"""
Model-serving boundary: image grounding, draft parsing, and audit assembly.

This module is the deep boundary between the vision-language model stack and the
rest of the application. It owns the two things that genuinely need the model -
turning a chest X-ray into a constrained grounded-finding list, and turning a
draft impression into the same label space - and then delegates the model-free
work to the authoritative pure-logic modules:

- the deterministic comparator lives in ``cxr_auditor.comparator``;
- the draft-impression parser lives in ``cxr_auditor.parser``.

``inference`` does not reimplement either; it wires them together. Every model
call flows through a single seam: a ``generate_fn`` callable that maps a rendered
text prompt to the model's raw completion (with an image bound to the closure for
the grounding turn). That same ``generate_fn`` shape is exactly what
``cxr_auditor.parser.parse_draft`` consumes, so the draft parser is driven through
its documented injected-callable contract rather than re-implemented here, and the
whole orchestration is testable with a fake ``generate_fn`` and no model.

A retry-on-invalid-JSON ladder (``run_with_retry``) wraps each generation. Each
attempt changes the conditions so a deterministic failure mode cannot simply
repeat: attempt one decodes greedily with the base prompt, attempt two appends a
corrective instruction (``RETRY_CORRECTIVE_SUFFIX``), and attempt three switches
to sampling (``RETRY_SAMPLING_SETTINGS``). Every parse failure is logged to
stdout with its full traceback and the offending raw model text, and the final
``SchemaParseError`` carries the last raw text for inspection. Draft parsing
degrades gracefully: when the draft cannot be parsed after its retries, the audit
proceeds image-only and records ``AuditOutcome.draft_parse_note`` so the user
interface can say so prominently. ``categorize_serving_error`` maps the
exceptions an audit call can surface (including the string-transported ZeroGPU
platform errors) to honest, user-facing messages.

The heavy stack (torch, transformers) is imported lazily via ``importlib`` inside
``load_model`` and ``_generate_text`` so importing this module on a pure-logic
checkout (no torch/transformers) succeeds; only the functions that need the stack
pay the import cost, and only when called.

ZeroGPU serving notes
---------------------
When this runs inside a ``@spaces.GPU`` worker, follow the platform contract:
load the model eagerly at module scope in the Space app via ``load_model`` (which
moves weights with the string device ``"cuda"``, never an int) using
``attn_implementation="sdpa"`` (Flash-Attention is not usable on the sm_120
backing GPU); and never return CUDA tensors across the worker boundary. The public
functions here return only plain pydantic objects and Python data, so they are
safe to call from a GPU worker.
"""

from __future__ import annotations

import importlib
import re
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cxr_auditor.comparator import ComparisonReport, compare
from cxr_auditor.parser import GenerateFn, parse_draft
from cxr_auditor.prompts import build_image_grounding_prompt
from cxr_auditor.schema import (
    AuditResult,
    DraftFinding,
    FindingStatus,
    ImageFinding,
    NormalizedBox,
    SchemaParseError,
    extract_finding_list,
    normalize_finding,
)

if TYPE_CHECKING:
    from PIL import Image

# Default base model identifier. The app overrides this with its ``HF_MODEL_ID``
# constant (sourced from the environment) once the user publishes their merged
# 16-bit model.
DEFAULT_MODEL_ID: str = "google/medgemma-1.5-4b-it"

# Default number of new tokens per generation. The constrained finding JSON for
# six findings is short; this leaves headroom without runaway decoding.
DEFAULT_MAX_NEW_TOKENS: int = 512

# Default retry budget for the invalid-JSON repair loop: one initial attempt plus
# this many additional re-generations.
DEFAULT_MAX_RETRIES: int = 2

# Draft-parsing retry budget: one initial attempt plus this many retries. Smaller
# than the grounding budget because draft parsing degrades gracefully (the audit
# proceeds image-only), and the combined worst case of grounding plus draft
# attempts must stay inside the GPU duration the serving app declares.
DRAFT_MAX_RETRIES: int = 1

# User-facing note recorded on the outcome when a supplied draft could not be
# parsed after all retries and the audit proceeded image-only.
DRAFT_PARSE_FAILURE_NOTE: str = "The draft text could not be parsed; results show image findings only."

# NVIDIA Nemotron-3 Nano 4B draft parser, run on the GPU through transformers. A
# dedicated, strong instruction-following text model parses the draft impression
# into the canonical label space far more reliably than the grounding-fine-tuned
# MedGemma did, and running it on the GPU (not CPU) keeps an audit fast. Its native
# transformers architecture (nemotron_h) is supported from transformers >= 5.3,
# which the Space already runs (MedGemma loads on it), so there is no risky major
# bump and no mamba CUDA kernel to build.
DRAFT_MODEL_ID: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

# Token budget for one draft parse. Nemotron reasons in a <think> block before the
# JSON (reasoning is enabled because it materially improves extraction accuracy; see
# _generate_draft_text), so the budget must hold both the reasoning and the short
# label JSON that follows. Greedy decoding stops at the end token well before this
# cap on a normal parse, so the headroom only guards against a truncated JSON on a
# longer-than-usual reasoning trace; the <think> block is stripped (see
# _strip_reasoning) before the JSON reaches the parser.
DRAFT_MAX_NEW_TOKENS: int = 1024

# System instruction that suppresses Nemotron's reasoning trace so the reply is the
# bare JSON array the draft parser expects.
_DRAFT_SYSTEM_INSTRUCTION: str = (
    "You are a precise radiology-label extractor. Do not show any reasoning. "
    "Respond with ONLY the requested JSON array - no <think> block, no prose, no markdown."
)

# Matches a Nemotron <think>...</think> reasoning block so it can be removed before
# JSON-array extraction (reasoning prose may contain brackets that would otherwise
# fool the tolerant array scanner).
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Corrective instruction appended to the prompt on retry attempts, so a retry
# never repeats the exact conditions that already failed deterministically.
RETRY_CORRECTIVE_SUFFIX: str = (
    "\nIMPORTANT: your previous reply was not one valid JSON array. "
    "Reply with ONE complete JSON array, starting with '[' and ending with ']'. "
    "Do not repeat elements. No prose."
)

# Truncation bound for raw model text echoed into stdout logs on parse failures.
_RAW_TEXT_LOG_LIMIT: int = 2000


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Decoding settings for one model generation.

    Attributes:
        do_sample: Whether to sample instead of decoding greedily.
        temperature: Sampling temperature; only forwarded when ``do_sample``.
        top_p: Nucleus-sampling probability mass; only forwarded when
            ``do_sample``.
    """

    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None


# Deterministic greedy decoding: the default for every first attempt.
GREEDY_SETTINGS: GenerationSettings = GenerationSettings()

# Mild sampling for the final retry attempt: enough randomness to escape a
# deterministic degenerate completion while keeping the constrained JSON shape
# likely.
RETRY_SAMPLING_SETTINGS: GenerationSettings = GenerationSettings(do_sample=True, temperature=0.4, top_p=0.9)

# A factory producing a ``GenerateFn`` bound to specific decoding settings. The
# retry ladder requests a fresh ``GenerateFn`` per attempt so attempt three can
# switch from greedy decoding to sampling.
type GenerateFnFactory = Callable[[GenerationSettings], GenerateFn]


def _attempt_settings(attempt: int) -> tuple[GenerationSettings, bool]:
    """Return the decoding plan for a 1-based retry-ladder attempt.

    Attempt 1 decodes greedily with the base prompt; attempt 2 keeps greedy
    decoding but appends the corrective suffix; attempt 3 and beyond switch to
    sampling (still with the suffix) so a deterministic failure cannot repeat
    verbatim.

    Args:
        attempt: The 1-based attempt number.

    Returns:
        A ``(settings, append_corrective_suffix)`` pair.
    """
    if attempt == 1:
        return GREEDY_SETTINGS, False
    if attempt == 2:
        return GREEDY_SETTINGS, True
    return RETRY_SAMPLING_SETTINGS, True


def _log_parse_failure(stage: str, attempt: int, error: SchemaParseError) -> None:
    """Print a parse failure's traceback and raw model text to stdout.

    Worker stdout reaches the serving platform's run logs, so this is the durable
    diagnostic channel for malformed model output: the full traceback shows where
    parsing failed and the delimited block shows exactly what the model emitted
    (truncated to ``_RAW_TEXT_LOG_LIMIT`` characters to keep log entries bounded).

    Args:
        stage: The pipeline stage that failed (for example ``"image_grounding"``).
        attempt: The 1-based attempt number that produced the failure.
        error: The parse error carrying the offending raw model text.
    """
    print(f"[cxr-auditor] parse failure: stage={stage} attempt={attempt}", flush=True)
    traceback.print_exception(error, file=sys.stdout)
    raw = error.raw_text
    if len(raw) > _RAW_TEXT_LOG_LIMIT:
        raw = f"{raw[:_RAW_TEXT_LOG_LIMIT]} ...[truncated]"
    print(f"[cxr-auditor] raw model text (stage={stage} attempt={attempt}) >>>\n{raw}\n<<<", flush=True)


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    """The complete result of one audit run.

    Bundles the serializable ``AuditResult`` (the canonical output object) with the
    richer ``ComparisonReport`` the user interface uses to draw per-item evidence
    (which box belongs to a missing finding, which to an urgent flag, and the draft
    span behind each unsupported claim).

    Attributes:
        result: The canonical ``AuditResult`` (image findings, draft findings,
            label-only audit, disclaimer, box format).
        comparison: The per-item comparator detail (boxes, urgency, draft spans).
        draft_parse_note: A user-facing note set when a supplied draft could not
            be parsed after all retries, so the audit proceeded image-only.
            ``None`` when no draft was supplied or the draft parsed.
    """

    result: AuditResult
    comparison: ComparisonReport
    draft_parse_note: str | None = None


def grounded_dicts_to_image_findings(grounded: list[dict[str, Any]]) -> list[ImageFinding]:
    """Convert MedGemma-style grounded dicts into validated ``ImageFinding`` objects.

    Each input dict follows the model's native grounding shape
    ``{"label": ..., "box_2d": [y0, x0, y1, x1] | null, "confidence": ..., "evidence": ...}``.
    Labels outside the canonical set are skipped: the model is constrained to the
    canonical labels, but tolerating one stray label keeps it from failing the
    whole turn. A box that fails validation is dropped while the finding is kept,
    so a malformed box never discards a real finding.

    Exact ``(label, box)`` duplicates are collapsed to the first occurrence so a
    finding the model emits twice at identical coordinates does not produce a
    duplicate row or a duplicate overlay box. Deduplication is on the
    ``(label, box)`` pair only, so two genuinely distinct findings survive: the
    same label localized to two different boxes (a bilateral finding) and two
    different labels at the same box (a region the model reads as both an opacity
    and a nodule) are each kept, because neither shares a ``(label, box)`` key.

    This is the image-side analogue of ``parser.parse_draft_findings``; it lives
    here because turning raw model grounding output into findings is part of the
    model-serving boundary.

    Args:
        grounded: The list of grounded finding dicts parsed from model output.

    Returns:
        Validated ``ImageFinding`` objects, in the order the model emitted them,
        with non-canonical labels removed and exact ``(label, box)`` duplicates
        collapsed.
    """
    findings: list[ImageFinding] = []
    seen: set[tuple[str, NormalizedBox | None]] = set()
    for item in grounded:
        raw_label = item.get("label")
        if not isinstance(raw_label, str):
            continue
        try:
            label = normalize_finding(raw_label)
        except ValueError:
            continue
        box = _coerce_box(item.get("box_2d"))
        key = (label, box)
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            ImageFinding(
                finding=label,
                status=FindingStatus.PRESENT,
                box=box,
                confidence=_coerce_confidence(item.get("confidence")),
                evidence=item.get("evidence") if isinstance(item.get("evidence"), str) else None,
            )
        )
    return findings


def _coerce_box(raw: Any) -> tuple[float, float, float, float] | None:
    """Coerce a raw ``box_2d`` value into a validated normalized box, or ``None``.

    A non-sequence, wrong-length, out-of-range, or top-left/bottom-right-inverted
    box yields ``None`` rather than raising, so a malformed box never discards an
    otherwise-valid finding.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        y0, x0, y1, x1 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= component <= 1.0 for component in (y0, x0, y1, x1)):
        return None
    if y1 < y0 or x1 < x0:
        return None
    return (y0, x0, y1, x1)


def _coerce_confidence(raw: Any) -> float | None:
    """Coerce a raw confidence value into a float in [0, 1], or ``None``."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def run_with_retry(
    generate_fn_factory: GenerateFnFactory,
    prompt: str,
    parse_fn: Callable[[str], list[Any]],
    max_retries: int = DEFAULT_MAX_RETRIES,
    *,
    stage: str = "generation",
) -> list[Any]:
    """Generate then parse, escalating the retry conditions on each failure.

    The model occasionally emits prose, a truncated array, or otherwise
    unparseable text - and a greedy decode of the same prompt fails the same way
    every time. Each attempt therefore changes the conditions per
    ``_attempt_settings``: attempt 1 is greedy with the base prompt, attempt 2 is
    greedy with ``RETRY_CORRECTIVE_SUFFIX`` appended, and attempt 3 onward samples
    (``RETRY_SAMPLING_SETTINGS``) with the suffix. Every failed attempt is logged
    to stdout with its traceback and raw model text. If every attempt fails, the
    last ``SchemaParseError`` is raised so its ``raw_text`` (the final raw
    completion) is available to the caller.

    Args:
        generate_fn_factory: Factory returning a ``GenerateFn`` for the decoding
            settings of each attempt.
        prompt: The rendered base prompt (the corrective suffix is appended to it
            on retry attempts).
        parse_fn: A tolerant parser turning raw text into a list (raises
            ``SchemaParseError`` on unparseable text).
        max_retries: Number of additional attempts after the first. Must be >= 0.
        stage: Stage label used in failure logs (keyword-only).

    Returns:
        The first successfully parsed list.

    Raises:
        ValueError: If ``max_retries`` is negative.
        SchemaParseError: If every attempt fails to parse.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")

    last_error: SchemaParseError | None = None
    for attempt in range(1, max_retries + 2):
        settings, corrective = _attempt_settings(attempt)
        generate_fn = generate_fn_factory(settings)
        attempt_prompt = prompt + RETRY_CORRECTIVE_SUFFIX if corrective else prompt
        raw_text = generate_fn(attempt_prompt)
        try:
            return parse_fn(raw_text)
        except SchemaParseError as exc:
            last_error = exc
            _log_parse_failure(stage, attempt, exc)
    assert last_error is not None  # loop runs at least once, so an error was set
    raise last_error


def make_generate_fn(
    model: Any,
    processor: Any,
    image: Image.Image,
    settings: GenerationSettings = GREEDY_SETTINGS,
) -> GenerateFn:
    """Build an image-bound ``generate_fn`` over a loaded model and processor.

    The returned closure captures the model, processor, image, and decoding
    settings, exposing the text-in/text-out ``GenerateFn`` shape that the
    grounding path, the retry ladder, and ``cxr_auditor.parser.parse_draft`` all
    consume. Binding the image into the closure lets the draft parser - which
    only knows about a text-prompt ``generate_fn`` - reuse the same single-turn
    multimodal model the grounding step uses, so the draft is parsed with the
    SAME model rather than a separate text-only stack.

    Args:
        model: A loaded vision-language model exposing ``generate``.
        processor: The matching transformers processor.
        image: The chest X-ray bound to every generation through this closure.
        settings: Decoding settings bound to every generation through this
            closure.

    Returns:
        A ``GenerateFn`` mapping a rendered prompt to the model's raw completion.
    """

    def _generate(prompt: str) -> str:
        return _generate_text(model, processor, prompt, image, settings=settings)

    return _generate


def _generate_fn_factory(model: Any, processor: Any, image: Image.Image) -> GenerateFnFactory:
    """Build a settings-to-``GenerateFn`` factory bound to one model and image.

    This is the shape the retry ladder consumes: each attempt requests a
    ``GenerateFn`` for its own decoding settings while the model, processor, and
    image stay fixed.

    Args:
        model: A loaded vision-language model exposing ``generate``.
        processor: The matching transformers processor.
        image: The chest X-ray bound to every generation.

    Returns:
        A factory mapping ``GenerationSettings`` to an image-bound ``GenerateFn``.
    """

    def _factory(settings: GenerationSettings) -> GenerateFn:
        return make_generate_fn(model, processor, image, settings=settings)

    return _factory


def generate_findings(
    image: Image.Image,
    *,
    model: Any,
    processor: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[ImageFinding]:
    """Ground an image into validated ``ImageFinding`` objects.

    Builds the pinned image-grounding prompt, generates through the escalating
    retry ladder, and assembles validated findings. This is the image-side entry
    point the app uses when it wants only the grounded findings (for example to
    draw boxes before a draft is supplied).

    Args:
        image: The chest X-ray as a PIL image.
        model: A loaded vision-language model (keyword-only).
        processor: The matching transformers processor (keyword-only).
        max_retries: Retry budget for the invalid-JSON ladder (keyword-only).

    Returns:
        The image-grounded findings with bounding-box evidence.

    Raises:
        SchemaParseError: If grounding output cannot be parsed after all retries.
    """
    grounded = run_with_retry(
        _generate_fn_factory(model, processor, image),
        build_image_grounding_prompt(),
        extract_finding_list,
        max_retries=max_retries,
        stage="image_grounding",
    )
    return grounded_dicts_to_image_findings(grounded)


def assemble_audit(
    image_findings: list[ImageFinding],
    draft_text: str | None = None,
    *,
    draft_factory: GenerateFnFactory,
    max_retries: int = DRAFT_MAX_RETRIES,
) -> AuditOutcome:
    """Assemble an audit from already-grounded image findings.

    Given the image findings the grounding step produced, this parses a supplied
    draft into the canonical label space through the injected text ``GenerateFn``
    factory (backed by the Nemotron draft parser in the serving app), runs the
    deterministic comparator, and bundles the ``AuditOutcome``. The comparator and
    the retry orchestration here are pure logic; the draft generation itself happens
    inside the injected factory, which the serving app runs on the GPU through
    transformers. Splitting assembly out from grounding lets the serving app ground
    the image and then parse the draft within a single ``@spaces.GPU`` window. Draft
    parsing degrades gracefully: a draft that cannot be parsed after its retries
    never fails the audit - the audit proceeds image-only and
    ``AuditOutcome.draft_parse_note`` records the degradation for the user interface.

    Args:
        image_findings: The grounded findings from ``generate_findings``.
        draft_text: The draft impression. ``None`` or whitespace skips draft parsing.
        draft_factory: Factory returning a text ``GenerateFn`` for each attempt's
            decoding settings (keyword-only).
        max_retries: Retry budget for the draft-parse ladder (keyword-only).

    Returns:
        The ``AuditOutcome`` (result, comparator detail, and the draft note when set).
    """
    draft_findings: list[DraftFinding] = []
    draft_parse_note: str | None = None
    cleaned_draft = (draft_text or "").strip()
    if cleaned_draft:
        try:
            draft_findings = _parse_draft_with_retry(cleaned_draft, draft_factory, max_retries=max_retries)
        except SchemaParseError:
            # Per-attempt details are already logged by _log_parse_failure; the
            # audit degrades to image-only rather than failing on the draft.
            print("[cxr-auditor] draft parsing failed after all retries; auditing image only", flush=True)
            draft_parse_note = DRAFT_PARSE_FAILURE_NOTE

    comparison = compare(image_findings, draft_findings)
    result = AuditResult(
        image_findings=image_findings,
        draft_findings=draft_findings,
        audit=comparison.audit,
    )
    return AuditOutcome(result=result, comparison=comparison, draft_parse_note=draft_parse_note)


def run_audit(
    image: Image.Image,
    draft_text: str | None = None,
    *,
    model: Any = None,
    processor: Any = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AuditOutcome:
    """Run the full single-model audit pipeline (grounding plus draft) in one call.

    Grounds the image into validated ``ImageFinding`` objects (``generate_findings``)
    and then assembles the audit (``assemble_audit``) with an image-bound MedGemma
    ``GenerateFn`` factory for the draft path, so this stays a single self-contained
    entry point for tests and any single-model caller. The serving app instead splits
    grounding from the draft parse - grounding the image and then parsing the draft
    with the Nemotron model, both on the GPU through transformers - via
    ``generate_findings`` and ``assemble_audit``. Draft parsing degrades gracefully
    (image-only with a note).

    The model and processor are passed in rather than loaded here so the caller (a
    Gradio Space) loads them once at module scope and this function only runs
    inference and assembles the result.

    Args:
        image: The chest X-ray as a PIL image.
        draft_text: The draft radiology impression. ``None`` or whitespace means no
            draft was supplied; draft parsing is skipped and the audit reports only
            image-side findings (and urgent flags).
        model: A loaded vision-language model exposing ``generate`` (keyword-only).
        processor: The matching transformers processor (keyword-only).
        max_retries: Retry budget for the image-grounding ladder (keyword-only).
            Draft parsing uses the fixed ``DRAFT_MAX_RETRIES`` budget because it
            degrades gracefully instead of failing.

    Returns:
        An ``AuditOutcome`` carrying the canonical ``AuditResult``, the per-item
        ``ComparisonReport``, and the draft-degradation note when it applies.

    Raises:
        ValueError: If ``model`` or ``processor`` is not supplied.
        SchemaParseError: If the image-grounding output cannot be parsed into a
            finding list after all retries.
    """
    if model is None or processor is None:
        raise ValueError("run_audit requires both a loaded model and processor")

    image_findings = generate_findings(image, model=model, processor=processor, max_retries=max_retries)
    return assemble_audit(
        image_findings,
        draft_text,
        draft_factory=_generate_fn_factory(model, processor, image),
        max_retries=DRAFT_MAX_RETRIES,
    )


def _with_corrective_suffix(generate_fn: GenerateFn) -> GenerateFn:
    """Wrap a ``GenerateFn`` so every prompt carries the corrective suffix.

    ``parser.parse_draft`` builds its prompt internally, so retry attempts inject
    ``RETRY_CORRECTIVE_SUFFIX`` by wrapping the callable rather than editing the
    prompt directly.

    Args:
        generate_fn: The inner ``GenerateFn`` to wrap.

    Returns:
        A ``GenerateFn`` that appends the corrective suffix to every prompt.
    """

    def _generate(prompt: str) -> str:
        return generate_fn(prompt + RETRY_CORRECTIVE_SUFFIX)

    return _generate


def _parse_draft_with_retry(
    draft_text: str,
    generate_fn_factory: GenerateFnFactory,
    max_retries: int = DRAFT_MAX_RETRIES,
) -> list[DraftFinding]:
    """Parse a draft through ``parser.parse_draft`` with the escalating ladder.

    Wraps the draft parser's injected-callable contract in the same attempt plan
    the grounding path uses (``_attempt_settings``): each attempt re-runs the full
    ``parse_draft`` (build prompt, generate, validate), retry attempts append the
    corrective suffix via a wrapped ``GenerateFn``, the first successful parse
    wins, and the final ``SchemaParseError`` propagates if all attempts fail.

    Args:
        draft_text: The non-empty draft impression to parse.
        generate_fn_factory: Factory returning an image-bound ``GenerateFn`` for
            each attempt's decoding settings.
        max_retries: Number of additional attempts after the first. Must be >= 0.

    Returns:
        Validated ``DraftFinding`` objects.

    Raises:
        ValueError: If ``max_retries`` is negative.
        SchemaParseError: If every attempt fails to parse.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")

    last_error: SchemaParseError | None = None
    for attempt in range(1, max_retries + 2):
        settings, corrective = _attempt_settings(attempt)
        generate_fn = generate_fn_factory(settings)
        if corrective:
            generate_fn = _with_corrective_suffix(generate_fn)
        try:
            return parse_draft(draft_text, generate_fn)
        except SchemaParseError as exc:
            last_error = exc
            _log_parse_failure("draft_parsing", attempt, exc)
    assert last_error is not None
    raise last_error


def audit(
    image: Image.Image,
    draft_text: str | None = None,
    *,
    model: Any = None,
    processor: Any = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> AuditResult:
    """Run the full audit pipeline and return the canonical ``AuditResult``.

    Thin wrapper over ``run_audit`` for callers that only need the serializable
    result and not the per-item comparator detail.

    Args:
        image: The chest X-ray as a PIL image.
        draft_text: The optional draft impression (``None``/whitespace skips it).
        model: A loaded vision-language model (keyword-only).
        processor: The matching transformers processor (keyword-only).
        max_retries: Retry budget for both invalid-JSON loops (keyword-only).

    Returns:
        The canonical ``AuditResult``.

    Raises:
        ValueError: If ``model`` or ``processor`` is not supplied.
        SchemaParseError: If the model output cannot be parsed after all retries.
    """
    return run_audit(image, draft_text, model=model, processor=processor, max_retries=max_retries).result


def load_model(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
    """Load the merged MedGemma model and its processor for serving.

    Heavy imports (torch, transformers) happen inside this function so importing
    ``cxr_auditor.inference`` never requires the vision stack. The model is loaded
    in bfloat16 with SDPA attention (Flash-Attention 3 is unavailable on the
    ZeroGPU sm_120 backing GPU) and moved to the string device ``"cuda"``. Under
    ZeroGPU the ``spaces`` package monkey-patches torch so this module-scope
    ``.to("cuda")`` registers weights for the forked GPU worker.

    Args:
        model_id: The Hugging Face model id to load (a merged 16-bit MedGemma).

    Returns:
        A ``(model, processor)`` tuple.

    Note:
        The model is loaded via the image-text-to-text auto classes
        (``AutoModelForImageTextToText`` / ``AutoProcessor``). If a future
        transformers release renames the auto class for this model, update the
        attribute names below; the documented interface is the one used here.
    """
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    return model, processor


def _strip_reasoning(text: str) -> str:
    """Remove a Nemotron ``<think>...</think>`` reasoning block from model text.

    Nemotron is a reasoning model; even with reasoning discouraged it can emit a
    ``<think>...</think>`` preamble before the JSON. Removing the block first keeps
    the reasoning's brackets from fooling the tolerant JSON-array scanner.

    Args:
        text: Raw text from the draft model.

    Returns:
        The text with any complete reasoning block removed and surrounding
        whitespace stripped.
    """
    return _THINK_PATTERN.sub("", text).strip()


def load_draft_model(model_id: str = DRAFT_MODEL_ID) -> tuple[Any, Any]:
    """Load the Nemotron-3 Nano 4B draft parser (text) on the GPU via transformers.

    Heavy imports (torch, transformers) happen inside this function so importing
    ``cxr_auditor.inference`` never requires the model stack. Nemotron's native
    transformers architecture (``nemotron_h``) is supported from transformers 5.3,
    which the Space already runs; using it avoids the bundled remote-code path (which
    needs a mamba CUDA kernel that cannot build on ZeroGPU). Because that kernel is
    absent, transformers logs a benign one-time notice ("the fast path is not
    available ... falling back to the naive implementation") for Nemotron's Mamba2
    layers; the naive PyTorch path is correct (only slightly slower), and the short
    greedy draft parse is unaffected, so this is expected and needs no action. The
    model is loaded in bfloat16 and moved to the string device ``"cuda"`` so ZeroGPU
    registers it for the forked GPU worker alongside MedGemma, and the draft parse
    runs fast on the GPU.

    Args:
        model_id: The Hugging Face Nemotron text-model repo id.

    Returns:
        A ``(model, tokenizer)`` tuple (typed ``Any`` because transformers is not
        importable in the dev environment).
    """
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    model.eval()
    # Nemotron's bundled generation_config ships top_p and temperature but no
    # do_sample, so under greedy decoding transformers logs "generation flags ...
    # may be ignored: ['top_p']". Clear those sampling-only fields so greedy
    # generation is clean; this does not change greedy output (the retry ladder
    # still passes top_p/temperature itself on the sampling attempts).
    model.generation_config.top_p = None
    model.generation_config.temperature = None
    model.generation_config.top_k = None
    return model, tokenizer


def _generate_draft_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    settings: GenerationSettings = GREEDY_SETTINGS,
) -> str:
    """Run one single-turn Nemotron text generation for draft parsing.

    Applies the tokenizer's chat template with Nemotron's reasoning ENABLED
    (``enable_thinking=True``): the model reasons inside a ``<think>...</think>``
    block before emitting the JSON array. The reasoning is load-bearing - for this
    small model it materially improves label-extraction accuracy (with reasoning
    suppressed the model drops and substitutes findings on multi-clause drafts), so
    it is kept on and the ``<think>`` block is removed afterward by
    ``_strip_reasoning`` so the parser receives only the JSON. Generates with the
    supplied decoding settings, decoding only the newly generated tokens.
    ``eos_token_id`` is pinned to the tokenizer's end token (the ChatML turn
    terminator) because Nemotron's generation config defaults to a different token
    that would not stop at the assistant boundary; ``pad_token_id`` is pinned to the
    same token to avoid a pad warning.

    Args:
        model: The loaded Nemotron causal-LM.
        tokenizer: The matching tokenizer.
        prompt: The fully rendered draft-parsing prompt.
        settings: Decoding settings for this generation.

    Returns:
        The model's cleaned reply text (prompt tokens stripped, reasoning removed).
    """
    torch = importlib.import_module("torch")

    messages = [
        {"role": "system", "content": _DRAFT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_ids = inputs["input_ids"]

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": DRAFT_MAX_NEW_TOKENS,
        "do_sample": settings.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    if settings.do_sample:
        if settings.temperature is not None:
            generate_kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            generate_kwargs["top_p"] = settings.top_p

    input_len = input_ids.shape[-1]
    with torch.inference_mode():
        generated = model.generate(input_ids, **generate_kwargs)
    reply = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
    return _strip_reasoning(reply)


def make_draft_generate_fn(draft_model: tuple[Any, Any], settings: GenerationSettings = GREEDY_SETTINGS) -> GenerateFn:
    """Build a text ``GenerateFn`` backed by the Nemotron draft model.

    Maps a rendered text prompt to the model's cleaned completion via
    ``_generate_draft_text`` (reasoning disabled, residual reasoning stripped) so the
    result is the JSON array the draft parser consumes. Decoding follows the retry
    ladder's settings: greedy by default, mild sampling on the final attempt.

    Args:
        draft_model: A ``(model, tokenizer)`` tuple from ``load_draft_model``.
        settings: Decoding settings bound to this generation.

    Returns:
        A ``GenerateFn`` mapping a prompt to the model's cleaned text completion.
    """
    model, tokenizer = draft_model

    def _generate(prompt: str) -> str:
        return _generate_draft_text(model, tokenizer, prompt, settings)

    return _generate


def draft_generate_fn_factory(draft_model: tuple[Any, Any]) -> GenerateFnFactory:
    """Build a settings-to-``GenerateFn`` factory bound to the Nemotron draft model.

    This is the shape the draft retry ladder consumes: each attempt requests a
    ``GenerateFn`` for its own decoding settings while the model stays fixed.

    Args:
        draft_model: A ``(model, tokenizer)`` tuple from ``load_draft_model``.

    Returns:
        A factory mapping ``GenerationSettings`` to a Nemotron-backed ``GenerateFn``.
    """

    def _factory(settings: GenerationSettings) -> GenerateFn:
        return make_draft_generate_fn(draft_model, settings)

    return _factory


def _generate_text(
    model: Any,
    processor: Any,
    prompt: str,
    image: Image.Image,
    settings: GenerationSettings = GREEDY_SETTINGS,
) -> str:
    """Run one single-turn multimodal generation and return the decoded reply.

    This is the only function that touches the model at inference time, and the
    single seam tests patch to drive the orchestration without a real model. It
    builds a single-turn chat message with the image and the prompt text, applies
    the processor's chat template, generates with the supplied decoding settings,
    and decodes only the newly generated tokens (slicing off the prompt) so the
    returned text is just the model's reply (a plain ``str``, never a tensor
    across a worker boundary). ``pad_token_id`` is pinned to the tokenizer's
    end-of-sequence token so generation runs without a pad-token warning.

    Heavy imports are local to keep module import free of the vision stack. The
    chat-message construction follows the transformers image-text-to-text
    convention; adjust the message schema if a future processor revision changes
    it.

    Args:
        model: The loaded vision-language model.
        processor: The matching processor.
        prompt: The fully rendered text prompt.
        image: The chest X-ray as a PIL image.
        settings: Decoding settings for this generation.

    Returns:
        The model's decoded reply text (prompt tokens stripped).
    """
    torch = importlib.import_module("torch")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": settings.do_sample,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    # Sampling knobs are forwarded only when sampling; transformers warns when
    # they accompany greedy decoding.
    if settings.do_sample:
        if settings.temperature is not None:
            generate_kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            generate_kwargs["top_p"] = settings.top_p

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    new_tokens = generated[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True)


# Titles the ZeroGPU scheduler attaches to the quota and scheduling errors it
# raises in the serving app's main process. These failures are platform-side: the
# uploaded image is never the cause, so the user message must not blame it.
_GPU_SCHEDULING_ERROR_TITLES: frozenset[str] = frozenset(
    {
        "ZeroGPU quota exceeded",
        "ZeroGPU illegal duration",
        "ZeroGPU pending credits exceeded",
        "ZeroGPU queue timeout",
        "ZeroGPU client error",
    }
)

# Title the ZeroGPU platform attaches when a worker exception was converted to a
# string-transported error whose message body is the worker exception class name.
_WORKER_ERROR_TITLE: str = "ZeroGPU worker error"

# Worker exception class names that mean "the model output could not be parsed".
_PARSE_ERROR_CLASS_NAMES: frozenset[str] = frozenset({"SchemaParseError"})

# Message body the ZeroGPU platform uses when it cuts a GPU task short.
_GPU_TASK_ABORTED_BODY: str = "GPU task aborted"

_GPU_QUOTA_MESSAGE: str = (
    "**GPU quota reached.** Free ZeroGPU time is temporarily exhausted, so this audit could not get a "
    "GPU slot - the image is not the problem. Wait a few minutes and press Run audit again."
)

_GPU_INTERRUPTED_MESSAGE: str = (
    "**GPU task was interrupted.** The platform cut the GPU run short before the audit finished. "
    "Please press Run audit again."
)


def _parse_failure_message(class_name: str) -> str:
    """Return the user-facing message for an unparseable-model-output failure."""
    return (
        f"**Could not analyze this image.** The model returned output that could not be parsed ({class_name}). "
        "Please try again, or use a clearer frontal chest X-ray."
    )


def _generic_failure_message(detail: str) -> str:
    """Return the user-facing message for an uncategorized failure."""
    return (
        f"**Audit failed.** An unexpected error occurred ({detail}). "
        "Please try again; if this keeps happening, check the Space logs."
    )


def categorize_serving_error(error: Exception) -> str:
    """Map an audit-time exception to an honest, user-facing Markdown message.

    The ZeroGPU platform never pickles worker exceptions across the process
    boundary: it transports the worker exception's class name as the message body
    of a gradio error object whose own class ``__name__`` is literally ``"Error"``
    and whose ``title`` attribute names the failure source. This categorizer
    therefore classifies on the exception's class name, ``title`` attribute, and
    message body - never on ``isinstance`` against gradio types - so it stays
    importable and unit-testable without gradio installed.

    Categories:
        - Quota and scheduling errors (a title in the known ZeroGPU set, or any
          title or body mentioning "quota") produce a GPU-quota message that
          never blames the image.
        - An aborted GPU task (body ``"GPU task aborted"``) produces a transient
          interrupted-please-retry message.
        - A worker error whose body names a parse-failure class, or a directly
          raised ``SchemaParseError``, produces the could-not-parse message
          naming the real exception class.
        - Anything else produces a generic failure message naming the true type.

    Args:
        error: The exception caught around the GPU audit call.

    Returns:
        A Markdown message suitable for the audit panel.
    """
    if isinstance(error, SchemaParseError):
        return _parse_failure_message(type(error).__name__)

    body = str(error)
    if type(error).__name__ == "Error":
        title = str(getattr(error, "title", ""))
        if title in _GPU_SCHEDULING_ERROR_TITLES or "quota" in title.lower() or "quota" in body.lower():
            return _GPU_QUOTA_MESSAGE
        if _GPU_TASK_ABORTED_BODY in body:
            return _GPU_INTERRUPTED_MESSAGE
        if title == _WORKER_ERROR_TITLE:
            if body in _PARSE_ERROR_CLASS_NAMES:
                return _parse_failure_message(body)
            return _generic_failure_message(body or type(error).__name__)
    return _generic_failure_message(type(error).__name__)


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL_ID",
    "DRAFT_MAX_NEW_TOKENS",
    "DRAFT_MAX_RETRIES",
    "DRAFT_MODEL_ID",
    "DRAFT_PARSE_FAILURE_NOTE",
    "GREEDY_SETTINGS",
    "RETRY_CORRECTIVE_SUFFIX",
    "RETRY_SAMPLING_SETTINGS",
    "AuditOutcome",
    "GenerateFnFactory",
    "GenerationSettings",
    "assemble_audit",
    "audit",
    "categorize_serving_error",
    "draft_generate_fn_factory",
    "generate_findings",
    "grounded_dicts_to_image_findings",
    "load_draft_model",
    "load_model",
    "make_draft_generate_fn",
    "make_generate_fn",
    "run_audit",
    "run_with_retry",
]
