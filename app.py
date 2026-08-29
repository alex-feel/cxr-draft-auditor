"""
CXR Draft Auditor - Hugging Face ZeroGPU Gradio Space entry point.

Research/educational QA only. NOT a medical device, NOT diagnosis, NOT for
clinical use. The app takes a chest X-ray plus an optional draft radiology
impression, grounds the image into a constrained finding set with bounding-box
evidence, parses the draft into the same label space, and runs a deterministic
comparator that flags MISSING findings, UNSUPPORTED claims, and URGENT review
flags. The image evidence is overlaid so a person can look again; the tool never
issues a verdict.

Presentation (the "Reading Room" UI)
------------------------------------
All presentation lives in ``reading_room_ui``: a dark PACS-inspired theme, the
custom CSS, and pure HTML builders for the header (with the RESEARCH PREVIEW
badge), the permanent disclaimer ribbon, the numbered input cards, the evidence
legend, the verdict cards with summary chips, the findings table with status
pills, and the footer. This module wires only layout and callbacks. Per the
Gradio 6 API, the theme and CSS are passed to ``demo.launch(...)``; passing them
to the ``gr.Blocks`` constructor is deprecated.

ZeroGPU rules honored
---------------------
1. ``spaces`` is imported at module load (via importlib; see the import note below)
   before any CUDA-touching import, so its torch CUDA monkey-patch is applied first.
   This module never imports torch at module scope; torch is imported lazily inside
   ``cxr_auditor.inference`` only when a GPU call runs. ZeroGPU activation is driven
   by the ``@spaces.GPU`` decorator (statically present below) plus the Space
   hardware tier, not by a static ``import spaces`` token, so the runtime import is
   functionally complete.
2. The model and processor are loaded eagerly at module scope and moved to the
   string device ``"cuda"`` (never an integer device id) inside
   ``inference.load_model``, so ZeroGPU packs the weights at startup and streams
   them into the forked GPU worker.
3. GPU inference runs inside a single ``@spaces.GPU(duration=...)``-decorated
   handler (the only GPU-touching function), with a small declared duration for
   queue priority and a lenient quota pre-check.
4. The handler never returns CUDA tensors: it returns an ``AuditOutcome`` built
   from plain Python objects and validated pydantic models, which crosses the
   worker boundary cleanly; the overlay image and HTML strings are built on CPU.
5. No mutable module-global state is written from handlers; module globals (model,
   processor, config) are read-only after startup, and every handler output is a
   per-request return value (no fixed output paths), so concurrent requests are
   safe.
6. ``attn_implementation="sdpa"`` is used (Flash-Attention 3 is unavailable on the
   ZeroGPU sm_120 Blackwell backing GPU).
7. ``torch.compile`` is not used (unsupported on ZeroGPU).
8. ``gr.Examples`` caching is left at the ZeroGPU defaults (lazy), never forced to
   eager (no GPU is attached at startup).
9. The Space ``requirements.txt`` does not list gradio/spaces/huggingface_hub
   (platform-managed) and leaves torch unpinned; see ``requirements.txt``.

Model id
--------
``HF_MODEL_ID`` defaults to a placeholder the user replaces with their published
merged 16-bit MedGemma model (via the ``HF_MODEL_ID`` Space variable). While it is
still the placeholder, the app boots and shows a clear configuration message
instead of attempting to load a model, so the Space comes up green for inspection.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import Any

from PIL import Image

import reading_room_ui as ui
from cxr_auditor.findings import display_name
from cxr_auditor.inference import (
    DEFAULT_MODEL_ID,
    DRAFT_MODEL_ID,
    AuditOutcome,
    assemble_audit,
    categorize_serving_error,
    draft_generate_fn_factory,
    generate_findings,
    load_draft_model,
    load_model,
    run_audit,
)
from cxr_auditor.render import annotate_evidence, findings_table_rows, result_json

# ``spaces`` and ``gradio`` are supplied by the Hugging Face Space platform (the
# Gradio base image ships them on every hardware tier) and are intentionally NOT
# installed in this project's pure-logic / dev environment. They are imported at
# module load via importlib and bound as ``Any``. This keeps the strict type gate
# (mypy / pyright / ty) green WITHOUT a
# bypass comment, while still importing ``spaces`` before any CUDA-touching import so
# its torch monkey-patch is applied first (this module never imports torch at module
# scope; the inference layer imports it lazily during a GPU call). Per Hugging
# Face's ZeroGPU documentation, ZeroGPU activation is driven by the ``@spaces.GPU``
# decorator (statically present below) plus the Space hardware tier, not by a static
# ``import spaces`` token, so this runtime import is functionally complete. ``spaces``
# is correctly omitted from the Space requirements.txt (the platform pins it).
spaces: Any = importlib.import_module("spaces")
gr: Any = importlib.import_module("gradio")

# The merged 16-bit model to serve. Replace the placeholder via the HF_MODEL_ID
# Space variable (Settings -> Variables) once you publish your fine-tuned model,
# or set it to the base model id below to serve the base MedGemma.
_PLACEHOLDER_MODEL_ID = "your-username/cxr-draft-auditor-medgemma-merged"
HF_MODEL_ID: str = os.environ.get("HF_MODEL_ID", _PLACEHOLDER_MODEL_ID)

# Documents the base model id for operators who want to serve it directly; the
# value is surfaced in the configuration notice so the relationship is visible.
BASE_MODEL_ID: str = DEFAULT_MODEL_ID

# GPU seconds to reserve per audit. Both models run on the GPU: MedGemma grounds the
# image and Nemotron parses the draft. Worst case is the grounding retry ladder (up
# to three 512-token generations) plus the draft generations; each is a single-turn
# 4B bf16 decode on a Blackwell card, so 90s is a generous bound. Measure on the live
# Space and set ``duration = round(measured_max * 1.4)``.
_GPU_DURATION = 90

# Truncation bound for raw model text echoed into stdout logs on audit failures.
_RAW_TEXT_LOG_LIMIT = 2000

_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def _model_is_configured() -> bool:
    """Return whether a real (non-placeholder) model id has been configured."""
    return HF_MODEL_ID not in ("", _PLACEHOLDER_MODEL_ID)


# Eager module-scope model load (ZeroGPU rule 2). When the model id is still the
# placeholder, skip the load so the Space boots and shows a configuration message.
# A failed load is captured rather than raised so the Space still comes up and the
# error is shown in the UI instead of crashing the boot.
_MODEL = None
_PROCESSOR = None
_LOAD_ERROR: str | None = None

if _model_is_configured():
    try:
        _MODEL, _PROCESSOR = load_model(HF_MODEL_ID)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        # A model-load failure (missing/gated repo, bad id, missing optional
        # backend, allocation error) is surfaced in the UI so the Space still
        # boots for inspection rather than crashing on import.
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"

# The draft-impression parser is NVIDIA Nemotron-3 Nano 4B, run on the GPU through
# transformers, loaded once at module scope as a (model, tokenizer) pair alongside
# MedGemma. A successful load builds the per-attempt GenerateFn factory the draft
# retry ladder consumes. If it fails to load (for example the download fails), the
# app degrades to parsing the draft with MedGemma instead of crashing (see on_run).
_DRAFT_MODEL: Any = None
_DRAFT_FACTORY: Any = None
_DRAFT_LOAD_ERROR: str | None = None

if _MODEL is not None and _PROCESSOR is not None:
    try:
        _DRAFT_MODEL = load_draft_model()
        _DRAFT_FACTORY = draft_generate_fn_factory(_DRAFT_MODEL)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        _DRAFT_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        print(
            "[cxr-auditor] Nemotron draft parser unavailable; the draft will be parsed by "
            f"MedGemma instead: {_DRAFT_LOAD_ERROR}",
            flush=True,
        )


def _model_ready() -> bool:
    """Return whether a model is configured, loaded, and ready to serve audits."""
    return _model_is_configured() and _LOAD_ERROR is None and _MODEL is not None and _PROCESSOR is not None


def _status_message() -> str:
    """Return a message describing a degraded model configuration state.

    Only meaningful when ``_model_ready()`` is ``False``; the Reading Room UI
    shows it as an amber notice card in the verdict panel (``ui.message_html``
    converts the ``**bold**`` and ```code``` spans to HTML).
    """
    if _LOAD_ERROR is not None:
        return (
            f"**Model failed to load** (`{HF_MODEL_ID}`): {_LOAD_ERROR} "
            "Check the model id and that the Space has access to it."
        )
    return (
        "**Model not configured.** Set the `HF_MODEL_ID` Space variable to your published "
        f"merged 16-bit model (placeholder is `{_PLACEHOLDER_MODEL_ID}`; base model is "
        f"`{BASE_MODEL_ID}`). The app is live for inspection but cannot run an audit until a "
        "model is configured."
    )


def _initial_verdict_html() -> str:
    """Return the verdict panel's boot/cleared state.

    A ready model gets the run-an-audit placeholder; an unconfigured or
    failed-to-load model gets the configuration notice, so the degraded state is
    visible the moment the Space opens.
    """
    if not _model_ready():
        return ui.message_html(_status_message(), "missing")
    return ui.verdict_placeholder()


@spaces.GPU(duration=_GPU_DURATION)
def _audit_on_gpu(image: Image.Image, draft_text: str | None) -> AuditOutcome:
    """Run the full audit on the GPU and return a pickle-safe outcome.

    This is the only GPU-touching function. Both models run on the GPU: MedGemma
    grounds the image, then (when the Nemotron draft parser loaded) Nemotron parses
    the draft into the canonical labels and the comparator assembles the audit; if
    Nemotron is unavailable the single-model fallback parses the draft with MedGemma.
    It returns an ``AuditOutcome`` built from plain Python objects and validated
    pydantic models (no CUDA tensors), so it crosses the ZeroGPU process boundary
    cleanly.

    Args:
        image: The uploaded chest X-ray as a PIL image.
        draft_text: The optional draft impression.

    Returns:
        The audit outcome (canonical result plus comparator detail).
    """
    if _DRAFT_FACTORY is not None:
        image_findings = generate_findings(image, model=_MODEL, processor=_PROCESSOR)
        return assemble_audit(image_findings, draft_text, draft_factory=_DRAFT_FACTORY)
    return run_audit(image, draft_text, model=_MODEL, processor=_PROCESSOR)


def on_run(image: Image.Image | None, draft_text: str | None) -> tuple[Image.Image | None, str, str, str]:
    """Gradio run handler: validate inputs, run the audit, build the panels.

    Runs on CPU except for the single ``_audit_on_gpu`` call, which grounds the
    image and parses the draft on the GPU. Returns the annotated overlay image plus
    the three HTML/JSON panel strings - all pickle-safe, CUDA-tensor-free values.

    Args:
        image: The uploaded chest X-ray (``None`` if the user submitted nothing).
        draft_text: The optional draft impression.

    Returns:
        ``(overlay_image, verdict_html, table_html, raw_json)``.

    Raises:
        gr.Error: When no image was uploaded (surfaced as a toast).
    """
    if image is None:
        raise gr.Error("Upload a chest X-ray first (or pick an example below).")
    if not _model_ready():
        return None, ui.message_html(_status_message(), "missing"), ui.table_html([]), ""

    try:
        outcome = _audit_on_gpu(image, draft_text)
    except Exception as exc:
        # Print the full traceback (and, when the exception carries it, the
        # offending raw model text) to stdout so the Space run logs capture every
        # failure, then show an honest category-specific message - GPU quota and
        # scheduling problems are never blamed on the image.
        traceback.print_exc(file=sys.stdout)
        raw_text = getattr(exc, "raw_text", None)
        if isinstance(raw_text, str) and raw_text:
            print(f"[cxr-auditor] offending raw model text >>>\n{raw_text[:_RAW_TEXT_LOG_LIMIT]}\n<<<", flush=True)
        return image, ui.message_html(categorize_serving_error(exc), "unsupported"), ui.table_html([]), ""

    overlay = annotate_evidence(image, outcome)
    verdict = ui.verdict_html(
        urgent=[display_name(label) for label in outcome.result.audit.urgent_review_flags],
        missing=[display_name(label) for label in outcome.result.audit.missing_findings],
        unsupported=[(display_name(claim.finding), claim.draft_span) for claim in outcome.comparison.unsupported],
        draft_parse_note=outcome.draft_parse_note,
    )
    return overlay, verdict, ui.table_html(findings_table_rows(outcome)), result_json(outcome)


def on_clear() -> tuple[None, str, None, str, str, str]:
    """Gradio clear handler: reset the inputs and all result panels.

    Returns:
        Cleared values for ``(image, draft, overlay, verdict, table, json)``.
    """
    return None, "", None, _initial_verdict_html(), ui.table_html([]), ""


def _example_path(name: str) -> str:
    """Return an absolute path under the examples directory for an asset name."""
    return os.path.join(_EXAMPLES_DIR, name)


def build_demo() -> Any:
    """Build the Reading Room gr.Blocks app: inputs left, evidence right.

    The left column holds the numbered input cards (X-ray, draft impression),
    the Run audit / Clear buttons, and the examples; the right column holds the
    image-grounded evidence with the color legend, then the audit verdict, the
    findings table, and the raw JSON, each spanning the full column width and
    stacked in reading order. The theme and CSS are applied at launch time
    (Gradio 6 moved them off the ``gr.Blocks`` constructor).

    Returns:
        The assembled (un-launched) ``gr.Blocks`` app.
    """
    with gr.Blocks(title="CXR Draft Auditor") as demo:
        gr.HTML(
            ui.header_html(
                HF_MODEL_ID.split("/")[-1],
                "Nemotron-3 Nano 4B",
                f"https://huggingface.co/{HF_MODEL_ID}",
                f"https://huggingface.co/{DRAFT_MODEL_ID}",
                "https://huggingface.co/blog/build-small-hackathon/chest-x-ray-draft-auditor",
            )
        )
        gr.HTML(ui.ribbon_html())

        with gr.Row(equal_height=False, elem_classes="rr-row"):
            with gr.Column(scale=4, min_width=380, elem_classes="rr-col"):
                with gr.Group(elem_classes="rr-panel"):
                    gr.HTML(ui.card_head("1", "Chest X-ray"))
                    # Gradio 6 controls the image toolbar via ``buttons``. The built-in
                    # fullscreen button throws "e.onclick is not a function" in the served
                    # bundle (it never opens), so we expose only the download button and drop
                    # the fullscreen and share buttons; the upload/clipboard sources and the
                    # clear control are unaffected.
                    image_in = gr.Image(
                        type="pil",
                        show_label=False,
                        sources=["upload", "clipboard"],
                        elem_id="xray-input",
                        height=320,
                        buttons=["download"],
                    )
                with gr.Group(elem_classes="rr-panel"):
                    gr.HTML(ui.card_head("2", "Draft impression", meta="optional"))
                    draft_in = gr.Textbox(
                        show_label=False,
                        placeholder="e.g. No acute cardiopulmonary abnormality.",
                        lines=4,
                        elem_id="draft-input",
                    )
                with gr.Row():
                    run_btn = gr.Button("Run audit", variant="primary", elem_id="run-btn", scale=4)
                    clear_btn = gr.Button("Clear", variant="secondary", elem_id="clear-btn", scale=1)
                gr.HTML(
                    '<p style="font-size:11.5px;line-height:1.55;color:var(--rr-faint);padding:0 4px;margin:0">'
                    "The auditor grounds findings on the image, parses your draft into the same label space, "
                    "and flags what is missing, unsupported, or urgent. MedGemma grounding and the NVIDIA "
                    "Nemotron draft parse both run on ZeroGPU, so an audit takes about half a minute.</p>"
                )
                # Illustrative examples reference asset paths under examples/. Only
                # those whose image file is actually present are registered: gradio
                # rejects an example pointing at a missing file (InvalidPathError),
                # so an unguarded list breaks the app whenever the images are
                # absent. Drop chest X-rays you are licensed to redistribute into
                # examples/ to populate this section; it appears automatically once
                # at least one example image exists.
                candidate_examples = [
                    [_example_path("cxr_example_1.png"), "No acute cardiopulmonary abnormality."],
                    [_example_path("cxr_example_2.png"), "Lungs are clear. No focal consolidation, mass, or effusion."],
                    [_example_path("cxr_example_3.png"), "Moderate left pleural effusion."],
                    [_example_path("cxr_example_4.png"), "Heart and lungs unremarkable. No acute abnormality."],
                ]
                available_examples = [example for example in candidate_examples if os.path.isfile(example[0])]
                if available_examples:
                    gr.Examples(
                        examples=available_examples,
                        inputs=[image_in, draft_in],
                        label="Examples (clean draft / draft that misses a finding / draft that over-calls a finding / draft that misses an urgent finding)",
                    )

            with gr.Column(scale=6, elem_classes="rr-col"):
                with gr.Group(elem_classes="rr-panel"):
                    gr.HTML(ui.card_head(None, "Image-grounded evidence", legend=True))
                    # This display image's fullscreen button works (unlike the editable input
                    # image's, which errors in the served bundle), so keep download + fullscreen
                    # here for inspecting the evidence boxes; drop only the share button.
                    overlay_out = gr.Image(
                        show_label=False,
                        elem_id="overlay-out",
                        height=520,
                        interactive=False,
                        buttons=["download", "fullscreen"],
                    )
                # The audit verdict, the findings table, and the raw JSON each span
                # the full width of the evidence column and stack in reading order:
                # the plain-English verdict first, then the structured findings, then
                # the raw AuditResult JSON last.
                with gr.Group(elem_classes="rr-panel"):
                    gr.HTML(ui.card_head(None, "Audit verdict"))
                    verdict_out = gr.HTML(_initial_verdict_html(), elem_classes="rr-pad")
                with gr.Group(elem_classes="rr-panel"):
                    gr.HTML(ui.card_head(None, "Findings"))
                    table_out = gr.HTML(ui.table_html([]), elem_classes="rr-tablewrap")
                with gr.Group(elem_classes="rr-panel"):
                    gr.HTML(ui.card_head(None, "Raw JSON", meta="AuditResult"))
                    json_out = gr.Code(language="json", show_label=False, elem_id="json-out", lines=12)

        gr.HTML(ui.footer_html(HF_MODEL_ID, DRAFT_MODEL_ID))

        run_btn.click(fn=on_run, inputs=[image_in, draft_in], outputs=[overlay_out, verdict_out, table_out, json_out])
        clear_btn.click(fn=on_clear, outputs=[image_in, draft_in, overlay_out, verdict_out, table_out, json_out])

    return demo


demo = build_demo()


if __name__ == "__main__":
    # Gradio 6 moved the app-level theme/css parameters from the gr.Blocks
    # constructor to launch(); passing them here is the non-deprecated API. The
    # head script defaults a first visit to dark and wires the header light/dark
    # toggle (it reloads with ?__theme= and remembers the choice in localStorage).
    demo.launch(theme=ui.build_theme(), css=ui.CSS, head=ui.HEAD)
