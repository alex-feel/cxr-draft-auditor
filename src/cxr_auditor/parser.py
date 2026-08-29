"""
Draft-report parser: map a draft impression into the canonical label space.

This is the PRIMARY draft parser. It prompts an injected text model (via the
pinned draft-parsing prompt) to extract which canonical findings a draft
impression asserts present and which it explicitly denies, then validates the
model's JSON list into ``DraftFinding`` objects. The serving app backs that
injected model with NVIDIA Nemotron-3 Nano 4B; this module stays model-agnostic.

The model is injected as a ``generate_fn`` callable rather than imported here, so
this module is pure logic: it has no torch/transformers dependency and is fully
unit-testable with a fake ``generate_fn`` that returns canned JSON. The lazy
model wiring lives in ``cxr_auditor.inference``.

An OPTIONAL, OFF-BY-DEFAULT CheXbert cross-check is provided as a documented
secondary labeler. It lazily imports ``f1chexbert`` only when explicitly invoked,
so it never becomes a hard dependency of the package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cxr_auditor.findings import CANONICAL_FINDING_SET, NO_FINDING
from cxr_auditor.prompts import build_draft_parsing_prompt
from cxr_auditor.schema import DraftFinding, FindingStatus, SchemaParseError, extract_finding_list, normalize_finding

# A ``generate_fn`` takes a fully-rendered text prompt and returns the model's raw
# text completion. The image-grounding path uses a different signature (it also
# takes an image); the draft parser only needs text in, text out.
type GenerateFn = Callable[[str], str]


def _coerce_status(raw: Any) -> FindingStatus:
    """Coerce a raw ``status`` value from model JSON into a ``FindingStatus``.

    The model is instructed to emit ``"present"`` or ``"absent"``. A missing or
    null status defaults to ``PRESENT`` (the model asserted the label by listing
    it). Any other string is matched case-insensitively; an unrecognized value
    falls back to ``PRESENT`` rather than discarding the finding, because a label
    the model chose to emit is a positive signal even when the status word drifts.
    """
    if raw is None:
        return FindingStatus.PRESENT
    if isinstance(raw, FindingStatus):
        return raw
    text = str(raw).strip().lower()
    if text == FindingStatus.ABSENT.value:
        return FindingStatus.ABSENT
    return FindingStatus.PRESENT


def parse_draft_findings(raw_text: str) -> list[DraftFinding]:
    """Parse a model's raw draft-label JSON into validated ``DraftFinding`` list.

    Performs tolerant JSON-array extraction (handling markdown fences and a bare
    object) and validates each element. Elements whose ``label`` is not a
    canonical finding are dropped rather than raising, so a single hallucinated
    out-of-vocabulary label does not discard the entire parse; the kept labels are
    deduplicated by ``(finding, status)`` preserving first-seen order.

    Args:
        raw_text: The model's raw completion for the draft-parsing prompt.

    Returns:
        Validated ``DraftFinding`` objects in first-seen order.

    Raises:
        SchemaParseError: If no JSON array/object can be extracted at all. The raw
            text is attached for a retry/repair loop.
    """
    elements = extract_finding_list(raw_text)

    findings: list[DraftFinding] = []
    seen: set[tuple[str, FindingStatus]] = set()
    for element in elements:
        label_raw = element.get("label")
        if not isinstance(label_raw, str):
            continue
        try:
            label = normalize_finding(label_raw)
        except ValueError:
            # Out-of-vocabulary label: skip this element, keep the rest.
            continue
        status = _coerce_status(element.get("status"))
        key = (label, status)
        if key in seen:
            continue
        seen.add(key)
        span = element.get("span")
        findings.append(
            DraftFinding(
                finding=label,
                status=status,
                span=span if isinstance(span, str) else None,
            )
        )
    return findings


def parse_draft(draft_text: str, generate_fn: GenerateFn) -> list[DraftFinding]:
    """Parse a draft impression into canonical ``DraftFinding`` objects.

    Builds the pinned draft-parsing prompt, calls the injected ``generate_fn`` to
    obtain the model's raw completion, and validates the result. The model is
    injected so this function is testable with a fake ``generate_fn`` returning
    canned JSON, with no model, network, or GPU.

    A draft that the model maps to the ``no_finding`` sentinel returns that single
    sentinel finding; callers (the comparator) treat it as "no positive asserted".

    Args:
        draft_text: The draft radiology impression to parse. Must be non-empty.
        generate_fn: Callable mapping a rendered text prompt to the model's raw
            text completion.

    Returns:
        Validated ``DraftFinding`` objects.

    Raises:
        ValueError: If ``draft_text`` is empty or whitespace only (raised by the
            prompt builder).
        SchemaParseError: If the model output cannot be parsed into a finding
            list at all.
    """
    prompt = build_draft_parsing_prompt(draft_text)
    raw_text = generate_fn(prompt)
    return parse_draft_findings(raw_text)


def chexbert_cross_check(draft_text: str, model_path: str | None = None) -> dict[str, FindingStatus]:
    """OPTIONAL, OFF-BY-DEFAULT CheXbert cross-check of a draft impression.

    This is a documented secondary labeler, NOT part of the primary parse path
    and NOT a package dependency. It lazily imports ``f1chexbert`` only when
    called, so installing or importing ``cxr_auditor`` never requires CheXbert.
    Enable it explicitly (and install the optional dependency yourself) when you
    want an independent label opinion to compare against the primary parser.

    CheXbert labels the 14 CheXpert conditions. Only the conditions that map onto
    the canonical six-finding set are returned; CheXbert's ``positive`` label maps
    to ``PRESENT`` and its ``negative`` label maps to ``ABSENT``. Uncertain and
    blank labels are omitted (the draft did not commit to the finding).

    Args:
        draft_text: The draft radiology impression to label.
        model_path: Optional path to a local ``chexbert.pth`` checkpoint. When
            ``None``, ``f1chexbert`` resolves its own default checkpoint.

    Returns:
        A mapping from canonical finding label to ``FindingStatus`` for the
        conditions CheXbert committed to.

    Raises:
        RuntimeError: If ``f1chexbert`` is not installed. The message points at
            the optional install so the cross-check stays opt-in.
    """
    # Lazy, dynamic import keeps f1chexbert (and its torch dependency) off the
    # package import path so the cross-check stays a documented opt-in, never a
    # hard dependency. importlib is used so static type-checkers do not require
    # the optional package to be installed to type-check this module.
    import importlib

    try:
        f1chexbert_module = importlib.import_module("f1chexbert")
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise RuntimeError(
            "CheXbert cross-check requires the optional 'f1chexbert' package, which is "
            "off by default. Install it explicitly (pip install f1chexbert) and supply a "
            "chexbert.pth checkpoint (for example from StanfordAIMI/RRG_scorers) to enable "
            "this secondary labeler."
        ) from exc
    f1_chexbert_cls = f1chexbert_module.F1CheXbert

    # CheXpert condition name (as f1chexbert returns) -> canonical finding label.
    # Conditions with no canonical counterpart are omitted from this map so they
    # are ignored. Names are lowercased for a case-insensitive lookup.
    chexpert_to_canonical = {
        "pleural effusion": "pleural_effusion",
        "pneumothorax": "pneumothorax",
        "consolidation": "lung_opacity_consolidation",
        "lung opacity": "lung_opacity_consolidation",
        "lung lesion": "nodule_mass",
        "cardiomegaly": "cardiomegaly",
        "no finding": NO_FINDING,
    }

    labeler = f1_chexbert_cls(model_path=model_path) if model_path is not None else f1_chexbert_cls()
    # ``get_label`` returns per-condition labels for a single report. The public
    # f1chexbert API exposes the condition list as ``labeler.target_names`` aligned
    # with the returned label vector.
    raw_labels = labeler.get_label(draft_text)

    result: dict[str, FindingStatus] = {}
    for condition, label_value in zip(labeler.target_names, raw_labels, strict=False):
        canonical = chexpert_to_canonical.get(condition.strip().lower())
        if canonical is None or canonical not in CANONICAL_FINDING_SET:
            continue
        status = _chexbert_label_to_status(label_value)
        if status is not None:
            result[canonical] = status
    return result


def _chexbert_label_to_status(label_value: Any) -> FindingStatus | None:
    """Map a CheXbert per-condition label code to a ``FindingStatus``.

    CheXbert emits ``1`` (positive), ``0`` (negative), ``-1`` (uncertain), and a
    blank for an unmentioned condition. Positive maps to ``PRESENT``, negative to
    ``ABSENT``; uncertain and blank yield ``None`` (no committed status).
    """
    try:
        code = int(label_value)
    except (TypeError, ValueError):
        return None
    if code == 1:
        return FindingStatus.PRESENT
    if code == 0:
        return FindingStatus.ABSENT
    return None


__all__ = [
    "GenerateFn",
    "SchemaParseError",
    "chexbert_cross_check",
    "parse_draft",
    "parse_draft_findings",
]
