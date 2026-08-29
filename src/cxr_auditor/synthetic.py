"""
Synthetic draft generation and labeled audit-case construction.

This module turns an image's known canonical positive findings into a realistic
draft radiology impression (via templated sentences) and then produces the three
labeled audit cases the auditor is evaluated against:

1. ``missing`` -- drop a present finding from the draft so the comparator must
   flag it as a missing finding.
2. ``unsupported`` -- add a finding the image does not contain so the comparator
   must flag it as an unsupported claim.
3. ``faithful`` -- a control draft that faithfully mentions exactly the image
   findings, so the comparator flags nothing.

Every generated case carries the ground-truth expected ``Audit`` labels, computed
deterministically from the image findings and the corruption applied, so the
synthetic corpus doubles as an evaluation set for the deterministic comparator.

Determinism
-----------
All randomness flows through an explicitly-seeded ``random.Random`` instance. The
seed is a required argument on the public entry points; the module never touches
the global ``random`` state and never reads the system clock. Identical seeds and
identical inputs always reproduce identical drafts and cases.

LLM paraphrase hook
-------------------
``generate_draft`` accepts an optional ``paraphraser`` callable so a downstream
caller can swap the templated text for an LLM paraphrase. It is off by default and
this module performs no network access; the templated generator is fully offline
and is the single source of truth for the label-to-text mapping.

Dependencies are limited to the standard library and the pure-logic schema/findings
modules, so the whole module imports and unit-tests without any model-serving stack.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

from cxr_auditor.findings import (
    CANONICAL_FINDING_SET,
    NO_FINDING,
    POSITIVE_FINDINGS,
    is_urgent,
)
from cxr_auditor.schema import (
    Audit,
    DraftFinding,
    FindingStatus,
    ImageFinding,
    normalize_finding,
)

# A paraphraser turns a templated draft impression into an alternative phrasing.
# It MUST be label-preserving: the returned text is treated as a stylistic variant
# of the same draft, so the expected audit labels are unchanged. Implementations
# are supplied by the caller (for example an LLM wrapper); none ships here.
type Paraphraser = Callable[[str], str]


class AuditCaseType(str, Enum):
    """The kind of corruption applied to a faithful draft to build a case.

    ``MISSING`` drops one image-present finding so the draft omits it.
    ``UNSUPPORTED`` adds one image-absent finding so the draft over-claims it.
    ``FAITHFUL`` applies no corruption and serves as the negative control.
    """

    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    FAITHFUL = "faithful"


# Templated impression sentences keyed by canonical finding. Each finding maps to
# a tuple of interchangeable phrasings; the seeded RNG selects one per draft so the
# corpus is varied yet reproducible. Every phrase is a plausible impression-line
# fragment that asserts the finding as present. This table is the single source of
# truth mapping a canonical label to draft prose.
_PRESENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "pleural_effusion": (
        "Blunting of the costophrenic angle consistent with a pleural effusion.",
        "Layering pleural effusion.",
        "Pleural effusion is present.",
    ),
    "pneumothorax": (
        "Pneumothorax is present.",
        "Lucency with absent lung markings consistent with a pneumothorax.",
        "Apical pneumothorax.",
    ),
    "lung_opacity_consolidation": (
        "Airspace opacity consistent with consolidation.",
        "Focal consolidation is present.",
        "Patchy lung opacity.",
    ),
    "nodule_mass": (
        "A pulmonary nodule is present.",
        "Mass-like opacity consistent with a nodule or mass.",
        "Nodular opacity is seen.",
    ),
    "cardiomegaly": (
        "Cardiomegaly with an enlarged cardiac silhouette.",
        "The cardiac silhouette is enlarged consistent with cardiomegaly.",
        "Cardiomegaly is present.",
    ),
}

# Impression line used when no positive finding is asserted (the negative sentinel).
_NO_FINDING_TEMPLATES: tuple[str, ...] = (
    "No acute cardiopulmonary abnormality.",
    "Clear lung fields with a normal cardiomediastinal silhouette.",
    "No acute findings.",
)


def _coerce_image_findings(
    findings: Iterable[ImageFinding | Mapping[str, Any]],
) -> list[ImageFinding]:
    """Coerce mixed image-finding inputs into validated ``ImageFinding`` models.

    Accepts already-built ``ImageFinding`` instances or plain dicts (which are
    validated through the pydantic model) so callers can pass either the parsed
    schema objects or raw dataset rows.
    """
    coerced: list[ImageFinding] = []
    for item in findings:
        if isinstance(item, ImageFinding):
            coerced.append(item)
        else:
            coerced.append(ImageFinding.model_validate(item))
    return coerced


def present_positive_labels(
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
) -> list[str]:
    """Return the canonical positive labels asserted present on the image.

    The negative sentinel ``no_finding`` and any finding with status
    ``ABSENT`` are excluded. Order follows the canonical finding order so the
    result is deterministic and duplicate labels collapse to one entry.

    Args:
        image_findings: The image-grounded findings (models or dicts).

    Returns:
        The sorted-by-canonical-order list of distinct present positive labels.
    """
    coerced = _coerce_image_findings(image_findings)
    present = {
        finding.finding for finding in coerced if finding.status is FindingStatus.PRESENT and finding.finding != NO_FINDING
    }
    return [label for label in POSITIVE_FINDINGS if label in present]


def absent_positive_labels(present_labels: Sequence[str]) -> list[str]:
    """Return the canonical positive labels NOT present on the image.

    These are the candidates an ``unsupported`` case can over-claim. Order follows
    the canonical finding order.

    Args:
        present_labels: The labels currently present on the image.

    Returns:
        The canonical positive labels absent from ``present_labels``.
    """
    present_set = set(present_labels)
    return [label for label in POSITIVE_FINDINGS if label not in present_set]


def _render_impression(
    labels: Sequence[str],
    rng: random.Random,
) -> str:
    """Render a templated draft impression for the given present labels.

    One phrasing is drawn per label from ``_PRESENT_TEMPLATES`` via ``rng`` and the
    sentences are joined in canonical order. An empty label set renders a single
    negative-study line drawn from ``_NO_FINDING_TEMPLATES``.
    """
    if not labels:
        return rng.choice(_NO_FINDING_TEMPLATES)
    ordered = [label for label in POSITIVE_FINDINGS if label in set(labels)]
    sentences = [rng.choice(_PRESENT_TEMPLATES[label]) for label in ordered]
    return " ".join(sentences)


def generate_draft(
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
    seed: int,
    *,
    paraphraser: Paraphraser | None = None,
) -> str:
    """Generate a faithful templated draft impression for an image's findings.

    Produces an impression that mentions exactly the image-present positive
    findings (or a negative-study line when there are none). The text is
    deterministic given ``seed`` and the inputs.

    Args:
        image_findings: The image-grounded findings (models or dicts). Only
            present positive findings drive the generated sentences.
        seed: Seed for the local ``random.Random`` instance. Required; identical
            seeds reproduce identical drafts.
        paraphraser: Optional label-preserving callable that rephrases the
            templated draft. Off by default. When supplied it is invoked with the
            templated text and its return value is used instead. The caller is
            responsible for keeping the paraphrase label-preserving; this module
            never alters the expected labels based on the paraphrase.

    Returns:
        The draft impression text.
    """
    rng = random.Random(seed)
    present = present_positive_labels(image_findings)
    draft = _render_impression(present, rng)
    if paraphraser is not None:
        draft = paraphraser(draft)
    return draft


def _draft_findings_for_labels(
    present_labels: Sequence[str],
    rng: random.Random,
) -> list[DraftFinding]:
    """Build the parsed ``DraftFinding`` list a faithful draft would yield.

    Mirrors what the draft parser extracts from the rendered impression: one
    ``PRESENT`` entry per positive label, or a single ``no_finding`` entry when
    the label set is empty. The verbatim ``span`` is the rendered sentence so the
    draft findings are traceable back to the draft text.
    """
    if not present_labels:
        return [
            DraftFinding(
                finding=NO_FINDING,
                status=FindingStatus.PRESENT,
                span=rng.choice(_NO_FINDING_TEMPLATES),
            )
        ]
    ordered = [label for label in POSITIVE_FINDINGS if label in set(present_labels)]
    return [
        DraftFinding(
            finding=label,
            status=FindingStatus.PRESENT,
            span=rng.choice(_PRESENT_TEMPLATES[label]),
        )
        for label in ordered
    ]


def compute_expected_audit(
    image_present_labels: Sequence[str],
    draft_findings: Sequence[DraftFinding],
) -> Audit:
    """Compute the deterministic comparator result for known finding sets.

    This is the same comparator the auditor applies, expressed over the pure label
    sets so the synthetic corpus can record its own ground-truth expected labels.

    - MISSING: an image-present positive label that the draft does not assert
      present (the draft omits it or marks it ``ABSENT``).
    - UNSUPPORTED: a positive label the draft asserts present that the image does
      not contain (``no_finding`` is the negative sentinel, never an unsupported
      claim).
    - URGENT: any image-present label on the urgent-review whitelist.

    Each list is ordered by canonical finding order and de-duplicated.

    Args:
        image_present_labels: The positive labels present on the image.
        draft_findings: The findings parsed from the draft.

    Returns:
        The expected ``Audit``.
    """
    image_present = {label for label in image_present_labels if label in CANONICAL_FINDING_SET and label != NO_FINDING}
    draft_present = {
        finding.finding
        for finding in draft_findings
        if finding.status is FindingStatus.PRESENT and finding.finding != NO_FINDING
    }

    missing = image_present - draft_present
    unsupported = draft_present - image_present
    urgent = {label for label in image_present if is_urgent(label)}

    def _ordered(labels: set[str]) -> list[str]:
        return [label for label in POSITIVE_FINDINGS if label in labels]

    return Audit(
        missing_findings=_ordered(missing),
        unsupported_claims=_ordered(unsupported),
        urgent_review_flags=_ordered(urgent),
    )


class AuditCase:
    """A single labeled synthetic audit case.

    Bundles the corruption type, the image-present positive labels, the generated
    draft text, the draft findings that text parses to, and the ground-truth
    expected ``Audit`` the deterministic comparator must reproduce.

    Attributes:
        case_type: Which corruption produced this case.
        image_present_labels: The positive labels present on the image.
        draft_text: The generated (possibly corrupted) draft impression.
        draft_findings: The findings the draft text parses to, in the canonical
            label space.
        expected_audit: The ground-truth comparator result for this case.
        corrupted_label: The single label dropped (``MISSING``) or added
            (``UNSUPPORTED``); ``None`` for a ``FAITHFUL`` control.
    """

    __slots__ = (
        "case_type",
        "image_present_labels",
        "draft_text",
        "draft_findings",
        "expected_audit",
        "corrupted_label",
    )

    def __init__(
        self,
        *,
        case_type: AuditCaseType,
        image_present_labels: Sequence[str],
        draft_text: str,
        draft_findings: Sequence[DraftFinding],
        expected_audit: Audit,
        corrupted_label: str | None,
    ) -> None:
        self.case_type = case_type
        self.image_present_labels = list(image_present_labels)
        self.draft_text = draft_text
        self.draft_findings = list(draft_findings)
        self.expected_audit = expected_audit
        self.corrupted_label = corrupted_label

    def __repr__(self) -> str:
        return (
            f"AuditCase(case_type={self.case_type.value!r}, "
            f"corrupted_label={self.corrupted_label!r}, "
            f"image_present_labels={self.image_present_labels!r})"
        )


def _build_case(
    case_type: AuditCaseType,
    image_present_labels: Sequence[str],
    draft_present_labels: Sequence[str],
    corrupted_label: str | None,
    rng: random.Random,
    paraphraser: Paraphraser | None,
) -> AuditCase:
    """Assemble one ``AuditCase`` from the image and draft label sets.

    ``draft_present_labels`` is the set of positive labels the draft asserts (which
    may differ from the image set by exactly the corruption). The draft text and
    findings are rendered from that set; the expected audit is computed from the
    image set versus the draft findings.
    """
    draft_findings = _draft_findings_for_labels(draft_present_labels, rng)
    draft_text = _render_impression(draft_present_labels, rng)
    if paraphraser is not None:
        draft_text = paraphraser(draft_text)
    expected = compute_expected_audit(image_present_labels, draft_findings)
    return AuditCase(
        case_type=case_type,
        image_present_labels=image_present_labels,
        draft_text=draft_text,
        draft_findings=draft_findings,
        expected_audit=expected,
        corrupted_label=corrupted_label,
    )


def make_audit_cases(
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
    seed: int,
    *,
    paraphraser: Paraphraser | None = None,
) -> list[AuditCase]:
    """Build the three labeled audit cases for one image's findings.

    Generates, in order, the ``missing``, ``unsupported``, and ``faithful`` cases.
    Each case is deterministic given ``seed`` and the inputs.

    Case construction:
    - ``missing``: drop one image-present finding from the draft. Requires at least
      one present finding; when the image has none this case is omitted (there is
      nothing to drop). The dropped finding is the first present label in canonical
      order, so the choice is reproducible.
    - ``unsupported``: add one image-absent finding to the draft. The added finding
      is the first absent positive label in canonical order. When the image already
      contains every positive finding this case is omitted (there is nothing to
      add).
    - ``faithful``: the draft asserts exactly the image findings; the expected audit
      is empty (apart from urgent flags, which fire on any present urgent finding
      regardless of the draft).

    Args:
        image_findings: The image-grounded findings (models or dicts).
        seed: Seed for deterministic generation. Required.
        paraphraser: Optional label-preserving paraphrase hook, off by default.

    Returns:
        The list of generated ``AuditCase`` objects, omitting any case that cannot
        be constructed for the given image (an image with no present findings has
        no ``missing`` case; an image with all positives has no ``unsupported``
        case). The ``faithful`` control is always present.
    """
    rng = random.Random(seed)
    present = present_positive_labels(image_findings)
    absent = absent_positive_labels(present)

    cases: list[AuditCase] = []

    if present:
        dropped = present[0]
        draft_labels = [label for label in present if label != dropped]
        cases.append(
            _build_case(
                AuditCaseType.MISSING,
                present,
                draft_labels,
                dropped,
                rng,
                paraphraser,
            )
        )

    if absent:
        added = absent[0]
        draft_labels = [label for label in POSITIVE_FINDINGS if label in set(present) | {added}]
        cases.append(
            _build_case(
                AuditCaseType.UNSUPPORTED,
                present,
                draft_labels,
                added,
                rng,
                paraphraser,
            )
        )

    cases.append(
        _build_case(
            AuditCaseType.FAITHFUL,
            present,
            present,
            None,
            rng,
            paraphraser,
        )
    )

    return cases


def make_audit_case_records(
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
    seed: int,
    *,
    image_id: str | None = None,
    paraphraser: Paraphraser | None = None,
) -> list[dict[str, object]]:
    """Build the three audit cases as plain JSON-serializable dict records.

    Convenience wrapper over ``make_audit_cases`` that flattens each case into a
    dict suitable for writing to JSONL or feeding an evaluation harness. The draft
    findings and expected audit are serialized via their pydantic models.

    Args:
        image_findings: The image-grounded findings (models or dicts).
        seed: Seed for deterministic generation. Required.
        image_id: Optional identifier carried onto every record for traceability.
        paraphraser: Optional label-preserving paraphrase hook, off by default.

    Returns:
        A list of dict records, one per constructed case.
    """
    records: list[dict[str, object]] = []
    for case in make_audit_cases(image_findings, seed, paraphraser=paraphraser):
        record: dict[str, object] = {
            "case_type": case.case_type.value,
            "image_present_labels": case.image_present_labels,
            "draft_text": case.draft_text,
            "draft_findings": [finding.model_dump() for finding in case.draft_findings],
            "expected_audit": case.expected_audit.model_dump(),
            "corrupted_label": case.corrupted_label,
        }
        if image_id is not None:
            record["image_id"] = image_id
        records.append(record)
    return records


def normalize_present_labels(labels: Iterable[str]) -> list[str]:
    """Normalize and validate a raw label list to distinct canonical positives.

    Each label is normalized via ``normalize_finding`` (so ``'Pleural Effusion'``
    and ``'pleural-effusion'`` both canonicalize), the negative sentinel is
    dropped, and the result is ordered by canonical finding order with duplicates
    collapsed.

    Args:
        labels: Raw positive labels (any casing / separator style).

    Returns:
        The distinct canonical positive labels in canonical order.

    Raises:
        ValueError: If a label does not normalize to a canonical finding.
    """
    canonical = {normalize_finding(label) for label in labels}
    canonical.discard(NO_FINDING)
    return [label for label in POSITIVE_FINDINGS if label in canonical]
