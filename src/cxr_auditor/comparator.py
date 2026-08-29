"""
Deterministic audit comparator: the product's trust core.

This module compares the image-derived findings (each carrying a label, a
present/absent status, and an optional bounding box) against the draft-derived
findings (each carrying a label and a present/absent status) and produces the
canonical ``Audit`` object plus the per-item evidence the user interface renders.

The comparator is pure logic: it depends only on the standard library, the
canonical finding-set definitions (``cxr_auditor.findings``), and the schema
types (``cxr_auditor.schema``). It never imports torch, transformers, or any
serving stack, so it is fully unit-testable without a GPU.

Audit rules
-----------
Let an image finding be *present* when its status is ``PRESENT`` and its label is
a canonical positive (``no_finding`` is the negative sentinel, never a positive
claim). Let a draft finding be *asserted* when its status is ``PRESENT`` and its
label is a canonical positive.

- MISSING: a positive finding present on the image whose label is absent from the
  draft entirely, or present in the draft with status ``ABSENT`` (explicitly
  denied). These are findings the image supports that the draft fails to report.
- UNSUPPORTED: a positive finding asserted by the draft whose label is absent from
  the image findings (or present there only with status ``ABSENT``). These are
  draft claims the image does not support.
- URGENT: any image-present positive finding whose label is on the urgent-review
  whitelist (``findings.URGENT_WHITELIST`` via ``findings.is_urgent``).

Matching is label-level (set-based), not per-lesion: every rule above keys on the
canonical label, so a label is judged present-or-absent once, no matter how many
boxes the image model localized for it. A finding the model grounds at two
spatially-distinct sites yields at most one MISSING, one UNSUPPORTED, and one
URGENT entry for that label; a second, separate lesion carrying the same label
that the draft happens to omit is not flagged on its own. The overlay still draws
every distinct box (see ``render``), so the overlay can legitimately show more
boxes than the audit lists or the findings table have rows for that label.

The canonical ordering of ``CANONICAL_FINDINGS`` is used for every output list so
the result is deterministic regardless of input ordering and free of duplicates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from cxr_auditor.findings import CANONICAL_FINDINGS, NO_FINDING, is_urgent
from cxr_auditor.schema import Audit, DraftFinding, FindingStatus, ImageFinding, NormalizedBox

# Canonical sort key: index of a label in the fixed presentation order. Labels
# are always canonical here (the schema validates them on construction), so a
# missing label never occurs; the lookup is total over the inputs.
_CANONICAL_ORDER: dict[str, int] = {label: index for index, label in enumerate(CANONICAL_FINDINGS)}


@dataclass(frozen=True, slots=True)
class MissingFinding:
    """A finding present on the image but absent or denied in the draft.

    Attributes:
        finding: The canonical positive finding label.
        box: The image bounding box for this finding, if the model localized one.
        urgent: Whether this finding is on the urgent-review whitelist.
    """

    finding: str
    box: NormalizedBox | None = None
    urgent: bool = False


@dataclass(frozen=True, slots=True)
class UnsupportedClaim:
    """A finding asserted by the draft but not supported by the image.

    Attributes:
        finding: The canonical positive finding label.
        draft_span: The verbatim draft phrase that produced the label, if known.
    """

    finding: str
    draft_span: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """The full comparator result with per-item evidence.

    ``audit`` is the canonical ``Audit`` object (plain label lists) that is
    embedded in the serialized ``AuditResult``. The detail lists carry the
    evidence (boxes, urgency flags, draft spans) the user interface renders
    alongside each flagged item. All lists are sorted in canonical finding order
    and deduplicated by label.

    Attributes:
        missing: Detailed missing findings (image-present, draft-absent/denied).
        unsupported: Detailed unsupported claims (draft-asserted, image-absent).
        urgent: Detailed urgent findings (image-present, on the whitelist).
        audit: The canonical label-only ``Audit`` for serialization.
    """

    missing: list[MissingFinding] = field(default_factory=list)
    unsupported: list[UnsupportedClaim] = field(default_factory=list)
    urgent: list[MissingFinding] = field(default_factory=list)
    audit: Audit = field(default_factory=Audit)


def _canonical_sort_key(label: str) -> int:
    """Return the canonical presentation index for a label."""
    return _CANONICAL_ORDER[label]


def _present_image_findings(image_findings: Iterable[ImageFinding]) -> dict[str, ImageFinding]:
    """Index positive, present image findings by label.

    Findings whose status is ``ABSENT`` or whose label is the ``no_finding``
    sentinel are excluded: only positive findings the image asserts as present
    participate in the audit. When the same positive label appears more than once
    with status ``PRESENT``, the first occurrence (in input order) wins, so its
    box and evidence are the ones surfaced. Indexing by label makes the audit
    label-level: a second present box of the same label (a spatially-distinct
    lesion) collapses into this one entry and is not compared against the draft on
    its own, so a draft that omits only that second site raises no separate flag.
    The overlay still draws every distinct box (see ``render``).
    """
    present: dict[str, ImageFinding] = {}
    for finding in image_findings:
        if finding.status is not FindingStatus.PRESENT:
            continue
        if finding.finding == NO_FINDING:
            continue
        present.setdefault(finding.finding, finding)
    return present


def _asserted_draft_findings(draft_findings: Iterable[DraftFinding]) -> dict[str, DraftFinding]:
    """Index positive, asserted draft findings by label.

    A label asserted present (status ``PRESENT``, not the ``no_finding``
    sentinel) is a positive claim. The first present occurrence wins so its span
    is surfaced. Labels the draft only denies (status ``ABSENT``) are not
    asserted and are excluded here.
    """
    asserted: dict[str, DraftFinding] = {}
    for finding in draft_findings:
        if finding.status is not FindingStatus.PRESENT:
            continue
        if finding.finding == NO_FINDING:
            continue
        asserted.setdefault(finding.finding, finding)
    return asserted


def compare(
    image_findings: Sequence[ImageFinding],
    draft_findings: Sequence[DraftFinding],
) -> ComparisonReport:
    """Compare image findings against draft findings deterministically.

    This is the auditor's trust core. It applies the MISSING / UNSUPPORTED /
    URGENT rules over the canonical positive label space and returns a
    ``ComparisonReport`` carrying both the canonical ``Audit`` (label-only lists,
    for serialization) and the per-item detail lists (boxes, urgency flags, draft
    spans) the user interface renders.

    The ``no_finding`` sentinel is treated as "no positive asserted": it never
    appears in any output list, and an image or draft finding carrying it
    contributes no positive claim. A draft that supplies no findings at all (empty
    sequence) yields no missing or unsupported items; the image findings still
    drive urgent flags.

    Args:
        image_findings: The image-grounded findings (label, status, optional box).
        draft_findings: The draft-derived findings (label, status). Empty when no
            draft was supplied.

    Returns:
        A ``ComparisonReport`` whose lists are sorted in canonical finding order
        and deduplicated by label.
    """
    present_image = _present_image_findings(image_findings)
    asserted_draft = _asserted_draft_findings(draft_findings)

    missing: list[MissingFinding] = []
    urgent: list[MissingFinding] = []
    for label in sorted(present_image, key=_canonical_sort_key):
        image_finding = present_image[label]
        label_is_urgent = is_urgent(label)
        if label_is_urgent:
            urgent.append(MissingFinding(finding=label, box=image_finding.box, urgent=True))
        if label not in asserted_draft:
            missing.append(MissingFinding(finding=label, box=image_finding.box, urgent=label_is_urgent))

    unsupported: list[UnsupportedClaim] = []
    for label in sorted(asserted_draft, key=_canonical_sort_key):
        if label not in present_image:
            draft_finding = asserted_draft[label]
            unsupported.append(UnsupportedClaim(finding=label, draft_span=draft_finding.span))

    audit = Audit(
        missing_findings=[item.finding for item in missing],
        unsupported_claims=[item.finding for item in unsupported],
        urgent_review_flags=[item.finding for item in urgent],
    )
    return ComparisonReport(missing=missing, unsupported=unsupported, urgent=urgent, audit=audit)


def build_audit(
    image_findings: Sequence[ImageFinding],
    draft_findings: Sequence[DraftFinding],
) -> Audit:
    """Return only the canonical ``Audit`` for the given findings.

    Convenience wrapper over ``compare`` for callers that need the label-only
    ``Audit`` object (the part embedded in the serialized ``AuditResult``) and do
    not need the per-item boxes or spans.

    Args:
        image_findings: The image-grounded findings.
        draft_findings: The draft-derived findings.

    Returns:
        The canonical ``Audit`` object.
    """
    return compare(image_findings, draft_findings).audit
