"""
Tests for the deterministic audit comparator.

These tests exercise the trust core exhaustively: every combination of a
finding's state on the image side (present, absent, silent) crossed with its
state on the draft side (asserted present, denied absent, silent), plus the
``no_finding`` sentinel handling, urgent-whitelist behavior, ordering,
deduplication, and the canonical ``Audit`` projection.

All inputs are tiny synthetic schema objects; no model, network, or GPU is used.
"""

from __future__ import annotations

import pytest

from cxr_auditor.comparator import (
    ComparisonReport,
    MissingFinding,
    UnsupportedClaim,
    build_audit,
    compare,
)
from cxr_auditor.findings import NO_FINDING
from cxr_auditor.schema import Audit, DraftFinding, FindingStatus, ImageFinding

EFFUSION = "pleural_effusion"
PNEUMO = "pneumothorax"
OPACITY = "lung_opacity_consolidation"
NODULE = "nodule_mass"
CARDIO = "cardiomegaly"

EFFUSION_BOX = (0.62, 0.08, 0.94, 0.40)
PNEUMO_BOX = (0.10, 0.55, 0.45, 0.95)
NODULE_BOX = (0.236, 0.316, 0.386, 0.422)


def img(label: str, status: FindingStatus = FindingStatus.PRESENT, box=None) -> ImageFinding:
    """Build an image finding tersely for table-driven tests."""
    return ImageFinding(finding=label, status=status, box=box)


def draft(label: str, status: FindingStatus = FindingStatus.PRESENT, span: str | None = None) -> DraftFinding:
    """Build a draft finding tersely for table-driven tests."""
    return DraftFinding(finding=label, status=status, span=span)


class TestMissingFindings:
    """Image-present findings that the draft omits or denies are MISSING."""

    def test_image_present_draft_silent_is_missing(self) -> None:
        report = compare([img(EFFUSION, box=EFFUSION_BOX)], [])
        assert report.audit.missing_findings == [EFFUSION]
        assert report.missing == [MissingFinding(finding=EFFUSION, box=EFFUSION_BOX, urgent=False)]

    def test_image_present_draft_denies_is_missing(self) -> None:
        report = compare(
            [img(EFFUSION, box=EFFUSION_BOX)],
            [draft(EFFUSION, status=FindingStatus.ABSENT, span="No pleural effusion")],
        )
        assert report.audit.missing_findings == [EFFUSION]

    def test_missing_carries_box(self) -> None:
        report = compare([img(EFFUSION, box=EFFUSION_BOX)], [])
        assert report.missing[0].box == EFFUSION_BOX

    def test_missing_box_is_none_when_unlocalized(self) -> None:
        report = compare([img(OPACITY, box=None)], [])
        assert report.missing[0].box is None

    def test_image_absent_status_is_not_missing(self) -> None:
        report = compare([img(EFFUSION, status=FindingStatus.ABSENT, box=EFFUSION_BOX)], [])
        assert report.audit.missing_findings == []
        assert report.missing == []


class TestUnsupportedClaims:
    """Draft-asserted findings the image does not support are UNSUPPORTED."""

    def test_draft_present_image_silent_is_unsupported(self) -> None:
        report = compare([], [draft(NODULE, span="suspicious nodule")])
        assert report.audit.unsupported_claims == [NODULE]
        assert report.unsupported == [UnsupportedClaim(finding=NODULE, draft_span="suspicious nodule")]

    def test_draft_present_image_denies_is_unsupported(self) -> None:
        report = compare(
            [img(NODULE, status=FindingStatus.ABSENT)],
            [draft(NODULE, span="nodule seen")],
        )
        assert report.audit.unsupported_claims == [NODULE]

    def test_unsupported_carries_draft_span(self) -> None:
        report = compare([], [draft(NODULE, span="ill-defined nodule in the RUL")])
        assert report.unsupported[0].draft_span == "ill-defined nodule in the RUL"

    def test_draft_absent_status_is_not_unsupported(self) -> None:
        report = compare([], [draft(NODULE, status=FindingStatus.ABSENT, span="no nodule")])
        assert report.audit.unsupported_claims == []
        assert report.unsupported == []


class TestAgreement:
    """A finding present on both image and draft is neither missing nor unsupported."""

    def test_present_both_sides_is_clean(self) -> None:
        report = compare(
            [img(EFFUSION, box=EFFUSION_BOX)],
            [draft(EFFUSION, span="Left pleural effusion")],
        )
        assert report.audit.missing_findings == []
        assert report.audit.unsupported_claims == []

    def test_both_silent_is_clean(self) -> None:
        report = compare([], [])
        assert report.audit == Audit()


class TestUrgent:
    """Image-present whitelist findings raise an urgent flag regardless of draft."""

    def test_image_pneumothorax_is_urgent(self) -> None:
        report = compare([img(PNEUMO, box=PNEUMO_BOX)], [])
        assert report.audit.urgent_review_flags == [PNEUMO]
        assert report.urgent[0].finding == PNEUMO
        assert report.urgent[0].box == PNEUMO_BOX
        assert report.urgent[0].urgent is True

    def test_urgent_flag_raised_even_when_draft_reports_it(self) -> None:
        report = compare(
            [img(PNEUMO, box=PNEUMO_BOX)],
            [draft(PNEUMO, span="small apical pneumothorax")],
        )
        assert report.audit.urgent_review_flags == [PNEUMO]
        assert report.audit.missing_findings == []

    def test_urgent_finding_also_missing_marks_missing_urgent(self) -> None:
        report = compare([img(PNEUMO, box=PNEUMO_BOX)], [])
        assert report.audit.missing_findings == [PNEUMO]
        assert report.missing[0].urgent is True

    def test_image_nodule_mass_is_urgent(self) -> None:
        # A pulmonary nodule/mass is a can't-miss possible malignancy: it must
        # fire urgent on the image side regardless of the draft.
        report = compare([img(NODULE, box=NODULE_BOX)], [])
        assert report.audit.urgent_review_flags == [NODULE]
        assert report.urgent[0].finding == NODULE
        assert report.urgent[0].box == NODULE_BOX
        assert report.urgent[0].urgent is True

    def test_nodule_mass_urgent_even_when_draft_reports_it(self) -> None:
        # Even when the draft faithfully reports the nodule, urgent still fires so
        # the can't-miss finding is always surfaced for radiologist review.
        report = compare(
            [img(NODULE, box=NODULE_BOX)],
            [draft(NODULE, span="ill-defined nodule in the right upper lobe")],
        )
        assert report.audit.urgent_review_flags == [NODULE]
        assert report.audit.missing_findings == []
        assert report.audit.unsupported_claims == []

    def test_both_whitelist_findings_fire_urgent_in_canonical_order(self) -> None:
        report = compare(
            [img(NODULE, box=NODULE_BOX), img(PNEUMO, box=PNEUMO_BOX)],
            [],
        )
        # Canonical order places pneumothorax before nodule_mass.
        assert report.audit.urgent_review_flags == [PNEUMO, NODULE]

    def test_non_whitelist_finding_is_not_urgent(self) -> None:
        report = compare([img(EFFUSION, box=EFFUSION_BOX)], [])
        assert report.audit.urgent_review_flags == []

    def test_image_absent_pneumothorax_is_not_urgent(self) -> None:
        report = compare([img(PNEUMO, status=FindingStatus.ABSENT)], [])
        assert report.audit.urgent_review_flags == []

    def test_image_absent_nodule_mass_is_not_urgent(self) -> None:
        report = compare([img(NODULE, status=FindingStatus.ABSENT)], [])
        assert report.audit.urgent_review_flags == []


class TestNoFindingSentinel:
    """The ``no_finding`` sentinel never produces a positive claim on either side."""

    def test_image_no_finding_produces_nothing(self) -> None:
        report = compare([img(NO_FINDING)], [draft(EFFUSION, span="effusion")])
        assert report.audit.missing_findings == []
        assert report.audit.unsupported_claims == [EFFUSION]

    def test_draft_no_finding_does_not_suppress_missing(self) -> None:
        report = compare(
            [img(EFFUSION, box=EFFUSION_BOX)],
            [draft(NO_FINDING, span="No acute cardiopulmonary abnormality")],
        )
        assert report.audit.missing_findings == [EFFUSION]
        assert report.audit.unsupported_claims == []

    def test_no_finding_both_sides_is_clean(self) -> None:
        report = compare([img(NO_FINDING)], [draft(NO_FINDING, span="normal study")])
        assert report.audit == Audit()


class TestOrderingAndDeduplication:
    """Output lists are canonical-ordered and deduplicated by label."""

    def test_missing_findings_are_canonical_ordered(self) -> None:
        # Supplied out of canonical order; cardiomegaly precedes nothing, effusion
        # is first in canonical order, opacity third.
        report = compare(
            [img(CARDIO), img(EFFUSION, box=EFFUSION_BOX), img(OPACITY)],
            [],
        )
        assert report.audit.missing_findings == [EFFUSION, OPACITY, CARDIO]

    def test_duplicate_image_label_deduplicated_first_box_wins(self) -> None:
        first_box = (0.10, 0.10, 0.20, 0.20)
        second_box = (0.30, 0.30, 0.40, 0.40)
        report = compare(
            [img(EFFUSION, box=first_box), img(EFFUSION, box=second_box)],
            [],
        )
        assert report.audit.missing_findings == [EFFUSION]
        assert report.missing[0].box == first_box

    def test_duplicate_draft_label_deduplicated_first_span_wins(self) -> None:
        report = compare(
            [],
            [draft(NODULE, span="first mention"), draft(NODULE, span="second mention")],
        )
        assert report.audit.unsupported_claims == [NODULE]
        assert report.unsupported[0].draft_span == "first mention"


class TestCombinedScenario:
    """A realistic mixed case exercises all three rules at once."""

    def test_mixed_missing_unsupported_urgent(self) -> None:
        report = compare(
            image_findings=[
                img(PNEUMO, box=PNEUMO_BOX),  # urgent + missing (draft silent)
                img(EFFUSION, box=EFFUSION_BOX),  # agreement
            ],
            draft_findings=[
                draft(EFFUSION, span="Left effusion"),  # agreement
                draft(NODULE, span="nodule RUL"),  # unsupported
            ],
        )
        assert report.audit.missing_findings == [PNEUMO]
        assert report.audit.unsupported_claims == [NODULE]
        assert report.audit.urgent_review_flags == [PNEUMO]


class TestBuildAudit:
    """``build_audit`` returns just the canonical Audit projection."""

    def test_build_audit_matches_compare(self) -> None:
        image = [img(PNEUMO, box=PNEUMO_BOX)]
        draft_list = [draft(NODULE, span="nodule")]
        assert build_audit(image, draft_list) == compare(image, draft_list).audit

    def test_build_audit_returns_audit_type(self) -> None:
        assert isinstance(build_audit([], []), Audit)


def test_comparison_report_defaults_are_empty() -> None:
    """A default-constructed report has empty detail lists and a clean audit."""
    report = ComparisonReport()
    assert report.missing == []
    assert report.unsupported == []
    assert report.urgent == []
    assert report.audit == Audit()


@pytest.mark.parametrize(
    ("image_status", "draft_status", "expect_missing", "expect_unsupported"),
    [
        (FindingStatus.PRESENT, None, True, False),
        (FindingStatus.PRESENT, FindingStatus.PRESENT, False, False),
        (FindingStatus.PRESENT, FindingStatus.ABSENT, True, False),
        (FindingStatus.ABSENT, None, False, False),
        (FindingStatus.ABSENT, FindingStatus.PRESENT, False, True),
        (FindingStatus.ABSENT, FindingStatus.ABSENT, False, False),
        (None, FindingStatus.PRESENT, False, True),
        (None, FindingStatus.ABSENT, False, False),
        (None, None, False, False),
    ],
)
def test_full_status_matrix(
    image_status: FindingStatus | None,
    draft_status: FindingStatus | None,
    expect_missing: bool,
    expect_unsupported: bool,
) -> None:
    """Every image-state x draft-state combination yields the expected verdict.

    ``None`` for a status means the side is silent (the finding is omitted from
    that list entirely).
    """
    label = EFFUSION
    image_findings = [] if image_status is None else [img(label, status=image_status, box=EFFUSION_BOX)]
    draft_findings = [] if draft_status is None else [draft(label, status=draft_status, span="phrase")]

    report = compare(image_findings, draft_findings)

    assert (label in report.audit.missing_findings) is expect_missing
    assert (label in report.audit.unsupported_claims) is expect_unsupported
