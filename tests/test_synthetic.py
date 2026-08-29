"""Tests for synthetic draft generation and labeled audit-case construction."""

from __future__ import annotations

import pytest

from cxr_auditor.findings import NO_FINDING, POSITIVE_FINDINGS
from cxr_auditor.schema import FindingStatus, ImageFinding
from cxr_auditor.synthetic import (
    AuditCase,
    AuditCaseType,
    absent_positive_labels,
    compute_expected_audit,
    generate_draft,
    make_audit_case_records,
    make_audit_cases,
    normalize_present_labels,
    present_positive_labels,
)


def _image_finding(label: str, *, status: FindingStatus = FindingStatus.PRESENT, box=None) -> ImageFinding:
    return ImageFinding(finding=label, status=status, box=box)


def _case_by_type(cases: list[AuditCase], case_type: AuditCaseType) -> AuditCase:
    matches = [case for case in cases if case.case_type is case_type]
    assert len(matches) == 1, f"expected exactly one {case_type} case, got {len(matches)}"
    return matches[0]


# --- present / absent label helpers -------------------------------------------------


def test_present_positive_labels_excludes_no_finding_and_absent() -> None:
    findings = [
        _image_finding("pleural_effusion"),
        _image_finding(NO_FINDING),
        _image_finding("pneumothorax", status=FindingStatus.ABSENT),
        _image_finding("cardiomegaly"),
    ]
    assert present_positive_labels(findings) == ["pleural_effusion", "cardiomegaly"]


def test_present_positive_labels_orders_by_canonical_order_and_dedupes() -> None:
    findings = [
        _image_finding("cardiomegaly"),
        _image_finding("pleural_effusion"),
        _image_finding("pleural_effusion", box=(0.1, 0.1, 0.2, 0.2)),
    ]
    result = present_positive_labels(findings)
    assert result == ["pleural_effusion", "cardiomegaly"]
    assert result == [label for label in POSITIVE_FINDINGS if label in set(result)]


def test_present_positive_labels_accepts_dicts() -> None:
    findings = [{"finding": "Pleural Effusion"}, {"finding": "cardiomegaly"}]
    assert present_positive_labels(findings) == ["pleural_effusion", "cardiomegaly"]


def test_absent_positive_labels_is_complement() -> None:
    present = ["pleural_effusion", "cardiomegaly"]
    absent = absent_positive_labels(present)
    assert set(absent) == set(POSITIVE_FINDINGS) - set(present)
    assert NO_FINDING not in absent


# --- draft generation ---------------------------------------------------------------


def test_generate_draft_is_deterministic_for_same_seed() -> None:
    findings = [_image_finding("pleural_effusion"), _image_finding("cardiomegaly")]
    first = generate_draft(findings, seed=7)
    second = generate_draft(findings, seed=7)
    assert first == second


def test_generate_draft_varies_with_seed() -> None:
    findings = [_image_finding("pleural_effusion"), _image_finding("cardiomegaly")]
    drafts = {generate_draft(findings, seed=seed) for seed in range(20)}
    # The template tables offer several phrasings per finding, so distinct seeds
    # must be able to produce more than one distinct draft.
    assert len(drafts) > 1


def test_generate_draft_empty_findings_is_negative_study() -> None:
    draft = generate_draft([], seed=1)
    assert draft
    # A negative-study line never asserts a positive finding sentence.
    assert "effusion" not in draft.lower()


def test_generate_draft_invokes_paraphraser_when_supplied() -> None:
    findings = [_image_finding("pleural_effusion")]
    calls: list[str] = []

    def paraphraser(text: str) -> str:
        calls.append(text)
        return "PARAPHRASED"

    draft = generate_draft(findings, seed=1, paraphraser=paraphraser)
    assert draft == "PARAPHRASED"
    assert len(calls) == 1


# --- compute_expected_audit ---------------------------------------------------------


def test_compute_expected_audit_flags_missing() -> None:
    from cxr_auditor.schema import DraftFinding

    audit = compute_expected_audit(
        image_present_labels=["pleural_effusion", "cardiomegaly"],
        draft_findings=[DraftFinding(finding="cardiomegaly", status=FindingStatus.PRESENT)],
    )
    assert audit.missing_findings == ["pleural_effusion"]
    assert audit.unsupported_claims == []


def test_compute_expected_audit_flags_unsupported() -> None:
    from cxr_auditor.schema import DraftFinding

    audit = compute_expected_audit(
        image_present_labels=["cardiomegaly"],
        draft_findings=[
            DraftFinding(finding="cardiomegaly", status=FindingStatus.PRESENT),
            DraftFinding(finding="nodule_mass", status=FindingStatus.PRESENT),
        ],
    )
    assert audit.unsupported_claims == ["nodule_mass"]
    assert audit.missing_findings == []


def test_compute_expected_audit_no_finding_draft_is_not_unsupported() -> None:
    from cxr_auditor.schema import DraftFinding

    audit = compute_expected_audit(
        image_present_labels=[],
        draft_findings=[DraftFinding(finding=NO_FINDING, status=FindingStatus.PRESENT)],
    )
    assert audit.unsupported_claims == []
    assert audit.missing_findings == []


def test_compute_expected_audit_absent_draft_finding_counts_as_missing() -> None:
    from cxr_auditor.schema import DraftFinding

    audit = compute_expected_audit(
        image_present_labels=["pneumothorax"],
        draft_findings=[DraftFinding(finding="pneumothorax", status=FindingStatus.ABSENT)],
    )
    assert audit.missing_findings == ["pneumothorax"]


def test_compute_expected_audit_flags_urgent() -> None:
    from cxr_auditor.schema import DraftFinding

    audit = compute_expected_audit(
        image_present_labels=["pneumothorax"],
        draft_findings=[DraftFinding(finding="pneumothorax", status=FindingStatus.PRESENT)],
    )
    assert audit.urgent_review_flags == ["pneumothorax"]
    # A faithfully reported urgent finding is not missing or unsupported.
    assert audit.missing_findings == []
    assert audit.unsupported_claims == []


def test_compute_expected_audit_orders_lists_canonically() -> None:
    from cxr_auditor.schema import DraftFinding

    audit = compute_expected_audit(
        image_present_labels=["cardiomegaly", "pleural_effusion", "nodule_mass"],
        draft_findings=[DraftFinding(finding="lung_opacity_consolidation", status=FindingStatus.PRESENT)],
    )
    expected_missing = [label for label in POSITIVE_FINDINGS if label in {"pleural_effusion", "nodule_mass", "cardiomegaly"}]
    assert audit.missing_findings == expected_missing


# --- make_audit_cases: the three labeled cases --------------------------------------


def test_make_audit_cases_yields_three_cases_for_mixed_image() -> None:
    findings = [_image_finding("pleural_effusion"), _image_finding("cardiomegaly")]
    cases = make_audit_cases(findings, seed=3)
    assert {case.case_type for case in cases} == {
        AuditCaseType.MISSING,
        AuditCaseType.UNSUPPORTED,
        AuditCaseType.FAITHFUL,
    }


def test_missing_case_expects_dropped_finding_in_missing_list() -> None:
    findings = [_image_finding("pleural_effusion"), _image_finding("cardiomegaly")]
    cases = make_audit_cases(findings, seed=3)
    missing_case = _case_by_type(cases, AuditCaseType.MISSING)

    dropped = missing_case.corrupted_label
    assert dropped is not None
    assert dropped in missing_case.image_present_labels
    # The dropped finding must NOT appear among the draft's present findings.
    draft_present = {f.finding for f in missing_case.draft_findings if f.status is FindingStatus.PRESENT}
    assert dropped not in draft_present
    # And it must be the (only) expected missing finding.
    assert missing_case.expected_audit.missing_findings == [dropped]
    assert missing_case.expected_audit.unsupported_claims == []


def test_unsupported_case_expects_added_finding_in_unsupported_list() -> None:
    findings = [_image_finding("cardiomegaly")]
    cases = make_audit_cases(findings, seed=11)
    unsupported_case = _case_by_type(cases, AuditCaseType.UNSUPPORTED)

    added = unsupported_case.corrupted_label
    assert added is not None
    assert added not in unsupported_case.image_present_labels
    draft_present = {f.finding for f in unsupported_case.draft_findings if f.status is FindingStatus.PRESENT}
    assert added in draft_present
    assert unsupported_case.expected_audit.unsupported_claims == [added]
    assert unsupported_case.expected_audit.missing_findings == []


def test_faithful_case_has_empty_audit_when_no_urgent_finding() -> None:
    findings = [_image_finding("cardiomegaly"), _image_finding("pleural_effusion")]
    cases = make_audit_cases(findings, seed=5)
    faithful_case = _case_by_type(cases, AuditCaseType.FAITHFUL)

    assert faithful_case.corrupted_label is None
    assert faithful_case.expected_audit.missing_findings == []
    assert faithful_case.expected_audit.unsupported_claims == []
    assert faithful_case.expected_audit.urgent_review_flags == []


def test_faithful_case_still_flags_urgent_finding() -> None:
    findings = [_image_finding("pneumothorax")]
    cases = make_audit_cases(findings, seed=5)
    faithful_case = _case_by_type(cases, AuditCaseType.FAITHFUL)
    assert faithful_case.expected_audit.urgent_review_flags == ["pneumothorax"]
    assert faithful_case.expected_audit.missing_findings == []
    assert faithful_case.expected_audit.unsupported_claims == []


def test_no_present_findings_omits_missing_case_but_keeps_faithful_and_unsupported() -> None:
    findings = [_image_finding(NO_FINDING)]
    cases = make_audit_cases(findings, seed=9)
    case_types = {case.case_type for case in cases}
    assert AuditCaseType.MISSING not in case_types
    assert AuditCaseType.UNSUPPORTED in case_types
    assert AuditCaseType.FAITHFUL in case_types

    faithful_case = _case_by_type(cases, AuditCaseType.FAITHFUL)
    assert faithful_case.expected_audit.missing_findings == []
    assert faithful_case.expected_audit.unsupported_claims == []


def test_all_positives_present_omits_unsupported_case() -> None:
    findings = [_image_finding(label) for label in POSITIVE_FINDINGS]
    cases = make_audit_cases(findings, seed=2)
    case_types = {case.case_type for case in cases}
    assert AuditCaseType.UNSUPPORTED not in case_types
    assert AuditCaseType.MISSING in case_types
    assert AuditCaseType.FAITHFUL in case_types


def test_make_audit_cases_is_deterministic() -> None:
    findings = [_image_finding("pleural_effusion"), _image_finding("nodule_mass")]
    first = make_audit_cases(findings, seed=42)
    second = make_audit_cases(findings, seed=42)
    assert [c.draft_text for c in first] == [c.draft_text for c in second]
    assert [c.expected_audit.model_dump() for c in first] == [c.expected_audit.model_dump() for c in second]


# --- make_audit_case_records --------------------------------------------------------


def test_make_audit_case_records_are_json_serializable_and_labeled() -> None:
    import json

    findings = [_image_finding("pleural_effusion"), _image_finding("cardiomegaly")]
    records = make_audit_case_records(findings, seed=4, image_id="abc123")
    assert records
    for record in records:
        # Must round-trip through JSON (deterministic JSONL eval set).
        json.dumps(record)
        assert record["image_id"] == "abc123"
        assert record["case_type"] in {t.value for t in AuditCaseType}
        assert "expected_audit" in record
        assert "draft_findings" in record


def test_make_audit_case_records_omits_image_id_when_not_supplied() -> None:
    findings = [_image_finding("cardiomegaly")]
    records = make_audit_case_records(findings, seed=4)
    for record in records:
        assert "image_id" not in record


# --- normalize_present_labels -------------------------------------------------------


def test_normalize_present_labels_canonicalizes_and_drops_sentinel() -> None:
    labels = ["Pleural Effusion", "pleural-effusion", "No Finding", "Cardiomegaly"]
    assert normalize_present_labels(labels) == ["pleural_effusion", "cardiomegaly"]


def test_normalize_present_labels_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="canonical"):
        normalize_present_labels(["fracture"])
