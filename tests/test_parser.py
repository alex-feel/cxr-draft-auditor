"""
Tests for the draft-report parser.

The parser is exercised with a fake ``generate_fn`` that returns canned model
text, so no model, network, or GPU is involved. Tests cover tolerant JSON
extraction, status coercion, out-of-vocabulary dropping, deduplication, the
``no_finding`` sentinel, and the off-by-default CheXbert cross-check raising when
the optional dependency is absent.
"""

from __future__ import annotations

import pytest

from cxr_auditor.parser import (
    GenerateFn,
    chexbert_cross_check,
    parse_draft,
    parse_draft_findings,
)
from cxr_auditor.schema import DraftFinding, FindingStatus, SchemaParseError


def fixed_generate_fn(output: str) -> GenerateFn:
    """Build a fake ``generate_fn`` that ignores its prompt and returns ``output``."""

    def _generate(_prompt: str) -> str:
        return output

    return _generate


class TestParseDraftFindings:
    """Raw model text is parsed into validated ``DraftFinding`` objects."""

    def test_parses_present_and_absent(self) -> None:
        raw = (
            '[{"label": "pleural_effusion", "status": "present", "span": "Left pleural effusion"},'
            ' {"label": "pneumothorax", "status": "absent", "span": "No pneumothorax"}]'
        )
        findings = parse_draft_findings(raw)
        assert findings == [
            DraftFinding(finding="pleural_effusion", status=FindingStatus.PRESENT, span="Left pleural effusion"),
            DraftFinding(finding="pneumothorax", status=FindingStatus.ABSENT, span="No pneumothorax"),
        ]

    def test_strips_markdown_fences(self) -> None:
        raw = '```json\n[{"label": "cardiomegaly", "status": "present"}]\n```'
        findings = parse_draft_findings(raw)
        assert findings == [DraftFinding(finding="cardiomegaly", status=FindingStatus.PRESENT, span=None)]

    def test_missing_status_defaults_present(self) -> None:
        findings = parse_draft_findings('[{"label": "nodule_mass", "span": "a nodule"}]')
        assert findings[0].status is FindingStatus.PRESENT

    def test_null_status_defaults_present(self) -> None:
        findings = parse_draft_findings('[{"label": "nodule_mass", "status": null}]')
        assert findings[0].status is FindingStatus.PRESENT

    def test_unknown_status_defaults_present(self) -> None:
        findings = parse_draft_findings('[{"label": "cardiomegaly", "status": "likely"}]')
        assert findings[0].status is FindingStatus.PRESENT

    def test_label_casing_and_spaces_normalized(self) -> None:
        findings = parse_draft_findings('[{"label": "Pleural Effusion", "status": "present"}]')
        assert findings[0].finding == "pleural_effusion"

    def test_out_of_vocabulary_label_dropped_keeps_rest(self) -> None:
        raw = '[{"label": "atelectasis", "status": "present"}, {"label": "pneumothorax", "status": "present"}]'
        findings = parse_draft_findings(raw)
        assert [f.finding for f in findings] == ["pneumothorax"]

    def test_non_string_label_skipped(self) -> None:
        raw = '[{"label": 5, "status": "present"}, {"label": "cardiomegaly"}]'
        findings = parse_draft_findings(raw)
        assert [f.finding for f in findings] == ["cardiomegaly"]

    def test_non_string_span_becomes_none(self) -> None:
        findings = parse_draft_findings('[{"label": "cardiomegaly", "span": 7}]')
        assert findings[0].span is None

    def test_duplicate_label_status_deduplicated(self) -> None:
        raw = (
            '[{"label": "cardiomegaly", "status": "present", "span": "first"},'
            ' {"label": "cardiomegaly", "status": "present", "span": "second"}]'
        )
        findings = parse_draft_findings(raw)
        assert len(findings) == 1
        assert findings[0].span == "first"

    def test_same_label_different_status_kept(self) -> None:
        raw = '[{"label": "pneumothorax", "status": "present"}, {"label": "pneumothorax", "status": "absent"}]'
        findings = parse_draft_findings(raw)
        assert {(f.finding, f.status) for f in findings} == {
            ("pneumothorax", FindingStatus.PRESENT),
            ("pneumothorax", FindingStatus.ABSENT),
        }

    def test_bare_object_is_wrapped(self) -> None:
        findings = parse_draft_findings('{"label": "no_finding", "status": "present"}')
        assert findings == [DraftFinding(finding="no_finding", status=FindingStatus.PRESENT, span=None)]

    def test_no_finding_sentinel_parsed(self) -> None:
        findings = parse_draft_findings(
            '[{"label": "no_finding", "status": "present", "span": "No acute cardiopulmonary abnormality"}]'
        )
        assert findings[0].finding == "no_finding"

    def test_prose_wrapped_array_extracted(self) -> None:
        raw = 'Here is the result: [{"label": "cardiomegaly", "status": "present"}] Done.'
        findings = parse_draft_findings(raw)
        assert [f.finding for f in findings] == ["cardiomegaly"]

    def test_unparseable_text_raises_with_raw(self) -> None:
        with pytest.raises(SchemaParseError) as excinfo:
            parse_draft_findings("the model said nothing parseable")
        assert excinfo.value.raw_text == "the model said nothing parseable"

    def test_truncated_array_salvaged_into_findings(self) -> None:
        # A reply that exhausts the token budget mid-array keeps its complete
        # leading elements (via the schema-level salvage) instead of raising.
        raw = '[{"label": "pleural_effusion", "status": "present", "span": "effusion"}, {"label": "pneumo'
        findings = parse_draft_findings(raw)
        assert findings == [DraftFinding(finding="pleural_effusion", status=FindingStatus.PRESENT, span="effusion")]


class TestParseDraft:
    """``parse_draft`` builds the prompt, calls generate_fn, and validates."""

    def test_end_to_end_with_fake_generate_fn(self) -> None:
        raw = '[{"label": "pleural_effusion", "status": "present", "span": "effusion"}]'
        findings = parse_draft("Left pleural effusion.", fixed_generate_fn(raw))
        assert findings == [DraftFinding(finding="pleural_effusion", status=FindingStatus.PRESENT, span="effusion")]

    def test_prompt_passed_to_generate_fn(self) -> None:
        captured: dict[str, str] = {}

        def _capture(prompt: str) -> str:
            captured["prompt"] = prompt
            return '[{"label": "no_finding", "status": "present"}]'

        parse_draft("No acute cardiopulmonary abnormality.", _capture)
        # The rendered prompt embeds the draft text and the canonical vocabulary.
        assert "No acute cardiopulmonary abnormality." in captured["prompt"]
        assert "pleural_effusion" in captured["prompt"]

    def test_empty_draft_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            parse_draft("   ", fixed_generate_fn("[]"))

    def test_empty_array_yields_no_findings(self) -> None:
        assert parse_draft("Some text.", fixed_generate_fn("[]")) == []


class TestChexbertCrossCheck:
    """The off-by-default CheXbert cross-check stays opt-in and lazy."""

    def test_raises_runtime_error_without_optional_dependency(self) -> None:
        # f1chexbert is intentionally NOT a project dependency; the cross-check
        # must report that clearly rather than importing it at module load.
        import importlib.util

        f1chexbert_installed = importlib.util.find_spec("f1chexbert") is not None
        if f1chexbert_installed:
            pytest.skip("f1chexbert is installed in this environment; the absent-dependency path cannot be exercised")

        with pytest.raises(RuntimeError, match="f1chexbert"):
            chexbert_cross_check("Left pleural effusion. No pneumothorax.")
