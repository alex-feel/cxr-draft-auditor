"""Tests for the canonical output schema, tolerant parsing, and box conversions."""

from __future__ import annotations

import pickle
from typing import Any

import pytest
from pydantic import ValidationError

from cxr_auditor.schema import (
    CANONICAL_BOX_FORMAT,
    DISCLAIMER_TEXT,
    Audit,
    AuditResult,
    DraftFinding,
    FindingStatus,
    ImageFinding,
    SchemaParseError,
    extract_finding_list,
    extract_first_json_object,
    from_qwen_box,
    normalize_finding,
    normalized_to_xyxy_abs,
    parse_model_output,
    salvage_finding_list,
    xyxy_abs_to_normalized,
)


def test_box_format_constant() -> None:
    assert CANONICAL_BOX_FORMAT == "normalized_y0x0y1x1"


def test_audit_result_defaults() -> None:
    result = AuditResult()
    assert result.image_findings == []
    assert result.draft_findings == []
    assert isinstance(result.audit, Audit)
    assert result.disclaimer == DISCLAIMER_TEXT
    assert result.box_format == CANONICAL_BOX_FORMAT


def test_full_audit_result_validates(full_audit_result_dict: dict[str, Any]) -> None:
    result = AuditResult.model_validate(full_audit_result_dict)
    assert result.image_findings[0].finding == "pleural_effusion"
    assert result.draft_findings[0].finding == "no_finding"
    assert result.audit.missing_findings == ["pleural_effusion"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pleural Effusion", "pleural_effusion"),
        ("pleural-effusion", "pleural_effusion"),
        ("  NODULE   MASS  ", "nodule_mass"),
        ("lung_opacity_consolidation", "lung_opacity_consolidation"),
    ],
)
def test_normalize_finding_absorbs_formatting_drift(raw: str, expected: str) -> None:
    assert normalize_finding(raw) == expected


def test_normalize_finding_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="not a canonical finding"):
        normalize_finding("aortic enlargement")


def test_image_finding_normalizes_finding_name() -> None:
    finding = ImageFinding(finding="Pleural Effusion", box=(0.1, 0.2, 0.3, 0.4))
    assert finding.finding == "pleural_effusion"
    assert finding.status is FindingStatus.PRESENT


def test_image_finding_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ImageFinding.model_validate({"finding": "pneumothorax", "unexpected": "x"})


def test_image_finding_rejects_location_field() -> None:
    # ``location`` is not part of the canonical image-finding schema: the grounding
    # prompt never asks for it and the model never emits it, so it must be rejected
    # under extra='forbid' like any other unknown field.
    with pytest.raises(ValidationError):
        ImageFinding.model_validate({"finding": "pneumothorax", "location": "right apex"})


def test_image_finding_box_may_be_none() -> None:
    finding = ImageFinding(finding="no_finding", box=None)
    assert finding.box is None


@pytest.mark.parametrize(
    "bad_box",
    [
        [0.1, 0.2, 0.3],  # too few components
        [0.1, 0.2, 0.3, 1.5],  # out of [0, 1]
        [0.5, 0.2, 0.3, 0.4],  # y1 < y0
        [0.1, 0.6, 0.3, 0.4],  # x1 < x0
    ],
)
def test_image_finding_rejects_bad_box(bad_box: list[float]) -> None:
    with pytest.raises(ValidationError):
        ImageFinding.model_validate({"finding": "pleural_effusion", "box": bad_box})


def test_image_finding_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        ImageFinding(finding="pneumothorax", confidence=1.5)


def test_draft_finding_absent_status() -> None:
    draft = DraftFinding(finding="pneumothorax", status=FindingStatus.ABSENT, span="No pneumothorax")
    assert draft.status is FindingStatus.ABSENT
    assert draft.finding == "pneumothorax"


def test_extract_first_json_object_from_prose() -> None:
    text = 'Here is the answer: {"a": 1, "nested": {"b": 2}} and that is all.'
    assert extract_first_json_object(text) == {"a": 1, "nested": {"b": 2}}


def test_extract_first_json_object_handles_braces_in_strings() -> None:
    text = 'prefix {"note": "a } brace in a string", "ok": true} suffix'
    obj = extract_first_json_object(text)
    assert obj["note"] == "a } brace in a string"
    assert obj["ok"] is True


def test_extract_first_json_object_no_object_raises() -> None:
    with pytest.raises(SchemaParseError):
        extract_first_json_object("there is no json here")


def test_extract_first_json_object_unbalanced_raises() -> None:
    with pytest.raises(SchemaParseError):
        extract_first_json_object('{"a": 1, "b": 2')


def test_extract_first_json_object_non_object_raises() -> None:
    with pytest.raises(SchemaParseError):
        extract_first_json_object("[1, 2, 3]")


def test_extract_first_json_object_handles_escaped_quotes() -> None:
    text = r'noise {"quote": "she said \"hi\" and { left a brace"} tail'
    obj = extract_first_json_object(text)
    assert obj["quote"] == 'she said "hi" and { left a brace'


def test_extract_first_json_object_invalid_slice_raises() -> None:
    # Balanced braces but the slice is not valid JSON (trailing comma).
    with pytest.raises(SchemaParseError, match="invalid"):
        extract_first_json_object('{"a": 1,}')


def test_parse_model_output_with_fences_and_prose() -> None:
    text = (
        "Sure, here you go:\n```json\n"
        '{"image_findings": [{"finding": "Pleural Effusion", "box": [0.1, 0.2, 0.3, 0.4]}]}\n'
        "```\nDone."
    )
    result = parse_model_output(text)
    assert result.image_findings[0].finding == "pleural_effusion"
    assert result.box_format == CANONICAL_BOX_FORMAT


def test_parse_model_output_invalid_schema_raises_with_raw_text() -> None:
    text = '{"image_findings": [{"finding": "definitely not a finding"}]}'
    with pytest.raises(SchemaParseError) as exc_info:
        parse_model_output(text)
    assert exc_info.value.raw_text == text


def test_extract_finding_list_medgemma_native(grounded_finding_dicts: list[dict[str, Any]]) -> None:
    text = 'Findings: [{"label": "pleural_effusion", "box_2d": [0.6, 0.1, 0.9, 0.4]}]'
    parsed = extract_finding_list(text)
    assert parsed[0]["label"] == "pleural_effusion"
    # The static fixture is itself a valid finding list shape.
    assert grounded_finding_dicts[0]["label"] == "pleural_effusion"


def test_extract_finding_list_with_code_fence() -> None:
    text = '```json\n[{"label": "pneumothorax", "box_2d": null}]\n```'
    parsed = extract_finding_list(text)
    assert parsed[0]["label"] == "pneumothorax"
    assert parsed[0]["box_2d"] is None


def test_extract_finding_list_wraps_single_object() -> None:
    text = '{"label": "cardiomegaly", "box_2d": [0.2, 0.2, 0.8, 0.8]}'
    parsed = extract_finding_list(text)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["label"] == "cardiomegaly"


def test_extract_finding_list_array_with_brackets_in_strings() -> None:
    text = '[{"label": "nodule_mass", "evidence": "mass [right upper lobe]"}]'
    parsed = extract_finding_list(text)
    assert parsed[0]["evidence"] == "mass [right upper lobe]"


def test_extract_finding_list_non_object_element_raises() -> None:
    with pytest.raises(SchemaParseError):
        extract_finding_list("[1, 2, 3]")


def test_extract_finding_list_no_json_raises() -> None:
    with pytest.raises(SchemaParseError):
        extract_finding_list("no json at all")


def test_extract_finding_list_handles_escaped_quotes_in_array() -> None:
    text = r'[{"evidence": "opacity with \"halo\" and ] bracket", "label": "nodule_mass"}]'
    parsed = extract_finding_list(text)
    assert parsed[0]["evidence"] == 'opacity with "halo" and ] bracket'
    assert parsed[0]["label"] == "nodule_mass"


def test_extract_finding_list_unbalanced_array_salvages_complete_elements() -> None:
    # A truncated generation leaves the array unclosed; the complete leading
    # element is recovered instead of discarding the whole reply.
    parsed = extract_finding_list('[{"label": "pneumothorax"}')
    assert parsed == [{"label": "pneumothorax"}]


def test_extract_finding_list_unsalvageable_truncation_raises() -> None:
    # A reply truncated before any element completes has nothing to salvage.
    with pytest.raises(SchemaParseError, match="balanced"):
        extract_finding_list('[{"label": ')


def test_extract_finding_list_invalid_array_slice_raises() -> None:
    with pytest.raises(SchemaParseError, match="invalid"):
        extract_finding_list('[{"label": "pneumothorax",}]')


def test_extract_finding_list_salvages_truncated_tail_keeping_leading_elements() -> None:
    text = (
        '[{"label": "pleural_effusion", "box_2d": [0.62, 0.08, 0.94, 0.4]},'
        ' {"label": "cardiomegaly", "box_2d": [0.5, 0.3, 0.9, 0.75]},'
        ' {"label": "nodule_mass", "box_2d": [0.2'
    )
    parsed = extract_finding_list(text)
    assert [element["label"] for element in parsed] == ["pleural_effusion", "cardiomegaly"]


def test_extract_finding_list_salvages_closed_array_with_invalid_tail_element() -> None:
    # The array closes, but a malformed tail element makes the whole slice
    # invalid JSON; the complete leading element is still recovered.
    text = '[{"label": "cardiomegaly"}, {"label": "pneumothorax",}]'
    parsed = extract_finding_list(text)
    assert parsed == [{"label": "cardiomegaly"}]


class TestSchemaParseErrorPickle:
    """``SchemaParseError`` survives the pickle round-trip a process boundary implies."""

    def test_round_trip_preserves_message_and_raw_text(self) -> None:
        error = SchemaParseError("no balanced JSON array found in model text", "the raw model text")
        restored = pickle.loads(pickle.dumps(error))
        assert isinstance(restored, SchemaParseError)
        assert str(restored) == "no balanced JSON array found in model text"
        assert restored.raw_text == "the raw model text"

    def test_round_trip_with_empty_message(self) -> None:
        restored = pickle.loads(pickle.dumps(SchemaParseError("", "tail")))
        assert str(restored) == ""
        assert restored.raw_text == "tail"

    def test_round_trip_of_subclass_relationship(self) -> None:
        # The error must remain catchable as ValueError after the round-trip.
        restored = pickle.loads(pickle.dumps(SchemaParseError("message", "raw")))
        assert isinstance(restored, ValueError)


class TestSalvageFindingList:
    """Truncated or degenerate arrays yield their complete leading elements."""

    def test_truncation_mid_element_recovers_leading(self) -> None:
        text = '[{"label": "pleural_effusion", "box_2d": null}, {"label": "pneumo'
        assert salvage_finding_list(text) == [{"label": "pleural_effusion", "box_2d": None}]

    def test_truncation_mid_box_array_recovers_leading(self) -> None:
        text = '[{"label": "cardiomegaly", "box_2d": [0.5, 0.3, 0.9, 0.75]}, {"label": "nodule_mass", "box_2d": [0.55, 0.'
        assert salvage_finding_list(text) == [{"label": "cardiomegaly", "box_2d": [0.5, 0.3, 0.9, 0.75]}]

    def test_truncation_between_elements_recovers_all_complete(self) -> None:
        text = '[{"label": "cardiomegaly"}, {"label": "nodule_mass"},'
        assert salvage_finding_list(text) == [{"label": "cardiomegaly"}, {"label": "nodule_mass"}]

    def test_garbage_prefix_before_array_is_skipped(self) -> None:
        text = 'Sure, here are the findings: [{"label": "pneumothorax"}, {"label": "card'
        assert salvage_finding_list(text) == [{"label": "pneumothorax"}]

    def test_unterminated_code_fence_still_salvages(self) -> None:
        # A truncated reply can open a fence and never close it; the fence regex
        # does not match, and the array is still found behind the backticks.
        text = '```json\n[{"label": "pneumothorax"}, {"label": "card'
        assert salvage_finding_list(text) == [{"label": "pneumothorax"}]

    def test_no_array_yields_empty(self) -> None:
        assert salvage_finding_list("no json at all") == []

    def test_truncation_inside_first_element_yields_empty(self) -> None:
        assert salvage_finding_list('[{"label": ') == []

    def test_invalid_element_stops_salvage(self) -> None:
        text = '[{"label": "cardiomegaly"}, {"label": "pneumothorax",}, {"label": "nodule_mass"}'
        assert salvage_finding_list(text) == [{"label": "cardiomegaly"}]

    def test_content_after_closed_array_is_not_salvaged(self) -> None:
        text = '[{"label": "cardiomegaly"}] trailing prose {"label": "pneumothorax"}'
        assert salvage_finding_list(text) == [{"label": "cardiomegaly"}]

    def test_repetition_degenerate_output_caps_at_64_elements(self) -> None:
        element = '{"label": "nodule_mass", "box_2d": [0.2, 0.3, 0.4, 0.5]}'
        text = "[" + ", ".join([element] * 100) + ', {"label": "trunca'
        salvaged = salvage_finding_list(text)
        assert len(salvaged) == 64
        assert all(item["label"] == "nodule_mass" for item in salvaged)


def test_normalized_to_xyxy_abs() -> None:
    assert normalized_to_xyxy_abs((0.1, 0.2, 0.3, 0.4), 1000, 2000) == (200.0, 200.0, 400.0, 600.0)


def test_xyxy_abs_to_normalized_roundtrip() -> None:
    box = (0.1, 0.2, 0.3, 0.4)
    abs_box = normalized_to_xyxy_abs(box, 1000, 2000)
    assert xyxy_abs_to_normalized(abs_box, 1000, 2000) == pytest.approx(box)


def test_xyxy_abs_to_normalized_clamps_overflow() -> None:
    # A box slightly past the image edge clamps to 1.0 rather than overflowing.
    result = xyxy_abs_to_normalized((-5.0, -5.0, 1005.0, 2005.0), 1000, 2000)
    assert result == (0.0, 0.0, 1.0, 1.0)


def test_from_qwen_box_matches_xyxy_conversion() -> None:
    qwen = (200.0, 200.0, 400.0, 600.0)
    assert from_qwen_box(qwen, 1000, 2000) == xyxy_abs_to_normalized(qwen, 1000, 2000)


@pytest.mark.parametrize("dimensions", [(0, 100), (100, 0), (-1, 100)])
def test_box_conversion_rejects_nonpositive_dimensions(dimensions: tuple[int, int]) -> None:
    width, height = dimensions
    with pytest.raises(ValueError, match="must be positive"):
        normalized_to_xyxy_abs((0.1, 0.1, 0.2, 0.2), width, height)
    with pytest.raises(ValueError, match="must be positive"):
        xyxy_abs_to_normalized((1.0, 1.0, 2.0, 2.0), width, height)
