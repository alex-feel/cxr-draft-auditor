"""Tests for SFT corpus construction, JSONL writing, and record validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cxr_auditor.findings import NO_FINDING
from cxr_auditor.prompts import build_image_grounding_prompt
from cxr_auditor.schema import FindingStatus, ImageFinding, extract_finding_list
from cxr_auditor.sft_dataset import (
    ASSISTANT_ROLE,
    IMAGE_PART_TYPE,
    TEXT_PART_TYPE,
    USER_ROLE,
    build_assistant_target_text,
    build_sft_corpus,
    build_sft_record,
    build_target_finding_dicts,
    describe_corpus_target_format,
    read_sft_jsonl,
    validate_sft_record,
    write_sft_jsonl,
)


def _image_finding(label: str, *, status: FindingStatus = FindingStatus.PRESENT, box=None, **kwargs) -> ImageFinding:
    return ImageFinding(finding=label, status=status, box=box, **kwargs)


# --- assistant target construction --------------------------------------------------


def test_build_target_finding_dicts_keeps_present_with_box() -> None:
    findings = [_image_finding("pleural_effusion", box=(0.62, 0.08, 0.94, 0.40), confidence=0.78, evidence="blunting")]
    target = build_target_finding_dicts(findings)
    assert target == [
        {
            "label": "pleural_effusion",
            "box_2d": [0.62, 0.08, 0.94, 0.40],
            "confidence": 0.78,
            "evidence": "blunting",
        }
    ]


def test_build_target_finding_dicts_emits_null_box_when_absent() -> None:
    findings = [_image_finding("cardiomegaly", box=None)]
    target = build_target_finding_dicts(findings)
    assert target == [{"label": "cardiomegaly", "box_2d": None}]


def test_build_target_finding_dicts_drops_absent_status_findings() -> None:
    findings = [
        _image_finding("pleural_effusion", box=(0.1, 0.1, 0.2, 0.2)),
        _image_finding("pneumothorax", status=FindingStatus.ABSENT),
    ]
    labels = [element["label"] for element in build_target_finding_dicts(findings)]
    assert labels == ["pleural_effusion"]


def test_build_target_finding_dicts_no_findings_yields_sentinel() -> None:
    target = build_target_finding_dicts([])
    assert target == [{"label": NO_FINDING, "box_2d": None}]


def test_build_assistant_target_text_is_parseable_finding_list() -> None:
    findings = [_image_finding("pleural_effusion", box=(0.1, 0.2, 0.3, 0.4))]
    text = build_assistant_target_text(findings)
    parsed = extract_finding_list(text)
    assert parsed[0]["label"] == "pleural_effusion"
    assert parsed[0]["box_2d"] == [0.1, 0.2, 0.3, 0.4]


# --- record construction ------------------------------------------------------------


def test_build_sft_record_shape(tiny_png_path: Path) -> None:
    findings = [_image_finding("cardiomegaly", box=(0.3, 0.2, 0.8, 0.8))]
    record = build_sft_record(tiny_png_path, findings)

    assert record["image_path"] == str(tiny_png_path)
    messages = record["messages"]
    assert [m["role"] for m in messages] == [USER_ROLE, ASSISTANT_ROLE]

    user_parts = messages[0]["content"]
    assert any(part["type"] == IMAGE_PART_TYPE for part in user_parts)
    text_part = next(part for part in user_parts if part["type"] == TEXT_PART_TYPE)
    assert text_part["text"] == build_image_grounding_prompt()

    assistant_parts = messages[1]["content"]
    assistant_text = next(part["text"] for part in assistant_parts if part["type"] == TEXT_PART_TYPE)
    parsed = extract_finding_list(assistant_text)
    assert parsed[0]["label"] == "cardiomegaly"


def test_build_sft_record_passes_its_own_validation(tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("nodule_mass", box=(0.4, 0.4, 0.6, 0.6))])
    # Must not raise.
    validate_sft_record(record)


# --- validation rejects malformed records -------------------------------------------


def test_validate_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        validate_sft_record(["not", "a", "dict"])


def test_validate_rejects_missing_image_path() -> None:
    with pytest.raises(ValueError, match="image_path"):
        validate_sft_record({"messages": []})


def test_validate_rejects_wrong_message_count(tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("cardiomegaly")])
    record["messages"] = record["messages"][:1]
    with pytest.raises(ValueError, match="exactly two turns"):
        validate_sft_record(record)


def test_validate_rejects_user_turn_without_image(tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("cardiomegaly")])
    user_parts = record["messages"][0]["content"]
    record["messages"][0]["content"] = [part for part in user_parts if part["type"] != IMAGE_PART_TYPE]
    with pytest.raises(ValueError, match="image part"):
        validate_sft_record(record)


def test_validate_rejects_noncanonical_assistant_label(tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("cardiomegaly")])
    record["messages"][1]["content"][0]["text"] = json.dumps([{"label": "fracture", "box_2d": None}])
    with pytest.raises(ValueError, match="canonical"):
        validate_sft_record(record)


def test_validate_rejects_swapped_roles(tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("cardiomegaly")])
    record["messages"][0]["role"] = ASSISTANT_ROLE
    with pytest.raises(ValueError, match="role 'user'"):
        validate_sft_record(record)


# --- JSONL round trip ---------------------------------------------------------------


def test_write_and_read_sft_jsonl_round_trip(tmp_path: Path, tiny_png_path: Path) -> None:
    corpus = build_sft_corpus(
        [
            (tiny_png_path, [_image_finding("pleural_effusion", box=(0.6, 0.1, 0.9, 0.4))]),
            (tiny_png_path, [_image_finding(NO_FINDING)]),
        ]
    )
    out = tmp_path / "sft" / "corpus.jsonl"
    written = write_sft_jsonl(corpus, out)
    assert written == 2
    assert out.exists()

    # Each physical line is exactly one JSON record (JSONL contract).
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)

    records = read_sft_jsonl(out)
    assert len(records) == 2
    assert records[0]["image_path"] == str(tiny_png_path)


def test_write_sft_jsonl_validates_by_default(tmp_path: Path, tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("cardiomegaly")])
    record["image_path"] = ""  # break it
    out = tmp_path / "broken.jsonl"
    with pytest.raises(ValueError, match="record 0 failed SFT validation"):
        write_sft_jsonl([record], out)
    # No partial file is written when validation fails before the write.
    assert not out.exists()


def test_read_sft_jsonl_skips_blank_lines(tmp_path: Path, tiny_png_path: Path) -> None:
    record = build_sft_record(tiny_png_path, [_image_finding("cardiomegaly")])
    out = tmp_path / "padded.jsonl"
    out.write_text(json.dumps(record) + "\n\n", encoding="utf-8")
    records = read_sft_jsonl(out)
    assert len(records) == 1


def test_read_sft_jsonl_reports_bad_json_line(tmp_path: Path) -> None:
    out = tmp_path / "bad.jsonl"
    out.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not valid JSON"):
        read_sft_jsonl(out)


# --- descriptor ---------------------------------------------------------------------


def test_describe_corpus_target_format() -> None:
    descriptor = describe_corpus_target_format()
    assert descriptor["box_format"] == "normalized_y0x0y1x1"
    assert descriptor["user_role"] == USER_ROLE
    assert descriptor["assistant_role"] == ASSISTANT_ROLE
