"""Tests for the VinDr-CXR-VQA join and gt_location rescaling."""

from __future__ import annotations

from typing import Any

import pytest

from cxr_auditor.data.vqa_join import (
    IMAGE_ID_HEX_LENGTH,
    collect_canonical_findings,
    is_valid_image_id,
    join_vqa_to_images,
    load_vqa_annotations,
    parse_vqa_row,
    parse_vqa_rows,
    rescale_gt_location,
)


def test_image_id_validation() -> None:
    assert is_valid_image_id("0a1b2c3d4e5f60718293a4b5c6d7e8f9")
    assert IMAGE_ID_HEX_LENGTH == 32
    assert not is_valid_image_id("too short")
    assert not is_valid_image_id("z" * 32)  # not hex


def test_rescale_gt_location_normalizes_by_original_dims() -> None:
    # gt_location is [x_min, y_min, x_max, y_max] in original full-res pixels.
    box = rescale_gt_location((108, 1810, 240, 2120), 2500, 3000)
    assert box == pytest.approx((1810 / 3000, 108 / 2500, 2120 / 3000, 240 / 2500))


def test_parse_row_rejects_wrong_length_gt_location() -> None:
    # A malformed gt_location arriving from untrusted JSON (typed Any) must be
    # rejected by the length guard rather than silently mis-rescaled.
    row: dict[str, Any] = {
        "image_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
        "gt_finding": "Pleural effusion",
        "gt_location": [1, 2, 3],
        "image_width": 100,
        "image_height": 100,
    }
    with pytest.raises(ValueError, match="exactly 4 components"):
        parse_vqa_row(row)


def test_parse_row_maps_finding_and_rescales_box(vqa_sample_rows: list[dict[str, Any]]) -> None:
    record = parse_vqa_row(vqa_sample_rows[0])
    assert record.image_id == "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
    assert record.finding == "pleural_effusion"
    assert record.box is not None
    # gt_location [108, 1810, 240, 2120] over image_width 2500, image_height 3000.
    assert record.box == pytest.approx((1810 / 3000, 108 / 2500, 2120 / 3000, 240 / 2500))


def test_parse_row_pneumothorax(vqa_sample_rows: list[dict[str, Any]]) -> None:
    record = parse_vqa_row(vqa_sample_rows[1])
    assert record.finding == "pneumothorax"
    assert record.box is not None
    assert record.box == pytest.approx((200 / 2400, 300 / 2000, 1400 / 2400, 900 / 2000))


def test_parse_row_no_finding_has_none_box(vqa_sample_rows: list[dict[str, Any]]) -> None:
    record = parse_vqa_row(vqa_sample_rows[2])
    assert record.finding == "no_finding"
    assert record.box is None
    assert record.native_finding == "No finding"


def test_parse_row_out_of_set_finding_yields_none() -> None:
    row = {
        "image_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
        "gt_finding": "Aortic enlargement",
        "gt_location": [500, 300, 1100, 800],
        "image_width": 2500,
        "image_height": 3000,
    }
    record = parse_vqa_row(row)
    assert record.finding is None
    # Box still rescales (geometry is valid) even when the label is out of set.
    assert record.box is not None


def test_parse_row_missing_dims_skips_box() -> None:
    row = {
        "image_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
        "gt_finding": "Pleural effusion",
        "gt_location": [108, 1810, 240, 2120],
    }
    record = parse_vqa_row(row)
    assert record.finding == "pleural_effusion"
    assert record.box is None


def test_join_groups_by_image_id(vqa_sample_rows: list[dict[str, Any]]) -> None:
    records = parse_vqa_rows(vqa_sample_rows)
    grouped = join_vqa_to_images(records)
    assert set(grouped) == {
        "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
        "1f2e3d4c5b6a70819a2b3c4d5e6f7081",
        "2a3b4c5d6e7f80910a1b2c3d4e5f6071",
    }


def test_join_restricts_to_available_images(vqa_sample_rows: list[dict[str, Any]]) -> None:
    records = parse_vqa_rows(vqa_sample_rows)
    grouped = join_vqa_to_images(records, available_image_ids={"0a1b2c3d4e5f60718293a4b5c6d7e8f9"})
    assert set(grouped) == {"0a1b2c3d4e5f60718293a4b5c6d7e8f9"}


def test_collect_canonical_findings_ignores_out_of_set(vqa_sample_rows: list[dict[str, Any]]) -> None:
    records = parse_vqa_rows(vqa_sample_rows)
    summary = collect_canonical_findings(records)
    assert summary["0a1b2c3d4e5f60718293a4b5c6d7e8f9"] == {"pleural_effusion"}
    assert summary["2a3b4c5d6e7f80910a1b2c3d4e5f6071"] == {"no_finding"}


def test_load_vqa_annotations_reads_file(tmp_path: Any, vqa_sample_rows: list[dict[str, Any]]) -> None:
    import json

    path = tmp_path / "data_v1.json"
    path.write_text(json.dumps(vqa_sample_rows), encoding="utf-8")
    records = load_vqa_annotations(path)
    assert [record.image_id for record in records] == [row["image_id"] for row in vqa_sample_rows]


def test_load_vqa_annotations_rejects_non_list(tmp_path: Any) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON array"):
        load_vqa_annotations(path)
