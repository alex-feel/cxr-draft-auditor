"""Tests for the NIH ChestX-ray14 BBox_List_2017 loader."""

from __future__ import annotations

import pytest

from cxr_auditor.data.nih_bbox import (
    NIH_REFERENCE_SIZE,
    iter_present_findings,
    parse_nih_bbox_csv,
    xywh_to_normalized,
)


def test_reference_size_constant() -> None:
    assert NIH_REFERENCE_SIZE == 1024


def test_xywh_to_normalized_converts_corner_plus_extent() -> None:
    # XYWH (x, y, w, h) -> xyxy (x, y, x+w, y+h) -> normalized [y0, x0, y1, x1].
    box = xywh_to_normalized(256, 512, 512, 256, 1024, 1024)
    assert box == pytest.approx((512 / 1024, 256 / 1024, 768 / 1024, 768 / 1024))


def test_parse_groups_and_maps_labels(nih_bbox_csv_text: str) -> None:
    records = parse_nih_bbox_csv(nih_bbox_csv_text)
    summary = iter_present_findings(records)
    # Image 1: Cardiomegaly + Effusion(->pleural_effusion).
    assert summary["00000001_000.png"] == {"cardiomegaly", "pleural_effusion"}
    # Image 3: Mass(->nodule_mass) + Pneumothorax.
    assert summary["00000003_000.png"] == {"nodule_mass", "pneumothorax"}


def test_parse_drops_out_of_set_label(nih_bbox_csv_text: str) -> None:
    records = parse_nih_bbox_csv(nih_bbox_csv_text)
    # Image 2 has only Atelectasis (out of set) -> the image group survives with
    # zero canonical boxes.
    image_two = next(record for record in records if record.image_id == "00000002_000.png")
    assert image_two.boxes == []


def test_parse_rescales_box_against_reference_size(nih_bbox_csv_text: str) -> None:
    records = parse_nih_bbox_csv(nih_bbox_csv_text)
    image_one = next(record for record in records if record.image_id == "00000001_000.png")
    assert image_one.original_width == 1024
    assert image_one.original_height == 1024
    cardio = next(box for box in image_one.boxes if box.finding == "cardiomegaly")
    # XYWH (256, 512, 512, 256) -> xyxy (256, 512, 768, 768) over 1024x1024.
    assert cardio.box is not None
    assert cardio.box == pytest.approx((512 / 1024, 256 / 1024, 768 / 1024, 768 / 1024))


def test_parse_accepts_plain_named_columns() -> None:
    text = "image_id,finding,x,y,w,h\nimg1,Effusion,10,20,30,40\n"
    records = parse_nih_bbox_csv(text, reference_width=100, reference_height=100)
    box = records[0].boxes[0]
    assert box.finding == "pleural_effusion"
    assert box.box is not None
    # XYWH (10, 20, 30, 40) -> xyxy (10, 20, 40, 60) -> normalized [y0, x0, y1, x1].
    assert box.box == pytest.approx((20 / 100, 10 / 100, 60 / 100, 40 / 100))


def test_parse_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        parse_nih_bbox_csv("image_id,finding\nimg1,Effusion\n")


def test_fixture_matches_real_export_header_shape(nih_bbox_csv_text: str) -> None:
    # The real BBox_List_2017.csv header is exactly
    # "Image Index,Finding Label,Bbox [x,y,w,h],,," (the bracketed column name
    # splits into Bbox [x / y / w / h], plus THREE trailing empty columns), and
    # rows carry float pixel coordinates with three trailing empties. The fixture
    # reproduces that shape so the parser is exercised against the genuine export.
    header = nih_bbox_csv_text.splitlines()[0]
    assert header == "Image Index,Finding Label,Bbox [x,y,w,h],,,"
    first_data_row = nih_bbox_csv_text.splitlines()[1]
    assert first_data_row.endswith(",,,")
    assert "256.0" in first_data_row


def test_parse_handles_real_export_with_float_coords_and_trailing_columns(nih_bbox_csv_text: str) -> None:
    # End-to-end over the real-shaped fixture: five rows, four canonical labels
    # mapped, Atelectasis dropped, boxes normalized against the 1024px reference.
    records = parse_nih_bbox_csv(nih_bbox_csv_text)
    summary = iter_present_findings(records)
    assert summary["00000001_000.png"] == {"cardiomegaly", "pleural_effusion"}
    assert summary["00000003_000.png"] == {"nodule_mass", "pneumothorax"}
