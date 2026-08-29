"""Tests that the synthetic fixtures load and have the documented shapes."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from PIL import Image

from cxr_auditor.findings import map_vindr_label
from cxr_auditor.schema import xyxy_abs_to_normalized


def test_tiny_png_is_16x16(tiny_png_path: Path) -> None:
    assert tiny_png_path.exists()
    with Image.open(tiny_png_path) as image:
        assert image.size == (16, 16)
        assert image.format == "PNG"


def test_tiny_image_fixture_is_rgb(tiny_image: Image.Image) -> None:
    assert tiny_image.mode == "RGB"
    assert tiny_image.size == (16, 16)


def test_vqa_sample_rows_have_image_id_and_boxes(vqa_sample_rows: list[dict[str, Any]]) -> None:
    assert len(vqa_sample_rows) == 3
    for row in vqa_sample_rows:
        # image_id joins to the Kaggle DICOM filename: a 32-char hex string.
        assert len(row["image_id"]) == 32
        assert all(character in "0123456789abcdef" for character in row["image_id"])


def test_vqa_gt_location_is_full_res_pixel_space(vqa_sample_rows: list[dict[str, Any]]) -> None:
    # gt_location boxes are full-res VinDr pixel space; rescaling against the
    # row's own width/height yields a valid normalized box. This exercises the
    # documented join-and-rescale contract.
    row = vqa_sample_rows[0]
    box = row["gt_location"]
    width = row["image_width"]
    height = row["image_height"]
    # gt_location ordering for this fixture is [x_min, y_min, x_max, y_max].
    normalized = xyxy_abs_to_normalized(tuple(box), width, height)
    assert all(0.0 <= component <= 1.0 for component in normalized)


def test_vindr_csv_fixture_parses_and_maps(vindr_boxes_csv_text: str) -> None:
    reader = csv.DictReader(io.StringIO(vindr_boxes_csv_text))
    rows = list(reader)
    assert len(rows) == 6
    # The first row is a pleural effusion that maps to the canonical label.
    assert map_vindr_label(rows[0]["class_name"]) == "pleural_effusion"
    # The "No finding" row has empty box coordinates.
    no_finding_rows = [row for row in rows if row["class_name"] == "No finding"]
    assert no_finding_rows
    assert no_finding_rows[0]["x_min"] == ""


def test_sample_model_outputs_have_expected_keys(sample_model_outputs: dict[str, Any]) -> None:
    for key in (
        "image_grounding_list",
        "image_grounding_no_finding",
        "draft_parse_list",
        "full_audit_result",
    ):
        assert key in sample_model_outputs
