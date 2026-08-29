"""Tests for the IU-Xray / Open-i report-row shaping (no boxes).

The fixture mirrors the real ``ykumards/open-i`` schema: ``uid`` (int), ``MeSH``,
``Problems``, ``image`` (a study-description string, NOT an image), ``indication``,
``comparison``, ``findings``, ``impression``, plus the binary ``img_frontal`` /
``img_lateral`` X-ray columns.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from cxr_auditor.data.iu_xray import extract_image, parse_iu_xray_row, parse_iu_xray_rows


def _jpeg_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """Encode a tiny solid-color RGB JPEG and return its raw bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_parse_row_extracts_findings_and_impression(iu_xray_rows: list[dict[str, Any]]) -> None:
    record = parse_iu_xray_row(iu_xray_rows[0])
    # uid is an int in the real schema; image_id is its string form.
    assert record.image_id == "1"
    assert "lungs are clear" in record.findings_text.lower()
    assert record.impression_text == "No acute cardiopulmonary abnormality."


def test_parse_row_missing_findings_yields_empty_string(iu_xray_rows: list[dict[str, Any]]) -> None:
    record = parse_iu_xray_row(iu_xray_rows[2])
    assert record.findings_text == ""
    assert record.impression_text == "Stable cardiomegaly. No pneumothorax."


def test_parse_rows_returns_record_per_row(iu_xray_rows: list[dict[str, Any]]) -> None:
    records = parse_iu_xray_rows(iu_xray_rows)
    assert [record.image_id for record in records] == ["1", "2", "3"]


def test_extract_image_none_when_no_image_columns(iu_xray_rows: list[dict[str, Any]]) -> None:
    # The fixture rows carry only the text 'image' study description, never an
    # img_frontal/img_lateral payload, so no image is extracted.
    assert extract_image(iu_xray_rows[0]) is None


def test_extract_image_ignores_text_image_column() -> None:
    # The 'image' column is a study-description string; it must never be returned
    # as if it were an image.
    assert extract_image({"image": "Xray Chest PA and Lateral"}) is None


def test_extract_image_decodes_frontal_bytes() -> None:
    image = extract_image({"img_frontal": _jpeg_bytes()})
    assert image is not None
    assert image.size == (8, 8)


def test_extract_image_prefers_frontal_over_lateral() -> None:
    frontal = _jpeg_bytes((100, 0, 0))
    lateral = _jpeg_bytes((0, 0, 100))
    image = extract_image({"img_frontal": frontal, "img_lateral": lateral})
    assert image is not None
    # The frontal payload is chosen; its average red channel dominates.
    rgb = image.convert("RGB").getpixel((0, 0))
    assert isinstance(rgb, tuple)
    assert rgb[0] > rgb[2]


def test_extract_image_falls_back_to_lateral_when_frontal_absent() -> None:
    image = extract_image({"img_lateral": _jpeg_bytes()})
    assert image is not None
    assert image.size == (8, 8)


def test_extract_image_accepts_already_decoded_image() -> None:
    decoded = Image.new("RGB", (4, 4))
    assert extract_image({"img_frontal": decoded}) is decoded


def test_records_carry_no_boxes() -> None:
    # The ReportRecord shape has no box field at all; this guards the contract
    # that IU-Xray is a no-boxes source.
    record = parse_iu_xray_row({"uid": 7, "findings": "f", "impression": "i"})
    assert not hasattr(record, "box")
    assert not hasattr(record, "boxes")
