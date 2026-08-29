"""Tests for the VinDr box loader and the original-versus-resized rescaling."""

from __future__ import annotations

import pytest

from cxr_auditor.data.records import ImageBoxRecord
from cxr_auditor.data.vindr import (
    VINDR_CSV_COLUMNS,
    iter_present_findings,
    load_vindr_dims_csv,
    parse_vindr_boxes_csv,
    project_boxes_to_mirror,
    rescale_box_to_normalized,
    vindr_image_path,
)
from cxr_auditor.schema import normalized_to_xyxy_abs


def _record_by_id(records: list[ImageBoxRecord], image_id: str) -> ImageBoxRecord:
    return next(record for record in records if record.image_id == image_id)


def test_rescale_box_normalizes_by_original_dims() -> None:
    # A box authored in 2500x3000 (w x h) original pixel space.
    box = rescale_box_to_normalized((250.0, 600.0, 1250.0, 2400.0), 2500, 3000)
    # y normalized by height 3000, x normalized by width 2500: [y0, x0, y1, x1].
    assert box == pytest.approx((0.2, 0.1, 0.8, 0.5))


def test_rescale_then_project_to_resized_mirror_roundtrips_proportionally() -> None:
    # The whole point of normalization: a box authored at 2500x3000 projects
    # proportionally onto a 512x512 mirror without knowing the original at draw time.
    normalized = rescale_box_to_normalized((250.0, 600.0, 1250.0, 2400.0), 2500, 3000)
    xyxy_on_mirror = normalized_to_xyxy_abs(normalized, 512, 512)
    assert xyxy_on_mirror == pytest.approx((0.1 * 512, 0.2 * 512, 0.5 * 512, 0.8 * 512))


def test_parse_groups_boxes_per_image_and_maps_labels(vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text, dims)

    summary = iter_present_findings(records)
    # First image: pleural effusion + cardiomegaly (both canonical).
    assert summary["0a1b2c3d4e5f60718293a4b5c6d7e8f9"] == {"pleural_effusion", "cardiomegaly"}
    # Second image: pneumothorax.
    assert summary["1f2e3d4c5b6a70819a2b3c4d5e6f7081"] == {"pneumothorax"}
    # Third image: a No finding row -> no_finding sentinel.
    assert summary["2a3b4c5d6e7f80910a1b2c3d4e5f6071"] == {"no_finding"}


def test_parse_drops_out_of_set_classes_but_keeps_image_group(vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text, dims)
    # The fourth image has Aortic enlargement (out-of-set -> dropped) and Lung
    # Opacity (canonical). The image group survives with only the canonical box.
    fourth = _record_by_id(records, "3b4c5d6e7f8091a2b3c4d5e6f7081923")
    assert [box.finding for box in fourth.boxes] == ["lung_opacity_consolidation"]


def test_no_finding_row_has_none_box(vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text, dims)
    third = _record_by_id(records, "2a3b4c5d6e7f80910a1b2c3d4e5f6071")
    assert len(third.boxes) == 1
    assert third.boxes[0].finding == "no_finding"
    assert third.boxes[0].box is None


def test_parse_rescales_box_using_provided_original_dims(vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text, dims)
    first = _record_by_id(records, "0a1b2c3d4e5f60718293a4b5c6d7e8f9")
    assert first.original_width == 2500
    assert first.original_height == 3000
    effusion = next(box for box in first.boxes if box.finding == "pleural_effusion")
    # Original box (108, 1810, 240, 2120) over 2500x3000 (w x h).
    assert effusion.box is not None
    assert effusion.box == pytest.approx((1810 / 3000, 108 / 2500, 2120 / 3000, 240 / 2500))


def test_parse_without_dims_keeps_label_but_drops_box(vindr_boxes_csv_text: str) -> None:
    # No dims supplied: positive findings are kept (label not lost) but box is None
    # because rescaling without the original dimensions would be wrong.
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text)
    first = _record_by_id(records, "0a1b2c3d4e5f60718293a4b5c6d7e8f9")
    assert first.original_width is None
    effusion = next(box for box in first.boxes if box.finding == "pleural_effusion")
    assert effusion.box is None


def test_load_dims_csv_dim0_dim1_convention(vindr_meta_csv_text: str) -> None:
    # dim0 = rows = height, dim1 = columns = width. The map stores (width, height).
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    assert dims["0a1b2c3d4e5f60718293a4b5c6d7e8f9"] == (2500, 3000)


def test_load_dims_csv_width_height_columns() -> None:
    text = "image_id,width,height\nabc,800,600\n"
    dims = load_vindr_dims_csv(text)
    assert dims["abc"] == (800, 600)


def test_load_dims_csv_rejects_missing_id_column() -> None:
    with pytest.raises(ValueError, match="image_id/id"):
        load_vindr_dims_csv("width,height\n800,600\n")


def test_load_dims_csv_rejects_missing_dimension_columns() -> None:
    with pytest.raises(ValueError, match="width/height"):
        load_vindr_dims_csv("image_id,foo\nabc,1\n")


def test_parse_rejects_wrong_header() -> None:
    with pytest.raises(ValueError, match="unexpected VinDr CSV header"):
        parse_vindr_boxes_csv("a,b,c\n1,2,3\n")


def test_csv_columns_constant_matches_contract() -> None:
    assert VINDR_CSV_COLUMNS == (
        "image_id",
        "class_name",
        "class_id",
        "rad_id",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    )


def test_project_boxes_to_mirror_inverts_normalization(vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text, dims)
    second = _record_by_id(records, "1f2e3d4c5b6a70819a2b3c4d5e6f7081")
    projected = project_boxes_to_mirror(second, 512, 512)
    finding, xyxy = projected[0]
    assert finding == "pneumothorax"
    # Original box (300, 200, 900, 1400) over 2000x2400 (w x h), drawn on 512x512.
    assert xyxy is not None
    assert xyxy == pytest.approx((300 / 2000 * 512, 200 / 2400 * 512, 900 / 2000 * 512, 1400 / 2400 * 512))


def test_project_boxes_passes_through_none_box(vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = parse_vindr_boxes_csv(vindr_boxes_csv_text, dims)
    third = _record_by_id(records, "2a3b4c5d6e7f80910a1b2c3d4e5f6071")
    projected = project_boxes_to_mirror(third, 512, 512)
    assert projected == [("no_finding", None)]


def test_vindr_image_path_uses_image_id_stem() -> None:
    path = vindr_image_path("/data/vindr512", "abc123", ".png")
    assert path.name == "abc123.png"
    assert path.parent.name == "vindr512"
