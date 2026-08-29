"""Tests for the ChestX-Det mirror parser (integer-index label segmentation)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cxr_auditor.data.chestxdet import (
    CHESTXDET_BACKGROUND_ID,
    CHESTXDET_ID_TO_LABEL,
    iter_present_findings,
    label_array_to_boxes,
    parse_chestxdet_example,
    parse_chestxdet_examples,
)


def test_id_to_label_taxonomy_matches_published_categories() -> None:
    # The 13-category ChestX-Det taxonomy, contiguous ids 1..13.
    assert sorted(CHESTXDET_ID_TO_LABEL) == list(range(1, 14))
    assert CHESTXDET_ID_TO_LABEL[6] == "Effusion"
    assert CHESTXDET_ID_TO_LABEL[3] == "Cardiomegaly"
    assert CHESTXDET_BACKGROUND_ID == 255


def test_label_array_to_boxes_tight_bbox_and_mapping() -> None:
    # A 4x4 id map: id 6 (Effusion) occupies the 2x2 block at rows 1-2, cols 1-2.
    label = np.full((4, 4), CHESTXDET_BACKGROUND_ID, dtype=np.int64)
    label[1:3, 1:3] = 6
    boxes = label_array_to_boxes(label)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.finding == "pleural_effusion"
    assert box.native_label == "Effusion"
    # Pixels span cols 1-2 and rows 1-2; the inclusive max pixel pushes the
    # bottom-right edge to (3, 3). Normalized [y0, x0, y1, x1] over 4x4.
    assert box.box is not None
    assert box.box == pytest.approx((1 / 4, 1 / 4, 3 / 4, 3 / 4))


def test_label_array_drops_background_and_out_of_set_ids() -> None:
    # id 9 is Fracture (out of canonical set); id 255 is background. Both dropped.
    label = np.full((3, 3), CHESTXDET_BACKGROUND_ID, dtype=np.int64)
    label[0, 0] = 9
    boxes = label_array_to_boxes(label)
    assert boxes == []


def test_label_array_emits_in_ascending_id_order() -> None:
    # id 10 (Mass) and id 6 (Effusion) present; emitted in ascending id order so
    # Effusion (6) precedes Mass (10).
    label = np.full((4, 4), CHESTXDET_BACKGROUND_ID, dtype=np.int64)
    label[0, 0] = 10
    label[3, 3] = 6
    findings = [box.finding for box in label_array_to_boxes(label)]
    assert findings == ["pleural_effusion", "nodule_mass"]


def test_parse_example_rejects_non_2d_label() -> None:
    with pytest.raises(ValueError, match="2-D"):
        parse_chestxdet_example({"image_id": "bad.png", "label": np.zeros((2, 2, 2, 2), dtype=np.int64)})


def test_parse_example_maps_present_categories(chestxdet_rows: list[dict[str, Any]]) -> None:
    record = parse_chestxdet_example(chestxdet_rows[0])
    summary = {box.finding for box in record.boxes}
    # id 6 -> pleural_effusion, id 4 -> lung_opacity_consolidation; id 9 (Fracture)
    # is out of set and dropped.
    assert summary == {"pleural_effusion", "lung_opacity_consolidation"}
    assert record.original_width == 6
    assert record.original_height == 6
    effusion = next(box for box in record.boxes if box.finding == "pleural_effusion")
    # id 6 occupies rows 1-2, cols 1-2 of the 6x6 grid.
    assert effusion.box is not None
    assert effusion.box == pytest.approx((1 / 6, 1 / 6, 3 / 6, 3 / 6))


def test_parse_example_three_channel_label_uses_first_channel() -> None:
    # Some mirror variants store the label as a 3-channel image with identical
    # channels; the first channel is the id map.
    base = np.full((3, 3), CHESTXDET_BACKGROUND_ID, dtype=np.int64)
    base[0, 0] = 3
    stacked = np.stack([base, base, base], axis=-1)
    record = parse_chestxdet_example({"image_id": "rgb.png", "label": stacked})
    assert {box.finding for box in record.boxes} == {"cardiomegaly"}


def test_parse_example_without_label_yields_empty_record() -> None:
    record = parse_chestxdet_example({"image_id": "no_label.png"})
    assert record.image_id == "no_label.png"
    assert record.boxes == []
    assert record.original_width is None
    assert record.original_height is None


def test_parse_example_without_image_id_uses_empty_string() -> None:
    label = np.full((2, 2), CHESTXDET_BACKGROUND_ID, dtype=np.int64)
    label[0, 0] = 6
    record = parse_chestxdet_example({"label": label})
    assert record.image_id == ""
    assert {box.finding for box in record.boxes} == {"pleural_effusion"}


def test_parse_examples_present_findings(chestxdet_rows: list[dict[str, Any]]) -> None:
    records = parse_chestxdet_examples(chestxdet_rows)
    summary = iter_present_findings(records)
    assert summary["cxd_0001.png"] == {"pleural_effusion", "lung_opacity_consolidation"}
    assert summary["cxd_0002.png"] == {"nodule_mass"}
