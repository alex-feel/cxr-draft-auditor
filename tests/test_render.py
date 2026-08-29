"""
Tests for the pure-logic presentation helpers in ``cxr_auditor.render``.

These helpers turn an ``AuditOutcome`` into the artifacts the Gradio app shows:
an annotated overlay image (PIL), a findings table (list of rows), an audit panel
(Markdown), and a raw JSON string. They depend only on PIL and the pure-logic
schema/comparator types - no gradio, torch, or GPU - so they are unit-testable
directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from cxr_auditor.comparator import compare
from cxr_auditor.inference import DRAFT_PARSE_FAILURE_NOTE, AuditOutcome
from cxr_auditor.render import (
    EvidenceCategory,
    EvidenceItem,
    OverlayBox,
    _draw_label,
    _iou,
    _rects_overlap,
    _resolve_band_rect,
    _scaled_font,
    annotate_evidence,
    audit_panel_markdown,
    categorize_image_findings,
    cluster_overlay_boxes,
    findings_table_rows,
    result_json,
)
from cxr_auditor.schema import AuditResult, DraftFinding, FindingStatus, ImageFinding


def _outcome(image_findings: list[ImageFinding], draft_findings: list[DraftFinding]) -> AuditOutcome:
    comparison = compare(image_findings, draft_findings)
    result = AuditResult(image_findings=image_findings, draft_findings=draft_findings, audit=comparison.audit)
    return AuditOutcome(result=result, comparison=comparison)


class TestCategorizeImageFindings:
    def test_supported_when_present_in_both(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))],
            [DraftFinding(finding="cardiomegaly")],
        )
        items = categorize_image_findings(outcome)
        assert len(items) == 1
        assert items[0].category is EvidenceCategory.SUPPORTED
        assert items[0].finding == "cardiomegaly"

    def test_missing_when_image_only(self) -> None:
        outcome = _outcome([ImageFinding(finding="pleural_effusion", box=(0.6, 0.1, 0.9, 0.4))], [])
        items = categorize_image_findings(outcome)
        assert items[0].category is EvidenceCategory.MISSING

    def test_urgent_takes_priority_over_missing(self) -> None:
        outcome = _outcome([ImageFinding(finding="pneumothorax", box=(0.1, 0.5, 0.4, 0.9))], [])
        items = categorize_image_findings(outcome)
        assert items[0].category is EvidenceCategory.URGENT
        assert items[0].urgent is True

    def test_urgent_even_when_supported_by_draft(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="pneumothorax", box=(0.1, 0.5, 0.4, 0.9))],
            [DraftFinding(finding="pneumothorax")],
        )
        items = categorize_image_findings(outcome)
        assert items[0].category is EvidenceCategory.URGENT

    def test_no_finding_image_produces_no_overlay_items(self) -> None:
        outcome = _outcome([ImageFinding(finding="no_finding")], [])
        assert categorize_image_findings(outcome) == []

    def test_absent_image_finding_excluded(self) -> None:
        outcome = _outcome([ImageFinding(finding="cardiomegaly", status=FindingStatus.ABSENT)], [])
        assert categorize_image_findings(outcome) == []


class TestClusterOverlayBoxes:
    def test_identical_boxes_different_labels_merge_into_one(self) -> None:
        box = (0.236, 0.316, 0.386, 0.422)
        items = [
            EvidenceItem(finding="lung_opacity_consolidation", box=box, category=EvidenceCategory.SUPPORTED, urgent=False),
            EvidenceItem(finding="nodule_mass", box=box, category=EvidenceCategory.MISSING, urgent=False),
        ]
        clusters = cluster_overlay_boxes(items)
        assert len(clusters) == 1
        assert isinstance(clusters[0], OverlayBox)
        assert set(clusters[0].labels) == {"lung_opacity_consolidation", "nodule_mass"}

    def test_distinct_regions_stay_separate(self) -> None:
        items = [
            EvidenceItem(
                finding="lung_opacity_consolidation",
                box=(0.236, 0.316, 0.386, 0.422),
                category=EvidenceCategory.MISSING,
                urgent=False,
            ),
            EvidenceItem(
                finding="lung_opacity_consolidation",
                box=(0.236, 0.531, 0.386, 0.622),
                category=EvidenceCategory.MISSING,
                urgent=False,
            ),
        ]
        clusters = cluster_overlay_boxes(items)
        assert len(clusters) == 2

    def test_merged_category_uses_severity_precedence(self) -> None:
        # SUPPORTED + MISSING at the same coordinates resolves to MISSING (the more
        # severe status) so the drawn color reflects status, not paint order.
        box = (0.1, 0.1, 0.5, 0.5)
        items = [
            EvidenceItem(finding="cardiomegaly", box=box, category=EvidenceCategory.SUPPORTED, urgent=False),
            EvidenceItem(finding="nodule_mass", box=box, category=EvidenceCategory.MISSING, urgent=False),
        ]
        clusters = cluster_overlay_boxes(items)
        assert clusters[0].category is EvidenceCategory.MISSING

    def test_urgent_member_makes_cluster_urgent(self) -> None:
        box = (0.1, 0.1, 0.5, 0.5)
        items = [
            EvidenceItem(finding="lung_opacity_consolidation", box=box, category=EvidenceCategory.MISSING, urgent=False),
            EvidenceItem(finding="nodule_mass", box=box, category=EvidenceCategory.URGENT, urgent=True),
        ]
        clusters = cluster_overlay_boxes(items)
        assert clusters[0].urgent is True
        assert clusters[0].category is EvidenceCategory.URGENT

    def test_duplicate_label_and_lower_severity_member_fold_in(self) -> None:
        # Folding a member that repeats an existing label must not duplicate the
        # label, and a less-severe member must not downgrade the cluster category.
        box = (0.1, 0.1, 0.5, 0.5)
        items = [
            EvidenceItem(finding="nodule_mass", box=box, category=EvidenceCategory.MISSING, urgent=False),
            EvidenceItem(finding="nodule_mass", box=box, category=EvidenceCategory.SUPPORTED, urgent=False),
        ]
        clusters = cluster_overlay_boxes(items)
        assert clusters[0].labels == ("nodule_mass",)
        assert clusters[0].category is EvidenceCategory.MISSING

    def test_boxless_items_are_skipped(self) -> None:
        items = [EvidenceItem(finding="no_finding", box=None, category=EvidenceCategory.SUPPORTED, urgent=False)]
        assert cluster_overlay_boxes(items) == []

    def test_iou_identical_boxes_is_one(self) -> None:
        box = (0.2, 0.2, 0.8, 0.8)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_iou_disjoint_boxes_is_zero(self) -> None:
        assert _iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0

    def test_iou_zero_area_boxes_is_zero(self) -> None:
        # Two degenerate (zero-area) boxes at the same point have no union area, so
        # IoU is defined as 0.0 rather than dividing by zero.
        point = (0.3, 0.3, 0.3, 0.3)
        assert _iou(point, point) == 0.0

    def test_cluster_order_is_stable_by_first_member(self) -> None:
        items = [
            EvidenceItem(finding="cardiomegaly", box=(0.5, 0.3, 0.7, 0.7), category=EvidenceCategory.SUPPORTED, urgent=False),
            EvidenceItem(
                finding="pleural_effusion", box=(0.0, 0.0, 0.2, 0.2), category=EvidenceCategory.MISSING, urgent=False
            ),
        ]
        clusters = cluster_overlay_boxes(items)
        assert clusters[0].labels == ("cardiomegaly",)
        assert clusters[1].labels == ("pleural_effusion",)


class TestScaledFont:
    def test_small_canvas_uses_readable_floor(self) -> None:
        # A tiny bitmap default font is unreadable, so small canvases get a floor.
        font = _scaled_font(200, 200)
        assert isinstance(font, ImageFont.FreeTypeFont)
        assert font.size == 14

    def test_capped_canvas_tracks_long_side(self) -> None:
        # round(1280 / 48) == 27: the size every canvas-capped large upload gets.
        font = _scaled_font(1280, 1024)
        assert isinstance(font, ImageFont.FreeTypeFont)
        assert font.size == 27

    def test_oversized_dimensions_clamp_to_cap(self) -> None:
        # annotate_evidence always passes a capped canvas; the clamp keeps the
        # function safe for arbitrary dimensions regardless of caller.
        font = _scaled_font(4096, 4096)
        assert isinstance(font, ImageFont.FreeTypeFont)
        assert font.size == 28


class TestLabelCollisionAvoidance:
    def test_rects_overlap_detects_shared_interior(self) -> None:
        assert _rects_overlap((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) is True

    def test_rects_overlap_edge_touching_is_not_overlap(self) -> None:
        # Flush horizontal and vertical neighbors share no interior area, so a
        # band nudged to sit exactly against another band needs no further nudge.
        assert _rects_overlap((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)) is False
        assert _rects_overlap((0.0, 0.0, 10.0, 10.0), (0.0, 10.0, 10.0, 20.0)) is False

    def test_lone_label_keeps_desired_rect(self) -> None:
        # With nothing placed yet the desired rectangle is returned unchanged, so
        # a lone label renders exactly where the anchored placement put it.
        desired = (40.0, 130.0, 240.0, 150.0)
        box = (40.0, 150.0, 200.0, 300.0)
        assert _resolve_band_rect(desired, box, 512, []) == desired

    def test_colliding_band_moves_below_its_own_box(self) -> None:
        # A placed band occupies the desired slot; the first fallback is the band
        # slot starting at the owning box's bottom edge, which is clear here.
        desired = (120.0, 130.0, 320.0, 150.0)
        box = (120.0, 150.0, 280.0, 300.0)
        placed = [(40.0, 130.0, 240.0, 150.0)]
        resolved = _resolve_band_rect(desired, box, 512, placed)
        assert resolved == (120.0, 300.0, 320.0, 320.0)
        assert not _rects_overlap(resolved, placed[0])

    def test_exhausted_budget_returns_clamped_on_image_rect(self) -> None:
        # A placed band covering the whole canvas leaves no clear slot; the
        # resolver must still return an on-image rectangle rather than fail.
        desired = (10.0, 30.0, 110.0, 50.0)
        box = (10.0, 50.0, 100.0, 90.0)
        placed = [(0.0, 0.0, 200.0, 200.0)]
        resolved = _resolve_band_rect(desired, box, 200, placed)
        assert resolved[1] >= 0.0
        assert resolved[3] <= 200.0

    def test_two_near_coincident_labels_produce_non_overlapping_bands(self) -> None:
        # Two boxes at the same height whose long labels would land on the same
        # row: the two band rectangles actually drawn must not overlap.
        canvas = Image.new("RGB", (512, 512), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        font = _scaled_font(512, 512)
        placed: list[tuple[float, float, float, float]] = []
        text = "Lung opacity / consolidation"
        for box in ((92.0, 153.0, 200.0, 317.0), (205.0, 153.0, 348.0, 317.0)):
            placed.append(
                _draw_label(draw, box, text, color=(243, 156, 18), font=font, image_size=(512, 512), placed_bands=placed)
            )
        assert len(placed) == 2
        assert not _rects_overlap(placed[0], placed[1])


class TestAnnotateEvidence:
    def test_returns_rgb_pil_image_same_size_below_cap(self, tiny_image: Image.Image) -> None:
        outcome = _outcome([ImageFinding(finding="pleural_effusion", box=(0.6, 0.1, 0.9, 0.4))], [])
        annotated = annotate_evidence(tiny_image, outcome)
        assert isinstance(annotated, Image.Image)
        assert annotated.mode == "RGB"
        assert annotated.size == tiny_image.size

    def test_large_input_downscaled_to_canvas_cap(self) -> None:
        # The overlay is an evidence visualization, not a diagnostic image: inputs
        # larger than 1280 px on the long side render on a downscaled canvas so
        # label and border proportions stay predictable for any upload size.
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.2, 0.2, 0.8, 0.8))], [])
        large = Image.new("L", (2560, 2048), color=60)
        annotated = annotate_evidence(large, outcome)
        assert annotated.size == (1280, 1024)
        assert annotated.mode == "RGB"

    def test_input_exactly_at_cap_keeps_size(self) -> None:
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.2, 0.2, 0.8, 0.8))], [])
        at_cap = Image.new("L", (1280, 640), color=60)
        annotated = annotate_evidence(at_cap, outcome)
        assert annotated.size == (1280, 640)

    def test_does_not_mutate_input(self, tiny_image: Image.Image) -> None:
        before = tiny_image.copy()
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))], [])
        annotate_evidence(tiny_image, outcome)
        assert tiny_image.tobytes() == before.tobytes()

    def test_overlay_changes_pixels_when_box_present(self, tiny_image: Image.Image) -> None:
        outcome = _outcome([ImageFinding(finding="pneumothorax", box=(0.0, 0.0, 1.0, 1.0))], [])
        annotated = annotate_evidence(tiny_image, outcome)
        assert annotated.tobytes() != tiny_image.convert("RGB").tobytes()

    def test_no_boxes_returns_unannotated_copy(self, tiny_image: Image.Image) -> None:
        outcome = _outcome([ImageFinding(finding="no_finding")], [])
        annotated = annotate_evidence(tiny_image, outcome)
        assert annotated.size == tiny_image.size
        assert annotated.mode == "RGB"

    def test_grayscale_input_is_converted(self) -> None:
        gray = Image.new("L", (32, 32), color=128)
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.2, 0.2, 0.8, 0.8))], [])
        annotated = annotate_evidence(gray, outcome)
        assert annotated.mode == "RGB"

    def test_duplicate_coincident_boxes_drawn_once(self) -> None:
        # Two findings the model emits at identical coordinates must render as ONE
        # box, not two stacked rectangles. A supported box drawn green then a missing
        # box drawn amber at the same pixels would, without merging, leave only the
        # later color; merging picks the more-severe (missing/amber) color, so the
        # box border carries the missing color, never the supported one.
        box = (0.2, 0.2, 0.8, 0.8)
        outcome = _outcome(
            [
                ImageFinding(finding="cardiomegaly", box=box),
                ImageFinding(finding="nodule_mass", box=box),
            ],
            [DraftFinding(finding="cardiomegaly")],
        )
        large = Image.new("RGB", (256, 256), color=(10, 10, 10))
        annotated = annotate_evidence(large, outcome)
        # The amber MISSING/urgent color must appear; the supported green must not
        # paint the merged region. nodule_mass is urgent so the cluster is red.
        colors = {color for _count, color in annotated.getcolors(maxcolors=100000) or []}
        assert (46, 204, 113) not in colors  # supported green must not win the merge

    def test_label_text_is_within_image_bounds_at_top_edge(self) -> None:
        # A box touching the top edge must still place its label on the image (below
        # the box top), never clipped above y=0.
        outcome = _outcome([ImageFinding(finding="pleural_effusion", box=(0.0, 0.1, 0.3, 0.5))], [])
        canvas = Image.new("RGB", (200, 200), color=(0, 0, 0))
        annotated = annotate_evidence(canvas, outcome)
        # The top row of pixels must contain non-background content (the label band or
        # box border), proving the label was not drawn off-image.
        top_row = [annotated.getpixel((x, 0)) for x in range(annotated.width)]
        assert any(pixel != (0, 0, 0) for pixel in top_row)

    def test_supported_box_drawn_in_green(self) -> None:
        # A finding present in both the image and the draft draws as a supported
        # (green) box - this is the only path that yields the supported color.
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.2, 0.2, 0.8, 0.8))],
            [DraftFinding(finding="cardiomegaly")],
        )
        canvas = Image.new("RGB", (256, 256), color=(10, 10, 10))
        annotated = annotate_evidence(canvas, outcome)
        colors = {color for _count, color in annotated.getcolors(maxcolors=100000) or []}
        assert (46, 204, 113) in colors

    def test_second_bilateral_label_renders_below_its_box(self) -> None:
        # Two same-label boxes over the left and right lung fields carry long label
        # bands at the same height; drawn independently the bands would overlap and
        # clip each other. The second band must move out of collision - below its
        # own box - so amber band fill appears strictly below the boxes' bottom edge.
        outcome = _outcome(
            [
                ImageFinding(finding="lung_opacity_consolidation", box=(0.30, 0.18, 0.62, 0.39)),
                ImageFinding(finding="lung_opacity_consolidation", box=(0.30, 0.40, 0.62, 0.68)),
            ],
            [],
        )
        canvas = Image.new("RGB", (512, 512), color=(0, 0, 0))
        annotated = annotate_evidence(canvas, outcome)
        box_bottom = round(0.62 * annotated.height)
        below_box = annotated.crop((0, box_bottom + 4, annotated.width, annotated.height))
        colors = {color for _count, color in below_box.getcolors(maxcolors=100000) or []}
        assert (243, 156, 18) in colors

    def test_label_band_clamps_to_right_edge(self) -> None:
        # A box flush against the right edge must keep its label band on-image: the
        # rightmost column carries band/border content rather than the band running
        # off the right side.
        outcome = _outcome([ImageFinding(finding="lung_opacity_consolidation", box=(0.3, 0.7, 0.6, 1.0))], [])
        canvas = Image.new("RGB", (200, 200), color=(0, 0, 0))
        annotated = annotate_evidence(canvas, outcome)
        right_column = [annotated.getpixel((annotated.width - 1, y)) for y in range(annotated.height)]
        assert any(pixel != (0, 0, 0) for pixel in right_column)


class TestFindingsTable:
    def test_rows_cover_image_and_draft_with_human_names(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="pleural_effusion", box=(0.6, 0.1, 0.9, 0.4), confidence=0.8)],
            [DraftFinding(finding="cardiomegaly", span="enlarged heart")],
        )
        rows = findings_table_rows(outcome)
        labels = {row[0] for row in rows}
        # Human display names, not raw snake_case.
        assert "Pleural effusion" in labels
        assert "Cardiomegaly (enlarged heart)" in labels
        assert "pleural_effusion" not in labels

    def test_rows_have_four_columns(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="pneumothorax", box=(0.1, 0.5, 0.4, 0.9))],
            [DraftFinding(finding="pneumothorax")],
        )
        rows = findings_table_rows(outcome)
        assert rows
        assert all(len(row) == 4 for row in rows)

    def test_status_word_is_not_split(self) -> None:
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))], [])
        rows = findings_table_rows(outcome)
        # Status column reads as a whole word, never a split token.
        assert rows[0][2] == "Present"

    def test_image_source_label_and_audit_phrase_are_plain_english(self) -> None:
        outcome = _outcome([ImageFinding(finding="pleural_effusion", box=(0.6, 0.1, 0.9, 0.4))], [])
        row = findings_table_rows(outcome)[0]
        assert row[1] == "Image"
        assert row[3] == "Missing from draft"

    def test_supported_image_finding_audit_phrase(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))],
            [DraftFinding(finding="cardiomegaly")],
        )
        row = findings_table_rows(outcome)[0]
        assert row[3] == "Supported by image"

    def test_absent_and_no_finding_image_rows_excluded(self) -> None:
        # An explicitly-absent image finding and the no_finding sentinel are not real
        # positive findings, so they never produce a row even when a genuine positive
        # finding is present alongside them.
        outcome = _outcome(
            [
                ImageFinding(finding="cardiomegaly", box=(0.3, 0.3, 0.7, 0.7)),
                ImageFinding(finding="pneumothorax", status=FindingStatus.ABSENT),
                ImageFinding(finding="no_finding"),
            ],
            [],
        )
        labels = [row[0] for row in findings_table_rows(outcome)]
        assert labels == ["Cardiomegaly (enlarged heart)"]

    def test_no_finding_draft_row_excluded(self) -> None:
        # A negative draft ("lungs are clear") parses to the no_finding sentinel; it
        # must not appear as a contradictory row beside a real image finding, and a
        # negative draft must behave like an empty one (no draft row at all).
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.3, 0.3, 0.7, 0.7))],
            [DraftFinding(finding="no_finding")],
        )
        rows = findings_table_rows(outcome)
        assert rows == [["Cardiomegaly (enlarged heart)", "Image", "Present", "Missing from draft"]]
        assert all(row[1] != "Draft" for row in rows)

    def test_unsupported_draft_row_has_phrase(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))],
            [DraftFinding(finding="pleural_effusion", span="effusion")],
        )
        draft_rows = [r for r in findings_table_rows(outcome) if r[1] == "Draft"]
        assert draft_rows[0][3] == "Unsupported claim"

    def test_duplicate_image_label_collapses_to_one_row_with_foci_hint(self) -> None:
        # The model can emit the same label at two distinct boxes; the table must show
        # ONE row per label (a non-expert never sees the finding repeated), and that row
        # carries an "(N foci)" hint so the single row reconciles with the two boxes the
        # overlay draws for the label.
        outcome = _outcome(
            [
                ImageFinding(finding="lung_opacity_consolidation", box=(0.236, 0.316, 0.386, 0.422)),
                ImageFinding(finding="lung_opacity_consolidation", box=(0.236, 0.531, 0.386, 0.622)),
            ],
            [],
        )
        rows = findings_table_rows(outcome)
        opacity_rows = [r for r in rows if r[0].startswith("Lung opacity / consolidation")]
        assert len(opacity_rows) == 1
        assert opacity_rows[0][0] == "Lung opacity / consolidation (2 foci)"

    def test_single_box_label_has_no_foci_hint(self) -> None:
        # A label localized once draws a single box, so its row reads plainly with no
        # foci hint.
        outcome = _outcome([ImageFinding(finding="pleural_effusion", box=(0.6, 0.1, 0.9, 0.4))], [])
        row = findings_table_rows(outcome)[0]
        assert row[0] == "Pleural effusion"
        assert "foci" not in row[0]

    def test_near_coincident_same_label_boxes_count_as_one_focus(self) -> None:
        # Two boxes that overlap enough to merge into one overlay cluster are a single
        # drawn region, so the label scores one focus and gets no "(N foci)" hint - the
        # hint tracks drawn boxes, not raw findings.
        box = (0.2, 0.2, 0.8, 0.8)
        outcome = _outcome(
            [
                ImageFinding(finding="lung_opacity_consolidation", box=box),
                ImageFinding(finding="lung_opacity_consolidation", box=box),
            ],
            [],
        )
        opacity_rows = [r for r in findings_table_rows(outcome) if r[0].startswith("Lung opacity / consolidation")]
        assert len(opacity_rows) == 1
        assert opacity_rows[0][0] == "Lung opacity / consolidation"

    def test_three_site_label_reports_three_foci(self) -> None:
        # Three spatially-distinct boxes of one label draw three regions, so the single
        # table row reports "(3 foci)".
        outcome = _outcome(
            [
                ImageFinding(finding="lung_opacity_consolidation", box=(0.10, 0.10, 0.20, 0.20)),
                ImageFinding(finding="lung_opacity_consolidation", box=(0.50, 0.10, 0.60, 0.20)),
                ImageFinding(finding="lung_opacity_consolidation", box=(0.10, 0.50, 0.20, 0.60)),
            ],
            [],
        )
        opacity_rows = [r for r in findings_table_rows(outcome) if r[0].startswith("Lung opacity / consolidation")]
        assert len(opacity_rows) == 1
        assert opacity_rows[0][0] == "Lung opacity / consolidation (3 foci)"

    def test_clear_study_yields_single_no_finding_row(self) -> None:
        # A clear study (no positive image finding, no draft claim) shows one honest
        # "No finding" row instead of an empty table or the pre-run placeholder.
        outcome = _outcome([], [])
        assert findings_table_rows(outcome) == [["No finding", "Image", "Present", ""]]

    def test_clear_study_with_negative_draft_shows_one_no_finding_row(self) -> None:
        # A clear image plus a negative draft yields a single "No finding" row, never
        # two contradictory rows (one from the image, one from the draft sentinel).
        outcome = _outcome(
            [ImageFinding(finding="no_finding")],
            [DraftFinding(finding="no_finding")],
        )
        assert findings_table_rows(outcome) == [["No finding", "Image", "Present", ""]]


class TestAuditPanel:
    def test_panel_lists_missing_unsupported_urgent_with_human_names(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="pneumothorax", box=(0.1, 0.5, 0.4, 0.9))],
            [DraftFinding(finding="cardiomegaly", span="big heart")],
        )
        panel = audit_panel_markdown(outcome)
        # Human display names, never raw snake_case.
        assert "Pneumothorax" in panel
        assert "Cardiomegaly (enlarged heart)" in panel
        assert "pneumothorax" not in panel
        assert "cardiomegaly" not in panel
        # Urgent must be surfaced prominently.
        assert "URGENT" in panel.upper()

    def test_panel_has_how_to_read_orientation_line(self) -> None:
        outcome = _outcome([ImageFinding(finding="pneumothorax", box=(0.1, 0.5, 0.4, 0.9))], [])
        panel = audit_panel_markdown(outcome)
        assert "how to read this" in panel.lower()

    def test_panel_keeps_unsupported_draft_span_quote(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))],
            [DraftFinding(finding="pleural_effusion", span="moderate effusion")],
        )
        panel = audit_panel_markdown(outcome)
        assert "moderate effusion" in panel

    def test_panel_reports_agreement_when_clean(self) -> None:
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))],
            [DraftFinding(finding="cardiomegaly")],
        )
        panel = audit_panel_markdown(outcome)
        assert "no" in panel.lower() or "none" in panel.lower()

    def test_panel_leads_with_draft_parse_note_when_set(self) -> None:
        base = _outcome([ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))], [])
        outcome = AuditOutcome(
            result=base.result,
            comparison=base.comparison,
            draft_parse_note=DRAFT_PARSE_FAILURE_NOTE,
        )
        panel = audit_panel_markdown(outcome)
        assert panel.startswith("**Draft not analyzed:**")
        assert DRAFT_PARSE_FAILURE_NOTE in panel
        assert "re-check the draft" in panel.lower()
        # The rest of the panel still renders after the note.
        assert "how to read this" in panel.lower()

    def test_panel_has_no_draft_note_by_default(self) -> None:
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))], [])
        panel = audit_panel_markdown(outcome)
        assert "Draft not analyzed" not in panel


class TestResultJson:
    def test_round_trips_to_audit_result(self) -> None:
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))], [])
        text = result_json(outcome)
        restored = AuditResult.model_validate_json(text)
        assert restored.box_format == "normalized_y0x0y1x1"
        assert [f.finding for f in restored.image_findings] == ["cardiomegaly"]

    def test_json_includes_disclaimer(self) -> None:
        outcome = _outcome([], [])
        text: Any = result_json(outcome)
        assert "NOT a medical device" in text

    def test_json_omits_always_null_optional_fields(self) -> None:
        # The fine-tuned model emits only {label, box_2d}; confidence and evidence
        # come back null. The displayed JSON must omit those nulls so a non-expert
        # does not see a wall of empty keys.
        outcome = _outcome([ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5))], [])
        text = result_json(outcome)
        assert '"confidence": null' not in text
        assert '"evidence": null' not in text
        assert '"location"' not in text
        # The meaningful fields survive.
        assert '"box"' in text
        assert '"box_format"' in text
        assert '"disclaimer"' in text

    def test_json_keeps_populated_optional_fields(self) -> None:
        # A finding that DOES carry confidence/evidence keeps them; exclude_none only
        # drops nulls, never populated values.
        outcome = _outcome(
            [ImageFinding(finding="cardiomegaly", box=(0.1, 0.1, 0.5, 0.5), confidence=0.7, evidence="enlarged heart")],
            [],
        )
        text = result_json(outcome)
        assert '"confidence": 0.7' in text
        assert "enlarged heart" in text
