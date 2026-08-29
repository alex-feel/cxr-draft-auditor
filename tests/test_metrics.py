"""Tests for the pure-logic evaluation metrics.

All fixtures are hand-checked: each IoU value, confusion case, and rate is
computed by hand in the test so a regression in the metric math is caught by an
exact assertion rather than a tautological round-trip.

These tests exercise only stdlib + numpy + pydantic logic; no torch, no
transformers, no network, no GPU.
"""

from __future__ import annotations

import math

import pytest

from cxr_auditor.eval.metrics import (
    AUDIT_FLAG_TYPES,
    AuditFlagMetrics,
    LocalizationResult,
    PresenceMetrics,
    PresenceReport,
    audit_flag_metrics,
    box_iou,
    localization_result,
    match_boxes,
    presence_metrics,
)
from cxr_auditor.schema import Audit


class TestBoxIoU:
    def test_identical_boxes_iou_is_one(self) -> None:
        box = (0.2, 0.2, 0.6, 0.6)
        assert box_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self) -> None:
        # No overlap on the x axis at all.
        a = (0.0, 0.0, 0.5, 0.4)
        b = (0.0, 0.6, 0.5, 1.0)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_edge_touching_boxes_iou_is_zero(self) -> None:
        # Share only the x=0.5 edge; intersection area is zero.
        a = (0.0, 0.0, 1.0, 0.5)
        b = (0.0, 0.5, 1.0, 1.0)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap_known_value(self) -> None:
        # a = [y0,x0,y1,x1] = [0, 0, 1, 1]  -> area 1.0 (in normalized^2 units).
        # b =                  [0, 0.5, 1, 1.5] but clamp not applied; use in-range.
        # Use a and b both within [0,1]: a=[0,0,1,0.6], b=[0,0.4,1,1.0].
        # a area = 1 * 0.6 = 0.6 ; b area = 1 * 0.6 = 0.6.
        # intersection x in [0.4, 0.6] width 0.2, y full -> area 0.2.
        # union = 0.6 + 0.6 - 0.2 = 1.0 ; IoU = 0.2 / 1.0 = 0.2.
        a = (0.0, 0.0, 1.0, 0.6)
        b = (0.0, 0.4, 1.0, 1.0)
        assert box_iou(a, b) == pytest.approx(0.2)

    def test_nested_box_known_value(self) -> None:
        # outer area = 0.8*0.8 = 0.64 ; inner area = 0.4*0.4 = 0.16.
        # intersection = inner = 0.16 ; union = outer = 0.64.
        # IoU = 0.16 / 0.64 = 0.25.
        outer = (0.1, 0.1, 0.9, 0.9)
        inner = (0.3, 0.3, 0.7, 0.7)
        assert box_iou(outer, inner) == pytest.approx(0.25)

    def test_quarter_overlap_known_value(self) -> None:
        # Two unit-ish squares each 0.6 x 0.6, offset by 0.3 in each axis.
        # a = [0, 0, 0.6, 0.6] ; b = [0.3, 0.3, 0.9, 0.9].
        # intersection = [0.3,0.3,0.6,0.6] -> 0.3 * 0.3 = 0.09.
        # each area = 0.36 ; union = 0.36 + 0.36 - 0.09 = 0.63.
        # IoU = 0.09 / 0.63 = 1/7.
        a = (0.0, 0.0, 0.6, 0.6)
        b = (0.3, 0.3, 0.9, 0.9)
        assert box_iou(a, b) == pytest.approx(1.0 / 7.0)

    def test_zero_area_predicted_box_iou_is_zero(self) -> None:
        # A degenerate (zero-area) box never localizes anything.
        degenerate = (0.5, 0.5, 0.5, 0.5)
        target = (0.4, 0.4, 0.6, 0.6)
        assert box_iou(degenerate, target) == pytest.approx(0.0)

    def test_both_zero_area_iou_is_zero(self) -> None:
        point = (0.5, 0.5, 0.5, 0.5)
        assert box_iou(point, point) == pytest.approx(0.0)

    def test_iou_is_symmetric(self) -> None:
        a = (0.1, 0.2, 0.5, 0.7)
        b = (0.3, 0.1, 0.8, 0.6)
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))


class TestMatchBoxes:
    def test_empty_inputs(self) -> None:
        matches, unmatched_pred, unmatched_gt = match_boxes([], [], iou_threshold=0.3)
        assert matches == []
        assert unmatched_pred == []
        assert unmatched_gt == []

    def test_single_match_above_threshold(self) -> None:
        preds = [(0.1, 0.1, 0.9, 0.9)]
        gts = [(0.3, 0.3, 0.7, 0.7)]  # IoU = 0.25 with the prediction
        # 0.25 < 0.3 -> no match at 0.3.
        matches, up, ug = match_boxes(preds, gts, iou_threshold=0.3)
        assert matches == []
        assert up == [0]
        assert ug == [0]

    def test_match_at_lower_threshold(self) -> None:
        preds = [(0.1, 0.1, 0.9, 0.9)]
        gts = [(0.3, 0.3, 0.7, 0.7)]  # IoU = 0.25
        matches, up, ug = match_boxes(preds, gts, iou_threshold=0.2)
        assert len(matches) == 1
        pred_idx, gt_idx, iou = matches[0]
        assert (pred_idx, gt_idx) == (0, 0)
        assert iou == pytest.approx(0.25)
        assert up == []
        assert ug == []

    def test_greedy_picks_highest_iou_first(self) -> None:
        # One gt overlaps two preds; the higher-IoU pred must win the match.
        gt = (0.0, 0.0, 1.0, 1.0)
        pred_strong = (0.0, 0.0, 1.0, 0.9)  # IoU 0.9
        pred_weak = (0.0, 0.0, 1.0, 0.4)  # IoU 0.4
        preds = [pred_weak, pred_strong]
        gts = [gt]
        matches, up, ug = match_boxes(preds, gts, iou_threshold=0.3)
        assert len(matches) == 1
        pred_idx, gt_idx, iou = matches[0]
        assert pred_idx == 1  # the strong prediction
        assert gt_idx == 0
        assert iou == pytest.approx(0.9)
        assert up == [0]  # the weak prediction is left unmatched
        assert ug == []

    def test_one_to_one_no_double_assignment(self) -> None:
        # Two preds and two gts, each pred best-overlaps a distinct gt.
        gts = [(0.0, 0.0, 0.4, 0.4), (0.6, 0.6, 1.0, 1.0)]
        preds = [(0.0, 0.0, 0.4, 0.4), (0.6, 0.6, 1.0, 1.0)]
        matches, up, ug = match_boxes(preds, gts, iou_threshold=0.5)
        assert len(matches) == 2
        assert up == []
        assert ug == []
        matched_pairs = {(m[0], m[1]) for m in matches}
        assert matched_pairs == {(0, 0), (1, 1)}


class TestLocalizationResult:
    def test_perfect_localization(self) -> None:
        preds = [(0.1, 0.1, 0.5, 0.5)]
        gts = [(0.1, 0.1, 0.5, 0.5)]
        result = localization_result(preds, gts, iou_threshold=0.5)
        assert isinstance(result, LocalizationResult)
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.localization_rate == pytest.approx(1.0)

    def test_localization_rate_definition(self) -> None:
        # 2 gts, 1 of them localized -> rate = 1/2.
        gts = [(0.0, 0.0, 0.4, 0.4), (0.6, 0.6, 1.0, 1.0)]
        preds = [(0.0, 0.0, 0.4, 0.4)]  # localizes the first gt only
        result = localization_result(preds, gts, iou_threshold=0.5)
        assert result.true_positives == 1
        assert result.false_negatives == 1
        assert result.false_positives == 0
        assert result.localization_rate == pytest.approx(0.5)

    def test_false_positive_counts(self) -> None:
        gts = [(0.0, 0.0, 0.4, 0.4)]
        preds = [(0.0, 0.0, 0.4, 0.4), (0.6, 0.6, 1.0, 1.0)]  # second is spurious
        result = localization_result(preds, gts, iou_threshold=0.5)
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 0
        assert result.localization_rate == pytest.approx(1.0)
        assert result.precision == pytest.approx(0.5)

    def test_empty_ground_truth_rate_is_nan(self) -> None:
        result = localization_result([(0.1, 0.1, 0.2, 0.2)], [], iou_threshold=0.5)
        assert result.true_positives == 0
        assert result.false_negatives == 0
        assert result.false_positives == 1
        assert math.isnan(result.localization_rate)

    def test_threshold_sensitivity(self) -> None:
        preds = [(0.1, 0.1, 0.9, 0.9)]
        gts = [(0.3, 0.3, 0.7, 0.7)]  # IoU 0.25
        loose = localization_result(preds, gts, iou_threshold=0.2)
        strict_acceptable = localization_result(preds, gts, iou_threshold=0.3)
        good = localization_result(preds, gts, iou_threshold=0.5)
        assert loose.localization_rate == pytest.approx(1.0)
        assert strict_acceptable.localization_rate == pytest.approx(0.0)
        assert good.localization_rate == pytest.approx(0.0)


class TestPresenceMetrics:
    def test_perfect_presence(self) -> None:
        predicted = [
            {"pleural_effusion", "cardiomegaly"},
            {"no_finding"},
        ]
        expected = [
            {"pleural_effusion", "cardiomegaly"},
            {"no_finding"},
        ]
        report = presence_metrics(predicted, expected)
        assert isinstance(report, PresenceReport)
        eff = report.per_finding["pleural_effusion"]
        assert isinstance(eff, PresenceMetrics)
        assert eff.precision == pytest.approx(1.0)
        assert eff.recall == pytest.approx(1.0)
        assert eff.f1 == pytest.approx(1.0)
        assert report.macro_f1 == pytest.approx(1.0)

    def test_known_confusion_case(self) -> None:
        # Image 1: predicts effusion (correct) and pneumothorax (false positive).
        # Image 2: misses effusion (false negative).
        predicted = [
            {"pleural_effusion", "pneumothorax"},
            set(),
        ]
        expected = [
            {"pleural_effusion"},
            {"pleural_effusion"},
        ]
        report = presence_metrics(predicted, expected)
        eff = report.per_finding["pleural_effusion"]
        # effusion: tp=1 (image1), fn=1 (image2), fp=0.
        assert eff.true_positives == 1
        assert eff.false_negatives == 1
        assert eff.false_positives == 0
        assert eff.precision == pytest.approx(1.0)
        assert eff.recall == pytest.approx(0.5)
        assert eff.f1 == pytest.approx(2.0 / 3.0)
        pneumo = report.per_finding["pneumothorax"]
        # pneumothorax: fp=1 (image1), no gt anywhere.
        assert pneumo.true_positives == 0
        assert pneumo.false_positives == 1
        assert pneumo.false_negatives == 0
        assert pneumo.precision == pytest.approx(0.0)
        assert math.isnan(pneumo.recall)

    def test_recall_undefined_when_no_positives(self) -> None:
        predicted = [{"pleural_effusion"}]
        expected: list[set[str]] = [set()]
        report = presence_metrics(predicted, expected)
        eff = report.per_finding["pleural_effusion"]
        assert eff.false_positives == 1
        assert eff.true_positives == 0
        assert math.isnan(eff.recall)
        assert eff.precision == pytest.approx(0.0)

    def test_precision_undefined_when_no_predictions(self) -> None:
        predicted: list[set[str]] = [set()]
        expected = [{"pleural_effusion"}]
        report = presence_metrics(predicted, expected)
        eff = report.per_finding["pleural_effusion"]
        assert eff.false_negatives == 1
        assert math.isnan(eff.precision)
        assert eff.recall == pytest.approx(0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            presence_metrics([{"pleural_effusion"}], [])

    def test_non_canonical_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="canonical"):
            presence_metrics([{"aortic_enlargement"}], [{"pleural_effusion"}])

    def test_macro_f1_skips_undefined_findings(self) -> None:
        # Only effusion has any signal; the macro average is over findings with a
        # defined F1, so a single perfect finding yields macro F1 of 1.0.
        predicted = [{"pleural_effusion"}]
        expected = [{"pleural_effusion"}]
        report = presence_metrics(predicted, expected)
        assert report.macro_f1 == pytest.approx(1.0)


class TestAuditFlagMetrics:
    def test_flag_types_constant(self) -> None:
        assert AUDIT_FLAG_TYPES == ("missing", "unsupported", "urgent")

    def test_perfect_audit_flags(self) -> None:
        predicted = [Audit(missing_findings=["pleural_effusion"], urgent_review_flags=["pneumothorax"])]
        expected = [Audit(missing_findings=["pleural_effusion"], urgent_review_flags=["pneumothorax"])]
        report = audit_flag_metrics(predicted, expected)
        missing = report["missing"]
        assert isinstance(missing, AuditFlagMetrics)
        assert missing.precision == pytest.approx(1.0)
        assert missing.recall == pytest.approx(1.0)
        urgent = report["urgent"]
        assert urgent.precision == pytest.approx(1.0)
        assert urgent.recall == pytest.approx(1.0)

    def test_missing_flag_confusion(self) -> None:
        # Case A: predicts missing effusion when expected is missing effusion (tp).
        # Case B: predicts missing pneumothorax when none expected (fp).
        # Case C: expects missing cardiomegaly but predicts none (fn).
        predicted = [
            Audit(missing_findings=["pleural_effusion"]),
            Audit(missing_findings=["pneumothorax"]),
            Audit(missing_findings=[]),
        ]
        expected = [
            Audit(missing_findings=["pleural_effusion"]),
            Audit(missing_findings=[]),
            Audit(missing_findings=["cardiomegaly"]),
        ]
        report = audit_flag_metrics(predicted, expected)
        missing = report["missing"]
        assert missing.true_positives == 1
        assert missing.false_positives == 1
        assert missing.false_negatives == 1
        assert missing.precision == pytest.approx(0.5)
        assert missing.recall == pytest.approx(0.5)
        assert missing.f1 == pytest.approx(0.5)

    def test_unsupported_and_urgent_independent(self) -> None:
        # The unsupported flag is exercised while missing/urgent stay empty.
        predicted = [Audit(unsupported_claims=["nodule_mass", "cardiomegaly"])]
        expected = [Audit(unsupported_claims=["nodule_mass"])]
        report = audit_flag_metrics(predicted, expected)
        unsupported = report["unsupported"]
        # tp = nodule_mass ; fp = cardiomegaly ; fn = none.
        assert unsupported.true_positives == 1
        assert unsupported.false_positives == 1
        assert unsupported.false_negatives == 0
        assert unsupported.precision == pytest.approx(0.5)
        assert unsupported.recall == pytest.approx(1.0)
        # missing and urgent had no flags anywhere -> undefined precision/recall.
        assert math.isnan(report["missing"].precision)
        assert math.isnan(report["urgent"].recall)

    def test_duplicate_flags_within_case_are_deduplicated(self) -> None:
        # A flag list is a set of labels per case; duplicates do not inflate counts.
        predicted = [Audit(missing_findings=["pleural_effusion", "pleural_effusion"])]
        expected = [Audit(missing_findings=["pleural_effusion"])]
        report = audit_flag_metrics(predicted, expected)
        missing = report["missing"]
        assert missing.true_positives == 1
        assert missing.false_positives == 0
        assert missing.precision == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            audit_flag_metrics([Audit()], [])

    def test_empty_inputs_yield_nan(self) -> None:
        report = audit_flag_metrics([], [])
        for flag in AUDIT_FLAG_TYPES:
            assert math.isnan(report[flag].precision)
            assert math.isnan(report[flag].recall)
            assert math.isnan(report[flag].f1)
