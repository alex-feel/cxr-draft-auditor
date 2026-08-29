"""Pure-logic evaluation metrics for the CXR Draft Auditor.

This module implements the quantitative metrics the project reports:

- ``box_iou`` and ``match_boxes``: intersection-over-union of bounding boxes and a
  greedy one-to-one matcher used to decide whether a predicted box localizes a
  ground-truth box.
- ``localization_result`` / ``LocalizationResult``: IoU-thresholded localization
  counts and rate. The project reports localization at two thresholds:
  ``0.3`` (acceptable) and ``0.5`` (good).
- ``presence_metrics`` / ``PresenceMetrics`` / ``PresenceReport``: per-finding
  precision, recall, and F1 for finding *presence* (independent of where a box
  lands), plus a macro-averaged F1 over findings with a defined score.
- ``audit_flag_metrics`` / ``AuditFlagMetrics``: precision and recall of the three
  deterministic audit flags (``missing``, ``unsupported``, ``urgent``) measured
  against expected audit objects on the synthetic stress set.

Boxes are the canonical normalized format ``[y0, x0, y1, x1]`` (see
``cxr_auditor.schema``). IoU is scale-invariant, so the metrics operate directly
on the normalized coordinates without converting to pixels.

Undefined ratios (a precision with no predictions, a recall with no positives, a
localization rate with no ground-truth boxes) are reported as ``float('nan')``
rather than a misleading ``0.0`` or ``1.0``. Callers that aggregate must skip NaN
entries; ``macro_f1`` does this internally.

The module depends only on the standard library and numpy. It never imports
torch, transformers, or gradio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cxr_auditor.findings import CANONICAL_FINDING_SET, POSITIVE_FINDINGS
from cxr_auditor.schema import Audit, NormalizedBox

# The three deterministic audit flag types, in a fixed presentation order. These
# correspond one-to-one to the list fields of ``cxr_auditor.schema.Audit``.
AUDIT_FLAG_TYPES: tuple[str, ...] = ("missing", "unsupported", "urgent")

# Mapping from the public flag-type name to the ``Audit`` attribute that holds it.
_FLAG_ATTRIBUTES: dict[str, str] = {
    "missing": "missing_findings",
    "unsupported": "unsupported_claims",
    "urgent": "urgent_review_flags",
}


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` or NaN when the denominator is zero.

    A zero denominator means the ratio is genuinely undefined (no predictions for
    precision, no positives for recall). Reporting NaN keeps an undefined cell
    from masquerading as a real score and lets aggregators skip it.
    """
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _f1_from_pr(precision: float, recall: float) -> float:
    """Combine precision and recall into an F1 score, propagating NaN.

    F1 is the harmonic mean of precision and recall. It is undefined (NaN) when
    either input is undefined, and is ``0.0`` when both are defined but their sum
    is zero (no true positives at all).
    """
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def box_iou(box_a: NormalizedBox, box_b: NormalizedBox) -> float:
    """Compute intersection-over-union of two normalized boxes.

    Both boxes are the canonical normalized format ``[y0, x0, y1, x1]`` with
    ``(y0, x0)`` the top-left and ``(y1, x1)`` the bottom-right corner. IoU is
    invariant to the shared normalization, so the result is identical whether the
    boxes are expressed in normalized or pixel coordinates as long as both use the
    same convention.

    Args:
        box_a: First normalized box.
        box_b: Second normalized box.

    Returns:
        The IoU in ``[0, 1]``. Returns ``0.0`` when either box has zero area or
        the boxes do not overlap.
    """
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b

    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_y0 = max(ay0, by0)
    inter_x0 = max(ax0, bx0)
    inter_y1 = min(ay1, by1)
    inter_x1 = min(ax1, bx1)
    inter_area = max(0.0, inter_y1 - inter_y0) * max(0.0, inter_x1 - inter_x0)
    if inter_area <= 0.0:
        return 0.0

    union_area = area_a + area_b - inter_area
    return inter_area / union_area


def _iou_matrix(pred_boxes: list[NormalizedBox], gt_boxes: list[NormalizedBox]) -> np.ndarray:
    """Build the dense IoU matrix between predicted and ground-truth boxes.

    Returns a ``(n_pred, n_gt)`` float array; ``matrix[i, j]`` is the IoU of
    predicted box ``i`` with ground-truth box ``j``.
    """
    matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            matrix[i, j] = box_iou(pred, gt)
    return matrix


def match_boxes(
    pred_boxes: list[NormalizedBox],
    gt_boxes: list[NormalizedBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match predicted boxes to ground-truth boxes by IoU.

    Pairs are formed by repeatedly taking the highest remaining IoU that clears
    ``iou_threshold`` and removing both the predicted and the ground-truth box
    from further consideration. This yields a one-to-one assignment in which a
    high-overlap prediction wins a contested ground-truth box over a low-overlap
    one. The greedy assignment is the standard rule for single-class localization
    scoring and is deterministic for a fixed input order.

    Args:
        pred_boxes: Predicted normalized boxes.
        gt_boxes: Ground-truth normalized boxes.
        iou_threshold: Minimum IoU for a pair to count as a match.

    Returns:
        A tuple ``(matches, unmatched_pred, unmatched_gt)`` where ``matches`` is a
        list of ``(pred_index, gt_index, iou)`` triples sorted by descending IoU,
        ``unmatched_pred`` is the sorted list of predicted indices left over
        (false positives), and ``unmatched_gt`` is the sorted list of ground-truth
        indices left over (false negatives).
    """
    matrix = _iou_matrix(pred_boxes, gt_boxes)

    # Enumerate every candidate pair that clears the threshold, then consume them
    # greedily from highest IoU down. Ties break by (pred_index, gt_index) so the
    # assignment is stable.
    candidates = [
        (matrix[i, j], i, j) for i in range(len(pred_boxes)) for j in range(len(gt_boxes)) if matrix[i, j] >= iou_threshold
    ]
    candidates.sort(key=lambda triple: (-triple[0], triple[1], triple[2]))

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        matches.append((i, j, float(iou)))
        used_pred.add(i)
        used_gt.add(j)

    unmatched_pred = sorted(i for i in range(len(pred_boxes)) if i not in used_pred)
    unmatched_gt = sorted(j for j in range(len(gt_boxes)) if j not in used_gt)
    return matches, unmatched_pred, unmatched_gt


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """IoU-thresholded localization counts for one finding class (or pooled).

    Attributes:
        iou_threshold: The IoU threshold used to decide a match.
        true_positives: Ground-truth boxes that a prediction localized.
        false_positives: Predicted boxes that did not match any ground-truth box.
        false_negatives: Ground-truth boxes left unlocalized.
    """

    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def localization_rate(self) -> float:
        """Fraction of ground-truth boxes that were localized (recall of boxes).

        Returns NaN when there are no ground-truth boxes (the rate is undefined).
        """
        return _safe_ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def precision(self) -> float:
        """Fraction of predicted boxes that localized a ground-truth box.

        Returns NaN when there were no predicted boxes.
        """
        return _safe_ratio(self.true_positives, self.true_positives + self.false_positives)


def localization_result(
    pred_boxes: list[NormalizedBox],
    gt_boxes: list[NormalizedBox],
    iou_threshold: float,
) -> LocalizationResult:
    """Score localization of predicted boxes against ground-truth boxes.

    A ground-truth box is localized when a predicted box matches it at or above
    ``iou_threshold`` under the greedy one-to-one assignment of ``match_boxes``.

    Args:
        pred_boxes: Predicted normalized boxes for one finding class.
        gt_boxes: Ground-truth normalized boxes for the same class.
        iou_threshold: The IoU threshold (for example ``0.3`` or ``0.5``).

    Returns:
        A ``LocalizationResult`` with true-positive, false-positive, and
        false-negative box counts at the threshold.
    """
    matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
    return LocalizationResult(
        iou_threshold=iou_threshold,
        true_positives=len(matches),
        false_positives=len(unmatched_pred),
        false_negatives=len(unmatched_gt),
    )


@dataclass(frozen=True, slots=True)
class PresenceMetrics:
    """Precision, recall, and F1 for the presence of one finding class.

    Presence is a per-image binary judgment: did the model assert finding X on an
    image where the ground truth asserts (or denies) X? Box position is ignored.

    Attributes:
        true_positives: Images where both prediction and ground truth assert it.
        false_positives: Images where only the prediction asserts it.
        false_negatives: Images where only the ground truth asserts it.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Precision; NaN when the finding was never predicted."""
        return _safe_ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall; NaN when the finding never appears in the ground truth."""
        return _safe_ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        """F1 score; NaN when either precision or recall is undefined."""
        return _f1_from_pr(self.precision, self.recall)


@dataclass(frozen=True, slots=True)
class PresenceReport:
    """Per-finding presence metrics plus a macro-averaged F1.

    Attributes:
        per_finding: Map from each positive finding label to its presence metrics.
        n_cases: Number of images (cases) scored.
    """

    per_finding: dict[str, PresenceMetrics]
    n_cases: int = field(default=0)

    @property
    def macro_f1(self) -> float:
        """Macro-average of the per-finding F1 over findings with a defined F1.

        Findings whose F1 is undefined (never predicted and never present, so both
        precision and recall are NaN) are excluded from the average. Returns NaN
        when no finding has a defined F1.
        """
        defined = [metrics.f1 for metrics in self.per_finding.values() if not math.isnan(metrics.f1)]
        if not defined:
            return float("nan")
        return sum(defined) / len(defined)


def _validate_label_sets(label_sets: list[set[str]], side: str) -> None:
    """Reject any label that is not a canonical positive finding.

    ``no_finding`` is the negative sentinel, not a positive presence label, so it
    is permitted in the input sets but contributes to no positive finding's
    counts. Any label outside the canonical set is a caller error.
    """
    for case in label_sets:
        for label in case:
            if label not in CANONICAL_FINDING_SET:
                raise ValueError(
                    f"{side} label {label!r} is not a canonical finding; expected one of {sorted(CANONICAL_FINDING_SET)}"
                )


def presence_metrics(
    predicted: list[set[str]],
    expected: list[set[str]],
) -> PresenceReport:
    """Compute per-finding presence precision, recall, and F1.

    Each element of ``predicted`` and ``expected`` is the set of canonical finding
    labels asserted present for one image. The two lists are aligned by index, so
    they must have the same length. The negative sentinel ``no_finding`` may
    appear in a set; it contributes to no positive finding's confusion counts (it
    represents the absence of any positive claim).

    Args:
        predicted: Per-image sets of predicted present findings.
        expected: Per-image sets of ground-truth present findings.

    Returns:
        A ``PresenceReport`` with metrics for each positive finding and a
        macro-averaged F1.

    Raises:
        ValueError: If the lists differ in length or contain a non-canonical
            label.
    """
    if len(predicted) != len(expected):
        raise ValueError(f"predicted and expected must have the same length, got {len(predicted)} and {len(expected)}")
    _validate_label_sets(predicted, "predicted")
    _validate_label_sets(expected, "expected")

    counts: dict[str, list[int]] = {label: [0, 0, 0] for label in POSITIVE_FINDINGS}
    for pred_set, exp_set in zip(predicted, expected, strict=True):
        for label in POSITIVE_FINDINGS:
            in_pred = label in pred_set
            in_exp = label in exp_set
            if in_pred and in_exp:
                counts[label][0] += 1
            elif in_pred and not in_exp:
                counts[label][1] += 1
            elif not in_pred and in_exp:
                counts[label][2] += 1

    per_finding = {
        label: PresenceMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
        for label, (tp, fp, fn) in counts.items()
    }
    return PresenceReport(per_finding=per_finding, n_cases=len(predicted))


@dataclass(frozen=True, slots=True)
class AuditFlagMetrics:
    """Precision, recall, and F1 for one audit flag type across all cases.

    Counts are pooled over every audit case: a flagged label in one case is one
    instance. Within a single case the flag labels are treated as a set, so a
    duplicate label in the same case is counted once.

    Attributes:
        flag_type: The flag type name (``missing``, ``unsupported``, ``urgent``).
        true_positives: Flagged labels present in both prediction and expectation.
        false_positives: Flagged labels predicted but not expected.
        false_negatives: Flagged labels expected but not predicted.
    """

    flag_type: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Precision; NaN when the flag was never predicted."""
        return _safe_ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall; NaN when the flag was never expected."""
        return _safe_ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        """F1 score; NaN when either precision or recall is undefined."""
        return _f1_from_pr(self.precision, self.recall)


def audit_flag_metrics(
    predicted: list[Audit],
    expected: list[Audit],
) -> dict[str, AuditFlagMetrics]:
    """Compute precision/recall/F1 for each audit flag type.

    For each flag type the comparison is set-based per case: the predicted and
    expected label sets for that flag are compared, and the per-case true
    positives, false positives, and false negatives are pooled across all cases.

    Args:
        predicted: Predicted ``Audit`` objects, one per case.
        expected: Expected ``Audit`` objects, aligned by index with ``predicted``.

    Returns:
        A dict keyed by each entry of ``AUDIT_FLAG_TYPES`` mapping to its
        ``AuditFlagMetrics``.

    Raises:
        ValueError: If the lists differ in length.
    """
    if len(predicted) != len(expected):
        raise ValueError(f"predicted and expected must have the same length, got {len(predicted)} and {len(expected)}")

    counts: dict[str, list[int]] = {flag: [0, 0, 0] for flag in AUDIT_FLAG_TYPES}
    for pred_audit, exp_audit in zip(predicted, expected, strict=True):
        for flag in AUDIT_FLAG_TYPES:
            attribute = _FLAG_ATTRIBUTES[flag]
            pred_set = set(getattr(pred_audit, attribute))
            exp_set = set(getattr(exp_audit, attribute))
            counts[flag][0] += len(pred_set & exp_set)
            counts[flag][1] += len(pred_set - exp_set)
            counts[flag][2] += len(exp_set - pred_set)

    return {
        flag: AuditFlagMetrics(
            flag_type=flag,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
        for flag, (tp, fp, fn) in counts.items()
    }
