"""Command-line evaluation harness for the CXR Draft Auditor.

Two evaluation modes are supported, selected with ``--mode``:

- ``image``: scores image-grounded findings against held-out box ground truth
  (VinDr-CXR / ChestX-Det / NIH ChestX-ray14). It reports per-finding presence
  precision / recall / F1 and IoU-thresholded localization at ``0.3`` (acceptable)
  and ``0.5`` (good), both per finding and pooled across findings.
- ``audit``: scores the deterministic audit flags (missing / unsupported /
  urgent) against an expected audit on the synthetic stress set, reporting
  precision / recall / F1 per flag type.

The CLI reads one JSON input file describing aligned predicted-vs-expected cases,
prints a human-readable metrics table, and optionally writes a JSON report.

Input JSON contracts
--------------------
Image mode (``--mode image``)::

    {"cases": [
        {"image_id": "<id>",
         "predicted": [{"finding": "pleural_effusion", "box": [y0, x0, y1, x1], ...}, ...],
         "expected":  [{"finding": "pleural_effusion", "box": [y0, x0, y1, x1], ...}, ...]},
        ...
    ]}

Each ``predicted`` / ``expected`` element is an ``ImageFinding`` (see
``cxr_auditor.schema``); boxes are the canonical normalized ``[y0, x0, y1, x1]``
format and may be ``null`` for non-localizable findings.

Audit mode (``--mode audit``)::

    {"cases": [
        {"case_id": "<id>",
         "predicted": {"missing_findings": [...], "unsupported_claims": [...], "urgent_review_flags": [...]},
         "expected":  {"missing_findings": [...], "unsupported_claims": [...], "urgent_review_flags": [...]}},
        ...
    ]}

The ``predicted`` / ``expected`` objects are ``Audit`` objects.

Optional RadEval / GREEN hook
-----------------------------
``maybe_score_with_radeval`` is a lazy, off-by-default bridge to the external
``RadEval`` package (which provides GREEN and other radiology-report generation
metrics). It is intentionally NOT a project dependency: the function imports the
package only when called and raises a clear, actionable ``RuntimeError`` when the
package is absent, so the rest of the harness runs (and its tests pass) without
any network access or heavyweight install.

This module is dependency-light (stdlib + numpy + pydantic). It imports no model
stack.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cxr_auditor.eval.metrics import (
    AUDIT_FLAG_TYPES,
    AuditFlagMetrics,
    LocalizationResult,
    PresenceReport,
    audit_flag_metrics,
    localization_result,
    presence_metrics,
)
from cxr_auditor.findings import NO_FINDING, POSITIVE_FINDINGS
from cxr_auditor.schema import Audit, ImageFinding, NormalizedBox
from cxr_auditor.schema import FindingStatus as _FindingStatus

# The two localization IoU thresholds reported by the harness. ``0.3`` is the
# "acceptable" bar and ``0.5`` the "good" bar used throughout the project.
LOCALIZATION_THRESHOLDS: tuple[float, ...] = (0.3, 0.5)


class ImageEvalCase(BaseModel):
    """One image's predicted and expected grounded findings.

    Attributes:
        image_id: Identifier of the image (32-char hex for VinDr, free-form
            otherwise). Used only for reporting and traceability.
        predicted: The model's grounded findings for the image.
        expected: The ground-truth grounded findings for the image.
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str
    predicted: list[ImageFinding] = Field(default_factory=list)
    expected: list[ImageFinding] = Field(default_factory=list)


class ImageEvalInput(BaseModel):
    """Top-level image-mode input: a list of aligned per-image cases."""

    model_config = ConfigDict(extra="forbid")

    cases: list[ImageEvalCase] = Field(default_factory=list)


class AuditEvalCase(BaseModel):
    """One synthetic stress case's predicted and expected audit objects.

    Attributes:
        case_id: Identifier of the synthetic case (for example ``drop_effusion``).
        predicted: The audit the comparator produced.
        expected: The audit the synthetic corruption recipe expects.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    predicted: Audit
    expected: Audit


class AuditEvalInput(BaseModel):
    """Top-level audit-mode input: a list of aligned per-case audit objects."""

    model_config = ConfigDict(extra="forbid")

    cases: list[AuditEvalCase] = Field(default_factory=list)


def load_image_input(path: Path) -> ImageEvalInput:
    """Load and validate an image-mode evaluation input file.

    Args:
        path: Path to the image-mode JSON input file.

    Returns:
        The validated ``ImageEvalInput``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON is malformed or fails schema validation
            (pydantic ``ValidationError`` is a ``ValueError`` subclass).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ImageEvalInput.model_validate(raw)


def load_audit_input(path: Path) -> AuditEvalInput:
    """Load and validate an audit-mode evaluation input file.

    Args:
        path: Path to the audit-mode JSON input file.

    Returns:
        The validated ``AuditEvalInput``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON is malformed or fails schema validation.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return AuditEvalInput.model_validate(raw)


def _present_label_set(findings: Sequence[ImageFinding]) -> set[str]:
    """Collapse a finding list to the set of positive labels asserted present.

    The negative sentinel ``no_finding`` is dropped: it represents the absence of
    any positive claim and is not a positive presence label.
    """
    return {
        finding.finding for finding in findings if finding.status is _FindingStatus.PRESENT and finding.finding != NO_FINDING
    }


def _boxes_by_finding(findings: Sequence[ImageFinding]) -> dict[str, list[NormalizedBox]]:
    """Group present, box-carrying findings by canonical label.

    Findings without a box (``box is None``) are skipped: there is nothing to
    localize. The negative sentinel ``no_finding`` is excluded.
    """
    grouped: dict[str, list[NormalizedBox]] = {label: [] for label in POSITIVE_FINDINGS}
    for finding in findings:
        if finding.status is not _FindingStatus.PRESENT:
            continue
        if finding.finding == NO_FINDING or finding.box is None:
            continue
        grouped[finding.finding].append(finding.box)
    return grouped


def _localization_result_to_dict(result: LocalizationResult) -> dict[str, Any]:
    """Serialize a ``LocalizationResult`` to a JSON-friendly dict."""
    return {
        "iou_threshold": result.iou_threshold,
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "localization_rate": result.localization_rate,
        "precision": result.precision,
    }


def _presence_report_to_dict(report: PresenceReport) -> dict[str, Any]:
    """Serialize a ``PresenceReport`` to a JSON-friendly dict."""
    return {
        "n_cases": report.n_cases,
        "macro_f1": report.macro_f1,
        "per_finding": {
            label: {
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }
            for label, metrics in report.per_finding.items()
        },
    }


def _audit_flag_to_dict(metrics: AuditFlagMetrics) -> dict[str, Any]:
    """Serialize an ``AuditFlagMetrics`` to a JSON-friendly dict."""
    return {
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def build_image_report(data: ImageEvalInput) -> dict[str, Any]:
    """Compute presence and localization metrics for image-mode input.

    Presence is scored per finding over all cases. Localization is scored per
    finding and pooled across findings at each of ``LOCALIZATION_THRESHOLDS``.

    Args:
        data: The validated image-mode input.

    Returns:
        A JSON-friendly report dict with ``n_cases``, ``presence``, and
        ``localization`` (keyed by the string form of each IoU threshold).
    """
    predicted_presence = [_present_label_set(case.predicted) for case in data.cases]
    expected_presence = [_present_label_set(case.expected) for case in data.cases]
    presence = presence_metrics(predicted_presence, expected_presence)

    localization: dict[str, Any] = {}
    for threshold in LOCALIZATION_THRESHOLDS:
        per_finding: dict[str, dict[str, Any]] = {}
        pooled_tp = pooled_fp = pooled_fn = 0
        for label in POSITIVE_FINDINGS:
            pred_boxes: list[NormalizedBox] = []
            gt_boxes: list[NormalizedBox] = []
            for case in data.cases:
                pred_boxes.extend(_boxes_by_finding(case.predicted)[label])
                gt_boxes.extend(_boxes_by_finding(case.expected)[label])
            result = localization_result(pred_boxes, gt_boxes, iou_threshold=threshold)
            # Skip findings that have neither a predicted nor a ground-truth box;
            # they carry no localization signal at all.
            if result.true_positives + result.false_positives + result.false_negatives == 0:
                continue
            per_finding[label] = _localization_result_to_dict(result)
            pooled_tp += result.true_positives
            pooled_fp += result.false_positives
            pooled_fn += result.false_negatives

        pooled = LocalizationResult(
            iou_threshold=threshold,
            true_positives=pooled_tp,
            false_positives=pooled_fp,
            false_negatives=pooled_fn,
        )
        localization[f"{threshold}"] = {
            "per_finding": per_finding,
            "pooled": _localization_result_to_dict(pooled),
        }

    return {
        "n_cases": len(data.cases),
        "presence": _presence_report_to_dict(presence),
        "localization": localization,
    }


def build_audit_report(data: AuditEvalInput) -> dict[str, Any]:
    """Compute audit-flag precision/recall/F1 for audit-mode input.

    Args:
        data: The validated audit-mode input.

    Returns:
        A JSON-friendly dict keyed by each entry of ``AUDIT_FLAG_TYPES``.
    """
    predicted = [case.predicted for case in data.cases]
    expected = [case.expected for case in data.cases]
    metrics = audit_flag_metrics(predicted, expected)
    return {flag: _audit_flag_to_dict(metrics[flag]) for flag in AUDIT_FLAG_TYPES}


def maybe_score_with_radeval(
    refs: Sequence[str],
    hyps: Sequence[str],
    _import_name: str = "RadEval",
) -> dict[str, float]:
    """Score reference vs hypothesis reports with the optional RadEval package.

    This is an off-by-default cross-check hook for free-text report-generation
    metrics (RadEval bundles GREEN, RadGraph-F1, and others). RadEval is NOT a
    project dependency: it is imported lazily here, only when this function runs.
    When the package is absent the function raises a clear ``RuntimeError`` that
    names the package and how to install it, rather than letting an ``ImportError``
    surface at module import time.

    Args:
        refs: Reference (ground-truth) report strings.
        hyps: Hypothesis (generated) report strings, aligned with ``refs``.
        _import_name: The package import name to load. Overridable so tests can
            assert the missing-dependency path without installing anything.

    Returns:
        A dict of metric name to score, as produced by RadEval.

    Raises:
        ValueError: If ``refs`` and ``hyps`` differ in length.
        RuntimeError: If the RadEval package cannot be imported.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs and hyps must have the same length, got {len(refs)} and {len(hyps)}")

    try:
        radeval = importlib.import_module(_import_name)
    except ImportError as exc:
        raise RuntimeError(
            f"RadEval is not installed (tried to import {_import_name!r}); this optional "
            "report-generation scorer is off by default. Install it with "
            "'uv pip install RadEval' to enable GREEN/RadGraph cross-checks."
        ) from exc

    # RadEval exposes a `RadEval` evaluator class whose instances are called with
    # `refs=` and `hyps=` keyword arguments and return a dict of metric scores.
    # The hook is written against that documented interface; the import-guard
    # above is what the test suite exercises without the package present.
    evaluator = radeval.RadEval()  # pragma: no cover - requires the optional package
    return dict(evaluator(refs=list(refs), hyps=list(hyps)))  # pragma: no cover


def _format_metric(value: float) -> str:
    """Format a metric for the text table, rendering NaN as ``n/a``."""
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def _render_image_table(report: dict[str, Any]) -> str:
    """Render the image-mode report as a fixed-width text table."""
    lines: list[str] = []
    lines.append(f"Image evaluation over {report['n_cases']} case(s)")
    lines.append("")
    lines.append("Presence (per finding): precision / recall / F1")
    lines.append(f"  {'finding':<28} {'prec':>7} {'recall':>7} {'f1':>7}  (tp/fp/fn)")
    presence = report["presence"]
    for label, metrics in presence["per_finding"].items():
        counts = f"({metrics['true_positives']}/{metrics['false_positives']}/{metrics['false_negatives']})"
        lines.append(
            f"  {label:<28} {_format_metric(metrics['precision']):>7} "
            f"{_format_metric(metrics['recall']):>7} {_format_metric(metrics['f1']):>7}  {counts}"
        )
    lines.append(f"  {'macro F1':<28} {_format_metric(presence['macro_f1']):>23}")
    lines.append("")
    for threshold, block in report["localization"].items():
        lines.append(f"Localization @ IoU {threshold}: rate (localized gt boxes / total gt boxes)")
        lines.append(f"  {'finding':<28} {'rate':>7} {'prec':>7}  (tp/fp/fn)")
        for label, metrics in block["per_finding"].items():
            counts = f"({metrics['true_positives']}/{metrics['false_positives']}/{metrics['false_negatives']})"
            lines.append(
                f"  {label:<28} {_format_metric(metrics['localization_rate']):>7} "
                f"{_format_metric(metrics['precision']):>7}  {counts}"
            )
        pooled = block["pooled"]
        pooled_counts = f"({pooled['true_positives']}/{pooled['false_positives']}/{pooled['false_negatives']})"
        lines.append(
            f"  {'pooled':<28} {_format_metric(pooled['localization_rate']):>7} "
            f"{_format_metric(pooled['precision']):>7}  {pooled_counts}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_audit_table(report: dict[str, Any]) -> str:
    """Render the audit-mode report as a fixed-width text table."""
    lines: list[str] = []
    lines.append("Audit flag evaluation: precision / recall / F1 per flag type")
    lines.append(f"  {'flag':<14} {'prec':>7} {'recall':>7} {'f1':>7}  (tp/fp/fn)")
    for flag in AUDIT_FLAG_TYPES:
        metrics = report[flag]
        counts = f"({metrics['true_positives']}/{metrics['false_positives']}/{metrics['false_negatives']})"
        lines.append(
            f"  {flag:<14} {_format_metric(metrics['precision']):>7} "
            f"{_format_metric(metrics['recall']):>7} {_format_metric(metrics['f1']):>7}  {counts}"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the evaluation CLI."""
    parser = argparse.ArgumentParser(
        prog="cxr-eval",
        description=(
            "Evaluate CXR Draft Auditor outputs. Mode 'image' scores grounded findings "
            "against held-out box datasets (presence P/R/F1 and IoU localization @0.3/@0.5); "
            "mode 'audit' scores the missing/unsupported/urgent flags on the synthetic stress set."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("image", "audit"),
        help="Evaluation mode: 'image' for grounded-finding metrics, 'audit' for audit-flag metrics.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the JSON input file describing aligned predicted-vs-expected cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON metrics report. When omitted, only the text table is printed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the evaluation CLI.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, ``1`` on a load/validation error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "image":
            image_report = build_image_report(load_image_input(args.input))
            table = _render_image_table(image_report)
            payload: dict[str, Any] = {"mode": "image", **image_report}
        else:
            audit_report = build_audit_report(load_audit_input(args.input))
            table = _render_audit_table(audit_report)
            payload = {"mode": "audit", "audit_flags": audit_report}
    except (OSError, ValueError) as exc:
        print(f"error: failed to evaluate {args.input}: {exc}", file=sys.stderr)
        return 1

    print(table, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote JSON report to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
