"""Tests for the evaluation CLI harness.

The CLI is exercised end to end against tiny in-repo JSON fixtures written into
``tmp_path``. No real dataset, no network, no GPU. The RadEval/GREEN hook is
tested only for its lazy-import guard (it must raise a clear error rather than
fail at module import time when the optional dependency is absent).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from cxr_auditor.eval.run_eval import (
    AuditEvalInput,
    ImageEvalInput,
    build_audit_report,
    build_image_report,
    load_audit_input,
    load_image_input,
    main,
    maybe_score_with_radeval,
)


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def image_eval_payload() -> dict[str, Any]:
    """A two-case image-eval payload with one perfect localization and one miss."""
    return {
        "cases": [
            {
                "image_id": "case_perfect",
                "predicted": [
                    {"finding": "pleural_effusion", "box": [0.6, 0.1, 0.9, 0.4]},
                ],
                "expected": [
                    {"finding": "pleural_effusion", "box": [0.6, 0.1, 0.9, 0.4]},
                ],
            },
            {
                "image_id": "case_missed_box",
                "predicted": [
                    {"finding": "cardiomegaly", "box": [0.1, 0.1, 0.2, 0.2]},
                ],
                "expected": [
                    {"finding": "cardiomegaly", "box": [0.4, 0.3, 0.9, 0.8]},
                ],
            },
        ]
    }


@pytest.fixture
def audit_eval_payload() -> dict[str, Any]:
    """A two-case audit-eval payload exercising missing/unsupported/urgent flags."""
    return {
        "cases": [
            {
                "case_id": "drop_effusion",
                "predicted": {"missing_findings": ["pleural_effusion"], "unsupported_claims": [], "urgent_review_flags": []},
                "expected": {"missing_findings": ["pleural_effusion"], "unsupported_claims": [], "urgent_review_flags": []},
            },
            {
                "case_id": "add_pneumo",
                "predicted": {"missing_findings": [], "unsupported_claims": ["pneumothorax"], "urgent_review_flags": []},
                "expected": {"missing_findings": [], "unsupported_claims": ["pneumothorax"], "urgent_review_flags": []},
            },
        ]
    }


class TestLoadImageInput:
    def test_parses_cases(self, tmp_path: Path, image_eval_payload: dict[str, Any]) -> None:
        path = _write_json(tmp_path / "image.json", image_eval_payload)
        parsed = load_image_input(path)
        assert isinstance(parsed, ImageEvalInput)
        assert len(parsed.cases) == 2
        first = parsed.cases[0]
        assert first.image_id == "case_perfect"
        assert first.predicted[0].finding == "pleural_effusion"

    def test_normalizes_finding_labels(self, tmp_path: Path) -> None:
        payload = {
            "cases": [
                {
                    "image_id": "x",
                    "predicted": [{"finding": "Pleural Effusion", "box": [0.6, 0.1, 0.9, 0.4]}],
                    "expected": [{"finding": "pleural-effusion", "box": [0.6, 0.1, 0.9, 0.4]}],
                }
            ]
        }
        path = _write_json(tmp_path / "image.json", payload)
        parsed = load_image_input(path)
        assert parsed.cases[0].predicted[0].finding == "pleural_effusion"
        assert parsed.cases[0].expected[0].finding == "pleural_effusion"


class TestLoadAuditInput:
    def test_parses_cases(self, tmp_path: Path, audit_eval_payload: dict[str, Any]) -> None:
        path = _write_json(tmp_path / "audit.json", audit_eval_payload)
        parsed = load_audit_input(path)
        assert isinstance(parsed, AuditEvalInput)
        assert len(parsed.cases) == 2
        assert parsed.cases[0].predicted.missing_findings == ["pleural_effusion"]


class TestBuildImageReport:
    def test_localization_and_presence(self, tmp_path: Path, image_eval_payload: dict[str, Any]) -> None:
        path = _write_json(tmp_path / "image.json", image_eval_payload)
        parsed = load_image_input(path)
        report = build_image_report(parsed)

        # Presence: effusion appears once (case 1, perfect tp); cardiomegaly once.
        effusion = report["presence"]["per_finding"]["pleural_effusion"]
        assert effusion["true_positives"] == 1
        assert effusion["recall"] == pytest.approx(1.0)
        cardio = report["presence"]["per_finding"]["cardiomegaly"]
        assert cardio["true_positives"] == 1  # present in both predicted and expected

        # Localization at 0.5: effusion box is identical (IoU 1.0 -> localized);
        # cardiomegaly boxes are disjoint (IoU 0.0 -> not localized).
        loc_50 = report["localization"]["0.5"]
        assert loc_50["per_finding"]["pleural_effusion"]["true_positives"] == 1
        assert loc_50["per_finding"]["pleural_effusion"]["localization_rate"] == pytest.approx(1.0)
        assert loc_50["per_finding"]["cardiomegaly"]["true_positives"] == 0
        assert loc_50["per_finding"]["cardiomegaly"]["localization_rate"] == pytest.approx(0.0)

        # Pooled localization across findings at 0.5 -> 1 of 2 gt boxes localized.
        assert loc_50["pooled"]["localization_rate"] == pytest.approx(0.5)

    def test_reports_both_thresholds(self, tmp_path: Path, image_eval_payload: dict[str, Any]) -> None:
        path = _write_json(tmp_path / "image.json", image_eval_payload)
        report = build_image_report(load_image_input(path))
        assert set(report["localization"].keys()) == {"0.3", "0.5"}

    def test_box_none_skipped_for_localization(self, tmp_path: Path) -> None:
        # A finding asserted with no box contributes to presence but not to
        # localization (there is no box to match).
        payload = {
            "cases": [
                {
                    "image_id": "diffuse",
                    "predicted": [{"finding": "no_finding", "box": None}],
                    "expected": [{"finding": "no_finding", "box": None}],
                }
            ]
        }
        path = _write_json(tmp_path / "image.json", payload)
        report = build_image_report(load_image_input(path))
        # no_finding is the negative sentinel and is not a positive finding, so it
        # does not appear in the localization per-finding table.
        assert "no_finding" not in report["localization"]["0.5"]["per_finding"]
        assert report["localization"]["0.5"]["pooled"]["true_positives"] == 0


class TestBuildAuditReport:
    def test_perfect_flags(self, tmp_path: Path, audit_eval_payload: dict[str, Any]) -> None:
        path = _write_json(tmp_path / "audit.json", audit_eval_payload)
        report = build_audit_report(load_audit_input(path))
        assert report["missing"]["precision"] == pytest.approx(1.0)
        assert report["missing"]["recall"] == pytest.approx(1.0)
        assert report["unsupported"]["precision"] == pytest.approx(1.0)
        # urgent never flagged anywhere -> undefined.
        assert math.isnan(report["urgent"]["precision"])


class TestMainCli:
    def test_image_mode_writes_report(
        self, tmp_path: Path, image_eval_payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        in_path = _write_json(tmp_path / "image.json", image_eval_payload)
        out_path = tmp_path / "report.json"
        exit_code = main(["--mode", "image", "--input", str(in_path), "--output", str(out_path)])
        assert exit_code == 0
        assert out_path.exists()
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written["mode"] == "image"
        assert "localization" in written
        # A human-readable table is printed to stdout.
        captured = capsys.readouterr()
        assert "localization" in captured.out.lower()

    def test_audit_mode_writes_report(
        self, tmp_path: Path, audit_eval_payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        in_path = _write_json(tmp_path / "audit.json", audit_eval_payload)
        out_path = tmp_path / "report.json"
        exit_code = main(["--mode", "audit", "--input", str(in_path), "--output", str(out_path)])
        assert exit_code == 0
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written["mode"] == "audit"
        assert "missing" in written["audit_flags"]

    def test_missing_input_file_exits_nonzero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out_path = tmp_path / "report.json"
        exit_code = main(["--mode", "image", "--input", str(tmp_path / "does_not_exist.json"), "--output", str(out_path)])
        assert exit_code != 0

    def test_stdout_only_when_no_output(
        self, tmp_path: Path, audit_eval_payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        in_path = _write_json(tmp_path / "audit.json", audit_eval_payload)
        exit_code = main(["--mode", "audit", "--input", str(in_path)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "missing" in captured.out.lower()


class TestRadEvalHook:
    def test_radeval_hook_raises_clear_error_when_absent(self) -> None:
        # The optional scorer must lazily import and raise a clear, actionable
        # error when the package is not installed, never crash at module import.
        with pytest.raises(RuntimeError, match="RadEval"):
            maybe_score_with_radeval(
                refs=["No acute cardiopulmonary process."],
                hyps=["Left pleural effusion."],
                _import_name="cxr_auditor_nonexistent_radeval_pkg",
            )

    def test_radeval_hook_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            maybe_score_with_radeval(refs=["a", "b"], hyps=["a"])
