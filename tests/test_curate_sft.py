"""Tests for the SFT corpus curation CLI (``scripts/curate_sft.py``).

The curation script is a user-run pure-logic CLI that is not part of the installed
``cxr_auditor`` package, so it is loaded here directly from its file path (the same
mechanism ``test_scripts_importable.py`` uses for the other scripts). These tests
exercise the box dedup/merge logic, the class-balance downsampling, and the
stratified train/val split on small in-memory fixtures with no real dataset, no
network, and no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cxr_auditor.findings import NO_FINDING
from cxr_auditor.sft_dataset import build_sft_record, validate_sft_record

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(module_name: str) -> ModuleType:
    """Load a script module from ``scripts/`` by file path."""
    path = SCRIPTS_DIR / f"{module_name}.py"
    qualified_name = f"_scripts_{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def curate_sft() -> ModuleType:
    """The imported ``scripts/curate_sft.py`` module."""
    return _load_script("curate_sft")


def _finding(label: str, box: tuple[float, float, float, float] | None) -> dict[str, Any]:
    """Build one assistant-target finding dict in MedGemma-native grounding form."""
    return {"label": label, "box_2d": list(box) if box is not None else None}


def _record(image_id: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one SFT record from raw finding dicts, matching prepare_sft output."""
    from cxr_auditor.schema import ImageFinding

    if findings == [_finding(NO_FINDING, None)]:
        image_findings: list[ImageFinding] = []
    else:
        image_findings = [
            ImageFinding(finding=element["label"], box=tuple(element["box_2d"]) if element["box_2d"] else None)
            for element in findings
        ]
    return build_sft_record(f"images/{image_id}.png", image_findings)


def _normal_record(image_id: str) -> dict[str, Any]:
    """Build a no_finding-only (normal study) SFT record."""
    return _record(image_id, [_finding(NO_FINDING, None)])


def test_resolve_cross_finding_overlaps_keeps_more_specific(curate_sft: ModuleType) -> None:
    same = (0.2, 0.2, 0.4, 0.4)
    findings = [
        _finding("lung_opacity_consolidation", same),
        _finding("nodule_mass", same),  # same region, more specific -> wins
        _finding("cardiomegaly", (0.5, 0.3, 0.7, 0.7)),  # distinct region -> kept
        _finding(NO_FINDING, None),  # null box -> passes through
    ]
    resolved = curate_sft.resolve_cross_finding_overlaps(findings, iou_threshold=0.6)
    labels = [element["label"] for element in resolved]
    assert "nodule_mass" in labels
    assert "lung_opacity_consolidation" not in labels
    assert "cardiomegaly" in labels
    assert NO_FINDING in labels


def test_resolve_cross_finding_overlaps_noop_at_zero_threshold(curate_sft: ModuleType) -> None:
    same = (0.2, 0.2, 0.4, 0.4)
    findings = [_finding("lung_opacity_consolidation", same), _finding("nodule_mass", same)]
    resolved = curate_sft.resolve_cross_finding_overlaps(findings, iou_threshold=0.0)
    assert len(resolved) == 2  # no-op: both kept


# --- box dedup / merge --------------------------------------------------------------


def test_curate_findings_drops_exact_duplicate_boxes(curate_sft: ModuleType) -> None:
    """Two identical (label, box) pairs collapse to a single finding."""
    findings = [
        _finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7)),
        _finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7)),
    ]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    assert curated == [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]


def test_curate_findings_merges_high_iou_boxes_into_mean(curate_sft: ModuleType) -> None:
    """Near-duplicate boxes above the IoU threshold merge to their mean box."""
    findings = [
        _finding("pleural_effusion", (0.60, 0.10, 0.90, 0.40)),
        _finding("pleural_effusion", (0.62, 0.12, 0.92, 0.42)),
    ]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    assert len(curated) == 1
    assert curated[0]["label"] == "pleural_effusion"
    mean_box = curated[0]["box_2d"]
    assert mean_box == pytest.approx([0.61, 0.11, 0.91, 0.41])


def test_curate_findings_keeps_distinct_non_overlapping_boxes(curate_sft: ModuleType) -> None:
    """Boxes below the IoU threshold are kept as separate findings."""
    findings = [
        _finding("nodule_mass", (0.10, 0.10, 0.20, 0.20)),
        _finding("nodule_mass", (0.70, 0.70, 0.85, 0.85)),
    ]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    assert len(curated) == 2


def test_curate_findings_merges_per_label_independently(curate_sft: ModuleType) -> None:
    """Overlapping boxes of different labels are never merged together."""
    findings = [
        _finding("cardiomegaly", (0.10, 0.10, 0.50, 0.50)),
        _finding("lung_opacity_consolidation", (0.11, 0.11, 0.51, 0.51)),
    ]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    labels = sorted(element["label"] for element in curated)
    assert labels == ["cardiomegaly", "lung_opacity_consolidation"]


def test_curate_findings_clusters_three_radiologist_triplication(curate_sft: ModuleType) -> None:
    """A 3-radiologist triplication of one finding collapses to one box."""
    findings = [
        _finding("lung_opacity_consolidation", (0.190, 0.292, 0.289, 0.392)),
        _finding("lung_opacity_consolidation", (0.190, 0.294, 0.290, 0.391)),
        _finding("lung_opacity_consolidation", (0.184, 0.303, 0.291, 0.389)),
    ]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    assert len(curated) == 1


def test_curate_findings_passes_no_finding_sentinel_unchanged(curate_sft: ModuleType) -> None:
    """The no_finding sentinel with a null box passes through unchanged."""
    findings = [_finding(NO_FINDING, None)]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    assert curated == [_finding(NO_FINDING, None)]


def test_curate_findings_passes_null_box_finding_unchanged(curate_sft: ModuleType) -> None:
    """A positive finding with a null box is preserved (no box to merge)."""
    findings = [_finding("cardiomegaly", None)]
    curated = curate_sft.curate_findings(findings, iou_merge_threshold=0.5)
    assert curated == [_finding("cardiomegaly", None)]


def test_curate_record_rewrites_assistant_target(curate_sft: ModuleType) -> None:
    """Curating a full record collapses duplicate boxes in its assistant target."""
    record = _record(
        "img1",
        [
            _finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7)),
            _finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7)),
        ],
    )
    curated = curate_sft.curate_record(record, iou_merge_threshold=0.5)
    target = json.loads(curated["messages"][1]["content"][0]["text"])
    assert target == [{"label": "cardiomegaly", "box_2d": [0.6, 0.3, 0.9, 0.7]}]
    # The curated record is still schema-valid.
    validate_sft_record(curated)


# --- record classification ----------------------------------------------------------


def test_is_normal_record_detects_no_finding_only(curate_sft: ModuleType) -> None:
    """A no_finding-only record is classified normal; a positive one is not."""
    assert curate_sft.is_normal_record(_normal_record("n1")) is True
    positive = _record("p1", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))])
    assert curate_sft.is_normal_record(positive) is False


def test_record_present_findings_returns_positive_label_set(curate_sft: ModuleType) -> None:
    """The present-finding set excludes the no_finding sentinel."""
    positive = _record(
        "p1",
        [
            _finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7)),
            _finding("pleural_effusion", (0.6, 0.1, 0.9, 0.4)),
        ],
    )
    assert curate_sft.record_present_findings(positive) == {"cardiomegaly", "pleural_effusion"}
    assert curate_sft.record_present_findings(_normal_record("n1")) == set()


# --- class balancing ----------------------------------------------------------------


def test_balance_keeps_all_positive_records(curate_sft: ModuleType) -> None:
    """Every positive record survives balancing regardless of the ratio."""
    positives = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(5)]
    normals = [_normal_record(f"n{i}") for i in range(50)]
    balanced = curate_sft.balance_records(positives + normals, normal_to_positive_ratio=1.0, seed=7)
    kept_positive = [record for record in balanced if not curate_sft.is_normal_record(record)]
    assert len(kept_positive) == 5


def test_balance_downsamples_normals_to_ratio(curate_sft: ModuleType) -> None:
    """Normals are downsampled to ratio * positives at a 1:1 default."""
    positives = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(5)]
    normals = [_normal_record(f"n{i}") for i in range(50)]
    balanced = curate_sft.balance_records(positives + normals, normal_to_positive_ratio=1.0, seed=7)
    kept_normal = [record for record in balanced if curate_sft.is_normal_record(record)]
    assert len(kept_normal) == 5


def test_balance_keeps_all_pneumothorax_records(curate_sft: ModuleType) -> None:
    """Pneumothorax records are always retained (rarest urgent finding)."""
    pneumo = [_record(f"x{i}", [_finding("pneumothorax", (0.1, 0.1, 0.3, 0.3))]) for i in range(3)]
    other_positives = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(2)]
    normals = [_normal_record(f"n{i}") for i in range(40)]
    balanced = curate_sft.balance_records(pneumo + other_positives + normals, normal_to_positive_ratio=0.5, seed=3)
    kept_pneumo = [record for record in balanced if "pneumothorax" in curate_sft.record_present_findings(record)]
    assert len(kept_pneumo) == 3


def test_balance_is_deterministic_for_a_fixed_seed(curate_sft: ModuleType) -> None:
    """The same seed yields the same downsampled normal selection."""
    positives = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(4)]
    normals = [_normal_record(f"n{i}") for i in range(40)]
    pool = positives + normals
    first = curate_sft.balance_records(pool, normal_to_positive_ratio=1.0, seed=11)
    second = curate_sft.balance_records(pool, normal_to_positive_ratio=1.0, seed=11)
    assert [record["image_path"] for record in first] == [record["image_path"] for record in second]


def test_balance_ratio_below_available_normals_keeps_fewer(curate_sft: ModuleType) -> None:
    """A ratio that asks for more normals than exist keeps all available normals."""
    positives = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(10)]
    normals = [_normal_record(f"n{i}") for i in range(3)]
    balanced = curate_sft.balance_records(positives + normals, normal_to_positive_ratio=1.0, seed=5)
    kept_normal = [record for record in balanced if curate_sft.is_normal_record(record)]
    assert len(kept_normal) == 3


# --- stratified split ---------------------------------------------------------------


def test_split_holds_out_val_fraction(curate_sft: ModuleType) -> None:
    """The validation split is approximately the requested fraction."""
    records = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(20)]
    train, val = curate_sft.stratified_split(records, val_fraction=0.2, seed=9)
    assert len(train) + len(val) == 20
    assert len(val) == 4


def test_split_is_disjoint_and_complete(curate_sft: ModuleType) -> None:
    """Train and val partition the input with no overlap and no loss."""
    records = [_record(f"p{i}", [_finding("nodule_mass", (0.1, 0.1, 0.2, 0.2))]) for i in range(15)]
    records += [_normal_record(f"n{i}") for i in range(15)]
    train, val = curate_sft.stratified_split(records, val_fraction=0.1, seed=2)
    train_paths = {record["image_path"] for record in train}
    val_paths = {record["image_path"] for record in val}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths | val_paths == {record["image_path"] for record in records}


def test_split_preserves_finding_presence_distribution(curate_sft: ModuleType) -> None:
    """Each finding stratum contributes proportionally to the validation split."""
    cardio = [_record(f"c{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(10)]
    effusion = [_record(f"e{i}", [_finding("pleural_effusion", (0.6, 0.1, 0.9, 0.4))]) for i in range(10)]
    normals = [_normal_record(f"n{i}") for i in range(10)]
    train, val = curate_sft.stratified_split(cardio + effusion + normals, val_fraction=0.2, seed=4)
    val_cardio = sum(1 for record in val if "cardiomegaly" in curate_sft.record_present_findings(record))
    val_effusion = sum(1 for record in val if "pleural_effusion" in curate_sft.record_present_findings(record))
    val_normal = sum(1 for record in val if curate_sft.is_normal_record(record))
    # Each stratum of 10 contributes 2 to a 20% validation split.
    assert val_cardio == 2
    assert val_effusion == 2
    assert val_normal == 2


def test_split_is_deterministic_for_a_fixed_seed(curate_sft: ModuleType) -> None:
    """The same seed yields the same train/val partition."""
    records = [_record(f"p{i}", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]) for i in range(20)]
    train_a, val_a = curate_sft.stratified_split(records, val_fraction=0.25, seed=13)
    train_b, val_b = curate_sft.stratified_split(records, val_fraction=0.25, seed=13)
    assert [r["image_path"] for r in val_a] == [r["image_path"] for r in val_b]
    assert [r["image_path"] for r in train_a] == [r["image_path"] for r in train_b]


# --- distribution summary -----------------------------------------------------------


def test_summarize_distribution_counts_normals_positives_and_labels(curate_sft: ModuleType) -> None:
    """The summary reports totals, the normal/positive split, and per-label counts."""
    records = [
        _record("p1", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))]),
        _record("p2", [_finding("pneumothorax", (0.1, 0.1, 0.3, 0.3))]),
        _normal_record("n1"),
    ]
    summary = curate_sft.summarize_distribution(records)
    assert summary.total == 3
    assert summary.normal == 1
    assert summary.positive == 2
    assert summary.per_label["cardiomegaly"] == 1
    assert summary.per_label["pneumothorax"] == 1
    assert summary.per_label[NO_FINDING] == 1


# --- end-to-end CLI -----------------------------------------------------------------


def _write_corpus(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file, one per line."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def test_cli_dry_run_writes_nothing(curate_sft: ModuleType, tmp_path: Path) -> None:
    """A dry run prints the plan and never writes the curated outputs."""
    corpus = tmp_path / "train.jsonl"
    _write_corpus(corpus, [_record("p1", [_finding("cardiomegaly", (0.6, 0.3, 0.9, 0.7))])])
    out_dir = tmp_path / "out"
    exit_code = curate_sft.main(["--input", str(corpus), "--out-dir", str(out_dir), "--dry-run"])
    assert exit_code == 0
    assert not (out_dir / "train.curated.jsonl").exists()


def test_cli_missing_input_errors(curate_sft: ModuleType, tmp_path: Path) -> None:
    """A missing input corpus reports a clear error and a nonzero exit code."""
    exit_code = curate_sft.main(["--input", str(tmp_path / "absent.jsonl"), "--out-dir", str(tmp_path / "out")])
    assert exit_code == 1


def test_cli_end_to_end_writes_curated_and_val(curate_sft: ModuleType, tmp_path: Path) -> None:
    """The full CLI path dedups boxes, balances, splits, and writes valid JSONL."""
    records: list[dict[str, Any]] = []
    # Positive cardiomegaly records, each with a triplicated near-duplicate box.
    for i in range(10):
        records.append(
            _record(
                f"p{i}",
                [
                    _finding("cardiomegaly", (0.60, 0.30, 0.90, 0.70)),
                    _finding("cardiomegaly", (0.61, 0.31, 0.91, 0.71)),
                    _finding("cardiomegaly", (0.60, 0.30, 0.90, 0.70)),
                ],
            )
        )
    # A heavy excess of normal studies that balancing must downsample.
    for i in range(60):
        records.append(_normal_record(f"n{i}"))

    corpus = tmp_path / "train.jsonl"
    _write_corpus(corpus, records)
    out_dir = tmp_path / "out"
    exit_code = curate_sft.main(
        [
            "--input",
            str(corpus),
            "--out-dir",
            str(out_dir),
            "--val-fraction",
            "0.2",
            "--normal-to-positive-ratio",
            "1.0",
            "--iou-merge-threshold",
            "0.5",
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0

    train_path = out_dir / "train.curated.jsonl"
    val_path = out_dir / "val.curated.jsonl"
    assert train_path.is_file()
    assert val_path.is_file()

    train_records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_records = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Balancing keeps all 10 positives and downsamples 60 normals to 10 (1:1),
    # for 20 curated records total, split 80/20 into 16 train and 4 val.
    total = len(train_records) + len(val_records)
    assert total == 20
    assert len(val_records) == 4

    # Every curated record is schema-valid and its boxes are deduped.
    for record in train_records + val_records:
        validate_sft_record(record)
        target = json.loads(record["messages"][1]["content"][0]["text"])
        if target[0]["label"] != NO_FINDING:
            # The triplicated cardiomegaly collapsed to a single box.
            assert len(target) == 1
