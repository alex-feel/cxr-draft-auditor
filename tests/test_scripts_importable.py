"""Importability and pure-logic smoke tests for the user-run helper scripts.

The scripts under ``scripts/`` are user-run CLIs that perform downloads
(``download_data.py``) or build the SFT corpus (``prepare_sft.py``). They are not
part of the installed ``cxr_auditor`` package and are not on ``pythonpath``, so
they are loaded here directly from their file paths. The primary purpose is to
catch syntax errors and import-time failures (importing must not require the heavy
``kaggle`` / ``huggingface_hub`` / ``datasets`` libraries, nor any network), and
to exercise the dependency-light CLI logic with the in-repo synthetic fixtures so
no real dataset, credential, or GPU is involved.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(module_name: str) -> ModuleType:
    """Load a script module from ``scripts/`` by file path.

    Loading by path rather than by package import keeps the scripts out of the
    installed package while still importing their real source, so any syntax or
    import-time error surfaces as a test failure.
    """
    path = SCRIPTS_DIR / f"{module_name}.py"
    qualified_name = f"_scripts_{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before executing so module-level introspection (for example the
    # dataclasses machinery resolving annotations via the module namespace) can
    # find the module by its qualified name.
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def download_data() -> ModuleType:
    """The imported ``scripts/download_data.py`` module."""
    return _load_script("download_data")


@pytest.fixture(scope="module")
def prepare_sft() -> ModuleType:
    """The imported ``scripts/prepare_sft.py`` module."""
    return _load_script("prepare_sft")


def test_download_data_imports(download_data: ModuleType) -> None:
    """Importing the download script exposes its CLI surface without heavy deps."""
    assert callable(download_data.main)
    assert callable(download_data.build_arg_parser)
    assert len(download_data.DATASET_SPECS) == 5


def test_prepare_sft_imports(prepare_sft: ModuleType) -> None:
    """Importing the SFT-prep script exposes its CLI surface without heavy deps."""
    assert callable(prepare_sft.main)
    assert callable(prepare_sft.build_arg_parser)
    assert callable(prepare_sft.build_grouped_findings)


def test_resolve_specs_all_and_default(download_data: ModuleType) -> None:
    """The ``all`` selector and an empty selection both expand to every dataset."""
    every = download_data.resolve_specs([download_data.ALL_SELECTOR])
    default = download_data.resolve_specs([])
    assert [spec.key for spec in every] == [spec.key for spec in download_data.DATASET_SPECS]
    assert [spec.key for spec in default] == [spec.key for spec in every]


def test_resolve_specs_subset_in_registry_order(download_data: ModuleType) -> None:
    """A subset is returned in registry order regardless of the listed order."""
    specs = download_data.resolve_specs(["chestxdet", "vqa"])
    assert [spec.key for spec in specs] == ["vqa", "chestxdet"]


def test_resolve_specs_rejects_unknown(download_data: ModuleType) -> None:
    """An unknown dataset selector is rejected with a helpful message."""
    with pytest.raises(ValueError, match="unknown dataset"):
        download_data.resolve_specs(["not_a_dataset"])


def test_format_plan_lists_destinations(download_data: ModuleType, tmp_path: Path) -> None:
    """The plan names each selected dataset and its computed destination path."""
    specs = download_data.resolve_specs(["vqa"])
    plan = download_data.format_plan(specs, tmp_path)
    assert "VinDr-CXR-VQA" in plan
    assert str(tmp_path / "vqa") in plan
    assert "LICENSE AND CREDENTIAL PREREQUISITES" in plan


def test_download_data_dry_run_no_network(download_data: ModuleType, tmp_path: Path) -> None:
    """A dry run prints the plan, writes nothing, and returns success."""
    exit_code = download_data.main(["--dry-run", "--data-dir", str(tmp_path), "vqa"])
    assert exit_code == 0
    # Dry run must not create the destination tree.
    assert not (tmp_path / "vqa").exists()


def test_prepare_sft_dry_run_no_inputs(prepare_sft: ModuleType, tmp_path: Path) -> None:
    """A dry run prints the plan and never touches the (empty) data tree."""
    exit_code = prepare_sft.main(["--dry-run", "--data-dir", str(tmp_path)])
    assert exit_code == 0
    assert not (tmp_path / "sft").exists()


def test_prepare_sft_missing_box_csv_errors(prepare_sft: ModuleType, tmp_path: Path) -> None:
    """A real run with no VinDr box CSV reports a clear error and a nonzero code."""
    exit_code = prepare_sft.main(["--data-dir", str(tmp_path)])
    assert exit_code == 1


def _write_vindr_tree(data_dir: Path, box_csv_text: str) -> None:
    """Lay out a tiny VinDr mirror tree with a box CSV and a meta CSV.

    The meta CSV gives each image a square original dimension so the box loader
    rescales boxes to the canonical normalized format.
    """
    mirror = data_dir / "vindr" / "vinbigdata-512-image-dataset"
    mirror.mkdir(parents=True)
    (mirror / "train.csv").write_text(box_csv_text, encoding="utf-8")

    image_ids = sorted({line.split(",", 1)[0] for line in box_csv_text.splitlines()[1:] if line.strip()})
    meta_lines = ["image_id,width,height"]
    meta_lines.extend(f"{image_id},3000,3000" for image_id in image_ids)
    (mirror / "train_meta.csv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")


def test_build_grouped_findings_from_fixture_csv(
    prepare_sft: ModuleType,
    tmp_path: Path,
    vindr_boxes_csv_text: str,
) -> None:
    """The box CSV is parsed and grouped into canonical per-image findings."""
    _write_vindr_tree(tmp_path, vindr_boxes_csv_text)
    grouped = prepare_sft.build_grouped_findings(
        tmp_path,
        box_csv=None,
        meta_csv=None,
        vqa_json=None,
    )
    # Every image_id from the fixture CSV appears, including the normal-study row.
    assert "0a1b2c3d4e5f60718293a4b5c6d7e8f9" in grouped
    assert "1f2e3d4c5b6a70819a2b3c4d5e6f7081" in grouped
    # The pneumothorax image carries a canonical pneumothorax finding with a box.
    pneumo = grouped["1f2e3d4c5b6a70819a2b3c4d5e6f7081"]
    findings = {finding.finding for finding in pneumo}
    assert "pneumothorax" in findings


def test_write_corpus_emits_valid_jsonl(
    prepare_sft: ModuleType,
    tmp_path: Path,
    vindr_boxes_csv_text: str,
) -> None:
    """The end-to-end pure-logic path writes a non-empty, valid SFT JSONL."""
    _write_vindr_tree(tmp_path, vindr_boxes_csv_text)
    output = tmp_path / "sft" / "train.jsonl"
    exit_code = prepare_sft.main(["--data-dir", str(tmp_path), "--output", str(output)])
    assert exit_code == 0
    assert output.is_file()

    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "expected at least one SFT record"
    for line in lines:
        record = json.loads(line)
        assert record["messages"][0]["role"] == "user"
        assert record["messages"][1]["role"] == "assistant"
        # The assistant target is a JSON list of canonical finding objects.
        target = record["messages"][1]["content"][0]["text"]
        parsed = json.loads(target)
        assert isinstance(parsed, list) and parsed


def test_record_to_image_findings_drops_native_label(prepare_sft: ModuleType) -> None:
    """Box records convert to ImageFinding objects carrying canonical labels."""
    from cxr_auditor.data.records import BoxRecord, ImageBoxRecord

    record = ImageBoxRecord(
        image_id="abc",
        boxes=[BoxRecord(finding="pneumothorax", box=(0.1, 0.2, 0.3, 0.4), native_label="Pneumothorax")],
    )
    findings = prepare_sft.record_to_image_findings(record)
    assert len(findings) == 1
    assert findings[0].finding == "pneumothorax"
    assert findings[0].box == (0.1, 0.2, 0.3, 0.4)
