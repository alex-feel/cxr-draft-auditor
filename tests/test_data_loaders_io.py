"""I/O wiring tests for the file- and dataset-backed loader entry points.

These exercise the thin functions that read a CSV/JSON from disk, load a mirror
PNG, or call ``datasets.load_dataset``. The dataset path is driven by a fake
``datasets`` module injected via the lazy import, so no network, credentials, or
heavy dependency is required. File reading uses the in-repo synthetic fixtures and
a generated tiny PNG.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from cxr_auditor.data import chestxdet as chestxdet_module
from cxr_auditor.data import iu_xray as iu_xray_module
from cxr_auditor.data.chestxdet import load_chestxdet
from cxr_auditor.data.iu_xray import load_iu_xray
from cxr_auditor.data.vindr import load_vindr_boxes, load_vindr_dims_csv, load_vindr_image


class _FakeDatasets:
    """Stand-in for the ``datasets`` module exposing ``load_dataset``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def load_dataset(self, _repo_id: str, split: str = "train") -> list[dict[str, Any]]:
        return self._rows


def _patch_datasets(monkeypatch: pytest.MonkeyPatch, module: Any, rows: list[dict[str, Any]]) -> None:
    fake = _FakeDatasets(rows)
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None) -> object:
        if name == "datasets":
            return fake
        return real_import(name, package)

    monkeypatch.setattr(module.importlib, "import_module", _fake_import)


def test_load_vindr_boxes_reads_file(tmp_path: Path, vindr_boxes_csv_text: str, vindr_meta_csv_text: str) -> None:
    csv_path = tmp_path / "boxes.csv"
    csv_path.write_text(vindr_boxes_csv_text, encoding="utf-8")
    dims = load_vindr_dims_csv(vindr_meta_csv_text)
    records = load_vindr_boxes(csv_path, dims)
    assert any(record.image_id == "0a1b2c3d4e5f60718293a4b5c6d7e8f9" for record in records)


def test_load_vindr_image_returns_image_and_dims(tmp_path: Path, tiny_png_path: Path) -> None:
    image_id = "abc"
    target = tmp_path / f"{image_id}.png"
    target.write_bytes(tiny_png_path.read_bytes())
    image, width, height = load_vindr_image(tmp_path, image_id)
    assert (width, height) == (16, 16)
    assert image.mode == "RGB"


def test_load_vindr_image_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no VinDr mirror image"):
        load_vindr_image(tmp_path, "does_not_exist")


def test_load_chestxdet_via_fake_datasets(monkeypatch: pytest.MonkeyPatch, chestxdet_rows: list[dict[str, Any]]) -> None:
    _patch_datasets(monkeypatch, chestxdet_module, chestxdet_rows)
    records = load_chestxdet(split="train")
    assert {record.image_id for record in records} == {"cxd_0001.png", "cxd_0002.png"}


def test_load_iu_xray_via_fake_datasets(monkeypatch: pytest.MonkeyPatch, iu_xray_rows: list[dict[str, Any]]) -> None:
    _patch_datasets(monkeypatch, iu_xray_module, iu_xray_rows)
    results = load_iu_xray(split="train")
    assert len(results) == 3
    image, findings_text, impression_text = results[0]
    assert image is None
    assert "lungs are clear" in findings_text.lower()
    assert impression_text == "No acute cardiopulmonary abnormality."
