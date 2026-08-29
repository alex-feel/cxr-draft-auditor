"""Regression guard: the SFT producer's image-path key matches the trainer's reader.

``sft_dataset.build_sft_record`` writes the image path under the ``image_path`` key;
the training entry point ``train/train_medgemma_lora.py`` reads it back in
``load_sft_dataset``. A divergence here (the historical ``image`` vs ``image_path``
mismatch) breaks training on the first JSONL line while every pure-logic test stays
green, because the train script is not import-covered by the package test suite.
This test runs the producer's output through the real consumer (with the optional
``datasets`` backend stubbed) so the two contracts can never silently drift apart.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from cxr_auditor.schema import FindingStatus, ImageFinding
from cxr_auditor.sft_dataset import build_sft_record, validate_sft_record

_TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "train" / "train_medgemma_lora.py"


def _load_train_module() -> types.ModuleType:
    """Import the training script by file path (it is a script, not an installed package)."""
    spec = importlib.util.spec_from_file_location("train_medgemma_lora", _TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sft_record_round_trips_through_trainer(
    tmp_path: Path,
    tiny_png_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record from build_sft_record loads back through the trainer's load_sft_dataset."""
    record = build_sft_record(
        tiny_png_path,
        [ImageFinding(finding="pleural_effusion", status=FindingStatus.PRESENT, box=(0.6, 0.1, 0.9, 0.4))],
    )
    validate_sft_record(record)  # producer is self-consistent

    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    # Stub the optional 'datasets' backend so the consumer runs without the train extra.
    fake_datasets = types.ModuleType("datasets")

    class _Dataset:
        @staticmethod
        def from_list(rows: list[object]) -> list[object]:
            return rows

    setattr(fake_datasets, "Dataset", _Dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    train = _load_train_module()
    rows = train.load_sft_dataset(jsonl, tiny_png_path.parent)

    # If the consumer read the wrong key, load_sft_dataset would have raised
    # ValueError("missing 'image_path'") before reaching here.
    assert len(rows) == 1
    assert set(rows[0]) == {"images", "messages"}


class _FakeImage:
    """A minimal stand-in for a PIL image exposing only ``convert``."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def convert(self, mode: str) -> "_FakeImage":
        return _FakeImage(mode)


class _FakeHubDataset:
    """A minimal stand-in for a loaded ``datasets.Dataset`` with column metadata and map."""

    def __init__(self, rows: list[dict[str, object]], column_names: list[str]) -> None:
        self._rows = rows
        self.column_names = column_names

    def map(self, fn: Callable[[dict[str, object]], dict[str, object]]) -> list[dict[str, object]]:
        return [fn(row) for row in self._rows]


def _install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch,
    dataset: _FakeHubDataset,
) -> dict[str, object]:
    """Stub the 'datasets' module so load_dataset returns the given fake dataset."""
    calls: dict[str, object] = {}

    def _load_dataset(repo_id: str, split: str) -> _FakeHubDataset:
        calls["repo_id"] = repo_id
        calls["split"] = split
        return dataset

    fake_datasets = types.ModuleType("datasets")
    setattr(fake_datasets, "load_dataset", _load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return calls


def test_load_hub_dataset_converts_images_to_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Hub-dataset path loads the split and converts each embedded image to RGB."""
    rows: list[dict[str, object]] = [{"images": [_FakeImage("L")], "messages": [{"role": "user", "content": []}]}]
    dataset = _FakeHubDataset(rows, ["images", "messages"])
    calls = _install_fake_datasets(monkeypatch, dataset)

    train = _load_train_module()
    result = train.load_hub_dataset("me/cxr-sft", "train")

    assert calls == {"repo_id": "me/cxr-sft", "split": "train"}
    assert len(result) == 1
    assert result[0]["images"][0].mode == "RGB"
    assert result[0]["messages"] == rows[0]["messages"]


def test_load_hub_dataset_rejects_missing_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub dataset lacking the required columns is rejected with a clear error."""
    dataset = _FakeHubDataset([], ["image", "text"])
    _install_fake_datasets(monkeypatch, dataset)

    train = _load_train_module()
    with pytest.raises(ValueError, match="must have 'images' and 'messages'"):
        train.load_hub_dataset("me/cxr-sft", "train")
