"""
Shared pytest fixtures backed by tiny synthetic in-repo fixtures.

Every fixture here is fabricated. There is no real chest X-ray, no real
radiology report, no real dataset row, and no network access. The fixtures exist
so the pure-logic modules (findings, schema, prompts) can be exercised end to end
without any GPU, torch, transformers, or gradio stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def tiny_png_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return the path to a 16x16 synthetic PNG.

    The image is a plain grayscale gradient. It carries no clinical meaning; it
    exists only so code paths that need a real on-disk PNG (image loading,
    width/height probing) have one to operate on.
    """
    path = tmp_path_factory.mktemp("images") / "tiny_cxr.png"
    image = Image.new("L", (16, 16))
    image.putdata([(x * 16 + y) % 256 for y in range(16) for x in range(16)])
    image.save(path, format="PNG")
    return path


@pytest.fixture
def tiny_image(tiny_png_path: Path) -> Image.Image:
    """Open the synthetic 16x16 PNG as a PIL image (RGB)."""
    with Image.open(tiny_png_path) as image:
        return image.convert("RGB")


@pytest.fixture(scope="session")
def sample_model_outputs() -> dict[str, Any]:
    """Load the static sample model-output fixtures as a dict."""
    with (FIXTURES_DIR / "sample_model_outputs.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def vqa_sample_rows() -> list[dict[str, Any]]:
    """Load the tiny synthetic VinDr-CXR-VQA data_v1.json snippet as a list."""
    with (FIXTURES_DIR / "vqa_data_v1_sample.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def vindr_boxes_csv_text() -> str:
    """Return the tiny synthetic VinDr-style boxes CSV as raw text."""
    return (FIXTURES_DIR / "vindr_boxes_sample.csv").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def vindr_meta_csv_text() -> str:
    """Return the tiny synthetic VinDr metadata CSV (dim0=height, dim1=width)."""
    return (FIXTURES_DIR / "vindr_meta_sample.csv").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def nih_bbox_csv_text() -> str:
    """Return the tiny synthetic NIH BBox_List_2017-style CSV as raw text."""
    return (FIXTURES_DIR / "nih_bbox_sample.csv").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def chestxdet_rows() -> list[dict[str, Any]]:
    """Return tiny synthetic ChestX-Det mirror rows (integer-index ``label`` maps).

    Each row carries an ``image_id`` and a small 2-D ``label`` grid of category ids
    (matching the real natealberti/ChestX-Det ``label`` column, where pixel value
    is the disease category id and ``255`` is background).
    """
    with (FIXTURES_DIR / "chestxdet_rows_sample.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def iu_xray_rows() -> list[dict[str, Any]]:
    """Return tiny synthetic Open-i mirror rows (real-report shape, no boxes)."""
    with (FIXTURES_DIR / "iu_xray_rows_sample.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def grounded_finding_dicts(sample_model_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """A small list of MedGemma-style grounded finding dicts ({label, box_2d, ...})."""
    return sample_model_outputs["image_grounding_list"]


@pytest.fixture
def full_audit_result_dict(sample_model_outputs: dict[str, Any]) -> dict[str, Any]:
    """A complete canonical AuditResult-shaped dict for schema validation tests."""
    return sample_model_outputs["full_audit_result"]
