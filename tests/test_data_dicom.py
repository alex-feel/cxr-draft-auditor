"""Tests for the DICOM windowing math and DICOM-to-PNG conversion.

The windowing math is exercised directly. The file-conversion paths
(``dicom_to_array`` / ``dicom_to_png`` / ``dicom_to_pil``) are exercised with a
fake ``pydicom`` module injected via the lazy import so the real I/O wiring runs
end to end through Pillow (a core dependency) without requiring ``pydicom`` or any
DICOM file.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cxr_auditor.data import dicom as dicom_module
from cxr_auditor.data.dicom import apply_window, dicom_to_array, dicom_to_png, dicom_to_pil


class _FakeDataset:
    """Minimal stand-in for a pydicom FileDataset used by the conversion path."""

    def __init__(self, pixel_array: np.ndarray, photometric: str = "MONOCHROME2") -> None:
        self.pixel_array = pixel_array
        self.PhotometricInterpretation = photometric
        self.WindowCenter = 500.0
        self.WindowWidth = 400.0


class _FakePydicom:
    """Stand-in for the ``pydicom`` module exposing ``dcmread``."""

    def __init__(self, dataset: _FakeDataset) -> None:
        self._dataset = dataset

    def dcmread(self, _path: str) -> _FakeDataset:
        return self._dataset


@pytest.fixture
def patch_pydicom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the lazy ``importlib.import_module('pydicom')`` to a fake module."""
    pixels = np.array([[100, 300, 500, 700, 900]], dtype=np.uint16)
    fake = _FakePydicom(_FakeDataset(pixels))
    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None) -> object:
        if name == "pydicom":
            return fake
        return real_import(name, package)

    monkeypatch.setattr(dicom_module.importlib, "import_module", _fake_import)


def test_explicit_window_maps_center_and_edges() -> None:
    # Window center 500, width 400 -> [300, 700] maps to [0, 255].
    pixels = np.array([[100, 300, 500, 700, 900]], dtype=np.uint16)
    out = apply_window(pixels, center=500.0, width=400.0)
    assert out[0, 0] == 0  # below window -> black
    assert out[0, 1] == 0  # at low edge
    assert out[0, 2] == 128  # center -> mid gray (round(0.5*255))
    assert out[0, 3] == 255  # at high edge
    assert out[0, 4] == 255  # above window -> white


def test_invert_for_monochrome1() -> None:
    pixels = np.array([[300, 500, 700]], dtype=np.uint16)
    out = apply_window(pixels, center=500.0, width=400.0, invert=True)
    # Inversion flips black and white relative to the non-inverted mapping.
    assert out[0, 0] == 255
    assert out[0, 2] == 0


def test_percentile_window_when_no_explicit_values() -> None:
    pixels = np.arange(0, 1000, dtype=np.uint16).reshape(1, 1000)
    out = apply_window(pixels, center=None, width=None, low_percentile=0.0, high_percentile=100.0)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_constant_image_returns_black_not_nan() -> None:
    pixels = np.full((4, 4), 700, dtype=np.uint16)
    out = apply_window(pixels, center=None, width=None)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_output_shape_and_dtype_preserved() -> None:
    pixels = np.random.default_rng(0).integers(0, 4096, size=(8, 8), dtype=np.uint16)
    out = apply_window(pixels)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_zero_width_falls_back_to_percentile() -> None:
    # A non-positive width is ignored in favor of the percentile window.
    pixels = np.array([[0, 128, 255]], dtype=np.uint16)
    out = apply_window(pixels, center=100.0, width=0.0, low_percentile=0.0, high_percentile=100.0)
    assert out[0, 0] == 0
    assert out[0, 2] == 255


def test_dicom_to_array_applies_embedded_window(patch_pydicom: None) -> None:
    array = dicom_to_array("ignored.dcm")
    # The fake dataset carries window center 500, width 400 over [100..900].
    assert array.dtype == np.uint8
    assert array[0, 0] == 0
    assert array[0, 3] == 255


def test_dicom_to_pil_returns_grayscale_image(patch_pydicom: None) -> None:
    image = dicom_to_pil("ignored.dcm")
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (5, 1)


def test_dicom_to_png_writes_file(patch_pydicom: None, tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "out.png"
    written = dicom_to_png("ignored.dcm", destination)
    assert written == destination
    assert destination.exists()
    with Image.open(destination) as reopened:
        assert reopened.mode == "L"
        assert reopened.size == (5, 1)
