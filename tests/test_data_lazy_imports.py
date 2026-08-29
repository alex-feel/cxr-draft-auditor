"""Guard that the data subpackage imports without heavy/network dependencies.

The pure parsing path must never pull torch, transformers, datasets, or pydicom
onto the import path. These tests import the subpackage and every module in it and
assert that the optional heavy modules are not imported as a side effect.
"""

from __future__ import annotations

import importlib
import sys


def test_data_package_imports_without_heavy_modules() -> None:
    # Drop any pre-imported heavy modules so this test observes the import side
    # effects of the data package in isolation.
    heavy = ("torch", "transformers", "datasets", "pydicom", "unsloth", "gradio")
    saved = {name: sys.modules.pop(name, None) for name in heavy}
    try:
        for module in (
            "cxr_auditor.data",
            "cxr_auditor.data.vindr",
            "cxr_auditor.data.vqa_join",
            "cxr_auditor.data.nih_bbox",
            "cxr_auditor.data.chestxdet",
            "cxr_auditor.data.dicom",
            "cxr_auditor.data.iu_xray",
            "cxr_auditor.data.records",
        ):
            importlib.import_module(module)
        for name in ("torch", "transformers", "datasets", "pydicom", "unsloth", "gradio"):
            assert name not in sys.modules, f"{name} was imported at module scope"
    finally:
        for name, value in saved.items():
            if value is not None:
                sys.modules[name] = value


def test_public_api_symbols_exported() -> None:
    import cxr_auditor.data as data

    for symbol in (
        "parse_vindr_boxes_csv",
        "load_vindr_dims_csv",
        "parse_vqa_rows",
        "parse_nih_bbox_csv",
        "parse_chestxdet_example",
        "parse_iu_xray_row",
        "ImageBoxRecord",
        "VqaBoxRecord",
        "ReportRecord",
        "BoxRecord",
    ):
        assert hasattr(data, symbol), f"missing public symbol {symbol}"
