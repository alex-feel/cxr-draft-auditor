#!/usr/bin/env python3
"""Fetch the CXR Draft Auditor datasets into the local ``data/`` tree.

This is a USER-RUN helper. It performs network access and assumes the user has
already accepted each dataset's license and configured the relevant credentials
(a Kaggle API token for the VinDr mirror, a Hugging Face token for any gated or
rate-limited Hub dataset). Nothing here accepts a license on the user's behalf or
stores credentials; see ``SETUP.md`` for the exact, prerequisite user actions.

Datasets and their on-disk layout (all under ``--data-dir``, default ``data/``)::

    data/
      vindr/
        vinbigdata-512-image-dataset/   # awsaf49/vinbigdata-512-image-dataset (Kaggle)
      vqa/
        data_v1.json                    # faizan711/VinDR-CXR-VQA (Hugging Face)
      chestxdet/                        # natealberti/ChestX-Det parquet (Hugging Face)
      nih/                              # alkzar90/NIH-Chest-X-ray-dataset + BBox_List_2017.csv
      open_i/                           # ykumards/open-i (Hugging Face, real reports)

This layout is what the loaders in :mod:`cxr_auditor.data` expect: the VinDr box
CSV and ``*_meta.csv`` plus the resized PNG mirror live under ``vindr/``; the VQA
``data_v1.json`` under ``vqa/``; the ChestX-Det parquet under ``chestxdet/``; the
NIH ``BBox_List_2017.csv`` under ``nih/``; the Open-i reports under ``open_i/``.

Usage::

    python scripts/download_data.py --dry-run                 # print the plan only
    python scripts/download_data.py                           # fetch every dataset
    python scripts/download_data.py vqa chestxdet             # fetch a subset
    python scripts/download_data.py --data-dir /mnt/data all  # custom destination

Design notes:

- The heavy / network libraries (``kaggle``, ``huggingface_hub``, ``datasets``)
  are imported lazily inside the per-dataset functions, so importing this module
  -- as the importability smoke test does -- never requires those packages,
  network access, or credentials.
- ``--dry-run`` prints exactly what each selected dataset would fetch and where it
  would land, without importing any heavy library or touching the network.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# License-acceptance and credential notes surfaced before any fetch. These are
# the user's responsibility; this script never performs them.
_LICENSE_NOTES: tuple[str, ...] = (
    "LICENSE AND CREDENTIAL PREREQUISITES (the user must complete these first):",
    "- VinDr mirror (Kaggle 'awsaf49/vinbigdata-512-image-dataset'): non-commercial",
    "  research use under the VinDr DUA (NOT CC0). A Kaggle API token must be placed",
    "  at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).",
    "- VinDr-CXR-VQA (Hugging Face 'faizan711/VinDR-CXR-VQA'): CC BY 4.0 annotations,",
    "  no images. Joins to VinDr pixels by the 32-char hex image_id.",
    "- ChestX-Det (Hugging Face 'natealberti/ChestX-Det'): Apache-2.0 annotations.",
    "- NIH ChestX-ray14 (Hugging Face 'alkzar90/NIH-Chest-X-ray-dataset' +",
    "  BBox_List_2017.csv): held-out box evaluation; boxes are absolute XYWH at 1024px.",
    "- Open-i / IU-Xray (Hugging Face 'ykumards/open-i'): CC BY-NC-ND real reports,",
    "  used only to validate the draft parser; no boxes.",
    "- A Hugging Face token (HF_TOKEN env var or 'hf auth login') is recommended for",
    "  Hub downloads and required for any dataset that becomes gated.",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing one fetchable dataset and where it lands on disk.

    Attributes:
        key: The short command-line selector (for example ``"vqa"``).
        title: A human-readable name for messages.
        source: Where the data comes from (for example ``"Kaggle"`` or
            ``"Hugging Face"``), shown in the plan.
        repo_id: The Kaggle slug or Hugging Face repo id.
        subdir: The destination subdirectory under the data root.
        license_note: A one-line license summary shown in the plan.
        fetch: The function that performs the actual download. It is invoked only
            outside ``--dry-run`` and is responsible for its own lazy imports.
    """

    key: str
    title: str
    source: str
    repo_id: str
    subdir: str
    license_note: str
    fetch: Callable[[Path], None]


def _ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_vindr_mirror(dest: Path) -> None:
    """Download the resized VinDr PNG mirror from Kaggle into ``dest``.

    Uses the Kaggle API (imported lazily). The mirror ships resized PNGs plus the
    competition CSVs; the loaders in :mod:`cxr_auditor.data.vindr` read the box CSV
    and ``*_meta.csv`` from this tree and the PNGs by ``<image_id>.png`` name.

    Args:
        dest: Destination directory for the unzipped Kaggle dataset.
    """
    kaggle = importlib.import_module("kaggle")

    _ensure_dir(dest)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "awsaf49/vinbigdata-512-image-dataset",
        path=str(dest),
        unzip=True,
        quiet=False,
    )


def fetch_vqa_annotations(dest: Path) -> None:
    """Download VinDr-CXR-VQA ``data_v1.json`` from the Hub into ``dest``.

    The dataset ships only ``data_v1.json`` (annotations, no images). The file is
    fetched with ``huggingface_hub.hf_hub_download`` (imported lazily) and copied to
    ``dest/data_v1.json`` so :func:`cxr_auditor.data.vqa_join.load_vqa_annotations`
    can read it directly.

    Args:
        dest: Destination directory for ``data_v1.json``.
    """
    hub = importlib.import_module("huggingface_hub")

    _ensure_dir(dest)
    cached = hub.hf_hub_download(
        repo_id="faizan711/VinDR-CXR-VQA",
        filename="data_v1.json",
        repo_type="dataset",
    )
    shutil.copyfile(cached, dest / "data_v1.json")


def fetch_chestxdet(dest: Path) -> None:
    """Download the ChestX-Det parquet mirror from the Hub into ``dest``.

    Snapshots the ``natealberti/ChestX-Det`` dataset repo (parquet under ``data/``
    plus ``id2label.json``) into ``dest`` with ``snapshot_download`` (imported
    lazily). :func:`cxr_auditor.data.chestxdet.load_chestxdet` loads the split via
    the ``datasets`` library from the Hub by repo id; this local snapshot makes the
    parquet available offline as well.

    Args:
        dest: Destination directory for the dataset snapshot.
    """
    hub = importlib.import_module("huggingface_hub")

    _ensure_dir(dest)
    hub.snapshot_download(
        repo_id="natealberti/ChestX-Det",
        repo_type="dataset",
        local_dir=str(dest),
    )


def fetch_nih_bbox(dest: Path) -> None:
    """Download NIH ChestX-ray14 box annotations from the Hub into ``dest``.

    Fetches ``data/BBox_List_2017.csv`` from ``alkzar90/NIH-Chest-X-ray-dataset``
    with ``hf_hub_download`` (imported lazily) so
    :func:`cxr_auditor.data.nih_bbox.load_nih_bbox_csv` can read it. The CSV is the
    held-out box-evaluation source; the boxes are absolute XYWH against the 1024px
    PNG release. The full image set is large and is not pulled here -- only the box
    CSV, which is what the box-evaluation path needs.

    Args:
        dest: Destination directory for ``BBox_List_2017.csv``.
    """
    hub = importlib.import_module("huggingface_hub")

    _ensure_dir(dest)
    # The CSV lives under the repo's ``data/`` directory, not the repo root
    # (requesting the bare filename returns a 404). The local copy keeps the
    # bare basename so the loader finds it at ``data/nih/BBox_List_2017.csv``.
    cached = hub.hf_hub_download(
        repo_id="alkzar90/NIH-Chest-X-ray-dataset",
        filename="data/BBox_List_2017.csv",
        repo_type="dataset",
    )
    shutil.copyfile(cached, dest / "BBox_List_2017.csv")


def fetch_open_i(dest: Path) -> None:
    """Download the Open-i / IU-Xray report dataset from the Hub into ``dest``.

    Snapshots ``ykumards/open-i`` (real radiology reports, no boxes) into ``dest``
    with ``snapshot_download`` (imported lazily). The reports validate the draft
    parser only; they are never used for box training.

    Args:
        dest: Destination directory for the dataset snapshot.
    """
    hub = importlib.import_module("huggingface_hub")

    _ensure_dir(dest)
    hub.snapshot_download(
        repo_id="ykumards/open-i",
        repo_type="dataset",
        local_dir=str(dest),
    )


# The dataset registry. Order is the fetch / plan order. The ``all`` selector
# expands to every key here.
DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="vindr",
        title="VinDr-CXR resized PNG mirror",
        source="Kaggle",
        repo_id="awsaf49/vinbigdata-512-image-dataset",
        subdir="vindr/vinbigdata-512-image-dataset",
        license_note="VinDr DUA, non-commercial research (NOT CC0)",
        fetch=fetch_vindr_mirror,
    ),
    DatasetSpec(
        key="vqa",
        title="VinDr-CXR-VQA annotations (data_v1.json, no images)",
        source="Hugging Face",
        repo_id="faizan711/VinDR-CXR-VQA",
        subdir="vqa",
        license_note="CC BY 4.0",
        fetch=fetch_vqa_annotations,
    ),
    DatasetSpec(
        key="chestxdet",
        title="ChestX-Det parquet mirror (second box source)",
        source="Hugging Face",
        repo_id="natealberti/ChestX-Det",
        subdir="chestxdet",
        license_note="Apache-2.0 annotations",
        fetch=fetch_chestxdet,
    ),
    DatasetSpec(
        key="nih",
        title="NIH ChestX-ray14 BBox_List_2017.csv (held-out box eval)",
        source="Hugging Face",
        repo_id="alkzar90/NIH-Chest-X-ray-dataset",
        subdir="nih",
        license_note="NIH Clinical Center open access",
        fetch=fetch_nih_bbox,
    ),
    DatasetSpec(
        key="open_i",
        title="Open-i / IU-Xray real reports (parser validation, no boxes)",
        source="Hugging Face",
        repo_id="ykumards/open-i",
        subdir="open_i",
        license_note="CC BY-NC-ND",
        fetch=fetch_open_i,
    ),
)

# Index for fast lookup by key.
_SPECS_BY_KEY: dict[str, DatasetSpec] = {spec.key: spec for spec in DATASET_SPECS}

# Sentinel selector that expands to every dataset.
ALL_SELECTOR = "all"


def resolve_specs(selectors: Sequence[str]) -> list[DatasetSpec]:
    """Resolve command-line selectors into the ordered list of dataset specs.

    Args:
        selectors: Dataset keys, or the single sentinel ``"all"``. An empty
            selection is treated as ``"all"``.

    Returns:
        The selected specs, de-duplicated, in registry order.

    Raises:
        ValueError: If a selector is neither ``"all"`` nor a known dataset key.
    """
    if not selectors or list(selectors) == [ALL_SELECTOR]:
        return list(DATASET_SPECS)

    chosen: set[str] = set()
    for selector in selectors:
        if selector == ALL_SELECTOR:
            return list(DATASET_SPECS)
        if selector not in _SPECS_BY_KEY:
            known = ", ".join([ALL_SELECTOR, *(spec.key for spec in DATASET_SPECS)])
            raise ValueError(f"unknown dataset {selector!r}; choose from: {known}")
        chosen.add(selector)
    # Return in registry order regardless of the order the user listed them.
    return [spec for spec in DATASET_SPECS if spec.key in chosen]


def format_plan(specs: Sequence[DatasetSpec], data_dir: Path) -> str:
    """Render the human-readable fetch plan for the selected datasets.

    Args:
        specs: The datasets that would be fetched.
        data_dir: The data root each dataset's ``subdir`` is relative to.

    Returns:
        A multi-line plan string (no side effects, no network).
    """
    lines: list[str] = ["CXR Draft Auditor data-fetch plan", ""]
    lines.extend(_LICENSE_NOTES)
    lines.append("")
    lines.append(f"Data root: {data_dir}")
    lines.append("")
    for index, spec in enumerate(specs, start=1):
        destination = data_dir / spec.subdir
        lines.append(f"{index}. {spec.title}")
        lines.append(f"   source:  {spec.source} :: {spec.repo_id}")
        lines.append(f"   license: {spec.license_note}")
        lines.append(f"   dest:    {destination}")
    return "\n".join(lines)


def run(specs: Sequence[DatasetSpec], data_dir: Path, *, dry_run: bool) -> int:
    """Print the plan and, unless ``dry_run``, fetch each selected dataset.

    Args:
        specs: The datasets to fetch.
        data_dir: The data root.
        dry_run: When true, only the plan is printed; no library is imported and
            no network access occurs.

    Returns:
        A process exit code: ``0`` on success, ``1`` if any fetch failed.
    """
    print(format_plan(specs, data_dir))
    if dry_run:
        print("\n[dry-run] No data fetched. Re-run without --dry-run to download.")
        return 0

    failures = 0
    for spec in specs:
        destination = data_dir / spec.subdir
        print(f"\n==> Fetching {spec.title} -> {destination}")
        try:
            spec.fetch(destination)
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {spec.title}: {exc}", file=sys.stderr)
        else:
            print(f"    done: {destination}")

    if failures:
        print(f"\n{failures} dataset(s) failed. See messages above.", file=sys.stderr)
        return 1
    print("\nAll selected datasets fetched.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the data-download CLI."""
    parser = argparse.ArgumentParser(
        prog="download_data.py",
        description=(
            "Fetch the CXR Draft Auditor datasets into the local data/ tree. "
            "USER-RUN: requires accepted licenses and configured credentials "
            "(see SETUP.md). Heavy libraries are imported only when fetching."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selectable = [ALL_SELECTOR, *(spec.key for spec in DATASET_SPECS)]
    parser.add_argument(
        "datasets",
        nargs="*",
        default=[ALL_SELECTOR],
        metavar="DATASET",
        help=f"Datasets to fetch (default: all). Choices: {', '.join(selectable)}.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for the downloaded datasets (default: ./data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fetch plan and exit without importing heavy libraries or touching the network.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        specs = resolve_specs(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    return run(specs, args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
