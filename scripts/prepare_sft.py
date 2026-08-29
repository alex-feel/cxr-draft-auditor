#!/usr/bin/env python3
"""Build the image-grounding SFT JSONL corpus from the laid-out ``data/`` tree.

This is a USER-RUN helper that runs AFTER ``scripts/download_data.py`` has fetched
the datasets. It reads the VinDr-CXR box annotations (and, when present, the
VinDr-CXR-VQA annotations) from disk, converts every box into the canonical
normalized format, builds one supervised-fine-tuning record per image (the pinned
image-grounding prompt plus the canonical finding JSON target), and writes the
corpus to ``data/sft/train.jsonl``.

It composes the project's pure-logic pipeline modules:

- :mod:`cxr_auditor.data.vindr` parses the VinDr box CSV and the ``*_meta.csv``
  original-dimension table, rescaling each box to the canonical normalized format.
- :mod:`cxr_auditor.data.vqa_join` parses the VinDr-CXR-VQA ``data_v1.json``
  annotations and joins them to the same images by ``image_id``.
- :mod:`cxr_auditor.sft_dataset` renders each image's findings into a
  chat-formatted SFT record and writes/validates the JSONL.

Only the standard library and the dependency-light ``cxr_auditor`` core are
imported; there is no torch, transformers, ``datasets``, or network access. The
script reads files already on disk and emits a JSONL, so it runs on a plain CPU
environment.

Expected inputs under ``--data-dir`` (default ``data/``), matching the layout
``scripts/download_data.py`` produces::

    data/vindr/vinbigdata-512-image-dataset/   # resized PNG mirror + CSVs
    data/vqa/data_v1.json                       # optional VQA annotations

The VinDr box CSV and meta CSV file names vary between mirrors, so their paths are
overridable; sensible defaults are tried in order.

Usage::

    python scripts/prepare_sft.py --dry-run
    python scripts/prepare_sft.py
    python scripts/prepare_sft.py --data-dir /mnt/data --output /mnt/data/sft/train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from cxr_auditor.data.records import ImageBoxRecord
from cxr_auditor.data.vindr import load_vindr_boxes, load_vindr_dims_csv
from cxr_auditor.data.vqa_join import join_vqa_to_images, load_vqa_annotations
from cxr_auditor.schema import ImageFinding
from cxr_auditor.sft_dataset import build_sft_record, write_sft_jsonl

# Default candidate file names for the VinDr box CSV, tried in order. Different
# mirrors of the VinBigData competition name the train annotation CSV differently.
_VINDR_BOX_CSV_CANDIDATES: tuple[str, ...] = (
    "train.csv",
    "train_downsampled.csv",
    "annotations_train.csv",
)

# Default candidate file names for the VinDr original-dimension meta CSV.
_VINDR_META_CSV_CANDIDATES: tuple[str, ...] = (
    "train_meta.csv",
    "train_original_dimension.csv",
    "train_dimensions.csv",
)

# The default relative image path template for an SFT record. The resized mirror
# names each PNG by the 32-character hex image_id.
_DEFAULT_IMAGE_PATH_TEMPLATE = "vindr/vinbigdata-512-image-dataset/vinbigdata/train/{image_id}.png"


def _first_existing(directory: Path, candidates: Sequence[str]) -> Path | None:
    """Return the first candidate file that exists under ``directory``."""
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    return None


def record_to_image_findings(record: ImageBoxRecord) -> list[ImageFinding]:
    """Convert a parsed per-image box record into canonical image findings.

    Each :class:`~cxr_auditor.data.records.BoxRecord` becomes an
    :class:`~cxr_auditor.schema.ImageFinding` carrying the canonical finding label
    and its normalized box. The box record's native provenance label is dropped
    here because the SFT target speaks only the canonical label space.

    Args:
        record: A parsed per-image box record.

    Returns:
        The image's canonical findings, in the record's box order.
    """
    return [ImageFinding(finding=box.finding, box=box.box) for box in record.boxes]


def merge_vqa_findings(
    grouped: dict[str, list[ImageFinding]],
    vqa_by_image: dict[str, list],
) -> None:
    """Augment per-image findings in place with canonical VQA-joined findings.

    The VinDr-CXR-VQA annotations cover the same VinDr images and occasionally add
    a canonical finding the box CSV did not localize. For every image already in
    ``grouped``, any VQA record with a canonical finding not yet present (by
    ``(finding, box)``) is appended, so the SFT target reflects both annotation
    sources without duplicating identical findings. VQA images absent from
    ``grouped`` are ignored here because the image pixels are keyed off the VinDr
    mirror that produced ``grouped``.

    Args:
        grouped: Mapping from ``image_id`` to canonical findings, mutated in place.
        vqa_by_image: Mapping from ``image_id`` to its VQA records (objects with
            ``finding`` and ``box`` attributes).
    """
    for image_id, findings in grouped.items():
        existing = {(finding.finding, finding.box) for finding in findings}
        for vqa_record in vqa_by_image.get(image_id, []):
            if vqa_record.finding is None:
                continue
            key = (vqa_record.finding, vqa_record.box)
            if key in existing:
                continue
            existing.add(key)
            findings.append(ImageFinding(finding=vqa_record.finding, box=vqa_record.box))


def build_grouped_findings(
    data_dir: Path,
    *,
    box_csv: Path | None,
    meta_csv: Path | None,
    vqa_json: Path | None,
) -> dict[str, list[ImageFinding]]:
    """Load the datasets from disk into per-image canonical findings.

    Args:
        data_dir: The data root.
        box_csv: Explicit VinDr box CSV path, or ``None`` to search the default
            candidates under ``data_dir/vindr/vinbigdata-512-image-dataset``.
        meta_csv: Explicit VinDr meta (original-dimension) CSV path, or ``None`` to
            search the default candidates. When no meta CSV is found, boxes are
            kept without rescaling (``box=None``) so labels are not lost.
        vqa_json: Explicit VinDr-CXR-VQA ``data_v1.json`` path, or ``None`` to use
            ``data_dir/vqa/data_v1.json`` when it exists.

    Returns:
        Mapping from ``image_id`` to its canonical findings.

    Raises:
        FileNotFoundError: If no VinDr box CSV can be located.
    """
    mirror_dir = data_dir / "vindr" / "vinbigdata-512-image-dataset"
    # The awsaf49 mirror nests the CSVs and image folders under a ``vinbigdata/``
    # subdirectory; other mirrors place them at the mirror's top level. Search the
    # nested directory first, then the top level.
    vindr_search_dirs = [mirror_dir / "vinbigdata", mirror_dir]
    resolved_box_csv = box_csv
    if resolved_box_csv is None:
        for candidate_dir in vindr_search_dirs:
            resolved_box_csv = _first_existing(candidate_dir, _VINDR_BOX_CSV_CANDIDATES)
            if resolved_box_csv is not None:
                break
    if resolved_box_csv is None:
        searched = [str(directory) for directory in vindr_search_dirs]
        raise FileNotFoundError(
            f"no VinDr box CSV found (looked for {list(_VINDR_BOX_CSV_CANDIDATES)} in {searched}); "
            "pass --box-csv explicitly"
        )

    resolved_meta_csv = meta_csv
    if resolved_meta_csv is None:
        for candidate_dir in vindr_search_dirs:
            resolved_meta_csv = _first_existing(candidate_dir, _VINDR_META_CSV_CANDIDATES)
            if resolved_meta_csv is not None:
                break
    dims_by_image = None
    if resolved_meta_csv is not None:
        dims_by_image = load_vindr_dims_csv(resolved_meta_csv.read_text(encoding="utf-8"))

    box_records = load_vindr_boxes(resolved_box_csv, dims_by_image)
    grouped = {record.image_id: record_to_image_findings(record) for record in box_records}

    resolved_vqa = vqa_json or (data_dir / "vqa" / "data_v1.json")
    if resolved_vqa.is_file():
        vqa_records = load_vqa_annotations(resolved_vqa)
        vqa_by_image = join_vqa_to_images(vqa_records, available_image_ids=grouped.keys())
        merge_vqa_findings(grouped, vqa_by_image)

    return grouped


def write_corpus(
    grouped: dict[str, list[ImageFinding]],
    output_path: Path,
    image_path_template: str,
) -> int:
    """Build and write the SFT corpus from grouped findings.

    Args:
        grouped: Mapping from ``image_id`` to its canonical findings.
        output_path: Destination JSONL path. Parent directories are created.
        image_path_template: A format string with one ``{image_id}`` field used to
            build each record's relative image path.

    Returns:
        The number of records written.
    """
    records = [
        build_sft_record(image_path_template.format(image_id=image_id), grouped[image_id]) for image_id in sorted(grouped)
    ]
    return write_sft_jsonl(records, output_path, validate=True)


def run(
    data_dir: Path,
    output_path: Path,
    *,
    box_csv: Path | None,
    meta_csv: Path | None,
    vqa_json: Path | None,
    image_path_template: str,
    dry_run: bool,
) -> int:
    """Build the corpus and write it, unless ``dry_run``.

    Args:
        data_dir: The data root.
        output_path: Destination JSONL path.
        box_csv: Explicit VinDr box CSV path, or ``None`` to auto-locate.
        meta_csv: Explicit VinDr meta CSV path, or ``None`` to auto-locate.
        vqa_json: Explicit VQA ``data_v1.json`` path, or ``None`` to auto-locate.
        image_path_template: The per-record relative image path template.
        dry_run: When true, report what would be built without reading datasets or
            writing the corpus.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a missing-input error.
    """
    if dry_run:
        print("CXR Draft Auditor SFT preparation plan")
        print(f"  data root:        {data_dir}")
        print(f"  output JSONL:     {output_path}")
        print(f"  image path tmpl:  {image_path_template}")
        print(f"  VinDr box CSV:    {box_csv or '(auto-locate under vindr/)'}")
        print(f"  VinDr meta CSV:   {meta_csv or '(auto-locate; optional, enables box rescaling)'}")
        print(f"  VQA data_v1.json: {vqa_json or '(auto-locate under vqa/; optional)'}")
        print("\n[dry-run] No datasets read and no corpus written.")
        return 0

    try:
        grouped = build_grouped_findings(data_dir, box_csv=box_csv, meta_csv=meta_csv, vqa_json=vqa_json)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    count = write_corpus(grouped, output_path, image_path_template)
    print(f"Wrote {count} SFT records for {len(grouped)} images to {output_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the SFT-preparation CLI."""
    parser = argparse.ArgumentParser(
        prog="prepare_sft.py",
        description=(
            "Build the image-grounding SFT JSONL from the laid-out data/ tree. "
            "Run scripts/download_data.py first. Pure-logic only: no torch, no "
            "network."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory of the downloaded datasets (default: ./data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSONL path (default: <data-dir>/sft/train.jsonl).",
    )
    parser.add_argument(
        "--box-csv",
        type=Path,
        default=None,
        help="Explicit VinDr box CSV path (default: auto-locate under the mirror dir).",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Explicit VinDr original-dimension meta CSV path (optional; enables box rescaling).",
    )
    parser.add_argument(
        "--vqa-json",
        type=Path,
        default=None,
        help="Explicit VinDr-CXR-VQA data_v1.json path (default: auto-locate under vqa/).",
    )
    parser.add_argument(
        "--image-path-template",
        default=_DEFAULT_IMAGE_PATH_TEMPLATE,
        help=f"Per-record relative image path template with a {{image_id}} field (default: {_DEFAULT_IMAGE_PATH_TEMPLATE!r}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without reading datasets or writing the corpus.",
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
    output_path = args.output or (args.data_dir / "sft" / "train.jsonl")
    return run(
        args.data_dir,
        output_path,
        box_csv=args.box_csv,
        meta_csv=args.meta_csv,
        vqa_json=args.vqa_json,
        image_path_template=args.image_path_template,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
