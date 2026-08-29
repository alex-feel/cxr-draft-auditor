#!/usr/bin/env python3
"""Publish the image-grounding SFT corpus as a private Hugging Face Dataset.

Hugging Face Jobs run in an ephemeral container that cannot see the local
``data/`` tree, so the chest X-ray pixels the training collator needs must be
delivered to the job. This script solves that the Hub-native way: it reads the
SFT JSON Lines corpus, loads every referenced PNG, and builds a
``datasets.Dataset`` whose ``images`` column embeds the image bytes directly
alongside the chat ``messages``. Pushing that dataset to the Hub with
``push_to_hub(private=True)`` makes the whole corpus -- pixels included --
loadable inside a Jobs container with a single ``load_dataset(<repo id>)`` call,
with no loose files to stage.

The published dataset has exactly two columns:

- ``images``: a list with one embedded image per record (the
  ``datasets.Image`` feature stores the pixels in the dataset itself).
- ``messages``: the chat-formatted user/assistant turns from the SFT record. The
  user turn references the image with an ``{"type": "image"}`` part and carries
  the grounding prompt; the assistant turn carries the canonical finding JSON the
  model is trained to emit.

This shape is exactly what TRL's ``DataCollatorForVisionLanguageModeling``
consumes when the SFT model is a vision-language model: the collator pairs each
record's ``images`` with the image placeholder in ``messages`` and masks the loss
to the assistant turn.

The heavy / network dependencies (``datasets``, ``PIL``, ``huggingface_hub``) are
imported lazily inside the functions that need them, so importing this module --
as the importability smoke test does -- never requires those packages, a network
connection, or credentials. ``--dry-run`` reports the plan (record count,
referenced-image count, approximate embedded size) without importing ``datasets``
or touching the network.

This script never pushes on its own initiative beyond the explicit
``push_to_hub`` call its CLI performs, and it never accepts a license or stores a
credential. The Hub token is read only from the ``HF_TOKEN`` environment
variable.

Usage::

    # Inspect the plan without any network access or heavy imports.
    python scripts/push_corpus_to_hub.py --hub-dataset-id me/cxr-sft --dry-run

    # Build and push the private dataset (requires HF_TOKEN in the environment).
    python scripts/push_corpus_to_hub.py --hub-dataset-id me/cxr-sft

    # Custom corpus / image root.
    python scripts/push_corpus_to_hub.py \
        --hub-dataset-id me/cxr-sft \
        --jsonl data/sft/train.jsonl \
        --image-root data

    # Publish the curated validation corpus as a second split in the same dataset
    # (so the trainer's --eval-split can load it for periodic evaluation).
    python scripts/push_corpus_to_hub.py \
        --hub-dataset-id me/cxr-sft \
        --jsonl data/sft/val.curated.jsonl \
        --split validation
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from cxr_auditor.sft_dataset import IMAGE_PART_TYPE, read_sft_jsonl

# Default corpus locations, tried in order: the curated corpus is preferred when
# it exists (it is the balanced/deduplicated training set), otherwise the full
# corpus the SFT builder writes.
_DEFAULT_JSONL_CANDIDATES: tuple[str, ...] = (
    "data/sft/train.curated.jsonl",
    "data/sft/train.jsonl",
)

# Default root that record image paths are resolved against. The SFT builder
# writes each ``image_path`` relative to the data root.
_DEFAULT_IMAGE_ROOT = "data"

# The default split name the corpus is published under on the Hub. The CLI
# ``--split`` flag overrides it so a curated validation corpus can be published as a
# second split (for example ``validation``) in the same dataset repo.
DEFAULT_SPLIT = "train"

# The two columns of the published dataset.
IMAGES_COLUMN = "images"
MESSAGES_COLUMN = "messages"


def default_jsonl(candidates: Sequence[str] = _DEFAULT_JSONL_CANDIDATES) -> Path:
    """Return the default corpus path: the curated corpus if present, else the full one.

    Args:
        candidates: Candidate corpus paths, in preference order.

    Returns:
        The first candidate that exists, or the last candidate as the fallback
        default when none exist (so the CLI reports a clear missing-file error
        against the expected location rather than against a nonexistent one).
    """
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    return Path(candidates[-1])


def resolve_record_image(record: dict[str, Any], image_root: Path) -> Path:
    """Resolve a single SFT record's image path against the image root.

    Args:
        record: One validated SFT record (carrying a string ``image_path``).
        image_root: Root directory relative image paths are resolved against.

    Returns:
        The resolved absolute image path (not checked for existence here).
    """
    return (image_root / str(record["image_path"])).resolve()


def collect_image_paths(records: Sequence[dict[str, Any]], image_root: Path) -> list[Path]:
    """Resolve every record's referenced image path against the image root.

    Args:
        records: The SFT records.
        image_root: Root directory relative image paths are resolved against.

    Returns:
        The resolved image paths, one per record, in record order. Paths may
        repeat when several records reference the same image.
    """
    return [resolve_record_image(record, image_root) for record in records]


def find_missing_images(image_paths: Sequence[Path]) -> list[Path]:
    """Return the subset of image paths that are not existing files.

    Args:
        image_paths: Resolved image paths.

    Returns:
        The paths that do not point at an existing file, preserving order and
        de-duplicating, so a missing image is reported once.
    """
    missing: list[Path] = []
    seen: set[Path] = set()
    for path in image_paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            missing.append(path)
    return missing


def estimate_embedded_bytes(image_paths: Sequence[Path]) -> int:
    """Approximate the embedded image payload size in bytes.

    The published dataset embeds each referenced image once (Arrow stores the
    distinct image bytes and references them per row), so the estimate sums the
    on-disk size of each distinct existing image file. Missing files contribute
    nothing; they are surfaced separately by ``find_missing_images``.

    Args:
        image_paths: Resolved image paths.

    Returns:
        The summed byte size of the distinct existing image files.
    """
    total = 0
    seen: set[Path] = set()
    for path in image_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            total += path.stat().st_size
    return total


def _format_size(num_bytes: int) -> str:
    """Format a byte count as a compact human-readable size string."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def summarize_plan(
    *,
    jsonl_path: Path,
    image_root: Path,
    hub_dataset_id: str,
    split: str,
    private: bool,
    record_count: int,
    distinct_image_count: int,
    missing_image_count: int,
    embedded_bytes: int,
) -> str:
    """Build the human-readable plan summary for ``--dry-run``.

    Args:
        jsonl_path: The corpus JSONL path.
        image_root: The image root directory.
        hub_dataset_id: The target Hub dataset repo id.
        split: The split name the corpus is published under.
        private: Whether the dataset would be pushed private.
        record_count: Number of SFT records (dataset rows).
        distinct_image_count: Number of distinct referenced images.
        missing_image_count: Number of distinct referenced images not on disk.
        embedded_bytes: Approximate embedded image payload size in bytes.

    Returns:
        A multi-line plan string.
    """
    visibility = "private" if private else "public"
    lines = [
        "CXR Draft Auditor SFT corpus push plan",
        f"  corpus JSONL:        {jsonl_path}",
        f"  image root:          {image_root}",
        f"  target dataset:      {hub_dataset_id} ({visibility})",
        f"  split:               {split}",
        f"  records (rows):      {record_count}",
        f"  distinct images:     {distinct_image_count}",
        f"  missing images:      {missing_image_count}",
        f"  approx embedded:     {_format_size(embedded_bytes)}",
    ]
    if missing_image_count:
        lines.append(
            f"  WARNING: {missing_image_count} referenced image(s) are missing; "
            "a real push would fail. Build the data/ tree first (see SETUP.md)."
        )
    return "\n".join(lines)


def _record_to_image_path_value(record: dict[str, Any]) -> str:
    """Extract the record's image path string."""
    return str(record["image_path"])


def iter_dataset_rows(records: Sequence[dict[str, Any]], image_root: Path) -> Iterator[dict[str, Any]]:
    """Yield the embedded-image dataset rows for the records.

    Each row has the ``{"images": [<PIL.Image>], "messages": [...]}`` shape the
    vision SFT collator consumes. The image referenced by each record is opened
    and converted to RGB (chest X-rays are commonly single-channel), so the
    pixels are embedded directly in the dataset rather than referenced by path.
    The record's ``messages`` are carried through unchanged; the user turn already
    references the image with an ``{"type": "image"}`` part.

    Args:
        records: The SFT records.
        image_root: Root directory relative image paths are resolved against.

    Yields:
        One dataset row per record, in record order.

    Raises:
        FileNotFoundError: If a referenced image is missing.
        ValueError: If a record's user turn carries no image part (the embedded
            image would have nothing to bind to).
    """
    image_module = importlib.import_module("PIL.Image")
    for index, record in enumerate(records):
        messages = record["messages"]
        if not _messages_have_image_part(messages):
            raise ValueError(
                f"record {index}: user turn has no '{IMAGE_PART_TYPE}' content part to bind the embedded image to"
            )
        image_path = resolve_record_image(record, image_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"record {index}: image not found: {image_path}")
        image = image_module.open(image_path).convert("RGB")
        yield {IMAGES_COLUMN: [image], MESSAGES_COLUMN: messages}


def _messages_have_image_part(messages: Any) -> bool:
    """Return whether any message content part is an image part."""
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == IMAGE_PART_TYPE:
                return True
    return False


def build_dataset(records: Sequence[dict[str, Any]], image_root: Path) -> Any:
    """Build the embedded-image ``datasets.Dataset`` from the SFT records.

    The ``images`` column is cast to the ``datasets.Image`` feature so the image
    bytes are stored inside the dataset (and travel with ``push_to_hub`` /
    ``load_dataset``) rather than being kept as loose file references.

    Args:
        records: The SFT records.
        image_root: Root directory relative image paths are resolved against.

    Returns:
        A ``datasets.Dataset`` with ``images`` and ``messages`` columns.

    Raises:
        FileNotFoundError: If a referenced image is missing.
        ValueError: If a record carries no image part.
    """
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'datasets' package is required to build and push the corpus but is not installed. "
            "Install it (pyarrow is pulled in automatically), then re-run:\n"
            "    uv pip install 'datasets>=3.0'\n"
            "or install the project's training extra:\n"
            "    uv pip install -e '.[train]'"
        ) from exc
    rows = list(iter_dataset_rows(records, image_root))
    dataset = datasets.Dataset.from_list(rows)
    return dataset.cast_column(IMAGES_COLUMN, datasets.Sequence(datasets.Image()))


def push_corpus(
    *,
    records: Sequence[dict[str, Any]],
    image_root: Path,
    hub_dataset_id: str,
    split: str,
    private: bool,
    token: str,
) -> str:
    """Build the embedded dataset and push it to the Hub as a private dataset.

    Pushing a second corpus to the same ``hub_dataset_id`` under a different
    ``split`` adds that split to the existing dataset (it does not replace the
    first split), so the train and validation corpora can share one dataset repo
    that the trainer loads by split name.

    Args:
        records: The SFT records.
        image_root: Root directory relative image paths are resolved against.
        hub_dataset_id: The target Hub dataset repo id (``namespace/name``).
        split: The split name to publish the corpus under.
        private: Whether to create the dataset repo private.
        token: The Hugging Face write token.

    Returns:
        The dataset repo id that was pushed.

    Raises:
        FileNotFoundError: If a referenced image is missing.
        ValueError: If a record carries no image part.
    """
    dataset = build_dataset(records, image_root)
    dataset.push_to_hub(hub_dataset_id, private=private, token=token, split=split)
    return hub_dataset_id


def run(
    *,
    jsonl_path: Path,
    image_root: Path,
    hub_dataset_id: str,
    split: str,
    private: bool,
    dry_run: bool,
) -> int:
    """Read the corpus, then push it (or print the plan under ``--dry-run``).

    Args:
        jsonl_path: The corpus JSONL path.
        image_root: The image root directory.
        hub_dataset_id: The target Hub dataset repo id.
        split: The split name to publish the corpus under.
        private: Whether to push the dataset private.
        dry_run: When true, report the plan and exit without importing
            ``datasets`` or touching the network.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a recoverable error
        (missing corpus, missing images, or a missing token on a real push).
    """
    if not jsonl_path.is_file():
        print(f"ERROR: corpus JSONL not found: {jsonl_path}", file=sys.stderr)
        return 1

    try:
        records = read_sft_jsonl(jsonl_path, validate=True)
    except ValueError as exc:
        print(f"ERROR: {jsonl_path}: {exc}", file=sys.stderr)
        return 1

    image_paths = collect_image_paths(records, image_root)
    missing = find_missing_images(image_paths)
    distinct_image_count = len({path for path in image_paths})
    embedded_bytes = estimate_embedded_bytes(image_paths)

    print(
        summarize_plan(
            jsonl_path=jsonl_path,
            image_root=image_root,
            hub_dataset_id=hub_dataset_id,
            split=split,
            private=private,
            record_count=len(records),
            distinct_image_count=distinct_image_count,
            missing_image_count=len(missing),
            embedded_bytes=embedded_bytes,
        )
    )

    if dry_run:
        print("\n[dry-run] No dataset built and nothing pushed.")
        return 0

    if missing:
        print(
            f"\nERROR: {len(missing)} referenced image(s) are missing; cannot push. "
            f"First missing: {missing[0]}",
            file=sys.stderr,
        )
        return 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set in the environment; cannot push.", file=sys.stderr)
        return 1

    pushed = push_corpus(
        records=records,
        image_root=image_root,
        hub_dataset_id=hub_dataset_id,
        split=split,
        private=private,
        token=token,
    )
    visibility = "private" if private else "public"
    print(
        f"\nPushed {len(records)} records to the {split!r} split of {visibility} dataset "
        f"https://huggingface.co/datasets/{pushed}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the corpus-push CLI."""
    parser = argparse.ArgumentParser(
        prog="push_corpus_to_hub.py",
        description=(
            "Publish the image-grounding SFT corpus as a private Hugging Face "
            "Dataset with embedded image bytes, so a Hugging Face Jobs container "
            "can load it (pixels included) via load_dataset. Build the data/ tree "
            "and the SFT JSONL first (see SETUP.md)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hub-dataset-id",
        required=True,
        help="Target Hub dataset repo id (namespace/name), for example me/cxr-sft.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=(
            "Corpus JSONL path (default: data/sft/train.curated.jsonl if present, "
            "else data/sft/train.jsonl)."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(_DEFAULT_IMAGE_ROOT),
        help=f"Root directory record image paths are resolved against (default: {_DEFAULT_IMAGE_ROOT!r}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=(
            f"Split name to publish the corpus under (default: {DEFAULT_SPLIT!r}). Run a second time with "
            "--split validation and --jsonl data/sft/val.curated.jsonl to add a validation split for --eval-split."
        ),
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Push the dataset public instead of private. The default is private (VinDr DUA non-commercial).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the push plan and exit without importing datasets or touching the network.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    args = build_arg_parser().parse_args(argv)
    jsonl_path = args.jsonl or default_jsonl()
    return run(
        jsonl_path=jsonl_path,
        image_root=args.image_root,
        hub_dataset_id=args.hub_dataset_id,
        split=args.split,
        private=not args.public,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
