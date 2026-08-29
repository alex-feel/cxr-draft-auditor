#!/usr/bin/env python3
"""Curate the image-grounding SFT corpus into balanced, deduplicated train/val splits.

This is a USER-RUN helper that runs AFTER ``scripts/prepare_sft.py`` has built the
raw corpus at ``data/sft/train.jsonl``. The raw corpus has two quality problems
that hurt a vision fine-tune:

- Box duplication: VinDr-CXR is triple-annotated, so a positive image carries many
  near-duplicate or exactly identical boxes per finding (one image can list dozens
  of overlapping boxes for the same finding). Training on those teaches the model
  to emit redundant boxes.
- Class imbalance: the corpus is dominated by normal (``no_finding``-only) studies,
  and the rarest urgent finding (``pneumothorax``) appears on only a handful of
  images. Training on that skew biases the model toward predicting "normal".

This script reads the raw JSONL, applies a curation step, and writes two files
under ``--out-dir``:

- ``train.curated.jsonl`` and ``val.curated.jsonl``: a stratified held-out split
  (default ten percent validation).

Curation has three stages:

1. Per record, per finding label, drop exact-duplicate boxes and merge boxes whose
   pairwise IoU meets ``--iou-merge-threshold`` into one representative (the mean
   box of the cluster), collapsing the triple-annotation into clean targets while
   keeping genuinely distinct (non-overlapping) boxes. The ``no_finding`` sentinel
   and findings with a null box pass through unchanged.
2. Keep every positive record and every record containing ``pneumothorax``;
   downsample ``no_finding``-only records to ``--normal-to-positive-ratio`` times
   the positive count, using an explicit seeded ``random.Random`` (never unseeded
   randomness, so the curation is reproducible).
3. Split the balanced corpus into train/val, stratifying on the per-record
   finding-presence signature so each split preserves the finding distribution as
   closely as the integer split sizes allow, seeded with the same ``--seed``.

The output records are re-emitted through the same builder ``prepare_sft`` uses
(:func:`cxr_auditor.sft_dataset.build_sft_record`), so the curated JSONL is in the
identical schema and is consumed unchanged by the corpus uploader and the trainer.

Only the standard library, numpy, and the dependency-light ``cxr_auditor`` core are
imported; there is no torch, transformers, ``datasets``, or network access.

Usage::

    python scripts/curate_sft.py --dry-run
    python scripts/curate_sft.py --input data/sft/train.jsonl --out-dir data/sft
    python scripts/curate_sft.py --normal-to-positive-ratio 1.5 --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cxr_auditor.eval.metrics import box_iou
from cxr_auditor.findings import CANONICAL_FINDINGS, NO_FINDING, specificity_rank
from cxr_auditor.schema import ImageFinding, NormalizedBox, extract_finding_list
from cxr_auditor.sft_dataset import build_sft_record, read_sft_jsonl

# Default curation parameters. They are exposed as CLI flags; the defaults encode
# the project's chosen balance target (roughly 1:1 normal:positive) and the IoU
# threshold above which two radiologist boxes for the same finding are treated as
# the same localization rather than two distinct ones.
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_NORMAL_TO_POSITIVE_RATIO = 1.0
DEFAULT_IOU_MERGE_THRESHOLD = 0.5
# IoU above which two boxes of DIFFERENT findings are treated as the same region;
# the less-specific finding is then dropped (see resolve_cross_finding_overlaps).
DEFAULT_CROSS_FINDING_IOU = 0.6
DEFAULT_SEED = 0

# Output file names written under --out-dir.
TRAIN_OUTPUT_NAME = "train.curated.jsonl"
VAL_OUTPUT_NAME = "val.curated.jsonl"

# The assistant turn is the second message; its first content part carries the
# JSON finding-list target. These indices match the schema enforced by
# ``cxr_auditor.sft_dataset.validate_sft_record``.
_ASSISTANT_MESSAGE_INDEX = 1
_TARGET_PART_INDEX = 0


def _assistant_target_text(record: dict[str, Any]) -> str:
    """Return the assistant-target JSON text of an SFT record."""
    return record["messages"][_ASSISTANT_MESSAGE_INDEX]["content"][_TARGET_PART_INDEX]["text"]


def record_present_findings(record: dict[str, Any]) -> set[str]:
    """Return the set of canonical positive findings asserted by a record.

    The ``no_finding`` sentinel is excluded, so a normal study returns the empty
    set. The result is the per-image finding-presence signature the balancing and
    stratification stages key off.

    Args:
        record: An SFT record (one parsed JSONL line).

    Returns:
        The set of canonical positive finding labels present in the assistant
        target.
    """
    findings = extract_finding_list(_assistant_target_text(record))
    return {element["label"] for element in findings if element.get("label") != NO_FINDING}


def is_normal_record(record: dict[str, Any]) -> bool:
    """Return whether a record is a normal study (``no_finding`` only)."""
    return not record_present_findings(record)


def _cluster_indices(boxes: list[NormalizedBox], iou_merge_threshold: float) -> list[list[int]]:
    """Group box indices into clusters connected by IoU >= the threshold.

    Two boxes are linked when their IoU meets ``iou_merge_threshold``; a cluster is
    a connected component of that link graph (so a chain of pairwise-overlapping
    boxes forms one cluster). Clustering is computed with a union-find over the
    boxes, which is deterministic for a fixed input order.

    Args:
        boxes: The boxes for a single finding label, in record order.
        iou_merge_threshold: Minimum IoU for two boxes to be linked.

    Returns:
        A list of clusters, each a list of indices into ``boxes``, with clusters
        and the indices within them in ascending order.
    """
    parent = list(range(len(boxes)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) >= iou_merge_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(boxes)):
        clusters[find(index)].append(index)
    return [clusters[root] for root in sorted(clusters)]


def _mean_box(boxes: list[NormalizedBox]) -> NormalizedBox:
    """Return the component-wise mean of a cluster of boxes.

    The mean of valid normalized boxes is itself a valid normalized box (each
    component stays in ``[0, 1]`` and the top-left/bottom-right ordering is
    preserved because the mean is monotone), so the representative needs no
    re-clamping.
    """
    count = len(boxes)
    y0 = sum(box[0] for box in boxes) / count
    x0 = sum(box[1] for box in boxes) / count
    y1 = sum(box[2] for box in boxes) / count
    x1 = sum(box[3] for box in boxes) / count
    return (y0, x0, y1, x1)


def curate_findings(
    findings: Sequence[dict[str, Any]],
    *,
    iou_merge_threshold: float,
) -> list[dict[str, Any]]:
    """Deduplicate and merge a record's assistant-target finding dicts.

    For each finding label, boxes whose pairwise IoU meets ``iou_merge_threshold``
    are clustered and each cluster is replaced by a single representative whose box
    is the cluster mean. Exact-duplicate boxes have IoU 1.0 and therefore always
    merge. Genuinely distinct (non-overlapping) boxes stay separate. Findings with
    a null box (the ``no_finding`` sentinel or a non-localized positive finding)
    carry no box to merge and are emitted once per label.

    The output preserves the canonical label presentation order, and within a label
    orders the merged boxes by their cluster's first occurrence, so the result is
    deterministic for a fixed input.

    Args:
        findings: The assistant-target finding dicts (``{label, box_2d, ...}``).
        iou_merge_threshold: Minimum IoU for two boxes of the same label to merge.

    Returns:
        The curated finding dicts.
    """
    boxed_by_label: dict[str, list[NormalizedBox]] = defaultdict(list)
    has_null_box: dict[str, bool] = defaultdict(bool)
    for element in findings:
        label = element["label"]
        box = element.get("box_2d")
        if box is None:
            has_null_box[label] = True
        else:
            boxed_by_label[label].append((box[0], box[1], box[2], box[3]))

    # Emit in canonical order, with any non-canonical label (there should be none
    # in a valid corpus) following deterministically by sorted name.
    ordered_labels = [label for label in CANONICAL_FINDINGS if label in boxed_by_label or has_null_box.get(label)]
    extra_labels = sorted(set(boxed_by_label) | set(has_null_box) - set(CANONICAL_FINDINGS))
    ordered_labels += [label for label in extra_labels if label not in ordered_labels]

    curated: list[dict[str, Any]] = []
    for label in ordered_labels:
        boxes = boxed_by_label.get(label, [])
        if boxes:
            for cluster in _cluster_indices(boxes, iou_merge_threshold):
                representative = _mean_box([boxes[index] for index in cluster])
                curated.append({"label": label, "box_2d": list(representative)})
        elif has_null_box.get(label):
            curated.append({"label": label, "box_2d": None})
    return curated


def resolve_cross_finding_overlaps(
    findings: Sequence[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Drop the less-specific of two DIFFERENT findings that share a region.

    After per-label dedup, one region can still carry two different labels (the
    triple annotation often tags a focal mass as both ``nodule_mass`` and
    ``lung_opacity_consolidation``). When two findings of different labels have
    boxes whose IoU meets ``iou_threshold``, the more specific finding (lower
    ``findings.specificity_rank``) is kept and the other is dropped, so the model
    learns one clean label per region instead of redundant double labels. Findings
    with a null box never overlap and pass through. With ``iou_threshold <= 0`` the
    step is a no-op. Input order is preserved among the kept findings.

    Args:
        findings: The per-label-deduped finding dicts (``{label, box_2d}``).
        iou_threshold: Minimum IoU for two different-label boxes to be one region.

    Returns:
        The findings with same-region cross-finding overlaps resolved.
    """
    if iou_threshold <= 0:
        return list(findings)

    boxed = [(index, element) for index, element in enumerate(findings) if element.get("box_2d") is not None]
    dropped: set[int] = set()
    for a in range(len(boxed)):
        index_a, element_a = boxed[a]
        for b in range(a + 1, len(boxed)):
            index_b, element_b = boxed[b]
            if element_a["label"] == element_b["label"]:
                continue
            box_a = element_a["box_2d"]
            box_b = element_b["box_2d"]
            iou = box_iou(
                (box_a[0], box_a[1], box_a[2], box_a[3]),
                (box_b[0], box_b[1], box_b[2], box_b[3]),
            )
            if iou >= iou_threshold:
                keep_a = specificity_rank(element_a["label"]) <= specificity_rank(element_b["label"])
                dropped.add(index_b if keep_a else index_a)

    return [element for index, element in enumerate(findings) if index not in dropped]


def curate_record(
    record: dict[str, Any],
    *,
    iou_merge_threshold: float,
    cross_finding_iou_threshold: float = DEFAULT_CROSS_FINDING_IOU,
) -> dict[str, Any]:
    """Return a copy of an SFT record with its boxes deduplicated and merged.

    Boxes are first merged per label, then same-region overlaps between DIFFERENT
    findings are resolved (the less-specific label is dropped). The record is then
    rebuilt through :func:`cxr_auditor.sft_dataset.build_sft_record`, so the output
    is in the identical schema as the raw corpus and is consumed unchanged by the
    trainer.

    Args:
        record: An SFT record (one parsed JSONL line).
        iou_merge_threshold: Minimum IoU for two boxes of the same label to merge.
        cross_finding_iou_threshold: Minimum IoU for two boxes of DIFFERENT labels
            to be treated as one region (the less-specific label is dropped).

    Returns:
        A new curated SFT record.
    """
    findings = extract_finding_list(_assistant_target_text(record))
    curated = curate_findings(findings, iou_merge_threshold=iou_merge_threshold)
    curated = resolve_cross_finding_overlaps(curated, iou_threshold=cross_finding_iou_threshold)

    present = [element for element in curated if element["label"] != NO_FINDING]
    image_findings = [
        ImageFinding(
            finding=element["label"],
            box=tuple(element["box_2d"]) if element["box_2d"] is not None else None,
        )
        for element in present
    ]
    return build_sft_record(record["image_path"], image_findings)


def balance_records(
    records: Sequence[dict[str, Any]],
    *,
    normal_to_positive_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Downsample normal studies to rebalance the corpus against positives.

    Every positive record is kept, and every record containing ``pneumothorax``
    (the rarest urgent finding) is kept even though it is already a positive, which
    makes the pneumothorax guarantee explicit and robust to a future change in how
    "positive" is computed. The ``no_finding``-only records are downsampled to
    ``round(normal_to_positive_ratio * n_positive)`` using a seeded
    ``random.Random`` so the selection is reproducible; when fewer normals exist
    than the target, all of them are kept.

    Record order is preserved: the kept normals are returned in their original
    relative order (the seeded shuffle decides which are kept, not the output
    order), and positives keep their order, so the function is deterministic.

    Args:
        records: The curated SFT records.
        normal_to_positive_ratio: Target ratio of normal to positive records.
        seed: Seed for the reproducible normal downsampling.

    Returns:
        The balanced records: all positives followed by the sampled normals, each
        group in its original relative order.
    """
    positives = [record for record in records if not is_normal_record(record)]
    normals = [record for record in records if is_normal_record(record)]

    target_normal = round(normal_to_positive_ratio * len(positives))
    if target_normal >= len(normals):
        kept_normals = normals
    else:
        rng = random.Random(seed)
        kept_indices = set(rng.sample(range(len(normals)), target_normal))
        kept_normals = [normal for index, normal in enumerate(normals) if index in kept_indices]

    return positives + kept_normals


def _stratum_key(record: dict[str, Any]) -> tuple[str, ...]:
    """Return the stratification key for a record (its finding-presence signature).

    Normal studies map to the single-element ``(NO_FINDING,)`` key; positive
    records map to the sorted tuple of their present findings, so co-occurring
    findings form their own stratum and are split proportionally.
    """
    present = record_present_findings(record)
    if not present:
        return (NO_FINDING,)
    return tuple(sorted(present))


def stratified_split(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train/val, stratified by finding-presence signature.

    Within each stratum (records sharing the same set of present findings) a
    ``round(len * val_fraction)`` slice is held out for validation after a seeded
    shuffle, so each stratum contributes proportionally and the overall finding
    distribution is preserved as closely as the integer split sizes allow. The
    split is reproducible for a fixed ``seed`` and the two parts partition the input
    with no overlap and no loss.

    Args:
        records: The balanced SFT records.
        val_fraction: Fraction of each stratum to place in the validation split.
        seed: Seed for the reproducible per-stratum shuffle.

    Returns:
        A ``(train, val)`` tuple of record lists.
    """
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[_stratum_key(record)].append(record)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        group = list(by_stratum[stratum])
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """A before/after distribution snapshot of an SFT corpus.

    Attributes:
        total: Total number of records.
        normal: Number of ``no_finding``-only (normal study) records.
        positive: Number of records asserting at least one positive finding.
        per_label: Count of records in which each canonical label is present
            (``no_finding`` counts the normal studies).
    """

    total: int
    normal: int
    positive: int
    per_label: dict[str, int]


def summarize_distribution(records: Sequence[dict[str, Any]]) -> DistributionSummary:
    """Compute the distribution summary of a corpus.

    Args:
        records: The SFT records to summarize.

    Returns:
        A :class:`DistributionSummary` with totals, the normal/positive split, and
        the per-label presence counts.
    """
    per_label: dict[str, int] = {label: 0 for label in CANONICAL_FINDINGS}
    normal = 0
    for record in records:
        present = record_present_findings(record)
        if present:
            for label in present:
                per_label[label] = per_label.get(label, 0) + 1
        else:
            normal += 1
            per_label[NO_FINDING] += 1
    total = len(records)
    return DistributionSummary(total=total, normal=normal, positive=total - normal, per_label=per_label)


def _format_summary(title: str, summary: DistributionSummary) -> str:
    """Render a distribution summary as an aligned multi-line block."""
    lines = [
        f"{title}:",
        f"  total:    {summary.total}",
        f"  normal:   {summary.normal}",
        f"  positive: {summary.positive}",
        "  per-label presence:",
    ]
    for label in CANONICAL_FINDINGS:
        lines.append(f"    {label:<28} {summary.per_label.get(label, 0)}")
    return "\n".join(lines)


def curate_corpus(
    records: Sequence[dict[str, Any]],
    *,
    iou_merge_threshold: float,
    normal_to_positive_ratio: float,
    val_fraction: float,
    seed: int,
    cross_finding_iou_threshold: float = DEFAULT_CROSS_FINDING_IOU,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the full curation pipeline on a raw corpus.

    Box dedup/merge runs first, then class balancing, then the stratified split, so
    the split sees the deduplicated finding signatures and the balanced class mix.

    Args:
        records: The raw SFT records.
        iou_merge_threshold: Minimum IoU for two boxes of the same label to merge.
        normal_to_positive_ratio: Target ratio of normal to positive records.
        val_fraction: Fraction of each stratum to hold out for validation.
        seed: Seed shared by the balancing and split stages.

    Returns:
        A ``(train, val)`` tuple of curated record lists.
    """
    deduped = [
        curate_record(
            record,
            iou_merge_threshold=iou_merge_threshold,
            cross_finding_iou_threshold=cross_finding_iou_threshold,
        )
        for record in records
    ]
    balanced = balance_records(deduped, normal_to_positive_ratio=normal_to_positive_ratio, seed=seed)
    return stratified_split(balanced, val_fraction=val_fraction, seed=seed)


def _write_jsonl(records: Sequence[dict[str, Any]], output_path: Path) -> int:
    """Write records to a JSONL file, one per line; returns the count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return len(records)


def run(
    input_path: Path,
    out_dir: Path,
    *,
    val_fraction: float,
    normal_to_positive_ratio: float,
    iou_merge_threshold: float,
    seed: int,
    dry_run: bool,
) -> int:
    """Curate the corpus and write the train/val splits, unless ``dry_run``.

    Args:
        input_path: Path to the raw ``train.jsonl`` corpus.
        out_dir: Directory the curated splits are written to.
        val_fraction: Fraction of each stratum held out for validation.
        normal_to_positive_ratio: Target ratio of normal to positive records.
        iou_merge_threshold: Minimum IoU for two boxes of the same label to merge.
        seed: Seed shared by the balancing and split stages.
        dry_run: When true, print the plan and exit without reading the corpus or
            writing any output.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a missing-input error.
    """
    train_path = out_dir / TRAIN_OUTPUT_NAME
    val_path = out_dir / VAL_OUTPUT_NAME

    if dry_run:
        print("CXR Draft Auditor SFT curation plan")
        print(f"  input corpus:           {input_path}")
        print(f"  output train:           {train_path}")
        print(f"  output val:             {val_path}")
        print(f"  val fraction:           {val_fraction}")
        print(f"  normal:positive ratio:  {normal_to_positive_ratio}")
        print(f"  IoU merge threshold:    {iou_merge_threshold}")
        print(f"  seed:                   {seed}")
        print("\n[dry-run] No corpus read and no output written.")
        return 0

    if not input_path.is_file():
        print(f"ERROR: input corpus not found: {input_path}", file=sys.stderr)
        return 1

    raw_records = read_sft_jsonl(input_path, validate=True)
    train, val = curate_corpus(
        raw_records,
        iou_merge_threshold=iou_merge_threshold,
        normal_to_positive_ratio=normal_to_positive_ratio,
        val_fraction=val_fraction,
        seed=seed,
    )

    print(_format_summary("BEFORE curation", summarize_distribution(raw_records)))
    print()
    print(_format_summary("AFTER curation (train)", summarize_distribution(train)))
    print()
    print(_format_summary("AFTER curation (val)", summarize_distribution(val)))
    print()

    n_train = _write_jsonl(train, train_path)
    n_val = _write_jsonl(val, val_path)
    print(f"Wrote {n_train} train records to {train_path}")
    print(f"Wrote {n_val} val records to {val_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the SFT-curation CLI."""
    parser = argparse.ArgumentParser(
        prog="curate_sft.py",
        description=(
            "Curate the raw SFT corpus: dedup/merge triple-annotated boxes, "
            "balance normal vs positive studies, and write a stratified train/val "
            "split. Run scripts/prepare_sft.py first. Pure-logic only: no torch, "
            "no network."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "sft" / "train.jsonl",
        help="Path to the raw SFT corpus (default: data/sft/train.jsonl).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "sft",
        help="Directory for the curated train/val JSONL (default: data/sft).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f"Fraction of each stratum held out for validation (default: {DEFAULT_VAL_FRACTION}).",
    )
    parser.add_argument(
        "--normal-to-positive-ratio",
        type=float,
        default=DEFAULT_NORMAL_TO_POSITIVE_RATIO,
        help=f"Target ratio of normal to positive records (default: {DEFAULT_NORMAL_TO_POSITIVE_RATIO}).",
    )
    parser.add_argument(
        "--iou-merge-threshold",
        type=float,
        default=DEFAULT_IOU_MERGE_THRESHOLD,
        help=f"Minimum IoU for two boxes of one label to merge (default: {DEFAULT_IOU_MERGE_THRESHOLD}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed shared by the balancing and split stages (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without reading the corpus or writing output.",
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
    return run(
        args.input,
        args.out_dir,
        val_fraction=args.val_fraction,
        normal_to_positive_ratio=args.normal_to_positive_ratio,
        iou_merge_threshold=args.iou_merge_threshold,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
