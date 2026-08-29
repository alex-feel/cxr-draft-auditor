# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "transformers>=4.56",
#     "datasets>=3.0",
#     "accelerate>=0.34",
#     "huggingface_hub>=0.26",
#     "pillow>=10.3",
#     "numpy>=1.26",
#     "pydantic>=2.7",
# ]
#
# # Pin torch (and its torchvision side-car) to a CUDA 12.6 build from the PyTorch
# # wheel index, exactly as train/hf_job_sft.py does. Without this, uv resolves the
# # latest torch, which since torch 2.11 ships CUDA 13 (cu13) wheels by default; a
# # cu13 build needs a CUDA 13 driver and cannot initialize the GPU on a node whose
# # driver is CUDA 12.x, which silently forces everything onto the CPU. A cu126
# # build runs on ANY CUDA 12.x or newer driver via CUDA minor-version backward
# # compatibility, so this pin is flavor-agnostic (a10g-large, a100-large, ...).
# # torch 2.8.0 has no cu130 wheel at all, so it can never drift to CUDA 13. Both
# # torch and torchvision MUST be routed to the index: a side-car left on PyPI would
# # pull a mismatched CUDA build and ImportError at load. `explicit = true` confines
# # the index to the named packages; everything else resolves from PyPI. `hf jobs uv
# # run` calls `uv run`, which honors these PEP 723 [tool.uv] sections.
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# torchvision = { index = "pytorch-cu126" }
# ///
"""Hugging Face Jobs evaluation of two CXR-grounding fine-tunes on one held-out set.

This is the Hugging Face Jobs entry point for the systematic v1-vs-v2 model
evaluation. It is a self-contained UV script (its dependencies are declared inline
in the PEP 723 header above), so it runs inside an ephemeral Jobs container with
nothing pre-installed and nothing pre-staged. It is the read-only inference
counterpart to ``train/hf_job_sft.py``: it loads each merged bf16 model, scores it
on the SAME held-out validation subset with greedy decoding matching production,
applies the SAME tolerant parsing as production, and publishes decision-grade
metrics so the served model can be chosen and a v3 retrain decision can be made.

Why this script does not reinvent the harness
----------------------------------------------
The quantitative metrics, the production prompt, and the tolerant model-output
parsing all already exist in the ``cxr_auditor`` package (``eval.metrics``,
``eval.run_eval``, ``prompts``, ``schema``, ``inference``). The package is not on
PyPI, so it cannot be a PEP 723 dependency. Instead it is shipped by uploading the
``src/cxr_auditor`` directory into the existing scripts dataset repo (under
``cxr_auditor/``); this script snapshot-downloads that repo and prepends it to
``sys.path`` before importing the package. Every metric, the prompt, and the parse
path therefore come from the same code the live Space runs, so the numbers are
faithful to production rather than a re-implementation.

Held-out integrity (no train leakage for either model)
------------------------------------------------------
v1 (``medgemma-cxr-auditor``) trained on ``cxr-sft`` and v2
(``medgemma-cxr-auditor-v2``) trained on ``cxr-sft-v2``; the two curations produced
DIFFERENT train/validation splits (their row counts differ), so a row in one corpus's
validation split is held out from THAT model by construction but is NOT guaranteed
held out from the OTHER model's training split. To enlarge the scarce urgent classes
while keeping every record held out from BOTH models, the evaluation subset is the
UNION of two fair, leakage-free pools, each held out from both models:

  * Pool A: ``cxr-sft/validation`` with any image also in ``cxr-sft-v2/train`` dropped.
  * Pool B: ``cxr-sft-v2/validation`` with any image also in ``cxr-sft/train`` dropped.

Each pool filters its own validation split against the OTHER model's training split,
matched by a re-encoding-robust downscaled-grayscale pixel hash, so a pool record is
held out from its own model (own validation split) and from the other model (filtered).
The two pools are unioned and deduplicated by the same image content hash (first pool
wins), so an image present in both validation splits is counted once and held out from
both by construction. Passing a single pool/exclude pair reproduces the original
single-pool behavior.

Ground truth
------------
The ground-truth grounded findings for each image are the assistant target in THAT
image's OWN source-pool record (the canonical finding JSON with boxes the corpus was
built from); the same per-image ground truth is used for BOTH models so the comparison
is apples-to-apples. One nuance is reported, not silently absorbed: the two pools use
different ground-truth conventions. Pool A (``cxr-sft``) retains same-region
opacity+nodule double-labels that the pool B (``cxr-sft-v2``) curation deduplicated to
the single specific label. Because that dedup resolves the opacity/nodule OVERLAP to
nodule, ``nodule_mass`` and ``pneumothorax`` presence are CONSISTENT across both
conventions, so urgent recall on the union is sound; generic
``lung_opacity_consolidation`` recall mixes the two conventions and is read with care.
The summary records this caveat and the per-pool counts so the reader can weigh it.

Pipeline
--------
1. Snapshot-download the scripts repo and import ``cxr_auditor`` from it.
2. For each pool/exclude pair, build its leakage-free held-out records (ground truth
   from that pair's own pool); union and deduplicate the pools by image content hash;
   stratify the union (all six canonical labels; every available pneumothorax and
   nodule_mass case kept so urgent recall is meaningful; remaining budget stratified
   across the other labels).
3. For each model in turn: load it (bf16, SDPA, greedy) exactly as the Space does,
   generate and tolerantly parse predictions for every subset image, free the GPU,
   then load the next model.
4. Compute per-finding precision/recall/F1, box IoU@0.3 and IoU@0.5, mean IoU on
   matched findings, and urgent recall for pneumothorax and nodule_mass, reusing the
   harness metrics; record raw parse failures.
5. Upload a summary JSON and a per-sample dump (including raw parse failures) to the
   scripts dataset repo under ``eval_results/`` so the results survive the job.

The heavy / GPU dependencies and the shipped ``cxr_auditor`` package are imported
lazily through ``importlib`` so the module imports for a syntax / type check without
a GPU stack or the package on ``sys.path``; at runtime inside the Jobs container the
imports resolve normally. The Hub token is read only from the ``HF_TOKEN``
environment variable (passed as a Jobs secret), never hard-coded.

Usage (inside a Jobs container; flags are passed as ``script_args``). The repeatable
pool flags zip positionally into ``(pool, pool_split, exclude, exclude_split)`` pairs::

    python hf_job_eval.py \
        --pool-dataset alex-feeel/cxr-sft --pool-split validation \
        --exclude-dataset alex-feeel/cxr-sft-v2 --exclude-split train \
        --pool-dataset alex-feeel/cxr-sft-v2 --pool-split validation \
        --exclude-dataset alex-feeel/cxr-sft --exclude-split train \
        --subset-size 480 \
        --results-basename union \
        --scripts-repo alex-feeel/cxr-auditor-scripts \
        --model alex-feeel/medgemma-cxr-auditor \
        --model alex-feeel/medgemma-cxr-auditor-v2

See ``train/hf_jobs.md`` for the training runbook this evaluation complements.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
import random
import sys
from typing import Any

# Default identifiers. These are the project's own namespaces; the two corpus
# datasets are private by design (VinDr-CXR data-use agreement -- see the model
# cards), and every one is a flag so a run can substitute its own repos without
# editing this file.
DEFAULT_POOL_DATASET = "alex-feeel/cxr-sft"
DEFAULT_POOL_SPLIT = "validation"
DEFAULT_EXCLUDE_DATASET = "alex-feeel/cxr-sft-v2"
DEFAULT_EXCLUDE_SPLIT = "train"
DEFAULT_SCRIPTS_REPO = "alex-feeel/cxr-auditor-scripts"
DEFAULT_MODELS: tuple[str, ...] = (
    "alex-feeel/medgemma-cxr-auditor",
    "alex-feeel/medgemma-cxr-auditor-v2",
)
DEFAULT_RESULTS_PREFIX = "eval_results"
# Default infix in the result filenames. The original single-pool run wrote
# ``summary_v1_vs_v2_latest.json``; keeping this as the default preserves that
# filename so single-pool reruns stay backward compatible. The union run passes a
# distinct basename (for example ``union``) so it does not clobber that file.
DEFAULT_RESULTS_BASENAME = "v1_vs_v2"

# The greedy decode budget for grounding. Matches the production serving default
# (cxr_auditor.inference.DEFAULT_MAX_NEW_TOKENS) so generation is faithful.
DEFAULT_MAX_NEW_TOKENS = 512

# Side length of the grayscale thumbnail used for the leakage de-duplication hash.
# Large enough that two distinct chest X-rays do not collide, small enough that the
# hash is invariant to PNG re-encoding between the two corpus builds.
_HASH_THUMB_SIZE = 64


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Every tunable is a flag so the runbook can override the datasets, the model
    list, the subset size, and the smoke controls via the Jobs ``script_args``
    without editing this file.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate two CXR-grounding fine-tunes on one held-out validation subset on Hugging Face Jobs.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="A merged model repo id to evaluate. Repeat for each model (default: v1 then v2).",
    )
    parser.add_argument(
        "--pool-dataset",
        dest="pool_datasets",
        action="append",
        default=None,
        help=(
            "Dataset an evaluation pool is drawn from. Repeat to evaluate on the UNION of several pools; "
            "the i-th --pool-dataset, --pool-split, --exclude-dataset, and --exclude-split form one ordered "
            "pool/exclude pair (default: a single cxr-sft/validation pool)."
        ),
    )
    parser.add_argument(
        "--pool-split",
        dest="pool_splits",
        action="append",
        default=None,
        help="Split of the matching --pool-dataset to draw from (repeat once per pool).",
    )
    parser.add_argument(
        "--exclude-dataset",
        dest="exclude_datasets",
        action="append",
        default=None,
        help=(
            "Dataset whose split is excluded from the matching pool by image hash (the OTHER model's training "
            "corpus). Repeat once per pool."
        ),
    )
    parser.add_argument(
        "--exclude-split",
        dest="exclude_splits",
        action="append",
        default=None,
        help="Split of the matching --exclude-dataset to remove from the pool (repeat once per pool).",
    )
    parser.add_argument(
        "--skip-leakage-filter",
        action="store_true",
        help="Skip the cross-corpus de-duplication filter (smoke runs only; the full run must NOT skip it).",
    )
    parser.add_argument(
        "--scripts-repo",
        default=DEFAULT_SCRIPTS_REPO,
        help="Dataset repo that carries the shipped cxr_auditor package and receives the results.",
    )
    parser.add_argument(
        "--results-prefix",
        default=DEFAULT_RESULTS_PREFIX,
        help="Path prefix inside the scripts repo where result files are written.",
    )
    parser.add_argument(
        "--results-basename",
        default=DEFAULT_RESULTS_BASENAME,
        help=(
            "Infix used in result filenames (summary_<basename>_latest.json, per_sample_<basename>_latest.json, "
            "and timestamped history copies). Use a distinct value (for example 'union') so a new run does not "
            "clobber an earlier run's results."
        ),
    )
    parser.add_argument("--subset-size", type=int, default=280, help="Target number of images in the stratified subset.")
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=-1,
        help="If > 0, cap the subset to the first N images after stratification (smoke runs).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Greedy decode budget per generation (matches production serving).",
    )
    parser.add_argument("--seed", type=int, default=20260612, help="Seed for the deterministic stratified sample.")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Compute and print metrics but skip the Hub upload (smoke runs / local dry tests).",
    )
    return parser


def import_shipped_package(scripts_repo: str, token: str | None) -> None:
    """Snapshot-download the scripts repo and put the shipped ``cxr_auditor`` on ``sys.path``.

    The ``cxr_auditor`` package is uploaded into the scripts dataset repo under
    ``cxr_auditor/``. Downloading the repo and prepending its root to ``sys.path``
    makes ``import cxr_auditor`` resolve to exactly the package the live Space runs,
    so the metrics, prompt, and parse path are not re-implementations.

    Args:
        scripts_repo: The dataset repo id holding the shipped package.
        token: The Hugging Face token (the repo may be private).

    Raises:
        RuntimeError: If the downloaded snapshot does not contain a ``cxr_auditor``
            package directory.
    """
    hub = importlib.import_module("huggingface_hub")
    local_root = hub.snapshot_download(
        repo_id=scripts_repo,
        repo_type="dataset",
        token=token,
        allow_patterns=["cxr_auditor/**"],
    )
    package_init = os.path.join(local_root, "cxr_auditor", "__init__.py")
    if not os.path.isfile(package_init):
        raise RuntimeError(
            f"the scripts repo {scripts_repo!r} does not contain a 'cxr_auditor/' package "
            f"(expected {package_init}); upload src/cxr_auditor into it first."
        )
    if local_root not in sys.path:
        sys.path.insert(0, local_root)


def image_content_hash(image: Any) -> str:
    """Compute a re-encoding-robust content hash of one image.

    The hash is taken over the bytes of a fixed-size grayscale thumbnail of the
    decoded pixels, so it is identical for the same chest X-ray regardless of how
    each corpus build re-encoded the PNG, while distinct X-rays do not collide at
    this resolution. Used to detect images shared between the evaluation pool and
    the other model's training split.

    Args:
        image: A ``PIL.Image`` instance.

    Returns:
        A hex SHA-1 digest of the normalized thumbnail.
    """
    image_module = importlib.import_module("PIL.Image")
    thumb = image.convert("L").resize((_HASH_THUMB_SIZE, _HASH_THUMB_SIZE), image_module.Resampling.BILINEAR)
    return hashlib.sha1(thumb.tobytes()).hexdigest()


def build_exclusion_hashes(dataset: Any) -> set[str]:
    """Hash every image in a dataset split to form a content-exclusion set.

    Args:
        dataset: A loaded ``datasets.Dataset`` with an ``images`` column whose first
            element is the chest X-ray for the row.

    Returns:
        The set of content hashes of every image in the split.
    """
    hashes: set[str] = set()
    for row in dataset:
        images = row["images"]
        if images:
            hashes.add(image_content_hash(images[0]))
    return hashes


def assistant_target_text(messages: Any) -> str | None:
    """Extract the assistant turn's target text from a chat-formatted record.

    Args:
        messages: The record's ``messages`` list (user turn then assistant turn).

    Returns:
        The assistant turn's text part, or ``None`` if the record has no assistant
        text part (a malformed row that should be skipped).
    """
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                return part["text"]
    return None


def present_labels(findings: list[Any], no_finding_label: str) -> set[str]:
    """Collapse a finding list to the set of positive labels asserted present.

    Args:
        findings: ``ImageFinding`` objects (predicted or ground-truth).
        no_finding_label: The canonical negative-sentinel label to exclude.

    Returns:
        The set of canonical positive labels present in the list.
    """
    return {finding.finding for finding in findings if finding.finding != no_finding_label}


def build_ground_truth(
    dataset: Any,
    cxr_auditor: Any,
    *,
    exclusion_hashes: set[str],
    source_pool: str,
) -> list[dict[str, Any]]:
    """Decode one pool split into per-image ground-truth records, dropping leaked images.

    Each kept record carries the image, its content hash, the source-pool tag, and
    the ground-truth ``ImageFinding`` objects parsed from THIS pool's own assistant
    target with the same tolerant parser production uses. Images whose hash is in
    ``exclusion_hashes`` (present in the paired exclude split, the other model's
    training split) are dropped so every record is held out from both models.

    Args:
        dataset: The loaded pool split.
        cxr_auditor: The shipped package's namespace (provides ``schema`` and
            ``inference`` parse helpers).
        exclusion_hashes: Content hashes to exclude (empty to keep every image).
        source_pool: A ``dataset:split`` tag recording which pool produced the
            record (so the union is auditable per pool).

    Returns:
        A list of ground-truth records, each
        ``{"image", "hash", "gt_findings", "source_pool"}``.
    """
    schema = cxr_auditor["schema"]
    inference = cxr_auditor["inference"]

    records: list[dict[str, Any]] = []
    for row in dataset:
        images = row["images"]
        if not images:
            continue
        image = images[0].convert("RGB")
        content_hash = image_content_hash(image)
        if content_hash in exclusion_hashes:
            continue
        target_text = assistant_target_text(row["messages"])
        if target_text is None:
            continue
        try:
            grounded = schema.extract_finding_list(target_text)
        except schema.SchemaParseError:
            continue
        gt_findings = inference.grounded_dicts_to_image_findings(grounded)
        records.append(
            {"image": image, "hash": content_hash, "gt_findings": gt_findings, "source_pool": source_pool}
        )
    return records


def resolve_pool_pairs(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    """Resolve the repeatable pool/exclude flags into ordered evaluation pairs.

    Each pair is ``(pool_dataset, pool_split, exclude_dataset, exclude_split)``: the
    i-th value of every repeatable flag forms the i-th pair. When no pool flags are
    passed the single default pool/exclude pair is returned, preserving the original
    single-pool behavior. All four lists must have equal length so the zip is
    unambiguous.

    Args:
        args: The parsed command-line namespace (carries ``pool_datasets``,
            ``pool_splits``, ``exclude_datasets``, ``exclude_splits``).

    Returns:
        The ordered list of pool/exclude pairs.

    Raises:
        SystemExit: If the four repeatable flags are passed with mismatched counts.
    """
    pool_datasets = args.pool_datasets if args.pool_datasets else [DEFAULT_POOL_DATASET]
    pool_splits = args.pool_splits if args.pool_splits else [DEFAULT_POOL_SPLIT]
    exclude_datasets = args.exclude_datasets if args.exclude_datasets else [DEFAULT_EXCLUDE_DATASET]
    exclude_splits = args.exclude_splits if args.exclude_splits else [DEFAULT_EXCLUDE_SPLIT]

    lengths = {
        "--pool-dataset": len(pool_datasets),
        "--pool-split": len(pool_splits),
        "--exclude-dataset": len(exclude_datasets),
        "--exclude-split": len(exclude_splits),
    }
    if len(set(lengths.values())) != 1:
        raise SystemExit(
            "the repeatable pool flags must be passed the same number of times so they zip into "
            f"ordered (pool, pool_split, exclude, exclude_split) pairs; got counts {lengths}"
        )
    return list(zip(pool_datasets, pool_splits, exclude_datasets, exclude_splits, strict=True))


def union_pool_records(
    per_pool_records: list[list[dict[str, Any]]],
    cxr_auditor: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Union per-pool ground-truth records and deduplicate by image content hash.

    The pools are merged in their given order; the first occurrence of each image
    content hash wins, so an image present in more than one pool's validation split
    is counted once and (because every pool's records are already held out from that
    pool's paired exclude split) is held out from both models by construction. Each
    retained record keeps the ``source_pool`` tag of the pool that first contributed
    it, making the union auditable per pool.

    Args:
        per_pool_records: One held-out ground-truth record list per pool, in pool
            order.
        cxr_auditor: The shipped package namespace (provides ``findings`` for the
            urgent-label and no-finding constants).

    Returns:
        A tuple ``(union_records, union_audit)`` where ``union_audit`` reports, per
        pool, the candidate count contributed to the deduplicated union and the
        urgent-label counts among those candidates, plus the union total and the
        number of cross-pool duplicates removed.
    """
    findings_module = cxr_auditor["findings"]
    no_finding = findings_module.NO_FINDING
    urgent_labels = sorted(findings_module.URGENT_WHITELIST)

    seen: set[str] = set()
    union_records: list[dict[str, Any]] = []
    duplicates_removed = 0
    raw_per_pool: dict[str, int] = {}
    per_pool_candidate_counts: dict[str, int] = {}
    per_pool_urgent_counts: dict[str, dict[str, int]] = {}

    for pool_records in per_pool_records:
        for record in pool_records:
            pool = record["source_pool"]
            raw_per_pool[pool] = raw_per_pool.get(pool, 0) + 1
            content_hash = record["hash"]
            if content_hash in seen:
                duplicates_removed += 1
                continue
            seen.add(content_hash)
            union_records.append(record)
            per_pool_candidate_counts[pool] = per_pool_candidate_counts.get(pool, 0) + 1
            urgent_for_pool = per_pool_urgent_counts.setdefault(pool, {label: 0 for label in urgent_labels})
            record_labels = present_labels(record["gt_findings"], no_finding)
            for label in urgent_labels:
                if label in record_labels:
                    urgent_for_pool[label] += 1

    union_audit: dict[str, Any] = {
        "raw_per_pool_candidate_counts": raw_per_pool,
        "per_pool_candidate_counts": per_pool_candidate_counts,
        "per_pool_urgent_counts": per_pool_urgent_counts,
        "union_total_candidates": len(union_records),
        "cross_pool_duplicates_removed": duplicates_removed,
    }
    return union_records, union_audit


def stratify_subset(
    records: list[dict[str, Any]],
    cxr_auditor: Any,
    *,
    subset_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a stratified subset covering all six labels with maximal urgent cases.

    Every available pneumothorax and nodule_mass image is kept (those urgent classes
    are scarce, so urgent recall stays as meaningful as the split allows), then the
    remaining budget is filled by a seeded round-robin over the other label strata so
    every canonical label is represented. The selection is deterministic for a fixed
    seed.

    Args:
        records: The leakage-free ground-truth records.
        cxr_auditor: The shipped package namespace (provides ``findings``).
        subset_size: Target number of images in the subset.
        seed: Seed for the deterministic fill.

    Returns:
        A tuple ``(subset, composition)`` where ``subset`` is the chosen records and
        ``composition`` describes the subset (per-label image counts, urgent counts,
        no-finding-only count, total, and the budget).
    """
    findings_module = cxr_auditor["findings"]
    no_finding = findings_module.NO_FINDING
    positive_findings: tuple[str, ...] = tuple(findings_module.POSITIVE_FINDINGS)
    urgent_labels = sorted(findings_module.URGENT_WHITELIST)

    rng = random.Random(seed)

    # Annotate each record with its present-label set once.
    for index, record in enumerate(records):
        record["index"] = index
        record["labels"] = present_labels(record["gt_findings"], no_finding)

    selected: dict[int, dict[str, Any]] = {}

    # Keep every urgent (pneumothorax, nodule_mass) image: these classes are scarce.
    for record in records:
        if record["labels"] & set(urgent_labels):
            selected[record["index"]] = record

    # Build per-stratum candidate pools (excluding already-selected) for the fill.
    other_labels = [label for label in positive_findings if label not in urgent_labels]
    strata: dict[str, list[dict[str, Any]]] = {label: [] for label in other_labels}
    strata[no_finding] = []
    for record in records:
        if record["index"] in selected:
            continue
        if not record["labels"]:
            strata[no_finding].append(record)
            continue
        for label in other_labels:
            if label in record["labels"]:
                strata[label].append(record)
    for pool in strata.values():
        rng.shuffle(pool)

    # Round-robin fill from the strata until the budget is met or pools are empty.
    stratum_keys = [*other_labels, no_finding]
    cursor = {label: 0 for label in stratum_keys}
    while len(selected) < subset_size:
        progressed = False
        for label in stratum_keys:
            if len(selected) >= subset_size:
                break
            pool = strata[label]
            while cursor[label] < len(pool):
                candidate = pool[cursor[label]]
                cursor[label] += 1
                if candidate["index"] not in selected:
                    selected[candidate["index"]] = candidate
                    progressed = True
                    break
        if not progressed:
            break

    subset = [selected[index] for index in sorted(selected)]

    # Per-label image counts within the subset (an image may carry several labels).
    label_counts = {label: 0 for label in positive_findings}
    no_finding_only = 0
    for record in subset:
        if not record["labels"]:
            no_finding_only += 1
        for label in record["labels"]:
            label_counts[label] += 1

    composition: dict[str, Any] = {
        "total_images": len(subset),
        "target_subset_size": subset_size,
        "per_label_image_counts": label_counts,
        "no_finding_only_images": no_finding_only,
        "urgent_label_counts": {label: label_counts[label] for label in urgent_labels},
        "seed": seed,
    }
    return subset, composition


def generate_prediction(
    image: Any,
    *,
    model: Any,
    processor: Any,
    cxr_auditor: Any,
    max_new_tokens: int,
) -> tuple[str, list[Any], bool]:
    """Generate and tolerantly parse one image's grounded findings, matching production.

    Builds the production grounding prompt, runs a single greedy generation through
    the same model seam the Space uses, and parses the raw completion with the same
    production tolerant parser (``schema.extract_finding_list``, which routes a
    truncated or degenerate array through ``schema.salvage_finding_list`` to recover
    its complete leading findings before declaring a parse failure). A single greedy
    generation captures production's FIRST attempt: production's first attempt is also
    greedy with the same base prompt and the same salvaging parser, so the recovered
    findings here equal production's first-attempt result. Production additionally
    escalates on a parse failure (a corrective-suffix greedy retry, then sampling) for
    the residual cases salvage cannot rescue; the eval does not re-generate, so a
    ``parse_ok`` of ``False`` here marks exactly those residual unsalvageable cases,
    and the raw text is still captured for the per-sample dump.

    Args:
        image: The chest X-ray as an RGB ``PIL.Image``.
        model: The loaded vision-language model.
        processor: The matching processor.
        cxr_auditor: The shipped package namespace.
        max_new_tokens: The greedy decode budget.

    Returns:
        A tuple ``(raw_text, predicted_findings, parse_ok)``. On a parse failure the
        predicted list is empty and ``parse_ok`` is ``False``.
    """
    prompts = cxr_auditor["prompts"]
    schema = cxr_auditor["schema"]
    inference = cxr_auditor["inference"]

    # Match the production max-new-tokens through the module constant the seam reads.
    inference.DEFAULT_MAX_NEW_TOKENS = max_new_tokens
    generate_fn = inference.make_generate_fn(model, processor, image)
    raw_text = generate_fn(prompts.build_image_grounding_prompt())
    try:
        grounded = schema.extract_finding_list(raw_text)
    except schema.SchemaParseError:
        return raw_text, [], False
    predicted = inference.grounded_dicts_to_image_findings(grounded)
    return raw_text, predicted, True


def mean_iou_on_matched(
    cases: list[Any],
    cxr_auditor: Any,
    *,
    iou_threshold: float,
) -> tuple[float | None, int]:
    """Mean IoU of greedily matched same-label predicted/ground-truth box pairs.

    Boxes are grouped per canonical finding label (the harness grouping helper) and
    matched within label by the harness greedy matcher at ``iou_threshold``; the IoU
    of every matched pair is averaged. This reuses the harness primitives rather than
    re-deriving matching or IoU.

    Args:
        cases: The per-image ``ImageEvalCase`` objects.
        cxr_auditor: The shipped package namespace.
        iou_threshold: Minimum IoU for a pair to count as matched.

    Returns:
        A tuple ``(mean_iou, n_pairs)``; ``mean_iou`` is ``None`` when no pair
        matched at the threshold.
    """
    metrics = cxr_auditor["metrics"]
    run_eval = cxr_auditor["run_eval"]
    findings_module = cxr_auditor["findings"]
    boxes_by_finding = run_eval._boxes_by_finding  # harness grouping helper (not a metric)

    matched_ious: list[float] = []
    for label in findings_module.POSITIVE_FINDINGS:
        pred_boxes: list[Any] = []
        gt_boxes: list[Any] = []
        for case in cases:
            pred_boxes.extend(boxes_by_finding(case.predicted)[label])
            gt_boxes.extend(boxes_by_finding(case.expected)[label])
        matches, _, _ = metrics.match_boxes(pred_boxes, gt_boxes, iou_threshold)
        matched_ious.extend(iou for _, _, iou in matches)

    if not matched_ious:
        return None, 0
    return sum(matched_ious) / len(matched_ious), len(matched_ious)


def evaluate_model(
    model_id: str,
    subset: list[dict[str, Any]],
    cxr_auditor: Any,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Load one model, score it on the subset, free the GPU, and return its metrics.

    Args:
        model_id: The merged model repo id to evaluate.
        subset: The shared held-out subset records.
        cxr_auditor: The shipped package namespace.
        max_new_tokens: The greedy decode budget per generation.

    Returns:
        A dict with the model id, its metrics block, and the per-sample dump rows.
    """
    torch = importlib.import_module("torch")
    inference = cxr_auditor["inference"]
    run_eval = cxr_auditor["run_eval"]
    findings_module = cxr_auditor["findings"]

    print(f"Loading model {model_id} ...", flush=True)
    model, processor = inference.load_model(model_id)

    cases: list[Any] = []
    per_sample: list[dict[str, Any]] = []
    parse_failures = 0
    for position, record in enumerate(subset):
        raw_text, predicted, parse_ok = generate_prediction(
            record["image"],
            model=model,
            processor=processor,
            cxr_auditor=cxr_auditor,
            max_new_tokens=max_new_tokens,
        )
        if not parse_ok:
            parse_failures += 1
        cases.append(
            run_eval.ImageEvalCase(
                image_id=f"{record['hash'][:16]}",
                predicted=predicted,
                expected=record["gt_findings"],
            )
        )
        per_sample.append(
            {
                "model": model_id,
                "pool_index": record["index"],
                "image_hash": record["hash"],
                "parse_ok": parse_ok,
                "raw_text": raw_text,
                "predicted": [finding.model_dump() for finding in predicted],
                "expected": [finding.model_dump() for finding in record["gt_findings"]],
            }
        )
        if (position + 1) % 25 == 0:
            print(f"  {model_id}: scored {position + 1}/{len(subset)} images", flush=True)

    report = run_eval.build_image_report(run_eval.ImageEvalInput(cases=cases))

    mean_iou_03, pairs_03 = mean_iou_on_matched(cases, cxr_auditor, iou_threshold=0.3)
    mean_iou_any, pairs_any = mean_iou_on_matched(cases, cxr_auditor, iou_threshold=1e-9)

    presence = report["presence"]["per_finding"]
    urgent_recall = {
        label: {
            "recall": presence[label]["recall"],
            "true_positives": presence[label]["true_positives"],
            "false_negatives": presence[label]["false_negatives"],
        }
        for label in sorted(findings_module.URGENT_WHITELIST)
    }

    metrics_block: dict[str, Any] = {
        "n_images": len(subset),
        "parse_failures": parse_failures,
        "presence": report["presence"],
        "localization_iou_0.3": report["localization"]["0.3"],
        "localization_iou_0.5": report["localization"]["0.5"],
        "mean_iou_matched_at_0.3": {"mean_iou": mean_iou_03, "n_pairs": pairs_03},
        "mean_iou_any_overlap": {"mean_iou": mean_iou_any, "n_pairs": pairs_any},
        "urgent_recall": urgent_recall,
    }

    print(f"Finished {model_id}: {parse_failures} parse failure(s) over {len(subset)} images.", flush=True)

    # Free the GPU before the next model is loaded.
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    importlib.import_module("gc").collect()

    return {"model_id": model_id, "metrics": metrics_block, "per_sample": per_sample}


def upload_results(
    *,
    scripts_repo: str,
    results_prefix: str,
    results_basename: str,
    summary: dict[str, Any],
    per_sample: list[dict[str, Any]],
    token: str,
    timestamp: str,
) -> dict[str, str]:
    """Write the summary and per-sample dump to the scripts dataset repo.

    Each file is written twice: a stable ``*_latest.json`` name a consumer can
    fetch unconditionally, and a timestamped copy that preserves history across reruns.
    The ``results_basename`` infix lets a distinct run (for example the union run)
    write to its own filenames without clobbering an earlier run's results.

    Args:
        scripts_repo: The dataset repo that receives the results.
        results_prefix: The path prefix inside the repo for result files.
        results_basename: The infix used in the result filenames
            (``summary_<basename>_latest.json`` and the per-sample / history copies).
        summary: The summary report object.
        per_sample: The per-sample dump rows (including raw parse failures).
        token: The Hugging Face write token.
        timestamp: A UTC timestamp string used in the history filenames.

    Returns:
        A mapping of logical name to the repo path each file was written to.
    """
    hub = importlib.import_module("huggingface_hub")
    api = hub.HfApi(token=token)

    summary_bytes = json.dumps(summary, indent=2).encode("utf-8")
    per_sample_bytes = json.dumps({"per_sample": per_sample}, indent=2).encode("utf-8")

    targets = {
        "summary_latest": f"{results_prefix}/summary_{results_basename}_latest.json",
        "summary_history": f"{results_prefix}/summary_{results_basename}_{timestamp}.json",
        "per_sample_latest": f"{results_prefix}/per_sample_{results_basename}_latest.json",
        "per_sample_history": f"{results_prefix}/per_sample_{results_basename}_{timestamp}.json",
    }
    payloads = {
        "summary_latest": summary_bytes,
        "summary_history": summary_bytes,
        "per_sample_latest": per_sample_bytes,
        "per_sample_history": per_sample_bytes,
    }
    for name, path_in_repo in targets.items():
        api.upload_file(
            path_or_fileobj=payloads[name],
            path_in_repo=path_in_repo,
            repo_id=scripts_repo,
            repo_type="dataset",
            commit_message=f"Add {results_basename} eval {name} ({timestamp})",
        )
        print(f"Uploaded {name} -> {scripts_repo}:{path_in_repo}", flush=True)
    return targets


def main(argv: list[str] | None = None) -> None:
    """Run the full two-model held-out evaluation and publish the results.

    Args:
        argv: Optional explicit argument vector (for testing). Defaults to
            ``sys.argv[1:]`` when ``None``.
    """
    args = build_arg_parser().parse_args(argv)
    models = args.models if args.models else list(DEFAULT_MODELS)

    hf_token = os.environ.get("HF_TOKEN")
    if not args.no_upload and not hf_token:
        raise SystemExit("HF_TOKEN is not present in the environment; pass it as a Jobs secret or use --no-upload")

    torch = importlib.import_module("torch")
    # Fail fast if no GPU is visible, before any download or paid compute.
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device is visible to torch; aborting before paid compute. "
            "The torch wheel likely mismatches the node's GPU driver."
        )

    # Ship and import the cxr_auditor package from the scripts repo.
    import_shipped_package(args.scripts_repo, hf_token)
    cxr_auditor = {
        "schema": importlib.import_module("cxr_auditor.schema"),
        "findings": importlib.import_module("cxr_auditor.findings"),
        "prompts": importlib.import_module("cxr_auditor.prompts"),
        "inference": importlib.import_module("cxr_auditor.inference"),
        "metrics": importlib.import_module("cxr_auditor.eval.metrics"),
        "run_eval": importlib.import_module("cxr_auditor.eval.run_eval"),
    }
    # Fail loudly if a stale, pre-salvage package was fetched: the eval must run the
    # CURRENT production parser whose ``extract_finding_list`` routes truncated and
    # degenerate arrays through ``salvage_finding_list``. A stale copy would silently
    # score salvageable outputs as parse failures, so this guard protects the result.
    if not hasattr(cxr_auditor["schema"], "salvage_finding_list"):
        raise SystemExit(
            "the shipped cxr_auditor.schema lacks salvage_finding_list; the scripts repo holds a stale "
            "pre-salvage package. Re-upload src/cxr_auditor into the scripts repo before evaluating."
        )
    print("Imported shipped cxr_auditor package (salvage parser present) from the scripts repo.", flush=True)

    datasets = importlib.import_module("datasets")

    pool_pairs = resolve_pool_pairs(args)
    print(f"Evaluating on the union of {len(pool_pairs)} held-out pool(s): {pool_pairs}", flush=True)

    # Build one held-out ground-truth record list per (pool, exclude) pair. Each
    # pair uses ITS OWN pool dataset for ground truth and ITS OWN paired exclude
    # split for the leakage filter, so every record is held out from both models.
    per_pool_records: list[list[dict[str, Any]]] = []
    per_pool_stats: list[dict[str, Any]] = []
    for pool_dataset, pool_split_name, exclude_dataset, exclude_split_name in pool_pairs:
        source_pool = f"{pool_dataset}:{pool_split_name}"

        exclusion_hashes: set[str] = set()
        if not args.skip_leakage_filter:
            print(f"Hashing {exclude_dataset}:{exclude_split_name} for leakage filtering of {source_pool} ...", flush=True)
            exclude_split = datasets.load_dataset(exclude_dataset, split=exclude_split_name, token=hf_token)
            exclusion_hashes = build_exclusion_hashes(exclude_split)
            print(f"  Exclusion set for {source_pool} has {len(exclusion_hashes)} image hashes.", flush=True)

        print(f"Loading evaluation pool {source_pool} ...", flush=True)
        pool_split = datasets.load_dataset(pool_dataset, split=pool_split_name, token=hf_token)
        pool_size = len(pool_split)
        records = build_ground_truth(
            pool_split,
            cxr_auditor,
            exclusion_hashes=exclusion_hashes,
            source_pool=source_pool,
        )
        dropped = pool_size - len(records)
        print(f"  {source_pool}: {pool_size} images; {dropped} dropped (leakage/parse); {len(records)} held-out.", flush=True)
        per_pool_records.append(records)
        per_pool_stats.append(
            {
                "pool_dataset": source_pool,
                "exclude_dataset": f"{exclude_dataset}:{exclude_split_name}",
                "pool_size": pool_size,
                "dropped_from_pool": dropped,
                "held_out_candidates": len(records),
            }
        )

    # Union the per-pool record lists and deduplicate by image content hash so an
    # image in more than one validation split is scored once (first pool wins) and
    # is held out from both models by construction.
    union_records, union_audit = union_pool_records(per_pool_records, cxr_auditor)
    print(
        f"Union: {union_audit['union_total_candidates']} held-out candidates "
        f"({union_audit['cross_pool_duplicates_removed']} cross-pool duplicates removed).",
        flush=True,
    )

    subset, composition = stratify_subset(union_records, cxr_auditor, subset_size=args.subset_size, seed=args.seed)
    if args.max_eval_samples and args.max_eval_samples > 0:
        subset = subset[: args.max_eval_samples]
        composition["total_images"] = len(subset)
        composition["capped_for_smoke"] = args.max_eval_samples
    composition["leakage_filter_applied"] = not args.skip_leakage_filter
    composition["pools"] = per_pool_stats
    composition["union_audit"] = union_audit
    print(f"Stratified subset: {composition}", flush=True)

    model_results = [evaluate_model(model_id, subset, cxr_auditor, max_new_tokens=args.max_new_tokens) for model_id in models]

    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pool_specs = [stat["pool_dataset"] for stat in per_pool_stats]
    summary: dict[str, Any] = {
        "evaluation": "cxr-auditor v1-vs-v2 held-out grounding metrics (union of both held-out splits)",
        "generated_at_utc": timestamp,
        "ground_truth_source": (
            "Per-pool assistant targets: each image's ground truth comes from its own source pool "
            f"({', '.join(pool_specs)}); the same per-image ground truth is used for both models."
        ),
        "ground_truth_caveat": (
            "The evaluation subset is the UNION of two held-out pools that use DIFFERENT ground-truth "
            "conventions. Pool cxr-sft retains same-region opacity+nodule double-labels; pool cxr-sft-v2 is "
            "deduplicated to the single specific label (the dedup resolves an opacity/nodule OVERLAP to nodule). "
            "Because that dedup keeps nodule_mass present, nodule_mass and pneumothorax presence are CONSISTENT "
            "across both conventions, so urgent recall on the union is sound. Generic lung_opacity_consolidation "
            "recall, however, mixes the two conventions and should be read with care: a lower v2 opacity recall "
            "can come partly from this ground-truth difference rather than a real localization regression. See "
            "subset_composition.union_audit for the per-pool candidate and urgent counts."
        ),
        "decoding": "greedy (do_sample=False), bf16, SDPA, matching production serving",
        "parsing": (
            "production tolerant parse: schema.extract_finding_list (routes truncated/degenerate arrays through "
            "schema.salvage_finding_list) + inference.grounded_dicts_to_image_findings"
        ),
        "subset_composition": composition,
        "models": [{"model_id": result["model_id"], "metrics": result["metrics"]} for result in model_results],
    }

    per_sample_all: list[dict[str, Any]] = []
    for result in model_results:
        per_sample_all.extend(result["per_sample"])

    print("SUMMARY:\n" + json.dumps(summary, indent=2), flush=True)

    if args.no_upload:
        print("[--no-upload] Skipping Hub upload of results.", flush=True)
        return

    assert hf_token is not None  # guarded at entry when uploading
    paths = upload_results(
        scripts_repo=args.scripts_repo,
        results_prefix=args.results_prefix,
        results_basename=args.results_basename,
        summary=summary,
        per_sample=per_sample_all,
        token=hf_token,
        timestamp=timestamp,
    )
    print(f"Results published to {args.scripts_repo} (dataset) under {args.results_prefix}/: {paths}", flush=True)


if __name__ == "__main__":
    main()
