"""
Unsloth FastVisionModel QLoRA fine-tuning for the CXR Draft Auditor.

This is a self-contained training entry point for ``google/medgemma-1.5-4b-it``.
It loads the base vision-language model in 4-bit, attaches a QLoRA adapter over
the vision encoder, language model, attention modules, and MLP modules, and runs
a supervised fine-tune whose target is the constrained finding JSON (with
normalized bounding boxes) the auditor consumes downstream. After training it
merges the adapter to a 16-bit checkpoint, verifies the merge actually captured
the adapter weights, and optionally pushes the merged model to the Hugging Face
Hub for the published fine-tuned model.

The script runs unchanged on a Kaggle free GPU (T4 / P100, F16-only) and on
Hugging Face Jobs (a10g-large, bf16). It detects bf16 support at runtime and
selects the appropriate mixed-precision mode. All identifiers, paths, and
hyperparameters are command-line arguments; the Hub token is read only from the
environment (``HF_TOKEN``), never hard-coded.

The training data is provided in exactly one of two ways:

- ``--dataset-jsonl`` (local / Kaggle): a local SFT JSON Lines file whose
  ``image_path`` fields are opened from disk under ``--image-root``.
- ``--dataset-hub-id`` (Hugging Face Jobs): a published Hugging Face Dataset repo
  (built by ``scripts/push_corpus_to_hub.py``) whose ``images`` column embeds the
  pixels, so it loads inside an ephemeral Jobs container that cannot see the local
  ``data/`` tree.

Heavy dependencies (torch, unsloth, trl, datasets) are imported lazily inside
``main`` so that importing this module for a syntax check or for documentation
does not require a GPU stack to be installed.

Input dataset
-------------
The training set is a JSON Lines file produced by the dataset builder
(``sft_dataset.py``). Each line is one example with this shape::

    {
      "image_path": "relative/or/absolute/path/to/cxr.png",
      "messages": [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "<image-grounding prompt text>"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "[{\"label\": \"pleural_effusion\", \"box_2d\": [0.62, 0.08, 0.94, 0.40], ...}]"}
        ]}
      ]
    }

``image_path`` is a filesystem path resolved relative to ``--image-root`` (default:
the JSONL file's own directory). The assistant turn carries the constrained
finding JSON list exactly as the auditor expects it (the serialization parseable
by ``cxr_auditor.schema.extract_finding_list``); the data collator masks the loss
to that assistant turn only, so the model learns to emit the JSON and not to
reproduce the prompt or the image tokens.

Usage
-----
Local / Kaggle (images on disk)::

    python train/train_medgemma_lora.py \
        --dataset-jsonl data/sft/train.jsonl \
        --image-root data \
        --output-dir outputs/medgemma-cxr-lora \
        --merged-dir outputs/medgemma-cxr-merged \
        --push-to-hub --hub-model-id your-username/medgemma-cxr-auditor

Hugging Face Jobs (images embedded in a published dataset)::

    python train/train_medgemma_lora.py \
        --dataset-hub-id your-username/cxr-sft \
        --output-dir outputs/medgemma-cxr-lora \
        --merged-dir outputs/medgemma-cxr-merged \
        --push-to-hub --hub-model-id your-username/medgemma-cxr-auditor

See ``train/kaggle_notebook.md`` and ``train/hf_jobs.md`` for the two
launch environments.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

# The fine-tuning stack (torch, unsloth, trl, datasets) is an optional extra and
# is loaded dynamically at call time via ``importlib.import_module``. This keeps
# the module importable for documentation and syntax checks without a GPU stack
# installed, and the dynamic import is the idiomatic representation of a
# genuinely optional dependency. ``Any`` is used for the values these modules
# return because their types are unavailable when the stack is absent.

# Base model identifier and the constrained-finding vocabulary live in the
# package so the training target matches what the auditor parses at inference.
DEFAULT_BASE_MODEL = "google/medgemma-1.5-4b-it"

# MedGemma boxing accuracy is resolution-sensitive: too few image tokens and the
# normalized box coordinates degrade. These bounds (in pixels, expressed as
# token-count proxies the processor understands) keep the chest X-ray at a
# resolution where the boxes stay faithful while bounding memory use. They are
# overridable from the command line.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Every tunable is exposed as a flag so the same script drives both the Kaggle
    and the Hugging Face Jobs launches without edits.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Unsloth FastVisionModel QLoRA fine-tune for MedGemma CXR grounding.",
    )

    # Model and data. The dataset is provided EITHER as a local SFT JSON Lines
    # file (--dataset-jsonl, the Kaggle / local path, images loaded from disk) OR
    # as a published Hugging Face Dataset repo id (--dataset-hub-id, the HF Jobs
    # path, images embedded in the dataset). Exactly one is required.
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model repo id to fine-tune.")
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Path to the SFT JSON Lines file produced by sft_dataset.py (local/Kaggle mode).",
    )
    parser.add_argument(
        "--dataset-hub-id",
        default=None,
        help=(
            "Hub dataset repo id with an embedded 'images' column and 'messages' "
            "(HF Jobs mode; built by scripts/push_corpus_to_hub.py)."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Split to load when --dataset-hub-id is used.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Optional held-out split name to load from --dataset-hub-id for periodic evaluation.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory image paths are resolved against. Defaults to the JSONL file's directory.",
    )
    parser.add_argument(
        "--eval-jsonl",
        default=None,
        help="Optional held-out JSON Lines file for periodic evaluation (local mode).",
    )

    # Output locations.
    parser.add_argument(
        "--output-dir",
        default="outputs/medgemma-cxr-lora",
        help="Trainer output directory (adapter checkpoints, logs).",
    )
    parser.add_argument(
        "--merged-dir",
        default="outputs/medgemma-cxr-merged",
        help="Directory the merged 16-bit model is written to.",
    )

    # QLoRA / adapter hyperparameters.
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Model max sequence length for loading (not used to truncate vision sequences during training).",
    )

    # Processor resolution bounds (bbox-accuracy sensitive).
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Processor min image pixels.")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Processor max image pixels.")

    # Optimization.
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs (1-3 recommended).")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, overrides --epochs and trains for a fixed step count.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging interval in steps.")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval in steps.")

    # Hub publishing. The token is read from the environment only.
    parser.add_argument("--push-to-hub", action="store_true", help="Push the merged 16-bit model to the Hub.")
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Target Hub repo id (username/name). Required when --push-to-hub is set.",
    )
    parser.add_argument("--hub-private", action="store_true", help="Create the Hub repo as private.")

    # Monitoring.
    parser.add_argument(
        "--report-to",
        default="trackio",
        help="Trainer reporting backend ('trackio', 'tensorboard', or 'none').",
    )
    parser.add_argument("--run-name", default="medgemma-cxr-auditor", help="Descriptive run name for monitoring.")

    # Merge-verification controls.
    parser.add_argument(
        "--skip-merge-verify",
        action="store_true",
        help="Skip the post-merge adapter-took-effect check. Not recommended.",
    )
    parser.add_argument(
        "--merge-verify-tolerance",
        type=float,
        default=1e-6,
        help="Minimum mean absolute weight delta between base and merged to consider the merge non-trivial.",
    )
    return parser


def load_sft_dataset(jsonl_path: Path, image_root: Path) -> Any:
    """Load the SFT JSON Lines file into an in-memory dataset of vision chat rows.

    Each output row has the ``{"images": [PIL.Image], "messages": [...]}`` shape
    that ``UnslothVisionDataCollator`` consumes. The image referenced by each
    line is opened (and converted to RGB, since chest X-rays are commonly
    single-channel) and substituted for the JSONL's path string.

    Args:
        jsonl_path: Path to the SFT JSON Lines file.
        image_root: Directory relative image paths are resolved against.

    Returns:
        A ``datasets.Dataset`` ready to hand to the trainer.

    Raises:
        FileNotFoundError: If the JSONL file or any referenced image is missing.
        ValueError: If a line is malformed or carries no usable image reference.
    """
    datasets = importlib.import_module("datasets")
    image_module = importlib.import_module("PIL.Image")

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"dataset JSONL not found: {jsonl_path}")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid JSON: {exc}") from exc

            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{jsonl_path}:{line_number}: missing or empty 'messages'")

            image_ref = record.get("image_path")
            if not isinstance(image_ref, str) or not image_ref:
                raise ValueError(f"{jsonl_path}:{line_number}: missing 'image_path'")

            image_path = (image_root / image_ref).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"{jsonl_path}:{line_number}: image not found: {image_path}")

            image = image_module.open(image_path).convert("RGB")
            rows.append({"images": [image], "messages": messages})

    if not rows:
        raise ValueError(f"no usable examples found in {jsonl_path}")
    return datasets.Dataset.from_list(rows)


def load_hub_dataset(hub_dataset_id: str, split: str) -> Any:
    """Load a published SFT dataset split from the Hub.

    The dataset (built by ``scripts/push_corpus_to_hub.py``) already embeds the
    chest X-ray pixels in an ``images`` column alongside the chat ``messages``, so
    no on-disk image staging is needed. This is the Hugging Face Jobs path: the
    ephemeral container cannot see the local ``data/`` tree, so the images travel
    inside the dataset. Each embedded image is normalized to RGB so the row shape
    matches what the local JSON Lines path produces.

    Args:
        hub_dataset_id: The Hub dataset repo id (``namespace/name``).
        split: The split to load (for example ``"train"``).

    Returns:
        A ``datasets.Dataset`` of ``{"images": [PIL.Image], "messages": [...]}``
        rows ready for the vision collator.

    Raises:
        ValueError: If the loaded dataset lacks the ``images`` or ``messages``
            columns the collator requires.
    """
    datasets = importlib.import_module("datasets")
    dataset = datasets.load_dataset(hub_dataset_id, split=split)

    column_names = getattr(dataset, "column_names", None)
    if column_names is not None and not ({"images", "messages"} <= set(column_names)):
        raise ValueError(
            f"Hub dataset {hub_dataset_id!r} split {split!r} must have 'images' and 'messages' columns; "
            f"found {sorted(column_names)}. Rebuild it with scripts/push_corpus_to_hub.py."
        )

    def _to_rgb(row: dict[str, Any]) -> dict[str, Any]:
        images = [image.convert("RGB") for image in row["images"]]
        return {"images": images, "messages": row["messages"]}

    return dataset.map(_to_rgb)


def select_precision() -> tuple[bool, bool]:
    """Select the mixed-precision mode for the available GPU.

    Kaggle's free T4 and P100 GPUs do not support bf16, so the script falls back
    to fp16 there; Ampere and newer (a10g, the ZeroGPU Blackwell) support bf16.

    Returns:
        A ``(use_bf16, use_fp16)`` pair with exactly one element true.
    """
    torch = importlib.import_module("torch")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def verify_merge_captured_adapter(
    base_model_id: str,
    merged_dir: Path,
    max_seq_length: int,
    tolerance: float,
) -> None:
    """Verify that the merged 16-bit checkpoint differs from the base model.

    Unsloth has a known failure mode on vision models where
    ``save_pretrained_merged`` silently writes the base weights without folding
    the adapter in. This reloads both the base model and the merged checkpoint
    and asserts that at least one shared parameter tensor differs by more than
    ``tolerance`` in mean absolute value. A trained adapter always perturbs the
    weights, so an all-equal result means the merge did not take effect.

    Args:
        base_model_id: The base model repo id that was fine-tuned.
        merged_dir: Directory holding the merged 16-bit checkpoint.
        max_seq_length: Max sequence length used when loading both models.
        tolerance: Minimum mean absolute delta to accept the merge as non-trivial.

    Raises:
        RuntimeError: If no shared parameter differs by more than ``tolerance``,
            indicating the adapter was not merged in.
    """
    torch = importlib.import_module("torch")
    unsloth = importlib.import_module("unsloth")
    fast_vision_model = unsloth.FastVisionModel

    base_model, _ = fast_vision_model.from_pretrained(
        base_model_id,
        load_in_4bit=False,
        dtype=torch.float16,
        max_seq_length=max_seq_length,
    )
    merged_model, _ = fast_vision_model.from_pretrained(
        str(merged_dir),
        load_in_4bit=False,
        dtype=torch.float16,
        max_seq_length=max_seq_length,
    )

    base_params = dict(base_model.named_parameters())
    max_delta = 0.0
    for name, merged_param in merged_model.named_parameters():
        base_param = base_params.get(name)
        if base_param is None or base_param.shape != merged_param.shape:
            continue
        delta = (merged_param.detach().float() - base_param.detach().float()).abs().mean().item()
        max_delta = max(max_delta, delta)
        if delta > tolerance:
            print(f"Merge verification passed: parameter {name!r} changed by mean |delta| {delta:.3e}.")
            return

    raise RuntimeError(
        "Merge verification FAILED: the merged checkpoint is identical to the base model "
        f"(max mean |delta| {max_delta:.3e} <= tolerance {tolerance:.3e}). The LoRA adapter was "
        "not folded in (Unsloth vision silent-merge failure). Do not publish this checkpoint; "
        "re-run the merge or save the adapter and merge with PEFT instead."
    )


def _resolve_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    """Resolve the train and optional eval datasets from the parsed arguments.

    Dispatches on which dataset source was provided: ``--dataset-jsonl`` opens
    images from the local ``data/`` tree (local / Kaggle mode); ``--dataset-hub-id``
    loads a published dataset whose images are already embedded (HF Jobs mode).
    Exactly one source is provided (validated in ``main``).

    Args:
        args: The parsed command-line arguments.

    Returns:
        A ``(train_dataset, eval_dataset)`` pair; ``eval_dataset`` is ``None`` when
        no held-out evaluation source was given.
    """
    if args.dataset_jsonl:
        dataset_jsonl = Path(args.dataset_jsonl).expanduser().resolve()
        image_root = Path(args.image_root).expanduser().resolve() if args.image_root else dataset_jsonl.parent
        train_dataset = load_sft_dataset(dataset_jsonl, image_root)
        eval_dataset = None
        if args.eval_jsonl:
            eval_jsonl = Path(args.eval_jsonl).expanduser().resolve()
            eval_dataset = load_sft_dataset(eval_jsonl, image_root)
        return train_dataset, eval_dataset

    train_dataset = load_hub_dataset(args.dataset_hub_id, args.dataset_split)
    eval_dataset = None
    if args.eval_split:
        eval_dataset = load_hub_dataset(args.dataset_hub_id, args.eval_split)
    return train_dataset, eval_dataset


def main(argv: list[str] | None = None) -> None:
    """Run the full fine-tune, merge, verify, and optional publish pipeline.

    Args:
        argv: Optional explicit argument vector (for testing). Defaults to
            ``sys.argv[1:]`` when ``None``.
    """
    args = build_arg_parser().parse_args(argv)

    # Validate the configuration and resolve paths before importing the heavy
    # stack, so misconfiguration fails fast with a clear message rather than only
    # after the GPU libraries load.
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--hub-model-id is required when --push-to-hub is set")

    if bool(args.dataset_jsonl) == bool(args.dataset_hub_id):
        raise SystemExit("provide exactly one of --dataset-jsonl (local) or --dataset-hub-id (HF Jobs)")

    hf_token = os.environ.get("HF_TOKEN")
    if args.push_to_hub and not hf_token:
        raise SystemExit("--push-to-hub set but HF_TOKEN is not present in the environment")

    trl = importlib.import_module("trl")
    unsloth = importlib.import_module("unsloth")
    unsloth_trainer = importlib.import_module("unsloth.trainer")
    sft_config_cls = trl.SFTConfig
    sft_trainer_cls = trl.SFTTrainer
    fast_vision_model = unsloth.FastVisionModel
    vision_data_collator_cls = unsloth_trainer.UnslothVisionDataCollator

    use_bf16, use_fp16 = select_precision()

    # Load the base VLM in 4-bit with Unsloth's fused-kernel gradient checkpointing.
    model, processor = fast_vision_model.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_length,
    )

    # The processor's image-token budget governs box-coordinate fidelity. Set the
    # bounds on whichever sub-processor (image processor / tokenizer) exposes them.
    _configure_processor_pixels(processor, args.min_pixels, args.max_pixels)

    # Attach the QLoRA adapter over every modality block.
    model = fast_vision_model.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
    )

    train_dataset, eval_dataset = _resolve_datasets(args)

    fast_vision_model.for_training(model)

    config = sft_config_cls(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
        # Vision-specific TRL settings: do not let TRL pre-tokenize or drop the
        # image/message columns, and do not truncate (which would cut image tokens).
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
    )

    trainer = sft_trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        data_collator=vision_data_collator_cls(model, processor),
        args=config,
    )

    trainer.train()

    # Merge the adapter into the base weights at 16-bit precision. The merged
    # checkpoint is what gets served (no quantization at inference on ZeroGPU
    # large) and published.
    merged_dir = Path(args.merged_dir).expanduser().resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), processor, save_method="merged_16bit")
    print(f"Merged 16-bit model written to {merged_dir}.")

    if not args.skip_merge_verify:
        verify_merge_captured_adapter(
            base_model_id=args.base_model,
            merged_dir=merged_dir,
            max_seq_length=args.max_seq_length,
            tolerance=args.merge_verify_tolerance,
        )

    if args.push_to_hub:
        model.push_to_hub_merged(
            args.hub_model_id,
            processor,
            save_method="merged_16bit",
            token=hf_token,
            private=args.hub_private,
        )
        print(f"Published merged 16-bit model to https://huggingface.co/{args.hub_model_id}.")


def _configure_processor_pixels(processor: Any, min_pixels: int, max_pixels: int) -> None:
    """Set the image-token pixel bounds on whichever component exposes them.

    Different VLM processors expose the resolution budget on the processor
    itself, on a nested ``image_processor``, or not at all. This sets the bounds
    where they exist and silently leaves processors that do not expose them
    untouched, so the call is safe across model families.

    Args:
        processor: The model processor returned by ``FastVisionModel.from_pretrained``.
        min_pixels: Lower bound on the image-token pixel budget.
        max_pixels: Upper bound on the image-token pixel budget.
    """
    for target in (processor, getattr(processor, "image_processor", None)):
        if target is None:
            continue
        if hasattr(target, "min_pixels"):
            target.min_pixels = min_pixels
        if hasattr(target, "max_pixels"):
            target.max_pixels = max_pixels
        size = getattr(target, "size", None)
        if isinstance(size, dict) and ("shortest_edge" in size or "longest_edge" in size):
            size["shortest_edge"] = min_pixels
            size["longest_edge"] = max_pixels


if __name__ == "__main__":
    main()
