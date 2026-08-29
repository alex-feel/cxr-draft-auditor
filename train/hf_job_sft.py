# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "trl>=0.21",
#     "peft>=0.13",
#     "transformers>=4.56",
#     "datasets>=3.0",
#     "bitsandbytes>=0.45.1",
#     "accelerate>=0.34",
#     "huggingface_hub>=0.26",
#     "pillow>=10.3",
#     "trackio",
# ]
#
# # Pin torch (and its torchvision side-car) to a CUDA 12.6 build from the PyTorch
# # wheel index. Without this, uv resolves the latest torch, which since torch 2.11
# # ships CUDA 13 (cu13) wheels by default; a cu13 build needs a CUDA 13 driver and
# # cannot initialize the GPU on a node whose driver is CUDA 12.x (e.g. a100-large
# # reports driver 12090 / CUDA 12.9), which silently forces everything onto the CPU
# # and makes the bitsandbytes 8-bit optimizer fail. A cu126 build runs on ANY CUDA
# # 12.x or newer driver via CUDA minor-version backward compatibility, so this pin
# # is flavor-agnostic (works on a100-large, a10g-large, rtx-pro-6000, ...). torch
# # 2.8.0 has no cu130 wheel at all, so it can never drift to CUDA 13. Both torch and
# # torchvision MUST be routed to the index: a side-car left on PyPI would pull a
# # mismatched CUDA build and ImportError at load. `explicit = true` confines the
# # index to the packages named in [tool.uv.sources]; everything else resolves from
# # PyPI. `hf jobs uv run` calls `uv run`, which honors these PEP 723 [tool.uv]
# # sections (unlike the `--torch-backend` flag, which uv ignores for `uv run`).
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# torchvision = { index = "pytorch-cu126" }
# ///
"""Hugging Face Jobs QLoRA fine-tune of MedGemma for CXR grounding (TRL backend).

This is the Hugging Face Jobs entry point. It is a self-contained UV script (its
dependencies are declared inline in the PEP 723 header above), so it runs inside
an ephemeral Jobs container with nothing pre-installed and nothing pre-staged.

It is the TRL counterpart to the Unsloth ``train/train_medgemma_lora.py`` used on
Kaggle / locally. Plain TRL ``SFTTrainer`` + PEFT QLoRA + bitsandbytes is chosen
for the Jobs environment because that stack installs cleanly on the freshly built
UV environment every Jobs run creates, whereas Unsloth's custom CUDA kernels are a
common cause of container setup failures; and on a Jobs cloud GPU (a10g-large 24
GB, or a larger flavor) the run is not memory-starved enough to need Unsloth's VRAM
savings. TRL has first-class vision-language SFT: it auto-selects
``DataCollatorForVisionLanguageModeling`` for a VLM and ``max_length=None`` keeps
the full sequence so image tokens are never truncated. (TRL does not yet support
assistant-only loss masking for VLMs, so the loss covers the full sequence; the
grounding prompt is constant across examples, so this costs little.)

Data delivery
-------------
The Jobs container cannot see the local ``data/`` tree, so the chest X-ray pixels
are delivered inside a published Hugging Face Dataset. ``scripts/push_corpus_to_hub.py``
builds that dataset: an ``images`` column embeds each PNG alongside the chat
``messages``. This script loads it with ``datasets.load_dataset(<hub id>)`` -- the
pixels travel in the dataset, no loose files.

Pipeline
--------
1. Load the published dataset split (embedded ``images`` + ``messages``).
2. Load ``google/medgemma-1.5-4b-it`` in 4-bit (NF4) via bitsandbytes.
3. Attach a QLoRA adapter (rank 16) over the language and attention/MLP modules.
4. Fine-tune with TRL ``SFTTrainer`` (vision collator, assistant-only loss).
5. Merge the adapter into 16-bit weights with PEFT ``merge_and_unload``.
6. Verify the merged checkpoint differs non-trivially from the base model (so a
   silently failed merge can never be published).
7. Push the merged 16-bit model and its processor to the Hub.

The heavy / GPU dependencies are imported lazily through ``importlib`` so the
module imports for a syntax / type check without a GPU stack present; at runtime
inside the Jobs container the imports resolve normally. The Hub token is read only
from the ``HF_TOKEN`` environment variable (passed as a Jobs secret), never
hard-coded.

Usage (inside a Jobs container; flags are passed as ``script_args``)::

    python hf_job_sft.py \
        --dataset-hub-id your-username/cxr-sft \
        --hub-model-id your-username/medgemma-cxr-auditor \
        --epochs 2 --lora-r 16

See ``train/hf_jobs.md`` for the full launch runbook.
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

# Default identifiers. The base model and QLoRA rank match the published
# training run; the dataset and target model ids are always provided on the command
# line so this script is namespace-agnostic.
DEFAULT_BASE_MODEL = "google/medgemma-1.5-4b-it"

# LoRA target modules: the attention projections (q/k/v/o) and MLP projections
# (gate/up/down). PEFT matches these names across the whole model, so the adapter
# attaches to both the Gemma-3 language tower and the vision tower's attention --
# light vision adaptation that helps grounded localization -- while the bulk of
# each tower stays frozen, keeping the 4-bit QLoRA run within a single 24 GB GPU.
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Every tunable is a flag so the runbook can override hyperparameters via the
    Jobs ``script_args`` without editing this file.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="TRL QLoRA fine-tune of MedGemma for CXR grounding on Hugging Face Jobs.",
    )

    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model repo id to fine-tune.")
    parser.add_argument(
        "--dataset-hub-id",
        required=True,
        help="Published SFT dataset repo id (embedded 'images' + 'messages'); built by push_corpus_to_hub.py.",
    )
    parser.add_argument("--dataset-split", default="train", help="Training split to load.")
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Optional held-out split for periodic evaluation.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=-1,
        help="If > 0, train on only the first N examples (fast smoke runs / bounding run time).",
    )

    parser.add_argument(
        "--output-dir",
        default="medgemma-cxr-lora",
        help="Trainer output directory (adapter checkpoints, logs).",
    )
    parser.add_argument(
        "--merged-dir",
        default="medgemma-cxr-merged",
        help="Directory the merged 16-bit model is written to.",
    )

    # QLoRA hyperparameters.
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    # Optimization.
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, overrides --epochs and trains for a fixed step count (use for smoke runs).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging interval in steps.")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval in steps.")

    # Hub publishing. The token is read from the environment only.
    parser.add_argument(
        "--hub-model-id",
        required=True,
        help="Target Hub repo id (username/name) for the merged 16-bit model.",
    )
    parser.add_argument("--hub-private", action="store_true", help="Create the model Hub repo as private.")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Train, merge, and verify but skip the Hub push (for a local dry test).",
    )

    # Monitoring.
    parser.add_argument(
        "--report-to",
        default="trackio",
        help="Trainer reporting backend ('trackio', 'tensorboard', or 'none').",
    )
    parser.add_argument("--run-name", default="medgemma-cxr-auditor", help="Descriptive run name for monitoring.")
    parser.add_argument(
        "--trackio-project",
        default="cxr-draft-auditor",
        help="Trackio project name to group runs under.",
    )

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


def normalize_dataset_images(dataset: Any) -> Any:
    """Normalize the embedded ``images`` column to RGB.

    Chest X-rays are commonly single-channel; converting to RGB matches what the
    processor and collator expect and keeps every row uniform. The dataset must
    already carry the ``images`` and ``messages`` columns the vision collator
    consumes (the published corpus does).

    Args:
        dataset: A loaded ``datasets.Dataset`` with ``images`` and ``messages``.

    Returns:
        The dataset with each embedded image converted to RGB.

    Raises:
        ValueError: If the dataset lacks the required columns.
    """
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None and not ({"images", "messages"} <= set(column_names)):
        raise ValueError(
            f"dataset must have 'images' and 'messages' columns; found {sorted(column_names)}. "
            "Rebuild it with scripts/push_corpus_to_hub.py."
        )

    def _to_rgb(row: dict[str, Any]) -> dict[str, Any]:
        images = [image.convert("RGB") for image in row["images"]]
        return {"images": images, "messages": row["messages"]}

    return dataset.map(_to_rgb)


def select_precision() -> tuple[bool, bool]:
    """Select the mixed-precision mode for the available GPU.

    bf16 is preferred on Ampere and newer (a10g, the ZeroGPU Blackwell); the
    script falls back to fp16 on older GPUs that lack bf16.

    Returns:
        A ``(use_bf16, use_fp16)`` pair with exactly one element true.
    """
    torch = importlib.import_module("torch")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def build_quantization_config() -> Any:
    """Build the bitsandbytes 4-bit (NF4) quantization config for QLoRA.

    Returns:
        A ``transformers.BitsAndBytesConfig`` for double-quantized NF4 4-bit
        loading with a bf16 compute dtype.
    """
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def verify_merge_captured_adapter(
    base_model_id: str,
    merged_dir: str,
    tolerance: float,
) -> None:
    """Verify that the merged 16-bit checkpoint differs from the base model.

    Reloads both the base model and the merged checkpoint and asserts that at
    least one shared parameter tensor differs by more than ``tolerance`` in mean
    absolute value. A trained adapter always perturbs the weights, so an all-equal
    result means the merge did not fold the adapter in and the checkpoint must not
    be published. This guard is kept regardless of training backend.

    Args:
        base_model_id: The base model repo id that was fine-tuned.
        merged_dir: Directory holding the merged 16-bit checkpoint.
        tolerance: Minimum mean absolute delta to accept the merge as non-trivial.

    Raises:
        RuntimeError: If no shared parameter differs by more than ``tolerance``.
    """
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    auto_model = transformers.AutoModelForImageTextToText

    base_model = auto_model.from_pretrained(base_model_id, dtype=torch.float16)
    merged_model = auto_model.from_pretrained(merged_dir, dtype=torch.float16)

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
        "not folded in. Do not publish this checkpoint; re-run the merge."
    )


def main(argv: list[str] | None = None) -> None:
    """Run the full Jobs fine-tune, merge, verify, and publish pipeline.

    Args:
        argv: Optional explicit argument vector (for testing). Defaults to
            ``sys.argv[1:]`` when ``None``.
    """
    args = build_arg_parser().parse_args(argv)

    hf_token = os.environ.get("HF_TOKEN")
    if not args.no_push and not hf_token:
        raise SystemExit("HF_TOKEN is not present in the environment; pass it as a Jobs secret or use --no-push")

    torch = importlib.import_module("torch")
    # Fail fast if no GPU is visible. A cu126-pinned torch runs on any CUDA 12.x
    # driver, but if the wheel ever mismatches the node driver, torch silently
    # falls back to CPU and the bitsandbytes 8-bit optimizer then dies many
    # minutes later (after the image map) with an opaque "tensors not on a GPU"
    # error. Aborting here turns that into a minute-one failure on paid compute.
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device is visible to torch; aborting before paid compute. "
            "The torch wheel likely mismatches the node's GPU driver."
        )

    datasets = importlib.import_module("datasets")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    trl = importlib.import_module("trl")

    auto_processor = transformers.AutoProcessor
    auto_model = transformers.AutoModelForImageTextToText
    lora_config_cls = peft.LoraConfig
    prepare_for_kbit = peft.prepare_model_for_kbit_training
    sft_config_cls = trl.SFTConfig
    sft_trainer_cls = trl.SFTTrainer

    use_bf16, use_fp16 = select_precision()

    # Group runs under the project the Trainer's own trackio backend reads from
    # the environment. report_to="trackio" makes the Trainer initialize the
    # trackio run itself, so no manual trackio.init() is needed (a second init
    # would race the Space creation); setting TRACKIO_PROJECT here gives that
    # single init the project grouping without any code path that could abort a
    # paid run if trackio is unreachable.
    if args.report_to == "trackio":
        os.environ.setdefault("TRACKIO_PROJECT", args.trackio_project)

    # Load the base VLM 4-bit (NF4) for QLoRA, plus its processor. Pin every
    # module to cuda:0 with device_map={"": 0}: on a single-GPU node "auto" can
    # offload layers to CPU/meta when it under-estimates free VRAM, and
    # offloaded/meta params receive no gradients while the bitsandbytes 8-bit
    # optimizer requires all tensors on the GPU. Explicit single-GPU placement is
    # the canonical QLoRA setting and removes that fragility.
    model = auto_model.from_pretrained(
        args.base_model,
        quantization_config=build_quantization_config(),
        device_map={"": 0},
        token=hf_token,
    )
    processor = auto_processor.from_pretrained(args.base_model, token=hf_token)
    # use_cache is incompatible with gradient checkpointing (the cache is never
    # used under checkpointing and the combination warns and can break the
    # recomputed activations); disable it before enabling checkpointing.
    model.config.use_cache = False
    # Non-reentrant checkpointing is required for QLoRA: the reentrant variant
    # combined with frozen quantized embeddings can silently drop gradients on
    # vision-language models. Route the flag through the single GC-enable path so
    # checkpointing is configured once, consistently.
    model = prepare_for_kbit(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = lora_config_cls(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(DEFAULT_LORA_TARGET_MODULES),
    )

    # The published dataset embeds the images; normalize them to RGB. Subset
    # (before the RGB map) when --max-train-samples is set, so a smoke run maps
    # only a handful of images instead of the whole corpus.
    raw_train = datasets.load_dataset(args.dataset_hub_id, split=args.dataset_split)
    if args.max_train_samples and args.max_train_samples > 0:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    train_dataset = normalize_dataset_images(raw_train)
    eval_dataset = None
    if args.eval_split:
        eval_dataset = normalize_dataset_images(datasets.load_dataset(args.dataset_hub_id, split=args.eval_split))

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
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
        # Vision-language SFT: do not pre-tokenize or drop the image/message
        # columns and do not truncate (which would cut image tokens). TRL does not
        # yet support assistant-only loss masking for VLMs, so the loss covers the
        # full sequence; the grounding prompt is constant across examples and is
        # learned trivially, while the variable finding JSON carries the signal.
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        assistant_only_loss=False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
    )

    trainer = sft_trainer_cls(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()

    # Save the trained LoRA adapter, then merge it into a CLEAN 16-bit base.
    # Folding the adapter directly into the 4-bit (bitsandbytes) training model is
    # lossy and version-dependent; the canonical QLoRA merge reloads the base in
    # bf16, applies the saved adapter, and merges. The 4-bit training model is
    # freed first, and the merge runs on CPU so it never competes with the training
    # GPU's memory. The merged model is what the ZeroGPU Space serves at bf16.
    peft_model_cls = peft.PeftModel

    adapter_dir = f"{args.output_dir}-adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(args.merged_dir)

    # The trained adapter is the expensive, irreplaceable result of the run; the
    # merge, the verification reload, and the push are all cheap, repeatable
    # post-processing. Push the small adapter (and the processor) to its own Hub
    # repo BEFORE merging so that any downstream merge / verify / push fault can
    # be recovered from the saved adapter instead of forcing a full retrain.
    if not args.no_push:
        adapter_repo_id = f"{args.hub_model_id}-adapter"
        trainer.model.push_to_hub(adapter_repo_id, token=hf_token, private=args.hub_private)
        processor.push_to_hub(adapter_repo_id, token=hf_token, private=args.hub_private)
        print(f"Pushed trained LoRA adapter to https://huggingface.co/{adapter_repo_id} (pre-merge safety copy).")

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # low_cpu_mem_usage streams the base weights in instead of materializing a
    # second full copy, keeping the CPU-side merge peak low.
    base_for_merge = auto_model.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    merged = peft_model_cls.from_pretrained(base_for_merge, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(args.merged_dir, safe_serialization=True)
    print(f"Merged 16-bit model written to {args.merged_dir} (merged from a clean bf16 base).")

    if not args.skip_merge_verify:
        verify_merge_captured_adapter(
            base_model_id=args.base_model,
            merged_dir=args.merged_dir,
            tolerance=args.merge_verify_tolerance,
        )

    if not args.no_push:
        merged.push_to_hub(args.hub_model_id, token=hf_token, private=args.hub_private)
        processor.push_to_hub(args.hub_model_id, token=hf_token, private=args.hub_private)
        print(f"Published merged 16-bit model to https://huggingface.co/{args.hub_model_id}.")


if __name__ == "__main__":
    main()
