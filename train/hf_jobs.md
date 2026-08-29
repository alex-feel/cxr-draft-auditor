# Training MedGemma CXR LoRA on Hugging Face Jobs

> Historical / reference runbook. This records how the served model was actually trained for the Build Small Hackathon (June 2026); the hosted demo has since ended. The `alex-feeel/cxr-sft*` corpus repositories named in the commands are private (VinDr-CXR data-use agreement; see the model cards) - substitute your own namespaces to reproduce.

This is the runbook for fine-tuning `google/medgemma-1.5-4b-it` (QLoRA) on Hugging Face Jobs cloud GPUs and publishing the merged 16-bit model that the Gradio / ZeroGPU Space serves. It is the managed-cloud alternative to the Kaggle free-GPU path (`train/kaggle_notebook.md`).

The Jobs container is ephemeral and cannot see your local `data/` tree, and nothing survives unless it is pushed to the Hub. Both problems are solved the Hub-native way: the chest X-ray pixels are delivered inside a published dataset (an embedded image column), and the trained model is pushed back to the Hub. Every step below is a USER action; this repository authors the scripts and the commands, it does not execute any network, upload, or training action on your behalf.

## Training backend decision: TRL, not Unsloth, on Jobs

The Jobs script `train/hf_job_sft.py` uses plain TRL `SFTTrainer` + PEFT QLoRA + bitsandbytes, not Unsloth. The Unsloth `train/train_medgemma_lora.py` remains the Kaggle / local path (it also gained a `--dataset-hub-id` mode, so it can consume the same published dataset if you prefer it). The reasons, in order:

1. TRL + PEFT + bitsandbytes is the stack Hugging Face Jobs is built around, and it installs cleanly on the fresh UV environment a Jobs run builds every time. Unsloth ships custom CUDA kernels plus `unsloth_zoo` with tight `torch` / `transformers` / `triton` version pinning, which is a frequent cause of container setup failures on Jobs.
2. The Unsloth value proposition is roughly 2x speed and roughly 60% less VRAM, which matters when a GPU is memory-starved. The 4-bit base weights are small (roughly 3 GB), so the speed/VRAM saving is not the deciding factor; the memory pressure in this run comes from the loss logits, addressed by the GPU flavor (see Hardware below), not by the training framework.
3. TRL has first-class vision-language SFT: it auto-selects `DataCollatorForVisionLanguageModeling` for a vision-language model and `max_length=None` keeps the full sequence so image tokens are never truncated. TRL does not yet support assistant-only loss masking for VLMs, so the script sets `assistant_only_loss=False` and trains on the full sequence; the grounding prompt is constant across examples, so the variable finding JSON still carries the learning signal.
4. The QLoRA merge is the canonical one: the script saves the trained adapter, reloads the base in bf16 (on CPU, off the training GPU), applies the adapter, and `merge_and_unload`s it into 16-bit weights -- it never folds the adapter into the lossy 4-bit training model. It then verifies the merged checkpoint differs non-trivially from the base before pushing, so a silently failed merge can never be published.

## Prerequisites (USER actions, not automated)

1. A Hugging Face account on a paid plan (Jobs are not available on the free tier; PRO is sufficient).
2. Acceptance of the MedGemma / HAI-DEF gated terms for `google/medgemma-1.5-4b-it` on the model page (self-serve, `gated: auto`).
3. A write-scoped Hugging Face token available locally so `$HF_TOKEN` resolves (`hf auth login`, or export `HF_TOKEN`).
4. The SFT corpus built on disk (`data/sft/train.jsonl`, or the balanced `data/sft/train.curated.jsonl` if you ran `scripts/curate_sft.py`). See `SETUP.md` for downloading the data and building the corpus.

## Step 1: Publish the SFT corpus as a private dataset

The Jobs container loads the training data with `load_dataset`, so the corpus and its pixels must live in a Hub dataset. `scripts/push_corpus_to_hub.py` reads the SFT JSONL, embeds each referenced PNG into a `datasets.Dataset` `images` column alongside the chat `messages`, and pushes it private. Inspect the plan first (no network, no heavy imports), then push:

```bash
# Preview: record count, distinct-image count, approximate embedded size.
python scripts/push_corpus_to_hub.py --hub-dataset-id alex-feeel/cxr-sft --dry-run

# Build and push the private dataset (reads HF_TOKEN from the environment).
python scripts/push_corpus_to_hub.py --hub-dataset-id alex-feeel/cxr-sft
```

The dataset defaults to `data/sft/train.curated.jsonl` when present, else `data/sft/train.jsonl`, with `--image-root data`. It is pushed private by default because the VinDr pixels are under a non-commercial DUA; do not pass `--public`.

If you ran `scripts/curate_sft.py` (which also writes `data/sft/val.curated.jsonl`) and want periodic evaluation during training, publish the validation corpus as a second split in the same dataset. `--split validation` adds the split to the existing dataset without replacing the `train` split:

```bash
python scripts/push_corpus_to_hub.py \
  --hub-dataset-id alex-feeel/cxr-sft \
  --jsonl data/sft/val.curated.jsonl \
  --split validation
```

Then pass `--eval-split validation` in Step 4 to evaluate on it. This is optional; omit it for a leaner run.

## Step 2: Make the Jobs script reachable by URL

`hf jobs uv run` takes a UV script as inline content or a URL, not a local path (the container has no access to your disk). Upload `train/hf_job_sft.py` to a Hub repo and use its `resolve` URL. To publish it as a script file:

```bash
hf repos create alex-feeel/cxr-auditor-scripts --type dataset
hf upload alex-feeel/cxr-auditor-scripts train/hf_job_sft.py hf_job_sft.py --repo-type dataset
# Reachable at:
# https://huggingface.co/datasets/alex-feeel/cxr-auditor-scripts/resolve/main/hf_job_sft.py
```

Re-run the `hf upload` whenever `train/hf_job_sft.py` changes: the Jobs container fetches the `resolve` URL, not your local file, so an edit that is not re-uploaded does not reach the run. The script pins `torch==2.8.0` + `torchvision==0.23.0` to the PyTorch `cu126` (CUDA 12.6) wheel index in its PEP 723 header; without that pin uv resolves the latest torch, which ships CUDA 13 wheels that cannot initialize the GPU on a CUDA 12.x driver (the node silently falls back to CPU and the 8-bit optimizer then fails). A `cu126` wheel runs on any CUDA 12.x-or-newer driver via minor-version backward compatibility, so the pin is flavor-agnostic.

## Step 3: Smoke-test the loop with a short run

Validate the whole path (container build, the `cu126` torch wheel resolving to a real GPU, dataset load, model load, a few real optimizer steps, merge, verify) cheaply before committing to the full run. `--max-steps` overrides the epoch count, `--max-train-samples` keeps the upfront image map to a handful of rows, and `--no-push` stops before the Hub push so the smoke run produces nothing to clean up. Run it on the same flavor as the full run so the smoke proves the torch/driver pairing on the exact target node.

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 1h \
  --secrets HF_TOKEN \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "https://huggingface.co/datasets/alex-feeel/cxr-auditor-scripts/resolve/main/hf_job_sft.py" \
  -- \
  --dataset-hub-id alex-feeel/cxr-sft \
  --hub-model-id alex-feeel/medgemma-cxr-auditor \
  --max-train-samples 64 \
  --max-steps 10 \
  --no-push
```

Confirm in the smoke logs: torch reports the GPU (no "No CUDA device is visible" abort and no CPU fallback), real optimizer steps advance 1..10 with a loss, no bitsandbytes "tensors not on the same GPU" error, and the run ends with `Merged 16-bit model written ...` and `Merge verification passed`. Only then launch the full run.

## Step 4: Launch the full fine-tune

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "https://huggingface.co/datasets/alex-feeel/cxr-auditor-scripts/resolve/main/hf_job_sft.py" \
  -- \
  --dataset-hub-id alex-feeel/cxr-sft \
  --hub-model-id alex-feeel/medgemma-cxr-auditor \
  --epochs 1 \
  --lora-r 16 \
  --lora-alpha 16 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accum 8 \
  --report-to trackio \
  --run-name medgemma-cxr-auditor
```

The default Jobs timeout is 30 minutes, far too short for a vision fine-tune, so `--timeout 3h` is set explicitly (it must exceed the expected run time plus buffer for the container build, model download, the merge, the merge-verification reload, and the Hub push). One epoch is used because the earlier run had already converged before reaching it (loss roughly 0.84, mean-token-accuracy roughly 0.96 at about three-quarters of an epoch), so a second epoch would roughly double the cost for little gain and risk overfitting. `--secrets HF_TOKEN` makes your local token available inside the container as `$HF_TOKEN`, which the script reads for the gated base-model download, the pre-merge adapter safety push, and the merged-model push. `--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces allocator fragmentation around the transient fp32 loss-logit spike (belt-and-suspenders on an 80 GB GPU, where the spike already fits with wide margin). All flags after `--` are passed to the script.

The script first pushes the trained LoRA adapter to `alex-feeel/medgemma-cxr-auditor-adapter` (a small safety copy) before it merges, verifies, and pushes the full merged model to `alex-feeel/medgemma-cxr-auditor`. If a later merge, verification, or push step fails, the adapter is already preserved on the Hub and the merge can be re-run from it without repeating training.

## Step 5: Monitor the run

- List jobs: `hf jobs ps`
- Follow logs: `hf jobs logs <job-id> --follow`
- Inspect a job: `hf jobs inspect <job-id>`

Watch for the `Merge verification passed` log line near the end: it confirms the LoRA adapter folded into the merged weights before the push. If you see `Merge verification FAILED`, the published model would be the unmodified base and the script aborts the push.

## Step 6: Point the Space at the resulting model

When the run finishes, the merged 16-bit model is at `https://huggingface.co/alex-feeel/medgemma-cxr-auditor`. The Space serves it at bf16 with no quantization. Set the model id as the Space's `HF_MODEL_ID` secret so the app loads your fine-tune instead of the base model:

```bash
hf spaces secrets add build-small-hackathon/cxr-draft-auditor \
  --secrets HF_MODEL_ID=alex-feeel/medgemma-cxr-auditor
hf spaces restart build-small-hackathon/cxr-draft-auditor
```

## Hardware and cost estimate

`a100-large` (one NVIDIA A100, 80 GB) is the recommended flavor. The 4-bit base weights, the LoRA adapter, the activations, and the 8-bit optimizer state together use well under half the card; the memory headroom is needed for a different reason. Because TRL trains the full sequence for vision-language models (assistant-only loss masking is not supported for VLMs, so no token is masked out), the Gemma-3 cross-entropy materializes the loss logits in fp32 over the full roughly 262000-token vocabulary, and the longest full-resolution image-plus-report sequences make that a multi-gigabyte transient spike at the loss step. On a 24 GB A10G that spike is what overflows memory near the end of an epoch; on the 80 GB A100 it fits with wide margin. The A100 supports bf16, which `select_precision()` selects automatically, and its host has 142 GB of RAM, far more than the CPU-side merge and verification reload need. The full SFT corpus is 15000 records, but it carries heavy three-radiologist box duplication and an 11215:3785 normal:positive imbalance; `scripts/curate_sft.py` balances it to roughly 7000 examples, which is the assumed size below.

| Item | Estimate |
| --- | --- |
| Flavor | a100-large (1x A100, 80 GB, bf16-capable, 142 GB host RAM) |
| Approximate rate | roughly 2.50 USD per GPU-hour |
| Container build + base-model download | 10 to 20 minutes |
| Fine-tune (1 epoch, roughly 7000 examples, 4-bit QLoRA) | 1.5 to 2.5 hours |
| Merge + verification reload + adapter and merged Hub push | 10 to 25 minutes |
| Total wall time | roughly 2 to 3 hours |
| Estimated total cost | roughly 5 to 8 USD |

Against the hackathon credit (20 USD) this leaves margin for one full run plus the short smoke run in Step 3. Keep `--batch-size 1` and rely on `--grad-accum 8` for the effective batch size; the per-step memory is dominated by the single-sequence loss logits, not by the batch, so raising the batch size would only increase pressure. If even more headroom is ever needed, `h200` (one NVIDIA H200, 141 GB) is the next single-GPU step up; the script needs no edits for a larger flavor, only the `--flavor` value changes.
