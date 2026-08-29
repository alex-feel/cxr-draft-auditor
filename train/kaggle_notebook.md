# Training MedGemma CXR LoRA on a Kaggle free GPU

This guide runs `train/train_medgemma_lora.py` on Kaggle's free GPU tier to produce the published fine-tuned model. Kaggle gives roughly 30 GPU-hours per week with a 9-hour session cap, which is enough for a 1-3 epoch QLoRA fine-tune of a 4B vision-language model.

This Kaggle path is the free-GPU alternative training backend (no Hugging Face PRO or paid Jobs quota required). It is NOT the path that produced the published model: the served `alex-feeel/medgemma-cxr-auditor` was trained on the Hugging Face Jobs + TRL path (`train/hf_jobs.md`). Use this Kaggle notebook when you lack a paid plan or prefer Unsloth.

## Prerequisites (USER actions, not automated)

1. A Kaggle account with phone verification (required to enable GPU and internet).
2. A Hugging Face account that has accepted the MedGemma / HAI-DEF gated-model terms for `google/medgemma-1.5-4b-it`. Visit the model page on Hugging Face and accept the license; access is self-serve (`gated: auto`).
3. A Hugging Face access token with write permission (write is needed only if you publish from inside the notebook; read is enough if you only download the base model and publish later from your workstation).
4. The SFT dataset prepared on the Kaggle side. The training script reads a JSON Lines file plus the chest X-ray PNGs the JSONL references. Attach the data as a Kaggle Dataset (preferred, persistent and within quota) or build it inside the session from the source datasets. The JSONL is produced by the data pipeline (`scripts/prepare_sft.py` / `sft_dataset.py`); see `SETUP.md` for how to build it and what each row contains.

## Step 1: Create the notebook and enable hardware

1. Create a new Kaggle Notebook (Code -> New Notebook).
2. Open the right-hand Settings panel.
3. Under Accelerator, select a GPU. On the free tier this is a T4 x2 or a P100. Both are F16-only (no bf16); the training script detects this at runtime and selects fp16 automatically, so no edits are needed.
4. Turn Internet ON (required to download the base model and the latest Unsloth wheel).
5. Leave Persistence on so the merged model survives between session restarts within the same notebook.

## Step 2: Store the Hugging Face token as a Kaggle secret

Do NOT paste the token into a code cell. Use Kaggle Secrets so the token never appears in the notebook source or output.

1. In the notebook, open Add-ons -> Secrets.
2. Add a secret named `HF_TOKEN` whose value is your Hugging Face write token.
3. Attach the secret to the notebook (toggle it on for this notebook).

Load it into the environment at the top of the notebook:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

The training script reads the token only from `os.environ["HF_TOKEN"]`; it never takes a token argument.

## Step 3: Install the latest Unsloth

Kaggle's base image lags behind, and the free T4/P100 GPUs require a current Unsloth for correct F16-only kernels. Install the latest wheel first.

```python
%%capture
import os
os.system("pip install --upgrade --no-cache-dir unsloth unsloth_zoo")
os.system("pip install --upgrade --no-cache-dir trl peft accelerate datasets")
```

Restart the kernel once after the install so the freshly installed Unsloth is the one imported (Unsloth patches transformers at import time).

## Step 4: Make the training script available

The script is a single file. Either add this repository as a Kaggle Dataset / GitHub source and reference the file directly, or paste the contents of `train/train_medgemma_lora.py` into a notebook cell that writes it to disk:

```python
from pathlib import Path

# If the repo is attached as a Kaggle Dataset at /kaggle/input/cxr-draft-auditor,
# copy the script; otherwise write it from a string you paste in.
src = Path("/kaggle/input/cxr-draft-auditor/train/train_medgemma_lora.py")
Path("train_medgemma_lora.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
```

## Step 5: Confirm the dataset paths

The script needs the JSONL path and the image root the JSONL paths resolve against. With the SFT data attached as a Kaggle Dataset:

```python
DATASET_JSONL = "/kaggle/input/cxr-sft/train.jsonl"
IMAGE_ROOT = "/kaggle/input/cxr-sft"          # PNG paths in the JSONL are relative to this
EVAL_JSONL = "/kaggle/input/cxr-sft/val.jsonl"  # optional
```

## Step 6: Run training in the background

Long fine-tunes outlast an interactive cell. Use Kaggle's Save Version -> Save & Run All (Commit) to run the notebook as a background batch job: it survives browser disconnects and runs until completion or the 9-hour cap. For interactive iteration, you can also launch the trainer as a detached subprocess, but the committed batch run is the reliable path for a full epoch sweep.

The training invocation (set `--epochs` to 1 first to confirm the loop, then 2-3 for the real run):

```python
import subprocess, sys

cmd = [
    sys.executable, "train_medgemma_lora.py",
    "--dataset-jsonl", DATASET_JSONL,
    "--image-root", IMAGE_ROOT,
    "--eval-jsonl", EVAL_JSONL,
    "--output-dir", "/kaggle/working/medgemma-cxr-lora",
    "--merged-dir", "/kaggle/working/medgemma-cxr-merged",
    "--epochs", "1",
    "--batch-size", "1",
    "--grad-accum", "8",
    "--learning-rate", "2e-4",
    "--lora-r", "16",
    "--lora-alpha", "16",
    "--report-to", "none",   # Trackio needs a Hub Space; "none" keeps the free run self-contained
]
subprocess.run(cmd, check=True)
```

To publish the merged model directly from the notebook, add `--push-to-hub --hub-model-id your-username/medgemma-cxr-auditor`. The script merges to 16-bit, verifies the adapter actually folded in (guarding the Unsloth vision silent-merge failure), and only then pushes.

## Step 7: Watch the clock and the quota

- The 9-hour session cap is hard. A 1-3 epoch QLoRA on a few thousand examples fits comfortably, but checkpoints are written to `--output-dir` every `--save-steps` so a capped run can resume.
- The free GPU quota is roughly 30 GPU-hours per week. Budget the epoch count accordingly; start with 1 epoch and a small `--max-steps` smoke run before committing the full sweep.
- F16-only hardware: do not pass any bf16 flag. The script's `select_precision()` returns fp16 on T4/P100 automatically.
- Out-of-memory: keep `--batch-size 1`; raise `--grad-accum` to keep the effective batch size up. Lower `--max-pixels` (for example to `768 * 28 * 28`) only if memory forces it, accepting some loss of bounding-box resolution.

## Step 8: Collect the output

The merged 16-bit model is written to `/kaggle/working/medgemma-cxr-merged`. If you did not push from the notebook, download the `/kaggle/working` output from the committed version's Output tab, or publish it later with the Hugging Face CLI (`hf upload your-username/medgemma-cxr-auditor /path/to/medgemma-cxr-merged`). The merged model is what the Gradio / ZeroGPU Space serves at bf16 with no quantization.
