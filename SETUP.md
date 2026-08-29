# SETUP - User-Action Checklist

This is the definitive list of actions that need your credentials, your payment, or your acceptance of a license. The repository itself never downloads datasets, accepts gated licenses, installs GPU packages, logs in to any service, or hits the network. Everything below is yours to do.

I wrote this as a reproduce-from-scratch guide. I have already built and deployed the project: the Space is `build-small-hackathon/cxr-draft-auditor` (the hosted demo has since ended and the Space is paused; its code stays public) and the served fine-tuned grounding model is `alex-feeel/medgemma-cxr-auditor-v2` (the v2 derivative, which superseded v1 `alex-feeel/medgemma-cxr-auditor` as served on 2026-06-12 after winning the held-out head-to-head). The served pipeline decomposes into three steps: the fine-tuned MedGemma grounds the image on the GPU, NVIDIA Nemotron-3 Nano 4B (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, run on the GPU through transformers) parses the draft impression into the same labels, and a deterministic comparator over the two label sets does the only judging. Nemotron's native architecture is `nemotron_h` (a Mamba2-Transformer hybrid) that transformers supports directly (transformers >= 5.3), so the Space needs no extra runtime and no CUDA build; both models run on the GPU, so a full audit takes roughly 15 to 30 seconds. The training steps below cover only the MedGemma fine-tune; the Nemotron draft parser uses an off-the-shelf published model and needs no training. The as-built training path was Hugging Face Jobs with TRL `SFTTrainer` + PEFT QLoRA (see `train/hf_jobs.md`); the Kaggle + Unsloth path described in Step 7 is the documented alternative, not the path that produced the served model. Reuse the real repo ids below when you want to point at the as-built artifacts, or substitute your own namespace when reproducing from scratch.

Each step is marked BLOCKER (the project cannot ship without it) or UPSIDE (nice to have, not on the critical path). Do the BLOCKER steps in order. The UPSIDE step can run in parallel from day one.

Conventions: commands assume a Unix-style shell. On Windows, use a bash shell (Git Bash or WSL) and forward slashes. Set `HF_TOKEN` in your environment rather than passing `--token` on the command line.

## Step 1 - Accept the MedGemma HAI-DEF gated license (BLOCKER)

The base model `google/medgemma-1.5-4b-it` is gated behind a self-serve Health AI Developer Foundations (HAI-DEF) click-through license. It is NOT PhysioNet-gated; a free Hugging Face account that accepts the terms gets immediate access.

1. Sign in to Hugging Face.
2. Open `https://huggingface.co/google/medgemma-1.5-4b-it`.
3. Read and accept the HAI-DEF Terms of Use on that page: `https://developers.google.com/health-ai-developer-foundations/terms`.
4. Confirm access is granted (the gate banner on the model page clears).

You must accept this before training or before the Space can load the base weights.

## Step 2 - Create a Hugging Face token and a Kaggle API token (BLOCKER)

You need a write-scoped Hugging Face token (for publishing the model and the Space) and a Kaggle API token (for downloading VinDr).

Hugging Face token:

1. Create a write token at `https://huggingface.co/settings/tokens`.
2. Log in the CLI and export the token:

   ```bash
   hf auth login
   export HF_TOKEN=hf_xxx
   hf auth whoami
   ```

3. In the `whoami` output, note `isPro` and `canPay`; they gate the hardware choices in Step 3.

Kaggle API token:

1. At `https://www.kaggle.com/settings`, under API, click "Create New Token". This downloads `kaggle.json`.
2. Place it where the Kaggle CLI expects it and lock down permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   pip install --upgrade kaggle
   kaggle --version
   ```

## Step 3 - Get Hugging Face PRO to host ZeroGPU (BLOCKER)

Hosting your own ZeroGPU Space requires the Space creator to be on a PRO (or Team or Enterprise) plan. PRO is $9 per month. Free accounts can only use other people's ZeroGPU Spaces, not host one.

1. Upgrade at `https://huggingface.co/settings/billing` (or the PRO subscription page).
2. Confirm `hf auth whoami` now reports `isPro: true`.
3. The hackathon references a $20 Hugging Face credit; if you have it, it can offset this cost.

## Step 4 - Create the Gradio Space under the build-small-hackathon org (BLOCKER)

The Space MUST live under the `build-small-hackathon` org namespace, or the submission does not count. You must already be a member of that org (registration closed June 3, 2026).

```bash
hf repos create build-small-hackathon/cxr-draft-auditor \
    --type space --space-sdk gradio \
    --flavor zero-a10g \
    --public --exist-ok
```

Notes:

- `--flavor zero-a10g` is the identifier for ZeroGPU. The actual backing GPU is the NVIDIA RTX Pro 6000 Blackwell; `large` (48 GB) fits the merged 4B at bf16 with no quantization.
- The Space README frontmatter must set `python_version: "3.12"`. The Space `requirements.txt` must NOT list `gradio`, `spaces`, or `huggingface_hub` (the platform manages them); it MUST list `transformers` (>= 5.3, so it can load the `nemotron_h` hybrid architecture natively), `accelerate`, and `torchvision`, and leave `torch` unpinned. No `llama.cpp` runtime is needed: both the MedGemma grounding model and the Nemotron draft parser run on the GPU through transformers.
- If you are ever not on PRO, you can still create a `cpu-basic` Space, push the ZeroGPU code, and request a community grant; PRO is the direct path.

## Step 5 - Download the data and build the SFT corpus (BLOCKER for the parts you train and evaluate on)

All sources below avoid PhysioNet. Read and accept each source's license before downloading. Several are non-commercial research only. The VinDr Data Use Agreement is non-commercial research only even when a downstream mirror is tagged CC0.

### Recommended path: the helper scripts

`scripts/download_data.py` wraps every download below into one credential-aware command and lays the data out under `data/` in exactly the structure the pipeline loaders expect. It imports the `kaggle` / `huggingface_hub` clients only when it actually fetches, so a dry run needs neither. Install the clients first (`uv pip install kaggle huggingface_hub`), confirm your tokens from Step 2 are in place, then:

```bash
python scripts/download_data.py --dry-run                # preview the plan, no network
python scripts/download_data.py                          # fetch all datasets
python scripts/download_data.py vqa chestxdet            # fetch a subset
```

The script writes to `data/vindr/vinbigdata-512-image-dataset/`, `data/vqa/data_v1.json`, `data/chestxdet/`, `data/nih/BBox_List_2017.csv`, and `data/open_i/`. After the data is on disk, build the supervised-fine-tuning corpus with `scripts/prepare_sft.py`, which runs the pure-logic pipeline (no torch, no network) and writes `data/sft/train.jsonl`, the training target for Step 7:

```bash
python scripts/prepare_sft.py --dry-run                  # preview the plan
python scripts/prepare_sft.py                            # write data/sft/train.jsonl
```

`prepare_sft.py` normalizes every VinDr box by the ORIGINAL image dimensions (from the VinDr meta CSV or the VQA annotations), not by the resized mirror, so the boxes project cleanly onto any mirror resolution; when a meta CSV is absent it still emits each finding label with a null box rather than dropping it. Override the auto-located inputs with `--box-csv`, `--meta-csv`, and `--vqa-json` if your mirror names them differently.

The manual per-dataset commands below remain available if you prefer to fetch a single source by hand; they reach the same data, but you must then place the files where the loaders expect them (the layout the script produces above).

### VinDr-CXR (primary boxes) - pick one route

Route A, the resized PNG mirror, needs no competition entry and is the fastest:

```bash
kaggle datasets download -d awsaf49/vinbigdata-512-image-dataset
unzip -q vinbigdata-512-image-dataset.zip -d data/vindr_512
```

Route B, the original competition (use Late Submission, accept the competition rules first on the competition page):

```bash
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection
unzip -q vinbigdata-chest-xray-abnormalities-detection.zip -d data/vindr_competition
```

The box annotations are `train.csv` with columns `image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max`.

### VinDr-CXR-VQA annotations (join by image_id) - no images

```bash
hf download faizan711/VinDR-CXR-VQA --repo-type dataset --local-dir data/vindr_vqa
```

This ships `data_v1.json` only. Join it to the VinDr pixels by `image_id` (a 32-character hex filename). The `gt_location` boxes are in original full-resolution pixel space; rescale them per image when pairing with a resized mirror.

### ChestX-Det (second box source)

```bash
hf download natealberti/ChestX-Det --repo-type dataset --local-dir data/chestxdet
```

Annotations are Apache-2.0.

### NIH ChestX-ray14 with BBox_List_2017.csv (held-out box eval)

```bash
hf download alkzar90/NIH-Chest-X-ray-dataset --repo-type dataset --local-dir data/nih
```

Box coordinates are absolute XYWH at 1024 px.

### IU-Xray / Open-i (real reports, parser realism validation; no boxes)

```bash
hf download ykumards/open-i --repo-type dataset --local-dir data/iu_xray
```

License is CC BY-NC-ND 4.0; used only to validate the parser.

## Step 6 - Fire the PadChest-GR access request (UPSIDE, day one, non-blocking)

PadChest-GR is the only dataset with the full image-plus-real-report-plus-box triple, but it is request-gated and not guaranteed to clear quickly. Request it on day one so it is available if it clears, but do NOT put it on the critical path; the synthetic-draft loop does not depend on it.

1. Go to the BIMCV PadChest-GR access page (`https://bimcv.cipf.es/`, PadChest-GR).
2. Submit the data-use request and wait for credentials.
3. If it clears in time, use it as a real-triple bonus; if not, ship without it.

## Step 7 - Run training (BLOCKER for the Fine-tuned badge)

Train the QLoRA fine-tune. Two paths exist; the served model `alex-feeel/medgemma-cxr-auditor-v2` (and its predecessor v1 `alex-feeel/medgemma-cxr-auditor`) were produced by the Hugging Face Jobs + TRL path.

As-built path (recommended): Hugging Face Jobs with TRL `SFTTrainer` + PEFT QLoRA + bitsandbytes on an `a100-large` (1x A100, 80 GB) GPU. The exact runbook lives in `train/hf_jobs.md`: it publishes the SFT corpus as a private dataset (`alex-feeel/cxr-sft`), submits the Jobs run, and pushes the LoRA adapter as a safety copy before merging the adapter into a clean bf16 base and verifying the merge. It needs a paid plan (PRO is sufficient) and is the path that produced the served model.

Alternative path: the free Kaggle GPU tier with Unsloth `FastVisionModel`. The exact notebook steps live in `train/kaggle_notebook.md` (attach a T4 or P100 GPU, add your `HF_TOKEN` as a Kaggle secret, run the QLoRA fine-tune at rank 16, alpha 16, learning rate around 2e-4, merge to 16-bit with `save_method='merged_16bit'`, then verify the merge captured the adapters before publishing).

Either way, VERIFY the merge captured the adapters before publishing (vision-model merges can silently drop them). Run an inference sanity check on the merged model and confirm the outputs differ from the base model on a held-out example.

## Step 8 - Publish the merged model to Hugging Face (BLOCKER for the Fine-tuned badge)

Publish the verified merged 16-bit model so the Space can load it and the badge criterion ("your app uses a fine-tuned model you've published on Hugging Face") is met. The as-built served model is `alex-feeel/medgemma-cxr-auditor-v2` (the v2 derivative; the example commands below use the v1 namespace `alex-feeel/medgemma-cxr-auditor` since the upload procedure is identical for either). Substitute your own namespace when reproducing from scratch. (The `train/hf_jobs.md` path pushes the merged model directly from the Jobs run, so the manual upload below applies when you train elsewhere, for example the Kaggle path.)

1. Create the model repo and upload the merged weights, the model card, and the Notice file:

   ```bash
   hf repos create alex-feeel/medgemma-cxr-auditor --type model --exist-ok
   hf upload alex-feeel/medgemma-cxr-auditor ./merged_model . --repo-type model
   hf upload alex-feeel/medgemma-cxr-auditor docs/MODEL_CARD.md README.md --repo-type model
   hf upload alex-feeel/medgemma-cxr-auditor NOTICE NOTICE --repo-type model
   ```

2. HAI-DEF compliance for the published derivative (required): the model is a MedGemma "Model Derivative". The model card (`docs/MODEL_CARD.md`) already carries the compliance block; confirm the published repo includes the `NOTICE` file, the modified-file notice, the propagated Section 3.2 use restrictions and Prohibited Use Policy, the link to the HAI-DEF Terms of Use, and the no-Google-endorsement line.

## Step 9 - Set the HF_MODEL_ID Space secret and deploy (BLOCKER)

Point the Space at the published model and push the app. `app.py` reads the model id from the `HF_MODEL_ID` environment variable and falls back to a placeholder only when it is unset, so the model id is supplied as a Space secret rather than hard-coded into the source.

1. Set `HF_MODEL_ID` as a Space secret to the published model id from Step 8, then restart the Space so it picks up the new value. The live Space currently points at the v2 derivative `alex-feeel/medgemma-cxr-auditor-v2` (which superseded v1 on 2026-06-12); substitute your own published model id when reproducing from scratch:

   ```bash
   hf spaces secrets add build-small-hackathon/cxr-draft-auditor \
       --secrets HF_MODEL_ID=alex-feeel/medgemma-cxr-auditor-v2
   hf spaces restart build-small-hackathon/cxr-draft-auditor
   ```

2. Push the app and its files to the Space:

   ```bash
   hf upload build-small-hackathon/cxr-draft-auditor app.py app.py --repo-type space
   hf upload build-small-hackathon/cxr-draft-auditor README.md requirements.txt . --repo-type space
   ```

3. Watch the build and the runtime logs, and confirm the model loads cleanly:

   ```bash
   hf spaces logs build-small-hackathon/cxr-draft-auditor --build --follow
   hf spaces logs build-small-hackathon/cxr-draft-auditor --follow
   hf spaces info build-small-hackathon/cxr-draft-auditor --expand runtime
   ```

4. Smoke-test the live endpoint with one MISSING case, one UNSUPPORTED case, and one negative control, and confirm the flags and boxes render. The draft cases (MISSING and UNSUPPORTED) exercise the Nemotron draft parser, so confirm the draft-parsed labels (including any explicit denials) come through and that a draft that cannot be parsed degrades to an image-only pass with a visible note rather than erroring. The served app loads the Nemotron BF16 weights (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) from the Hub at startup and runs them on the GPU through transformers; the gated MedGemma base is what requires the HF token, so no additional secret is needed for the public Nemotron weights.

## Quick blocker-vs-upside summary

| Step | What | Type |
| --- | --- | --- |
| 1 | Accept MedGemma HAI-DEF license | BLOCKER |
| 2 | Hugging Face token + Kaggle token | BLOCKER |
| 3 | Hugging Face PRO for ZeroGPU hosting | BLOCKER |
| 4 | Create Space under build-small-hackathon org | BLOCKER |
| 5 | Download data and build the SFT corpus (download_data.py, prepare_sft.py) | BLOCKER |
| 6 | Request PadChest-GR | UPSIDE |
| 7 | Train (HF Jobs/TRL as-built, or Kaggle/Unsloth), merge, verify merge | BLOCKER |
| 8 | Publish merged model with HAI-DEF compliance | BLOCKER |
| 9 | Set the HF_MODEL_ID Space secret and deploy the Space | BLOCKER |
