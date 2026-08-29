# CXR Draft Auditor

> RESEARCH / EDUCATIONAL QA ONLY. This project is NOT a medical device, NOT a diagnostic tool, and NOT a substitute for a qualified radiologist. It must NEVER be used for clinical decision-making, screening, or patient care. See [DISCLAIMER.md](DISCLAIMER.md).

A small multimodal quality-assurance tool I built around a fine-tuned MedGemma plus NVIDIA Nemotron. It takes a chest X-ray plus an optional draft radiology impression, uses the fine-tuned MedGemma to extract a constrained set of findings WITH image-grounded bounding-box evidence, uses NVIDIA Nemotron-3 Nano 4B (on the GPU through transformers) to parse the draft into the same label space, and runs a deterministic comparator that flags MISSING findings, UNSUPPORTED claims, and URGENT review flags.

The point of the tool is the audit loop, not a verdict: it surfaces where a human draft and the image appear to disagree, and shows the image evidence so a person can look again.

## What it does

1. Image to grounded findings. The fine-tuned MedGemma vision-language model (on the GPU) emits a constrained JSON list of findings over a fixed 6-finding label space, each with a normalized bounding box.
2. Draft to labels. NVIDIA Nemotron-3 Nano 4B, run on the GPU through transformers, parses the draft impression into the same 6-finding label space (asserted vs denied), keeping the verbatim draft span for each label. It reasons briefly before emitting the labels, which improves extraction accuracy, and the reasoning trace is stripped before parsing. If the draft cannot be parsed, the audit degrades to an image-only pass with a visible note.
3. Deterministic comparison. A pure-logic comparator (no model) flags:
   - MISSING: present in the image findings, absent or denied in the draft.
   - UNSUPPORTED: asserted in the draft, absent from the image findings.
   - URGENT: any flagged finding on the urgent whitelist (pneumothorax and nodule/mass) is surfaced for radiologist review.

## Canonical finding set

`pleural_effusion`, `pneumothorax`, `lung_opacity_consolidation`, `nodule_mass`, `cardiomegaly`, `no_finding`.

## Architecture

| Layer | Module | Dependencies |
| --- | --- | --- |
| Canonical labels and dataset mappings | `cxr_auditor.findings` | stdlib only |
| Output schema, tolerant JSON parsing, box conversions | `cxr_auditor.schema` | numpy, pydantic |
| Pinned model prompt templates | `cxr_auditor.prompts` | stdlib only |
| Deterministic audit comparator (MISSING / UNSUPPORTED / URGENT) | `cxr_auditor.comparator` | stdlib, numpy, pydantic |
| Quantitative evaluation metrics (presence, localization, audit flags) | `cxr_auditor.eval.metrics` | numpy |

The pure-logic core (findings, schema, prompts, comparator, metrics) depends only on stdlib, numpy, and pydantic, so it is unit-testable WITHOUT torch, transformers, or gradio. The heavy stacks are optional extras (`[vision]`, `[train]`, `[app]`) and are imported lazily.

## Models

Grounding model (image to findings): `google/medgemma-1.5-4b-it` (HAI-DEF license), fine-tuned. MedGemma natively emits CXR bounding boxes as a JSON list of `{label, box_2d: [y0, x0, y1, x1]}` normalized to `[0, 1]` with `(y0, x0)` top-left and `(y1, x1)` bottom-right. The canonical box format constant is `normalized_y0x0y1x1`. It runs on the GPU.

Draft parser (draft to labels): `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (NVIDIA Nemotron Open Model License), run on the GPU through transformers in bf16. Its native `nemotron_h` (Mamba2-Transformer hybrid) architecture is supported by transformers directly (>= 5.3), so no extra runtime and no CUDA build are needed. A grounding-fine-tuned MedGemma parses free-text drafts unreliably, whereas a dedicated instruction-following text model parses the draft cleanly, including explicit denials with the verbatim span; the model reasons before emitting the labels and the reasoning trace is stripped before parsing. The deterministic comparator over the two label sets remains the only judgment layer.

The fine-tuned MedGemma weights I trained are a MedGemma Model Derivative governed by the Health AI Developer Foundations (HAI-DEF) Terms of Use, NOT Apache-2.0. The Apache-2.0 license covers only this repository's own application code (see `pyproject.toml` and `LICENSE`); it does not relicense the model weights. See the model card and [NOTICE](NOTICE) for the HAI-DEF compliance terms.

## Links

The hosted demo ran for the Build Small Hackathon (June 2026) and has since ended; the Space is paused, and its copy of the app code stays public.

- Write-up: https://www.alexfeel.info/projects/cxr-draft-auditor/
- Space (hosted the demo): https://huggingface.co/spaces/build-small-hackathon/cxr-draft-auditor
- Fine-tuned model (was served): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2 (v2; supersedes v1 `alex-feeel/medgemma-cxr-auditor` as of 2026-06-12)
- Open traces dataset: https://huggingface.co/datasets/build-small-hackathon/cxr-draft-auditor-traces
- Hugging Face collection: https://huggingface.co/collections/alex-feeel/cxr-draft-auditor-6a92d0128d80b609bb5c8ce2

Built by Aleksandr Filippov for the Build Small Hackathon (June 2026).

## Install

```bash
uv sync                      # core only
uv sync --extra dev          # core plus test tooling
uv sync --extra vision       # plus the inference stack (torch, transformers)
uv sync --extra train        # plus the fine-tuning stack (unsloth, trl, peft)
uv sync --extra app          # plus gradio (local runs only)
```

## Test

```bash
uv run pytest
```

Tests use tiny in-repo synthetic fixtures only. No real datasets, no network, and no GPU are required or used.

## Data and credentials

This repository never downloads datasets, accepts gated licenses, installs GPU packages, or logs in to any service. Those are user-only actions documented separately in `SETUP.md`.
