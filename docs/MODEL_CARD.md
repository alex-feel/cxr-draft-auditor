---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
base_model: google/medgemma-1.5-4b-it
tags:
  - chest-x-ray
  - radiology
  - vision-language-model
  - medgemma
  - object-detection
  - image-text-to-text
  - model-derivative
  - research
  - not-a-medical-device
pipeline_tag: image-text-to-text
library_name: transformers
---

# CXR Draft Auditor v1 - Fine-tuned MedGemma Derivative (superseded as served)

> RESEARCH / EDUCATIONAL QA ONLY. This model is NOT a medical device, NOT a diagnostic tool, and NOT a substitute for a qualified radiologist. It MUST NOT be used for clinical decision-making, screening, triage, or patient care. Its outputs are frequently wrong.

> SUPERSEDED AS SERVED. As of 2026-06-12, this v1 model has been superseded by `alex-feeel/medgemma-cxr-auditor-v2` as the model behind the CXR Draft Auditor demo. This repository is retained for reference and reproducibility. See the Evaluation section for the held-out comparison: with the production parser, v2 won the head-to-head on localization, presence, and urgent recall.

> Update (2026-08-29): the hosted demo has ended and the Space is paused. The Space's code, this model, and the full project repository (https://github.com/alex-feel/cxr-draft-auditor) remain public.

This is a fine-tuned, merged 16-bit derivative of `google/medgemma-1.5-4b-it`. I fine-tuned it for one job: to emit a constrained, image-grounded finding set for chest radiographs (CXR) as structured JSON with normalized bounding boxes; the fine-tuning corpus is image-grounding only and includes no draft-text-to-labels examples, so draft parsing was never a fine-tuning target. It was the first model behind my CXR Draft Auditor demo, which runs a deterministic comparator over the two label sets to flag MISSING findings, UNSUPPORTED claims, and URGENT review flags; the demo now serves the v2 derivative.

> SERVING-APP NOTE. In the live CXR Draft Auditor app, this MedGemma model is responsible for the image-grounding step (image to grounded findings with bounding boxes). The draft-impression parsing step is handled by NVIDIA Nemotron-3 Nano 4B (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) run on the GPU through `transformers`, because a dedicated instruction-following text model parses free-text drafts (including explicit denials, with the verbatim span) more reliably than the grounding-fine-tuned MedGemma did. Nemotron's native `nemotron_h` architecture (a Mamba2-Transformer hybrid) is supported directly by `transformers`, so the Space needs no extra runtime and no CUDA build. This model was fine-tuned for grounding only; the project keeps a draft-parse prompt from its early single-model design and the instruct base will attempt it, but draft parsing was never a fine-tuning target here, so the served pipeline uses Nemotron for that step and reserves MedGemma for grounding.

## Model details

- Developed by: Aleksandr Filippov (Build Small Hackathon, June 2026).
- Model type: vision-language model (multimodal, image-text-to-text), fine-tuned for constrained CXR finding extraction with bounding-box grounding (image grounding only; draft-text parsing was not a fine-tuning target).
- Base model: `google/medgemma-1.5-4b-it` (approximately 4.30B parameters).
- Fine-tuning method: QLoRA (4-bit NF4, bitsandbytes) with TRL `SFTTrainer` + PEFT, rank 16, alpha 16, learning rate 2e-4, 1 epoch, trained on a single A100 via Hugging Face Jobs. I applied LoRA to the attention and MLP projections, which by name match covers both the Gemma-3 language tower and the vision-tower attention. I then merged the adapter into a clean bf16 base (not folded into the 4-bit model) and verified the merge captured the adapter weights (a non-trivial weight delta versus the base) before publishing. I also provide an Unsloth `FastVisionModel` path for Kaggle / local training.
- Output dtype: bf16 (merged 16-bit).
- Language: English.
- License: Health AI Developer Foundations (HAI-DEF). See the HAI-DEF compliance block below.

## Related repositories

- Code repository (full project: package, training and evaluation scripts, tests): https://github.com/alex-feel/cxr-draft-auditor
- v2 (the SERVED model, supersedes this one): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2 - retrained for 2 epochs on a re-curated, deduplicated corpus. On a held-out evaluation with the production parser it won the head-to-head on localization, presence macro-F1, and urgent recall, so it replaced this v1 model as the served model on 2026-06-12.
- Demo Space (served v2; previously served this model; the hosted demo has ended): https://huggingface.co/spaces/build-small-hackathon/cxr-draft-auditor
- LoRA adapter for this model, pushed as a pre-merge safety copy: https://huggingface.co/alex-feeel/medgemma-cxr-auditor-adapter
- v2 LoRA adapter (pre-merge safety copy): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2-adapter

## Intended use

The intended use is research and education: studying whether a small vision-language model can surface apparent disagreements between a human-written draft impression and image-grounded findings, and demonstrating that audit loop with visible box evidence.

In scope:

- Research and educational experimentation with image-grounded CXR finding extraction.
- Demonstrations of an audit loop that compares a draft impression against image-grounded findings.
- Methods research on constrained-label parsing and deterministic comparison.

Out of scope (prohibited):

- Any clinical use: diagnosis, screening, triage, treatment, patient management, or any other clinical decision-making.
- Any use as, or as a component of, a medical device.
- Any use that would require regulatory authorization that has not been obtained.

The Clinical Use restriction in the HAI-DEF Terms of Use applies to this derivative.

## Canonical finding set

The model is constrained to a fixed set of six labels:

- `pleural_effusion`
- `pneumothorax`
- `lung_opacity_consolidation`
- `nodule_mass`
- `cardiomegaly`
- `no_finding`

`no_finding` is the negative sentinel and is mutually exclusive with the five positive findings. The urgent whitelist is `{pneumothorax, nodule_mass}` (a collapsed lung and a possible-malignancy mass are both can't-miss findings) and is extensible to other canonical positives.

## Output format

The model emits a JSON list of findings. Each finding carries a `label` from the canonical set, an optional `box_2d` of `[y0, x0, y1, x1]` normalized to `[0, 1]` with `(y0, x0)` at the top-left corner and `(y1, x1)` at the bottom-right corner (the y axis is normalized by image height, the x axis by image width), and (in the canonical schema) optional confidence and evidence fields, which the current model does not populate and which are therefore omitted from the user-facing output. This is the MedGemma-native grounding format. The canonical box-format identifier used throughout the project is `normalized_y0x0y1x1`.

The project's prompt set also includes a draft-parse prompt (a leftover from the early single-model design) that asks for a JSON list of `{label, status, span}` objects over the same six labels, where `status` is `present` (asserted) or `absent` (explicitly denied) and `span` is the verbatim draft phrase that produced the label. This model was fine-tuned for grounding only and was never trained on that task, so it parses drafts unreliably; in the live serving app the draft-parsing role is filled by NVIDIA Nemotron-3 Nano 4B run on the GPU through `transformers`, which emits the same `{label, status, span}` schema, while this MedGemma model is used for image grounding.

## Training data

The fine-tuning and evaluation data are drawn entirely from sources that do not require PhysioNet credentialing. Each source carries its own license, and several are non-commercial research only. Acceptance and compliance with every applicable data-use agreement is the user's responsibility.

- VinDr-CXR via Kaggle (primary bounding-box source). Obtained from the VinBigData Chest X-ray Abnormalities Detection competition or a public resized PNG mirror. License: VinDr Data Use Agreement, non-commercial research only (NOT CC0, regardless of any CC0 tag on a downstream mirror).
- VinDr-CXR-VQA (`faizan711/VinDR-CXR-VQA`, `data_v1.json` only, no images). Joined to VinDr pixels by `image_id` (a 32-character hex filename). The `gt_location` boxes are in original full-resolution pixel space and are rescaled per image when paired with a resized mirror. License: the community mirror's dataset card declares CC BY 4.0 for the annotations, but this is a third-party redistribution and the annotation license is UNVERIFIED against the original authors; confirm it before publishing a derived model. The paired images remain under the VinDr DUA (non-commercial research) regardless.
- ChestX-Det (`natealberti/ChestX-Det` HF mirror, second bounding-box source). License: Apache-2.0 annotations.
- NIH ChestX-ray14 with `BBox_List_2017.csv` (`alkzar90/NIH-Chest-X-ray-dataset`, held-out bounding-box evaluation). Box coordinates are absolute XYWH at 1024 px.
- IU-Xray / Open-i (`ykumards/open-i`, real reports, no boxes). Used only to validate the draft parser on realistic report text. License: CC BY-NC-ND 4.0.

Native dataset labels are mapped into the six-finding canonical space (for example, VinDr `Lung Opacity`, `Consolidation`, and `Infiltration` all map to `lung_opacity_consolidation`; `Nodule/Mass` maps to `nodule_mass`). Native labels with no canonical counterpart (for example, `Aortic enlargement`, `Atelectasis`, `Calcification`, `ILD`, `Pleural thickening`, `Pulmonary fibrosis`) are dropped.

### Synthetic audit data

The audit loop is built and stress-tested on synthetic drafts generated from box labels and then corrupted into three cases: drop a present finding (a MISSING case), add an absent finding (an UNSUPPORTED case), and a faithful draft (a negative control); these synthetic drafts construct known-ground-truth audit cases and are not model training data (the fine-tune is image-grounding only). Real IU-Xray reports are used ONLY to validate the draft parser on realistic phrasing, never as box supervision.

## Evaluation

I trained and sanity-checked this model for the Build Small Hackathon. It has NOT undergone a clinical evaluation, but I did evaluate it head-to-head against v2 on a held-out research split (below). The training metrics that follow are training-distribution numbers, not validated performance.

### Training metrics

One epoch over roughly 6,800 curated, class-balanced VinDr-CXR grounding examples (I deduplicate the triple-radiologist boxes per finding and resolve same-region cross-finding overlaps to the more specific label). Final training loss approximately 0.97 and mean next-token accuracy approximately 0.97 on the training distribution. The next-token accuracy mostly reflects the model learning the constrained JSON output format and the canonical label space; it is NOT a measure of localization quality.

### Held-out comparison against v2 (decision-grade)

I compared this v1 model head-to-head with v2 on a held-out set of 273 chest X-rays, none of which appears in either model's training split. The evaluation uses a single greedy generation (`do_sample=False`, bf16, SDPA) matching production serving, the production tolerant parser the live Space uses (`schema.extract_finding_list`, which routes truncated or degenerate arrays through `schema.salvage_finding_list`), and per-image ground truth from the SFT corpus validation split. Both models had zero parse failures on all 273 images. v2 won the head-to-head: it beats or ties v1 on presence macro-F1, localization (decisively), and urgent recall, which is why I replaced this model with v2 as served. The numbers below are reproduced from the v2 model card so this card stands alone.

Presence is per-finding F1 with the prevalence count N (positive images for that finding). Localization is pooled across findings as the IoU localization rate with its precision. Urgent recall is the recall on the two can't-miss urgent-whitelist findings.

| Finding | N | v1 presence F1 | v2 presence F1 |
| --- | --- | --- | --- |
| `cardiomegaly` | 133 | 0.826 | 0.863 |
| `lung_opacity_consolidation` | 70 | 0.767 | 0.729 |
| `pleural_effusion` | 13 | 0.72 | 0.58 |
| `nodule_mass` (urgent) | 9 | 0.27 | 0.50 |
| `pneumothorax` (urgent) | 1 | recall 0/1 | recall 1/1 |
| Presence macro-F1 (all findings) | - | 0.646 | 0.735 |

| Metric | v1 | v2 |
| --- | --- | --- |
| Parse failures (of 273) | 0 | 0 |
| Localization IoU@0.3 rate / precision | 0.484 / 0.613 | 0.633 / 0.791 |
| Localization IoU@0.5 rate / precision | 0.360 / 0.456 | 0.531 / 0.664 |
| Mean IoU on matched boxes | 0.614 | 0.700 |
| Urgent recall - `nodule_mass` | 3/9 | 4/9 |
| Urgent recall - `pneumothorax` | 0/1 | 1/1 |

Caveats: the urgent classes are scarce (`nodule_mass` N=9, `pneumothorax` N=1), so urgent-recall figures are directional, not statistically robust; the ground truth comes from SFT corpus validation targets that retain same-region opacity-plus-nodule double-labels, which slightly depresses v2's generic `lung_opacity_consolidation` recall as a convention difference rather than a real regression; the evaluation uses a single greedy generation per image; and this is a research-split comparison, not a clinical evaluation. This model is not a medical device. See the v2 model card for the full Evaluation section and the salvage-parser correction history.

## Limitations and hallucination warnings

- The model hallucinates. It invents findings that are not present, misses findings that are present, and emits boxes that do not localize the structure named. Treat every output as unverified.
- The constrained six-finding label space is intentionally small. Findings outside that set (for example, aortic enlargement, atelectasis, fractures, tubes and lines) are not represented and will be silently absent. Their absence in the output means nothing about whether they are present in the image.
- Bounding boxes are approximate. A plausible-looking box is not evidence that the model reasoned about the correct region.
- The URGENT review flag (a small whitelist of pneumothorax and nodule/mass) is a demonstration heuristic, not a safety mechanism. The absence of an URGENT flag does NOT mean an image is normal or safe.
- The audit comparator is deterministic and only as good as the two label sets it compares. A MISSING or UNSUPPORTED flag reflects a disagreement between two model outputs, not ground truth.
- Performance on images outside the training distribution (different scanners, pediatric images, lateral views, post-operative anatomy, devices) is unknown and likely poor.
- The model is English-only and was tuned on a narrow distribution of chest radiographs with a single pinned grounding prompt; off-template prompts or out-of-distribution images may degrade its grounding output.
- This model has not been evaluated for fairness across demographic groups; the source datasets have known and unmeasured biases.

## How to use

This model previously served my CXR Draft Auditor Gradio Space; the Space now serves the v2 derivative (`alex-feeel/medgemma-cxr-auditor-v2`). To run this v1 model for reference or reproducibility, inference uses vanilla `transformers` (`AutoModelForImageTextToText`) with `attn_implementation='sdpa'` (Flash-Attention 3 is not usable on the ZeroGPU sm_120 backend). The merged 16-bit model fits the ZeroGPU `large` tier (48 GB) at bf16 with no quantization. Use the pinned prompt templates from `cxr_auditor.prompts`; MedGemma is prompt-sensitive and single-turn.

## HAI-DEF compliance (required)

This model is a MedGemma "Model Derivative" within the meaning of the Health AI Developer Foundations (HAI-DEF) Terms of Use. The following statements are part of the distribution conditions for that derivative.

- HAI-DEF Terms of Use: https://developers.google.com/health-ai-developer-foundations/terms
- Prohibited Use Policy (incorporated by reference into the Terms): https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy
- Notice file: a `NOTICE` file is distributed with this repository and the model. It states verbatim: "HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at https://developers.google.com/health-ai-developer-foundations/terms".
- Modified-file notice: this model is a modified work derived from `google/medgemma-1.5-4b-it`. All files modified relative to the upstream MedGemma distribution carry a prominent "modified" notice. The weights themselves were modified by QLoRA fine-tuning and merging.
- Agreement propagation: a copy of the HAI-DEF Agreement is provided to all recipients. The Section 3.2 use restrictions of the HAI-DEF Terms (including, without limitation, the Clinical Use restriction, the prohibition on uses that would make Google a device manufacturer, and the Prohibited Use Policy) are propagated as an enforceable provision governing the use and further distribution of this derivative. Recipients are hereby notified that their use and any further distribution of this model are subject to Section 3.2.
- No endorsement: Google does not endorse this model, this software, or its author. "MedGemma", "Gemma", and "Google" are trademarks of Google LLC and are used here only for accurate attribution of the base model. No trademark license or endorsement is granted or implied.
- Health regulatory authorization: where applicable, the user must obtain any required Health Regulatory Authorization before any use beyond the research and educational scope stated above. No such authorization has been sought for this derivative, and none of its intended uses require it because clinical use is out of scope and prohibited.

## Citation

If this work is referenced, cite the base model (`google/medgemma-1.5-4b-it`) and the data sources listed above under their respective licenses, and link to the CXR Draft Auditor Space (`https://huggingface.co/spaces/build-small-hackathon/cxr-draft-auditor`).

## Disclaimer

See `DISCLAIMER.md` in the repository. Research and educational QA only. NOT a medical device, NOT diagnosis, NOT for clinical use. Outputs are frequently wrong; always consult a qualified radiologist.
