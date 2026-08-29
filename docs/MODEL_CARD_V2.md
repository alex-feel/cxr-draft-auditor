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

# CXR Draft Auditor v2 - Fine-tuned MedGemma Derivative (the SERVED model)

> RESEARCH / EDUCATIONAL QA ONLY. This model is NOT a medical device, NOT a diagnostic tool, and NOT a substitute for a qualified radiologist. It MUST NOT be used for clinical decision-making, screening, triage, or patient care. Its outputs are frequently wrong.

> THIS IS THE SERVED MODEL. As of 2026-06-12, this v2 model is the model behind the CXR Draft Auditor demo. The live Space `build-small-hackathon/cxr-draft-auditor` loads it (bf16, SDPA attention). It supersedes v1 (`alex-feeel/medgemma-cxr-auditor`), which is retained for reference and reproducibility. See the Evaluation section below for the held-out, decision-grade comparison that motivated the switch.

> Update (2026-08-29): the hosted demo has ended and the Space is paused. The Space's code, this model, and the full project repository (https://github.com/alex-feel/cxr-draft-auditor) remain public.

This is the second fine-tuned, merged 16-bit derivative of `google/medgemma-1.5-4b-it` I trained for the CXR Draft Auditor project, and the one the demo serves. Like v1, I fine-tuned it for one job: to emit a constrained, image-grounded finding set for chest radiographs (CXR) as structured JSON with normalized bounding boxes; the fine-tuning corpus is image-grounding only and includes no draft-text-to-labels examples, so draft parsing was never a fine-tuning target. It differs from v1 only in the training corpus and the number of epochs.

> SERVING-APP NOTE. In the live CXR Draft Auditor app, this model does the image-grounding step (image to grounded findings with bounding boxes) on the GPU. The draft-impression parsing step is handled by NVIDIA Nemotron-3 Nano 4B (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, bf16) run on the GPU through `transformers`. I split the pipeline this way because a dedicated instruction-following text model parses free-text drafts (including explicit denials, with the verbatim span) more reliably than this grounding-fine-tuned MedGemma did; Nemotron also reasons briefly before emitting its label JSON, which improves extraction on multi-clause drafts. This model was fine-tuned for grounding only; the project keeps a draft-parse prompt from its early single-model design and the instruct base will attempt it, but draft parsing was never a fine-tuning target here, so the served pipeline reserves this model for grounding and uses Nemotron for the draft. Because both models run on the GPU, a full audit takes roughly 15 to 30 seconds; the deterministic comparator over the two label sets remains the only judgment layer.

## Why a fine-tune, not the base model

The base `google/medgemma-1.5-4b-it` already carries the medical knowledge for chest radiographs, but out of the box it does not reliably honor the contract this pipeline depends on: a clean, on-vocabulary JSON list of `{label, box_2d}` that the deterministic comparator can read without guessing. In a small illustrative pass over the demo's four example images (an illustration, not a benchmark), the stock base, given the same production grounding prompt, narrated step-by-step reasoning instead of emitting JSON on one image (so nothing parsed), called a real `cardiomegaly` case normal on another, and on the mass image labeled the mass as `lung_opacity_consolidation` while also emitting a contradictory `no_finding`, which dropped the URGENT flag. That is a reliability and format-discipline gap, not a knowledge gap. The fine-tune does not teach the model to see a chest X-ray, which it already can; it disciplines it into reliable, parseable, on-vocabulary findings-with-boxes that the comparator can trust. The Evaluation section below is the next step: once a fine-tune existed, the served-model choice was the head-to-head v1-versus-v2 comparison.

## Model details

- Developed by: Aleksandr Filippov (Build Small Hackathon, June 2026).
- Model type: vision-language model (multimodal, image-text-to-text), fine-tuned for constrained CXR finding extraction with bounding-box grounding (image grounding only; draft-text parsing was not a fine-tuning target).
- Base model: `google/medgemma-1.5-4b-it` (approximately 4.30B parameters).
- Fine-tuning method: QLoRA (4-bit NF4, bitsandbytes) with TRL `SFTTrainer` + PEFT, rank 16, alpha 16, learning rate 2e-4, batch size 1, gradient accumulation 8, 2 epochs, trained on a single A100 (80 GB) via Hugging Face Jobs. I applied LoRA to the attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), which by name match covers both the Gemma-3 language tower and the vision-tower attention. I pushed the adapter to `alex-feeel/medgemma-cxr-auditor-v2-adapter` as a safety copy before merging, then merged it into a clean bf16 base (not folded into the lossy 4-bit training model), and verified the merge captured the adapter weights (a non-trivial weight delta versus the base) before publishing.
- Output dtype: bf16 (merged 16-bit).
- Language: English.
- License: Health AI Developer Foundations (HAI-DEF). See the HAI-DEF compliance block below.

## Related repositories

- Code repository (full project: package, training and evaluation scripts, tests): https://github.com/alex-feel/cxr-draft-auditor
- Demo Space (served this model; the hosted demo has ended): https://huggingface.co/spaces/build-small-hackathon/cxr-draft-auditor
- LoRA adapter for this v2 model (pre-merge safety copy): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2-adapter
- v1 (superseded as served, retained for reference and reproducibility): https://huggingface.co/alex-feeel/medgemma-cxr-auditor
- LoRA adapter for v1 (pre-merge safety copy): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-adapter

## What changed relative to v1

v2 keeps the v1 recipe (same base model, label space, prompts, output contract, and LoRA configuration) and changes two things: the training corpus and the epoch count.

- Training corpus: I trained v2 on `alex-feeel/cxr-sft-v2`, a re-curated, deduplicated version of the v1 corpus. The re-curation performs, in order: per-label box deduplication and merging (boxes of the SAME finding whose pairwise IoU meets the 0.5 merge threshold are clustered via union-find connected components and replaced by the cluster mean, so triple-radiologist near-duplicate boxes collapse to one representative while genuinely distinct boxes stay separate); cross-finding overlap resolution (when boxes of two DIFFERENT findings localize the same region with IoU >= 0.6, only the more specific label is kept, using my specificity order, most specific first: `pneumothorax`, `nodule_mass`, `pleural_effusion`, `cardiomegaly`, `lung_opacity_consolidation`); class balancing (all positive records and all pneumothorax records are kept, `no_finding`-only records are downsampled to roughly 1:1 normal:positive); and a stratified 90/10 train/validation split. The cross-finding step targets the v1 failure mode of labeling one region as both `lung_opacity_consolidation` and `nodule_mass`.
- Epochs: 2 (v1 used 1).

The corpus repositories (`alex-feeel/cxr-sft` and `alex-feeel/cxr-sft-v2`) are private because the VinDr-CXR pixels are under a non-commercial data-use agreement and cannot be redistributed.

## Evaluation

I evaluated v2 head-to-head against v1 on a held-out set of 273 chest X-rays, none of which appears in either model's training split. The evaluation uses a single greedy generation (`do_sample=False`, bf16, SDPA) that matches production serving exactly, and the same production tolerant parser the live Space uses (`schema.extract_finding_list`, which routes truncated or degenerate arrays through `schema.salvage_finding_list`). The ground truth for each image is its own assistant target from the SFT corpus validation split. With this parser, v2 had zero parse failures on all 273 images, and it beats or ties v1 on every axis that matters: presence accuracy, localization, and urgent recall. This held-out comparison is the reason v2 is the served model.

### Per-finding results

Presence is per-finding F1 with the prevalence count N (positive images for that finding in the held-out set). Localization is reported pooled across findings as the IoU@0.3 and IoU@0.5 localization rate with its precision. Urgent recall is the recall on the two can't-miss findings on the urgent whitelist.

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

### Why v2 supersedes v1 as served

With the production salvage parser, v2 wins on localization decisively (every IoU rate and precision is higher, and mean IoU on matched boxes rises from 0.614 to 0.700), wins on presence macro-F1 (0.735 versus 0.646), and wins on urgent recall (`nodule_mass` 4/9 versus 3/9, `pneumothorax` 1/1 versus 0/1). My first impression that v2 had "regressed on urgent and crashed" turned out to be an artifact of a pre-salvage parser that discarded v2's outputs: 6 of v2's 9 `nodule_mass` cases were parse failures before I fixed the parser, so the urgent signal was being thrown away rather than missed by the model. With the corrected parser those outputs are recovered, and v2 is the better model on the safety-adjacent urgent axis as well as on boxes and presence. That is why I serve v2.

### Caveats

- Urgent classes are scarce. The held-out VinDr validation data is exhausted for urgent cases, so `nodule_mass` (N=9) and especially `pneumothorax` (N=1) numbers are directional, not statistically robust. The urgent-recall figures show v2 at or above v1, but a single-positive `pneumothorax` cell cannot carry statistical weight.
- The ground truth slightly understates v2's generic-opacity recall. The ground truth comes from the SFT corpus validation targets, which retain the same-region opacity-plus-nodule double-labels that v2's corpus curation deliberately deduplicated. This depresses v2's `lung_opacity_consolidation` recall relative to v1 (0.729 versus 0.767 F1) and is a ground-truth convention difference, NOT a real localization regression. Urgent recall is unaffected: the dedup keeps `nodule_mass` present, so `nodule_mass` and `pneumothorax` presence are consistent across both conventions.
- The evaluation uses a single greedy generation per image, matching production. It is not a sampled or ensembled estimate, and it is not a clinical or regulatory evaluation.
- Research and educational use only. These metrics describe behavior on a narrow held-out research split; they do not establish fitness for any clinical purpose. This model is not a medical device.

## Intended use

The intended use is research and education: studying whether a small vision-language model can surface apparent disagreements between a human-written draft impression and image-grounded findings, and demonstrating that audit loop with visible box evidence. This is the model the demo Space serves, within that research-and-education scope only; it is not intended for deployment in any clinical setting.

In scope:

- Research and educational experimentation with image-grounded CXR finding extraction.
- Demonstrations of an audit loop that compares a draft impression against image-grounded findings.
- Reproducible comparison of the v1 and v2 training recipes (corpus curation and epoch count).
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

The model emits a JSON list of findings. Each finding carries a `label` from the canonical set, an optional `box_2d` of `[y0, x0, y1, x1]` normalized to `[0, 1]` with `(y0, x0)` at the top-left corner and `(y1, x1)` at the bottom-right corner (the y axis is normalized by image height, the x axis by image width), and (in the canonical schema) optional confidence and evidence fields, which the model does not populate and which are therefore omitted from the user-facing output. This is the MedGemma-native grounding format. The canonical box-format identifier used throughout the project is `normalized_y0x0y1x1`.

The project's prompt set also includes a draft-parse prompt (a leftover from the early single-model design) that asks for a JSON list of `{label, status, span}` objects over the same six labels, where `status` is `present` (asserted) or `absent` (explicitly denied) and `span` is the verbatim draft phrase that produced the label. This model was fine-tuned for grounding only and was never trained on that task, so it parses drafts unreliably; in the live serving app the draft-parsing role is filled by NVIDIA Nemotron-3 Nano 4B (through `transformers` on the GPU), which emits the same `{label, status, span}` schema, while this MedGemma model is used for the image-grounding step.

Like any generative model, this model can emit output that does not parse cleanly against the schema (for example a truncated or malformed array). The served Space recovers such output with the production tolerant parser (`schema.extract_finding_list`, which routes truncated or degenerate arrays through `schema.salvage_finding_list`); on the held-out evaluation set this parser yielded zero parse failures across all 273 images. Any consumer of this model that does not use that parser must still treat schema-invalid output as a possible outcome and handle it.

## Training data

The training data sources, label mapping, and licensing are the same as for v1; only the curation differs (see "What changed relative to v1" above). Each source carries its own license, and several are non-commercial research only. Acceptance and compliance with every applicable data-use agreement is the user's responsibility.

- VinDr-CXR via Kaggle (primary bounding-box source). License: VinDr Data Use Agreement, non-commercial research only (NOT CC0, regardless of any CC0 tag on a downstream mirror).
- VinDr-CXR-VQA (`faizan711/VinDR-CXR-VQA`, `data_v1.json` only, no images). The community mirror's dataset card declares CC BY 4.0 for the annotations, but this is a third-party redistribution and the annotation license is UNVERIFIED against the original authors. The paired images remain under the VinDr DUA (non-commercial research) regardless.
- ChestX-Det (`natealberti/ChestX-Det` HF mirror, second bounding-box source). License: Apache-2.0 annotations.
- NIH ChestX-ray14 with `BBox_List_2017.csv` (`alkzar90/NIH-Chest-X-ray-dataset`, held-out bounding-box evaluation).
- IU-Xray / Open-i (`ykumards/open-i`, real reports, no boxes). Used only to validate the draft parser on realistic report text. License: CC BY-NC-ND 4.0.

Native dataset labels are mapped into the six-finding canonical space; native labels with no canonical counterpart are dropped.

## Limitations and hallucination warnings

All v1 limitations apply:

- The model hallucinates. It invents findings that are not present, misses findings that are present, and emits boxes that do not localize the structure named. Treat every output as unverified.
- Urgent recall is imperfect and the urgent-class evaluation is small. On the held-out set this model recovered 4 of 9 `nodule_mass` cases and the single `pneumothorax` case; both numbers are at or above v1, but they are directional, not statistically robust, because the urgent classes are scarce (see the Evaluation caveats). A missed urgent finding is possible.
- The constrained six-finding label space is intentionally small. Findings outside that set are not represented and will be silently absent. Their absence in the output means nothing about whether they are present in the image.
- Bounding boxes are approximate. A plausible-looking box is not evidence that the model reasoned about the correct region.
- The URGENT review flag (a small whitelist of pneumothorax and nodule/mass) is a demonstration heuristic, not a safety mechanism. The absence of an URGENT flag does NOT mean an image is normal or safe.
- Performance on images outside the training distribution (different scanners, pediatric images, lateral views, post-operative anatomy, devices) is unknown and likely poor.
- The model is English-only and was tuned on a narrow distribution of chest radiographs with a single pinned grounding prompt; off-template prompts or out-of-distribution images may degrade its grounding output.
- This model has not been evaluated for fairness across demographic groups; the source datasets have known and unmeasured biases.

## How to use

This is the model my CXR Draft Auditor Gradio Space serves for the image-grounding step. Inference uses vanilla `transformers` (`AutoModelForImageTextToText`) at bf16 with `attn_implementation='sdpa'` (Flash-Attention 3 is not usable on the ZeroGPU sm_120 backend). The merged 16-bit model fits the ZeroGPU `large` tier (48 GB) at bf16 with no quantization. Use the pinned prompt templates from `cxr_auditor.prompts`; MedGemma is prompt-sensitive and single-turn. The Space recovers schema-invalid output with the production tolerant parser (`schema.extract_finding_list`); any consumer that does not use that parser must handle schema-invalid output itself. In the served pipeline the draft impression is parsed separately by NVIDIA Nemotron-3 Nano 4B (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`), which also runs on the GPU through `transformers`, and the two label sets are then compared by the deterministic comparator.

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

See `DISCLAIMER.md` in the project repository. Research and educational QA only. NOT a medical device, NOT diagnosis, NOT for clinical use. Outputs are frequently wrong; always consult a qualified radiologist.
