---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
base_model: google/medgemma-1.5-4b-it
tags:
  - lora
  - qlora
  - chest-x-ray
  - radiology
  - vision-language-model
  - medgemma
  - image-text-to-text
  - model-derivative
  - research
  - not-a-medical-device
pipeline_tag: image-text-to-text
library_name: peft
---

# CXR Draft Auditor - LoRA Adapter for v2 (pre-merge safety copy)

> RESEARCH / EDUCATIONAL QA ONLY. This adapter is NOT a medical device, NOT a diagnostic tool, and NOT a substitute for a qualified radiologist. It MUST NOT be used for clinical decision-making, screening, triage, or patient care. Outputs of the adapted model are frequently wrong.

> Update (2026-08-29): the CXR Draft Auditor hosted demo has ended and the Space is paused. The Space's code and the full project repository (https://github.com/alex-feel/cxr-draft-auditor) remain public.

This repository holds the unmerged QLoRA LoRA adapter I trained over `google/medgemma-1.5-4b-it` for the v2 variant of the CXR Draft Auditor project. I pushed it to the Hub as a safety copy BEFORE the merge step, following the procedure I use across the project (push the adapter first, then merge into a clean bf16 base and verify a non-trivial weight delta, so a silent merge can never be published). Applying this adapter to the bf16 base model reproduces the merged v2 model, `alex-feeel/medgemma-cxr-auditor-v2`, which is the model my CXR Draft Auditor demo Space serves as of 2026-06-12 (it superseded v1). On a held-out evaluation with the production parser, v2 won the head-to-head over v1 on localization, presence, and urgent recall; see the merged v2 card for the full Evaluation section.

## Adapter details

- Developed by: Aleksandr Filippov (Build Small Hackathon, June 2026).
- Base model: `google/medgemma-1.5-4b-it` (HAI-DEF, gated; you must accept the HAI-DEF terms on the base repository to download it).
- Corresponding merged model: `alex-feeel/medgemma-cxr-auditor-v2` (the model SERVED by the demo Space as of 2026-06-12).
- Adapter type: LoRA (PEFT), rank 16, alpha 16, dropout 0.0, no bias, task type `CAUSAL_LM`, saved with PEFT 0.19.1.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (attention and MLP projections; by name match this covers both the Gemma-3 language tower and the vision-tower attention).
- Training: TRL `SFTTrainer` + PEFT QLoRA (4-bit NF4 via bitsandbytes), learning rate 2e-4, batch size 1, gradient accumulation 8, 2 epochs over the re-curated, deduplicated corpus `alex-feeel/cxr-sft-v2` (private; the VinDr-CXR pixels are under a non-commercial data-use agreement and cannot be redistributed), on a single A100 (80 GB) via Hugging Face Jobs.
- Contents: `adapter_config.json` and `adapter_model.safetensors` (approximately 66 MB), plus the tokenizer and processor configuration files saved alongside the adapter.

For the task description, canonical finding set, output format, training data and licenses, what changed relative to v1, and the held-out evaluation that motivated serving v2, see the merged v2 model card: https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2. No separate evaluation exists for this adapter; the merged v2 model carries the full results.

## How to load with PEFT

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base_id = "google/medgemma-1.5-4b-it"  # gated: accept the HAI-DEF terms first
base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "alex-feeel/medgemma-cxr-auditor-v2-adapter")
processor = AutoProcessor.from_pretrained(base_id)
```

To reproduce the published merged model, call `model.merge_and_unload()` on the loaded `PeftModel` (on the bf16 base, never on a 4-bit quantized base) and verify the merged weights differ non-trivially from the base before using or distributing the result.

## License and usage constraints

Identical to the merged model cards. This adapter is a MedGemma "Model Derivative" within the meaning of the Health AI Developer Foundations (HAI-DEF) Terms of Use, and the following distribution conditions apply.

- HAI-DEF Terms of Use: https://developers.google.com/health-ai-developer-foundations/terms
- Prohibited Use Policy (incorporated by reference into the Terms): https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy
- Notice file: a `NOTICE` file is distributed with this repository. It states verbatim: "HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at https://developers.google.com/health-ai-developer-foundations/terms".
- Agreement propagation: the Section 3.2 use restrictions of the HAI-DEF Terms (including, without limitation, the Clinical Use restriction, the prohibition on uses that would make Google a device manufacturer, and the Prohibited Use Policy) are propagated as an enforceable provision governing the use and further distribution of this adapter. Recipients are hereby notified that their use and any further distribution are subject to Section 3.2.
- No endorsement: Google does not endorse this adapter, this software, or its author. "MedGemma", "Gemma", and "Google" are trademarks of Google LLC and are used here only for accurate attribution of the base model.
- Any clinical use (diagnosis, screening, triage, treatment, patient management) is out of scope and prohibited.

## Links

- Code repository (full project: package, training and evaluation scripts, tests): https://github.com/alex-feel/cxr-draft-auditor
- Merged v2 model (SERVED): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2
- v1 model (superseded as served): https://huggingface.co/alex-feeel/medgemma-cxr-auditor
- v1 adapter (pre-merge safety copy): https://huggingface.co/alex-feeel/medgemma-cxr-auditor-adapter
- Demo Space (serves v2): https://huggingface.co/spaces/build-small-hackathon/cxr-draft-auditor
