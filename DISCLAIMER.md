# Disclaimer

## Not a medical device

CXR Draft Auditor is a research and educational quality-assurance prototype. It is NOT a medical device. It has NOT been cleared, approved, certified, or registered by the U.S. Food and Drug Administration, the European Medicines Agency, the conformity-assessment bodies behind the CE mark, or any other regulatory authority anywhere.

## Not diagnosis, not clinical use

The tool does NOT diagnose disease, does NOT make clinical recommendations, and does NOT provide a second medical opinion. Its outputs (findings, bounding boxes, MISSING/UNSUPPORTED/URGENT flags) are illustrative artifacts produced by two machine-learning models (a fine-tuned MedGemma that grounds the image and NVIDIA Nemotron that parses the draft) and a deterministic comparator. They are frequently wrong. They MUST NOT be used for screening, triage, treatment, patient management, or any other clinical decision-making.

## Intended use

The only intended use is research and education: studying whether a small vision-language model can surface apparent disagreements between a human-written draft impression and image-grounded findings, and demonstrating that audit loop. Any other use is out of scope and unsupported.

## No patient data

This repository contains no patient data and downloads none. The synthetic fixtures used by the tests are fabricated and contain no real images or reports. Users who supply their own images are responsible for ensuring those images are de-identified and that they hold the rights to use them. Do not upload protected health information.

## Model and data licenses are the user's responsibility

The grounding base model (`google/medgemma-1.5-4b-it`) is distributed under the Health AI Developer Foundations (HAI-DEF) terms. The draft parser (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, run on the GPU through transformers) is distributed under the NVIDIA Nemotron Open Model License. The datasets referenced for evaluation and fine-tuning (VinDr-CXR, VinDr-CXR-VQA, ChestX-Det, NIH ChestX-ray14, IU-Xray / Open-i) each carry their own licenses and data-use agreements, several of which are non-commercial research only. It is the user's responsibility to read, accept, and comply with every applicable model and dataset license before downloading or using any of them. See `SETUP.md` for the precise, user-only access steps.

## No warranty

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from, out of, or in connection with the software or its use.

## Urgent findings

The URGENT review flag (a small whitelist of pneumothorax and nodule/mass) is a demonstration heuristic, not a safety mechanism. The absence of an URGENT flag does NOT mean an image is normal or safe. A genuinely urgent clinical situation requires immediate evaluation by a qualified clinician, never this tool.
