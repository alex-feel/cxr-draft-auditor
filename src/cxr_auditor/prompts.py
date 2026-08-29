"""
Pinned prompt templates for the CXR Draft Auditor.

Both serving models are prompt-sensitive and single-turn, so the prompts here are
fixed templates rather than dynamically composed strings. Two prompts are defined:

1. Image grounding (``IMAGE_GROUNDING_PROMPT`` / ``build_image_grounding_prompt``):
   instruct the grounding VLM (a fine-tuned MedGemma) to emit a JSON list of
   constrained findings, each with a ``box_2d`` normalized to ``[0, 1]`` in the
   canonical ``[y0, x0, y1, x1]`` ordering. This matches MedGemma's native CXR
   bounding-box emission.
2. Draft parsing (``DRAFT_PARSING_PROMPT`` / ``build_draft_parsing_prompt``):
   instruct the draft text model (NVIDIA Nemotron-3 Nano 4B) to extract the
   findings a draft impression asserts or denies, into the same six-label space,
   as a JSON list.

Both prompts embed the canonical label list and a couple of in-context examples.
The module is stdlib-only; the label list is sourced from ``cxr_auditor.findings``
so there is a single source of truth for the vocabulary.
"""

from __future__ import annotations

from cxr_auditor.findings import CANONICAL_FINDINGS, NO_FINDING

# Rendered, comma-separated canonical label list for embedding in prompt text.
_LABEL_LIST = ", ".join(CANONICAL_FINDINGS)


# Image-grounding prompt. The model receives a single chest X-ray image alongside
# this text and must return ONLY a JSON list. ``{label_list}`` is filled with the
# canonical findings; ``{no_finding}`` with the negative sentinel label.
IMAGE_GROUNDING_PROMPT: str = """\
You are a chest X-ray finding extractor for a research quality-assurance tool. \
This is NOT diagnosis and NOT clinical use.

Look at the chest X-ray and report ONLY findings drawn from this fixed label set:
{label_list}

Rules:
- Use ONLY the labels above, spelled exactly as written (lowercase, underscores).
- Return a JSON list. Each element is an object with these keys:
  - "label": one label from the set above.
  - "box_2d": [y0, x0, y1, x1], the bounding box normalized to [0, 1], where
    (y0, x0) is the top-left corner and (y1, x1) is the bottom-right corner.
    y is the vertical axis (top=0, bottom=1); x is the horizontal axis
    (left=0, right=1).
  - "confidence": a number in [0, 1].
  - "evidence": a short phrase describing the visual evidence.
- If a finding is genuinely present but not localizable to a box, set "box_2d" to null.
- If there is no abnormal finding, return a single element with "label": "{no_finding}".
- Output ONLY the JSON list. No prose, no markdown fences, no commentary.

Example output for an image with a left-sided pleural effusion:
[{{"label": "pleural_effusion", "box_2d": [0.62, 0.08, 0.94, 0.40], "confidence": 0.78, "evidence": "blunting and opacity at the left costophrenic angle"}}]

Example output for a normal image:
[{{"label": "{no_finding}", "box_2d": null, "confidence": 0.90, "evidence": "clear lung fields, normal cardiomediastinal silhouette"}}]
"""


# Draft-parsing prompt. The model receives the draft impression text and must
# return ONLY a JSON list mapping the draft into the canonical label space.
# ``{label_list}`` is the canonical findings; ``{draft_report}`` is the draft text.
DRAFT_PARSING_PROMPT: str = """\
You are a radiology-report label extractor for a research quality-assurance tool. \
This is NOT diagnosis and NOT clinical use.

Read the draft chest X-ray impression below and extract which findings it ASSERTS \
as present and which it explicitly DENIES, using ONLY this fixed label set:
{label_list}

Label guidance - map the draft's wording to the closest label:
- nodule_mass: nodule, mass, tumor, tumour, neoplasm, carcinoma, carcinoid, lung \
cancer, malignancy, lesion, coin lesion, spiculated lesion.
- lung_opacity_consolidation: opacity, consolidation, infiltrate, infiltration, \
airspace disease, pneumonia, atelectasis.
- pleural_effusion: pleural effusion, effusion, pleural fluid, blunting of the \
costophrenic angle.
- pneumothorax: pneumothorax, collapsed lung, absent lung markings.
- cardiomegaly: cardiomegaly, enlarged heart, enlarged cardiac silhouette, \
increased cardiothoracic ratio.

Rules:
- Use ONLY the labels above, spelled exactly as written (lowercase, underscores).
- Return a JSON list. Each element is an object with these keys:
  - "label": one label from the set above.
  - "status": "present" if the draft asserts the finding, "absent" if the draft
    explicitly denies it.
  - "span": the verbatim phrase from the draft that supports this label.
- Do NOT include a label the draft does not mention at all.
- If the draft asserts a normal study (for example "no acute cardiopulmonary
  abnormality"), emit "{no_finding}" with status "present".
- Output ONLY the JSON list. No prose, no markdown fences, no commentary.

Example draft: "Left pleural effusion. No pneumothorax."
Example output:
[{{"label": "pleural_effusion", "status": "present", "span": "Left pleural effusion"}}, {{"label": "pneumothorax", "status": "absent", "span": "No pneumothorax"}}]

Example draft: "No acute cardiopulmonary abnormality."
Example output:
[{{"label": "{no_finding}", "status": "present", "span": "No acute cardiopulmonary abnormality"}}]

Draft impression:
{draft_report}
"""


def build_image_grounding_prompt() -> str:
    """Build the pinned image-grounding prompt text.

    The chest X-ray image itself is supplied to the model separately (as the
    image part of the multimodal turn); this returns only the text part.

    Returns:
        The fully rendered image-grounding prompt with the canonical label list
        embedded.
    """
    return IMAGE_GROUNDING_PROMPT.format(label_list=_LABEL_LIST, no_finding=NO_FINDING)


def build_draft_parsing_prompt(draft_report: str) -> str:
    """Build the pinned draft-parsing prompt text for a given draft impression.

    Args:
        draft_report: The draft radiology impression to parse. Leading and
            trailing whitespace is stripped.

    Returns:
        The fully rendered draft-parsing prompt with the canonical label list and
        the draft text embedded.

    Raises:
        ValueError: If ``draft_report`` is empty or whitespace only.
    """
    cleaned = draft_report.strip()
    if not cleaned:
        raise ValueError("draft_report must be non-empty")
    return DRAFT_PARSING_PROMPT.format(
        label_list=_LABEL_LIST,
        no_finding=NO_FINDING,
        draft_report=cleaned,
    )
