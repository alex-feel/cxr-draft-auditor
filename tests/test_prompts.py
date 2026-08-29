"""Tests for the pinned MedGemma prompt templates."""

from __future__ import annotations

import json
import re

import pytest

from cxr_auditor.findings import CANONICAL_FINDINGS, NO_FINDING
from cxr_auditor.prompts import (
    build_draft_parsing_prompt,
    build_image_grounding_prompt,
)
from cxr_auditor.schema import extract_finding_list


def test_image_grounding_prompt_embeds_all_canonical_labels() -> None:
    prompt = build_image_grounding_prompt()
    for label in CANONICAL_FINDINGS:
        assert label in prompt


def test_image_grounding_prompt_specifies_box_convention() -> None:
    prompt = build_image_grounding_prompt()
    assert "box_2d" in prompt
    assert "[y0, x0, y1, x1]" in prompt
    assert "top-left" in prompt
    assert "bottom-right" in prompt


def test_image_grounding_prompt_in_context_examples_are_valid_json() -> None:
    prompt = build_image_grounding_prompt()
    # Every JSON array embedded as an in-context example must itself parse and be
    # extractable by the production finding-list parser.
    arrays = re.findall(r"\[\{.*?\}\]", prompt)
    assert arrays, "expected at least one in-context example array"
    for array_text in arrays:
        parsed = extract_finding_list(array_text)
        assert parsed[0]["label"] in CANONICAL_FINDINGS


def test_draft_parsing_prompt_embeds_draft_text() -> None:
    draft = "Left pleural effusion. No pneumothorax."
    prompt = build_draft_parsing_prompt(draft)
    assert draft in prompt
    for label in CANONICAL_FINDINGS:
        assert label in prompt


def test_draft_parsing_prompt_strips_whitespace() -> None:
    prompt = build_draft_parsing_prompt("   Cardiomegaly.   ")
    assert "Cardiomegaly." in prompt
    assert "   Cardiomegaly.   " not in prompt


def test_draft_parsing_prompt_in_context_examples_are_valid_json() -> None:
    prompt = build_draft_parsing_prompt("Some draft.")
    arrays = re.findall(r"\[\{.*?\}\]", prompt)
    assert arrays, "expected at least one in-context example array"
    for array_text in arrays:
        parsed = json.loads(array_text)
        for element in parsed:
            assert element["label"] in CANONICAL_FINDINGS
            assert element["status"] in {"present", "absent"}


def test_draft_parsing_prompt_mentions_no_finding_sentinel() -> None:
    prompt = build_draft_parsing_prompt("No acute cardiopulmonary abnormality.")
    assert NO_FINDING in prompt


def test_build_draft_parsing_prompt_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_draft_parsing_prompt("   ")


def test_prompts_contain_not_clinical_use_guardrail() -> None:
    assert "NOT" in build_image_grounding_prompt()
    assert "NOT" in build_draft_parsing_prompt("x")
