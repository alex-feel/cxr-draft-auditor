"""
Supervised fine-tuning (SFT) corpus construction for the image-grounding task.

This module turns image-grounded findings into the chat-formatted records an
Unsloth/TRL vision SFT run consumes, and writes/validates them as JSONL. Each
record pairs one chest X-ray with the pinned grounding prompt (the user turn) and
the canonical finding JSON with boxes (the assistant turn the model is trained to
emit).

Record shape
------------
Each JSONL line is one training example::

    {
      "image_path": "<path to the image file>",
      "messages": [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "<grounding prompt>"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<canonical finding JSON list>"}
        ]}
      ]
    }

The user content is a list of parts (an image placeholder plus the prompt text) so
a vision chat template can interleave the pixels with the instruction. The image
pixels themselves are referenced by ``image_path`` and loaded by the training
collator; the JSONL carries the path, not the bytes. The assistant content is the
target the loss is masked to: a JSON list of ``{label, box_2d, ...}`` objects in
MedGemma's native grounding format, which ``schema.extract_finding_list`` parses.

Dependencies are limited to the standard library and the pure-logic
schema/findings/prompts modules, so the whole module imports and unit-tests
without any model-serving or training stack.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cxr_auditor.findings import NO_FINDING
from cxr_auditor.prompts import build_image_grounding_prompt
from cxr_auditor.schema import (
    CANONICAL_BOX_FORMAT,
    FindingStatus,
    ImageFinding,
    extract_finding_list,
    normalize_finding,
)

# Roles used in the chat-formatted records.
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

# Content-part type identifiers used inside a message's content list.
IMAGE_PART_TYPE = "image"
TEXT_PART_TYPE = "text"


def _coerce_image_findings(
    findings: Iterable[ImageFinding | Mapping[str, Any]],
) -> list[ImageFinding]:
    """Coerce mixed image-finding inputs into validated ``ImageFinding`` models."""
    coerced: list[ImageFinding] = []
    for item in findings:
        if isinstance(item, ImageFinding):
            coerced.append(item)
        else:
            coerced.append(ImageFinding.model_validate(item))
    return coerced


def build_target_finding_dicts(
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the assistant-target finding dicts in MedGemma-native grounding form.

    Each present finding becomes an object with ``label``, ``box_2d`` (the
    canonical normalized ``[y0, x0, y1, x1]`` box or ``null``), and, when set on
    the source finding, ``confidence`` and ``evidence``. ``ABSENT`` findings are
    dropped (the grounding target asserts only what is on the image). An empty set
    of present findings yields a single ``no_finding`` element so the model learns
    to emit the negative sentinel rather than an empty list.

    Args:
        image_findings: The image-grounded findings (models or dicts).

    Returns:
        The list of finding dicts the assistant turn should emit.
    """
    coerced = _coerce_image_findings(image_findings)
    present = [finding for finding in coerced if finding.status is FindingStatus.PRESENT and finding.finding != NO_FINDING]

    if not present:
        return [{"label": NO_FINDING, "box_2d": None}]

    target: list[dict[str, Any]] = []
    for finding in present:
        element: dict[str, Any] = {
            "label": finding.finding,
            "box_2d": list(finding.box) if finding.box is not None else None,
        }
        if finding.confidence is not None:
            element["confidence"] = finding.confidence
        if finding.evidence is not None:
            element["evidence"] = finding.evidence
        target.append(element)
    return target


def build_assistant_target_text(
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
) -> str:
    """Serialize the assistant-target finding list to a compact JSON string.

    The string is exactly what the model is trained to emit and is parseable by
    ``schema.extract_finding_list``.

    Args:
        image_findings: The image-grounded findings (models or dicts).

    Returns:
        A JSON array string of finding objects.
    """
    return json.dumps(build_target_finding_dicts(image_findings), ensure_ascii=False)


def build_sft_record(
    image_path: str | Path,
    image_findings: Iterable[ImageFinding | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one chat-formatted SFT record for an image and its findings.

    Args:
        image_path: Path to the chest X-ray image file. Stored as a string on the
            record; the image bytes are loaded by the training collator, not here.
        image_findings: The image-grounded findings (models or dicts). Used to
            build the assistant target.

    Returns:
        The SFT record dict (one JSONL line).
    """
    prompt = build_image_grounding_prompt()
    target_text = build_assistant_target_text(image_findings)
    return {
        "image_path": str(image_path),
        "messages": [
            {
                "role": USER_ROLE,
                "content": [
                    {"type": IMAGE_PART_TYPE},
                    {"type": TEXT_PART_TYPE, "text": prompt},
                ],
            },
            {
                "role": ASSISTANT_ROLE,
                "content": [
                    {"type": TEXT_PART_TYPE, "text": target_text},
                ],
            },
        ],
    }


def _validate_content_parts(parts: Any, *, require_image: bool, role: str) -> None:
    """Validate a message ``content`` list of typed parts.

    Args:
        parts: The candidate content value.
        require_image: Whether an image part is required (the user turn).
        role: The role being validated, for error messages.

    Raises:
        ValueError: If the content is malformed or missing a required part.
    """
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"{role} content must be a non-empty list of parts")

    has_image = False
    has_text = False
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError(f"{role} content part must be an object")
        part_type = part.get("type")
        if part_type == IMAGE_PART_TYPE:
            has_image = True
        elif part_type == TEXT_PART_TYPE:
            if not isinstance(part.get("text"), str) or not part["text"]:
                raise ValueError(f"{role} text part must carry a non-empty 'text' string")
            has_text = True
        else:
            raise ValueError(f"{role} content part has unknown type {part_type!r}")

    if require_image and not has_image:
        raise ValueError(f"{role} content must include an image part")
    if not has_text:
        raise ValueError(f"{role} content must include a text part")


def validate_sft_record(record: Any) -> None:
    """Validate a single SFT record against the chat-formatted contract.

    Checks the top-level keys, the user/assistant message structure, that the user
    turn carries both an image part and the grounding prompt text, and that the
    assistant turn's text parses to a valid canonical finding list whose labels are
    all canonical.

    Args:
        record: The candidate record (typically a dict parsed from a JSONL line).

    Raises:
        ValueError: If the record violates the contract. The message identifies the
            offending field so a producer can repair it.
    """
    if not isinstance(record, dict):
        raise ValueError("SFT record must be an object")

    image_path = record.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("SFT record must carry a non-empty string 'image_path'")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("SFT record 'messages' must be a list of exactly two turns")

    user_message, assistant_message = messages
    if not isinstance(user_message, dict) or user_message.get("role") != USER_ROLE:
        raise ValueError("first message must have role 'user'")
    if not isinstance(assistant_message, dict) or assistant_message.get("role") != ASSISTANT_ROLE:
        raise ValueError("second message must have role 'assistant'")

    _validate_content_parts(user_message.get("content"), require_image=True, role=USER_ROLE)
    _validate_content_parts(assistant_message.get("content"), require_image=False, role=ASSISTANT_ROLE)

    assistant_text = next(part["text"] for part in assistant_message["content"] if part.get("type") == TEXT_PART_TYPE)
    findings = extract_finding_list(assistant_text)
    if not findings:
        raise ValueError("assistant target must contain at least one finding object")
    for element in findings:
        label = element.get("label")
        if not isinstance(label, str):
            raise ValueError("assistant finding element must carry a 'label' string")
        # normalize_finding raises ValueError if the label is not canonical.
        normalize_finding(label)


def write_sft_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    validate: bool = True,
) -> int:
    """Write SFT records to a JSONL file, one record per line.

    Args:
        records: The SFT records to write (typically from ``build_sft_record``).
        output_path: Destination JSONL path. Parent directories are created.
        validate: When true (default) every record is validated via
            ``validate_sft_record`` before writing, so a malformed record fails
            fast rather than producing a corpus that breaks training.

    Returns:
        The number of records written.

    Raises:
        ValueError: If ``validate`` is true and any record is invalid. No partial
            file is left behind in that case.
    """
    path = Path(output_path)
    materialized = list(records)
    if validate:
        for index, record in enumerate(materialized):
            try:
                validate_sft_record(record)
            except ValueError as exc:
                raise ValueError(f"record {index} failed SFT validation: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return len(materialized)


def read_sft_jsonl(input_path: str | Path, *, validate: bool = True) -> list[dict[str, Any]]:
    """Read an SFT JSONL file into a list of records.

    Args:
        input_path: Source JSONL path.
        validate: When true (default) every parsed record is validated via
            ``validate_sft_record``.

    Returns:
        The parsed records in file order.

    Raises:
        ValueError: If a line is not valid JSON, or ``validate`` is true and a
            record is invalid. The error identifies the offending line number.
    """
    path = Path(input_path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
            if validate:
                try:
                    validate_sft_record(record)
                except ValueError as exc:
                    raise ValueError(f"line {line_number} failed SFT validation: {exc}") from exc
            records.append(record)
    return records


def build_sft_corpus(
    examples: Sequence[tuple[str | Path, Iterable[ImageFinding | Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Build an SFT corpus from (image_path, findings) pairs.

    Args:
        examples: A sequence of ``(image_path, image_findings)`` pairs.

    Returns:
        The list of SFT records, one per example, in input order.
    """
    return [build_sft_record(image_path, findings) for image_path, findings in examples]


def describe_corpus_target_format() -> dict[str, str]:
    """Return a small descriptor of the assistant-target format for documentation.

    Useful for emitting a corpus manifest alongside the JSONL so a training run
    records which box format and roles the targets use.

    Returns:
        A dict with the box format and the two chat roles.
    """
    return {
        "box_format": CANONICAL_BOX_FORMAT,
        "user_role": USER_ROLE,
        "assistant_role": ASSISTANT_ROLE,
    }
