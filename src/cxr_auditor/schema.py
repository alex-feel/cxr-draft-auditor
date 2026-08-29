"""
Canonical output schema, tolerant model-text parsing, and box conversions.

This module defines the pydantic v2 models for the auditor's canonical output
JSON, a tolerant parser that turns raw model text into a validated object, and
helpers for converting bounding boxes between the canonical normalized format and
absolute pixel coordinates.

Box format
----------
The canonical box format is the string constant ``CANONICAL_BOX_FORMAT`` whose
value is ``'normalized_y0x0y1x1'``. A box is a 4-tuple of floats in ``[0, 1]``
ordered ``[y0, x0, y1, x1]`` where ``(y0, x0)`` is the top-left corner and
``(y1, x1)`` is the bottom-right corner, normalized by image height (the y axis)
and image width (the x axis). This matches MedGemma's native ``box_2d`` emission.

Dependencies are limited to the standard library, numpy, and pydantic so the
schema can be imported and validated without any model-serving stack.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The canonical bounding-box format identifier. A box is [y0, x0, y1, x1],
# each component a float in [0, 1], (y0, x0) top-left, (y1, x1) bottom-right.
CANONICAL_BOX_FORMAT: str = "normalized_y0x0y1x1"

# The standard disclaimer string embedded in every canonical output object.
DISCLAIMER_TEXT: str = (
    "Research/educational QA only. NOT a medical device, NOT diagnosis, "
    "NOT for clinical use. Outputs are frequently wrong; always consult a "
    "qualified radiologist."
)

# A normalized box: four floats in [0, 1] ordered y0, x0, y1, x1.
type NormalizedBox = tuple[float, float, float, float]

# An absolute-pixel xyxy box: x_min, y_min, x_max, y_max in pixels.
type XYXYBox = tuple[float, float, float, float]


def normalize_finding(value: str) -> str:
    """Normalize a raw finding label to a canonical finding string.

    Absorbs the common formatting drift a vision-language model produces when
    emitting label names: surrounding whitespace, casing, and the use of spaces
    or hyphens as word separators where the canonical form uses underscores (for
    example ``'Pleural Effusion'`` and ``'pleural-effusion'`` both normalize to
    ``'pleural_effusion'``). A genuinely unknown label that does not match the
    canonical set after normalization is rejected.

    Args:
        value: The raw finding label as produced by the model or a dataset.

    Returns:
        The canonical finding string.

    Raises:
        ValueError: If the normalized label is not one of the six canonical
            findings.
    """
    from cxr_auditor.findings import CANONICAL_FINDING_SET

    normalized = re.sub(r"[\s\-]+", "_", value.strip().lower())
    if normalized not in CANONICAL_FINDING_SET:
        raise ValueError(f"finding {value!r} is not a canonical finding; expected one of {sorted(CANONICAL_FINDING_SET)}")
    return normalized


class FindingStatus(str, Enum):
    """Whether a finding is asserted present or explicitly denied.

    ``PRESENT`` means the source (image or draft) asserts the finding exists.
    ``ABSENT`` means the source explicitly denies it (for example a draft saying
    "no pneumothorax"). Findings that a draft does not mention at all are simply
    omitted from the finding list rather than recorded as ``ABSENT``.
    """

    PRESENT = "present"
    ABSENT = "absent"


class ImageFinding(BaseModel):
    """A single image-grounded finding with a normalized bounding box.

    Attributes:
        finding: A canonical finding string (validated against the canonical set).
        status: Whether the finding is present or absent on the image.
        box: The normalized bounding box ``[y0, x0, y1, x1]`` in ``[0, 1]``.
            ``None`` is permitted for findings with no localizable box (for
            example ``no_finding`` or a diffuse global finding).
        confidence: Optional model confidence in ``[0, 1]``.
        evidence: Optional short free-text rationale describing the image
            evidence (for example "blunting of the left costophrenic angle").
    """

    model_config = ConfigDict(extra="forbid")

    finding: str
    status: FindingStatus = FindingStatus.PRESENT
    box: NormalizedBox | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None

    @field_validator("finding")
    @classmethod
    def _validate_finding(cls, value: str) -> str:
        return normalize_finding(value)

    @field_validator("box")
    @classmethod
    def _validate_box(cls, value: NormalizedBox | None) -> NormalizedBox | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError(f"box must have exactly 4 components, got {len(value)}")
        y0, x0, y1, x1 = value
        for component in value:
            if not 0.0 <= component <= 1.0:
                raise ValueError(f"box components must be in [0, 1], got {value!r}")
        if y1 < y0 or x1 < x0:
            raise ValueError(f"box must satisfy y1 >= y0 and x1 >= x0 (top-left to bottom-right), got {value!r}")
        return (float(y0), float(x0), float(y1), float(x1))


class DraftFinding(BaseModel):
    """A finding extracted from a draft impression, in the canonical label space.

    Attributes:
        finding: A canonical finding string (validated against the canonical set).
        status: Whether the draft asserts the finding present or denies it.
        span: Optional verbatim text span from the draft that produced this label.
    """

    model_config = ConfigDict(extra="forbid")

    finding: str
    status: FindingStatus = FindingStatus.PRESENT
    span: str | None = None

    @field_validator("finding")
    @classmethod
    def _validate_finding(cls, value: str) -> str:
        return normalize_finding(value)


class Audit(BaseModel):
    """The deterministic comparator result.

    Attributes:
        missing_findings: Canonical findings present on the image but absent or
            denied in the draft.
        unsupported_claims: Canonical findings asserted in the draft but absent
            from the image findings.
        urgent_review_flags: Canonical findings present on the image that are on
            the urgent-review whitelist.
    """

    model_config = ConfigDict(extra="forbid")

    missing_findings: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    urgent_review_flags: list[str] = Field(default_factory=list)


class AuditResult(BaseModel):
    """The complete canonical output object emitted by the auditor.

    Attributes:
        image_findings: The image-grounded findings with bounding-box evidence.
        draft_findings: The draft impression parsed into the canonical labels.
            Empty when no draft was supplied.
        audit: The deterministic comparator result.
        disclaimer: The standard research/educational disclaimer. Defaults to
            ``DISCLAIMER_TEXT``.
        box_format: The bounding-box format identifier. Always
            ``CANONICAL_BOX_FORMAT``.
    """

    model_config = ConfigDict(extra="forbid")

    image_findings: list[ImageFinding] = Field(default_factory=list)
    draft_findings: list[DraftFinding] = Field(default_factory=list)
    audit: Audit = Field(default_factory=Audit)
    disclaimer: str = DISCLAIMER_TEXT
    box_format: str = CANONICAL_BOX_FORMAT


class SchemaParseError(ValueError):
    """Raised when raw model text cannot be parsed into a valid schema object.

    Carries the original raw text so a retry/repair loop can inspect it.
    """

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text

    def __reduce__(self) -> tuple[type[SchemaParseError], tuple[str, str]]:
        """Support pickling across process boundaries (for example a GPU worker).

        The default exception reduction re-invokes the class with ``self.args``
        only, which omits the required ``raw_text`` argument; returning both
        constructor arguments keeps the error fully reconstructable.

        Returns:
            The ``(callable, args)`` pair pickle uses to rebuild the error.
        """
        return (type(self), (str(self.args[0]) if self.args else "", self.raw_text))


def _find_balanced_end(text: str, start: int, open_char: str, close_char: str) -> int:
    """Find the index of the close delimiter balancing ``text[start]``.

    Walks the string from ``start`` (which must point at ``open_char``) tracking
    nesting depth while respecting JSON string literals and escape sequences, so
    a delimiter inside a quoted string never affects the depth count.

    Args:
        text: The text to scan.
        start: Index of the opening delimiter to balance.
        open_char: The opening delimiter (for example ``"{"`` or ``"["``).
        close_char: The matching closing delimiter (``"}"`` or ``"]"``).

    Returns:
        The index of the balancing close delimiter, or ``-1`` when the text ends
        before the delimiter closes.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced top-level JSON object from raw model text.

    Vision-language models frequently wrap their JSON in prose, markdown code
    fences, or trailing commentary. This scans for the first ``{`` and walks the
    string tracking brace depth (while respecting JSON string literals and escape
    sequences) to find the matching close brace, then parses that slice.

    Args:
        text: Raw model output that contains a JSON object somewhere inside it.

    Returns:
        The parsed object as a dict.

    Raises:
        SchemaParseError: If no balanced JSON object is found, or the candidate
            slice is not valid JSON, or the top-level value is not an object.
    """
    start = text.find("{")
    if start == -1:
        raise SchemaParseError("no JSON object found in model text", text)

    end = _find_balanced_end(text, start, "{", "}")
    if end == -1:
        raise SchemaParseError("no balanced JSON object found in model text", text)

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaParseError(f"candidate JSON slice is invalid: {exc}", text) from exc

    if not isinstance(parsed, dict):
        raise SchemaParseError("top-level JSON value is not an object", text)
    return parsed


def parse_model_output(text: str) -> AuditResult:
    """Parse raw model text into a validated ``AuditResult``.

    Performs tolerant JSON extraction (``extract_first_json_object``) followed by
    pydantic validation. This is the strict parser used after a model produces a
    full canonical output object.

    Args:
        text: Raw model output containing the canonical JSON somewhere inside.

    Returns:
        A validated ``AuditResult``.

    Raises:
        SchemaParseError: If extraction or validation fails. The raw text is
            attached so a repair/retry loop can re-prompt the model.
    """
    obj = extract_first_json_object(text)
    try:
        return AuditResult.model_validate(obj)
    except ValueError as exc:
        raise SchemaParseError(f"model JSON failed schema validation: {exc}", text) from exc


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the text is fenced."""
    fence = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    return text


# Upper bound on elements recovered by ``salvage_finding_list``. A degenerate
# repetition loop can emit the same element until the token budget is exhausted;
# the cap bounds the salvage work while comfortably exceeding any realistic
# finding count for one image.
_SALVAGE_MAX_ELEMENTS: int = 64


def salvage_finding_list(text: str) -> list[dict[str, Any]]:
    """Recover the complete leading elements of a truncated or malformed array.

    A generation that exhausts its token budget mid-array leaves the JSON array
    unclosed (or its tail malformed), which would otherwise discard every element
    the model emitted. This walks the array's elements from the first ``[``,
    extracting each balanced ``{...}`` slice and parsing it independently, and
    stops at the first incomplete or invalid element, at the array's closing
    bracket, or after ``_SALVAGE_MAX_ELEMENTS`` elements - so the complete
    leading elements survive a broken tail. Markdown code fences are stripped
    before scanning.

    Args:
        text: Raw model output containing at least the head of a JSON array.

    Returns:
        The successfully recovered element dicts, possibly empty.
    """
    stripped = _strip_code_fences(text)
    array_start = stripped.find("[")
    if array_start == -1:
        return []

    elements: list[dict[str, Any]] = []
    cursor = array_start + 1
    while len(elements) < _SALVAGE_MAX_ELEMENTS:
        element_start = stripped.find("{", cursor)
        if element_start == -1:
            break
        # A closing bracket between elements means the array ended; anything
        # after it lies outside the array and must not be salvaged into it.
        if "]" in stripped[cursor:element_start]:
            break
        element_end = _find_balanced_end(stripped, element_start, "{", "}")
        if element_end == -1:
            break
        candidate = stripped[element_start : element_end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if not isinstance(parsed, dict):
            break
        elements.append(parsed)
        cursor = element_end + 1
    return elements


def extract_finding_list(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array of finding dicts from raw model text.

    MedGemma's native grounding output is a JSON *list* of ``{label, box_2d}``
    objects rather than a wrapping object. This finds the first balanced
    top-level JSON array and parses it. A bare object is tolerated and wrapped in
    a single-element list. When the array never closes or its slice is invalid
    JSON (a truncated or degenerate generation), the complete leading elements
    are recovered via ``salvage_finding_list`` before declaring failure.

    Args:
        text: Raw model output containing a JSON array (or a single object).

    Returns:
        A list of dicts (one per finding). Non-dict array elements are rejected.

    Raises:
        SchemaParseError: If no balanced JSON array/object is found and nothing
            can be salvaged, the slice is invalid JSON and nothing can be
            salvaged, or a well-formed array contains a non-object element.
    """
    stripped = _strip_code_fences(text)

    array_start = stripped.find("[")
    object_start = stripped.find("{")
    # If an object appears before any array (or no array exists), fall back to
    # single-object extraction and wrap it.
    if array_start == -1 or (object_start != -1 and object_start < array_start):
        return [extract_first_json_object(stripped)]

    end = _find_balanced_end(stripped, array_start, "[", "]")
    if end == -1:
        salvaged = salvage_finding_list(text)
        if salvaged:
            return salvaged
        raise SchemaParseError("no balanced JSON array found in model text", text)

    candidate = stripped[array_start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        salvaged = salvage_finding_list(text)
        if salvaged:
            return salvaged
        raise SchemaParseError(f"candidate JSON array is invalid: {exc}", text) from exc

    if not isinstance(parsed, list):
        raise SchemaParseError("top-level JSON value is not an array", text)
    for element in parsed:
        if not isinstance(element, dict):
            raise SchemaParseError("JSON array element is not an object", text)
    return parsed


def normalized_to_xyxy_abs(box: NormalizedBox, width: int, height: int) -> XYXYBox:
    """Convert a canonical normalized box to absolute-pixel xyxy coordinates.

    Args:
        box: Canonical normalized box ``[y0, x0, y1, x1]`` in ``[0, 1]``.
        width: Image width in pixels (x axis).
        height: Image height in pixels (y axis).

    Returns:
        An xyxy box ``(x_min, y_min, x_max, y_max)`` in pixels.

    Raises:
        ValueError: If width or height is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    y0, x0, y1, x1 = box
    return (x0 * width, y0 * height, x1 * width, y1 * height)


def xyxy_abs_to_normalized(box: XYXYBox, width: int, height: int) -> NormalizedBox:
    """Convert an absolute-pixel xyxy box to the canonical normalized format.

    Args:
        box: An xyxy box ``(x_min, y_min, x_max, y_max)`` in pixels.
        width: Image width in pixels (x axis).
        height: Image height in pixels (y axis).

    Returns:
        A canonical normalized box ``[y0, x0, y1, x1]``. Components are clamped to
        ``[0, 1]`` to absorb sub-pixel rounding at the image edges.

    Raises:
        ValueError: If width or height is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    x_min, y_min, x_max, y_max = box

    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, value))

    return (
        _clamp(y_min / height),
        _clamp(x_min / width),
        _clamp(y_max / height),
        _clamp(x_max / width),
    )


def from_qwen_box(box: tuple[float, float, float, float], width: int, height: int) -> NormalizedBox:
    """Convert a Qwen-style absolute xyxy box to the canonical normalized format.

    Qwen2.5-VL / Qwen3-VL grounding emits boxes as absolute pixel
    ``[x_min, y_min, x_max, y_max]`` against the input image resolution. This is
    the same ordering as ``XYXYBox``; the helper exists so call sites reading
    Qwen output are explicit about the source convention.

    Args:
        box: A Qwen-style absolute-pixel box ``[x_min, y_min, x_max, y_max]``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A canonical normalized box ``[y0, x0, y1, x1]``.
    """
    return xyxy_abs_to_normalized(box, width, height)
