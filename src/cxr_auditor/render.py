"""
Presentation helpers: turn an audit outcome into the artifacts the app displays.

The Gradio app shows four things for every audit: an annotated overlay image with
colored evidence boxes, a structured findings table, a Markdown audit panel, and
the raw canonical JSON. This module builds all four from an
``inference.AuditOutcome`` and depends only on PIL and the pure-logic
schema/comparator types - no gradio, torch, or GPU - so the visual logic is
unit-testable on its own.

Color legend
------------
- green: a draft claim the image supports (present in both image and draft).
- amber: a finding present on the image that the draft misses or denies.
- red: a draft claim the image does not support, and any urgent finding.
- urgent: an image-present finding on the urgent whitelist (for example a
  pneumothorax or a pulmonary nodule/mass) is drawn red with a thick, doubled
  border and an "(URGENT)" tag so it stands out regardless of whether the draft
  mentioned it.

Every label shown to a person uses the human-readable display name from
``findings.display_name`` (the single source of truth), never the raw
``snake_case`` canonical label; the raw label is kept only in the machine JSON.

Overlay deduplication
---------------------
The model sometimes localizes two different findings (for example a lung opacity
and a nodule) to the same region. Drawing one rectangle per finding would stack
several boxes and labels at identical pixels, and the last-drawn color would mask
the others. ``cluster_overlay_boxes`` merges spatially-overlapping boxes into one
``OverlayBox`` whose color is chosen by status severity, so each physical region
is drawn exactly once with the correct, order-independent color and a single
combined label.

The findings table and the audit panel are deduplicated by canonical label (the
audit reasons about each label once; see ``comparator``), while the overlay keeps
one box per distinct region. A label the model localizes at several separate
sites therefore appears once in the table and panel but draws several boxes, so
the overlay can legitimately show more boxes than the table or panel have rows.
``findings_table_rows`` appends an "(N foci)" hint to such a label so a reader can
reconcile the single table row with the several boxes on the image.

Label collision avoidance
-------------------------
Distinct regions can still carry long labels at nearly the same height (for
example bilateral opacities over the two lung fields), where independently
placed label bands would overlap and clip each other. Each band is therefore
checked against the bands already drawn on the canvas and nudged vertically
(below its own box first) until it overlaps none of them, always staying
clamped on-image; a lone label keeps its exact anchored position.

Unsupported claims are draft-only and have no image box, so they appear in the
table and the audit panel rather than as overlay boxes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from PIL import Image, ImageDraw, ImageFont

from cxr_auditor.findings import display_name, is_urgent
from cxr_auditor.inference import AuditOutcome
from cxr_auditor.schema import FindingStatus, NormalizedBox, XYXYBox, normalized_to_xyxy_abs

# Pillow's scalable default font resolves to ``FreeTypeFont`` when FreeType is
# available and falls back to the bitmap ``ImageFont`` otherwise; both are valid
# arguments to ``ImageDraw.text`` and ``ImageDraw.textbbox``.
type LoadedFont = ImageFont.FreeTypeFont | ImageFont.ImageFont

# The overlay is an evidence visualization, not a diagnostic image, so it is
# rendered on a canvas capped at this many pixels on the long side. Gradio shows
# the overlay fit-to-column (roughly 700-900 px wide), so native resolution beyond
# the cap adds no on-screen legibility; bounding the canvas makes font and
# line-width proportions predictable for every input size.
_MAX_OVERLAY_LONG_SIDE = 1280

# Label font band in canvas pixels. The floor keeps labels readable on small
# canvases; the cap keeps a long combined label modest on canvas-capped large ones.
_MIN_LABEL_FONT_SIZE = 14
_MAX_LABEL_FONT_SIZE = 28

# Vertical search budget for nudging a colliding label band: this many
# band-height steps downward and again upward. At the tallest band a capped
# canvas produces (about 37 px) eight steps sweep roughly 300 px each way -
# ample clearance for the handful of labels one audit draws - while keeping
# the candidate count, and therefore the worst-case drawing work, bounded.
_LABEL_NUDGE_ATTEMPTS = 8

# RGB colors for each evidence category. Chosen for contrast on a grayscale X-ray.
_SUPPORTED_COLOR = (46, 204, 113)
_MISSING_COLOR = (243, 156, 18)
# Unsupported draft claims and urgent findings share the same red: both are the
# highest-attention category in their respective channel (unsupported in the
# table/panel, urgent on the overlay), so they intentionally use one named value.
_ALERT_COLOR = (231, 76, 60)


class EvidenceCategory(str, Enum):
    """The audit category an overlay box belongs to.

    ``SUPPORTED`` and ``MISSING`` describe image-present findings; ``UNSUPPORTED``
    describes a draft-only claim (no image box). ``URGENT`` overrides the others
    for any image-present finding on the urgent whitelist.
    """

    SUPPORTED = "supported"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    URGENT = "urgent"


# Severity precedence for resolving the category of a merged cluster: the most
# severe member category wins the single drawn box, so an overlapping
# supported+missing region draws as missing and a region that includes an urgent
# finding draws as urgent. Higher number = more severe.
_CATEGORY_SEVERITY: dict[EvidenceCategory, int] = {
    EvidenceCategory.SUPPORTED: 0,
    EvidenceCategory.MISSING: 1,
    EvidenceCategory.UNSUPPORTED: 2,
    EvidenceCategory.URGENT: 3,
}


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One image-grounded finding categorized for overlay drawing.

    Attributes:
        finding: The canonical finding label.
        box: The normalized box, or ``None`` when the model localized none.
        category: The audit category that drives the box color and style.
        urgent: Whether the finding is on the urgent-review whitelist.
        confidence: Optional model confidence in ``[0, 1]``.
    """

    finding: str
    box: NormalizedBox | None
    category: EvidenceCategory
    urgent: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class OverlayBox:
    """A cluster of overlapping evidence items collapsed into one drawn box.

    A physical region the model labels with several findings (for example an
    opacity and a nodule at the same coordinates) becomes a single ``OverlayBox``
    so the overlay draws one rectangle and one combined label there instead of
    several stacked, illegible ones.

    Attributes:
        box: The representative normalized box for the cluster (the first member's
            box; overlapping members share essentially the same region).
        category: The most-severe member category (drives the box color).
        labels: The member canonical findings, in first-seen order with duplicates
            removed.
        urgent: Whether any member is on the urgent-review whitelist.
    """

    box: NormalizedBox
    category: EvidenceCategory
    labels: tuple[str, ...]
    urgent: bool


def categorize_image_findings(outcome: AuditOutcome) -> list[EvidenceItem]:
    """Categorize each image-present finding for the overlay.

    A present image finding is ``SUPPORTED`` when the draft also asserts it,
    ``MISSING`` when the draft omits or denies it, and ``URGENT`` when its label is
    on the urgent whitelist (urgency overrides the supported/missing distinction so
    a critical finding is always visually prominent). The ``no_finding`` sentinel
    and any ``ABSENT`` image finding contribute no overlay item.

    Args:
        outcome: The audit outcome (result plus comparator detail).

    Returns:
        Evidence items in the order the model emitted the findings.
    """
    missing_labels = set(outcome.result.audit.missing_findings)
    items: list[EvidenceItem] = []
    for finding in outcome.result.image_findings:
        if finding.status is not FindingStatus.PRESENT:
            continue
        if not _is_positive(finding.finding):
            continue
        urgent = is_urgent(finding.finding)
        if urgent:
            category = EvidenceCategory.URGENT
        elif finding.finding in missing_labels:
            category = EvidenceCategory.MISSING
        else:
            category = EvidenceCategory.SUPPORTED
        items.append(
            EvidenceItem(
                finding=finding.finding,
                box=finding.box,
                category=category,
                urgent=urgent,
                confidence=finding.confidence,
            )
        )
    return items


def _is_positive(label: str) -> bool:
    """Return whether a label is a positive finding (not the negative sentinel)."""
    from cxr_auditor.findings import NO_FINDING

    return label != NO_FINDING


def _color_for(category: EvidenceCategory) -> tuple[int, int, int]:
    """Return the RGB color for an evidence category."""
    if category is EvidenceCategory.SUPPORTED:
        return _SUPPORTED_COLOR
    if category is EvidenceCategory.MISSING:
        return _MISSING_COLOR
    return _ALERT_COLOR


def _iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """Return the intersection-over-union of two normalized boxes.

    Both boxes are ``[y0, x0, y1, x1]`` in ``[0, 1]``. The computation is scale-free
    (it works directly in normalized space), so no image dimensions are needed.
    Returns ``0.0`` when the boxes do not overlap or either has degenerate area.
    """
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    inter_y0 = max(ay0, by0)
    inter_x0 = max(ax0, bx0)
    inter_y1 = min(ay1, by1)
    inter_x1 = min(ax1, bx1)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_w = max(0.0, inter_x1 - inter_x0)
    intersection = inter_h * inter_w
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def cluster_overlay_boxes(items: list[EvidenceItem], iou_threshold: float = 0.9) -> list[OverlayBox]:
    """Merge spatially-overlapping evidence items into one drawn box each.

    A greedy single pass: each localized item either joins the first existing
    cluster whose representative box overlaps it (IoU >= ``iou_threshold``) or
    starts a new cluster. Merging collapses the stacked rectangles the model
    produces when it labels one region with several findings (for example an
    opacity and a nodule at identical coordinates) into a single box. The cluster's
    category is the most severe member category (so the color reflects status, not
    paint order), and the cluster is urgent if any member is urgent.

    Items with no box (``box is None``) contribute no overlay and are skipped.
    Cluster order is stable: clusters appear in the order their first member did.

    Args:
        items: The categorized image findings to draw.
        iou_threshold: Minimum intersection-over-union for two boxes to merge.

    Returns:
        One ``OverlayBox`` per distinct physical region, in first-seen order.
    """
    clusters: list[_ClusterAccumulator] = []
    for item in items:
        if item.box is None:
            continue
        merged = False
        for cluster in clusters:
            if _iou(cluster.box, item.box) >= iou_threshold:
                cluster.add(item)
                merged = True
                break
        if not merged:
            clusters.append(_ClusterAccumulator(item))
    return [cluster.finalize() for cluster in clusters]


@dataclass(slots=True)
class _ClusterAccumulator:
    """Mutable helper that accumulates overlapping items into one ``OverlayBox``."""

    box: NormalizedBox
    category: EvidenceCategory
    labels: list[str]
    urgent: bool

    def __init__(self, item: EvidenceItem) -> None:
        assert item.box is not None  # callers skip box-less items before adding
        self.box = item.box
        self.category = item.category
        self.labels = [item.finding]
        self.urgent = item.urgent

    def add(self, item: EvidenceItem) -> None:
        """Fold another overlapping item into this cluster."""
        if item.finding not in self.labels:
            self.labels.append(item.finding)
        if _CATEGORY_SEVERITY[item.category] > _CATEGORY_SEVERITY[self.category]:
            self.category = item.category
        self.urgent = self.urgent or item.urgent

    def finalize(self) -> OverlayBox:
        """Return the immutable ``OverlayBox`` for this cluster."""
        return OverlayBox(box=self.box, category=self.category, labels=tuple(self.labels), urgent=self.urgent)


def _scaled_font(width: int, height: int) -> LoadedFont:
    """Return a built-in font scaled to the canvas so labels are legible, never huge.

    Pillow's default font is scalable, so the label text tracks the canvas size
    (roughly one-48th of the longer side) without bundling any ``.ttf`` asset. The
    size is clamped to ``[_MIN_LABEL_FONT_SIZE, _MAX_LABEL_FONT_SIZE]``: the floor
    keeps labels readable on small canvases, and the cap (which the
    ``_MAX_OVERLAY_LONG_SIDE`` canvas bound makes effective) keeps a long combined
    label from dominating the overlay.
    """
    size = round(max(width, height) / 48)
    return ImageFont.load_default(size=min(_MAX_LABEL_FONT_SIZE, max(_MIN_LABEL_FONT_SIZE, size)))


def _combined_label(cluster: OverlayBox) -> str:
    """Return the human-readable label for a cluster.

    Joins the member findings' display names with ``" / "`` (so an opacity+nodule
    region reads as one combined label) and appends an ``" (URGENT)"`` tag when the
    cluster is urgent.
    """
    text = " / ".join(display_name(label) for label in cluster.labels)
    if cluster.urgent:
        text += " (URGENT)"
    return text


def _rects_overlap(a: XYXYBox, b: XYXYBox) -> bool:
    """Return whether two pixel rectangles share interior area.

    Rectangles are ``(left, top, right, bottom)``. Edge-touching rectangles do
    not count as overlapping, so nudged label bands may sit flush against one
    another without triggering a further nudge.
    """
    a_left, a_top, a_right, a_bottom = a
    b_left, b_top, b_right, b_bottom = b
    return a_left < b_right and b_left < a_right and a_top < b_bottom and b_top < a_bottom


def _resolve_band_rect(
    desired: XYXYBox,
    box_xyxy: XYXYBox,
    image_height: int,
    placed_bands: Sequence[XYXYBox],
) -> XYXYBox:
    """Return the label-band rectangle to draw, nudged vertically clear of placed bands.

    A desired rectangle that overlaps no already-placed band is returned
    unchanged, so a lone label renders exactly at its anchored position. On
    collision the band keeps its horizontal extent (it stays anchored to its
    box) and tries vertical positions in order: just below the box's bottom
    edge, then band-height steps downward from there, then band-height steps
    upward from the desired position, up to ``_LABEL_NUDGE_ATTEMPTS`` steps per
    direction. Every candidate is clamped fully on-image; when no candidate
    clears all placed bands the last clamped candidate is returned, so the
    result is always an on-image rectangle and rendering never fails.

    Args:
        desired: The preferred band rectangle ``(left, top, right, bottom)``,
            already clamped on-image by the caller.
        box_xyxy: The owning box in absolute pixels ``(x_min, y_min, x_max,
            y_max)``, anchoring the below-box candidate.
        image_height: The canvas height in pixels, for vertical clamping.
        placed_bands: Band rectangles already drawn on this canvas.

    Returns:
        The chosen band rectangle ``(left, top, right, bottom)``.
    """
    left, desired_top, right, desired_bottom = desired
    band_height = desired_bottom - desired_top
    lowest_top = max(0.0, image_height - band_height)
    below_box_top = min(box_xyxy[3], lowest_top)

    candidate_tops = [desired_top, below_box_top]
    candidate_tops.extend(min(below_box_top + step * band_height, lowest_top) for step in range(1, _LABEL_NUDGE_ATTEMPTS + 1))
    candidate_tops.extend(max(0.0, desired_top - step * band_height) for step in range(1, _LABEL_NUDGE_ATTEMPTS + 1))

    band = desired
    for top in candidate_tops:
        band = (left, top, right, top + band_height)
        if not any(_rects_overlap(band, placed) for placed in placed_bands):
            return band
    return band


def _draw_label(
    draw: ImageDraw.ImageDraw,
    box_xyxy: XYXYBox,
    text: str,
    *,
    color: tuple[int, int, int],
    font: LoadedFont,
    image_size: tuple[int, int],
    placed_bands: Sequence[XYXYBox],
) -> XYXYBox:
    """Draw a filled, high-contrast label band anchored to a box, clamped on-image.

    The label sits just above the box top by default, but flips to just below the
    box top when there is no room above (the box touches the top edge), and its
    left edge is clamped so the band never runs off the right side of the image.
    When the resulting band would overlap a band already drawn on this canvas it
    is nudged vertically clear (see ``_resolve_band_rect``) so neighboring labels
    never clip each other. The band is filled with ``color`` and the text is
    drawn in a contrasting ink so it is legible over a grayscale X-ray.

    Args:
        draw: The active drawing context.
        box_xyxy: The box in absolute pixels ``(x_min, y_min, x_max, y_max)``.
        text: The label text (already human-readable).
        color: The band fill color (the cluster's status color).
        font: The scaled font to measure and render with.
        image_size: ``(width, height)`` of the canvas, for on-image clamping.
        placed_bands: Band rectangles already drawn on this canvas, used to
            resolve collisions; the caller records the returned rectangle.

    Returns:
        The band rectangle ``(left, top, right, bottom)`` actually drawn.
    """
    image_width, image_height = image_size
    x_min, y_min, _x_max, _y_max = box_xyxy

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad = max(2, text_height // 4)
    band_height = text_height + 2 * pad
    band_width = text_width + 2 * pad

    band_top = y_min - band_height
    if band_top < 0:
        # No room above the box; place the band just below the box top instead.
        band_top = y_min
    band_left = x_min
    if band_left + band_width > image_width:
        band_left = max(0, image_width - band_width)
    band_top = max(0, min(band_top, image_height - band_height))

    desired = (band_left, band_top, band_left + band_width, band_top + band_height)
    band = _resolve_band_rect(desired, box_xyxy, image_height, placed_bands)

    draw.rectangle(band, fill=color)
    draw.text((band[0] + pad, band[1] + pad), text, fill=_contrast_ink(color), font=font)
    return band


def _contrast_ink(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white, whichever reads better on ``color``.

    Uses the standard perceived-luminance formula; light backgrounds get black
    text and dark backgrounds get white text.
    """
    red, green, blue = color
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def annotate_evidence(image: Image.Image, outcome: AuditOutcome) -> Image.Image:
    """Draw one colored, labeled evidence box per physical region.

    Returns a new RGB image; the input is never mutated. The drawing canvas is
    capped at ``_MAX_OVERLAY_LONG_SIDE`` pixels on the long side (aspect ratio
    preserved; smaller inputs keep their size): the overlay is an evidence
    visualization shown fit-to-column, so the bounded canvas keeps label and border
    proportions predictable for every input size. Overlapping findings are merged
    (see ``cluster_overlay_boxes``) so each region is drawn exactly once, colored by
    audit status (see the module color legend). Urgent regions are drawn in red with
    a doubled, thicker border and an "(URGENT)" tag so they stand out. Every label
    uses the human-readable display name, and label bands that would overlap an
    earlier band are nudged vertically clear of it (see ``_resolve_band_rect``) so
    neighboring labels stay readable. Findings without a localizable box
    contribute no drawing (they still appear in the table and panel).

    Args:
        image: The source chest X-ray (any PIL mode; converted to RGB).
        outcome: The audit outcome to visualize.

    Returns:
        A new RGB ``Image.Image`` with boxes drawn, downscaled to the canvas cap
        when the input's long side exceeds it.
    """
    canvas = image.convert("RGB")
    if max(canvas.size) > _MAX_OVERLAY_LONG_SIDE:
        scale = _MAX_OVERLAY_LONG_SIDE / max(canvas.size)
        new_size = (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale)))
        canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    font = _scaled_font(width, height)

    # Line width scales with the same bounded long side the font uses, so borders
    # stay proportionate on both tiny test fixtures and canvas-capped X-rays.
    base_width = max(2, round(max(width, height) / 320))

    # Bands already drawn on this canvas; each new label is nudged clear of them.
    placed_bands: list[XYXYBox] = []
    for cluster in cluster_overlay_boxes(categorize_image_findings(outcome)):
        box_xyxy = normalized_to_xyxy_abs(cluster.box, width, height)
        x_min, y_min, x_max, y_max = box_xyxy
        color = _color_for(cluster.category)
        line_width = base_width * 2 if cluster.urgent else base_width
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=line_width)
        if cluster.urgent:
            # A doubled inset border emphasizes urgent findings.
            inset = line_width
            draw.rectangle(
                (x_min + inset, y_min + inset, x_max - inset, y_max - inset),
                outline=color,
                width=max(1, base_width),
            )
        placed_bands.append(
            _draw_label(
                draw,
                box_xyxy,
                _combined_label(cluster),
                color=color,
                font=font,
                image_size=(width, height),
                placed_bands=placed_bands,
            )
        )

    return canvas


# Source labels and plain-English audit-flag phrases shown in the findings table.
_SOURCE_IMAGE = "Image"
_SOURCE_DRAFT = "Draft"
_FLAG_SUPPORTED = "Supported by image"
_FLAG_MISSING = "Missing from draft"
_FLAG_URGENT = "Urgent - review"
_FLAG_UNSUPPORTED = "Unsupported claim"
_FLAG_NONE = ""


def _foci_counts(outcome: AuditOutcome) -> dict[str, int]:
    """Count the distinct overlay regions drawn for each canonical label.

    Mirrors the overlay pipeline (``cluster_overlay_boxes(categorize_image_findings(...))``)
    so the count matches what the image actually shows: each cluster is one drawn
    region, and a label is counted once per cluster it belongs to. A label the model
    localizes at two separate sites scores 2; two near-coincident boxes that merge
    into one cluster score 1; a present finding with no box scores 0 (it draws
    nothing). ``findings_table_rows`` uses a count above one to append an "(N foci)"
    hint, reconciling the single deduplicated row with the several boxes the overlay
    draws.
    """
    counts: dict[str, int] = {}
    for cluster in cluster_overlay_boxes(categorize_image_findings(outcome)):
        for label in cluster.labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def findings_table_rows(outcome: AuditOutcome) -> list[list[str]]:
    """Build the structured findings table rows.

    Each row is ``[finding, source, status, audit_flag]`` where ``finding`` is the
    human-readable display name, ``source`` is ``"Image"`` or ``"Draft"``,
    ``status`` is ``"Present"`` or ``"Absent"``, and ``audit_flag`` is a plain
    English phrase. Image findings are deduplicated by canonical label (the model
    can localize one label to several boxes; the table shows it once), so a
    non-expert never sees the same finding listed repeatedly; when the overlay
    draws several distinct boxes for one label, that label's single row carries an
    "(N foci)" hint so the one row reconciles with the several boxes on the image.
    The ``no_finding``
    sentinel is never listed beside positive findings - it is mutually exclusive
    with them - so a negative draft ("lungs are clear") adds no contradictory row;
    a genuinely clear study instead shows a single ``no_finding`` row.

    Args:
        outcome: The audit outcome.

    Returns:
        A list of four-column string rows. For an audited outcome the list is
        always non-empty: a clear study yields a single ``"No finding"`` row.
    """
    from cxr_auditor.findings import NO_FINDING

    rows: list[list[str]] = []

    missing = set(outcome.result.audit.missing_findings)
    urgent = set(outcome.result.audit.urgent_review_flags)
    foci_by_label = _foci_counts(outcome)
    seen_image_labels: set[str] = set()
    for finding in outcome.result.image_findings:
        if finding.status is not FindingStatus.PRESENT or not _is_positive(finding.finding):
            continue
        if finding.finding in seen_image_labels:
            continue
        seen_image_labels.add(finding.finding)
        if finding.finding in urgent:
            flag = _FLAG_URGENT
        elif finding.finding in missing:
            flag = _FLAG_MISSING
        else:
            flag = _FLAG_SUPPORTED
        name = display_name(finding.finding)
        foci = foci_by_label.get(finding.finding, 0)
        if foci > 1:
            name = f"{name} ({foci} foci)"
        rows.append([name, _SOURCE_IMAGE, _status_word(finding.status), flag])

    unsupported = set(outcome.result.audit.unsupported_claims)
    for draft_finding in outcome.result.draft_findings:
        # The no_finding sentinel means "nothing to report", never a positive
        # claim, so it is not listed beside real findings (mutually exclusive with
        # positives). The image loop already excludes it; excluding it here too
        # keeps a negative draft from showing a contradictory "No finding" row and
        # makes a negative draft behave like an empty one.
        if draft_finding.finding == NO_FINDING:
            continue
        is_unsupported = draft_finding.finding in unsupported and draft_finding.status is FindingStatus.PRESENT
        flag = _FLAG_UNSUPPORTED if is_unsupported else _FLAG_NONE
        rows.append([display_name(draft_finding.finding), _SOURCE_DRAFT, _status_word(draft_finding.status), flag])

    # A genuinely clear study (no positive image finding, no draft claim or denial)
    # produces no rows above. Show a single honest "No finding" row so the table
    # reflects the clear result rather than the pre-run placeholder; this is the one
    # place the sentinel appears, and only when nothing positive contradicts it.
    if not rows:
        rows.append([display_name(NO_FINDING), _SOURCE_IMAGE, _status_word(FindingStatus.PRESENT), _FLAG_NONE])

    return rows


def _status_word(status: FindingStatus) -> str:
    """Return the human-readable status word for a finding status."""
    return "Present" if status is FindingStatus.PRESENT else "Absent"


def audit_panel_markdown(outcome: AuditOutcome) -> str:
    """Render the audit verdict as a plain-English Markdown panel.

    When the outcome carries a draft-degradation note (the draft could not be
    parsed and the audit proceeded image-only), that note leads the panel so the
    user re-checks the draft manually. Then comes a short "How to read this"
    orientation line, urgent flags first (most important), missing findings,
    and unsupported claims, with per-item detail (draft spans for unsupported
    claims). Every finding is shown by its human-readable display name. When
    nothing is flagged, reports agreement.

    Args:
        outcome: The audit outcome.

    Returns:
        A Markdown string.
    """
    audit = outcome.result.audit
    lines: list[str] = []
    if outcome.draft_parse_note is not None:
        lines.extend(
            [
                f"**Draft not analyzed:** {outcome.draft_parse_note} "
                "This audit reflects the image only - re-check the draft text manually.",
                "",
            ]
        )
    lines.extend(
        [
            "**How to read this:** this panel compares what the AI sees in the image against the draft text. "
            "It is a research aid, not a diagnosis - always confirm with a qualified radiologist.",
            "",
        ]
    )

    if audit.urgent_review_flags:
        lines.append("### URGENT - needs radiologist review")
        for label in audit.urgent_review_flags:
            name = display_name(label)
            lines.append(f"- **{name}** is visible on the image and is a can't-miss finding - have a radiologist review it.")
        lines.append("")

    if audit.missing_findings:
        lines.append("### Missing from the draft")
        for label in audit.missing_findings:
            lines.append(f"- **{display_name(label)}** appears on the image but the draft does not report it.")
        lines.append("")

    if audit.unsupported_claims:
        lines.append("### Unsupported claims in the draft")
        for claim in outcome.comparison.unsupported:
            span = f' (draft text: "{claim.draft_span}")' if claim.draft_span else ""
            name = display_name(claim.finding)
            lines.append(f"- **{name}** is stated in the draft but the image does not support it{span}.")
        lines.append("")

    if not (audit.urgent_review_flags or audit.missing_findings or audit.unsupported_claims):
        lines.append("### No discrepancies found")
        lines.append("The draft and the image findings agree across the finding set the tool checks.")

    return "\n".join(lines).strip()


def result_json(outcome: AuditOutcome) -> str:
    """Serialize the canonical ``AuditResult`` to an indented JSON string.

    Optional fields that the model left empty (for example ``confidence`` and
    ``evidence`` when the checkpoint emits only a label and a box) are omitted via
    ``exclude_none`` so the displayed JSON shows only the fields that carry a value,
    rather than a wall of ``null`` keys. The disclaimer, box format, and the audit
    lists are always present (they are never ``None``), and the cleaned JSON still
    round-trips through ``AuditResult.model_validate_json``.

    Args:
        outcome: The audit outcome.

    Returns:
        The ``AuditResult`` as pretty-printed JSON with empty optionals omitted.
    """
    return outcome.result.model_dump_json(indent=2, exclude_none=True)
