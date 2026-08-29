"""
Canonical finding-set definitions and dataset label-space mappings.

This module is the single source of truth for the constrained finding vocabulary
the whole project operates over, the urgent-review whitelist, and the explicit
mapping tables that translate each external dataset's native label space into the
canonical set.

It is intentionally stdlib-only so that every downstream module (schema,
comparator, metrics, prompts) can depend on it without pulling in numpy, pydantic,
torch, or any heavier dependency.

The canonical set has exactly six labels. ``no_finding`` is a mutually informative
sentinel: it means "no canonical positive finding is asserted", and is never
combined with a positive finding in the same finding list.
"""

from __future__ import annotations

from enum import Enum


class Finding(str, Enum):
    """The canonical constrained finding set (exactly six labels).

    Membership is closed: any external label that does not correspond to one of
    these six maps to ``None`` and is ignored by the canonical pipeline.
    """

    PLEURAL_EFFUSION = "pleural_effusion"
    PNEUMOTHORAX = "pneumothorax"
    LUNG_OPACITY_CONSOLIDATION = "lung_opacity_consolidation"
    NODULE_MASS = "nodule_mass"
    CARDIOMEGALY = "cardiomegaly"
    NO_FINDING = "no_finding"


# Ordered tuple of the canonical label strings, in a fixed presentation order.
# Use this when a stable iteration order matters (prompts, metric tables, UI).
CANONICAL_FINDINGS: tuple[str, ...] = tuple(member.value for member in Finding)

# Fast membership set of canonical label strings.
CANONICAL_FINDING_SET: frozenset[str] = frozenset(CANONICAL_FINDINGS)

# The sentinel "negative" label. Mutually exclusive with positive findings.
NO_FINDING: str = Finding.NO_FINDING.value

# Positive findings are every canonical label except the negative sentinel.
POSITIVE_FINDINGS: tuple[str, ...] = tuple(label for label in CANONICAL_FINDINGS if label != NO_FINDING)

# Findings that, when flagged on the image side, must be surfaced for radiologist
# review regardless of the draft. This frozenset is the single source of truth for
# urgency: the comparator (urgent_review_flags), the overlay renderer (prominent
# urgent styling), and the synthetic expected-audit generator all consult it via
# ``is_urgent``. The whitelist is intentionally extensible: add canonical labels
# here to widen the urgent set. Every entry MUST be a canonical positive finding.
#
# Clinical rationale for the membership:
# - pneumothorax: an acute, time-critical "can't-miss" -- a collapsed lung can
#   progress to tension physiology and is an emergency if overlooked.
# - nodule_mass: a pulmonary nodule or mass is a "can't-miss" possible malignancy;
#   missing it in a draft can delay a lung-cancer diagnosis. Flagging it for
#   radiologist review whenever the image shows one is the safety-conscious default.
URGENT_WHITELIST: frozenset[str] = frozenset(
    {
        Finding.PNEUMOTHORAX.value,
        Finding.NODULE_MASS.value,
    }
)


# Human-readable display names for each canonical finding. This is the single
# source of truth for how a label is shown to a non-expert: the overlay, the
# findings table, and the audit panel all read it through ``display_name`` so the
# same wording appears everywhere. The raw canonical label is kept only in the
# machine-facing JSON.
CANONICAL_DISPLAY_NAMES: dict[str, str] = {
    Finding.PLEURAL_EFFUSION.value: "Pleural effusion",
    Finding.PNEUMOTHORAX.value: "Pneumothorax",
    Finding.LUNG_OPACITY_CONSOLIDATION.value: "Lung opacity / consolidation",
    Finding.NODULE_MASS.value: "Nodule / mass",
    Finding.CARDIOMEGALY.value: "Cardiomegaly (enlarged heart)",
    Finding.NO_FINDING.value: "No finding",
}


# Specificity order for resolving cross-finding overlaps during corpus curation:
# when two findings of DIFFERENT labels localize to the SAME region (heavily
# overlapping boxes), the more SPECIFIC finding is kept. A focal mass that also
# reads as an opacity should be reported as the mass, not as generic opacity, so
# the model is trained on one clean label per region instead of redundant double
# labels. Order is most-specific first; a lower index is more specific and wins on
# overlap. ``no_finding`` is omitted (it carries no box and never overlaps).
FINDING_SPECIFICITY_ORDER: tuple[str, ...] = (
    Finding.PNEUMOTHORAX.value,
    Finding.NODULE_MASS.value,
    Finding.PLEURAL_EFFUSION.value,
    Finding.CARDIOMEGALY.value,
    Finding.LUNG_OPACITY_CONSOLIDATION.value,
)

_SPECIFICITY_RANK: dict[str, int] = {label: index for index, label in enumerate(FINDING_SPECIFICITY_ORDER)}


def specificity_rank(label: str) -> int:
    """Return a finding's specificity rank (lower is more specific, kept on overlap).

    Used by curation to resolve same-region cross-finding overlaps. A label not in
    ``FINDING_SPECIFICITY_ORDER`` ranks last (treated as least specific).
    """
    return _SPECIFICITY_RANK.get(label, len(FINDING_SPECIFICITY_ORDER))


# Mapping from VinDr-CXR's native 14 abnormality classes plus "No finding" to the
# canonical set. Keys are the dataset's native class names exactly as they appear
# in VinDr label spaces. A value of ``None`` means the native class has no
# canonical counterpart and is ignored. Lookups are case-insensitive via
# ``map_label`` / ``map_vindr_label``; keys here are stored lowercased.
VINDR_TO_CANONICAL: dict[str, str | None] = {
    "aortic enlargement": None,
    "atelectasis": None,
    "calcification": None,
    "cardiomegaly": Finding.CARDIOMEGALY.value,
    "consolidation": Finding.LUNG_OPACITY_CONSOLIDATION.value,
    "ild": None,
    "infiltration": Finding.LUNG_OPACITY_CONSOLIDATION.value,
    "lung opacity": Finding.LUNG_OPACITY_CONSOLIDATION.value,
    "nodule/mass": Finding.NODULE_MASS.value,
    "other lesion": None,
    "pleural effusion": Finding.PLEURAL_EFFUSION.value,
    "pleural thickening": None,
    "pneumothorax": Finding.PNEUMOTHORAX.value,
    "pulmonary fibrosis": None,
    "no finding": Finding.NO_FINDING.value,
}


# Mapping from the eight NIH ChestX-ray14 BBox_List_2017 pathologies to the
# canonical set. NIH box annotations cover only these eight classes. Native names
# are stored lowercased.
NIH_TO_CANONICAL: dict[str, str | None] = {
    "atelectasis": None,
    "cardiomegaly": Finding.CARDIOMEGALY.value,
    "effusion": Finding.PLEURAL_EFFUSION.value,
    "infiltrate": Finding.LUNG_OPACITY_CONSOLIDATION.value,
    "mass": Finding.NODULE_MASS.value,
    "nodule": Finding.NODULE_MASS.value,
    "pneumonia": Finding.LUNG_OPACITY_CONSOLIDATION.value,
    "pneumothorax": Finding.PNEUMOTHORAX.value,
}


# Mapping from ChestX-Det categories to the canonical set. ChestX-Det annotates a
# broader category list; classes without a canonical counterpart map to ``None``.
# Native names are stored lowercased.
CHESTXDET_TO_CANONICAL: dict[str, str | None] = {
    "atelectasis": None,
    "calcification": None,
    "cardiomegaly": Finding.CARDIOMEGALY.value,
    "consolidation": Finding.LUNG_OPACITY_CONSOLIDATION.value,
    "diffuse nodule": Finding.NODULE_MASS.value,
    "effusion": Finding.PLEURAL_EFFUSION.value,
    "emphysema": None,
    "fibrosis": None,
    "fracture": None,
    "mass": Finding.NODULE_MASS.value,
    "nodule": Finding.NODULE_MASS.value,
    "pleural thickening": None,
    "pneumothorax": Finding.PNEUMOTHORAX.value,
}


# Registry of every dataset mapping table by a short dataset key. Used by
# ``map_label`` to dispatch to the correct table.
DATASET_MAPPINGS: dict[str, dict[str, str | None]] = {
    "vindr": VINDR_TO_CANONICAL,
    "nih": NIH_TO_CANONICAL,
    "chestxdet": CHESTXDET_TO_CANONICAL,
}


def map_label(native_label: str, dataset: str) -> str | None:
    """Map a single native dataset label to the canonical finding string.

    The lookup is case-insensitive and tolerant of surrounding whitespace.

    Args:
        native_label: The dataset's native class name (any casing).
        dataset: Dataset key, one of the keys of ``DATASET_MAPPINGS``
            (``'vindr'``, ``'nih'``, ``'chestxdet'``).

    Returns:
        The canonical finding string, or ``None`` when the native label has no
        canonical counterpart (and should be ignored) or is unknown to the table.

    Raises:
        KeyError: If ``dataset`` is not a registered dataset key.
    """
    table = DATASET_MAPPINGS[dataset]
    return table.get(native_label.strip().lower())


def map_vindr_label(native_label: str) -> str | None:
    """Map a single VinDr-CXR native label to the canonical finding string.

    Convenience wrapper over ``map_label(native_label, 'vindr')``.
    """
    return map_label(native_label, "vindr")


def map_nih_label(native_label: str) -> str | None:
    """Map a single NIH ChestX-ray14 box label to the canonical finding string.

    Convenience wrapper over ``map_label(native_label, 'nih')``.
    """
    return map_label(native_label, "nih")


def map_chestxdet_label(native_label: str) -> str | None:
    """Map a single ChestX-Det category to the canonical finding string.

    Convenience wrapper over ``map_label(native_label, 'chestxdet')``.
    """
    return map_label(native_label, "chestxdet")


def is_canonical(label: str) -> bool:
    """Return whether ``label`` is one of the six canonical finding strings."""
    return label in CANONICAL_FINDING_SET


def is_urgent(label: str) -> bool:
    """Return whether ``label`` is on the urgent-review whitelist."""
    return label in URGENT_WHITELIST


def display_name(label: str) -> str:
    """Return the human-readable display name for a canonical finding label.

    Looks the label up in ``CANONICAL_DISPLAY_NAMES``. For any label not in the
    map (an unexpected value that slipped past validation), it falls back to a
    capitalized, space-separated rendering of the raw label so a non-expert never
    sees raw ``snake_case`` text on screen.

    Args:
        label: A canonical finding string (or any string, for the fallback).

    Returns:
        The display name to show the user.
    """
    return CANONICAL_DISPLAY_NAMES.get(label, label.replace("_", " ").capitalize())
