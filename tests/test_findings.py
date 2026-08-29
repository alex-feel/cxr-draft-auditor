"""Tests for the canonical finding set and dataset label-space mappings."""

from __future__ import annotations

import csv
import io

import pytest

from cxr_auditor.findings import (
    CANONICAL_DISPLAY_NAMES,
    CANONICAL_FINDING_SET,
    CANONICAL_FINDINGS,
    CHESTXDET_TO_CANONICAL,
    NIH_TO_CANONICAL,
    NO_FINDING,
    POSITIVE_FINDINGS,
    URGENT_WHITELIST,
    VINDR_TO_CANONICAL,
    Finding,
    display_name,
    is_canonical,
    is_urgent,
    map_chestxdet_label,
    map_label,
    map_nih_label,
    map_vindr_label,
)


def test_canonical_set_has_exactly_six_findings() -> None:
    assert len(CANONICAL_FINDINGS) == 6
    assert len(CANONICAL_FINDING_SET) == 6


def test_canonical_findings_match_enum_values() -> None:
    assert set(CANONICAL_FINDINGS) == {member.value for member in Finding}


def test_no_finding_sentinel_excluded_from_positive_findings() -> None:
    assert NO_FINDING == "no_finding"
    assert NO_FINDING not in POSITIVE_FINDINGS
    assert set(POSITIVE_FINDINGS) == CANONICAL_FINDING_SET - {NO_FINDING}
    assert len(POSITIVE_FINDINGS) == 5


def test_urgent_whitelist_contains_pneumothorax_and_nodule_mass_and_is_canonical() -> None:
    assert Finding.PNEUMOTHORAX.value in URGENT_WHITELIST
    assert Finding.NODULE_MASS.value in URGENT_WHITELIST
    assert URGENT_WHITELIST <= CANONICAL_FINDING_SET


def test_urgent_whitelist_excludes_non_can_not_miss_findings() -> None:
    # Findings that are not acute or possible-malignancy "can't-miss" items stay
    # off the urgent list so urgent stays a meaningful, high-signal flag.
    assert Finding.CARDIOMEGALY.value not in URGENT_WHITELIST
    assert Finding.PLEURAL_EFFUSION.value not in URGENT_WHITELIST
    assert Finding.LUNG_OPACITY_CONSOLIDATION.value not in URGENT_WHITELIST
    assert NO_FINDING not in URGENT_WHITELIST


def test_is_canonical_and_is_urgent() -> None:
    assert is_canonical("pleural_effusion")
    assert not is_canonical("aortic_enlargement")
    assert is_urgent("pneumothorax")
    assert is_urgent("nodule_mass")
    assert not is_urgent("cardiomegaly")


def test_display_names_cover_every_canonical_finding() -> None:
    # Every canonical label must have a display name so a new label can never ship
    # without one (the map and the canonical set are kept in lockstep).
    assert set(CANONICAL_DISPLAY_NAMES) == CANONICAL_FINDING_SET


def test_display_name_returns_human_readable_for_canonical_labels() -> None:
    assert display_name("lung_opacity_consolidation") == "Lung opacity / consolidation"
    assert display_name("nodule_mass") == "Nodule / mass"
    assert display_name("cardiomegaly") == "Cardiomegaly (enlarged heart)"
    assert display_name("no_finding") == "No finding"


def test_display_name_never_returns_raw_snake_case() -> None:
    for label in CANONICAL_FINDINGS:
        assert "_" not in display_name(label)


def test_display_name_falls_back_gracefully_for_unknown_label() -> None:
    # An unexpected label that slips past validation must still render readably
    # (capitalized, spaced) rather than as raw snake_case.
    assert display_name("aortic_enlargement") == "Aortic enlargement"


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("Pleural effusion", Finding.PLEURAL_EFFUSION.value),
        ("Cardiomegaly", Finding.CARDIOMEGALY.value),
        ("Consolidation", Finding.LUNG_OPACITY_CONSOLIDATION.value),
        ("Infiltration", Finding.LUNG_OPACITY_CONSOLIDATION.value),
        ("Lung Opacity", Finding.LUNG_OPACITY_CONSOLIDATION.value),
        ("Nodule/Mass", Finding.NODULE_MASS.value),
        ("Pneumothorax", Finding.PNEUMOTHORAX.value),
        ("No finding", Finding.NO_FINDING.value),
        ("Aortic enlargement", None),
        ("Atelectasis", None),
        ("Calcification", None),
        ("ILD", None),
        ("Other lesion", None),
        ("Pleural thickening", None),
        ("Pulmonary fibrosis", None),
    ],
)
def test_vindr_mapping(native: str, expected: str | None) -> None:
    assert map_vindr_label(native) == expected


def test_vindr_table_covers_all_fourteen_classes_plus_no_finding() -> None:
    assert len(VINDR_TO_CANONICAL) == 15


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("Cardiomegaly", Finding.CARDIOMEGALY.value),
        ("Effusion", Finding.PLEURAL_EFFUSION.value),
        ("Infiltrate", Finding.LUNG_OPACITY_CONSOLIDATION.value),
        ("Mass", Finding.NODULE_MASS.value),
        ("Nodule", Finding.NODULE_MASS.value),
        ("Pneumonia", Finding.LUNG_OPACITY_CONSOLIDATION.value),
        ("Pneumothorax", Finding.PNEUMOTHORAX.value),
        ("Atelectasis", None),
    ],
)
def test_nih_mapping(native: str, expected: str | None) -> None:
    assert map_nih_label(native) == expected


def test_nih_table_covers_eight_box_pathologies() -> None:
    assert len(NIH_TO_CANONICAL) == 8


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("Cardiomegaly", Finding.CARDIOMEGALY.value),
        ("Effusion", Finding.PLEURAL_EFFUSION.value),
        ("Consolidation", Finding.LUNG_OPACITY_CONSOLIDATION.value),
        ("Mass", Finding.NODULE_MASS.value),
        ("Nodule", Finding.NODULE_MASS.value),
        ("Diffuse Nodule", Finding.NODULE_MASS.value),
        ("Pneumothorax", Finding.PNEUMOTHORAX.value),
        ("Fracture", None),
        ("Emphysema", None),
    ],
)
def test_chestxdet_mapping(native: str, expected: str | None) -> None:
    assert map_chestxdet_label(native) == expected


def test_mapping_is_case_and_whitespace_insensitive() -> None:
    assert map_vindr_label("  PLEURAL EFFUSION  ") == Finding.PLEURAL_EFFUSION.value
    assert map_vindr_label("pleural effusion") == Finding.PLEURAL_EFFUSION.value


def test_map_label_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        map_label("Pneumothorax", "not_a_dataset")


def test_map_label_unknown_native_label_returns_none() -> None:
    assert map_vindr_label("Completely Made Up Finding") is None


def test_every_mapped_value_is_canonical_or_none() -> None:
    for table in (VINDR_TO_CANONICAL, NIH_TO_CANONICAL, CHESTXDET_TO_CANONICAL):
        for value in table.values():
            assert value is None or value in CANONICAL_FINDING_SET


def test_vindr_csv_fixture_rows_map_correctly(vindr_boxes_csv_text: str) -> None:
    reader = csv.DictReader(io.StringIO(vindr_boxes_csv_text))
    mapped = [map_vindr_label(row["class_name"]) for row in reader]
    # Aortic enlargement and a second positive row are present; verify both the
    # canonical hits and the ignored (None) class appear.
    assert Finding.PLEURAL_EFFUSION.value in mapped
    assert Finding.CARDIOMEGALY.value in mapped
    assert Finding.PNEUMOTHORAX.value in mapped
    assert Finding.NO_FINDING.value in mapped
    assert None in mapped  # Aortic enlargement maps to None
