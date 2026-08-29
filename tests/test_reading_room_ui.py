"""
Tests for the pure presentation builders in ``reading_room_ui``.

The module's top level imports only the standard library (gradio is touched
lazily inside ``build_theme`` alone), so every HTML builder and the CSS string
are exercised here in the pure-logic dev environment without gradio installed.
The status-pill mapping is pinned against the exact audit-flag phrases that
``cxr_auditor.render.findings_table_rows`` emits, so a wording change in the
renderer fails loudly here instead of silently downgrading pills to dashes.
"""

from __future__ import annotations

import importlib.util

import pytest

import reading_room_ui as ui
from cxr_auditor.comparator import compare
from cxr_auditor.inference import DRAFT_PARSE_FAILURE_NOTE, AuditOutcome
from cxr_auditor.render import findings_table_rows
from cxr_auditor.schema import AuditResult, DraftFinding, FindingStatus, ImageFinding

EM_DASH = chr(0x2014)
LEFT_DOUBLE_QUOTE = chr(0x201C)
RIGHT_DOUBLE_QUOTE = chr(0x201D)


def _outcome(
    image_findings: list[ImageFinding],
    draft_findings: list[DraftFinding],
    draft_parse_note: str | None = None,
) -> AuditOutcome:
    comparison = compare(image_findings, draft_findings)
    result = AuditResult(image_findings=image_findings, draft_findings=draft_findings, audit=comparison.audit)
    return AuditOutcome(result=result, comparison=comparison, draft_parse_note=draft_parse_note)


def _all_flags_outcome() -> AuditOutcome:
    """An outcome whose table rows carry every audit-flag phrase plus an empty flag."""
    return _outcome(
        [
            ImageFinding(finding="pneumothorax", box=(0.1, 0.5, 0.4, 0.9)),
            ImageFinding(finding="pleural_effusion", box=(0.6, 0.1, 0.9, 0.4)),
            ImageFinding(finding="cardiomegaly", box=(0.3, 0.3, 0.7, 0.7)),
        ],
        [
            DraftFinding(finding="cardiomegaly"),
            DraftFinding(finding="lung_opacity_consolidation", span="patchy opacity"),
            DraftFinding(finding="nodule_mass", status=FindingStatus.ABSENT),
        ],
    )


class TestFlagToStatusMapping:
    def test_mapping_keys_are_the_exact_render_phrases(self) -> None:
        assert ui._FLAG_TO_STATUS == {
            "Supported by image": "supported",
            "Missing from draft": "missing",
            "Urgent - review": "urgent",
            "Unsupported claim": "unsupported",
        }

    def test_every_flag_findings_table_rows_emits_is_mapped(self) -> None:
        rows = findings_table_rows(_all_flags_outcome())
        emitted = {row[3] for row in rows}
        assert emitted == {
            "Urgent - review",
            "Missing from draft",
            "Supported by image",
            "Unsupported claim",
            "",
        }
        for flag in emitted - {""}:
            assert flag in ui._FLAG_TO_STATUS

    def test_each_mapped_flag_renders_a_tinted_pill(self) -> None:
        for flag, status in ui._FLAG_TO_STATUS.items():
            color, bg = ui._STATUS[status]
            pill = ui._pill(flag)
            assert 'class="rr-pill"' in pill
            assert color in pill
            assert bg in pill
            assert flag in pill

    def test_unknown_or_empty_flag_renders_a_dash_placeholder(self) -> None:
        for flag in ("", "Something new"):
            pill = ui._pill(flag)
            assert "rr-pill" not in pill
            assert EM_DASH in pill


class TestTableHtml:
    def test_empty_rows_render_the_placeholder(self) -> None:
        html_out = ui.table_html([])
        assert "rr-empty" in html_out
        assert "Run an audit to see the structured findings." in html_out
        assert "<table" not in html_out

    def test_rows_render_a_table_with_headers_and_pills(self) -> None:
        rows = findings_table_rows(_all_flags_outcome())
        html_out = ui.table_html(rows)
        for header in ("Finding", "Source", "Status", "Audit flag"):
            assert f"<th>{header}</th>" in html_out
        assert html_out.count("<tr>") == len(rows) + 1  # header row + one per finding
        assert html_out.count('class="rr-pill"') == 4  # the absent draft row gets a dash
        for phrase in ("Urgent - review", "Missing from draft", "Supported by image", "Unsupported claim"):
            assert phrase in html_out
        assert "Pneumothorax" in html_out
        assert "Cardiomegaly (enlarged heart)" in html_out

    def test_cell_text_is_escaped(self) -> None:
        html_out = ui.table_html([["<b>finding</b>", "Image", "Present", "Supported by image"]])
        assert "<b>finding</b>" not in html_out
        assert "&lt;b&gt;finding&lt;/b&gt;" in html_out


class TestVerdictHtml:
    def test_all_empty_reports_agreement(self) -> None:
        html_out = ui.verdict_html(urgent=[], missing=[], unsupported=[], draft_parse_note=None)
        assert "No discrepancies found" in html_out
        assert "rr-chips" not in html_out
        green, green_bg = ui._STATUS["supported"]
        assert green in html_out
        assert green_bg in html_out

    def test_flag_lists_render_chips_and_tinted_cards(self) -> None:
        html_out = ui.verdict_html(
            urgent=["Pneumothorax"],
            missing=["Pleural effusion", "Cardiomegaly (enlarged heart)"],
            unsupported=[("Nodule / mass", "a carcinoid tumor")],
            draft_parse_note=None,
        )
        assert "1 URGENT" in html_out
        assert "2 MISSING" in html_out
        assert "1 UNSUPPORTED" in html_out
        assert html_out.count('class="rr-item"') == 4
        assert "Pneumothorax" in html_out
        assert "have a radiologist review it." in html_out
        assert "Appears on the image but the draft does not report it." in html_out
        assert f"({LEFT_DOUBLE_QUOTE}a carcinoid tumor{RIGHT_DOUBLE_QUOTE})" in html_out
        assert "No discrepancies found" not in html_out

    def test_unsupported_without_span_uses_the_generic_sentence(self) -> None:
        html_out = ui.verdict_html(
            urgent=[],
            missing=[],
            unsupported=[("Nodule / mass", None)],
            draft_parse_note=None,
        )
        assert "Stated in the draft but the image does not support it." in html_out
        assert LEFT_DOUBLE_QUOTE not in html_out

    def test_draft_parse_note_leads_the_panel(self) -> None:
        html_out = ui.verdict_html(
            urgent=["Pneumothorax"],
            missing=[],
            unsupported=[],
            draft_parse_note=DRAFT_PARSE_FAILURE_NOTE,
        )
        note_position = html_out.index("Draft not analyzed")
        chips_position = html_out.index("rr-chips")
        assert note_position < chips_position
        assert DRAFT_PARSE_FAILURE_NOTE in html_out
        assert "Re-check the draft text manually." in html_out

    def test_names_and_spans_are_escaped(self) -> None:
        html_out = ui.verdict_html(
            urgent=["<script>alert(1)</script>"],
            missing=[],
            unsupported=[("Nodule / mass", '<img src="x">')],
            draft_parse_note=None,
        )
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out
        assert '<img src="x">' not in html_out


class TestVerdictPlaceholder:
    def test_placeholder_prompts_for_a_run(self) -> None:
        html_out = ui.verdict_placeholder()
        assert "rr-empty" in html_out
        assert "Run audit" in html_out


class TestHeaderRibbonFooter:
    def test_header_carries_wordmark_chips_and_preview_badge(self) -> None:
        html_out = ui.header_html(
            "my-model & co",
            "Nemotron & co",
            "https://huggingface.co/acme/my-model",
            "https://huggingface.co/nvidia/nemotron",
            "https://huggingface.co/blog/acme/my-post",
        )
        assert 'id="rr-header"' in html_out
        assert "CXR Draft Auditor" in html_out
        assert "second-look QA for draft impressions" in html_out
        assert "RESEARCH PREVIEW" in html_out
        # Both model chips appear, with names HTML-escaped.
        assert "my-model &amp; co" in html_out
        assert "Nemotron &amp; co" in html_out
        assert html_out.count('class="chip model"') == 2
        # Each chip is a whole-chip link to its model page, opened in a new tab.
        assert 'href="https://huggingface.co/acme/my-model"' in html_out
        assert 'href="https://huggingface.co/nvidia/nemotron"' in html_out
        assert html_out.count('target="_blank"') >= 3
        # The BLOG POST chip links to the published write-up.
        assert 'class="chip blog"' in html_out
        assert 'href="https://huggingface.co/blog/acme/my-post"' in html_out

    def test_ribbon_carries_the_disclaimer(self) -> None:
        html_out = ui.ribbon_html()
        assert 'id="rr-ribbon"' in html_out
        assert "Not a medical device, not a diagnosis, not for clinical use." in html_out
        assert "qualified radiologist" in html_out

    def test_footer_carries_both_model_ids_and_serving_details(self) -> None:
        html_out = ui.footer_html("alex-feeel/medgemma-cxr-auditor-v2", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        assert 'id="rr-footer"' in html_out
        assert "alex-feeel/medgemma-cxr-auditor-v2" in html_out
        assert "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" in html_out
        assert "ZeroGPU" in html_out
        assert "Build Small Hackathon" in html_out
        # Both model ids and the hackathon name are links opening in a new tab.
        assert 'href="https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2"' in html_out
        assert 'href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"' in html_out
        assert 'href="https://huggingface.co/build-small-hackathon"' in html_out


class TestCardHeadAndLegend:
    def test_step_badge_and_title(self) -> None:
        html_out = ui.card_head("1", "Chest X-ray")
        assert '<span class="step">1</span>' in html_out
        assert "Chest X-ray" in html_out
        assert "rr-legend" not in html_out

    def test_no_step_with_meta(self) -> None:
        html_out = ui.card_head(None, "Raw JSON", meta="AuditResult")
        assert 'class="step"' not in html_out
        assert '<span class="meta">AuditResult</span>' in html_out

    def test_legend_variant_embeds_the_color_legend(self) -> None:
        html_out = ui.card_head(None, "Image-grounded evidence", legend=True)
        assert "rr-legend" in html_out
        for entry in ("supported", "missing from draft", "urgent"):
            assert entry in html_out

    def test_legend_omits_unsupported_which_is_never_an_image_box(self) -> None:
        # An unsupported claim is asserted in the draft but absent from the image,
        # so the overlay never draws an unsupported box (see
        # cxr_auditor.render.categorize_image_findings, which categorizes only
        # image findings as supported/missing/urgent). The image legend must not
        # advertise it as a box; it surfaces only in the verdict and findings table.
        html_out = ui.legend_html()
        assert "unsupported" not in html_out
        for entry in ("supported", "missing from draft", "urgent"):
            assert entry in html_out

    def test_legend_uses_the_status_color_vars(self) -> None:
        # The legend reads the status CSS variables (not baked hex) so it recolors
        # with the active light/dark mode, like the pills and verdict cards.
        html_out = ui.legend_html()
        for var in ("var(--rr-green)", "var(--rr-amber)", "var(--rr-red)"):
            assert var in html_out
        assert "double" in html_out  # the urgent chip mirrors the doubled overlay border

    def test_title_is_escaped(self) -> None:
        html_out = ui.card_head(None, "<i>title</i>")
        assert "<i>title</i>" not in html_out
        assert "&lt;i&gt;title&lt;/i&gt;" in html_out


class TestMessageHtml:
    def test_bold_and_code_spans_become_html(self) -> None:
        html_out = ui.message_html("**Model not configured.** Set the `HF_MODEL_ID` Space variable.")
        assert "<strong>Model not configured.</strong>" in html_out
        assert "<code>HF_MODEL_ID</code>" in html_out
        assert "**" not in html_out
        assert "`" not in html_out

    def test_message_text_is_escaped_before_conversion(self) -> None:
        html_out = ui.message_html("**Bad <input>** seen")
        assert "<input>" not in html_out
        assert "<strong>Bad &lt;input&gt;</strong>" in html_out

    def test_status_drives_the_tint(self) -> None:
        amber, amber_bg = ui._STATUS["missing"]
        assert amber in ui.message_html("notice", "missing")
        assert amber_bg in ui.message_html("notice", "missing")
        red, red_bg = ui._STATUS["unsupported"]
        assert red in ui.message_html("failure", "unsupported")
        assert red_bg in ui.message_html("failure", "unsupported")

    def test_unknown_status_raises(self) -> None:
        with pytest.raises(KeyError):
            ui.message_html("notice", "celebratory")


class TestThemeToggle:
    def test_header_includes_the_theme_toggle_button(self) -> None:
        html_out = ui.header_html(
            "medgemma-cxr-auditor-v2",
            "Nemotron-3 Nano 4B",
            "https://huggingface.co/alex-feeel/medgemma-cxr-auditor-v2",
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "https://huggingface.co/blog/build-small-hackathon/chest-x-ray-draft-auditor",
        )
        assert 'id="rr-theme-toggle"' in html_out
        assert 'class="rr-toggle"' in html_out

    def test_head_script_defaults_dark_and_wires_the_toggle(self) -> None:
        # The injected head script forces a first-visit default, wires the toggle,
        # and persists the choice; it switches via the native ?__theme= parameter.
        assert "rr-theme-toggle" in ui.HEAD
        assert "__theme" in ui.HEAD
        assert "cxr-theme" in ui.HEAD  # the localStorage persistence key
        assert "'dark'" in ui.HEAD


class TestCssAndTheme:
    def test_css_defines_both_palettes_under_a_dark_selector(self) -> None:
        # Light values live in :root; dark values override under body.dark, so the
        # single served stylesheet adapts to the class Gradio adds in dark mode.
        assert "body.dark" in ui.CSS
        for palette in (ui.LIGHT, ui.DARK):
            assert palette["page"] in ui.CSS
            assert palette["accent"] in ui.CSS
        for var in ("--rr-page", "--rr-panel", "--rr-green", "--rr-amber", "--rr-red", "--rr-accent"):
            assert var in ui.CSS
        for selector in ("#rr-header", "#rr-ribbon", "#rr-footer", ".rr-panel", ".rr-pill", "table.rr-table"):
            assert selector in ui.CSS

    def test_build_theme_needs_gradio_only_when_called(self) -> None:
        if importlib.util.find_spec("gradio") is None:
            with pytest.raises(ModuleNotFoundError):
                ui.build_theme()
        else:  # pragma: no cover - exercised only where gradio is installed
            assert ui.build_theme() is not None
