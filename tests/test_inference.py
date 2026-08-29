"""
Tests for the model-serving boundary in ``cxr_auditor.inference``.

These tests exercise only the pure-logic surface of the module: the
grounded-dict-to-``ImageFinding`` assembly helper, the escalating retry ladder,
the serving-error categorizer, and the ``run_audit`` / ``audit`` orchestration
driven with a fake model. The deterministic comparator and the draft parser are
owned (and tested) by ``cxr_auditor.comparator`` and ``cxr_auditor.parser``; this
module delegates to them, so the tests here verify the wiring, not the
comparator/parser internals. No torch, transformers, or GPU is imported or
required.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from cxr_auditor.inference import (
    DRAFT_PARSE_FAILURE_NOTE,
    GREEDY_SETTINGS,
    RETRY_CORRECTIVE_SUFFIX,
    RETRY_SAMPLING_SETTINGS,
    AuditOutcome,
    GenerateFnFactory,
    GenerationSettings,
    _strip_reasoning,
    assemble_audit,
    audit,
    categorize_serving_error,
    generate_findings,
    grounded_dicts_to_image_findings,
    make_generate_fn,
    run_audit,
    run_with_retry,
)
from cxr_auditor.parser import GenerateFn
from cxr_auditor.schema import AuditResult, FindingStatus, ImageFinding, SchemaParseError, extract_finding_list


def _text_factory(reply: str) -> GenerateFnFactory:
    """A draft ``GenerateFn`` factory whose every attempt returns a canned reply."""

    def _factory(_settings: GenerationSettings) -> GenerateFn:
        def _generate(_prompt: str) -> str:
            return reply

        return _generate

    return _factory


class TestStripReasoning:
    def test_removes_a_closed_think_block(self) -> None:
        assert _strip_reasoning("<think>weighing options</think>\n[1, 2]") == "[1, 2]"

    def test_passes_text_without_a_block_through(self) -> None:
        assert _strip_reasoning("[1, 2]") == "[1, 2]"

    def test_is_case_insensitive_and_dotall(self) -> None:
        assert _strip_reasoning("<THINK>line one\nline two [stray</THINK>[2]") == "[2]"


class TestAssembleAudit:
    def test_parses_the_draft_through_the_injected_factory(self) -> None:
        image_findings = [ImageFinding(finding="cardiomegaly", box=(0.3, 0.3, 0.7, 0.7))]
        reply = '[{"label": "cardiomegaly", "status": "present", "span": "enlarged heart"}]'
        outcome = assemble_audit(image_findings, "Cardiomegaly is present.", draft_factory=_text_factory(reply))
        assert outcome.draft_parse_note is None
        assert [d.finding for d in outcome.result.draft_findings] == ["cardiomegaly"]
        # Present in both the image and the draft, so cardiomegaly is not missing.
        assert "cardiomegaly" not in outcome.result.audit.missing_findings

    def test_no_draft_skips_parsing(self) -> None:
        image_findings = [ImageFinding(finding="cardiomegaly", box=(0.3, 0.3, 0.7, 0.7))]
        outcome = assemble_audit(image_findings, None, draft_factory=_text_factory("[]"))
        assert outcome.result.draft_findings == []
        assert outcome.draft_parse_note is None

    def test_unparseable_draft_degrades_to_image_only(self) -> None:
        image_findings = [ImageFinding(finding="cardiomegaly", box=(0.3, 0.3, 0.7, 0.7))]
        outcome = assemble_audit(image_findings, "Some draft text.", draft_factory=_text_factory("no json here at all"))
        assert outcome.result.draft_findings == []
        assert outcome.draft_parse_note == DRAFT_PARSE_FAILURE_NOTE
        assert any(f.finding == "cardiomegaly" for f in outcome.result.image_findings)


class TestGroundedDictsToImageFindings:
    def test_basic_conversion(self, grounded_finding_dicts: list[dict[str, Any]]) -> None:
        findings = grounded_dicts_to_image_findings(grounded_finding_dicts)
        assert [f.finding for f in findings] == ["pleural_effusion", "pneumothorax"]
        assert findings[0].box == (0.62, 0.08, 0.94, 0.40)
        assert findings[0].status is FindingStatus.PRESENT
        assert findings[0].confidence == 0.78

    def test_skip_non_canonical_labels(self) -> None:
        findings = grounded_dicts_to_image_findings(
            [
                {"label": "aortic_enlargement", "box_2d": [0.1, 0.1, 0.2, 0.2]},
                {"label": "cardiomegaly", "box_2d": None},
            ]
        )
        assert [f.finding for f in findings] == ["cardiomegaly"]

    def test_normalizes_label_drift(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": "Pleural Effusion", "box_2d": None}])
        assert findings[0].finding == "pleural_effusion"

    def test_no_finding_grounding_yields_single_no_finding(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": "no_finding", "box_2d": None, "confidence": 0.9}])
        assert len(findings) == 1
        assert findings[0].finding == "no_finding"
        assert findings[0].box is None

    def test_malformed_box_is_dropped_but_finding_kept(self) -> None:
        findings = grounded_dicts_to_image_findings(
            [
                {"label": "cardiomegaly", "box_2d": [0.1, 0.2, 0.3]},
                {"label": "pleural_effusion", "box_2d": [2.0, 0.0, 3.0, 1.0]},
                {"label": "nodule_mass", "box_2d": [0.9, 0.1, 0.2, 0.4]},
            ]
        )
        assert [f.finding for f in findings] == ["cardiomegaly", "pleural_effusion", "nodule_mass"]
        assert all(f.box is None for f in findings)

    def test_out_of_range_confidence_dropped(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": "cardiomegaly", "box_2d": None, "confidence": 9.0}])
        assert findings[0].confidence is None

    def test_non_string_label_skipped(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": 5, "box_2d": None}, {"label": "cardiomegaly"}])
        assert [f.finding for f in findings] == ["cardiomegaly"]

    def test_non_numeric_box_element_dropped_but_finding_kept(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": "cardiomegaly", "box_2d": ["a", 0.1, 0.2, 0.3]}])
        assert [f.finding for f in findings] == ["cardiomegaly"]
        assert findings[0].box is None

    def test_inverted_box_dropped(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": "cardiomegaly", "box_2d": [0.9, 0.9, 0.1, 0.1]}])
        assert findings[0].box is None

    def test_non_numeric_confidence_dropped(self) -> None:
        findings = grounded_dicts_to_image_findings([{"label": "cardiomegaly", "box_2d": None, "confidence": "high"}])
        assert findings[0].confidence is None

    def test_exact_label_and_box_duplicate_collapsed_first_wins(self) -> None:
        # The model sometimes emits the identical (label, box) finding twice. The
        # redundant repeat is collapsed so it does not produce a duplicate row or a
        # duplicate overlay box; the first occurrence (with its confidence) wins.
        findings = grounded_dicts_to_image_findings(
            [
                {"label": "nodule_mass", "box_2d": [0.236, 0.316, 0.386, 0.422], "confidence": 0.8},
                {"label": "nodule_mass", "box_2d": [0.236, 0.316, 0.386, 0.422], "confidence": 0.5},
            ]
        )
        assert len(findings) == 1
        assert findings[0].finding == "nodule_mass"
        assert findings[0].box == (0.236, 0.316, 0.386, 0.422)
        assert findings[0].confidence == 0.8

    def test_same_label_distinct_boxes_both_kept(self) -> None:
        # The same label localized to two genuinely different regions (for example a
        # bilateral opacity) is two real findings; neither is dropped.
        findings = grounded_dicts_to_image_findings(
            [
                {"label": "lung_opacity_consolidation", "box_2d": [0.236, 0.316, 0.386, 0.422]},
                {"label": "lung_opacity_consolidation", "box_2d": [0.236, 0.531, 0.386, 0.622]},
            ]
        )
        assert len(findings) == 2
        assert findings[0].box == (0.236, 0.316, 0.386, 0.422)
        assert findings[1].box == (0.236, 0.531, 0.386, 0.622)

    def test_distinct_labels_at_same_box_both_kept(self) -> None:
        # A single physical region the model labels with two different findings (for
        # example opacity AND nodule_mass at the same coordinates) is two distinct
        # findings; the pipeline keeps both labels (a region can be both).
        box = [0.236, 0.316, 0.386, 0.422]
        findings = grounded_dicts_to_image_findings(
            [
                {"label": "lung_opacity_consolidation", "box_2d": box},
                {"label": "nodule_mass", "box_2d": box},
            ]
        )
        assert [f.finding for f in findings] == ["lung_opacity_consolidation", "nodule_mass"]

    def test_unlocalized_label_duplicate_collapsed(self) -> None:
        # Two box-less repeats of the same label collapse to one (identical key).
        findings = grounded_dicts_to_image_findings(
            [
                {"label": "cardiomegaly", "box_2d": None},
                {"label": "cardiomegaly", "box_2d": None},
            ]
        )
        assert len(findings) == 1


class _FakeProcessor:
    """A minimal stand-in for a transformers processor.

    Records each prompt the orchestration generates against (so a test can assert
    how many model turns were issued and what each prompt carried) and the
    decoding settings each turn used (so a test can assert the retry ladder's
    escalation).
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.settings: list[GenerationSettings] = []


class _FakeModel:
    """A minimal stand-in for a vision-language model."""


class TestRunAudit:
    """Drive ``run_audit`` / ``audit`` with a fake model via the ``_generate_text`` seam."""

    def _patch_generation(self, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> None:
        queued = list(replies)

        def _fake_generate_text(model: Any, processor: Any, prompt: str, image: Any, settings: GenerationSettings) -> str:
            processor.prompts.append(prompt)
            processor.settings.append(settings)
            return queued.pop(0)

        monkeypatch.setattr("cxr_auditor.inference._generate_text", _fake_generate_text)

    def test_image_only(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        grounding_reply = '[{"label": "pneumothorax", "box_2d": [0.1, 0.5, 0.4, 0.9], "confidence": 0.7}]'
        self._patch_generation(monkeypatch, [grounding_reply])
        outcome = run_audit(tiny_image, draft_text=None, model=_FakeModel(), processor=_FakeProcessor())
        assert isinstance(outcome, AuditOutcome)
        assert [f.finding for f in outcome.result.image_findings] == ["pneumothorax"]
        assert outcome.result.draft_findings == []
        assert outcome.result.audit.urgent_review_flags == ["pneumothorax"]
        assert outcome.result.audit.missing_findings == ["pneumothorax"]
        # The comparison detail carries the box for the overlay.
        assert outcome.comparison.urgent[0].box == (0.1, 0.5, 0.4, 0.9)

    def test_with_draft(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        grounding_reply = '[{"label": "pleural_effusion", "box_2d": [0.6, 0.1, 0.9, 0.4]}]'
        draft_reply = '[{"label": "cardiomegaly", "status": "present", "span": "enlarged heart"}]'
        self._patch_generation(monkeypatch, [grounding_reply, draft_reply])
        outcome = run_audit(
            tiny_image,
            draft_text="Enlarged cardiac silhouette.",
            model=_FakeModel(),
            processor=_FakeProcessor(),
        )
        assert [f.finding for f in outcome.result.image_findings] == ["pleural_effusion"]
        assert [d.finding for d in outcome.result.draft_findings] == ["cardiomegaly"]
        assert outcome.result.audit.missing_findings == ["pleural_effusion"]
        assert outcome.result.audit.unsupported_claims == ["cardiomegaly"]
        assert outcome.comparison.unsupported[0].draft_span == "enlarged heart"
        assert outcome.draft_parse_note is None

    def test_blank_draft_treated_as_no_draft(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        grounding_reply = '[{"label": "no_finding", "box_2d": null}]'
        self._patch_generation(monkeypatch, [grounding_reply])
        processor = _FakeProcessor()
        outcome = run_audit(tiny_image, draft_text="   ", model=_FakeModel(), processor=processor)
        assert outcome.result.draft_findings == []
        # A single grounding call only (no draft parse) means one generation call.
        assert len(processor.prompts) == 1

    def test_audit_wrapper_returns_audit_result(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        grounding_reply = '[{"label": "cardiomegaly", "box_2d": null}]'
        self._patch_generation(monkeypatch, [grounding_reply])
        result = audit(tiny_image, model=_FakeModel(), processor=_FakeProcessor())
        assert isinstance(result, AuditResult)
        assert [f.finding for f in result.image_findings] == ["cardiomegaly"]

    def test_requires_model_and_processor(self, tiny_image: Any) -> None:
        with pytest.raises(ValueError, match="model and processor"):
            run_audit(tiny_image, model=None, processor=None)


def _recording_factory(
    replies: Iterator[str],
    log: list[tuple[GenerationSettings, str]],
) -> GenerateFnFactory:
    """Build a fake factory that records each attempt's settings and prompt."""

    def _factory(settings: GenerationSettings) -> GenerateFn:
        def _generate(prompt: str) -> str:
            log.append((settings, prompt))
            return next(replies)

        return _generate

    return _factory


class TestRunWithRetry:
    """The retry ladder escalates conditions on every failed attempt."""

    def test_succeeds_on_first_attempt_greedy_base_prompt(self) -> None:
        log: list[tuple[GenerationSettings, str]] = []
        factory = _recording_factory(iter(['[{"label": "cardiomegaly"}]']), log)

        parsed = run_with_retry(factory, "PROMPT", extract_finding_list, max_retries=2)
        assert parsed == [{"label": "cardiomegaly"}]
        assert log == [(GREEDY_SETTINGS, "PROMPT")]

    def test_attempt_two_appends_corrective_suffix_still_greedy(self) -> None:
        log: list[tuple[GenerationSettings, str]] = []
        factory = _recording_factory(iter(["not json at all", '[{"label": "pneumothorax"}]']), log)

        parsed = run_with_retry(factory, "PROMPT", extract_finding_list, max_retries=2)
        assert parsed == [{"label": "pneumothorax"}]
        assert log[0] == (GREEDY_SETTINGS, "PROMPT")
        assert log[1] == (GREEDY_SETTINGS, "PROMPT" + RETRY_CORRECTIVE_SUFFIX)

    def test_attempt_three_switches_to_sampling_with_suffix(self) -> None:
        log: list[tuple[GenerationSettings, str]] = []
        factory = _recording_factory(iter(["bad", "still bad", '[{"label": "cardiomegaly"}]']), log)

        parsed = run_with_retry(factory, "PROMPT", extract_finding_list, max_retries=2)
        assert parsed == [{"label": "cardiomegaly"}]
        assert log[2] == (RETRY_SAMPLING_SETTINGS, "PROMPT" + RETRY_CORRECTIVE_SUFFIX)
        assert RETRY_SAMPLING_SETTINGS.do_sample is True

    def test_raises_last_error_after_exhausting_retries(self) -> None:
        log: list[tuple[GenerationSettings, str]] = []
        factory = _recording_factory(iter(["still not json", "still not json"]), log)

        with pytest.raises(SchemaParseError) as excinfo:
            run_with_retry(factory, "PROMPT", extract_finding_list, max_retries=1)
        assert excinfo.value.raw_text == "still not json"
        assert len(log) == 2

    def test_negative_max_retries_rejected(self) -> None:
        log: list[tuple[GenerationSettings, str]] = []
        factory = _recording_factory(iter(["[]"]), log)

        with pytest.raises(ValueError, match="max_retries"):
            run_with_retry(factory, "PROMPT", extract_finding_list, max_retries=-1)

    def test_parse_failure_logs_traceback_and_truncated_raw_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Every failed attempt prints its traceback and a bounded raw-text block to
        # stdout, which is the channel the serving platform's run logs capture.
        long_garbage = "x" * 3000
        log: list[tuple[GenerationSettings, str]] = []
        factory = _recording_factory(iter([long_garbage, '[{"label": "cardiomegaly"}]']), log)

        run_with_retry(factory, "PROMPT", extract_finding_list, max_retries=1, stage="image_grounding")
        captured = capsys.readouterr().out
        assert "parse failure: stage=image_grounding attempt=1" in captured
        assert "SchemaParseError" in captured
        assert "...[truncated]" in captured
        assert "x" * 100 in captured


class TestGenerateFindingsSeam:
    """``generate_findings`` grounds an image through the retry loop."""

    def test_generate_findings_via_patched_seam(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        reply = '[{"label": "pleural_effusion", "box_2d": [0.6, 0.1, 0.9, 0.4], "confidence": 0.8}]'

        def _fake_generate_text(_model: Any, _processor: Any, _prompt: str, _image: Any, settings: GenerationSettings) -> str:
            return reply

        monkeypatch.setattr("cxr_auditor.inference._generate_text", _fake_generate_text)
        findings = generate_findings(tiny_image, model=_FakeModel(), processor=_FakeProcessor())
        assert [f.finding for f in findings] == ["pleural_effusion"]
        assert isinstance(findings[0], ImageFinding)
        assert findings[0].box == (0.6, 0.1, 0.9, 0.4)

    def test_degenerate_truncated_grounding_salvaged_on_first_attempt(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        # A degenerate generation that repeats one finding until the token budget
        # truncates the array mid-element is salvaged on attempt 1 and deduplicated
        # down to a single finding - no retry, no failure.
        element = '{"label": "nodule_mass", "box_2d": [0.2, 0.3, 0.4, 0.5]}'
        reply = "[" + ", ".join([element] * 80) + ', {"label": "nodule_mass", "box_2d": [0.2'
        calls: list[str] = []

        def _fake_generate_text(_model: Any, _processor: Any, prompt: str, _image: Any, settings: GenerationSettings) -> str:
            calls.append(prompt)
            return reply

        monkeypatch.setattr("cxr_auditor.inference._generate_text", _fake_generate_text)
        findings = generate_findings(tiny_image, model=_FakeModel(), processor=_FakeProcessor())
        assert len(calls) == 1
        assert [f.finding for f in findings] == ["nodule_mass"]
        assert findings[0].box == (0.2, 0.3, 0.4, 0.5)

    def test_make_generate_fn_binds_image_and_settings(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        seen: dict[str, Any] = {}

        def _fake_generate_text(_model: Any, _processor: Any, prompt: str, image: Any, settings: GenerationSettings) -> str:
            seen["prompt"] = prompt
            seen["image"] = image
            seen["settings"] = settings
            return "[]"

        monkeypatch.setattr("cxr_auditor.inference._generate_text", _fake_generate_text)
        generate_fn = make_generate_fn(_FakeModel(), _FakeProcessor(), tiny_image, settings=RETRY_SAMPLING_SETTINGS)
        assert generate_fn("the grounding prompt") == "[]"
        assert seen["prompt"] == "the grounding prompt"
        assert seen["image"] is tiny_image
        assert seen["settings"] == RETRY_SAMPLING_SETTINGS

    def test_make_generate_fn_defaults_to_greedy(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        seen: dict[str, Any] = {}

        def _fake_generate_text(_model: Any, _processor: Any, _prompt: str, _image: Any, settings: GenerationSettings) -> str:
            seen["settings"] = settings
            return "[]"

        monkeypatch.setattr("cxr_auditor.inference._generate_text", _fake_generate_text)
        generate_fn = make_generate_fn(_FakeModel(), _FakeProcessor(), tiny_image)
        assert generate_fn("prompt") == "[]"
        assert seen["settings"] == GREEDY_SETTINGS


class TestRunAuditRetries:
    """The orchestration escalates retries and degrades gracefully on the draft."""

    def _patch_generation(self, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> None:
        queued = list(replies)

        def _fake_generate_text(model: Any, processor: Any, prompt: str, image: Any, settings: GenerationSettings) -> str:
            processor.prompts.append(prompt)
            processor.settings.append(settings)
            return queued.pop(0)

        monkeypatch.setattr("cxr_auditor.inference._generate_text", _fake_generate_text)

    def test_grounding_retried_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        self._patch_generation(monkeypatch, ["garbage prose", '[{"label": "cardiomegaly", "box_2d": null}]'])
        processor = _FakeProcessor()
        outcome = run_audit(tiny_image, model=_FakeModel(), processor=processor)
        assert [f.finding for f in outcome.result.image_findings] == ["cardiomegaly"]
        # The second grounding attempt carried the corrective suffix, still greedy.
        assert processor.prompts[1].endswith(RETRY_CORRECTIVE_SUFFIX)
        assert processor.settings == [GREEDY_SETTINGS, GREEDY_SETTINGS]

    def test_grounding_failure_raises_after_full_ladder(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        self._patch_generation(monkeypatch, ["bad one", "bad two", "bad three"])
        processor = _FakeProcessor()
        with pytest.raises(SchemaParseError) as excinfo:
            run_audit(tiny_image, model=_FakeModel(), processor=processor)
        assert excinfo.value.raw_text == "bad three"
        # Three attempts: greedy, greedy + suffix, sampling + suffix.
        assert processor.settings == [GREEDY_SETTINGS, GREEDY_SETTINGS, RETRY_SAMPLING_SETTINGS]
        assert not processor.prompts[0].endswith(RETRY_CORRECTIVE_SUFFIX)
        assert processor.prompts[1].endswith(RETRY_CORRECTIVE_SUFFIX)
        assert processor.prompts[2].endswith(RETRY_CORRECTIVE_SUFFIX)

    def test_draft_parsed_through_parser_injection_seam(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        # First reply grounds the image, second drives parser.parse_draft.
        self._patch_generation(
            monkeypatch,
            [
                '[{"label": "no_finding", "box_2d": null}]',
                '[{"label": "nodule_mass", "status": "present", "span": "a nodule"}]',
            ],
        )
        outcome = run_audit(
            tiny_image,
            draft_text="Round nodule in the RUL.",
            model=_FakeModel(),
            processor=_FakeProcessor(),
        )
        assert [d.finding for d in outcome.result.draft_findings] == ["nodule_mass"]
        assert outcome.result.audit.unsupported_claims == ["nodule_mass"]
        assert outcome.comparison.unsupported[0].draft_span == "a nodule"

    def test_draft_parse_retried_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        self._patch_generation(
            monkeypatch,
            [
                '[{"label": "no_finding", "box_2d": null}]',  # grounding ok
                "the draft model rambled in prose",  # draft attempt 1 fails
                '[{"label": "cardiomegaly", "status": "present"}]',  # draft attempt 2 ok
            ],
        )
        processor = _FakeProcessor()
        outcome = run_audit(
            tiny_image,
            draft_text="Enlarged heart.",
            model=_FakeModel(),
            processor=processor,
        )
        assert [d.finding for d in outcome.result.draft_findings] == ["cardiomegaly"]
        assert outcome.draft_parse_note is None
        # The second draft attempt carried the corrective suffix.
        assert processor.prompts[2].endswith(RETRY_CORRECTIVE_SUFFIX)

    def test_draft_parse_failure_degrades_to_image_only(self, monkeypatch: pytest.MonkeyPatch, tiny_image: Any) -> None:
        # The draft budget is two attempts; when both fail the audit proceeds
        # image-only with a machine-readable note instead of failing.
        self._patch_generation(
            monkeypatch,
            [
                '[{"label": "pleural_effusion", "box_2d": [0.6, 0.1, 0.9, 0.4]}]',  # grounding ok
                "unparseable draft reply one",
                "unparseable draft reply two",
            ],
        )
        processor = _FakeProcessor()
        outcome = run_audit(
            tiny_image,
            draft_text="Left pleural effusion.",
            model=_FakeModel(),
            processor=processor,
        )
        assert outcome.draft_parse_note == DRAFT_PARSE_FAILURE_NOTE
        assert outcome.result.draft_findings == []
        assert [f.finding for f in outcome.result.image_findings] == ["pleural_effusion"]
        # Image-only semantics: missing flags still fire against the empty draft.
        assert outcome.result.audit.missing_findings == ["pleural_effusion"]
        # Exactly one grounding call plus two draft attempts were issued.
        assert len(processor.prompts) == 3


class Error(Exception):
    """Stand-in matching the gradio error class shape the ZeroGPU platform raises.

    The real class's ``__name__`` is literally ``"Error"`` and it carries a
    ``title`` attribute; the categorizer classifies on exactly those, so this
    stand-in exercises it without installing gradio.
    """

    def __init__(self, message: str, title: str = "") -> None:
        super().__init__(message)
        self.title = title


class TestCategorizeServingError:
    """Platform and parse errors map to honest, category-specific messages."""

    @pytest.mark.parametrize(
        "title",
        [
            "ZeroGPU quota exceeded",
            "ZeroGPU illegal duration",
            "ZeroGPU pending credits exceeded",
            "ZeroGPU queue timeout",
            "ZeroGPU client error",
        ],
    )
    def test_scheduling_titles_yield_quota_message(self, title: str) -> None:
        message = categorize_serving_error(Error("60s requested vs. 30s left", title=title))
        assert "GPU quota reached" in message
        assert "could not be parsed" not in message

    def test_unknown_title_mentioning_quota_yields_quota_message(self) -> None:
        message = categorize_serving_error(Error("something", title="ZeroGPU usage quota hit"))
        assert "GPU quota reached" in message

    def test_body_mentioning_quota_yields_quota_message(self) -> None:
        message = categorize_serving_error(Error("Daily quota limit reached", title="Some new title"))
        assert "GPU quota reached" in message

    def test_aborted_gpu_task_yields_interrupted_message(self) -> None:
        message = categorize_serving_error(Error("GPU task aborted", title="ZeroGPU worker error"))
        assert "interrupted" in message
        assert "could not be parsed" not in message

    def test_worker_error_carrying_parse_class_yields_parse_message(self) -> None:
        message = categorize_serving_error(Error("SchemaParseError", title="ZeroGPU worker error"))
        assert "could not be parsed (SchemaParseError)" in message

    def test_worker_error_carrying_other_class_yields_generic_with_real_name(self) -> None:
        message = categorize_serving_error(Error("RuntimeError", title="ZeroGPU worker error"))
        assert "RuntimeError" in message
        assert "could not be parsed" not in message

    def test_worker_error_with_empty_body_falls_back_to_type_name(self) -> None:
        message = categorize_serving_error(Error("", title="ZeroGPU worker error"))
        assert "(Error)" in message

    def test_direct_schema_parse_error_yields_parse_message(self) -> None:
        message = categorize_serving_error(SchemaParseError("no balanced JSON array found", "raw"))
        assert "could not be parsed (SchemaParseError)" in message

    def test_unrelated_exception_yields_generic_with_type_name(self) -> None:
        message = categorize_serving_error(ValueError("boom"))
        assert "(ValueError)" in message
        assert "could not be parsed" not in message
