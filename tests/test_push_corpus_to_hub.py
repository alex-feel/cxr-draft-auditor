"""Pure-logic tests for the SFT-corpus push helper ``scripts/push_corpus_to_hub.py``.

The script publishes the SFT corpus as a private Hugging Face Dataset with the
chest X-ray pixels embedded, so a Hugging Face Jobs container can load it without
the local ``data/`` tree. These tests exercise the dependency-light surface: the
default-corpus resolution, the image-path collection / missing detection / size
estimate, the dry-run plan, and the embedded-row builder (which uses PIL, the only
heavy dependency installed in the dev environment). The ``datasets`` push path is
not exercised here (that library is the training extra); the row builder it depends
on is, so the producer cannot silently diverge from the dataset shape the trainer
consumes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from cxr_auditor.schema import FindingStatus, ImageFinding
from cxr_auditor.sft_dataset import build_sft_record

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "push_corpus_to_hub.py"


def _load_script() -> ModuleType:
    """Import the push script by file path (it is a script, not an installed package)."""
    spec = importlib.util.spec_from_file_location("push_corpus_to_hub", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def push() -> ModuleType:
    """The imported ``scripts/push_corpus_to_hub.py`` module."""
    return _load_script()


def _write_corpus(tmp_path: Path, image_paths: list[str]) -> Path:
    """Write a tiny SFT JSONL referencing the given relative image paths."""
    records = [
        build_sft_record(
            image_path,
            [ImageFinding(finding="pneumothorax", status=FindingStatus.PRESENT, box=(0.6, 0.1, 0.9, 0.4))],
        )
        for image_path in image_paths
    ]
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return jsonl


def test_module_imports_without_heavy_deps(push: ModuleType) -> None:
    """Importing the script exposes its CLI surface without datasets / network."""
    assert callable(push.main)
    assert callable(push.build_arg_parser)
    assert callable(push.run)


def test_default_jsonl_prefers_curated(push: ModuleType, tmp_path: Path) -> None:
    """The curated corpus is preferred over the full corpus when it exists."""
    curated = tmp_path / "curated.jsonl"
    full = tmp_path / "full.jsonl"
    curated.write_text("{}\n", encoding="utf-8")
    full.write_text("{}\n", encoding="utf-8")
    chosen = push.default_jsonl([str(curated), str(full)])
    assert chosen == curated


def test_default_jsonl_falls_back_to_full(push: ModuleType, tmp_path: Path) -> None:
    """When the curated corpus is absent, the full corpus is chosen."""
    full = tmp_path / "full.jsonl"
    full.write_text("{}\n", encoding="utf-8")
    chosen = push.default_jsonl([str(tmp_path / "curated.jsonl"), str(full)])
    assert chosen == full


def test_default_jsonl_returns_last_when_none_exist(push: ModuleType, tmp_path: Path) -> None:
    """With no candidate present, the last candidate is returned as the default target."""
    candidates = [str(tmp_path / "curated.jsonl"), str(tmp_path / "full.jsonl")]
    chosen = push.default_jsonl(candidates)
    assert chosen == Path(candidates[-1])


def test_collect_and_missing_images(push: ModuleType, tmp_path: Path, tiny_png_path: Path) -> None:
    """Image paths resolve against the root; missing files are reported once."""
    image_root = tiny_png_path.parent
    records = [
        {"image_path": tiny_png_path.name},
        {"image_path": "does_not_exist.png"},
        {"image_path": "does_not_exist.png"},
    ]
    paths = push.collect_image_paths(records, image_root)
    assert len(paths) == 3
    missing = push.find_missing_images(paths)
    assert len(missing) == 1
    assert missing[0].name == "does_not_exist.png"


def test_estimate_embedded_bytes_counts_distinct(push: ModuleType, tiny_png_path: Path) -> None:
    """The size estimate sums distinct existing files, not per-row duplicates."""
    image_root = tiny_png_path.parent
    one = push.collect_image_paths([{"image_path": tiny_png_path.name}], image_root)
    repeated = push.collect_image_paths(
        [{"image_path": tiny_png_path.name}, {"image_path": tiny_png_path.name}], image_root
    )
    size_one = push.estimate_embedded_bytes(one)
    size_repeated = push.estimate_embedded_bytes(repeated)
    assert size_one > 0
    # The same image referenced twice is counted once.
    assert size_one == size_repeated


def test_summarize_plan_reports_counts_and_missing_warning(push: ModuleType, tmp_path: Path) -> None:
    """The plan names the target, split, counts, and warns when images are missing."""
    plan = push.summarize_plan(
        jsonl_path=tmp_path / "train.jsonl",
        image_root=tmp_path,
        hub_dataset_id="me/cxr-sft",
        split="train",
        private=True,
        record_count=3,
        distinct_image_count=2,
        missing_image_count=1,
        embedded_bytes=2048,
    )
    assert "me/cxr-sft (private)" in plan
    assert "split:               train" in plan
    assert "records (rows):      3" in plan
    assert "distinct images:     2" in plan
    assert "WARNING" in plan


def test_summarize_plan_reports_custom_split(push: ModuleType, tmp_path: Path) -> None:
    """A non-default split name (for the validation corpus) is shown in the plan."""
    plan = push.summarize_plan(
        jsonl_path=tmp_path / "val.curated.jsonl",
        image_root=tmp_path,
        hub_dataset_id="me/cxr-sft",
        split="validation",
        private=True,
        record_count=755,
        distinct_image_count=755,
        missing_image_count=0,
        embedded_bytes=4096,
    )
    assert "split:               validation" in plan


def test_iter_dataset_rows_embeds_rgb_images(push: ModuleType, tiny_png_path: Path) -> None:
    """Each row carries an RGB PIL image and the record's messages."""
    record = build_sft_record(
        tiny_png_path.name,
        [ImageFinding(finding="cardiomegaly", status=FindingStatus.PRESENT, box=(0.3, 0.3, 0.7, 0.7))],
    )
    rows = list(push.iter_dataset_rows([record], tiny_png_path.parent))
    assert len(rows) == 1
    assert set(rows[0]) == {"images", "messages"}
    image = rows[0]["images"][0]
    assert image.mode == "RGB"
    assert rows[0]["messages"] == record["messages"]


def test_iter_dataset_rows_missing_image_raises(push: ModuleType, tmp_path: Path) -> None:
    """A referenced image that is not on disk fails fast."""
    record = build_sft_record("absent.png", [ImageFinding(finding="no_finding", status=FindingStatus.ABSENT)])
    with pytest.raises(FileNotFoundError):
        list(push.iter_dataset_rows([record], tmp_path))


def test_iter_dataset_rows_rejects_record_without_image_part(push: ModuleType, tiny_png_path: Path) -> None:
    """A record whose user turn carries no image part is rejected."""
    record = {
        "image_path": tiny_png_path.name,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "no image part here"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "[{\"label\": \"no_finding\"}]"}]},
        ],
    }
    with pytest.raises(ValueError, match="no 'image' content part"):
        list(push.iter_dataset_rows([record], tiny_png_path.parent))


def test_run_dry_run_no_network(push: ModuleType, tmp_path: Path, tiny_png_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A dry run prints the plan, builds no dataset, and returns success."""
    jsonl = _write_corpus(tmp_path, [tiny_png_path.name])
    exit_code = push.run(
        jsonl_path=jsonl,
        image_root=tiny_png_path.parent,
        hub_dataset_id="me/cxr-sft",
        split="train",
        private=True,
        dry_run=True,
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "me/cxr-sft (private)" in out


def test_run_missing_corpus_errors(push: ModuleType, tmp_path: Path) -> None:
    """A missing corpus file reports a clear error and a nonzero code."""
    exit_code = push.run(
        jsonl_path=tmp_path / "absent.jsonl",
        image_root=tmp_path,
        hub_dataset_id="me/cxr-sft",
        split="train",
        private=True,
        dry_run=True,
    )
    assert exit_code == 1


def test_run_missing_images_blocks_push(push: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-dry run with missing images refuses to push and returns nonzero."""
    jsonl = _write_corpus(tmp_path, ["absent.png"])
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    exit_code = push.run(
        jsonl_path=jsonl,
        image_root=tmp_path,
        hub_dataset_id="me/cxr-sft",
        split="train",
        private=True,
        dry_run=False,
    )
    assert exit_code == 1


def test_run_no_token_blocks_push(push: ModuleType, tmp_path: Path, tiny_png_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-dry run without HF_TOKEN refuses to push and returns nonzero."""
    jsonl = _write_corpus(tmp_path, [tiny_png_path.name])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    exit_code = push.run(
        jsonl_path=jsonl,
        image_root=tiny_png_path.parent,
        hub_dataset_id="me/cxr-sft",
        split="train",
        private=True,
        dry_run=False,
    )
    assert exit_code == 1


def test_main_dry_run_uses_explicit_jsonl(push: ModuleType, tmp_path: Path, tiny_png_path: Path) -> None:
    """The CLI wires --jsonl, --image-root, and --hub-dataset-id through to a dry run."""
    jsonl = _write_corpus(tmp_path, [tiny_png_path.name])
    exit_code = push.main(
        [
            "--hub-dataset-id",
            "me/cxr-sft",
            "--jsonl",
            str(jsonl),
            "--image-root",
            str(tiny_png_path.parent),
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_main_split_defaults_to_train(
    push: ModuleType, tmp_path: Path, tiny_png_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --split the corpus is published under the 'train' split."""
    jsonl = _write_corpus(tmp_path, [tiny_png_path.name])
    push.main(
        [
            "--hub-dataset-id",
            "me/cxr-sft",
            "--jsonl",
            str(jsonl),
            "--image-root",
            str(tiny_png_path.parent),
            "--dry-run",
        ]
    )
    assert "split:               train" in capsys.readouterr().out


def test_main_split_override_publishes_validation_split(
    push: ModuleType, tmp_path: Path, tiny_png_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--split validation routes the curated val corpus to a second split."""
    jsonl = _write_corpus(tmp_path, [tiny_png_path.name])
    exit_code = push.main(
        [
            "--hub-dataset-id",
            "me/cxr-sft",
            "--jsonl",
            str(jsonl),
            "--image-root",
            str(tiny_png_path.parent),
            "--split",
            "validation",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert "split:               validation" in capsys.readouterr().out
