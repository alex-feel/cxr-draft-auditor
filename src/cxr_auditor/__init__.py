"""CXR Draft Auditor: research/educational QA for chest X-ray draft impressions.

This package provides the pure-logic core (canonical findings, output schema,
and model prompt templates) that the inference, training, and app layers import.
The core depends only on the standard library, numpy, and pydantic; heavier
stacks (torch, transformers, unsloth, gradio) are optional extras imported
lazily so importing this package never requires a GPU stack.

The package root re-exports the key public API so callers can write
``from cxr_auditor import audit, AuditResult, Finding`` without reaching into
submodules. The re-exported ``audit`` / ``run_audit`` entry points themselves
import their vision stack lazily (inside ``load_model`` / ``_generate_text``), so
importing ``cxr_auditor`` still pulls no torch or transformers onto the import
path.

NOT a medical device. NOT diagnosis. Research and educational use only.
"""

from __future__ import annotations

from cxr_auditor.comparator import (
    ComparisonReport,
    MissingFinding,
    UnsupportedClaim,
    build_audit,
    compare,
)
from cxr_auditor.findings import (
    CANONICAL_FINDING_SET,
    CANONICAL_FINDINGS,
    NO_FINDING,
    POSITIVE_FINDINGS,
    URGENT_WHITELIST,
    Finding,
    is_canonical,
    is_urgent,
    map_label,
)
from cxr_auditor.inference import AuditOutcome, audit, run_audit
from cxr_auditor.schema import (
    CANONICAL_BOX_FORMAT,
    DISCLAIMER_TEXT,
    Audit,
    AuditResult,
    DraftFinding,
    FindingStatus,
    ImageFinding,
    SchemaParseError,
    parse_model_output,
)

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_BOX_FORMAT",
    "CANONICAL_FINDINGS",
    "CANONICAL_FINDING_SET",
    "DISCLAIMER_TEXT",
    "NO_FINDING",
    "POSITIVE_FINDINGS",
    "URGENT_WHITELIST",
    "Audit",
    "AuditOutcome",
    "AuditResult",
    "ComparisonReport",
    "DraftFinding",
    "Finding",
    "FindingStatus",
    "ImageFinding",
    "MissingFinding",
    "SchemaParseError",
    "UnsupportedClaim",
    "__version__",
    "audit",
    "build_audit",
    "compare",
    "is_canonical",
    "is_urgent",
    "map_label",
    "parse_model_output",
    "run_audit",
]
