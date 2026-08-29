"""Evaluation subpackage: pure-logic metrics and a CLI evaluation harness.

The ``metrics`` module is dependency-light (stdlib + numpy) so localization,
presence, and audit-flag metrics can be unit-tested without any model-serving
stack. The ``run_eval`` module wires those metrics into an argparse CLI that
reads predictions and ground truth from JSON and writes a JSON report.
"""

from __future__ import annotations
