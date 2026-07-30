"""Rotate a private Calibration-A guard without exposing its prompt text.

This helper preserves the reusable fit and selection roles, creates a fresh
family-disjoint guard from system entropy, and publishes a new prompt-free
corpus commitment.  Its report contains only hashes and counts.  The private
guard role file remains ignored by Git and must not be opened by the campaign
until its durable manifest-global claim has been created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Protocol, Sequence

from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)
from .gemma3_l3_l4_progressive_a_pilot import CORPUS_ID, PROFILE


class _Chooser(Protocol):
    def choice(self, values: Sequence[str]) -> str: ...


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_FIT_INPUT = _LOCAL_ROOT / "progressive-a-pilot-v1.fit.json"
DEFAULT_SELECTION_INPUT = (
    _LOCAL_ROOT / "progressive-a-pilot-v1.selection.json"
)
DEFAULT_ROTATED_GUARD_INPUT = (
    _LOCAL_ROOT / "progressive-a-staged-v2.guard.json"
)
DEFAULT_ROTATED_CORPUS_ARTIFACT = (
    _LOCAL_ROOT / "progressive-a-staged-v2.corpus.json"
)

_FAMILIES = (
    "progressive-guard-v2-investigation-m",
    "progressive-guard-v2-reconciliation-n",
    "progressive-guard-v2-planning-o",
    "progressive-guard-v2-validation-p",
)
_ROLES = (
    (
        "A {operator} is investigating {incident}. "
        "One source reports {signal_a}, while another reports {signal_b}. "
        "A recent change affected {component}, but the timing is uncertain. "
        "Construct an ordered investigation that separates observations from "
        "hypotheses, names a reversible first action, and states what evidence "
        "would justify escalating to a costly intervention."
    ),
    (
        "A research team must reconcile {record_a} with {record_b}. "
        "The records use different {difference}, and each omits a detail the "
        "other includes. Explain how to create a shared account without "
        "silently erasing disagreement. Include provenance, uncertainty, and "
        "a test that could distinguish a true conflict from a conversion or "
        "sampling artifact."
    ),
    (
        "A planner is choosing between {option_a} and {option_b}. "
        "The first improves {benefit_a} but increases {cost_a}; the second "
        "improves {benefit_b} but depends on {assumption}. Give a conditional "
        "policy rather than a universal winner, identify the measurements "
        "needed before deployment, and describe a failure case hidden by "
        "average performance."
    ),
    (
        "A technical team replaced {old_stage} with {new_stage}. "
        "Initial checks show {success}, but {unknown} has not been measured. "
        "Describe a validation ladder that distinguishes functional fidelity, "
        "resource accounting, and wall-clock performance. State which claims "
        "are currently supported, which remain provisional, and what held-out "
        "evidence would falsify the proposed improvement."
    ),
)
_VALUES = {
    "operator": (
        "rail dispatcher",
        "water-system technician",
        "observatory engineer",
        "manufacturing supervisor",
    ),
    "incident": (
        "an intermittent control failure",
        "a sudden sensor disagreement",
        "an unexplained scheduling delay",
        "a repeated quality alert",
    ),
    "signal_a": (
        "a sharp one-minute spike",
        "a gradual drift over several hours",
        "normal behavior followed by a missing interval",
        "a warning limited to one geographic zone",
    ),
    "signal_b": (
        "stable neighboring measurements",
        "a delayed but similar anomaly",
        "operator notes that disagree with the timestamp",
        "a confidence drop without a value change",
    ),
    "component": (
        "clock synchronization",
        "unit conversion",
        "routing configuration",
        "calibration metadata",
    ),
    "record_a": (
        "a field notebook",
        "an automated event log",
        "a hand-entered inventory",
        "a sequence of instrument readings",
    ),
    "record_b": (
        "a later summary report",
        "a second instrument archive",
        "a shipment manifest",
        "an independent observer timeline",
    ),
    "difference": (
        "time zones and rounding rules",
        "identifiers and aggregation windows",
        "units and missing-value conventions",
        "sampling intervals and confidence labels",
    ),
    "option_a": (
        "a centralized dispatch policy",
        "a compact approximate component",
        "a direct high-capacity route",
        "a fixed preventive schedule",
    ),
    "option_b": (
        "a local adaptive policy",
        "a larger conditional correction",
        "a longer redundant route",
        "an event-triggered schedule",
    ),
    "benefit_a": (
        "mean response time",
        "storage efficiency",
        "peak throughput",
        "predictability",
    ),
    "cost_a": (
        "single-point sensitivity",
        "worst-case error",
        "congestion risk",
        "unnecessary maintenance",
    ),
    "benefit_b": (
        "fault isolation",
        "difficult-case fidelity",
        "resilience under disruption",
        "resource targeting",
    ),
    "assumption": (
        "a reliable confidence estimate",
        "timely local measurements",
        "an accurate demand forecast",
        "stable failure detection",
    ),
    "old_stage": (
        "a dense matrix pipeline",
        "a multi-pass numerical routine",
        "a rule-based routing stage",
        "a full intermediate materialization",
    ),
    "new_stage": (
        "a graph executor",
        "a factorized one-pass runtime",
        "a learned conditional router",
        "a compact residual program",
    ),
    "success": (
        "agreement on a small development panel",
        "lower logical parameter accounting",
        "matching output shapes and finite values",
        "stable execution on short sequences",
    ),
    "unknown": (
        "family-disjoint behavior",
        "end-to-end latency without tracing",
        "worst-case memory traffic",
        "long-sequence error accumulation",
    ),
}


def _render_private_guard(chooser: _Chooser) -> tuple[str, ...]:
    prompts = []
    for template in _ROLES:
        values = {
            key: chooser.choice(options)
            for key, options in _VALUES.items()
            if "{" + key + "}" in template
        }
        prompts.append(template.format(**values))
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("rotated guard prompts unexpectedly collided")
    return tuple(prompts)


def rotate_gemma3_l3_l4_progressive_a_guard(
    *,
    fit_input: Path | str = DEFAULT_FIT_INPUT,
    selection_input: Path | str = DEFAULT_SELECTION_INPUT,
    guard_output: Path | str = DEFAULT_ROTATED_GUARD_INPUT,
    corpus_output: Path | str = DEFAULT_ROTATED_CORPUS_ARTIFACT,
    chooser: _Chooser | None = None,
) -> dict[str, object]:
    """Publish a fresh private guard and prompt-free corpus commitment."""

    private_prompts = _render_private_guard(
        random.SystemRandom() if chooser is None else chooser
    )
    guard_path = Path(guard_output)
    guard_file_sha256 = write_gemma3_l3_l4_progressive_a_role_input(
        guard_path,
        corpus_id=CORPUS_ID,
        profile=PROFILE,
        role="calibration_a_guard",
        prompts=private_prompts,
        family_ids=_FAMILIES,
    )
    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    legacy.validate_integrity()
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id=CORPUS_ID,
        profile=PROFILE,
        tokenizer_contract=dict(legacy.metadata()["tokenizer"]),
        role_input_paths={
            "calibration_a_fit": Path(fit_input),
            "calibration_a_selection": Path(selection_input),
            "calibration_a_guard": guard_path,
        },
    )
    corpus_file_sha256 = write_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_output,
        artifact,
    )
    guard_view = next(
        view
        for view in artifact.role_views
        if view.role == "calibration_a_guard"
    )
    return {
        "schema": "fisher_graph.gemma3_l3_l4_progressive_guard_rotation",
        "format_version": 1,
        "corpus_artifact_sha256": artifact.artifact_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "guard_manifest_sha256": guard_view.manifest_sha256,
        "guard_file_sha256": guard_file_sha256,
        "guard_example_count": guard_view.example_count,
        "guard_family_ids": guard_view.family_ids,
        "prompt_text_exposed": False,
        "token_ids_exposed": False,
        "calibration_b_opened": False,
        "reason": "prior_guard_procedurally_compromised_by_out_of_band_read",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_INPUT)
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=DEFAULT_SELECTION_INPUT,
    )
    parser.add_argument(
        "--guard-output",
        type=Path,
        default=DEFAULT_ROTATED_GUARD_INPUT,
    )
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=DEFAULT_ROTATED_CORPUS_ARTIFACT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = rotate_gemma3_l3_l4_progressive_a_guard(
        fit_input=args.fit_input,
        selection_input=args.selection_input,
        guard_output=args.guard_output,
        corpus_output=args.corpus_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
