"""Freeze fresh layer-10 selection and guard roles without exposing guard text.

The upstream Fisher fit is intentionally reused: this rung asks whether the
same compiler map transfers to a different Gemma layer.  Selection and guard
authority are new and disjoint from the layer-17 shape-flow campaign.  All
prompt-bearing role files remain below ``.local-runs`` and are ignored by Git.
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
    Gemma3L3L4ProgressiveACorpusArtifact,
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)
from .gemma3_l3_l4_progressive_a_guard_rotation import (
    _render_private_guard,
)
from .gemma3_l3_l4_progressive_a_pilot import CORPUS_ID, PROFILE


class _Chooser(Protocol):
    def choice(self, values: Sequence[str]) -> str: ...


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_REUSED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
DEFAULT_PRIOR_CORPUS_ARTIFACT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
DEFAULT_SELECTION_OUTPUT = (
    _LOCAL_ROOT / "layer10-shape-flow-v1.selection.json"
)
DEFAULT_GUARD_OUTPUT = _LOCAL_ROOT / "layer10-shape-flow-v1.guard.json"
DEFAULT_CORPUS_OUTPUT = _LOCAL_ROOT / "layer10-shape-flow-v1.corpus.json"

_SELECTION_PROMPTS = (
    (
        "A satellite operations team sees a slow disagreement between two "
        "orientation estimates. One estimate uses star-camera observations; "
        "the other integrates gyroscope readings. Describe a diagnostic "
        "sequence that separates sensor bias, timing drift, and a coordinate "
        "conversion error before any irreversible recalibration."
    ),
    (
        "A data engineer can store a large lookup table or reconstruct it "
        "from compact factors. Compare parameter storage, memory traffic, "
        "arithmetic work, numerical error, and latency. State a measurement "
        "that would reveal when the apparently smaller representation is "
        "actually slower in deployment."
    ),
    (
        "A coastal planner must choose between a fixed evacuation schedule "
        "and a route policy that changes with tide, traffic, and bridge "
        "availability. Give a conditional policy, identify the observations "
        "needed at decision time, and describe a rare case hidden by average "
        "travel-time statistics."
    ),
    (
        "Two laboratories report different values for the same material. "
        "Their instruments use different calibration dates, units, and "
        "sampling intervals. Explain how to reconcile the records while "
        "preserving provenance and how to test whether the disagreement is "
        "physical rather than a conversion artifact."
    ),
    (
        "A compiler replaces several dense operations with a graph of local "
        "generators and conditional edges. Design an evaluation that keeps "
        "node approximation error separate from routing error, accumulated "
        "trajectory error, and the cost of deciding which edge to execute."
    ),
    (
        "An archive contains an official timetable, a witness diary, and a "
        "newspaper summary of one event. Construct a timeline without "
        "confusing plans, observations, and later interpretation. Mark what "
        "is directly supported, what is inferred, and what remains disputed."
    ),
    (
        "A factory schedules preventive maintenance using either fixed "
        "intervals or condition-triggered inspections. Compare the policies "
        "under uncertain sensor reliability, downtime cost, and rare severe "
        "failures. Explain why mean cost alone is not an adequate safety "
        "criterion."
    ),
    (
        "A verification team has unit tests, a simulation, and a short "
        "shadow run for a new numerical runtime. Arrange these into an "
        "evidence ladder. For each stage, state the claim it supports, the "
        "failure it can expose, and the stronger claim it cannot establish."
    ),
)
_SELECTION_FAMILIES = (
    "layer10-selection-sensor-fusion-q",
    "layer10-selection-resource-accounting-r",
    "layer10-selection-conditional-policy-s",
    "layer10-selection-data-reconciliation-t",
    "layer10-selection-graph-diagnostics-u",
    "layer10-selection-source-provenance-v",
    "layer10-selection-risk-scheduling-w",
    "layer10-selection-validation-ladder-x",
)
_GUARD_FAMILIES = (
    "layer10-guard-investigation-y",
    "layer10-guard-reconciliation-z",
    "layer10-guard-planning-aa",
    "layer10-guard-validation-ab",
)


def _load_prompt_free_artifact(path: Path | str) -> Gemma3L3L4ProgressiveACorpusArtifact:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("prior corpus artifact must contain one JSON object")
    return Gemma3L3L4ProgressiveACorpusArtifact.from_dict(raw)


def prepare_gemma3_layer10_shape_flow_corpus(
    *,
    fit_input: Path | str = DEFAULT_REUSED_FIT_INPUT,
    prior_corpus_artifact: Path | str = DEFAULT_PRIOR_CORPUS_ARTIFACT,
    selection_output: Path | str = DEFAULT_SELECTION_OUTPUT,
    guard_output: Path | str = DEFAULT_GUARD_OUTPUT,
    corpus_output: Path | str = DEFAULT_CORPUS_OUTPUT,
    chooser: _Chooser | None = None,
) -> dict[str, object]:
    """Publish fresh selection/guard roles and a prompt-free commitment."""

    guard_prompts = _render_private_guard(
        random.SystemRandom() if chooser is None else chooser
    )
    if set(guard_prompts) & set(_SELECTION_PROMPTS):
        raise RuntimeError("layer-10 selection and guard prompts collided")
    selection_path = Path(selection_output)
    guard_path = Path(guard_output)
    role_file_sha256s = {
        "calibration_a_selection": (
            write_gemma3_l3_l4_progressive_a_role_input(
                selection_path,
                corpus_id=CORPUS_ID,
                profile=PROFILE,
                role="calibration_a_selection",
                prompts=_SELECTION_PROMPTS,
                family_ids=_SELECTION_FAMILIES,
            )
        ),
        "calibration_a_guard": (
            write_gemma3_l3_l4_progressive_a_role_input(
                guard_path,
                corpus_id=CORPUS_ID,
                profile=PROFILE,
                role="calibration_a_guard",
                prompts=guard_prompts,
                family_ids=_GUARD_FAMILIES,
            )
        ),
    }
    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    legacy.validate_integrity()
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id=CORPUS_ID,
        profile=PROFILE,
        tokenizer_contract=dict(legacy.metadata()["tokenizer"]),
        role_input_paths={
            "calibration_a_fit": Path(fit_input),
            "calibration_a_selection": selection_path,
            "calibration_a_guard": guard_path,
        },
    )
    prior = _load_prompt_free_artifact(prior_corpus_artifact)
    prior_prompt_hashes = {
        prompt_sha256
        for role in ("calibration_a_selection", "calibration_a_guard")
        for prompt_sha256 in prior.role_view(role).ordered_prompt_sha256s
    }
    prior_family_ids = {
        family_id
        for role in ("calibration_a_selection", "calibration_a_guard")
        for family_id in prior.role_view(role).family_ids
    }
    new_prompt_hashes = {
        prompt_sha256
        for role in ("calibration_a_selection", "calibration_a_guard")
        for prompt_sha256 in artifact.role_view(role).ordered_prompt_sha256s
    }
    new_family_ids = {
        family_id
        for role in ("calibration_a_selection", "calibration_a_guard")
        for family_id in artifact.role_view(role).family_ids
    }
    prompt_overlap = len(prior_prompt_hashes & new_prompt_hashes)
    family_overlap = len(prior_family_ids & new_family_ids)
    if prompt_overlap or family_overlap:
        raise RuntimeError("layer-10 decision roles overlap layer-17 authority")
    corpus_file_sha256 = write_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_output,
        artifact,
    )
    selection_view = artifact.role_view("calibration_a_selection")
    guard_view = artifact.role_view("calibration_a_guard")
    return {
        "schema": "fisher_graph.gemma3_layer10_shape_flow_corpus",
        "format_version": 1,
        "layer_ordinal": 10,
        "corpus_artifact_sha256": artifact.artifact_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "reused_fit_role_manifest_sha256": (
            artifact.role_view("calibration_a_fit").manifest_sha256
        ),
        "selection_role_manifest_sha256": selection_view.manifest_sha256,
        "selection_example_count": selection_view.example_count,
        "selection_family_ids": selection_view.family_ids,
        "guard_role_manifest_sha256": guard_view.manifest_sha256,
        "guard_example_count": guard_view.example_count,
        "guard_family_ids": guard_view.family_ids,
        "role_input_file_sha256s": role_file_sha256s,
        "prior_layer17_prompt_overlap_count": prompt_overlap,
        "prior_layer17_family_overlap_count": family_overlap,
        "upstream_fisher_fit_reused": True,
        "selection_reused": False,
        "guard_reused": False,
        "prompt_text_exposed": False,
        "token_ids_exposed": False,
        "calibration_b_opened": False,
        "outputs_ignored_by_git": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_REUSED_FIT_INPUT)
    parser.add_argument(
        "--prior-corpus-artifact",
        type=Path,
        default=DEFAULT_PRIOR_CORPUS_ARTIFACT,
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=DEFAULT_SELECTION_OUTPUT,
    )
    parser.add_argument("--guard-output", type=Path, default=DEFAULT_GUARD_OUTPUT)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = prepare_gemma3_layer10_shape_flow_corpus(
        fit_input=arguments.fit_input,
        prior_corpus_artifact=arguments.prior_corpus_artifact,
        selection_output=arguments.selection_output,
        guard_output=arguments.guard_output,
        corpus_output=arguments.corpus_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
