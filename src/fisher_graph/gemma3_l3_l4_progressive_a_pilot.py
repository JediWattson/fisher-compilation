"""Create the open development-only pilot corpus for the Gemma A campaign.

The role JSONs and prompt-free corpus artifact are written under
``.local-runs`` and are ignored by Git.  The prompts below are new open
development material.  They are not derived from structured-v9 and never read
or repartition Calibration B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT,
    DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    DEFAULT_PROGRESSIVE_A_GUARD_INPUT,
    DEFAULT_PROGRESSIVE_A_SELECTION_INPUT,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)


CORPUS_ID = "gemma3-l3-l4-progressive-a-pilot-v1"
PROFILE = "pilot"

FIT_PROMPTS = (
    (
        "A harbor engineer reviews a storm report before sunrise. She lists "
        "the tide height, wind direction, cable tension, and the time each "
        "sensor was calibrated. Then she explains to a new apprentice why a "
        "single alarming measurement should be compared with nearby sensors "
        "before anyone closes the bridge. Write a clear continuation that "
        "keeps the sequence of evidence, uncertainty, and action intact."
    ),
    (
        "During a winter field study, a biologist records tracks near a "
        "frozen stream. The first marks are wide and shallow, the second set "
        "is narrow and deep, and melting snow has blurred both trails. The "
        "team must decide whether one animal changed direction or two animals "
        "crossed at different times. Continue the analysis step by step, "
        "separating direct observations from plausible interpretations."
    ),
    (
        "A patient music teacher introduces syncopation with four measures on "
        "a whiteboard. First the class claps a steady pulse, then accents move "
        "between beats, and finally two groups perform the patterns together. "
        "One student hears only disorder. Explain how the teacher can use the "
        "unchanged pulse as a reference so the shifted accents become easier "
        "to recognize and reproduce."
    ),
    (
        "In a small archive, a curator finds three letters describing the "
        "same journey. One letter emphasizes weather, another lists expenses, "
        "and the third remembers conversations at each stop. Dates overlap "
        "but several place names differ in spelling. Describe a careful "
        "method for building one timeline while preserving disagreements "
        "instead of silently choosing the most vivid account."
    ),
    (
        "A compiler workshop compares two implementations of a numerical "
        "pipeline. The first stores a large intermediate table and performs "
        "few transformations. The second reconstructs that table from compact "
        "factors but performs extra matrix products. Explain why parameter "
        "count, memory traffic, arithmetic work, and wall clock latency are "
        "related yet distinct measurements, using one concrete example for "
        "each tradeoff."
    ),
    (
        "A city planner evaluates two bus routes that serve the same six "
        "neighborhoods. Route A is shorter but transfers passengers at a busy "
        "intersection. Route B travels farther and has fewer transfers, yet "
        "it is delayed by school traffic each afternoon. Compare the routes "
        "without declaring a universal winner, and identify which additional "
        "measurements would change the recommendation for commuters."
    ),
    (
        "Follow these instructions precisely. Read a paragraph, identify its "
        "main claim, list two supporting observations, and mark any sentence "
        "that introduces a new assumption. Next rewrite the claim so it can "
        "be tested, but do not add evidence that was absent from the original "
        "paragraph. Finally state one outcome that would weaken the revised "
        "claim and one outcome that would strengthen it."
    ),
    (
        "A laboratory notebook contains measurements from four instruments "
        "with different units. Convert each quantity to a shared unit system, "
        "retain the original values in an audit column, and flag readings that "
        "fall outside the calibrated range. Explain the order of operations "
        "and why rounding should happen only after comparisons and aggregate "
        "calculations have been completed."
    ),
)

FIT_FAMILIES = (
    "progressive-fit-evidence-a",
    "progressive-fit-evidence-a",
    "progressive-fit-analogy-b",
    "progressive-fit-analogy-b",
    "progressive-fit-comparison-c",
    "progressive-fit-comparison-c",
    "progressive-fit-procedure-d",
    "progressive-fit-procedure-d",
)

SELECTION_PROMPTS = (
    (
        "A telescope control system reports a pointing error after a software "
        "update. The star catalog is unchanged, the motor encoder passes its "
        "self-test, and the error grows with distance from the horizon. Build "
        "a causal troubleshooting order that distinguishes coordinate "
        "conversion, timing, and mechanical alignment. Explain what each test "
        "would rule out before proposing a repair."
    ),
    (
        "An editor must shorten a technical explanation while keeping its "
        "reasoning auditable. The draft defines a metric, gives a counterexample, "
        "compares two systems, and ends with a qualified conclusion. Describe "
        "which repetitions can be compressed, which definitions must remain, "
        "and how to preserve the counterexample so the shorter version does "
        "not sound more certain than the evidence allows."
    ),
    (
        "A warehouse robot chooses between a direct aisle and a longer route "
        "around a loading zone. The direct aisle usually saves time, but its "
        "camera confidence drops when reflective packaging is present. Give "
        "a conditional policy that uses confidence, congestion, and battery "
        "state. Then explain why average travel time alone cannot validate "
        "the safety of that policy."
    ),
    (
        "Consider a model whose compact component gives an approximate answer "
        "and whose corrective component is invoked only for sensitive inputs. "
        "Describe an evaluation that measures answer fidelity, activation "
        "frequency, worst-case compute, and the cost of deciding whether the "
        "correction is needed. Make clear which result would demonstrate true "
        "conditional savings rather than merely moving work into a router."
    ),
)

SELECTION_FAMILIES = (
    "progressive-selection-causal-e",
    "progressive-selection-editing-f",
    "progressive-selection-routing-g",
    "progressive-selection-compute-h",
)

GUARD_PROMPTS = (
    (
        "A flood-response coordinator receives reports from river gauges, "
        "weather radar, and local crews. One gauge jumps suddenly, radar shows "
        "moderate rain, and a crew reports debris near the sensor. Form a "
        "decision that protects residents without treating every source as "
        "equally reliable. Include a reversible immediate action, a method to "
        "verify the anomaly, and a condition that would justify escalation."
    ),
    (
        "A historian compares an official schedule with a personal diary and "
        "a newspaper printed the following morning. The schedule says when an "
        "event was planned, the diary says when one witness arrived, and the "
        "newspaper summarizes the public outcome. Explain how these sources "
        "can support a timeline without confusing plans, observations, and "
        "later interpretation."
    ),
    (
        "A software team replaces one stage of a production pipeline with a "
        "graph executor. Unit tests pass, but a held-out workload has not been "
        "run and latency measurements include diagnostic tracing. State what "
        "the team has demonstrated, what remains unknown, and the minimum "
        "shadow deployment evidence needed before claiming either compression "
        "or speed."
    ),
    (
        "A clinician explains a screening result using sensitivity, "
        "specificity, prevalence, and follow-up testing. Without giving medical "
        "advice, write an educational explanation of why a positive screen is "
        "not identical to a diagnosis. Preserve the distinction between the "
        "test's measured behavior and the decision process that follows from "
        "an individual's broader evidence."
    ),
)

GUARD_FAMILIES = (
    "progressive-guard-risk-i",
    "progressive-guard-sources-j",
    "progressive-guard-deployment-k",
    "progressive-guard-calibration-l",
)


def prepare_gemma3_l3_l4_progressive_a_pilot(
    *,
    fit_output: Path | str = DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    selection_output: Path | str = DEFAULT_PROGRESSIVE_A_SELECTION_INPUT,
    guard_output: Path | str = DEFAULT_PROGRESSIVE_A_GUARD_INPUT,
    corpus_output: Path | str = DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT,
) -> dict[str, object]:
    """Exclusively publish the three role files and prompt-free commitment."""

    paths = {
        "calibration_a_fit": Path(fit_output),
        "calibration_a_selection": Path(selection_output),
        "calibration_a_guard": Path(guard_output),
    }
    role_payloads = {
        "calibration_a_fit": (FIT_PROMPTS, FIT_FAMILIES),
        "calibration_a_selection": (
            SELECTION_PROMPTS,
            SELECTION_FAMILIES,
        ),
        "calibration_a_guard": (GUARD_PROMPTS, GUARD_FAMILIES),
    }
    file_sha256s = {}
    for role, path in paths.items():
        prompts, families = role_payloads[role]
        file_sha256s[role] = (
            write_gemma3_l3_l4_progressive_a_role_input(
                path,
                corpus_id=CORPUS_ID,
                profile=PROFILE,
                role=role,  # type: ignore[arg-type]
                prompts=prompts,
                family_ids=families,
            )
        )
    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    legacy.validate_integrity()
    tokenizer_contract = dict(legacy.metadata()["tokenizer"])
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id=CORPUS_ID,
        profile=PROFILE,
        tokenizer_contract=tokenizer_contract,
        role_input_paths=paths,  # type: ignore[arg-type]
    )
    corpus_file_sha256 = write_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_output,
        artifact,
    )
    return {
        "corpus_id": CORPUS_ID,
        "profile": PROFILE,
        "corpus_artifact_sha256": artifact.artifact_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "tokenizer_contract_sha256": (
            artifact.tokenizer_contract_sha256
        ),
        "role_input_file_sha256s": file_sha256s,
        "roles": {
            view.role: {
                "manifest_sha256": view.manifest_sha256,
                "example_count": view.example_count,
                "family_ids": view.family_ids,
            }
            for view in artifact.role_views
        },
        "structured_v9_reused": False,
        "calibration_b_opened": False,
        "outputs_ignored_by_git": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_SELECTION_INPUT,
    )
    parser.add_argument(
        "--guard-output",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_GUARD_INPUT,
    )
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_gemma3_l3_l4_progressive_a_pilot(
        fit_output=args.fit_output,
        selection_output=args.selection_output,
        guard_output=args.guard_output,
        corpus_output=args.corpus_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
