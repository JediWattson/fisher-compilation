"""Freeze the expanded fit-only panel without opening protected A roles.

The prompt text below is open adaptive-development material.  This command
reads one prompt-free parent corpus artifact, writes one new A-fit role, and
copies the parent's selection and guard preclaim views into a replacement
artifact.  It accepts no selection or guard input paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    _load_progressive_a_artifact,
    gemma3_l3_l4_progressive_a_fit_replacement_lineage,
    replace_gemma3_l3_l4_progressive_a_fit_role,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)


__all__ = [
    "DEFAULT_EXPANDED_FIT_CORPUS_OUTPUT",
    "DEFAULT_EXPANDED_FIT_INPUT_OUTPUT",
    "EXPANDED_FIT_FAMILIES",
    "EXPANDED_FIT_PROMPTS",
    "build_parser",
    "main",
    "prepare_gemma3_l3_l4_progressive_a_expanded_fit",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_PARENT_CORPUS = (
    _LOCAL_ROOT / "progressive-a-loss-v3.corpus.json"
)
DEFAULT_PARENT_CORPUS_ARTIFACT_SHA256 = (
    "55015297b5f06006ac0a03fbb3fa38a15b4d6815625fe35b76ad4a52e3aa066b"
)
DEFAULT_EXPANDED_FIT_INPUT_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
DEFAULT_EXPANDED_FIT_CORPUS_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)

_FAMILIES = (
    "progressive-fit-v2-temporal-dependency-q",
    "progressive-fit-v2-state-invariant-r",
    "progressive-fit-v2-constraint-propagation-s",
    "progressive-fit-v2-counterfactual-isolation-t",
    "progressive-fit-v2-hierarchical-composition-u",
    "progressive-fit-v2-uncertainty-update-v",
    "progressive-fit-v2-reference-frame-w",
    "progressive-fit-v2-budget-allocation-x",
)

_ROUND_ONE_PROMPTS = (
    (
        "A refrigerated medicine shipment passes through a packing room, a "
        "truck, and a clinic freezer. The temperature log rises after a "
        "sensor battery is replaced, but the seal remains intact and a "
        "second thermometer changes later. Reconstruct the most defensible "
        "chronology, explain which event could affect each later reading, "
        "and state the assumptions needed before deciding whether the "
        "shipment ever left its safe range."
    ),
    (
        "A library begins with 420 catalogued maps. Forty maps move to an "
        "exhibit, twelve return for repair, three repair records are later "
        "found to describe duplicates, and seven uncatalogued maps enter the "
        "archive. Build an auditable inventory update that keeps physical "
        "items distinct from records. Identify the quantity that should "
        "remain invariant across transfers and explain how each correction "
        "changes, or does not change, that quantity."
    ),
    (
        "A conference has four rooms, five speakers, two shared projectors, "
        "and one interpreter. The keynote must precede both workshops, one "
        "speaker cannot attend before noon, and the interpreter is required "
        "in sessions that currently overlap. Propagate these constraints in "
        "a clear order, give one feasible partial schedule, and identify the "
        "single assumption whose removal would create the most flexibility."
    ),
    (
        "A greenhouse overheats on three afternoons. Heating commands appear "
        "normal, roof vents open late, cloud cover differs each day, and a "
        "new humidity sensor was installed before the first incident. Design "
        "counterfactual checks that isolate ventilation timing from weather "
        "and sensor calibration. For each check, say what should remain fixed "
        "and what observation would weaken the proposed cause."
    ),
    (
        "A museum catalog groups individual objects into display cases and "
        "cases into galleries. One object moves between cases, a case closes "
        "for conservation, and the visitor guide reports totals at all three "
        "levels. Explain how to update the hierarchy from the object level "
        "upward without double counting. State which totals are local, which "
        "are inherited, and how to audit a disagreement in the gallery total."
    ),
    (
        "A ceramic bowl is tentatively attributed to one workshop because of "
        "its glaze and shape. A new clay analysis supports a neighboring "
        "region, while an excavation note reveals that several storage boxes "
        "were relabeled. Update the competing explanations without treating "
        "either clue as decisive. Separate prior evidence, new evidence, and "
        "source reliability, then name the next observation that would most "
        "change the balance."
    ),
    (
        "An emergency crew reads a building plan that has been rotated ninety "
        "degrees relative to the street. A leak is marked east of the stair, "
        "a valve is north of the elevator in plan coordinates, and the radio "
        "operator describes everything using street directions. Translate "
        "the relations into one reference frame, preserve the original "
        "statements for audit, and explain how an unnoticed mirror operation "
        "would reveal itself."
    ),
    (
        "A town has a fixed maintenance budget for bridge inspection, water "
        "main repair, tree removal, and playground resurfacing. Two jobs are "
        "legally urgent, one repair prevents a larger expected loss, and the "
        "playground project has matching funds that expire soon. Propose a "
        "transparent allocation order, show how dependencies change the "
        "effective cost, and identify which uncertain estimate most affects "
        "the final recommendation."
    ),
)

_ROUND_TWO_PROMPTS = (
    (
        "A theater rehearsal log lists lighting cues, sound cues, and actor "
        "entrances. A lighting reset shifts cue numbers after scene two, the "
        "sound operator follows the old sheet, and an entrance occurs before "
        "both logs record it. Rebuild the causal sequence rather than merely "
        "sorting timestamps. Mark which later events depend on the reset and "
        "which discrepancy could instead come from unsynchronized clocks."
    ),
    (
        "During a database migration, customer records may be copied, split "
        "into household members, merged after duplicate detection, or "
        "rejected for review. The source contains 800 people but only 730 "
        "account rows. Describe a ledger that preserves the invariant number "
        "of represented people while row counts change. Explain how to treat "
        "a rejected row and how to detect a merge that silently loses a person."
    ),
    (
        "Three field crews share two vehicles and one calibrated survey unit. "
        "Each site has a permit window, the river site requires two crews "
        "together, and the mountain vehicle must return before the night "
        "shift. Work through the constraint propagation needed to determine "
        "which assignments are impossible. Give a feasible ordering if one "
        "exists and identify any hidden assumption about travel time."
    ),
    (
        "A photography lab sees a blue color shift in exported images. The "
        "camera files look normal on one monitor, a new display profile was "
        "installed, exports use two different applications, and only some "
        "prints show the shift. Construct counterfactual comparisons that "
        "separate capture, display, export, and printing stages. Explain what "
        "result at each comparison would redirect the investigation."
    ),
    (
        "A festival plan groups booths into themed zones and zones into the "
        "whole site. Two booths share one power circuit, a food booth moves "
        "to another zone, and a closed walkway changes the capacity of two "
        "zones. Recompute the plan from booth constraints upward. Distinguish "
        "attributes that aggregate by addition from constraints that must be "
        "re-evaluated at each higher level."
    ),
    (
        "A factory initially labels a surface defect as a tooling problem. A "
        "second inspection finds the defect on two machines, but only during "
        "one material batch, and the first inspector used a different light. "
        "Update the probabilities of tooling, material, and inspection causes "
        "in stages. State which evidence is conditionally related and avoid "
        "counting the same observation twice."
    ),
    (
        "A support team in New York hands a case to Berlin on the night a "
        "daylight-saving change occurs. The ticket system stores UTC, one "
        "email shows local time, and a handwritten note omits its zone. Put "
        "the events into a single time reference, identify the remaining "
        "ambiguity, and explain why adding a fixed number of hours to every "
        "timestamp is not a safe method."
    ),
    (
        "An observatory allocates a limited month of telescope time among a "
        "rare transient, a long survey, instrument calibration, and weather "
        "contingency. Calibration enables every science program, the transient "
        "window cannot move, and survey value grows unevenly with additional "
        "hours. Construct an allocation that exposes opportunity costs and "
        "dependencies. Then explain how the plan should change if the weather "
        "forecast becomes less reliable."
    ),
)

EXPANDED_FIT_PROMPTS = _ROUND_ONE_PROMPTS + _ROUND_TWO_PROMPTS
EXPANDED_FIT_FAMILIES = _FAMILIES + _FAMILIES


def _default_tokenizer_contract() -> dict[str, object]:
    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    protocol.validate_integrity()
    return dict(protocol.metadata()["tokenizer"])


def prepare_gemma3_l3_l4_progressive_a_expanded_fit(
    *,
    parent_corpus_path: Path | str = DEFAULT_PARENT_CORPUS,
    expected_parent_artifact_sha256: str = (
        DEFAULT_PARENT_CORPUS_ARTIFACT_SHA256
    ),
    fit_output: Path | str = DEFAULT_EXPANDED_FIT_INPUT_OUTPUT,
    corpus_output: Path | str = DEFAULT_EXPANDED_FIT_CORPUS_OUTPUT,
    tokenizer_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish the fixed 16-example, eight-family fit-only replacement."""

    fit_destination = Path(fit_output)
    corpus_destination = Path(corpus_output)
    if fit_destination.exists() or corpus_destination.exists():
        raise FileExistsError("refusing to overwrite expanded fit output")
    contract = (
        dict(tokenizer_contract)
        if tokenizer_contract is not None
        else _default_tokenizer_contract()
    )
    parent = _load_progressive_a_artifact(
        parent_corpus_path,
        expected_artifact_sha256=expected_parent_artifact_sha256,
        tokenizer_contract=contract,
    )
    fit_file_sha256 = write_gemma3_l3_l4_progressive_a_role_input(
        fit_destination,
        corpus_id=parent.corpus_id,
        profile=parent.profile,
        role="calibration_a_fit",
        prompts=EXPANDED_FIT_PROMPTS,
        family_ids=EXPANDED_FIT_FAMILIES,
    )
    replacement = replace_gemma3_l3_l4_progressive_a_fit_role(
        parent_corpus_path,
        fit_input_path=fit_destination,
        expected_parent_artifact_sha256=(
            expected_parent_artifact_sha256
        ),
        tokenizer_contract=contract,
    )
    corpus_file_sha256 = (
        write_gemma3_l3_l4_progressive_a_corpus_artifact(
            corpus_destination,
            replacement,
        )
    )
    lineage = gemma3_l3_l4_progressive_a_fit_replacement_lineage(
        parent,
        replacement,
    )
    return {
        "schema": "fisher_graph.gemma3_l3_l4_expanded_fit_freeze",
        "format_version": 1,
        "example_count": len(EXPANDED_FIT_PROMPTS),
        "family_count": len(set(EXPANDED_FIT_FAMILIES)),
        "examples_per_family": 2,
        "fit_input_file_sha256": fit_file_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "fit_manifest_sha256": replacement.role_view(
            "calibration_a_fit"
        ).manifest_sha256,
        "lineage": lineage,
        "selection_input_capability_present": False,
        "selection_opened": False,
        "guard_input_capability_present": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "outputs_ignored_by_git": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-corpus",
        type=Path,
        default=DEFAULT_PARENT_CORPUS,
    )
    parser.add_argument(
        "--expected-parent-corpus-sha256",
        default=DEFAULT_PARENT_CORPUS_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=DEFAULT_EXPANDED_FIT_INPUT_OUTPUT,
    )
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=DEFAULT_EXPANDED_FIT_CORPUS_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_gemma3_l3_l4_progressive_a_expanded_fit(
        parent_corpus_path=args.parent_corpus,
        expected_parent_artifact_sha256=(
            args.expected_parent_corpus_sha256
        ),
        fit_output=args.fit_output,
        corpus_output=args.corpus_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
