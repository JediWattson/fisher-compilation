"""Frozen downstream-retention pilot for a compiled Gemma graph candidate.

This evaluator deliberately answers a narrower question than ordinary
next-token fidelity: does a frozen graph candidate preserve ground-truth
answers on a small deterministic forced-choice panel?

The panel is claimed before its prompt-bearing rows are parsed.  The claim
binds the candidate, raw panel bytes, evaluator implementation, fixed scoring
contract, and fixed acceptance thresholds.  A crashed invocation may resume
only when all of those bytes remain identical.  No fitting, gain search,
threshold override, or subset selection is exposed here.

This remains a handcrafted 60-item pilot, not an external benchmark or fresh
population-level validation.  Its purpose is to decide whether broader
compiled-scope experiments are warranted without mislabeling native top-1
agreement as downstream accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Literal

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .gemma3_experiment import load_gemma3, resolve_gemma3_huggingface_paths
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_l3_l4_progressive_guard_ledger import (
    load_gemma3_l3_l4_progressive_guard_claim,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecution,
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_state_conditioned_modal_graph_artifact import (
    load_gemma3_state_conditioned_modal_graph_candidate,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    DEFAULT_CORPUS_ARTIFACT,
    DEFAULT_CORPUS_FIT,
    DEFAULT_CORPUS_GUARD,
    DEFAULT_CORPUS_SELECTION,
    DEFAULT_GAIN_CANDIDATE_OUTPUT,
    _load_corpus,
    restore_gemma3_state_conditioned_shape_flow_runtime,
)


__all__ = [
    "DEFAULT_DOWNSTREAM_OUTPUT",
    "DEFAULT_DOWNSTREAM_PANEL",
    "ForcedChoiceConditionResult",
    "ForcedChoiceExample",
    "ForcedChoicePanel",
    "assess_gemma3_downstream_retention",
    "build_parser",
    "evaluate_forced_choice_retention",
    "load_forced_choice_panel",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_DOWNSTREAM_PANEL = Path("examples/gemma3_downstream_retention_v1.json")
DEFAULT_DOWNSTREAM_OUTPUT = _LOCAL_ROOT / "state-conditioned-downstream-v1.json"
_EXPECTED_PANEL_FILE_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
_EXPECTED_GUARD_ASSESSMENT_FILE_SHA256 = (
    "36971b1da3108dfb16fce5f4596f68fc6c80240d230332725a7e016398c0a4a7"
)
_PANEL_SCHEMA = "fisher_graph.gemma3_downstream_retention_panel"
_REPORT_SCHEMA = "fisher_graph.gemma3_downstream_retention_assessment"
_CLAIM_SCHEMA = "fisher_graph.gemma3_downstream_retention_claim"
_PANEL_ID = "gemma3-single-token-forced-choice-v1"
_EXPECTED_EXAMPLES = 60
_EXPECTED_FAMILIES = 6
_EXAMPLES_PER_FAMILY = 10
_CHOICE_COUNT = 4
_MINIMUM_ACCURACY_RETENTION = 0.90
_MINIMUM_NATIVE_WIN_PRESERVATION = 0.90
_MINIMUM_WILSON_LOWER = 0.80
_MAXIMUM_ACCURACY_DROP = 0.05
_MAXIMUM_CHOICE_NLL_INCREASE = 0.10
_MINIMUM_QUALIFIED_FAMILY_FRACTION = 0.80
_MINIMUM_NATIVE_CORRECT = 30
_MINIMUM_QUALIFIED_FAMILIES = 5
_FAMILY_NATIVE_CHANCE_MARGIN = 0.10
_MINIMUM_FAMILY_NATIVE_CORRECT = 3
_WILSON_Z_ONE_SIDED_90 = 1.2815515655446004
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ConditionName = Literal["native", "edgeless", "candidate"]


def _progress(message: str) -> None:
    print(f"[downstream-retention] {message}", file=sys.stderr, flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return _sha256_bytes(domain + _canonical_json_bytes(value))


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable identifier")
    return value


@dataclass(frozen=True, slots=True)
class ForcedChoiceExample:
    example_id: str
    family_id: str
    prompt: str
    choices: tuple[str, ...]
    correct_choice: int

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, label="example_id")
        _require_identifier(self.family_id, label="family_id")
        if (
            not isinstance(self.prompt, str)
            or not self.prompt
            or self.prompt != self.prompt.strip()
        ):
            raise ValueError("prompt must be canonical nonempty text")
        if (
            type(self.choices) is not tuple
            or len(self.choices) != _CHOICE_COUNT
            or len(set(self.choices)) != _CHOICE_COUNT
            or any(
                not isinstance(choice, str)
                or not choice
                or choice != choice.strip()
                for choice in self.choices
            )
        ):
            raise ValueError("choices must contain four unique canonical strings")
        if (
            type(self.correct_choice) is not int
            or not 0 <= self.correct_choice < len(self.choices)
        ):
            raise ValueError("correct_choice is outside the choice catalog")

    @property
    def prompt_sha256(self) -> str:
        return gemma3_l3_l4_graph_organized_svd_prompt_sha256(self.prompt)


@dataclass(frozen=True, slots=True)
class ForcedChoicePanel:
    panel_id: str
    examples: tuple[ForcedChoiceExample, ...]
    file_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.panel_id, label="panel_id")
        _require_sha256(self.file_sha256, label="panel file")
        if type(self.examples) is not tuple or not self.examples:
            raise ValueError("panel must contain at least one example")
        ids = tuple(example.example_id for example in self.examples)
        prompts = tuple(example.prompt_sha256 for example in self.examples)
        if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
            raise ValueError("panel example and prompt identities must be unique")
        counts = Counter(example.family_id for example in self.examples)
        if not counts:
            raise ValueError("panel must contain at least one family")

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(sorted({example.family_id for example in self.examples}))

    @property
    def semantic_sha256(self) -> str:
        return _domain_sha256(
            b"fisher-graph:gemma3-downstream-panel:v1\0",
            {
                "panel_id": self.panel_id,
                "examples": tuple(
                    {
                        "example_id": example.example_id,
                        "family_id": example.family_id,
                        "prompt": example.prompt,
                        "choices": example.choices,
                        "correct_choice": example.correct_choice,
                    }
                    for example in self.examples
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class ForcedChoiceConditionResult:
    name: ConditionName
    predictions: tuple[int, ...]
    restricted_choice_nll: tuple[float, ...]
    gold_margins: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.name not in {"native", "edgeless", "candidate"}:
            raise ValueError("invalid forced-choice condition")
        count = len(self.predictions)
        if (
            type(self.predictions) is not tuple
            or type(self.restricted_choice_nll) is not tuple
            or type(self.gold_margins) is not tuple
            or count == 0
            or len(self.restricted_choice_nll) != count
            or len(self.gold_margins) != count
            or any(
                type(prediction) is not int
                or not 0 <= prediction < _CHOICE_COUNT
                for prediction in self.predictions
            )
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.restricted_choice_nll
            )
            or any(not math.isfinite(value) for value in self.gold_margins)
        ):
            raise ValueError("forced-choice condition rows are invalid")


def load_forced_choice_panel(
    path: Path | str = DEFAULT_DOWNSTREAM_PANEL,
    *,
    expected_file_sha256: str = _EXPECTED_PANEL_FILE_SHA256,
) -> ForcedChoicePanel:
    """Strict-load the exact prompt-bearing panel after its claim boundary."""

    source = Path(path)
    encoded = source.read_bytes()
    observed = _sha256_bytes(encoded)
    if observed != _require_sha256(expected_file_sha256, label="expected panel"):
        raise ValueError("forced-choice panel bytes differ from the frozen protocol")
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("forced-choice panel is not strict UTF-8 JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "format_version",
        "panel_id",
        "choice_count",
        "examples",
    }:
        raise ValueError("forced-choice panel fields differ from the schema")
    if (
        raw["schema"] != _PANEL_SCHEMA
        or raw["format_version"] != 1
        or raw["panel_id"] != _PANEL_ID
        or raw["choice_count"] != _CHOICE_COUNT
        or not isinstance(raw["examples"], list)
    ):
        raise ValueError("forced-choice panel header is invalid")
    examples: list[ForcedChoiceExample] = []
    for row in raw["examples"]:
        if not isinstance(row, dict) or set(row) != {
            "example_id",
            "family_id",
            "prompt",
            "choices",
            "correct_choice",
        }:
            raise ValueError("forced-choice example fields differ from the schema")
        choices = row["choices"]
        if not isinstance(choices, list):
            raise ValueError("forced-choice choices must be a list")
        examples.append(
            ForcedChoiceExample(
                example_id=row["example_id"],
                family_id=row["family_id"],
                prompt=row["prompt"],
                choices=tuple(choices),
                correct_choice=row["correct_choice"],
            )
        )
    panel = ForcedChoicePanel(
        panel_id=raw["panel_id"],
        examples=tuple(examples),
        file_sha256=observed,
    )
    counts = Counter(example.family_id for example in panel.examples)
    if (
        panel.panel_id != _PANEL_ID
        or len(panel.examples) != _EXPECTED_EXAMPLES
        or len(counts) != _EXPECTED_FAMILIES
        or set(counts.values()) != {_EXAMPLES_PER_FAMILY}
    ):
        raise ValueError("v1 panel must contain six families of ten examples")
    return panel


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _wilson_lower(successes: int, observations: int) -> float | None:
    if observations == 0:
        return None
    if not 0 <= successes <= observations:
        raise ValueError("Wilson counts are invalid")
    p = successes / observations
    z = _WILSON_Z_ONE_SIDED_90
    denominator = 1.0 + z * z / observations
    center = p + z * z / (2.0 * observations)
    radius = z * math.sqrt(
        p * (1.0 - p) / observations
        + z * z / (4.0 * observations * observations)
    )
    return (center - radius) / denominator


def _condition_summary(
    panel: ForcedChoicePanel,
    result: ForcedChoiceConditionResult,
) -> dict[str, object]:
    correct = tuple(
        prediction == example.correct_choice
        for prediction, example in zip(
            result.predictions,
            panel.examples,
            strict=True,
        )
    )
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(panel.examples):
        by_family[example.family_id].append(index)
    family = {
        name: {
            "example_count": len(indices),
            "correct_count": sum(int(correct[index]) for index in indices),
            "accuracy": _mean(tuple(float(correct[index]) for index in indices)),
            "restricted_choice_nll": _mean(
                tuple(result.restricted_choice_nll[index] for index in indices)
            ),
            "mean_gold_margin": _mean(
                tuple(result.gold_margins[index] for index in indices)
            ),
        }
        for name, indices in sorted(by_family.items())
    }
    return {
        "example_count": len(panel.examples),
        "correct_count": sum(int(value) for value in correct),
        "accuracy": _mean(tuple(float(value) for value in correct)),
        "restricted_choice_nll": _mean(result.restricted_choice_nll),
        "mean_gold_margin": _mean(result.gold_margins),
        "prediction_stream_sha256": _domain_sha256(
            b"fisher-graph:gemma3-downstream-predictions:v1\0",
            {
                "condition": result.name,
                "predictions": result.predictions,
            },
        ),
        "families": family,
    }


def evaluate_forced_choice_retention(
    panel: ForcedChoicePanel,
    *,
    native: ForcedChoiceConditionResult,
    edgeless: ForcedChoiceConditionResult,
    candidate: ForcedChoiceConditionResult,
    prequalified_family_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Reduce frozen condition outputs into paired retention gates."""

    if (native.name, edgeless.name, candidate.name) != (
        "native",
        "edgeless",
        "candidate",
    ):
        raise ValueError("forced-choice conditions are not canonically ordered")
    if any(
        len(result.predictions) != len(panel.examples)
        for result in (native, edgeless, candidate)
    ):
        raise ValueError("condition row counts differ from the panel")
    summaries = {
        result.name: _condition_summary(panel, result)
        for result in (native, edgeless, candidate)
    }
    gold = tuple(example.correct_choice for example in panel.examples)
    native_correct_rows = tuple(
        prediction == answer
        for prediction, answer in zip(native.predictions, gold, strict=True)
    )
    candidate_correct_rows = tuple(
        prediction == answer
        for prediction, answer in zip(candidate.predictions, gold, strict=True)
    )
    edgeless_correct_rows = tuple(
        prediction == answer
        for prediction, answer in zip(edgeless.predictions, gold, strict=True)
    )
    native_correct = sum(native_correct_rows)
    candidate_correct = sum(candidate_correct_rows)
    both_correct = sum(
        left and right
        for left, right in zip(
            native_correct_rows,
            candidate_correct_rows,
            strict=True,
        )
    )
    native_only = sum(
        left and not right
        for left, right in zip(
            native_correct_rows,
            candidate_correct_rows,
            strict=True,
        )
    )
    candidate_only = sum(
        right and not left
        for left, right in zip(
            native_correct_rows,
            candidate_correct_rows,
            strict=True,
        )
    )
    both_wrong = len(panel.examples) - both_correct - native_only - candidate_only
    accuracy_ratio = (
        None if native_correct == 0 else candidate_correct / native_correct
    )
    accuracy_retention = (
        None if accuracy_ratio is None else min(1.0, accuracy_ratio)
    )
    native_win_preservation = (
        None if native_correct == 0 else both_correct / native_correct
    )
    accuracy_delta = (candidate_correct - native_correct) / len(panel.examples)

    family_rows: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(panel.examples):
        family_rows[example.family_id].append(index)
    panel_family_ids = tuple(sorted(family_rows))
    if prequalified_family_ids is None:
        prequalified: frozenset[str] | None = None
    else:
        supplied = tuple(prequalified_family_ids)
        if (
            len(supplied) != len(set(supplied))
            or set(supplied) != set(panel_family_ids)
        ):
            raise ValueError(
                "prequalified families must exactly cover the evaluation panel"
            )
        prequalified = frozenset(supplied)
    family_comparisons: dict[str, dict[str, object]] = {}
    qualified: list[str] = []
    for family_id, indices in sorted(family_rows.items()):
        native_family_correct = sum(native_correct_rows[index] for index in indices)
        candidate_family_correct = sum(
            candidate_correct_rows[index] for index in indices
        )
        both_family_correct = sum(
            native_correct_rows[index] and candidate_correct_rows[index]
            for index in indices
        )
        native_accuracy = native_family_correct / len(indices)
        eligible = (
            family_id in prequalified
            if prequalified is not None
            else (
                native_family_correct >= _MINIMUM_FAMILY_NATIVE_CORRECT
                and native_accuracy
                >= 1.0 / _CHOICE_COUNT + _FAMILY_NATIVE_CHANCE_MARGIN
            )
        )
        if eligible:
            qualified.append(family_id)
        family_comparisons[family_id] = {
            "example_count": len(indices),
            "native_correct_count": native_family_correct,
            "candidate_correct_count": candidate_family_correct,
            "both_correct_count": both_family_correct,
            "native_only_correct_count": (
                native_family_correct - both_family_correct
            ),
            "accuracy_retained_fraction": (
                None
                if native_family_correct == 0
                else min(1.0, candidate_family_correct / native_family_correct)
            ),
            "native_win_preservation": (
                None
                if native_family_correct == 0
                else both_family_correct / native_family_correct
            ),
            "primary_eligible": eligible,
            "eligibility_source": (
                "separate_native_only_qualification_split"
                if prequalified is not None
                else "evaluation_native_denominator"
            ),
        }
    adequate = (
        native_correct >= _MINIMUM_NATIVE_CORRECT
        and len(qualified) >= _MINIMUM_QUALIFIED_FAMILIES
    )
    small_loss_families = sum(
        int(
            int(family_comparisons[name]["native_only_correct_count"]) <= 1
        )
        for name in qualified
    )
    required_small_loss_families = math.ceil(
        _MINIMUM_QUALIFIED_FAMILY_FRACTION * len(qualified)
    )
    maximum_family_losses = max(
        (
            int(family_comparisons[name]["native_only_correct_count"])
            for name in qualified
        ),
        default=0,
    )
    wilson = _wilson_lower(both_correct, native_correct)
    candidate_nll = float(summaries["candidate"]["restricted_choice_nll"])
    native_nll = float(summaries["native"]["restricted_choice_nll"])
    edgeless_nll = float(summaries["edgeless"]["restricted_choice_nll"])
    candidate_accuracy = float(summaries["candidate"]["accuracy"])
    edgeless_accuracy = float(summaries["edgeless"]["accuracy"])
    gates = {
        "suite_adequate": adequate,
        "global_accuracy_retention_at_least_0_90": (
            accuracy_retention is not None
            and accuracy_retention >= _MINIMUM_ACCURACY_RETENTION
        ),
        "global_native_win_preservation_at_least_0_90": (
            native_win_preservation is not None
            and native_win_preservation >= _MINIMUM_NATIVE_WIN_PRESERVATION
        ),
        "native_win_preservation_one_sided_90pct_wilson_lower_at_least_0_80": (
            wilson is not None and wilson >= _MINIMUM_WILSON_LOWER
        ),
        "absolute_accuracy_drop_at_most_0_05": (
            accuracy_delta >= -_MAXIMUM_ACCURACY_DROP
        ),
        "at_least_80pct_qualified_families_lose_at_most_one": (
            adequate and small_loss_families >= required_small_loss_families
        ),
        "no_qualified_family_loses_more_than_two": (
            adequate and maximum_family_losses <= 2
        ),
        "candidate_choice_nll_minus_native_at_most_0_10": (
            candidate_nll - native_nll <= _MAXIMUM_CHOICE_NLL_INCREASE
        ),
    }
    passed = all(gates.values())
    return {
        "status": (
            "downstream_retention_pilot_pass"
            if passed
            else (
                "inconclusive_native_denominator"
                if not adequate
                else "downstream_retention_pilot_fail"
            )
        ),
        "passed": passed,
        "conditions": summaries,
        "paired_candidate_vs_native": {
            "native_correct_count": native_correct,
            "candidate_correct_count": candidate_correct,
            "both_correct_count": both_correct,
            "native_only_correct_count": native_only,
            "candidate_only_correct_count": candidate_only,
            "both_wrong_count": both_wrong,
            "accuracy_ratio_to_native": accuracy_ratio,
            "accuracy_retained_fraction": accuracy_retention,
            "native_win_preservation": native_win_preservation,
            "native_win_preservation_one_sided_90pct_wilson_lower": wilson,
            "accuracy_delta": accuracy_delta,
            "restricted_choice_nll_delta": candidate_nll - native_nll,
            "restricted_choice_perplexity_multiplier": math.exp(
                candidate_nll - native_nll
            ),
            "prediction_agreement": _mean(
                tuple(
                    float(left == right)
                    for left, right in zip(
                        native.predictions,
                        candidate.predictions,
                        strict=True,
                    )
                )
            ),
        },
        "conditional_edge_value_added": {
            "candidate_choice_nll_below_edgeless": candidate_nll < edgeless_nll,
            "candidate_accuracy_not_below_edgeless": (
                candidate_accuracy >= edgeless_accuracy
            ),
            "restricted_choice_nll_improvement_over_edgeless": (
                edgeless_nll - candidate_nll
            ),
            "accuracy_delta_over_edgeless": (
                candidate_accuracy - edgeless_accuracy
            ),
            "edgeless_correct_candidate_wrong_count": sum(
                left and not right
                for left, right in zip(
                    edgeless_correct_rows,
                    candidate_correct_rows,
                    strict=True,
                )
            ),
            "edgeless_wrong_candidate_correct_count": sum(
                not left and right
                for left, right in zip(
                    edgeless_correct_rows,
                    candidate_correct_rows,
                    strict=True,
                )
            ),
        },
        "family_comparisons": family_comparisons,
        "adequacy": {
            "minimum_native_correct": _MINIMUM_NATIVE_CORRECT,
            "observed_native_correct": native_correct,
            "minimum_qualified_families": _MINIMUM_QUALIFIED_FAMILIES,
            "observed_qualified_families": len(qualified),
            "qualified_family_ids": tuple(qualified),
            "family_eligibility_source": (
                "separate_native_only_qualification_split"
                if prequalified is not None
                else "evaluation_native_denominator"
            ),
        },
        "gates": gates,
    }


def _candidate_metadata(raw: Mapping[str, object]) -> dict[str, str]:
    experiment = raw.get("experiment")
    selection = raw.get("selection")
    pipeline = raw.get("compiler_pipeline")
    if not isinstance(experiment, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("candidate metadata is unavailable")
    if (
        not isinstance(pipeline, Mapping)
        or selection.get("promotion_passed") is not True
    ):
        raise ValueError("candidate did not pass compiler promotion")
    values = {
        "scientific_payload_sha256": raw.get("scientific_payload_sha256"),
        "dynamic_graph_sha256": selection.get("dynamic_graph_sha256"),
        "compiler_pipeline_sha256": selection.get("compiler_pipeline_sha256"),
        "interaction_promotion_sha256": selection.get(
            "interaction_promotion_sha256"
        ),
    }
    return {
        name: _require_sha256(value, label=f"candidate {name}")
        for name, value in values.items()
    }


def _validate_guard_assessment(
    path: Path,
    *,
    candidate: Mapping[str, str],
    expected_tensor_file: str,
    expected_guard_manifest_sha256: str,
    expected_guard_assessment_file_sha256: str = (
        _EXPECTED_GUARD_ASSESSMENT_FILE_SHA256
    ),
) -> tuple[dict[str, object], str]:
    encoded = path.read_bytes()
    file_sha256 = _sha256_bytes(encoded)
    if file_sha256 != _require_sha256(
        expected_guard_assessment_file_sha256,
        label="expected guard assessment file",
    ):
        raise ValueError("guard assessment bytes differ from the frozen result")
    def reject_nonfinite_constant(value: str) -> object:
        raise ValueError(f"guard assessment contains non-finite {value}")

    raw = json.loads(
        encoded.decode("utf-8"),
        parse_constant=reject_nonfinite_constant,
    )
    expected_top_level = {
        "behavior",
        "candidate",
        "flow",
        "format_version",
        "guard",
        "guard_nll_improvement_over_edgeless",
        "resource_summary",
        "routing_execution",
        "schema",
        "scientific_status",
        "source_model_unchanged",
    }
    if not isinstance(raw, dict) or set(raw) != expected_top_level:
        raise ValueError("guard assessment must contain one JSON object")
    assessment_candidate = raw.get("candidate")
    status = raw.get("scientific_status")
    guard = raw.get("guard")
    behavior = raw.get("behavior")
    routing = raw.get("routing_execution")
    flow = raw.get("flow")
    resources = raw.get("resource_summary")
    if (
        raw.get("schema")
        != "fisher_graph.gemma3_state_conditioned_shape_flow_assessment"
        or raw.get("format_version") != 1
        or not isinstance(assessment_candidate, Mapping)
        or set(assessment_candidate)
        != {
            "compiler_pipeline_sha256",
            "dynamic_graph_sha256",
            "interaction_promotion_sha256",
            "scientific_payload_sha256",
            "tensor_file",
        }
        or not isinstance(status, Mapping)
        or set(status)
        != {
            "fresh_validation",
            "guard_claimed_before_materialization",
            "heldout_confirmation",
            "open_development",
            "role",
            "test_data_used",
        }
        or status.get("role") != "family_disjoint_calibration_a_guard"
        or status.get("guard_claimed_before_materialization") is not True
        or status.get("open_development") is not True
        or status.get("heldout_confirmation") is not False
        or status.get("fresh_validation") is not False
        or status.get("test_data_used") is not False
        or raw.get("source_model_unchanged") is not True
    ):
        raise ValueError("guard assessment identity or status is invalid")
    for name, expected in candidate.items():
        if assessment_candidate.get(name) != expected:
            raise ValueError("guard assessment candidate binding drifted")
    if assessment_candidate.get("tensor_file") != expected_tensor_file:
        raise ValueError("guard assessment tensor-file binding drifted")
    if (
        not isinstance(guard, Mapping)
        or set(guard)
        != {
            "claim_sha256",
            "example_count",
            "family_ids",
            "role_manifest_sha256",
            "tokenized_split_sha256",
        }
        or guard.get("role_manifest_sha256")
        != _require_sha256(
            expected_guard_manifest_sha256,
            label="expected guard manifest",
        )
        or not isinstance(guard.get("example_count"), int)
        or isinstance(guard.get("example_count"), bool)
        or int(guard["example_count"]) <= 0
        or not isinstance(guard.get("family_ids"), list)
        or len(guard["family_ids"]) != int(guard["example_count"])
        or not all(
            isinstance(family_id, str)
            and _IDENTIFIER.fullmatch(family_id) is not None
            for family_id in guard["family_ids"]
        )
        or len(set(guard["family_ids"])) != len(guard["family_ids"])
    ):
        raise ValueError("guard assessment manifest binding is invalid")
    guard_claim_sha256 = _require_sha256(
        guard.get("claim_sha256"),
        label="guard claim",
    )
    _require_sha256(
        guard.get("tokenized_split_sha256"),
        label="guard tokenized split",
    )
    try:
        persisted_claim = load_gemma3_l3_l4_progressive_guard_claim(
            protocol_sha256=candidate["compiler_pipeline_sha256"],
            guard_manifest_sha256=expected_guard_manifest_sha256,
            challenger_receipt_sha256=candidate[
                "interaction_promotion_sha256"
            ],
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "guard assessment lacks an authenticated persisted claim"
        ) from error
    if persisted_claim.claim_sha256 != guard_claim_sha256:
        raise ValueError("guard assessment claim binding drifted")

    def finite_float(value: object, *, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} must be finite")
        return float(value)

    def positive_int(value: object, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        return value

    if (
        not isinstance(behavior, Mapping)
        or set(behavior)
        != {
            "assessment_role",
            "conditions",
            "execution_path",
            "graph_comparison",
            "heldout_confirmation",
            "latency_or_kernel_speed_claim",
            "logical_valid_tokens",
            "native",
            "resource_accounting",
            "supervised_tokens",
        }
        or behavior.get("assessment_role") != "open_development_assessment"
        or behavior.get("heldout_confirmation") is not False
        or behavior.get("latency_or_kernel_speed_claim") is not False
        or not isinstance(behavior.get("conditions"), Mapping)
    ):
        raise ValueError("guard behavioral assessment is invalid")
    conditions = behavior["conditions"]
    assert isinstance(conditions, Mapping)
    if set(conditions) != {
        "edgeless_graph",
        "interacting_graph",
        "matched_deletion",
    }:
        raise ValueError("guard comparison condition set is invalid")
    edgeless = conditions.get("edgeless_graph")
    interacting = conditions.get("interacting_graph")
    matched_deletion = conditions.get("matched_deletion")
    condition_keys = {
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "nll_per_token",
        "top1_agreement_to_native",
    }
    if not all(
        isinstance(condition, Mapping) and set(condition) == condition_keys
        for condition in (edgeless, interacting, matched_deletion)
    ):
        raise ValueError("guard comparison conditions are unavailable")
    assert isinstance(edgeless, Mapping)
    assert isinstance(interacting, Mapping)
    assert isinstance(matched_deletion, Mapping)
    native = behavior.get("native")
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError("guard native behavior is invalid")
    native_nll = finite_float(
        native["nll_per_token"],
        label="guard native NLL",
    )
    if native_nll < 0.0:
        raise ValueError("guard native NLL must be non-negative")
    for name, condition in (
        ("edgeless", edgeless),
        ("interacting", interacting),
        ("matched deletion", matched_deletion),
    ):
        nll = finite_float(
            condition["nll_per_token"],
            label=f"guard {name} NLL",
        )
        delta = finite_float(
            condition["delta_nll_per_token"],
            label=f"guard {name} delta NLL",
        )
        kl = finite_float(
            condition["native_to_candidate_kl_per_token"],
            label=f"guard {name} KL",
        )
        agreement = finite_float(
            condition["top1_agreement_to_native"],
            label=f"guard {name} top-1 agreement",
        )
        if (
            nll < 0.0
            or kl < 0.0
            or not 0.0 <= agreement <= 1.0
            or not math.isclose(
                delta,
                nll - native_nll,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"guard {name} metrics are inconsistent")
    improvement = raw.get("guard_nll_improvement_over_edgeless")
    if (
        isinstance(improvement, bool)
        or not isinstance(improvement, (int, float))
        or not math.isfinite(float(improvement))
        or improvement <= 0.0
        or not math.isclose(
            float(improvement),
            float(edgeless["nll_per_token"])
            - float(interacting["nll_per_token"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("candidate lacks a recomputable positive guard result")

    graph_comparison = behavior.get("graph_comparison")
    expected_graph_keys = {
        "deletion_equivalence_atol",
        "deletion_equivalence_rtol",
        "deletion_equivalence_scope",
        "deletion_max_abs_logit_difference",
        "deletion_paths_agree",
        "edgeless_edge_count",
        "interacting_edge_count",
        "interaction_parameter_delta",
        "matched_deletion_resource_scope",
        "node_artifacts_identical",
        "node_count",
        "nodewise_dense_agrees_with_edgeless",
        "nodewise_dense_equivalence_atol",
        "nodewise_dense_equivalence_rtol",
        "nodewise_dense_equivalence_scope",
        "nodewise_dense_max_abs_logit_difference",
        "nodewise_dense_supplied",
    }
    if (
        not isinstance(graph_comparison, Mapping)
        or set(graph_comparison) != expected_graph_keys
        or graph_comparison.get("deletion_paths_agree") is not True
        or graph_comparison.get("node_artifacts_identical") is not True
        or graph_comparison.get("nodewise_dense_supplied") is not False
        or graph_comparison.get("deletion_equivalence_scope")
        != "supervised_logits"
        or finite_float(
            graph_comparison.get("deletion_max_abs_logit_difference"),
            label="guard deletion equivalence",
        )
        != 0.0
        or graph_comparison.get("edgeless_edge_count") != 0
        or positive_int(
            graph_comparison.get("interacting_edge_count"),
            label="guard interacting edge count",
        )
        <= 0
        or positive_int(
            graph_comparison.get("interaction_parameter_delta"),
            label="guard interaction parameter delta",
        )
        <= 0
        or positive_int(
            graph_comparison.get("node_count"),
            label="guard node count",
        )
        <= 0
    ):
        raise ValueError("guard graph comparison receipt is invalid")

    behavior_valid_tokens = positive_int(
        behavior.get("logical_valid_tokens"),
        label="guard logical valid tokens",
    )
    supervised_tokens = positive_int(
        behavior.get("supervised_tokens"),
        label="guard supervised tokens",
    )
    if supervised_tokens > behavior_valid_tokens:
        raise ValueError("guard token accounting is invalid")
    if (
        not isinstance(routing, Mapping)
        or set(routing)
        != {
            "exactly_one_selected_edge_per_valid_token",
            "selected_edge_rows",
            "selected_edge_rows_by_interaction",
            "valid_tokens",
        }
        or routing.get("exactly_one_selected_edge_per_valid_token") is not True
        or routing.get("selected_edge_rows") != routing.get("valid_tokens")
    ):
        raise ValueError("guard routing execution receipt is invalid")
    routing_tokens = positive_int(
        routing.get("valid_tokens"),
        label="guard routing valid tokens",
    )
    selected_rows = positive_int(
        routing.get("selected_edge_rows"),
        label="guard selected edge rows",
    )
    rows_by_interaction = routing.get("selected_edge_rows_by_interaction")
    if (
        routing_tokens != behavior_valid_tokens
        or selected_rows != routing_tokens
        or not isinstance(rows_by_interaction, Mapping)
        or len(rows_by_interaction)
        != int(graph_comparison["interacting_edge_count"])
        or not all(
            isinstance(name, str)
            and name
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for name, count in rows_by_interaction.items()
        )
        or sum(int(count) for count in rows_by_interaction.values())
        != selected_rows
    ):
        raise ValueError("guard routing row accounting is invalid")

    expected_resource_keys = {
        "candidate_whole_model_learned_parameters",
        "logical_executed_modal_graph_macs",
        "logical_modal_graph_macs",
        "logical_valid_tokens",
        "modal_graph_learned_parameters",
        "native_removed_learned_parameters",
        "source_whole_model_learned_parameters",
    }
    if not isinstance(resources, Mapping) or set(resources) != expected_resource_keys:
        raise ValueError("guard resource summary is invalid")
    resource_values = {
        name: positive_int(value, label=f"guard resource {name}")
        for name, value in resources.items()
    }
    if (
        resource_values["logical_valid_tokens"] != routing_tokens
        or resource_values["logical_executed_modal_graph_macs"]
        > resource_values["logical_modal_graph_macs"]
        or resource_values["modal_graph_learned_parameters"]
        >= resource_values["native_removed_learned_parameters"]
        or resource_values["candidate_whole_model_learned_parameters"]
        != resource_values["source_whole_model_learned_parameters"]
        - resource_values["native_removed_learned_parameters"]
        + resource_values["modal_graph_learned_parameters"]
    ):
        raise ValueError("guard resource accounting is inconsistent")

    if (
        not isinstance(flow, Mapping)
        or flow.get("assessment_read_only") is not True
        or flow.get("coefficients_fitted") is not False
        or flow.get("evaluation_kind")
        != "fisher_graph.state_conditioned_modal_flow_evaluation"
        or flow.get("source_free") is not True
        or flow.get("routed_graph_uses_source_state_only") is not True
        or flow.get("teacher_used_for_scoring_only") is not True
        or positive_int(
            flow.get("observations"),
            label="guard flow observations",
        )
        != routing_tokens
        or positive_int(
            flow.get("residual_width"),
            label="guard flow residual width",
        )
        <= 0
        or not isinstance(flow.get("families"), list)
        or len(flow["families"]) != int(guard["example_count"])
        or not isinstance(flow.get("interaction_artifact_sha256s"), list)
        or len(flow["interaction_artifact_sha256s"])
        != int(graph_comparison["interacting_edge_count"])
        or not all(
            isinstance(value, str) and _SHA256.fullmatch(value) is not None
            for value in flow["interaction_artifact_sha256s"]
        )
    ):
        raise ValueError("guard flow receipt is invalid")
    return raw, file_sha256


def _claim_payload(
    *,
    candidate: Mapping[str, str],
    guard_assessment_sha256: str,
    panel_file_sha256: str,
    evaluator_file_sha256: str,
) -> dict[str, object]:
    return {
        "schema": _CLAIM_SCHEMA,
        "format_version": 1,
        "candidate": dict(candidate),
        "guard_assessment_sha256": _require_sha256(
            guard_assessment_sha256,
            label="guard assessment",
        ),
        "panel_id": _PANEL_ID,
        "panel_file_sha256": panel_file_sha256,
        "evaluator_file_sha256": evaluator_file_sha256,
        "scoring_contract": {
            "conditions": ("native", "edgeless", "candidate"),
            "choice_count": _CHOICE_COUNT,
            "choice_tokenization": "one_leading_space_token_per_choice",
            "score": "restricted_next_token_log_softmax",
            "reduction": "item_then_equal_item_global_and_equal_item_family",
            "candidate_refit_or_search": False,
        },
        "thresholds": {
            "minimum_accuracy_retention": _MINIMUM_ACCURACY_RETENTION,
            "minimum_native_win_preservation": (
                _MINIMUM_NATIVE_WIN_PRESERVATION
            ),
            "minimum_native_win_wilson_lower": _MINIMUM_WILSON_LOWER,
            "maximum_accuracy_drop": _MAXIMUM_ACCURACY_DROP,
            "maximum_choice_nll_increase": _MAXIMUM_CHOICE_NLL_INCREASE,
            "minimum_native_correct": _MINIMUM_NATIVE_CORRECT,
            "minimum_qualified_families": _MINIMUM_QUALIFIED_FAMILIES,
            "minimum_qualified_family_fraction_with_at_most_one_loss": (
                _MINIMUM_QUALIFIED_FAMILY_FRACTION
            ),
            "maximum_losses_per_qualified_family": 2,
        },
    }


def _write_or_resume_claim(path: Path, payload: Mapping[str, object]) -> str:
    claim_sha256 = _domain_sha256(
        b"fisher-graph:gemma3-downstream-claim:v1\0",
        payload,
    )
    claim = {**dict(payload), "claim_sha256": claim_sha256}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical_json_bytes(existing) != _canonical_json_bytes(claim):
            raise FileExistsError("downstream panel is claimed by different bytes")
        return claim_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return claim_sha256


def _tokenize_panel(
    tokenizer: object,
    panel: ForcedChoicePanel,
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, str]:
    if not callable(tokenizer) or not hasattr(tokenizer, "encode"):
        raise TypeError("tokenizer lacks the required callable/encode interface")
    tokenized = tokenizer(
        [example.prompt for example in panel.examples],
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    if not isinstance(tokenized, Mapping):
        raise TypeError("tokenizer did not return a model-input mapping")
    model_inputs = {
        name: value.to(device=device)
        for name, value in tokenized.items()
        if isinstance(value, Tensor) and name in {"input_ids", "attention_mask"}
    }
    if set(model_inputs) != {"input_ids", "attention_mask"}:
        raise ValueError("tokenized panel lacks input_ids or attention_mask")
    choice_ids: list[list[int]] = []
    for example in panel.examples:
        current: list[int] = []
        for choice in example.choices:
            encoded = tokenizer.encode(
                " " + choice,
                add_special_tokens=False,
            )
            if not isinstance(encoded, list) or len(encoded) != 1:
                raise ValueError(
                    "every forced choice must be exactly one leading-space token"
                )
            current.append(int(encoded[0]))
        choice_ids.append(current)
    choices = torch.tensor(choice_ids, dtype=torch.long, device=device)
    stream_sha256 = _domain_sha256(
        b"fisher-graph:gemma3-downstream-tokenized-stream:v1\0",
        {
            "input_ids": model_inputs["input_ids"].detach().cpu().tolist(),
            "attention_mask": (
                model_inputs["attention_mask"].detach().cpu().tolist()
            ),
            "choice_ids": choices.detach().cpu().tolist(),
        },
    )
    return model_inputs, choices, stream_sha256


def _extract_logits(output: object) -> Tensor:
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, Mapping):
        value = output.get("logits")
        if isinstance(value, Tensor):
            return value
    raise TypeError("model output does not expose logits")


def _score_batches(
    *,
    name: ConditionName,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor | None,
    model_inputs: Mapping[str, Tensor],
    choice_ids: Tensor,
    gold: Tensor,
    batch_size: int,
) -> tuple[ForcedChoiceConditionResult, dict[str, object] | None]:
    predictions: list[int] = []
    nlls: list[float] = []
    margins: list[float] = []
    resources: dict[str, object] | None = None
    observations = int(choice_ids.shape[0])
    for start in range(0, observations, batch_size):
        stop = min(start + batch_size, observations)
        batch = {key: value[start:stop] for key, value in model_inputs.items()}
        with torch.no_grad():
            if executor is None:
                logits = adapter.forward(batch).logits
            else:
                execution = executor.run(batch, condition="generated")
                logits = _extract_logits(execution.model_output)
                if resources is None:
                    resources = _runtime_resource_summary(execution)
        mask = batch["attention_mask"].to(dtype=torch.bool)
        positions = torch.arange(
            mask.shape[1],
            device=mask.device,
        ).expand_as(mask)
        last = positions.masked_fill(~mask, -1).amax(dim=1)
        if bool((last < 0).any()):
            raise ValueError("forced-choice prompt cannot be empty")
        row = torch.arange(stop - start, device=logits.device)
        final_logits = logits[row, last.to(device=logits.device)]
        ids = choice_ids[start:stop].to(device=logits.device)
        scores = final_logits.gather(1, ids)
        restricted = torch.log_softmax(scores.to(dtype=torch.float64), dim=-1)
        answers = gold[start:stop].to(device=logits.device)
        predicted = scores.argmax(dim=-1)
        gold_values = scores.gather(1, answers[:, None]).squeeze(1)
        distractors = scores.clone()
        distractors.scatter_(1, answers[:, None], -torch.inf)
        gold_margin = gold_values - distractors.amax(dim=-1)
        predictions.extend(int(value) for value in predicted.cpu().tolist())
        nlls.extend(
            float(value)
            for value in (-restricted.gather(1, answers[:, None]).squeeze(1))
            .cpu()
            .tolist()
        )
        margins.extend(float(value) for value in gold_margin.cpu().tolist())
        _progress(f"{name}: scored {stop}/{observations}")
    return (
        ForcedChoiceConditionResult(
            name=name,
            predictions=tuple(predictions),
            restricted_choice_nll=tuple(nlls),
            gold_margins=tuple(margins),
        ),
        resources,
    )


def _runtime_resource_summary(
    execution: Gemma3ModalGeneratorGraphExecution,
) -> dict[str, object]:
    valid = execution.valid_tokens
    if valid <= 0:
        raise ValueError("runtime resource accounting has no valid tokens")
    return {
        "source_whole_model_learned_parameters": (
            execution.source_whole_model_learned_parameters
        ),
        "candidate_whole_model_learned_parameters": (
            execution.candidate_whole_model_learned_parameters
        ),
        "native_removed_learned_parameters": (
            execution.native_removed_learned_parameters
        ),
        "modal_graph_learned_parameters": execution.modal_graph_learned_parameters,
        "net_stored_parameter_savings": execution.net_stored_parameter_savings,
        "logical_native_removed_macs_per_token": (
            execution.logical_linear_macs_native_removed / valid
        ),
        "logical_modal_graph_macs_per_token": (
            execution.logical_modal_graph_macs / valid
        ),
        "logical_executed_modal_graph_macs_per_token": (
            execution.logical_executed_modal_graph_macs / valid
        ),
        "replacement_scope": execution.replacement_scope,
    }


def assess_gemma3_downstream_retention(
    *,
    candidate_path: Path | str = DEFAULT_GAIN_CANDIDATE_OUTPUT,
    output: Path | str = DEFAULT_DOWNSTREAM_OUTPUT,
    panel_path: Path | str = DEFAULT_DOWNSTREAM_PANEL,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 10,
) -> dict[str, object]:
    """Run the fixed no-refit downstream pilot exactly once per protocol."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite downstream assessment")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    panel_file = Path(panel_path)
    panel_bytes = panel_file.read_bytes()
    panel_file_sha256 = _sha256_bytes(panel_bytes)
    if panel_file_sha256 != _EXPECTED_PANEL_FILE_SHA256:
        raise ValueError("downstream panel bytes differ from the frozen protocol")

    raw = load_gemma3_state_conditioned_modal_graph_candidate(candidate_path)
    candidate = _candidate_metadata(raw)
    splits = raw.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("candidate split metadata is unavailable")
    guard_path = Path(candidate_path).with_suffix(".assessment.json")
    guard, guard_assessment_sha256 = _validate_guard_assessment(
        guard_path,
        candidate=candidate,
        expected_tensor_file=Path(candidate_path).name,
        expected_guard_manifest_sha256=str(
            splits["guard_role_manifest_sha256"]
        ),
    )
    evaluator_sha256 = _sha256_bytes(Path(__file__).read_bytes())
    claim_payload = _claim_payload(
        candidate=candidate,
        guard_assessment_sha256=guard_assessment_sha256,
        panel_file_sha256=panel_file_sha256,
        evaluator_file_sha256=evaluator_sha256,
    )
    claim_path = destination.with_suffix(".claim.json")
    claim_sha256 = _write_or_resume_claim(claim_path, claim_payload)
    _progress("claimed frozen candidate, panel, evaluator, and thresholds")

    panel = load_forced_choice_panel(panel_file)
    pipeline, edgeless_graph, dynamic_graph, lowerings = (
        restore_gemma3_state_conditioned_shape_flow_runtime(raw)
    )
    del pipeline
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("candidate experiment metadata is unavailable")
    model_id = experiment.get("model_id")
    revision = experiment.get("requested_revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("candidate model binding is invalid")

    corpus = _load_corpus(
        corpus_artifact_path=DEFAULT_CORPUS_ARTIFACT,
        corpus_fit_path=DEFAULT_CORPUS_FIT,
        corpus_selection_path=DEFAULT_CORPUS_SELECTION,
        corpus_guard_path=DEFAULT_CORPUS_GUARD,
    )
    prior_prompt_hashes = set()
    prior_family_ids = set()
    for role in (
        "calibration_a_fit",
        "calibration_a_selection",
        "calibration_a_guard",
    ):
        view = corpus.preclaim_view(role)  # type: ignore[arg-type]
        prior_prompt_hashes.update(view.ordered_prompt_sha256s)
        prior_family_ids.update(view.family_ids)
    task_prompt_hashes = {example.prompt_sha256 for example in panel.examples}
    if task_prompt_hashes & prior_prompt_hashes:
        raise ValueError("downstream panel overlaps a candidate development prompt")

    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        raise ValueError("requested downstream device is unavailable")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("loading pinned local Gemma after the panel claim")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    model_fingerprint = adapter.model_fingerprint()
    if model_fingerprint != experiment.get("adapter_model_fingerprint"):
        raise ValueError("downstream model fingerprint differs from candidate")
    model_inputs, choice_ids, stream_sha256 = _tokenize_panel(
        tokenizer,
        panel,
        device=device,
    )
    gold = torch.tensor(
        [example.correct_choice for example in panel.examples],
        dtype=torch.long,
        device=device,
    )
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless_graph,
        tuple(lowerings[name] for name in edgeless_graph.traversal_order),
    )
    candidate_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        dynamic_graph,
        tuple(lowerings[name] for name in dynamic_graph.traversal_order),
    )
    native, _ = _score_batches(
        name="native",
        adapter=adapter,
        executor=None,
        model_inputs=model_inputs,
        choice_ids=choice_ids,
        gold=gold,
        batch_size=batch_size,
    )
    edgeless, edgeless_resources = _score_batches(
        name="edgeless",
        adapter=adapter,
        executor=edgeless_executor,
        model_inputs=model_inputs,
        choice_ids=choice_ids,
        gold=gold,
        batch_size=batch_size,
    )
    candidate_result, candidate_resources = _score_batches(
        name="candidate",
        adapter=adapter,
        executor=candidate_executor,
        model_inputs=model_inputs,
        choice_ids=choice_ids,
        gold=gold,
        batch_size=batch_size,
    )
    evaluation = evaluate_forced_choice_retention(
        panel,
        native=native,
        edgeless=edgeless,
        candidate=candidate_result,
    )
    report: dict[str, object] = {
        "schema": _REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "role": "post_candidate_frozen_downstream_retention_pilot",
            "candidate_refit_or_search": False,
            "task_suite_used_for_candidate_selection": False,
            "externally_standardized_benchmark": False,
            "fresh_validation": False,
            "test_data_used": False,
            "claim_written_before_prompt_materialization": True,
        },
        "candidate": candidate,
        "prior_guard": {
            "assessment_file": guard_path.name,
            "assessment_file_sha256": guard_assessment_sha256,
            "guard_nll_improvement_over_edgeless": guard[
                "guard_nll_improvement_over_edgeless"
            ],
        },
        "panel": {
            "panel_id": panel.panel_id,
            "panel_file_sha256": panel.file_sha256,
            "panel_semantic_sha256": panel.semantic_sha256,
            "tokenized_stream_sha256": stream_sha256,
            "claim_sha256": claim_sha256,
            "evaluator_file_sha256": evaluator_sha256,
            "example_count": len(panel.examples),
            "choice_count": _CHOICE_COUNT,
            "family_ids": panel.family_ids,
            "examples_per_family": _EXAMPLES_PER_FAMILY,
            "candidate_development_prompt_overlap_count": len(
                task_prompt_hashes & prior_prompt_hashes
            ),
            "candidate_development_family_label_overlap_count": len(
                set(panel.family_ids) & prior_family_ids
            ),
            "contains_prompt_text": False,
            "contains_choice_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
        },
        "evaluation": evaluation,
        "resources": {
            "edgeless": edgeless_resources,
            "candidate": candidate_resources,
        },
        "source_model_unchanged": adapter.model_fingerprint()
        == model_fingerprint,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _progress(
        f"wrote {destination}; status={evaluation['status']}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Gemma downstream-retention pilot.",
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_GAIN_CANDIDATE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DOWNSTREAM_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = assess_gemma3_downstream_retention(
        candidate_path=arguments.candidate,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    evaluation = result["evaluation"]
    assert isinstance(evaluation, Mapping)
    print(json.dumps({"status": evaluation["status"], "passed": evaluation["passed"]}))
    return 0 if evaluation["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
