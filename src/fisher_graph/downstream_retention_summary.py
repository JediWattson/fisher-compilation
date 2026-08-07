"""Source-safe projection for a local downstream-retention assessment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re


_ASSESSMENT_SCHEMA = (
    "fisher_graph.gemma3_native_qualified_downstream_retention_assessment"
)
_SUMMARY_SCHEMA = (
    "fisher_graph.state_conditioned_downstream_retention_summary"
)
_AMBIGUOUS_FAMILIES = frozenset(
    {"animal_movement", "object_material", "country_continent"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_CANDIDATE = {
    "scientific_payload_sha256": (
        "0897a54a5a52e544db67f05164313980e02ccb4e54fa5b4b00a49064b0478fd7"
    ),
    "dynamic_graph_sha256": (
        "11bb160721528c2a7ea326add60be5277729cda7ac3bf949d8f08adfce2bcbbf"
    ),
    "compiler_pipeline_sha256": (
        "36c33b40eef3a32c3c39900c4d3ab2a79ef5a329641c895c5d1ea0b2d60f6efb"
    ),
    "interaction_promotion_sha256": (
        "172e8dd5ef167f6d8df09f4f5d869ef43010b9c480094de37ebb1370927d17f8"
    ),
}
_EXPECTED_GATE_NAMES = frozenset(
    {
        "absolute_accuracy_drop_at_most_0_05",
        "at_least_80pct_qualified_families_lose_at_most_one",
        "candidate_choice_nll_minus_native_at_most_0_10",
        "global_accuracy_retention_at_least_0_90",
        "global_native_win_preservation_at_least_0_90",
        "native_win_preservation_one_sided_90pct_wilson_lower_at_least_0_80",
        "no_qualified_family_loses_more_than_two",
        "suite_adequate",
    }
)
_SUMMARY_HASH_DOMAIN = b"fisher-graph:downstream-retention-summary:v2\0"
_WILSON_Z_ONE_SIDED_90 = 1.2815515655446004


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} fields are invalid")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _require_close(
    observed: object,
    expected: float,
    *,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    value = _require_finite(observed, label=label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} is inconsistent")


def _wilson_lower(successes: int, observations: int) -> float:
    if observations <= 0:
        raise ValueError("Wilson observations must be positive")
    probability = successes / observations
    z2 = _WILSON_Z_ONE_SIDED_90 * _WILSON_Z_ONE_SIDED_90
    denominator = 1.0 + z2 / observations
    center = probability + z2 / (2.0 * observations)
    spread = _WILSON_Z_ONE_SIDED_90 * math.sqrt(
        probability * (1.0 - probability) / observations
        + z2 / (4.0 * observations * observations)
    )
    return (center - spread) / denominator


def build_source_safe_downstream_retention_summary(
    assessment: Mapping[str, object],
    *,
    assessment_file_sha256: str,
    as_of: str,
) -> dict[str, object]:
    """Project one authenticated local assessment without private task rows."""

    if (
        assessment.get("schema") != _ASSESSMENT_SCHEMA
        or assessment.get("format_version") != 1
    ):
        raise ValueError("downstream assessment schema is invalid")
    _exact_keys(
        assessment,
        {
            "bank",
            "candidate",
            "evaluation",
            "format_version",
            "model",
            "prior_guard",
            "qualification",
            "resources",
            "schema",
            "scientific_status",
            "source_model_unchanged",
        },
        label="assessment",
    )
    _require_sha256(
        assessment_file_sha256,
        label="assessment file",
    )
    if not isinstance(as_of, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of) is None:
        raise ValueError("as_of must use YYYY-MM-DD")
    status = _mapping(assessment.get("scientific_status"), label="status")
    _exact_keys(
        status,
        {
            "candidate_executed_during_family_qualification",
            "candidate_refit_or_search",
            "externally_standardized_benchmark",
            "fresh_validation",
            "prior_candidate_diagnostic_informed_panel_redesign",
            "role",
            "status",
            "task_suite_used_for_candidate_selection",
            "test_data_used",
        },
        label="scientific status",
    )
    if (
        status.get("status") != "downstream_retention_pilot_pass"
        or status.get("role")
        != "native_qualified_downstream_retention_diagnostic"
        or status.get("candidate_executed_during_family_qualification")
        is not False
        or status.get("candidate_refit_or_search") is not False
        or status.get("task_suite_used_for_candidate_selection") is not False
        or status.get("prior_candidate_diagnostic_informed_panel_redesign")
        is not True
        or status.get("externally_standardized_benchmark") is not False
        or status.get("fresh_validation") is not False
        or status.get("test_data_used") is not False
    ):
        raise ValueError("only a passed assessment can be published")
    model = _mapping(assessment.get("model"), label="model")
    _exact_keys(
        model,
        {"adapter_model_fingerprint", "model_id", "revision"},
        label="model",
    )
    if model.get("model_id") != "google/gemma-3-270m":
        raise ValueError("assessment model is not the frozen Gemma target")
    if (
        not isinstance(model.get("revision"), str)
        or _REVISION.fullmatch(str(model["revision"])) is None
    ):
        raise ValueError("model revision must be a 40-character Git commit")
    _require_sha256(
        model.get("adapter_model_fingerprint"),
        label="adapter model fingerprint",
    )
    candidate = _mapping(assessment.get("candidate"), label="candidate")
    _exact_keys(candidate, set(_EXPECTED_CANDIDATE), label="candidate")
    if dict(candidate) != _EXPECTED_CANDIDATE:
        raise ValueError("assessment does not bind the frozen 0.5x candidate")
    guard = _mapping(assessment.get("prior_guard"), label="prior guard")
    _exact_keys(
        guard,
        {
            "assessment_file",
            "assessment_file_sha256",
            "guard_nll_improvement_over_edgeless",
        },
        label="prior guard",
    )
    if guard.get("assessment_file") != (
        "state-conditioned-shape-flow-gain-dev-v2.assessment.json"
    ):
        raise ValueError("prior guard assessment identity drifted")
    _require_sha256(
        guard.get("assessment_file_sha256"),
        label="prior guard assessment file",
    )
    guard_improvement = _require_finite(
        guard.get("guard_nll_improvement_over_edgeless"),
        label="prior guard improvement",
    )
    if guard_improvement <= 0.0:
        raise ValueError("prior guard did not beat its edgeless control")
    bank = _mapping(assessment.get("bank"), label="bank")
    _exact_keys(
        bank,
        {
            "bank_file_sha256",
            "bank_id",
            "candidate_development_prompt_overlap_count",
            "claim_sha256",
            "contains_prompt_or_choice_text",
            "contains_token_ids_or_logits",
            "evaluation_stream_sha256",
            "evaluator_file_sha256",
            "qualification_stream_sha256",
            "shared_evaluator_file_sha256",
        },
        label="bank",
    )
    if (
        bank.get("bank_id") != "gemma3-native-qualified-forced-choice-v2"
        or bank.get("contains_prompt_or_choice_text") is not False
        or bank.get("contains_token_ids_or_logits") is not False
        or bank.get("candidate_development_prompt_overlap_count") != 0
    ):
        raise ValueError("downstream bank identity or source-safety drifted")
    for name in (
        "bank_file_sha256",
        "claim_sha256",
        "evaluation_stream_sha256",
        "evaluator_file_sha256",
        "qualification_stream_sha256",
        "shared_evaluator_file_sha256",
    ):
        _require_sha256(bank.get(name), label=f"bank {name}")
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    current_sources = {
        "evaluator_file_sha256": package_root
        / "gemma3_downstream_retention_v2_experiment.py",
        "shared_evaluator_file_sha256": package_root
        / "gemma3_downstream_retention_experiment.py",
        "bank_file_sha256": repository_root
        / "examples"
        / "gemma3_downstream_qualification_v2.json",
    }
    for receipt_name, source_path in current_sources.items():
        if (
            not source_path.is_file()
            or _sha256(source_path.read_bytes()) != bank[receipt_name]
        ):
            raise ValueError(
                f"current source bytes do not match {receipt_name}"
            )
    qualification = _mapping(
        assessment.get("qualification"),
        label="qualification",
    )
    _exact_keys(
        qualification,
        {
            "eligible_family_ids",
            "family_native_correct_counts",
            "minimum_native_correct_per_family",
            "selected_family_ids",
            "sufficient_eligible_families",
        },
        label="qualification",
    )
    selected = qualification.get("selected_family_ids")
    eligible = qualification.get("eligible_family_ids")
    family_native_counts = qualification.get("family_native_correct_counts")
    minimum_family_correct = _require_int(
        qualification.get("minimum_native_correct_per_family"),
        label="qualification minimum native correct",
        minimum=1,
    )
    if (
        not isinstance(selected, list)
        or len(selected) != 6
        or len(set(selected)) != 6
        or not all(isinstance(name, str) and name for name in selected)
        or not isinstance(eligible, list)
        or not all(isinstance(name, str) and name for name in eligible)
        or len(eligible) != len(set(eligible))
        or tuple(selected) != tuple(eligible[:6])
        or qualification.get("sufficient_eligible_families") is not True
        or not isinstance(family_native_counts, Mapping)
        or len(family_native_counts) != 8
    ):
        raise ValueError("selected family receipt is invalid")
    derived_eligible = [
        name
        for name, count in family_native_counts.items()
        if _require_int(
            count,
            label=f"qualification count for {name}",
        )
        >= minimum_family_correct
    ]
    if set(eligible) != set(derived_eligible):
        raise ValueError("eligible family receipt is inconsistent")

    evaluation = _mapping(assessment.get("evaluation"), label="evaluation")
    _exact_keys(
        evaluation,
        {
            "adequacy",
            "conditional_edge_value_added",
            "conditions",
            "family_comparisons",
            "gates",
            "paired_candidate_vs_native",
            "passed",
            "status",
        },
        label="evaluation",
    )
    conditions = _mapping(evaluation.get("conditions"), label="conditions")
    _exact_keys(conditions, {"candidate", "edgeless", "native"}, label="conditions")
    native = _mapping(conditions.get("native"), label="native")
    edgeless = _mapping(conditions.get("edgeless"), label="edgeless")
    compiled = _mapping(conditions.get("candidate"), label="candidate result")
    condition_keys = {
        "accuracy",
        "correct_count",
        "example_count",
        "families",
        "mean_gold_margin",
        "prediction_stream_sha256",
        "restricted_choice_nll",
    }
    condition_family_keys = {
        "accuracy",
        "correct_count",
        "example_count",
        "mean_gold_margin",
        "restricted_choice_nll",
    }
    for name, condition in (
        ("native", native),
        ("edgeless", edgeless),
        ("candidate", compiled),
    ):
        _exact_keys(condition, condition_keys, label=f"{name} condition")
        count = _require_int(
            condition.get("correct_count"),
            label=f"{name} correct count",
        )
        examples = _require_int(
            condition.get("example_count"),
            label=f"{name} example count",
            minimum=1,
        )
        if examples != 60 or count > examples:
            raise ValueError(f"{name} condition count is invalid")
        _require_close(
            condition.get("accuracy"),
            count / examples,
            label=f"{name} accuracy",
        )
        _require_finite(
            condition.get("mean_gold_margin"),
            label=f"{name} mean gold margin",
        )
        nll = _require_finite(
            condition.get("restricted_choice_nll"),
            label=f"{name} restricted choice NLL",
        )
        if nll < 0.0:
            raise ValueError(f"{name} restricted choice NLL is negative")
        _require_sha256(
            condition.get("prediction_stream_sha256"),
            label=f"{name} prediction stream",
        )
        family_results = _mapping(
            condition.get("families"),
            label=f"{name} family results",
        )
        if set(family_results) != set(selected):
            raise ValueError(f"{name} family coverage drifted")
        family_correct_total = 0
        family_example_total = 0
        for family_name, raw_family in family_results.items():
            family = _mapping(
                raw_family,
                label=f"{name} family {family_name}",
            )
            _exact_keys(
                family,
                condition_family_keys,
                label=f"{name} family {family_name}",
            )
            family_correct = _require_int(
                family.get("correct_count"),
                label=f"{name} family {family_name} correct",
            )
            family_examples = _require_int(
                family.get("example_count"),
                label=f"{name} family {family_name} examples",
                minimum=1,
            )
            if family_examples != 10 or family_correct > family_examples:
                raise ValueError(f"{name} family accounting is invalid")
            _require_close(
                family.get("accuracy"),
                family_correct / family_examples,
                label=f"{name} family {family_name} accuracy",
            )
            _require_finite(
                family.get("mean_gold_margin"),
                label=f"{name} family {family_name} margin",
            )
            if _require_finite(
                family.get("restricted_choice_nll"),
                label=f"{name} family {family_name} NLL",
            ) < 0.0:
                raise ValueError(f"{name} family NLL is negative")
            family_correct_total += family_correct
            family_example_total += family_examples
        if family_correct_total != count or family_example_total != examples:
            raise ValueError(f"{name} family totals are inconsistent")

    paired = _mapping(
        evaluation.get("paired_candidate_vs_native"),
        label="paired result",
    )
    _exact_keys(
        paired,
        {
            "accuracy_delta",
            "accuracy_ratio_to_native",
            "accuracy_retained_fraction",
            "both_correct_count",
            "both_wrong_count",
            "candidate_correct_count",
            "candidate_only_correct_count",
            "native_correct_count",
            "native_only_correct_count",
            "native_win_preservation",
            "native_win_preservation_one_sided_90pct_wilson_lower",
            "prediction_agreement",
            "restricted_choice_nll_delta",
            "restricted_choice_perplexity_multiplier",
        },
        label="paired result",
    )
    native_correct = _require_int(
        native["correct_count"],
        label="native correct",
        minimum=1,
    )
    candidate_correct = _require_int(
        compiled["correct_count"],
        label="candidate correct",
    )
    both_correct = _require_int(
        paired.get("both_correct_count"),
        label="both correct",
    )
    native_only = _require_int(
        paired.get("native_only_correct_count"),
        label="native only correct",
    )
    candidate_only = _require_int(
        paired.get("candidate_only_correct_count"),
        label="candidate only correct",
    )
    both_wrong = _require_int(
        paired.get("both_wrong_count"),
        label="both wrong",
    )
    if (
        native_correct != both_correct + native_only
        or candidate_correct != both_correct + candidate_only
        or 60 != both_correct + native_only + candidate_only + both_wrong
        or paired.get("native_correct_count") != native_correct
        or paired.get("candidate_correct_count") != candidate_correct
    ):
        raise ValueError("paired correctness counts are inconsistent")
    accuracy_ratio = candidate_correct / native_correct
    native_win_preservation = both_correct / native_correct
    wilson_lower = _wilson_lower(both_correct, native_correct)
    candidate_nll = float(compiled["restricted_choice_nll"])
    native_nll = float(native["restricted_choice_nll"])
    edgeless_nll = float(edgeless["restricted_choice_nll"])
    _require_close(
        paired.get("accuracy_ratio_to_native"),
        accuracy_ratio,
        label="accuracy ratio",
    )
    _require_close(
        paired.get("accuracy_retained_fraction"),
        min(1.0, accuracy_ratio),
        label="accuracy retained fraction",
    )
    _require_close(
        paired.get("accuracy_delta"),
        (candidate_correct - native_correct) / 60,
        label="accuracy delta",
    )
    _require_close(
        paired.get("native_win_preservation"),
        native_win_preservation,
        label="native win preservation",
    )
    _require_close(
        paired.get("native_win_preservation_one_sided_90pct_wilson_lower"),
        wilson_lower,
        label="Wilson lower",
    )
    _require_close(
        paired.get("restricted_choice_nll_delta"),
        candidate_nll - native_nll,
        label="restricted choice NLL delta",
    )
    _require_close(
        paired.get("restricted_choice_perplexity_multiplier"),
        math.exp(candidate_nll - native_nll),
        label="restricted choice perplexity multiplier",
    )
    prediction_agreement = _require_finite(
        paired.get("prediction_agreement"),
        label="prediction agreement",
    )
    if not 0.0 <= prediction_agreement <= 1.0:
        raise ValueError("prediction agreement is outside [0, 1]")
    prediction_disagreements = (1.0 - prediction_agreement) * 60
    if not math.isclose(
        prediction_disagreements,
        round(prediction_disagreements),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("prediction agreement does not map to whole rows")

    edge = _mapping(
        evaluation.get("conditional_edge_value_added"),
        label="edge value",
    )
    _exact_keys(
        edge,
        {
            "accuracy_delta_over_edgeless",
            "candidate_accuracy_not_below_edgeless",
            "candidate_choice_nll_below_edgeless",
            "edgeless_correct_candidate_wrong_count",
            "edgeless_wrong_candidate_correct_count",
            "restricted_choice_nll_improvement_over_edgeless",
        },
        label="edge value",
    )
    edgeless_correct = _require_int(
        edgeless["correct_count"],
        label="edgeless correct",
    )
    _require_close(
        edge.get("accuracy_delta_over_edgeless"),
        (candidate_correct - edgeless_correct) / 60,
        label="edge accuracy delta",
    )
    edge_nll_improvement = edgeless_nll - candidate_nll
    _require_close(
        edge.get("restricted_choice_nll_improvement_over_edgeless"),
        edge_nll_improvement,
        label="edge NLL improvement",
    )
    if (
        edge.get("candidate_accuracy_not_below_edgeless")
        is not (candidate_correct >= edgeless_correct)
        or edge.get("candidate_choice_nll_below_edgeless")
        is not (candidate_nll < edgeless_nll)
    ):
        raise ValueError("edge value booleans are inconsistent")
    _require_int(
        edge.get("edgeless_correct_candidate_wrong_count"),
        label="edgeless-only correct",
    )
    _require_int(
        edge.get("edgeless_wrong_candidate_correct_count"),
        label="candidate-only versus edgeless",
    )

    families = _mapping(
        evaluation.get("family_comparisons"),
        label="family comparisons",
    )
    if set(families) != set(selected):
        raise ValueError("paired family coverage drifted")
    family_keys = {
        "accuracy_retained_fraction",
        "both_correct_count",
        "candidate_correct_count",
        "eligibility_source",
        "example_count",
        "native_correct_count",
        "native_only_correct_count",
        "native_win_preservation",
        "primary_eligible",
    }
    native_losses_by_family: list[int] = []
    family_native_total = 0
    family_candidate_total = 0
    family_both_total = 0
    for name, raw_family in families.items():
        family = _mapping(raw_family, label=f"paired family {name}")
        _exact_keys(family, family_keys, label=f"paired family {name}")
        family_native = _require_int(
            family.get("native_correct_count"),
            label=f"paired family {name} native correct",
            minimum=1,
        )
        family_candidate = _require_int(
            family.get("candidate_correct_count"),
            label=f"paired family {name} candidate correct",
        )
        family_both = _require_int(
            family.get("both_correct_count"),
            label=f"paired family {name} both correct",
        )
        family_loss = _require_int(
            family.get("native_only_correct_count"),
            label=f"paired family {name} native only",
        )
        if (
            family.get("example_count") != 10
            or family.get("primary_eligible") is not True
            or family.get("eligibility_source")
            != "separate_native_only_qualification_split"
            or family_native != family_both + family_loss
        ):
            raise ValueError(f"paired family {name} receipt is invalid")
        _require_close(
            family.get("accuracy_retained_fraction"),
            min(1.0, family_candidate / family_native),
            label=f"paired family {name} retention",
        )
        _require_close(
            family.get("native_win_preservation"),
            family_both / family_native,
            label=f"paired family {name} preservation",
        )
        native_losses_by_family.append(family_loss)
        family_native_total += family_native
        family_candidate_total += family_candidate
        family_both_total += family_both
    if (
        family_native_total != native_correct
        or family_candidate_total != candidate_correct
        or family_both_total != both_correct
    ):
        raise ValueError("paired family totals are inconsistent")

    adequacy = _mapping(evaluation.get("adequacy"), label="adequacy")
    _exact_keys(
        adequacy,
        {
            "family_eligibility_source",
            "minimum_native_correct",
            "minimum_qualified_families",
            "observed_native_correct",
            "observed_qualified_families",
            "qualified_family_ids",
        },
        label="adequacy",
    )
    if (
        adequacy.get("family_eligibility_source")
        != "separate_native_only_qualification_split"
        or adequacy.get("observed_native_correct") != native_correct
        or adequacy.get("observed_qualified_families") != len(selected)
        or set(adequacy.get("qualified_family_ids", [])) != set(selected)
    ):
        raise ValueError("adequacy receipt is inconsistent")
    minimum_native_correct = _require_int(
        adequacy.get("minimum_native_correct"),
        label="minimum native correct",
        minimum=1,
    )
    minimum_qualified_families = _require_int(
        adequacy.get("minimum_qualified_families"),
        label="minimum qualified families",
        minimum=1,
    )
    suite_adequate = (
        native_correct >= minimum_native_correct
        and len(selected) >= minimum_qualified_families
    )
    computed_gates = {
        "absolute_accuracy_drop_at_most_0_05": (
            (native_correct - candidate_correct) / 60 <= 0.05
        ),
        "at_least_80pct_qualified_families_lose_at_most_one": (
            sum(loss <= 1 for loss in native_losses_by_family)
            >= math.ceil(0.80 * len(selected))
        ),
        "candidate_choice_nll_minus_native_at_most_0_10": (
            candidate_nll - native_nll <= 0.10
        ),
        "global_accuracy_retention_at_least_0_90": (
            candidate_correct >= math.ceil(0.90 * native_correct)
        ),
        "global_native_win_preservation_at_least_0_90": (
            both_correct >= math.ceil(0.90 * native_correct)
        ),
        "native_win_preservation_one_sided_90pct_wilson_lower_at_least_0_80": (
            wilson_lower >= 0.80
        ),
        "no_qualified_family_loses_more_than_two": all(
            loss <= 2 for loss in native_losses_by_family
        ),
        "suite_adequate": suite_adequate,
    }
    gates = _mapping(evaluation.get("gates"), label="gates")
    if (
        set(gates) != _EXPECTED_GATE_NAMES
        or any(
            gates.get(name) is not expected
            for name, expected in computed_gates.items()
        )
        or not all(computed_gates.values())
        or evaluation.get("passed") is not True
        or evaluation.get("status") != "downstream_retention_pilot_pass"
    ):
        raise ValueError("downstream assessment gates are not all passed")

    resources = _mapping(assessment.get("resources"), label="resources")
    _exact_keys(resources, {"candidate", "edgeless"}, label="resources")
    compiled_resources = _mapping(
        resources.get("candidate"),
        label="compiled resources",
    )
    resource_keys = {
        "candidate_whole_model_learned_parameters",
        "logical_executed_modal_graph_macs_per_token",
        "logical_modal_graph_macs_per_token",
        "logical_native_removed_macs_per_token",
        "modal_graph_learned_parameters",
        "native_removed_learned_parameters",
        "net_stored_parameter_savings",
        "replacement_scope",
        "source_whole_model_learned_parameters",
    }
    _exact_keys(compiled_resources, resource_keys, label="compiled resources")
    if compiled_resources.get("replacement_scope") != (
        "partial_native_mlp_mode_replacement"
    ):
        raise ValueError("compiled replacement scope drifted")
    native_removed = _require_int(
        compiled_resources.get("native_removed_learned_parameters"),
        label="native removed parameters",
        minimum=1,
    )
    graph_parameters = _require_int(
        compiled_resources.get("modal_graph_learned_parameters"),
        label="modal graph parameters",
        minimum=1,
    )
    net_savings = _require_int(
        compiled_resources.get("net_stored_parameter_savings"),
        label="net parameter savings",
        minimum=1,
    )
    source_parameters = _require_int(
        compiled_resources.get("source_whole_model_learned_parameters"),
        label="source whole-model parameters",
        minimum=1,
    )
    candidate_parameters = _require_int(
        compiled_resources.get("candidate_whole_model_learned_parameters"),
        label="candidate whole-model parameters",
        minimum=1,
    )
    native_macs = _require_finite(
        compiled_resources.get("logical_native_removed_macs_per_token"),
        label="native removed MACs",
    )
    dense_macs = _require_finite(
        compiled_resources.get("logical_modal_graph_macs_per_token"),
        label="dense graph MACs",
    )
    executed_macs = _require_finite(
        compiled_resources.get("logical_executed_modal_graph_macs_per_token"),
        label="executed graph MACs",
    )
    if (
        native_removed - graph_parameters != net_savings
        or source_parameters - net_savings != candidate_parameters
        or not native_macs >= dense_macs >= executed_macs >= 0.0
    ):
        raise ValueError("compiled resource accounting is inconsistent")
    if assessment.get("source_model_unchanged") is not True:
        raise ValueError("source model changed during assessment")

    ambiguous_selected = [
        name for name in selected if name in _AMBIGUOUS_FAMILIES
    ]
    paired_swap_families = [
        name
        for name, value in sorted(families.items())
        if isinstance(value, Mapping)
        and int(value["native_only_correct_count"]) > 0
        and int(value["candidate_correct_count"])
        - int(value["both_correct_count"])
        > 0
    ]
    summary: dict[str, object] = {
        "schema": _SUMMARY_SCHEMA,
        "format_version": 2,
        "as_of": as_of,
        "status": (
            "declared_label_retention_pilot_pass_with_label_ambiguity"
            if ambiguous_selected
            else "declared_label_retention_pilot_pass"
        ),
        "model": dict(model),
        "candidate": {
            **dict(candidate),
            "chosen_signed_gain": 0.5,
            "candidate_refit_or_search_during_task_evaluation": False,
        },
        "protocol": {
            "bank_id": bank["bank_id"],
            "bank_file_sha256": bank["bank_file_sha256"],
            "qualification_family_count": 8,
            "qualification_examples_per_family": 5,
            "minimum_native_correct_per_family": qualification[
                "minimum_native_correct_per_family"
            ],
            "selected_family_count": len(selected),
            "evaluation_examples_per_selected_family": 10,
            "evaluation_example_count": native["example_count"],
            "choice_count": 4,
            "choice_scoring": "restricted_single_next_token_log_softmax",
            "candidate_executed_during_family_qualification": False,
            "selected_family_ids": selected,
            "eligible_family_count": len(qualification["eligible_family_ids"]),
            "candidate_development_prompt_overlap_count": bank[
                "candidate_development_prompt_overlap_count"
            ],
        },
        "declared_label_results": {
            "native_correct": native["correct_count"],
            "native_accuracy": native["accuracy"],
            "edgeless_correct": edgeless["correct_count"],
            "edgeless_accuracy": edgeless["accuracy"],
            "candidate_correct": compiled["correct_count"],
            "candidate_accuracy": compiled["accuracy"],
            "candidate_accuracy_retained_fraction": paired[
                "accuracy_retained_fraction"
            ],
            "native_and_candidate_both_correct": paired["both_correct_count"],
            "native_only_correct": paired["native_only_correct_count"],
            "candidate_only_correct": paired["candidate_only_correct_count"],
            "native_win_preservation": paired["native_win_preservation"],
            "descriptive_item_level_wilson_lower": paired[
                "native_win_preservation_one_sided_90pct_wilson_lower"
            ],
            "candidate_prediction_agreement_with_native": paired[
                "prediction_agreement"
            ],
            "candidate_prediction_disagreement_count": round(
                prediction_disagreements
            ),
            "all_predeclared_retention_gates_passed": True,
        },
        "restricted_choice_likelihood": {
            "native_nll": native["restricted_choice_nll"],
            "edgeless_nll": edgeless["restricted_choice_nll"],
            "candidate_nll": compiled["restricted_choice_nll"],
            "candidate_minus_native_nll": paired[
                "restricted_choice_nll_delta"
            ],
            "candidate_minus_edgeless_nll": -float(
                edge["restricted_choice_nll_improvement_over_edgeless"]
            ),
            "candidate_choice_perplexity_multiplier_vs_native": paired[
                "restricted_choice_perplexity_multiplier"
            ],
            "conditional_edge_improves_edgeless_choice_nll": edge[
                "candidate_choice_nll_below_edgeless"
            ],
            "conditional_edge_changes_accuracy_vs_edgeless": (
                float(edge["accuracy_delta_over_edgeless"]) != 0.0
            ),
            "edgeless_correct_candidate_wrong_count": edge[
                "edgeless_correct_candidate_wrong_count"
            ],
            "edgeless_wrong_candidate_correct_count": edge[
                "edgeless_wrong_candidate_correct_count"
            ],
        },
        "resource_accounting": {
            "native_replaced_slice_parameters": compiled_resources[
                "native_removed_learned_parameters"
            ],
            "compiled_graph_parameters": compiled_resources[
                "modal_graph_learned_parameters"
            ],
            "net_replaced_slice_parameter_savings": compiled_resources[
                "net_stored_parameter_savings"
            ],
            "replaced_slice_parameter_savings_fraction": (
                net_savings / native_removed
            ),
            "source_whole_model_parameters": compiled_resources[
                "source_whole_model_learned_parameters"
            ],
            "candidate_whole_model_parameters": compiled_resources[
                "candidate_whole_model_learned_parameters"
            ],
            "whole_model_parameter_savings_fraction": (
                net_savings / source_parameters
            ),
            "native_replaced_slice_matrix_macs_per_token": compiled_resources[
                "logical_native_removed_macs_per_token"
            ],
            "compiled_dense_matrix_macs_per_token": compiled_resources[
                "logical_modal_graph_macs_per_token"
            ],
            "compiled_executed_matrix_macs_per_token": compiled_resources[
                "logical_executed_modal_graph_macs_per_token"
            ],
            "executed_matrix_mac_savings_fraction": (
                1.0 - executed_macs / native_macs
            ),
            "latency_or_kernel_speed_claim": False,
        },
        "label_validity_audit": {
            "label_audit_passed": False,
            "ambiguous_selected_family_ids": ambiguous_selected,
            "paired_loss_and_repair_family_ids": paired_swap_families,
            "sole_observed_loss_repair_pair_depends_on_ambiguous_family": (
                paired_swap_families == ["object_material"]
            ),
            "population_confidence_claim_from_wilson_bound": False,
            "wilson_scope": (
                "descriptive_item_level_screen_only_rows_are_templated_and_"
                "family_clustered"
            ),
        },
        "integrity": {
            "assessment_file_sha256": assessment_file_sha256,
            "downstream_claim_sha256": bank["claim_sha256"],
            "evaluator_file_sha256": bank["evaluator_file_sha256"],
            "shared_evaluator_file_sha256": bank[
                "shared_evaluator_file_sha256"
            ],
            "guard_assessment_file_sha256": guard[
                "assessment_file_sha256"
            ],
            "qualification_stream_sha256": bank[
                "qualification_stream_sha256"
            ],
            "evaluation_stream_sha256": bank["evaluation_stream_sha256"],
            "prediction_stream_sha256s": {
                name: value["prediction_stream_sha256"]
                for name, value in (
                    ("native", native),
                    ("edgeless", edgeless),
                    ("candidate", compiled),
                )
            },
            "source_model_unchanged": assessment["source_model_unchanged"],
            "exact_evaluator_source_available_in_checkout": True,
            "contains_prompt_or_choice_text": False,
            "contains_token_ids_or_logits": False,
            "contains_activations_gradients_or_weights": False,
            "source_safe": True,
        },
        "scientific_boundary": {
            "audit_replay_after_evaluator_hardening": True,
            "post_candidate_frozen_diagnostic": True,
            "prior_candidate_diagnostic_informed_panel_redesign": True,
            "externally_standardized_benchmark": False,
            "fresh_validation": False,
            "test_data_used": False,
            "whole_model_compression_proven": False,
        },
        "interpretation": (
            "The frozen half-strength layer-17 graph passed every declared "
            f"retention gate and matched native {candidate_correct}-of-60 "
            "declared-label accuracy while slightly improving "
            "restricted-choice NLL over native and edgeless. An audit found "
            "multiple-valid-answer "
            "templates, including the family containing the sole loss and "
            "repair, so this supports a viable lossy-compression pilot but "
            "not external downstream qualification."
        ),
        "next_rung": (
            "external_standardized_task_subset_then_standalone_same_topology_"
            "layer_10_candidate_with_fresh_guard_families"
        ),
    }
    summary["summary_payload_sha256"] = hashlib.sha256(
        _SUMMARY_HASH_DOMAIN + _canonical_json_bytes(summary)
    ).hexdigest()
    return summary


def load_and_build_source_safe_summary(
    path: Path | str,
    *,
    as_of: str,
) -> dict[str, object]:
    source = Path(path)
    encoded = source.read_bytes()
    def reject_nonfinite_constant(value: str) -> object:
        raise ValueError(f"downstream assessment contains non-finite {value}")

    raw = json.loads(
        encoded.decode("utf-8"),
        parse_constant=reject_nonfinite_constant,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("downstream assessment must contain one JSON object")
    return build_source_safe_downstream_retention_summary(
        raw,
        assessment_file_sha256=_sha256(encoded),
        as_of=as_of,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a source-safe downstream-retention summary.",
    )
    parser.add_argument("assessment", type=Path)
    parser.add_argument("--as-of", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = load_and_build_source_safe_summary(
        arguments.assessment,
        as_of=arguments.as_of,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
