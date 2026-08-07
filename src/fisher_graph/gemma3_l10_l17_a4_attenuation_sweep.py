"""Diagnostic attenuation sweep for the sealed Gemma A4 fold executables.

This rung asks a deliberately narrow question: does the failed A4 Layer-17
full-block correction contain a useful direction whose magnitude was simply
too large?  It restores each already-fitted LOFO executable and scales only
its decoded ``layer.17.mlp.delta`` contribution.  No generator is re-fit.

``alpha=0`` is *not* the frozen uncorrected source composition.  It keeps
generated Layer 10, deletes the compact Layer-17 MLP, and adds none of the A4
correction.  The frozen uncorrected source and the prior A3 trajectory-
corrected composition are carried as distinct authenticated benchmarks.  This
is calibration-A development evidence only: it makes no held-out, serving,
resource, latency, or whole-model-compilation claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_full_block_closure_bundle import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    load_gemma3_l10_l17_full_block_closure_fold_bundle,
    restore_gemma3_l10_l17_full_block_closure_fold,
)
from .gemma3_l10_l17_full_block_closure_lofo import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    _canonical_json_bytes,
    _reject_forbidden_output_fields,
    load_gemma3_l10_l17_full_block_closure_lofo_report,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _authenticate_before_fit_access,
    _merge_corrected_composition_lowerings,
    _source_lowering_maps,
    _validate_source_runtime_catalog,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_family_lofo_authority import (
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_open_a_capacity_evaluation import (
    _add_comparison,
    _add_native,
    _candidate_comparison,
    _file_sha256,
    _finalize_metric_accumulator,
    _model_logits,
    _native_nll,
    _new_metric_accumulator,
    _selected_logits_and_targets,
    _validate_metric_container,
)
from .gemma3_layer17_v8_fit_lofo import _blocks_to_device, _family_blocks
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .modal_generator_graph import ModalGeneratorGraphPlan


__all__ = [
    "A4_ATTENUATION_ALPHA_LADDER",
    "DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT",
    "GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_FORMAT_VERSION",
    "GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_SCHEMA",
    "aggregate_a4_attenuation_folds",
    "build_a4_attenuation_sweep_report",
    "load_gemma3_l10_l17_a4_attenuation_sweep_report",
    "run_gemma3_l10_l17_a4_attenuation_sweep",
    "save_gemma3_l10_l17_a4_attenuation_sweep_report",
    "score_a4_attenuation_fold",
    "validate_gemma3_l10_l17_a4_attenuation_sweep_report",
]


GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a4_postdelta_attenuation_sweep"
)
GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a4-postdelta-attenuation-sweep-v1.json"
)

# Frozen before looking at attenuation outcomes.  Powers of two give a
# logarithmic view while retaining exact binary floating-point scalars.
A4_ATTENUATION_ALPHA_LADDER: tuple[tuple[str, float], ...] = (
    ("alpha_0", 0.0),
    ("alpha_2m14", 2.0**-14),
    ("alpha_2m12", 2.0**-12),
    ("alpha_2m10", 2.0**-10),
    ("alpha_2m8", 2.0**-8),
    ("alpha_2m6", 2.0**-6),
    ("alpha_2m4", 2.0**-4),
    ("alpha_2m2", 2.0**-2),
    ("alpha_1", 1.0),
)

_CONDITIONS = tuple(name for name, _ in A4_ATTENUATION_ALPHA_LADDER)
_NONZERO_CONDITIONS = _CONDITIONS[1:]
_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-attenuation-report:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FAMILY_COUNT = 8
_VOCABULARY_CHUNK_SIZE = 16_384
_METRIC_FIELDS = (
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
)
_COMPARISON_FIELDS = (
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
)
_REPORT_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "source_a4_report",
    "fold_executable_bundle",
    "composition_bundle",
    "runtime",
    "alpha_ladder",
    "alpha_semantics",
    "source_benchmarks",
    "folds",
    "aggregate",
    "comparisons",
    "diagnosis",
    "alpha_one_exact_overlay_replay",
    "alpha_one_source_report_metric_replay",
    "selection_opened",
    "guard_opened",
    "calibration_b_opened",
    "validation_opened",
    "test_opened",
    "full_model_logits_scored",
    "full_model_compiled",
    "heldout_confirmation",
    "serving_authorized",
    "resource_or_latency_claim",
    "refit_performed",
    "safety",
    "report_sha256",
}
_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_gradient_or_projection_tensors": False,
    "contains_model_or_candidate_weights": False,
    "source_safe": True,
}


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ladder_records() -> list[dict[str, object]]:
    exponents: tuple[int | None, ...] = (
        None,
        -14,
        -12,
        -10,
        -8,
        -6,
        -4,
        -2,
        0,
    )
    return [
        {
            "condition_id": name,
            "alpha": alpha,
            "power_of_two_exponent": exponent,
        }
        for (name, alpha), exponent in zip(
            A4_ATTENUATION_ALPHA_LADDER,
            exponents,
            strict=True,
        )
    ]


def _metric_copy(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_METRIC_FIELDS):
        raise ValueError(f"{label} metric fields are invalid")
    record = {
        field: _finite(value.get(field), label=f"{label} {field}")
        for field in _METRIC_FIELDS
    }
    if (
        record["nll_per_token"] < 0.0
        or record["native_to_candidate_kl_per_token"] < 0.0
        or not 0.0 <= record["top1_agreement_to_native"] <= 1.0
    ):
        raise ValueError(f"{label} metric range is invalid")
    return record


def score_a4_attenuation_fold(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    """Score one held family without resource-accounting the intervention."""

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("attenuation batches must contain CalibrationBatch values")
    if executor.affected_layer_ordinals != (10, 17):
        raise ValueError("attenuation executor must compose Layers 10 and 17")
    if 17 not in executor.post_feedforward_delta_layer_ordinals:
        raise ValueError("attenuation executor lacks the Layer17 delta boundary")

    accumulator = _new_metric_accumulator(_CONDITIONS)
    native_model = adapter.module
    exact_alpha_one = True
    with executor.validated_transaction():
        for batch in materialized:
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output), batch
            )
            _add_native(
                accumulator,
                nll_sum=_native_nll(native_logits, targets),
                token_count=targets.numel(),
            )
            for condition, alpha in A4_ATTENUATION_ALPHA_LADDER:
                with torch.no_grad():
                    candidate_output = (
                        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
                            lambda: native_model(**call_inputs),
                            layer_ordinal=17,
                            alpha=alpha,
                            expected_forward_calls=1,
                        )
                    )
                candidate_logits, candidate_targets = (
                    _selected_logits_and_targets(
                        _model_logits(candidate_output), batch
                    )
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{condition} held targets drifted")
                _add_comparison(
                    accumulator,
                    condition,
                    _candidate_comparison(
                        native_logits,
                        candidate_logits,
                        targets,
                        vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                    ),
                )
                if alpha == 1.0:
                    with torch.no_grad():
                        ordinary_output = executor.run_with_generated_overlay(
                            lambda: native_model(**call_inputs),
                            expected_forward_calls=1,
                        )
                    ordinary_logits, ordinary_targets = (
                        _selected_logits_and_targets(
                            _model_logits(ordinary_output), batch
                        )
                    )
                    exact_alpha_one = exact_alpha_one and torch.equal(
                        candidate_targets, ordinary_targets
                    ) and torch.equal(candidate_logits, ordinary_logits)

    if not exact_alpha_one:
        raise RuntimeError("alpha=1 does not exactly replay the ordinary overlay")
    metrics = _finalize_metric_accumulator(
        accumulator,
        conditions=_CONDITIONS,
    )
    _validate_metric_container(
        metrics,
        label="attenuation fold",
        conditions=_CONDITIONS,
    )
    return {
        **metrics,
        "execution_path": "full_model_logits_a4_postdelta_attenuation",
        "application_boundary": "layer.17.mlp.delta",
        "alpha_zero_semantics": (
            "generated_layer10_plus_compact_layer17_deletion"
        ),
        "alpha_one_exact_overlay_replay": True,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "refit_performed": False,
    }


def _validate_fold_evaluation(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    fields = {
        "supervised_tokens",
        "native",
        "conditions",
        "execution_path",
        "application_boundary",
        "alpha_zero_semantics",
        "alpha_one_exact_overlay_replay",
        "full_model_logits_scored",
        "full_model_compiled",
        "heldout_confirmation",
        "serving_authorized",
        "resource_or_latency_claim",
        "refit_performed",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    if type(value.get("supervised_tokens")) is not int or int(
        value["supervised_tokens"]
    ) <= 0:
        raise ValueError(f"{label} supervised-token count is invalid")
    _validate_metric_container(value, label=label, conditions=_CONDITIONS)
    if (
        value.get("execution_path")
        != "full_model_logits_a4_postdelta_attenuation"
        or value.get("application_boundary") != "layer.17.mlp.delta"
        or value.get("alpha_zero_semantics")
        != "generated_layer10_plus_compact_layer17_deletion"
        or value.get("alpha_one_exact_overlay_replay") is not True
        or value.get("full_model_logits_scored") is not True
        or any(
            value.get(field) is not False
            for field in (
                "full_model_compiled",
                "heldout_confirmation",
                "serving_authorized",
                "resource_or_latency_claim",
                "refit_performed",
            )
        )
    ):
        raise ValueError(f"{label} claim boundary is invalid")
    return value


def _aggregate_metric_container(
    evaluations: Sequence[Mapping[str, object]],
    *,
    token_weighted: bool,
) -> dict[str, object]:
    denominators = [
        int(value["supervised_tokens"]) if token_weighted else 1
        for value in evaluations
    ]
    denominator = sum(denominators)
    native_nll = sum(
        weight * float(value["native"]["nll_per_token"])  # type: ignore[index]
        for value, weight in zip(evaluations, denominators, strict=True)
    ) / denominator
    conditions: dict[str, object] = {}
    for name in _CONDITIONS:
        nll = sum(
            weight
            * float(value["conditions"][name]["nll_per_token"])  # type: ignore[index]
            for value, weight in zip(evaluations, denominators, strict=True)
        ) / denominator
        kl = sum(
            weight
            * float(
                value["conditions"][name][  # type: ignore[index]
                    "native_to_candidate_kl_per_token"
                ]
            )
            for value, weight in zip(evaluations, denominators, strict=True)
        ) / denominator
        top1 = sum(
            weight
            * float(
                value["conditions"][name][  # type: ignore[index]
                    "top1_agreement_to_native"
                ]
            )
            for value, weight in zip(evaluations, denominators, strict=True)
        ) / denominator
        conditions[name] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": kl,
            "top1_agreement_to_native": top1,
        }
    result = {
        "supervised_tokens": sum(
            int(value["supervised_tokens"]) for value in evaluations
        ),
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
    }
    _validate_metric_container(
        result,
        label=("micro" if token_weighted else "equal-family macro"),
        conditions=_CONDITIONS,
    )
    return result


def aggregate_a4_attenuation_folds(
    evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    values = tuple(evaluations)
    if len(values) != _FAMILY_COUNT:
        raise ValueError("attenuation aggregate requires exactly eight folds")
    for index, value in enumerate(values):
        _validate_fold_evaluation(value, label=f"fold {index} evaluation")
    return {
        "family_count": _FAMILY_COUNT,
        "completed_fold_count": _FAMILY_COUNT,
        "micro": _aggregate_metric_container(values, token_weighted=True),
        "equal_family_macro": _aggregate_metric_container(
            values, token_weighted=False
        ),
    }


def _source_benchmarks(a4_report: Mapping[str, object]) -> dict[str, object]:
    folds = a4_report.get("folds")
    aggregate = a4_report.get("aggregate")
    if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence):
        raise TypeError("source A4 folds are unavailable")
    if not isinstance(aggregate, Mapping):
        raise TypeError("source A4 aggregate is unavailable")
    by_family: dict[str, object] = {}
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise TypeError("source A4 fold is invalid")
        evaluation = fold.get("evaluation")
        conditions = (
            evaluation.get("conditions")
            if isinstance(evaluation, Mapping)
            else None
        )
        if not isinstance(conditions, Mapping):
            raise TypeError("source A4 fold metrics are unavailable")
        alias = str(fold["held_family_alias"])
        by_family[alias] = {
            "alpha_one_a4_composition": _metric_copy(
                conditions.get("a4_full_block_corrected_composition"),
                label=f"{alias} A4 composition",
            ),
            "frozen_uncorrected_composition": _metric_copy(
                conditions.get("frozen_uncorrected_composition"),
                label=f"{alias} frozen uncorrected composition",
            ),
        }

    def aggregate_pair(name: str) -> dict[str, object]:
        value = aggregate.get(name)
        conditions = value.get("conditions") if isinstance(value, Mapping) else None
        if not isinstance(conditions, Mapping):
            raise TypeError(f"source A4 {name} metrics are unavailable")
        return {
            "alpha_one_a4_composition": _metric_copy(
                conditions.get("a4_full_block_corrected_composition"),
                label=f"{name} A4 composition",
            ),
            "frozen_uncorrected_composition": _metric_copy(
                conditions.get("frozen_uncorrected_composition"),
                label=f"{name} frozen uncorrected composition",
            ),
        }

    if len(by_family) != _FAMILY_COUNT:
        raise ValueError("source A4 family benchmark coverage is incomplete")
    prior = a4_report.get("prior_a3_comparison")
    prior_macro = (
        prior.get("equal_family_macro") if isinstance(prior, Mapping) else None
    )
    prior_metric = _metric_copy(
        prior_macro,
        label="prior A3 corrected composition equal-family macro",
    )
    equal_family_macro = aggregate_pair("equal_family_macro")
    equal_family_macro["prior_a3_corrected_composition"] = prior_metric
    return {
        "by_family": by_family,
        "micro": aggregate_pair("micro"),
        "equal_family_macro": equal_family_macro,
    }


def _metric_differences(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, float]:
    return {
        f"{field}_difference": float(candidate[field]) - float(reference[field])
        for field in _COMPARISON_FIELDS
    }


def _comparison_table(
    *,
    folds: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    source_benchmarks: Mapping[str, object],
) -> dict[str, object]:
    def one_scope(
        conditions: Mapping[str, object],
        external: Mapping[str, object],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for name in _NONZERO_CONDITIONS:
            candidate = conditions[name]
            if not isinstance(candidate, Mapping):
                raise TypeError("attenuation metric is unavailable")
            result[name] = {
                "alpha": dict(A4_ATTENUATION_ALPHA_LADDER)[name],
                "versus_alpha_zero_deletion": _metric_differences(
                    candidate, conditions["alpha_0"]  # type: ignore[arg-type]
                ),
                "versus_alpha_one_a4": _metric_differences(
                    candidate, conditions["alpha_1"]  # type: ignore[arg-type]
                ),
                "versus_frozen_uncorrected_composition": _metric_differences(
                    candidate,
                    external["frozen_uncorrected_composition"],  # type: ignore[arg-type]
                ),
            }
            prior = external.get("prior_a3_corrected_composition")
            if isinstance(prior, Mapping):
                result[name]["versus_prior_a3_corrected_composition"] = (  # type: ignore[index]
                    _metric_differences(candidate, prior)
                )
        return result

    by_family: dict[str, object] = {}
    benchmark_families = source_benchmarks.get("by_family")
    if not isinstance(benchmark_families, Mapping):
        raise TypeError("source family benchmarks are unavailable")
    for fold in folds:
        alias = str(fold["held_family_alias"])
        evaluation = fold["evaluation"]
        if not isinstance(evaluation, Mapping) or not isinstance(
            evaluation.get("conditions"), Mapping
        ):
            raise TypeError("attenuation fold conditions are unavailable")
        external = benchmark_families.get(alias)
        if not isinstance(external, Mapping):
            raise TypeError("source family benchmark is unavailable")
        by_family[alias] = one_scope(
            evaluation["conditions"],  # type: ignore[arg-type]
            external,
        )

    result: dict[str, object] = {"by_family": by_family}
    for scope in ("micro", "equal_family_macro"):
        value = aggregate.get(scope)
        external = source_benchmarks.get(scope)
        if not isinstance(value, Mapping) or not isinstance(
            value.get("conditions"), Mapping
        ) or not isinstance(external, Mapping):
            raise TypeError(f"{scope} attenuation comparison is unavailable")
        result[scope] = one_scope(
            value["conditions"],  # type: ignore[arg-type]
            external,
        )
    return result


def _attenuation_diagnosis(
    *,
    aggregate: Mapping[str, object],
    source_benchmarks: Mapping[str, object],
) -> dict[str, object]:
    """Classify whether scaling reveals useful, competitive A4 direction."""

    macro = aggregate.get("equal_family_macro")
    external = source_benchmarks.get("equal_family_macro")
    conditions = macro.get("conditions") if isinstance(macro, Mapping) else None
    frozen = (
        external.get("frozen_uncorrected_composition")
        if isinstance(external, Mapping)
        else None
    )
    prior_a3 = (
        external.get("prior_a3_corrected_composition")
        if isinstance(external, Mapping)
        else None
    )
    if (
        not isinstance(conditions, Mapping)
        or not isinstance(frozen, Mapping)
        or not isinstance(prior_a3, Mapping)
    ):
        raise TypeError("attenuation diagnosis inputs are unavailable")
    eligible = _CONDITIONS[1:-1]
    best = min(
        eligible,
        key=lambda name: (
            float(conditions[name]["native_to_candidate_kl_per_token"]),  # type: ignore[index]
            float(conditions[name]["delta_nll_per_token"]),  # type: ignore[index]
            dict(A4_ATTENUATION_ALPHA_LADDER)[name],
        ),
    )
    best_metrics = conditions[best]
    alpha_zero = conditions["alpha_0"]
    alpha_one = conditions["alpha_1"]
    assert isinstance(best_metrics, Mapping)
    assert isinstance(alpha_zero, Mapping)
    assert isinstance(alpha_one, Mapping)
    improves_alpha_zero = (
        float(best_metrics["native_to_candidate_kl_per_token"])
        < float(alpha_zero["native_to_candidate_kl_per_token"])
        and float(best_metrics["delta_nll_per_token"])
        < float(alpha_zero["delta_nll_per_token"])
    )
    improves_alpha_one = (
        float(best_metrics["native_to_candidate_kl_per_token"])
        < float(alpha_one["native_to_candidate_kl_per_token"])
        and float(best_metrics["delta_nll_per_token"])
        < float(alpha_one["delta_nll_per_token"])
    )
    beats_frozen_source = (
        float(best_metrics["native_to_candidate_kl_per_token"])
        <= float(frozen["native_to_candidate_kl_per_token"])
        and float(best_metrics["delta_nll_per_token"])
        <= float(frozen["delta_nll_per_token"])
        and float(best_metrics["top1_agreement_to_native"])
        >= float(frozen["top1_agreement_to_native"])
    )
    beats_prior_a3 = (
        float(best_metrics["native_to_candidate_kl_per_token"])
        <= float(prior_a3["native_to_candidate_kl_per_token"])
        and float(best_metrics["delta_nll_per_token"])
        <= float(prior_a3["delta_nll_per_token"])
        and float(best_metrics["top1_agreement_to_native"])
        >= float(prior_a3["top1_agreement_to_native"])
    )
    competitive = beats_frozen_source and beats_prior_a3
    if improves_alpha_zero and competitive:
        classification = "direction_contains_competitive_usable_signal"
    elif improves_alpha_zero:
        classification = "direction_contains_usable_signal_but_not_competitive"
    else:
        classification = "no_macro_usable_signal_on_frozen_ladder"
    return {
        "selection_metric": (
            "equal_family_macro_native_to_candidate_kl_then_delta_nll"
        ),
        "eligible_conditions": list(eligible),
        "best_intermediate_condition": best,
        "best_intermediate_alpha": dict(A4_ATTENUATION_ALPHA_LADDER)[best],
        "best_intermediate_metrics": dict(best_metrics),
        "improves_alpha_zero_on_macro_kl_and_delta_nll": improves_alpha_zero,
        "improves_alpha_one_on_macro_kl_and_delta_nll": improves_alpha_one,
        "beats_frozen_uncorrected_composition_on_macro_kl_delta_nll_and_top1": (
            beats_frozen_source
        ),
        "beats_prior_a3_corrected_composition_on_macro_kl_delta_nll_and_top1": (
            beats_prior_a3
        ),
        "competitive_with_both_source_benchmarks": competitive,
        "classification": classification,
    }


def _metrics_equal(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def build_a4_attenuation_sweep_report(
    *,
    source_a4_report: Mapping[str, object],
    source_a4_report_binding: Mapping[str, object],
    fold_executable_bundle: Mapping[str, object],
    composition_bundle: Mapping[str, object],
    runtime: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fold_values = [dict(value) for value in folds]
    evaluations = [value["evaluation"] for value in fold_values]
    if any(not isinstance(value, Mapping) for value in evaluations):
        raise TypeError("attenuation fold evaluation is unavailable")
    aggregate = aggregate_a4_attenuation_folds(evaluations)  # type: ignore[arg-type]
    benchmarks = _source_benchmarks(source_a4_report)
    benchmark_families = benchmarks["by_family"]
    assert isinstance(benchmark_families, Mapping)
    replay = all(
        _metrics_equal(
            value["evaluation"]["conditions"]["alpha_1"],  # type: ignore[index]
            benchmark_families[str(value["held_family_alias"])][  # type: ignore[index]
                "alpha_one_a4_composition"
            ],
        )
        for value in fold_values
    )
    if not replay:
        raise RuntimeError("alpha=1 metrics do not replay the source A4 report")
    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_SCHEMA,
        "format_version": GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_FORMAT_VERSION,
        "scientific_role": "calibration_a_fit_a4_attenuation_diagnostic",
        "source_a4_report": dict(source_a4_report_binding),
        "fold_executable_bundle": dict(fold_executable_bundle),
        "composition_bundle": dict(composition_bundle),
        "runtime": dict(runtime),
        "alpha_ladder": _ladder_records(),
        "alpha_semantics": {
            "alpha_zero": (
                "generated_layer10_plus_compact_layer17_deletion_not_"
                "frozen_source_composition"
            ),
            "intermediate_alpha": (
                "scale_only_decoded_a4_layer17_post_feedforward_delta"
            ),
            "alpha_one": "exact_ordinary_a4_generated_overlay",
            "frozen_uncorrected_composition": (
                "external_frozen_source_composition_benchmark"
            ),
            "prior_a3_corrected_composition": (
                "aggregate_only_prior_trajectory_corrected_benchmark"
            ),
        },
        "source_benchmarks": benchmarks,
        "folds": fold_values,
        "aggregate": aggregate,
        "comparisons": _comparison_table(
            folds=fold_values,
            aggregate=aggregate,
            source_benchmarks=benchmarks,
        ),
        "diagnosis": _attenuation_diagnosis(
            aggregate=aggregate,
            source_benchmarks=benchmarks,
        ),
        "alpha_one_exact_overlay_replay": all(
            value["evaluation"].get("alpha_one_exact_overlay_replay") is True  # type: ignore[union-attr]
            for value in fold_values
        ),
        "alpha_one_source_report_metric_replay": replay,
        "selection_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "refit_performed": False,
        "safety": dict(_SAFETY),
    }
    _reject_forbidden_output_fields(payload)
    report = {
        **payload,
        "report_sha256": _domain_sha256(_REPORT_DOMAIN, payload),
    }
    return validate_gemma3_l10_l17_a4_attenuation_sweep_report(report)


def _validate_source_binding(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} is unavailable")
    for field in value:
        if field.endswith("sha256"):
            _require_sha256(value[field], label=f"{label} {field}")
    return value


def validate_gemma3_l10_l17_a4_attenuation_sweep_report(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on metric, lineage, ladder, or claim-boundary drift."""

    if not isinstance(raw, Mapping) or set(raw) != _REPORT_FIELDS:
        raise ValueError("A4 attenuation report fields are invalid")
    _reject_forbidden_output_fields(raw)
    if (
        raw.get("schema") != GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_SCHEMA
        or raw.get("format_version")
        != GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_FORMAT_VERSION
        or raw.get("scientific_role")
        != "calibration_a_fit_a4_attenuation_diagnostic"
        or raw.get("alpha_ladder") != _ladder_records()
        or raw.get("alpha_semantics")
        != {
            "alpha_zero": (
                "generated_layer10_plus_compact_layer17_deletion_not_"
                "frozen_source_composition"
            ),
            "intermediate_alpha": (
                "scale_only_decoded_a4_layer17_post_feedforward_delta"
            ),
            "alpha_one": "exact_ordinary_a4_generated_overlay",
            "frozen_uncorrected_composition": (
                "external_frozen_source_composition_benchmark"
            ),
            "prior_a3_corrected_composition": (
                "aggregate_only_prior_trajectory_corrected_benchmark"
            ),
        }
        or raw.get("alpha_one_exact_overlay_replay") is not True
        or raw.get("alpha_one_source_report_metric_replay") is not True
        or raw.get("full_model_logits_scored") is not True
        or raw.get("safety") != _SAFETY
        or any(
            raw.get(field) is not False
            for field in (
                "selection_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
                "full_model_compiled",
                "heldout_confirmation",
                "serving_authorized",
                "resource_or_latency_claim",
                "refit_performed",
            )
        )
    ):
        raise ValueError("A4 attenuation claim or ladder boundary is invalid")
    source = _validate_source_binding(
        raw.get("source_a4_report"), label="source A4 report"
    )
    fold_bundle = _validate_source_binding(
        raw.get("fold_executable_bundle"), label="fold executable bundle"
    )
    composition = _validate_source_binding(
        raw.get("composition_bundle"), label="composition bundle"
    )
    required_source = {"file", "file_sha256", "report_sha256", "protocol_sha256"}
    required_fold = {
        "file",
        "file_sha256",
        "scientific_payload_sha256",
        "protocol_sha256",
    }
    required_composition = {
        "file",
        "file_sha256",
        "composition_payload_sha256",
        "primary_graph_sha256",
    }
    if set(source) != required_source or set(fold_bundle) != required_fold or set(
        composition
    ) != required_composition:
        raise ValueError("A4 attenuation source binding fields are invalid")
    if source["protocol_sha256"] != fold_bundle["protocol_sha256"]:
        raise ValueError("A4 report and fold bundle protocols differ")
    runtime = raw.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "model_id",
        "requested_revision",
        "model_fingerprint",
        "device",
        "dtype",
        "local_files_only",
        "vocabulary_chunk_size",
    }:
        raise ValueError("A4 attenuation runtime fields are invalid")
    if (
        not isinstance(runtime.get("model_id"), str)
        or not isinstance(runtime.get("requested_revision"), str)
        or _REVISION.fullmatch(str(runtime["requested_revision"])) is None
        or _SHA256.fullmatch(str(runtime.get("model_fingerprint"))) is None
        or runtime.get("local_files_only") is not True
        or runtime.get("vocabulary_chunk_size") != _VOCABULARY_CHUNK_SIZE
    ):
        raise ValueError("A4 attenuation runtime identity is invalid")

    folds_raw = raw.get("folds")
    if isinstance(folds_raw, (str, bytes)) or not isinstance(folds_raw, Sequence):
        raise TypeError("A4 attenuation folds are unavailable")
    folds = tuple(folds_raw)
    if len(folds) != _FAMILY_COUNT:
        raise ValueError("A4 attenuation requires exactly eight folds")
    aliases: list[str] = []
    evaluations: list[Mapping[str, object]] = []
    fold_fields = {
        "fold_index",
        "fold_id",
        "held_family_alias",
        "protocol_fold_sha256",
        "a4_layer17_graph_sha256",
        "a4_composition_graph_sha256",
        "evaluation",
    }
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or set(fold) != fold_fields:
            raise ValueError(f"attenuation fold {index} fields are invalid")
        if fold.get("fold_index") != index:
            raise ValueError("attenuation fold order drifted")
        alias = fold.get("held_family_alias")
        if not isinstance(alias, str) or not alias:
            raise ValueError("attenuation held-family alias is invalid")
        aliases.append(alias)
        _require_sha256(
            fold.get("protocol_fold_sha256"), label="protocol fold"
        )
        _require_sha256(
            fold.get("a4_layer17_graph_sha256"), label="A4 Layer17 graph"
        )
        _require_sha256(
            fold.get("a4_composition_graph_sha256"), label="A4 composition graph"
        )
        evaluations.append(
            _validate_fold_evaluation(
                fold.get("evaluation"), label=f"fold {index} evaluation"
            )
        )
    if len(set(aliases)) != _FAMILY_COUNT:
        raise ValueError("attenuation held-family coverage is incomplete")

    benchmarks = raw.get("source_benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != {
        "by_family",
        "micro",
        "equal_family_macro",
    }:
        raise ValueError("attenuation source benchmarks are invalid")
    benchmark_families = benchmarks.get("by_family")
    if not isinstance(benchmark_families, Mapping) or set(
        benchmark_families
    ) != set(aliases):
        raise ValueError("attenuation source family benchmarks are incomplete")
    for scope, value in (*benchmark_families.items(), ("micro", benchmarks["micro"])):
        if not isinstance(value, Mapping) or set(value) != {
            "alpha_one_a4_composition",
            "frozen_uncorrected_composition",
        }:
            raise ValueError(f"{scope} source benchmark fields are invalid")
        _metric_copy(value["alpha_one_a4_composition"], label=f"{scope} alpha1")
        _metric_copy(
            value["frozen_uncorrected_composition"], label=f"{scope} frozen"
        )
    macro_benchmarks = benchmarks["equal_family_macro"]
    if not isinstance(macro_benchmarks, Mapping) or set(macro_benchmarks) != {
        "alpha_one_a4_composition",
        "frozen_uncorrected_composition",
        "prior_a3_corrected_composition",
    }:
        raise ValueError("equal-family macro source benchmark fields are invalid")
    for name, value in macro_benchmarks.items():
        _metric_copy(value, label=f"equal-family macro {name}")
    for alias, evaluation in zip(aliases, evaluations, strict=True):
        if not _metrics_equal(
            evaluation["conditions"]["alpha_1"],  # type: ignore[index]
            benchmark_families[alias]["alpha_one_a4_composition"],  # type: ignore[index]
        ):
            raise ValueError("alpha=1 metrics do not replay source A4 family")

    aggregate = aggregate_a4_attenuation_folds(evaluations)
    if not _metrics_equal(raw.get("aggregate"), aggregate):
        raise ValueError("A4 attenuation aggregate was not reproduced")
    comparisons = _comparison_table(
        folds=folds,  # type: ignore[arg-type]
        aggregate=aggregate,
        source_benchmarks=benchmarks,
    )
    if not _metrics_equal(raw.get("comparisons"), comparisons):
        raise ValueError("A4 attenuation comparisons were not reproduced")
    diagnosis = _attenuation_diagnosis(
        aggregate=aggregate,
        source_benchmarks=benchmarks,
    )
    if not _metrics_equal(raw.get("diagnosis"), diagnosis):
        raise ValueError("A4 attenuation diagnosis was not reproduced")
    supplied = _require_sha256(raw.get("report_sha256"), label="A4 attenuation report")
    payload = {key: raw[key] for key in _REPORT_FIELDS if key != "report_sha256"}
    if supplied != _domain_sha256(_REPORT_DOMAIN, payload):
        raise ValueError("A4 attenuation report hash mismatch")
    return json.loads(_canonical_json_bytes(raw).decode("utf-8"))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_gemma3_l10_l17_a4_attenuation_sweep_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A4 attenuation report")
    validated = validate_gemma3_l10_l17_a4_attenuation_sweep_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(validated) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                "refusing to overwrite A4 attenuation report"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_gemma3_l10_l17_a4_attenuation_sweep_report(
    path: Path | str,
) -> dict[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("A4 attenuation report is not strict JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("A4 attenuation report must contain one object")
    return validate_gemma3_l10_l17_a4_attenuation_sweep_report(raw)


def _progress(message: str) -> None:
    print(f"[a4-attenuation] {message}", flush=True)


def _fold_bundle_binding(
    *,
    path: Path,
    bundle: Mapping[str, object],
    a4_report: Mapping[str, object],
) -> dict[str, object]:
    receipt = a4_report.get("fold_executable_bundle")
    if not isinstance(receipt, Mapping):
        raise TypeError("source A4 fold-bundle receipt is unavailable")
    file_sha256 = _file_sha256(path)
    if (
        file_sha256 != receipt.get("tensor_file_sha256")
        or path.name != receipt.get("tensor_file")
        or bundle.get("scientific_payload_sha256")
        != receipt.get("scientific_payload_sha256")
        or bundle.get("protocol_sha256") != receipt.get("protocol_sha256")
    ):
        raise ValueError("A4 fold bundle differs from the source report")
    return {
        "file": path.name,
        "file_sha256": file_sha256,
        "scientific_payload_sha256": bundle["scientific_payload_sha256"],
        "protocol_sha256": bundle["protocol_sha256"],
    }


def run_gemma3_l10_l17_a4_attenuation_sweep(
    *,
    revision: str,
    output: Path | str = DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT,
    source_a4_report_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT
    ),
    fold_bundle_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE
    ),
    composition_bundle_path: Path | str = DEFAULT_COMPOSITION_BUNDLE_PATH,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run the no-refit logarithmic attenuation diagnostic."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A4 attenuation report")
    source_path = Path(source_a4_report_path)
    executable_path = Path(fold_bundle_path)

    _progress("preflight: authenticate strict A4 report and fold executables")
    source_report = load_gemma3_l10_l17_full_block_closure_lofo_report(
        source_path
    )
    fold_bundle = load_gemma3_l10_l17_full_block_closure_fold_bundle(
        executable_path
    )
    fold_binding = _fold_bundle_binding(
        path=executable_path,
        bundle=fold_bundle,
        a4_report=source_report,
    )
    source_runtime = source_report.get("runtime")
    source_authorization = source_report.get("authorization")
    if not isinstance(source_runtime, Mapping) or not isinstance(
        source_authorization, Mapping
    ):
        raise TypeError("source A4 runtime/authorization is unavailable")
    if (
        source_runtime.get("model_id") != model_id
        or source_runtime.get("requested_revision") != revision
        or source_runtime.get("device") != device_name
        or source_runtime.get("dtype") != dtype
    ):
        raise ValueError("diagnostic runtime must exactly replay source A4 runtime")

    # This repeats the source's pre-fit authorization order: authenticate the
    # composition before opening calibration-A fit prompts.
    bundle, authority, _, source_fit_authorization = (
        _authenticate_before_fit_access(
            bundle_path=composition_bundle_path,
            corpus_receipt_path=corpus_receipt_path,
            corpus_artifact_path=corpus_artifact_path,
            fit_input_path=fit_input_path,
        )
    )
    if (
        not _metrics_equal(
            source_authorization.get("bundle"),
            source_fit_authorization.get("bundle"),
        )
        or not _metrics_equal(
            source_authorization.get("fit_authority"),
            source_fit_authorization.get("fit_authority"),
        )
        or source_authorization.get("fit_authority_sha256")
        != source_fit_authorization.get("fit_authority_sha256")
    ):
        raise ValueError("live A-fit authority differs from source A4 report")
    bundle_binding = getattr(bundle, "binding", None)
    primary_graph = getattr(bundle, "primary", None)
    if not isinstance(bundle_binding, Mapping) or not isinstance(
        primary_graph, ModalGeneratorGraphPlan
    ):
        raise TypeError("authenticated composition runtime is unavailable")
    protocol = source_report.get("protocol")
    catalog = source_authorization.get("source_runtime_catalog")
    if not isinstance(protocol, Mapping):
        raise TypeError("source A4 protocol is unavailable")
    _validate_source_runtime_catalog(
        catalog,
        protocol=protocol,
        bundle_binding=bundle_binding,
    )
    if (
        fold_bundle.get("source_runtime_catalog_sha256")
        != catalog.get("catalog_sha256")  # type: ignore[union-attr]
        or fold_bundle.get("source_composition_graph_sha256")
        != primary_graph.artifact_sha256
        or fold_bundle.get("model_fingerprint")
        != primary_graph.model_fingerprint
    ):
        raise ValueError("A4 fold bundle is not bound to the live composition")

    layer10_graph, _, layer10_lowerings, _ = _source_lowering_maps(bundle)
    source_graph_sha256 = primary_graph.artifact_sha256
    source_layer10_sha256 = layer10_graph.artifact_sha256
    source_layer10_lowerings = tuple(
        layer10_lowerings[name].artifact_sha256
        for name in layer10_graph.traversal_order
    )

    randomness = source_runtime.get("randomness")
    inherited = (
        randomness.get("inherits_seed_and_execution_recipe_from")
        if isinstance(randomness, Mapping)
        else None
    )
    seed = inherited.get("torch_seed") if isinstance(inherited, Mapping) else None
    if type(seed) is not int:
        raise ValueError("source A4 deterministic seed is unavailable")
    torch.manual_seed(seed)
    device = resolve_torch_device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned local Gemma checkpoint")
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
    if adapter.model_fingerprint() != source_runtime.get("model_fingerprint"):
        raise ValueError("live Gemma fingerprint differs from source A4 run")

    _progress("tokenize: replay calibration_a_fit family blocks")
    raw_blocks, materialization = materialize_gemma3_layer17_family_lofo(
        authority, tokenizer
    )
    validate_gemma3_layer17_family_lofo_materialization_metadata(materialization)
    fit_collection = source_report.get("fit_collection")
    if not isinstance(fit_collection, Mapping) or not _metrics_equal(
        materialization, fit_collection.get("materialization")
    ):
        raise ValueError("A-fit materialization differs from source A4 run")
    blocks = _blocks_to_device(_family_blocks(raw_blocks), device)
    held_batches = dict(blocks)
    source_folds = source_report.get("folds")
    bundle_folds = fold_bundle.get("folds")
    if (
        isinstance(source_folds, (str, bytes))
        or not isinstance(source_folds, Sequence)
        or isinstance(bundle_folds, (str, bytes))
        or not isinstance(bundle_folds, Sequence)
        or len(source_folds) != _FAMILY_COUNT
        or len(bundle_folds) != _FAMILY_COUNT
    ):
        raise ValueError("A4 source fold catalogs are incomplete")

    fold_results: list[dict[str, object]] = []
    for index, (source_fold, bundle_fold) in enumerate(
        zip(source_folds, bundle_folds, strict=True)
    ):
        if not isinstance(source_fold, Mapping) or not isinstance(
            bundle_fold, Mapping
        ):
            raise TypeError("A4 source fold record is invalid")
        held = str(source_fold["held_family_alias"])
        _progress(f"fold {index + 1}/{_FAMILY_COUNT}: sweep held {held}")
        graph, lowerings = restore_gemma3_l10_l17_full_block_closure_fold(
            fold_bundle, index
        )
        if (
            bundle_fold.get("held_family_alias") != held
            or graph.artifact_sha256
            != source_fold.get("corrected_layer17_graph_sha256")
            or {
                name: lowerings[name].artifact_sha256
                for name in graph.traversal_order
            }
            != source_fold.get("corrected_lowering_sha256_by_node")
        ):
            raise ValueError(f"A4 executable fold {index} is cross-bound wrong")
        composition = replace_layer_nodes_in_composed_graph(
            primary_graph, graph, layer_ordinal=17
        )
        if composition.artifact_sha256 != source_fold.get(
            "corrected_primary_graph_sha256"
        ):
            raise ValueError("restored A4 composition graph drifted")
        merged = _merge_corrected_composition_lowerings(
            composition,
            layer10_lowerings_by_node=layer10_lowerings,
            corrected_layer17_lowerings_by_node=lowerings,
        )
        executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            composition,
            merged,
            post_feedforward_delta_layer_ordinals=(17,),
        )
        evaluation = score_a4_attenuation_fold(
            adapter=adapter,
            executor=executor,
            batches=held_batches[held],
        )
        source_conditions = source_fold["evaluation"]["conditions"]  # type: ignore[index]
        if not _metrics_equal(
            evaluation["conditions"]["alpha_1"],  # type: ignore[index]
            source_conditions["a4_full_block_corrected_composition"],
        ):
            raise RuntimeError("alpha=1 metric replay differs from source A4")
        fold_results.append(
            {
                "fold_index": index,
                "fold_id": source_fold["fold_id"],
                "held_family_alias": held,
                "protocol_fold_sha256": source_fold["protocol_fold_sha256"],
                "a4_layer17_graph_sha256": graph.artifact_sha256,
                "a4_composition_graph_sha256": composition.artifact_sha256,
                "evaluation": evaluation,
            }
        )

    if (
        primary_graph.artifact_sha256 != source_graph_sha256
        or layer10_graph.artifact_sha256 != source_layer10_sha256
        or tuple(
            layer10_lowerings[name].artifact_sha256
            for name in layer10_graph.traversal_order
        )
        != source_layer10_lowerings
        or adapter.model_fingerprint() != source_runtime.get("model_fingerprint")
    ):
        raise RuntimeError("attenuation sweep mutated frozen source state")

    source_binding = {
        "file": source_path.name,
        "file_sha256": _file_sha256(source_path),
        "report_sha256": source_report["report_sha256"],
        "protocol_sha256": protocol["artifact_sha256"],
    }
    composition_binding = {
        "file": Path(composition_bundle_path).name,
        "file_sha256": bundle_binding["bundle_file_sha256"],
        "composition_payload_sha256": bundle_binding[
            "composition_payload_sha256"
        ],
        "primary_graph_sha256": primary_graph.artifact_sha256,
    }
    report = build_a4_attenuation_sweep_report(
        source_a4_report=source_report,
        source_a4_report_binding=source_binding,
        fold_executable_bundle=fold_binding,
        composition_bundle=composition_binding,
        runtime={
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "device": str(device),
            "dtype": dtype,
            "local_files_only": True,
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
        },
        folds=fold_results,
    )
    _progress("report: publish strict tensor-free JSON without overwrite")
    return save_gemma3_l10_l17_a4_attenuation_sweep_report(destination, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed Gemma A4 Layer17 post-delta attenuation sweep."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT,
    )
    parser.add_argument(
        "--source-a4-report",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--fold-bundle",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    )
    parser.add_argument(
        "--composition-bundle",
        type=Path,
        default=DEFAULT_COMPOSITION_BUNDLE_PATH,
    )
    parser.add_argument("--corpus-receipt", type=Path, default=DEFAULT_RECEIPT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_gemma3_l10_l17_a4_attenuation_sweep(
        revision=arguments.revision,
        output=arguments.output,
        source_a4_report_path=arguments.source_a4_report,
        fold_bundle_path=arguments.fold_bundle,
        composition_bundle_path=arguments.composition_bundle,
        corpus_receipt_path=arguments.corpus_receipt,
        corpus_artifact_path=arguments.corpus,
        fit_input_path=arguments.fit_input,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
