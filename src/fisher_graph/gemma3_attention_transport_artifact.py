"""Strict source-safe artifacts for the Gemma attention transport A/B/C rung.

The executable tensor artifact contains only the selected shared generator
bank, one frozen chart atlas, and the chart-conditioned routers bound to both.
Calibration captures, native V/O weights, teacher outputs, gradients, sparse
oracle rows, prompt text, and token rows are deliberately forbidden.

The logical payload digest authenticates exact tensor dtype, shape, and bytes.
The companion canonical JSON report binds that digest to the serialized tensor
file.  Publication refuses overwrites, makes the tensor artifact durable first,
and links the report last as the commit marker.  These hashes detect accidental
or uncoordinated mutation; they are not signatures or external provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import unicodedata

import torch
from torch import Tensor

from .gemma3_attention_transport_compiler import (
    AttentionTransportResourceAccounting,
    ChartConditionedAttentionTransportRouter,
    EXACT_SPARSE_ORACLE_SOLVER,
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    HELDOUT_ORACLE_COEFFICIENT_RIDGE,
    chart_conditioned_attention_transport_router_from_state_dict,
    fisher_attention_transport_generator_bank_from_state_dict,
    frozen_attention_chart_atlas_from_state_dict,
)


ATTENTION_TRANSPORT_ARTIFACT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport_abc.v1"
)
ATTENTION_TRANSPORT_ARTIFACT_FORMAT_VERSION = 1
ATTENTION_TRANSPORT_REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport_abc.report.v1"
)
ATTENTION_TRANSPORT_REPORT_FORMAT_VERSION = 1

_PAYLOAD_DOMAIN = b"fisher_graph:gemma3-attention-transport:payload:v1\0"
_REPORT_DOMAIN = b"fisher_graph:gemma3-attention-transport:report:v1\0"
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SCIENTIFIC_STATUS: dict[str, object] = {
    "abc_role_firewall": True,
    "attention_boundary": "pre_output_rmsnorm_operator_output",
    "behavioral_nll_measured": False,
    "behavioral_quality_claimed": False,
    "deployment_or_latency_improvement_claimed": False,
    "native_output_rmsnorm_retained": True,
    "native_qk_retained": True,
    "operator_boundary_only": True,
    "parameter_storage_reported": True,
    "post_rmsnorm_diagnostic_measured": True,
    "scope": "gemma3_attention_value_output_transport_only",
    "teacher_only_fisher_fit_separated_from_deployment_router": True,
}
_SAFETY: dict[str, bool] = {
    "contains_attention_probabilities": False,
    "contains_calibration_captures": False,
    "contains_generator_bank_weights": True,
    "contains_native_value_or_output_weights": False,
    "contains_oracle_outputs_or_coefficients": False,
    "contains_prompt_text": False,
    "contains_router_weights": True,
    "contains_teacher_outputs_or_gradients": False,
    "contains_token_ids": False,
    "contains_tokenizer_state": False,
    "executable_attention_transport": True,
}
_FORBIDDEN_KEYS = frozenset(
    {
        "activation_rows",
        "attention_probabilities",
        "capture",
        "captures",
        "gradient_rows",
        "head_contexts",
        "input_ids",
        "labels",
        "model_state_dict",
        "native_output_weight",
        "native_value_weight",
        "normalized_inputs",
        "operator_output_score_gradients",
        "operator_outputs",
        "oracle_coefficients",
        "oracle_outputs",
        "oracle_support",
        "output_projection_weight",
        "probabilities",
        "prompt",
        "prompt_text",
        "prompts",
        "raw_activation_rows",
        "raw_capture_rows",
        "raw_gradient_rows",
        "raw_prompt_rows",
        "raw_token_rows",
        "score_gradients",
        "source_model_weights",
        "source_parameter_values",
        "source_state_dict",
        "support",
        "targets",
        "teacher_outputs",
        "teacher_targets",
        "text",
        "token_ids",
        "tokenizer_state",
        "value_projection_weight",
    }
)
_PAYLOAD_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "protocol",
    "splits",
    "selection",
    "generator_bank",
    "atlas",
    "routers",
    "resource_accounting",
    "evaluations",
    "safety",
    "scientific_payload_sha256",
}
_REPORT_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "protocol",
    "splits",
    "selection",
    "generator_bank",
    "atlas",
    "routers",
    "resource_accounting",
    "evaluations",
    "safety",
    "artifact",
    "report_sha256",
}
_RESOURCE_FIELDS = {
    "native_v_o_parameters",
    "generator_bank_parameters",
    "chart_parameters",
    "router_parameters",
    "compiled_parameters",
    "parameter_reduction_fraction",
    "parameter_compression_eligible",
    "native_qk_retained",
    "native_output_rmsnorm_retained",
    "logical_speedup_claimed",
}
_MODEL_FIELDS = {
    "model_id",
    "model_revision",
    "hidden_width",
    "runtime_qualification_receipt_sha256",
    "runtime_execution_receipt_sha256",
}
_PROTOCOL_FIELDS = {
    "protocol_id",
    "a_fit_only",
    "b_selection_only",
    "c_confirmation_only",
    "no_post_c_refit_or_selection",
    "rows_per_batch_and_invocation",
    "expected_rows_per_role",
    "candidate_configs",
    "gates",
    "implementation_source_sha256s",
    "preflight_receipt_sha256",
    "dense_lexical_parity_receipt_sha256",
}
_CANDIDATE_CONFIG_FIELDS = {
    "candidate_id",
    "config_artifact_sha256",
    "generator_count",
    "generator_rank",
    "bank_steps",
    "bank_learning_rate",
    "euclidean_floor",
    "chart_count",
    "chart_rank",
    "chart_temperature_scale",
    "chart_lloyd_iterations",
    "active_count",
    "router_training_oracle",
    "router_steps",
    "router_learning_rate",
}
_TRAINING_ORACLE_FIELDS = {
    "split",
    "solver_kind",
    "coefficient_ridge",
    "heldout_metric_used",
}
_GATE_FIELDS = {
    "gates_artifact_sha256",
    "max_relative_fisher_loss",
    "max_oracle_router_fisher_gap",
    "oracle_gap_numerical_tolerance",
    "max_postnorm_normalized_rmse",
    "min_postnorm_cosine",
    "max_sparse_probability_renormalization_error",
    "max_effective_support_fraction",
    "require_parameter_compression",
    "minimum_aggregate_rows",
}
_IMPLEMENTATION_SOURCE_PATHS = frozenset(
    {
        "src/fisher_graph/gemma3_attention_transport_compiler.py",
        "src/fisher_graph/gemma3_attention_transport_runtime.py",
        "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
        "src/fisher_graph/gemma3_attention_transport_artifact.py",
        "src/fisher_graph/gemma3_attention_transport_experiment.py",
        "pyproject.toml",
    }
)
_SPLIT_FIELDS = {
    "A_split_artifact_sha256",
    "B_split_artifact_sha256",
    "C_split_artifact_sha256",
    "A_receipt_sha256s",
    "B_receipt_sha256s",
    "C_receipt_sha256s",
    "A_B_C_receipts_disjoint",
}
_SELECTION_FIELDS = {
    "selected_candidate_id",
    "selected_candidate_artifact_sha256",
    "selection_lock_artifact_sha256",
    "B_selection_artifact_sha256",
    "B_decision_artifact_sha256",
    "C_confirmation_artifact_sha256",
    "B_passed",
    "C_passed",
    "selection_rule",
}
_SELECTION_RULE = (
    "minimum_B_relative_fisher_then_compiled_parameters_then_K_rank_"
    "active_count_then_candidate_id"
)
_EVALUATION_FIELDS = {"A", "B", "C", "all_B_candidates", "runtime"}
_A_EVALUATION_FIELDS = {"initial_bank", "final_bank"}
_METRIC_FIELDS = {
    "row_count",
    "fisher_loss",
    "directional_mse",
    "euclidean_mse",
    "normalized_rmse",
    "cosine",
}
_DECISION_FIELDS = {
    "candidate_id",
    "candidate_artifact_sha256",
    "evaluation_artifact_sha256",
    "relative_fisher_pass",
    "oracle_gap_pass",
    "postnorm_nrmse_pass",
    "postnorm_cosine_pass",
    "sparse_renormalization_pass",
    "effective_support_pass",
    "global_support_budget_pass",
    "resource_pass",
    "minimum_rows_pass",
    "passed",
    "evaluation",
}
_DECISION_PASS_FIELDS = (
    "relative_fisher_pass",
    "oracle_gap_pass",
    "postnorm_nrmse_pass",
    "postnorm_cosine_pass",
    "sparse_renormalization_pass",
    "effective_support_pass",
    "global_support_budget_pass",
    "resource_pass",
    "minimum_rows_pass",
)
_AGGREGATE_EVALUATION_FIELDS = {
    "role",
    "split_artifact_sha256",
    "row_count",
    "invocation_count",
    "fisher_loss",
    "zero_predictor_fisher_loss",
    "relative_fisher_loss",
    "oracle_fisher_loss",
    "oracle_contract",
    "oracle_relative_fisher_loss",
    "oracle_router_fisher_gap",
    "operator_normalized_rmse",
    "operator_cosine",
    "postnorm_normalized_rmse",
    "postnorm_cosine",
    "per_chart_active_count",
    "effective_support_count",
    "sparse_probability_renormalization_max_error",
    "generator_usage",
    "dead_generator_count",
    "resources",
    "per_invocation",
}
_INVOCATION_EVALUATION_FIELDS = {
    "invocation_id",
    "operator",
    "oracle",
    "oracle_contract",
    "zero_predictor_fisher_loss",
    "relative_fisher_loss",
    "oracle_relative_fisher_loss",
    "oracle_router_fisher_gap",
    "postnorm",
    "per_chart_active_count",
    "effective_support_count",
    "sparse_probability_renormalization_max_error",
    "generator_usage",
    "dead_generator_count",
}
_ORACLE_CONTRACT_FIELDS = {
    "solver_kind",
    "active_count_limit",
    "coefficient_ridge",
    "heldout_ridge_free",
}
_COUNT_SUMMARY_FIELDS = {"minimum", "mean", "maximum"}
_SUPPORT_SUMMARY_FIELDS = {"minimum", "mean", "maximum", "fraction"}
_POSTNORM_FIELDS = {"row_count", "normalized_rmse", "cosine"}
_RUNTIME_FIELDS = {
    "counters",
    "prepublication_peak_resident_set_bytes",
    "resident_set_cap_bytes",
    "publication_guarded",
    "logical_speedup_claimed",
}
_RUNTIME_COUNTER_FIELDS = {
    "role_batch_counts",
    "chart_backbone_forward_count",
    "chart_terminal_cotangent_vjp_count",
    "dense_parity_smoke_count",
    "parity_backbone_forward_count",
    "parity_terminal_cotangent_vjp_count",
    "captured_example_count",
    "supplemental_route_receipt_count",
    "canonical_gradient_targets_per_vjp",
    "auxiliary_gradient_targets_per_vjp",
    "gradient_targets_per_vjp",
    "contains_prompt_text",
    "contains_token_ids",
    "contains_raw_activations",
    "contains_raw_gradients",
    "contains_logits",
}
_EXPECTED_ROLE_BATCH_COUNTS = [
    ["A_chart_transport_fit", 45],
    ["B_neighbor_discovery", 45],
    ["C_neighbor_confirmation", 45],
]
_ROWS_PER_BATCH_AND_INVOCATION = 16
_EXPECTED_ROWS_PER_ROLE = 12_960


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict canonical JSON used by report and lineage digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _normalize_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("-", "_")


def _assert_machine_string(value: str, *, path: tuple[str, ...]) -> None:
    if (
        not value
        or len(value.encode("utf-8")) > 16_384
        or any(
            character.isspace()
            or unicodedata.category(character).startswith("C")
            for character in value
        )
    ):
        location = ".".join(path) or "<root>"
        raise ValueError(f"artifact contains non-machine string at {location}")


def _assert_source_safe(
    value: object,
    *,
    path: tuple[str, ...] = (),
    tensors_allowed: bool = True,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact mapping keys must be strings")
            _assert_machine_string(key, path=(*path, "<key>"))
            if _normalize_key(key) in _FORBIDDEN_KEYS:
                location = ".".join((*path, key))
                raise ValueError(f"artifact contains forbidden field {location}")
            _assert_source_safe(
                child,
                path=(*path, key),
                tensors_allowed=tensors_allowed,
            )
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_source_safe(
                child,
                path=(*path, str(index)),
                tensors_allowed=tensors_allowed,
            )
        return
    if isinstance(value, str):
        _assert_machine_string(value, path=path)
        return
    if isinstance(value, Tensor):
        if not tensors_allowed:
            location = ".".join(path) or "<root>"
            raise ValueError(f"metadata contains a tensor at {location}")
        if value.layout != torch.strided:
            raise ValueError("artifact tensors must use strided layout")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("artifact tensors must be finite")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(
        "artifact contains unsupported scalar " f"{type(value).__qualname__}"
    )


def _assert_metadata(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty mapping")
    _assert_source_safe(value, path=(label,), tensors_allowed=False)


def _update_payload_digest(digest: object, value: object) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a SHA-256 object")
    if value is None:
        digest.update(b"N;")
    elif isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
    elif type(value) is int:
        digest.update(f"I{value};".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific payload floats must be finite")
        digest.update(f"F{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"S{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.layout != torch.strided:
            raise ValueError("scientific payload tensors must be strided")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("scientific payload tensors must be finite")
        digest.update(b"T")
        _update_payload_digest(digest, str(tensor.dtype))
        _update_payload_digest(digest, tuple(tensor.shape))
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(f"{len(raw)}:".encode("ascii"))
        digest.update(raw)
        digest.update(b";")
    elif isinstance(value, Mapping):
        keys = sorted(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("scientific payload mapping keys must be strings")
        digest.update(f"M{len(keys)}[".encode("ascii"))
        for key in keys:
            _update_payload_digest(digest, key)
            _update_payload_digest(digest, value[key])
        digest.update(b"];")
    elif isinstance(value, tuple):
        digest.update(f"U{len(value)}[".encode("ascii"))
        for child in value:
            _update_payload_digest(digest, child)
        digest.update(b"];")
    elif isinstance(value, list):
        digest.update(f"L{len(value)}[".encode("ascii"))
        for child in value:
            _update_payload_digest(digest, child)
        digest.update(b"];")
    else:
        raise TypeError(
            "scientific payload contains unsupported "
            f"{type(value).__qualname__}"
        )


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + canonical_json_bytes(value)).hexdigest()


def _resource_state(
    value: AttentionTransportResourceAccounting | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(value, AttentionTransportResourceAccounting):
        state = asdict(value)
    elif isinstance(value, Mapping):
        state = dict(value)
    else:
        raise TypeError("resource_accounting has the wrong type")
    if set(state) != _RESOURCE_FIELDS:
        raise ValueError("attention transport resource fields are invalid")
    for name in (
        "native_v_o_parameters",
        "generator_bank_parameters",
        "chart_parameters",
        "router_parameters",
        "compiled_parameters",
    ):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError(f"resource {name} is invalid")
    if state["native_v_o_parameters"] <= 0:
        raise ValueError("native attention parameter count must be positive")
    reduction = state["parameter_reduction_fraction"]
    if (
        isinstance(reduction, bool)
        or not isinstance(reduction, (int, float))
        or not math.isfinite(float(reduction))
    ):
        raise ValueError("resource reduction fraction is invalid")
    for name in (
        "parameter_compression_eligible",
        "native_qk_retained",
        "native_output_rmsnorm_retained",
        "logical_speedup_claimed",
    ):
        if type(state[name]) is not bool:
            raise ValueError(f"resource {name} must be boolean")
    return state


def _validate_resource_accounting(
    state: Mapping[str, object],
    *,
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    routers: Sequence[ChartConditionedAttentionTransportRouter],
) -> dict[str, object]:
    result = _resource_state(state)
    native = int(result["native_v_o_parameters"])
    bank_count = bank.stored_parameter_count
    chart_count = atlas.stored_parameter_count
    router_count = sum(router.stored_parameter_count for router in routers)
    compiled = bank_count + chart_count + router_count
    expected_reduction = 1.0 - compiled / native
    if (
        result["generator_bank_parameters"] != bank_count
        or result["chart_parameters"] != chart_count
        or result["router_parameters"] != router_count
        or result["compiled_parameters"] != compiled
        or not math.isclose(
            float(result["parameter_reduction_fraction"]),
            expected_reduction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or result["parameter_compression_eligible"] is not (compiled < native)
        or result["native_qk_retained"] is not True
        or result["native_output_rmsnorm_retained"] is not True
        or result["logical_speedup_claimed"] is not False
    ):
        raise ValueError("attention transport resource accounting does not close")
    result["parameter_reduction_fraction"] = float(
        result["parameter_reduction_fraction"]
    )
    return result


def _exact_dict(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _require_machine_value(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a machine string")
    _assert_machine_string(value, path=tuple(label.split(".")))
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    return value


def _validate_resource_metadata(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be resource accounting")
    state = _resource_state(value)
    native = int(state["native_v_o_parameters"])
    compiled = int(state["compiled_parameters"])
    expected_compiled = sum(
        int(state[name])
        for name in (
            "generator_bank_parameters",
            "chart_parameters",
            "router_parameters",
        )
    )
    if (
        compiled != expected_compiled
        or not math.isclose(
            float(state["parameter_reduction_fraction"]),
            1.0 - compiled / native,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or state["parameter_compression_eligible"] is not (compiled < native)
        or state["native_qk_retained"] is not True
        or state["native_output_rmsnorm_retained"] is not True
        or state["logical_speedup_claimed"] is not False
    ):
        raise ValueError(f"{label} does not close")
    return state


def _validate_metric_payload(
    value: object,
    *,
    label: str,
    expected_rows: int | None = None,
) -> dict[str, object]:
    metric = _exact_dict(value, _METRIC_FIELDS, label=label)
    rows = _require_exact_int(metric["row_count"], label=f"{label}.row_count", minimum=1)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{label} row count drifted")
    for field in (
        "fisher_loss",
        "directional_mse",
        "euclidean_mse",
        "normalized_rmse",
    ):
        _require_exact_float(metric[field], label=f"{label}.{field}", minimum=0.0)
    cosine = _require_exact_float(metric["cosine"], label=f"{label}.cosine")
    if not -1.0 - 1.0e-10 <= cosine <= 1.0 + 1.0e-10:
        raise ValueError(f"{label}.cosine is outside [-1, 1]")
    return metric


def _validate_oracle_contract(
    value: object,
    *,
    label: str,
    active_count: int,
) -> None:
    contract = _exact_dict(value, _ORACLE_CONTRACT_FIELDS, label=label)
    if (
        contract["solver_kind"] != EXACT_SPARSE_ORACLE_SOLVER
        or contract["active_count_limit"] != active_count
        or type(contract["coefficient_ridge"]) is not float
        or contract["coefficient_ridge"] != HELDOUT_ORACLE_COEFFICIENT_RIDGE
        or contract["heldout_ridge_free"] is not True
    ):
        raise ValueError(f"{label} is not the held-out sparse-oracle contract")


def _validate_count_summary(
    value: object,
    *,
    label: str,
    fields: set[str],
    maximum_limit: int,
) -> dict[str, object]:
    summary = _exact_dict(value, fields, label=label)
    minimum = _require_exact_int(summary["minimum"], label=f"{label}.minimum")
    mean = _require_exact_float(summary["mean"], label=f"{label}.mean", minimum=0.0)
    maximum = _require_exact_int(summary["maximum"], label=f"{label}.maximum")
    if not minimum <= mean <= maximum <= maximum_limit:
        raise ValueError(f"{label} bounds are inconsistent")
    return summary


def _validate_usage(
    value: object,
    *,
    label: str,
    generator_count: int,
    dead_generator_count: object,
) -> None:
    if type(value) is not list or len(value) != generator_count:
        raise ValueError(f"{label} has the wrong generator count")
    usage = [
        _require_exact_float(item, label=f"{label}.{index}", minimum=0.0)
        for index, item in enumerate(value)
    ]
    if not math.isclose(sum(usage), 1.0, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise ValueError(f"{label} is not normalized")
    dead = _require_exact_int(
        dead_generator_count,
        label=f"{label}.dead_generator_count",
    )
    if dead != sum(item == 0.0 for item in usage):
        raise ValueError(f"{label} dead-generator count drifted")


def _validate_invocation_evaluation(
    value: object,
    *,
    label: str,
    generator_count: int,
    active_count: int,
) -> tuple[str, int]:
    evaluation = _exact_dict(value, _INVOCATION_EVALUATION_FIELDS, label=label)
    invocation_id = _require_machine_value(
        evaluation["invocation_id"],
        label=f"{label}.invocation_id",
    )
    operator = _validate_metric_payload(evaluation["operator"], label=f"{label}.operator")
    oracle = _validate_metric_payload(evaluation["oracle"], label=f"{label}.oracle")
    if operator["row_count"] != oracle["row_count"]:
        raise ValueError(f"{label} operator/oracle rows differ")
    _validate_oracle_contract(
        evaluation["oracle_contract"],
        label=f"{label}.oracle_contract",
        active_count=active_count,
    )
    zero = _require_exact_float(
        evaluation["zero_predictor_fisher_loss"],
        label=f"{label}.zero_predictor_fisher_loss",
        minimum=0.0,
    )
    relative = _require_exact_float(
        evaluation["relative_fisher_loss"],
        label=f"{label}.relative_fisher_loss",
        minimum=0.0,
    )
    oracle_relative = _require_exact_float(
        evaluation["oracle_relative_fisher_loss"],
        label=f"{label}.oracle_relative_fisher_loss",
        minimum=0.0,
    )
    gap = _require_exact_float(
        evaluation["oracle_router_fisher_gap"],
        label=f"{label}.oracle_router_fisher_gap",
    )
    if zero <= 0.0 or not (
        math.isclose(
            relative,
            float(operator["fisher_loss"]) / zero,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            oracle_relative,
            float(oracle["fisher_loss"]) / zero,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            gap,
            relative - oracle_relative,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"{label} relative Fisher metrics do not close")
    postnorm = _exact_dict(evaluation["postnorm"], _POSTNORM_FIELDS, label=f"{label}.postnorm")
    if postnorm["row_count"] != operator["row_count"]:
        raise ValueError(f"{label} postnorm rows drifted")
    _require_exact_float(
        postnorm["normalized_rmse"],
        label=f"{label}.postnorm.normalized_rmse",
        minimum=0.0,
    )
    _require_exact_float(postnorm["cosine"], label=f"{label}.postnorm.cosine")
    active = _validate_count_summary(
        evaluation["per_chart_active_count"],
        label=f"{label}.per_chart_active_count",
        fields=_COUNT_SUMMARY_FIELDS,
        maximum_limit=generator_count,
    )
    if (
        active["minimum"] != active_count
        or active["mean"] != float(active_count)
        or active["maximum"] != active_count
    ):
        raise ValueError(f"{label} per-chart TopK count drifted")
    support = _validate_count_summary(
        evaluation["effective_support_count"],
        label=f"{label}.effective_support_count",
        fields=_SUPPORT_SUMMARY_FIELDS,
        maximum_limit=generator_count,
    )
    fraction = _require_exact_float(
        support["fraction"],
        label=f"{label}.effective_support_count.fraction",
        minimum=0.0,
    )
    if not math.isclose(
        fraction,
        float(support["mean"]) / generator_count,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{label} support fraction drifted")
    _require_exact_float(
        evaluation["sparse_probability_renormalization_max_error"],
        label=f"{label}.sparse_probability_renormalization_max_error",
        minimum=0.0,
    )
    _validate_usage(
        evaluation["generator_usage"],
        label=f"{label}.generator_usage",
        generator_count=generator_count,
        dead_generator_count=evaluation["dead_generator_count"],
    )
    return invocation_id, int(operator["row_count"])


def _validate_aggregate_evaluation(
    value: object,
    *,
    label: str,
    role: str,
    split_sha256: str,
    generator_count: int,
    active_count: int,
    router_invocation_ids: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    evaluation = _exact_dict(value, _AGGREGATE_EVALUATION_FIELDS, label=label)
    if evaluation["role"] != role:
        raise ValueError(f"{label} role must be {role}")
    if _require_sha256(
        evaluation["split_artifact_sha256"],
        label=f"{label}.split_artifact_sha256",
    ) != split_sha256:
        raise ValueError(f"{label} split binding drifted")
    row_count = _require_exact_int(
        evaluation["row_count"],
        label=f"{label}.row_count",
        minimum=1,
    )
    if row_count != _EXPECTED_ROWS_PER_ROLE:
        raise ValueError(f"{label} row count drifted")
    invocation_count = _require_exact_int(
        evaluation["invocation_count"],
        label=f"{label}.invocation_count",
        minimum=1,
    )
    if invocation_count != len(router_invocation_ids):
        raise ValueError(f"{label} invocation count drifted")
    for field in (
        "fisher_loss",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
        "oracle_fisher_loss",
        "oracle_relative_fisher_loss",
        "operator_normalized_rmse",
        "postnorm_normalized_rmse",
        "sparse_probability_renormalization_max_error",
    ):
        _require_exact_float(evaluation[field], label=f"{label}.{field}", minimum=0.0)
    for field in ("oracle_router_fisher_gap", "operator_cosine", "postnorm_cosine"):
        _require_exact_float(evaluation[field], label=f"{label}.{field}")
    zero = float(evaluation["zero_predictor_fisher_loss"])
    if zero <= 0.0 or not (
        math.isclose(
            float(evaluation["relative_fisher_loss"]),
            float(evaluation["fisher_loss"]) / zero,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(evaluation["oracle_relative_fisher_loss"]),
            float(evaluation["oracle_fisher_loss"]) / zero,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(evaluation["oracle_router_fisher_gap"]),
            float(evaluation["relative_fisher_loss"])
            - float(evaluation["oracle_relative_fisher_loss"]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"{label} aggregate Fisher metrics do not close")
    _validate_oracle_contract(
        evaluation["oracle_contract"],
        label=f"{label}.oracle_contract",
        active_count=active_count,
    )
    active = _validate_count_summary(
        evaluation["per_chart_active_count"],
        label=f"{label}.per_chart_active_count",
        fields=_COUNT_SUMMARY_FIELDS,
        maximum_limit=generator_count,
    )
    if (
        active["minimum"] != active_count
        or active["mean"] != float(active_count)
        or active["maximum"] != active_count
    ):
        raise ValueError(f"{label} aggregate per-chart TopK count drifted")
    support = _validate_count_summary(
        evaluation["effective_support_count"],
        label=f"{label}.effective_support_count",
        fields=_SUPPORT_SUMMARY_FIELDS,
        maximum_limit=generator_count,
    )
    fraction = _require_exact_float(
        support["fraction"],
        label=f"{label}.effective_support_count.fraction",
        minimum=0.0,
    )
    if not math.isclose(
        fraction,
        float(support["mean"]) / generator_count,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{label} aggregate support fraction drifted")
    _validate_usage(
        evaluation["generator_usage"],
        label=f"{label}.generator_usage",
        generator_count=generator_count,
        dead_generator_count=evaluation["dead_generator_count"],
    )
    per_invocation = evaluation["per_invocation"]
    if type(per_invocation) is not list or len(per_invocation) != invocation_count:
        raise ValueError(f"{label} per-invocation catalog is invalid")
    observed = tuple(
        _validate_invocation_evaluation(
            item,
            label=f"{label}.per_invocation.{index}",
            generator_count=generator_count,
            active_count=active_count,
        )
        for index, item in enumerate(per_invocation)
    )
    if (
        tuple(item[0] for item in observed) != router_invocation_ids
        or sum(item[1] for item in observed) != row_count
    ):
        raise ValueError(f"{label} per-invocation lineage or rows drifted")
    resources = _validate_resource_metadata(
        evaluation["resources"],
        label=f"{label}.resources",
    )
    return evaluation, resources


def _validate_decision(
    value: object,
    *,
    label: str,
    role: str,
    split_sha256: str,
    config: Mapping[str, object],
    gates: Mapping[str, object],
    router_invocation_ids: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    decision = _exact_dict(value, _DECISION_FIELDS, label=label)
    if decision["candidate_id"] != config["candidate_id"]:
        raise ValueError(f"{label} candidate id drifted")
    _require_sha256(
        decision["candidate_artifact_sha256"],
        label=f"{label}.candidate_artifact_sha256",
    )
    _require_sha256(
        decision["evaluation_artifact_sha256"],
        label=f"{label}.evaluation_artifact_sha256",
    )
    evaluation, resources = _validate_aggregate_evaluation(
        decision["evaluation"],
        label=f"{label}.evaluation",
        role=role,
        split_sha256=split_sha256,
        generator_count=int(config["generator_count"]),
        active_count=int(config["active_count"]),
        router_invocation_ids=router_invocation_ids,
    )
    expected_flags = {
        "relative_fisher_pass": float(evaluation["relative_fisher_loss"])
        <= float(gates["max_relative_fisher_loss"]),
        "oracle_gap_pass": -float(gates["oracle_gap_numerical_tolerance"])
        <= float(evaluation["oracle_router_fisher_gap"])
        <= float(gates["max_oracle_router_fisher_gap"]),
        "postnorm_nrmse_pass": float(evaluation["postnorm_normalized_rmse"])
        <= float(gates["max_postnorm_normalized_rmse"]),
        "postnorm_cosine_pass": float(evaluation["postnorm_cosine"])
        >= float(gates["min_postnorm_cosine"]),
        "sparse_renormalization_pass": float(
            evaluation["sparse_probability_renormalization_max_error"]
        )
        <= float(gates["max_sparse_probability_renormalization_error"]),
        "effective_support_pass": float(
            evaluation["effective_support_count"]["fraction"]
        )
        <= float(gates["max_effective_support_fraction"]),
        "global_support_budget_pass": int(
            evaluation["oracle_contract"]["active_count_limit"]
        )
        == int(config["active_count"])
        and int(evaluation["effective_support_count"]["maximum"])
        <= int(config["active_count"]),
        "resource_pass": bool(resources["parameter_compression_eligible"])
        if gates["require_parameter_compression"] is True
        else True,
        "minimum_rows_pass": int(evaluation["row_count"])
        >= int(gates["minimum_aggregate_rows"]),
    }
    if any(type(decision[field]) is not bool for field in (*_DECISION_PASS_FIELDS, "passed")):
        raise ValueError(f"{label} pass flags must be boolean")
    if any(decision[field] is not expected for field, expected in expected_flags.items()):
        raise ValueError(f"{label} pass flags contradict its evaluation")
    if decision["passed"] is not all(expected_flags.values()):
        raise ValueError(f"{label} terminal pass flag is inconsistent")
    return decision, resources


def _validate_runtime_metadata(value: object) -> None:
    runtime = _exact_dict(value, _RUNTIME_FIELDS, label="evaluations.runtime")
    counters = _exact_dict(
        runtime["counters"],
        _RUNTIME_COUNTER_FIELDS,
        label="evaluations.runtime.counters",
    )
    if counters["role_batch_counts"] != _EXPECTED_ROLE_BATCH_COUNTS:
        raise ValueError("runtime role batch counts do not prove complete A/B/C")
    expected_exact_counts = {
        "chart_backbone_forward_count": 135,
        "chart_terminal_cotangent_vjp_count": 135,
        "dense_parity_smoke_count": 1,
        "parity_backbone_forward_count": 1,
        "parity_terminal_cotangent_vjp_count": 1,
        "canonical_gradient_targets_per_vjp": 73,
        "auxiliary_gradient_targets_per_vjp": 19,
        "gradient_targets_per_vjp": 92,
    }
    if any(counters[field] != expected for field, expected in expected_exact_counts.items()):
        raise ValueError("runtime exact execution counters drifted")
    _require_exact_int(
        counters["captured_example_count"],
        label="evaluations.runtime.counters.captured_example_count",
        minimum=1,
    )
    _require_exact_int(
        counters["supplemental_route_receipt_count"],
        label="evaluations.runtime.counters.supplemental_route_receipt_count",
    )
    for field in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations",
        "contains_raw_gradients",
        "contains_logits",
    ):
        if counters[field] is not False:
            raise ValueError(f"runtime counter safety field {field} must be false")
    peak = _require_exact_int(
        runtime["prepublication_peak_resident_set_bytes"],
        label="evaluations.runtime.prepublication_peak_resident_set_bytes",
        minimum=1,
    )
    cap = _require_exact_int(
        runtime["resident_set_cap_bytes"],
        label="evaluations.runtime.resident_set_cap_bytes",
        minimum=1,
    )
    if (
        peak > cap
        or runtime["publication_guarded"] is not True
        or runtime["logical_speedup_claimed"] is not False
    ):
        raise ValueError("runtime publication resource contract is invalid")


def _validate_success_metadata(
    *,
    model: object,
    protocol: object,
    splits: object,
    selection: object,
    evaluations: object,
    generator_count: int,
    generator_rank: int,
    hidden_width: int,
    chart_count: int,
    chart_rank: int,
    router_invocation_ids: tuple[str, ...],
    router_active_counts: tuple[int, ...],
    resource_accounting: Mapping[str, object],
) -> None:
    model_state = _exact_dict(model, _MODEL_FIELDS, label="model")
    _require_machine_value(model_state["model_id"], label="model.model_id")
    _require_machine_value(model_state["model_revision"], label="model.model_revision")
    if _require_exact_int(
        model_state["hidden_width"],
        label="model.hidden_width",
        minimum=1,
    ) != hidden_width:
        raise ValueError("model hidden width differs from executable bank")
    for field in (
        "runtime_qualification_receipt_sha256",
        "runtime_execution_receipt_sha256",
    ):
        _require_sha256(model_state[field], label=f"model.{field}")

    protocol_state = _exact_dict(protocol, _PROTOCOL_FIELDS, label="protocol")
    if (
        protocol_state["protocol_id"] != "all_length_uniform_row_canary_v1"
        or protocol_state["a_fit_only"] is not True
        or protocol_state["b_selection_only"] is not True
        or protocol_state["c_confirmation_only"] is not True
        or protocol_state["no_post_c_refit_or_selection"] is not True
        or protocol_state["rows_per_batch_and_invocation"]
        != _ROWS_PER_BATCH_AND_INVOCATION
        or protocol_state["expected_rows_per_role"] != _EXPECTED_ROWS_PER_ROLE
    ):
        raise ValueError("fixed A/B/C protocol contract drifted")
    configs = protocol_state["candidate_configs"]
    if type(configs) is not list or not configs:
        raise ValueError("protocol candidate configs must be a nonempty list")
    config_values: list[dict[str, object]] = []
    for index, value in enumerate(configs):
        config = _exact_dict(
            value,
            _CANDIDATE_CONFIG_FIELDS,
            label=f"protocol.candidate_configs.{index}",
        )
        _require_machine_value(config["candidate_id"], label=f"candidate.{index}.id")
        _require_sha256(
            config["config_artifact_sha256"],
            label=f"candidate.{index}.config_artifact_sha256",
        )
        for field in (
            "generator_count",
            "generator_rank",
            "bank_steps",
            "chart_count",
            "chart_rank",
            "chart_lloyd_iterations",
            "active_count",
            "router_steps",
        ):
            _require_exact_int(config[field], label=f"candidate.{index}.{field}", minimum=1)
        for field in (
            "bank_learning_rate",
            "euclidean_floor",
            "chart_temperature_scale",
            "router_learning_rate",
        ):
            _require_exact_float(config[field], label=f"candidate.{index}.{field}", minimum=0.0)
        if int(config["active_count"]) > int(config["generator_count"]):
            raise ValueError("candidate active count exceeds its generators")
        teacher = _exact_dict(
            config["router_training_oracle"],
            _TRAINING_ORACLE_FIELDS,
            label=f"candidate.{index}.router_training_oracle",
        )
        if (
            teacher["split"] != "A_only"
            or teacher["solver_kind"] != EXACT_SPARSE_ORACLE_SOLVER
            or type(teacher["coefficient_ridge"]) is not float
            or teacher["coefficient_ridge"] < 0.0
            or teacher["heldout_metric_used"] is not False
        ):
            raise ValueError("candidate router teacher contract is invalid")
        config_values.append(config)
    candidate_ids = [str(config["candidate_id"]) for config in config_values]
    if candidate_ids != sorted(set(candidate_ids)):
        raise ValueError("protocol candidate catalog is not canonical and unique")
    gates = _exact_dict(protocol_state["gates"], _GATE_FIELDS, label="protocol.gates")
    _require_sha256(gates["gates_artifact_sha256"], label="protocol.gates.artifact")
    for field in _GATE_FIELDS - {
        "gates_artifact_sha256",
        "require_parameter_compression",
        "minimum_aggregate_rows",
    }:
        _require_exact_float(gates[field], label=f"protocol.gates.{field}")
    if (
        gates["require_parameter_compression"] is not True
        or gates["minimum_aggregate_rows"] != _EXPECTED_ROWS_PER_ROLE
    ):
        raise ValueError("protocol gate invariants drifted")
    sources = _exact_dict(
        protocol_state["implementation_source_sha256s"],
        set(_IMPLEMENTATION_SOURCE_PATHS),
        label="protocol.implementation_source_sha256s",
    )
    for path, digest in sources.items():
        _require_sha256(digest, label=f"implementation source {path}")
    for field in (
        "preflight_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
    ):
        _require_sha256(protocol_state[field], label=f"protocol.{field}")

    split_state = _exact_dict(splits, _SPLIT_FIELDS, label="splits")
    split_digests = tuple(
        _require_sha256(split_state[f"{role}_split_artifact_sha256"], label=f"split {role}")
        for role in "ABC"
    )
    if len(set(split_digests)) != 3 or split_state["A_B_C_receipts_disjoint"] is not True:
        raise ValueError("A/B/C split identity contract is invalid")
    receipt_sets: list[set[str]] = []
    for role in "ABC":
        receipts = split_state[f"{role}_receipt_sha256s"]
        if type(receipts) is not list or not receipts:
            raise ValueError(f"split {role} receipt catalog is invalid")
        for index, digest in enumerate(receipts):
            _require_sha256(digest, label=f"split {role} receipt {index}")
        if receipts != sorted(set(receipts)):
            raise ValueError(f"split {role} receipt catalog is not canonical")
        receipt_sets.append(set(receipts))
    if any(receipt_sets[left] & receipt_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("A/B/C receipt catalogs are not disjoint")

    selection_state = _exact_dict(selection, _SELECTION_FIELDS, label="selection")
    selected_id = _require_machine_value(
        selection_state["selected_candidate_id"],
        label="selection.selected_candidate_id",
    )
    for field in (
        "selected_candidate_artifact_sha256",
        "selection_lock_artifact_sha256",
        "B_selection_artifact_sha256",
        "B_decision_artifact_sha256",
        "C_confirmation_artifact_sha256",
    ):
        _require_sha256(selection_state[field], label=f"selection.{field}")
    if (
        selection_state["B_passed"] is not True
        or type(selection_state["C_passed"]) is not bool
        or selection_state["selection_rule"] != _SELECTION_RULE
    ):
        raise ValueError("selection terminal contract is invalid")
    configs_by_id = {str(config["candidate_id"]): config for config in config_values}
    if selected_id not in configs_by_id:
        raise ValueError("selected candidate is absent from the frozen catalog")
    selected_config = configs_by_id[selected_id]
    if (
        int(selected_config["generator_count"]) != generator_count
        or int(selected_config["generator_rank"]) != generator_rank
        or int(selected_config["chart_count"]) != chart_count
        or int(selected_config["chart_rank"]) != chart_rank
        or not router_invocation_ids
        or router_invocation_ids != tuple(sorted(set(router_invocation_ids)))
        or any(
            value != int(selected_config["active_count"])
            for value in router_active_counts
        )
    ):
        raise ValueError("selected candidate does not bind the executable components")

    evaluation_state = _exact_dict(evaluations, _EVALUATION_FIELDS, label="evaluations")
    a_evaluation = _exact_dict(
        evaluation_state["A"],
        _A_EVALUATION_FIELDS,
        label="evaluations.A",
    )
    _validate_metric_payload(
        a_evaluation["initial_bank"],
        label="evaluations.A.initial_bank",
        expected_rows=_EXPECTED_ROWS_PER_ROLE,
    )
    _validate_metric_payload(
        a_evaluation["final_bank"],
        label="evaluations.A.final_bank",
        expected_rows=_EXPECTED_ROWS_PER_ROLE,
    )
    all_b = evaluation_state["all_B_candidates"]
    if type(all_b) is not list or len(all_b) != len(config_values):
        raise ValueError("all-B candidate decision catalog is invalid")
    validated_b: list[tuple[dict[str, object], dict[str, object]]] = []
    for index, (decision, config) in enumerate(zip(all_b, config_values, strict=True)):
        validated_b.append(
            _validate_decision(
                decision,
                label=f"evaluations.all_B_candidates.{index}",
                role="B",
                split_sha256=split_digests[1],
                config=config,
                gates=gates,
                router_invocation_ids=router_invocation_ids,
            )
        )
    selected_index = candidate_ids.index(selected_id)
    b_decision, b_resources = _validate_decision(
        evaluation_state["B"],
        label="evaluations.B",
        role="B",
        split_sha256=split_digests[1],
        config=selected_config,
        gates=gates,
        router_invocation_ids=router_invocation_ids,
    )
    if b_decision != validated_b[selected_index][0] or b_decision["passed"] is not True:
        raise ValueError("selected B decision does not bind the all-B catalog")
    passing = [
        (config, decision, resources)
        for config, (decision, resources) in zip(config_values, validated_b, strict=True)
        if decision["passed"] is True
    ]
    if not passing:
        raise ValueError("success artifact has no passing B candidate")
    expected_selected = min(
        passing,
        key=lambda item: (
            float(item[1]["evaluation"]["relative_fisher_loss"]),
            int(item[2]["compiled_parameters"]),
            int(item[0]["generator_count"]),
            int(item[0]["generator_rank"]),
            int(item[0]["active_count"]),
            str(item[0]["candidate_id"]),
        ),
    )[0]["candidate_id"]
    if expected_selected != selected_id:
        raise ValueError("selected candidate contradicts the frozen B selection rule")
    c_decision, c_resources = _validate_decision(
        evaluation_state["C"],
        label="evaluations.C",
        role="C",
        split_sha256=split_digests[2],
        config=selected_config,
        gates=gates,
        router_invocation_ids=router_invocation_ids,
    )
    selected_artifact = selection_state["selected_candidate_artifact_sha256"]
    if (
        b_decision["candidate_id"] != selected_id
        or c_decision["candidate_id"] != selected_id
        or b_decision["candidate_artifact_sha256"] != selected_artifact
        or c_decision["candidate_artifact_sha256"] != selected_artifact
        or selection_state["C_passed"] is not c_decision["passed"]
        or b_resources != dict(resource_accounting)
        or c_resources != dict(resource_accounting)
    ):
        raise ValueError("selected candidate B/C/component bindings are inconsistent")
    _validate_runtime_metadata(evaluation_state["runtime"])


def _authenticated_components(
    *,
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    routers: Sequence[ChartConditionedAttentionTransportRouter],
) -> tuple[
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    tuple[ChartConditionedAttentionTransportRouter, ...],
]:
    if not isinstance(bank, FisherAttentionTransportGeneratorBank):
        raise TypeError("bank must be a Fisher attention transport bank")
    if not isinstance(atlas, FrozenAttentionChartAtlas):
        raise TypeError("atlas must be a frozen attention chart atlas")
    if isinstance(routers, (str, bytes)) or not isinstance(routers, Sequence):
        raise TypeError("routers must be a materialized sequence")
    bank_copy = fisher_attention_transport_generator_bank_from_state_dict(
        bank.state_dict()
    )
    atlas_copy = frozen_attention_chart_atlas_from_state_dict(atlas.state_dict())
    router_values = tuple(routers)
    if not router_values or any(
        not isinstance(router, ChartConditionedAttentionTransportRouter)
        for router in router_values
    ):
        raise ValueError("routers must contain transport routers")
    restored = tuple(
        chart_conditioned_attention_transport_router_from_state_dict(
            router.state_dict(),
            atlas=atlas_copy,
        )
        for router in sorted(router_values, key=lambda item: item.invocation_id)
    )
    invocation_ids = tuple(router.invocation_id for router in restored)
    if len(set(invocation_ids)) != len(invocation_ids):
        raise ValueError("router invocation ids must be unique")
    if any(
        router.generator_bank_sha256 != bank_copy.artifact_sha256
        or router.atlas.artifact_sha256 != atlas_copy.artifact_sha256
        or router.generator_count != bank_copy.generator_count
        or atlas_copy.hidden_width != bank_copy.hidden_width
        or not any(
            head_id.startswith(f"{router.invocation_id}/query-head.")
            for head_id in bank_copy.head_ids
        )
        for router in restored
    ):
        raise ValueError("router, atlas, and generator bank lineage is invalid")
    return bank_copy, atlas_copy, restored


def build_attention_transport_payload(
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
    selection: Mapping[str, object],
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    routers: Sequence[ChartConditionedAttentionTransportRouter],
    evaluations: Mapping[str, object],
    resource_accounting: (
        AttentionTransportResourceAccounting | Mapping[str, object]
    ),
) -> dict[str, object]:
    """Build and fully authenticate one selected source-free payload."""

    for label, value in (
        ("model", model),
        ("protocol", protocol),
        ("splits", splits),
        ("selection", selection),
        ("evaluations", evaluations),
    ):
        _assert_metadata(value, label=label)
    bank_copy, atlas_copy, router_values = _authenticated_components(
        bank=bank,
        atlas=atlas,
        routers=routers,
    )
    resources = _validate_resource_accounting(
        _resource_state(resource_accounting),
        bank=bank_copy,
        atlas=atlas_copy,
        routers=router_values,
    )
    without_digest: dict[str, object] = {
        "schema": ATTENTION_TRANSPORT_ARTIFACT_SCHEMA,
        "format_version": ATTENTION_TRANSPORT_ARTIFACT_FORMAT_VERSION,
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "model": dict(model),
        "protocol": dict(protocol),
        "splits": dict(splits),
        "selection": dict(selection),
        "generator_bank": bank_copy.state_dict(),
        "atlas": atlas_copy.state_dict(),
        "routers": [router.state_dict() for router in router_values],
        "resource_accounting": resources,
        "evaluations": dict(evaluations),
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    payload = {
        **without_digest,
        "scientific_payload_sha256": _payload_sha256(without_digest),
    }
    _validate_and_restore_payload(payload)
    return payload


def _validate_and_restore_payload(
    raw: Mapping[str, object],
) -> tuple[
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    tuple[ChartConditionedAttentionTransportRouter, ...],
]:
    if type(raw) is not dict or set(raw) != _PAYLOAD_FIELDS:
        raise ValueError("attention transport payload fields are invalid")
    if (
        raw["schema"] != ATTENTION_TRANSPORT_ARTIFACT_SCHEMA
        or raw["format_version"]
        != ATTENTION_TRANSPORT_ARTIFACT_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("attention transport payload envelope is invalid")
    for label in ("model", "protocol", "splits", "selection", "evaluations"):
        _assert_metadata(raw[label], label=label)
    _assert_source_safe(raw)
    observed = _require_sha256(
        raw["scientific_payload_sha256"],
        label="scientific_payload_sha256",
    )
    unsigned = dict(raw)
    unsigned.pop("scientific_payload_sha256")
    if observed != _payload_sha256(unsigned):
        raise ValueError("attention transport scientific payload digest drifted")
    bank_state = raw["generator_bank"]
    atlas_state = raw["atlas"]
    router_states = raw["routers"]
    if (
        not isinstance(bank_state, Mapping)
        or not isinstance(atlas_state, Mapping)
        or type(router_states) is not list
        or not router_states
        or any(not isinstance(state, Mapping) for state in router_states)
    ):
        raise ValueError("attention transport component states are invalid")
    bank = fisher_attention_transport_generator_bank_from_state_dict(bank_state)
    atlas = frozen_attention_chart_atlas_from_state_dict(atlas_state)
    routers = tuple(
        chart_conditioned_attention_transport_router_from_state_dict(
            state,
            atlas=atlas,
        )
        for state in router_states
    )
    if tuple(router.invocation_id for router in routers) != tuple(
        sorted(router.invocation_id for router in routers)
    ):
        raise ValueError("attention transport router catalog is not canonical")
    authenticated = _authenticated_components(
        bank=bank,
        atlas=atlas,
        routers=routers,
    )
    resources = _validate_resource_accounting(
        raw["resource_accounting"],
        bank=authenticated[0],
        atlas=authenticated[1],
        routers=authenticated[2],
    )
    _validate_success_metadata(
        model=raw["model"],
        protocol=raw["protocol"],
        splits=raw["splits"],
        selection=raw["selection"],
        evaluations=raw["evaluations"],
        generator_count=authenticated[0].generator_count,
        generator_rank=authenticated[0].generator_rank,
        hidden_width=authenticated[0].hidden_width,
        chart_count=authenticated[1].chart_count,
        chart_rank=authenticated[1].chart_rank,
        router_invocation_ids=tuple(
            router.invocation_id for router in authenticated[2]
        ),
        router_active_counts=tuple(
            router.active_count for router in authenticated[2]
        ),
        resource_accounting=resources,
    )
    return authenticated


def _bank_summary(bank: FisherAttentionTransportGeneratorBank) -> dict[str, object]:
    return {
        "artifact_sha256": bank.artifact_sha256,
        "generator_count": bank.generator_count,
        "generator_rank": bank.generator_rank,
        "head_count": len(bank.head_ids),
        "hidden_width": bank.hidden_width,
        "stored_parameter_count": bank.stored_parameter_count,
    }


def _atlas_summary(atlas: FrozenAttentionChartAtlas) -> dict[str, object]:
    return {
        "artifact_sha256": atlas.artifact_sha256,
        "chart_count": atlas.chart_count,
        "chart_rank": atlas.chart_rank,
        "hidden_width": atlas.hidden_width,
        "stored_parameter_count": atlas.stored_parameter_count,
        "temperature": float(atlas.temperature),
    }


def _router_summary(
    router: ChartConditionedAttentionTransportRouter,
) -> dict[str, object]:
    return {
        "active_count": router.active_count,
        "artifact_sha256": router.artifact_sha256,
        "atlas_sha256": router.atlas.artifact_sha256,
        "generator_bank_sha256": router.generator_bank_sha256,
        "generator_count": router.generator_count,
        "invocation_id": router.invocation_id,
        "stored_parameter_count": router.stored_parameter_count,
    }


def _require_tensor_file_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or not value.endswith(".pt")
    ):
        raise ValueError("tensor file must be a safe .pt basename")
    _assert_machine_string(value, path=("artifact", "tensor_file"))
    return value


def build_attention_transport_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
    tensor_file_sha256: str,
) -> dict[str, object]:
    """Build a compact canonical tensor-free report for one strict payload."""

    bank, atlas, routers = _validate_and_restore_payload(payload)
    file_name = _require_tensor_file_name(tensor_file)
    file_sha = _require_sha256(tensor_file_sha256, label="tensor_file_sha256")
    without_digest: dict[str, object] = {
        "schema": ATTENTION_TRANSPORT_REPORT_SCHEMA,
        "format_version": ATTENTION_TRANSPORT_REPORT_FORMAT_VERSION,
        "scientific_status": raw_copy(payload["scientific_status"]),
        "model": raw_copy(payload["model"]),
        "protocol": raw_copy(payload["protocol"]),
        "splits": raw_copy(payload["splits"]),
        "selection": raw_copy(payload["selection"]),
        "generator_bank": _bank_summary(bank),
        "atlas": _atlas_summary(atlas),
        "routers": [_router_summary(router) for router in routers],
        "resource_accounting": raw_copy(payload["resource_accounting"]),
        "evaluations": raw_copy(payload["evaluations"]),
        "safety": dict(_SAFETY),
        "artifact": {
            "tensor_file": file_name,
            "tensor_file_sha256": file_sha,
            "scientific_payload_sha256": payload[
                "scientific_payload_sha256"
            ],
        },
    }
    _assert_source_safe(without_digest, tensors_allowed=False)
    report = {
        **without_digest,
        "report_sha256": _report_sha256(without_digest),
    }
    _validate_report(report)
    return report


def raw_copy(value: object) -> object:
    """Copy supported JSON-like metadata without accepting arbitrary objects."""

    if isinstance(value, Mapping):
        return {str(key): raw_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [raw_copy(child) for child in value]
    if isinstance(value, tuple):
        return [raw_copy(child) for child in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("report metadata contains a non-JSON value")


def _validate_report(raw: Mapping[str, object]) -> dict[str, object]:
    if type(raw) is not dict or set(raw) != _REPORT_FIELDS:
        raise ValueError("attention transport report fields are invalid")
    if (
        raw["schema"] != ATTENTION_TRANSPORT_REPORT_SCHEMA
        or raw["format_version"] != ATTENTION_TRANSPORT_REPORT_FORMAT_VERSION
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("attention transport report envelope is invalid")
    _assert_source_safe(raw, tensors_allowed=False)
    for label in ("model", "protocol", "splits", "selection", "evaluations"):
        _assert_metadata(raw[label], label=label)
    observed = _require_sha256(raw["report_sha256"], label="report_sha256")
    unsigned = dict(raw)
    unsigned.pop("report_sha256")
    if observed != _report_sha256(unsigned):
        raise ValueError("attention transport report digest drifted")
    artifact = raw["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "tensor_file",
        "tensor_file_sha256",
        "scientific_payload_sha256",
    }:
        raise ValueError("attention transport report artifact binding is invalid")
    _require_tensor_file_name(artifact["tensor_file"])
    _require_sha256(artifact["tensor_file_sha256"], label="tensor_file_sha256")
    _require_sha256(
        artifact["scientific_payload_sha256"],
        label="scientific_payload_sha256",
    )
    bank = raw["generator_bank"]
    atlas = raw["atlas"]
    routers = raw["routers"]
    resources = _resource_state(raw["resource_accounting"])
    if (
        not isinstance(bank, Mapping)
        or set(bank)
        != {
            "artifact_sha256",
            "generator_count",
            "generator_rank",
            "head_count",
            "hidden_width",
            "stored_parameter_count",
        }
        or not isinstance(atlas, Mapping)
        or set(atlas)
        != {
            "artifact_sha256",
            "chart_count",
            "chart_rank",
            "hidden_width",
            "stored_parameter_count",
            "temperature",
        }
        or type(routers) is not list
        or not routers
    ):
        raise ValueError("attention transport report summaries are invalid")
    for digest, label in (
        (bank["artifact_sha256"], "bank artifact_sha256"),
        (atlas["artifact_sha256"], "atlas artifact_sha256"),
    ):
        _require_sha256(digest, label=label)
    expected_router_fields = {
        "active_count",
        "artifact_sha256",
        "atlas_sha256",
        "generator_bank_sha256",
        "generator_count",
        "invocation_id",
        "stored_parameter_count",
    }
    invocation_ids: list[str] = []
    router_parameters = 0
    for router in routers:
        if not isinstance(router, Mapping) or set(router) != expected_router_fields:
            raise ValueError("attention transport router report is invalid")
        for field in (
            "artifact_sha256",
            "atlas_sha256",
            "generator_bank_sha256",
        ):
            _require_sha256(router[field], label=f"router {field}")
        if (
            router["atlas_sha256"] != atlas["artifact_sha256"]
            or router["generator_bank_sha256"] != bank["artifact_sha256"]
            or router["generator_count"] != bank["generator_count"]
            or type(router["active_count"]) is not int
            or router["active_count"] <= 0
            or not isinstance(router["invocation_id"], str)
            or not router["invocation_id"]
            or type(router["stored_parameter_count"]) is not int
            or router["stored_parameter_count"] <= 0
        ):
            raise ValueError("attention transport router report lineage drifted")
        invocation_ids.append(str(router["invocation_id"]))
        router_parameters += int(router["stored_parameter_count"])
    if invocation_ids != sorted(set(invocation_ids)):
        raise ValueError("attention transport report router order is invalid")
    integer_summary_fields = (
        *((bank, name) for name in (
            "generator_count",
            "generator_rank",
            "head_count",
            "hidden_width",
            "stored_parameter_count",
        )),
        *((atlas, name) for name in (
            "chart_count",
            "chart_rank",
            "hidden_width",
            "stored_parameter_count",
        )),
    )
    if any(type(mapping[name]) is not int or mapping[name] <= 0 for mapping, name in integer_summary_fields):
        raise ValueError("attention transport report dimensions are invalid")
    if (
        bank["hidden_width"] != atlas["hidden_width"]
        or resources["generator_bank_parameters"]
        != bank["stored_parameter_count"]
        or resources["chart_parameters"] != atlas["stored_parameter_count"]
        or resources["router_parameters"] != router_parameters
        or resources["compiled_parameters"]
        != (
            int(bank["stored_parameter_count"])
            + int(atlas["stored_parameter_count"])
            + router_parameters
        )
    ):
        raise ValueError("attention transport report resource summary drifted")
    _validate_success_metadata(
        model=raw["model"],
        protocol=raw["protocol"],
        splits=raw["splits"],
        selection=raw["selection"],
        evaluations=raw["evaluations"],
        generator_count=int(bank["generator_count"]),
        generator_rank=int(bank["generator_rank"]),
        hidden_width=int(bank["hidden_width"]),
        chart_count=int(atlas["chart_count"]),
        chart_rank=int(atlas["chart_rank"]),
        router_invocation_ids=tuple(invocation_ids),
        router_active_counts=tuple(int(router["active_count"]) for router in routers),
        resource_accounting=resources,
    )
    return dict(raw)


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate attention report key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite attention report constant: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("attention transport report JSON is invalid") from error


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} identity or size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_payload_bytes(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError("attention transport artifact byte size is invalid")
    try:
        value = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("attention transport tensor artifact is invalid") from error
    if type(value) is not dict:
        raise TypeError("attention transport artifact must be an exact dict")
    _validate_and_restore_payload(value)
    return value


def load_attention_transport_artifact(path: Path | str) -> dict[str, object]:
    """Strict CPU/weights-only load of one executable transport payload."""

    source = Path(path)
    raw = _read_regular_bytes(
        source,
        maximum_bytes=_MAX_ARTIFACT_BYTES,
        label="attention transport artifact",
    )
    return _load_payload_bytes(raw)


def _strict_load_report_bytes(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_REPORT_BYTES:
        raise ValueError("attention transport report byte size is invalid")
    value = _strict_json_loads(raw)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError("attention transport report must be canonical JSON")
    return _validate_report(value)


def strict_load_attention_transport_report(
    report_path: Path | str,
    *,
    artifact_path: Path | str | None = None,
) -> dict[str, object]:
    """Strict-load and cross-authenticate report and executable artifact."""

    report_source = Path(report_path)
    report_raw = _read_regular_bytes(
        report_source,
        maximum_bytes=_MAX_REPORT_BYTES,
        label="attention transport report",
    )
    report = _strict_load_report_bytes(report_raw)
    binding = report["artifact"]
    if not isinstance(binding, Mapping):
        raise AssertionError("validated report lost artifact binding")
    tensor_name = _require_tensor_file_name(binding["tensor_file"])
    source = (
        report_source.parent / tensor_name
        if artifact_path is None
        else Path(artifact_path)
    )
    if source.parent != report_source.parent or source.name != tensor_name:
        raise ValueError("attention transport artifact/report paths differ")
    tensor_raw = _read_regular_bytes(
        source,
        maximum_bytes=_MAX_ARTIFACT_BYTES,
        label="attention transport artifact",
    )
    tensor_sha = hashlib.sha256(tensor_raw).hexdigest()
    if tensor_sha != binding["tensor_file_sha256"]:
        raise ValueError("attention transport tensor file digest drifted")
    payload = _load_payload_bytes(tensor_raw)
    expected = build_attention_transport_report(
        payload,
        tensor_file=tensor_name,
        tensor_file_sha256=tensor_sha,
    )
    if canonical_json_bytes(report) != canonical_json_bytes(expected):
        raise ValueError("attention transport report differs from tensor artifact")
    return report


def _require_output_parent(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError("attention artifact output parent is absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("attention artifact output parent must be a real directory")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_tensor_temporary(
    destination: Path,
    payload: Mapping[str, object],
) -> tuple[Path, tuple[int, int]]:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    owner = _inode_identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if _inode_identity(temporary.lstat()) != owner:
            raise RuntimeError("attention tensor temporary was replaced")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if _inode_identity(temporary.lstat()) == owner:
                temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary, owner


def _write_report_temporary(
    destination: Path,
    raw: bytes,
) -> tuple[Path, tuple[int, int]]:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    owner = _inode_identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if _inode_identity(temporary.lstat()) != owner:
            raise RuntimeError("attention report temporary was replaced")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if _inode_identity(temporary.lstat()) == owner:
                temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary, owner


def _link_temporary(
    temporary: Path,
    destination: Path,
    *,
    owner: tuple[int, int],
) -> None:
    if _inode_identity(temporary.lstat()) != owner:
        raise RuntimeError("attention artifact temporary changed before publication")
    os.link(temporary, destination, follow_symlinks=False)
    if (
        _inode_identity(destination.lstat()) != owner
        or _inode_identity(temporary.lstat()) != owner
    ):
        raise RuntimeError("attention artifact publication identity drifted")
    temporary.unlink()
    if _inode_identity(destination.lstat()) != owner:
        raise RuntimeError("attention artifact target changed after publication")


def _unlink_if_owned(path: Path, owner: tuple[int, int] | None) -> None:
    if owner is None:
        return
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(info.st_mode) and _inode_identity(info) == owner:
        path.unlink()


def save_attention_transport_artifact(
    path: Path | str,
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
    selection: Mapping[str, object],
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    routers: Sequence[ChartConditionedAttentionTransportRouter],
    evaluations: Mapping[str, object],
    resource_accounting: (
        AttentionTransportResourceAccounting | Mapping[str, object]
    ),
    report_path: Path | str | None = None,
    publication_guard: Callable[[str], None],
) -> dict[str, object]:
    """Atomically publish tensor artifact first and canonical report last."""

    output = Path(path)
    report_output = (
        output.with_suffix(".json") if report_path is None else Path(report_path)
    )
    if (
        output.suffix != ".pt"
        or report_output.suffix != ".json"
        or output.parent != report_output.parent
        or output == report_output
    ):
        raise ValueError("attention artifact outputs must be sibling .pt/.json files")
    if not callable(publication_guard):
        raise TypeError("publication_guard must be callable")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_output_parent(output.parent)
    if (
        output.exists()
        or output.is_symlink()
        or report_output.exists()
        or report_output.is_symlink()
    ):
        raise FileExistsError("refusing to overwrite attention transport artifact")

    payload = build_attention_transport_payload(
        model=model,
        protocol=protocol,
        splits=splits,
        selection=selection,
        bank=bank,
        atlas=atlas,
        routers=routers,
        evaluations=evaluations,
        resource_accounting=resource_accounting,
    )
    tensor_temp: Path | None = None
    report_temp: Path | None = None
    tensor_owner: tuple[int, int] | None = None
    report_owner: tuple[int, int] | None = None
    tensor_attempted = False
    report_attempted = False
    primary: BaseException | None = None
    report: dict[str, object] | None = None
    try:
        tensor_temp, tensor_owner = _write_tensor_temporary(output, payload)
        tensor_raw = _read_regular_bytes(
            tensor_temp,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
            label="attention tensor temporary",
        )
        tensor_sha = hashlib.sha256(tensor_raw).hexdigest()
        if _load_payload_bytes(tensor_raw)["scientific_payload_sha256"] != payload[
            "scientific_payload_sha256"
        ]:
            raise RuntimeError("attention tensor temporary strict reload drifted")
        report = build_attention_transport_report(
            payload,
            tensor_file=output.name,
            tensor_file_sha256=tensor_sha,
        )
        report_raw = canonical_json_bytes(report)
        _strict_load_report_bytes(report_raw)
        report_temp, report_owner = _write_report_temporary(
            report_output,
            report_raw,
        )
        publication_guard("artifact_temporaries_sealed")

        tensor_attempted = True
        _link_temporary(tensor_temp, output, owner=tensor_owner)
        tensor_temp = None
        _fsync_directory(output.parent)
        loaded_payload = load_attention_transport_artifact(output)
        if loaded_payload["scientific_payload_sha256"] != payload[
            "scientific_payload_sha256"
        ]:
            raise RuntimeError("published attention tensor strict reload drifted")

        # The canonical report is the commit marker and is always linked last.
        report_attempted = True
        _link_temporary(report_temp, report_output, owner=report_owner)
        report_temp = None
        _fsync_directory(report_output.parent)
        loaded_report = strict_load_attention_transport_report(
            report_output,
            artifact_path=output,
        )
        if canonical_json_bytes(loaded_report) != report_raw:
            raise RuntimeError("published attention report strict reload drifted")
        publication_guard("artifact_publication_verified")
    except BaseException as error:
        primary = error
        cleanup_errors: list[BaseException] = []
        if report_attempted:
            try:
                _unlink_if_owned(report_output, report_owner)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            _fsync_directory(report_output.parent)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if tensor_attempted:
            try:
                _unlink_if_owned(output, tensor_owner)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            _fsync_directory(output.parent)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        for cleanup_error in cleanup_errors:
            error.add_note(
                "attention publication cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
    finally:
        temporary_cleanup_errors: list[BaseException] = []
        for temporary, owner in (
            (report_temp, report_owner),
            (tensor_temp, tensor_owner),
        ):
            if temporary is None:
                continue
            try:
                _unlink_if_owned(temporary, owner)
            except BaseException as cleanup_error:
                temporary_cleanup_errors.append(cleanup_error)
        if temporary_cleanup_errors:
            try:
                _fsync_directory(output.parent)
            except BaseException as cleanup_error:
                temporary_cleanup_errors.append(cleanup_error)
        if primary is not None:
            for cleanup_error in temporary_cleanup_errors:
                primary.add_note(
                    "attention temporary cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        elif temporary_cleanup_errors:
            primary = temporary_cleanup_errors[0]
            for cleanup_error in temporary_cleanup_errors[1:]:
                primary.add_note(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
    if primary is not None:
        raise primary
    if report is None:
        raise AssertionError("attention artifact publication produced no report")
    return report


__all__ = [
    "ATTENTION_TRANSPORT_ARTIFACT_FORMAT_VERSION",
    "ATTENTION_TRANSPORT_ARTIFACT_SCHEMA",
    "ATTENTION_TRANSPORT_REPORT_FORMAT_VERSION",
    "ATTENTION_TRANSPORT_REPORT_SCHEMA",
    "build_attention_transport_payload",
    "build_attention_transport_report",
    "canonical_json_bytes",
    "load_attention_transport_artifact",
    "save_attention_transport_artifact",
    "strict_load_attention_transport_report",
]
