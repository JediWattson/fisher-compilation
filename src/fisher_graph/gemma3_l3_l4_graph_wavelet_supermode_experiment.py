"""Fit-only graph-wavelet supermode development rung for Gemma L3 -> L4.

The experiment reconstructs the authenticated signed graph-wavelet GOMP
rank-64 parent basis, then freezes several rank-45..52 pruning and merging
paths using origins 8/24/40 only.  Every conditional target refit and fit
evaluation is completed before response values at origins 16/32 are
validated or read.

This is an open-development structural-response experiment.  It performs no
model execution, reads no prompts or token ids, and makes no latency,
natural-prompt, NLL, whole-model, or deployment claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from types import MappingProxyType

import torch
from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorAccounting,
    ConditionalSpectralGeneratorPlan,
    evaluate_conditional_spectral_generator,
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    SELECTION_ORIGINS,
    Gemma3SpectralSource,
    _file_sha256,
    _reserve_outputs,
    _response_binding,
    _stage_json,
    _stage_torch,
    _validate_output_path,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    DIFFUSION_SCALES,
    EIGENVALUE_TOLERANCE,
    EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256,
    Gemma3GraphWaveletCandidate,
    _floating_tensor_geometry,
    _float_tensor,
    _laplacian,
    _tensor_sha256 as _parent_tensor_sha256,
    load_gemma3_graph_wavelet_candidate,
)
from .graph_spectral_source_basis import fit_graph_source_bases
from .graph_wavelet_analysis import (
    build_spectral_graph_wavelet_frame,
    fit_graph_wavelet_omp_subspace,
)
from .graph_wavelet_supermode_merge import (
    FitOnlyGraphWaveletSupermodePath,
    GraphWaveletSupermodePair,
    _ACTION_POLICY as _PATH_ACTION_POLICY,
    _ALGORITHM as _PATH_ALGORITHM,
    _ARTIFACT_KIND as _PATH_ARTIFACT_KIND,
    _FIT_SCOPE as _PATH_FIT_SCOPE,
    _LOFO_SEMANTICS as _PATH_LOFO_SEMANTICS,
    _ONE_HOT_CONTROL_SEMANTICS as _PATH_ONE_HOT_CONTROL_SEMANTICS,
    _PAIR_SEMANTICS as _PATH_PAIR_SEMANTICS,
    _TOPOLOGY_SEMANTICS as _PATH_TOPOLOGY_SEMANTICS,
    _action_from_metadata as _path_action_from_metadata,
    _json_sha256 as _path_json_sha256,
    fit_graph_wavelet_supermode_merge,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "Gemma3GraphWaveletSupermodeCandidate",
    "analyze_gemma3_l3_l4_graph_wavelet_supermodes",
    "build_parser",
    "compile_gemma3_graph_wavelet_supermode_candidate",
    "describe_gemma3_l3_l4_graph_wavelet_supermodes",
    "load_gemma3_graph_wavelet_supermode_candidate",
    "main",
]


DEFAULT_PARENT_ARTIFACT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-dev-v2.pt"
)
DEFAULT_PARENT_ARTIFACT_SHA256 = (
    "7659dbbc2547f0222f4ee8eb28b587b4d05a5dce0fa403f1801975083b8914f2"
)
DEFAULT_PARENT_TENSOR_FILE_SHA256 = (
    "cba8148ae17aceac95e975a4593335b96cc011977d65285d43d091454ade5d3d"
)
DEFAULT_PARENT_REPORT_SHA256 = (
    "3d81cec7d34e97872266ef0c019257c55c5ff231a776f6cfb9426d8d70fb858c"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-supermode-dev-v1.pt"
)

PARENT_RANK = 64
RANKS = tuple(range(45, 53))
PERMUTED_GRAPH_LOCAL_SEEDS = (1729, 3253, 4513, 6421)
MINIMUM_SQUARED_LOADING = 0.10
MINIMUM_LOFO_LOADING_STABILITY = 0.90
TOPOLOGY_TOP_K = 8
MAXIMUM_SELECTION_RELATIVE_ERROR = 0.20
MINIMUM_SELECTION_COSINE = 0.98
MINIMUM_COMPILED_PLAN_REDUCTION = 0.20
MINIMUM_PERMUTED_CONTROL_WINS = 3
MINIMUM_SSE_RECOVERY_OVER_GOMP = 0.05

PERMUTED_METHODS = tuple(
    f"permuted_graph_local_seed_{seed}"
    for seed in PERMUTED_GRAPH_LOCAL_SEEDS
)
METHOD_ORDER = (
    "gomp_prefix",
    "diagonal_energy_prune",
    "signed_gfa_prefix",
    "signed_gfa_fit_energy",
    "fit_svd",
    "response_only_merge",
    "graph_local_merge",
    "graph_local_one_hot",
    *PERMUTED_METHODS,
)
PATH_METHODS = (
    "response_only_merge",
    "graph_local_merge",
    *PERMUTED_METHODS,
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_wavelet_supermode_development"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-graph-wavelet-supermode:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-graph-wavelet-supermode-report:v1\0"
)
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-graph-wavelet-supermode-tensor:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_raw_response_tensors": False,
    "contains_compiled_plan_tensors": False,
    "contains_parent_basis_tensor": False,
    "contains_supermode_path_tensors": False,
    "metadata_only_candidate": True,
    "artifact_must_remain_outside_git": True,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(tuple(int(v) for v in tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be a string-keyed mapping")
    normalized = json.loads(
        _canonical_json_bytes(dict(value)).decode("ascii")
    )
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must normalize to an object")
    return MappingProxyType(normalized)


def _canonical_rows(
    value: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("rate rows must be a sequence")
    return tuple(
        _canonical_mapping(row, label="rate row")
        for row in value
    )


def _exact_int_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an integer sequence")
    result = tuple(value)
    if any(type(item) is not int for item in result):
        raise ValueError(f"{label} must be an integer sequence")
    return result  # type: ignore[return-value]


def _evaluation_metadata(value: object) -> dict[str, object]:
    metadata = getattr(value, "metadata", None)
    if not callable(metadata):
        raise TypeError("conditional evaluation lacks metadata")
    result = metadata()
    if not isinstance(result, Mapping):
        raise TypeError("conditional evaluation metadata is invalid")
    return dict(result)


def _basis_locality(basis: Tensor, laplacian: Tensor) -> dict[str, object]:
    probability = basis.square()
    effective_support = 1.0 / probability.square().sum(dim=0)
    graph_variation = torch.diagonal(basis.T @ laplacian @ basis)
    return {
        "column_count": int(basis.shape[1]),
        "mean_effective_node_support": float(effective_support.mean()),
        "maximum_effective_node_support": float(effective_support.max()),
        "mean_signed_graph_quadratic_form": float(graph_variation.mean()),
        "maximum_signed_graph_quadratic_form": float(graph_variation.max()),
        "hop_locality_claim": False,
    }


def _fit_energy_ordered_basis(
    basis: Tensor,
    weighted_fit_response: Tensor,
) -> tuple[Tensor, tuple[int, ...]]:
    coefficients = basis.T @ weighted_fit_response.reshape(
        weighted_fit_response.shape[0],
        -1,
    )
    energy = coefficients.square().sum(dim=1)
    order = tuple(
        sorted(
            range(basis.shape[1]),
            key=lambda index: (-float(energy[index]), index),
        )
    )
    indices = torch.tensor(order, dtype=torch.int64)
    return basis.index_select(1, indices).contiguous(), order


_SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "tensor_file_sha256",
        "report_file_sha256",
        "report_payload_sha256",
        "mapping_artifact_sha256",
        "response_artifact_sha256",
        "source_model_sha256",
    }
)
_PARENT_RECEIPT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "source_receipt",
        "graph_artifact_sha256",
        "signed_frame_artifact_sha256",
        "signed_gomp_subspace_artifact_sha256",
        "q64_source_basis_sha256",
        "q64_selected_packet_order_sha256",
    }
)
_PROTOCOL_FIELDS = frozenset(
    {
        "experiment_variant",
        "response_component",
        "response_tensor_order",
        "measured_origins",
        "fit_origins",
        "selection_origins",
        "source_modes",
        "target_modes",
        "lag_count",
        "fft_length",
        "parent_rank",
        "ranks",
        "method_order",
        "primary_method",
        "permuted_graph_local_seeds",
        "minimum_squared_loading",
        "minimum_lofo_loading_stability",
        "topology_top_k",
        "all_paths_bases_and_plans_frozen_before_selection_read",
        "target_rank_is_full",
        "selection_is_open_development_not_confirmation",
    }
)
_PATH_HASH_PAYLOAD_FIELDS = (
    "artifact_kind",
    "format_version",
    "algorithm",
    "method",
    "minimum_rank",
    "parent_rank",
    "node_count",
    "minimum_squared_loading",
    "minimum_lofo_loading_stability",
    "topology_top_k",
    "permutation_seed",
    "fit_response_shape",
    "fit_response_sha256",
    "fit_fold_response_shapes",
    "fit_fold_response_sha256s",
    "signed_laplacian_sha256",
    "parent_basis_sha256",
    "response_gram_sha256",
    "fit_fold_response_gram_sha256s",
    "native_topology_matrix_sha256",
    "selection_topology_matrix_sha256",
    "topology_permutation_sha256",
    "selected_actions",
    "heldout_input_used",
    "pair_semantics",
    "action_policy",
    "topology_semantics",
    "lofo_semantics",
    "one_hot_control_semantics",
    "fit_scope",
)
_PATH_REPORT_FIELDS = frozenset(
    {
        *_PATH_HASH_PAYLOAD_FIELDS,
        "artifact_sha256",
        "available_ranks",
        "merge_action_count",
        "singleton_prune_action_count",
        "action_diagnostics",
        "rank_path",
    }
)
_PATH_RANK_ROW_FIELDS = frozenset(
    {
        "rank",
        "active_action_count",
        "active_merge_count",
        "active_singleton_prune_count",
        "mixed_basis_sha256",
        "one_hot_control_basis_sha256",
        "cumulative_action_loss",
        "mean_merge_relative_response_loss",
        "mean_merge_mixing_participation",
        "minimum_merge_lofo_loading_stability",
        "mean_merge_topology_interaction",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "plan_sha256",
        "response_binding_sha256",
        "evaluation_origins",
        "fit_origin_overlap",
        "weighted_target_frobenius",
        "weighted_residual_frobenius",
        "weighted_relative_error",
        "weighted_cosine",
        "per_origin_weighted_relative_errors",
        "per_origin_weighted_cosines",
        "fit_was_not_recomputed",
    }
)
_COEFFICIENT_PAYLOAD_FIELDS = frozenset(
    {
        "source_basis_float64_scalars",
        "target_basis_float64_scalars",
        "fit_knot_core_float64_scalars",
        "compiled_plan_float64_scalars",
        "compiled_plan_coefficient_fraction_of_full_rank",
        "prepared_runtime_storage_bytes",
        "prepared_runtime_storage_fraction_of_full_rank",
        "storage_gate_uses_complete_prepared_runtime_bytes",
        "equal_rank_payload_match",
        "basis_construction_metadata_excluded_from_plan_payload",
    }
)
_BASIS_LOCALITY_FIELDS = frozenset(
    {
        "column_count",
        "mean_effective_node_support",
        "maximum_effective_node_support",
        "mean_signed_graph_quadratic_form",
        "maximum_signed_graph_quadratic_form",
        "hop_locality_claim",
    }
)
_PATH_ROW_FIELDS = frozenset(
    {
        "path_artifact_sha256",
        "active_action_count",
        "active_selected_pair_action_count",
        "active_genuine_merge_count",
        "active_paired_delete_count",
        "active_singleton_prune_count",
        "action_prefix_sha256",
        "path_basis_kind",
    }
)
_BASE_GATE_FIELDS = frozenset(
    {
        "passes_fidelity_gate",
        "passes_storage_gate",
        "passes_fidelity_and_storage_gate",
        "passes_merge_recovery_gate",
        "passes_controlled_compression_gate",
        "passes_compute_gate",
    }
)
_PRIMARY_ROW_FIELDS = frozenset(
    {
        "equal_rank_heldout_relative_errors",
        "permuted_graph_local_summary",
        "one_hot_damage_vs_gomp",
        "merge_improvement_vs_one_hot",
        "one_hot_gap_recovery_fraction",
        "sse_recovery_vs_equal_rank_gomp",
        "per_origin_sse_recovery_vs_equal_rank_gomp",
        "minimum_required_sse_recovery",
        "passes_sse_recovery_gate",
        "passes_genuine_merge_gate",
        "passes_one_hot_gate",
        "passes_response_only_gate",
        "passes_permutation_gate",
        "passes_gomp_gate",
        "passes_diagonal_prune_gate",
        "passes_gfa_gate",
        "passes_svd_gate",
    }
)
_ROW_FIELDS = frozenset(
    {
        "method",
        "rank",
        "source_rank",
        "target_rank",
        "source_basis_kind",
        "source_basis_sha256",
        "plan_artifact_sha256",
        "fit_weighted_kernels_sha256",
        "fit_evaluation",
        "heldout_evaluation",
        "basis_locality",
        "coefficient_payload",
        "plan_accounting",
        *_PATH_ROW_FIELDS,
        *_BASE_GATE_FIELDS,
    }
)
_ORDERED_METHODS = frozenset(
    {
        "gomp_prefix",
        "diagonal_energy_prune",
        "signed_gfa_prefix",
        "signed_gfa_fit_energy",
    }
)
_ORDER_ROW_FIELDS = frozenset(
    {
        "fit_only_parent_order_prefix",
        "fit_only_parent_order_prefix_sha256",
    }
)
_RESOURCE_ACCOUNTING_FIELDS = frozenset(
    {
        "model_load_count",
        "tokenizer_load_count",
        "prompt_text_read_count",
        "token_id_read_count",
        "model_forward_count",
        "new_response_measurement_count",
        "authenticated_source_tensor_file_count",
        "authenticated_source_tensor_file_bytes",
        "authenticated_parent_tensor_file_count",
        "authenticated_parent_tensor_file_bytes",
        "fit_response_float64_scalars",
        "selection_response_float64_scalars",
        "fit_only_graph_eigendecomposition_count",
        "wavelet_frame_eigendecomposition_count",
        "fit_only_graph_wavelet_gomp_count",
        "fit_svd_reference_count",
        "supermode_path_fit_count",
        "conditional_plan_fit_count",
        "conditional_plan_heldout_evaluation_count",
        "raw_response_tensors_serialized",
        "compiled_plan_tensors_serialized",
        "parent_basis_tensor_serialized",
        "supermode_path_tensors_serialized",
    }
)
_CONCLUSION_FIELDS = frozenset(
    {
        "primary_method",
        "merge_recovery_passing_ranks",
        "controlled_compression_passing_ranks",
        "selected_rank",
        "development_nominee_rank",
        "development_nominee_plan_artifact_sha256",
        "selected_plan_artifact_sha256",
        "fit_and_selection_are_disjoint",
        "parent_q64_reconstructed_and_verified",
        "all_methods_have_exact_equal_rank_plan_payload",
        "compute_gate_measured",
        "model_compression_claim",
        "latency_or_speed_claim",
        "whole_model_replacement_claim",
        "natural_prompt_or_nll_fidelity_measured",
        "fixed_reference_structural_response_only",
    }
)


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ")


def _finite_float(
    value: object,
    *,
    label: str,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _require_close(
    actual: object,
    expected: float,
    *,
    label: str,
    tolerance: float = 1.0e-12,
) -> None:
    value = _finite_float(actual, label=label)
    if not math.isclose(
        value,
        expected,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError(f"{label} differs")


def _require_exact_bool(
    value: object,
    expected: bool,
    *,
    label: str,
) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} differs")


def _source_basis_kind(method: str) -> str:
    if method == "gomp_prefix":
        return "fit_only_graph_wavelet_gomp"
    if method == "signed_gfa_prefix":
        return "signed_phase_graph_low_frequency"
    if method == "response_only_merge":
        return "fit_only_graph_wavelet_response_only_supermodes"
    if method == "graph_local_merge":
        return "fit_only_graph_wavelet_local_supermodes"
    if method in PERMUTED_METHODS:
        return (
            "fit_only_graph_wavelet_permuted_topology_"
            "supermode_control"
        )
    return "fixed_orthonormal_control"


def _path_key(method: str) -> str | None:
    if method == "graph_local_one_hot":
        return "graph_local_merge"
    if method in PATH_METHODS:
        return method
    return None


def _validate_receipts(
    *,
    source_receipt: Mapping[str, object],
    parent_receipt: Mapping[str, object],
) -> None:
    _require_exact_fields(
        source_receipt,
        _SOURCE_RECEIPT_FIELDS,
        label="supermode source receipt",
    )
    for field in _SOURCE_RECEIPT_FIELDS:
        _require_sha256(
            source_receipt[field],
            label=f"supermode source receipt {field}",
        )
    _require_exact_fields(
        parent_receipt,
        _PARENT_RECEIPT_FIELDS,
        label="supermode parent receipt",
    )
    nested_source = parent_receipt.get("source_receipt")
    if (
        not isinstance(nested_source, Mapping)
        or dict(nested_source) != dict(source_receipt)
    ):
        raise ValueError("supermode parent source receipt differs")
    for field in _PARENT_RECEIPT_FIELDS - {"source_receipt"}:
        _require_sha256(
            parent_receipt[field],
            label=f"supermode parent receipt {field}",
        )


def _validate_protocol(
    protocol: Mapping[str, object],
) -> tuple[int, int, int]:
    _require_exact_fields(
        protocol,
        _PROTOCOL_FIELDS,
        label="supermode protocol",
    )
    if (
        protocol.get("experiment_variant")
        != "fit_only_graph_wavelet_supermode_merge"
        or protocol.get("response_component")
        != "local_central_odd_tangent"
        or protocol.get("response_tensor_order")
        != "source_origin_lag_target"
        or _exact_int_tuple(
            protocol.get("measured_origins"),
            label="measured origins",
        )
        != INTERIOR_ORIGINS
        or _exact_int_tuple(
            protocol.get("fit_origins"),
            label="fit origins",
        )
        != FIT_ORIGINS
        or _exact_int_tuple(
            protocol.get("selection_origins"),
            label="selection origins",
        )
        != SELECTION_ORIGINS
        or _exact_int_tuple(protocol.get("ranks"), label="ranks") != RANKS
        or tuple(protocol.get("method_order", ())) != METHOD_ORDER
        or _exact_int_tuple(
            protocol.get("permuted_graph_local_seeds"),
            label="permuted graph-local seeds",
        )
        != PERMUTED_GRAPH_LOCAL_SEEDS
        or protocol.get("primary_method") != "graph_local_merge"
        or protocol.get("parent_rank") != PARENT_RANK
        or protocol.get("topology_top_k") != TOPOLOGY_TOP_K
    ):
        raise ValueError("supermode frozen protocol differs")
    for field in (
        "all_paths_bases_and_plans_frozen_before_selection_read",
        "target_rank_is_full",
        "selection_is_open_development_not_confirmation",
    ):
        _require_exact_bool(
            protocol[field],
            True,
            label=f"supermode protocol {field}",
        )
    _require_close(
        protocol["minimum_squared_loading"],
        MINIMUM_SQUARED_LOADING,
        label="minimum squared loading",
    )
    _require_close(
        protocol["minimum_lofo_loading_stability"],
        MINIMUM_LOFO_LOADING_STABILITY,
        label="minimum LOFO loading stability",
    )
    source_modes = protocol.get("source_modes")
    target_modes = protocol.get("target_modes")
    lag_count = protocol.get("lag_count")
    fft_length = protocol.get("fft_length")
    if (
        type(source_modes) is not int
        or source_modes != PARENT_RANK
        or type(target_modes) is not int
        or target_modes <= 0
        or type(lag_count) is not int
        or lag_count <= 0
        or type(fft_length) is not int
        or fft_length < lag_count
    ):
        raise ValueError("supermode protocol tensor geometry differs")
    if not set(FIT_ORIGINS).isdisjoint(SELECTION_ORIGINS):
        raise RuntimeError("supermode origin split is not disjoint")
    return source_modes, target_modes, lag_count


def _validate_path_receipts(
    *,
    path_receipts: Mapping[str, object],
    source_modes: int,
    target_modes: int,
    lag_count: int,
) -> None:
    if set(path_receipts) != set(PATH_METHODS):
        raise ValueError("supermode path receipt methods differ")
    common_binding: tuple[object, ...] | None = None
    available_ranks = tuple(range(PARENT_RANK, RANKS[0] - 1, -1))
    expected_fit_shape = (
        source_modes,
        len(FIT_ORIGINS),
        lag_count,
        target_modes,
    )
    expected_fold_shapes = tuple(
        (source_modes, 1, lag_count, target_modes)
        for _ in FIT_ORIGINS
    )
    for name in PATH_METHODS:
        raw = path_receipts.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError("supermode path receipt is invalid")
        _require_exact_fields(
            raw,
            _PATH_REPORT_FIELDS,
            label=f"supermode path receipt {name}",
        )
        expected_core_method = (
            "permuted_graph_local"
            if name in PERMUTED_METHODS
            else name.removesuffix("_merge")
        )
        expected_seed = (
            PERMUTED_GRAPH_LOCAL_SEEDS[PERMUTED_METHODS.index(name)]
            if name in PERMUTED_METHODS
            else 0
        )
        if (
            raw.get("artifact_kind") != _PATH_ARTIFACT_KIND
            or raw.get("format_version") != 1
            or raw.get("algorithm") != _PATH_ALGORITHM
            or raw.get("pair_semantics") != _PATH_PAIR_SEMANTICS
            or raw.get("action_policy") != _PATH_ACTION_POLICY
            or raw.get("topology_semantics")
            != _PATH_TOPOLOGY_SEMANTICS
            or raw.get("lofo_semantics") != _PATH_LOFO_SEMANTICS
            or raw.get("one_hot_control_semantics")
            != _PATH_ONE_HOT_CONTROL_SEMANTICS
            or raw.get("fit_scope") != _PATH_FIT_SCOPE
            or raw.get("method") != expected_core_method
            or raw.get("minimum_rank") != RANKS[0]
            or raw.get("parent_rank") != PARENT_RANK
            or raw.get("node_count") != source_modes
            or raw.get("permutation_seed") != expected_seed
            or raw.get("topology_top_k") != TOPOLOGY_TOP_K
            or raw.get("heldout_input_used") is not False
            or _exact_int_tuple(
                raw.get("available_ranks"),
                label=f"{name} available ranks",
            )
            != available_ranks
            or _exact_int_tuple(
                raw.get("fit_response_shape"),
                label=f"{name} fit response shape",
            )
            != expected_fit_shape
        ):
            raise ValueError("supermode path receipt provenance differs")
        _require_close(
            raw["minimum_squared_loading"],
            MINIMUM_SQUARED_LOADING,
            label=f"{name} minimum squared loading",
        )
        _require_close(
            raw["minimum_lofo_loading_stability"],
            MINIMUM_LOFO_LOADING_STABILITY,
            label=f"{name} minimum LOFO stability",
        )
        fold_shapes_raw = raw.get("fit_fold_response_shapes")
        if (
            not isinstance(fold_shapes_raw, Sequence)
            or isinstance(fold_shapes_raw, (str, bytes))
            or tuple(
                _exact_int_tuple(value, label=f"{name} fit fold shape")
                for value in fold_shapes_raw
            )
            != expected_fold_shapes
        ):
            raise ValueError("supermode path fit-fold geometry differs")
        for field in (
            "fit_response_sha256",
            "signed_laplacian_sha256",
            "parent_basis_sha256",
            "response_gram_sha256",
            "native_topology_matrix_sha256",
            "selection_topology_matrix_sha256",
            "topology_permutation_sha256",
        ):
            _require_sha256(
                raw[field],
                label=f"supermode path {name} {field}",
            )
        for field in (
            "fit_fold_response_sha256s",
            "fit_fold_response_gram_sha256s",
        ):
            values = raw.get(field)
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or len(values) != len(FIT_ORIGINS)
            ):
                raise ValueError("supermode path fit-fold hashes differ")
            for value in values:
                _require_sha256(value, label=f"supermode path {name} {field}")
        path_payload = {
            field: raw[field] for field in _PATH_HASH_PAYLOAD_FIELDS
        }
        artifact_sha256 = _require_sha256(
            raw.get("artifact_sha256"),
            label="supermode path artifact",
        )
        if _path_json_sha256(path_payload) != artifact_sha256:
            raise ValueError("supermode path artifact hash mismatch")

        selected_actions = raw.get("selected_actions")
        action_diagnostics = raw.get("action_diagnostics")
        if (
            not isinstance(selected_actions, Sequence)
            or isinstance(selected_actions, (str, bytes))
            or not isinstance(action_diagnostics, Sequence)
            or isinstance(action_diagnostics, (str, bytes))
            or selected_actions != action_diagnostics
            or len(action_diagnostics) != PARENT_RANK - RANKS[0]
        ):
            raise ValueError("supermode path action diagnostics differ")
        actions = []
        used_endpoints: set[int] = set()
        previous_action_loss = -math.inf
        for index, action_raw in enumerate(action_diagnostics, start=1):
            if not isinstance(action_raw, Mapping):
                raise ValueError("supermode path action is invalid")
            action = _path_action_from_metadata(action_raw)
            if (
                action.action_order != index
                or action.rank_after_action != PARENT_RANK - index
                or set(action.endpoints) & used_endpoints
                or any(
                    endpoint >= PARENT_RANK for endpoint in action.endpoints
                )
                or action.action_loss < previous_action_loss
            ):
                raise ValueError("supermode path action sequence differs")
            if isinstance(action, GraphWaveletSupermodePair) and (
                action.minimum_squared_loading
                < MINIMUM_SQUARED_LOADING
                or action.lofo_loading_stability
                < MINIMUM_LOFO_LOADING_STABILITY
                or (
                    expected_core_method != "response_only"
                    and (
                        not action.topology_union_top_k_eligible
                        or action.topology_interaction <= 0.0
                    )
                )
            ):
                raise ValueError("supermode path merge gate differs")
            used_endpoints.update(action.endpoints)
            previous_action_loss = action.action_loss
            actions.append(action)
        merge_count = sum(
            isinstance(action, GraphWaveletSupermodePair)
            for action in actions
        )
        if (
            raw.get("merge_action_count") != merge_count
            or raw.get("singleton_prune_action_count")
            != len(actions) - merge_count
        ):
            raise ValueError("supermode path action counts differ")

        rank_path = raw.get("rank_path")
        if (
            not isinstance(rank_path, Sequence)
            or isinstance(rank_path, (str, bytes))
            or len(rank_path) != len(available_ranks)
        ):
            raise ValueError("supermode path rank report differs")
        for rank, rank_raw in zip(available_ranks, rank_path, strict=True):
            if not isinstance(rank_raw, Mapping):
                raise ValueError("supermode path rank row is invalid")
            _require_exact_fields(
                rank_raw,
                _PATH_RANK_ROW_FIELDS,
                label="supermode path rank row",
            )
            active = actions[: PARENT_RANK - rank]
            active_merges = tuple(
                action
                for action in active
                if isinstance(action, GraphWaveletSupermodePair)
            )
            if (
                rank_raw.get("rank") != rank
                or rank_raw.get("active_action_count") != len(active)
                or rank_raw.get("active_merge_count") != len(active_merges)
                or rank_raw.get("active_singleton_prune_count")
                != len(active) - len(active_merges)
            ):
                raise ValueError("supermode path rank action counts differ")
            _require_sha256(
                rank_raw["mixed_basis_sha256"],
                label="supermode path mixed basis",
            )
            _require_sha256(
                rank_raw["one_hot_control_basis_sha256"],
                label="supermode path one-hot basis",
            )
            expected_diagnostics = (
                math.fsum(action.action_loss for action in active),
                (
                    math.fsum(
                        action.relative_response_loss
                        for action in active_merges
                    )
                    / len(active_merges)
                    if active_merges
                    else 0.0
                ),
                (
                    math.fsum(
                        action.mixing_participation
                        for action in active_merges
                    )
                    / len(active_merges)
                    if active_merges
                    else 0.0
                ),
                (
                    min(
                        action.lofo_loading_stability
                        for action in active_merges
                    )
                    if active_merges
                    else 0.0
                ),
                (
                    math.fsum(
                        action.topology_interaction
                        for action in active_merges
                    )
                    / len(active_merges)
                    if active_merges
                    else 0.0
                ),
            )
            for field, expected in zip(
                (
                    "cumulative_action_loss",
                    "mean_merge_relative_response_loss",
                    "mean_merge_mixing_participation",
                    "minimum_merge_lofo_loading_stability",
                    "mean_merge_topology_interaction",
                ),
                expected_diagnostics,
                strict=True,
            ):
                _require_close(
                    rank_raw[field],
                    expected,
                    label=f"supermode path rank {field}",
                )
        binding = (
            raw["fit_response_shape"],
            raw["fit_response_sha256"],
            raw["fit_fold_response_shapes"],
            raw["fit_fold_response_sha256s"],
            raw["signed_laplacian_sha256"],
            raw["parent_basis_sha256"],
            raw["response_gram_sha256"],
            raw["fit_fold_response_gram_sha256s"],
            raw["native_topology_matrix_sha256"],
        )
        if common_binding is None:
            common_binding = binding
        elif binding != common_binding:
            raise ValueError("supermode path authenticated fit binding differs")


def _validate_evaluation(
    value: object,
    *,
    plan_sha256: str,
    origins: tuple[int, ...],
    overlap: tuple[int, ...],
    label: str,
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} receipt is missing")
    _require_exact_fields(value, _EVALUATION_FIELDS, label=label)
    if (
        value.get("plan_sha256") != plan_sha256
        or _exact_int_tuple(
            value.get("evaluation_origins"),
            label=f"{label} origins",
        )
        != origins
        or _exact_int_tuple(
            value.get("fit_origin_overlap"),
            label=f"{label} fit overlap",
        )
        != overlap
    ):
        raise ValueError(f"{label} provenance differs")
    _require_exact_bool(
        value["fit_was_not_recomputed"],
        True,
        label=f"{label} fit-was-not-recomputed",
    )
    response_binding = _require_sha256(
        value.get("response_binding_sha256"),
        label=f"{label} response binding",
    )
    target_norm = _finite_float(
        value.get("weighted_target_frobenius"),
        label=f"{label} target norm",
        nonnegative=True,
    )
    residual_norm = _finite_float(
        value.get("weighted_residual_frobenius"),
        label=f"{label} residual norm",
        nonnegative=True,
    )
    _require_close(
        value.get("weighted_relative_error"),
        residual_norm / max(target_norm, torch.finfo(torch.float64).eps),
        label=f"{label} relative error",
    )
    cosine = _finite_float(
        value.get("weighted_cosine"),
        label=f"{label} cosine",
    )
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"{label} cosine lies outside [-1, 1]")
    for field, bounded in (
        ("per_origin_weighted_relative_errors", False),
        ("per_origin_weighted_cosines", True),
    ):
        metrics = value.get(field)
        if (
            not isinstance(metrics, Sequence)
            or isinstance(metrics, (str, bytes))
            or len(metrics) != len(origins)
        ):
            raise ValueError(f"{label} per-origin metrics differ")
        for metric in metrics:
            number = _finite_float(
                metric,
                label=f"{label} {field}",
                nonnegative=not bounded,
            )
            if bounded and not -1.0 <= number <= 1.0:
                raise ValueError(f"{label} cosine lies outside [-1, 1]")
    return response_binding


def _validate_plan_accounting(
    *,
    row: Mapping[str, object],
    source_modes: int,
    target_modes: int,
    lag_count: int,
    fft_length: int,
    full_plan_scalars: int,
    full_prepared_storage_bytes: int,
) -> float:
    rank = int(row["rank"])
    spectrum_bins = fft_length // 2 + 1
    source_spectrum_rank = min(
        source_modes,
        2 * len(FIT_ORIGINS) * spectrum_bins * target_modes,
    )
    target_spectrum_rank = min(
        target_modes,
        2 * source_modes * len(FIT_ORIGINS) * spectrum_bins,
    )
    expected_accounting = ConditionalSpectralGeneratorAccounting(
        source_modes=source_modes,
        target_modes=target_modes,
        source_rank=rank,
        target_rank=target_modes,
        knot_count=len(FIT_ORIGINS),
        lag_count=lag_count,
        source_spectrum_rank=source_spectrum_rank,
        target_spectrum_rank=target_spectrum_rank,
    ).metadata()
    accounting = row.get("plan_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("supermode row plan accounting is missing")
    if set(accounting) != set(expected_accounting):
        raise ValueError("supermode plan accounting fields differ")
    for field, expected in expected_accounting.items():
        actual = accounting[field]
        if isinstance(expected, float):
            _require_close(
                actual,
                expected,
                label=f"supermode plan accounting {field}",
                tolerance=1.0e-15,
            )
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError("supermode plan accounting geometry differs")

    payload = row.get("coefficient_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("supermode coefficient payload is missing")
    _require_exact_fields(
        payload,
        _COEFFICIENT_PAYLOAD_FIELDS,
        label="supermode coefficient payload",
    )
    expected_source = int(
        expected_accounting["source_basis_coefficient_count"]
    )
    expected_target = int(
        expected_accounting["target_basis_coefficient_count"]
    )
    expected_core = int(expected_accounting["core_coefficient_count"])
    expected_stored = int(expected_accounting["stored_coefficient_count"])
    expected_prepared = int(expected_accounting["prepared_storage_bytes"])
    for field, expected in (
        ("source_basis_float64_scalars", expected_source),
        ("target_basis_float64_scalars", expected_target),
        ("fit_knot_core_float64_scalars", expected_core),
        ("compiled_plan_float64_scalars", expected_stored),
        ("prepared_runtime_storage_bytes", expected_prepared),
    ):
        if type(payload.get(field)) is not int or payload.get(field) != expected:
            raise ValueError("supermode coefficient accounting differs")
    _require_close(
        payload["compiled_plan_coefficient_fraction_of_full_rank"],
        expected_stored / full_plan_scalars,
        label="supermode compiled-plan coefficient fraction",
        tolerance=1.0e-15,
    )
    prepared_fraction = expected_prepared / full_prepared_storage_bytes
    _require_close(
        payload["prepared_runtime_storage_fraction_of_full_rank"],
        prepared_fraction,
        label="supermode prepared-runtime storage fraction",
        tolerance=1.0e-15,
    )
    for field in (
        "storage_gate_uses_complete_prepared_runtime_bytes",
        "equal_rank_payload_match",
        "basis_construction_metadata_excluded_from_plan_payload",
    ):
        _require_exact_bool(
            payload[field],
            True,
            label=f"supermode coefficient payload {field}",
        )
    return prepared_fraction


def _validate_candidate_geometry(
    *,
    source_receipt: Mapping[str, object],
    parent_receipt: Mapping[str, object],
    protocol: Mapping[str, object],
    path_receipts: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    conclusions: Mapping[str, object],
    resource_accounting: Mapping[str, object],
) -> None:
    _validate_receipts(
        source_receipt=source_receipt,
        parent_receipt=parent_receipt,
    )
    source_modes, target_modes, lag_count = _validate_protocol(protocol)
    fft_length = int(protocol["fft_length"])
    _validate_path_receipts(
        path_receipts=path_receipts,
        source_modes=source_modes,
        target_modes=target_modes,
        lag_count=lag_count,
    )

    expected_sequence = tuple(
        (method, rank)
        for method in METHOD_ORDER
        for rank in RANKS
    )
    actual_sequence = tuple(
        (row.get("method"), row.get("rank"))
        for row in rows
    )
    if actual_sequence != expected_sequence:
        raise ValueError("supermode row method and rank sequence differs")
    spectrum_bins = fft_length // 2 + 1
    full_accounting = ConditionalSpectralGeneratorAccounting(
        source_modes=source_modes,
        target_modes=target_modes,
        source_rank=source_modes,
        target_rank=target_modes,
        knot_count=len(FIT_ORIGINS),
        lag_count=lag_count,
        source_spectrum_rank=min(
            source_modes,
            2 * len(FIT_ORIGINS) * spectrum_bins * target_modes,
        ),
        target_spectrum_rank=min(
            target_modes,
            2 * source_modes * len(FIT_ORIGINS) * spectrum_bins,
        ),
    )
    full_plan_scalars = full_accounting.stored_coefficient_count
    full_prepared_storage_bytes = full_accounting.prepared_storage_bytes
    response_binding: str | None = None
    weighted_fit_sha256: str | None = None
    order_prefixes: dict[str, tuple[int, ...]] = {}
    by_method_rank: dict[tuple[str, int], Mapping[str, object]] = {}

    for row in rows:
        if type(row.get("method")) is not str or type(row.get("rank")) is not int:
            raise ValueError("supermode row method or rank type differs")
        method = row["method"]
        rank = row["rank"]
        expected_fields = set(_ROW_FIELDS)
        if method in _ORDERED_METHODS:
            expected_fields.update(_ORDER_ROW_FIELDS)
        if method == "graph_local_merge":
            expected_fields.update(_PRIMARY_ROW_FIELDS)
        if set(row) != expected_fields:
            raise ValueError("supermode row fields differ")
        if (
            row.get("source_rank") != rank
            or row.get("target_rank") != target_modes
            or row.get("source_basis_kind") != _source_basis_kind(method)
        ):
            raise ValueError("supermode row rank or source kind differs")
        source_basis_sha256 = _require_sha256(
            row.get("source_basis_sha256"),
            label="source basis",
        )
        plan_sha256 = _require_sha256(
            row.get("plan_artifact_sha256"),
            label="conditional plan",
        )
        fit_weighted_sha256 = _require_sha256(
            row.get("fit_weighted_kernels_sha256"),
            label="fit weighted kernels",
        )
        if weighted_fit_sha256 is None:
            weighted_fit_sha256 = fit_weighted_sha256
        elif fit_weighted_sha256 != weighted_fit_sha256:
            raise ValueError("supermode plan fit tensor binding differs")
        locality = row.get("basis_locality")
        if not isinstance(locality, Mapping):
            raise ValueError("supermode basis-locality receipt is missing")
        _require_exact_fields(
            locality,
            _BASIS_LOCALITY_FIELDS,
            label="supermode basis-locality receipt",
        )
        mean_support = _finite_float(
            locality["mean_effective_node_support"],
            label="mean effective node support",
        )
        maximum_support = _finite_float(
            locality["maximum_effective_node_support"],
            label="maximum effective node support",
        )
        mean_variation = _finite_float(
            locality["mean_signed_graph_quadratic_form"],
            label="mean signed graph quadratic form",
        )
        maximum_variation = _finite_float(
            locality["maximum_signed_graph_quadratic_form"],
            label="maximum signed graph quadratic form",
        )
        if (
            locality.get("column_count") != rank
            or mean_support < 1.0 - 1.0e-10
            or maximum_support + 1.0e-12 < mean_support
            or maximum_support > source_modes + 1.0e-8
            or maximum_variation + 1.0e-12 < mean_variation
        ):
            raise ValueError("supermode basis-locality geometry differs")
        _require_exact_bool(
            locality["hop_locality_claim"],
            False,
            label="supermode hop-locality claim",
        )
        prepared_fraction = _validate_plan_accounting(
            row=row,
            source_modes=source_modes,
            target_modes=target_modes,
            lag_count=lag_count,
            fft_length=fft_length,
            full_plan_scalars=full_plan_scalars,
            full_prepared_storage_bytes=full_prepared_storage_bytes,
        )
        fit_binding = _validate_evaluation(
            row.get("fit_evaluation"),
            plan_sha256=plan_sha256,
            origins=FIT_ORIGINS,
            overlap=FIT_ORIGINS,
            label="supermode fit evaluation",
        )
        heldout_binding = _validate_evaluation(
            row.get("heldout_evaluation"),
            plan_sha256=plan_sha256,
            origins=SELECTION_ORIGINS,
            overlap=(),
            label="supermode heldout evaluation",
        )
        if fit_binding != heldout_binding:
            raise ValueError("supermode evaluation response bindings differ")
        if response_binding is None:
            response_binding = fit_binding
        elif fit_binding != response_binding:
            raise ValueError("supermode row response binding differs")

        if method in _ORDERED_METHODS:
            order = _exact_int_tuple(
                row.get("fit_only_parent_order_prefix"),
                label="fit-only parent order prefix",
            )
            if (
                len(order) != rank
                or len(set(order)) != rank
                or any(value < 0 for value in order)
                or (
                    method != "gomp_prefix"
                    and any(value >= source_modes for value in order)
                )
                or row.get("fit_only_parent_order_prefix_sha256")
                != _json_sha256(order, domain=_TENSOR_DOMAIN)
            ):
                raise ValueError("supermode fit-only order prefix differs")
            previous = order_prefixes.get(method)
            if previous is not None and order[: len(previous)] != previous:
                raise ValueError("supermode fit-only order is not nested")
            order_prefixes[method] = order

        path_key = _path_key(method)
        if path_key is None:
            expected_path_values = {
                "path_artifact_sha256": None,
                "active_action_count": 0,
                "active_selected_pair_action_count": 0,
                "active_genuine_merge_count": 0,
                "active_paired_delete_count": 0,
                "active_singleton_prune_count": 0,
                "action_prefix_sha256": None,
                "path_basis_kind": None,
            }
        else:
            receipt = path_receipts[path_key]
            if not isinstance(receipt, Mapping):
                raise ValueError("supermode row path receipt is invalid")
            diagnostics = receipt["action_diagnostics"]
            if not isinstance(diagnostics, Sequence):
                raise ValueError("supermode row path diagnostics are invalid")
            active = tuple(diagnostics[: PARENT_RANK - rank])
            selected_pairs = sum(
                isinstance(action, Mapping)
                and action.get("action_kind") == "merge"
                for action in active
            )
            singleton_prunes = len(active) - selected_pairs
            one_hot = method == "graph_local_one_hot"
            expected_path_values = {
                "path_artifact_sha256": receipt["artifact_sha256"],
                "active_action_count": len(active),
                "active_selected_pair_action_count": selected_pairs,
                "active_genuine_merge_count": (
                    0 if one_hot else selected_pairs
                ),
                "active_paired_delete_count": (
                    selected_pairs if one_hot else 0
                ),
                "active_singleton_prune_count": singleton_prunes,
                "action_prefix_sha256": _json_sha256(
                    active,
                    domain=_TENSOR_DOMAIN,
                ),
                "path_basis_kind": (
                    "one_hot_control" if one_hot else "dense_supermode"
                ),
            }
        if any(
            row.get(field) != expected
            for field, expected in expected_path_values.items()
        ):
            raise ValueError("supermode row path binding differs")

        heldout = row["heldout_evaluation"]
        if not isinstance(heldout, Mapping):
            raise ValueError("supermode heldout evaluation is invalid")
        fidelity_gate = (
            float(heldout["weighted_relative_error"])
            <= MAXIMUM_SELECTION_RELATIVE_ERROR
            and float(heldout["weighted_cosine"])
            >= MINIMUM_SELECTION_COSINE
        )
        storage_gate = (
            1.0 - prepared_fraction
            >= MINIMUM_COMPILED_PLAN_REDUCTION
        )
        for field, expected in (
            ("passes_fidelity_gate", fidelity_gate),
            ("passes_storage_gate", storage_gate),
            (
                "passes_fidelity_and_storage_gate",
                fidelity_gate and storage_gate,
            ),
            ("passes_compute_gate", False),
        ):
            _require_exact_bool(
                row[field],
                expected,
                label=f"supermode row {field}",
            )
        if method != "graph_local_merge":
            for field in (
                "passes_merge_recovery_gate",
                "passes_controlled_compression_gate",
            ):
                _require_exact_bool(
                    row[field],
                    False,
                    label=f"supermode row {field}",
                )
        by_method_rank[(method, rank)] = row
        del source_basis_sha256

    for rank in RANKS:
        primary = by_method_rank[("graph_local_merge", rank)]
        heldout = primary["heldout_evaluation"]
        if not isinstance(heldout, Mapping):
            raise ValueError("supermode primary heldout evaluation is invalid")
        primary_error = float(heldout["weighted_relative_error"])
        errors = {
            method: float(
                by_method_rank[(method, rank)]["heldout_evaluation"][
                    "weighted_relative_error"
                ]  # type: ignore[index]
            )
            for method in METHOD_ORDER
            if method != "graph_local_merge"
        }
        reported_errors = primary.get("equal_rank_heldout_relative_errors")
        if (
            not isinstance(reported_errors, Mapping)
            or set(reported_errors) != set(errors)
        ):
            raise ValueError("supermode equal-rank error report differs")
        for method, expected in errors.items():
            _require_close(
                reported_errors[method],
                expected,
                label=f"supermode equal-rank error {method}",
            )
        permuted_errors = tuple(errors[method] for method in PERMUTED_METHODS)
        permuted_win_count = sum(
            primary_error <= error + 1.0e-12
            for error in permuted_errors
        )
        reported_summary = primary.get("permuted_graph_local_summary")
        expected_summary = {
            "minimum": min(permuted_errors),
            "median": float(median(permuted_errors)),
            "maximum": max(permuted_errors),
            "primary_win_count": permuted_win_count,
            "required_win_count": MINIMUM_PERMUTED_CONTROL_WINS,
        }
        if (
            not isinstance(reported_summary, Mapping)
            or set(reported_summary) != set(expected_summary)
        ):
            raise ValueError("supermode permuted-control summary differs")
        for field, expected in expected_summary.items():
            if isinstance(expected, float):
                _require_close(
                    reported_summary[field],
                    expected,
                    label=f"supermode permutation summary {field}",
                )
            elif (
                type(reported_summary[field]) is not int
                or reported_summary[field] != expected
            ):
                raise ValueError("supermode permuted-control summary differs")

        one_hot_error = errors["graph_local_one_hot"]
        gomp_error = errors["gomp_prefix"]
        one_hot_gap = one_hot_error - gomp_error
        merge_improvement = one_hot_error - primary_error
        one_hot_gap_recovery = (
            merge_improvement / one_hot_gap
            if one_hot_gap > 1.0e-12
            else None
        )
        sse_recovery = (
            1.0 - (primary_error / gomp_error) ** 2
            if gomp_error > torch.finfo(torch.float64).tiny
            else None
        )
        primary_per_origin = tuple(
            float(value)
            for value in heldout[
                "per_origin_weighted_relative_errors"
            ]  # type: ignore[index]
        )
        gomp_heldout = by_method_rank[("gomp_prefix", rank)][
            "heldout_evaluation"
        ]
        if not isinstance(gomp_heldout, Mapping):
            raise ValueError("supermode GOMP evaluation is invalid")
        gomp_per_origin = tuple(
            float(value)
            for value in gomp_heldout[
                "per_origin_weighted_relative_errors"
            ]  # type: ignore[index]
        )
        per_origin_recovery = tuple(
            (
                1.0 - (primary_value / gomp_value) ** 2
                if gomp_value > torch.finfo(torch.float64).tiny
                else 0.0
            )
            for primary_value, gomp_value in zip(
                primary_per_origin,
                gomp_per_origin,
                strict=True,
            )
        )
        genuine_merge_gate = (
            int(primary["active_genuine_merge_count"]) >= 1
        )
        sse_recovery_gate = (
            sse_recovery is not None
            and sse_recovery >= MINIMUM_SSE_RECOVERY_OVER_GOMP
            and all(value > 0.0 for value in per_origin_recovery)
        )
        one_hot_gate = primary_error <= one_hot_error + 1.0e-12
        response_gate = (
            primary_error <= errors["response_only_merge"] + 1.0e-12
        )
        permutation_gate = (
            primary_error <= float(median(permuted_errors)) + 1.0e-12
            and permuted_win_count >= MINIMUM_PERMUTED_CONTROL_WINS
        )
        gomp_gate = primary_error <= gomp_error + 1.0e-12
        diagonal_gate = (
            primary_error <= errors["diagonal_energy_prune"] + 1.0e-12
        )
        gfa_gate = primary_error <= min(
            errors["signed_gfa_prefix"],
            errors["signed_gfa_fit_energy"],
        ) + 1.0e-12
        svd_gate = primary_error <= errors["fit_svd"] + 1.0e-12
        merge_recovery_gate = (
            bool(primary["passes_fidelity_and_storage_gate"])
            and sse_recovery_gate
            and genuine_merge_gate
            and one_hot_gate
            and response_gate
            and permutation_gate
            and diagonal_gate
        )
        controlled_gate = (
            merge_recovery_gate
            and gomp_gate
            and diagonal_gate
            and gfa_gate
            and svd_gate
            and bool(primary["passes_compute_gate"])
        )
        for field, expected in (
            ("one_hot_damage_vs_gomp", one_hot_gap),
            ("merge_improvement_vs_one_hot", merge_improvement),
            ("sse_recovery_vs_equal_rank_gomp", sse_recovery),
        ):
            if expected is None:
                if primary.get(field) is not None:
                    raise ValueError(f"supermode primary {field} differs")
            else:
                _require_close(
                    primary.get(field),
                    expected,
                    label=f"supermode primary {field}",
                )
        if one_hot_gap_recovery is None:
            if primary.get("one_hot_gap_recovery_fraction") is not None:
                raise ValueError("supermode one-hot recovery differs")
        else:
            _require_close(
                primary.get("one_hot_gap_recovery_fraction"),
                one_hot_gap_recovery,
                label="supermode one-hot recovery",
            )
        reported_per_origin = primary.get(
            "per_origin_sse_recovery_vs_equal_rank_gomp"
        )
        if (
            not isinstance(reported_per_origin, Sequence)
            or isinstance(reported_per_origin, (str, bytes))
            or len(reported_per_origin) != len(per_origin_recovery)
        ):
            raise ValueError("supermode per-origin SSE recovery differs")
        for actual, expected in zip(
            reported_per_origin,
            per_origin_recovery,
            strict=True,
        ):
            _require_close(
                actual,
                expected,
                label="supermode per-origin SSE recovery",
            )
        _require_close(
            primary.get("minimum_required_sse_recovery"),
            MINIMUM_SSE_RECOVERY_OVER_GOMP,
            label="supermode minimum SSE recovery",
        )
        for field, expected in (
            ("passes_sse_recovery_gate", sse_recovery_gate),
            ("passes_genuine_merge_gate", genuine_merge_gate),
            ("passes_one_hot_gate", one_hot_gate),
            ("passes_response_only_gate", response_gate),
            ("passes_permutation_gate", permutation_gate),
            ("passes_gomp_gate", gomp_gate),
            ("passes_diagonal_prune_gate", diagonal_gate),
            ("passes_gfa_gate", gfa_gate),
            ("passes_svd_gate", svd_gate),
            ("passes_merge_recovery_gate", merge_recovery_gate),
            ("passes_controlled_compression_gate", controlled_gate),
        ):
            _require_exact_bool(
                primary[field],
                expected,
                label=f"supermode primary {field}",
            )

    _require_exact_fields(
        conclusions,
        _CONCLUSION_FIELDS,
        label="supermode conclusions",
    )
    recovery_ranks = tuple(
        rank
        for rank in RANKS
        if bool(
            by_method_rank[("graph_local_merge", rank)][
                "passes_merge_recovery_gate"
            ]
        )
    )
    controlled_ranks = tuple(
        rank
        for rank in RANKS
        if bool(
            by_method_rank[("graph_local_merge", rank)][
                "passes_controlled_compression_gate"
            ]
        )
    )
    selected_rank = controlled_ranks[0] if controlled_ranks else None
    nominee_rank = min(recovery_ranks) if recovery_ranks else None
    selected_plan = (
        by_method_rank[("graph_local_merge", selected_rank)][
            "plan_artifact_sha256"
        ]
        if selected_rank is not None
        else None
    )
    nominee_plan = (
        by_method_rank[("graph_local_merge", nominee_rank)][
            "plan_artifact_sha256"
        ]
        if nominee_rank is not None
        else None
    )
    if (
        conclusions.get("primary_method") != "graph_local_merge"
        or _exact_int_tuple(
            conclusions.get("merge_recovery_passing_ranks"),
            label="merge recovery passing ranks",
        )
        != recovery_ranks
        or _exact_int_tuple(
            conclusions.get("controlled_compression_passing_ranks"),
            label="controlled compression passing ranks",
        )
        != controlled_ranks
        or conclusions.get("selected_rank") != selected_rank
        or conclusions.get("development_nominee_rank") != nominee_rank
        or conclusions.get("selected_plan_artifact_sha256") != selected_plan
        or conclusions.get("development_nominee_plan_artifact_sha256")
        != nominee_plan
    ):
        raise ValueError("supermode conclusions differ from rate rows")
    for field, expected in (
        ("fit_and_selection_are_disjoint", True),
        ("parent_q64_reconstructed_and_verified", True),
        ("all_methods_have_exact_equal_rank_plan_payload", True),
        ("compute_gate_measured", False),
        ("model_compression_claim", False),
        ("latency_or_speed_claim", False),
        ("whole_model_replacement_claim", False),
        ("natural_prompt_or_nll_fidelity_measured", False),
        ("fixed_reference_structural_response_only", True),
    ):
        _require_exact_bool(
            conclusions[field],
            expected,
            label=f"supermode conclusion {field}",
        )

    _require_exact_fields(
        resource_accounting,
        _RESOURCE_ACCOUNTING_FIELDS,
        label="supermode resource accounting",
    )
    for field in (
        "authenticated_source_tensor_file_bytes",
        "authenticated_parent_tensor_file_bytes",
    ):
        value = resource_accounting[field]
        if type(value) is not int or value < 0:
            raise ValueError("supermode authenticated file bytes differ")
    expected_plan_count = len(METHOD_ORDER) * len(RANKS)
    expected_resources = {
        "model_load_count": 0,
        "tokenizer_load_count": 0,
        "prompt_text_read_count": 0,
        "token_id_read_count": 0,
        "model_forward_count": 0,
        "new_response_measurement_count": 0,
        "authenticated_source_tensor_file_count": 1,
        "authenticated_parent_tensor_file_count": 1,
        "fit_response_float64_scalars": (
            source_modes
            * len(FIT_ORIGINS)
            * lag_count
            * target_modes
        ),
        "selection_response_float64_scalars": (
            source_modes
            * len(SELECTION_ORIGINS)
            * lag_count
            * target_modes
        ),
        "fit_only_graph_eigendecomposition_count": 2,
        "wavelet_frame_eigendecomposition_count": 1,
        "fit_only_graph_wavelet_gomp_count": 1,
        "fit_svd_reference_count": 1,
        "supermode_path_fit_count": len(PATH_METHODS),
        "conditional_plan_fit_count": expected_plan_count,
        "conditional_plan_heldout_evaluation_count": expected_plan_count,
        "raw_response_tensors_serialized": False,
        "compiled_plan_tensors_serialized": False,
        "parent_basis_tensor_serialized": False,
        "supermode_path_tensors_serialized": False,
    }
    for field, expected in expected_resources.items():
        actual = resource_accounting[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("supermode resource accounting differs")


@dataclass(frozen=True, slots=True, eq=False)
class Gemma3GraphWaveletSupermodeCandidate:
    """Authenticated metadata-only supermode rate-curve candidate."""

    source_receipt: Mapping[str, object]
    parent_receipt: Mapping[str, object]
    protocol: Mapping[str, object]
    path_receipts: Mapping[str, object]
    rate_rows: tuple[Mapping[str, object], ...]
    conclusions: Mapping[str, object]
    resource_accounting: Mapping[str, object]
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.format_version != _FORMAT_VERSION:
            raise ValueError("supermode candidate header differs")
        for field in (
            "source_receipt",
            "parent_receipt",
            "protocol",
            "path_receipts",
            "conclusions",
            "resource_accounting",
        ):
            object.__setattr__(
                self,
                field,
                _canonical_mapping(getattr(self, field), label=field),
            )
        object.__setattr__(self, "rate_rows", _canonical_rows(self.rate_rows))
        _validate_candidate_geometry(
            source_receipt=self.source_receipt,
            parent_receipt=self.parent_receipt,
            protocol=self.protocol,
            path_receipts=self.path_receipts,
            rows=self.rate_rows,
            conclusions=self.conclusions,
            resource_accounting=self.resource_accounting,
        )
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="candidate artifact",
                )
                != computed
            ):
                raise ValueError("supermode candidate hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_receipt": dict(self.source_receipt),
            "parent_receipt": dict(self.parent_receipt),
            "protocol": dict(self.protocol),
            "path_receipts": dict(self.path_receipts),
            "rate_rows": tuple(dict(row) for row in self.rate_rows),
            "conclusions": dict(self.conclusions),
            "resource_accounting": dict(self.resource_accounting),
            "safety": _SAFETY,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("supermode candidate hash mismatch")
        _validate_candidate_geometry(
            source_receipt=self.source_receipt,
            parent_receipt=self.parent_receipt,
            protocol=self.protocol,
            path_receipts=self.path_receipts,
            rows=self.rate_rows,
            conclusions=self.conclusions,
            resource_accounting=self.resource_accounting,
        )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        value: object,
    ) -> "Gemma3GraphWaveletSupermodeCandidate":
        expected = {
            "schema",
            "format_version",
            "source_receipt",
            "parent_receipt",
            "protocol",
            "path_receipts",
            "rate_rows",
            "conclusions",
            "resource_accounting",
            "safety",
            "artifact_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("safety") != _SAFETY
        ):
            raise ValueError("supermode candidate state fields differ")
        return cls(
            source_receipt=value["source_receipt"],  # type: ignore[arg-type]
            parent_receipt=value["parent_receipt"],  # type: ignore[arg-type]
            protocol=value["protocol"],  # type: ignore[arg-type]
            path_receipts=value["path_receipts"],  # type: ignore[arg-type]
            rate_rows=tuple(value["rate_rows"]),  # type: ignore[arg-type]
            conclusions=value["conclusions"],  # type: ignore[arg-type]
            resource_accounting=value[
                "resource_accounting"
            ],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class _BasisFamily:
    method: str
    bases: Mapping[int, Tensor]
    source_basis_kind: str
    order: tuple[int, ...] | None = None
    path_key: str | None = None
    one_hot_control: bool = False


def _path_action_metadata(
    path: FitOnlyGraphWaveletSupermodePath,
    rank: int,
    *,
    one_hot_control: bool,
) -> dict[str, object]:
    actions = tuple(path.action_prefix(rank))
    selected_pair_count = sum(
        isinstance(action, GraphWaveletSupermodePair)
        for action in actions
    )
    metadata = tuple(
        action.metadata()
        if callable(getattr(action, "metadata", None))
        else dict(action)  # type: ignore[arg-type]
        for action in actions
    )
    return {
        "path_artifact_sha256": path.artifact_sha256,
        "active_action_count": len(actions),
        "active_selected_pair_action_count": selected_pair_count,
        "active_genuine_merge_count": (
            0 if one_hot_control else selected_pair_count
        ),
        "active_paired_delete_count": (
            selected_pair_count if one_hot_control else 0
        ),
        "active_singleton_prune_count": sum(
            not isinstance(action, GraphWaveletSupermodePair)
            for action in actions
        ),
        "action_prefix_sha256": _json_sha256(
            metadata,
            domain=_TENSOR_DOMAIN,
        ),
    }


def _row_prefix(
    *,
    family: _BasisFamily,
    rank: int,
    basis: Tensor,
    plan: ConditionalSpectralGeneratorPlan,
    fit_evaluation: object,
    signed_laplacian: Tensor,
    path: FitOnlyGraphWaveletSupermodePath | None,
    full_plan_scalars: int,
    full_prepared_storage_bytes: int,
) -> dict[str, object]:
    source_scalars = int(plan.source_basis.numel())
    target_scalars = int(plan.target_basis.numel())
    core_scalars = int(plan.knot_cores.numel())
    plan_scalars = plan.stored_coefficient_count
    plan_accounting = plan.accounting().metadata()
    prepared_storage_bytes = int(
        plan_accounting["prepared_storage_bytes"]
    )
    result: dict[str, object] = {
        "method": family.method,
        "rank": rank,
        "source_rank": plan.source_rank,
        "target_rank": plan.target_rank,
        "source_basis_kind": family.source_basis_kind,
        "source_basis_sha256": _tensor_sha256(basis),
        "plan_artifact_sha256": plan.artifact_sha256,
        "fit_weighted_kernels_sha256": plan.fit_weighted_kernels_sha256,
        "fit_evaluation": _evaluation_metadata(fit_evaluation),
        "heldout_evaluation": None,
        "basis_locality": _basis_locality(basis, signed_laplacian),
        "coefficient_payload": {
            "source_basis_float64_scalars": source_scalars,
            "target_basis_float64_scalars": target_scalars,
            "fit_knot_core_float64_scalars": core_scalars,
            "compiled_plan_float64_scalars": plan_scalars,
            "compiled_plan_coefficient_fraction_of_full_rank": (
                plan_scalars / full_plan_scalars
            ),
            "prepared_runtime_storage_bytes": prepared_storage_bytes,
            "prepared_runtime_storage_fraction_of_full_rank": (
                prepared_storage_bytes / full_prepared_storage_bytes
            ),
            "storage_gate_uses_complete_prepared_runtime_bytes": True,
            "equal_rank_payload_match": True,
            "basis_construction_metadata_excluded_from_plan_payload": True,
        },
        "plan_accounting": plan_accounting,
    }
    if family.order is not None:
        result["fit_only_parent_order_prefix"] = family.order[:rank]
        result["fit_only_parent_order_prefix_sha256"] = _json_sha256(
            family.order[:rank],
            domain=_TENSOR_DOMAIN,
        )
    if path is not None:
        result.update(
            _path_action_metadata(
                path,
                rank,
                one_hot_control=family.one_hot_control,
            )
        )
        result["path_basis_kind"] = (
            "one_hot_control"
            if family.one_hot_control
            else "dense_supermode"
        )
    else:
        result["path_artifact_sha256"] = None
        result["active_action_count"] = 0
        result["active_selected_pair_action_count"] = 0
        result["active_genuine_merge_count"] = 0
        result["active_paired_delete_count"] = 0
        result["active_singleton_prune_count"] = 0
        result["action_prefix_sha256"] = None
        result["path_basis_kind"] = None
    return result


def _parent_rate_row(
    parent: Gemma3GraphWaveletCandidate,
    *,
    rank: int,
) -> Mapping[str, object]:
    matches = tuple(
        row
        for row in parent.rate_rows
        if row.get("method") == "signed_graph_wavelet_omp"
        and row.get("source_rank") == rank
    )
    if len(matches) != 1:
        raise ValueError("parent signed graph-wavelet row is missing")
    return matches[0]


def _compile_from_response(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    parent: Gemma3GraphWaveletCandidate,
    *,
    response_binding_sha256: str,
    expected_graph_basis_artifact_sha256: str,
    source_receipt: Mapping[str, object],
    fft_length: int,
    target_rank: int | None = None,
    source_tensor_file_bytes: int = 0,
    parent_tensor_file_bytes: int = 0,
) -> Gemma3GraphWaveletSupermodeCandidate:
    """Internal seam with a literal fit-before-selection value boundary."""

    if not isinstance(parent, Gemma3GraphWaveletCandidate):
        raise TypeError("parent must be a graph-wavelet candidate")
    parent.validate_integrity()
    response_geometry = _floating_tensor_geometry(
        responses,
        label="responses",
        ndim=4,
    )
    scales = _float_tensor(source_scales, label="source_scales", ndim=1)
    measured_origins = tuple(origins)
    if (
        measured_origins != INTERIOR_ORIGINS
        or len(measured_origins) != response_geometry.shape[1]
        or response_geometry.shape[0] != PARENT_RANK
        or scales.numel() != PARENT_RANK
    ):
        raise ValueError("supermode response geometry or origins differ")
    if target_rank is None:
        target_rank = int(response_geometry.shape[3])
    if target_rank != response_geometry.shape[3]:
        raise ValueError("supermode rung requires a full target basis")
    expected_graph = _require_sha256(
        expected_graph_basis_artifact_sha256,
        label="expected graph basis",
    )
    fit_indices = torch.tensor(
        [measured_origins.index(origin) for origin in FIT_ORIGINS],
        dtype=torch.int64,
        device=response_geometry.device,
    )
    fit_kernels = _float_tensor(
        response_geometry.index_select(1, fit_indices),
        label="fit responses",
        ndim=4,
    )
    weighted_fit = (
        fit_kernels * scales.view(-1, 1, 1, 1)
    ).contiguous()
    fit_folds = tuple(
        weighted_fit[:, index : index + 1].contiguous()
        for index in range(len(FIT_ORIGINS))
    )

    graph = fit_graph_source_bases(
        fit_kernels,
        scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=response_binding_sha256,
        fft_length=fft_length,
    )
    if (
        graph.artifact_sha256 != expected_graph
        or graph.artifact_sha256
        != parent.graph_receipt.get("artifact_sha256")
        or graph.fit_weighted_kernels_sha256
        != parent.graph_receipt.get("fit_weighted_kernels_sha256")
    ):
        raise ValueError("supermode graph provenance differs from parent")
    signed_laplacian = _laplacian(
        graph.signed_eigenvalues,
        graph.signed_eigenvectors,
    )
    signed_frame = build_spectral_graph_wavelet_frame(
        signed_laplacian,
        diffusion_scales=DIFFUSION_SCALES,
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
    )
    parent_signed_frame = parent.frame_receipts.get("signed")
    if (
        not isinstance(parent_signed_frame, Mapping)
        or signed_frame.artifact_sha256
        != parent_signed_frame.get("artifact_sha256")
    ):
        raise ValueError("supermode signed frame provenance differs")
    fit_signals = tuple(
        weighted_fit[:, index].contiguous()
        for index in range(weighted_fit.shape[1])
    )
    parent_subspace = fit_graph_wavelet_omp_subspace(
        signed_frame,
        fit_signals,
        max_rank=PARENT_RANK,
    )
    parent_subspace_receipt = parent_signed_frame.get(
        "fit_only_omp_subspace"
    )
    if (
        not isinstance(parent_subspace_receipt, Mapping)
        or parent_subspace.artifact_sha256
        != parent_subspace_receipt.get("artifact_sha256")
    ):
        raise ValueError("supermode parent GOMP subspace differs")
    q64 = parent_subspace.orthonormal_basis
    parent_rank64 = _parent_rate_row(parent, rank=PARENT_RANK)
    if (
        _parent_tensor_sha256(q64)
        != parent_rank64.get("source_basis_sha256")
        or tuple(parent_subspace.selected_flat_atom_indices)
        != tuple(parent_rank64.get("selected_packet_indices", ()))
    ):
        raise ValueError("supermode reconstructed Q64 differs from parent")
    parent_source = dict(parent.source_receipt)
    if any(
        parent_source.get(key) != value
        for key, value in source_receipt.items()
    ):
        raise ValueError("supermode source receipt differs from parent")

    response_path = fit_graph_wavelet_supermode_merge(
        q64,
        weighted_fit,
        signed_laplacian,
        minimum_rank=RANKS[0],
        method="response_only",
        fit_fold_responses=fit_folds,
        minimum_squared_loading=MINIMUM_SQUARED_LOADING,
        minimum_lofo_loading_stability=(
            MINIMUM_LOFO_LOADING_STABILITY
        ),
        topology_top_k=TOPOLOGY_TOP_K,
    )
    graph_path = fit_graph_wavelet_supermode_merge(
        q64,
        weighted_fit,
        signed_laplacian,
        minimum_rank=RANKS[0],
        method="graph_local",
        fit_fold_responses=fit_folds,
        minimum_squared_loading=MINIMUM_SQUARED_LOADING,
        minimum_lofo_loading_stability=(
            MINIMUM_LOFO_LOADING_STABILITY
        ),
        topology_top_k=TOPOLOGY_TOP_K,
    )
    permuted_paths = {
        method_name: fit_graph_wavelet_supermode_merge(
            q64,
            weighted_fit,
            signed_laplacian,
            minimum_rank=RANKS[0],
            method="permuted_graph_local",
            fit_fold_responses=fit_folds,
            minimum_squared_loading=MINIMUM_SQUARED_LOADING,
            minimum_lofo_loading_stability=(
                MINIMUM_LOFO_LOADING_STABILITY
            ),
            topology_top_k=TOPOLOGY_TOP_K,
            permutation_seed=seed,
        )
        for method_name, seed in zip(
            PERMUTED_METHODS,
            PERMUTED_GRAPH_LOCAL_SEEDS,
            strict=True,
        )
    }

    diagonal_basis, diagonal_order = _fit_energy_ordered_basis(
        q64,
        weighted_fit,
    )
    gfa_prefix = graph.signed_eigenvectors
    gfa_energy_basis, gfa_energy_order = _fit_energy_ordered_basis(
        gfa_prefix,
        weighted_fit,
    )
    full_svd = fit_conditional_spectral_generator(
        fit_kernels,
        scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        PARENT_RANK,
        target_rank,
        response_binding_sha256=response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=fft_length,
    )
    families: dict[str, _BasisFamily] = {
        "gomp_prefix": _BasisFamily(
            method="gomp_prefix",
            bases=MappingProxyType(
                {rank: q64[:, :rank].clone() for rank in RANKS}
            ),
            source_basis_kind="fit_only_graph_wavelet_gomp",
            order=tuple(parent_subspace.selected_flat_atom_indices),
        ),
        "diagonal_energy_prune": _BasisFamily(
            method="diagonal_energy_prune",
            bases=MappingProxyType(
                {rank: diagonal_basis[:, :rank].clone() for rank in RANKS}
            ),
            source_basis_kind="fixed_orthonormal_control",
            order=diagonal_order,
        ),
        "signed_gfa_prefix": _BasisFamily(
            method="signed_gfa_prefix",
            bases=MappingProxyType(
                {rank: gfa_prefix[:, :rank].clone() for rank in RANKS}
            ),
            source_basis_kind="signed_phase_graph_low_frequency",
            order=tuple(range(PARENT_RANK)),
        ),
        "signed_gfa_fit_energy": _BasisFamily(
            method="signed_gfa_fit_energy",
            bases=MappingProxyType(
                {
                    rank: gfa_energy_basis[:, :rank].clone()
                    for rank in RANKS
                }
            ),
            source_basis_kind="fixed_orthonormal_control",
            order=gfa_energy_order,
        ),
        "fit_svd": _BasisFamily(
            method="fit_svd",
            bases=MappingProxyType(
                {
                    rank: full_svd.source_basis[:, :rank].clone()
                    for rank in RANKS
                }
            ),
            source_basis_kind="fixed_orthonormal_control",
        ),
        "response_only_merge": _BasisFamily(
            method="response_only_merge",
            bases=MappingProxyType(
                {rank: response_path.basis(rank) for rank in RANKS}
            ),
            source_basis_kind=(
                "fit_only_graph_wavelet_response_only_supermodes"
            ),
            path_key="response_only_merge",
        ),
        "graph_local_merge": _BasisFamily(
            method="graph_local_merge",
            bases=MappingProxyType(
                {rank: graph_path.basis(rank) for rank in RANKS}
            ),
            source_basis_kind="fit_only_graph_wavelet_local_supermodes",
            path_key="graph_local_merge",
        ),
        "graph_local_one_hot": _BasisFamily(
            method="graph_local_one_hot",
            bases=MappingProxyType(
                {
                    rank: graph_path.one_hot_control_basis(rank)
                    for rank in RANKS
                }
            ),
            source_basis_kind="fixed_orthonormal_control",
            path_key="graph_local_merge",
            one_hot_control=True,
        ),
        **{
            method_name: _BasisFamily(
                method=method_name,
                bases=MappingProxyType(
                    {rank: path.basis(rank) for rank in RANKS}
                ),
                source_basis_kind=(
                    "fit_only_graph_wavelet_permuted_topology_"
                    "supermode_control"
                ),
                path_key=method_name,
            )
            for method_name, path in permuted_paths.items()
        },
    }
    if tuple(families) != METHOD_ORDER:
        raise RuntimeError("supermode family order drifted")
    paths: dict[str, FitOnlyGraphWaveletSupermodePath] = {
        "response_only_merge": response_path,
        "graph_local_merge": graph_path,
        **permuted_paths,
    }

    full_plan_scalars = (
        PARENT_RANK * PARENT_RANK
        + response_geometry.shape[3] * target_rank
        + len(FIT_ORIGINS)
        * response_geometry.shape[2]
        * PARENT_RANK
        * target_rank
    )
    full_prepared_storage_bytes = (
        (full_plan_scalars + PARENT_RANK) * torch.float64.itemsize
        + len(FIT_ORIGINS) * 8
    )
    plans: list[ConditionalSpectralGeneratorPlan] = []
    fit_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        family = families[method]
        for rank in RANKS:
            basis = family.bases[rank]
            plan = fit_conditional_spectral_generator_with_source_basis(
                fit_kernels,
                scales,
                FIT_ORIGINS,
                FIT_ORIGINS,
                basis,
                target_rank,
                source_basis_kind=family.source_basis_kind,  # type: ignore[arg-type]
                source_basis_fit_weighted_kernels_sha256=(
                    graph.fit_weighted_kernels_sha256
                ),
                response_binding_sha256=response_binding_sha256,
                input_transform="standardized_linear",
                fft_length=fft_length,
            )
            fit_evaluation = evaluate_conditional_spectral_generator(
                plan,
                fit_kernels,
                FIT_ORIGINS,
                FIT_ORIGINS,
                response_binding_sha256=response_binding_sha256,
            )
            plans.append(plan)
            fit_rows.append(
                _row_prefix(
                    family=family,
                    rank=rank,
                    basis=basis,
                    plan=plan,
                    fit_evaluation=fit_evaluation,
                    signed_laplacian=signed_laplacian,
                    path=(
                        paths[family.path_key]
                        if family.path_key is not None
                        else None
                    ),
                    full_plan_scalars=full_plan_scalars,
                    full_prepared_storage_bytes=(
                        full_prepared_storage_bytes
                    ),
                )
            )

    # No operation above this boundary validates or reads selection values.
    kernels = _float_tensor(
        response_geometry,
        label="responses",
        ndim=4,
    )
    rate_rows: list[dict[str, object]] = []
    for plan, prefix in zip(plans, fit_rows, strict=True):
        heldout = evaluate_conditional_spectral_generator(
            plan,
            kernels,
            measured_origins,
            SELECTION_ORIGINS,
            response_binding_sha256=response_binding_sha256,
            require_heldout=True,
        )
        row = dict(prefix)
        row["heldout_evaluation"] = _evaluation_metadata(heldout)
        row["passes_fidelity_gate"] = (
            heldout.weighted_relative_error
            <= MAXIMUM_SELECTION_RELATIVE_ERROR
            and heldout.weighted_cosine >= MINIMUM_SELECTION_COSINE
        )
        prepared_storage_fraction = float(
            row["coefficient_payload"][  # type: ignore[index]
                "prepared_runtime_storage_fraction_of_full_rank"
            ]
        )
        row["passes_storage_gate"] = (
            1.0 - prepared_storage_fraction
            >= MINIMUM_COMPILED_PLAN_REDUCTION
        )
        row["passes_fidelity_and_storage_gate"] = (
            bool(row["passes_fidelity_gate"])
            and bool(row["passes_storage_gate"])
        )
        row["passes_merge_recovery_gate"] = False
        row["passes_controlled_compression_gate"] = False
        row["passes_compute_gate"] = False
        rate_rows.append(row)

    by_method_rank = {
        (str(row["method"]), int(row["rank"])): row
        for row in rate_rows
    }
    for rank in RANKS:
        primary = by_method_rank[("graph_local_merge", rank)]
        primary_error = float(
            primary["heldout_evaluation"][  # type: ignore[index]
                "weighted_relative_error"
            ]
        )
        errors = {
            method: float(
                by_method_rank[(method, rank)][
                    "heldout_evaluation"
                ]["weighted_relative_error"]  # type: ignore[index]
            )
            for method in METHOD_ORDER
            if method != "graph_local_merge"
        }
        permuted_errors = tuple(errors[method] for method in PERMUTED_METHODS)
        one_hot_error = errors["graph_local_one_hot"]
        gomp_error = errors["gomp_prefix"]
        one_hot_gap = one_hot_error - gomp_error
        merge_improvement = one_hot_error - primary_error
        one_hot_gap_recovery = (
            merge_improvement / one_hot_gap
            if one_hot_gap > 1.0e-12
            else None
        )
        sse_recovery_vs_gomp = (
            1.0 - (primary_error / gomp_error) ** 2
            if gomp_error > torch.finfo(torch.float64).tiny
            else None
        )
        primary_per_origin = tuple(
            float(value)
            for value in primary["heldout_evaluation"][  # type: ignore[index]
                "per_origin_weighted_relative_errors"
            ]
        )
        gomp_per_origin = tuple(
            float(value)
            for value in by_method_rank[("gomp_prefix", rank)][
                "heldout_evaluation"
            ][  # type: ignore[index]
                "per_origin_weighted_relative_errors"
            ]
        )
        per_origin_sse_recovery = tuple(
            (
                1.0 - (primary_value / gomp_value) ** 2
                if gomp_value > torch.finfo(torch.float64).tiny
                else 0.0
            )
            for primary_value, gomp_value in zip(
                primary_per_origin,
                gomp_per_origin,
                strict=True,
            )
        )
        genuine_merge_gate = (
            int(primary["active_genuine_merge_count"]) >= 1
        )
        sse_recovery_gate = (
            sse_recovery_vs_gomp is not None
            and sse_recovery_vs_gomp
            >= MINIMUM_SSE_RECOVERY_OVER_GOMP
            and all(value > 0.0 for value in per_origin_sse_recovery)
        )
        permuted_win_count = sum(
            primary_error <= error + 1.0e-12
            for error in permuted_errors
        )
        one_hot_gate = primary_error <= one_hot_error + 1.0e-12
        response_gate = (
            primary_error <= errors["response_only_merge"] + 1.0e-12
        )
        permutation_gate = (
            primary_error <= float(median(permuted_errors)) + 1.0e-12
            and permuted_win_count >= MINIMUM_PERMUTED_CONTROL_WINS
        )
        gomp_gate = primary_error <= gomp_error + 1.0e-12
        diagonal_gate = (
            primary_error <= errors["diagonal_energy_prune"] + 1.0e-12
        )
        best_gfa_error = min(
            errors["signed_gfa_prefix"],
            errors["signed_gfa_fit_energy"],
        )
        gfa_gate = primary_error <= best_gfa_error + 1.0e-12
        svd_gate = primary_error <= errors["fit_svd"] + 1.0e-12
        primary["equal_rank_heldout_relative_errors"] = errors
        primary["permuted_graph_local_summary"] = {
            "minimum": min(permuted_errors),
            "median": float(median(permuted_errors)),
            "maximum": max(permuted_errors),
            "primary_win_count": permuted_win_count,
            "required_win_count": MINIMUM_PERMUTED_CONTROL_WINS,
        }
        primary["one_hot_damage_vs_gomp"] = one_hot_gap
        primary["merge_improvement_vs_one_hot"] = merge_improvement
        primary["one_hot_gap_recovery_fraction"] = one_hot_gap_recovery
        primary["sse_recovery_vs_equal_rank_gomp"] = (
            sse_recovery_vs_gomp
        )
        primary["per_origin_sse_recovery_vs_equal_rank_gomp"] = (
            per_origin_sse_recovery
        )
        primary["minimum_required_sse_recovery"] = (
            MINIMUM_SSE_RECOVERY_OVER_GOMP
        )
        primary["passes_sse_recovery_gate"] = sse_recovery_gate
        primary["passes_genuine_merge_gate"] = genuine_merge_gate
        primary["passes_one_hot_gate"] = one_hot_gate
        primary["passes_response_only_gate"] = response_gate
        primary["passes_permutation_gate"] = permutation_gate
        primary["passes_gomp_gate"] = gomp_gate
        primary["passes_diagonal_prune_gate"] = diagonal_gate
        primary["passes_gfa_gate"] = gfa_gate
        primary["passes_svd_gate"] = svd_gate
        primary["passes_merge_recovery_gate"] = (
            bool(primary["passes_fidelity_and_storage_gate"])
            and sse_recovery_gate
            and genuine_merge_gate
            and one_hot_gate
            and response_gate
            and permutation_gate
            and diagonal_gate
        )
        primary["passes_compute_gate"] = False
        primary["passes_controlled_compression_gate"] = (
            bool(primary["passes_merge_recovery_gate"])
            and gomp_gate
            and diagonal_gate
            and gfa_gate
            and svd_gate
            and bool(primary["passes_compute_gate"])
        )

    primary_rows = tuple(
        row for row in rate_rows if row["method"] == "graph_local_merge"
    )
    recovery_passing = tuple(
        row for row in primary_rows if row["passes_merge_recovery_gate"]
    )
    controlled_passing = tuple(
        row
        for row in primary_rows
        if row["passes_controlled_compression_gate"]
    )
    selected = controlled_passing[0] if controlled_passing else None
    development_nominee = (
        min(recovery_passing, key=lambda row: int(row["rank"]))
        if recovery_passing
        else None
    )
    path_receipts = {
        name: path.report()
        for name, path in paths.items()
    }
    protocol = {
        "experiment_variant": "fit_only_graph_wavelet_supermode_merge",
        "response_component": "local_central_odd_tangent",
        "response_tensor_order": "source_origin_lag_target",
        "measured_origins": measured_origins,
        "fit_origins": FIT_ORIGINS,
        "selection_origins": SELECTION_ORIGINS,
        "source_modes": PARENT_RANK,
        "target_modes": int(response_geometry.shape[3]),
        "lag_count": int(response_geometry.shape[2]),
        "fft_length": fft_length,
        "parent_rank": PARENT_RANK,
        "ranks": RANKS,
        "method_order": METHOD_ORDER,
        "primary_method": "graph_local_merge",
        "permuted_graph_local_seeds": PERMUTED_GRAPH_LOCAL_SEEDS,
        "minimum_squared_loading": MINIMUM_SQUARED_LOADING,
        "minimum_lofo_loading_stability": (
            MINIMUM_LOFO_LOADING_STABILITY
        ),
        "topology_top_k": TOPOLOGY_TOP_K,
        "all_paths_bases_and_plans_frozen_before_selection_read": True,
        "target_rank_is_full": True,
        "selection_is_open_development_not_confirmation": True,
    }
    parent_receipt = {
        "artifact_sha256": parent.artifact_sha256,
        "source_receipt": dict(parent.source_receipt),
        "graph_artifact_sha256": parent.graph_receipt["artifact_sha256"],
        "signed_frame_artifact_sha256": (
            parent.frame_receipts["signed"]["artifact_sha256"]  # type: ignore[index]
        ),
        "signed_gomp_subspace_artifact_sha256": (
            parent.frame_receipts["signed"][  # type: ignore[index]
                "fit_only_omp_subspace"
            ]["artifact_sha256"]  # type: ignore[index]
        ),
        "q64_source_basis_sha256": parent_rank64["source_basis_sha256"],
        "q64_selected_packet_order_sha256": (
            parent_rank64["selected_packet_order_sha256"]
        ),
    }
    conclusions = {
        "primary_method": "graph_local_merge",
        "merge_recovery_passing_ranks": tuple(
            int(row["rank"]) for row in recovery_passing
        ),
        "controlled_compression_passing_ranks": tuple(
            int(row["rank"]) for row in controlled_passing
        ),
        "selected_rank": (
            int(selected["rank"]) if selected is not None else None
        ),
        "development_nominee_rank": (
            int(development_nominee["rank"])
            if development_nominee is not None
            else None
        ),
        "development_nominee_plan_artifact_sha256": (
            development_nominee["plan_artifact_sha256"]
            if development_nominee is not None
            else None
        ),
        "selected_plan_artifact_sha256": (
            selected["plan_artifact_sha256"]
            if selected is not None
            else None
        ),
        "fit_and_selection_are_disjoint": True,
        "parent_q64_reconstructed_and_verified": True,
        "all_methods_have_exact_equal_rank_plan_payload": True,
        "compute_gate_measured": False,
        "model_compression_claim": False,
        "latency_or_speed_claim": False,
        "whole_model_replacement_claim": False,
        "natural_prompt_or_nll_fidelity_measured": False,
        "fixed_reference_structural_response_only": True,
    }
    resource_accounting = {
        "model_load_count": 0,
        "tokenizer_load_count": 0,
        "prompt_text_read_count": 0,
        "token_id_read_count": 0,
        "model_forward_count": 0,
        "new_response_measurement_count": 0,
        "authenticated_source_tensor_file_count": 1,
        "authenticated_source_tensor_file_bytes": source_tensor_file_bytes,
        "authenticated_parent_tensor_file_count": 1,
        "authenticated_parent_tensor_file_bytes": parent_tensor_file_bytes,
        "fit_response_float64_scalars": int(weighted_fit.numel()),
        "selection_response_float64_scalars": (
            PARENT_RANK
            * len(SELECTION_ORIGINS)
            * response_geometry.shape[2]
            * response_geometry.shape[3]
        ),
        "fit_only_graph_eigendecomposition_count": 2,
        "wavelet_frame_eigendecomposition_count": 1,
        "fit_only_graph_wavelet_gomp_count": 1,
        "fit_svd_reference_count": 1,
        "supermode_path_fit_count": len(paths),
        "conditional_plan_fit_count": len(plans),
        "conditional_plan_heldout_evaluation_count": len(plans),
        "raw_response_tensors_serialized": False,
        "compiled_plan_tensors_serialized": False,
        "parent_basis_tensor_serialized": False,
        "supermode_path_tensors_serialized": False,
    }
    return Gemma3GraphWaveletSupermodeCandidate(
        source_receipt=source_receipt,
        parent_receipt=parent_receipt,
        protocol=protocol,
        path_receipts=path_receipts,
        rate_rows=tuple(rate_rows),
        conclusions=conclusions,
        resource_accounting=resource_accounting,
    )


def compile_gemma3_graph_wavelet_supermode_candidate(
    source: Gemma3SpectralSource,
    parent: Gemma3GraphWaveletCandidate,
    *,
    source_tensor_file_bytes: int = 0,
    parent_tensor_file_bytes: int = 0,
) -> Gemma3GraphWaveletSupermodeCandidate:
    """Compile the fit-only supermode rung from authenticated inputs."""

    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    if not isinstance(parent, Gemma3GraphWaveletCandidate):
        raise TypeError("parent must be a Gemma3GraphWaveletCandidate")
    source.mapping.validate_integrity()
    parent.validate_integrity()
    if source.mapping.impulse_logical_positions != INTERIOR_ORIGINS:
        raise ValueError("source origins differ from the frozen split")
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    response_binding = _response_binding(
        source,
        component="local_central_odd_tangent",
    )
    return _compile_from_response(
        local.impulse_responses,
        source.source_mode_standard_deviations,
        source.mapping.impulse_logical_positions,
        parent,
        response_binding_sha256=response_binding,
        expected_graph_basis_artifact_sha256=(
            EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256
        ),
        source_receipt={
            "tensor_file_sha256": source.file_sha256,
            "report_file_sha256": source.report_file_sha256,
            "report_payload_sha256": source.report_payload_sha256,
            "mapping_artifact_sha256": source.mapping.artifact_sha256,
            "response_artifact_sha256": local.artifact_sha256,
            "source_model_sha256": source.binding.get("source_model_sha256"),
        },
        fft_length=source.mapping.fft_length,
        source_tensor_file_bytes=source_tensor_file_bytes,
        parent_tensor_file_bytes=parent_tensor_file_bytes,
    )


def _publish_candidate(
    candidate: Gemma3GraphWaveletSupermodeCandidate,
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    reservation = _reserve_outputs((output, report_path))
    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch(candidate.state_dict(), output)
        tensor_digest = _file_sha256(tensor_stage)
        report: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "candidate": candidate.metadata(),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": tensor_digest,
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
            "scientific_status": dict(candidate.conclusions),
            "safety": _SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        report_stage = _stage_json(report, report_path)
        reservation.publish((tensor_stage, report_stage))
        return report
    finally:
        reservation.release()
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def load_gemma3_graph_wavelet_supermode_candidate(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> Gemma3GraphWaveletSupermodeCandidate:
    """Strictly authenticate and restore one metadata-only candidate."""

    source = Path(path)
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected artifact",
    )
    expected_tensor = _require_sha256(
        expected_tensor_file_sha256,
        label="expected tensor file",
    )
    expected_report = _require_sha256(
        expected_report_sha256,
        label="expected report",
    )
    actual_tensor = _file_sha256(source)
    if actual_tensor != expected_tensor:
        raise ValueError("supermode tensor file hash differs")
    with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    claimed_report = _require_sha256(
        report.get("report_sha256"),
        label="report SHA-256",
    )
    payload = dict(report)
    payload.pop("report_sha256")
    if (
        claimed_report != expected_report
        or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed_report
        or report.get("artifact", {}).get("tensor_file_sha256")
        != actual_tensor
    ):
        raise ValueError("supermode report binding differs")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    candidate = Gemma3GraphWaveletSupermodeCandidate.from_state_dict(raw)
    if (
        candidate.artifact_sha256 != expected_artifact
        or report.get("candidate", {}).get("artifact_sha256")
        != expected_artifact
        or _canonical_json_bytes(report["candidate"])
        != _canonical_json_bytes(candidate.metadata())
    ):
        raise ValueError("supermode logical artifact differs")
    return candidate


def analyze_gemma3_l3_l4_graph_wavelet_supermodes(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    source_artifact_sha256: str = DEFAULT_INTERIOR_ARTIFACT_SHA256,
    source_report_sha256: str = DEFAULT_INTERIOR_REPORT_SHA256,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    parent_artifact_sha256: str = DEFAULT_PARENT_ARTIFACT_SHA256,
    parent_tensor_file_sha256: str = DEFAULT_PARENT_TENSOR_FILE_SHA256,
    parent_report_sha256: str = DEFAULT_PARENT_REPORT_SHA256,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Strict-load the pinned source and v2 parent, then publish once."""

    if (
        source_artifact_sha256 != DEFAULT_INTERIOR_ARTIFACT_SHA256
        or source_report_sha256 != DEFAULT_INTERIOR_REPORT_SHA256
    ):
        raise ValueError("supermode source receipt must remain pinned")
    if (
        parent_artifact_sha256 != DEFAULT_PARENT_ARTIFACT_SHA256
        or parent_tensor_file_sha256
        != DEFAULT_PARENT_TENSOR_FILE_SHA256
        or parent_report_sha256 != DEFAULT_PARENT_REPORT_SHA256
    ):
        raise ValueError("supermode parent receipt must remain pinned")
    destination = _validate_output_path(output, suffix=".pt")
    if destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite supermode report")
    source_path = Path(source_artifact_path)
    parent_path = Path(parent_artifact_path)
    source = load_gemma3_spectral_source(
        source_path,
        expected_file_sha256=source_artifact_sha256,
        expected_report_sha256=source_report_sha256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        parent_path,
        expected_artifact_sha256=parent_artifact_sha256,
        expected_tensor_file_sha256=parent_tensor_file_sha256,
        expected_report_sha256=parent_report_sha256,
    )
    candidate = compile_gemma3_graph_wavelet_supermode_candidate(
        source,
        parent,
        source_tensor_file_bytes=source_path.stat().st_size,
        parent_tensor_file_bytes=parent_path.stat().st_size,
    )
    return _publish_candidate(candidate, output=destination)


def describe_gemma3_l3_l4_graph_wavelet_supermodes() -> dict[str, object]:
    """Describe the frozen offline resource contract without opening files."""

    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "source": {
            "artifact": str(DEFAULT_INTERIOR_ARTIFACT),
            "tensor_file_sha256": DEFAULT_INTERIOR_ARTIFACT_SHA256,
            "report_payload_sha256": DEFAULT_INTERIOR_REPORT_SHA256,
        },
        "parent": {
            "artifact": str(DEFAULT_PARENT_ARTIFACT),
            "artifact_sha256": DEFAULT_PARENT_ARTIFACT_SHA256,
            "tensor_file_sha256": DEFAULT_PARENT_TENSOR_FILE_SHA256,
            "report_payload_sha256": DEFAULT_PARENT_REPORT_SHA256,
        },
        "protocol": {
            "fit_origins": FIT_ORIGINS,
            "selection_origins": SELECTION_ORIGINS,
            "parent_rank": PARENT_RANK,
            "ranks": RANKS,
            "method_order": METHOD_ORDER,
            "primary_method": "graph_local_merge",
            "permuted_graph_local_seeds": PERMUTED_GRAPH_LOCAL_SEEDS,
            "minimum_squared_loading": MINIMUM_SQUARED_LOADING,
            "minimum_lofo_loading_stability": (
                MINIMUM_LOFO_LOADING_STABILITY
            ),
            "topology_top_k": TOPOLOGY_TOP_K,
            "selection_is_open_development_not_confirmation": True,
        },
        "resource_contract": {
            "model_load_count": 0,
            "tokenizer_load_count": 0,
            "prompt_text_read_count": 0,
            "token_id_read_count": 0,
            "model_forward_count": 0,
            "new_response_measurement_count": 0,
        },
        "safety": _SAFETY,
        "claims": {
            "natural_prompt_or_nll_fidelity_measured": False,
            "whole_model_replacement_claim": False,
            "model_compression_claim": False,
            "latency_or_speed_claim": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemma L3->L4 graph-wavelet supermode development rung",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_INTERIOR_ARTIFACT,
    )
    analyze.add_argument(
        "--parent-artifact",
        type=Path,
        default=DEFAULT_PARENT_ARTIFACT,
    )
    analyze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        result = describe_gemma3_l3_l4_graph_wavelet_supermodes()
    else:
        result = analyze_gemma3_l3_l4_graph_wavelet_supermodes(
            source_artifact_path=args.source_artifact,
            parent_artifact_path=args.parent_artifact,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
