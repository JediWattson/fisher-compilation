"""Frozen Fisher-generator and causal-innovation development plan.

This module is the zero-forward boundary between hypothesis generation on the
already-open token-Fisher panel and a new family-disjoint derivative
collection.  It deliberately does not collect examples, fit a conditional
model, inspect finite displacements, or compile a provider.

The generator basis is derived once from the authenticated cumulative
six-coordinate Fisher matrix.  Real and imaginary coordinates are kept in
separate three-dimensional blocks and the leading raw-Fisher eigenvector from
each block becomes one orthonormal generator.  A deterministic sign convention
removes the otherwise arbitrary eigenvector sign.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .gemma3_l3_l4_iterative_fisher_corrective_development import (
    GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA,
    validate_gemma_iterative_fisher_corrective_development_report,
)
from .gemma3_l3_l4_iterative_token_fisher_development import (
    TOKEN_FISHER_DEVELOPMENT_SCHEMA,
    validate_gemma_iterative_token_fisher_development_report,
)
from .token_loss_fisher import (
    CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    token_loss_fisher_prompt_record_from_dict,
)


__all__ = [
    "GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA",
    "build_gemma_iterative_generator_innovation_plan",
    "replay_gemma_iterative_generator_innovation_plan",
    "validate_gemma_iterative_generator_innovation_plan",
]


GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_generator_innovation_plan.v1"
)

_PLAN_DOMAIN = (
    b"fisher-graph:gemma-iterative-generator-innovation-plan:v1\0"
)
_BASIS_DOMAIN = (
    b"fisher-graph:gemma-iterative-generator-innovation-basis:v1\0"
)
_FISHER_DOMAIN = (
    b"fisher-graph:gemma-iterative-generator-innovation-fisher:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS_PER_FAMILY = 2
_SOURCE_INDICES = tuple(
    int(index)
    for index in CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES
)
_CHANNELS = (
    ("real", (0, 2, 4)),
    ("imag", (1, 3, 5)),
)
_MINIMUM_EIGENGAP_RATIO = 2.0
_MINIMUM_LOFO_BASIS_COSINE = 0.98
_MINIMUM_FISHER_TRACE_COVERAGE = 0.50
_NUMERICAL_TOLERANCE = 1.0e-10


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _matrix(
    value: object,
    *,
    rows: int,
    columns: int,
    label: str,
) -> Tensor:
    if not isinstance(value, (tuple, list)) or len(value) != rows:
        raise ValueError(f"{label} must contain {rows} rows")
    normalized: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (tuple, list)) or len(row) != columns:
            raise ValueError(
                f"{label}[{row_index}] must contain {columns} scalars"
            )
        normalized.append(
            tuple(
                _finite(item, label=f"{label}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    return torch.tensor(tuple(normalized), dtype=torch.float64)


def _matrix_payload(value: Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _oriented_leading_component(
    fisher: Tensor,
) -> tuple[Tensor, Tensor, int]:
    symmetric = ((fisher + fisher.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    vector = eigenvectors[:, -1].contiguous()
    anchor = int(torch.argmax(vector.abs()))
    if float(vector[anchor]) < 0.0:
        vector = -vector
    return eigenvalues, vector.contiguous(), anchor


def _family_fishers(
    token_fisher_report: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, Tensor], tuple[str, ...]]:
    rows = token_fisher_report.get("prompt_fisher_records")
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("token Fisher report omitted prompt records")
    grouped: dict[str, list[Tensor]] = defaultdict(list)
    prompt_receipts: list[str] = []
    selected = torch.tensor(_SOURCE_INDICES, dtype=torch.int64)
    coordinate_names: tuple[str, ...] | None = None
    for raw in rows:
        record = token_loss_fisher_prompt_record_from_dict(raw)
        record.validate_integrity()
        names = tuple(record.coordinate_names[index] for index in _SOURCE_INDICES)
        if coordinate_names is None:
            coordinate_names = names
        elif coordinate_names != names:
            raise ValueError("cumulative token Fisher coordinates differ")
        full = torch.tensor(
            record.fisher_second_moment,
            dtype=torch.float64,
        )
        grouped[record.family_id].append(
            full.index_select(0, selected)
            .index_select(1, selected)
            .contiguous()
        )
        prompt_receipts.append(record.prompt_record_sha256)
    counts = Counter({family: len(values) for family, values in grouped.items()})
    if (
        len(grouped) != _EXPECTED_FAMILIES
        or set(counts.values()) != {_EXPECTED_PROMPTS_PER_FAMILY}
        or coordinate_names is None
    ):
        raise ValueError("generator source family geometry differs")
    family = {
        family_id: (
            sum(values, torch.zeros_like(values[0])) / len(values)
        ).contiguous()
        for family_id, values in sorted(grouped.items())
    }
    return (
        coordinate_names,
        family,
        tuple(sorted(prompt_receipts)),
    )


def _derive_basis(
    *,
    coordinate_names: Sequence[str],
    family_fishers: Mapping[str, Tensor],
) -> dict[str, object]:
    names = tuple(str(name) for name in coordinate_names)
    families = tuple(sorted(family_fishers))
    if (
        len(names) != 6
        or len(set(names)) != 6
        or len(families) != _EXPECTED_FAMILIES
    ):
        raise ValueError("generator basis source geometry differs")
    matrices: dict[str, Tensor] = {}
    for family_id in families:
        matrix = family_fishers[family_id].to(
            device="cpu",
            dtype=torch.float64,
        )
        if (
            matrix.shape != (6, 6)
            or not bool(torch.isfinite(matrix).all())
            or float((matrix - matrix.T).abs().max())
            > _NUMERICAL_TOLERANCE
            or float(torch.linalg.eigvalsh((matrix + matrix.T) * 0.5).min())
            < -_NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"{family_id} source Fisher is invalid")
        matrices[family_id] = ((matrix + matrix.T) * 0.5).contiguous()
    global_fisher = (
        sum(matrices.values(), torch.zeros((6, 6), dtype=torch.float64))
        / len(matrices)
    ).contiguous()
    if float(torch.trace(global_fisher)) <= 0.0:
        raise ValueError("generator source Fisher has no trace")

    basis = torch.zeros((6, 2), dtype=torch.float64)
    components: dict[str, object] = {}
    lofo: dict[str, dict[str, float]] = {family: {} for family in families}
    for column, (channel, raw_indices) in enumerate(_CHANNELS):
        indices = torch.tensor(raw_indices, dtype=torch.int64)
        block = (
            global_fisher.index_select(0, indices)
            .index_select(1, indices)
            .contiguous()
        )
        eigenvalues, vector, anchor = _oriented_leading_component(block)
        second = float(eigenvalues[-2])
        eigengap = (
            math.inf if second <= 0.0 else float(eigenvalues[-1]) / second
        )
        if not math.isfinite(eigengap):
            raise ValueError("generator eigengap must be finite")
        basis[indices, column] = vector
        fold_cosines: dict[str, float] = {}
        for held_family in families:
            fold = (
                sum(
                    (
                        value
                        for family_id, value in matrices.items()
                        if family_id != held_family
                    ),
                    torch.zeros((6, 6), dtype=torch.float64),
                )
                / (len(matrices) - 1)
            )
            fold_block = (
                fold.index_select(0, indices)
                .index_select(1, indices)
                .contiguous()
            )
            _fold_values, fold_vector, _fold_anchor = (
                _oriented_leading_component(fold_block)
            )
            cosine = abs(float(torch.dot(vector, fold_vector)))
            fold_cosines[held_family] = cosine
            lofo[held_family][channel] = cosine
        components[channel] = {
            "basis_column": column,
            "source_coordinate_indices": raw_indices,
            "source_coordinate_names": tuple(names[index] for index in raw_indices),
            "eigenvalues_ascending": tuple(float(item) for item in eigenvalues),
            "leading_eigenvector": tuple(float(item) for item in vector),
            "sign_anchor_local_index": anchor,
            "sign_orientation": (
                "largest_absolute_entry_positive_then_lowest_index"
            ),
            "leading_to_second_eigenvalue_ratio": eigengap,
            "minimum_lofo_absolute_cosine": min(fold_cosines.values()),
            "lofo_absolute_cosine_by_held_family": fold_cosines,
        }

    gram = basis.T @ basis
    coverage = float(
        torch.trace(basis.T @ global_fisher @ basis)
        / torch.trace(global_fisher)
    )
    gate_results = {
        "basis_is_orthonormal": bool(
            torch.allclose(
                gram,
                torch.eye(2, dtype=torch.float64),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "minimum_channel_eigengap_passed": min(
            float(components[channel]["leading_to_second_eigenvalue_ratio"])
            for channel, _indices in _CHANNELS
        )
        >= _MINIMUM_EIGENGAP_RATIO,
        "minimum_lofo_stability_passed": min(
            float(components[channel]["minimum_lofo_absolute_cosine"])
            for channel, _indices in _CHANNELS
        )
        >= _MINIMUM_LOFO_BASIS_COSINE,
        "minimum_trace_coverage_passed": (
            coverage >= _MINIMUM_FISHER_TRACE_COVERAGE
        ),
    }
    family_payload = {
        family_id: {
            "fisher_second_moment": _matrix_payload(matrix),
            "fisher_sha256": _sha256(
                _FISHER_DOMAIN,
                {
                    "family_id": family_id,
                    "coordinate_names": names,
                    "fisher_second_moment": _matrix_payload(matrix),
                },
            ),
        }
        for family_id, matrix in matrices.items()
    }
    payload: dict[str, object] = {
        "method": "raw_channel_factored_fisher_leading_eigenvectors",
        "rank": 2,
        "source_coordinate_indices": _SOURCE_INDICES,
        "source_coordinate_names": names,
        "channel_partition": {
            channel: indices for channel, indices in _CHANNELS
        },
        "family_fisher_second_moments": family_payload,
        "family_balanced_fisher_second_moment": _matrix_payload(global_fisher),
        "family_balanced_fisher_sha256": _sha256(
            _FISHER_DOMAIN,
            {
                "coordinate_names": names,
                "family_balanced_fisher_second_moment": (
                    _matrix_payload(global_fisher)
                ),
            },
        ),
        "basis_matrix_source_coordinates_by_generator": _matrix_payload(basis),
        "basis_gram_matrix": _matrix_payload(gram),
        "components": components,
        "lofo_absolute_cosine_by_held_family": lofo,
        "fisher_trace_coverage_fraction": coverage,
        "gates": {
            "minimum_channel_leading_to_second_eigenvalue_ratio": (
                _MINIMUM_EIGENGAP_RATIO
            ),
            "minimum_lofo_absolute_cosine": (
                _MINIMUM_LOFO_BASIS_COSINE
            ),
            "minimum_fisher_trace_coverage_fraction": (
                _MINIMUM_FISHER_TRACE_COVERAGE
            ),
        },
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
    }
    return {
        **payload,
        "basis_sha256": _sha256(_BASIS_DOMAIN, payload),
    }


def _innovation_recipe() -> dict[str, object]:
    decay = 2.0 ** (-1.0 / 16.0)
    return {
        "name": "causal_parent_top2_modal_innovation_softsign",
        "source": {
            "stream": "unmodified_accepted_parent_h4_modal_output",
            "mode_selection": (
                "top2_lag_b_output_modes_by_descending_positive_kernel_norm"
            ),
            "normalization": (
                "divide_each_selected_mode_by_its_fixed_parent_lag_b_"
                "output_kernel_norm"
            ),
            "parent_artifact_identity_must_match_plan_lineage": True,
            "candidate_output_read": False,
        },
        "channels": ("real", "imag"),
        "half_life_active_positions": 16,
        "decay_per_active_position": decay,
        "state": {
            "initial_ew_sum_by_channel": (0.0, 0.0),
            "initial_ew_mass": 0.0,
            "runtime_float_scalars_per_sequence": 3,
            "chunk_boundary_state_is_explicit": True,
        },
        "active_position_update": (
            "x=normalized_parent_top2; prior=0_if_mass_zero_else_sum/mass; "
            "raw=x-prior; h=raw/(1+abs(raw)); emit_h_then_"
            "sum=decay*sum+x_and_mass=decay*mass+1"
        ),
        "prior_is_computed_before_current_position_update": True,
        "softsign_is_elementwise": True,
        "output_interval_is_open_minus_one_to_one": True,
        "inactive_or_padding_position": "emit_zero_and_do_not_update_state",
        "causal_read_boundary": (
            "current_parent_modal_value_and_strictly_earlier_active_"
            "parent_modal_values_only"
        ),
        "forbidden_inputs": (
            "token_loss",
            "compensation_target",
            "token_gradient",
            "future_token",
            "future_activation",
            "candidate_output",
            "prompt_id",
            "family_id",
            "final_sequence_length",
        ),
    }


def _activation_tangent_design() -> dict[str, object]:
    return {
        "source_tangent_bank": (
            "cumulative_six_coordinate_activation_position_tangents_T6"
        ),
        "generator_projection": (
            "TU[position,generator,:]=sum_k_T6[position,k,:]*U[k,generator]"
        ),
        "coordinate_order": (
            "generator_real_shared",
            "generator_imag_shared",
            "generator_real_innovation",
            "generator_imag_innovation",
        ),
        "activation_position_columns": (
            "TU_real",
            "TU_imag",
            "h_real_at_position_times_TU_real",
            "h_imag_at_position_times_TU_imag",
        ),
        "innovation_multiplication_boundary": (
            "multiply_each_activation_position_tangent_before_contracting_"
            "with_each_supervised_token_gradient"
        ),
        "post_aggregation_token_score_multiplication_forbidden": True,
        "coefficient_map": (
            "theta(h)=U[:,real]*(beta_real_shared+h_real*"
            "beta_real_innovation)+U[:,imag]*(beta_imag_shared+"
            "h_imag*beta_imag_innovation)"
        ),
        "source_basis_is_fixed": True,
        "learned_scalar_coefficient_count": 4,
    }


def _controls() -> dict[str, object]:
    return {
        "ordered_arms": (
            "parent_zero_correction",
            "legacy_original_shared_coordinates",
            "fixed_u_shared",
            "fixed_u_conditional_innovation",
        ),
        "parent_zero_correction": "all_correction_coefficients_exactly_zero",
        "legacy_original_shared_coordinates": (
            "fit_only_source_coordinates_shared_real_and_shared_imag"
        ),
        "fixed_u_shared": (
            "fit_two_generator_shared_columns_with_innovation_coefficients_"
            "exactly_zero"
        ),
        "fixed_u_conditional_innovation": (
            "fit_all_four_preregistered_generator_columns"
        ),
        "primary_candidate": "fixed_u_conditional_innovation",
        "success_requires_macro_improvement_over": (
            "parent_zero_correction",
            "legacy_original_shared_coordinates",
            "fixed_u_shared",
        ),
        "alternative_basis_screening_on_new_panel_allowed": False,
        "feature_revision_on_new_panel_allowed": False,
    }


def _nested_screen() -> dict[str, object]:
    return {
        "panel_geometry": {
            "examples": 16,
            "families": 8,
            "prompts_per_family": 2,
            "new_family_ids_required": True,
            "disjoint_from_basis_source_examples": True,
            "disjoint_from_basis_source_families": True,
            "disjoint_from_prior_selection_and_calibration_panels": True,
        },
        "outer_split": "leave_one_family_out",
        "inner_split": "leave_one_training_family_out",
        "statistical_weighting": (
            "equal_family_then_equal_prompt_within_family_then_equal_"
            "supervised_token_within_prompt"
        ),
        "tokens_are_independent_split_units": False,
        "standardization": "fit_on_outer_or_inner_training_families_only",
        "conditional_ridge": {
            "applies_to": (
                "generator_real_innovation",
                "generator_imag_innovation",
            ),
            "shared_coordinate_ridge": 1.0e-6,
            "finite_lambda_grid": (
                1.0e-1,
                1.0,
                10.0,
            ),
            "shared_only_candidate": (
                "exact_infinite_lambda_with_both_innovation_coefficients_zero"
            ),
            "selection_metric": "inner_family_macro_rmse",
            "selection_rule": (
                "one_standard_error_toward_shared_only_or_larger_lambda"
            ),
            "ties_prefer": (
                "shared_only_then_larger_finite_lambda_then_smaller_"
                "coefficient_norm"
            ),
            "held_outer_family_used_for_selection": False,
        },
        "gates": {
            "all_outer_standardized_design_rank": 4,
            "maximum_median_outer_standardized_condition_number": 100.0,
            "minimum_mean_pairwise_outer_coefficient_cosine": 0.90,
            "minimum_conditional_residual_design_energy_fraction": 0.05,
            "minimum_parent_family_macro_relative_rmse_improvement": 0.02,
            "minimum_parent_family_win_count": 6,
            "minimum_worst_family_relative_rmse_improvement": -0.02,
            "minimum_conditional_minus_fixed_u_shared_macro_improvement": (
                0.005
            ),
            "minimum_conditional_minus_legacy_shared_macro_improvement": 0.0,
            "minimum_material_conditional_family_win_count": 5,
            "minimum_material_conditional_family_improvement": 0.001,
            "minimum_outer_folds_with_nonzero_conditional_route": 5,
            "minimum_fixed_u_new_panel_fisher_trace_coverage_fraction": 0.50,
        },
        "new_panel_can_refit_generator_basis": False,
        "derivative_screen_can_compile_provider": False,
    }


def _trust_region() -> dict[str, object]:
    return {
        "operator_norm_bound": 0.25,
        "corner_axes": (
            "balance_feature_g",
            "occupancy_feature_o",
            "innovation_real",
            "innovation_imag",
        ),
        "corner_axis_values": (-1.0, 1.0),
        "required_corner_count": 16,
        "proof": (
            "evaluate_route_operator_norm_at_every_corner_of_"
            "{g,o,h_real,h_imag}_in_{-1,+1}^4"
        ),
        "projection": (
            "one_global_nonnegative_radial_scale_shared_by_all_four_"
            "generator_coefficients"
        ),
        "coordinatewise_clipping_allowed": False,
        "projection_must_be_fit_without_held_family": True,
    }


def _phase_boundaries() -> dict[str, object]:
    return {
        "phase_a_freeze": {
            "input": "already_open_authenticated_token_fisher_and_corrective",
            "model_forwards": 0,
            "new_panel_opened": False,
            "output": "this_hashed_basis_feature_and_test_plan",
        },
        "phase_b_derivative_screen": {
            "may_start_after": "phase_a_plan_hash_is_published",
            "collection": (
                "new_family_disjoint_16_example_8_family_exact_token_"
                "jacobian_panel"
            ),
            "source_model_forwards": 16,
            "retained_parent_vjp_forwards": 16,
            "total_model_forwards": 32,
            "backward_call_formula": (
                "sum_over_prompts_ceil(supervised_token_count/8)"
            ),
            "finite_displacements_visible": False,
            "provider_compilation_allowed": False,
        },
        "phase_c_finite_displacement": {
            "may_start_only_if": "every_preregistered_phase_b_gate_passes",
            "data": "out_of_fold_exact_finite_displacements_opened_once",
            "exact_outputs_may_refit_or_select": False,
            "gates": {
                "minimum_predicted_exact_correlation": 0.99,
                "minimum_predicted_exact_sign_agreement": 0.95,
                "maximum_rmse_delta_nll_per_token": 0.01,
                "maximum_worst_prompt_absolute_error_delta_nll_per_token": (
                    0.025
                ),
                "maximum_predicted_exact_family_macro_gap": 0.01,
            },
        },
        "phase_d_provider_shadow": {
            "may_start_only_if": "every_phase_c_gate_passes",
            "action": (
                "compile_mean_reference_provider_then_open_one_untouched_"
                "family_disjoint_shadow_nll_panel"
            ),
            "shadow_outputs_may_refit_or_select": False,
            "absolute_gates": {
                "maximum_mean_delta_nll_per_token": 0.05,
                "minimum_token_top1_agreement": 0.95,
                "maximum_mean_kl_per_token": 0.05,
                "maximum_prompt_p90_delta_nll_per_token": 0.10,
                "minimum_prompt_p10_top1_agreement": 0.90,
            },
        },
    }


def _validate_sources(
    *,
    token_fisher_report: Mapping[str, object],
    token_fisher_report_file_sha256: str,
    corrective_report: Mapping[str, object],
    corrective_report_file_sha256: str,
) -> tuple[str, str, str, str]:
    validate_gemma_iterative_token_fisher_development_report(
        token_fisher_report
    )
    validate_gemma_iterative_fisher_corrective_development_report(
        corrective_report
    )
    token_logical = _require_sha256(
        token_fisher_report.get("report_sha256"),
        label="token Fisher report",
    )
    token_file = _require_sha256(
        token_fisher_report_file_sha256,
        label="token Fisher report file",
    )
    corrective_logical = _require_sha256(
        corrective_report.get("report_sha256"),
        label="corrective report",
    )
    corrective_file = _require_sha256(
        corrective_report_file_sha256,
        label="corrective report file",
    )
    corrective_lineage = _mapping(
        corrective_report.get("lineage"),
        label="corrective lineage",
    )
    if (
        corrective_lineage.get("token_fisher_report_sha256")
        != token_logical
        or corrective_lineage.get("token_fisher_report_file_sha256")
        != token_file
    ):
        raise ValueError("corrective report is not bound to token Fisher source")
    decision = _mapping(
        corrective_report.get("decision"),
        label="corrective decision",
    )
    if (
        decision.get("next_step")
        != (
            "collect_preregistered_new_causal_feature_on_"
            "new_family_disjoint_data"
        )
        or decision.get("provider_compiled") is not False
        or decision.get("runtime_claim_authorized") is not False
        or decision.get("fresh_confirmation_authorized") is not False
    ):
        raise ValueError(
            "corrective source does not authorize this preregistered rung"
        )
    return token_logical, token_file, corrective_logical, corrective_file


def build_gemma_iterative_generator_innovation_plan(
    *,
    token_fisher_report: Mapping[str, object],
    token_fisher_report_file_sha256: str,
    corrective_report: Mapping[str, object],
    corrective_report_file_sha256: str,
) -> dict[str, object]:
    """Build the fixed-U causal-innovation plan without model execution."""

    (
        token_logical,
        token_file,
        corrective_logical,
        corrective_file,
    ) = _validate_sources(
        token_fisher_report=token_fisher_report,
        token_fisher_report_file_sha256=token_fisher_report_file_sha256,
        corrective_report=corrective_report,
        corrective_report_file_sha256=corrective_report_file_sha256,
    )
    coordinate_names, family_fishers, prompt_receipts = _family_fishers(
        token_fisher_report
    )
    basis = _derive_basis(
        coordinate_names=coordinate_names,
        family_fishers=family_fishers,
    )
    source_graph = _mapping(
        _mapping(
            token_fisher_report.get("analysis"),
            label="token Fisher analysis",
        ).get("cumulative_coupling_graph"),
        label="cumulative coupling graph",
    )
    source_graph_fisher = _matrix(
        source_graph.get("family_balanced_fisher_second_moment"),
        rows=6,
        columns=6,
        label="cumulative family-balanced Fisher",
    )
    derived_fisher = _matrix(
        basis["family_balanced_fisher_second_moment"],
        rows=6,
        columns=6,
        label="derived family-balanced Fisher",
    )
    if (
        tuple(source_graph.get("coordinate_names", ())) != coordinate_names
        or not torch.allclose(
            source_graph_fisher,
            derived_fisher,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        or basis["passed"] is not True
    ):
        raise ValueError("authenticated Fisher source cannot freeze this basis")
    token_lineage = _mapping(
        token_fisher_report.get("lineage"),
        label="token Fisher lineage",
    )
    payload: dict[str, object] = {
        "schema": GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA,
        "lineage": {
            "token_fisher_report_sha256": token_logical,
            "token_fisher_report_file_sha256": token_file,
            "token_fisher_schema": TOKEN_FISHER_DEVELOPMENT_SCHEMA,
            "corrective_report_sha256": corrective_logical,
            "corrective_report_file_sha256": corrective_file,
            "corrective_schema": GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA,
            "token_fisher_model_and_parent_lineage": dict(
                sorted(token_lineage.items())
            ),
            "token_fisher_prompt_record_sha256s": prompt_receipts,
        },
        "frozen_generator_basis": basis,
        "causal_innovation_feature": _innovation_recipe(),
        "activation_tangent_design": _activation_tangent_design(),
        "controls": _controls(),
        "nested_family_screen": _nested_screen(),
        "trust_region": _trust_region(),
        "phase_boundaries": _phase_boundaries(),
        "decision": {
            "plan_frozen": True,
            "basis_selection_complete": True,
            "new_family_disjoint_panel_opened": False,
            "derivative_screen_executed": False,
            "finite_displacement_opened": False,
            "provider_compiled": False,
            "runtime_or_compression_claim_authorized": False,
            "next_step": (
                "prepare_then_collect_preregistered_new_family_disjoint_"
                "generator_innovation_panel"
            ),
        },
        "resources": {
            "source_model_forwards": 0,
            "parent_model_forwards": 0,
            "candidate_model_forwards": 0,
            "fresh_panel_forwards": 0,
            "backward_calls": 0,
            "reused_authenticated_prompt_sufficient_statistics": (
                len(prompt_receipts)
            ),
        },
        "audit": {
            "adaptive_hypothesis_generation_only": True,
            "new_panel_prompts_seen": False,
            "new_panel_outputs_seen": False,
            "basis_fixed_before_new_panel": True,
            "feature_fixed_before_new_panel": True,
            "controls_and_gates_fixed_before_new_panel": True,
            "raw_fisher_not_standardized_correlation_used_for_basis": True,
            "fisher_coupling_used_as_causal_direction": False,
            "prompt_or_family_identity_available_to_runtime_feature": False,
            "future_or_target_information_available_to_runtime_feature": False,
            "provider_compiled": False,
        },
    }
    return {
        **payload,
        "plan_sha256": _sha256(_PLAN_DOMAIN, payload),
    }


def _basis_from_serialized_source(value: Mapping[str, object]) -> dict[str, object]:
    names = tuple(value.get("source_coordinate_names", ()))
    family_payload = _mapping(
        value.get("family_fisher_second_moments"),
        label="serialized family Fishers",
    )
    family: dict[str, Tensor] = {}
    for family_id, raw in family_payload.items():
        row = _mapping(raw, label=f"{family_id} serialized Fisher")
        matrix = _matrix(
            row.get("fisher_second_moment"),
            rows=6,
            columns=6,
            label=f"{family_id} Fisher",
        )
        expected_receipt = _sha256(
            _FISHER_DOMAIN,
            {
                "family_id": family_id,
                "coordinate_names": names,
                "fisher_second_moment": _matrix_payload(matrix),
            },
        )
        if row.get("fisher_sha256") != expected_receipt:
            raise ValueError(f"{family_id} Fisher hash mismatch")
        family[str(family_id)] = matrix
    return _derive_basis(coordinate_names=names, family_fishers=family)


def validate_gemma_iterative_generator_innovation_plan(
    plan: object,
) -> None:
    """Validate the plan, including a standalone basis reconstruction."""

    value = _mapping(plan, label="generator innovation plan")
    expected = {
        "schema",
        "lineage",
        "frozen_generator_basis",
        "causal_innovation_feature",
        "activation_tangent_design",
        "controls",
        "nested_family_screen",
        "trust_region",
        "phase_boundaries",
        "decision",
        "resources",
        "audit",
        "plan_sha256",
    }
    if set(value) != expected:
        raise ValueError("generator innovation plan fields differ")
    if value.get("schema") != GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA:
        raise ValueError("generator innovation plan schema differs")
    lineage = _mapping(value.get("lineage"), label="plan lineage")
    if set(lineage) != {
        "token_fisher_report_sha256",
        "token_fisher_report_file_sha256",
        "token_fisher_schema",
        "corrective_report_sha256",
        "corrective_report_file_sha256",
        "corrective_schema",
        "token_fisher_model_and_parent_lineage",
        "token_fisher_prompt_record_sha256s",
    }:
        raise ValueError("generator innovation lineage fields differ")
    for key in (
        "token_fisher_report_sha256",
        "token_fisher_report_file_sha256",
        "corrective_report_sha256",
        "corrective_report_file_sha256",
    ):
        _require_sha256(lineage.get(key), label=f"plan lineage {key}")
    if (
        lineage.get("token_fisher_schema") != TOKEN_FISHER_DEVELOPMENT_SCHEMA
        or lineage.get("corrective_schema")
        != GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA
    ):
        raise ValueError("generator innovation source schema differs")
    parent_lineage = _mapping(
        lineage.get("token_fisher_model_and_parent_lineage"),
        label="model and parent lineage",
    )
    if not parent_lineage:
        raise ValueError("generator plan omitted model and parent lineage")
    for key, receipt in parent_lineage.items():
        _require_sha256(receipt, label=f"model and parent lineage {key}")
    prompt_receipts = tuple(lineage.get("token_fisher_prompt_record_sha256s", ()))
    if (
        len(prompt_receipts) != 16
        or prompt_receipts != tuple(sorted(set(prompt_receipts)))
    ):
        raise ValueError("generator plan prompt-record lineage differs")
    for receipt in prompt_receipts:
        _require_sha256(receipt, label="generator prompt record")

    basis = _mapping(
        value.get("frozen_generator_basis"),
        label="frozen generator basis",
    )
    rebuilt_basis = _basis_from_serialized_source(basis)
    if not _canonical_equal(rebuilt_basis, basis):
        raise ValueError("frozen generator basis does not reconstruct")
    if basis.get("passed") is not True:
        raise ValueError("frozen generator basis gates did not pass")

    expected_constants = {
        "causal_innovation_feature": _innovation_recipe(),
        "activation_tangent_design": _activation_tangent_design(),
        "controls": _controls(),
        "nested_family_screen": _nested_screen(),
        "trust_region": _trust_region(),
        "phase_boundaries": _phase_boundaries(),
    }
    for key, expected_value in expected_constants.items():
        if not _canonical_equal(value.get(key), expected_value):
            raise ValueError(f"generator innovation {key} differs")
    if value.get("decision") != {
        "plan_frozen": True,
        "basis_selection_complete": True,
        "new_family_disjoint_panel_opened": False,
        "derivative_screen_executed": False,
        "finite_displacement_opened": False,
        "provider_compiled": False,
        "runtime_or_compression_claim_authorized": False,
        "next_step": (
            "prepare_then_collect_preregistered_new_family_disjoint_"
            "generator_innovation_panel"
        ),
    }:
        raise ValueError("generator innovation decision differs")
    if value.get("resources") != {
        "source_model_forwards": 0,
        "parent_model_forwards": 0,
        "candidate_model_forwards": 0,
        "fresh_panel_forwards": 0,
        "backward_calls": 0,
        "reused_authenticated_prompt_sufficient_statistics": 16,
    }:
        raise ValueError("generator innovation resource receipt differs")
    audit = _mapping(value.get("audit"), label="generator innovation audit")
    true_keys = {
        "adaptive_hypothesis_generation_only",
        "basis_fixed_before_new_panel",
        "feature_fixed_before_new_panel",
        "controls_and_gates_fixed_before_new_panel",
        "raw_fisher_not_standardized_correlation_used_for_basis",
    }
    false_keys = {
        "new_panel_prompts_seen",
        "new_panel_outputs_seen",
        "fisher_coupling_used_as_causal_direction",
        "prompt_or_family_identity_available_to_runtime_feature",
        "future_or_target_information_available_to_runtime_feature",
        "provider_compiled",
    }
    if (
        set(audit) != true_keys | false_keys
        or any(audit.get(key) is not True for key in true_keys)
        or any(audit.get(key) is not False for key in false_keys)
    ):
        raise ValueError("generator innovation audit differs")
    payload = dict(value)
    receipt = payload.pop("plan_sha256")
    if receipt != _sha256(_PLAN_DOMAIN, payload):
        raise ValueError("generator innovation plan hash mismatch")


def replay_gemma_iterative_generator_innovation_plan(
    *,
    token_fisher_report: Mapping[str, object],
    token_fisher_report_file_sha256: str,
    corrective_report: Mapping[str, object],
    corrective_report_file_sha256: str,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the complete plan from both authenticated source reports."""

    validate_gemma_iterative_generator_innovation_plan(plan)
    rebuilt = build_gemma_iterative_generator_innovation_plan(
        token_fisher_report=token_fisher_report,
        token_fisher_report_file_sha256=token_fisher_report_file_sha256,
        corrective_report=corrective_report,
        corrective_report_file_sha256=corrective_report_file_sha256,
    )
    if not _canonical_equal(rebuilt, plan):
        raise ValueError(
            "generator innovation plan does not replay from source reports"
        )
    return rebuilt
