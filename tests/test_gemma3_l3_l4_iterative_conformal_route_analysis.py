from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

import fisher_graph.gemma3_l3_l4_iterative_conformal_route_analysis as analysis
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_conformal_route import (
    GemmaIterativeConformalRouteFitRecord,
    fit_gemma_iterative_conformal_route_fold,
    gemma_causal_top2_conformal_route_provider_artifact_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_deltas(
    *,
    wins: int = 8,
    loss_delta: float = 0.10,
) -> tuple[float, ...]:
    return tuple(
        0.08 if family_index < wins else loss_delta
        for family_index in range(8)
        for _ in range(2)
    )


def _inputs(
    *,
    candidate_deltas: tuple[float, ...] | None = None,
    candidate_kl: float = 0.9,
    balance_feature_std: float = 0.10,
    top2_modal_energy_fraction: float = 0.75,
    design: str = "stable",
    parent_delta: float = 0.10,
    learned_parameter_count: int = 4,
    include_retained_receipt: bool = True,
    top_mode_indices: tuple[int, int] = (0, 1),
) -> dict[str, object]:
    if candidate_deltas is None:
        candidate_deltas = _candidate_deltas()
    if len(candidate_deltas) != 16:
        raise ValueError("candidate_deltas must contain sixteen values")

    lineage = {
        "parent_artifact_sha256": _digest("parent-artifact"),
        "parent_h4_head_sha256": _digest("parent-h4"),
        "accepted_x4_head_sha256": _digest("accepted-x4"),
        "bridge_binding_sha256": _digest("bridge"),
        "model_sha256": _digest("model"),
        "adapter_execution_sha256": _digest("adapter-execution"),
        "fit_manifest_sha256": _digest("manifest"),
        "factorial_report_sha256": _digest("factorial-report"),
        "factorial_report_file_sha256": _digest("factorial-file"),
        "prior_iteration_report_sha256": _digest(
            "iteration-three-state-experts-report"
        ),
        "prior_iteration_report_file_sha256": _digest(
            "iteration-three-state-experts-file"
        ),
        "prior_iteration_collection_sha256": _digest(
            "iteration-three-state-experts-collection"
        ),
    }
    decoder_sha256 = _digest("parent-h4-decoder")
    lag_kernel_sha256 = _digest("parent-h4-lag-kernel")
    top_mode_norms = (2.0, 1.0)
    parent: list[GemmaH4DampingFiniteNLLObservation] = []
    candidate: list[GemmaH4DampingFiniteNLLObservation] = []
    fits: list[GemmaIterativeConformalRouteFitRecord] = []
    manifest: dict[str, str] = {}

    flat_index = 0
    for family_index in range(8):
        family_id = f"family-{family_index}"
        for example_index in range(2):
            example_id = f"{family_id}-example-{example_index}"
            manifest[example_id] = family_id
            candidate_delta = candidate_deltas[flat_index]
            angle = 2.0 * math.pi * (flat_index + 0.5) / 16.0
            if design == "stable":
                jacobian = (
                    1.0,
                    math.cos(angle),
                    math.sin(angle),
                    math.cos(2.0 * angle),
                )
            elif design == "rank3":
                x = (flat_index + 1) / 20.0
                jacobian = (1.0, x, x * x, x * x)
            elif design == "unsupported":
                jacobian = (
                    1.0,
                    math.cos(angle),
                    math.sin(angle),
                    0.0,
                )
            elif design == "ill_conditioned":
                scale = 1.0e-3
                jacobian = (
                    1.0,
                    scale * math.cos(angle),
                    scale * math.sin(angle),
                    scale * math.cos(2.0 * angle),
                )
            else:
                raise ValueError("unknown design")
            flat_index += 1
            source_logits_sha256 = _digest(f"source:{example_id}")
            targets_sha256 = _digest(f"targets:{example_id}")
            parent_row = GemmaH4DampingFiniteNLLObservation(
                example_id=example_id,
                family_id=family_id,
                supervised_tokens=10,
                source_summed_nll=10.0,
                candidate_summed_nll=10.0 + 10.0 * parent_delta,
                source_to_candidate_summed_kl=1.0,
                top1_matches=8,
                source_logits_sha256=source_logits_sha256,
                candidate_logits_sha256=_digest(
                    f"parent-candidate:{example_id}"
                ),
                targets_sha256=targets_sha256,
            )
            candidate_row = GemmaH4DampingFiniteNLLObservation(
                example_id=example_id,
                family_id=family_id,
                supervised_tokens=10,
                source_summed_nll=10.0,
                candidate_summed_nll=10.0 + 10.0 * candidate_delta,
                source_to_candidate_summed_kl=candidate_kl,
                top1_matches=8,
                source_logits_sha256=source_logits_sha256,
                candidate_logits_sha256=_digest(
                    f"conformal-candidate:{example_id}"
                ),
                targets_sha256=targets_sha256,
            )
            fit = GemmaIterativeConformalRouteFitRecord(
                example_id=example_id,
                family_id=family_id,
                model_inputs_sha256=_digest(f"inputs:{example_id}"),
                parent_execution_sha256=_digest(
                    f"parent-execution:{example_id}"
                ),
                parent_observation_sha256=parent_row.observation_sha256,
                parent_h4_artifact_sha256=lineage[
                    "parent_h4_head_sha256"
                ],
                prefix_sha256=_digest(f"prefix:{example_id}"),
                gradient_sha256=_digest(f"gradient:{example_id}"),
                parent_modal_sha256=_digest(
                    f"parent-modal:{example_id}"
                ),
                balance_feature_sha256=_digest(
                    f"balance-feature:{example_id}"
                ),
                shared_gated_feature_sha256=_digest(
                    f"shared-feature:{example_id}"
                ),
                contrast_gated_feature_sha256=_digest(
                    f"contrast-feature:{example_id}"
                ),
                supervised_tokens=10,
                parent_signed_delta_nll_per_token=parent_delta,
                jacobian_by_conformal_coefficient=jacobian,
                active_row_count=10,
                top_mode_indices=top_mode_indices,
                top_mode_norms=top_mode_norms,
                balance_feature_std=balance_feature_std,
                top2_modal_energy_fraction=top2_modal_energy_fraction,
            )
            parent.append(parent_row)
            candidate.append(candidate_row)
            fits.append(fit)

    folds = [
        fit_gemma_iterative_conformal_route_fold(
            tuple(
                fit for fit in fits if fit.family_id != held_family_id
            ),
            held_family_id=held_family_id,
        ).to_dict()
        for held_family_id in sorted(set(manifest.values()))
    ]
    fold_by_family = {
        str(row["held_family_id"]): row for row in folds
    }
    provider_by_family = {
        family_id: (
            gemma_causal_top2_conformal_route_provider_artifact_sha256(
                parent_artifact_sha256=lineage[
                    "parent_artifact_sha256"
                ],
                parent_h4_artifact_sha256=lineage[
                    "parent_h4_head_sha256"
                ],
                bridge_binding_sha256=lineage[
                    "bridge_binding_sha256"
                ],
                decoder_sha256=decoder_sha256,
                lag_kernel_sha256=lag_kernel_sha256,
                fold_receipt_sha256=str(fold["fold_receipt_sha256"]),
                top_mode_indices=top_mode_indices,
                top_mode_norms=top_mode_norms,
                coefficients_by_conformal_coefficient=fold[
                    "coefficients_by_conformal_coefficient"
                ],
            )
        )
        for family_id, fold in fold_by_family.items()
    }
    fit_by_example = {fit.example_id: fit for fit in fits}
    candidate_by_example = {row.example_id: row for row in candidate}
    model_inputs_by_example = {
        fit.example_id: fit.model_inputs_sha256 for fit in fits
    }
    parent_execution_by_example = {
        fit.example_id: fit.parent_execution_sha256 for fit in fits
    }
    candidate_execution_by_example = {
        example_id: _digest(f"candidate-execution:{example_id}")
        for example_id in sorted(manifest)
    }
    oof_rows: list[dict[str, object]] = []
    for parent_row in parent:
        fit = fit_by_example[parent_row.example_id]
        candidate_row = candidate_by_example[parent_row.example_id]
        fold = fold_by_family[parent_row.family_id]
        coefficients = tuple(
            float(value)
            for value in fold["coefficients_by_conformal_coefficient"]
        )
        predicted = (
            fit.parent_signed_delta_nll_per_token
            + sum(
                left * right
                for left, right in zip(
                    fit.jacobian_by_conformal_coefficient,
                    coefficients,
                    strict=True,
                )
            )
        )
        oof_rows.append(
            {
                "example_id": parent_row.example_id,
                "family_id": parent_row.family_id,
                "held_family_id": parent_row.family_id,
                "parent_signed_delta_nll_per_token": (
                    fit.parent_signed_delta_nll_per_token
                ),
                "predicted_candidate_signed_delta_nll_per_token": predicted,
                "exact_candidate_signed_delta_nll_per_token": (
                    candidate_row.candidate_summed_nll
                    - candidate_row.source_summed_nll
                )
                / candidate_row.supervised_tokens,
                "jacobian_by_conformal_coefficient": (
                    fit.jacobian_by_conformal_coefficient
                ),
                "coefficients_by_conformal_coefficient": coefficients,
                "train_example_ids": fold["train_example_ids"],
                "train_family_ids": fold["train_family_ids"],
                "fit_record_sha256": fit.fit_record_sha256,
                "fold_receipt_sha256": fold["fold_receipt_sha256"],
                "provider_artifact_sha256": provider_by_family[
                    parent_row.family_id
                ],
                "candidate_execution_sha256": (
                    candidate_execution_by_example[parent_row.example_id]
                ),
                "candidate_observation_sha256": (
                    candidate_row.observation_sha256
                ),
            }
        )

    resource_payload = {
        "learned_parameter_count": learned_parameter_count,
        "logical_macs_per_token_upper_bound": 8,
        "derived_constant_float_count": 2,
        "prepared_float_scalar_count": 6,
        "runtime_state_float_count_per_sequence": 2,
        "nonlinear_scalar_ops_per_token_upper_bound": 5,
        "linear_accumulator_scalar_ops_per_token_upper_bound": 4,
        "zero_denominator_comparisons_per_token_upper_bound": 1,
        "parent_decoder_invocations_per_token": 1,
        "serving_model_forward_count": 1,
        "parent_head_reused_not_duplicated": True,
        "parent_artifact_sha256": lineage["parent_artifact_sha256"],
        "parent_h4_head_sha256": lineage["parent_h4_head_sha256"],
        "candidate_provider_artifact_sha256_by_family": provider_by_family,
        "residual_width": 640,
    }
    resources = {
        **resource_payload,
        "resource_receipt_sha256": analysis._sha256(  # noqa: SLF001
            analysis._RESOURCE_DOMAIN,  # noqa: SLF001
            resource_payload,
        ),
    }
    audit = {
        "execution_mode": (
            "fit_only_two_phase_family_blocked_iterative_conformal_route"
        ),
        "conformal_matrix_shape": (2, 2),
        "conformal_coefficient_order": (
            "shared_real",
            "shared_imag",
            "contrast_real",
            "contrast_imag",
        ),
        "route_state_semantics": (
            "top2_parent_lag_b_modal_cumulative_balance_v1"
        ),
        "conformal_route_semantics": (
            "delta=(g*selected_top2)@C(a0+g*a1,b0+g*b1)"
        ),
        "endpoint_operator_norm_bound": 0.25,
        "example_count": 16,
        "family_count": 8,
        "outer_fold_count": 8,
        "phase_a_source_forward_count": 16,
        "phase_a_parent_vjp_forward_count": 16,
        "phase_b_source_forward_count": 16,
        "phase_b_candidate_forward_count": 16,
        "total_model_forward_count": 64,
        "model_forward_count_per_example": 4,
        "one_semantic_candidate_per_iteration": True,
        "family_blocked_leave_one_family_out": True,
        "source_rerun_between_phases": True,
        "source_identity_equal_across_phases": True,
        "parent_observation_count": 16,
        "candidate_observation_count": 16,
        "fit_record_count": 16,
        "fit_records_scalar_hash_only": True,
        "candidate_executions_released_between_examples": True,
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_activations_retained": False,
        "gradient_tensors_retained": False,
        "model_weights_retained": False,
        "source_model_sha256": lineage["model_sha256"],
        "source_execution_sha256": lineage[
            "adapter_execution_sha256"
        ],
        "parent_artifact_sha256": lineage["parent_artifact_sha256"],
        "parent_h4_artifact_sha256": lineage[
            "parent_h4_head_sha256"
        ],
        "parent_h4_decoder_sha256": decoder_sha256,
        "parent_h4_lag_kernel_sha256": lag_kernel_sha256,
        "accepted_x4_head_sha256": lineage["accepted_x4_head_sha256"],
        "fit_manifest_sha256": lineage["fit_manifest_sha256"],
        "residual_width": 640,
        "parent_prepared_float_scalar_count": 438_144,
        "parent_logical_macs_per_token_upper_bound": 252_736,
        "bridge_binding_sha256": lineage["bridge_binding_sha256"],
        "parent_execution_sha256s": tuple(
            sorted(parent_execution_by_example.values())
        ),
        "parent_execution_sha256_by_example": (
            parent_execution_by_example
        ),
        "model_inputs_sha256_by_example": model_inputs_by_example,
        "candidate_execution_sha256s": tuple(
            sorted(candidate_execution_by_example.values())
        ),
        "candidate_execution_sha256_by_example": (
            candidate_execution_by_example
        ),
        "fold_provider_artifact_sha256s": tuple(
            sorted(provider_by_family.values())
        ),
        "fold_provider_artifact_sha256_by_family": provider_by_family,
        "fold_trust_projection_count": sum(
            bool(row["trust_projection_applied"]) for row in folds
        ),
        "trust_projection_interpretation": (
            "global_radial_endpoint_operator_norm_projection_is_"
            "linearization_extrapolation"
        ),
        "routed_parent_decoder_mode_indices": top_mode_indices,
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }

    retained_receipt = None
    if include_retained_receipt:
        full_fit = fit_gemma_iterative_conformal_route_fold(
            fits,
            held_family_id="__full_fit__",
        )
        retained_payload = {
            "provider_artifact_sha256": (
                gemma_causal_top2_conformal_route_provider_artifact_sha256(
                    parent_artifact_sha256=lineage[
                        "parent_artifact_sha256"
                    ],
                    parent_h4_artifact_sha256=lineage[
                        "parent_h4_head_sha256"
                    ],
                    bridge_binding_sha256=lineage[
                        "bridge_binding_sha256"
                    ],
                    decoder_sha256=decoder_sha256,
                    lag_kernel_sha256=lag_kernel_sha256,
                    fold_receipt_sha256=full_fit.fold_receipt_sha256,
                    top_mode_indices=top_mode_indices,
                    top_mode_norms=top_mode_norms,
                    coefficients_by_conformal_coefficient=(
                        full_fit.coefficients_by_conformal_coefficient
                    ),
                )
            ),
            "parent_artifact_sha256": lineage[
                "parent_artifact_sha256"
            ],
            "parent_h4_head_sha256": lineage[
                "parent_h4_head_sha256"
            ],
            "bridge_binding_sha256": lineage[
                "bridge_binding_sha256"
            ],
            "learned_parameter_count": learned_parameter_count,
            "logical_macs_per_token_upper_bound": 8,
            "derived_constant_float_count": 2,
            "prepared_float_scalar_count": 6,
            "runtime_state_float_count_per_sequence": 2,
            "nonlinear_scalar_ops_per_token_upper_bound": 5,
            "linear_accumulator_scalar_ops_per_token_upper_bound": 4,
            "zero_denominator_comparisons_per_token_upper_bound": 1,
            "parent_decoder_invocations_per_token": 1,
            "full_fit": full_fit.to_dict(),
        }
        retained_receipt = {
            **retained_payload,
            "retention_receipt_sha256": analysis._sha256(  # noqa: SLF001
                analysis._RETENTION_DOMAIN,  # noqa: SLF001
                retained_payload,
            ),
        }
    return {
        "parent_observations": parent,
        "candidate_observations": candidate,
        "oof_rows": oof_rows,
        "fit_records": fits,
        "fold_receipts": folds,
        "manifest": manifest,
        "lineage": lineage,
        "resources": resources,
        "audit": audit,
        "retained_full_fit_receipt": retained_receipt,
        "provisional": False,
    }


def _resign(report: dict[str, object]) -> None:
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = analysis._sha256(  # noqa: SLF001
        analysis._REPORT_DOMAIN,  # noqa: SLF001
        payload,
    )


def test_passing_conformal_report_replays_every_gate() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs()
    )
    analysis.validate_gemma_iterative_conformal_route_report(report)

    assert report["schema"] == (
        "fisher_graph.gemma3_l3_l4_iterative_conformal_route_analysis"
    )
    assert report["semantics"]["iteration"] == 4
    assert report["decision"]["retained"] is True
    assert report["decision"]["behavior_relative_gates"]["passed"] is True
    assert report["decision"]["scientific_gates"]["passed"] is True
    assert report["decision"]["resource_gates"]["passed"] is True
    metrics = report["metrics"][
        "conformal_support_condition_and_stability"
    ]
    assert metrics["full_rank_fold_count"] == 8
    assert metrics["all_coordinates_supported_fold_count"] == 8
    assert metrics["fold_coefficient_pair_count"] == 28
    assert metrics["mean_pairwise_fold_coefficient_cosine"] >= 0.90
    linearization_correlation = report["metrics"]["linearization"][
        "predicted_vs_exact_correlation"
    ]
    assert -1.0 <= linearization_correlation <= 1.0


def test_correlation_centers_nonzero_offset_pairs() -> None:
    assert analysis._correlation(  # noqa: SLF001
        (10.0, 11.0, 12.0, 13.0),
        (110.0, 112.0, 114.0, 116.0),
    ) == pytest.approx(1.0)


def test_nonleading_modes_json_round_trip_and_reordering_replay() -> None:
    inputs = _inputs(top_mode_indices=(3, 1))
    inputs["parent_observations"] = tuple(
        reversed(inputs["parent_observations"])
    )
    inputs["candidate_observations"] = tuple(
        reversed(inputs["candidate_observations"])
    )
    inputs["fit_records"] = tuple(reversed(inputs["fit_records"]))
    inputs["fold_receipts"] = tuple(reversed(inputs["fold_receipts"]))
    inputs["oof_rows"] = tuple(reversed(inputs["oof_rows"]))
    report = analysis.build_gemma_iterative_conformal_route_report(**inputs)
    serialized = json.loads(json.dumps(report))
    analysis.validate_gemma_iterative_conformal_route_report(serialized)


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    (
        (
            "balance_feature_std",
            0.049,
            "family_macro_balance_feature_std_at_least_0_05",
        ),
        (
            "top2_modal_energy_fraction",
            0.499,
            "family_macro_top2_modal_energy_fraction_at_least_0_5",
        ),
    ),
)
def test_scientific_feature_gates_reject(
    field: str,
    value: float,
    gate: str,
) -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs(
            **{field: value},
            include_retained_receipt=False,
        )
    )
    assert report["decision"]["scientific_gates"][gate] is False
    assert report["decision"]["retained"] is False


@pytest.mark.parametrize(
    ("design", "gate"),
    (
        ("rank3", "all_fold_weighted_design_ranks_exactly_4"),
        (
            "unsupported",
            "all_4_conformal_coordinates_supported_in_every_fold",
        ),
        (
            "ill_conditioned",
            "median_fold_normal_condition_number_at_most_100",
        ),
    ),
)
def test_design_scientific_gates_reject(
    design: str,
    gate: str,
) -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs(design=design, include_retained_receipt=False)
    )
    assert report["decision"]["scientific_gates"][gate] is False
    assert report["decision"]["retained"] is False


def test_fold_coefficient_stability_uses_28_unordered_pairs() -> None:
    inputs = _inputs()
    fits = tuple(fit.to_dict() for fit in inputs["fit_records"])
    folds = copy.deepcopy(inputs["fold_receipts"])
    for index, fold in enumerate(folds):
        fold["coefficients_by_conformal_coefficient"] = (
            (1.0, 0.0, 0.0, 0.0)
            if index < 4
            else (0.0, 1.0, 0.0, 0.0)
        )
    metrics, gates = analysis._scientific_metrics(  # noqa: SLF001
        fits,
        folds,
    )
    assert metrics["fold_coefficient_pair_count"] == 28
    assert metrics["mean_pairwise_fold_coefficient_cosine"] < 0.90
    assert gates[
        "mean_pairwise_fold_coefficient_cosine_at_least_0_90"
    ] is False


def test_condition_gate_uses_the_even_sample_median() -> None:
    inputs = _inputs()
    fits = tuple(fit.to_dict() for fit in inputs["fit_records"])
    folds = copy.deepcopy(inputs["fold_receipts"])
    at_threshold = (1.0, 2.0, 3.0, 100.0, 100.0, 500.0, 600.0, 700.0)
    for fold, value in zip(folds, at_threshold, strict=True):
        fold["normal_condition_number"] = value
    metrics, gates = analysis._scientific_metrics(  # noqa: SLF001
        fits,
        folds,
    )
    assert metrics["median_fold_normal_condition_number"] == 100.0
    assert gates[
        "median_fold_normal_condition_number_at_most_100"
    ] is True

    folds[4]["normal_condition_number"] = 100.2
    metrics, gates = analysis._scientific_metrics(  # noqa: SLF001
        fits,
        folds,
    )
    assert metrics["median_fold_normal_condition_number"] == 100.1
    assert gates[
        "median_fold_normal_condition_number_at_most_100"
    ] is False


def test_valid_zero_fit_panel_is_rejected_by_explicit_gate_and_replays() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs(
            parent_delta=0.0,
            candidate_deltas=(0.0,) * 16,
            include_retained_receipt=False,
        )
    )
    analysis.validate_gemma_iterative_conformal_route_report(report)
    metrics = report["metrics"][
        "conformal_support_condition_and_stability"
    ]
    assert metrics["zero_norm_fold_count"] == 8
    assert len(metrics["zero_norm_held_family_ids"]) == 8
    assert set(metrics["fold_coefficient_norm_by_held_family"].values()) == {
        0.0
    }
    assert report["decision"]["scientific_gates"][
        "all_fold_coefficient_norms_positive"
    ] is False
    assert report["decision"]["retained"] is False


def test_exactly_six_family_wins_passes_behavior() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs(
            candidate_deltas=_candidate_deltas(
                wins=6,
                loss_delta=0.101,
            ),
        )
    )
    assert report["decision"]["behavior_relative_gates"]["passed"] is True
    assert report["metrics"]["paired"]["strict_family_win_count"] == 6


def test_five_family_wins_rejects_behavior() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs(
            candidate_deltas=_candidate_deltas(
                wins=5,
                loss_delta=0.10,
            ),
            include_retained_receipt=False,
        )
    )
    gates = report["decision"]["behavior_relative_gates"]
    assert gates["strict_family_win_count_at_least_6_of_8"] is False
    assert report["decision"]["retained"] is False


def test_worst_family_and_secondary_fidelity_gates_reject() -> None:
    deltas = list(_candidate_deltas())
    deltas[-2:] = [0.103, 0.103]
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs(
            candidate_deltas=tuple(deltas),
            candidate_kl=1.03,
            include_retained_receipt=False,
        )
    )
    gates = report["decision"]["behavior_relative_gates"]
    assert gates["worst_family_improvement_at_least_minus_2pct"] is False
    assert gates["family_macro_kl_regression_at_most_2pct"] is False


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    (
        (
            "learned_parameter_count",
            5,
            "learned_parameter_count_exactly_4",
        ),
        (
            "logical_macs_per_token_upper_bound",
            9,
            "logical_macs_per_token_exactly_8",
        ),
        (
            "derived_constant_float_count",
            3,
            "derived_constant_float_count_exactly_2",
        ),
        (
            "prepared_float_scalar_count",
            7,
            "prepared_float_scalar_count_exactly_6",
        ),
        (
            "runtime_state_float_count_per_sequence",
            3,
            "runtime_state_float_count_per_sequence_exactly_2",
        ),
        (
            "nonlinear_scalar_ops_per_token_upper_bound",
            6,
            "nonlinear_scalar_ops_per_token_upper_bound_exactly_5",
        ),
        (
            "linear_accumulator_scalar_ops_per_token_upper_bound",
            5,
            (
                "linear_accumulator_scalar_ops_per_token_upper_bound_"
                "exactly_4"
            ),
        ),
        (
            "zero_denominator_comparisons_per_token_upper_bound",
            2,
            (
                "zero_denominator_comparisons_per_token_upper_bound_"
                "exactly_1"
            ),
        ),
        (
            "parent_decoder_invocations_per_token",
            2,
            "parent_decoder_invocations_per_token_exactly_1",
        ),
        (
            "serving_model_forward_count",
            2,
            "serving_model_forward_count_exactly_1",
        ),
        (
            "parent_head_reused_not_duplicated",
            False,
            "parent_head_reused_not_duplicated",
        ),
    ),
)
def test_resource_envelope_is_exact_and_rejection_has_no_full_fit(
    field: str,
    value: object,
    gate: str,
) -> None:
    inputs = _inputs(include_retained_receipt=False)
    resources = dict(inputs["resources"])
    resources[field] = value
    payload = dict(resources)
    payload.pop("resource_receipt_sha256")
    resources["resource_receipt_sha256"] = analysis._sha256(  # noqa: SLF001
        analysis._RESOURCE_DOMAIN,  # noqa: SLF001
        payload,
    )
    inputs["resources"] = resources
    report = analysis.build_gemma_iterative_conformal_route_report(**inputs)
    assert report["decision"]["resource_gates"][gate] is False
    assert report["decision"]["retained"] is False
    assert report["retained_full_fit"] is None


def test_immediate_iteration_three_lineage_is_exact() -> None:
    inputs = _inputs()
    inputs["lineage"] = dict(inputs["lineage"])
    inputs["lineage"].pop("prior_iteration_collection_sha256")
    with pytest.raises(ValueError, match="lineage fields"):
        analysis.build_gemma_iterative_conformal_route_report(**inputs)

    inputs = _inputs()
    inputs["lineage"] = dict(inputs["lineage"])
    inputs["lineage"]["prior_iteration_report_sha256"] = inputs[
        "lineage"
    ]["factorial_report_sha256"]
    with pytest.raises(ValueError, match="aliases"):
        analysis.build_gemma_iterative_conformal_route_report(**inputs)


def test_frozen_parent_lineage_cannot_contradict_execution() -> None:
    inputs = _inputs()
    inputs["lineage"] = dict(inputs["lineage"])
    inputs["lineage"]["parent_artifact_sha256"] = _digest(
        "different-parent"
    )
    with pytest.raises(ValueError, match="contradict"):
        analysis.build_gemma_iterative_conformal_route_report(**inputs)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "match"),
    (
        (
            "oof_rows",
            "predicted_candidate_signed_delta_nll_per_token",
            123.0,
            "does not replay",
        ),
        (
            "fold_receipts",
            "post_projection_endpoint_operator_norms",
            (0.0, 0.0),
            "fold coefficients do not replay",
        ),
        (
            "fit_records",
            "shared_gated_feature_sha256",
            "0" * 64,
            "fit-record hash mismatch",
        ),
    ),
)
def test_scalar_receipt_tampering_is_rejected_after_resigning(
    section: str,
    field: str,
    replacement: object,
    match: str,
) -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs()
    )
    tampered = copy.deepcopy(report)
    tampered[section][0][field] = replacement
    _resign(tampered)
    with pytest.raises(ValueError, match=match):
        analysis.validate_gemma_iterative_conformal_route_report(tampered)


def test_candidate_execution_and_observation_receipts_are_bound() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs()
    )
    for field in (
        "candidate_execution_sha256",
        "candidate_observation_sha256",
    ):
        tampered = copy.deepcopy(report)
        tampered["oof_rows"][0][field] = "0" * 64
        _resign(tampered)
        with pytest.raises(ValueError, match="does not replay"):
            analysis.validate_gemma_iterative_conformal_route_report(tampered)


def test_retained_full_fit_is_exact_and_retained_only() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs()
    )
    tampered = copy.deepcopy(report)
    tampered["retained_full_fit"]["full_fit"][
        "normal_condition_number"
    ] += 1.0
    _resign(tampered)
    with pytest.raises(ValueError, match="full fit does not replay"):
        analysis.validate_gemma_iterative_conformal_route_report(tampered)

    rejected_inputs = _inputs(
        candidate_deltas=_candidate_deltas(wins=0, loss_delta=0.12),
        include_retained_receipt=False,
    )
    rejected_inputs["retained_full_fit_receipt"] = report[
        "retained_full_fit"
    ]
    with pytest.raises(ValueError, match="rejected.*full fit"):
        analysis.build_gemma_iterative_conformal_route_report(
            **rejected_inputs
        )


def test_report_hash_and_derived_decisions_are_replayed() -> None:
    report = analysis.build_gemma_iterative_conformal_route_report(
        **_inputs()
    )
    tampered_hash = copy.deepcopy(report)
    tampered_hash["decision"]["retained"] = False
    with pytest.raises(ValueError, match="report hash mismatch"):
        analysis.validate_gemma_iterative_conformal_route_report(
            tampered_hash
        )

    resigned = copy.deepcopy(tampered_hash)
    _resign(resigned)
    with pytest.raises(ValueError, match="derived state"):
        analysis.validate_gemma_iterative_conformal_route_report(resigned)


def test_nonfinite_and_tensor_like_payloads_are_rejected() -> None:
    inputs = _inputs()
    inputs["audit"] = dict(inputs["audit"])
    inputs["audit"]["raw_tensor"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="audit fields"):
        analysis.build_gemma_iterative_conformal_route_report(**inputs)

    inputs = _inputs()
    inputs["oof_rows"] = copy.deepcopy(inputs["oof_rows"])
    inputs["oof_rows"][0][
        "predicted_candidate_signed_delta_nll_per_token"
    ] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analysis.build_gemma_iterative_conformal_route_report(**inputs)
