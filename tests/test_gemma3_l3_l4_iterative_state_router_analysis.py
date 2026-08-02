from __future__ import annotations

import copy
import hashlib
import json

import pytest

import fisher_graph.gemma3_l3_l4_iterative_state_router_analysis as analysis
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_state_router import (
    GemmaIterativeStateRouterFitRecord,
    fit_gemma_iterative_state_router_fold,
    gemma_causal_top2_balance_provider_artifact_sha256,
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
    rank4: bool = True,
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
        "prior_iteration_report_sha256": _digest("iteration-one-report"),
        "prior_iteration_report_file_sha256": _digest(
            "iteration-one-report-file"
        ),
        "prior_iteration_collection_sha256": _digest(
            "iteration-one-live-collection"
        ),
    }
    decoder_sha256 = _digest("parent-h4-decoder")
    lag_kernel_sha256 = _digest("parent-h4-lag-kernel")
    top_mode_norms = (2.0, 1.0)
    parent: list[GemmaH4DampingFiniteNLLObservation] = []
    candidate: list[GemmaH4DampingFiniteNLLObservation] = []
    fits: list[GemmaIterativeStateRouterFitRecord] = []
    manifest: dict[str, str] = {}

    flat_index = 0
    for family_index in range(8):
        family_id = f"family-{family_index}"
        for example_index in range(2):
            example_id = f"{family_id}-example-{example_index}"
            manifest[example_id] = family_id
            candidate_delta = candidate_deltas[flat_index]
            x = (flat_index + 1) / 20.0
            jacobian = (
                (1.0, x, x * x, x * x * x)
                if rank4
                else (1.0, 0.0, 0.0, 0.0)
            )
            flat_index += 1
            source_logits_sha256 = _digest(f"source:{example_id}")
            targets_sha256 = _digest(f"targets:{example_id}")
            parent_row = GemmaH4DampingFiniteNLLObservation(
                example_id=example_id,
                family_id=family_id,
                supervised_tokens=10,
                source_summed_nll=10.0,
                candidate_summed_nll=11.0,
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
                    f"router-candidate:{example_id}"
                ),
                targets_sha256=targets_sha256,
            )
            fit = GemmaIterativeStateRouterFitRecord(
                example_id=example_id,
                family_id=family_id,
                model_inputs_sha256=_digest(f"inputs:{example_id}"),
                parent_execution_sha256=_digest(
                    f"parent-execution:{example_id}"
                ),
                parent_observation_sha256=(
                    parent_row.observation_sha256
                ),
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
                gated_modal_sha256=_digest(
                    f"gated-modal:{example_id}"
                ),
                supervised_tokens=10,
                parent_signed_delta_nll_per_token=0.10,
                jacobian_by_route_edge=jacobian,
                active_row_count=5 + example_index,
                top_mode_indices=top_mode_indices,
                top_mode_norms=top_mode_norms,
                balance_feature_std=balance_feature_std,
                top2_modal_energy_fraction=top2_modal_energy_fraction,
            )
            parent.append(parent_row)
            candidate.append(candidate_row)
            fits.append(fit)

    folds = [
        fit_gemma_iterative_state_router_fold(
            tuple(
                fit
                for fit in fits
                if fit.family_id != held_family_id
            ),
            held_family_id=held_family_id,
        ).to_dict()
        for held_family_id in sorted(set(manifest.values()))
    ]
    fold_by_family = {
        str(row["held_family_id"]): row for row in folds
    }
    provider_by_family = {
        family_id: gemma_causal_top2_balance_provider_artifact_sha256(
            parent_artifact_sha256=lineage["parent_artifact_sha256"],
            parent_h4_artifact_sha256=lineage[
                "parent_h4_head_sha256"
            ],
            bridge_binding_sha256=lineage["bridge_binding_sha256"],
            decoder_sha256=decoder_sha256,
            lag_kernel_sha256=lag_kernel_sha256,
            fold_receipt_sha256=str(fold["fold_receipt_sha256"]),
            top_mode_indices=top_mode_indices,
            top_mode_norms=top_mode_norms,
            coefficients_by_route_edge=fold[
                "coefficients_by_route_edge"
            ],
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
            for value in fold["coefficients_by_route_edge"]
        )
        predicted = (
            fit.parent_signed_delta_nll_per_token
            + sum(
                left * right
                for left, right in zip(
                    fit.jacobian_by_route_edge,
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
                "jacobian_by_route_edge": fit.jacobian_by_route_edge,
                "coefficients_by_route_edge": coefficients,
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
        "learned_parameter_count": 4,
        "logical_macs_per_token_upper_bound": 6,
        "derived_constant_float_count": 2,
        "runtime_state_float_count_per_sequence": 2,
        "nonlinear_scalar_ops_per_token_upper_bound": 5,
        "serving_model_forward_count": 1,
        "parent_head_reused_not_duplicated": True,
        "parent_artifact_sha256": lineage["parent_artifact_sha256"],
        "parent_h4_head_sha256": lineage["parent_h4_head_sha256"],
        "candidate_provider_artifact_sha256_by_family": (
            provider_by_family
        ),
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
            "fit_only_two_phase_family_blocked_iterative_state_router"
        ),
        "example_count": 16,
        "family_count": 8,
        "outer_fold_count": 8,
        "route_matrix_shape": (2, 2),
        "route_edge_order": ("0_to_0", "0_to_1", "1_to_0", "1_to_1"),
        "routed_parent_decoder_mode_indices": top_mode_indices,
        "route_state_semantics": (
            "top2_parent_lag_b_modal_cumulative_balance_v1"
        ),
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
            "operator_norm_projection_is_linearization_extrapolation"
        ),
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }

    retained_receipt = None
    if include_retained_receipt:
        full_fit = fit_gemma_iterative_state_router_fold(
            fits,
            held_family_id="__full_fit__",
        )
        retained_payload = {
            "provider_artifact_sha256": (
                gemma_causal_top2_balance_provider_artifact_sha256(
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
                    coefficients_by_route_edge=(
                        full_fit.coefficients_by_route_edge
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
            "learned_parameter_count": 4,
            "logical_macs_per_token_upper_bound": 6,
            "derived_constant_float_count": 2,
            "runtime_state_float_count_per_sequence": 2,
            "nonlinear_scalar_ops_per_token_upper_bound": 5,
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


def test_passing_report_replays_all_retention_gates() -> None:
    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    analysis.validate_gemma_iterative_state_router_report(report)

    assert report["decision"]["retained"] is True
    assert report["decision"]["behavior_relative_gates"]["passed"] is True
    assert report["decision"]["scientific_gates"]["passed"] is True
    assert report["decision"]["resource_gates"]["passed"] is True
    paired = report["metrics"]["paired"]
    assert paired["baseline_arm_id"] == "accepted_x4_plus_lag_b_parent"
    assert paired["challenger_arm_id"] == (
        "causal_top2_modal_state_router"
    )
    assert paired["strict_family_win_count"] == 8
    assert paired[
        "family_macro_mean_prompt_absolute_delta_nll_per_token"
    ]["relative_improvement"] == pytest.approx(0.20)
    assert {
        "parent",
        "state_router",
        "relative_improvement",
    } == set(
        paired[
            "family_macro_mean_prompt_absolute_delta_nll_per_token"
        ]
    )
    linearization = report["metrics"]["linearization"]
    assert linearization["prompt_count"] == 16
    assert 0.0 <= linearization[
        "predicted_vs_exact_sign_agreement"
    ] <= 1.0
    assert linearization["predicted_vs_exact_rmse"] >= 0.0
    activity = report["metrics"]["router_activity_and_coverage"]
    assert activity["full_rank_fold_count"] == 8
    assert activity["total_active_row_count"] == 88
    assert set(activity["fold_coverage_fraction_by_route_edge"]) == {
        "0_to_0",
        "0_to_1",
        "1_to_0",
        "1_to_1",
    }


def test_nonleading_top_modes_and_json_round_trip_replay() -> None:
    report = analysis.build_gemma_iterative_state_router_report(
        **_inputs(top_mode_indices=(3, 1))
    )
    serialized = json.loads(json.dumps(report))
    analysis.validate_gemma_iterative_state_router_report(serialized)
    assert serialized["execution"][
        "routed_parent_decoder_mode_indices"
    ] == [3, 1]


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
def test_scientific_feature_gates_are_hard_retention_gates(
    field: str,
    value: float,
    gate: str,
) -> None:
    inputs = _inputs(
        **{field: value},
        include_retained_receipt=False,
    )
    report = analysis.build_gemma_iterative_state_router_report(**inputs)
    assert report["decision"]["scientific_gates"][gate] is False
    assert report["decision"]["retained"] is False


def test_every_fold_must_have_exact_route_rank_four() -> None:
    report = analysis.build_gemma_iterative_state_router_report(
        **_inputs(rank4=False, include_retained_receipt=False)
    )
    assert report["decision"]["scientific_gates"][
        "all_fold_weighted_design_ranks_exactly_4"
    ] is False
    assert report["metrics"]["router_activity_and_coverage"][
        "full_rank_fold_count"
    ] == 0
    assert report["decision"]["retained"] is False


def test_exactly_six_family_wins_passes_the_behavior_ladder() -> None:
    report = analysis.build_gemma_iterative_state_router_report(
        **_inputs(
            candidate_deltas=_candidate_deltas(
                wins=6,
                loss_delta=0.101,
            )
        )
    )
    assert report["metrics"]["paired"]["strict_family_win_count"] == 6
    assert report["decision"]["behavior_relative_gates"]["passed"] is True
    assert report["decision"]["retained"] is True


def test_five_family_wins_fails_even_when_macro_and_worst_pass() -> None:
    report = analysis.build_gemma_iterative_state_router_report(
        **_inputs(
            candidate_deltas=_candidate_deltas(
                wins=5,
                loss_delta=0.10,
            ),
            include_retained_receipt=False,
        )
    )
    gates = report["decision"]["behavior_relative_gates"]
    assert gates["family_macro_error_regression_at_most_2pct"] is True
    assert gates["worst_family_improvement_at_least_minus_2pct"] is True
    assert gates["strict_family_win_count_at_least_6_of_8"] is False
    assert report["decision"]["retained"] is False


def test_worst_family_cannot_regress_more_than_two_percent() -> None:
    report = analysis.build_gemma_iterative_state_router_report(
        **_inputs(
            candidate_deltas=_candidate_deltas(
                wins=7,
                loss_delta=0.103,
            ),
            include_retained_receipt=False,
        )
    )
    gates = report["decision"]["behavior_relative_gates"]
    assert gates["strict_family_win_count_at_least_6_of_8"] is True
    assert gates["family_macro_error_regression_at_most_2pct"] is True
    assert gates["worst_family_improvement_at_least_minus_2pct"] is False
    assert report["decision"]["retained"] is False


def test_secondary_metric_regression_is_a_hard_gate() -> None:
    report = analysis.build_gemma_iterative_state_router_report(
        **_inputs(
            candidate_kl=1.03,
            include_retained_receipt=False,
        )
    )
    gates = report["decision"]["behavior_relative_gates"]
    assert gates["family_macro_kl_regression_at_most_2pct"] is False
    assert report["decision"]["retained"] is False


def test_fold_coefficients_must_replay_from_family_disjoint_rows() -> None:
    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    changed = copy.deepcopy(report)
    coefficients = list(
        changed["fold_receipts"][0]["coefficients_by_route_edge"]
    )
    coefficients[0] += 0.001
    changed["fold_receipts"][0]["coefficients_by_route_edge"] = coefficients
    _resign(changed)

    with pytest.raises(
        ValueError,
        match="fold coefficients do not replay",
    ):
        analysis.validate_gemma_iterative_state_router_report(changed)


@pytest.mark.parametrize(
    "receipt_name",
    (
        "provider_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_observation_sha256",
    ),
)
def test_oof_rows_bind_provider_execution_and_observation(
    receipt_name: str,
) -> None:
    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    changed = copy.deepcopy(report)
    target = changed["oof_rows"][0]
    replacement = next(
        row[receipt_name]
        for row in changed["oof_rows"][1:]
        if row[receipt_name] != target[receipt_name]
    )
    target[receipt_name] = replacement
    _resign(changed)

    with pytest.raises(ValueError, match="OOF row does not replay"):
        analysis.validate_gemma_iterative_state_router_report(changed)


def test_fit_record_binds_transient_tensor_receipts() -> None:
    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["fit_records"][0]["balance_feature_sha256"] = _digest(
        "different-balance-feature"
    )
    _resign(changed)

    with pytest.raises(ValueError, match="fit-record hash mismatch"):
        analysis.validate_gemma_iterative_state_router_report(changed)


def test_resource_receipt_tamper_is_rejected() -> None:
    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["resources"]["resource_receipt_sha256"] = _digest(
        "tampered-resource"
    )
    _resign(changed)

    with pytest.raises(ValueError, match="resource receipt hash mismatch"):
        analysis.validate_gemma_iterative_state_router_report(changed)


def test_retained_report_requires_and_replays_full_fit_provider() -> None:
    values = _inputs(include_retained_receipt=False)
    with pytest.raises(ValueError, match="must bind its full-fit provider"):
        analysis.build_gemma_iterative_state_router_report(**values)

    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    assert report["retained_full_fit"]["full_fit"][
        "held_family_id"
    ] == "__full_fit__"
    changed = copy.deepcopy(report)
    changed["retained_full_fit"]["provider_artifact_sha256"] = _digest(
        "wrong-full-fit-provider"
    )
    _resign(changed)
    with pytest.raises(
        ValueError,
        match="retained router provider lineage or resources differ",
    ):
        analysis.validate_gemma_iterative_state_router_report(changed)


def test_prior_iteration_collection_is_required_and_independent() -> None:
    missing = _inputs()
    del missing["lineage"]["prior_iteration_collection_sha256"]
    with pytest.raises(ValueError, match="lineage fields differ"):
        analysis.build_gemma_iterative_state_router_report(**missing)

    aliased = _inputs()
    aliased["lineage"]["prior_iteration_collection_sha256"] = aliased[
        "lineage"
    ]["prior_iteration_report_sha256"]
    with pytest.raises(ValueError, match="aliases its prerequisite"):
        analysis.build_gemma_iterative_state_router_report(**aliased)


def test_validator_replays_derived_decision_after_resigning() -> None:
    report = analysis.build_gemma_iterative_state_router_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["decision"]["retained"] = False
    _resign(changed)
    with pytest.raises(ValueError, match="derived state does not replay"):
        analysis.validate_gemma_iterative_state_router_report(changed)


def test_report_is_deterministic_under_input_reordering() -> None:
    values = _inputs()
    forward = analysis.build_gemma_iterative_state_router_report(**values)
    for key in (
        "parent_observations",
        "candidate_observations",
        "oof_rows",
        "fit_records",
        "fold_receipts",
    ):
        values[key] = list(reversed(values[key]))
    reverse = analysis.build_gemma_iterative_state_router_report(**values)
    assert reverse == forward
