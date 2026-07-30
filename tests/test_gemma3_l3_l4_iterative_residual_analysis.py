from __future__ import annotations

import copy
import hashlib

import pytest

import fisher_graph.gemma3_l3_l4_iterative_residual_analysis as analysis
import fisher_graph.gemma3_l3_l4_iterative_residual_boost as boost
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_residual_boost import (
    GemmaIterativeResidualFitRecord,
    fit_gemma_iterative_residual_fold,
    gemma_causal_position_scale_provider_artifact_sha256,
)


def _hash(character: str) -> str:
    return character * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _at(
    value: float | tuple[float, ...],
    index: int,
) -> float:
    return float(value if isinstance(value, float) else value[index])


def _inputs(
    *,
    parent_delta: float | tuple[float, ...] = 0.10,
    candidate_delta: float | tuple[float, ...] = 0.08,
    candidate_kl: float = 0.9,
    candidate_matches: int = 8,
    learned_parameter_count: int = 4,
    include_retained_receipt: bool = True,
) -> dict[str, object]:
    parent: list[GemmaH4DampingFiniteNLLObservation] = []
    candidate: list[GemmaH4DampingFiniteNLLObservation] = []
    fits: list[GemmaIterativeResidualFitRecord] = []
    manifest: dict[str, str] = {}
    examples_by_family: dict[str, list[str]] = {}
    flat_index = 0
    for family_index in range(8):
        family = f"family-{family_index}"
        examples_by_family[family] = []
        for example_index in range(2):
            example = f"{family}-example-{example_index}"
            parent_value = _at(parent_delta, flat_index)
            candidate_value = _at(candidate_delta, flat_index)
            flat_index += 1
            examples_by_family[family].append(example)
            manifest[example] = family
            source_hash = f"{family_index:x}{example_index:x}".ljust(64, "a")
            target_hash = f"{example_index:x}{family_index:x}".ljust(64, "b")
            parent_row = GemmaH4DampingFiniteNLLObservation(
                example_id=example,
                family_id=family,
                supervised_tokens=10,
                source_summed_nll=10.0,
                candidate_summed_nll=10.0 + 10.0 * parent_value,
                source_to_candidate_summed_kl=1.0,
                top1_matches=8,
                source_logits_sha256=source_hash,
                candidate_logits_sha256=(
                    f"{family_index:x}{example_index:x}".ljust(64, "c")
                ),
                targets_sha256=target_hash,
            )
            candidate_row = GemmaH4DampingFiniteNLLObservation(
                example_id=example,
                family_id=family,
                supervised_tokens=10,
                source_summed_nll=10.0,
                candidate_summed_nll=10.0 + 10.0 * candidate_value,
                source_to_candidate_summed_kl=candidate_kl,
                top1_matches=candidate_matches,
                source_logits_sha256=source_hash,
                candidate_logits_sha256=(
                    f"{family_index:x}{example_index:x}".ljust(64, "d")
                ),
                targets_sha256=target_hash,
            )
            fit = GemmaIterativeResidualFitRecord(
                example_id=example,
                family_id=family,
                model_inputs_sha256=(
                    f"{family_index:x}{example_index:x}".ljust(64, "e")
                ),
                parent_execution_sha256=(
                    f"{family_index:x}{example_index:x}".ljust(64, "f")
                ),
                parent_observation_sha256=parent_row.observation_sha256,
                supervised_tokens=10,
                parent_signed_delta_nll_per_token=parent_value,
                jacobian_by_bin=(1.0, 0.0, 0.0, 0.0),
                active_rows_by_bin=(4, 4, 8, 4),
            )
            parent.append(parent_row)
            candidate.append(candidate_row)
            fits.append(fit)

    lineage = {
        "parent_artifact_sha256": _hash("1"),
        "parent_h4_head_sha256": _hash("2"),
        "accepted_x4_head_sha256": _hash("3"),
        "bridge_binding_sha256": _hash("4"),
        "model_sha256": _hash("5"),
        "adapter_execution_sha256": _hash("6"),
        "fit_manifest_sha256": _hash("7"),
        "factorial_report_sha256": _hash("8"),
        "factorial_report_file_sha256": _hash("9"),
    }
    folds = [
        fit_gemma_iterative_residual_fold(
            tuple(fit for fit in fits if fit.family_id != family),
            held_family_id=family,
        ).to_dict()
        for family in sorted(examples_by_family)
    ]
    fold_by_family = {str(row["held_family_id"]): row for row in folds}
    provider_by_family = {
        family: gemma_causal_position_scale_provider_artifact_sha256(
            parent_artifact_sha256=lineage["parent_artifact_sha256"],
            parent_h4_artifact_sha256=lineage[
                "parent_h4_head_sha256"
            ],
            bridge_binding_sha256=lineage["bridge_binding_sha256"],
            fold_receipt_sha256=str(fold["fold_receipt_sha256"]),
            coefficients_by_bin=fold["coefficients_by_bin"],
        )
        for family, fold in fold_by_family.items()
    }
    fit_by_example = {fit.example_id: fit for fit in fits}
    candidate_by_example = {row.example_id: row for row in candidate}
    candidate_execution_by_example = {
        example: _digest(f"candidate-execution:{example}")
        for example in sorted(manifest)
    }
    model_inputs_by_example = {
        fit.example_id: fit.model_inputs_sha256 for fit in fits
    }
    parent_execution_by_example = {
        fit.example_id: fit.parent_execution_sha256 for fit in fits
    }
    oof = []
    for parent_row in parent:
        candidate_row = candidate_by_example[parent_row.example_id]
        fold = fold_by_family[parent_row.family_id]
        fit = fit_by_example[parent_row.example_id]
        coefficients = tuple(float(value) for value in fold["coefficients_by_bin"])
        predicted = (
            fit.parent_signed_delta_nll_per_token
            + sum(
                left * right
                for left, right in zip(
                    fit.jacobian_by_bin,
                    coefficients,
                    strict=True,
                )
            )
        )
        oof.append(
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
                "jacobian_by_bin": fit.jacobian_by_bin,
                "coefficients_by_bin": coefficients,
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
        "logical_macs_per_token_upper_bound": 640,
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
            "fit_only_two_phase_family_blocked_iterative_residual"
        ),
        "example_count": 16,
        "family_count": 8,
        "outer_fold_count": 8,
        "position_bin_count": 4,
        "position_bin_semantics": (
            "causal_logical_position_[0_3]_[4_7]_[8_15]_[16_plus]"
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
        "source_execution_sha256": lineage["adapter_execution_sha256"],
        "parent_artifact_sha256": lineage["parent_artifact_sha256"],
        "parent_h4_artifact_sha256": lineage["parent_h4_head_sha256"],
        "accepted_x4_head_sha256": lineage["accepted_x4_head_sha256"],
        "fit_manifest_sha256": lineage["fit_manifest_sha256"],
        "residual_width": 640,
        "parent_prepared_float_scalar_count": 438_144,
        "parent_logical_macs_per_token_upper_bound": 252_736,
        "bridge_binding_sha256": lineage["bridge_binding_sha256"],
        "parent_execution_sha256s": tuple(
            sorted(parent_execution_by_example.values())
        ),
        "parent_execution_sha256_by_example": parent_execution_by_example,
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
        "fold_linearization_extrapolation_count": sum(
            bool(row["linearization_extrapolation"]) for row in folds
        ),
        "coefficient_clipping_interpretation": (
            "linearization_extrapolation_not_free_improvement"
        ),
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }
    retained_receipt = None
    if include_retained_receipt:
        full_fit = fit_gemma_iterative_residual_fold(
            fits,
            held_family_id="__full_fit__",
        )
        retained_payload = {
            "provider_artifact_sha256": (
                gemma_causal_position_scale_provider_artifact_sha256(
                    parent_artifact_sha256=lineage[
                        "parent_artifact_sha256"
                    ],
                    parent_h4_artifact_sha256=lineage[
                        "parent_h4_head_sha256"
                    ],
                    bridge_binding_sha256=lineage[
                        "bridge_binding_sha256"
                    ],
                    fold_receipt_sha256=full_fit.fold_receipt_sha256,
                    coefficients_by_bin=full_fit.coefficients_by_bin,
                )
            ),
            "parent_artifact_sha256": lineage["parent_artifact_sha256"],
            "parent_h4_head_sha256": lineage["parent_h4_head_sha256"],
            "bridge_binding_sha256": lineage["bridge_binding_sha256"],
            "learned_parameter_count": learned_parameter_count,
            "logical_macs_per_token_upper_bound": 640,
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
        "oof_rows": oof,
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


def test_relative_candidate_is_retained_before_absolute_fidelity() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    analysis.validate_gemma_iterative_residual_report(report)
    assert report["decision"]["retained"] is True
    assert report["decision"]["ready_for_new_selection"] is False
    paired = report["metrics"]["paired"]
    assert paired[
        "family_macro_mean_prompt_absolute_delta_nll_per_token"
    ]["relative_improvement"] == pytest.approx(0.20)
    assert paired["strict_family_win_count"] == 8
    secondary_names = {
        row["metric"] for row in paired["secondary_metrics"]
    }
    assert (
        "per_prompt_p90_top1_disagreement_to_source"
        in secondary_names
    )
    assert (
        "per_prompt_p10_top1_disagreement_to_source"
        not in secondary_names
    )
    assert (
        "prompt_p90_top1_disagreement_regression_at_most_2pct"
        in paired["gates"]
    )
    assert (
        "prompt_p10_top1_disagreement_regression_at_most_2pct"
        not in paired["gates"]
    )
    linear = report["metrics"]["linearization"]
    assert linear["prompt_count"] == 16
    assert linear["predicted_vs_exact_rmse"] > 0.0


def test_prompt_absolute_metric_rejects_signed_cancellation() -> None:
    parent_deltas = tuple(
        0.10 if index % 2 == 0 else -0.10 for index in range(16)
    )
    candidate_deltas = tuple(
        0.11 if index % 2 == 0 else -0.11 for index in range(16)
    )
    values = _inputs(
        parent_delta=parent_deltas,
        candidate_delta=candidate_deltas,
        include_retained_receipt=False,
    )
    report = analysis.build_gemma_iterative_residual_report(**values)
    assert report["decision"]["retained"] is False
    assert report["metrics"]["parent"]["aggregate"][
        "delta_nll_per_token"
    ] == pytest.approx(0.0)
    assert report["metrics"]["candidate"]["aggregate"][
        "delta_nll_per_token"
    ] == pytest.approx(0.0)
    assert report["metrics"]["paired"][
        "family_macro_mean_prompt_absolute_delta_nll_per_token"
    ]["relative_improvement"] == pytest.approx(-0.10)
    assert (
        report["metrics"]["paired"]["strict_family_win_count"]
        == 0
    )


def test_resource_envelope_is_a_hard_retention_gate() -> None:
    values = _inputs(
        learned_parameter_count=5,
        include_retained_receipt=False,
    )
    report = analysis.build_gemma_iterative_residual_report(**values)
    assert report["decision"]["behavior_relative_gates_passed"] is True
    assert report["decision"]["resource_gates"]["passed"] is False
    assert report["decision"]["retained"] is False


def test_validator_replays_derived_decision_after_resigning() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["decision"]["retained"] = False
    _resign(changed)
    with pytest.raises(ValueError, match="derived state"):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_fold_leakage_after_resigning() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    fold = changed["fold_receipts"][0]
    held = fold["held_family_id"]
    held_example = next(
        example
        for example, family in changed["manifest"][
            "family_by_example"
        ].items()
        if family == held
    )
    train_examples = list(fold["train_example_ids"])
    train_examples[0] = held_example
    fold["train_example_ids"] = train_examples
    _resign(changed)
    with pytest.raises(ValueError, match="leaks"):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_self_consistent_arbitrary_fold_coefficients() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    original = dict(changed["fold_receipts"][0])
    original.pop("fold_receipt_sha256")
    coefficients = tuple(original["coefficients_by_bin"])
    original["coefficients_by_bin"] = (
        coefficients[0] + 0.01,
        *coefficients[1:],
    )
    changed["fold_receipts"][0] = boost.GemmaIterativeResidualFoldFit(
        **original
    ).to_dict()
    _resign(changed)

    with pytest.raises(
        ValueError,
        match="fold coefficients do not replay",
    ):
        analysis.validate_gemma_iterative_residual_report(changed)


@pytest.mark.parametrize(
    "receipt_name",
    (
        "provider_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_observation_sha256",
    ),
)
def test_validator_rejects_oof_provider_execution_or_observation_rebinding(
    receipt_name: str,
) -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
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
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_fit_signed_delta_contradiction() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    original = dict(changed["fit_records"][0])
    original.pop("fit_record_sha256")
    original["parent_signed_delta_nll_per_token"] += 1.0
    changed["fit_records"][0] = GemmaIterativeResidualFitRecord(
        **original
    ).to_dict()
    _resign(changed)

    with pytest.raises(
        ValueError,
        match="fit record differs from its parent observation",
    ):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_lineage_rebinding() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["lineage"]["model_sha256"] = _hash("a")
    _resign(changed)

    with pytest.raises(
        ValueError,
        match="lineage, execution, and resource receipts contradict",
    ):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_resource_receipt_tamper() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["resources"]["resource_receipt_sha256"] = _hash("a")
    _resign(changed)

    with pytest.raises(ValueError, match="resource receipt hash mismatch"):
        analysis.validate_gemma_iterative_residual_report(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    (("raw_prompt", "SECRET"), ("raw_token_ids", [1, 2, 3])),
)
def test_validator_rejects_raw_payloads_hidden_in_execution_audit(
    field: str,
    value: object,
) -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["execution"][field] = value
    _resign(changed)

    with pytest.raises(
        ValueError,
        match="iterative execution audit fields differ",
    ):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_clipping_count_that_contradicts_folds() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["execution"]["fold_linearization_extrapolation_count"] = 1
    _resign(changed)

    with pytest.raises(
        ValueError,
        match="execution clipping count differs from replayed fold fits",
    ):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_retained_report_binds_and_replays_full_fit_provider() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    retained = report["retained_full_fit"]
    assert retained is not None
    assert retained["full_fit"]["held_family_id"] == "__full_fit__"
    assert len(retained["full_fit"]["train_example_ids"]) == 16
    assert len(retained["full_fit"]["train_family_ids"]) == 8
    analysis.validate_gemma_iterative_residual_report(report)

    changed = copy.deepcopy(report)
    changed["retained_full_fit"]["provider_artifact_sha256"] = _hash("a")
    _resign(changed)
    with pytest.raises(
        ValueError,
        match="retained provider lineage or resources differ",
    ):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_validator_rejects_collection_receipt_tamper() -> None:
    report = analysis.build_gemma_iterative_residual_report(**_inputs())
    changed = copy.deepcopy(report)
    changed["collection_sha256"] = _hash("a")
    _resign(changed)

    with pytest.raises(ValueError, match="derived state does not replay"):
        analysis.validate_gemma_iterative_residual_report(changed)


def test_report_is_deterministic_under_input_reordering() -> None:
    values = _inputs()
    forward = analysis.build_gemma_iterative_residual_report(**values)
    for name in (
        "parent_observations",
        "candidate_observations",
        "oof_rows",
        "fit_records",
        "fold_receipts",
    ):
        values[name] = list(reversed(values[name]))
    reverse = analysis.build_gemma_iterative_residual_report(**values)
    assert reverse == forward
