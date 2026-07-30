from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_residual_boost import (
    GemmaCausalPositionScaleH4Provider,
    GemmaIterativeResidualFitRecord,
    GemmaIterativeResidualFoldFit,
    build_gemma_iterative_residual_fit_record,
    causal_position_bin_indices,
    fit_gemma_iterative_residual_fold,
)


_HASH = "a" * 64
_BRIDGE = "b" * 64


class _ParentH4(Gemma3L3L4CorrectionProvider):
    site = "layer.4.output"
    artifact_sha256 = _HASH
    bridge_binding_sha256 = _BRIDGE
    width = 3
    prepared_float_scalar_count = 12
    logical_macs_per_token_upper_bound = 12

    def validate_integrity(self) -> None:
        if self.artifact_sha256 != _HASH:
            raise RuntimeError("drift")

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(realized_state)
        result[prefix.target_affected_mask] = 1
        return result


def _prefix(length: int = 20) -> Gemma3L3L4OnePassPrefix:
    grid = (1, length)
    positions = torch.arange(length, dtype=torch.int64).unsqueeze(0)
    valid = torch.ones(grid, dtype=torch.bool)
    return Gemma3L3L4OnePassPrefix(
        source_modes=torch.zeros(1, length, 2),
        clamped_y3=torch.zeros(1, length, 3),
        predicted_target_modal_delta=torch.zeros(1, length, 2),
        decoded_base_x4_delta=torch.zeros(1, length, 3),
        logical_positions=positions,
        valid_target_mask=valid,
        source_eligible_mask=valid,
        target_affected_mask=valid,
        bridge_binding_sha256=_BRIDGE,
    )


def _fold(
    coefficients: tuple[float, float, float, float],
) -> GemmaIterativeResidualFoldFit:
    return GemmaIterativeResidualFoldFit(
        held_family_id="held",
        train_example_ids=("e0",),
        train_family_ids=("f0",),
        train_fit_record_sha256s=("c" * 64,),
        coefficients_by_bin=coefficients,
        unsupported_bin_indices=(),
        active_rows_by_bin=(4, 4, 8, 4),
        weighted_column_norm_by_bin=(1.0, 1.0, 1.0, 1.0),
        normal_condition_number=1.0,
        linearized_rmse_before=1.0,
        linearized_rmse_after=0.5,
        linearization_extrapolation=False,
    )


def _record(
    example: str,
    family: str,
    *,
    delta: float,
    jacobian: tuple[float, float, float, float],
    active: tuple[int, int, int, int] = (1, 1, 1, 1),
) -> GemmaIterativeResidualFitRecord:
    return GemmaIterativeResidualFitRecord(
        example_id=example,
        family_id=family,
        model_inputs_sha256="1" * 64,
        parent_execution_sha256="2" * 64,
        parent_observation_sha256="3" * 64,
        supervised_tokens=2,
        parent_signed_delta_nll_per_token=delta,
        jacobian_by_bin=jacobian,
        active_rows_by_bin=active,
    )


def test_causal_position_bins_do_not_depend_on_a_future_suffix() -> None:
    short = torch.arange(10, dtype=torch.int64)
    long = torch.arange(30, dtype=torch.int64)
    assert causal_position_bin_indices(short).tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
    ]
    assert torch.equal(
        causal_position_bin_indices(short),
        causal_position_bin_indices(long)[:10],
    )


def test_composite_provider_scales_parent_and_stays_off_support() -> None:
    prefix = _prefix()
    mask = prefix.target_affected_mask.clone()
    mask[:, 19] = False
    prefix = Gemma3L3L4OnePassPrefix(
        source_modes=prefix.source_modes,
        clamped_y3=prefix.clamped_y3,
        predicted_target_modal_delta=prefix.predicted_target_modal_delta,
        decoded_base_x4_delta=prefix.decoded_base_x4_delta,
        logical_positions=prefix.logical_positions,
        valid_target_mask=prefix.valid_target_mask,
        source_eligible_mask=prefix.source_eligible_mask,
        target_affected_mask=mask,
        bridge_binding_sha256=_BRIDGE,
    )
    provider = GemmaCausalPositionScaleH4Provider(
        parent_h4=_ParentH4(),
        parent_artifact_sha256="d" * 64,
        fold_fit=_fold((-0.5, -0.25, 0.25, 0.5)),
    )
    correction = provider.correction(prefix, torch.zeros(1, 20, 3))
    assert torch.equal(correction[0, 0], torch.full((3,), 0.5))
    assert torch.equal(correction[0, 4], torch.full((3,), 0.75))
    assert torch.equal(correction[0, 8], torch.full((3,), 1.25))
    assert torch.equal(correction[0, 16], torch.full((3,), 1.5))
    assert torch.equal(correction[0, 19], torch.zeros(3))
    assert provider.marginal_prepared_float_scalar_count == 4
    assert provider.marginal_logical_macs_per_token_upper_bound == 3


def test_fold_fit_is_family_balanced_and_freezes_unsupported_bins() -> None:
    records = []
    for family_index in range(7):
        family = f"f{family_index}"
        for example_index in range(2):
            records.append(
                _record(
                    f"{family}-{example_index}",
                    family,
                    delta=-0.1,
                    jacobian=(1.0, 0.0, 0.0, 0.0),
                    active=(1, 0, 0, 0),
                )
            )
    fit = fit_gemma_iterative_residual_fold(
        records,
        held_family_id="held",
    )
    assert fit.coefficients_by_bin[0] == pytest.approx(0.1, rel=1e-5)
    assert fit.coefficients_by_bin[1:] == (0.0, 0.0, 0.0)
    assert fit.unsupported_bin_indices == (1, 2, 3)
    assert fit.train_family_ids == tuple(f"f{i}" for i in range(7))


def test_fold_fit_rejects_held_family_leakage() -> None:
    with pytest.raises(ValueError, match="held family leaked"):
        fit_gemma_iterative_residual_fold(
            [_record("e0", "held", delta=0.1, jacobian=(1, 0, 0, 0))],
            held_family_id="held",
        )


@dataclass
class _Example:
    example_id: str = "example"
    family_id: str = "family"
    model_inputs_sha256: str = "4" * 64


@dataclass
class _Execution:
    prefix: Gemma3L3L4OnePassPrefix
    artifact_sha256: str = "5" * 64
    model_inputs_sha256: str = "4" * 64

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()


def test_fit_record_reduces_vjp_to_four_scalars() -> None:
    prefix = _prefix()
    observation = GemmaH4DampingFiniteNLLObservation(
        example_id="example",
        family_id="family",
        supervised_tokens=2,
        source_summed_nll=4.0,
        candidate_summed_nll=4.4,
        source_to_candidate_summed_kl=0.2,
        top1_matches=1,
        source_logits_sha256="6" * 64,
        candidate_logits_sha256="7" * 64,
        targets_sha256="8" * 64,
    )
    record = build_gemma_iterative_residual_fit_record(
        example=_Example(),
        parent_execution=_Execution(prefix),
        gradient=torch.ones(1, 20, 3),
        lag_b_correction=torch.ones(1, 20, 3),
        parent_observation=observation,
    )
    assert record.parent_signed_delta_nll_per_token == pytest.approx(0.2)
    assert record.active_rows_by_bin == (4, 4, 8, 4)
    assert record.jacobian_by_bin == pytest.approx((6.0, 6.0, 12.0, 6.0))
    assert set(record.to_dict()) == {
        "example_id",
        "family_id",
        "model_inputs_sha256",
        "parent_execution_sha256",
        "parent_observation_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_bin",
        "active_rows_by_bin",
        "fit_record_sha256",
    }
