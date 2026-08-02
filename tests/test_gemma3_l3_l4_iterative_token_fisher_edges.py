from __future__ import annotations

import copy
from dataclasses import dataclass
import json

import pytest
import torch

from fisher_graph import gemma3_l3_l4_iterative_occupancy_route as route
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
    GemmaIterativeTokenOccupancyTangentRecord,
    build_gemma_iterative_token_occupancy_tangent_record,
    parse_gemma_iterative_token_occupancy_tangent_record,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
)


_EXPECTED_COMBINED_ORDER = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
    "ew_occupancy_contrast_real",
    "ew_occupancy_contrast_imag",
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _parent() -> GemmaCausalResidualHead:
    return GemmaCausalResidualHead(
        site="layer.4.output",
        parent_runtime_binding_sha256=_sha(1),
        residual_map_sha256=_sha(2),
        analysis_artifact_sha256=_sha(3),
        fit_manifest_sha256=_sha(4),
        bridge_binding_sha256=_sha(5),
        decoder=torch.eye(3, dtype=torch.float64),
        lag_kernel=torch.tensor(
            [
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.1],
                ]
            ],
            dtype=torch.float64,
        ),
        state_kernel=torch.empty((0, 0), dtype=torch.float64),
        conditioning="l3_source_modes",
        ridge=1.0e-6,
        fit_row_count=8,
        family_ids=("fit-a", "fit-b"),
        fit_sequence_sha256s=(_sha(6), _sha(7)),
        fit_objective="candidate_nll_vjp_metric_ridge_v1",
        weighted_residual_rmse=1.0,
        normalized_nll_direction_rmse=1.0,
        linearized_nll_residual_rmse=1.0,
    )


def _prefix(
    *,
    active: torch.Tensor | None = None,
) -> Gemma3L3L4OnePassPrefix:
    source_modes = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.5],
                [0.5, -1.0],
            ]
        ],
        dtype=torch.float64,
    )
    batch, length, _rank = source_modes.shape
    valid = torch.ones((batch, length), dtype=torch.bool)
    if active is None:
        active = valid.clone()
    return Gemma3L3L4OnePassPrefix(
        source_modes=source_modes,
        clamped_y3=torch.zeros(batch, length, 3, dtype=torch.float64),
        predicted_target_modal_delta=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        decoded_base_x4_delta=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        logical_positions=torch.arange(
            length,
            dtype=torch.int64,
        ).expand(batch, -1),
        valid_target_mask=valid,
        source_eligible_mask=valid.clone(),
        target_affected_mask=active,
        bridge_binding_sha256=_sha(5),
    )


class _Example:
    example_id: str = "example"
    family_id: str = "family"
    model_inputs_sha256: str = _sha(30)


@dataclass
class _Execution:
    prefix: Gemma3L3L4OnePassPrefix
    candidate_h4: torch.Tensor
    h4_head_sha256: str
    artifact_sha256: str = _sha(31)
    model_inputs_sha256: str = _sha(30)

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()


def _observation(
    supervised_tokens: int = 5,
) -> GemmaH4DampingFiniteNLLObservation:
    return GemmaH4DampingFiniteNLLObservation(
        example_id="example",
        family_id="family",
        supervised_tokens=supervised_tokens,
        source_summed_nll=4.0,
        candidate_summed_nll=4.5,
        source_to_candidate_summed_kl=0.2,
        top1_matches=min(3, supervised_tokens),
        source_logits_sha256=_sha(32),
        candidate_logits_sha256=_sha(33),
        targets_sha256=_sha(34),
    )


def _summed_gradient() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.25, -0.5, 0.1],
                [1.0, 0.25, -0.2],
                [-0.5, 0.75, 0.3],
                [0.2, -0.1, 0.4],
                [-0.7, 0.5, -0.6],
            ]
        ],
        dtype=torch.float64,
    )


def _causal_token_loss_gradient_bank(
    summed_gradient: torch.Tensor,
) -> torch.Tensor:
    """Split one summed-loss VJP into exact, nontrivial causal token VJPs."""

    _batch, length, _width = summed_gradient.shape
    result = torch.zeros(
        length,
        *summed_gradient.shape,
        dtype=summed_gradient.dtype,
    )
    last = length - 1
    for activation_position in range(length):
        value = summed_gradient[:, activation_position]
        if activation_position == last:
            result[last, :, activation_position] = value
        else:
            # The current token loss and the final later token loss each own
            # half of this activation-position gradient. This makes the bank
            # genuinely cross-position while retaining an exact binary sum.
            result[
                activation_position,
                :,
                activation_position,
            ] = 0.5 * value
            result[last, :, activation_position] = 0.5 * value
    assert torch.equal(result.sum(dim=0), summed_gradient)
    return result


def _build(
    token_loss_gradients: torch.Tensor | None = None,
    *,
    supervised_token_logical_positions: tuple[int, ...] = (0, 1, 2, 3, 4),
    active: torch.Tensor | None = None,
    supervised_tokens: int = 5,
) -> GemmaIterativeTokenOccupancyTangentRecord:
    parent = _parent()
    prefix = _prefix(active=active)
    if token_loss_gradients is None:
        token_loss_gradients = _causal_token_loss_gradient_bank(
            _summed_gradient()
        )
    return build_gemma_iterative_token_occupancy_tangent_record(
        example=_Example(),
        parent_execution=_Execution(
            prefix=prefix,
            candidate_h4=torch.zeros(
                1,
                5,
                3,
                dtype=torch.float64,
            ),
            h4_head_sha256=parent.artifact_sha256,
        ),
        token_loss_gradients=token_loss_gradients,
        supervised_token_logical_positions=(
            supervised_token_logical_positions
        ),
        parent_h4=parent,
        parent_observation=_observation(supervised_tokens),
    )


def _old_prompt_record(
    gradient: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    supervised_tokens: int = 5,
) -> route.GemmaIterativeOccupancyConformalRouteFitRecord:
    parent = _parent()
    return route.build_gemma_iterative_occupancy_conformal_route_fit_record(
        example=_Example(),
        parent_execution=_Execution(
            prefix=_prefix(active=active),
            candidate_h4=torch.zeros(
                1,
                5,
                3,
                dtype=torch.float64,
            ),
            h4_head_sha256=parent.artifact_sha256,
        ),
        gradient=gradient,
        parent_h4=parent,
        parent_observation=_observation(supervised_tokens),
    )


def _combined_rows(
    record: GemmaIterativeTokenOccupancyTangentRecord,
) -> torch.Tensor:
    return torch.tensor(
        [
            row.tangent_by_combined_occupancy_coordinate
            for row in record.rows
        ],
        dtype=torch.float64,
    )


def test_combined_eight_coordinate_rows_sum_to_both_old_prompt_jacobians(
) -> None:
    summed_gradient = _summed_gradient()
    bank = _causal_token_loss_gradient_bank(summed_gradient)
    token_record = _build(bank)
    old_record = _old_prompt_record(summed_gradient)

    assert TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER == (
        _EXPECTED_COMBINED_ORDER
    )
    assert token_record.coordinate_order == _EXPECTED_COMBINED_ORDER
    assert tuple(
        row.supervised_token_logical_position
        for row in token_record.rows
    ) == (0, 1, 2, 3, 4)
    assert tuple(
        row.supervised_token_ordinal for row in token_record.rows
    ) == (0, 1, 2, 3, 4)

    combined_prompt = _combined_rows(token_record).sum(dim=0) / 5
    cumulative = combined_prompt.index_select(
        0,
        torch.tensor((0, 1, 2, 3, 4, 5)),
    )
    ew = combined_prompt.index_select(
        0,
        torch.tensor((0, 1, 2, 3, 6, 7)),
    )
    torch.testing.assert_close(
        cumulative,
        torch.tensor(
            old_record
            .jacobian_by_cumulative_occupancy_conformal_coefficient,
            dtype=torch.float64,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        ew,
        torch.tensor(
            old_record.jacobian_by_ew_occupancy_conformal_coefficient,
            dtype=torch.float64,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert tuple(cumulative.tolist()) == pytest.approx(
        token_record
        .jacobian_by_cumulative_occupancy_conformal_coefficient,
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert tuple(ew.tolist()) == pytest.approx(
        token_record.jacobian_by_ew_occupancy_conformal_coefficient,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_builder_canonicalizes_supervised_token_input_order_exactly() -> None:
    bank = _causal_token_loss_gradient_bank(_summed_gradient())
    forward = _build(bank)
    reverse = _build(
        bank.flip(0),
        supervised_token_logical_positions=(4, 3, 2, 1, 0),
    )
    assert reverse.to_dict() == forward.to_dict()
    assert (
        reverse.token_tangent_record_sha256
        == forward.token_tangent_record_sha256
    )

    with pytest.raises(ValueError, match="unique"):
        _build(
            bank,
            supervised_token_logical_positions=(0, 1, 1, 3, 4),
        )


def test_inactive_padding_is_excluded_from_every_token_tangent() -> None:
    active = torch.tensor([[True, True, True, False, False]])
    clean = torch.zeros(3, 1, 5, 3, dtype=torch.float64)
    clean[0, 0, 0] = torch.tensor((0.5, -0.25, 0.1))
    clean[1, 0, 0] = torch.tensor((0.2, 0.1, -0.3))
    clean[1, 0, 1] = torch.tensor((0.4, -0.2, 0.5))
    clean[2, 0, :3] = torch.tensor(
        (
            (0.1, 0.2, 0.3),
            (-0.3, 0.1, 0.2),
            (0.6, -0.4, 0.25),
        )
    )
    poisoned_padding = clean.clone()
    poisoned_padding[:, :, 3:] = torch.tensor(
        (
            (
                (1.0e9, -2.0e9, 3.0e9),
                (-4.0e9, 5.0e9, -6.0e9),
            ),
        ),
        dtype=torch.float64,
    )

    clean_record = _build(
        clean,
        supervised_token_logical_positions=(0, 1, 2),
        active=active,
        supervised_tokens=3,
    )
    poisoned_record = _build(
        poisoned_padding,
        supervised_token_logical_positions=(0, 1, 2),
        active=active,
        supervised_tokens=3,
    )
    torch.testing.assert_close(
        _combined_rows(poisoned_record),
        _combined_rows(clean_record),
        rtol=0.0,
        atol=0.0,
    )

    clean_old = _old_prompt_record(
        clean.sum(dim=0),
        active=active,
        supervised_tokens=3,
    )
    poisoned_old = _old_prompt_record(
        poisoned_padding.sum(dim=0),
        active=active,
        supervised_tokens=3,
    )
    assert (
        clean_old.jacobian_by_cumulative_occupancy_conformal_coefficient
        == poisoned_old
        .jacobian_by_cumulative_occupancy_conformal_coefficient
    )
    assert (
        clean_old.jacobian_by_ew_occupancy_conformal_coefficient
        == poisoned_old.jacobian_by_ew_occupancy_conformal_coefficient
    )


def test_nonzero_active_future_gradient_is_rejected_as_noncausal() -> None:
    bank = _causal_token_loss_gradient_bank(_summed_gradient())
    assert torch.equal(
        bank[0, :, 1:],
        torch.zeros_like(bank[0, :, 1:]),
    )
    noncausal = bank.clone()
    noncausal[0, 0, 1, 0] = 1.0e-4

    with pytest.raises(ValueError, match="future"):
        _build(noncausal)


def test_record_json_replay_and_nested_order_or_hash_tamper_fail_closed(
) -> None:
    record = _build()
    payload = json.loads(json.dumps(record.to_dict()))
    restored = parse_gemma_iterative_token_occupancy_tangent_record(
        payload
    )
    assert restored.to_dict() == record.to_dict()
    restored.validate_integrity()

    reversed_rows = copy.deepcopy(payload)
    reversed_rows["rows"].reverse()
    with pytest.raises(ValueError, match="canonical|order|hash"):
        parse_gemma_iterative_token_occupancy_tangent_record(reversed_rows)

    changed_tangent = copy.deepcopy(payload)
    changed_tangent["rows"][0][
        "tangent_by_combined_occupancy_coordinate"
    ][0] += 0.125
    with pytest.raises(ValueError, match="hash"):
        parse_gemma_iterative_token_occupancy_tangent_record(
            changed_tangent
        )

    changed_position = copy.deepcopy(payload)
    changed_position["rows"][0]["supervised_token_logical_position"] = 99
    with pytest.raises(ValueError, match="canonical|position|hash"):
        parse_gemma_iterative_token_occupancy_tangent_record(
            changed_position
        )

    changed_checksum = copy.deepcopy(payload)
    changed_checksum[
        "jacobian_by_ew_occupancy_conformal_coefficient"
    ][4] += 0.25
    with pytest.raises(ValueError, match="hash"):
        parse_gemma_iterative_token_occupancy_tangent_record(
            changed_checksum
        )

    unknown = copy.deepcopy(payload)
    unknown["model_state_dict"] = {"weight": [1.0]}
    with pytest.raises(ValueError, match="fields"):
        parse_gemma_iterative_token_occupancy_tangent_record(unknown)

    object.__setattr__(
        record.rows[0],
        "tangent_by_combined_occupancy_coordinate",
        (
            99.0,
            *record.rows[0].tangent_by_combined_occupancy_coordinate[1:],
        ),
    )
    with pytest.raises(RuntimeError, match="drifted"):
        record.validate_integrity()
