from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest
import torch
from torch import Tensor

from fisher_graph.causal_modal_pair import (
    MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS,
    CausalModalPairPlan,
    EdgeTornModalPairBoundaryContract,
    PreparedCausalModalPair,
    bind_causal_modal_pair_plan,
)
from fisher_graph.causal_edge_jvp import estimate_causal_edge_jvp
from fisher_graph.modal_connectivity_modes import (
    CausalBoundaryTransfer,
    MessageMoments,
    ModalBoundaryPort,
    ModalConnectivityFactor,
    factor_modal_connectivity,
)


FLOAT64 = torch.float64
Y3_MEAN = torch.tensor([0.25, -0.5, 1.5], dtype=FLOAT64)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _boundary(
    **overrides: object,
) -> EdgeTornModalPairBoundaryContract:
    base = EdgeTornModalPairBoundaryContract.for_y3_mean(
        stage3_source_name="fixture.layer3.modal_source",
        stage4_target_name="fixture.layer4.modal_target",
        x4_reference_torn_base_name=(
            "fixture.layer4.mean_source.reference.torn_base"
        ),
        y3_mean=Y3_MEAN,
    )
    values: dict[str, object] = {
        "stage3_source_name": base.stage3_source_name,
        "stage4_target_name": base.stage4_target_name,
        "x4_reference_torn_base_name": (
            base.x4_reference_torn_base_name
        ),
        "y3_mean_sha256": base.y3_mean_sha256,
        "x4_semantics": base.x4_semantics,
        "centered_source_edge_present_in_x4": False,
        "ordinary_path_input_authorized": False,
        "source_replacement_authority": False,
    }
    values.update(overrides)
    return EdgeTornModalPairBoundaryContract(**values)  # type: ignore[arg-type]


def _plan(*, zero_edge: bool = False) -> CausalModalPairPlan:
    edge = torch.tensor(
        [
            [[0.5, -0.25]],
            [[-0.75, 0.4]],
        ],
        dtype=FLOAT64,
    )
    if zero_edge:
        edge.zero_()
    return CausalModalPairPlan(
        boundary_contract=_boundary(),
        source_artifact_sha256=_sha("causal-pair-source"),
        direct_jacobian_proof_sha256=_sha("signed-direct-jvp-proof"),
        x3_mean=torch.tensor([0.5, -1.0], dtype=FLOAT64),
        y3_mean=Y3_MEAN,
        x4_mean=torch.tensor([1.0, 0.0, -0.5, 2.0], dtype=FLOAT64),
        y4_mean=torch.tensor([-1.0, 0.75], dtype=FLOAT64),
        R3=torch.tensor([[2.0, -0.5]], dtype=FLOAT64),
        P3=torch.tensor([[1.0], [-2.0], [0.25]], dtype=FLOAT64),
        R4=torch.tensor(
            [[1.0, 0.0, -1.0, 0.5], [-0.25, 2.0, 0.0, 1.0]],
            dtype=FLOAT64,
        ),
        P4=torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=FLOAT64),
        K=edge,
    )


def _inputs() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor(
            [
                [[1.0, -2.0], [0.0, 0.5], [2.0, -1.0]],
                [[-1.0, 1.0], [0.5, -0.5], [3.0, 2.0]],
            ],
            dtype=FLOAT64,
        ),
        torch.tensor(
            [
                [
                    [1.5, 0.0, -1.0, 2.0],
                    [0.0, 1.0, 0.5, -1.0],
                    [2.0, -0.5, 1.0, 0.0],
                ],
                [
                    [-1.0, 0.0, 2.0, 1.0],
                    [0.5, -2.0, -0.5, 2.5],
                    [1.0, 1.0, 0.0, -2.0],
                ],
            ],
            dtype=FLOAT64,
        ),
    )


def _expanded_grid(
    leading_shape: tuple[int, ...],
    logical_positions: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    sequence_length = leading_shape[-1]
    assert logical_positions.shape in {
        (sequence_length,),
        leading_shape,
    }
    assert valid_mask.shape in {(sequence_length,), leading_shape}
    positions = (
        logical_positions.expand(leading_shape)
        if logical_positions.shape == (sequence_length,)
        else logical_positions
    )
    mask = (
        valid_mask.expand(leading_shape)
        if valid_mask.shape == (sequence_length,)
        else valid_mask
    )
    return positions, mask


def _manual_factorized(
    plan: CausalModalPairPlan,
    x3: Tensor,
    x4_reference: Tensor,
    *,
    logical_positions: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    leading_shape = tuple(x3.shape[:-1])
    positions, mask = _expanded_grid(
        leading_shape,
        logical_positions,
        valid_mask,
    )
    m3 = (x3 - plan.x3_mean) @ plan.R3.T
    y3 = m3 @ plan.P3.T + plan.y3_mean
    sequence_length = leading_shape[-1]
    flat_m3 = m3.reshape(-1, sequence_length, plan.rank3)
    flat_positions = positions.reshape(-1, sequence_length)
    flat_mask = mask.reshape(-1, sequence_length)
    flat_edge = torch.zeros(
        (flat_m3.shape[0], sequence_length, plan.rank4),
        dtype=m3.dtype,
    )
    for outer in range(flat_m3.shape[0]):
        for target in range(sequence_length):
            if not bool(flat_mask[outer, target]):
                continue
            for source in range(sequence_length):
                if not bool(flat_mask[outer, source]):
                    continue
                lag = int(
                    flat_positions[outer, target]
                    - flat_positions[outer, source]
                )
                if 0 <= lag < plan.lag_count:
                    flat_edge[outer, target] += (
                        flat_m3[outer, source] @ plan.K[lag]
                    )
    edge = flat_edge.reshape(*leading_shape, plan.rank4)
    m4 = (x4_reference - plan.x4_mean) @ plan.R4.T + edge
    y4 = m4 @ plan.P4.T + plan.y4_mean
    return y3, y4


def _contiguous_grid() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.ones(3, dtype=torch.bool),
    )


def _binding_factor(
    name: str,
    *,
    causal_order: int,
) -> ModalConnectivityFactor:
    source = _sha(f"binding:{name}")
    input_port = ModalBoundaryPort(
        name=f"{name}.input",
        direction="input",
        causal_order=causal_order,
        width=2,
        owner_id=name,
    )
    output_port = ModalBoundaryPort(
        name=f"{name}.output",
        direction="output",
        causal_order=causal_order,
        width=2,
        owner_id=name,
    )
    offset = (
        torch.tensor([0.25, -0.5], dtype=FLOAT64)
        if causal_order == 3
        else torch.zeros(2, dtype=FLOAT64)
    )
    transfer = CausalBoundaryTransfer(
        source_level_sha256=source,
        input_ports=(input_port,),
        output_ports=(output_port,),
        input_prefixes=((input_port.name,),),
        transfer_matrices=(torch.eye(2, dtype=FLOAT64),),
        affine_offsets=(offset,),
    )

    def moments(port: ModalBoundaryPort) -> MessageMoments:
        return MessageMoments(
            port=port,
            source_level_sha256=source,
            reduction_id="binding-lineage",
            sample_count=12,
            mean=torch.zeros(2, dtype=FLOAT64),
            covariance=torch.eye(2, dtype=FLOAT64),
            fisher=torch.eye(2, dtype=FLOAT64),
        )

    return factor_modal_connectivity(
        transfer,
        (moments(input_port),),
        (moments(output_port),),
        retained_ranks=2,
    ).factors[0]


def test_boundary_contract_binds_the_mean_and_rejects_wrong_base_kinds() -> None:
    boundary = _boundary()
    assert (
        boundary.x4_semantics
        == MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS
    )
    assert boundary.centered_source_edge_present_in_x4 is False
    assert boundary.ordinary_path_input_authorized is False
    assert boundary.source_replacement_authority is False
    boundary.validate_integrity()

    for forbidden_name in (
        "fixture.layer4.zero_source.reference.torn_base",
        "fixture.layer4.ordinary.mean_source.reference.torn_base",
        "fixture.layer4.native.mean_source.reference.torn_base",
    ):
        with pytest.raises(ValueError, match="mean-source reference torn base"):
            _boundary(x4_reference_torn_base_name=forbidden_name)
    with pytest.raises(ValueError, match="mean-source reference"):
        _boundary(x4_semantics="zero_source_edge_torn_base")
    with pytest.raises(ValueError, match="ordinary/native-path"):
        _boundary(ordinary_path_input_authorized=True)
    with pytest.raises(ValueError, match="replacement authority"):
        _boundary(source_replacement_authority=True)
    with pytest.raises(ValueError, match="centered source edge"):
        _boundary(centered_source_edge_present_in_x4=True)

    with pytest.raises(ValueError, match="not bound to the plan y3_mean"):
        CausalModalPairPlan(
            boundary_contract=_boundary(y3_mean_sha256=_sha("wrong-mean")),
            source_artifact_sha256=_sha("source"),
            direct_jacobian_proof_sha256=_sha("proof"),
            x3_mean=torch.zeros(2),
            y3_mean=Y3_MEAN,
            x4_mean=torch.zeros(4),
            y4_mean=torch.zeros(2),
            R3=torch.ones(1, 2),
            P3=torch.ones(3, 1),
            R4=torch.ones(2, 4),
            P4=torch.ones(2, 2),
            K=torch.zeros(1, 1, 2),
        )


def test_plan_is_canonical_cloned_and_authenticated() -> None:
    source_R3 = torch.tensor([[2.0, -0.5]], dtype=torch.float32)
    plan = CausalModalPairPlan(
        boundary_contract=_boundary(),
        source_artifact_sha256=_sha("source"),
        direct_jacobian_proof_sha256=_sha("proof"),
        x3_mean=torch.zeros(2),
        y3_mean=Y3_MEAN,
        x4_mean=torch.zeros(4),
        y4_mean=torch.zeros(2),
        R3=source_R3,
        P3=torch.ones(3, 1),
        R4=torch.ones(2, 4),
        P4=torch.ones(2, 2),
        K=torch.zeros(2, 1, 2),
    )
    source_R3.add_(100.0)

    assert plan.R3.dtype == FLOAT64
    assert plan.R3.device.type == "cpu"
    assert plan.R3.is_contiguous()
    torch.testing.assert_close(
        plan.R3,
        torch.tensor([[2.0, -0.5]], dtype=FLOAT64),
    )
    assert len(plan.artifact_sha256) == 64
    assert plan.source_sha256 == plan.source_artifact_sha256
    assert plan.proof_sha256 == plan.direct_jacobian_proof_sha256
    plan.validate_integrity()

    with pytest.raises(FrozenInstanceError):
        plan.source_artifact_sha256 = _sha("changed")  # type: ignore[misc]
    plan.K[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        plan.validate_integrity()


def test_binder_authenticates_factors_mean_grid_and_jvp_proof() -> None:
    stage3 = _binding_factor("binding.layer3", causal_order=3)
    stage4 = _binding_factor("binding.layer4", causal_order=4)
    positions = torch.arange(4)
    mask = torch.ones(4, dtype=torch.bool)
    baseline = stage3.output_mean.view(1, 1, -1).expand(1, 4, -1)
    edge_fit = estimate_causal_edge_jvp(
        lambda value: value,
        baseline_source=baseline,
        logical_positions=positions,
        valid_mask=mask,
        source_decoder=stage3.prolongation,
        target_encoder=stage4.restriction.T,
        max_lag=0,
        probe_count=8,
        probe_seed=9,
        ridge=0.0,
    )

    plan = bind_causal_modal_pair_plan(
        stage3,
        stage4,
        edge_fit,
        baseline_source=baseline,
        logical_positions=positions,
        valid_mask=mask,
        x4_reference_torn_base_name=(
            "binding.layer4.mean_source.reference.torn_base"
        ),
    )

    plan.validate_integrity()
    assert plan.direct_jacobian_proof_sha256 == edge_fit.artifact_sha256
    assert plan.boundary_contract.stage3_source_name == (
        stage3.output_port.name
    )
    assert plan.boundary_contract.stage4_target_name == (
        stage4.input_ports[0].name
    )
    torch.testing.assert_close(plan.K, edge_fit.kernel)

    with pytest.raises(ValueError, match="expanded stage3 output mean"):
        bind_causal_modal_pair_plan(
            stage3,
            stage4,
            edge_fit,
            baseline_source=baseline + 1.0,
            logical_positions=positions,
            valid_mask=mask,
            x4_reference_torn_base_name=(
                "binding.layer4.mean_source.reference.torn_base"
            ),
        )
    with pytest.raises(ValueError, match="logical_positions do not match"):
        bind_causal_modal_pair_plan(
            stage3,
            stage4,
            edge_fit,
            baseline_source=baseline,
            logical_positions=torch.tensor([0, 1, 2, 4]),
            valid_mask=mask,
            x4_reference_torn_base_name=(
                "binding.layer4.mean_source.reference.torn_base"
            ),
        )


def test_factorized_materialized_and_prepared_dense_paths_match() -> None:
    plan = _plan()
    x3, x4_reference = _inputs()
    positions, mask = _contiguous_grid()
    expected_y3, expected_y4 = _manual_factorized(
        plan,
        x3,
        x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )

    factor_y3, factor_y4 = plan.execute_factorized(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    dense = plan.materialize_dense()
    dense.validate_integrity()
    dense_y3, dense_y4 = dense.execute(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    prepared_y3, prepared_y4 = plan.prepare(
        device="cpu",
        dtype=FLOAT64,
    ).execute_dense(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )

    for actual, expected in (
        (factor_y3, expected_y3),
        (factor_y4, expected_y4),
        (dense_y3, expected_y3),
        (dense_y4, expected_y4),
        (prepared_y3, expected_y3),
        (prepared_y4, expected_y4),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert dense.source_plan_sha256 == plan.artifact_sha256


def test_logical_gaps_and_invalid_rows_never_fall_back_to_tensor_offsets() -> None:
    plan = _plan()
    x3, x4_reference = _inputs()
    positions = torch.tensor([0, 2, 3], dtype=torch.int64)
    mask = torch.tensor([True, True, True])
    expected = _manual_factorized(
        plan,
        x3,
        x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    actual = plan.execute_factorized(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)

    contiguous_expected = _manual_factorized(
        plan,
        x3,
        x4_reference,
        logical_positions=torch.tensor([0, 1, 2]),
        valid_mask=mask,
    )
    assert not torch.allclose(actual[1], contiguous_expected[1])

    masked = torch.tensor([True, False, True])
    masked_expected = _manual_factorized(
        plan,
        x3,
        x4_reference,
        logical_positions=positions,
        valid_mask=masked,
    )
    masked_actual = plan.execute_dense(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=masked,
    )
    for actual_value, expected_value in zip(
        masked_actual,
        masked_expected,
        strict=True,
    ):
        torch.testing.assert_close(actual_value, expected_value)


def test_mean_source_reference_and_nonzero_edge_are_applied_once() -> None:
    x3, x4_reference = _inputs()
    positions, mask = _contiguous_grid()
    zero_plan = _plan(zero_edge=True)
    _, zero_y4 = zero_plan.execute_factorized(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    base_m4 = (x4_reference - zero_plan.x4_mean) @ zero_plan.R4.T
    expected_base_y4 = base_m4 @ zero_plan.P4.T + zero_plan.y4_mean
    torch.testing.assert_close(zero_y4, expected_base_y4)

    mean_x3 = zero_plan.x3_mean.view(1, 1, -1).expand_as(x3)
    _, mean_source_y4 = _plan().execute_factorized(
        mean_x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    torch.testing.assert_close(mean_source_y4, expected_base_y4)

    plan = _plan()
    _, interacting_y4 = plan.execute_factorized(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    _, expected_y4 = _manual_factorized(
        plan,
        x3,
        x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    expected_edge_y4 = expected_y4 - expected_base_y4
    actual_edge_y4 = interacting_y4 - expected_base_y4
    torch.testing.assert_close(actual_edge_y4, expected_edge_y4)
    assert not torch.allclose(actual_edge_y4, 2.0 * expected_edge_y4)


def test_accounting_uses_the_explicit_logical_grid_and_counts_staged_state() -> None:
    accounting = _plan().accounting()
    positions, mask = _contiguous_grid()

    assert accounting.mean_stored_scalar_count == 11
    assert accounting.factorized_linear_stored_scalar_count == 21
    assert accounting.factorized_stored_scalar_count == 32
    assert accounting.dense_linear_stored_scalar_count == 22
    assert accounting.dense_stored_scalar_count == 33
    assert accounting.prepared_unique_stored_scalar_count == 54
    assert accounting.factorized_storage_bytes == 256
    assert accounting.dense_storage_bytes == 264
    assert accounting.prepared_unique_storage_bytes == 432

    assert accounting.factorized_local_macs_per_row == 17
    assert accounting.factorized_edge_macs_per_causal_pair == 2
    assert accounting.dense_local_macs_per_row == 14
    assert accounting.dense_edge_macs_per_causal_pair == 4
    assert accounting.causal_pair_count(
        (2, 3),
        logical_positions=positions,
        valid_mask=mask,
    ) == 10
    assert accounting.factorized_macs(
        (2, 3),
        logical_positions=positions,
        valid_mask=mask,
    ) == 6 * 17 + 10 * 2
    assert accounting.dense_macs(
        (2, 3),
        logical_positions=positions,
        valid_mask=mask,
    ) == 6 * 14 + 10 * 4

    gap_positions = torch.tensor([0, 2, 3], dtype=torch.int64)
    assert accounting.causal_pair_count(
        (2, 3),
        logical_positions=gap_positions,
        valid_mask=mask,
    ) == 8
    assert accounting.causal_pair_count(
        (2, 3),
        logical_positions=gap_positions,
        valid_mask=torch.tensor([True, False, True]),
    ) == 4
    assert accounting.factorized_carry_scalar_count((2, 3)) == 6
    assert accounting.dense_carry_scalar_count((2, 3)) == 12
    assert accounting.staged_carry_storage_bytes(
        (2, 3),
        position_bytes=8,
    ) == 102


def test_prepared_session_owns_then_clears_modal_and_grid_state() -> None:
    plan = _plan()
    runtime = PreparedCausalModalPair.from_plan(
        plan,
        device="cpu",
        dtype=FLOAT64,
    )
    x3, x4_reference = _inputs()
    positions, mask = _contiguous_grid()
    expected_y3, expected_y4 = _manual_factorized(
        plan,
        x3,
        x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    session = runtime.new_session()

    y3 = session.stage3(
        x3,
        logical_positions=positions,
        valid_mask=mask,
    )
    assert session.state == "stage3_complete"
    assert session.has_pending_stage3 is True
    assert session.pending_modal_carry_scalar_count == 6
    assert session.pending_logical_position_value_count == 6
    assert session.pending_valid_mask_value_count == 6
    assert session.pending_carry_storage_bytes == 102
    torch.testing.assert_close(y3, expected_y3)

    fresh_positions, fresh_mask = _contiguous_grid()
    positions.fill_(99)
    mask.zero_()
    y4 = session.stage4_from_mean_source_reference_torn_base(
        x4_reference,
        logical_positions=fresh_positions,
        valid_mask=fresh_mask,
    )
    assert session.state == "complete"
    assert session.has_pending_stage3 is False
    assert session.pending_carry_storage_bytes == 0
    torch.testing.assert_close(y4, expected_y4)

    with pytest.raises(RuntimeError, match="fresh or explicitly reset"):
        session.stage3(
            x3,
            logical_positions=fresh_positions,
            valid_mask=fresh_mask,
        )
    assert session.state == "failed"
    session.reset()
    assert session.state == "ready"
    session.close()
    assert session.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.reset()


def test_session_failures_clear_all_staged_state() -> None:
    runtime = _plan().prepare(device="cpu", dtype=FLOAT64)
    x3, x4_reference = _inputs()
    positions, mask = _contiguous_grid()

    session = runtime.new_session()
    with pytest.raises(RuntimeError, match="preceding stage3"):
        session.stage4_from_mean_source_reference_torn_base(
            x4_reference,
            logical_positions=positions,
            valid_mask=mask,
        )
    assert session.state == "failed"
    assert session.pending_carry_storage_bytes == 0

    session.reset()
    session.stage3(x3, logical_positions=positions, valid_mask=mask)
    with pytest.raises(ValueError, match="must equal"):
        session.stage4_from_mean_source_reference_torn_base(
            x4_reference,
            logical_positions=torch.tensor([0, 1, 3]),
            valid_mask=mask,
        )
    assert session.state == "failed"
    assert session.pending_carry_storage_bytes == 0

    session.reset()
    session.stage3(x3, logical_positions=positions, valid_mask=mask)
    with pytest.raises(ValueError, match="share stage3's leading shape"):
        session.stage4_from_mean_source_reference_torn_base(
            x4_reference[:, :2],
            logical_positions=torch.tensor([0, 1]),
            valid_mask=torch.ones(2, dtype=torch.bool),
        )
    assert session.pending_carry_storage_bytes == 0

    session.reset()
    with pytest.raises(ValueError, match="wrong width"):
        session.stage3(
            x3[..., :1],
            logical_positions=positions,
            valid_mask=mask,
        )
    assert session.state == "failed"

    reentrant = runtime.new_session()
    reentrant._in_call = True
    with pytest.raises(RuntimeError, match="not reentrant"):
        reentrant.stage3(
            x3,
            logical_positions=positions,
            valid_mask=mask,
        )
    reentrant._in_call = False
    reentrant.close()


@pytest.mark.parametrize(
    ("positions", "mask", "message"),
    [
        (
            torch.tensor([0.0, 1.0, 2.0]),
            torch.ones(3, dtype=torch.bool),
            "int32 or torch.int64",
        ),
        (
            torch.tensor([0, 2, 1]),
            torch.ones(3, dtype=torch.bool),
            "strictly increasing",
        ),
        (
            torch.tensor([0, -1, 2]),
            torch.ones(3, dtype=torch.bool),
            "nonnegative",
        ),
        (
            torch.tensor([0, 1, 2]),
            torch.ones(2, dtype=torch.bool),
            "must have shape",
        ),
    ],
)
def test_logical_grid_validation_fails_closed(
    positions: Tensor,
    mask: Tensor,
    message: str,
) -> None:
    x3, x4_reference = _inputs()
    with pytest.raises((TypeError, ValueError), match=message):
        _plan().execute_factorized(
            x3,
            x4_mean_source_reference_torn_base=x4_reference,
            logical_positions=positions,
            valid_mask=mask,
        )


def test_prepared_hot_paths_do_not_revalidate_or_convert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    runtime = plan.prepare(device="cpu", dtype=FLOAT64)
    x3, x4_reference = _inputs()
    positions, mask = _contiguous_grid()

    def unexpected_validation(self: CausalModalPairPlan) -> None:
        raise AssertionError("prepared hot path revalidated its source plan")

    def unexpected_conversion(
        self: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        raise AssertionError("prepared hot path called Tensor.to")

    monkeypatch.setattr(
        CausalModalPairPlan,
        "validate_integrity",
        unexpected_validation,
    )
    monkeypatch.setattr(torch.Tensor, "to", unexpected_conversion)

    factorized = runtime.execute_factorized(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    dense = runtime.execute_dense(
        x3,
        x4_mean_source_reference_torn_base=x4_reference,
        logical_positions=positions,
        valid_mask=mask,
    )
    for factorized_value, dense_value in zip(
        factorized,
        dense,
        strict=True,
    ):
        torch.testing.assert_close(factorized_value, dense_value)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_preparation_rejects_non_runtime_floating_dtypes(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="supported floating"):
        _plan().prepare(device="cpu", dtype=dtype)
