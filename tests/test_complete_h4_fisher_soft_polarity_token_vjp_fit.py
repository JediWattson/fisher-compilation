from __future__ import annotations

import json

import pytest
import torch

from fisher_graph.complete_h4_fisher_soft_polarity_token_vjp_fit import (
    SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER,
    SoftPolarityTokenVJPFitArguments,
    SoftPolarityTokenVJPPromptRecord,
    aggregate_soft_polarity_token_vjp_records,
    build_soft_polarity_token_vjp_natural_direction,
    contract_soft_polarity_token_h4_vjps,
    fit_soft_polarity_token_vjp_step,
)
from fisher_graph.complete_h4_fisher_soft_polarity_token_vjp_protocol import (
    build_soft_polarity_token_vjp_scalar_fit_output,
)


def _record(
    family_id: str,
    example_id: str,
    q: torch.Tensor,
    kl: torch.Tensor,
    *,
    reference_b: float = 0.125,
    reference_a: float = -0.25,
) -> SoftPolarityTokenVJPPromptRecord:
    return SoftPolarityTokenVJPPromptRecord(
        feature_id="signed_log_c2",
        family_id=family_id,
        example_id=example_id,
        reference_b=reference_b,
        reference_a=reference_a,
        derivative_convention="reverse_token_vjp_at_realized_post_cast_h4",
        derivative_artifact_sha256s=("a" * 64, "b" * 64),
        token_teacher_kl=kl.to(dtype=torch.float64),
        token_parameter_gradients=q.to(dtype=torch.float64),
    )


def test_exact_causal_contraction_matches_direct_toy_autograd() -> None:
    tangents = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [0.0, 0.0]]],
            [[[-2.0, 1.0], [1.5, 0.25], [2.0, -3.0], [0.0, 0.0]]],
        ],
        dtype=torch.float64,
    )
    theta = torch.tensor((0.4, -0.7), dtype=torch.float64, requires_grad=True)
    h4 = torch.einsum("p,pbsw->bsw", theta, tangents)
    coefficients = (
        torch.tensor(
            [[[2.0, -1.0], [0.5, 3.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[[-1.0, 0.25], [2.0, -0.5], [1.5, 4.0], [0.0, 0.0]]],
            dtype=torch.float64,
        ),
    )
    losses = tuple((h4 * coefficient).sum() for coefficient in coefficients)
    token_h4_gradients = torch.stack(
        [
            torch.autograd.grad(loss, h4, retain_graph=True)[0]
            for loss in losses
        ]
    )
    expected = torch.stack(
        [
            torch.autograd.grad(loss, theta, retain_graph=True)[0]
            for loss in losses
        ]
    )

    actual = contract_soft_polarity_token_h4_vjps(
        token_h4_gradients=token_h4_gradients,
        local_h4_tangents=tangents,
        canonical_support_mask=torch.tensor([[True, True, True, False]]),
        supervised_indices=torch.tensor(((0, 1), (0, 2)), dtype=torch.int64),
    )
    assert SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER == (
        "field_bias",
        "field_slope",
    )
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert torch.allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_contraction_rejects_non_support_and_future_leakage() -> None:
    gradients = torch.zeros(1, 1, 3, 1, dtype=torch.float64)
    tangents = torch.zeros(2, 1, 3, 1, dtype=torch.float64)
    tangents[0, 0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="outside canonical support"):
        contract_soft_polarity_token_h4_vjps(
            token_h4_gradients=gradients,
            local_h4_tangents=tangents,
            canonical_support_mask=torch.tensor([[True, False, False]]),
            supervised_indices=torch.tensor(((0, 0),), dtype=torch.int64),
        )

    tangents.zero_()
    tangents[0, 0, :, 0] = 1.0
    gradients[0, 0, 2, 0] = 1.0
    with pytest.raises(ValueError, match="future or cross-batch leakage"):
        contract_soft_polarity_token_h4_vjps(
            token_h4_gradients=gradients,
            local_h4_tangents=tangents,
            canonical_support_mask=torch.tensor([[True, True, True]]),
            supervised_indices=torch.tensor(((0, 1),), dtype=torch.int64),
        )


def test_contraction_rejects_noncanonical_supervised_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        contract_soft_polarity_token_h4_vjps(
            token_h4_gradients=torch.zeros(2, 1, 2, 1, dtype=torch.float64),
            local_h4_tangents=torch.zeros(2, 1, 2, 1, dtype=torch.float64),
            canonical_support_mask=torch.ones(1, 2, dtype=torch.bool),
            supervised_indices=torch.tensor(((0, 1), (0, 0)), dtype=torch.int64),
        )


def test_prompt_record_is_defensive_tamper_evident_and_metadata_only() -> None:
    source_kl = torch.tensor((1.0, 2.0), dtype=torch.float64)
    source_q = torch.tensor(((1.0, 0.0), (0.0, 2.0)), dtype=torch.float64)
    record = _record("family-a", "example-a", source_q, source_kl)
    source_kl.add_(10.0)
    source_q.add_(10.0)
    assert torch.equal(
        record.token_teacher_kl_tensor(),
        torch.tensor((1.0, 2.0), dtype=torch.float64),
    )
    metadata = record.metadata()
    assert metadata["raw_token_teacher_kl_serialized"] is False
    assert metadata["raw_token_parameter_gradients_serialized"] is False
    assert metadata["raw_tensors_serialized"] is False
    assert "tensor(" not in json.dumps(metadata)

    record.token_teacher_kl[0] = 99.0
    with pytest.raises(RuntimeError, match="prompt record drifted"):
        record.validate_integrity()


def test_aggregation_weights_families_prompts_and_tokens_at_their_own_levels() -> None:
    records = (
        _record(
            "family-a",
            "a-one-token",
            torch.tensor(((1.0, 0.0),)),
            torch.tensor((2.0,)),
        ),
        _record(
            "family-a",
            "a-five-tokens",
            torch.tensor(((3.0, 0.0),)).repeat(5, 1),
            torch.full((5,), 4.0),
        ),
        _record(
            "family-b",
            "b-three-tokens",
            torch.tensor(((0.0, 2.0),)).repeat(3, 1),
            torch.full((3,), 5.0),
        ),
    )
    aggregate = aggregate_soft_polarity_token_vjp_records(
        reversed(records), held_family_id="held"
    )
    assert torch.equal(
        aggregate.mean_parameter_gradient,
        torch.tensor((1.0, 1.0), dtype=torch.float64),
    )
    assert torch.equal(
        aggregate.residual_gradient_c,
        torch.tensor((3.5, 5.0), dtype=torch.float64),
    )
    assert torch.equal(
        aggregate.gradient_gram,
        torch.diag(torch.tensor((2.5, 2.0), dtype=torch.float64)),
    )
    assert aggregate.mean_token_teacher_kl == pytest.approx(4.0)
    assert aggregate.family_prompt_counts == (
        ("family-a", 2),
        ("family-b", 1),
    )
    assert aggregate.family_token_counts == (
        ("family-a", 6),
        ("family-b", 3),
    )
    assert aggregate.prompt_count == 3
    assert aggregate.token_count == 9
    assert aggregate.metadata()["family_weighting"] == "families_equal"


def test_aggregation_rejects_held_family_and_tampered_evidence() -> None:
    record = _record(
        "held",
        "example",
        torch.tensor(((1.0, 0.0),)),
        torch.tensor((1.0,)),
    )
    with pytest.raises(ValueError, match="Held-family|held-family"):
        aggregate_soft_polarity_token_vjp_records(
            (record,), held_family_id="held"
        )

    clean = _record(
        "train",
        "example-clean",
        torch.tensor(((1.0, 0.0),)),
        torch.tensor((1.0,)),
    )
    clean.token_parameter_gradients[0, 0] = 9.0
    with pytest.raises(RuntimeError, match="prompt record drifted"):
        aggregate_soft_polarity_token_vjp_records(
            (clean,), held_family_id="held"
        )


def test_natural_direction_matches_trace_ridge_formula_and_protocol_contract() -> None:
    aggregate = aggregate_soft_polarity_token_vjp_records(
        (
            _record(
                "train",
                "natural-direction",
                torch.tensor(((2.0, 0.0), (0.0, 1.0))),
                torch.ones(2),
            ),
        ),
        held_family_id="held",
    )
    result = build_soft_polarity_token_vjp_natural_direction(
        aggregate, ridge_multiplier=0.1
    )
    expected_trace = 2.5
    expected_tau = 1.25
    expected_damping = 0.125
    expected_raw = -torch.linalg.solve(
        aggregate.gradient_gram
        + expected_damping * torch.eye(2, dtype=torch.float64),
        aggregate.mean_parameter_gradient,
    )
    expected_direction = expected_raw / expected_raw.abs().max()
    assert result.no_op is False
    assert result.gradient_gram_trace.hex() == expected_trace.hex()
    assert result.tau.hex() == expected_tau.hex()
    assert result.damping.hex() == expected_damping.hex()
    assert torch.equal(result.raw_direction, expected_raw)
    assert torch.equal(result.direction, expected_direction)
    assert result.direction_linf.hex() == 1.0.hex()
    assert result.predicted_derivative < 0.0
    assert result.metadata()["method"] == (
        "mean_kl_natural_opg_trace_scaled_ridge_linf_direction"
    )

    scalar = build_soft_polarity_token_vjp_scalar_fit_output(
        direction_metadata=result.metadata(),
        aggregate_metadata=aggregate.metadata(),
        primary_secant_receipt_sha256="c" * 64,
        audit_secant_receipt_sha256="d" * 64,
        secant_stability={
            "cosine_by_parameter": (1.0, 1.0),
            "audit_to_primary_norm_ratio_by_parameter": (1.0, 1.0),
            "passed": True,
        },
    )
    assert scalar["direction_linf"] == 1.0
    assert scalar["fit_artifact_sha256"] == result.artifact_sha256


def test_natural_direction_uses_tau_floor_and_binds_tensor_tamper() -> None:
    tiny = 2.0**-30
    aggregate = aggregate_soft_polarity_token_vjp_records(
        (
            _record(
                "train",
                "tau-floor",
                torch.tensor(((tiny, 0.0), (0.0, tiny))),
                torch.ones(2),
            ),
        ),
        held_family_id="held",
    )
    result = build_soft_polarity_token_vjp_natural_direction(
        aggregate, ridge_multiplier=1.0
    )
    assert result.tau.hex() == (2.0**-24).hex()
    assert result.damping.hex() == result.tau.hex()
    result.direction[0] = 0.0
    with pytest.raises(RuntimeError, match="natural direction receipt drifted"):
        result.validate_integrity()


def test_natural_direction_degenerate_cases_are_explicit_no_ops() -> None:
    zero_gradient = aggregate_soft_polarity_token_vjp_records(
        (
            _record(
                "train",
                "zero-gradient",
                torch.tensor(((1.0, 0.0), (-1.0, 0.0))),
                torch.ones(2),
            ),
        ),
        held_family_id="held",
    )
    zero = build_soft_polarity_token_vjp_natural_direction(
        zero_gradient, ridge_multiplier=1.0
    )
    assert zero.no_op is True
    assert zero.no_op_reason == "zero_mean_kl_gradient"
    assert zero.direction_linf == 0.0

    singular = aggregate_soft_polarity_token_vjp_records(
        (
            _record(
                "train",
                "singular",
                torch.tensor(((1.0, 0.0), (2.0, 0.0))),
                torch.ones(2),
            ),
        ),
        held_family_id="held",
    )
    undamped = build_soft_polarity_token_vjp_natural_direction(
        singular, ridge_multiplier=0.0
    )
    assert undamped.no_op is True
    assert undamped.no_op_reason == "singular_damped_system"
    damped = build_soft_polarity_token_vjp_natural_direction(
        singular, ridge_multiplier=1.0
    )
    assert damped.no_op is False
    assert damped.direction_linf == 1.0
    assert damped.predicted_derivative < 0.0


def test_full_rank_fit_is_damped_trust_bounded_and_descending() -> None:
    record = _record(
        "train",
        "full-rank",
        torch.eye(2, dtype=torch.float64),
        torch.tensor((2.0, 4.0), dtype=torch.float64),
    )
    aggregate = aggregate_soft_polarity_token_vjp_records(
        (record,), held_family_id="held"
    )
    arguments = SoftPolarityTokenVJPFitArguments(
        damping=0.5,
        trust_l2_bound=0.25,
        solver_kind="mean_kl_natural_opg",
    )
    fit = fit_soft_polarity_token_vjp_step(aggregate, arguments=arguments)
    assert fit.no_op is False
    assert fit.no_op_reason is None
    assert fit.applied_step_l2 == pytest.approx(0.25)
    assert fit.trust_scale < 1.0
    assert fit.predicted_derivative < 0.0
    assert fit.proposed_b == pytest.approx(
        aggregate.reference_b + float(fit.applied_step[0])
    )
    assert fit.proposed_a == pytest.approx(
        aggregate.reference_a + float(fit.applied_step[1])
    )
    assert fit.metadata()["method"] == (
        "one_step_damped_mean_KL_natural_gradient_with_OPG"
    )
    assert fit.solver_kind == "mean_kl_natural_opg"
    assert torch.equal(fit.selected_rhs, aggregate.mean_parameter_gradient)


def test_solver_kind_selects_and_authenticates_natural_or_residual_rhs() -> None:
    aggregate = aggregate_soft_polarity_token_vjp_records(
        (
            _record(
                "train",
                "solver-selection",
                torch.eye(2, dtype=torch.float64),
                torch.tensor((2.0, 4.0), dtype=torch.float64),
            ),
        ),
        held_family_id="held",
    )
    natural = fit_soft_polarity_token_vjp_step(
        aggregate,
        arguments=SoftPolarityTokenVJPFitArguments(
            damping=0.5,
            trust_l2_bound=10.0,
            solver_kind="mean_kl_natural_opg",
        ),
    )
    residual = fit_soft_polarity_token_vjp_step(
        aggregate,
        arguments=SoftPolarityTokenVJPFitArguments(
            damping=0.5,
            trust_l2_bound=10.0,
            solver_kind="squared_kl_residual_gn",
        ),
    )
    assert torch.equal(
        natural.selected_rhs, torch.tensor((0.5, 0.5), dtype=torch.float64)
    )
    assert torch.equal(
        residual.selected_rhs, torch.tensor((1.0, 2.0), dtype=torch.float64)
    )
    assert torch.equal(
        natural.raw_step, torch.tensor((-0.5, -0.5), dtype=torch.float64)
    )
    assert torch.equal(
        residual.raw_step, torch.tensor((-1.0, -2.0), dtype=torch.float64)
    )
    assert natural.metadata()["selected_rhs_sha256"] != (
        residual.metadata()["selected_rhs_sha256"]
    )
    assert residual.metadata()["method"] == (
        "one_step_damped_squared_KL_residual_Gauss_Newton_with_OPG"
    )
    with pytest.raises(ValueError, match="solver_kind"):
        SoftPolarityTokenVJPFitArguments(
            damping=0.5,
            trust_l2_bound=1.0,
            solver_kind="undeclared",
        )


def _assert_exact_no_op(
    q: torch.Tensor,
    kl: torch.Tensor,
    *,
    solver_kind: str,
    damping: float,
    reason: str,
) -> None:
    aggregate = aggregate_soft_polarity_token_vjp_records(
        (_record("train", f"example-{reason}", q, kl),),
        held_family_id="held",
    )
    fit = fit_soft_polarity_token_vjp_step(
        aggregate,
        arguments=SoftPolarityTokenVJPFitArguments(
            damping=damping,
            trust_l2_bound=0.5,
            solver_kind=solver_kind,
        ),
    )
    assert fit.no_op is True
    assert fit.no_op_reason == reason
    assert torch.equal(fit.applied_step, torch.zeros(2, dtype=torch.float64))
    assert fit.applied_step_l2 == 0.0
    assert fit.predicted_derivative == 0.0
    assert fit.proposed_b.hex() == aggregate.reference_b.hex()
    assert fit.proposed_a.hex() == aggregate.reference_a.hex()


def test_zero_natural_and_residual_gradients_are_exact_no_ops() -> None:
    _assert_exact_no_op(
        torch.zeros(2, 2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        solver_kind="mean_kl_natural_opg",
        damping=1.0,
        reason="zero_mean_kl_gradient",
    )
    _assert_exact_no_op(
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        solver_kind="squared_kl_residual_gn",
        damping=1.0,
        reason="zero_residual_gradient",
    )


def test_positive_damping_solves_rank_deficient_supported_evidence() -> None:
    q = torch.tensor(((1.0, 0.0), (2.0, 0.0)), dtype=torch.float64)
    aggregate = aggregate_soft_polarity_token_vjp_records(
        (_record("train", "rank-one", q, torch.ones(2)),),
        held_family_id="held",
    )
    fit = fit_soft_polarity_token_vjp_step(
        aggregate,
        arguments=SoftPolarityTokenVJPFitArguments(
            damping=1.0,
            trust_l2_bound=0.5,
            solver_kind="mean_kl_natural_opg",
        ),
    )
    assert fit.no_op is False
    assert fit.predicted_derivative < 0.0
    assert fit.applied_step[0] < 0.0
    assert fit.applied_step[1] == 0.0


def test_undamped_singular_system_is_an_exact_no_op() -> None:
    _assert_exact_no_op(
        torch.tensor(((1.0, 0.0), (2.0, 0.0)), dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        solver_kind="mean_kl_natural_opg",
        damping=0.0,
        reason="singular_damped_system",
    )
