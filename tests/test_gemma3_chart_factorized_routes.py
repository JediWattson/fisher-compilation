from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.gemma3_attention_generator import (
    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID,
    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT,
    AttentionRouteProbabilitySimplexAudit,
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    Gemma3AttentionRouteFactorSnapshot,
    attention_route_probability_simplex_forward_error_bound,
    compile_exact_attention_generator,
    derive_attention_route_allowed_mask,
    validate_attention_route_probability_simplex,
)
from fisher_graph.gemma3_chart_mode_surfaces import (
    FACTORIZED_ATTENTION_ROUTE_KERNEL_ID,
    FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE,
    FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE,
    AttentionRouteStreamContext,
    NativeRoPESpec,
    build_attention_postnorm_bridge_receipt,
    build_attention_route_chart_surface,
    build_attention_target_carrier_signature,
    build_factorized_attention_route_aggregates,
    build_mutation_carrier_chart_surface,
    bind_carrier_signature_to_chart_mode_spec,
    factorized_attention_route_kernel_receipt,
)
from fisher_graph.chart_energy_neighbor_graph import (
    CAUSAL_SEQUENCE_NATIVE_ABI,
    DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    ChartModeSpec,
    NativeChartABI,
    canonical_carrier_comparison_abi,
)


HIDDEN = 640
HEADS = 2
HEAD_DIMENSION = 2


def _sha(index: int) -> str:
    return f"{index:064x}"


def _normalization(width: int) -> NormalizationSpec:
    return NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1.0e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )


def _generator(
    *,
    attention_kind: str = "global_causal",
    window_size: int | None = None,
):
    config = Gemma3AttentionGeneratorConfig(
        invocation_id="mutation.1",
        residual_width=HIDDEN,
        attention=AttentionSpec(
            kind=attention_kind,
            query_heads=HEADS,
            key_value_heads=1,
            head_dimension=HEAD_DIMENSION,
            query_scale=HEAD_DIMENSION**-0.5,
            qk_norm=True,
            window_size=window_size,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=HEAD_DIMENSION,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        input_norm=_normalization(HIDDEN),
        output_norm=_normalization(HIDDEN),
        qk_norm=_normalization(HEAD_DIMENSION),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=None,
    )
    random = torch.Generator().manual_seed(8_140)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(
            rows,
            columns,
            dtype=torch.float64,
            generator=random,
        ) / 25.0

    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=torch.zeros(HIDDEN, dtype=torch.float64),
            q_weight=matrix(HEADS * HEAD_DIMENSION, HIDDEN),
            k_weight=matrix(HEAD_DIMENSION, HIDDEN),
            v_weight=matrix(HEAD_DIMENSION, HIDDEN),
            o_weight=matrix(HIDDEN, HEADS * HEAD_DIMENSION),
            output_norm_weight=torch.zeros(HIDDEN, dtype=torch.float64),
            q_norm_weight=torch.zeros(HEAD_DIMENSION, dtype=torch.float64),
            k_norm_weight=torch.zeros(HEAD_DIMENSION, dtype=torch.float64),
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _sequence(length: int = 4) -> SequenceContext:
    valid = torch.ones(1, length, dtype=torch.bool)
    positions = torch.arange(length, dtype=torch.int64)[None, :]
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )


def _factorized_fixture(*, zero_operator_gradient: bool = False):
    generator = _generator()
    sequence = _sequence()
    hidden = torch.randn(
        1,
        sequence.query_length,
        HIDDEN,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(8_141),
    )
    execution = generator(
        hidden,
        sequence,
        capture_route_factors=True,
    )
    snapshot = execution.route_factor_snapshot
    assert isinstance(snapshot, Gemma3AttentionRouteFactorSnapshot)
    query_valid = sequence.query_valid_mask.clone()
    query_valid[:, -1] = False
    native_context = AttentionRouteStreamContext(
        example_ids=("example.factorized.1",),
        context_fingerprints=(_sha(1),),
        query_valid_mask=query_valid,
        key_valid_mask=sequence.key_valid_mask,
        query_logical_positions=sequence.logical_positions,
        key_logical_positions=sequence.key_logical_positions,
        query_head_count=HEADS,
        window_size=None,
        rope=NativeRoPESpec(
            theta=10_000.0,
            rotary_dimension=HEAD_DIMENSION,
            scaling_kind="none",
        ),
    )
    random = torch.Generator().manual_seed(8_142)
    operator_gradient = torch.randn(
        execution.operator_output.shape,
        dtype=torch.float64,
        generator=random,
    )
    if zero_operator_gradient:
        operator_gradient.zero_()
    postnorm_gradient = torch.randn(
        execution.residual_delta.shape,
        dtype=torch.float64,
        generator=random,
    )
    bridge = build_attention_postnorm_bridge_receipt(
        operator_output=execution.operator_output,
        operator_output_score_gradient=operator_gradient,
        postnorm_delta=execution.residual_delta,
        postnorm_state_sha256=_sha(2),
        postnorm_config_sha256=_sha(3),
        execution_fingerprint_sha256=_sha(4),
    )
    carrier = build_attention_target_carrier_signature(
        mode_id="mode.mutation.1",
        block_ordinal=0,
        native_abi_sha256=_sha(5),
        comparison_abi_sha256=_sha(6),
        source_state_sha256=_sha(7),
        execution_fingerprint_sha256=_sha(4),
        score_objective_receipt_sha256=_sha(8),
        attention_context_binding_sha256=native_context.sha256,
        postnorm_bridge=bridge,
        example_ids=native_context.example_ids,
        query_valid_mask=query_valid,
        query_logical_positions=sequence.logical_positions,
        target_carrier_values=execution.residual_delta,
        target_carrier_score_gradients=postnorm_gradient,
    )
    return (
        snapshot,
        execution,
        operator_gradient,
        postnorm_gradient,
        native_context,
        bridge,
        carrier,
    )


def _exhaustive_surface(fixture):
    (
        snapshot,
        execution,
        operator_gradient,
        postnorm_gradient,
        context,
        _,
        _,
    ) = fixture
    projected_values = torch.einsum(
        "bhkd,hdo->bhko",
        snapshot.repeated_values.to(torch.float64),
        snapshot.output_projection_weight.to(torch.float64)
        .reshape(HIDDEN, HEADS, HEAD_DIMENSION)
        .permute(1, 2, 0),
    )
    route_values = (
        snapshot.probabilities.permute(0, 2, 1, 3)[..., None].to(
            torch.float64
        )
        * projected_values[:, None, :, :, :]
    )
    allowed = (
        context.query_valid_mask[:, :, None, None]
        & context.key_valid_mask[:, None, None, :]
        & (
            context.key_logical_positions[:, None, None, :]
            <= context.query_logical_positions[:, :, None, None]
        )
    ).expand(1, context.query_valid_mask.shape[1], HEADS, -1)
    fixture_operator = execution.operator_output.masked_fill(
        ~context.query_valid_mask.unsqueeze(-1),
        0,
    )
    fixture_operator_gradient = operator_gradient.masked_fill(
        ~context.query_valid_mask.unsqueeze(-1),
        0,
    )
    return build_attention_route_chart_surface(
        mode_id="mode.mutation.1",
        block_ordinal=0,
        native_abi_sha256=_sha(5),
        comparison_abi_sha256=_sha(6),
        source_state_sha256=_sha(7),
        execution_fingerprint_sha256=_sha(4),
        score_objective_receipt_sha256=_sha(8),
        example_ids=context.example_ids,
        context_fingerprints=context.context_fingerprints,
        query_valid_mask=context.query_valid_mask,
        key_valid_mask=context.key_valid_mask,
        query_logical_positions=context.query_logical_positions,
        key_logical_positions=context.key_logical_positions,
        native_route_mask=allowed,
        target_carrier_values=execution.residual_delta,
        target_carrier_score_gradients=postnorm_gradient,
        attention_operator_output=fixture_operator,
        attention_operator_output_score_gradients=fixture_operator_gradient,
        route_operator_output_values=route_values,
        postnorm_state_sha256=_sha(2),
        postnorm_config_sha256=_sha(3),
        rope=context.rope,
        window_size=None,
    )


def test_exact_attention_route_snapshot_is_detached_factorized_and_hashed() -> None:
    snapshot, execution, *_ = _factorized_fixture()
    assert snapshot.probabilities.device.type == "cpu"
    assert snapshot.repeated_values.device.type == "cpu"
    assert snapshot.output_projection_weight.device.type == "cpu"
    assert snapshot.probabilities.requires_grad is False
    assert snapshot.repeated_values.requires_grad is False
    assert snapshot.metadata()["full_route_vectors_materialized"] is False
    assert snapshot.metadata()["contains_token_ids"] is False
    assert snapshot.metadata()["output_projection_bias_present"] is False
    assert len(snapshot.sha256) == 64
    assert snapshot.probabilities.shape == (1, HEADS, 4, 4)
    assert snapshot.repeated_values.shape == (1, HEADS, 4, HEAD_DIMENSION)
    before = execution.operator_output.detach().clone()
    snapshot.probabilities.zero_()
    assert torch.equal(execution.operator_output, before)
    with pytest.raises(ValueError, match="bias-free O projection"):
        replace(snapshot, output_projection_bias_present=True)


def _native_float32_route_probabilities(
    *,
    length: int,
    scale: float,
    seed: int,
    window_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(length, dtype=torch.int64)[None, :]
    valid = torch.ones(1, length, dtype=torch.bool)
    allowed = derive_attention_route_allowed_mask(
        query_valid_mask=valid,
        key_valid_mask=valid,
        query_logical_positions=positions,
        key_logical_positions=positions,
        window_size=window_size,
    )
    scores = torch.randn(
        1,
        1,
        length,
        length,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float32,
    ) * scale
    scores = scores.masked_fill(
        ~allowed[:, None, :, :],
        torch.finfo(torch.float32).min,
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    probabilities = probabilities.masked_fill(~allowed[:, None, :, :], 0)
    return probabilities, allowed, valid, positions


def test_float32_softmax_simplex_bound_covers_production_support_sweep() -> None:
    maximum_observed_residual = 0.0
    lengths = (96, 511, 512, 513, 514, 576, 640, 768)
    scale_and_seed = ((0.0, 0), (0.1, 1), (1.0, 7), (100.0, 29))
    for length in lengths:
        for window_size in (None, 512):
            for scale, seed in scale_and_seed:
                probabilities, allowed, valid, _ = (
                    _native_float32_route_probabilities(
                        length=length,
                        scale=scale,
                        seed=seed,
                        window_size=window_size,
                    )
                )
                audit = validate_attention_route_probability_simplex(
                    probabilities,
                    allowed_mask=allowed[:, None, :, :],
                    valid_row_mask=valid[:, None, :],
                )
                assert audit.formula_id == (
                    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID
                )
                assert audit.support_maximum == min(
                    length,
                    window_size or length,
                )
                assert audit.maximum_absolute_mass_residual <= (
                    audit.maximum_forward_error_bound
                )
                assert audit.minimum_error_margin >= 0.0
                maximum_observed_residual = max(
                    maximum_observed_residual,
                    audit.maximum_absolute_mass_residual,
                )
    # The regression is meaningful: legitimate native rows exceed the old
    # effective 2e-7 fixed allclose gate.
    assert maximum_observed_residual > 2.0e-7
    assert ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT == 768
    assert (
        attention_route_probability_simplex_forward_error_bound(768)
        < 1.0e-4
    )


def test_production_batch_and_head_shape_rejects_old_float32_sum_gate() -> None:
    batch = 2
    heads = 4
    length = 96
    positions = torch.arange(length, dtype=torch.int64)[None, :].expand(
        batch,
        -1,
    )
    valid = torch.ones(batch, length, dtype=torch.bool)
    allowed = derive_attention_route_allowed_mask(
        query_valid_mask=valid,
        key_valid_mask=valid,
        query_logical_positions=positions,
        key_logical_positions=positions,
        window_size=None,
    )
    scores = torch.randn(
        batch,
        heads,
        length,
        length,
        generator=torch.Generator().manual_seed(0),
        dtype=torch.float32,
    ).masked_fill(
        ~allowed[:, None, :, :],
        torch.finfo(torch.float32).min,
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    probabilities = probabilities.masked_fill(~allowed[:, None, :, :], 0)
    assert float((probabilities.sum(dim=-1) - 1.0).abs().max()) > 2.0e-7
    audit = validate_attention_route_probability_simplex(
        probabilities,
        allowed_mask=allowed[:, None, :, :].expand_as(probabilities),
        valid_row_mask=valid[:, None, :].expand(batch, heads, length),
    )
    assert audit.valid_row_count == batch * heads * length
    snapshot = Gemma3AttentionRouteFactorSnapshot(
        invocation_id="mutation.1",
        generator_kind="exact_full_rank",
        attention_kind="global_causal",
        config_sha256=_sha(101),
        artifact_sha256=_sha(102),
        probabilities=probabilities,
        repeated_values=torch.zeros(
            batch,
            heads,
            length,
            2,
            dtype=torch.float32,
        ),
        output_projection_weight=torch.zeros(
            heads * 2,
            heads * 2,
            dtype=torch.float32,
        ),
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        query_logical_positions=positions,
        key_logical_positions=positions.clone(),
        query_head_count=heads,
        key_value_head_count=2,
        head_dimension=2,
        window_size=None,
        rope_theta=10_000.0,
        rope_scaling_kind="none",
        rope_scaling_factor=None,
    )
    assert snapshot.probability_simplex_audit.sha256 == audit.sha256


def test_float32_softmax_simplex_bound_accepts_ties_and_extreme_logits() -> None:
    for scale in (0.0, 1.0e4):
        probabilities, allowed, valid, _ = _native_float32_route_probabilities(
            length=768,
            scale=scale,
            seed=113,
            window_size=None,
        )
        audit = validate_attention_route_probability_simplex(
            probabilities,
            allowed_mask=allowed[:, None, :, :],
            valid_row_mask=valid[:, None, :],
        )
        assert audit.valid_row_count == 768
        assert len(audit.sha256) == 64


def test_float32_softmax_simplex_rejects_mass_mask_dtype_and_support_tamper() -> None:
    length = 768
    probabilities, allowed, valid, _ = _native_float32_route_probabilities(
        length=length,
        scale=0.1,
        seed=1,
        window_size=None,
    )
    changed_mass = probabilities.clone()
    changed_mass[0, 0, -1, 0] += 1.0e-4
    with pytest.raises(ValueError, match="simplex error bound"):
        validate_attention_route_probability_simplex(
            changed_mass,
            allowed_mask=allowed[:, None, :, :],
            valid_row_mask=valid[:, None, :],
        )

    changed_mask = probabilities.clone()
    changed_mask[0, 0, 0, 1] = 1.0e-6
    with pytest.raises(ValueError, match="masked route probabilities"):
        validate_attention_route_probability_simplex(
            changed_mask,
            allowed_mask=allowed[:, None, :, :],
            valid_row_mask=valid[:, None, :],
        )
    with pytest.raises(ValueError, match="torch.float32"):
        validate_attention_route_probability_simplex(
            probabilities.to(torch.float64),
            allowed_mask=allowed[:, None, :, :],
            valid_row_mask=valid[:, None, :],
        )
    with pytest.raises(ValueError, match=r"\[1, 768\]"):
        attention_route_probability_simplex_forward_error_bound(769)


def test_simplex_validator_enforces_invalid_rows_support_and_unit_interval() -> None:
    probabilities = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    allowed = torch.tensor(
        [[[[True, False], [False, False]]]],
        dtype=torch.bool,
    )
    valid_rows = torch.tensor([[[True, False]]], dtype=torch.bool)
    audit = validate_attention_route_probability_simplex(
        probabilities,
        allowed_mask=allowed,
        valid_row_mask=valid_rows,
    )
    assert audit.valid_row_count == 1
    assert audit.support_minimum == audit.support_maximum == 1

    invalid_nonzero = probabilities.clone()
    invalid_nonzero[0, 0, 1, 0] = 1.0e-6
    with pytest.raises(ValueError, match="masked route probabilities"):
        validate_attention_route_probability_simplex(
            invalid_nonzero,
            allowed_mask=allowed,
            valid_row_mask=valid_rows,
        )
    zero_support = allowed.clone()
    zero_support[0, 0, 0, 0] = False
    zero_probabilities = torch.zeros_like(probabilities)
    with pytest.raises(ValueError, match="positive support"):
        validate_attention_route_probability_simplex(
            zero_probabilities,
            allowed_mask=zero_support,
            valid_row_mask=valid_rows,
        )
    for changed_value in (-1.0e-6, 1.0 + 1.0e-6):
        changed = probabilities.clone()
        changed[0, 0, 0, 0] = changed_value
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_attention_route_probability_simplex(
                changed,
                allowed_mask=allowed,
                valid_row_mask=valid_rows,
            )


def test_snapshot_binds_simplex_audit_and_consumer_rejects_mutation() -> None:
    snapshot, execution, operator_gradient, _, context, bridge, carrier = (
        _factorized_fixture()
    )
    audit = snapshot.probability_simplex_audit
    assert snapshot.metadata()["probability_simplex_audit_sha256"] == audit.sha256
    snapshot.probabilities[0, 0, -1, 0] += 1.0e-4
    with pytest.raises(ValueError, match="simplex error bound"):
        build_factorized_attention_route_aggregates(
            carrier=carrier,
            block_ordinal=0,
            context=context,
            postnorm_bridge=bridge,
            route_factors=snapshot,
            attention_operator_output=execution.operator_output,
            attention_operator_output_score_gradients=operator_gradient,
        )


def test_snapshot_rejects_non_cpu_probability_and_kind_window_forgery() -> None:
    snapshot, *_ = _factorized_fixture()
    with pytest.raises(ValueError, match="CPU torch.float32"):
        replace(
            snapshot,
            probabilities=torch.empty_like(
                snapshot.probabilities,
                device="meta",
            ),
        )
    with pytest.raises(ValueError, match="global attention"):
        replace(snapshot, window_size=2)


def test_simplex_audit_and_factorized_receipt_reject_coordinated_rehash() -> None:
    fixture = _factorized_fixture()
    snapshot, execution, operator_gradient, _, context, bridge, carrier = fixture
    (receipt,) = build_factorized_attention_route_aggregates(
        carrier=carrier,
        block_ordinal=0,
        context=context,
        postnorm_bridge=bridge,
        route_factors=snapshot,
        attention_operator_output=execution.operator_output,
        attention_operator_output_score_gradients=operator_gradient,
    )
    audit = snapshot.probability_simplex_audit
    with pytest.raises(ValueError, match="frozen formula"):
        replace(audit, maximum_forward_error_bound=0.5)
    with pytest.raises(ValueError, match="error margin"):
        replace(audit, minimum_error_margin=0.5)
    with pytest.raises(ValueError, match="support inventory"):
        replace(
            audit,
            maximum_absolute_mass_residual=0.0,
            minimum_error_margin=0.0,
        )
    with pytest.raises(ValueError, match="support inventory"):
        replace(
            audit,
            maximum_absolute_mass_residual=(
                audit.maximum_forward_error_bound - 1.0e-9
            ),
            minimum_error_margin=1.0e-7,
        )

    forged_audit = AttentionRouteProbabilitySimplexAudit(
        formula_id=receipt.probability_simplex_formula_id,
        source_softmax_dtype="torch.float32",
        source_softmax_device="cpu",
        audit_accumulation_dtype="torch.float64",
        valid_row_count=receipt.probability_simplex_valid_row_count - 1,
        support_minimum=receipt.probability_simplex_support_minimum,
        support_maximum=receipt.probability_simplex_support_maximum,
        maximum_absolute_mass_residual=(
            receipt.probability_simplex_maximum_absolute_mass_residual
        ),
        maximum_forward_error_bound=(
            receipt.probability_simplex_maximum_forward_error_bound
        ),
        minimum_error_margin=receipt.probability_simplex_minimum_error_margin,
    )
    with pytest.raises(ValueError, match="row coverage"):
        replace(
            receipt,
            probability_simplex_valid_row_count=forged_audit.valid_row_count,
            probability_simplex_audit_sha256=forged_audit.sha256,
        )
    with pytest.raises(ValueError, match="kernel receipt"):
        replace(receipt, kernel_receipt_sha256=_sha(999))
    impossible_support_audit = AttentionRouteProbabilitySimplexAudit(
        formula_id=receipt.probability_simplex_formula_id,
        source_softmax_dtype="torch.float32",
        source_softmax_device="cpu",
        audit_accumulation_dtype="torch.float64",
        valid_row_count=receipt.probability_simplex_valid_row_count,
        support_minimum=1,
        support_maximum=5,
        maximum_absolute_mass_residual=(
            receipt.probability_simplex_maximum_absolute_mass_residual
        ),
        maximum_forward_error_bound=(
            attention_route_probability_simplex_forward_error_bound(5)
        ),
        minimum_error_margin=(
            attention_route_probability_simplex_forward_error_bound(1)
            - receipt.probability_simplex_maximum_absolute_mass_residual
        ),
    )
    with pytest.raises(ValueError, match="support inventory"):
        replace(
            receipt,
            probability_simplex_support_minimum=1,
            probability_simplex_support_maximum=5,
            probability_simplex_maximum_forward_error_bound=(
                impossible_support_audit.maximum_forward_error_bound
            ),
            probability_simplex_minimum_error_margin=(
                impossible_support_audit.minimum_error_margin
            ),
            probability_simplex_audit_sha256=impossible_support_audit.sha256,
        )
    with pytest.raises(ValueError, match="identically across heads"):
        replace(
            receipt,
            head_route_counts=(1, receipt.route_count - 1),
        )


def test_route_capture_rejects_dense_vs_logical_window_support_drift() -> None:
    generator = _generator(
        attention_kind="sliding_causal",
        window_size=2,
    )
    valid = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.tensor([[0, 10, 11, 12]], dtype=torch.int64)
    sequence = SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )
    hidden = torch.zeros(1, 4, HIDDEN, dtype=torch.float64)
    with pytest.raises(ValueError, match="logical route support"):
        generator(hidden, sequence, capture_route_factors=True)


def test_factorized_route_aggregate_matches_exhaustive_fixture_exactly() -> None:
    fixture = _factorized_fixture()
    snapshot, execution, operator_gradient, _, context, bridge, carrier = fixture
    exhaustive = _exhaustive_surface(fixture)
    (receipt,) = build_factorized_attention_route_aggregates(
        carrier=carrier,
        block_ordinal=0,
        context=context,
        postnorm_bridge=bridge,
        route_factors=snapshot,
        attention_operator_output=execution.operator_output,
        attention_operator_output_score_gradients=operator_gradient,
        distance_bin_upper_bounds=(1, 2, 4),
        top_mass_route_count=8,
    )
    route_indices = exhaustive.context_binding.native_route_mask.nonzero(
        as_tuple=False
    )
    coordinates = exhaustive.route_coordinates.to(torch.float64)
    intrinsic = exhaustive.route_intrinsic_fisher_energy.to(torch.float64)
    assert receipt.route_count == len(route_indices)
    assert receipt.probability_simplex_formula_id == (
        ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID
    )
    assert receipt.probability_simplex_valid_row_count == (
        HEADS * receipt.target_count
    )
    assert receipt.probability_simplex_maximum_absolute_mass_residual <= (
        receipt.probability_simplex_maximum_forward_error_bound
    )
    assert len(receipt.probability_simplex_audit_sha256) == 64
    torch.testing.assert_close(
        torch.tensor(
            receipt.intrinsic_route_fisher_energy,
            dtype=torch.float64,
        ),
        intrinsic.sum(),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    for head in range(HEADS):
        selector = route_indices[:, 2] == head
        assert receipt.head_route_counts[head] == int(selector.sum())
        torch.testing.assert_close(
            torch.tensor(
                receipt.head_intrinsic_fisher_energy[head],
                dtype=torch.float64,
            ),
            intrinsic[selector].sum(),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        torch.testing.assert_close(
            receipt.head_coordinate_mean[head],
            coordinates[selector].mean(dim=0),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        torch.testing.assert_close(
            receipt.head_coordinate_second_moment[head],
            coordinates[selector].square().mean(dim=0),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    top_energy = torch.sort(intrinsic, descending=True, stable=True).values[:8].sum()
    assert receipt.top_intrinsic_fisher_energy_fraction == pytest.approx(
        float(top_energy / intrinsic.sum()),
        rel=1.0e-15,
        abs=1.0e-15,
    )
    assert receipt.operator_output_closure_nrmse <= 1.0e-12
    assert receipt.operator_contribution_closure_nrmse <= 1.0e-12
    assert receipt.materialized_full_width_route_scalar_count == 0
    assert receipt.peak_route_energy_scalar_count <= (
        HEADS
        * FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE
        * FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE
    )
    assert receipt.peak_route_energy_scalar_count < (
        receipt.route_count * HIDDEN
    )
    assert receipt.supplemental_evidence() == (
        receipt.route_catalog_sha256,
        receipt.sha256,
    )


def test_factorized_zero_intrinsic_is_valid_with_nonzero_common_energy() -> None:
    fixture = _factorized_fixture(zero_operator_gradient=True)
    snapshot, execution, operator_gradient, _, context, bridge, carrier = fixture
    (receipt,) = build_factorized_attention_route_aggregates(
        carrier=carrier,
        block_ordinal=0,
        context=context,
        postnorm_bridge=bridge,
        route_factors=snapshot,
        attention_operator_output=execution.operator_output,
        attention_operator_output_score_gradients=operator_gradient,
    )
    assert receipt.intrinsic_route_fisher_energy == 0.0
    assert receipt.common_postnorm_target_fisher_energy > 0.0
    assert receipt.intrinsic_positive_target_count == 0
    assert receipt.intrinsic_zero_target_count == receipt.target_count
    assert receipt.common_positive_target_count > 0
    assert receipt.top_intrinsic_fisher_energy_fraction == 0.0


def test_factorized_kernel_receipt_is_fixed_and_backend_authenticated() -> None:
    receipt = factorized_attention_route_kernel_receipt()
    assert receipt["kernel_id"] == FACTORIZED_ATTENTION_ROUTE_KERNEL_ID
    assert receipt["query_chunk_size"] == 32
    assert receipt["key_chunk_size"] == 128
    assert receipt["full_width_route_vectors_materialized"] is False
    assert receipt["source_softmax_dtype"] == "torch.float32"
    assert receipt["source_softmax_device"] == "cpu"
    assert receipt["probability_simplex_audit_accumulation_dtype"] == (
        "torch.float64"
    )
    assert receipt["probability_simplex_forward_error_formula_id"] == (
        ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID
    )
    assert receipt["probability_simplex_maximum_support"] == 768
    assert receipt["stored_probability_bytes_renormalized"] is False
    assert isinstance(receipt["torch_git_version"], str)
    assert receipt["torch_git_version"]
    assert len(receipt["torch_build_config_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64


def test_generic_mutation_surface_separates_static_and_panel_receipts() -> None:
    fixture = _factorized_fixture()
    _, execution, _, postnorm_gradient, context, _, _ = fixture
    native = NativeChartABI(
        abi_kind=CAUSAL_SEQUENCE_NATIVE_ABI,
        abi_name="attention.residual_h640.causal_sequence.v1",
        abi_version=1,
        input_signature="raw_carrier_sequence.float.h640",
        output_signature="postnorm_delta.float.h640",
        context_signature="causal_mask.positions.rope.exact_context",
    )
    comparison = canonical_carrier_comparison_abi()
    static_receipt = _sha(41)
    surface = build_mutation_carrier_chart_surface(
        mode_id="mode.mutation.1",
        native_abi_sha256=native.sha256,
        comparison_abi_sha256=comparison.sha256,
        static_comparison_adapter_receipt_sha256=static_receipt,
        source_state_sha256=_sha(42),
        execution_fingerprint_sha256=_sha(43),
        score_objective_receipt_sha256=_sha(44),
        context_site_id="mutation.1.input",
        contribution_site_id="mutation.1.output",
        example_ids=context.example_ids,
        context_fingerprints=context.context_fingerprints,
        valid_mask=context.query_valid_mask,
        logical_positions=context.query_logical_positions,
        mutation_context_values=torch.randn_like(execution.residual_delta),
        mutation_output_values=execution.residual_delta,
        mutation_output_score_gradients=postnorm_gradient,
    )
    assert surface.carrier.mapping_receipt_sha256 == static_receipt
    assert surface.panel_mapping_receipt_sha256 != static_receipt
    assert len(surface.context_catalog_sha256) == 64
    mode_spec = ChartModeSpec(
        mode_id="mode.mutation.1",
        mode_kind="attention",
        state_fingerprint_sha256=_sha(42),
        event_id="mutation.1",
        event_index=1,
        context_site_id="mutation.1.input",
        contribution_site_id="mutation.1.output",
        site_placement_sha256=_sha(45),
        native_abi=native,
        comparison_abi=comparison,
        comparison_adapter_kind=DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        comparison_adapter_receipt_sha256=static_receipt,
    )
    assert (
        bind_carrier_signature_to_chart_mode_spec(surface.carrier, mode_spec)
        is surface.carrier
    )


def test_exact_executor_exports_18_factor_snapshots_and_auxiliary_vjps() -> None:
    # Reuse the repository's small source-free 18-block exact fixture; this
    # remains model-free and exercises the production compiled-attention seam.
    from test_gemma3_causal_carrier_transport_compiler import (  # noqa: PLC0415
        _sequence as exact_sequence,
        _toy_exact_executor,
    )

    executor = _toy_exact_executor()
    snapshots: list[tuple[str, int, Gemma3AttentionRouteFactorSnapshot]] = []
    observed = executor.forward_components_with_observer(
        torch.tensor([[1, 2, 3], [0, 4, 5]], dtype=torch.long),
        exact_sequence(),
        observer=lambda **_: None,
        retain_gradient_tape=True,
        attention_route_factor_observer=lambda **payload: snapshots.append(
            (
                payload["event_id"],
                payload["block_ordinal"],
                payload["snapshot"],
            )
        ),
    )
    assert tuple(item[0] for item in snapshots) == tuple(
        f"mutation.{index}" for index in range(1, 37, 2)
    )
    assert tuple(item[1] for item in snapshots) == tuple(range(18))
    assert all(
        isinstance(item[2], Gemma3AttentionRouteFactorSnapshot)
        for item in snapshots
    )
    assert observed.gradient_tape is not None
    bundle = observed.gradient_tape.score_gradients_with_auxiliary(
        observed.execution.logits.square().sum()
    )
    assert len(bundle.site_gradients) == 73
    assert len(bundle.attention_operator_output_gradients) == 18
    assert bundle.terminal_normalized_output_gradient.shape == (2, 3, 4)
