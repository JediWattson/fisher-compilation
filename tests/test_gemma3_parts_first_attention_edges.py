from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

from fisher_graph import gemma3_parts_first_attention_edges as attention_edges
from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.gemma3_attention_generator import (
    AttentionCoordinatePacket,
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
    compile_svd_reduced_attention_generator,
)
from fisher_graph.gemma3_parts_first_attention_edges import (
    ATTENTION_EDGE_PROBABILITY_FORMULA_ID,
    AttentionEdgeClosureReceipt,
    PartsFirstAttentionEdgeExtraction,
    compile_parts_first_attention_edges,
)


WIDTH = 8
HEADS = 2
KV_HEADS = 1
HEAD_DIMENSION = 4


def _norm(width: int) -> NormalizationSpec:
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
    rope_scaling_kind: str | None = None,
    rope_scaling_factor: float | None = None,
    logit_softcap: float | None = None,
    projection_bias: bool = False,
    dropout: float = 0.0,
    cancellation_output: bool = False,
):
    config = Gemma3AttentionGeneratorConfig(
        invocation_id="mutation.1",
        residual_width=WIDTH,
        attention=AttentionSpec(
            kind=attention_kind,
            query_heads=HEADS,
            key_value_heads=KV_HEADS,
            head_dimension=HEAD_DIMENSION,
            query_scale=HEAD_DIMENSION**-0.5,
            qk_norm=True,
            window_size=window_size,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=HEAD_DIMENSION,
                scaling_kind=rope_scaling_kind,
                scaling_factor=rope_scaling_factor,
            ),
            cache_kind="none",
        ),
        input_norm=_norm(WIDTH),
        output_norm=_norm(WIDTH),
        qk_norm=_norm(HEAD_DIMENSION),
        projection_bias=projection_bias,
        dropout=dropout,
        logit_softcap=logit_softcap,
    )
    random = torch.Generator().manual_seed(91_337)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(
            rows,
            columns,
            generator=random,
            dtype=torch.float32,
        ) / 7.0

    output = matrix(WIDTH, HEADS * HEAD_DIMENSION)
    if cancellation_output:
        first = matrix(WIDTH, HEAD_DIMENSION)
        output = torch.cat((first, -first + 1.0e-7), dim=1)
    bias = lambda width: torch.zeros(width, dtype=torch.float32)  # noqa: E731
    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=matrix(1, WIDTH).reshape(WIDTH),
            q_weight=matrix(HEADS * HEAD_DIMENSION, WIDTH),
            k_weight=matrix(KV_HEADS * HEAD_DIMENSION, WIDTH),
            v_weight=matrix(KV_HEADS * HEAD_DIMENSION, WIDTH),
            o_weight=output,
            output_norm_weight=matrix(1, WIDTH).reshape(WIDTH),
            q_norm_weight=matrix(1, HEAD_DIMENSION).reshape(HEAD_DIMENSION),
            k_norm_weight=matrix(1, HEAD_DIMENSION).reshape(HEAD_DIMENSION),
            q_bias=bias(HEADS * HEAD_DIMENSION) if projection_bias else None,
            k_bias=bias(KV_HEADS * HEAD_DIMENSION) if projection_bias else None,
            v_bias=bias(KV_HEADS * HEAD_DIMENSION) if projection_bias else None,
            o_bias=bias(WIDTH) if projection_bias else None,
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _sequence(
    *,
    valid: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> SequenceContext:
    if positions is None:
        positions = torch.arange(valid.shape[1], dtype=torch.int64)[None, :]
        positions = positions.expand(valid.shape[0], -1).clone()
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


def _inputs(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    random = torch.Generator().manual_seed(91_338)
    normalized = torch.randn(
        batch,
        length,
        WIDTH,
        generator=random,
        dtype=torch.float32,
    )
    gradient = torch.randn(
        batch,
        length,
        WIDTH,
        generator=random,
        dtype=torch.float32,
    )
    return normalized, gradient


def _exhaustive_reference(
    generator,
    normalized: torch.Tensor,
    delta_gradient: torch.Tensor,
    sequence: SequenceContext,
):
    effective = normalized.masked_fill(
        ~sequence.query_valid_mask.unsqueeze(-1),
        0,
    )
    leaf = effective.clone().requires_grad_(True)
    packet = AttentionCoordinatePacket(
        values=leaf,
        valid_token_mask=sequence.query_valid_mask,
        coordinate_system_sha256=generator.input_coordinate_system_sha256,
        semantic_role="normalized_input",
        source_invocation_id=generator.config.invocation_id,
    )
    replay = generator.forward_from_coordinates(packet, sequence)
    operator_gradient = torch.autograd.grad(
        replay.residual_delta,
        replay.operator_output,
        grad_outputs=delta_gradient,
    )[0].detach()
    attention = generator.config.attention
    batch, length, _ = effective.shape
    query = F.linear(effective, generator.q_weight).view(
        batch,
        length,
        attention.query_heads,
        attention.head_dimension,
    ).transpose(1, 2)
    key = F.linear(effective, generator.k_weight).view(
        batch,
        length,
        attention.key_value_heads,
        attention.head_dimension,
    ).transpose(1, 2)
    value = F.linear(effective, generator.v_weight).view(
        batch,
        length,
        attention.key_value_heads,
        attention.head_dimension,
    ).transpose(1, 2)
    assert generator.config.qk_norm is not None
    assert generator.q_norm_weight is not None
    assert generator.k_norm_weight is not None
    query = generator._rms_norm(  # noqa: SLF001 - independent committed reference
        query,
        generator.q_norm_weight,
        generator.config.qk_norm,
    )
    key = generator._rms_norm(  # noqa: SLF001
        key,
        generator.k_norm_weight,
        generator.config.qk_norm,
    )
    cos, sin = generator._rotary_cos_sin(  # noqa: SLF001
        sequence.logical_positions,
        dtype=torch.float32,
    )
    query = query * cos.unsqueeze(1) + generator._rotate_half(  # noqa: SLF001
        query
    ) * sin.unsqueeze(1)
    key = key * cos.unsqueeze(1) + generator._rotate_half(  # noqa: SLF001
        key
    ) * sin.unsqueeze(1)
    repeats = attention.query_heads // attention.key_value_heads
    repeated_key = generator._repeat_key_value(key, repeats)  # noqa: SLF001
    repeated_value = generator._repeat_key_value(value, repeats)  # noqa: SLF001
    scores = torch.matmul(query, repeated_key.transpose(-2, -1)) * float(
        attention.query_scale
    )
    if generator.config.logit_softcap is not None:
        cap = float(generator.config.logit_softcap)
        scores = torch.tanh(scores / cap) * cap
    allowed = generator._allowed_pairs(sequence)  # noqa: SLF001
    expanded_allowed = allowed[:, None].expand_as(scores)
    scores = scores.masked_fill(
        ~expanded_allowed,
        torch.finfo(torch.float32).min,
    )
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32).masked_fill(
        ~expanded_allowed,
        0,
    )
    output_slices = generator.output_decoder_weight.reshape(
        WIDTH,
        attention.query_heads,
        attention.head_dimension,
    ).permute(1, 2, 0)
    projected = torch.einsum(
        "bhkd,hdo->bhko",
        repeated_value,
        output_slices,
    ).to(torch.float64)
    probability64 = probability.to(torch.float64)
    gradient64 = operator_gradient.to(torch.float64)
    unit_signed = torch.einsum("bqo,bhko->bhqk", gradient64, projected)
    signed = (probability64 * unit_signed).masked_fill(~expanded_allowed, 0)
    energy = (
        probability64.square()
        * torch.einsum(
            "bqo,bhko->bhqk",
            gradient64.square(),
            projected.square(),
        )
    ).masked_fill(~expanded_allowed, 0)
    tangent = (
        signed - probability64 * signed.sum(dim=-1, keepdim=True)
    ).masked_fill(~expanded_allowed, 0)
    return allowed, probability, signed, energy, tangent


def test_full_gqa_rope_edge_tensors_match_exhaustive_reference_and_close() -> None:
    generator = _generator(logit_softcap=4.0)
    valid = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    positions = torch.tensor(
        [[0, 3, 7, 8, 20], [4, 9, 30, 31, 32]],
        dtype=torch.int64,
    )
    sequence = _sequence(valid=valid, positions=positions)
    normalized, gradient = _inputs(2, 5)

    extraction = compile_parts_first_attention_edges(
        generator,
        normalized,
        gradient,
        sequence,
    )
    allowed, probability, signed, energy, tangent = _exhaustive_reference(
        generator,
        normalized,
        gradient,
        sequence,
    )

    assert isinstance(extraction, PartsFirstAttentionEdgeExtraction)
    assert extraction.tensor_shape == (2, HEADS, 5, 5)
    assert torch.equal(extraction.allowed_mask, allowed)
    # The kernel derives the RoPE frequencies from the public configuration,
    # while the independent oracle consumes the generator's cached buffer.
    # Those paths can differ by one float32 ULP without changing the native
    # attention operator.
    torch.testing.assert_close(
        extraction.probabilities,
        probability,
        atol=torch.finfo(torch.float32).eps,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(extraction.signed_responses, signed)
    torch.testing.assert_close(
        extraction.intrinsic_fisher_energies,
        energy,
        atol=torch.finfo(torch.float32).eps,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(extraction.softmax_tangents, tangent)
    assert extraction.closure.passed is True
    assert extraction.closure.probability_formula_id == (
        ATTENTION_EDGE_PROBABILITY_FORMULA_ID
    )
    assert extraction.closure.native_edge_count == int(allowed.sum()) * HEADS
    assert extraction.closure.masked_probability_max_abs == 0.0
    assert extraction.closure.intrinsic_fisher_energy_total == pytest.approx(
        float(extraction.intrinsic_fisher_energies.sum()),
        rel=1.0e-15,
    )
    assert extraction.metadata()["full_width_route_vectors_materialized"] is False
    assert extraction.metadata()["logical_positions_define_visibility"] is False


def test_sliding_visibility_uses_tensor_ordinals_not_gapped_logical_positions() -> None:
    generator = _generator(
        attention_kind="sliding_causal",
        window_size=2,
        rope_scaling_kind="linear",
        rope_scaling_factor=2.0,
    )
    valid = torch.tensor([[True, True, True, True]])
    positions = torch.tensor([[0, 100, 101, 500]], dtype=torch.int64)
    sequence = _sequence(valid=valid, positions=positions)
    normalized, gradient = _inputs(1, 4)

    extraction = compile_parts_first_attention_edges(
        generator,
        normalized,
        gradient,
        sequence,
    )

    expected = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
            ]
        ]
    )
    assert torch.equal(extraction.allowed_mask, expected)
    expanded = expected[:, None].expand_as(extraction.probabilities)
    assert torch.count_nonzero(extraction.probabilities[~expanded]) == 0
    torch.testing.assert_close(
        extraction.probabilities.sum(dim=-1),
        torch.ones(1, HEADS, 4),
        atol=1.0e-6,
        rtol=0,
    )


def test_padding_zeroes_every_invalid_route_response_energy_and_tangent() -> None:
    generator = _generator(attention_kind="sliding_causal", window_size=2)
    valid = torch.tensor([[True, True, False, True]])
    sequence = _sequence(
        valid=valid,
        positions=torch.tensor([[0, 1, 100, 101]], dtype=torch.int64),
    )
    normalized, gradient = _inputs(1, 4)

    extraction = compile_parts_first_attention_edges(
        generator,
        normalized,
        gradient,
        sequence,
    )

    assert not bool(extraction.allowed_mask[0, 2].any())
    # q=3 has an ordinal window {2,3}; key 2 is padding, so only self remains.
    assert torch.equal(
        extraction.allowed_mask[0, 3],
        torch.tensor([False, False, False, True]),
    )
    expanded = extraction.allowed_mask[:, None].expand_as(extraction.probabilities)
    for values in (
        extraction.probabilities,
        extraction.signed_responses,
        extraction.intrinsic_fisher_energies,
        extraction.softmax_tangents,
    ):
        assert torch.count_nonzero(values[~expanded]) == 0
    tangent_row_sums = extraction.softmax_tangents.sum(dim=-1).abs()
    assert float(tangent_row_sums.max()) <= (
        extraction.closure.softmax_tangent_row_sum_max_abs_limit
    )
    assert torch.count_nonzero(tangent_row_sums[:, :, ~valid[0]]) == 0


def test_extraction_tensors_are_defensive_and_hashes_bind_payloads() -> None:
    generator = _generator()
    sequence = _sequence(valid=torch.ones(1, 4, dtype=torch.bool))
    normalized, gradient = _inputs(1, 4)
    extraction = compile_parts_first_attention_edges(
        generator,
        normalized,
        gradient,
        sequence,
    )
    original_probability = extraction.probabilities
    original_response = extraction.signed_responses
    original_sha = extraction.sha256

    extraction.probabilities.zero_()
    extraction.signed_responses.fill_(123.0)

    assert torch.equal(extraction.probabilities, original_probability)
    assert torch.equal(extraction.signed_responses, original_response)
    assert extraction.sha256 == original_sha
    metadata_json = json.dumps(extraction.metadata(), sort_keys=True)
    assert "_bytes" not in metadata_json
    assert "_probabilities_bytes" not in metadata_json
    assert original_sha in metadata_json
    changed = bytearray(extraction._probabilities_bytes)  # noqa: SLF001
    changed[0] ^= 1
    with pytest.raises(
        ValueError,
        match="tensor payload is invalid|probabilities hash drifted",
    ):
        replace(extraction, _probabilities_bytes=bytes(changed))
    zero_responses = torch.zeros_like(extraction.signed_responses)
    with pytest.raises(ValueError, match="softmax tangent relation drifted"):
        replace(
            extraction,
            _signed_responses_bytes=attention_edges._tensor_bytes(  # noqa: SLF001
                zero_responses
            ),
            signed_responses_sha256=attention_edges._tensor_sha256(  # noqa: SLF001
                zero_responses
            ),
            extraction_sha256="",
        )
    zero_tangents = torch.zeros_like(extraction.softmax_tangents)
    stale_response_closure = replace(
        extraction.closure,
        softmax_tangent_row_sum_max_abs=0.0,
        softmax_tangent_row_sum_max_abs_limit=2.0**-53,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="head response receipt drifted"):
        replace(
            extraction,
            _signed_responses_bytes=attention_edges._tensor_bytes(  # noqa: SLF001
                zero_responses
            ),
            signed_responses_sha256=attention_edges._tensor_sha256(  # noqa: SLF001
                zero_responses
            ),
            _softmax_tangents_bytes=attention_edges._tensor_bytes(  # noqa: SLF001
                zero_tangents
            ),
            softmax_tangents_sha256=attention_edges._tensor_sha256(  # noqa: SLF001
                zero_tangents
            ),
            closure=stale_response_closure,
            extraction_sha256="",
        )
    with pytest.raises(ValueError, match="extraction hash drifted"):
        replace(extraction, extraction_sha256="0" * 64)
    with pytest.raises(ValueError, match="may only be built"):
        replace(extraction, extraction_sha256="")


def test_nrmse_is_diagnostic_under_near_cancellation() -> None:
    generator = _generator(cancellation_output=True)
    sequence = _sequence(valid=torch.ones(1, 5, dtype=torch.bool))
    normalized, gradient = _inputs(1, 5)
    extraction = compile_parts_first_attention_edges(
        generator,
        normalized,
        gradient,
        sequence,
    )
    receipt = extraction.closure

    # The real contraction is close to zero because the two output-head
    # slices cancel.  Its relative diagnostic is therefore large even though
    # the mixed absolute/relative closure remains comfortably inside bounds.
    assert receipt.total_response_nrmse > 1.0e-5
    assert receipt.total_response_max_abs_error <= (
        receipt.total_response_max_abs_limit
    )

    diagnostic = replace(
        receipt,
        operator_output_nrmse=5.0e-3,
        head_response_nrmse=5.0e-3,
        total_response_nrmse=5.0e-3,
        receipt_sha256="",
    )

    assert isinstance(diagnostic, AttentionEdgeClosureReceipt)
    assert diagnostic.passed is True
    assert diagnostic.operator_output_max_abs_error <= (
        diagnostic.operator_output_max_abs_limit
    )


def test_kernel_fails_closed_on_arm_dtype_bias_dropout_and_bad_inputs() -> None:
    generator = _generator()
    sequence = _sequence(valid=torch.ones(1, 4, dtype=torch.bool))
    normalized, gradient = _inputs(1, 4)
    reduced = compile_svd_reduced_attention_generator(
        generator,
        input_rank=4,
        output_rank=4,
    ).float()

    with pytest.raises(ValueError, match="exact_full_rank"):
        compile_parts_first_attention_edges(
            reduced,
            normalized,
            gradient,
            sequence,
        )
    with pytest.raises(ValueError, match="CPU-float32"):
        compile_parts_first_attention_edges(
            _generator().double(),
            normalized.double(),
            gradient.double(),
            sequence,
        )
    with pytest.raises(ValueError, match="bias-free"):
        compile_parts_first_attention_edges(
            _generator(projection_bias=True),
            normalized,
            gradient,
            sequence,
        )
    with pytest.raises(ValueError, match="zero attention dropout"):
        compile_parts_first_attention_edges(
            _generator(dropout=0.1),
            normalized,
            gradient,
            sequence,
        )
    bad = normalized.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="normalized_input"):
        compile_parts_first_attention_edges(
            generator,
            bad,
            gradient,
            sequence,
        )
    with pytest.raises(ValueError, match="postnorm_delta_score_gradient"):
        compile_parts_first_attention_edges(
            generator,
            normalized,
            gradient[:, :-1],
            sequence,
        )


def test_receipts_reject_numerical_and_hash_tampering() -> None:
    generator = _generator()
    sequence = _sequence(valid=torch.ones(1, 3, dtype=torch.bool))
    normalized, gradient = _inputs(1, 3)
    receipt = compile_parts_first_attention_edges(
        generator,
        normalized,
        gradient,
        sequence,
    ).closure

    with pytest.raises(ValueError, match="numerical closure"):
        replace(
            receipt,
            total_response_max_abs_error=(
                receipt.total_response_max_abs_limit + 1.0
            ),
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="receipt hash drifted"):
        replace(receipt, receipt_sha256="f" * 64)
