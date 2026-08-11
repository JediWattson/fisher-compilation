from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from fisher_graph.adapters import (
    AttentionSpec,
    NormalizationSpec,
    RopeSpec,
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
)
from fisher_graph import gemma3_joint_cross_block_chart_executor as joint
from fisher_graph.gemma3_joint_cross_block_chart_executor import (
    ExactTokenLocalChartGenerator,
    Gemma3JointCrossBlockChartExecutor,
    JointGraphGeneratorMutation,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)
from fisher_graph.gemma3_parts_first_attention_edge_fidelity import (
    ATTENTION_EDGE_FIDELITY_K_VALUES,
    attention_route_tensors,
    replay_exact_with_attention_masks,
    select_global_attention_route_mask,
    sparse_attention_from_normalized,
)


WIDTH = 4
HEADS = 2
HEAD_DIMENSION = 2


def _norm(width: int) -> NormalizationSpec:
    return NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )


def _generator(event_id: str = "mutation.1", *, global_attention: bool = False):
    norm = _norm(WIDTH)
    config = Gemma3AttentionGeneratorConfig(
        invocation_id=event_id,
        residual_width=WIDTH,
        attention=AttentionSpec(
            kind="global_causal" if global_attention else "sliding_causal",
            query_heads=HEADS,
            key_value_heads=1,
            head_dimension=HEAD_DIMENSION,
            query_scale=HEAD_DIMENSION**-0.5,
            qk_norm=True,
            window_size=None if global_attention else 512,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=HEAD_DIMENSION,
            ),
            cache_kind="none",
        ),
        input_norm=norm,
        output_norm=norm,
        qk_norm=replace(norm, width=HEAD_DIMENSION),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=7.0,
    )
    random = torch.Generator().manual_seed(2101 + int(event_id.split(".")[-1]))

    def values(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=random, dtype=torch.float32) * 0.1

    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=values(WIDTH),
            q_weight=values(HEADS * HEAD_DIMENSION, WIDTH),
            k_weight=values(HEAD_DIMENSION, WIDTH),
            v_weight=values(HEAD_DIMENSION, WIDTH),
            o_weight=values(WIDTH, HEADS * HEAD_DIMENSION),
            output_norm_weight=values(WIDTH),
            q_norm_weight=values(HEAD_DIMENSION),
            k_norm_weight=values(HEAD_DIMENSION),
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _sequence(length: int = 4) -> SequenceContext:
    valid = torch.ones(1, length, dtype=torch.bool)
    positions = torch.arange(length, dtype=torch.int64).reshape(1, -1)
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


def test_all_routes_reconstruct_public_operator_and_delta_bitwise() -> None:
    generator = _generator()
    sequence = _sequence()
    normalized = torch.randn(1, 4, WIDTH, generator=torch.Generator().manual_seed(9))

    factors = attention_route_tensors(generator, normalized, sequence)

    assert factors.all_routes_operator_literal_parity_passed is True
    assert factors.all_routes_delta_literal_parity_passed is True
    assert factors.route_output_energies.shape == factors.probabilities.shape
    assert bool((factors.route_output_energies >= 0).all())


def test_all_routes_preserve_invalid_operator_bytes_but_zero_delta() -> None:
    generator = _generator()
    valid = torch.tensor([[True, True, False]])
    positions = torch.tensor([[5, 19, 1_003]], dtype=torch.int64)
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
    normalized = torch.randn(
        1, 3, WIDTH, generator=torch.Generator().manual_seed(91)
    )
    factors = attention_route_tensors(generator, normalized, sequence)
    selection = factors.allowed_mask[:, None].expand_as(factors.probabilities)
    sparse = sparse_attention_from_normalized(
        generator,
        normalized,
        sequence,
        selection,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
    )

    assert factors.probabilities[:, :, 2].count_nonzero() == 0
    assert torch.equal(
        factors.execution_probabilities[:, :, 2].sum(dim=-1),
        torch.ones(1, HEADS),
    )
    assert factors.reference_operator_output[:, 2].count_nonzero() > 0
    assert factors.reference_residual_delta[:, 2].count_nonzero() == 0
    assert torch.equal(
        sparse.operator_output, factors.reference_operator_output
    )
    assert torch.equal(
        sparse.residual_delta, factors.reference_residual_delta
    )


def test_global_selection_is_nested_and_can_starve_heads() -> None:
    probabilities = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.6, 0.3, 0.1]],
          [[1.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.5, 0.3, 0.2]]]],
        dtype=torch.float32,
    )
    allowed = torch.tensor(
        [[[True, False, False], [True, True, False], [True, True, True]]]
    )
    fisher = probabilities.to(torch.float64).square()
    route_output = fisher * 0.5
    masks = [
        select_global_attention_route_mask(
            probabilities,
            fisher,
            route_output,
            allowed,
            ranker="fisher",
            k=k,
            row_catalog_id="row",
            layer_ordinal=0,
        )
        for k in ATTENTION_EDGE_FIDELITY_K_VALUES
    ]

    for smaller, larger in zip(masks, masks[1:]):
        assert not bool((smaller & ~larger).any())
    assert masks[0][0, 0, 2].sum() == 1
    assert masks[0][0, 1, 2].sum() == 0


def test_random_salts_are_deterministic_distinct_and_support_matched() -> None:
    generator = _generator()
    sequence = _sequence(8)
    normalized = torch.randn(1, 8, WIDTH, generator=torch.Generator().manual_seed(10))
    factors = attention_route_tensors(generator, normalized, sequence)
    fisher = torch.ones_like(factors.probabilities, dtype=torch.float64)

    primary = select_global_attention_route_mask(
        factors.probabilities,
        fisher,
        factors.route_output_energies,
        factors.allowed_mask,
        ranker="deterministic_random",
        k=8,
        row_catalog_id="row",
        layer_ordinal=2,
        random_salt="primary",
    )
    repeated = select_global_attention_route_mask(
        factors.probabilities,
        fisher,
        factors.route_output_energies,
        factors.allowed_mask,
        ranker="deterministic_random",
        k=8,
        row_catalog_id="row",
        layer_ordinal=2,
        random_salt="primary",
    )
    replicate = select_global_attention_route_mask(
        factors.probabilities,
        fisher,
        factors.route_output_energies,
        factors.allowed_mask,
        ranker="deterministic_random",
        k=8,
        row_catalog_id="row",
        layer_ordinal=2,
        random_salt="replicate_1",
    )

    assert torch.equal(primary, repeated)
    assert int(primary.sum()) == int(replicate.sum())
    assert not torch.equal(primary, replicate)


def test_sparse_renormalization_zero_fills_starved_heads() -> None:
    generator = _generator()
    sequence = _sequence()
    normalized = torch.randn(1, 4, WIDTH, generator=torch.Generator().manual_seed(11))
    factors = attention_route_tensors(generator, normalized, sequence)
    selection = torch.zeros_like(factors.probabilities, dtype=torch.bool)
    selection[:, 0, :, 0] = factors.allowed_mask[:, :, 0]

    execution = sparse_attention_from_normalized(
        generator,
        normalized,
        sequence,
        selection,
        renormalize_rows=torch.ones(1, dtype=torch.bool),
    )

    assert torch.equal(
        execution.routed_probabilities[:, 1],
        torch.zeros_like(execution.routed_probabilities[:, 1]),
    )
    assert not bool(execution.active_head_query_mask[:, 1].any())
    assert torch.equal(
        execution.routed_probabilities[:, 0].sum(dim=-1),
        torch.ones(1, 4),
    )


def _executor() -> Gemma3JointCrossBlockChartExecutor:
    schedule = joint.build_joint_cross_block_event_schedule()
    mutations: list[JointGraphGeneratorMutation] = []
    for event in schedule[1:-2]:
        if event.event_kind == "attention_mutation":
            mutations.append(
                JointGraphGeneratorMutation(
                    event_spec=event,
                    compiled_operator=_generator(
                        event.event_id,
                        global_attention=event.attention_kind == "global",
                    ),
                    source_checkpoint_coefficients_retained=True,
                )
            )
        else:
            mutations.append(
                JointGraphGeneratorMutation(
                    event_spec=event,
                    input_norm=nn.Identity(),
                    chart_generator=ExactTokenLocalChartGenerator(
                        operator=nn.Identity(),
                        hidden_width=WIDTH,
                        chart_count=1,
                        generator_rank=WIDTH,
                    ),
                    output_norm=nn.Identity(),
                    source_checkpoint_coefficients_retained=True,
                )
            )
    lexical = SharedTiedLexicalFactor(
        exact_weight=torch.randn(
            9, WIDTH, generator=torch.Generator().manual_seed(12)
        )
    )
    return Gemma3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=tuple(mutations),
        transports=(),
        terminal_norm=nn.Identity(),
        final_logit_softcap=9.0,
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )


def test_whole_stack_all_route_replay_is_literal_and_sparse_changes_logits() -> None:
    executor = _executor()
    sequence = _sequence(3)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    mutation_inputs: dict[str, torch.Tensor] = {}
    mutation_deltas: dict[str, torch.Tensor] = {}
    lexical_outputs: list[torch.Tensor] = []

    def observer(*, event_id: str, role: str, values: torch.Tensor) -> None:
        if event_id == "lexical.ingress" and role == "output":
            lexical_outputs.append(values)
            return
        if not event_id.startswith("mutation."):
            return
        if role == "input":
            mutation_inputs[event_id] = values
        elif role == "output":
            mutation_deltas[event_id] = values

    baseline = executor.forward_components_with_observer(
        input_ids, sequence, observer=observer
    ).execution
    allowed = torch.tril(torch.ones(3, 3, dtype=torch.bool)).reshape(1, 1, 3, 3)
    all_routes = tuple(allowed.expand(1, HEADS, 3, 3).clone() for _ in range(18))

    dense = replay_exact_with_attention_masks(
        executor,
        input_ids,
        sequence,
        all_routes,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
    )
    assert torch.equal(dense.output, baseline.output)
    assert torch.equal(dense.normalized_output, baseline.normalized_output)
    assert torch.equal(dense.logits, baseline.logits)
    assert len(dense.mutation_residual_deltas) == 36
    assert len(dense.mutation_carriers) == 36
    valid = sequence.query_valid_mask.unsqueeze(-1)
    expected_carrier = lexical_outputs[0]
    for mutation, (actual_delta, actual_carrier) in enumerate(
        zip(
            dense.mutation_residual_deltas,
            dense.mutation_carriers,
            strict=True,
        ),
        start=1,
    ):
        event_id = f"mutation.{mutation}"
        expected_delta = mutation_deltas[event_id]
        expected_carrier = torch.where(
            valid,
            expected_carrier + expected_delta.masked_fill(~valid, 0),
            lexical_outputs[0],
        )
        assert torch.equal(actual_delta, expected_delta)
        assert torch.equal(actual_carrier, expected_carrier)

    sparse_masks = tuple(mask & (torch.arange(3).view(1, 1, 1, 3) == 0) for mask in all_routes)
    sparse = replay_exact_with_attention_masks(
        executor,
        input_ids,
        sequence,
        sparse_masks,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
        egress_query_ordinal=2,
    )
    assert sparse.logits.shape == (1, 1, 9)
    assert not torch.equal(sparse.logits[:, 0], baseline.logits[:, 2])


def test_whole_stack_dense_replay_is_literal_with_padding_and_gapped_positions() -> None:
    executor = _executor()
    valid = torch.tensor([[True, True, False]])
    positions = torch.tensor([[7, 29, 1_009]], dtype=torch.int64)
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
    input_ids = torch.tensor([[1, 2, 0]], dtype=torch.int64)
    baseline = executor.forward_components(input_ids, sequence)
    allowed = (
        valid[:, :, None]
        & valid[:, None, :]
        & torch.tril(torch.ones(3, 3, dtype=torch.bool)).unsqueeze(0)
    )
    all_routes = tuple(
        allowed[:, None].expand(1, HEADS, 3, 3).clone()
        for _ in range(18)
    )

    dense = replay_exact_with_attention_masks(
        executor,
        input_ids,
        sequence,
        all_routes,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
    )

    assert torch.equal(dense.output, baseline.output)
    assert torch.equal(dense.normalized_output, baseline.normalized_output)
    assert torch.equal(dense.logits, baseline.logits)
    assert all(
        torch.equal(carrier[:, 2], dense.mutation_carriers[0][:, 2])
        for carrier in dense.mutation_carriers[1:]
    )


def test_selector_rejects_nonrandom_replicate_salt() -> None:
    generator = _generator()
    sequence = _sequence()
    factors = attention_route_tensors(generator, torch.zeros(1, 4, WIDTH), sequence)
    with pytest.raises(ValueError, match="random replicates"):
        select_global_attention_route_mask(
            factors.probabilities,
            torch.zeros_like(factors.probabilities, dtype=torch.float64),
            factors.route_output_energies,
            factors.allowed_mask,
            ranker="probability",
            k=8,
            row_catalog_id="row",
            layer_ordinal=0,
            random_salt="replicate_1",
        )
