from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from fisher_graph.gemma3_chart_mode_surfaces import (
    MAX_EXHAUSTIVE_ATTENTION_ROUTE_ROWS,
    AttentionRouteAggregateReceipt,
    AttentionRouteChartAccumulator,
    AttentionRouteChartSurface,
    AttentionRouteStreamContext,
    CarrierChartSignature,
    LexicalEgressChartSurface,
    LexicalIngressChartSurface,
    NativeRoPESpec,
    TiedLexicalStateLineage,
    build_attention_postnorm_bridge_receipt,
    build_attention_route_chart_surface,
    build_attention_target_carrier_signature,
    build_lexical_egress_chart_surface,
    build_lexical_egress_from_terminal_score_gradient,
    build_lexical_ingress_chart_surface,
    pull_back_post_softcap_logit_covector,
    bind_carrier_signature_to_chart_mode_spec,
    strict_load_attention_route_aggregate,
)
from fisher_graph.chart_energy_neighbor_graph import (
    CAUSAL_SEQUENCE_NATIVE_ABI,
    DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
    EMBEDDING_CARRIER_H640_ADAPTER,
    LEXICAL_INGRESS_NATIVE_ABI,
    LEXICAL_READOUT_NATIVE_ABI,
    SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
    ChartModeSpec,
    NativeChartABI,
    canonical_carrier_comparison_abi,
)


HIDDEN = 640


def _sha(index: int) -> str:
    return f"{index:064x}"


def _rope(*, theta: float = 10_000.0) -> NativeRoPESpec:
    return NativeRoPESpec(
        theta=theta,
        rotary_dimension=256,
        scaling_kind="none",
    )


def _native_causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    heads: int,
    window_size: int | None = None,
) -> torch.Tensor:
    mask = (
        key_positions[:, None, None, :]
        <= query_positions[:, :, None, None]
    ).expand(
        query_positions.shape[0],
        query_positions.shape[1],
        heads,
        key_positions.shape[1],
    )
    if window_size is not None:
        mask = mask & (
            key_positions[:, None, None, :]
            > query_positions[:, :, None, None] - window_size
        ).expand_as(mask)
    return mask.clone()


def _attention_fixture() -> tuple[AttentionRouteChartSurface, dict[str, object]]:
    generator = torch.Generator().manual_seed(77)
    query_valid = torch.tensor([[True, True, True]])
    key_valid = query_valid.clone()
    positions = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    native = _native_causal_mask(positions, positions, heads=2)
    target = torch.randn(1, 3, HIDDEN, generator=generator)
    target_gradient = torch.randn(1, 3, HIDDEN, generator=generator)
    operator_gradient = torch.randn(1, 3, HIDDEN, generator=generator)
    routes = torch.randn(1, 3, 2, 3, HIDDEN, generator=generator)
    operator = (routes * native[..., None]).sum(dim=(2, 3))
    kwargs: dict[str, object] = {
        "mode_id": "mode.mutation.1",
        "block_ordinal": 0,
        "native_abi_sha256": _sha(1),
        "comparison_abi_sha256": _sha(2),
        "source_state_sha256": _sha(3),
        "execution_fingerprint_sha256": _sha(4),
        "score_objective_receipt_sha256": _sha(8),
        "example_ids": ("example.safe.1",),
        "context_fingerprints": (_sha(5),),
        "query_valid_mask": query_valid,
        "key_valid_mask": key_valid,
        "query_logical_positions": positions,
        "key_logical_positions": positions,
        "native_route_mask": native,
        "target_carrier_values": target,
        "target_carrier_score_gradients": target_gradient,
        "attention_operator_output": operator,
        "attention_operator_output_score_gradients": operator_gradient,
        "route_operator_output_values": routes,
        "postnorm_state_sha256": _sha(6),
        "postnorm_config_sha256": _sha(7),
        "rope": _rope(),
        "window_size": None,
    }
    return build_attention_route_chart_surface(**kwargs), kwargs


def _stream_context(kwargs: dict[str, object]) -> AttentionRouteStreamContext:
    return AttentionRouteStreamContext(
        example_ids=kwargs["example_ids"],  # type: ignore[arg-type]
        context_fingerprints=kwargs["context_fingerprints"],  # type: ignore[arg-type]
        query_valid_mask=kwargs["query_valid_mask"],  # type: ignore[arg-type]
        key_valid_mask=kwargs["key_valid_mask"],  # type: ignore[arg-type]
        query_logical_positions=kwargs["query_logical_positions"],  # type: ignore[arg-type]
        key_logical_positions=kwargs["key_logical_positions"],  # type: ignore[arg-type]
        query_head_count=2,
        window_size=None,
        rope=kwargs["rope"],  # type: ignore[arg-type]
    )


def _route_stream(
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = kwargs["native_route_mask"]
    routes = kwargs["route_operator_output_values"]
    assert isinstance(mask, torch.Tensor) and isinstance(routes, torch.Tensor)
    return mask.nonzero(as_tuple=False), routes[mask]


def _bridge(kwargs: dict[str, object]):
    return build_attention_postnorm_bridge_receipt(
        operator_output=kwargs["attention_operator_output"],  # type: ignore[arg-type]
        operator_output_score_gradient=kwargs[  # type: ignore[arg-type]
            "attention_operator_output_score_gradients"
        ],
        postnorm_delta=kwargs["target_carrier_values"],  # type: ignore[arg-type]
        postnorm_state_sha256=kwargs["postnorm_state_sha256"],  # type: ignore[arg-type]
        postnorm_config_sha256=kwargs["postnorm_config_sha256"],  # type: ignore[arg-type]
        execution_fingerprint_sha256=kwargs[  # type: ignore[arg-type]
            "execution_fingerprint_sha256"
        ],
    )


def test_attention_surface_keys_every_native_causal_route_and_conserves_energy() -> None:
    surface, kwargs = _attention_fixture()
    assert surface.carrier.hidden_width == HIDDEN
    assert len(surface.carrier.row_keys) == 3
    assert len(surface.route_row_keys) == 12
    assert surface.route_row_keys[0].state_dict() == {
        "context_fingerprint_sha256": _sha(5),
        "example_id": "example.safe.1",
        "source_position": 0,
        "target_position": 0,
        "query_head": 0,
    }
    assert surface.route_row_keys[-1].source_position == 2
    assert surface.route_row_keys[-1].target_position == 2
    assert surface.route_row_keys[-1].query_head == 1
    target = kwargs["target_carrier_values"]
    target_gradient = kwargs["target_carrier_score_gradients"]
    assert isinstance(target, torch.Tensor) and isinstance(target_gradient, torch.Tensor)
    assert torch.equal(
        surface.carrier.coordinates,
        (target * target_gradient).reshape(3, HIDDEN),
    )
    operator_gradient = kwargs["attention_operator_output_score_gradients"]
    routes = kwargs["route_operator_output_values"]
    mask = kwargs["native_route_mask"]
    assert all(isinstance(value, torch.Tensor) for value in (operator_gradient, routes, mask))
    first_route = routes[mask][0]  # type: ignore[index]
    assert torch.equal(
        surface.route_coordinates[0],
        first_route * operator_gradient[0, 0],  # type: ignore[index]
    )
    for target_index in range(3):
        selector = surface.route_target_row_indices == target_index
        assert torch.isclose(surface.route_energy_fraction[selector].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.isclose(
            surface.route_allocated_target_fisher_energy[selector].sum(),
            surface.carrier.fisher_energy[target_index],
        )
    assert (
        surface.postnorm_bridge.state_dict()[
            "additive_equality_across_nonlinear_postnorm_claimed"
        ]
        is False
    )
    assert surface.postnorm_bridge.state_dict()["route_additive_closure_site"] == "attention.operator_output"


def test_attention_surface_strict_state_rejects_tensor_and_hash_tampering() -> None:
    surface, _ = _attention_fixture()
    state = surface.state_dict()
    loaded = AttentionRouteChartSurface.from_state_dict(state)
    assert loaded.sha256 == surface.sha256
    tampered = deepcopy(state)
    tampered["route_coordinates"][0, 0] += 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="intrinsic route energy"):
        AttentionRouteChartSurface.from_state_dict(tampered)
    tampered = deepcopy(state)
    tampered["surface_sha256"] = _sha(99)
    with pytest.raises(ValueError, match="surface_sha256"):
        AttentionRouteChartSurface.from_state_dict(tampered)


def test_attention_mask_window_and_rope_are_exact_native_bindings() -> None:
    _, kwargs = _attention_fixture()
    future = kwargs["native_route_mask"].clone()  # type: ignore[union-attr]
    future[0, 0, 0, 2] = True
    kwargs["native_route_mask"] = future
    with pytest.raises(ValueError, match="exactly match"):
        build_attention_route_chart_surface(**kwargs)

    _, kwargs = _attention_fixture()
    kwargs["window_size"] = 2
    with pytest.raises(ValueError, match="exactly match"):
        build_attention_route_chart_surface(**kwargs)
    kwargs["native_route_mask"] = _native_causal_mask(
        kwargs["query_logical_positions"],  # type: ignore[arg-type]
        kwargs["key_logical_positions"],  # type: ignore[arg-type]
        heads=2,
        window_size=2,
    )
    kwargs["attention_operator_output"] = (
        kwargs["route_operator_output_values"]  # type: ignore[operator]
        * kwargs["native_route_mask"][..., None]  # type: ignore[index]
    ).sum(dim=(2, 3))
    windowed = build_attention_route_chart_surface(**kwargs)
    assert windowed.context_binding.window_size == 2

    original, kwargs = _attention_fixture()
    kwargs["rope"] = _rope(theta=1_000_000.0)
    changed = build_attention_route_chart_surface(**kwargs)
    assert changed.context_binding.sha256 != original.context_binding.sha256
    assert changed.sha256 != original.sha256


def test_streamed_route_aggregate_is_chunk_invariant_bounded_and_strict() -> None:
    surface, kwargs = _attention_fixture()
    context = _stream_context(kwargs)
    indices, values = _route_stream(kwargs)

    def run(cuts: tuple[int, ...]) -> AttentionRouteAggregateReceipt:
        accumulator = AttentionRouteChartAccumulator(
            carrier=surface.carrier,
            block_ordinal=0,
            context=context,
            postnorm_bridge=_bridge(kwargs),
            attention_operator_output=kwargs[  # type: ignore[arg-type]
                "attention_operator_output"
            ],
            attention_operator_output_score_gradients=kwargs[  # type: ignore[arg-type]
                "attention_operator_output_score_gradients"
            ],
            distance_bin_upper_bounds=(1, 2),
            top_mass_route_count=4,
        )
        assert accumulator.resident_scalar_count_upper_bound < 15_000
        start = 0
        for stop in cuts:
            accumulator.add_chunk(
                route_indices=indices[start:stop],
                route_carrier_values=values[start:stop],
            )
            start = stop
        return accumulator.finish()

    whole = run((12,))
    chunked = run((1, 5, 9, 12))
    assert whole.sha256 == chunked.sha256
    assert whole.route_count == 12
    assert whole.positive_target_count == 3
    assert sum(whole.distance_bin_counts) == 12
    assert sum(whole.head_route_counts) == 12
    assert whole.allocated_target_fisher_energy == whole.target_fisher_energy
    with pytest.raises(ValueError, match="verified tolerance"):
        replace(whole, operator_output_closure_nrmse=1.0)
    with pytest.raises(ValueError, match="verified tolerance"):
        replace(
            whole,
            operator_output_closure_max_abs=(
                whole.operator_output_closure_max_abs_limit + 1.0
            ),
        )
    state = whole.state_dict()
    assert state["full_route_rows_persisted"] is False
    assert state["full_route_coordinates_persisted"] is False
    assert strict_load_attention_route_aggregate(state).sha256 == whole.sha256
    tampered = deepcopy(state)
    tampered["route_row_catalog_sha256"] = _sha(98)
    with pytest.raises(ValueError, match="failed validation"):
        strict_load_attention_route_aggregate(tampered)
    for key, replacement, match in (
        ("route_count", True, "exact integer"),
        ("intrinsic_route_fisher_energy", float("nan"), "finite number"),
        ("operator_output_closure_nrmse", float("nan"), "finite number"),
    ):
        tampered = deepcopy(state)
        tampered[key] = replacement
        with pytest.raises(ValueError, match=match):
            strict_load_attention_route_aggregate(tampered)
    tampered = deepcopy(state)
    tampered["distance_bin_counts"][0] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="exact integer"):
        strict_load_attention_route_aggregate(tampered)


def test_streamed_route_aggregate_is_chunk_invariant_beyond_internal_block() -> None:
    generator = torch.Generator().manual_seed(7701)
    sequence_length = 32
    query_heads = 4
    valid = torch.ones(1, sequence_length, dtype=torch.bool)
    positions = torch.arange(sequence_length, dtype=torch.int64)[None, :]
    native = _native_causal_mask(positions, positions, heads=query_heads)
    indices = native.nonzero(as_tuple=False)
    route_values = torch.randn(
        len(indices),
        HIDDEN,
        dtype=torch.float64,
        generator=generator,
    )
    operator = torch.zeros(
        sequence_length,
        HIDDEN,
        dtype=torch.float64,
    )
    operator.index_add_(0, indices[:, 1], route_values)
    operator = operator[None, :, :]
    operator_gradient = torch.randn(
        1,
        sequence_length,
        HIDDEN,
        dtype=torch.float64,
        generator=generator,
    )
    postnorm_delta = torch.randn(
        1,
        sequence_length,
        HIDDEN,
        dtype=torch.float64,
        generator=generator,
    )
    postnorm_gradient = torch.randn(
        1,
        sequence_length,
        HIDDEN,
        dtype=torch.float64,
        generator=generator,
    )
    context = AttentionRouteStreamContext(
        example_ids=("example.long.safe",),
        context_fingerprints=(_sha(81),),
        query_valid_mask=valid,
        key_valid_mask=valid,
        query_logical_positions=positions,
        key_logical_positions=positions,
        query_head_count=query_heads,
        window_size=None,
        rope=_rope(),
    )
    bridge = build_attention_postnorm_bridge_receipt(
        operator_output=operator,
        operator_output_score_gradient=operator_gradient,
        postnorm_delta=postnorm_delta,
        postnorm_state_sha256=_sha(82),
        postnorm_config_sha256=_sha(83),
        execution_fingerprint_sha256=_sha(84),
    )
    carrier = build_attention_target_carrier_signature(
        mode_id="mode.mutation.9",
        block_ordinal=8,
        native_abi_sha256=_sha(85),
        comparison_abi_sha256=_sha(86),
        source_state_sha256=_sha(87),
        execution_fingerprint_sha256=_sha(84),
        score_objective_receipt_sha256=_sha(88),
        attention_context_binding_sha256=context.sha256,
        postnorm_bridge=bridge,
        example_ids=("example.long.safe",),
        query_valid_mask=valid,
        query_logical_positions=positions,
        target_carrier_values=postnorm_delta,
        target_carrier_score_gradients=postnorm_gradient,
    )

    def run(chunk_rows: int) -> AttentionRouteAggregateReceipt:
        accumulator = AttentionRouteChartAccumulator(
            carrier=carrier,
            block_ordinal=8,
            context=context,
            postnorm_bridge=bridge,
            attention_operator_output=operator,
            attention_operator_output_score_gradients=operator_gradient,
            top_mass_route_count=16,
        )
        assert accumulator.resident_scalar_count_upper_bound < len(indices) * HIDDEN
        for start in range(0, len(indices), chunk_rows):
            stop = min(start + chunk_rows, len(indices))
            accumulator.add_chunk(
                route_indices=indices[start:stop],
                route_carrier_values=route_values[start:stop],
            )
        return accumulator.finish()

    whole = run(len(indices))
    arbitrarily_chunked = run(17)
    assert len(indices) > 2_048
    assert whole.sha256 == arbitrarily_chunked.sha256
    assert whole.route_row_catalog_sha256 == arbitrarily_chunked.route_row_catalog_sha256
    assert whole.reconstructed_operator_output_sha256 == (
        arbitrarily_chunked.reconstructed_operator_output_sha256
    )


def test_rope_strict_parser_rejects_bool_and_nan_numeric_aliases() -> None:
    state = _rope().state_dict()
    tampered = dict(state)
    tampered["rotary_dimension"] = True
    with pytest.raises(ValueError, match="exact integer"):
        NativeRoPESpec.from_state_dict(tampered)
    tampered = dict(state)
    tampered["theta"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        NativeRoPESpec.from_state_dict(tampered)


def test_streaming_rejects_missing_duplicate_and_masked_routes() -> None:
    surface, kwargs = _attention_fixture()
    context = _stream_context(kwargs)
    indices, values = _route_stream(kwargs)

    def accumulator() -> AttentionRouteChartAccumulator:
        return AttentionRouteChartAccumulator(
            carrier=surface.carrier,
            block_ordinal=0,
            context=context,
            postnorm_bridge=_bridge(kwargs),
            attention_operator_output=kwargs[  # type: ignore[arg-type]
                "attention_operator_output"
            ],
            attention_operator_output_score_gradients=kwargs[  # type: ignore[arg-type]
                "attention_operator_output_score_gradients"
            ],
        )

    missing = accumulator()
    missing.add_chunk(route_indices=indices[:-1], route_carrier_values=values[:-1])
    with pytest.raises(ValueError, match="complete native route catalog"):
        missing.finish()

    duplicate = accumulator()
    duplicate.add_chunk(route_indices=indices[:1], route_carrier_values=values[:1])
    with pytest.raises(ValueError, match="strict canonical"):
        duplicate.add_chunk(route_indices=indices[:1], route_carrier_values=values[:1])

    masked = accumulator()
    with pytest.raises(ValueError, match="masked or noncausal"):
        masked.add_chunk(
            route_indices=torch.tensor([[0, 0, 0, 2]]),
            route_carrier_values=torch.ones(1, HIDDEN),
        )

    forged = accumulator()
    forged_values = values.clone()
    forged_values[0, 0] += 1.0
    forged.add_chunk(route_indices=indices, route_carrier_values=forged_values)
    with pytest.raises(ValueError, match="do not close to operator output"):
        forged.finish()


def test_terminal_zero_gradient_target_is_retained_with_zero_route_energy() -> None:
    _, kwargs = _attention_fixture()
    kwargs["target_carrier_score_gradients"] = kwargs[
        "target_carrier_score_gradients"
    ].clone()  # type: ignore[union-attr]
    kwargs["target_carrier_score_gradients"][:, -1] = 0.0  # type: ignore[index]
    kwargs["attention_operator_output_score_gradients"] = kwargs[
        "attention_operator_output_score_gradients"
    ].clone()  # type: ignore[union-attr]
    kwargs["attention_operator_output_score_gradients"][:, -1] = 0.0  # type: ignore[index]
    surface = build_attention_route_chart_surface(**kwargs)
    assert surface.carrier.fisher_energy[-1] == 0
    final_routes = surface.route_target_row_indices == 2
    assert surface.route_intrinsic_fisher_energy[final_routes].sum() == 0
    assert surface.route_energy_fraction[final_routes].sum() == 0
    assert surface.route_allocated_target_fisher_energy[final_routes].sum() == 0

    context = _stream_context(kwargs)
    indices, values = _route_stream(kwargs)
    accumulator = AttentionRouteChartAccumulator(
        carrier=surface.carrier,
        block_ordinal=0,
        context=context,
        postnorm_bridge=_bridge(kwargs),
        attention_operator_output=kwargs["attention_operator_output"],  # type: ignore[arg-type]
        attention_operator_output_score_gradients=kwargs[  # type: ignore[arg-type]
            "attention_operator_output_score_gradients"
        ],
    )
    accumulator.add_chunk(route_indices=indices, route_carrier_values=values)
    receipt = accumulator.finish()
    assert receipt.positive_target_count == 2
    assert receipt.zero_target_count == 1
    assert receipt.target_count == 3


def test_exhaustive_attention_builder_is_fixture_bounded() -> None:
    length = 200
    positions = torch.arange(length, dtype=torch.int64)[None, :]
    mask = torch.ones(1, length, dtype=torch.bool)
    native = _native_causal_mask(positions, positions, heads=4)
    assert int(native.sum()) > MAX_EXHAUSTIVE_ATTENTION_ROUTE_ROWS
    with pytest.raises(ValueError, match="small-fixture-only"):
        build_attention_route_chart_surface(
            mode_id="mode.mutation.1",
            block_ordinal=0,
            native_abi_sha256=_sha(1),
            comparison_abi_sha256=_sha(2),
            source_state_sha256=_sha(3),
            execution_fingerprint_sha256=_sha(4),
            score_objective_receipt_sha256=_sha(8),
            example_ids=("long.safe",),
            context_fingerprints=(_sha(5),),
            query_valid_mask=mask,
            key_valid_mask=mask,
            query_logical_positions=positions,
            key_logical_positions=positions,
            native_route_mask=native,
            target_carrier_values=torch.ones(1, length, HIDDEN),
            target_carrier_score_gradients=torch.ones(1, length, HIDDEN),
            attention_operator_output=torch.ones(1, length, HIDDEN),
            attention_operator_output_score_gradients=torch.ones(1, length, HIDDEN),
            route_operator_output_values=torch.ones(1, 1, 1, 1, 1),
            postnorm_state_sha256=_sha(6),
            postnorm_config_sha256=_sha(7),
            rope=_rope(),
            window_size=None,
        )


def _lexical_fixture():
    lineage = TiedLexicalStateLineage(
        tied_state_sha256=_sha(20),
        vocabulary_size=7,
        hidden_width=HIDDEN,
        embedding_scale=HIDDEN**0.5,
    )
    valid = torch.tensor([[True, True]])
    positions = torch.tensor([[0, 1]], dtype=torch.int64)
    generator = torch.Generator().manual_seed(91)
    values = torch.randn(1, 2, HIDDEN, generator=generator)
    gradients = torch.randn(1, 2, HIDDEN, generator=generator)
    common = {
        "native_abi_sha256": _sha(21),
        "comparison_abi_sha256": _sha(22),
        "static_comparison_adapter_receipt_sha256": _sha(20),
        "execution_fingerprint_sha256": _sha(23),
        "lineage": lineage,
        "example_ids": ("lex.safe",),
        "context_fingerprints": (_sha(24),),
        "valid_mask": valid,
        "logical_positions": positions,
    }
    return lineage, common, values, gradients


def test_lexical_ingress_uses_generalized_contribution_and_normalized_energy() -> None:
    lineage, common, values, gradients = _lexical_fixture()
    ingress = build_lexical_ingress_chart_surface(
        **common,
        score_objective_receipt_sha256=_sha(47),
        ingress_lookup_receipt_sha256=_sha(48),
        ingress_carrier_values=values,
        ingress_carrier_score_gradients=gradients,
    )
    assert isinstance(ingress, LexicalIngressChartSurface)
    assert ingress.lineage.sha256 == lineage.sha256
    assert ingress.carrier.mapping_receipt_sha256 == _sha(20)
    assert ingress.static_comparison_adapter_receipt_sha256 == _sha(20)
    assert ingress.panel_mapping_receipt_sha256 != _sha(20)
    assert torch.equal(ingress.carrier.coordinates, (values * gradients)[0])
    assert torch.isclose(
        ingress.carrier.normalized_activation_energy.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert torch.isclose(
        ingress.carrier.normalized_fisher_energy.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert LexicalIngressChartSurface.from_state_dict(ingress.state_dict()).sha256 == ingress.sha256


def test_selected_readout_pullback_matches_softcap_autograd_and_is_validation_only() -> None:
    lineage, common, terminal, _ = _lexical_fixture()
    generator = torch.Generator().manual_seed(92)
    weights = torch.randn(1, 3, HIDDEN, generator=generator)
    covector = torch.tensor([[[0.2, -0.1, -0.1], [-0.4, 0.1, 0.3]]])
    cap = 7.0
    analytic = pull_back_post_softcap_logit_covector(
        terminal_carrier=terminal,
        selected_readout_weight_rows=weights,
        post_softcap_logit_covector=covector,
        final_logit_softcap=cap,
    )
    primal = terminal.clone().requires_grad_(True)
    logits = torch.einsum("bsh,bch->bsc", primal, weights)
    post = cap * torch.tanh(logits / cap)
    (post * covector).sum().backward()
    assert torch.allclose(analytic, primal.grad, rtol=1e-5, atol=1e-6)

    egress = build_lexical_egress_chart_surface(
        **{**common, "native_abi_sha256": _sha(25)},
        terminal_carrier=terminal,
        selected_readout_weight_rows=weights,
        post_softcap_logit_covector=covector,
        readout_row_catalog_sha256=_sha(26),
        objective_receipt_sha256=_sha(27),
        terminal_norm_state_sha256=_sha(28),
        readout_state_sha256=_sha(29),
        final_logit_softcap=cap,
    )
    assert egress.pullback_kind == "validation_selected_logit_covector"
    assert egress.carrier.mapping_receipt_sha256 == _sha(20)
    assert egress.static_comparison_adapter_receipt_sha256 == _sha(20)
    assert egress.panel_mapping_receipt_sha256 != _sha(20)
    assert torch.equal(egress.carrier.coordinates, (terminal * analytic)[0])
    assert LexicalEgressChartSurface.from_state_dict(egress.state_dict()).sha256 == egress.sha256

    with pytest.raises(ValueError, match="centered"):
        build_lexical_egress_chart_surface(
            **{**common, "native_abi_sha256": _sha(25)},
            terminal_carrier=terminal,
            selected_readout_weight_rows=weights,
            post_softcap_logit_covector=torch.ones_like(covector),
            readout_row_catalog_sha256=_sha(26),
            objective_receipt_sha256=_sha(27),
            terminal_norm_state_sha256=_sha(28),
            readout_state_sha256=_sha(29),
            final_logit_softcap=cap,
        )


def test_direct_terminal_gradient_is_canonical_scalable_egress_path() -> None:
    lineage, common, terminal, gradient = _lexical_fixture()
    egress = build_lexical_egress_from_terminal_score_gradient(
        **{**common, "native_abi_sha256": _sha(30)},
        terminal_carrier=terminal,
        terminal_carrier_score_gradient=gradient,
        objective_receipt_sha256=_sha(31),
        terminal_norm_state_sha256=_sha(32),
        readout_state_sha256=_sha(33),
        readout_mapping_receipt_sha256=_sha(34),
        terminal_score_gradient_receipt_sha256=_sha(35),
        analytic_parity_receipt_sha256=_sha(36),
        final_logit_softcap=7.0,
    )
    assert egress.pullback_kind == "full_next_token_nll_direct_terminal_gradient"
    assert egress.carrier.mapping_receipt_sha256 == _sha(20)
    assert egress.static_comparison_adapter_receipt_sha256 == _sha(20)
    assert egress.panel_mapping_receipt_sha256 != _sha(20)
    assert egress.lineage.sha256 == lineage.sha256
    assert torch.equal(egress.carrier.coordinates, (terminal * gradient)[0])
    state = egress.state_dict()
    prohibited_keys = {"logits", "raw_logits", "full_vocabulary_logits", "input_ids"}
    assert prohibited_keys.isdisjoint(_walk_keys(state))
    assert all(
        not (isinstance(value, torch.Tensor) and value.ndim > 2)
        for value in _walk_values(state)
    )
    assert LexicalEgressChartSurface.from_state_dict(state).sha256 == egress.sha256


def _walk_values(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _walk_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            result.add(str(key))
            result.update(_walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(_walk_keys(item))
    return result


def _nontensor_skeleton(value: object) -> object:
    if isinstance(value, dict):
        return {key: _nontensor_skeleton(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nontensor_skeleton(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    return value


def test_lexical_endpoints_share_state_lineage_but_have_distinct_native_abis() -> None:
    lineage, common, values, gradients = _lexical_fixture()
    ingress = build_lexical_ingress_chart_surface(
        **common,
        score_objective_receipt_sha256=_sha(47),
        ingress_lookup_receipt_sha256=_sha(48),
        ingress_carrier_values=values,
        ingress_carrier_score_gradients=gradients,
    )
    egress = build_lexical_egress_from_terminal_score_gradient(
        **{**common, "native_abi_sha256": _sha(40)},
        terminal_carrier=values,
        terminal_carrier_score_gradient=gradients,
        objective_receipt_sha256=_sha(41),
        terminal_norm_state_sha256=_sha(42),
        readout_state_sha256=_sha(43),
        readout_mapping_receipt_sha256=_sha(44),
        terminal_score_gradient_receipt_sha256=_sha(45),
        analytic_parity_receipt_sha256=_sha(46),
        final_logit_softcap=None,
    )
    assert ingress.lineage.tied_state_sha256 == egress.lineage.tied_state_sha256
    assert ingress.lineage.sha256 == egress.lineage.sha256
    assert ingress.carrier.native_abi_sha256 != egress.carrier.native_abi_sha256
    assert ingress.carrier.comparison_abi_sha256 == egress.carrier.comparison_abi_sha256
    assert ingress.carrier.row_keys == egress.carrier.row_keys


def test_lexical_static_adapter_receipts_bind_across_distinct_panels() -> None:
    lineage, common, values, gradients = _lexical_fixture()
    comparison = canonical_carrier_comparison_abi()
    ingress_native = NativeChartABI(
        abi_kind=LEXICAL_INGRESS_NATIVE_ABI,
        abi_name="lexical_ingress.lookup_to_h640.v1",
        abi_version=1,
        input_signature="token_identity.lookup",
        output_signature="initial_carrier.float.h640",
        context_signature="aligned_prediction_query_positions",
    )
    ingress_static = _sha(80)
    ingress_common = {
        **common,
        "native_abi_sha256": ingress_native.sha256,
        "comparison_abi_sha256": comparison.sha256,
        "static_comparison_adapter_receipt_sha256": ingress_static,
    }
    ingress_a = build_lexical_ingress_chart_surface(
        **ingress_common,
        score_objective_receipt_sha256=_sha(81),
        ingress_lookup_receipt_sha256=_sha(82),
        ingress_carrier_values=values,
        ingress_carrier_score_gradients=gradients,
    )
    ingress_b = build_lexical_ingress_chart_surface(
        **{
            **ingress_common,
            "execution_fingerprint_sha256": _sha(83),
            "context_fingerprints": (_sha(84),),
        },
        score_objective_receipt_sha256=_sha(85),
        ingress_lookup_receipt_sha256=_sha(86),
        ingress_carrier_values=values + 1.0,
        ingress_carrier_score_gradients=gradients * 2.0,
    )
    assert ingress_a.carrier.mapping_receipt_sha256 == ingress_static
    assert ingress_b.carrier.mapping_receipt_sha256 == ingress_static
    assert (
        ingress_a.panel_mapping_receipt_sha256
        != ingress_b.panel_mapping_receipt_sha256
    )
    with pytest.raises(ValueError, match="static comparison adapter"):
        replace(
            ingress_a,
            static_comparison_adapter_receipt_sha256=(
                ingress_a.panel_mapping_receipt_sha256
            ),
        )
    ingress_spec = ChartModeSpec(
        mode_id="mode.lexical.ingress",
        mode_kind="lexical_ingress",
        state_fingerprint_sha256=lineage.tied_state_sha256,
        event_id="lexical.ingress",
        event_index=0,
        context_site_id="lexical.ingress.output",
        contribution_site_id="lexical.ingress.output",
        site_placement_sha256=_sha(87),
        native_abi=ingress_native,
        comparison_abi=comparison,
        comparison_adapter_kind=EMBEDDING_CARRIER_H640_ADAPTER,
        comparison_adapter_receipt_sha256=ingress_static,
    )
    assert bind_carrier_signature_to_chart_mode_spec(
        ingress_a.carrier,
        ingress_spec,
    ) is ingress_a.carrier
    assert bind_carrier_signature_to_chart_mode_spec(
        ingress_b.carrier,
        ingress_spec,
    ) is ingress_b.carrier
    with pytest.raises(ValueError, match="adapter receipt"):
        bind_carrier_signature_to_chart_mode_spec(
            ingress_a.carrier,
            replace(
                ingress_spec,
                comparison_adapter_receipt_sha256=(
                    ingress_a.panel_mapping_receipt_sha256
                ),
            ),
        )

    egress_native = NativeChartABI(
        abi_kind=LEXICAL_READOUT_NATIVE_ABI,
        abi_name="lexical_egress.tied_vocab_softcap_readout.v1",
        abi_version=1,
        input_signature="terminal_carrier.float.h640",
        output_signature="summed_next_token_nll.pullback.float.h640",
        context_signature="tied_readout.softcap.aligned_targets",
    )
    egress_static = _sha(88)
    egress_common = {
        **common,
        "native_abi_sha256": egress_native.sha256,
        "comparison_abi_sha256": comparison.sha256,
        "static_comparison_adapter_receipt_sha256": egress_static,
    }
    egress_a = build_lexical_egress_from_terminal_score_gradient(
        **egress_common,
        terminal_carrier=values,
        terminal_carrier_score_gradient=gradients,
        objective_receipt_sha256=_sha(89),
        terminal_norm_state_sha256=_sha(90),
        readout_state_sha256=_sha(91),
        readout_mapping_receipt_sha256=_sha(92),
        terminal_score_gradient_receipt_sha256=_sha(93),
        analytic_parity_receipt_sha256=_sha(94),
        final_logit_softcap=30.0,
    )
    egress_b = build_lexical_egress_from_terminal_score_gradient(
        **{
            **egress_common,
            "execution_fingerprint_sha256": _sha(95),
            "context_fingerprints": (_sha(96),),
        },
        terminal_carrier=values + 1.0,
        terminal_carrier_score_gradient=gradients * 2.0,
        objective_receipt_sha256=_sha(97),
        terminal_norm_state_sha256=_sha(90),
        readout_state_sha256=_sha(91),
        readout_mapping_receipt_sha256=_sha(92),
        terminal_score_gradient_receipt_sha256=_sha(98),
        analytic_parity_receipt_sha256=_sha(99),
        final_logit_softcap=30.0,
    )
    assert egress_a.carrier.mapping_receipt_sha256 == egress_static
    assert egress_b.carrier.mapping_receipt_sha256 == egress_static
    assert (
        egress_a.panel_mapping_receipt_sha256
        != egress_b.panel_mapping_receipt_sha256
    )
    with pytest.raises(ValueError, match="static comparison adapter"):
        replace(
            egress_a,
            static_comparison_adapter_receipt_sha256=(
                egress_a.panel_mapping_receipt_sha256
            ),
        )
    egress_spec = ChartModeSpec(
        mode_id="mode.lexical.egress",
        mode_kind="lexical_egress",
        state_fingerprint_sha256=lineage.tied_state_sha256,
        event_id="lexical.egress",
        event_index=38,
        context_site_id="terminal.rmsnorm.output",
        contribution_site_id="terminal.rmsnorm.output",
        site_placement_sha256=_sha(100),
        native_abi=egress_native,
        comparison_abi=comparison,
        comparison_adapter_kind=SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
        comparison_adapter_receipt_sha256=egress_static,
    )
    assert bind_carrier_signature_to_chart_mode_spec(
        egress_a.carrier,
        egress_spec,
    ) is egress_a.carrier
    assert bind_carrier_signature_to_chart_mode_spec(
        egress_b.carrier,
        egress_spec,
    ) is egress_b.carrier
    with pytest.raises(ValueError, match="adapter receipt"):
        bind_carrier_signature_to_chart_mode_spec(
            egress_b.carrier,
            replace(
                egress_spec,
                comparison_adapter_receipt_sha256=(
                    egress_b.panel_mapping_receipt_sha256
                ),
            ),
        )


def test_common_signature_strict_hash_and_safety_markers_reject_tampering() -> None:
    _, common, values, gradients = _lexical_fixture()
    ingress = build_lexical_ingress_chart_surface(
        **common,
        score_objective_receipt_sha256=_sha(47),
        ingress_lookup_receipt_sha256=_sha(48),
        ingress_carrier_values=values,
        ingress_carrier_score_gradients=gradients,
    )
    state = ingress.carrier.state_dict()
    assert CarrierChartSignature.from_state_dict(state).sha256 == ingress.carrier.sha256
    tampered = deepcopy(state)
    tampered["carrier_values"][0, 0] += 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="coordinates must be"):
        CarrierChartSignature.from_state_dict(tampered)
    tampered = deepcopy(state)
    tampered["contains_token_ids"] = True
    with pytest.raises(ValueError, match="contract drifted"):
        CarrierChartSignature.from_state_dict(tampered)


def test_carrier_signature_binds_exactly_to_graph_core_mode_spec() -> None:
    _, kwargs = _attention_fixture()
    native = NativeChartABI(
        abi_kind=CAUSAL_SEQUENCE_NATIVE_ABI,
        abi_name="attention.residual_h640.causal_sequence.v1",
        abi_version=1,
        input_signature="raw_carrier_sequence.float.h640",
        output_signature="postnorm_delta.float.h640",
        context_signature="causal_mask.positions.rope.exact_context",
    )
    comparison = canonical_carrier_comparison_abi()
    kwargs["native_abi_sha256"] = native.sha256
    kwargs["comparison_abi_sha256"] = comparison.sha256
    surface = build_attention_route_chart_surface(**kwargs)
    mode_spec = ChartModeSpec(
        mode_id="mode.mutation.1",
        mode_kind="attention",
        state_fingerprint_sha256=surface.carrier.source_state_sha256,
        event_id="mutation.1",
        event_index=1,
        context_site_id="mutation.1.input",
        contribution_site_id="mutation.1.output",
        site_placement_sha256=_sha(70),
        native_abi=native,
        comparison_abi=comparison,
        comparison_adapter_kind=DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        comparison_adapter_receipt_sha256=surface.carrier.mapping_receipt_sha256,
    )
    assert (
        bind_carrier_signature_to_chart_mode_spec(surface.carrier, mode_spec)
        is surface.carrier
    )
    with pytest.raises(ValueError, match="adapter receipt"):
        bind_carrier_signature_to_chart_mode_spec(
            surface.carrier,
            replace(mode_spec, comparison_adapter_receipt_sha256=_sha(71)),
        )
