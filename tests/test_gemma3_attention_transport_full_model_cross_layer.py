from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from fisher_graph.gemma3_attention_transport_full_model_cross_layer import (
    DYNAMIC_ROUTER_FEATURE_NAMES,
    FIXED_INVOCATION_FEATURE_NAMES,
    GLOBAL_ATTENTION_INVOCATION_IDS,
    INVOCATION_IDS,
    ROUTER_FEATURE_NAMES,
    FullModelCrossLayerAttentionTransport,
    SharedCrossLayerGeneratorBank,
    SharedTop2CrossLayerRouter,
    check_full_model_cross_layer_attention_transport_state_dict,
    fixed_invocation_coordinates,
    fixed_invocation_encoding,
    full_model_cross_layer_attention_transport_from_state_dict,
    shared_cross_layer_generator_bank_from_state_dict,
)


def _factors() -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(4, dtype=torch.float64)
    left = torch.stack(
        (
            identity[:, (0, 1)],
            identity[:, (1, 2)],
            identity[:, (2, 3)],
        )
    )
    right = torch.stack(
        (
            identity[:, (1, 0)],
            identity[:, (2, 1)],
            identity[:, (3, 2)],
        )
    )
    return left, right


def _bank() -> SharedCrossLayerGeneratorBank:
    left, right = _factors()
    raw_offsets = torch.arange(36, dtype=torch.float64).reshape(6, 3, 2) / 100.0
    site_head_coefficients = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.1, 0.7, 0.3]],
            [[0.8, 0.1, 0.2], [0.2, 0.9, -0.2]],
            [[0.7, -0.2, 0.4], [0.4, 0.8, 0.1]],
            [[0.6, 0.3, 0.1], [0.5, 0.7, -0.1]],
            [[0.5, 0.4, -0.2], [0.6, 0.6, 0.2]],
            [[0.4, -0.1, 0.5], [0.7, 0.5, 0.3]],
        ],
        dtype=torch.float64,
    )
    return SharedCrossLayerGeneratorBank(
        left_factors=left,
        base_singular_diagonal=torch.tensor(
            [[1.0, 0.8], [0.7, 0.5], [0.6, 0.4]],
            dtype=torch.float64,
        ),
        depth_singular_slopes=torch.tensor(
            [[0.10, -0.05], [0.04, 0.02], [-0.03, 0.06]],
            dtype=torch.float64,
        ),
        site_singular_offsets=raw_offsets,
        right_factors=right,
        site_head_coefficients=site_head_coefficients,
    )


def _router(bank: SharedCrossLayerGeneratorBank) -> SharedTop2CrossLayerRouter:
    feature_count = len(ROUTER_FEATURE_NAMES)
    selection_weight = torch.zeros(3, feature_count, dtype=torch.float64)
    selection_weight[0, 0] = 0.2
    selection_weight[1, -1] = 0.1
    selection_weight[2, -2] = -0.1
    amplitude_weight = torch.zeros_like(selection_weight)
    amplitude_weight[0, -3] = 0.05
    amplitude_weight[1, -2] = -0.02
    return SharedTop2CrossLayerRouter(
        generator_bank_sha256=bank.artifact_sha256,
        selection_weight=selection_weight,
        selection_bias=torch.tensor([3.0, 2.0, -2.0], dtype=torch.float64),
        amplitude_weight=amplitude_weight,
        amplitude_bias=torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64),
    )


def _transport() -> FullModelCrossLayerAttentionTransport:
    bank = _bank()
    return FullModelCrossLayerAttentionTransport(bank=bank, router=_router(bank))


def _head_contexts() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 2.0, -1.0, 0.5], [0.5, -0.5, 1.5, 2.0]],
            [[-0.5, 1.0, 0.2, 0.8], [1.2, 0.4, -0.3, 0.1]],
        ],
        dtype=torch.float32,
    )


def test_fixed_encoding_covers_the_exact_3_by_6_catalog_without_parameters() -> None:
    assert INVOCATION_IDS == tuple(
        f"mutation.{2 * index + 1}" for index in range(18)
    )
    encodings = torch.stack(
        tuple(fixed_invocation_encoding(value) for value in INVOCATION_IDS)
    )
    assert encodings.shape == (18, len(FIXED_INVOCATION_FEATURE_NAMES))
    assert torch.unique(encodings, dim=0).shape[0] == 18

    for ordinal, invocation_id in enumerate(INVOCATION_IDS):
        coordinates = fixed_invocation_coordinates(invocation_id)
        block, site = divmod(ordinal, 6)
        assert coordinates.ordinal == ordinal
        assert coordinates.macroblock_index == block
        assert coordinates.site_index == site
        expected_depth = float((2 * ordinal - 17) / 17)
        assert coordinates.normalized_depth == expected_depth
        assert coordinates.is_global_attention is (site == 5)
        encoding = encodings[ordinal]
        assert encoding[0].item() == expected_depth
        assert encoding[1:7].sum().item() == 1.0
        assert encoding[1 + site].item() == 1.0
        assert encoding[-2:].tolist() == (
            [0.0, 1.0] if site == 5 else [1.0, 0.0]
        )
    assert tuple(
        value
        for value in INVOCATION_IDS
        if fixed_invocation_coordinates(value).is_global_attention
    ) == GLOBAL_ATTENTION_INVOCATION_IDS


def test_bank_centers_gamma_and_shares_site_head_mixing_across_depth() -> None:
    bank = _bank()
    assert torch.allclose(
        bank.site_singular_offsets.sum(dim=0),
        torch.zeros(3, 2, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    contexts = _head_contexts()
    early = bank.execute(invocation_id="mutation.1", head_contexts=contexts)
    middle = bank.execute(invocation_id="mutation.13", head_contexts=contexts)
    late = bank.execute(invocation_id="mutation.25", head_contexts=contexts)

    # All three have site zero, so c[site, head, generator] is identical.
    assert torch.equal(early.contextual_inputs, middle.contextual_inputs)
    assert torch.equal(middle.contextual_inputs, late.contextual_inputs)
    site_gamma = bank.site_singular_offsets[0]
    assert torch.allclose(
        early.effective_singular_diagonal,
        bank.base_singular_diagonal
        + fixed_invocation_coordinates("mutation.1").normalized_depth
        * bank.depth_singular_slopes
        + site_gamma,
    )
    assert torch.allclose(
        middle.effective_singular_diagonal,
        bank.base_singular_diagonal
        + fixed_invocation_coordinates("mutation.13").normalized_depth
        * bank.depth_singular_slopes
        + site_gamma,
    )
    assert torch.allclose(
        late.effective_singular_diagonal,
        bank.base_singular_diagonal
        + fixed_invocation_coordinates("mutation.25").normalized_depth
        * bank.depth_singular_slopes
        + site_gamma,
    )
    assert torch.allclose(
        middle.effective_singular_diagonal,
        (
            early.effective_singular_diagonal
            + late.effective_singular_diagonal
        )
        / 2.0,
    )


def test_synthetic_forward_has_exact_top2_and_closes_the_output_equation() -> None:
    transport = _transport()
    contexts = _head_contexts()
    for invocation_id in INVOCATION_IDS:
        result = transport(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        router = result.router_execution
        assert result.bank_execution.contextual_inputs.shape == (2, 3, 4)
        assert result.bank_execution.latent_coordinates.shape == (2, 3, 2)
        assert result.bank_execution.candidate_outputs.shape == (2, 3, 4)
        assert router.dynamic_features.shape == (
            2,
            3,
            len(DYNAMIC_ROUTER_FEATURE_NAMES),
        )
        assert router.router_features.shape == (2, 3, len(ROUTER_FEATURE_NAMES))
        assert router.active_mask.sum(dim=-1).tolist() == [2, 2]
        assert torch.equal(
            router.sparse_selection_probabilities == 0.0,
            ~router.active_mask,
        )
        assert torch.allclose(
            router.sparse_selection_probabilities.sum(dim=-1),
            torch.ones(2, dtype=torch.float64),
        )
        assert torch.allclose(
            router.generator_coefficients,
            router.sparse_selection_probabilities * router.signed_amplitudes,
        )
        expected = torch.einsum(
            "nk,nkh->nh",
            router.generator_coefficients,
            result.bank_execution.candidate_outputs,
        )
        assert result.operator_outputs.dtype == torch.float64
        assert result.operator_outputs.shape == (2, 4)
        assert torch.equal(result.operator_outputs, expected)


def test_top2_ties_are_deterministically_resolved_by_generator_ordinal() -> None:
    bank = _bank()
    router = SharedTop2CrossLayerRouter(
        generator_bank_sha256=bank.artifact_sha256,
        selection_weight=torch.zeros(3, len(ROUTER_FEATURE_NAMES)),
        selection_bias=torch.zeros(3),
        amplitude_weight=torch.zeros(3, len(ROUTER_FEATURE_NAMES)),
        amplitude_bias=torch.ones(3),
    )
    execution = router.execute(
        invocation_id="mutation.1",
        dynamic_features=torch.zeros(4, 3, len(DYNAMIC_ROUTER_FEATURE_NAMES)),
    )
    assert execution.active_mask.tolist() == [[True, True, False]] * 4


def test_parameter_accounting_includes_site_mixing_and_has_no_layer_table() -> None:
    transport = _transport()
    without_native = transport.parameter_accounting()
    assert without_native.generator_factor_parameters == 2 * 3 * 4 * 2
    assert without_native.singular_diagonal_parameters == (1 + 1 + 6) * 3 * 2
    assert without_native.site_head_mixing_parameters == 6 * 2 * 3
    assert without_native.bank_parameters == 132
    assert without_native.router_parameters == 2 * 3 * (len(ROUTER_FEATURE_NAMES) + 1)
    assert without_native.fixed_encoding_parameters == 0
    assert without_native.free_layer_id_parameters == 0
    assert without_native.compiled_parameters == 210
    assert without_native.native_v_o_parameter_count is None
    assert without_native.parameter_reduction_fraction is None
    assert without_native.parameter_compression_eligible is None

    with_native = transport.parameter_accounting(native_v_o_parameter_count=1_000)
    assert with_native.parameter_reduction_fraction == pytest.approx(0.79)
    assert with_native.parameter_compression_eligible is True

    state = transport.state_dict()
    assert state["invocation_ids"] == list(INVOCATION_IDS)
    bank_state = state["bank"]
    router_state = state["router"]
    assert isinstance(bank_state, dict)
    assert isinstance(router_state, dict)
    assert bank_state["site_head_coefficients"].shape[0] == 6
    assert bank_state["site_singular_offsets"].shape[0] == 6
    assert router_state["selection_weight"].shape[0] == 3
    assert all(
        not (isinstance(value, torch.Tensor) and value.shape[0] == 18)
        for value in (*bank_state.values(), *router_state.values())
        if not (isinstance(value, torch.Tensor) and value.ndim == 0)
    )


def test_state_round_trip_replays_exactly_and_checker_reconstructs() -> None:
    transport = _transport()
    state = transport.state_dict()
    loaded = full_model_cross_layer_attention_transport_from_state_dict(state)
    checked = check_full_model_cross_layer_attention_transport_state_dict(state)
    assert loaded.artifact_sha256 == transport.artifact_sha256
    assert checked.artifact_sha256 == transport.artifact_sha256
    expected = transport.forward(
        invocation_id="mutation.23",
        head_contexts=_head_contexts(),
    )
    actual = loaded.forward(
        invocation_id="mutation.23",
        head_contexts=_head_contexts(),
    )
    assert torch.equal(actual.operator_outputs, expected.operator_outputs)
    assert torch.equal(
        actual.router_execution.active_mask,
        expected.router_execution.active_mask,
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "artifact_hash",
        "bank_tensor",
        "router_tensor",
        "bank_binding",
        "gamma_gauge",
        "safety_flag",
        "catalog",
        "extra_field",
    ),
)
def test_checker_rejects_state_tampering(tamper: str) -> None:
    state = deepcopy(_transport().state_dict())
    bank = state["bank"]
    router = state["router"]
    assert isinstance(bank, dict)
    assert isinstance(router, dict)
    if tamper == "artifact_hash":
        state["artifact_sha256"] = "0" * 64
    elif tamper == "bank_tensor":
        bank["base_singular_diagonal"][0, 0] += 0.1
    elif tamper == "router_tensor":
        router["selection_bias"][0] += 0.1
    elif tamper == "bank_binding":
        router["generator_bank_sha256"] = "0" * 64
    elif tamper == "gamma_gauge":
        bank["site_singular_offsets"][0, 0, 0] += 0.1
    elif tamper == "safety_flag":
        state["contains_teacher_outputs_or_gradients"] = True
    elif tamper == "catalog":
        state["invocation_ids"] = list(reversed(INVOCATION_IDS))
    else:
        state["unexpected"] = None
    with pytest.raises(ValueError):
        check_full_model_cross_layer_attention_transport_state_dict(state)


def test_public_artifacts_clone_inputs_and_fail_closed_after_internal_mutation() -> None:
    left, right = _factors()
    base = torch.ones(3, 2, dtype=torch.float64)
    slopes = torch.zeros(3, 2, dtype=torch.float64)
    offsets = torch.zeros(6, 3, 2, dtype=torch.float64)
    head_coefficients = torch.ones(6, 2, 3, dtype=torch.float64)
    bank = SharedCrossLayerGeneratorBank(
        left_factors=left,
        base_singular_diagonal=base,
        depth_singular_slopes=slopes,
        site_singular_offsets=offsets,
        right_factors=right,
        site_head_coefficients=head_coefficients,
    )
    sealed_hash = bank.artifact_sha256
    left.zero_()
    base.zero_()
    head_coefficients.zero_()
    assert bank.artifact_sha256 == sealed_hash
    assert bool((bank.left_factors != 0.0).any())
    assert bool((bank.base_singular_diagonal != 0.0).any())
    assert bool((bank.site_head_coefficients != 0.0).any())

    bank.left_factors[0, 0, 0] += 0.25
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = bank.artifact_sha256

    intact_bank = _bank()
    router = _router(intact_bank)
    router.selection_bias[0] += 0.25
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = router.artifact_sha256


def test_bank_loader_requires_canonical_zero_sum_gamma() -> None:
    state = _bank().state_dict()
    state["site_singular_offsets"][0, 0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="zero-sum"):
        shared_cross_layer_generator_bank_from_state_dict(state)


def test_forward_rejects_unknown_invocations_bad_head_geometry_and_nonfinite() -> None:
    transport = _transport()
    with pytest.raises(ValueError, match="fixed 18-site catalog"):
        transport.forward(invocation_id="mutation.37", head_contexts=_head_contexts())
    with pytest.raises(ValueError, match="query-head count"):
        transport.forward(
            invocation_id="mutation.1",
            head_contexts=torch.zeros(2, 3, 4),
        )
    nonfinite = _head_contexts()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        transport.forward(invocation_id="mutation.1", head_contexts=nonfinite)


def test_constructor_rejects_non_top2_router_and_cross_bank_binding() -> None:
    bank = _bank()
    with pytest.raises(ValueError, match="Top-2"):
        SharedTop2CrossLayerRouter(
            generator_bank_sha256=bank.artifact_sha256,
            selection_weight=torch.zeros(3, len(ROUTER_FEATURE_NAMES)),
            selection_bias=torch.zeros(3),
            amplitude_weight=torch.zeros(3, len(ROUTER_FEATURE_NAMES)),
            amplitude_bias=torch.zeros(3),
            active_count=1,
        )
    other_bank = SharedCrossLayerGeneratorBank(
        left_factors=bank.left_factors,
        base_singular_diagonal=bank.base_singular_diagonal + 0.1,
        depth_singular_slopes=bank.depth_singular_slopes,
        site_singular_offsets=bank.site_singular_offsets,
        right_factors=bank.right_factors,
        site_head_coefficients=bank.site_head_coefficients,
    )
    with pytest.raises(ValueError, match="not bound"):
        FullModelCrossLayerAttentionTransport(
            bank=other_bank,
            router=_router(bank),
        )
