from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fisher_graph.gemma3_cross_layer_chart_transport_executor import (
    CrossLayerChartMessageSession,
    CrossLayerChartTransportEdge,
    Gemma3CrossLayerChartGraphExecutor,
    Gemma3FactorizedChartGeneratorLayer,
)
from fisher_graph.gemma3_cross_layer_chart_transport_experiment import (
    FrozenChartTransportStage,
    prepare_cross_layer_chart_transport_experiment,
    sequentially_freeze_cross_layer_chart_stack,
    sequentially_freeze_cross_layer_chart_transport_stack,
)
from fisher_graph.gemma3_cross_layer_chart_transport_protocol import (
    PINNED_DEVELOPMENT_PANEL_PATH,
    PINNED_DEVELOPMENT_PANEL_SHA256,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    CrossLayerChartTransportProtocol,
    build_cross_layer_chart_transport_protocol,
    compiled_chart_prefix_catalog_sha256,
    compiled_chart_transport_stage_sha256,
    require_aligned_transport_rows,
)


_MODEL = "1" * 64
_ROWS = "2" * 64


def _layer(
    ordinal: int,
    *,
    prefix_stages: tuple[str, ...] = (),
    centers: torch.Tensor | None = None,
    local_encoders: torch.Tensor | None = None,
    chart_decoders: torch.Tensor | None = None,
) -> Gemma3FactorizedChartGeneratorLayer:
    return Gemma3FactorizedChartGeneratorLayer(
        affine_weight=torch.tensor(
            [[0.2, 0.0], [0.0, 0.3], [0.1, -0.1]]
        ),
        affine_bias=torch.tensor([0.1, -0.1, 0.2]),
        mode_decoder_weight=torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]
        ),
        chart_centers=(
            torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
            if centers is None
            else centers
        ),
        chart_bases=torch.tensor(
            [[[1.0], [0.0]], [[1.0], [0.0]]]
        ),
        chart_distance_scales=torch.ones(2),
        local_generator_encoders=(
            torch.tensor([[[0.2], [0.5]], [[-0.1], [0.25]]])
            if local_encoders is None
            else local_encoders
        ),
        chart_mode_decoders=(
            torch.tensor(
                [
                    [[1.0, 0.0, 0.25]],
                    [[0.0, 1.0, -0.25]],
                ]
            )
            if chart_decoders is None
            else chart_decoders
        ),
        chart_temperature=1.0,
        model_fingerprint=_MODEL,
        layer_ordinal=ordinal,
        fit_row_identity_sha256=_ROWS,
        compiled_prefix_catalog_sha256=(
            compiled_chart_prefix_catalog_sha256(prefix_stages)
        ),
    )


def _edge(
    source: Gemma3FactorizedChartGeneratorLayer,
    target: Gemma3FactorizedChartGeneratorLayer,
    *,
    source_chart: int = 0,
    target_chart: int = 1,
    scale: float = 2.0,
) -> CrossLayerChartTransportEdge:
    return CrossLayerChartTransportEdge(
        source_layer_ordinal=source.layer_ordinal,
        target_layer_ordinal=target.layer_ordinal,
        source_chart_index=source_chart,
        target_chart_index=target_chart,
        transport_weight=torch.tensor([[scale]]),
        model_fingerprint=_MODEL,
        fit_row_identity_sha256=_ROWS,
        source_layer_artifact_sha256=source.artifact_sha256,
        target_layer_artifact_sha256=target.artifact_sha256,
    )


def _execute_two_layers(
    source: Gemma3FactorizedChartGeneratorLayer,
    target: Gemma3FactorizedChartGeneratorLayer,
    edge: CrossLayerChartTransportEdge,
    *,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])
    valid = torch.ones(2, dtype=torch.bool)
    session = CrossLayerChartMessageSession(
        edges=(edge,),
        valid_positions=valid,
        condition=condition,
    )
    source_execution = source.execute_with_incoming(inputs)
    source_latents = source_execution.emitted_generator_latents
    session.publish(0, source_latents)
    incoming = session.incoming_for(target_layer=target, edges=(edge,))
    target_execution = target.execute_with_incoming(inputs, incoming)
    session.publish(1, target_execution.emitted_generator_latents)
    session.finish(expected_layers=(0, 1))
    return target_execution.output, incoming


def test_transport_matches_manual_two_layer_generator_contraction() -> None:
    source = _layer(0)
    source_stage = compiled_chart_transport_stage_sha256(
        layer_ordinal=0,
        layer_artifact_sha256=source.artifact_sha256,
        incoming_edge_artifact_sha256s=(),
    )
    target = _layer(1, prefix_stages=(source_stage,))
    edge = _edge(source, target)
    output, incoming = _execute_two_layers(
        source,
        target,
        edge,
        condition="transport",
    )
    inputs = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])
    source_execution = source.execute_with_incoming(inputs)
    expected_incoming = torch.zeros(2, 2, 1)
    expected_incoming[:, 1, 0] = (
        source_execution.emitted_generator_latents[:, 0, 0] * 2.0
    )
    target_membership, target_coordinates, target_local = (
        target.encode_local_generator_latents(inputs)
    )
    emitted = target_membership[..., None] * (
        target_local + expected_incoming
    )
    states = target.affine_generators(inputs) + torch.einsum(
        "ncg,cgk->nk",
        emitted,
        target.chart_mode_decoders,
    )
    expected_output = target.mode_decoder(states)

    assert target_coordinates.shape == (2, 2, 1)
    assert torch.allclose(incoming, expected_incoming)
    assert torch.allclose(output, expected_output)


def test_zero_transport_is_exact_independent_factorized_chart_baseline() -> None:
    source = _layer(0)
    source_stage = compiled_chart_transport_stage_sha256(
        layer_ordinal=0,
        layer_artifact_sha256=source.artifact_sha256,
        incoming_edge_artifact_sha256s=(),
    )
    target = _layer(1, prefix_stages=(source_stage,))
    edge = _edge(source, target)
    output, incoming = _execute_two_layers(
        source,
        target,
        edge,
        condition="zero_transport",
    )
    inputs = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])
    assert torch.count_nonzero(incoming) == 0
    assert torch.equal(output, target(inputs))


def test_source_chart_permutation_with_edge_permutation_is_invariant() -> None:
    source = _layer(0)
    source_stage = compiled_chart_transport_stage_sha256(
        layer_ordinal=0,
        layer_artifact_sha256=source.artifact_sha256,
        incoming_edge_artifact_sha256s=(),
    )
    target = _layer(1, prefix_stages=(source_stage,))
    reference, _ = _execute_two_layers(
        source,
        target,
        _edge(source, target, source_chart=0),
        condition="transport",
    )
    permutation = torch.tensor([1, 0])
    permuted = _layer(
        0,
        centers=source.chart_centers[permutation],
        local_encoders=source.local_generator_encoders[permutation],
        chart_decoders=source.chart_mode_decoders[permutation],
    )
    observed, _ = _execute_two_layers(
        permuted,
        target,
        _edge(permuted, target, source_chart=1),
        condition="transport",
    )
    assert torch.allclose(observed, reference)


def test_transport_rejects_backward_duplicate_or_misaligned_edges() -> None:
    with pytest.raises(ValueError, match="later layer"):
        CrossLayerChartTransportEdge(
            source_layer_ordinal=1,
            target_layer_ordinal=0,
            source_chart_index=0,
            target_chart_index=0,
            transport_weight=torch.eye(1),
            model_fingerprint=_MODEL,
            fit_row_identity_sha256=_ROWS,
            source_layer_artifact_sha256="3" * 64,
            target_layer_artifact_sha256="4" * 64,
        )
    source = _layer(0)
    source_stage = compiled_chart_transport_stage_sha256(
        layer_ordinal=0,
        layer_artifact_sha256=source.artifact_sha256,
        incoming_edge_artifact_sha256s=(),
    )
    target = _layer(1, prefix_stages=(source_stage,))
    edge = _edge(source, target)

    class Adapter:
        layers = (object(), object())

        @staticmethod
        def model_fingerprint() -> str:
            return _MODEL

    with pytest.raises(ValueError, match="duplicate"):
        Gemma3CrossLayerChartGraphExecutor(
            Adapter(),
            layers={0: source, 1: target},
            edges=(edge, edge),
        )
    misaligned = CrossLayerChartTransportEdge(
        source_layer_ordinal=0,
        target_layer_ordinal=1,
        source_chart_index=0,
        target_chart_index=1,
        transport_weight=torch.eye(1),
        model_fingerprint=_MODEL,
        fit_row_identity_sha256="9" * 64,
        source_layer_artifact_sha256=source.artifact_sha256,
        target_layer_artifact_sha256=target.artifact_sha256,
    )
    with pytest.raises(ValueError, match="lineage"):
        Gemma3CrossLayerChartGraphExecutor(
            Adapter(),
            layers={0: source, 1: target},
            edges=(misaligned,),
        )


def test_target_before_source_fails_and_abort_clears_session() -> None:
    source = _layer(0)
    source_stage = compiled_chart_transport_stage_sha256(
        layer_ordinal=0,
        layer_artifact_sha256=source.artifact_sha256,
        incoming_edge_artifact_sha256s=(),
    )
    target = _layer(1, prefix_stages=(source_stage,))
    edge = _edge(source, target)
    session = CrossLayerChartMessageSession(
        edges=(edge,),
        valid_positions=torch.ones(2, dtype=torch.bool),
        condition="transport",
    )
    with pytest.raises(RuntimeError, match="before its source"):
        session.incoming_for(target_layer=target, edges=(edge,))
    session.abort()
    assert session.closed is True
    assert session._messages == {}


def test_fit_requires_exact_source_target_token_row_identity() -> None:
    rows = (("example-a", 0), ("example-a", 1), ("example-b", 0))
    digest = require_aligned_transport_rows(rows, rows)
    assert len(digest) == 64
    with pytest.raises(ValueError, match="not exactly aligned"):
        require_aligned_transport_rows(rows, tuple(reversed(rows)))
    with pytest.raises(ValueError, match="example/position pairs"):
        require_aligned_transport_rows((("example-a", 0, 1),), (("example-a", 0, 1),))


class _NativeTripwireMLP(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("native MLP projection executed")


class _NativePassthroughMLP(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _NativeTripwireMLP()


class _FakeGemmaBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((_FakeBlock(), _FakeBlock()))


class _FakeGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeGemmaBody()
        self.fail_after_layer_zero = False

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask
        assert use_cache is False
        assert return_dict is True
        hidden = self.model.layers[0].mlp(hidden_states)
        if self.fail_after_layer_zero:
            raise RuntimeError("injected model failure")
        hidden = self.model.layers[1].mlp(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeAdapter:
    def __init__(self) -> None:
        self.module = _FakeGemma()
        self.layers = (object(), object())

    @staticmethod
    def model_fingerprint() -> str:
        return _MODEL

    @staticmethod
    def prepare_sequence(
        model_inputs: dict[str, torch.Tensor],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            query_valid_mask=model_inputs["attention_mask"].bool()
        )


def test_full_executor_bypasses_native_restores_and_resets_after_failure() -> None:
    adapter = _FakeAdapter()
    source = _layer(0)
    source_stage = compiled_chart_transport_stage_sha256(
        layer_ordinal=0,
        layer_artifact_sha256=source.artifact_sha256,
        incoming_edge_artifact_sha256s=(),
    )
    target = _layer(1, prefix_stages=(source_stage,))
    edge = _edge(source, target)
    executor = Gemma3CrossLayerChartGraphExecutor(
        adapter,
        layers={0: source, 1: target},
        edges=(edge,),
    )
    original_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    inputs = {
        "hidden_states": torch.tensor(
            [[[-1.0, 0.5], [1.0, -0.5]]]
        ),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    adapter.module.fail_after_layer_zero = True
    with pytest.raises(RuntimeError, match="injected model failure"):
        executor.run(inputs)
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == original_mlps
    assert executor._active is False

    adapter.module.fail_after_layer_zero = False
    execution = executor.run(inputs)
    assert execution.layer_calls == (0, 1)
    assert execution.transport_edge_count == 1
    assert execution.peak_live_chart_packets == 1
    assert execution.peak_live_chart_packet_bytes == 16
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == original_mlps
    assert executor.stored_coefficient_count == 71
    assert executor.estimated_macs_per_token == 65


def test_partial_prefix_callback_captures_valid_packets_and_restores() -> None:
    adapter = _FakeAdapter()
    adapter.module.model.layers[1].mlp = _NativePassthroughMLP()
    source = _layer(0)
    executor = Gemma3CrossLayerChartGraphExecutor(
        adapter,
        layers={0: source},
        edges=(),
    )
    original_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    inputs = {
        "hidden_states": torch.tensor(
            [[[-1.0, 0.5], [0.25, 0.75], [1.0, -0.5]]]
        ),
        "attention_mask": torch.tensor([[1, 0, 1]]),
    }
    installed_during_callback: tuple[bool, bool] | None = None

    def callback() -> SimpleNamespace:
        nonlocal installed_during_callback
        installed_during_callback = (
            adapter.module.model.layers[0].mlp is not original_mlps[0],
            adapter.module.model.layers[1].mlp is original_mlps[1],
        )
        return adapter.module(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    model_output, receipt = executor.run_with_overlay(
        inputs,
        callback=callback,
        capture_emitted_layer_ordinals=(0,),
    )
    expected_execution = source.execute_with_incoming(inputs["hidden_states"])
    expected_packet = expected_execution.emitted_generator_latents[
        inputs["attention_mask"].bool()
    ]

    assert installed_during_callback == (True, True)
    assert torch.equal(model_output.last_hidden_state, expected_execution.output)
    assert receipt.layer_calls == (0,)
    assert receipt.transport_edge_count == 0
    assert receipt.peak_live_chart_packets == 0
    assert receipt.peak_live_chart_packet_bytes == 0
    assert tuple(receipt.captured_emitted_generator_latents) == (0,)
    captured = receipt.captured_emitted_generator_latents[0]
    assert captured.device.type == "cpu"
    assert captured.dtype == torch.float32
    assert captured.shape == (2, 2, 1)
    assert torch.equal(captured, expected_packet)
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == original_mlps


def test_prefix_callback_failure_discards_capture_and_resets_executor() -> None:
    adapter = _FakeAdapter()
    adapter.module.model.layers[1].mlp = _NativePassthroughMLP()
    source = _layer(0)
    executor = Gemma3CrossLayerChartGraphExecutor(
        adapter,
        layers={0: source},
        edges=(),
    )
    original_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    inputs = {
        "hidden_states": torch.tensor([[[-1.0, 0.5], [1.0, -0.5]]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    def fail_after_prefix() -> None:
        adapter.module.model.layers[0].mlp(inputs["hidden_states"])
        raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="injected callback failure"):
        executor.run_with_overlay(
            inputs,
            callback=fail_after_prefix,
            capture_emitted_layer_ordinals=(0,),
        )
    assert executor._active is False
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == original_mlps

    execution = executor.run(inputs)
    assert execution.layer_calls == (0,)
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == original_mlps


def test_prefix_callback_rejects_capture_outside_compiled_prefix() -> None:
    adapter = _FakeAdapter()
    source = _layer(0)
    executor = Gemma3CrossLayerChartGraphExecutor(
        adapter,
        layers={0: source},
        edges=(),
    )
    inputs = {
        "hidden_states": torch.tensor([[[-1.0, 0.5]]]),
        "attention_mask": torch.tensor([[1]]),
    }
    with pytest.raises(ValueError, match="compiled prefix layers"):
        executor.run_with_overlay(
            inputs,
            callback=lambda: None,
            capture_emitted_layer_ordinals=(1,),
        )


def test_protocol_freezes_nested_controls_resources_and_claim_boundary() -> None:
    protocol = CrossLayerChartTransportProtocol()
    state = protocol.state_dict()
    arm_ids = tuple(arm["arm_id"] for arm in state["arms"])
    assert arm_ids == (
        "native",
        "exact_all_mode_graph",
        "one_shot_independent_dense_chart_reproduction",
        "progressive_independent_factorized_chart",
        "progressive_factorized_chart_transport",
        "budget_matched_factorized_chart_transport",
        "transport_zero_edge_control",
        "transport_row_shuffled_negative_control",
    )
    assert protocol.local_factorized_coefficients_per_layer(4) == 66_080
    assert protocol.maximum_transport_coefficients_per_layer(4, 2) == 256
    assert state["claim_boundary"]["compression_success_claimed"] is False
    assert state["model_experiment_executed"] is False
    assert len(state["protocol_sha256"]) == 64


@dataclass(frozen=True)
class _Frozen:
    layer_ordinal: int
    artifact_sha256: str
    compiled_prefix_catalog_sha256: str


def test_sequential_setup_freezes_each_layer_into_the_next_prefix() -> None:
    observed: list[tuple[int, tuple[int, ...], str]] = []

    def capture(
        ordinal: int,
        prefix: tuple[_Frozen, ...],
        catalog: str,
    ) -> str:
        observed.append(
            (ordinal, tuple(layer.layer_ordinal for layer in prefix), catalog)
        )
        return f"capture-{ordinal}"

    def fit(
        ordinal: int,
        capture_value: str,
        prefix: tuple[_Frozen, ...],
        catalog: str,
    ) -> _Frozen:
        assert capture_value == f"capture-{ordinal}"
        assert tuple(layer.layer_ordinal for layer in prefix) == tuple(
            range(ordinal)
        )
        return _Frozen(ordinal, f"{ordinal + 10:064x}", catalog)

    frozen, receipts = sequentially_freeze_cross_layer_chart_stack(
        layer_count=3,
        capture_layer=capture,
        fit_and_freeze_layer=fit,
    )
    assert tuple(layer.layer_ordinal for layer in frozen) == (0, 1, 2)
    assert tuple(receipt.layer_ordinal for receipt in receipts) == (0, 1, 2)
    assert tuple(prefix for _, prefix, _ in observed) == ((), (0,), (0, 1))
    assert receipts[2].compiled_prefix_artifact_sha256s == (
        frozen[0].artifact_sha256,
        frozen[1].artifact_sha256,
    )


def test_prepare_writes_setup_only_protocol_without_model_fields(tmp_path) -> None:
    destination = tmp_path / "protocol.json"
    state = prepare_cross_layer_chart_transport_experiment(output=destination)
    loaded = build_cross_layer_chart_transport_protocol()
    assert state == loaded
    assert destination.exists()
    assert state["model_weights_loaded"] is False
    assert state["model_experiment_executed"] is False


def test_prefix_catalog_requires_exact_contiguous_order() -> None:
    assert len(compiled_chart_prefix_catalog_sha256(())) == 64
    with pytest.raises(ValueError, match="prefix lineage"):
        compiled_chart_prefix_catalog_sha256(("not-a-digest",))


@dataclass(frozen=True)
class _FrozenEdge:
    target_layer_ordinal: int
    artifact_sha256: str


def test_composite_stage_lineage_binds_inbound_edges_into_next_prefix() -> None:
    inbound = {
        0: (),
        1: (_FrozenEdge(1, "a" * 64),),
        2: (_FrozenEdge(2, "c" * 64), _FrozenEdge(2, "b" * 64)),
    }
    observed_catalogs: list[str] = []

    def capture(
        ordinal: int,
        prefix: tuple[FrozenChartTransportStage, ...],
        catalog: str,
    ) -> str:
        expected = compiled_chart_prefix_catalog_sha256(
            tuple(stage.stage_artifact_sha256 for stage in prefix)
        )
        assert catalog == expected
        observed_catalogs.append(catalog)
        return f"capture-{ordinal}"

    def fit(
        ordinal: int,
        capture_value: str,
        prefix: tuple[FrozenChartTransportStage, ...],
        catalog: str,
    ) -> tuple[_Frozen, tuple[_FrozenEdge, ...]]:
        assert capture_value == f"capture-{ordinal}"
        assert len(prefix) == ordinal
        return (
            _Frozen(ordinal, f"{ordinal + 20:064x}", catalog),
            inbound[ordinal],
        )

    stages, receipts = sequentially_freeze_cross_layer_chart_transport_stack(
        layer_count=3,
        capture_layer=capture,
        fit_and_freeze_stage=fit,
    )

    expected_stage_one = compiled_chart_transport_stage_sha256(
        layer_ordinal=1,
        layer_artifact_sha256=stages[1].layer.artifact_sha256,
        incoming_edge_artifact_sha256s=("a" * 64,),
    )
    assert stages[1].stage_artifact_sha256 == expected_stage_one
    assert receipts[1].incoming_edge_artifact_sha256s == ("a" * 64,)
    assert receipts[2].incoming_edge_artifact_sha256s == (
        "b" * 64,
        "c" * 64,
    )
    assert receipts[2].compiled_prefix_stage_sha256s == (
        stages[0].stage_artifact_sha256,
        stages[1].stage_artifact_sha256,
    )
    assert observed_catalogs[2] == compiled_chart_prefix_catalog_sha256(
        receipts[2].compiled_prefix_stage_sha256s
    )

    changed_stage_one = compiled_chart_transport_stage_sha256(
        layer_ordinal=1,
        layer_artifact_sha256=stages[1].layer.artifact_sha256,
        incoming_edge_artifact_sha256s=("d" * 64,),
    )
    assert changed_stage_one != stages[1].stage_artifact_sha256
    assert compiled_chart_prefix_catalog_sha256(
        (stages[0].stage_artifact_sha256, changed_stage_one)
    ) != observed_catalogs[2]


def test_composite_stage_lineage_rejects_invalid_inbound_edge_sets() -> None:
    layer_artifact = "a" * 64
    edge_artifact = "b" * 64
    with pytest.raises(ValueError, match="stage lineage"):
        compiled_chart_transport_stage_sha256(
            layer_ordinal=0,
            layer_artifact_sha256=layer_artifact,
            incoming_edge_artifact_sha256s=(edge_artifact,),
        )
    with pytest.raises(ValueError, match="stage lineage"):
        compiled_chart_transport_stage_sha256(
            layer_ordinal=1,
            layer_artifact_sha256=layer_artifact,
            incoming_edge_artifact_sha256s=(edge_artifact, edge_artifact),
        )


def test_protocol_preregisters_runtime_numerics_and_guard_decision() -> None:
    protocol = CrossLayerChartTransportProtocol()
    state = protocol.state_dict()
    runtime = state["pinned_runtime"]
    assert runtime == {
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "tokenizer_revision": PINNED_MODEL_REVISION,
        "device": "cpu",
        "dtype": "float32",
        "deterministic_seed": 1729,
        "development_panel_path": str(PINNED_DEVELOPMENT_PANEL_PATH),
        "development_panel_sha256": PINNED_DEVELOPMENT_PANEL_SHA256,
    }
    panel = Path(__file__).parents[1] / PINNED_DEVELOPMENT_PANEL_PATH
    assert hashlib.sha256(panel.read_bytes()).hexdigest() == (
        PINNED_DEVELOPMENT_PANEL_SHA256
    )
    assert state["chart_temperature_ladder"] == (
        protocol.chart_temperature_ladder
    )
    assert state["frozen_numerics"] == {
        "global_generator_rank_per_candidate_stack": True,
        "weighted_affine_solver": "dual_ridge_float64",
        "affine_relative_ridge": 1.0e-3,
        "local_chart_solver": "joint_weighted_ridge_float64",
        "local_chart_relative_ridge": 1.0e-2,
        "chart_geometry_kmeans_iterations": 12,
        "factorization": "per_chart_canonical_truncated_svd",
        "transport_solver": "deterministic_block_ridge",
        "block_coordinate_sweeps": 8,
        "monotonic_objective_tolerance": (
            "128_times_float64_epsilon_times_max_1_baseline_sse"
        ),
        "row_fisher_clamp": (0.1, 10.0),
    }
    assert state["guard_acceptance_thresholds"] == {
        "top1_agreement_to_native_minimum": 0.90,
        "delta_nll_per_token_maximum": 0.10,
        "native_to_candidate_kl_per_token_maximum": 0.10,
        "transport_gain_over_zero_edge_must_be_positive": True,
        "row_shuffled_gain_fraction_of_real_maximum": 0.25,
        "family_bootstrap_lower_bound_must_pass": True,
    }
    assert state["data_firewall"][
        "development_result_may_claim_heldout_confirmation"
    ] is False
    assert state["claim_boundary"]["compression_success_claimed"] is False
    assert state["model_experiment_executed"] is False


def test_temperature_ladder_is_validated_and_changes_protocol_identity() -> None:
    baseline = CrossLayerChartTransportProtocol().state_dict()["protocol_sha256"]
    narrowed = CrossLayerChartTransportProtocol(
        chart_temperature_ladder=(1.0,)
    ).state_dict()["protocol_sha256"]
    assert narrowed != baseline
    with pytest.raises(ValueError, match="temperature ladder"):
        CrossLayerChartTransportProtocol(chart_temperature_ladder=(0.0, 1.0))
    with pytest.raises(ValueError, match="temperature ladder"):
        CrossLayerChartTransportProtocol(chart_temperature_ladder=(1.0, 1.0))
