from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGeneratorConfig,
    compile_exact_attention_generator,
)
from fisher_graph.gemma3_attention_transport_runtime import (
    AttentionTransportCapturedInvocation,
    Gemma3AttentionTransportCaptureRuntime,
    sample_attention_transport_captured_invocation,
)
from fisher_graph.gemma3_chart_native_neighbor_experiment import (
    FIT_ROLE,
    CapturedBatch,
    CapturedExampleRows,
    CapturedModeRows,
)
from fisher_graph.gemma3_chart_native_neighbor_runtime import (
    Gemma3ChartNativeCaptureRuntime,
)


HIDDEN = 4
LENGTH = 4
BATCH = 2


def _normalization(width: int) -> NormalizationSpec:
    return NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=1.0e-6,
        affine=True,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )


def _attention(event_id: str, *, seed: int):
    random = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(
            rows,
            columns,
            dtype=torch.float32,
            generator=random,
        ) / 5.0

    head_dimension = 2
    config = Gemma3AttentionGeneratorConfig(
        invocation_id=event_id,
        residual_width=HIDDEN,
        attention=AttentionSpec(
            kind="global_causal",
            query_heads=2,
            key_value_heads=1,
            head_dimension=head_dimension,
            query_scale=head_dimension**-0.5,
            qk_norm=True,
            window_size=None,
            rope=RopeSpec(
                kind="rotary",
                theta=10_000.0,
                rotary_dimension=head_dimension,
                scaling_kind=None,
                scaling_factor=None,
            ),
            cache_kind="none",
        ),
        input_norm=_normalization(HIDDEN),
        output_norm=_normalization(HIDDEN),
        qk_norm=_normalization(head_dimension),
        projection_bias=False,
        dropout=0.0,
        logit_softcap=None,
    )
    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=torch.tensor(
                [0.0, 0.05, -0.025, 0.1],
                dtype=torch.float32,
            ),
            q_weight=matrix(4, HIDDEN),
            k_weight=matrix(2, HIDDEN),
            v_weight=matrix(2, HIDDEN),
            o_weight=matrix(HIDDEN, 4),
            output_norm_weight=torch.tensor(
                [0.1, -0.05, 0.025, 0.075],
                dtype=torch.float32,
            ),
            q_norm_weight=torch.zeros(head_dimension),
            k_norm_weight=torch.zeros(head_dimension),
        ),
        source_checkpoint_coefficients_retained=True,
    )


def _sequence() -> SequenceContext:
    valid = torch.ones(BATCH, LENGTH, dtype=torch.bool)
    positions = torch.arange(LENGTH, dtype=torch.int64)[None].expand(
        BATCH,
        -1,
    )
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


def _transport_state() -> SimpleNamespace:
    sequence = _sequence()
    observations: dict[str, torch.Tensor] = {}
    routes: dict[str, object] = {}
    operator_outputs: dict[str, torch.Tensor] = {}
    operator_gradients: dict[str, torch.Tensor] = {}
    source_executions: dict[str, object] = {}
    mutations: list[object] = []
    for mutation_index in range(1, 37):
        event_id = f"mutation.{mutation_index}"
        if mutation_index % 2 == 0:
            mutations.append(SimpleNamespace(compiled_operator=None))
            continue
        generator = _attention(event_id, seed=71_000 + mutation_index)
        residual = torch.randn(
            BATCH,
            LENGTH,
            HIDDEN,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(72_000 + mutation_index),
        )
        execution = generator(
            residual,
            sequence,
            capture_route_factors=True,
        )
        assert execution.normalized_input is not None
        assert execution.route_factor_snapshot is not None
        operator_site = f"{event_id}.operator_output"
        gradient = torch.full_like(
            execution.operator_output,
            float(mutation_index) / 37.0,
        )
        observations[f"{event_id}.input"] = execution.normalized_input
        observations[f"{event_id}.output"] = execution.residual_delta
        routes[event_id] = execution.route_factor_snapshot
        operator_outputs[operator_site] = execution.operator_output
        operator_gradients[operator_site] = gradient
        source_executions[event_id] = execution
        mutations.append(SimpleNamespace(compiled_operator=generator))
    observed = SimpleNamespace(
        execution=SimpleNamespace(
            attention_operator_outputs=operator_outputs,
        )
    )
    bundle = SimpleNamespace(
        attention_operator_output_gradients=operator_gradients,
    )
    return SimpleNamespace(
        backbone=(observed, observations, routes, sequence),
        bundle=bundle,
        mutations=tuple(mutations),
        source_executions=source_executions,
        operator_gradients=operator_gradients,
    )


def _captured_batch() -> CapturedBatch:
    example = CapturedExampleRows(
        family_id="test-family",
        example_id="test-example",
        token_count=2,
        row_keys=(("test-family", "test-example", 0, 1),),
        modes=(
            CapturedModeRows(
                mode_id="mode.test",
                context_values=None,
                comparison_values=None,
                generalized_contributions=None,
            ),
        ),
        context_fingerprint_sha256="b" * 64,
        capture_receipt_sha256="c" * 64,
    )
    return CapturedBatch(
        role=FIT_ROLE,
        examples=(example,),
        lexical_nll_receipt_sha256="d" * 64,
        execution_receipt_sha256="e" * 64,
    )


def _runtime_shell(
    state: SimpleNamespace,
    sink,
) -> Gemma3AttentionTransportCaptureRuntime:
    runtime = object.__new__(Gemma3AttentionTransportCaptureRuntime)
    runtime._transport_sink = sink
    runtime._transport_capture_active = False
    runtime._transport_backbone = None
    runtime._transport_gradient_bundle = None
    runtime._executor = SimpleNamespace(mutations=state.mutations)
    return runtime


def _install_base_capture(
    monkeypatch: pytest.MonkeyPatch,
    state: SimpleNamespace,
    captured: CapturedBatch,
    *,
    failure_phase: str | None = None,
) -> dict[str, int]:
    calls = {"capture": 0, "backbone": 0, "validate": 0}

    def run_backbone(self, *, input_ids, capture_routes):
        del self, input_ids
        assert capture_routes is True
        calls["backbone"] += 1
        return state.backbone

    def validate(bundle):
        assert bundle is state.bundle
        calls["validate"] += 1

    def capture_batch(self, **kwargs):
        del kwargs
        calls["capture"] += 1
        self._run_backbone(
            input_ids=torch.ones(1, 2, dtype=torch.int64),
            capture_routes=True,
        )
        if failure_phase == "backbone":
            raise RuntimeError("injected base capture failure")
        self._validate_gradient_bundle(state.bundle)
        if failure_phase == "gradient":
            raise RuntimeError("injected base capture failure")
        return captured

    monkeypatch.setattr(
        Gemma3ChartNativeCaptureRuntime,
        "_run_backbone",
        run_backbone,
    )
    monkeypatch.setattr(
        Gemma3ChartNativeCaptureRuntime,
        "_validate_gradient_bundle",
        staticmethod(validate),
    )
    monkeypatch.setattr(
        Gemma3ChartNativeCaptureRuntime,
        "capture_batch",
        capture_batch,
    )
    return calls


def _capture(
    runtime: Gemma3AttentionTransportCaptureRuntime,
):
    return runtime.capture_batch(
        preflight=object(),
        role=FIT_ROLE,
        raw_examples=(),
        role_manifest={},
    )


def _assert_ephemeral_state_cleared(
    runtime: Gemma3AttentionTransportCaptureRuntime,
) -> None:
    assert runtime._transport_capture_active is False
    assert runtime._transport_backbone is None
    assert runtime._transport_gradient_bundle is None


def test_runtime_calls_base_once_and_emits_exact_18_shifted_nll_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _transport_state()
    base_capture = _captured_batch()
    emissions: list[AttentionTransportCapturedInvocation] = []
    runtime = _runtime_shell(state, emissions.append)
    calls = _install_base_capture(monkeypatch, state, base_capture)

    returned = _capture(runtime)

    assert returned is base_capture
    assert calls == {"capture": 1, "backbone": 1, "validate": 1}
    assert len(emissions) == 18
    assert tuple(row.capture.invocation_id for row in emissions) == tuple(
        f"mutation.{index}" for index in range(1, 37, 2)
    )
    for row in emissions:
        event_id = row.capture.invocation_id
        execution = state.source_executions[event_id]
        route = execution.route_factor_snapshot
        assert route is not None
        operator_site = f"{event_id}.operator_output"
        assert row.role == FIT_ROLE
        assert row.base_capture_receipt_sha256 == "e" * 64
        assert row.capture.row_count == BATCH * (LENGTH - 1)
        assert torch.equal(
            row.capture.normalized_inputs,
            execution.normalized_input.detach().to(torch.float64),
        )
        assert torch.equal(
            row.capture.operator_outputs,
            execution.operator_output.detach().to(torch.float64),
        )
        assert torch.equal(
            row.capture.operator_output_score_gradients,
            state.operator_gradients[operator_site].to(torch.float64),
        )
        assert bool(route.query_valid_mask[:, -1].all())
        assert bool(row.capture.query_valid_mask[:, :-1].all())
        assert not bool(row.capture.query_valid_mask[:, -1].any())
        assert torch.equal(
            row.capture.attention_probabilities[:, :, :-1],
            route.probabilities[:, :, :-1].to(torch.float64),
        )
        assert not bool(
            torch.count_nonzero(
                row.capture.attention_probabilities[:, :, -1]
            )
        )
        expected_postnorm = execution.residual_delta[
            row.capture.query_valid_mask
        ].to(torch.float64)
        assert torch.equal(row.postnorm_targets, expected_postnorm)
        assert torch.allclose(
            row.output_norm.apply(row.capture.valid_operator_outputs),
            row.postnorm_targets,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
    _assert_ephemeral_state_cleared(runtime)

    sampled = sample_attention_transport_captured_invocation(
        emissions[0],
        maximum_rows=2,
    )
    expected_indices = torch.tensor([1, 4], dtype=torch.int64)
    assert sampled.capture.row_count == 2
    assert torch.equal(
        sampled.capture.valid_operator_outputs,
        emissions[0].capture.valid_operator_outputs.index_select(
            0,
            expected_indices,
        ),
    )
    assert torch.equal(
        sampled.postnorm_targets,
        emissions[0].postnorm_targets.index_select(0, expected_indices),
    )


def test_postnorm_target_receipt_rejects_a_nonclosing_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _transport_state()
    emissions: list[AttentionTransportCapturedInvocation] = []
    runtime = _runtime_shell(state, emissions.append)
    _install_base_capture(monkeypatch, state, _captured_batch())
    _capture(runtime)

    with pytest.raises(ValueError, match="postnorm delta does not close"):
        replace(
            emissions[0],
            postnorm_targets=emissions[0].postnorm_targets + 0.25,
        )


def test_runtime_reuses_one_head_operator_catalog_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _transport_state()
    emissions: list[AttentionTransportCapturedInvocation] = []
    runtime = _runtime_shell(state, emissions.append)
    calls = _install_base_capture(monkeypatch, state, _captured_batch())

    _capture(runtime)
    _capture(runtime)

    assert calls == {"capture": 2, "backbone": 2, "validate": 2}
    assert len(emissions) == 36
    for invocation_index in range(18):
        first = emissions[invocation_index]
        second = emissions[18 + invocation_index]
        assert first.head_operator_catalog is second.head_operator_catalog
        first_rows = sample_attention_transport_captured_invocation(
            first,
            maximum_rows=2,
        )
        second_rows = sample_attention_transport_captured_invocation(
            second,
            maximum_rows=2,
        )
        assert (
            first_rows.capture.head_operators
            is second_rows.capture.head_operators
        )
        assert (
            first_rows.capture.head_operator_catalog_identity
            == second_rows.capture.head_operator_catalog_identity
        )


@pytest.mark.parametrize("failure_phase", ["backbone", "gradient"])
def test_runtime_clears_ephemeral_state_when_base_capture_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    state = _transport_state()
    emissions: list[AttentionTransportCapturedInvocation] = []
    runtime = _runtime_shell(state, emissions.append)
    calls = _install_base_capture(
        monkeypatch,
        state,
        _captured_batch(),
        failure_phase=failure_phase,
    )

    with pytest.raises(RuntimeError, match="injected base capture failure"):
        _capture(runtime)

    assert calls["capture"] == 1
    assert calls["backbone"] == 1
    assert calls["validate"] == int(failure_phase == "gradient")
    assert emissions == []
    _assert_ephemeral_state_cleared(runtime)


def test_runtime_clears_ephemeral_state_when_sink_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _transport_state()
    sink_calls = 0

    def rejecting_sink(_capture):
        nonlocal sink_calls
        sink_calls += 1
        raise RuntimeError("injected sink failure")

    runtime = _runtime_shell(state, rejecting_sink)
    calls = _install_base_capture(monkeypatch, state, _captured_batch())

    with pytest.raises(RuntimeError, match="injected sink failure"):
        _capture(runtime)

    assert calls == {"capture": 1, "backbone": 1, "validate": 1}
    assert sink_calls == 1
    _assert_ephemeral_state_cleared(runtime)
