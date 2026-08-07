from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import fisher_graph.gemma3_layer17_full_block_closure_capture as capture
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_layer17_node_rank_ladder import LAYER17_FRAGMENT_IDS
from fisher_graph.gemma3_layer17_trajectory_row_capture import (
    Gemma3Layer17TrajectoryRowPair,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import LayerFragmentRows
from fisher_graph.gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


MODEL_SHA = "a" * 64
SELECTION_SHA = "b" * 64
LAYER10_GRAPH_SHA = "c" * 64
LAYER10_LOWERING_SHAS = ("d" * 64, "e" * 64)
LAYER17_GRAPH_SHA = "1" * 64
LAYER17_LOWERING_SHAS = ("2" * 64, "3" * 64, "4" * 64, "5" * 64)
INPUT_SITE = "layer.17.mlp.normalized_input"
POST_ATTENTION_SITE = "layer.17.post_attention"
POST_FF_DELTA_SITE = "layer.17.mlp.delta"
BLOCK_OUTPUT_SITE = "layer.17.output"
OUTPUT_SITE = "layer.17.mlp.operator_output"


def _batches() -> tuple[CalibrationBatch, ...]:
    return (
        CalibrationBatch(
            model_inputs={
                "input_ids": torch.tensor([[0, 10], [1, 11]], dtype=torch.long)
            },
            targets=torch.zeros((2, 2), dtype=torch.long),
            valid_positions=torch.tensor(
                [[True, False], [True, False]],
                dtype=torch.bool,
            ),
            example_ids=("example-a", "example-b"),
        ),
        CalibrationBatch(
            model_inputs={"input_ids": torch.tensor([[2, 12]], dtype=torch.long)},
            targets=torch.zeros((1, 2), dtype=torch.long),
            valid_positions=torch.tensor([[True, False]], dtype=torch.bool),
            example_ids=("example-c",),
        ),
    )


def _aligned_rows(*, offset: float) -> AlignedFragmentRows:
    keys = (("example-a", 0), ("example-b", 0), ("example-c", 0))
    inputs = torch.arange(6, dtype=torch.float64).reshape(3, 2) + offset
    return AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=inputs.clone(),
                contributions=torch.full(
                    (3, 3),
                    offset + index + 1.0,
                    dtype=torch.float64,
                ),
                fisher_weights=torch.arange(1, 4, dtype=torch.float64) + index,
                sequences=3,
            )
            for index, fragment_id in enumerate(LAYER17_FRAGMENT_IDS)
        },
        row_keys=keys,
    )


def _trajectory() -> Gemma3Layer17TrajectoryRowPair:
    native_rows = _aligned_rows(offset=0.0)
    compiled_rows = _aligned_rows(offset=0.25)
    compiled_full = torch.tensor(
        [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0], [26.0, 27.0, 28.0]],
        dtype=torch.float64,
    )
    selected = sum(
        (
            rows.contributions
            for rows in compiled_rows.rows_by_fragment.values()
        ),
        torch.zeros_like(compiled_full),
    )
    return Gemma3Layer17TrajectoryRowPair(
        native_rows=native_rows,
        compiled_rows=compiled_rows,
        native_full_mlp_output=compiled_full + 10.0,
        compiled_full_mlp_output=compiled_full,
        compiled_compact_retained_mlp_output=compiled_full - selected,
        model_fingerprint=MODEL_SHA,
        layer17_selection_sha256=SELECTION_SHA,
        layer17_leaf_activation_site=INPUT_SITE,
        fragment_ids=LAYER17_FRAGMENT_IDS,
        layer10_graph_sha256=LAYER10_GRAPH_SHA,
        layer10_traversal_order=("layer10/a", "layer10/b"),
        ordered_layer10_lowering_sha256s=LAYER10_LOWERING_SHAS,
        layer17_compact_graph_sha256=LAYER17_GRAPH_SHA,
        layer17_compact_traversal_order=tuple(
            f"layer17/{index}" for index in range(4)
        ),
        ordered_layer17_compact_lowering_sha256s=LAYER17_LOWERING_SHAS,
    )


class _FakeSelection:
    execution_order = (
        SimpleNamespace(output_site=OUTPUT_SITE),
        SimpleNamespace(output_site=OUTPUT_SITE),
        SimpleNamespace(output_site=OUTPUT_SITE),
        SimpleNamespace(output_site=OUTPUT_SITE),
    )


class _FakeAdapter:
    def __init__(
        self,
        executor: "_FakeLayer10Executor",
        *,
        native: dict[str, Tensor],
        compiled: dict[str, Tensor],
        compiled_position_offset: int = 0,
    ) -> None:
        self.executor = executor
        self.native = native
        self.compiled = compiled
        self.compiled_position_offset = compiled_position_offset
        self.forward_receipts: list[dict[str, object]] = []

    def forward(
        self,
        model_inputs: dict[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
        retain_gradients: bool,
    ) -> object:
        index = int(model_inputs["input_ids"][0, 0].item())
        source = self.compiled if self.executor.overlay_active else self.native
        self.forward_receipts.append(
            {
                "index": index,
                "overlay": self.executor.overlay_active,
                "grad_enabled": torch.is_grad_enabled(),
                "capture_sites": capture_sites,
                "retain_gradients": retain_gradients,
            }
        )
        activations = {
            site: torch.stack((source[site][index], source[site][index] + 99.0))
            .unsqueeze(0)
            .to(dtype=torch.float32)
            for site in capture_sites
        }
        position_offset = (
            self.compiled_position_offset if self.executor.overlay_active else 0
        )
        return SimpleNamespace(
            activations=activations,
            sequence=SimpleNamespace(
                query_valid_mask=torch.tensor([[True, True]], dtype=torch.bool),
                logical_positions=torch.tensor(
                    [[position_offset, position_offset + 1]],
                    dtype=torch.long,
                ),
            ),
        )


class _FakeLayer10Executor:
    def __init__(self) -> None:
        self.overlay_active = False
        self.overlay_calls: list[int] = []

    def run_with_generated_overlay(
        self,
        callback: object,
        *,
        expected_forward_calls: int,
    ) -> object:
        self.overlay_calls.append(expected_forward_calls)
        self.overlay_active = True
        try:
            return callback()  # type: ignore[operator]
        finally:
            self.overlay_active = False


class _FakeLayer17Executor:
    def __init__(self, compact_post_ff_delta: Tensor) -> None:
        self.compact_post_ff_delta = compact_post_ff_delta
        self.calls: list[tuple[int, Tensor]] = []

    def execute_compact_post_feedforward_delta_rows(
        self,
        layer_ordinal: int,
        normalized_inputs: Tensor,
    ) -> Tensor:
        self.calls.append((layer_ordinal, normalized_inputs.clone()))
        return self.compact_post_ff_delta.clone()


def _activation_values() -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    native_post_attention = torch.tensor(
        [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
    )
    native_delta = torch.tensor(
        [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]]
    )
    compiled_post_attention = native_post_attention + torch.tensor(
        [[0.5, -0.25, 0.75], [1.0, 0.0, -0.5], [-0.5, 0.25, 1.0]]
    )
    compiled_delta = native_delta + 2.0
    native = {
        POST_ATTENTION_SITE: native_post_attention,
        POST_FF_DELTA_SITE: native_delta,
        BLOCK_OUTPUT_SITE: native_post_attention + native_delta,
    }
    compiled = {
        POST_ATTENTION_SITE: compiled_post_attention,
        POST_FF_DELTA_SITE: compiled_delta,
        BLOCK_OUTPUT_SITE: compiled_post_attention + compiled_delta,
    }
    return native, compiled


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def _install_capture_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    trajectory: Gemma3Layer17TrajectoryRowPair,
    *,
    algebraic_post_ff_delta: Tensor,
) -> None:
    monkeypatch.setattr(
        capture,
        "capture_gemma3_layer17_native_and_layer10_rows",
        lambda *_args, **_kwargs: trajectory,
    )
    monkeypatch.setattr(
        capture,
        "_layer17_activation_sites",
        lambda *_args, **_kwargs: (
            POST_ATTENTION_SITE,
            POST_FF_DELTA_SITE,
            BLOCK_OUTPUT_SITE,
        ),
    )
    monkeypatch.setattr(
        capture,
        "_apply_live_post_feedforward_delta_rows",
        lambda *_args, **_kwargs: algebraic_post_ff_delta.clone(),
    )


def test_a4_capture_uses_activation_only_residual_rows_and_exact_postnorm_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()
    native, compiled = _activation_values()
    compact_post_ff_delta = torch.tensor(
        [[0.25, 0.5, 0.75], [0.5, 0.75, 1.0], [0.75, 1.0, 1.25]],
        dtype=torch.float64,
    )
    algebraic_post_ff_delta = compact_post_ff_delta + torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1e-8], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    layer10_executor = _FakeLayer10Executor()
    adapter = _FakeAdapter(
        layer10_executor,
        native=native,
        compiled=compiled,
    )
    layer17_executor = _FakeLayer17Executor(compact_post_ff_delta)
    _install_capture_dependencies(
        monkeypatch,
        trajectory,
        algebraic_post_ff_delta=algebraic_post_ff_delta,
    )

    result = capture.capture_gemma3_layer17_full_block_closure(
        adapter,  # type: ignore[arg-type]
        _batches(),
        selection=_FakeSelection(),  # type: ignore[arg-type]
        leaf_activation_site=INPUT_SITE,
        layer10_executor=layer10_executor,  # type: ignore[arg-type]
        layer17_executor=layer17_executor,  # type: ignore[arg-type]
    )

    expected_target = (
        native[BLOCK_OUTPUT_SITE]
        - compiled[POST_ATTENTION_SITE]
        - compact_post_ff_delta
    ).to(dtype=torch.float64)
    torch.testing.assert_close(result.a4_full_block_closure_target, expected_target)
    torch.testing.assert_close(
        result.a4_full_block_closure_target - result.native_delta_only_closure,
        result.residual_stream_closure_offset,
    )
    assert not torch.equal(
        result.compiled_compact_retained_post_feedforward_delta,
        trajectory.compiled_compact_retained_mlp_output,
    )
    assert result.native_rows is trajectory.native_rows
    assert result.compiled_rows is trajectory.compiled_rows
    assert layer10_executor.overlay_calls == [3]
    assert layer10_executor.overlay_active is False
    assert len(layer17_executor.calls) == 1
    ordinal, compact_inputs = layer17_executor.calls[0]
    assert ordinal == 17
    torch.testing.assert_close(
        compact_inputs,
        trajectory.compiled_rows.rows_by_fragment[LAYER17_FRAGMENT_IDS[0]].inputs,
    )
    assert len(adapter.forward_receipts) == 6
    assert [receipt["overlay"] for receipt in adapter.forward_receipts] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert all(
        receipt["grad_enabled"] is False for receipt in adapter.forward_receipts
    )
    assert all(
        receipt["capture_sites"]
        == (POST_ATTENTION_SITE, POST_FF_DELTA_SITE, BLOCK_OUTPUT_SITE)
        for receipt in adapter.forward_receipts
    )

    metadata = result.metadata()
    assert metadata["target"] == {
        "variant": "A4_full_block_closure",
        "symbol": "g_block",
        "formula": (
            "native_layer17_block_output-"
            "compiled_layer17_post_attention_residual-"
            "exact_compact_retained_layer17_post_feedforward_delta"
        ),
        "application_boundary": "layer.17.mlp.delta",
        "uses_exact_compact_post_feedforward_delta": True,
        "uses_raw_compact_mlp_output": False,
    }
    auxiliary = metadata["activation_only_capture"]
    assert auxiliary["uses_autograd_grad"] is False
    assert auxiliary["post_attention_derived_from_block_output_subtraction"] is False
    assert metadata["alignment"][
        "activation_fisher_and_activation_only_row_keys_equal"
    ] is True
    for name in (
        "native_block_decomposition",
        "compiled_block_decomposition",
        "a4_reconstruction",
        "a4_equivalent_formula",
        "a4_minus_delta_only_closure_offset_identity",
    ):
        assert metadata["audits"][name]["max_abs_difference"] == pytest.approx(0.0)
        assert metadata["audits"][name]["rms_difference"] == pytest.approx(0.0)
    compact_audit = metadata["audits"]["compact_post_feedforward_replay"]
    assert compact_audit["max_abs_difference"] == pytest.approx(1e-8)
    assert compact_audit["rms_difference"] == pytest.approx(1e-8 / 3.0)
    assert metadata["capture_sha256"] == result.capture_sha256
    assert metadata["trajectory_capture"]["capture_sha256"] == (
        trajectory.capture_sha256
    )
    assert not _contains_tensor(metadata)

    result.native_block_output[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="tensors or lineage drifted"):
        result.metadata()


def test_a4_capture_rejects_auxiliary_row_misalignment_before_compact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory()
    native, compiled = _activation_values()
    compact_post_ff_delta = torch.zeros((3, 3), dtype=torch.float64)
    layer10_executor = _FakeLayer10Executor()
    adapter = _FakeAdapter(
        layer10_executor,
        native=native,
        compiled=compiled,
        compiled_position_offset=1,
    )
    layer17_executor = _FakeLayer17Executor(compact_post_ff_delta)
    _install_capture_dependencies(
        monkeypatch,
        trajectory,
        algebraic_post_ff_delta=compact_post_ff_delta,
    )

    with pytest.raises(ValueError, match="row keys are not aligned"):
        capture.capture_gemma3_layer17_full_block_closure(
            adapter,  # type: ignore[arg-type]
            _batches(),
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
            layer10_executor=layer10_executor,  # type: ignore[arg-type]
            layer17_executor=layer17_executor,  # type: ignore[arg-type]
        )
    assert layer17_executor.calls == []


def test_a4_capture_rejects_direct_row_key_or_tensor_shape_drift() -> None:
    trajectory = _trajectory()
    native, compiled = _activation_values()
    compact = torch.zeros((3, 3), dtype=torch.float64)
    fields = {
        "trajectory_rows": trajectory,
        "native_activation_row_keys": trajectory.native_rows.row_keys,
        "compiled_activation_row_keys": trajectory.native_rows.row_keys,
        "native_post_attention_residual": native[POST_ATTENTION_SITE],
        "native_post_feedforward_delta": native[POST_FF_DELTA_SITE],
        "native_block_output": native[BLOCK_OUTPUT_SITE],
        "compiled_post_attention_residual": compiled[POST_ATTENTION_SITE],
        "compiled_post_feedforward_delta": compiled[POST_FF_DELTA_SITE],
        "compiled_block_output": compiled[BLOCK_OUTPUT_SITE],
        "compiled_compact_retained_post_feedforward_delta": compact,
        "algebraic_compact_retained_post_feedforward_delta": compact,
    }
    with pytest.raises(ValueError, match="row keys are not aligned"):
        capture.Gemma3Layer17FullBlockClosureCapture(
            **{
                **fields,
                "compiled_activation_row_keys": (
                    ("example-a", 0),
                    ("example-b", 0),
                    ("wrong", 0),
                ),
            }
        )
    with pytest.raises(ValueError, match="compiled block output must be"):
        capture.Gemma3Layer17FullBlockClosureCapture(
            **{
                **fields,
                "compiled_block_output": torch.zeros((2, 3)),
            }
        )


def test_layer17_activation_sites_resolve_exact_structured_boundaries() -> None:
    attention = SimpleNamespace(
        kind="attention",
        output_site=POST_ATTENTION_SITE,
    )
    feed_forward = SimpleNamespace(
        kind="feed_forward",
        input_site=POST_ATTENTION_SITE,
        normalized_input_site=INPUT_SITE,
        operator_output_site=OUTPUT_SITE,
        delta_site=POST_FF_DELTA_SITE,
        output_site=BLOCK_OUTPUT_SITE,
    )
    layer17 = SimpleNamespace(
        id="layer.17",
        output_site=BLOCK_OUTPUT_SITE,
        transformer=SimpleNamespace(stages=(attention, feed_forward)),
    )
    adapter = SimpleNamespace(layers=(*([object()] * 17), layer17))

    assert capture._layer17_activation_sites(
        adapter,  # type: ignore[arg-type]
        selection=_FakeSelection(),  # type: ignore[arg-type]
        leaf_activation_site=INPUT_SITE,
    ) == (POST_ATTENTION_SITE, POST_FF_DELTA_SITE, BLOCK_OUTPUT_SITE)

    feed_forward.delta_site = "layer.17.mlp.wrong_delta"
    with pytest.raises(ValueError, match="activation sites drifted"):
        capture._layer17_activation_sites(
            adapter,  # type: ignore[arg-type]
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
        )


def test_live_postfeedforward_audit_replay_uses_source_dtype_and_no_grad() -> None:
    class _ScaleNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
            self.grad_modes: list[bool] = []

        def forward(self, value: Tensor) -> Tensor:
            self.grad_modes.append(torch.is_grad_enabled())
            return value * self.weight

    norm = _ScaleNorm()
    source_layer = SimpleNamespace(post_feedforward_layernorm=norm)
    adapter = SimpleNamespace(source_module=lambda _layer_id: source_layer)
    raw = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float64,
    )

    result = capture._apply_live_post_feedforward_delta_rows(
        adapter,  # type: ignore[arg-type]
        17,
        raw,
    )

    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, raw * 2.0)
    assert norm.grad_modes == [False]
