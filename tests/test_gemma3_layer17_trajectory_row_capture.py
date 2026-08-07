from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

import fisher_graph.gemma3_layer17_trajectory_row_capture as capture
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_layer17_node_rank_ladder import LAYER17_FRAGMENT_IDS
from fisher_graph.gemma3_modal_generator_dev_experiment import LayerFragmentRows
from fisher_graph.gemma3_modal_generator_terminal_fanin import AlignedFragmentRows
from fisher_graph.streaming_analysis import ActivationScoreGradientRows


MODEL_SHA = "a" * 64
SELECTION_SHA = "b" * 64
GRAPH_SHA = "c" * 64
LOWERING_SHAS = ("d" * 64, "e" * 64)
LAYER17_GRAPH_SHA = "1" * 64
LAYER17_LOWERING_SHAS = ("2" * 64, "3" * 64, "4" * 64, "5" * 64)
INPUT_SITE = "model.layers.17.mlp.input"
OUTPUT_SITE = "model.layers.17.mlp.residual_delta"
FRAGMENT_SHAS = ("6" * 64, "7" * 64, "8" * 64, "9" * 64)
FRAGMENT_CHANNELS = ((0,), (2,), (4,), (6,))
LAYER17_NODE_NAMES = tuple(
    f"layer17/node-{index}" for index in range(len(LAYER17_FRAGMENT_IDS))
)


class _FakeSelection:
    layer_ordinal = 17
    source_model_sha256 = MODEL_SHA
    artifact_sha256 = SELECTION_SHA
    fragment_ids = LAYER17_FRAGMENT_IDS
    execution_order = tuple(
        SimpleNamespace(
            input_site=INPUT_SITE,
            output_site=OUTPUT_SITE,
            fragment_id=fragment_id,
            artifact_sha256=fragment_sha,
            layer_ordinal=17,
            channel_indices=channels,
        )
        for fragment_id, fragment_sha, channels in zip(
            LAYER17_FRAGMENT_IDS,
            FRAGMENT_SHAS,
            FRAGMENT_CHANNELS,
            strict=True,
        )
    )

    def validate_integrity(self) -> None:
        return None


class _FakeGraphPlan:
    def __init__(self, layer_ordinal: int, *, interactions: tuple[object, ...]) -> None:
        self.model_fingerprint = MODEL_SHA
        self.artifact_sha256 = (
            GRAPH_SHA if layer_ordinal == 10 else LAYER17_GRAPH_SHA
        )
        self.traversal_order = (
            ("layer10/source", "layer10/target")
            if layer_ordinal == 10
            else LAYER17_NODE_NAMES
        )
        self.interactions = interactions

    def validate_integrity(self) -> None:
        return None


class _FakeExecutor:
    def __init__(
        self,
        adapter: object,
        *,
        affected: tuple[int, ...] = (10,),
        interactions: tuple[object, ...] = (),
        compact_output: Tensor | None = None,
    ) -> None:
        self.adapter = adapter
        layer_ordinal = affected[0]
        self.graph_plan = _FakeGraphPlan(
            layer_ordinal,
            interactions=interactions,
        )
        self._affected_ordinals = affected
        if layer_ordinal == 10:
            self._bound_nodes = tuple(
                SimpleNamespace(
                    node=SimpleNamespace(name=name),
                    lowering=SimpleNamespace(artifact_sha256=lowering_sha),
                    fragment=SimpleNamespace(layer_ordinal=10),
                )
                for name, lowering_sha in zip(
                    self.graph_plan.traversal_order,
                    LOWERING_SHAS,
                    strict=True,
                )
            )
            self.compiled_mlps: dict[str, object] = {}
        else:
            self._bound_nodes = tuple(
                SimpleNamespace(
                    node=SimpleNamespace(
                        name=name,
                        input_boundary=fragment.input_site,
                        output_boundary=fragment.output_site,
                    ),
                    lowering=SimpleNamespace(artifact_sha256=lowering_sha),
                    fragment=fragment,
                )
                for name, lowering_sha, fragment in zip(
                    self.graph_plan.traversal_order,
                    LAYER17_LOWERING_SHAS,
                    _FakeSelection.execution_order,
                    strict=True,
                )
            )
            removed = tuple(
                sorted(
                    channel
                    for channels in FRAGMENT_CHANNELS
                    for channel in channels
                )
            )
            self.compiled_mlps = {
                "17": SimpleNamespace(removed_mode_indices=removed)
            }
        self.overlay_active = False
        self.overlay_calls: list[int] = []
        self.compact_output = compact_output
        self.compact_calls: list[tuple[int, Tensor]] = []

    @property
    def affected_layer_ordinals(self) -> tuple[int, ...]:
        return self._affected_ordinals

    def run_with_generated_overlay(
        self,
        callback: Any,
        *,
        expected_forward_calls: int,
    ) -> Any:
        self.overlay_calls.append(expected_forward_calls)
        self.overlay_active = True
        try:
            return callback()
        finally:
            self.overlay_active = False

    def execute_compact_mlp_rows(
        self,
        layer_ordinal: int,
        normalized_inputs: Tensor,
    ) -> Tensor:
        self.compact_calls.append((layer_ordinal, normalized_inputs.clone()))
        if self.compact_output is None:
            raise RuntimeError("fake compact output was not configured")
        return self.compact_output.clone()


class _FakeAdapter:
    def model_fingerprint(self) -> str:
        return MODEL_SHA


def _batches() -> tuple[CalibrationBatch, ...]:
    def batch(prefix: str, size: int) -> CalibrationBatch:
        return CalibrationBatch(
            model_inputs={"input_ids": torch.arange(size * 2).reshape(size, 2)},
            targets=torch.zeros((size, 2), dtype=torch.long),
            valid_positions=torch.ones((size, 2), dtype=torch.bool),
            example_ids=tuple(f"{prefix}-{index}" for index in range(size)),
        )

    return batch("a", 2), batch("b", 1)


def _aligned_rows(
    *,
    offset: float,
    row_keys: tuple[tuple[str, int], ...] | None = None,
    fragment_ids: tuple[str, ...] = LAYER17_FRAGMENT_IDS,
    sequences: int = 3,
) -> AlignedFragmentRows:
    keys = (
        (("a-0", 0), ("a-1", 0), ("b-0", 0))
        if row_keys is None
        else row_keys
    )
    base = torch.arange(len(keys) * 2, dtype=torch.float64).reshape(len(keys), 2)
    rows = {
        fragment_id: LayerFragmentRows(
            inputs=base + offset,
            contributions=torch.full(
                (len(keys), 3),
                offset + fragment_index + 1.0,
                dtype=torch.float64,
            ),
            fisher_weights=torch.arange(
                1,
                len(keys) + 1,
                dtype=torch.float64,
            )
            + fragment_index,
            sequences=sequences,
        )
        for fragment_index, fragment_id in enumerate(fragment_ids)
    }
    return AlignedFragmentRows(rows_by_fragment=rows, row_keys=keys)


def _install_synthetic_capture(
    monkeypatch: pytest.MonkeyPatch,
    executor: _FakeExecutor,
    *,
    native_rows: AlignedFragmentRows,
    compiled_rows: AlignedFragmentRows,
    native_full: Tensor,
    compiled_full: Tensor,
) -> list[bool]:
    observed_collector_conditions: list[bool] = []

    def fake_stream(
        _model: object,
        _batches: object,
        *,
        activation_names: tuple[str, ...],
        **_kwargs: object,
    ):
        full = compiled_full if executor.overlay_active else native_full
        keys = (
            compiled_rows.row_keys
            if executor.overlay_active
            else native_rows.row_keys
        )
        for index, (example_id, position) in enumerate(keys):
            activations = {
                name: (
                    full[index : index + 1]
                    if name == OUTPUT_SITE
                    else torch.zeros((1, 3), dtype=torch.float64)
                )
                for name in activation_names
            }
            yield ActivationScoreGradientRows(
                activations=activations,
                score_gradients={
                    name: torch.ones_like(value)
                    for name, value in activations.items()
                },
                logical_positions=torch.tensor([position], dtype=torch.int64),
                loss=0.0,
                example_id=example_id,
            )

    def fake_aligned_collector(
        _adapter: object,
        calibration_batches: object,
        *,
        selection: object,
        leaf_activation_site: str,
        row_factory: Any,
    ) -> AlignedFragmentRows:
        assert tuple(batch.batch_size for batch in calibration_batches) == (2, 1)
        assert selection.__class__ is _FakeSelection
        assert leaf_activation_site == INPUT_SITE
        observed_collector_conditions.append(executor.overlay_active)
        tuple(
            row_factory(
                _adapter,
                calibration_batches,
                activation_names=(INPUT_SITE, "model.layers.17.mlp.down_input"),
                score_objective=SimpleNamespace(),
                leaf_activation_name=leaf_activation_site,
                accumulation_dtype=torch.float64,
            )
        )
        return compiled_rows if executor.overlay_active else native_rows

    monkeypatch.setattr(capture, "SameLayerFragmentSelection", _FakeSelection)
    monkeypatch.setattr(capture, "Gemma3ModalGeneratorGraphExecutor", _FakeExecutor)
    monkeypatch.setattr(
        capture,
        "iter_activation_score_gradient_rows",
        fake_stream,
    )
    monkeypatch.setattr(
        capture,
        "_collect_same_layer_native_rows",
        fake_aligned_collector,
    )
    return observed_collector_conditions


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def test_capture_pairs_same_pass_full_outputs_and_commits_a3_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    executor = _FakeExecutor(adapter)
    native_rows = _aligned_rows(offset=0.0)
    compiled_rows = _aligned_rows(offset=0.25)
    native_full = torch.tensor(
        [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]],
        dtype=torch.float64,
    )
    compiled_full = torch.tensor(
        [[7.0, 18.0, 29.0], [8.0, 19.0, 30.0], [9.0, 20.0, 31.0]],
        dtype=torch.float64,
    )
    expected_selected = sum(
        (
            rows.contributions
            for rows in compiled_rows.rows_by_fragment.values()
        ),
        torch.zeros_like(compiled_full),
    )
    algebraic_retained = compiled_full - expected_selected
    compact_retained = algebraic_retained + torch.tensor(
        [[0.125, 0.0, -0.25], [0.0, 0.5, 0.0], [-0.125, 0.0, 0.25]],
        dtype=torch.float64,
    )
    layer17_executor = _FakeExecutor(
        adapter,
        affected=(17,),
        compact_output=compact_retained,
    )
    observed = _install_synthetic_capture(
        monkeypatch,
        executor,
        native_rows=native_rows,
        compiled_rows=compiled_rows,
        native_full=native_full,
        compiled_full=compiled_full,
    )

    result = capture.capture_gemma3_layer17_native_and_layer10_rows(
        adapter,  # type: ignore[arg-type]
        _batches(),
        selection=_FakeSelection(),  # type: ignore[arg-type]
        leaf_activation_site=INPUT_SITE,
        layer10_executor=executor,  # type: ignore[arg-type]
        layer17_executor=layer17_executor,  # type: ignore[arg-type]
    )

    assert observed == [False, True]
    assert executor.overlay_calls == [3]
    assert executor.overlay_active is False
    assert result.native_rows is native_rows
    assert result.compiled_rows is compiled_rows
    torch.testing.assert_close(result.native_full_mlp_output, native_full)
    torch.testing.assert_close(result.compiled_full_mlp_output, compiled_full)
    torch.testing.assert_close(
        result.compiled_compact_retained_mlp_output,
        compact_retained,
    )
    assert len(layer17_executor.compact_calls) == 1
    compact_ordinal, compact_inputs = layer17_executor.compact_calls[0]
    assert compact_ordinal == 17
    torch.testing.assert_close(
        compact_inputs,
        compiled_rows.rows_by_fragment[LAYER17_FRAGMENT_IDS[0]].inputs,
    )
    torch.testing.assert_close(
        result.a3_correction_target,
        native_full - compact_retained,
    )
    torch.testing.assert_close(
        result.algebraic_compact_retained_mlp_output,
        algebraic_retained,
    )

    metadata = result.metadata()
    assert metadata["condition"] == "generated"
    assert metadata["affected_layer_ordinals"] == (10,)
    assert metadata["layer10"] == {
        "graph_sha256": GRAPH_SHA,
        "traversal_order": executor.graph_plan.traversal_order,
        "ordered_lowering_sha256s": LOWERING_SHAS,
    }
    assert metadata["layer17"]["teacher_role"].startswith("native_layer17")
    assert "after_frozen_layer10" in metadata["layer17"]["input_roles"]["compiled"]
    assert metadata["layer17"]["compact_executor"] == {
        "graph_sha256": LAYER17_GRAPH_SHA,
        "traversal_order": LAYER17_NODE_NAMES,
        "ordered_lowering_sha256s": LAYER17_LOWERING_SHAS,
        "interaction_count": 0,
        "affected_layer_ordinals": (17,),
    }
    assert metadata["layer17"]["a3_target_role"].endswith(
        "exact_authenticated_compact_retained_output"
    )
    assert set(metadata["tensor_sha256s"]["native_rows"]) == set(
        LAYER17_FRAGMENT_IDS
    )
    assert set(metadata["tensor_sha256s"]["full_mlp_outputs"]) == {
        "native_sha256",
        "compiled_sha256",
        "exact_compact_retained_sha256",
        "algebraic_compact_retained_audit_sha256",
        "compact_retained_audit_difference_sha256",
        "a3_correction_target_sha256",
    }
    difference = compact_retained - algebraic_retained
    audit = metadata["compact_retained_numerical_audit"]
    assert audit["difference_definition"] == (
        "exact_compact_retained_minus_compiled_full_minus_"
        "selected_compiled_contributions"
    )
    assert audit["max_abs_difference"] == pytest.approx(
        float(difference.abs().max().item())
    )
    assert audit["rms_difference"] == pytest.approx(
        float(torch.sqrt(torch.mean(difference.square())).item())
    )
    assert metadata["capture_sha256"] == result.capture_sha256
    assert not _contains_tensor(metadata)

    result.native_full_mlp_output[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="tensors or lineage drifted"):
        result.metadata()


@pytest.mark.parametrize(
    ("compiled_rows", "message"),
    (
        (
            _aligned_rows(
                offset=0.25,
                fragment_ids=tuple(reversed(LAYER17_FRAGMENT_IDS)),
            ),
            "fragment catalog",
        ),
        (
            _aligned_rows(
                offset=0.25,
                row_keys=(("a-0", 0), ("a-1", 0), ("different", 0)),
            ),
            "row keys differ",
        ),
        (_aligned_rows(offset=0.25, sequences=2), "sequence counts differ"),
        (
            _aligned_rows(
                offset=0.25,
                row_keys=(
                    ("a-0", 0),
                    ("a-1", 0),
                    ("b-0", 0),
                    ("b-0", 1),
                ),
            ),
            "observation counts differ",
        ),
    ),
)
def test_capture_rejects_misaligned_native_and_compiled_rows(
    monkeypatch: pytest.MonkeyPatch,
    compiled_rows: AlignedFragmentRows,
    message: str,
) -> None:
    adapter = _FakeAdapter()
    executor = _FakeExecutor(adapter)
    native_rows = _aligned_rows(offset=0.0)
    native_full = torch.zeros((3, 3), dtype=torch.float64)
    compiled_full = torch.zeros(
        (compiled_rows.observations, 3),
        dtype=torch.float64,
    )
    layer17_executor = _FakeExecutor(
        adapter,
        affected=(17,),
        compact_output=compiled_full,
    )
    _install_synthetic_capture(
        monkeypatch,
        executor,
        native_rows=native_rows,
        compiled_rows=compiled_rows,
        native_full=native_full,
        compiled_full=compiled_full,
    )

    with pytest.raises(ValueError, match=message):
        capture.capture_gemma3_layer17_native_and_layer10_rows(
            adapter,  # type: ignore[arg-type]
            _batches(),
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
            layer10_executor=executor,  # type: ignore[arg-type]
            layer17_executor=layer17_executor,  # type: ignore[arg-type]
        )


def test_capture_rejects_non_layer10_executor_before_row_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    executor = _FakeExecutor(adapter, affected=(10, 17))
    layer17_executor = _FakeExecutor(
        adapter,
        affected=(17,),
        compact_output=torch.zeros((3, 3), dtype=torch.float64),
    )
    monkeypatch.setattr(capture, "SameLayerFragmentSelection", _FakeSelection)
    monkeypatch.setattr(capture, "Gemma3ModalGeneratorGraphExecutor", _FakeExecutor)
    monkeypatch.setattr(
        capture,
        "_capture_rows_and_full_output",
        lambda *_args, **_kwargs: pytest.fail("rows must not be collected"),
    )

    with pytest.raises(ValueError, match="exact Layer10-only graph"):
        capture.capture_gemma3_layer17_native_and_layer10_rows(
            adapter,  # type: ignore[arg-type]
            _batches(),
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
            layer10_executor=executor,  # type: ignore[arg-type]
            layer17_executor=layer17_executor,  # type: ignore[arg-type]
        )


def test_capture_requires_exact_edgeless_layer17_executor_before_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    layer10_executor = _FakeExecutor(adapter)
    layer17_executor = _FakeExecutor(
        adapter,
        affected=(17,),
        interactions=(object(),),
        compact_output=torch.zeros((3, 3), dtype=torch.float64),
    )
    monkeypatch.setattr(capture, "SameLayerFragmentSelection", _FakeSelection)
    monkeypatch.setattr(capture, "Gemma3ModalGeneratorGraphExecutor", _FakeExecutor)
    monkeypatch.setattr(
        capture,
        "_capture_rows_and_full_output",
        lambda *_args, **_kwargs: pytest.fail("rows must not be collected"),
    )

    with pytest.raises(ValueError, match="must be edgeless"):
        capture.capture_gemma3_layer17_native_and_layer10_rows(
            adapter,  # type: ignore[arg-type]
            _batches(),
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
            layer10_executor=layer10_executor,  # type: ignore[arg-type]
            layer17_executor=layer17_executor,  # type: ignore[arg-type]
        )


def test_capture_rejects_layer17_topology_drift_before_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    layer10_executor = _FakeExecutor(adapter)
    layer17_executor = _FakeExecutor(
        adapter,
        affected=(17,),
        compact_output=torch.zeros((3, 3), dtype=torch.float64),
    )
    layer17_executor._bound_nodes = tuple(
        (
            SimpleNamespace(
                node=bound.node,
                lowering=bound.lowering,
                fragment=SimpleNamespace(
                    **{
                        **vars(bound.fragment),
                        "artifact_sha256": "0" * 64,
                    }
                ),
            )
            if index == 0
            else bound
        )
        for index, bound in enumerate(layer17_executor._bound_nodes)
    )
    monkeypatch.setattr(capture, "SameLayerFragmentSelection", _FakeSelection)
    monkeypatch.setattr(capture, "Gemma3ModalGeneratorGraphExecutor", _FakeExecutor)
    monkeypatch.setattr(
        capture,
        "_capture_rows_and_full_output",
        lambda *_args, **_kwargs: pytest.fail("rows must not be collected"),
    )

    with pytest.raises(ValueError, match="topology differs"):
        capture.capture_gemma3_layer17_native_and_layer10_rows(
            adapter,  # type: ignore[arg-type]
            _batches(),
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
            layer10_executor=layer10_executor,  # type: ignore[arg-type]
            layer17_executor=layer17_executor,  # type: ignore[arg-type]
        )


def test_capture_rejects_nonidentical_compiled_inputs_before_compact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter()
    layer10_executor = _FakeExecutor(adapter)
    native_rows = _aligned_rows(offset=0.0)
    compiled_rows = _aligned_rows(offset=0.25)
    compiled_rows.rows_by_fragment[LAYER17_FRAGMENT_IDS[1]].inputs[0, 0] += 1.0
    full = torch.zeros((3, 3), dtype=torch.float64)
    layer17_executor = _FakeExecutor(
        adapter,
        affected=(17,),
        compact_output=full,
    )
    _install_synthetic_capture(
        monkeypatch,
        layer10_executor,
        native_rows=native_rows,
        compiled_rows=compiled_rows,
        native_full=full,
        compiled_full=full,
    )

    with pytest.raises(ValueError, match="compiled fragment inputs"):
        capture.capture_gemma3_layer17_native_and_layer10_rows(
            adapter,  # type: ignore[arg-type]
            _batches(),
            selection=_FakeSelection(),  # type: ignore[arg-type]
            leaf_activation_site=INPUT_SITE,
            layer10_executor=layer10_executor,  # type: ignore[arg-type]
            layer17_executor=layer17_executor,  # type: ignore[arg-type]
        )
    assert layer17_executor.compact_calls == []
