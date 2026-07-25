from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from fisher_graph.associative import (
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
)
from fisher_graph.fused_executor import (
    FusedToyTransformer,
    PackedTriangularFusedTwoLayerModalStack,
    load_lazy_fused_modal_stack,
)
from fisher_graph.mlx_benchmark import benchmark_mlx_modal_stack
from fisher_graph.mlx_executor import (
    MLXFusedToyTransformer,
    MLXModalStackOutput,
    MLXPackedTriangularFusedTwoLayerModalStack,
    mlx_array_from_torch,
    mlx_array_to_numpy,
    mlx_is_installed,
    mlx_metal_is_usable,
    mlx_runtime_provenance,
)
from fisher_graph.training import load_checkpoint


ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "associative_recall"
)


_METAL_USABLE = mlx_metal_is_usable()
pytestmark = pytest.mark.skipif(
    not _METAL_USABLE,
    reason="MLX Metal execution is unavailable",
)
mx = pytest.importorskip("mlx.core") if _METAL_USABLE else None


@dataclass(frozen=True)
class _RuntimeBundle:
    lazy: object
    packed: PackedTriangularFusedTwoLayerModalStack
    torch_model: FusedToyTransformer
    mlx_stack: MLXPackedTriangularFusedTwoLayerModalStack
    mlx_model: MLXFusedToyTransformer


@pytest.fixture(scope="module")
def runtimes() -> _RuntimeBundle:
    lazy, _, _ = load_lazy_fused_modal_stack(
        ARTIFACTS / "fused_modal_runtime.pt",
        sidecar_root=ARTIFACTS,
    )
    status_before = lazy.instrumentation_status()
    packed = PackedTriangularFusedTwoLayerModalStack.from_lazy(
        lazy
    ).eval()
    teacher, _ = load_checkpoint(ARTIFACTS / "checkpoint.pt")
    teacher.eval()
    torch_model = FusedToyTransformer.from_teacher(
        teacher,
        packed,
    ).eval()
    mlx_stack = MLXPackedTriangularFusedTwoLayerModalStack.from_torch(
        packed,
        backend="metal",
    )
    mlx_model = MLXFusedToyTransformer.from_torch(
        torch_model,
        stack=mlx_stack,
        backend="metal",
    )
    assert lazy.instrumentation_status() == status_before
    return _RuntimeBundle(
        lazy=lazy,
        packed=packed,
        torch_model=torch_model,
        mlx_stack=mlx_stack,
        mlx_model=mlx_model,
    )


def _torch_hidden(
    packed: PackedTriangularFusedTwoLayerModalStack,
    batch_size: int,
    *,
    seed: int,
) -> torch.Tensor:
    return packed.first_input_mean + 0.05 * torch.randn(
        batch_size,
        packed.sequence_length,
        packed.width,
        generator=torch.Generator().manual_seed(seed),
    )


def test_mlx_optional_package_and_state_conversion_are_explicit(
    runtimes: _RuntimeBundle,
) -> None:
    assert mlx_is_installed()
    stack = runtimes.mlx_stack
    torch_state = {
        name: value
        for name, value in runtimes.packed.state_dict().items()
        if value.is_floating_point()
    }

    assert set(stack.state_dict()) == set(torch_state)
    for name, value in stack.state_dict().items():
        converted = mlx_array_to_numpy(value)
        source = torch_state[name].numpy()
        assert converted.dtype == np.float32
        assert converted.shape == source.shape
        assert np.array_equal(converted, source)
    assert not hasattr(stack, "causal_target_indices")
    assert not hasattr(stack, "causal_source_indices")
    assert stack.state_bytes == 124_544
    assert runtimes.mlx_model.state_bytes == 130_688

    provenance = mlx_runtime_provenance(stack)
    assert provenance["backend"] == "metal"
    assert provenance["default_backend"] == "metal"
    assert provenance["available_execution_backends"] == [
        "eager",
        "compiled",
        "metal",
    ]
    assert provenance["causal_pair_count"] == 36
    assert (
        provenance["causal_pair_order"]
        == "target_major_lower_triangle"
    )
    assert provenance["weights_updated"] is False
    assert provenance["serialized_artifact"] is False
    assert provenance["supports_activation_capture"] is True
    assert provenance["capture_backend"] == "ordinary_mlx_graph"
    assert provenance["metal_kernel"] == "fisher_packed_causal_gelu"
    assert len(provenance["source_float_state_sha256"]) == 64
    assert provenance["state_matches_source"] is True
    assert (
        provenance["state_sha256"]
        == provenance["source_float_state_sha256"]
    )
    assert (
        provenance["source_provenance"]
        == runtimes.packed.runtime_provenance()
    )

    reordered = copy.deepcopy(runtimes.packed)
    reordered.causal_source_indices = torch.flip(
        reordered.causal_source_indices,
        dims=(0,),
    )
    with pytest.raises(ValueError, match="canonical target-major"):
        MLXPackedTriangularFusedTwoLayerModalStack.from_torch(
            reordered,
            backend="metal",
        )


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_mlx_reference_compiled_and_metal_preserve_stack_behavior(
    runtimes: _RuntimeBundle,
    batch_size: int,
) -> None:
    hidden = _torch_hidden(
        runtimes.packed,
        batch_size,
        seed=1700 + batch_size,
    )
    with torch.no_grad():
        expected = runtimes.packed(hidden).numpy()
    mlx_hidden = mlx_array_from_torch(hidden)

    actual: dict[str, np.ndarray] = {}
    for backend in ("eager", "compiled", "metal"):
        actual[backend] = mlx_array_to_numpy(
            runtimes.mlx_stack(mlx_hidden, backend=backend)
        )
        assert actual[backend].shape == expected.shape
        assert actual[backend].dtype == np.float32
        assert np.isfinite(actual[backend]).all()

    for backend, output in actual.items():
        relative_rms = np.linalg.norm(output - expected) / np.linalg.norm(
            expected
        )
        maximum_difference = np.max(np.abs(output - expected))
        assert relative_rms < (0.004 if backend != "metal" else 0.002)
        assert maximum_difference < (5.0 if backend != "metal" else 2.0)


def test_mlx_metal_is_causal_batch_independent_and_deterministic(
    runtimes: _RuntimeBundle,
) -> None:
    hidden = _torch_hidden(runtimes.packed, 3, seed=1801)
    mlx_hidden = mlx_array_from_torch(hidden)
    with pytest.raises(ValueError, match="batch must be nonempty"):
        runtimes.mlx_stack(mlx_hidden[:0], backend="metal")
    baseline = mlx_array_to_numpy(
        runtimes.mlx_stack(mlx_hidden, backend="metal")
    )

    repeated = [
        mlx_array_to_numpy(
            runtimes.mlx_stack(mlx_hidden, backend="metal")
        )
        for _ in range(10)
    ]
    assert all(np.array_equal(baseline, value) for value in repeated)

    single = mlx_array_to_numpy(
        runtimes.mlx_stack(
            mlx_array_from_torch(hidden[:1]),
            backend="metal",
        )
    )
    batch_relative_rms = np.linalg.norm(
        single[0] - baseline[0]
    ) / np.linalg.norm(baseline[0])
    assert batch_relative_rms < 0.002

    for last_visible in (0, 3, 6):
        changed = hidden.clone()
        changed[:, last_visible + 1 :] += torch.randn(
            changed[:, last_visible + 1 :].shape,
            generator=torch.Generator().manual_seed(
                1900 + last_visible
            ),
        )
        changed_output = mlx_array_to_numpy(
            runtimes.mlx_stack(
                mlx_array_from_torch(changed),
                backend="metal",
            )
        )
        assert np.array_equal(
            baseline[:, : last_visible + 1],
            changed_output[:, : last_visible + 1],
        )

    class OversizedValues:
        shape = (
            2**32,
            runtimes.mlx_stack.sequence_length,
            runtimes.mlx_stack.width,
        )
        dtype = mx.float32

    with pytest.raises(ValueError, match="32-bit index limit"):
        runtimes.mlx_stack._causal_stage_metal(
            OversizedValues(),
            runtimes.mlx_stack.packed_first_input_kernel,
            runtimes.mlx_stack.first_hidden_bias,
        )


def test_mlx_capture_routes_to_differentiable_reference_graph(
    runtimes: _RuntimeBundle,
) -> None:
    hidden = _torch_hidden(runtimes.packed, 2, seed=2001)
    mlx_hidden = mlx_array_from_torch(hidden)
    captured = runtimes.mlx_stack(
        mlx_hidden,
        capture_activations=True,
    )

    assert isinstance(captured, MLXModalStackOutput)
    assert set(captured.activations) == {
        "layer.0.modal.hidden",
        "layer.1.modal.hidden",
        "layer.1.output",
    }
    eager = runtimes.mlx_stack(mlx_hidden, backend="eager")
    assert np.array_equal(
        mlx_array_to_numpy(captured.output),
        mlx_array_to_numpy(eager),
    )

    def loss_fn(value):
        output = runtimes.mlx_stack(value, backend="eager")
        return mx.mean(mx.square(output))

    gradient = mx.grad(loss_fn)(mlx_hidden)
    gradient_numpy = mlx_array_to_numpy(gradient)
    assert gradient_numpy.shape == hidden.shape
    assert np.isfinite(gradient_numpy).all()
    assert np.linalg.norm(gradient_numpy) > 0


def test_mlx_full_model_preserves_validation_behavior(
    runtimes: _RuntimeBundle,
) -> None:
    report = json.loads(
        (ARTIFACTS / "fisher_report.json").read_text()
    )
    task = AssociativeRecallTaskConfig(**report["task"]["config"])
    validation = build_associative_recall_splits(task).validation
    expected = associative_recall_answer_logits(
        runtimes.torch_model,
        validation,
    )
    mlx_inputs = mlx_array_from_torch(validation.input_ids)

    outputs: dict[str, torch.Tensor] = {}
    for backend in ("eager", "compiled", "metal"):
        logits = runtimes.mlx_model(
            mlx_inputs,
            backend=backend,
        ).logits
        outputs[backend] = torch.from_numpy(
            mlx_array_to_numpy(logits)[:, -1]
        )
        assert torch.isfinite(outputs[backend]).all()
        assert torch.equal(
            outputs[backend].argmax(dim=-1),
            expected.argmax(dim=-1),
        )
        metrics = associative_recall_metrics_from_logits(
            validation,
            outputs[backend],
        )
        expected_metrics = associative_recall_metrics_from_logits(
            validation,
            expected,
        )
        assert metrics.answer_accuracy == expected_metrics.answer_accuracy
        assert (
            metrics.paired_context_accuracy
            == expected_metrics.paired_context_accuracy
        )
        assert abs(metrics.hard_nll - expected_metrics.hard_nll) < 3e-4
        maximum_logit_difference = torch.max(
            torch.abs(outputs[backend] - expected)
        )
        limit = 0.01 if backend == "metal" else 0.025
        assert maximum_logit_difference < limit


def test_mlx_model_capture_and_input_guards(
    runtimes: _RuntimeBundle,
) -> None:
    tokens = torch.randint(
        runtimes.torch_model.config.vocab_size,
        (2, 8),
        generator=torch.Generator().manual_seed(2101),
    )
    mlx_tokens = mlx_array_from_torch(tokens)
    captured = runtimes.mlx_model(
        mlx_tokens,
        capture_activations=True,
    )
    assert captured.activations is not None
    assert set(captured.activations) == {
        "embedding.token",
        "embedding.position",
        "embedding.output",
        "layer.0.modal.hidden",
        "layer.1.modal.hidden",
        "layer.1.output",
        "final_norm",
        "logits",
    }
    assert np.isfinite(mlx_array_to_numpy(captured.logits)).all()

    with pytest.raises(ValueError, match="sequence length"):
        runtimes.mlx_model(
            mlx_array_from_torch(tokens[:, :-1]),
        )
    with pytest.raises(ValueError, match="batch must be nonempty"):
        runtimes.mlx_model(
            mlx_array_from_torch(tokens[:0]),
        )
    with pytest.raises(ValueError, match="integer index dtype"):
        runtimes.mlx_model(
            mlx_array_from_torch(tokens.float()),
        )
    negative_tokens = tokens.clone()
    negative_tokens[0, 0] = -1
    with pytest.raises(ValueError, match="configured vocabulary"):
        runtimes.mlx_model(mlx_array_from_torch(negative_tokens))
    out_of_range_tokens = tokens.clone()
    out_of_range_tokens[0, 0] = runtimes.torch_model.config.vocab_size
    with pytest.raises(ValueError, match="configured vocabulary"):
        runtimes.mlx_model(mlx_array_from_torch(out_of_range_tokens))
    with pytest.raises(ValueError, match="does not support padding"):
        runtimes.mlx_model(
            mlx_tokens,
            attention_mask=mlx_array_from_torch(
                torch.tensor(
                    [[True] * 8, [True] * 7 + [False]]
                )
            ),
        )
    with pytest.raises(ValueError, match="does not yet support"):
        runtimes.mlx_model(
            mlx_tokens,
            activation_interventions={"layer.0.modal.hidden": object()},
        )

    foreign_source = copy.deepcopy(runtimes.torch_model)
    foreign_source.stack.packed_bridge_kernel[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="not derived from"):
        MLXFusedToyTransformer.from_torch(
            foreign_source,
            stack=runtimes.mlx_stack,
            backend="metal",
        )

    original_bridge = runtimes.mlx_stack._packed_bridge_kernel
    with pytest.raises(AttributeError, match="immutable"):
        runtimes.mlx_stack.packed_bridge_kernel = original_bridge + 1.0
    try:
        object.__setattr__(
            runtimes.mlx_stack,
            "_packed_bridge_kernel",
            original_bridge + 1.0,
        )
        with pytest.raises(ValueError, match="not derived from"):
            MLXFusedToyTransformer.from_torch(
                runtimes.torch_model,
                stack=runtimes.mlx_stack,
                backend="metal",
            )
    finally:
        object.__setattr__(
            runtimes.mlx_stack,
            "_packed_bridge_kernel",
            original_bridge,
        )

    with pytest.raises(AttributeError, match="immutable"):
        runtimes.mlx_model.final_norm_eps = 1.0
    with pytest.raises(AttributeError, match="immutable"):
        runtimes.mlx_model.stack = runtimes.mlx_stack
    with pytest.raises(AttributeError, match="immutable"):
        runtimes.mlx_stack.backend = "eager"
    with pytest.raises(AttributeError, match="immutable"):
        runtimes.mlx_model.backend = "eager"

    provenance_before = mlx_runtime_provenance(runtimes.mlx_stack)
    exported_state = runtimes.mlx_stack.state_dict()
    exported_state["packed_bridge_kernel"][0, 0, 0] = 123.0
    public_weight = runtimes.mlx_stack.packed_bridge_kernel
    public_weight[0, 0, 0] = 456.0
    mx.eval(exported_state["packed_bridge_kernel"], public_weight)
    provenance_after = mlx_runtime_provenance(runtimes.mlx_stack)
    assert (
        provenance_after["state_sha256"]
        == provenance_before["state_sha256"]
    )


def test_mlx_benchmark_self_validates_exact_measured_outputs(
    runtimes: _RuntimeBundle,
) -> None:
    report = benchmark_mlx_modal_stack(
        runtimes.mlx_stack,
        batch_sizes=(1,),
        rounds=1,
        warmup_calls=0,
        minimum_warmup_seconds=0.0,
        iterations_per_round=1,
    )

    assert report["output_validation"]["gate_passed"] is True
    batch = report["output_validation"]["batches"][0]
    assert batch["batch_size"] == 1
    assert set(batch["systems"]) == {
        "mlx_dense_compiled",
        "mlx_packed_compiled",
        "mlx_packed_metal",
    }
