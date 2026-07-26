from dataclasses import replace

import pytest
import torch

from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.structured_layer_distillation import (
    StructuredLayerDistillationScales,
    StructuredLayerDistillationWeights,
    StructuredLayerProvenance,
    StructuredLayerTargets,
    StructuredOutputFisherMetric,
    capture_structured_layer_targets,
    estimate_structured_layer_scales,
    initialize_structured_rmsnorms_from_targets_,
    structured_layer_distillation_loss,
    structured_layer_provenance,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)

from test_structured_transformer_layer_executor import (
    _layer_spec,
    _sequence,
)


def _targets_from_teacher(
    teacher: StructuredTransformerLayerExecutor,
    hidden: torch.Tensor,
    mask: torch.Tensor,
) -> StructuredLayerTargets:
    sequence = _sequence(mask)
    with torch.no_grad():
        result = teacher.forward_components(hidden, sequence)
    return StructuredLayerTargets(
        provenance=StructuredLayerProvenance(
            layer_id="layer.0",
            output_site="layer.0.output",
            source_segment_fingerprint=teacher.execution_fingerprint(),
        ),
        sequence=sequence,
        block_input=hidden.detach().clone(),
        normalized_attention_input=(
            result.normalized_attention_input.detach().clone()
        ),
        attention_operator_output=(
            result.attention_operator_output.detach().clone()
        ),
        attention_delta=result.attention_delta.detach().clone(),
        post_attention=result.post_attention.detach().clone(),
        normalized_feed_forward_input=(
            result.normalized_feed_forward_input.detach().clone()
        ),
        feed_forward_operator_output=(
            result.feed_forward_operator_output.detach().clone()
        ),
        feed_forward_delta=result.feed_forward_delta.detach().clone(),
        output=result.output.detach().clone(),
        teacher_logits=None,
    )


def test_structured_loss_supervises_stages_and_full_fisher_metric() -> None:
    torch.manual_seed(913)
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    student = StructuredTransformerLayerExecutor(config)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    targets = _targets_from_teacher(teacher, hidden, mask)
    prediction = student.forward_components(hidden, targets.sequence)
    fisher_positions = mask.clone()
    fisher_positions[:, 0] = False
    weights = StructuredLayerDistillationWeights(output_fisher=0.5)
    loss = structured_layer_distillation_loss(
        prediction,
        targets,
        mask,
        weights=weights,
        fisher_positions=fisher_positions,
        output_fisher=StructuredOutputFisherMetric.from_raw_fisher(
            provenance=targets.provenance,
            calibration_split_sha256="a" * 64,
            delta_scale=torch.ones(8),
            raw_fisher=torch.eye(8),
        ),
    )

    assert loss.total.requires_grad
    assert loss.attention_delta.item() > 0
    assert loss.feed_forward_delta.item() > 0
    assert loss.output.item() > 0
    assert loss.output_fisher.item() > 0
    loss.total.backward()
    assert all(
        parameter.grad is not None
        for parameter in student.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in teacher.parameters()
    )


def test_structured_distillation_smoke_reduces_heldout_boundary_error() -> None:
    torch.manual_seed(117)
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec(
            attention_kind="global_causal",
            window_size=None,
        )
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    torch.manual_seed(991)
    student = StructuredTransformerLayerExecutor(config)
    train_hidden = tuple(
        torch.randn(4, 6, 8)
        for _ in range(8)
    )
    heldout_hidden = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)
    train_targets = tuple(
        _targets_from_teacher(teacher, values, mask)
        for values in train_hidden
    )
    scales = estimate_structured_layer_scales(
        train_targets,
        calibration_split_sha256="b" * 64,
    )
    assert isinstance(scales, StructuredLayerDistillationScales)
    assert scales.width == 8
    heldout_targets = _targets_from_teacher(teacher, heldout_hidden, mask)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=4e-3,
        weight_decay=0.0,
    )

    def heldout_mse() -> float:
        student.eval()
        with torch.no_grad():
            output = student(
                heldout_hidden,
                heldout_targets.sequence,
            )
            return float(
                (output - heldout_targets.output).square().mean().item()
            )

    initial = heldout_mse()
    student.train()
    for step in range(240):
        batch = step % len(train_hidden)
        optimizer.zero_grad(set_to_none=True)
        prediction = student.forward_components(
            train_hidden[batch],
            train_targets[batch].sequence,
        )
        loss = structured_layer_distillation_loss(
            prediction,
            train_targets[batch],
            mask,
            scales=scales,
        )
        loss.total.backward()
        optimizer.step()
    final = heldout_mse()

    assert final < initial * 0.50


def test_activation_derived_rmsnorm_initialization_recovers_teacher_gains_and_ignores_padding() -> None:
    torch.manual_seed(7_441)
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec(
            attention_kind="global_causal",
            window_size=None,
        )
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    norm_names = (
        "attention_input_norm",
        "attention_output_norm",
        "feed_forward_input_norm",
        "feed_forward_output_norm",
    )
    with torch.no_grad():
        for index, name in enumerate(norm_names):
            values = torch.linspace(
                -0.25 + index,
                8.0 + 11.0 * index,
                teacher.width,
            )
            getattr(teacher, name).weight.copy_(values)
    hidden = torch.randn(3, 6, teacher.width)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
        ]
    )
    targets = _targets_from_teacher(teacher, hidden, mask)
    changed = {}
    for field in (
        "block_input",
        "normalized_attention_input",
        "attention_operator_output",
        "attention_delta",
        "post_attention",
        "normalized_feed_forward_input",
        "feed_forward_operator_output",
        "feed_forward_delta",
        "output",
    ):
        value = getattr(targets, field).clone()
        value[~mask] = 1e6
        changed[field] = value
    targets = replace(targets, **changed)

    torch.manual_seed(1_997)
    student = StructuredTransformerLayerExecutor(config)
    untouched = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
        if not any(
            name == f"{norm_name}.weight"
            for norm_name in norm_names
        )
    }
    report = initialize_structured_rmsnorms_from_targets_(
        student,
        (targets,),
        calibration_split_sha256="9" * 64,
    )

    assert not student.owns_source_model_weights
    assert report["source_module_or_parameter_read"] is False
    assert report["direct_source_tensor_copy"] is False
    assert report["valid_rows"] == int(mask.sum().item())
    for name in norm_names:
        torch.testing.assert_close(
            getattr(student, name).weight,
            getattr(teacher, name).weight,
            rtol=1e-5,
            atol=1e-5,
        )
        norm_report = report["normalizations"][name]
        assert norm_report["identified_coordinates"] == student.width
        assert norm_report["fit_nrmse"] < 1e-6
    for name, expected in untouched.items():
        torch.testing.assert_close(student.state_dict()[name], expected)


def test_mixed_coordinate_and_global_energy_loss_matches_manual_formula() -> None:
    torch.manual_seed(7_119)
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    hidden = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, False]])
    targets = _targets_from_teacher(teacher, hidden, mask)
    prediction = teacher.forward_components(hidden, targets.sequence)
    error = torch.zeros_like(prediction.output)
    error[0, 0, 0] = 2.0
    error[0, 1, 1] = 12.0
    prediction = replace(
        prediction,
        output=prediction.output + error,
    )
    scales = estimate_structured_layer_scales(
        (targets,),
        calibration_split_sha256="8" * 64,
    )
    output_scale = torch.arange(1, 9, dtype=torch.float64)
    scales = replace(scales, output=output_scale)
    weights = StructuredLayerDistillationWeights(
        normalized_attention_input=0.0,
        attention_operator_output=0.0,
        attention_delta=0.0,
        post_attention=0.0,
        normalized_feed_forward_input=0.0,
        feed_forward_operator_output=0.0,
        feed_forward_delta=0.0,
        output=1.0,
        output_fisher=0.0,
    )
    loss = structured_layer_distillation_loss(
        prediction,
        targets,
        mask,
        weights=weights,
        scales=scales,
        coordinate_loss_weight=2.0,
        energy_loss_weight=3.0,
    )
    valid_error = error[mask].float()
    expected_coordinate = (
        valid_error / output_scale.float()
    ).square().mean()
    expected_energy = (
        valid_error.square().mean()
        / output_scale.float().square().mean()
    )

    torch.testing.assert_close(
        loss.coordinate_total,
        expected_coordinate,
    )
    torch.testing.assert_close(loss.energy_total, expected_energy)
    torch.testing.assert_close(
        loss.output,
        2.0 * expected_coordinate + 3.0 * expected_energy,
    )
    torch.testing.assert_close(loss.total, loss.output)


def test_structured_scales_floor_dormant_coordinates_relative_to_stage() -> None:
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    hidden = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    targets = _targets_from_teacher(teacher, hidden, mask)
    coordinate_rms = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    targets = replace(
        targets,
        normalized_attention_input=coordinate_rms.expand(1, 2, 8),
    )

    scales = estimate_structured_layer_scales(
        (targets,),
        calibration_split_sha256="c" * 64,
        floor=1e-4,
        relative_median_floor=0.1,
    )

    # torch.median uses the lower middle coordinate for an even width, so
    # the stage-relative floor is 0.1 * 3.0 rather than the absolute 1e-4.
    torch.testing.assert_close(
        scales.normalized_attention_input,
        torch.tensor(
            [0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize(
    "relative_floor",
    (-0.01, 1.01, float("nan"), True),
)
def test_structured_scales_reject_invalid_relative_median_floor(
    relative_floor: object,
) -> None:
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    targets = _targets_from_teacher(
        teacher,
        torch.zeros(1, 2, 8),
        torch.ones(1, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="relative median"):
        estimate_structured_layer_scales(
            (targets,),
            calibration_split_sha256="d" * 64,
            relative_median_floor=relative_floor,  # type: ignore[arg-type]
        )


def test_output_fisher_metric_validates_once_and_canonicalizes() -> None:
    provenance = StructuredLayerProvenance(
        layer_id="layer.0",
        output_site="layer.0.output",
        source_segment_fingerprint="c" * 64,
    )
    scale = torch.arange(1, 9, dtype=torch.float32)
    raw_metric = torch.diag(torch.arange(1, 9, dtype=torch.float32))
    output_fisher = StructuredOutputFisherMetric(
        provenance=provenance,
        calibration_split_sha256="d" * 64,
        delta_scale=scale,
        standardized_coordinate_metric=raw_metric,
    )

    assert output_fisher.width == 8
    assert output_fisher.delta_scale.dtype is torch.float64
    assert (
        output_fisher.standardized_coordinate_metric.dtype
        is torch.float64
    )
    assert output_fisher.delta_scale.device.type == "cpu"
    scale.zero_()
    raw_metric.zero_()
    assert bool((output_fisher.delta_scale > 0).all())
    assert float(
        output_fisher.standardized_coordinate_metric.trace().item()
    ) == 36.0


@pytest.mark.parametrize(
    ("scale", "metric", "message"),
    (
        (
            torch.tensor([1.0, float("nan")]),
            torch.eye(2),
            "finite",
        ),
        (
            torch.tensor([1.0, 0.0]),
            torch.eye(2),
            "strictly positive",
        ),
        (
            torch.ones(2),
            torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
            "symmetric",
        ),
        (
            torch.ones(2),
            torch.diag(torch.tensor([1.0, -0.25])),
            "positive semidefinite",
        ),
        (
            torch.ones(2),
            torch.zeros(2, 2),
            "nonzero sensitivity",
        ),
        (
            torch.ones(640),
            torch.diag(
                torch.cat(
                    (
                        torch.ones(639),
                        torch.tensor([-1e-6]),
                    )
                )
            ),
            "positive semidefinite",
        ),
    ),
)
def test_output_fisher_metric_rejects_invalid_quadratics(
    scale: torch.Tensor,
    metric: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StructuredOutputFisherMetric(
            provenance=StructuredLayerProvenance(
                layer_id="layer.0",
                output_site="layer.0.output",
                source_segment_fingerprint="e" * 64,
            ),
            calibration_split_sha256="f" * 64,
            delta_scale=scale,
            standardized_coordinate_metric=metric,
        )


def test_raw_fisher_transform_preserves_nonuniform_dense_quadratic() -> None:
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    hidden = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    targets = _targets_from_teacher(teacher, hidden, mask)
    prediction = teacher.forward_components(hidden, targets.sequence)
    error = torch.tensor(
        [0.25, -0.50, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0]
    )
    changed_output = prediction.output.detach().clone()
    changed_output[0, 2] += error
    prediction = replace(prediction, output=changed_output)
    scale = torch.arange(1, 9, dtype=torch.float64)
    factor = torch.arange(1, 65, dtype=torch.float64).reshape(8, 8)
    raw_fisher = factor.transpose(0, 1) @ factor
    output_fisher = StructuredOutputFisherMetric.from_raw_fisher(
        provenance=targets.provenance,
        calibration_split_sha256="1" * 64,
        delta_scale=scale,
        raw_fisher=raw_fisher,
    )
    fisher_positions = torch.tensor(
        [[False, False, True]],
        dtype=torch.bool,
    )
    loss = structured_layer_distillation_loss(
        prediction,
        targets,
        mask,
        fisher_positions=fisher_positions,
        output_fisher=output_fisher,
    )
    expected = (
        error.double()
        @ raw_fisher
        @ error.double()
        / error.numel()
    )

    torch.testing.assert_close(
        loss.output_fisher.double(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_structured_loss_rejects_cross_layer_calibration_state() -> None:
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    hidden = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    targets = _targets_from_teacher(teacher, hidden, mask)
    prediction = teacher.forward_components(hidden, targets.sequence)
    scales = estimate_structured_layer_scales(
        (targets,),
        calibration_split_sha256="2" * 64,
    )
    wrong_provenance = replace(
        targets.provenance,
        layer_id="layer.1",
        output_site="layer.1.output",
    )

    with pytest.raises(ValueError, match="provenance"):
        structured_layer_distillation_loss(
            prediction,
            targets,
            mask,
            scales=replace(scales, provenance=wrong_provenance),
        )


def test_structured_loss_rejects_native_padding_supervision() -> None:
    config = StructuredTransformerLayerExecutorConfig.from_layer_spec(
        _layer_spec()
    )
    teacher = StructuredTransformerLayerExecutor(config).eval()
    student = StructuredTransformerLayerExecutor(config).eval()
    hidden = torch.randn(1, 4, 8)
    captured_mask = torch.tensor([[True, True, False, False]])
    targets = _targets_from_teacher(teacher, hidden, captured_mask)
    prediction = student.forward_components(hidden, targets.sequence)

    with pytest.raises(ValueError, match="subset"):
        structured_layer_distillation_loss(
            prediction,
            targets,
            torch.ones_like(captured_mask),
        )


def test_optional_real_gemma_capture_exposes_exact_residual_stages() -> None:
    try:
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    except ImportError:
        pytest.skip("optional Transformers dependency is not installed")

    torch.manual_seed(612)
    config = Gemma3TextConfig(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        sliding_window=4,
        layer_types=["sliding_attention"],
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    adapter = Gemma3CausalLMAdapter(
        Gemma3ForCausalLM(config).eval()
    )
    inputs = {
        "input_ids": torch.tensor([[1, 7, 3, 9, 2]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
    }
    live_provenance = structured_layer_provenance(adapter, "layer.0")
    with pytest.raises(ValueError, match="provenance"):
        capture_structured_layer_targets(
            adapter,
            "layer.0",
            inputs,
            provenance=replace(
                live_provenance,
                source_segment_fingerprint="0" * 64,
            ),
        )
    targets = capture_structured_layer_targets(
        adapter,
        "layer.0",
        inputs,
        teacher_logit_positions=inputs["attention_mask"],
        provenance=live_provenance,
    )

    torch.testing.assert_close(
        targets.post_attention,
        targets.block_input + targets.attention_delta,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        targets.output,
        targets.post_attention + targets.feed_forward_delta,
        rtol=0.0,
        atol=0.0,
    )
    assert targets.normalized_attention_input.shape == (1, 5, 16)
    assert targets.attention_operator_output.shape == (1, 5, 16)
    assert targets.teacher_logits is not None
    assert targets.teacher_logits.shape == (5, 48)
