import copy
from dataclasses import replace

import pytest
import torch

from fisher_graph.activations import ActivationTrace
from fisher_graph.adapters import StructuredOperatorSites
from fisher_graph.structured_layer_distillation import (
    StructuredLayerProvenance,
    StructuredLayerTargets,
    refit_structured_terminal_projections_from_targets_,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)

from test_structured_transformer_layer_executor import (
    _layer_spec,
    _sequence,
)


def _config(
    *,
    projection_bias: bool = False,
    operator_sites: bool = False,
) -> StructuredTransformerLayerExecutorConfig:
    layer = _layer_spec(
        attention_kind="global_causal",
        window_size=None,
    )
    assert layer.transformer is not None
    sites = (
        StructuredOperatorSites(
            attention_query_projection="layer.0.attention.query_projection",
            attention_query_normalized="layer.0.attention.query_normalized",
            attention_key_projection="layer.0.attention.key_projection",
            attention_key_normalized="layer.0.attention.key_normalized",
            attention_value_projection="layer.0.attention.value_projection",
            attention_context="layer.0.attention.context",
            feed_forward_gate_projection="layer.0.mlp.gate_projection",
            feed_forward_up_projection="layer.0.mlp.up_projection",
            feed_forward_down_input="layer.0.mlp.down_input",
        )
        if operator_sites
        else None
    )
    transformer = replace(
        layer.transformer,
        attention_projection_bias=projection_bias,
        feed_forward=replace(
            layer.transformer.feed_forward,
            projection_bias=projection_bias,
        ),
        operator_sites=sites,
    )
    return StructuredTransformerLayerExecutorConfig.from_layer_spec(
        replace(layer, transformer=transformer)
    )


def _targets_from_executor(
    executor: StructuredTransformerLayerExecutor,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    *,
    provenance: StructuredLayerProvenance,
) -> StructuredLayerTargets:
    sequence = _sequence(mask)
    with torch.no_grad():
        result = executor.forward_components(hidden, sequence)
    return StructuredLayerTargets(
        provenance=provenance,
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
        attention_projection_input=(
            result.attention_projection_input.detach().clone()
        ),
        feed_forward_projection_input=(
            result.feed_forward_projection_input.detach().clone()
        ),
    )


def _teacher_student_and_targets(
    *,
    projection_bias: bool,
    masks: tuple[torch.Tensor, ...],
) -> tuple[
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutor,
    tuple[StructuredLayerTargets, ...],
]:
    torch.manual_seed(91_771)
    teacher = StructuredTransformerLayerExecutor(
        _config(projection_bias=projection_bias)
    ).eval()
    student = copy.deepcopy(teacher).eval()
    with torch.no_grad():
        student.attention.o_proj.weight.zero_()
        student.feed_forward.down_proj.weight.zero_()
        if student.attention.o_proj.bias is not None:
            student.attention.o_proj.bias.zero_()
        if student.feed_forward.down_proj.bias is not None:
            student.feed_forward.down_proj.bias.zero_()
    provenance = StructuredLayerProvenance(
        layer_id="layer.0",
        output_site="layer.0.output",
        source_segment_fingerprint=teacher.execution_fingerprint(),
    )
    generator = torch.Generator().manual_seed(220_019)
    targets = tuple(
        _targets_from_executor(
            teacher,
            torch.randn(
                mask.shape[0],
                mask.shape[1],
                teacher.width,
                generator=generator,
            ),
            mask,
            provenance=provenance,
        )
        for mask in masks
    )
    return teacher, student, targets


def test_executor_exposes_both_pre_terminal_projection_features() -> None:
    torch.manual_seed(4_093)
    executor = StructuredTransformerLayerExecutor(_config()).eval()
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    hidden = torch.randn(2, 4, executor.width)
    sequence = _sequence(mask)
    trace = ActivationTrace(retain_grad=False)
    result = executor.forward_components(
        hidden,
        sequence,
        trace=trace,
    )

    attention_features = executor.attention_projection_features(
        result.normalized_attention_input,
        sequence,
    )
    feed_forward_features = (
        executor.feed_forward_projection_features(
            result.normalized_feed_forward_input
        )
    )
    torch.testing.assert_close(
        result.attention_projection_input,
        attention_features,
    )
    torch.testing.assert_close(
        result.feed_forward_projection_input,
        feed_forward_features,
    )
    torch.testing.assert_close(
        result.attention_operator_output,
        executor.attention.o_proj(result.attention_projection_input),
    )
    torch.testing.assert_close(
        result.feed_forward_operator_output,
        executor.feed_forward.down_proj(
            result.feed_forward_projection_input
        ),
    )
    assert trace["structured_layer.attention.pre_o_proj"].shape == (
        2,
        4,
        executor.attention.o_proj.in_features,
    )
    assert trace["structured_layer.mlp.pre_down_proj"].shape == (
        2,
        4,
        executor.feed_forward.down_proj.in_features,
    )


def test_structured_operator_sites_strict_config_and_artifact_roundtrip() -> None:
    config = _config(operator_sites=True)
    raw = config.to_dict()
    restored_config = StructuredTransformerLayerExecutorConfig.from_dict(
        raw
    )
    assert restored_config == config
    assert restored_config.transformer.operator_sites is not None
    legacy = copy.deepcopy(raw)
    legacy_transformer = legacy["transformer"]
    assert isinstance(legacy_transformer, dict)
    legacy_transformer.pop("operator_sites")
    restored_legacy = (
        StructuredTransformerLayerExecutorConfig.from_dict(legacy)
    )
    assert restored_legacy.transformer.operator_sites is None
    assert restored_legacy.to_dict() == legacy
    legacy_executor = StructuredTransformerLayerExecutor(
        restored_legacy
    ).eval()
    legacy_state = legacy_executor.artifact_state_dict()
    legacy_roundtrip = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            legacy_state
        )
    )
    assert legacy_roundtrip.config.to_dict() == legacy
    assert (
        legacy_roundtrip.execution_fingerprint()
        == legacy_state["execution_fingerprint"]
    )

    for mutation in ("missing", "extra"):
        invalid = copy.deepcopy(raw)
        sites = invalid["transformer"]["operator_sites"]
        assert isinstance(sites, dict)
        if mutation == "missing":
            sites.pop("attention_context")
        else:
            sites["unexpected"] = "layer.0.unexpected"
        with pytest.raises(
            ValueError,
            match="structured operator sites fields are invalid",
        ):
            StructuredTransformerLayerExecutorConfig.from_dict(invalid)

    executor = StructuredTransformerLayerExecutor(config).eval()
    restored = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        executor.artifact_state_dict()
    )
    assert restored.config == config
    assert restored.execution_fingerprint() == executor.execution_fingerprint()


def test_terminal_projection_ridge_refit_recovers_biased_operators() -> None:
    masks = (
        torch.ones(4, 7, dtype=torch.bool),
        torch.ones(3, 6, dtype=torch.bool),
        torch.ones(5, 5, dtype=torch.bool),
        torch.ones(2, 8, dtype=torch.bool),
    )
    teacher, student, targets = _teacher_student_and_targets(
        projection_bias=True,
        masks=masks,
    )
    student.train()
    with torch.no_grad():
        for index, module in enumerate(
            (
                student.attention.q_proj,
                student.attention.k_proj,
                student.attention.v_proj,
                student.feed_forward.gate_proj,
                student.feed_forward.up_proj,
            ),
            start=1,
        ):
            module.weight.add_(0.25 * index)
            if module.bias is not None:
                module.bias.sub_(0.10 * index)
    untouched = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
        if not (
            name.startswith("attention.o_proj.")
            or name.startswith("feed_forward.down_proj.")
        )
    }

    report = refit_structured_terminal_projections_from_targets_(
        student,
        targets,
        calibration_split_sha256="d" * 64,
        ridge=1e-10,
    )

    assert report["valid_rows"] == sum(
        int(mask.sum().item()) for mask in masks
    )
    assert report["provenance"] == {
        "layer_id": targets[0].provenance.layer_id,
        "output_site": targets[0].provenance.output_site,
        "source_segment_fingerprint": (
            targets[0].provenance.source_segment_fingerprint
        ),
    }
    assert report["regularization"] == {
        "ridge": 1e-10,
        "objective": (
            "summed_squared_operator_error_plus_ridge_weight_l2"
        ),
        "bias_regularized": False,
    }
    for name in ("attention.o_proj", "feed_forward.down_proj"):
        projection = report["projections"][name]
        assert projection["bias_fitted"] is True
        assert projection["normal_equations_dtype"] == "float64"
        assert projection["normal_equations_device"] == "cpu"
        assert projection["pre_refit_operator_nrmse"] > 0.9
        assert projection["post_refit_operator_nrmse"] < 1e-5
        assert (
            projection["post_refit_operator_nrmse"]
            < projection["pre_refit_operator_nrmse"] * 1e-4
        )
    torch.testing.assert_close(
        student.attention.o_proj.weight,
        teacher.attention.o_proj.weight,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        student.attention.o_proj.bias,
        teacher.attention.o_proj.bias,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        student.feed_forward.down_proj.weight,
        teacher.feed_forward.down_proj.weight,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        student.feed_forward.down_proj.bias,
        teacher.feed_forward.down_proj.bias,
        rtol=1e-4,
        atol=1e-5,
    )
    for name, expected in untouched.items():
        torch.testing.assert_close(
            student.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert not student.owns_source_model_weights
    assert student.training
    assert report["source_module_or_parameter_read"] is False
    assert report["direct_source_tensor_copy"] is False
    assert report["source_weight_origin_before"] is False
    assert report["source_weight_origin_after"] is False


def test_terminal_projection_refit_excludes_padding_and_is_deterministic() -> None:
    masks = (
        torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, False, False, False],
            ]
        ),
        torch.tensor(
            [
                [True, True, True, True, True, False],
                [True, True, False, False, False, False],
            ]
        ),
    )
    _teacher, base_student, targets = _teacher_student_and_targets(
        projection_bias=False,
        masks=masks,
    )
    poisoned_targets = []
    for target in targets:
        invalid = ~target.sequence.query_valid_mask
        replacements = {}
        for name in (
            "attention_projection_input",
            "attention_operator_output",
            "feed_forward_projection_input",
            "feed_forward_operator_output",
        ):
            value = getattr(target, name).clone()
            value[invalid] = 100_000.0
            replacements[name] = value
        poisoned_targets.append(replace(target, **replacements))

    clean = copy.deepcopy(base_student).eval()
    poisoned = copy.deepcopy(base_student).eval()
    repeated = copy.deepcopy(base_student).eval()
    clean_report = refit_structured_terminal_projections_from_targets_(
        clean,
        targets,
        calibration_split_sha256="e" * 64,
        ridge=1e-7,
    )
    poisoned_report = refit_structured_terminal_projections_from_targets_(
        poisoned,
        tuple(poisoned_targets),
        calibration_split_sha256="e" * 64,
        ridge=1e-7,
    )
    repeated_report = refit_structured_terminal_projections_from_targets_(
        repeated,
        targets,
        calibration_split_sha256="e" * 64,
        ridge=1e-7,
    )

    assert clean_report == poisoned_report
    assert clean_report == repeated_report
    assert clean_report["valid_rows"] == sum(
        int(mask.sum().item()) for mask in masks
    )
    for name, value in clean.state_dict().items():
        torch.testing.assert_close(
            poisoned.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            repeated.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_terminal_projection_refit_rejects_source_weight_origin() -> None:
    mask = torch.ones(2, 5, dtype=torch.bool)
    _teacher, executor, targets = _teacher_student_and_targets(
        projection_bias=False,
        masks=(mask,),
    )
    without_projection_inputs = replace(
        targets[0],
        attention_projection_input=None,
        feed_forward_projection_input=None,
    )
    with pytest.raises(
        ValueError,
        match="contain both projection inputs",
    ):
        refit_structured_terminal_projections_from_targets_(
            executor,
            (without_projection_inputs,),
            calibration_split_sha256="f" * 64,
        )
    with torch.no_grad():
        executor._weight_origin.fill_(1)
    before = {
        name: value.detach().clone()
        for name, value in executor.state_dict().items()
    }

    with pytest.raises(
        ValueError,
        match="refuses source-weight-contaminated executors",
    ):
        refit_structured_terminal_projections_from_targets_(
            executor,
            targets,
            calibration_split_sha256="f" * 64,
        )

    for name, expected in before.items():
        torch.testing.assert_close(
            executor.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
