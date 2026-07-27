from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import pytest
import torch
from torch import Tensor

from fisher_graph.adapters.base import (
    FeedForwardSpec,
    LayerSpec,
    NormalizationSpec,
    ResidualStageSpec,
    StructuredOperatorSites,
    TransformerLayerSemantics,
    module_state_fingerprint,
)
from fisher_graph.adapters.toy import ToyTransformerAdapter
from fisher_graph.compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
    ScoreObjective,
)
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_whole_model_mode_graph_discovery import (
    GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_SCHEMA,
    run_gemma3_whole_model_mode_graph_discovery,
)
from fisher_graph.instrumentation import InstrumentedModel
from fisher_graph.model import ToyTransformer
from fisher_graph.modes import collect_adapter_score_gradients
from fisher_graph.streaming_analysis import (
    ActivationScoreGradientRows,
    iter_activation_score_gradient_rows,
)
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryResult,
    CrossBlockDiscoverySketch,
    CrossBlockExactCriteria,
    CrossBlockSketchConfig,
)


class _CountingToyTransformer(ToyTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.forward_calls = 0

    def forward(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def _structured_semantics(
    layer_id: str,
    *,
    residual_width: int,
    intermediate_width: int,
) -> TransformerLayerSemantics:
    norm = NormalizationSpec(
        kind="layer_norm",
        width=residual_width,
        epsilon=1e-5,
        affine=True,
    )
    qk_norm = NormalizationSpec(
        kind="layer_norm",
        width=residual_width // 2,
        epsilon=1e-5,
        affine=True,
    )
    attention_stage = ResidualStageSpec(
        id=f"{layer_id}.attention",
        kind="attention",
        input_site=f"{layer_id}.input",
        normalized_input_site=f"{layer_id}.ln1",
        operator_output_site=f"{layer_id}.attention.output",
        delta_site=f"{layer_id}.attention.delta",
        output_site=f"{layer_id}.post_attention",
    )
    feed_forward_stage = ResidualStageSpec(
        id=f"{layer_id}.feed_forward",
        kind="feed_forward",
        input_site=f"{layer_id}.post_attention",
        normalized_input_site=f"{layer_id}.ln2",
        operator_output_site=f"{layer_id}.mlp.output",
        delta_site=f"{layer_id}.mlp.delta",
        output_site=f"{layer_id}.output",
    )
    operator_sites = StructuredOperatorSites(
        attention_query_projection=f"{layer_id}.attention.query",
        attention_query_normalized=f"{layer_id}.attention.query_norm",
        attention_key_projection=f"{layer_id}.attention.key",
        attention_key_normalized=f"{layer_id}.attention.key_norm",
        attention_value_projection=f"{layer_id}.attention.value",
        attention_context=f"{layer_id}.attention.context_heads",
        feed_forward_gate_projection=f"{layer_id}.mlp.pre_activation",
        feed_forward_up_projection=f"{layer_id}.mlp.up_projection",
        feed_forward_down_input=f"{layer_id}.mlp.activated",
    )
    return TransformerLayerSemantics(
        residual_layout="sequential_attention_then_feed_forward_residual",
        attention_input_norm=norm,
        attention_output_norm=norm,
        qk_norm=qk_norm,
        attention_projection_bias=True,
        attention_dropout=0.0,
        attention_logit_softcap=None,
        feed_forward_input_norm=norm,
        feed_forward_output_norm=norm,
        feed_forward=FeedForwardSpec(
            kind="dense",
            intermediate_width=intermediate_width,
            activation="gelu",
            projection_bias=True,
        ),
        stages=(attention_stage, feed_forward_stage),
        operator_sites=operator_sites,
    )


class _StructuredToyAdapter(ToyTransformerAdapter):
    """Add Gemma-like structured MLP metadata to real toy capture sites."""

    def __init__(self, model: ToyTransformer) -> None:
        super().__init__(model)
        layers: list[LayerSpec] = []
        for source_layer in self._layers:
            layer_id = source_layer.id
            layers.append(
                LayerSpec(
                    id=layer_id,
                    ordinal=source_layer.ordinal,
                    input_site=f"{layer_id}.input",
                    output_site=f"{layer_id}.output",
                    residual_width=source_layer.residual_width,
                    kind="structured_toy_decoder",
                    attention=source_layer.attention,
                    source_path=source_layer.source_path,
                    transformer=_structured_semantics(
                        layer_id,
                        residual_width=source_layer.residual_width,
                        intermediate_width=model.config.d_ff,
                    ),
                )
            )
        self._layers = tuple(layers)


class _RecordingRowFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.first_pass_rows: list[ActivationScoreGradientRows] = []

    def __call__(
        self,
        model: InstrumentedModel,
        calibration_batches: Iterable[CalibrationBatch],
        *,
        activation_names: Collection[str],
        score_objective: ScoreObjective,
        leaf_activation_name: str | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> Iterable[ActivationScoreGradientRows]:
        call_index = len(self.calls)
        self.calls.append(
            {
                "activation_names": tuple(activation_names),
                "leaf_activation_name": leaf_activation_name,
                "accumulation_dtype": accumulation_dtype,
            }
        )
        rows = iter_activation_score_gradient_rows(
            model,
            calibration_batches,
            activation_names=activation_names,
            score_objective=score_objective,
            leaf_activation_name=leaf_activation_name,
            accumulation_dtype=accumulation_dtype,
        )

        def record() -> Iterable[ActivationScoreGradientRows]:
            try:
                for row in rows:
                    if call_index == 0:
                        self.first_pass_rows.append(row)
                    yield row
            finally:
                rows.close()

        return record()


def _fixture(
    *,
    layers: int = 3,
) -> tuple[
    _CountingToyTransformer,
    _StructuredToyAdapter,
    CalibrationBatch,
]:
    torch.manual_seed(9701)
    config = TransformerConfig(
        vocab_size=17,
        max_sequence_length=6,
        d_model=4,
        n_heads=2,
        n_layers=layers,
        d_ff=5,
        dropout=0.0,
    )
    model = _CountingToyTransformer(config)
    adapter = _StructuredToyAdapter(model)
    valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    batch = CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 0],
                ]
            ),
            "attention_mask": valid,
        },
        targets=torch.tensor(
            [
                [2, 3, 4, -100],
                [6, 7, -100, -100],
            ]
        ),
        valid_positions=valid,
        example_ids=("fit-family-a", "fit-family-b"),
    )
    return model, adapter, batch


def _all_tensors(value: object) -> list[Tensor]:
    tensors: list[Tensor] = []
    if isinstance(value, Tensor):
        tensors.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            tensors.extend(_all_tensors(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(_all_tensors(item))
    return tensors


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in forbidden or _contains_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_whole_model_runner_streams_two_exact_full_suffix_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, adapter, batch = _fixture(layers=3)
    reference_model = _CountingToyTransformer(model.config)
    reference_model.load_state_dict(model.state_dict())
    reference_adapter = _StructuredToyAdapter(reference_model)
    mlp_sites = tuple(
        (
            layer.transformer.operator_sites.feed_forward_down_input
            if layer.transformer is not None
            and layer.transformer.operator_sites is not None
            else ""
        )
        for layer in adapter.layers
    )
    reference = collect_adapter_score_gradients(
        reference_adapter,
        (batch,),
        activation_names=mlp_sites,
        score_objective=CausalLanguageModelNLL(),
    )

    model.eval()
    model.requires_grad_(False)
    before_state = module_state_fingerprint(model)
    before_parameters = tuple(
        (
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter._version,
            parameter.requires_grad,
        )
        for parameter in model.parameters()
    )
    row_factory = _RecordingRowFactory()
    real_grad = torch.autograd.grad
    backward_calls = 0

    def counting_grad(*args: object, **kwargs: object):
        nonlocal backward_calls
        backward_calls += 1
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting_grad)
    artifact, report = run_gemma3_whole_model_mode_graph_discovery(
        adapter,
        (batch,),
        calibration_fit_split_sha256="a" * 64,
        family_fold_assignment={
            "fit-family-a": 0,
            "fit-family-b": 1,
        },
        sketch_config=CrossBlockSketchConfig(
            sketch_size=11,
            sketch_seed=7,
            per_layer_pool_size=3,
            neighbors_per_mode=2,
            proxy_min_signed_correlation=-1.0,
        ),
        exact_criteria=CrossBlockExactCriteria(
            min_row_signed_correlation=1.0,
            min_sequence_signed_correlation=1.0,
            min_energy_balance=1.0,
            min_absolute_activation_correlation=1.0,
            max_activation_rank_one_tail_fraction=0.0,
            min_coactivity=1.0,
            min_fold_signed_correlation=1.0,
        ),
        row_factory=row_factory,
        row_factory_id="tests.recording_rows.v1",
    )

    assert model.forward_calls == 4
    assert backward_calls == 4
    assert len(row_factory.calls) == 2
    expected_sites = ("layer.0.input", *mlp_sites)
    assert all(
        call["activation_names"] == expected_sites
        and call["leaf_activation_name"] == "layer.0.input"
        and call["accumulation_dtype"] == torch.float64
        for call in row_factory.calls
    )
    for site in mlp_sites:
        actual_gradients = torch.cat(
            [
                row.score_gradients[site]
                for row in row_factory.first_pass_rows
            ],
            dim=0,
        )
        torch.testing.assert_close(
            actual_gradients,
            reference.samples[site].score_gradients.double(),
            rtol=1e-10,
            atol=1e-10,
        )
        assert torch.count_nonzero(actual_gradients) > 0

    assert module_state_fingerprint(model) == before_state
    assert tuple(
        (
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter._version,
            parameter.requires_grad,
        )
        for parameter in model.parameters()
    ) == before_parameters
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training

    assert artifact["schema"] == (
        GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_SCHEMA
    )
    assert report["scientific_status"]["calibration_a_guard_opened"] is False
    assert report["scientific_status"]["calibration_b_opened"] is False
    assert report["protocol"]["streaming_passes"] == 2
    assert report["protocol"]["guard_or_heldout_data_consumed"] is False
    assert report["protocol"]["family_folds"][
        "family_disjoint_assignment_supplied"
    ] is True
    assert report["model"]["layer_count"] == 3
    assert tuple(
        layer["activation_site"] for layer in report["model"]["layers"]
    ) == mlp_sites
    sketch_state = artifact["sketch_state"]
    assert isinstance(sketch_state, Mapping)
    restored_sketch = CrossBlockDiscoverySketch.from_state_dict(sketch_state)
    discovery_state = artifact["discovery_state"]
    assert isinstance(discovery_state, Mapping)
    restored_discovery = CrossBlockDiscoveryResult.from_state_dict(
        discovery_state
    )
    assert (
        restored_discovery.sketch_artifact_sha256
        == restored_sketch.artifact_sha256
    )
    assert len(restored_sketch.modes) == 3 * model.config.d_ff
    for layer_ordinal in range(3):
        layer_ranks = tuple(
            mode.key.fisher_rank
            for mode in restored_sketch.modes
            if mode.key.layer_ordinal == layer_ordinal
        )
        assert tuple(sorted(layer_ranks)) == tuple(range(model.config.d_ff))

    tensors = _all_tensors(artifact)
    assert tensors
    assert max(tensor.numel() for tensor in tensors) <= 11
    assert all(
        tensor.device.type == "cpu" and tensor.dtype == torch.float64
        for tensor in tensors
    )
    assert not _all_tensors(report)
    assert not _contains_key(
        artifact,
        {
            "model_state_dict",
            "executor_state_dict",
            "source_state_dict",
            "weights",
            "parameters",
        },
    )
    assert artifact["safety"]["contains_source_model_weights"] is False
    assert artifact["safety"]["contains_executable_weights"] is False
    assert artifact["safety"]["authorizes_calibration_a_guard"] is False
    assert artifact["safety"]["authorizes_calibration_b"] is False


def test_two_layer_no_qualifying_edge_is_a_valid_discovery() -> None:
    model, adapter, batch = _fixture(layers=2)
    model.eval()
    model.requires_grad_(False)

    artifact, report = run_gemma3_whole_model_mode_graph_discovery(
        adapter,
        (batch,),
        calibration_fit_split_sha256="b" * 64,
        sketch_config=CrossBlockSketchConfig(
            sketch_size=13,
            sketch_seed=19,
            per_layer_pool_size=4,
            neighbors_per_mode=3,
            proxy_min_signed_correlation=-1.0,
        ),
        exact_criteria=CrossBlockExactCriteria(
            min_row_signed_correlation=1.0,
            min_sequence_signed_correlation=1.0,
            min_energy_balance=1.0,
            min_absolute_activation_correlation=1.0,
            max_activation_rank_one_tail_fraction=0.0,
            min_coactivity=1.0,
        ),
    )

    assert report["scientific_status"]["discovery_completed"] is True
    assert report["scientific_status"]["scientific_compression_success"] is False
    assert report["model"]["layer_count"] == 2
    assert report["discovery"]["selected_hypotheses"] == ()
    assert artifact["safety"]["discovery_only"] is True
    assert artifact["safety"]["authorizes_executor_construction"] is False


def test_runner_rejects_non_materialized_batches_and_unfrozen_sources() -> None:
    model, adapter, batch = _fixture(layers=2)
    model.eval()
    model.requires_grad_(False)
    with pytest.raises(TypeError, match="materialized sequence"):
        run_gemma3_whole_model_mode_graph_discovery(
            adapter,
            (value for value in (batch,)),  # type: ignore[arg-type]
            calibration_fit_split_sha256="c" * 64,
        )

    model.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen source parameters"):
        run_gemma3_whole_model_mode_graph_discovery(
            adapter,
            (batch,),
            calibration_fit_split_sha256="c" * 64,
        )


def test_runner_requires_exact_family_fold_coverage() -> None:
    model, adapter, batch = _fixture(layers=2)
    model.eval()
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="exact fit example ids"):
        run_gemma3_whole_model_mode_graph_discovery(
            adapter,
            (batch,),
            calibration_fit_split_sha256="d" * 64,
            family_fold_assignment={"fit-family-a": 0},
        )
