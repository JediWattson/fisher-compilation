from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from fisher_graph.generator_interaction_map import (
    GeneratorInteractionMapAccumulator,
    GeneratorInteractionMapProvenance,
)
from fisher_graph.gemma3_full_mlp_stack_executor import (
    Gemma3FullMLPStackExecutor,
)
from fisher_graph.gemma3_generator_causal_intervention import (
    FrozenGemma3GeneratorCausalInterventionExecutor,
)

from test_gemma3_full_mlp_stack_executor import _full_replacements
from test_gemma3_modal_generator_executor import _adapter, _batch


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture() -> tuple[
    object,
    FrozenGemma3GeneratorCausalInterventionExecutor,
]:
    adapter = _adapter()
    full = Gemma3FullMLPStackExecutor(
        adapter,
        _full_replacements(adapter),
    )
    return adapter, FrozenGemma3GeneratorCausalInterventionExecutor(full)


def test_visits_baseline_then_every_single_generator_suppression() -> None:
    adapter, executor = _fixture()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()
    observed: list[object] = []

    def visit(execution: object) -> tuple[int | None, torch.Tensor]:
        observed.append(execution)
        return (
            execution.muted_layer_ordinal,
            execution.model_output.logits.detach().clone(),
        )

    results = executor.visit_baseline_and_single_suppressions(
        _batch().model_inputs,
        visitor=visit,
    )

    assert tuple(value[0] for value in results) == (None, 0, 1, 2)
    baseline = results[0][1]
    assert all(not torch.equal(baseline, value[1]) for value in results[1:])
    assert executor.layer_count == 3
    assert len(set(executor.generator_plan_sha256s)) == 3
    assert executor.generator_ids == (
        "layer.0.cluster.0",
        "layer.1.cluster.0",
        "layer.2.cluster.0",
    )
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_sha256

    for index, execution in enumerate(observed):
        assert execution.generated_layer_ordinals == (0, 1, 2)
        assert execution.valid_tokens == 6
        assert execution.logical_generator_macs == 6 * 3 * 20
        assert execution.logical_generator_bias_additions == 6 * 3 * 4
        assert execution.observational_only is True
        assert execution.mutation_authority is False
        assert execution.merge_authority is False
        assert execution.pruning_authority is False
        if index == 0:
            assert execution.suppressed_layer_ordinals == ()
            assert execution.logical_suppressed_generator_macs == 0
            assert (
                execution.logical_executed_generator_macs
                == execution.logical_generator_macs
            )
        else:
            assert execution.suppressed_layer_ordinals == (index - 1,)
            assert execution.logical_suppressed_generator_macs == 6 * 20
            assert (
                execution.logical_executed_generator_macs
                == execution.logical_generator_macs - 6 * 20
            )


def test_restores_native_stack_when_model_or_visitor_raises() -> None:
    adapter, executor = _fixture()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()

    def fail_visitor(execution: object) -> None:
        if execution.muted_layer_ordinal == 0:
            raise RuntimeError("sentinel visitor failure")

    with pytest.raises(RuntimeError, match="sentinel visitor failure"):
        executor.visit_baseline_and_single_suppressions(
            _batch().model_inputs,
            visitor=fail_visitor,
        )
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_sha256

    def fail_model(
        _module: nn.Module,
        _arguments: tuple[object, ...],
    ) -> None:
        raise RuntimeError("sentinel model failure")

    handle = executor.executor.compiled_mlps[
        "2"
    ].generator_input_proj.register_forward_pre_hook(fail_model)
    try:
        with pytest.raises(RuntimeError, match="sentinel model failure"):
            executor.visit_baseline_and_single_suppressions(
                _batch().model_inputs,
                visitor=lambda _execution: None,
            )
    finally:
        handle.remove()
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_sha256


def test_rejects_reentrancy_without_corrupting_source() -> None:
    adapter, executor = _fixture()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)

    def recurse(_execution: object) -> None:
        executor.visit_baseline_and_single_suppressions(
            _batch().model_inputs,
            visitor=lambda _value: None,
        )

    with pytest.raises(RuntimeError, match="not reentrant"):
        executor.visit_baseline_and_single_suppressions(
            _batch().model_inputs,
            visitor=recurse,
        )
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps


def test_visits_complete_interaction_map_with_ephemeral_generator_outputs() -> None:
    adapter, executor = _fixture()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()
    expected_baseline = executor.executor.run(
        _batch().model_inputs,
        condition="generated",
    ).model_output.logits.detach().clone()
    observed: list[
        tuple[
            tuple[int, ...],
            tuple[torch.Tensor | None, ...] | None,
        ]
    ] = []
    observed_logits: list[torch.Tensor] = []

    def visit(execution: object) -> tuple[int, ...]:
        observed_logits.append(execution.model_output.logits.detach().clone())
        traces = (
            None
            if execution.generated_residuals is None
            else tuple(
                None if value is None else value.detach().clone()
                for value in execution.generated_residuals
            )
        )
        observed.append((execution.suppressed_layer_ordinals, traces))
        return execution.suppressed_layer_ordinals

    results = executor.visit_generator_interaction_map(
        _batch().model_inputs,
        visitor=visit,
    )

    assert results == (
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    )
    assert torch.equal(observed_logits[0], expected_baseline)
    baseline = observed[0][1]
    assert baseline is not None
    assert all(isinstance(value, torch.Tensor) for value in baseline)
    for suppressed, traces in observed:
        if len(suppressed) == 2:
            assert traces is None
            continue
        assert traces is not None
        assert tuple(
            ordinal for ordinal, value in enumerate(traces) if value is None
        ) == suppressed
    assert observed[2][1] is not None
    assert observed[3][1] is not None
    assert torch.equal(baseline[0], observed[2][1][0])
    assert torch.equal(baseline[0], observed[3][1][0])
    assert torch.equal(baseline[1], observed[3][1][1])
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_sha256


def test_interaction_map_pair_failure_restores_source_stack() -> None:
    adapter, executor = _fixture()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()

    def fail_on_pair(execution: object) -> None:
        if execution.suppressed_layer_ordinals == (0, 2):
            raise RuntimeError("sentinel pair visitor failure")

    with pytest.raises(RuntimeError, match="sentinel pair visitor failure"):
        executor.visit_generator_interaction_map(
            _batch().model_inputs,
            visitor=fail_on_pair,
        )

    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_sha256


def test_interaction_map_rejects_noncanonical_pair_schedule() -> None:
    _, executor = _fixture()
    with pytest.raises(ValueError, match="canonical pairs"):
        executor.visit_generator_interaction_map(
            _batch().model_inputs,
            visitor=lambda _execution: None,
            joint_pairs=((1, 2), (0, 1)),
        )


def test_interaction_executor_streams_directly_into_generic_map_core() -> None:
    _, executor = _fixture()
    batch = _batch()
    accumulator = GeneratorInteractionMapAccumulator(
        generator_ids=executor.generator_ids,
        provenance=GeneratorInteractionMapProvenance(
            source_model_sha256=_sha("model"),
            generator_catalog_sha256=_sha("catalog"),
            evaluation_split_sha256=_sha("split"),
            objective_sha256=_sha("objective"),
        ),
        anchor_count=2,
    )

    def visit(execution: object) -> None:
        logits = execution.model_output.logits
        suppressed = execution.suppressed_layer_ordinals
        if len(suppressed) <= 1:
            assert execution.generated_residuals is not None
            outputs = {
                generator_id: value
                for generator_id, value in zip(
                    executor.generator_ids,
                    execution.generated_residuals,
                    strict=True,
                )
                if value is not None
            }
        else:
            assert execution.generated_residuals is None
            outputs = {}
        if not suppressed:
            accumulator.begin_batch(
                example_ids=batch.example_ids or (),
                baseline_logits=logits,
                targets=batch.targets,
                supervised_mask=batch.targets != -100,
                valid_mask=batch.valid_positions,
                baseline_generator_outputs=outputs,
            )
        elif len(suppressed) == 1:
            accumulator.add_singleton(
                executor.generator_ids[suppressed[0]],
                logits,
                outputs,
            )
        else:
            accumulator.add_joint(
                executor.generator_ids[suppressed[0]],
                executor.generator_ids[suppressed[1]],
                logits,
            )

    executor.visit_generator_interaction_map(
        batch.model_inputs,
        visitor=visit,
    )
    accumulator.finish_batch()
    analysis = accumulator.finalize()

    assert analysis.generator_count == 3
    assert analysis.pair_count == 3
    assert analysis.directed_edge_count == 3
    assert analysis.prompt_count == 2
    assert analysis.prompt_nll_second_differences.shape == (3, 2)
    assert analysis.prompt_directed_response_rms.shape == (3, 2)
