from __future__ import annotations

import pytest
import torch
from torch import nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.compiler.calibration import CausalLanguageModelNLL
from fisher_graph.full_mlp_stack_trajectory import (
    FrozenFullMLPStackTrajectoryExecutor,
    evaluate_full_mlp_stack_trajectory,
)
from fisher_graph.gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)
from fisher_graph.streaming_analysis import (
    iter_activation_score_gradient_rows,
)

from test_gemma3_modal_generator_executor import (
    _adapter,
    _batch,
    _bound_plan,
)


def _replacements(adapter: object):
    return tuple(
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=ordinal,
            removed_mode_indices=tuple(range(6)),
            generator_plan=_bound_plan(adapter, ordinal),
        )
        for ordinal in range(3)
    )


def _fixture():
    adapter = _adapter()
    return adapter, FrozenFullMLPStackTrajectoryExecutor(
        adapter,
        _replacements(adapter),
    )


def test_arbitrary_subset_reuses_one_compiled_catalog_and_accounts_exactly():
    adapter, executor = _fixture()
    batch = _batch()
    source_model = adapter.module
    source_mlps = tuple(layer.mlp for layer in source_model.model.layers)
    source_sha256 = adapter.model_fingerprint()
    compiled_ids = tuple(id(value) for value in executor.compiled_mlps.values())

    execution = executor.run_subset(
        batch.model_inputs,
        generated_layer_ordinals=(0, 2),
    )

    per_layer_native = tuple(
        value.native_removed_parameter_count
        for value in executor.compiled_mlps.values()
    )
    per_layer_generator = tuple(
        value.generator_parameter_count
        for value in executor.compiled_mlps.values()
    )
    per_layer_macs = tuple(
        value.generator_macs_per_token
        for value in executor.compiled_mlps.values()
    )
    per_layer_bias = tuple(
        value.generator_bias_additions_per_token
        for value in executor.compiled_mlps.values()
    )
    accounting = execution.accounting
    assert execution.generated_layer_ordinals == (0, 2)
    assert accounting.replaced_layer_count == 2
    assert accounting.removed_mode_count == 12
    assert accounting.logical_native_mlp_parameters_removed == (
        per_layer_native[0] + per_layer_native[2]
    )
    assert accounting.logical_generator_subset_learned_parameters == (
        per_layer_generator[0] + per_layer_generator[2]
    )
    assert accounting.logical_generator_macs == 6 * (
        per_layer_macs[0] + per_layer_macs[2]
    )
    assert accounting.logical_generator_bias_additions == 6 * (
        per_layer_bias[0] + per_layer_bias[2]
    )
    assert accounting.logical_candidate_mlp_macs == (
        accounting.logical_native_mlp_macs_retained
        + accounting.logical_generator_macs
    )
    assert accounting.net_logical_macs_saved == (
        accounting.logical_native_mlp_stack_macs_baseline
        - accounting.logical_candidate_mlp_macs
    )
    assert accounting.experimental_resident_compiled_full_stack_learned_parameters == sum(
        per_layer_generator
    )
    assert tuple(id(value) for value in executor.compiled_mlps.values()) == (
        compiled_ids
    )
    assert tuple(layer.mlp for layer in source_model.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_sha256


def test_subset_restores_source_and_compiled_state_after_model_error():
    adapter, executor = _fixture()
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()
    compiled_sha256 = tuple(
        module_state_fingerprint(value)
        for value in executor.compiled_mlps.values()
    )

    def fail(
        _module: nn.Module,
        _arguments: tuple[object, ...],
    ) -> None:
        raise RuntimeError("sentinel trajectory failure")

    handle = executor.compiled_mlps[
        "2"
    ].generator_input_proj.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="sentinel trajectory failure"):
            executor.run_subset(
                batch.model_inputs,
                generated_layer_ordinals=(1, 2),
            )
    finally:
        handle.remove()

    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == source_mlps
    assert adapter.model_fingerprint() == source_sha256
    assert tuple(
        module_state_fingerprint(value)
        for value in executor.compiled_mlps.values()
    ) == compiled_sha256
    recovered = executor.run_subset(
        batch.model_inputs,
        generated_layer_ordinals=(1,),
    )
    assert recovered.generated_layer_ordinals == (1,)


def test_streaming_visitor_restores_source_when_consumer_raises():
    adapter, executor = _fixture()
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()

    def consume_native(_output: object) -> None:
        return None

    def fail_consumer(_execution: object) -> None:
        raise RuntimeError("sentinel consumer failure")

    with pytest.raises(RuntimeError, match="sentinel consumer failure"):
        executor.visit_native_and_subsets(
            batch.model_inputs,
            generated_layer_subsets=((0,), (0, 1)),
            native_visitor=consume_native,
            subset_visitor=fail_consumer,
        )

    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == source_mlps
    assert adapter.model_fingerprint() == source_sha256


def test_subset_overlay_callback_supports_one_instrumented_backward():
    adapter, executor = _fixture()
    batch = _batch().sample(0)
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()
    transformer = adapter.layers[1].transformer
    assert transformer is not None
    feed_forward = next(
        stage
        for stage in transformer.stages
        if stage.kind == "feed_forward"
    )
    requested = (
        feed_forward.normalized_input_site,
        feed_forward.operator_output_site,
    )
    callback_calls = 0

    def collect_rows():
        nonlocal callback_calls
        callback_calls += 1
        return tuple(
            iter_activation_score_gradient_rows(
                adapter,
                (batch,),
                activation_names=requested,
                score_objective=CausalLanguageModelNLL(),
                leaf_activation_name=(
                    feed_forward.normalized_input_site
                ),
                accumulation_dtype=torch.float64,
            )
        )

    rows = executor.run_with_subset_overlay(
        generated_layer_ordinals=(0,),
        callback=collect_rows,
    )

    assert callback_calls == 1
    assert len(rows) == 1
    assert rows[0].example_id == batch.example_ids[0]
    assert set(rows[0].activations) == set(requested)
    assert set(rows[0].score_gradients) == set(requested)
    assert all(
        bool(torch.isfinite(value).all())
        for value in rows[0].score_gradients.values()
    )
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        source_mlps
    )
    assert adapter.model_fingerprint() == source_sha256


def test_subset_overlay_callback_supports_multiple_instrumented_backwards():
    adapter, executor = _fixture()
    batch = _batch()
    transformer = adapter.layers[2].transformer
    assert transformer is not None
    feed_forward = next(
        stage
        for stage in transformer.stages
        if stage.kind == "feed_forward"
    )
    requested = (feed_forward.normalized_input_site,)

    def collect_rows():
        return tuple(
            iter_activation_score_gradient_rows(
                adapter,
                (batch,),
                activation_names=requested,
                score_objective=CausalLanguageModelNLL(),
                leaf_activation_name=requested[0],
            )
        )

    rows = executor.run_with_subset_overlay(
        generated_layer_ordinals=(0, 1),
        callback=collect_rows,
        expected_forward_calls=2,
    )

    assert tuple(row.example_id for row in rows) == batch.example_ids


def test_subset_overlay_callback_restores_source_after_callback_error():
    adapter, executor = _fixture()
    batch = _batch().sample(0)
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()

    def fail_after_forward() -> None:
        adapter.forward(batch.model_inputs)
        raise RuntimeError("sentinel callback failure")

    with pytest.raises(RuntimeError, match="sentinel callback failure"):
        executor.run_with_subset_overlay(
            generated_layer_ordinals=(0, 1),
            callback=fail_after_forward,
        )

    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        source_mlps
    )
    assert adapter.model_fingerprint() == source_sha256
    recovered = executor.run_subset(
        batch.model_inputs,
        generated_layer_ordinals=(0, 1),
    )
    assert recovered.generated_layer_ordinals == (0, 1)


def test_subset_overlay_callback_requires_exactly_one_forward():
    adapter, executor = _fixture()

    with pytest.raises(
        RuntimeError,
        match="not every trajectory overlay executed exactly once",
    ):
        executor.run_with_subset_overlay(
            generated_layer_ordinals=(0,),
            callback=lambda: "callback result without a forward",
        )

    recovered = executor.run_subset(
        _batch().sample(0).model_inputs,
        generated_layer_ordinals=(0,),
    )
    assert recovered.generated_layer_ordinals == (0,)


def test_subset_overlay_callback_rejects_wrong_expected_forward_count():
    adapter, executor = _fixture()
    batch = _batch().sample(0)
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()

    with pytest.raises(
        RuntimeError,
        match="not every trajectory overlay executed exactly 2 times",
    ):
        executor.run_with_subset_overlay(
            generated_layer_ordinals=(0, 1),
            callback=lambda: adapter.forward(batch.model_inputs),
            expected_forward_calls=2,
        )

    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        source_mlps
    )
    assert adapter.model_fingerprint() == source_sha256


def test_subset_overlay_callback_rejects_too_many_forwards_and_restores():
    adapter, executor = _fixture()
    batch = _batch().sample(0)
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_sha256 = adapter.model_fingerprint()

    def run_twice() -> None:
        adapter.forward(batch.model_inputs)
        adapter.forward(batch.model_inputs)

    with pytest.raises(
        RuntimeError,
        match="each frozen trajectory overlay may execute only once",
    ):
        executor.run_with_subset_overlay(
            generated_layer_ordinals=(0, 1),
            callback=run_twice,
        )

    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        source_mlps
    )
    assert adapter.model_fingerprint() == source_sha256


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_subset_overlay_callback_requires_positive_exact_forward_count(value):
    _adapter_value, executor = _fixture()

    with pytest.raises(
        ValueError,
        match="expected_forward_calls must be a positive integer",
    ):
        executor.run_with_subset_overlay(
            generated_layer_ordinals=(0,),
            callback=lambda: None,
            expected_forward_calls=value,
        )


def test_subset_overlay_callback_preserves_reentrancy_guard():
    adapter, executor = _fixture()
    batch = _batch().sample(0)

    def recurse() -> object:
        return executor.run_with_subset_overlay(
            generated_layer_ordinals=(1,),
            callback=lambda: adapter.forward(batch.model_inputs),
        )

    with pytest.raises(RuntimeError, match="not reentrant"):
        executor.run_with_subset_overlay(
            generated_layer_ordinals=(0,),
            callback=recurse,
        )

    recovered = executor.run_subset(
        batch.model_inputs,
        generated_layer_ordinals=(0,),
    )
    assert recovered.generated_layer_ordinals == (0,)


def test_all_depth_evaluation_deduplicates_full_endpoint_and_is_exact():
    adapter, executor = _fixture()
    batch = _batch()
    forward_calls = 0

    def count_forward(
        _module: nn.Module,
        _arguments: tuple[object, ...],
    ) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = adapter.module.register_forward_pre_hook(count_forward)
    try:
        report = evaluate_full_mlp_stack_trajectory(
            adapter,
            executor,
            (batch,),
            expected_example_ids=batch.example_ids,
            expected_mode_counts_by_layer=(6, 6, 6),
            vocabulary_chunk_size=2,
        )
    finally:
        handle.remove()

    assert forward_calls == 6
    assert report["condition_order"] == (
        "native",
        "prefix_01",
        "prefix_02",
        "suffix_01",
        "suffix_02",
        "full_stack",
    )
    assert report["trajectory_condition_ids"] == {
        "prefix": ("prefix_01", "prefix_02", "full_stack"),
        "suffix": ("suffix_01", "suffix_02", "full_stack"),
    }
    assert report["full_stack_endpoint_evaluated_once"] is True
    assert set(report["conditions"]) == set(report["condition_order"])
    assert report["conditions"]["native"] == {
        "nll_per_token": report["conditions"]["native"]["nll_per_token"],
        "delta_nll_per_token": 0.0,
        "native_to_candidate_kl_per_token": 0.0,
        "top1_agreement_to_native": 1.0,
    }
    assert report["assessment_role"] == "open_development_assessment"
    assert report["heldout_confirmation"] is False
    assert report["latency_or_kernel_speed_claim"] is False
    assert report["interpretation_guard"] == {
        "marginal_changes_are_path_conditioned": True,
        "marginal_changes_are_not_isolated_layer_causal_effects": True,
    }

    native_resources = report["resource_accounting"]["native"]
    full_resources = report["resource_accounting"]["full_stack"]
    assert native_resources["generated_layer_ordinals"] == ()
    assert native_resources["logical_net_stored_parameter_savings"] == 0
    assert native_resources["logical_generator_macs"] == 0
    assert full_resources["generated_layer_ordinals"] == (0, 1, 2)
    assert full_resources["logical_native_mlp_parameters_retained"] == 0
    assert full_resources["logical_candidate_mlp_macs"] == (
        full_resources["logical_generator_macs"]
    )
    assert report["supervised_tokens"] == 6
    assert report["logical_valid_tokens"] == 6
    assert len(report["marginal_changes"]["prefix"]) == 3
    assert len(report["marginal_changes"]["suffix"]) == 3


def test_evaluation_metrics_are_invariant_to_assessment_batching():
    adapter_a, executor_a = _fixture()
    batch = _batch()
    together = evaluate_full_mlp_stack_trajectory(
        adapter_a,
        executor_a,
        (batch,),
        expected_example_ids=batch.example_ids,
        expected_mode_counts_by_layer=(6, 6, 6),
        vocabulary_chunk_size=2,
    )
    adapter_b, executor_b = _fixture()
    split = evaluate_full_mlp_stack_trajectory(
        adapter_b,
        executor_b,
        (batch.sample(0), batch.sample(1)),
        expected_example_ids=batch.example_ids,
        expected_mode_counts_by_layer=(6, 6, 6),
        vocabulary_chunk_size=2,
    )

    for condition_id in together["condition_order"]:
        for metric in together["conditions"][condition_id]:
            assert together["conditions"][condition_id][metric] == pytest.approx(
                split["conditions"][condition_id][metric],
                rel=0.0,
                abs=1e-15,
            )
        assert together["resource_accounting"][condition_id] == (
            split["resource_accounting"][condition_id]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("membership", "declared example membership"),
        ("mode_count", "exactly cover every declared"),
        ("role", "cannot relabel"),
    ),
)
def test_evaluation_rejects_leakage_or_scope_drift(
    mutation: str,
    message: str,
):
    adapter, executor = _fixture()
    batch = _batch()
    expected_ids = batch.example_ids
    mode_counts = (6, 6, 6)
    role = "open_development_assessment"
    if mutation == "membership":
        assert expected_ids is not None
        expected_ids = tuple(reversed(expected_ids))
    elif mutation == "mode_count":
        mode_counts = (6, 6, 5)
    elif mutation == "role":
        role = "heldout_confirmation"
    else:
        raise AssertionError("unknown mutation")

    with pytest.raises(ValueError, match=message):
        evaluate_full_mlp_stack_trajectory(
            adapter,
            executor,
            (batch,),
            expected_example_ids=expected_ids,
            expected_mode_counts_by_layer=mode_counts,
            vocabulary_chunk_size=2,
            assessment_role=role,
        )


@pytest.mark.parametrize(
    "subset",
    (
        (1, 0),
        (0, 0),
        (-1,),
        (3,),
    ),
)
def test_executor_rejects_noncanonical_subsets(subset):
    _adapter_value, executor = _fixture()
    with pytest.raises(ValueError, match="unique, increasing, in-range"):
        executor.run_subset(
            _batch().model_inputs,
            generated_layer_ordinals=subset,
        )
