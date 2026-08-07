"""Full-logit scoring for an A5d source-owned plus additive-residual graph.

The ordinary trajectory-correction scorer assumes that one graph owns every
generated contribution.  A5d deliberately has two runtimes at Layer 17: the
already-qualified source graph owns native deletion, while an optional
zero-mean graph contributes a scaled residual after the live feed-forward
RMSNorm.  This scorer authenticates and accounts for both traversals without
reinterpreting the additive graph as another native replacement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack

import torch

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_l10_l17_open_a_progressive_evaluation import _record_execution
from .gemma3_layer17_open_a_capacity_evaluation import (
    _add_comparison,
    _add_native,
    _candidate_comparison,
    _finalize_metric_accumulator,
    _model_logits,
    _native_nll,
    _new_metric_accumulator,
    _selected_logits_and_targets,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecution,
    Gemma3ModalGeneratorGraphExecutor,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_graph_rung_evaluation import (
    _GRAPH_LOGICAL_FIELDS,
    _GRAPH_STATIC_FIELDS,
    _execution_fields,
)


__all__ = [
    "score_a5d_source_anchored_residual_fold",
]


_CONDITIONS = (
    "layer10_only",
    "selected_layer17_only",
    "frozen_uncorrected_composition",
    "selected_composition",
    "matched_double_deletion",
)
_VOCABULARY_CHUNK_SIZE = 16_384


def _additive_plan(
    executor: Gemma3ModalGeneratorGraphExecutor,
) -> ModalGeneratorGraphPlan | None:
    plan = executor.additive_post_feedforward_graph_plan
    if plan is None:
        return None
    if not isinstance(plan, ModalGeneratorGraphPlan):
        raise TypeError("A5d executor does not expose an additive graph plan")
    plan.validate_integrity()
    return plan


def _validate_executor_panel(
    *,
    layer10: Gemma3ModalGeneratorGraphExecutor,
    layer17: Gemma3ModalGeneratorGraphExecutor,
    frozen_composition: Gemma3ModalGeneratorGraphExecutor,
    selected_composition: Gemma3ModalGeneratorGraphExecutor,
) -> bool:
    executors = (layer10, layer17, frozen_composition, selected_composition)
    if len({id(value) for value in executors}) != len(executors):
        raise ValueError("A5d scoring executors must be distinct")
    if (
        layer10.affected_layer_ordinals != (10,)
        or layer17.affected_layer_ordinals != (17,)
        or frozen_composition.affected_layer_ordinals != (10, 17)
        or selected_composition.affected_layer_ordinals != (10, 17)
        or layer10.post_feedforward_delta_layer_ordinals
        or layer17.post_feedforward_delta_layer_ordinals
        or frozen_composition.post_feedforward_delta_layer_ordinals
        or selected_composition.post_feedforward_delta_layer_ordinals
        or _additive_plan(layer10) is not None
        or _additive_plan(frozen_composition) is not None
        or layer17.graph_plan.interactions
        or len(layer10.graph_plan.interactions) != 3
        or len(frozen_composition.graph_plan.interactions) != 3
        or len(selected_composition.graph_plan.interactions) != 3
        or selected_composition.graph_plan.artifact_sha256
        != frozen_composition.graph_plan.artifact_sha256
    ):
        raise ValueError("A5d executor ownership/topology differs from protocol")

    layer17_additive = _additive_plan(layer17)
    composition_additive = _additive_plan(selected_composition)
    if (layer17_additive is None) != (composition_additive is None):
        raise ValueError("A5d selected executors disagree on residual presence")
    if layer17_additive is None:
        if (
            layer17.additive_post_feedforward_layer_ordinals
            or selected_composition.additive_post_feedforward_layer_ordinals
            or layer17.additive_post_feedforward_lowering_artifact_sha256s
            or selected_composition.additive_post_feedforward_lowering_artifact_sha256s
        ):
            raise ValueError("A5d fallback retained additive runtime metadata")
        return False

    assert composition_additive is not None
    if (
        layer17_additive.artifact_sha256 != composition_additive.artifact_sha256
        or layer17_additive.interactions
        or composition_additive.interactions
        or layer17.additive_post_feedforward_layer_ordinals != (17,)
        or selected_composition.additive_post_feedforward_layer_ordinals != (17,)
        or layer17.additive_post_feedforward_scale
        != selected_composition.additive_post_feedforward_scale
        or not 0.0 < layer17.additive_post_feedforward_scale <= 1.0
        or layer17.additive_post_feedforward_lowering_artifact_sha256s
        != selected_composition.additive_post_feedforward_lowering_artifact_sha256s
    ):
        raise ValueError("A5d additive graph identity/scale differs across scope")
    return True


def _validate_dual_execution(
    execution: Gemma3ModalGeneratorGraphExecution,
    executor: Gemma3ModalGeneratorGraphExecutor,
    *,
    condition: str,
    label: str,
) -> None:
    if execution.condition != condition:
        raise RuntimeError(f"{label} execution condition drifted")
    expected_primary = (
        executor.graph_plan.traversal_order if condition == "generated" else ()
    )
    if execution.graph_execution.traversal_order != expected_primary:
        raise RuntimeError(f"{label} owning graph traversal drifted")
    additive_plan = _additive_plan(executor)
    additive_execution = execution.additive_graph_execution
    if additive_plan is None:
        if additive_execution is not None:
            raise RuntimeError(f"{label} unexpectedly executed an additive graph")
    else:
        if additive_execution is None:
            raise RuntimeError(f"{label} omitted additive traversal evidence")
        expected_additive = (
            additive_plan.traversal_order if condition == "generated" else ()
        )
        if additive_execution.traversal_order != expected_additive:
            raise RuntimeError(f"{label} additive graph traversal drifted")
    expected_nodes = len(executor.graph_plan.nodes) + (
        0 if additive_plan is None else len(additive_plan.nodes)
    )
    if execution.graph_node_count != expected_nodes:
        raise RuntimeError(f"{label} total graph node accounting drifted")
    expected_parameters = executor.graph_plan.parameter_count + (
        0 if additive_plan is None else additive_plan.parameter_count
    )
    if execution.modal_graph_learned_parameters != expected_parameters:
        raise RuntimeError(f"{label} total graph parameter accounting drifted")


def _compact_resources(
    *,
    executor: Gemma3ModalGeneratorGraphExecutor,
    static: Mapping[str, object],
    totals: Mapping[str, int],
    logical_valid_tokens: int,
    peak_live_modal_width: int,
) -> dict[str, object]:
    if logical_valid_tokens <= 0:
        raise ValueError("A5d logical valid-token total must be positive")
    for field in _GRAPH_LOGICAL_FIELDS:
        value = totals.get(field)
        if type(value) is not int or value % logical_valid_tokens:
            raise RuntimeError(f"A5d {field} is not an exact per-token total")
    additive = _additive_plan(executor)
    owning_parameters = executor.graph_plan.parameter_count
    additive_parameters = 0 if additive is None else additive.parameter_count
    owning_macs = executor.graph_plan.macs_per_token
    additive_macs = 0 if additive is None else additive.macs_per_token
    owning_additions = executor.graph_plan.accounting.elementwise_additions_per_token
    additive_additions = (
        0 if additive is None else additive.accounting.elementwise_additions_per_token
    )
    source_parameters = int(static["source_whole_model_learned_parameters"])
    removed_parameters = int(static["native_removed_learned_parameters"])
    total_parameters = owning_parameters + additive_parameters
    candidate_parameters = int(static["candidate_whole_model_learned_parameters"])
    executed_macs = (
        totals["logical_executed_modal_graph_macs"] // logical_valid_tokens
    )
    net_macs = totals["net_logical_macs_saved"] // logical_valid_tokens
    dense_macs = owning_macs + additive_macs
    if (
        int(static["graph_node_count"])
        != len(executor.graph_plan.nodes)
        + (0 if additive is None else len(additive.nodes))
        or int(static["modal_graph_learned_parameters"]) != total_parameters
        or int(static["net_stored_parameter_savings"])
        != removed_parameters - total_parameters
        or candidate_parameters
        != source_parameters - removed_parameters + total_parameters
        or totals["logical_linear_macs_native_removed"]
        != removed_parameters * logical_valid_tokens
        or totals["logical_modal_graph_macs"]
        != dense_macs * logical_valid_tokens
        or totals["logical_modal_graph_additions"]
        != (owning_additions + additive_additions) * logical_valid_tokens
        or net_macs != removed_parameters - executed_macs
    ):
        raise RuntimeError("A5d exact resource identities drifted")
    return {
        "replaced_layer_count": int(static["replaced_layer_count"]),
        "owning_graph_node_count": len(executor.graph_plan.nodes),
        "additive_graph_node_count": 0 if additive is None else len(additive.nodes),
        "total_graph_node_count": int(static["graph_node_count"]),
        "owning_interaction_count": len(executor.graph_plan.interactions),
        "additive_interaction_count": (
            0 if additive is None else len(additive.interactions)
        ),
        "total_interaction_count": (
            len(executor.graph_plan.interactions)
            + (0 if additive is None else len(additive.interactions))
        ),
        "native_removed_parameters": removed_parameters,
        "owning_graph_parameters": owning_parameters,
        "additive_graph_parameters": additive_parameters,
        "total_graph_parameters": total_parameters,
        "net_parameter_savings": int(static["net_stored_parameter_savings"]),
        "candidate_whole_model_learned_parameters": candidate_parameters,
        "owning_dense_graph_macs_per_token": owning_macs,
        "additive_dense_graph_macs_per_token": additive_macs,
        "total_dense_graph_macs_per_token": dense_macs,
        "executed_graph_macs_per_token": executed_macs,
        "net_executed_macs_saved_per_token": net_macs,
        "owning_dense_graph_additions_per_token": owning_additions,
        "additive_dense_graph_additions_per_token": additive_additions,
        "total_dense_graph_additions_per_token": (
            owning_additions + additive_additions
        ),
        "executed_graph_additions_per_token": (
            totals["logical_executed_modal_graph_additions"]
            // logical_valid_tokens
        ),
        "executed_peak_live_modal_width": peak_live_modal_width,
    }


def score_a5d_source_anchored_residual_fold(
    *,
    adapter: Gemma3CausalLMAdapter,
    layer10_executor: Gemma3ModalGeneratorGraphExecutor,
    selected_layer17_executor: Gemma3ModalGeneratorGraphExecutor,
    frozen_composition_executor: Gemma3ModalGeneratorGraphExecutor,
    selected_composition_executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    """Score one untouched outer-family example through full model logits."""

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("A5d held batches must contain CalibrationBatch values")
    identities = tuple(
        example_id
        for batch in materialized
        for example_id in (batch.example_ids or ())
    )
    if (
        any(batch.example_ids is None for batch in materialized)
        or not identities
        or len(identities) != len(set(identities))
    ):
        raise ValueError("A5d held batches require unique example identities")
    _validate_executor_panel(
        layer10=layer10_executor,
        layer17=selected_layer17_executor,
        frozen_composition=frozen_composition_executor,
        selected_composition=selected_composition_executor,
    )
    executors = {
        "layer10_only": layer10_executor,
        "selected_layer17_only": selected_layer17_executor,
        "frozen_uncorrected_composition": frozen_composition_executor,
        "selected_composition": selected_composition_executor,
    }
    aggregate = _new_metric_accumulator(_CONDITIONS)
    static_by_condition: dict[str, dict[str, object]] = {}
    totals_by_condition: dict[str, dict[str, int]] = {}
    peak_by_condition: dict[str, int] = {}
    logical_valid_tokens = 0
    native_model = adapter.module
    with ExitStack() as stack:
        for executor in executors.values():
            stack.enter_context(executor.validated_transaction())
        for batch in materialized:
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output), batch
            )
            _add_native(
                aggregate,
                nll_sum=_native_nll(native_logits, targets),
                token_count=targets.numel(),
            )
            expected_valid = int(batch.valid_positions.sum().item())
            observed_valid: list[int] = []
            for name, executor in executors.items():
                with torch.no_grad():
                    execution = executor.run(batch.model_inputs, condition="generated")
                _validate_dual_execution(
                    execution, executor, condition="generated", label=name
                )
                logits, candidate_targets = _selected_logits_and_targets(
                    _model_logits(execution.model_output), batch
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"A5d {name} held targets drifted")
                _add_comparison(
                    aggregate,
                    name,
                    _candidate_comparison(
                        native_logits,
                        logits,
                        targets,
                        vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                    ),
                )
                _record_execution(
                    static_by_condition,
                    totals_by_condition,
                    peak_by_condition,
                    condition=name,
                    execution=execution,
                )
                observed_valid.append(execution.valid_tokens)

            with torch.no_grad():
                deletion = selected_composition_executor.run(
                    batch.model_inputs, condition="deletion"
                )
            _validate_dual_execution(
                deletion,
                selected_composition_executor,
                condition="deletion",
                label="matched_double_deletion",
            )
            deletion_logits, deletion_targets = _selected_logits_and_targets(
                _model_logits(deletion.model_output), batch
            )
            if not torch.equal(targets, deletion_targets):
                raise RuntimeError("A5d matched-deletion targets drifted")
            _add_comparison(
                aggregate,
                "matched_double_deletion",
                _candidate_comparison(
                    native_logits,
                    deletion_logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                ),
            )
            _record_execution(
                static_by_condition,
                totals_by_condition,
                peak_by_condition,
                condition="matched_double_deletion",
                execution=deletion,
            )
            if any(
                getattr(deletion, field) != 0
                for field in (
                    "logical_executed_modal_graph_macs",
                    "logical_executed_modal_graph_additions",
                    "peak_live_modal_width",
                )
            ):
                raise RuntimeError("A5d deletion executed graph work")
            deletion_static = _execution_fields(
                deletion, _GRAPH_STATIC_FIELDS, label="matched_double_deletion"
            )
            if deletion_static != static_by_condition[
                "selected_composition"
            ]:
                raise RuntimeError("A5d generated/deletion replacement scope drifted")
            observed_valid.append(deletion.valid_tokens)
            if set(observed_valid) != {expected_valid}:
                raise RuntimeError("A5d conditions disagree on valid-token count")
            logical_valid_tokens += expected_valid

    metrics = _finalize_metric_accumulator(aggregate, conditions=_CONDITIONS)
    executor_by_condition = {
        **executors,
        "matched_double_deletion": selected_composition_executor,
    }
    resources = {
        name: _compact_resources(
            executor=executor_by_condition[name],
            static=static_by_condition[name],
            totals=totals_by_condition[name],
            logical_valid_tokens=logical_valid_tokens,
            peak_live_modal_width=peak_by_condition[name],
        )
        for name in _CONDITIONS
    }
    conditions: dict[str, dict[str, object]] = {}
    for name, metric in metrics["conditions"].items():
        executor = executor_by_condition[name]
        additive = _additive_plan(executor)
        conditions[name] = {
            **metric,
            "owning_graph_sha256": executor.graph_plan.artifact_sha256,
            "additive_graph_sha256": (
                None if additive is None else additive.artifact_sha256
            ),
        }
    source_parameters = int(
        static_by_condition["layer10_only"][
            "source_whole_model_learned_parameters"
        ]
    )
    return {
        "assessment_role": (
            "calibration_a_outer_family_bounded_source_anchored_residual"
        ),
        "outer_fold_index": 0,
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": metrics["supervised_tokens"],
        "logical_valid_tokens": logical_valid_tokens,
        "source_whole_model_learned_parameters": source_parameters,
        "native": metrics["native"],
        "conditions": conditions,
        "resource_accounting": resources,
        "exact_resources_match_frozen_executable": True,
        "latency_or_kernel_speed_claim": False,
    }
