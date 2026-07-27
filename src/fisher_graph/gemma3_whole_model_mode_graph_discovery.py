"""Pure fit-only whole-model mode-graph discovery for Gemma 3 adapters.

The runner in this module is deliberately narrower than an experiment entry
point.  It accepts an already constructed :class:`ModelAdapter` and an
already materialized calibration-A fit split, streams that exact split twice,
and returns an analysis-only artifact plus a JSON-safe report.

Every native MLP ``feed_forward_down_input`` is differentiated against the
same summed per-sequence language-model NLL.  A single equal-value detached
leaf at ``layer.0.input`` cuts the frozen prefix while retaining the complete
native suffix for every MLP score gradient.  The first pass builds bounded
sketches; the second pass replays only the frozen shortlist for exact pair
moments.  Neither output authorizes a held-out guard, an executor, or
calibration B.
"""

from __future__ import annotations

from collections.abc import (
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from typing import Iterator, Protocol

import torch
from torch import Tensor, nn

from .adapters.base import ModelAdapter, module_state_fingerprint
from .compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
    ScoreObjective,
)
from .instrumentation import InstrumentedModel
from .streaming_analysis import (
    ActivationScoreGradientRows,
    iter_activation_score_gradient_rows,
)
from .structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryProvenance,
    CrossBlockExactCriteria,
    CrossBlockLayerSpec,
    CrossBlockSketchConfig,
    build_cross_block_discovery_sketch,
    replay_cross_block_discovery_shortlist,
)


GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_SCHEMA = (
    "fisher_graph.gemma3_whole_model_mode_graph_discovery"
)
GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_FORMAT_VERSION = 1

_OBJECTIVE_DOMAIN = b"fisher_graph.gemma3_whole_model.objective.v1\0"
_FOLD_ASSIGNMENT_DOMAIN = (
    b"fisher_graph.gemma3_whole_model.family_fold_assignment.v1\0"
)
_STANDARD_ROW_FACTORY_ID = (
    "fisher_graph.streaming_analysis."
    "iter_activation_score_gradient_rows.v1"
)


class ActivationRowFactory(Protocol):
    """Replaceable exact row-stream boundary used by both discovery passes."""

    def __call__(
        self,
        model: InstrumentedModel,
        calibration_batches: Iterable[CalibrationBatch],
        *,
        activation_names: Collection[str],
        score_objective: ScoreObjective,
        leaf_activation_name: str | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> Iterable[ActivationScoreGradientRows]: ...


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _assert_no_tensor(value: object, *, label: str) -> None:
    if isinstance(value, Tensor):
        raise RuntimeError(f"{label} unexpectedly contains a Tensor")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_tensor(item, label=f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_tensor(item, label=f"{label}[{index}]")


def _source_parameter_snapshot(
    module: nn.Module,
) -> tuple[tuple[str, int, int, int, bool], ...]:
    return tuple(
        (
            name,
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter._version,
            parameter.requires_grad,
        )
        for name, parameter in module.named_parameters()
    )


def _validate_frozen_source(module: nn.Module) -> None:
    training_modules = tuple(
        name or "<root>"
        for name, child in module.named_modules()
        if child.training
    )
    if training_modules:
        raise ValueError(
            "whole-model discovery requires an eval-mode source; "
            f"training modules: {list(training_modules)}"
        )
    trainable_parameters = tuple(
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if trainable_parameters:
        raise ValueError(
            "whole-model discovery requires frozen source parameters; "
            f"trainable parameters: {list(trainable_parameters)}"
        )
    parameter_gradients = tuple(
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    )
    if parameter_gradients:
        raise ValueError(
            "whole-model discovery requires empty source gradients; "
            f"parameters with gradients: {list(parameter_gradients)}"
        )


def _materialized_fit_batches(
    batches: Sequence[CalibrationBatch],
) -> tuple[tuple[CalibrationBatch, ...], tuple[str, ...]]:
    if isinstance(batches, (str, bytes)) or not isinstance(
        batches,
        Sequence,
    ):
        raise TypeError(
            "calibration_fit_batches must be a materialized sequence"
        )
    materialized = tuple(batches)
    if not materialized:
        raise ValueError("calibration_fit_batches cannot be empty")
    if any(not isinstance(batch, CalibrationBatch) for batch in materialized):
        raise TypeError(
            "calibration_fit_batches must contain CalibrationBatch values"
        )
    missing_ids = tuple(
        index
        for index, batch in enumerate(materialized)
        if batch.example_ids is None
    )
    if missing_ids:
        raise ValueError(
            "whole-model discovery requires exact example_ids for every "
            f"fit batch; missing batch indices: {list(missing_ids)}"
        )
    example_ids = tuple(
        example_id
        for batch in materialized
        for example_id in (batch.example_ids or ())
    )
    if len(set(example_ids)) != len(example_ids):
        raise ValueError(
            "fit example_ids must be globally unique across materialized "
            "batches"
        )
    return materialized, example_ids


def _whole_model_layer_specs(
    adapter: ModelAdapter,
) -> tuple[tuple[CrossBlockLayerSpec, ...], str, tuple[str, ...]]:
    layers = adapter.layers
    if len(layers) < 2:
        raise ValueError(
            "whole-model cross-block discovery requires at least two layers"
        )
    ordinals = tuple(layer.ordinal for layer in layers)
    if ordinals != tuple(range(len(layers))):
        raise ValueError(
            "Gemma whole-model layers must be in contiguous ordinal order "
            "starting at zero"
        )
    first_layer = layers[0]
    if (
        first_layer.id != "layer.0"
        or first_layer.input_site != "layer.0.input"
    ):
        raise ValueError(
            "Gemma whole-model discovery requires layer.0.input as its "
            "single detached leaf"
        )
    leaf_site = adapter.activation_site(first_layer.input_site)
    if (
        leaf_site.axes != ("batch", "sequence", "feature")
        or leaf_site.width != first_layer.residual_width
        or not leaf_site.modal_eligible
        or not leaf_site.intervenable
    ):
        raise ValueError(
            "layer.0.input is not a canonical intervenable residual site"
        )

    specs: list[CrossBlockLayerSpec] = []
    for layer in layers:
        transformer = layer.transformer
        if transformer is None or transformer.operator_sites is None:
            raise ValueError(
                f"layer {layer.id!r} lacks structured MLP operator sites"
            )
        activation_site = (
            transformer.operator_sites.feed_forward_down_input
        )
        site = adapter.activation_site(activation_site)
        if (
            site.owner_layer != layer.id
            or site.axes != ("batch", "sequence", "feature")
            or site.width != transformer.feed_forward.intermediate_width
            or not site.modal_eligible
        ):
            raise ValueError(
                f"layer {layer.id!r} feed-forward down-input schema drifted"
            )
        specs.append(
            CrossBlockLayerSpec(
                layer_id=layer.id,
                layer_ordinal=layer.ordinal,
                activation_site=activation_site,
                width=transformer.feed_forward.intermediate_width,
            )
        )
    activation_sites = tuple(spec.activation_site for spec in specs)
    if len(set(activation_sites)) != len(activation_sites):
        raise ValueError("whole-model MLP activation sites must be unique")
    return tuple(specs), first_layer.input_site, activation_sites


def _resolved_row_factory_id(
    row_factory: ActivationRowFactory,
    row_factory_id: str | None,
    *,
    is_standard: bool,
) -> str:
    if row_factory_id is not None:
        if not isinstance(row_factory_id, str) or not row_factory_id:
            raise ValueError("row_factory_id must be nonempty when supplied")
        return row_factory_id
    if is_standard:
        return _STANDARD_ROW_FACTORY_ID
    module = getattr(row_factory, "__module__", None)
    name = getattr(row_factory, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(name, str)
        or not name
    ):
        raise ValueError(
            "a custom row factory must supply a stable row_factory_id"
        )
    return f"{module}.{name}"


def _resolved_fold_assignment(
    example_ids: tuple[str, ...],
    assignment: Mapping[str, int] | None,
    criteria: CrossBlockExactCriteria,
) -> tuple[
    Mapping[str, int] | None,
    CrossBlockExactCriteria,
    dict[str, object],
]:
    if assignment is None:
        return None, criteria, {
            "strategy": (
                "none"
                if criteria.fold_count < 2
                else "deterministic_example_id_hash"
            ),
            "family_disjoint_assignment_supplied": False,
            "fold_count": criteria.fold_count,
            "assignment_sha256": None,
            "examples_per_fold": None,
        }
    if not isinstance(assignment, Mapping):
        raise TypeError("family_fold_assignment must be a mapping")
    if any(
        not isinstance(example_id, str)
        or not example_id
        or type(fold) is not int
        or fold < 0
        for example_id, fold in assignment.items()
    ):
        raise ValueError(
            "family_fold_assignment must map nonempty example ids to "
            "nonnegative integer folds"
        )
    if set(assignment) != set(example_ids):
        missing = sorted(set(example_ids) - set(assignment))
        extra = sorted(set(assignment) - set(example_ids))
        raise ValueError(
            "family_fold_assignment must cover the exact fit example ids; "
            f"missing={missing}, extra={extra}"
        )
    normalized = {
        example_id: assignment[example_id]
        for example_id in sorted(assignment)
    }
    fold_ids = tuple(sorted(set(normalized.values())))
    if len(fold_ids) < 2 or fold_ids != tuple(range(len(fold_ids))):
        raise ValueError(
            "family folds must contain contiguous ids starting at zero and "
            "use at least two folds"
        )
    fold_count = len(fold_ids)
    if criteria.fold_count not in (0, fold_count):
        raise ValueError(
            "exact criteria fold_count disagrees with "
            "family_fold_assignment"
        )
    if criteria.fold_count == 0:
        criteria = replace(criteria, fold_count=fold_count)
    counts = tuple(
        sum(fold == index for fold in normalized.values())
        for index in range(fold_count)
    )
    return normalized, criteria, {
        "strategy": "caller_supplied_family_disjoint_assignment",
        "family_disjoint_assignment_supplied": True,
        "fold_count": fold_count,
        "assignment_sha256": _json_sha256(
            tuple(normalized.items()),
            domain=_FOLD_ASSIGNMENT_DOMAIN,
        ),
        "examples_per_fold": counts,
    }


@contextmanager
def _open_row_stream(
    row_factory: ActivationRowFactory,
    adapter: ModelAdapter,
    batches: tuple[CalibrationBatch, ...],
    *,
    activation_names: tuple[str, ...],
    objective: CausalLanguageModelNLL,
    leaf_activation_name: str,
) -> Iterator[Iterable[ActivationScoreGradientRows]]:
    rows = row_factory(
        adapter,
        batches,
        activation_names=activation_names,
        score_objective=objective,
        leaf_activation_name=leaf_activation_name,
        accumulation_dtype=torch.float64,
    )
    if not isinstance(rows, Iterable):
        raise TypeError(
            "activation row factory must return an iterable row stream"
        )
    try:
        yield rows
    finally:
        close = getattr(rows, "close", None)
        if callable(close):
            close()


def _validate_analysis_only_state(value: object) -> None:
    """Reject executable/source tensors while permitting bounded statistics."""

    forbidden_keys = {
        "model_state_dict",
        "executor_state_dict",
        "source_state_dict",
        "optimizer_state_dict",
        "weights",
        "parameters",
    }

    def walk(item: object, path: tuple[str, ...]) -> None:
        if isinstance(item, Tensor):
            if (
                item.device.type != "cpu"
                or item.dtype != torch.float64
                or not torch.isfinite(item).all()
            ):
                raise RuntimeError(
                    "discovery artifact tensors must be finite CPU float64 "
                    f"analysis statistics at {'.'.join(path)}"
                )
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise RuntimeError(
                        "discovery artifact mapping keys must be strings"
                    )
                if key in forbidden_keys:
                    raise RuntimeError(
                        "discovery artifact contains a forbidden executable "
                        f"or source field at {'.'.join((*path, key))}"
                    )
                walk(nested, (*path, key))
        elif isinstance(item, (tuple, list)):
            for index, nested in enumerate(item):
                walk(nested, (*path, str(index)))

    walk(value, ())


def run_gemma3_whole_model_mode_graph_discovery(
    adapter: ModelAdapter,
    calibration_fit_batches: Sequence[CalibrationBatch],
    *,
    calibration_fit_split_sha256: str,
    family_fold_assignment: Mapping[str, int] | None = None,
    sketch_config: CrossBlockSketchConfig = CrossBlockSketchConfig(),
    exact_criteria: CrossBlockExactCriteria = CrossBlockExactCriteria(),
    ignore_index: int = -100,
    row_factory: ActivationRowFactory | None = None,
    row_factory_id: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Discover whole-model cross-block MLP hypotheses on fit data only.

    ``calibration_fit_batches`` must be a replayable, materialized sequence.
    The function performs exactly two row-stream passes.  The injectable row
    factory is a narrow performance seam: a future batch-aware implementation
    can replace the present per-sequence iterator without changing the
    discovery or artifact contracts, but it must preserve the independent
    per-sequence summed-score semantics recorded below.
    """

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    split_sha256 = _require_sha256(
        calibration_fit_split_sha256,
        label="calibration_fit_split_sha256",
    )
    if not isinstance(sketch_config, CrossBlockSketchConfig):
        raise TypeError("sketch_config must be a CrossBlockSketchConfig")
    if not isinstance(exact_criteria, CrossBlockExactCriteria):
        raise TypeError("exact_criteria must be a CrossBlockExactCriteria")
    if type(ignore_index) is not int:
        raise TypeError("ignore_index must be an integer")
    resolved_factory = (
        iter_activation_score_gradient_rows
        if row_factory is None
        else row_factory
    )
    if not callable(resolved_factory):
        raise TypeError("row_factory must be callable")
    factory_id = _resolved_row_factory_id(
        resolved_factory,
        row_factory_id,
        is_standard=row_factory is None,
    )
    batches, example_ids = _materialized_fit_batches(
        calibration_fit_batches
    )
    layer_specs, leaf_site, mlp_sites = _whole_model_layer_specs(adapter)
    fold_assignment, criteria, fold_report = _resolved_fold_assignment(
        example_ids,
        family_fold_assignment,
        exact_criteria,
    )

    source = adapter.module
    _validate_frozen_source(source)
    source_state_before = module_state_fingerprint(source)
    source_parameters_before = _source_parameter_snapshot(source)
    model_fingerprint_before = adapter.model_fingerprint()
    execution_fingerprint_before = adapter.execution_fingerprint()
    objective_descriptor = {
        "name": "causal_language_model_nll",
        "target_kind": "hard_ground_truth_tokens",
        "reduction": "sum_per_independent_sequence",
        "normalizer": "valid_activation_positions",
        "ignore_index": ignore_index,
        "gradient_leaf": (
            "equal_value_detached_leaf_at_first_transformer_input"
        ),
        "gradient_scope": (
            "complete_native_model_suffix_for_every_mlp_down_input"
        ),
    }
    objective_sha256 = _json_sha256(
        objective_descriptor,
        domain=_OBJECTIVE_DOMAIN,
    )
    provenance = CrossBlockDiscoveryProvenance(
        model_fingerprint=model_fingerprint_before,
        calibration_split_sha256=split_sha256,
        objective_sha256=objective_sha256,
        score_reduction="sum_per_independent_sequence",
        normalizer="valid_activation_positions",
    )
    objective = CausalLanguageModelNLL(ignore_index=ignore_index)
    activation_names = (leaf_site, *mlp_sites)

    with _open_row_stream(
        resolved_factory,
        adapter,
        batches,
        activation_names=activation_names,
        objective=objective,
        leaf_activation_name=leaf_site,
    ) as rows:
        sketch = build_cross_block_discovery_sketch(
            rows,
            layer_specs=layer_specs,
            provenance=provenance,
            config=sketch_config,
        )
    with _open_row_stream(
        resolved_factory,
        adapter,
        batches,
        activation_names=activation_names,
        objective=objective,
        leaf_activation_name=leaf_site,
    ) as rows:
        discovery = replay_cross_block_discovery_shortlist(
            rows,
            sketch=sketch,
            criteria=criteria,
            fold_assignment=fold_assignment,
        )

    source_state_after = module_state_fingerprint(source)
    source_parameters_after = _source_parameter_snapshot(source)
    model_fingerprint_after = adapter.model_fingerprint()
    execution_fingerprint_after = adapter.execution_fingerprint()
    gradients_observed = tuple(
        name
        for name, parameter in source.named_parameters()
        if parameter.grad is not None
    )
    if (
        source_state_after != source_state_before
        or source_parameters_after != source_parameters_before
        or model_fingerprint_after != model_fingerprint_before
        or execution_fingerprint_after != execution_fingerprint_before
        or gradients_observed
    ):
        raise RuntimeError(
            "source model changed during whole-model mode discovery"
        )

    sketch_state = sketch.state_dict()
    discovery_state = discovery.state_dict()
    _validate_analysis_only_state(sketch_state)
    _validate_analysis_only_state(discovery_state)
    artifact: dict[str, object] = {
        "schema": GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_SCHEMA,
        "format_version": (
            GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_FORMAT_VERSION
        ),
        "sketch_state": sketch_state,
        "discovery_state": discovery_state,
        "binding": {
            "model_fingerprint": model_fingerprint_before,
            "execution_fingerprint": execution_fingerprint_before,
            "calibration_fit_split_sha256": split_sha256,
            "objective_sha256": objective_sha256,
        },
        "safety": {
            "contains_source_model_weights": False,
            "contains_executable_weights": False,
            "contains_optimizer_state": False,
            "contains_corpus_rows": False,
            "contains_activation_rows": False,
            "contains_score_gradient_rows": False,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "contains_teacher_targets": False,
            "discovery_only": True,
            "authorizes_static_merge": False,
            "authorizes_execution": False,
            "authorizes_executor_construction": False,
            "authorizes_calibration_a_guard": False,
            "authorizes_b": False,
            "authorizes_calibration_b": False,
        },
    }
    _validate_analysis_only_state(artifact)

    report: dict[str, object] = {
        "schema": GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_SCHEMA,
        "format_version": (
            GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_FORMAT_VERSION
        ),
        "scientific_status": {
            "outcome": "fit_only_whole_model_mode_graph_discovered",
            "discovery_completed": True,
            "calibration_a_fit_opened": True,
            "calibration_a_fit_completed": True,
            "scientific_compression_success": False,
            "executable_candidate_built": False,
            "calibration_a_guard_opened": False,
            "calibration_b_opened": False,
            "heldout_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "authorizes_execution": False,
            "authorizes_calibration_a_guard": False,
            "authorizes_b": False,
            "authorizes_calibration_b": False,
        },
        "model": {
            "model_fingerprint": model_fingerprint_before,
            "execution_fingerprint": execution_fingerprint_before,
            "layer_count": len(layer_specs),
            "mode_count": sum(spec.width for spec in layer_specs),
            "layers": tuple(spec.metadata() for spec in layer_specs),
        },
        "protocol": {
            "calibration_role": "calibration_a_fit_only",
            "calibration_fit_split_sha256": split_sha256,
            "materialized_batch_count": len(batches),
            "independent_sequence_count": len(example_ids),
            "streaming_passes": 2,
            "pass_1": "bounded_count_sketch_and_shortlist",
            "pass_2": "exact_shortlist_moments_only",
            "row_factory_id": factory_id,
            "row_factory_scope": "one_independent_sequence_per_row_value",
            "activation_sites": activation_names,
            "detached_leaf_sites": (leaf_site,),
            "objective": objective_descriptor,
            "objective_sha256": objective_sha256,
            "fisher_rank_source": (
                "pass_1_descending_signed_influence_energy_per_layer"
            ),
            "family_folds": fold_report,
            "guard_or_heldout_data_consumed": False,
        },
        "sketch": sketch.metadata(),
        "discovery": discovery.metadata(),
        "source_audit": {
            "source_model_executed": True,
            "source_model_role": "calibration_a_fit_teacher_only",
            "source_state_sha256_before": source_state_before,
            "source_state_sha256_after": source_state_after,
            "source_model_fingerprint_before": model_fingerprint_before,
            "source_model_fingerprint_after": model_fingerprint_after,
            "source_execution_fingerprint_before": (
                execution_fingerprint_before
            ),
            "source_execution_fingerprint_after": (
                execution_fingerprint_after
            ),
            "source_parameter_objects_preserved": True,
            "source_parameter_storages_preserved": True,
            "source_parameter_versions_preserved": True,
            "source_parameter_requires_grad_preserved": True,
            "source_parameter_gradients_observed": False,
            "source_parameters_frozen": True,
            "source_eval_mode_preserved": True,
        },
        "artifact": {
            "contains_source_model_weights": False,
            "contains_executable_weights": False,
            "contains_corpus_rows": False,
            "contains_activation_or_score_gradient_rows": False,
            "contains_bounded_analysis_tensors": True,
            "discovery_artifact_sha256": discovery.artifact_sha256,
            "authorizes_execution": False,
            "authorizes_executor_construction": False,
            "authorizes_calibration_a_guard": False,
            "authorizes_b": False,
            "authorizes_calibration_b": False,
        },
    }
    _assert_no_tensor(report, label="whole-model discovery report")
    return artifact, report


__all__ = [
    "ActivationRowFactory",
    "GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_FORMAT_VERSION",
    "GEMMA3_WHOLE_MODEL_MODE_GRAPH_DISCOVERY_SCHEMA",
    "run_gemma3_whole_model_mode_graph_discovery",
]
