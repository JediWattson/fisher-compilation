"""Paired Layer-17 rows on native and frozen Layer-10 graph trajectories.

The helper in this module deliberately owns no corpus or split access. Callers
provide already-materialized calibration batches, and the returned metadata
contains only hashes and structural provenance. Ephemeral row tensors remain on
the frozen result object and are never embedded in its metadata commitment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, ScoreObjective
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows
from .gemma3_same_layer_shape_flow import SameLayerFragmentSelection
from .gemma3_state_conditioned_shape_flow_experiment import (
    _collect_same_layer_native_rows,
)
from .streaming_analysis import (
    ActivationScoreGradientRows,
    iter_activation_score_gradient_rows,
)


__all__ = [
    "GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_FORMAT_VERSION",
    "GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_SCHEMA",
    "Gemma3Layer17TrajectoryRowPair",
    "capture_gemma3_layer17_native_and_layer10_rows",
]


GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_SCHEMA = (
    "fisher_graph.gemma3_layer17_native_layer10_trajectory_rows"
)
GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_FORMAT_VERSION = 2

_CAPTURE_DOMAIN = (
    b"fisher-graph:gemma3-layer17-native-layer10-trajectory-rows:v2\0"
)
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-layer17-native-layer10-trajectory-tensor:v2\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION = "generated"
_AFFECTED_LAYER_ORDINALS = (10,)
_LAYER17_ORDINAL = 17
_NATIVE_TEACHER_ROLE = (
    "native_layer17_fragment_residual_contribution_on_each_trajectory"
)
_NATIVE_INPUT_ROLE = "native_model_layer17_normalized_mlp_input"
_COMPILED_INPUT_ROLE = (
    "layer17_normalized_mlp_input_after_frozen_layer10_generated_graph"
)
_NATIVE_FULL_OUTPUT_ROLE = "native_model_layer17_full_mlp_output"
_COMPILED_FULL_OUTPUT_ROLE = (
    "native_layer17_full_mlp_output_after_frozen_layer10_generated_graph"
)
_EXACT_COMPACT_RETAINED_OUTPUT_ROLE = (
    "authenticated_layer17_compact_mlp_output_on_layer10_compiled_input"
)
_ALGEBRAIC_COMPACT_RETAINED_AUDIT_ROLE = (
    "compiled_full_minus_selected_compiled_contributions_numerical_audit_only"
)
_A3_TARGET_ROLE = (
    "native_full_minus_exact_authenticated_compact_retained_output"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise TypeError("tensor hash input must be a strided Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": tuple(int(width) for width in canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_full_output(
    value: Tensor,
    *,
    observations: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] != observations
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a floating [observations, output] Tensor"
        )
    canonical = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError(f"{label} must be finite")
    return canonical


def _fragment_tensor_hashes(
    rows: AlignedFragmentRows,
    fragment_ids: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    return {
        fragment_id: {
            "inputs_sha256": _tensor_sha256(
                rows.rows_by_fragment[fragment_id].inputs
            ),
            "contributions_sha256": _tensor_sha256(
                rows.rows_by_fragment[fragment_id].contributions
            ),
            "fisher_weights_sha256": _tensor_sha256(
                rows.rows_by_fragment[fragment_id].fisher_weights
            ),
        }
        for fragment_id in fragment_ids
    }


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_tensor(child) for child in value)
    return False


def _shared_fragment_inputs(
    rows: AlignedFragmentRows,
    fragment_ids: tuple[str, ...],
    *,
    label: str,
) -> Tensor:
    """Return the exact common same-layer normalized input row matrix."""

    reference = rows.rows_by_fragment[fragment_ids[0]].inputs
    for fragment_id in fragment_ids[1:]:
        value = rows.rows_by_fragment[fragment_id].inputs
        if value.shape != reference.shape or not torch.equal(value, reference):
            raise ValueError(
                f"{label} fragment inputs do not share exact values"
            )
    return reference


def _difference_statistics(value: Tensor) -> tuple[float, float]:
    canonical = value.detach().to(device="cpu", dtype=torch.float64)
    if canonical.numel() == 0 or not bool(torch.isfinite(canonical).all()):
        raise ValueError("compact-retained audit difference must be finite")
    return (
        float(canonical.abs().max().item()),
        float(torch.sqrt(torch.mean(canonical.square())).item()),
    )


@dataclass(frozen=True, slots=True)
class Gemma3Layer17TrajectoryRowPair:
    """Ephemeral paired rows plus their tensor-free lineage commitment."""

    native_rows: AlignedFragmentRows
    compiled_rows: AlignedFragmentRows
    native_full_mlp_output: Tensor
    compiled_full_mlp_output: Tensor
    compiled_compact_retained_mlp_output: Tensor
    model_fingerprint: str
    layer17_selection_sha256: str
    layer17_leaf_activation_site: str
    fragment_ids: tuple[str, ...]
    layer10_graph_sha256: str
    layer10_traversal_order: tuple[str, ...]
    ordered_layer10_lowering_sha256s: tuple[str, ...]
    layer17_compact_graph_sha256: str
    layer17_compact_traversal_order: tuple[str, ...]
    ordered_layer17_compact_lowering_sha256s: tuple[str, ...]
    capture_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.native_rows, AlignedFragmentRows) or not isinstance(
            self.compiled_rows,
            AlignedFragmentRows,
        ):
            raise TypeError("native_rows and compiled_rows must be aligned rows")
        for label, value in (
            ("model fingerprint", self.model_fingerprint),
            ("layer17 selection", self.layer17_selection_sha256),
            ("layer10 graph", self.layer10_graph_sha256),
            ("layer17 compact graph", self.layer17_compact_graph_sha256),
        ):
            _require_sha256(value, label=label)
        if (
            not isinstance(self.layer17_leaf_activation_site, str)
            or not self.layer17_leaf_activation_site
        ):
            raise ValueError("layer17 leaf activation site must be nonempty")
        if (
            type(self.fragment_ids) is not tuple
            or not self.fragment_ids
            or len(self.fragment_ids) != len(set(self.fragment_ids))
            or any(
                not isinstance(value, str) or not value
                for value in self.fragment_ids
            )
        ):
            raise ValueError("fragment_ids must be a nonempty unique tuple")
        native_catalog = tuple(self.native_rows.rows_by_fragment)
        compiled_catalog = tuple(self.compiled_rows.rows_by_fragment)
        if native_catalog != self.fragment_ids or compiled_catalog != self.fragment_ids:
            raise ValueError("paired rows do not match the exact fragment catalog")
        if self.native_rows.observations != self.compiled_rows.observations:
            raise ValueError("native and compiled observation counts differ")
        if self.native_rows.sequences != self.compiled_rows.sequences:
            raise ValueError("native and compiled sequence counts differ")
        if self.native_rows.row_keys != self.compiled_rows.row_keys:
            raise ValueError("native and compiled row keys differ")
        if self.native_rows.row_key_sha256 != self.compiled_rows.row_key_sha256:
            raise ValueError("native and compiled row-key hashes differ")
        for fragment_id in self.fragment_ids:
            native = self.native_rows.rows_by_fragment[fragment_id]
            compiled = self.compiled_rows.rows_by_fragment[fragment_id]
            if (
                native.inputs.shape != compiled.inputs.shape
                or native.contributions.shape != compiled.contributions.shape
                or native.fisher_weights.shape != compiled.fisher_weights.shape
            ):
                raise ValueError("native and compiled fragment tensor shapes differ")
        _shared_fragment_inputs(
            self.native_rows,
            self.fragment_ids,
            label="native",
        )
        _shared_fragment_inputs(
            self.compiled_rows,
            self.fragment_ids,
            label="compiled",
        )
        if (
            type(self.layer10_traversal_order) is not tuple
            or not self.layer10_traversal_order
            or len(self.layer10_traversal_order)
            != len(set(self.layer10_traversal_order))
            or any(
                not isinstance(value, str) or not value
                for value in self.layer10_traversal_order
            )
        ):
            raise ValueError("layer10 traversal order must be nonempty and unique")
        if (
            type(self.ordered_layer10_lowering_sha256s) is not tuple
            or len(self.ordered_layer10_lowering_sha256s)
            != len(self.layer10_traversal_order)
        ):
            raise ValueError("ordered Layer10 lowering catalog is incomplete")
        for value in self.ordered_layer10_lowering_sha256s:
            _require_sha256(value, label="Layer10 lowering")
        if (
            type(self.layer17_compact_traversal_order) is not tuple
            or len(self.layer17_compact_traversal_order) != len(self.fragment_ids)
            or len(self.layer17_compact_traversal_order)
            != len(set(self.layer17_compact_traversal_order))
            or any(
                not isinstance(value, str) or not value
                for value in self.layer17_compact_traversal_order
            )
        ):
            raise ValueError(
                "Layer17 compact traversal must exactly cover the fragments"
            )
        if (
            type(self.ordered_layer17_compact_lowering_sha256s) is not tuple
            or len(self.ordered_layer17_compact_lowering_sha256s)
            != len(self.layer17_compact_traversal_order)
        ):
            raise ValueError("ordered Layer17 compact lowering catalog is incomplete")
        for value in self.ordered_layer17_compact_lowering_sha256s:
            _require_sha256(value, label="Layer17 compact lowering")

        native_full = _canonical_full_output(
            self.native_full_mlp_output,
            observations=self.native_rows.observations,
            label="native full MLP output",
        )
        compiled_full = _canonical_full_output(
            self.compiled_full_mlp_output,
            observations=self.compiled_rows.observations,
            label="compiled-trajectory full MLP output",
        )
        if native_full.shape != compiled_full.shape:
            raise ValueError("native and compiled full MLP output shapes differ")
        compact_retained = _canonical_full_output(
            self.compiled_compact_retained_mlp_output,
            observations=self.compiled_rows.observations,
            label="compiled-trajectory compact retained MLP output",
        )
        if native_full.shape != compact_retained.shape:
            raise ValueError("compact retained and full MLP output shapes differ")
        if any(
            rows.contributions.shape != native_full.shape
            for rows in self.compiled_rows.rows_by_fragment.values()
        ):
            raise ValueError("compiled fragment contributions and full output differ")
        object.__setattr__(self, "native_full_mlp_output", native_full)
        object.__setattr__(self, "compiled_full_mlp_output", compiled_full)
        object.__setattr__(
            self,
            "compiled_compact_retained_mlp_output",
            compact_retained,
        )

        computed = hashlib.sha256(
            _CAPTURE_DOMAIN + _canonical_json_bytes(self._metadata_payload())
        ).hexdigest()
        if self.capture_sha256 == "":
            object.__setattr__(self, "capture_sha256", computed)
        elif _require_sha256(
            self.capture_sha256,
            label="capture",
        ) != computed:
            raise ValueError("trajectory row capture hash mismatch")

    @property
    def compiled_selected_fragment_contribution(self) -> Tensor:
        values = tuple(
            self.compiled_rows.rows_by_fragment[fragment_id].contributions
            for fragment_id in self.fragment_ids
        )
        result = torch.zeros_like(values[0])
        for value in values:
            result = result + value
        return result

    @property
    def a3_correction_target(self) -> Tensor:
        """Return native-full minus the exact compact retained replay."""

        return (
            self.native_full_mlp_output
            - self.compiled_compact_retained_mlp_output
        )

    @property
    def algebraic_compact_retained_mlp_output(self) -> Tensor:
        """Full-minus-selected retained term, for numerical audit only."""

        return (
            self.compiled_full_mlp_output
            - self.compiled_selected_fragment_contribution
        )

    @property
    def compact_retained_audit_difference(self) -> Tensor:
        """Exact runtime retained output minus its algebraic audit value."""

        return (
            self.compiled_compact_retained_mlp_output
            - self.algebraic_compact_retained_mlp_output
        )

    def _metadata_payload(self) -> dict[str, object]:
        audit_max_abs, audit_rms = _difference_statistics(
            self.compact_retained_audit_difference
        )
        return {
            "schema": GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_SCHEMA,
            "format_version": GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_FORMAT_VERSION,
            "scientific_role": "paired_native_and_layer10_compiled_layer17_rows",
            "source_safe": True,
            "contains_tensors": False,
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_token_ids": False,
            "model_fingerprint": self.model_fingerprint,
            "condition": _CONDITION,
            "affected_layer_ordinals": _AFFECTED_LAYER_ORDINALS,
            "layer10": {
                "graph_sha256": self.layer10_graph_sha256,
                "traversal_order": self.layer10_traversal_order,
                "ordered_lowering_sha256s": (
                    self.ordered_layer10_lowering_sha256s
                ),
            },
            "layer17": {
                "layer_ordinal": _LAYER17_ORDINAL,
                "selection_sha256": self.layer17_selection_sha256,
                "leaf_activation_site": self.layer17_leaf_activation_site,
                "fragment_ids": self.fragment_ids,
                "compact_executor": {
                    "graph_sha256": self.layer17_compact_graph_sha256,
                    "traversal_order": self.layer17_compact_traversal_order,
                    "ordered_lowering_sha256s": (
                        self.ordered_layer17_compact_lowering_sha256s
                    ),
                    "interaction_count": 0,
                    "affected_layer_ordinals": (_LAYER17_ORDINAL,),
                },
                "teacher_role": _NATIVE_TEACHER_ROLE,
                "input_roles": {
                    "native": _NATIVE_INPUT_ROLE,
                    "compiled": _COMPILED_INPUT_ROLE,
                },
                "full_output_roles": {
                    "native": _NATIVE_FULL_OUTPUT_ROLE,
                    "compiled": _COMPILED_FULL_OUTPUT_ROLE,
                },
                "compact_retained_output_role": (
                    _EXACT_COMPACT_RETAINED_OUTPUT_ROLE
                ),
                "algebraic_compact_retained_audit_role": (
                    _ALGEBRAIC_COMPACT_RETAINED_AUDIT_ROLE
                ),
                "a3_target_role": _A3_TARGET_ROLE,
            },
            "alignment": {
                "fragment_count": len(self.fragment_ids),
                "observations": self.native_rows.observations,
                "sequences": self.native_rows.sequences,
                "row_key_sha256": self.native_rows.row_key_sha256,
            },
            "tensor_sha256s": {
                "native_rows": _fragment_tensor_hashes(
                    self.native_rows,
                    self.fragment_ids,
                ),
                "compiled_rows": _fragment_tensor_hashes(
                    self.compiled_rows,
                    self.fragment_ids,
                ),
                "full_mlp_outputs": {
                    "native_sha256": _tensor_sha256(
                        self.native_full_mlp_output
                    ),
                    "compiled_sha256": _tensor_sha256(
                        self.compiled_full_mlp_output
                    ),
                    "exact_compact_retained_sha256": _tensor_sha256(
                        self.compiled_compact_retained_mlp_output
                    ),
                    "algebraic_compact_retained_audit_sha256": _tensor_sha256(
                        self.algebraic_compact_retained_mlp_output
                    ),
                    "compact_retained_audit_difference_sha256": _tensor_sha256(
                        self.compact_retained_audit_difference
                    ),
                    "a3_correction_target_sha256": _tensor_sha256(
                        self.a3_correction_target
                    ),
                },
            },
            "compact_retained_numerical_audit": {
                "role": _ALGEBRAIC_COMPACT_RETAINED_AUDIT_ROLE,
                "difference_definition": (
                    "exact_compact_retained_minus_compiled_full_minus_"
                    "selected_compiled_contributions"
                ),
                "max_abs_difference": audit_max_abs,
                "rms_difference": audit_rms,
            },
        }

    def metadata(self) -> dict[str, object]:
        payload = self._metadata_payload()
        if _contains_tensor(payload):
            raise RuntimeError("trajectory row metadata contains a Tensor")
        computed = hashlib.sha256(
            _CAPTURE_DOMAIN + _canonical_json_bytes(payload)
        ).hexdigest()
        if computed != self.capture_sha256:
            raise RuntimeError("trajectory row capture tensors or lineage drifted")
        return {**payload, "capture_sha256": self.capture_sha256}


def _capture_rows_and_full_output(
    adapter: Gemma3CausalLMAdapter,
    batches: tuple[CalibrationBatch, ...],
    *,
    selection: SameLayerFragmentSelection,
    leaf_activation_site: str,
) -> tuple[AlignedFragmentRows, Tensor]:
    """Extend the existing aligned collector with its same-pass full output."""

    output_site = selection.execution_order[0].output_site
    outputs: list[Tensor] = []
    output_row_keys: list[tuple[str, int]] = []

    def row_factory(
        model: Gemma3CausalLMAdapter,
        calibration_batches: Iterable[CalibrationBatch],
        *,
        activation_names: Sequence[str],
        score_objective: ScoreObjective,
        leaf_activation_name: str | None = None,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> Iterable[ActivationScoreGradientRows]:
        requested = tuple(dict.fromkeys((*activation_names, output_site)))
        raw_rows = iter_activation_score_gradient_rows(
            model,
            calibration_batches,
            activation_names=requested,
            score_objective=score_objective,
            leaf_activation_name=leaf_activation_name,
            accumulation_dtype=accumulation_dtype,
        )

        def observe() -> Iterable[ActivationScoreGradientRows]:
            iterator = iter(raw_rows)
            try:
                for row in iterator:
                    if output_site not in row.activations or row.example_id is None:
                        raise ValueError(
                            "full Layer17 output row is missing activation or identity"
                        )
                    output = row.activations[output_site].detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    ).contiguous()
                    if (
                        output.ndim != 2
                        or output.shape[0] != row.logical_positions.numel()
                        or not bool(torch.isfinite(output).all())
                    ):
                        raise ValueError("full Layer17 output rows are invalid")
                    outputs.append(output)
                    output_row_keys.extend(
                        (row.example_id, int(position))
                        for position in row.logical_positions.tolist()
                    )
                    yield row
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

        return observe()

    rows = _collect_same_layer_native_rows(
        adapter,
        batches,
        selection=selection,
        leaf_activation_site=leaf_activation_site,
        row_factory=row_factory,
    )
    if not isinstance(rows, AlignedFragmentRows):
        raise TypeError("same-layer collector returned invalid aligned rows")
    if tuple(output_row_keys) != rows.row_keys or not outputs:
        raise RuntimeError("full Layer17 outputs are not aligned to fragment rows")
    return rows, torch.cat(outputs, dim=0)


def _layer10_executor_binding(
    adapter: Gemma3CausalLMAdapter,
    selection: SameLayerFragmentSelection,
    executor: Gemma3ModalGeneratorGraphExecutor,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    if not isinstance(executor, Gemma3ModalGeneratorGraphExecutor):
        raise TypeError("layer10_executor must be a graph executor")
    if executor.adapter is not adapter:
        raise ValueError("Layer10 executor and row adapter differ")
    executor.graph_plan.validate_integrity()
    if executor.affected_layer_ordinals != _AFFECTED_LAYER_ORDINALS:
        raise ValueError("row capture requires an exact Layer10-only graph")
    bound_nodes = getattr(executor, "_bound_nodes", None)
    if type(bound_nodes) is not tuple or not bound_nodes:
        raise ValueError("Layer10 executor has no authenticated bound nodes")
    traversal_order = tuple(bound.node.name for bound in bound_nodes)
    if traversal_order != executor.graph_plan.traversal_order or any(
        bound.fragment.layer_ordinal != 10 for bound in bound_nodes
    ):
        raise ValueError("Layer10 executor bound-node order drifted")
    lowering_sha256s = tuple(
        _require_sha256(
            bound.lowering.artifact_sha256,
            label="Layer10 lowering",
        )
        for bound in bound_nodes
    )
    model_fingerprint = _require_sha256(
        executor.graph_plan.model_fingerprint,
        label="graph model fingerprint",
    )
    if selection.source_model_sha256 != model_fingerprint:
        raise ValueError("Layer10 graph and Layer17 selection model differ")
    if adapter.model_fingerprint() != model_fingerprint:
        raise ValueError("live adapter and Layer10 graph model differ")
    return (
        _require_sha256(
            executor.graph_plan.artifact_sha256,
            label="Layer10 graph",
        ),
        traversal_order,
        lowering_sha256s,
        model_fingerprint,
    )


def _layer17_compact_executor_binding(
    adapter: Gemma3CausalLMAdapter,
    selection: SameLayerFragmentSelection,
    executor: Gemma3ModalGeneratorGraphExecutor,
    *,
    model_fingerprint: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Authenticate the exact edgeless Layer17 compact replay operator."""

    if not isinstance(executor, Gemma3ModalGeneratorGraphExecutor):
        raise TypeError("layer17_executor must be a graph executor")
    if executor.adapter is not adapter:
        raise ValueError("Layer17 executor and row adapter differ")
    executor.graph_plan.validate_integrity()
    if executor.affected_layer_ordinals != (_LAYER17_ORDINAL,):
        raise ValueError("row capture requires an exact Layer17-only graph")
    if executor.graph_plan.interactions:
        raise ValueError("Layer17 compact replay graph must be edgeless")
    if executor.graph_plan.model_fingerprint != model_fingerprint:
        raise ValueError("Layer10 and Layer17 executor models differ")

    bound_nodes = getattr(executor, "_bound_nodes", None)
    if type(bound_nodes) is not tuple or not bound_nodes:
        raise ValueError("Layer17 executor has no authenticated bound nodes")
    traversal_order = tuple(bound.node.name for bound in bound_nodes)
    selected = tuple(selection.execution_order)
    if (
        traversal_order != executor.graph_plan.traversal_order
        or len(bound_nodes) != len(selected)
        or tuple(bound.fragment.fragment_id for bound in bound_nodes)
        != selection.fragment_ids
        or tuple(
            bound.fragment.artifact_sha256 for bound in bound_nodes
        )
        != tuple(fragment.artifact_sha256 for fragment in selected)
        or any(
            bound.fragment.layer_ordinal != _LAYER17_ORDINAL
            or bound.node.input_boundary != fragment.input_site
            or bound.node.output_boundary != fragment.output_site
            for bound, fragment in zip(bound_nodes, selected, strict=True)
        )
    ):
        raise ValueError(
            "Layer17 compact executor topology differs from the selection"
        )
    if tuple(executor.compiled_mlps) != (str(_LAYER17_ORDINAL),):
        raise ValueError("Layer17 compact MLP catalog is not exact")
    compact = executor.compiled_mlps[str(_LAYER17_ORDINAL)]
    selected_channels = tuple(
        sorted(
            channel
            for fragment in selected
            for channel in fragment.channel_indices
        )
    )
    if compact.removed_mode_indices != selected_channels:
        raise ValueError("Layer17 compact removed-mode union drifted")
    lowering_sha256s = tuple(
        _require_sha256(
            bound.lowering.artifact_sha256,
            label="Layer17 compact lowering",
        )
        for bound in bound_nodes
    )
    return (
        _require_sha256(
            executor.graph_plan.artifact_sha256,
            label="Layer17 compact graph",
        ),
        traversal_order,
        lowering_sha256s,
    )


def capture_gemma3_layer17_native_and_layer10_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    selection: SameLayerFragmentSelection,
    leaf_activation_site: str,
    layer10_executor: Gemma3ModalGeneratorGraphExecutor,
    layer17_executor: Gemma3ModalGeneratorGraphExecutor,
) -> Gemma3Layer17TrajectoryRowPair:
    """Capture paired rows plus the exact retained Layer17 compact replay."""

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("batches must contain CalibrationBatch values")
    if not isinstance(selection, SameLayerFragmentSelection):
        raise TypeError("selection must be a SameLayerFragmentSelection")
    selection.validate_integrity()
    if selection.layer_ordinal != _LAYER17_ORDINAL:
        raise ValueError("row capture requires a Layer17 selection")
    selected_leaf = selection.execution_order[0].input_site
    if leaf_activation_site != selected_leaf:
        raise ValueError("leaf activation site must be the Layer17 normalized input")
    graph_sha256, traversal, lowerings, model_fingerprint = (
        _layer10_executor_binding(adapter, selection, layer10_executor)
    )
    (
        layer17_graph_sha256,
        layer17_traversal,
        layer17_lowerings,
    ) = _layer17_compact_executor_binding(
        adapter,
        selection,
        layer17_executor,
        model_fingerprint=model_fingerprint,
    )

    native_rows, native_full = _capture_rows_and_full_output(
        adapter,
        materialized,
        selection=selection,
        leaf_activation_site=leaf_activation_site,
    )
    _shared_fragment_inputs(
        native_rows,
        tuple(selection.fragment_ids),
        label="native",
    )

    def capture_compiled() -> tuple[AlignedFragmentRows, Tensor]:
        return _capture_rows_and_full_output(
            adapter,
            materialized,
            selection=selection,
            leaf_activation_site=leaf_activation_site,
        )

    compiled_rows, compiled_full = layer10_executor.run_with_generated_overlay(
        capture_compiled,
        expected_forward_calls=sum(batch.batch_size for batch in materialized),
    )
    compiled_inputs = _shared_fragment_inputs(
        compiled_rows,
        tuple(selection.fragment_ids),
        label="compiled",
    )
    compact_retained = layer17_executor.execute_compact_mlp_rows(
        _LAYER17_ORDINAL,
        compiled_inputs,
    )
    return Gemma3Layer17TrajectoryRowPair(
        native_rows=native_rows,
        compiled_rows=compiled_rows,
        native_full_mlp_output=native_full,
        compiled_full_mlp_output=compiled_full,
        compiled_compact_retained_mlp_output=compact_retained,
        model_fingerprint=model_fingerprint,
        layer17_selection_sha256=selection.artifact_sha256,
        layer17_leaf_activation_site=leaf_activation_site,
        fragment_ids=tuple(selection.fragment_ids),
        layer10_graph_sha256=graph_sha256,
        layer10_traversal_order=traversal,
        ordered_layer10_lowering_sha256s=lowerings,
        layer17_compact_graph_sha256=layer17_graph_sha256,
        layer17_compact_traversal_order=layer17_traversal,
        ordered_layer17_compact_lowering_sha256s=layer17_lowerings,
    )
