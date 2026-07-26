"""Deterministic Fisher/Taylor compression of structured Gemma MLP width.

This module implements the first post-parent compression rung without loading
or inspecting a native model. A score-gradient capture ranks complete gated
MLP units, construction slices paired gate/up rows and down-projection
columns from a source-free parent executor, and a target transform prepares
activation-only inputs for the existing terminal-projection ridge refit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .structured_layer_distillation import (
    StructuredLayerProvenance,
    StructuredLayerTargets,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_MLP_FISHER_TAYLOR_ALGORITHM = (
    "valid_row_squared_activation_score_gradient_stable_topk_v1"
)
STRUCTURED_MLP_COMPRESSION_SCHEMA = (
    "fisher_graph.structured_mlp_width_compression"
)
STRUCTURED_MLP_COMPRESSION_FORMAT_VERSION = 1
GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH = 2_048
GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH = 1_536

_SCORE_DOMAIN = b"fisher_graph.structured_mlp.score.v1\0"
_BATCH_DOMAIN = b"fisher_graph.structured_mlp.batch.v1\0"
_BATCH_SET_DOMAIN = b"fisher_graph.structured_mlp.batch_set.v1\0"
_SELECTION_DOMAIN = b"fisher_graph.structured_mlp.selection.v1\0"
_INDEX_DOMAIN = b"fisher_graph.structured_mlp.indices.v1\0"
_STATE_DOMAIN = b"fisher_graph.structured_mlp.state.v1\0"
_CONFIG_DOMAIN = b"fisher_graph.structured_mlp.config.v1\0"
_REFIT_INPUT_DOMAIN = b"fisher_graph.structured_mlp.refit_input.v1\0"


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
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


def _tensor_mapping_sha256(
    values: Mapping[str, Tensor],
    *,
    domain: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for name in sorted(values):
        tensor = values[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _provenance_dict(
    provenance: StructuredLayerProvenance,
) -> dict[str, str]:
    return {
        "layer_id": provenance.layer_id,
        "output_site": provenance.output_site,
        "source_segment_fingerprint": (
            provenance.source_segment_fingerprint
        ),
    }


@dataclass(frozen=True, slots=True)
class StructuredMLPFisherTaylorBatch:
    """Pre-down activations and their score gradients for one batch."""

    provenance: StructuredLayerProvenance
    batch_id: str
    projection_input: Tensor
    score_gradient: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError(
                "provenance must be StructuredLayerProvenance"
            )
        if not isinstance(self.batch_id, str) or not self.batch_id:
            raise ValueError("batch_id must be nonempty")
        if (
            not isinstance(self.projection_input, Tensor)
            or not isinstance(self.score_gradient, Tensor)
            or self.projection_input.ndim != 3
            or self.projection_input.shape != self.score_gradient.shape
            or self.projection_input.shape[-1] <= 0
            or not self.projection_input.is_floating_point()
            or not self.score_gradient.is_floating_point()
            or self.projection_input.device != self.score_gradient.device
            or not isinstance(self.valid_mask, Tensor)
            or self.valid_mask.dtype is not torch.bool
            or self.valid_mask.shape != self.projection_input.shape[:2]
            or self.valid_mask.device != self.projection_input.device
            or not bool(self.valid_mask.any())
        ):
            raise ValueError(
                "Fisher/Taylor batch tensors have incompatible schemas"
            )
        if (
            not bool(
                torch.isfinite(
                    self.projection_input[self.valid_mask]
                ).all()
            )
            or not bool(
                torch.isfinite(
                    self.score_gradient[self.valid_mask]
                ).all()
            )
        ):
            raise ValueError(
                "Fisher/Taylor valid activation rows must be finite"
            )

    @property
    def source_width(self) -> int:
        return int(self.projection_input.shape[-1])

    @property
    def valid_rows(self) -> int:
        return int(self.valid_mask.sum().item())

    @classmethod
    def from_structured_targets(
        cls,
        targets: StructuredLayerTargets,
        score_gradient: Tensor,
        *,
        batch_id: str,
    ) -> StructuredMLPFisherTaylorBatch:
        if not isinstance(targets, StructuredLayerTargets):
            raise TypeError("targets must be StructuredLayerTargets")
        if targets.feed_forward_projection_input is None:
            raise ValueError(
                "structured targets do not contain pre-down features"
            )
        return cls(
            provenance=targets.provenance,
            batch_id=batch_id,
            projection_input=targets.feed_forward_projection_input,
            score_gradient=score_gradient,
            valid_mask=targets.sequence.query_valid_mask,
        )

    def input_sha256(self) -> str:
        valid = self.valid_mask
        return _tensor_mapping_sha256(
            {
                "projection_input": self.projection_input[valid],
                "score_gradient": self.score_gradient[valid],
            },
            domain=_BATCH_DOMAIN,
        )


def _selection_payload(
    *,
    provenance: StructuredLayerProvenance,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
    source_width: int,
    retained_width: int,
    valid_rows: int,
    batch_ids: tuple[str, ...],
    input_batches_sha256: str,
    unit_scores_sha256: str,
    ranked_indices: tuple[int, ...],
    selected_indices: tuple[int, ...],
) -> dict[str, object]:
    return {
        "algorithm": STRUCTURED_MLP_FISHER_TAYLOR_ALGORITHM,
        "provenance": _provenance_dict(provenance),
        "calibration_split_sha256": calibration_split_sha256,
        "activation_site": activation_site,
        "parent_executor_fingerprint": parent_executor_fingerprint,
        "source_width": source_width,
        "retained_width": retained_width,
        "valid_rows": valid_rows,
        "batch_ids": batch_ids,
        "input_batches_sha256": input_batches_sha256,
        "unit_scores_sha256": unit_scores_sha256,
        "ranked_indices": ranked_indices,
        "selected_indices": selected_indices,
        "score_semantics": (
            "mean_valid_row_squared_activation_times_score_gradient"
        ),
        "tie_break": "stable_source_unit_index",
        "construction_order": "ascending_source_unit_index",
    }


@dataclass(frozen=True, slots=True)
class StructuredMLPUnitSelection:
    """Authenticated deterministic selection of complete gated MLP units."""

    provenance: StructuredLayerProvenance
    calibration_split_sha256: str
    activation_site: str
    parent_executor_fingerprint: str
    source_width: int
    retained_width: int
    valid_rows: int
    batch_ids: tuple[str, ...]
    input_batches_sha256: str
    unit_scores: Tensor
    unit_scores_sha256: str
    ranked_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    selection_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError(
                "provenance must be StructuredLayerProvenance"
            )
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        _require_sha256(
            self.parent_executor_fingerprint,
            label="parent_executor_fingerprint",
        )
        _require_sha256(
            self.input_batches_sha256,
            label="input_batches_sha256",
        )
        _require_sha256(
            self.unit_scores_sha256,
            label="unit_scores_sha256",
        )
        _require_sha256(
            self.selection_sha256,
            label="selection_sha256",
        )
        if not isinstance(self.activation_site, str) or not self.activation_site:
            raise ValueError("activation_site must be nonempty")
        if (
            type(self.source_width) is not int
            or type(self.retained_width) is not int
            or not 0 < self.retained_width < self.source_width
            or type(self.valid_rows) is not int
            or self.valid_rows <= 0
            or type(self.batch_ids) is not tuple
            or not self.batch_ids
            or tuple(sorted(self.batch_ids)) != self.batch_ids
            or len(set(self.batch_ids)) != len(self.batch_ids)
            or any(
                not isinstance(batch_id, str) or not batch_id
                for batch_id in self.batch_ids
            )
        ):
            raise ValueError("MLP unit selection metadata is invalid")
        scores = self.unit_scores.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        if (
            scores.shape != (self.source_width,)
            or not bool(torch.isfinite(scores).all())
            or bool((scores < 0).any())
            or float(scores.sum().item()) <= 0
        ):
            raise ValueError(
                "unit_scores must be one finite nonnegative vector with "
                "positive total importance"
            )
        object.__setattr__(self, "unit_scores", scores.clone())
        if _tensor_mapping_sha256(
            {"unit_scores": scores},
            domain=_SCORE_DOMAIN,
        ) != self.unit_scores_sha256:
            raise ValueError("unit score digest is invalid")
        expected_ranked = tuple(
            int(index)
            for index in torch.argsort(
                scores,
                descending=True,
                stable=True,
            ).tolist()
        )
        expected_selected = tuple(
            sorted(expected_ranked[: self.retained_width])
        )
        if (
            self.ranked_indices != expected_ranked
            or self.selected_indices != expected_selected
            or len(self.ranked_indices) != self.source_width
            or len(self.selected_indices) != self.retained_width
        ):
            raise ValueError(
                "ranked or selected MLP unit indices are invalid"
            )
        payload = _selection_payload(
            provenance=self.provenance,
            calibration_split_sha256=self.calibration_split_sha256,
            activation_site=self.activation_site,
            parent_executor_fingerprint=(
                self.parent_executor_fingerprint
            ),
            source_width=self.source_width,
            retained_width=self.retained_width,
            valid_rows=self.valid_rows,
            batch_ids=self.batch_ids,
            input_batches_sha256=self.input_batches_sha256,
            unit_scores_sha256=self.unit_scores_sha256,
            ranked_indices=self.ranked_indices,
            selected_indices=self.selected_indices,
        )
        if _json_sha256(
            payload,
            domain=_SELECTION_DOMAIN,
        ) != self.selection_sha256:
            raise ValueError("MLP unit selection digest is invalid")

    @property
    def removed_width(self) -> int:
        return self.source_width - self.retained_width

    @property
    def retained_score_fraction(self) -> float:
        total = float(self.unit_scores.sum().item())
        if total <= 0:
            return 0.0
        selected = torch.tensor(
            self.selected_indices,
            dtype=torch.long,
        )
        return float(self.unit_scores[selected].sum().item() / total)

    def metadata(self) -> dict[str, object]:
        payload = _selection_payload(
            provenance=self.provenance,
            calibration_split_sha256=self.calibration_split_sha256,
            activation_site=self.activation_site,
            parent_executor_fingerprint=(
                self.parent_executor_fingerprint
            ),
            source_width=self.source_width,
            retained_width=self.retained_width,
            valid_rows=self.valid_rows,
            batch_ids=self.batch_ids,
            input_batches_sha256=self.input_batches_sha256,
            unit_scores_sha256=self.unit_scores_sha256,
            ranked_indices=self.ranked_indices,
            selected_indices=self.selected_indices,
        )
        return {
            **payload,
            "removed_width": self.removed_width,
            "retained_score_fraction": self.retained_score_fraction,
            "selection_sha256": self.selection_sha256,
        }


def select_fisher_taylor_mlp_units(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
    retained_width: int,
    expected_source_width: int | None = None,
) -> StructuredMLPUnitSelection:
    """Rank paired MLP units by squared first-order score damage."""

    if not batches:
        raise ValueError("Fisher/Taylor batches cannot be empty")
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    _require_sha256(
        parent_executor_fingerprint,
        label="parent_executor_fingerprint",
    )
    if not isinstance(activation_site, str) or not activation_site:
        raise ValueError("activation_site must be nonempty")
    if type(retained_width) is not int:
        raise TypeError("retained_width must be an integer")
    if expected_source_width is not None and (
        type(expected_source_width) is not int
        or expected_source_width <= 0
    ):
        raise ValueError(
            "expected_source_width must be a positive integer"
        )
    if any(
        not isinstance(batch, StructuredMLPFisherTaylorBatch)
        for batch in batches
    ):
        raise TypeError(
            "batches must contain StructuredMLPFisherTaylorBatch values"
        )
    ordered = tuple(sorted(batches, key=lambda batch: batch.batch_id))
    batch_ids = tuple(batch.batch_id for batch in ordered)
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Fisher/Taylor batch ids must be unique")
    provenance = ordered[0].provenance
    source_width = ordered[0].source_width
    if (
        not 0 < retained_width < source_width
        or (
            expected_source_width is not None
            and source_width != expected_source_width
        )
        or any(
            batch.provenance != provenance
            or batch.source_width != source_width
            for batch in ordered
        )
    ):
        raise ValueError(
            "Fisher/Taylor batches, widths, or provenance are inconsistent"
        )

    score_sum = torch.zeros(source_width, dtype=torch.float64)
    valid_rows = 0
    batch_records = []
    for batch in ordered:
        valid = batch.valid_mask
        activations = batch.projection_input[valid].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        gradients = batch.score_gradient[valid].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        score_sum.add_(
            (activations * gradients).square().sum(dim=0)
        )
        valid_rows += activations.shape[0]
        batch_records.append(
            {
                "batch_id": batch.batch_id,
                "valid_rows": batch.valid_rows,
                "input_sha256": batch.input_sha256(),
            }
        )
    if (
        valid_rows <= 0
        or not bool(torch.isfinite(score_sum).all())
        or float(score_sum.sum().item()) <= 0
    ):
        raise RuntimeError("Fisher/Taylor score accumulation is invalid")
    scores = score_sum / valid_rows
    ranked = tuple(
        int(index)
        for index in torch.argsort(
            scores,
            descending=True,
            stable=True,
        ).tolist()
    )
    selected = tuple(sorted(ranked[:retained_width]))
    scores_sha256 = _tensor_mapping_sha256(
        {"unit_scores": scores},
        domain=_SCORE_DOMAIN,
    )
    input_batches_sha256 = _json_sha256(
        batch_records,
        domain=_BATCH_SET_DOMAIN,
    )
    payload = _selection_payload(
        provenance=provenance,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        source_width=source_width,
        retained_width=retained_width,
        valid_rows=valid_rows,
        batch_ids=batch_ids,
        input_batches_sha256=input_batches_sha256,
        unit_scores_sha256=scores_sha256,
        ranked_indices=ranked,
        selected_indices=selected,
    )
    return StructuredMLPUnitSelection(
        provenance=provenance,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        source_width=source_width,
        retained_width=retained_width,
        valid_rows=valid_rows,
        batch_ids=batch_ids,
        input_batches_sha256=input_batches_sha256,
        unit_scores=scores,
        unit_scores_sha256=scores_sha256,
        ranked_indices=ranked,
        selected_indices=selected,
        selection_sha256=_json_sha256(
            payload,
            domain=_SELECTION_DOMAIN,
        ),
    )


def select_gemma_mlp_first_rung_units(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
) -> StructuredMLPUnitSelection:
    """Select the preregistered first Gemma MLP rung, 2048 -> 1536."""

    return select_fisher_taylor_mlp_units(
        batches,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        retained_width=GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH,
        expected_source_width=GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
    )


def _state_sha256(
    executor: StructuredTransformerLayerExecutor,
) -> str:
    return _tensor_mapping_sha256(
        executor.state_dict(),
        domain=_STATE_DOMAIN,
    )


def _config_sha256(
    executor: StructuredTransformerLayerExecutor,
) -> str:
    return _json_sha256(
        executor.config.to_dict(),
        domain=_CONFIG_DOMAIN,
    )


def _indices_sha256(indices: tuple[int, ...]) -> str:
    return _json_sha256(indices, domain=_INDEX_DOMAIN)


def _copy_compressed_state(
    source: StructuredTransformerLayerExecutor,
    destination: StructuredTransformerLayerExecutor,
    selection: StructuredMLPUnitSelection,
) -> tuple[str, ...]:
    source_state = source.state_dict()
    destination_state = destination.state_dict()
    if set(source_state) != set(destination_state):
        raise ValueError(
            "structured executor state schema drifted during compression"
        )
    selected = torch.tensor(
        selection.selected_indices,
        dtype=torch.long,
    )
    row_slices = {
        "feed_forward.gate_proj.weight",
        "feed_forward.up_proj.weight",
    }
    if source.feed_forward.gate_proj.bias is not None:
        row_slices.update(
            {
                "feed_forward.gate_proj.bias",
                "feed_forward.up_proj.bias",
            }
        )
    column_slices = {"feed_forward.down_proj.weight"}
    expected_changed = row_slices | column_slices
    candidate: dict[str, Tensor] = {}
    preserved = []
    for name, source_value in source_state.items():
        destination_value = destination_state[name]
        local_indices = selected.to(device=source_value.device)
        if name in row_slices:
            value = source_value.index_select(0, local_indices)
        elif name in column_slices:
            value = source_value.index_select(1, local_indices)
        else:
            value = source_value
            preserved.append(name)
        if value.shape != destination_value.shape:
            raise ValueError(
                f"compressed tensor shape drifted for {name!r}: "
                f"{tuple(value.shape)} != {tuple(destination_value.shape)}"
            )
        candidate[name] = value.detach().to(
            device=destination_value.device,
            dtype=destination_value.dtype,
        ).clone()
    changed_shapes = {
        name
        for name in source_state
        if source_state[name].shape != destination_state[name].shape
    }
    if changed_shapes != expected_changed:
        raise ValueError(
            "compressed MLP tensor schema drifted: "
            f"expected={sorted(expected_changed)}, "
            f"observed={sorted(changed_shapes)}"
        )
    destination.load_state_dict(candidate, strict=True)
    return tuple(sorted(preserved))


def build_width_compressed_structured_executor(
    source: StructuredTransformerLayerExecutor,
    selection: StructuredMLPUnitSelection,
) -> tuple[StructuredTransformerLayerExecutor, dict[str, object]]:
    """Construct a source-free executor with paired intermediate-unit slices."""

    if not isinstance(source, StructuredTransformerLayerExecutor):
        raise TypeError(
            "source must be StructuredTransformerLayerExecutor"
        )
    if not isinstance(selection, StructuredMLPUnitSelection):
        raise TypeError("selection must be StructuredMLPUnitSelection")
    if source.owns_source_model_weights:
        raise ValueError(
            "MLP compression refuses source-weight-contaminated executors"
        )
    parent_fingerprint = source.execution_fingerprint()
    if parent_fingerprint != selection.parent_executor_fingerprint:
        raise ValueError(
            "MLP selection does not match the parent executor fingerprint"
        )
    source_feed_forward = source.config.transformer.feed_forward
    if source_feed_forward.intermediate_width != selection.source_width:
        raise ValueError(
            "MLP selection width does not match the parent executor"
        )
    residual_width = source.width
    projection_bias = source_feed_forward.projection_bias
    if (
        source.feed_forward.gate_proj.weight.shape
        != (selection.source_width, residual_width)
        or source.feed_forward.up_proj.weight.shape
        != (selection.source_width, residual_width)
        or source.feed_forward.down_proj.weight.shape
        != (residual_width, selection.source_width)
        or (source.feed_forward.gate_proj.bias is not None)
        is not projection_bias
        or (source.feed_forward.up_proj.bias is not None)
        is not projection_bias
        or (source.feed_forward.down_proj.bias is not None)
        is not projection_bias
    ):
        raise ValueError("parent executor MLP parameter schema drifted")
    operator_sites = source.config.transformer.operator_sites
    if (
        operator_sites is not None
        and operator_sites.feed_forward_down_input
        != selection.activation_site
    ):
        raise ValueError(
            "MLP selection activation site does not match executor schema"
        )
    compressed_feed_forward = replace(
        source_feed_forward,
        intermediate_width=selection.retained_width,
    )
    compressed_transformer = replace(
        source.config.transformer,
        feed_forward=compressed_feed_forward,
    )
    compressed_config = replace(
        source.config,
        transformer=compressed_transformer,
    )
    cuda_devices = []
    if source.device.type == "cuda":
        cuda_devices.append(
            source.device.index
            if source.device.index is not None
            else torch.cuda.current_device()
        )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(0)
        destination = StructuredTransformerLayerExecutor(
            compressed_config,
            dtype=source.dtype,
            device=source.device,
        )
    source_modules = dict(source.named_modules())
    destination_modules = dict(destination.named_modules())
    if set(source_modules) != set(destination_modules):
        raise ValueError(
            "structured executor module schema drifted during compression"
        )
    for name, module in destination_modules.items():
        module.training = source_modules[name].training
    preserved_tensors = _copy_compressed_state(
        source,
        destination,
        selection,
    )
    if destination.owns_source_model_weights:
        raise RuntimeError(
            "MLP compression changed the source-weight origin"
        )

    removed_width = selection.removed_width
    per_unit_bias_parameters = int(
        source.feed_forward.gate_proj.bias is not None
    ) + int(source.feed_forward.up_proj.bias is not None)
    expected_parameter_reduction = removed_width * (
        3 * residual_width + per_unit_bias_parameters
    )
    actual_parameter_reduction = (
        source.learned_parameter_count
        - destination.learned_parameter_count
    )
    if actual_parameter_reduction != expected_parameter_reduction:
        raise RuntimeError(
            "compressed MLP parameter accounting drifted"
        )
    source_macs = 3 * residual_width * selection.source_width
    compressed_macs = (
        3 * residual_width * selection.retained_width
    )
    source_bias_additions = (
        2 * selection.source_width + residual_width
        if source_feed_forward.projection_bias
        else 0
    )
    compressed_bias_additions = (
        2 * selection.retained_width + residual_width
        if source_feed_forward.projection_bias
        else 0
    )
    report: dict[str, object] = {
        "schema": STRUCTURED_MLP_COMPRESSION_SCHEMA,
        "format_version": STRUCTURED_MLP_COMPRESSION_FORMAT_VERSION,
        "algorithm": STRUCTURED_MLP_FISHER_TAYLOR_ALGORITHM,
        "rung": {
            "source_intermediate_width": selection.source_width,
            "retained_intermediate_width": selection.retained_width,
            "removed_intermediate_width": removed_width,
        },
        "provenance": {
            **_provenance_dict(selection.provenance),
            "calibration_split_sha256": (
                selection.calibration_split_sha256
            ),
            "activation_site": selection.activation_site,
            "parent_executor_fingerprint": parent_fingerprint,
            "compressed_executor_fingerprint": (
                destination.execution_fingerprint()
            ),
        },
        "selection": selection.metadata(),
        "pairing": {
            "selected_indices_sha256": _indices_sha256(
                selection.selected_indices
            ),
            "construction_order": "ascending_source_unit_index",
            "gate_rows": selection.selected_indices,
            "up_rows": selection.selected_indices,
            "down_columns": selection.selected_indices,
        },
        "parameters": {
            "source_full_layer": source.learned_parameter_count,
            "compressed_full_layer": (
                destination.learned_parameter_count
            ),
            "removed_full_layer": actual_parameter_reduction,
            "expected_removed_from_mlp_slices": (
                expected_parameter_reduction
            ),
            "retained_ratio": (
                destination.learned_parameter_count
                / source.learned_parameter_count
            ),
        },
        "compute_per_valid_token": {
            "scope": (
                "gate_up_down_linear_weight_matmuls_only; "
                "nonlinear_activation_flops_excluded_as "
                "implementation_dependent; attention_and_norm_unchanged"
            ),
            "macs": {
                "source": source_macs,
                "compressed": compressed_macs,
                "removed": source_macs - compressed_macs,
            },
            "flops_two_per_mac": {
                "source": 2 * source_macs,
                "compressed": 2 * compressed_macs,
                "removed": 2 * (source_macs - compressed_macs),
            },
            "bias_additions": {
                "source": source_bias_additions,
                "compressed": compressed_bias_additions,
                "removed": (
                    source_bias_additions - compressed_bias_additions
                ),
            },
            "gate_up_multiplications": {
                "source": selection.source_width,
                "compressed": selection.retained_width,
                "removed": removed_width,
            },
            "nonlinear_activation_elements": {
                "source": selection.source_width,
                "compressed": selection.retained_width,
                "removed": removed_width,
            },
        },
        "digests": {
            "selection_sha256": selection.selection_sha256,
            "source_config_sha256": _config_sha256(source),
            "compressed_config_sha256": _config_sha256(destination),
            "source_state_sha256": _state_sha256(source),
            "compressed_state_sha256": _state_sha256(destination),
        },
        "preservation": {
            "preserved_tensor_count": len(preserved_tensors),
            "preserved_tensor_names": preserved_tensors,
            "attention_preserved": True,
            "normalizations_preserved": True,
            "projection_bias_schema_preserved": True,
            "native_source_parameter_read": False,
            "parent_compiled_executor_parameter_read": True,
            "source_weight_origin_before": False,
            "source_weight_origin_after": False,
        },
    }
    return destination, report


def prepare_width_compressed_mlp_refit_targets(
    batches: Sequence[StructuredLayerTargets],
    selection: StructuredMLPUnitSelection,
) -> tuple[tuple[StructuredLayerTargets, ...], dict[str, object]]:
    """Select pre-down feature columns for activation-only ridge refitting."""

    if not isinstance(selection, StructuredMLPUnitSelection):
        raise TypeError("selection must be StructuredMLPUnitSelection")
    if not batches:
        raise ValueError("refit target batches cannot be empty")
    selected_batches = []
    valid_rows = 0
    feature_records = []
    for batch_index, targets in enumerate(batches):
        if (
            not isinstance(targets, StructuredLayerTargets)
            or targets.provenance != selection.provenance
            or targets.attention_projection_input is None
            or targets.feed_forward_projection_input is None
            or targets.feed_forward_projection_input.shape[-1]
            != selection.source_width
        ):
            raise ValueError(
                "refit targets do not match selection provenance or width"
            )
        indices = torch.tensor(
            selection.selected_indices,
            dtype=torch.long,
            device=targets.feed_forward_projection_input.device,
        )
        compressed_features = (
            targets.feed_forward_projection_input.index_select(
                -1,
                indices,
            )
        )
        selected_batches.append(
            replace(
                targets,
                feed_forward_projection_input=compressed_features,
            )
        )
        valid = targets.sequence.query_valid_mask
        rows = int(valid.sum().item())
        valid_rows += rows
        feature_records.append(
            {
                "batch_index": batch_index,
                "valid_rows": rows,
                "feature_sha256": _tensor_mapping_sha256(
                    {
                        "feed_forward_projection_input": (
                            compressed_features[valid]
                        )
                    },
                    domain=_REFIT_INPUT_DOMAIN,
                ),
            }
        )
    report = {
        "schema": (
            "fisher_graph.structured_mlp_width_compression_refit_inputs"
        ),
        "format_version": 1,
        "selection_sha256": selection.selection_sha256,
        "provenance": _provenance_dict(selection.provenance),
        "batches": len(selected_batches),
        "valid_rows": valid_rows,
        "source_intermediate_width": selection.source_width,
        "retained_intermediate_width": selection.retained_width,
        "selected_indices_sha256": _indices_sha256(
            selection.selected_indices
        ),
        "ordered_valid_feature_sha256": _json_sha256(
            feature_records,
            domain=_REFIT_INPUT_DOMAIN,
        ),
        "padding_rows_excluded_from_digest": True,
        "activation_only": True,
    }
    return tuple(selected_batches), report


__all__ = [
    "GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH",
    "GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH",
    "STRUCTURED_MLP_COMPRESSION_FORMAT_VERSION",
    "STRUCTURED_MLP_COMPRESSION_SCHEMA",
    "STRUCTURED_MLP_FISHER_TAYLOR_ALGORITHM",
    "StructuredMLPFisherTaylorBatch",
    "StructuredMLPUnitSelection",
    "build_width_compressed_structured_executor",
    "prepare_width_compressed_mlp_refit_targets",
    "select_fisher_taylor_mlp_units",
    "select_gemma_mlp_first_rung_units",
]
