"""Two-forward exact-X4 H4 observations for frozen-basis reconstruction.

The complete-H4 projection experiments originally obtained one fit sequence
from a three-forward shadow followed by a two-forward native/exact-X4 pair.
That shadow is unnecessary when the purpose is only to reproduce the already
authenticated fit tensors.  This module collects the same native H4,
clamped-Y3/exact-X4 H4, and prompt-mean-NLL gradient in exactly two model
forwards and one backward while binding every retained tensor to the one-pass
bridge, model inputs, execution grid, and objective.

The observation is fit-only.  It neither installs a serving correction nor
authorizes the candidate logits it computes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassBridge,
    Gemma3L3L4OnePassPrefix,
    _canonical_json_bytes,
    _complete_h4_nll_objective_receipt_sha256,
    _execution_grid_sha256,
    _require_sha256,
    _runtime_tensor_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
    validate_gemma3_l3_l4_shadow_model_inputs_sha256,
)


__all__ = [
    "Gemma3L3L4ExactX4FitObservation",
    "collect_gemma3_l3_l4_exact_x4_fit_observation",
]


_X3_SITE = "layer.3.mlp.normalized_input"
_Y3_SITE = "layer.3.mlp.operator_output"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_OBSERVATION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-exact-x4-fit-observation:v1\0"
)


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _validate_supervision(
    *,
    supervised_indices: Tensor,
    supervised_targets: Tensor,
    valid_target_mask: Tensor,
    vocabulary: int,
    ignore_index: int,
) -> None:
    if (
        not isinstance(supervised_indices, Tensor)
        or supervised_indices.dtype != torch.int64
        or supervised_indices.device.type != "cpu"
        or supervised_indices.ndim != 2
        or supervised_indices.shape[1:] != (2,)
        or supervised_indices.shape[0] <= 0
        or not supervised_indices.is_contiguous()
    ):
        raise ValueError(
            "supervised_indices must be nonempty contiguous CPU int64 [N, 2]"
        )
    count = int(supervised_indices.shape[0])
    if (
        not isinstance(supervised_targets, Tensor)
        or supervised_targets.dtype != torch.int64
        or supervised_targets.device.type != "cpu"
        or supervised_targets.shape != (count,)
        or not supervised_targets.is_contiguous()
    ):
        raise ValueError(
            "supervised_targets must be contiguous CPU int64 [N]"
        )
    if type(ignore_index) is not int:
        raise TypeError("exact-X4 NLL ignore_index must be an integer")
    batches = supervised_indices[:, 0]
    positions = supervised_indices[:, 1]
    batch_size, sequence_length = valid_target_mask.shape
    if bool(
        (batches < 0).any()
        or (batches >= batch_size).any()
        or (positions < 0).any()
        or (positions >= sequence_length).any()
    ):
        raise ValueError("supervised indices escape the execution grid")
    ordinals = batches * sequence_length + positions
    if count > 1 and not bool(torch.all(ordinals[1:] > ordinals[:-1])):
        raise ValueError("supervised indices must be unique batch-major sorted")
    valid_cpu = valid_target_mask.detach().to(device="cpu")
    if not bool(valid_cpu[batches, positions].all()):
        raise ValueError("supervised indices include padding rows")
    if bool(
        (supervised_targets == ignore_index).any()
        or (supervised_targets < 0).any()
        or (supervised_targets >= vocabulary).any()
    ):
        raise ValueError(
            "supervised targets must be non-ignored vocabulary ids"
        )


def _validate_supervision_tensor_abi(
    supervised_indices: Tensor,
    supervised_targets: Tensor,
) -> None:
    if (
        not isinstance(supervised_indices, Tensor)
        or supervised_indices.dtype != torch.int64
        or supervised_indices.device.type != "cpu"
        or supervised_indices.ndim != 2
        or supervised_indices.shape[1:] != (2,)
        or supervised_indices.shape[0] <= 0
        or not supervised_indices.is_contiguous()
    ):
        raise ValueError(
            "supervised_indices must be nonempty contiguous CPU int64 [N, 2]"
        )
    if (
        not isinstance(supervised_targets, Tensor)
        or supervised_targets.dtype != torch.int64
        or supervised_targets.device.type != "cpu"
        or supervised_targets.shape != (int(supervised_indices.shape[0]),)
        or not supervised_targets.is_contiguous()
    ):
        raise ValueError(
            "supervised_targets must be contiguous CPU int64 [N]"
        )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ExactX4FitObservation:
    """Authenticated native/exact-X4 H4 tensors from two model forwards."""

    native_x4: Tensor
    native_h4: Tensor
    incomplete_h4: Tensor
    h4_gradient: Tensor
    prefix: Gemma3L3L4OnePassPrefix
    native_logits_sha256: str
    partial_exact_x4_logits_sha256: str
    supervised_indices_sha256: str
    supervised_targets_sha256: str
    supervised_token_count: int
    objective_ignore_index: int
    objective_mean_nll: float
    objective_receipt_sha256: str
    model_inputs_sha256: str
    execution_grid_sha256: str
    adapter_execution_sha256: str
    bridge_binding_sha256: str
    callback_order: tuple[str, ...]
    native_x4_sha256: str = ""
    native_h4_sha256: str = ""
    incomplete_h4_sha256: str = ""
    h4_gradient_sha256: str = ""
    artifact_sha256: str = ""

    _TENSOR_HASH_FIELDS = (
        ("native_x4", "native_x4_sha256", "native X4"),
        ("native_h4", "native_h4_sha256", "native H4"),
        ("incomplete_h4", "incomplete_h4_sha256", "incomplete H4"),
        ("h4_gradient", "h4_gradient_sha256", "H4 gradient"),
    )

    def __post_init__(self) -> None:
        self._validate_structure()
        for tensor_name, hash_name, label in self._TENSOR_HASH_FIELDS:
            computed = _runtime_tensor_sha256(getattr(self, tensor_name))
            supplied = getattr(self, hash_name)
            if supplied:
                if _require_sha256(supplied, label=label) != computed:
                    raise ValueError(f"{label} hash mismatch")
            else:
                object.__setattr__(self, hash_name, computed)
        computed = self._computed_artifact_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="exact-X4 fit observation artifact",
            ) != computed:
                raise ValueError("exact-X4 fit observation artifact mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _validate_structure(self) -> None:
        self.prefix.validate_integrity()
        if (
            not isinstance(self.native_h4, Tensor)
            or self.native_h4.ndim != 3
            or self.native_h4.numel() == 0
            or not self.native_h4.is_floating_point()
        ):
            raise ValueError("exact-X4 fit residual geometry is invalid")
        shape = self.native_h4.shape
        if (
            not isinstance(self.native_x4, Tensor)
            or self.native_x4.shape != shape
            or not self.native_x4.is_floating_point()
        ):
            raise ValueError("exact-X4 fit residual geometry is invalid")
        for value, label in (
            (self.native_x4, "native X4"),
            (self.native_h4, "native H4"),
            (self.incomplete_h4, "incomplete H4"),
            (self.h4_gradient, "H4 gradient"),
        ):
            if (
                not isinstance(value, Tensor)
                or value.shape != shape
                or value.dtype != self.native_h4.dtype
                or value.device != self.native_h4.device
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{label} differs from the residual ABI")
        if self.prefix.clamped_y3.shape != shape:
            raise ValueError("exact-X4 fit prefix differs from residual geometry")
        for value, label in (
            (self.native_logits_sha256, "native logits"),
            (self.partial_exact_x4_logits_sha256, "partial exact-X4 logits"),
            (self.supervised_indices_sha256, "supervised indices"),
            (self.supervised_targets_sha256, "supervised targets"),
            (self.objective_receipt_sha256, "objective receipt"),
            (self.model_inputs_sha256, "model inputs"),
            (self.execution_grid_sha256, "execution grid"),
            (self.adapter_execution_sha256, "adapter execution"),
            (self.bridge_binding_sha256, "bridge binding"),
        ):
            _require_sha256(value, label=label)
        if (
            self.prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or type(self.objective_ignore_index) is not int
            or not math.isfinite(self.objective_mean_nll)
            or self.objective_mean_nll < 0.0
            or self.callback_order
            != (
                "native.x3",
                "native.y3",
                "native.x4",
                "native.h4",
                "exact_x4.y3",
                "exact_x4.x4",
                "exact_x4.h4",
            )
        ):
            raise ValueError("exact-X4 fit execution metadata differs")
        expected_grid = _execution_grid_sha256(
            self.prefix.logical_positions,
            self.prefix.valid_target_mask,
            self.prefix.source_eligible_mask,
            self.prefix.target_affected_mask,
        )
        if expected_grid != self.execution_grid_sha256:
            raise ValueError("exact-X4 fit execution-grid hash differs")
        expected_objective = _complete_h4_nll_objective_receipt_sha256(
            supervised_indices_sha256=self.supervised_indices_sha256,
            supervised_targets_sha256=self.supervised_targets_sha256,
            partial_exact_x4_logits_sha256=(
                self.partial_exact_x4_logits_sha256
            ),
            ignore_index=self.objective_ignore_index,
            reduction="mean",
            supervised_token_count=self.supervised_token_count,
            mean_nll=self.objective_mean_nll,
        )
        if expected_objective != self.objective_receipt_sha256:
            raise ValueError("exact-X4 fit objective receipt differs")

    def _computed_artifact_sha256(self) -> str:
        payload = {
            "schema": "fisher_graph.gemma3_l3_l4_exact_x4_fit_observation",
            "format_version": 1,
            "prefix_artifact_sha256": self.prefix.artifact_sha256,
            "native_logits_sha256": self.native_logits_sha256,
            "partial_exact_x4_logits_sha256": (
                self.partial_exact_x4_logits_sha256
            ),
            "supervised_indices_sha256": self.supervised_indices_sha256,
            "supervised_targets_sha256": self.supervised_targets_sha256,
            "supervised_token_count": self.supervised_token_count,
            "objective_ignore_index": self.objective_ignore_index,
            "objective_reduction": "mean",
            "objective_mean_nll": self.objective_mean_nll,
            "objective_receipt_sha256": self.objective_receipt_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "callback_order": self.callback_order,
            "tensor_sha256s": {
                tensor_name: getattr(self, hash_name)
                for tensor_name, hash_name, _label in self._TENSOR_HASH_FIELDS
            },
            "model_forward_count": 2,
            "backward_count": 1,
            "simultaneously_live_full_vocabulary_logit_tensor_peak": 1,
            "fit_only": True,
            "serving_authorized": False,
        }
        return hashlib.sha256(
            _OBSERVATION_DOMAIN + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self._validate_structure()
        for tensor_name, hash_name, label in self._TENSOR_HASH_FIELDS:
            if _runtime_tensor_sha256(getattr(self, tensor_name)) != (
                _require_sha256(getattr(self, hash_name), label=label)
            ):
                raise RuntimeError(f"{label} tensor payload drifted")
        if self._computed_artifact_sha256() != _require_sha256(
            self.artifact_sha256,
            label="exact-X4 fit observation artifact",
        ):
            raise RuntimeError("exact-X4 fit observation payload drifted")

    @property
    def complete_h4_support_mask(self) -> Tensor:
        self.validate_integrity()
        return self.prefix.complete_h4_causal_support_mask()

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        support = self.prefix.complete_h4_causal_support_mask()
        graph_core = self.prefix.target_affected_mask
        return {
            "schema": "fisher_graph.gemma3_l3_l4_exact_x4_fit_observation",
            "format_version": 1,
            "fit_only": True,
            "serving_authorized": False,
            "model_forward_count": 2,
            "backward_count": 1,
            "simultaneously_live_full_vocabulary_logit_tensor_peak": 1,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "prefix_artifact_sha256": self.prefix.artifact_sha256,
            "native_x4_sha256": self.native_x4_sha256,
            "native_h4_sha256": self.native_h4_sha256,
            "incomplete_h4_sha256": self.incomplete_h4_sha256,
            "h4_gradient_sha256": self.h4_gradient_sha256,
            "native_logits_sha256": self.native_logits_sha256,
            "partial_exact_x4_logits_sha256": (
                self.partial_exact_x4_logits_sha256
            ),
            "supervised_indices_sha256": self.supervised_indices_sha256,
            "supervised_targets_sha256": self.supervised_targets_sha256,
            "supervised_token_count": self.supervised_token_count,
            "objective_ignore_index": self.objective_ignore_index,
            "objective_reduction": "mean",
            "objective_mean_nll": self.objective_mean_nll,
            "objective_receipt_sha256": self.objective_receipt_sha256,
            "complete_h4_support_rows": int(support.sum()),
            "graph_target_affected_rows": int(graph_core.sum()),
            "complete_h4_support_outside_graph_rows": int(
                (support & ~graph_core).sum()
            ),
            "callback_order": self.callback_order,
            "artifact_sha256": self.artifact_sha256,
        }


def collect_gemma3_l3_l4_exact_x4_fit_observation(
    bridge: Gemma3L3L4OnePassBridge,
    adapter: Gemma3CausalLMAdapter,
    model_inputs: Mapping[str, Tensor],
    *,
    supervised_indices: Tensor,
    supervised_targets: Tensor,
    ignore_index: int = -100,
) -> Gemma3L3L4ExactX4FitObservation:
    """Collect the frozen complete-H4 fit tensors in two forwards and one VJP."""

    if not isinstance(bridge, Gemma3L3L4OnePassBridge):
        raise TypeError("bridge must be a Gemma3L3L4OnePassBridge")
    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    if not isinstance(model_inputs, Mapping):
        raise TypeError("model_inputs must be a mapping")
    bridge.validate_integrity()
    bridge._authenticate_adapter(adapter)
    _validate_supervision_tensor_abi(
        supervised_indices,
        supervised_targets,
    )
    if type(ignore_index) is not int:
        raise TypeError("exact-X4 NLL ignore_index must be an integer")
    model_inputs_sha256 = gemma3_l3_l4_shadow_model_inputs_sha256(
        model_inputs
    )
    indices_sha256 = _runtime_tensor_sha256(supervised_indices)
    targets_sha256 = _runtime_tensor_sha256(supervised_targets)
    indices_snapshot = supervised_indices.detach().clone().contiguous()
    targets_snapshot = supervised_targets.detach().clone().contiguous()
    sequence = adapter.prepare_sequence(model_inputs)
    callback_order: list[str] = []
    x3: Tensor | None = None
    native_y3: Tensor | None = None
    native_x4: Tensor | None = None
    native_h4: Tensor | None = None
    prefix: Gemma3L3L4OnePassPrefix | None = None

    def native_x3(original: Tensor) -> Tensor:
        nonlocal x3
        callback_order.append("native.x3")
        if x3 is not None:
            raise RuntimeError("native X3 callback repeated")
        x3 = original
        return original

    def native_y3_callback(original: Tensor) -> Tensor:
        nonlocal native_y3, prefix
        callback_order.append("native.y3")
        if x3 is None or native_y3 is not None or prefix is not None:
            raise RuntimeError("native Y3 callback order differs")
        native_y3 = original
        prefix = bridge._prepare_prefix(
            x3=x3,
            native_y3=original,
            logical_positions=sequence.logical_positions,
            valid_mask=sequence.query_valid_mask,
        )
        return original

    def native_x4_callback(original: Tensor) -> Tensor:
        nonlocal native_x4
        callback_order.append("native.x4")
        if prefix is None or native_x4 is not None:
            raise RuntimeError("native X4 callback order differs")
        native_x4 = original
        return original

    def native_h4_callback(original: Tensor) -> Tensor:
        nonlocal native_h4
        callback_order.append("native.h4")
        if native_x4 is None or native_h4 is not None:
            raise RuntimeError("native H4 callback order differs")
        native_h4 = original
        return original

    try:
        with torch.no_grad():
            native_run = adapter.forward(
                model_inputs,
                capture_sites=(),
                interventions={
                    _X3_SITE: native_x3,
                    _Y3_SITE: native_y3_callback,
                    _X4_SITE: native_x4_callback,
                    _H4_SITE: native_h4_callback,
                },
            )
        bridge._authenticate_adapter(adapter)
        bridge.validate_integrity()
        if (
            x3 is None
            or native_y3 is None
            or native_x4 is None
            or native_h4 is None
            or prefix is None
            or tuple(callback_order)
            != ("native.x3", "native.y3", "native.x4", "native.h4")
        ):
            raise RuntimeError("native boundary collection was incomplete")
        prefix.validate_integrity()
        if (
            not torch.equal(
                native_run.sequence.logical_positions,
                sequence.logical_positions,
            )
            or not torch.equal(
                native_run.sequence.query_valid_mask,
                sequence.query_valid_mask,
            )
        ):
            raise RuntimeError("native fit-observation grid differs")
        _validate_supervision(
            supervised_indices=indices_snapshot,
            supervised_targets=targets_snapshot,
            valid_target_mask=prefix.valid_target_mask,
            vocabulary=int(native_run.logits.shape[-1]),
            ignore_index=ignore_index,
        )
        native_logits_sha256 = _runtime_tensor_sha256(native_run.logits)
        native_y3_sha256 = _runtime_tensor_sha256(native_y3)
        native_x4_snapshot = native_x4.detach().clone().contiguous()
        native_h4_snapshot = native_h4.detach().clone().contiguous()
        # Only the hash is retained.  Releasing the native run before the
        # exact-X4 pass keeps the peak at one full-vocabulary logits tensor.
        del native_run

        incomplete_h4_leaf: Tensor | None = None

        def exact_y3(original: Tensor) -> Tensor:
            callback_order.append("exact_x4.y3")
            if _runtime_tensor_sha256(original) != native_y3_sha256:
                raise RuntimeError("exact-X4 pass reached a different native Y3")
            return prefix.clamped_y3

        def exact_x4(original: Tensor) -> Tensor:
            callback_order.append("exact_x4.x4")
            if not (
                original.shape == native_x4_snapshot.shape
                and original.dtype == native_x4_snapshot.dtype
                and original.device == native_x4_snapshot.device
                and bool(torch.isfinite(original).all())
            ):
                raise RuntimeError("exact-X4 reference carrier differs")
            return native_x4_snapshot

        def exact_h4(original: Tensor) -> Tensor:
            nonlocal incomplete_h4_leaf
            callback_order.append("exact_x4.h4")
            if incomplete_h4_leaf is not None:
                raise RuntimeError("exact-X4 H4 callback repeated")
            if (
                original.shape != native_h4_snapshot.shape
                or original.dtype != native_h4_snapshot.dtype
                or original.device != native_h4_snapshot.device
                or not bool(torch.isfinite(original).all())
            ):
                raise RuntimeError("exact-X4 incomplete H4 differs")
            incomplete_h4_leaf = original.detach().requires_grad_(True)
            return incomplete_h4_leaf

        bridge._authenticate_adapter(adapter)
        bridge.validate_integrity()
        with torch.enable_grad():
            partial_run = adapter.forward(
                model_inputs,
                capture_sites=(),
                interventions={
                    _Y3_SITE: exact_y3,
                    _X4_SITE: exact_x4,
                    _H4_SITE: exact_h4,
                },
            )
            if incomplete_h4_leaf is None:
                raise RuntimeError("exact-X4 pass omitted the H4 boundary")
            indices_on_logits = indices_snapshot.to(partial_run.logits.device)
            targets_on_logits = targets_snapshot.to(partial_run.logits.device)
            logits = partial_run.logits[
                indices_on_logits[:, 0],
                indices_on_logits[:, 1],
            ]
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            loss = F.cross_entropy(
                logits,
                targets_on_logits,
                ignore_index=ignore_index,
                reduction="mean",
            )
            if (
                loss.ndim != 0
                or not loss.is_floating_point()
                or not bool(torch.isfinite(loss))
            ):
                raise RuntimeError("exact-X4 mean NLL is invalid")
            (gradient,) = torch.autograd.grad(
                loss,
                (incomplete_h4_leaf,),
                retain_graph=False,
                create_graph=False,
            )
        bridge._authenticate_adapter(adapter)
        bridge.validate_integrity()
        if (
            tuple(callback_order)
            != (
                "native.x3",
                "native.y3",
                "native.x4",
                "native.h4",
                "exact_x4.y3",
                "exact_x4.x4",
                "exact_x4.h4",
            )
            or not torch.equal(
                partial_run.sequence.logical_positions,
                sequence.logical_positions,
            )
            or not torch.equal(
                partial_run.sequence.query_valid_mask,
                sequence.query_valid_mask,
            )
        ):
            raise RuntimeError("exact-X4 fit pass callback or grid differs")
        if (
            gradient.shape != incomplete_h4_leaf.shape
            or gradient.dtype != incomplete_h4_leaf.dtype
            or gradient.device != incomplete_h4_leaf.device
            or not bool(torch.isfinite(gradient).all())
        ):
            raise RuntimeError("exact-X4 H4 gradient differs")
        if not _bitwise_equal(native_x4, native_x4_snapshot) or not (
            _bitwise_equal(native_h4, native_h4_snapshot)
        ):
            raise RuntimeError("native boundaries mutated during exact-X4 pass")
        partial_logits_sha256 = _runtime_tensor_sha256(partial_run.logits)
        mean_nll = float(loss.detach().to(device="cpu", dtype=torch.float64))
        objective_receipt = _complete_h4_nll_objective_receipt_sha256(
            supervised_indices_sha256=indices_sha256,
            supervised_targets_sha256=targets_sha256,
            partial_exact_x4_logits_sha256=partial_logits_sha256,
            ignore_index=ignore_index,
            reduction="mean",
            supervised_token_count=int(indices_snapshot.shape[0]),
            mean_nll=mean_nll,
        )
        execution_grid_sha256 = _execution_grid_sha256(
            prefix.logical_positions,
            prefix.valid_target_mask,
            prefix.source_eligible_mask,
            prefix.target_affected_mask,
        )
        result = Gemma3L3L4ExactX4FitObservation(
            native_x4=native_x4_snapshot,
            native_h4=native_h4_snapshot,
            incomplete_h4=(
                incomplete_h4_leaf.detach().clone().contiguous()
            ),
            h4_gradient=gradient.detach().clone().contiguous(),
            prefix=Gemma3L3L4OnePassPrefix(
                source_modes=(
                    prefix.source_modes.detach().clone().contiguous()
                ),
                clamped_y3=(
                    prefix.clamped_y3.detach().clone().contiguous()
                ),
                predicted_target_modal_delta=(
                    prefix.predicted_target_modal_delta.detach()
                    .clone()
                    .contiguous()
                ),
                decoded_base_x4_delta=(
                    prefix.decoded_base_x4_delta.detach()
                    .clone()
                    .contiguous()
                ),
                logical_positions=(
                    prefix.logical_positions.detach().clone().contiguous()
                ),
                valid_target_mask=(
                    prefix.valid_target_mask.detach().clone().contiguous()
                ),
                source_eligible_mask=(
                    prefix.source_eligible_mask.detach().clone().contiguous()
                ),
                target_affected_mask=(
                    prefix.target_affected_mask.detach().clone().contiguous()
                ),
                bridge_binding_sha256=prefix.bridge_binding_sha256,
                artifact_sha256=prefix.artifact_sha256,
            ),
            native_logits_sha256=native_logits_sha256,
            partial_exact_x4_logits_sha256=partial_logits_sha256,
            supervised_indices_sha256=indices_sha256,
            supervised_targets_sha256=targets_sha256,
            supervised_token_count=int(indices_snapshot.shape[0]),
            objective_ignore_index=ignore_index,
            objective_mean_nll=mean_nll,
            objective_receipt_sha256=objective_receipt,
            model_inputs_sha256=model_inputs_sha256,
            execution_grid_sha256=execution_grid_sha256,
            adapter_execution_sha256=bridge._adapter_execution_sha256,
            bridge_binding_sha256=bridge.bridge_binding_sha256,
            callback_order=tuple(callback_order),
        )
        result.validate_integrity()
        return result
    finally:
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            model_inputs_sha256,
        )
        if _runtime_tensor_sha256(supervised_indices) != indices_sha256:
            raise RuntimeError("supervised indices mutated during observation")
        if _runtime_tensor_sha256(supervised_targets) != targets_sha256:
            raise RuntimeError("supervised targets mutated during observation")
        bridge._authenticate_adapter(adapter)
        bridge.validate_integrity()
