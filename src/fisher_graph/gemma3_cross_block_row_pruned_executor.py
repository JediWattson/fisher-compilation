"""Minimal directed cross-block merged-supermode executor for Gemma.

This is sharing/merging, not deletion.  The anchor generator remains full
width and its signed scalar is shared forward.  The consumer physically omits
its redundant gate row and up row, while retaining every other native
generator row.  Retained features and the shared carry are scattered into the
native full-width down-projection input, so the complete consumer down matrix
(including its decoder column and accumulation order) remains unchanged.
Deleting the consumer without replacement remains an ablation control only.

Equivalence is defined on valid query positions.  Invalid padded rows carry
zero because the physically removed consumer generator is unavailable there;
causal masking prevents those rows from influencing valid queries.

The local operator exposes both affected MLP calls for instrumentation.  The
model runner temporarily overlays those calls at their native ``layer.mlp``
locations, letting the original model execute every intervening transformer
layer.  The carry is runtime state, not an extra residual-stream coordinate.
Consequently this makes an honest local compute claim without pretending that
a source-free layer-6-through-layer-15 window already exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .gemma3_cross_block_replacement_oracle import (
    validate_gemma3_cross_block_replacement_oracle_artifact,
)
from .structured_mlp_cross_block_plan import (
    StructuredMLPCrossBlockPlan,
    UnresolvedCrossBlockCarryProposal,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_ACTIVATIONS = frozenset(
    ("gelu", "gelu_pytorch_tanh", "silu", "swish")
)
_ARTIFACT_KIND = (
    "fisher_graph.gemma3_directed_cross_block_merged_supermode_executor"
)
_ARTIFACT_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = (
    b"fisher_graph.gemma3_directed_cross_block_merged_supermode_executor.v1\0"
)
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "binding",
    "residual_width",
    "source_intermediate_width",
    "source_anchor_mlp_fingerprint",
    "source_consumer_mlp_fingerprint",
    "source_pair_parameter_count",
    "dtype",
    "model_state_dict",
    "execution_fingerprint",
    "contains_complete_source_model",
    "requires_compatible_base_model",
}
_BINDING_FIELDS = {
    "source_model_fingerprint",
    "source_execution_fingerprint",
    "source_plan_artifact_sha256",
    "source_replacement_oracle_artifact_sha256",
    "proposal_id",
    "anchor_layer_id",
    "anchor_source_index",
    "consumer_layer_id",
    "consumer_source_index",
    "carry_scale",
    "activation",
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported merged-supermode dtype: {dtype}")
    return name


def _storage_pointers(module: nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel() > 0
    }
    pointers.discard(0)
    return pointers


def _linear(
    in_features: int,
    out_features: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Linear:
    return nn.Linear(
        in_features,
        out_features,
        bias=False,
        dtype=dtype,
        device=device,
    )


def _apply_activation(name: str, values: Tensor) -> Tensor:
    if name == "gelu_pytorch_tanh":
        return F.gelu(values, approximate="tanh")
    if name == "gelu":
        return F.gelu(values)
    if name in ("silu", "swish"):
        return F.silu(values)
    raise ValueError(f"unsupported Gemma MLP activation: {name!r}")


def _validate_source_mlp(
    module: nn.Module,
    *,
    label: str,
) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if not all(isinstance(value, nn.Linear) for value in (gate, up, down)):
        raise TypeError(
            f"{label} must expose linear gate_proj, up_proj, and down_proj"
        )
    assert isinstance(gate, nn.Linear)
    assert isinstance(up, nn.Linear)
    assert isinstance(down, nn.Linear)
    if any(value.bias is not None for value in (gate, up, down)):
        raise ValueError("Gemma row pruning requires bias-free projections")
    if (
        gate.weight.shape != up.weight.shape
        or down.in_features != gate.out_features
        or down.out_features != gate.in_features
        or any(
            value.weight.dtype != gate.weight.dtype
            or value.weight.device != gate.weight.device
            for value in (up, down)
        )
    ):
        raise ValueError(f"{label} projection shapes or tensor types disagree")
    return gate, up, down


class _ArtifactMLP(nn.Module):
    """Shape-only frozen parent used to reconstruct a source-free artifact."""

    def __init__(
        self,
        width: int,
        intermediate: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        device = torch.device("cpu")
        self.gate_proj = _linear(
            width,
            intermediate,
            dtype=dtype,
            device=device,
        )
        self.up_proj = _linear(
            width,
            intermediate,
            dtype=dtype,
            device=device,
        )
        self.down_proj = _linear(
            intermediate,
            width,
            dtype=dtype,
            device=device,
        )
        self.requires_grad_(False)
        self.eval()


@dataclass(frozen=True, slots=True)
class Gemma3CrossBlockRowPrunedBinding:
    """Authenticated source and native-coordinate identity for one executor."""

    source_model_fingerprint: str
    source_execution_fingerprint: str
    source_plan_artifact_sha256: str
    source_replacement_oracle_artifact_sha256: str
    proposal_id: str
    anchor_layer_id: str
    anchor_source_index: int
    consumer_layer_id: str
    consumer_source_index: int
    carry_scale: float
    activation: str

    def __post_init__(self) -> None:
        for label, value in (
            ("source_model_fingerprint", self.source_model_fingerprint),
            (
                "source_execution_fingerprint",
                self.source_execution_fingerprint,
            ),
            (
                "source_plan_artifact_sha256",
                self.source_plan_artifact_sha256,
            ),
            (
                "source_replacement_oracle_artifact_sha256",
                self.source_replacement_oracle_artifact_sha256,
            ),
        ):
            _require_sha256(value, label=label)
        for label, value in (
            ("proposal_id", self.proposal_id),
            ("anchor_layer_id", self.anchor_layer_id),
            ("consumer_layer_id", self.consumer_layer_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if (
            type(self.anchor_source_index) is not int
            or self.anchor_source_index < 0
            or type(self.consumer_source_index) is not int
            or self.consumer_source_index < 0
        ):
            raise ValueError("native source indices must be nonnegative")
        if (
            not isinstance(self.carry_scale, float)
            or not math.isfinite(self.carry_scale)
            or self.carry_scale == 0.0
        ):
            raise ValueError("carry_scale must be finite and nonzero")
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError("unsupported Gemma MLP activation")

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gemma3CrossBlockAnchorExecution:
    """Full anchor output and the signed token-local scalar carry."""

    output: Tensor
    carried_scalar: Tensor
    native_anchor_feature: Tensor


@dataclass(frozen=True, slots=True)
class Gemma3CrossBlockRowPrunedExecution:
    """Inspectable result when independent anchor/consumer inputs are known."""

    anchor_output: Tensor
    consumer_output: Tensor
    carried_scalar: Tensor
    native_anchor_feature: Tensor
    retained_consumer_features: Tensor


@dataclass(frozen=True, slots=True)
class Gemma3CrossBlockModelExecution:
    """One native full-model prefill with the two MLP overlays installed."""

    model_output: object
    carried_scalar: Tensor
    native_anchor_feature: Tensor
    retained_consumer_features: Tensor
    consumer_mlp_output: Tensor
    anchor_overlay_calls: int
    consumer_overlay_calls: int
    intervening_native_layer_ids: tuple[str, ...]
    source_whole_model_parameters: int
    candidate_whole_model_learned_parameters: int


@dataclass(frozen=True, slots=True)
class Gemma3CrossBlockRowPrunedAccounting:
    """Exact local coefficient and linear/scalar-MAC ledger."""

    query_tokens: int
    valid_tokens: int
    residual_width: int
    source_intermediate_width: int
    candidate_intermediate_width: int
    source_pair_learned_parameters: int
    candidate_pair_learned_parameters: int
    removed_learned_parameters: int
    fixed_carry_scale_coefficients: int
    net_stored_coefficient_savings: int
    preserved_consumer_decoder_parameters: int
    source_pair_logical_linear_macs: int
    candidate_pair_logical_linear_macs: int
    removed_logical_linear_macs: int
    carry_scale_logical_macs: int
    net_logical_arithmetic_macs_saved: int
    source_pair_dense_linear_macs: int
    candidate_pair_dense_linear_macs: int
    removed_dense_linear_macs: int
    carry_scale_dense_macs: int
    net_dense_arithmetic_macs_saved: int
    source_whole_model_parameters: int | None
    candidate_whole_model_learned_parameters: int | None
    candidate_whole_model_stored_coefficients: int | None

    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "parameter_savings_semantics": (
                "directed_sharing_removes_one_consumer_gate_row_and_up_row"
            ),
            "preserved_decoder_semantics": (
                "complete_native_consumer_down_projection_is_preserved"
            ),
            "mac_scope": (
                "gate_up_down_linear_macs_plus_explicit_carry_scale_multiply"
            ),
            "activation_masking_additions_and_kernel_launches_excluded": True,
            "kernel_speedup_claimed": False,
            "latency_measured": False,
        }


class Gemma3CrossBlockRowPrunedExecutor(nn.Module):
    """Two affected Gemma MLPs with one physically pruned consumer row."""

    def __init__(
        self,
        anchor_mlp: nn.Module,
        consumer_mlp: nn.Module,
        *,
        binding: Gemma3CrossBlockRowPrunedBinding,
    ) -> None:
        super().__init__()
        if not isinstance(binding, Gemma3CrossBlockRowPrunedBinding):
            raise TypeError(
                "binding must be a Gemma3CrossBlockRowPrunedBinding"
            )
        if (
            anchor_mlp.training
            or consumer_mlp.training
            or any(
                parameter.requires_grad
                for module in (anchor_mlp, consumer_mlp)
                for parameter in module.parameters()
            )
        ):
            raise ValueError(
                "row-pruned compilation requires frozen eval source MLPs"
            )
        anchor_gate, anchor_up, anchor_down = _validate_source_mlp(
            anchor_mlp,
            label="anchor_mlp",
        )
        consumer_gate, consumer_up, consumer_down = _validate_source_mlp(
            consumer_mlp,
            label="consumer_mlp",
        )
        if (
            anchor_gate.in_features != consumer_gate.in_features
            or anchor_gate.out_features != consumer_gate.out_features
        ):
            raise ValueError("anchor and consumer MLP dimensions must match")
        width = anchor_gate.in_features
        intermediate = anchor_gate.out_features
        if (
            binding.anchor_source_index >= intermediate
            or binding.consumer_source_index >= intermediate
            or intermediate <= 1
        ):
            raise ValueError("binding source index is outside the MLP")
        dtype = anchor_gate.weight.dtype
        device = anchor_gate.weight.device
        if (
            consumer_gate.weight.dtype != dtype
            or consumer_gate.weight.device != device
        ):
            raise ValueError("anchor and consumer MLP tensor types must match")

        source_anchor_fingerprint = module_state_fingerprint(anchor_mlp)
        source_consumer_fingerprint = module_state_fingerprint(consumer_mlp)
        source_storage = _storage_pointers(anchor_mlp) | _storage_pointers(
            consumer_mlp
        )

        self.anchor_gate_proj = _linear(
            width,
            intermediate,
            dtype=dtype,
            device=device,
        )
        self.anchor_up_proj = _linear(
            width,
            intermediate,
            dtype=dtype,
            device=device,
        )
        self.anchor_down_proj = _linear(
            intermediate,
            width,
            dtype=dtype,
            device=device,
        )
        retained = torch.tensor(
            tuple(
                index
                for index in range(intermediate)
                if index != binding.consumer_source_index
            ),
            dtype=torch.long,
            device=device,
        )
        self.consumer_gate_proj = _linear(
            width,
            intermediate - 1,
            dtype=dtype,
            device=device,
        )
        self.consumer_up_proj = _linear(
            width,
            intermediate - 1,
            dtype=dtype,
            device=device,
        )
        self.consumer_down_proj = _linear(
            intermediate,
            width,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "retained_consumer_source_indices",
            retained,
            # This is structural metadata derivable from the binding, not a
            # stored candidate coefficient.
            persistent=False,
        )
        self.register_buffer(
            "carry_scale",
            torch.tensor(
                [binding.carry_scale],
                dtype=dtype,
                device=device,
            ),
            persistent=True,
        )

        with torch.no_grad():
            self.anchor_gate_proj.weight.copy_(anchor_gate.weight)
            self.anchor_up_proj.weight.copy_(anchor_up.weight)
            self.anchor_down_proj.weight.copy_(anchor_down.weight)
            self.consumer_gate_proj.weight.copy_(
                consumer_gate.weight.index_select(0, retained)
            )
            self.consumer_up_proj.weight.copy_(
                consumer_up.weight.index_select(0, retained)
            )
            self.consumer_down_proj.weight.copy_(
                consumer_down.weight
            )

        self.binding = binding
        self.residual_width = width
        self.source_intermediate_width = intermediate
        self._source_anchor_mlp_fingerprint = source_anchor_fingerprint
        self._source_consumer_mlp_fingerprint = source_consumer_fingerprint
        self._source_pair_parameter_count = sum(
            parameter.numel()
            for module in (anchor_mlp, consumer_mlp)
            for parameter in module.parameters()
        )
        if source_storage & _storage_pointers(self):
            raise RuntimeError("row-pruned executor aliases source storage")
        if (
            module_state_fingerprint(anchor_mlp)
            != source_anchor_fingerprint
            or module_state_fingerprint(consumer_mlp)
            != source_consumer_fingerprint
        ):
            raise RuntimeError("row-pruned executor mutated a source MLP")
        expected_source = 6 * width * intermediate
        expected_candidate = expected_source - 2 * width
        if self._source_pair_parameter_count != expected_source:
            raise ValueError(
                "source MLP pair parameter count is incompatible with "
                "bias-free Gemma gating"
            )
        if self.learned_parameter_count != expected_candidate:
            raise RuntimeError("row-pruned learned-parameter count drifted")
        self.requires_grad_(False)
        self.eval()

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor_gate_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.anchor_gate_proj.weight.device

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return 1

    @property
    def source_pair_parameter_count(self) -> int:
        return self._source_pair_parameter_count

    @property
    def consumer_decoder_column(self) -> Tensor:
        """View the preserved native consumer decoder column."""

        return self.consumer_down_proj.weight[
            :, self.binding.consumer_source_index
        ]

    def _validate_values(
        self,
        values: Tensor,
        valid_positions: Tensor,
        *,
        label: str,
    ) -> None:
        if (
            not isinstance(values, Tensor)
            or values.ndim != 3
            or values.shape[-1] != self.residual_width
            or values.dtype != self.dtype
            or values.device != self.device
        ):
            raise ValueError(
                f"{label} must have shape [batch, sequence, residual_width] "
                "with the executor dtype and device"
            )
        if (
            not isinstance(valid_positions, Tensor)
            or valid_positions.dtype is not torch.bool
            or valid_positions.shape != values.shape[:2]
            or valid_positions.device != values.device
        ):
            raise ValueError(
                "valid_positions must be a colocated [batch, sequence] "
                "boolean mask"
            )

    def forward_anchor(
        self,
        normalized_hidden_states: Tensor,
        valid_positions: Tensor,
    ) -> Gemma3CrossBlockAnchorExecution:
        """Execute the full anchor MLP and emit a valid-query signed carry."""

        self._validate_values(
            normalized_hidden_states,
            valid_positions,
            label="anchor normalized_hidden_states",
        )
        features = _apply_activation(
            self.binding.activation,
            self.anchor_gate_proj(normalized_hidden_states),
        ) * self.anchor_up_proj(normalized_hidden_states)
        native_feature = features[..., self.binding.anchor_source_index]
        carried = (native_feature * self.carry_scale).masked_fill(
            ~valid_positions,
            0,
        )
        return Gemma3CrossBlockAnchorExecution(
            output=self.anchor_down_proj(features),
            carried_scalar=carried,
            native_anchor_feature=native_feature,
        )

    def forward_consumer(
        self,
        normalized_hidden_states: Tensor,
        carried_scalar: Tensor,
        valid_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Execute retained rows and inject carry on valid query positions."""

        self._validate_values(
            normalized_hidden_states,
            valid_positions,
            label="consumer normalized_hidden_states",
        )
        if (
            not isinstance(carried_scalar, Tensor)
            or carried_scalar.shape != valid_positions.shape
            or carried_scalar.dtype != self.dtype
            or carried_scalar.device != self.device
            or bool(carried_scalar[~valid_positions].count_nonzero())
        ):
            raise ValueError(
                "carried_scalar must be a colocated, invalid-row-zeroed "
                "[batch, sequence] tensor"
            )
        retained_features = _apply_activation(
            self.binding.activation,
            self.consumer_gate_proj(normalized_hidden_states),
        ) * self.consumer_up_proj(normalized_hidden_states)
        full_down_input = torch.zeros(
            *retained_features.shape[:-1],
            self.source_intermediate_width,
            dtype=retained_features.dtype,
            device=retained_features.device,
        )
        full_down_input.index_copy_(
            -1,
            self.retained_consumer_source_indices,
            retained_features,
        )
        full_down_input[..., self.binding.consumer_source_index] = (
            carried_scalar
        )
        return self.consumer_down_proj(full_down_input), retained_features

    def forward(
        self,
        anchor_normalized_hidden_states: Tensor,
        consumer_normalized_hidden_states: Tensor,
        valid_positions: Tensor,
    ) -> Gemma3CrossBlockRowPrunedExecution:
        """Execute both affected MLPs when both native inputs are available."""

        anchor = self.forward_anchor(
            anchor_normalized_hidden_states,
            valid_positions,
        )
        consumer_output, retained = self.forward_consumer(
            consumer_normalized_hidden_states,
            anchor.carried_scalar,
            valid_positions,
        )
        return Gemma3CrossBlockRowPrunedExecution(
            anchor_output=anchor.output,
            consumer_output=consumer_output,
            carried_scalar=anchor.carried_scalar,
            native_anchor_feature=anchor.native_anchor_feature,
            retained_consumer_features=retained,
        )

    def logical_accounting(
        self,
        valid_positions: Tensor,
        *,
        source_whole_model_parameters: int | None = None,
    ) -> Gemma3CrossBlockRowPrunedAccounting:
        if (
            not isinstance(valid_positions, Tensor)
            or valid_positions.dtype is not torch.bool
            or valid_positions.ndim != 2
        ):
            raise ValueError(
                "valid_positions must have shape [batch, sequence]"
            )
        if source_whole_model_parameters is not None and (
            type(source_whole_model_parameters) is not int
            or source_whole_model_parameters
            < self.source_pair_parameter_count
        ):
            raise ValueError(
                "source_whole_model_parameters is smaller than the MLP pair"
            )
        query_tokens = valid_positions.numel()
        valid_tokens = int(valid_positions.sum().item())
        width = self.residual_width
        intermediate = self.source_intermediate_width
        source_per_token = 6 * width * intermediate
        candidate_linear_per_token = source_per_token - 2 * width
        removed_parameters = (
            self.source_pair_parameter_count - self.learned_parameter_count
        )
        if removed_parameters != 2 * width:
            raise RuntimeError("row-pruned parameter accounting drifted")
        whole_candidate = (
            None
            if source_whole_model_parameters is None
            else source_whole_model_parameters - removed_parameters
        )
        return Gemma3CrossBlockRowPrunedAccounting(
            query_tokens=query_tokens,
            valid_tokens=valid_tokens,
            residual_width=width,
            source_intermediate_width=intermediate,
            candidate_intermediate_width=intermediate - 1,
            source_pair_learned_parameters=self.source_pair_parameter_count,
            candidate_pair_learned_parameters=self.learned_parameter_count,
            removed_learned_parameters=removed_parameters,
            fixed_carry_scale_coefficients=1,
            net_stored_coefficient_savings=removed_parameters - 1,
            preserved_consumer_decoder_parameters=width,
            source_pair_logical_linear_macs=valid_tokens * source_per_token,
            candidate_pair_logical_linear_macs=(
                valid_tokens * candidate_linear_per_token
            ),
            removed_logical_linear_macs=valid_tokens * 2 * width,
            carry_scale_logical_macs=valid_tokens,
            net_logical_arithmetic_macs_saved=(
                valid_tokens * (2 * width - 1)
            ),
            source_pair_dense_linear_macs=query_tokens * source_per_token,
            candidate_pair_dense_linear_macs=(
                query_tokens * candidate_linear_per_token
            ),
            removed_dense_linear_macs=query_tokens * 2 * width,
            carry_scale_dense_macs=query_tokens,
            net_dense_arithmetic_macs_saved=(
                query_tokens * (2 * width - 1)
            ),
            source_whole_model_parameters=source_whole_model_parameters,
            candidate_whole_model_learned_parameters=whole_candidate,
            candidate_whole_model_stored_coefficients=(
                None if whole_candidate is None else whole_candidate + 1
            ),
        )

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "gemma3_directed_cross_block_merged_supermode",
            "binding": self.binding.metadata(),
            "residual_width": self.residual_width,
            "source_intermediate_width": self.source_intermediate_width,
            "candidate_consumer_intermediate_width": (
                self.source_intermediate_width - 1
            ),
            "source_anchor_mlp_fingerprint": (
                self._source_anchor_mlp_fingerprint
            ),
            "source_consumer_mlp_fingerprint": (
                self._source_consumer_mlp_fingerprint
            ),
            "source_pair_parameter_count": self.source_pair_parameter_count,
            "candidate_pair_parameter_count": self.learned_parameter_count,
            "fixed_runtime_coefficient_count": (
                self.fixed_runtime_coefficient_count
            ),
            "runtime_carry_scale": float(self.carry_scale.item()),
            "physically_skipped_consumer_gate_rows": 1,
            "physically_skipped_consumer_up_rows": 1,
            "preserved_consumer_down_columns": 1,
            "full_consumer_down_projection_preserved": True,
            "consumer_down_input_scatter_width": (
                self.source_intermediate_width
            ),
            "equivalence_domain": "valid_query_positions",
            "invalid_query_policy": "zero_carried_coordinate",
            "anchor_generator_retained_and_shared": True,
            "consumer_generator_removed": True,
            "consumer_decoder_retained": True,
            "deletion_is_ablation_control_only": True,
            "compression_semantics": "directed_cross_block_merged_supermode",
            "intervening_layers_included": False,
            "native_in_model_overlay_available": True,
            "source_free_window_claimed": False,
            "kernel_speedup_claimed": False,
            "fit_only_experimental_executor": True,
            "materialized_whole_model_candidate": False,
            "authorizes_guard": False,
            "authorizes_model_replacement": False,
        }

    def execution_fingerprint(self) -> str:
        """Hash executable tensors together with their authenticated meaning."""

        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                {
                    "architecture": self.architecture_manifest(),
                    "module_training": tuple(
                        (name, module.training)
                        for name, module in self.named_modules()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(module_state_fingerprint(self).encode("ascii"))
        return digest.hexdigest()

    def artifact_state_dict(self) -> dict[str, object]:
        """Return a strict, CPU-resident local executor artifact.

        The artifact materializes the two affected candidate MLPs, but not the
        unchanged remainder of Gemma.  Installing it still requires a base
        model whose model and execution fingerprints match ``binding``.
        """

        if any(module.training for module in self.modules()):
            raise RuntimeError(
                "merged-supermode artifacts require every module in eval "
                "mode"
            )
        state = {
            name: value.detach().to(device="cpu").clone()
            for name, value in self.state_dict().items()
        }
        for name, value in state.items():
            if not value.is_floating_point() or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"merged-supermode state {name!r} is invalid"
                )
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _ARTIFACT_FORMAT_VERSION,
            "binding": self.binding.metadata(),
            "residual_width": self.residual_width,
            "source_intermediate_width": self.source_intermediate_width,
            "source_anchor_mlp_fingerprint": (
                self._source_anchor_mlp_fingerprint
            ),
            "source_consumer_mlp_fingerprint": (
                self._source_consumer_mlp_fingerprint
            ),
            "source_pair_parameter_count": self.source_pair_parameter_count,
            "dtype": _dtype_name(self.dtype),
            "model_state_dict": state,
            "execution_fingerprint": self.execution_fingerprint(),
            "contains_complete_source_model": False,
            "requires_compatible_base_model": True,
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> Gemma3CrossBlockRowPrunedExecutor:
        """Strictly reconstruct the local executor without source MLP objects."""

        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError("merged-supermode artifact fields are invalid")
        dtype_name = state["dtype"]
        execution_fingerprint = state["execution_fingerprint"]
        if (
            state["artifact_kind"] != _ARTIFACT_KIND
            or state["format_version"] != _ARTIFACT_FORMAT_VERSION
            or state["contains_complete_source_model"] is not False
            or state["requires_compatible_base_model"] is not True
            or not isinstance(dtype_name, str)
            or dtype_name not in _DTYPES
            or not isinstance(execution_fingerprint, str)
            or _SHA256.fullmatch(execution_fingerprint) is None
        ):
            raise ValueError("unsupported merged-supermode artifact")
        binding_raw = state["binding"]
        if (
            not isinstance(binding_raw, Mapping)
            or set(binding_raw) != _BINDING_FIELDS
        ):
            raise ValueError(
                "merged-supermode artifact binding fields are invalid"
            )
        binding = Gemma3CrossBlockRowPrunedBinding(**dict(binding_raw))
        width = state["residual_width"]
        intermediate = state["source_intermediate_width"]
        source_count = state["source_pair_parameter_count"]
        if (
            type(width) is not int
            or width <= 0
            or type(intermediate) is not int
            or intermediate <= 1
            or type(source_count) is not int
            or source_count != 6 * width * intermediate
            or binding.anchor_source_index >= intermediate
            or binding.consumer_source_index >= intermediate
        ):
            raise ValueError(
                "merged-supermode artifact dimensions are invalid"
            )
        anchor_fingerprint = _require_sha256(
            state["source_anchor_mlp_fingerprint"],
            label="source_anchor_mlp_fingerprint",
        )
        consumer_fingerprint = _require_sha256(
            state["source_consumer_mlp_fingerprint"],
            label="source_consumer_mlp_fingerprint",
        )
        dtype = _DTYPES[dtype_name]
        with torch.random.fork_rng(devices=()):
            torch.manual_seed(0)
            anchor = _ArtifactMLP(
                width,
                intermediate,
                dtype=dtype,
            )
            consumer = _ArtifactMLP(
                width,
                intermediate,
                dtype=dtype,
            )
            result = cls(anchor, consumer, binding=binding)
        result._source_anchor_mlp_fingerprint = anchor_fingerprint
        result._source_consumer_mlp_fingerprint = consumer_fingerprint
        result._source_pair_parameter_count = source_count

        raw_model_state = state["model_state_dict"]
        if not isinstance(raw_model_state, Mapping):
            raise ValueError(
                "merged-supermode artifact model state is invalid"
            )
        expected = result.state_dict()
        if set(raw_model_state) != set(expected):
            raise ValueError(
                "merged-supermode artifact model-state fields are invalid"
            )
        restored: dict[str, Tensor] = {}
        for name, expected_value in expected.items():
            value = raw_model_state[name]
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.shape != expected_value.shape
                or value.dtype != expected_value.dtype
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"merged-supermode state {name!r} has invalid schema"
                )
            restored[name] = value.clone()
        result.load_state_dict(restored, strict=True)
        result.to(device=map_location)
        result.requires_grad_(False)
        result.eval()
        if result.execution_fingerprint() != execution_fingerprint:
            raise ValueError(
                "merged-supermode execution fingerprint mismatch"
            )
        return result

    @classmethod
    def from_validated_oracle(
        cls,
        adapter: Gemma3CausalLMAdapter,
        plan: StructuredMLPCrossBlockPlan,
        oracle_artifact: Mapping[str, object],
    ) -> Gemma3CrossBlockRowPrunedExecutor:
        """Build the local executor from an authenticated fit-only oracle."""

        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(plan, StructuredMLPCrossBlockPlan):
            raise TypeError("plan must be a StructuredMLPCrossBlockPlan")
        validate_gemma3_cross_block_replacement_oracle_artifact(
            oracle_artifact
        )
        binding = oracle_artifact["binding"]
        proposal_metadata = oracle_artifact["proposal"]
        fit = oracle_artifact["fit"]
        comparisons = oracle_artifact["comparisons"]
        if not all(
            isinstance(value, Mapping)
            for value in (binding, proposal_metadata, fit, comparisons)
        ):
            raise ValueError("replacement oracle metadata is invalid")
        if comparisons.get("core_paired_oracle") is None:
            raise ValueError(
                "row-pruned construction requires a complete paired oracle"
            )
        if (
            binding["source_model_fingerprint"]
            != adapter.model_fingerprint()
            or binding["source_execution_fingerprint"]
            != adapter.execution_fingerprint()
            or binding["source_plan_artifact_sha256"]
            != plan.artifact_sha256
            or binding["source_discovery_artifact_sha256"]
            != plan.source_discovery_artifact_sha256
        ):
            raise ValueError(
                "replacement oracle does not bind this adapter and plan"
            )
        proposal_id = proposal_metadata.get("proposal_id")
        proposals = tuple(
            proposal
            for proposal in plan.proposals
            if proposal.proposal_id == proposal_id
        )
        if len(proposals) != 1:
            raise ValueError(
                "replacement oracle proposal is absent from the plan"
            )
        proposal = proposals[0]
        if (
            proposal_metadata.get("anchor") != proposal.anchor.metadata()
            or proposal_metadata.get("consumer")
            != proposal.consumer.metadata()
        ):
            raise ValueError(
                "replacement oracle native coordinates do not match plan"
            )
        anchor_layer = adapter.layer(proposal.anchor.layer_id)
        consumer_layer = adapter.layer(proposal.consumer.layer_id)
        if (
            anchor_layer.transformer is None
            or consumer_layer.transformer is None
            or anchor_layer.transformer.feed_forward.activation
            != consumer_layer.transformer.feed_forward.activation
        ):
            raise ValueError("Gemma proposal MLP activation metadata differs")
        anchor_source_layer = adapter.source_module(
            proposal.anchor.layer_id
        )
        consumer_source_layer = adapter.source_module(
            proposal.consumer.layer_id
        )
        anchor_mlp = getattr(anchor_source_layer, "mlp", None)
        consumer_mlp = getattr(consumer_source_layer, "mlp", None)
        if not isinstance(anchor_mlp, nn.Module) or not isinstance(
            consumer_mlp,
            nn.Module,
        ):
            raise TypeError("Gemma source layers do not expose MLP modules")
        artifact_sha256 = _require_sha256(
            oracle_artifact["artifact_sha256"],
            label="replacement_oracle_artifact_sha256",
        )
        selected_scale = fit.get("selected_scale")
        if (
            isinstance(selected_scale, bool)
            or not isinstance(selected_scale, (float, int))
        ):
            raise ValueError("replacement oracle selected scale is invalid")
        return cls(
            anchor_mlp,
            consumer_mlp,
            binding=Gemma3CrossBlockRowPrunedBinding(
                source_model_fingerprint=adapter.model_fingerprint(),
                source_execution_fingerprint=(
                    adapter.execution_fingerprint()
                ),
                source_plan_artifact_sha256=plan.artifact_sha256,
                source_replacement_oracle_artifact_sha256=artifact_sha256,
                proposal_id=proposal.proposal_id,
                anchor_layer_id=proposal.anchor.layer_id,
                anchor_source_index=proposal.anchor_source_index,
                consumer_layer_id=proposal.consumer.layer_id,
                consumer_source_index=proposal.consumer_source_index,
                carry_scale=float(selected_scale),
                activation=(
                    anchor_layer.transformer.feed_forward.activation
                ),
            ),
        )


@dataclass(slots=True)
class _InModelCarryState:
    valid_positions: Tensor
    anchor_execution: Gemma3CrossBlockAnchorExecution | None = None
    retained_consumer_features: Tensor | None = None
    consumer_mlp_output: Tensor | None = None
    anchor_calls: int = 0
    consumer_calls: int = 0


class _AnchorMLPOverlay(nn.Module):
    def __init__(
        self,
        executor: Gemma3CrossBlockRowPrunedExecutor,
        state: _InModelCarryState,
    ) -> None:
        super().__init__()
        self.executor = executor
        self._state = state
        self.eval()

    @property
    def gate_proj(self) -> nn.Linear:
        return self.executor.anchor_gate_proj

    @property
    def up_proj(self) -> nn.Linear:
        return self.executor.anchor_up_proj

    @property
    def down_proj(self) -> nn.Linear:
        return self.executor.anchor_down_proj

    def forward(self, normalized_hidden_states: Tensor) -> Tensor:
        if self._state.anchor_calls != 0:
            raise RuntimeError(
                "the merged-supermode anchor may execute only once per "
                "prefill"
            )
        execution = self.executor.forward_anchor(
            normalized_hidden_states,
            self._state.valid_positions,
        )
        self._state.anchor_execution = execution
        self._state.anchor_calls += 1
        return execution.output


class _ConsumerMLPOverlay(nn.Module):
    def __init__(
        self,
        executor: Gemma3CrossBlockRowPrunedExecutor,
        state: _InModelCarryState,
    ) -> None:
        super().__init__()
        # Do not register the executor a second time in the model tree.  It is
        # owned by the anchor overlay; this non-module lookup only shares the
        # runtime operator with the later consumer call.
        object.__setattr__(self, "_executor", executor)
        self._state = state
        self.eval()

    @property
    def executor(self) -> Gemma3CrossBlockRowPrunedExecutor:
        executor = object.__getattribute__(self, "_executor")
        assert isinstance(executor, Gemma3CrossBlockRowPrunedExecutor)
        return executor

    @property
    def gate_proj(self) -> nn.Linear:
        return self.executor.consumer_gate_proj

    @property
    def up_proj(self) -> nn.Linear:
        return self.executor.consumer_up_proj

    @property
    def down_proj(self) -> nn.Linear:
        return self.executor.consumer_down_proj

    def forward(self, normalized_hidden_states: Tensor) -> Tensor:
        anchor = self._state.anchor_execution
        if (
            anchor is None
            or self._state.anchor_calls != 1
            or self._state.consumer_calls != 0
        ):
            raise RuntimeError(
                "the merged-supermode consumer requires exactly one earlier "
                "anchor call"
            )
        output, retained = self.executor.forward_consumer(
            normalized_hidden_states,
            anchor.carried_scalar,
            self._state.valid_positions,
        )
        self._state.retained_consumer_features = retained
        self._state.consumer_mlp_output = output
        self._state.consumer_calls += 1
        return output


class Gemma3CrossBlockModelExecutor:
    """Run a full native Gemma prefill with two temporary MLP overlays.

    The original anchor and consumer MLP objects are restored even when model
    execution fails.  All layers strictly between them remain attached and
    execute through the model's native forward path.
    """

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        executor: Gemma3CrossBlockRowPrunedExecutor,
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(executor, Gemma3CrossBlockRowPrunedExecutor):
            raise TypeError(
                "executor must be a Gemma3CrossBlockRowPrunedExecutor"
            )
        self._adapter = adapter
        self._executor = executor
        self._active = False
        self._validate_live_source()

    @property
    def adapter(self) -> Gemma3CausalLMAdapter:
        return self._adapter

    @property
    def executor(self) -> Gemma3CrossBlockRowPrunedExecutor:
        return self._executor

    def _resolve_live_source(
        self,
    ) -> tuple[int, nn.Module, int, nn.Module]:
        anchor_spec = self._adapter.layer(
            self._executor.binding.anchor_layer_id
        )
        consumer_spec = self._adapter.layer(
            self._executor.binding.consumer_layer_id
        )
        if anchor_spec.ordinal >= consumer_spec.ordinal:
            raise ValueError(
                "cross-block anchor must precede its consumer"
            )
        anchor_layer = self._adapter.source_module(anchor_spec.id)
        consumer_layer = self._adapter.source_module(consumer_spec.id)
        anchor_mlp = getattr(anchor_layer, "mlp", None)
        consumer_mlp = getattr(consumer_layer, "mlp", None)
        if not isinstance(anchor_mlp, nn.Module) or not isinstance(
            consumer_mlp,
            nn.Module,
        ):
            raise TypeError("Gemma source layers do not expose MLP modules")
        return (
            anchor_spec.ordinal,
            anchor_mlp,
            consumer_spec.ordinal,
            consumer_mlp,
        )

    def _validate_live_source(self) -> None:
        model = self._adapter.module
        if (
            model.training
            or any(
                parameter.requires_grad
                for parameter in model.parameters()
            )
        ):
            raise ValueError(
                "in-model merged-supermode execution requires a frozen "
                "eval source model"
            )
        if (
            self._adapter.model_fingerprint()
            != self._executor.binding.source_model_fingerprint
        ):
            raise ValueError(
                "executor binding does not authenticate the live model"
            )
        if (
            self._adapter.execution_fingerprint()
            != self._executor.binding.source_execution_fingerprint
        ):
            raise ValueError(
                "executor binding does not authenticate the live execution "
                "configuration"
            )
        _, anchor_mlp, _, consumer_mlp = self._resolve_live_source()
        if (
            module_state_fingerprint(anchor_mlp)
            != self._executor._source_anchor_mlp_fingerprint
            or module_state_fingerprint(consumer_mlp)
            != self._executor._source_consumer_mlp_fingerprint
        ):
            raise ValueError(
                "live Gemma MLP state differs from the compiled source"
            )

    def __call__(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3CrossBlockModelExecution:
        if self._active:
            raise RuntimeError(
                "in-model merged-supermode execution is not reentrant"
            )
        self._validate_live_source()
        context = self._adapter.prepare_sequence(model_inputs)
        valid_positions = context.query_valid_mask.to(
            device=self._executor.device
        )
        (
            anchor_ordinal,
            anchor_mlp,
            consumer_ordinal,
            consumer_mlp,
        ) = self._resolve_live_source()
        layers = getattr(getattr(self._adapter.module, "model"), "layers")
        source_parameters = sum(
            parameter.numel()
            for parameter in self._adapter.module.parameters()
        )
        state = _InModelCarryState(valid_positions=valid_positions)
        anchor_overlay = _AnchorMLPOverlay(self._executor, state)
        consumer_overlay = _ConsumerMLPOverlay(self._executor, state)
        self._active = True
        try:
            layers[anchor_ordinal].mlp = anchor_overlay
            try:
                layers[consumer_ordinal].mlp = consumer_overlay
                candidate_parameters = sum(
                    parameter.numel()
                    for parameter in self._adapter.module.parameters()
                )
                expected_parameters = (
                    source_parameters
                    - self._executor.source_pair_parameter_count
                    + self._executor.learned_parameter_count
                )
                if candidate_parameters != expected_parameters:
                    raise RuntimeError(
                        "in-model learned-parameter accounting drifted"
                    )
                call_inputs: dict[str, object] = dict(model_inputs)
                call_inputs["use_cache"] = False
                call_inputs["return_dict"] = True
                model_output = self._adapter.module(**call_inputs)
            finally:
                layers[consumer_ordinal].mlp = consumer_mlp
        finally:
            layers[anchor_ordinal].mlp = anchor_mlp
            self._active = False
        anchor = state.anchor_execution
        retained = state.retained_consumer_features
        consumer_output = state.consumer_mlp_output
        if (
            anchor is None
            or retained is None
            or consumer_output is None
            or state.anchor_calls != 1
            or state.consumer_calls != 1
        ):
            raise RuntimeError(
                "full-model prefill did not execute each MLP overlay once"
            )
        return Gemma3CrossBlockModelExecution(
            model_output=model_output,
            carried_scalar=anchor.carried_scalar,
            native_anchor_feature=anchor.native_anchor_feature,
            retained_consumer_features=retained,
            consumer_mlp_output=consumer_output,
            anchor_overlay_calls=state.anchor_calls,
            consumer_overlay_calls=state.consumer_calls,
            intervening_native_layer_ids=tuple(
                self._adapter.layers[index].id
                for index in range(anchor_ordinal + 1, consumer_ordinal)
            ),
            source_whole_model_parameters=source_parameters,
            candidate_whole_model_learned_parameters=(
                candidate_parameters
            ),
        )


Gemma3DirectedCrossBlockMergedSupermodeExecutor = (
    Gemma3CrossBlockRowPrunedExecutor
)
Gemma3DirectedCrossBlockMergedSupermodeModelExecutor = (
    Gemma3CrossBlockModelExecutor
)


__all__ = [
    "Gemma3CrossBlockAnchorExecution",
    "Gemma3CrossBlockModelExecution",
    "Gemma3CrossBlockModelExecutor",
    "Gemma3DirectedCrossBlockMergedSupermodeExecutor",
    "Gemma3DirectedCrossBlockMergedSupermodeModelExecutor",
    "Gemma3CrossBlockRowPrunedAccounting",
    "Gemma3CrossBlockRowPrunedBinding",
    "Gemma3CrossBlockRowPrunedExecution",
    "Gemma3CrossBlockRowPrunedExecutor",
]
