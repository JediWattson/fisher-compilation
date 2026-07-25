"""Residual-separated, state-conditioned causal execution in modal space.

The constant weighted-Jacobian executor is useful for measuring whether one
fixed tangent graph is adequate.  It cannot express the empirical finding that
cross-token transport changes with the current activation regime.  This module
provides the next deliberately small reference executor:

```
y_t = skip(x_t) + x_t W_same + b
      + sum_(s < t) sum_e router(x_t, x_s, log(1 + lag))[e] V_e U_e x_s
```

The lag-zero path is explicit and independent of the causal experts.  Expert
edges exist only for earlier tensor slots whose logical position is strictly
smaller than the query position.  There are no learned position-pair tables or
future-edge parameter slots.  A shared low-width router chooses among shared
low-rank experts for every allowed positive-lag edge.  Relative lag enters
through one learned router-width vector, so the parameter shapes remain
independent of runtime sequence length and absolute position offsets.

This is a trainable modal-coordinate primitive.  Codecs, teacher fitting, and
nonlinear residual reconstruction intentionally live outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


RouterActivation = Literal["tanh", "silu"]

_ARTIFACT_KIND = "fisher_graph.residual_gated_causal_modal_executor"
_FORMAT_VERSION = 1
_CONFIG_FIELDS = {
    "input_modes",
    "output_modes",
    "expert_count",
    "expert_rank",
    "router_width",
    "same_position_skip",
    "max_positive_lag",
    "router_activation",
}
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "config",
    "dtype",
    "model_state_dict",
}
_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _require_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported executor dtype: {dtype}")
    return name


@dataclass(frozen=True, slots=True)
class GatedCausalModalExecutorConfig:
    """Shape and routing contract for the gated causal modal executor."""

    input_modes: int
    output_modes: int
    expert_count: int
    expert_rank: int
    router_width: int
    same_position_skip: bool = True
    max_positive_lag: int | None = None
    router_activation: RouterActivation = "tanh"

    def __post_init__(self) -> None:
        for label in (
            "input_modes",
            "output_modes",
            "expert_count",
            "expert_rank",
            "router_width",
        ):
            _require_positive_integer(getattr(self, label), label=label)
        if type(self.same_position_skip) is not bool:
            raise TypeError("same_position_skip must be boolean")
        if (
            self.same_position_skip
            and self.input_modes != self.output_modes
        ):
            raise ValueError(
                "same_position_skip requires equal input and output modes"
            )
        if self.max_positive_lag is not None:
            _require_positive_integer(
                self.max_positive_lag,
                label="max_positive_lag",
            )
        if self.router_activation not in ("tanh", "silu"):
            raise ValueError(
                "router_activation must be either 'tanh' or 'silu'"
            )


@dataclass(frozen=True, slots=True)
class GatedCausalModalComponents:
    """Inspectable decomposition of one executor call.

    ``router_probabilities`` and ``positive_lag_mask`` have shapes
    ``[batch, query, key, expert]`` and ``[batch, query, key]``.  Probabilities
    are exactly zero outside the positive-lag mask and sum to one over experts
    on each allowed edge.
    """

    output: Tensor
    same_position_output: Tensor
    positive_lag_output: Tensor
    router_probabilities: Tensor
    positive_lag_mask: Tensor


@dataclass(frozen=True, slots=True)
class GatedCausalExecutionAccounting:
    """Logical coefficient and multiply-accumulate accounting.

    MAC counts describe an ideal sparse implementation of the mathematical
    graph.  They exclude additions, bias application, masking, router
    activation, and softmax.  They are therefore an analytic comparison, not a
    claim about the speed of the dense PyTorch reference kernel.
    """

    batch_size: int
    sequence_length: int
    valid_query_tokens: int
    valid_key_tokens: int
    active_positive_lag_queries: int
    active_positive_lag_keys: int
    positive_lag_edges: int
    same_position_parameter_count: int
    expert_parameter_count: int
    router_parameter_count: int
    total_parameter_count: int
    same_position_mac_count: int
    expert_input_mac_count: int
    router_projection_mac_count: int
    router_lag_mac_count: int
    router_edge_mac_count: int
    expert_mixture_mac_count: int
    expert_output_mac_count: int
    total_mac_count: int
    dense_affine_reference_mac_count: int

    @property
    def mac_to_dense_affine_ratio(self) -> float:
        if self.dense_affine_reference_mac_count == 0:
            return 0.0
        return self.total_mac_count / self.dense_affine_reference_mac_count


class ResidualGatedCausalModalExecutor(nn.Module):
    """A variable-length modal executor with local and positive-lag paths.

    The same-position component is

    ``x_t`` (when ``same_position_skip`` is enabled) plus
    ``x_t @ same_position_weight + same_position_bias``.

    A positive-lag expert first maps a source coordinate through
    ``expert_input_weight[e]``.  The causal router mixes those source latents
    by expert for one query, and ``expert_output_weight[e]`` maps each mixed
    latent to output coordinates.  No expert can see its query's own input
    through this path.
    """

    def __init__(
        self,
        config: GatedCausalModalExecutorConfig,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, GatedCausalModalExecutorConfig):
            raise TypeError("config must be a GatedCausalModalExecutorConfig")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("executor dtype must be floating point")
        _dtype_name(dtype)
        self.config = config

        factory = {"dtype": dtype, "device": device}
        self.same_position_weight = nn.Parameter(
            torch.empty(
                config.input_modes,
                config.output_modes,
                **factory,
            )
        )
        self.same_position_bias = nn.Parameter(
            torch.empty(config.output_modes, **factory)
        )
        self.expert_input_weight = nn.Parameter(
            torch.empty(
                config.expert_count,
                config.input_modes,
                config.expert_rank,
                **factory,
            )
        )
        self.expert_output_weight = nn.Parameter(
            torch.empty(
                config.expert_count,
                config.expert_rank,
                config.output_modes,
                **factory,
            )
        )
        self.router_query_weight = nn.Parameter(
            torch.empty(
                config.input_modes,
                config.router_width,
                **factory,
            )
        )
        self.router_key_weight = nn.Parameter(
            torch.empty(
                config.input_modes,
                config.router_width,
                **factory,
            )
        )
        self.router_output_weight = nn.Parameter(
            torch.empty(
                config.router_width,
                config.expert_count,
                **factory,
            )
        )
        self.router_bias = nn.Parameter(
            torch.empty(config.expert_count, **factory)
        )
        self.router_lag_weight = nn.Parameter(
            torch.empty(config.router_width, **factory)
        )
        self.reset_parameters()

    @property
    def input_modes(self) -> int:
        return self.config.input_modes

    @property
    def output_modes(self) -> int:
        return self.config.output_modes

    @property
    def dtype(self) -> torch.dtype:
        return self.same_position_weight.dtype

    @property
    def device(self) -> torch.device:
        return self.same_position_weight.device

    def reset_parameters(self) -> None:
        # With a skip, the executor starts as an exact identity rather than a
        # random perturbation of a pretrained residual stream.
        if self.config.same_position_skip:
            nn.init.zeros_(self.same_position_weight)
        else:
            nn.init.xavier_uniform_(self.same_position_weight)
        nn.init.zeros_(self.same_position_bias)
        nn.init.xavier_uniform_(self.expert_input_weight)
        nn.init.xavier_uniform_(self.expert_output_weight)
        nn.init.xavier_uniform_(self.router_query_weight)
        nn.init.xavier_uniform_(self.router_key_weight)
        nn.init.xavier_uniform_(self.router_output_weight)
        nn.init.zeros_(self.router_bias)
        nn.init.zeros_(self.router_lag_weight)

    @property
    def same_position_parameter_count(self) -> int:
        return self.input_modes * self.output_modes + self.output_modes

    @property
    def expert_parameter_count(self) -> int:
        config = self.config
        return (
            config.expert_count
            * config.expert_rank
            * (config.input_modes + config.output_modes)
        )

    @property
    def router_parameter_count(self) -> int:
        config = self.config
        return (
            2 * config.input_modes * config.router_width
            + config.router_width * config.expert_count
            + config.expert_count
            + config.router_width
        )

    @property
    def learned_parameter_count(self) -> int:
        return (
            self.same_position_parameter_count
            + self.expert_parameter_count
            + self.router_parameter_count
        )

    def _validate_coordinates(self, coordinates: Tensor) -> None:
        if not isinstance(coordinates, Tensor):
            raise TypeError("modal coordinates must be a Tensor")
        if (
            coordinates.ndim != 3
            or coordinates.shape[0] == 0
            or coordinates.shape[1] == 0
            or coordinates.shape[2] != self.input_modes
        ):
            raise ValueError(
                "modal coordinates must have nonempty shape "
                "[batch, sequence, input_modes]"
            )
        if not coordinates.is_floating_point():
            raise TypeError("modal coordinates must be floating point")
        if (
            coordinates.dtype != self.dtype
            or coordinates.device != self.device
        ):
            raise ValueError(
                "modal coordinates must match executor dtype and device"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError("modal coordinates must be finite")

    def _normalize_sequence_inputs(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None,
        key_valid_mask: Tensor | None,
        logical_positions: Tensor | None,
        key_logical_positions: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate_coordinates(coordinates)
        batch_size, sequence_length, _ = coordinates.shape
        if query_valid_mask is None and key_valid_mask is None:
            query_valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=coordinates.device,
            )
            key_valid_mask = query_valid_mask
        elif query_valid_mask is None:
            query_valid_mask = key_valid_mask
        elif key_valid_mask is None:
            key_valid_mask = query_valid_mask
        assert query_valid_mask is not None
        assert key_valid_mask is not None
        for label, mask in (
            ("query_valid_mask", query_valid_mask),
            ("key_valid_mask", key_valid_mask),
        ):
            if not isinstance(mask, Tensor):
                raise TypeError(f"{label} must be a Tensor")
            if mask.dtype is not torch.bool:
                raise ValueError(f"{label} must be boolean")
            if mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"{label} must have shape "
                    f"{(batch_size, sequence_length)}"
                )
            if mask.device != coordinates.device:
                raise ValueError(
                    f"{label} must share the coordinate device"
                )

        if logical_positions is None:
            logical_positions = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=coordinates.device,
            ).unsqueeze(0).expand(batch_size, -1)
        if key_logical_positions is None:
            key_logical_positions = logical_positions
        for label, positions in (
            ("logical_positions", logical_positions),
            ("key_logical_positions", key_logical_positions),
        ):
            if not isinstance(positions, Tensor):
                raise TypeError(f"{label} must be a Tensor")
            if positions.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{label} must use an integer dtype")
            if positions.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"{label} must have shape "
                    f"{(batch_size, sequence_length)}"
                )
            if positions.device != coordinates.device:
                raise ValueError(
                    f"{label} must share the coordinate device"
                )
        self._validate_logical_position_invariants(
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        return (
            query_valid_mask,
            key_valid_mask,
            logical_positions,
            key_logical_positions,
        )

    @staticmethod
    def _validate_logical_position_invariants(
        *,
        query_valid_mask: Tensor,
        key_valid_mask: Tensor,
        logical_positions: Tensor,
        key_logical_positions: Tensor,
    ) -> None:
        """Validate invariants shared by execution and analytic accounting."""

        if (logical_positions[query_valid_mask] < 0).any():
            raise ValueError("valid query logical positions cannot be negative")
        if (key_logical_positions[key_valid_mask] < 0).any():
            raise ValueError("valid key logical positions cannot be negative")

        # This mirrors the dynamic executor's prefill contract and makes a
        # lower tensor index an unambiguous earlier key slot.
        batch_size, sequence_length = key_valid_mask.shape
        seen = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=key_valid_mask.device,
        )
        last_position = torch.zeros(
            batch_size,
            dtype=key_logical_positions.dtype,
            device=key_logical_positions.device,
        )
        for index in range(sequence_length):
            valid = key_valid_mask[:, index]
            position = key_logical_positions[:, index]
            if (valid & seen & (position <= last_position)).any():
                raise ValueError(
                    "valid key logical positions must be strictly increasing"
                )
            last_position = torch.where(valid, position, last_position)
            seen = seen | valid

    def _positive_lag_mask(
        self,
        *,
        query_valid_mask: Tensor,
        key_valid_mask: Tensor,
        logical_positions: Tensor,
        key_logical_positions: Tensor,
    ) -> Tensor:
        sequence_length = query_valid_mask.shape[1]
        slot_causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query_valid_mask.device,
        ).tril(diagonal=-1)
        relative_lag = (
            logical_positions.unsqueeze(2)
            - key_logical_positions.unsqueeze(1)
        )
        allowed = (
            query_valid_mask.unsqueeze(2)
            & key_valid_mask.unsqueeze(1)
            & slot_causal.unsqueeze(0)
            & (relative_lag > 0)
        )
        if self.config.max_positive_lag is not None:
            allowed = allowed & (
                relative_lag <= self.config.max_positive_lag
            )
        return allowed

    def _activate_router(self, values: Tensor) -> Tensor:
        if self.config.router_activation == "tanh":
            return torch.tanh(values)
        return F.silu(values)

    def _execute(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None,
        key_valid_mask: Tensor | None,
        logical_positions: Tensor | None,
        key_logical_positions: Tensor | None,
        collect_router: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            query_valid_mask,
            key_valid_mask,
            logical_positions,
            key_logical_positions,
        ) = self._normalize_sequence_inputs(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        batch_size, sequence_length, _ = coordinates.shape
        query_safe = torch.where(
            query_valid_mask.unsqueeze(-1),
            coordinates,
            torch.zeros_like(coordinates),
        )
        key_safe = torch.where(
            key_valid_mask.unsqueeze(-1),
            coordinates,
            torch.zeros_like(coordinates),
        )

        same_position = (
            query_safe @ self.same_position_weight
            + self.same_position_bias
        )
        if self.config.same_position_skip:
            same_position = same_position + query_safe
        same_position = torch.where(
            query_valid_mask.unsqueeze(-1),
            same_position,
            torch.zeros_like(same_position),
        )

        positive_lag_mask = self._positive_lag_mask(
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        query_route = query_safe @ self.router_query_weight
        key_route = key_safe @ self.router_key_weight
        source_latent = torch.einsum(
            "bsi,eir->bser",
            key_safe,
            self.expert_input_weight,
        )
        positive_lag = coordinates.new_zeros(
            batch_size,
            sequence_length,
            self.output_modes,
        )
        if collect_router:
            router_probabilities = coordinates.new_zeros(
                batch_size,
                sequence_length,
                sequence_length,
                self.config.expert_count,
            )
        else:
            # This tensor is not returned by ``forward``; a zero-width key axis
            # avoids allocating the full quadratic inspection buffer.
            router_probabilities = coordinates.new_zeros(
                batch_size,
                sequence_length,
                0,
                self.config.expert_count,
            )

        # Every target explicitly slices only earlier source slots.  The
        # logical-lag mask then removes equal-position or otherwise ineligible
        # earlier slots.  There is no code path that indexes a future slot.
        for target in range(1, sequence_length):
            allowed = positive_lag_mask[:, target, :target]
            pair_hidden = self._activate_router(
                query_route[:, target].unsqueeze(1)
                + key_route[:, :target]
                + torch.log1p(
                    (
                        logical_positions[:, target].unsqueeze(1)
                        - key_logical_positions[:, :target]
                    )
                    .clamp_min(0)
                    .to(dtype=coordinates.dtype)
                ).unsqueeze(-1)
                * self.router_lag_weight
            )
            logits = (
                pair_hidden @ self.router_output_weight
                + self.router_bias
            )
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = torch.where(
                allowed.unsqueeze(-1),
                probabilities,
                torch.zeros_like(probabilities),
            )
            expert_state = torch.einsum(
                "bse,bser->ber",
                probabilities,
                source_latent[:, :target],
            )
            cross = torch.einsum(
                "ber,ero->bo",
                expert_state,
                self.expert_output_weight,
            )
            positive_lag[:, target] = cross
            if collect_router:
                router_probabilities[:, target, :target] = probabilities

        positive_lag = torch.where(
            query_valid_mask.unsqueeze(-1),
            positive_lag,
            torch.zeros_like(positive_lag),
        )
        output = same_position + positive_lag
        return (
            output,
            same_position,
            positive_lag,
            router_probabilities,
            positive_lag_mask,
        )

    def forward(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> Tensor:
        """Execute modal coordinates without allocating router diagnostics."""

        output, _, _, _, _ = self._execute(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
            collect_router=False,
        )
        return output

    def forward_components(
        self,
        coordinates: Tensor,
        *,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> GatedCausalModalComponents:
        """Execute and expose the local, cross-token, and router components."""

        (
            output,
            same_position,
            positive_lag,
            router_probabilities,
            positive_lag_mask,
        ) = self._execute(
            coordinates,
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
            collect_router=True,
        )
        return GatedCausalModalComponents(
            output=output,
            same_position_output=same_position,
            positive_lag_output=positive_lag,
            router_probabilities=router_probabilities,
            positive_lag_mask=positive_lag_mask,
        )

    def execution_accounting(
        self,
        sequence_length: int,
        *,
        batch_size: int = 1,
        query_valid_mask: Tensor | None = None,
        key_valid_mask: Tensor | None = None,
        logical_positions: Tensor | None = None,
        key_logical_positions: Tensor | None = None,
    ) -> GatedCausalExecutionAccounting:
        """Return analytic logical MAC counts for one sequence contract."""

        _require_positive_integer(
            sequence_length,
            label="sequence_length",
        )
        _require_positive_integer(batch_size, label="batch_size")
        shape = (batch_size, sequence_length)

        if query_valid_mask is None and key_valid_mask is None:
            query_valid_mask = torch.ones(shape, dtype=torch.bool)
            key_valid_mask = query_valid_mask
        elif query_valid_mask is None:
            query_valid_mask = key_valid_mask
        elif key_valid_mask is None:
            key_valid_mask = query_valid_mask
        assert query_valid_mask is not None
        assert key_valid_mask is not None
        for label, mask in (
            ("query_valid_mask", query_valid_mask),
            ("key_valid_mask", key_valid_mask),
        ):
            if not isinstance(mask, Tensor):
                raise TypeError(f"{label} must be a Tensor")
            if mask.dtype is not torch.bool or mask.shape != shape:
                raise ValueError(
                    f"{label} must be boolean with shape {shape}"
                )
        accounting_device = query_valid_mask.device
        if key_valid_mask.device != accounting_device:
            raise ValueError("accounting masks must share a device")

        if logical_positions is None:
            logical_positions = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=accounting_device,
            ).unsqueeze(0).expand(batch_size, -1)
        if key_logical_positions is None:
            key_logical_positions = logical_positions
        for label, positions in (
            ("logical_positions", logical_positions),
            ("key_logical_positions", key_logical_positions),
        ):
            if not isinstance(positions, Tensor):
                raise TypeError(f"{label} must be a Tensor")
            if (
                positions.dtype not in (torch.int32, torch.int64)
                or positions.shape != shape
                or positions.device != accounting_device
            ):
                raise ValueError(
                    f"{label} must be an integer Tensor with shape {shape} "
                    "on the mask device"
                )
        self._validate_logical_position_invariants(
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )

        allowed = self._positive_lag_mask(
            query_valid_mask=query_valid_mask,
            key_valid_mask=key_valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=key_logical_positions,
        )
        active_queries = allowed.any(dim=2)
        active_keys = allowed.any(dim=1)
        valid_queries = int(query_valid_mask.sum().item())
        valid_keys = int(key_valid_mask.sum().item())
        query_count = int(active_queries.sum().item())
        key_count = int(active_keys.sum().item())
        edge_count = int(allowed.sum().item())
        config = self.config

        same_macs = valid_queries * config.input_modes * config.output_modes
        expert_input_macs = (
            key_count
            * config.expert_count
            * config.input_modes
            * config.expert_rank
        )
        router_projection_macs = (
            (query_count + key_count)
            * config.input_modes
            * config.router_width
        )
        router_edge_macs = (
            edge_count * config.router_width * config.expert_count
        )
        router_lag_macs = edge_count * config.router_width
        expert_mixture_macs = (
            edge_count * config.expert_count * config.expert_rank
        )
        expert_output_macs = (
            query_count
            * config.expert_count
            * config.expert_rank
            * config.output_modes
        )
        total_macs = (
            same_macs
            + expert_input_macs
            + router_projection_macs
            + router_lag_macs
            + router_edge_macs
            + expert_mixture_macs
            + expert_output_macs
        )
        dense_reference_macs = (
            (valid_queries + edge_count)
            * config.input_modes
            * config.output_modes
        )
        return GatedCausalExecutionAccounting(
            batch_size=batch_size,
            sequence_length=sequence_length,
            valid_query_tokens=valid_queries,
            valid_key_tokens=valid_keys,
            active_positive_lag_queries=query_count,
            active_positive_lag_keys=key_count,
            positive_lag_edges=edge_count,
            same_position_parameter_count=(
                self.same_position_parameter_count
            ),
            expert_parameter_count=self.expert_parameter_count,
            router_parameter_count=self.router_parameter_count,
            total_parameter_count=self.learned_parameter_count,
            same_position_mac_count=same_macs,
            expert_input_mac_count=expert_input_macs,
            router_projection_mac_count=router_projection_macs,
            router_lag_mac_count=router_lag_macs,
            router_edge_mac_count=router_edge_macs,
            expert_mixture_mac_count=expert_mixture_macs,
            expert_output_mac_count=expert_output_macs,
            total_mac_count=total_macs,
            dense_affine_reference_mac_count=dense_reference_macs,
        )

    def _validated_cpu_state(self) -> dict[str, Tensor]:
        state = {}
        for name, value in sorted(super().state_dict().items()):
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.dtype != self.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"executor state {name!r} must be finite {self.dtype}"
                )
            state[name] = value.detach().to(device="cpu").clone()
        return state

    def artifact_state_dict(self) -> dict[str, object]:
        """Return a deterministic, weights-only-safe executor artifact."""

        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "config": asdict(self.config),
            "dtype": _dtype_name(self.dtype),
            "model_state_dict": self._validated_cpu_state(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        device: torch.device | str | None = None,
    ) -> ResidualGatedCausalModalExecutor:
        """Strictly restore a deterministic artifact.

        Unknown or missing fields, noncanonical config types, state key/shape
        drift, dtype drift, and nonfinite values are rejected.
        """

        if not isinstance(state, Mapping) or set(state) != _ARTIFACT_FIELDS:
            raise ValueError("gated executor artifact fields are invalid")
        if state["artifact_kind"] != _ARTIFACT_KIND:
            raise ValueError("unsupported gated executor artifact kind")
        if (
            type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("unsupported gated executor format version")

        raw_config = state["config"]
        if (
            not isinstance(raw_config, Mapping)
            or set(raw_config) != _CONFIG_FIELDS
        ):
            raise ValueError("gated executor config fields are invalid")
        config = GatedCausalModalExecutorConfig(
            input_modes=raw_config["input_modes"],
            output_modes=raw_config["output_modes"],
            expert_count=raw_config["expert_count"],
            expert_rank=raw_config["expert_rank"],
            router_width=raw_config["router_width"],
            same_position_skip=raw_config["same_position_skip"],
            max_positive_lag=raw_config["max_positive_lag"],
            router_activation=raw_config["router_activation"],
        )
        dtype_name = state["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
            raise ValueError("gated executor artifact dtype is invalid")
        dtype = _DTYPES[dtype_name]
        executor = cls(config, dtype=dtype, device=device)

        raw_model_state = state["model_state_dict"]
        expected_state = super(
            ResidualGatedCausalModalExecutor,
            executor,
        ).state_dict()
        if (
            not isinstance(raw_model_state, Mapping)
            or set(raw_model_state) != set(expected_state)
        ):
            raise ValueError("gated executor model state fields are invalid")
        restored: dict[str, Tensor] = {}
        for name, expected in expected_state.items():
            value = raw_model_state[name]
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"gated executor state {name!r} must be a Tensor"
                )
            if value.device.type != "cpu":
                raise ValueError(
                    f"gated executor state {name!r} must be on CPU"
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"gated executor state {name!r} has the wrong dtype"
                )
            if value.shape != expected.shape:
                raise ValueError(
                    f"gated executor state {name!r} has the wrong shape"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"gated executor state {name!r} must be finite"
                )
            restored[name] = value.detach().to(device=device).clone()
        executor.load_state_dict(restored, strict=True)
        return executor


__all__ = [
    "GatedCausalExecutionAccounting",
    "GatedCausalModalComponents",
    "GatedCausalModalExecutorConfig",
    "ResidualGatedCausalModalExecutor",
    "RouterActivation",
]
