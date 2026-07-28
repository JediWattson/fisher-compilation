"""Authenticated edge-torn execution for one staged causal modal pair.

The pair has two node-local maps and one signed causal lag edge::

    m3 = R3 (x3 - mean_x3)
    y3 = mean_y3 + P3 m3

    m4 = R4 (x4_reference - mean_x4) + sum_lag K[lag](m3)
    y4 = mean_y4 + P4 m4

``x4_reference`` is structurally important.  It is the prompt- and
sequence-shaped downstream input produced with the L3 source fixed at
``mean_y3``.  The modal source state is centered at that same mean, so the
signed ``K`` edge restores only departures from the reference exactly once.
It is not an ordinary-path L4 input.  A candidate still needs an external
stationarity and fidelity gate: this runtime authenticates the declared
reference, not the scientific adequacy of a stationary lag map.

Every execution also requires logical positions and a validity mask.  Lag is
the logical target position minus the logical source position; physical tensor
offsets are never used as a fallback.

Plans are canonical CPU/float64 scientific artifacts.  Public offline
execution reauthenticates them.  Prepared runtimes validate and copy once,
then provide conversion-free factorized, dense-control, and staged hot paths.
The boundary contract authenticates the declared semantics and mean binding;
it cannot prove that an external tensor provider actually honored them.
Neither the contract nor this analysis runtime grants source-model replacement
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .causal_edge_jvp import CausalEdgeJVPFit
from .modal_connectivity_modes import ModalConnectivityFactor


__all__ = [
    "MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS",
    "CausalModalPairAccounting",
    "CausalModalPairPlan",
    "EdgeTornModalPairBoundaryContract",
    "MaterializedCausalModalPair",
    "PreparedCausalModalPair",
    "PreparedCausalModalPairSession",
    "bind_causal_modal_pair_plan",
]


MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS = (
    "mean_source_reference_edge_torn_base"
)

_FORMAT_VERSION = 1
_BOUNDARY_KIND = "fisher_graph.edge_torn_modal_pair_boundary"
_PLAN_KIND = "fisher_graph.causal_modal_pair_plan"
_MATERIALIZED_KIND = "fisher_graph.materialized_causal_modal_pair"
_BOUNDARY_DOMAIN = b"fisher_graph.edge_torn_modal_pair_boundary.v1\0"
_PLAN_DOMAIN = b"fisher_graph.causal_modal_pair_plan.v1\0"
_MATERIALIZED_DOMAIN = b"fisher_graph.materialized_causal_modal_pair.v1\0"
_BINDING_DOMAIN = b"fisher_graph.causal_modal_pair_binding.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.causal_modal_pair_tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_RUNTIME_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_LOGICAL_POSITION_DTYPES = frozenset({torch.int32, torch.int64})


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical name")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _canonical_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    canonical = _canonical_tensor(
        value,
        label=label,
        ndim=value.ndim if isinstance(value, Tensor) else 0,
    )
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_copy(
    value: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return (
        value.detach()
        .to(device=device, dtype=dtype)
        .contiguous()
        .clone()
    )


def _leading_shape_product(leading_shape: tuple[int, ...]) -> int:
    if (
        type(leading_shape) is not tuple
        or not leading_shape
        or any(type(value) is not int or value <= 0 for value in leading_shape)
    ):
        raise ValueError(
            "leading_shape must be a nonempty tuple of positive integers"
        )
    return math.prod(leading_shape)


def _validate_runtime_input(
    value: Tensor,
    *,
    label: str,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, ...]:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    if value.ndim < 2:
        raise ValueError(
            f"{label} must expose a sequence axis before its feature axis"
        )
    if value.shape[-1] != width:
        raise ValueError(f"{label} has the wrong width")
    if value.shape[-2] <= 0:
        raise ValueError(f"{label} cannot have an empty sequence axis")
    if value.device != device:
        raise ValueError(f"{label} is on the wrong device")
    if value.dtype != dtype:
        raise ValueError(f"{label} has the wrong dtype")
    return tuple(value.shape[:-1])


def _validate_logical_grid(
    logical_positions: Tensor,
    valid_mask: Tensor,
    *,
    leading_shape: tuple[int, ...],
    device: torch.device,
    label: str,
) -> tuple[Tensor, Tensor]:
    """Validate and broadcast one explicit logical sequence grid."""

    if not isinstance(logical_positions, Tensor):
        raise TypeError(f"{label} logical_positions must be a Tensor")
    if logical_positions.dtype not in _LOGICAL_POSITION_DTYPES:
        raise TypeError(
            f"{label} logical_positions must use torch.int32 or torch.int64"
        )
    if logical_positions.device != device:
        raise ValueError(f"{label} logical_positions are on the wrong device")
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError(f"{label} valid_mask must be a boolean Tensor")
    if valid_mask.device != device:
        raise ValueError(f"{label} valid_mask is on the wrong device")

    sequence_length = leading_shape[-1]
    broadcast_shape = (sequence_length,)
    for name, value in (
        ("logical_positions", logical_positions),
        ("valid_mask", valid_mask),
    ):
        if tuple(value.shape) not in {broadcast_shape, leading_shape}:
            raise ValueError(
                f"{label} {name} must have shape {broadcast_shape} "
                f"or the full leading shape {leading_shape}"
            )
    positions = (
        logical_positions.expand(leading_shape)
        if tuple(logical_positions.shape) == broadcast_shape
        else logical_positions
    )
    mask = (
        valid_mask.expand(leading_shape)
        if tuple(valid_mask.shape) == broadcast_shape
        else valid_mask
    )
    if ((positions < 0) & mask).any():
        raise ValueError(
            f"{label} valid logical_positions must be nonnegative"
        )

    # Valid positions define a logical sequence, not merely a bag of labels.
    # Enforce strict increase even when invalid physical rows create gaps.
    source_position = positions.unsqueeze(-1)
    target_position = positions.unsqueeze(-2)
    source_valid = mask.unsqueeze(-1)
    target_valid = mask.unsqueeze(-2)
    physical_order = torch.triu(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )
    ordered_valid_pairs = source_valid & target_valid & physical_order
    if (
        ordered_valid_pairs
        & (target_position <= source_position)
    ).any():
        raise ValueError(
            f"{label} valid logical_positions must be strictly increasing"
        )
    return positions, mask


def _causal_logical_lag_row_map(
    values: Tensor,
    kernels: Tensor,
    *,
    source_logical_positions: Tensor,
    source_valid_mask: Tensor,
    target_logical_positions: Tensor,
    target_valid_mask: Tensor,
) -> Tensor:
    """Apply ``K[lag,input,output]`` only to explicit valid logical pairs."""

    sequence_length = values.shape[-2]
    outer_count = math.prod(values.shape[:-2]) or 1
    flat_values = values.reshape(outer_count, sequence_length, values.shape[-1])
    flat_output = values.new_zeros(
        (outer_count * sequence_length, kernels.shape[-1])
    )
    logical_lags = (
        target_logical_positions.unsqueeze(-1)
        - source_logical_positions.unsqueeze(-2)
    ).reshape(outer_count, sequence_length, sequence_length)
    valid_pairs = (
        target_valid_mask.unsqueeze(-1)
        & source_valid_mask.unsqueeze(-2)
    ).reshape(outer_count, sequence_length, sequence_length)
    for lag in range(kernels.shape[0]):
        pair_indices = (valid_pairs & (logical_lags == lag)).nonzero(
            as_tuple=False
        )
        if pair_indices.numel() == 0:
            continue
        outer_indices = pair_indices[:, 0]
        target_indices = pair_indices[:, 1]
        source_indices = pair_indices[:, 2]
        mapped = (
            flat_values[outer_indices, source_indices] @ kernels[lag]
        )
        flat_target_indices = (
            outer_indices * sequence_length + target_indices
        )
        flat_output = flat_output.index_add(
            0,
            flat_target_indices,
            mapped,
        )
    return flat_output.reshape(
        (*target_logical_positions.shape, kernels.shape[-1])
    )


@dataclass(frozen=True, slots=True)
class EdgeTornModalPairBoundaryContract:
    """Authenticated declaration of the pair's intended L4 input.

    The L4 base is the response with the L3 source fixed at the exact
    authenticated ``y3_mean``.  Its name must make ``mean_source``,
    ``reference``, and ``torn_base`` explicit.  Ordinary/native-path authority
    is deliberately inexpressible as a valid declaration.  Execution still
    needs an authenticated provider because a tensor's values do not carry
    this provenance.
    """

    stage3_source_name: str
    stage4_target_name: str
    x4_reference_torn_base_name: str
    y3_mean_sha256: str
    x4_semantics: str = MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS
    centered_source_edge_present_in_x4: bool = False
    ordinary_path_input_authorized: bool = False
    source_replacement_authority: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _BOUNDARY_KIND
    format_version: int = _FORMAT_VERSION

    @classmethod
    def for_y3_mean(
        cls,
        *,
        stage3_source_name: str,
        stage4_target_name: str,
        x4_reference_torn_base_name: str,
        y3_mean: Tensor,
    ) -> EdgeTornModalPairBoundaryContract:
        """Build a boundary bound to the canonical mean-source tensor."""

        return cls(
            stage3_source_name=stage3_source_name,
            stage4_target_name=stage4_target_name,
            x4_reference_torn_base_name=x4_reference_torn_base_name,
            y3_mean_sha256=_tensor_sha256(y3_mean, label="y3_mean"),
        )

    def __post_init__(self) -> None:
        _require_name(self.stage3_source_name, label="stage3_source_name")
        _require_name(self.stage4_target_name, label="stage4_target_name")
        _require_name(
            self.x4_reference_torn_base_name,
            label="x4_reference_torn_base_name",
        )
        _require_sha256(self.y3_mean_sha256, label="y3_mean_sha256")
        if self.stage3_source_name == self.stage4_target_name:
            raise ValueError("stage3 and stage4 boundary names must differ")
        normalized_x4_name = (
            self.x4_reference_torn_base_name.lower().replace("-", "_")
        )
        if (
            "mean_source" not in normalized_x4_name
            or "reference" not in normalized_x4_name
            or "torn_base" not in normalized_x4_name
            or "ordinary" in normalized_x4_name
            or "native" in normalized_x4_name
        ):
            raise ValueError(
                "x4_reference_torn_base_name must explicitly name a "
                "mean-source reference torn base and cannot name an "
                "ordinary or native path"
            )
        if self.x4_semantics != MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS:
            raise ValueError(
                "x4_semantics must declare the mean-source reference "
                "edge-torn base"
            )
        if (
            self.centered_source_edge_present_in_x4 is not False
            or self.ordinary_path_input_authorized is not False
            or self.source_replacement_authority is not False
        ):
            raise ValueError(
                "an edge-torn boundary cannot retain the centered source "
                "edge or grant ordinary/native-path or replacement authority"
            )
        if (
            self.artifact_kind != _BOUNDARY_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("edge-torn boundary artifact header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_BOUNDARY_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="boundary artifact_sha256",
                )
                != computed
            ):
                raise ValueError("edge-torn boundary artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "stage3_source_name": self.stage3_source_name,
            "stage4_target_name": self.stage4_target_name,
            "x4_reference_torn_base_name": (
                self.x4_reference_torn_base_name
            ),
            "y3_mean_sha256": self.y3_mean_sha256,
            "x4_semantics": self.x4_semantics,
            "centered_source_edge_present_in_x4": (
                self.centered_source_edge_present_in_x4
            ),
            "ordinary_path_input_authorized": (
                self.ordinary_path_input_authorized
            ),
            "source_replacement_authority": (
                self.source_replacement_authority
            ),
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_BOUNDARY_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("edge-torn boundary artifact hash mismatch")


@dataclass(frozen=True, slots=True)
class CausalModalPairAccounting:
    """Exact scalar, MAC, and staged-carry counts for one pair.

    Stored-scalar counts include all four explicit means.  Linear MAC counts
    exclude centering, bias addition, lag accumulation, allocation, and
    dispatch.  A causal pair is one source/target row pair admitted by a lag.
    """

    x3_width: int
    y3_width: int
    x4_width: int
    y4_width: int
    rank3: int
    rank4: int
    lag_count: int
    bytes_per_scalar: int

    def __post_init__(self) -> None:
        for name in (
            "x3_width",
            "y3_width",
            "x4_width",
            "y4_width",
            "rank3",
            "rank4",
            "lag_count",
            "bytes_per_scalar",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def mean_stored_scalar_count(self) -> int:
        return self.x3_width + self.y3_width + self.x4_width + self.y4_width

    @property
    def factorized_linear_stored_scalar_count(self) -> int:
        return (
            self.rank3 * self.x3_width
            + self.y3_width * self.rank3
            + self.rank4 * self.x4_width
            + self.y4_width * self.rank4
            + self.lag_count * self.rank3 * self.rank4
        )

    @property
    def factorized_stored_scalar_count(self) -> int:
        return (
            self.mean_stored_scalar_count
            + self.factorized_linear_stored_scalar_count
        )

    @property
    def dense_linear_stored_scalar_count(self) -> int:
        return (
            self.y3_width * self.x3_width
            + self.y4_width * self.x4_width
            + self.lag_count * self.y4_width * self.x3_width
        )

    @property
    def dense_stored_scalar_count(self) -> int:
        return self.mean_stored_scalar_count + self.dense_linear_stored_scalar_count

    @property
    def prepared_unique_stored_scalar_count(self) -> int:
        """Means shared once plus both factorized and materialized maps."""

        return (
            self.mean_stored_scalar_count
            + self.factorized_linear_stored_scalar_count
            + self.dense_linear_stored_scalar_count
        )

    @property
    def factorized_storage_bytes(self) -> int:
        return self.factorized_stored_scalar_count * self.bytes_per_scalar

    @property
    def dense_storage_bytes(self) -> int:
        return self.dense_stored_scalar_count * self.bytes_per_scalar

    @property
    def prepared_unique_storage_bytes(self) -> int:
        return (
            self.prepared_unique_stored_scalar_count * self.bytes_per_scalar
        )

    @property
    def factorized_local_macs_per_row(self) -> int:
        return (
            self.rank3 * self.x3_width
            + self.y3_width * self.rank3
            + self.rank4 * self.x4_width
            + self.y4_width * self.rank4
        )

    @property
    def factorized_edge_macs_per_causal_pair(self) -> int:
        return self.rank3 * self.rank4

    @property
    def dense_local_macs_per_row(self) -> int:
        return (
            self.y3_width * self.x3_width
            + self.y4_width * self.x4_width
        )

    @property
    def dense_edge_macs_per_causal_pair(self) -> int:
        return self.x3_width * self.y4_width

    @property
    def factorized_carry_scalars_per_source_row(self) -> int:
        return self.rank3

    @property
    def dense_carry_scalars_per_source_row(self) -> int:
        return self.x3_width

    @property
    def staged_logical_position_values_per_source_row(self) -> int:
        return 1

    @property
    def staged_valid_mask_values_per_source_row(self) -> int:
        return 1

    def causal_pair_count(
        self,
        leading_shape: tuple[int, ...],
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> int:
        """Count valid pairs admitted by explicit logical lags."""

        _leading_shape_product(leading_shape)
        positions, mask = _validate_logical_grid(
            logical_positions,
            valid_mask,
            leading_shape=leading_shape,
            device=logical_positions.device,
            label="accounting",
        )
        logical_lags = positions.unsqueeze(-1) - positions.unsqueeze(-2)
        valid_pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        admitted = (
            valid_pairs
            & (logical_lags >= 0)
            & (logical_lags < self.lag_count)
        )
        return int(admitted.sum().item())

    def factorized_macs(
        self,
        leading_shape: tuple[int, ...],
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> int:
        rows = _leading_shape_product(leading_shape)
        return (
            rows * self.factorized_local_macs_per_row
            + self.causal_pair_count(
                leading_shape,
                logical_positions=logical_positions,
                valid_mask=valid_mask,
            )
            * self.factorized_edge_macs_per_causal_pair
        )

    def dense_macs(
        self,
        leading_shape: tuple[int, ...],
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> int:
        rows = _leading_shape_product(leading_shape)
        return (
            rows * self.dense_local_macs_per_row
            + self.causal_pair_count(
                leading_shape,
                logical_positions=logical_positions,
                valid_mask=valid_mask,
            )
            * self.dense_edge_macs_per_causal_pair
        )

    def factorized_carry_scalar_count(
        self,
        leading_shape: tuple[int, ...],
    ) -> int:
        return (
            _leading_shape_product(leading_shape)
            * self.factorized_carry_scalars_per_source_row
        )

    def dense_carry_scalar_count(
        self,
        leading_shape: tuple[int, ...],
    ) -> int:
        return (
            _leading_shape_product(leading_shape)
            * self.dense_carry_scalars_per_source_row
        )

    def staged_carry_storage_bytes(
        self,
        leading_shape: tuple[int, ...],
        *,
        position_bytes: int,
        mask_bytes: int = 1,
    ) -> int:
        """Count factorized modal state plus copied grid metadata."""

        if type(position_bytes) is not int or position_bytes <= 0:
            raise ValueError("position_bytes must be a positive integer")
        if type(mask_bytes) is not int or mask_bytes <= 0:
            raise ValueError("mask_bytes must be a positive integer")
        rows = _leading_shape_product(leading_shape)
        return (
            rows
            * self.factorized_carry_scalars_per_source_row
            * self.bytes_per_scalar
            + rows
            * self.staged_logical_position_values_per_source_row
            * position_bytes
            + rows
            * self.staged_valid_mask_values_per_source_row
            * mask_bytes
        )


@dataclass(frozen=True, slots=True, eq=False)
class CausalModalPairPlan:
    """Canonical, proof-bound factorized plan for one edge-torn pair.

    ``K`` has shape ``[lag, rank3, rank4]`` and uses row-vector orientation:
    ``m3[..., source, :] @ K[lag]`` contributes to the rank-4 state at
    ``target = source + lag``.
    """

    boundary_contract: EdgeTornModalPairBoundaryContract
    source_artifact_sha256: str
    direct_jacobian_proof_sha256: str
    x3_mean: Tensor
    y3_mean: Tensor
    x4_mean: Tensor
    y4_mean: Tensor
    R3: Tensor
    P3: Tensor
    R4: Tensor
    P4: Tensor
    K: Tensor
    artifact_sha256: str = ""
    artifact_kind: str = _PLAN_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.boundary_contract,
            EdgeTornModalPairBoundaryContract,
        ):
            raise TypeError(
                "boundary_contract must be an "
                "EdgeTornModalPairBoundaryContract"
            )
        self.boundary_contract.validate_integrity()
        _require_sha256(
            self.source_artifact_sha256,
            label="source_artifact_sha256",
        )
        _require_sha256(
            self.direct_jacobian_proof_sha256,
            label="direct_jacobian_proof_sha256",
        )
        if (
            self.artifact_kind != _PLAN_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("causal modal pair artifact header is invalid")

        tensor_dimensions = {
            "x3_mean": 1,
            "y3_mean": 1,
            "x4_mean": 1,
            "y4_mean": 1,
            "R3": 2,
            "P3": 2,
            "R4": 2,
            "P4": 2,
            "K": 3,
        }
        for name, ndim in tensor_dimensions.items():
            canonical = _canonical_tensor(
                getattr(self, name),
                label=name,
                ndim=ndim,
            )
            object.__setattr__(self, name, canonical)
        self._validate_shapes()
        if (
            self.boundary_contract.y3_mean_sha256
            != _tensor_sha256(self.y3_mean, label="y3_mean")
        ):
            raise ValueError(
                "boundary contract is not bound to the plan y3_mean"
            )

        computed = _json_sha256(self._hash_payload(), domain=_PLAN_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="plan artifact_sha256",
                )
                != computed
            ):
                raise ValueError("causal modal pair artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def x3_width(self) -> int:
        return self.x3_mean.numel()

    @property
    def y3_width(self) -> int:
        return self.y3_mean.numel()

    @property
    def x4_width(self) -> int:
        return self.x4_mean.numel()

    @property
    def y4_width(self) -> int:
        return self.y4_mean.numel()

    @property
    def rank3(self) -> int:
        return self.R3.shape[0]

    @property
    def rank4(self) -> int:
        return self.R4.shape[0]

    @property
    def lag_count(self) -> int:
        return self.K.shape[0]

    @property
    def source_sha256(self) -> str:
        return self.source_artifact_sha256

    @property
    def proof_sha256(self) -> str:
        return self.direct_jacobian_proof_sha256

    def _validate_shapes(self) -> None:
        for name in ("x3_mean", "y3_mean", "x4_mean", "y4_mean"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.ndim != 1
                or value.numel() <= 0
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} is not a nonempty canonical mean tensor"
                )
        rank3 = self.R3.shape[0]
        rank4 = self.R4.shape[0]
        if rank3 <= 0 or rank4 <= 0 or self.K.shape[0] <= 0:
            raise ValueError("modal ranks and lag count must be positive")
        expected = {
            "R3": (rank3, self.x3_width),
            "P3": (self.y3_width, rank3),
            "R4": (rank4, self.x4_width),
            "P4": (self.y4_width, rank4),
            "K": (self.K.shape[0], rank3, rank4),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or tuple(value.shape) != shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} is not the canonical tensor with shape {shape}"
                )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "boundary_contract_sha256": self.boundary_contract.artifact_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "direct_jacobian_proof_sha256": (
                self.direct_jacobian_proof_sha256
            ),
            "y3_mean_sha256": self.boundary_contract.y3_mean_sha256,
            "x4_semantics": MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS,
            "lag_axis": "nonnegative_integer_source_to_target_lag",
            "edge_orientation": "K[lag,rank3,rank4]",
            "stationarity_status": "requires_external_fidelity_gate",
            "tensor_sha256s": {
                name: _tensor_sha256(getattr(self, name), label=name)
                for name in (
                    "x3_mean",
                    "y3_mean",
                    "x4_mean",
                    "y4_mean",
                    "R3",
                    "P3",
                    "R4",
                    "P4",
                    "K",
                )
            },
        }

    def validate_integrity(self) -> None:
        self.boundary_contract.validate_integrity()
        self._validate_shapes()
        if (
            self.boundary_contract.y3_mean_sha256
            != _tensor_sha256(self.y3_mean, label="y3_mean")
        ):
            raise ValueError(
                "boundary contract is not bound to the plan y3_mean"
            )
        if (
            _json_sha256(self._hash_payload(), domain=_PLAN_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("causal modal pair artifact hash mismatch")

    def accounting(
        self,
        *,
        bytes_per_scalar: int = 8,
    ) -> CausalModalPairAccounting:
        self.validate_integrity()
        return CausalModalPairAccounting(
            x3_width=self.x3_width,
            y3_width=self.y3_width,
            x4_width=self.x4_width,
            y4_width=self.y4_width,
            rank3=self.rank3,
            rank4=self.rank4,
            lag_count=self.lag_count,
            bytes_per_scalar=bytes_per_scalar,
        )

    def materialize_dense(self) -> MaterializedCausalModalPair:
        return MaterializedCausalModalPair.from_plan(self)

    def prepare(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedCausalModalPair:
        return PreparedCausalModalPair.from_plan(
            self,
            device=device,
            dtype=dtype,
        )

    def execute_factorized(
        self,
        x3: Tensor,
        *,
        x4_mean_source_reference_torn_base: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reauthenticate and execute the factorized plan offline."""

        self.validate_integrity()
        if not isinstance(x3, Tensor):
            raise TypeError("x3 must be a Tensor")
        if x3.dtype not in _RUNTIME_DTYPES:
            raise ValueError("x3 must use a supported floating Torch dtype")
        prepared = self.prepare(device=x3.device, dtype=x3.dtype)
        return prepared.execute_factorized(
            x3,
            x4_mean_source_reference_torn_base=(
                x4_mean_source_reference_torn_base
            ),
            logical_positions=logical_positions,
            valid_mask=valid_mask,
        )

    def execute_dense(
        self,
        x3: Tensor,
        *,
        x4_mean_source_reference_torn_base: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reauthenticate and execute the materialized equivalent offline."""

        return self.materialize_dense().execute(
            x3,
            x4_mean_source_reference_torn_base=(
                x4_mean_source_reference_torn_base
            ),
            logical_positions=logical_positions,
            valid_mask=valid_mask,
        )


def bind_causal_modal_pair_plan(
    stage3: ModalConnectivityFactor,
    stage4: ModalConnectivityFactor,
    edge_fit: CausalEdgeJVPFit,
    *,
    baseline_source: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
    x4_reference_torn_base_name: str,
) -> CausalModalPairPlan:
    """Bind two exact factors and one JVP proof into a pair plan.

    This authenticates the modal bases, centered linearization point, logical
    grid, and fitted kernel.  It deliberately does not authenticate a runtime
    provider for the prompt-conditioned reference-base tensor; that remains a
    separate compiler and fidelity gate.
    """

    if not isinstance(stage3, ModalConnectivityFactor) or not isinstance(
        stage4,
        ModalConnectivityFactor,
    ):
        raise TypeError("stage3 and stage4 must be ModalConnectivityFactor values")
    if not isinstance(edge_fit, CausalEdgeJVPFit):
        raise TypeError("edge_fit must be a CausalEdgeJVPFit")
    stage3.validate_integrity()
    stage4.validate_integrity()
    edge_fit.validate_integrity()
    if len(stage3.input_ports) != 1 or len(stage4.input_ports) != 1:
        raise ValueError("causal modal pair factors must each have one input port")
    if (
        stage3.output_port.causal_order
        >= stage4.input_ports[0].causal_order
    ):
        raise ValueError("stage3 must causally precede the stage4 input")
    if (
        stage3.reduction_id != stage4.reduction_id
        or stage3.sample_count != stage4.sample_count
    ):
        raise ValueError("causal modal pair factors must share moment lineage")
    if (
        stage3.retained_rank != edge_fit.source_rank
        or stage4.retained_rank != edge_fit.target_rank
    ):
        raise ValueError("factor ranks do not match the fitted JVP edge")
    if (
        not isinstance(baseline_source, Tensor)
        or baseline_source.ndim != 3
        or baseline_source.shape[0] != 1
        or baseline_source.shape[2] != stage3.output_port.width
    ):
        raise ValueError(
            "baseline_source must have shape [1, S, stage3 output width]"
        )
    expected_mean = stage3.output_mean.to(
        device=baseline_source.device,
        dtype=baseline_source.dtype,
    ).view(1, 1, -1).expand_as(baseline_source)
    if not torch.equal(baseline_source, expected_mean):
        raise ValueError(
            "baseline_source must be the expanded stage3 output mean"
        )
    edge_fit.validate_binding(
        baseline_source=baseline_source,
        source_decoder=stage3.prolongation,
        target_encoder=stage4.restriction.T,
        logical_positions=logical_positions,
        valid_mask=valid_mask,
    )
    boundary = EdgeTornModalPairBoundaryContract.for_y3_mean(
        stage3_source_name=stage3.output_port.name,
        stage4_target_name=stage4.input_ports[0].name,
        x4_reference_torn_base_name=x4_reference_torn_base_name,
        y3_mean=stage3.output_mean,
    )
    source_artifact_sha256 = _json_sha256(
        {
            "stage3_factor_sha256": stage3.artifact_sha256,
            "stage4_factor_sha256": stage4.artifact_sha256,
            "reduction_id": stage3.reduction_id,
            "sample_count": stage3.sample_count,
        },
        domain=_BINDING_DOMAIN,
    )
    return CausalModalPairPlan(
        boundary_contract=boundary,
        source_artifact_sha256=source_artifact_sha256,
        direct_jacobian_proof_sha256=edge_fit.artifact_sha256,
        x3_mean=stage3.input_mean,
        y3_mean=stage3.output_mean,
        x4_mean=stage4.input_mean,
        y4_mean=stage4.output_mean,
        R3=stage3.restriction,
        P3=stage3.prolongation,
        R4=stage4.restriction,
        P4=stage4.prolongation,
        K=edge_fit.kernel,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class MaterializedCausalModalPair:
    """Authenticated dense materialization of a causal pair plan."""

    source_plan_sha256: str
    boundary_contract: EdgeTornModalPairBoundaryContract
    source_artifact_sha256: str
    direct_jacobian_proof_sha256: str
    rank3: int
    rank4: int
    x3_mean: Tensor
    y3_mean: Tensor
    x4_mean: Tensor
    y4_mean: Tensor
    stage3_matrix: Tensor
    stage4_torn_base_matrix: Tensor
    lag_edge_matrices: Tensor
    artifact_sha256: str
    artifact_kind: str
    format_version: int

    @classmethod
    def from_plan(
        cls,
        plan: CausalModalPairPlan,
    ) -> MaterializedCausalModalPair:
        return cls(plan)

    def __init__(self, plan: CausalModalPairPlan) -> None:
        if not isinstance(plan, CausalModalPairPlan):
            raise TypeError("plan must be a CausalModalPairPlan")
        plan.validate_integrity()
        stage3_matrix = (plan.P3 @ plan.R3).contiguous()
        stage4_matrix = (plan.P4 @ plan.R4).contiguous()
        lag_matrices = torch.stack(
            tuple(
                plan.P4 @ plan.K[lag].T @ plan.R3
                for lag in range(plan.lag_count)
            ),
            dim=0,
        ).contiguous()
        object.__setattr__(self, "source_plan_sha256", plan.artifact_sha256)
        object.__setattr__(
            self,
            "boundary_contract",
            plan.boundary_contract,
        )
        object.__setattr__(
            self,
            "source_artifact_sha256",
            plan.source_artifact_sha256,
        )
        object.__setattr__(
            self,
            "direct_jacobian_proof_sha256",
            plan.direct_jacobian_proof_sha256,
        )
        object.__setattr__(self, "rank3", plan.rank3)
        object.__setattr__(self, "rank4", plan.rank4)
        for name in ("x3_mean", "y3_mean", "x4_mean", "y4_mean"):
            object.__setattr__(self, name, getattr(plan, name).clone())
        object.__setattr__(self, "stage3_matrix", stage3_matrix)
        object.__setattr__(
            self,
            "stage4_torn_base_matrix",
            stage4_matrix,
        )
        object.__setattr__(self, "lag_edge_matrices", lag_matrices)
        object.__setattr__(self, "artifact_kind", _MATERIALIZED_KIND)
        object.__setattr__(self, "format_version", _FORMAT_VERSION)
        object.__setattr__(
            self,
            "artifact_sha256",
            _json_sha256(
                self._hash_payload(),
                domain=_MATERIALIZED_DOMAIN,
            ),
        )

    @property
    def x3_width(self) -> int:
        return self.x3_mean.numel()

    @property
    def x4_width(self) -> int:
        return self.x4_mean.numel()

    @property
    def y3_width(self) -> int:
        return self.y3_mean.numel()

    @property
    def y4_width(self) -> int:
        return self.y4_mean.numel()

    @property
    def lag_count(self) -> int:
        return self.lag_edge_matrices.shape[0]

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_plan_sha256": self.source_plan_sha256,
            "boundary_contract_sha256": self.boundary_contract.artifact_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "direct_jacobian_proof_sha256": (
                self.direct_jacobian_proof_sha256
            ),
            "rank3": self.rank3,
            "rank4": self.rank4,
            "y3_mean_sha256": self.boundary_contract.y3_mean_sha256,
            "x4_semantics": MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS,
            "stationarity_status": "requires_external_fidelity_gate",
            "tensor_sha256s": {
                name: _tensor_sha256(getattr(self, name), label=name)
                for name in (
                    "x3_mean",
                    "y3_mean",
                    "x4_mean",
                    "y4_mean",
                    "stage3_matrix",
                    "stage4_torn_base_matrix",
                    "lag_edge_matrices",
                )
            },
        }

    def validate_integrity(self) -> None:
        self.boundary_contract.validate_integrity()
        if (
            self.stage3_matrix.shape
            != (self.y3_width, self.x3_width)
            or self.stage4_torn_base_matrix.shape
            != (self.y4_width, self.x4_width)
            or self.lag_edge_matrices.shape
            != (self.lag_count, self.y4_width, self.x3_width)
            or self.lag_count <= 0
        ):
            raise ValueError("materialized causal pair tensor shapes drifted")
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_MATERIALIZED_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("materialized causal pair artifact hash mismatch")

    def accounting(
        self,
        *,
        bytes_per_scalar: int = 8,
    ) -> CausalModalPairAccounting:
        self.validate_integrity()
        return CausalModalPairAccounting(
            x3_width=self.x3_width,
            y3_width=self.y3_width,
            x4_width=self.x4_width,
            y4_width=self.y4_width,
            rank3=self.rank3,
            rank4=self.rank4,
            lag_count=self.lag_count,
            bytes_per_scalar=bytes_per_scalar,
        )

    def execute(
        self,
        x3: Tensor,
        *,
        x4_mean_source_reference_torn_base: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reauthenticate and execute the dense materialization offline."""

        self.validate_integrity()
        if not isinstance(x3, Tensor):
            raise TypeError("x3 must be a Tensor")
        if x3.dtype not in _RUNTIME_DTYPES:
            raise ValueError("x3 must use a supported floating Torch dtype")
        device = x3.device
        dtype = x3.dtype
        x3_leading = _validate_runtime_input(
            x3,
            label="x3",
            width=self.x3_width,
            device=device,
            dtype=dtype,
        )
        x4_leading = _validate_runtime_input(
            x4_mean_source_reference_torn_base,
            label="x4_mean_source_reference_torn_base",
            width=self.x4_width,
            device=device,
            dtype=dtype,
        )
        if x4_leading != x3_leading:
            raise ValueError(
                "x4_mean_source_reference_torn_base must share x3's "
                "leading shape"
            )
        positions, mask = _validate_logical_grid(
            logical_positions,
            valid_mask,
            leading_shape=x3_leading,
            device=device,
            label="causal modal pair",
        )
        x3_mean = self.x3_mean.to(device=device, dtype=dtype)
        y3_mean = self.y3_mean.to(device=device, dtype=dtype)
        x4_mean = self.x4_mean.to(device=device, dtype=dtype)
        y4_mean = self.y4_mean.to(device=device, dtype=dtype)
        stage3 = self.stage3_matrix.to(device=device, dtype=dtype)
        stage4 = self.stage4_torn_base_matrix.to(
            device=device,
            dtype=dtype,
        )
        edge = self.lag_edge_matrices.to(device=device, dtype=dtype)
        centered_x3 = x3 - x3_mean
        y3 = centered_x3 @ stage3.T + y3_mean
        base_y4 = (
            (x4_mean_source_reference_torn_base - x4_mean)
            @ stage4.T
            + y4_mean
        )
        edge_y4 = _causal_logical_lag_row_map(
            centered_x3,
            edge.transpose(-1, -2),
            source_logical_positions=positions,
            source_valid_mask=mask,
            target_logical_positions=positions,
            target_valid_mask=mask,
        )
        return y3, base_y4 + edge_y4


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PreparedCausalModalPair:
    """Validate-once, device-resident factorized and dense pair controls."""

    plan_sha256: str
    boundary_contract: EdgeTornModalPairBoundaryContract
    source_artifact_sha256: str
    direct_jacobian_proof_sha256: str
    device: torch.device
    dtype: torch.dtype
    accounting: CausalModalPairAccounting
    _x3_mean: Tensor
    _y3_mean: Tensor
    _x4_mean: Tensor
    _y4_mean: Tensor
    _R3: Tensor
    _P3: Tensor
    _R4: Tensor
    _P4: Tensor
    _K: Tensor
    _stage3_matrix: Tensor
    _stage4_torn_base_matrix: Tensor
    _lag_edge_matrices: Tensor

    @classmethod
    def from_plan(
        cls,
        plan: CausalModalPairPlan,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedCausalModalPair:
        return cls(plan, device=device, dtype=dtype)

    def __init__(
        self,
        plan: CausalModalPairPlan,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(plan, CausalModalPairPlan):
            raise TypeError("plan must be a CausalModalPairPlan")
        if dtype not in _RUNTIME_DTYPES:
            raise ValueError("dtype must be a supported floating Torch dtype")
        try:
            runtime_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a Torch device") from error
        plan.validate_integrity()

        runtime_tensors = {
            name: _runtime_copy(
                getattr(plan, name),
                device=runtime_device,
                dtype=dtype,
            )
            for name in (
                "x3_mean",
                "y3_mean",
                "x4_mean",
                "y4_mean",
                "R3",
                "P3",
                "R4",
                "P4",
                "K",
            )
        }
        R3 = runtime_tensors["R3"]
        P3 = runtime_tensors["P3"]
        R4 = runtime_tensors["R4"]
        P4 = runtime_tensors["P4"]
        K = runtime_tensors["K"]
        stage3_matrix = (P3 @ R3).contiguous()
        stage4_matrix = (P4 @ R4).contiguous()
        lag_matrices = torch.stack(
            tuple(
                P4 @ K[lag].T @ R3
                for lag in range(plan.lag_count)
            ),
            dim=0,
        ).contiguous()

        object.__setattr__(self, "plan_sha256", plan.artifact_sha256)
        object.__setattr__(
            self,
            "boundary_contract",
            plan.boundary_contract,
        )
        object.__setattr__(
            self,
            "source_artifact_sha256",
            plan.source_artifact_sha256,
        )
        object.__setattr__(
            self,
            "direct_jacobian_proof_sha256",
            plan.direct_jacobian_proof_sha256,
        )
        object.__setattr__(self, "device", runtime_device)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "accounting",
            plan.accounting(
                bytes_per_scalar=torch.empty(
                    (),
                    dtype=dtype,
                ).element_size()
            ),
        )
        for public_name in (
            "x3_mean",
            "y3_mean",
            "x4_mean",
            "y4_mean",
            "R3",
            "P3",
            "R4",
            "P4",
            "K",
        ):
            object.__setattr__(
                self,
                f"_{public_name}",
                runtime_tensors[public_name],
            )
        object.__setattr__(self, "_stage3_matrix", stage3_matrix)
        object.__setattr__(
            self,
            "_stage4_torn_base_matrix",
            stage4_matrix,
        )
        object.__setattr__(self, "_lag_edge_matrices", lag_matrices)

    def _validate_x3(self, x3: Tensor) -> tuple[int, ...]:
        return _validate_runtime_input(
            x3,
            label="x3",
            width=self.accounting.x3_width,
            device=self.device,
            dtype=self.dtype,
        )

    def _validate_x4_reference(
        self,
        x4_reference: Tensor,
    ) -> tuple[int, ...]:
        return _validate_runtime_input(
            x4_reference,
            label="x4_mean_source_reference_torn_base",
            width=self.accounting.x4_width,
            device=self.device,
            dtype=self.dtype,
        )

    def _validate_grid(
        self,
        logical_positions: Tensor,
        valid_mask: Tensor,
        *,
        leading_shape: tuple[int, ...],
        label: str,
    ) -> tuple[Tensor, Tensor]:
        return _validate_logical_grid(
            logical_positions,
            valid_mask,
            leading_shape=leading_shape,
            device=self.device,
            label=label,
        )

    def _factorized_stage3(self, x3: Tensor) -> tuple[Tensor, Tensor]:
        m3 = (x3 - self._x3_mean) @ self._R3.T
        y3 = m3 @ self._P3.T + self._y3_mean
        return y3, m3

    def _factorized_stage4(
        self,
        x4_reference: Tensor,
        m3: Tensor,
        *,
        source_logical_positions: Tensor,
        source_valid_mask: Tensor,
        target_logical_positions: Tensor,
        target_valid_mask: Tensor,
    ) -> Tensor:
        edge = _causal_logical_lag_row_map(
            m3,
            self._K,
            source_logical_positions=source_logical_positions,
            source_valid_mask=source_valid_mask,
            target_logical_positions=target_logical_positions,
            target_valid_mask=target_valid_mask,
        )
        m4 = (x4_reference - self._x4_mean) @ self._R4.T + edge
        return m4 @ self._P4.T + self._y4_mean

    def new_session(self) -> PreparedCausalModalPairSession:
        return PreparedCausalModalPairSession(self)

    def execute_factorized(
        self,
        x3: Tensor,
        *,
        x4_mean_source_reference_torn_base: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Execute through a fresh single-use staged session."""

        session = self.new_session()
        try:
            y3 = session.stage3(
                x3,
                logical_positions=logical_positions,
                valid_mask=valid_mask,
            )
            y4 = session.stage4_from_mean_source_reference_torn_base(
                x4_mean_source_reference_torn_base,
                logical_positions=logical_positions,
                valid_mask=valid_mask,
            )
            return y3, y4
        finally:
            session.close()

    def execute_dense(
        self,
        x3: Tensor,
        *,
        x4_mean_source_reference_torn_base: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Execute the prepared materialized control."""

        leading = self._validate_x3(x3)
        x4_leading = self._validate_x4_reference(
            x4_mean_source_reference_torn_base
        )
        if x4_leading != leading:
            raise ValueError(
                "x4_mean_source_reference_torn_base must share x3's "
                "leading shape"
            )
        positions, mask = self._validate_grid(
            logical_positions,
            valid_mask,
            leading_shape=leading,
            label="causal modal pair",
        )
        centered_x3 = x3 - self._x3_mean
        y3 = centered_x3 @ self._stage3_matrix.T + self._y3_mean
        base_y4 = (
            (x4_mean_source_reference_torn_base - self._x4_mean)
            @ self._stage4_torn_base_matrix.T
            + self._y4_mean
        )
        edge_y4 = _causal_logical_lag_row_map(
            centered_x3,
            self._lag_edge_matrices.transpose(-1, -2),
            source_logical_positions=positions,
            source_valid_mask=mask,
            target_logical_positions=positions,
            target_valid_mask=mask,
        )
        return y3, base_y4 + edge_y4


class PreparedCausalModalPairSession:
    """Single-forward staged state for a prepared factorized pair.

    A session is intentionally single-use unless explicitly reset.  Every
    failed call clears retained modal state and enters ``failed``.  Successful
    stage 4 clears the carry before returning and enters ``complete``.
    """

    __slots__ = (
        "_runtime",
        "_state",
        "_m3",
        "_leading_shape",
        "_logical_positions",
        "_valid_mask",
        "_in_call",
    )

    def __init__(self, runtime: PreparedCausalModalPair) -> None:
        if not isinstance(runtime, PreparedCausalModalPair):
            raise TypeError("runtime must be a PreparedCausalModalPair")
        self._runtime = runtime
        self._state = "ready"
        self._m3: Tensor | None = None
        self._leading_shape: tuple[int, ...] | None = None
        self._logical_positions: Tensor | None = None
        self._valid_mask: Tensor | None = None
        self._in_call = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def has_pending_stage3(self) -> bool:
        return self._state == "stage3_complete" and self._m3 is not None

    @property
    def pending_modal_carry_scalar_count(self) -> int:
        return 0 if self._m3 is None else self._m3.numel()

    @property
    def pending_logical_position_value_count(self) -> int:
        if self._logical_positions is None:
            return 0
        return self._logical_positions.numel()

    @property
    def pending_valid_mask_value_count(self) -> int:
        if self._valid_mask is None:
            return 0
        return self._valid_mask.numel()

    @property
    def pending_carry_storage_bytes(self) -> int:
        values = (self._m3, self._logical_positions, self._valid_mask)
        return sum(
            value.numel() * value.element_size()
            for value in values
            if value is not None
        )

    def _clear(self, state: str) -> None:
        self._m3 = None
        self._leading_shape = None
        self._logical_positions = None
        self._valid_mask = None
        self._state = state

    def _enter(self) -> None:
        if self._in_call:
            raise RuntimeError("causal modal pair session is not reentrant")
        self._in_call = True

    def stage3(
        self,
        x3: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Emit ``y3`` and retain modal state plus its explicit logical grid."""

        self._enter()
        try:
            if self._state != "ready":
                raise RuntimeError(
                    "stage3 requires a fresh or explicitly reset session"
                )
            leading_shape = self._runtime._validate_x3(x3)
            positions, mask = self._runtime._validate_grid(
                logical_positions,
                valid_mask,
                leading_shape=leading_shape,
                label="stage3",
            )
            y3, m3 = self._runtime._factorized_stage3(x3)
            self._m3 = m3
            self._leading_shape = leading_shape
            # Own the grid used to interpret the retained source rows.  This
            # prevents caller mutation between the two stages.
            self._logical_positions = positions.contiguous().clone()
            self._valid_mask = mask.contiguous().clone()
            self._state = "stage3_complete"
            return y3
        except Exception:
            self._clear("failed")
            raise
        finally:
            self._in_call = False

    def stage4_from_mean_source_reference_torn_base(
        self,
        x4_mean_source_reference_torn_base: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Consume the mean-source reference base and clear all staged state."""

        self._enter()
        try:
            if (
                self._state != "stage3_complete"
                or self._m3 is None
                or self._leading_shape is None
                or self._logical_positions is None
                or self._valid_mask is None
            ):
                raise RuntimeError(
                    "stage4 requires one preceding stage3 in this session"
                )
            x4_leading = self._runtime._validate_x4_reference(
                x4_mean_source_reference_torn_base
            )
            if x4_leading != self._leading_shape:
                raise ValueError(
                    "x4_mean_source_reference_torn_base must share "
                    "stage3's leading shape"
                )
            target_positions, target_mask = self._runtime._validate_grid(
                logical_positions,
                valid_mask,
                leading_shape=x4_leading,
                label="stage4",
            )
            if (
                target_positions.dtype != self._logical_positions.dtype
                or not torch.equal(
                    target_positions,
                    self._logical_positions,
                )
                or not torch.equal(target_mask, self._valid_mask)
            ):
                raise ValueError(
                    "stage4 logical_positions and valid_mask must equal "
                    "the stage3 logical grid"
                )
            y4 = self._runtime._factorized_stage4(
                x4_mean_source_reference_torn_base,
                self._m3,
                source_logical_positions=self._logical_positions,
                source_valid_mask=self._valid_mask,
                target_logical_positions=target_positions,
                target_valid_mask=target_mask,
            )
            self._clear("complete")
            return y4
        except Exception:
            self._clear("failed")
            raise
        finally:
            self._in_call = False

    def reset(self) -> None:
        if self._in_call:
            raise RuntimeError(
                "cannot reset a causal modal pair session during execution"
            )
        if self._state == "closed":
            raise RuntimeError("cannot reset a closed causal modal pair session")
        self._clear("ready")

    def abort(self) -> None:
        if self._in_call:
            raise RuntimeError(
                "cannot abort a causal modal pair session during execution"
            )
        if self._state == "closed":
            return
        self._clear("aborted")

    def close(self) -> None:
        if self._in_call:
            raise RuntimeError(
                "cannot close a causal modal pair session during execution"
            )
        self._clear("closed")

    def __enter__(self) -> PreparedCausalModalPairSession:
        if self._state == "closed":
            raise RuntimeError("cannot enter a closed causal modal pair session")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.close()
