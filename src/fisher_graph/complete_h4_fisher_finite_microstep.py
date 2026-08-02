"""Authenticated finite factor-space microsteps for V19 providers.

V20 asks whether V19's first factor-space Adam step merely overshot.  This
module therefore interpolates the *actual* V19 tensors, without changing
their gauge or runtime math::

    L(a)    = L0    + a * (L1    - L0)
    R(a)    = R0    + a * (R1    - R0)
    beta(a) = beta0 + a * (beta1 - beta0)

The three paths isolate the direction factors, the pedal weight/bias, or both.
Signed alpha lies in ``[-1, 1]`` and always uses the same positive Adam
proposal endpoint: negative alpha traverses the affine line in the opposite
direction; it does not authenticate a reflected proposal.  Alpha zero and
positive one use explicit endpoint branches, so selected tensors are bitwise
equal to their relevant endpoint.  The selected serving object is an
ordinary :class:`AutonomousCompleteH4FisherFiniteJointPedalProvider`; V20 adds
no runtime tensor, ABI, or matrix MAC.  A separate authenticated receipt binds
the endpoint artifacts, path, alpha, selected tensor hashes, and experiment
protocol.  Serialization stores that receipt plus only the selected V19
serving state--never endpoint tensors, gradients, Adam moments, logits, or
examples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import (
    _float_tensor,
    _require_sha256,
    _sha256,
    _tensor_sha256,
)
from .complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    _fit_receipt_sha256,
    autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict,
    autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict,
)


__all__ = [
    "FISHER_FINITE_MICROSTEP_PATHS",
    "FisherFiniteMicrostepParameters",
    "FisherFiniteMicrostepReceipt",
    "FisherFiniteMicrostepResult",
    "autonomous_complete_h4_fisher_finite_microstep_from_state_dict",
    "autonomous_complete_h4_fisher_finite_microstep_state_dict",
    "build_autonomous_complete_h4_fisher_finite_microstep",
    "fisher_finite_microstep_selected_tensor_sha256s",
    "interpolate_fisher_finite_microstep_parameters",
]


FISHER_FINITE_MICROSTEP_PATHS = frozenset(
    {"direction_only", "pedal_only", "joint"}
)

_PARAMETER_DOMAIN = b"fisher-graph:complete-h4-fisher-finite-microstep:parameters:v1\0"
_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-fisher-finite-microstep:receipt:v1\0"
_STATE_SCHEMA = "fisher_graph.complete_h4_fisher_finite_microstep_state.v1"
_STATE_KEYS = frozenset(
    {"schema", "format_version", "receipt", "selected_provider_state"}
)
_TENSOR_NAMES = (
    "direction_left",
    "direction_right",
    "pedal_weight",
    "pedal_bias",
)
_RECEIPT_METADATA_KEYS = frozenset(
    {
        "schema",
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "selected_provider_artifact_sha256",
        "parameter_artifact_sha256",
        "microstep_path",
        "alpha",
        "microstep_protocol_sha256",
        "microstep_evidence_sha256",
        "selected_tensor_sha256s",
        "prepared_float_scalar_count",
        "logical_macs_per_token_upper_bound",
        "rank",
        "conditional_rank",
        "runtime_provider_type",
        "endpoint_tensors_gradients_or_optimizer_state_serialized",
        "artifact_sha256",
    }
)


def _path(value: object) -> str:
    if not isinstance(value, str) or value not in FISHER_FINITE_MICROSTEP_PATHS:
        raise ValueError("finite microstep path differs")
    return value


def _alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("finite microstep alpha must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < -1.0 or selected > 1.0:
        raise ValueError("finite microstep alpha must lie in [-1, 1]")
    if selected == 0.0:
        return 0.0
    if selected == 1.0:
        return 1.0
    return selected


def fisher_finite_microstep_selected_tensor_sha256s(
    value: object,
) -> dict[str, str]:
    """Hash the four selected V19 tensors in the receipt's hash domain."""

    return {name: _tensor_sha256(getattr(value, name)) for name in _TENSOR_NAMES}


def _validate_endpoint_pair(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
) -> None:
    if not isinstance(
        base_provider, AutonomousCompleteH4FisherFiniteJointPedalProvider
    ) or not isinstance(
        proposal_provider, AutonomousCompleteH4FisherFiniteJointPedalProvider
    ):
        raise TypeError("finite microstep endpoints must be V19 providers")
    base_provider.validate_integrity()
    proposal_provider.validate_integrity()
    if (
        base_provider.pedal_mode != "conditional"
        or proposal_provider.pedal_mode != "conditional"
        or base_provider.bridge_binding_sha256
        != proposal_provider.bridge_binding_sha256
        or base_provider.parent_provider.artifact_sha256
        != proposal_provider.parent_provider.artifact_sha256
        or base_provider.start_provider_artifact_sha256
        != proposal_provider.start_provider_artifact_sha256
        or base_provider.fit_protocol_sha256
        != proposal_provider.fit_protocol_sha256
        or base_provider.fit_row_count != proposal_provider.fit_row_count
        or base_provider.fit_family_ids != proposal_provider.fit_family_ids
        or base_provider.fit_sequence_sha256s
        != proposal_provider.fit_sequence_sha256s
        or base_provider.coordinate_objective
        != proposal_provider.coordinate_objective
        or base_provider.trust_fraction != proposal_provider.trust_fraction
        or base_provider.rank != proposal_provider.rank
        or base_provider.conditional_rank != proposal_provider.conditional_rank
        or base_provider.prepared_float_scalar_count
        != proposal_provider.prepared_float_scalar_count
        or base_provider.logical_macs_per_token_upper_bound
        != proposal_provider.logical_macs_per_token_upper_bound
    ):
        raise ValueError("finite microstep endpoint lineage or geometry differs")
    for name in ("router_weight", "router_bias", "coordinate_scales"):
        if not torch.equal(getattr(base_provider, name), getattr(proposal_provider, name)):
            raise ValueError("finite microstep endpoint router tensors differ")


@dataclass(frozen=True, slots=True)
class FisherFiniteMicrostepParameters:
    """Immutable selected V19 tensors and their interpolation identity."""

    direction_left: Tensor
    direction_right: Tensor
    pedal_weight: Tensor
    pedal_bias: Tensor
    base_provider_artifact_sha256: str
    proposal_provider_artifact_sha256: str
    microstep_path: str
    alpha: float
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        tensors = {
            "direction_left": _float_tensor(
                self.direction_left,
                label="finite microstep direction_left",
                ndim=2,
            ),
            "direction_right": _float_tensor(
                self.direction_right,
                label="finite microstep direction_right",
                ndim=2,
            ),
            "pedal_weight": _float_tensor(
                self.pedal_weight,
                label="finite microstep pedal_weight",
                ndim=1,
            ),
            "pedal_bias": _float_tensor(
                self.pedal_bias,
                label="finite microstep pedal_bias",
                ndim=1,
            ),
        }
        if (
            tensors["direction_left"].shape[1]
            != tensors["direction_right"].shape[0]
            or tensors["direction_left"].shape[0]
            != 3 * tensors["direction_right"].shape[1]
            or tensors["pedal_weight"].shape != (3,)
            or tensors["pedal_bias"].shape != (1,)
        ):
            raise ValueError("finite microstep parameter geometry differs")
        for name, tensor in tensors.items():
            object.__setattr__(self, name, tensor)
        _require_sha256(
            self.base_provider_artifact_sha256,
            label="finite microstep base provider",
        )
        _require_sha256(
            self.proposal_provider_artifact_sha256,
            label="finite microstep proposal provider",
        )
        object.__setattr__(self, "microstep_path", _path(self.microstep_path))
        object.__setattr__(self, "alpha", _alpha(self.alpha))
        computed = _sha256(_PARAMETER_DOMAIN, self._payload())
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="finite microstep parameters",
            ) != computed:
                raise ValueError("finite microstep parameter artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def rank(self) -> int:
        return int(self.direction_right.shape[1])

    @property
    def conditional_rank(self) -> int:
        return int(self.direction_right.shape[0])

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.fisher_finite_microstep_parameters.v1",
            "base_provider_artifact_sha256": self.base_provider_artifact_sha256,
            "proposal_provider_artifact_sha256": (
                self.proposal_provider_artifact_sha256
            ),
            "microstep_path": self.microstep_path,
            "alpha": self.alpha,
            "selected_tensor_sha256s": (
                fisher_finite_microstep_selected_tensor_sha256s(self)
            ),
            "interpolation_semantics": (
                "factor_space_L_R_and_pedal_logit_parameters"
            ),
        }

    def validate_integrity(self) -> None:
        if _sha256(_PARAMETER_DOMAIN, self._payload()) != self.artifact_sha256:
            raise RuntimeError("finite microstep parameter payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def interpolate_fisher_finite_microstep_parameters(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    microstep_path: str,
    alpha: float,
) -> FisherFiniteMicrostepParameters:
    """Interpolate the exact first-Adam factor/pedal tensors."""

    _validate_endpoint_pair(base_provider, proposal_provider)
    path = _path(microstep_path)
    fraction = _alpha(alpha)

    def interpolate(name: str, *, selected: bool) -> Tensor:
        base = getattr(base_provider, name)
        proposal = getattr(proposal_provider, name)
        if not selected or fraction == 0.0:
            return base
        if fraction == 1.0:
            return proposal
        return (base + fraction * (proposal - base)).contiguous()

    direction_selected = path in {"direction_only", "joint"}
    pedal_selected = path in {"pedal_only", "joint"}
    result = FisherFiniteMicrostepParameters(
        direction_left=interpolate("direction_left", selected=direction_selected),
        direction_right=interpolate(
            "direction_right", selected=direction_selected
        ),
        pedal_weight=interpolate("pedal_weight", selected=pedal_selected),
        pedal_bias=interpolate("pedal_bias", selected=pedal_selected),
        base_provider_artifact_sha256=base_provider.artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider.artifact_sha256,
        microstep_path=path,
        alpha=fraction,
    )
    result.validate_integrity()
    return result


def _materialize_selected_provider(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    parameters: FisherFiniteMicrostepParameters,
    *,
    protocol: str,
    evidence: str,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    if parameters.alpha == 0.0:
        return base_provider
    if parameters.alpha == 1.0 and parameters.microstep_path == "joint":
        return proposal_provider
    receipt = _fit_receipt_sha256(
        parent_provider=base_provider.parent_provider,
        router_weight=base_provider.router_weight,
        router_bias=base_provider.router_bias,
        coordinate_scales=base_provider.coordinate_scales,
        direction_left=parameters.direction_left,
        direction_right=parameters.direction_right,
        pedal_weight=parameters.pedal_weight,
        pedal_bias=parameters.pedal_bias,
        start_provider_artifact_sha256=(
            base_provider.start_provider_artifact_sha256
        ),
        fit_protocol_sha256=protocol,
        fit_evidence_sha256=evidence,
        fit_row_count=base_provider.fit_row_count,
        fit_family_ids=base_provider.fit_family_ids,
        fit_sequence_sha256s=base_provider.fit_sequence_sha256s,
        coordinate_objective=base_provider.coordinate_objective,
        pedal_mode="conditional",
        trust_fraction=base_provider.trust_fraction,
    )
    return AutonomousCompleteH4FisherFiniteJointPedalProvider(
        parent_provider=base_provider.parent_provider,
        router_weight=base_provider.router_weight,
        router_bias=base_provider.router_bias,
        coordinate_scales=base_provider.coordinate_scales,
        direction_left=parameters.direction_left,
        direction_right=parameters.direction_right,
        pedal_weight=parameters.pedal_weight,
        pedal_bias=parameters.pedal_bias,
        start_provider_artifact_sha256=(
            base_provider.start_provider_artifact_sha256
        ),
        fit_protocol_sha256=protocol,
        fit_evidence_sha256=evidence,
        fit_receipt_sha256=receipt,
        trust_fraction=base_provider.trust_fraction,
        fit_row_count=base_provider.fit_row_count,
        fit_family_ids=base_provider.fit_family_ids,
        fit_sequence_sha256s=base_provider.fit_sequence_sha256s,
        coordinate_objective=base_provider.coordinate_objective,
        pedal_mode="conditional",
    )


@dataclass(frozen=True, slots=True)
class FisherFiniteMicrostepReceipt:
    """Scalar/hash-only provenance for one selected V19 provider."""

    base_provider_artifact_sha256: str
    proposal_provider_artifact_sha256: str
    selected_provider_artifact_sha256: str
    parameter_artifact_sha256: str
    microstep_path: str
    alpha: float
    microstep_protocol_sha256: str
    microstep_evidence_sha256: str
    selected_tensor_sha256s: Mapping[str, str]
    prepared_float_scalar_count: int
    logical_macs_per_token_upper_bound: int
    rank: int
    conditional_rank: int
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "base_provider_artifact_sha256",
            "proposal_provider_artifact_sha256",
            "selected_provider_artifact_sha256",
            "parameter_artifact_sha256",
            "microstep_protocol_sha256",
            "microstep_evidence_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        object.__setattr__(self, "microstep_path", _path(self.microstep_path))
        object.__setattr__(self, "alpha", _alpha(self.alpha))
        hashes = dict(self.selected_tensor_sha256s)
        if set(hashes) != set(_TENSOR_NAMES):
            raise ValueError("finite microstep selected tensor hashes differ")
        for name, value in hashes.items():
            _require_sha256(value, label=f"finite microstep selected {name}")
        object.__setattr__(self, "selected_tensor_sha256s", hashes)
        for name in (
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
            "rank",
            "conditional_rank",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"finite microstep {name} must be positive")
        computed = _sha256(_RECEIPT_DOMAIN, self._payload())
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="finite microstep receipt",
            ) != computed:
                raise ValueError("finite microstep receipt artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.fisher_finite_microstep_receipt.v1",
            "base_provider_artifact_sha256": self.base_provider_artifact_sha256,
            "proposal_provider_artifact_sha256": (
                self.proposal_provider_artifact_sha256
            ),
            "selected_provider_artifact_sha256": (
                self.selected_provider_artifact_sha256
            ),
            "parameter_artifact_sha256": self.parameter_artifact_sha256,
            "microstep_path": self.microstep_path,
            "alpha": self.alpha,
            "microstep_protocol_sha256": self.microstep_protocol_sha256,
            "microstep_evidence_sha256": self.microstep_evidence_sha256,
            "selected_tensor_sha256s": dict(self.selected_tensor_sha256s),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "logical_macs_per_token_upper_bound": (
                self.logical_macs_per_token_upper_bound
            ),
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "runtime_provider_type": (
                "AutonomousCompleteH4FisherFiniteJointPedalProvider"
            ),
            "endpoint_tensors_gradients_or_optimizer_state_serialized": False,
        }

    def validate_integrity(self) -> None:
        if _sha256(_RECEIPT_DOMAIN, self._payload()) != self.artifact_sha256:
            raise RuntimeError("finite microstep receipt payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_metadata(cls, value: Mapping[str, object]) -> FisherFiniteMicrostepReceipt:
        if not isinstance(value, Mapping) or set(value) != _RECEIPT_METADATA_KEYS:
            raise ValueError("finite microstep receipt metadata fields differ")
        if (
            value.get("schema")
            != "fisher_graph.fisher_finite_microstep_receipt.v1"
            or value.get("runtime_provider_type")
            != "AutonomousCompleteH4FisherFiniteJointPedalProvider"
            or value.get(
                "endpoint_tensors_gradients_or_optimizer_state_serialized"
            )
            is not False
            or not isinstance(value.get("selected_tensor_sha256s"), Mapping)
        ):
            raise ValueError("finite microstep receipt metadata contract differs")
        return cls(
            base_provider_artifact_sha256=value[  # type: ignore[arg-type]
                "base_provider_artifact_sha256"
            ],
            proposal_provider_artifact_sha256=value[  # type: ignore[arg-type]
                "proposal_provider_artifact_sha256"
            ],
            selected_provider_artifact_sha256=value[  # type: ignore[arg-type]
                "selected_provider_artifact_sha256"
            ],
            parameter_artifact_sha256=value[  # type: ignore[arg-type]
                "parameter_artifact_sha256"
            ],
            microstep_path=value["microstep_path"],  # type: ignore[arg-type]
            alpha=value["alpha"],  # type: ignore[arg-type]
            microstep_protocol_sha256=value[  # type: ignore[arg-type]
                "microstep_protocol_sha256"
            ],
            microstep_evidence_sha256=value[  # type: ignore[arg-type]
                "microstep_evidence_sha256"
            ],
            selected_tensor_sha256s=value[  # type: ignore[arg-type]
                "selected_tensor_sha256s"
            ],
            prepared_float_scalar_count=value[  # type: ignore[arg-type]
                "prepared_float_scalar_count"
            ],
            logical_macs_per_token_upper_bound=value[  # type: ignore[arg-type]
                "logical_macs_per_token_upper_bound"
            ],
            rank=value["rank"],  # type: ignore[arg-type]
            conditional_rank=value["conditional_rank"],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class FisherFiniteMicrostepResult:
    """An unchanged-ABI V19 provider paired with its V20 receipt."""

    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    receipt: FisherFiniteMicrostepReceipt

    def __post_init__(self) -> None:
        if not isinstance(
            self.provider, AutonomousCompleteH4FisherFiniteJointPedalProvider
        ) or not isinstance(self.receipt, FisherFiniteMicrostepReceipt):
            raise TypeError("finite microstep result types differ")
        self.validate_integrity()

    @property
    def artifact_sha256(self) -> str:
        return self.receipt.artifact_sha256

    def validate_integrity(self) -> None:
        self.provider.validate_integrity()
        self.receipt.validate_integrity()
        parameters = FisherFiniteMicrostepParameters(
            direction_left=self.provider.direction_left,
            direction_right=self.provider.direction_right,
            pedal_weight=self.provider.pedal_weight,
            pedal_bias=self.provider.pedal_bias,
            base_provider_artifact_sha256=(
                self.receipt.base_provider_artifact_sha256
            ),
            proposal_provider_artifact_sha256=(
                self.receipt.proposal_provider_artifact_sha256
            ),
            microstep_path=self.receipt.microstep_path,
            alpha=self.receipt.alpha,
        )
        if (
            self.receipt.selected_provider_artifact_sha256
            != self.provider.artifact_sha256
            or self.receipt.parameter_artifact_sha256
            != parameters.artifact_sha256
            or dict(self.receipt.selected_tensor_sha256s)
            != fisher_finite_microstep_selected_tensor_sha256s(self.provider)
            or self.receipt.prepared_float_scalar_count
            != self.provider.prepared_float_scalar_count
            or self.receipt.logical_macs_per_token_upper_bound
            != self.provider.logical_macs_per_token_upper_bound
            or self.receipt.rank != self.provider.rank
            or self.receipt.conditional_rank != self.provider.conditional_rank
        ):
            raise RuntimeError("finite microstep result binding drifted")


def build_autonomous_complete_h4_fisher_finite_microstep(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    microstep_path: str,
    alpha: float,
    microstep_protocol_sha256: str,
    microstep_evidence_sha256: str,
) -> FisherFiniteMicrostepResult:
    """Materialize a V19 provider plus scalar/hash-only V20 provenance."""

    _validate_endpoint_pair(base_provider, proposal_provider)
    protocol = _require_sha256(
        microstep_protocol_sha256,
        label="finite microstep protocol",
    )
    evidence = _require_sha256(
        microstep_evidence_sha256,
        label="finite microstep evidence",
    )
    parameters = interpolate_fisher_finite_microstep_parameters(
        base_provider,
        proposal_provider,
        microstep_path=microstep_path,
        alpha=alpha,
    )
    provider = _materialize_selected_provider(
        base_provider,
        proposal_provider,
        parameters,
        protocol=protocol,
        evidence=evidence,
    )
    if (
        provider.prepared_float_scalar_count
        != base_provider.prepared_float_scalar_count
        or provider.logical_macs_per_token_upper_bound
        != base_provider.logical_macs_per_token_upper_bound
    ):
        raise RuntimeError("finite microstep resources differ from V19")
    receipt = FisherFiniteMicrostepReceipt(
        base_provider_artifact_sha256=base_provider.artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider.artifact_sha256,
        selected_provider_artifact_sha256=provider.artifact_sha256,
        parameter_artifact_sha256=parameters.artifact_sha256,
        microstep_path=parameters.microstep_path,
        alpha=parameters.alpha,
        microstep_protocol_sha256=protocol,
        microstep_evidence_sha256=evidence,
        selected_tensor_sha256s=(
            fisher_finite_microstep_selected_tensor_sha256s(provider)
        ),
        prepared_float_scalar_count=provider.prepared_float_scalar_count,
        logical_macs_per_token_upper_bound=(
            provider.logical_macs_per_token_upper_bound
        ),
        rank=provider.rank,
        conditional_rank=provider.conditional_rank,
    )
    return FisherFiniteMicrostepResult(provider=provider, receipt=receipt)


def autonomous_complete_h4_fisher_finite_microstep_state_dict(
    result: FisherFiniteMicrostepResult,
) -> dict[str, object]:
    """Serialize only the receipt and selected V19 serving provider."""

    if not isinstance(result, FisherFiniteMicrostepResult):
        raise TypeError("finite microstep state requires a result")
    result.validate_integrity()
    return {
        "schema": _STATE_SCHEMA,
        "format_version": 1,
        "receipt": result.receipt.metadata(),
        "selected_provider_state": (
            autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict(
                result.provider
            )
        ),
    }


def autonomous_complete_h4_fisher_finite_microstep_from_state_dict(
    state: Mapping[str, object],
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None = None,
    expected_start_provider_artifact_sha256: str | None = None,
    expected_base_provider_artifact_sha256: str | None = None,
    expected_proposal_provider_artifact_sha256: str | None = None,
) -> FisherFiniteMicrostepResult:
    """Restore a selected V19 provider with strict V20 receipt bindings."""

    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ValueError("finite microstep state fields differ")
    if state.get("schema") != _STATE_SCHEMA or state.get("format_version") != 1:
        raise ValueError("finite microstep state schema differs")
    receipt_state = state.get("receipt")
    provider_state = state.get("selected_provider_state")
    if not isinstance(receipt_state, Mapping) or not isinstance(
        provider_state, Mapping
    ):
        raise ValueError("finite microstep nested state differs")
    receipt = FisherFiniteMicrostepReceipt.from_metadata(receipt_state)
    expected = _require_sha256(
        expected_artifact_sha256,
        label="expected finite microstep receipt",
    )
    if receipt.artifact_sha256 != expected:
        raise ValueError("finite microstep receipt differs from expected")
    for actual, expected_value, label in (
        (
            receipt.base_provider_artifact_sha256,
            expected_base_provider_artifact_sha256,
            "base provider",
        ),
        (
            receipt.proposal_provider_artifact_sha256,
            expected_proposal_provider_artifact_sha256,
            "proposal provider",
        ),
    ):
        if expected_value is not None and actual != _require_sha256(
            expected_value,
            label=f"expected finite microstep {label}",
        ):
            raise ValueError(f"finite microstep {label} differs from expected")
    provider = (
        autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
            provider_state,
            expected_artifact_sha256=(
                receipt.selected_provider_artifact_sha256
            ),
            expected_bridge_binding_sha256=expected_bridge_binding_sha256,
            expected_start_provider_artifact_sha256=(
                expected_start_provider_artifact_sha256
            ),
        )
    )
    return FisherFiniteMicrostepResult(provider=provider, receipt=receipt)
