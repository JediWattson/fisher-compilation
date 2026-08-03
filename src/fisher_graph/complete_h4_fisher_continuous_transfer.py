"""Continuous, Fisher-coordinate transfer between authenticated V19 endpoints.

The V20c provider keeps a compatible base/proposal pair fixed and replaces a
single global microstep scalar with a smooth row-local gain.  The only live
state used to obtain that gain is the base provider's bounded two-coordinate
Fisher router output.  No prompt family, target, logit, gradient, or optimizer
state is a serving input.

For direction features ``F=[c1*h,c2*h,c1*c2*h]``, endpoint factors
``(L0,R0)`` and differences ``(dL,dR)``, a row gain ``g`` is applied exactly
in factor space::

    F (L0 + g dL) (R0 + g dR)
      = F L0 R0 + g(F dL R0 + F L0 dR) + g^2 F dL dR

The expansion matters because ``g`` may differ by row.  A constant gain is
therefore identical to the existing finite-microstep tensor interpolation,
including the quadratic factor interaction.  Pedal logits are interpolated
with the same gain, after which the direction is pointwise bounded at 0.25 of
the parent modal norm and multiplied by the sigmoid pedal.

This is deliberately an experimental provider.  It retains both endpoints
and executes the extra endpoint terms; its metadata accounts for those costs
rather than presenting the object as an inference optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import (
    _float_tensor,
    _require_sha256,
    _sha256,
    _tensor_sha256,
)
from .complete_h4_fisher_conditional_pedal import (
    fisher_xy_pedal_features,
    fisher_xy_pointwise_bounded_direction,
)
from .complete_h4_fisher_finite_joint_pedal import (
    FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    _finite_runtime_tensor,
    fisher_finite_joint_direction_features,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_CONTINUOUS_RESPONSE_LAWS",
    "FISHER_CONTINUOUS_RESPONSE_SOURCES",
    "FISHER_CONTINUOUS_SIGNED_LOG_KAPPA",
    "FISHER_CONTINUOUS_TRANSFER_TRUST_FRACTION",
    "AutonomousCompleteH4FisherContinuousTransferProvider",
    "build_autonomous_complete_h4_fisher_continuous_axis_response",
    "build_autonomous_complete_h4_fisher_continuous_constant_control",
    "build_autonomous_complete_h4_fisher_continuous_transfer",
    "fisher_continuous_bilinear_box_max_abs",
    "fisher_continuous_bilinear_corner_values",
    "fisher_continuous_factor_direction",
    "fisher_continuous_pedal_logit",
    "fisher_continuous_response_features",
    "fisher_continuous_response_gain",
    "fisher_continuous_transfer_modal_terms",
]


FISHER_CONTINUOUS_RESPONSE_LAWS = frozenset({"linear", "signed_log"})
FISHER_CONTINUOUS_RESPONSE_SOURCES = frozenset(
    {"constant", "direct", "tanh_projection"}
)
FISHER_CONTINUOUS_SIGNED_LOG_KAPPA = 9.0
FISHER_CONTINUOUS_TRANSFER_TRUST_FRACTION = (
    FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION
)

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-continuous-transfer:provider:v1\0"
)


def _response_law(value: object) -> str:
    if not isinstance(value, str) or value not in FISHER_CONTINUOUS_RESPONSE_LAWS:
        raise ValueError("continuous-transfer response law differs")
    return value


def _response_source(value: object) -> str:
    if not isinstance(value, str) or value not in FISHER_CONTINUOUS_RESPONSE_SOURCES:
        raise ValueError("continuous-transfer response source differs")
    return value


def _polarity(value: object) -> int:
    if type(value) is not int or value not in {-1, 1}:
        raise ValueError("continuous-transfer polarity must be exactly -1 or 1")
    return value


def _signed_log_kappa(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("continuous-transfer signed-log kappa must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("continuous-transfer signed-log kappa must be positive")
    return result


def fisher_continuous_bilinear_corner_values(
    response_weight: Tensor,
) -> tuple[float, float, float, float]:
    """Return ``w1*c1 + w2*c2 + w12*c1*c2`` at the four box corners.

    The frozen order is ``(-1,-1), (-1,+1), (+1,-1), (+1,+1)``.  A
    multi-affine scalar reaches its extrema on ``[-1,1]^2`` at these corners,
    so the values form a global direct-projection certificate.
    """

    weight = _finite_runtime_tensor(
        response_weight,
        label="continuous-transfer response weight certificate",
        ndim=1,
    ).to(device="cpu")
    if weight.shape != (3,):
        raise ValueError("continuous-transfer response weight geometry differs")
    return tuple(
        float(weight[0] * c1 + weight[1] * c2 + weight[2] * c1 * c2)
        for c1, c2 in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    )


def fisher_continuous_bilinear_box_max_abs(response_weight: Tensor) -> float:
    """Return the exact maximum absolute direct projection on ``[-1,1]^2``."""

    return max(
        abs(value)
        for value in fisher_continuous_bilinear_corner_values(response_weight)
    )


def _validate_endpoint_pair(
    base: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal: AutonomousCompleteH4FisherFiniteJointPedalProvider,
) -> None:
    """Require the same endpoint lineage and geometry as V20 microsteps."""

    if not isinstance(
        base, AutonomousCompleteH4FisherFiniteJointPedalProvider
    ) or not isinstance(proposal, AutonomousCompleteH4FisherFiniteJointPedalProvider):
        raise TypeError("continuous-transfer endpoints must be V19 providers")
    base.validate_integrity()
    proposal.validate_integrity()
    if (
        base.pedal_mode != "conditional"
        or proposal.pedal_mode != "conditional"
        or base.bridge_binding_sha256 != proposal.bridge_binding_sha256
        or base.parent_provider.artifact_sha256
        != proposal.parent_provider.artifact_sha256
        or base.start_provider_artifact_sha256
        != proposal.start_provider_artifact_sha256
        or base.fit_protocol_sha256 != proposal.fit_protocol_sha256
        or base.fit_row_count != proposal.fit_row_count
        or base.fit_family_ids != proposal.fit_family_ids
        or base.fit_sequence_sha256s != proposal.fit_sequence_sha256s
        or base.coordinate_objective != proposal.coordinate_objective
        or base.trust_fraction != proposal.trust_fraction
        or base.rank != proposal.rank
        or base.conditional_rank != proposal.conditional_rank
        or base.prepared_float_scalar_count
        != proposal.prepared_float_scalar_count
        or base.logical_macs_per_token_upper_bound
        != proposal.logical_macs_per_token_upper_bound
    ):
        raise ValueError("continuous-transfer endpoint lineage or geometry differs")
    for name in ("router_weight", "router_bias", "coordinate_scales"):
        if not torch.equal(getattr(base, name), getattr(proposal, name)):
            raise ValueError("continuous-transfer endpoint router tensors differ")


def fisher_continuous_response_features(coordinates: Tensor) -> Tensor:
    """Return differentiable raw ``[c1,c2,c1*c2]`` Fisher features."""

    bounded = _finite_runtime_tensor(
        coordinates,
        label="continuous-transfer bounded coordinates",
        ndim=2,
    )
    if bounded.shape[1] != 2 or bool((bounded.abs() >= 1.0).any()):
        raise ValueError("continuous-transfer coordinate geometry differs")
    c1 = bounded[:, :1]
    c2 = bounded[:, 1:]
    return torch.cat((c1, c2, c1 * c2), dim=1)


def fisher_continuous_response_gain(
    coordinates: Tensor,
    response_weight: Tensor,
    *,
    response_source: str,
    response_law: str,
    polarity: int,
    signed_log_kappa: float = FISHER_CONTINUOUS_SIGNED_LOG_KAPPA,
) -> Tensor:
    """Return a differentiable shared direction/pedal gain.

    ``direct`` permits exact fixed-axis sentinels: weights ``[1,0,0]`` and
    ``[0,1,0]`` produce the raw first and second bounded coordinates.  General
    fitted projections can use ``tanh_projection`` to remain inside ``(-1,1)``.
    ``constant`` ignores the (required-zero) weight and emits exact ``+1`` or
    ``-1`` controls through ``polarity``.
    """

    source = _response_source(response_source)
    law = _response_law(response_law)
    sign = _polarity(polarity)
    kappa = _signed_log_kappa(signed_log_kappa)
    features = fisher_continuous_response_features(coordinates)
    weight = _finite_runtime_tensor(
        response_weight,
        label="continuous-transfer response weight",
        ndim=1,
    ).to(features.device)
    if weight.shape != (3,):
        raise ValueError("continuous-transfer response weight geometry differs")
    projection = features @ weight
    if source == "constant":
        if bool((weight != 0.0).any()):
            raise ValueError("continuous-transfer constant weight must be zero")
        # Keep a zero-gradient connection for callers using the pure fit path.
        z = torch.ones_like(projection) + 0.0 * projection
    elif source == "direct":
        z = projection
        if bool((z.abs() >= 1.0).any()):
            raise ValueError("continuous-transfer direct projection escaped (-1,1)")
    else:
        z = torch.tanh(projection)
    if law == "signed_log":
        z = torch.asinh(kappa * z) / math.asinh(kappa)
    gain = float(sign) * z
    if not bool(torch.isfinite(gain).all()) or bool((gain.abs() > 1.0).any()):
        raise RuntimeError("continuous-transfer response gain became invalid")
    return gain.contiguous()


def fisher_continuous_factor_direction(
    direction_features: Tensor,
    base_left: Tensor,
    base_right: Tensor,
    proposal_left: Tensor,
    proposal_right: Tensor,
    gain: Tensor,
) -> Tensor:
    """Expand exact row-varying factor interpolation, including ``g^2``."""

    features = _finite_runtime_tensor(
        direction_features,
        label="continuous-transfer direction features",
        ndim=2,
    )
    left0 = _finite_runtime_tensor(
        base_left, label="continuous-transfer base left", ndim=2
    ).to(features.device)
    right0 = _finite_runtime_tensor(
        base_right, label="continuous-transfer base right", ndim=2
    ).to(features.device)
    left1 = _finite_runtime_tensor(
        proposal_left, label="continuous-transfer proposal left", ndim=2
    ).to(features.device)
    right1 = _finite_runtime_tensor(
        proposal_right, label="continuous-transfer proposal right", ndim=2
    ).to(features.device)
    row_gain = _finite_runtime_tensor(
        gain, label="continuous-transfer row gain", ndim=1
    ).to(features.device)
    if (
        left0.shape != left1.shape
        or right0.shape != right1.shape
        or features.shape[1] != left0.shape[0]
        or left0.shape[1] != right0.shape[0]
        or row_gain.shape != (features.shape[0],)
        or bool((row_gain.abs() > 1.0).any())
    ):
        raise ValueError("continuous-transfer factor geometry differs")
    delta_left = left1 - left0
    delta_right = right1 - right0
    features_base = features @ left0
    features_delta = features @ delta_left
    base_term = features_base @ right0
    linear_term = features_delta @ right0 + features_base @ delta_right
    quadratic_term = features_delta @ delta_right
    direction = (
        base_term
        + row_gain.unsqueeze(1) * linear_term
        + row_gain.square().unsqueeze(1) * quadratic_term
    )
    if not bool(torch.isfinite(direction).all()):
        raise RuntimeError("continuous-transfer factor direction became nonfinite")
    return direction.contiguous()


def fisher_continuous_pedal_logit(
    coordinates: Tensor,
    base_weight: Tensor,
    base_bias: Tensor,
    proposal_weight: Tensor,
    proposal_bias: Tensor,
    gain: Tensor,
) -> Tensor:
    """Return exact row-wise interpolation of endpoint sigmoid logits."""

    features = fisher_xy_pedal_features(coordinates)
    weight0 = _finite_runtime_tensor(
        base_weight, label="continuous-transfer base pedal weight", ndim=1
    ).to(features.device)
    bias0 = _finite_runtime_tensor(
        base_bias, label="continuous-transfer base pedal bias", ndim=1
    ).to(features.device)
    weight1 = _finite_runtime_tensor(
        proposal_weight,
        label="continuous-transfer proposal pedal weight",
        ndim=1,
    ).to(features.device)
    bias1 = _finite_runtime_tensor(
        proposal_bias, label="continuous-transfer proposal pedal bias", ndim=1
    ).to(features.device)
    row_gain = _finite_runtime_tensor(
        gain, label="continuous-transfer pedal gain", ndim=1
    ).to(features.device)
    if (
        weight0.shape != (3,)
        or weight1.shape != (3,)
        or bias0.shape != (1,)
        or bias1.shape != (1,)
        or row_gain.shape != (features.shape[0],)
        or bool((row_gain.abs() > 1.0).any())
    ):
        raise ValueError("continuous-transfer pedal geometry differs")
    base_logit = features @ weight0 + bias0[0]
    logit_delta = features @ (weight1 - weight0) + (bias1[0] - bias0[0])
    result = base_logit + row_gain * logit_delta
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("continuous-transfer pedal logit became nonfinite")
    return result.contiguous()


def fisher_continuous_transfer_modal_terms(
    parent_modal: Tensor,
    coordinates: Tensor,
    base_direction_left: Tensor,
    base_direction_right: Tensor,
    proposal_direction_left: Tensor,
    proposal_direction_right: Tensor,
    base_pedal_weight: Tensor,
    base_pedal_bias: Tensor,
    proposal_pedal_weight: Tensor,
    proposal_pedal_bias: Tensor,
    response_weight: Tensor,
    *,
    response_source: str,
    response_law: str,
    polarity: int,
    signed_log_kappa: float = FISHER_CONTINUOUS_SIGNED_LOG_KAPPA,
    trust_fraction: float = FISHER_CONTINUOUS_TRANSFER_TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return differentiable ``(gain,q,b,logit,pedal,delta)`` terms."""

    parent = _finite_runtime_tensor(
        parent_modal, label="continuous-transfer parent modal", ndim=2
    )
    bounded_coordinates = _finite_runtime_tensor(
        coordinates, label="continuous-transfer coordinates", ndim=2
    ).to(parent.device)
    if bounded_coordinates.shape != (parent.shape[0], 2):
        raise ValueError("continuous-transfer parent/coordinate rows differ")
    gain = fisher_continuous_response_gain(
        bounded_coordinates,
        response_weight,
        response_source=response_source,
        response_law=response_law,
        polarity=polarity,
        signed_log_kappa=signed_log_kappa,
    )
    features = fisher_finite_joint_direction_features(parent, bounded_coordinates)
    direction = fisher_continuous_factor_direction(
        features,
        base_direction_left,
        base_direction_right,
        proposal_direction_left,
        proposal_direction_right,
        gain,
    )
    bounded = fisher_xy_pointwise_bounded_direction(
        parent,
        direction,
        trust_fraction=trust_fraction,
    )
    logit = fisher_continuous_pedal_logit(
        bounded_coordinates,
        base_pedal_weight,
        base_pedal_bias,
        proposal_pedal_weight,
        proposal_pedal_bias,
        gain,
    )
    pedal = torch.sigmoid(logit)
    delta = pedal.unsqueeze(1) * bounded
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("continuous-transfer modal delta became nonfinite")
    return gain, direction, bounded, logit, pedal, delta.contiguous()


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherContinuousTransferProvider(
    Gemma3L3L4CorrectionProvider
):
    """Immutable runtime provider for a smooth V19 endpoint transfer."""

    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    response_weight: Tensor
    response_source: str
    response_law: str
    polarity: int
    signed_log_kappa: float
    transfer_protocol_sha256: str
    transfer_evidence_sha256: str
    trust_fraction: float = FISHER_CONTINUOUS_TRANSFER_TRUST_FRACTION
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        weight = _float_tensor(
            self.response_weight,
            label="continuous-transfer response_weight",
            ndim=1,
        )
        if weight.shape != (3,):
            raise ValueError("continuous-transfer response weight geometry differs")
        source = _response_source(self.response_source)
        if source == "constant" and bool((weight != 0.0).any()):
            raise ValueError("continuous-transfer constant weight must be zero")
        if (
            source == "direct"
            and fisher_continuous_bilinear_box_max_abs(weight) > 1.0
        ):
            raise ValueError(
                "continuous-transfer direct weight lacks a global [-1,1] box certificate"
            )
        object.__setattr__(self, "response_weight", weight)
        object.__setattr__(self, "response_source", source)
        object.__setattr__(self, "response_law", _response_law(self.response_law))
        object.__setattr__(self, "polarity", _polarity(self.polarity))
        object.__setattr__(
            self,
            "signed_log_kappa",
            _signed_log_kappa(self.signed_log_kappa),
        )
        for name in ("transfer_protocol_sha256", "transfer_evidence_sha256"):
            _require_sha256(getattr(self, name), label=name)
        if self.trust_fraction != FISHER_CONTINUOUS_TRANSFER_TRUST_FRACTION:
            raise ValueError("continuous-transfer trust fraction is frozen at 0.25")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="continuous-transfer provider"
            ) != computed:
                raise ValueError("continuous-transfer provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def parent_provider(self):
        return self.base_provider.parent_provider

    @property
    def bridge_binding_sha256(self) -> str:
        return self.base_provider.bridge_binding_sha256

    @property
    def rank(self) -> int:
        return self.base_provider.rank

    @property
    def conditional_rank(self) -> int:
        return self.base_provider.conditional_rank

    @property
    def incremental_prepared_float_scalar_count(self) -> int:
        # Retain both complete endpoint increments (including the otherwise
        # equal router tensors) and three response weights.  The common parent
        # artifact is counted once below.
        return int(
            self.base_provider.incremental_prepared_float_scalar_count
            + self.proposal_provider.incremental_prepared_float_scalar_count
            + 3
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.parent_provider.prepared_float_scalar_count
            + self.incremental_prepared_float_scalar_count
        )

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        # Router + response projection + two left-factor products + four
        # right-factor products + two pedal logits.
        return int(
            2 * self.rank
            + 3
            + 10 * self.rank * self.conditional_rank
            + 6
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            self.parent_provider.logical_macs_per_token_upper_bound
            + self.incremental_logical_macs_per_token_upper_bound
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.autonomous_complete_h4_fisher_continuous_transfer_provider.v1"
            ),
            "site": self.site,
            "write_scope": self.write_scope,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "parent_provider_artifact_sha256": self.parent_provider.artifact_sha256,
            "base_provider_artifact_sha256": self.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": (
                self.proposal_provider.artifact_sha256
            ),
            "start_provider_artifact_sha256": (
                self.base_provider.start_provider_artifact_sha256
            ),
            "transfer_protocol_sha256": self.transfer_protocol_sha256,
            "transfer_evidence_sha256": self.transfer_evidence_sha256,
            "response_weight_sha256": _tensor_sha256(self.response_weight),
            "response_source": self.response_source,
            "response_law": self.response_law,
            "polarity": self.polarity,
            "signed_log_kappa": self.signed_log_kappa,
            "trust_fraction": self.trust_fraction,
            "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
            "runtime_forbidden_inputs": (
                "native_h4",
                "targets",
                "logits",
                "gradients",
                "family_ids",
                "fit_examples",
                "optimizer_state",
            ),
            "response_semantics": (
                "shared_signed_coordinate_gain_for_exact_factor_and_pedal_logit_transfer"
            ),
            "factor_semantics": (
                "FL0R0_plus_g_FdLR0_plus_FL0dR_plus_g_squared_FdLdR"
            ),
            "bounded_direction_semantics": (
                "pointwise_q_norm_at_most_0.25_parent_modal_norm"
            ),
            "experimental_serving_status": (
                "analysis_only_retains_two_endpoints_and_executes_extra_endpoint_terms"
            ),
            "prepared_payload_deduplication_semantics": (
                "common_parent_artifact_once_both_complete_endpoint_increments_retained"
            ),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("continuous-transfer provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "incremental_runtime_parameter_bytes_float64": (
                self.incremental_prepared_float_scalar_count * 8
            ),
            "runtime_parameter_bytes_float64": self.prepared_float_scalar_count * 8,
            "incremental_logical_macs_per_token_upper_bound": (
                self.incremental_logical_macs_per_token_upper_bound
            ),
            "logical_macs_per_token_upper_bound": (
                self.logical_macs_per_token_upper_bound
            ),
            "logical_macs_accounting_scope": (
                "experimental_dense_upper_bound_includes_both_endpoint_factor_paths_"
                "and_response_projection_excludes_norms_asinh_tanh_sigmoid_and_hashes"
            ),
            "runtime_state_float_scalars_per_sequence": 0,
            "pointwise_trust_certificate_scope": (
                "emitted_modal_amplitude_not_full_nonlinear_jacobian_or_lipschitz"
            ),
        }

    def bounded_coordinates(self, parent_modal: Tensor) -> Tensor:
        return self.base_provider.bounded_coordinates(parent_modal)

    def response_gain(self, coordinates: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(coordinates, Tensor)
            or coordinates.ndim < 2
            or coordinates.shape[-1] != 2
            or not coordinates.is_floating_point()
            or not bool(torch.isfinite(coordinates).all())
            or bool((coordinates.abs() >= 1.0).any())
        ):
            raise ValueError("continuous-transfer response coordinates differ")
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_continuous_response_gain(
            flat,
            self.response_weight.to(flat.device),
            response_source=self.response_source,
            response_law=self.response_law,
            polarity=self.polarity,
            signed_log_kappa=self.signed_log_kappa,
        )
        return gain.reshape(original_shape).contiguous()

    def terms_from_parent(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Expose pure runtime terms for fit, replay, and trust auditing."""

        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("continuous-transfer parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        original_shape = parent.shape
        flat_parent = parent.reshape(-1, self.rank)
        flat_coordinates = bounded.reshape(-1, 2)
        terms = fisher_continuous_transfer_modal_terms(
            flat_parent,
            flat_coordinates,
            self.base_provider.direction_left.to(parent.device),
            self.base_provider.direction_right.to(parent.device),
            self.proposal_provider.direction_left.to(parent.device),
            self.proposal_provider.direction_right.to(parent.device),
            self.base_provider.pedal_weight.to(parent.device),
            self.base_provider.pedal_bias.to(parent.device),
            self.proposal_provider.pedal_weight.to(parent.device),
            self.proposal_provider.pedal_bias.to(parent.device),
            self.response_weight.to(parent.device),
            response_source=self.response_source,
            response_law=self.response_law,
            polarity=self.polarity,
            signed_log_kappa=self.signed_log_kappa,
            trust_fraction=self.trust_fraction,
        )
        gain, direction, bounded_direction, logit, pedal, delta = terms
        leading = original_shape[:-1]
        return (
            gain.reshape(leading).contiguous(),
            direction.reshape(original_shape).contiguous(),
            bounded_direction.reshape(original_shape).contiguous(),
            logit.reshape(leading).contiguous(),
            pedal.reshape(leading).contiguous(),
            delta.reshape(original_shape).contiguous(),
        )

    def unbounded_direction(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> Tensor:
        return self.terms_from_parent(parent_modal, coordinates)[1]

    def pedal_logits(self, coordinates: Tensor) -> Tensor:
        """Return row-wise interpolated pedal logits from Fisher coordinates."""

        gain = self.response_gain(coordinates)
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2).to(dtype=torch.float64)
        logit = fisher_continuous_pedal_logit(
            flat,
            self.base_provider.pedal_weight.to(flat.device),
            self.base_provider.pedal_bias.to(flat.device),
            self.proposal_provider.pedal_weight.to(flat.device),
            self.proposal_provider.pedal_bias.to(flat.device),
            gain.reshape(-1),
        )
        return logit.reshape(original_shape).contiguous()

    def pedal_values(self, coordinates: Tensor) -> Tensor:
        return torch.sigmoid(self.pedal_logits(coordinates)).contiguous()

    def _modal_terms(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        parent = self.parent_provider.modal_correction(prefix, realized_state)
        coordinates = self.bounded_coordinates(parent)
        gain, direction, bounded, logit, pedal, delta = self.terms_from_parent(
            parent, coordinates
        )
        return parent, coordinates, gain, direction, bounded, logit, pedal, delta

    def modal_correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        prefix.validate_integrity()
        prefix_sha = prefix.artifact_sha256
        realized_sha = _tensor_sha256(realized_state)
        parent, _coordinates, _gain, _direction, _bounded, _logit, _pedal, delta = (
            self._modal_terms(prefix, realized_state)
        )
        modal = parent + delta
        support = prefix.complete_h4_causal_support_mask().to(modal.device)
        modal = modal.masked_fill((~support).unsqueeze(-1), 0.0)
        if (
            prefix.artifact_sha256 != prefix_sha
            or _tensor_sha256(realized_state) != realized_sha
        ):
            raise RuntimeError("continuous-transfer provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("continuous-transfer modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("continuous-transfer modal correction escaped support")
        self.validate_integrity()
        prefix.validate_integrity()
        return modal.contiguous()

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        return self.parent_provider.decode_modal(
            prefix,
            self.modal_correction(prefix, realized_state),
            like=realized_state,
        )


def build_autonomous_complete_h4_fisher_continuous_transfer(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    response_law: str,
    response_source: str,
    response_weight: Tensor,
    polarity: int,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
    signed_log_kappa: float = FISHER_CONTINUOUS_SIGNED_LOG_KAPPA,
) -> AutonomousCompleteH4FisherContinuousTransferProvider:
    """Build and hash-bind one continuous transfer provider directly."""

    return AutonomousCompleteH4FisherContinuousTransferProvider(
        base_provider=base_provider,
        proposal_provider=proposal_provider,
        response_weight=response_weight,
        response_source=response_source,
        response_law=response_law,
        polarity=polarity,
        signed_log_kappa=signed_log_kappa,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )


def build_autonomous_complete_h4_fisher_continuous_constant_control(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    alpha: int,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherContinuousTransferProvider:
    """Build an exact constant ``alpha in {-1,+1}`` endpoint/mirror control."""

    return build_autonomous_complete_h4_fisher_continuous_transfer(
        base_provider,
        proposal_provider,
        response_law="linear",
        response_source="constant",
        response_weight=torch.zeros(3, dtype=torch.float64),
        polarity=_polarity(alpha),
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )


def build_autonomous_complete_h4_fisher_continuous_axis_response(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    coordinate_index: int,
    response_law: str,
    polarity: int,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
    signed_log_kappa: float = FISHER_CONTINUOUS_SIGNED_LOG_KAPPA,
) -> AutonomousCompleteH4FisherContinuousTransferProvider:
    """Build an exact raw-coordinate linear or signed-log sentinel."""

    if type(coordinate_index) is not int or coordinate_index not in {0, 1}:
        raise ValueError("continuous-transfer coordinate index must be 0 or 1")
    weight = torch.zeros(3, dtype=torch.float64)
    weight[coordinate_index] = 1.0
    return build_autonomous_complete_h4_fisher_continuous_transfer(
        base_provider,
        proposal_provider,
        response_law=response_law,
        response_source="direct",
        response_weight=weight,
        polarity=polarity,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
        signed_log_kappa=signed_log_kappa,
    )
