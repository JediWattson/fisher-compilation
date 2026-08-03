"""Smooth signed-amplitude routing between authenticated V19 endpoints.

V20f keeps the complete-H4 endpoint pair and pointwise trust certificate used
by :mod:`complete_h4_fisher_continuous_transfer`, but factorizes the response
into a fixed amplitude envelope and a learned soft polarity field.  For the
base provider's bounded Fisher coordinates ``c=(c1,c2)`` the frozen formula is

``phi(c) = [1, c1, c2, c1*c2]``

``envelope(c) = asinh(9*c2) / asinh(9)``

``polarity(c) = tanh(eta @ phi(c))``

``gain(c) = envelope(c) * polarity(c)``.

There is no data-dependent routing branch.  Both factors are in ``[-1,1]``
for every coordinate in the closed box, so ``abs(gain) <= 1`` is an analytic
global certificate rather than an empirical range check.  Only the four
float64 values in ``eta`` are new fitted state.  Family IDs, prompts, targets,
logits, gradients, and optimizer state are not runtime inputs.
Accepted coefficients are limited to ``float64_max / 8`` in absolute value,
which leaves factor-two accumulation headroom for the four-term router dot.

This remains an analysis provider: it retains both endpoint increments and
executes the same expanded factor interpolation as V20c.  Its accounting
therefore does not claim an inference optimization.
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
    fisher_xy_pointwise_bounded_direction,
)
from .complete_h4_fisher_continuous_transfer import (
    AutonomousCompleteH4FisherContinuousTransferProvider,
    _validate_endpoint_pair,
    build_autonomous_complete_h4_fisher_continuous_axis_response,
    fisher_continuous_factor_direction,
    fisher_continuous_pedal_logit,
)
from .complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
    _finite_runtime_tensor,
    fisher_finite_joint_direction_features,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)


__all__ = [
    "FISHER_SOFT_POLARITY_ETA_COUNT",
    "FISHER_SOFT_POLARITY_ETA_MAX_ABS",
    "FISHER_SOFT_POLARITY_FEATURE_NAMES",
    "FISHER_SOFT_POLARITY_SIGNED_LOG_KAPPA",
    "FISHER_SOFT_POLARITY_TRUST_FRACTION",
    "AutonomousCompleteH4FisherSoftPolarityProvider",
    "build_autonomous_complete_h4_fisher_soft_polarity",
    "build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control",
    "build_autonomous_complete_h4_fisher_soft_polarity_zero_control",
    "fisher_soft_polarity_box_certificate",
    "fisher_soft_polarity_constant_tensor_sha256s",
    "fisher_soft_polarity_envelope",
    "fisher_soft_polarity_features",
    "fisher_soft_polarity_gain",
    "fisher_soft_polarity_modal_terms",
    "fisher_soft_polarity_value",
]


_SOFT_POLARITY_ETA_COUNT = 4
_SOFT_POLARITY_FEATURE_NAMES = ("one", "c1", "c2", "c1_times_c2")
_SOFT_POLARITY_SIGNED_LOG_KAPPA = 9.0
_SOFT_POLARITY_TRUST_FRACTION = (
    FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION
)
# Four bounded features enter the router dot product.  Keeping every
# coefficient below max/8 leaves a factor-of-two accumulation margin even at
# a box corner, so accepted finite eta values cannot create an inf-minus-inf
# cancellation in float64.
_SOFT_POLARITY_ETA_MAX_ABS = float(torch.finfo(torch.float64).max / 8.0)

# Public constants are compatibility aliases only.  Runtime arithmetic,
# validation, metadata, and accounting always use the private authorities
# above so rebinding an exported name cannot alter an authenticated provider.
FISHER_SOFT_POLARITY_ETA_COUNT = _SOFT_POLARITY_ETA_COUNT
FISHER_SOFT_POLARITY_ETA_MAX_ABS = _SOFT_POLARITY_ETA_MAX_ABS
FISHER_SOFT_POLARITY_FEATURE_NAMES = _SOFT_POLARITY_FEATURE_NAMES
FISHER_SOFT_POLARITY_SIGNED_LOG_KAPPA = _SOFT_POLARITY_SIGNED_LOG_KAPPA
FISHER_SOFT_POLARITY_TRUST_FRACTION = _SOFT_POLARITY_TRUST_FRACTION

_H4_SITE = "layer.4.output"
_PROVIDER_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity:provider:v1\0"
)
_CONSTANT_DOMAIN = (
    b"fisher-graph:autonomous-complete-h4-fisher-soft-polarity:constant:v1\0"
)
_FEATURE_EXPONENTS = torch.tensor(
    ((0, 0), (1, 0), (0, 1), (1, 1)), dtype=torch.int64
)
_ENVELOPE_AXIS = torch.tensor((0.0, 1.0), dtype=torch.float64)
_SIGNED_LOG_KAPPA_TENSOR = torch.tensor(
    (_SOFT_POLARITY_SIGNED_LOG_KAPPA,), dtype=torch.float64
)
_CONSTANT_TENSOR_SHA256S = {
    "feature_exponents": _tensor_sha256(_FEATURE_EXPONENTS),
    "envelope_axis": _tensor_sha256(_ENVELOPE_AXIS),
    "signed_log_kappa": _tensor_sha256(_SIGNED_LOG_KAPPA_TENSOR),
}
_CONSTANT_BUNDLE_SHA256 = _sha256(
    _CONSTANT_DOMAIN,
    {
        "feature_names": _SOFT_POLARITY_FEATURE_NAMES,
        "eta_count": _SOFT_POLARITY_ETA_COUNT,
        "eta_max_abs": _SOFT_POLARITY_ETA_MAX_ABS,
        "trust_fraction": _SOFT_POLARITY_TRUST_FRACTION,
        "constant_tensor_sha256s": _CONSTANT_TENSOR_SHA256S,
        "formula": "asinh_9c2_over_asinh_9_times_tanh_eta_dot_1_c1_c2_c1c2",
    },
)


def fisher_soft_polarity_constant_tensor_sha256s() -> dict[str, str]:
    """Return the immutable hashes of every frozen formula tensor."""

    return dict(_CONSTANT_TENSOR_SHA256S)


def _validate_constant_tensors() -> None:
    observed = {
        "feature_exponents": _tensor_sha256(_FEATURE_EXPONENTS),
        "envelope_axis": _tensor_sha256(_ENVELOPE_AXIS),
        "signed_log_kappa": _tensor_sha256(_SIGNED_LOG_KAPPA_TENSOR),
    }
    if observed != _CONSTANT_TENSOR_SHA256S:
        raise RuntimeError("soft-polarity frozen formula tensors drifted")


def _authenticated_signed_log_kappa() -> float:
    """Return the hash-checked private kappa used by every runtime surface."""

    _validate_constant_tensors()
    return float(_SIGNED_LOG_KAPPA_TENSOR[0])


def _coordinates(value: object, *, label: str) -> Tensor:
    selected = _finite_runtime_tensor(value, label=label, ndim=2)
    if selected.shape[1] != 2 or bool((selected.abs() > 1.0).any()):
        raise ValueError(f"{label} must have shape [N,2] inside [-1,1]")
    return selected


def _eta(value: object, *, detach: bool, label: str) -> Tensor:
    if detach:
        selected = _float_tensor(value, label=label, ndim=1)
    else:
        selected = _finite_runtime_tensor(value, label=label, ndim=1)
    if selected.shape != (_SOFT_POLARITY_ETA_COUNT,):
        raise ValueError(f"{label} must contain exactly four float64 values")
    if bool((selected.abs() > _SOFT_POLARITY_ETA_MAX_ABS).any()):
        raise ValueError(
            f"{label} exceeds the certified float64 numerical magnitude limit"
        )
    return selected


def fisher_soft_polarity_features(coordinates: Tensor) -> Tensor:
    """Return differentiable ``[1,c1,c2,c1*c2]`` router features."""

    bounded = _coordinates(coordinates, label="soft-polarity coordinates")
    c1 = bounded[:, :1]
    c2 = bounded[:, 1:]
    return torch.cat((torch.ones_like(c1), c1, c2, c1 * c2), dim=1).contiguous()


def fisher_soft_polarity_envelope(coordinates: Tensor) -> Tensor:
    """Return the fixed signed-log second-coordinate amplitude envelope."""

    bounded = _coordinates(coordinates, label="soft-polarity coordinates")
    kappa = _authenticated_signed_log_kappa()
    result = torch.asinh(kappa * bounded[:, 1])
    result = result / math.asinh(kappa)
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("soft-polarity envelope violated its analytic bound")
    return result.contiguous()


def fisher_soft_polarity_value(coordinates: Tensor, eta: Tensor) -> Tensor:
    """Return the smooth ``tanh(eta dot phi)`` polarity/amplitude factor."""

    features = fisher_soft_polarity_features(coordinates)
    weight = _eta(eta, detach=False, label="soft-polarity eta").to(features.device)
    result = torch.tanh(features @ weight)
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("soft-polarity value violated its analytic bound")
    return result.contiguous()


def fisher_soft_polarity_gain(coordinates: Tensor, eta: Tensor) -> Tensor:
    """Return the frozen continuous gain, analytically bounded by one."""

    envelope = fisher_soft_polarity_envelope(coordinates)
    polarity = fisher_soft_polarity_value(coordinates, eta)
    result = envelope * polarity
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0).any()):
        raise RuntimeError("soft-polarity gain violated its analytic box certificate")
    return result.contiguous()


def fisher_soft_polarity_box_certificate(eta: Tensor) -> dict[str, object]:
    """Return the global certificate for any accepted finite four-vector.

    The certificate is independent of fitted coefficient magnitude: ``tanh``
    bounds the learned factor while the normalized ``asinh`` bounds the fixed
    envelope on the complete coordinate box.  The accepted eta magnitude
    leaves float64 accumulation headroom for all four bounded router features,
    so the projection is numerically finite before applying ``tanh``.
    """

    _validate_constant_tensors()
    selected = _eta(eta, detach=True, label="soft-polarity certificate eta")
    return {
        "schema": "fisher_graph.fisher_soft_polarity_box_certificate.v1",
        "coordinate_box": ((-1.0, 1.0), (-1.0, 1.0)),
        "eta_count": _SOFT_POLARITY_ETA_COUNT,
        "eta_max_abs": _SOFT_POLARITY_ETA_MAX_ABS,
        "router_projection_max_abs_upper_bound": (
            _SOFT_POLARITY_ETA_COUNT * _SOFT_POLARITY_ETA_MAX_ABS
        ),
        "envelope_max_abs": 1.0,
        "polarity_max_abs": 1.0,
        "gain_max_abs": 1.0,
        "proof": "abs_normalized_asinh_at_most_one_times_abs_tanh_at_most_one",
        "numerical_totality_proof": (
            "four_bounded_features_times_eta_max_over_eight_leaves_"
            "factor_two_float64_accumulation_headroom"
        ),
        "eta_sha256": _tensor_sha256(selected),
        "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
    }


def fisher_soft_polarity_modal_terms(
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
    eta: Tensor,
    *,
    trust_fraction: float = _SOFT_POLARITY_TRUST_FRACTION,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return differentiable ``(gain,q,b,logit,pedal,delta)`` terms."""

    parent = _finite_runtime_tensor(
        parent_modal, label="soft-polarity parent modal", ndim=2
    )
    bounded_coordinates = _coordinates(
        coordinates, label="soft-polarity coordinates"
    ).to(parent.device)
    if bounded_coordinates.shape != (parent.shape[0], 2):
        raise ValueError("soft-polarity parent and coordinate rows differ")
    if trust_fraction != _SOFT_POLARITY_TRUST_FRACTION:
        raise ValueError("soft-polarity trust fraction is frozen at 0.25")
    gain = fisher_soft_polarity_gain(
        bounded_coordinates, _eta(eta, detach=False, label="soft-polarity eta")
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
        raise RuntimeError("soft-polarity modal delta became nonfinite")
    return gain, direction, bounded, logit, pedal, delta.contiguous()


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4FisherSoftPolarityProvider(
    Gemma3L3L4CorrectionProvider
):
    """Immutable four-scalar soft-polarity provider over a V19 endpoint pair."""

    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    eta: Tensor
    transfer_protocol_sha256: str
    transfer_evidence_sha256: str
    trust_fraction: float = _SOFT_POLARITY_TRUST_FRACTION
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_constant_tensors()
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        selected_eta = _eta(self.eta, detach=True, label="soft-polarity eta")
        object.__setattr__(self, "eta", selected_eta)
        for name in ("transfer_protocol_sha256", "transfer_evidence_sha256"):
            _require_sha256(getattr(self, name), label=name)
        if self.trust_fraction != _SOFT_POLARITY_TRUST_FRACTION:
            raise ValueError("soft-polarity trust fraction is frozen at 0.25")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256, label="soft-polarity provider"
            ) != computed:
                raise ValueError("soft-polarity provider artifact hash mismatch")
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
        return int(
            self.base_provider.incremental_prepared_float_scalar_count
            + self.proposal_provider.incremental_prepared_float_scalar_count
            + _SOFT_POLARITY_ETA_COUNT
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return (
            self.parent_provider.prepared_float_scalar_count
            + self.incremental_prepared_float_scalar_count
        )

    @property
    def incremental_logical_macs_per_token_upper_bound(self) -> int:
        # Same endpoint-factor and pedal accounting as V20c.  The learned
        # response projection is now a four-term dot product instead of three.
        return int(
            2 * self.rank
            + _SOFT_POLARITY_ETA_COUNT
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
                "fisher_graph.autonomous_complete_h4_fisher_soft_polarity_provider.v1"
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
            "eta_sha256": _tensor_sha256(self.eta),
            "eta_float64_scalar_count": _SOFT_POLARITY_ETA_COUNT,
            "eta_max_abs": _SOFT_POLARITY_ETA_MAX_ABS,
            "constant_tensor_sha256s": dict(_CONSTANT_TENSOR_SHA256S),
            "constant_bundle_sha256": _CONSTANT_BUNDLE_SHA256,
            "feature_names": _SOFT_POLARITY_FEATURE_NAMES,
            "signed_log_kappa": _authenticated_signed_log_kappa(),
            "trust_fraction": self.trust_fraction,
            "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
            "runtime_forbidden_inputs": (
                "native_h4",
                "targets",
                "logits",
                "gradients",
                "family_ids",
                "prompt_text",
                "token_ids",
                "fit_examples",
                "optimizer_state",
            ),
            "gain_formula": (
                "asinh_9c2_over_asinh_9_times_tanh_eta_dot_1_c1_c2_c1c2"
            ),
            "routing_control_flow": "none_validation_guards_only",
            "global_gain_certificate": "absolute_gain_at_most_one",
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
        _validate_constant_tensors()
        _validate_endpoint_pair(self.base_provider, self.proposal_provider)
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("soft-polarity provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "conditional_rank": self.conditional_rank,
            "box_certificate": fisher_soft_polarity_box_certificate(self.eta),
            "incremental_prepared_float_scalar_count": (
                self.incremental_prepared_float_scalar_count
            ),
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "soft_polarity_fitted_float_scalar_count": (
                _SOFT_POLARITY_ETA_COUNT
            ),
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
            "soft_polarity_router_dot_macs_per_token": 4,
            "soft_polarity_elementwise_scalar_arithmetic_per_token": 4,
            "soft_polarity_nonlinear_scalar_ops_per_token": 2,
            "soft_polarity_elementwise_scope": (
                "c1c2_product_kappa_scale_asinh_normalization_and_envelope_polarity_product"
            ),
            "soft_polarity_nonlinear_scope": "one_asinh_and_one_tanh",
            "logical_macs_accounting_scope": (
                "experimental_dense_upper_bound_includes_both_endpoint_factor_paths_"
                "four_term_router_dot_and_two_pedal_logits_elementwise_and_nonlinear_"
                "operations_reported_separately"
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
            or bool((coordinates.abs() > 1.0).any())
        ):
            raise ValueError("soft-polarity response coordinates differ")
        original_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 2)
        gain = fisher_soft_polarity_gain(flat, self.eta.to(flat.device))
        return gain.reshape(original_shape).contiguous()

    def terms_from_parent(
        self,
        parent_modal: Tensor,
        coordinates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Expose pure runtime terms for fitting, replay, and trust auditing."""

        self.validate_integrity()
        if (
            not isinstance(parent_modal, Tensor)
            or parent_modal.ndim < 2
            or parent_modal.shape[-1] != self.rank
            or not parent_modal.is_floating_point()
            or not bool(torch.isfinite(parent_modal).all())
        ):
            raise ValueError("soft-polarity parent modal geometry differs")
        parent = parent_modal.to(dtype=torch.float64)
        bounded = (
            self.bounded_coordinates(parent)
            if coordinates is None
            else coordinates.to(device=parent.device, dtype=torch.float64)
        )
        if bounded.shape != (*parent.shape[:-1], 2):
            raise ValueError("soft-polarity parent and coordinate geometry differs")
        original_shape = parent.shape
        flat_parent = parent.reshape(-1, self.rank)
        flat_coordinates = bounded.reshape(-1, 2)
        gain, direction, bounded_direction, logit, pedal, delta = (
            fisher_soft_polarity_modal_terms(
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
                self.eta.to(parent.device),
                trust_fraction=self.trust_fraction,
            )
        )
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
            raise RuntimeError("soft-polarity provider mutated a runtime input")
        if bool(support.any()) and not bool(torch.isfinite(modal[support]).all()):
            raise RuntimeError("soft-polarity modal correction became nonfinite")
        if bool((modal[~support] != 0.0).any()):
            raise RuntimeError("soft-polarity modal correction escaped support")
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


def build_autonomous_complete_h4_fisher_soft_polarity(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    eta: Tensor,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolarityProvider:
    """Build and hash-bind one V20f soft-polarity provider."""

    return AutonomousCompleteH4FisherSoftPolarityProvider(
        base_provider=base_provider,
        proposal_provider=proposal_provider,
        eta=eta,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )


def build_autonomous_complete_h4_fisher_soft_polarity_zero_control(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherSoftPolarityProvider:
    """Build the exact zero-gain/base-endpoint member of the V20f formula."""

    return build_autonomous_complete_h4_fisher_soft_polarity(
        base_provider,
        proposal_provider,
        eta=torch.zeros(_SOFT_POLARITY_ETA_COUNT, dtype=torch.float64),
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
    )


def build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
    base_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    polarity: int,
    transfer_protocol_sha256: str,
    transfer_evidence_sha256: str,
) -> AutonomousCompleteH4FisherContinuousTransferProvider:
    """Build the exact fixed ``+/-`` signed-log envelope comparison arm."""

    if type(polarity) is not int or polarity not in {-1, 1}:
        raise ValueError("soft-polarity control polarity must be exactly -1 or 1")
    return build_autonomous_complete_h4_fisher_continuous_axis_response(
        base_provider,
        proposal_provider,
        coordinate_index=1,
        response_law="signed_log",
        polarity=polarity,
        transfer_protocol_sha256=transfer_protocol_sha256,
        transfer_evidence_sha256=transfer_evidence_sha256,
        signed_log_kappa=_authenticated_signed_log_kappa(),
    )
