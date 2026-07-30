"""Candidate-bound causal residual heads for progressive Gemma L3/L4 repair.

The locked rank-64 shadow runtime remains a source-authoritative measurement
tool.  This module lowers its private A-fit residual traces into immutable
one-prefill candidates built on ``Gemma3L3L4OnePassBridge``:

* an X4 head repairs ``layer.4.mlp.normalized_input``;
* an H4 head repairs the complete ``layer.4.output`` carrier; and
* every head is a homogeneous, family-balanced causal finite-displacement
  ridge fit over L3 source modes; and
* an H4 head may add one compact local feature by either reusing its output
  decoder or using an explicit independent state encoder to encode the
  realized, post-X4 H4 carrier before correction.

Heads are installed sequentially.  In particular, a joint candidate is made
by accepting an X4 candidate, remeasuring its H4 residual on the next
progressive iteration, and then adding H4.  This avoids pretending that an
H4 fit measured before the nonlinear X4 intervention is conditionally valid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor, nn

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .causal_edge_jvp import apply_causal_lag_convolution
from .compiler.progressive import (
    MutationProposal,
    ProgressiveCandidate,
    ProgressivePhase,
    ProgressiveResourceFootprint,
    ResidualMap,
)
from .compiler.calibration import CausalLanguageModelNLL
from .conditional_quadratic_edge import (
    build_causal_lagged_modal_design,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    Gemma3L3L4OnePassBridge,
    Gemma3L3L4OnePassExecution,
    Gemma3L3L4OnePassPrefix,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    derive_gemma3_l3_l4_supervised_boundary,
)
from .gemma3_l3_l4_progressive_worker import (
    GemmaCarrierResidualAnalysis,
    GemmaL3L4DevelopmentObservation,
    GemmaProgressiveExample,
    GemmaProgressiveExecutable,
    GemmaTwoHeadFitSequence,
    LegacyRank64GemmaProgressiveExecutable,
)
from .radial_finite_displacement_correction import (
    family_balanced_row_weights,
)


__all__ = [
    "GEMMA_TWO_HEAD_COMPUTE_SCOPE",
    "GEMMA_TWO_HEAD_PARAMETER_SCOPE",
    "GEMMA_TWO_HEAD_RUNTIME_DTYPE",
    "GEMMA_TWO_HEAD_RUNTIME_ID",
    "GemmaCausalResidualHead",
    "GemmaL3L4TwoHeadArtifact",
    "GemmaL3L4TwoHeadExecutable",
    "GemmaL3L4TwoHeadMutationLowerer",
    "ResidualHeadConditioning",
    "ResidualHeadFitObjective",
    "fit_gemma_causal_residual_head",
]


HeadSite = str
ResidualHeadConditioning = Literal[
    "l3_source_modes",
    "l3_source_modes_plus_realized_h4_decoder_modes_v1",
    "l3_source_modes_plus_independent_realized_h4_modes_v1",
]
ResidualHeadFitObjective = Literal[
    "hidden_residual_ridge",
    "source_nll_vjp_metric_ridge_v1",
    "candidate_nll_vjp_metric_ridge_v1",
]

_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_HEAD_SITES = frozenset({_X4_SITE, _H4_SITE})
_FIT_OBJECTIVES = frozenset(
    {
        "hidden_residual_ridge",
        "source_nll_vjp_metric_ridge_v1",
        "candidate_nll_vjp_metric_ridge_v1",
    }
)
_HEAD_CONDITIONINGS = frozenset(
    {
        "l3_source_modes",
        "l3_source_modes_plus_realized_h4_decoder_modes_v1",
        "l3_source_modes_plus_independent_realized_h4_modes_v1",
    }
)
_REALIZED_H4_CONDITIONINGS = frozenset(
    {
        "l3_source_modes_plus_realized_h4_decoder_modes_v1",
        "l3_source_modes_plus_independent_realized_h4_modes_v1",
    }
)
_INDEPENDENT_STATE_CONDITIONING = (
    "l3_source_modes_plus_independent_realized_h4_modes_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN = b"fisher-graph:gemma-l3-l4-two-head-lowerer:v1\0"
_EXECUTION_ABI = "gemma3-l3-l4-one-pass-causal-residual-heads-v2"
_STATE_METADATA_KEY = "metadata_utf8"
_HEAD_STATE_SCHEMA_V3 = (
    "fisher_graph.gemma_l3_l4_causal_residual_head_state.v3"
)
_HEAD_STATE_SCHEMA_V4 = (
    "fisher_graph.gemma_l3_l4_causal_residual_head_state.v4"
)
_ARTIFACT_STATE_SCHEMA = (
    "fisher_graph.gemma_l3_l4_two_head_artifact_state.v2"
)
_HEAD_STATE_KEYS_V3 = frozenset(
    {_STATE_METADATA_KEY, "decoder", "lag_kernel", "state_kernel"}
)
_HEAD_STATE_KEYS_V4 = frozenset(
    {
        _STATE_METADATA_KEY,
        "decoder",
        "lag_kernel",
        "state_kernel",
        "state_encoder",
    }
)
_HEAD_METADATA_KEYS_V3 = frozenset(
    {
        "schema",
        "format_version",
        "site",
        "parent_runtime_binding_sha256",
        "residual_map_sha256",
        "analysis_artifact_sha256",
        "fit_manifest_sha256",
        "bridge_binding_sha256",
        "decoder_sha256",
        "lag_kernel_sha256",
        "state_kernel_sha256",
        "conditioning",
        "ridge",
        "fit_row_count",
        "family_ids",
        "fit_sequence_sha256s",
        "fit_objective",
        "weighted_residual_rmse",
        "normalized_nll_direction_rmse",
        "linearized_nll_residual_rmse",
        "artifact_sha256",
    }
)
_HEAD_METADATA_KEYS_V4 = frozenset(
    {*_HEAD_METADATA_KEYS_V3, "state_encoder_sha256"}
)
_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "parent_artifact_sha256",
        "parent_receipt_sha256",
        "residual_map_sha256",
        "analysis_artifact_sha256",
        "bridge_binding_sha256",
        "live_model_sha256",
        "adapter_execution_sha256",
        "head_count",
        "head_sites",
        "head_artifact_sha256s",
        "recipe_sha256",
        "artifact_sha256",
        "execution_sha256",
        "runtime_binding_sha256",
    }
)
GEMMA_TWO_HEAD_PARAMETER_SCOPE = (
    "logical-model-parameters-plus-prepared-modal-float-coefficients"
)
GEMMA_TWO_HEAD_COMPUTE_SCOPE = (
    "torch-linear-weight-macs-plus-modal-linear-mac-upper-bound"
)
GEMMA_TWO_HEAD_RUNTIME_ID = "torch-gemma-l3-l4-one-pass-overlay"
GEMMA_TWO_HEAD_RUNTIME_DTYPE = "native-model-plus-float64-modal"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(kind: str, value: object) -> str:
    return hashlib.sha256(
        _HASH_DOMAIN
        + kind.encode("ascii")
        + b"\0"
        + _canonical_json_bytes(value)
    ).hexdigest()


def _state_metadata_tensor(value: Mapping[str, object]) -> Tensor:
    encoded = _canonical_json_bytes(value)
    return torch.tensor(tuple(encoded), dtype=torch.uint8)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate state metadata key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _state_metadata(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() <= 0
    ):
        raise ValueError(
            f"{label} metadata must be a nonempty strided uint8 vector"
        )
    encoded = bytes(
        value.detach().to(device="cpu").contiguous().tolist()
    )
    try:
        decoded = json.loads(
            encoded.decode("ascii", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} metadata is not canonical JSON") from error
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} metadata must contain a JSON object")
    if _canonical_json_bytes(decoded) != encoded:
        raise ValueError(f"{label} metadata is not canonically encoded")
    return decoded


def _strict_mapping_keys(
    value: object,
    *,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    observed = frozenset(value)
    if observed != expected:
        missing = tuple(sorted(expected - observed))
        extra = tuple(sorted(observed - expected))
        raise ValueError(
            f"{label} keys differ: missing={missing!r}, extra={extra!r}"
        )
    return value


def _strict_tensor_state(
    value: object,
    *,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Tensor]:
    state = _strict_mapping_keys(
        value,
        expected=expected,
        label=label,
    )
    if any(not isinstance(item, Tensor) for item in state.values()):
        raise TypeError(f"{label} values must all be Tensors")
    return state  # type: ignore[return-value]


def _state_tensor_copy(value: object, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise ValueError(f"{label} must be a materialized strided Tensor")
    return value.detach().to(device="cpu").contiguous().clone()


def _strict_metadata_keys(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    return _strict_mapping_keys(
        value,
        expected=expected,
        label=f"{label} metadata",
    )


def _metadata_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _metadata_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _metadata_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    return value


def _metadata_string_tuple(
    value: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return tuple(value)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise TypeError("hash inputs must be materialized strided tensors")
    canonical = value.detach().to(device="cpu").contiguous()
    return _sha256(
        "tensor",
        {
            "dtype": str(canonical.dtype),
            "shape": tuple(int(width) for width in canonical.shape),
            "bytes_sha256": hashlib.sha256(
                canonical.view(torch.uint8).numpy().tobytes(order="C")
            ).hexdigest(),
        },
    )


def _canonical_head_site(value: object) -> HeadSite:
    if value not in _HEAD_SITES:
        raise ValueError("head site must be X4 or H4")
    return str(value)


def _canonical_fit_objective(
    value: object,
) -> ResidualHeadFitObjective:
    if value not in _FIT_OBJECTIVES:
        raise ValueError(
            "fit_objective must be hidden_residual_ridge or "
            "source_nll_vjp_metric_ridge_v1 or "
            "candidate_nll_vjp_metric_ridge_v1"
        )
    return str(value)  # type: ignore[return-value]


def _canonical_head_conditioning(
    value: object,
) -> ResidualHeadConditioning:
    if value not in _HEAD_CONDITIONINGS:
        raise ValueError(
            "conditioning must be l3_source_modes or "
            "l3_source_modes_plus_realized_h4_decoder_modes_v1 or "
            "l3_source_modes_plus_independent_realized_h4_modes_v1"
        )
    return str(value)  # type: ignore[return-value]


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _canonical_sign_rows(value: Tensor) -> Tensor:
    rows = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    pivots = rows.abs().argmax(dim=1)
    signs = rows.gather(1, pivots.unsqueeze(1)).squeeze(1)
    return (
        rows
        * torch.where(
            signs < 0.0,
            -torch.ones_like(signs),
            torch.ones_like(signs),
        ).unsqueeze(1)
    ).contiguous()


def _overflow_safe_weighted_rms(
    design: Tensor,
    weights: Tensor,
) -> Tensor:
    maximum = design.abs().amax(dim=0)
    nonzero = maximum > 0
    scaled = torch.zeros_like(design)
    if bool(nonzero.any()):
        scaled[:, nonzero] = design[:, nonzero] / maximum[nonzero]
    rms = torch.sqrt((scaled.square() * weights.unsqueeze(1)).sum(dim=0))
    return (maximum * rms).contiguous()


def _normalized_nll_directions(gradient: Tensor) -> Tensor:
    """Normalize per-row NLL VJPs without changing family/example mass."""

    norms = torch.linalg.vector_norm(gradient, dim=1)
    maximum = float(norms.max()) if norms.numel() else 0.0
    result = torch.zeros_like(gradient)
    if maximum == 0.0:
        return result
    threshold = math.sqrt(torch.finfo(torch.float64).eps) * maximum
    active = norms > threshold
    if bool(active.any()):
        result[active] = gradient[active] / norms[active].unsqueeze(1)
    return result.contiguous()


def _solve_source_nll_vjp_metric_ridge(
    *,
    standardized_design: Tensor,
    weighted_design: Tensor,
    weights: Tensor,
    target: Tensor,
    residual: Tensor,
    normalized_gradient: Tensor,
    decoder: Tensor,
    ridge: float,
    initial: Tensor,
) -> Tensor:
    """Solve the bounded ``I + uu.T`` loss metric with matrix-free CG."""

    projected_gradient = normalized_gradient @ decoder.T
    directional_target = (
        normalized_gradient * residual
    ).sum(dim=1)
    if not bool(projected_gradient.abs().any()):
        return initial.contiguous()

    weighted_target = target * weights.sqrt().unsqueeze(1)
    right_hand_side = weighted_design.T @ weighted_target
    right_hand_side = right_hand_side + standardized_design.T @ (
        (weights * directional_target).unsqueeze(1)
        * projected_gradient
    )

    def operator(value: Tensor) -> Tensor:
        prediction = standardized_design @ value
        directional_prediction = (
            prediction * projected_gradient
        ).sum(dim=1)
        return (
            weighted_design.T @ (weighted_design @ value)
            + ridge * value
            + standardized_design.T
            @ (
                (weights * directional_prediction).unsqueeze(1)
                * projected_gradient
            )
        )

    diagonal = (
        weighted_design.square().sum(dim=0).unsqueeze(1)
        + ridge
        + standardized_design.square().T
        @ (weights.unsqueeze(1) * projected_gradient.square())
    )
    if (
        not bool(torch.isfinite(right_hand_side).all())
        or not bool(torch.isfinite(diagonal).all())
        or bool((diagonal <= 0.0).any())
    ):
        raise RuntimeError("loss-metric ridge system is invalid")

    solution = initial.detach().clone()
    residual_vector = right_hand_side - operator(solution)
    right_norm = max(
        float(torch.linalg.vector_norm(right_hand_side)),
        1.0,
    )
    tolerance = 1.0e-10 * right_norm
    if float(torch.linalg.vector_norm(residual_vector)) <= tolerance:
        return solution.contiguous()
    preconditioned = residual_vector / diagonal
    search = preconditioned.clone()
    residual_dot = (residual_vector * preconditioned).sum()
    maximum_iterations = 512
    for _ in range(maximum_iterations):
        operated = operator(search)
        denominator = (search * operated).sum()
        if (
            not bool(torch.isfinite(denominator))
            or float(denominator) <= 0.0
        ):
            raise RuntimeError(
                "loss-metric conjugate-gradient curvature is invalid"
            )
        step = residual_dot / denominator
        solution = solution + step * search
        residual_vector = residual_vector - step * operated
        if (
            float(torch.linalg.vector_norm(residual_vector))
            <= tolerance
        ):
            return solution.contiguous()
        preconditioned = residual_vector / diagonal
        next_residual_dot = (
            residual_vector * preconditioned
        ).sum()
        if (
            not bool(torch.isfinite(next_residual_dot))
            or float(next_residual_dot) < 0.0
        ):
            raise RuntimeError(
                "loss-metric conjugate-gradient residual is invalid"
            )
        search = (
            preconditioned
            + (next_residual_dot / residual_dot) * search
        )
        residual_dot = next_residual_dot
    relative_residual = float(
        torch.linalg.vector_norm(residual_vector)
    ) / right_norm
    raise RuntimeError(
        "loss-metric conjugate-gradient fit did not converge: "
        f"relative residual {relative_residual:.3e}"
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalResidualHead(Gemma3L3L4CorrectionProvider):
    """One authenticated stationary causal finite-displacement head."""

    site: HeadSite
    parent_runtime_binding_sha256: str
    residual_map_sha256: str
    analysis_artifact_sha256: str
    fit_manifest_sha256: str
    bridge_binding_sha256: str
    decoder: Tensor
    lag_kernel: Tensor
    state_kernel: Tensor
    conditioning: ResidualHeadConditioning
    ridge: float
    fit_row_count: int
    family_ids: tuple[str, ...]
    fit_sequence_sha256s: tuple[str, ...]
    fit_objective: ResidualHeadFitObjective
    weighted_residual_rmse: float
    normalized_nll_direction_rmse: float
    linearized_nll_residual_rmse: float
    artifact_sha256: str = ""
    state_encoder: Tensor | None = None

    def __post_init__(self) -> None:
        site = _canonical_head_site(self.site)
        object.__setattr__(self, "site", site)
        for name in (
            "parent_runtime_binding_sha256",
            "residual_map_sha256",
            "analysis_artifact_sha256",
            "fit_manifest_sha256",
            "bridge_binding_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        decoder = self.decoder
        kernel = self.lag_kernel
        state_kernel = self.state_kernel
        state_encoder = self.state_encoder
        if (
            isinstance(state_encoder, Tensor)
            and state_encoder.shape == (0, 0)
            and state_encoder.layout == torch.strided
            and state_encoder.device.type != "meta"
            and state_encoder.is_floating_point()
        ):
            state_encoder = None
            object.__setattr__(self, "state_encoder", None)
        conditioning = _canonical_head_conditioning(self.conditioning)
        object.__setattr__(self, "conditioning", conditioning)
        if (
            not isinstance(decoder, Tensor)
            or decoder.ndim != 2
            or not decoder.is_floating_point()
            or decoder.shape[0] <= 0
            or decoder.shape[1] <= 0
            or not isinstance(kernel, Tensor)
            or kernel.ndim != 3
            or not kernel.is_floating_point()
            or kernel.shape[0] <= 0
            or kernel.shape[1] <= 0
            or kernel.shape[2] != decoder.shape[0]
            or not isinstance(state_kernel, Tensor)
            or state_kernel.ndim != 2
            or not state_kernel.is_floating_point()
            or not bool(torch.isfinite(decoder).all())
            or not bool(torch.isfinite(kernel).all())
            or not bool(torch.isfinite(state_kernel).all())
        ):
            raise ValueError("residual head tensor geometry is invalid")
        if conditioning == "l3_source_modes":
            expected_state_shape = (0, 0)
            if state_encoder is not None:
                raise ValueError(
                    "source-only conditioning cannot have a state encoder"
                )
        elif (
            conditioning
            == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        ):
            expected_state_shape = (
                decoder.shape[0],
                decoder.shape[0],
            )
            if state_encoder is not None:
                raise ValueError(
                    "legacy decoder conditioning cannot have a state encoder"
                )
        else:
            if (
                not isinstance(state_encoder, Tensor)
                or state_encoder.layout != torch.strided
                or state_encoder.device.type == "meta"
                or state_encoder.ndim != 2
                or not state_encoder.is_floating_point()
                or state_encoder.shape[0] <= 0
                or state_encoder.shape[0] > state_encoder.shape[1]
                or state_encoder.shape[1] != decoder.shape[1]
                or not bool(torch.isfinite(state_encoder).all())
            ):
                raise ValueError(
                    "independent residual head state encoder geometry "
                    "is invalid"
                )
            expected_state_shape = (
                state_encoder.shape[0],
                decoder.shape[0],
            )
        if state_kernel.shape != expected_state_shape:
            raise ValueError(
                "residual head state kernel differs from its conditioning"
            )
        if (
            conditioning in _REALIZED_H4_CONDITIONINGS
            and site != _H4_SITE
        ):
            raise ValueError(
                "realized-H4 conditioning is valid only for the H4 head"
            )
        identity = decoder @ decoder.T
        error = float(
            (
                identity
                - torch.eye(
                    decoder.shape[0],
                    dtype=decoder.dtype,
                    device=decoder.device,
                )
            )
            .abs()
            .max()
        )
        if error > 1.0e-8:
            raise ValueError("residual head decoder rows must be orthonormal")
        if state_encoder is not None:
            state_identity = state_encoder @ state_encoder.T
            state_error = float(
                (
                    state_identity
                    - torch.eye(
                        state_encoder.shape[0],
                        dtype=state_encoder.dtype,
                        device=state_encoder.device,
                    )
                )
                .abs()
                .max()
            )
            if state_error > 1.0e-8:
                raise ValueError(
                    "residual head state encoder rows must be orthonormal"
                )
        object.__setattr__(
            self,
            "ridge",
            _positive_finite(self.ridge, label="ridge"),
        )
        _positive_int(self.fit_row_count, label="fit_row_count")
        if (
            type(self.family_ids) is not tuple
            or not self.family_ids
            or self.family_ids != tuple(sorted(set(self.family_ids)))
            or any(
                not isinstance(value, str) or not value
                for value in self.family_ids
            )
        ):
            raise ValueError("family_ids must be canonical and nonempty")
        if (
            type(self.fit_sequence_sha256s) is not tuple
            or not self.fit_sequence_sha256s
            or self.fit_sequence_sha256s
            != tuple(sorted(set(self.fit_sequence_sha256s)))
        ):
            raise ValueError(
                "fit sequence identities must be canonical and nonempty"
        )
        for value in self.fit_sequence_sha256s:
            _require_sha256(value, label="fit sequence identity")
        object.__setattr__(
            self,
            "fit_objective",
            _canonical_fit_objective(self.fit_objective),
        )
        for name in (
            "weighted_residual_rmse",
            "normalized_nll_direction_rmse",
            "linearized_nll_residual_rmse",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="head artifact",
            ) != computed:
                raise ValueError("residual head artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def rank(self) -> int:
        return int(self.decoder.shape[0])

    @property
    def width(self) -> int:
        return int(self.decoder.shape[1])

    @property
    def source_rank(self) -> int:
        return int(self.lag_kernel.shape[1])

    @property
    def lag_count(self) -> int:
        return int(self.lag_kernel.shape[0])

    @property
    def state_rank(self) -> int:
        if self.conditioning == "l3_source_modes":
            return 0
        if self.state_encoder is None:
            return self.rank
        return int(self.state_encoder.shape[0])

    @property
    def prepared_float_scalar_count(self) -> int:
        return int(
            self.decoder.numel()
            + self.lag_kernel.numel()
            + self.state_kernel.numel()
            + (
                0
                if self.state_encoder is None
                else self.state_encoder.numel()
            )
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        base = (
            self.lag_count * self.source_rank * self.rank
            + self.rank * self.width
        )
        if self.conditioning in _REALIZED_H4_CONDITIONINGS:
            base += (
                self.width * self.state_rank
                + self.state_rank * self.rank
            )
        return base

    def _computed_sha256(self) -> str:
        payload = {
            "format_version": 3,
            "site": self.site,
            "parent_runtime_binding_sha256": (
                self.parent_runtime_binding_sha256
            ),
            "residual_map_sha256": self.residual_map_sha256,
            "analysis_artifact_sha256": (
                self.analysis_artifact_sha256
            ),
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "decoder_sha256": _tensor_sha256(self.decoder),
            "lag_kernel_sha256": _tensor_sha256(self.lag_kernel),
            "state_kernel_sha256": _tensor_sha256(self.state_kernel),
            "conditioning": self.conditioning,
            "ridge": self.ridge,
            "fit_row_count": self.fit_row_count,
            "family_ids": self.family_ids,
            "fit_sequence_sha256s": self.fit_sequence_sha256s,
            "fit_objective": self.fit_objective,
            "weighted_residual_rmse": self.weighted_residual_rmse,
            "normalized_nll_direction_rmse": (
                self.normalized_nll_direction_rmse
            ),
            "linearized_nll_residual_rmse": (
                self.linearized_nll_residual_rmse
            ),
            "fit_kind": self.fit_objective,
            "causal_direction": (
                "target_t_reads_source_t_minus_lag"
                if self.conditioning == "l3_source_modes"
                else (
                    "target_t_reads_source_t_minus_lag_and_"
                    "pre_correction_realized_h4_t"
                )
            ),
        }
        if self.state_encoder is not None:
            payload["format_version"] = 4
            payload["state_encoder_sha256"] = _tensor_sha256(
                self.state_encoder
            )
            payload["causal_direction"] = (
                "target_t_reads_source_t_minus_lag_and_"
                "independently_encoded_pre_correction_realized_h4_t"
            )
        return _sha256("causal-residual-head", payload)

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("residual head tensor payload drifted")

    def state_dict(self) -> dict[str, Tensor]:
        """Return a strict tensor-only state suitable for ``weights_only``."""

        self.validate_integrity()
        decoder = _state_tensor_copy(
            self.decoder,
            label="residual head decoder",
        )
        lag_kernel = _state_tensor_copy(
            self.lag_kernel,
            label="residual head lag kernel",
        )
        state_kernel = _state_tensor_copy(
            self.state_kernel,
            label="residual head state kernel",
        )
        metadata = {
            "schema": _HEAD_STATE_SCHEMA_V3,
            "format_version": 3,
            "site": self.site,
            "parent_runtime_binding_sha256": (
                self.parent_runtime_binding_sha256
            ),
            "residual_map_sha256": self.residual_map_sha256,
            "analysis_artifact_sha256": self.analysis_artifact_sha256,
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "decoder_sha256": _tensor_sha256(decoder),
            "lag_kernel_sha256": _tensor_sha256(lag_kernel),
            "state_kernel_sha256": _tensor_sha256(state_kernel),
            "conditioning": self.conditioning,
            "ridge": self.ridge,
            "fit_row_count": self.fit_row_count,
            "family_ids": self.family_ids,
            "fit_sequence_sha256s": self.fit_sequence_sha256s,
            "fit_objective": self.fit_objective,
            "weighted_residual_rmse": self.weighted_residual_rmse,
            "normalized_nll_direction_rmse": (
                self.normalized_nll_direction_rmse
            ),
            "linearized_nll_residual_rmse": (
                self.linearized_nll_residual_rmse
            ),
            "artifact_sha256": self.artifact_sha256,
        }
        state = {
            _STATE_METADATA_KEY: _state_metadata_tensor(metadata),
            "decoder": decoder,
            "lag_kernel": lag_kernel,
            "state_kernel": state_kernel,
        }
        if self.state_encoder is not None:
            state_encoder = _state_tensor_copy(
                self.state_encoder,
                label="residual head state encoder",
            )
            metadata["schema"] = _HEAD_STATE_SCHEMA_V4
            metadata["format_version"] = 4
            metadata["state_encoder_sha256"] = _tensor_sha256(
                state_encoder
            )
            state[_STATE_METADATA_KEY] = _state_metadata_tensor(metadata)
            state["state_encoder"] = state_encoder
        self.validate_integrity()
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GemmaCausalResidualHead:
        """Strict-load a tensor-only head state and reauthenticate its hash."""

        if not isinstance(state, Mapping):
            raise TypeError("residual head state must be a mapping")
        if any(not isinstance(key, str) for key in state):
            raise ValueError("residual head state keys must be strings")
        is_v4 = "state_encoder" in state
        state_keys = (
            _HEAD_STATE_KEYS_V4 if is_v4 else _HEAD_STATE_KEYS_V3
        )
        metadata_keys = (
            _HEAD_METADATA_KEYS_V4
            if is_v4
            else _HEAD_METADATA_KEYS_V3
        )
        expected_schema = (
            _HEAD_STATE_SCHEMA_V4 if is_v4 else _HEAD_STATE_SCHEMA_V3
        )
        expected_version = 4 if is_v4 else 3
        tensors = _strict_tensor_state(
            state,
            expected=state_keys,
            label="residual head state",
        )
        metadata = _strict_metadata_keys(
            _state_metadata(
                tensors[_STATE_METADATA_KEY],
                label="residual head state",
            ),
            expected=metadata_keys,
            label="residual head state",
        )
        if (
            metadata["schema"] != expected_schema
            or _metadata_int(
                metadata["format_version"],
                label="head state format_version",
            )
            != expected_version
        ):
            raise ValueError("residual head state schema or version differs")
        decoder = _state_tensor_copy(
            tensors["decoder"],
            label="residual head decoder",
        )
        lag_kernel = _state_tensor_copy(
            tensors["lag_kernel"],
            label="residual head lag kernel",
        )
        state_kernel = _state_tensor_copy(
            tensors["state_kernel"],
            label="residual head state kernel",
        )
        state_encoder = (
            _state_tensor_copy(
                tensors["state_encoder"],
                label="residual head state encoder",
            )
            if is_v4
            else None
        )
        if (
            _tensor_sha256(decoder)
            != _require_sha256(
                metadata["decoder_sha256"],
                label="serialized decoder",
            )
            or _tensor_sha256(lag_kernel)
            != _require_sha256(
                metadata["lag_kernel_sha256"],
                label="serialized lag kernel",
            )
            or _tensor_sha256(state_kernel)
            != _require_sha256(
                metadata["state_kernel_sha256"],
                label="serialized state kernel",
            )
            or (
                state_encoder is not None
                and _tensor_sha256(state_encoder)
                != _require_sha256(
                    metadata["state_encoder_sha256"],
                    label="serialized state encoder",
                )
            )
        ):
            raise ValueError("serialized residual head tensor hash differs")
        result = cls(
            site=_metadata_string(
                metadata["site"],
                label="head state site",
            ),
            parent_runtime_binding_sha256=_require_sha256(
                metadata["parent_runtime_binding_sha256"],
                label="serialized parent runtime binding",
            ),
            residual_map_sha256=_require_sha256(
                metadata["residual_map_sha256"],
                label="serialized residual map",
            ),
            analysis_artifact_sha256=_require_sha256(
                metadata["analysis_artifact_sha256"],
                label="serialized residual analysis",
            ),
            fit_manifest_sha256=_require_sha256(
                metadata["fit_manifest_sha256"],
                label="serialized fit manifest",
            ),
            bridge_binding_sha256=_require_sha256(
                metadata["bridge_binding_sha256"],
                label="serialized bridge binding",
            ),
            decoder=decoder,
            lag_kernel=lag_kernel,
            state_kernel=state_kernel,
            conditioning=_canonical_head_conditioning(
                metadata["conditioning"]
            ),
            ridge=_metadata_float(
                metadata["ridge"],
                label="head state ridge",
            ),
            fit_row_count=_metadata_int(
                metadata["fit_row_count"],
                label="head state fit_row_count",
            ),
            family_ids=_metadata_string_tuple(
                metadata["family_ids"],
                label="head state family_ids",
            ),
            fit_sequence_sha256s=_metadata_string_tuple(
                metadata["fit_sequence_sha256s"],
                label="head state fit_sequence_sha256s",
            ),
            fit_objective=_canonical_fit_objective(
                metadata["fit_objective"]
            ),
            weighted_residual_rmse=_metadata_float(
                metadata["weighted_residual_rmse"],
                label="head state weighted_residual_rmse",
            ),
            normalized_nll_direction_rmse=_metadata_float(
                metadata["normalized_nll_direction_rmse"],
                label="head state normalized_nll_direction_rmse",
            ),
            linearized_nll_residual_rmse=_metadata_float(
                metadata["linearized_nll_residual_rmse"],
                label="head state linearized_nll_residual_rmse",
            ),
            artifact_sha256=_require_sha256(
                metadata["artifact_sha256"],
                label="serialized head artifact",
            ),
            state_encoder=state_encoder,
        )
        result.validate_integrity()
        return result

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        prefix.validate_integrity()
        if (
            prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or prefix.source_modes.shape[-1] != self.source_rank
            or prefix.clamped_y3.shape[-1] != self.width
            or not isinstance(realized_state, Tensor)
            or realized_state.shape != prefix.clamped_y3.shape
            or not realized_state.is_floating_point()
        ):
            raise ValueError("residual head and bridge geometry differ")
        active_state = prefix.target_affected_mask.to(realized_state.device)
        if (
            self.conditioning in _REALIZED_H4_CONDITIONINGS
            and bool(active_state.any())
            and not bool(torch.isfinite(realized_state[active_state]).all())
        ):
            raise ValueError("realized H4 state is nonfinite on active rows")
        result = torch.zeros_like(prefix.clamped_y3)
        for batch in range(prefix.source_modes.shape[0]):
            modal = apply_causal_lag_convolution(
                prefix.source_modes[batch],
                kernel=self.lag_kernel,
                logical_positions=prefix.logical_positions[batch],
                valid_mask=prefix.valid_target_mask[batch],
            )
            if self.conditioning in _REALIZED_H4_CONDITIONINGS:
                encoder_tensor = (
                    self.decoder
                    if self.state_encoder is None
                    else self.state_encoder
                )
                encoder = encoder_tensor.to(
                    device=realized_state.device,
                    dtype=torch.float64,
                )
                realized_modes = (
                    realized_state[batch].to(torch.float64) @ encoder.T
                ).to(device=modal.device, dtype=modal.dtype)
                modal = modal + realized_modes @ self.state_kernel.to(
                    device=modal.device,
                    dtype=modal.dtype,
                )
            decoded = modal @ self.decoder.to(
                device=modal.device,
                dtype=modal.dtype,
            )
            active = prefix.target_affected_mask[batch].to(decoded.device)
            if bool(active.any()):
                result[batch, active.to(result.device)] = decoded[active].to(
                    device=result.device,
                    dtype=result.dtype,
                )
        inactive = ~prefix.target_affected_mask
        if bool(inactive.any()) and not bool((result[inactive] == 0).all()):
            raise RuntimeError("residual head wrote outside target support")
        self.validate_integrity()
        return result


def fit_gemma_causal_residual_head(
    *,
    site: HeadSite,
    sequences: Sequence[GemmaTwoHeadFitSequence],
    directions: Tensor,
    parent_runtime_binding_sha256: str,
    residual_map_sha256: str,
    analysis_artifact_sha256: str,
    fit_manifest_sha256: str,
    bridge_binding_sha256: str,
    lag_count: int,
    ridge: float,
    fit_objective: ResidualHeadFitObjective = "hidden_residual_ridge",
    conditioning: ResidualHeadConditioning = "l3_source_modes",
) -> GemmaCausalResidualHead:
    """Fit one deterministic family-balanced causal residual head."""

    selected_site = _canonical_head_site(site)
    selected_objective = _canonical_fit_objective(fit_objective)
    selected_conditioning = _canonical_head_conditioning(conditioning)
    if (
        selected_conditioning in _REALIZED_H4_CONDITIONINGS
        and selected_site != _H4_SITE
    ):
        raise ValueError("realized-H4 conditioning is H4-only")
    if selected_conditioning == _INDEPENDENT_STATE_CONDITIONING:
        raise ValueError(
            "independent realized-H4 conditioning requires an explicit "
            "state encoder and prefit state kernel"
        )
    lag_count = _positive_int(lag_count, label="lag_count")
    ridge = _positive_finite(ridge, label="ridge")
    for name, value in (
        ("parent runtime", parent_runtime_binding_sha256),
        ("residual map", residual_map_sha256),
        ("analysis artifact", analysis_artifact_sha256),
        ("fit manifest", fit_manifest_sha256),
        ("bridge binding", bridge_binding_sha256),
    ):
        _require_sha256(value, label=name)
    if (
        isinstance(sequences, (str, bytes))
        or not isinstance(sequences, Sequence)
        or not sequences
        or any(
            not isinstance(value, GemmaTwoHeadFitSequence)
            for value in sequences
        )
    ):
        raise ValueError("head fit requires private two-head fit sequences")
    ordered = tuple(
        sorted(
            (value.detached_copy() for value in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    if len({value.artifact_sha256 for value in ordered}) != len(ordered):
        raise ValueError("head fit sequences must be unique")
    decoder = _canonical_sign_rows(directions)
    width = ordered[0].width
    source_rank = ordered[0].source_rank
    if (
        decoder.ndim != 2
        or decoder.shape[0] <= 0
        or decoder.shape[1] != width
    ):
        raise ValueError("head directions differ from fit boundary width")
    design_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    residual_rows: list[Tensor] = []
    gradient_rows: list[Tensor] = []
    family_rows: list[str] = []
    example_rows: list[str] = []
    sequence_ids: list[str] = []
    for sequence in ordered:
        if (
            sequence.width != width
            or sequence.source_rank != source_rank
            or sequence.runtime_binding_sha256
            != parent_runtime_binding_sha256
        ):
            raise ValueError("head fit sequence geometry or parent differs")
        design = build_causal_lagged_modal_design(
            sequence.source_modes,
            logical_positions=sequence.logical_positions,
            valid_mask=sequence.valid_target_mask,
            lag_count=lag_count,
        )
        if (
            selected_conditioning
            == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        ):
            valid_candidate_h4 = sequence.candidate_h4[
                sequence.valid_target_mask
            ]
            if not bool(torch.isfinite(valid_candidate_h4).all()):
                raise ValueError(
                    "realized-H4 conditioning requires finite valid rows"
                )
            realized_h4_modes = (
                sequence.candidate_h4.to(torch.float64) @ decoder.T
            )
            design = torch.cat(
                (design.to(torch.float64), realized_h4_modes),
                dim=1,
            )
        selected = sequence.target_affected_mask
        residual = (
            sequence.x4_residual_rows
            if selected_site == _X4_SITE
            else sequence.h4_residual_rows
        )
        if (
            selected_objective
            == "candidate_nll_vjp_metric_ridge_v1"
        ):
            if selected_site != _H4_SITE:
                raise ValueError(
                    "candidate-conditioned VJP objective is H4-only"
                )
            if sequence.candidate_h4_loss_gradient is None:
                raise ValueError(
                    "candidate-conditioned H4 fit lacks its VJP"
                )
            gradient = sequence.candidate_h4_loss_gradient[selected]
        else:
            gradient = (
                sequence.x4_loss_gradient[selected]
                if selected_site == _X4_SITE
                else sequence.h4_loss_gradient[selected]
            )
        target = residual @ decoder.T
        design_rows.append(design[selected])
        target_rows.append(target)
        residual_rows.append(residual)
        gradient_rows.append(gradient)
        family_rows.extend([sequence.family_id] * sequence.affected_rows)
        example_rows.extend([sequence.example_id] * sequence.affected_rows)
        sequence_ids.append(sequence.artifact_sha256)
    design = torch.cat(design_rows, dim=0).to(torch.float64)
    target = torch.cat(target_rows, dim=0).to(torch.float64)
    full_residual = torch.cat(residual_rows, dim=0).to(torch.float64)
    loss_gradient = torch.cat(gradient_rows, dim=0).to(torch.float64)
    weights = family_balanced_row_weights(
        tuple(family_rows),
        tuple(example_rows),
    ).to(torch.float64)
    if (
        design.shape[0] != target.shape[0]
        or design.shape[0] != weights.shape[0]
        or design.shape[1]
        != (
            lag_count * source_rank
            + (
                decoder.shape[0]
                if selected_conditioning
                == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
                else 0
            )
        )
        or full_residual.shape != (design.shape[0], width)
        or loss_gradient.shape != full_residual.shape
        or not bool(torch.isfinite(design).all())
        or not bool(torch.isfinite(target).all())
        or not bool(torch.isfinite(full_residual).all())
        or not bool(torch.isfinite(loss_gradient).all())
    ):
        raise ValueError("head fit design or target geometry is invalid")
    rms = _overflow_safe_weighted_rms(design, weights)
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    standardized = design / scales
    root_weights = weights.sqrt().unsqueeze(1)
    weighted_design = standardized * root_weights
    weighted_target = target * root_weights
    gram = weighted_design.T @ weighted_design
    cross = weighted_design.T @ weighted_target
    standardized_coefficients = torch.linalg.solve(
        gram
        + ridge
        * torch.eye(
            gram.shape[0],
            dtype=torch.float64,
        ),
        cross,
    )
    normalized_gradient = _normalized_nll_directions(loss_gradient)
    if selected_objective in (
        "source_nll_vjp_metric_ridge_v1",
        "candidate_nll_vjp_metric_ridge_v1",
    ):
        standardized_coefficients = (
            _solve_source_nll_vjp_metric_ridge(
                standardized_design=standardized,
                weighted_design=weighted_design,
                weights=weights,
                target=target,
                residual=full_residual,
                normalized_gradient=normalized_gradient,
                decoder=decoder,
                ridge=ridge,
                initial=standardized_coefficients,
            )
        )
    coefficients = (
        standardized_coefficients / scales.unsqueeze(1)
    ).contiguous()
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("causal residual ridge fit became nonfinite")
    prediction = design @ coefficients
    full_error = full_residual - prediction @ decoder
    weighted_residual_rmse = float(
        torch.sqrt(
            (
                (prediction - target).square().sum(dim=1)
                * weights
            ).sum()
            / target.shape[1]
        )
    )
    normalized_nll_direction_rmse = float(
        torch.sqrt(
            (
                (
                    normalized_gradient * full_error
                ).sum(dim=1).square()
                * weights
            ).sum()
        )
    )
    linearized_nll_residual_rmse = float(
        torch.sqrt(
            (
                (loss_gradient * full_error).sum(dim=1).square()
                * weights
            ).sum()
        )
    )
    lag_coefficient_count = lag_count * source_rank
    kernel = coefficients[:lag_coefficient_count].reshape(
        lag_count,
        source_rank,
        decoder.shape[0],
    ).contiguous()
    state_kernel = (
        coefficients[lag_coefficient_count:].contiguous()
        if selected_conditioning
        == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        else torch.empty((0, 0), dtype=torch.float64)
    )
    return GemmaCausalResidualHead(
        site=selected_site,
        parent_runtime_binding_sha256=parent_runtime_binding_sha256,
        residual_map_sha256=residual_map_sha256,
        analysis_artifact_sha256=analysis_artifact_sha256,
        fit_manifest_sha256=fit_manifest_sha256,
        bridge_binding_sha256=bridge_binding_sha256,
        decoder=decoder,
        lag_kernel=kernel,
        state_kernel=state_kernel,
        conditioning=selected_conditioning,
        ridge=ridge,
        fit_row_count=int(design.shape[0]),
        family_ids=tuple(sorted(set(family_rows))),
        fit_sequence_sha256s=tuple(sorted(sequence_ids)),
        fit_objective=selected_objective,
        weighted_residual_rmse=weighted_residual_rmse,
        normalized_nll_direction_rmse=(
            normalized_nll_direction_rmse
        ),
        linearized_nll_residual_rmse=(
            linearized_nll_residual_rmse
        ),
    )


@dataclass(frozen=True, slots=True)
class GemmaL3L4TwoHeadArtifact:
    """Immutable one-pass bridge overlay with at most one head per boundary."""

    parent_artifact_sha256: str
    parent_receipt_sha256: str
    residual_map_sha256: str
    analysis_artifact_sha256: str
    bridge_binding_sha256: str
    live_model_sha256: str
    adapter_execution_sha256: str
    heads: tuple[GemmaCausalResidualHead, ...]
    recipe_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "parent_artifact_sha256",
            "parent_receipt_sha256",
            "residual_map_sha256",
            "analysis_artifact_sha256",
            "bridge_binding_sha256",
            "live_model_sha256",
            "adapter_execution_sha256",
            "recipe_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            type(self.heads) is not tuple
            or not self.heads
            or any(
                not isinstance(value, GemmaCausalResidualHead)
                for value in self.heads
            )
            or tuple(value.site for value in self.heads)
            != tuple(sorted(value.site for value in self.heads))
            or len({value.site for value in self.heads}) != len(self.heads)
            or any(
                value.bridge_binding_sha256
                != self.bridge_binding_sha256
                for value in self.heads
            )
        ):
            raise ValueError("two-head artifact heads are invalid")
        for head in self.heads:
            head.validate_integrity()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="two-head artifact",
            ) != computed:
                raise ValueError("two-head artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def execution_sha256(self) -> str:
        return _sha256(
            "candidate-execution",
            {
                "abi": _EXECUTION_ABI,
                "artifact_sha256": self.artifact_sha256,
                "adapter_execution_sha256": (
                    self.adapter_execution_sha256
                ),
                "model_forward_count": 1,
            },
        )

    @property
    def runtime_binding_sha256(self) -> str:
        return _sha256(
            "candidate-runtime-binding",
            {
                "execution_sha256": self.execution_sha256,
                "bridge_binding_sha256": self.bridge_binding_sha256,
                "live_model_sha256": self.live_model_sha256,
                "head_sha256s": tuple(
                    value.artifact_sha256 for value in self.heads
                ),
            },
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return sum(
            value.prepared_float_scalar_count for value in self.heads
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return sum(
            value.logical_macs_per_token_upper_bound
            for value in self.heads
        )

    def head(self, site: HeadSite) -> GemmaCausalResidualHead | None:
        selected = _canonical_head_site(site)
        return next(
            (value for value in self.heads if value.site == selected),
            None,
        )

    def _computed_sha256(self) -> str:
        return _sha256(
            "two-head-artifact",
            {
                "format_version": 2,
                "parent_artifact_sha256": self.parent_artifact_sha256,
                "parent_receipt_sha256": self.parent_receipt_sha256,
                "residual_map_sha256": self.residual_map_sha256,
                "analysis_artifact_sha256": (
                    self.analysis_artifact_sha256
                ),
                "bridge_binding_sha256": self.bridge_binding_sha256,
                "live_model_sha256": self.live_model_sha256,
                "adapter_execution_sha256": (
                    self.adapter_execution_sha256
                ),
                "head_sha256s": tuple(
                    value.artifact_sha256 for value in self.heads
                ),
                "recipe_sha256": self.recipe_sha256,
                "execution_abi": _EXECUTION_ABI,
                "model_forward_count": 1,
                "native_boundary_fallback": False,
            },
        )

    def validate_integrity(self) -> None:
        for head in self.heads:
            head.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("two-head artifact payload drifted")

    def state_dict(self) -> dict[str, Tensor]:
        """Return a flat tensor-only artifact state with strict head slots."""

        self.validate_integrity()
        metadata = {
            "schema": _ARTIFACT_STATE_SCHEMA,
            "format_version": 2,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "parent_receipt_sha256": self.parent_receipt_sha256,
            "residual_map_sha256": self.residual_map_sha256,
            "analysis_artifact_sha256": self.analysis_artifact_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "live_model_sha256": self.live_model_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "head_count": len(self.heads),
            "head_sites": tuple(head.site for head in self.heads),
            "head_artifact_sha256s": tuple(
                head.artifact_sha256 for head in self.heads
            ),
            "recipe_sha256": self.recipe_sha256,
            "artifact_sha256": self.artifact_sha256,
            "execution_sha256": self.execution_sha256,
            "runtime_binding_sha256": self.runtime_binding_sha256,
        }
        state = {
            _STATE_METADATA_KEY: _state_metadata_tensor(metadata),
        }
        for index, head in enumerate(self.heads):
            for name, value in head.state_dict().items():
                state[f"heads.{index}.{name}"] = value
        self.validate_integrity()
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GemmaL3L4TwoHeadArtifact:
        """Strict-load a tensor-only artifact and every nested head."""

        if not isinstance(state, Mapping):
            raise TypeError("two-head artifact state must be a mapping")
        if any(not isinstance(key, str) for key in state):
            raise ValueError("two-head artifact state keys must be strings")
        if any(not isinstance(value, Tensor) for value in state.values()):
            raise TypeError(
                "two-head artifact state values must all be Tensors"
            )
        try:
            metadata_tensor = state[_STATE_METADATA_KEY]
        except KeyError as error:
            raise ValueError(
                "two-head artifact state lacks metadata_utf8"
            ) from error
        metadata = _strict_metadata_keys(
            _state_metadata(
                metadata_tensor,
                label="two-head artifact state",
            ),
            expected=_ARTIFACT_METADATA_KEYS,
            label="two-head artifact state",
        )
        if (
            metadata["schema"] != _ARTIFACT_STATE_SCHEMA
            or _metadata_int(
                metadata["format_version"],
                label="artifact state format_version",
            )
            != 2
        ):
            raise ValueError("two-head artifact state schema or version differs")
        head_count = _metadata_int(
            metadata["head_count"],
            label="artifact state head_count",
        )
        if not 1 <= head_count <= len(_HEAD_SITES):
            raise ValueError("artifact state head_count is invalid")
        head_sites = _metadata_string_tuple(
            metadata["head_sites"],
            label="artifact state head_sites",
        )
        head_sha256s = _metadata_string_tuple(
            metadata["head_artifact_sha256s"],
            label="artifact state head_artifact_sha256s",
        )
        if (
            len(head_sites) != head_count
            or len(head_sha256s) != head_count
            or head_sites != tuple(sorted(set(head_sites)))
        ):
            raise ValueError("artifact state head index is invalid")
        for site in head_sites:
            _canonical_head_site(site)
        for value in head_sha256s:
            _require_sha256(value, label="serialized head artifact")
        heads: list[GemmaCausalResidualHead] = []
        expected_state_keys = {_STATE_METADATA_KEY}
        for index in range(head_count):
            prefix = f"heads.{index}."
            nested_state = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            head = GemmaCausalResidualHead.from_state_dict(
                nested_state
            )
            heads.append(head)
            expected_state_keys.update(
                f"{prefix}{name}" for name in nested_state
            )
        _strict_tensor_state(
            state,
            expected=frozenset(expected_state_keys),
            label="two-head artifact state",
        )
        if (
            tuple(head.site for head in heads) != head_sites
            or tuple(head.artifact_sha256 for head in heads)
            != head_sha256s
        ):
            raise ValueError("serialized artifact head binding differs")
        result = cls(
            parent_artifact_sha256=_require_sha256(
                metadata["parent_artifact_sha256"],
                label="serialized parent artifact",
            ),
            parent_receipt_sha256=_require_sha256(
                metadata["parent_receipt_sha256"],
                label="serialized parent receipt",
            ),
            residual_map_sha256=_require_sha256(
                metadata["residual_map_sha256"],
                label="serialized residual map",
            ),
            analysis_artifact_sha256=_require_sha256(
                metadata["analysis_artifact_sha256"],
                label="serialized residual analysis",
            ),
            bridge_binding_sha256=_require_sha256(
                metadata["bridge_binding_sha256"],
                label="serialized bridge binding",
            ),
            live_model_sha256=_require_sha256(
                metadata["live_model_sha256"],
                label="serialized live model",
            ),
            adapter_execution_sha256=_require_sha256(
                metadata["adapter_execution_sha256"],
                label="serialized adapter execution",
            ),
            heads=tuple(heads),
            recipe_sha256=_require_sha256(
                metadata["recipe_sha256"],
                label="serialized recipe",
            ),
            artifact_sha256=_require_sha256(
                metadata["artifact_sha256"],
                label="serialized two-head artifact",
            ),
        )
        if (
            result.execution_sha256
            != _require_sha256(
                metadata["execution_sha256"],
                label="serialized candidate execution",
            )
            or result.runtime_binding_sha256
            != _require_sha256(
                metadata["runtime_binding_sha256"],
                label="serialized runtime binding",
            )
        ):
            raise ValueError(
                "serialized artifact execution or runtime binding differs"
            )
        result.validate_integrity()
        return result


class GemmaL3L4TwoHeadExecutable:
    """One-prefill executor plus source-authoritative A-panel observer."""

    def __init__(
        self,
        *,
        adapter: Gemma3CausalLMAdapter,
        shadow_runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        bridge: Gemma3L3L4OnePassBridge,
        source_probe: LegacyRank64GemmaProgressiveExecutable,
        artifact: GemmaL3L4TwoHeadArtifact,
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(
            shadow_runtime,
            Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        ):
            raise TypeError("shadow_runtime must be the locked runtime")
        if not isinstance(bridge, Gemma3L3L4OnePassBridge):
            raise TypeError("bridge must be a one-pass bridge")
        if not isinstance(
            source_probe,
            LegacyRank64GemmaProgressiveExecutable,
        ):
            raise TypeError("source_probe must be the rank-64 measurement probe")
        if not isinstance(artifact, GemmaL3L4TwoHeadArtifact):
            raise TypeError("artifact must be a two-head artifact")
        self._adapter = adapter
        self._shadow_runtime = shadow_runtime
        self._bridge = bridge
        self._source_probe = source_probe
        self._artifact = artifact
        self._objective = CausalLanguageModelNLL()
        self._authenticate()

    @property
    def candidate_artifact_sha256(self) -> str:
        return self._artifact.artifact_sha256

    @property
    def candidate_execution_sha256(self) -> str:
        return self._artifact.execution_sha256

    @property
    def runtime_binding_sha256(self) -> str:
        return self._artifact.runtime_binding_sha256

    def _authenticate(self, *, check_adapter: bool = True) -> None:
        self._artifact.validate_integrity()
        self._bridge.validate_integrity()
        self._shadow_runtime.validate_integrity()
        try:
            self._source_probe.validate_integrity()
        except Exception as error:
            raise ValueError(
                "source probe authentication failed"
            ) from error
        if (
            self._artifact.bridge_binding_sha256
            != self._bridge.bridge_binding_sha256
            or self._artifact.live_model_sha256
            != self._shadow_runtime.live_model_sha256
            or self._artifact.adapter_execution_sha256
            != self._shadow_runtime.adapter_execution_sha256
            or self._source_probe.candidate_artifact_sha256
            != self._shadow_runtime.candidate_artifact_sha256
            or self._source_probe.candidate_execution_sha256
            != self._shadow_runtime.adapter_execution_sha256
            or self._source_probe.runtime_binding_sha256
            != self._shadow_runtime.runtime_binding_sha256
        ):
            raise ValueError("two-head executable lineage differs")
        if check_adapter and (
            self._adapter.model_fingerprint()
            != self._artifact.live_model_sha256
            or self._adapter.execution_fingerprint()
            != self._artifact.adapter_execution_sha256
        ):
            raise ValueError("live two-head adapter state differs")

    def execute(
        self,
        model_inputs: Mapping[str, Tensor],
    ) -> Gemma3L3L4OnePassExecution:
        # The bridge authenticates the live adapter before and after its
        # forward. Avoid repeating those full-model fingerprints here.
        self._authenticate(check_adapter=False)
        x4 = self._artifact.head(_X4_SITE)
        h4 = self._artifact.head(_H4_SITE)
        result = self._bridge.execute(
            self._adapter,
            model_inputs,
            x4_head=x4,
            h4_head=h4,
        )
        if result.model_forward_count != 1:
            raise RuntimeError("two-head prefill execution was not one-pass")
        self._authenticate(check_adapter=False)
        return result

    def _candidate_h4_loss_gradient(
        self,
        example: GemmaProgressiveExample,
        *,
        expected: Gemma3L3L4OnePassExecution,
    ) -> Tensor:
        x4 = self._artifact.head(_X4_SITE)
        h4 = self._artifact.head(_H4_SITE)
        measured, gradient = self._bridge.execute_h4_vjp(
            self._adapter,
            example.batch.model_inputs,
            x4_head=x4,
            h4_head=h4,
            objective=lambda run: self._objective(run, example.batch),
        )
        if measured.artifact_sha256 != expected.artifact_sha256:
            raise RuntimeError(
                "candidate-conditioned H4 VJP pass differs from the "
                "authenticated one-pass execution"
            )
        return (
            gradient[0]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )

    @staticmethod
    def _gather_logits(logits: Tensor, indices: Tensor) -> Tensor:
        return (
            logits[0]
            .index_select(0, indices.to(logits.device))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )

    def observe(
        self,
        example: GemmaProgressiveExample,
        *,
        collect_carrier_fisher: bool,
    ) -> GemmaL3L4DevelopmentObservation:
        if not isinstance(example, GemmaProgressiveExample):
            raise TypeError("example must be a GemmaProgressiveExample")
        example.validate_integrity()
        base = self._source_probe.observe(
            example,
            collect_carrier_fisher=False,
        )
        with torch.no_grad():
            execution = self.execute(example.batch.model_inputs)
        prefix = execution.prefix
        input_ids = example.batch.model_inputs["input_ids"]
        indices, derived_targets = (
            derive_gemma3_l3_l4_supervised_boundary(
                input_ids,
                prefix.valid_target_mask,
            )
        )
        affected_supervised = (
            prefix.target_affected_mask[0]
            .detach()
            .to(device="cpu")
            .index_select(0, indices)
        )
        if not bool(affected_supervised.any()):
            raise ValueError(
                "example has no causally affected supervised tokens"
            )
        targets = derived_targets.contiguous()
        if (
            not torch.equal(targets, base.targets)
            or execution.model_inputs_sha256
            != example.model_inputs_sha256
            or int(prefix.valid_target_mask.sum()) != base.valid_target_rows
            or int(prefix.target_affected_mask.sum())
            != base.affected_target_rows
        ):
            raise ValueError("two-head observation grid differs from the probe")
        x4_delta = execution.candidate_x4 - execution.reference_x4
        encoded = self._shadow_runtime.encode_target_delta(
            torch.where(
                prefix.target_affected_mask.unsqueeze(-1),
                x4_delta,
                torch.zeros_like(x4_delta),
            )
        )
        affected = prefix.target_affected_mask.to(encoded.device)
        residual_rows: Tensor | None = None
        gradient_rows: Tensor | None = None
        complete_error: float | None = None
        fit_sequence: GemmaTwoHeadFitSequence | None = None
        if collect_carrier_fisher:
            source_fit = self._source_probe.observe(
                example,
                collect_carrier_fisher=True,
            )
            seed_trace = source_fit.two_head_fit_sequence
            if seed_trace is None:
                raise ValueError("source probe omitted its two-head fit trace")
            candidate_h4_loss_gradient = (
                self._candidate_h4_loss_gradient(
                    example,
                    expected=execution,
                )
            )
            fit_sequence = GemmaTwoHeadFitSequence(
                example_id=example.example_id,
                family_id=example.family_id,
                model_inputs_sha256=example.model_inputs_sha256,
                runtime_binding_sha256=self.runtime_binding_sha256,
                source_modes=seed_trace.source_modes.detach().clone(),
                logical_positions=(
                    seed_trace.logical_positions.detach().clone()
                ),
                valid_target_mask=(
                    seed_trace.valid_target_mask.detach().clone()
                ),
                source_eligible_mask=(
                    seed_trace.source_eligible_mask.detach().clone()
                ),
                target_affected_mask=(
                    seed_trace.target_affected_mask.detach().clone()
                ),
                native_x4=seed_trace.native_x4.detach().clone(),
                candidate_x4=(
                    execution.candidate_x4[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .contiguous()
                ),
                native_h4=seed_trace.native_h4.detach().clone(),
                candidate_h4=(
                    execution.candidate_h4[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .contiguous()
                ),
                x4_loss_gradient=(
                    seed_trace.x4_loss_gradient.detach().clone()
                ),
                h4_loss_gradient=(
                    seed_trace.h4_loss_gradient.detach().clone()
                ),
                candidate_h4_loss_gradient=(
                    candidate_h4_loss_gradient
                ),
            )
            residual_rows = fit_sequence.h4_residual_rows.contiguous()
            gradient_rows = candidate_h4_loss_gradient[
                fit_sequence.target_affected_mask
            ].contiguous()
            complete_error = (
                source_fit.complete_boundary_oracle_max_abs_logit_error
            )
        return GemmaL3L4DevelopmentObservation(
            example_id=example.example_id,
            family_id=example.family_id,
            model_inputs_sha256=example.model_inputs_sha256,
            runtime_binding_sha256=self.runtime_binding_sha256,
            source_logits=base.source_logits,
            candidate_logits=self._gather_logits(
                execution.logits,
                indices,
            ),
            projection_oracle_logits=base.projection_oracle_logits,
            carrier_oracle_logits=base.carrier_oracle_logits,
            targets=targets,
            source_target_modes=base.source_target_modes,
            candidate_target_modes=(
                encoded[affected]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            source_full_width_delta=base.source_full_width_delta,
            projection_full_width_delta=(
                base.projection_full_width_delta
            ),
            valid_target_rows=base.valid_target_rows,
            affected_target_rows=base.affected_target_rows,
            carrier_residual_rows=residual_rows,
            carrier_loss_gradient_rows=gradient_rows,
            complete_boundary_oracle_max_abs_logit_error=complete_error,
            two_head_fit_sequence=fit_sequence,
        )


@dataclass(frozen=True, slots=True)
class _PlannedCandidate:
    parent_receipt_sha256: str
    proposal_receipt_sha256: str
    artifact: GemmaL3L4TwoHeadArtifact
    resources: ProgressiveResourceFootprint


class GemmaL3L4TwoHeadMutationLowerer:
    """Deterministic sequential X4/H4 lowerer for the progressive worker."""

    def __init__(
        self,
        *,
        adapter: Gemma3CausalLMAdapter,
        shadow_runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        source_probe: LegacyRank64GemmaProgressiveExecutable,
        head_rank: int = 16,
        lag_count: int = 32,
        ridge: float = 1.0e-6,
        h4_fit_objective: ResidualHeadFitObjective = (
            "hidden_residual_ridge"
        ),
        h4_conditioning: ResidualHeadConditioning = "l3_source_modes",
        proposal_schedule: str = "parallel_single_heads",
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(
            shadow_runtime,
            Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        ):
            raise TypeError("shadow_runtime must be the locked runtime")
        if not isinstance(
            source_probe,
            LegacyRank64GemmaProgressiveExecutable,
        ):
            raise TypeError("source_probe must be the legacy measurement probe")
        self._adapter = adapter
        self._shadow_runtime = shadow_runtime
        self._source_probe = source_probe
        self._bridge = shadow_runtime.export_one_pass_bridge()
        self._head_rank = _positive_int(head_rank, label="head_rank")
        self._lag_count = _positive_int(lag_count, label="lag_count")
        self._ridge = _positive_finite(ridge, label="ridge")
        self._h4_fit_objective = _canonical_fit_objective(
            h4_fit_objective
        )
        self._h4_conditioning = _canonical_head_conditioning(
            h4_conditioning
        )
        if proposal_schedule not in (
            "parallel_single_heads",
            "x4_then_h4",
        ):
            raise ValueError(
                "proposal_schedule must be parallel_single_heads or "
                "x4_then_h4"
            )
        self._proposal_schedule = proposal_schedule
        self._artifacts: dict[str, GemmaL3L4TwoHeadArtifact] = {}
        self._candidate_receipts: dict[str, str] = {}
        self._plans: dict[str, _PlannedCandidate] = {}
        self._authenticate()

    @property
    def bridge(self) -> Gemma3L3L4OnePassBridge:
        return self._bridge

    def artifact_for(
        self,
        candidate: ProgressiveCandidate,
    ) -> GemmaL3L4TwoHeadArtifact:
        """Export an isolated authenticated artifact for one built candidate."""

        if not isinstance(candidate, ProgressiveCandidate):
            raise TypeError("candidate must be a ProgressiveCandidate")
        self._authenticate()
        try:
            artifact = self._artifacts[candidate.artifact_sha256]
            candidate_receipt = self._candidate_receipts[
                candidate.artifact_sha256
            ]
        except KeyError as error:
            raise ValueError(
                "candidate artifact was not built by this lowerer"
            ) from error
        artifact.validate_integrity()
        if (
            candidate_receipt != candidate.receipt_sha256
            or artifact.artifact_sha256 != candidate.artifact_sha256
            or artifact.execution_sha256 != candidate.execution_sha256
            or artifact.runtime_binding_sha256
            != candidate.runtime_binding_sha256
        ):
            raise ValueError(
                "candidate receipt or executable artifact binding differs"
            )
        exported = GemmaL3L4TwoHeadArtifact.from_state_dict(
            artifact.state_dict()
        )
        self._authenticate()
        return exported

    def _authenticate(self) -> None:
        self._bridge.validate_integrity()
        self._shadow_runtime.validate_integrity()
        try:
            self._source_probe.validate_integrity()
        except Exception as error:
            raise ValueError(
                "source probe authentication failed"
            ) from error
        if (
            self._bridge.parent_runtime_binding_sha256
            != self._shadow_runtime.runtime_binding_sha256
            or self._source_probe.candidate_artifact_sha256
            != self._shadow_runtime.candidate_artifact_sha256
            or self._source_probe.candidate_execution_sha256
            != self._shadow_runtime.adapter_execution_sha256
            or self._source_probe.runtime_binding_sha256
            != self._shadow_runtime.runtime_binding_sha256
            or self._adapter.model_fingerprint()
            != self._shadow_runtime.live_model_sha256
            or self._adapter.execution_fingerprint()
            != self._shadow_runtime.adapter_execution_sha256
        ):
            raise ValueError("lowerer model or bridge lineage differs")

    @staticmethod
    def _unique_model_resources(
        model: nn.Module,
    ) -> tuple[int, int, int]:
        parameters = tuple(model.parameters())
        logical = sum(int(value.numel()) for value in parameters)
        runtime_bytes = sum(
            int(value.numel() * value.element_size())
            for value in parameters
        )
        linear_macs = sum(
            int(module.weight.numel())
            for module in model.modules()
            if isinstance(module, nn.Linear)
        )
        if logical <= 0 or runtime_bytes <= 0 or linear_macs <= 0:
            raise ValueError("retained Gemma resource scope is empty")
        return logical, runtime_bytes, linear_macs

    def _resources(
        self,
        *,
        parent: ProgressiveCandidate,
        artifact: GemmaL3L4TwoHeadArtifact,
    ) -> ProgressiveResourceFootprint:
        model_parameters, model_bytes, model_macs = (
            self._unique_model_resources(self._adapter.module)
        )
        if (
            parent.resources.parameter_scope
            != GEMMA_TWO_HEAD_PARAMETER_SCOPE
            or parent.resources.compute_scope
            != GEMMA_TWO_HEAD_COMPUTE_SCOPE
            or parent.resources.runtime_id != GEMMA_TWO_HEAD_RUNTIME_ID
            or parent.resources.runtime_dtype
            != GEMMA_TWO_HEAD_RUNTIME_DTYPE
        ):
            raise ValueError(
                "parent resource scopes/runtime must use the audited "
                "two-head definitions"
            )
        bridge_float_bytes = (
            self._bridge.prepared_float_scalar_count
            * torch.tensor([], dtype=torch.float64).element_size()
        )
        bridge_integer_bytes = (
            self._bridge.prepared_runtime_parameter_bytes
            - bridge_float_bytes
        )
        head_floats = artifact.prepared_float_scalar_count
        accounting = {
            "format_version": 1,
            "execution_sha256": artifact.execution_sha256,
            "scope": {
                "parameter": parent.resources.parameter_scope,
                "compute": parent.resources.compute_scope,
                "runtime_id": parent.resources.runtime_id,
                "runtime_dtype": parent.resources.runtime_dtype,
                "sequence_sha256": (
                    parent.resources.sequence_scope_sha256
                ),
            },
            "retained_model": {
                "unique_parameter_count": model_parameters,
                "runtime_parameter_bytes": model_bytes,
                "linear_weight_macs_per_token": model_macs,
            },
            "compiled_bridge": {
                "prepared_float_scalar_count": (
                    self._bridge.prepared_float_scalar_count
                ),
                "prepared_integer_value_count": (
                    self._bridge.prepared_integer_value_count
                ),
                "runtime_parameter_bytes": (
                    self._bridge.prepared_runtime_parameter_bytes
                ),
                "logical_macs_per_token_upper_bound": (
                    self._bridge.logical_macs_per_token_upper_bound
                ),
            },
            "residual_heads": {
                "prepared_float_scalar_count": head_floats,
                "logical_macs_per_token_upper_bound": (
                    artifact.logical_macs_per_token_upper_bound
                ),
            },
            "serving_model_forward_count": 1,
            "native_boundary_fallback": False,
            "measurement_passes_excluded_from_serving_cost": True,
        }
        return ProgressiveResourceFootprint(
            candidate_execution_sha256=artifact.execution_sha256,
            accounting_artifact_sha256=_sha256(
                "resource-accounting",
                accounting,
            ),
            parameter_scope=GEMMA_TWO_HEAD_PARAMETER_SCOPE,
            compute_scope=GEMMA_TWO_HEAD_COMPUTE_SCOPE,
            runtime_id=GEMMA_TWO_HEAD_RUNTIME_ID,
            runtime_dtype=GEMMA_TWO_HEAD_RUNTIME_DTYPE,
            sequence_scope_sha256=(
                parent.resources.sequence_scope_sha256
            ),
            compiled_learned_parameters=(
                self._bridge.prepared_float_scalar_count + head_floats
            ),
            retained_source_learned_parameters=model_parameters,
            support_learned_parameters=0,
            compiled_runtime_parameter_bytes=(
                bridge_float_bytes + head_floats * 8
            ),
            retained_source_runtime_parameter_bytes=model_bytes,
            support_runtime_parameter_bytes=bridge_integer_bytes,
            compiled_logical_macs_per_token=(
                self._bridge.logical_macs_per_token_upper_bound
                + artifact.logical_macs_per_token_upper_bound
            ),
            retained_source_logical_macs_per_token=model_macs,
            support_logical_macs_per_token=0,
            cost_complete=True,
            incomplete_cost_reasons=(),
        )

    def _parent_heads(
        self,
        parent: ProgressiveCandidate,
    ) -> dict[HeadSite, GemmaCausalResidualHead]:
        artifact = self._artifacts.get(parent.artifact_sha256)
        if artifact is None:
            if parent.mutation_kind != "seed":
                raise ValueError(
                    "non-seed parent was not built by this lowerer"
                )
            return {}
        artifact.validate_integrity()
        if artifact.runtime_binding_sha256 != parent.runtime_binding_sha256:
            raise ValueError("parent artifact and candidate runtime differ")
        return {head.site: head for head in artifact.heads}

    def _fit_artifact(
        self,
        *,
        parent: ProgressiveCandidate,
        residual_map: ResidualMap,
        analysis: GemmaCarrierResidualAnalysis,
        site: HeadSite,
        directions: Tensor,
    ) -> GemmaL3L4TwoHeadArtifact:
        existing = self._parent_heads(parent)
        if site in existing:
            raise ValueError("parent already contains this residual head")
        count = min(self._head_rank, int(directions.shape[0]))
        if count <= 0:
            raise ValueError("residual map exposes no direction for this head")
        selected = directions[:count].detach().clone()
        head = fit_gemma_causal_residual_head(
            site=site,
            sequences=analysis.fit_sequences,
            directions=selected,
            parent_runtime_binding_sha256=parent.runtime_binding_sha256,
            residual_map_sha256=residual_map.receipt_sha256,
            analysis_artifact_sha256=analysis.artifact_sha256,
            fit_manifest_sha256=analysis.fit_manifest_sha256,
            bridge_binding_sha256=self._bridge.bridge_binding_sha256,
            lag_count=self._lag_count,
            ridge=self._ridge,
            fit_objective=(
                self._h4_fit_objective
                if site == _H4_SITE
                else "hidden_residual_ridge"
            ),
            conditioning=(
                self._h4_conditioning
                if site == _H4_SITE
                else "l3_source_modes"
            ),
        )
        heads = tuple(
            sorted(
                (*existing.values(), head),
                key=lambda value: value.site,
            )
        )
        recipe = _sha256(
            "lowering-recipe",
            {
                "format_version": 1,
                "parent_artifact_sha256": parent.artifact_sha256,
                "parent_receipt_sha256": parent.receipt_sha256,
                "residual_map_sha256": residual_map.receipt_sha256,
                "analysis_artifact_sha256": analysis.artifact_sha256,
                "site": site,
                "head_rank": count,
                "lag_count": self._lag_count,
                "ridge": self._ridge,
                "fit_objective": head.fit_objective,
                "conditioning": head.conditioning,
                "head_sha256s": tuple(
                    value.artifact_sha256 for value in heads
                ),
                "joint_policy": (
                    "sequential_remeasure_after_first_head"
                ),
                "proposal_schedule": self._proposal_schedule,
            },
        )
        return GemmaL3L4TwoHeadArtifact(
            parent_artifact_sha256=parent.artifact_sha256,
            parent_receipt_sha256=parent.receipt_sha256,
            residual_map_sha256=residual_map.receipt_sha256,
            analysis_artifact_sha256=analysis.artifact_sha256,
            bridge_binding_sha256=self._bridge.bridge_binding_sha256,
            live_model_sha256=self._shadow_runtime.live_model_sha256,
            adapter_execution_sha256=(
                self._shadow_runtime.adapter_execution_sha256
            ),
            heads=heads,
            recipe_sha256=recipe,
        )

    def propose(
        self,
        *,
        parent: ProgressiveCandidate,
        residual_map: ResidualMap,
        analysis: GemmaCarrierResidualAnalysis,
        phase: ProgressivePhase,
    ) -> Sequence[MutationProposal]:
        self._authenticate()
        if phase != "repair":
            return ()
        if (
            residual_map.receipt_sha256
            != analysis.residual_map(iteration=parent.iteration).receipt_sha256
            or residual_map.analysis_artifact_sha256
            != analysis.artifact_sha256
            or residual_map.candidate_artifact_sha256
            != parent.artifact_sha256
            or residual_map.candidate_receipt_sha256
            != parent.receipt_sha256
            or not analysis.fit_sequences
            or analysis.x4_directions is None
        ):
            raise ValueError("lowerer residual map or private fit binding differs")
        existing = self._parent_heads(parent)
        candidates: list[
            tuple[HeadSite, str, str, Tensor, tuple[int, ...]]
        ] = []
        # A joint candidate is valid only in X4 -> remeasure -> H4 order.
        # Installing X4 underneath an already-fitted H4 head would change the
        # nonlinear carrier that the retained H4 kernel was trained against.
        if _X4_SITE not in existing and _H4_SITE not in existing:
            x4_count = min(
                self._head_rank,
                int(analysis.x4_directions.shape[0]),
            )
            start = int(analysis.directions.shape[0])
            candidates.append(
                (
                    _X4_SITE,
                    "add_residual_edge",
                    "x4",
                    analysis.x4_directions,
                    tuple(range(start, start + x4_count)),
                )
            )
        if (
            _H4_SITE not in existing
            and (
                self._proposal_schedule == "parallel_single_heads"
                or _X4_SITE in existing
            )
        ):
            h4_count = min(
                self._head_rank,
                int(analysis.directions.shape[0]),
            )
            candidates.append(
                (
                    _H4_SITE,
                    "widen_carrier",
                    "h4",
                    analysis.directions,
                    tuple(range(h4_count)),
                )
            )
        proposals: list[MutationProposal] = []
        for site, mutation, label, directions, ranks in candidates:
            artifact = self._fit_artifact(
                parent=parent,
                residual_map=residual_map,
                analysis=analysis,
                site=site,
                directions=directions,
            )
            resources = self._resources(parent=parent, artifact=artifact)
            proposal = MutationProposal(
                proposal_id=(
                    f"gemma-l3-l4-{label}-r"
                    f"{artifact.head(site).rank}-i{parent.iteration + 1}"
                ),
                phase="repair",
                mutation_kind=mutation,  # type: ignore[arg-type]
                parent_artifact_sha256=parent.artifact_sha256,
                parent_receipt_sha256=parent.receipt_sha256,
                residual_map_sha256=residual_map.receipt_sha256,
                recipe_sha256=artifact.recipe_sha256,
                target_ranks=ranks,
                resources=resources,
            )
            self._plans[proposal.receipt_sha256] = _PlannedCandidate(
                parent_receipt_sha256=parent.receipt_sha256,
                proposal_receipt_sha256=proposal.receipt_sha256,
                artifact=artifact,
                resources=resources,
            )
            proposals.append(proposal)
        self._authenticate()
        return tuple(proposals)

    def build(
        self,
        *,
        parent: ProgressiveCandidate,
        proposal: MutationProposal,
        analysis: GemmaCarrierResidualAnalysis,
    ) -> tuple[ProgressiveCandidate, GemmaProgressiveExecutable]:
        self._authenticate()
        try:
            planned = self._plans[proposal.receipt_sha256]
        except KeyError as error:
            raise ValueError("proposal was not fitted by this lowerer") from error
        artifact = planned.artifact
        artifact.validate_integrity()
        if (
            planned.proposal_receipt_sha256 != proposal.receipt_sha256
            or planned.parent_receipt_sha256 != parent.receipt_sha256
            or proposal.parent_artifact_sha256 != parent.artifact_sha256
            or proposal.parent_receipt_sha256 != parent.receipt_sha256
            or proposal.residual_map_sha256
            != artifact.residual_map_sha256
            or proposal.recipe_sha256 != artifact.recipe_sha256
            or proposal.resources.receipt_sha256
            != planned.resources.receipt_sha256
            or proposal.resources.receipt_sha256
            != self._resources(
                parent=parent,
                artifact=artifact,
            ).receipt_sha256
            or analysis.artifact_sha256
            != artifact.analysis_artifact_sha256
        ):
            raise ValueError("planned proposal, parent, or analysis differs")
        child = ProgressiveCandidate(
            candidate_id=(
                f"{parent.candidate_id}.{proposal.proposal_id}"
            ),
            iteration=parent.iteration + 1,
            artifact_sha256=artifact.artifact_sha256,
            execution_sha256=artifact.execution_sha256,
            runtime_binding_sha256=artifact.runtime_binding_sha256,
            resources=proposal.resources,
            mutation_kind=proposal.mutation_kind,
            parent_artifact_sha256=parent.artifact_sha256,
            proposal_sha256=proposal.receipt_sha256,
        )
        executable = GemmaL3L4TwoHeadExecutable(
            adapter=self._adapter,
            shadow_runtime=self._shadow_runtime,
            bridge=self._bridge,
            source_probe=self._source_probe,
            artifact=artifact,
        )
        if (
            executable.candidate_artifact_sha256
            != child.artifact_sha256
            or executable.candidate_execution_sha256
            != child.execution_sha256
            or executable.runtime_binding_sha256
            != child.runtime_binding_sha256
        ):
            raise RuntimeError("built child and executable identities differ")
        self._artifacts[child.artifact_sha256] = artifact
        self._candidate_receipts[child.artifact_sha256] = (
            child.receipt_sha256
        )
        self._authenticate()
        return child, executable
