"""Model-free local multimode flow for Gemma 3 MLP residual maps.

The first rung keeps attention native and replaces only a token-local MLP
residual map.  For a normalized MLP input ``x`` and residual carrier ``h``, a
fixed chart atlas supplies centers ``mu_i`` and orthonormal bases ``B_i``::

    z_i = B_i.T @ (x - mu_i)
    normal_squared_i = ||x - mu_i||^2 - ||z_i||^2
    q_i = softmax_i(-normal_squared_i / tau_q)
    pi_ik = softmax_k((a_ik @ z_i + b_ik) / tau_pi)
    t_ik = T_ik @ z_i + c_ik
    u_i = sum_k pi_ik t_ik
    G_i = B_i @ u_i
    v_hat = sum_i q_i G_i
    h_hat = h + v_hat

The module is deliberately independent of model hooks, corpora, clustering,
and optimizer policy.  Result objects expose tensors for fitting; their
receipts contain hashes, dimensions, closure errors, and resource counts but
never raw activations, weights, gradients, token IDs, or prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Final

import torch
from torch import Tensor, nn


__all__ = [
    "AffineLocalLayerMultimodeFlow",
    "FixedLayerChartAtlas",
    "GEMMA3_CHART_COUNT_LADDER",
    "GEMMA3_LAYER_COUNT",
    "GEMMA3_MODES_PER_CHART_LADDER",
    "LAYER_MULTIMODE_FLOW_EQUATION_ID",
    "LayerMultimodeFlowConfig",
    "LayerMultimodeFlowLoss",
    "LayerMultimodeFlowLossReceipt",
    "LayerMultimodeFlowReceipt",
    "LayerMultimodeFlowResult",
    "LayerMultimodeResourceAccounting",
    "ResidualReconstructionReceipt",
    "ResidualReconstructionResult",
    "layer_multimode_flow_loss",
    "reconstruct_residual_rows",
    "reconstruct_all_layer_residual_rows",
]


GEMMA3_LAYER_COUNT: Final = 18
GEMMA3_CHART_COUNT_LADDER: Final = (2, 1)
GEMMA3_MODES_PER_CHART_LADDER: Final = (4, 2, 1)
LAYER_MULTIMODE_FLOW_EQUATION_ID: Final = (
    "normal_residual_chart_softmax.affine_z_router."
    "affine_local_coordinate_generator.coordinate_fuse_single_chart_decode.v2"
)
_RESOURCE_FORMULA_ID: Final = (
    "retained_rmsnorm_fixed_atlas_affine_router_generator."
    "fused_coordinate_decode.matrix_mac_2chr_cmr_cmrr.v2"
)
_LOSS_FORMULA_ID: Final = (
    "weighted_mse_plus_chart_conditional_mode_balance_entropy_plus_"
    "chart_weighted_mode_cosine_square.v1"
)
_SCHEMA: Final = "fisher_graph.gemma3_layer_multimode_flow.v1"
_FORMAT_VERSION: Final = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-tensor:v1\0"
_ATLAS_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-atlas:v1\0"
_CONFIG_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-config:v1\0"
_MODEL_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-model:v1\0"
_RESOURCE_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-resource:v1\0"
_FLOW_RECEIPT_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-receipt:v1\0"
_RECONSTRUCTION_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode-flow-reconstruction:v1\0"
)
_LOSS_DOMAIN = b"fisher_graph:gemma3-layer-multimode-flow-loss:v1\0"
_FLOAT64_ORTHONORMAL_ATOL = 1.0e-10
_SIMPLEX_ATOL = 1.0e-10
_FLOW_TENSOR_NAMES = (
    "chart_inputs",
    "chart_velocities",
    "fused_chart_coordinates",
    "per_mode_coordinates",
    "pi",
    "predicted_output",
    "predicted_velocity",
    "q",
    "residual_states",
    "z",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _layer_ordinal(value: object) -> int:
    result = _strict_int(value, label="layer_ordinal")
    if result >= GEMMA3_LAYER_COUNT:
        raise ValueError("layer_ordinal must be in the 18-layer Gemma stack")
    return result


def _cpu_float64_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
    allow_requires_grad: bool = False,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must be on CPU")
    if value.dtype != torch.float64:
        raise ValueError(f"{label} must have dtype torch.float64")
    if value.requires_grad and not allow_requires_grad:
        raise ValueError(f"{label} must be detached")
    if value.ndim != ndim or any(int(width) <= 0 for width in value.shape):
        raise ValueError(f"{label} must be a nonempty rank-{ndim} tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _detached_cpu_float64(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    tensor = _cpu_float64_tensor(value, label=label, ndim=ndim)
    return tensor.detach().clone().contiguous()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"dtype": "float64_le", "shape": list(tensor.shape)}
        )
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _hash_pairs(value: tuple[tuple[str, str], ...], *, label: str) -> None:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a nonempty tuple")
    names: list[str] = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(f"{label} entries must be exact pairs")
        name, digest = pair
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} names must be nonempty")
        _require_sha256(digest, label=f"{label} digest")
        names.append(name)
    if tuple(names) != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError(f"{label} must use sorted unique names")


def _closure_statistics(
    actual: Tensor,
    expected: Tensor,
    *,
    absolute_floor: float = 1.0e-10,
    relative_factor: float = 1.0e-10,
) -> tuple[float, float, float]:
    residual = actual.detach().to(torch.float64) - expected.detach().to(
        torch.float64
    )
    maximum = float(residual.abs().max())
    expected64 = expected.detach().to(torch.float64)
    limit = absolute_floor + relative_factor * float(expected64.abs().max())
    nrmse = float(
        residual.square().mean().sqrt()
        / expected64.square().mean().sqrt().clamp_min(1.0e-15)
    )
    return maximum, limit, nrmse


def _tensor_free_safety() -> dict[str, bool]:
    return {
        "contains_activation_rows": False,
        "contains_gradient_rows": False,
        "contains_parameter_values": False,
        "contains_prompt_text": False,
        "contains_raw_tensors": False,
        "contains_token_ids": False,
    }


@dataclass(frozen=True, slots=True)
class FixedLayerChartAtlas:
    """Frozen per-layer centers and column-orthonormal tangent bases."""

    centers: Tensor
    bases: Tensor
    source_fit_sha256: str
    atlas_sha256: str = ""

    def __post_init__(self) -> None:
        centers = _detached_cpu_float64(
            self.centers,
            label="centers",
            ndim=2,
        )
        bases = _detached_cpu_float64(
            self.bases,
            label="bases",
            ndim=3,
        )
        _require_sha256(self.source_fit_sha256, label="source_fit_sha256")
        if bases.shape[:2] != centers.shape:
            raise ValueError("basis chart/hidden axes must match centers")
        if int(centers.shape[0]) not in GEMMA3_CHART_COUNT_LADDER:
            raise ValueError("chart count is outside the frozen C=(2,1) ladder")
        if bases.shape[2] > centers.shape[1]:
            raise ValueError("chart tangent rank cannot exceed hidden width")
        gram = torch.einsum("chr,chs->crs", bases, bases)
        identity = torch.eye(bases.shape[2], dtype=torch.float64).expand_as(gram)
        orthonormal_error = float((gram - identity).abs().max())
        if orthonormal_error > _FLOAT64_ORTHONORMAL_ATOL:
            raise ValueError("chart basis columns must be float64 orthonormal")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "bases", bases)
        computed = _digest(_ATLAS_DOMAIN, self._payload())
        if self.atlas_sha256 == "":
            object.__setattr__(self, "atlas_sha256", computed)
        elif _require_sha256(
            self.atlas_sha256,
            label="atlas_sha256",
        ) != computed:
            raise ValueError("fixed chart atlas hash mismatch")

    @property
    def chart_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.centers.shape[1])

    @property
    def tangent_rank(self) -> int:
        return int(self.bases.shape[2])

    @property
    def orthonormal_max_abs_error(self) -> float:
        gram = torch.einsum("chr,chs->crs", self.bases, self.bases)
        identity = torch.eye(self.tangent_rank, dtype=torch.float64).expand_as(gram)
        return float((gram - identity).abs().max())

    def _payload(self) -> dict[str, object]:
        return {
            "basis_sha256": _tensor_sha256(self.bases),
            "center_sha256": _tensor_sha256(self.centers),
            "chart_count": self.chart_count,
            "format_version": _FORMAT_VERSION,
            "hidden_width": self.hidden_width,
            "orthonormal_max_abs_error": self.orthonormal_max_abs_error,
            "schema": _SCHEMA,
            "source_fit_sha256": self.source_fit_sha256,
            "tangent_rank": self.tangent_rank,
        }

    def receipt(self) -> dict[str, object]:
        return {
            **self._payload(),
            "atlas_sha256": self.atlas_sha256,
            **_tensor_free_safety(),
        }

    def assert_integrity(self) -> None:
        if _digest(_ATLAS_DOMAIN, self._payload()) != self.atlas_sha256:
            raise ValueError("fixed chart atlas tensors changed after construction")


@dataclass(frozen=True, slots=True)
class LayerMultimodeFlowConfig:
    layer_ordinal: int
    modes_per_chart: int
    chart_temperature: float
    mode_temperature: float
    source_mlp_intermediate_width: int
    # The compiled map predicts the already post-output-RMSNorm teacher
    # velocity.  It therefore retains only the input RMSNorm used to form x;
    # the native output RMSNorm is folded into the teacher target and removed.
    retained_rmsnorm_vector_count: int = 1
    initialization_seed: int = 0
    fit_chart_temperature: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ordinal", _layer_ordinal(self.layer_ordinal))
        _strict_int(self.modes_per_chart, label="modes_per_chart", minimum=1)
        if self.modes_per_chart not in GEMMA3_MODES_PER_CHART_LADDER:
            raise ValueError("modes_per_chart is outside the frozen M=(4,2,1) ladder")
        object.__setattr__(
            self,
            "chart_temperature",
            _finite_float(
                self.chart_temperature,
                label="chart_temperature",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "mode_temperature",
            _finite_float(
                self.mode_temperature,
                label="mode_temperature",
                strictly_positive=True,
            ),
        )
        _strict_int(
            self.source_mlp_intermediate_width,
            label="source_mlp_intermediate_width",
            minimum=1,
        )
        _strict_int(
            self.retained_rmsnorm_vector_count,
            label="retained_rmsnorm_vector_count",
        )
        seed = _strict_int(self.initialization_seed, label="initialization_seed")
        if seed >= 2**63:
            raise ValueError("initialization_seed must fit signed int64")
        if type(self.fit_chart_temperature) is not bool:
            raise TypeError("fit_chart_temperature must be bool")

    def payload(self) -> dict[str, object]:
        return {
            "chart_temperature": self.chart_temperature,
            "fit_chart_temperature": self.fit_chart_temperature,
            "initialization_seed": self.initialization_seed,
            "layer_ordinal": self.layer_ordinal,
            "mode_temperature": self.mode_temperature,
            "modes_per_chart": self.modes_per_chart,
            "retained_rmsnorm_vector_count": self.retained_rmsnorm_vector_count,
            "source_mlp_intermediate_width": (
                self.source_mlp_intermediate_width
            ),
        }

    @property
    def config_sha256(self) -> str:
        return _digest(_CONFIG_DOMAIN, self.payload())


@dataclass(frozen=True, slots=True)
class LayerMultimodeResourceAccounting:
    hidden_width: int
    chart_count: int
    tangent_rank: int
    modes_per_chart: int
    source_mlp_intermediate_width: int
    retained_rmsnorm_parameter_count: int
    source_mlp_rmsnorm_parameter_count: int
    source_mlp_matrix_parameter_count: int
    source_total_parameter_count: int
    fixed_atlas_parameter_count: int
    chart_temperature_parameter_count: int
    router_parameter_count: int
    local_generator_parameter_count: int
    candidate_total_parameter_count: int
    net_parameter_savings: int
    source_matrix_macs_per_token: int
    candidate_matrix_and_mixture_macs_per_token: int
    net_macs_per_token: int
    formula_id: str = _RESOURCE_FORMULA_ID
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for label in (
            "hidden_width",
            "chart_count",
            "tangent_rank",
            "modes_per_chart",
            "source_mlp_intermediate_width",
            "source_mlp_matrix_parameter_count",
            "source_mlp_rmsnorm_parameter_count",
            "source_total_parameter_count",
            "fixed_atlas_parameter_count",
            "chart_temperature_parameter_count",
            "router_parameter_count",
            "local_generator_parameter_count",
            "candidate_total_parameter_count",
            "source_matrix_macs_per_token",
            "candidate_matrix_and_mixture_macs_per_token",
        ):
            _strict_int(getattr(self, label), label=label, minimum=1)
        _strict_int(
            self.retained_rmsnorm_parameter_count,
            label="retained_rmsnorm_parameter_count",
        )
        if type(self.net_parameter_savings) is not int:
            raise TypeError("net_parameter_savings must be an exact integer")
        if type(self.net_macs_per_token) is not int:
            raise TypeError("net_macs_per_token must be an exact integer")
        expected = _resource_counts(
            hidden_width=self.hidden_width,
            chart_count=self.chart_count,
            tangent_rank=self.tangent_rank,
            modes_per_chart=self.modes_per_chart,
            source_mlp_intermediate_width=self.source_mlp_intermediate_width,
            retained_rmsnorm_parameter_count=(
                self.retained_rmsnorm_parameter_count
            ),
        )
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("multimode resource formulas do not close")
        if self.formula_id != _RESOURCE_FORMULA_ID:
            raise ValueError("resource formula_id is invalid")
        computed = _digest(_RESOURCE_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="receipt_sha256",
        ) != computed:
            raise ValueError("resource receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            label: getattr(self, label)
            for label in (
                "candidate_matrix_and_mixture_macs_per_token",
                "candidate_total_parameter_count",
                "chart_temperature_parameter_count",
                "chart_count",
                "fixed_atlas_parameter_count",
                "formula_id",
                "hidden_width",
                "local_generator_parameter_count",
                "modes_per_chart",
                "net_macs_per_token",
                "net_parameter_savings",
                "retained_rmsnorm_parameter_count",
                "router_parameter_count",
                "source_matrix_macs_per_token",
                "source_mlp_intermediate_width",
                "source_mlp_matrix_parameter_count",
                "source_mlp_rmsnorm_parameter_count",
                "source_total_parameter_count",
                "tangent_rank",
            )
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}


def _resource_counts(
    *,
    hidden_width: int,
    chart_count: int,
    tangent_rank: int,
    modes_per_chart: int,
    source_mlp_intermediate_width: int,
    retained_rmsnorm_parameter_count: int,
) -> dict[str, int]:
    h = hidden_width
    c = chart_count
    r = tangent_rank
    k = modes_per_chart
    intermediate = source_mlp_intermediate_width
    retained = retained_rmsnorm_parameter_count
    source_matrices = 3 * h * intermediate
    atlas = c * h + c * h * r
    chart_temperature = 1
    router = c * k * (r + 1)
    local = c * k * (r * r + r)
    source_norms = 2 * h
    source_total = source_norms + source_matrices
    candidate_total = retained + atlas + chart_temperature + router + local
    source_macs = source_matrices
    # Exact deployed matrix-MAC ABI. Modes are fused in tangent coordinates,
    # so every chart performs one B_i decode rather than K H-wide decodes.
    # Nonlinearities, softmax, norms, adds, scalar distance reductions, and
    # coordinate-wise mixtures are disclosed but not called matrix MACs.
    candidate_macs = (
        c * h * r  # chart encode B_i^T (x-mu_i)
        + c * h * r  # one fused chart decode B_i u_i
        + c * k * r  # affine router
        + c * k * r * r  # affine local coordinate generator
    )
    return {
        "source_mlp_matrix_parameter_count": source_matrices,
        "source_mlp_rmsnorm_parameter_count": source_norms,
        "source_total_parameter_count": source_total,
        "fixed_atlas_parameter_count": atlas,
        "chart_temperature_parameter_count": chart_temperature,
        "router_parameter_count": router,
        "local_generator_parameter_count": local,
        "candidate_total_parameter_count": candidate_total,
        "net_parameter_savings": source_total - candidate_total,
        "source_matrix_macs_per_token": source_macs,
        "candidate_matrix_and_mixture_macs_per_token": candidate_macs,
        "net_macs_per_token": source_macs - candidate_macs,
    }


def _build_resource_accounting(
    atlas: FixedLayerChartAtlas,
    config: LayerMultimodeFlowConfig,
) -> LayerMultimodeResourceAccounting:
    retained = config.retained_rmsnorm_vector_count * atlas.hidden_width
    counts = _resource_counts(
        hidden_width=atlas.hidden_width,
        chart_count=atlas.chart_count,
        tangent_rank=atlas.tangent_rank,
        modes_per_chart=config.modes_per_chart,
        source_mlp_intermediate_width=config.source_mlp_intermediate_width,
        retained_rmsnorm_parameter_count=retained,
    )
    return LayerMultimodeResourceAccounting(
        hidden_width=atlas.hidden_width,
        chart_count=atlas.chart_count,
        tangent_rank=atlas.tangent_rank,
        modes_per_chart=config.modes_per_chart,
        source_mlp_intermediate_width=config.source_mlp_intermediate_width,
        retained_rmsnorm_parameter_count=retained,
        **counts,
    )


@dataclass(frozen=True, slots=True)
class ResidualReconstructionReceipt:
    layer_ordinal: int
    source_capture_sha256: str
    row_count: int
    hidden_width: int
    residual_state_sha256: str
    teacher_velocity_sha256: str
    expected_output_sha256: str
    reconstructed_output_sha256: str
    closure_max_abs: float
    closure_max_abs_limit: float
    closure_nrmse: float
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ordinal", _layer_ordinal(self.layer_ordinal))
        _require_sha256(self.source_capture_sha256, label="source_capture_sha256")
        _strict_int(self.row_count, label="row_count", minimum=1)
        _strict_int(self.hidden_width, label="hidden_width", minimum=1)
        for label in (
            "residual_state_sha256",
            "teacher_velocity_sha256",
            "expected_output_sha256",
            "reconstructed_output_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        maximum = _finite_float(
            self.closure_max_abs,
            label="closure_max_abs",
            minimum=0.0,
        )
        limit = _finite_float(
            self.closure_max_abs_limit,
            label="closure_max_abs_limit",
            strictly_positive=True,
        )
        nrmse = _finite_float(
            self.closure_nrmse,
            label="closure_nrmse",
            minimum=0.0,
        )
        if maximum > limit or nrmse > 1.0e-10:
            raise ValueError("captured h+v residual reconstruction does not close")
        computed = _digest(_RECONSTRUCTION_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="receipt_sha256",
        ) != computed:
            raise ValueError("residual reconstruction receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "closure_max_abs": self.closure_max_abs,
            "closure_max_abs_limit": self.closure_max_abs_limit,
            "closure_nrmse": self.closure_nrmse,
            "expected_output_sha256": self.expected_output_sha256,
            "format_version": _FORMAT_VERSION,
            "hidden_width": self.hidden_width,
            "layer_ordinal": self.layer_ordinal,
            "reconstructed_output_sha256": self.reconstructed_output_sha256,
            "residual_state_sha256": self.residual_state_sha256,
            "row_count": self.row_count,
            "schema": _SCHEMA,
            "source_capture_sha256": self.source_capture_sha256,
            "teacher_velocity_sha256": self.teacher_velocity_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "receipt_sha256": self.receipt_sha256,
            **_tensor_free_safety(),
        }


@dataclass(frozen=True, slots=True)
class ResidualReconstructionResult:
    reconstructed_output: Tensor
    receipt: ResidualReconstructionReceipt

    def __post_init__(self) -> None:
        self.assert_integrity()

    def assert_integrity(self) -> None:
        output = _cpu_float64_tensor(
            self.reconstructed_output,
            label="reconstructed_output",
            ndim=2,
            allow_requires_grad=True,
        )
        if (
            output.shape
            != (self.receipt.row_count, self.receipt.hidden_width)
            or _tensor_sha256(output)
            != self.receipt.reconstructed_output_sha256
        ):
            raise ValueError("residual reconstruction result changed after receipt")


def reconstruct_residual_rows(
    *,
    layer_ordinal: int,
    residual_states: Tensor,
    teacher_velocity: Tensor,
    expected_output: Tensor,
    source_capture_sha256: str,
) -> ResidualReconstructionResult:
    """Prove the captured local residual identity ``h' = h + v_T``."""

    layer = _layer_ordinal(layer_ordinal)
    states = _cpu_float64_tensor(
        residual_states,
        label="residual_states",
        ndim=2,
    )
    velocity = _cpu_float64_tensor(
        teacher_velocity,
        label="teacher_velocity",
        ndim=2,
    )
    expected = _cpu_float64_tensor(
        expected_output,
        label="expected_output",
        ndim=2,
    )
    _require_sha256(source_capture_sha256, label="source_capture_sha256")
    if states.shape != velocity.shape or states.shape != expected.shape:
        raise ValueError("h, v_T, and h' capture rows must have identical shapes")
    reconstructed = (states + velocity).contiguous()
    maximum, limit, nrmse = _closure_statistics(reconstructed, expected)
    receipt = ResidualReconstructionReceipt(
        layer_ordinal=layer,
        source_capture_sha256=source_capture_sha256,
        row_count=int(states.shape[0]),
        hidden_width=int(states.shape[1]),
        residual_state_sha256=_tensor_sha256(states),
        teacher_velocity_sha256=_tensor_sha256(velocity),
        expected_output_sha256=_tensor_sha256(expected),
        reconstructed_output_sha256=_tensor_sha256(reconstructed),
        closure_max_abs=maximum,
        closure_max_abs_limit=limit,
        closure_nrmse=nrmse,
    )
    result = ResidualReconstructionResult(reconstructed, receipt)
    result.assert_integrity()
    return result


def reconstruct_all_layer_residual_rows(
    *,
    residual_states: Tensor,
    teacher_velocities: Tensor,
    expected_outputs: Tensor,
    source_capture_sha256s: tuple[str, ...],
) -> tuple[ResidualReconstructionResult, ...]:
    """Apply the pure residual closure audit to all 18 physical layers."""

    states = _cpu_float64_tensor(
        residual_states,
        label="residual_states",
        ndim=3,
    )
    velocities = _cpu_float64_tensor(
        teacher_velocities,
        label="teacher_velocities",
        ndim=3,
    )
    outputs = _cpu_float64_tensor(
        expected_outputs,
        label="expected_outputs",
        ndim=3,
    )
    if (
        states.shape != velocities.shape
        or states.shape != outputs.shape
        or states.shape[0] != GEMMA3_LAYER_COUNT
    ):
        raise ValueError("all-layer residual rows must be aligned [18,N,H]")
    if (
        type(source_capture_sha256s) is not tuple
        or len(source_capture_sha256s) != GEMMA3_LAYER_COUNT
    ):
        raise ValueError("source_capture_sha256s must contain exactly 18 digests")
    return tuple(
        reconstruct_residual_rows(
            layer_ordinal=layer,
            residual_states=states[layer],
            teacher_velocity=velocities[layer],
            expected_output=outputs[layer],
            source_capture_sha256=source_capture_sha256s[layer],
        )
        for layer in range(GEMMA3_LAYER_COUNT)
    )


@dataclass(frozen=True, slots=True)
class LayerMultimodeFlowReceipt:
    layer_ordinal: int
    source_capture_sha256: str
    atlas_sha256: str
    config_sha256: str
    model_parameter_sha256: str
    resource_receipt_sha256: str
    equation_id: str
    row_count: int
    hidden_width: int
    chart_count: int
    tangent_rank: int
    modes_per_chart: int
    tensor_sha256s: tuple[tuple[str, str], ...]
    q_simplex_max_abs_error: float
    pi_simplex_max_abs_error: float
    velocity_closure_max_abs: float
    output_closure_max_abs: float
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ordinal", _layer_ordinal(self.layer_ordinal))
        for label in (
            "source_capture_sha256",
            "atlas_sha256",
            "config_sha256",
            "model_parameter_sha256",
            "resource_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.equation_id != LAYER_MULTIMODE_FLOW_EQUATION_ID:
            raise ValueError("multimode flow equation_id is invalid")
        for label in (
            "row_count",
            "hidden_width",
            "chart_count",
            "tangent_rank",
            "modes_per_chart",
        ):
            _strict_int(getattr(self, label), label=label, minimum=1)
        _hash_pairs(self.tensor_sha256s, label="tensor_sha256s")
        if tuple(name for name, _ in self.tensor_sha256s) != _FLOW_TENSOR_NAMES:
            raise ValueError("flow tensor hash inventory is not exact")
        if self.chart_count not in GEMMA3_CHART_COUNT_LADDER:
            raise ValueError("flow receipt chart_count is outside the ladder")
        if self.modes_per_chart not in GEMMA3_MODES_PER_CHART_LADDER:
            raise ValueError("flow receipt modes_per_chart is outside the ladder")
        q_error = _finite_float(
            self.q_simplex_max_abs_error,
            label="q_simplex_max_abs_error",
            minimum=0.0,
        )
        pi_error = _finite_float(
            self.pi_simplex_max_abs_error,
            label="pi_simplex_max_abs_error",
            minimum=0.0,
        )
        velocity_error = _finite_float(
            self.velocity_closure_max_abs,
            label="velocity_closure_max_abs",
            minimum=0.0,
        )
        output_error = _finite_float(
            self.output_closure_max_abs,
            label="output_closure_max_abs",
            minimum=0.0,
        )
        if (
            q_error > _SIMPLEX_ATOL
            or pi_error > _SIMPLEX_ATOL
            or velocity_error > 1.0e-10
            or output_error > 1.0e-10
        ):
            raise ValueError("multimode flow equation/simplex closure failed")
        computed = _digest(_FLOW_RECEIPT_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="receipt_sha256",
        ) != computed:
            raise ValueError("multimode flow receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "chart_count": self.chart_count,
            "config_sha256": self.config_sha256,
            "equation_id": self.equation_id,
            "deploy_forward_uses_fused_coordinates": True,
            "format_version": _FORMAT_VERSION,
            "hidden_width": self.hidden_width,
            "layer_ordinal": self.layer_ordinal,
            "model_parameter_sha256": self.model_parameter_sha256,
            "modes_per_chart": self.modes_per_chart,
            "normal_residual_uses_orthonormal_pythagorean_identity": True,
            "output_closure_max_abs": self.output_closure_max_abs,
            "pi_simplex_max_abs_error": self.pi_simplex_max_abs_error,
            "q_simplex_max_abs_error": self.q_simplex_max_abs_error,
            "resource_receipt_sha256": self.resource_receipt_sha256,
            "row_count": self.row_count,
            "schema": _SCHEMA,
            "source_capture_sha256": self.source_capture_sha256,
            "tangent_rank": self.tangent_rank,
            "tensor_sha256s": [list(value) for value in self.tensor_sha256s],
            "per_mode_hwide_velocities_materialized": False,
            "velocity_closure_max_abs": self.velocity_closure_max_abs,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "receipt_sha256": self.receipt_sha256,
            **_tensor_free_safety(),
        }


@dataclass(frozen=True, slots=True)
class LayerMultimodeFlowResult:
    z: Tensor
    q: Tensor
    pi: Tensor
    per_mode_coordinates: Tensor
    fused_chart_coordinates: Tensor
    chart_velocities: Tensor
    predicted_velocity: Tensor
    predicted_output: Tensor
    resource_accounting: LayerMultimodeResourceAccounting
    receipt: LayerMultimodeFlowReceipt

    def __post_init__(self) -> None:
        self.assert_integrity()

    @property
    def joint_routing(self) -> Tensor:
        return self.q.unsqueeze(-1) * self.pi

    def assert_integrity(self) -> None:
        shapes = {
            "z": (
                self.receipt.row_count,
                self.receipt.chart_count,
                self.receipt.tangent_rank,
            ),
            "q": (self.receipt.row_count, self.receipt.chart_count),
            "pi": (
                self.receipt.row_count,
                self.receipt.chart_count,
                self.receipt.modes_per_chart,
            ),
            "per_mode_coordinates": (
                self.receipt.row_count,
                self.receipt.chart_count,
                self.receipt.modes_per_chart,
                self.receipt.tangent_rank,
            ),
            "fused_chart_coordinates": (
                self.receipt.row_count,
                self.receipt.chart_count,
                self.receipt.tangent_rank,
            ),
            "chart_velocities": (
                self.receipt.row_count,
                self.receipt.chart_count,
                self.receipt.hidden_width,
            ),
            "predicted_velocity": (
                self.receipt.row_count,
                self.receipt.hidden_width,
            ),
            "predicted_output": (
                self.receipt.row_count,
                self.receipt.hidden_width,
            ),
        }
        tensors = {
            "z": self.z,
            "q": self.q,
            "pi": self.pi,
            "per_mode_coordinates": self.per_mode_coordinates,
            "fused_chart_coordinates": self.fused_chart_coordinates,
            "chart_velocities": self.chart_velocities,
            "predicted_velocity": self.predicted_velocity,
            "predicted_output": self.predicted_output,
        }
        receipt_hashes = dict(self.receipt.tensor_sha256s)
        for name, tensor in tensors.items():
            _cpu_float64_tensor(
                tensor,
                label=name,
                ndim=len(shapes[name]),
                allow_requires_grad=True,
            )
            if tuple(tensor.shape) != shapes[name]:
                raise ValueError(f"{name} shape drifted from flow receipt")
            if receipt_hashes.get(name) != _tensor_sha256(tensor):
                raise ValueError(f"{name} changed after flow receipt construction")
        if bool((self.q < 0.0).any()) or bool((self.q > 1.0).any()):
            raise ValueError("q must be a nonnegative simplex")
        if bool((self.pi < 0.0).any()) or bool((self.pi > 1.0).any()):
            raise ValueError("pi must be a nonnegative simplex")
        q_error = float((self.q.sum(dim=-1) - 1.0).abs().max().detach())
        pi_error = float((self.pi.sum(dim=-1) - 1.0).abs().max().detach())
        if not math.isclose(
            q_error,
            self.receipt.q_simplex_max_abs_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ) or not math.isclose(
            pi_error,
            self.receipt.pi_simplex_max_abs_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError("simplex evidence drifted from flow tensors")
        fused = torch.einsum(
            "nck,nckr->ncr",
            self.pi,
            self.per_mode_coordinates,
        )
        if (
            float(
                (fused - self.fused_chart_coordinates)
                .abs()
                .max()
                .detach()
            )
            > 1.0e-10
        ):
            raise ValueError("within-chart coordinate fusion does not close")
        velocity = torch.einsum(
            "nc,nch->nh",
            self.q,
            self.chart_velocities,
        )
        if (
            float(
                (velocity - self.predicted_velocity).abs().max().detach()
            )
            > 1.0e-10
        ):
            raise ValueError("chart velocity mixture does not close")
        if (
            self.resource_accounting.receipt_sha256
            != self.receipt.resource_receipt_sha256
        ):
            raise ValueError("resource accounting drifted from flow receipt")


class AffineLocalLayerMultimodeFlow(nn.Module):
    """First-rung affine router and affine local chart generators."""

    atlas: FixedLayerChartAtlas
    config: LayerMultimodeFlowConfig

    def __init__(
        self,
        *,
        atlas: FixedLayerChartAtlas,
        config: LayerMultimodeFlowConfig,
    ) -> None:
        super().__init__()
        if type(atlas) is not FixedLayerChartAtlas:
            raise TypeError("atlas must be a FixedLayerChartAtlas")
        if type(config) is not LayerMultimodeFlowConfig:
            raise TypeError("config must be a LayerMultimodeFlowConfig")
        atlas.assert_integrity()
        if atlas.chart_count == 1 and (
            config.chart_temperature != 1.0
            or config.fit_chart_temperature
        ):
            raise ValueError("C=1 requires serialized tau_c=1.0 and frozen")
        self.atlas = atlas
        self.config = config
        self.register_buffer("centers", atlas.centers.clone(), persistent=True)
        self.register_buffer("bases", atlas.bases.clone(), persistent=True)
        c = atlas.chart_count
        k = config.modes_per_chart
        r = atlas.tangent_rank
        self.router_weight = nn.Parameter(torch.empty(c, k, r, dtype=torch.float64))
        self.router_bias = nn.Parameter(torch.zeros(c, k, dtype=torch.float64))
        self.generator_weight = nn.Parameter(
            torch.empty(c, k, r, r, dtype=torch.float64)
        )
        self.generator_bias = nn.Parameter(
            torch.empty(c, k, r, dtype=torch.float64)
        )
        self.log_chart_temperature = nn.Parameter(
            torch.tensor(
                math.log(config.chart_temperature),
                dtype=torch.float64,
            ),
            requires_grad=config.fit_chart_temperature,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.initialization_seed)
        scale = 1.0 / math.sqrt(self.atlas.tangent_rank)
        with torch.no_grad():
            self.router_weight.copy_(
                torch.randn(
                    self.router_weight.shape,
                    dtype=torch.float64,
                    generator=generator,
                )
                * (0.05 * scale)
            )
            self.router_bias.zero_()
            self.generator_weight.copy_(
                torch.randn(
                    self.generator_weight.shape,
                    dtype=torch.float64,
                    generator=generator,
                )
                * (0.05 * scale)
            )
            self.generator_bias.copy_(
                torch.randn(
                    self.generator_bias.shape,
                    dtype=torch.float64,
                    generator=generator,
                )
                * 0.01
            )

    @property
    def resource_accounting(self) -> LayerMultimodeResourceAccounting:
        return _build_resource_accounting(self.atlas, self.config)

    @property
    def model_parameter_sha256(self) -> str:
        return _digest(
            _MODEL_DOMAIN,
            {
                "atlas_sha256": self.atlas.atlas_sha256,
                "config_sha256": self.config.config_sha256,
                "generator_bias_sha256": _tensor_sha256(self.generator_bias),
                "generator_weight_sha256": _tensor_sha256(
                    self.generator_weight
                ),
                "log_chart_temperature_sha256": _tensor_sha256(
                    self.log_chart_temperature
                ),
                "router_bias_sha256": _tensor_sha256(self.router_bias),
                "router_weight_sha256": _tensor_sha256(self.router_weight),
            },
        )

    def forward(
        self,
        *,
        chart_inputs: Tensor,
        residual_states: Tensor,
        source_capture_sha256: str,
    ) -> LayerMultimodeFlowResult:
        x = _cpu_float64_tensor(
            chart_inputs,
            label="chart_inputs",
            ndim=2,
            allow_requires_grad=True,
        )
        h = _cpu_float64_tensor(
            residual_states,
            label="residual_states",
            ndim=2,
            allow_requires_grad=True,
        )
        _require_sha256(source_capture_sha256, label="source_capture_sha256")
        if x.shape != h.shape or x.shape[1] != self.atlas.hidden_width:
            raise ValueError("chart inputs and residual states must align [N,H]")
        for label, parameter in self.named_parameters():
            _cpu_float64_tensor(
                parameter,
                label=label,
                ndim=parameter.ndim,
                allow_requires_grad=True,
            )
        self.atlas.assert_integrity()
        if not torch.equal(self.centers, self.atlas.centers) or not torch.equal(
            self.bases,
            self.atlas.bases,
        ):
            raise ValueError("module chart buffers drifted from frozen atlas")

        centered = x[:, None, :] - self.centers[None, :, :]
        z = torch.einsum("nch,chr->ncr", centered, self.bases)
        # B_i has orthonormal columns, so Pythagoras yields the squared
        # normal residual without a separate B_i z_i reconstruction.
        normal_squared = (
            centered.square().sum(dim=-1) - z.square().sum(dim=-1)
        ).clamp_min(0.0)
        chart_temperature = self.log_chart_temperature.exp()
        if not bool(torch.isfinite(chart_temperature)) or float(
            chart_temperature.detach()
        ) <= 0.0:
            raise ValueError("serialized chart temperature must remain positive")
        q = torch.softmax(
            -normal_squared / chart_temperature,
            dim=-1,
        )
        router_logits = (
            torch.einsum("ncr,ckr->nck", z, self.router_weight)
            + self.router_bias[None, :, :]
        )
        pi = torch.softmax(
            router_logits / self.config.mode_temperature,
            dim=-1,
        )
        per_mode_coordinates = (
            torch.einsum("ncr,cksr->ncks", z, self.generator_weight)
            + self.generator_bias[None, :, :, :]
        )
        fused_chart_coordinates = torch.einsum(
            "nck,nckr->ncr",
            pi,
            per_mode_coordinates,
        )
        chart_velocities = torch.einsum(
            "ncr,chr->nch",
            fused_chart_coordinates,
            self.bases,
        )
        predicted_velocity = torch.einsum(
            "nc,nch->nh",
            q,
            chart_velocities,
        )
        predicted_output = h + predicted_velocity

        q_error = float((q.sum(dim=-1) - 1.0).abs().max().detach())
        pi_error = float((pi.sum(dim=-1) - 1.0).abs().max().detach())
        # Independently replay the user's unfused sum without materializing
        # an [N,C,M,H] tensor.  The deployed path above remains one decode per
        # chart; this audit contraction proves algebraic equivalence to
        # sum_i q_i sum_k pi_ik B_i t_ik(z_i).
        velocity_check = torch.einsum(
            "nc,nck,nckr,chr->nh",
            q,
            pi,
            per_mode_coordinates,
            self.bases,
        )
        velocity_error = float(
            (predicted_velocity - velocity_check).abs().max().detach()
        )
        output_error = float(
            (predicted_output - (h + predicted_velocity)).abs().max().detach()
        )
        tensors = {
            "chart_inputs": x,
            "chart_velocities": chart_velocities,
            "fused_chart_coordinates": fused_chart_coordinates,
            "per_mode_coordinates": per_mode_coordinates,
            "pi": pi,
            "predicted_output": predicted_output,
            "predicted_velocity": predicted_velocity,
            "q": q,
            "residual_states": h,
            "z": z,
        }
        resources = self.resource_accounting
        receipt = LayerMultimodeFlowReceipt(
            layer_ordinal=self.config.layer_ordinal,
            source_capture_sha256=source_capture_sha256,
            atlas_sha256=self.atlas.atlas_sha256,
            config_sha256=self.config.config_sha256,
            model_parameter_sha256=self.model_parameter_sha256,
            resource_receipt_sha256=resources.receipt_sha256,
            equation_id=LAYER_MULTIMODE_FLOW_EQUATION_ID,
            row_count=int(x.shape[0]),
            hidden_width=self.atlas.hidden_width,
            chart_count=self.atlas.chart_count,
            tangent_rank=self.atlas.tangent_rank,
            modes_per_chart=self.config.modes_per_chart,
            tensor_sha256s=tuple(
                sorted((name, _tensor_sha256(value)) for name, value in tensors.items())
            ),
            q_simplex_max_abs_error=q_error,
            pi_simplex_max_abs_error=pi_error,
            velocity_closure_max_abs=velocity_error,
            output_closure_max_abs=output_error,
        )
        result = LayerMultimodeFlowResult(
            z=z,
            q=q,
            pi=pi,
            per_mode_coordinates=per_mode_coordinates,
            fused_chart_coordinates=fused_chart_coordinates,
            chart_velocities=chart_velocities,
            predicted_velocity=predicted_velocity,
            predicted_output=predicted_output,
            resource_accounting=resources,
            receipt=receipt,
        )
        result.assert_integrity()
        return result


@dataclass(frozen=True, slots=True)
class LayerMultimodeFlowLossReceipt:
    flow_receipt_sha256: str
    teacher_velocity_sha256: str
    row_weights_sha256: str
    coefficients: tuple[tuple[str, float], ...]
    term_values: tuple[tuple[str, float], ...]
    formula_id: str = _LOSS_FORMULA_ID
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for label in (
            "flow_receipt_sha256",
            "teacher_velocity_sha256",
            "row_weights_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.formula_id != _LOSS_FORMULA_ID:
            raise ValueError("flow-loss formula_id is invalid")
        expected_coefficients = (
            "distinctness",
            "fidelity",
            "load_balance",
            "sparsity",
        )
        expected_terms = (
            "distinctness",
            "fidelity",
            "load_balance",
            "sparsity",
            "total",
            "weighted_distinctness",
            "weighted_fidelity",
            "weighted_load_balance",
            "weighted_sparsity",
        )
        if (
            type(self.coefficients) is not tuple
            or tuple(name for name, _ in self.coefficients)
            != expected_coefficients
            or type(self.term_values) is not tuple
            or tuple(name for name, _ in self.term_values) != expected_terms
        ):
            raise ValueError("flow-loss term catalogs are invalid")
        coefficients = {
            name: _finite_float(value, label=f"{name} coefficient", minimum=0.0)
            for name, value in self.coefficients
        }
        if coefficients["fidelity"] <= 0.0:
            raise ValueError("fidelity coefficient must be strictly positive")
        terms = {
            name: _finite_float(value, label=f"{name} term", minimum=0.0)
            for name, value in self.term_values
        }
        for name in expected_coefficients:
            if not math.isclose(
                terms[f"weighted_{name}"],
                coefficients[name] * terms[name],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("weighted flow-loss term does not close")
        if not math.isclose(
            terms["total"],
            math.fsum(terms[f"weighted_{name}"] for name in expected_coefficients),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("total flow loss does not close")
        computed = _digest(_LOSS_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="receipt_sha256",
        ) != computed:
            raise ValueError("flow-loss receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "coefficients": [list(value) for value in self.coefficients],
            "flow_receipt_sha256": self.flow_receipt_sha256,
            "format_version": _FORMAT_VERSION,
            "formula_id": self.formula_id,
            "row_weights_sha256": self.row_weights_sha256,
            "schema": _SCHEMA,
            "teacher_velocity_sha256": self.teacher_velocity_sha256,
            "term_values": [list(value) for value in self.term_values],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "receipt_sha256": self.receipt_sha256,
            **_tensor_free_safety(),
        }


@dataclass(frozen=True, slots=True)
class LayerMultimodeFlowLoss:
    total: Tensor
    fidelity: Tensor
    load_balance: Tensor
    sparsity: Tensor
    distinctness: Tensor
    weighted_fidelity: Tensor
    weighted_load_balance: Tensor
    weighted_sparsity: Tensor
    weighted_distinctness: Tensor
    receipt: LayerMultimodeFlowLossReceipt


def _coefficient(value: object, *, label: str) -> float:
    return _finite_float(value, label=label, minimum=0.0)


def layer_multimode_flow_loss(
    *,
    result: LayerMultimodeFlowResult,
    teacher_velocity: Tensor,
    row_weights: Tensor | None = None,
    fidelity_coefficient: float = 1.0,
    load_balance_coefficient: float = 0.0,
    sparsity_coefficient: float = 0.0,
    distinctness_coefficient: float = 0.0,
) -> LayerMultimodeFlowLoss:
    """Return explicit fidelity and optional routing regularizer terms."""

    if type(result) is not LayerMultimodeFlowResult:
        raise TypeError("result must be a LayerMultimodeFlowResult")
    result.assert_integrity()
    teacher = _cpu_float64_tensor(
        teacher_velocity,
        label="teacher_velocity",
        ndim=2,
    )
    if teacher.shape != result.predicted_velocity.shape:
        raise ValueError("teacher velocity must match predicted [N,H] rows")
    n = result.receipt.row_count
    if row_weights is None:
        weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
    else:
        weights = _cpu_float64_tensor(
            row_weights,
            label="row_weights",
            ndim=1,
        )
        if weights.shape != (n,) or bool((weights < 0.0).any()):
            raise ValueError("row_weights must be nonnegative and row-aligned")
        weight_sum = weights.sum()
        if not bool(torch.isfinite(weight_sum)) or float(weight_sum) <= 0.0:
            raise ValueError("row_weights must have positive finite mass")
        weights = weights / weight_sum
    coefficients = {
        "distinctness": _coefficient(
            distinctness_coefficient,
            label="distinctness_coefficient",
        ),
        "fidelity": _coefficient(
            fidelity_coefficient,
            label="fidelity_coefficient",
        ),
        "load_balance": _coefficient(
            load_balance_coefficient,
            label="load_balance_coefficient",
        ),
        "sparsity": _coefficient(
            sparsity_coefficient,
            label="sparsity_coefficient",
        ),
    }
    if coefficients["fidelity"] <= 0.0:
        raise ValueError("fidelity_coefficient must be strictly positive")

    row_mse = (result.predicted_velocity - teacher).square().mean(dim=-1)
    fidelity = (weights * row_mse).sum()
    # Charts describe where a row lies in state geometry; the regularizers
    # must not force those chart masses toward uniformity or harden q.  Modes
    # describe dynamics *within* a chart, so balance and sparsity are computed
    # conditionally from pi and then averaged under the observed chart mass.
    # Stop mode-specialization objectives at the chart boundary.  Charts are
    # fit only by teacher fidelity; otherwise a conditional mode-balancing or
    # distinctness term could move tau_c indirectly.
    q_for_mode_regularizers = result.q.detach()
    joint = q_for_mode_regularizers[:, :, None] * result.pi
    chart_mass = (weights[:, None] * q_for_mode_regularizers).sum(dim=0)
    conditional_mode_mass = (
        weights[:, None, None] * joint
    ).sum(dim=0) / chart_mass[:, None].clamp_min(
        torch.finfo(torch.float64).tiny
    )
    mode_count = result.receipt.modes_per_chart
    target = 1.0 / mode_count
    load_balance = (
        chart_mass
        * mode_count
        * (conditional_mode_mass - target).square().mean(dim=-1)
    ).sum()
    pi_safe = result.pi.clamp_min(torch.finfo(torch.float64).tiny)
    if mode_count == 1:
        sparsity = result.predicted_velocity.new_zeros(())
    else:
        sparsity = (
            weights
            * (
                q_for_mode_regularizers
                * (-(result.pi * pi_safe.log()).sum(dim=-1))
            ).sum(dim=-1)
            / math.log(mode_count)
        ).sum()

    # B_i is orthonormal, so coordinate-space cosine is exactly the same as
    # H-wide G_ik cosine without materializing any per-mode H-wide velocity.
    normalized_modes = result.per_mode_coordinates / torch.linalg.vector_norm(
        result.per_mode_coordinates,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-12)
    cosine = torch.einsum(
        "nckh,nclh->nckl",
        normalized_modes,
        normalized_modes,
    )
    k = result.receipt.modes_per_chart
    if k == 1:
        distinctness = result.predicted_velocity.new_zeros(())
    else:
        pair_mask = torch.triu(
            torch.ones(k, k, dtype=torch.bool),
            diagonal=1,
        )
        per_row_chart_distinctness = cosine.square()[..., pair_mask].mean(dim=-1)
        distinctness = (
            weights[:, None]
            * q_for_mode_regularizers
            * per_row_chart_distinctness
        ).sum()

    weighted_fidelity = coefficients["fidelity"] * fidelity
    weighted_load_balance = coefficients["load_balance"] * load_balance
    weighted_sparsity = coefficients["sparsity"] * sparsity
    weighted_distinctness = coefficients["distinctness"] * distinctness
    total = (
        weighted_fidelity
        + weighted_load_balance
        + weighted_sparsity
        + weighted_distinctness
    )
    terms = {
        "distinctness": distinctness,
        "fidelity": fidelity,
        "load_balance": load_balance,
        "sparsity": sparsity,
        "total": total,
        "weighted_distinctness": weighted_distinctness,
        "weighted_fidelity": weighted_fidelity,
        "weighted_load_balance": weighted_load_balance,
        "weighted_sparsity": weighted_sparsity,
    }
    receipt = LayerMultimodeFlowLossReceipt(
        flow_receipt_sha256=result.receipt.receipt_sha256,
        teacher_velocity_sha256=_tensor_sha256(teacher),
        row_weights_sha256=_tensor_sha256(weights),
        coefficients=tuple(sorted(coefficients.items())),
        term_values=tuple(
            sorted((name, float(value.detach())) for name, value in terms.items())
        ),
    )
    return LayerMultimodeFlowLoss(
        total=total,
        fidelity=fidelity,
        load_balance=load_balance,
        sparsity=sparsity,
        distinctness=distinctness,
        weighted_fidelity=weighted_fidelity,
        weighted_load_balance=weighted_load_balance,
        weighted_sparsity=weighted_sparsity,
        weighted_distinctness=weighted_distinctness,
        receipt=receipt,
    )
