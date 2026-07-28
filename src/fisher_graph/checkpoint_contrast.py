"""Tensor-free checkpoint localization for finite activation contrasts.

The diagnostic compares two activation states through an ordered,
differentiable checkpoint function.  It records:

* endpoint secants at every checkpoint;
* a midpoint Jacobian-vector product in the full endpoint direction;
* a contrast-aligned vector-Jacobian product back through every checkpoint;
* the input/output adjoint identity; and
* conservative, reason-coded localization.

This module deliberately does not fit an executor or choose a rank.  A weak
output contrast is reported as uninformative rather than being assigned to a
checkpoint.  Callers may therefore use it to investigate a failed test panel
without turning target-conditioned diagnostics into compiler inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import Tensor


__all__ = [
    "CheckpointContrastReport",
    "CheckpointContrastRow",
    "CheckpointContrastThresholds",
    "analyze_checkpoint_contrast",
]


CheckpointFunction = Callable[[Tensor], tuple[Tensor, ...]]


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _finite_positive(value: object, *, label: str) -> float:
    result = _finite_nonnegative(value, label=label)
    if result == 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": tuple(int(width) for width in tensor.shape),
        }
    )
    return hashlib.sha256(
        header + b"\0" + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _l2(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)))


def _cosine(left: Tensor, right: Tensor, *, floor: float) -> float | None:
    left64 = left.detach().to(torch.float64).flatten()
    right64 = right.detach().to(torch.float64).flatten()
    left_norm = float(torch.linalg.vector_norm(left64))
    right_norm = float(torch.linalg.vector_norm(right64))
    if left_norm <= floor or right_norm <= floor:
        return None
    return float(torch.dot(left64, right64) / (left_norm * right_norm))


def _masked(value: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return value
    selected = mask
    if selected.device != value.device:
        selected = selected.to(device=value.device)
    if selected.dtype != torch.bool:
        raise TypeError("output_mask must use torch.bool")
    if selected.shape == value.shape:
        return torch.where(selected, value, torch.zeros_like(value))
    if selected.shape == value.shape[:-1]:
        return torch.where(
            selected.unsqueeze(-1),
            value,
            torch.zeros_like(value),
        )
    raise ValueError(
        "output_mask must match the output or all output axes except feature"
    )


def _validate_outputs(
    outputs: object,
    *,
    checkpoint_names: tuple[str, ...],
    label: str,
) -> tuple[Tensor, ...]:
    if not isinstance(outputs, tuple):
        raise TypeError(f"{label} checkpoint function output must be a tuple")
    if len(outputs) != len(checkpoint_names):
        raise ValueError(
            f"{label} checkpoint count differs from checkpoint_names"
        )
    for name, value in zip(checkpoint_names, outputs, strict=True):
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"checkpoint {name!r} returned an invalid Tensor")
    return outputs


@dataclass(frozen=True, slots=True)
class CheckpointContrastThresholds:
    """Frozen numerical and classification thresholds for one diagnostic."""

    numeric_floor_epsilon_multiplier: float = 4.0
    resolved_noise_multiplier: float = 8.0
    maximum_linearization_relative_error: float = 0.25
    maximum_adjoint_relative_error: float = 1e-4
    maximum_causal_leakage_fraction: float = 1e-10
    localization_contraction_ratio: float = 0.25
    localization_dominance_ratio: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "numeric_floor_epsilon_multiplier",
            "resolved_noise_multiplier",
        ):
            _finite_positive(getattr(self, name), label=name)
        for name in (
            "maximum_linearization_relative_error",
            "maximum_adjoint_relative_error",
            "maximum_causal_leakage_fraction",
            "localization_contraction_ratio",
            "localization_dominance_ratio",
        ):
            value = _finite_nonnegative(getattr(self, name), label=name)
            if value > 1.0:
                raise ValueError(f"{name} must not exceed one")

    def state_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointContrastRow:
    """Scalar and hash-only evidence for one ordered checkpoint."""

    checkpoint_name: str
    left_sha256: str
    right_sha256: str
    midpoint_sha256: str
    jvp_sha256: str
    left_l2: float
    right_l2: float
    secant_l2: float
    repeat_noise_l2: float
    numeric_floor_l2: float
    symmetric_relative_separation: float
    resolved: bool
    jvp_l2: float
    midpoint_jvp_relative_response: float
    jvp_secant_cosine: float | None
    midpoint_linearization_relative_error: float
    vjp_l2: float | None
    vjp_secant_inner_product: float | None
    contrast_aligned_fraction: float | None
    cumulative_jvp_gain_from_input: float | None

    def state_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointContrastReport:
    """Authenticated, tensor-free result from one endpoint contrast."""

    checkpoint_names: tuple[str, ...]
    input_left_sha256: str
    input_right_sha256: str
    input_midpoint_sha256: str
    input_tangent_sha256: str
    output_checkpoint_index: int
    output_secant_l2: float
    output_symmetric_relative_separation: float
    output_resolved: bool
    output_jvp_secant_cosine: float | None
    output_midpoint_linearization_relative_error: float
    adjoint_left_inner_product: float
    adjoint_right_inner_product: float
    adjoint_relative_error: float
    causal_leakage_fraction: float
    classification: str
    localized_transition: str | None
    reason_codes: tuple[str, ...]
    rows: tuple[CheckpointContrastRow, ...]
    thresholds: CheckpointContrastThresholds
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_names or len(self.rows) != len(
            self.checkpoint_names
        ):
            raise ValueError("checkpoint contrast report has invalid rows")
        payload = self._payload()
        computed = hashlib.sha256(
            b"fisher-graph:checkpoint-contrast:v1\0"
            + _canonical_json_bytes(payload)
        ).hexdigest()
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("checkpoint contrast artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "checkpoint_names": list(self.checkpoint_names),
            "input_left_sha256": self.input_left_sha256,
            "input_right_sha256": self.input_right_sha256,
            "input_midpoint_sha256": self.input_midpoint_sha256,
            "input_tangent_sha256": self.input_tangent_sha256,
            "output_checkpoint_index": self.output_checkpoint_index,
            "output_secant_l2": self.output_secant_l2,
            "output_symmetric_relative_separation": (
                self.output_symmetric_relative_separation
            ),
            "output_resolved": self.output_resolved,
            "output_jvp_secant_cosine": self.output_jvp_secant_cosine,
            "output_midpoint_linearization_relative_error": (
                self.output_midpoint_linearization_relative_error
            ),
            "adjoint_left_inner_product": self.adjoint_left_inner_product,
            "adjoint_right_inner_product": self.adjoint_right_inner_product,
            "adjoint_relative_error": self.adjoint_relative_error,
            "causal_leakage_fraction": self.causal_leakage_fraction,
            "classification": self.classification,
            "localized_transition": self.localized_transition,
            "reason_codes": list(self.reason_codes),
            "rows": [row.state_dict() for row in self.rows],
            "thresholds": self.thresholds.state_dict(),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _causal_leakage_fraction(
    *,
    input_tangent: Tensor,
    output_jvp: Tensor,
    input_valid_mask: Tensor | None,
    output_mask: Tensor | None,
    logical_positions: Tensor | None,
) -> float:
    if (
        input_valid_mask is None
        or output_mask is None
        or logical_positions is None
    ):
        return 0.0
    valid = input_valid_mask.to(device="cpu", dtype=torch.bool)
    positions = logical_positions.detach().to(device="cpu", dtype=torch.int64)
    if valid.shape != positions.shape:
        raise ValueError("input_valid_mask and logical_positions must align")
    if input_tangent.shape[: valid.ndim] != valid.shape:
        raise ValueError("input_valid_mask does not match input_tangent")
    output_valid = output_mask.detach().to(device="cpu", dtype=torch.bool)
    if output_valid.shape == output_jvp.shape:
        output_valid = output_valid.any(dim=-1)
    if output_valid.shape != positions.shape:
        raise ValueError("output_mask does not match logical positions")

    tangent_energy = (
        input_tangent.detach().to(device="cpu", dtype=torch.float64)
        .square()
        .flatten(start_dim=valid.ndim)
        .sum(dim=-1)
    )
    active_source = valid & (tangent_energy > 0.0)
    if not bool(active_source.any()):
        return 0.0
    source_minimum = int(positions[active_source].min())
    response = (
        output_jvp.detach().to(device="cpu", dtype=torch.float64)
        .square()
        .flatten(start_dim=output_valid.ndim)
        .sum(dim=-1)
    )
    scored = output_valid
    total = float(response[scored].sum())
    if total == 0.0:
        return 0.0
    leaked = float(response[scored & (positions < source_minimum)].sum())
    return leaked / total


def analyze_checkpoint_contrast(
    checkpoint_function: CheckpointFunction,
    *,
    checkpoint_names: Sequence[str],
    left_input: Tensor,
    right_input: Tensor,
    output_checkpoint_index: int = -1,
    output_weight: Tensor | None = None,
    output_mask: Tensor | None = None,
    input_valid_mask: Tensor | None = None,
    logical_positions: Tensor | None = None,
    thresholds: CheckpointContrastThresholds = CheckpointContrastThresholds(),
) -> CheckpointContrastReport:
    """Analyze one finite contrast without retaining raw tensors.

    ``checkpoint_function`` must be deterministic and return the same ordered
    tuple for every call.  The midpoint JVP uses the complete finite endpoint
    direction, so its tangent is directly comparable to the endpoint secant.
    """

    if not callable(checkpoint_function):
        raise TypeError("checkpoint_function must be callable")
    names = tuple(checkpoint_names)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("checkpoint_names must be unique nonempty strings")
    if not isinstance(thresholds, CheckpointContrastThresholds):
        raise TypeError("thresholds must be CheckpointContrastThresholds")
    for label, value in (("left_input", left_input), ("right_input", right_input)):
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{label} must be a finite floating Tensor")
    if (
        left_input.shape != right_input.shape
        or left_input.dtype != right_input.dtype
        or left_input.device != right_input.device
    ):
        raise ValueError("contrast endpoint inputs must have identical layouts")
    output_index = output_checkpoint_index
    if output_index < 0:
        output_index += len(names)
    if not 0 <= output_index < len(names):
        raise ValueError("output_checkpoint_index is outside checkpoint_names")

    midpoint = (left_input + right_input) * 0.5
    tangent = right_input - left_input
    with torch.no_grad():
        left = _validate_outputs(
            checkpoint_function(left_input),
            checkpoint_names=names,
            label="left",
        )
        right = _validate_outputs(
            checkpoint_function(right_input),
            checkpoint_names=names,
            label="right",
        )
        left_repeat = _validate_outputs(
            checkpoint_function(left_input),
            checkpoint_names=names,
            label="left repeat",
        )
        right_repeat = _validate_outputs(
            checkpoint_function(right_input),
            checkpoint_names=names,
            label="right repeat",
        )
    for index, name in enumerate(names):
        expected = left[index]
        for collection in (right, left_repeat, right_repeat):
            value = collection[index]
            if (
                value.shape != expected.shape
                or value.dtype != expected.dtype
                or value.device != expected.device
            ):
                raise ValueError(f"checkpoint {name!r} layout changed across calls")

    with torch.enable_grad():
        midpoint_outputs, jvps = torch.autograd.functional.jvp(
            checkpoint_function,
            midpoint,
            tangent,
            create_graph=False,
            strict=True,
        )
    midpoint_outputs = _validate_outputs(
        midpoint_outputs,
        checkpoint_names=names,
        label="midpoint",
    )
    jvps = _validate_outputs(
        jvps,
        checkpoint_names=names,
        label="JVP",
    )

    output_left = _masked(left[output_index], output_mask)
    output_right = _masked(right[output_index], output_mask)
    output_delta = output_right - output_left
    output_scale = max(
        0.5 * (_l2(output_left) + _l2(output_right)),
        torch.finfo(output_delta.dtype).tiny,
    )
    output_floor = max(
        _l2(_masked(left_repeat[output_index] - left[output_index], output_mask)),
        _l2(_masked(right_repeat[output_index] - right[output_index], output_mask)),
        torch.finfo(output_delta.dtype).eps
        * thresholds.numeric_floor_epsilon_multiplier
        * output_scale,
    )
    output_delta_l2 = _l2(output_delta)
    if output_weight is None:
        cotangent = output_delta / max(output_delta_l2, output_floor)
    else:
        if (
            not isinstance(output_weight, Tensor)
            or output_weight.shape != left[output_index].shape
            or output_weight.dtype != left[output_index].dtype
            or output_weight.device != left[output_index].device
            or not bool(torch.isfinite(output_weight).all())
        ):
            raise ValueError("output_weight must match the output checkpoint")
        cotangent = _masked(output_weight, output_mask)

    midpoint_leaf = midpoint.detach().requires_grad_(True)
    with torch.enable_grad():
        vjp_outputs = _validate_outputs(
            checkpoint_function(midpoint_leaf),
            checkpoint_names=names,
            label="VJP midpoint",
        )
        scalar = (
            _masked(vjp_outputs[output_index], output_mask) * cotangent
        ).sum()
        input_gradient = torch.autograd.grad(
            scalar,
            midpoint_leaf,
            retain_graph=True,
            allow_unused=False,
        )[0]
        checkpoint_gradients = torch.autograd.grad(
            scalar,
            vjp_outputs,
            retain_graph=False,
            allow_unused=True,
        )

    rows: list[CheckpointContrastRow] = []
    input_jvp_l2 = max(_l2(tangent), torch.finfo(tangent.dtype).tiny)
    output_denominator = max(output_delta_l2, output_floor)
    for index, name in enumerate(names):
        finite_delta = right[index] - left[index]
        secant_l2 = _l2(finite_delta)
        left_l2 = _l2(left[index])
        right_l2 = _l2(right[index])
        scale = max(
            0.5 * (left_l2 + right_l2),
            torch.finfo(finite_delta.dtype).tiny,
        )
        repeat_noise = max(
            _l2(left_repeat[index] - left[index]),
            _l2(right_repeat[index] - right[index]),
        )
        floor = max(
            repeat_noise,
            torch.finfo(finite_delta.dtype).eps
            * thresholds.numeric_floor_epsilon_multiplier
            * scale,
        )
        gradient = checkpoint_gradients[index]
        vjp_l2 = None if gradient is None else _l2(gradient)
        inner = (
            None
            if gradient is None
            else float(
                torch.dot(
                    gradient.detach().to(torch.float64).flatten(),
                    finite_delta.detach().to(torch.float64).flatten(),
                )
            )
        )
        rows.append(
            CheckpointContrastRow(
                checkpoint_name=name,
                left_sha256=_tensor_sha256(left[index]),
                right_sha256=_tensor_sha256(right[index]),
                midpoint_sha256=_tensor_sha256(midpoint_outputs[index]),
                jvp_sha256=_tensor_sha256(jvps[index]),
                left_l2=left_l2,
                right_l2=right_l2,
                secant_l2=secant_l2,
                repeat_noise_l2=repeat_noise,
                numeric_floor_l2=floor,
                symmetric_relative_separation=secant_l2 / scale,
                resolved=(
                    secant_l2
                    > thresholds.resolved_noise_multiplier * floor
                ),
                jvp_l2=_l2(jvps[index]),
                midpoint_jvp_relative_response=(
                    _l2(jvps[index]) / scale
                ),
                jvp_secant_cosine=_cosine(
                    jvps[index],
                    finite_delta,
                    floor=floor,
                ),
                midpoint_linearization_relative_error=(
                    _l2(jvps[index] - finite_delta)
                    / max(secant_l2, floor)
                ),
                vjp_l2=vjp_l2,
                vjp_secant_inner_product=inner,
                contrast_aligned_fraction=(
                    None if inner is None else inner / output_denominator
                ),
                cumulative_jvp_gain_from_input=(
                    _l2(jvps[index]) / input_jvp_l2
                ),
            )
        )

    output_row = rows[output_index]
    output_jvp = _masked(jvps[output_index], output_mask)
    adjoint_left = float(
        torch.dot(
            input_gradient.detach().to(torch.float64).flatten(),
            tangent.detach().to(torch.float64).flatten(),
        )
    )
    adjoint_right = float(
        torch.dot(
            cotangent.detach().to(torch.float64).flatten(),
            output_jvp.detach().to(torch.float64).flatten(),
        )
    )
    adjoint_relative_error = abs(adjoint_left - adjoint_right) / max(
        abs(adjoint_left),
        abs(adjoint_right),
        output_floor,
    )
    causal_leakage = _causal_leakage_fraction(
        input_tangent=tangent,
        output_jvp=output_jvp,
        input_valid_mask=input_valid_mask,
        output_mask=output_mask,
        logical_positions=logical_positions,
    )

    reasons: list[str] = []
    localized: str | None = None
    if not output_row.resolved:
        classification = "uninformative_low_output_contrast"
        reasons.append("output_contrast_not_resolved_above_repeat_noise")
    elif (
        output_row.midpoint_linearization_relative_error
        > thresholds.maximum_linearization_relative_error
    ):
        classification = "nonlinear_or_finite_displacement"
        reasons.append("midpoint_jvp_does_not_match_endpoint_secant")
    elif adjoint_relative_error > thresholds.maximum_adjoint_relative_error:
        classification = "numerically_invalid"
        reasons.append("jvp_vjp_adjoint_identity_failed")
    elif causal_leakage > thresholds.maximum_causal_leakage_fraction:
        classification = "noncausal_or_invalid"
        reasons.append("jvp_has_response_before_earliest_changed_source")
    else:
        contractions: list[tuple[float, int]] = []
        for index in range(1, len(rows)):
            previous = rows[index - 1].midpoint_jvp_relative_response
            current = rows[index].midpoint_jvp_relative_response
            if previous > 0.0:
                contractions.append((current / previous, index))
        contractions.sort(key=lambda value: (value[0], value[1]))
        if (
            contractions
            and contractions[0][0]
            <= thresholds.localization_contraction_ratio
            and (
                len(contractions) == 1
                or contractions[0][0]
                <= (
                    contractions[1][0]
                    * thresholds.localization_dominance_ratio
                )
            )
        ):
            index = contractions[0][1]
            localized = f"{names[index - 1]} -> {names[index]}"
            classification = "localized_attenuation"
        else:
            classification = "distributed_or_inconclusive"
            reasons.append("no_single_validated_dominant_contraction")

    return CheckpointContrastReport(
        checkpoint_names=names,
        input_left_sha256=_tensor_sha256(left_input),
        input_right_sha256=_tensor_sha256(right_input),
        input_midpoint_sha256=_tensor_sha256(midpoint),
        input_tangent_sha256=_tensor_sha256(tangent),
        output_checkpoint_index=output_index,
        output_secant_l2=output_delta_l2,
        output_symmetric_relative_separation=(
            output_delta_l2 / output_scale
        ),
        output_resolved=output_row.resolved,
        output_jvp_secant_cosine=output_row.jvp_secant_cosine,
        output_midpoint_linearization_relative_error=(
            output_row.midpoint_linearization_relative_error
        ),
        adjoint_left_inner_product=adjoint_left,
        adjoint_right_inner_product=adjoint_right,
        adjoint_relative_error=adjoint_relative_error,
        causal_leakage_fraction=causal_leakage,
        classification=classification,
        localized_transition=localized,
        reason_codes=tuple(reasons),
        rows=tuple(rows),
        thresholds=thresholds,
    )
