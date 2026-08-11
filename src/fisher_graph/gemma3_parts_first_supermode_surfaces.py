"""Exact operator-boundary surfaces for parts-first Gemma 3 supermodes.

This module does not learn a generator.  It exposes exact additive parts of
three native operators so a later experiment can discover *groups of parts*
and compose those groups as an ordered path through the model:

* tied lexical hidden columns at embedding ingress and raw-logit egress;
* query-head contributions after the attention output projection; and
* MLP intermediate-channel contributions after the down projection.

Only contributions at the same boundary may be summed.  A cross-boundary
supermode is an ordered program, not a sum of vectors from different depths.
Nonlinear normalization, routing, masking, RoPE, and softmax remain explicit
shared substrate and are outside the additive claims made here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor


PART_SURFACE_ABI = "fisher_graph.gemma3_parts_first_surface.v1"
PART_SURFACE_DOMAIN = b"fisher_graph.gemma3_parts_first_surface.v1\0"
DEFAULT_CLOSURE_ATOL = 1.0e-4
DEFAULT_CLOSURE_RTOL = 1.0e-5
DEFAULT_CLOSURE_NRMSE_LIMIT = 1.0e-5


def _require_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{label} must use float32 or float64 audit precision")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{label} must be rank {ndim}")
    if value.numel() == 0:
        raise ValueError(f"{label} must be nonempty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _require_integer_tensor(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{label} must be an integer Tensor")
    return value


def _canonical_groups(
    groups: Sequence[Sequence[int]],
    *,
    width: int,
    exhaustive: bool,
) -> tuple[tuple[int, ...], ...]:
    if type(width) is not int or width <= 0:
        raise ValueError("width must be positive")
    if not isinstance(groups, Sequence) or not groups:
        raise ValueError("groups must be a nonempty sequence")
    canonical: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group in groups:
        if not isinstance(group, Sequence) or not group:
            raise ValueError("every part group must be nonempty")
        values = tuple(group)
        if any(type(value) is not int for value in values):
            raise TypeError("part indices must be exact ints")
        if tuple(sorted(set(values))) != values:
            raise ValueError("part indices must be sorted and unique")
        if values[0] < 0 or values[-1] >= width:
            raise ValueError("part index is outside the operator width")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError("physical partition groups must be disjoint")
        seen.update(values)
        canonical.append(values)
    if exhaustive and seen != set(range(width)):
        raise ValueError("physical partition groups must be exhaustive")
    return tuple(canonical)


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        PART_SURFACE_ABI.encode("utf-8")
        + b"\0"
        + header
        + b"\0"
        + detached.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _membership_sha256(groups: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps(
        [list(group) for group in groups],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        PART_SURFACE_ABI.encode("utf-8") + b"\0groups\0" + payload
    ).hexdigest()


def _surface_sha256(
    *,
    operator_kind: str,
    boundary_id: str,
    reference_source: str,
    part_axis_width: int,
    membership_sha256: str,
    group_contribution_sha256s: tuple[str, ...],
    reconstructed_output_sha256: str,
    reference_output_sha256: str,
    max_abs_error: float,
    closure_atol: float,
    closure_rtol: float,
    closure_max_abs_limit: float,
    closure_nrmse: float,
    closure_nrmse_limit: float,
    additive_closure_passed: bool,
) -> str:
    payload = json.dumps(
        {
            "operator_kind": operator_kind,
            "boundary_id": boundary_id,
            "reference_source": reference_source,
            "part_axis_width": part_axis_width,
            "membership_sha256": membership_sha256,
            "group_contribution_sha256s": list(
                group_contribution_sha256s
            ),
            "reconstructed_output_sha256": reconstructed_output_sha256,
            "reference_output_sha256": reference_output_sha256,
            "max_abs_error": max_abs_error,
            "closure_atol": closure_atol,
            "closure_rtol": closure_rtol,
            "closure_max_abs_limit": closure_max_abs_limit,
            "closure_nrmse": closure_nrmse,
            "closure_nrmse_limit": closure_nrmse_limit,
            "additive_closure_passed": additive_closure_passed,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(PART_SURFACE_DOMAIN + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AdditivePartSurface:
    """One same-boundary partition and its numerical reconstruction evidence.

    ``provided_operator_output`` means the caller supplied an independent
    reference tensor.  This mathematical surface does not authenticate how
    that tensor was captured; the experiment layer must bind a typed native
    observer receipt before making a native-parity claim.
    """

    operator_kind: str
    boundary_id: str
    reference_source: str
    part_axis_width: int
    groups: tuple[tuple[int, ...], ...]
    group_contributions: tuple[Tensor, ...]
    reconstructed_output: Tensor
    reference_output: Tensor
    membership_sha256: str
    group_contribution_sha256s: tuple[str, ...]
    reconstructed_output_sha256: str
    reference_output_sha256: str
    max_abs_error: float
    closure_atol: float
    closure_rtol: float
    closure_max_abs_limit: float
    closure_nrmse: float
    closure_nrmse_limit: float
    additive_closure_passed: bool
    surface_sha256: str

    def __post_init__(self) -> None:
        cloned_contributions = tuple(
            contribution.detach().clone().contiguous()
            for contribution in self.group_contributions
        )
        cloned_reconstructed = (
            self.reconstructed_output.detach().clone().contiguous()
        )
        cloned_reference = self.reference_output.detach().clone().contiguous()
        object.__setattr__(
            self,
            "group_contributions",
            cloned_contributions,
        )
        object.__setattr__(
            self,
            "reconstructed_output",
            cloned_reconstructed,
        )
        object.__setattr__(self, "reference_output", cloned_reference)
        self.assert_integrity()

    def assert_integrity(self) -> None:
        """Recompute hashes and additive closure from the stored tensors."""

        if self.operator_kind not in {
            "tied_lexical_ingress",
            "tied_lexical_raw_egress",
            "attention_query_heads_after_O",
            "mlp_channels_after_down",
        }:
            raise ValueError("unsupported operator_kind")
        if not isinstance(self.boundary_id, str) or not self.boundary_id:
            raise ValueError("boundary_id must be nonempty")
        if self.reference_source not in {
            "provided_operator_output",
            "recomputed_formula",
        }:
            raise ValueError("reference_source is invalid")
        canonical = _canonical_groups(
            self.groups,
            width=self.part_axis_width,
            exhaustive=True,
        )
        if canonical != self.groups:
            raise ValueError("groups are not canonical")
        if len(self.group_contributions) != len(self.groups):
            raise ValueError("one contribution is required per group")
        if len(self.group_contribution_sha256s) != len(self.groups):
            raise ValueError("one contribution hash is required per group")
        reference = _require_float_tensor(
            self.reference_output,
            label="reference_output",
        )
        reconstructed = _require_float_tensor(
            self.reconstructed_output,
            label="reconstructed_output",
        )
        if reconstructed.shape != reference.shape:
            raise ValueError("reconstruction and reference shapes differ")
        for index, contribution in enumerate(self.group_contributions):
            checked = _require_float_tensor(
                contribution,
                label="group_contribution",
            )
            if checked.shape != reference.shape:
                raise ValueError("group contribution shape is inconsistent")
            if self.group_contribution_sha256s[index] != _tensor_sha256(checked):
                raise ValueError("group contribution hash does not close")
        expected_membership = _membership_sha256(self.groups)
        if self.membership_sha256 != expected_membership:
            raise ValueError("membership hash does not close")
        if self.reconstructed_output_sha256 != _tensor_sha256(reconstructed):
            raise ValueError("reconstructed output hash does not close")
        if self.reference_output_sha256 != _tensor_sha256(reference):
            raise ValueError("reference output hash does not close")
        if (
            not isinstance(self.max_abs_error, float)
            or not math.isfinite(self.max_abs_error)
            or self.max_abs_error < 0
        ):
            raise ValueError("max_abs_error must be finite and nonnegative")
        for label, tolerance in (
            ("closure_atol", self.closure_atol),
            ("closure_rtol", self.closure_rtol),
            ("closure_max_abs_limit", self.closure_max_abs_limit),
            ("closure_nrmse", self.closure_nrmse),
            ("closure_nrmse_limit", self.closure_nrmse_limit),
        ):
            if (
                not isinstance(tolerance, float)
                or not math.isfinite(tolerance)
                or tolerance < 0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")
        if type(self.additive_closure_passed) is not bool:
            raise TypeError("additive_closure_passed must be bool")
        derived_reconstruction = torch.stack(
            self.group_contributions,
            dim=0,
        ).sum(dim=0)
        if _tensor_sha256(derived_reconstruction) != _tensor_sha256(
            reconstructed
        ):
            raise ValueError("stored reconstruction is not the sum of parts")
        derived_error = float(
            (derived_reconstruction - reference).abs().max().item()
        )
        if self.max_abs_error != derived_error:
            raise ValueError("max_abs_error does not close")
        derived_max_abs_limit = float(
            self.closure_atol
            + self.closure_rtol * float(reference.abs().max().item())
        )
        if self.closure_max_abs_limit != derived_max_abs_limit:
            raise ValueError("closure_max_abs_limit does not close")
        reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
        derived_nrmse = float(
            ((derived_reconstruction - reference).square().mean().sqrt()
            / reference_rms).item()
        )
        if self.closure_nrmse != derived_nrmse:
            raise ValueError("closure_nrmse does not close")
        derived_pass = bool(
            derived_error <= derived_max_abs_limit
            and derived_nrmse <= self.closure_nrmse_limit
        )
        if self.additive_closure_passed is not derived_pass:
            raise ValueError("additive closure decision does not close")
        derived_surface_sha256 = _surface_sha256(
            operator_kind=self.operator_kind,
            boundary_id=self.boundary_id,
            reference_source=self.reference_source,
            part_axis_width=self.part_axis_width,
            membership_sha256=self.membership_sha256,
            group_contribution_sha256s=self.group_contribution_sha256s,
            reconstructed_output_sha256=self.reconstructed_output_sha256,
            reference_output_sha256=self.reference_output_sha256,
            max_abs_error=self.max_abs_error,
            closure_atol=self.closure_atol,
            closure_rtol=self.closure_rtol,
            closure_max_abs_limit=self.closure_max_abs_limit,
            closure_nrmse=self.closure_nrmse,
            closure_nrmse_limit=self.closure_nrmse_limit,
            additive_closure_passed=self.additive_closure_passed,
        )
        if self.surface_sha256 != derived_surface_sha256:
            raise ValueError("surface semantic hash does not close")

    def score_coordinates(self, operator_score_gradient: Tensor) -> Tensor:
        """Return per-group ``<part output, dScore/d operator output>``."""

        self.assert_integrity()
        gradient = _require_float_tensor(
            operator_score_gradient,
            label="operator_score_gradient",
        )
        if gradient.shape != self.reference_output.shape:
            raise ValueError("operator gradient shape does not match surface")
        leading = gradient.shape[:-1]
        values = tuple(
            (contribution * gradient).sum(dim=-1)
            for contribution in self.group_contributions
        )
        return torch.stack(values, dim=-1).reshape(*leading, len(values))


def _surface(
    *,
    operator_kind: str,
    boundary_id: str,
    reference_source: str,
    part_axis_width: int,
    groups: tuple[tuple[int, ...], ...],
    contributions: Iterable[Tensor],
    reference: Tensor,
    atol: float,
    rtol: float,
    nrmse_limit: float,
) -> AdditivePartSurface:
    parts = tuple(value.contiguous() for value in contributions)
    if not parts:
        raise ValueError("at least one contribution is required")
    reconstructed = torch.stack(parts, dim=0).sum(dim=0).contiguous()
    reference = reference.contiguous()
    error = float((reconstructed - reference).abs().max().item())
    max_abs_limit = float(atol + rtol * float(reference.abs().max().item()))
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    nrmse = float(
        ((reconstructed - reference).square().mean().sqrt() / reference_rms).item()
    )
    passed = bool(error <= max_abs_limit and nrmse <= nrmse_limit)
    contribution_sha256s = tuple(_tensor_sha256(value) for value in parts)
    membership_sha256 = _membership_sha256(groups)
    reconstructed_sha256 = _tensor_sha256(reconstructed)
    reference_sha256 = _tensor_sha256(reference)
    surface_sha256 = _surface_sha256(
        operator_kind=operator_kind,
        boundary_id=boundary_id,
        reference_source=reference_source,
        part_axis_width=part_axis_width,
        membership_sha256=membership_sha256,
        group_contribution_sha256s=contribution_sha256s,
        reconstructed_output_sha256=reconstructed_sha256,
        reference_output_sha256=reference_sha256,
        max_abs_error=error,
        closure_atol=float(atol),
        closure_rtol=float(rtol),
        closure_max_abs_limit=max_abs_limit,
        closure_nrmse=nrmse,
        closure_nrmse_limit=float(nrmse_limit),
        additive_closure_passed=passed,
    )
    return AdditivePartSurface(
        operator_kind=operator_kind,
        boundary_id=boundary_id,
        reference_source=reference_source,
        part_axis_width=part_axis_width,
        groups=groups,
        group_contributions=parts,
        reconstructed_output=reconstructed,
        reference_output=reference,
        membership_sha256=membership_sha256,
        group_contribution_sha256s=contribution_sha256s,
        reconstructed_output_sha256=reconstructed_sha256,
        reference_output_sha256=reference_sha256,
        max_abs_error=error,
        closure_atol=float(atol),
        closure_rtol=float(rtol),
        closure_max_abs_limit=max_abs_limit,
        closure_nrmse=nrmse,
        closure_nrmse_limit=float(nrmse_limit),
        additive_closure_passed=passed,
        surface_sha256=surface_sha256,
    )


def decompose_mlp_channel_groups(
    *,
    down_input: Tensor,
    down_projection_weight: Tensor,
    groups: Sequence[Sequence[int]],
    boundary_id: str,
    reference_output: Tensor | None = None,
    atol: float = DEFAULT_CLOSURE_ATOL,
    rtol: float = DEFAULT_CLOSURE_RTOL,
    nrmse_limit: float = DEFAULT_CLOSURE_NRMSE_LIMIT,
) -> AdditivePartSurface:
    """Partition ``down_input @ W_down.T`` into channel groups.

    ``down_input[..., j]`` already contains the native activation and
    gate/up product.  Channel ``j`` therefore owns exactly one additive output
    ``down_input[..., j] * W_down[:, j]`` at this boundary.
    """

    activation = _require_float_tensor(down_input, label="down_input")
    weight = _require_float_tensor(
        down_projection_weight,
        label="down_projection_weight",
        ndim=2,
    )
    if activation.ndim < 2:
        raise ValueError("down_input must have at least two dimensions")
    intermediate = int(activation.shape[-1])
    if weight.shape[1] != intermediate:
        raise ValueError("down projection width does not match down_input")
    canonical = _canonical_groups(
        groups,
        width=intermediate,
        exhaustive=True,
    )
    activation_audit = activation.to(torch.float64)
    weight_audit = weight.to(torch.float64)
    contributions = []
    for group in canonical:
        index = torch.tensor(
            group,
            dtype=torch.int64,
            device=activation_audit.device,
        )
        contribution = activation_audit.index_select(
            -1, index
        ) @ weight_audit.index_select(
            1,
            index,
        ).transpose(0, 1)
        contributions.append(contribution)
    native = activation_audit @ weight_audit.transpose(0, 1)
    if reference_output is not None:
        reference = _require_float_tensor(
            reference_output,
            label="reference_output",
        ).to(torch.float64)
        if reference.shape != native.shape:
            raise ValueError("reference_output shape is inconsistent")
    else:
        reference = native
    return _surface(
        operator_kind="mlp_channels_after_down",
        boundary_id=boundary_id,
        reference_source=(
            "provided_operator_output"
            if reference_output is not None
            else "recomputed_formula"
        ),
        part_axis_width=intermediate,
        groups=canonical,
        contributions=contributions,
        reference=reference,
        atol=atol,
        rtol=rtol,
        nrmse_limit=nrmse_limit,
    )


def decompose_attention_head_groups(
    *,
    probabilities: Tensor,
    repeated_values: Tensor,
    output_projection_weight: Tensor,
    groups: Sequence[Sequence[int]],
    boundary_id: str,
    reference_output: Tensor | None = None,
    atol: float = DEFAULT_CLOSURE_ATOL,
    rtol: float = DEFAULT_CLOSURE_RTOL,
    nrmse_limit: float = DEFAULT_CLOSURE_NRMSE_LIMIT,
) -> AdditivePartSurface:
    """Partition native attention output by query head after ``O``.

    K/V values and the probability router are shared substrate.  A query-head
    part owns only its already-routed contribution through the corresponding
    output-projection column block; this function does not duplicate K/V.
    """

    probability = _require_float_tensor(
        probabilities,
        label="probabilities",
        ndim=4,
    )
    values = _require_float_tensor(
        repeated_values,
        label="repeated_values",
        ndim=4,
    )
    weight = _require_float_tensor(
        output_projection_weight,
        label="output_projection_weight",
        ndim=2,
    )
    batch, heads, query_width, key_width = probability.shape
    if values.shape[:3] != (batch, heads, key_width):
        raise ValueError("attention probability/value axes do not align")
    head_dimension = int(values.shape[3])
    if weight.shape[1] != heads * head_dimension:
        raise ValueError("output projection does not match head geometry")
    canonical = _canonical_groups(groups, width=heads, exhaustive=True)
    probability_audit = probability.to(torch.float64)
    values_audit = values.to(torch.float64)
    weight_audit = weight.to(torch.float64)
    output_slices = weight_audit.reshape(
        weight.shape[0], heads, head_dimension
    ).permute(1, 2, 0)
    head_outputs = tuple(
        torch.einsum(
            "bqk,bkd,do->bqo",
            probability_audit[:, head],
            values_audit[:, head],
            output_slices[head],
        )
        for head in range(heads)
    )
    contributions = tuple(
        torch.stack(tuple(head_outputs[head] for head in group), dim=0).sum(
            dim=0
        )
        for group in canonical
    )
    native = torch.stack(head_outputs, dim=0).sum(dim=0)
    if native.shape != (batch, query_width, weight.shape[0]):
        raise AssertionError("internal attention output shape is invalid")
    if reference_output is not None:
        reference = _require_float_tensor(
            reference_output,
            label="reference_output",
        ).to(torch.float64)
        if reference.shape != native.shape:
            raise ValueError("reference_output shape is inconsistent")
    else:
        reference = native
    return _surface(
        operator_kind="attention_query_heads_after_O",
        boundary_id=boundary_id,
        reference_source=(
            "provided_operator_output"
            if reference_output is not None
            else "recomputed_formula"
        ),
        part_axis_width=heads,
        groups=canonical,
        contributions=contributions,
        reference=reference,
        atol=atol,
        rtol=rtol,
        nrmse_limit=nrmse_limit,
    )


def decompose_tied_lexical_ingress_column_groups(
    *,
    input_ids: Tensor,
    tied_weight: Tensor,
    embed_scale: float,
    groups: Sequence[Sequence[int]],
    boundary_id: str = "lexical.ingress",
    reference_output: Tensor | None = None,
    atol: float = DEFAULT_CLOSURE_ATOL,
    rtol: float = DEFAULT_CLOSURE_RTOL,
    nrmse_limit: float = DEFAULT_CLOSURE_NRMSE_LIMIT,
) -> AdditivePartSurface:
    """Partition a tied embedding lookup by hidden columns."""

    ids = _require_integer_tensor(input_ids, label="input_ids")
    weight = _require_float_tensor(tied_weight, label="tied_weight", ndim=2)
    if (
        not isinstance(embed_scale, (int, float))
        or isinstance(embed_scale, bool)
        or not math.isfinite(float(embed_scale))
        or embed_scale <= 0
    ):
        raise ValueError("embed_scale must be finite and positive")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= weight.shape[0]):
        raise ValueError("input token id is outside the vocabulary")
    hidden = int(weight.shape[1])
    canonical = _canonical_groups(groups, width=hidden, exhaustive=True)
    weight_audit = weight.to(torch.float64)
    native = weight_audit[ids.to(torch.int64)] * float(embed_scale)
    contributions = []
    for group in canonical:
        contribution = torch.zeros_like(native)
        index = torch.tensor(group, dtype=torch.int64, device=weight_audit.device)
        contribution.index_copy_(-1, index, native.index_select(-1, index))
        contributions.append(contribution)
    if reference_output is not None:
        reference = _require_float_tensor(
            reference_output,
            label="reference_output",
        ).to(torch.float64)
        if reference.shape != native.shape:
            raise ValueError("reference_output shape is inconsistent")
    else:
        reference = native
    return _surface(
        operator_kind="tied_lexical_ingress",
        boundary_id=boundary_id,
        reference_source=(
            "provided_operator_output"
            if reference_output is not None
            else "recomputed_formula"
        ),
        part_axis_width=hidden,
        groups=canonical,
        contributions=contributions,
        reference=reference,
        atol=atol,
        rtol=rtol,
        nrmse_limit=nrmse_limit,
    )


def decompose_tied_lexical_egress_column_groups(
    *,
    hidden_states: Tensor,
    tied_weight: Tensor,
    groups: Sequence[Sequence[int]],
    boundary_id: str = "lexical.raw_egress",
    reference_output: Tensor | None = None,
    atol: float = DEFAULT_CLOSURE_ATOL,
    rtol: float = DEFAULT_CLOSURE_RTOL,
    nrmse_limit: float = DEFAULT_CLOSURE_NRMSE_LIMIT,
) -> AdditivePartSurface:
    """Partition raw tied logits by the same hidden-column ownership."""

    hidden_value = _require_float_tensor(
        hidden_states,
        label="hidden_states",
    )
    weight = _require_float_tensor(tied_weight, label="tied_weight", ndim=2)
    if hidden_value.ndim < 2 or hidden_value.shape[-1] != weight.shape[1]:
        raise ValueError("hidden_states do not match the tied hidden width")
    hidden = int(weight.shape[1])
    canonical = _canonical_groups(groups, width=hidden, exhaustive=True)
    hidden_audit = hidden_value.to(torch.float64)
    weight_audit = weight.to(torch.float64)
    native = hidden_audit @ weight_audit.transpose(0, 1)
    contributions = []
    for group in canonical:
        index = torch.tensor(group, dtype=torch.int64, device=weight_audit.device)
        contributions.append(
            hidden_audit.index_select(-1, index)
            @ weight_audit.index_select(1, index).transpose(0, 1)
        )
    if reference_output is not None:
        reference = _require_float_tensor(
            reference_output,
            label="reference_output",
        ).to(torch.float64)
        if reference.shape != native.shape:
            raise ValueError("reference_output shape is inconsistent")
    else:
        reference = native
    return _surface(
        operator_kind="tied_lexical_raw_egress",
        boundary_id=boundary_id,
        reference_source=(
            "provided_operator_output"
            if reference_output is not None
            else "recomputed_formula"
        ),
        part_axis_width=hidden,
        groups=canonical,
        contributions=contributions,
        reference=reference,
        atol=atol,
        rtol=rtol,
        nrmse_limit=nrmse_limit,
    )
