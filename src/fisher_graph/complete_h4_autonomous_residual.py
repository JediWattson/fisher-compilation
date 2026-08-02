"""Autonomous complete-H4 finite-residual provider.

The provider in this module is fitted offline from native/base H4 pairs, but
its serving ABI is deliberately source free: it reads only the authenticated
one-pass prefix and the realized pre-correction H4 carrier.  Native H4,
targets, logits, and reverse-VJP gradients are never runtime inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Literal

import torch
from torch import Tensor

from .conditional_quadratic_edge import build_causal_lagged_modal_design
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from .radial_finite_displacement_correction import (
    family_balanced_row_weights,
)


__all__ = [
    "AutonomousCompleteH4FitObjective",
    "AutonomousCompleteH4ResidualProvider",
    "AutonomousCompleteH4TrainingSequence",
    "autonomous_complete_h4_residual_provider_from_state_dict",
    "autonomous_complete_h4_residual_provider_state_dict",
    "fit_autonomous_complete_h4_residual",
    "fit_autonomous_complete_h4_output_decoder",
    "load_autonomous_complete_h4_residual_provider",
    "save_autonomous_complete_h4_residual_provider",
]


AutonomousCompleteH4FitObjective = Literal[
    "hidden_residual_ridge",
    "reverse_vjp_row_weighted_ridge_v1",
]

_OBJECTIVES = frozenset(
    {"hidden_residual_ridge", "reverse_vjp_row_weighted_ridge_v1"}
)
_H4_SITE = "layer.4.output"
_SOURCE_RANK = 64
_H4_WIDTH = 640
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:autonomous-complete-h4:tensor:v1\0"
_SEQUENCE_DOMAIN = b"fisher-graph:autonomous-complete-h4:sequence:v1\0"
_PROVIDER_DOMAIN = b"fisher-graph:autonomous-complete-h4:provider:v1\0"
_PROVIDER_TENSOR_SCHEMA = (
    "fisher_graph.autonomous_complete_h4_residual_provider_tensor.v1"
)
_PROVIDER_STATE_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "provider_artifact_sha256",
        "bridge_binding_sha256",
        "ridge",
        "fit_objective",
        "fit_row_count",
        "fit_family_ids",
        "fit_sequence_sha256s",
        "weighted_residual_rmse",
        "vjp_weight_floor",
        "vjp_weight_ceiling",
        "observed_vjp_multiplier_min",
        "observed_vjp_multiplier_max",
        "fit_weight_sha256",
        "tensors",
    }
)
_PROVIDER_TENSOR_KEYS = frozenset(
    {
        "output_decoder",
        "lag_source_kernel",
        "state_kernel",
        "bias",
        "state_encoder",
    }
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(v) for v in canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _float_tensor(value: object, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or value.layout != torch.strided
        or value.device.type == "meta"
        or any(int(width) <= 0 for width in value.shape)
    ):
        raise ValueError(f"{label} must be a materialized floating rank-{ndim} tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _mask(value: object, *, length: int, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != (length,) or value.dtype != torch.bool:
        raise ValueError(f"{label} must be boolean [S]")
    return value.detach().to(device="cpu").contiguous().clone()


def _orthonormal_rows(value: object, *, width: int, label: str) -> Tensor:
    result = _float_tensor(value, label=label, ndim=2)
    if result.shape[1] != width or result.shape[0] > width:
        raise ValueError(f"{label} must have shape [rank, {width}]")
    identity = torch.eye(result.shape[0], dtype=torch.float64)
    if float((result @ result.T - identity).abs().max()) > 1.0e-10:
        raise ValueError(f"{label} rows must be orthonormal")
    return result


def _derived_support(positions: Tensor, valid: Tensor, source: Tensor) -> Tensor:
    support = torch.zeros_like(valid)
    source_positions = positions[source]
    if source_positions.numel() == 0:
        return support
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    support[indices] = (
        positions[indices].unsqueeze(1) >= source_positions.unsqueeze(0)
    ).any(dim=1)
    return support


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4TrainingSequence:
    """One private offline trace; tensors are copied into canonical CPU form."""

    example_id: str
    family_id: str
    source_modes: Tensor
    logical_positions: Tensor
    valid_mask: Tensor
    source_mask: Tensor
    support_mask: Tensor
    base_h4: Tensor
    native_h4: Tensor
    reverse_vjp_gradients: Tensor | None = None
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="example_id")
        _identifier(self.family_id, label="family_id")
        source_modes = _float_tensor(self.source_modes, label="source_modes", ndim=2)
        if source_modes.shape[1] != _SOURCE_RANK:
            raise ValueError("source_modes must have shape [S, 64]")
        length = int(source_modes.shape[0])
        if (
            not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.shape != (length,)
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("logical_positions must be integer [S]")
        positions = self.logical_positions.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()
        valid = _mask(self.valid_mask, length=length, label="valid_mask")
        source = _mask(self.source_mask, length=length, label="source_mask")
        support = _mask(self.support_mask, length=length, label="support_mask")
        if bool((source & ~valid).any()) or bool((support & ~valid).any()):
            raise ValueError("source/support masks must be subsets of valid_mask")
        selected_positions = positions[valid]
        if (
            selected_positions.numel() == 0
            or bool((selected_positions < 0).any())
            or (
                selected_positions.numel() > 1
                and not bool(torch.all(selected_positions[1:] > selected_positions[:-1]))
            )
        ):
            raise ValueError("valid logical positions must be nonnegative and increasing")
        if not torch.equal(support, _derived_support(positions, valid, source)):
            raise ValueError("support_mask is not the complete-H4 causal closure")
        if not bool(support.any()):
            raise ValueError("training sequence must contain complete-H4 support")
        if bool((source_modes[~source] != 0).any()):
            raise ValueError("source_modes must be zero off source_mask")
        base = _float_tensor(self.base_h4, label="base_h4", ndim=2)
        native = _float_tensor(self.native_h4, label="native_h4", ndim=2)
        if base.shape != (length, _H4_WIDTH) or native.shape != base.shape:
            raise ValueError("base_h4/native_h4 must have shape [S, 640]")
        gradient = None
        if self.reverse_vjp_gradients is not None:
            gradient = _float_tensor(
                self.reverse_vjp_gradients,
                label="reverse_vjp_gradients",
                ndim=2,
            )
            if gradient.shape != base.shape:
                raise ValueError("reverse_vjp_gradients must match [S, 640]")
        for name, value in (
            ("source_modes", source_modes),
            ("logical_positions", positions),
            ("valid_mask", valid),
            ("source_mask", source),
            ("support_mask", support),
            ("base_h4", base),
            ("native_h4", native),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reverse_vjp_gradients", gradient)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(
                _SEQUENCE_DOMAIN,
                {
                    "example_id": self.example_id,
                    "family_id": self.family_id,
                    "tensor_sha256s": {
                        name: _tensor_sha256(getattr(self, name))
                        for name in (
                            "source_modes",
                            "logical_positions",
                            "valid_mask",
                            "source_mask",
                            "support_mask",
                            "base_h4",
                            "native_h4",
                        )
                    }
                    | {
                        "reverse_vjp_gradients": (
                            None if gradient is None else _tensor_sha256(gradient)
                        )
                    },
                },
            ),
        )


def fit_autonomous_complete_h4_output_decoder(
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    *,
    rank: int,
) -> Tensor:
    """Derive a deterministic residual PCA decoder from training traces only."""

    if type(rank) is not int or rank <= 0 or rank > _H4_WIDTH:
        raise ValueError("rank must be in [1, 640]")
    ordered = _ordered_sequences(sequences)
    residual = torch.cat(
        [(s.native_h4 - s.base_h4)[s.support_mask] for s in ordered], dim=0
    )
    if rank > min(residual.shape):
        raise ValueError("rank exceeds the residual training matrix rank bound")
    _u, _singular, vh = torch.linalg.svd(residual, full_matrices=False)
    decoder = vh[:rank].contiguous()
    # Remove the otherwise arbitrary SVD row sign deterministically.
    for row in range(rank):
        pivot = int(decoder[row].abs().argmax())
        if float(decoder[row, pivot]) < 0.0:
            decoder[row].neg_()
    return _orthonormal_rows(decoder, width=_H4_WIDTH, label="output_decoder")


@dataclass(frozen=True, slots=True)
class AutonomousCompleteH4ResidualProvider(Gemma3L3L4CorrectionProvider):
    """Immutable source-free complete-H4 correction provider."""

    bridge_binding_sha256: str
    output_decoder: Tensor
    lag_source_kernel: Tensor
    state_kernel: Tensor
    bias: Tensor
    ridge: float
    fit_objective: AutonomousCompleteH4FitObjective
    fit_row_count: int
    fit_family_ids: tuple[str, ...]
    fit_sequence_sha256s: tuple[str, ...]
    weighted_residual_rmse: float
    vjp_weight_floor: float = 0.5
    vjp_weight_ceiling: float = 2.0
    observed_vjp_multiplier_min: float = 1.0
    observed_vjp_multiplier_max: float = 1.0
    fit_weight_sha256: str = ""
    state_encoder: Tensor | None = None
    site: str = field(init=False, default=_H4_SITE)
    write_scope: str = field(init=False, default="complete_h4_causal_support")
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.bridge_binding_sha256, label="bridge_binding_sha256")
        decoder = _orthonormal_rows(self.output_decoder, width=_H4_WIDTH, label="output_decoder")
        encoder = None
        if self.state_encoder is not None:
            encoder = _orthonormal_rows(self.state_encoder, width=_H4_WIDTH, label="state_encoder")
        effective_state_rank = decoder.shape[0] if encoder is None else encoder.shape[0]
        lag_kernel = _float_tensor(self.lag_source_kernel, label="lag_source_kernel", ndim=3)
        state_kernel = _float_tensor(self.state_kernel, label="state_kernel", ndim=2)
        bias = _float_tensor(self.bias, label="bias", ndim=1)
        if lag_kernel.shape[1:] != (_SOURCE_RANK, decoder.shape[0]):
            raise ValueError("lag_source_kernel must have shape [L, 64, Rout]")
        if state_kernel.shape != (effective_state_rank, decoder.shape[0]):
            raise ValueError("state_kernel must have shape [Rstate, Rout]")
        if bias.shape != (decoder.shape[0],):
            raise ValueError("bias must have shape [Rout]")
        if self.fit_objective not in _OBJECTIVES:
            raise ValueError("fit_objective is invalid")
        if not math.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        if type(self.fit_row_count) is not int or self.fit_row_count <= 0:
            raise ValueError("fit_row_count must be positive")
        if (
            type(self.fit_family_ids) is not tuple
            or not self.fit_family_ids
            or self.fit_family_ids != tuple(sorted(set(self.fit_family_ids)))
        ):
            raise ValueError("fit_family_ids must be canonical")
        if (
            type(self.fit_sequence_sha256s) is not tuple
            or not self.fit_sequence_sha256s
            or self.fit_sequence_sha256s != tuple(sorted(set(self.fit_sequence_sha256s)))
        ):
            raise ValueError("fit_sequence_sha256s must be canonical")
        for value in self.fit_sequence_sha256s:
            _require_sha256(value, label="fit sequence SHA-256")
        for name in (
            "weighted_residual_rmse",
            "vjp_weight_floor",
            "vjp_weight_ceiling",
            "observed_vjp_multiplier_min",
            "observed_vjp_multiplier_max",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not (0.0 < self.vjp_weight_floor <= 1.0 <= self.vjp_weight_ceiling):
            raise ValueError("VJP multiplier bounds must bracket one")
        if not (
            self.vjp_weight_floor <= self.observed_vjp_multiplier_min
            <= self.observed_vjp_multiplier_max <= self.vjp_weight_ceiling
        ):
            raise ValueError("observed VJP multipliers exceed their bounds")
        _require_sha256(self.fit_weight_sha256, label="fit_weight_sha256")
        for name, value in (
            ("output_decoder", decoder),
            ("lag_source_kernel", lag_kernel),
            ("state_kernel", state_kernel),
            ("bias", bias),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "state_encoder", encoder)
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(self.artifact_sha256, label="provider artifact") != computed:
                raise ValueError("autonomous provider artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def rank(self) -> int:
        return int(self.output_decoder.shape[0])

    @property
    def state_rank(self) -> int:
        return self.rank if self.state_encoder is None else int(self.state_encoder.shape[0])

    @property
    def source_rank(self) -> int:
        return _SOURCE_RANK

    @property
    def lag_count(self) -> int:
        return int(self.lag_source_kernel.shape[0])

    @property
    def prepared_float_scalar_count(self) -> int:
        return int(
            self.output_decoder.numel()
            + self.lag_source_kernel.numel()
            + self.state_kernel.numel()
            + self.bias.numel()
            + (0 if self.state_encoder is None else self.state_encoder.numel())
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return int(
            self.lag_count * self.source_rank * self.rank
            + _H4_WIDTH * self.state_rank
            + self.state_rank * self.rank
            + self.rank * _H4_WIDTH
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.autonomous_complete_h4_residual_provider.v1",
            "site": self.site,
            "write_scope": self.write_scope,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "tensor_sha256s": {
                "output_decoder": _tensor_sha256(self.output_decoder),
                "lag_source_kernel": _tensor_sha256(self.lag_source_kernel),
                "state_kernel": _tensor_sha256(self.state_kernel),
                "bias": _tensor_sha256(self.bias),
                "state_encoder": None if self.state_encoder is None else _tensor_sha256(self.state_encoder),
            },
            "ridge": self.ridge,
            "fit_objective": self.fit_objective,
            "fit_row_count": self.fit_row_count,
            "fit_family_ids": self.fit_family_ids,
            "fit_sequence_sha256s": self.fit_sequence_sha256s,
            "weighted_residual_rmse": self.weighted_residual_rmse,
            "vjp_weight_floor": self.vjp_weight_floor,
            "vjp_weight_ceiling": self.vjp_weight_ceiling,
            "observed_vjp_multiplier_min": self.observed_vjp_multiplier_min,
            "observed_vjp_multiplier_max": self.observed_vjp_multiplier_max,
            "fit_weight_sha256": self.fit_weight_sha256,
            "runtime_inputs": ("one_pass_prefix", "realized_pre_correction_h4"),
            "runtime_forbidden_inputs": ("native_h4", "targets", "logits", "gradients"),
        }

    def _computed_sha256(self) -> str:
        return _sha256(_PROVIDER_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("autonomous complete-H4 provider payload drifted")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "rank": self.rank,
            "state_rank": self.state_rank,
            "source_rank": self.source_rank,
            "lag_count": self.lag_count,
            "prepared_float_scalar_count": self.prepared_float_scalar_count,
            "logical_macs_per_token_upper_bound": self.logical_macs_per_token_upper_bound,
        }

    def modal_correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        """Return the source-free float64 correction before H4 decoding.

        Conditional providers may extend this modal value and then invoke
        :meth:`decode_modal` exactly once.  Keeping the extension in modal
        space avoids independently rounding a decoded parent correction.
        """

        self.validate_integrity()
        prefix.validate_integrity()
        if (
            prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or prefix.source_modes.shape[-1] != _SOURCE_RANK
            or prefix.clamped_y3.shape[-1] != _H4_WIDTH
            or not isinstance(realized_state, Tensor)
            or realized_state.shape != prefix.clamped_y3.shape
            or not realized_state.is_floating_point()
        ):
            raise ValueError("autonomous provider and one-pass geometry differ")
        prefix_sha = prefix.artifact_sha256
        realized_sha = _tensor_sha256(realized_state)
        source = prefix.source_modes.to(dtype=torch.float64)
        source = source.masked_fill(
            (~prefix.source_eligible_mask).to(source.device).unsqueeze(-1), 0.0
        )
        design = build_causal_lagged_modal_design(
            source,
            logical_positions=prefix.logical_positions.to(source.device),
            valid_mask=prefix.valid_target_mask.to(source.device),
            lag_count=self.lag_count,
        )
        modal = design @ self.lag_source_kernel.reshape(-1, self.rank).to(design.device)
        encoder = self.output_decoder if self.state_encoder is None else self.state_encoder
        state_modes = realized_state.to(dtype=torch.float64) @ encoder.to(realized_state.device).T
        modal = modal.to(state_modes.device) + state_modes @ self.state_kernel.to(state_modes.device)
        modal = modal + self.bias.to(modal.device)
        support = prefix.complete_h4_causal_support_mask().to(modal.device)
        modal = modal.masked_fill((~support).unsqueeze(-1), 0.0)
        if (
            prefix.artifact_sha256 != prefix_sha
            or _tensor_sha256(realized_state) != realized_sha
        ):
            raise RuntimeError("autonomous provider mutated a runtime input")
        if bool((modal[~support] != 0).any()):
            raise RuntimeError("autonomous provider wrote outside modal support")
        self.validate_integrity()
        prefix.validate_integrity()
        return modal.contiguous()

    def decode_modal(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        modal: Tensor,
        *,
        like: Tensor,
    ) -> Tensor:
        """Decode one authenticated modal correction into float64 H4 rows."""

        self.validate_integrity()
        prefix.validate_integrity()
        if (
            prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or not isinstance(modal, Tensor)
            or modal.shape != (*prefix.clamped_y3.shape[:2], self.rank)
            or not modal.is_floating_point()
            or not isinstance(like, Tensor)
            or like.shape != prefix.clamped_y3.shape
            or not like.is_floating_point()
        ):
            raise ValueError("autonomous modal decode geometry differs")
        support = prefix.complete_h4_causal_support_mask().to(modal.device)
        if (
            bool(support.any())
            and not bool(torch.isfinite(modal[support]).all())
        ):
            raise ValueError("autonomous modal correction is nonfinite")
        if bool((modal[~support] != 0).any()):
            raise ValueError("autonomous modal correction escapes support")
        decoded = modal.to(dtype=torch.float64) @ self.output_decoder.to(
            device=modal.device,
            dtype=torch.float64,
        )
        # Complete-H4 corrections intentionally remain float64 until the
        # bridge adds them to a float64 view of the realized carrier and casts
        # that sum once.  Returning the carrier dtype here would independently
        # round the delta and can select the wrong float32 value at a midpoint.
        result = torch.zeros_like(like, dtype=torch.float64)
        result_support = support.to(result.device)
        result[result_support] = decoded[support].to(
            device=result.device,
            dtype=torch.float64,
        )
        if bool((result[~result_support] != 0).any()):
            raise RuntimeError("autonomous provider wrote outside complete-H4 support")
        self.validate_integrity()
        prefix.validate_integrity()
        return result

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        """Return one decoded source-free correction under cast-once semantics."""

        return self.decode_modal(
            prefix,
            self.modal_correction(prefix, realized_state),
            like=realized_state,
        )


def _ordered_sequences(
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> tuple[AutonomousCompleteH4TrainingSequence, ...]:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence) or not sequences:
        raise ValueError("fit requires autonomous complete-H4 training sequences")
    if any(not isinstance(v, AutonomousCompleteH4TrainingSequence) for v in sequences):
        raise TypeError("fit sequence type differs")
    ordered = tuple(sorted(sequences, key=lambda v: (v.family_id, v.example_id)))
    if len({v.artifact_sha256 for v in ordered}) != len(ordered):
        raise ValueError("fit sequences must be unique")
    return ordered


def _bounded_vjp_multipliers(
    gradients: Tensor,
    example_ids: tuple[str, ...],
    *,
    floor: float,
    ceiling: float,
) -> Tensor:
    salience = torch.linalg.vector_norm(gradients, dim=1)
    result = torch.ones_like(salience)
    for example in sorted(set(example_ids)):
        indices = torch.tensor([i for i, value in enumerate(example_ids) if value == example])
        selected = salience.index_select(0, indices)
        mean = float(selected.mean())
        raw = torch.ones_like(selected) if mean == 0.0 else (selected / mean).clamp(floor, ceiling)
        normalized = raw / raw.mean()
        delta = normalized - 1.0
        alpha = 1.0
        if bool((delta > 0).any()):
            alpha = min(alpha, float(((ceiling - 1.0) / delta[delta > 0]).min()))
        if bool((delta < 0).any()):
            alpha = min(alpha, float(((floor - 1.0) / delta[delta < 0]).min()))
        bounded = 1.0 + max(0.0, min(1.0, alpha)) * delta
        result.index_copy_(0, indices, bounded)
    return result.contiguous()


def fit_autonomous_complete_h4_residual(
    *,
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    output_decoder: Tensor,
    bridge_binding_sha256: str,
    lag_count: int,
    ridge: float,
    state_encoder: Tensor | None = None,
    fit_objective: AutonomousCompleteH4FitObjective = "hidden_residual_ridge",
    vjp_weight_floor: float = 0.5,
    vjp_weight_ceiling: float = 2.0,
) -> AutonomousCompleteH4ResidualProvider:
    """Fit a deterministic family-balanced autonomous residual provider."""

    ordered = _ordered_sequences(sequences)
    _require_sha256(bridge_binding_sha256, label="bridge_binding_sha256")
    if type(lag_count) is not int or lag_count <= 0:
        raise ValueError("lag_count must be positive")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if fit_objective not in _OBJECTIVES:
        raise ValueError("fit_objective is invalid")
    if not (0.0 < vjp_weight_floor <= 1.0 <= vjp_weight_ceiling):
        raise ValueError("VJP multiplier bounds must bracket one")
    decoder = _orthonormal_rows(output_decoder, width=_H4_WIDTH, label="output_decoder")
    encoder = None if state_encoder is None else _orthonormal_rows(
        state_encoder, width=_H4_WIDTH, label="state_encoder"
    )
    effective_encoder = decoder if encoder is None else encoder
    design_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    gradient_rows: list[Tensor] = []
    families: list[str] = []
    examples: list[str] = []
    for sequence in ordered:
        source_design = build_causal_lagged_modal_design(
            sequence.source_modes,
            logical_positions=sequence.logical_positions,
            valid_mask=sequence.valid_mask,
            lag_count=lag_count,
        )
        state_modes = sequence.base_h4 @ effective_encoder.T
        design = torch.cat(
            (source_design, state_modes, torch.ones((source_design.shape[0], 1), dtype=torch.float64)),
            dim=1,
        )
        selected = sequence.support_mask
        design_rows.append(design[selected])
        target_rows.append(((sequence.native_h4 - sequence.base_h4) @ decoder.T)[selected])
        if fit_objective == "reverse_vjp_row_weighted_ridge_v1":
            if sequence.reverse_vjp_gradients is None:
                raise ValueError("reverse-VJP weighted fit requires every sequence gradient")
            gradient_rows.append(sequence.reverse_vjp_gradients[selected])
        count = int(selected.sum())
        families.extend([sequence.family_id] * count)
        examples.extend([sequence.example_id] * count)
    design = torch.cat(design_rows, dim=0)
    target = torch.cat(target_rows, dim=0)
    family_tuple = tuple(families)
    example_tuple = tuple(examples)
    weights = family_balanced_row_weights(family_tuple, example_tuple)
    multipliers = torch.ones_like(weights)
    if fit_objective == "reverse_vjp_row_weighted_ridge_v1":
        multipliers = _bounded_vjp_multipliers(
            torch.cat(gradient_rows, dim=0),
            example_tuple,
            floor=vjp_weight_floor,
            ceiling=vjp_weight_ceiling,
        )
        weights = (weights * multipliers).contiguous()
        weights = weights / weights.sum()
    rms = torch.sqrt((weights.unsqueeze(1) * design.square()).sum(dim=0))
    scales = torch.where(rms > math.sqrt(torch.finfo(torch.float64).eps), rms, torch.ones_like(rms))
    standardized = design / scales
    root = weights.sqrt().unsqueeze(1)
    weighted_design = standardized * root
    weighted_target = target * root
    gram = weighted_design.T @ weighted_design
    coefficients_std = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64),
        weighted_design.T @ weighted_target,
    )
    coefficients = coefficients_std / scales.unsqueeze(1)
    prediction = design @ coefficients
    weighted_rmse = float(
        torch.sqrt((weights.unsqueeze(1) * (prediction - target).square()).sum() / decoder.shape[0])
    )
    lag_width = lag_count * _SOURCE_RANK
    state_width = int(effective_encoder.shape[0])
    return AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=bridge_binding_sha256,
        output_decoder=decoder,
        lag_source_kernel=coefficients[:lag_width].reshape(lag_count, _SOURCE_RANK, decoder.shape[0]),
        state_kernel=coefficients[lag_width : lag_width + state_width],
        bias=coefficients[-1],
        ridge=ridge,
        fit_objective=fit_objective,
        fit_row_count=int(design.shape[0]),
        fit_family_ids=tuple(sorted(set(families))),
        fit_sequence_sha256s=tuple(sorted(v.artifact_sha256 for v in ordered)),
        weighted_residual_rmse=weighted_rmse,
        vjp_weight_floor=vjp_weight_floor,
        vjp_weight_ceiling=vjp_weight_ceiling,
        observed_vjp_multiplier_min=float(multipliers.min()),
        observed_vjp_multiplier_max=float(multipliers.max()),
        fit_weight_sha256=_tensor_sha256(weights),
        state_encoder=encoder,
    )


def autonomous_complete_h4_residual_provider_state_dict(
    provider: AutonomousCompleteH4ResidualProvider,
) -> dict[str, object]:
    """Return the strict source-free tensor state for one fitted provider."""

    if not isinstance(provider, AutonomousCompleteH4ResidualProvider):
        raise TypeError("provider must be AutonomousCompleteH4ResidualProvider")
    provider.validate_integrity()
    return {
        "schema": _PROVIDER_TENSOR_SCHEMA,
        "format_version": 1,
        "provider_artifact_sha256": provider.artifact_sha256,
        "bridge_binding_sha256": provider.bridge_binding_sha256,
        "ridge": float(provider.ridge),
        "fit_objective": provider.fit_objective,
        "fit_row_count": provider.fit_row_count,
        "fit_family_ids": provider.fit_family_ids,
        "fit_sequence_sha256s": provider.fit_sequence_sha256s,
        "weighted_residual_rmse": float(provider.weighted_residual_rmse),
        "vjp_weight_floor": float(provider.vjp_weight_floor),
        "vjp_weight_ceiling": float(provider.vjp_weight_ceiling),
        "observed_vjp_multiplier_min": float(
            provider.observed_vjp_multiplier_min
        ),
        "observed_vjp_multiplier_max": float(
            provider.observed_vjp_multiplier_max
        ),
        "fit_weight_sha256": provider.fit_weight_sha256,
        "tensors": {
            "output_decoder": provider.output_decoder.detach().clone(),
            "lag_source_kernel": provider.lag_source_kernel.detach().clone(),
            "state_kernel": provider.state_kernel.detach().clone(),
            "bias": provider.bias.detach().clone(),
            "state_encoder": (
                None
                if provider.state_encoder is None
                else provider.state_encoder.detach().clone()
            ),
        },
    }


def _strict_state_float(state: Mapping[str, object], name: str) -> float:
    value = state.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"provider state {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"provider state {name} must be finite")
    return result


def autonomous_complete_h4_residual_provider_from_state_dict(
    state: Mapping[str, object],
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None = None,
) -> AutonomousCompleteH4ResidualProvider:
    """Restore and reauthenticate an exact provider tensor state.

    An externally bound expected artifact digest is mandatory.  The digest in
    the tensor file is therefore a consistency receipt, never a caller-trusted
    replacement for the candidate frozen by its protocol report.
    """

    if not isinstance(state, Mapping) or set(state) != _PROVIDER_STATE_KEYS:
        raise ValueError("provider state fields differ")
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected provider artifact",
    )
    embedded_artifact = _require_sha256(
        state.get("provider_artifact_sha256"),
        label="embedded provider artifact",
    )
    if embedded_artifact != expected_artifact:
        raise ValueError("provider state artifact differs from expected")
    bridge_binding = _require_sha256(
        state.get("bridge_binding_sha256"),
        label="provider state bridge binding",
    )
    if expected_bridge_binding_sha256 is not None:
        expected_bridge = _require_sha256(
            expected_bridge_binding_sha256,
            label="expected provider bridge binding",
        )
        if bridge_binding != expected_bridge:
            raise ValueError("provider state bridge binding differs from expected")
    if (
        state.get("schema") != _PROVIDER_TENSOR_SCHEMA
        or state.get("format_version") != 1
        or type(state.get("fit_row_count")) is not int
        or not isinstance(state.get("fit_objective"), str)
        or type(state.get("fit_family_ids")) is not tuple
        or type(state.get("fit_sequence_sha256s")) is not tuple
        or not isinstance(state.get("fit_weight_sha256"), str)
    ):
        raise ValueError("provider state scalar contract differs")
    tensors = state.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _PROVIDER_TENSOR_KEYS:
        raise ValueError("provider state tensor fields differ")
    for name in (
        "output_decoder",
        "lag_source_kernel",
        "state_kernel",
        "bias",
    ):
        if not isinstance(tensors.get(name), Tensor):
            raise ValueError(f"provider state {name} must be a tensor")
    if tensors.get("state_encoder") is not None and not isinstance(
        tensors.get("state_encoder"), Tensor
    ):
        raise ValueError("provider state state_encoder must be a tensor or null")
    provider = AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=bridge_binding,
        output_decoder=tensors["output_decoder"],
        lag_source_kernel=tensors["lag_source_kernel"],
        state_kernel=tensors["state_kernel"],
        bias=tensors["bias"],
        ridge=_strict_state_float(state, "ridge"),
        fit_objective=state["fit_objective"],  # type: ignore[arg-type]
        fit_row_count=state["fit_row_count"],  # type: ignore[arg-type]
        fit_family_ids=state["fit_family_ids"],  # type: ignore[arg-type]
        fit_sequence_sha256s=state["fit_sequence_sha256s"],  # type: ignore[arg-type]
        weighted_residual_rmse=_strict_state_float(
            state, "weighted_residual_rmse"
        ),
        vjp_weight_floor=_strict_state_float(state, "vjp_weight_floor"),
        vjp_weight_ceiling=_strict_state_float(state, "vjp_weight_ceiling"),
        observed_vjp_multiplier_min=_strict_state_float(
            state, "observed_vjp_multiplier_min"
        ),
        observed_vjp_multiplier_max=_strict_state_float(
            state, "observed_vjp_multiplier_max"
        ),
        fit_weight_sha256=state["fit_weight_sha256"],  # type: ignore[arg-type]
        state_encoder=tensors["state_encoder"],  # type: ignore[arg-type]
        artifact_sha256=embedded_artifact,
    )
    provider.validate_integrity()
    if provider.artifact_sha256 != expected_artifact:
        raise RuntimeError("restored provider artifact authentication drifted")
    return provider


def _read_regular_provider_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("provider tensor path is not a readable regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError("provider tensor path must be a nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise RuntimeError("provider tensor file changed while reading")
    return payload


def _provider_from_bytes(
    payload: bytes,
    *,
    expected_artifact_sha256: str,
    expected_bridge_binding_sha256: str | None,
) -> AutonomousCompleteH4ResidualProvider:
    try:
        state = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("provider tensor payload is invalid") from error
    if not isinstance(state, Mapping):
        raise ValueError("provider tensor payload must contain a state mapping")
    return autonomous_complete_h4_residual_provider_from_state_dict(
        state,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
    )


def save_autonomous_complete_h4_residual_provider(
    provider: AutonomousCompleteH4ResidualProvider,
    path: Path | str,
) -> dict[str, object]:
    """Atomically publish one authenticated provider without overwrite."""

    if not isinstance(provider, AutonomousCompleteH4ResidualProvider):
        raise TypeError("provider must be AutonomousCompleteH4ResidualProvider")
    provider.validate_integrity()
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("provider tensor output must use .pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite autonomous provider")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    stage = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(
                autonomous_complete_h4_residual_provider_state_dict(provider),
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        payload = _read_regular_provider_file(stage)
        restored = _provider_from_bytes(
            payload,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_bridge_binding_sha256=provider.bridge_binding_sha256,
        )
        if restored.artifact_sha256 != provider.artifact_sha256:
            raise RuntimeError("staged provider roundtrip drifted")
        try:
            os.link(stage, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite autonomous provider"
            ) from error
        published = True
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return {
            "path": destination.as_posix(),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "file_bytes": len(payload),
            "provider_artifact_sha256": provider.artifact_sha256,
            "bridge_binding_sha256": provider.bridge_binding_sha256,
        }
    except BaseException:
        if published:
            raise RuntimeError(
                "provider publication durability is uncertain"
            )
        raise
    finally:
        stage.unlink(missing_ok=True)


def load_autonomous_complete_h4_residual_provider(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_file_sha256: str | None = None,
    expected_bridge_binding_sha256: str | None = None,
) -> AutonomousCompleteH4ResidualProvider:
    """Load one tensor file under external artifact and optional file binds."""

    payload = _read_regular_provider_file(Path(path))
    if expected_file_sha256 is not None:
        expected_file = _require_sha256(
            expected_file_sha256,
            label="expected provider tensor file",
        )
        if hashlib.sha256(payload).hexdigest() != expected_file:
            raise ValueError("provider tensor file differs from expected")
    return _provider_from_bytes(
        payload,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_bridge_binding_sha256=expected_bridge_binding_sha256,
    )
