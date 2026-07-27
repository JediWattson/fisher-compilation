"""Fisher-weighted computational modes for parameter-cluster contributions.

This module implements one explicit compiler boundary::

    parameter cluster -> native residual rows -> computational modes

For a declared parameter cluster (or layer fragment), let ``Y`` contain the
cluster's dense contribution at a residual-stream output site.  The fit
routine derives a *frozen* affine, orthonormal codec using only ``Y_fit`` and
its Fisher row weights:

``modes = (Y - mean) @ basis``

``Y_hat = modes @ basis.T + mean``

The modal coordinates are therefore targets for a later modal generator; the
generator is not involved in deriving their meaning.  Evaluation rows and
weights are used only to report held-out distortion.  Rank selection is
either absent (the complete predeclared ladder is returned) or fixed before
the fit.  There is no evaluation-driven rank choice.  The rate curve reports
the fit-derived affine mean as a separate baseline rather than calling it a
rank-zero learned mode, and rejects the obvious leakage case where both source
tensor pairs are exactly identical after canonicalization across fit and
evaluation.

Artifacts authenticate all executable tensors and scientific provenance while
retaining no source rows, Fisher row weights, prompt text, or source weights.
The decoder is the transpose of the encoder and shares its storage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "ComputationalModeBasis",
    "ComputationalModeBinding",
    "ComputationalModeConfig",
    "ComputationalModeMetrics",
    "ComputationalModeRateCurve",
    "ComputationalModeRatePoint",
    "ComputationalModeSpectrum",
    "computational_mode_site_sha256",
    "fit_computational_mode_rate_curve",
]


_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_BINDING_KIND = "fisher_graph.computational_mode_binding"
_CONFIG_KIND = "fisher_graph.computational_mode_config"
_BASIS_KIND = "fisher_graph.computational_mode_basis"
_POINT_KIND = "fisher_graph.computational_mode_rate_point"
_CURVE_KIND = "fisher_graph.computational_mode_rate_curve"

_BINDING_DOMAIN = b"fisher_graph.computational_modes.binding.v1\0"
_CONFIG_DOMAIN = b"fisher_graph.computational_modes.config.v1\0"
_BASIS_DOMAIN = b"fisher_graph.computational_modes.basis.v1\0"
_POINT_DOMAIN = b"fisher_graph.computational_modes.point.v1\0"
_CURVE_DOMAIN = b"fisher_graph.computational_modes.curve.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.computational_modes.tensor.v1\0"
_SOURCE_TENSOR_DOMAIN = (
    b"fisher_graph.computational_modes.source_tensor.v1\0"
)
_SITE_DOMAIN = b"fisher_graph.computational_modes.output_site.v1\0"

_SOURCE_KINDS = frozenset({"parameter_cluster", "layer_fragment"})
_SELECTION_RULES = frozenset({"return_all", "fixed_rank"})
_FIT_DEFINITION = "fisher_row_weighted_centered_svd_on_fit_contributions"
_COORDINATE_DEFINITION = "center_then_project_onto_orthonormal_residual_basis"
_DELETION_DEFINITION = "replace_cluster_residual_contribution_with_zero"
_MEAN_ONLY_DEFINITION = (
    "replace_cluster_residual_contribution_with_fit_weighted_mean"
)
_BASIS_ORDERING_DEFINITION = (
    "descending_singular_energy_with_projector_canonicalized_tied_and_null_"
    "subspaces"
)

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_fit_rows": False,
    "contains_raw_eval_rows": False,
    "contains_fisher_row_weights": False,
    "contains_computational_mode_basis": True,
    "evaluation_used_for_basis_fit": False,
    "evaluation_used_for_rank_selection": False,
    "executable_codec": True,
}


def _json_sha256(value: object, *, domain: bytes) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be a source-safe identifier using only letters, "
            "numbers, '.', '_', ':', '/', and '-'"
        )
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _expect_fields(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _canonical_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or value.numel() <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a nonempty floating rank-{ndim} Tensor"
        )
    result = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result.clone()


def _tensor_sha256(
    value: Tensor,
    *,
    label: str,
    domain: bytes = _TENSOR_DOMAIN,
) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be a finite contiguous CPU float64 Tensor"
        )
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        f"{tuple(value.shape)}\0float64\0".encode("utf-8")
    )
    digest.update(value.detach().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _as_rows(value: Tensor, *, label: str) -> Tensor:
    result = _canonical_tensor(value, label=label, ndim=2)
    if result.shape[0] <= 0 or result.shape[1] <= 0:
        raise ValueError(f"{label} must have positive row and residual widths")
    return result


def _as_weights(
    value: Tensor,
    *,
    rows: int,
    label: str,
) -> Tensor:
    result = _canonical_tensor(value, label=label, ndim=1)
    if result.shape != (rows,):
        raise ValueError(f"{label} must have shape ({rows},)")
    if bool((result < 0).any()) or float(result.sum().item()) <= 0.0:
        raise ValueError(
            f"{label} must be nonnegative with positive total mass"
        )
    return result


def computational_mode_site_sha256(output_site: str) -> str:
    """Return the domain-separated hash of a source-safe output-site name."""

    site = _require_identifier(output_site, label="output_site")
    digest = hashlib.sha256()
    digest.update(_SITE_DOMAIN)
    digest.update(site.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ComputationalModeBinding:
    """Authenticated cluster, model, Fisher, output-site, and split binding."""

    mode_set_id: str
    source_kind: str
    output_site: str
    output_site_sha256: str
    source_model_sha256: str
    parameter_catalog_sha256: str
    fisher_coupling_sha256: str
    parameter_cluster_sha256: str
    fit_split_sha256: str
    eval_split_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _BINDING_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.mode_set_id, label="mode_set_id")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                "source_kind must be 'parameter_cluster' or "
                "'layer_fragment'"
            )
        site = _require_identifier(self.output_site, label="output_site")
        if (
            _require_sha256(
                self.output_site_sha256,
                label="output_site_sha256",
            )
            != computational_mode_site_sha256(site)
        ):
            raise ValueError("output_site_sha256 does not match output_site")
        for name in (
            "source_model_sha256",
            "parameter_catalog_sha256",
            "fisher_coupling_sha256",
            "parameter_cluster_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.fit_split_sha256 == self.eval_split_sha256:
            raise ValueError("fit and evaluation split hashes must differ")
        if (
            self.artifact_kind != _BINDING_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("computational-mode binding header is invalid")
        computed = _json_sha256(
            self._payload(),
            domain=_BINDING_DOMAIN,
        )
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != computed:
                raise ValueError("computational-mode binding hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @classmethod
    def create(
        cls,
        *,
        mode_set_id: str,
        source_kind: str,
        output_site: str,
        source_model_sha256: str,
        parameter_catalog_sha256: str,
        fisher_coupling_sha256: str,
        parameter_cluster_sha256: str,
        fit_split_sha256: str,
        eval_split_sha256: str,
    ) -> ComputationalModeBinding:
        return cls(
            mode_set_id=mode_set_id,
            source_kind=source_kind,
            output_site=output_site,
            output_site_sha256=computational_mode_site_sha256(output_site),
            source_model_sha256=source_model_sha256,
            parameter_catalog_sha256=parameter_catalog_sha256,
            fisher_coupling_sha256=fisher_coupling_sha256,
            parameter_cluster_sha256=parameter_cluster_sha256,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "mode_set_id": self.mode_set_id,
            "source_kind": self.source_kind,
            "output_site": self.output_site,
            "output_site_sha256": self.output_site_sha256,
            "source_model_sha256": self.source_model_sha256,
            "parameter_catalog_sha256": self.parameter_catalog_sha256,
            "fisher_coupling_sha256": self.fisher_coupling_sha256,
            "parameter_cluster_sha256": self.parameter_cluster_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "eval_split_sha256": self.eval_split_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeBinding:
        fields = {
            "artifact_kind",
            "format_version",
            "mode_set_id",
            "source_kind",
            "output_site",
            "output_site_sha256",
            "source_model_sha256",
            "parameter_catalog_sha256",
            "fisher_coupling_sha256",
            "parameter_cluster_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "artifact_sha256",
        }
        _expect_fields(state, fields, label="computational-mode binding")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ComputationalModeConfig:
    """Predeclared rank ladder and non-adaptive selection policy."""

    ranks: tuple[int, ...]
    selection_rule: str = "return_all"
    selected_rank: int | None = None
    artifact_sha256: str = ""
    artifact_kind: str = _CONFIG_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.ranks) is not tuple or not self.ranks:
            raise ValueError("ranks must be a nonempty tuple")
        for index, rank in enumerate(self.ranks):
            _require_int(rank, label=f"ranks[{index}]", minimum=1)
        if self.ranks != tuple(sorted(set(self.ranks))):
            raise ValueError("ranks must be unique and strictly increasing")
        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError("selection_rule is unsupported")
        if self.selection_rule == "return_all":
            if self.selected_rank is not None:
                raise ValueError("return_all cannot declare selected_rank")
        elif (
            type(self.selected_rank) is not int
            or self.selected_rank not in self.ranks
        ):
            raise ValueError(
                "fixed_rank requires selected_rank in the rank ladder"
            )
        if (
            self.artifact_kind != _CONFIG_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("computational-mode config header is invalid")
        computed = _json_sha256(self._payload(), domain=_CONFIG_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != computed:
                raise ValueError("computational-mode config hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "ranks": self.ranks,
            "selection_rule": self.selection_rule,
            "selected_rank": self.selected_rank,
            "fit_definition": _FIT_DEFINITION,
            "coordinate_definition": _COORDINATE_DEFINITION,
            "deletion_definition": _DELETION_DEFINITION,
            "mean_only_definition": _MEAN_ONLY_DEFINITION,
            "mean_only_is_learned_mode": False,
            "basis_ordering_definition": _BASIS_ORDERING_DEFINITION,
            "rank_selection_uses_evaluation": False,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeConfig:
        fields = {
            "artifact_kind",
            "format_version",
            "ranks",
            "selection_rule",
            "selected_rank",
            "fit_definition",
            "coordinate_definition",
            "deletion_definition",
            "mean_only_definition",
            "mean_only_is_learned_mode",
            "basis_ordering_definition",
            "rank_selection_uses_evaluation",
            "artifact_sha256",
        }
        _expect_fields(state, fields, label="computational-mode config")
        if (
            state["fit_definition"] != _FIT_DEFINITION
            or state["coordinate_definition"] != _COORDINATE_DEFINITION
            or state["deletion_definition"] != _DELETION_DEFINITION
            or state["mean_only_definition"] != _MEAN_ONLY_DEFINITION
            or state["mean_only_is_learned_mode"] is not False
            or state["basis_ordering_definition"]
            != _BASIS_ORDERING_DEFINITION
            or state["rank_selection_uses_evaluation"] is not False
        ):
            raise ValueError("computational-mode config semantics are invalid")
        return cls(
            ranks=state["ranks"],  # type: ignore[arg-type]
            selection_rule=state["selection_rule"],  # type: ignore[arg-type]
            selected_rank=state["selected_rank"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ComputationalModeMetrics:
    """Reconstruction metrics for one fixed split and one prediction."""

    observations: int
    residual_width: int
    mse: float
    rmse: float
    weighted_mse: float
    weighted_rmse: float
    weighted_nrmse: float
    max_abs_error: float
    weighted_target_rms: float

    def __post_init__(self) -> None:
        _require_int(self.observations, label="observations", minimum=1)
        _require_int(self.residual_width, label="residual_width", minimum=1)
        for name in (
            "mse",
            "rmse",
            "weighted_mse",
            "weighted_rmse",
            "weighted_nrmse",
            "max_abs_error",
            "weighted_target_rms",
        ):
            object.__setattr__(
                self,
                name,
                _require_float(
                    getattr(self, name),
                    label=name,
                    minimum=0.0,
                ),
            )

    def metadata(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "residual_width": self.residual_width,
            "mse": self.mse,
            "rmse": self.rmse,
            "weighted_mse": self.weighted_mse,
            "weighted_rmse": self.weighted_rmse,
            "weighted_nrmse": self.weighted_nrmse,
            "max_abs_error": self.max_abs_error,
            "weighted_target_rms": self.weighted_target_rms,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeMetrics:
        fields = {
            "observations",
            "residual_width",
            "mse",
            "rmse",
            "weighted_mse",
            "weighted_rmse",
            "weighted_nrmse",
            "max_abs_error",
            "weighted_target_rms",
        }
        _expect_fields(state, fields, label="computational-mode metrics")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ComputationalModeSpectrum:
    """Fit-only energy summary for one retained prefix."""

    total_centered_energy: float
    rank_one_energy: float
    rank_one_energy_fraction: float
    retained_energy: float
    retained_energy_fraction: float
    tail_energy: float
    tail_energy_fraction: float

    def __post_init__(self) -> None:
        for name in (
            "total_centered_energy",
            "rank_one_energy",
            "retained_energy",
            "tail_energy",
        ):
            object.__setattr__(
                self,
                name,
                _require_float(
                    getattr(self, name),
                    label=name,
                    minimum=0.0,
                ),
            )
        for name in (
            "rank_one_energy_fraction",
            "retained_energy_fraction",
            "tail_energy_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _require_float(
                    getattr(self, name),
                    label=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        tolerance = 1e-10 * max(self.total_centered_energy, 1.0)
        if (
            abs(
                self.retained_energy
                + self.tail_energy
                - self.total_centered_energy
            )
            > tolerance
            or abs(
                self.retained_energy_fraction
                + self.tail_energy_fraction
                - (1.0 if self.total_centered_energy > 0.0 else 0.0)
            )
            > 1e-10
            or self.rank_one_energy
            > self.total_centered_energy + tolerance
        ):
            raise ValueError("computational-mode spectrum is inconsistent")

    def metadata(self) -> dict[str, object]:
        return {
            "total_centered_energy": self.total_centered_energy,
            "rank_one_energy": self.rank_one_energy,
            "rank_one_energy_fraction": self.rank_one_energy_fraction,
            "retained_energy": self.retained_energy,
            "retained_energy_fraction": self.retained_energy_fraction,
            "tail_energy": self.tail_energy,
            "tail_energy_fraction": self.tail_energy_fraction,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeSpectrum:
        fields = {
            "total_centered_energy",
            "rank_one_energy",
            "rank_one_energy_fraction",
            "retained_energy",
            "retained_energy_fraction",
            "tail_energy",
            "tail_energy_fraction",
        }
        _expect_fields(state, fields, label="computational-mode spectrum")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ComputationalModeBasis:
    """One authenticated affine orthonormal residual-space mode codec."""

    binding: ComputationalModeBinding
    config: ComputationalModeConfig
    rank: int
    mean_bias: Tensor
    encoder_basis: Tensor
    mean_bias_sha256: str = ""
    encoder_basis_sha256: str = ""
    decoder_basis_sha256: str = ""
    artifact_sha256: str = ""
    artifact_kind: str = _BASIS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ComputationalModeBinding):
            raise TypeError("binding must be ComputationalModeBinding")
        if not isinstance(self.config, ComputationalModeConfig):
            raise TypeError("config must be ComputationalModeConfig")
        rank = _require_int(self.rank, label="rank", minimum=1)
        if rank not in self.config.ranks:
            raise ValueError("rank is outside the configured ladder")
        mean = _canonical_tensor(
            self.mean_bias,
            label="mean_bias",
            ndim=1,
        )
        encoder = _canonical_tensor(
            self.encoder_basis,
            label="encoder_basis",
            ndim=2,
        )
        if encoder.shape != (mean.shape[0], rank):
            raise ValueError(
                "encoder_basis must have shape [residual_width, rank]"
            )
        gram = encoder.T @ encoder
        if not torch.allclose(
            gram,
            torch.eye(rank, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-11,
        ):
            raise ValueError("encoder_basis columns must be orthonormal")
        _validate_canonical_signs(encoder)
        object.__setattr__(self, "mean_bias", mean)
        object.__setattr__(self, "encoder_basis", encoder)
        actual_hashes = {
            "mean_bias_sha256": _tensor_sha256(
                mean,
                label="mean_bias",
            ),
            "encoder_basis_sha256": _tensor_sha256(
                encoder,
                label="encoder_basis",
            ),
            "decoder_basis_sha256": _tensor_sha256(
                encoder.T.contiguous(),
                label="decoder_basis",
            ),
        }
        for name, actual in actual_hashes.items():
            supplied = getattr(self, name)
            if supplied:
                _require_sha256(supplied, label=name)
                if supplied != actual:
                    raise ValueError(f"{name} does not match tensor")
            else:
                object.__setattr__(self, name, actual)
        if (
            self.artifact_kind != _BASIS_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("computational-mode basis header is invalid")
        computed = _json_sha256(self._payload(), domain=_BASIS_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != computed:
                raise ValueError("computational-mode basis hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def residual_width(self) -> int:
        return int(self.mean_bias.shape[0])

    @property
    def mean(self) -> Tensor:
        return self.mean_bias.clone()

    @property
    def bias(self) -> Tensor:
        return self.mean_bias.clone()

    @property
    def decoder_basis(self) -> Tensor:
        return self.encoder_basis.T.contiguous().clone()

    @property
    def basis_scalar_count(self) -> int:
        return self.residual_width * self.rank

    @property
    def bias_scalar_count(self) -> int:
        return self.residual_width

    @property
    def stored_scalar_count(self) -> int:
        return self.basis_scalar_count + self.bias_scalar_count

    @property
    def storage_bytes_float64(self) -> int:
        return 8 * self.stored_scalar_count

    @property
    def encode_projection_macs_per_row(self) -> int:
        return self.residual_width * self.rank

    @property
    def decode_projection_macs_per_row(self) -> int:
        return self.residual_width * self.rank

    @property
    def round_trip_projection_macs_per_row(self) -> int:
        return 2 * self.residual_width * self.rank

    @property
    def encode_center_additions_per_row(self) -> int:
        return self.residual_width

    @property
    def decode_bias_additions_per_row(self) -> int:
        return self.residual_width

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "binding_sha256": self.binding.artifact_sha256,
            "config_sha256": self.config.artifact_sha256,
            "rank": self.rank,
            "residual_width": self.residual_width,
            "mean_bias_sha256": self.mean_bias_sha256,
            "encoder_basis_sha256": self.encoder_basis_sha256,
            "decoder_basis_sha256": self.decoder_basis_sha256,
            "basis_scalar_count": self.basis_scalar_count,
            "bias_scalar_count": self.bias_scalar_count,
            "stored_scalar_count": self.stored_scalar_count,
            "storage_bytes_float64": self.storage_bytes_float64,
            "encode_projection_macs_per_row": (
                self.encode_projection_macs_per_row
            ),
            "decode_projection_macs_per_row": (
                self.decode_projection_macs_per_row
            ),
            "round_trip_projection_macs_per_row": (
                self.round_trip_projection_macs_per_row
            ),
            "encode_center_additions_per_row": (
                self.encode_center_additions_per_row
            ),
            "decode_bias_additions_per_row": (
                self.decode_bias_additions_per_row
            ),
            "encoder_decoder_share_storage": True,
        }

    def validate_integrity(self) -> None:
        self.binding.__post_init__()
        self.config.__post_init__()
        for value, expected, label in (
            (self.mean_bias, self.mean_bias_sha256, "mean_bias"),
            (
                self.encoder_basis,
                self.encoder_basis_sha256,
                "encoder_basis",
            ),
            (
                self.encoder_basis.T.contiguous(),
                self.decoder_basis_sha256,
                "decoder_basis",
            ),
        ):
            if _tensor_sha256(value, label=label) != expected:
                raise ValueError(f"{label}_sha256 does not match tensor")
        if _json_sha256(self._payload(), domain=_BASIS_DOMAIN) != (
            self.artifact_sha256
        ):
            raise ValueError("computational-mode basis hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "binding": self.binding.state_dict(),
            "config": self.config.state_dict(),
            "mean_bias": self.mean_bias.clone(),
            "encoder_basis": self.encoder_basis.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeBasis:
        fields = {
            "artifact_kind",
            "format_version",
            "binding_sha256",
            "config_sha256",
            "rank",
            "residual_width",
            "mean_bias_sha256",
            "encoder_basis_sha256",
            "decoder_basis_sha256",
            "basis_scalar_count",
            "bias_scalar_count",
            "stored_scalar_count",
            "storage_bytes_float64",
            "encode_projection_macs_per_row",
            "decode_projection_macs_per_row",
            "round_trip_projection_macs_per_row",
            "encode_center_additions_per_row",
            "decode_bias_additions_per_row",
            "encoder_decoder_share_storage",
            "artifact_sha256",
            "binding",
            "config",
            "mean_bias",
            "encoder_basis",
        }
        _expect_fields(state, fields, label="computational-mode basis")
        if not isinstance(state["binding"], Mapping) or not isinstance(
            state["config"],
            Mapping,
        ):
            raise TypeError("basis binding and config must be mappings")
        binding = ComputationalModeBinding.from_state_dict(state["binding"])
        config = ComputationalModeConfig.from_state_dict(state["config"])
        if state["binding_sha256"] != binding.artifact_sha256:
            raise ValueError("binding_sha256 does not match nested binding")
        if state["config_sha256"] != config.artifact_sha256:
            raise ValueError("config_sha256 does not match nested config")
        result = cls(
            binding=binding,
            config=config,
            rank=state["rank"],  # type: ignore[arg-type]
            mean_bias=state["mean_bias"],  # type: ignore[arg-type]
            encoder_basis=state["encoder_basis"],  # type: ignore[arg-type]
            mean_bias_sha256=state["mean_bias_sha256"],  # type: ignore[arg-type]
            encoder_basis_sha256=state["encoder_basis_sha256"],  # type: ignore[arg-type]
            decoder_basis_sha256=state["decoder_basis_sha256"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        expected = result._payload()
        for name in (
            "residual_width",
            "basis_scalar_count",
            "bias_scalar_count",
            "stored_scalar_count",
            "storage_bytes_float64",
            "encode_projection_macs_per_row",
            "decode_projection_macs_per_row",
            "round_trip_projection_macs_per_row",
            "encode_center_additions_per_row",
            "decode_bias_additions_per_row",
            "encoder_decoder_share_storage",
        ):
            if state[name] != expected[name]:
                raise ValueError(f"serialized {name} is inconsistent")
        return result

    def encode(self, contributions: Tensor) -> Tensor:
        """Project residual contributions into this frozen modal basis."""

        self.validate_integrity()
        if (
            not isinstance(contributions, Tensor)
            or not contributions.is_floating_point()
            or contributions.ndim < 1
            or contributions.shape[-1] != self.residual_width
        ):
            raise ValueError(
                "contributions must be floating with trailing residual width"
            )
        dtype = contributions.dtype
        device = contributions.device
        mean = self.mean_bias.to(device=device, dtype=dtype)
        basis = self.encoder_basis.to(device=device, dtype=dtype)
        return (contributions - mean) @ basis

    def decode(self, modal_coordinates: Tensor) -> Tensor:
        """Decode generated modal coordinates to a residual contribution."""

        self.validate_integrity()
        if (
            not isinstance(modal_coordinates, Tensor)
            or not modal_coordinates.is_floating_point()
            or modal_coordinates.ndim < 1
            or modal_coordinates.shape[-1] != self.rank
        ):
            raise ValueError(
                "modal_coordinates must be floating with trailing mode rank"
            )
        dtype = modal_coordinates.dtype
        device = modal_coordinates.device
        decoder = self.encoder_basis.T.to(device=device, dtype=dtype)
        mean = self.mean_bias.to(device=device, dtype=dtype)
        return modal_coordinates @ decoder + mean

    def reconstruct(self, contributions: Tensor) -> Tensor:
        """Encode and decode residual rows with the frozen affine codec."""

        return self.decode(self.encode(contributions))


@dataclass(frozen=True, slots=True)
class ComputationalModeRatePoint:
    """One predeclared mode rank and its fit/evaluation rate-distortion."""

    basis: ComputationalModeBasis
    spectrum: ComputationalModeSpectrum
    fit_reconstruction: ComputationalModeMetrics
    fit_deletion: ComputationalModeMetrics
    fit_mean_only: ComputationalModeMetrics
    eval_reconstruction: ComputationalModeMetrics
    eval_deletion: ComputationalModeMetrics
    eval_mean_only: ComputationalModeMetrics
    fit_error_to_deletion_ratio: float
    eval_error_to_deletion_ratio: float
    fit_error_to_mean_only_ratio: float
    eval_error_to_mean_only_ratio: float
    artifact_sha256: str = ""
    artifact_kind: str = _POINT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.basis, ComputationalModeBasis):
            raise TypeError("basis must be ComputationalModeBasis")
        if not isinstance(self.spectrum, ComputationalModeSpectrum):
            raise TypeError("spectrum must be ComputationalModeSpectrum")
        for name in (
            "fit_reconstruction",
            "fit_deletion",
            "fit_mean_only",
            "eval_reconstruction",
            "eval_deletion",
            "eval_mean_only",
        ):
            if not isinstance(getattr(self, name), ComputationalModeMetrics):
                raise TypeError(f"{name} must be ComputationalModeMetrics")
        for name in (
            "fit_error_to_deletion_ratio",
            "eval_error_to_deletion_ratio",
            "fit_error_to_mean_only_ratio",
            "eval_error_to_mean_only_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _require_float(
                    getattr(self, name),
                    label=name,
                    minimum=0.0,
                ),
            )
        if (
            self.artifact_kind != _POINT_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("computational-mode point header is invalid")
        computed = _json_sha256(self._payload(), domain=_POINT_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != computed:
                raise ValueError("computational-mode point hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def rank(self) -> int:
        return self.basis.rank

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "basis_sha256": self.basis.artifact_sha256,
            "rank": self.rank,
            "spectrum": self.spectrum.metadata(),
            "fit_reconstruction": self.fit_reconstruction.metadata(),
            "fit_deletion": self.fit_deletion.metadata(),
            "fit_mean_only": self.fit_mean_only.metadata(),
            "eval_reconstruction": self.eval_reconstruction.metadata(),
            "eval_deletion": self.eval_deletion.metadata(),
            "eval_mean_only": self.eval_mean_only.metadata(),
            "fit_error_to_deletion_ratio": (
                self.fit_error_to_deletion_ratio
            ),
            "eval_error_to_deletion_ratio": (
                self.eval_error_to_deletion_ratio
            ),
            "fit_error_to_mean_only_ratio": (
                self.fit_error_to_mean_only_ratio
            ),
            "eval_error_to_mean_only_ratio": (
                self.eval_error_to_mean_only_ratio
            ),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "basis": self.basis.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeRatePoint:
        fields = {
            "artifact_kind",
            "format_version",
            "basis_sha256",
            "rank",
            "spectrum",
            "fit_reconstruction",
            "fit_deletion",
            "fit_mean_only",
            "eval_reconstruction",
            "eval_deletion",
            "eval_mean_only",
            "fit_error_to_deletion_ratio",
            "eval_error_to_deletion_ratio",
            "fit_error_to_mean_only_ratio",
            "eval_error_to_mean_only_ratio",
            "artifact_sha256",
            "basis",
        }
        _expect_fields(state, fields, label="computational-mode point")
        basis = ComputationalModeBasis.from_state_dict(
            state["basis"]  # type: ignore[arg-type]
        )
        if state["basis_sha256"] != basis.artifact_sha256:
            raise ValueError("basis_sha256 does not match nested basis")
        if state["rank"] != basis.rank:
            raise ValueError("serialized point rank is inconsistent")
        return cls(
            basis=basis,
            spectrum=ComputationalModeSpectrum.from_state_dict(
                state["spectrum"]  # type: ignore[arg-type]
            ),
            fit_reconstruction=ComputationalModeMetrics.from_state_dict(
                state["fit_reconstruction"]  # type: ignore[arg-type]
            ),
            fit_deletion=ComputationalModeMetrics.from_state_dict(
                state["fit_deletion"]  # type: ignore[arg-type]
            ),
            fit_mean_only=ComputationalModeMetrics.from_state_dict(
                state["fit_mean_only"]  # type: ignore[arg-type]
            ),
            eval_reconstruction=ComputationalModeMetrics.from_state_dict(
                state["eval_reconstruction"]  # type: ignore[arg-type]
            ),
            eval_deletion=ComputationalModeMetrics.from_state_dict(
                state["eval_deletion"]  # type: ignore[arg-type]
            ),
            eval_mean_only=ComputationalModeMetrics.from_state_dict(
                state["eval_mean_only"]  # type: ignore[arg-type]
            ),
            fit_error_to_deletion_ratio=state[
                "fit_error_to_deletion_ratio"
            ],  # type: ignore[arg-type]
            eval_error_to_deletion_ratio=state[
                "eval_error_to_deletion_ratio"
            ],  # type: ignore[arg-type]
            fit_error_to_mean_only_ratio=state[
                "fit_error_to_mean_only_ratio"
            ],  # type: ignore[arg-type]
            eval_error_to_mean_only_ratio=state[
                "eval_error_to_mean_only_ratio"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ComputationalModeRateCurve:
    """Authenticated fit-only mode ladder with held-out measurements."""

    binding: ComputationalModeBinding
    config: ComputationalModeConfig
    fit_rows: int
    eval_rows: int
    residual_width: int
    fit_contributions_sha256: str
    eval_contributions_sha256: str
    fit_fisher_weights_sha256: str
    eval_fisher_weights_sha256: str
    points: tuple[ComputationalModeRatePoint, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _CURVE_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_raw_fit_rows: bool = False
    contains_raw_eval_rows: bool = False
    contains_fisher_row_weights: bool = False
    contains_computational_mode_basis: bool = True
    evaluation_used_for_basis_fit: bool = False
    evaluation_used_for_rank_selection: bool = False
    executable_codec: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ComputationalModeBinding):
            raise TypeError("binding must be ComputationalModeBinding")
        if not isinstance(self.config, ComputationalModeConfig):
            raise TypeError("config must be ComputationalModeConfig")
        _require_int(self.fit_rows, label="fit_rows", minimum=1)
        _require_int(self.eval_rows, label="eval_rows", minimum=1)
        _require_int(
            self.residual_width,
            label="residual_width",
            minimum=1,
        )
        for name in (
            "fit_contributions_sha256",
            "eval_contributions_sha256",
            "fit_fisher_weights_sha256",
            "eval_fisher_weights_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.fit_contributions_sha256
            == self.eval_contributions_sha256
            and self.fit_fisher_weights_sha256
            == self.eval_fisher_weights_sha256
        ):
            raise ValueError(
                "fit and evaluation contribution/weight source tensors "
                "must not be identical"
            )
        if type(self.points) is not tuple or not self.points:
            raise ValueError("points must be a nonempty tuple")
        if any(
            not isinstance(point, ComputationalModeRatePoint)
            for point in self.points
        ):
            raise TypeError("points contain an invalid item")
        if tuple(point.rank for point in self.points) != self.config.ranks:
            raise ValueError("point ranks must equal the configured ladder")
        first = self.points[0]
        for point in self.points:
            if (
                point.basis.binding.artifact_sha256
                != self.binding.artifact_sha256
                or point.basis.config.artifact_sha256
                != self.config.artifact_sha256
                or point.basis.residual_width != self.residual_width
                or point.fit_reconstruction.observations != self.fit_rows
                or point.eval_reconstruction.observations != self.eval_rows
                or point.fit_deletion != first.fit_deletion
                or point.eval_deletion != first.eval_deletion
                or point.fit_mean_only != first.fit_mean_only
                or point.eval_mean_only != first.eval_mean_only
            ):
                raise ValueError(
                    "point does not match curve provenance, shape, or "
                    "shared baselines"
                )
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError(
                    "computational-mode safety metadata is invalid"
                )
        if (
            self.artifact_kind != _CURVE_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("computational-mode curve header is invalid")
        computed = _json_sha256(self._payload(), domain=_CURVE_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != computed:
                raise ValueError("computational-mode curve hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def selected_point(self) -> ComputationalModeRatePoint | None:
        if self.config.selection_rule != "fixed_rank":
            return None
        return self.point_for_rank(self.config.selected_rank)  # type: ignore[arg-type]

    @property
    def selected_basis(self) -> ComputationalModeBasis | None:
        point = self.selected_point
        return None if point is None else point.basis

    @property
    def fit_mean_only(self) -> ComputationalModeMetrics:
        """Fit error of the affine mean before any learned mode is applied."""

        return self.points[0].fit_mean_only

    @property
    def eval_mean_only(self) -> ComputationalModeMetrics:
        """Evaluation error of the fit-derived mean with no learned mode."""

        return self.points[0].eval_mean_only

    def point_for_rank(self, rank: int) -> ComputationalModeRatePoint:
        for point in self.points:
            if point.rank == rank:
                return point
        raise KeyError(f"rank {rank} is not in the predeclared ladder")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "binding_sha256": self.binding.artifact_sha256,
            "config_sha256": self.config.artifact_sha256,
            "fit_rows": self.fit_rows,
            "eval_rows": self.eval_rows,
            "residual_width": self.residual_width,
            "fit_contributions_sha256": self.fit_contributions_sha256,
            "eval_contributions_sha256": self.eval_contributions_sha256,
            "fit_fisher_weights_sha256": (
                self.fit_fisher_weights_sha256
            ),
            "eval_fisher_weights_sha256": (
                self.eval_fisher_weights_sha256
            ),
            "point_sha256s": tuple(
                point.artifact_sha256 for point in self.points
            ),
            **_SAFETY_METADATA,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "binding": self.binding.metadata(),
            "config": self.config.metadata(),
            "points": tuple(point.metadata() for point in self.points),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "binding": self.binding.state_dict(),
            "config": self.config.state_dict(),
            "points": tuple(point.state_dict() for point in self.points),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ComputationalModeRateCurve:
        fields = {
            "artifact_kind",
            "format_version",
            "binding_sha256",
            "config_sha256",
            "fit_rows",
            "eval_rows",
            "residual_width",
            "fit_contributions_sha256",
            "eval_contributions_sha256",
            "fit_fisher_weights_sha256",
            "eval_fisher_weights_sha256",
            "point_sha256s",
            *set(_SAFETY_METADATA),
            "binding",
            "config",
            "points",
            "artifact_sha256",
        }
        _expect_fields(state, fields, label="computational-mode curve")
        binding = ComputationalModeBinding.from_state_dict(
            state["binding"]  # type: ignore[arg-type]
        )
        config = ComputationalModeConfig.from_state_dict(
            state["config"]  # type: ignore[arg-type]
        )
        if binding.artifact_sha256 != state["binding_sha256"]:
            raise ValueError("binding_sha256 does not match nested binding")
        if config.artifact_sha256 != state["config_sha256"]:
            raise ValueError("config_sha256 does not match nested config")
        if type(state["points"]) is not tuple:
            raise TypeError("computational-mode points must be a tuple")
        points = tuple(
            ComputationalModeRatePoint.from_state_dict(value)
            for value in state["points"]
        )
        if state["point_sha256s"] != tuple(
            point.artifact_sha256 for point in points
        ):
            raise ValueError("point_sha256s do not match nested points")
        return cls(
            binding=binding,
            config=config,
            fit_rows=state["fit_rows"],  # type: ignore[arg-type]
            eval_rows=state["eval_rows"],  # type: ignore[arg-type]
            residual_width=state["residual_width"],  # type: ignore[arg-type]
            fit_contributions_sha256=state[
                "fit_contributions_sha256"
            ],  # type: ignore[arg-type]
            eval_contributions_sha256=state[
                "eval_contributions_sha256"
            ],  # type: ignore[arg-type]
            fit_fisher_weights_sha256=state[
                "fit_fisher_weights_sha256"
            ],  # type: ignore[arg-type]
            eval_fisher_weights_sha256=state[
                "eval_fisher_weights_sha256"
            ],  # type: ignore[arg-type]
            points=points,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name]  # type: ignore[misc]
                for name in _SAFETY_METADATA
            },
        )


def _validate_canonical_signs(basis: Tensor) -> None:
    for column in range(basis.shape[1]):
        vector = basis[:, column]
        pivot = int(torch.argmax(vector.abs()).item())
        if float(vector[pivot].item()) < 0.0:
            raise ValueError("encoder_basis signs are not canonical")


def _canonicalize_basis_signs(basis: Tensor) -> Tensor:
    result = basis.clone()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(torch.argmax(vector.abs()).item())
        if float(vector[pivot].item()) < 0.0:
            result[:, column].neg_()
    return result.contiguous()


def _canonical_basis_from_projector(
    projector: Tensor,
    *,
    dimension: int,
) -> Tensor:
    """Choose a coordinate-ordered basis using only a subspace projector."""

    width = projector.shape[0]
    if (
        projector.shape != (width, width)
        or projector.dtype != torch.float64
        or projector.device.type != "cpu"
        or dimension <= 0
        or dimension > width
    ):
        raise ValueError("canonical projector basis inputs are invalid")
    # Project coordinate axes in a fixed order, then use two-pass modified
    # Gram-Schmidt.  The result depends on the subspace projector rather than
    # on an arbitrary SVD rotation inside that subspace.
    tolerance = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(width, 1)
    )
    vectors: list[Tensor] = []
    for coordinate in range(width):
        candidate = projector[:, coordinate].clone()
        for _ in range(2):
            for prior in vectors:
                candidate -= prior * torch.dot(prior, candidate)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= tolerance:
            continue
        vectors.append(candidate / norm)
        if len(vectors) == dimension:
            break
    if len(vectors) != dimension:
        raise RuntimeError(
            "could not derive the requested deterministic projector basis"
        )
    return _canonicalize_basis_signs(
        torch.stack(vectors, dim=1).contiguous()
    )


def _canonical_orthogonal_complement_basis(
    supported_basis: Tensor,
    *,
    dimension: int,
) -> Tensor:
    """Choose coordinate-ordered directions orthogonal to a fixed basis."""

    if (
        supported_basis.ndim != 2
        or supported_basis.dtype != torch.float64
        or supported_basis.device.type != "cpu"
        or dimension <= 0
        or supported_basis.shape[1] + dimension
        > supported_basis.shape[0]
    ):
        raise ValueError("canonical complement basis inputs are invalid")
    width = supported_basis.shape[0]
    tolerance = (
        64.0
        * torch.finfo(torch.float64).eps
        * max(width, 1)
    )
    vectors: list[Tensor] = []
    for coordinate in range(width):
        candidate = torch.zeros(width, dtype=torch.float64)
        candidate[coordinate] = 1.0
        for _ in range(2):
            if supported_basis.shape[1]:
                candidate -= supported_basis @ (
                    supported_basis.T @ candidate
                )
            for prior in vectors:
                candidate -= prior * torch.dot(prior, candidate)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= tolerance:
            continue
        vectors.append(candidate / norm)
        if len(vectors) == dimension:
            break
    if len(vectors) != dimension:
        raise RuntimeError(
            "could not derive the requested deterministic complement basis"
        )
    return _canonicalize_basis_signs(
        torch.stack(vectors, dim=1).contiguous()
    )


def _canonicalize_svd_basis(
    basis: Tensor,
    singular_values: Tensor,
    *,
    matrix_shape: tuple[int, int],
    requested_count: int | None = None,
) -> Tensor:
    """Remove arbitrary rotations from tied and numerically null SVD spaces."""

    if (
        basis.ndim != 2
        or singular_values.ndim != 1
        or basis.shape[1] != singular_values.shape[0]
        or basis.dtype != torch.float64
        or singular_values.dtype != torch.float64
        or basis.device.type != "cpu"
        or singular_values.device.type != "cpu"
    ):
        raise ValueError("SVD basis canonicalization inputs are invalid")
    count = singular_values.numel()
    if count == 0:
        return basis.clone().contiguous()
    retained = count if requested_count is None else requested_count
    if type(retained) is not int or not 0 < retained <= count:
        raise ValueError("requested SVD basis count is invalid")

    eps = torch.finfo(torch.float64).eps
    size = max(*matrix_shape, 1)
    largest = float(singular_values[0].item())
    zero_tolerance = 8.0 * eps * size * largest
    supported = int((singular_values > zero_tolerance).sum().item())
    blocks: list[Tensor] = []
    start = 0
    # Only materialize the prefix used by the declared rate curve.  Native MLP
    # fragment residuals are often highly rank-deficient (for example, 72
    # removed channels decoded into a 640-wide residual).  Constructing an
    # arbitrary 568-vector numerical null-space completion is unnecessary and
    # can accumulate enough roundoff to fail a strict full-basis Gram check.
    # A tied block that crosses the requested boundary is canonicalized as one
    # subspace and sliced only after its deterministic basis is constructed.
    while start < supported and start < retained:
        stop = start + 1
        while stop < supported:
            left = float(singular_values[start].item())
            right = float(singular_values[stop].item())
            local_scale = max(abs(left), abs(right))
            tie_tolerance = 8.0 * eps * size * local_scale
            if abs(left - right) > tie_tolerance:
                break
            stop += 1
        block = basis[:, start:stop]
        if stop - start == 1:
            blocks.append(_canonicalize_basis_signs(block))
        else:
            blocks.append(
                _canonical_basis_from_projector(
                    block @ block.T,
                    dimension=stop - start,
                )
            )
        start = stop

    supported_basis = (
        torch.cat(blocks, dim=1)
        if blocks
        else basis.new_empty((basis.shape[0], 0))
    )
    null_count = max(retained - supported, 0)
    if null_count:
        blocks.append(
            _canonical_orthogonal_complement_basis(
                supported_basis,
                dimension=null_count,
            )
        )
    result = torch.cat(blocks, dim=1)[:, :retained].contiguous()
    # Projector-derived tied blocks and near-null complements are individually
    # canonical, but concatenating many numerically estimated subspaces can
    # accumulate a small cross-block Gram error.  Ordered reduced QR preserves
    # the span of every column prefix (and therefore every declared ladder
    # point); sign canonicalization then removes QR's diagonal-sign freedom.
    result, _ = torch.linalg.qr(result, mode="reduced")
    result = _canonicalize_basis_signs(result)
    if not torch.allclose(
        result.T @ result,
        torch.eye(retained, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-11,
    ):
        raise RuntimeError("canonical SVD basis lost orthonormality")
    return result


def _metrics(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor,
) -> ComputationalModeMetrics:
    error = prediction - target
    squared = error.square()
    mse = float(squared.mean().item())
    per_row_mse = squared.mean(dim=1)
    weight_sum = weights.sum()
    weighted_mse = float(
        ((weights @ per_row_mse) / weight_sum).item()
    )
    target_per_row = target.square().mean(dim=1)
    weighted_target_energy = float(
        ((weights @ target_per_row) / weight_sum).item()
    )
    weighted_target_rms = math.sqrt(max(weighted_target_energy, 0.0))
    weighted_rmse = math.sqrt(max(weighted_mse, 0.0))
    if weighted_target_rms == 0.0:
        weighted_nrmse = 0.0 if weighted_rmse == 0.0 else (
            weighted_rmse / torch.finfo(torch.float64).tiny
        )
    else:
        weighted_nrmse = weighted_rmse / weighted_target_rms
    return ComputationalModeMetrics(
        observations=target.shape[0],
        residual_width=target.shape[1],
        mse=mse,
        rmse=math.sqrt(max(mse, 0.0)),
        weighted_mse=weighted_mse,
        weighted_rmse=weighted_rmse,
        weighted_nrmse=weighted_nrmse,
        max_abs_error=float(error.abs().max().item()),
        weighted_target_rms=weighted_target_rms,
    )


def _error_ratio(
    reconstruction: ComputationalModeMetrics,
    deletion: ComputationalModeMetrics,
) -> float:
    if deletion.weighted_mse == 0.0:
        return 0.0 if reconstruction.weighted_mse == 0.0 else (
            reconstruction.weighted_mse
            / torch.finfo(torch.float64).tiny
        )
    return reconstruction.weighted_mse / deletion.weighted_mse


def _error_to_mean_only_ratio(
    reconstruction: ComputationalModeMetrics,
    mean_only: ComputationalModeMetrics,
) -> float:
    """Return modal error / mean-only error, treating a perfect tie as one."""

    if mean_only.weighted_mse == 0.0:
        return 1.0 if reconstruction.weighted_mse == 0.0 else (
            reconstruction.weighted_mse
            / torch.finfo(torch.float64).tiny
        )
    return reconstruction.weighted_mse / mean_only.weighted_mse


def _spectrum(
    singular_values: Tensor,
    *,
    rank: int,
    total_centered_energy: float,
    weight_sum: float,
) -> ComputationalModeSpectrum:
    component_energy = singular_values.square() / weight_sum
    rank_one = (
        float(component_energy[0].item())
        if component_energy.numel()
        else 0.0
    )
    retained = float(component_energy[:rank].sum().item())
    # Use the directly accumulated total so numerical energy outside a
    # returned thin-SVD basis is not silently discarded.
    retained = min(max(retained, 0.0), total_centered_energy)
    tail = max(total_centered_energy - retained, 0.0)
    if total_centered_energy == 0.0:
        rank_one_fraction = retained_fraction = tail_fraction = 0.0
    else:
        rank_one_fraction = min(
            max(rank_one / total_centered_energy, 0.0),
            1.0,
        )
        retained_fraction = min(
            max(retained / total_centered_energy, 0.0),
            1.0,
        )
        tail_fraction = max(1.0 - retained_fraction, 0.0)
    return ComputationalModeSpectrum(
        total_centered_energy=total_centered_energy,
        rank_one_energy=min(rank_one, total_centered_energy),
        rank_one_energy_fraction=rank_one_fraction,
        retained_energy=retained,
        retained_energy_fraction=retained_fraction,
        tail_energy=tail,
        tail_energy_fraction=tail_fraction,
    )


def fit_computational_mode_rate_curve(
    fit_contributions: Tensor,
    fit_fisher_weights: Tensor,
    eval_contributions: Tensor,
    eval_fisher_weights: Tensor,
    ranks: Sequence[int],
    *,
    binding: ComputationalModeBinding,
    selection_rule: str = "return_all",
    selected_rank: int | None = None,
) -> ComputationalModeRateCurve:
    """Fit a Fisher-weighted mode basis from fit contributions only.

    ``eval_contributions`` and ``eval_fisher_weights`` are read only after the
    complete ordered basis has been frozen.  They cannot affect the basis,
    rank ladder, fixed selected rank, mean, orientation, or resource counts.
    """

    if not isinstance(binding, ComputationalModeBinding):
        raise TypeError("binding must be ComputationalModeBinding")
    fit = _as_rows(fit_contributions, label="fit_contributions")
    evaluation = _as_rows(
        eval_contributions,
        label="eval_contributions",
    )
    if evaluation.shape[1] != fit.shape[1]:
        raise ValueError(
            "fit and evaluation residual widths must be identical"
        )
    fit_weights = _as_weights(
        fit_fisher_weights,
        rows=fit.shape[0],
        label="fit_fisher_weights",
    )
    eval_weights = _as_weights(
        eval_fisher_weights,
        rows=evaluation.shape[0],
        label="eval_fisher_weights",
    )
    fit_contributions_sha256 = _tensor_sha256(
        fit,
        label="fit_contributions",
        domain=_SOURCE_TENSOR_DOMAIN,
    )
    eval_contributions_sha256 = _tensor_sha256(
        evaluation,
        label="eval_contributions",
        domain=_SOURCE_TENSOR_DOMAIN,
    )
    fit_fisher_weights_sha256 = _tensor_sha256(
        fit_weights,
        label="fit_fisher_weights",
        domain=_SOURCE_TENSOR_DOMAIN,
    )
    eval_fisher_weights_sha256 = _tensor_sha256(
        eval_weights,
        label="eval_fisher_weights",
        domain=_SOURCE_TENSOR_DOMAIN,
    )
    if (
        fit_contributions_sha256 == eval_contributions_sha256
        and fit_fisher_weights_sha256 == eval_fisher_weights_sha256
    ):
        raise ValueError(
            "fit and evaluation contribution/weight source tensors "
            "must not be identical"
        )
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be a sequence of integers")
    config = ComputationalModeConfig(
        ranks=tuple(ranks),
        selection_rule=selection_rule,
        selected_rank=selected_rank,
    )
    max_available_rank = min(fit.shape)
    if config.ranks[-1] > max_available_rank:
        raise ValueError(
            "largest mode rank exceeds the thin-SVD rank available from "
            "fit rows and residual width"
        )

    weight_sum_tensor = fit_weights.sum()
    weight_sum = float(weight_sum_tensor.item())
    mean = (
        (fit_weights[:, None] * fit).sum(dim=0) / weight_sum_tensor
    ).contiguous()
    centered = fit - mean
    weighted_centered = (
        centered * torch.sqrt(fit_weights)[:, None]
    ).contiguous()

    # The SVD is performed exactly once.  Every ladder point receives a
    # prefix, which guarantees nested mode spaces.  Evaluation is intentionally
    # absent from this computation.
    _, singular_values, vh = torch.linalg.svd(
        weighted_centered,
        full_matrices=False,
    )
    full_basis = _canonicalize_svd_basis(
        vh.T.contiguous(),
        singular_values,
        matrix_shape=tuple(weighted_centered.shape),
        requested_count=config.ranks[-1],
    )
    total_centered_energy = float(
        (weighted_centered.square().sum() / weight_sum_tensor).item()
    )

    zero_fit = torch.zeros_like(fit)
    zero_eval = torch.zeros_like(evaluation)
    fit_deletion = _metrics(fit, zero_fit, fit_weights)
    eval_deletion = _metrics(evaluation, zero_eval, eval_weights)
    fit_mean_only = _metrics(
        fit,
        mean.unsqueeze(0).expand_as(fit),
        fit_weights,
    )
    eval_mean_only = _metrics(
        evaluation,
        mean.unsqueeze(0).expand_as(evaluation),
        eval_weights,
    )

    points: list[ComputationalModeRatePoint] = []
    for rank in config.ranks:
        basis = ComputationalModeBasis(
            binding=binding,
            config=config,
            rank=rank,
            mean_bias=mean,
            encoder_basis=full_basis[:, :rank],
        )
        fit_reconstruction = _metrics(
            fit,
            basis.reconstruct(fit),
            fit_weights,
        )
        eval_reconstruction = _metrics(
            evaluation,
            basis.reconstruct(evaluation),
            eval_weights,
        )
        points.append(
            ComputationalModeRatePoint(
                basis=basis,
                spectrum=_spectrum(
                    singular_values,
                    rank=rank,
                    total_centered_energy=total_centered_energy,
                    weight_sum=weight_sum,
                ),
                fit_reconstruction=fit_reconstruction,
                fit_deletion=fit_deletion,
                fit_mean_only=fit_mean_only,
                eval_reconstruction=eval_reconstruction,
                eval_deletion=eval_deletion,
                eval_mean_only=eval_mean_only,
                fit_error_to_deletion_ratio=_error_ratio(
                    fit_reconstruction,
                    fit_deletion,
                ),
                eval_error_to_deletion_ratio=_error_ratio(
                    eval_reconstruction,
                    eval_deletion,
                ),
                fit_error_to_mean_only_ratio=_error_to_mean_only_ratio(
                    fit_reconstruction,
                    fit_mean_only,
                ),
                eval_error_to_mean_only_ratio=_error_to_mean_only_ratio(
                    eval_reconstruction,
                    eval_mean_only,
                ),
            )
        )

    return ComputationalModeRateCurve(
        binding=binding,
        config=config,
        fit_rows=fit.shape[0],
        eval_rows=evaluation.shape[0],
        residual_width=fit.shape[1],
        fit_contributions_sha256=fit_contributions_sha256,
        eval_contributions_sha256=eval_contributions_sha256,
        fit_fisher_weights_sha256=fit_fisher_weights_sha256,
        eval_fisher_weights_sha256=eval_fisher_weights_sha256,
        points=tuple(points),
    )
