"""Deterministic reduced-rank generators for Fisher-discovered mode clusters.

A modal generator replaces the dense residual contribution produced by a
cluster of native modes.  Its runtime input must be a cheap layer or block
input that remains available after the native modes are removed:

``Y_hat = (X @ input_factor) @ output_factor + bias``.

The fitter in this module performs weighted reduced-rank regression.  Fisher
weights affect the fit objective, while an explicit rank ladder exposes the
parameter/error and MAC/error trade-off.  No prompt text, raw activation rows,
native source weights, or native target rows are retained in the artifacts.
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
    "ModalGeneratorBinding",
    "ModalGeneratorConfig",
    "ModalGeneratorFactors",
    "ModalGeneratorMetrics",
    "ModalGeneratorPlan",
    "ModalGeneratorRateCurve",
    "ModalGeneratorRateDistortionPoint",
    "apply_modal_generator",
    "fit_modal_generator_rate_curve",
    "modal_generator_site_sha256",
]


_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_BINDING_KIND = "fisher_graph.modal_generator_binding"
_CONFIG_KIND = "fisher_graph.modal_generator_config"
_FACTORS_KIND = "fisher_graph.modal_generator_factors"
_PLAN_KIND = "fisher_graph.modal_generator_plan"
_POINT_KIND = "fisher_graph.modal_generator_rate_distortion_point"
_CURVE_KIND = "fisher_graph.modal_generator_rate_curve"

_BINDING_DOMAIN = b"fisher_graph.modal_generator.binding.v1\0"
_CONFIG_DOMAIN = b"fisher_graph.modal_generator.config.v1\0"
_FACTORS_DOMAIN = b"fisher_graph.modal_generator.factors.v1\0"
_PLAN_DOMAIN = b"fisher_graph.modal_generator.plan.v1\0"
_POINT_DOMAIN = b"fisher_graph.modal_generator.point.v1\0"
_CURVE_DOMAIN = b"fisher_graph.modal_generator.curve.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.modal_generator.tensor.v1\0"
_SOURCE_TENSOR_DOMAIN = b"fisher_graph.modal_generator.source_tensor.v1\0"
_SITE_DOMAIN = b"fisher_graph.modal_generator.site.v1\0"

_INPUT_KINDS = frozenset(
    {
        "native_layer_input",
        "native_block_input",
    }
)
_TARGET_KINDS = frozenset(
    {
        "cluster_residual_contribution",
        "computational_mode_coordinates",
    }
)
_BIAS_MAC_POLICIES = frozenset(
    {
        "count_bias_additions",
        "matrix_multiplies_only",
    }
)
_SELECTION_RULES = frozenset({"return_all", "fixed_rank"})

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_raw_activation_rows": False,
    "contains_native_mode_activation_rows": False,
    "contains_native_target_rows": False,
    "contains_generator_weights": True,
    "executable": True,
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
            f"{label} must be a source-safe identifier using only "
            "letters, numbers, '.', '_', ':', '/', and '-'"
        )
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if type(value) not in (float, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _expect_fields(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


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
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be a finite CPU float64 Tensor")
    canonical = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        f"{tuple(canonical.shape)}\0float64\0".encode("utf-8")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _as_float64_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a nonempty floating matrix")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result.clone()


def _as_weights(
    value: Tensor | None,
    *,
    observations: int,
    label: str,
) -> Tensor:
    if value is None:
        return torch.ones(observations, dtype=torch.float64)
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.shape[0] != observations
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a floating vector with shape "
            f"({observations},)"
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{label} must contain finite nonnegative values")
    if float(result.sum().item()) <= 0:
        raise ValueError(f"{label} must have positive total weight")
    return result.clone()


def modal_generator_site_sha256(site: str) -> str:
    """Hash a source-safe runtime site identifier."""

    value = _require_identifier(site, label="site")
    digest = hashlib.sha256()
    digest.update(_SITE_DOMAIN)
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModalGeneratorBinding:
    """Authenticated source, catalog, site, and split bindings."""

    generator_id: str
    input_kind: str
    input_site: str
    input_site_sha256: str
    target_kind: str
    output_site: str
    output_site_sha256: str
    source_model_sha256: str
    input_catalog_sha256: str
    output_catalog_sha256: str
    cluster_plan_sha256: str
    fit_split_sha256: str
    eval_split_sha256: str
    fisher_coupling_sha256: str | None = None
    computational_mode_basis_sha256: str | None = None
    parameter_cluster_fragment_sha256: str | None = None
    source_generator_plan_sha256: str | None = None
    artifact_sha256: str = ""
    artifact_kind: str = _BINDING_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.generator_id, label="generator_id")
        if self.input_kind not in _INPUT_KINDS:
            raise ValueError(
                "input_kind must identify a native layer or block input; "
                "native cluster targets or mode activations are circular"
            )
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError(
                "target_kind must identify a dense cluster residual "
                "contribution or frozen computational-mode coordinates"
            )
        input_site = _require_identifier(
            self.input_site,
            label="input_site",
        )
        output_site = _require_identifier(
            self.output_site,
            label="output_site",
        )
        if input_site == output_site:
            raise ValueError(
                "input_site and output_site must differ; a generator may "
                "not consume its own native target"
            )
        if (
            _require_sha256(
                self.input_site_sha256,
                label="input_site_sha256",
            )
            != modal_generator_site_sha256(input_site)
        ):
            raise ValueError("input_site_sha256 does not match input_site")
        if (
            _require_sha256(
                self.output_site_sha256,
                label="output_site_sha256",
            )
            != modal_generator_site_sha256(output_site)
        ):
            raise ValueError("output_site_sha256 does not match output_site")
        for name in (
            "source_model_sha256",
            "input_catalog_sha256",
            "output_catalog_sha256",
            "cluster_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name in (
            "fisher_coupling_sha256",
            "computational_mode_basis_sha256",
            "parameter_cluster_fragment_sha256",
            "source_generator_plan_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, label=name)
        if self.target_kind == "computational_mode_coordinates":
            if self.fisher_coupling_sha256 is None:
                raise ValueError(
                    "coordinate-target generators must bind the Fisher "
                    "coupling artifact"
                )
            if self.computational_mode_basis_sha256 is None:
                raise ValueError(
                    "coordinate-target generators must bind the "
                    "computational-mode basis"
                )
            if self.parameter_cluster_fragment_sha256 is None:
                raise ValueError(
                    "coordinate-target generators must bind the individual "
                    "parameter-cluster layer fragment"
                )
        if self.input_catalog_sha256 == self.output_catalog_sha256:
            raise ValueError(
                "input and output catalogs must differ; a generator may "
                "not declare its native target catalog as its input"
            )
        if self.fit_split_sha256 == self.eval_split_sha256:
            raise ValueError(
                "fit and evaluation split hashes must differ; hashes alone "
                "do not prove membership disjointness"
            )
        if (
            self.artifact_kind != _BINDING_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator binding header is invalid")

        expected = _json_sha256(
            self._payload(),
            domain=_BINDING_DOMAIN,
        )
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != expected:
                raise ValueError("modal generator binding hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    @classmethod
    def create(
        cls,
        *,
        generator_id: str,
        input_kind: str,
        input_site: str,
        output_site: str,
        source_model_sha256: str,
        input_catalog_sha256: str,
        output_catalog_sha256: str,
        cluster_plan_sha256: str,
        fit_split_sha256: str,
        eval_split_sha256: str,
        target_kind: str = "cluster_residual_contribution",
        fisher_coupling_sha256: str | None = None,
        computational_mode_basis_sha256: str | None = None,
        parameter_cluster_fragment_sha256: str | None = None,
        source_generator_plan_sha256: str | None = None,
    ) -> ModalGeneratorBinding:
        return cls(
            generator_id=generator_id,
            input_kind=input_kind,
            input_site=input_site,
            input_site_sha256=modal_generator_site_sha256(input_site),
            target_kind=target_kind,
            output_site=output_site,
            output_site_sha256=modal_generator_site_sha256(output_site),
            source_model_sha256=source_model_sha256,
            input_catalog_sha256=input_catalog_sha256,
            output_catalog_sha256=output_catalog_sha256,
            cluster_plan_sha256=cluster_plan_sha256,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
            fisher_coupling_sha256=fisher_coupling_sha256,
            computational_mode_basis_sha256=(
                computational_mode_basis_sha256
            ),
            parameter_cluster_fragment_sha256=(
                parameter_cluster_fragment_sha256
            ),
            source_generator_plan_sha256=source_generator_plan_sha256,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "generator_id": self.generator_id,
            "input_kind": self.input_kind,
            "input_site": self.input_site,
            "input_site_sha256": self.input_site_sha256,
            "target_kind": self.target_kind,
            "output_site": self.output_site,
            "output_site_sha256": self.output_site_sha256,
            "source_model_sha256": self.source_model_sha256,
            "input_catalog_sha256": self.input_catalog_sha256,
            "output_catalog_sha256": self.output_catalog_sha256,
            "cluster_plan_sha256": self.cluster_plan_sha256,
            "fit_split_sha256": self.fit_split_sha256,
            "eval_split_sha256": self.eval_split_sha256,
            "fisher_coupling_sha256": self.fisher_coupling_sha256,
            "computational_mode_basis_sha256": (
                self.computational_mode_basis_sha256
            ),
            "parameter_cluster_fragment_sha256": (
                self.parameter_cluster_fragment_sha256
            ),
            "source_generator_plan_sha256": (
                self.source_generator_plan_sha256
            ),
        }

    def validate_integrity(self) -> None:
        expected = _json_sha256(
            self._payload(),
            domain=_BINDING_DOMAIN,
        )
        if self.artifact_sha256 != expected:
            raise ValueError("modal generator binding hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorBinding:
        fields = {
            "artifact_kind",
            "format_version",
            "generator_id",
            "input_kind",
            "input_site",
            "input_site_sha256",
            "target_kind",
            "output_site",
            "output_site_sha256",
            "source_model_sha256",
            "input_catalog_sha256",
            "output_catalog_sha256",
            "cluster_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "fisher_coupling_sha256",
            "computational_mode_basis_sha256",
            "parameter_cluster_fragment_sha256",
            "source_generator_plan_sha256",
            "artifact_sha256",
        }
        _expect_fields(state, fields, label="modal generator binding")
        _require_sha256(state["artifact_sha256"], label="artifact_sha256")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalGeneratorConfig:
    """Predeclared rank ladder, objective, accounting, and selection rule."""

    ranks: tuple[int, ...]
    fit_intercept: bool = True
    ridge: float = 0.0
    bias_mac_policy: str = "count_bias_additions"
    selection_rule: str = "return_all"
    selected_rank: int | None = None
    artifact_sha256: str = ""
    artifact_kind: str = _CONFIG_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.ranks) is not tuple or not self.ranks:
            raise ValueError("ranks must be a nonempty tuple")
        for index, rank in enumerate(self.ranks):
            _require_exact_int(rank, label=f"ranks[{index}]", minimum=1)
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("ranks must be unique and strictly increasing")
        if type(self.fit_intercept) is not bool:
            raise TypeError("fit_intercept must be a bool")
        ridge = _require_finite_float(
            self.ridge,
            label="ridge",
            minimum=0.0,
        )
        object.__setattr__(self, "ridge", ridge)
        if self.bias_mac_policy not in _BIAS_MAC_POLICIES:
            raise ValueError("bias_mac_policy is unsupported")
        if self.selection_rule not in _SELECTION_RULES:
            raise ValueError("selection_rule is unsupported")
        if self.selection_rule == "return_all":
            if self.selected_rank is not None:
                raise ValueError(
                    "return_all cannot declare a selected_rank"
                )
        else:
            if (
                type(self.selected_rank) is not int
                or self.selected_rank not in self.ranks
            ):
                raise ValueError(
                    "fixed_rank requires a selected_rank in the rank ladder"
                )
        if (
            self.artifact_kind != _CONFIG_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator config header is invalid")
        expected = _json_sha256(self._payload(), domain=_CONFIG_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != expected:
                raise ValueError("modal generator config hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "ranks": self.ranks,
            "fit_intercept": self.fit_intercept,
            "ridge": self.ridge,
            "bias_mac_policy": self.bias_mac_policy,
            "selection_rule": self.selection_rule,
            "selected_rank": self.selected_rank,
            "objective": "weighted_reduced_rank_least_squares",
            "weight_normalizer": "mean_one_over_fit_rows",
        }

    def validate_integrity(self) -> None:
        expected = _json_sha256(
            self._payload(),
            domain=_CONFIG_DOMAIN,
        )
        if self.artifact_sha256 != expected:
            raise ValueError("modal generator config hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorConfig:
        fields = {
            "artifact_kind",
            "format_version",
            "ranks",
            "fit_intercept",
            "ridge",
            "bias_mac_policy",
            "selection_rule",
            "selected_rank",
            "objective",
            "weight_normalizer",
            "artifact_sha256",
        }
        _expect_fields(state, fields, label="modal generator config")
        if state["objective"] != "weighted_reduced_rank_least_squares":
            raise ValueError("modal generator objective is invalid")
        if state["weight_normalizer"] != "mean_one_over_fit_rows":
            raise ValueError("modal generator weight normalizer is invalid")
        _require_sha256(state["artifact_sha256"], label="artifact_sha256")
        return cls(
            ranks=state["ranks"],  # type: ignore[arg-type]
            fit_intercept=state["fit_intercept"],  # type: ignore[arg-type]
            ridge=state["ridge"],  # type: ignore[arg-type]
            bias_mac_policy=state["bias_mac_policy"],  # type: ignore[arg-type]
            selection_rule=state["selection_rule"],  # type: ignore[arg-type]
            selected_rank=state["selected_rank"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModalGeneratorFactors:
    """Factorized executable map with authenticated CPU float64 tensors."""

    rank: int
    input_factor: Tensor
    output_factor: Tensor
    bias: Tensor | None
    input_factor_sha256: str = ""
    output_factor_sha256: str = ""
    bias_sha256: str | None = ""
    artifact_sha256: str = ""
    artifact_kind: str = _FACTORS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        rank = _require_exact_int(self.rank, label="rank", minimum=1)
        if (
            not isinstance(self.input_factor, Tensor)
            or self.input_factor.device.type != "cpu"
            or self.input_factor.dtype != torch.float64
            or self.input_factor.ndim != 2
            or self.input_factor.shape[0] <= 0
            or self.input_factor.shape[1] != rank
            or not torch.isfinite(self.input_factor).all()
        ):
            raise ValueError(
                "input_factor must be a finite CPU float64 "
                "[input_width, rank] Tensor"
            )
        if (
            not isinstance(self.output_factor, Tensor)
            or self.output_factor.device.type != "cpu"
            or self.output_factor.dtype != torch.float64
            or self.output_factor.ndim != 2
            or self.output_factor.shape[0] != rank
            or self.output_factor.shape[1] <= 0
            or not torch.isfinite(self.output_factor).all()
        ):
            raise ValueError(
                "output_factor must be a finite CPU float64 "
                "[rank, output_width] Tensor"
            )
        if self.bias is None:
            if self.bias_sha256 not in ("", None):
                raise ValueError(
                    "bias_sha256 must be absent when bias is absent"
                )
            object.__setattr__(self, "bias_sha256", None)
        elif (
            not isinstance(self.bias, Tensor)
            or self.bias.device.type != "cpu"
            or self.bias.dtype != torch.float64
            or self.bias.shape != (self.output_factor.shape[1],)
            or not torch.isfinite(self.bias).all()
        ):
            raise ValueError(
                "bias must be absent or a finite CPU float64 "
                "output-width vector"
            )
        for name in ("input_factor", "output_factor"):
            object.__setattr__(
                self,
                name,
                getattr(self, name).detach().clone().contiguous(),
            )
        if self.bias is not None:
            object.__setattr__(
                self,
                "bias",
                self.bias.detach().clone().contiguous(),
            )
        hashes = {
            name: _tensor_sha256(getattr(self, name), label=name)
            for name in ("input_factor", "output_factor")
        }
        if self.bias is not None:
            hashes["bias"] = _tensor_sha256(self.bias, label="bias")
        for name, expected in hashes.items():
            field = f"{name}_sha256"
            supplied = getattr(self, field)
            if supplied:
                _require_sha256(supplied, label=field)
                if supplied != expected:
                    raise ValueError(f"{field} does not match tensor")
            else:
                object.__setattr__(self, field, expected)
        if (
            self.artifact_kind != _FACTORS_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator factors header is invalid")
        expected_artifact = _json_sha256(
            self._payload(),
            domain=_FACTORS_DOMAIN,
        )
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != expected_artifact:
                raise ValueError("modal generator factors hash mismatch")
        else:
            object.__setattr__(
                self,
                "artifact_sha256",
                expected_artifact,
            )

    @property
    def input_width(self) -> int:
        return int(self.input_factor.shape[0])

    @property
    def output_width(self) -> int:
        return int(self.output_factor.shape[1])

    @property
    def has_bias(self) -> bool:
        return self.bias is not None

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "rank": self.rank,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "has_bias": self.has_bias,
            "input_factor_sha256": self.input_factor_sha256,
            "output_factor_sha256": self.output_factor_sha256,
            "bias_sha256": self.bias_sha256,
        }

    def validate_integrity(self) -> None:
        rank = _require_exact_int(self.rank, label="rank", minimum=1)
        if (
            not isinstance(self.input_factor, Tensor)
            or self.input_factor.device.type != "cpu"
            or self.input_factor.dtype != torch.float64
            or self.input_factor.ndim != 2
            or self.input_factor.shape[0] <= 0
            or self.input_factor.shape[1] != rank
            or not torch.isfinite(self.input_factor).all()
        ):
            raise ValueError(
                "input_factor must be a finite CPU float64 "
                "[input_width, rank] Tensor"
            )
        if (
            not isinstance(self.output_factor, Tensor)
            or self.output_factor.device.type != "cpu"
            or self.output_factor.dtype != torch.float64
            or self.output_factor.ndim != 2
            or self.output_factor.shape[0] != rank
            or self.output_factor.shape[1] <= 0
            or not torch.isfinite(self.output_factor).all()
        ):
            raise ValueError(
                "output_factor must be a finite CPU float64 "
                "[rank, output_width] Tensor"
            )
        if self.bias is None:
            if self.bias_sha256 is not None:
                raise ValueError(
                    "bias_sha256 must be absent when bias is absent"
                )
        elif (
            not isinstance(self.bias, Tensor)
            or self.bias.device.type != "cpu"
            or self.bias.dtype != torch.float64
            or self.bias.shape != (self.output_factor.shape[1],)
            or not torch.isfinite(self.bias).all()
        ):
            raise ValueError(
                "bias must be absent or a finite CPU float64 "
                "output-width vector"
            )

        expected_hashes = {
            "input_factor_sha256": _tensor_sha256(
                self.input_factor,
                label="input_factor",
            ),
            "output_factor_sha256": _tensor_sha256(
                self.output_factor,
                label="output_factor",
            ),
        }
        if self.bias is not None:
            expected_hashes["bias_sha256"] = _tensor_sha256(
                self.bias,
                label="bias",
            )
        for field, expected in expected_hashes.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} does not match tensor")
        expected_artifact = _json_sha256(
            self._payload(),
            domain=_FACTORS_DOMAIN,
        )
        if self.artifact_sha256 != expected_artifact:
            raise ValueError("modal generator factors hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "input_factor": self.input_factor.clone(),
            "output_factor": self.output_factor.clone(),
            "bias": None if self.bias is None else self.bias.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorFactors:
        fields = {
            "artifact_kind",
            "format_version",
            "rank",
            "input_width",
            "output_width",
            "has_bias",
            "input_factor_sha256",
            "output_factor_sha256",
            "bias_sha256",
            "artifact_sha256",
            "input_factor",
            "output_factor",
            "bias",
        }
        _expect_fields(state, fields, label="modal generator factors")
        for field in (
            "input_factor_sha256",
            "output_factor_sha256",
            "artifact_sha256",
        ):
            _require_sha256(state[field], label=field)
        if state["bias_sha256"] is not None:
            _require_sha256(state["bias_sha256"], label="bias_sha256")
        result = cls(
            rank=state["rank"],  # type: ignore[arg-type]
            input_factor=state["input_factor"],  # type: ignore[arg-type]
            output_factor=state["output_factor"],  # type: ignore[arg-type]
            bias=state["bias"],  # type: ignore[arg-type]
            input_factor_sha256=state["input_factor_sha256"],  # type: ignore[arg-type]
            output_factor_sha256=state["output_factor_sha256"],  # type: ignore[arg-type]
            bias_sha256=state["bias_sha256"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if (
            _require_exact_int(
                state["input_width"],
                label="input_width",
                minimum=1,
            )
            != result.input_width
            or _require_exact_int(
                state["output_width"],
                label="output_width",
                minimum=1,
            )
            != result.output_width
            or state["has_bias"] is not result.has_bias
        ):
            raise ValueError("serialized factor widths do not match tensors")
        return result

    def apply(self, inputs: Tensor) -> Tensor:
        return apply_modal_generator(inputs, self)


@dataclass(frozen=True, slots=True)
class ModalGeneratorMetrics:
    """Rate-distortion metrics on one fixed split."""

    observations: int
    mse: float
    nrmse: float
    weighted_mse: float
    weighted_nrmse: float
    cosine_similarity: float
    weighted_cosine_similarity: float
    max_abs_error: float
    target_rms: float
    weighted_target_rms: float

    def __post_init__(self) -> None:
        _require_exact_int(
            self.observations,
            label="observations",
            minimum=1,
        )
        for name in (
            "mse",
            "nrmse",
            "weighted_mse",
            "weighted_nrmse",
            "max_abs_error",
            "target_rms",
            "weighted_target_rms",
        ):
            object.__setattr__(
                self,
                name,
                _require_finite_float(
                    getattr(self, name),
                    label=name,
                    minimum=0.0,
                ),
            )
        for name in (
            "cosine_similarity",
            "weighted_cosine_similarity",
        ):
            value = _require_finite_float(getattr(self, name), label=name)
            if value < -1.000000000001 or value > 1.000000000001:
                raise ValueError(f"{name} must be in [-1, 1]")
            object.__setattr__(
                self,
                name,
                max(-1.0, min(1.0, value)),
            )

    def metadata(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "mse": self.mse,
            "nrmse": self.nrmse,
            "weighted_mse": self.weighted_mse,
            "weighted_nrmse": self.weighted_nrmse,
            "cosine_similarity": self.cosine_similarity,
            "weighted_cosine_similarity": (
                self.weighted_cosine_similarity
            ),
            "max_abs_error": self.max_abs_error,
            "target_rms": self.target_rms,
            "weighted_target_rms": self.weighted_target_rms,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorMetrics:
        fields = {
            "observations",
            "mse",
            "nrmse",
            "weighted_mse",
            "weighted_nrmse",
            "cosine_similarity",
            "weighted_cosine_similarity",
            "max_abs_error",
            "target_rms",
            "weighted_target_rms",
        }
        _expect_fields(state, fields, label="modal generator metrics")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalGeneratorPlan:
    """One authenticated, independently executable generator candidate."""

    binding: ModalGeneratorBinding
    config: ModalGeneratorConfig
    factors: ModalGeneratorFactors
    parameter_count: int
    macs_per_token: int
    artifact_sha256: str = ""
    artifact_kind: str = _PLAN_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModalGeneratorBinding):
            raise TypeError("binding must be ModalGeneratorBinding")
        if not isinstance(self.config, ModalGeneratorConfig):
            raise TypeError("config must be ModalGeneratorConfig")
        if not isinstance(self.factors, ModalGeneratorFactors):
            raise TypeError("factors must be ModalGeneratorFactors")
        if self.factors.rank not in self.config.ranks:
            raise ValueError("factor rank is not in the configured ladder")
        # Strictly canonicalize legacy all-zero placeholder biases.  This
        # preserves construction compatibility while ensuring a bias-free
        # executable artifact does not retain or authenticate nonexistent
        # coefficients.
        if not self.config.fit_intercept and self.factors.bias is not None:
            if bool(torch.count_nonzero(self.factors.bias)):
                raise ValueError(
                    "bias must be absent when fit_intercept is false"
                )
            object.__setattr__(
                self,
                "factors",
                ModalGeneratorFactors(
                    rank=self.factors.rank,
                    input_factor=self.factors.input_factor,
                    output_factor=self.factors.output_factor,
                    bias=None,
                ),
            )
        expected_parameters, expected_macs = _resource_counts(
            input_width=self.factors.input_width,
            output_width=self.factors.output_width,
            rank=self.factors.rank,
            fit_intercept=self.config.fit_intercept,
            bias_mac_policy=self.config.bias_mac_policy,
        )
        if (
            _require_exact_int(
                self.parameter_count,
                label="parameter_count",
                minimum=1,
            )
            != expected_parameters
        ):
            raise ValueError("parameter_count does not match factor shapes")
        if (
            _require_exact_int(
                self.macs_per_token,
                label="macs_per_token",
                minimum=1,
            )
            != expected_macs
        ):
            raise ValueError("macs_per_token does not match accounting policy")
        if self.config.fit_intercept is not self.factors.has_bias:
            raise ValueError(
                "factor bias presence must exactly match fit_intercept"
            )
        if (
            self.artifact_kind != _PLAN_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator plan header is invalid")
        expected = _json_sha256(self._payload(), domain=_PLAN_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != expected:
                raise ValueError("modal generator plan hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    @property
    def rank(self) -> int:
        return self.factors.rank

    @property
    def input_width(self) -> int:
        return self.factors.input_width

    @property
    def output_width(self) -> int:
        return self.factors.output_width

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "binding_sha256": self.binding.artifact_sha256,
            "config_sha256": self.config.artifact_sha256,
            "factors_sha256": self.factors.artifact_sha256,
            "rank": self.rank,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "parameter_count": self.parameter_count,
            "macs_per_token": self.macs_per_token,
        }

    def validate_integrity(self) -> None:
        self.binding.validate_integrity()
        self.config.validate_integrity()
        self.factors.validate_integrity()
        if self.factors.rank not in self.config.ranks:
            raise ValueError("factor rank is not in the configured ladder")
        expected_parameters, expected_macs = _resource_counts(
            input_width=self.factors.input_width,
            output_width=self.factors.output_width,
            rank=self.factors.rank,
            fit_intercept=self.config.fit_intercept,
            bias_mac_policy=self.config.bias_mac_policy,
        )
        if self.parameter_count != expected_parameters:
            raise ValueError("parameter_count does not match factor shapes")
        if self.macs_per_token != expected_macs:
            raise ValueError("macs_per_token does not match accounting policy")
        if self.config.fit_intercept is not self.factors.has_bias:
            raise ValueError(
                "factor bias presence must exactly match fit_intercept"
            )
        expected = _json_sha256(self._payload(), domain=_PLAN_DOMAIN)
        if self.artifact_sha256 != expected:
            raise ValueError("modal generator plan hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "binding": self.binding.state_dict(),
            "config": self.config.state_dict(),
            "factors": self.factors.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorPlan:
        fields = {
            "artifact_kind",
            "format_version",
            "binding_sha256",
            "config_sha256",
            "factors_sha256",
            "rank",
            "input_width",
            "output_width",
            "parameter_count",
            "macs_per_token",
            "artifact_sha256",
            "binding",
            "config",
            "factors",
        }
        _expect_fields(state, fields, label="modal generator plan")
        binding = ModalGeneratorBinding.from_state_dict(
            state["binding"]  # type: ignore[arg-type]
        )
        config = ModalGeneratorConfig.from_state_dict(
            state["config"]  # type: ignore[arg-type]
        )
        factors = ModalGeneratorFactors.from_state_dict(
            state["factors"]  # type: ignore[arg-type]
        )
        for field, actual in (
            ("binding_sha256", binding.artifact_sha256),
            ("config_sha256", config.artifact_sha256),
            ("factors_sha256", factors.artifact_sha256),
        ):
            if _require_sha256(state[field], label=field) != actual:
                raise ValueError(f"{field} does not match nested artifact")
        result = cls(
            binding=binding,
            config=config,
            factors=factors,
            parameter_count=state["parameter_count"],  # type: ignore[arg-type]
            macs_per_token=state["macs_per_token"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        for field, actual in (
            ("rank", result.rank),
            ("input_width", result.input_width),
            ("output_width", result.output_width),
        ):
            if (
                _require_exact_int(
                    state[field],
                    label=field,
                    minimum=1,
                )
                != actual
            ):
                raise ValueError(f"serialized {field} is inconsistent")
        return result

    def apply(self, inputs: Tensor) -> Tensor:
        return apply_modal_generator(inputs, self)


@dataclass(frozen=True, slots=True)
class ModalGeneratorRateDistortionPoint:
    """One rank and its fixed fit/evaluation measurements."""

    plan: ModalGeneratorPlan
    fit_metrics: ModalGeneratorMetrics
    eval_metrics: ModalGeneratorMetrics
    artifact_sha256: str = ""
    artifact_kind: str = _POINT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ModalGeneratorPlan):
            raise TypeError("plan must be ModalGeneratorPlan")
        if not isinstance(self.fit_metrics, ModalGeneratorMetrics):
            raise TypeError("fit_metrics must be ModalGeneratorMetrics")
        if not isinstance(self.eval_metrics, ModalGeneratorMetrics):
            raise TypeError("eval_metrics must be ModalGeneratorMetrics")
        if (
            self.artifact_kind != _POINT_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("rate-distortion point header is invalid")
        expected = _json_sha256(self._payload(), domain=_POINT_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != expected:
                raise ValueError("rate-distortion point hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    @property
    def rank(self) -> int:
        return self.plan.rank

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "plan_sha256": self.plan.artifact_sha256,
            "rank": self.rank,
            "fit_metrics": self.fit_metrics.metadata(),
            "eval_metrics": self.eval_metrics.metadata(),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "plan": self.plan.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorRateDistortionPoint:
        fields = {
            "artifact_kind",
            "format_version",
            "plan_sha256",
            "rank",
            "fit_metrics",
            "eval_metrics",
            "artifact_sha256",
            "plan",
        }
        _expect_fields(state, fields, label="rate-distortion point")
        plan = ModalGeneratorPlan.from_state_dict(
            state["plan"]  # type: ignore[arg-type]
        )
        if (
            _require_sha256(
                state["plan_sha256"],
                label="plan_sha256",
            )
            != plan.artifact_sha256
        ):
            raise ValueError("plan_sha256 does not match nested plan")
        if (
            _require_exact_int(
                state["rank"],
                label="rank",
                minimum=1,
            )
            != plan.rank
        ):
            raise ValueError("serialized point rank is inconsistent")
        return cls(
            plan=plan,
            fit_metrics=ModalGeneratorMetrics.from_state_dict(
                state["fit_metrics"]  # type: ignore[arg-type]
            ),
            eval_metrics=ModalGeneratorMetrics.from_state_dict(
                state["eval_metrics"]  # type: ignore[arg-type]
            ),
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModalGeneratorRateCurve:
    """Authenticated rank ladder with a literal zero/deletion baseline."""

    binding: ModalGeneratorBinding
    config: ModalGeneratorConfig
    input_width: int
    output_width: int
    zero_fit_metrics: ModalGeneratorMetrics
    zero_eval_metrics: ModalGeneratorMetrics
    points: tuple[ModalGeneratorRateDistortionPoint, ...]
    fit_inputs_sha256: str
    fit_targets_sha256: str
    fit_fisher_weights_sha256: str
    eval_inputs_sha256: str
    eval_targets_sha256: str
    eval_fisher_weights_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _CURVE_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_raw_activation_rows: bool = False
    contains_native_mode_activation_rows: bool = False
    contains_native_target_rows: bool = False
    contains_generator_weights: bool = True
    executable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModalGeneratorBinding):
            raise TypeError("binding must be ModalGeneratorBinding")
        if not isinstance(self.config, ModalGeneratorConfig):
            raise TypeError("config must be ModalGeneratorConfig")
        input_width = _require_exact_int(
            self.input_width,
            label="input_width",
            minimum=1,
        )
        output_width = _require_exact_int(
            self.output_width,
            label="output_width",
            minimum=1,
        )
        if not isinstance(self.zero_fit_metrics, ModalGeneratorMetrics):
            raise TypeError("zero_fit_metrics must be ModalGeneratorMetrics")
        if not isinstance(self.zero_eval_metrics, ModalGeneratorMetrics):
            raise TypeError("zero_eval_metrics must be ModalGeneratorMetrics")
        for name in (
            "fit_inputs_sha256",
            "fit_targets_sha256",
            "fit_fisher_weights_sha256",
            "eval_inputs_sha256",
            "eval_targets_sha256",
            "eval_fisher_weights_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        fit_source_triplet = (
            self.fit_inputs_sha256,
            self.fit_targets_sha256,
            self.fit_fisher_weights_sha256,
        )
        eval_source_triplet = (
            self.eval_inputs_sha256,
            self.eval_targets_sha256,
            self.eval_fisher_weights_sha256,
        )
        if fit_source_triplet == eval_source_triplet:
            raise ValueError(
                "fit and evaluation source triplets must differ"
            )
        if type(self.points) is not tuple or not self.points:
            raise ValueError("points must be a nonempty tuple")
        if any(
            not isinstance(point, ModalGeneratorRateDistortionPoint)
            for point in self.points
        ):
            raise TypeError("points contain an invalid item")
        ranks = tuple(point.rank for point in self.points)
        if ranks != self.config.ranks:
            raise ValueError("point ranks must equal the configured ladder")
        for point in self.points:
            plan = point.plan
            if (
                plan.binding.artifact_sha256
                != self.binding.artifact_sha256
                or plan.config.artifact_sha256
                != self.config.artifact_sha256
                or plan.input_width != input_width
                or plan.output_width != output_width
            ):
                raise ValueError(
                    "point plan does not match curve binding, config, or widths"
                )
            if (
                point.fit_metrics.observations
                != self.zero_fit_metrics.observations
                or point.eval_metrics.observations
                != self.zero_eval_metrics.observations
            ):
                raise ValueError(
                    "point metric observation counts do not match baselines"
                )
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError("modal generator safety metadata is invalid")
        if (
            self.artifact_kind != _CURVE_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal generator rate curve header is invalid")
        expected_hash = _json_sha256(self._payload(), domain=_CURVE_DOMAIN)
        if self.artifact_sha256:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != expected_hash:
                raise ValueError("modal generator rate curve hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", expected_hash)

    @property
    def selected_plan(self) -> ModalGeneratorPlan | None:
        if self.config.selection_rule == "return_all":
            return None
        return self.point_for_rank(self.config.selected_rank).plan

    def point_for_rank(
        self,
        rank: int | None,
    ) -> ModalGeneratorRateDistortionPoint:
        if type(rank) is not int:
            raise KeyError("rank must be an integer")
        for point in self.points:
            if point.rank == rank:
                return point
        raise KeyError(f"rank {rank} is not in the generator curve")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            **_SAFETY_METADATA,
            "binding_sha256": self.binding.artifact_sha256,
            "config_sha256": self.config.artifact_sha256,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "fit_inputs_sha256": self.fit_inputs_sha256,
            "fit_targets_sha256": self.fit_targets_sha256,
            "fit_fisher_weights_sha256": (
                self.fit_fisher_weights_sha256
            ),
            "eval_inputs_sha256": self.eval_inputs_sha256,
            "eval_targets_sha256": self.eval_targets_sha256,
            "eval_fisher_weights_sha256": (
                self.eval_fisher_weights_sha256
            ),
            "zero_fit_metrics": self.zero_fit_metrics.metadata(),
            "zero_eval_metrics": self.zero_eval_metrics.metadata(),
            "point_sha256s": tuple(
                point.artifact_sha256 for point in self.points
            ),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "binding": self.binding.state_dict(),
            "config": self.config.state_dict(),
            "points": tuple(point.state_dict() for point in self.points),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorRateCurve:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "binding_sha256",
            "config_sha256",
            "input_width",
            "output_width",
            "fit_inputs_sha256",
            "fit_targets_sha256",
            "fit_fisher_weights_sha256",
            "eval_inputs_sha256",
            "eval_targets_sha256",
            "eval_fisher_weights_sha256",
            "zero_fit_metrics",
            "zero_eval_metrics",
            "point_sha256s",
            "artifact_sha256",
            "binding",
            "config",
            "points",
        }
        _expect_fields(state, fields, label="modal generator rate curve")
        for name, expected in _SAFETY_METADATA.items():
            if state[name] is not expected:
                raise ValueError("modal generator safety metadata is invalid")
        binding = ModalGeneratorBinding.from_state_dict(
            state["binding"]  # type: ignore[arg-type]
        )
        config = ModalGeneratorConfig.from_state_dict(
            state["config"]  # type: ignore[arg-type]
        )
        if (
            _require_sha256(
                state["binding_sha256"],
                label="binding_sha256",
            )
            != binding.artifact_sha256
            or _require_sha256(
                state["config_sha256"],
                label="config_sha256",
            )
            != config.artifact_sha256
        ):
            raise ValueError("nested binding or config hash mismatch")
        raw_points = state["points"]
        if type(raw_points) is not tuple:
            raise TypeError("serialized points must be a tuple")
        points = tuple(
            ModalGeneratorRateDistortionPoint.from_state_dict(value)
            for value in raw_points  # type: ignore[arg-type]
        )
        raw_hashes = state["point_sha256s"]
        if type(raw_hashes) is not tuple or tuple(
            _require_sha256(value, label="point_sha256")
            for value in raw_hashes
        ) != tuple(point.artifact_sha256 for point in points):
            raise ValueError("point_sha256s do not match nested points")
        return cls(
            binding=binding,
            config=config,
            input_width=state["input_width"],  # type: ignore[arg-type]
            output_width=state["output_width"],  # type: ignore[arg-type]
            fit_inputs_sha256=state[
                "fit_inputs_sha256"
            ],  # type: ignore[arg-type]
            fit_targets_sha256=state[
                "fit_targets_sha256"
            ],  # type: ignore[arg-type]
            fit_fisher_weights_sha256=state[
                "fit_fisher_weights_sha256"
            ],  # type: ignore[arg-type]
            eval_inputs_sha256=state[
                "eval_inputs_sha256"
            ],  # type: ignore[arg-type]
            eval_targets_sha256=state[
                "eval_targets_sha256"
            ],  # type: ignore[arg-type]
            eval_fisher_weights_sha256=state[
                "eval_fisher_weights_sha256"
            ],  # type: ignore[arg-type]
            zero_fit_metrics=ModalGeneratorMetrics.from_state_dict(
                state["zero_fit_metrics"]  # type: ignore[arg-type]
            ),
            zero_eval_metrics=ModalGeneratorMetrics.from_state_dict(
                state["zero_eval_metrics"]  # type: ignore[arg-type]
            ),
            points=points,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name]
                for name in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )


def _resource_counts(
    *,
    input_width: int,
    output_width: int,
    rank: int,
    fit_intercept: bool,
    bias_mac_policy: str,
) -> tuple[int, int]:
    matrix_coefficients = input_width * rank + rank * output_width
    bias_coefficients = output_width if fit_intercept else 0
    parameters = matrix_coefficients + bias_coefficients
    macs = matrix_coefficients
    if (
        fit_intercept
        and bias_mac_policy == "count_bias_additions"
    ):
        macs += output_width
    return parameters, macs


def _canonicalize_component_signs(
    input_factor: Tensor,
    output_factor: Tensor,
) -> tuple[Tensor, Tensor]:
    result_input = input_factor.clone()
    result_output = output_factor.clone()
    for index in range(output_factor.shape[0]):
        row = result_output[index]
        pivot = int(row.abs().argmax().item())
        if float(row[pivot].item()) < 0:
            result_input[:, index].neg_()
            result_output[index].neg_()
    return result_input, result_output


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
    tolerance = (
        64.0 * torch.finfo(torch.float64).eps * max(width, 1)
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


def _canonicalize_eigenbasis(
    basis: Tensor,
    eigenvalues: Tensor,
) -> Tensor:
    """Remove arbitrary rotations from tied and null eigenspaces."""

    if (
        basis.ndim != 2
        or basis.shape[0] != basis.shape[1]
        or eigenvalues.shape != (basis.shape[1],)
        or basis.dtype != torch.float64
        or eigenvalues.dtype != torch.float64
        or basis.device.type != "cpu"
        or eigenvalues.device.type != "cpu"
    ):
        raise ValueError("eigenbasis canonicalization inputs are invalid")
    count = eigenvalues.numel()
    if count == 0:
        return basis.clone().contiguous()
    eps = torch.finfo(torch.float64).eps
    largest = max(float(eigenvalues[0].abs().item()), 0.0)
    zero_tolerance = 8.0 * eps * max(count, 1) * largest
    supported = int((eigenvalues > zero_tolerance).sum().item())
    blocks: list[Tensor] = []
    start = 0
    while start < supported:
        stop = start + 1
        while stop < supported:
            left = float(eigenvalues[start].item())
            right = float(eigenvalues[stop].item())
            local_scale = max(abs(left), abs(right))
            tie_tolerance = (
                8.0 * eps * max(count, 1) * local_scale
            )
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
    null_count = count - supported
    if null_count:
        null_projector = torch.eye(count, dtype=torch.float64)
        if supported:
            null_projector -= supported_basis @ supported_basis.T
        blocks.append(
            _canonical_basis_from_projector(
                null_projector,
                dimension=null_count,
            )
        )
    result = torch.cat(blocks, dim=1).contiguous()
    if not torch.allclose(
        result.T @ result,
        torch.eye(count, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-11,
    ):
        raise RuntimeError("canonical eigenbasis lost orthonormality")
    return result


def _cosine(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor | None,
) -> float:
    if weights is None:
        left = target.reshape(-1)
        right = prediction.reshape(-1)
    else:
        root = weights.sqrt().unsqueeze(1)
        left = (target * root).reshape(-1)
        right = (prediction * root).reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    if left_norm == 0 and right_norm == 0:
        return 1.0
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(torch.dot(left, right).item()) / (
        left_norm * right_norm
    )


def _normalized_rmse(error_power: float, target_power: float) -> float:
    if target_power == 0.0:
        if error_power == 0.0:
            return 0.0
        target_power = torch.finfo(torch.float64).tiny
    return math.sqrt(error_power / target_power)


def _metrics(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor,
) -> ModalGeneratorMetrics:
    error = prediction - target
    mse = float(error.square().mean().item())
    target_power = float(target.square().mean().item())
    normalized_weights = weights / weights.sum()
    weighted_row_error = error.square().mean(dim=1)
    weighted_row_target = target.square().mean(dim=1)
    weighted_mse = float(
        torch.dot(normalized_weights, weighted_row_error).item()
    )
    weighted_target_power = float(
        torch.dot(normalized_weights, weighted_row_target).item()
    )
    return ModalGeneratorMetrics(
        observations=target.shape[0],
        mse=mse,
        nrmse=_normalized_rmse(mse, target_power),
        weighted_mse=weighted_mse,
        weighted_nrmse=_normalized_rmse(
            weighted_mse,
            weighted_target_power,
        ),
        cosine_similarity=_cosine(target, prediction, None),
        weighted_cosine_similarity=_cosine(
            target,
            prediction,
            normalized_weights,
        ),
        max_abs_error=float(error.abs().max().item()),
        target_rms=math.sqrt(target_power),
        weighted_target_rms=math.sqrt(weighted_target_power),
    )


def apply_modal_generator(
    inputs: Tensor,
    generator: ModalGeneratorPlan | ModalGeneratorFactors,
) -> Tensor:
    """Execute a fitted generator without consulting native cluster modes."""

    if isinstance(generator, ModalGeneratorPlan):
        generator.validate_integrity()
        factors = generator.factors
    elif isinstance(generator, ModalGeneratorFactors):
        generator.validate_integrity()
        factors = generator
    else:
        raise TypeError(
            "generator must be ModalGeneratorPlan or ModalGeneratorFactors"
        )
    if (
        not isinstance(inputs, Tensor)
        or not inputs.is_floating_point()
        or inputs.ndim < 1
        or inputs.shape[-1] != factors.input_width
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError(
            "inputs must be a finite floating Tensor whose final dimension "
            "matches the generator input width"
        )
    input_factor = factors.input_factor.to(
        device=inputs.device,
        dtype=inputs.dtype,
    )
    output_factor = factors.output_factor.to(
        device=inputs.device,
        dtype=inputs.dtype,
    )
    bias = (
        None
        if factors.bias is None
        else factors.bias.to(device=inputs.device, dtype=inputs.dtype)
    )
    if (
        not torch.isfinite(input_factor).all()
        or not torch.isfinite(output_factor).all()
        or (bias is not None and not torch.isfinite(bias).all())
    ):
        raise ValueError(
            "generator factors are not finite in the runtime dtype"
        )
    result = (inputs @ input_factor) @ output_factor
    return result if bias is None else result + bias


def fit_modal_generator_rate_curve(
    X_fit: Tensor,
    Y_fit: Tensor,
    fisher_weights_fit: Tensor,
    X_eval: Tensor,
    Y_eval: Tensor,
    ranks: Sequence[int],
    *,
    binding: ModalGeneratorBinding,
    fisher_weights_eval: Tensor | None = None,
    fit_intercept: bool = True,
    ridge: float = 0.0,
    bias_mac_policy: str = "count_bias_additions",
    selection_rule: str = "return_all",
    selected_rank: int | None = None,
) -> ModalGeneratorRateCurve:
    """Fit a deterministic Fisher-weighted reduced-rank generator ladder.

    Evaluation rows never influence the fitted factors.  ``selection_rule``
    may either return the complete predeclared ladder without selecting a
    winner, or select one explicitly predeclared ``fixed_rank``.  Evaluation
    metrics are therefore descriptive and cannot silently tune the artifact.
    """

    if not isinstance(binding, ModalGeneratorBinding):
        raise TypeError("binding must be ModalGeneratorBinding")
    fit_inputs = _as_float64_matrix(X_fit, label="X_fit")
    fit_targets = _as_float64_matrix(Y_fit, label="Y_fit")
    eval_inputs = _as_float64_matrix(X_eval, label="X_eval")
    eval_targets = _as_float64_matrix(Y_eval, label="Y_eval")
    if fit_inputs.shape[0] != fit_targets.shape[0]:
        raise ValueError("X_fit and Y_fit row counts must match")
    if eval_inputs.shape[0] != eval_targets.shape[0]:
        raise ValueError("X_eval and Y_eval row counts must match")
    if fit_inputs.shape[1] != eval_inputs.shape[1]:
        raise ValueError("fit and evaluation input widths must match")
    if fit_targets.shape[1] != eval_targets.shape[1]:
        raise ValueError("fit and evaluation output widths must match")
    config = ModalGeneratorConfig(
        ranks=tuple(ranks),
        fit_intercept=fit_intercept,
        ridge=ridge,
        bias_mac_policy=bias_mac_policy,
        selection_rule=selection_rule,
        selected_rank=selected_rank,
    )
    maximum_rank = min(fit_inputs.shape[1], fit_targets.shape[1])
    if config.ranks[-1] > maximum_rank:
        raise ValueError(
            "requested ranks cannot exceed min(input_width, output_width)"
        )
    fit_weights = _as_weights(
        fisher_weights_fit,
        observations=fit_inputs.shape[0],
        label="fisher_weights_fit",
    )
    eval_weights = _as_weights(
        fisher_weights_eval,
        observations=eval_inputs.shape[0],
        label="fisher_weights_eval",
    )
    source_hashes = {
        "fit_inputs_sha256": _tensor_sha256(
            fit_inputs,
            label="X_fit",
            domain=_SOURCE_TENSOR_DOMAIN,
        ),
        "fit_targets_sha256": _tensor_sha256(
            fit_targets,
            label="Y_fit",
            domain=_SOURCE_TENSOR_DOMAIN,
        ),
        "fit_fisher_weights_sha256": _tensor_sha256(
            fit_weights,
            label="fisher_weights_fit",
            domain=_SOURCE_TENSOR_DOMAIN,
        ),
        "eval_inputs_sha256": _tensor_sha256(
            eval_inputs,
            label="X_eval",
            domain=_SOURCE_TENSOR_DOMAIN,
        ),
        "eval_targets_sha256": _tensor_sha256(
            eval_targets,
            label="Y_eval",
            domain=_SOURCE_TENSOR_DOMAIN,
        ),
        "eval_fisher_weights_sha256": _tensor_sha256(
            eval_weights,
            label="fisher_weights_eval",
            domain=_SOURCE_TENSOR_DOMAIN,
        ),
    }
    fit_source_triplet = (
        source_hashes["fit_inputs_sha256"],
        source_hashes["fit_targets_sha256"],
        source_hashes["fit_fisher_weights_sha256"],
    )
    eval_source_triplet = (
        source_hashes["eval_inputs_sha256"],
        source_hashes["eval_targets_sha256"],
        source_hashes["eval_fisher_weights_sha256"],
    )
    if fit_source_triplet == eval_source_triplet:
        raise ValueError(
            "fit and evaluation source triplets must differ; exact source "
            "reuse is obvious evaluation leakage"
        )

    fit_weight_sum = fit_weights.sum()
    if config.fit_intercept:
        fit_input_mean = (
            fit_inputs * fit_weights.unsqueeze(1)
        ).sum(dim=0) / fit_weight_sum
        fit_target_mean = (
            fit_targets * fit_weights.unsqueeze(1)
        ).sum(dim=0) / fit_weight_sum
    else:
        fit_input_mean = torch.zeros(
            fit_inputs.shape[1],
            dtype=torch.float64,
        )
        fit_target_mean = torch.zeros(
            fit_targets.shape[1],
            dtype=torch.float64,
        )
    centered_inputs = fit_inputs - fit_input_mean
    centered_targets = fit_targets - fit_target_mean

    # Mean-normalized weights make ridge invariant to multiplying every
    # Fisher weight by the same positive constant.
    normalized_fit_weights = (
        fit_weights * (fit_inputs.shape[0] / fit_weight_sum)
    )
    root_weights = normalized_fit_weights.sqrt().unsqueeze(1)
    weighted_inputs = centered_inputs * root_weights
    weighted_targets = centered_targets * root_weights
    normalizer = float(fit_inputs.shape[0])
    gram = weighted_inputs.T @ weighted_inputs / normalizer
    cross = weighted_inputs.T @ weighted_targets / normalizer
    if config.ridge:
        gram = gram + config.ridge * torch.eye(
            gram.shape[0],
            dtype=torch.float64,
        )
    full_coefficient = torch.linalg.pinv(
        gram,
        hermitian=True,
    ) @ cross

    # Under ridge, reduced-rank truncation must use the same penalized
    # coefficient metric as the full solve.  Omitting the ridge term here
    # selects directions for a different objective.
    output_covariance = full_coefficient.T @ gram @ full_coefficient
    output_covariance = (
        output_covariance + output_covariance.T
    ) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(output_covariance)
    eigenvalues = eigenvalues.flip(dims=(0,)).clamp_min(0.0)
    directions = eigenvectors.flip(dims=(1,))
    directions = _canonicalize_eigenbasis(directions, eigenvalues)

    zero_fit = torch.zeros_like(fit_targets)
    zero_eval = torch.zeros_like(eval_targets)
    points: list[ModalGeneratorRateDistortionPoint] = []
    for rank in config.ranks:
        output_directions = directions[:, :rank]
        input_factor = full_coefficient @ output_directions
        output_factor = output_directions.T
        input_factor, output_factor = _canonicalize_component_signs(
            input_factor,
            output_factor,
        )
        coefficient = input_factor @ output_factor
        if config.fit_intercept:
            bias = fit_target_mean - fit_input_mean @ coefficient
        else:
            bias = None
        factors = ModalGeneratorFactors(
            rank=rank,
            input_factor=input_factor,
            output_factor=output_factor,
            bias=bias,
        )
        parameter_count, macs_per_token = _resource_counts(
            input_width=factors.input_width,
            output_width=factors.output_width,
            rank=rank,
            fit_intercept=config.fit_intercept,
            bias_mac_policy=config.bias_mac_policy,
        )
        plan = ModalGeneratorPlan(
            binding=binding,
            config=config,
            factors=factors,
            parameter_count=parameter_count,
            macs_per_token=macs_per_token,
        )
        fit_prediction = factors.apply(fit_inputs)
        eval_prediction = factors.apply(eval_inputs)
        points.append(
            ModalGeneratorRateDistortionPoint(
                plan=plan,
                fit_metrics=_metrics(
                    fit_targets,
                    fit_prediction,
                    fit_weights,
                ),
                eval_metrics=_metrics(
                    eval_targets,
                    eval_prediction,
                    eval_weights,
                ),
            )
        )

    return ModalGeneratorRateCurve(
        binding=binding,
        config=config,
        input_width=fit_inputs.shape[1],
        output_width=fit_targets.shape[1],
        **source_hashes,
        zero_fit_metrics=_metrics(
            fit_targets,
            zero_fit,
            fit_weights,
        ),
        zero_eval_metrics=_metrics(
            eval_targets,
            zero_eval,
            eval_weights,
        ),
        points=tuple(points),
    )
