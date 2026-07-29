"""Compile a controlled graph-vs-SVD L3-to-L4 rate-distortion ladder.

The experiment uses the pinned prompt-free five-origin Gemma measurement.
Origins 8/24/40 fit every graph, basis, target decoder, and causal core;
origins 16/32 are read only after each plan is frozen.  The resulting plans
are executable fixed-reference modal-delta maps, but they are not complete
Gemma block replacements and do not establish NLL, task fidelity, latency, or
whole-model compression.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    SELECTION_ORIGINS,
    Gemma3SpectralSource,
    _file_sha256,
    _reserve_outputs,
    _response_binding,
    _stage_json,
    _stage_torch,
    _validate_output_path,
    load_gemma3_spectral_source,
)
from .graph_spectral_source_basis import (
    FitOnlyGraphSourceBasis,
    fit_graph_source_bases,
)


__all__ = [
    "DEFAULT_CUTOFFS",
    "DEFAULT_OUTPUT",
    "DEFAULT_RANDOM_SEEDS",
    "DEFAULT_TARGET_RANK",
    "Gemma3GraphSpectralRateDistortionCandidate",
    "build_parser",
    "compile_gemma3_graph_spectral_rate_distortion_candidate",
    "compile_gemma3_l3_l4_graph_spectral_rate_distortion",
    "load_gemma3_graph_spectral_rate_distortion_candidate",
    "main",
]


DEFAULT_CUTOFFS = (8, 16, 32, 45, 52)
DEFAULT_TARGET_RANK = 64
DEFAULT_RANDOM_SEEDS = (
    1729,
    3253,
    4513,
    6421,
    8191,
    10007,
    12289,
    16381,
)
DEFAULT_PERMUTATION_SEED = 271828
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-spectral-rate-distortion-dev-v1.pt"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_graph_spectral_rate_distortion_development"
)
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-graph-spectral-rate-distortion:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-graph-spectral-rate-distortion-report:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OFFLINE_MAX_RELATIVE_ERROR = math.sqrt(0.05)
_OFFLINE_MAX_P90_RELATIVE_ERROR = 0.30
_MIN_RESOURCE_REDUCTION = 0.20
_BASIS_FAMILIES = frozenset(
    {
        "signed_gfa",
        "magnitude_gfa",
        "svd",
        "native_prefix",
        "signed_row_permutation",
        "random_haar",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rate rows must be a sequence")
    result = json.loads(_canonical_json_bytes(list(rows)).decode("ascii"))
    if not isinstance(result, list) or any(
        not isinstance(row, dict) for row in result
    ):
        raise TypeError("rate rows must contain mappings")
    return tuple(result)


def _relative_error(prediction: Tensor, target: Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(target)),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.vector_norm(prediction - target)) / denominator


def _cosine(prediction: Tensor, target: Tensor) -> float:
    left = prediction.reshape(-1)
    right = target.reshape(-1)
    denominator = float(torch.linalg.vector_norm(left)) * float(
        torch.linalg.vector_norm(right)
    )
    if denominator <= torch.finfo(torch.float64).eps:
        return 1.0 if bool((left == right).all()) else 0.0
    return max(-1.0, min(1.0, float(torch.dot(left, right)) / denominator))


def _profile(value: Tensor, *, dimension: int) -> tuple[float, ...]:
    squared = value.square()
    reduce = tuple(index for index in range(value.ndim) if index != dimension)
    energy = squared.sum(dim=reduce)
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps:
        return tuple(0.0 for _ in range(int(energy.numel())))
    return tuple(float(item) for item in energy / total)


def _source_projection(basis: Tensor, value: Tensor) -> Tensor:
    flattened = value.reshape(value.shape[0], -1)
    return (
        basis @ (basis.T @ flattened)
    ).reshape_as(value).contiguous()


def _random_orthonormal(
    width: int,
    *,
    seed: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(
        (width, width),
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(raw)
    for column in range(width):
        pivot = int(torch.argmax(basis[:, column].abs()))
        if float(basis[pivot, column]) < 0.0:
            basis[:, column] *= -1.0
    return basis.contiguous()


def _metric_row(
    *,
    plan_key: str,
    basis_family: str,
    rank: int,
    random_seed: int | None,
    plan: ConditionalSpectralGeneratorPlan,
    weighted_truth: Tensor,
    origins: tuple[int, ...],
    sequence_positions: Tensor,
    valid_mask: Tensor,
    source_mask: Tensor,
) -> dict[str, object]:
    fit_predictions = torch.stack(
        tuple(plan.weighted_kernel_at_origin(origin) for origin in FIT_ORIGINS)
    )
    fit_truth = torch.stack(
        tuple(weighted_truth[:, origins.index(origin)] for origin in FIT_ORIGINS)
    )
    selection_predictions = torch.stack(
        tuple(
            plan.weighted_kernel_at_origin(origin)
            for origin in SELECTION_ORIGINS
        )
    )
    selection_truth = torch.stack(
        tuple(
            weighted_truth[:, origins.index(origin)]
            for origin in SELECTION_ORIGINS
        )
    )
    selection_errors = tuple(
        _relative_error(prediction, target)
        for prediction, target in zip(
            selection_predictions,
            selection_truth,
            strict=True,
        )
    )
    selection_cosines = tuple(
        _cosine(prediction, target)
        for prediction, target in zip(
            selection_predictions,
            selection_truth,
            strict=True,
        )
    )
    compiled_error = _relative_error(
        selection_predictions,
        selection_truth,
    )
    residual = selection_predictions - selection_truth
    oracle_projection = _source_projection(
        plan.source_basis,
        selection_truth.transpose(0, 1).contiguous(),
    ).transpose(0, 1).contiguous()
    accounting = plan.accounting().metadata()
    runtime = plan.prepare(device="cpu", dtype=torch.float64)
    execution = runtime.execution_accounting(
        logical_positions=sequence_positions,
        valid_mask=valid_mask,
        source_mask=source_mask,
    ).metadata()
    coefficient_fraction = float(
        accounting["coefficient_fraction_of_dense_fit_knots"]
    )
    mac_fraction = (
        int(execution["factorized_linear_macs"])
        / int(execution["dense_control_linear_macs"])
    )
    current_runtime_multiply_count = (
        int(execution["factorized_linear_macs"])
        + int(execution["core_interpolation_multiplies"])
    )
    current_runtime_multiply_fraction = (
        current_runtime_multiply_count
        / int(execution["dense_control_linear_macs"])
    )
    per_origin = tuple(
        {
            "origin": origin,
            "compiled_weighted_relative_error": error,
            "compiled_weighted_cosine": cosine,
        }
        for origin, error, cosine in zip(
            SELECTION_ORIGINS,
            selection_errors,
            selection_cosines,
            strict=True,
        )
    )
    return {
        "plan_key": plan_key,
        "basis_family": basis_family,
        "rank": rank,
        "random_seed": random_seed,
        "plan_artifact_sha256": plan.artifact_sha256,
        "fit_weighted_relative_error": _relative_error(
            fit_predictions,
            fit_truth,
        ),
        "fit_weighted_cosine": _cosine(fit_predictions, fit_truth),
        "selection_compiled_weighted_relative_error": compiled_error,
        "selection_compiled_weighted_cosine": _cosine(
            selection_predictions,
            selection_truth,
        ),
        "selection_compiled_energy_retained": max(
            0.0,
            1.0 - compiled_error**2,
        ),
        "selection_compiled_p90_origin_relative_error": float(
            torch.quantile(
                torch.tensor(selection_errors, dtype=torch.float64),
                0.90,
            )
        ),
        "selection_compiled_worst_origin_relative_error": max(
            selection_errors
        ),
        "selection_compiled_worst_origin_cosine": min(selection_cosines),
        "selection_by_origin": per_origin,
        "selection_oracle_source_projection_relative_error": _relative_error(
            oracle_projection,
            selection_truth,
        ),
        "selection_residual_energy_profile_by_lag": _profile(
            residual,
            dimension=2,
        ),
        "selection_residual_energy_profile_by_target_mode": _profile(
            residual,
            dimension=3,
        ),
        "stored_coefficient_count": int(
            accounting["stored_coefficient_count"]
        ),
        "prepared_float_scalar_count": int(
            accounting["prepared_float_scalar_count"]
        ),
        "coefficient_fraction_of_dense_fit_knots": coefficient_fraction,
        "stored_coefficient_reduction_fraction": 1.0 - coefficient_fraction,
        "sequence_execution_accounting": execution,
        "factorized_linear_mac_fraction_of_dense": mac_fraction,
        "factorized_linear_mac_reduction_fraction": 1.0 - mac_fraction,
        "current_runtime_factorized_plus_uncached_interpolation_multiplies": (
            current_runtime_multiply_count
        ),
        "current_runtime_multiply_fraction_of_dense_kernel_application": (
            current_runtime_multiply_fraction
        ),
        "current_runtime_multiply_reduction_fraction_vs_dense_kernel_"
        "application": 1.0 - current_runtime_multiply_fraction,
        "current_runtime_compute_gate_requires_cached_or_fused_"
        "interpolation": current_runtime_multiply_fraction > 1.0,
        "passes_offline_energy_gate": (
            compiled_error <= _OFFLINE_MAX_RELATIVE_ERROR
        ),
        "passes_offline_p90_gate": (
            float(
                torch.quantile(
                    torch.tensor(selection_errors, dtype=torch.float64),
                    0.90,
                )
            )
            <= _OFFLINE_MAX_P90_RELATIVE_ERROR
        ),
        "passes_resource_gate": (
            1.0 - coefficient_fraction >= _MIN_RESOURCE_REDUCTION
            and 1.0 - current_runtime_multiply_fraction
            >= _MIN_RESOURCE_REDUCTION
        ),
        "matches_or_beats_same_rank_svd": False,
        "passes_controlled_compression_candidate_gate": False,
    }


@dataclass(frozen=True, slots=True)
class Gemma3GraphSpectralRateDistortionCandidate:
    """Strict bundle of executable rate candidates and held-out-origin rows."""

    source_artifact_file_sha256: str
    source_report_file_sha256: str
    source_report_payload_sha256: str
    source_mapping_artifact_sha256: str
    binding: Mapping[str, object]
    model: Mapping[str, object]
    response_binding_sha256: str
    cutoffs: tuple[int, ...]
    target_rank: int
    random_seeds: tuple[int, ...]
    permutation_seed: int
    graph_basis: FitOnlyGraphSourceBasis
    plan_keys: tuple[str, ...]
    plans: tuple[ConditionalSpectralGeneratorPlan, ...]
    rate_rows: tuple[Mapping[str, object], ...]
    conclusions: Mapping[str, object]
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        for field in (
            "source_artifact_file_sha256",
            "source_report_file_sha256",
            "source_report_payload_sha256",
            "source_mapping_artifact_sha256",
            "response_binding_sha256",
        ):
            _sha256(getattr(self, field), label=field)
        if self.schema != _SCHEMA or self.format_version != _FORMAT_VERSION:
            raise ValueError("rate-distortion candidate header drifted")
        if not isinstance(self.graph_basis, FitOnlyGraphSourceBasis):
            raise TypeError("graph_basis must be a FitOnlyGraphSourceBasis")
        self.graph_basis.validate_integrity()
        if (
            self.graph_basis.response_binding_sha256
            != self.response_binding_sha256
        ):
            raise ValueError("graph basis response binding differs")
        for field in ("binding", "model", "conclusions"):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field} must be a mapping")
            object.__setattr__(
                self,
                field,
                json.loads(_canonical_json_bytes(value).decode("ascii")),
            )
        cutoffs = tuple(self.cutoffs)
        if (
            not cutoffs
            or tuple(sorted(set(cutoffs))) != cutoffs
            or any(
                type(rank) is not int
                or not 1 <= rank <= self.graph_basis.source_modes
                for rank in cutoffs
            )
        ):
            raise ValueError("cutoffs are invalid")
        object.__setattr__(self, "cutoffs", cutoffs)
        if type(self.target_rank) is not int or self.target_rank <= 0:
            raise ValueError("target_rank must be positive")
        seeds = tuple(self.random_seeds)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise ValueError("random_seeds are invalid")
        object.__setattr__(self, "random_seeds", seeds)
        if type(self.permutation_seed) is not int or self.permutation_seed < 0:
            raise ValueError("permutation_seed is invalid")
        keys = tuple(self.plan_keys)
        plans = tuple(self.plans)
        if (
            not keys
            or len(keys) != len(plans)
            or len(set(keys)) != len(keys)
            or any(
                not isinstance(key, str) or not key
                for key in keys
            )
            or any(
                not isinstance(plan, ConditionalSpectralGeneratorPlan)
                for plan in plans
            )
        ):
            raise ValueError("plan bundle is invalid")
        object.__setattr__(self, "plan_keys", keys)
        object.__setattr__(self, "plans", plans)
        rows = _canonical_rows(self.rate_rows)
        if len(rows) != len(plans):
            raise ValueError("rate rows do not match plans")
        for key, plan, row in zip(keys, plans, rows, strict=True):
            plan.validate_integrity()
            if (
                row.get("plan_key") != key
                or row.get("plan_artifact_sha256") != plan.artifact_sha256
                or plan.response_binding_sha256
                != self.response_binding_sha256
                or plan.fit_weighted_kernels_sha256
                != self.graph_basis.fit_weighted_kernels_sha256
                or plan.fit_knot_origins != FIT_ORIGINS
                or row.get("basis_family") not in _BASIS_FAMILIES
                or row.get("rank") not in cutoffs
            ):
                raise ValueError("rate row and plan provenance differ")
        expected_controls = {
            (family, seed): basis
            for family, seed, basis in _plan_specs(
                self.graph_basis,
                random_seeds=seeds,
                permutation_seed=self.permutation_seed,
            )
        }
        for plan, row in zip(plans, rows, strict=True):
            family = str(row["basis_family"])
            if family == "svd":
                continue
            seed_value = row["random_seed"]
            seed = int(seed_value) if seed_value is not None else None
            expected_basis = expected_controls[(family, seed)]
            rank = int(row["rank"])
            if not torch.equal(
                plan.source_basis,
                expected_basis[:, :rank],
            ):
                raise ValueError(
                    "compiled source basis differs from its graph/control "
                    "provenance"
                )
        object.__setattr__(self, "rate_rows", rows)
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _sha256(self.artifact_sha256, label="artifact_sha256")
                != computed
            ):
                raise ValueError("rate-distortion artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_artifact_file_sha256": (
                self.source_artifact_file_sha256
            ),
            "source_report_file_sha256": self.source_report_file_sha256,
            "source_report_payload_sha256": (
                self.source_report_payload_sha256
            ),
            "source_mapping_artifact_sha256": (
                self.source_mapping_artifact_sha256
            ),
            "binding": dict(self.binding),
            "model": dict(self.model),
            "response_binding_sha256": self.response_binding_sha256,
            "cutoffs": self.cutoffs,
            "target_rank": self.target_rank,
            "random_seeds": self.random_seeds,
            "permutation_seed": self.permutation_seed,
            "graph_basis_artifact_sha256": self.graph_basis.artifact_sha256,
            "plan_keys": self.plan_keys,
            "plan_artifact_sha256s": tuple(
                plan.artifact_sha256 for plan in self.plans
            ),
            "rate_rows": self.rate_rows,
            "conclusions": dict(self.conclusions),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        self.graph_basis.validate_integrity()
        for plan in self.plans:
            plan.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("rate-distortion artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "graph_basis": self.graph_basis.metadata(),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "graph_basis": self.graph_basis.state_dict(),
            "plans": tuple(plan.state_dict() for plan in self.plans),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "Gemma3GraphSpectralRateDistortionCandidate":
        if not isinstance(raw, Mapping):
            raise TypeError("rate-distortion state must be a mapping")
        expected = {
            "schema",
            "format_version",
            "source_artifact_file_sha256",
            "source_report_file_sha256",
            "source_report_payload_sha256",
            "source_mapping_artifact_sha256",
            "binding",
            "model",
            "response_binding_sha256",
            "cutoffs",
            "target_rank",
            "random_seeds",
            "permutation_seed",
            "graph_basis_artifact_sha256",
            "graph_basis",
            "plan_keys",
            "plan_artifact_sha256s",
            "plans",
            "rate_rows",
            "conclusions",
            "artifact_sha256",
        }
        if set(raw) != expected:
            raise ValueError("rate-distortion state fields differ")
        graph = FitOnlyGraphSourceBasis.from_state_dict(raw["graph_basis"])
        if raw["graph_basis_artifact_sha256"] != graph.artifact_sha256:
            raise ValueError("serialized graph basis hash differs")
        plans_raw = raw["plans"]
        if not isinstance(plans_raw, (tuple, list)):
            raise TypeError("serialized plans must be a sequence")
        plans = tuple(
            ConditionalSpectralGeneratorPlan.from_state_dict(plan)
            for plan in plans_raw
        )
        if tuple(raw["plan_artifact_sha256s"]) != tuple(  # type: ignore[arg-type]
            plan.artifact_sha256 for plan in plans
        ):
            raise ValueError("serialized plan hashes differ")
        return cls(
            source_artifact_file_sha256=raw[
                "source_artifact_file_sha256"
            ],  # type: ignore[arg-type]
            source_report_file_sha256=raw[
                "source_report_file_sha256"
            ],  # type: ignore[arg-type]
            source_report_payload_sha256=raw[
                "source_report_payload_sha256"
            ],  # type: ignore[arg-type]
            source_mapping_artifact_sha256=raw[
                "source_mapping_artifact_sha256"
            ],  # type: ignore[arg-type]
            binding=raw["binding"],  # type: ignore[arg-type]
            model=raw["model"],  # type: ignore[arg-type]
            response_binding_sha256=raw[
                "response_binding_sha256"
            ],  # type: ignore[arg-type]
            cutoffs=tuple(raw["cutoffs"]),  # type: ignore[arg-type]
            target_rank=raw["target_rank"],  # type: ignore[arg-type]
            random_seeds=tuple(raw["random_seeds"]),  # type: ignore[arg-type]
            permutation_seed=raw["permutation_seed"],  # type: ignore[arg-type]
            graph_basis=graph,
            plan_keys=tuple(raw["plan_keys"]),  # type: ignore[arg-type]
            plans=plans,
            rate_rows=tuple(raw["rate_rows"]),  # type: ignore[arg-type]
            conclusions=raw["conclusions"],  # type: ignore[arg-type]
            artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
            schema=raw["schema"],  # type: ignore[arg-type]
            format_version=raw["format_version"],  # type: ignore[arg-type]
        )


def _plan_specs(
    graph: FitOnlyGraphSourceBasis,
    *,
    random_seeds: tuple[int, ...],
    permutation_seed: int,
) -> tuple[tuple[str, int | None, Tensor], ...]:
    modes = graph.source_modes
    permutation_generator = torch.Generator().manual_seed(permutation_seed)
    permutation = torch.randperm(
        modes,
        generator=permutation_generator,
    )
    result: list[tuple[str, int | None, Tensor]] = [
        ("signed_gfa", None, graph.signed_eigenvectors),
        ("magnitude_gfa", None, graph.magnitude_eigenvectors),
        ("native_prefix", None, torch.eye(modes, dtype=torch.float64)),
        (
            "signed_row_permutation",
            permutation_seed,
            graph.signed_eigenvectors[permutation].contiguous(),
        ),
    ]
    result.extend(
        (
            "random_haar",
            seed,
            _random_orthonormal(modes, seed=seed),
        )
        for seed in random_seeds
    )
    return tuple(result)


def _finalize_rows(
    rows: list[dict[str, object]],
    *,
    cutoffs: tuple[int, ...],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rank in cutoffs:
        by_family: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            if row["rank"] == rank:
                by_family.setdefault(
                    str(row["basis_family"]),
                    [],
                ).append(row)
        svd_error = float(
            by_family["svd"][0][
                "selection_compiled_weighted_relative_error"
            ]
        )
        random_errors = sorted(
            float(row["selection_compiled_weighted_relative_error"])
            for row in by_family["random_haar"]
        )
        random_median = float(
            torch.median(torch.tensor(random_errors, dtype=torch.float64))
        )
        signed = by_family["signed_gfa"][0]
        magnitude = by_family["magnitude_gfa"][0]
        native = by_family["native_prefix"][0]
        permutation = by_family["signed_row_permutation"][0]
        for row in (
            candidate
            for candidates in by_family.values()
            for candidate in candidates
        ):
            error = float(
                row["selection_compiled_weighted_relative_error"]
            )
            row["matches_or_beats_same_rank_svd"] = error <= svd_error + 1e-12
        signed_error = float(
            signed["selection_compiled_weighted_relative_error"]
        )
        beats_graph_controls = (
            signed_error
            <= float(
                magnitude["selection_compiled_weighted_relative_error"]
            )
            and signed_error
            <= float(native["selection_compiled_weighted_relative_error"])
            and signed_error <= random_median
            and signed_error
            <= float(
                permutation["selection_compiled_weighted_relative_error"]
            )
        )
        signed["passes_controlled_compression_candidate_gate"] = (
            bool(signed["passes_offline_energy_gate"])
            and bool(signed["passes_offline_p90_gate"])
            and bool(signed["passes_resource_gate"])
            and bool(signed["matches_or_beats_same_rank_svd"])
            and beats_graph_controls
        )
        summaries.append(
            {
                "rank": rank,
                "signed_gfa_selection_relative_error": signed_error,
                "magnitude_gfa_selection_relative_error": float(
                    magnitude[
                        "selection_compiled_weighted_relative_error"
                    ]
                ),
                "svd_selection_relative_error": svd_error,
                "native_prefix_selection_relative_error": float(
                    native[
                        "selection_compiled_weighted_relative_error"
                    ]
                ),
                "signed_row_permutation_selection_relative_error": float(
                    permutation[
                        "selection_compiled_weighted_relative_error"
                    ]
                ),
                "random_haar_selection_relative_error_min": min(
                    random_errors
                ),
                "random_haar_selection_relative_error_median": random_median,
                "random_haar_selection_relative_error_max": max(
                    random_errors
                ),
                "signed_gfa_matches_or_beats_svd": bool(
                    signed["matches_or_beats_same_rank_svd"]
                ),
                "signed_gfa_beats_graph_controls": beats_graph_controls,
                "signed_gfa_passes_controlled_compression_candidate_gate": (
                    bool(
                        signed[
                            "passes_controlled_compression_candidate_gate"
                        ]
                    )
                ),
                "stored_coefficient_count": signed[
                    "stored_coefficient_count"
                ],
                "coefficient_fraction_of_dense_fit_knots": signed[
                    "coefficient_fraction_of_dense_fit_knots"
                ],
                "factorized_linear_macs": signed[
                    "sequence_execution_accounting"
                ]["factorized_linear_macs"],
                "dense_control_linear_macs": signed[
                    "sequence_execution_accounting"
                ]["dense_control_linear_macs"],
                "factorized_linear_mac_fraction_of_dense": signed[
                    "factorized_linear_mac_fraction_of_dense"
                ],
                "current_runtime_factorized_plus_uncached_interpolation_"
                "multiplies": signed[
                    "current_runtime_factorized_plus_uncached_interpolation_"
                    "multiplies"
                ],
                "current_runtime_multiply_fraction_of_dense_kernel_"
                "application": signed[
                    "current_runtime_multiply_fraction_of_dense_kernel_"
                    "application"
                ],
            }
        )
    passing = tuple(
        summary["rank"]
        for summary in summaries
        if summary[
            "signed_gfa_passes_controlled_compression_candidate_gate"
        ]
    )
    conclusions = {
        "by_rank": tuple(summaries),
        "signed_gfa_passing_ranks": passing,
        "signed_gfa_passed_controlled_compression_candidate_gate": bool(
            passing
        ),
        "smallest_passing_signed_gfa_rank": (
            min(passing) if passing else None
        ),
        "heldout_origins_are_open_development_selection_not_confirmation": True,
        "oracle_projection_metrics_are_not_executor_predictions": True,
        "compiled_selection_metrics_use_fit_frozen_cores": True,
        "complete_model_replacement_authority": False,
        "model_compression_claim": False,
        "nll_or_task_fidelity_measured": False,
        "latency_measured": False,
        "analytic_macs_are_not_a_speed_claim": True,
    }
    return tuple(rows), conclusions


def _graph_prefix_diagnostics(
    graph: FitOnlyGraphSourceBasis,
    *,
    cutoffs: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for family, values, energy in (
        (
            "signed_gfa",
            graph.signed_eigenvalues,
            graph.signed_projection_energy,
        ),
        (
            "magnitude_gfa",
            graph.magnitude_eigenvalues,
            graph.magnitude_projection_energy,
        ),
    ):
        for rank in cutoffs:
            right_gap = (
                float(values[rank] - values[rank - 1])
                if rank < graph.source_modes
                else None
            )
            result.append(
                {
                    "basis_family": family,
                    "rank": rank,
                    "cumulative_fit_projection_energy": float(
                        energy[:rank].sum()
                    ),
                    "fit_projection_relative_error": math.sqrt(
                        max(0.0, 1.0 - float(energy[:rank].sum()))
                    ),
                    "eigenvalue_at_prefix_end": float(values[rank - 1]),
                    "eigenvalue_after_prefix": (
                        float(values[rank])
                        if rank < graph.source_modes
                        else None
                    ),
                    "right_eigenvalue_gap": right_gap,
                    "splits_numerically_tied_eigenspace": (
                        right_gap is not None
                        and abs(right_gap)
                        <= 1e-10
                        * max(
                            1.0,
                            abs(float(values[rank - 1])),
                            abs(float(values[rank])),
                        )
                    ),
                }
            )
    return tuple(result)


def compile_gemma3_graph_spectral_rate_distortion_candidate(
    source: Gemma3SpectralSource,
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    target_rank: int = DEFAULT_TARGET_RANK,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
) -> Gemma3GraphSpectralRateDistortionCandidate:
    """Fit all candidates on 8/24/40 and score frozen plans on 16/32."""

    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    origins = source.mapping.impulse_logical_positions
    if origins != INTERIOR_ORIGINS:
        raise ValueError("source origins differ from the frozen split")
    cutoff_tuple = tuple(cutoffs)
    if (
        not cutoff_tuple
        or tuple(sorted(set(cutoff_tuple))) != cutoff_tuple
        or any(
            type(rank) is not int
            or not 1 <= rank <= source.mapping.source_rank
            for rank in cutoff_tuple
        )
    ):
        raise ValueError("cutoffs are invalid")
    seed_tuple = tuple(random_seeds)
    if (
        not seed_tuple
        or len(set(seed_tuple)) != len(seed_tuple)
        or any(type(seed) is not int or seed < 0 for seed in seed_tuple)
    ):
        raise ValueError("random_seeds are invalid")
    if type(target_rank) is not int or not (
        1 <= target_rank <= source.mapping.target_rank
    ):
        raise ValueError("target_rank is invalid")
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    sigma = source.source_mode_standard_deviations
    response_binding = _response_binding(
        source,
        component="local_central_odd_tangent",
    )
    graph = fit_graph_source_bases(
        local.impulse_responses,
        sigma,
        origins,
        FIT_ORIGINS,
        response_binding_sha256=response_binding,
        fft_length=source.mapping.fft_length,
    )
    weighted_truth = (
        local.impulse_responses * sigma.view(-1, 1, 1, 1)
    )
    sequence_positions = torch.tensor(
        tuple(source.protocol["logical_positions"]),  # type: ignore[arg-type]
        dtype=torch.int64,
    )
    valid_mask = torch.tensor(
        tuple(source.protocol["valid_mask"]),  # type: ignore[arg-type]
        dtype=torch.bool,
    )
    source_mask = (
        (sequence_positions >= FIT_ORIGINS[0])
        & (sequence_positions <= FIT_ORIGINS[-1])
        & valid_mask
    )
    plan_keys: list[str] = []
    plans: list[ConditionalSpectralGeneratorPlan] = []
    rows: list[dict[str, object]] = []
    for rank in cutoff_tuple:
        svd_plan = fit_conditional_spectral_generator(
            local.impulse_responses,
            sigma,
            origins,
            FIT_ORIGINS,
            rank,
            target_rank,
            response_binding_sha256=response_binding,
            input_transform="standardized_linear",
            fft_length=source.mapping.fft_length,
        )
        key = f"svd:r{rank}"
        plan_keys.append(key)
        plans.append(svd_plan)
        rows.append(
            _metric_row(
                plan_key=key,
                basis_family="svd",
                rank=rank,
                random_seed=None,
                plan=svd_plan,
                weighted_truth=weighted_truth,
                origins=origins,
                sequence_positions=sequence_positions,
                valid_mask=valid_mask,
                source_mask=source_mask,
            )
        )
    for family, seed, full_basis in _plan_specs(
        graph,
        random_seeds=seed_tuple,
        permutation_seed=permutation_seed,
    ):
        source_kind = (
            "signed_phase_graph_low_frequency"
            if family == "signed_gfa"
            else "phase_blind_magnitude_graph_low_frequency"
            if family == "magnitude_gfa"
            else "fixed_orthonormal_control"
        )
        for rank in cutoff_tuple:
            plan = fit_conditional_spectral_generator_with_source_basis(
                local.impulse_responses,
                sigma,
                origins,
                FIT_ORIGINS,
                full_basis[:, :rank],
                target_rank,
                source_basis_kind=source_kind,
                source_basis_fit_weighted_kernels_sha256=(
                    graph.fit_weighted_kernels_sha256
                ),
                response_binding_sha256=response_binding,
                input_transform="standardized_linear",
                fft_length=source.mapping.fft_length,
            )
            qualifier = f":seed{seed}" if seed is not None else ""
            key = f"{family}{qualifier}:r{rank}"
            plan_keys.append(key)
            plans.append(plan)
            rows.append(
                _metric_row(
                    plan_key=key,
                    basis_family=family,
                    rank=rank,
                    random_seed=seed,
                    plan=plan,
                    weighted_truth=weighted_truth,
                    origins=origins,
                    sequence_positions=sequence_positions,
                    valid_mask=valid_mask,
                    source_mask=source_mask,
                )
            )
    final_rows, conclusions = _finalize_rows(
        rows,
        cutoffs=cutoff_tuple,
    )
    conclusions["graph_prefix_diagnostics"] = _graph_prefix_diagnostics(
        graph,
        cutoffs=cutoff_tuple,
    )
    conclusions["target_rank"] = target_rank
    conclusions["target_axis_is_uncompressed"] = (
        target_rank == source.mapping.target_rank
    )
    conclusions["explicit_full_target_basis_is_counted_in_storage_and_macs"] = (
        target_rank == source.mapping.target_rank
    )
    conclusions[
        "signed_graph_retains_real_phase_alignment_not_quadrature_direction"
    ] = True
    return Gemma3GraphSpectralRateDistortionCandidate(
        source_artifact_file_sha256=source.file_sha256,
        source_report_file_sha256=source.report_file_sha256,
        source_report_payload_sha256=source.report_payload_sha256,
        source_mapping_artifact_sha256=source.mapping.artifact_sha256,
        binding=source.binding,
        model=source.model,
        response_binding_sha256=response_binding,
        cutoffs=cutoff_tuple,
        target_rank=target_rank,
        random_seeds=seed_tuple,
        permutation_seed=permutation_seed,
        graph_basis=graph,
        plan_keys=tuple(plan_keys),
        plans=tuple(plans),
        rate_rows=final_rows,
        conclusions=conclusions,
    )


def _claim_boundaries() -> dict[str, object]:
    return {
        "fit_only_graph_and_plan_compilation": True,
        "heldout_origin_frozen_executor_evaluation": True,
        "phase_blind_svd_native_random_and_permutation_controls": True,
        "fixed_reference_modal_delta_executor": True,
        "complete_gemma_block_replacement": False,
        "prompt_conditioned_reference_provider": False,
        "family_disjoint_prompt_validation": False,
        "nll_kl_or_top1_fidelity": False,
        "downstream_task_accuracy": False,
        "whole_model_parameter_compression": False,
        "runtime_latency_or_speed": False,
        "analytic_storage_and_mac_accounting_only": True,
        "development_only": True,
    }


_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_fit_only_graph_basis_tensors": True,
    "contains_compiled_plan_tensors": True,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _publish_candidate(
    candidate: Gemma3GraphSpectralRateDistortionCandidate,
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    reservation = _reserve_outputs((output, report_path))
    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch(candidate.state_dict(), output)
        tensor_digest = _file_sha256(tensor_stage)
        report: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "candidate": candidate.metadata(),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": tensor_digest,
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
            "scientific_status": _claim_boundaries(),
            "safety": _SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        _canonical_json_bytes(report)
        report_stage = _stage_json(report, report_path)
        reservation.publish((tensor_stage, report_stage))
        return report
    finally:
        reservation.release()
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def compile_gemma3_l3_l4_graph_spectral_rate_distortion(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    source_artifact_sha256: str = DEFAULT_INTERIOR_ARTIFACT_SHA256,
    source_report_sha256: str = DEFAULT_INTERIOR_REPORT_SHA256,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Strict-load the pinned source, compile the ladder, and publish once."""

    if source_artifact_sha256 != DEFAULT_INTERIOR_ARTIFACT_SHA256:
        raise ValueError("source must equal the pinned interior tensor")
    if source_report_sha256 != DEFAULT_INTERIOR_REPORT_SHA256:
        raise ValueError("source must equal the pinned interior report")
    destination = _validate_output_path(output, suffix=".pt")
    if destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite rate-distortion report")
    source = load_gemma3_spectral_source(
        source_artifact_path,
        expected_file_sha256=source_artifact_sha256,
        expected_report_sha256=source_report_sha256,
        expected_origins=INTERIOR_ORIGINS,
    )
    candidate = compile_gemma3_graph_spectral_rate_distortion_candidate(
        source
    )
    return _publish_candidate(candidate, output=destination)


def load_gemma3_graph_spectral_rate_distortion_candidate(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str | None = None,
) -> Gemma3GraphSpectralRateDistortionCandidate:
    """Authenticate and restore one frozen rate-distortion bundle."""

    source = Path(path)
    expected = _sha256(expected_file_sha256, label="expected tensor file")
    actual = _file_sha256(source)
    if actual != expected:
        raise ValueError("rate-distortion tensor file hash differs")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    candidate = Gemma3GraphSpectralRateDistortionCandidate.from_state_dict(raw)
    if expected_report_sha256 is not None:
        with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        claimed = _sha256(
            report.get("report_sha256"),
            label="expected report payload",
        )
        payload = dict(report)
        payload.pop("report_sha256")
        if (
            claimed
            != _sha256(
                expected_report_sha256,
                label="expected report payload",
            )
            or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed
            or report["artifact"]["tensor_file_sha256"] != actual
            or report["candidate"]["artifact_sha256"]
            != candidate.artifact_sha256
        ):
            raise ValueError("rate-distortion report binding differs")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the development-only signed-GFA versus matched-control "
            "Gemma L3-to-L4 rate-distortion ladder."
        )
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_INTERIOR_ARTIFACT,
    )
    parser.add_argument(
        "--source-artifact-sha256",
        default=DEFAULT_INTERIOR_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--source-report-sha256",
        default=DEFAULT_INTERIOR_REPORT_SHA256,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = compile_gemma3_l3_l4_graph_spectral_rate_distortion(
        source_artifact_path=arguments.source_artifact,
        source_artifact_sha256=arguments.source_artifact_sha256,
        source_report_sha256=arguments.source_report_sha256,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
