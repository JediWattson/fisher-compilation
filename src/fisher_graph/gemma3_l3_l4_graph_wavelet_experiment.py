"""Prompt-free graph-wavelet development rung for the Gemma L3->L4 map.

This runner consumes the already-measured five-origin structural response.
All graph construction, atom ordering, QR bases, target decoders, and causal
cores are frozen from origins 8/24/40 before origins 16/32 are opened for
development selection.  It performs no model execution and does not load
prompts, tokens, a tokenizer, or model weights.

The primary construction is deterministic group OMP over normalized atoms of
a signed spectral graph-wavelet Parseval frame.  The selected atoms are
orthonormalized incrementally, giving one nested source basis at each matched
vector-packet budget.  Each basis is lowered through the existing
``ConditionalSpectralGeneratorPlan`` with a full target basis, so held-out
metrics include fit-knot interpolation rather than an oracle projection of
the held-out response.

Only compact graph eigensystems and spectral filter multipliers are
serialized.  Raw response tensors, dense per-filter matrices, and transient
compiled plans are deliberately excluded from the publication.
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
from statistics import median
from types import MappingProxyType

import torch
from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorAccounting,
    ConditionalSpectralGeneratorPlan,
    evaluate_conditional_spectral_generator,
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
    fit_graph_source_bases,
)
from .graph_wavelet_analysis import (
    DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES,
    SpectralGraphWaveletFrame,
    build_spectral_graph_wavelet_frame,
    fit_graph_wavelet_omp_subspace,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256",
    "Gemma3GraphWaveletCandidate",
    "analyze_gemma3_l3_l4_graph_wavelets",
    "build_parser",
    "compile_gemma3_graph_wavelet_candidate",
    "describe_gemma3_l3_l4_graph_wavelets",
    "load_gemma3_graph_wavelet_candidate",
    "main",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-dev-v2.pt"
)
EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256 = (
    "855a047ef20ca3e11a105d7d62752575381ce6eeccd7d12bf98b72dc43067730"
)
PACKET_BUDGETS = (8, 16, 32, 45, 52, 64)
DIFFUSION_SCALES = DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES
EIGENVALUE_TOLERANCE = 1.0e-12
PERMUTATION_SEED = 271828
RANDOM_ORTHONORMAL_SEEDS = (
    1729,
    3253,
    4513,
    6421,
    8191,
    10007,
    12289,
    16381,
)
RANDOM_CONTROL_METHODS = tuple(
    f"seeded_random_orthonormal_fit_energy_seed_{seed}"
    for seed in RANDOM_ORTHONORMAL_SEEDS
)
MAXIMUM_SELECTION_RELATIVE_ERROR = 0.20
MINIMUM_SELECTION_COSINE = 0.98
MINIMUM_COMPILED_PLAN_REDUCTION = 0.20
MINIMUM_LOCALITY_REDUCTION = 0.20
MAXIMUM_LOCALITY_FIDELITY_LOSS = 0.01
MINIMUM_RANDOM_CONTROL_WINS = 6

METHOD_ORDER = (
    "signed_graph_wavelet_omp",
    "magnitude_graph_wavelet_omp",
    "signed_graph_fourier_prefix",
    "signed_graph_fourier_fit_energy",
    "permuted_signed_graph_wavelet_omp",
    *RANDOM_CONTROL_METHODS,
    "fit_svd_prefix",
    "native_mode_omp",
)

_TRANSFORM_METADATA_SEMANTICS = {
    "signed_graph_wavelet_omp": (
        "compact_signed_graph_eigensystem_and_spectral_kernels"
    ),
    "magnitude_graph_wavelet_omp": (
        "compact_magnitude_graph_eigensystem_and_spectral_kernels"
    ),
    "signed_graph_fourier_prefix": "compact_signed_graph_eigensystem",
    "signed_graph_fourier_fit_energy": (
        "compact_signed_graph_eigensystem_and_fit_energy_order"
    ),
    "permuted_signed_graph_wavelet_omp": (
        "compact_signed_graph_eigensystem_spectral_kernels_and_permutation"
    ),
    **{
        method: "seeded_orthonormal_basis_and_fit_energy_order"
        for method in RANDOM_CONTROL_METHODS
    },
    "fit_svd_prefix": "fit_only_full_svd_source_basis",
    "native_mode_omp": "implicit_native_coordinates",
}

_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_wavelet_development"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-graph-wavelet:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-graph-wavelet-report:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l3-l4-graph-wavelet-tensor:v1\0"
_FRAME_TENSOR_DOMAIN = b"fisher-graph:canonical-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_V1_ARTIFACT_SHA256 = (
    "ba2a76556a71471017d8fb3de93b8a23974104517bcead3813afa5e862c5f36c"
)
_LEGACY_V1_TENSOR_FILE_SHA256 = (
    "7df16688a9810905631f1b7afa0120cb07b4c306b80c1b6cd770280aa5902895"
)
_LEGACY_V1_REPORT_SHA256 = (
    "7905cb065a71fd248a4e4992668d405f5bfd8a91288421a94132ff0d74eae8ba"
)

_COMPACT_TENSOR_FIELDS = (
    "signed_eigenvalues",
    "signed_eigenvectors",
    "signed_spectral_kernels",
    "magnitude_eigenvalues",
    "magnitude_eigenvectors",
    "magnitude_spectral_kernels",
)

_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_raw_response_tensors": False,
    "contains_dense_per_scale_operators": False,
    "contains_compiled_plan_tensors": False,
    "contains_compact_graph_eigensystems": True,
    "artifact_must_remain_outside_git": True,
}


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256_with_domain(value: Tensor, *, domain: bytes) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(tuple(int(v) for v in tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    return _tensor_sha256_with_domain(value, domain=_TENSOR_DOMAIN)


def _frame_tensor_sha256(value: Tensor) -> str:
    return _tensor_sha256_with_domain(value, domain=_FRAME_TENSOR_DOMAIN)


def _floating_tensor_geometry(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    """Validate tensor metadata without inspecting or copying its values."""

    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    if (
        value.ndim != ndim
        or any(int(width) <= 0 for width in value.shape)
    ):
        raise ValueError(f"{label} must be nonempty rank-{ndim} data")
    return value


def _float_tensor(value: object, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if (
        result.ndim != ndim
        or any(int(width) <= 0 for width in result.shape)
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError(f"{label} must be finite nonempty rank-{ndim} data")
    return result


def _canonical_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be a string-keyed mapping")
    normalized = json.loads(
        _canonical_json_bytes(dict(value)).decode("ascii")
    )
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must normalize to an object")
    return MappingProxyType(normalized)


def _canonical_rows(
    value: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("rate rows must be a sequence")
    return tuple(
        _canonical_mapping(row, label="rate row")
        for row in value
    )


def _relative_error(reference: Tensor, candidate: Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(reference)),
        torch.finfo(torch.float64).tiny,
    )
    return float(torch.linalg.vector_norm(candidate - reference)) / denominator


def _cosine(left: Tensor, right: Tensor) -> float:
    first = left.reshape(-1)
    second = right.reshape(-1)
    denominator = float(torch.linalg.vector_norm(first)) * float(
        torch.linalg.vector_norm(second)
    )
    if denominator <= torch.finfo(torch.float64).tiny:
        return 1.0 if bool(torch.equal(first, second)) else 0.0
    return max(-1.0, min(1.0, float(torch.dot(first, second)) / denominator))


def _laplacian(eigenvalues: Tensor, eigenvectors: Tensor) -> Tensor:
    result = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T
    return ((result + result.T) / 2.0).contiguous()


def _frame_dictionary(
    frame: SpectralGraphWaveletFrame,
) -> tuple[Tensor, tuple[int, ...]]:
    """Return normalized frame atoms as columns and their flattened indices."""

    frame.validate_integrity()
    atoms = (
        frame.filter_matrices.permute(0, 2, 1)
        .reshape(frame.frame_coefficient_count, frame.node_count)
        .T.contiguous()
    )
    norms = torch.linalg.vector_norm(atoms, dim=0)
    floor = max(
        torch.finfo(torch.float64).eps,
        float(norms.max()) * math.sqrt(torch.finfo(torch.float64).eps),
    )
    active = tuple(
        index for index in range(atoms.shape[1]) if float(norms[index]) > floor
    )
    if len(active) < frame.node_count:
        raise RuntimeError("graph-wavelet frame has too few supported atoms")
    ordinals = torch.tensor(active, dtype=torch.int64)
    selected = atoms.index_select(1, ordinals)
    selected /= norms.index_select(0, ordinals).unsqueeze(0)
    return selected.contiguous(), active


def _canonicalize_columns(value: Tensor) -> Tensor:
    result = value.clone()
    for column in range(result.shape[1]):
        pivot = int(torch.argmax(result[:, column].abs()))
        if float(result[pivot, column]) < 0.0:
            result[:, column] *= -1.0
    return result.contiguous()


def _random_orthonormal(node_count: int, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        (node_count, node_count),
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(raw)
    return _canonicalize_columns(basis)


@dataclass(frozen=True, slots=True)
class _NestedBasisFamily:
    method: str
    available_packet_count: int
    selected_packet_order: tuple[int, ...]
    bases: Mapping[int, Tensor]
    source_basis_kind: str
    transform_metadata_semantics: str
    transform_metadata_float64_scalars: int
    transform_metadata_integer_scalars: int = 0


def _group_omp_nested_bases(
    dictionary: Tensor,
    packet_indices: Sequence[int],
    fit_target: Tensor,
    budgets: Sequence[int],
    *,
    method: str,
    source_basis_kind: str = "fixed_orthonormal_control",
    transform_metadata_float64_scalars: int,
    transform_metadata_integer_scalars: int = 0,
) -> _NestedBasisFamily:
    """Fit one deterministic nested group-OMP order and canonical QR bases."""

    atoms = _float_tensor(dictionary, label=f"{method} dictionary", ndim=2)
    target = _float_tensor(fit_target, label="fit target", ndim=4)
    if atoms.shape[0] != target.shape[0]:
        raise ValueError("dictionary and fit target source widths differ")
    packets = tuple(packet_indices)
    if (
        len(packets) != atoms.shape[1]
        or len(set(packets)) != len(packets)
        or any(type(value) is not int or value < 0 for value in packets)
    ):
        raise ValueError("packet indices do not identify dictionary columns")
    selected_budgets = tuple(budgets)
    if (
        not selected_budgets
        or tuple(sorted(set(selected_budgets))) != selected_budgets
        or selected_budgets[-1] > atoms.shape[0]
    ):
        raise ValueError("OMP budgets are invalid")
    norms = torch.linalg.vector_norm(atoms, dim=0)
    if bool((norms <= torch.finfo(torch.float64).eps).any()):
        raise ValueError("OMP dictionary contains a zero atom")
    atoms = (atoms / norms.unsqueeze(0)).contiguous()
    flattened = target.reshape(target.shape[0], -1)
    residual = flattened.clone()
    remaining = set(range(atoms.shape[1]))
    q_columns: list[Tensor] = []
    order: list[int] = []
    bases: dict[int, Tensor] = {}
    independence_floor = (
        256.0 * torch.finfo(torch.float64).eps * max(1, atoms.shape[0])
    )
    while len(q_columns) < selected_budgets[-1]:
        correlations = atoms.T @ residual
        scores = torch.linalg.vector_norm(correlations, dim=1)
        candidates = sorted(
            remaining,
            key=lambda index: (-float(scores[index]), packets[index]),
        )
        chosen: int | None = None
        chosen_column: Tensor | None = None
        for candidate in candidates:
            column = atoms[:, candidate].clone()
            for existing in q_columns:
                column -= torch.dot(existing, column) * existing
            norm = float(torch.linalg.vector_norm(column))
            if norm > independence_floor:
                chosen = candidate
                chosen_column = column / norm
                break
        if chosen is None or chosen_column is None:
            raise RuntimeError("group OMP could not reach the requested rank")
        if float(torch.dot(chosen_column, atoms[:, chosen])) < 0.0:
            chosen_column *= -1.0
        q_columns.append(chosen_column.contiguous())
        order.append(packets[chosen])
        remaining.remove(chosen)
        q = torch.stack(q_columns, dim=1).contiguous()
        residual = flattened - q @ (q.T @ flattened)
        if len(q_columns) in selected_budgets:
            bases[len(q_columns)] = q.clone()
    return _NestedBasisFamily(
        method=method,
        available_packet_count=len(packets),
        selected_packet_order=tuple(order),
        bases=MappingProxyType(bases),
        source_basis_kind=source_basis_kind,
        transform_metadata_semantics=_TRANSFORM_METADATA_SEMANTICS[method],
        transform_metadata_float64_scalars=(
            transform_metadata_float64_scalars
        ),
        transform_metadata_integer_scalars=(
            transform_metadata_integer_scalars
        ),
    )


def _prefix_family(
    basis: Tensor,
    budgets: Sequence[int],
    *,
    method: str,
    source_basis_kind: str,
    transform_metadata_float64_scalars: int,
    transform_metadata_integer_scalars: int = 0,
) -> _NestedBasisFamily:
    canonical = _float_tensor(basis, label=f"{method} basis", ndim=2)
    if canonical.shape[1] > canonical.shape[0]:
        raise ValueError("prefix basis cannot be wider than its source axis")
    identity = torch.eye(canonical.shape[1], dtype=torch.float64)
    if not torch.allclose(
        canonical.T @ canonical,
        identity,
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise ValueError("prefix basis must be orthonormal")
    selected_budgets = tuple(budgets)
    if selected_budgets[-1] > canonical.shape[1]:
        raise ValueError("prefix budgets exceed the supplied basis")
    return _NestedBasisFamily(
        method=method,
        available_packet_count=canonical.shape[1],
        selected_packet_order=tuple(range(canonical.shape[1])),
        bases=MappingProxyType(
            {
                budget: canonical[:, :budget].clone()
                for budget in selected_budgets
            }
        ),
        source_basis_kind=source_basis_kind,
        transform_metadata_semantics=_TRANSFORM_METADATA_SEMANTICS[method],
        transform_metadata_float64_scalars=(
            transform_metadata_float64_scalars
        ),
        transform_metadata_integer_scalars=(
            transform_metadata_integer_scalars
        ),
    )


def _fit_energy_ordered_basis(
    basis: Tensor,
    fit_target: Tensor,
) -> tuple[Tensor, tuple[int, ...]]:
    """Order one orthonormal basis by fit-only grouped coefficient energy."""

    canonical = _float_tensor(basis, label="fit-energy basis", ndim=2)
    target = _float_tensor(fit_target, label="fit-energy target", ndim=4)
    if canonical.shape[0] != target.shape[0]:
        raise ValueError("fit-energy basis and target source widths differ")
    coefficients = canonical.T @ target.reshape(target.shape[0], -1)
    energy = coefficients.square().sum(dim=1)
    order = tuple(
        sorted(
            range(canonical.shape[1]),
            key=lambda index: (-float(energy[index]), index),
        )
    )
    ordinals = torch.tensor(order, dtype=torch.int64)
    return canonical.index_select(1, ordinals).contiguous(), order


def _basis_locality(basis: Tensor, laplacian: Tensor) -> dict[str, object]:
    probability = basis.square()
    effective = 1.0 / probability.square().sum(dim=0)
    peak = probability.max(dim=0).values
    variation = torch.diagonal(basis.T @ laplacian @ basis)
    return {
        "column_count": int(basis.shape[1]),
        "mean_effective_node_support": float(effective.mean()),
        "p90_effective_node_support": float(
            torch.quantile(effective, 0.90)
        ),
        "p95_effective_node_support": float(
            torch.quantile(effective, 0.95)
        ),
        "maximum_effective_node_support": float(effective.max()),
        "mean_peak_node_energy_fraction": float(peak.mean()),
        "p10_peak_node_energy_fraction": float(torch.quantile(peak, 0.10)),
        "p05_peak_node_energy_fraction": float(torch.quantile(peak, 0.05)),
        "minimum_peak_node_energy_fraction": float(peak.min()),
        "mean_signed_graph_quadratic_form": float(variation.mean()),
        "maximum_signed_graph_quadratic_form": float(variation.max()),
        "hop_locality_claim": False,
    }


def _selected_atom_locality(
    dictionary: Tensor,
    packet_indices: Sequence[int],
    selected_order: Sequence[int],
    budget: int,
) -> dict[str, object]:
    lookup = {packet: ordinal for ordinal, packet in enumerate(packet_indices)}
    ordinals = torch.tensor(
        [lookup[packet] for packet in selected_order[:budget]],
        dtype=torch.int64,
    )
    atoms = dictionary.index_select(1, ordinals)
    atoms /= torch.linalg.vector_norm(atoms, dim=0).unsqueeze(0)
    probability = atoms.square()
    effective = 1.0 / probability.square().sum(dim=0)
    return {
        "selected_atom_count": budget,
        "mean_effective_node_support": float(effective.mean()),
        "p90_effective_node_support": float(
            torch.quantile(effective, 0.90)
        ),
        "p95_effective_node_support": float(
            torch.quantile(effective, 0.95)
        ),
        "maximum_effective_node_support": float(effective.max()),
        "mean_peak_node_energy_fraction": float(
            probability.max(dim=0).values.mean()
        ),
        "hop_locality_claim": False,
    }


def _frame_exactness(
    frame: SpectralGraphWaveletFrame,
    signal: Tensor,
) -> dict[str, object]:
    coefficients = frame.analyze(signal)
    reconstruction = frame.synthesize(coefficients)
    signal_energy = float(signal.square().sum())
    coefficient_energy = float(coefficients.values.square().sum())
    return {
        "frame_artifact_sha256": frame.artifact_sha256,
        "frame_coefficient_count": frame.frame_coefficient_count,
        "full_frame_reconstruction_relative_error": _relative_error(
            signal,
            reconstruction,
        ),
        "full_frame_reconstruction_cosine": _cosine(
            signal,
            reconstruction,
        ),
        "parseval_energy_relative_error": (
            abs(coefficient_energy - signal_energy)
            / max(signal_energy, torch.finfo(torch.float64).tiny)
        ),
        "tight_partition_maximum_error": (
            frame.tight_partition_maximum_error
        ),
        "tight_operator_maximum_error": frame.tight_operator_maximum_error,
        "dense_filter_matrices_serialized": False,
    }


def _evaluation_metadata(value: object) -> dict[str, object]:
    method = getattr(value, "metadata", None)
    if not callable(method):
        raise TypeError("conditional spectral evaluation lacks metadata")
    result = method()
    if not isinstance(result, Mapping):
        raise TypeError("conditional spectral evaluation metadata is invalid")
    return dict(result)


def _plan_row_prefix(
    *,
    family: _NestedBasisFamily,
    budget: int,
    basis: Tensor,
    plan: ConditionalSpectralGeneratorPlan,
    fit_evaluation: object,
    selected_atom_locality: Mapping[str, object] | None,
    signed_laplacian: Tensor,
    dense_full_rank_plan_scalars: int,
) -> dict[str, object]:
    core_scalars = int(plan.knot_cores.numel())
    source_basis_scalars = int(plan.source_basis.numel())
    target_basis_scalars = int(plan.target_basis.numel())
    plan_scalars = plan.stored_coefficient_count
    return {
        "method": family.method,
        "vector_packet_budget": budget,
        "available_packet_count": family.available_packet_count,
        "selected_packet_indices": family.selected_packet_order[:budget],
        "selected_packet_order_sha256": _json_sha256(
            family.selected_packet_order[:budget],
            domain=_TENSOR_DOMAIN,
        ),
        "source_basis_sha256": _tensor_sha256(basis),
        "source_basis_kind": family.source_basis_kind,
        "source_rank": plan.source_rank,
        "target_rank": plan.target_rank,
        "plan_artifact_sha256": plan.artifact_sha256,
        "fit_weighted_kernels_sha256": plan.fit_weighted_kernels_sha256,
        "fit_evaluation": _evaluation_metadata(fit_evaluation),
        "heldout_evaluation": None,
        "q_locality": _basis_locality(basis, signed_laplacian),
        "selected_atom_locality": (
            dict(selected_atom_locality)
            if selected_atom_locality is not None
            else None
        ),
        "coefficient_payload": {
            "fit_knot_core_float64_scalars": core_scalars,
            "source_basis_float64_scalars": source_basis_scalars,
            "target_basis_float64_scalars": target_basis_scalars,
            "compiled_plan_float64_scalars": plan_scalars,
            "compiler_transform_metadata_float64_scalars": (
                family.transform_metadata_float64_scalars
            ),
            "compiler_transform_metadata_integer_scalars": (
                family.transform_metadata_integer_scalars
            ),
            "compiler_transform_metadata_semantics": (
                family.transform_metadata_semantics
            ),
            "standalone_compiler_plus_plan_float64_scalars": (
                plan_scalars + family.transform_metadata_float64_scalars
            ),
            "compiled_plan_fraction_of_full_rank": (
                plan_scalars / dense_full_rank_plan_scalars
            ),
        },
        "plan_accounting": plan.accounting().metadata(),
    }


def _exact_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _exact_int_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an integer sequence")
    result = tuple(value)
    if any(type(item) is not int for item in result):
        raise ValueError(f"{label} must be an integer sequence")
    return result  # type: ignore[return-value]


def _expected_transform_metadata_counts(
    method: str,
    *,
    source_modes: int,
    maximum_budget: int,
    wavelet_frame_coefficient_counts: Mapping[str, int],
) -> tuple[int, int]:
    if method in wavelet_frame_coefficient_counts:
        integer_count = (
            source_modes
            if method == "permuted_signed_graph_wavelet_omp"
            else 0
        )
        return (
            source_modes * source_modes
            + source_modes
            + wavelet_frame_coefficient_counts[method],
            integer_count,
        )
    if method == "signed_graph_fourier_prefix":
        return source_modes * source_modes + source_modes, 0
    if method == "signed_graph_fourier_fit_energy":
        return source_modes * source_modes + source_modes, source_modes
    if method in RANDOM_CONTROL_METHODS:
        return source_modes * source_modes, source_modes
    if method == "fit_svd_prefix":
        return source_modes * maximum_budget, 0
    if method == "native_mode_omp":
        return 0, 0
    raise RuntimeError("graph-wavelet transform accounting method drifted")


def _validate_compact_frame_receipts(
    frame_receipts: Mapping[str, object],
    compact_tensors: Mapping[str, Tensor],
) -> None:
    for receipt_name, field_prefix in (
        ("signed", "signed"),
        ("magnitude", "magnitude"),
    ):
        receipt = frame_receipts.get(receipt_name)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"{receipt_name} frame receipt is missing")
        hashes = receipt.get("tensor_sha256s")
        if not isinstance(hashes, Mapping):
            raise ValueError(
                f"{receipt_name} frame tensor receipts are missing"
            )
        for tensor_name in (
            "eigenvalues",
            "eigenvectors",
            "spectral_kernels",
        ):
            field = f"{field_prefix}_{tensor_name}"
            if hashes.get(tensor_name) != _frame_tensor_sha256(
                compact_tensors[field]
            ):
                raise ValueError(
                    f"compact {field_prefix} tensors differ from frame receipt"
                )


def _validate_rate_row_geometry(
    *,
    rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
    frame_receipts: Mapping[str, object],
    conclusions: Mapping[str, object],
    artifact_sha256: object,
) -> None:
    source_modes = _exact_positive_int(
        protocol.get("source_modes"),
        label="protocol source_modes",
    )
    target_modes = _exact_positive_int(
        protocol.get("target_modes"),
        label="protocol target_modes",
    )
    lag_count = _exact_positive_int(
        protocol.get("lag_count"),
        label="protocol lag_count",
    )
    budgets = _exact_int_tuple(
        protocol.get("packet_budgets"),
        label="protocol packet_budgets",
    )
    if (
        not budgets
        or budgets != tuple(sorted(set(budgets)))
        or budgets[-1] > source_modes
        or any(budget <= 0 for budget in budgets)
    ):
        raise ValueError("protocol packet budgets are invalid")
    method_order = tuple(protocol.get("method_order", ()))
    if method_order != METHOD_ORDER:
        raise ValueError("protocol method order differs")
    measured_origins = _exact_int_tuple(
        protocol.get("measured_origins"),
        label="protocol measured_origins",
    )
    fit_origins = _exact_int_tuple(
        protocol.get("fit_origins"),
        label="protocol fit_origins",
    )
    selection_origins = _exact_int_tuple(
        protocol.get("selection_origins"),
        label="protocol selection_origins",
    )
    split_is_disjoint = set(fit_origins).isdisjoint(selection_origins)
    split_exhausts_measurement = (
        set(fit_origins) | set(selection_origins) == set(measured_origins)
        and len(fit_origins) + len(selection_origins) == len(measured_origins)
    )
    if (
        measured_origins != INTERIOR_ORIGINS
        or fit_origins != FIT_ORIGINS
        or selection_origins != SELECTION_ORIGINS
        or not split_is_disjoint
        or not split_exhausts_measurement
    ):
        raise ValueError("protocol frozen origin split differs")
    if (
        conclusions.get("fit_and_selection_are_disjoint")
        is not split_is_disjoint
    ):
        raise ValueError("origin split conclusion differs")

    wavelet_receipt_names = {
        "signed_graph_wavelet_omp": "signed",
        "magnitude_graph_wavelet_omp": "magnitude",
        "permuted_signed_graph_wavelet_omp": "permuted_signed_control",
    }
    wavelet_frame_coefficient_counts: dict[str, int] = {}
    for method, receipt_name in wavelet_receipt_names.items():
        receipt = frame_receipts.get(receipt_name)
        if not isinstance(receipt, Mapping):
            raise ValueError("wavelet frame receipt is missing")
        exactness = receipt.get("exactness")
        if not isinstance(exactness, Mapping):
            raise ValueError("wavelet frame exactness receipt is missing")
        frame_count = _exact_positive_int(
            receipt.get("frame_coefficient_count"),
            label="wavelet frame coefficient count",
        )
        if (
            receipt.get("node_count") != source_modes
            or exactness.get("frame_coefficient_count") != frame_count
        ):
            raise ValueError("wavelet frame accounting differs")
        wavelet_frame_coefficient_counts[method] = frame_count

    expected_sequence = tuple(
        (method, budget)
        for method in METHOD_ORDER
        for budget in budgets
    )
    actual_sequence = tuple(
        (
            row.get("method"),
            row.get("vector_packet_budget"),
        )
        for row in rows
    )
    if actual_sequence != expected_sequence:
        raise ValueError("rate row method and budget sequence differs")

    full_rank_plan_scalars = (
        source_modes * source_modes
        + target_modes * target_modes
        + len(fit_origins)
        * lag_count
        * source_modes
        * target_modes
    )
    transform_metadata_schema: str | None = None
    for method in METHOD_ORDER:
        previous_prefix: tuple[int, ...] = ()
        available_packet_count: int | None = None
        for budget in budgets:
            row = rows[expected_sequence.index((method, budget))]
            source_rank = _exact_positive_int(
                row.get("source_rank"),
                label="rate row source_rank",
            )
            target_rank = _exact_positive_int(
                row.get("target_rank"),
                label="rate row target_rank",
            )
            if source_rank != budget or target_rank != target_modes:
                raise ValueError("rate row rank geometry differs")
            available = _exact_positive_int(
                row.get("available_packet_count"),
                label="rate row available_packet_count",
            )
            if available_packet_count is None:
                available_packet_count = available
            if available != available_packet_count or available < budget:
                raise ValueError("rate row packet capacity differs")
            selected = _exact_int_tuple(
                row.get("selected_packet_indices"),
                label="rate row selected_packet_indices",
            )
            if (
                len(selected) != budget
                or len(set(selected)) != budget
                or any(packet < 0 or packet >= available for packet in selected)
                or selected[: len(previous_prefix)] != previous_prefix
            ):
                raise ValueError("rate row selected packet prefix differs")
            if row.get("selected_packet_order_sha256") != _json_sha256(
                selected,
                domain=_TENSOR_DOMAIN,
            ):
                raise ValueError("rate row selected packet hash differs")
            previous_prefix = selected

            accounting = row.get("plan_accounting")
            payload = row.get("coefficient_payload")
            if not isinstance(accounting, Mapping) or not isinstance(
                payload,
                Mapping,
            ):
                raise ValueError("rate row accounting is missing")
            expected_source_scalars = source_modes * budget
            expected_target_scalars = target_modes * target_rank
            expected_core_scalars = (
                len(fit_origins)
                * lag_count
                * budget
                * target_rank
            )
            expected_plan_scalars = (
                expected_source_scalars
                + expected_target_scalars
                + expected_core_scalars
            )
            expected_accounting = ConditionalSpectralGeneratorAccounting(
                source_modes=source_modes,
                target_modes=target_modes,
                lag_count=lag_count,
                knot_count=len(fit_origins),
                source_rank=budget,
                target_rank=target_rank,
                source_spectrum_rank=source_modes,
                target_spectrum_rank=target_modes,
            ).metadata()
            if _canonical_json_bytes(dict(accounting)) != _canonical_json_bytes(
                expected_accounting
            ):
                raise ValueError("rate row plan accounting geometry differs")
            payload_geometry = {
                "fit_knot_core_float64_scalars": expected_core_scalars,
                "source_basis_float64_scalars": expected_source_scalars,
                "target_basis_float64_scalars": expected_target_scalars,
                "compiled_plan_float64_scalars": expected_plan_scalars,
            }
            if any(
                payload.get(field) != expected
                for field, expected in payload_geometry.items()
            ):
                raise ValueError(
                    "rate row coefficient accounting geometry differs"
                )
            transform_floats = payload.get(
                "compiler_transform_metadata_float64_scalars"
            )
            transform_integers = payload.get(
                "compiler_transform_metadata_integer_scalars"
            )
            expected_transform_floats, expected_transform_integers = (
                _expected_transform_metadata_counts(
                    method,
                    source_modes=source_modes,
                    maximum_budget=budgets[-1],
                    wavelet_frame_coefficient_counts=(
                        wavelet_frame_coefficient_counts
                    ),
                )
            )
            if (
                transform_floats != expected_transform_floats
                or type(transform_floats) is not int
                or transform_integers != expected_transform_integers
                or type(transform_integers) is not int
                or payload.get("standalone_compiler_plus_plan_float64_scalars")
                != expected_plan_scalars + expected_transform_floats
            ):
                raise ValueError("rate row transform accounting differs")
            has_current_semantics = (
                "compiler_transform_metadata_semantics" in payload
            )
            has_legacy_semantics = (
                "eigensystem_and_filter_metadata_counted" in payload
            )
            if has_current_semantics == has_legacy_semantics:
                raise ValueError(
                    "rate row transform metadata schema differs"
                )
            if has_current_semantics:
                if (
                    payload.get("compiler_transform_metadata_semantics")
                    != _TRANSFORM_METADATA_SEMANTICS[method]
                ):
                    raise ValueError(
                        "rate row transform metadata semantics differ"
                    )
                row_schema = "current"
            # Read-only compatibility for the authenticated v1 publication;
            # newly compiled candidates never emit this blanket legacy field.
            else:
                if (
                    payload.get("eigensystem_and_filter_metadata_counted")
                    is not True
                ):
                    raise ValueError(
                        "rate row transform metadata semantics are missing"
                    )
                row_schema = "legacy"
            if transform_metadata_schema is None:
                transform_metadata_schema = row_schema
            elif transform_metadata_schema != row_schema:
                raise ValueError(
                    "rate row transform metadata schema is mixed"
                )
            fraction = payload.get(
                "compiled_plan_fraction_of_full_rank"
            )
            if (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not math.isclose(
                    float(fraction),
                    expected_plan_scalars / full_rank_plan_scalars,
                    rel_tol=1.0e-15,
                    abs_tol=1.0e-15,
                )
            ):
                raise ValueError("rate row plan fraction geometry differs")
        wavelet_receipt_name = wavelet_receipt_names.get(method)
        if wavelet_receipt_name is not None:
            receipt = frame_receipts.get(wavelet_receipt_name)
            if not isinstance(receipt, Mapping):
                raise ValueError("wavelet frame receipt is missing")
            subspace = receipt.get("fit_only_omp_subspace")
            exactness = receipt.get("exactness")
            if not isinstance(subspace, Mapping) or not isinstance(
                exactness,
                Mapping,
            ):
                raise ValueError("wavelet fit receipt is missing")
            frozen_order = _exact_int_tuple(
                subspace.get("selected_flat_atom_indices"),
                label="wavelet selected packet order",
            )
            if (
                previous_prefix != frozen_order[: budgets[-1]]
                or subspace.get("max_rank") != budgets[-1]
                or subspace.get("node_count") != source_modes
                or exactness.get("frame_coefficient_count")
                != available_packet_count
            ):
                raise ValueError(
                    "rate row selected packets differ from wavelet fit receipt"
                )
    if (
        transform_metadata_schema == "legacy"
        and artifact_sha256 != _LEGACY_V1_ARTIFACT_SHA256
    ):
        raise ValueError(
            "legacy transform metadata candidate identity differs"
        )


@dataclass(frozen=True, slots=True, eq=False)
class Gemma3GraphWaveletCandidate:
    """Compact authenticated report bundle; no response or plan tensors."""

    source_receipt: Mapping[str, object]
    protocol: Mapping[str, object]
    graph_receipt: Mapping[str, object]
    frame_receipts: Mapping[str, object]
    rate_rows: tuple[Mapping[str, object], ...]
    conclusions: Mapping[str, object]
    resource_accounting: Mapping[str, object]
    signed_eigenvalues: Tensor
    signed_eigenvectors: Tensor
    signed_spectral_kernels: Tensor
    magnitude_eigenvalues: Tensor
    magnitude_eigenvectors: Tensor
    magnitude_spectral_kernels: Tensor
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.format_version != _FORMAT_VERSION:
            raise ValueError("graph-wavelet candidate header differs")
        for field in (
            "source_receipt",
            "protocol",
            "graph_receipt",
            "frame_receipts",
            "conclusions",
            "resource_accounting",
        ):
            object.__setattr__(
                self,
                field,
                _canonical_mapping(getattr(self, field), label=field),
            )
        object.__setattr__(self, "rate_rows", _canonical_rows(self.rate_rows))
        for field in _COMPACT_TENSOR_FIELDS:
            object.__setattr__(
                self,
                field,
                _float_tensor(
                    getattr(self, field),
                    label=field,
                    ndim=1 if field.endswith("eigenvalues") else 2,
                ),
            )
        nodes = int(self.signed_eigenvalues.numel())
        if (
            self.signed_eigenvectors.shape != (nodes, nodes)
            or self.magnitude_eigenvalues.shape != (nodes,)
            or self.magnitude_eigenvectors.shape != (nodes, nodes)
            or self.signed_spectral_kernels.shape[1] != nodes
            or self.magnitude_spectral_kernels.shape
            != self.signed_spectral_kernels.shape
            or self.protocol.get("source_modes") != nodes
        ):
            raise ValueError("graph-wavelet candidate tensor geometry differs")
        _validate_compact_frame_receipts(
            self.frame_receipts,
            {
                field: getattr(self, field)
                for field in _COMPACT_TENSOR_FIELDS
            },
        )
        _validate_rate_row_geometry(
            rows=self.rate_rows,
            protocol=self.protocol,
            frame_receipts=self.frame_receipts,
            conclusions=self.conclusions,
            artifact_sha256=self.artifact_sha256,
        )
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("graph-wavelet candidate hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_receipt": dict(self.source_receipt),
            "protocol": dict(self.protocol),
            "graph_receipt": dict(self.graph_receipt),
            "frame_receipts": dict(self.frame_receipts),
            "rate_rows": tuple(dict(row) for row in self.rate_rows),
            "conclusions": dict(self.conclusions),
            "resource_accounting": dict(self.resource_accounting),
            "compact_tensor_sha256s": {
                field: _tensor_sha256(getattr(self, field))
                for field in _COMPACT_TENSOR_FIELDS
            },
            "compact_tensor_shapes": {
                field: tuple(int(v) for v in getattr(self, field).shape)
                for field in _COMPACT_TENSOR_FIELDS
            },
            "safety": _SAFETY,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("graph-wavelet candidate hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            **{
                field: getattr(self, field).clone()
                for field in _COMPACT_TENSOR_FIELDS
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        value: object,
    ) -> "Gemma3GraphWaveletCandidate":
        if not isinstance(value, Mapping):
            raise TypeError("graph-wavelet candidate state must be a mapping")
        expected = {
            "schema",
            "format_version",
            "source_receipt",
            "protocol",
            "graph_receipt",
            "frame_receipts",
            "rate_rows",
            "conclusions",
            "resource_accounting",
            "compact_tensor_sha256s",
            "compact_tensor_shapes",
            "safety",
            "artifact_sha256",
            *_COMPACT_TENSOR_FIELDS,
        }
        if set(value) != expected or value["safety"] != _SAFETY:
            raise ValueError("graph-wavelet candidate state fields differ")
        result = cls(
            source_receipt=value["source_receipt"],  # type: ignore[arg-type]
            protocol=value["protocol"],  # type: ignore[arg-type]
            graph_receipt=value["graph_receipt"],  # type: ignore[arg-type]
            frame_receipts=value["frame_receipts"],  # type: ignore[arg-type]
            rate_rows=tuple(value["rate_rows"]),  # type: ignore[arg-type]
            conclusions=value["conclusions"],  # type: ignore[arg-type]
            resource_accounting=value[
                "resource_accounting"
            ],  # type: ignore[arg-type]
            **{
                field: value[field]
                for field in _COMPACT_TENSOR_FIELDS
            },
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )
        metadata = result.metadata()
        if (
            value["compact_tensor_sha256s"]
            != metadata["compact_tensor_sha256s"]
            or value["compact_tensor_shapes"]
            != metadata["compact_tensor_shapes"]
        ):
            raise ValueError("graph-wavelet compact tensor receipt differs")
        return result


def _compile_from_response(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    *,
    response_binding_sha256: str,
    expected_graph_basis_artifact_sha256: str,
    source_receipt: Mapping[str, object],
    fft_length: int,
    packet_budgets: Sequence[int] = PACKET_BUDGETS,
    target_rank: int | None = None,
    source_tensor_file_bytes: int = 0,
) -> Gemma3GraphWaveletCandidate:
    """Internal test seam and implementation of the frozen offline analysis."""

    response_geometry = _floating_tensor_geometry(
        responses,
        label="responses",
        ndim=4,
    )
    scales = _float_tensor(source_scales, label="source_scales", ndim=1)
    measured_origins = tuple(origins)
    budgets = tuple(packet_budgets)
    if measured_origins != INTERIOR_ORIGINS:
        raise ValueError("graph-wavelet source must use the frozen five origins")
    if len(measured_origins) != response_geometry.shape[1]:
        raise ValueError("origins must match the response origin axis")
    if response_geometry.shape[0] != scales.numel():
        raise ValueError("response and source scale widths differ")
    source_modes = int(response_geometry.shape[0])
    if budgets != tuple(sorted(set(budgets))) or budgets[-1] > source_modes:
        raise ValueError("packet budgets must be nested within source width")
    if target_rank is None:
        target_rank = int(response_geometry.shape[3])
    if target_rank != response_geometry.shape[3]:
        raise ValueError("graph-wavelet rung requires a full target basis")
    expected_graph = _require_sha256(
        expected_graph_basis_artifact_sha256,
        label="expected graph basis artifact",
    )
    fit_ordinals = torch.tensor(
        [measured_origins.index(origin) for origin in FIT_ORIGINS],
        dtype=torch.int64,
        device=response_geometry.device,
    )
    fit_kernels = _float_tensor(
        response_geometry.index_select(1, fit_ordinals),
        label="fit responses",
        ndim=4,
    )

    # Every fit API below receives a tensor containing fit origins only.  The
    # held-out values have not yet been validated, copied, or read.
    graph = fit_graph_source_bases(
        fit_kernels,
        scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=response_binding_sha256,
        fft_length=fft_length,
    )
    if graph.artifact_sha256 != expected_graph:
        raise ValueError("fit-only graph basis differs from the pinned artifact")

    signed_laplacian = _laplacian(
        graph.signed_eigenvalues,
        graph.signed_eigenvectors,
    )
    magnitude_laplacian = _laplacian(
        graph.magnitude_eigenvalues,
        graph.magnitude_eigenvectors,
    )
    signed_frame = build_spectral_graph_wavelet_frame(
        signed_laplacian,
        diffusion_scales=DIFFUSION_SCALES,
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
    )
    magnitude_frame = build_spectral_graph_wavelet_frame(
        magnitude_laplacian,
        diffusion_scales=DIFFUSION_SCALES,
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
    )
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    permutation = torch.randperm(
        source_modes,
        generator=generator,
        dtype=torch.int64,
    )
    permuted_laplacian = (
        signed_laplacian.index_select(0, permutation)
        .index_select(1, permutation)
        .contiguous()
    )
    permuted_frame = build_spectral_graph_wavelet_frame(
        permuted_laplacian,
        diffusion_scales=DIFFUSION_SCALES,
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
    )

    fit_target = (
        fit_kernels
        * scales.view(-1, 1, 1, 1)
    ).contiguous()
    signed_dictionary, signed_packets = _frame_dictionary(signed_frame)
    magnitude_dictionary, magnitude_packets = _frame_dictionary(
        magnitude_frame
    )
    permuted_dictionary, permuted_packets = _frame_dictionary(permuted_frame)
    compact_wavelet_scalars = (
        source_modes * source_modes
        + source_modes
        + signed_frame.filter_count * source_modes
    )
    families: dict[str, _NestedBasisFamily] = {}
    fit_signals = tuple(
        fit_target[:, ordinal].contiguous()
        for ordinal in range(fit_target.shape[1])
    )
    signed_omp = fit_graph_wavelet_omp_subspace(
        signed_frame,
        fit_signals,
        max_rank=budgets[-1],
    )
    magnitude_omp = fit_graph_wavelet_omp_subspace(
        magnitude_frame,
        fit_signals,
        max_rank=budgets[-1],
    )
    permuted_omp = fit_graph_wavelet_omp_subspace(
        permuted_frame,
        fit_signals,
        max_rank=budgets[-1],
    )

    def wavelet_family(
        method: str,
        subspace: object,
        *,
        integer_metadata: int = 0,
    ) -> _NestedBasisFamily:
        selected = tuple(
            getattr(subspace, "selected_flat_atom_indices")
        )
        basis = getattr(subspace, "orthonormal_basis")
        if not isinstance(basis, Tensor):
            raise TypeError("graph-wavelet OMP subspace lacks a basis")
        return _NestedBasisFamily(
            method=method,
            available_packet_count=(
                signed_frame.frame_coefficient_count
            ),
            selected_packet_order=selected,
            bases=MappingProxyType(
                {
                    budget: basis[:, :budget].clone()
                    for budget in budgets
                }
            ),
            source_basis_kind="fit_only_graph_wavelet_gomp",
            transform_metadata_semantics=(
                _TRANSFORM_METADATA_SEMANTICS[method]
            ),
            transform_metadata_float64_scalars=(
                compact_wavelet_scalars
            ),
            transform_metadata_integer_scalars=integer_metadata,
        )

    families["signed_graph_wavelet_omp"] = wavelet_family(
        "signed_graph_wavelet_omp",
        signed_omp,
    )
    families["magnitude_graph_wavelet_omp"] = wavelet_family(
        "magnitude_graph_wavelet_omp",
        magnitude_omp,
    )
    families["signed_graph_fourier_prefix"] = _prefix_family(
        signed_frame.eigenvectors,
        budgets,
        method="signed_graph_fourier_prefix",
        source_basis_kind="signed_phase_graph_low_frequency",
        transform_metadata_float64_scalars=(
            source_modes * source_modes + source_modes
        ),
    )
    fit_energy_basis, fit_energy_order = _fit_energy_ordered_basis(
        signed_frame.eigenvectors,
        fit_target,
    )
    families["signed_graph_fourier_fit_energy"] = _prefix_family(
        fit_energy_basis,
        budgets,
        method="signed_graph_fourier_fit_energy",
        source_basis_kind="fixed_orthonormal_control",
        transform_metadata_float64_scalars=(
            source_modes * source_modes + source_modes
        ),
        transform_metadata_integer_scalars=source_modes,
    )
    families["signed_graph_fourier_fit_energy"] = _NestedBasisFamily(
        method="signed_graph_fourier_fit_energy",
        available_packet_count=source_modes,
        selected_packet_order=fit_energy_order,
        bases=families["signed_graph_fourier_fit_energy"].bases,
        source_basis_kind="fixed_orthonormal_control",
        transform_metadata_semantics=(
            _TRANSFORM_METADATA_SEMANTICS[
                "signed_graph_fourier_fit_energy"
            ]
        ),
        transform_metadata_float64_scalars=(
            source_modes * source_modes + source_modes
        ),
        transform_metadata_integer_scalars=source_modes,
    )
    families["permuted_signed_graph_wavelet_omp"] = wavelet_family(
        "permuted_signed_graph_wavelet_omp",
        permuted_omp,
        integer_metadata=source_modes,
    )
    for seed, method in zip(
        RANDOM_ORTHONORMAL_SEEDS,
        RANDOM_CONTROL_METHODS,
        strict=True,
    ):
        random_basis = _random_orthonormal(source_modes, seed=seed)
        random_fit_basis, random_fit_order = _fit_energy_ordered_basis(
            random_basis,
            fit_target,
        )
        random_family = _prefix_family(
            random_fit_basis,
            budgets,
            method=method,
            source_basis_kind="fixed_orthonormal_control",
            transform_metadata_float64_scalars=(
                source_modes * source_modes
            ),
            transform_metadata_integer_scalars=source_modes,
        )
        families[method] = _NestedBasisFamily(
            method=method,
            available_packet_count=source_modes,
            selected_packet_order=random_fit_order,
            bases=random_family.bases,
            source_basis_kind="fixed_orthonormal_control",
            transform_metadata_semantics=(
                _TRANSFORM_METADATA_SEMANTICS[method]
            ),
            transform_metadata_float64_scalars=(
                source_modes * source_modes
            ),
            transform_metadata_integer_scalars=source_modes,
        )
    full_svd = fit_conditional_spectral_generator(
        fit_kernels,
        scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        budgets[-1],
        target_rank,
        response_binding_sha256=response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=fft_length,
    )
    families["fit_svd_prefix"] = _prefix_family(
        full_svd.source_basis,
        budgets,
        method="fit_svd_prefix",
        source_basis_kind="fixed_orthonormal_control",
        transform_metadata_float64_scalars=(
            source_modes * budgets[-1]
        ),
    )
    identity = torch.eye(source_modes, dtype=torch.float64)
    families["native_mode_omp"] = _group_omp_nested_bases(
        identity,
        tuple(range(source_modes)),
        fit_target,
        budgets,
        method="native_mode_omp",
        transform_metadata_float64_scalars=0,
    )
    if tuple(families) != METHOD_ORDER:
        raise RuntimeError("graph-wavelet method order drifted")

    dense_full_rank_plan_scalars = (
        source_modes * source_modes
        + response_geometry.shape[3] * target_rank
        + len(FIT_ORIGINS)
        * response_geometry.shape[2]
        * source_modes
        * target_rank
    )
    plans: list[ConditionalSpectralGeneratorPlan] = []
    fit_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        family = families[method]
        for budget in budgets:
            basis = family.bases[budget]
            plan = fit_conditional_spectral_generator_with_source_basis(
                fit_kernels,
                scales,
                FIT_ORIGINS,
                FIT_ORIGINS,
                basis,
                target_rank,
                source_basis_kind=family.source_basis_kind,  # type: ignore[arg-type]
                source_basis_fit_weighted_kernels_sha256=(
                    graph.fit_weighted_kernels_sha256
                ),
                response_binding_sha256=response_binding_sha256,
                input_transform="standardized_linear",
                fft_length=fft_length,
            )
            fit_evaluation = evaluate_conditional_spectral_generator(
                plan,
                fit_kernels,
                FIT_ORIGINS,
                FIT_ORIGINS,
                response_binding_sha256=response_binding_sha256,
            )
            atom_locality: Mapping[str, object] | None = None
            if method == "signed_graph_wavelet_omp":
                atom_locality = _selected_atom_locality(
                    signed_dictionary,
                    signed_packets,
                    family.selected_packet_order,
                    budget,
                )
            elif method == "magnitude_graph_wavelet_omp":
                atom_locality = _selected_atom_locality(
                    magnitude_dictionary,
                    magnitude_packets,
                    family.selected_packet_order,
                    budget,
                )
            elif method == "permuted_signed_graph_wavelet_omp":
                atom_locality = _selected_atom_locality(
                    permuted_dictionary,
                    permuted_packets,
                    family.selected_packet_order,
                    budget,
                )
            plans.append(plan)
            fit_rows.append(
                _plan_row_prefix(
                    family=family,
                    budget=budget,
                    basis=basis,
                    plan=plan,
                    fit_evaluation=fit_evaluation,
                    selected_atom_locality=atom_locality,
                    signed_laplacian=signed_laplacian,
                    dense_full_rank_plan_scalars=(
                        dense_full_rank_plan_scalars
                    ),
                )
            )

    # All graph/frame/basis/plan objects and fit evaluations are frozen before
    # this first validation and read of the selection response values.
    kernels = _float_tensor(
        response_geometry,
        label="responses",
        ndim=4,
    )
    rate_rows: list[dict[str, object]] = []
    for plan, prefix in zip(plans, fit_rows, strict=True):
        heldout = evaluate_conditional_spectral_generator(
            plan,
            kernels,
            measured_origins,
            SELECTION_ORIGINS,
            response_binding_sha256=response_binding_sha256,
            require_heldout=True,
        )
        row = dict(prefix)
        row["heldout_evaluation"] = _evaluation_metadata(heldout)
        row["passes_frozen_fidelity_gate"] = (
            row["method"] == "signed_graph_wavelet_omp"
            and heldout.weighted_relative_error
            <= MAXIMUM_SELECTION_RELATIVE_ERROR
            and heldout.weighted_cosine >= MINIMUM_SELECTION_COSINE
        )
        row["passes_topology_control_gate"] = False
        row["passes_graph_fourier_gate"] = False
        row["passes_locality_gate"] = False
        row["passes_localized_structural_candidate_gate"] = False
        row["passes_svd_gate"] = False
        row["passes_storage_gate"] = False
        row["passes_fidelity_and_storage_gate"] = False
        row["passes_compute_gate"] = False
        row["passes_controlled_candidate_gate"] = False
        rate_rows.append(row)

    by_method_budget = {
        (str(row["method"]), int(row["vector_packet_budget"])): row
        for row in rate_rows
    }
    for row in rate_rows:
        if row["method"] != "signed_graph_wavelet_omp":
            continue
        budget = int(row["vector_packet_budget"])
        primary_error = float(
            row["heldout_evaluation"]["weighted_relative_error"]  # type: ignore[index]
        )
        topology_controls = {
            method: float(
                by_method_budget[(method, budget)][
                    "heldout_evaluation"
                ][  # type: ignore[index]
                    "weighted_relative_error"
                ]
            )
            for method in (
                "magnitude_graph_wavelet_omp",
                "native_mode_omp",
                "permuted_signed_graph_wavelet_omp",
            )
        }
        random_errors = tuple(
            float(
                by_method_budget[(method, budget)][
                    "heldout_evaluation"
                ][  # type: ignore[index]
                    "weighted_relative_error"
                ]
            )
            for method in RANDOM_CONTROL_METHODS
        )
        graph_fourier_errors = {
            method: float(
                by_method_budget[(method, budget)][
                    "heldout_evaluation"
                ][  # type: ignore[index]
                    "weighted_relative_error"
                ]
            )
            for method in (
                "signed_graph_fourier_prefix",
                "signed_graph_fourier_fit_energy",
            )
        }
        best_graph_fourier_method = min(
            graph_fourier_errors,
            key=graph_fourier_errors.__getitem__,
        )
        best_graph_fourier_error = graph_fourier_errors[
            best_graph_fourier_method
        ]
        best_graph_fourier_row = by_method_budget[
            (best_graph_fourier_method, budget)
        ]
        primary_support = float(
            row["q_locality"]["mean_effective_node_support"]  # type: ignore[index]
        )
        graph_fourier_support = float(
            best_graph_fourier_row["q_locality"][  # type: ignore[index]
                "mean_effective_node_support"
            ]
        )
        locality_reduction = 1.0 - (
            primary_support / graph_fourier_support
        )
        graph_fourier_fidelity_loss = (
            primary_error - best_graph_fourier_error
        )
        svd_error = float(
            by_method_budget[("fit_svd_prefix", budget)][
                "heldout_evaluation"
            ]["weighted_relative_error"]  # type: ignore[index]
        )
        compiled_fraction = float(
            row["coefficient_payload"][  # type: ignore[index]
                "compiled_plan_fraction_of_full_rank"
            ]
        )
        random_win_count = sum(
            primary_error <= error + 1.0e-12 for error in random_errors
        )
        topology_gate = (
            all(
                primary_error <= error + 1.0e-12
                for error in topology_controls.values()
            )
            and primary_error <= float(median(random_errors)) + 1.0e-12
            and random_win_count >= MINIMUM_RANDOM_CONTROL_WINS
        )
        graph_fourier_gate = (
            primary_error <= best_graph_fourier_error + 1.0e-12
        )
        locality_gate = (
            locality_reduction >= MINIMUM_LOCALITY_REDUCTION
            and graph_fourier_fidelity_loss
            <= MAXIMUM_LOCALITY_FIDELITY_LOSS
        )
        svd_gate = primary_error <= svd_error + 1.0e-12
        storage_gate = (
            1.0 - compiled_fraction >= MINIMUM_COMPILED_PLAN_REDUCTION
        )
        row["topology_control_heldout_relative_errors"] = topology_controls
        row["random_control_heldout_relative_error"] = {
            "minimum": min(random_errors),
            "median": float(median(random_errors)),
            "maximum": max(random_errors),
            "primary_win_count": random_win_count,
            "required_win_count": MINIMUM_RANDOM_CONTROL_WINS,
        }
        row["graph_fourier_heldout_relative_errors"] = (
            graph_fourier_errors
        )
        row["best_graph_fourier_method"] = best_graph_fourier_method
        row["graph_fourier_fidelity_loss"] = (
            graph_fourier_fidelity_loss
        )
        row["q_locality_reduction_vs_best_graph_fourier"] = (
            locality_reduction
        )
        row["fit_svd_heldout_relative_error"] = svd_error
        row["passes_topology_control_gate"] = topology_gate
        row["passes_graph_fourier_gate"] = graph_fourier_gate
        row["passes_locality_gate"] = locality_gate
        row["passes_localized_structural_candidate_gate"] = (
            bool(row["passes_frozen_fidelity_gate"])
            and topology_gate
            and locality_gate
        )
        row["passes_svd_gate"] = svd_gate
        row["passes_storage_gate"] = storage_gate
        row["passes_fidelity_and_storage_gate"] = (
            bool(row["passes_frozen_fidelity_gate"]) and storage_gate
        )
        # This offline artifact has analytic plan storage but no matched
        # sequence execution or latency measurement.  Compute qualification
        # therefore remains fail-closed rather than inferred from rank.
        row["passes_compute_gate"] = False
        row["passes_controlled_candidate_gate"] = (
            bool(row["passes_frozen_fidelity_gate"])
            and topology_gate
            and graph_fourier_gate
            and svd_gate
            and storage_gate
            and bool(row["passes_compute_gate"])
        )

    controlled_passing = tuple(
        row
        for row in rate_rows
        if row["passes_controlled_candidate_gate"]
    )
    localized_passing = tuple(
        row
        for row in rate_rows
        if row["passes_localized_structural_candidate_gate"]
    )
    selected = (
        min(
            controlled_passing,
            key=lambda row: (
                int(
                    row["coefficient_payload"][  # type: ignore[index]
                        "standalone_compiler_plus_plan_float64_scalars"
                    ]
                ),
                int(row["vector_packet_budget"]),
            ),
        )
        if controlled_passing
        else None
    )
    localized_nominee = (
        min(
            localized_passing,
            key=lambda row: int(row["vector_packet_budget"]),
        )
        if localized_passing
        else None
    )
    weighted_all = (kernels * scales.view(-1, 1, 1, 1)).contiguous()
    frame_receipts = {
        "signed": {
            **signed_frame.metadata(),
            "exactness": _frame_exactness(signed_frame, weighted_all),
            "locality_summary": signed_frame.scale_localization_summary(),
            "fit_only_omp_subspace": signed_omp.metadata(),
        },
        "magnitude": {
            **magnitude_frame.metadata(),
            "exactness": _frame_exactness(magnitude_frame, weighted_all),
            "locality_summary": magnitude_frame.scale_localization_summary(),
            "fit_only_omp_subspace": magnitude_omp.metadata(),
        },
        "permuted_signed_control": {
            **permuted_frame.metadata(),
            "exactness": _frame_exactness(permuted_frame, weighted_all),
            "permutation": tuple(int(value) for value in permutation.tolist()),
            "conjugated_laplacian_control": True,
            "center_reorder_only": False,
            "fit_only_omp_subspace": permuted_omp.metadata(),
        },
    }
    compact_scalars = sum(
        int(tensor.numel())
        for tensor in (
            signed_frame.eigenvalues,
            signed_frame.eigenvectors,
            signed_frame.spectral_kernels,
            magnitude_frame.eigenvalues,
            magnitude_frame.eigenvectors,
            magnitude_frame.spectral_kernels,
        )
    )
    protocol = {
        "response_component": "local_central_odd_tangent",
        "response_tensor_order": "source_origin_lag_target",
        "source_modes": source_modes,
        "target_modes": int(kernels.shape[3]),
        "lag_count": int(kernels.shape[2]),
        "fft_length": fft_length,
        "measured_origins": measured_origins,
        "fit_origins": FIT_ORIGINS,
        "selection_origins": SELECTION_ORIGINS,
        "packet_budgets": budgets,
        "diffusion_scales": DIFFUSION_SCALES,
        "eigenvalue_tolerance": EIGENVALUE_TOLERANCE,
        "permutation_seed": PERMUTATION_SEED,
        "random_orthonormal_seeds_used": RANDOM_ORTHONORMAL_SEEDS,
        "method_order": METHOD_ORDER,
        "primary_atom_selection": (
            "fit_only_group_omp_by_residual_group_correlation_"
            "deterministic_packet_index_tie_break"
        ),
        "primary_basis_construction": (
            "nested_incremental_modified_gram_schmidt_qr"
        ),
        "heldout_masks_or_subspaces_refit": False,
        "target_rank_is_full": True,
        "selection_gate": {
            "maximum_weighted_relative_error": (
                MAXIMUM_SELECTION_RELATIVE_ERROR
            ),
            "minimum_weighted_cosine": MINIMUM_SELECTION_COSINE,
            "minimum_compiled_plan_reduction": (
                MINIMUM_COMPILED_PLAN_REDUCTION
            ),
            "minimum_locality_reduction": MINIMUM_LOCALITY_REDUCTION,
            "maximum_locality_fidelity_loss": (
                MAXIMUM_LOCALITY_FIDELITY_LOSS
            ),
            "minimum_random_control_wins": (
                MINIMUM_RANDOM_CONTROL_WINS
            ),
            "requires_same_budget_topology_control_gate": True,
            "requires_same_budget_graph_fourier_gate": True,
            "requires_same_budget_fit_svd_gate": True,
            "requires_measured_compute_gate": True,
            "selection": (
                "minimal_standalone_compiler_plus_plan_storage_then_budget"
            ),
        },
    }
    resource_accounting = {
        "model_load_count": 0,
        "tokenizer_load_count": 0,
        "prompt_text_read_count": 0,
        "token_id_read_count": 0,
        "model_forward_count": 0,
        "new_response_measurement_count": 0,
        "authenticated_source_tensor_file_count": 1,
        "authenticated_source_tensor_file_bytes": source_tensor_file_bytes,
        "measured_response_float64_scalars_read": int(kernels.numel()),
        "fit_response_float64_scalars": int(fit_target.numel()),
        "selection_response_float64_scalars": (
            source_modes
            * len(SELECTION_ORIGINS)
            * kernels.shape[2]
            * kernels.shape[3]
        ),
        "fit_only_graph_eigendecomposition_count": 2,
        "wavelet_frame_eigendecomposition_count": 3,
        "fit_svd_reference_count": 1,
        "random_qr_count": len(RANDOM_ORTHONORMAL_SEEDS),
        "conditional_plan_fit_count": len(plans),
        "conditional_plan_heldout_evaluation_count": len(plans),
        "compact_serialized_float64_scalars": compact_scalars,
        "compact_serialized_float64_bytes": compact_scalars * 8,
        "raw_response_tensors_serialized": False,
        "dense_per_scale_operators_serialized": False,
        "transient_plan_tensors_serialized": False,
    }
    graph_receipt = {
        "artifact_sha256": graph.artifact_sha256,
        "response_binding_sha256": graph.response_binding_sha256,
        "fit_weighted_kernels_sha256": graph.fit_weighted_kernels_sha256,
        "fit_origins": graph.fit_origins,
        "fft_length": graph.fft_length,
        "heldout_origins_used_for_basis": (
            graph.heldout_origins_used_for_basis
        ),
        "signed_graph_semantics": graph.signed_graph_semantics,
        "magnitude_graph_semantics": graph.magnitude_graph_semantics,
    }
    conclusions = {
        "selected_signed_graph_wavelet_budget": (
            int(selected["vector_packet_budget"])
            if selected is not None
            else None
        ),
        "selected_plan_artifact_sha256": (
            selected["plan_artifact_sha256"]
            if selected is not None
            else None
        ),
        "signed_graph_wavelet_controlled_passing_budgets": tuple(
            int(row["vector_packet_budget"]) for row in controlled_passing
        ),
        "signed_graph_wavelet_fidelity_passing_budgets": tuple(
            int(row["vector_packet_budget"])
            for row in rate_rows
            if row["method"] == "signed_graph_wavelet_omp"
            and row["passes_frozen_fidelity_gate"]
        ),
        "signed_graph_wavelet_storage_passing_budgets": tuple(
            int(row["vector_packet_budget"])
            for row in rate_rows
            if row["method"] == "signed_graph_wavelet_omp"
            and row["passes_storage_gate"]
        ),
        "signed_graph_wavelet_fidelity_and_storage_passing_budgets": tuple(
            int(row["vector_packet_budget"])
            for row in rate_rows
            if row["method"] == "signed_graph_wavelet_omp"
            and row["passes_fidelity_and_storage_gate"]
        ),
        "localized_structural_nominee_budget": (
            int(localized_nominee["vector_packet_budget"])
            if localized_nominee is not None
            else None
        ),
        "localized_structural_passing_budgets": tuple(
            int(row["vector_packet_budget"])
            for row in localized_passing
        ),
        "winner_requires_fidelity_topology_graph_fourier_svd_storage_and_"
        "compute_gates": True,
        "compute_gate_measured": False,
        "random_control_panel_size": len(RANDOM_ORTHONORMAL_SEEDS),
        "fit_and_selection_are_disjoint": True,
        "selection_is_open_development_not_confirmation": True,
        "full_frame_reconstruction_is_a_numerical_identity_check": True,
        "packet_rows_are_frozen_fit_knot_interpolated_executors": True,
        "hop_locality_not_used_for_claims_because_graph_is_dense": True,
        "q_locality_is_coordinate_support_and_graph_quadratic_form_only": True,
        "fixed_reference_structural_response_only": True,
        "natural_prompt_or_nll_fidelity_measured": False,
        "whole_model_replacement_claim": False,
        "model_compression_claim": False,
        "latency_or_speed_claim": False,
    }
    return Gemma3GraphWaveletCandidate(
        source_receipt=source_receipt,
        protocol=protocol,
        graph_receipt=graph_receipt,
        frame_receipts=frame_receipts,
        rate_rows=tuple(rate_rows),
        conclusions=conclusions,
        resource_accounting=resource_accounting,
        signed_eigenvalues=signed_frame.eigenvalues,
        signed_eigenvectors=signed_frame.eigenvectors,
        signed_spectral_kernels=signed_frame.spectral_kernels,
        magnitude_eigenvalues=magnitude_frame.eigenvalues,
        magnitude_eigenvectors=magnitude_frame.eigenvectors,
        magnitude_spectral_kernels=magnitude_frame.spectral_kernels,
    )


def compile_gemma3_graph_wavelet_candidate(
    source: Gemma3SpectralSource,
) -> Gemma3GraphWaveletCandidate:
    """Compile the frozen offline rung from one authenticated Gemma source."""

    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    source.mapping.validate_integrity()
    if source.mapping.impulse_logical_positions != INTERIOR_ORIGINS:
        raise ValueError("source origins differ from the frozen split")
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    response_binding = _response_binding(
        source,
        component="local_central_odd_tangent",
    )
    return _compile_from_response(
        local.impulse_responses,
        source.source_mode_standard_deviations,
        source.mapping.impulse_logical_positions,
        response_binding_sha256=response_binding,
        expected_graph_basis_artifact_sha256=(
            EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256
        ),
        source_receipt={
            "tensor_file_sha256": source.file_sha256,
            "report_file_sha256": source.report_file_sha256,
            "report_payload_sha256": source.report_payload_sha256,
            "mapping_artifact_sha256": source.mapping.artifact_sha256,
            "response_artifact_sha256": local.artifact_sha256,
            "source_model_sha256": source.binding.get("source_model_sha256"),
        },
        fft_length=source.mapping.fft_length,
        source_tensor_file_bytes=0,
    )


def _publish_candidate(
    candidate: Gemma3GraphWaveletCandidate,
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
            "scientific_status": dict(candidate.conclusions),
            "safety": _SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        report_stage = _stage_json(report, report_path)
        reservation.publish((tensor_stage, report_stage))
        return report
    finally:
        reservation.release()
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def analyze_gemma3_l3_l4_graph_wavelets(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    source_artifact_sha256: str = DEFAULT_INTERIOR_ARTIFACT_SHA256,
    source_report_sha256: str = DEFAULT_INTERIOR_REPORT_SHA256,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Strict-load the pinned measurement, analyze it, and publish once."""

    if source_artifact_sha256 != DEFAULT_INTERIOR_ARTIFACT_SHA256:
        raise ValueError("source tensor must equal the pinned interior artifact")
    if source_report_sha256 != DEFAULT_INTERIOR_REPORT_SHA256:
        raise ValueError("source report must equal the pinned interior report")
    destination = _validate_output_path(output, suffix=".pt")
    if destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite graph-wavelet report")
    source_path = Path(source_artifact_path)
    source = load_gemma3_spectral_source(
        source_path,
        expected_file_sha256=source_artifact_sha256,
        expected_report_sha256=source_report_sha256,
        expected_origins=INTERIOR_ORIGINS,
    )
    candidate = compile_gemma3_graph_wavelet_candidate(source)
    # File byte accounting is intentionally added before hashing/publication.
    accounting = dict(candidate.resource_accounting)
    accounting["authenticated_source_tensor_file_bytes"] = (
        source_path.stat().st_size
    )
    candidate = Gemma3GraphWaveletCandidate(
        source_receipt=candidate.source_receipt,
        protocol=candidate.protocol,
        graph_receipt=candidate.graph_receipt,
        frame_receipts=candidate.frame_receipts,
        rate_rows=candidate.rate_rows,
        conclusions=candidate.conclusions,
        resource_accounting=accounting,
        signed_eigenvalues=candidate.signed_eigenvalues,
        signed_eigenvectors=candidate.signed_eigenvectors,
        signed_spectral_kernels=candidate.signed_spectral_kernels,
        magnitude_eigenvalues=candidate.magnitude_eigenvalues,
        magnitude_eigenvectors=candidate.magnitude_eigenvectors,
        magnitude_spectral_kernels=candidate.magnitude_spectral_kernels,
    )
    return _publish_candidate(candidate, output=destination)


def load_gemma3_graph_wavelet_candidate(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> Gemma3GraphWaveletCandidate:
    """Authenticate and restore one compact graph-wavelet publication."""

    source = Path(path)
    expected_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected artifact",
    )
    expected_tensor = _require_sha256(
        expected_tensor_file_sha256,
        label="expected tensor file",
    )
    expected_report = _require_sha256(
        expected_report_sha256,
        label="expected report",
    )
    actual_tensor = _file_sha256(source)
    if actual_tensor != expected_tensor:
        raise ValueError("graph-wavelet tensor file hash differs")
    with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    claimed_report = _require_sha256(
        report.get("report_sha256"),
        label="report SHA-256",
    )
    payload = dict(report)
    payload.pop("report_sha256")
    if (
        claimed_report != expected_report
        or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed_report
        or report["artifact"]["tensor_file_sha256"] != actual_tensor
    ):
        raise ValueError("graph-wavelet report binding differs")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    candidate = Gemma3GraphWaveletCandidate.from_state_dict(raw)
    if (
        candidate.artifact_sha256 != expected_artifact
        or report["candidate"]["artifact_sha256"] != expected_artifact
        or _canonical_json_bytes(report["candidate"])
        != _canonical_json_bytes(candidate.metadata())
    ):
        raise ValueError("graph-wavelet logical artifact differs")
    if (
        candidate.artifact_sha256 == _LEGACY_V1_ARTIFACT_SHA256
        and (
            actual_tensor != _LEGACY_V1_TENSOR_FILE_SHA256
            or claimed_report != _LEGACY_V1_REPORT_SHA256
        )
    ):
        raise ValueError("graph-wavelet legacy publication receipt differs")
    return candidate


def describe_gemma3_l3_l4_graph_wavelets() -> dict[str, object]:
    """Describe the preregistered offline rung without opening an artifact."""

    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "source": {
            "artifact": str(DEFAULT_INTERIOR_ARTIFACT),
            "tensor_file_sha256": DEFAULT_INTERIOR_ARTIFACT_SHA256,
            "report_payload_sha256": DEFAULT_INTERIOR_REPORT_SHA256,
            "expected_graph_basis_artifact_sha256": (
                EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256
            ),
        },
        "protocol": {
            "fit_origins": FIT_ORIGINS,
            "selection_origins": SELECTION_ORIGINS,
            "packet_budgets": PACKET_BUDGETS,
            "diffusion_scales": DIFFUSION_SCALES,
            "method_order": METHOD_ORDER,
            "primary_method": "signed_graph_wavelet_omp",
            "selection_gate": {
                "maximum_weighted_relative_error": (
                    MAXIMUM_SELECTION_RELATIVE_ERROR
                ),
                "minimum_weighted_cosine": MINIMUM_SELECTION_COSINE,
                "minimum_compiled_plan_reduction": (
                    MINIMUM_COMPILED_PLAN_REDUCTION
                ),
                "minimum_locality_reduction": (
                    MINIMUM_LOCALITY_REDUCTION
                ),
                "maximum_locality_fidelity_loss": (
                    MAXIMUM_LOCALITY_FIDELITY_LOSS
                ),
                "minimum_random_control_wins": (
                    MINIMUM_RANDOM_CONTROL_WINS
                ),
                "compute_gate_is_fail_closed_without_measurement": True,
            },
        },
        "resource_contract": {
            "model_load_count": 0,
            "tokenizer_load_count": 0,
            "prompt_text_read_count": 0,
            "token_id_read_count": 0,
            "model_forward_count": 0,
            "new_response_measurement_count": 0,
        },
        "safety": _SAFETY,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the pinned prompt-free Gemma L3-to-L4 response with "
            "fit-only graph-wavelet packet bases."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe")
    analyze = commands.add_parser("analyze")
    analyze.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_INTERIOR_ARTIFACT,
    )
    analyze.add_argument(
        "--source-artifact-sha256",
        default=DEFAULT_INTERIOR_ARTIFACT_SHA256,
    )
    analyze.add_argument(
        "--source-report-sha256",
        default=DEFAULT_INTERIOR_REPORT_SHA256,
    )
    analyze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "describe":
        result = describe_gemma3_l3_l4_graph_wavelets()
    else:
        result = analyze_gemma3_l3_l4_graph_wavelets(
            source_artifact_path=arguments.source_artifact,
            source_artifact_sha256=arguments.source_artifact_sha256,
            source_report_sha256=arguments.source_report_sha256,
            output=arguments.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
