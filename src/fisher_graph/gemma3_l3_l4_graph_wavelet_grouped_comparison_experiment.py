"""Matched grouped graph-wavelet basis comparison for Gemma L3 -> L4.

The experiment reconstructs the authenticated rank-64 graph-wavelet GOMP
parent from origins 8/24/40, freezes every rank-45 source basis and compensated
conditional plan, and only then reads origins 16/32.  It compares:

* existing GOMP, diagonal, GFA, global-SVD, and pair-supermode references;
* graph-partitioned local SVD at 4, 8, and 16 balanced groups;
* graph-partitioned local graph-Fourier bases at the same granularities;
* signed and magnitude topology, matched one-hot controls, and four permuted
  signed-topology local-SVD controls.

The artifact is metadata only.  It contains no raw response, Q64, partition,
or compiled-plan tensors and makes no NLL, model-compression, or speed claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import Tensor

from .conditional_spectral_generator import (
    fit_conditional_spectral_generator,
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
from .gemma3_l3_l4_graph_wavelet_comparison_support import (
    MatchedBasisFamily,
    evaluate_frozen_plan_panel,
    freeze_matched_plan_panel,
    reconstruct_authenticated_q64_fit_context,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256,
    Gemma3GraphWaveletCandidate,
    _laplacian,
    _tensor_sha256,
    load_gemma3_graph_wavelet_candidate,
)
from .gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
    MINIMUM_LOFO_LOADING_STABILITY,
    MINIMUM_SQUARED_LOADING,
    PERMUTED_GRAPH_LOCAL_SEEDS,
    TOPOLOGY_TOP_K,
)
from .graph_wavelet_grouped_basis import (
    FitOnlyGraphWaveletGroupedBasis,
    GraphWaveletTopologyPartition,
    fit_graph_wavelet_grouped_basis,
    fit_graph_wavelet_topology_partition,
    grouped_basis_one_hot_control,
    grouped_basis_projector_overlap,
)
from .graph_wavelet_supermode_merge import (
    GraphWaveletSupermodePair,
    fit_graph_wavelet_supermode_merge,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "Gemma3GraphWaveletGroupedComparisonCandidate",
    "analyze_gemma3_l3_l4_graph_wavelet_grouped_comparison",
    "build_parser",
    "compile_gemma3_graph_wavelet_grouped_comparison_candidate",
    "describe_gemma3_l3_l4_graph_wavelet_grouped_comparison",
    "load_gemma3_graph_wavelet_grouped_comparison_candidate",
    "main",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-grouped-comparison-dev-v1.pt"
)
TARGET_RANK = 45
GROUP_COUNTS = (4, 8, 16)
CONTROL_GROUP_COUNT = 8
MAXIMUM_SELECTION_RELATIVE_ERROR = 0.20
MINIMUM_SELECTION_COSINE = 0.98

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_graph_wavelet_grouped_comparison_development"
)
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-wavelet-grouped-comparison:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-wavelet-grouped-comparison-report:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_raw_response_tensors": False,
    "contains_compiled_plan_tensors": False,
    "contains_parent_basis_tensor": False,
    "contains_partition_tensors": False,
    "metadata_only_candidate": True,
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


def _metadata(value: object, *, label: str) -> object:
    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(
            _canonical_json_bytes(thaw(value)).decode("ascii")
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain JSON metadata only") from error


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _method_order(
    group_counts: Sequence[int],
    *,
    control_group_count: int,
    permutation_seeds: Sequence[int],
) -> tuple[str, ...]:
    grouped = tuple(
        f"{topology}_{method}_g{group_count}"
        for topology in ("signed", "magnitude")
        for group_count in group_counts
        for method in ("local_svd", "cluster_gfa")
    )
    return (
        "gomp_prefix",
        "diagonal_energy_prune",
        "signed_gfa_fit_energy",
        "fit_svd",
        "graph_local_pair_supermode",
        *grouped,
        f"signed_local_svd_g{control_group_count}_one_hot",
        f"magnitude_local_svd_g{control_group_count}_one_hot",
        *(
            f"permuted_signed_local_svd_g{control_group_count}_seed_{seed}"
            for seed in permutation_seeds
        ),
    )


def _energy_ordered_basis(
    basis: Tensor,
    weighted_fit: Tensor,
) -> tuple[Tensor, tuple[int, ...]]:
    coordinates = (
        basis.T @ weighted_fit.reshape(weighted_fit.shape[0], -1)
    )
    energies = coordinates.square().sum(dim=1)
    order = tuple(
        sorted(
            range(basis.shape[1]),
            key=lambda index: (-float(energies[index]), index),
        )
    )
    index = torch.tensor(order, dtype=torch.int64)
    return basis.index_select(1, index).contiguous(), order


def _basis_mixing(parent_basis: Tensor, basis: Tensor) -> dict[str, object]:
    probability = (parent_basis.T @ basis).square()
    participation = 1.0 / probability.square().sum(dim=0)
    contributor_counts = (probability >= 0.10).sum(dim=0)
    return {
        "mean_parent_coordinate_participation": float(participation.mean()),
        "maximum_parent_coordinate_participation": float(
            participation.max()
        ),
        "columns_with_at_least_three_ten_percent_contributors": int(
            (contributor_counts >= 3).sum()
        ),
        "maximum_ten_percent_contributor_count": int(
            contributor_counts.max()
        ),
        "dense_loading_threshold_is_squared_loading": 0.10,
    }


def _rank_allocation(
    family: FitOnlyGraphWaveletGroupedBasis,
    *,
    rank: int,
    group_count: int,
) -> tuple[int, ...]:
    return tuple(
        family.component_group_ordinals[:rank].count(group)
        for group in range(group_count)
    )


def _lofo_overlaps(
    *,
    parent_basis: Tensor,
    partition: GraphWaveletTopologyPartition,
    full_family: FitOnlyGraphWaveletGroupedBasis,
    fit_folds: Sequence[Tensor],
    fit_origins: Sequence[int],
    response_binding_sha256: str,
    parent_subspace_artifact_sha256: str,
    method: str,
    rank: int,
) -> tuple[float, ...]:
    overlaps = []
    for omitted in range(len(fit_folds)):
        remaining = torch.cat(
            [
                fold
                for index, fold in enumerate(fit_folds)
                if index != omitted
            ],
            dim=1,
        ).contiguous()
        replay = fit_graph_wavelet_grouped_basis(
            parent_basis,
            remaining,
            partition,
            method=method,  # type: ignore[arg-type]
            fit_origins=tuple(
                origin
                for index, origin in enumerate(fit_origins)
                if index != omitted
            ),
            response_binding_sha256=response_binding_sha256,
            parent_subspace_artifact_sha256=(
                parent_subspace_artifact_sha256
            ),
        )
        overlaps.append(
            grouped_basis_projector_overlap(
                full_family.prefix(rank),
                replay.prefix(rank),
            )
        )
    return tuple(overlaps)


def _grouped_receipt(
    *,
    parent_basis: Tensor,
    partition: GraphWaveletTopologyPartition,
    family: FitOnlyGraphWaveletGroupedBasis,
    fit_folds: Sequence[Tensor],
    fit_origins: Sequence[int],
    response_binding_sha256: str,
    parent_subspace_artifact_sha256: str,
    rank: int,
    topology_kind: str,
) -> dict[str, object]:
    overlaps = _lofo_overlaps(
        parent_basis=parent_basis,
        partition=partition,
        full_family=family,
        fit_folds=fit_folds,
        fit_origins=fit_origins,
        response_binding_sha256=response_binding_sha256,
        parent_subspace_artifact_sha256=(
            parent_subspace_artifact_sha256
        ),
        method=family.method,
        rank=rank,
    )
    return {
        "construction_kind": family.method,
        "topology_kind": topology_kind,
        "partition": partition.metadata(),
        "grouped_basis": family.metadata(),
        "rank_allocation": _rank_allocation(
            family,
            rank=rank,
            group_count=partition.group_count,
        ),
        "lofo_rank_projector_overlaps": overlaps,
        "minimum_lofo_rank_projector_overlap": min(overlaps),
        "basis_mixing": _basis_mixing(
            parent_basis,
            family.prefix(rank),
        ),
        "cross_group_rotations_permitted": False,
        "global_svd_equivalence_claim": False,
        "heldout_values_used_for_construction": False,
    }


@dataclass(frozen=True, slots=True)
class Gemma3GraphWaveletGroupedComparisonCandidate:
    """Authenticated metadata-only matched comparison result."""

    source_receipt: Mapping[str, object]
    parent_receipt: Mapping[str, object]
    protocol: Mapping[str, object]
    construction_receipts: Mapping[str, object]
    rate_rows: tuple[Mapping[str, object], ...]
    conclusions: Mapping[str, object]
    resource_accounting: Mapping[str, object]
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.format_version != _FORMAT_VERSION:
            raise ValueError("grouped comparison candidate header differs")
        for field in (
            "source_receipt",
            "parent_receipt",
            "protocol",
            "construction_receipts",
            "conclusions",
            "resource_accounting",
        ):
            normalized = _metadata(getattr(self, field), label=field)
            if not isinstance(normalized, dict):
                raise TypeError(f"{field} must normalize to an object")
            object.__setattr__(self, field, normalized)
        normalized_rows = _metadata(self.rate_rows, label="rate_rows")
        if not isinstance(normalized_rows, list) or any(
            not isinstance(row, dict) for row in normalized_rows
        ):
            raise TypeError("rate_rows must normalize to objects")
        object.__setattr__(
            self,
            "rate_rows",
            tuple(normalized_rows),  # type: ignore[arg-type]
        )
        self._validate_semantics()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            ) != computed:
                raise ValueError("grouped comparison candidate hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_receipt": dict(self.source_receipt),
            "parent_receipt": dict(self.parent_receipt),
            "protocol": dict(self.protocol),
            "construction_receipts": dict(self.construction_receipts),
            "rate_rows": tuple(dict(row) for row in self.rate_rows),
            "conclusions": dict(self.conclusions),
            "resource_accounting": dict(self.resource_accounting),
            "safety": _SAFETY,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)

    def _validate_semantics(self) -> None:
        methods = tuple(self.protocol.get("method_order", ()))
        rank = self.protocol.get("target_source_rank")
        if (
            not methods
            or len(self.rate_rows) != len(methods)
            or tuple(row.get("method") for row in self.rate_rows) != methods
            or any(row.get("rank") != rank for row in self.rate_rows)
            or set(self.construction_receipts) != set(methods)
            or _contains_tensor(self._payload())
        ):
            raise ValueError("grouped comparison method panel differs")
        payloads = {
            (
                row["coefficient_payload"][
                    "compiled_plan_float64_scalars"
                ],
                row["coefficient_payload"][
                    "prepared_runtime_storage_bytes"
                ],
            )
            for row in self.rate_rows
        }
        if (
            len(payloads) != 1
            or any(row.get("heldout_evaluation") is None for row in self.rate_rows)
            or self.conclusions.get(
                "all_methods_have_exact_equal_rank_plan_payload"
            )
            is not True
            or any(
                self.resource_accounting.get(field) != 0
                for field in (
                    "model_load_count",
                    "tokenizer_load_count",
                    "prompt_text_read_count",
                    "token_id_read_count",
                    "model_forward_count",
                    "new_response_measurement_count",
                )
            )
        ):
            raise ValueError("grouped comparison accounting differs")

    def validate_integrity(self) -> None:
        self._validate_semantics()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("grouped comparison candidate hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        value: object,
    ) -> "Gemma3GraphWaveletGroupedComparisonCandidate":
        if not isinstance(value, Mapping):
            raise TypeError("grouped comparison state must be a mapping")
        expected = {
            "schema",
            "format_version",
            "source_receipt",
            "parent_receipt",
            "protocol",
            "construction_receipts",
            "rate_rows",
            "conclusions",
            "resource_accounting",
            "safety",
            "artifact_sha256",
        }
        if set(value) != expected or value.get("safety") != _SAFETY:
            raise ValueError("grouped comparison state fields differ")
        return cls(
            source_receipt=value["source_receipt"],  # type: ignore[arg-type]
            parent_receipt=value["parent_receipt"],  # type: ignore[arg-type]
            protocol=value["protocol"],  # type: ignore[arg-type]
            construction_receipts=value[
                "construction_receipts"
            ],  # type: ignore[arg-type]
            rate_rows=tuple(value["rate_rows"]),  # type: ignore[arg-type]
            conclusions=value["conclusions"],  # type: ignore[arg-type]
            resource_accounting=value[
                "resource_accounting"
            ],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )


def _compile_from_response(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    parent: Gemma3GraphWaveletCandidate,
    *,
    response_binding_sha256: str,
    expected_graph_basis_artifact_sha256: str,
    source_receipt: Mapping[str, object],
    fft_length: int,
    target_source_rank: int = TARGET_RANK,
    group_counts: Sequence[int] = GROUP_COUNTS,
    control_group_count: int = CONTROL_GROUP_COUNT,
    permutation_seeds: Sequence[int] = PERMUTED_GRAPH_LOCAL_SEEDS,
    source_tensor_file_bytes: int = 0,
    parent_tensor_file_bytes: int = 0,
) -> Gemma3GraphWaveletGroupedComparisonCandidate:
    """Compile all fit-frozen methods before crossing the selection boundary."""

    counts = tuple(group_counts)
    seeds = tuple(permutation_seeds)
    context = reconstruct_authenticated_q64_fit_context(
        responses,
        source_scales,
        origins,
        parent,
        response_binding_sha256=response_binding_sha256,
        expected_graph_basis_artifact_sha256=(
            expected_graph_basis_artifact_sha256
        ),
        source_receipt=source_receipt,
        fft_length=fft_length,
        expected_parent_rank=int(source_scales.numel()),
    )
    parent_rank = context.q64.shape[1]
    if (
        not 1 <= target_source_rank < parent_rank
        or not counts
        or tuple(sorted(set(counts))) != counts
        or control_group_count not in counts
        or any(
            parent_rank % count != 0
            or ((parent_rank // count) & (parent_rank // count - 1))
            for count in counts
        )
    ):
        raise ValueError("grouped comparison rank or group schedule differs")
    topology_top_k = min(TOPOLOGY_TOP_K, parent_rank - 1)
    method_order = _method_order(
        counts,
        control_group_count=control_group_count,
        permutation_seeds=seeds,
    )
    q = context.q64
    weighted_fit = context.weighted_fit
    signed_laplacian = context.signed_laplacian
    magnitude_laplacian = _laplacian(
        context.graph.magnitude_eigenvalues,
        context.graph.magnitude_eigenvectors,
    )

    families: list[MatchedBasisFamily] = []
    receipts: dict[str, object] = {}

    def add_family(
        method: str,
        basis: Tensor,
        source_basis_kind: str,
        receipt: Mapping[str, object],
    ) -> None:
        complete_receipt = {
            **dict(receipt),
            "basis_sha256": _tensor_sha256(basis),
            "basis_mixing": _basis_mixing(q, basis),
        }
        families.append(
            MatchedBasisFamily(
                method=method,
                source_basis_kind=source_basis_kind,
                bases={target_source_rank: basis},
                construction_receipt=complete_receipt,
            )
        )
        receipts[method] = complete_receipt

    add_family(
        "gomp_prefix",
        q[:, :target_source_rank].clone(),
        "fit_only_graph_wavelet_gomp",
        {
            "construction_kind": "parent_gomp_prefix",
            "parent_subspace_artifact_sha256": (
                context.parent_subspace.artifact_sha256
            ),
            "selected_packet_prefix": (
                context.parent_packet_order[:target_source_rank]
            ),
        },
    )
    diagonal_basis, diagonal_order = _energy_ordered_basis(q, weighted_fit)
    add_family(
        "diagonal_energy_prune",
        diagonal_basis[:, :target_source_rank],
        "fixed_orthonormal_control",
        {
            "construction_kind": "parent_coordinate_fit_energy_prune",
            "fit_only_parent_order": diagonal_order,
        },
    )
    gfa_basis, gfa_order = _energy_ordered_basis(
        context.graph.signed_eigenvectors,
        weighted_fit,
    )
    add_family(
        "signed_gfa_fit_energy",
        gfa_basis[:, :target_source_rank],
        "fixed_orthonormal_control",
        {
            "construction_kind": "signed_graph_fourier_fit_energy_order",
            "fit_only_parent_order": gfa_order,
        },
    )
    global_plan = fit_conditional_spectral_generator(
        context.fit_kernels,
        context.source_scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        target_source_rank,
        context.target_modes,
        response_binding_sha256=context.response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=context.fft_length,
    )
    add_family(
        "fit_svd",
        global_plan.source_basis,
        "fixed_orthonormal_control",
        {
            "construction_kind": "unrestricted_global_fit_svd_ceiling",
            "construction_plan_artifact_sha256": global_plan.artifact_sha256,
            "localized_method_claim": False,
        },
    )
    pair_path = fit_graph_wavelet_supermode_merge(
        q,
        weighted_fit,
        signed_laplacian,
        minimum_rank=target_source_rank,
        method="graph_local",
        fit_fold_responses=context.fit_folds,
        minimum_squared_loading=MINIMUM_SQUARED_LOADING,
        minimum_lofo_loading_stability=MINIMUM_LOFO_LOADING_STABILITY,
        topology_top_k=topology_top_k,
    )
    pair_actions = pair_path.action_prefix(target_source_rank)
    add_family(
        "graph_local_pair_supermode",
        pair_path.basis(target_source_rank),
        "fit_only_graph_wavelet_local_supermodes",
        {
            "construction_kind": "endpoint_disjoint_pair_supermode",
            "path_artifact_sha256": pair_path.artifact_sha256,
            "action_count": len(pair_actions),
            "genuine_merge_count": sum(
                isinstance(action, GraphWaveletSupermodePair)
                for action in pair_actions
            ),
        },
    )

    grouped: dict[
        tuple[str, int, str],
        tuple[GraphWaveletTopologyPartition, FitOnlyGraphWaveletGroupedBasis],
    ] = {}
    for topology_kind, laplacian in (
        ("signed", signed_laplacian),
        ("magnitude", magnitude_laplacian),
    ):
        for group_count in counts:
            partition = fit_graph_wavelet_topology_partition(
                q,
                laplacian,
                group_count=group_count,
                topology_top_k=topology_top_k,
            )
            for core_method, method_suffix, source_kind in (
                (
                    "wavelet_local_svd",
                    "local_svd",
                    "fit_only_graph_wavelet_local_block_svd",
                ),
                (
                    "wavelet_cluster_gfa",
                    "cluster_gfa",
                    "fit_only_graph_wavelet_cluster_spectral",
                ),
            ):
                result = fit_graph_wavelet_grouped_basis(
                    q,
                    weighted_fit,
                    partition,
                    method=core_method,  # type: ignore[arg-type]
                    fit_origins=FIT_ORIGINS,
                    response_binding_sha256=(
                        context.response_binding_sha256
                    ),
                    parent_subspace_artifact_sha256=(
                        context.parent_subspace.artifact_sha256
                    ),
                )
                method = f"{topology_kind}_{method_suffix}_g{group_count}"
                receipt = _grouped_receipt(
                    parent_basis=q,
                    partition=partition,
                    family=result,
                    fit_folds=context.fit_folds,
                    fit_origins=FIT_ORIGINS,
                    response_binding_sha256=(
                        context.response_binding_sha256
                    ),
                    parent_subspace_artifact_sha256=(
                        context.parent_subspace.artifact_sha256
                    ),
                    rank=target_source_rank,
                    topology_kind=topology_kind,
                )
                add_family(
                    method,
                    result.prefix(target_source_rank),
                    source_kind,
                    receipt,
                )
                grouped[(topology_kind, group_count, core_method)] = (
                    partition,
                    result,
                )

    for topology_kind in ("signed", "magnitude"):
        partition, result = grouped[
            (topology_kind, control_group_count, "wavelet_local_svd")
        ]
        method = (
            f"{topology_kind}_local_svd_g"
            f"{control_group_count}_one_hot"
        )
        add_family(
            method,
            grouped_basis_one_hot_control(
                q,
                weighted_fit,
                partition,
                result,
                rank=target_source_rank,
            ),
            "fixed_orthonormal_control",
            {
                "construction_kind": "matched_group_allocation_one_hot_control",
                "topology_kind": topology_kind,
                "partition_artifact_sha256": partition.artifact_sha256,
                "source_grouped_basis_artifact_sha256": (
                    result.artifact_sha256
                ),
                "rank_allocation": _rank_allocation(
                    result,
                    rank=target_source_rank,
                    group_count=control_group_count,
                ),
                "local_rotations_permitted": False,
            },
        )

    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(
            parent_rank,
            generator=generator,
            dtype=torch.int64,
        )
        permuted_laplacian = signed_laplacian.index_select(
            0,
            permutation,
        ).index_select(1, permutation).contiguous()
        partition = fit_graph_wavelet_topology_partition(
            q,
            permuted_laplacian,
            group_count=control_group_count,
            topology_top_k=topology_top_k,
        )
        result = fit_graph_wavelet_grouped_basis(
            q,
            weighted_fit,
            partition,
            method="wavelet_local_svd",
            fit_origins=FIT_ORIGINS,
            response_binding_sha256=context.response_binding_sha256,
            parent_subspace_artifact_sha256=(
                context.parent_subspace.artifact_sha256
            ),
        )
        method = (
            f"permuted_signed_local_svd_g{control_group_count}_seed_{seed}"
        )
        receipt = _grouped_receipt(
            parent_basis=q,
            partition=partition,
            family=result,
            fit_folds=context.fit_folds,
            fit_origins=FIT_ORIGINS,
            response_binding_sha256=context.response_binding_sha256,
            parent_subspace_artifact_sha256=(
                context.parent_subspace.artifact_sha256
            ),
            rank=target_source_rank,
            topology_kind="permuted_signed",
        )
        receipt["permutation_seed"] = seed
        receipt["topology_permutation"] = tuple(int(v) for v in permutation)
        add_family(
            method,
            result.prefix(target_source_rank),
            "fixed_orthonormal_control",
            receipt,
        )

    if tuple(family.method for family in families) != method_order:
        raise RuntimeError("grouped comparison method order drifted")
    panel = freeze_matched_plan_panel(
        context,
        families,
        ranks=(target_source_rank,),
    )

    # This call is the first operation that validates and reads selection values.
    frozen_rows = evaluate_frozen_plan_panel(
        panel,
        responses,
        tuple(origins),
        selection_origins=SELECTION_ORIGINS,
    )
    rows = []
    for frozen in frozen_rows:
        row = dict(frozen)
        heldout = row["heldout_evaluation"]
        row["passes_fidelity_gate"] = (
            heldout["weighted_relative_error"]  # type: ignore[index]
            <= MAXIMUM_SELECTION_RELATIVE_ERROR
            and heldout["weighted_cosine"]  # type: ignore[index]
            >= MINIMUM_SELECTION_COSINE
        )
        row["basis_mixing"] = receipts[str(row["method"])][  # type: ignore[index]
            "basis_mixing"
        ]
        rows.append(row)

    by_method = {str(row["method"]): row for row in rows}
    new_methods = tuple(
        method
        for method in method_order
        if "_local_svd_" in method or "_cluster_gfa_" in method
        if "_one_hot" not in method and not method.startswith("permuted_")
    )
    best_new = min(
        new_methods,
        key=lambda method: float(
            by_method[method]["heldout_evaluation"][
                "weighted_relative_error"
            ]
        ),
    )
    signed_primary = f"signed_local_svd_g{control_group_count}"
    magnitude_primary = f"magnitude_local_svd_g{control_group_count}"
    signed_error = float(
        by_method[signed_primary]["heldout_evaluation"][
            "weighted_relative_error"
        ]
    )
    permuted_errors = tuple(
        float(
            by_method[
                f"permuted_signed_local_svd_g"
                f"{control_group_count}_seed_{seed}"
            ]["heldout_evaluation"]["weighted_relative_error"]
        )
        for seed in seeds
    )
    one_hot_error = float(
        by_method[
            f"signed_local_svd_g{control_group_count}_one_hot"
        ]["heldout_evaluation"]["weighted_relative_error"]
    )
    gfa_error = float(
        by_method["signed_gfa_fit_energy"]["heldout_evaluation"][
            "weighted_relative_error"
        ]
    )
    pair_error = float(
        by_method["graph_local_pair_supermode"]["heldout_evaluation"][
            "weighted_relative_error"
        ]
    )
    gomp_error = float(
        by_method["gomp_prefix"]["heldout_evaluation"][
            "weighted_relative_error"
        ]
    )
    global_svd_error = float(
        by_method["fit_svd"]["heldout_evaluation"][
            "weighted_relative_error"
        ]
    )
    full_accounting = panel.full_accounting
    first_accounting = panel.plans[0].accounting()
    conclusions = {
        "best_new_method": best_new,
        "best_new_selection_weighted_relative_error": float(
            by_method[best_new]["heldout_evaluation"][
                "weighted_relative_error"
            ]
        ),
        "best_new_selection_weighted_cosine": float(
            by_method[best_new]["heldout_evaluation"]["weighted_cosine"]
        ),
        "signed_grouped_primary_method": signed_primary,
        "controlled_development_nominee_method": signed_primary,
        "controlled_development_nominee_selection_weighted_relative_error": (
            signed_error
        ),
        "controlled_development_nominee_sse_recovery_vs_signed_gfa": (
            1.0 - (signed_error / gfa_error) ** 2
        ),
        "controlled_development_nominee_sse_recovery_vs_pair_supermode": (
            1.0 - (signed_error / pair_error) ** 2
        ),
        "controlled_development_nominee_sse_recovery_vs_gomp": (
            1.0 - (signed_error / gomp_error) ** 2
        ),
        "signed_grouped_primary_beats_one_hot_control": (
            signed_error < one_hot_error
        ),
        "signed_grouped_primary_permuted_control_win_count": sum(
            signed_error < error for error in permuted_errors
        ),
        "signed_grouped_primary_permuted_control_count": len(
            permuted_errors
        ),
        "magnitude_grouped_primary_error": float(
            by_method[magnitude_primary]["heldout_evaluation"][
                "weighted_relative_error"
            ]
        ),
        "signed_topology_equal_group_observed_better_than_magnitude": (
            signed_error
            < float(
                by_method[magnitude_primary]["heldout_evaluation"][
                    "weighted_relative_error"
                ]
            )
        ),
        "signed_topology_superiority_claim": False,
        "cluster_gfa_beats_local_svd_at_any_matched_setting": any(
            float(
                by_method[f"{topology}_cluster_gfa_g{count}"][
                    "heldout_evaluation"
                ]["weighted_relative_error"]
            )
            < float(
                by_method[f"{topology}_local_svd_g{count}"][
                    "heldout_evaluation"
                ]["weighted_relative_error"]
            )
            for topology in ("signed", "magnitude")
            for count in counts
        ),
        "global_svd_ceiling_still_better_than_every_localized_method": (
            global_svd_error
            < min(
                float(
                    by_method[method]["heldout_evaluation"][
                        "weighted_relative_error"
                    ]
                )
                for method in new_methods
            )
        ),
        "locality_fidelity_curve": tuple(
            {
                "topology": topology,
                "group_count": count,
                "maximum_block_size": parent_rank // count,
                "local_svd_error": float(
                    by_method[f"{topology}_local_svd_g{count}"][
                        "heldout_evaluation"
                    ]["weighted_relative_error"]
                ),
                "cluster_gfa_error": float(
                    by_method[f"{topology}_cluster_gfa_g{count}"][
                        "heldout_evaluation"
                    ]["weighted_relative_error"]
                ),
            }
            for topology in ("signed", "magnitude")
            for count in counts
        ),
        "prepared_runtime_storage_reduction_from_rank64": (
            1.0
            - first_accounting.prepared_storage_bytes
            / full_accounting.prepared_storage_bytes
        ),
        "all_methods_have_exact_equal_rank_plan_payload": True,
        "all_bases_and_plans_frozen_before_selection_read": True,
        "fit_and_selection_are_disjoint": True,
        "selection_is_open_development_not_confirmation": True,
        "compute_gate_measured": False,
        "model_compression_claim": False,
        "latency_or_speed_claim": False,
        "whole_model_replacement_claim": False,
        "natural_prompt_or_nll_fidelity_measured": False,
        "fixed_reference_structural_response_only": True,
    }
    protocol = {
        "experiment_variant": "fit_only_graph_wavelet_grouped_comparison",
        "response_component": "local_central_odd_tangent",
        "response_tensor_order": "source_origin_lag_target",
        "measured_origins": tuple(origins),
        "fit_origins": FIT_ORIGINS,
        "selection_origins": SELECTION_ORIGINS,
        "source_modes": context.source_modes,
        "target_modes": context.target_modes,
        "lag_count": context.lag_count,
        "fft_length": context.fft_length,
        "parent_rank": parent_rank,
        "target_source_rank": target_source_rank,
        "group_counts": counts,
        "control_group_count": control_group_count,
        "topology_top_k": topology_top_k,
        "permutation_seeds": seeds,
        "method_order": method_order,
        "target_rank_is_full": True,
        "all_bases_and_plans_frozen_before_selection_read": True,
        "selection_is_open_development_not_confirmation": True,
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
        "authenticated_parent_tensor_file_count": 1,
        "authenticated_parent_tensor_file_bytes": parent_tensor_file_bytes,
        "fit_response_float64_scalars": int(weighted_fit.numel()),
        "selection_response_float64_scalars": (
            context.source_modes
            * len(SELECTION_ORIGINS)
            * context.lag_count
            * context.target_modes
        ),
        "topology_partition_fit_count": 2 * len(counts) + len(seeds),
        "grouped_basis_fit_count": 4 * len(counts) + len(seeds),
        "grouped_basis_lofo_fit_count": (
            (4 * len(counts) + len(seeds)) * len(FIT_ORIGINS)
        ),
        "conditional_plan_fit_count": len(panel.plans),
        "conditional_plan_heldout_evaluation_count": len(panel.plans),
        "raw_response_tensors_serialized": False,
        "compiled_plan_tensors_serialized": False,
        "parent_basis_tensor_serialized": False,
        "partition_tensors_serialized": False,
    }
    return Gemma3GraphWaveletGroupedComparisonCandidate(
        source_receipt=context.source_receipt,
        parent_receipt=context.parent_receipt,
        protocol=protocol,
        construction_receipts=receipts,
        rate_rows=tuple(rows),
        conclusions=conclusions,
        resource_accounting=resource_accounting,
    )


def compile_gemma3_graph_wavelet_grouped_comparison_candidate(
    source: Gemma3SpectralSource,
    parent: Gemma3GraphWaveletCandidate,
    *,
    source_tensor_file_bytes: int = 0,
    parent_tensor_file_bytes: int = 0,
) -> Gemma3GraphWaveletGroupedComparisonCandidate:
    """Compile the comparison from strict-loaded prompt-free artifacts."""

    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    if not isinstance(parent, Gemma3GraphWaveletCandidate):
        raise TypeError("parent must be a Gemma3GraphWaveletCandidate")
    source.mapping.validate_integrity()
    parent.validate_integrity()
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    return _compile_from_response(
        local.impulse_responses,
        source.source_mode_standard_deviations,
        source.mapping.impulse_logical_positions,
        parent,
        response_binding_sha256=_response_binding(
            source,
            component="local_central_odd_tangent",
        ),
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
        source_tensor_file_bytes=source_tensor_file_bytes,
        parent_tensor_file_bytes=parent_tensor_file_bytes,
    )


def _publish_candidate(
    candidate: Gemma3GraphWaveletGroupedComparisonCandidate,
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


def load_gemma3_graph_wavelet_grouped_comparison_candidate(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> Gemma3GraphWaveletGroupedComparisonCandidate:
    """Strictly authenticate and restore one comparison candidate."""

    source = Path(path)
    if _file_sha256(source) != _require_sha256(
        expected_tensor_file_sha256,
        label="expected tensor file",
    ):
        raise ValueError("grouped comparison tensor file hash differs")
    with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    claimed_report = _require_sha256(
        report.get("report_sha256"),
        label="report SHA-256",
    )
    payload = dict(report)
    payload.pop("report_sha256")
    if (
        claimed_report
        != _require_sha256(
            expected_report_sha256,
            label="expected report",
        )
        or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed_report
        or report.get("artifact", {}).get("tensor_file_sha256")
        != _file_sha256(source)
    ):
        raise ValueError("grouped comparison report binding differs")
    candidate = Gemma3GraphWaveletGroupedComparisonCandidate.from_state_dict(
        torch.load(source, map_location="cpu", weights_only=True)
    )
    if (
        candidate.artifact_sha256
        != _require_sha256(
            expected_artifact_sha256,
            label="expected artifact",
        )
        or report.get("candidate", {}).get("artifact_sha256")
        != candidate.artifact_sha256
        or _canonical_json_bytes(report["candidate"])
        != _canonical_json_bytes(candidate.metadata())
    ):
        raise ValueError("grouped comparison logical artifact differs")
    return candidate


def analyze_gemma3_l3_l4_graph_wavelet_grouped_comparison(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Strict-load the pinned source and parent, then publish once."""

    destination = _validate_output_path(output, suffix=".pt")
    if destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite grouped comparison report")
    source_path = Path(source_artifact_path)
    parent_path = Path(parent_artifact_path)
    source = load_gemma3_spectral_source(
        source_path,
        expected_file_sha256=DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        parent_path,
        expected_artifact_sha256=DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = compile_gemma3_graph_wavelet_grouped_comparison_candidate(
        source,
        parent,
        source_tensor_file_bytes=source_path.stat().st_size,
        parent_tensor_file_bytes=parent_path.stat().st_size,
    )
    return _publish_candidate(candidate, output=destination)


def describe_gemma3_l3_l4_graph_wavelet_grouped_comparison() -> dict[str, object]:
    """Describe the frozen resource contract without opening artifacts."""

    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "source": {
            "artifact": str(DEFAULT_INTERIOR_ARTIFACT),
            "tensor_file_sha256": DEFAULT_INTERIOR_ARTIFACT_SHA256,
            "report_payload_sha256": DEFAULT_INTERIOR_REPORT_SHA256,
        },
        "parent": {
            "artifact": str(DEFAULT_PARENT_ARTIFACT),
            "artifact_sha256": DEFAULT_PARENT_ARTIFACT_SHA256,
            "tensor_file_sha256": DEFAULT_PARENT_TENSOR_FILE_SHA256,
            "report_payload_sha256": DEFAULT_PARENT_REPORT_SHA256,
        },
        "protocol": {
            "fit_origins": FIT_ORIGINS,
            "selection_origins": SELECTION_ORIGINS,
            "target_source_rank": TARGET_RANK,
            "group_counts": GROUP_COUNTS,
            "control_group_count": CONTROL_GROUP_COUNT,
            "method_order": _method_order(
                GROUP_COUNTS,
                control_group_count=CONTROL_GROUP_COUNT,
                permutation_seeds=PERMUTED_GRAPH_LOCAL_SEEDS,
            ),
            "selection_is_open_development_not_confirmation": True,
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
        "claims": {
            "natural_prompt_or_nll_fidelity_measured": False,
            "whole_model_replacement_claim": False,
            "model_compression_claim": False,
            "latency_or_speed_claim": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemma L3->L4 grouped graph-wavelet comparison",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_INTERIOR_ARTIFACT,
    )
    analyze.add_argument(
        "--parent-artifact",
        type=Path,
        default=DEFAULT_PARENT_ARTIFACT,
    )
    analyze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        result = describe_gemma3_l3_l4_graph_wavelet_grouped_comparison()
    else:
        result = analyze_gemma3_l3_l4_graph_wavelet_grouped_comparison(
            source_artifact_path=args.source_artifact,
            parent_artifact_path=args.parent_artifact,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
