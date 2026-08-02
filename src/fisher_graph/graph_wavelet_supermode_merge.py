"""Fit-only mixed graph-wavelet supermode merging and singleton pruning.

The compiler receives an orthonormal parent basis ``Q`` with shape ``[N, K]``,
a weighted fit response with shape ``[N, ...]``, fit-only fold responses, and
a signed graph Laplacian.  It emits a deterministic nested path from rank
``K`` down to a requested minimum rank.

Every possible two-direction pair is summarized by the leading eigenvector
of its ``2 x 2`` fit-response Gram block.  A pair is admitted as a genuine
merge only when:

* its smaller squared loading clears a preregistered mixing floor;
* its leading loading is stable under leave-one-fit-fold-out replay; and
* graph-local methods place the edge in the undirected union of per-node
  topology top-k neighborhoods.

Admitted merge actions and every one-hot singleton-prune action then compete
in one list.  The action cost is absolute lost fit-response energy:
``lambda_min`` for a merge and coordinate energy for a prune.  Actions are
sorted by ``(loss, merge-before-prune, i, j)`` and greedily accepted while
their original parent endpoints remain unused.  This is deliberately not an
all-pair matching: a cheap true merge may win, while a one-hot-like pair is
forced to compete honestly as singleton pruning.

``graph_local`` topology is

``abs(Q).T @ abs(offdiag(L)) @ abs(Q)``.

``permuted_graph_local`` is the topology-null control: it deterministically
permutes the rows of ``Q`` relative to the unchanged ``L`` before computing
that same score.  Response moments are never permuted.

The result exposes both the mixed basis and an equal-rank one-hot control in
which every selected merge is replaced by keeping its higher-response-energy
endpoint.  No held-out/evaluation argument exists anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal, Mapping, Sequence, TypeAlias

import torch
from torch import Tensor


GraphWaveletSupermodeMethod = Literal[
    "response_only",
    "graph_local",
    "permuted_graph_local",
]

__all__ = [
    "FitOnlyGraphWaveletSupermodePath",
    "GraphWaveletSingletonPrune",
    "GraphWaveletSupermodeAction",
    "GraphWaveletSupermodeMethod",
    "GraphWaveletSupermodePair",
    "fit_graph_wavelet_supermode_merge",
]


_METHODS = frozenset(
    {"response_only", "graph_local", "permuted_graph_local"}
)
_ARTIFACT_KIND = "fisher_graph.fit_only_graph_wavelet_supermode_path"
_FORMAT_VERSION = 1
_ALGORITHM = (
    "fit_only_loss_competitive_genuine_pair_merge_or_singleton_prune_v1"
)
_PAIR_SEMANTICS = (
    "canonical_leading_eigenvector_of_pair_weighted_fit_response_gram"
)
_ACTION_POLICY = (
    "eligible_merges_and_all_singleton_prunes_sorted_by_absolute_loss_"
    "merge_before_prune_then_parent_indices_endpoint_disjoint_greedy"
)
_TOPOLOGY_SEMANTICS = (
    "undirected_union_top_k_of_abs_q_transpose_abs_offdiag_laplacian_abs_q"
)
_LOFO_SEMANTICS = (
    "minimum_absolute_cosine_full_loading_vs_each_leave_one_fit_fold_out_"
    "loading"
)
_ONE_HOT_CONTROL_SEMANTICS = (
    "same_selected_actions_pair_merge_becomes_keep_higher_coordinate_"
    "response_energy_stable_parent_index_tie"
)
_FIT_SCOPE = "caller_supplied_weighted_fit_response_and_fit_folds_only"
_ARTIFACT_DOMAIN = (
    b"fisher_graph.fit_only_graph_wavelet_supermode_path.v1\0"
)
_TENSOR_DOMAIN = (
    b"fisher_graph.fit_only_graph_wavelet_supermode_path.tensor.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ORTHOGONALITY_TOLERANCE = 2.0e-10


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        _ARTIFACT_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            list(tensor.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _float64_tensor(
    value: object,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result.clone()


def _int64_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != 1:
        raise ValueError(f"{label} must have one dimension")
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{label} must use an integer dtype")
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _unit_interval(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not -1.0e-12 <= float(value) <= 1.0 + 1.0e-12
    ):
        raise ValueError(f"{label} must lie in [0, 1]")
    return min(max(float(value), 0.0), 1.0)


def _nonnegative_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _symmetric(
    value: Tensor,
    *,
    label: str,
    require_psd: bool,
) -> Tensor:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{label} must be square")
    scale = max(float(value.abs().max()), 1.0)
    tolerance = (
        torch.finfo(torch.float64).eps
        * scale
        * max(1, value.shape[0])
        * 1024.0
    )
    if not torch.allclose(value, value.T, atol=tolerance, rtol=0.0):
        raise ValueError(f"{label} must be symmetric")
    result = ((value + value.T) * 0.5).contiguous()
    if require_psd and float(torch.linalg.eigvalsh(result).min()) < -tolerance:
        raise ValueError(f"{label} must be positive semidefinite")
    return result


def _canonicalize_columns(value: Tensor) -> Tensor:
    result = value.clone()
    for column in range(result.shape[1]):
        pivot = int(torch.argmax(result[:, column].abs()))
        if float(result[pivot, column]) < 0.0:
            result[:, column].neg_()
    return result.contiguous()


def _orthonormal(value: Tensor) -> bool:
    return bool(
        torch.allclose(
            value.T @ value,
            torch.eye(value.shape[1], dtype=torch.float64),
            atol=_ORTHOGONALITY_TOLERANCE,
            rtol=_ORTHOGONALITY_TOLERANCE,
        )
    )


def _canonical_top_mode(block: Tensor) -> tuple[Tensor, float, float]:
    """Return canonical top loading and ascending nonnegative eigenvalues."""

    eigenvalues, eigenvectors = torch.linalg.eigh(
        ((block + block.T) * 0.5).contiguous()
    )
    scale = max(float(eigenvalues.abs().max()), 1.0)
    tolerance = torch.finfo(torch.float64).eps * scale * 256.0
    lower = max(float(eigenvalues[0]), 0.0)
    upper = max(float(eigenvalues[1]), 0.0)
    if upper - lower <= tolerance:
        loading = torch.tensor([1.0, 0.0], dtype=torch.float64)
    else:
        loading = eigenvectors[:, 1].clone()
        loading /= torch.linalg.vector_norm(loading)
        pivot = int(torch.argmax(loading.abs()))
        if float(loading[pivot]) < 0.0:
            loading.neg_()
    return loading.contiguous(), lower, upper


def _leading_direction_identifiable(
    lower: float,
    upper: float,
    *,
    reference_energy: float,
) -> bool:
    """Return whether a leading eigendirection carries resolvable energy."""

    scale = max(abs(lower), abs(upper), abs(reference_energy), 1.0)
    tolerance = torch.finfo(torch.float64).eps * scale * 256.0
    return upper > tolerance and upper - lower > tolerance


def _pair_block(matrix: Tensor, first: int, second: int) -> Tensor:
    indices = torch.tensor((first, second), dtype=torch.int64)
    return matrix.index_select(0, indices).index_select(1, indices)


def _topology_matrix(parent_basis: Tensor, laplacian: Tensor) -> Tensor:
    off_diagonal = laplacian.clone()
    off_diagonal.fill_diagonal_(0.0)
    absolute_adjacency = off_diagonal.abs()
    absolute_basis = parent_basis.abs()
    result = (
        absolute_basis.T @ absolute_adjacency @ absolute_basis
    ).contiguous()
    return _symmetric(
        result,
        label="topology interaction matrix",
        require_psd=False,
    )


def _top_k_neighbors(matrix: Tensor, top_k: int) -> tuple[frozenset[int], ...]:
    width = int(matrix.shape[0])
    retained = min(top_k, width - 1)
    return tuple(
        frozenset(
            sorted(
                (other for other in range(width) if other != index),
                key=lambda other: (-float(matrix[index, other]), other),
            )[:retained]
        )
        for index in range(width)
    )


@dataclass(frozen=True, slots=True)
class _PairCandidate:
    first: int
    second: int
    loadings: tuple[float, float]
    action_loss: float
    pair_response_energy: float
    retained_response_energy: float
    relative_response_loss: float
    minimum_squared_loading: float
    mixing_participation: float
    lofo_loading_stability: float
    topology_interaction: float
    native_topology_interaction: float
    topology_union_top_k_eligible: bool
    one_hot_retained_parent_index: int
    one_hot_removed_parent_index: int

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.first, self.second


@dataclass(frozen=True, slots=True)
class GraphWaveletSupermodePair:
    """One accepted genuine 2->1 merge action."""

    action_order: int
    rank_after_action: int
    first_parent_index: int
    second_parent_index: int
    loadings: tuple[float, float]
    one_hot_retained_parent_index: int
    one_hot_removed_parent_index: int
    action_loss: float
    pair_response_energy: float
    retained_response_energy: float
    relative_response_loss: float
    minimum_squared_loading: float
    mixing_participation: float
    lofo_loading_stability: float
    topology_interaction: float
    native_topology_interaction: float
    topology_union_top_k_eligible: bool
    action_kind: str = "merge"

    def __post_init__(self) -> None:
        _positive_int(self.action_order, label="action_order")
        _positive_int(self.rank_after_action, label="rank_after_action")
        first = _nonnegative_int(
            self.first_parent_index,
            label="first_parent_index",
        )
        second = _nonnegative_int(
            self.second_parent_index,
            label="second_parent_index",
        )
        if first >= second:
            raise ValueError("merge endpoints must be strictly ascending")
        if (
            type(self.loadings) is not tuple
            or len(self.loadings) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.loadings
            )
        ):
            raise ValueError("merge loadings must be two finite floats")
        loadings = tuple(float(value) for value in self.loadings)
        if not math.isclose(
            math.fsum(value * value for value in loadings),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ValueError("merge loadings must have unit norm")
        pivot = max(range(2), key=lambda index: abs(loadings[index]))
        if loadings[pivot] < 0.0:
            raise ValueError("merge loading sign is not canonical")
        retained_index = _nonnegative_int(
            self.one_hot_retained_parent_index,
            label="one_hot_retained_parent_index",
        )
        removed_index = _nonnegative_int(
            self.one_hot_removed_parent_index,
            label="one_hot_removed_parent_index",
        )
        if {retained_index, removed_index} != {first, second}:
            raise ValueError("one-hot control endpoints differ from merge")
        loss = _nonnegative_float(self.action_loss, label="action_loss")
        total = _nonnegative_float(
            self.pair_response_energy,
            label="pair_response_energy",
        )
        retained = _nonnegative_float(
            self.retained_response_energy,
            label="retained_response_energy",
        )
        relative = _unit_interval(
            self.relative_response_loss,
            label="relative_response_loss",
        )
        minimum_loading = _unit_interval(
            self.minimum_squared_loading,
            label="minimum_squared_loading",
        )
        participation = _unit_interval(
            self.mixing_participation,
            label="mixing_participation",
        )
        stability = _unit_interval(
            self.lofo_loading_stability,
            label="lofo_loading_stability",
        )
        topology = _nonnegative_float(
            self.topology_interaction,
            label="topology_interaction",
        )
        native_topology = _nonnegative_float(
            self.native_topology_interaction,
            label="native_topology_interaction",
        )
        if type(self.topology_union_top_k_eligible) is not bool:
            raise TypeError("topology eligibility must be boolean")
        if self.action_kind != "merge":
            raise ValueError("merge action_kind differs")
        scale = max(total, retained, loss, 1.0)
        if (
            not math.isclose(
                retained + loss,
                total,
                rel_tol=0.0,
                abs_tol=2.0e-10 * scale,
            )
            or not math.isclose(
                relative,
                loss / total
                if total > torch.finfo(torch.float64).tiny
                else 0.0,
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
        ):
            raise ValueError("merge response accounting differs")
        squared = tuple(value * value for value in loadings)
        if (
            not math.isclose(
                minimum_loading,
                min(squared),
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
            or not math.isclose(
                participation,
                4.0 * squared[0] * squared[1],
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
        ):
            raise ValueError("merge loading diagnostics differ")
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "action_loss", loss)
        object.__setattr__(self, "pair_response_energy", total)
        object.__setattr__(self, "retained_response_energy", retained)
        object.__setattr__(self, "relative_response_loss", relative)
        object.__setattr__(self, "minimum_squared_loading", minimum_loading)
        object.__setattr__(self, "mixing_participation", participation)
        object.__setattr__(self, "lofo_loading_stability", stability)
        object.__setattr__(self, "topology_interaction", topology)
        object.__setattr__(
            self,
            "native_topology_interaction",
            native_topology,
        )

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.first_parent_index, self.second_parent_index

    def metadata(self) -> dict[str, object]:
        return {
            "action_kind": self.action_kind,
            "action_order": self.action_order,
            "rank_after_action": self.rank_after_action,
            "first_parent_index": self.first_parent_index,
            "second_parent_index": self.second_parent_index,
            "loadings": self.loadings,
            "one_hot_retained_parent_index": (
                self.one_hot_retained_parent_index
            ),
            "one_hot_removed_parent_index": (
                self.one_hot_removed_parent_index
            ),
            "action_loss": self.action_loss,
            "pair_response_energy": self.pair_response_energy,
            "retained_response_energy": self.retained_response_energy,
            "relative_response_loss": self.relative_response_loss,
            "minimum_squared_loading": self.minimum_squared_loading,
            "mixing_participation": self.mixing_participation,
            "lofo_loading_stability": self.lofo_loading_stability,
            "topology_interaction": self.topology_interaction,
            "native_topology_interaction": (
                self.native_topology_interaction
            ),
            "topology_union_top_k_eligible": (
                self.topology_union_top_k_eligible
            ),
        }


@dataclass(frozen=True, slots=True)
class GraphWaveletSingletonPrune:
    """One accepted one-hot parent-coordinate deletion action."""

    action_order: int
    rank_after_action: int
    parent_index: int
    action_loss: float
    action_kind: str = "singleton_prune"

    def __post_init__(self) -> None:
        _positive_int(self.action_order, label="action_order")
        _positive_int(self.rank_after_action, label="rank_after_action")
        _nonnegative_int(self.parent_index, label="parent_index")
        object.__setattr__(
            self,
            "action_loss",
            _nonnegative_float(self.action_loss, label="action_loss"),
        )
        if self.action_kind != "singleton_prune":
            raise ValueError("singleton prune action_kind differs")

    @property
    def endpoints(self) -> tuple[int]:
        return (self.parent_index,)

    def metadata(self) -> dict[str, object]:
        return {
            "action_kind": self.action_kind,
            "action_order": self.action_order,
            "rank_after_action": self.rank_after_action,
            "parent_index": self.parent_index,
            "action_loss": self.action_loss,
        }


GraphWaveletSupermodeAction: TypeAlias = (
    GraphWaveletSupermodePair | GraphWaveletSingletonPrune
)


def _pair_candidate(
    *,
    response_gram: Tensor,
    fold_response_grams: Sequence[Tensor],
    native_topology: Tensor,
    selection_topology: Tensor,
    topology_neighbors: Sequence[frozenset[int]],
    first: int,
    second: int,
) -> _PairCandidate:
    loading, lower, upper = _canonical_top_mode(
        _pair_block(response_gram, first, second)
    )
    fold_stabilities = []
    for heldout_gram in fold_response_grams:
        training_block = _pair_block(
            response_gram - heldout_gram,
            first,
            second,
        )
        training_loading, training_lower, training_upper = (
            _canonical_top_mode(training_block)
        )
        if not _leading_direction_identifiable(
            training_lower,
            training_upper,
            reference_energy=upper,
        ):
            fold_stabilities.append(0.0)
        else:
            fold_stabilities.append(
                abs(float(torch.dot(loading, training_loading)))
            )
    stability = min(fold_stabilities)
    squared = loading.square()
    total = lower + upper
    first_energy = float(response_gram[first, first])
    second_energy = float(response_gram[second, second])
    if second_energy > first_energy:
        one_hot_retained = second
        one_hot_removed = first
    else:
        one_hot_retained = first
        one_hot_removed = second
    return _PairCandidate(
        first=first,
        second=second,
        loadings=(float(loading[0]), float(loading[1])),
        action_loss=lower,
        pair_response_energy=total,
        retained_response_energy=upper,
        relative_response_loss=(
            lower / total
            if total > torch.finfo(torch.float64).tiny
            else 0.0
        ),
        minimum_squared_loading=float(squared.min()),
        mixing_participation=float(4.0 * squared[0] * squared[1]),
        lofo_loading_stability=stability,
        topology_interaction=float(selection_topology[first, second]),
        native_topology_interaction=float(native_topology[first, second]),
        topology_union_top_k_eligible=(
            second in topology_neighbors[first]
            or first in topology_neighbors[second]
        ),
        one_hot_retained_parent_index=one_hot_retained,
        one_hot_removed_parent_index=one_hot_removed,
    )


def _select_actions(
    *,
    parent_rank: int,
    minimum_rank: int,
    method: GraphWaveletSupermodeMethod,
    response_gram: Tensor,
    fold_response_grams: Sequence[Tensor],
    native_topology: Tensor,
    selection_topology: Tensor,
    minimum_squared_loading: float,
    minimum_lofo_loading_stability: float,
    topology_top_k: int,
) -> tuple[GraphWaveletSupermodeAction, ...]:
    neighbors = _top_k_neighbors(selection_topology, topology_top_k)
    pairs = []
    for first in range(parent_rank):
        for second in range(first + 1, parent_rank):
            candidate = _pair_candidate(
                response_gram=response_gram,
                fold_response_grams=fold_response_grams,
                native_topology=native_topology,
                selection_topology=selection_topology,
                topology_neighbors=neighbors,
                first=first,
                second=second,
            )
            if (
                candidate.minimum_squared_loading
                < minimum_squared_loading
                or candidate.lofo_loading_stability
                < minimum_lofo_loading_stability
                or (
                    method != "response_only"
                    and (
                        not candidate.topology_union_top_k_eligible
                        or candidate.topology_interaction <= 0.0
                    )
                )
            ):
                continue
            pairs.append(candidate)
    candidates: list[
        tuple[
            tuple[float, int, int, int],
            _PairCandidate | int,
        ]
    ] = [
        (
            (
                candidate.action_loss,
                0,
                candidate.first,
                candidate.second,
            ),
            candidate,
        )
        for candidate in pairs
    ]
    candidates.extend(
        (
            (
                float(response_gram[index, index]),
                1,
                index,
                index,
            ),
            index,
        )
        for index in range(parent_rank)
    )
    candidates.sort(key=lambda item: item[0])
    reductions = parent_rank - minimum_rank
    used: set[int] = set()
    selected: list[GraphWaveletSupermodeAction] = []
    selected_pair_count = 0
    for _key, candidate in candidates:
        if (
            isinstance(candidate, _PairCandidate)
            and selected_pair_count >= minimum_rank
        ):
            # A pair consumes two original endpoints but removes only one
            # rank.  More than ``minimum_rank`` accepted pairs would leave
            # too few untouched singleton endpoints to finish the path.
            continue
        endpoints = (
            candidate.endpoints
            if isinstance(candidate, _PairCandidate)
            else (candidate,)
        )
        if any(endpoint in used for endpoint in endpoints):
            continue
        order = len(selected) + 1
        rank_after = parent_rank - order
        if isinstance(candidate, _PairCandidate):
            action: GraphWaveletSupermodeAction = (
                GraphWaveletSupermodePair(
                    action_order=order,
                    rank_after_action=rank_after,
                    first_parent_index=candidate.first,
                    second_parent_index=candidate.second,
                    loadings=candidate.loadings,
                    one_hot_retained_parent_index=(
                        candidate.one_hot_retained_parent_index
                    ),
                    one_hot_removed_parent_index=(
                        candidate.one_hot_removed_parent_index
                    ),
                    action_loss=candidate.action_loss,
                    pair_response_energy=(
                        candidate.pair_response_energy
                    ),
                    retained_response_energy=(
                        candidate.retained_response_energy
                    ),
                    relative_response_loss=(
                        candidate.relative_response_loss
                    ),
                    minimum_squared_loading=(
                        candidate.minimum_squared_loading
                    ),
                    mixing_participation=(
                        candidate.mixing_participation
                    ),
                    lofo_loading_stability=(
                        candidate.lofo_loading_stability
                    ),
                    topology_interaction=(
                        candidate.topology_interaction
                    ),
                    native_topology_interaction=(
                        candidate.native_topology_interaction
                    ),
                    topology_union_top_k_eligible=(
                        candidate.topology_union_top_k_eligible
                    ),
                )
            )
        else:
            action = GraphWaveletSingletonPrune(
                action_order=order,
                rank_after_action=rank_after,
                parent_index=candidate,
                action_loss=float(response_gram[candidate, candidate]),
            )
        selected.append(action)
        if isinstance(candidate, _PairCandidate):
            selected_pair_count += 1
        used.update(endpoints)
        if len(selected) == reductions:
            break
    if len(selected) != reductions:
        raise RuntimeError("mixed action selector could not reach minimum_rank")
    return tuple(selected)


def _basis_from_actions(
    parent_basis: Tensor,
    actions: Sequence[GraphWaveletSupermodeAction],
    *,
    one_hot_control: bool,
) -> Tensor:
    merge_by_first = {
        action.first_parent_index: action
        for action in actions
        if isinstance(action, GraphWaveletSupermodePair)
    }
    merged_seconds = {
        action.second_parent_index
        for action in actions
        if isinstance(action, GraphWaveletSupermodePair)
    }
    removed = {
        action.parent_index
        for action in actions
        if isinstance(action, GraphWaveletSingletonPrune)
    }
    if one_hot_control:
        removed.update(
            action.one_hot_removed_parent_index
            for action in actions
            if isinstance(action, GraphWaveletSupermodePair)
        )
    columns = []
    for index in range(parent_basis.shape[1]):
        if index in removed:
            continue
        if one_hot_control:
            columns.append(parent_basis[:, index].clone())
            continue
        merge = merge_by_first.get(index)
        if merge is not None:
            columns.append(
                (
                    merge.loadings[0]
                    * parent_basis[:, merge.first_parent_index]
                    + merge.loadings[1]
                    * parent_basis[:, merge.second_parent_index]
                ).contiguous()
            )
        elif index not in merged_seconds:
            columns.append(parent_basis[:, index].clone())
    if not columns:
        raise RuntimeError("action path produced an empty basis")
    return torch.stack(columns, dim=1).contiguous()


def _action_from_metadata(value: Mapping[str, object]) -> GraphWaveletSupermodeAction:
    kind = value.get("action_kind")
    if kind == "singleton_prune":
        expected = {
            "action_kind",
            "action_order",
            "rank_after_action",
            "parent_index",
            "action_loss",
        }
        if set(value) != expected:
            raise ValueError("singleton action metadata fields are invalid")
        if any(
            type(value[field]) is not int
            for field in (
                "action_order",
                "rank_after_action",
                "parent_index",
            )
        ):
            raise ValueError("singleton action integer fields are invalid")
        return GraphWaveletSingletonPrune(
            action_order=value["action_order"],  # type: ignore[arg-type]
            rank_after_action=value["rank_after_action"],  # type: ignore[arg-type]
            parent_index=value["parent_index"],  # type: ignore[arg-type]
            action_loss=float(value["action_loss"]),
        )
    if kind != "merge":
        raise ValueError("action metadata kind is invalid")
    expected = {
        "action_kind",
        "action_order",
        "rank_after_action",
        "first_parent_index",
        "second_parent_index",
        "loadings",
        "one_hot_retained_parent_index",
        "one_hot_removed_parent_index",
        "action_loss",
        "pair_response_energy",
        "retained_response_energy",
        "relative_response_loss",
        "minimum_squared_loading",
        "mixing_participation",
        "lofo_loading_stability",
        "topology_interaction",
        "native_topology_interaction",
        "topology_union_top_k_eligible",
    }
    if set(value) != expected:
        raise ValueError("merge action metadata fields are invalid")
    if any(
        type(value[field]) is not int
        for field in (
            "action_order",
            "rank_after_action",
            "first_parent_index",
            "second_parent_index",
            "one_hot_retained_parent_index",
            "one_hot_removed_parent_index",
        )
    ):
        raise ValueError("merge action integer fields are invalid")
    if type(value["topology_union_top_k_eligible"]) is not bool:
        raise ValueError("merge action topology eligibility is invalid")
    raw_loadings = value["loadings"]
    if not isinstance(raw_loadings, (tuple, list)):
        raise TypeError("merge loadings metadata must be a sequence")
    return GraphWaveletSupermodePair(
        action_order=value["action_order"],  # type: ignore[arg-type]
        rank_after_action=value["rank_after_action"],  # type: ignore[arg-type]
        first_parent_index=value["first_parent_index"],  # type: ignore[arg-type]
        second_parent_index=value["second_parent_index"],  # type: ignore[arg-type]
        loadings=tuple(float(item) for item in raw_loadings),  # type: ignore[arg-type]
        one_hot_retained_parent_index=value[  # type: ignore[arg-type]
            "one_hot_retained_parent_index"
        ],
        one_hot_removed_parent_index=value[  # type: ignore[arg-type]
            "one_hot_removed_parent_index"
        ],
        action_loss=float(value["action_loss"]),
        pair_response_energy=float(value["pair_response_energy"]),
        retained_response_energy=float(value["retained_response_energy"]),
        relative_response_loss=float(value["relative_response_loss"]),
        minimum_squared_loading=float(value["minimum_squared_loading"]),
        mixing_participation=float(value["mixing_participation"]),
        lofo_loading_stability=float(value["lofo_loading_stability"]),
        topology_interaction=float(value["topology_interaction"]),
        native_topology_interaction=float(
            value["native_topology_interaction"]
        ),
        topology_union_top_k_eligible=value[  # type: ignore[arg-type]
            "topology_union_top_k_eligible"
        ],
    )


@dataclass(frozen=True, slots=True)
class FitOnlyGraphWaveletSupermodePath:
    """Authenticated nested mixed merge/prune path."""

    method: GraphWaveletSupermodeMethod
    minimum_rank: int
    minimum_squared_loading: float
    minimum_lofo_loading_stability: float
    topology_top_k: int
    permutation_seed: int
    fit_response_shape: tuple[int, ...]
    fit_response_sha256: str
    fit_fold_response_shapes: tuple[tuple[int, ...], ...]
    fit_fold_response_sha256s: tuple[str, ...]
    signed_laplacian_sha256: str
    parent_basis: Tensor
    response_gram: Tensor
    fit_fold_response_grams: tuple[Tensor, ...]
    native_topology_matrix: Tensor
    selection_topology_matrix: Tensor
    topology_permutation: Tensor
    selected_actions: tuple[GraphWaveletSupermodeAction, ...]
    heldout_input_used: bool = False
    algorithm: str = _ALGORITHM
    pair_semantics: str = _PAIR_SEMANTICS
    action_policy: str = _ACTION_POLICY
    topology_semantics: str = _TOPOLOGY_SEMANTICS
    lofo_semantics: str = _LOFO_SEMANTICS
    one_hot_control_semantics: str = _ONE_HOT_CONTROL_SEMANTICS
    fit_scope: str = _FIT_SCOPE
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError("method is invalid")
        minimum_rank = _positive_int(
            self.minimum_rank,
            label="minimum_rank",
        )
        minimum_loading = _unit_interval(
            self.minimum_squared_loading,
            label="minimum_squared_loading",
        )
        if minimum_loading > 0.5:
            raise ValueError("minimum_squared_loading cannot exceed 0.5")
        minimum_stability = _unit_interval(
            self.minimum_lofo_loading_stability,
            label="minimum_lofo_loading_stability",
        )
        topology_top_k = _positive_int(
            self.topology_top_k,
            label="topology_top_k",
        )
        permutation_seed = _nonnegative_int(
            self.permutation_seed,
            label="permutation_seed",
        )
        if self.method != "permuted_graph_local" and permutation_seed != 0:
            raise ValueError(
                "only permuted_graph_local may use a nonzero permutation seed"
            )
        parent = _float64_tensor(
            self.parent_basis,
            label="parent_basis",
            ndim=2,
        )
        node_count, parent_rank = parent.shape
        if (
            node_count <= 0
            or parent_rank <= 1
            or parent_rank > node_count
            or not _orthonormal(parent)
            or not torch.equal(parent, _canonicalize_columns(parent))
        ):
            raise ValueError(
                "parent_basis must be canonical, nonempty, and orthonormal"
            )
        if not 1 <= minimum_rank < parent_rank:
            raise ValueError("minimum_rank must lie in [1, parent_rank)")
        if not 1 <= topology_top_k <= parent_rank - 1:
            raise ValueError("topology_top_k must be below parent rank")
        fit_shape = tuple(self.fit_response_shape)
        fold_shapes = tuple(tuple(shape) for shape in self.fit_fold_response_shapes)
        fold_hashes = tuple(self.fit_fold_response_sha256s)
        if (
            len(fit_shape) < 2
            or fit_shape[0] != node_count
            or any(type(value) is not int or value <= 0 for value in fit_shape)
            or len(fold_shapes) < 2
            or len(fold_shapes) != len(fold_hashes)
            or any(
                len(shape) < 2
                or shape[0] != node_count
                or any(type(value) is not int or value <= 0 for value in shape)
                for shape in fold_shapes
            )
        ):
            raise ValueError("fit response or fold geometry is invalid")
        _require_sha256(
            self.fit_response_sha256,
            label="fit_response_sha256",
        )
        for value in fold_hashes:
            _require_sha256(value, label="fit fold response SHA-256")
        _require_sha256(
            self.signed_laplacian_sha256,
            label="signed_laplacian_sha256",
        )
        response_gram = _symmetric(
            _float64_tensor(
                self.response_gram,
                label="response_gram",
                ndim=2,
            ),
            label="response_gram",
            require_psd=True,
        )
        fold_grams = tuple(
            _symmetric(
                _float64_tensor(
                    value,
                    label="fit_fold_response_gram",
                    ndim=2,
                ),
                label="fit_fold_response_gram",
                require_psd=True,
            )
            for value in self.fit_fold_response_grams
        )
        native_topology = _symmetric(
            _float64_tensor(
                self.native_topology_matrix,
                label="native_topology_matrix",
                ndim=2,
            ),
            label="native_topology_matrix",
            require_psd=False,
        )
        selection_topology = _symmetric(
            _float64_tensor(
                self.selection_topology_matrix,
                label="selection_topology_matrix",
                ndim=2,
            ),
            label="selection_topology_matrix",
            require_psd=False,
        )
        expected_shape = (parent_rank, parent_rank)
        if (
            response_gram.shape != expected_shape
            or len(fold_grams) != len(fold_shapes)
            or any(value.shape != expected_shape for value in fold_grams)
            or native_topology.shape != expected_shape
            or selection_topology.shape != expected_shape
            or float(torch.trace(response_gram))
            <= torch.finfo(torch.float64).tiny
        ):
            raise ValueError("fit moment geometry or energy is invalid")
        fold_sum = torch.stack(fold_grams).sum(dim=0)
        scale = max(float(response_gram.abs().max()), 1.0)
        if not torch.allclose(
            fold_sum,
            response_gram,
            atol=2.0e-10 * scale,
            rtol=2.0e-10,
        ):
            raise ValueError("fit fold response Grams do not partition response")
        permutation = _int64_tensor(
            self.topology_permutation,
            label="topology_permutation",
        )
        if (
            permutation.shape != (node_count,)
            or not torch.equal(
                torch.sort(permutation).values,
                torch.arange(node_count, dtype=torch.int64),
            )
        ):
            raise ValueError("topology_permutation is invalid")
        if (
            self.method != "permuted_graph_local"
            and not torch.equal(
                permutation,
                torch.arange(node_count, dtype=torch.int64),
            )
        ):
            raise ValueError("non-null methods require identity topology")
        if self.method != "permuted_graph_local":
            if not torch.equal(selection_topology, native_topology):
                raise ValueError(
                    "non-null selection topology must equal native topology"
                )
        else:
            expected_permutation = torch.randperm(
                node_count,
                generator=torch.Generator(device="cpu").manual_seed(
                    permutation_seed
                ),
                dtype=torch.int64,
            )
            if not torch.equal(permutation, expected_permutation):
                raise ValueError(
                    "permuted topology does not replay permutation_seed"
                )
        actions = tuple(self.selected_actions)
        if (
            len(actions) != parent_rank - minimum_rank
            or any(
                not isinstance(
                    action,
                    (GraphWaveletSupermodePair, GraphWaveletSingletonPrune),
                )
                for action in actions
            )
        ):
            raise ValueError("selected action path is invalid")
        expected_actions = _select_actions(
            parent_rank=parent_rank,
            minimum_rank=minimum_rank,
            method=self.method,
            response_gram=response_gram,
            fold_response_grams=fold_grams,
            native_topology=native_topology,
            selection_topology=selection_topology,
            minimum_squared_loading=minimum_loading,
            minimum_lofo_loading_stability=minimum_stability,
            topology_top_k=topology_top_k,
        )
        if actions != expected_actions:
            raise ValueError("selected mixed action path is not canonical")
        if (
            self.heldout_input_used is not False
            or self.algorithm != _ALGORITHM
            or self.pair_semantics != _PAIR_SEMANTICS
            or self.action_policy != _ACTION_POLICY
            or self.topology_semantics != _TOPOLOGY_SEMANTICS
            or self.lofo_semantics != _LOFO_SEMANTICS
            or self.one_hot_control_semantics
            != _ONE_HOT_CONTROL_SEMANTICS
            or self.fit_scope != _FIT_SCOPE
            or self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("supermode path provenance differs")
        object.__setattr__(self, "minimum_rank", minimum_rank)
        object.__setattr__(
            self,
            "minimum_squared_loading",
            minimum_loading,
        )
        object.__setattr__(
            self,
            "minimum_lofo_loading_stability",
            minimum_stability,
        )
        object.__setattr__(self, "topology_top_k", topology_top_k)
        object.__setattr__(self, "permutation_seed", permutation_seed)
        object.__setattr__(self, "fit_response_shape", fit_shape)
        object.__setattr__(self, "fit_fold_response_shapes", fold_shapes)
        object.__setattr__(
            self,
            "fit_fold_response_sha256s",
            fold_hashes,
        )
        object.__setattr__(self, "parent_basis", parent)
        object.__setattr__(self, "response_gram", response_gram)
        object.__setattr__(
            self,
            "fit_fold_response_grams",
            fold_grams,
        )
        object.__setattr__(
            self,
            "native_topology_matrix",
            native_topology,
        )
        object.__setattr__(
            self,
            "selection_topology_matrix",
            selection_topology,
        )
        object.__setattr__(self, "topology_permutation", permutation)
        object.__setattr__(self, "selected_actions", actions)
        self._validate_bases()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("supermode path artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def node_count(self) -> int:
        return int(self.parent_basis.shape[0])

    @property
    def parent_rank(self) -> int:
        return int(self.parent_basis.shape[1])

    @property
    def available_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.parent_rank, self.minimum_rank - 1, -1))

    @property
    def selected_pairs(self) -> tuple[GraphWaveletSupermodePair, ...]:
        return tuple(
            action
            for action in self.selected_actions
            if isinstance(action, GraphWaveletSupermodePair)
        )

    @property
    def singleton_prunes(self) -> tuple[GraphWaveletSingletonPrune, ...]:
        return tuple(
            action
            for action in self.selected_actions
            if isinstance(action, GraphWaveletSingletonPrune)
        )

    def _actions_for_rank(
        self,
        rank: int,
    ) -> tuple[GraphWaveletSupermodeAction, ...]:
        if (
            type(rank) is not int
            or not self.minimum_rank <= rank <= self.parent_rank
        ):
            raise ValueError(
                "rank must lie between minimum_rank and parent_rank"
            )
        return self.selected_actions[: self.parent_rank - rank]

    def _basis_without_validation(
        self,
        rank: int,
        *,
        one_hot_control: bool,
    ) -> Tensor:
        return _basis_from_actions(
            self.parent_basis,
            self._actions_for_rank(rank),
            one_hot_control=one_hot_control,
        )

    def basis(self, rank: int) -> Tensor:
        """Return the mixed merge/prune basis at one nested rank."""

        self.validate_integrity()
        return self._basis_without_validation(
            rank,
            one_hot_control=False,
        ).clone()

    def one_hot_control_basis(self, rank: int) -> Tensor:
        """Return the equal-rank one-hot/delete control for the same actions."""

        self.validate_integrity()
        return self._basis_without_validation(
            rank,
            one_hot_control=True,
        ).clone()

    def paired_delete_control_basis(self, rank: int) -> Tensor:
        """Compatibility spelling for :meth:`one_hot_control_basis`."""

        return self.one_hot_control_basis(rank)

    def action_prefix(
        self,
        rank: int,
    ) -> tuple[GraphWaveletSupermodeAction, ...]:
        self.validate_integrity()
        return self._actions_for_rank(rank)

    def _validate_bases(self) -> None:
        previous_mixed = self.parent_basis
        previous_control = self.parent_basis
        for rank in range(self.parent_rank - 1, self.minimum_rank - 1, -1):
            mixed = self._basis_without_validation(
                rank,
                one_hot_control=False,
            )
            control = self._basis_without_validation(
                rank,
                one_hot_control=True,
            )
            if (
                mixed.shape != (self.node_count, rank)
                or control.shape != (self.node_count, rank)
                or not _orthonormal(mixed)
                or not _orthonormal(control)
            ):
                raise ValueError("mixed path basis geometry differs")
            mixed_residual = mixed - previous_mixed @ (
                previous_mixed.T @ mixed
            )
            control_residual = control - previous_control @ (
                previous_control.T @ control
            )
            if (
                float(torch.linalg.vector_norm(mixed_residual))
                > _ORTHOGONALITY_TOLERANCE
                or float(torch.linalg.vector_norm(control_residual))
                > _ORTHOGONALITY_TOLERANCE
            ):
                raise ValueError("mixed or one-hot control path is not nested")
            previous_mixed = mixed
            previous_control = control

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "algorithm": self.algorithm,
            "method": self.method,
            "minimum_rank": self.minimum_rank,
            "parent_rank": self.parent_rank,
            "node_count": self.node_count,
            "minimum_squared_loading": self.minimum_squared_loading,
            "minimum_lofo_loading_stability": (
                self.minimum_lofo_loading_stability
            ),
            "topology_top_k": self.topology_top_k,
            "permutation_seed": self.permutation_seed,
            "fit_response_shape": self.fit_response_shape,
            "fit_response_sha256": self.fit_response_sha256,
            "fit_fold_response_shapes": self.fit_fold_response_shapes,
            "fit_fold_response_sha256s": self.fit_fold_response_sha256s,
            "signed_laplacian_sha256": self.signed_laplacian_sha256,
            "parent_basis_sha256": _tensor_sha256(self.parent_basis),
            "response_gram_sha256": _tensor_sha256(self.response_gram),
            "fit_fold_response_gram_sha256s": tuple(
                _tensor_sha256(value)
                for value in self.fit_fold_response_grams
            ),
            "native_topology_matrix_sha256": _tensor_sha256(
                self.native_topology_matrix
            ),
            "selection_topology_matrix_sha256": _tensor_sha256(
                self.selection_topology_matrix
            ),
            "topology_permutation_sha256": _tensor_sha256(
                self.topology_permutation
            ),
            "selected_actions": tuple(
                action.metadata() for action in self.selected_actions
            ),
            "heldout_input_used": self.heldout_input_used,
            "pair_semantics": self.pair_semantics,
            "action_policy": self.action_policy,
            "topology_semantics": self.topology_semantics,
            "lofo_semantics": self.lofo_semantics,
            "one_hot_control_semantics": self.one_hot_control_semantics,
            "fit_scope": self.fit_scope,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload())

    def validate_integrity(self) -> None:
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("supermode path artifact hash mismatch")
        expected = _select_actions(
            parent_rank=self.parent_rank,
            minimum_rank=self.minimum_rank,
            method=self.method,
            response_gram=self.response_gram,
            fold_response_grams=self.fit_fold_response_grams,
            native_topology=self.native_topology_matrix,
            selection_topology=self.selection_topology_matrix,
            minimum_squared_loading=self.minimum_squared_loading,
            minimum_lofo_loading_stability=(
                self.minimum_lofo_loading_stability
            ),
            topology_top_k=self.topology_top_k,
        )
        if expected != self.selected_actions:
            raise ValueError("supermode selected-action integrity differs")
        self._validate_bases()

    def report(self) -> dict[str, object]:
        """Return tensor-free action and nested-rank diagnostics."""

        self.validate_integrity()
        rank_rows = []
        for rank in self.available_ranks:
            active = self._actions_for_rank(rank)
            active_pairs = tuple(
                action
                for action in active
                if isinstance(action, GraphWaveletSupermodePair)
            )
            rank_rows.append(
                {
                    "rank": rank,
                    "active_action_count": len(active),
                    "active_merge_count": len(active_pairs),
                    "active_singleton_prune_count": (
                        len(active) - len(active_pairs)
                    ),
                    "mixed_basis_sha256": _tensor_sha256(
                        self._basis_without_validation(
                            rank,
                            one_hot_control=False,
                        )
                    ),
                    "one_hot_control_basis_sha256": _tensor_sha256(
                        self._basis_without_validation(
                            rank,
                            one_hot_control=True,
                        )
                    ),
                    "cumulative_action_loss": math.fsum(
                        action.action_loss for action in active
                    ),
                    "mean_merge_relative_response_loss": (
                        math.fsum(
                            action.relative_response_loss
                            for action in active_pairs
                        )
                        / len(active_pairs)
                        if active_pairs
                        else 0.0
                    ),
                    "mean_merge_mixing_participation": (
                        math.fsum(
                            action.mixing_participation
                            for action in active_pairs
                        )
                        / len(active_pairs)
                        if active_pairs
                        else 0.0
                    ),
                    "minimum_merge_lofo_loading_stability": (
                        min(
                            action.lofo_loading_stability
                            for action in active_pairs
                        )
                        if active_pairs
                        else 0.0
                    ),
                    "mean_merge_topology_interaction": (
                        math.fsum(
                            action.topology_interaction
                            for action in active_pairs
                        )
                        / len(active_pairs)
                        if active_pairs
                        else 0.0
                    ),
                }
            )
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "available_ranks": self.available_ranks,
            "merge_action_count": len(self.selected_pairs),
            "singleton_prune_action_count": len(self.singleton_prunes),
            "action_diagnostics": tuple(
                action.metadata() for action in self.selected_actions
            ),
            "rank_path": tuple(rank_rows),
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "method": self.method,
            "minimum_rank": self.minimum_rank,
            "minimum_squared_loading": self.minimum_squared_loading,
            "minimum_lofo_loading_stability": (
                self.minimum_lofo_loading_stability
            ),
            "topology_top_k": self.topology_top_k,
            "permutation_seed": self.permutation_seed,
            "fit_response_shape": self.fit_response_shape,
            "fit_response_sha256": self.fit_response_sha256,
            "fit_fold_response_shapes": self.fit_fold_response_shapes,
            "fit_fold_response_sha256s": self.fit_fold_response_sha256s,
            "signed_laplacian_sha256": self.signed_laplacian_sha256,
            "parent_basis": self.parent_basis.clone(),
            "response_gram": self.response_gram.clone(),
            "fit_fold_response_grams": tuple(
                value.clone() for value in self.fit_fold_response_grams
            ),
            "native_topology_matrix": self.native_topology_matrix.clone(),
            "selection_topology_matrix": (
                self.selection_topology_matrix.clone()
            ),
            "topology_permutation": self.topology_permutation.clone(),
            "selected_actions": tuple(
                action.metadata() for action in self.selected_actions
            ),
            "heldout_input_used": self.heldout_input_used,
            "algorithm": self.algorithm,
            "pair_semantics": self.pair_semantics,
            "action_policy": self.action_policy,
            "topology_semantics": self.topology_semantics,
            "lofo_semantics": self.lofo_semantics,
            "one_hot_control_semantics": self.one_hot_control_semantics,
            "fit_scope": self.fit_scope,
            "artifact_sha256": self.artifact_sha256,
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, object],
    ) -> FitOnlyGraphWaveletSupermodePath:
        expected = {
            "method",
            "minimum_rank",
            "minimum_squared_loading",
            "minimum_lofo_loading_stability",
            "topology_top_k",
            "permutation_seed",
            "fit_response_shape",
            "fit_response_sha256",
            "fit_fold_response_shapes",
            "fit_fold_response_sha256s",
            "signed_laplacian_sha256",
            "parent_basis",
            "response_gram",
            "fit_fold_response_grams",
            "native_topology_matrix",
            "selection_topology_matrix",
            "topology_permutation",
            "selected_actions",
            "heldout_input_used",
            "algorithm",
            "pair_semantics",
            "action_policy",
            "topology_semantics",
            "lofo_semantics",
            "one_hot_control_semantics",
            "fit_scope",
            "artifact_sha256",
            "artifact_kind",
            "format_version",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("supermode path state fields are invalid")
        raw_shape = value["fit_response_shape"]
        raw_fold_shapes = value["fit_fold_response_shapes"]
        raw_fold_hashes = value["fit_fold_response_sha256s"]
        raw_fold_grams = value["fit_fold_response_grams"]
        raw_actions = value["selected_actions"]
        for raw, label in (
            (raw_shape, "fit_response_shape"),
            (raw_fold_shapes, "fit_fold_response_shapes"),
            (raw_fold_hashes, "fit_fold_response_sha256s"),
            (raw_fold_grams, "fit_fold_response_grams"),
            (raw_actions, "selected_actions"),
        ):
            if not isinstance(raw, (tuple, list)):
                raise TypeError(f"{label} state must be a sequence")
        return cls(
            method=value["method"],  # type: ignore[arg-type]
            minimum_rank=int(value["minimum_rank"]),
            minimum_squared_loading=float(
                value["minimum_squared_loading"]
            ),
            minimum_lofo_loading_stability=float(
                value["minimum_lofo_loading_stability"]
            ),
            topology_top_k=int(value["topology_top_k"]),
            permutation_seed=int(value["permutation_seed"]),
            fit_response_shape=tuple(int(item) for item in raw_shape),
            fit_response_sha256=str(value["fit_response_sha256"]),
            fit_fold_response_shapes=tuple(
                tuple(int(item) for item in shape)  # type: ignore[arg-type]
                for shape in raw_fold_shapes
            ),
            fit_fold_response_sha256s=tuple(
                str(item) for item in raw_fold_hashes
            ),
            signed_laplacian_sha256=str(
                value["signed_laplacian_sha256"]
            ),
            parent_basis=value["parent_basis"],  # type: ignore[arg-type]
            response_gram=value["response_gram"],  # type: ignore[arg-type]
            fit_fold_response_grams=tuple(
                item for item in raw_fold_grams  # type: ignore[misc]
            ),
            native_topology_matrix=value[  # type: ignore[arg-type]
                "native_topology_matrix"
            ],
            selection_topology_matrix=value[  # type: ignore[arg-type]
                "selection_topology_matrix"
            ],
            topology_permutation=value[  # type: ignore[arg-type]
                "topology_permutation"
            ],
            selected_actions=tuple(
                _action_from_metadata(item)  # type: ignore[arg-type]
                for item in raw_actions
            ),
            heldout_input_used=bool(value["heldout_input_used"]),
            algorithm=str(value["algorithm"]),
            pair_semantics=str(value["pair_semantics"]),
            action_policy=str(value["action_policy"]),
            topology_semantics=str(value["topology_semantics"]),
            lofo_semantics=str(value["lofo_semantics"]),
            one_hot_control_semantics=str(
                value["one_hot_control_semantics"]
            ),
            fit_scope=str(value["fit_scope"]),
            artifact_sha256=str(value["artifact_sha256"]),
            artifact_kind=str(value["artifact_kind"]),
            format_version=int(value["format_version"]),
        )


def fit_graph_wavelet_supermode_merge(
    parent_basis: Tensor,
    weighted_fit_response: Tensor,
    signed_laplacian: Tensor,
    *,
    minimum_rank: int,
    method: GraphWaveletSupermodeMethod,
    fit_fold_responses: Sequence[Tensor],
    minimum_squared_loading: float = 0.10,
    minimum_lofo_loading_stability: float = 0.90,
    topology_top_k: int = 8,
    permutation_seed: int = 0,
) -> FitOnlyGraphWaveletSupermodePath:
    """Compile a deterministic fit-only mixed supermode/pruning path."""

    if method not in _METHODS:
        raise ValueError("method is invalid")
    minimum_rank = _positive_int(minimum_rank, label="minimum_rank")
    minimum_loading = _unit_interval(
        minimum_squared_loading,
        label="minimum_squared_loading",
    )
    if minimum_loading > 0.5:
        raise ValueError("minimum_squared_loading cannot exceed 0.5")
    minimum_stability = _unit_interval(
        minimum_lofo_loading_stability,
        label="minimum_lofo_loading_stability",
    )
    topology_top_k = _positive_int(
        topology_top_k,
        label="topology_top_k",
    )
    permutation_seed = _nonnegative_int(
        permutation_seed,
        label="permutation_seed",
    )
    if method != "permuted_graph_local" and permutation_seed != 0:
        raise ValueError(
            "only permuted_graph_local may use a nonzero permutation seed"
        )
    parent = _canonicalize_columns(
        _float64_tensor(
            parent_basis,
            label="parent_basis",
            ndim=2,
        )
    )
    node_count, parent_rank = parent.shape
    if (
        node_count <= 0
        or parent_rank <= 1
        or parent_rank > node_count
        or not _orthonormal(parent)
    ):
        raise ValueError("parent_basis must be nonempty and orthonormal")
    if not 1 <= minimum_rank < parent_rank:
        raise ValueError("minimum_rank must lie in [1, parent_rank)")
    if topology_top_k >= parent_rank:
        raise ValueError("topology_top_k must be below parent rank")
    response = _float64_tensor(
        weighted_fit_response,
        label="weighted_fit_response",
    )
    if response.ndim < 2 or response.shape[0] != node_count:
        raise ValueError(
            "weighted_fit_response must have shape [nodes, ...]"
        )
    if (
        isinstance(fit_fold_responses, (str, bytes))
        or not isinstance(fit_fold_responses, Sequence)
        or len(fit_fold_responses) < 2
    ):
        raise ValueError(
            "fit_fold_responses must contain at least two fit-only folds"
        )
    folds = tuple(
        _float64_tensor(
            value,
            label="fit_fold_response",
        )
        for value in fit_fold_responses
    )
    if any(value.ndim < 2 or value.shape[0] != node_count for value in folds):
        raise ValueError("every fit fold must have shape [nodes, ...]")
    laplacian = _symmetric(
        _float64_tensor(
            signed_laplacian,
            label="signed_laplacian",
            ndim=2,
        ),
        label="signed_laplacian",
        require_psd=True,
    )
    if laplacian.shape != (node_count, node_count):
        raise ValueError("signed_laplacian and parent basis widths differ")
    coordinates = parent.T @ response.reshape(node_count, -1)
    response_gram = _symmetric(
        (coordinates @ coordinates.T).contiguous(),
        label="response_gram",
        require_psd=True,
    )
    if float(torch.trace(response_gram)) <= torch.finfo(torch.float64).tiny:
        raise ValueError("weighted_fit_response must contain nonzero energy")
    fold_grams = tuple(
        _symmetric(
            (
                (parent.T @ fold.reshape(node_count, -1))
                @ (parent.T @ fold.reshape(node_count, -1)).T
            ).contiguous(),
            label="fit_fold_response_gram",
            require_psd=True,
        )
        for fold in folds
    )
    fold_sum = torch.stack(fold_grams).sum(dim=0)
    scale = max(float(response_gram.abs().max()), 1.0)
    if not torch.allclose(
        fold_sum,
        response_gram,
        atol=2.0e-10 * scale,
        rtol=2.0e-10,
    ):
        raise ValueError(
            "fit_fold_responses must exactly partition weighted_fit_response "
            "in response-Gram space"
        )
    native_topology = _topology_matrix(parent, laplacian)
    if method == "permuted_graph_local":
        permutation = torch.randperm(
            node_count,
            generator=torch.Generator(device="cpu").manual_seed(
                permutation_seed
            ),
            dtype=torch.int64,
        )
        selection_topology = _topology_matrix(
            parent.index_select(0, permutation),
            laplacian,
        )
    else:
        permutation = torch.arange(node_count, dtype=torch.int64)
        selection_topology = native_topology
    actions = _select_actions(
        parent_rank=parent_rank,
        minimum_rank=minimum_rank,
        method=method,
        response_gram=response_gram,
        fold_response_grams=fold_grams,
        native_topology=native_topology,
        selection_topology=selection_topology,
        minimum_squared_loading=minimum_loading,
        minimum_lofo_loading_stability=minimum_stability,
        topology_top_k=topology_top_k,
    )
    return FitOnlyGraphWaveletSupermodePath(
        method=method,
        minimum_rank=minimum_rank,
        minimum_squared_loading=minimum_loading,
        minimum_lofo_loading_stability=minimum_stability,
        topology_top_k=topology_top_k,
        permutation_seed=permutation_seed,
        fit_response_shape=tuple(int(value) for value in response.shape),
        fit_response_sha256=_tensor_sha256(response),
        fit_fold_response_shapes=tuple(
            tuple(int(value) for value in fold.shape)
            for fold in folds
        ),
        fit_fold_response_sha256s=tuple(
            _tensor_sha256(fold) for fold in folds
        ),
        signed_laplacian_sha256=_tensor_sha256(laplacian),
        parent_basis=parent,
        response_gram=response_gram,
        fit_fold_response_grams=fold_grams,
        native_topology_matrix=native_topology,
        selection_topology_matrix=selection_topology,
        topology_permutation=permutation,
        selected_actions=actions,
    )
