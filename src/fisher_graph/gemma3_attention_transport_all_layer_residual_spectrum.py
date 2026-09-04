"""All-layer residual-spectrum allocation for Gemma V/O transport.

The spectrum rung begins from a completed, externally supplied
``GlobalSiteRankOneFisherAttentionTransportGeneratorBank``.  It never fits or
mutates that parent.  Instead, it appends an ordered sequence of
invocation-local rank-one components.  Sites may repeat in the sequence.  For
component ``m`` assigned to invocation ``i_m`` the additive candidate is

    R[m,k](xi) = l[m,k] (r[m,k]^T xi).

The stored factor geometry is ``[M,K,H]``.  Every right vector has unit norm
and a deterministic positive max-absolute pivot; component order is stable and
part of the artifact identity.  Off-site execution returns the parent output
bit-for-bit.

Each proposal freezes freshly reselected exact signed Top-2 coefficients,
computes the Fisher-loss gradient with respect to an implicit dense operator,
and extracts one projected top singular pair per generator using deterministic
hashed multistarts and matrix-free power iteration.  A joint exact ``K x K``
Fisher solve chooses component amplitudes.  Proposal coefficients remain
frozen only for construction; acceptance always reselects the exact sparse
oracle on the designated evaluation panel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
from torch import Tensor

from .gemma3_attention_transport_compiler import (
    AttentionTransportBatch,
    _cpu_float64,
    _tensor_sha256,
    samplewise_joint_fisher_loss,
)
from .gemma3_attention_transport_global_rank1 import (
    GLOBAL_ATTENTION_INVOCATION_IDS,
    _canonicalize_rank_one_factors,
    _is_canonical_rank_one_gauge,
)
from .gemma3_attention_transport_global_site_rank1 import (
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict,
)
from .gemma3_attention_transport_hierarchical_edge import (
    _capture_values,
    _exact_sparse_coefficients_by_capture,
    _mutation_ordinal,
)
from .gemma3_attention_transport_sparse_v2 import (
    SparseAttentionTransportOracleEvaluation,
    evaluate_sparse_attention_transport_bank,
)


ALL_LAYER_RESIDUAL_COMPONENT_SEMANTICS = (
    "ordered_invocation_local_rank_one_residual_components_v1"
)
ALL_LAYER_RESIDUAL_GAUGE_SEMANTICS = (
    "unit_right_per_component_generator_max_abs_positive_v1"
)
ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS = (
    "matrix_free_fisher_gradient_projected_top_singular_pair_v1"
)
ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS = (
    "exact_joint_k_by_k_fisher_amplitude_solve_v1"
)
ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS = (
    "fresh_reselected_exact_sparse_oracle_strict_positive_gain_v1"
)
ALL_LAYER_RESIDUAL_GREEDY_SEMANTICS = (
    "budget_two_cached_off_site_marginals_numeric_mutation_tiebreak_v1"
)

_BANK_DOMAIN = b"fisher_graph:attention-transport-all-layer-residual-bank:v1\0"
_HASHED_START_DOMAIN = b"fisher_graph:residual-spectrum-hashed-start:v1\0"
_PROFILE_DOMAIN = b"fisher_graph:residual-spectrum-allocation-profile:v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def residual_allocation_profile_sha256(
    allocation_profile: Sequence[str],
) -> str:
    """Hash an ordered, possibly empty allocation profile."""

    if isinstance(allocation_profile, (str, bytes)) or not isinstance(
        allocation_profile,
        Sequence,
    ):
        raise TypeError("allocation_profile must be a materialized sequence")
    values = tuple(allocation_profile)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("allocation profile entries must be nonempty strings")
    return hashlib.sha256(
        _PROFILE_DOMAIN + _canonical_json_bytes(list(values))
    ).hexdigest()


def _canonicalize_allocated_factors(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> tuple[Tensor, Tensor]:
    if (
        isinstance(left, Tensor)
        and isinstance(right, Tensor)
        and left.ndim == 3
        and right.ndim == 3
        and left.shape == right.shape
        and left.shape[0] == 0
        and left.shape[1] > 0
        and left.shape[2] > 0
        and left.is_floating_point()
        and right.is_floating_point()
    ):
        return (
            left.detach().to(device="cpu", dtype=torch.float64).clone().contiguous(),
            right.detach().to(device="cpu", dtype=torch.float64).clone().contiguous(),
        )
    left_value = _cpu_float64(
        left,
        label="allocated residual left vectors",
        ndim=3,
    )
    right_value = _cpu_float64(
        right,
        label="allocated residual right vectors",
        ndim=3,
    )
    if left_value.shape != right_value.shape:
        raise ValueError("allocated residual factor shapes do not match")
    components, generators, hidden = left_value.shape
    if components == 0:
        return left_value.clone().contiguous(), right_value.clone().contiguous()
    flat_left = left_value.reshape(components * generators, hidden)
    flat_right = right_value.reshape(components * generators, hidden)
    if _is_canonical_rank_one_gauge(
        flat_left,
        flat_right,
        gauge_floor=gauge_floor,
    ):
        return left_value.clone().contiguous(), right_value.clone().contiguous()
    canonical_left, canonical_right = _canonicalize_rank_one_factors(
        flat_left,
        flat_right,
        gauge_floor=gauge_floor,
    )
    return (
        canonical_left.reshape(components, generators, hidden).contiguous(),
        canonical_right.reshape(components, generators, hidden).contiguous(),
    )


def _is_canonical_allocated_gauge(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> bool:
    try:
        if (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.ndim == 3
            and right.ndim == 3
            and left.shape == right.shape
            and left.shape[0] == 0
            and left.shape[1] > 0
            and left.shape[2] > 0
            and left.is_floating_point()
            and right.is_floating_point()
        ):
            return True
        left_value = _cpu_float64(
            left,
            label="allocated residual left vectors",
            ndim=3,
        )
        right_value = _cpu_float64(
            right,
            label="allocated residual right vectors",
            ndim=3,
        )
        if left_value.shape != right_value.shape:
            return False
        components, generators, hidden = left_value.shape
        if components == 0:
            return True
        return _is_canonical_rank_one_gauge(
            left_value.reshape(components * generators, hidden),
            right_value.reshape(components * generators, hidden),
            gauge_floor=gauge_floor,
        )
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class AllLayerResidualSpectrumGeneratorBank(
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank
):
    """Site-rank-one parent plus ordered invocation-local residual components."""

    allocated_invocation_ids: tuple[str, ...]
    allocated_residual_left_vectors: Tensor
    allocated_residual_right_vectors: Tensor
    parent_site_rank_one_bank_sha256: str
    _allocated_tensor_sha256s: tuple[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _parent_bank_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _allocated_artifact_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank.__post_init__(self)
        parent_sha = (
            GlobalSiteRankOneFisherAttentionTransportGeneratorBank.artifact_sha256.fget(
                self
            )
        )
        if (
            not _is_sha256(self.parent_site_rank_one_bank_sha256)
            or self.parent_site_rank_one_bank_sha256 != parent_sha
        ):
            raise ValueError("allocated residual parent lineage does not close")
        if (
            not isinstance(self.allocated_invocation_ids, tuple)
            or any(
                invocation_id not in self.hierarchy.invocation_ids
                for invocation_id in self.allocated_invocation_ids
            )
        ):
            raise ValueError("allocated residual invocation sequence is invalid")
        left, right = _canonicalize_allocated_factors(
            self.allocated_residual_left_vectors,
            self.allocated_residual_right_vectors,
            gauge_floor=self.gauge_floor,
        )
        expected = (
            len(self.allocated_invocation_ids),
            self.generator_count,
            self.hidden_width,
        )
        if left.shape != expected:
            raise ValueError("allocated residual factors have the wrong geometry")
        object.__setattr__(self, "allocated_residual_left_vectors", left)
        object.__setattr__(self, "allocated_residual_right_vectors", right)
        tensor_hashes = (_tensor_sha256(left), _tensor_sha256(right))
        object.__setattr__(self, "_allocated_tensor_sha256s", tensor_hashes)
        object.__setattr__(self, "_parent_bank_sha256", parent_sha)
        payload = self._allocated_artifact_payload(tensor_hashes)
        object.__setattr__(
            self,
            "_allocated_artifact_sha256",
            hashlib.sha256(_BANK_DOMAIN + _canonical_json_bytes(payload)).hexdigest(),
        )

    def _allocated_artifact_payload(
        self,
        tensor_hashes: tuple[str, str],
    ) -> dict[str, object]:
        return {
            "acceptance_semantics": ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS,
            "allocated_invocation_ids": list(self.allocated_invocation_ids),
            "allocated_parameter_count": self.allocated_parameter_count,
            "allocated_residual_left_vectors_sha256": tensor_hashes[0],
            "allocated_residual_right_vectors_sha256": tensor_hashes[1],
            "amplitude_semantics": ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS,
            "component_semantics": ALL_LAYER_RESIDUAL_COMPONENT_SEMANTICS,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": ALL_LAYER_RESIDUAL_GAUGE_SEMANTICS,
            "generator_count": self.generator_count,
            "hidden_width": self.hidden_width,
            "parent_site_rank_one_bank_sha256": self._parent_bank_sha256,
            "proposal_semantics": ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS,
            "rank_profile": [list(value) for value in self.rank_profile],
        }

    def _validate_allocated_unchanged(self) -> None:
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank._validate_global_site_rank_one_unchanged(
            self
        )
        current = (
            _tensor_sha256(self.allocated_residual_left_vectors),
            _tensor_sha256(self.allocated_residual_right_vectors),
        )
        if current != self._allocated_tensor_sha256s:
            raise RuntimeError("allocated residual bank changed after construction")
        if not _is_canonical_allocated_gauge(
            self.allocated_residual_left_vectors,
            self.allocated_residual_right_vectors,
            gauge_floor=self.gauge_floor,
        ):
            raise RuntimeError("allocated residual gauge drifted")
        parent_sha = (
            GlobalSiteRankOneFisherAttentionTransportGeneratorBank.artifact_sha256.fget(
                self
            )
        )
        if parent_sha != self._parent_bank_sha256:
            raise RuntimeError("allocated residual parent bank changed")

    @property
    def allocated_component_count(self) -> int:
        return len(self.allocated_invocation_ids)

    @property
    def allocated_parameter_count(self) -> int:
        expected = (
            2
            * self.allocated_component_count
            * self.generator_count
            * self.hidden_width
        )
        observed = int(
            self.allocated_residual_left_vectors.numel()
            + self.allocated_residual_right_vectors.numel()
        )
        if observed != expected:
            raise RuntimeError("allocated residual parameter count drifted")
        return observed

    @property
    def stored_parameter_count(self) -> int:
        parent = (
            GlobalSiteRankOneFisherAttentionTransportGeneratorBank.stored_parameter_count.fget(
                self
            )
        )
        return int(parent + self.allocated_parameter_count)

    @property
    def allocated_rank_profile(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                invocation_id,
                self.allocated_invocation_ids.count(invocation_id),
            )
            for invocation_id in self.hierarchy.invocation_ids
        )

    @property
    def rank_profile(self) -> tuple[tuple[str, int], ...]:
        """Return total invocation-local residual rank, including the parent."""

        return tuple(
            (
                invocation_id,
                (1 if invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS else 0)
                + self.allocated_invocation_ids.count(invocation_id),
            )
            for invocation_id in self.hierarchy.invocation_ids
        )

    @property
    def artifact_sha256(self) -> str:
        self._validate_allocated_unchanged()
        return self._allocated_artifact_sha256

    def __hash__(self) -> int:
        return hash((AllLayerResidualSpectrumGeneratorBank, self.artifact_sha256))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AllLayerResidualSpectrumGeneratorBank)
            and self.artifact_sha256 == other.artifact_sha256
        )

    def allocated_candidate_residuals_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        """Return ordered allocated residual candidates, or exact zero off-site."""

        self._validate_allocated_unchanged()
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        result = torch.zeros_like(contextual)
        for component_index, component_site in enumerate(
            self.allocated_invocation_ids
        ):
            if component_site != invocation_id:
                continue
            left = self.allocated_residual_left_vectors[component_index]
            right = self.allocated_residual_right_vectors[component_index]
            activation = torch.einsum("nkh,kh->nk", contextual, right)
            result = result + activation[:, :, None] * left[None]
        return result.contiguous()

    def candidate_outputs_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        parent = GlobalSiteRankOneFisherAttentionTransportGeneratorBank.candidate_outputs_from_head_contexts(
            self,
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        if invocation_id not in self.allocated_invocation_ids:
            return parent
        return (
            parent
            + self.allocated_candidate_residuals_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=head_contexts,
            )
        ).contiguous()

    def state_dict(self) -> dict[str, object]:
        self._validate_allocated_unchanged()
        return {
            "acceptance_semantics": ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS,
            "allocated_invocation_ids": list(self.allocated_invocation_ids),
            "allocated_residual_left_vectors": (
                self.allocated_residual_left_vectors.clone()
            ),
            "allocated_residual_right_vectors": (
                self.allocated_residual_right_vectors.clone()
            ),
            "amplitude_semantics": ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS,
            "artifact_sha256": self._allocated_artifact_sha256,
            "component_semantics": ALL_LAYER_RESIDUAL_COMPONENT_SEMANTICS,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": ALL_LAYER_RESIDUAL_GAUGE_SEMANTICS,
            "parent_site_rank_one_bank": (
                GlobalSiteRankOneFisherAttentionTransportGeneratorBank.state_dict(
                    self
                )
            ),
            "parent_site_rank_one_bank_sha256": self._parent_bank_sha256,
            "proposal_semantics": ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS,
            "rank_profile": [list(value) for value in self.rank_profile],
        }


def all_layer_residual_spectrum_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> AllLayerResidualSpectrumGeneratorBank:
    """Reload a source-free allocated bank and verify complete lineage."""

    expected = {
        "acceptance_semantics",
        "allocated_invocation_ids",
        "allocated_residual_left_vectors",
        "allocated_residual_right_vectors",
        "amplitude_semantics",
        "artifact_sha256",
        "component_semantics",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "gauge_floor",
        "gauge_semantics",
        "parent_site_rank_one_bank",
        "parent_site_rank_one_bank_sha256",
        "proposal_semantics",
        "rank_profile",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("all-layer residual bank state fields are invalid")
    raw_parent = state["parent_site_rank_one_bank"]
    raw_ids = state["allocated_invocation_ids"]
    gauge_floor = state["gauge_floor"]
    if (
        not isinstance(raw_parent, Mapping)
        or not isinstance(raw_ids, list)
        or any(not isinstance(value, str) for value in raw_ids)
        or state["acceptance_semantics"]
        != ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS
        or state["amplitude_semantics"]
        != ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS
        or state["component_semantics"]
        != ALL_LAYER_RESIDUAL_COMPONENT_SEMANTICS
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_raw_teacher_outputs_or_gradients"] is not False
        or state["gauge_semantics"] != ALL_LAYER_RESIDUAL_GAUGE_SEMANTICS
        or state["proposal_semantics"] != ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS
        or not _is_sha256(state["artifact_sha256"])
        or not _is_sha256(state["parent_site_rank_one_bank_sha256"])
        or isinstance(gauge_floor, bool)
        or not isinstance(gauge_floor, (int, float))
        or not math.isfinite(float(gauge_floor))
        or float(gauge_floor) <= 0.0
        or not _is_canonical_allocated_gauge(
            state["allocated_residual_left_vectors"],
            state["allocated_residual_right_vectors"],
            gauge_floor=float(gauge_floor),
        )
    ):
        raise ValueError("all-layer residual bank safety semantics are invalid")
    parent = (
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            raw_parent
        )
    )
    if (
        state["parent_site_rank_one_bank_sha256"] != parent.artifact_sha256
        or float(gauge_floor) != parent.gauge_floor
    ):
        raise ValueError("all-layer residual parent lineage drifted")
    bank = AllLayerResidualSpectrumGeneratorBank(
        left_factors=parent.left_factors,
        singular_scales=parent.singular_scales,
        right_factors=parent.right_factors,
        head_coefficients=parent.head_coefficients,
        head_ids=parent.head_ids,
        hierarchy=parent.hierarchy,
        block_scale_contrasts=parent.block_scale_contrasts,
        site_scale_contrasts=parent.site_scale_contrasts,
        signed_feature_floor=parent.signed_feature_floor,
        global_site_residual_left_vectors=(
            parent.global_site_residual_left_vectors
        ),
        global_site_residual_right_vectors=(
            parent.global_site_residual_right_vectors
        ),
        shared_rank_one_bank_sha256=parent.shared_rank_one_bank_sha256,
        gauge_floor=parent.gauge_floor,
        allocated_invocation_ids=tuple(raw_ids),
        allocated_residual_left_vectors=state[
            "allocated_residual_left_vectors"
        ],
        allocated_residual_right_vectors=state[
            "allocated_residual_right_vectors"
        ],
        parent_site_rank_one_bank_sha256=parent.artifact_sha256,
    )
    if (
        state["rank_profile"] != [list(value) for value in bank.rank_profile]
        or state["artifact_sha256"] != bank.artifact_sha256
    ):
        raise ValueError("all-layer residual artifact lineage drifted")
    return bank


def all_layer_residual_spectrum_generator_bank_from_parent(
    parent: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
) -> AllLayerResidualSpectrumGeneratorBank:
    """Wrap a supplied parent with an exactly empty allocation sequence."""

    if not isinstance(
        parent,
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("parent must be a global site rank-one bank")
    parent._validate_global_site_rank_one_unchanged()
    empty = torch.empty(
        0,
        parent.generator_count,
        parent.hidden_width,
        dtype=torch.float64,
    )
    return AllLayerResidualSpectrumGeneratorBank(
        left_factors=parent.left_factors,
        singular_scales=parent.singular_scales,
        right_factors=parent.right_factors,
        head_coefficients=parent.head_coefficients,
        head_ids=parent.head_ids,
        hierarchy=parent.hierarchy,
        block_scale_contrasts=parent.block_scale_contrasts,
        site_scale_contrasts=parent.site_scale_contrasts,
        signed_feature_floor=parent.signed_feature_floor,
        global_site_residual_left_vectors=(
            parent.global_site_residual_left_vectors
        ),
        global_site_residual_right_vectors=(
            parent.global_site_residual_right_vectors
        ),
        shared_rank_one_bank_sha256=parent.shared_rank_one_bank_sha256,
        gauge_floor=parent.gauge_floor,
        allocated_invocation_ids=(),
        allocated_residual_left_vectors=empty,
        allocated_residual_right_vectors=empty.clone(),
        parent_site_rank_one_bank_sha256=parent.artifact_sha256,
    )


def _append_allocated_component(
    bank: AllLayerResidualSpectrumGeneratorBank,
    *,
    invocation_id: str,
    left: Tensor,
    right: Tensor,
) -> AllLayerResidualSpectrumGeneratorBank:
    if not isinstance(bank, AllLayerResidualSpectrumGeneratorBank):
        raise TypeError("bank must be an all-layer residual spectrum bank")
    if invocation_id not in bank.hierarchy.invocation_ids:
        raise ValueError("allocated invocation is absent from the parent hierarchy")
    left_value = _cpu_float64(left, label="new component left", ndim=2)
    right_value = _cpu_float64(right, label="new component right", ndim=2)
    if left_value.shape != (bank.generator_count, bank.hidden_width):
        raise ValueError("new component factor geometry is invalid")
    canonical_left, canonical_right = _canonicalize_rank_one_factors(
        left_value,
        right_value,
        gauge_floor=bank.gauge_floor,
    )
    return AllLayerResidualSpectrumGeneratorBank(
        left_factors=bank.left_factors,
        singular_scales=bank.singular_scales,
        right_factors=bank.right_factors,
        head_coefficients=bank.head_coefficients,
        head_ids=bank.head_ids,
        hierarchy=bank.hierarchy,
        block_scale_contrasts=bank.block_scale_contrasts,
        site_scale_contrasts=bank.site_scale_contrasts,
        signed_feature_floor=bank.signed_feature_floor,
        global_site_residual_left_vectors=(
            bank.global_site_residual_left_vectors
        ),
        global_site_residual_right_vectors=(
            bank.global_site_residual_right_vectors
        ),
        shared_rank_one_bank_sha256=bank.shared_rank_one_bank_sha256,
        gauge_floor=bank.gauge_floor,
        allocated_invocation_ids=bank.allocated_invocation_ids + (invocation_id,),
        allocated_residual_left_vectors=torch.cat(
            (bank.allocated_residual_left_vectors, canonical_left[None]),
            dim=0,
        ),
        allocated_residual_right_vectors=torch.cat(
            (bank.allocated_residual_right_vectors, canonical_right[None]),
            dim=0,
        ),
        parent_site_rank_one_bank_sha256=bank.parent_site_rank_one_bank_sha256,
    )


@dataclass(frozen=True, slots=True)
class AllLayerResidualSpectrumConfig:
    """Deterministic K8/R8/Top2-compatible residual-spectrum protocol."""

    generator_count: int = 8
    generator_rank: int = 8
    active_count: int = 2
    euclidean_floor: float = 1.0e-3
    coefficient_ridge: float = 0.0
    amplitude_ridge: float = 0.0
    power_iterations: int = 16
    deterministic_start_count: int = 4
    allocation_budget: int = 2
    gauge_floor: float = 1.0e-12
    minimum_relative_fisher_gain: float = 1.0e-12
    tie_tolerance: float = 1.0e-15

    def __post_init__(self) -> None:
        for label, value, minimum in (
            ("generator_count", self.generator_count, 1),
            ("generator_rank", self.generator_rank, 1),
            ("active_count", self.active_count, 1),
            ("power_iterations", self.power_iterations, 1),
            ("deterministic_start_count", self.deterministic_start_count, 1),
            ("allocation_budget", self.allocation_budget, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} is outside its valid range")
        if self.active_count > self.generator_count:
            raise ValueError("active_count exceeds generator inventory")
        for label, value, positive in (
            ("euclidean_floor", self.euclidean_floor, True),
            ("coefficient_ridge", self.coefficient_ridge, False),
            ("amplitude_ridge", self.amplitude_ridge, False),
            ("gauge_floor", self.gauge_floor, True),
            ("minimum_relative_fisher_gain", self.minimum_relative_fisher_gain, True),
            ("tie_tolerance", self.tie_tolerance, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (positive and float(value) <= 0.0)
                or (not positive and float(value) < 0.0)
            ):
                raise ValueError(f"{label} is invalid")

    def validate_bank(
        self,
        bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    ) -> None:
        if (
            not isinstance(
                bank,
                GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
            )
            or bank.generator_count != self.generator_count
            or bank.generator_rank != self.generator_rank
            or bank.gauge_floor != float(self.gauge_floor)
        ):
            raise ValueError("residual-spectrum config does not match the bank")


ResidualSpectrumProgressCallback = Callable[[str, int, int, str | None], None]


def _notify_progress(
    callback: ResidualSpectrumProgressCallback | None,
    phase: str,
    completed: int,
    total: int,
    invocation_id: str | None,
) -> None:
    if callback is not None:
        callback(phase, completed, total, invocation_id)


def _site_captures(
    captures: Sequence[AttentionTransportBatch],
    *,
    invocation_id: str | None = None,
) -> tuple[str, tuple[AttentionTransportBatch, ...]]:
    values = _capture_values(captures)
    ids = {capture.invocation_id for capture in values}
    if len(ids) != 1:
        raise ValueError("site marginal captures must contain exactly one invocation")
    observed = next(iter(ids))
    if invocation_id is not None and observed != invocation_id:
        raise ValueError("proposal and evaluation site invocations do not match")
    return observed, values


def _hashed_start_vector(
    *,
    bank_sha256: str,
    invocation_id: str,
    generator_index: int,
    start_index: int,
    hidden_width: int,
) -> Tensor:
    key = _canonical_json_bytes(
        {
            "bank_sha256": bank_sha256,
            "generator_index": generator_index,
            "hidden_width": hidden_width,
            "invocation_id": invocation_id,
            "start_index": start_index,
        }
    )
    raw = hashlib.shake_256(_HASHED_START_DOMAIN + key).digest(8 * hidden_width)
    denominator = float(2**64 - 1)
    values = [
        2.0 * int.from_bytes(raw[index : index + 8], "little") / denominator
        - 1.0
        for index in range(0, len(raw), 8)
    ]
    return torch.tensor(values, dtype=torch.float64)


def _prior_right_vectors(
    bank: AllLayerResidualSpectrumGeneratorBank,
    *,
    invocation_id: str,
    generator_index: int,
) -> Tensor:
    values: list[Tensor] = []
    if invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
        site_index = GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
        values.append(
            bank.global_site_residual_right_vectors[
                site_index,
                generator_index,
            ]
        )
    values.extend(
        bank.allocated_residual_right_vectors[component_index, generator_index]
        for component_index, component_site in enumerate(
            bank.allocated_invocation_ids
        )
        if component_site == invocation_id
    )
    if not values:
        return torch.empty(0, bank.hidden_width, dtype=torch.float64)
    return torch.stack(tuple(values)).contiguous()


def _right_projection(
    prior_rights: Tensor,
    *,
    hidden_width: int,
    gauge_floor: float,
) -> tuple[Callable[[Tensor], Tensor], int]:
    if prior_rights.shape == (0, hidden_width):
        return lambda value: value, 0
    if prior_rights.ndim != 2 or prior_rights.shape[1] != hidden_width:
        raise ValueError("prior residual right-vector geometry is invalid")
    gram = prior_rights @ prior_rights.T
    gram_pinv = torch.linalg.pinv(gram, hermitian=True)
    rank = int(torch.linalg.matrix_rank(prior_rights, tol=gauge_floor))

    def project(value: Tensor) -> Tensor:
        return (
            value
            - prior_rights.T @ (gram_pinv @ (prior_rights @ value))
        ).contiguous()

    return project, rank


def _matrix_free_top_singular_pair(
    contextual_inputs: Tensor,
    output_gradient: Tensor,
    coefficients: Tensor,
    *,
    bank_sha256: str,
    invocation_id: str,
    generator_index: int,
    prior_rights: Tensor,
    config: AllLayerResidualSpectrumConfig,
) -> tuple[Tensor, Tensor, float, int]:
    """Return a projected top pair without materializing an ``H x H`` matrix."""

    hidden_width = contextual_inputs.shape[-1]
    contexts = contextual_inputs[:, generator_index]
    beta = coefficients[:, generator_index]
    project, projection_rank = _right_projection(
        prior_rights,
        hidden_width=hidden_width,
        gauge_floor=config.gauge_floor,
    )

    def gradient_times_right(right: Tensor) -> Tensor:
        activation = contexts @ right
        return torch.einsum(
            "n,nh->h",
            beta * activation,
            output_gradient,
        )

    def gradient_transpose_times_left(left: Tensor) -> Tensor:
        activation = output_gradient @ left
        return torch.einsum("n,nh->h", beta * activation, contexts)

    best_sigma = -math.inf
    best_left: Tensor | None = None
    best_right: Tensor | None = None
    floor = float(config.gauge_floor)
    for start_index in range(config.deterministic_start_count):
        right = project(
            _hashed_start_vector(
                bank_sha256=bank_sha256,
                invocation_id=invocation_id,
                generator_index=generator_index,
                start_index=start_index,
                hidden_width=hidden_width,
            )
        )
        right_norm = torch.linalg.vector_norm(right)
        if float(right_norm) <= floor:
            continue
        right = right / right_norm
        left: Tensor | None = None
        for _ in range(config.power_iterations):
            mapped_left = gradient_times_right(right)
            left_norm = torch.linalg.vector_norm(mapped_left)
            if float(left_norm) <= floor:
                left = None
                break
            left = mapped_left / left_norm
            mapped_right = project(gradient_transpose_times_left(left))
            right_norm = torch.linalg.vector_norm(mapped_right)
            if float(right_norm) <= floor:
                left = None
                break
            right = mapped_right / right_norm
        if left is None:
            continue
        mapped_left = gradient_times_right(right)
        sigma = float(torch.linalg.vector_norm(mapped_left))
        if sigma > best_sigma:
            best_sigma = sigma
            best_left = mapped_left / max(sigma, floor)
            best_right = right.clone()

    if best_left is None or best_right is None:
        # The projected right space can be exhausted in tiny geometries, or
        # the Fisher gradient can vanish throughout it.  Return an identically
        # zero product so the amplitude solve cannot revive a direction from
        # outside the allowed complement.  Strict acceptance will reject the
        # resulting no-op proposal.
        best_left = torch.zeros(hidden_width, dtype=torch.float64)
        best_right = torch.zeros(hidden_width, dtype=torch.float64)
        return best_left, best_right, 0.0, projection_rank
    pivot = int(best_right.abs().argmax())
    if float(best_right[pivot]) < 0.0:
        best_left = -best_left
        best_right = -best_right
    return (
        best_left.contiguous(),
        best_right.contiguous(),
        float(max(best_sigma, 0.0)),
        projection_rank,
    )


def _exact_fisher_amplitudes(
    directional_candidates: Tensor,
    baseline_prediction: Tensor,
    target: Tensor,
    score_gradient: Tensor,
    *,
    euclidean_floor: float,
    amplitude_ridge: float,
) -> Tensor:
    """Solve all generator amplitudes jointly in the exact Fisher quadratic."""

    rows, generators, hidden = directional_candidates.shape
    if (
        baseline_prediction.shape != (rows, hidden)
        or target.shape != (rows, hidden)
        or score_gradient.shape != (rows, hidden)
    ):
        raise ValueError("Fisher amplitude inputs do not align")
    projected = torch.einsum(
        "nkh,nh->nk",
        directional_candidates,
        score_gradient,
    )
    gram = (
        float(euclidean_floor)
        * torch.einsum(
            "nkh,njh->kj",
            directional_candidates,
            directional_candidates,
        )
        / rows
        + torch.einsum("nk,nj->kj", projected, projected) / rows
        + float(amplitude_ridge)
        * torch.eye(generators, dtype=torch.float64)
    )
    error = baseline_prediction - target
    projected_error = torch.einsum("nh,nh->n", error, score_gradient)
    gradient = (
        float(euclidean_floor)
        * torch.einsum("nkh,nh->k", directional_candidates, error)
        / rows
        + torch.einsum("nk,n->k", projected, projected_error) / rows
    )
    return (
        torch.linalg.solve(gram, -gradient)
        if amplitude_ridge > 0.0
        else torch.linalg.pinv(gram, hermitian=True) @ (-gradient)
    ).contiguous()


@dataclass(frozen=True, slots=True)
class ResidualSpectrumSiteMarginal:
    """Source-free proposal factors and metrics for one invocation."""

    invocation_id: str
    source_bank_sha256: str
    proposal_bank: AllLayerResidualSpectrumGeneratorBank
    proposal_fit_current_oracle: SparseAttentionTransportOracleEvaluation
    proposal_fit_candidate_oracle: SparseAttentionTransportOracleEvaluation
    evaluation_current_oracle: SparseAttentionTransportOracleEvaluation
    evaluation_candidate_oracle: SparseAttentionTransportOracleEvaluation
    frozen_current_fisher_loss: float
    frozen_candidate_fisher_loss: float
    frozen_relative_fisher_gain: float
    evaluation_relative_fisher_gain: float
    generator_singular_values: tuple[float, ...]
    generator_amplitudes: tuple[float, ...]
    right_projection_ranks: tuple[int, ...]
    minimum_relative_fisher_gain: float

    def __post_init__(self) -> None:
        evaluations = (
            self.proposal_fit_current_oracle,
            self.proposal_fit_candidate_oracle,
            self.evaluation_current_oracle,
            self.evaluation_candidate_oracle,
        )
        if (
            not isinstance(self.invocation_id, str)
            or not _is_sha256(self.source_bank_sha256)
            or not isinstance(self.proposal_bank, AllLayerResidualSpectrumGeneratorBank)
            or not self.proposal_bank.allocated_invocation_ids
            or self.proposal_bank.allocated_invocation_ids[-1] != self.invocation_id
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in evaluations
            )
            or any(
                tuple(key for key, _loss in value.per_invocation_relative_fisher_loss)
                != (self.invocation_id,)
                for value in evaluations
            )
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in (
                    self.frozen_current_fisher_loss,
                    self.frozen_candidate_fisher_loss,
                    *self.generator_singular_values,
                )
            )
            or any(
                not math.isfinite(float(value))
                for value in (
                    self.frozen_relative_fisher_gain,
                    self.evaluation_relative_fisher_gain,
                    *self.generator_amplitudes,
                )
            )
            or len(self.generator_singular_values)
            != self.proposal_bank.generator_count
            or len(self.generator_amplitudes) != self.proposal_bank.generator_count
            or len(self.right_projection_ranks)
            != self.proposal_bank.generator_count
            or any(type(value) is not int or value < 0 for value in self.right_projection_ranks)
            or isinstance(self.minimum_relative_fisher_gain, bool)
            or not isinstance(self.minimum_relative_fisher_gain, (int, float))
            or not math.isfinite(float(self.minimum_relative_fisher_gain))
            or float(self.minimum_relative_fisher_gain) <= 0.0
            or not math.isclose(
                self.frozen_relative_fisher_gain,
                (
                    self.frozen_current_fisher_loss
                    - self.frozen_candidate_fisher_loss
                )
                / self.proposal_fit_current_oracle.zero_predictor_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                self.evaluation_relative_fisher_gain,
                self.evaluation_current_oracle.relative_fisher_loss
                - self.evaluation_candidate_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("residual-spectrum site marginal is invalid")

    @property
    def strictly_improves(self) -> bool:
        return (
            self.frozen_relative_fisher_gain
            > float(self.minimum_relative_fisher_gain)
            and self.proposal_fit_relative_fisher_gain
            > float(self.minimum_relative_fisher_gain)
            and
            self.evaluation_relative_fisher_gain
            > float(self.minimum_relative_fisher_gain)
        )

    @property
    def proposal_fit_relative_fisher_gain(self) -> float:
        return float(
            self.proposal_fit_current_oracle.relative_fisher_loss
            - self.proposal_fit_candidate_oracle.relative_fisher_loss
        )

    @property
    def evaluation_fisher_loss_sum_gain(self) -> float:
        rows = self.evaluation_current_oracle.output_metrics.row_count
        return float(
            rows
            * (
                self.evaluation_current_oracle.output_metrics.fisher_loss
                - self.evaluation_candidate_oracle.output_metrics.fisher_loss
            )
        )

    @property
    def proposed_left_vectors(self) -> Tensor:
        return self.proposal_bank.allocated_residual_left_vectors[-1].clone()

    @property
    def proposed_right_vectors(self) -> Tensor:
        return self.proposal_bank.allocated_residual_right_vectors[-1].clone()


def compute_site_residual_spectrum_marginal(
    bank: AllLayerResidualSpectrumGeneratorBank,
    proposal_captures: Sequence[AttentionTransportBatch],
    evaluation_captures: Sequence[AttentionTransportBatch],
    *,
    config: AllLayerResidualSpectrumConfig,
    progress_callback: ResidualSpectrumProgressCallback | None = None,
) -> ResidualSpectrumSiteMarginal:
    """Build one site proposal on one panel and reselect on another panel."""

    if not isinstance(bank, AllLayerResidualSpectrumGeneratorBank):
        raise TypeError("bank must be an all-layer residual spectrum bank")
    if not isinstance(config, AllLayerResidualSpectrumConfig):
        raise TypeError("config must be an all-layer residual spectrum config")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    config.validate_bank(bank)
    invocation_id, fit_captures = _site_captures(proposal_captures)
    _evaluation_id, heldout_captures = _site_captures(
        evaluation_captures,
        invocation_id=invocation_id,
    )
    if invocation_id not in bank.hierarchy.invocation_ids:
        raise ValueError("site marginal invocation is absent from bank hierarchy")
    coefficients_by_capture = _exact_sparse_coefficients_by_capture(
        bank,
        fit_captures,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    coefficients = torch.cat(coefficients_by_capture)
    contextual = torch.cat(
        tuple(
            bank.contextual_inputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=capture.head_contexts,
            )
            for capture in fit_captures
        )
    )
    baseline_candidates = torch.cat(
        tuple(bank.candidate_outputs(capture) for capture in fit_captures)
    )
    baseline_prediction = torch.einsum(
        "nk,nkh->nh",
        coefficients,
        baseline_candidates,
    ).contiguous()
    target = torch.cat(
        tuple(capture.valid_operator_outputs for capture in fit_captures)
    )
    score_gradient = torch.cat(
        tuple(capture.valid_score_gradients for capture in fit_captures)
    )
    differentiable_prediction = baseline_prediction.clone().requires_grad_(True)
    frozen_current_loss_tensor = samplewise_joint_fisher_loss(
        differentiable_prediction,
        target,
        score_gradient,
        euclidean_floor=config.euclidean_floor,
    )
    output_gradient = torch.autograd.grad(
        frozen_current_loss_tensor,
        differentiable_prediction,
    )[0].detach()

    left_directions = []
    right_directions = []
    singular_values = []
    projection_ranks = []
    for generator_index in range(bank.generator_count):
        left, right, singular, projection_rank = _matrix_free_top_singular_pair(
            contextual,
            output_gradient,
            coefficients,
            bank_sha256=bank.artifact_sha256,
            invocation_id=invocation_id,
            generator_index=generator_index,
            prior_rights=_prior_right_vectors(
                bank,
                invocation_id=invocation_id,
                generator_index=generator_index,
            ),
            config=config,
        )
        left_directions.append(left)
        right_directions.append(right)
        singular_values.append(singular)
        projection_ranks.append(projection_rank)
        _notify_progress(
            progress_callback,
            "site_power_iteration",
            generator_index + 1,
            bank.generator_count,
            invocation_id,
        )
    left_direction = torch.stack(tuple(left_directions))
    right_direction = torch.stack(tuple(right_directions))
    activation = torch.einsum("nkh,kh->nk", contextual, right_direction)
    directional_candidates = (
        coefficients[:, :, None]
        * activation[:, :, None]
        * left_direction[None]
    ).contiguous()
    amplitudes = _exact_fisher_amplitudes(
        directional_candidates,
        baseline_prediction,
        target,
        score_gradient,
        euclidean_floor=config.euclidean_floor,
        amplitude_ridge=config.amplitude_ridge,
    )
    proposed_left = amplitudes[:, None] * left_direction
    proposal_bank = _append_allocated_component(
        bank,
        invocation_id=invocation_id,
        left=proposed_left,
        right=right_direction,
    )
    frozen_candidate_prediction = (
        baseline_prediction
        + torch.einsum("k,nkh->nh", amplitudes, directional_candidates)
    ).contiguous()
    # The directional tensor already contains beta.  The allocated bank must
    # reproduce that fixed-coefficient prediction up to summation roundoff.
    proposal_candidates = torch.cat(
        tuple(proposal_bank.candidate_outputs(capture) for capture in fit_captures)
    )
    replay_prediction = torch.einsum(
        "nk,nkh->nh",
        coefficients,
        proposal_candidates,
    ).contiguous()
    if not torch.allclose(
        replay_prediction,
        frozen_candidate_prediction,
        atol=2.0e-12,
        rtol=2.0e-12,
    ):
        raise RuntimeError("residual-spectrum amplitude replay does not close")
    frozen_current_loss = float(frozen_current_loss_tensor.detach())
    frozen_candidate_loss = float(
        samplewise_joint_fisher_loss(
            replay_prediction,
            target,
            score_gradient,
            euclidean_floor=config.euclidean_floor,
        )
    )
    tolerance = 1.0e-11 * max(1.0, abs(frozen_current_loss))
    if frozen_candidate_loss > frozen_current_loss + tolerance:
        raise RuntimeError("exact Fisher amplitude solve regressed frozen support")
    fit_current = evaluate_sparse_attention_transport_bank(
        bank,
        fit_captures,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    fit_candidate = evaluate_sparse_attention_transport_bank(
        proposal_bank,
        fit_captures,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    if not math.isclose(
        fit_current.output_metrics.fisher_loss,
        frozen_current_loss,
        rel_tol=1.0e-11,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("frozen proposal coefficients are not the exact oracle")
    evaluation_current = evaluate_sparse_attention_transport_bank(
        bank,
        heldout_captures,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    evaluation_candidate = evaluate_sparse_attention_transport_bank(
        proposal_bank,
        heldout_captures,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    return ResidualSpectrumSiteMarginal(
        invocation_id=invocation_id,
        source_bank_sha256=bank.artifact_sha256,
        proposal_bank=proposal_bank,
        proposal_fit_current_oracle=fit_current,
        proposal_fit_candidate_oracle=fit_candidate,
        evaluation_current_oracle=evaluation_current,
        evaluation_candidate_oracle=evaluation_candidate,
        frozen_current_fisher_loss=frozen_current_loss,
        frozen_candidate_fisher_loss=frozen_candidate_loss,
        frozen_relative_fisher_gain=(
            frozen_current_loss - frozen_candidate_loss
        )
        / fit_current.zero_predictor_fisher_loss,
        evaluation_relative_fisher_gain=(
            evaluation_current.relative_fisher_loss
            - evaluation_candidate.relative_fisher_loss
        ),
        generator_singular_values=tuple(float(value) for value in singular_values),
        generator_amplitudes=tuple(float(value) for value in amplitudes),
        right_projection_ranks=tuple(projection_ranks),
        minimum_relative_fisher_gain=config.minimum_relative_fisher_gain,
    )


def _all_layer_capture_groups(
    captures: Sequence[AttentionTransportBatch],
    invocation_ids: tuple[str, ...],
) -> tuple[
    tuple[AttentionTransportBatch, ...],
    dict[str, tuple[AttentionTransportBatch, ...]],
]:
    values = _capture_values(captures)
    observed = {capture.invocation_id for capture in values}
    if observed != set(invocation_ids):
        raise ValueError("residual spectrum requires captures from all 18 layers")
    return values, {
        invocation_id: tuple(
            capture
            for capture in values
            if capture.invocation_id == invocation_id
        )
        for invocation_id in invocation_ids
    }


@dataclass(frozen=True, slots=True)
class AllLayerFirstResidualSpectrum:
    """The first independently proposed residual marginal at every layer."""

    source_bank_sha256: str
    invocation_ids: tuple[str, ...]
    proposal_fit_current_oracle: SparseAttentionTransportOracleEvaluation
    evaluation_current_oracle: SparseAttentionTransportOracleEvaluation
    per_site: tuple[ResidualSpectrumSiteMarginal, ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.source_bank_sha256)
            or len(self.invocation_ids) != 18
            or len(set(self.invocation_ids)) != 18
            or not isinstance(
                self.proposal_fit_current_oracle,
                SparseAttentionTransportOracleEvaluation,
            )
            or not isinstance(
                self.evaluation_current_oracle,
                SparseAttentionTransportOracleEvaluation,
            )
            or len(self.per_site) != 18
            or any(
                not isinstance(value, ResidualSpectrumSiteMarginal)
                for value in self.per_site
            )
            or tuple(value.invocation_id for value in self.per_site)
            != self.invocation_ids
            or any(
                value.source_bank_sha256 != self.source_bank_sha256
                for value in self.per_site
            )
        ):
            raise ValueError("all-layer first residual spectrum is invalid")
        fit_by_site = dict(
            self.proposal_fit_current_oracle.per_invocation_relative_fisher_loss
        )
        evaluation_by_site = dict(
            self.evaluation_current_oracle.per_invocation_relative_fisher_loss
        )
        if any(
            not math.isclose(
                marginal.proposal_fit_current_oracle.relative_fisher_loss,
                fit_by_site[marginal.invocation_id],
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                marginal.evaluation_current_oracle.relative_fisher_loss,
                evaluation_by_site[marginal.invocation_id],
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            for marginal in self.per_site
        ):
            raise ValueError("all-layer first-spectrum baselines do not close")

    @property
    def evaluation_all_layer_normalized_fisher_gains(
        self,
    ) -> tuple[tuple[str, float], ...]:
        rows = self.evaluation_current_oracle.output_metrics.row_count
        zero = self.evaluation_current_oracle.zero_predictor_fisher_loss
        return tuple(
            (
                marginal.invocation_id,
                marginal.evaluation_fisher_loss_sum_gain / rows / zero,
            )
            for marginal in self.per_site
        )


def compute_all_layer_first_residual_spectrum(
    bank: AllLayerResidualSpectrumGeneratorBank,
    proposal_captures: Sequence[AttentionTransportBatch],
    evaluation_captures: Sequence[AttentionTransportBatch],
    *,
    config: AllLayerResidualSpectrumConfig,
    progress_callback: ResidualSpectrumProgressCallback | None = None,
) -> AllLayerFirstResidualSpectrum:
    """Compute the full 18-site marginal spectrum in numeric layer order."""

    if not isinstance(bank, AllLayerResidualSpectrumGeneratorBank):
        raise TypeError("bank must be an all-layer residual spectrum bank")
    config.validate_bank(bank)
    invocation_ids = tuple(bank.hierarchy.invocation_ids)
    fit_values, fit_groups = _all_layer_capture_groups(
        proposal_captures,
        invocation_ids,
    )
    evaluation_values, evaluation_groups = _all_layer_capture_groups(
        evaluation_captures,
        invocation_ids,
    )
    current_fit = evaluate_sparse_attention_transport_bank(
        bank,
        fit_values,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    current_evaluation = evaluate_sparse_attention_transport_bank(
        bank,
        evaluation_values,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    marginals = []
    for site_index, invocation_id in enumerate(invocation_ids):
        marginal = compute_site_residual_spectrum_marginal(
            bank,
            fit_groups[invocation_id],
            evaluation_groups[invocation_id],
            config=config,
            progress_callback=progress_callback,
        )
        marginals.append(marginal)
        _notify_progress(
            progress_callback,
            "first_spectrum_site",
            site_index + 1,
            len(invocation_ids),
            invocation_id,
        )
    return AllLayerFirstResidualSpectrum(
        source_bank_sha256=bank.artifact_sha256,
        invocation_ids=invocation_ids,
        proposal_fit_current_oracle=current_fit,
        evaluation_current_oracle=current_evaluation,
        per_site=tuple(marginals),
    )


@dataclass(frozen=True, slots=True)
class ResidualSpectrumAllocationStep:
    """One accepted greedy or fixed-profile allocation step."""

    allocation_index: int
    invocation_id: str
    source_bank_sha256: str
    result_bank_sha256: str
    marginal_source_bank_sha256: str
    used_cached_off_site_marginal: bool
    selection_all_layer_normalized_fisher_gain: float
    marginal: ResidualSpectrumSiteMarginal
    proposal_fit_before: SparseAttentionTransportOracleEvaluation
    proposal_fit_after: SparseAttentionTransportOracleEvaluation
    evaluation_before: SparseAttentionTransportOracleEvaluation
    evaluation_after: SparseAttentionTransportOracleEvaluation

    def __post_init__(self) -> None:
        evaluations = (
            self.proposal_fit_before,
            self.proposal_fit_after,
            self.evaluation_before,
            self.evaluation_after,
        )
        if (
            type(self.allocation_index) is not int
            or self.allocation_index < 0
            or self.invocation_id != self.marginal.invocation_id
            or not _is_sha256(self.source_bank_sha256)
            or not _is_sha256(self.result_bank_sha256)
            or self.source_bank_sha256 == self.result_bank_sha256
            or self.marginal_source_bank_sha256
            != self.marginal.source_bank_sha256
            or type(self.used_cached_off_site_marginal) is not bool
            or not math.isfinite(
                float(self.selection_all_layer_normalized_fisher_gain)
            )
            or float(self.selection_all_layer_normalized_fisher_gain)
            <= float(self.marginal.minimum_relative_fisher_gain)
            or not isinstance(self.marginal, ResidualSpectrumSiteMarginal)
            or not self.marginal.strictly_improves
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in evaluations
            )
            or self.proposal_fit_after.relative_fisher_loss
            > self.proposal_fit_before.relative_fisher_loss + 1.0e-12
            or self.evaluation_after.relative_fisher_loss
            >= self.evaluation_before.relative_fisher_loss
            - float(self.marginal.minimum_relative_fisher_gain)
            or not math.isclose(
                float(self.selection_all_layer_normalized_fisher_gain),
                self.evaluation_before.relative_fisher_loss
                - self.evaluation_after.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("residual-spectrum allocation step is invalid")


@dataclass(frozen=True, slots=True)
class ResidualSpectrumAllocationRound:
    """Independently checkable 18-site cache snapshot before one decision."""

    allocation_index: int
    current_bank_sha256: str
    evaluation_row_count: int
    evaluation_zero_predictor_fisher_loss: float
    marginals: tuple[ResidualSpectrumSiteMarginal, ...]
    normalized_fisher_mass_gains: tuple[tuple[str, float], ...]
    selected_invocation_id: str | None
    minimum_relative_fisher_gain: float
    tie_tolerance: float

    def __post_init__(self) -> None:
        if (
            type(self.allocation_index) is not int
            or self.allocation_index < 0
            or not _is_sha256(self.current_bank_sha256)
            or type(self.evaluation_row_count) is not int
            or self.evaluation_row_count <= 0
            or isinstance(self.evaluation_zero_predictor_fisher_loss, bool)
            or not isinstance(
                self.evaluation_zero_predictor_fisher_loss,
                (int, float),
            )
            or not math.isfinite(float(self.evaluation_zero_predictor_fisher_loss))
            or float(self.evaluation_zero_predictor_fisher_loss) <= 0.0
            or len(self.marginals) != 18
            or any(
                not isinstance(value, ResidualSpectrumSiteMarginal)
                for value in self.marginals
            )
            or tuple(value.invocation_id for value in self.marginals)
            != tuple(
                sorted(
                    (value.invocation_id for value in self.marginals),
                    key=_mutation_ordinal,
                )
            )
            or tuple(key for key, _value in self.normalized_fisher_mass_gains)
            != tuple(value.invocation_id for value in self.marginals)
            or any(
                not math.isfinite(float(value))
                for _key, value in self.normalized_fisher_mass_gains
            )
            or (
                self.selected_invocation_id is not None
                and self.selected_invocation_id
                not in {value.invocation_id for value in self.marginals}
            )
            or isinstance(self.minimum_relative_fisher_gain, bool)
            or not isinstance(self.minimum_relative_fisher_gain, (int, float))
            or not math.isfinite(float(self.minimum_relative_fisher_gain))
            or float(self.minimum_relative_fisher_gain) <= 0.0
            or isinstance(self.tie_tolerance, bool)
            or not isinstance(self.tie_tolerance, (int, float))
            or not math.isfinite(float(self.tie_tolerance))
            or float(self.tie_tolerance) < 0.0
            or any(
                float(value.minimum_relative_fisher_gain)
                != float(self.minimum_relative_fisher_gain)
                for value in self.marginals
            )
        ):
            raise ValueError("residual-spectrum allocation round is invalid")
        gains = dict(self.normalized_fisher_mass_gains)
        for marginal in self.marginals:
            expected = (
                marginal.evaluation_fisher_loss_sum_gain
                / self.evaluation_row_count
                / float(self.evaluation_zero_predictor_fisher_loss)
            )
            if gains[marginal.invocation_id] != expected:
                raise ValueError("allocation-round normalized gain does not close")
        eligible = tuple(
            value
            for value in self.marginals
            if value.strictly_improves
            and gains[value.invocation_id]
            > float(self.minimum_relative_fisher_gain)
        )
        if not eligible:
            if self.selected_invocation_id is not None:
                raise ValueError("allocation round selected an ineligible candidate")
            return
        if self.selected_invocation_id is None:
            raise ValueError("allocation round omitted an eligible candidate")
        best_gain = max(gains[value.invocation_id] for value in eligible)
        tied = tuple(
            value
            for value in eligible
            if gains[value.invocation_id]
            >= best_gain - float(self.tie_tolerance)
        )
        expected_selected = min(
            tied,
            key=lambda value: _mutation_ordinal(value.invocation_id),
        ).invocation_id
        if self.selected_invocation_id != expected_selected:
            raise ValueError("allocation round did not select the numeric argmax")

    @property
    def source_bank_sha256(self) -> str:
        """Alias emphasizing that this is the source of the round decision."""

        return self.current_bank_sha256


@dataclass(frozen=True, slots=True)
class GreedyResidualSpectrumAllocation:
    """Two-step inner-panel greedy allocation with cached off-site marginals."""

    initial_bank_sha256: str
    final_bank: AllLayerResidualSpectrumGeneratorBank
    allocation_profile: tuple[str, ...]
    first_spectrum: AllLayerFirstResidualSpectrum
    steps: tuple[ResidualSpectrumAllocationStep, ...]
    candidate_rounds: tuple[ResidualSpectrumAllocationRound, ...]
    initial_proposal_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_proposal_fit_oracle: SparseAttentionTransportOracleEvaluation
    initial_evaluation_oracle: SparseAttentionTransportOracleEvaluation
    final_evaluation_oracle: SparseAttentionTransportOracleEvaluation
    proposal_fit_relative_fisher_loss_history: tuple[float, ...]
    evaluation_relative_fisher_loss_history: tuple[float, ...]
    allocation_budget: int

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.initial_bank_sha256)
            or not isinstance(self.final_bank, AllLayerResidualSpectrumGeneratorBank)
            or type(self.allocation_budget) is not int
            or self.allocation_budget <= 0
            or len(self.allocation_profile) > self.allocation_budget
            or len(self.steps) != len(self.allocation_profile)
            or tuple(step.allocation_index for step in self.steps)
            != tuple(range(len(self.steps)))
            or tuple(step.invocation_id for step in self.steps)
            != self.allocation_profile
            or any(
                not isinstance(value, ResidualSpectrumAllocationRound)
                for value in self.candidate_rounds
            )
            or tuple(value.allocation_index for value in self.candidate_rounds)
            != tuple(range(len(self.candidate_rounds)))
            or self.final_bank.allocated_invocation_ids
            != self.allocation_profile
            or not isinstance(self.first_spectrum, AllLayerFirstResidualSpectrum)
            or self.first_spectrum.source_bank_sha256 != self.initial_bank_sha256
            or len(self.proposal_fit_relative_fisher_loss_history)
            != len(self.allocation_profile) + 1
            or len(self.evaluation_relative_fisher_loss_history)
            != len(self.allocation_profile) + 1
            or self.proposal_fit_relative_fisher_loss_history[0]
            != self.initial_proposal_fit_oracle.relative_fisher_loss
            or self.proposal_fit_relative_fisher_loss_history[-1]
            != self.final_proposal_fit_oracle.relative_fisher_loss
            or self.evaluation_relative_fisher_loss_history[0]
            != self.initial_evaluation_oracle.relative_fisher_loss
            or self.evaluation_relative_fisher_loss_history[-1]
            != self.final_evaluation_oracle.relative_fisher_loss
            or any(
                right > left + 1.0e-12
                for history in (
                    self.proposal_fit_relative_fisher_loss_history,
                    self.evaluation_relative_fisher_loss_history,
                )
                for left, right in zip(history, history[1:])
            )
        ):
            raise ValueError("greedy residual-spectrum allocation is invalid")
        exhausted = len(self.steps) == self.allocation_budget
        expected_round_count = len(self.steps) if exhausted else len(self.steps) + 1
        if (
            len(self.candidate_rounds) != expected_round_count
            or (
                self.candidate_rounds
                and self.candidate_rounds[0].current_bank_sha256
                != self.initial_bank_sha256
            )
            or tuple(
                value.selected_invocation_id
                for value in self.candidate_rounds[: len(self.steps)]
            )
            != self.allocation_profile
            or any(
                round_value.current_bank_sha256 != step.source_bank_sha256
                for round_value, step in zip(
                    self.candidate_rounds,
                    self.steps,
                    strict=False,
                )
            )
            or (
                not exhausted
                and (
                    not self.candidate_rounds
                    or self.candidate_rounds[-1].selected_invocation_id is not None
                    or self.candidate_rounds[-1].current_bank_sha256
                    != self.final_bank.artifact_sha256
                )
            )
            or (
                self.steps
                and self.steps[-1].result_bank_sha256
                != self.final_bank.artifact_sha256
            )
            or (
                not self.steps
                and self.final_bank.artifact_sha256 != self.initial_bank_sha256
            )
        ):
            raise ValueError("greedy residual-spectrum round lineage is invalid")
        first_round = self.candidate_rounds[0]
        if (
            tuple(value.invocation_id for value in first_round.marginals)
            != self.first_spectrum.invocation_ids
            or tuple(
                value.proposal_bank.artifact_sha256
                for value in first_round.marginals
            )
            != tuple(
                value.proposal_bank.artifact_sha256
                for value in self.first_spectrum.per_site
            )
        ):
            raise ValueError("first allocation round does not match first spectrum")

    @property
    def budget_exhausted(self) -> bool:
        return len(self.allocation_profile) == self.allocation_budget

    @property
    def allocation_profile_sha256(self) -> str:
        return residual_allocation_profile_sha256(self.allocation_profile)


def _apply_cached_marginal(
    bank: AllLayerResidualSpectrumGeneratorBank,
    marginal: ResidualSpectrumSiteMarginal,
) -> AllLayerResidualSpectrumGeneratorBank:
    return _append_allocated_component(
        bank,
        invocation_id=marginal.invocation_id,
        left=marginal.proposed_left_vectors,
        right=marginal.proposed_right_vectors,
    )


def _residual_spectrum_allocation_round(
    *,
    allocation_index: int,
    current_bank: AllLayerResidualSpectrumGeneratorBank,
    current_evaluation: SparseAttentionTransportOracleEvaluation,
    cache: Mapping[str, ResidualSpectrumSiteMarginal],
    invocation_ids: tuple[str, ...],
    config: AllLayerResidualSpectrumConfig,
) -> ResidualSpectrumAllocationRound:
    marginals = tuple(cache[invocation_id] for invocation_id in invocation_ids)
    rows = current_evaluation.output_metrics.row_count
    zero = current_evaluation.zero_predictor_fisher_loss
    gains = tuple(
        (
            marginal.invocation_id,
            marginal.evaluation_fisher_loss_sum_gain / rows / zero,
        )
        for marginal in marginals
    )
    gain_by_site = dict(gains)
    eligible = tuple(
        value
        for value in marginals
        if value.strictly_improves
        and gain_by_site[value.invocation_id]
        > float(config.minimum_relative_fisher_gain)
    )
    selected: str | None = None
    if eligible:
        best_gain = max(gain_by_site[value.invocation_id] for value in eligible)
        tied = tuple(
            value
            for value in eligible
            if gain_by_site[value.invocation_id]
            >= best_gain - float(config.tie_tolerance)
        )
        selected = min(
            tied,
            key=lambda value: _mutation_ordinal(value.invocation_id),
        ).invocation_id
    return ResidualSpectrumAllocationRound(
        allocation_index=allocation_index,
        current_bank_sha256=current_bank.artifact_sha256,
        evaluation_row_count=rows,
        evaluation_zero_predictor_fisher_loss=zero,
        marginals=marginals,
        normalized_fisher_mass_gains=gains,
        selected_invocation_id=selected,
        minimum_relative_fisher_gain=config.minimum_relative_fisher_gain,
        tie_tolerance=config.tie_tolerance,
    )


def greedily_allocate_residual_spectrum(
    parent_bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    proposal_captures: Sequence[AttentionTransportBatch],
    evaluation_captures: Sequence[AttentionTransportBatch],
    *,
    config: AllLayerResidualSpectrumConfig,
    progress_callback: ResidualSpectrumProgressCallback | None = None,
) -> GreedyResidualSpectrumAllocation:
    """Greedily allocate a fixed budget, caching every unaffected marginal."""

    if not isinstance(
        parent_bank,
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("parent_bank must be a global site rank-one bank")
    config.validate_bank(parent_bank)
    current = all_layer_residual_spectrum_generator_bank_from_parent(parent_bank)
    invocation_ids = tuple(current.hierarchy.invocation_ids)
    fit_values, fit_groups = _all_layer_capture_groups(
        proposal_captures,
        invocation_ids,
    )
    evaluation_values, evaluation_groups = _all_layer_capture_groups(
        evaluation_captures,
        invocation_ids,
    )
    first_spectrum = compute_all_layer_first_residual_spectrum(
        current,
        fit_values,
        evaluation_values,
        config=config,
        progress_callback=progress_callback,
    )
    cache = {value.invocation_id: value for value in first_spectrum.per_site}
    initial_fit = first_spectrum.proposal_fit_current_oracle
    initial_evaluation = first_spectrum.evaluation_current_oracle
    current_fit = initial_fit
    current_evaluation = initial_evaluation
    fit_history = [float(initial_fit.relative_fisher_loss)]
    evaluation_history = [float(initial_evaluation.relative_fisher_loss)]
    steps = []
    profile = []
    candidate_rounds = []

    for allocation_index in range(config.allocation_budget):
        allocation_round = _residual_spectrum_allocation_round(
            allocation_index=allocation_index,
            current_bank=current,
            current_evaluation=current_evaluation,
            cache=cache,
            invocation_ids=invocation_ids,
            config=config,
        )
        candidate_rounds.append(allocation_round)
        if allocation_round.selected_invocation_id is None:
            break
        selected = cache[allocation_round.selected_invocation_id]
        allocation_gain = dict(allocation_round.normalized_fisher_mass_gains)[
            selected.invocation_id
        ]
        source_sha = current.artifact_sha256
        used_cached = selected.source_bank_sha256 != source_sha
        candidate = (
            _apply_cached_marginal(current, selected)
            if used_cached
            else selected.proposal_bank
        )
        site_fit_before = evaluate_sparse_attention_transport_bank(
            current,
            fit_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        site_fit_after = evaluate_sparse_attention_transport_bank(
            candidate,
            fit_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        site_evaluation_before = evaluate_sparse_attention_transport_bank(
            current,
            evaluation_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        site_evaluation_after = evaluate_sparse_attention_transport_bank(
            candidate,
            evaluation_groups[selected.invocation_id],
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        observed_gain = (
            site_evaluation_before.relative_fisher_loss
            - site_evaluation_after.relative_fisher_loss
        )
        if (
            observed_gain <= config.minimum_relative_fisher_gain
            or not math.isclose(
                observed_gain,
                selected.evaluation_relative_fisher_gain,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError("cached residual-spectrum marginal did not replay")
        candidate_fit = evaluate_sparse_attention_transport_bank(
            candidate,
            fit_values,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        candidate_evaluation = evaluate_sparse_attention_transport_bank(
            candidate,
            evaluation_values,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        if (
            candidate_fit.relative_fisher_loss
            > current_fit.relative_fisher_loss + 1.0e-12
            or candidate_evaluation.relative_fisher_loss
            >= current_evaluation.relative_fisher_loss
            - float(config.minimum_relative_fisher_gain)
        ):
            raise RuntimeError("accepted greedy allocation regressed an aggregate")
        steps.append(
            ResidualSpectrumAllocationStep(
                allocation_index=allocation_index,
                invocation_id=selected.invocation_id,
                source_bank_sha256=source_sha,
                result_bank_sha256=candidate.artifact_sha256,
                marginal_source_bank_sha256=selected.source_bank_sha256,
                used_cached_off_site_marginal=used_cached,
                selection_all_layer_normalized_fisher_gain=allocation_gain,
                marginal=selected,
                proposal_fit_before=current_fit,
                proposal_fit_after=candidate_fit,
                evaluation_before=current_evaluation,
                evaluation_after=candidate_evaluation,
            )
        )
        profile.append(selected.invocation_id)
        current = candidate
        current_fit = candidate_fit
        current_evaluation = candidate_evaluation
        fit_history.append(float(current_fit.relative_fisher_loss))
        evaluation_history.append(float(current_evaluation.relative_fisher_loss))
        _notify_progress(
            progress_callback,
            "greedy_allocation",
            allocation_index + 1,
            config.allocation_budget,
            selected.invocation_id,
        )
        if allocation_index + 1 < config.allocation_budget:
            # Only the selected site's candidate space changed.  Every off-site
            # candidate and held-out marginal remains exactly cacheable.
            cache[selected.invocation_id] = compute_site_residual_spectrum_marginal(
                current,
                fit_groups[selected.invocation_id],
                evaluation_groups[selected.invocation_id],
                config=config,
                progress_callback=progress_callback,
            )

    return GreedyResidualSpectrumAllocation(
        initial_bank_sha256=first_spectrum.source_bank_sha256,
        final_bank=current,
        allocation_profile=tuple(profile),
        first_spectrum=first_spectrum,
        steps=tuple(steps),
        candidate_rounds=tuple(candidate_rounds),
        initial_proposal_fit_oracle=initial_fit,
        final_proposal_fit_oracle=current_fit,
        initial_evaluation_oracle=initial_evaluation,
        final_evaluation_oracle=current_evaluation,
        proposal_fit_relative_fisher_loss_history=tuple(fit_history),
        evaluation_relative_fisher_loss_history=tuple(evaluation_history),
        allocation_budget=config.allocation_budget,
    )


@dataclass(frozen=True, slots=True)
class FixedProfileResidualSpectrumFit:
    """Full-fit replay of a sealed allocation profile with fail-closed steps."""

    initial_bank_sha256: str
    final_bank: AllLayerResidualSpectrumGeneratorBank
    allocation_profile: tuple[str, ...]
    steps: tuple[ResidualSpectrumAllocationStep, ...]
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_relative_fisher_loss_history: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.initial_bank_sha256)
            or not isinstance(self.final_bank, AllLayerResidualSpectrumGeneratorBank)
            or len(self.steps) != len(self.allocation_profile)
            or tuple(step.allocation_index for step in self.steps)
            != tuple(range(len(self.steps)))
            or tuple(step.invocation_id for step in self.steps)
            != self.allocation_profile
            or self.final_bank.allocated_invocation_ids
            != self.allocation_profile
            or len(self.fit_relative_fisher_loss_history)
            != len(self.allocation_profile) + 1
            or self.fit_relative_fisher_loss_history[0]
            != self.initial_fit_oracle.relative_fisher_loss
            or self.fit_relative_fisher_loss_history[-1]
            != self.final_fit_oracle.relative_fisher_loss
            or any(
                right >= left
                for left, right in zip(
                    self.fit_relative_fisher_loss_history,
                    self.fit_relative_fisher_loss_history[1:],
                )
            )
        ):
            raise ValueError("fixed-profile residual-spectrum fit is invalid")
        if (
            self.steps
            and self.steps[-1].result_bank_sha256
            != self.final_bank.artifact_sha256
        ) or (
            not self.steps
            and self.final_bank.artifact_sha256 != self.initial_bank_sha256
        ):
            raise ValueError("fixed-profile residual-spectrum lineage is invalid")

    @property
    def allocated_component_count(self) -> int:
        return len(self.allocation_profile)

    @property
    def allocation_profile_sha256(self) -> str:
        return residual_allocation_profile_sha256(self.allocation_profile)


def replay_fixed_residual_allocation_profile(
    parent_bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    full_fit_captures: Sequence[AttentionTransportBatch],
    allocation_profile: Sequence[str],
    *,
    config: AllLayerResidualSpectrumConfig,
    progress_callback: ResidualSpectrumProgressCallback | None = None,
) -> FixedProfileResidualSpectrumFit:
    """Refit only a preselected site sequence on the full fit panel.

    Site identity is fixed before this function runs.  Every newly computed
    component must strictly improve both frozen and freshly reselected exact
    Top-2 objectives; otherwise replay aborts without returning an artifact.
    """

    if isinstance(allocation_profile, (str, bytes)) or not isinstance(
        allocation_profile,
        Sequence,
    ):
        raise TypeError("allocation_profile must be a materialized sequence")
    profile = tuple(allocation_profile)
    if len(profile) > config.allocation_budget:
        raise ValueError("fixed allocation profile exceeds the budget")
    config.validate_bank(parent_bank)
    current = all_layer_residual_spectrum_generator_bank_from_parent(parent_bank)
    invocation_ids = tuple(current.hierarchy.invocation_ids)
    if any(value not in invocation_ids for value in profile):
        raise ValueError("fixed allocation profile contains an unknown invocation")
    fit_values, fit_groups = _all_layer_capture_groups(
        full_fit_captures,
        invocation_ids,
    )
    initial = evaluate_sparse_attention_transport_bank(
        current,
        fit_values,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.coefficient_ridge,
    )
    current_evaluation = initial
    history = [float(initial.relative_fisher_loss)]
    steps = []
    for allocation_index, invocation_id in enumerate(profile):
        marginal = compute_site_residual_spectrum_marginal(
            current,
            fit_groups[invocation_id],
            fit_groups[invocation_id],
            config=config,
            progress_callback=progress_callback,
        )
        if not marginal.strictly_improves:
            raise RuntimeError(
                "fixed-profile residual component failed strict improvement"
            )
        candidate = marginal.proposal_bank
        candidate_evaluation = evaluate_sparse_attention_transport_bank(
            candidate,
            fit_values,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )
        aggregate_gain = (
            current_evaluation.relative_fisher_loss
            - candidate_evaluation.relative_fisher_loss
        )
        if aggregate_gain <= float(config.minimum_relative_fisher_gain):
            raise RuntimeError(
                "fixed-profile replay failed the all-layer improvement threshold"
            )
        steps.append(
            ResidualSpectrumAllocationStep(
                allocation_index=allocation_index,
                invocation_id=invocation_id,
                source_bank_sha256=current.artifact_sha256,
                result_bank_sha256=candidate.artifact_sha256,
                marginal_source_bank_sha256=marginal.source_bank_sha256,
                used_cached_off_site_marginal=False,
                selection_all_layer_normalized_fisher_gain=aggregate_gain,
                marginal=marginal,
                proposal_fit_before=current_evaluation,
                proposal_fit_after=candidate_evaluation,
                evaluation_before=current_evaluation,
                evaluation_after=candidate_evaluation,
            )
        )
        current = candidate
        current_evaluation = candidate_evaluation
        history.append(float(current_evaluation.relative_fisher_loss))
        _notify_progress(
            progress_callback,
            "fixed_profile_replay",
            allocation_index + 1,
            len(profile),
            invocation_id,
        )
    return FixedProfileResidualSpectrumFit(
        initial_bank_sha256=(
            all_layer_residual_spectrum_generator_bank_from_parent(
                parent_bank
            ).artifact_sha256
        ),
        final_bank=current,
        allocation_profile=profile,
        steps=tuple(steps),
        initial_fit_oracle=initial,
        final_fit_oracle=current_evaluation,
        fit_relative_fisher_loss_history=tuple(history),
    )


@dataclass(frozen=True, slots=True)
class ResidualSpectrumBankAllLayerEvaluation:
    """Exact sparse all/local/global and numeric per-layer metrics for one bank."""

    bank_kind: str
    all_layers: SparseAttentionTransportOracleEvaluation
    local_attention: SparseAttentionTransportOracleEvaluation
    global_attention: SparseAttentionTransportOracleEvaluation
    per_invocation: tuple[
        tuple[str, SparseAttentionTransportOracleEvaluation], ...
    ]

    def __post_init__(self) -> None:
        if (
            self.bank_kind not in ("parent_site_rank_one", "allocated_residual")
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in (
                    self.all_layers,
                    self.local_attention,
                    self.global_attention,
                )
            )
            or len(self.per_invocation) != 18
            or len({key for key, _value in self.per_invocation}) != 18
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                or tuple(
                    invocation_id
                    for invocation_id, _loss in value.per_invocation_relative_fisher_loss
                )
                != (key,)
                for key, value in self.per_invocation
            )
            or self.local_attention.output_metrics.row_count
            + self.global_attention.output_metrics.row_count
            != self.all_layers.output_metrics.row_count
        ):
            raise ValueError("residual-spectrum all-layer bank evaluation is invalid")
        total_rows = self.all_layers.output_metrics.row_count
        weighted_fisher = sum(
            value.output_metrics.row_count * value.output_metrics.fisher_loss
            for _key, value in self.per_invocation
        ) / total_rows
        weighted_zero = sum(
            value.output_metrics.row_count * value.zero_predictor_fisher_loss
            for _key, value in self.per_invocation
        ) / total_rows
        if (
            not math.isclose(
                weighted_fisher,
                self.all_layers.output_metrics.fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                weighted_zero,
                self.all_layers.zero_predictor_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("residual-spectrum per-layer metrics do not close")

    @property
    def per_invocation_fisher_loss_sums(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (
                key,
                float(
                    value.output_metrics.row_count
                    * value.output_metrics.fisher_loss
                ),
            )
            for key, value in self.per_invocation
        )


@dataclass(frozen=True, slots=True)
class AllocatedResidualAllLayerDiagnostic:
    """Paired outer-panel comparison of the parent and allocated banks."""

    invocation_ids: tuple[str, ...]
    parent_bank_sha256: str
    allocated_bank_sha256: str
    allocation_profile: tuple[str, ...]
    active_generator_count_limit: int
    coefficient_ridge: float
    parent: ResidualSpectrumBankAllLayerEvaluation
    allocated: ResidualSpectrumBankAllLayerEvaluation

    def __post_init__(self) -> None:
        if (
            len(self.invocation_ids) != 18
            or len(set(self.invocation_ids)) != 18
            or not _is_sha256(self.parent_bank_sha256)
            or not _is_sha256(self.allocated_bank_sha256)
            or type(self.active_generator_count_limit) is not int
            or self.active_generator_count_limit <= 0
            or isinstance(self.coefficient_ridge, bool)
            or not isinstance(self.coefficient_ridge, (int, float))
            or not math.isfinite(float(self.coefficient_ridge))
            or float(self.coefficient_ridge) < 0.0
            or not isinstance(self.parent, ResidualSpectrumBankAllLayerEvaluation)
            or not isinstance(self.allocated, ResidualSpectrumBankAllLayerEvaluation)
            or self.parent.bank_kind != "parent_site_rank_one"
            or self.allocated.bank_kind != "allocated_residual"
            or tuple(key for key, _value in self.parent.per_invocation)
            != self.invocation_ids
            or tuple(key for key, _value in self.allocated.per_invocation)
            != self.invocation_ids
            or self.parent.all_layers.output_metrics.row_count
            != self.allocated.all_layers.output_metrics.row_count
        ):
            raise ValueError("allocated residual all-layer diagnostic is invalid")

    @property
    def per_invocation_all_layer_normalized_fisher_gains(
        self,
    ) -> tuple[tuple[str, float], ...]:
        total_rows = self.parent.all_layers.output_metrics.row_count
        zero = self.parent.all_layers.zero_predictor_fisher_loss
        parent = dict(self.parent.per_invocation)
        allocated = dict(self.allocated.per_invocation)
        return tuple(
            (
                invocation_id,
                float(
                    parent[invocation_id].output_metrics.row_count
                    * (
                        parent[invocation_id].output_metrics.fisher_loss
                        - allocated[invocation_id].output_metrics.fisher_loss
                    )
                    / total_rows
                    / zero
                ),
            )
            for invocation_id in self.invocation_ids
        )


def _evaluate_bank_all_layers(
    bank_kind: str,
    bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    captures: tuple[AttentionTransportBatch, ...],
    invocation_ids: tuple[str, ...],
    *,
    config: AllLayerResidualSpectrumConfig,
) -> ResidualSpectrumBankAllLayerEvaluation:
    local = tuple(
        capture
        for capture in captures
        if capture.invocation_id not in GLOBAL_ATTENTION_INVOCATION_IDS
    )
    global_values = tuple(
        capture
        for capture in captures
        if capture.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    )

    def evaluate(values: tuple[AttentionTransportBatch, ...]) -> SparseAttentionTransportOracleEvaluation:
        return evaluate_sparse_attention_transport_bank(
            bank,
            values,
            active_count=config.active_count,
            euclidean_floor=config.euclidean_floor,
            coefficient_ridge=config.coefficient_ridge,
        )

    return ResidualSpectrumBankAllLayerEvaluation(
        bank_kind=bank_kind,
        all_layers=evaluate(captures),
        local_attention=evaluate(local),
        global_attention=evaluate(global_values),
        per_invocation=tuple(
            (
                invocation_id,
                evaluate(
                    tuple(
                        capture
                        for capture in captures
                        if capture.invocation_id == invocation_id
                    )
                ),
            )
            for invocation_id in invocation_ids
        ),
    )


def evaluate_allocated_residual_bank_all_layers(
    parent_bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    allocated_bank: AllLayerResidualSpectrumGeneratorBank,
    outer_captures: Sequence[AttentionTransportBatch],
    *,
    config: AllLayerResidualSpectrumConfig,
    progress_callback: ResidualSpectrumProgressCallback | None = None,
) -> AllocatedResidualAllLayerDiagnostic:
    """Evaluate a frozen allocation once on an untouched all-layer panel."""

    config.validate_bank(parent_bank)
    config.validate_bank(allocated_bank)
    if (
        not isinstance(allocated_bank, AllLayerResidualSpectrumGeneratorBank)
        or allocated_bank.parent_site_rank_one_bank_sha256
        != parent_bank.artifact_sha256
    ):
        raise ValueError("allocated bank is not descended from the supplied parent")
    invocation_ids = tuple(parent_bank.hierarchy.invocation_ids)
    values, _groups = _all_layer_capture_groups(outer_captures, invocation_ids)
    parent_evaluation = _evaluate_bank_all_layers(
        "parent_site_rank_one",
        parent_bank,
        values,
        invocation_ids,
        config=config,
    )
    _notify_progress(
        progress_callback,
        "outer_evaluation",
        1,
        2,
        None,
    )
    allocated_evaluation = _evaluate_bank_all_layers(
        "allocated_residual",
        allocated_bank,
        values,
        invocation_ids,
        config=config,
    )
    _notify_progress(
        progress_callback,
        "outer_evaluation",
        2,
        2,
        None,
    )
    return AllocatedResidualAllLayerDiagnostic(
        invocation_ids=invocation_ids,
        parent_bank_sha256=parent_bank.artifact_sha256,
        allocated_bank_sha256=allocated_bank.artifact_sha256,
        allocation_profile=allocated_bank.allocated_invocation_ids,
        active_generator_count_limit=config.active_count,
        coefficient_ridge=config.coefficient_ridge,
        parent=parent_evaluation,
        allocated=allocated_evaluation,
    )


__all__ = [
    "ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS",
    "ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS",
    "ALL_LAYER_RESIDUAL_COMPONENT_SEMANTICS",
    "ALL_LAYER_RESIDUAL_GAUGE_SEMANTICS",
    "ALL_LAYER_RESIDUAL_GREEDY_SEMANTICS",
    "ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS",
    "AllocatedResidualAllLayerDiagnostic",
    "AllLayerFirstResidualSpectrum",
    "AllLayerResidualSpectrumConfig",
    "AllLayerResidualSpectrumGeneratorBank",
    "FixedProfileResidualSpectrumFit",
    "GreedyResidualSpectrumAllocation",
    "ResidualSpectrumAllocationRound",
    "ResidualSpectrumAllocationStep",
    "ResidualSpectrumBankAllLayerEvaluation",
    "ResidualSpectrumProgressCallback",
    "ResidualSpectrumSiteMarginal",
    "all_layer_residual_spectrum_generator_bank_from_parent",
    "all_layer_residual_spectrum_generator_bank_from_state_dict",
    "compute_all_layer_first_residual_spectrum",
    "compute_site_residual_spectrum_marginal",
    "evaluate_allocated_residual_bank_all_layers",
    "greedily_allocate_residual_spectrum",
    "replay_fixed_residual_allocation_profile",
    "residual_allocation_profile_sha256",
]
