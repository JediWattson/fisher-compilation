"""Targeted rank expansion for Gemma attention-transport generators.

This A-only rung starts from a completed independent global-site rank-one bank
and adds one more rank-one operator direction at the two unresolved global sites,
``mutation.11`` and ``mutation.23``.  ``mutation.35`` remains rank one and all
15 local-attention sites remain exactly equal to the parent bank::

    C_init[i,k](xi) = C_site-rank1[i,k](xi) + 0 (r2[i,k]^T xi)

After nesting is verified exactly, both target-site components are jointly
refit, giving ``G_hier + l1' (r1'^T xi) + l2 (r2^T xi)``.  Component order is
stable (inherited first, additive second), but—as with any rank-two
factorization—the factors are not claimed to be operator-unique.

The second-factor tensors have shape ``[2,K,H]``.  Their right vectors use the
same unit-norm, positive-pivot gauge as the first residual and their left
vectors initialize to exact zero.  Thus initialization is bit-exact to the
parent and the added inventory is exactly ``2 * 2 * K * H`` parameters.

Fitting alternates exact signed Top-2 coefficients with a short second-only
warmup and then a true joint refit of both components at each target site.  A
per-site checkpoint is accepted only after a
fresh exact sparse Fisher oracle strictly improves at that site.  Diagnostic
helpers retain metrics only: one measures tied versus untied global coupling,
and one evaluates all 18 attention invocations for the hierarchical, shared,
site-rank-one, step-matched continued-rank-one, and targeted-rank-two banks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
from torch import Tensor, nn

from .gemma3_attention_transport_compiler import (
    AttentionTransportBatch,
    _cpu_float64,
    _tensor_sha256,
    samplewise_joint_fisher_loss,
)
from .gemma3_attention_transport_global_rank1 import (
    GLOBAL_ATTENTION_INVOCATION_IDS,
    GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER,
    GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
    GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
    GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
    GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER,
    GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION,
    GlobalRankOneCouplingDiagnostic,
    GlobalRankOneFisherAttentionTransportGeneratorBank,
    GlobalRankOneSiteCouplingDiagnostic,
    _canonicalize_rank_one_factors,
    _diagnostic_oracle_evaluation,
    _fixed_support_prediction,
    _global_captures,
    _is_canonical_rank_one_gauge,
    _solve_grouped_untied_diagnostic_oracle,
    _solve_tied_diagnostic_oracle,
)
from .gemma3_attention_transport_global_site_rank1 import (
    GlobalSiteRankOneAttentionTransportBankFit,
    GlobalSiteRankOneAttentionTransportFitConfig,
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    _global_site_rank_one_bank_with_factors,
    fit_global_site_rank_one_attention_transport_generator_bank,
    global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict,
)
from .gemma3_attention_transport_hierarchical_edge import (
    HierarchicalFisherAttentionTransportGeneratorBank,
    _capture_values,
    _exact_sparse_coefficients_by_capture,
)
from .gemma3_attention_transport_sparse_v2 import (
    SparseAttentionTransportOracleEvaluation,
    evaluate_sparse_attention_transport_bank,
)


TARGETED_SITE_RANK_TWO_INVOCATION_IDS = (
    "mutation.11",
    "mutation.23",
)
TARGETED_SITE_RANK_PROFILE = (
    ("mutation.11", 2),
    ("mutation.23", 2),
    ("mutation.35", 1),
)
TARGETED_SITE_RANK_TWO_RESIDUAL_SEMANTICS = (
    "joint_target_site_rank_two_refit_at_mutation_11_and_23_v1"
)
TARGETED_SITE_RANK_TWO_GAUGE_SEMANTICS = (
    "unit_second_right_per_target_site_generator_max_abs_positive_v1"
)
TARGETED_SITE_RANK_TWO_COMPONENT_ORDER_SEMANTICS = (
    "component_1_inherited_then_jointly_refit_component_2_additive_stable_order_v1"
)
TARGETED_SITE_RANK_TWO_INITIALIZATION_SEMANTICS = (
    "zero_second_left_deterministic_right_orthogonal_to_first_when_possible_v1"
)
TARGETED_SITE_RANK_TWO_ACCEPTANCE_SEMANTICS = (
    "joint_target_site_rank_two_strict_exact_per_site_acceptance_v1"
)
TARGETED_SITE_RANK_TWO_WARMUP_SEMANTICS = (
    "round1_second_component_adam_first_step_backtracking_v1"
)

_BANK_DOMAIN = b"fisher_graph:attention-transport-targeted-site-rank-two-bank:v1\0"


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


def _target_index(invocation_id: str) -> int | None:
    if not isinstance(invocation_id, str):
        raise TypeError("invocation_id must be a string")
    try:
        return TARGETED_SITE_RANK_TWO_INVOCATION_IDS.index(invocation_id)
    except ValueError:
        return None


def _canonicalize_second_factors(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> tuple[Tensor, Tensor]:
    left_value = _cpu_float64(left, label="second residual left vectors", ndim=3)
    right_value = _cpu_float64(
        right,
        label="second residual right vectors",
        ndim=3,
    )
    if left_value.shape != right_value.shape:
        raise ValueError("second residual factor shapes do not match")
    if left_value.shape[0] != len(TARGETED_SITE_RANK_TWO_INVOCATION_IDS):
        raise ValueError("second residual factors have the wrong site count")
    sites, generators, hidden = left_value.shape
    flat_left = left_value.reshape(sites * generators, hidden)
    flat_right = right_value.reshape(sites * generators, hidden)
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
        canonical_left.reshape(sites, generators, hidden).contiguous(),
        canonical_right.reshape(sites, generators, hidden).contiguous(),
    )


def _canonicalize_target_first_factors(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> tuple[Tensor, Tensor]:
    """Canonicalize each mutable first-component site in isolation."""

    left_value = _cpu_float64(
        left,
        label="target first residual left vectors",
        ndim=3,
    )
    right_value = _cpu_float64(
        right,
        label="target first residual right vectors",
        ndim=3,
    )
    if left_value.shape != right_value.shape:
        raise ValueError("target first residual factor shapes do not match")
    if left_value.shape[0] != len(TARGETED_SITE_RANK_TWO_INVOCATION_IDS):
        raise ValueError("target first residual factors have the wrong site count")
    canonical_left = []
    canonical_right = []
    for target_left, target_right in zip(
        left_value,
        right_value,
        strict=True,
    ):
        if _is_canonical_rank_one_gauge(
            target_left,
            target_right,
            gauge_floor=gauge_floor,
        ):
            site_left = target_left.clone().contiguous()
            site_right = target_right.clone().contiguous()
        else:
            site_left, site_right = _canonicalize_rank_one_factors(
                target_left,
                target_right,
                gauge_floor=gauge_floor,
            )
        canonical_left.append(site_left)
        canonical_right.append(site_right)
    return (
        torch.stack(canonical_left).contiguous(),
        torch.stack(canonical_right).contiguous(),
    )


def _is_canonical_second_gauge(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> bool:
    try:
        left_value = _cpu_float64(
            left,
            label="second residual left vectors",
            ndim=3,
        )
        right_value = _cpu_float64(
            right,
            label="second residual right vectors",
            ndim=3,
        )
        if left_value.shape != right_value.shape or left_value.shape[0] != 2:
            return False
        sites, generators, hidden = left_value.shape
        return _is_canonical_rank_one_gauge(
            left_value.reshape(sites * generators, hidden),
            right_value.reshape(sites * generators, hidden),
            gauge_floor=gauge_floor,
        )
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class TargetedSiteRankTwoFisherAttentionTransportGeneratorBank(
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank
):
    """Nested then jointly refit rank-two directions at two global sites."""

    second_residual_left_vectors: Tensor
    second_residual_right_vectors: Tensor
    initial_site_rank_one_bank_sha256: str
    _second_tensor_sha256s: tuple[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _parent_site_rank_one_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _targeted_artifact_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank.__post_init__(self)
        if not _is_sha256(self.initial_site_rank_one_bank_sha256):
            raise ValueError("initial site rank-one bank SHA-256 is invalid")
        parent_sha = (
            GlobalSiteRankOneFisherAttentionTransportGeneratorBank.artifact_sha256.fget(
                self
            )
        )
        left, right = _canonicalize_second_factors(
            self.second_residual_left_vectors,
            self.second_residual_right_vectors,
            gauge_floor=self.gauge_floor,
        )
        expected = (
            len(TARGETED_SITE_RANK_TWO_INVOCATION_IDS),
            self.generator_count,
            self.hidden_width,
        )
        if left.shape != expected:
            raise ValueError("second residual factors have the wrong geometry")
        object.__setattr__(self, "second_residual_left_vectors", left)
        object.__setattr__(self, "second_residual_right_vectors", right)
        tensor_hashes = (_tensor_sha256(left), _tensor_sha256(right))
        object.__setattr__(self, "_second_tensor_sha256s", tensor_hashes)
        object.__setattr__(self, "_parent_site_rank_one_sha256", parent_sha)
        payload = self._targeted_artifact_payload(tensor_hashes)
        object.__setattr__(
            self,
            "_targeted_artifact_sha256",
            hashlib.sha256(_BANK_DOMAIN + _canonical_json_bytes(payload)).hexdigest(),
        )

    def _targeted_artifact_payload(
        self,
        tensor_hashes: tuple[str, str],
    ) -> dict[str, object]:
        return {
            "acceptance_semantics": TARGETED_SITE_RANK_TWO_ACCEPTANCE_SEMANTICS,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "component_order_semantics": (
                TARGETED_SITE_RANK_TWO_COMPONENT_ORDER_SEMANTICS
            ),
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": TARGETED_SITE_RANK_TWO_GAUGE_SEMANTICS,
            "generator_count": self.generator_count,
            "hidden_width": self.hidden_width,
            "initialization_semantics": (
                TARGETED_SITE_RANK_TWO_INITIALIZATION_SEMANTICS
            ),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "rank_profile": [list(value) for value in TARGETED_SITE_RANK_PROFILE],
            "residual_parameter_count": self.targeted_residual_parameter_count,
            "residual_semantics": TARGETED_SITE_RANK_TWO_RESIDUAL_SEMANTICS,
            "router_semantics": GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
            "second_residual_left_vectors_sha256": tensor_hashes[0],
            "second_residual_parameter_count": (
                self.second_residual_parameter_count
            ),
            "second_residual_right_vectors_sha256": tensor_hashes[1],
            "current_first_component_bank_sha256": (
                self._parent_site_rank_one_sha256
            ),
            "initial_site_rank_one_bank_sha256": (
                self.initial_site_rank_one_bank_sha256
            ),
            "targeted_invocation_ids": list(
                TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            ),
        }

    def _validate_targeted_unchanged(self) -> None:
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank._validate_global_site_rank_one_unchanged(
            self
        )
        current = (
            _tensor_sha256(self.second_residual_left_vectors),
            _tensor_sha256(self.second_residual_right_vectors),
        )
        if current != self._second_tensor_sha256s:
            raise RuntimeError("targeted rank-two bank changed after construction")
        if not _is_canonical_second_gauge(
            self.second_residual_left_vectors,
            self.second_residual_right_vectors,
            gauge_floor=self.gauge_floor,
        ):
            raise RuntimeError("targeted rank-two generator gauge drifted")
        parent_sha = (
            GlobalSiteRankOneFisherAttentionTransportGeneratorBank.artifact_sha256.fget(
                self
            )
        )
        if parent_sha != self._parent_site_rank_one_sha256:
            raise RuntimeError("targeted rank-two parent bank changed")

    @property
    def parent_site_rank_one_sha256(self) -> str:
        self._validate_targeted_unchanged()
        return self._parent_site_rank_one_sha256

    @property
    def second_residual_parameter_count(self) -> int:
        expected = (
            2
            * len(TARGETED_SITE_RANK_TWO_INVOCATION_IDS)
            * self.generator_count
            * self.hidden_width
        )
        observed = int(
            self.second_residual_left_vectors.numel()
            + self.second_residual_right_vectors.numel()
        )
        if observed != expected:
            raise RuntimeError("second residual parameter count drifted")
        return observed

    @property
    def targeted_residual_parameter_count(self) -> int:
        return int(
            self.global_site_rank_one_parameter_count
            + self.second_residual_parameter_count
        )

    @property
    def stored_parameter_count(self) -> int:
        parent = (
            GlobalSiteRankOneFisherAttentionTransportGeneratorBank.stored_parameter_count.fget(
                self
            )
        )
        return int(parent + self.second_residual_parameter_count)

    @property
    def artifact_sha256(self) -> str:
        self._validate_targeted_unchanged()
        return self._targeted_artifact_sha256

    def __hash__(self) -> int:
        return hash(
            (
                TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
                self.artifact_sha256,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(
                other,
                TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
            )
            and self.artifact_sha256 == other.artifact_sha256
        )

    def second_rank_one_candidate_residuals_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        """Return ``[N,K,H]`` second residuals, exactly zero off-target."""

        self._validate_targeted_unchanged()
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        index = _target_index(invocation_id)
        if index is None:
            return torch.zeros_like(contextual)
        left = self.second_residual_left_vectors[index]
        right = self.second_residual_right_vectors[index]
        activation = torch.einsum("nkh,kh->nk", contextual, right)
        return (activation[:, :, None] * left[None]).contiguous()

    def total_global_site_candidate_residuals_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        """Return the first plus optional second residual candidate tensor."""

        first = GlobalSiteRankOneFisherAttentionTransportGeneratorBank.global_site_rank_one_candidate_residuals_from_head_contexts(
            self,
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        if _target_index(invocation_id) is None:
            return first
        return (
            first
            + self.second_rank_one_candidate_residuals_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=head_contexts,
            )
        ).contiguous()

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
        if _target_index(invocation_id) is None:
            return parent
        return (
            parent
            + self.second_rank_one_candidate_residuals_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=head_contexts,
            )
        ).contiguous()

    def state_dict(self) -> dict[str, object]:
        self._validate_targeted_unchanged()
        return {
            "acceptance_semantics": TARGETED_SITE_RANK_TWO_ACCEPTANCE_SEMANTICS,
            "artifact_sha256": self._targeted_artifact_sha256,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "component_order_semantics": (
                TARGETED_SITE_RANK_TWO_COMPONENT_ORDER_SEMANTICS
            ),
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": TARGETED_SITE_RANK_TWO_GAUGE_SEMANTICS,
            "initialization_semantics": (
                TARGETED_SITE_RANK_TWO_INITIALIZATION_SEMANTICS
            ),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "rank_profile": [list(value) for value in TARGETED_SITE_RANK_PROFILE],
            "residual_semantics": TARGETED_SITE_RANK_TWO_RESIDUAL_SEMANTICS,
            "router_semantics": GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
            "second_residual_left_vectors": (
                self.second_residual_left_vectors.clone()
            ),
            "second_residual_right_vectors": (
                self.second_residual_right_vectors.clone()
            ),
            "site_rank_one_bank": (
                GlobalSiteRankOneFisherAttentionTransportGeneratorBank.state_dict(
                    self
                )
            ),
            "current_first_component_bank_sha256": (
                self._parent_site_rank_one_sha256
            ),
            "initial_site_rank_one_bank_sha256": (
                self.initial_site_rank_one_bank_sha256
            ),
            "targeted_invocation_ids": list(
                TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            ),
        }


def targeted_site_rank_two_fisher_attention_transport_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> TargetedSiteRankTwoFisherAttentionTransportGeneratorBank:
    """Reload a source-free targeted bank and verify every lineage edge."""

    expected = {
        "acceptance_semantics",
        "artifact_sha256",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "component_order_semantics",
        "gauge_floor",
        "gauge_semantics",
        "initialization_semantics",
        "native_output_rmsnorm_retained",
        "native_qk_retained",
        "rank_profile",
        "residual_semantics",
        "router_semantics",
        "second_residual_left_vectors",
        "second_residual_right_vectors",
        "site_rank_one_bank",
        "current_first_component_bank_sha256",
        "initial_site_rank_one_bank_sha256",
        "targeted_invocation_ids",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("targeted site rank-two bank state fields are invalid")
    raw_parent = state["site_rank_one_bank"]
    gauge_floor = state["gauge_floor"]
    if (
        not isinstance(raw_parent, Mapping)
        or state["acceptance_semantics"]
        != TARGETED_SITE_RANK_TWO_ACCEPTANCE_SEMANTICS
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_raw_teacher_outputs_or_gradients"] is not False
        or state["component_order_semantics"]
        != TARGETED_SITE_RANK_TWO_COMPONENT_ORDER_SEMANTICS
        or state["gauge_semantics"] != TARGETED_SITE_RANK_TWO_GAUGE_SEMANTICS
        or state["initialization_semantics"]
        != TARGETED_SITE_RANK_TWO_INITIALIZATION_SEMANTICS
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
        or state["rank_profile"]
        != [list(value) for value in TARGETED_SITE_RANK_PROFILE]
        or state["residual_semantics"]
        != TARGETED_SITE_RANK_TWO_RESIDUAL_SEMANTICS
        or state["router_semantics"] != GLOBAL_RANK_ONE_ROUTER_SEMANTICS
        or state["targeted_invocation_ids"]
        != list(TARGETED_SITE_RANK_TWO_INVOCATION_IDS)
        or not _is_sha256(state["artifact_sha256"])
        or not _is_sha256(state["current_first_component_bank_sha256"])
        or not _is_sha256(state["initial_site_rank_one_bank_sha256"])
        or isinstance(gauge_floor, bool)
        or not isinstance(gauge_floor, (int, float))
        or not math.isfinite(float(gauge_floor))
        or float(gauge_floor) <= 0.0
        or not _is_canonical_second_gauge(
            state["second_residual_left_vectors"],
            state["second_residual_right_vectors"],
            gauge_floor=float(gauge_floor),
        )
    ):
        raise ValueError("targeted site rank-two bank safety semantics are invalid")
    parent = (
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            raw_parent
        )
    )
    if (
        state["current_first_component_bank_sha256"] != parent.artifact_sha256
        or float(gauge_floor) != parent.gauge_floor
    ):
        raise ValueError("targeted site rank-two parent lineage drifted")
    bank = TargetedSiteRankTwoFisherAttentionTransportGeneratorBank(
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
        second_residual_left_vectors=state["second_residual_left_vectors"],
        second_residual_right_vectors=state["second_residual_right_vectors"],
        initial_site_rank_one_bank_sha256=state[
            "initial_site_rank_one_bank_sha256"
        ],
    )
    if bank.artifact_sha256 != state["artifact_sha256"]:
        raise ValueError("targeted site rank-two artifact lineage drifted")
    return bank


def _deterministic_orthogonal_second_right_vectors(
    parent: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    *,
    gauge_floor: float,
) -> Tensor:
    """Choose deterministic unit rights orthogonal to component one."""

    values = torch.empty(
        len(TARGETED_SITE_RANK_TWO_INVOCATION_IDS),
        parent.generator_count,
        parent.hidden_width,
        dtype=torch.float64,
    )
    for target_index, invocation_id in enumerate(
        TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    ):
        site_index = GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
        for generator_index in range(parent.generator_count):
            first = parent.global_site_residual_right_vectors[
                site_index,
                generator_index,
            ]
            # The least-aligned coordinate has the largest orthogonal residue.
            coordinate_order = torch.argsort(first.abs(), stable=True).tolist()
            orthogonal: Tensor | None = None
            for coordinate in coordinate_order:
                basis = torch.zeros_like(first)
                basis[int(coordinate)] = 1.0
                candidate = basis - torch.dot(basis, first) * first
                norm = torch.linalg.vector_norm(candidate)
                if float(norm) > float(gauge_floor):
                    orthogonal = candidate / norm
                    break
            if orthogonal is None:
                raise ValueError(
                    "hidden width cannot supply an independent second direction"
                )
            values[target_index, generator_index] = orthogonal
    zero = torch.zeros_like(values)
    _left, canonical = _canonicalize_second_factors(
        zero,
        values,
        gauge_floor=gauge_floor,
    )
    return canonical


def _initialize_targeted_site_rank_two_bank(
    parent: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    *,
    gauge_floor: float,
) -> TargetedSiteRankTwoFisherAttentionTransportGeneratorBank:
    if not isinstance(
        parent,
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("parent must be a global site rank-one bank")
    parent._validate_global_site_rank_one_unchanged()
    if parent.hidden_width < 2:
        raise ValueError("targeted rank two requires hidden width of at least two")
    if parent.gauge_floor != float(gauge_floor):
        raise ValueError("targeted and parent gauge floors must match")
    right = _deterministic_orthogonal_second_right_vectors(
        parent,
        gauge_floor=gauge_floor,
    )
    return TargetedSiteRankTwoFisherAttentionTransportGeneratorBank(
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
        second_residual_left_vectors=torch.zeros_like(right),
        second_residual_right_vectors=right,
        initial_site_rank_one_bank_sha256=parent.artifact_sha256,
    )


def _targeted_bank_with_factors(
    template: TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
    first_left: Tensor,
    first_right: Tensor,
    second_left: Tensor,
    second_right: Tensor,
) -> TargetedSiteRankTwoFisherAttentionTransportGeneratorBank:
    canonical_first_left, canonical_first_right = _canonicalize_target_first_factors(
        first_left,
        first_right,
        gauge_floor=template.gauge_floor,
    )
    first_left_value = template.global_site_residual_left_vectors.clone()
    first_right_value = template.global_site_residual_right_vectors.clone()
    for target_index, invocation_id in enumerate(
        TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    ):
        site_index = GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
        first_left_value[site_index] = canonical_first_left[target_index]
        first_right_value[site_index] = canonical_first_right[target_index]
    # Construct a current first-component bank first so its source-free hash is
    # bound independently from the original nested initialization lineage.
    current_parent = GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
        left_factors=template.left_factors,
        singular_scales=template.singular_scales,
        right_factors=template.right_factors,
        head_coefficients=template.head_coefficients,
        head_ids=template.head_ids,
        hierarchy=template.hierarchy,
        block_scale_contrasts=template.block_scale_contrasts,
        site_scale_contrasts=template.site_scale_contrasts,
        signed_feature_floor=template.signed_feature_floor,
        global_site_residual_left_vectors=first_left_value,
        global_site_residual_right_vectors=first_right_value,
        shared_rank_one_bank_sha256=template.shared_rank_one_bank_sha256,
        gauge_floor=template.gauge_floor,
    )
    canonical_second_left, canonical_second_right = _canonicalize_second_factors(
        second_left,
        second_right,
        gauge_floor=template.gauge_floor,
    )
    return TargetedSiteRankTwoFisherAttentionTransportGeneratorBank(
        left_factors=current_parent.left_factors,
        singular_scales=current_parent.singular_scales,
        right_factors=current_parent.right_factors,
        head_coefficients=current_parent.head_coefficients,
        head_ids=current_parent.head_ids,
        hierarchy=current_parent.hierarchy,
        block_scale_contrasts=current_parent.block_scale_contrasts,
        site_scale_contrasts=current_parent.site_scale_contrasts,
        signed_feature_floor=current_parent.signed_feature_floor,
        global_site_residual_left_vectors=(
            current_parent.global_site_residual_left_vectors
        ),
        global_site_residual_right_vectors=(
            current_parent.global_site_residual_right_vectors
        ),
        shared_rank_one_bank_sha256=current_parent.shared_rank_one_bank_sha256,
        gauge_floor=current_parent.gauge_floor,
        second_residual_left_vectors=canonical_second_left,
        second_residual_right_vectors=canonical_second_right,
        initial_site_rank_one_bank_sha256=(
            template.initial_site_rank_one_bank_sha256
        ),
    )


@dataclass(frozen=True, slots=True)
class TargetedSiteRankTwoAttentionTransportFitConfig:
    """Protocol for a nested, jointly refit rank-(2,2,1) A-only bank."""

    site_rank_one: GlobalSiteRankOneAttentionTransportFitConfig
    alternating_rounds: int = 4
    steps_per_target_site_per_round: int = 48
    second_component_warmup_steps: int = 8
    warmup_max_backtracks: int = 24
    warmup_backtrack_factor: float = 0.5
    learning_rate: float = 1.0e-2
    residual_ridge: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    gauge_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(
            self.site_rank_one,
            GlobalSiteRankOneAttentionTransportFitConfig,
        ):
            raise TypeError("site_rank_one must be a site rank-one fit config")
        if type(self.alternating_rounds) is not int or self.alternating_rounds <= 0:
            raise ValueError("alternating_rounds must be positive")
        if (
            type(self.steps_per_target_site_per_round) is not int
            or self.steps_per_target_site_per_round <= 0
        ):
            raise ValueError("steps_per_target_site_per_round must be positive")
        if (
            type(self.second_component_warmup_steps) is not int
            or self.second_component_warmup_steps < 0
        ):
            raise ValueError("second_component_warmup_steps must be nonnegative")
        if (
            type(self.warmup_max_backtracks) is not int
            or self.warmup_max_backtracks < 0
        ):
            raise ValueError("warmup_max_backtracks must be nonnegative")
        if (
            isinstance(self.warmup_backtrack_factor, bool)
            or not isinstance(self.warmup_backtrack_factor, (int, float))
            or not math.isfinite(float(self.warmup_backtrack_factor))
            or not 0.0 < float(self.warmup_backtrack_factor) < 1.0
        ):
            raise ValueError("warmup_backtrack_factor must be strictly between zero and one")
        for label, value in (
            ("learning_rate", self.learning_rate),
            ("residual_ridge", self.residual_ridge),
            ("gradient_clip_norm", self.gradient_clip_norm),
            ("gauge_floor", self.gauge_floor),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive")
        predecessor = self.site_rank_one
        if (
            float(self.learning_rate) != float(predecessor.learning_rate)
            or float(self.residual_ridge) != float(predecessor.residual_ridge)
            or float(self.gradient_clip_norm)
            != float(predecessor.gradient_clip_norm)
            or float(self.gauge_floor) != float(predecessor.gauge_floor)
        ):
            raise ValueError(
                "targeted optimizer, ridge, clipping, and gauge must match predecessor"
            )

    @classmethod
    def fixed_k8_r8_top2(
        cls,
    ) -> TargetedSiteRankTwoAttentionTransportFitConfig:
        return cls(
            site_rank_one=(
                GlobalSiteRankOneAttentionTransportFitConfig.fixed_k8_r8_top2()
            )
        )


@dataclass(frozen=True, slots=True)
class ContinuedTargetSiteRankOneControlFit:
    """Step-matched continued-rank-one control from the same parent."""

    bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank
    initial_site_rank_one_bank_sha256: str
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    initial_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_round_relative_fisher_losses: tuple[float, ...]
    fit_round_global_relative_fisher_losses: tuple[float, ...]
    fit_round_target_site_relative_fisher_losses: tuple[
        tuple[str, tuple[float, ...]], ...
    ]
    optimizer_steps_per_target_site_by_round: tuple[int, ...]

    def __post_init__(self) -> None:
        all_history = self.fit_round_relative_fisher_losses
        global_history = self.fit_round_global_relative_fisher_losses
        site_histories = self.fit_round_target_site_relative_fisher_losses
        evaluations = (
            self.initial_fit_oracle,
            self.final_fit_oracle,
            self.initial_global_fit_oracle,
            self.final_global_fit_oracle,
        )
        if (
            not isinstance(
                self.bank,
                GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
            )
            or not _is_sha256(self.initial_site_rank_one_bank_sha256)
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in evaluations
            )
            or len(self.optimizer_steps_per_target_site_by_round)
            != len(all_history) - 1
            or any(
                type(value) is not int or value <= 0
                for value in self.optimizer_steps_per_target_site_by_round
            )
            or len(all_history) < 2
            or len(global_history) != len(all_history)
            or tuple(key for key, _values in site_histories)
            != TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            or any(len(values) != len(all_history) for _key, values in site_histories)
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in all_history
                + global_history
                + tuple(
                    item for _key, values in site_histories for item in values
                )
            )
            or any(
                right > left + 1.0e-12
                for history in (
                    all_history,
                    global_history,
                    *(values for _key, values in site_histories),
                )
                for left, right in zip(history, history[1:])
            )
            or not math.isclose(
                all_history[0],
                self.initial_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                all_history[-1],
                self.final_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                global_history[0],
                self.initial_global_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                global_history[-1],
                self.final_global_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("continued target-site rank-one control is invalid")
        initial_sites = dict(
            self.initial_global_fit_oracle.per_invocation_relative_fisher_loss
        )
        final_sites = dict(
            self.final_global_fit_oracle.per_invocation_relative_fisher_loss
        )
        if any(
            key not in initial_sites
            or key not in final_sites
            or not math.isclose(
                values[0],
                initial_sites[key],
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                values[-1],
                final_sites[key],
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            for key, values in site_histories
        ):
            raise ValueError("continued rank-one site histories do not close")


@dataclass(frozen=True, slots=True)
class TargetedSiteRankTwoAttentionTransportBankFit:
    """Metrics-only fit result for the nested targeted rank expansion."""

    bank: TargetedSiteRankTwoFisherAttentionTransportGeneratorBank
    site_rank_one_fit: GlobalSiteRankOneAttentionTransportBankFit
    continued_site_rank_one_control: ContinuedTargetSiteRankOneControlFit
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    initial_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_round_relative_fisher_losses: tuple[float, ...]
    fit_round_global_relative_fisher_losses: tuple[float, ...]
    fit_round_target_site_relative_fisher_losses: tuple[
        tuple[str, tuple[float, ...]], ...
    ]
    warmup_fixed_support_relative_decreases: tuple[tuple[str, float], ...]
    warmup_accepted_learning_rates: tuple[tuple[str, float], ...]
    warmup_backtrack_counts: tuple[tuple[str, int], ...]
    activation_warmup_steps_per_target_site: int
    joint_optimizer_steps_per_target_site_per_round: int
    optimizer_steps_per_target_site_by_round: tuple[int, ...]

    def __post_init__(self) -> None:
        all_history = self.fit_round_relative_fisher_losses
        global_history = self.fit_round_global_relative_fisher_losses
        target_histories = self.fit_round_target_site_relative_fisher_losses
        evaluations = (
            self.initial_fit_oracle,
            self.final_fit_oracle,
            self.initial_global_fit_oracle,
            self.final_global_fit_oracle,
        )
        if (
            not isinstance(
                self.bank,
                TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
            )
            or not isinstance(
                self.site_rank_one_fit,
                GlobalSiteRankOneAttentionTransportBankFit,
            )
            or not isinstance(
                self.continued_site_rank_one_control,
                ContinuedTargetSiteRankOneControlFit,
            )
            or self.continued_site_rank_one_control.initial_site_rank_one_bank_sha256
            != self.site_rank_one_fit.bank.artifact_sha256
            or self.continued_site_rank_one_control.initial_fit_oracle
            != self.site_rank_one_fit.final_fit_oracle
            or self.continued_site_rank_one_control.initial_global_fit_oracle
            != self.site_rank_one_fit.final_global_fit_oracle
            or self.bank.initial_site_rank_one_bank_sha256
            != self.site_rank_one_fit.bank.artifact_sha256
            or any(
                not isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in evaluations
            )
            or self.initial_fit_oracle != self.site_rank_one_fit.final_fit_oracle
            or self.initial_global_fit_oracle
            != self.site_rank_one_fit.final_global_fit_oracle
            or len(all_history) < 2
            or len(global_history) != len(all_history)
            or len(
                self.continued_site_rank_one_control.fit_round_relative_fisher_losses
            )
            != len(all_history)
            or type(self.activation_warmup_steps_per_target_site) is not int
            or self.activation_warmup_steps_per_target_site < 0
            or type(self.joint_optimizer_steps_per_target_site_per_round) is not int
            or self.joint_optimizer_steps_per_target_site_per_round <= 0
            or self.optimizer_steps_per_target_site_by_round
            != self.continued_site_rank_one_control.optimizer_steps_per_target_site_by_round
            or len(self.optimizer_steps_per_target_site_by_round)
            != len(all_history) - 1
            or self.optimizer_steps_per_target_site_by_round
            != (
                (
                    self.activation_warmup_steps_per_target_site
                    + self.joint_optimizer_steps_per_target_site_per_round
                ),
                *(
                    self.joint_optimizer_steps_per_target_site_per_round
                    for _ in range(len(all_history) - 2)
                ),
            )
            or tuple(key for key, _values in target_histories)
            != TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            or any(len(values) != len(all_history) for _key, values in target_histories)
            or tuple(key for key, _value in self.warmup_fixed_support_relative_decreases)
            != TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            or tuple(key for key, _value in self.warmup_accepted_learning_rates)
            != TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            or tuple(key for key, _value in self.warmup_backtrack_counts)
            != TARGETED_SITE_RANK_TWO_INVOCATION_IDS
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in all_history
                + global_history
                + tuple(
                    item
                    for _key, values in target_histories
                    for item in values
                )
                + tuple(
                    value
                    for _key, value in self.warmup_fixed_support_relative_decreases
                )
                + tuple(
                    value
                    for _key, value in self.warmup_accepted_learning_rates
                )
            )
            or any(
                type(value) is not int or value < 0
                for _key, value in self.warmup_backtrack_counts
            )
            or (
                self.activation_warmup_steps_per_target_site > 0
                and any(
                    float(value) <= 0.0
                    for _key, value in self.warmup_fixed_support_relative_decreases
                )
            )
            or (
                self.activation_warmup_steps_per_target_site > 0
                and any(
                    float(value) <= 0.0
                    for _key, value in self.warmup_accepted_learning_rates
                )
            )
            or (
                self.activation_warmup_steps_per_target_site == 0
                and (
                    any(
                        float(value) != 0.0
                        for _key, value in self.warmup_accepted_learning_rates
                    )
                    or any(
                        value != 0
                        for _key, value in self.warmup_backtrack_counts
                    )
                )
            )
            or any(
                right > left + 1.0e-12
                for history in (
                    all_history,
                    global_history,
                    *(values for _key, values in target_histories),
                )
                for left, right in zip(history, history[1:])
            )
            or not math.isclose(
                all_history[0],
                self.initial_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                all_history[-1],
                self.final_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                global_history[0],
                self.initial_global_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                global_history[-1],
                self.final_global_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("targeted site rank-two bank fit is invalid")
        initial_sites = dict(
            self.initial_global_fit_oracle.per_invocation_relative_fisher_loss
        )
        final_sites = dict(
            self.final_global_fit_oracle.per_invocation_relative_fisher_loss
        )
        if any(
            key not in initial_sites
            or key not in final_sites
            or not math.isclose(
                values[0],
                initial_sites[key],
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                values[-1],
                final_sites[key],
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            for key, values in target_histories
        ):
            raise ValueError("targeted site rank-two site histories do not close")
        mutation_35_index = GLOBAL_ATTENTION_INVOCATION_IDS.index("mutation.35")
        parent = self.site_rank_one_fit.bank
        if (
            self.bank.hierarchical_bank_sha256 != parent.hierarchical_bank_sha256
            or self.bank.shared_rank_one_bank_sha256
            != parent.shared_rank_one_bank_sha256
            or not torch.equal(
                self.bank.global_site_residual_left_vectors[mutation_35_index],
                parent.global_site_residual_left_vectors[mutation_35_index],
            )
            or not torch.equal(
                self.bank.global_site_residual_right_vectors[mutation_35_index],
                parent.global_site_residual_right_vectors[mutation_35_index],
            )
        ):
            raise ValueError("targeted fit changed frozen non-target structure")

    @property
    def rank_profile(self) -> tuple[tuple[str, int], ...]:
        return TARGETED_SITE_RANK_PROFILE


def _target_site_oracle_evaluations(
    bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> dict[str, SparseAttentionTransportOracleEvaluation]:
    return {
        invocation_id: evaluate_sparse_attention_transport_bank(
            bank,
            tuple(
                capture
                for capture in captures
                if capture.invocation_id == invocation_id
            ),
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        for invocation_id in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    }


def _validate_parent_fit_against_captures(
    site_fit: GlobalSiteRankOneAttentionTransportBankFit,
    captures: tuple[AttentionTransportBatch, ...],
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> tuple[
    SparseAttentionTransportOracleEvaluation,
    SparseAttentionTransportOracleEvaluation,
]:
    expected_head_ids = {
        f"{capture.invocation_id}/query-head.{head_index}"
        for capture in captures
        for head_index in range(capture.query_head_count)
    }
    if (
        not isinstance(site_fit, GlobalSiteRankOneAttentionTransportBankFit)
        or set(site_fit.bank.head_ids) != expected_head_ids
    ):
        raise ValueError("site rank-one fit does not match the A capture catalog")
    current_all = evaluate_sparse_attention_transport_bank(
        site_fit.bank,
        captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    current_global = evaluate_sparse_attention_transport_bank(
        site_fit.bank,
        _global_captures(captures),
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    if (
        current_all != site_fit.final_fit_oracle
        or current_global != site_fit.final_global_fit_oracle
    ):
        raise ValueError(
            "site rank-one fit evaluations do not match the current A captures"
        )
    return current_all, current_global


def _fit_continued_target_site_rank_one_control(
    captures: tuple[AttentionTransportBatch, ...],
    *,
    parent_fit: GlobalSiteRankOneAttentionTransportBankFit,
    config: TargetedSiteRankTwoAttentionTransportFitConfig,
    initial: SparseAttentionTransportOracleEvaluation,
    initial_global: SparseAttentionTransportOracleEvaluation,
) -> ContinuedTargetSiteRankOneControlFit:
    shared_config = config.site_rank_one.shared_rank_one.hierarchical.shared
    active_count = shared_config.active_count
    euclidean_floor = shared_config.euclidean_floor
    coefficient_ridge = shared_config.oracle_coefficient_ridge
    global_captures = _global_captures(captures)
    best_bank = parent_fit.bank
    best_evaluation = initial
    best_global_evaluation = initial_global
    best_sites = _target_site_oracle_evaluations(
        best_bank,
        global_captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    history = [float(initial.relative_fisher_loss)]
    global_history = [float(initial_global.relative_fisher_loss)]
    site_histories = {
        key: [float(value.relative_fisher_loss)]
        for key, value in best_sites.items()
    }
    step_schedule = (
        config.second_component_warmup_steps
        + config.steps_per_target_site_per_round,
        *(
            config.steps_per_target_site_per_round
            for _ in range(config.alternating_rounds - 1)
        ),
    )
    for round_index in range(1, config.alternating_rounds + 1):
        step_count = step_schedule[round_index - 1]
        for invocation_id in TARGETED_SITE_RANK_TWO_INVOCATION_IDS:
            site_index = GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
            site_captures = tuple(
                capture
                for capture in captures
                if capture.invocation_id == invocation_id
            )
            coefficients = _exact_sparse_coefficients_by_capture(
                best_bank,
                site_captures,
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            pairs = tuple(zip(site_captures, coefficients, strict=True))
            row_count = float(sum(capture.row_count for capture, _ in pairs))
            left = nn.Parameter(
                best_bank.global_site_residual_left_vectors[site_index].clone()
            )
            right = nn.Parameter(
                best_bank.global_site_residual_right_vectors[site_index].clone()
            )
            optimizer = torch.optim.Adam(
                (left, right),
                lr=float(config.learning_rate),
            )

            def objective(candidate_left: Tensor, candidate_right: Tensor) -> Tensor:
                unit_right = candidate_right / torch.linalg.vector_norm(
                    candidate_right,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(float(config.gauge_floor))
                value = torch.zeros((), dtype=torch.float64)
                for capture, beta in pairs:
                    prediction = _fixed_support_prediction(
                        best_bank,
                        capture,
                        beta,
                        candidate_left,
                        unit_right,
                    )
                    value = value + samplewise_joint_fisher_loss(
                        prediction,
                        capture.valid_operator_outputs,
                        capture.valid_score_gradients,
                        euclidean_floor=euclidean_floor,
                    ) * capture.row_count / row_count
                return value + float(config.residual_ridge) * candidate_left.square().mean()

            best_fixed = math.inf
            checkpoint: tuple[Tensor, Tensor] | None = None
            for _ in range(step_count):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(left, right)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("continued rank-one control became nonfinite")
                if float(loss.detach()) < best_fixed:
                    best_fixed = float(loss.detach())
                    checkpoint = (left.detach().clone(), right.detach().clone())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (left, right),
                    max_norm=float(config.gradient_clip_norm),
                    error_if_nonfinite=True,
                )
                optimizer.step()
            terminal = objective(left, right)
            if not bool(torch.isfinite(terminal)):
                raise RuntimeError("continued rank-one terminal loss is nonfinite")
            if float(terminal.detach()) < best_fixed:
                checkpoint = (left.detach().clone(), right.detach().clone())
            if checkpoint is None:
                raise RuntimeError("continued rank-one retained no checkpoint")
            checkpoint_left, checkpoint_right = checkpoint
            checkpoint_right = checkpoint_right / torch.linalg.vector_norm(
                checkpoint_right,
                dim=-1,
                keepdim=True,
            ).clamp_min(float(config.gauge_floor))
            candidate_left = best_bank.global_site_residual_left_vectors.clone()
            candidate_right = best_bank.global_site_residual_right_vectors.clone()
            candidate_left[site_index] = checkpoint_left
            candidate_right[site_index] = checkpoint_right
            candidate = _global_site_rank_one_bank_with_factors(
                best_bank,
                candidate_left,
                candidate_right,
            )
            candidate_site = evaluate_sparse_attention_transport_bank(
                candidate,
                site_captures,
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            if (
                candidate_site.relative_fisher_loss
                < best_sites[invocation_id].relative_fisher_loss - 1.0e-12
            ):
                best_bank = candidate
                best_sites[invocation_id] = candidate_site
        evaluation = evaluate_sparse_attention_transport_bank(
            best_bank,
            captures,
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        global_evaluation = evaluate_sparse_attention_transport_bank(
            best_bank,
            global_captures,
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        if (
            evaluation.relative_fisher_loss
            > best_evaluation.relative_fisher_loss + 1.0e-12
            or global_evaluation.relative_fisher_loss
            > best_global_evaluation.relative_fisher_loss + 1.0e-12
        ):
            raise RuntimeError("continued rank-one control regressed")
        best_evaluation = evaluation
        best_global_evaluation = global_evaluation
        history.append(float(evaluation.relative_fisher_loss))
        global_history.append(float(global_evaluation.relative_fisher_loss))
        for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS:
            site_histories[key].append(float(best_sites[key].relative_fisher_loss))
    return ContinuedTargetSiteRankOneControlFit(
        bank=best_bank,
        initial_site_rank_one_bank_sha256=parent_fit.bank.artifact_sha256,
        initial_fit_oracle=initial,
        final_fit_oracle=best_evaluation,
        initial_global_fit_oracle=initial_global,
        final_global_fit_oracle=best_global_evaluation,
        fit_round_relative_fisher_losses=tuple(history),
        fit_round_global_relative_fisher_losses=tuple(global_history),
        fit_round_target_site_relative_fisher_losses=tuple(
            (key, tuple(site_histories[key]))
            for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        ),
        optimizer_steps_per_target_site_by_round=tuple(step_schedule),
    )


def fit_targeted_site_rank_two_attention_transport_generator_bank(
    a_captures: Sequence[AttentionTransportBatch],
    *,
    config: TargetedSiteRankTwoAttentionTransportFitConfig,
    site_rank_one_fit: GlobalSiteRankOneAttentionTransportBankFit | None = None,
    round_observer: Callable[
        [
            int,
            int,
            SparseAttentionTransportOracleEvaluation,
            SparseAttentionTransportOracleEvaluation,
        ],
        None,
    ]
    | None = None,
) -> TargetedSiteRankTwoAttentionTransportBankFit:
    """Fit true rank two at mutations 11/23 from an exact rank-one nesting.

    The supplied or freshly completed parent is evaluated again on the current
    captures and must reproduce its sealed final metrics exactly.  A separate
    continued-rank-one branch receives the same optimizer-step schedule, making
    the targeted-versus-control difference an attributable capacity effect.
    """

    if not isinstance(config, TargetedSiteRankTwoAttentionTransportFitConfig):
        raise TypeError("config must be a targeted site rank-two fit config")
    if round_observer is not None and not callable(round_observer):
        raise TypeError("round_observer must be callable")
    captures = _capture_values(a_captures)
    parent_fit = (
        fit_global_site_rank_one_attention_transport_generator_bank(
            captures,
            config=config.site_rank_one,
        )
        if site_rank_one_fit is None
        else site_rank_one_fit
    )
    shared_config = config.site_rank_one.shared_rank_one.hierarchical.shared
    active_count = shared_config.active_count
    euclidean_floor = shared_config.euclidean_floor
    coefficient_ridge = shared_config.oracle_coefficient_ridge
    if (
        parent_fit.bank.generator_count != shared_config.generator_count
        or parent_fit.bank.generator_rank != shared_config.generator_rank
        or parent_fit.bank.gauge_floor != float(config.gauge_floor)
    ):
        raise ValueError("site rank-one fit does not match targeted geometry")
    initial, initial_global = _validate_parent_fit_against_captures(
        parent_fit,
        captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    continued_control = _fit_continued_target_site_rank_one_control(
        captures,
        parent_fit=parent_fit,
        config=config,
        initial=initial,
        initial_global=initial_global,
    )
    best_bank = _initialize_targeted_site_rank_two_bank(
        parent_fit.bank,
        gauge_floor=config.gauge_floor,
    )
    # Zero-left nested initialization must preserve every candidate bit-for-bit.
    for capture in captures:
        if not torch.equal(
            best_bank.candidate_outputs(capture),
            parent_fit.bank.candidate_outputs(capture),
        ):
            raise RuntimeError("targeted rank-two initialization is not bit-exact")
    nested_initial = evaluate_sparse_attention_transport_bank(
        best_bank,
        captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    global_captures = _global_captures(captures)
    nested_initial_global = evaluate_sparse_attention_transport_bank(
        best_bank,
        global_captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    if nested_initial != initial or nested_initial_global != initial_global:
        raise RuntimeError("targeted rank-two nested evaluation does not close")
    best_evaluation = initial
    best_global_evaluation = initial_global
    best_sites = _target_site_oracle_evaluations(
        best_bank,
        global_captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    history = [float(initial.relative_fisher_loss)]
    global_history = [float(initial_global.relative_fisher_loss)]
    site_histories = {
        key: [float(value.relative_fisher_loss)]
        for key, value in best_sites.items()
    }
    warmup_decrease = {
        key: 0.0 for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    }
    warmup_accepted_learning_rate = {
        key: 0.0 for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    }
    warmup_backtrack_count = {
        key: 0 for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    }

    for round_index in range(1, config.alternating_rounds + 1):
        for target_index, invocation_id in enumerate(
            TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        ):
            site_index = GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
            site_captures = tuple(
                capture
                for capture in captures
                if capture.invocation_id == invocation_id
            )
            coefficients = _exact_sparse_coefficients_by_capture(
                best_bank,
                site_captures,
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            pairs = tuple(zip(site_captures, coefficients, strict=True))
            if not pairs:
                raise RuntimeError("targeted rank-two fit has no target captures")
            row_count = float(sum(capture.row_count for capture, _ in pairs))

            def objective(
                first_left: Tensor,
                first_right: Tensor,
                second_left: Tensor,
                second_right: Tensor,
            ) -> Tensor:
                unit_first_right = first_right / torch.linalg.vector_norm(
                    first_right,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(float(config.gauge_floor))
                unit_second_right = second_right / torch.linalg.vector_norm(
                    second_right,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(float(config.gauge_floor))
                value = torch.zeros((), dtype=torch.float64)
                for capture, beta in pairs:
                    base = HierarchicalFisherAttentionTransportGeneratorBank.candidate_outputs_from_head_contexts(
                        best_bank,
                        invocation_id=invocation_id,
                        head_contexts=capture.head_contexts,
                    )
                    contextual = best_bank.contextual_inputs_from_head_contexts(
                        invocation_id=invocation_id,
                        head_contexts=capture.head_contexts,
                    )
                    first_activation = torch.einsum(
                        "nkh,kh->nk",
                        contextual,
                        unit_first_right,
                    )
                    second_activation = torch.einsum(
                        "nkh,kh->nk",
                        contextual,
                        unit_second_right,
                    )
                    candidates = (
                        base
                        + first_activation[:, :, None] * first_left[None]
                        + second_activation[:, :, None] * second_left[None]
                    )
                    prediction = torch.einsum("nk,nkh->nh", beta, candidates)
                    value = value + samplewise_joint_fisher_loss(
                        prediction,
                        capture.valid_operator_outputs,
                        capture.valid_score_gradients,
                        euclidean_floor=euclidean_floor,
                    ) * capture.row_count / row_count
                return value + float(config.residual_ridge) * (
                    first_left.square().mean() + second_left.square().mean()
                )

            initial_first_left = (
                best_bank.global_site_residual_left_vectors[site_index].clone()
            )
            initial_first_right = (
                best_bank.global_site_residual_right_vectors[site_index].clone()
            )
            initial_second_left = (
                best_bank.second_residual_left_vectors[target_index].clone()
            )
            initial_second_right = (
                best_bank.second_residual_right_vectors[target_index].clone()
            )

            warm_checkpoint = initial_second_left.clone()
            if round_index == 1 and config.second_component_warmup_steps > 0:
                warm_left = nn.Parameter(initial_second_left.clone())
                warm_initial_tensor = objective(
                    initial_first_left,
                    initial_first_right,
                    warm_left,
                    initial_second_right,
                )
                if not bool(torch.isfinite(warm_initial_tensor)):
                    raise RuntimeError("targeted second-component preflight is nonfinite")
                warm_gradient = torch.autograd.grad(
                    warm_initial_tensor,
                    warm_left,
                    retain_graph=False,
                )[0]
                gradient_norm = float(torch.linalg.vector_norm(warm_gradient))
                if not math.isfinite(gradient_norm):
                    raise RuntimeError("targeted second-component gradient is nonfinite")
                warm_initial = float(warm_initial_tensor.detach())
                if gradient_norm <= 1.0e-15:
                    raise RuntimeError(
                        "second component has no usable left-factor gradient: "
                        f"invocation_id={invocation_id} "
                        f"initial_loss={warm_initial:.17g} "
                        f"gradient_norm={gradient_norm:.17g}"
                    )
                tolerance = 1.0e-15 * max(1.0, abs(warm_initial))
                warm_best = math.inf
                best_trial = math.inf
                accepted_learning_rate: float | None = None
                accepted_backtrack_count: int | None = None
                warm_optimizer: torch.optim.Adam | None = None
                smallest_learning_rate = float(config.learning_rate)
                for backtrack_count in range(config.warmup_max_backtracks + 1):
                    trial_learning_rate = float(config.learning_rate) * (
                        float(config.warmup_backtrack_factor) ** backtrack_count
                    )
                    smallest_learning_rate = trial_learning_rate
                    trial_left = nn.Parameter(initial_second_left.clone())
                    trial_optimizer = torch.optim.Adam(
                        (trial_left,),
                        lr=trial_learning_rate,
                    )
                    trial_optimizer.zero_grad(set_to_none=True)
                    trial_loss = objective(
                        initial_first_left,
                        initial_first_right,
                        trial_left,
                        initial_second_right,
                    )
                    if not bool(torch.isfinite(trial_loss)):
                        raise RuntimeError(
                            "targeted second-component backtracking preflight is nonfinite"
                        )
                    trial_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (trial_left,),
                        max_norm=float(config.gradient_clip_norm),
                        error_if_nonfinite=True,
                    )
                    trial_optimizer.step()
                    observed_tensor = objective(
                        initial_first_left,
                        initial_first_right,
                        trial_left,
                        initial_second_right,
                    )
                    if not bool(torch.isfinite(observed_tensor)):
                        continue
                    observed = float(observed_tensor.detach())
                    best_trial = min(best_trial, observed)
                    if observed < warm_initial - tolerance:
                        warm_left = trial_left
                        warm_optimizer = trial_optimizer
                        warm_best = observed
                        warm_checkpoint = warm_left.detach().clone()
                        accepted_learning_rate = trial_learning_rate
                        accepted_backtrack_count = backtrack_count
                        break
                if (
                    warm_optimizer is None
                    or accepted_learning_rate is None
                    or accepted_backtrack_count is None
                ):
                    raise RuntimeError(
                        "second-component activation warmup did not decrease loss: "
                        f"invocation_id={invocation_id} "
                        f"initial_loss={warm_initial:.17g} "
                        f"best_trial_loss={best_trial:.17g} "
                        f"gradient_norm={gradient_norm:.17g} "
                        f"smallest_learning_rate={smallest_learning_rate:.17g} "
                        f"max_backtracks={config.warmup_max_backtracks}"
                    )
                # The accepted trial is the first of the fixed warmup-step
                # budget.  Retain its Adam moments, then complete exactly W-1
                # more updates at the accepted learning rate.
                for _ in range(1, config.second_component_warmup_steps):
                    warm_optimizer.zero_grad(set_to_none=True)
                    warm_loss = objective(
                        initial_first_left,
                        initial_first_right,
                        warm_left,
                        initial_second_right,
                    )
                    if not bool(torch.isfinite(warm_loss)):
                        raise RuntimeError("targeted second-component warmup is nonfinite")
                    warm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (warm_left,),
                        max_norm=float(config.gradient_clip_norm),
                        error_if_nonfinite=True,
                    )
                    warm_optimizer.step()
                    observed = float(
                        objective(
                            initial_first_left,
                            initial_first_right,
                            warm_left,
                            initial_second_right,
                        ).detach()
                    )
                    if observed < warm_best:
                        warm_best = observed
                        warm_checkpoint = warm_left.detach().clone()
                if warm_best >= warm_initial - tolerance:
                    raise RuntimeError(
                        "accepted second-component activation warmup lost its "
                        f"strict decrease: invocation_id={invocation_id} "
                        f"initial_loss={warm_initial:.17g} "
                        f"best_loss={warm_best:.17g} "
                        f"accepted_learning_rate={accepted_learning_rate:.17g}"
                    )
                warmup_decrease[invocation_id] += (
                    warm_initial - warm_best
                ) / max(abs(warm_initial), 1.0e-30)
                warmup_accepted_learning_rate[invocation_id] = (
                    accepted_learning_rate
                )
                warmup_backtrack_count[invocation_id] = (
                    accepted_backtrack_count
                )

            first_left = nn.Parameter(initial_first_left)
            first_right = nn.Parameter(initial_first_right)
            second_left = nn.Parameter(warm_checkpoint)
            second_right = nn.Parameter(initial_second_right)
            optimizer = torch.optim.Adam(
                (first_left, first_right, second_left, second_right),
                lr=float(config.learning_rate),
            )
            best_fixed = math.inf
            checkpoint: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
            for _ in range(config.steps_per_target_site_per_round):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(
                    first_left,
                    first_right,
                    second_left,
                    second_right,
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("joint targeted rank-two fit became nonfinite")
                if float(loss.detach()) < best_fixed:
                    best_fixed = float(loss.detach())
                    checkpoint = (
                        first_left.detach().clone(),
                        first_right.detach().clone(),
                        second_left.detach().clone(),
                        second_right.detach().clone(),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (first_left, first_right, second_left, second_right),
                    max_norm=float(config.gradient_clip_norm),
                    error_if_nonfinite=True,
                )
                optimizer.step()
            terminal = objective(
                first_left,
                first_right,
                second_left,
                second_right,
            )
            if not bool(torch.isfinite(terminal)):
                raise RuntimeError("joint targeted terminal loss is nonfinite")
            if float(terminal.detach()) < best_fixed:
                checkpoint = (
                    first_left.detach().clone(),
                    first_right.detach().clone(),
                    second_left.detach().clone(),
                    second_right.detach().clone(),
                )
            if checkpoint is None:
                raise RuntimeError("joint targeted fit retained no checkpoint")
            checkpoint_first_left, checkpoint_first_right, checkpoint_second_left, checkpoint_second_right = checkpoint
            checkpoint_first_right = checkpoint_first_right / torch.linalg.vector_norm(
                checkpoint_first_right,
                dim=-1,
                keepdim=True,
            ).clamp_min(float(config.gauge_floor))
            checkpoint_second_right = checkpoint_second_right / torch.linalg.vector_norm(
                checkpoint_second_right,
                dim=-1,
                keepdim=True,
            ).clamp_min(float(config.gauge_floor))
            candidate_first_left = torch.stack(
                tuple(
                    best_bank.global_site_residual_left_vectors[
                        GLOBAL_ATTENTION_INVOCATION_IDS.index(key)
                    ]
                    for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
                )
            )
            candidate_first_right = torch.stack(
                tuple(
                    best_bank.global_site_residual_right_vectors[
                        GLOBAL_ATTENTION_INVOCATION_IDS.index(key)
                    ]
                    for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
                )
            )
            candidate_second_left = best_bank.second_residual_left_vectors.clone()
            candidate_second_right = best_bank.second_residual_right_vectors.clone()
            candidate_first_left[target_index] = checkpoint_first_left
            candidate_first_right[target_index] = checkpoint_first_right
            candidate_second_left[target_index] = checkpoint_second_left
            candidate_second_right[target_index] = checkpoint_second_right
            candidate = _targeted_bank_with_factors(
                best_bank,
                candidate_first_left,
                candidate_first_right,
                candidate_second_left,
                candidate_second_right,
            )
            candidate_site = evaluate_sparse_attention_transport_bank(
                candidate,
                site_captures,
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            if (
                candidate_site.relative_fisher_loss
                < best_sites[invocation_id].relative_fisher_loss - 1.0e-12
            ):
                best_bank = candidate
                best_sites[invocation_id] = candidate_site

        evaluation = evaluate_sparse_attention_transport_bank(
            best_bank,
            captures,
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        global_evaluation = evaluate_sparse_attention_transport_bank(
            best_bank,
            global_captures,
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        if (
            evaluation.relative_fisher_loss
            > best_evaluation.relative_fisher_loss + 1.0e-12
            or global_evaluation.relative_fisher_loss
            > best_global_evaluation.relative_fisher_loss + 1.0e-12
        ):
            raise RuntimeError("accepted targeted rank-two update regressed")
        best_evaluation = evaluation
        best_global_evaluation = global_evaluation
        history.append(float(evaluation.relative_fisher_loss))
        global_history.append(float(global_evaluation.relative_fisher_loss))
        for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS:
            site_histories[key].append(float(best_sites[key].relative_fisher_loss))
        if round_observer is not None:
            round_observer(
                round_index,
                config.alternating_rounds,
                evaluation,
                global_evaluation,
            )

    # Non-target component one is inherited exactly; local outputs never receive
    # either targeted component by construction.
    mutation_35_index = GLOBAL_ATTENTION_INVOCATION_IDS.index("mutation.35")
    if (
        not torch.equal(
            best_bank.global_site_residual_left_vectors[mutation_35_index],
            parent_fit.bank.global_site_residual_left_vectors[mutation_35_index],
        )
        or not torch.equal(
            best_bank.global_site_residual_right_vectors[mutation_35_index],
            parent_fit.bank.global_site_residual_right_vectors[mutation_35_index],
        )
    ):
        raise RuntimeError("mutation.35 changed during targeted rank-two fitting")
    return TargetedSiteRankTwoAttentionTransportBankFit(
        bank=best_bank,
        site_rank_one_fit=parent_fit,
        continued_site_rank_one_control=continued_control,
        initial_fit_oracle=initial,
        final_fit_oracle=best_evaluation,
        initial_global_fit_oracle=initial_global,
        final_global_fit_oracle=best_global_evaluation,
        fit_round_relative_fisher_losses=tuple(history),
        fit_round_global_relative_fisher_losses=tuple(global_history),
        fit_round_target_site_relative_fisher_losses=tuple(
            (key, tuple(site_histories[key]))
            for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        ),
        warmup_fixed_support_relative_decreases=tuple(
            (key, float(warmup_decrease[key]))
            for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        ),
        warmup_accepted_learning_rates=tuple(
            (key, float(warmup_accepted_learning_rate[key]))
            for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        ),
        warmup_backtrack_counts=tuple(
            (key, int(warmup_backtrack_count[key]))
            for key in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        ),
        activation_warmup_steps_per_target_site=(
            config.second_component_warmup_steps
        ),
        joint_optimizer_steps_per_target_site_per_round=(
            config.steps_per_target_site_per_round
        ),
        optimizer_steps_per_target_site_by_round=(
            config.second_component_warmup_steps
            + config.steps_per_target_site_per_round,
            *(
                config.steps_per_target_site_per_round
                for _ in range(config.alternating_rounds - 1)
            ),
        ),
    )


def evaluate_targeted_site_rank_two_coupling_diagnostic(
    bank: TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
    a_captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int = 2,
    euclidean_floor: float = 1.0e-3,
    coefficient_ridge: float = 0.0,
) -> GlobalRankOneCouplingDiagnostic:
    """Measure tied and untied ceilings for base versus total residual.

    The residual group is the sum of component one and component two.  Support
    remains counted by generator identity, so the untied cases only decouple
    the hierarchical-base and total-residual amplitudes; no targets,
    coefficients, supports, gradients, or activations are retained.
    """

    if not isinstance(
        bank,
        TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("bank must be a targeted site rank-two bank")
    bank._validate_targeted_unchanged()
    captures = _global_captures(_capture_values(a_captures))
    if type(active_count) is not int or not 0 < active_count <= bank.generator_count:
        raise ValueError("active_count is outside the generator inventory")
    for label, value, positive in (
        ("euclidean_floor", euclidean_floor, True),
        ("coefficient_ridge", coefficient_ridge, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (positive and float(value) <= 0.0)
            or (not positive and float(value) < 0.0)
        ):
            raise ValueError(f"{label} is invalid")

    per_site: list[GlobalRankOneSiteCouplingDiagnostic] = []
    predictions: dict[str, list[Tensor]] = {
        "baseline": [],
        "tied": [],
        "fixed_support_untied": [],
        "reselected_support_untied": [],
    }
    targets: list[Tensor] = []
    gradients: list[Tensor] = []
    for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
        site_captures = tuple(
            capture
            for capture in captures
            if capture.invocation_id == invocation_id
        )
        base = torch.cat(
            tuple(
                HierarchicalFisherAttentionTransportGeneratorBank.candidate_outputs_from_head_contexts(
                    bank,
                    invocation_id=invocation_id,
                    head_contexts=capture.head_contexts,
                )
                for capture in site_captures
            )
        )
        residual = torch.cat(
            tuple(
                bank.total_global_site_candidate_residuals_from_head_contexts(
                    invocation_id=invocation_id,
                    head_contexts=capture.head_contexts,
                )
                for capture in site_captures
            )
        )
        target = torch.cat(
            tuple(capture.valid_operator_outputs for capture in site_captures)
        )
        gradient = torch.cat(
            tuple(capture.valid_score_gradients for capture in site_captures)
        )
        baseline_prediction, _baseline_support = _solve_tied_diagnostic_oracle(
            base,
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
        )
        tied_prediction, tied_support = _solve_tied_diagnostic_oracle(
            base + residual,
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
        )
        fixed_prediction = _solve_grouped_untied_diagnostic_oracle(
            base,
            residual,
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            fixed_generator_support=tied_support,
        )
        reselected_prediction = _solve_grouped_untied_diagnostic_oracle(
            base,
            residual,
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            fixed_generator_support=None,
        )
        site_evaluations = {
            "baseline": _diagnostic_oracle_evaluation(
                baseline_prediction,
                target,
                gradient,
                active_count=active_count,
                euclidean_floor=float(euclidean_floor),
                coefficient_ridge=float(coefficient_ridge),
                solver_kind=GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER,
            ),
            "tied": _diagnostic_oracle_evaluation(
                tied_prediction,
                target,
                gradient,
                active_count=active_count,
                euclidean_floor=float(euclidean_floor),
                coefficient_ridge=float(coefficient_ridge),
                solver_kind=GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER,
            ),
            "fixed": _diagnostic_oracle_evaluation(
                fixed_prediction,
                target,
                gradient,
                active_count=active_count,
                euclidean_floor=float(euclidean_floor),
                coefficient_ridge=float(coefficient_ridge),
                solver_kind=GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
            ),
            "reselected": _diagnostic_oracle_evaluation(
                reselected_prediction,
                target,
                gradient,
                active_count=active_count,
                euclidean_floor=float(euclidean_floor),
                coefficient_ridge=float(coefficient_ridge),
                solver_kind=GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
            ),
        }
        per_site.append(
            GlobalRankOneSiteCouplingDiagnostic(
                invocation_id=invocation_id,
                baseline=site_evaluations["baseline"],
                tied=site_evaluations["tied"],
                fixed_support_untied=site_evaluations["fixed"],
                reselected_support_untied=site_evaluations["reselected"],
            )
        )
        for key, prediction in (
            ("baseline", baseline_prediction),
            ("tied", tied_prediction),
            ("fixed_support_untied", fixed_prediction),
            ("reselected_support_untied", reselected_prediction),
        ):
            predictions[key].append(prediction)
        targets.append(target)
        gradients.append(gradient)

    target = torch.cat(targets)
    gradient = torch.cat(gradients)
    aggregate = {
        key: _diagnostic_oracle_evaluation(
            torch.cat(predictions[key]),
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            solver_kind=solver,
        )
        for key, solver in (
            ("baseline", GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER),
            ("tied", GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER),
            (
                "fixed_support_untied",
                GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
            ),
            (
                "reselected_support_untied",
                GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
            ),
        )
    }
    total_rows = sum(value.baseline.row_count for value in per_site)
    global_zero = aggregate["baseline"].zero_predictor_fisher_loss
    leave_one_out = tuple(
        (
            invocation_id,
            float(
                sum(
                    site.baseline.row_count
                    * (
                        site.baseline.fisher_loss
                        if site.invocation_id == invocation_id
                        else site.tied.fisher_loss
                    )
                    for site in per_site
                )
                / total_rows
                / global_zero
            ),
        )
        for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    )
    return GlobalRankOneCouplingDiagnostic(
        global_attention_invocation_ids=GLOBAL_ATTENTION_INVOCATION_IDS,
        active_generator_count_limit=active_count,
        coefficient_ridge=float(coefficient_ridge),
        ridge_parameterization=GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION,
        baseline=aggregate["baseline"],
        tied=aggregate["tied"],
        fixed_support_untied=aggregate["fixed_support_untied"],
        reselected_support_untied=aggregate["reselected_support_untied"],
        leave_one_site_residual_off_relative_fisher_losses=leave_one_out,
        per_site=tuple(per_site),
    )


@dataclass(frozen=True, slots=True)
class AllLayerSparseBankDiagnostic:
    """Aggregate and full per-invocation exact sparse metrics for one bank."""

    bank_kind: str
    all_layers: SparseAttentionTransportOracleEvaluation
    local_attention: SparseAttentionTransportOracleEvaluation
    global_attention: SparseAttentionTransportOracleEvaluation
    per_invocation: tuple[
        tuple[str, SparseAttentionTransportOracleEvaluation], ...
    ]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bank_kind, str)
            or not self.bank_kind
            or not isinstance(
                self.all_layers,
                SparseAttentionTransportOracleEvaluation,
            )
            or not isinstance(
                self.local_attention,
                SparseAttentionTransportOracleEvaluation,
            )
            or not isinstance(
                self.global_attention,
                SparseAttentionTransportOracleEvaluation,
            )
            or not self.per_invocation
            or len({key for key, _value in self.per_invocation})
            != len(self.per_invocation)
            or any(
                not isinstance(key, str)
                or not isinstance(value, SparseAttentionTransportOracleEvaluation)
                or tuple(
                    invocation_id
                    for invocation_id, _relative in (
                        value.per_invocation_relative_fisher_loss
                    )
                )
                != (key,)
                for key, value in self.per_invocation
            )
            or {
                key
                for key, _value in self.all_layers.per_invocation_relative_fisher_loss
            }
            != {key for key, _value in self.per_invocation}
        ):
            raise ValueError("all-layer sparse bank diagnostic is invalid")
        total_rows = sum(
            value.output_metrics.row_count for _key, value in self.per_invocation
        )
        if total_rows != self.all_layers.output_metrics.row_count:
            raise ValueError("all-layer bank diagnostic row counts do not close")
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
            or any(
                not math.isclose(
                    value.relative_fisher_loss,
                    dict(
                        self.all_layers.per_invocation_relative_fisher_loss
                    )[key],
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                )
                for key, value in self.per_invocation
            )
        ):
            raise ValueError("all-layer bank diagnostic metrics do not close")
        local_ids = tuple(
            key
            for key, _value in self.per_invocation
            if key not in GLOBAL_ATTENTION_INVOCATION_IDS
        )
        if (
            {
                key
                for key, _value in self.local_attention.per_invocation_relative_fisher_loss
            }
            != set(local_ids)
            or {
                key
                for key, _value in self.global_attention.per_invocation_relative_fisher_loss
            }
            != set(GLOBAL_ATTENTION_INVOCATION_IDS)
            or self.local_attention.output_metrics.row_count
            + self.global_attention.output_metrics.row_count
            != self.all_layers.output_metrics.row_count
        ):
            raise ValueError("all-layer local/global partitions do not close")

    @property
    def aggregate(self) -> SparseAttentionTransportOracleEvaluation:
        """Backward-compatible alias for the all-layer aggregate."""

        return self.all_layers

    @property
    def per_invocation_fisher_loss_sums(self) -> tuple[tuple[str, float], ...]:
        """Return row-weighted sums used to rank absolute error burden."""

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
class AllLayerAttentionTransportDiagnostic:
    """A metrics-only five-bank comparison covering all 18 V/O sites."""

    invocation_ids: tuple[str, ...]
    active_generator_count_limit: int
    coefficient_ridge: float
    hierarchical: AllLayerSparseBankDiagnostic
    shared_rank_one: AllLayerSparseBankDiagnostic
    site_rank_one: AllLayerSparseBankDiagnostic
    continued_site_rank_one: AllLayerSparseBankDiagnostic
    targeted_site_rank_two: AllLayerSparseBankDiagnostic

    def __post_init__(self) -> None:
        banks = (
            self.hierarchical,
            self.shared_rank_one,
            self.site_rank_one,
            self.continued_site_rank_one,
            self.targeted_site_rank_two,
        )
        if (
            len(self.invocation_ids) != 18
            or len(set(self.invocation_ids)) != 18
            or type(self.active_generator_count_limit) is not int
            or self.active_generator_count_limit <= 0
            or isinstance(self.coefficient_ridge, bool)
            or not isinstance(self.coefficient_ridge, (int, float))
            or not math.isfinite(float(self.coefficient_ridge))
            or float(self.coefficient_ridge) < 0.0
            or any(not isinstance(value, AllLayerSparseBankDiagnostic) for value in banks)
            or tuple(value.bank_kind for value in banks)
            != (
                "hierarchical",
                "shared_rank_one",
                "site_rank_one",
                "continued_site_rank_one",
                "targeted_site_rank_two",
            )
            or any(
                tuple(key for key, _value in bank.per_invocation)
                != self.invocation_ids
                or bank.aggregate.active_count_limit
                != self.active_generator_count_limit
                or bank.aggregate.coefficient_ridge
                != float(self.coefficient_ridge)
                for bank in banks
            )
            or len({bank.aggregate.output_metrics.row_count for bank in banks}) != 1
        ):
            raise ValueError("all-layer attention transport diagnostic is invalid")


def _hierarchical_view(
    bank: GlobalRankOneFisherAttentionTransportGeneratorBank,
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    return HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=bank.left_factors,
        singular_scales=bank.singular_scales,
        right_factors=bank.right_factors,
        head_coefficients=bank.head_coefficients,
        head_ids=bank.head_ids,
        hierarchy=bank.hierarchy,
        block_scale_contrasts=bank.block_scale_contrasts,
        site_scale_contrasts=bank.site_scale_contrasts,
        signed_feature_floor=bank.signed_feature_floor,
    )


def _all_layer_bank_diagnostic(
    bank_kind: str,
    bank: HierarchicalFisherAttentionTransportGeneratorBank,
    captures: tuple[AttentionTransportBatch, ...],
    invocation_ids: tuple[str, ...],
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> AllLayerSparseBankDiagnostic:
    aggregate = evaluate_sparse_attention_transport_bank(
        bank,
        captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    return AllLayerSparseBankDiagnostic(
        bank_kind=bank_kind,
        all_layers=aggregate,
        local_attention=evaluate_sparse_attention_transport_bank(
            bank,
            tuple(
                capture
                for capture in captures
                if capture.invocation_id not in GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        ),
        global_attention=evaluate_sparse_attention_transport_bank(
            bank,
            tuple(
                capture
                for capture in captures
                if capture.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        ),
        per_invocation=tuple(
            (
                invocation_id,
                evaluate_sparse_attention_transport_bank(
                    bank,
                    tuple(
                        capture
                        for capture in captures
                        if capture.invocation_id == invocation_id
                    ),
                    active_count=active_count,
                    euclidean_floor=euclidean_floor,
                    coefficient_ridge=coefficient_ridge,
                ),
            )
            for invocation_id in invocation_ids
        ),
    )


def evaluate_all_layer_attention_transport_diagnostic(
    fit: TargetedSiteRankTwoAttentionTransportBankFit,
    a_captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int = 2,
    euclidean_floor: float = 1.0e-3,
    coefficient_ridge: float = 0.0,
) -> AllLayerAttentionTransportDiagnostic:
    """Evaluate all 18 layers for every nested bank and the fair control."""

    if not isinstance(fit, TargetedSiteRankTwoAttentionTransportBankFit):
        raise TypeError("fit must be a targeted site rank-two bank fit")
    captures = _capture_values(a_captures)
    invocation_ids = tuple(fit.bank.hierarchy.invocation_ids)
    if (
        len(invocation_ids) != 18
        or {capture.invocation_id for capture in captures} != set(invocation_ids)
    ):
        raise ValueError("all-layer diagnostic requires all 18 invocations")
    if type(active_count) is not int or not 0 < active_count <= fit.bank.generator_count:
        raise ValueError("active_count is outside the generator inventory")
    for label, value, positive in (
        ("euclidean_floor", euclidean_floor, True),
        ("coefficient_ridge", coefficient_ridge, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (positive and float(value) <= 0.0)
            or (not positive and float(value) < 0.0)
        ):
            raise ValueError(f"{label} is invalid")
    shared = fit.site_rank_one_fit.shared_rank_one_fit.bank
    hierarchy = _hierarchical_view(shared)
    site = fit.site_rank_one_fit.bank
    control = fit.continued_site_rank_one_control.bank
    banks = tuple(
        _all_layer_bank_diagnostic(
            kind,
            bank,
            captures,
            invocation_ids,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
        )
        for kind, bank in (
            ("hierarchical", hierarchy),
            ("shared_rank_one", shared),
            ("site_rank_one", site),
            ("continued_site_rank_one", control),
            ("targeted_site_rank_two", fit.bank),
        )
    )
    return AllLayerAttentionTransportDiagnostic(
        invocation_ids=invocation_ids,
        active_generator_count_limit=active_count,
        coefficient_ridge=float(coefficient_ridge),
        hierarchical=banks[0],
        shared_rank_one=banks[1],
        site_rank_one=banks[2],
        continued_site_rank_one=banks[3],
        targeted_site_rank_two=banks[4],
    )


__all__ = [
    "AllLayerAttentionTransportDiagnostic",
    "AllLayerSparseBankDiagnostic",
    "ContinuedTargetSiteRankOneControlFit",
    "TARGETED_SITE_RANK_PROFILE",
    "TARGETED_SITE_RANK_TWO_ACCEPTANCE_SEMANTICS",
    "TARGETED_SITE_RANK_TWO_COMPONENT_ORDER_SEMANTICS",
    "TARGETED_SITE_RANK_TWO_GAUGE_SEMANTICS",
    "TARGETED_SITE_RANK_TWO_INITIALIZATION_SEMANTICS",
    "TARGETED_SITE_RANK_TWO_INVOCATION_IDS",
    "TARGETED_SITE_RANK_TWO_RESIDUAL_SEMANTICS",
    "TARGETED_SITE_RANK_TWO_WARMUP_SEMANTICS",
    "TargetedSiteRankTwoAttentionTransportBankFit",
    "TargetedSiteRankTwoAttentionTransportFitConfig",
    "TargetedSiteRankTwoFisherAttentionTransportGeneratorBank",
    "evaluate_all_layer_attention_transport_diagnostic",
    "evaluate_targeted_site_rank_two_coupling_diagnostic",
    "fit_targeted_site_rank_two_attention_transport_generator_bank",
    "targeted_site_rank_two_fisher_attention_transport_generator_bank_from_state_dict",
]
