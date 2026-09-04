"""Site-specific rank-one residuals for Gemma attention transport.

This additive A-only rung starts from the completed shared global rank-one
fit and broadcasts that residual into three independent banks, one for each
global-attention invocation.  The deployed candidate at global site ``i`` is

    C[i,k](xi) = G_hier[i,k](xi) + l[i,k] (r[i,k]^T xi),

while every local-attention invocation is exactly the hierarchical candidate.
Each ``r[i,k]`` has unit norm with a deterministic positive max-absolute
pivot, so all scale lives in ``l[i,k]``.  The residual inventory is therefore
exactly ``3 * 2 * K * H`` stored parameters.

Fitting alternates exact signed Top-2 coefficients with fixed-support factor
optimization.  A site update is retained only when the exact sparse Fisher
oracle for that same site improves strictly.  The coupling diagnostic is
A-only and returns the existing metrics-only diagnostic type; it never keeps
oracle supports, coefficients, targets, gradients, or activations.
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
    _metrics,
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
    GlobalRankOneAttentionTransportBankFit,
    GlobalRankOneAttentionTransportFitConfig,
    GlobalRankOneCouplingDiagnostic,
    GlobalRankOneFisherAttentionTransportGeneratorBank,
    GlobalRankOneSiteCouplingDiagnostic,
    _canonicalize_rank_one_factors,
    _diagnostic_oracle_evaluation,
    _exact_sparse_coefficients_by_capture,
    _fixed_support_prediction,
    _global_captures,
    _is_canonical_rank_one_gauge,
    _solve_grouped_untied_diagnostic_oracle,
    _solve_tied_diagnostic_oracle,
    fit_global_rank_one_attention_transport_generator_bank,
)
from .gemma3_attention_transport_hierarchical_edge import (
    HierarchicalFisherAttentionTransportGeneratorBank,
    _capture_values,
    hierarchical_fisher_attention_transport_generator_bank_from_state_dict,
)
from .gemma3_attention_transport_sparse_v2 import (
    SparseAttentionTransportOracleEvaluation,
    evaluate_sparse_attention_transport_bank,
)


GLOBAL_SITE_RANK_ONE_RESIDUAL_SEMANTICS = (
    "global_attention_site_specific_rank_one_residual_v1"
)
GLOBAL_SITE_RANK_ONE_GAUGE_SEMANTICS = (
    "unit_right_per_site_generator_max_abs_positive_scale_in_left_v1"
)
GLOBAL_SITE_RANK_ONE_INITIALIZATION_SEMANTICS = (
    "broadcast_completed_shared_global_rank_one_fit_v1"
)
GLOBAL_SITE_RANK_ONE_ACCEPTANCE_SEMANTICS = (
    "strict_exact_per_site_sparse_fisher_oracle_improvement_v1"
)

_BANK_DOMAIN = b"fisher_graph:attention-transport-global-site-rank-one-bank:v1\0"


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


def _site_index(invocation_id: str) -> int | None:
    if not isinstance(invocation_id, str):
        raise TypeError("invocation_id must be a string")
    try:
        return GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
    except ValueError:
        return None


def _canonicalize_site_rank_one_factors(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> tuple[Tensor, Tensor]:
    left_value = _cpu_float64(
        left,
        label="global site residual left vectors",
        ndim=3,
    )
    right_value = _cpu_float64(
        right,
        label="global site residual right vectors",
        ndim=3,
    )
    if left_value.shape != right_value.shape:
        raise ValueError("global site residual factor shapes do not match")
    if left_value.shape[0] != len(GLOBAL_ATTENTION_INVOCATION_IDS):
        raise ValueError("global site residual factors have the wrong site count")
    sites, generators, hidden = left_value.shape
    flattened_left = left_value.reshape(sites * generators, hidden)
    flattened_right = right_value.reshape(sites * generators, hidden)
    # Broadcasting a completed shared fit must be bit-exact.  Re-normalizing
    # factors that are already in the accepted canonical gauge can introduce
    # an otherwise harmless ULP of drift and break that initialization
    # invariant, so retain canonical input tensors verbatim.
    if _is_canonical_rank_one_gauge(
        flattened_left,
        flattened_right,
        gauge_floor=gauge_floor,
    ):
        return left_value.clone().contiguous(), right_value.clone().contiguous()
    canonical_left, canonical_right = _canonicalize_rank_one_factors(
        flattened_left,
        flattened_right,
        gauge_floor=gauge_floor,
    )
    return (
        canonical_left.reshape(sites, generators, hidden).contiguous(),
        canonical_right.reshape(sites, generators, hidden).contiguous(),
    )


def _is_canonical_site_rank_one_gauge(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> bool:
    try:
        left_value = _cpu_float64(
            left,
            label="global site residual left vectors",
            ndim=3,
        )
        right_value = _cpu_float64(
            right,
            label="global site residual right vectors",
            ndim=3,
        )
        if left_value.shape != right_value.shape or left_value.shape[0] != 3:
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
class GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
    HierarchicalFisherAttentionTransportGeneratorBank
):
    """Hierarchical bank plus an independent rank-one residual per global site."""

    global_site_residual_left_vectors: Tensor
    global_site_residual_right_vectors: Tensor
    shared_rank_one_bank_sha256: str
    gauge_floor: float = 1.0e-12
    _global_site_rank_one_tensor_sha256s: tuple[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _hierarchical_bank_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _global_site_rank_one_artifact_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        HierarchicalFisherAttentionTransportGeneratorBank.__post_init__(self)
        if (
            isinstance(self.gauge_floor, bool)
            or not isinstance(self.gauge_floor, (int, float))
            or not math.isfinite(float(self.gauge_floor))
            or float(self.gauge_floor) <= 0.0
        ):
            raise ValueError("global site rank-one gauge floor must be positive")
        if not _is_sha256(self.shared_rank_one_bank_sha256):
            raise ValueError("shared rank-one bank SHA-256 is invalid")
        left, right = _canonicalize_site_rank_one_factors(
            self.global_site_residual_left_vectors,
            self.global_site_residual_right_vectors,
            gauge_floor=float(self.gauge_floor),
        )
        expected = (
            len(GLOBAL_ATTENTION_INVOCATION_IDS),
            self.generator_count,
            self.hidden_width,
        )
        if left.shape != expected:
            raise ValueError("global site residual factors have the wrong geometry")
        if not set(GLOBAL_ATTENTION_INVOCATION_IDS).issubset(
            self.hierarchy.invocation_ids
        ):
            raise ValueError("hierarchy omits a required global-attention site")
        object.__setattr__(self, "global_site_residual_left_vectors", left)
        object.__setattr__(self, "global_site_residual_right_vectors", right)
        object.__setattr__(self, "gauge_floor", float(self.gauge_floor))
        tensor_hashes = (_tensor_sha256(left), _tensor_sha256(right))
        hierarchical_hash = (
            HierarchicalFisherAttentionTransportGeneratorBank.artifact_sha256.fget(
                self
            )
        )
        object.__setattr__(
            self,
            "_global_site_rank_one_tensor_sha256s",
            tensor_hashes,
        )
        object.__setattr__(self, "_hierarchical_bank_sha256", hierarchical_hash)
        payload = self._global_site_rank_one_artifact_payload(tensor_hashes)
        object.__setattr__(
            self,
            "_global_site_rank_one_artifact_sha256",
            hashlib.sha256(_BANK_DOMAIN + _canonical_json_bytes(payload)).hexdigest(),
        )

    def _global_site_rank_one_artifact_payload(
        self,
        tensor_hashes: tuple[str, str],
    ) -> dict[str, object]:
        return {
            "acceptance_semantics": GLOBAL_SITE_RANK_ONE_ACCEPTANCE_SEMANTICS,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": GLOBAL_SITE_RANK_ONE_GAUGE_SEMANTICS,
            "generator_count": self.generator_count,
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "global_site_residual_left_vectors_sha256": tensor_hashes[0],
            "global_site_residual_right_vectors_sha256": tensor_hashes[1],
            "hidden_width": self.hidden_width,
            "hierarchical_bank_sha256": self._hierarchical_bank_sha256,
            "initialization_semantics": (
                GLOBAL_SITE_RANK_ONE_INITIALIZATION_SEMANTICS
            ),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "residual_parameter_count": self.global_site_rank_one_parameter_count,
            "residual_semantics": GLOBAL_SITE_RANK_ONE_RESIDUAL_SEMANTICS,
            "router_semantics": GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
            "shared_rank_one_bank_sha256": self.shared_rank_one_bank_sha256,
        }

    def _validate_global_site_rank_one_unchanged(self) -> None:
        HierarchicalFisherAttentionTransportGeneratorBank._validate_unchanged(self)
        current = (
            _tensor_sha256(self.global_site_residual_left_vectors),
            _tensor_sha256(self.global_site_residual_right_vectors),
        )
        if current != self._global_site_rank_one_tensor_sha256s:
            raise RuntimeError(
                "global site rank-one generator bank changed after construction"
            )
        if not _is_canonical_site_rank_one_gauge(
            self.global_site_residual_left_vectors,
            self.global_site_residual_right_vectors,
            gauge_floor=self.gauge_floor,
        ):
            raise RuntimeError("global site rank-one generator gauge drifted")

    @property
    def hierarchical_bank_sha256(self) -> str:
        self._validate_global_site_rank_one_unchanged()
        return self._hierarchical_bank_sha256

    @property
    def global_site_rank_one_parameter_count(self) -> int:
        expected = (
            2
            * len(GLOBAL_ATTENTION_INVOCATION_IDS)
            * self.generator_count
            * self.hidden_width
        )
        observed = int(
            self.global_site_residual_left_vectors.numel()
            + self.global_site_residual_right_vectors.numel()
        )
        if observed != expected:
            raise RuntimeError("global site rank-one parameter count drifted")
        return observed

    @property
    def stored_parameter_count(self) -> int:
        hierarchical = (
            HierarchicalFisherAttentionTransportGeneratorBank.stored_parameter_count.fget(
                self
            )
        )
        return int(hierarchical + self.global_site_rank_one_parameter_count)

    @property
    def artifact_sha256(self) -> str:
        self._validate_global_site_rank_one_unchanged()
        return self._global_site_rank_one_artifact_sha256

    def __hash__(self) -> int:
        return hash(
            (
                GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
                self.artifact_sha256,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(
                other,
                GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
            )
            and self.artifact_sha256 == other.artifact_sha256
        )

    def global_site_rank_one_candidate_residuals_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        """Return ``[N,K,H]`` residual candidates, exactly zero off-global."""

        self._validate_global_site_rank_one_unchanged()
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        index = _site_index(invocation_id)
        if index is None:
            return torch.zeros_like(contextual)
        left = self.global_site_residual_left_vectors[index]
        right = self.global_site_residual_right_vectors[index]
        activation = torch.einsum("nkh,kh->nk", contextual, right)
        return (activation[:, :, None] * left[None]).contiguous()

    def candidate_outputs_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        base = (
            HierarchicalFisherAttentionTransportGeneratorBank.candidate_outputs_from_head_contexts(
                self,
                invocation_id=invocation_id,
                head_contexts=head_contexts,
            )
        )
        if _site_index(invocation_id) is None:
            return base
        return (
            base
            + self.global_site_rank_one_candidate_residuals_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=head_contexts,
            )
        ).contiguous()

    def state_dict(self) -> dict[str, object]:
        self._validate_global_site_rank_one_unchanged()
        return {
            "acceptance_semantics": GLOBAL_SITE_RANK_ONE_ACCEPTANCE_SEMANTICS,
            "artifact_sha256": self._global_site_rank_one_artifact_sha256,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": GLOBAL_SITE_RANK_ONE_GAUGE_SEMANTICS,
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "global_site_residual_left_vectors": (
                self.global_site_residual_left_vectors.clone()
            ),
            "global_site_residual_right_vectors": (
                self.global_site_residual_right_vectors.clone()
            ),
            "hierarchical_bank": (
                HierarchicalFisherAttentionTransportGeneratorBank.state_dict(self)
            ),
            "hierarchical_bank_sha256": self._hierarchical_bank_sha256,
            "initialization_semantics": (
                GLOBAL_SITE_RANK_ONE_INITIALIZATION_SEMANTICS
            ),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "residual_semantics": GLOBAL_SITE_RANK_ONE_RESIDUAL_SEMANTICS,
            "router_semantics": GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
            "shared_rank_one_bank_sha256": self.shared_rank_one_bank_sha256,
        }


def global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> GlobalSiteRankOneFisherAttentionTransportGeneratorBank:
    """Reload a source-free site-specific bank and verify its full lineage."""

    expected = {
        "acceptance_semantics",
        "artifact_sha256",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "gauge_floor",
        "gauge_semantics",
        "global_attention_invocation_ids",
        "global_site_residual_left_vectors",
        "global_site_residual_right_vectors",
        "hierarchical_bank",
        "hierarchical_bank_sha256",
        "initialization_semantics",
        "native_output_rmsnorm_retained",
        "native_qk_retained",
        "residual_semantics",
        "router_semantics",
        "shared_rank_one_bank_sha256",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("global site rank-one bank state fields are invalid")
    raw_hierarchical = state["hierarchical_bank"]
    raw_ids = state["global_attention_invocation_ids"]
    if (
        not isinstance(raw_hierarchical, Mapping)
        or raw_ids != list(GLOBAL_ATTENTION_INVOCATION_IDS)
        or state["acceptance_semantics"]
        != GLOBAL_SITE_RANK_ONE_ACCEPTANCE_SEMANTICS
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_raw_teacher_outputs_or_gradients"] is not False
        or state["gauge_semantics"] != GLOBAL_SITE_RANK_ONE_GAUGE_SEMANTICS
        or state["initialization_semantics"]
        != GLOBAL_SITE_RANK_ONE_INITIALIZATION_SEMANTICS
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
        or state["residual_semantics"]
        != GLOBAL_SITE_RANK_ONE_RESIDUAL_SEMANTICS
        or state["router_semantics"] != GLOBAL_RANK_ONE_ROUTER_SEMANTICS
        or not _is_sha256(state["artifact_sha256"])
        or not _is_sha256(state["hierarchical_bank_sha256"])
        or not _is_sha256(state["shared_rank_one_bank_sha256"])
    ):
        raise ValueError("global site rank-one bank safety semantics are invalid")
    gauge_floor = state["gauge_floor"]
    if (
        isinstance(gauge_floor, bool)
        or not isinstance(gauge_floor, (int, float))
        or not math.isfinite(float(gauge_floor))
        or float(gauge_floor) <= 0.0
        or not _is_canonical_site_rank_one_gauge(
            state["global_site_residual_left_vectors"],
            state["global_site_residual_right_vectors"],
            gauge_floor=float(gauge_floor),
        )
    ):
        raise ValueError("global site rank-one bank state gauge is invalid")
    hierarchical = (
        hierarchical_fisher_attention_transport_generator_bank_from_state_dict(
            raw_hierarchical
        )
    )
    bank = GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
        left_factors=hierarchical.left_factors,
        singular_scales=hierarchical.singular_scales,
        right_factors=hierarchical.right_factors,
        head_coefficients=hierarchical.head_coefficients,
        head_ids=hierarchical.head_ids,
        hierarchy=hierarchical.hierarchy,
        block_scale_contrasts=hierarchical.block_scale_contrasts,
        site_scale_contrasts=hierarchical.site_scale_contrasts,
        signed_feature_floor=hierarchical.signed_feature_floor,
        global_site_residual_left_vectors=state[
            "global_site_residual_left_vectors"
        ],
        global_site_residual_right_vectors=state[
            "global_site_residual_right_vectors"
        ],
        shared_rank_one_bank_sha256=state["shared_rank_one_bank_sha256"],
        gauge_floor=float(gauge_floor),
    )
    if (
        state["hierarchical_bank_sha256"] != bank.hierarchical_bank_sha256
        or state["artifact_sha256"] != bank.artifact_sha256
    ):
        raise ValueError("global site rank-one bank artifact lineage drifted")
    return bank


@dataclass(frozen=True, slots=True)
class GlobalSiteRankOneAttentionTransportFitConfig:
    """Fixed A-only protocol for independent global-site residual banks."""

    shared_rank_one: GlobalRankOneAttentionTransportFitConfig
    alternating_rounds: int = 4
    steps_per_site_per_round: int = 48
    learning_rate: float = 1.0e-2
    residual_ridge: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    gauge_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(
            self.shared_rank_one,
            GlobalRankOneAttentionTransportFitConfig,
        ):
            raise TypeError("shared_rank_one must be a global rank-one fit config")
        if type(self.alternating_rounds) is not int or self.alternating_rounds <= 0:
            raise ValueError("alternating_rounds must be positive")
        if (
            type(self.steps_per_site_per_round) is not int
            or self.steps_per_site_per_round <= 0
        ):
            raise ValueError("steps_per_site_per_round must be positive")
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
        if float(self.gauge_floor) != float(self.shared_rank_one.gauge_floor):
            raise ValueError(
                "gauge_floor must equal the completed shared rank-one gauge floor"
            )

    @classmethod
    def fixed_k8_r8_top2(cls) -> GlobalSiteRankOneAttentionTransportFitConfig:
        return cls(
            shared_rank_one=(
                GlobalRankOneAttentionTransportFitConfig.fixed_k8_r8_top2()
            )
        )


@dataclass(frozen=True, slots=True)
class GlobalSiteRankOneAttentionTransportBankFit:
    bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank
    shared_rank_one_fit: GlobalRankOneAttentionTransportBankFit
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    initial_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_round_relative_fisher_losses: tuple[float, ...]
    fit_round_global_relative_fisher_losses: tuple[float, ...]
    fit_round_site_relative_fisher_losses: tuple[
        tuple[str, tuple[float, ...]], ...
    ]

    def __post_init__(self) -> None:
        all_history = self.fit_round_relative_fisher_losses
        global_history = self.fit_round_global_relative_fisher_losses
        site_histories = self.fit_round_site_relative_fisher_losses
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
            or not isinstance(
                self.shared_rank_one_fit,
                GlobalRankOneAttentionTransportBankFit,
            )
            or self.bank.shared_rank_one_bank_sha256
            != self.shared_rank_one_fit.bank.artifact_sha256
            or self.bank.hierarchical_bank_sha256
            != self.shared_rank_one_fit.bank.hierarchical_bank_sha256
            or self.bank.gauge_floor != self.shared_rank_one_fit.bank.gauge_floor
            or not all(
                isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in evaluations
            )
            or self.initial_fit_oracle
            != self.shared_rank_one_fit.final_fit_oracle
            or self.initial_global_fit_oracle
            != self.shared_rank_one_fit.final_global_fit_oracle
            or len(all_history) < 2
            or len(global_history) != len(all_history)
            or tuple(key for key, _values in site_histories)
            != GLOBAL_ATTENTION_INVOCATION_IDS
            or any(len(values) != len(all_history) for _key, values in site_histories)
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in all_history
                + global_history
                + tuple(
                    item
                    for _key, values in site_histories
                    for item in values
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
            or tuple(
                key
                for key, _value in (
                    self.initial_global_fit_oracle.per_invocation_relative_fisher_loss
                )
            )
            != GLOBAL_ATTENTION_INVOCATION_IDS
            or tuple(
                key
                for key, _value in (
                    self.final_global_fit_oracle.per_invocation_relative_fisher_loss
                )
            )
            != GLOBAL_ATTENTION_INVOCATION_IDS
            or any(
                not math.isclose(
                    values[0],
                    dict(
                        self.initial_global_fit_oracle.per_invocation_relative_fisher_loss
                    )[key],
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    values[-1],
                    dict(
                        self.final_global_fit_oracle.per_invocation_relative_fisher_loss
                    )[key],
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                )
                for key, values in site_histories
            )
        ):
            raise ValueError("global site rank-one bank fit is invalid")

    @property
    def global_attention_invocation_ids(self) -> tuple[str, str, str]:
        return GLOBAL_ATTENTION_INVOCATION_IDS


def _broadcast_shared_rank_one_bank(
    shared: GlobalRankOneFisherAttentionTransportGeneratorBank,
    *,
    gauge_floor: float,
) -> GlobalSiteRankOneFisherAttentionTransportGeneratorBank:
    if not isinstance(
        shared,
        GlobalRankOneFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("shared must be a global rank-one generator bank")
    shared._validate_global_rank_one_unchanged()
    site_count = len(GLOBAL_ATTENTION_INVOCATION_IDS)
    return GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
        left_factors=shared.left_factors,
        singular_scales=shared.singular_scales,
        right_factors=shared.right_factors,
        head_coefficients=shared.head_coefficients,
        head_ids=shared.head_ids,
        hierarchy=shared.hierarchy,
        block_scale_contrasts=shared.block_scale_contrasts,
        site_scale_contrasts=shared.site_scale_contrasts,
        signed_feature_floor=shared.signed_feature_floor,
        global_site_residual_left_vectors=(
            shared.global_residual_left_vectors[None]
            .expand(site_count, -1, -1)
            .clone()
        ),
        global_site_residual_right_vectors=(
            shared.global_residual_right_vectors[None]
            .expand(site_count, -1, -1)
            .clone()
        ),
        shared_rank_one_bank_sha256=shared.artifact_sha256,
        gauge_floor=gauge_floor,
    )


def _global_site_rank_one_bank_with_factors(
    template: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    left: Tensor,
    right: Tensor,
) -> GlobalSiteRankOneFisherAttentionTransportGeneratorBank:
    canonical_left, canonical_right = _canonicalize_site_rank_one_factors(
        left,
        right,
        gauge_floor=template.gauge_floor,
    )
    return GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
        left_factors=template.left_factors,
        singular_scales=template.singular_scales,
        right_factors=template.right_factors,
        head_coefficients=template.head_coefficients,
        head_ids=template.head_ids,
        hierarchy=template.hierarchy,
        block_scale_contrasts=template.block_scale_contrasts,
        site_scale_contrasts=template.site_scale_contrasts,
        signed_feature_floor=template.signed_feature_floor,
        global_site_residual_left_vectors=canonical_left,
        global_site_residual_right_vectors=canonical_right,
        shared_rank_one_bank_sha256=template.shared_rank_one_bank_sha256,
        gauge_floor=template.gauge_floor,
    )


def _site_oracle_evaluations(
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
        for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    }


def fit_global_site_rank_one_attention_transport_generator_bank(
    a_captures: Sequence[AttentionTransportBatch],
    *,
    config: GlobalSiteRankOneAttentionTransportFitConfig,
    shared_rank_one_fit: GlobalRankOneAttentionTransportBankFit | None = None,
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
) -> GlobalSiteRankOneAttentionTransportBankFit:
    """Fit independent global-site residuals using only A captures.

    Each site begins as an exact broadcast of the completed shared rank-one
    bank.  Within a round, exact signed Top-2 coefficients are recomputed
    before optimizing each site.  The fixed-support checkpoint is accepted
    only when a fresh exact sparse oracle strictly improves on that site.
    """

    if not isinstance(config, GlobalSiteRankOneAttentionTransportFitConfig):
        raise TypeError("config must be a global site rank-one fit config")
    if round_observer is not None and not callable(round_observer):
        raise TypeError("round_observer must be callable")
    captures = _capture_values(a_captures)
    global_captures = _global_captures(captures)
    shared_fit = (
        fit_global_rank_one_attention_transport_generator_bank(
            captures,
            config=config.shared_rank_one,
        )
        if shared_rank_one_fit is None
        else shared_rank_one_fit
    )
    expected_head_ids = {
        f"{capture.invocation_id}/query-head.{head_index}"
        for capture in captures
        for head_index in range(capture.query_head_count)
    }
    shared_config = config.shared_rank_one.hierarchical.shared
    if (
        not isinstance(shared_fit, GlobalRankOneAttentionTransportBankFit)
        or shared_fit.bank.generator_count != shared_config.generator_count
        or shared_fit.bank.generator_rank != shared_config.generator_rank
        or set(shared_fit.bank.head_ids) != expected_head_ids
        or shared_fit.bank.gauge_floor != float(config.gauge_floor)
    ):
        raise ValueError("shared rank-one fit does not match the A capture catalog")

    best_bank = _broadcast_shared_rank_one_bank(
        shared_fit.bank,
        gauge_floor=config.gauge_floor,
    )
    active_count = shared_config.active_count
    euclidean_floor = shared_config.euclidean_floor
    coefficient_ridge = shared_config.oracle_coefficient_ridge
    initial = evaluate_sparse_attention_transport_bank(
        best_bank,
        captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    initial_global = evaluate_sparse_attention_transport_bank(
        best_bank,
        global_captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    best_site_evaluations = _site_oracle_evaluations(
        best_bank,
        global_captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    best_evaluation = initial
    best_global_evaluation = initial_global
    history = [float(initial.relative_fisher_loss)]
    global_history = [float(initial_global.relative_fisher_loss)]
    site_histories = {
        invocation_id: [
            float(best_site_evaluations[invocation_id].relative_fisher_loss)
        ]
        for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    }

    for round_index in range(1, config.alternating_rounds + 1):
        for site_index, invocation_id in enumerate(
            GLOBAL_ATTENTION_INVOCATION_IDS
        ):
            site_captures = tuple(
                capture
                for capture in captures
                if capture.invocation_id == invocation_id
            )
            sparse_coefficients = _exact_sparse_coefficients_by_capture(
                best_bank,
                site_captures,
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            site_fit_pairs = tuple(
                (capture, beta)
                for capture, beta in zip(
                    site_captures,
                    sparse_coefficients,
                    strict=True,
                )
            )
            if not site_fit_pairs:
                raise RuntimeError("global site fit has no captures for one site")
            site_row_count = float(
                sum(capture.row_count for capture, _beta in site_fit_pairs)
            )
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

            def fixed_support_objective(
                candidate_left: Tensor,
                candidate_right: Tensor,
            ) -> Tensor:
                right_norms = torch.linalg.vector_norm(
                    candidate_right,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(float(config.gauge_floor))
                unit_right = candidate_right / right_norms
                objective = torch.zeros((), dtype=torch.float64)
                for capture, beta in site_fit_pairs:
                    prediction = _fixed_support_prediction(
                        best_bank,
                        capture,
                        beta,
                        candidate_left,
                        unit_right,
                    )
                    objective = objective + (
                        samplewise_joint_fisher_loss(
                            prediction,
                            capture.valid_operator_outputs,
                            capture.valid_score_gradients,
                            euclidean_floor=euclidean_floor,
                        )
                        * capture.row_count
                        / site_row_count
                    )
                return objective + float(config.residual_ridge) * (
                    candidate_left.square().mean()
                )

            best_fixed_loss = math.inf
            best_fixed_factors: tuple[Tensor, Tensor] | None = None
            for _ in range(config.steps_per_site_per_round):
                optimizer.zero_grad(set_to_none=True)
                objective = fixed_support_objective(left, right)
                if not bool(torch.isfinite(objective)):
                    raise RuntimeError("global site rank-one fit became nonfinite")
                observed = float(objective.detach())
                if observed < best_fixed_loss:
                    best_fixed_loss = observed
                    best_fixed_factors = (
                        left.detach().clone(),
                        right.detach().clone(),
                    )
                objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    (left, right),
                    max_norm=float(config.gradient_clip_norm),
                    error_if_nonfinite=True,
                )
                optimizer.step()

            terminal_objective = fixed_support_objective(left, right)
            if not bool(torch.isfinite(terminal_objective)):
                raise RuntimeError(
                    "global site rank-one terminal fit became nonfinite"
                )
            if float(terminal_objective.detach()) < best_fixed_loss:
                best_fixed_factors = (
                    left.detach().clone(),
                    right.detach().clone(),
                )
            if best_fixed_factors is None:
                raise RuntimeError("global site rank-one fit retained no checkpoint")
            checkpoint_left, checkpoint_right = best_fixed_factors
            checkpoint_unit_right = checkpoint_right / torch.linalg.vector_norm(
                checkpoint_right,
                dim=-1,
                keepdim=True,
            ).clamp_min(float(config.gauge_floor))
            candidate_left = (
                best_bank.global_site_residual_left_vectors.clone()
            )
            candidate_right = (
                best_bank.global_site_residual_right_vectors.clone()
            )
            candidate_left[site_index] = checkpoint_left
            candidate_right[site_index] = checkpoint_unit_right
            candidate = _global_site_rank_one_bank_with_factors(
                best_bank,
                candidate_left,
                candidate_right,
            )
            candidate_site_evaluation = evaluate_sparse_attention_transport_bank(
                candidate,
                tuple(capture for capture, _beta in site_fit_pairs),
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            if (
                candidate_site_evaluation.relative_fisher_loss
                < best_site_evaluations[
                    invocation_id
                ].relative_fisher_loss
                - 1.0e-12
            ):
                best_bank = candidate
                best_site_evaluations[invocation_id] = (
                    candidate_site_evaluation
                )

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
            raise RuntimeError(
                "accepted site update regressed the aggregate exact oracle"
            )
        best_evaluation = evaluation
        best_global_evaluation = global_evaluation
        history.append(float(best_evaluation.relative_fisher_loss))
        global_history.append(float(best_global_evaluation.relative_fisher_loss))
        for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
            site_histories[invocation_id].append(
                float(
                    best_site_evaluations[
                        invocation_id
                    ].relative_fisher_loss
                )
            )
        if round_observer is not None:
            round_observer(
                round_index,
                config.alternating_rounds,
                best_evaluation,
                best_global_evaluation,
            )

    return GlobalSiteRankOneAttentionTransportBankFit(
        bank=best_bank,
        shared_rank_one_fit=shared_fit,
        initial_fit_oracle=initial,
        final_fit_oracle=best_evaluation,
        initial_global_fit_oracle=initial_global,
        final_global_fit_oracle=best_global_evaluation,
        fit_round_relative_fisher_losses=tuple(history),
        fit_round_global_relative_fisher_losses=tuple(global_history),
        fit_round_site_relative_fisher_losses=tuple(
            (invocation_id, tuple(site_histories[invocation_id]))
            for invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
        ),
    )


def evaluate_global_site_rank_one_coupling_diagnostic(
    bank: GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    a_captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int = 2,
    euclidean_floor: float = 1.0e-3,
    coefficient_ridge: float = 0.0,
) -> GlobalRankOneCouplingDiagnostic:
    """Return A-only tied/untied ceilings for the site-specific residual bank.

    ``tied`` means one coefficient multiplies the hierarchical base plus that
    invocation's own residual.  Both untied cases count support by generator
    identity while allowing separate base and residual amplitudes.  Only
    aggregate metrics are returned.
    """

    if not isinstance(
        bank,
        GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    ):
        raise TypeError("bank must be a global site rank-one generator bank")
    bank._validate_global_site_rank_one_unchanged()
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
    joined_targets: list[Tensor] = []
    joined_gradients: list[Tensor] = []
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
                bank.global_site_rank_one_candidate_residuals_from_head_contexts(
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
        fixed_untied_prediction = _solve_grouped_untied_diagnostic_oracle(
            base,
            residual,
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            fixed_generator_support=tied_support,
        )
        reselected_untied_prediction = _solve_grouped_untied_diagnostic_oracle(
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
            "fixed_support_untied": _diagnostic_oracle_evaluation(
                fixed_untied_prediction,
                target,
                gradient,
                active_count=active_count,
                euclidean_floor=float(euclidean_floor),
                coefficient_ridge=float(coefficient_ridge),
                solver_kind=GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
            ),
            "reselected_support_untied": _diagnostic_oracle_evaluation(
                reselected_untied_prediction,
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
                fixed_support_untied=site_evaluations[
                    "fixed_support_untied"
                ],
                reselected_support_untied=site_evaluations[
                    "reselected_support_untied"
                ],
            )
        )
        for key, prediction in (
            ("baseline", baseline_prediction),
            ("tied", tied_prediction),
            ("fixed_support_untied", fixed_untied_prediction),
            ("reselected_support_untied", reselected_untied_prediction),
        ):
            predictions[key].append(prediction)
        joined_targets.append(target)
        joined_gradients.append(gradient)

    target = torch.cat(joined_targets)
    gradient = torch.cat(joined_gradients)
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
        reselected_support_untied=aggregate[
            "reselected_support_untied"
        ],
        leave_one_site_residual_off_relative_fisher_losses=leave_one_out,
        per_site=tuple(per_site),
    )


__all__ = [
    "GLOBAL_SITE_RANK_ONE_ACCEPTANCE_SEMANTICS",
    "GLOBAL_SITE_RANK_ONE_GAUGE_SEMANTICS",
    "GLOBAL_SITE_RANK_ONE_INITIALIZATION_SEMANTICS",
    "GLOBAL_SITE_RANK_ONE_RESIDUAL_SEMANTICS",
    "GlobalSiteRankOneAttentionTransportBankFit",
    "GlobalSiteRankOneAttentionTransportFitConfig",
    "GlobalSiteRankOneFisherAttentionTransportGeneratorBank",
    "evaluate_global_site_rank_one_coupling_diagnostic",
    "fit_global_site_rank_one_attention_transport_generator_bank",
    "global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict",
]
