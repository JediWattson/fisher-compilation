"""Global-attention-only rank-one residuals for attention transport.

This additive rung keeps the crossed hierarchical bank and its signed ``2K``
router context unchanged.  It gives each generator exactly one additional
operator direction,

    R[k] = l[k] r[k]^T,

which is applied to the head-coefficient-weighted contextual input ``xi[k]``
only at Gemma's three global-attention invocations.  The right vector has unit
norm and a deterministic sign; all rank-one scale lives in the left vector.
Consequently the residual inventory contains exactly ``2 * K * H`` stored
parameters and a zero left factor reproduces the hierarchical bank exactly.

The production fitter alternates exact signed Top-2 coefficients with fixed-
support optimization of only these A-side residual factors.  A candidate is
retained only when both the exact all-A and global-only sparse Fisher oracles
improve.  A separate A-only diagnostic can split the hierarchical base from
the shared residual and compare tied against generator-grouped untied Top-2
amplitudes without opening or accepting on a held-out panel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import math

import torch
from torch import Tensor, nn

from .gemma3_attention_transport_compiler import (
    AttentionTransportBatch,
    AttentionTransportMetrics,
    _cpu_float64,
    _metrics,
    _tensor_sha256,
    samplewise_joint_fisher_loss,
    solve_fisher_sparse_transport_oracle,
)
from .gemma3_attention_transport_hierarchical_edge import (
    HierarchicalFisherAttentionTransportGeneratorBank,
    HierarchicalSparseAttentionTransportBankFit,
    HierarchicalSparseAttentionTransportFitConfig,
    _capture_values,
    _exact_sparse_coefficients_by_capture,
    fit_hierarchical_sparse_attention_transport_generator_bank,
    hierarchical_fisher_attention_transport_generator_bank_from_state_dict,
)
from .gemma3_attention_transport_sparse_v2 import (
    SparseAttentionTransportOracleEvaluation,
    evaluate_sparse_attention_transport_bank,
)


GLOBAL_ATTENTION_INVOCATION_IDS = (
    "mutation.11",
    "mutation.23",
    "mutation.35",
)
GLOBAL_RANK_ONE_RESIDUAL_SEMANTICS = (
    "global_attention_only_rank_one_residual_v1"
)
GLOBAL_RANK_ONE_GAUGE_SEMANTICS = (
    "unit_right_max_abs_positive_scale_in_left_v1"
)
GLOBAL_RANK_ONE_ROUTER_SEMANTICS = "inherit_hierarchical_signed_2k_unchanged"
GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION = (
    "sum_difference_equal_tied_embedding_v1"
)
GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER = (
    "exact_generator_support_topk_hierarchical_base_v1"
)
GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER = (
    "exact_generator_support_topk_tied_base_plus_residual_v1"
)
GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER = (
    "exact_untied_amplitudes_on_tied_generator_support_v1"
)
GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER = (
    "exact_grouped_generator_support_topk_untied_amplitudes_v1"
)

_BANK_DOMAIN = b"fisher_graph:attention-transport-global-rank-one-bank:v1\0"


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


def is_global_attention_invocation(invocation_id: str) -> bool:
    """Return whether ``invocation_id`` is one of Gemma's global sites."""

    if not isinstance(invocation_id, str):
        raise TypeError("invocation_id must be a string")
    return invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS


def _canonicalize_rank_one_factors(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> tuple[Tensor, Tensor]:
    """Put rank-one factors in the unique unit-right, positive-pivot gauge."""

    left_value = _cpu_float64(left, label="global residual left vectors", ndim=2)
    right_value = _cpu_float64(
        right,
        label="global residual right vectors",
        ndim=2,
    )
    if left_value.shape != right_value.shape:
        raise ValueError("global residual factor shapes do not match")
    norms = torch.linalg.vector_norm(right_value, dim=-1)
    degenerate = norms <= float(gauge_floor)
    if bool(degenerate.any()):
        # A zero right vector denotes the zero matrix regardless of the left
        # vector.  Give that zero product one deterministic stored gauge.
        right_value = right_value.clone()
        left_value = left_value.clone()
        right_value[degenerate] = 0.0
        right_value[degenerate, 0] = 1.0
        left_value[degenerate] = 0.0
        norms = torch.linalg.vector_norm(right_value, dim=-1)
    canonical_right = right_value / norms[:, None]
    canonical_left = left_value * norms[:, None]
    pivot_indices = canonical_right.abs().argmax(dim=-1)
    pivot_values = canonical_right.gather(1, pivot_indices[:, None]).squeeze(1)
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    canonical_left = canonical_left * signs[:, None]
    canonical_right = canonical_right * signs[:, None]
    return canonical_left.contiguous(), canonical_right.contiguous()


def _is_canonical_rank_one_gauge(
    left: Tensor,
    right: Tensor,
    *,
    gauge_floor: float,
) -> bool:
    try:
        canonical_left, canonical_right = _canonicalize_rank_one_factors(
            left,
            right,
            gauge_floor=gauge_floor,
        )
        left_value = _cpu_float64(left, label="global residual left vectors", ndim=2)
        right_value = _cpu_float64(
            right,
            label="global residual right vectors",
            ndim=2,
        )
    except (TypeError, ValueError):
        return False
    return bool(
        torch.allclose(
            left_value,
            canonical_left,
            atol=2.0e-15,
            rtol=2.0e-15,
        )
        and torch.allclose(
            right_value,
            canonical_right,
            atol=2.0e-15,
            rtol=2.0e-15,
        )
    )


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class GlobalRankOneFisherAttentionTransportGeneratorBank(
    HierarchicalFisherAttentionTransportGeneratorBank
):
    """Hierarchical bank plus one rank-one residual per generator.

    Residuals act on the already head-mixed contextual input ``xi[k]``.  They
    are deliberately absent from ``router_context_features_from_head_contexts``;
    this rung changes bank expressivity while holding the signed router fixed.
    """

    global_residual_left_vectors: Tensor
    global_residual_right_vectors: Tensor
    gauge_floor: float = 1.0e-12
    _global_rank_one_tensor_sha256s: tuple[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _hierarchical_bank_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _global_rank_one_artifact_sha256: str = field(
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
            raise ValueError("global rank-one gauge floor must be finite and positive")
        left, right = _canonicalize_rank_one_factors(
            self.global_residual_left_vectors,
            self.global_residual_right_vectors,
            gauge_floor=float(self.gauge_floor),
        )
        expected = (self.generator_count, self.hidden_width)
        if left.shape != expected:
            raise ValueError("global residual factors have the wrong bank geometry")
        if not set(GLOBAL_ATTENTION_INVOCATION_IDS).issubset(
            self.hierarchy.invocation_ids
        ):
            raise ValueError("hierarchy omits a required global-attention site")
        object.__setattr__(self, "global_residual_left_vectors", left)
        object.__setattr__(self, "global_residual_right_vectors", right)
        object.__setattr__(self, "gauge_floor", float(self.gauge_floor))
        tensor_hashes = (_tensor_sha256(left), _tensor_sha256(right))
        hierarchical_hash = (
            HierarchicalFisherAttentionTransportGeneratorBank.artifact_sha256.fget(
                self
            )
        )
        object.__setattr__(self, "_global_rank_one_tensor_sha256s", tensor_hashes)
        object.__setattr__(self, "_hierarchical_bank_sha256", hierarchical_hash)
        payload = self._global_rank_one_artifact_payload(tensor_hashes)
        object.__setattr__(
            self,
            "_global_rank_one_artifact_sha256",
            hashlib.sha256(_BANK_DOMAIN + _canonical_json_bytes(payload)).hexdigest(),
        )

    def _global_rank_one_artifact_payload(
        self,
        tensor_hashes: tuple[str, str],
    ) -> dict[str, object]:
        return {
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": GLOBAL_RANK_ONE_GAUGE_SEMANTICS,
            "generator_count": self.generator_count,
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "global_residual_left_vectors_sha256": tensor_hashes[0],
            "global_residual_right_vectors_sha256": tensor_hashes[1],
            "hidden_width": self.hidden_width,
            "hierarchical_bank_sha256": self._hierarchical_bank_sha256,
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "residual_parameter_count": self.global_rank_one_parameter_count,
            "residual_semantics": GLOBAL_RANK_ONE_RESIDUAL_SEMANTICS,
            "router_semantics": GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
        }

    def _validate_global_rank_one_unchanged(self) -> None:
        HierarchicalFisherAttentionTransportGeneratorBank._validate_unchanged(self)
        current = (
            _tensor_sha256(self.global_residual_left_vectors),
            _tensor_sha256(self.global_residual_right_vectors),
        )
        if current != self._global_rank_one_tensor_sha256s:
            raise RuntimeError("global rank-one generator bank changed after construction")
        if not _is_canonical_rank_one_gauge(
            self.global_residual_left_vectors,
            self.global_residual_right_vectors,
            gauge_floor=self.gauge_floor,
        ):
            raise RuntimeError("global rank-one generator gauge drifted")

    @property
    def hierarchical_bank_sha256(self) -> str:
        self._validate_global_rank_one_unchanged()
        return self._hierarchical_bank_sha256

    @property
    def global_rank_one_parameter_count(self) -> int:
        return int(
            self.global_residual_left_vectors.numel()
            + self.global_residual_right_vectors.numel()
        )

    @property
    def stored_parameter_count(self) -> int:
        hierarchical = (
            HierarchicalFisherAttentionTransportGeneratorBank.stored_parameter_count.fget(
                self
            )
        )
        return int(hierarchical + self.global_rank_one_parameter_count)

    @property
    def artifact_sha256(self) -> str:
        self._validate_global_rank_one_unchanged()
        return self._global_rank_one_artifact_sha256

    def __hash__(self) -> int:
        return hash(
            (
                GlobalRankOneFisherAttentionTransportGeneratorBank,
                self.artifact_sha256,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(
                other,
                GlobalRankOneFisherAttentionTransportGeneratorBank,
            )
            and self.artifact_sha256 == other.artifact_sha256
        )

    def global_rank_one_candidate_residuals_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        """Return ``[N,K,H]`` residual candidates, exactly zero off-global."""

        self._validate_global_rank_one_unchanged()
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        if not is_global_attention_invocation(invocation_id):
            return torch.zeros_like(contextual)
        activation = torch.einsum(
            "nkh,kh->nk",
            contextual,
            self.global_residual_right_vectors,
        )
        return (
            activation[:, :, None]
            * self.global_residual_left_vectors[None]
        ).contiguous()

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
        if not is_global_attention_invocation(invocation_id):
            return base
        return (
            base
            + self.global_rank_one_candidate_residuals_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=head_contexts,
            )
        ).contiguous()

    def state_dict(self) -> dict[str, object]:
        self._validate_global_rank_one_unchanged()
        return {
            "artifact_sha256": self._global_rank_one_artifact_sha256,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "gauge_floor": float(self.gauge_floor),
            "gauge_semantics": GLOBAL_RANK_ONE_GAUGE_SEMANTICS,
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "global_residual_left_vectors": (
                self.global_residual_left_vectors.clone()
            ),
            "global_residual_right_vectors": (
                self.global_residual_right_vectors.clone()
            ),
            "hierarchical_bank": (
                HierarchicalFisherAttentionTransportGeneratorBank.state_dict(self)
            ),
            "hierarchical_bank_sha256": self._hierarchical_bank_sha256,
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "residual_semantics": GLOBAL_RANK_ONE_RESIDUAL_SEMANTICS,
            "router_semantics": GLOBAL_RANK_ONE_ROUTER_SEMANTICS,
        }


def global_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> GlobalRankOneFisherAttentionTransportGeneratorBank:
    """Reload a source-free global rank-one bank and verify all lineage."""

    expected = {
        "artifact_sha256",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "gauge_floor",
        "gauge_semantics",
        "global_attention_invocation_ids",
        "global_residual_left_vectors",
        "global_residual_right_vectors",
        "hierarchical_bank",
        "hierarchical_bank_sha256",
        "native_output_rmsnorm_retained",
        "native_qk_retained",
        "residual_semantics",
        "router_semantics",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("global rank-one bank state fields are invalid")
    raw_hierarchical = state["hierarchical_bank"]
    raw_ids = state["global_attention_invocation_ids"]
    if (
        not isinstance(raw_hierarchical, Mapping)
        or raw_ids != list(GLOBAL_ATTENTION_INVOCATION_IDS)
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_raw_teacher_outputs_or_gradients"] is not False
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
        or not _is_sha256(state["artifact_sha256"])
        or not _is_sha256(state["hierarchical_bank_sha256"])
        or state["gauge_semantics"] != GLOBAL_RANK_ONE_GAUGE_SEMANTICS
        or state["residual_semantics"] != GLOBAL_RANK_ONE_RESIDUAL_SEMANTICS
        or state["router_semantics"] != GLOBAL_RANK_ONE_ROUTER_SEMANTICS
    ):
        raise ValueError("global rank-one bank state safety semantics are invalid")
    gauge_floor = state["gauge_floor"]
    if (
        isinstance(gauge_floor, bool)
        or not isinstance(gauge_floor, (int, float))
        or not math.isfinite(float(gauge_floor))
        or float(gauge_floor) <= 0.0
        or not _is_canonical_rank_one_gauge(
            state["global_residual_left_vectors"],
            state["global_residual_right_vectors"],
            gauge_floor=float(gauge_floor),
        )
    ):
        raise ValueError("global rank-one bank state gauge is invalid")
    hierarchical = (
        hierarchical_fisher_attention_transport_generator_bank_from_state_dict(
            raw_hierarchical
        )
    )
    bank = GlobalRankOneFisherAttentionTransportGeneratorBank(
        left_factors=hierarchical.left_factors,
        singular_scales=hierarchical.singular_scales,
        right_factors=hierarchical.right_factors,
        head_coefficients=hierarchical.head_coefficients,
        head_ids=hierarchical.head_ids,
        hierarchy=hierarchical.hierarchy,
        block_scale_contrasts=hierarchical.block_scale_contrasts,
        site_scale_contrasts=hierarchical.site_scale_contrasts,
        signed_feature_floor=hierarchical.signed_feature_floor,
        global_residual_left_vectors=state["global_residual_left_vectors"],
        global_residual_right_vectors=state["global_residual_right_vectors"],
        gauge_floor=float(gauge_floor),
    )
    if (
        state["hierarchical_bank_sha256"] != bank.hierarchical_bank_sha256
        or state["artifact_sha256"] != bank.artifact_sha256
    ):
        raise ValueError("global rank-one bank artifact lineage drifted")
    return bank


@dataclass(frozen=True, slots=True)
class GlobalRankOneAttentionTransportFitConfig:
    """Fixed A-only protocol for the global-attention rank-one rung."""

    hierarchical: HierarchicalSparseAttentionTransportFitConfig
    alternating_rounds: int = 4
    steps_per_round: int = 48
    learning_rate: float = 1.0e-2
    residual_ridge: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    gauge_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(
            self.hierarchical,
            HierarchicalSparseAttentionTransportFitConfig,
        ):
            raise TypeError("hierarchical must be a hierarchical fit config")
        if type(self.alternating_rounds) is not int or self.alternating_rounds <= 0:
            raise ValueError("alternating_rounds must be positive")
        if type(self.steps_per_round) is not int or self.steps_per_round <= 0:
            raise ValueError("steps_per_round must be positive")
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

    @classmethod
    def fixed_k8_r8_top2(cls) -> GlobalRankOneAttentionTransportFitConfig:
        return cls(
            hierarchical=(
                HierarchicalSparseAttentionTransportFitConfig.fixed_k8_r8_top2()
            )
        )


@dataclass(frozen=True, slots=True)
class GlobalRankOneAttentionTransportBankFit:
    bank: GlobalRankOneFisherAttentionTransportGeneratorBank
    hierarchical_fit: HierarchicalSparseAttentionTransportBankFit
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    initial_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_global_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_round_relative_fisher_losses: tuple[float, ...]
    fit_round_global_relative_fisher_losses: tuple[float, ...]

    def __post_init__(self) -> None:
        all_history = self.fit_round_relative_fisher_losses
        global_history = self.fit_round_global_relative_fisher_losses
        if (
            not isinstance(
                self.bank,
                GlobalRankOneFisherAttentionTransportGeneratorBank,
            )
            or not isinstance(
                self.hierarchical_fit,
                HierarchicalSparseAttentionTransportBankFit,
            )
            or self.bank.hierarchical_bank_sha256
            != self.hierarchical_fit.bank.artifact_sha256
            or not all(
                isinstance(value, SparseAttentionTransportOracleEvaluation)
                for value in (
                    self.initial_fit_oracle,
                    self.final_fit_oracle,
                    self.initial_global_fit_oracle,
                    self.final_global_fit_oracle,
                )
            )
            or len(all_history) < 2
            or len(global_history) != len(all_history)
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in all_history + global_history
            )
            or any(
                right > left + 1.0e-12
                for left, right in zip(all_history, all_history[1:])
            )
            or any(
                right > left + 1.0e-12
                for left, right in zip(global_history, global_history[1:])
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
            raise ValueError("global rank-one bank fit is invalid")

    @property
    def global_attention_invocation_ids(self) -> tuple[str, str, str]:
        return GLOBAL_ATTENTION_INVOCATION_IDS


_GLOBAL_RANK_ONE_DIAGNOSTIC_SOLVERS = frozenset(
    {
        GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER,
        GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER,
        GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
        GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
    }
)


@dataclass(frozen=True, slots=True)
class GlobalRankOneGroupedOracleEvaluation:
    """Coefficient-free metrics for one generator-group sparse oracle."""

    output_metrics: AttentionTransportMetrics
    zero_predictor_fisher_loss: float
    relative_fisher_loss: float
    active_generator_count_limit: int
    coefficient_ridge: float
    solver_kind: str

    def __post_init__(self) -> None:
        metrics = self.output_metrics
        if (
            not isinstance(metrics, AttentionTransportMetrics)
            or type(metrics.row_count) is not int
            or metrics.row_count <= 0
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in (
                    metrics.fisher_loss,
                    metrics.directional_mse,
                    metrics.euclidean_mse,
                    metrics.normalized_rmse,
                )
            )
            or not math.isfinite(float(metrics.cosine))
            or abs(float(metrics.cosine)) > 1.0 + 1.0e-12
            or not math.isfinite(float(self.zero_predictor_fisher_loss))
            or float(self.zero_predictor_fisher_loss) <= 0.0
            or not math.isfinite(float(self.relative_fisher_loss))
            or float(self.relative_fisher_loss) < 0.0
            or type(self.active_generator_count_limit) is not int
            or self.active_generator_count_limit <= 0
            or isinstance(self.coefficient_ridge, bool)
            or not isinstance(self.coefficient_ridge, (int, float))
            or not math.isfinite(float(self.coefficient_ridge))
            or float(self.coefficient_ridge) < 0.0
            or self.solver_kind not in _GLOBAL_RANK_ONE_DIAGNOSTIC_SOLVERS
            or not math.isclose(
                float(self.relative_fisher_loss),
                float(metrics.fisher_loss)
                / float(self.zero_predictor_fisher_loss),
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("global rank-one grouped oracle evaluation is invalid")
        object.__setattr__(
            self,
            "zero_predictor_fisher_loss",
            float(self.zero_predictor_fisher_loss),
        )
        object.__setattr__(self, "relative_fisher_loss", float(self.relative_fisher_loss))
        object.__setattr__(self, "coefficient_ridge", float(self.coefficient_ridge))

    @property
    def row_count(self) -> int:
        return self.output_metrics.row_count

    @property
    def fisher_loss(self) -> float:
        return self.output_metrics.fisher_loss


@dataclass(frozen=True, slots=True)
class GlobalRankOneSiteCouplingDiagnostic:
    """A-only base/residual coupling ceilings for one global invocation."""

    invocation_id: str
    baseline: GlobalRankOneGroupedOracleEvaluation
    tied: GlobalRankOneGroupedOracleEvaluation
    fixed_support_untied: GlobalRankOneGroupedOracleEvaluation
    reselected_support_untied: GlobalRankOneGroupedOracleEvaluation

    def __post_init__(self) -> None:
        evaluations = (
            self.baseline,
            self.tied,
            self.fixed_support_untied,
            self.reselected_support_untied,
        )
        if (
            self.invocation_id not in GLOBAL_ATTENTION_INVOCATION_IDS
            or any(
                not isinstance(value, GlobalRankOneGroupedOracleEvaluation)
                for value in evaluations
            )
            or tuple(value.solver_kind for value in evaluations)
            != (
                GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER,
                GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER,
                GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
                GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
            )
            or len({value.row_count for value in evaluations}) != 1
            or len(
                {value.active_generator_count_limit for value in evaluations}
            )
            != 1
            or len({value.coefficient_ridge for value in evaluations}) != 1
            or any(
                not math.isclose(
                    value.zero_predictor_fisher_loss,
                    evaluations[0].zero_predictor_fisher_loss,
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                )
                for value in evaluations[1:]
            )
        ):
            raise ValueError("global rank-one per-site coupling diagnostic is invalid")
        if evaluations[0].coefficient_ridge == 0.0:
            tolerance = 1.0e-11 * max(
                1.0,
                evaluations[1].fisher_loss,
                evaluations[2].fisher_loss,
            )
            if (
                evaluations[2].fisher_loss
                > evaluations[1].fisher_loss + tolerance
                or evaluations[3].fisher_loss
                > evaluations[2].fisher_loss + tolerance
            ):
                raise ValueError(
                    "ridge-free untied global rank-one oracle regressed"
                )


@dataclass(frozen=True, slots=True)
class GlobalRankOneCouplingDiagnostic:
    """Aggregate A-only evidence for shared-site and tied-amplitude coupling."""

    global_attention_invocation_ids: tuple[str, str, str]
    active_generator_count_limit: int
    coefficient_ridge: float
    ridge_parameterization: str
    baseline: GlobalRankOneGroupedOracleEvaluation
    tied: GlobalRankOneGroupedOracleEvaluation
    fixed_support_untied: GlobalRankOneGroupedOracleEvaluation
    reselected_support_untied: GlobalRankOneGroupedOracleEvaluation
    leave_one_site_residual_off_relative_fisher_losses: tuple[
        tuple[str, float], ...
    ]
    per_site: tuple[GlobalRankOneSiteCouplingDiagnostic, ...]

    def __post_init__(self) -> None:
        evaluations = (
            self.baseline,
            self.tied,
            self.fixed_support_untied,
            self.reselected_support_untied,
        )
        expected_sites = GLOBAL_ATTENTION_INVOCATION_IDS
        if (
            self.global_attention_invocation_ids != expected_sites
            or type(self.active_generator_count_limit) is not int
            or self.active_generator_count_limit <= 0
            or isinstance(self.coefficient_ridge, bool)
            or not isinstance(self.coefficient_ridge, (int, float))
            or not math.isfinite(float(self.coefficient_ridge))
            or float(self.coefficient_ridge) < 0.0
            or self.ridge_parameterization
            != GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION
            or any(
                not isinstance(value, GlobalRankOneGroupedOracleEvaluation)
                for value in evaluations
            )
            or tuple(value.solver_kind for value in evaluations)
            != (
                GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER,
                GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER,
                GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
                GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
            )
            or any(
                value.active_generator_count_limit
                != self.active_generator_count_limit
                or value.coefficient_ridge != float(self.coefficient_ridge)
                for value in evaluations
            )
            or any(
                not isinstance(value, GlobalRankOneSiteCouplingDiagnostic)
                for value in self.per_site
            )
            or tuple(value.invocation_id for value in self.per_site)
            != expected_sites
            or any(
                value.baseline.active_generator_count_limit
                != self.active_generator_count_limit
                or value.baseline.coefficient_ridge
                != float(self.coefficient_ridge)
                for value in self.per_site
            )
            or tuple(
                key
                for key, _value in (
                    self.leave_one_site_residual_off_relative_fisher_losses
                )
            )
            != expected_sites
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for _key, value in (
                    self.leave_one_site_residual_off_relative_fisher_losses
                )
            )
        ):
            raise ValueError("global rank-one coupling diagnostic is invalid")

        total_rows = sum(value.baseline.row_count for value in self.per_site)
        if any(value.row_count != total_rows for value in evaluations):
            raise ValueError("global rank-one diagnostic row counts do not close")
        global_zero = sum(
            value.baseline.row_count
            * value.baseline.zero_predictor_fisher_loss
            for value in self.per_site
        ) / total_rows
        if any(
            not math.isclose(
                value.zero_predictor_fisher_loss,
                global_zero,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            for value in evaluations
        ):
            raise ValueError("global rank-one diagnostic zero loss does not close")
        for field_name, aggregate in zip(
            (
                "baseline",
                "tied",
                "fixed_support_untied",
                "reselected_support_untied",
            ),
            evaluations,
            strict=True,
        ):
            for metric_name in (
                "fisher_loss",
                "directional_mse",
                "euclidean_mse",
            ):
                expected_metric = sum(
                    getattr(value, field_name).row_count
                    * getattr(
                        getattr(value, field_name).output_metrics,
                        metric_name,
                    )
                    for value in self.per_site
                ) / total_rows
                observed_metric = getattr(
                    aggregate.output_metrics,
                    metric_name,
                )
                if not math.isclose(
                    observed_metric,
                    expected_metric,
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "global rank-one diagnostic invocation metrics do not close"
                    )
        tied_losses = {
            value.invocation_id: value.tied.fisher_loss for value in self.per_site
        }
        baseline_losses = {
            value.invocation_id: value.baseline.fisher_loss
            for value in self.per_site
        }
        row_counts = {
            value.invocation_id: value.baseline.row_count for value in self.per_site
        }
        for invocation_id, observed in (
            self.leave_one_site_residual_off_relative_fisher_losses
        ):
            disabled_loss = sum(
                row_counts[key]
                * (
                    baseline_losses[key]
                    if key == invocation_id
                    else tied_losses[key]
                )
                for key in expected_sites
            ) / total_rows
            expected_relative = disabled_loss / global_zero
            if not math.isclose(
                float(observed),
                expected_relative,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "global rank-one leave-one-site loss does not close"
                )
        object.__setattr__(self, "coefficient_ridge", float(self.coefficient_ridge))


def _global_captures(
    captures: Sequence[AttentionTransportBatch],
) -> tuple[AttentionTransportBatch, ...]:
    values = tuple(
        capture
        for capture in captures
        if capture.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    )
    observed = {capture.invocation_id for capture in values}
    if observed != set(GLOBAL_ATTENTION_INVOCATION_IDS):
        raise ValueError("A captures omit a global-attention invocation")
    return values


def _diagnostic_oracle_evaluation(
    prediction: Tensor,
    target: Tensor,
    gradient: Tensor,
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
    solver_kind: str,
) -> GlobalRankOneGroupedOracleEvaluation:
    metrics = _metrics(prediction, target, gradient, euclidean_floor)
    zero = _metrics(
        torch.zeros_like(target),
        target,
        gradient,
        euclidean_floor,
    )
    return GlobalRankOneGroupedOracleEvaluation(
        output_metrics=metrics,
        zero_predictor_fisher_loss=float(zero.fisher_loss),
        relative_fisher_loss=float(metrics.fisher_loss / zero.fisher_loss),
        active_generator_count_limit=active_count,
        coefficient_ridge=coefficient_ridge,
        solver_kind=solver_kind,
    )


def _solve_tied_diagnostic_oracle(
    candidates: Tensor,
    target: Tensor,
    gradient: Tensor,
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> tuple[Tensor, Tensor]:
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        target,
        gradient,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    prediction = torch.einsum(
        "nk,nkh->nh",
        oracle.coefficients,
        candidates,
    ).contiguous()
    return prediction, oracle.support


def _untied_sum_difference_atoms(base: Tensor, residual: Tensor) -> Tensor:
    if base.shape != residual.shape or base.ndim != 3:
        raise ValueError("global rank-one diagnostic candidates do not align")
    return torch.stack((base + residual, base - residual), dim=2).reshape(
        base.shape[0],
        2 * base.shape[1],
        base.shape[2],
    ).contiguous()


def _solve_grouped_untied_diagnostic_oracle(
    base: Tensor,
    residual: Tensor,
    target: Tensor,
    gradient: Tensor,
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
    fixed_generator_support: Tensor | None,
) -> Tensor:
    """Solve two amplitudes while counting support by generator identity.

    The two coordinates for each selected generator are ``z = base + residual``
    and ``d = base - residual``.  Thus ``gamma * z + delta * d`` has raw base
    and residual amplitudes ``gamma + delta`` and ``gamma - delta``.  The ridge
    ``lambda * (gamma**2 + delta**2)`` contains the tied solution at
    ``delta = 0`` with exactly the same penalty as the tied oracle.
    """

    atoms = _untied_sum_difference_atoms(base, residual)
    targets = _cpu_float64(target, label="diagnostic target", ndim=2)
    gradients = _cpu_float64(gradient, label="diagnostic gradient", ndim=2)
    rows, atom_count, hidden = atoms.shape
    generator_count = atom_count // 2
    if (
        targets.shape != (rows, hidden)
        or gradients.shape != targets.shape
        or type(active_count) is not int
        or not 0 < active_count <= generator_count
    ):
        raise ValueError("global rank-one grouped oracle geometry is invalid")

    support: Tensor | None = None
    if fixed_generator_support is not None:
        if (
            not isinstance(fixed_generator_support, Tensor)
            or fixed_generator_support.dtype != torch.bool
            or fixed_generator_support.shape != (rows, generator_count)
        ):
            raise ValueError("fixed generator support does not align")
        support = fixed_generator_support.detach().to(device="cpu").clone()
        if bool((support.sum(dim=-1) > active_count).any()):
            raise ValueError("fixed generator support exceeds Top-K")

    projected_gradient = torch.einsum("nph,nh->np", atoms, gradients)
    gram = (
        float(euclidean_floor)
        * torch.einsum("nph,nqh->npq", atoms, atoms)
        + projected_gradient[:, :, None] * projected_gradient[:, None, :]
    )
    if coefficient_ridge > 0.0:
        gram = gram + float(coefficient_ridge) * torch.eye(
            atom_count,
            dtype=torch.float64,
        )[None]
    right = (
        float(euclidean_floor) * torch.einsum("nph,nh->np", atoms, targets)
        + projected_gradient
        * torch.einsum("nh,nh->n", gradients, targets)[:, None]
    )
    target_energy = (
        float(euclidean_floor) * targets.square().sum(dim=-1)
        + torch.einsum("nh,nh->n", gradients, targets).square()
    )

    def solve_support(
        selected_generators: tuple[int, ...],
        row_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        atom_indices = torch.tensor(
            tuple(
                coordinate
                for generator_index in selected_generators
                for coordinate in (2 * generator_index, 2 * generator_index + 1)
            ),
            dtype=torch.long,
        )
        selected_gram = gram if row_indices is None else gram.index_select(0, row_indices)
        selected_right = (
            right if row_indices is None else right.index_select(0, row_indices)
        )
        block = selected_gram.index_select(1, atom_indices).index_select(
            2,
            atom_indices,
        )
        block_right = selected_right.index_select(1, atom_indices)
        values = (
            torch.linalg.solve(block, block_right.unsqueeze(-1)).squeeze(-1)
            if coefficient_ridge > 0.0
            else torch.einsum(
                "nij,nj->ni",
                torch.linalg.pinv(block, hermitian=True),
                block_right,
            )
        )
        return atom_indices, values

    coefficients = torch.zeros(rows, atom_count, dtype=torch.float64)
    if support is not None:
        covered = torch.zeros(rows, dtype=torch.bool)
        for support_size in range(0, active_count + 1):
            for selected in combinations(range(generator_count), support_size):
                pattern = torch.zeros(generator_count, dtype=torch.bool)
                if selected:
                    pattern[list(selected)] = True
                row_indices = torch.nonzero(
                    (support == pattern[None]).all(dim=-1),
                    as_tuple=False,
                ).squeeze(-1)
                if row_indices.numel() == 0:
                    continue
                covered[row_indices] = True
                if not selected:
                    continue
                atom_indices, values = solve_support(selected, row_indices)
                rows_value = coefficients.index_select(0, row_indices)
                rows_value[:, atom_indices] = values
                coefficients.index_copy_(0, row_indices, rows_value)
        if not bool(covered.all()):
            raise RuntimeError("fixed grouped oracle omitted one support pattern")
    else:
        objectives = target_energy.clone()
        machine_epsilon = torch.finfo(torch.float64).eps
        for support_size in range(1, active_count + 1):
            for selected in combinations(range(generator_count), support_size):
                atom_indices, values = solve_support(selected)
                candidate_objective = (
                    target_energy
                    - (
                        right.index_select(1, atom_indices) * values
                    ).sum(dim=-1)
                ).clamp_min(0.0)
                tolerance = (
                    256.0
                    * machine_epsilon
                    * torch.maximum(torch.ones_like(objectives), objectives.abs())
                )
                improved = candidate_objective < objectives - tolerance
                if not bool(improved.any()):
                    continue
                candidate_coefficients = torch.zeros_like(coefficients)
                candidate_coefficients[:, atom_indices] = values
                coefficients = torch.where(
                    improved[:, None],
                    candidate_coefficients,
                    coefficients,
                )
                objectives = torch.where(
                    improved,
                    candidate_objective,
                    objectives,
                )
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("global rank-one grouped oracle became nonfinite")
    return torch.einsum("np,nph->nh", coefficients, atoms).contiguous()


def evaluate_global_rank_one_coupling_diagnostic(
    bank: GlobalRankOneFisherAttentionTransportGeneratorBank,
    a_captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int = 2,
    euclidean_floor: float = 1.0e-3,
    coefficient_ridge: float = 0.0,
) -> GlobalRankOneCouplingDiagnostic:
    """Measure shared-site and tied-amplitude bottlenecks on A only.

    ``active_count`` always counts generator identities.  The independently
    reselected untied oracle may use both its base and residual amplitude for a
    selected generator, but it may never select more generator identities than
    the tied oracle.  Returned objects retain metrics only, never oracle
    supports, coefficients, targets, or gradients.
    """

    if not isinstance(bank, GlobalRankOneFisherAttentionTransportGeneratorBank):
        raise TypeError("bank must be a global rank-one generator bank")
    bank._validate_global_rank_one_unchanged()
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
            capture for capture in captures if capture.invocation_id == invocation_id
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
                bank.global_rank_one_candidate_residuals_from_head_contexts(
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
        predictions["baseline"].append(baseline_prediction)
        predictions["tied"].append(tied_prediction)
        predictions["fixed_support_untied"].append(fixed_untied_prediction)
        predictions["reselected_support_untied"].append(
            reselected_untied_prediction
        )
        joined_targets.append(target)
        joined_gradients.append(gradient)

    target = torch.cat(joined_targets)
    gradient = torch.cat(joined_gradients)
    aggregate = {
        "baseline": _diagnostic_oracle_evaluation(
            torch.cat(predictions["baseline"]),
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            solver_kind=GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER,
        ),
        "tied": _diagnostic_oracle_evaluation(
            torch.cat(predictions["tied"]),
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            solver_kind=GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER,
        ),
        "fixed_support_untied": _diagnostic_oracle_evaluation(
            torch.cat(predictions["fixed_support_untied"]),
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            solver_kind=GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER,
        ),
        "reselected_support_untied": _diagnostic_oracle_evaluation(
            torch.cat(predictions["reselected_support_untied"]),
            target,
            gradient,
            active_count=active_count,
            euclidean_floor=float(euclidean_floor),
            coefficient_ridge=float(coefficient_ridge),
            solver_kind=GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER,
        ),
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


def _deterministic_initial_right_vectors(
    bank: HierarchicalFisherAttentionTransportGeneratorBank,
    *,
    gauge_floor: float,
) -> Tensor:
    """Choose a deterministic unit start outside each current input span.

    Orthogonality is initialization only.  Subsequent fitting may move into
    the current span, which preserves useful off-diagonal combinations.
    """

    result = []
    for factors in bank.right_factors:
        leverage = factors.square().sum(dim=-1)
        index = int(torch.argmin(leverage))
        direction = torch.zeros(bank.hidden_width, dtype=torch.float64)
        direction[index] = 1.0
        direction = direction - factors @ factors[index]
        norm = float(torch.linalg.vector_norm(direction))
        if norm <= float(gauge_floor):
            direction = factors[:, 0]
        direction = direction / torch.linalg.vector_norm(direction)
        pivot = int(torch.argmax(direction.abs()))
        if float(direction[pivot]) < 0.0:
            direction = -direction
        result.append(direction)
    return torch.stack(result).contiguous()


def _zero_global_rank_one_bank(
    hierarchical: HierarchicalFisherAttentionTransportGeneratorBank,
    *,
    gauge_floor: float,
) -> GlobalRankOneFisherAttentionTransportGeneratorBank:
    right = _deterministic_initial_right_vectors(
        hierarchical,
        gauge_floor=gauge_floor,
    )
    return GlobalRankOneFisherAttentionTransportGeneratorBank(
        left_factors=hierarchical.left_factors,
        singular_scales=hierarchical.singular_scales,
        right_factors=hierarchical.right_factors,
        head_coefficients=hierarchical.head_coefficients,
        head_ids=hierarchical.head_ids,
        hierarchy=hierarchical.hierarchy,
        block_scale_contrasts=hierarchical.block_scale_contrasts,
        site_scale_contrasts=hierarchical.site_scale_contrasts,
        signed_feature_floor=hierarchical.signed_feature_floor,
        global_residual_left_vectors=torch.zeros_like(right),
        global_residual_right_vectors=right,
        gauge_floor=gauge_floor,
    )


def _global_rank_one_bank_with_factors(
    template: GlobalRankOneFisherAttentionTransportGeneratorBank,
    left: Tensor,
    right: Tensor,
) -> GlobalRankOneFisherAttentionTransportGeneratorBank:
    canonical_left, canonical_right = _canonicalize_rank_one_factors(
        left,
        right,
        gauge_floor=template.gauge_floor,
    )
    return GlobalRankOneFisherAttentionTransportGeneratorBank(
        left_factors=template.left_factors,
        singular_scales=template.singular_scales,
        right_factors=template.right_factors,
        head_coefficients=template.head_coefficients,
        head_ids=template.head_ids,
        hierarchy=template.hierarchy,
        block_scale_contrasts=template.block_scale_contrasts,
        site_scale_contrasts=template.site_scale_contrasts,
        signed_feature_floor=template.signed_feature_floor,
        global_residual_left_vectors=canonical_left,
        global_residual_right_vectors=canonical_right,
        gauge_floor=template.gauge_floor,
    )


def _fixed_support_prediction(
    bank: GlobalRankOneFisherAttentionTransportGeneratorBank,
    capture: AttentionTransportBatch,
    sparse_coefficients: Tensor,
    left: Tensor,
    unit_right: Tensor,
) -> Tensor:
    base_candidates = (
        HierarchicalFisherAttentionTransportGeneratorBank.candidate_outputs_from_head_contexts(
            bank,
            invocation_id=capture.invocation_id,
            head_contexts=capture.head_contexts,
        )
    )
    if capture.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
        contextual = bank.contextual_inputs_from_head_contexts(
            invocation_id=capture.invocation_id,
            head_contexts=capture.head_contexts,
        )
        activation = torch.einsum("nkh,kh->nk", contextual, unit_right)
        candidates = base_candidates + activation[:, :, None] * left[None]
    else:
        candidates = base_candidates
    return torch.einsum(
        "nk,nkh->nh",
        sparse_coefficients,
        candidates,
    ).contiguous()


def fit_global_rank_one_attention_transport_generator_bank(
    a_captures: Sequence[AttentionTransportBatch],
    *,
    config: GlobalRankOneAttentionTransportFitConfig,
    hierarchical_fit: HierarchicalSparseAttentionTransportBankFit | None = None,
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
) -> GlobalRankOneAttentionTransportBankFit:
    """Fit global-only rank-one residuals on A with exact Top-2 alternation."""

    if not isinstance(config, GlobalRankOneAttentionTransportFitConfig):
        raise TypeError("config must be a global rank-one fit config")
    if round_observer is not None and not callable(round_observer):
        raise TypeError("round_observer must be callable")
    captures = _capture_values(a_captures)
    global_captures = _global_captures(captures)
    hierarchy_fit = (
        fit_hierarchical_sparse_attention_transport_generator_bank(
            captures,
            config=config.hierarchical,
        )
        if hierarchical_fit is None
        else hierarchical_fit
    )
    if (
        not isinstance(
            hierarchy_fit,
            HierarchicalSparseAttentionTransportBankFit,
        )
        or hierarchy_fit.bank.generator_count
        != config.hierarchical.shared.generator_count
        or hierarchy_fit.bank.generator_rank
        != config.hierarchical.shared.generator_rank
        or {head_id for head_id in hierarchy_fit.bank.head_ids}
        != {
            f"{capture.invocation_id}/query-head.{head_index}"
            for capture in captures
            for head_index in range(capture.query_head_count)
        }
    ):
        raise ValueError("hierarchical fit does not match the A capture catalog")

    current = _zero_global_rank_one_bank(
        hierarchy_fit.bank,
        gauge_floor=config.gauge_floor,
    )
    active_count = config.hierarchical.shared.active_count
    euclidean_floor = config.hierarchical.shared.euclidean_floor
    coefficient_ridge = config.hierarchical.shared.oracle_coefficient_ridge
    initial = evaluate_sparse_attention_transport_bank(
        current,
        captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    initial_global = evaluate_sparse_attention_transport_bank(
        current,
        global_captures,
        active_count=active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=coefficient_ridge,
    )
    best_bank = current
    best_evaluation = initial
    best_global_evaluation = initial_global
    history = [float(initial.relative_fisher_loss)]
    global_history = [float(initial_global.relative_fisher_loss)]
    global_row_count = float(
        sum(capture.row_count for capture in global_captures)
    )

    for round_index in range(1, config.alternating_rounds + 1):
        sparse_coefficients = _exact_sparse_coefficients_by_capture(
            best_bank,
            captures,
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        global_fit_pairs = tuple(
            (capture, beta)
            for capture, beta in zip(
                captures,
                sparse_coefficients,
                strict=True,
            )
            if capture.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
        )
        left = nn.Parameter(best_bank.global_residual_left_vectors.clone())
        right = nn.Parameter(best_bank.global_residual_right_vectors.clone())
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
            for capture, beta in global_fit_pairs:
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
                    / global_row_count
                )
            objective = objective + float(config.residual_ridge) * (
                candidate_left.square().mean()
            )
            return objective

        best_fixed_loss = math.inf
        best_fixed_factors: tuple[Tensor, Tensor] | None = None
        for _ in range(config.steps_per_round):
            optimizer.zero_grad(set_to_none=True)
            objective = fixed_support_objective(left, right)
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("global rank-one fit became nonfinite")
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
            raise RuntimeError("global rank-one terminal fit became nonfinite")
        if float(terminal_objective.detach()) < best_fixed_loss:
            best_fixed_factors = (
                left.detach().clone(),
                right.detach().clone(),
            )
        if best_fixed_factors is None:
            raise RuntimeError("global rank-one fit retained no checkpoint")
        checkpoint_left, checkpoint_right = best_fixed_factors
        checkpoint_unit_right = checkpoint_right / torch.linalg.vector_norm(
            checkpoint_right,
            dim=-1,
            keepdim=True,
        ).clamp_min(float(config.gauge_floor))
        candidate = _global_rank_one_bank_with_factors(
            best_bank,
            checkpoint_left,
            checkpoint_unit_right,
        )
        evaluation = evaluate_sparse_attention_transport_bank(
            candidate,
            captures,
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        if (
            evaluation.relative_fisher_loss
            < best_evaluation.relative_fisher_loss - 1.0e-12
        ):
            candidate_global_evaluation = evaluate_sparse_attention_transport_bank(
                candidate,
                global_captures,
                active_count=active_count,
                euclidean_floor=euclidean_floor,
                coefficient_ridge=coefficient_ridge,
            )
            # Only global predictions can change, so an all-A improvement and
            # a global regression would signal numerical inconsistency.
            if (
                candidate_global_evaluation.relative_fisher_loss
                < best_global_evaluation.relative_fisher_loss - 1.0e-12
            ):
                best_bank = candidate
                best_evaluation = evaluation
                best_global_evaluation = candidate_global_evaluation
        history.append(float(best_evaluation.relative_fisher_loss))
        global_history.append(float(best_global_evaluation.relative_fisher_loss))
        if round_observer is not None:
            round_observer(
                round_index,
                config.alternating_rounds,
                best_evaluation,
                best_global_evaluation,
            )

    return GlobalRankOneAttentionTransportBankFit(
        bank=best_bank,
        hierarchical_fit=hierarchy_fit,
        initial_fit_oracle=initial,
        final_fit_oracle=best_evaluation,
        initial_global_fit_oracle=initial_global,
        final_global_fit_oracle=best_global_evaluation,
        fit_round_relative_fisher_losses=tuple(history),
        fit_round_global_relative_fisher_losses=tuple(global_history),
    )


__all__ = [
    "GLOBAL_ATTENTION_INVOCATION_IDS",
    "GLOBAL_RANK_ONE_BASELINE_ORACLE_SOLVER",
    "GLOBAL_RANK_ONE_FIXED_UNTIED_ORACLE_SOLVER",
    "GLOBAL_RANK_ONE_GAUGE_SEMANTICS",
    "GLOBAL_RANK_ONE_RESELECTED_UNTIED_ORACLE_SOLVER",
    "GLOBAL_RANK_ONE_RESIDUAL_SEMANTICS",
    "GLOBAL_RANK_ONE_ROUTER_SEMANTICS",
    "GLOBAL_RANK_ONE_TIED_ORACLE_SOLVER",
    "GLOBAL_RANK_ONE_UNTIED_RIDGE_PARAMETERIZATION",
    "GlobalRankOneAttentionTransportBankFit",
    "GlobalRankOneAttentionTransportFitConfig",
    "GlobalRankOneCouplingDiagnostic",
    "GlobalRankOneFisherAttentionTransportGeneratorBank",
    "GlobalRankOneGroupedOracleEvaluation",
    "GlobalRankOneSiteCouplingDiagnostic",
    "evaluate_global_rank_one_coupling_diagnostic",
    "fit_global_rank_one_attention_transport_generator_bank",
    "global_rank_one_fisher_attention_transport_generator_bank_from_state_dict",
    "is_global_attention_invocation",
]
