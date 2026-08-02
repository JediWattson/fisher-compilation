"""Source-authoritative Gemma shadowing for one conditional spectral plan.

This is the candidate-bound sibling of the older graph-organized SVD shadow
runtime.  The measurement carrier and three-pass intervention protocol are
identical, but the numerical executor is a frozen
``ConditionalSpectralGeneratorPlan`` rather than a graph-pack wrapper.

Only the all-on path is supported.  Candidate boundaries and logits remain
metrics-only; the untouched source-model values are always authoritative.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math

import torch
from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    PreparedConditionalSpectralGenerator,
)
from .gemma3_l3_l4_basis_package import Gemma3L3L4BasisPackage
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    ShadowArm,
    _ADAPTER_EXECUTION_BINDING_SCOPES,
    _LOCKED_FACTORIZED_ADAPTER_EXECUTION_SHA256,
    _MAX_DUAL_IDENTITY_ERROR,
    _MAX_R4_CONDITION,
    _basis_copy,
    _canonical_json_bytes,
    _require_sha256,
    _runtime_tensor_sha256,
)
from .graph_organized_svd import GraphOrganizedSVDExecutionAccounting


__all__ = ["Gemma3L3L4ConditionalSpectralShadowRuntime"]


_RUNTIME_BINDING_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-conditional-spectral-shadow-runtime:v1\0"
)


class _PreparedAllOnConditionalExecutor(
    PreparedConditionalSpectralGenerator
):
    """Adapt the generic conditional executor to the locked shadow ABI."""

    @property
    def pack_count(self) -> int:
        return 1

    def graph_execution_accounting(
        self,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        pack_mask: Tensor,
        source_mask: Tensor,
    ) -> GraphOrganizedSVDExecutionAccounting:
        base = super().execution_accounting(
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        expected_shape = (
            base.batch_size,
            base.sequence_length,
            1,
        )
        if (
            not isinstance(pack_mask, Tensor)
            or pack_mask.dtype != torch.bool
            or pack_mask.device != self.device
            or tuple(pack_mask.shape) != expected_shape
            or not torch.equal(pack_mask[..., 0], source_mask)
        ):
            raise ValueError("all-on conditional pack mask differs")

        positions = logical_positions
        sources = source_mask
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if sources.ndim == 1:
            sources = sources.unsqueeze(0)
        knot_set = frozenset(self.fit_knot_origins)
        interpolated_sources = sum(
            int(
                source
                and int(position) not in knot_set
            )
            for source, position in zip(
                sources.reshape(-1).tolist(),
                positions.reshape(-1).tolist(),
                strict=True,
            )
        )
        return GraphOrganizedSVDExecutionAccounting(
            batch_size=base.batch_size,
            sequence_length=base.sequence_length,
            valid_source_rows=base.valid_source_rows,
            valid_target_rows=base.valid_target_rows,
            admitted_causal_pairs=base.admitted_causal_pairs,
            active_pack_instances=base.valid_source_rows,
            active_rank_instances=(
                base.valid_source_rows * base.source_rank
            ),
            interpolated_active_rank_instances=(
                interpolated_sources * base.source_rank
            ),
            admitted_active_rank_pairs=(
                base.admitted_causal_pairs * base.source_rank
            ),
            admitted_active_pack_pairs=base.admitted_causal_pairs,
            source_modes=base.source_modes,
            target_modes=base.target_modes,
            source_rank=base.source_rank,
            pack_count=1,
            lag_count=base.lag_count,
            router_evaluated=False,
        )


class Gemma3L3L4ConditionalSpectralShadowRuntime(
    Gemma3L3L4GraphOrganizedSVDShadowRuntime
):
    """Authenticated three-pass shadow for one frozen all-on plan."""

    @classmethod
    def from_signed_g8_candidate(
        cls,
        candidate: object,
        basis: Gemma3L3L4BasisPackage,
        *,
        expected_basis_payload_sha256: str,
        expected_live_model_sha256: str,
        expected_adapter_execution_sha256: str,
        adapter_execution_binding_scope: str = "locked_factorized_refit",
        analysis_device: torch.device | str = "cpu",
    ) -> "Gemma3L3L4ConditionalSpectralShadowRuntime":
        """Bind the exact audited signed-g8 wrapper, never loose metadata."""

        from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
            Gemma3L3L4GraphWaveletSignedG8Candidate,
        )

        if not isinstance(
            candidate,
            Gemma3L3L4GraphWaveletSignedG8Candidate,
        ):
            raise TypeError("candidate must be the frozen signed-g8 wrapper")
        candidate.validate_frozen_identity()
        return cls(
            candidate.plan,
            basis,
            candidate_artifact_sha256=candidate.artifact_sha256,
            candidate_method=candidate.method,
            candidate_binding=candidate.binding,
            candidate_model=candidate.model,
            expected_plan_artifact_sha256=(
                candidate.plan.artifact_sha256
            ),
            expected_basis_payload_sha256=(
                expected_basis_payload_sha256
            ),
            expected_live_model_sha256=expected_live_model_sha256,
            expected_adapter_execution_sha256=(
                expected_adapter_execution_sha256
            ),
            adapter_execution_binding_scope=(
                adapter_execution_binding_scope
            ),
            analysis_device=analysis_device,
        )

    def __init__(
        self,
        plan: ConditionalSpectralGeneratorPlan,
        basis: Gemma3L3L4BasisPackage,
        *,
        candidate_artifact_sha256: str,
        candidate_method: str,
        candidate_binding: Mapping[str, object],
        candidate_model: Mapping[str, object],
        expected_plan_artifact_sha256: str,
        expected_basis_payload_sha256: str,
        expected_live_model_sha256: str,
        expected_adapter_execution_sha256: str,
        adapter_execution_binding_scope: str = "locked_factorized_refit",
        analysis_device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(plan, ConditionalSpectralGeneratorPlan):
            raise TypeError("plan must be a ConditionalSpectralGeneratorPlan")
        plan.validate_integrity()
        if plan.artifact_sha256 != _require_sha256(
            expected_plan_artifact_sha256,
            label="expected deployment plan",
        ):
            raise ValueError("conditional deployment plan identity differs")
        if not isinstance(candidate_method, str) or not candidate_method:
            raise ValueError("candidate_method must be nonempty text")
        candidate_sha256 = _require_sha256(
            candidate_artifact_sha256,
            label="candidate artifact",
        )
        if not isinstance(candidate_binding, Mapping):
            raise TypeError("candidate_binding must be a mapping")
        if not isinstance(candidate_model, Mapping):
            raise TypeError("candidate_model must be a mapping")

        authenticated_basis = _basis_copy(basis)
        if authenticated_basis.basis_payload_sha256 != _require_sha256(
            expected_basis_payload_sha256,
            label="expected basis payload",
        ):
            raise ValueError("basis package differs from the frozen identity")
        basis_binding = authenticated_basis.binding()
        if any(
            _canonical_json_bytes(candidate_binding.get(name))
            != _canonical_json_bytes(value)
            for name, value in basis_binding.items()
        ):
            raise ValueError("candidate and basis projection lineage differ")
        if (
            candidate_model.get("source_model_sha256")
            != authenticated_basis.source_model_sha256
            or plan.source_modes > authenticated_basis.residual_width
            or plan.target_modes > authenticated_basis.residual_width
        ):
            raise ValueError("candidate model or modal geometry differs")
        expected_scales = (
            authenticated_basis.source_mode_standard_deviations(
                plan.source_modes
            )
        )
        if not torch.equal(plan.source_scales, expected_scales):
            raise ValueError("candidate source scales differ from the basis")

        r4 = authenticated_basis.R4[: plan.target_modes].contiguous()
        singular_values = torch.linalg.svdvals(r4)
        if (
            singular_values.shape != (plan.target_modes,)
            or not bool(torch.isfinite(singular_values).all())
            or float(singular_values[-1]) <= 0.0
        ):
            raise ValueError("R4 target restriction is not full row rank")
        condition = float(singular_values[0] / singular_values[-1])
        if not math.isfinite(condition) or condition > _MAX_R4_CONDITION:
            raise ValueError("R4 target restriction is too ill-conditioned")
        target_decoder = torch.linalg.pinv(
            r4.T,
            atol=0.0,
            rtol=1.0e-12,
        ).contiguous()
        identity_error = float(
            (
                target_decoder @ r4.T
                - torch.eye(plan.target_modes, dtype=torch.float64)
            )
            .abs()
            .max()
        )
        if (
            not math.isfinite(identity_error)
            or identity_error > _MAX_DUAL_IDENTITY_ERROR
        ):
            raise ValueError("R4 target dual failed its right-inverse check")

        runtime_device = torch.device(analysis_device)
        live_model_sha256 = _require_sha256(
            expected_live_model_sha256,
            label="expected live model",
        )
        adapter_execution_sha256 = _require_sha256(
            expected_adapter_execution_sha256,
            label="expected adapter execution",
        )
        if (
            adapter_execution_binding_scope
            not in _ADAPTER_EXECUTION_BINDING_SCOPES
        ):
            raise ValueError(
                "adapter_execution_binding_scope must be "
                "locked_factorized_refit or generic_test"
            )
        if (
            adapter_execution_binding_scope == "locked_factorized_refit"
            and adapter_execution_sha256
            != _LOCKED_FACTORIZED_ADAPTER_EXECUTION_SHA256
        ):
            raise ValueError(
                "locked factorized-refit adapter execution fingerprint differs"
            )

        self._candidate_sha256 = candidate_sha256
        self._candidate_method = candidate_method
        self._basis_sha256 = authenticated_basis.basis_payload_sha256
        self._plan = plan
        self._plan_key = candidate_method
        self._plan_sha256 = plan.artifact_sha256
        self._source_model_sha256 = authenticated_basis.source_model_sha256
        self._live_model_sha256 = live_model_sha256
        self._adapter_execution_sha256 = adapter_execution_sha256
        self._adapter_execution_binding_scope = (
            adapter_execution_binding_scope
        )
        self._device = runtime_device
        self._graph = _PreparedAllOnConditionalExecutor(
            plan,
            device=runtime_device,
            dtype=torch.float64,
        )
        self._x3_mean = authenticated_basis.x3_mean.to(
            runtime_device
        ).contiguous().clone()
        self._r3 = authenticated_basis.R3[: plan.source_modes].to(
            runtime_device
        ).contiguous().clone()
        self._p3 = authenticated_basis.P3[:, : plan.source_modes].to(
            runtime_device
        ).contiguous().clone()
        self._r4 = r4.to(runtime_device).contiguous().clone()
        self._target_decoder = target_decoder.to(
            runtime_device
        ).contiguous().clone()
        self._residual_width = authenticated_basis.residual_width
        self._r4_condition = condition
        self._target_dual_identity_error = identity_error
        self._runtime_binding_sha256 = self._computed_runtime_binding_sha256()
        self._expected_runtime_header = self._runtime_header()
        self._expected_internal_tensor_sha256s = {
            name: _runtime_tensor_sha256(value)
            for name, value in self._internal_tensors().items()
        }
        self.validate_integrity()

    def _runtime_binding_payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.gemma3_l3_l4_conditional_spectral_"
                "shadow_runtime_binding"
            ),
            "format_version": 1,
            "candidate_artifact_sha256": self._candidate_sha256,
            "candidate_method": self._candidate_method,
            "basis_payload_sha256": self._basis_sha256,
            "plan_artifact_sha256": self._plan_sha256,
            "raw_source_model_sha256": self._source_model_sha256,
            "live_factorized_model_sha256": self._live_model_sha256,
            "adapter_execution_sha256": self._adapter_execution_sha256,
            "adapter_execution_binding_scope": (
                self._adapter_execution_binding_scope
            ),
            "analysis_device": str(self._device),
            "residual_width": self._residual_width,
            "source_modes": self._plan.source_modes,
            "source_rank": self._plan.source_rank,
            "target_modes": self._plan.target_modes,
            "fit_knot_origins": self._plan.fit_knot_origins,
            "lag_count": self._plan.lag_count,
            "executor_kind": "conditional_spectral_generator_all_on",
            "routing_supported": False,
            "candidate_serving_authorized": False,
            "native_x4_fallback_policy": (
                "authoritative_native_boundary_outside_target_affected_mask"
            ),
        }

    def _computed_runtime_binding_sha256(self) -> str:
        return hashlib.sha256(
            _RUNTIME_BINDING_DOMAIN
            + _canonical_json_bytes(self._runtime_binding_payload())
        ).hexdigest()

    def _runtime_header(self) -> tuple[object, ...]:
        executor = self._graph
        return (
            self._candidate_sha256,
            self._candidate_method,
            self._basis_sha256,
            self._plan_sha256,
            self._source_model_sha256,
            self._live_model_sha256,
            self._adapter_execution_sha256,
            self._adapter_execution_binding_scope,
            self._runtime_binding_sha256,
            str(self._device),
            self._residual_width,
            self._r4_condition,
            self._target_dual_identity_error,
            executor.plan_sha256,
            executor.fit_knot_origins,
            executor.source_modes,
            executor.source_rank,
            executor.target_modes,
            executor.pack_count,
            executor.lag_count,
            False,
        )

    def _internal_tensors(self) -> dict[str, Tensor]:
        if not isinstance(self._graph, _PreparedAllOnConditionalExecutor):
            raise RuntimeError("prepared conditional runtime type drifted")
        buffers = dict(self._graph.named_buffers(recurse=True))
        if set(buffers) != {
            "source_scales",
            "source_basis",
            "target_basis",
            "knot_cores",
            "knot_positions",
        }:
            raise RuntimeError("prepared conditional buffer set drifted")
        return {
            "basis.x3_mean": self._x3_mean,
            "basis.R3": self._r3,
            "basis.P3": self._p3,
            "basis.R4": self._r4,
            "decoder.target_dual": self._target_decoder,
            **{
                f"executor.{name}": value
                for name, value in buffers.items()
            },
        }

    def validate_integrity(self) -> None:
        try:
            self._plan.validate_integrity()
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "frozen conditional deployment plan drifted after binding"
            ) from error
        if self._plan.artifact_sha256 != self._plan_sha256:
            raise RuntimeError("conditional deployment plan identity drifted")
        if not isinstance(self._graph, _PreparedAllOnConditionalExecutor):
            raise RuntimeError("prepared conditional runtime type drifted")
        if dict(self._graph.named_parameters(recurse=True)):
            raise RuntimeError(
                "prepared conditional runtime unexpectedly acquired parameters"
            )
        if self._runtime_header() != self._expected_runtime_header:
            raise RuntimeError("shadow runtime execution geometry drifted")
        if (
            self._computed_runtime_binding_sha256()
            != self._runtime_binding_sha256
        ):
            raise RuntimeError("shadow runtime binding payload drifted")
        tensors = self._internal_tensors()
        if set(tensors) != set(self._expected_internal_tensor_sha256s):
            raise RuntimeError("shadow runtime internal tensor set drifted")
        for name, value in tensors.items():
            if (
                not isinstance(value, Tensor)
                or value.device != self._device
                or not value.is_contiguous()
                or (
                    value.is_floating_point()
                    and not bool(torch.isfinite(value).all())
                )
                or _runtime_tensor_sha256(value)
                != self._expected_internal_tensor_sha256s[name]
            ):
                raise RuntimeError(
                    f"shadow runtime internal tensor {name} drifted"
                )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._runtime_binding_payload(),
            "runtime_binding_sha256": self._runtime_binding_sha256,
            "source_model_role": "raw_artifact_lineage",
            "live_model_role": "executed_factorized_model_state",
            "adapter_execution_role": (
                "executed_factorized_model_nontensor_semantics"
            ),
            "partial_edge_only": True,
            "all_on_only": True,
            "validate_on_use_integrity": True,
            "authenticated_internal_tensor_count": len(
                self._expected_internal_tensor_sha256s
            ),
            "native_x4_fallback_used_for_metrics_only": True,
            "R4_right_inverse_condition_number": self._r4_condition,
            "R4_right_inverse_identity_max_abs_error": (
                self._target_dual_identity_error
            ),
            "P4_used_as_target_decoder": False,
            "candidate_serving_authorized": False,
            "one_pass_bridge_export_authorized": False,
        }

    def _graph_execute(
        self,
        source_modes: Tensor,
        logical_positions: Tensor,
        valid_target_mask: Tensor,
        source_eligible_mask: Tensor,
        *,
        arm: ShadowArm,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        GraphOrganizedSVDExecutionAccounting | None,
    ]:
        prediction = torch.zeros(
            (*source_eligible_mask.shape, self.target_modes),
            dtype=torch.float64,
            device=self._device,
        )
        pack_mask = torch.zeros(
            (*source_eligible_mask.shape, 1),
            dtype=torch.bool,
            device=self._device,
        )
        scores = torch.zeros_like(pack_mask, dtype=torch.float64)
        if arm == "identity":
            return prediction, pack_mask, scores, None
        if arm != "all_on":
            raise RuntimeError("conditional execution reached an invalid arm")
        active_batches = torch.nonzero(
            source_eligible_mask.any(dim=1),
            as_tuple=False,
        ).flatten()
        if active_batches.numel() == 0:
            return prediction, pack_mask, scores, None
        active_on_analysis = active_batches.to(self._device)
        active_source = source_modes.index_select(0, active_on_analysis)
        active_positions = logical_positions.index_select(
            0,
            active_batches.to(logical_positions.device),
        ).to(self._device)
        active_valid = valid_target_mask.index_select(
            0,
            active_batches.to(valid_target_mask.device),
        ).to(self._device)
        active_sources = source_eligible_mask.index_select(
            0,
            active_batches.to(source_eligible_mask.device),
        ).to(self._device)
        active_prediction = self._graph(
            active_source,
            logical_positions=active_positions,
            valid_mask=active_valid,
            source_mask=active_sources,
        )
        active_pack_mask = active_sources.unsqueeze(-1)
        accounting = self._graph.graph_execution_accounting(
            logical_positions=active_positions,
            valid_mask=active_valid,
            pack_mask=active_pack_mask,
            source_mask=active_sources,
        )
        prediction.index_copy_(0, active_on_analysis, active_prediction)
        pack_mask.index_copy_(0, active_on_analysis, active_pack_mask)
        return prediction, pack_mask, scores, accounting

    def export_one_pass_bridge(self):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "one-pass bridge export is not authorized for this development "
            "candidate"
        )
