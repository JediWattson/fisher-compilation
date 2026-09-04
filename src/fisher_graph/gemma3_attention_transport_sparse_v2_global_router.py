"""Global-support router for sparse Gemma attention transport.

The v1 chart router sparsifies each chart independently and only then mixes
the chart predictions.  Its effective row support can therefore be the union
of several chart supports.  This module preserves chart-conditioned logits
and signed amplitudes while moving sparsification after membership mixing::

    l[n, k] = sum_c m[n, c] l[n, c, k]
    a[n, k] = sum_c m[n, c] a[n, c, k]
    p[n] = TopKRenormalize(softmax(l[n]))
    beta[n, k] = p[n, k] a[n, k]

Consequently every executable coefficient row has at most ``active_count``
nonzero entries, matching the support semantics of the exact sparse Fisher
oracle used as the teacher and held-out comparator.  The generator bank and
chart atlas remain frozen throughout router fitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json

import torch
from torch import Tensor, nn

from .gemma3_attention_transport_compiler import (
    AttentionTransportBatch,
    AttentionTransportCapture,
    AttentionTransportRowBatch,
    AttentionTransportRouterEvaluation,
    ChartConditionedAttentionTransportRouter,
    ChartConditionedRouterConfig,
    ChartConditionedRouterExecution,
    ChartConditionedRouterFit,
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    _capture_catalog,
    _cpu_float64,
    _metrics,
    _tensor_sha256,
    evaluate_chart_conditioned_attention_transport_router,
    samplewise_joint_fisher_loss,
    solve_fisher_sparse_transport_oracle,
)


_GLOBAL_ROUTER_DOMAIN = (
    b"fisher_graph:attention-transport-global-support-router:v2\0"
)
GLOBAL_SUPPORT_ROUTER_SEMANTICS = "membership_mix_then_global_topk"


GlobalSupportRouterExecution = ChartConditionedRouterExecution


@dataclass(frozen=True, slots=True, eq=False)
class GlobalSupportAttentionTransportRouter(
    ChartConditionedAttentionTransportRouter
):
    """Frozen, content-addressed chart router with one global TopK support."""

    invocation_id: str
    generator_bank_sha256: str
    atlas: FrozenAttentionChartAtlas
    selection_weight: Tensor
    selection_bias: Tensor
    amplitude_weight: Tensor
    amplitude_bias: Tensor
    active_count: int
    _parameter_sha256s: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _atlas_sha256: str = field(init=False, repr=False, compare=False)
    _artifact_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ChartConditionedAttentionTransportRouter.__post_init__(self)
        parameters = (
            self.selection_weight,
            self.selection_bias,
            self.amplitude_weight,
            self.amplitude_bias,
        )
        parameter_sha256s = tuple(_tensor_sha256(value) for value in parameters)
        object.__setattr__(self, "_parameter_sha256s", parameter_sha256s)
        object.__setattr__(self, "_atlas_sha256", self.atlas.artifact_sha256)
        payload = self._artifact_payload(parameter_sha256s)
        object.__setattr__(
            self,
            "_artifact_sha256",
            hashlib.sha256(
                _GLOBAL_ROUTER_DOMAIN
                + json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )

    def _artifact_payload(
        self,
        parameter_sha256s: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "active_count": self.active_count,
            "amplitude_bias_sha256": parameter_sha256s[3],
            "amplitude_weight_sha256": parameter_sha256s[2],
            "atlas_sha256": self._atlas_sha256,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "generator_bank_sha256": self.generator_bank_sha256,
            "invocation_id": self.invocation_id,
            "selection_bias_sha256": parameter_sha256s[1],
            "selection_weight_sha256": parameter_sha256s[0],
            "support_semantics": GLOBAL_SUPPORT_ROUTER_SEMANTICS,
        }

    def _validate_unchanged(self) -> None:
        current = tuple(
            _tensor_sha256(value)
            for value in (
                self.selection_weight,
                self.selection_bias,
                self.amplitude_weight,
                self.amplitude_bias,
            )
        )
        if current != self._parameter_sha256s:
            raise RuntimeError("global-support router changed after construction")
        if self.atlas.artifact_sha256 != self._atlas_sha256:
            raise RuntimeError("global-support router atlas changed after construction")

    def __hash__(self) -> int:
        self._validate_unchanged()
        return hash((GlobalSupportAttentionTransportRouter, self._artifact_sha256))

    def __eq__(self, other: object) -> bool:
        self._validate_unchanged()
        if not isinstance(other, GlobalSupportAttentionTransportRouter):
            return False
        other._validate_unchanged()
        return self._artifact_sha256 == other._artifact_sha256

    @property
    def generator_count(self) -> int:
        return int(self.selection_bias.shape[1])

    @property
    def stored_parameter_count(self) -> int:
        return int(
            self.selection_weight.numel()
            + self.selection_bias.numel()
            + self.amplitude_weight.numel()
            + self.amplitude_bias.numel()
        )

    @property
    def artifact_sha256(self) -> str:
        self._validate_unchanged()
        return self._artifact_sha256

    def state_dict(self) -> dict[str, object]:
        self._validate_unchanged()
        return {
            "active_count": self.active_count,
            "amplitude_bias": self.amplitude_bias.clone(),
            "amplitude_weight": self.amplitude_weight.clone(),
            "artifact_sha256": self._artifact_sha256,
            "atlas_sha256": self._atlas_sha256,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "generator_bank_sha256": self.generator_bank_sha256,
            "invocation_id": self.invocation_id,
            "selection_bias": self.selection_bias.clone(),
            "selection_weight": self.selection_weight.clone(),
            "support_semantics": GLOBAL_SUPPORT_ROUTER_SEMANTICS,
        }

    def execute(
        self,
        chart_inputs: Tensor,
        contextual_features: Tensor,
    ) -> ChartConditionedRouterExecution:
        self._validate_unchanged()
        coordinates, memberships = self.atlas.coordinates_and_memberships(chart_inputs)
        context = _cpu_float64(
            contextual_features,
            label="router contextual_features",
            ndim=2,
        )
        if context.shape != (coordinates.shape[0], self.generator_count):
            raise ValueError("router contextual features do not align with rows")
        features = torch.cat(
            (
                coordinates,
                context[:, None, :].expand(-1, self.atlas.chart_count, -1),
            ),
            dim=-1,
        )
        chart_logits = (
            torch.einsum("ncf,ckf->nck", features, self.selection_weight)
            + self.selection_bias[None]
        )
        mixed_logits = torch.einsum("nc,nck->nk", memberships, chart_logits)
        probabilities = torch.softmax(mixed_logits, dim=-1)
        indices = torch.topk(
            probabilities,
            k=self.active_count,
            dim=-1,
            sorted=True,
        ).indices
        active = torch.zeros_like(probabilities, dtype=torch.bool)
        active.scatter_(-1, indices, True)
        sparse = probabilities.masked_fill(~active, 0.0)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        chart_amplitudes = (
            torch.einsum("ncf,ckf->nck", features, self.amplitude_weight)
            + self.amplitude_bias[None]
        )
        mixed_amplitudes = torch.einsum(
            "nc,nck->nk",
            memberships,
            chart_amplitudes,
        )
        coefficients = sparse * mixed_amplitudes
        expanded_probabilities = probabilities[:, None, :].expand(
            -1,
            self.atlas.chart_count,
            -1,
        ).contiguous()
        expanded_sparse = sparse[:, None, :].expand_as(
            expanded_probabilities
        ).contiguous()
        expanded_amplitudes = mixed_amplitudes[:, None, :].expand_as(
            expanded_probabilities
        ).contiguous()
        expanded_active = active[:, None, :].expand_as(
            expanded_probabilities
        ).contiguous()
        return ChartConditionedRouterExecution(
            chart_coordinates=coordinates,
            chart_memberships=memberships,
            contextual_features=context,
            selection_probabilities=expanded_probabilities,
            sparse_selection_probabilities=expanded_sparse,
            signed_amplitudes=expanded_amplitudes,
            active_mask=expanded_active,
            generator_coefficients=coefficients,
        )

    def predict_operator_outputs(
        self,
        bank: FisherAttentionTransportGeneratorBank,
        *,
        head_contexts: Tensor,
        chart_inputs: Tensor,
    ) -> Tensor:
        if not isinstance(bank, FisherAttentionTransportGeneratorBank):
            raise TypeError("bank must be a FisherAttentionTransportGeneratorBank")
        if bank.artifact_sha256 != self.generator_bank_sha256:
            raise ValueError("router is not bound to this generator bank")
        if bank.generator_count != self.generator_count:
            raise ValueError("router and generator inventory do not align")
        context = bank.router_context_features_from_head_contexts(
            invocation_id=self.invocation_id,
            head_contexts=head_contexts,
        )
        execution = self.execute(chart_inputs, context)
        candidates = bank.candidate_outputs_from_head_contexts(
            invocation_id=self.invocation_id,
            head_contexts=head_contexts,
        )
        if candidates.shape[:2] != execution.generator_coefficients.shape:
            raise ValueError("router rows do not align with contextual candidates")
        return torch.einsum(
            "nk,nkh->nh",
            execution.generator_coefficients,
            candidates,
        ).contiguous()


def global_support_attention_transport_router_from_state_dict(
    state: Mapping[str, object],
    *,
    atlas: FrozenAttentionChartAtlas,
    bank: FisherAttentionTransportGeneratorBank,
) -> GlobalSupportAttentionTransportRouter:
    """Reload a router only when both frozen parent artifacts match."""

    expected = {
        "active_count",
        "amplitude_bias",
        "amplitude_weight",
        "artifact_sha256",
        "atlas_sha256",
        "contains_oracle_outputs_or_coefficients",
        "contains_teacher_outputs_or_gradients",
        "generator_bank_sha256",
        "invocation_id",
        "selection_bias",
        "selection_weight",
        "support_semantics",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("global-support router state fields are invalid")
    if (
        not isinstance(atlas, FrozenAttentionChartAtlas)
        or not isinstance(bank, FisherAttentionTransportGeneratorBank)
    ):
        raise TypeError("router parents must be a frozen atlas and generator bank")
    if (
        state["atlas_sha256"] != atlas.artifact_sha256
        or state["generator_bank_sha256"] != bank.artifact_sha256
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
        or state["support_semantics"] != GLOBAL_SUPPORT_ROUTER_SEMANTICS
    ):
        raise ValueError("global-support router state lineage is invalid")
    if atlas.hidden_width != bank.hidden_width:
        raise ValueError("global-support router parent dimensions are invalid")
    router = GlobalSupportAttentionTransportRouter(
        invocation_id=state["invocation_id"],
        generator_bank_sha256=state["generator_bank_sha256"],
        atlas=atlas,
        selection_weight=state["selection_weight"],
        selection_bias=state["selection_bias"],
        amplitude_weight=state["amplitude_weight"],
        amplitude_bias=state["amplitude_bias"],
        active_count=state["active_count"],
    )
    if router.generator_count != bank.generator_count:
        raise ValueError("global-support router generator inventory is invalid")
    if state["artifact_sha256"] != router.artifact_sha256:
        raise ValueError("global-support router artifact hash drifted")
    return router


GlobalSupportRouterFit = ChartConditionedRouterFit
GlobalSupportRouterEvaluation = AttentionTransportRouterEvaluation


def _capture_values(
    capture: AttentionTransportBatch | Sequence[AttentionTransportBatch],
) -> tuple[AttentionTransportBatch, ...]:
    if isinstance(capture, (AttentionTransportCapture, AttentionTransportRowBatch)):
        return (capture,)
    if isinstance(capture, Sequence) and not isinstance(capture, (str, bytes)):
        values = tuple(capture)
        if values and all(
            isinstance(value, (AttentionTransportCapture, AttentionTransportRowBatch))
            for value in values
        ):
            return values
    raise TypeError("capture must contain attention transport captures")


def _features_and_memberships(
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    *,
    invocation_id: str,
    head_contexts: Tensor,
    chart_inputs: Tensor,
) -> tuple[Tensor, Tensor]:
    coordinates, memberships = atlas.coordinates_and_memberships(chart_inputs)
    contextual = bank.router_context_features_from_head_contexts(
        invocation_id=invocation_id,
        head_contexts=head_contexts,
    )
    features = torch.cat(
        (
            coordinates,
            contextual[:, None, :].expand(-1, atlas.chart_count, -1),
        ),
        dim=-1,
    )
    return features, memberships.detach()


def _training_execution(
    features: Tensor,
    memberships: Tensor,
    selection_weight: Tensor,
    selection_bias: Tensor,
    amplitude_weight: Tensor,
    amplitude_bias: Tensor,
    *,
    active_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    chart_logits = (
        torch.einsum("ncf,ckf->nck", features, selection_weight)
        + selection_bias[None]
    )
    mixed_logits = torch.einsum("nc,nck->nk", memberships, chart_logits)
    probabilities = torch.softmax(mixed_logits, dim=-1)
    indices = torch.topk(probabilities, k=active_count, dim=-1).indices
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    hard = probabilities.masked_fill(~mask, 0.0)
    hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    straight_through = probabilities + (hard - probabilities).detach()
    chart_amplitudes = (
        torch.einsum("ncf,ckf->nck", features, amplitude_weight)
        + amplitude_bias[None]
    )
    mixed_amplitudes = torch.einsum(
        "nc,nck->nk",
        memberships,
        chart_amplitudes,
    )
    return (
        straight_through * mixed_amplitudes,
        probabilities,
        hard,
        mixed_amplitudes,
    )


def fit_global_support_attention_transport_router(
    bank: FisherAttentionTransportGeneratorBank,
    capture: AttentionTransportBatch | Sequence[AttentionTransportBatch],
    atlas: FrozenAttentionChartAtlas,
    *,
    config: ChartConditionedRouterConfig,
) -> GlobalSupportRouterFit:
    """Fit one invocation router against the exact sparse Fisher teacher."""

    if not isinstance(bank, FisherAttentionTransportGeneratorBank):
        raise TypeError("bank must be a FisherAttentionTransportGeneratorBank")
    values, invocations, _, _ = _capture_catalog(_capture_values(capture))
    if len(invocations) != 1:
        raise ValueError("one router fit may cover only one invocation")
    representative = invocations[0]
    if not isinstance(atlas, FrozenAttentionChartAtlas):
        raise TypeError("atlas must be a FrozenAttentionChartAtlas")
    if not isinstance(config, ChartConditionedRouterConfig):
        raise TypeError("config must be a ChartConditionedRouterConfig")
    if (
        atlas.hidden_width != bank.hidden_width
        or config.active_count > bank.generator_count
    ):
        raise ValueError("atlas, router, and bank dimensions are inconsistent")

    candidates = torch.cat(tuple(bank.candidate_outputs(value) for value in values))
    targets = torch.cat(tuple(value.valid_operator_outputs for value in values))
    gradients = torch.cat(tuple(value.valid_score_gradients for value in values))
    chart_inputs = torch.cat(tuple(value.valid_chart_inputs for value in values))
    head_contexts = torch.cat(tuple(value.head_contexts for value in values))
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        targets,
        gradients,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.oracle_ridge,
    )
    oracle_prediction = torch.einsum("nk,nkh->nh", oracle.coefficients, candidates)
    features, memberships = _features_and_memberships(
        bank,
        atlas,
        invocation_id=representative.invocation_id,
        head_contexts=head_contexts,
        chart_inputs=chart_inputs,
    )
    charts = atlas.chart_count
    generators = bank.generator_count
    feature_count = atlas.chart_rank + generators
    target_mass = oracle.coefficients.abs()
    row_mass = target_mass.sum(dim=-1, keepdim=True)
    target_probabilities = target_mass / row_mass.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    zero_rows = row_mass.squeeze(-1) == 0.0
    if bool(zero_rows.any()):
        target_probabilities[zero_rows] = 1.0 / generators
    target_usage = target_probabilities.mean(dim=0)

    selection_weight = nn.Parameter(
        torch.zeros(charts, generators, feature_count, dtype=torch.float64)
    )
    selection_bias = nn.Parameter(
        target_usage.clamp_min(1.0e-6).log()[None].expand(charts, -1).clone()
    )
    amplitude_weight = nn.Parameter(
        torch.zeros(charts, generators, feature_count, dtype=torch.float64)
    )
    initial_indices = torch.topk(
        target_usage,
        k=config.active_count,
    ).indices
    initial_active = torch.zeros_like(target_usage, dtype=torch.bool)
    initial_active[initial_indices] = True
    initial_sparse = target_usage.masked_fill(~initial_active, 0.0)
    initial_sparse = initial_sparse / initial_sparse.sum().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    mean_beta = oracle.coefficients.mean(dim=0)
    initial_amplitude = torch.where(
        initial_active,
        mean_beta / initial_sparse.clamp_min(1.0e-6),
        torch.zeros_like(mean_beta),
    )
    amplitude_bias = nn.Parameter(
        initial_amplitude[None].expand(charts, -1).clone()
    )
    parameters = (
        selection_weight,
        selection_bias,
        amplitude_weight,
        amplitude_bias,
    )
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    tiny = torch.finfo(torch.float64).tiny
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        beta, probabilities, _, _ = _training_execution(
            features,
            memberships,
            selection_weight,
            selection_bias,
            amplitude_weight,
            amplitude_bias,
            active_count=config.active_count,
        )
        prediction = torch.einsum("nk,nkh->nh", beta, candidates)
        output_loss = samplewise_joint_fisher_loss(
            prediction,
            targets,
            gradients,
            euclidean_floor=config.euclidean_floor,
        )
        log_probabilities = probabilities.clamp_min(tiny).log()
        target_log = target_probabilities.clamp_min(tiny).log()
        route_kl = (
            target_probabilities * (target_log - log_probabilities)
        ).sum(dim=-1).mean()
        entropy = (-(probabilities * log_probabilities).sum(dim=-1)).mean()
        predicted_usage = probabilities.mean(dim=0)
        balance = (predicted_usage - target_usage).square().mean()
        coefficient_mse = (beta - oracle.coefficients).square().mean()
        loss = (
            output_loss
            + float(config.route_kl_coefficient) * route_kl
            + float(config.entropy_coefficient) * entropy
            + float(config.balance_coefficient) * balance
            + float(config.coefficient_mse_coefficient) * coefficient_mse
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("global-support router fit became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float(config.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()

    router = GlobalSupportAttentionTransportRouter(
        invocation_id=representative.invocation_id,
        generator_bank_sha256=bank.artifact_sha256,
        atlas=atlas,
        selection_weight=selection_weight.detach(),
        selection_bias=selection_bias.detach(),
        amplitude_weight=amplitude_weight.detach(),
        amplitude_bias=amplitude_bias.detach(),
        active_count=config.active_count,
    )
    router_prediction = router.predict_operator_outputs(
        bank,
        head_contexts=head_contexts,
        chart_inputs=chart_inputs,
    )
    return GlobalSupportRouterFit(
        router=router,
        oracle=oracle,
        oracle_output_metrics=_metrics(
            oracle_prediction,
            targets,
            gradients,
            config.euclidean_floor,
        ),
        router_output_metrics=_metrics(
            router_prediction,
            targets,
            gradients,
            config.euclidean_floor,
        ),
    )


def evaluate_global_support_attention_transport_router(
    bank: FisherAttentionTransportGeneratorBank,
    router: GlobalSupportAttentionTransportRouter,
    captures: Sequence[AttentionTransportBatch],
    *,
    euclidean_floor: float,
) -> GlobalSupportRouterEvaluation:
    """Evaluate against an exact ridge-free oracle with the same support cap."""

    if not isinstance(router, GlobalSupportAttentionTransportRouter):
        raise TypeError("router must be a GlobalSupportAttentionTransportRouter")
    return evaluate_chart_conditioned_attention_transport_router(
        bank,
        router,
        captures,
        euclidean_floor=euclidean_floor,
    )


__all__ = [
    "GLOBAL_SUPPORT_ROUTER_SEMANTICS",
    "GlobalSupportAttentionTransportRouter",
    "GlobalSupportRouterEvaluation",
    "GlobalSupportRouterExecution",
    "GlobalSupportRouterFit",
    "evaluate_global_support_attention_transport_router",
    "fit_global_support_attention_transport_router",
    "global_support_attention_transport_router_from_state_dict",
]
