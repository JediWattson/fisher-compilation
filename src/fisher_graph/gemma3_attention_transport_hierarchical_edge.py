"""Crossed hierarchical sparse attention transport with signed router edges.

This additive v1 keeps the sparse-v2 shared generator directions frozen and
adds only signed diagonal scale contrasts.  Gemma's eighteen attention sites
form a crossed 3-by-6 layout: three six-layer macroblocks and six repeated
sites within each macroblock.  Fixed zero-sum Helmert contrasts make

    G[i, k] = G_shared[k] + G_block[block(i), k] + G_site[site(i), k]

identifiable without storing one unconstrained residual per invocation.

The router retains chart-conditioned logits and amplitudes, but mixes both
over chart membership before applying one global TopK.  Its source-free
context contains two features per generator: log1p contextual RMS and the
signed cosine between the generator latent and its effective signed diagonal.
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
    AttentionTransportBankFit,
    AttentionTransportBatch,
    AttentionTransportCapture,
    AttentionTransportMetrics,
    AttentionTransportRowBatch,
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
    samplewise_joint_fisher_loss,
    solve_fisher_sparse_transport_oracle,
)
from .gemma3_attention_transport_sparse_v2 import (
    SparseAttentionTransportOracleEvaluation,
    SparseAwareAttentionTransportBankFit,
    SparseAwareAttentionTransportFitConfig,
    evaluate_sparse_attention_transport_bank,
    fit_sparse_aware_attention_transport_generator_bank,
)


_LAYOUT_DOMAIN = b"fisher_graph:attention-transport-hierarchy-layout:v1\0"
_BANK_DOMAIN = b"fisher_graph:attention-transport-hierarchical-bank:v1\0"
_ROUTER_DOMAIN = b"fisher_graph:attention-transport-hierarchical-router:v1\0"
HIERARCHY_SEMANTICS = "crossed_3x6_zero_sum_signed_scale_contrasts"
SIGNED_ROUTER_FEATURE_SEMANTICS = "log1p_context_rms_and_scale_latent_cosine"
GLOBAL_SUPPORT_SEMANTICS = "membership_mix_then_single_global_topk"


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


def _mutation_ordinal(invocation_id: str) -> int:
    prefix = "mutation."
    if not isinstance(invocation_id, str) or not invocation_id.startswith(prefix):
        raise ValueError("hierarchy invocation id must use mutation.<odd> form")
    suffix = invocation_id[len(prefix) :]
    if not suffix.isascii() or not suffix.isdecimal():
        raise ValueError("hierarchy invocation suffix must be decimal")
    mutation = int(suffix)
    if mutation <= 0 or mutation % 2 != 1:
        raise ValueError("hierarchy mutation suffix must be positive and odd")
    return (mutation - 1) // 2


def _helmert_zero_sum(count: int) -> Tensor:
    """Return deterministic orthonormal contrasts perpendicular to all-ones."""

    if type(count) is not int or count < 2:
        raise ValueError("Helmert contrast count must be at least two")
    result = torch.zeros(count, count - 1, dtype=torch.float64)
    for column in range(count - 1):
        scale = math.sqrt(float((column + 1) * (column + 2)))
        result[: column + 1, column] = 1.0 / scale
        result[column + 1, column] = -float(column + 1) / scale
    return result


@dataclass(frozen=True, slots=True)
class AttentionTransportHierarchyLayout:
    """Numeric Gemma invocation mapping for a crossed macroblock/site layout."""

    invocation_ids: tuple[str, ...]
    block_count: int = 3
    site_count: int = 6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.invocation_ids, tuple)
            or len(self.invocation_ids) != self.block_count * self.site_count
            or type(self.block_count) is not int
            or self.block_count != 3
            or type(self.site_count) is not int
            or self.site_count != 6
            or len(set(self.invocation_ids)) != len(self.invocation_ids)
        ):
            raise ValueError("hierarchy layout must contain one crossed 3x6 catalog")
        ordinals = tuple(_mutation_ordinal(value) for value in self.invocation_ids)
        expected = tuple(range(self.block_count * self.site_count))
        if ordinals != expected:
            raise ValueError(
                "hierarchy invocation ids must be numeric mutation.1 through "
                "mutation.35 in odd order"
            )

    @classmethod
    def gemma3_270m_attention(
        cls,
        invocation_ids: Sequence[str] | None = None,
    ) -> AttentionTransportHierarchyLayout:
        expected = tuple(f"mutation.{2 * index + 1}" for index in range(18))
        if invocation_ids is None:
            values = expected
        elif isinstance(invocation_ids, Sequence) and not isinstance(
            invocation_ids, (str, bytes)
        ):
            values = tuple(sorted(invocation_ids, key=_mutation_ordinal))
        else:
            raise TypeError("invocation_ids must be a materialized sequence")
        if values != expected:
            raise ValueError("Gemma hierarchy invocation catalog drifted")
        return cls(invocation_ids=values)

    @property
    def block_contrast(self) -> Tensor:
        return _helmert_zero_sum(self.block_count)

    @property
    def site_contrast(self) -> Tensor:
        return _helmert_zero_sum(self.site_count)

    def coordinates(self, invocation_id: str) -> tuple[int, int]:
        ordinal = _mutation_ordinal(invocation_id)
        if ordinal >= len(self.invocation_ids) or self.invocation_ids[ordinal] != invocation_id:
            raise ValueError("invocation is absent from hierarchy layout")
        return ordinal // self.site_count, ordinal % self.site_count

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "block_count": self.block_count,
            "contrast_kind": "helmert_zero_sum_v1",
            "hierarchy_semantics": HIERARCHY_SEMANTICS,
            "invocation_ids": list(self.invocation_ids),
            "site_count": self.site_count,
        }
        return hashlib.sha256(_LAYOUT_DOMAIN + _canonical_json_bytes(payload)).hexdigest()

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "block_count": self.block_count,
            "contrast_kind": "helmert_zero_sum_v1",
            "hierarchy_semantics": HIERARCHY_SEMANTICS,
            "invocation_ids": list(self.invocation_ids),
            "site_count": self.site_count,
        }


def attention_transport_hierarchy_layout_from_state_dict(
    state: Mapping[str, object],
) -> AttentionTransportHierarchyLayout:
    expected = {
        "artifact_sha256",
        "block_count",
        "contrast_kind",
        "hierarchy_semantics",
        "invocation_ids",
        "site_count",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("hierarchy layout state fields are invalid")
    raw_ids = state["invocation_ids"]
    if (
        not isinstance(raw_ids, list)
        or state["contrast_kind"] != "helmert_zero_sum_v1"
        or state["hierarchy_semantics"] != HIERARCHY_SEMANTICS
    ):
        raise ValueError("hierarchy layout state semantics are invalid")
    layout = AttentionTransportHierarchyLayout(
        invocation_ids=tuple(raw_ids),
        block_count=state["block_count"],
        site_count=state["site_count"],
    )
    if state["artifact_sha256"] != layout.artifact_sha256:
        raise ValueError("hierarchy layout artifact hash drifted")
    return layout


@dataclass(frozen=True, slots=True, eq=False)
class HierarchicalFisherAttentionTransportGeneratorBank(
    FisherAttentionTransportGeneratorBank
):
    """Shared U/V bank with crossed signed diagonal scale contrasts."""

    hierarchy: AttentionTransportHierarchyLayout
    block_scale_contrasts: Tensor
    site_scale_contrasts: Tensor
    signed_feature_floor: float = 1.0e-12
    _hierarchical_tensor_sha256s: tuple[str, ...] = field(
        init=False, repr=False, compare=False
    )
    _shared_bank_sha256: str = field(init=False, repr=False, compare=False)
    _hierarchy_sha256: str = field(init=False, repr=False, compare=False)
    _artifact_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        FisherAttentionTransportGeneratorBank.__post_init__(self)
        if not isinstance(self.hierarchy, AttentionTransportHierarchyLayout):
            raise TypeError("hierarchy must be an AttentionTransportHierarchyLayout")
        block = _cpu_float64(
            self.block_scale_contrasts,
            label="block_scale_contrasts",
            ndim=3,
        )
        site = _cpu_float64(
            self.site_scale_contrasts,
            label="site_scale_contrasts",
            ndim=3,
        )
        expected_tail = (self.generator_count, self.generator_rank)
        if (
            block.shape != (self.hierarchy.block_count - 1,) + expected_tail
            or site.shape != (self.hierarchy.site_count - 1,) + expected_tail
            or isinstance(self.signed_feature_floor, bool)
            or not isinstance(self.signed_feature_floor, (int, float))
            or not math.isfinite(float(self.signed_feature_floor))
            or float(self.signed_feature_floor) <= 0.0
        ):
            raise ValueError("hierarchical bank contrast shapes or floor are invalid")
        bank_invocations = {
            head_id.split("/query-head.", 1)[0] for head_id in self.head_ids
        }
        if bank_invocations != set(self.hierarchy.invocation_ids):
            raise ValueError("hierarchical bank head catalog does not match layout")
        object.__setattr__(self, "block_scale_contrasts", block)
        object.__setattr__(self, "site_scale_contrasts", site)
        object.__setattr__(self, "signed_feature_floor", float(self.signed_feature_floor))
        shared_hash = FisherAttentionTransportGeneratorBank.artifact_sha256.fget(self)
        hierarchy_hash = self.hierarchy.artifact_sha256
        hashes = tuple(
            _tensor_sha256(value)
            for value in (
                self.left_factors,
                self.singular_scales,
                self.right_factors,
                self.head_coefficients,
                block,
                site,
            )
        )
        object.__setattr__(self, "_hierarchical_tensor_sha256s", hashes)
        object.__setattr__(self, "_shared_bank_sha256", shared_hash)
        object.__setattr__(self, "_hierarchy_sha256", hierarchy_hash)
        object.__setattr__(
            self,
            "_artifact_sha256",
            hashlib.sha256(
                _BANK_DOMAIN
                + _canonical_json_bytes(self._artifact_payload(hashes))
            ).hexdigest(),
        )

    def _artifact_payload(self, hashes: tuple[str, ...]) -> dict[str, object]:
        return {
            "block_scale_contrasts_sha256": hashes[4],
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "generator_count": self.generator_count,
            "generator_rank": self.generator_rank,
            "head_coefficients_sha256": hashes[3],
            "head_ids": list(self.head_ids),
            "hierarchy_semantics": HIERARCHY_SEMANTICS,
            "hierarchy_sha256": self._hierarchy_sha256,
            "hidden_width": self.hidden_width,
            "left_factors_sha256": hashes[0],
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "right_factors_sha256": hashes[2],
            "shared_bank_sha256": self._shared_bank_sha256,
            "signed_feature_floor": float(self.signed_feature_floor),
            "signed_router_feature_semantics": SIGNED_ROUTER_FEATURE_SEMANTICS,
            "singular_scales_sha256": hashes[1],
            "site_scale_contrasts_sha256": hashes[5],
        }

    def _validate_unchanged(self) -> None:
        current = tuple(
            _tensor_sha256(value)
            for value in (
                self.left_factors,
                self.singular_scales,
                self.right_factors,
                self.head_coefficients,
                self.block_scale_contrasts,
                self.site_scale_contrasts,
            )
        )
        if current != self._hierarchical_tensor_sha256s:
            raise RuntimeError("hierarchical generator bank changed after construction")
        if self.hierarchy.artifact_sha256 != self._hierarchy_sha256:
            raise RuntimeError("hierarchical generator layout changed after construction")

    @property
    def shared_bank_sha256(self) -> str:
        self._validate_unchanged()
        return self._shared_bank_sha256

    @property
    def hierarchy_parameter_count(self) -> int:
        return int(
            self.block_scale_contrasts.numel()
            + self.site_scale_contrasts.numel()
        )

    @property
    def stored_parameter_count(self) -> int:
        shared = FisherAttentionTransportGeneratorBank.stored_parameter_count.fget(self)
        return int(shared + self.hierarchy_parameter_count)

    @property
    def artifact_sha256(self) -> str:
        self._validate_unchanged()
        return self._artifact_sha256

    def __hash__(self) -> int:
        return hash((HierarchicalFisherAttentionTransportGeneratorBank, self.artifact_sha256))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HierarchicalFisherAttentionTransportGeneratorBank)
            and self.artifact_sha256 == other.artifact_sha256
        )

    def effective_signed_diagonal(self, invocation_id: str) -> Tensor:
        self._validate_unchanged()
        block_index, site_index = self.hierarchy.coordinates(invocation_id)
        block_delta = torch.einsum(
            "p,pkr->kr",
            self.hierarchy.block_contrast[block_index],
            self.block_scale_contrasts,
        )
        site_delta = torch.einsum(
            "q,qkr->kr",
            self.hierarchy.site_contrast[site_index],
            self.site_scale_contrasts,
        )
        return (self.singular_scales + block_delta + site_delta).contiguous()

    def candidate_outputs_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        latent = torch.einsum("nki,kir->nkr", contextual, self.right_factors)
        latent = latent * self.effective_signed_diagonal(invocation_id)[None]
        return torch.einsum("nkr,kor->nko", latent, self.left_factors).contiguous()

    def router_context_features_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        rms = contextual.square().mean(dim=-1).sqrt()
        rho = torch.log1p(rms)
        latent = torch.einsum("nki,kir->nkr", contextual, self.right_factors)
        diagonal = self.effective_signed_diagonal(invocation_id)
        numerator = (latent * diagonal[None]).sum(dim=-1)
        denominator = (
            torch.linalg.vector_norm(latent, dim=-1)
            * torch.linalg.vector_norm(diagonal, dim=-1)[None]
        ).clamp_min(float(self.signed_feature_floor))
        alignment = (numerator / denominator).clamp(-1.0, 1.0)
        return torch.cat((rho, alignment), dim=-1).contiguous()

    def state_dict(self) -> dict[str, object]:
        self._validate_unchanged()
        return {
            "artifact_sha256": self._artifact_sha256,
            "block_scale_contrasts": self.block_scale_contrasts.clone(),
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "head_coefficients": self.head_coefficients.clone(),
            "head_ids": list(self.head_ids),
            "hierarchy": self.hierarchy.state_dict(),
            "hierarchy_semantics": HIERARCHY_SEMANTICS,
            "left_factors": self.left_factors.clone(),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "right_factors": self.right_factors.clone(),
            "shared_bank_sha256": self._shared_bank_sha256,
            "signed_feature_floor": float(self.signed_feature_floor),
            "signed_router_feature_semantics": SIGNED_ROUTER_FEATURE_SEMANTICS,
            "singular_scales": self.singular_scales.clone(),
            "site_scale_contrasts": self.site_scale_contrasts.clone(),
        }


def hierarchical_fisher_attention_transport_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    expected = {
        "artifact_sha256",
        "block_scale_contrasts",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "head_coefficients",
        "head_ids",
        "hierarchy",
        "hierarchy_semantics",
        "left_factors",
        "native_output_rmsnorm_retained",
        "native_qk_retained",
        "right_factors",
        "shared_bank_sha256",
        "signed_feature_floor",
        "signed_router_feature_semantics",
        "singular_scales",
        "site_scale_contrasts",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("hierarchical bank state fields are invalid")
    raw_ids = state["head_ids"]
    raw_layout = state["hierarchy"]
    if (
        not isinstance(raw_ids, list)
        or not isinstance(raw_layout, Mapping)
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_raw_teacher_outputs_or_gradients"] is not False
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
        or state["hierarchy_semantics"] != HIERARCHY_SEMANTICS
        or state["signed_router_feature_semantics"]
        != SIGNED_ROUTER_FEATURE_SEMANTICS
    ):
        raise ValueError("hierarchical bank state safety declarations are invalid")
    bank = HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=state["left_factors"],
        singular_scales=state["singular_scales"],
        right_factors=state["right_factors"],
        head_coefficients=state["head_coefficients"],
        head_ids=tuple(raw_ids),
        hierarchy=attention_transport_hierarchy_layout_from_state_dict(raw_layout),
        block_scale_contrasts=state["block_scale_contrasts"],
        site_scale_contrasts=state["site_scale_contrasts"],
        signed_feature_floor=state["signed_feature_floor"],
    )
    if (
        state["shared_bank_sha256"] != bank.shared_bank_sha256
        or state["artifact_sha256"] != bank.artifact_sha256
    ):
        raise ValueError("hierarchical bank artifact lineage drifted")
    return bank


@dataclass(frozen=True, slots=True, eq=False)
class HierarchicalEdgeGlobalSupportRouter(
    ChartConditionedAttentionTransportRouter
):
    """Chart-conditioned router with signed 2K features and one global TopK."""

    invocation_id: str
    generator_bank_sha256: str
    atlas: FrozenAttentionChartAtlas
    selection_weight: Tensor
    selection_bias: Tensor
    amplitude_weight: Tensor
    amplitude_bias: Tensor
    active_count: int
    _parameter_sha256s: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _atlas_sha256: str = field(init=False, repr=False, compare=False)
    _artifact_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("hierarchical router invocation_id must be nonempty")
        if not _is_sha256(self.generator_bank_sha256):
            raise ValueError("hierarchical router bank hash is invalid")
        if not isinstance(self.atlas, FrozenAttentionChartAtlas):
            raise TypeError("atlas must be a FrozenAttentionChartAtlas")
        selection_weight = _cpu_float64(
            self.selection_weight, label="selection_weight", ndim=3
        )
        selection_bias = _cpu_float64(
            self.selection_bias, label="selection_bias", ndim=2
        )
        amplitude_weight = _cpu_float64(
            self.amplitude_weight, label="amplitude_weight", ndim=3
        )
        amplitude_bias = _cpu_float64(
            self.amplitude_bias, label="amplitude_bias", ndim=2
        )
        generators = int(selection_bias.shape[1])
        expected_prefix = (self.atlas.chart_count, generators)
        expected_features = self.atlas.chart_rank + 2 * generators
        if (
            selection_bias.shape != expected_prefix
            or amplitude_bias.shape != expected_prefix
            or selection_weight.shape != expected_prefix + (expected_features,)
            or amplitude_weight.shape != selection_weight.shape
            or type(self.active_count) is not int
            or not 0 < self.active_count <= generators
        ):
            raise ValueError("hierarchical router parameter shapes are invalid")
        for name, value in (
            ("selection_weight", selection_weight),
            ("selection_bias", selection_bias),
            ("amplitude_weight", amplitude_weight),
            ("amplitude_bias", amplitude_bias),
        ):
            object.__setattr__(self, name, value)
        hashes = tuple(
            _tensor_sha256(value)
            for value in (
                selection_weight,
                selection_bias,
                amplitude_weight,
                amplitude_bias,
            )
        )
        object.__setattr__(self, "_parameter_sha256s", hashes)
        object.__setattr__(self, "_atlas_sha256", self.atlas.artifact_sha256)
        object.__setattr__(
            self,
            "_artifact_sha256",
            hashlib.sha256(
                _ROUTER_DOMAIN
                + _canonical_json_bytes(
                    {
                        "active_count": self.active_count,
                        "amplitude_bias_sha256": hashes[3],
                        "amplitude_weight_sha256": hashes[2],
                        "atlas_sha256": self.atlas.artifact_sha256,
                        "contains_oracle_outputs_or_coefficients": False,
                        "contains_teacher_outputs_or_gradients": False,
                        "generator_bank_sha256": self.generator_bank_sha256,
                        "invocation_id": self.invocation_id,
                        "selection_bias_sha256": hashes[1],
                        "selection_weight_sha256": hashes[0],
                        "signed_router_feature_semantics": (
                            SIGNED_ROUTER_FEATURE_SEMANTICS
                        ),
                        "support_semantics": GLOBAL_SUPPORT_SEMANTICS,
                    }
                )
            ).hexdigest(),
        )

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
            raise RuntimeError("hierarchical router changed after construction")
        if self.atlas.artifact_sha256 != self._atlas_sha256:
            raise RuntimeError("hierarchical router atlas changed after construction")

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

    def __hash__(self) -> int:
        return hash((HierarchicalEdgeGlobalSupportRouter, self.artifact_sha256))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HierarchicalEdgeGlobalSupportRouter)
            and self.artifact_sha256 == other.artifact_sha256
        )

    def execute(
        self,
        chart_inputs: Tensor,
        contextual_features: Tensor,
    ) -> ChartConditionedRouterExecution:
        self._validate_unchanged()
        coordinates, memberships = self.atlas.coordinates_and_memberships(chart_inputs)
        context = _cpu_float64(
            contextual_features,
            label="hierarchical router contextual_features",
            ndim=2,
        )
        if context.shape != (coordinates.shape[0], 2 * self.generator_count):
            raise ValueError("hierarchical router features do not align with rows")
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
            probabilities, k=self.active_count, dim=-1, sorted=True
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
            "nc,nck->nk", memberships, chart_amplitudes
        )
        coefficients = sparse * mixed_amplitudes
        charts = self.atlas.chart_count
        expanded_probabilities = probabilities[:, None, :].expand(-1, charts, -1).contiguous()
        expanded_sparse = sparse[:, None, :].expand(-1, charts, -1).contiguous()
        expanded_amplitudes = mixed_amplitudes[:, None, :].expand(-1, charts, -1).contiguous()
        expanded_active = active[:, None, :].expand(-1, charts, -1).contiguous()
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
        if not isinstance(bank, HierarchicalFisherAttentionTransportGeneratorBank):
            raise TypeError("router requires a hierarchical generator bank")
        if bank.artifact_sha256 != self.generator_bank_sha256:
            raise ValueError("router is not bound to this hierarchical bank")
        context = bank.router_context_features_from_head_contexts(
            invocation_id=self.invocation_id,
            head_contexts=head_contexts,
        )
        execution = self.execute(chart_inputs, context)
        candidates = bank.candidate_outputs_from_head_contexts(
            invocation_id=self.invocation_id,
            head_contexts=head_contexts,
        )
        return torch.einsum(
            "nk,nkh->nh", execution.generator_coefficients, candidates
        ).contiguous()

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
            "signed_router_feature_semantics": SIGNED_ROUTER_FEATURE_SEMANTICS,
            "support_semantics": GLOBAL_SUPPORT_SEMANTICS,
        }


def hierarchical_global_support_attention_transport_router_from_state_dict(
    state: Mapping[str, object],
    *,
    atlas: FrozenAttentionChartAtlas,
    bank: HierarchicalFisherAttentionTransportGeneratorBank,
) -> HierarchicalEdgeGlobalSupportRouter:
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
        "signed_router_feature_semantics",
        "support_semantics",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("hierarchical router state fields are invalid")
    if (
        state["atlas_sha256"] != atlas.artifact_sha256
        or state["generator_bank_sha256"] != bank.artifact_sha256
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
        or state["signed_router_feature_semantics"]
        != SIGNED_ROUTER_FEATURE_SEMANTICS
        or state["support_semantics"] != GLOBAL_SUPPORT_SEMANTICS
    ):
        raise ValueError("hierarchical router state lineage is invalid")
    router = HierarchicalEdgeGlobalSupportRouter(
        invocation_id=state["invocation_id"],
        generator_bank_sha256=state["generator_bank_sha256"],
        atlas=atlas,
        selection_weight=state["selection_weight"],
        selection_bias=state["selection_bias"],
        amplitude_weight=state["amplitude_weight"],
        amplitude_bias=state["amplitude_bias"],
        active_count=state["active_count"],
    )
    if state["artifact_sha256"] != router.artifact_sha256:
        raise ValueError("hierarchical router artifact hash drifted")
    return router


hierarchical_edge_global_support_router_from_state_dict = (
    hierarchical_global_support_attention_transport_router_from_state_dict
)


@dataclass(frozen=True, slots=True)
class HierarchicalSparseAttentionTransportFitConfig:
    """Fixed A-only protocol for crossed signed-scale residual fitting."""

    shared: SparseAwareAttentionTransportFitConfig
    alternating_rounds: int = 4
    hierarchy_ridge: float = 1.0e-8
    normal_equation_jitter: float = 1.0e-12
    signed_feature_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.shared, SparseAwareAttentionTransportFitConfig):
            raise TypeError("shared must be a SparseAwareAttentionTransportFitConfig")
        if type(self.alternating_rounds) is not int or self.alternating_rounds != 4:
            raise ValueError("hierarchical v1 requires exactly four alternating rounds")
        for label, value in (
            ("hierarchy_ridge", self.hierarchy_ridge),
            ("normal_equation_jitter", self.normal_equation_jitter),
            ("signed_feature_floor", self.signed_feature_floor),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive")

    @classmethod
    def fixed_k8_r8_top2(cls) -> HierarchicalSparseAttentionTransportFitConfig:
        return cls(
            shared=SparseAwareAttentionTransportFitConfig(
                generator_count=8,
                generator_rank=8,
                active_count=2,
                dense_warm_start_steps=96,
                alternating_rounds=4,
                steps_per_round=32,
                learning_rate=1.0e-2,
                euclidean_floor=1.0e-3,
                oracle_coefficient_ridge=1.0e-8,
                dense_head_coefficient_ridge=1.0e-8,
                gradient_clip_norm=10.0,
                gauge_floor=1.0e-12,
            )
        )


@dataclass(frozen=True, slots=True)
class HierarchicalSparseAttentionTransportBankFit:
    bank: HierarchicalFisherAttentionTransportGeneratorBank
    shared_sparse_fit: SparseAwareAttentionTransportBankFit
    initial_fit_oracle: SparseAttentionTransportOracleEvaluation
    final_fit_oracle: SparseAttentionTransportOracleEvaluation
    fit_round_relative_fisher_losses: tuple[float, ...]
    hierarchy_layout_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.bank,
                HierarchicalFisherAttentionTransportGeneratorBank,
            )
            or not isinstance(
                self.shared_sparse_fit,
                SparseAwareAttentionTransportBankFit,
            )
            or not isinstance(
                self.initial_fit_oracle,
                SparseAttentionTransportOracleEvaluation,
            )
            or not isinstance(
                self.final_fit_oracle,
                SparseAttentionTransportOracleEvaluation,
            )
            or self.bank.shared_bank_sha256
            != self.shared_sparse_fit.bank.artifact_sha256
            or self.hierarchy_layout_sha256
            != self.bank.hierarchy.artifact_sha256
            or not _is_sha256(self.hierarchy_layout_sha256)
            or len(self.fit_round_relative_fisher_losses) != 5
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in self.fit_round_relative_fisher_losses
            )
            or any(
                right > left + 1.0e-12
                for left, right in zip(
                    self.fit_round_relative_fisher_losses,
                    self.fit_round_relative_fisher_losses[1:],
                )
            )
            or not math.isclose(
                self.fit_round_relative_fisher_losses[0],
                self.initial_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                self.fit_round_relative_fisher_losses[-1],
                self.final_fit_oracle.relative_fisher_loss,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("hierarchical sparse bank fit is invalid")

    @property
    def dense_warm_start(self) -> AttentionTransportBankFit:
        return self.shared_sparse_fit.dense_warm_start

    @property
    def shared_bank(self) -> FisherAttentionTransportGeneratorBank:
        return self.shared_sparse_fit.bank


def _capture_values(
    captures: Sequence[AttentionTransportBatch],
) -> tuple[AttentionTransportBatch, ...]:
    if isinstance(captures, (str, bytes)) or not isinstance(captures, Sequence):
        raise TypeError("captures must be a materialized sequence")
    values = tuple(captures)
    if not values or any(
        not isinstance(value, (AttentionTransportCapture, AttentionTransportRowBatch))
        for value in values
    ):
        raise ValueError("captures must contain attention transport captures")
    return values


def _group_capture_indices_numeric(
    captures: Sequence[AttentionTransportBatch],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: dict[str, list[int]] = {}
    for index, capture in enumerate(captures):
        groups.setdefault(capture.invocation_id, []).append(index)
    return tuple(
        (invocation_id, tuple(groups[invocation_id]))
        for invocation_id in sorted(groups, key=_mutation_ordinal)
    )


def _exact_sparse_coefficients_by_capture(
    bank: FisherAttentionTransportGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> tuple[Tensor, ...]:
    coefficients: list[Tensor | None] = [None] * len(captures)
    for _invocation_id, indices in _group_capture_indices_numeric(captures):
        candidates = tuple(bank.candidate_outputs(captures[index]) for index in indices)
        row_counts = tuple(int(value.shape[0]) for value in candidates)
        oracle = solve_fisher_sparse_transport_oracle(
            torch.cat(candidates),
            torch.cat(
                tuple(captures[index].valid_operator_outputs for index in indices)
            ),
            torch.cat(
                tuple(captures[index].valid_score_gradients for index in indices)
            ),
            active_count=active_count,
            euclidean_floor=euclidean_floor,
            coefficient_ridge=coefficient_ridge,
        )
        for index, value in zip(
            indices,
            oracle.coefficients.split(row_counts),
            strict=True,
        ):
            coefficients[index] = value.detach().clone().contiguous()
    if any(value is None for value in coefficients):
        raise RuntimeError("hierarchical sparse oracle omitted one capture")
    return tuple(value for value in coefficients if value is not None)


def _zero_hierarchical_bank(
    shared: FisherAttentionTransportGeneratorBank,
    hierarchy: AttentionTransportHierarchyLayout,
    *,
    signed_feature_floor: float,
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    return HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=shared.left_factors,
        singular_scales=shared.singular_scales,
        right_factors=shared.right_factors,
        head_coefficients=shared.head_coefficients,
        head_ids=shared.head_ids,
        hierarchy=hierarchy,
        block_scale_contrasts=torch.zeros(
            hierarchy.block_count - 1,
            shared.generator_count,
            shared.generator_rank,
            dtype=torch.float64,
        ),
        site_scale_contrasts=torch.zeros(
            hierarchy.site_count - 1,
            shared.generator_count,
            shared.generator_rank,
            dtype=torch.float64,
        ),
        signed_feature_floor=signed_feature_floor,
    )


def _hierarchical_bank_with_coordinates(
    template: HierarchicalFisherAttentionTransportGeneratorBank,
    coordinates: Tensor,
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    local = template.generator_count * template.generator_rank
    block_size = (template.hierarchy.block_count - 1) * local
    if coordinates.ndim != 1 or coordinates.numel() != template.hierarchy_parameter_count:
        raise ValueError("hierarchical coordinate vector has the wrong shape")
    block = coordinates[:block_size].reshape(
        template.hierarchy.block_count - 1,
        template.generator_count,
        template.generator_rank,
    )
    site = coordinates[block_size:].reshape(
        template.hierarchy.site_count - 1,
        template.generator_count,
        template.generator_rank,
    )
    return HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=template.left_factors,
        singular_scales=template.singular_scales,
        right_factors=template.right_factors,
        head_coefficients=template.head_coefficients,
        head_ids=template.head_ids,
        hierarchy=template.hierarchy,
        block_scale_contrasts=block,
        site_scale_contrasts=site,
        signed_feature_floor=template.signed_feature_floor,
    )


def _invocation_contrast_design(
    hierarchy: AttentionTransportHierarchyLayout,
    invocation_id: str,
    *,
    local_parameter_count: int,
) -> Tensor:
    block_index, site_index = hierarchy.coordinates(invocation_id)
    block_row = hierarchy.block_contrast[block_index]
    site_row = hierarchy.site_contrast[site_index]
    total_contrasts = block_row.numel() + site_row.numel()
    design = torch.zeros(
        local_parameter_count,
        total_contrasts * local_parameter_count,
        dtype=torch.float64,
    )
    identity = torch.eye(local_parameter_count, dtype=torch.float64)
    for contrast_index, coefficient in enumerate(
        torch.cat((block_row, site_row))
    ):
        begin = contrast_index * local_parameter_count
        design[:, begin : begin + local_parameter_count] = coefficient * identity
    return design


def _solve_hierarchical_contrast_normal_equations(
    shared: FisherAttentionTransportGeneratorBank,
    hierarchy: AttentionTransportHierarchyLayout,
    captures: Sequence[AttentionTransportBatch],
    oracle_coefficients: Sequence[Tensor],
    *,
    euclidean_floor: float,
    hierarchy_ridge: float,
    normal_equation_jitter: float,
) -> Tensor:
    """Accumulate 64-local equations, then map each invocation into 448 DOFs."""

    local_count = shared.generator_count * shared.generator_rank
    global_count = (
        hierarchy.block_count - 1 + hierarchy.site_count - 1
    ) * local_count
    global_gram = torch.zeros(global_count, global_count, dtype=torch.float64)
    global_rhs = torch.zeros(global_count, dtype=torch.float64)
    left_directions = shared.left_factors.permute(0, 2, 1).reshape(
        local_count,
        shared.hidden_width,
    )
    left_gram = left_directions @ left_directions.T
    total_rows = 0

    coefficients_by_index = tuple(oracle_coefficients)
    for invocation_id, indices in _group_capture_indices_numeric(captures):
        local_gram = torch.zeros(local_count, local_count, dtype=torch.float64)
        local_rhs = torch.zeros(local_count, dtype=torch.float64)
        for index in indices:
            capture = captures[index]
            beta = coefficients_by_index[index]
            contextual = shared.contextual_inputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=capture.head_contexts,
            )
            latent = torch.einsum(
                "nki,kir->nkr", contextual, shared.right_factors
            )
            local_amplitudes = (beta[:, :, None] * latent).reshape(
                capture.row_count,
                local_count,
            )
            shared_prediction = torch.einsum(
                "nk,nkh->nh", beta, shared.candidate_outputs(capture)
            )
            residual = capture.valid_operator_outputs - shared_prediction
            gradients = capture.valid_score_gradients
            gradient_on_left = gradients @ left_directions.T
            directional_design = local_amplitudes * gradient_on_left
            local_gram += directional_design.T @ directional_design
            local_gram += float(euclidean_floor) * (
                (local_amplitudes.T @ local_amplitudes) * left_gram
            )
            directional_target = (gradients * residual).sum(dim=-1)
            local_rhs += directional_design.T @ directional_target
            residual_on_left = residual @ left_directions.T
            local_rhs += float(euclidean_floor) * (
                local_amplitudes * residual_on_left
            ).sum(dim=0)
            total_rows += capture.row_count
        design = _invocation_contrast_design(
            hierarchy,
            invocation_id,
            local_parameter_count=local_count,
        )
        global_gram += design.T @ local_gram @ design
        global_rhs += design.T @ local_rhs

    if total_rows <= 0:
        raise RuntimeError("hierarchical normal equations observed no rows")
    global_gram = global_gram / float(total_rows)
    global_rhs = global_rhs / float(total_rows)
    global_gram = 0.5 * (global_gram + global_gram.T)
    diagonal = float(hierarchy_ridge) + float(normal_equation_jitter)
    global_gram = global_gram + diagonal * torch.eye(
        global_count, dtype=torch.float64
    )
    factor, info = torch.linalg.cholesky_ex(global_gram, check_errors=False)
    if int(info) != 0:
        raise RuntimeError("hierarchical Fisher normal equations are not positive definite")
    solution = torch.cholesky_solve(global_rhs[:, None], factor).squeeze(-1)
    if not bool(torch.isfinite(solution).all()):
        raise RuntimeError("hierarchical Fisher normal-equation solution is nonfinite")
    return solution.contiguous()


def fit_hierarchical_sparse_attention_transport_generator_bank(
    a_captures: Sequence[AttentionTransportBatch],
    *,
    config: HierarchicalSparseAttentionTransportFitConfig,
    shared_fit: SparseAwareAttentionTransportBankFit | None = None,
    round_observer: Callable[
        [int, int, SparseAttentionTransportOracleEvaluation], None
    ]
    | None = None,
) -> HierarchicalSparseAttentionTransportBankFit:
    """Fit all deployable hierarchy coordinates on A, and A, only."""

    if not isinstance(config, HierarchicalSparseAttentionTransportFitConfig):
        raise TypeError("config must be a hierarchical sparse fit config")
    if round_observer is not None and not callable(round_observer):
        raise TypeError("round_observer must be callable")
    captures = _capture_values(a_captures)
    _, invocations, _, _ = _capture_catalog(captures)
    hierarchy = AttentionTransportHierarchyLayout.gemma3_270m_attention(
        tuple(value.invocation_id for value in invocations)
    )
    shared = (
        fit_sparse_aware_attention_transport_generator_bank(
            captures,
            config=config.shared,
        )
        if shared_fit is None
        else shared_fit
    )
    if (
        not isinstance(shared, SparseAwareAttentionTransportBankFit)
        or shared.bank.generator_count != config.shared.generator_count
        or shared.bank.generator_rank != config.shared.generator_rank
        or set(shared.bank.head_ids)
        != {
            head_id
            for value in invocations
            for head_id in value.head_ids
        }
    ):
        raise ValueError("shared sparse fit does not match hierarchical A captures")
    current = _zero_hierarchical_bank(
        shared.bank,
        hierarchy,
        signed_feature_floor=config.signed_feature_floor,
    )
    initial = evaluate_sparse_attention_transport_bank(
        current,
        captures,
        active_count=config.shared.active_count,
        euclidean_floor=config.shared.euclidean_floor,
        coefficient_ridge=config.shared.oracle_coefficient_ridge,
    )
    best_bank = current
    best_evaluation = initial
    history = [float(initial.relative_fisher_loss)]
    for round_index in range(1, config.alternating_rounds + 1):
        coefficients = _exact_sparse_coefficients_by_capture(
            best_bank,
            captures,
            active_count=config.shared.active_count,
            euclidean_floor=config.shared.euclidean_floor,
            coefficient_ridge=config.shared.oracle_coefficient_ridge,
        )
        coordinates = _solve_hierarchical_contrast_normal_equations(
            shared.bank,
            hierarchy,
            captures,
            coefficients,
            euclidean_floor=config.shared.euclidean_floor,
            hierarchy_ridge=config.hierarchy_ridge,
            normal_equation_jitter=config.normal_equation_jitter,
        )
        candidate = _hierarchical_bank_with_coordinates(current, coordinates)
        evaluation = evaluate_sparse_attention_transport_bank(
            candidate,
            captures,
            active_count=config.shared.active_count,
            euclidean_floor=config.shared.euclidean_floor,
            coefficient_ridge=config.shared.oracle_coefficient_ridge,
        )
        if (
            evaluation.relative_fisher_loss
            < best_evaluation.relative_fisher_loss - 1.0e-12
        ):
            best_bank = candidate
            best_evaluation = evaluation
        history.append(float(best_evaluation.relative_fisher_loss))
        if round_observer is not None:
            round_observer(round_index, config.alternating_rounds, best_evaluation)
    return HierarchicalSparseAttentionTransportBankFit(
        bank=best_bank,
        shared_sparse_fit=shared,
        initial_fit_oracle=initial,
        final_fit_oracle=best_evaluation,
        fit_round_relative_fisher_losses=tuple(history),
        hierarchy_layout_sha256=hierarchy.artifact_sha256,
    )


def _training_global_router_execution(
    features: Tensor,
    memberships: Tensor,
    selection_weight: Tensor,
    selection_bias: Tensor,
    amplitude_weight: Tensor,
    amplitude_bias: Tensor,
    *,
    active_count: int,
) -> tuple[Tensor, Tensor]:
    chart_logits = (
        torch.einsum("ncf,ckf->nck", features, selection_weight)
        + selection_bias[None]
    )
    probabilities = torch.softmax(
        torch.einsum("nc,nck->nk", memberships, chart_logits),
        dim=-1,
    )
    indices = torch.topk(probabilities, k=active_count, dim=-1).indices
    active = torch.zeros_like(probabilities, dtype=torch.bool)
    active.scatter_(-1, indices, True)
    hard = probabilities.masked_fill(~active, 0.0)
    hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    straight_through = probabilities + (hard - probabilities).detach()
    chart_amplitudes = (
        torch.einsum("ncf,ckf->nck", features, amplitude_weight)
        + amplitude_bias[None]
    )
    mixed_amplitudes = torch.einsum(
        "nc,nck->nk", memberships, chart_amplitudes
    )
    return straight_through * mixed_amplitudes, probabilities


def fit_hierarchical_edge_attention_transport_router(
    bank: HierarchicalFisherAttentionTransportGeneratorBank,
    capture: AttentionTransportBatch | Sequence[AttentionTransportBatch],
    atlas: FrozenAttentionChartAtlas,
    *,
    config: ChartConditionedRouterConfig,
) -> ChartConditionedRouterFit:
    """Fit one signed-edge global-support router using one A invocation."""

    if not isinstance(bank, HierarchicalFisherAttentionTransportGeneratorBank):
        raise TypeError("bank must be a hierarchical generator bank")
    values = (
        (capture,)
        if isinstance(capture, (AttentionTransportCapture, AttentionTransportRowBatch))
        else _capture_values(capture)
    )
    captures, invocations, _, _ = _capture_catalog(values)
    if len(invocations) != 1:
        raise ValueError("one hierarchical router fit may cover one invocation")
    representative = invocations[0]
    bank.hierarchy.coordinates(representative.invocation_id)
    if not isinstance(atlas, FrozenAttentionChartAtlas):
        raise TypeError("atlas must be a FrozenAttentionChartAtlas")
    if not isinstance(config, ChartConditionedRouterConfig):
        raise TypeError("config must be a ChartConditionedRouterConfig")
    if (
        atlas.hidden_width != bank.hidden_width
        or config.active_count > bank.generator_count
    ):
        raise ValueError("hierarchical atlas, router, and bank dimensions drifted")

    candidates = torch.cat(tuple(bank.candidate_outputs(value) for value in captures))
    targets = torch.cat(tuple(value.valid_operator_outputs for value in captures))
    gradients = torch.cat(tuple(value.valid_score_gradients for value in captures))
    chart_inputs = torch.cat(tuple(value.valid_chart_inputs for value in captures))
    head_contexts = torch.cat(tuple(value.head_contexts for value in captures))
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        targets,
        gradients,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.oracle_ridge,
    )
    oracle_prediction = torch.einsum("nk,nkh->nh", oracle.coefficients, candidates)
    coordinates, memberships = atlas.coordinates_and_memberships(chart_inputs)
    context = bank.router_context_features_from_head_contexts(
        invocation_id=representative.invocation_id,
        head_contexts=head_contexts,
    )
    features = torch.cat(
        (
            coordinates,
            context[:, None, :].expand(-1, atlas.chart_count, -1),
        ),
        dim=-1,
    )
    charts = atlas.chart_count
    generators = bank.generator_count
    feature_count = atlas.chart_rank + 2 * generators
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
    initial_indices = torch.topk(target_usage, k=config.active_count).indices
    initial_active = torch.zeros_like(target_usage, dtype=torch.bool)
    initial_active[initial_indices] = True
    initial_sparse = target_usage.masked_fill(~initial_active, 0.0)
    initial_sparse = initial_sparse / initial_sparse.sum().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    mean_beta = oracle.coefficients.mean(dim=0)
    amplitude_bias = nn.Parameter(
        torch.where(
            initial_active,
            mean_beta / initial_sparse.clamp_min(1.0e-6),
            torch.zeros_like(mean_beta),
        )[None].expand(charts, -1).clone()
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
        beta, probabilities = _training_global_router_execution(
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
            raise RuntimeError("hierarchical signed-edge router fit became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float(config.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()

    router = HierarchicalEdgeGlobalSupportRouter(
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
    return ChartConditionedRouterFit(
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


fit_hierarchical_global_support_attention_transport_router = (
    fit_hierarchical_edge_attention_transport_router
)
HierarchicalGlobalSupportAttentionTransportRouter = (
    HierarchicalEdgeGlobalSupportRouter
)


__all__ = [
    "AttentionTransportHierarchyLayout",
    "GLOBAL_SUPPORT_SEMANTICS",
    "HIERARCHY_SEMANTICS",
    "HierarchicalEdgeGlobalSupportRouter",
    "HierarchicalFisherAttentionTransportGeneratorBank",
    "HierarchicalGlobalSupportAttentionTransportRouter",
    "HierarchicalSparseAttentionTransportBankFit",
    "HierarchicalSparseAttentionTransportFitConfig",
    "SIGNED_ROUTER_FEATURE_SEMANTICS",
    "attention_transport_hierarchy_layout_from_state_dict",
    "fit_hierarchical_edge_attention_transport_router",
    "fit_hierarchical_global_support_attention_transport_router",
    "fit_hierarchical_sparse_attention_transport_generator_bank",
    "hierarchical_edge_global_support_router_from_state_dict",
    "hierarchical_fisher_attention_transport_generator_bank_from_state_dict",
    "hierarchical_global_support_attention_transport_router_from_state_dict",
]
