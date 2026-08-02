"""A16 held-family endpoint token-Fisher and finite tail ladder.

The frozen D320 carrier is the endpoint.  For each whole-family-held-out fold,
the other seven A16 families fit a full basis of ``null(D320)`` from the actual
one-pass residual and order it by exact token-wise H4 VJP energy.  The held
family then evaluates D320+K for K=8,16,32,64,320 with finite one-pass
executions.  K320 is produced by the same projection formula as every smaller
arm; it is never replaced by an exact-residual provider.

This deliberately uses native held-prompt residuals to instantiate corrections
and is therefore truth-leaking hypothesis evidence only.  It fits no serving
provider and makes no compression, speed, or deployment claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from .complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
    CompleteH4TailHeldFamilyFit,
    complete_h4_tail_gate_scores,
    fit_complete_h4_tail_held_family,
    project_complete_h4_tail_prefix,
    project_complete_h4_tail_rows,
)
from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import (
    AuthenticatedCompleteH4TransferProvider,
    _ValidatedRank320BasisContract,
    _load_committed_basis,
    _native_boundary,
    _retokenize,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    DEFAULT_OUTPUT as DEFAULT_MATERIALIZATION_REPORT,
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _canonical_json_bytes,
    _require_sha256,
    _runtime_tensor_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_TRANSFER_REPORT",
    "MATERIALIZATION_REPORT_FILE_SHA256",
    "MATERIALIZATION_REPORT_SHA256",
    "TRANSFER_REPORT_FILE_SHA256",
    "TRANSFER_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_TRANSFER_REPORT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "informed-rank320-one-pass-transfer-a-fit16-dev-v1.json"
)
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-lofo-finite-ladder-a-fit16-dev-v1.json"
)

MATERIALIZATION_REPORT_SHA256 = (
    "2689e749f8f25b9a597df006b16613e6860a37811b1afe662b5ad0e524147bc4"
)
MATERIALIZATION_REPORT_FILE_SHA256 = (
    "c09f7b95ca983fceed64935be00e2fb5b507b62725e40cbd6b172672d3a1a56c"
)
TRANSFER_REPORT_SHA256 = (
    "98f084c78e8d8c8d52ed2b7ae9544ba2711c6e96d6c31d3af6453d3b0b8544fd"
)
TRANSFER_REPORT_FILE_SHA256 = (
    "056e34a36adc512f0fbb5d9677a4796a43ef4aa778c5ebb27b06a4cd87d6131a"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_lofo.v1"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-tail-token-fisher-lofo:v1\0"
_PROVIDER_DOMAIN = b"fisher-graph:gemma3-l3-l4-tail-finite-provider:v1\0"
_H4_SITE = "layer.4.output"
_WIDTH = 640
_D_RANK = 320
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_ORDINARY_TOKENS = 931
_EXPECTED_SUPPORT_TOKENS = 803
_EXPECTED_GRAPH_CORE_TOKENS = 790
_EXPECTED_CAUSAL_TAIL_TOKENS = 13
_EXPECTED_SUPPORT_ROWS = 819
_EXPECTED_GRAPH_CORE_ROWS = 802
_EXPECTED_CAUSAL_TAIL_ROWS = 17
_TAIL_RANKS = (8, 16, 32, 64, 320)
_VJP_CHUNK_SIZE = 8


def _validated_tail_ranks(ranks: tuple[int, ...]) -> tuple[int, ...]:
    """Validate a fixed finite-ladder rank tuple shared by follow-up screens."""

    if (
        type(ranks) is not tuple
        or len(ranks) < 2
        or any(type(rank) is not int for rank in ranks)
        or any(rank < 1 or rank > _D_RANK for rank in ranks)
        or tuple(sorted(set(ranks))) != ranks
        or 64 not in ranks
        or ranks[-1] != _D_RANK
    ):
        raise ValueError(
            "tail ranks must be a strictly increasing fixed tuple in [1, 320] "
            "containing rank 64 and ending with the rank-320 sentinel"
        )
    return ranks


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_report(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
    label: str,
) -> dict[str, object]:
    selected = Path(path)
    if _file_sha256(selected) != _require_sha256(
        expected_file_sha256, label=f"{label} file"
    ):
        raise ValueError(f"{label} file SHA-256 differs")
    value = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("report_sha256") != _require_sha256(
        expected_report_sha256, label=f"{label} report"
    ):
        raise ValueError(f"{label} report SHA-256 differs")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _domain_sha256(payload: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(payload)).hexdigest()


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    )


class _AuthenticatedFiniteTailProvider(Gemma3L3L4CorrectionProvider):
    """Single-use truth-leaking D320+K correction for one held prompt."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "rank",
        "fold_artifact_sha256",
        "model_inputs_sha256",
        "bridge_binding_sha256",
        "prefix_artifact_sha256",
        "base_h4_sha256",
        "support_mask_sha256",
        "correction_sha256",
        "_support",
        "_correction",
        "_used",
    )

    def __init__(
        self,
        *,
        rank: int,
        fold_artifact_sha256: str,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        if type(rank) is not int or rank < 1 or rank > _D_RANK:
            raise ValueError("finite tail provider rank must be in [1, 320]")
        if (
            not isinstance(base_h4, Tensor)
            or base_h4.ndim != 3
            or base_h4.shape[-1] != _WIDTH
            or not base_h4.is_floating_point()
            or not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not isinstance(correction, Tensor)
            or correction.shape != base_h4.shape
            or not correction.is_floating_point()
        ):
            raise ValueError("finite tail provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").contiguous()
        delta = correction.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if (
            not bool(torch.isfinite(delta).all())
            or bool((delta[~support] != 0).any())
        ):
            raise ValueError("finite tail correction escapes support")
        self.site = _H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.rank = rank
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="finite tail fold artifact"
        )
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="finite tail model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="finite tail bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="finite tail prefix"
        )
        self.base_h4_sha256 = _runtime_tensor_sha256(base_h4)
        self._support = support
        self._correction = delta
        self.support_mask_sha256 = _runtime_tensor_sha256(support)
        self.correction_sha256 = _runtime_tensor_sha256(delta)
        self._used = False
        self.artifact_sha256 = self._computed_sha256()
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.complete_h4_finite_tail_provider.v1",
            "rank": self.rank,
            "site": self.site,
            "write_scope": self.write_scope,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": "P_D320_R_plus_P_training_fisher_prefix_of_I_minus_P_D320_R",
            "exact_residual_provider_used": False,
            "single_use": True,
            "truth_leaking_hypothesis_use_only": True,
            "serving_authorized": False,
        }

    def _computed_sha256(self) -> str:
        return _domain_sha256(self._payload(), domain=_PROVIDER_DOMAIN)

    @property
    def used(self) -> bool:
        return self._used

    def validate_integrity(self) -> None:
        if (
            self.site != _H4_SITE
            or self.write_scope != "complete_h4_causal_support"
            or _runtime_tensor_sha256(self._support) != self.support_mask_sha256
            or _runtime_tensor_sha256(self._correction) != self.correction_sha256
            or bool((self._correction[~self._support] != 0).any())
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise RuntimeError("finite tail provider payload drifted")

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("finite tail provider cannot be reused")
        prefix.validate_integrity()
        if (
            prefix.artifact_sha256 != self.prefix_artifact_sha256
            or prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or _runtime_tensor_sha256(realized_state) != self.base_h4_sha256
            or _runtime_tensor_sha256(
                prefix.complete_h4_causal_support_mask()
                .detach()
                .to(device="cpu")
                .contiguous()
            )
            != self.support_mask_sha256
        ):
            raise RuntimeError("finite tail provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(slots=True)
class _LiveEndpointTrace:
    example: object = field(repr=False)
    example_id: str
    family_id: str
    model_inputs_sha256: str
    supervised_indices_sha256: str
    supervised_targets_sha256: str
    endpoint_indices_sha256: str
    endpoint_targets_sha256: str
    prefix: Gemma3L3L4OnePassPrefix = field(repr=False)
    base_x4_sha256: str
    base_h4: Tensor = field(repr=False)
    native_h4: Tensor = field(repr=False)
    support_indices: Tensor = field(repr=False)
    selected_by_ledger: Mapping[str, Tensor] = field(repr=False)
    endpoint: CompleteH4TailEndpointExample = field(repr=False)
    native_token_nll: Tensor = field(repr=False)
    d320_token_nll: Tensor = field(repr=False)
    native_ordinary_token_nll: Tensor = field(repr=False)
    d320_ordinary_token_nll: Tensor = field(repr=False)
    native_logits_sha256: str
    endpoint_vjp_artifact_sha256: str
    endpoint_execution_artifact_sha256: str
    endpoint_provider_artifact_sha256: str
    backward_call_count: int
    maximum_future_gradient_abs: float
    future_gradient_nonzero_count: int
    causality_receipt_sha256: str


def _target_grid(
    model_inputs: Mapping[str, Tensor],
    indices: Tensor,
    targets: Tensor,
) -> Tensor:
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("tail diagnostic requires a batch-one input grid")
    result = torch.full_like(input_ids, -100, dtype=torch.int64)
    result[0, indices.to(result.device)] = targets.to(result.device)
    return result


def _selected_token_nll(logits: Tensor, indices: Tensor, targets: Tensor) -> Tensor:
    selected = frozen._select_sequence_rows(logits, indices)
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    return F.cross_entropy(
        selected,
        targets.to(selected.device),
        reduction="none",
    ).detach().to(device="cpu", dtype=torch.float64).contiguous()


def _collect_endpoint_traces(
    *,
    context: object,
    basis: Tensor,
    basis_binding: Mapping[str, str],
    transfer_receipts: Mapping[str, Mapping[str, object]],
) -> tuple[list[_LiveEndpointTrace], dict[str, int]]:
    adapter = getattr(context, "adapter")
    bridge = getattr(context, "bridge")
    tokenize = getattr(context, "tokenize")
    examples = tuple(getattr(context, "examples"))
    bridge.validate_integrity()
    bridge_sha256 = _require_sha256(
        bridge.bridge_binding_sha256, label="tail diagnostic bridge"
    )
    basis_contract = _ValidatedRank320BasisContract.build(basis, basis_binding)
    traces: list[_LiveEndpointTrace] = []
    resources = {
        "base_forward_count": 0,
        "native_forward_count": 0,
        "endpoint_token_vjp_forward_count": 0,
        "endpoint_token_vjp_backward_call_count": 0,
        "ordinary_supervised_token_count": 0,
        "endpoint_support_supervised_token_count": 0,
        "complete_h4_support_row_count": 0,
        "graph_core_row_count": 0,
        "causal_tail_row_count": 0,
        "graph_core_supervised_token_count": 0,
        "causal_tail_supervised_token_count": 0,
    }
    for example in sorted(examples, key=lambda value: value.example_id):
        example_id = _identifier(example.example_id, label="example_id")
        family_id = _identifier(example.family_id, label="family_id")
        prior = transfer_receipts.get(example_id)
        if prior is None:
            raise ValueError("pinned transfer omitted an A16 prompt receipt")
        model_inputs, indices, targets = _retokenize(tokenize, example)
        model_inputs_sha256 = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        indices_sha256 = _runtime_tensor_sha256(indices)
        targets_sha256 = _runtime_tensor_sha256(targets)
        if (
            prior.get("family_id") != family_id
            or prior.get("supervised_indices_sha256") != indices_sha256
            or prior.get("supervised_targets_sha256") != targets_sha256
        ):
            raise RuntimeError("endpoint supervision differs from pinned transfer")
        base = bridge.execute(adapter, model_inputs)
        resources["base_forward_count"] += 1
        if getattr(base, "model_forward_count", None) != 1:
            raise RuntimeError("base endpoint execution forward count differs")
        prefix = base.prefix
        prefix.validate_integrity()
        base_h4 = base.candidate_h4
        if (
            prior.get("model_inputs_sha256") != model_inputs_sha256
            or prior.get("bridge_binding_sha256") != bridge_sha256
            or prior.get("prefix_artifact_sha256") != prefix.artifact_sha256
            or prior.get("base_candidate_h4_sha256")
            != _runtime_tensor_sha256(base_h4)
            or prior.get("base_candidate_x4_sha256")
            != _runtime_tensor_sha256(base.candidate_x4)
        ):
            raise RuntimeError("endpoint base identity differs from pinned transfer")
        source_logits, native_h4, native_positions, native_valid = _native_boundary(
            adapter, model_inputs
        )
        resources["native_forward_count"] += 1
        if (
            native_h4.shape != base_h4.shape
            or not _bitwise_equal(native_positions, prefix.logical_positions)
            or not _bitwise_equal(native_valid, prefix.valid_target_mask)
        ):
            raise RuntimeError("native/base endpoint boundary differs")
        source_logits_sha256 = _runtime_tensor_sha256(source_logits)
        if (
            source_logits_sha256 != prior.get("source_logits_sha256")
            or _runtime_tensor_sha256(native_h4)
            != prior.get("native_h4_sha256")
        ):
            raise RuntimeError("native boundary differs from pinned transfer")
        support = prefix.complete_h4_causal_support_mask().detach().to(device="cpu")
        core = prefix.target_affected_mask.detach().to(device="cpu")
        support_indices = torch.nonzero(support[0], as_tuple=False).flatten().to(
            dtype=torch.int64
        )
        positions_cpu = indices.detach().to(device="cpu")
        support_supervised = support[0].index_select(0, positions_cpu)
        core_supervised = core[0].index_select(0, positions_cpu)
        endpoint_selection = torch.nonzero(
            support_supervised, as_tuple=False
        ).flatten().to(dtype=torch.int64)
        endpoint_indices = indices.index_select(
            0, endpoint_selection.to(indices.device)
        )
        endpoint_targets = targets.index_select(
            0, endpoint_selection.to(targets.device)
        )
        selected_by_ledger = {
            "ordinary": torch.arange(indices.numel(), dtype=torch.int64),
            "complete_h4_support": endpoint_selection,
            "graph_core": torch.nonzero(
                core_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
            "causal_tail": torch.nonzero(
                support_supervised & ~core_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
        }
        support_mask_sha256 = _runtime_tensor_sha256(support)
        if (
            prior.get("complete_h4_support_mask_sha256")
            != support_mask_sha256
            or prior.get("complete_h4_support_rows") != int(support.sum())
            or prior.get("graph_core_rows") != int(core.sum())
            or prior.get("causal_tail_rows") != int((support & ~core).sum())
        ):
            raise RuntimeError("endpoint support differs from pinned transfer")
        provider = AuthenticatedCompleteH4TransferProvider(
            role="rank320_projection",
            model_inputs_sha256=model_inputs_sha256,
            bridge_binding_sha256=bridge_sha256,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base_h4,
            native_h4=native_h4,
            basis_contract=basis_contract,
            support_mask=support,
        )
        token_vjp = bridge.execute_h4_token_nll_vjps(
            adapter,
            model_inputs,
            targets=_target_grid(model_inputs, endpoint_indices, endpoint_targets),
            vjp_chunk_size=_VJP_CHUNK_SIZE,
            h4_head=provider,
        )
        token_vjp.validate_integrity()
        provider.validate_integrity()
        if not provider.used:
            raise RuntimeError("authenticated D320 endpoint provider was not consumed")
        resources["endpoint_token_vjp_forward_count"] += 1
        resources["endpoint_token_vjp_backward_call_count"] += (
            token_vjp.backward_call_count
        )
        resources["ordinary_supervised_token_count"] += int(indices.numel())
        resources["endpoint_support_supervised_token_count"] += int(
            endpoint_indices.numel()
        )
        resources["complete_h4_support_row_count"] += int(support.sum())
        resources["graph_core_row_count"] += int(core.sum())
        resources["causal_tail_row_count"] += int((support & ~core).sum())
        resources["graph_core_supervised_token_count"] += int(
            selected_by_ledger["graph_core"].numel()
        )
        resources["causal_tail_supervised_token_count"] += int(
            selected_by_ledger["causal_tail"].numel()
        )
        expected_grid = torch.stack(
            (torch.zeros_like(endpoint_indices), endpoint_indices), dim=1
        ).to(
            device=token_vjp.supervised_indices.device, dtype=torch.int64
        )
        if not torch.equal(token_vjp.supervised_indices, expected_grid):
            raise RuntimeError("endpoint token VJP order differs")
        projected_provider = prior.get("projected_provider")
        if not isinstance(projected_provider, Mapping):
            raise ValueError("pinned transfer projected provider receipt differs")
        endpoint_execution = token_vjp.execution
        if (
            endpoint_execution.model_inputs_sha256 != model_inputs_sha256
            or endpoint_execution.bridge_binding_sha256 != bridge_sha256
            or endpoint_execution.prefix.artifact_sha256
            != prefix.artifact_sha256
            or endpoint_execution.h4_head_sha256 != provider.artifact_sha256
            or _runtime_tensor_sha256(endpoint_execution.candidate_x4)
            != prior.get("base_candidate_x4_sha256")
            or provider.artifact_sha256
            != projected_provider.get("artifact_sha256")
            or _runtime_tensor_sha256(endpoint_execution.candidate_h4)
            != prior.get("projected_h4_sha256")
            or _runtime_tensor_sha256(endpoint_execution.logits)
            != prior.get("projected_logits_sha256")
        ):
            raise RuntimeError(
                "endpoint token VJP does not reproduce pinned D320 transfer"
            )
        native_token_nll = _selected_token_nll(
            source_logits, endpoint_indices, endpoint_targets
        )
        native_ordinary_token_nll = _selected_token_nll(
            source_logits, indices, targets
        )
        d320_token_nll = token_vjp.token_losses.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        selected_d320_nll = _selected_token_nll(
            token_vjp.execution.logits, endpoint_indices, endpoint_targets
        )
        d320_ordinary_token_nll = _selected_token_nll(
            token_vjp.execution.logits, indices, targets
        )
        if not torch.allclose(
            d320_token_nll, selected_d320_nll, rtol=0.0, atol=1.0e-6
        ):
            raise RuntimeError("endpoint token VJP loss authority differs")
        base_cpu = base_h4.detach().to(device="cpu", dtype=torch.float64)
        native_cpu = native_h4.detach().to(device="cpu", dtype=torch.float64)
        residual_rows = (
            native_cpu[0].index_select(0, support_indices)
            - base_cpu[0].index_select(0, support_indices)
        ).contiguous()
        gradient_rows = (
            token_vjp.h4_gradients.detach()
            .to(device="cpu", dtype=torch.float64)[:, 0]
            .index_select(1, support_indices)
            .contiguous()
        )
        projected_rows = ((residual_rows @ basis.T) @ basis).contiguous()
        expected_candidate = base_h4.detach().to(device="cpu").clone()
        support_on_base = support_indices.to(base_h4.device)
        expected_candidate[0].index_copy_(
            0,
            support_indices,
            (
                base_h4.detach()[0]
                .index_select(0, support_on_base)
                .to(device="cpu", dtype=torch.float64)
                + projected_rows
            ).to(dtype=base_h4.dtype),
        )
        if not _bitwise_equal(
            endpoint_execution.candidate_h4.detach().to(device="cpu"),
            expected_candidate,
        ):
            raise RuntimeError("endpoint candidate H4 is not base plus D320")
        logical_cpu = prefix.logical_positions.detach().to(device="cpu")
        supervised_logical = logical_cpu[0].index_select(
            0, endpoint_indices.detach().to(device="cpu")
        )
        support_logical = logical_cpu[0].index_select(0, support_indices)
        maximum_future = 0.0
        future_nonzero = 0
        for token_index in range(int(endpoint_indices.numel())):
            later = support_logical > supervised_logical[token_index]
            if bool(later.any()):
                future = gradient_rows[token_index, later]
                maximum_future = max(maximum_future, float(future.abs().max()))
                future_nonzero += int((future != 0).sum())
        causality_payload = {
            "example_id": example_id,
            "supervised_token_count": int(endpoint_indices.numel()),
            "support_row_count": int(support_indices.numel()),
            "maximum_future_gradient_abs_hex": maximum_future.hex(),
            "future_gradient_nonzero_count": future_nonzero,
            "token_vjp_artifact_sha256": token_vjp.artifact_sha256,
        }
        causality_receipt = _domain_sha256(
            causality_payload,
            domain=b"fisher-graph:complete-h4-tail-token-causality:v1\0",
        )
        if maximum_future != 0.0 or future_nonzero != 0:
            raise RuntimeError("endpoint token VJP leaks into logically later H4 rows")
        endpoint = CompleteH4TailEndpointExample(
            example_id=example_id,
            family_id=family_id,
            residual_rows=residual_rows,
            token_h4_gradients=gradient_rows,
            compensation_target=(native_token_nll - d320_token_nll).contiguous(),
        )
        traces.append(
            _LiveEndpointTrace(
                example=example,
                example_id=example_id,
                family_id=family_id,
                model_inputs_sha256=model_inputs_sha256,
                supervised_indices_sha256=_runtime_tensor_sha256(indices),
                supervised_targets_sha256=_runtime_tensor_sha256(targets),
                endpoint_indices_sha256=_runtime_tensor_sha256(endpoint_indices),
                endpoint_targets_sha256=_runtime_tensor_sha256(endpoint_targets),
                prefix=prefix,
                base_x4_sha256=_runtime_tensor_sha256(base.candidate_x4),
                base_h4=base_h4.detach().clone().contiguous(),
                native_h4=native_h4.detach().to(device="cpu").contiguous(),
                support_indices=support_indices,
                selected_by_ledger={
                    key: value.clone().contiguous()
                    for key, value in selected_by_ledger.items()
                },
                endpoint=endpoint,
                native_token_nll=native_token_nll,
                d320_token_nll=d320_token_nll,
                native_ordinary_token_nll=native_ordinary_token_nll,
                d320_ordinary_token_nll=d320_ordinary_token_nll,
                native_logits_sha256=source_logits_sha256,
                endpoint_vjp_artifact_sha256=token_vjp.artifact_sha256,
                endpoint_execution_artifact_sha256=(
                    token_vjp.execution.artifact_sha256
                ),
                endpoint_provider_artifact_sha256=provider.artifact_sha256,
                backward_call_count=token_vjp.backward_call_count,
                maximum_future_gradient_abs=maximum_future,
                future_gradient_nonzero_count=future_nonzero,
                causality_receipt_sha256=causality_receipt,
            )
        )
        del base, source_logits, native_h4, token_vjp, model_inputs
    expected_accounting = {
        "ordinary_supervised_token_count": _EXPECTED_ORDINARY_TOKENS,
        "endpoint_support_supervised_token_count": _EXPECTED_SUPPORT_TOKENS,
        "complete_h4_support_row_count": _EXPECTED_SUPPORT_ROWS,
        "graph_core_row_count": _EXPECTED_GRAPH_CORE_ROWS,
        "causal_tail_row_count": _EXPECTED_CAUSAL_TAIL_ROWS,
        "graph_core_supervised_token_count": _EXPECTED_GRAPH_CORE_TOKENS,
        "causal_tail_supervised_token_count": _EXPECTED_CAUSAL_TAIL_TOKENS,
    }
    if any(resources[key] != value for key, value in expected_accounting.items()):
        raise RuntimeError("endpoint support/token accounting differs")
    return traces, resources


def _finite_observations(
    *,
    context: object,
    traces: Sequence[_LiveEndpointTrace],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    ranks: tuple[int, ...] = _TAIL_RANKS,
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
]:
    ranks = _validated_tail_ranks(ranks)
    adapter = getattr(context, "adapter")
    bridge = getattr(context, "bridge")
    tokenize = getattr(context, "tokenize")
    observations: list[dict[str, object]] = []
    forward_by_rank = {rank: 0 for rank in ranks}
    ledgers = (
        "ordinary",
        "complete_h4_support",
        "graph_core",
        "causal_tail",
    )
    manifests = {
        ledger: {
            trace.example_id: trace.family_id
            for trace in traces
            if trace.selected_by_ledger[ledger].numel() > 0
        }
        for ledger in ledgers
    }
    fidelity_accumulators = {
        rank: {
            ledger: SourceAuthoritativeShadowFidelityAccumulator(
                manifests[ledger], gates=ESTABLISHED_SHADOW_FIDELITY_GATES
            )
            for ledger in ledgers
        }
        for rank in ranks
    }
    geometry_traces: list[object] = []
    executed_rows_by_rank: dict[int, dict[str, Tensor]] = {
        rank: {} for rank in ranks
    }
    for trace in traces:
        geometry_traces.append(
            SimpleNamespace(
                example=trace.example,
                fit_sequence=CompleteH4ProjectionFitSequence(
                    example_id=trace.example_id,
                    family_id=trace.family_id,
                    residual_rows=trace.endpoint.residual_rows,
                ),
                support_indices=trace.support_indices,
                graph_core_rows=(
                    trace.prefix.target_affected_mask.detach()
                    .to(device="cpu")[0]
                    .index_select(0, trace.support_indices)
                ),
            )
        )
    finite_native_forward_count = 0
    for trace in traces:
        fit = fits[trace.family_id]
        scores = complete_h4_tail_gate_scores(
            trace.endpoint, fit.ordered_basis_rows()
        )
        residual = trace.endpoint.residual_rows
        supported_rows = ((residual @ basis.T) @ basis).contiguous()
        tail_rows = project_complete_h4_tail_rows(residual, basis)
        model_inputs, indices, targets = _retokenize(tokenize, trace.example)
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            != trace.model_inputs_sha256
            or _runtime_tensor_sha256(indices) != trace.supervised_indices_sha256
            or _runtime_tensor_sha256(targets) != trace.supervised_targets_sha256
        ):
            raise RuntimeError("finite ladder retokenization drifted")
        source_logits, finite_native_h4, finite_native_positions, finite_native_valid = (
            _native_boundary(adapter, model_inputs)
        )
        finite_native_forward_count += 1
        if (
            _runtime_tensor_sha256(source_logits) != trace.native_logits_sha256
            or not _bitwise_equal(
                finite_native_h4.detach().to(device="cpu"), trace.native_h4
            )
            or not _bitwise_equal(
                finite_native_positions, trace.prefix.logical_positions
            )
            or not _bitwise_equal(
                finite_native_valid, trace.prefix.valid_target_mask
            )
        ):
            raise RuntimeError("finite ladder native boundary drifted")
        source_selected = frozen._select_sequence_rows(source_logits, indices)
        support = trace.prefix.complete_h4_causal_support_mask().detach().to(
            device="cpu"
        )
        for rank in ranks:
            tail_prefix = project_complete_h4_tail_prefix(
                tail_rows, fit, rank=rank
            )
            correction_rows = (supported_rows + tail_prefix).contiguous()
            correction = torch.zeros(
                trace.base_h4.shape, dtype=torch.float64, device="cpu"
            )
            correction[0].index_copy_(0, trace.support_indices, correction_rows)
            provider = _AuthenticatedFiniteTailProvider(
                rank=rank,
                fold_artifact_sha256=fit.artifact_sha256,
                model_inputs_sha256=trace.model_inputs_sha256,
                bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
                prefix_artifact_sha256=trace.prefix.artifact_sha256,
                base_h4=trace.base_h4,
                support_mask=support,
                correction=correction,
            )
            execution = bridge.execute(adapter, model_inputs, h4_head=provider)
            forward_by_rank[rank] += 1
            provider.validate_integrity()
            if (
                getattr(execution, "model_forward_count", None) != 1
                or not provider.used
                or execution.model_inputs_sha256 != trace.model_inputs_sha256
                or execution.bridge_binding_sha256
                != trace.prefix.bridge_binding_sha256
                or execution.prefix.artifact_sha256
                != trace.prefix.artifact_sha256
                or execution.h4_head_sha256 != provider.artifact_sha256
                or _runtime_tensor_sha256(execution.candidate_x4)
                != trace.base_x4_sha256
            ):
                raise RuntimeError("finite ladder execution count/binding differs")
            expected_h4 = trace.base_h4.clone()
            support_on_live = trace.support_indices.to(expected_h4.device)
            expected_h4[0].index_copy_(
                0,
                support_on_live,
                (
                    trace.base_h4[0]
                    .index_select(0, support_on_live)
                    .to(dtype=torch.float64)
                    + correction_rows.to(trace.base_h4.device)
                ).to(dtype=trace.base_h4.dtype),
            )
            if not _bitwise_equal(
                execution.candidate_h4.detach(), expected_h4
            ):
                raise RuntimeError("finite ladder H4 is not base plus D320+K")
            candidate_nll = _selected_token_nll(execution.logits, indices, targets)
            endpoint_selection = trace.selected_by_ledger[
                "complete_h4_support"
            ]
            candidate_endpoint_nll = _selected_token_nll(
                execution.logits,
                indices.index_select(0, endpoint_selection.to(indices.device)),
                targets.index_select(0, endpoint_selection.to(targets.device)),
            )
            candidate_selected = frozen._select_sequence_rows(
                execution.logits, indices
            )
            for ledger, selected in trace.selected_by_ledger.items():
                if selected.numel() == 0:
                    continue
                fidelity_accumulators[rank][ledger].add(
                    ShadowFidelityExample(
                        example_id=trace.example_id,
                        family_id=trace.family_id,
                        source_logits=source_selected.index_select(
                            0, selected.to(source_selected.device)
                        ),
                        candidate_logits=candidate_selected.index_select(
                            0, selected.to(candidate_selected.device)
                        ),
                        targets=targets.index_select(
                            0, selected.to(targets.device)
                        ),
                    )
                )
            actual_rows = (
                execution.candidate_h4.detach().to(
                    device="cpu", dtype=torch.float64
                )[0].index_select(0, trace.support_indices)
                - trace.base_h4.to(device="cpu", dtype=torch.float64)[0].index_select(
                    0, trace.support_indices
                )
            ).contiguous()
            executed_rows_by_rank[rank][trace.example_id] = actual_rows
            prediction = scores[:, :rank].sum(dim=1)
            target = trace.endpoint.compensation_target
            reconstruction_error = float(
                (tail_prefix - tail_rows).abs().max()
            ) if rank == fit.rank else None
            observation = {
                    "example_id": trace.example_id,
                    "family_id": trace.family_id,
                    "rank": rank,
                    "fold_artifact_sha256": fit.artifact_sha256,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "execution_artifact_sha256": execution.artifact_sha256,
                    "token_score_matrix_sha256": _runtime_tensor_sha256(scores),
                    "native_mean_nll": float(trace.native_token_nll.mean()),
                    "d320_mean_nll": float(trace.d320_token_nll.mean()),
                    "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
                    "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
                    "endpoint_baseline_mse": float(target.square().mean()),
                    "endpoint_prediction_mse": float(
                        (prediction - target).square().mean()
                    ),
                    "candidate_h4_bitwise_native": _bitwise_equal(
                        execution.candidate_h4.detach().to(device="cpu"),
                        trace.native_h4,
                    ),
                    "candidate_logits_bitwise_native": (
                        _runtime_tensor_sha256(execution.logits)
                        == trace.native_logits_sha256
                    ),
                    "full_tail_reconstruction_max_abs_error": reconstruction_error,
                    "exact_residual_provider_used": False,
                    "executed_correction_rows_sha256": _runtime_tensor_sha256(
                        actual_rows
                    ),
                }
            observation["observation_sha256"] = _domain_sha256(
                observation,
                domain=b"fisher-graph:complete-h4-tail-finite-observation:v1\0",
            )
            observations.append(observation)
            del (
                execution,
                candidate_nll,
                candidate_endpoint_nll,
                provider,
                correction,
                candidate_selected,
            )
        del (
            model_inputs,
            source_logits,
            source_selected,
            finite_native_h4,
            finite_native_positions,
            finite_native_valid,
        )
    behavioral = {
        rank: {
            ledger: fidelity_accumulators[rank][ledger].finalize()
            for ledger in ledgers
        }
        for rank in ranks
    }
    geometry = {
        rank: ladder._geometry_with_examples(
            geometry_traces,
            executed_rows_by_rank[rank],
            candidate_semantics=(
                f"actual_cast_once_d320_plus_training_only_fisher_tail_k{rank}"
            ),
        )
        for rank in ranks
    }
    return observations, {
        "finite_forward_count": sum(forward_by_rank.values()),
        "finite_native_forward_count": finite_native_forward_count,
        **{f"finite_rank_{rank}_forward_count": count for rank, count in forward_by_rank.items()},
    }, behavioral, geometry


def _family_macro(values: Sequence[dict[str, object]], key: str) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for value in values:
        by_family[str(value["family_id"])].append(float(value[key]))
    return math.fsum(
        math.fsum(rows) / len(rows) for rows in by_family.values()
    ) / len(by_family)


def _finite_observation_set_sha256(
    observations: Sequence[Mapping[str, object]],
    *,
    expected_example_count: int = _EXPECTED_EXAMPLES,
    ranks: tuple[int, ...] = _TAIL_RANKS,
) -> str:
    """Authenticate the complete example-by-rank finite observation grid."""

    fixed_ranks = _validated_tail_ranks(ranks)
    expected_pairs = expected_example_count * len(fixed_ranks)
    if len(observations) != expected_pairs:
        raise ValueError("finite observation count differs")
    identities: set[tuple[str, int]] = set()
    receipts: list[str] = []
    for raw in observations:
        row = dict(raw)
        receipt = row.pop("observation_sha256", None)
        example_id = _identifier(row.get("example_id"), label="finite example_id")
        rank = row.get("rank")
        if type(rank) is not int or rank not in fixed_ranks:
            raise ValueError("finite observation rank differs")
        identity = (example_id, rank)
        if identity in identities:
            raise ValueError("finite observation grid has a duplicate")
        identities.add(identity)
        expected_receipt = _domain_sha256(
            row,
            domain=b"fisher-graph:complete-h4-tail-finite-observation:v1\0",
        )
        if receipt != expected_receipt:
            raise RuntimeError("finite observation receipt drifted")
        receipts.append(expected_receipt)
    observed_ranks = {rank for _example, rank in identities}
    examples = {example for example, _rank in identities}
    if observed_ranks != set(fixed_ranks) or len(examples) != expected_example_count:
        raise ValueError("finite observation grid is incomplete")
    return _domain_sha256(
        tuple(receipts),
        domain=b"fisher-graph:complete-h4-tail-finite-observation-set:v1\0",
    )


def _summarize_observations(
    observations: Sequence[dict[str, object]],
    *,
    ranks: tuple[int, ...] = _TAIL_RANKS,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    ranks = _validated_tail_ranks(ranks)
    arms: list[dict[str, object]] = []
    for rank in ranks:
        selected = tuple(row for row in observations if row["rank"] == rank)
        before_mse = _family_macro(selected, "endpoint_baseline_mse")
        after_mse = _family_macro(selected, "endpoint_prediction_mse")
        native = _family_macro(selected, "native_mean_nll")
        d320 = _family_macro(selected, "d320_mean_nll")
        candidate = _family_macro(selected, "candidate_mean_nll")
        family_before: dict[str, list[float]] = defaultdict(list)
        family_after: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            family = str(row["family_id"])
            family_before[family].append(
                abs(float(row["d320_mean_nll"]) - float(row["native_mean_nll"]))
            )
            family_after[family].append(
                abs(float(row["candidate_mean_nll"]) - float(row["native_mean_nll"]))
            )
        family_improvements = tuple(
            1.0
            - (math.fsum(family_after[family]) / len(family_after[family]))
            / max(
                math.fsum(family_before[family]) / len(family_before[family]),
                torch.finfo(torch.float64).tiny,
            )
            for family in sorted(family_before)
        )
        before_gap = math.fsum(
            math.fsum(family_before[family]) / len(family_before[family])
            for family in sorted(family_before)
        ) / len(family_before)
        after_gap = math.fsum(
            math.fsum(family_after[family]) / len(family_after[family])
            for family in sorted(family_after)
        ) / len(family_after)
        arms.append(
            {
                "tail_rank": rank,
                "family_macro_endpoint_rmse_before": math.sqrt(before_mse),
                "family_macro_endpoint_rmse_after": math.sqrt(after_mse),
                "family_macro_endpoint_relative_rmse_improvement": (
                    1.0
                    - math.sqrt(after_mse)
                    / max(
                        math.sqrt(before_mse),
                        torch.finfo(torch.float64).tiny,
                    )
                ),
                "family_macro_native_mean_nll": native,
                "family_macro_d320_mean_nll": d320,
                "family_macro_candidate_mean_nll": candidate,
                "family_macro_absolute_nll_gap_before": before_gap,
                "family_macro_absolute_nll_gap_after": after_gap,
                "family_macro_relative_absolute_nll_gap_improvement": (
                    1.0 - after_gap / max(before_gap, torch.finfo(torch.float64).tiny)
                ),
                "family_win_count": sum(value > 0.0 for value in family_improvements),
                "worst_family_relative_nll_gap_improvement": min(family_improvements),
                "every_prompt_h4_bitwise_native": all(
                    bool(row["candidate_h4_bitwise_native"]) for row in selected
                ),
                "every_prompt_logits_bitwise_native": all(
                    bool(row["candidate_logits_bitwise_native"]) for row in selected
                ),
                "maximum_full_tail_reconstruction_abs_error": max(
                    float(row["full_tail_reconstruction_max_abs_error"] or 0.0)
                    for row in selected
                ),
            }
        )
    k64 = next(row for row in arms if row["tail_rank"] == 64)
    k320 = next(row for row in arms if row["tail_rank"] == 320)
    gates = {
        "k64_endpoint_relative_rmse_improvement_at_least_2pct": (
            float(k64["family_macro_endpoint_relative_rmse_improvement"]) >= 0.02
        ),
        "k64_finite_nll_gap_improvement_at_least_2pct": (
            float(k64["family_macro_relative_absolute_nll_gap_improvement"]) >= 0.02
        ),
        "k64_family_win_count_at_least_6_of_8": int(k64["family_win_count"]) >= 6,
        "k64_worst_family_regression_at_most_2pct": (
            float(k64["worst_family_relative_nll_gap_improvement"]) >= -0.02
        ),
        "k320_full_tail_reconstruction_at_most_1e_minus_9": (
            float(k320["maximum_full_tail_reconstruction_abs_error"]) <= 1.0e-9
        ),
        "k320_every_prompt_h4_bitwise_native": bool(
            k320["every_prompt_h4_bitwise_native"]
        ),
        "k320_every_prompt_logits_bitwise_native": bool(
            k320["every_prompt_logits_bitwise_native"]
        ),
    }
    return arms, gates


def _build_resource_accounting(
    traces: Sequence[_LiveEndpointTrace],
    *,
    endpoint_resources: Mapping[str, int],
    finite_resources: Mapping[str, int],
    ranks: tuple[int, ...] = _TAIL_RANKS,
) -> dict[str, object]:
    """Return exact executed counts and declared CPU-analysis MAC subtotals."""

    ranks = _validated_tail_ranks(ranks)
    expected_backward = sum(
        (trace.endpoint.supervised_tokens + _VJP_CHUNK_SIZE - 1)
        // _VJP_CHUNK_SIZE
        for trace in traces
    )
    if (
        endpoint_resources["endpoint_token_vjp_backward_call_count"]
        != expected_backward
    ):
        raise RuntimeError("endpoint VJP backward accounting differs")
    support_rows = sum(
        int(trace.endpoint.residual_rows.shape[0]) for trace in traces
    )
    one_pass_gate_score_macs = sum(
        (
            int(trace.endpoint.residual_rows.shape[0]) * _D_RANK * _WIDTH
            + trace.endpoint.supervised_tokens
            * int(trace.endpoint.residual_rows.shape[0])
            * _D_RANK
            * _WIDTH
            + trace.endpoint.supervised_tokens
            * int(trace.endpoint.residual_rows.shape[0])
            * _D_RANK
        )
        for trace in traces
    )
    training_order_gate_score_macs = 7 * one_pass_gate_score_macs
    held_score_macs = one_pass_gate_score_macs
    gate_score_macs = training_order_gate_score_macs + held_score_macs
    endpoint_d320_macs = 4 * support_rows * _D_RANK * _WIDTH
    finite_preparation_macs = 4 * support_rows * _D_RANK * _WIDTH
    finite_tail_macs_by_rank = {
        str(rank): 2 * support_rows * rank * _WIDTH for rank in ranks
    }
    return {
        **endpoint_resources,
        **finite_resources,
        "vjp_chunk_size": _VJP_CHUNK_SIZE,
        "expected_backward_call_count_from_supervised_tokens": expected_backward,
        "total_model_forward_count": (
            endpoint_resources["base_forward_count"]
            + endpoint_resources["native_forward_count"]
            + endpoint_resources["endpoint_token_vjp_forward_count"]
            + finite_resources["finite_forward_count"]
            + finite_resources["finite_native_forward_count"]
        ),
        "complete_h4_support_row_count": support_rows,
        "peak_simultaneously_retained_full_vocabulary_tensor_count": 3,
        "peak_full_vocabulary_residency_reason": (
            "base_execution_logits_plus_native_logits_plus_endpoint_vjp_"
            "logits_during_collection"
        ),
        "endpoint_d320_projection_logical_macs": endpoint_d320_macs,
        "endpoint_d320_projection_logical_macs_semantics": (
            "one_provider_projection_plus_one_independent_expected_candidate_"
            "projection"
        ),
        "finite_preparation_d320_and_tail_projection_logical_macs": (
            finite_preparation_macs
        ),
        "finite_tail_prefix_projection_logical_macs_by_rank": (
            finite_tail_macs_by_rank
        ),
        "finite_total_projection_logical_macs": (
            finite_preparation_macs + sum(finite_tail_macs_by_rank.values())
        ),
        "endpoint_token_fisher_gate_score_logical_macs": gate_score_macs,
        "training_only_fisher_order_gate_score_logical_macs": (
            training_order_gate_score_macs
        ),
        "held_endpoint_gate_score_logical_macs": held_score_macs,
        "endpoint_token_fisher_prompt_fold_contraction_count": (
            len(traces) * _EXPECTED_FAMILIES
        ),
        "wide_complement_fit_count": _EXPECTED_FAMILIES,
        "canonical_complement_construction_count": _EXPECTED_FAMILIES,
        "canonical_complement_coordinate_axis_two_pass_mgs": (
            "cpu_float64_analysis_setup_explicitly_excluded_from_logical_mac_"
            "subtotals"
        ),
        "wide_complement_training_tail_projection_count": (
            len(traces) * (_EXPECTED_FAMILIES - 1)
        ),
        "wide_complement_training_tail_projection_logical_macs": (
            2
            * (_EXPECTED_FAMILIES - 1)
            * support_rows
            * _D_RANK
            * _WIDTH
        ),
        "wide_complement_training_coordinate_projection_logical_macs": (
            (_EXPECTED_FAMILIES - 1)
            * support_rows
            * _D_RANK
            * _WIDTH
        ),
        "wide_complement_full_tail_reconstruction_logical_macs": (
            2
            * (_EXPECTED_FAMILIES - 1)
            * support_rows
            * _D_RANK
            * _WIDTH
        ),
        "wide_complement_ambient_basis_lift_logical_macs": (
            _EXPECTED_FAMILIES * _D_RANK * _D_RANK * _WIDTH
        ),
        "wide_complement_rank": _D_RANK,
        "fisher_order_sort_count": _EXPECTED_FAMILIES,
        "fisher_order_keys_per_sort": _D_RANK,
        "wide_covariance_complete_frame_eigh_and_sort_setup": (
            "cpu_float64_analysis_only_explicitly_excluded_from_reported_"
            "logical_mac_subtotals"
        ),
        "serving_learned_parameter_count": (
            "not_applicable_no_serving_artifact"
        ),
        "serving_logical_macs_per_token": (
            "not_applicable_no_serving_artifact"
        ),
    }


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if (
        destination.is_absolute()
        or destination.suffix != ".json"
        or not destination.parts
        or destination.parts[0] != ".local-runs"
        or ".." in destination.parts
    ):
        raise ValueError("tail token-Fisher output must be repo-relative JSON under .local-runs")
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(report, domain=_REPORT_DOMAIN)
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic(
    *,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned A16 LOFO endpoint/Fisher finite tail ladder."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite tail token-Fisher report")
    materialization = _load_pinned_report(
        materialization_report_path,
        expected_file_sha256=MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=MATERIALIZATION_REPORT_SHA256,
        label="rank320 materialization",
    )
    transfer = _load_pinned_report(
        transfer_report_path,
        expected_file_sha256=TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=TRANSFER_REPORT_SHA256,
        label="rank320 transfer",
    )
    comparison = frozen._mapping(transfer.get("comparison"), label="transfer comparison")
    if comparison.get("pass_pattern") != "11100000":
        raise ValueError("pinned transfer pass pattern differs")
    raw_transfer_receipts = transfer.get("prompt_receipts")
    if not isinstance(raw_transfer_receipts, list):
        raise ValueError("pinned transfer prompt receipts differ")
    transfer_receipts: dict[str, Mapping[str, object]] = {}
    for raw_receipt in raw_transfer_receipts:
        if not isinstance(raw_receipt, Mapping):
            raise ValueError("pinned transfer prompt receipt differs")
        receipt_example_id = _identifier(
            raw_receipt.get("example_id"), label="transfer receipt example_id"
        )
        if receipt_example_id in transfer_receipts:
            raise ValueError("pinned transfer has duplicate prompt receipts")
        transfer_receipts[receipt_example_id] = raw_receipt
    if len(transfer_receipts) != _EXPECTED_EXAMPLES:
        raise ValueError("pinned transfer prompt receipt count differs")
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = _collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != _EXPECTED_EXAMPLES or len(families) != _EXPECTED_FAMILIES:
            raise RuntimeError("A16 endpoint panel shape differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        (
            observations,
            finite_resources,
            behavioral_by_rank,
            geometry_by_rank,
        ) = _finite_observations(
            context=context, traces=traces, basis=basis, fits=fits
        )
        context.validate_immutable_inputs()
    finally:
        context.close()
    arms, secondary_gates = _summarize_observations(observations)
    finite_observation_set_sha256 = _finite_observation_set_sha256(
        observations
    )
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    )
    fidelity_and_geometry_pass_by_rank = {
        rank: (
            bool(geometry_by_rank[rank]["gates"]["passed"])
            and all(
                bool(behavioral_by_rank[rank][ledger]["gates"]["passed"])
                for ledger in (
                    "ordinary",
                    "complete_h4_support",
                    "graph_core",
                    "causal_tail",
                )
            )
        )
        for rank in _TAIL_RANKS
    }
    passing_bounded_ranks = tuple(
        rank
        for rank in _TAIL_RANKS
        if rank <= 64 and fidelity_and_geometry_pass_by_rank[rank]
    )
    smallest_passing_bounded_rank = (
        None if not passing_bounded_ranks else min(passing_bounded_ranks)
    )
    k320_arm = next(row for row in arms if row["tail_rank"] == 320)
    primary_gates = {
        "all_endpoint_token_vjps_have_zero_future_gradient": causality_passed,
        "at_least_one_k_le_64_clears_all_established_fidelity_and_geometry_gates": (
            smallest_passing_bounded_rank is not None
        ),
        "k320_full_fitted_span_clears_established_fidelity_and_geometry_gates": (
            fidelity_and_geometry_pass_by_rank[320]
        ),
        "k320_every_prompt_h4_bitwise_native": bool(
            k320_arm["every_prompt_h4_bitwise_native"]
        ),
        "k320_every_prompt_logits_bitwise_native": bool(
            k320_arm["every_prompt_logits_bitwise_native"]
        ),
    }
    resources = _build_resource_accounting(
        traces,
        endpoint_resources=endpoint_resources,
        finite_resources=finite_resources,
    )
    prompt_receipts = tuple(
        {
            **trace.endpoint.metadata(),
            "model_inputs_sha256": trace.model_inputs_sha256,
            "base_x4_sha256": trace.base_x4_sha256,
            "supervised_indices_sha256": trace.supervised_indices_sha256,
            "supervised_targets_sha256": trace.supervised_targets_sha256,
            "endpoint_support_indices_sha256": trace.endpoint_indices_sha256,
            "endpoint_support_targets_sha256": trace.endpoint_targets_sha256,
            "endpoint_support_supervised_token_count": (
                trace.endpoint.supervised_tokens
            ),
            "native_logits_sha256": trace.native_logits_sha256,
            "endpoint_vjp_artifact_sha256": trace.endpoint_vjp_artifact_sha256,
            "endpoint_execution_artifact_sha256": trace.endpoint_execution_artifact_sha256,
            "endpoint_provider_artifact_sha256": trace.endpoint_provider_artifact_sha256,
            "backward_call_count": trace.backward_call_count,
            "compensation_target_semantics": (
                "native_token_nll_minus_d320_endpoint_token_nll"
            ),
            "compensation_target_sign_used_in_fisher_q2_ordering": False,
            "maximum_future_gradient_abs": trace.maximum_future_gradient_abs,
            "future_gradient_nonzero_count": trace.future_gradient_nonzero_count,
            "causality_receipt_sha256": trace.causality_receipt_sha256,
        }
        for trace in traces
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_hypothesis_use_only",
            "split": (
                "whole_family_leave_one_out_for_tail_basis_and_fisher_order_only"
            ),
            "frozen_d320_was_fit_on_all_a16_families": True,
            "end_to_end_candidate_is_family_disjoint": False,
            "frozen_supported_basis_rank": _D_RANK,
            "tail_width": _D_RANK,
            "tail_ranks": _TAIL_RANKS,
            "tail_definition": "E=(I-P_D320)(native_H4-base_graph_H4)",
            "endpoint": "actual_cast_once_D320_one_pass_graph_execution",
            "basis_fit": "training_family_equal_unweighted_tail_covariance_full_complement",
            "order": "training_only_family_prompt_token_equal_endpoint_vjp_square",
            "compensation_target_semantics": (
                "native_token_nll_minus_d320_endpoint_token_nll"
            ),
            "compensation_target_sign_used_in_fisher_q2_ordering": False,
            "prompt_mean_preliminary_order_reused": False,
            "held_residual_or_gradient_used_for_fit_or_order": False,
            "endpoint_token_fisher_ledger": "complete_h4_support_803",
            "finite_shadow_ledgers": {
                "ordinary": 931,
                "complete_h4_support": 803,
                "graph_core": 790,
                "causal_tail": 13,
            },
            "finite_arm": "P_D320_R_plus_P_first_K_training_fisher_ordered_tail_E",
            "k320_uses_full_fitted_complement_span": True,
            "k320_exact_residual_provider_substitution": False,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": TRANSFER_REPORT_SHA256,
            "transfer_pass_pattern": "11100000",
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": prompt_receipts,
        "finite_ladder": arms,
        "established_behavioral_fidelity_by_rank": {
            str(rank): behavioral_by_rank[rank] for rank in _TAIL_RANKS
        },
        "executed_cast_once_geometry_by_rank": {
            str(rank): geometry_by_rank[rank] for rank in _TAIL_RANKS
        },
        "fidelity_and_geometry_pass_by_rank": {
            str(rank): fidelity_and_geometry_pass_by_rank[rank]
            for rank in _TAIL_RANKS
        },
        "smallest_tail_rank_at_most_64_clearing_established_gates": (
            smallest_passing_bounded_rank
        ),
        "finite_observation_receipts": tuple(observations),
        "finite_observation_set_sha256": finite_observation_set_sha256,
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "secondary_first_order_gate_results": tuple(
            sorted(secondary_gates.items())
        ),
        "passed": all(primary_gates.values()),
        "classification": (
            "tail_endpoint_fisher_finite_ladder_supported"
            if all(primary_gates.values())
            else "tail_endpoint_fisher_finite_ladder_not_supported"
        ),
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "tail_basis_and_fisher_order_family_disjoint_only": True,
            "frozen_d320_contains_same_a_held_family_information": True,
            "end_to_end_candidate_family_disjoint": False,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "next_rung_only_if_supported": "freeze_recipe_then_fresh_family_disjoint_confirmation",
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "contains_token_score_matrices": False,
            "contains_basis_coefficients": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
            "artifact_must_remain_outside_git": True,
        },
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run A16 held-family endpoint token-Fisher finite tail ladder."
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--transfer-report", type=Path, default=DEFAULT_TRANSFER_REPORT)
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic(
        materialization_report_path=args.materialization_report,
        transfer_report_path=args.transfer_report,
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":
    main()
