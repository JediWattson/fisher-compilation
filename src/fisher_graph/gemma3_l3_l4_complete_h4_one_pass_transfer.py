"""Fixed-D320 one-pass complete-H4 carrier-transfer oracle.

This rung asks a deliberately narrow question: does the exact same
tail-informed 320-dimensional subspace that passed the frozen complete-H4
capacity factorial still describe the residual left by the deployable
one-pass graph carrier?

The basis is fixed before this run.  For each A16 prompt the runner executes
one base one-pass H4 VJP, one native source pass, one exact complete-H4
ceiling, and one D320-projected correction.  The exact and projected
corrections are delivered by single-use, integrity-bound providers over the
complete causal H4 support.  Only scalar metrics and hashes may be published.
This remains a same-A truth-leaking transfer oracle; it is not a learned
coordinate generator, a serving artifact, or a compression/speed result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
    ImmutableFloat64Matrix,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_sidecar import (
    DEFAULT_OUTPUT as DEFAULT_BASIS_SIDECAR,
    PARENT_FILE_SHA256,
    PARENT_REPORT_SHA256,
    CompleteH4Rank320BasisSidecar,
    load_complete_h4_rank320_basis_sidecar,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    DEFAULT_OUTPUT as DEFAULT_BASIS_MATERIALIZATION_REPORT,
    load_complete_h4_rank320_basis_materialization_report,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_V3_DOMAIN,
    _COMPLETE_H4_PROJECTION_DEFINITION,
    _COMPLETE_H4_TAIL_INFORMED_PROJECTION_CONSTRUCTION,
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
    "DEFAULT_BASIS_SIDECAR",
    "DEFAULT_BASIS_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "AuthenticatedCompleteH4TransferProvider",
    "run_gemma3_l3_l4_complete_h4_one_pass_transfer",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-"
    "tail-informed-rank320-one-pass-transfer-a-fit16-dev-v1.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_complete_h4_tail_informed_rank320_"
    "one_pass_transfer_development"
)
_FORMAT_VERSION = 1
_ROLE = "reused_calibration_a_truth_leaking_one_pass_h4_transfer_oracle"
_ARM_ID = "tail_informed.rank320.one_pass_transfer"
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-one-pass-transfer:v1\0"
)
_PROVIDER_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-transfer-provider:v1\0"
)
_PROMPT_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-transfer-prompt:v1\0"
)
_H4_SITE = "layer.4.output"
_PROJECTION_RANK = 320
_RESIDUAL_WIDTH = 640
_EXPECTED_PROMPTS = 16
_EXPECTED_SUPPORT_ROWS = 819
_EXPECTED_GRAPH_CORE_ROWS = 802
_EXPECTED_CAUSAL_TAIL_ROWS = 17
_EXPECTED_LEDGER_TOKENS = {
    "ordinary": 931,
    "complete_h4_support": 803,
    "graph_core": 790,
    "causal_tail": 13,
}
_PROJECTION_MACS = 2 * _EXPECTED_SUPPORT_ROWS * _PROJECTION_RANK * _RESIDUAL_WIDTH

_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer_state": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_tensors": False,
    "contains_gradient_tensors": False,
    "contains_basis_coefficients": False,
    "contains_scalar_metrics": True,
    "truth_leaking_same_a_transfer_oracle": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        isinstance(left, Tensor)
        and isinstance(right, Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _domain_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


_VALIDATED_BASIS_TOKEN = object()


class _ValidatedRank320BasisContract:
    """One-time D320 validation receipt shared by all prompt providers."""

    __slots__ = ("_basis", "binding", "artifact_sha256")

    def __init__(
        self,
        *,
        basis: Tensor,
        binding: Mapping[str, str],
        _token: object,
    ) -> None:
        if _token is not _VALIDATED_BASIS_TOKEN:
            raise TypeError("validated D320 contracts require the closed factory")
        _validate_basis_contract(basis, binding)
        self._basis = ImmutableFloat64Matrix.from_tensor(
            basis,
            label="validated fixed rank320 basis",
        )
        self.binding = dict(binding)
        self.artifact_sha256 = _domain_sha256(
            {
                "schema": "fisher_graph.validated_rank320_basis_contract",
                "format_version": 1,
                "binding": self.binding,
                "projection_rank": _PROJECTION_RANK,
                "residual_width": _RESIDUAL_WIDTH,
                "orthonormality_checked_once": True,
            },
            domain=b"fisher-graph:validated-rank320-basis-contract:v1\0",
        )
        self.validate_integrity()

    @classmethod
    def build(
        cls,
        basis: Tensor,
        binding: Mapping[str, str],
    ) -> "_ValidatedRank320BasisContract":
        return cls(
            basis=basis,
            binding=binding,
            _token=_VALIDATED_BASIS_TOKEN,
        )

    def validate_integrity(self) -> None:
        rebuilt = ImmutableFloat64Matrix(
            row_count=self._basis.row_count,
            width=self._basis.width,
            _little_endian_bytes=self._basis._little_endian_bytes,
        )
        if (
            rebuilt.shape != (_PROJECTION_RANK, _RESIDUAL_WIDTH)
            or rebuilt.matrix_sha256 != self._basis.matrix_sha256
            or rebuilt.matrix_sha256
            != self.binding.get("basis_matrix_sha256")
            or self.artifact_sha256
            != _domain_sha256(
                {
                    "schema": "fisher_graph.validated_rank320_basis_contract",
                    "format_version": 1,
                    "binding": self.binding,
                    "projection_rank": _PROJECTION_RANK,
                    "residual_width": _RESIDUAL_WIDTH,
                    "orthonormality_checked_once": True,
                },
                domain=(
                    b"fisher-graph:validated-rank320-basis-contract:v1\0"
                ),
            )
        ):
            raise RuntimeError("validated D320 basis receipt drifted")

    def basis_tensor(self) -> Tensor:
        self.validate_integrity()
        rebuilt = ImmutableFloat64Matrix(
            row_count=self._basis.row_count,
            width=self._basis.width,
            _little_endian_bytes=self._basis._little_endian_bytes,
        )
        return rebuilt.to_tensor()


class AuthenticatedCompleteH4TransferProvider(Gemma3L3L4CorrectionProvider):
    """One-use exact or projected correction bound to one realized H4.

    The provider is intentionally prompt-local.  Its artifact authenticates
    the model input, bridge, prefix, base/native boundaries, frozen D320
    basis, causal support, and correction bytes.  Reusing it for another
    execution (even an identical one) raises rather than silently weakening
    the one-provider/one-forward receipt.
    """

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "role",
        "model_inputs_sha256",
        "bridge_binding_sha256",
        "prefix_artifact_sha256",
        "base_h4_sha256",
        "native_h4_sha256",
        "basis_sidecar_artifact_sha256",
        "basis_matrix_sha256",
        "basis_runtime_tensor_sha256",
        "projection_basis_artifact_sha256",
        "support_mask_sha256",
        "correction_sha256",
        "_correction",
        "_support_mask",
        "_used",
    )

    def __init__(
        self,
        *,
        role: Literal["exact_ceiling", "rank320_projection"],
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        native_h4: Tensor,
        basis_contract: _ValidatedRank320BasisContract,
        support_mask: Tensor,
    ) -> None:
        if role not in ("exact_ceiling", "rank320_projection"):
            raise ValueError("transfer provider role differs")
        if (
            not isinstance(base_h4, Tensor)
            or not isinstance(native_h4, Tensor)
            or base_h4.shape != native_h4.shape
            or base_h4.ndim != 3
            or base_h4.shape[-1] != _RESIDUAL_WIDTH
            or not base_h4.is_floating_point()
            or native_h4.dtype != base_h4.dtype
            or native_h4.device != base_h4.device
            or not bool(torch.isfinite(base_h4).all())
            or not bool(torch.isfinite(native_h4).all())
        ):
            raise ValueError("transfer provider H4 boundaries differ")
        if (
            not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not bool(support_mask.any())
        ):
            raise ValueError("transfer provider support mask differs")
        support_cpu = support_mask.detach().to(device="cpu").contiguous()
        if not isinstance(basis_contract, _ValidatedRank320BasisContract):
            raise TypeError("transfer provider requires a validated D320 contract")
        basis_contract.validate_integrity()
        basis_binding = basis_contract.binding
        required_basis = (
            "logical_artifact_sha256",
            "basis_matrix_sha256",
            "runtime_tensor_sha256",
            "projection_basis_artifact_sha256",
        )
        if set(basis_binding) != set(required_basis):
            raise ValueError("transfer provider basis binding keyset differs")

        base_live_cpu = base_h4.detach().to(device="cpu").contiguous()
        native_live_cpu = native_h4.detach().to(device="cpu").contiguous()
        if not _bitwise_equal(
            base_live_cpu[~support_cpu],
            native_live_cpu[~support_cpu],
        ):
            raise ValueError("native/base H4 differs outside causal support")
        exact_residual = torch.zeros(
            base_h4.shape,
            device="cpu",
            dtype=torch.float64,
        )
        exact_residual[support_cpu] = (
            native_live_cpu[support_cpu].to(torch.float64)
            - base_live_cpu[support_cpu].to(torch.float64)
        )
        derived_correction = exact_residual
        if role == "rank320_projection":
            basis = basis_contract.basis_tensor()
            residual_rows = exact_residual[support_cpu]
            derived_correction = torch.zeros_like(exact_residual)
            derived_correction[support_cpu] = (
                (residual_rows @ basis.T) @ basis
            )

        self.site = _H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.role = role
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256,
            label="transfer provider model inputs",
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256,
            label="transfer provider bridge binding",
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256,
            label="transfer provider prefix artifact",
        )
        self.base_h4_sha256 = _runtime_tensor_sha256(base_h4)
        self.native_h4_sha256 = _runtime_tensor_sha256(native_h4)
        self.basis_sidecar_artifact_sha256 = _require_sha256(
            basis_binding["logical_artifact_sha256"],
            label="rank320 sidecar artifact",
        )
        self.basis_matrix_sha256 = _require_sha256(
            basis_binding["basis_matrix_sha256"],
            label="rank320 basis matrix",
        )
        self.basis_runtime_tensor_sha256 = _require_sha256(
            basis_binding["runtime_tensor_sha256"],
            label="rank320 runtime tensor",
        )
        self.projection_basis_artifact_sha256 = _require_sha256(
            basis_binding["projection_basis_artifact_sha256"],
            label="rank320 projection basis artifact",
        )
        self._correction = derived_correction.detach().clone().contiguous()
        self._support_mask = support_cpu.clone().contiguous()
        self.support_mask_sha256 = _runtime_tensor_sha256(self._support_mask)
        self.correction_sha256 = _runtime_tensor_sha256(self._correction)
        self._used = False
        self.artifact_sha256 = self._computed_artifact_sha256()
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.authenticated_complete_h4_transfer_provider",
            "format_version": 1,
            "role": self.role,
            "site": self.site,
            "write_scope": self.write_scope,
            "arm_id": _ARM_ID,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "native_h4_sha256": self.native_h4_sha256,
            "basis_sidecar_artifact_sha256": (
                self.basis_sidecar_artifact_sha256
            ),
            "basis_matrix_sha256": self.basis_matrix_sha256,
            "basis_runtime_tensor_sha256": self.basis_runtime_tensor_sha256,
            "projection_basis_artifact_sha256": (
                self.projection_basis_artifact_sha256
            ),
            "projection_rank": _PROJECTION_RANK,
            "residual_width": _RESIDUAL_WIDTH,
            "projection_ordering": (
                COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
            ),
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_dtype": "cpu_float64",
            "single_use": True,
            "truth_leaking_same_a": True,
            "serving_authorized": False,
        }

    def _computed_artifact_sha256(self) -> str:
        return _domain_sha256(self._payload(), domain=_PROVIDER_DOMAIN)

    @property
    def used(self) -> bool:
        return self._used

    def validate_integrity(self) -> None:
        if (
            self.site != _H4_SITE
            or self.write_scope != "complete_h4_causal_support"
            or type(self._used) is not bool
            or _runtime_tensor_sha256(self._support_mask)
            != self.support_mask_sha256
            or _runtime_tensor_sha256(self._correction)
            != self.correction_sha256
            or bool((self._correction[~self._support_mask] != 0).any())
            or self._computed_artifact_sha256()
            != _require_sha256(
                self.artifact_sha256,
                label="transfer provider artifact",
            )
        ):
            raise RuntimeError("complete-H4 transfer provider payload drifted")

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("complete-H4 transfer provider cannot be reused")
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
            raise RuntimeError(
                "complete-H4 transfer provider reached another execution"
            )
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class _ExpectedAccounting:
    prompts: int = _EXPECTED_PROMPTS
    support_rows: int = _EXPECTED_SUPPORT_ROWS
    graph_core_rows: int = _EXPECTED_GRAPH_CORE_ROWS
    causal_tail_rows: int = _EXPECTED_CAUSAL_TAIL_ROWS
    ledger_tokens: Mapping[str, int] = field(
        default_factory=lambda: dict(_EXPECTED_LEDGER_TOKENS)
    )

    def __post_init__(self) -> None:
        if (
            type(self.prompts) is not int
            or self.prompts <= 0
            or type(self.support_rows) is not int
            or self.support_rows <= 0
            or type(self.graph_core_rows) is not int
            or self.graph_core_rows <= 0
            or type(self.causal_tail_rows) is not int
            or self.causal_tail_rows <= 0
            or self.graph_core_rows + self.causal_tail_rows
            != self.support_rows
            or set(self.ledger_tokens) != set(_EXPECTED_LEDGER_TOKENS)
            or any(type(value) is not int or value <= 0 for value in self.ledger_tokens.values())
        ):
            raise ValueError("one-pass transfer expected accounting differs")


@dataclass(slots=True)
class _BaseTrace:
    example: object
    prompt_sha256: str
    model_inputs_sha256: str
    supervised_indices_sha256: str
    supervised_targets_sha256: str
    supervised_token_count: int
    prefix: Gemma3L3L4OnePassPrefix
    candidate_x4: Tensor = field(repr=False)
    candidate_h4: Tensor = field(repr=False)
    gradient_rows: Tensor = field(repr=False)
    support_indices: Tensor = field(repr=False)
    graph_core_rows: Tensor = field(repr=False)
    selected_by_ledger: Mapping[str, Tensor] = field(repr=False)
    base_execution_artifact_sha256: str
    base_h4_gradient_sha256: str


def _basis_binding(
    sidecar: CompleteH4Rank320BasisSidecar,
    *,
    expected_logical_artifact_sha256: str,
) -> tuple[Tensor, dict[str, str]]:
    if not isinstance(sidecar, CompleteH4Rank320BasisSidecar):
        raise TypeError("basis sidecar has the wrong type")
    sidecar.validate_integrity()
    expected = _require_sha256(
        expected_logical_artifact_sha256,
        label="expected rank320 sidecar artifact",
    )
    if sidecar.artifact_sha256 != expected:
        raise ValueError("rank320 sidecar logical artifact differs")
    basis = sidecar.basis_tensor()
    return basis, {
        "logical_artifact_sha256": sidecar.artifact_sha256,
        "basis_matrix_sha256": sidecar.basis_matrix_sha256,
        "runtime_tensor_sha256": sidecar.runtime_tensor_sha256,
        "projection_basis_artifact_sha256": (
            sidecar.projection_basis_artifact_sha256
        ),
    }


def _load_committed_basis(
    *,
    materialization_report_path: Path | str,
    expected_materialization_report_sha256: str,
    basis_sidecar_path: Path | str | None,
) -> tuple[Tensor, dict[str, str], dict[str, object]]:
    expected_report = _require_sha256(
        expected_materialization_report_sha256,
        label="expected rank320 materialization report",
    )
    marker = load_complete_h4_rank320_basis_materialization_report(
        materialization_report_path
    )
    if marker.get("report_sha256") != expected_report:
        raise ValueError("rank320 materialization report SHA-256 differs")
    artifact = frozen._mapping(
        marker.get("artifact"),
        label="rank320 materialization artifact",
    )
    receipt = frozen._mapping(
        artifact.get("basis_sidecar"),
        label="rank320 materialization sidecar receipt",
    )
    marker_sidecar = receipt.get("file")
    if not isinstance(marker_sidecar, str) or not marker_sidecar:
        raise ValueError("rank320 materialization marker omitted its sidecar")
    selected_sidecar = (
        Path(marker_sidecar)
        if basis_sidecar_path is None
        else Path(basis_sidecar_path)
    )
    marker_resolved = Path(marker_sidecar).absolute().resolve(strict=False)
    selected_resolved = selected_sidecar.absolute().resolve(strict=False)
    if selected_resolved != marker_resolved:
        raise ValueError(
            "explicit rank320 sidecar does not match the commit marker"
        )
    sidecar = load_complete_h4_rank320_basis_sidecar(selected_sidecar)
    logical_artifact = _require_sha256(
        receipt.get("logical_artifact_sha256"),
        label="materialized rank320 sidecar logical artifact",
    )
    basis, binding = _basis_binding(
        sidecar,
        expected_logical_artifact_sha256=logical_artifact,
    )
    marker_file_sha256 = _require_sha256(
        artifact.get("file_sha256"),
        label="rank320 materialization marker file",
    )
    sidecar_file_sha256 = _require_sha256(
        receipt.get("file_sha256"),
        label="rank320 materialized sidecar file",
    )
    if receipt.get("committable") is not False:
        raise ValueError("rank320 materialized sidecar committability differs")
    return basis, binding, {
        "materialization_report_file": str(
            Path(materialization_report_path)
        ),
        "materialization_report_file_sha256": marker_file_sha256,
        "materialization_report_sha256": expected_report,
        "basis_sidecar_file": str(marker_resolved),
        "basis_sidecar_file_sha256": sidecar_file_sha256,
        "basis_sidecar_file_bytes": receipt.get("file_bytes"),
        "basis_sidecar_logical_artifact_sha256": logical_artifact,
        "consumer_required_commit_marker_and_sidecar": True,
        "explicit_sidecar_override_matched_marker": (
            basis_sidecar_path is not None
        ),
    }


def _validate_basis_contract(
    basis: Tensor,
    binding: Mapping[str, str],
) -> None:
    required = {
        "logical_artifact_sha256",
        "basis_matrix_sha256",
        "runtime_tensor_sha256",
        "projection_basis_artifact_sha256",
    }
    if (
        not isinstance(basis, Tensor)
        or basis.shape != (_PROJECTION_RANK, _RESIDUAL_WIDTH)
        or basis.dtype != torch.float64
        or basis.device.type != "cpu"
        or not basis.is_contiguous()
        or basis.requires_grad
        or not bool(torch.isfinite(basis).all())
        or set(binding) != required
    ):
        raise ValueError("fixed rank320 basis contract differs")
    for name in required:
        _require_sha256(binding[name], label=f"rank320 {name}")
    runtime_sha256 = _runtime_tensor_sha256(basis)
    if (
        runtime_sha256 != binding["runtime_tensor_sha256"]
        or ImmutableFloat64Matrix.from_tensor(
            basis,
            label="fixed rank320 basis",
        ).matrix_sha256
        != binding["basis_matrix_sha256"]
    ):
        raise ValueError("fixed rank320 basis hash binding differs")
    gram = basis @ basis.T
    identity = torch.eye(_PROJECTION_RANK, dtype=torch.float64)
    orthonormal_error = float((gram - identity).abs().max())
    if (
        not math.isfinite(orthonormal_error)
        or orthonormal_error > 1.0e-10
    ):
        raise ValueError("fixed rank320 basis is not orthonormal")
    projection_artifact = _projection_basis_artifact_from_receipt(
        runtime_tensor_sha256=runtime_sha256,
        orthonormal_max_abs_error=orthonormal_error,
    )
    if projection_artifact != binding["projection_basis_artifact_sha256"]:
        raise ValueError("fixed rank320 basis hash binding differs")


def _projection_basis_artifact_from_receipt(
    *,
    runtime_tensor_sha256: str,
    orthonormal_max_abs_error: float,
) -> str:
    """Rebuild V3 identity from the already-computed one-time Gram receipt."""

    runtime_sha256 = _require_sha256(
        runtime_tensor_sha256,
        label="rank320 projection runtime tensor",
    )
    if (
        not math.isfinite(orthonormal_max_abs_error)
        or orthonormal_max_abs_error < 0.0
    ):
        raise ValueError("rank320 orthonormality receipt differs")
    payload = {
        "schema": "fisher_graph.gemma3_l3_l4_complete_h4_projection_basis",
        "format_version": 3,
        "projection_basis_sha256": runtime_sha256,
        "projection_rank": _PROJECTION_RANK,
        "projection_width": _RESIDUAL_WIDTH,
        "projection_ordering": (
            COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
        ),
        "projection_definition": _COMPLETE_H4_PROJECTION_DEFINITION,
        "orthonormal_max_abs_error": orthonormal_max_abs_error,
        "fit_weighting": "unweighted",
        "basis_construction": (
            _COMPLETE_H4_TAIL_INFORMED_PROJECTION_CONSTRUCTION
        ),
    }
    return hashlib.sha256(
        _COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_V3_DOMAIN
        + _canonical_json_bytes(payload)
    ).hexdigest()
def _mean_supervised_nll(
    supervised_indices: Tensor,
    supervised_targets: Tensor,
) -> Callable[[object], Tensor]:
    positions = supervised_indices.detach().clone().to(dtype=torch.int64)
    targets = supervised_targets.detach().clone().to(dtype=torch.int64)

    def objective(run: object) -> Tensor:
        logits = getattr(run, "logits", None)
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError("one-pass H4 VJP logits differ")
        selected = frozen._select_sequence_rows(logits, positions)
        if selected.shape[0] != targets.shape[0]:
            raise ValueError("one-pass H4 VJP supervision differs")
        live_targets = targets.to(selected.device)
        if selected.dtype in (torch.float16, torch.bfloat16):
            selected = selected.float()
        return F.cross_entropy(selected, live_targets, reduction="mean")

    return objective


def _native_boundary(
    adapter: object,
    model_inputs: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    forward = getattr(adapter, "forward", None)
    if not callable(forward):
        raise TypeError("adapter must expose forward")
    with torch.inference_mode():
        run = forward(
            model_inputs,
            capture_sites=(_H4_SITE,),
            interventions={},
        )
    logits = getattr(run, "logits", None)
    activations = getattr(run, "activations", None)
    sequence = getattr(run, "sequence", None)
    logical_positions = getattr(sequence, "logical_positions", None)
    valid_target_mask = getattr(sequence, "query_valid_mask", None)
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or not isinstance(activations, Mapping)
        or not isinstance(activations.get(_H4_SITE), Tensor)
        or not isinstance(logical_positions, Tensor)
        or not isinstance(valid_target_mask, Tensor)
    ):
        raise ValueError("native source boundary was not captured")
    h4 = activations[_H4_SITE]
    if (
        h4.ndim != 3
        or h4.shape[-1] != _RESIDUAL_WIDTH
        or logits.shape[:2] != h4.shape[:2]
        or not h4.is_floating_point()
        or not bool(torch.isfinite(h4).all())
        or not bool(torch.isfinite(logits).all())
        or logical_positions.shape != h4.shape[:2]
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or valid_target_mask.shape != h4.shape[:2]
        or valid_target_mask.dtype != torch.bool
    ):
        raise ValueError("native source boundary geometry differs")
    return (
        logits.detach(),
        h4.detach(),
        logical_positions.detach(),
        valid_target_mask.detach(),
    )


def _retokenize(
    tokenize: Callable[[object], tuple[Mapping[str, Tensor], Tensor, Tensor]],
    example: object,
) -> tuple[Mapping[str, Tensor], Tensor, Tensor]:
    model_inputs, supervised_indices, supervised_targets = tokenize(example)
    if (
        not isinstance(model_inputs, Mapping)
        or not isinstance(supervised_indices, Tensor)
        or supervised_indices.ndim != 1
        or supervised_indices.dtype != torch.int64
        or supervised_indices.numel() <= 0
        or not isinstance(supervised_targets, Tensor)
        or supervised_targets.shape != supervised_indices.shape
        or supervised_targets.dtype != torch.int64
    ):
        raise ValueError("one-pass transfer tokenizer output differs")
    return model_inputs, supervised_indices, supervised_targets


def _prefix_matches(
    expected: Gemma3L3L4OnePassPrefix,
    observed: Gemma3L3L4OnePassPrefix,
) -> bool:
    expected.validate_integrity()
    observed.validate_integrity()
    fields = (
        "source_modes",
        "clamped_y3",
        "predicted_target_modal_delta",
        "decoded_base_x4_delta",
        "logical_positions",
        "valid_target_mask",
        "source_eligible_mask",
        "target_affected_mask",
    )
    return (
        expected.artifact_sha256 == observed.artifact_sha256
        and expected.bridge_binding_sha256
        == observed.bridge_binding_sha256
        and all(
            _bitwise_equal(getattr(expected, name), getattr(observed, name))
            for name in fields
        )
    )


def _execution_matches_base(
    trace: _BaseTrace,
    execution: object,
) -> bool:
    prefix = getattr(execution, "prefix", None)
    candidate_x4 = getattr(execution, "candidate_x4", None)
    model_inputs_sha256 = getattr(execution, "model_inputs_sha256", None)
    bridge_binding_sha256 = getattr(execution, "bridge_binding_sha256", None)
    return (
        isinstance(prefix, Gemma3L3L4OnePassPrefix)
        and isinstance(candidate_x4, Tensor)
        and model_inputs_sha256 == trace.model_inputs_sha256
        and bridge_binding_sha256 == trace.prefix.bridge_binding_sha256
        and _prefix_matches(trace.prefix, prefix)
        and _bitwise_equal(trace.candidate_x4, candidate_x4)
    )


def _evaluate_fixed_rank320_transfer(
    *,
    examples: Sequence[object],
    tokenize: Callable[[object], tuple[Mapping[str, Tensor], Tensor, Tensor]],
    adapter: object,
    bridge: object,
    basis: Tensor,
    basis_binding: Mapping[str, str],
    expected: _ExpectedAccounting = _ExpectedAccounting(),
) -> dict[str, object]:
    """Execute the fixed one-pass transfer protocol over an authenticated panel."""

    basis_contract = _ValidatedRank320BasisContract.build(
        basis,
        basis_binding,
    )
    bridge_validate = getattr(bridge, "validate_integrity", None)
    if not callable(bridge_validate):
        raise TypeError("one-pass bridge must expose validate_integrity")
    bridge_validate()
    bridge_binding_sha256 = _require_sha256(
        getattr(bridge, "bridge_binding_sha256", None),
        label="one-pass bridge binding",
    )
    ordered = sorted(
        tuple(examples),
        key=lambda value: _identifier(
            getattr(value, "example_id", None),
            label="example_id",
        ),
    )
    if len(ordered) != expected.prompts:
        raise ValueError("one-pass transfer prompt count differs")
    identities: set[str] = set()
    base_traces: list[_BaseTrace] = []
    manifests: dict[str, dict[str, str]] = {
        name: {} for name in _EXPECTED_LEDGER_TOKENS
    }
    ledger_token_counts = {name: 0 for name in manifests}
    total_support_rows = 0
    total_core_rows = 0
    total_tail_rows = 0
    total_padding_rows = 0

    # First pass: one VJP per prompt.  Full-vocabulary base logits are dropped
    # immediately; only boundary tensors and scalar/hash receipts survive.
    for example in ordered:
        example_id = _identifier(
            getattr(example, "example_id", None), label="example_id"
        )
        family_id = _identifier(
            getattr(example, "family_id", None), label="family_id"
        )
        if example_id in identities:
            raise ValueError("one-pass transfer panel has duplicate examples")
        identities.add(example_id)
        model_inputs, supervised_indices, supervised_targets = _retokenize(
            tokenize, example
        )
        model_inputs_sha256 = gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs
        )
        execute_h4_vjp = getattr(bridge, "execute_h4_vjp", None)
        if not callable(execute_h4_vjp):
            raise TypeError("one-pass bridge must expose execute_h4_vjp")
        base, gradient = execute_h4_vjp(
            adapter,
            model_inputs,
            objective=_mean_supervised_nll(
                supervised_indices,
                supervised_targets,
            ),
        )
        prefix = getattr(base, "prefix", None)
        base_x4 = getattr(base, "candidate_x4", None)
        base_h4 = getattr(base, "candidate_h4", None)
        if (
            not isinstance(prefix, Gemma3L3L4OnePassPrefix)
            or not isinstance(base_x4, Tensor)
            or not isinstance(base_h4, Tensor)
            or not isinstance(gradient, Tensor)
            or base_h4.shape != gradient.shape
            or gradient.dtype != base_h4.dtype
            or gradient.device != base_h4.device
            or not bool(torch.isfinite(gradient).all())
            or base_h4.shape[-1] != _RESIDUAL_WIDTH
            or getattr(base, "model_inputs_sha256", None)
            != model_inputs_sha256
            or getattr(base, "bridge_binding_sha256", None)
            != bridge_binding_sha256
            or getattr(base, "model_forward_count", None) != 1
        ):
            raise RuntimeError("base one-pass H4 VJP binding differs")
        prefix.validate_integrity()
        support = prefix.complete_h4_causal_support_mask()[0].detach().to(
            device="cpu"
        )
        core = prefix.target_affected_mask[0].detach().to(device="cpu")
        valid = prefix.valid_target_mask[0].detach().to(device="cpu")
        if (
            bool((core & ~support).any())
            or bool((support & ~valid).any())
            or int(base_h4.shape[0]) != 1
        ):
            raise RuntimeError("one-pass complete-H4 support differs")
        support_indices = torch.nonzero(
            support, as_tuple=False
        ).flatten().to(dtype=torch.int64)
        graph_core_rows = core.index_select(0, support_indices)
        positions_cpu = supervised_indices.detach().to(device="cpu")
        if bool(
            (positions_cpu < 0).any()
            or (positions_cpu >= support.shape[0]).any()
        ):
            raise ValueError("supervised positions escape one-pass grid")
        support_supervised = support.index_select(0, positions_cpu)
        core_supervised = core.index_select(0, positions_cpu)
        selected = {
            "ordinary": torch.arange(
                positions_cpu.numel(), dtype=torch.int64
            ),
            "complete_h4_support": torch.nonzero(
                support_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
            "graph_core": torch.nonzero(
                core_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
            "causal_tail": torch.nonzero(
                support_supervised & ~core_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
        }
        for ledger, indices in selected.items():
            ledger_token_counts[ledger] += int(indices.numel())
            if indices.numel() > 0:
                manifests[ledger][example_id] = family_id
        device_support = support_indices.to(base_h4.device)
        gradient_rows = gradient[0].index_select(
            0, device_support
        ).detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(gradient_rows).all()):
            raise RuntimeError("base one-pass H4 VJP is nonfinite")
        base_traces.append(
            _BaseTrace(
                example=example,
                prompt_sha256=frozen._prompt_sha256(
                    _identifier(
                        getattr(example, "prompt", None),
                        label="prompt",
                    )
                ),
                model_inputs_sha256=model_inputs_sha256,
                supervised_indices_sha256=_runtime_tensor_sha256(
                    supervised_indices
                ),
                supervised_targets_sha256=_runtime_tensor_sha256(
                    supervised_targets
                ),
                supervised_token_count=int(supervised_indices.numel()),
                prefix=prefix,
                candidate_x4=base_x4.detach().clone(),
                candidate_h4=base_h4.detach().clone(),
                gradient_rows=gradient_rows,
                support_indices=support_indices,
                graph_core_rows=graph_core_rows,
                selected_by_ledger={
                    name: value.detach().clone().contiguous()
                    for name, value in selected.items()
                },
                base_execution_artifact_sha256=_require_sha256(
                    getattr(base, "artifact_sha256", None),
                    label="base one-pass execution artifact",
                ),
                base_h4_gradient_sha256=_runtime_tensor_sha256(gradient),
            )
        )
        total_support_rows += int(support.sum())
        total_core_rows += int(core.sum())
        total_tail_rows += int((support & ~core).sum())
        total_padding_rows += int((support & ~valid).sum())
        del base, gradient, model_inputs

    if (
        total_support_rows != expected.support_rows
        or total_core_rows != expected.graph_core_rows
        or total_tail_rows != expected.causal_tail_rows
        or total_padding_rows != 0
        or ledger_token_counts != dict(expected.ledger_tokens)
        or any(not manifest for manifest in manifests.values())
    ):
        raise RuntimeError("one-pass transfer support/token accounting differs")

    accumulators = {
        name: SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
        )
        for name, manifest in manifests.items()
    }
    geometry_traces: list[Any] = []
    executed_rows: dict[str, Tensor] = {}
    prompt_receipts: list[dict[str, object]] = []
    exact_h4_bitwise = True
    exact_logits_bitwise = True
    exact_h4_max_error = 0.0
    exact_logits_max_error = 0.0
    projection_write_rows = 0

    # Second pass: native, exact ceiling, and projected correction.  Exact
    # logits are released before projected logits are created, so at most two
    # full-vocabulary tensors are simultaneously resident.
    execute = getattr(bridge, "execute", None)
    if not callable(execute):
        raise TypeError("one-pass bridge must expose execute")
    for trace in base_traces:
        example = trace.example
        example_id = _identifier(
            getattr(example, "example_id", None), label="example_id"
        )
        family_id = _identifier(
            getattr(example, "family_id", None), label="family_id"
        )
        model_inputs, supervised_indices, supervised_targets = _retokenize(
            tokenize, example
        )
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            != trace.model_inputs_sha256
            or _runtime_tensor_sha256(supervised_indices)
            != trace.supervised_indices_sha256
            or _runtime_tensor_sha256(supervised_targets)
            != trace.supervised_targets_sha256
        ):
            raise RuntimeError("one-pass transfer retokenization drifted")
        (
            source_logits,
            native_h4,
            native_logical_positions,
            native_valid_target_mask,
        ) = _native_boundary(adapter, model_inputs)
        if native_h4.shape != trace.candidate_h4.shape:
            raise RuntimeError("native/base one-pass H4 geometry differs")
        if (
            not _bitwise_equal(
                native_logical_positions,
                trace.prefix.logical_positions,
            )
            or not _bitwise_equal(
                native_valid_target_mask,
                trace.prefix.valid_target_mask,
            )
        ):
            raise RuntimeError("native/base one-pass sequence grid differs")
        support = trace.prefix.complete_h4_causal_support_mask().detach().to(
            device="cpu"
        )
        support_on_h4 = support.to(native_h4.device)
        if not _bitwise_equal(
            native_h4[~support_on_h4],
            trace.candidate_h4.to(native_h4.device)[~support_on_h4],
        ):
            raise RuntimeError("native/base H4 differs outside causal support")
        base_cpu = trace.candidate_h4.detach().to(
            device="cpu", dtype=torch.float64
        )
        native_cpu = native_h4.detach().to(device="cpu", dtype=torch.float64)
        exact_provider = AuthenticatedCompleteH4TransferProvider(
            role="exact_ceiling",
            model_inputs_sha256=trace.model_inputs_sha256,
            bridge_binding_sha256=bridge_binding_sha256,
            prefix_artifact_sha256=trace.prefix.artifact_sha256,
            base_h4=trace.candidate_h4,
            native_h4=native_h4,
            basis_contract=basis_contract,
            support_mask=support,
        )
        exact = execute(
            adapter,
            model_inputs,
            h4_head=exact_provider,
        )
        exact_provider.validate_integrity()
        if (
            not exact_provider.used
            or not _execution_matches_base(trace, exact)
            or getattr(exact, "h4_head_sha256", None)
            != exact_provider.artifact_sha256
        ):
            raise RuntimeError("exact one-pass rerun binding differs")
        exact_h4 = getattr(exact, "candidate_h4", None)
        exact_logits = getattr(exact, "logits", None)
        if not isinstance(exact_h4, Tensor) or not isinstance(exact_logits, Tensor):
            raise RuntimeError("exact one-pass ceiling omitted tensors")
        h4_is_bitwise = _bitwise_equal(exact_h4, native_h4)
        logits_are_bitwise = _bitwise_equal(exact_logits, source_logits)
        h4_error = float(
            (exact_h4.to(torch.float64) - native_h4.to(torch.float64))
            .abs()
            .max()
        )
        logits_error = float(
            (exact_logits.to(torch.float64) - source_logits.to(torch.float64))
            .abs()
            .max()
        )
        if (
            not h4_is_bitwise
            or not logits_are_bitwise
            or h4_error != 0.0
            or logits_error != 0.0
        ):
            raise RuntimeError("exact one-pass H4 ceiling is not authoritative")
        exact_h4_bitwise &= h4_is_bitwise
        exact_logits_bitwise &= logits_are_bitwise
        exact_h4_max_error = max(exact_h4_max_error, h4_error)
        exact_logits_max_error = max(exact_logits_max_error, logits_error)
        exact_metadata = exact_provider.metadata()
        del exact, exact_h4, exact_logits

        support_indices = trace.support_indices
        residual_rows = (
            native_cpu[0].index_select(0, support_indices)
            - base_cpu[0].index_select(0, support_indices)
        ).contiguous()
        projected_provider = AuthenticatedCompleteH4TransferProvider(
            role="rank320_projection",
            model_inputs_sha256=trace.model_inputs_sha256,
            bridge_binding_sha256=bridge_binding_sha256,
            prefix_artifact_sha256=trace.prefix.artifact_sha256,
            base_h4=trace.candidate_h4,
            native_h4=native_h4,
            basis_contract=basis_contract,
            support_mask=support,
        )
        projected = execute(
            adapter,
            model_inputs,
            h4_head=projected_provider,
        )
        projected_provider.validate_integrity()
        if (
            not projected_provider.used
            or not _execution_matches_base(trace, projected)
            or getattr(projected, "h4_head_sha256", None)
            != projected_provider.artifact_sha256
        ):
            raise RuntimeError("projected one-pass rerun binding differs")
        projected_h4 = getattr(projected, "candidate_h4", None)
        projected_logits = getattr(projected, "logits", None)
        if (
            not isinstance(projected_h4, Tensor)
            or not isinstance(projected_logits, Tensor)
        ):
            raise RuntimeError("projected one-pass execution omitted tensors")
        actual_rows = (
            projected_h4[0]
            .index_select(0, support_indices.to(projected_h4.device))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            - trace.candidate_h4[0]
            .index_select(0, support_indices.to(trace.candidate_h4.device))
            .detach()
            .to(device="cpu", dtype=torch.float64)
        ).contiguous()
        executed_rows[example_id] = actual_rows
        fit_sequence = CompleteH4ProjectionFitSequence(
            example_id=example_id,
            family_id=family_id,
            residual_rows=residual_rows,
            gradient_rows=trace.gradient_rows,
        )
        geometry_traces.append(
            SimpleNamespace(
                example=example,
                fit_sequence=fit_sequence,
                support_indices=support_indices,
                graph_core_rows=trace.graph_core_rows,
            )
        )
        source_selected = frozen._select_sequence_rows(
            source_logits, supervised_indices
        )
        candidate_selected = frozen._select_sequence_rows(
            projected_logits, supervised_indices
        )
        for ledger, selected in trace.selected_by_ledger.items():
            if selected.numel() == 0:
                continue
            accumulators[ledger].add(
                ShadowFidelityExample(
                    example_id=example_id,
                    family_id=family_id,
                    source_logits=source_selected.index_select(
                        0, selected.to(source_selected.device)
                    ),
                    candidate_logits=candidate_selected.index_select(
                        0, selected.to(candidate_selected.device)
                    ),
                    targets=supervised_targets.index_select(
                        0, selected.to(supervised_targets.device)
                    ),
                )
            )
        projection_write_rows += int(support.sum())
        projected_metadata = projected_provider.metadata()
        receipt = {
            "example_id": example_id,
            "family_id": family_id,
            "prompt_sha256": trace.prompt_sha256,
            "model_inputs_sha256": trace.model_inputs_sha256,
            "supervised_indices_sha256": trace.supervised_indices_sha256,
            "supervised_targets_sha256": trace.supervised_targets_sha256,
            "supervised_token_count": trace.supervised_token_count,
            "bridge_binding_sha256": bridge_binding_sha256,
            "prefix_artifact_sha256": trace.prefix.artifact_sha256,
            "base_execution_artifact_sha256": (
                trace.base_execution_artifact_sha256
            ),
            "base_candidate_x4_sha256": _runtime_tensor_sha256(
                trace.candidate_x4
            ),
            "base_candidate_h4_sha256": _runtime_tensor_sha256(
                trace.candidate_h4
            ),
            "base_h4_gradient_sha256": trace.base_h4_gradient_sha256,
            "transfer_sequence": fit_sequence.metadata(),
            "native_h4_sha256": _runtime_tensor_sha256(native_h4),
            "source_logits_sha256": _runtime_tensor_sha256(source_logits),
            "complete_h4_support_mask_sha256": _runtime_tensor_sha256(
                support
            ),
            "exact_provider": exact_metadata,
            "projected_provider": projected_metadata,
            "projected_h4_sha256": _runtime_tensor_sha256(projected_h4),
            "projected_logits_sha256": _runtime_tensor_sha256(
                projected_logits
            ),
            "exact_h4_bitwise_authoritative": h4_is_bitwise,
            "exact_logits_bitwise_authoritative": logits_are_bitwise,
            "exact_h4_max_abs_error": h4_error,
            "exact_logits_max_abs_error": logits_error,
            "prefix_candidate_x4_base_h4_replay_matched": True,
            "complete_h4_support_rows": int(support.sum()),
            "graph_core_rows": int(trace.graph_core_rows.sum()),
            "causal_tail_rows": int((~trace.graph_core_rows).sum()),
        }
        prompt_receipts.append(
            {
                **receipt,
                "receipt_sha256": _domain_sha256(
                    receipt, domain=_PROMPT_RECEIPT_DOMAIN
                ),
            }
        )
        del (
            model_inputs,
            source_logits,
            native_h4,
            native_logical_positions,
            native_valid_target_mask,
            projected,
            projected_h4,
            projected_logits,
            source_selected,
            candidate_selected,
        )

    behavioral = {
        name: accumulator.finalize()
        for name, accumulator in accumulators.items()
    }
    geometry = ladder._geometry_with_examples(
        geometry_traces,
        executed_rows,
        candidate_semantics=(
            "actual_cast_once_rank320_correction_executed_by_one_pass_h4_head"
        ),
    )
    semantics = dict(geometry["semantics"])  # type: ignore[arg-type]
    semantics["source"] = "native_h4_minus_base_one_pass_candidate_h4"
    semantics["transfer_from_frozen_exact_x4_capacity_basis"] = True
    geometry["semantics"] = semantics
    exact_ceiling = {
        "expected_prompt_count": expected.prompts,
        "observed_prompt_count": len(prompt_receipts),
        "every_prompt_h4_bitwise_authoritative": exact_h4_bitwise,
        "every_prompt_logits_bitwise_authoritative": exact_logits_bitwise,
        "maximum_h4_absolute_error": exact_h4_max_error,
        "maximum_logits_absolute_error": exact_logits_max_error,
        "passed": (
            len(prompt_receipts) == expected.prompts
            and exact_h4_bitwise
            and exact_logits_bitwise
            and exact_h4_max_error == 0.0
            and exact_logits_max_error == 0.0
        ),
    }
    support_integrity = {
        "expected_complete_h4_support_rows": expected.support_rows,
        "observed_complete_h4_support_rows": total_support_rows,
        "graph_core_rows": total_core_rows,
        "causal_tail_rows": total_tail_rows,
        "projection_write_rows": projection_write_rows,
        "projection_padding_write_rows": total_padding_rows,
        "ledger_supervised_tokens": dict(ledger_token_counts),
        "prompt_receipt_count": len(prompt_receipts),
        "passed": (
            total_support_rows == expected.support_rows
            and total_core_rows == expected.graph_core_rows
            and total_tail_rows == expected.causal_tail_rows
            and projection_write_rows == expected.support_rows
            and total_padding_rows == 0
            and ledger_token_counts == dict(expected.ledger_tokens)
            and len(prompt_receipts) == expected.prompts
        ),
    }
    comparison = ladder.classify_projection_ladder_arm(
        fit_weighting="unweighted",
        rank=_PROJECTION_RANK,
        identity_validated=True,
        exact_h4_ceiling=exact_ceiling,
        support_integrity=support_integrity,
        boundary_geometry=geometry,
        ordinary_behavioral=behavioral["ordinary"],
        support_behavioral=behavioral["complete_h4_support"],
        graph_core_behavioral=behavioral["graph_core"],
        causal_tail_behavioral=behavioral["causal_tail"],
    )
    passed = comparison["later_lofo_fitting_authorized"] is True
    comparison = {
        **comparison,
        "arm_id": _ARM_ID,
        "basis_family": "tail_informed",
        "classification": (
            "fixed_rank320_one_pass_transfer_validated"
            if passed
            else "fixed_rank320_one_pass_transfer_insufficient"
        ),
        "success_authorizes": (
            "family_disjoint_learned_coordinates_on_fixed_one_pass_d320"
            if passed
            else None
        ),
        "later_lofo_fitting_authorized": False,
        "family_disjoint_learned_coordinate_development_authorized": passed,
    }
    projection_macs = (
        2 * total_support_rows * _PROJECTION_RANK * _RESIDUAL_WIDTH
    )
    return {
        "arm_id": _ARM_ID,
        "basis_binding": dict(basis_binding),
        "bridge_binding_sha256": bridge_binding_sha256,
        "exact_h4_ceiling": exact_ceiling,
        "support_integrity": support_integrity,
        "executed_cast_once_geometry": geometry,
        "behavioral_ledgers": behavioral,
        "comparison": comparison,
        "prompt_receipts": tuple(prompt_receipts),
        "resources": {
            "model_forward_count": 4 * expected.prompts,
            "backward_count": expected.prompts,
            "base_h4_vjp_forwards": expected.prompts,
            "native_source_forwards": expected.prompts,
            "exact_ceiling_forwards": expected.prompts,
            "rank320_projection_forwards": expected.prompts,
            "full_vocabulary_tensor_peak": 2,
            "projection_rank": _PROJECTION_RANK,
            "residual_width": _RESIDUAL_WIDTH,
            "projected_support_rows": total_support_rows,
            "logical_projection_macs": projection_macs,
            "logical_projection_macs_formula": "2 * rows * rank * width",
            "expected_live_a16_projection_macs": _PROJECTION_MACS,
            "transfer_contract_basis_orthonormality_check_macs": (
                _PROJECTION_RANK * _PROJECTION_RANK * _RESIDUAL_WIDTH
            ),
            "transfer_contract_basis_check_count": 1,
            "sidecar_deserialization_and_reauthentication_compute": (
                "setup_excluded_and_uninstrumented"
            ),
            "oracle_projection_macs_exclude_basis_setup_and_authentication": True,
        },
    }


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    parts = destination.parts
    if (
        destination.is_absolute()
        or destination.suffix != ".json"
        or len(parts) < 2
        or parts[0] != ".local-runs"
        or parts.count(".local-runs") != 1
        or ".." in parts
    ):
        raise ValueError(
            "one-pass transfer output must be JSON under .local-runs as a "
            "lexical repo-relative path without traversal or nesting"
        )
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(
            report, domain=_REPORT_DOMAIN
        )
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": frozen._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def _live_context_binding(context: object) -> dict[str, object]:
    parent = frozen._mapping(
        getattr(context, "parent", None),
        label="one-pass transfer parent factorial",
    )
    panel = frozen._mapping(
        getattr(context, "panel_receipt", None),
        label="one-pass transfer panel receipt",
    )
    lineage = frozen._mapping(
        parent.get("lineage"),
        label="one-pass transfer parent lineage",
    )
    fit_runtime = getattr(context, "fit_runtime", None)
    runtime = getattr(context, "runtime", None)
    fit_runtime_metadata_method = getattr(fit_runtime, "metadata", None)
    runtime_metadata_method = getattr(runtime, "metadata", None)
    bridge = getattr(context, "bridge", None)
    carrier_preflight = frozen._mapping(
        getattr(context, "carrier_preflight", None),
        label="one-pass transfer carrier preflight",
    )
    if not callable(fit_runtime_metadata_method) or not callable(
        runtime_metadata_method
    ):
        raise TypeError("one-pass transfer context runtimes have no metadata")
    fit_runtime_metadata = frozen._mapping(
        fit_runtime_metadata_method(),
        label="one-pass transfer D320 source runtime metadata",
    )
    runtime_metadata = frozen._mapping(
        runtime_metadata_method(),
        label="one-pass transfer graph target runtime metadata",
    )
    fit_runtime_sha256 = _require_sha256(
        fit_runtime_metadata.get("runtime_binding_sha256"),
        label="one-pass transfer D320 source runtime binding",
    )
    graph_runtime_sha256 = _require_sha256(
        runtime_metadata.get("runtime_binding_sha256"),
        label="one-pass transfer graph target runtime binding",
    )
    graph_bridge_sha256 = _require_sha256(
        getattr(bridge, "bridge_binding_sha256", None),
        label="one-pass transfer graph target bridge binding",
    )
    graph_bridge_parent_sha256 = _require_sha256(
        getattr(bridge, "parent_runtime_binding_sha256", None),
        label="one-pass transfer graph bridge parent runtime",
    )
    panel_file_sha256 = _require_sha256(
        panel.get("file_sha256"),
        label="one-pass transfer panel file",
    )
    factorized_model_sha256 = _require_sha256(
        lineage.get("factorized_live_model_sha256"),
        label="one-pass transfer factorized model",
    )
    factorized_execution_sha256 = _require_sha256(
        lineage.get("factorized_adapter_execution_sha256"),
        label="one-pass transfer factorized adapter execution",
    )
    parent_runtime = frozen._mapping(
        parent.get("runtime_binding"),
        label="one-pass transfer parent fit runtime",
    )
    if (
        parent.get("report_sha256") != PARENT_REPORT_SHA256
        or panel.get("example_count") != _EXPECTED_PROMPTS
        or fit_runtime_sha256
        != parent_runtime.get("runtime_binding_sha256")
        or graph_runtime_sha256 != graph_bridge_parent_sha256
        or fit_runtime_metadata.get("live_factorized_model_sha256")
        != factorized_model_sha256
        or fit_runtime_metadata.get("adapter_execution_sha256")
        != factorized_execution_sha256
        or runtime_metadata.get("live_model_sha256")
        != factorized_model_sha256
        or runtime_metadata.get("adapter_execution_sha256")
        != factorized_execution_sha256
        or carrier_preflight.get("d320_source_runtime_binding_sha256")
        != fit_runtime_sha256
        or carrier_preflight.get("graph_target_runtime_binding_sha256")
        != graph_runtime_sha256
        or carrier_preflight.get("graph_target_bridge_binding_sha256")
        != graph_bridge_sha256
        or carrier_preflight.get(
            "fit_consumed_prefix_fields_bitwise_identical"
        )
        is not True
        or carrier_preflight.get(
            "graph_prediction_admitted_to_exact_x4_fit_lineage"
        )
        is not False
    ):
        raise RuntimeError("one-pass transfer live context lineage differs")
    return {
        "successful_factorial_file_sha256": PARENT_FILE_SHA256,
        "successful_factorial_report_sha256": PARENT_REPORT_SHA256,
        "panel_file_sha256": panel_file_sha256,
        "panel_example_count": _EXPECTED_PROMPTS,
        "d320_source_conditional_runtime_binding_sha256": (
            fit_runtime_sha256
        ),
        "graph_target_runtime_binding_sha256": graph_runtime_sha256,
        "graph_target_bridge_binding_sha256": graph_bridge_sha256,
        "graph_target_bridge_parent_runtime_binding_sha256": (
            graph_bridge_parent_sha256
        ),
        "fit_consumed_prefix_fields_bitwise_identical": True,
        "graph_prediction_admitted_to_d320_fit_lineage": False,
        "factorized_live_model_sha256": factorized_model_sha256,
        "factorized_adapter_execution_sha256": (
            factorized_execution_sha256
        ),
    }


def run_gemma3_l3_l4_complete_h4_one_pass_transfer(
    *,
    materialization_report_path: Path | str = (
        DEFAULT_BASIS_MATERIALIZATION_REPORT
    ),
    expected_materialization_report_sha256: str,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the live A16 fixed-D320 one-pass carrier-transfer oracle.

    Live setup is imported lazily from the authenticated D320 materializer so
    both rungs share one frozen model/panel/bridge construction path.
    """

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite one-pass transfer report")
    basis, binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=(
            expected_materialization_report_sha256
        ),
        basis_sidecar_path=basis_sidecar_path,
    )
    try:
        from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
            prepare_complete_h4_rank320_live_context,
        )
    except ImportError as error:  # pragma: no cover - transitional fail closed
        raise RuntimeError(
            "authenticated rank320 live-context materializer is unavailable"
        ) from error

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    context_binding: dict[str, object] | None = None
    try:
        result = _evaluate_fixed_rank320_transfer(
            examples=context.examples,
            tokenize=context.tokenize,
            adapter=context.adapter,
            bridge=context.bridge,
            basis=basis,
            basis_binding=binding,
        )
        context.validate_immutable_inputs()
        context_binding = _live_context_binding(context)
    finally:
        context.close()
    if context_binding is None:
        raise RuntimeError("one-pass transfer context binding was omitted")
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": _ROLE,
        "artifact": {
            "file": str(destination),
            "committable": False,
        },
        "protocol": {
            "single_preregistered_arm": _ARM_ID,
            "panel": "reused_calibration_a_fit16",
            "basis_is_fixed_before_transfer": True,
            "rank_reopening_allowed": False,
            "global_basis_control_reopened": False,
            "per_prompt_execution_order": (
                "base_one_pass_h4_vjp_then_native_source_then_exact_h4_"
                "ceiling_then_fixed_d320_projection"
            ),
            "projected_residual_definition": (
                "P=((native_h4-base_one_pass_candidate_h4)@D.T)@D"
            ),
            "complete_h4_write_arithmetic": (
                "float64_realized_plus_float64_delta_then_one_live_dtype_cast"
            ),
        },
        "live_context_binding": context_binding,
        "basis_materialization_binding": materialization_binding,
        **result,
        "scientific_status": {
            "same_a_truth_leaking_transfer_only": True,
            "family_disjoint_learned_coordinates_fitted": False,
            "candidate_serving_authorized": False,
            "generator_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
        },
        "safety": dict(_SAFETY),
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed tail-informed D320 basis on the deployable "
            "one-pass complete-H4 carrier."
        )
    )
    parser.add_argument(
        "--basis-materialization-report",
        type=Path,
        default=DEFAULT_BASIS_MATERIALIZATION_REPORT,
    )
    parser.add_argument(
        "--basis-materialization-report-sha256",
        required=True,
        help="Expected report SHA-256 printed by D320 materialization.",
    )
    parser.add_argument(
        "--basis-sidecar",
        type=Path,
        default=None,
        help=(
            "Optional explicit sidecar path; it must resolve to the exact "
            "path committed by the materialization report."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_one_pass_transfer(
        materialization_report_path=args.basis_materialization_report,
        expected_materialization_report_sha256=(
            args.basis_materialization_report_sha256
        ),
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    comparison = frozen._mapping(
        report["comparison"], label="one-pass transfer comparison"
    )
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {comparison['classification']}")
    print(f"pass pattern: {comparison['pass_pattern']}")


if __name__ == "__main__":
    main()
