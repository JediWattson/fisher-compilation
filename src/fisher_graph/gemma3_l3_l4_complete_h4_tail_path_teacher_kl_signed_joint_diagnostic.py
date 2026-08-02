"""GL4 path-integrated teacher-KL signed-joint complete-H4 diagnostic.

This is a separate, fail-closed same-Calibration-A hypothesis screen.  It
integrates exact token teacher-KL H4 gradients along the finite D320-to-native
tail path before fitting any direction.  The four Gauss--Legendre nodes are
executed sequentially so a prompt retains one integrated gradient bank rather
than four full node banks.  Whole-family LOFO fitting and the finite
K=8/16/32/64 plus exact K=320 ladder are delegated to the authenticated
endpoint signed-joint diagnostic's established evaluator.

The source/native residual and native teacher logits remain truth leaking.
Nothing produced here is a serving provider, compression artifact, or speed
claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as endpoint
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as v1
from .causal_edge_transport import gauss_legendre_unit_interval
from .complete_h4_tail_signed_joint_projector import (
    CompleteH4TailSignedJointHeldFamilyFit,
    fit_complete_h4_tail_signed_joint_held_family,
)
from .complete_h4_tail_path_teacher_kl import (
    CompleteH4TailPathTeacherKLAccumulator,
    CompleteH4TailPathTeacherKLEvidence,
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
    complete_h4_tail_path_as_endpoint_example,
    complete_h4_tail_path_basis_contraction,
    complete_h4_tail_path_direct_contraction,
    summarize_complete_h4_tail_path_ftc_closure,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
    canonical_orthogonal_complement_rows,
    project_complete_h4_tail_rows,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import (
    AuthenticatedCompleteH4TransferProvider,
    _ValidatedRank320BasisContract,
    _load_committed_basis,
    _native_boundary,
    _retokenize,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _require_sha256,
    _runtime_tensor_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)


__all__ = [
    "DEFAULT_ENDPOINT_REPORT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRANSFER_REPORT",
    "ENDPOINT_REPORT_FILE_SHA256",
    "ENDPOINT_REPORT_SHA256",
    "PATH_QUADRATURE_ORDER",
    "run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic",
    "main",
]


DEFAULT_ENDPOINT_REPORT = endpoint.DEFAULT_OUTPUT
DEFAULT_MATERIALIZATION_REPORT = endpoint.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = endpoint.DEFAULT_TRANSFER_REPORT
DEFAULT_OUTPUT = v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "teacher-kl-path-gl4-signed-joint-lofo-finite-ladder-a-fit16-dev-v1.json"
)

# These pins were inserted only after the endpoint run closed and its file and
# logical report identities were independently computed.  The run API still
# exposes explicit expected-pin parameters so tests and future protocol
# versions cannot silently substitute a path-matching report.
ENDPOINT_REPORT_FILE_SHA256 = (
    "3ff52693fd863171fa5f130ac7a7ab32d3ab3e848fd51bee08b227d3a7eb635a"
)
ENDPOINT_REPORT_SHA256 = (
    "325b41db460435c653839b7e5c760449d6a9f1adfd02029f049873289ea8cd35"
)
PATH_QUADRATURE_ORDER = 4
PATH_RANKS = endpoint.SIGNED_JOINT_RANKS
_PATH_MAX_RANK = 64
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_teacher_kl_path_gl4_"
    "signed_joint_lofo.v1"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-tail-teacher-kl-path-gl4-signed-joint:v1\0"
)
_PATH_PROVIDER_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-path-node-provider:v1\0"
)
_PATH_NODE_RECEIPT_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-path-node-receipt:v1\0"
)
_PATH_PROMPT_RECEIPT_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-path-prompt-receipt:v1\0"
)
_PATH_COMPARISON_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-path-endpoint-comparison:v1\0"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _canonical_hex(value: object, *, label: str, positive: bool = False) -> tuple[str, float]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical float hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical float hex string") from error
    if (
        not math.isfinite(parsed)
        or parsed.hex() != value
        or (positive and parsed <= 0.0)
    ):
        raise ValueError(f"{label} must be finite and canonical")
    return value, parsed


def _endpoint_prompt_receipts(
    report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    rows = report.get("prompt_receipts")
    if not isinstance(rows, (list, tuple)) or len(rows) != v1._EXPECTED_EXAMPLES:
        raise ValueError("endpoint signed report prompt grid differs")
    result: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        row = _mapping(raw, label="endpoint signed prompt receipt")
        example_id = v1._identifier(
            row.get("example_id"), label="endpoint signed prompt example"
        )
        family_id = v1._identifier(
            row.get("family_id"), label="endpoint signed prompt family"
        )
        if example_id in result:
            raise ValueError("endpoint signed report has duplicate prompts")
        for key in (
            "artifact_sha256",
            "model_inputs_sha256",
            "base_x4_sha256",
            "supervised_indices_sha256",
            "supervised_targets_sha256",
            "endpoint_support_indices_sha256",
            "endpoint_support_targets_sha256",
            "compensation_target_sha256",
            "native_logits_sha256",
            "teacher_logits_sha256",
            "teacher_kl_vjp_artifact_sha256",
            "endpoint_execution_artifact_sha256",
            "endpoint_provider_artifact_sha256",
            "causality_receipt_sha256",
        ):
            _require_sha256(row.get(key), label=f"endpoint prompt {key}")
        if (
            family_id != row.get("family_id")
            or row.get("endpoint_support_supervised_token_count") is None
            or row.get("maximum_future_gradient_abs") != 0.0
            or row.get("future_gradient_nonzero_count") != 0
        ):
            raise ValueError("endpoint signed prompt semantics differ")
        result[example_id] = row
    if len({str(row["family_id"]) for row in result.values()}) != v1._EXPECTED_FAMILIES:
        raise ValueError("endpoint signed report family grid differs")
    return result


def _load_endpoint_report(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    """Authenticate the exact endpoint signed-joint run used as control."""

    file_pin = _require_sha256(
        expected_file_sha256, label="endpoint signed report file"
    )
    report_pin = _require_sha256(
        expected_report_sha256, label="endpoint signed report payload"
    )
    report = v1._load_pinned_report(
        path,
        expected_file_sha256=file_pin,
        expected_report_sha256=report_pin,
        label="teacher-KL endpoint signed-joint control",
    )
    binding = _mapping(report.get("input_binding"), label="endpoint input binding")
    protocol = _mapping(report.get("protocol"), label="endpoint protocol")
    science = _mapping(report.get("scientific_status"), label="endpoint science")
    safety = _mapping(report.get("safety"), label="endpoint safety")
    raw_observations = report.get("finite_observation_receipts")
    if not isinstance(raw_observations, (list, tuple)):
        raise ValueError("endpoint signed finite observations differ")
    observations = [
        _mapping(row, label="endpoint signed finite observation")
        for row in raw_observations
    ]
    observation_set = endpoint._finite_observation_set_sha256(observations)
    primary = report.get("primary_gate_results")
    if not isinstance(primary, (list, tuple)):
        raise ValueError("endpoint signed primary gates differ")
    primary_map = dict(primary)
    expected_classification = (
        "same_a_teacher_kl_signed_joint_bounded_fidelity_supported"
        if report.get("passed") is True
        else "same_a_teacher_kl_signed_joint_bounded_fidelity_not_supported"
    )
    if (
        report.get("schema") != endpoint._SCHEMA
        or report.get("classification") != expected_classification
        or tuple(protocol.get("requested_tail_ranks", ())) != endpoint.SIGNED_JOINT_RANKS
        or protocol.get("teacher_objective")
        != "token_KL(native_teacher||D320_candidate)"
        or binding.get("materialization_report_file_sha256")
        != v1.MATERIALIZATION_REPORT_FILE_SHA256
        or binding.get("materialization_report_sha256")
        != v1.MATERIALIZATION_REPORT_SHA256
        or binding.get("transfer_report_file_sha256")
        != v1.TRANSFER_REPORT_FILE_SHA256
        or binding.get("transfer_report_sha256") != v1.TRANSFER_REPORT_SHA256
        or report.get("finite_observation_set_sha256") != observation_set
        or primary_map.get("all_teacher_kl_vjps_have_zero_future_gradient") is not True
        or primary_map.get("shared_k320_every_prompt_h4_bitwise_native") is not True
        or primary_map.get("shared_k320_every_prompt_logits_bitwise_native") is not True
        or science.get("same_a_truth_leaking_hypothesis_use_only") is not True
        or science.get("candidate_serving_authorized") is not False
        or science.get("compression_claim") is not False
        or safety.get("contains_activation_tensors") is not False
        or safety.get("contains_gradient_tensors") is not False
        or safety.get("artifact_must_remain_outside_git") is not True
    ):
        raise ValueError("endpoint signed-joint control semantics differ")
    return report, _endpoint_prompt_receipts(report)


class _AuthenticatedPathNodeH4Provider(Gemma3L3L4CorrectionProvider):
    """Single-use GL4 H4 provider bound to both finite path endpoints."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "model_inputs_sha256",
        "bridge_binding_sha256",
        "prefix_artifact_sha256",
        "node_index",
        "alpha_hex",
        "quadrature_weight_hex",
        "base_h4_sha256",
        "d320_h4_sha256",
        "native_h4_sha256",
        "support_mask_sha256",
        "supported_rows_sha256",
        "tail_rows_sha256",
        "correction_sha256",
        "_support",
        "_correction",
        "_used",
    )

    def __init__(
        self,
        *,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        node_index: int,
        alpha_hex: str,
        quadrature_weight_hex: str,
        base_h4: Tensor,
        d320_h4: Tensor,
        native_h4: Tensor,
        support_mask: Tensor,
        supported_rows: Tensor,
        tail_rows: Tensor,
    ) -> None:
        alpha_canonical, alpha = _canonical_hex(alpha_hex, label="path alpha")
        weight_canonical, _weight = _canonical_hex(
            quadrature_weight_hex, label="path quadrature weight", positive=True
        )
        if (
            type(node_index) is not int
            or not 0 <= node_index < PATH_QUADRATURE_ORDER
            or alpha_canonical != GL4_UNIT_INTERVAL_NODES[node_index].hex()
            or weight_canonical != GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
        ):
            raise ValueError("path provider must use the exact indexed GL4 pair")
        if (
            not isinstance(base_h4, Tensor)
            or not isinstance(d320_h4, Tensor)
            or not isinstance(native_h4, Tensor)
            or base_h4.ndim != 3
            or base_h4.shape[0] != 1
            or base_h4.shape[-1] != v1._WIDTH
            or d320_h4.shape != base_h4.shape
            or native_h4.shape != base_h4.shape
            or not base_h4.is_floating_point()
            or d320_h4.dtype != base_h4.dtype
            or native_h4.dtype != base_h4.dtype
            or d320_h4.device != base_h4.device
            or native_h4.device != base_h4.device
            or not bool(torch.isfinite(base_h4).all())
            or not bool(torch.isfinite(d320_h4).all())
            or not bool(torch.isfinite(native_h4).all())
            or not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not bool(support_mask.any())
        ):
            raise ValueError("path provider boundary geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        row_count = int(support.sum())
        if (
            not isinstance(supported_rows, Tensor)
            or not isinstance(tail_rows, Tensor)
            or supported_rows.shape != (row_count, v1._WIDTH)
            or tail_rows.shape != supported_rows.shape
            or not supported_rows.is_floating_point()
            or not tail_rows.is_floating_point()
        ):
            raise ValueError("path provider row geometry differs")
        supported = supported_rows.detach().to(
            device="cpu", dtype=torch.float64
        ).clone().contiguous()
        tail = tail_rows.detach().to(
            device="cpu", dtype=torch.float64
        ).clone().contiguous()
        if not bool(torch.isfinite(supported).all()) or not bool(torch.isfinite(tail).all()):
            raise ValueError("path provider rows must be finite")
        base_cpu = base_h4.detach().to(device="cpu").contiguous()
        d320_cpu = d320_h4.detach().to(device="cpu").contiguous()
        native_cpu = native_h4.detach().to(device="cpu").contiguous()
        if (
            not endpoint.v1._bitwise_equal(base_cpu[~support], d320_cpu[~support])
            or not endpoint.v1._bitwise_equal(base_cpu[~support], native_cpu[~support])
        ):
            raise ValueError("path provider endpoints differ outside support")

        d320_expected = base_cpu.clone()
        native_expected = base_cpu.clone()
        support_indices = torch.nonzero(support[0], as_tuple=False).flatten()
        base_rows = base_cpu[0].index_select(0, support_indices).to(torch.float64)
        d320_expected[0].index_copy_(
            0, support_indices, (base_rows + supported).to(dtype=base_cpu.dtype)
        )
        native_expected[0].index_copy_(
            0,
            support_indices,
            (base_rows + supported + tail).to(dtype=base_cpu.dtype),
        )
        if not endpoint.v1._bitwise_equal(d320_expected, d320_cpu):
            raise ValueError("path provider supported rows do not reconstruct D320")
        if not endpoint.v1._bitwise_equal(native_expected, native_cpu):
            raise ValueError("path provider full tail does not reconstruct native H4")

        correction = torch.zeros(base_cpu.shape, dtype=torch.float64, device="cpu")
        correction[0].index_copy_(0, support_indices, supported + alpha * tail)
        self.site = v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="path provider model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="path provider bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="path provider prefix"
        )
        self.node_index = node_index
        self.alpha_hex = alpha_canonical
        self.quadrature_weight_hex = weight_canonical
        self.base_h4_sha256 = _runtime_tensor_sha256(base_cpu)
        self.d320_h4_sha256 = _runtime_tensor_sha256(d320_cpu)
        self.native_h4_sha256 = _runtime_tensor_sha256(native_cpu)
        self.support_mask_sha256 = _runtime_tensor_sha256(support)
        self.supported_rows_sha256 = _runtime_tensor_sha256(supported)
        self.tail_rows_sha256 = _runtime_tensor_sha256(tail)
        self._support = support
        self._correction = correction.contiguous()
        self.correction_sha256 = _runtime_tensor_sha256(self._correction)
        self._used = False
        self.artifact_sha256 = self._computed_sha256()
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.complete_h4_tail_teacher_kl_path_node_provider.v1",
            "site": self.site,
            "write_scope": self.write_scope,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "quadrature_rule": "gauss_legendre_order_4_on_unit_interval",
            "node_index": self.node_index,
            "alpha_hex": self.alpha_hex,
            "quadrature_weight_hex": self.quadrature_weight_hex,
            "base_h4_sha256": self.base_h4_sha256,
            "d320_h4_sha256": self.d320_h4_sha256,
            "native_h4_sha256": self.native_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "supported_rows_sha256": self.supported_rows_sha256,
            "tail_rows_sha256": self.tail_rows_sha256,
            "correction_sha256": self.correction_sha256,
            "trajectory": (
                "base_plus_D320_supported_projection_plus_alpha_times_"
                "canonical_null_tail"
            ),
            "single_use": True,
            "truth_leaking_hypothesis_use_only": True,
            "serving_authorized": False,
        }

    def _computed_sha256(self) -> str:
        return v1._domain_sha256(self._payload(), domain=_PATH_PROVIDER_DOMAIN)

    @property
    def used(self) -> bool:
        return self._used

    def validate_integrity(self) -> None:
        if (
            _runtime_tensor_sha256(self._support) != self.support_mask_sha256
            or _runtime_tensor_sha256(self._correction) != self.correction_sha256
            or bool((self._correction[~self._support] != 0).any())
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise RuntimeError("path-node provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("path-node provider cannot be reused")
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
            raise RuntimeError("path-node provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(slots=True)
class _LivePathTeacherKLTrace:
    """Duck-compatible finite trace plus scalar/hash-only path provenance."""

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
    teacher_logits_sha256: str
    d320_execution_artifact_sha256: str
    d320_provider_artifact_sha256: str
    path_evidence_artifact_sha256: str
    path_node_receipts: tuple[Mapping[str, object], ...]
    maximum_future_gradient_abs: float
    future_gradient_nonzero_count: int
    prompt_path_payload: Mapping[str, object]
    prompt_path_receipt_sha256: str


def _finite_endpoint_from_path_evidence(
    evidence: CompleteH4TailPathTeacherKLEvidence,
    *,
    full_native_minus_base_residual_rows: Tensor,
    supported_basis: Tensor,
) -> CompleteH4TailEndpointExample:
    """Build the finite-evaluator duck endpoint without dropping D320.

    Pure path evidence owns ``E=(I-P_D)R`` because GL4 integrates the
    D320-to-native displacement.  The reused finite evaluator, however,
    expects full ``R=native-base`` so it can reconstruct ``P_D R + prefix(E)``.
    This adapter proves those two records have the same complement before
    returning the full-R record.
    """

    evidence.validate_integrity()
    full_residual = full_native_minus_base_residual_rows.detach().to(
        device="cpu", dtype=torch.float64
    ).clone().contiguous()
    basis = supported_basis.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    if (
        full_residual.shape != evidence.residual_rows.shape
        or basis.ndim != 2
        or basis.shape[1] != evidence.width
        or not bool(torch.isfinite(full_residual).all())
    ):
        raise ValueError("finite path endpoint geometry differs")
    reconstructed_tail = project_complete_h4_tail_rows(full_residual, basis)
    if not torch.allclose(
        reconstructed_tail,
        evidence.residual_rows,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "finite full-R endpoint and FTC path evidence disagree in null(D320)"
        )
    path_endpoint = complete_h4_tail_path_as_endpoint_example(evidence)
    finite_endpoint = CompleteH4TailEndpointExample(
        example_id=evidence.example_id,
        family_id=evidence.family_id,
        residual_rows=full_residual,
        token_h4_gradients=path_endpoint.token_h4_gradients,
        compensation_target=path_endpoint.compensation_target,
    )
    # Any signed-joint direction is in null(D320), so equality of the
    # complement rows proves equal amplitudes and therefore equal q_t scores.
    if (
        not torch.equal(
            finite_endpoint.token_h4_gradients,
            path_endpoint.token_h4_gradients,
        )
        or not torch.equal(
            finite_endpoint.compensation_target,
            path_endpoint.compensation_target,
        )
    ):
        raise RuntimeError("finite/path endpoint objective tensors differ")
    return finite_endpoint


def _expected_path_candidate_h4(
    *,
    base_h4: Tensor,
    support_indices: Tensor,
    supported_rows: Tensor,
    tail_rows: Tensor,
    alpha: float,
) -> Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("path candidate alpha must be in the unit interval")
    expected = base_h4.detach().clone().contiguous()
    indices = support_indices.detach().to(
        device=expected.device, dtype=torch.int64
    ).contiguous()
    base_rows = expected[0].index_select(0, indices).to(torch.float64)
    correction_rows = (
        supported_rows.detach().to(device=expected.device, dtype=torch.float64)
        + alpha
        * tail_rows.detach().to(device=expected.device, dtype=torch.float64)
    )
    expected[0].index_copy_(
        0, indices, (base_rows + correction_rows).to(dtype=expected.dtype)
    )
    return expected


def _causal_future_gradient_summary(
    *,
    gradient_rows: Tensor,
    endpoint_indices: Tensor,
    support_indices: Tensor,
    logical_positions: Tensor,
) -> tuple[float, int]:
    logical = logical_positions.detach().to(device="cpu")
    supervised_logical = logical[0].index_select(
        0, endpoint_indices.detach().to(device="cpu", dtype=torch.int64)
    )
    support_logical = logical[0].index_select(
        0, support_indices.detach().to(device="cpu", dtype=torch.int64)
    )
    maximum = 0.0
    nonzero = 0
    for token_index in range(int(endpoint_indices.numel())):
        future_mask = support_logical > supervised_logical[token_index]
        if bool(future_mask.any()):
            future = gradient_rows[token_index, future_mask]
            maximum = max(maximum, float(future.abs().max()))
            nonzero += int((future != 0).sum())
    return maximum, nonzero


def _collect_path_teacher_kl_traces(
    *,
    context: object,
    basis: Tensor,
    basis_binding: Mapping[str, str],
    transfer_receipts: Mapping[str, Mapping[str, object]],
    endpoint_prompt_receipts: Mapping[str, Mapping[str, object]],
) -> tuple[
    list[_LivePathTeacherKLTrace],
    tuple[CompleteH4TailPathTeacherKLEvidence, ...],
    dict[str, int],
]:
    """Stream four authenticated GL4 VJPs per A16 prompt."""

    adapter = getattr(context, "adapter")
    bridge = getattr(context, "bridge")
    tokenize = getattr(context, "tokenize")
    examples = tuple(getattr(context, "examples"))
    bridge.validate_integrity()
    bridge_sha256 = _require_sha256(
        bridge.bridge_binding_sha256, label="path teacher-KL bridge"
    )
    basis_contract = _ValidatedRank320BasisContract.build(basis, basis_binding)
    traces: list[_LivePathTeacherKLTrace] = []
    evidence_values: list[CompleteH4TailPathTeacherKLEvidence] = []
    resources = {
        "base_forward_count": 0,
        "native_teacher_forward_count": 0,
        "d320_boundary_forward_count": 0,
        "path_teacher_kl_vjp_forward_count": 0,
        "path_teacher_kl_vjp_backward_call_count": 0,
        "path_quadrature_node_count": 0,
        "ordinary_supervised_token_count": 0,
        "endpoint_support_supervised_token_count": 0,
        "complete_h4_support_row_count": 0,
        "graph_core_row_count": 0,
        "causal_tail_row_count": 0,
        "graph_core_supervised_token_count": 0,
        "causal_tail_supervised_token_count": 0,
    }

    for example in sorted(examples, key=lambda value: value.example_id):
        example_id = v1._identifier(example.example_id, label="path example_id")
        family_id = v1._identifier(example.family_id, label="path family_id")
        prior = transfer_receipts.get(example_id)
        endpoint_prior = endpoint_prompt_receipts.get(example_id)
        if prior is None or endpoint_prior is None:
            raise ValueError("pinned transfer/endpoint report omitted an A16 prompt")
        model_inputs, indices, targets = _retokenize(tokenize, example)
        model_inputs_sha256 = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        indices_sha256 = _runtime_tensor_sha256(indices)
        targets_sha256 = _runtime_tensor_sha256(targets)
        if (
            prior.get("family_id") != family_id
            or endpoint_prior.get("family_id") != family_id
            or prior.get("model_inputs_sha256") != model_inputs_sha256
            or endpoint_prior.get("model_inputs_sha256") != model_inputs_sha256
            or prior.get("supervised_indices_sha256") != indices_sha256
            or endpoint_prior.get("supervised_indices_sha256") != indices_sha256
            or prior.get("supervised_targets_sha256") != targets_sha256
            or endpoint_prior.get("supervised_targets_sha256") != targets_sha256
        ):
            raise RuntimeError("path supervision differs from pinned evidence")

        base = bridge.execute(adapter, model_inputs)
        resources["base_forward_count"] += 1
        if getattr(base, "model_forward_count", None) != 1:
            raise RuntimeError("path base execution forward count differs")
        prefix = base.prefix
        prefix.validate_integrity()
        base_h4 = base.candidate_h4
        base_x4_sha256 = _runtime_tensor_sha256(base.candidate_x4)
        if (
            prefix.bridge_binding_sha256 != bridge_sha256
            or prior.get("bridge_binding_sha256") != bridge_sha256
            or prior.get("prefix_artifact_sha256") != prefix.artifact_sha256
            or prior.get("base_candidate_h4_sha256")
            != _runtime_tensor_sha256(base_h4)
            or prior.get("base_candidate_x4_sha256") != base_x4_sha256
            or endpoint_prior.get("base_x4_sha256") != base_x4_sha256
        ):
            raise RuntimeError("path base identity differs from pinned evidence")

        teacher_logits, native_h4, native_positions, native_valid = _native_boundary(
            adapter, model_inputs
        )
        resources["native_teacher_forward_count"] += 1
        teacher_logits_sha256 = _runtime_tensor_sha256(teacher_logits)
        if (
            native_h4.shape != base_h4.shape
            or not v1._bitwise_equal(native_positions, prefix.logical_positions)
            or not v1._bitwise_equal(native_valid, prefix.valid_target_mask)
            or prior.get("source_logits_sha256") != teacher_logits_sha256
            or prior.get("native_h4_sha256") != _runtime_tensor_sha256(native_h4)
            or endpoint_prior.get("native_logits_sha256") != teacher_logits_sha256
            or endpoint_prior.get("teacher_logits_sha256") != teacher_logits_sha256
        ):
            raise RuntimeError("path native boundary differs from pinned evidence")

        support = prefix.complete_h4_causal_support_mask().detach().to(device="cpu")
        core = prefix.target_affected_mask.detach().to(device="cpu")
        if support.shape[0] != 1:
            raise RuntimeError("path protocol requires one prompt per execution")
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
        endpoint_grid = endpoint._canonical_support_grid(
            endpoint_indices.detach().to(device="cpu", dtype=torch.int64)
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
        if (
            prior.get("complete_h4_support_mask_sha256")
            != _runtime_tensor_sha256(support)
            or prior.get("complete_h4_support_rows") != int(support.sum())
            or prior.get("graph_core_rows") != int(core.sum())
            or prior.get("causal_tail_rows") != int((support & ~core).sum())
            or endpoint_prior.get("endpoint_support_indices_sha256")
            != _runtime_tensor_sha256(endpoint_indices)
            or endpoint_prior.get("endpoint_support_targets_sha256")
            != _runtime_tensor_sha256(endpoint_targets)
            or endpoint_prior.get("endpoint_support_supervised_token_count")
            != int(endpoint_indices.numel())
        ):
            raise RuntimeError("path support grid differs from pinned evidence")

        base_cpu = base_h4.detach().to(device="cpu", dtype=torch.float64)
        native_cpu = native_h4.detach().to(device="cpu", dtype=torch.float64)
        full_residual_rows = (
            native_cpu[0].index_select(0, support_indices)
            - base_cpu[0].index_select(0, support_indices)
        ).contiguous()
        supported_rows = ((full_residual_rows @ basis.T) @ basis).contiguous()
        tail_rows = project_complete_h4_tail_rows(full_residual_rows, basis)

        d320_provider = AuthenticatedCompleteH4TransferProvider(
            role="rank320_projection",
            model_inputs_sha256=model_inputs_sha256,
            bridge_binding_sha256=bridge_sha256,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base_h4,
            native_h4=native_h4,
            basis_contract=basis_contract,
            support_mask=support,
        )
        d320_execution = bridge.execute(
            adapter, model_inputs, h4_head=d320_provider
        )
        resources["d320_boundary_forward_count"] += 1
        d320_provider.validate_integrity()
        projected_prior = _mapping(
            prior.get("projected_provider"), label="pinned D320 provider"
        )
        expected_d320_h4 = _expected_path_candidate_h4(
            base_h4=base_h4,
            support_indices=support_indices,
            supported_rows=supported_rows,
            tail_rows=tail_rows,
            alpha=0.0,
        )
        expected_native_h4 = _expected_path_candidate_h4(
            base_h4=base_h4,
            support_indices=support_indices,
            supported_rows=supported_rows,
            tail_rows=tail_rows,
            alpha=1.0,
        )
        if (
            getattr(d320_execution, "model_forward_count", None) != 1
            or not d320_provider.used
            or d320_provider.artifact_sha256
            != projected_prior.get("artifact_sha256")
            or d320_provider.artifact_sha256
            != endpoint_prior.get("endpoint_provider_artifact_sha256")
            or d320_execution.artifact_sha256
            != endpoint_prior.get("endpoint_execution_artifact_sha256")
            or _runtime_tensor_sha256(d320_execution.candidate_h4)
            != prior.get("projected_h4_sha256")
            or _runtime_tensor_sha256(d320_execution.logits)
            != prior.get("projected_logits_sha256")
            or not v1._bitwise_equal(
                d320_execution.candidate_h4.detach(), expected_d320_h4
            )
            or not v1._bitwise_equal(
                expected_native_h4.detach().to(device="cpu"),
                native_h4.detach().to(device="cpu"),
            )
        ):
            raise RuntimeError("fresh path D320/native sentinels differ")

        native_token_nll = v1._selected_token_nll(
            teacher_logits, endpoint_indices, endpoint_targets
        )
        native_ordinary_token_nll = v1._selected_token_nll(
            teacher_logits, indices, targets
        )
        d320_token_nll = v1._selected_token_nll(
            d320_execution.logits, endpoint_indices, endpoint_targets
        )
        d320_ordinary_token_nll = v1._selected_token_nll(
            d320_execution.logits, indices, targets
        )
        source_token_kl = endpoint._selected_token_teacher_kl(
            teacher_logits, d320_execution.logits, endpoint_indices
        )
        native_token_kl = endpoint._selected_token_teacher_kl(
            teacher_logits, teacher_logits, endpoint_indices
        )
        if bool((native_token_kl != 0).any()):
            raise RuntimeError("native teacher-KL endpoint is not exact zero")
        d320_h4 = d320_execution.candidate_h4.detach().clone().contiguous()
        d320_execution_sha256 = d320_execution.artifact_sha256
        d320_provider_sha256 = d320_provider.artifact_sha256
        del d320_execution, d320_provider

        accumulator = CompleteH4TailPathTeacherKLAccumulator(
            example_id=example_id,
            family_id=family_id,
            residual_rows=tail_rows,
            source_token_teacher_kl=source_token_kl,
            native_token_teacher_kl=native_token_kl,
        )
        live_node_receipts: list[Mapping[str, object]] = []
        prompt_maximum_future = 0.0
        prompt_future_nonzero = 0
        for node_index, (alpha, weight) in enumerate(
            zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS)
        ):
            path_provider = _AuthenticatedPathNodeH4Provider(
                model_inputs_sha256=model_inputs_sha256,
                bridge_binding_sha256=bridge_sha256,
                prefix_artifact_sha256=prefix.artifact_sha256,
                node_index=node_index,
                alpha_hex=alpha.hex(),
                quadrature_weight_hex=weight.hex(),
                base_h4=base_h4,
                d320_h4=d320_h4,
                native_h4=native_h4,
                support_mask=support,
                supported_rows=supported_rows,
                tail_rows=tail_rows,
            )
            node_vjp = bridge.execute_h4_token_teacher_kl_vjps(
                adapter,
                model_inputs,
                teacher_logits=teacher_logits,
                supervised_indices=endpoint_grid,
                vjp_chunk_size=v1._VJP_CHUNK_SIZE,
                h4_head=path_provider,
            )
            resources["path_teacher_kl_vjp_forward_count"] += 1
            resources["path_teacher_kl_vjp_backward_call_count"] += (
                node_vjp.backward_call_count
            )
            resources["path_quadrature_node_count"] += 1
            node_vjp.validate_integrity()
            path_provider.validate_integrity()
            expected_node_h4 = _expected_path_candidate_h4(
                base_h4=base_h4,
                support_indices=support_indices,
                supported_rows=supported_rows,
                tail_rows=tail_rows,
                alpha=alpha,
            )
            independent_node_kl = endpoint._selected_token_teacher_kl(
                teacher_logits, node_vjp.execution.logits, endpoint_indices
            )
            token_kl = node_vjp.token_kl_divergences.detach().to(
                device="cpu", dtype=torch.float64
            ).contiguous()
            gradient_rows = (
                node_vjp.h4_gradients.detach()
                .to(device="cpu", dtype=torch.float64)[:, 0]
                .index_select(1, support_indices)
                .contiguous()
            )
            maximum_future, future_nonzero = _causal_future_gradient_summary(
                gradient_rows=gradient_rows,
                endpoint_indices=endpoint_indices,
                support_indices=support_indices,
                logical_positions=prefix.logical_positions,
            )
            if (
                not path_provider.used
                or node_vjp.execution.model_inputs_sha256 != model_inputs_sha256
                or node_vjp.execution.bridge_binding_sha256 != bridge_sha256
                or node_vjp.execution.prefix.artifact_sha256
                != prefix.artifact_sha256
                or node_vjp.execution.h4_head_sha256
                != path_provider.artifact_sha256
                or node_vjp.teacher_logits_sha256 != teacher_logits_sha256
                or _runtime_tensor_sha256(node_vjp.execution.candidate_x4)
                != base_x4_sha256
                or not torch.equal(
                    node_vjp.supervised_indices.detach().to(device="cpu"),
                    endpoint_grid,
                )
                or not torch.allclose(
                    token_kl, independent_node_kl, rtol=0.0, atol=1.0e-6
                )
                or not v1._bitwise_equal(
                    node_vjp.execution.candidate_h4.detach(), expected_node_h4
                )
                or maximum_future != 0.0
                or future_nonzero != 0
            ):
                raise RuntimeError("path GL4 node execution or causality differs")
            core_receipt = accumulator.add_node(
                node_index=node_index,
                path_fraction=alpha,
                quadrature_weight=weight,
                token_h4_gradients=gradient_rows,
                token_teacher_kl=token_kl,
                vjp_artifact_sha256=node_vjp.artifact_sha256,
                provider_artifact_sha256=path_provider.artifact_sha256,
                execution_artifact_sha256=node_vjp.execution.artifact_sha256,
                maximum_future_gradient_abs=maximum_future,
                future_gradient_nonzero_count=future_nonzero,
            )
            live_receipt: dict[str, object] = {
                "example_id": example_id,
                "family_id": family_id,
                "node_index": node_index,
                "path_fraction_hex": alpha.hex(),
                "quadrature_weight_hex": weight.hex(),
                "core_node_receipt_sha256": core_receipt.artifact_sha256,
                "provider_artifact_sha256": path_provider.artifact_sha256,
                "execution_artifact_sha256": node_vjp.execution.artifact_sha256,
                "vjp_artifact_sha256": node_vjp.artifact_sha256,
                "candidate_h4_sha256": _runtime_tensor_sha256(
                    node_vjp.execution.candidate_h4
                ),
                "candidate_logits_sha256": _runtime_tensor_sha256(
                    node_vjp.execution.logits
                ),
                "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
                "maximum_future_gradient_abs_hex": maximum_future.hex(),
                "future_gradient_nonzero_count": future_nonzero,
                "executed_h4_equals_base_plus_supported_plus_alpha_tail": True,
                "raw_gradient_or_logits_serialized": False,
            }
            live_receipt["receipt_sha256"] = v1._domain_sha256(
                live_receipt, domain=_PATH_NODE_RECEIPT_DOMAIN
            )
            live_node_receipts.append(live_receipt)
            prompt_maximum_future = max(prompt_maximum_future, maximum_future)
            prompt_future_nonzero += future_nonzero
            del node_vjp, path_provider, gradient_rows, token_kl

        path_evidence = accumulator.finalize()
        finite_endpoint = _finite_endpoint_from_path_evidence(
            path_evidence,
            full_native_minus_base_residual_rows=full_residual_rows,
            supported_basis=basis,
        )
        finite_metadata = finite_endpoint.metadata()
        if (
            finite_metadata["residual_sha256"]
            != endpoint_prior.get("residual_sha256")
            or finite_metadata["compensation_target_sha256"]
            != endpoint_prior.get("compensation_target_sha256")
            or prompt_maximum_future != 0.0
            or prompt_future_nonzero != 0
        ):
            raise RuntimeError("path finite endpoint differs from pinned boundaries")
        residual_reconstruction_error = float(
            (supported_rows + tail_rows - full_residual_rows).abs().max()
        )
        if residual_reconstruction_error > 1.0e-9:
            raise RuntimeError("P_D R plus E does not reconstruct the full residual")
        prompt_payload: dict[str, object] = {
            "example_id": example_id,
            "family_id": family_id,
            "path_evidence_artifact_sha256": path_evidence.artifact_sha256,
            "finite_endpoint_artifact_sha256": finite_endpoint.artifact_sha256,
            "pinned_endpoint_record_artifact_sha256": endpoint_prior.get(
                "artifact_sha256"
            ),
            "d320_execution_artifact_sha256": d320_execution_sha256,
            "d320_provider_artifact_sha256": d320_provider_sha256,
            "base_h4_sha256": _runtime_tensor_sha256(base_h4),
            "d320_h4_sha256": _runtime_tensor_sha256(d320_h4),
            "native_h4_sha256": _runtime_tensor_sha256(native_h4),
            "support_mask_sha256": _runtime_tensor_sha256(support),
            "supported_rows_sha256": _runtime_tensor_sha256(supported_rows),
            "tail_rows_sha256": _runtime_tensor_sha256(tail_rows),
            "node_receipt_sha256s": tuple(
                str(row["receipt_sha256"]) for row in live_node_receipts
            ),
            "maximum_future_gradient_abs_hex": prompt_maximum_future.hex(),
            "future_gradient_nonzero_count": prompt_future_nonzero,
            "full_residual_reconstruction_max_abs_error_hex": (
                residual_reconstruction_error.hex()
            ),
            "d320_h4_bitwise_pinned": True,
            "P_D_R_plus_E_H4_bitwise_native": True,
            "native_self_teacher_KL_exact_zero": True,
            "typed_path_evidence_uses_E": True,
            "finite_duck_endpoint_uses_full_R": True,
            "complement_coordinate_equivalence_proven": True,
            "raw_evidence_serialized": False,
        }
        prompt_receipt_sha256 = v1._domain_sha256(
            prompt_payload, domain=_PATH_PROMPT_RECEIPT_DOMAIN
        )
        traces.append(
            _LivePathTeacherKLTrace(
                example=example,
                example_id=example_id,
                family_id=family_id,
                model_inputs_sha256=model_inputs_sha256,
                supervised_indices_sha256=indices_sha256,
                supervised_targets_sha256=targets_sha256,
                endpoint_indices_sha256=_runtime_tensor_sha256(endpoint_indices),
                endpoint_targets_sha256=_runtime_tensor_sha256(endpoint_targets),
                prefix=prefix,
                base_x4_sha256=base_x4_sha256,
                base_h4=base_h4.detach().clone().contiguous(),
                native_h4=native_h4.detach().to(device="cpu").clone().contiguous(),
                support_indices=support_indices.clone().contiguous(),
                selected_by_ledger={
                    key: value.clone().contiguous()
                    for key, value in selected_by_ledger.items()
                },
                endpoint=finite_endpoint,
                native_token_nll=native_token_nll,
                d320_token_nll=d320_token_nll,
                native_ordinary_token_nll=native_ordinary_token_nll,
                d320_ordinary_token_nll=d320_ordinary_token_nll,
                native_logits_sha256=teacher_logits_sha256,
                teacher_logits_sha256=teacher_logits_sha256,
                d320_execution_artifact_sha256=d320_execution_sha256,
                d320_provider_artifact_sha256=d320_provider_sha256,
                path_evidence_artifact_sha256=path_evidence.artifact_sha256,
                path_node_receipts=tuple(live_node_receipts),
                maximum_future_gradient_abs=prompt_maximum_future,
                future_gradient_nonzero_count=prompt_future_nonzero,
                prompt_path_payload=dict(prompt_payload),
                prompt_path_receipt_sha256=prompt_receipt_sha256,
            )
        )
        evidence_values.append(path_evidence)
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
        del base, teacher_logits, native_h4, d320_h4, model_inputs

    derived_backward_calls = PATH_QUADRATURE_ORDER * sum(
        (trace.endpoint.supervised_tokens + v1._VJP_CHUNK_SIZE - 1)
        // v1._VJP_CHUNK_SIZE
        for trace in traces
    )
    if derived_backward_calls != 436:
        raise RuntimeError("path-derived GL4 backward count differs from frozen A16")
    expected_counts = {
        "base_forward_count": v1._EXPECTED_EXAMPLES,
        "native_teacher_forward_count": v1._EXPECTED_EXAMPLES,
        "d320_boundary_forward_count": v1._EXPECTED_EXAMPLES,
        "path_teacher_kl_vjp_forward_count": (
            PATH_QUADRATURE_ORDER * v1._EXPECTED_EXAMPLES
        ),
        "path_teacher_kl_vjp_backward_call_count": derived_backward_calls,
        "path_quadrature_node_count": PATH_QUADRATURE_ORDER * v1._EXPECTED_EXAMPLES,
        "ordinary_supervised_token_count": v1._EXPECTED_ORDINARY_TOKENS,
        "endpoint_support_supervised_token_count": v1._EXPECTED_SUPPORT_TOKENS,
        "complete_h4_support_row_count": v1._EXPECTED_SUPPORT_ROWS,
        "graph_core_row_count": v1._EXPECTED_GRAPH_CORE_ROWS,
        "causal_tail_row_count": v1._EXPECTED_CAUSAL_TAIL_ROWS,
        "graph_core_supervised_token_count": v1._EXPECTED_GRAPH_CORE_TOKENS,
        "causal_tail_supervised_token_count": v1._EXPECTED_CAUSAL_TAIL_TOKENS,
    }
    if resources != expected_counts:
        raise RuntimeError("path collection resource/support accounting differs")
    return traces, tuple(evidence_values), resources


def _closure_gate_results(
    closure: object,
    *,
    evidence: Sequence[CompleteH4TailPathTeacherKLEvidence],
    supported_basis: Tensor,
    traces: Sequence[_LivePathTeacherKLTrace],
) -> tuple[dict[str, bool], dict[str, object]]:
    family_summaries = tuple(getattr(closure, "family_summaries"))
    complement = canonical_orthogonal_complement_rows(supported_basis)
    contraction_errors: list[Tensor] = []
    direct_values: list[Tensor] = []
    for value in evidence:
        direct = complete_h4_tail_path_direct_contraction(value)
        through_complete_complement = complete_h4_tail_path_basis_contraction(
            value, complement
        )
        direct_values.append(direct)
        contraction_errors.append(through_complete_complement - direct)
    maximum_contraction_error = max(
        float(value.abs().max()) for value in contraction_errors
    )
    contraction_error_norm = math.sqrt(
        sum(float(value.square().sum()) for value in contraction_errors)
    )
    direct_norm = math.sqrt(sum(float(value.square().sum()) for value in direct_values))
    relative_contraction_error = contraction_error_norm / max(
        direct_norm, 64.0 * torch.finfo(torch.float64).eps
    )
    execution_invariants = {
        "every_GL4_node_has_zero_future_gradient": all(
            trace.maximum_future_gradient_abs == 0.0
            and trace.future_gradient_nonzero_count == 0
            for trace in traces
        ),
        "native_self_teacher_KL_is_exact_zero": all(
            bool((value.native_token_teacher_kl == 0).all()) for value in evidence
        ),
        "every_prompt_P_D_R_plus_E_reconstructs_full_R_at_most_1e_minus_9": all(
            float.fromhex(
                str(
                    trace.prompt_path_payload[
                        "full_residual_reconstruction_max_abs_error_hex"
                    ]
                )
            )
            <= 1.0e-9
            for trace in traces
        ),
        "every_prompt_D320_H4_matches_pinned_endpoint": all(
            trace.prompt_path_payload.get("d320_h4_bitwise_pinned") is True
            for trace in traces
        ),
        "every_prompt_full_tail_H4_is_bitwise_native": all(
            trace.prompt_path_payload.get("P_D_R_plus_E_H4_bitwise_native") is True
            for trace in traces
        ),
        "direct_and_complete_complement_contraction_agree": (
            maximum_contraction_error <= 1.0e-9
            and relative_contraction_error <= 1.0e-9
        ),
    }
    gates = {
        "family_macro_relative_rmse_at_most_0_05": (
            float(getattr(closure, "relative_rmse")) <= 0.05
        ),
        "every_family_relative_rmse_at_most_0_10": all(
            float(value.relative_rmse) <= 0.10 for value in family_summaries
        ),
        "family_macro_path_integral_target_cosine_at_least_0_99": (
            float(getattr(closure, "cosine")) >= 0.99
        ),
        **execution_invariants,
    }
    diagnostics = {
        "complete_complement_rank": int(complement.shape[0]),
        "maximum_direct_minus_complete_complement_contraction_abs_error": (
            maximum_contraction_error
        ),
        "relative_direct_minus_complete_complement_contraction_l2_error": (
            relative_contraction_error
        ),
        "direct_complete_complement_absolute_tolerance": 1.0e-9,
        "direct_complete_complement_relative_tolerance": 1.0e-9,
    }
    return gates, diagnostics


def _path_resource_accounting(
    traces: Sequence[_LivePathTeacherKLTrace],
    *,
    collection_resources: Mapping[str, int],
    finite_resources: Mapping[str, int] | None = None,
    signed_fits: Mapping[str, CompleteH4TailSignedJointHeldFamilyFit] | None = None,
) -> dict[str, object]:
    """Exact executed counts plus auditable logical-MAC accounting."""

    collection = dict(collection_resources)
    collection_forward_count = (
        collection["base_forward_count"]
        + collection["native_teacher_forward_count"]
        + collection["d320_boundary_forward_count"]
        + collection["path_teacher_kl_vjp_forward_count"]
    )
    if (
        collection_forward_count != 112
        or collection["path_teacher_kl_vjp_backward_call_count"] != 436
        or collection["path_quadrature_node_count"] != 64
    ):
        raise RuntimeError("path collection accounting differs before reporting")
    support_rows = sum(trace.endpoint.residual_rows.shape[0] for trace in traces)
    token_row_width = sum(
        trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * v1._WIDTH
        for trace in traces
    )
    result: dict[str, object] = {
        **collection,
        "quadrature_order": PATH_QUADRATURE_ORDER,
        "vjp_chunk_size": v1._VJP_CHUNK_SIZE,
        "collection_model_forward_count": collection_forward_count,
        "GL4_weighted_gradient_accumulation_scalar_multiply_adds": (
            2 * PATH_QUADRATURE_ORDER * token_row_width
        ),
        "path_provider_supported_plus_alpha_tail_scalar_multiply_adds": (
            2 * PATH_QUADRATURE_ORDER * support_rows * v1._WIDTH
        ),
        "path_node_executed_H4_independent_reconstruction_scalar_adds": (
            2 * PATH_QUADRATURE_ORDER * support_rows * v1._WIDTH
        ),
        "fresh_D320_transfer_plus_independent_supported_tail_projection_logical_macs": (
            6 * support_rows * v1._D_RANK * v1._WIDTH
        ),
        "path_full_residual_and_tail_scalar_subtractions": (
            2 * support_rows * v1._WIDTH
        ),
        "peak_simultaneously_retained_full_sequence_vocabulary_tensor_count": 4,
        "peak_full_vocabulary_residency_reason": (
            "base_logits_plus_native_teacher_logits_plus_detached_teacher_"
            "snapshot_plus_one_transient_GL4_candidate_logits"
        ),
        "full_GL4_node_gradient_banks_simultaneously_retained": 1,
        "full_GL4_node_gradient_banks_retained_after_prompt_finalize": 0,
        "all_four_GL4_node_banks_retained_together": False,
        "serving_learned_parameter_count": "not_applicable_no_serving_artifact",
        "serving_logical_macs_per_token": "not_applicable_no_serving_artifact",
    }
    if finite_resources is None or signed_fits is None:
        result.update(
            {
                "finite_evaluation_executed": False,
                "total_model_forward_count": collection_forward_count,
                "signed_joint_fit_executed": False,
            }
        )
        return result

    finite = dict(finite_resources)
    if (
        finite.get("finite_native_forward_count") != v1._EXPECTED_EXAMPLES
        or finite.get("finite_candidate_forward_count") != 144
        or finite.get("finite_signed_joint_forward_count") != 64
        or finite.get("finite_fixed_pca_diagonal_forward_count") != 64
        or finite.get("finite_shared_exact_sentinel_forward_count")
        != v1._EXPECTED_EXAMPLES
    ):
        raise RuntimeError("path finite evaluation accounting differs")
    by_family = {
        family: tuple(trace for trace in traces if trace.family_id != family)
        for family in sorted(signed_fits)
    }
    c = v1._D_RANK
    w = v1._WIDTH
    coordinate_transform_macs = sum(
        trace.endpoint.residual_rows.shape[0] * w * c
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * w
        * c
        for training in by_family.values()
        for trace in training
    )
    operator_and_score_macs = 0
    low_rank_deflation_macs = 0
    eigh_calls = 0
    for family, fit in signed_fits.items():
        attempts = fit.rank + (
            0 if fit.stop_reason == "requested_rank_reached" else 1
        )
        eigh_calls += attempts
        for step in range(attempts):
            for trace in by_family[family]:
                rows = trace.endpoint.residual_rows.shape[0]
                tokens = trace.endpoint.supervised_tokens
                operator_and_score_macs += (
                    tokens * rows * c
                    + c * rows * c
                    + rows * c
                    + tokens * rows * c
                    + tokens * rows
                )
            if step:
                low_rank_deflation_macs += (
                    3 * step * c * c + 3 * step * step * c
                )
    pca_projection_and_covariance_macs = sum(
        3 * trace.endpoint.residual_rows.shape[0] * c * w
        + trace.endpoint.residual_rows.shape[0] * c * c
        for training in by_family.values()
        for trace in training
    )
    pca_q2_order_macs = sum(
        trace.endpoint.residual_rows.shape[0] * c * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * c
        * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * c
        for training in by_family.values()
        for trace in training
    )
    pca_gain_rescore_macs = sum(
        trace.endpoint.residual_rows.shape[0] * _PATH_MAX_RANK * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * _PATH_MAX_RANK
        * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * _PATH_MAX_RANK
        for training in by_family.values()
        for trace in training
    )
    bounded_prefix_macs = {
        str(rank): sum(
            2
            * trace.endpoint.residual_rows.shape[0]
            * min(rank, signed_fits[trace.family_id].rank)
            * w
            + trace.endpoint.residual_rows.shape[0]
            * min(rank, signed_fits[trace.family_id].rank)
            for trace in traces
        )
        for rank in PATH_RANKS[:-1]
    }
    result.update(
        {
            **finite,
            "finite_evaluation_executed": True,
            "finite_evaluation_model_forward_count": (
                int(finite["finite_native_forward_count"])
                + int(finite["finite_candidate_forward_count"])
            ),
            "total_model_forward_count": (
                collection_forward_count
                + int(finite["finite_native_forward_count"])
                + int(finite["finite_candidate_forward_count"])
            ),
            "signed_joint_fit_executed": True,
            "signed_joint_fold_count": v1._EXPECTED_FAMILIES,
            "signed_joint_actual_retained_directions_by_fold": {
                family: fit.rank for family, fit in sorted(signed_fits.items())
            },
            "signed_joint_streamed_coordinate_transform_logical_macs": (
                coordinate_transform_macs
            ),
            "signed_joint_streamed_operator_and_direction_score_logical_macs": (
                operator_and_score_macs
            ),
            "signed_joint_low_rank_U_factor_deflation_logical_macs": (
                low_rank_deflation_macs
            ),
            "signed_joint_low_rank_deflation_formula": (
                "per_step_3_k_C_squared_plus_3_k_squared_C_including_U_orthonormality_check"
            ),
            "signed_joint_dense_PMP_deflation_used": False,
            "signed_joint_symmetric_eigh_320_by_320_call_count": eigh_calls,
            "path_signed_joint_bounded_prefix_logical_macs_by_rank": (
                bounded_prefix_macs
            ),
            "matched_PCA_fold_count": v1._EXPECTED_FAMILIES,
            "matched_PCA_full_complement_projection_and_covariance_logical_macs": (
                pca_projection_and_covariance_macs
            ),
            "matched_PCA_q2_order_logical_macs": pca_q2_order_macs,
            "matched_PCA_signed_gain_rescore_logical_macs": pca_gain_rescore_macs,
            "matched_PCA_symmetric_eigh_320_by_320_call_count": (
                v1._EXPECTED_FAMILIES
            ),
            "endpoint_parent_arithmetic_frozen_under_prior_implementation": True,
            (
                "current_path_low_rank_deflation_is_tolerance_equivalent_"
                "not_bitwise_parent_replay"
            ): True,
        }
    )
    if result["total_model_forward_count"] != 272:
        raise RuntimeError("path total forward accounting differs")
    if result["finite_evaluation_model_forward_count"] != 160:
        raise RuntimeError("path finite forward accounting differs")
    return result


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
                "file_sha256": v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def _prompt_report_receipts(
    traces: Sequence[_LivePathTeacherKLTrace],
    evidence: Sequence[CompleteH4TailPathTeacherKLEvidence],
) -> tuple[dict[str, object], ...]:
    evidence_by_id = {value.example_id: value for value in evidence}
    rows: list[dict[str, object]] = []
    for trace in sorted(traces, key=lambda value: value.example_id):
        value = evidence_by_id[trace.example_id]
        value.validate_integrity()
        payload = dict(trace.prompt_path_payload)
        if (
            v1._domain_sha256(payload, domain=_PATH_PROMPT_RECEIPT_DOMAIN)
            != trace.prompt_path_receipt_sha256
        ):
            raise RuntimeError("path prompt execution receipt drifted")
        rows.append(
            {
                **payload,
                "prompt_path_receipt_sha256": trace.prompt_path_receipt_sha256,
                "path_evidence": value.metadata(),
                "finite_full_R_duck_endpoint": trace.endpoint.metadata(),
                "live_GL4_node_execution_receipts": tuple(
                    dict(row) for row in trace.path_node_receipts
                ),
                "raw_path_evidence_serialized": False,
            }
        )
    return tuple(rows)


def _path_vs_frozen_endpoint_comparison(
    *,
    endpoint_report: Mapping[str, object],
    path_arms: Sequence[Mapping[str, object]],
    path_behavioral_by_method: Mapping[
        str, Mapping[int, Mapping[str, object]]
    ],
    path_geometry_by_method: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    endpoint_ladders = _mapping(
        endpoint_report.get("finite_ladder_by_method"),
        label="frozen endpoint ladders",
    )
    endpoint_signed = endpoint_ladders.get("signed_joint")
    if not isinstance(endpoint_signed, (list, tuple)):
        raise ValueError("frozen endpoint signed ladder differs")
    endpoint_arms = {
        int(row["tail_rank"]): _mapping(row, label="frozen endpoint arm")
        for row in endpoint_signed
        if isinstance(row, Mapping)
    }
    path_by_rank = {int(row["tail_rank"]): row for row in path_arms}
    endpoint_behavior = _mapping(
        endpoint_report.get("established_behavioral_fidelity_by_method_rank"),
        label="frozen endpoint behavior",
    )
    endpoint_geometry = _mapping(
        endpoint_report.get("executed_cast_once_geometry_by_method_rank"),
        label="frozen endpoint geometry",
    )
    rows: list[dict[str, object]] = []
    for rank in PATH_RANKS:
        method = "shared_exact_sentinel" if rank == v1._D_RANK else "signed_joint"
        path_arm = path_by_rank[rank]
        endpoint_arm = endpoint_arms[rank]
        path_rank_behavior = path_behavioral_by_method[method][rank]
        endpoint_method_behavior = _mapping(
            endpoint_behavior.get(method),
            label="frozen endpoint comparison behavior method",
        )
        endpoint_rank = _mapping(
            endpoint_method_behavior.get(str(rank)),
            label="frozen endpoint behavior rank",
        )
        behavior_deltas: dict[str, object] = {}
        for ledger in (
            "ordinary",
            "complete_h4_support",
            "graph_core",
            "causal_tail",
        ):
            path_report = _mapping(
                path_rank_behavior[ledger], label="path behavior ledger"
            )
            endpoint_report_ledger = _mapping(
                endpoint_rank.get(ledger), label="frozen endpoint behavior ledger"
            )
            path_aggregate = _mapping(
                path_report.get("aggregate"), label="path behavior aggregate"
            )
            endpoint_aggregate = _mapping(
                endpoint_report_ledger.get("aggregate"),
                label="endpoint behavior aggregate",
            )
            behavior_deltas[ledger] = {
                "path_minus_endpoint_absolute_delta_nll_per_token": (
                    abs(float(path_aggregate["delta_nll_per_token"]))
                    - abs(float(endpoint_aggregate["delta_nll_per_token"]))
                ),
                "path_minus_endpoint_source_to_candidate_kl_per_token": (
                    float(path_aggregate["source_to_candidate_kl_per_token"])
                    - float(endpoint_aggregate["source_to_candidate_kl_per_token"])
                ),
                "path_minus_endpoint_top1_agreement": (
                    float(path_aggregate["top1_agreement_to_source"])
                    - float(endpoint_aggregate["top1_agreement_to_source"])
                ),
            }
        path_pooled = _mapping(
            path_geometry_by_method[method][rank].get("pooled"),
            label="path pooled geometry",
        )
        endpoint_method_geometry = _mapping(
            endpoint_geometry.get(method),
            label="frozen endpoint comparison geometry method",
        )
        endpoint_rank_geometry = _mapping(
            endpoint_method_geometry.get(str(rank)),
            label="frozen endpoint geometry rank",
        )
        endpoint_pooled = _mapping(
            endpoint_rank_geometry.get("pooled"),
            label="frozen endpoint pooled geometry",
        )
        geometry_deltas = {
            stratum: {
                "path_minus_endpoint_cosine": (
                    float(_mapping(path_pooled[stratum], label="path stratum")["cosine"])
                    - float(
                        _mapping(
                            endpoint_pooled[stratum], label="endpoint stratum"
                        )["cosine"]
                    )
                ),
                "path_minus_endpoint_normalized_rmse": (
                    float(
                        _mapping(path_pooled[stratum], label="path stratum")[
                            "normalized_rmse"
                        ]
                    )
                    - float(
                        _mapping(
                            endpoint_pooled[stratum], label="endpoint stratum"
                        )["normalized_rmse"]
                    )
                ),
            }
            for stratum in ("full", "graph_core", "causal_tail")
        }
        row: dict[str, object] = {
            "rank": rank,
            "path_minus_endpoint_family_macro_endpoint_rmse_after": (
                float(path_arm["family_macro_endpoint_rmse_after"])
                - float(endpoint_arm["family_macro_endpoint_rmse_after"])
            ),
            "path_minus_endpoint_family_macro_absolute_nll_gap_after": (
                float(path_arm["family_macro_absolute_nll_gap_after"])
                - float(endpoint_arm["family_macro_absolute_nll_gap_after"])
            ),
            "behavioral_deltas": behavior_deltas,
            "geometry_deltas": geometry_deltas,
            "negative_error_or_KL_delta_favors_path": True,
            "positive_top1_or_cosine_delta_favors_path": True,
            "endpoint_parent_is_frozen_not_bitwise_refit": True,
        }
        row["comparison_sha256"] = v1._domain_sha256(
            row, domain=_PATH_COMPARISON_DOMAIN
        )
        rows.append(row)
    return tuple(rows)


def _common_report(
    *,
    destination: Path,
    endpoint_report_path: Path | str,
    endpoint_report: Mapping[str, object],
    expected_endpoint_report_file_sha256: str,
    expected_endpoint_report_sha256: str,
    materialization_report_path: Path | str,
    transfer_report_path: Path | str,
    materialization: Mapping[str, object],
    materialization_binding: Mapping[str, object],
    basis_binding: Mapping[str, str],
    traces: Sequence[_LivePathTeacherKLTrace],
    evidence: Sequence[CompleteH4TailPathTeacherKLEvidence],
    closure: object,
    closure_gates: Mapping[str, bool],
    closure_diagnostics: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_hypothesis_use_only",
            "split": "whole_family_leave_one_out_after_global_GL4_closure",
            "frozen_d320_was_fit_on_all_a16_families": True,
            "end_to_end_candidate_is_family_disjoint": False,
            "frozen_supported_basis_rank": v1._D_RANK,
            "tail_width": v1._D_RANK,
            "requested_tail_ranks": PATH_RANKS,
            "teacher_objective": "token_KL(native_teacher||path_candidate)",
            "path_source": "authenticated_D320_complete_H4",
            "path_destination": "authenticated_native_complete_H4",
            "path_geometry": (
                "base_plus_P_D_R_plus_alpha_times_E_where_E_equals_I_minus_P_D_times_R"
            ),
            "quadrature_rule": "gauss_legendre_order_4_on_unit_interval",
            "quadrature_nodes_hex": tuple(
                value.hex() for value in GL4_UNIT_INTERVAL_NODES
            ),
            "quadrature_weights_hex": tuple(
                value.hex() for value in GL4_UNIT_INTERVAL_WEIGHTS
            ),
            "FTC_orientation": "native_KL_minus_D320_KL_equals_path_integral",
            "closure_is_hard_pre_fit_precondition": True,
            "closure_thresholds": {
                "family_macro_relative_rmse_max": 0.05,
                "each_family_relative_rmse_max": 0.10,
                "family_macro_cosine_min": 0.99,
                "direct_complete_complement_max_abs_error": 1.0e-9,
                "direct_complete_complement_relative_l2_error": 1.0e-9,
            },
            "pure_path_evidence_residual": "E_equals_null_D320_tail",
            "finite_duck_endpoint_residual": "full_R_equals_native_minus_base",
            "finite_duck_complement_coordinate_equivalence_required": True,
            "signed_joint_order": (
                "training_only_equal_family_prompt_token_GL4_path_integrated_"
                "signed_joint_greedy_rotation"
            ),
            "signed_joint_gain_constraint": "closed_interval_0_2",
            "same_objective_control": (
                "training_only_E_tail_PCA_then_GL4_integrated_teacher_KL_q2_"
                "order_with_matched_signed_gains"
            ),
            "finite_signed_arm": "D320_plus_gain_scaled_path_signed_joint_tail_prefix",
            "finite_control_arm": "D320_plus_gain_scaled_path_fixed_PCA_q2_tail_prefix",
            "shared_k320_arm": "D320_plus_ungained_exact_orthogonal_complement_tail",
            "shared_k320_exact_residual_provider_substitution": False,
            "finite_shadow_ledgers": {
                "ordinary": v1._EXPECTED_ORDINARY_TOKENS,
                "complete_h4_support": v1._EXPECTED_SUPPORT_TOKENS,
                "graph_core": v1._EXPECTED_GRAPH_CORE_TOKENS,
                "causal_tail": v1._EXPECTED_CAUSAL_TAIL_TOKENS,
            },
        },
        "input_binding": {
            "endpoint_signed_report_file": str(endpoint_report_path),
            "endpoint_signed_report_file_sha256": (
                expected_endpoint_report_file_sha256
            ),
            "endpoint_signed_report_sha256": expected_endpoint_report_sha256,
            "endpoint_signed_report_schema": endpoint_report.get("schema"),
            "endpoint_signed_report_classification": endpoint_report.get(
                "classification"
            ),
            "endpoint_signed_report_passed": endpoint_report.get("passed"),
            "endpoint_parent_frozen_under_prior_dense_deflation_arithmetic": True,
            "endpoint_parent_not_refit_under_current_low_rank_arithmetic": True,
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": dict(materialization_binding),
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "FTC_closure": getattr(closure, "metadata")(),
        "FTC_closure_gate_results": tuple(sorted(closure_gates.items())),
        "FTC_closure_diagnostics": dict(closure_diagnostics),
        "prompt_path_receipts": _prompt_report_receipts(traces, evidence),
        "resources": dict(resources),
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "tail_direction_order_and_gain_whole_family_disjoint": True,
            "frozen_d320_contains_same_a_held_family_information": True,
            "native_teacher_and_tail_used_for_path_evidence": True,
            "end_to_end_candidate_family_disjoint": False,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "endpoint_parent_is_frozen_historical_control": True,
            "current_low_rank_deflation_is_tolerance_equivalent_not_bitwise_parent_replay": True,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "contains_direction_or_basis_tensors": False,
            "contains_GL4_node_gradient_banks": False,
            "contains_token_score_matrices": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
            "artifact_must_remain_outside_git": True,
        },
    }


def run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic(
    *,
    endpoint_report_path: Path | str = DEFAULT_ENDPOINT_REPORT,
    expected_endpoint_report_file_sha256: str = ENDPOINT_REPORT_FILE_SHA256,
    expected_endpoint_report_sha256: str = ENDPOINT_REPORT_SHA256,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned GL4 path-integrated teacher-KL signed-joint screen."""

    destination = v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite teacher-KL GL4 path report")
    endpoint_report, endpoint_prompt_receipts = _load_endpoint_report(
        endpoint_report_path,
        expected_file_sha256=expected_endpoint_report_file_sha256,
        expected_report_sha256=expected_endpoint_report_sha256,
    )
    materialization = v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=v1.MATERIALIZATION_REPORT_SHA256,
        label="rank320 materialization",
    )
    transfer = v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=v1.TRANSFER_REPORT_SHA256,
        label="rank320 transfer",
    )
    transfer_receipts = endpoint._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    observations: list[dict[str, object]] | None = None
    finite_resources: dict[str, int] | None = None
    behavioral: dict[str, dict[int, dict[str, object]]] | None = None
    geometry: dict[str, dict[int, dict[str, object]]] | None = None
    signed_fits: dict[str, CompleteH4TailSignedJointHeldFamilyFit] | None = None
    pca_control_fits: dict[str, endpoint._FixedPCASignedControlFit] | None = None
    try:
        traces, path_evidence, collection_resources = _collect_path_teacher_kl_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
            endpoint_prompt_receipts=endpoint_prompt_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if (
            len(traces) != v1._EXPECTED_EXAMPLES
            or len(path_evidence) != v1._EXPECTED_EXAMPLES
            or len(families) != v1._EXPECTED_FAMILIES
        ):
            raise RuntimeError("A16 GL4 path panel shape differs")
        closure = summarize_complete_h4_tail_path_ftc_closure(path_evidence)
        closure_gates, closure_diagnostics = _closure_gate_results(
            closure,
            evidence=path_evidence,
            supported_basis=basis,
            traces=traces,
        )
        closure_passed = all(closure_gates.values())
        if closure_passed:
            path_fit_endpoints = tuple(
                complete_h4_tail_path_as_endpoint_example(value)
                for value in path_evidence
            )
            signed_fits = {
                family: fit_complete_h4_tail_signed_joint_held_family(
                    path_fit_endpoints,
                    supported_basis=basis,
                    held_family_id=family,
                    max_directions=_PATH_MAX_RANK,
                )
                for family in families
            }
            pca_control_fits = {
                family: endpoint._fit_fixed_pca_signed_control(
                    path_fit_endpoints,
                    supported_basis=basis,
                    held_family_id=family,
                )
                for family in families
            }
            observations, finite_resources, behavioral, geometry = (
                endpoint._finite_dual_arm_observations(
                    context=context,
                    traces=traces,
                    basis=basis,
                    signed_fits=signed_fits,
                    pca_control_fits=pca_control_fits,
                )
            )
        context.validate_immutable_inputs()
    finally:
        context.close()

    if not closure_passed:
        resources = _path_resource_accounting(
            traces, collection_resources=collection_resources
        )
        report = _common_report(
            destination=destination,
            endpoint_report_path=endpoint_report_path,
            endpoint_report=endpoint_report,
            expected_endpoint_report_file_sha256=(
                expected_endpoint_report_file_sha256
            ),
            expected_endpoint_report_sha256=expected_endpoint_report_sha256,
            materialization_report_path=materialization_report_path,
            transfer_report_path=transfer_report_path,
            materialization=materialization,
            materialization_binding=materialization_binding,
            basis_binding=basis_binding,
            traces=traces,
            evidence=path_evidence,
            closure=closure,
            closure_gates=closure_gates,
            closure_diagnostics=closure_diagnostics,
            resources=resources,
        )
        report.update(
            {
                "fit_and_finite_evaluation_status": {
                    "signed_joint_fit_executed": False,
                    "matched_PCA_fit_executed": False,
                    "finite_ladder_executed": False,
                    "reason": "GL4_FTC_closure_hard_precondition_failed",
                    "no_fit_or_finite_fields_populated": True,
                },
                "primary_gate_results": tuple(sorted(closure_gates.items())),
                "passed": False,
                "classification": "same_a_GL4_FTC_closure_not_supported",
            }
        )
        return _publish(report, output=destination)

    if (
        observations is None
        or finite_resources is None
        or behavioral is None
        or geometry is None
        or signed_fits is None
        or pca_control_fits is None
    ):
        raise RuntimeError("path closure passed without finite evaluation outputs")
    observation_set_sha256 = endpoint._finite_observation_set_sha256(observations)
    arms_by_method, secondary_gates_by_method = (
        endpoint._summarize_dual_arm_observations(observations)
    )
    paired_path_algorithm_comparison = endpoint._paired_method_comparison(
        arms_by_method=arms_by_method,
        behavioral=behavioral,
        geometry=geometry,
    )
    direction_availability = {
        rank: all(fit.rank >= rank for fit in signed_fits.values())
        for rank in PATH_RANKS[:-1]
    }
    fidelity_and_geometry_pass: dict[str, dict[int, bool]] = {
        "signed_joint": {},
        "fixed_pca_diagonal": {},
        "shared_exact_sentinel": {},
    }
    for method in ("signed_joint", "fixed_pca_diagonal"):
        for rank in PATH_RANKS[:-1]:
            fidelity_and_geometry_pass[method][rank] = (
                bool(geometry[method][rank]["gates"]["passed"])
                and all(
                    bool(behavioral[method][rank][ledger]["gates"]["passed"])
                    for ledger in (
                        "ordinary",
                        "complete_h4_support",
                        "graph_core",
                        "causal_tail",
                    )
                )
            )
    sentinel_passed = (
        bool(geometry["shared_exact_sentinel"][v1._D_RANK]["gates"]["passed"])
        and all(
            bool(
                behavioral["shared_exact_sentinel"][v1._D_RANK][ledger][
                    "gates"
                ]["passed"]
            )
            for ledger in (
                "ordinary",
                "complete_h4_support",
                "graph_core",
                "causal_tail",
            )
        )
    )
    fidelity_and_geometry_pass["shared_exact_sentinel"][v1._D_RANK] = (
        sentinel_passed
    )
    signed_passing_ranks = tuple(
        rank
        for rank in PATH_RANKS[:-1]
        if direction_availability[rank]
        and fidelity_and_geometry_pass["signed_joint"][rank]
    )
    pca_passing_ranks = tuple(
        rank
        for rank in PATH_RANKS[:-1]
        if fidelity_and_geometry_pass["fixed_pca_diagonal"][rank]
    )
    smallest_signed_rank = min(signed_passing_ranks) if signed_passing_ranks else None
    smallest_pca_rank = min(pca_passing_ranks) if pca_passing_ranks else None
    if smallest_signed_rank is None:
        matched_algorithm_classification = "path_signed_joint_no_bounded_gate_pass"
    elif smallest_pca_rank is None or smallest_signed_rank < smallest_pca_rank:
        matched_algorithm_classification = (
            "path_signed_joint_lower_rank_gate_advantage_over_matched_PCA_q2"
        )
    elif smallest_signed_rank == smallest_pca_rank:
        matched_algorithm_classification = (
            "path_signed_joint_and_matched_PCA_q2_first_pass_at_same_rank"
        )
    else:
        matched_algorithm_classification = (
            "path_matched_PCA_q2_lower_rank_gate_advantage"
        )
    sentinel_arm = next(
        row
        for row in arms_by_method["signed_joint"]
        if row["tail_rank"] == v1._D_RANK
    )
    primary_gates = {
        **closure_gates,
        "shared_k320_full_tail_reconstruction_at_most_1e_minus_9": (
            float(sentinel_arm["maximum_full_tail_reconstruction_abs_error"])
            <= 1.0e-9
        ),
        "shared_k320_every_prompt_h4_bitwise_native": bool(
            sentinel_arm["every_prompt_h4_bitwise_native"]
        ),
        "shared_k320_every_prompt_logits_bitwise_native": bool(
            sentinel_arm["every_prompt_logits_bitwise_native"]
        ),
        "shared_k320_clears_established_fidelity_and_geometry_gates": (
            sentinel_passed
        ),
        "at_least_one_available_path_signed_joint_k_le_64_clears_all_established_gates": (
            bool(signed_passing_ranks)
        ),
    }
    resources = _path_resource_accounting(
        traces,
        collection_resources=collection_resources,
        finite_resources=finite_resources,
        signed_fits=signed_fits,
    )
    path_vs_endpoint = _path_vs_frozen_endpoint_comparison(
        endpoint_report=endpoint_report,
        path_arms=arms_by_method["signed_joint"],
        path_behavioral_by_method=behavioral,
        path_geometry_by_method=geometry,
    )
    report = _common_report(
        destination=destination,
        endpoint_report_path=endpoint_report_path,
        endpoint_report=endpoint_report,
        expected_endpoint_report_file_sha256=expected_endpoint_report_file_sha256,
        expected_endpoint_report_sha256=expected_endpoint_report_sha256,
        materialization_report_path=materialization_report_path,
        transfer_report_path=transfer_report_path,
        materialization=materialization,
        materialization_binding=materialization_binding,
        basis_binding=basis_binding,
        traces=traces,
        evidence=path_evidence,
        closure=closure,
        closure_gates=closure_gates,
        closure_diagnostics=closure_diagnostics,
        resources=resources,
    )
    report.update(
        {
            "fit_and_finite_evaluation_status": {
                "signed_joint_fit_executed": True,
                "matched_PCA_fit_executed": True,
                "finite_ladder_executed": True,
                "reason": "GL4_FTC_closure_hard_precondition_passed",
            },
            "path_signed_joint_folds": tuple(
                signed_fits[family].metadata() for family in families
            ),
            "path_fixed_pca_same_objective_control_folds": tuple(
                pca_control_fits[family].metadata() for family in families
            ),
            "path_signed_joint_direction_availability_by_requested_rank": {
                str(rank): direction_availability[rank]
                for rank in PATH_RANKS[:-1]
            },
            "finite_ladder_by_method": arms_by_method,
            "established_behavioral_fidelity_by_method_rank": {
                method: {str(rank): value for rank, value in ranks.items()}
                for method, ranks in behavioral.items()
            },
            "executed_cast_once_geometry_by_method_rank": {
                method: {str(rank): value for rank, value in ranks.items()}
                for method, ranks in geometry.items()
            },
            "fidelity_and_geometry_pass_by_method_rank": {
                method: {str(rank): value for rank, value in ranks.items()}
                for method, ranks in fidelity_and_geometry_pass.items()
            },
            "smallest_path_signed_joint_rank_clearing_established_gates": (
                smallest_signed_rank
            ),
            "smallest_path_fixed_pca_control_rank_clearing_established_gates": (
                smallest_pca_rank
            ),
            "matched_path_teacher_kl_algorithm_comparison_classification": (
                matched_algorithm_classification
            ),
            "same_path_teacher_kl_paired_method_comparison": (
                paired_path_algorithm_comparison
            ),
            "path_vs_frozen_endpoint_signed_joint_comparison": path_vs_endpoint,
            "finite_observation_receipts": tuple(observations),
            "finite_observation_set_sha256": observation_set_sha256,
            "primary_gate_results": tuple(sorted(primary_gates.items())),
            "secondary_first_order_gate_results_by_method": (
                secondary_gates_by_method
            ),
            "passed": all(primary_gates.values()),
            "classification": (
                "same_a_GL4_path_teacher_kl_signed_joint_bounded_fidelity_supported"
                if all(primary_gates.values())
                else "same_a_GL4_path_teacher_kl_signed_joint_bounded_fidelity_not_supported"
            ),
        }
    )
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authenticated A16 GL4 path teacher-KL signed-joint ladder."
        )
    )
    parser.add_argument(
        "--endpoint-report", type=Path, default=DEFAULT_ENDPOINT_REPORT
    )
    parser.add_argument(
        "--endpoint-report-file-sha256",
        default=ENDPOINT_REPORT_FILE_SHA256,
        help="Expected byte SHA-256 of the frozen endpoint parent.",
    )
    parser.add_argument(
        "--endpoint-report-sha256",
        default=ENDPOINT_REPORT_SHA256,
        help="Expected logical report SHA-256 of the frozen endpoint parent.",
    )
    parser.add_argument(
        "--materialization-report", type=Path, default=DEFAULT_MATERIALIZATION_REPORT
    )
    parser.add_argument(
        "--transfer-report", type=Path, default=DEFAULT_TRANSFER_REPORT
    )
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = (
        run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic(
            endpoint_report_path=args.endpoint_report,
            expected_endpoint_report_file_sha256=(
                args.endpoint_report_file_sha256
            ),
            expected_endpoint_report_sha256=args.endpoint_report_sha256,
            materialization_report_path=args.materialization_report,
            transfer_report_path=args.transfer_report,
            basis_sidecar_path=args.basis_sidecar,
            output=args.output,
            cache_dir=args.cache_dir,
        )
    )
    print(f"wrote {result['artifact']['file']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
