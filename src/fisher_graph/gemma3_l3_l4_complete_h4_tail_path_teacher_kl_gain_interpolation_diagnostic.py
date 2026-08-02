"""Predeclared K64 gain-interpolation screen over the frozen GL4 path fit.

This v2 diagnostic does not alter the published GL4 path-v1 protocol.  It
recollects the four-node path evidence through v1's authenticated lower-level
collector and requires exact replay of the v1 closure, prompt receipts, and
all eight whole-family-LOFO signed-joint fits before any finite candidate is
executed.

For each held-family K64 fit, five variants are fixed in advance::

    g_beta = g + beta * (1 - g), beta in (0, .25, .5, .75, 1)

Thus beta=0 is the published gained K64 arm and beta=1 preserves the same
directions while removing attenuation.  Every beta is reported; held-family
outcomes do not select one.  A single ungained K320 exact-complement sentinel
is shared across the five summaries.

The native residual and teacher logits remain truth leaking.  This is a
same-Calibration-A hypothesis screen, not a serving provider, compression
artifact, or speed claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic as path_v1
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as endpoint
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from .complete_h4_tail_path_teacher_kl import (
    CompleteH4TailPathTeacherKLEvidence,
    complete_h4_tail_path_as_endpoint_example,
    summarize_complete_h4_tail_path_ftc_closure,
)
from .complete_h4_tail_signed_joint_projector import (
    CompleteH4TailSignedJointHeldFamilyFit,
    complete_h4_tail_signed_joint_scores,
    fit_complete_h4_tail_signed_joint_held_family,
)
from .complete_h4_tail_token_fisher import project_complete_h4_tail_rows
from .gemma3_l3_l4_complete_h4_one_pass_transfer import (
    _load_committed_basis,
    _native_boundary,
    _retokenize,
)
from .gemma3_l3_l4_complete_h4_projection import CompleteH4ProjectionFitSequence
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
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PATH_V1_REPORT",
    "GAIN_INTERPOLATION_BETAS",
    "PATH_V1_REPORT_FILE_SHA256",
    "PATH_V1_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_gain_interpolation_diagnostic",
    "main",
]


DEFAULT_PATH_V1_REPORT = path_v1.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "teacher-kl-path-gl4-signed-joint-k64-gain-interpolation-lofo-"
    "a-fit16-dev-v2.json"
)
PATH_V1_REPORT_FILE_SHA256 = (
    "1031cf3c11354e7e59a4bb6adf616eea9ea92f39e4b3619f015bcf4bb7bd91a2"
)
PATH_V1_REPORT_SHA256 = (
    "4fbb5b8106f4048753d484eb319db813236cb5b31289db864359ab6a48de1dc2"
)
GAIN_INTERPOLATION_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_K64 = 64
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_teacher_kl_path_gl4_"
    "signed_joint_k64_gain_interpolation_lofo.v2"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-path-gl4-k64-gain-interpolation:v2\0"
)
_PROVIDER_DOMAIN = (
    b"fisher-graph:complete-h4-path-gl4-k64-gain-interpolation-provider:v2\0"
)
_OBSERVATION_DOMAIN = (
    b"fisher-graph:complete-h4-path-gl4-k64-gain-interpolation-observation:v2\0"
)
_OBSERVATION_SET_DOMAIN = (
    b"fisher-graph:complete-h4-path-gl4-k64-gain-interpolation-observation-set:v2\0"
)
_REPLAY_DOMAIN = (
    b"fisher-graph:complete-h4-path-gl4-k64-gain-interpolation-replay:v2\0"
)
_BETA_IDS = ("beta_0", "beta_0_25", "beta_0_5", "beta_0_75", "beta_1")
_LEDGERS = (
    "ordinary",
    "complete_h4_support",
    "graph_core",
    "causal_tail",
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _json_identity(value: object) -> str:
    return token_v1._domain_sha256(value, domain=_REPLAY_DOMAIN)


def _beta_index(beta: float) -> int:
    if type(beta) is not float or beta not in GAIN_INTERPOLATION_BETAS:
        raise ValueError("beta must be one of the predeclared gain interpolants")
    return GAIN_INTERPOLATION_BETAS.index(beta)


def _beta_id(beta: float) -> str:
    return _BETA_IDS[_beta_index(beta)]


def _load_path_v1_report(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
) -> dict[str, object]:
    """Authenticate the completed GL4 path-v1 parent and its observation set."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=_require_sha256(
            expected_file_sha256, label="path-v1 report file"
        ),
        expected_report_sha256=_require_sha256(
            expected_report_sha256, label="path-v1 report payload"
        ),
        label="completed GL4 path-v1 parent",
    )
    protocol = _mapping(report.get("protocol"), label="path-v1 protocol")
    binding = _mapping(report.get("input_binding"), label="path-v1 input binding")
    fit_status = _mapping(
        report.get("fit_and_finite_evaluation_status"), label="path-v1 fit status"
    )
    science = _mapping(
        report.get("scientific_status"), label="path-v1 scientific status"
    )
    safety = _mapping(report.get("safety"), label="path-v1 safety")
    resources = _mapping(report.get("resources"), label="path-v1 resources")
    closure = _mapping(report.get("FTC_closure"), label="path-v1 FTC closure")
    folds = report.get("path_signed_joint_folds")
    prompts = report.get("prompt_path_receipts")
    raw_observations = report.get("finite_observation_receipts")
    raw_gates = report.get("FTC_closure_gate_results")
    if (
        report.get("schema") != path_v1._SCHEMA
        or report.get("classification")
        != "same_a_GL4_path_teacher_kl_signed_joint_bounded_fidelity_not_supported"
        or report.get("passed") is not False
        or tuple(protocol.get("requested_tail_ranks", ())) != path_v1.PATH_RANKS
        or protocol.get("quadrature_rule")
        != "gauss_legendre_order_4_on_unit_interval"
        or protocol.get("teacher_objective")
        != "token_KL(native_teacher||path_candidate)"
        or protocol.get("split")
        != "whole_family_leave_one_out_after_global_GL4_closure"
        or binding.get("endpoint_signed_report_file_sha256")
        != path_v1.ENDPOINT_REPORT_FILE_SHA256
        or binding.get("endpoint_signed_report_sha256")
        != path_v1.ENDPOINT_REPORT_SHA256
        or binding.get("materialization_report_file_sha256")
        != token_v1.MATERIALIZATION_REPORT_FILE_SHA256
        or binding.get("materialization_report_sha256")
        != token_v1.MATERIALIZATION_REPORT_SHA256
        or binding.get("transfer_report_file_sha256")
        != token_v1.TRANSFER_REPORT_FILE_SHA256
        or binding.get("transfer_report_sha256")
        != token_v1.TRANSFER_REPORT_SHA256
        or fit_status.get("signed_joint_fit_executed") is not True
        or fit_status.get("finite_ladder_executed") is not True
        or not isinstance(folds, (list, tuple))
        or len(folds) != token_v1._EXPECTED_FAMILIES
        or not isinstance(prompts, (list, tuple))
        or len(prompts) != token_v1._EXPECTED_EXAMPLES
        or not isinstance(raw_observations, (list, tuple))
        or not isinstance(raw_gates, (list, tuple))
        or not all(bool(value) for _name, value in raw_gates)
        or resources.get("collection_model_forward_count") != 112
        or resources.get("path_teacher_kl_vjp_backward_call_count") != 436
        or resources.get("total_model_forward_count") != 272
        or science.get("same_a_truth_leaking_hypothesis_use_only") is not True
        or science.get("candidate_serving_authorized") is not False
        or science.get("compression_claim") is not False
        or safety.get("contains_activation_tensors") is not False
        or safety.get("contains_gradient_tensors") is not False
        or safety.get("contains_direction_or_basis_tensors") is not False
        or safety.get("artifact_must_remain_outside_git") is not True
    ):
        raise ValueError("completed GL4 path-v1 parent semantics differ")
    fold_rows = tuple(_mapping(row, label="path-v1 signed fold") for row in folds)
    if (
        len({str(row.get("held_family_id")) for row in fold_rows})
        != token_v1._EXPECTED_FAMILIES
        or any(
            tuple(row.get("ambient_directions_shape", ()))
            != (_K64, token_v1._WIDTH)
            for row in fold_rows
        )
        or any(row.get("requested_max_directions") != _K64 for row in fold_rows)
    ):
        raise ValueError("path-v1 K64 signed fold grid differs")
    observations = [
        _mapping(row, label="path-v1 finite observation")
        for row in raw_observations
    ]
    if (
        endpoint._finite_observation_set_sha256(observations)
        != report.get("finite_observation_set_sha256")
    ):
        raise RuntimeError("path-v1 finite observation set drifted")
    evidence_hashes = tuple(closure.get("evidence_artifact_sha256s", ()))
    prompt_evidence_hashes = tuple(
        _mapping(row, label="path-v1 prompt receipt").get(
            "path_evidence_artifact_sha256"
        )
        for row in prompts
    )
    if (
        len(evidence_hashes) != token_v1._EXPECTED_EXAMPLES
        or tuple(sorted(evidence_hashes)) != tuple(sorted(prompt_evidence_hashes))
    ):
        raise ValueError("path-v1 closure/prompt evidence identities differ")
    return report


def _interpolated_gains(
    fit: CompleteH4TailSignedJointHeldFamilyFit, *, beta: float
) -> Tensor:
    fit.validate_integrity()
    _beta_index(beta)
    if fit.rank != _K64:
        raise ValueError("gain interpolation requires the exact K64 path fit")
    gains = torch.tensor(fit.gains, dtype=torch.float64)
    interpolated = gains + beta * (torch.ones_like(gains) - gains)
    if interpolated.shape != (_K64,) or not bool(torch.isfinite(interpolated).all()):
        raise RuntimeError("interpolated K64 gains are invalid")
    return interpolated.contiguous()


def _gain_interpolated_tail_and_prediction(
    *,
    tail_rows: Tensor,
    endpoint_example: object,
    fit: CompleteH4TailSignedJointHeldFamilyFit,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    tail = tail_rows.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if tail.ndim != 2 or tail.shape[1] != token_v1._WIDTH:
        raise ValueError("gain-interpolation tail geometry differs")
    gains = _interpolated_gains(fit, beta=beta)
    directions = fit.directions_tensor()
    prefix = (((tail @ directions.T) * gains) @ directions).contiguous()
    scores = complete_h4_tail_signed_joint_scores(endpoint_example, directions)
    prediction = (scores @ gains).contiguous()
    return prefix, prediction, gains


class _AuthenticatedGainInterpolationFiniteProvider(Gemma3L3L4CorrectionProvider):
    """Single-use beta-bound K64 provider or shared exact K320 sentinel."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "rank",
        "variant_id",
        "beta_index",
        "beta_hex",
        "interpolated_gains_sha256",
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
        beta: float | None,
        interpolated_gains: Tensor | None,
        fold_artifact_sha256: str,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        if rank == _K64:
            if beta is None or interpolated_gains is None:
                raise ValueError("K64 interpolation provider requires beta gains")
            index = _beta_index(beta)
            gains = interpolated_gains.detach().to(
                device="cpu", dtype=torch.float64
            ).clone().contiguous()
            if gains.shape != (_K64,) or not bool(torch.isfinite(gains).all()):
                raise ValueError("K64 interpolation gains differ")
            variant_id = _BETA_IDS[index]
            beta_hex: str | None = beta.hex()
            gains_sha: str | None = _runtime_tensor_sha256(gains)
        elif rank == token_v1._D_RANK:
            if beta is not None or interpolated_gains is not None:
                raise ValueError("K320 sentinel must be ungained")
            index = None
            variant_id = "shared_exact_sentinel"
            beta_hex = None
            gains_sha = None
        else:
            raise ValueError("gain-interpolation provider rank must be K64 or K320")
        if (
            not isinstance(base_h4, Tensor)
            or base_h4.ndim != 3
            or base_h4.shape[-1] != token_v1._WIDTH
            or not base_h4.is_floating_point()
            or not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not isinstance(correction, Tensor)
            or correction.shape != base_h4.shape
            or not correction.is_floating_point()
        ):
            raise ValueError("gain-interpolation provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        delta = correction.detach().to(
            device="cpu", dtype=torch.float64
        ).clone().contiguous()
        if not bool(torch.isfinite(delta).all()) or bool((delta[~support] != 0).any()):
            raise ValueError("gain-interpolation correction escapes support")
        self.site = token_v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.rank = rank
        self.variant_id = variant_id
        self.beta_index = index
        self.beta_hex = beta_hex
        self.interpolated_gains_sha256 = gains_sha
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="path K64 fold"
        )
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="gain-interpolation model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="gain-interpolation bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="gain-interpolation prefix"
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
            "schema": "fisher_graph.complete_h4_path_gain_interpolation_provider.v2",
            "site": self.site,
            "write_scope": self.write_scope,
            "rank": self.rank,
            "variant_id": self.variant_id,
            "beta_index": self.beta_index,
            "beta_hex": self.beta_hex,
            "interpolated_gains_sha256": self.interpolated_gains_sha256,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": (
                "P_D320_R_plus_same_path_signed_K64_directions_times_"
                "g_plus_beta_times_one_minus_g"
                if self.rank == _K64
                else "P_D320_R_plus_ungained_exact_orthogonal_complement_tail"
            ),
            "held_family_used_to_select_beta": False,
            "exact_residual_provider_substitution_used": False,
            "single_use": True,
            "truth_leaking_hypothesis_use_only": True,
            "serving_authorized": False,
        }

    def _computed_sha256(self) -> str:
        return token_v1._domain_sha256(self._payload(), domain=_PROVIDER_DOMAIN)

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
            raise RuntimeError("gain-interpolation provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("gain-interpolation provider cannot be reused")
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
            raise RuntimeError("gain-interpolation provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _exact_path_v1_replay_receipt(
    *,
    parent_report: Mapping[str, object],
    traces: Sequence[object],
    evidence: Sequence[CompleteH4TailPathTeacherKLEvidence],
    closure: object,
    closure_gates: Mapping[str, bool],
    closure_diagnostics: Mapping[str, object],
    collection_resources: Mapping[str, int],
    signed_fits: Mapping[str, CompleteH4TailSignedJointHeldFamilyFit],
) -> dict[str, object]:
    """Fail unless recollection and K64 refit exactly reproduce path-v1."""

    fresh_prompts = path_v1._prompt_report_receipts(traces, evidence)
    parent_prompts = parent_report.get("prompt_path_receipts")
    fresh_folds = tuple(
        signed_fits[family].metadata() for family in sorted(signed_fits)
    )
    parent_folds = parent_report.get("path_signed_joint_folds")
    fresh_closure = getattr(closure, "metadata")()
    parent_closure = parent_report.get("FTC_closure")
    fresh_gate_rows = tuple(sorted(closure_gates.items()))
    parent_gate_rows = parent_report.get("FTC_closure_gate_results")
    parent_diagnostics = parent_report.get("FTC_closure_diagnostics")
    parent_resources = _mapping(
        parent_report.get("resources"), label="path-v1 replay resources"
    )
    expected_collection = {
        key: parent_resources.get(key) for key in collection_resources
    }
    checks = {
        "closure_metadata_exact": _json_identity(fresh_closure)
        == _json_identity(parent_closure),
        "closure_gates_exact": _json_identity(fresh_gate_rows)
        == _json_identity(parent_gate_rows),
        "closure_diagnostics_exact": _json_identity(dict(closure_diagnostics))
        == _json_identity(parent_diagnostics),
        "prompt_and_evidence_receipts_exact": _json_identity(fresh_prompts)
        == _json_identity(parent_prompts),
        "signed_joint_K64_folds_exact": _json_identity(fresh_folds)
        == _json_identity(parent_folds),
        "collection_resources_exact": dict(collection_resources)
        == expected_collection,
        "all_eight_fresh_fits_reach_exact_K64": (
            len(signed_fits) == token_v1._EXPECTED_FAMILIES
            and all(fit.rank == _K64 for fit in signed_fits.values())
        ),
        "all_fresh_closure_gates_pass": all(closure_gates.values()),
    }
    if not all(checks.values()):
        failed = ", ".join(key for key, value in checks.items() if not value)
        raise RuntimeError(
            f"exact GL4 path-v1 replay failed before finite evaluation: {failed}"
        )
    receipt: dict[str, object] = {
        "parent_file_sha256": PATH_V1_REPORT_FILE_SHA256,
        "parent_report_sha256": PATH_V1_REPORT_SHA256,
        "checks": tuple(sorted(checks.items())),
        "fresh_closure_artifact_sha256": fresh_closure.get("artifact_sha256"),
        "fresh_evidence_artifact_sha256s": tuple(
            value.artifact_sha256 for value in evidence
        ),
        "fresh_prompt_receipt_set_sha256": _json_identity(fresh_prompts),
        "fresh_fit_artifact_sha256s": tuple(
            signed_fits[family].artifact_sha256 for family in sorted(signed_fits)
        ),
        "finite_evaluation_executed_before_exact_replay": False,
    }
    receipt["replay_receipt_sha256"] = token_v1._domain_sha256(
        receipt, domain=_REPLAY_DOMAIN
    )
    return receipt


def _finite_gain_interpolation_observations(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    signed_fits: Mapping[str, CompleteH4TailSignedJointHeldFamilyFit],
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    """Execute five predeclared K64 gain variants plus one shared K320."""

    adapter = getattr(context, "adapter")
    bridge = getattr(context, "bridge")
    tokenize = getattr(context, "tokenize")
    keys = _BETA_IDS + ("shared_exact_sentinel",)
    manifests = {
        ledger: {
            trace.example_id: trace.family_id
            for trace in traces
            if trace.selected_by_ledger[ledger].numel() > 0
        }
        for ledger in _LEDGERS
    }
    fidelity = {
        key: {
            ledger: SourceAuthoritativeShadowFidelityAccumulator(
                manifests[ledger], gates=ESTABLISHED_SHADOW_FIDELITY_GATES
            )
            for ledger in _LEDGERS
        }
        for key in keys
    }
    geometry_traces = [
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
        for trace in traces
    ]
    executed_rows: dict[str, dict[str, Tensor]] = {key: {} for key in keys}
    observations: list[dict[str, object]] = []
    forward_by_key = {key: 0 for key in keys}
    native_forwards = 0

    for trace in traces:
        fit = signed_fits[trace.family_id]
        fit.validate_integrity()
        if fit.rank != _K64:
            raise RuntimeError("finite interpolation received a non-K64 fit")
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
            raise RuntimeError("gain-interpolation finite retokenization drifted")
        source_logits, native_h4, native_positions, native_valid = _native_boundary(
            adapter, model_inputs
        )
        native_forwards += 1
        if (
            _runtime_tensor_sha256(source_logits) != trace.native_logits_sha256
            or not token_v1._bitwise_equal(
                native_h4.detach().to(device="cpu"), trace.native_h4
            )
            or not token_v1._bitwise_equal(
                native_positions, trace.prefix.logical_positions
            )
            or not token_v1._bitwise_equal(
                native_valid, trace.prefix.valid_target_mask
            )
        ):
            raise RuntimeError("gain-interpolation finite native boundary drifted")
        source_selected = frozen._select_sequence_rows(source_logits, indices)
        support = trace.prefix.complete_h4_causal_support_mask().detach().to(
            device="cpu"
        )
        arms: list[
            tuple[str, int, float | None, Tensor, Tensor, Tensor | None]
        ] = []
        for beta in GAIN_INTERPOLATION_BETAS:
            prefix_rows, prediction, gains = _gain_interpolated_tail_and_prediction(
                tail_rows=tail_rows,
                endpoint_example=trace.endpoint,
                fit=fit,
                beta=beta,
            )
            arms.append((_beta_id(beta), _K64, beta, prefix_rows, prediction, gains))
        sentinel_prediction = torch.einsum(
            "rw,trw->t", tail_rows, trace.endpoint.token_h4_gradients
        ).contiguous()
        arms.append(
            (
                "shared_exact_sentinel",
                token_v1._D_RANK,
                None,
                tail_rows.clone().contiguous(),
                sentinel_prediction,
                None,
            )
        )

        for variant_id, rank, beta, tail_prefix, prediction, gains in arms:
            correction_rows = (supported_rows + tail_prefix).contiguous()
            correction = torch.zeros(
                trace.base_h4.shape, dtype=torch.float64, device="cpu"
            )
            correction[0].index_copy_(0, trace.support_indices, correction_rows)
            provider = _AuthenticatedGainInterpolationFiniteProvider(
                rank=rank,
                beta=beta,
                interpolated_gains=gains,
                fold_artifact_sha256=fit.artifact_sha256,
                model_inputs_sha256=trace.model_inputs_sha256,
                bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
                prefix_artifact_sha256=trace.prefix.artifact_sha256,
                base_h4=trace.base_h4,
                support_mask=support,
                correction=correction,
            )
            execution = bridge.execute(adapter, model_inputs, h4_head=provider)
            forward_by_key[variant_id] += 1
            provider.validate_integrity()
            if (
                getattr(execution, "model_forward_count", None) != 1
                or not provider.used
                or execution.model_inputs_sha256 != trace.model_inputs_sha256
                or execution.bridge_binding_sha256
                != trace.prefix.bridge_binding_sha256
                or execution.prefix.artifact_sha256 != trace.prefix.artifact_sha256
                or execution.h4_head_sha256 != provider.artifact_sha256
                or _runtime_tensor_sha256(execution.candidate_x4)
                != trace.base_x4_sha256
            ):
                raise RuntimeError("gain-interpolation finite execution binding differs")
            expected_h4 = trace.base_h4.clone()
            support_live = trace.support_indices.to(expected_h4.device)
            expected_h4[0].index_copy_(
                0,
                support_live,
                (
                    trace.base_h4[0]
                    .index_select(0, support_live)
                    .to(dtype=torch.float64)
                    + correction_rows.to(trace.base_h4.device)
                ).to(dtype=trace.base_h4.dtype),
            )
            if not token_v1._bitwise_equal(
                execution.candidate_h4.detach(), expected_h4
            ):
                raise RuntimeError("gain-interpolation finite H4 differs")
            candidate_nll = token_v1._selected_token_nll(
                execution.logits, indices, targets
            )
            endpoint_selection = trace.selected_by_ledger["complete_h4_support"]
            endpoint_indices = indices.index_select(
                0, endpoint_selection.to(indices.device)
            )
            endpoint_targets = targets.index_select(
                0, endpoint_selection.to(targets.device)
            )
            candidate_endpoint_nll = token_v1._selected_token_nll(
                execution.logits, endpoint_indices, endpoint_targets
            )
            candidate_endpoint_kl = endpoint._selected_token_teacher_kl(
                source_logits, execution.logits, endpoint_indices
            )
            candidate_selected = frozen._select_sequence_rows(
                execution.logits, indices
            )
            for ledger, selected in trace.selected_by_ledger.items():
                if selected.numel() == 0:
                    continue
                fidelity[variant_id][ledger].add(
                    ShadowFidelityExample(
                        example_id=trace.example_id,
                        family_id=trace.family_id,
                        source_logits=source_selected.index_select(
                            0, selected.to(source_selected.device)
                        ),
                        candidate_logits=candidate_selected.index_select(
                            0, selected.to(candidate_selected.device)
                        ),
                        targets=targets.index_select(0, selected.to(targets.device)),
                    )
                )
            actual_rows = (
                execution.candidate_h4.detach().to(device="cpu", dtype=torch.float64)[0]
                .index_select(0, trace.support_indices)
                - trace.base_h4.to(device="cpu", dtype=torch.float64)[0].index_select(
                    0, trace.support_indices
                )
            ).contiguous()
            executed_rows[variant_id][trace.example_id] = actual_rows
            observation: dict[str, object] = {
                "example_id": trace.example_id,
                "family_id": trace.family_id,
                "method": (
                    "gain_interpolated_path_signed_K64"
                    if rank == _K64
                    else "shared_exact_sentinel"
                ),
                "variant_id": variant_id,
                "rank": rank,
                "requested_rank": rank,
                "beta_index": _beta_index(beta) if beta is not None else None,
                "beta_hex": beta.hex() if beta is not None else None,
                "gain_formula": (
                    "g_plus_beta_times_one_minus_g" if beta is not None else None
                ),
                "interpolated_gains_sha256": (
                    _runtime_tensor_sha256(gains) if gains is not None else None
                ),
                "effective_direction_count": rank,
                "fold_artifact_sha256": fit.artifact_sha256,
                "provider_artifact_sha256": provider.artifact_sha256,
                "execution_artifact_sha256": execution.artifact_sha256,
                "native_mean_nll": float(trace.native_token_nll.mean()),
                "d320_mean_nll": float(trace.d320_token_nll.mean()),
                "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
                "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
                "d320_mean_teacher_kl": float(
                    (-trace.endpoint.compensation_target).mean()
                ),
                "candidate_mean_teacher_kl": float(candidate_endpoint_kl.mean()),
                "endpoint_baseline_mse": float(
                    trace.endpoint.compensation_target.square().mean()
                ),
                "endpoint_prediction_mse": float(
                    (prediction - trace.endpoint.compensation_target).square().mean()
                ),
                "candidate_h4_bitwise_native": token_v1._bitwise_equal(
                    execution.candidate_h4.detach().to(device="cpu"), trace.native_h4
                ),
                "candidate_logits_bitwise_native": (
                    _runtime_tensor_sha256(execution.logits)
                    == trace.native_logits_sha256
                ),
                "full_tail_reconstruction_max_abs_error": (
                    float((tail_prefix - tail_rows).abs().max())
                    if rank == token_v1._D_RANK
                    else None
                ),
                "exact_residual_provider_substitution_used": False,
                "executed_correction_rows_sha256": _runtime_tensor_sha256(actual_rows),
            }
            observation["observation_sha256"] = token_v1._domain_sha256(
                observation, domain=_OBSERVATION_DOMAIN
            )
            observations.append(observation)
            del execution, provider, correction, candidate_selected
        del model_inputs, source_logits, source_selected, native_h4

    behavioral = {
        key: {ledger: fidelity[key][ledger].finalize() for ledger in _LEDGERS}
        for key in keys
    }
    geometry = {
        key: token_v1.ladder._geometry_with_examples(
            geometry_traces,
            executed_rows[key],
            candidate_semantics=(
                "actual_cast_once_d320_plus_signed_joint_teacher_kl_tail_k64"
                if key == "beta_0"
                else (
                    "actual_cast_once_d320_plus_shared_exact_sentinel_"
                    "teacher_kl_tail_k320"
                    if key == "shared_exact_sentinel"
                    else f"actual_cast_once_d320_plus_{key}_path_signed_tail_k64"
                )
            ),
        )
        for key in keys
    }
    resources = {
        "finite_native_forward_count": native_forwards,
        "finite_candidate_forward_count": sum(forward_by_key.values()),
        "finite_gain_interpolation_forward_count": sum(
            forward_by_key[key] for key in _BETA_IDS
        ),
        "finite_shared_exact_sentinel_forward_count": forward_by_key[
            "shared_exact_sentinel"
        ],
        "finite_forward_count_by_variant": dict(forward_by_key),
    }
    if (
        native_forwards != token_v1._EXPECTED_EXAMPLES
        or resources["finite_candidate_forward_count"] != 96
        or resources["finite_gain_interpolation_forward_count"] != 80
        or resources["finite_shared_exact_sentinel_forward_count"] != 16
        or any(value != 16 for value in forward_by_key.values())
    ):
        raise RuntimeError("gain-interpolation finite forward accounting differs")
    return observations, resources, behavioral, geometry


def _finite_observation_set_sha256(
    observations: Sequence[Mapping[str, object]],
) -> str:
    expected = {
        **{key: (_K64, index) for index, key in enumerate(_BETA_IDS)},
        "shared_exact_sentinel": (token_v1._D_RANK, None),
    }
    if len(observations) != token_v1._EXPECTED_EXAMPLES * len(expected):
        raise ValueError("gain-interpolation observation count differs")
    identities: dict[tuple[str, str], str] = {}
    families: dict[str, str] = {}
    for raw in observations:
        row = dict(raw)
        receipt = row.pop("observation_sha256", None)
        example_id = token_v1._identifier(
            row.get("example_id"), label="gain-interpolation observation example"
        )
        family_id = token_v1._identifier(
            row.get("family_id"), label="gain-interpolation observation family"
        )
        variant_id = row.get("variant_id")
        if not isinstance(variant_id, str) or variant_id not in expected:
            raise ValueError("gain-interpolation observation variant differs")
        rank, beta_index = expected[variant_id]
        expected_method = (
            "shared_exact_sentinel"
            if beta_index is None
            else "gain_interpolated_path_signed_K64"
        )
        if (
            row.get("rank") != rank
            or row.get("requested_rank") != rank
            or row.get("effective_direction_count") != rank
            or row.get("method") != expected_method
            or row.get("beta_index") != beta_index
            or (
                beta_index is not None
                and row.get("beta_hex")
                != GAIN_INTERPOLATION_BETAS[beta_index].hex()
            )
            or (beta_index is None and row.get("beta_hex") is not None)
            or (
                beta_index is not None
                and row.get("gain_formula")
                != "g_plus_beta_times_one_minus_g"
            )
            or (beta_index is None and row.get("gain_formula") is not None)
            or (
                beta_index is not None
                and not isinstance(row.get("interpolated_gains_sha256"), str)
            )
            or (
                beta_index is None
                and row.get("interpolated_gains_sha256") is not None
            )
        ):
            raise ValueError("gain-interpolation observation protocol differs")
        identity = (example_id, variant_id)
        if identity in identities:
            raise ValueError("gain-interpolation observation grid has a duplicate")
        if families.setdefault(example_id, family_id) != family_id:
            raise ValueError("gain-interpolation observation family drifted")
        expected_receipt = token_v1._domain_sha256(row, domain=_OBSERVATION_DOMAIN)
        if receipt != expected_receipt:
            raise RuntimeError("gain-interpolation observation receipt drifted")
        identities[identity] = expected_receipt
    if (
        len(families) != token_v1._EXPECTED_EXAMPLES
        or any(
            (example_id, variant_id) not in identities
            for example_id in families
            for variant_id in expected
        )
    ):
        raise ValueError("gain-interpolation observation grid is incomplete")
    return token_v1._domain_sha256(
        tuple((*key, identities[key]) for key in sorted(identities)),
        domain=_OBSERVATION_SET_DOMAIN,
    )


def _summarize_by_beta(
    observations: Sequence[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, bool]]]:
    sentinel = [
        row for row in observations if row["variant_id"] == "shared_exact_sentinel"
    ]
    ladders: dict[str, list[dict[str, object]]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for variant_id in _BETA_IDS:
        selected = [row for row in observations if row["variant_id"] == variant_id]
        arms, arm_gates = token_v1._summarize_observations(
            selected + sentinel, ranks=(_K64, token_v1._D_RANK)
        )
        ladders[variant_id] = arms
        gates[variant_id] = arm_gates
    return ladders, gates


def _stable_parent_beta_zero_comparison(
    *,
    parent_report: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
    ladders: Mapping[str, Sequence[Mapping[str, object]]],
    behavioral: Mapping[str, Mapping[str, object]],
    geometry: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    parent_observations = parent_report.get("finite_observation_receipts")
    if not isinstance(parent_observations, (list, tuple)):
        raise ValueError("path-v1 beta-zero observations differ")
    parent_k64 = {
        str(row["example_id"]): _mapping(row, label="path-v1 K64 observation")
        for row in parent_observations
        if isinstance(row, Mapping)
        and row.get("method") == "signed_joint"
        and row.get("rank") == _K64
    }
    fresh_k64 = {
        str(row["example_id"]): row
        for row in observations
        if row.get("variant_id") == "beta_0"
    }
    stable_fields = (
        "family_id",
        "native_mean_nll",
        "d320_mean_nll",
        "candidate_mean_nll",
        "ordinary_candidate_mean_nll",
        "d320_mean_teacher_kl",
        "candidate_mean_teacher_kl",
        "endpoint_baseline_mse",
        "endpoint_prediction_mse",
        "candidate_h4_bitwise_native",
        "candidate_logits_bitwise_native",
        "full_tail_reconstruction_max_abs_error",
        "executed_correction_rows_sha256",
    )
    prompt_metrics_exact = (
        parent_k64.keys() == fresh_k64.keys()
        and all(
            all(
                parent_k64[key].get(field) == fresh_k64[key].get(field)
                for field in stable_fields
            )
            for key in parent_k64
        )
    )
    parent_ladders = _mapping(
        parent_report.get("finite_ladder_by_method"), label="path-v1 ladders"
    )
    parent_signed = parent_ladders.get("signed_joint")
    if not isinstance(parent_signed, (list, tuple)):
        raise ValueError("path-v1 signed ladder differs")
    parent_arm = next(row for row in parent_signed if row.get("tail_rank") == _K64)
    fresh_arm = next(row for row in ladders["beta_0"] if row.get("tail_rank") == _K64)
    parent_behavior = _mapping(
        _mapping(
            parent_report.get("established_behavioral_fidelity_by_method_rank"),
            label="path-v1 behavior",
        ).get("signed_joint"),
        label="path-v1 signed behavior",
    ).get(str(_K64))
    parent_geometry = _mapping(
        _mapping(
            parent_report.get("executed_cast_once_geometry_by_method_rank"),
            label="path-v1 geometry",
        ).get("signed_joint"),
        label="path-v1 signed geometry",
    ).get(str(_K64))
    checks = {
        "beta_zero_prompt_metrics_exact": prompt_metrics_exact,
        "beta_zero_family_macro_arm_exact": _json_identity(fresh_arm)
        == _json_identity(parent_arm),
        "beta_zero_four_behavior_ledgers_exact": _json_identity(
            behavioral["beta_0"]
        )
        == _json_identity(parent_behavior),
        "beta_zero_cast_once_geometry_exact": _json_identity(geometry["beta_0"])
        == _json_identity(parent_geometry),
    }
    return {
        "checks": tuple(sorted(checks.items())),
        "all_checks_passed": all(checks.values()),
        "comparison_is_post_finite_control_not_beta_selection": True,
    }


def _resource_accounting(
    *,
    collection: Mapping[str, int],
    finite: Mapping[str, object],
    traces: Sequence[object],
    path_v1_resources: Mapping[str, object],
) -> dict[str, object]:
    collection_forwards = (
        collection["base_forward_count"]
        + collection["native_teacher_forward_count"]
        + collection["d320_boundary_forward_count"]
        + collection["path_teacher_kl_vjp_forward_count"]
    )
    if (
        collection_forwards != 112
        or collection["path_teacher_kl_vjp_backward_call_count"] != 436
        or finite.get("finite_native_forward_count") != 16
        or finite.get("finite_candidate_forward_count") != 96
    ):
        raise RuntimeError("gain-interpolation total resource accounting differs")
    support_rows = sum(int(trace.endpoint.residual_rows.shape[0]) for trace in traces)
    first_order_rescore_macs = sum(
        len(GAIN_INTERPOLATION_BETAS)
        * (
            int(trace.endpoint.residual_rows.shape[0]) * token_v1._WIDTH * _K64
            + int(trace.endpoint.supervised_tokens)
            * int(trace.endpoint.residual_rows.shape[0])
            * token_v1._WIDTH
            * _K64
            + int(trace.endpoint.supervised_tokens)
            * int(trace.endpoint.residual_rows.shape[0])
            * _K64
            + int(trace.endpoint.supervised_tokens) * _K64
        )
        for trace in traces
    )
    replay_fit_keys = (
        "signed_joint_streamed_coordinate_transform_logical_macs",
        "signed_joint_streamed_operator_and_direction_score_logical_macs",
        "signed_joint_low_rank_U_factor_deflation_logical_macs",
        "signed_joint_symmetric_eigh_320_by_320_call_count",
    )
    if any(type(path_v1_resources.get(key)) is not int for key in replay_fit_keys):
        raise ValueError("path-v1 signed-fit resource receipt differs")
    finite_forwards = int(finite["finite_native_forward_count"]) + int(
        finite["finite_candidate_forward_count"]
    )
    result = {
        **dict(collection),
        **dict(finite),
        "collection_model_forward_count": collection_forwards,
        "finite_evaluation_model_forward_count": finite_forwards,
        "total_model_forward_count": collection_forwards + finite_forwards,
        "path_teacher_kl_vjp_backward_call_count": 436,
        "predeclared_gain_variant_count": len(GAIN_INTERPOLATION_BETAS),
        "K64_gain_interpolation_scalar_ops_during_finite_preparation": (
            3
            * len(GAIN_INTERPOLATION_BETAS)
            * token_v1._EXPECTED_EXAMPLES
            * _K64
        ),
        "K64_gain_interpolation_scalar_ops_during_fold_receipt_reporting": (
            3
            * len(GAIN_INTERPOLATION_BETAS)
            * token_v1._EXPECTED_FAMILIES
            * _K64
        ),
        "finite_K64_tail_projection_logical_macs": (
            2
            * len(GAIN_INTERPOLATION_BETAS)
            * support_rows
            * _K64
            * token_v1._WIDTH
        ),
        "finite_shared_K320_tail_copy_scalar_count": (
            support_rows * token_v1._WIDTH
        ),
        "finite_K64_first_order_rescore_logical_macs": first_order_rescore_macs,
        "signed_joint_K64_fit_replay_count": token_v1._EXPECTED_FAMILIES,
        "signed_joint_K64_fit_replay_direction_count_per_fold": _K64,
        "signed_joint_K64_fit_replay_resource_identity": {
            key: path_v1_resources[key] for key in replay_fit_keys
        },
        "signed_joint_K64_fit_replay_resource_identity_is_exact_path_v1": True,
        "PCA_control_fit_or_forward_count": 0,
        "full_GL4_node_gradient_banks_simultaneously_retained": 1,
        "all_four_GL4_node_banks_retained_together": False,
        "peak_simultaneously_retained_full_sequence_vocabulary_tensor_count": 4,
        "serving_learned_parameter_count": "not_applicable_no_serving_artifact",
        "serving_logical_macs_per_token": "not_applicable_no_serving_artifact",
    }
    if result["finite_evaluation_model_forward_count"] != 112 or result[
        "total_model_forward_count"
    ] != 224:
        raise RuntimeError("gain-interpolation 112/224 forward ledger differs")
    return result


def _fold_gain_receipts(
    signed_fits: Mapping[str, CompleteH4TailSignedJointHeldFamilyFit],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for family in sorted(signed_fits):
        fit = signed_fits[family]
        fit.validate_integrity()
        original = torch.tensor(fit.gains, dtype=torch.float64)
        variants = []
        for beta in GAIN_INTERPOLATION_BETAS:
            values = _interpolated_gains(fit, beta=beta)
            variants.append(
                {
                    "variant_id": _beta_id(beta),
                    "beta_index": _beta_index(beta),
                    "beta_hex": beta.hex(),
                    "gains_sha256": _runtime_tensor_sha256(values),
                    "gain_minimum": float(values.min()),
                    "gain_mean": float(values.mean()),
                    "gain_maximum": float(values.max()),
                }
            )
        rows.append(
            {
                "held_family_id": family,
                "fit_artifact_sha256": fit.artifact_sha256,
                "directions_sha256": fit.metadata()["ambient_directions_sha256"],
                "direction_count": fit.rank,
                "original_gains_sha256": _runtime_tensor_sha256(original),
                "original_gain_minimum": float(original.min()),
                "original_gain_mean": float(original.mean()),
                "original_gain_maximum": float(original.max()),
                "predeclared_variants": tuple(variants),
                "held_family_used_to_select_beta": False,
            }
        )
    return tuple(rows)


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
                "file_sha256": token_v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_gain_interpolation_diagnostic(
    *,
    path_v1_report_path: Path | str = DEFAULT_PATH_V1_REPORT,
    expected_path_v1_report_file_sha256: str = PATH_V1_REPORT_FILE_SHA256,
    expected_path_v1_report_sha256: str = PATH_V1_REPORT_SHA256,
    endpoint_report_path: Path | str = path_v1.DEFAULT_ENDPOINT_REPORT,
    materialization_report_path: Path | str = path_v1.DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = path_v1.DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run exact path-v1 replay followed by the fixed five-beta K64 screen."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite gain-interpolation report")
    parent = _load_path_v1_report(
        path_v1_report_path,
        expected_file_sha256=expected_path_v1_report_file_sha256,
        expected_report_sha256=expected_path_v1_report_sha256,
    )
    parent_binding = _mapping(parent["input_binding"], label="path-v1 binding")
    if (
        str(endpoint_report_path)
        != str(parent_binding.get("endpoint_signed_report_file"))
        or str(materialization_report_path)
        != str(parent_binding.get("materialization_report_file"))
        or str(transfer_report_path)
        != str(parent_binding.get("transfer_report_file"))
    ):
        raise ValueError("v2 input paths must exactly reuse the path-v1 parents")
    endpoint_report, endpoint_prompts = path_v1._load_endpoint_report(
        endpoint_report_path,
        expected_file_sha256=path_v1.ENDPOINT_REPORT_FILE_SHA256,
        expected_report_sha256=path_v1.ENDPOINT_REPORT_SHA256,
    )
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="rank320 transfer",
    )
    transfer_receipts = endpoint._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, evidence, collection = path_v1._collect_path_teacher_kl_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
            endpoint_prompt_receipts=endpoint_prompts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if (
            len(traces) != token_v1._EXPECTED_EXAMPLES
            or len(evidence) != token_v1._EXPECTED_EXAMPLES
            or len(families) != token_v1._EXPECTED_FAMILIES
        ):
            raise RuntimeError("gain-interpolation A16 path panel shape differs")
        closure = summarize_complete_h4_tail_path_ftc_closure(evidence)
        closure_gates, closure_diagnostics = path_v1._closure_gate_results(
            closure,
            evidence=evidence,
            supported_basis=basis,
            traces=traces,
        )
        path_fit_endpoints = tuple(
            complete_h4_tail_path_as_endpoint_example(value) for value in evidence
        )
        signed_fits = {
            family: fit_complete_h4_tail_signed_joint_held_family(
                path_fit_endpoints,
                supported_basis=basis,
                held_family_id=family,
                max_directions=_K64,
            )
            for family in families
        }
        replay = _exact_path_v1_replay_receipt(
            parent_report=parent,
            traces=traces,
            evidence=evidence,
            closure=closure,
            closure_gates=closure_gates,
            closure_diagnostics=closure_diagnostics,
            collection_resources=collection,
            signed_fits=signed_fits,
        )
        # This call is deliberately after the exact replay hard precondition.
        observations, finite, behavioral, geometry = (
            _finite_gain_interpolation_observations(
                context=context,
                traces=traces,
                basis=basis,
                signed_fits=signed_fits,
            )
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    observation_set = _finite_observation_set_sha256(observations)
    ladders, secondary_gates = _summarize_by_beta(observations)
    sentinel_geometry = geometry["shared_exact_sentinel"]
    sentinel_behavior = behavioral["shared_exact_sentinel"]
    sentinel_passed = bool(sentinel_geometry["gates"]["passed"]) and all(
        bool(sentinel_behavior[ledger]["gates"]["passed"])
        for ledger in _LEDGERS
    )
    beta_pass: dict[str, bool] = {}
    for variant_id in _BETA_IDS:
        beta_pass[variant_id] = bool(geometry[variant_id]["gates"]["passed"]) and all(
            bool(behavioral[variant_id][ledger]["gates"]["passed"])
            for ledger in _LEDGERS
        )
    parent_beta_zero = _stable_parent_beta_zero_comparison(
        parent_report=parent,
        observations=observations,
        ladders=ladders,
        behavioral=behavioral,
        geometry=geometry,
    )
    sentinel_arm = next(
        row
        for row in ladders["beta_0"]
        if row["tail_rank"] == token_v1._D_RANK
    )
    infrastructure_gates = {
        "exact_path_v1_recollection_closure_and_fit_replay_completed_before_finite": True,
        "all_five_predeclared_betas_executed_on_all_16_prompts": (
            finite["finite_gain_interpolation_forward_count"] == 80
        ),
        "beta_zero_finite_control_exactly_reproduces_path_v1_K64": bool(
            parent_beta_zero["all_checks_passed"]
        ),
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
        "shared_k320_clears_established_fidelity_and_geometry_gates": sentinel_passed,
    }
    hypothesis_gate = bool(tuple(key for key, value in beta_pass.items() if value))
    primary_gates = {
        **infrastructure_gates,
        "at_least_one_predeclared_K64_beta_clears_all_established_gates": hypothesis_gate,
    }
    resources = _resource_accounting(
        collection=collection,
        finite=finite,
        traces=traces,
        path_v1_resources=_mapping(
            parent.get("resources"), label="path-v1 resource accounting"
        ),
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_hypothesis_use_only",
            "path_evidence_collector": "published_path_v1_lower_level_GL4_collector",
            "path_v1_exact_replay_is_hard_pre_finite_precondition": True,
            "split": "same_whole_family_leave_one_out_path_signed_K64_folds",
            "tail_rank": _K64,
            "predeclared_beta_hex_grid": tuple(
                beta.hex() for beta in GAIN_INTERPOLATION_BETAS
            ),
            "predeclared_beta_ids": _BETA_IDS,
            "gain_formula": "g_beta_equals_g_plus_beta_times_one_minus_g",
            "beta_zero_semantics": "published_path_v1_gained_K64",
            "beta_one_semantics": "same_K64_directions_with_unit_gains",
            "all_betas_reported_without_held_family_selection": True,
            "selection_performed": False,
            "selected_beta": None,
            "shared_exact_sentinel_rank": token_v1._D_RANK,
            "same_four_finite_shadow_ledgers": _LEDGERS,
            "matched_PCA_arm_reexecuted": False,
        },
        "input_binding": {
            "path_v1_report_file": str(path_v1_report_path),
            "path_v1_report_file_sha256": expected_path_v1_report_file_sha256,
            "path_v1_report_sha256": expected_path_v1_report_sha256,
            "path_v1_schema": parent.get("schema"),
            "path_v1_classification": parent.get("classification"),
            "endpoint_report_file": str(endpoint_report_path),
            "endpoint_report_file_sha256": path_v1.ENDPOINT_REPORT_FILE_SHA256,
            "endpoint_report_sha256": path_v1.ENDPOINT_REPORT_SHA256,
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": token_v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": token_v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": token_v1.TRANSFER_REPORT_SHA256,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "basis_materialization_binding": dict(materialization_binding),
            "endpoint_schema": endpoint_report.get("schema"),
            "materialization_schema": materialization.get("schema"),
        },
        "exact_path_v1_replay": replay,
        "fresh_FTC_closure": getattr(closure, "metadata")(),
        "fresh_FTC_closure_gate_results": tuple(sorted(closure_gates.items())),
        "fresh_FTC_closure_diagnostics": dict(closure_diagnostics),
        "fold_gain_interpolation_receipts": _fold_gain_receipts(signed_fits),
        "finite_ladder_by_predeclared_beta": ladders,
        "established_behavioral_fidelity_by_beta": behavioral,
        "executed_cast_once_geometry_by_beta": geometry,
        "fidelity_and_geometry_pass_by_beta": beta_pass,
        "predeclared_betas_clearing_all_established_gates": tuple(
            key for key in _BETA_IDS if beta_pass[key]
        ),
        "shared_exact_sentinel_passed": sentinel_passed,
        "path_v1_beta_zero_post_finite_reproduction": parent_beta_zero,
        "finite_observation_receipts": tuple(observations),
        "finite_observation_set_sha256": observation_set,
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "infrastructure_gate_results": tuple(sorted(infrastructure_gates.items())),
        "secondary_first_order_gate_results_by_beta": secondary_gates,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "frozen_d320_contains_same_a_held_family_information": True,
            "native_teacher_and_tail_used_for_path_evidence": True,
            "tail_direction_order_and_original_gain_whole_family_disjoint": True,
            "held_family_used_to_choose_beta": False,
            "all_betas_predeclared_and_reported": True,
            "winner_selection_or_model_authorization_performed": False,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "contains_direction_or_basis_tensors": False,
            "contains_GL4_node_gradient_banks": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
            "artifact_must_remain_outside_git": True,
        },
        "passed": all(primary_gates.values()),
        "classification": (
            "same_a_GL4_path_signed_K64_gain_interpolation_hypothesis_supported"
            if all(primary_gates.values())
            else "same_a_GL4_path_signed_K64_gain_interpolation_hypothesis_not_supported"
        ),
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen GL4 path fit, then run the fixed K64 gain-beta screen."
        )
    )
    parser.add_argument("--path-v1-report", type=Path, default=DEFAULT_PATH_V1_REPORT)
    parser.add_argument(
        "--path-v1-report-file-sha256", default=PATH_V1_REPORT_FILE_SHA256
    )
    parser.add_argument("--path-v1-report-sha256", default=PATH_V1_REPORT_SHA256)
    parser.add_argument(
        "--endpoint-report", type=Path, default=path_v1.DEFAULT_ENDPOINT_REPORT
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=path_v1.DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument(
        "--transfer-report", type=Path, default=path_v1.DEFAULT_TRANSFER_REPORT
    )
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_gain_interpolation_diagnostic(
        path_v1_report_path=args.path_v1_report,
        expected_path_v1_report_file_sha256=args.path_v1_report_file_sha256,
        expected_path_v1_report_sha256=args.path_v1_report_sha256,
        endpoint_report_path=args.endpoint_report,
        materialization_report_path=args.materialization_report,
        transfer_report_path=args.transfer_report,
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"wrote {result['artifact']['file']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
