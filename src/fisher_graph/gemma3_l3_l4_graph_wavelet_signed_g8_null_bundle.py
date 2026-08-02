"""Pre-register the signed-g8 fresh structural confirmation panel.

This module is intentionally fit-only.  It reconstructs the authenticated
Q64 parent from the already-open origins 8/24/40, compiles the frozen native
signed-g8 plan and 63 deterministic size-matched random-partition plans, and
publishes only hashes, accounting, and fit metrics.  No fresh response tensor
is opened while the panel is built.

The plans themselves are not serialized here.  A later confirmation runner
must replay them from the same fit inputs and match every receipt before it is
allowed to read the separately measured confirmation artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    evaluate_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    Gemma3SpectralSource,
    _file_sha256,
    _reserve_outputs,
    _response_binding,
    _stage_json,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_graph_wavelet_comparison_support import (
    reconstruct_authenticated_q64_fit_context,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256,
    Gemma3GraphWaveletCandidate,
    _tensor_sha256,
    load_gemma3_graph_wavelet_candidate,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
    DEFAULT_FROZEN_ARTIFACT_SHA256,
    DEFAULT_FROZEN_REPORT_SHA256,
    DEFAULT_FROZEN_TENSOR_FILE_SHA256,
    DEFAULT_OUTPUT as DEFAULT_CANDIDATE_ARTIFACT,
    EXPECTED_PARENT_SUBSPACE_ARTIFACT_SHA256,
    EXPECTED_Q64_SHA256,
    GROUP_COUNT,
    SOURCE_BASIS_KIND,
    TARGET_SOURCE_RANK,
    Gemma3L3L4GraphWaveletSignedG8Candidate,
    load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate,
)
from .gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
)
from .graph_wavelet_grouped_basis import fit_graph_wavelet_grouped_basis
from .graph_wavelet_random_partition_confirmation import (
    PRIMARY_NULL_GATES,
    RANDOM_CONTROL_COUNT,
    BalancedRandomPartitionPanel,
    derive_balanced_random_partition_panel,
    grouped_parent_basis_sha256,
)


__all__ = [
    "DEFAULT_NULL_BUNDLE_ARTIFACT_SHA256",
    "DEFAULT_NULL_BUNDLE_FILE_SHA256",
    "DEFAULT_NULL_BUNDLE_REPORT_SHA256",
    "DEFAULT_OUTPUT",
    "FRESH_CONFIRMATION_ORIGINS",
    "Gemma3L3L4GraphWaveletSignedG8NullBundle",
    "compile_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle",
    "freeze_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle",
    "load_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle",
    "main",
    "replay_gemma3_l3_l4_graph_wavelet_signed_g8_null_plans",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-signed-g8-null-bundle-v1.json"
)
DEFAULT_NULL_BUNDLE_ARTIFACT_SHA256 = (
    "18079c3b49375137198263ba06c04c9f84fb4efe50a2d266b58a0b01d9cc93b8"
)
DEFAULT_NULL_BUNDLE_FILE_SHA256 = (
    "3ee4208163f7631120f015d5a8e37821ab06233d5a886360ddbaa434f6050e54"
)
DEFAULT_NULL_BUNDLE_REPORT_SHA256 = (
    "cac2e3cb3f60e1620cd5b9271599b715b5e32a5c3d06cdddc89fe767ac545e9c"
)

FRESH_CONFIRMATION_ORIGINS = (12, 28, 36)
FRESH_SEQUENCE_LENGTH = 72
FRESH_MAX_LAG = 31
FRESH_FFT_LENGTH = 64

_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-signed-g8-null-bundle:v1\0"
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-signed-g8-null-bundle-report:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FIELDS = {
    "role",
    "control_ordinal",
    "partition_artifact_sha256",
    "grouped_basis_artifact_sha256",
    "grouped_full_basis_sha256",
    "rank45_basis_sha256",
    "plan_artifact_sha256",
    "plan_accounting",
    "fit_evaluation",
}
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_fresh_response_values": False,
    "contains_fit_response_values": False,
    "contains_parent_basis_tensor": False,
    "contains_partition_tensor": False,
    "contains_compiled_plan_tensors": False,
    "metadata_only": True,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}
_CLAIMS = {
    "null_panel_frozen_before_fresh_measurement": True,
    "controls_are_size_matched_direct_partitions": True,
    "controls_use_graph_topology": False,
    "controls_use_fresh_response_values": False,
    "candidate_refit_performed": False,
    "fresh_structural_fidelity_measured": False,
    "natural_prompt_fidelity_measured": False,
    "compression_claim": False,
    "speed_or_latency_claim": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _metadata(value: object, *, label: str) -> dict[str, object]:
    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(child) for child in item]
        return item

    try:
        normalized = json.loads(
            _canonical_json_bytes(thaw(value)).decode("ascii")
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain JSON metadata only") from error
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must normalize to an object")
    return normalized


def _same_metadata(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _source_receipt(source: Gemma3SpectralSource) -> dict[str, object]:
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    return {
        "tensor_file_sha256": source.file_sha256,
        "report_file_sha256": source.report_file_sha256,
        "report_payload_sha256": source.report_payload_sha256,
        "mapping_artifact_sha256": source.mapping.artifact_sha256,
        "response_artifact_sha256": local.artifact_sha256,
        "source_model_sha256": source.binding.get("source_model_sha256"),
    }


def _fit_evaluation(
    plan: ConditionalSpectralGeneratorPlan,
    fit_kernels: Tensor,
    *,
    response_binding_sha256: str,
) -> dict[str, object]:
    return evaluate_conditional_spectral_generator(
        plan,
        fit_kernels,
        FIT_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=response_binding_sha256,
    ).metadata()


def _plan_receipt(
    *,
    role: str,
    control_ordinal: int | None,
    partition_artifact_sha256: str,
    grouped_basis_artifact_sha256: str,
    grouped_full_basis_sha256: str,
    rank45_basis_sha256: str,
    plan: ConditionalSpectralGeneratorPlan,
    fit_kernels: Tensor,
    response_binding_sha256: str,
) -> dict[str, object]:
    return {
        "role": role,
        "control_ordinal": control_ordinal,
        "partition_artifact_sha256": partition_artifact_sha256,
        "grouped_basis_artifact_sha256": grouped_basis_artifact_sha256,
        "grouped_full_basis_sha256": grouped_full_basis_sha256,
        "rank45_basis_sha256": rank45_basis_sha256,
        "plan_artifact_sha256": plan.artifact_sha256,
        "plan_accounting": plan.accounting().metadata(),
        "fit_evaluation": _fit_evaluation(
            plan,
            fit_kernels,
            response_binding_sha256=response_binding_sha256,
        ),
    }


def _validate_receipt(
    value: Mapping[str, object],
    *,
    role: str,
    ordinal: int | None,
    partition_artifact_sha256: str,
) -> dict[str, object]:
    receipt = _metadata(value, label=f"{role} plan receipt")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError(f"{role} plan receipt fields differ")
    for field in (
        "partition_artifact_sha256",
        "grouped_basis_artifact_sha256",
        "grouped_full_basis_sha256",
        "rank45_basis_sha256",
        "plan_artifact_sha256",
    ):
        _require_sha256(receipt.get(field), label=f"{role}.{field}")
    fit = receipt.get("fit_evaluation")
    accounting = receipt.get("plan_accounting")
    if (
        receipt.get("role") != role
        or receipt.get("control_ordinal") != ordinal
        or receipt.get("partition_artifact_sha256")
        != partition_artifact_sha256
        or not isinstance(fit, dict)
        or tuple(fit.get("evaluation_origins", ())) != FIT_ORIGINS
        or tuple(fit.get("fit_origin_overlap", ())) != FIT_ORIGINS
        or fit.get("plan_sha256") != receipt.get("plan_artifact_sha256")
        or fit.get("fit_was_not_recomputed") is not True
        or not isinstance(accounting, dict)
        or accounting.get("source_modes") != 64
        or accounting.get("source_rank") != TARGET_SOURCE_RANK
        or accounting.get("target_modes") != 64
        or accounting.get("target_rank") != 64
        or accounting.get("lag_count") != 32
    ):
        raise ValueError(f"{role} plan receipt semantics differ")
    return receipt


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphWaveletSignedG8NullBundle:
    """Metadata-only frozen native and 63-control plan panel."""

    candidate_artifact_sha256: str
    source_receipt: Mapping[str, object]
    parent_receipt: Mapping[str, object]
    panel: BalancedRandomPartitionPanel
    native_plan_receipt: Mapping[str, object]
    control_plan_receipts: tuple[Mapping[str, object], ...]
    preregistered_confirmation: Mapping[str, object]
    claims: Mapping[str, object]
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.format_version != _FORMAT_VERSION:
            raise ValueError("null bundle header differs")
        _require_sha256(
            self.candidate_artifact_sha256,
            label="candidate_artifact_sha256",
        )
        if not isinstance(self.panel, BalancedRandomPartitionPanel):
            raise TypeError("panel must be a BalancedRandomPartitionPanel")
        self.panel.validate_integrity()
        for field in (
            "source_receipt",
            "parent_receipt",
            "preregistered_confirmation",
            "claims",
        ):
            object.__setattr__(
                self,
                field,
                _metadata(getattr(self, field), label=field),
            )
        native = _validate_receipt(
            self.native_plan_receipt,
            role="native_signed_g8",
            ordinal=None,
            partition_artifact_sha256=(
                self.panel.native_partition_artifact_sha256
            ),
        )
        controls = tuple(
            _validate_receipt(
                receipt,
                role="random_partition_control",
                ordinal=ordinal,
                partition_artifact_sha256=(
                    self.panel.controls[ordinal].artifact_sha256
                ),
            )
            for ordinal, receipt in enumerate(self.control_plan_receipts)
        )
        object.__setattr__(self, "native_plan_receipt", native)
        object.__setattr__(self, "control_plan_receipts", controls)
        self._validate_semantics()
        computed = _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("null bundle artifact hash differs")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _validate_semantics(self) -> None:
        if (
            self.panel.candidate_artifact_sha256
            != self.candidate_artifact_sha256
            or len(self.control_plan_receipts) != RANDOM_CONTROL_COUNT
            or not _same_metadata(self.claims, _CLAIMS)
            or not _same_metadata(
                self.preregistered_confirmation,
                _preregistered_confirmation(),
            )
            or self.native_plan_receipt.get("plan_artifact_sha256")
            is None
            or len(
                {
                    receipt["partition_artifact_sha256"]
                    for receipt in self.control_plan_receipts
                }
            )
            != RANDOM_CONTROL_COUNT
        ):
            raise ValueError("null bundle panel or claim semantics differ")
        accounting = self.native_plan_receipt["plan_accounting"]
        if any(
            receipt["plan_accounting"] != accounting
            for receipt in self.control_plan_receipts
        ):
            raise ValueError("null bundle plan payloads are not size matched")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "source_receipt": dict(self.source_receipt),
            "parent_receipt": dict(self.parent_receipt),
            "panel": self.panel.metadata(),
            "native_plan_receipt": dict(self.native_plan_receipt),
            "control_plan_receipts": tuple(
                dict(receipt) for receipt in self.control_plan_receipts
            ),
            "preregistered_confirmation": dict(
                self.preregistered_confirmation
            ),
            "claims": dict(self.claims),
            "safety": _SAFETY,
        }

    def validate_integrity(self) -> None:
        self.panel.validate_integrity()
        self._validate_semantics()
        if (
            _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("null bundle artifact hash differs")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        value: object,
    ) -> "Gemma3L3L4GraphWaveletSignedG8NullBundle":
        if not isinstance(value, Mapping):
            raise TypeError("null bundle state must be a mapping")
        expected = {
            "schema",
            "format_version",
            "candidate_artifact_sha256",
            "source_receipt",
            "parent_receipt",
            "panel",
            "native_plan_receipt",
            "control_plan_receipts",
            "preregistered_confirmation",
            "claims",
            "safety",
            "artifact_sha256",
        }
        if set(value) != expected or value.get("safety") != _SAFETY:
            raise ValueError("null bundle state fields differ")
        panel_metadata = value.get("panel")
        if not isinstance(panel_metadata, Mapping):
            raise TypeError("null bundle panel metadata must be a mapping")
        panel = derive_balanced_random_partition_panel(
            candidate_artifact_sha256=str(
                panel_metadata.get("candidate_artifact_sha256")
            ),
            parent_basis_sha256=str(
                panel_metadata.get("parent_basis_sha256")
            ),
            native_partition_artifact_sha256=str(
                panel_metadata.get("native_partition_artifact_sha256")
            ),
            native_groups=panel_metadata.get("native_groups", ()),  # type: ignore[arg-type]
        )
        if not _same_metadata(panel.metadata(), panel_metadata):
            raise ValueError("serialized null panel differs from replay")
        receipts = value.get("control_plan_receipts")
        if not isinstance(receipts, (tuple, list)):
            raise TypeError("control plan receipts must be a sequence")
        return cls(
            candidate_artifact_sha256=value[  # type: ignore[arg-type]
                "candidate_artifact_sha256"
            ],
            source_receipt=value["source_receipt"],  # type: ignore[arg-type]
            parent_receipt=value["parent_receipt"],  # type: ignore[arg-type]
            panel=panel,
            native_plan_receipt=value[  # type: ignore[arg-type]
                "native_plan_receipt"
            ],
            control_plan_receipts=tuple(receipts),  # type: ignore[arg-type]
            preregistered_confirmation=value[  # type: ignore[arg-type]
                "preregistered_confirmation"
            ],
            claims=value["claims"],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )


def _preregistered_confirmation() -> dict[str, object]:
    return {
        "fresh_origins": FRESH_CONFIRMATION_ORIGINS,
        "sequence_length": FRESH_SEQUENCE_LENGTH,
        "max_lag": FRESH_MAX_LAG,
        "fft_length": FRESH_FFT_LENGTH,
        "response_component": "local_central_odd_tangent",
        "source_modes": 64,
        "source_rank": TARGET_SOURCE_RANK,
        "target_modes": 64,
        "target_rank": 64,
        "native_mode_family_count": GROUP_COUNT,
        "random_control_count": RANDOM_CONTROL_COUNT,
        "maximum_native_pooled_and_per_origin_relative_error": 0.20,
        "minimum_native_pooled_and_per_origin_cosine": 0.98,
        "native_pooled_sse_must_not_exceed_signed_gfa": True,
        "global_svd_role": "descriptive_ceiling_only",
        "random_null_gates": PRIMARY_NULL_GATES.metadata(),
        "fresh_measurement_must_be_separate_artifact": True,
        "candidate_or_control_refit_on_fresh_values": False,
    }


def _reconstruct_context(
    source: Gemma3SpectralSource,
    parent: Gemma3GraphWaveletCandidate,
):
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    receipt = _source_receipt(source)
    response_binding = _response_binding(
        source,
        component="local_central_odd_tangent",
    )
    context = reconstruct_authenticated_q64_fit_context(
        local.impulse_responses,
        source.source_mode_standard_deviations,
        source.mapping.impulse_logical_positions,
        parent,
        response_binding_sha256=response_binding,
        expected_graph_basis_artifact_sha256=(
            EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256
        ),
        source_receipt=receipt,
        fft_length=source.mapping.fft_length,
        expected_parent_rank=64,
    )
    if (
        _tensor_sha256(context.q64) != EXPECTED_Q64_SHA256
        or context.parent_subspace.artifact_sha256
        != EXPECTED_PARENT_SUBSPACE_ARTIFACT_SHA256
    ):
        raise ValueError("null bundle Q64 reconstruction differs")
    return context, receipt


def replay_gemma3_l3_l4_graph_wavelet_signed_g8_null_plans(
    source: Gemma3SpectralSource,
    parent: Gemma3GraphWaveletCandidate,
    candidate: Gemma3L3L4GraphWaveletSignedG8Candidate,
) -> tuple[
    BalancedRandomPartitionPanel,
    tuple[ConditionalSpectralGeneratorPlan, ...],
    tuple[dict[str, object], ...],
]:
    """Rebuild the 63 control plans and their exact frozen receipts."""

    candidate.validate_frozen_identity()
    context, _ = _reconstruct_context(source, parent)
    partition_metadata = candidate.construction_receipt.get("partition")
    if not isinstance(partition_metadata, Mapping):
        raise ValueError("signed-g8 candidate lacks native partition metadata")
    panel = derive_balanced_random_partition_panel(
        candidate_artifact_sha256=candidate.artifact_sha256,
        parent_basis_sha256=grouped_parent_basis_sha256(context.q64),
        native_partition_artifact_sha256=str(
            partition_metadata.get("artifact_sha256")
        ),
        native_groups=partition_metadata.get("groups", ()),  # type: ignore[arg-type]
    )
    plans: list[ConditionalSpectralGeneratorPlan] = []
    receipts: list[dict[str, object]] = []
    for control in panel.controls:
        grouped = fit_graph_wavelet_grouped_basis(
            context.q64,
            context.weighted_fit,
            control,  # type: ignore[arg-type]
            method="wavelet_local_svd",
            fit_origins=FIT_ORIGINS,
            response_binding_sha256=context.response_binding_sha256,
            parent_subspace_artifact_sha256=(
                context.parent_subspace.artifact_sha256
            ),
        )
        basis = grouped.prefix(TARGET_SOURCE_RANK)
        plan = fit_conditional_spectral_generator_with_source_basis(
            context.fit_kernels,
            context.source_scales,
            FIT_ORIGINS,
            FIT_ORIGINS,
            basis,
            context.target_modes,
            source_basis_kind=SOURCE_BASIS_KIND,
            source_basis_fit_weighted_kernels_sha256=(
                context.graph.fit_weighted_kernels_sha256
            ),
            response_binding_sha256=context.response_binding_sha256,
            input_transform="standardized_linear",
            fft_length=context.fft_length,
        )
        plans.append(plan)
        receipts.append(
            _plan_receipt(
                role="random_partition_control",
                control_ordinal=control.control_ordinal,
                partition_artifact_sha256=control.artifact_sha256,
                grouped_basis_artifact_sha256=grouped.artifact_sha256,
                grouped_full_basis_sha256=str(
                    grouped.metadata()["basis_sha256"]
                ),
                rank45_basis_sha256=_tensor_sha256(basis),
                plan=plan,
                fit_kernels=context.fit_kernels,
                response_binding_sha256=context.response_binding_sha256,
            )
        )
    return panel, tuple(plans), tuple(receipts)


def compile_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle(
    source: Gemma3SpectralSource,
    parent: Gemma3GraphWaveletCandidate,
    candidate: Gemma3L3L4GraphWaveletSignedG8Candidate,
) -> Gemma3L3L4GraphWaveletSignedG8NullBundle:
    """Compile the complete metadata-only panel from opened fit values."""

    panel, _, control_receipts = (
        replay_gemma3_l3_l4_graph_wavelet_signed_g8_null_plans(
            source,
            parent,
            candidate,
        )
    )
    context, source_receipt = _reconstruct_context(source, parent)
    construction = candidate.construction_receipt
    partition = construction["partition"]
    grouped = construction["grouped_basis"]
    if not isinstance(partition, Mapping) or not isinstance(grouped, Mapping):
        raise ValueError("signed-g8 construction metadata differs")
    native_receipt = _plan_receipt(
        role="native_signed_g8",
        control_ordinal=None,
        partition_artifact_sha256=str(partition["artifact_sha256"]),
        grouped_basis_artifact_sha256=str(grouped["artifact_sha256"]),
        grouped_full_basis_sha256=str(grouped["basis_sha256"]),
        rank45_basis_sha256=str(construction["basis_sha256"]),
        plan=candidate.plan,
        fit_kernels=context.fit_kernels,
        response_binding_sha256=context.response_binding_sha256,
    )
    return Gemma3L3L4GraphWaveletSignedG8NullBundle(
        candidate_artifact_sha256=candidate.artifact_sha256,
        source_receipt=source_receipt,
        parent_receipt=context.parent_receipt,
        panel=panel,
        native_plan_receipt=native_receipt,
        control_plan_receipts=control_receipts,
        preregistered_confirmation=_preregistered_confirmation(),
        claims=_CLAIMS,
    )


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("null bundle output must be JSON under .local-runs")
    return destination


def _publish_bundle(
    bundle: Gemma3L3L4GraphWaveletSignedG8NullBundle,
    *,
    output: Path | str,
) -> dict[str, object]:
    destination = _validate_output(output)
    reservation = _reserve_outputs((destination,))
    stage: Path | None = None
    try:
        report: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "bundle": bundle.metadata(),
            "artifact": {
                "file": str(destination),
                "committable": False,
            },
            "safety": _SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        stage = _stage_json(report, destination)
        reservation.publish((stage,))
        result = dict(report)
        result["artifact"] = {
            **dict(report["artifact"]),  # type: ignore[arg-type]
            "file_sha256": _file_sha256(destination),
            "file_bytes": destination.stat().st_size,
        }
        return result
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def load_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
    expected_artifact_sha256: str,
) -> Gemma3L3L4GraphWaveletSignedG8NullBundle:
    """Strictly authenticate one published null bundle."""

    source = Path(path)
    if _file_sha256(source) != _require_sha256(
        expected_file_sha256,
        label="expected null bundle file",
    ):
        raise ValueError("null bundle file hash differs")
    with source.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise TypeError("null bundle report must be an object")
    claimed = _require_sha256(
        report.get("report_sha256"),
        label="null bundle report",
    )
    payload = dict(report)
    payload.pop("report_sha256")
    if (
        set(report)
        != {
            "schema",
            "format_version",
            "bundle",
            "artifact",
            "safety",
            "report_sha256",
        }
        or report.get("schema") != _SCHEMA
        or report.get("format_version") != _FORMAT_VERSION
        or report.get("safety") != _SAFETY
        or claimed
        != _require_sha256(
            expected_report_sha256,
            label="expected null bundle report",
        )
        or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed
    ):
        raise ValueError("null bundle report binding differs")
    bundle = Gemma3L3L4GraphWaveletSignedG8NullBundle.from_state_dict(
        report.get("bundle")
    )
    if bundle.artifact_sha256 != _require_sha256(
        expected_artifact_sha256,
        label="expected null bundle artifact",
    ):
        raise ValueError("null bundle logical artifact differs")
    return bundle


def freeze_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Strict-load the frozen lineage, compile all controls, and publish."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite null bundle")
    source = load_gemma3_spectral_source(
        source_artifact_path,
        expected_file_sha256=DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        parent_artifact_path,
        expected_artifact_sha256=DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_FROZEN_REPORT_SHA256,
    )
    bundle = compile_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle(
        source,
        parent,
        candidate,
    )
    return _publish_bundle(bundle, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the signed-g8 63-control confirmation bundle",
    )
    parser.add_argument("--source-artifact", default=DEFAULT_INTERIOR_ARTIFACT)
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument(
        "--candidate-artifact",
        default=DEFAULT_CANDIDATE_ARTIFACT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = freeze_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle(
        source_artifact_path=arguments.source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        output=arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
