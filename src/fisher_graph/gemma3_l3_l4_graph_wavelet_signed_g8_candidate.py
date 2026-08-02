"""Frozen executable signed-g8 candidate for the Gemma L3 -> L4 edge.

The grouped-comparison artifact deliberately contains metadata only.  This
module replays its fit-only signed eight-group construction from the pinned
source and parent artifacts, checks that every audited identity is reproduced,
and freezes exactly one ``ConditionalSpectralGeneratorPlan`` for later
source-authoritative shadow execution.

The artifact contains no model weights, tokenizer, prompts, token IDs, raw
response tensor, Q64 parent basis, graph matrices, or partition matrices.  The
rank-45 source basis is already folded into the serialized conditional plan.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
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
    _stage_torch,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_graph_wavelet_comparison_support import (
    reconstruct_authenticated_q64_fit_context,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256,
    Gemma3GraphWaveletCandidate,
    _tensor_sha256 as _graph_tensor_sha256,
    load_gemma3_graph_wavelet_candidate,
)
from .gemma3_l3_l4_graph_wavelet_grouped_comparison_experiment import (
    Gemma3GraphWaveletGroupedComparisonCandidate,
    _basis_mixing,
    _grouped_receipt,
    load_gemma3_graph_wavelet_grouped_comparison_candidate,
)
from .gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
    TOPOLOGY_TOP_K,
)
from .graph_wavelet_grouped_basis import (
    fit_graph_wavelet_grouped_basis,
    fit_graph_wavelet_topology_partition,
)


__all__ = [
    "DEFAULT_COMPARISON_ARTIFACT",
    "DEFAULT_FROZEN_ARTIFACT_SHA256",
    "DEFAULT_FROZEN_REPORT_SHA256",
    "DEFAULT_FROZEN_TENSOR_FILE_SHA256",
    "DEFAULT_OUTPUT",
    "Gemma3L3L4GraphWaveletSignedG8Candidate",
    "compile_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
    "freeze_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
    "load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
    "main",
]


DEFAULT_COMPARISON_ARTIFACT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-grouped-comparison-dev-v1.pt"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-signed-g8-frozen-v1.pt"
)
DEFAULT_FROZEN_ARTIFACT_SHA256 = (
    "36b7bbf9e42f1e6e9b3182dc7e853580303fee4a38172ba9cad86724a8a6086b"
)
DEFAULT_FROZEN_TENSOR_FILE_SHA256 = (
    "9fa9b3e1fd93da96e92f40392030a130b2ca70381bb3d37916c817cf53821515"
)
DEFAULT_FROZEN_REPORT_SHA256 = (
    "3607b59d82f3389417fcd9aaa17131d8b1ec6e1c7fbb5eba66eb4cd561c0bd57"
)

METHOD = "signed_local_svd_g8"
SOURCE_BASIS_KIND = "fit_only_graph_wavelet_local_block_svd"
TARGET_SOURCE_RANK = 45
GROUP_COUNT = 8

EXPECTED_COMPARISON_ARTIFACT_SHA256 = (
    "2169782723cf11307fa3f86adc20700f67bbbff2c8491f3f984134ac2963eab8"
)
EXPECTED_COMPARISON_TENSOR_FILE_SHA256 = (
    "7e9e22fb5bb2125f1089511f3e8f0c4d541bb70976a929e82f7ce01f88fab28c"
)
EXPECTED_COMPARISON_REPORT_SHA256 = (
    "bc6913b1772ed6967199e0ef3052336adbe5a78b7015fb79397484d29e029424"
)
EXPECTED_COMPARISON_REPORT_FILE_SHA256 = (
    "4547ab750da159305974f7cc4a0f0707781c4e7eeb071e39a4030c175b2eabdc"
)
EXPECTED_SOURCE_MODEL_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
EXPECTED_Q64_SHA256 = (
    "7e8f09dc12f2e4f30db2cbf1f7c02d18794f658f7e9bb6f8577bedce4b198fcf"
)
EXPECTED_PARTITION_ARTIFACT_SHA256 = (
    "ad6e49c586622a2ecb273320311bffe6a11fc1f2e2e74c6063118599e54b67a9"
)
EXPECTED_GROUPED_BASIS_ARTIFACT_SHA256 = (
    "3b1dec171bf1d8dea744083a8fd33608c5894097fbd65b661e5364fbde7b2a62"
)
EXPECTED_GROUPED_FULL_BASIS_SHA256 = (
    "72673cacfcfc7946c4ffce08d11c6e053a0cebbdf97b19aa3220ca0b6a7ac8ec"
)
EXPECTED_RANK45_GRAPH_BASIS_SHA256 = (
    "d10ac55886cf131593678fdf52d794c6dd79d3280b26d23ff95b6a889b76ecda"
)
EXPECTED_PLAN_ARTIFACT_SHA256 = (
    "c4943189dd42ebfe92aa31d1b2168b585f3218110aabd26990ec821d254e8e21"
)
EXPECTED_RESPONSE_BINDING_SHA256 = (
    "71a8ec2b5e108256a96c81c1cf5855280054828816e20d28233d5ce8796c28cb"
)
EXPECTED_FIT_WEIGHTED_KERNELS_SHA256 = (
    "04143f6714bd6983fac6397fb580dbfbf5d8a938ec2486a56e09310487c33cd8"
)
EXPECTED_PARENT_SUBSPACE_ARTIFACT_SHA256 = (
    "cfa9968b22922abbcf534348192e5626dc22e261af8f0d2839de5484beb75d02"
)
EXPECTED_LOCAL_BLOCK_RANK_SEMANTICS = (
    "fit_only_graph_wavelet_topology_partitioned_block_svd_"
    "orthonormal_source_subspace"
)
EXPECTED_PLAN_TENSOR_SHA256S = {
    "source_scales": (
        "16387c303cf6dccce3a22a2194cc3b358c28634fb82fc6d50a070b07a49792b8"
    ),
    "source_basis": (
        "4a4ace5c40703214493f35116d3e56312f4754b7f2bd49f0943fc3662d1d48a2"
    ),
    "target_basis": (
        "eb95e6ed0193d504dc36eb02ac2df09c81bbc7a3882a08d0fec99e274893e435"
    ),
    "knot_cores": (
        "0a65cbfb7cee67d52f7272ebf659e203d2ebc7e298658a627a0a73f9da352da5"
    ),
    "source_singular_values": (
        "043230f10129372fdd912b5719e24ba8fbb2b43b591cae2eebacd0a55840bd31"
    ),
    "target_singular_values": (
        "0308f60ee69385b9432f2c5b9408cd26d5add0cbc284facc8c6886d6e4a35c16"
    ),
}

_EXPECTED_SOURCE_RECEIPT = {
    "mapping_artifact_sha256": (
        "60f3c1802335c0f9d3182d626dd724c071deeb41bdea3000fc659f0df67aefc9"
    ),
    "report_file_sha256": (
        "34952df5a68241b179ccbe50a9966c514a18c191424e64ea3302080c08e119c5"
    ),
    "report_payload_sha256": DEFAULT_INTERIOR_REPORT_SHA256,
    "response_artifact_sha256": (
        "6f6df8dedcf4751b87cab82ae9e8d237867a8a11d6721846e98a37e63d050bf7"
    ),
    "source_model_sha256": EXPECTED_SOURCE_MODEL_SHA256,
    "tensor_file_sha256": DEFAULT_INTERIOR_ARTIFACT_SHA256,
}
_EXPECTED_PARENT_RECEIPT = {
    "artifact_sha256": DEFAULT_PARENT_ARTIFACT_SHA256,
    "graph_artifact_sha256": EXPECTED_GRAPH_BASIS_ARTIFACT_SHA256,
    "q64_selected_packet_order_sha256": (
        "e4aae982c3866c4c429230198735489868097a65dce8c3374e976d7f90dc8730"
    ),
    "q64_source_basis_sha256": EXPECTED_Q64_SHA256,
    "signed_frame_artifact_sha256": (
        "edac3ba54408c16ab27b205108a706ac8a7cec2993a03075654303b761cdb35e"
    ),
    "signed_gomp_subspace_artifact_sha256": (
        EXPECTED_PARENT_SUBSPACE_ARTIFACT_SHA256
    ),
}
_EXPECTED_COMPARISON_RECEIPT = {
    "artifact_sha256": EXPECTED_COMPARISON_ARTIFACT_SHA256,
    "tensor_file_sha256": EXPECTED_COMPARISON_TENSOR_FILE_SHA256,
    "report_file_sha256": EXPECTED_COMPARISON_REPORT_FILE_SHA256,
    "report_payload_sha256": EXPECTED_COMPARISON_REPORT_SHA256,
    "selected_method": METHOD,
}

_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_frozen"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-signed-g8-frozen:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-signed-g8-frozen-report:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_raw_response_tensors": False,
    "contains_q64_parent_basis_tensor": False,
    "contains_graph_or_partition_tensors": False,
    "contains_compiled_plan_tensors": True,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}
_CLAIMS = {
    "fit_only_signed_g8_candidate_frozen": True,
    "source_authoritative_shadow_only": True,
    "candidate_outputs_must_not_be_served": True,
    "complete_gemma_block_replacement": False,
    "whole_model_replacement": False,
    "natural_prompt_fidelity_measured": False,
    "nll_kl_or_top1_fidelity_measured": False,
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


def _selected_row(
    comparison: Gemma3GraphWaveletGroupedComparisonCandidate,
) -> Mapping[str, object]:
    rows = tuple(
        row for row in comparison.rate_rows if row.get("method") == METHOD
    )
    if len(rows) != 1:
        raise ValueError("grouped comparison lacks exactly one signed-g8 row")
    return rows[0]


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphWaveletSignedG8Candidate:
    """One integrity-bound executable signed-g8 conditional plan."""

    plan: ConditionalSpectralGeneratorPlan
    binding: Mapping[str, object]
    model: Mapping[str, object]
    source_receipt: Mapping[str, object]
    parent_receipt: Mapping[str, object]
    comparison_receipt: Mapping[str, object]
    construction_receipt: Mapping[str, object]
    selection_receipt: Mapping[str, object]
    protocol: Mapping[str, object]
    claims: Mapping[str, object]
    method: str = METHOD
    source_basis_kind: str = SOURCE_BASIS_KIND
    artifact_sha256: str = ""
    schema: str = _SCHEMA
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ConditionalSpectralGeneratorPlan):
            raise TypeError("plan must be a ConditionalSpectralGeneratorPlan")
        self.plan.validate_integrity()
        for field in (
            "binding",
            "model",
            "source_receipt",
            "parent_receipt",
            "comparison_receipt",
            "construction_receipt",
            "selection_receipt",
            "protocol",
            "claims",
        ):
            object.__setattr__(
                self,
                field,
                _metadata(getattr(self, field), label=field),
            )
        self._validate_semantics()
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            ) != computed:
                raise ValueError("signed-g8 candidate artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "method": self.method,
            "source_basis_kind": self.source_basis_kind,
            "binding": dict(self.binding),
            "model": dict(self.model),
            "source_receipt": dict(self.source_receipt),
            "parent_receipt": dict(self.parent_receipt),
            "comparison_receipt": dict(self.comparison_receipt),
            "construction_receipt": dict(self.construction_receipt),
            "selection_receipt": dict(self.selection_receipt),
            "protocol": dict(self.protocol),
            "claims": dict(self.claims),
            "plan": self.plan.metadata(),
            "safety": _SAFETY,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)

    def _validate_semantics(self) -> None:
        plan = self.plan
        selection = self.selection_receipt
        construction = self.construction_receipt
        partition = construction.get("partition")
        grouped = construction.get("grouped_basis")
        fit_evaluation = selection.get("fit_evaluation")
        heldout_evaluation = selection.get("heldout_evaluation")
        if (
            self.schema != _SCHEMA
            or self.format_version != _FORMAT_VERSION
            or self.method != METHOD
            or self.source_basis_kind != SOURCE_BASIS_KIND
            or plan.source_modes != 64
            or plan.source_rank != TARGET_SOURCE_RANK
            or plan.target_modes != 64
            or plan.target_rank != 64
            or plan.fit_knot_origins != FIT_ORIGINS
            or plan.lag_count != 32
            or plan.fft_length != 64
            or plan.input_transform != "standardized_linear"
            or plan.rank_semantics != EXPECTED_LOCAL_BLOCK_RANK_SEMANTICS
            or plan.heldout_origins_used_for_fit is not False
            or plan.cross_mode_terms_measured is not False
        ):
            raise ValueError("signed-g8 plan ABI differs")
        if (
            not isinstance(partition, Mapping)
            or partition.get("group_count") != GROUP_COUNT
            or tuple(partition.get("group_sizes", ())) != (8,) * GROUP_COUNT
            or partition.get("artifact_sha256")
            != construction.get("partition", {}).get("artifact_sha256")
            or not isinstance(grouped, Mapping)
            or grouped.get("method") != "wavelet_local_svd"
            or grouped.get("partition_artifact_sha256")
            != partition.get("artifact_sha256")
            or construction.get("construction_kind") != "wavelet_local_svd"
            or construction.get("topology_kind") != "signed"
            or construction.get("cross_group_rotations_permitted") is not False
            or construction.get("global_svd_equivalence_claim") is not False
            or construction.get("heldout_values_used_for_construction") is not False
            or len(tuple(construction.get("rank_allocation", ()))) != GROUP_COUNT
            or sum(construction.get("rank_allocation", ()))
            != TARGET_SOURCE_RANK
        ):
            raise ValueError("signed-g8 construction receipt differs")
        if (
            selection.get("method") != METHOD
            or selection.get("rank") != TARGET_SOURCE_RANK
            or selection.get("source_rank") != TARGET_SOURCE_RANK
            or selection.get("target_rank") != 64
            or selection.get("source_basis_kind") != SOURCE_BASIS_KIND
            or selection.get("plan_artifact_sha256") != plan.artifact_sha256
            or selection.get("passes_fidelity_gate") is not True
            or not isinstance(fit_evaluation, Mapping)
            or fit_evaluation.get("plan_sha256") != plan.artifact_sha256
            or not isinstance(heldout_evaluation, Mapping)
            or heldout_evaluation.get("plan_sha256") != plan.artifact_sha256
            or tuple(heldout_evaluation.get("fit_origin_overlap", ())) != ()
            or selection.get("plan_accounting") != plan.accounting().metadata()
        ):
            raise ValueError("signed-g8 selection receipt differs")
        protocol = self.protocol
        if (
            protocol.get("method") != METHOD
            or protocol.get("source_basis_kind") != SOURCE_BASIS_KIND
            or tuple(protocol.get("fit_origins", ())) != FIT_ORIGINS
            or protocol.get("source_modes") != 64
            or protocol.get("source_rank") != TARGET_SOURCE_RANK
            or protocol.get("target_modes") != 64
            or protocol.get("target_rank") != 64
            or protocol.get("lag_count") != 32
            or protocol.get("fft_length") != 64
            or protocol.get("source_origin_minimum") != 8
            or protocol.get("source_origin_maximum") != 40
            or protocol.get("candidate_serving_authorized") is not False
            or protocol.get("supported_shadow_arms") != ["identity", "all_on"]
        ):
            raise ValueError("signed-g8 frozen protocol differs")
        if self.claims != _CLAIMS:
            raise ValueError("signed-g8 claim boundary differs")
        if (
            self.binding.get("source_model_sha256")
            != self.model.get("source_model_sha256")
            or self.source_receipt.get("source_model_sha256")
            != self.binding.get("source_model_sha256")
        ):
            raise ValueError("signed-g8 model binding differs")

    def _validate_pinned_identity(self) -> None:
        plan_metadata = self.plan.metadata()
        tensor_hashes = plan_metadata.get("tensor_sha256s")
        construction = self.construction_receipt
        partition = construction.get("partition")
        grouped = construction.get("grouped_basis")
        if (
            self.plan.artifact_sha256 != EXPECTED_PLAN_ARTIFACT_SHA256
            or tensor_hashes != EXPECTED_PLAN_TENSOR_SHA256S
            or self.plan.response_binding_sha256
            != EXPECTED_RESPONSE_BINDING_SHA256
            or self.plan.fit_weighted_kernels_sha256
            != EXPECTED_FIT_WEIGHTED_KERNELS_SHA256
            or self.source_receipt != _EXPECTED_SOURCE_RECEIPT
            or self.parent_receipt != _EXPECTED_PARENT_RECEIPT
            or self.comparison_receipt != _EXPECTED_COMPARISON_RECEIPT
            or self.binding.get("source_model_sha256")
            != EXPECTED_SOURCE_MODEL_SHA256
            or self.model.get("source_model_sha256")
            != EXPECTED_SOURCE_MODEL_SHA256
            or not isinstance(partition, Mapping)
            or partition.get("artifact_sha256")
            != EXPECTED_PARTITION_ARTIFACT_SHA256
            or not isinstance(grouped, Mapping)
            or grouped.get("artifact_sha256")
            != EXPECTED_GROUPED_BASIS_ARTIFACT_SHA256
            or grouped.get("basis_sha256")
            != EXPECTED_GROUPED_FULL_BASIS_SHA256
            or grouped.get("parent_subspace_artifact_sha256")
            != EXPECTED_PARENT_SUBSPACE_ARTIFACT_SHA256
            or construction.get("basis_sha256")
            != EXPECTED_RANK45_GRAPH_BASIS_SHA256
            or self.selection_receipt.get("source_basis_sha256")
            != EXPECTED_RANK45_GRAPH_BASIS_SHA256
        ):
            raise ValueError("signed-g8 candidate differs from frozen identity")

    def validate_integrity(self) -> None:
        self.plan.validate_integrity()
        self._validate_semantics()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("signed-g8 candidate artifact hash mismatch")

    def validate_frozen_identity(self) -> None:
        """Validate self-consistency and every audited signed-g8 identity."""

        self.validate_integrity()
        self._validate_pinned_identity()

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "plan": self.plan.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: object,
    ) -> "Gemma3L3L4GraphWaveletSignedG8Candidate":
        if not isinstance(value, Mapping):
            raise TypeError("signed-g8 candidate state must be a mapping")
        expected = {
            "schema",
            "format_version",
            "method",
            "source_basis_kind",
            "binding",
            "model",
            "source_receipt",
            "parent_receipt",
            "comparison_receipt",
            "construction_receipt",
            "selection_receipt",
            "protocol",
            "claims",
            "plan",
            "safety",
            "artifact_sha256",
        }
        if set(value) != expected or value.get("safety") != _SAFETY:
            raise ValueError("signed-g8 candidate state fields differ")
        plan_state = value["plan"]
        if not isinstance(plan_state, Mapping):
            raise TypeError("serialized signed-g8 plan must be a mapping")
        return cls(
            plan=ConditionalSpectralGeneratorPlan.from_state_dict(plan_state),
            binding=value["binding"],  # type: ignore[arg-type]
            model=value["model"],  # type: ignore[arg-type]
            source_receipt=value["source_receipt"],  # type: ignore[arg-type]
            parent_receipt=value["parent_receipt"],  # type: ignore[arg-type]
            comparison_receipt=value[
                "comparison_receipt"
            ],  # type: ignore[arg-type]
            construction_receipt=value[
                "construction_receipt"
            ],  # type: ignore[arg-type]
            selection_receipt=value[
                "selection_receipt"
            ],  # type: ignore[arg-type]
            protocol=value["protocol"],  # type: ignore[arg-type]
            claims=value["claims"],  # type: ignore[arg-type]
            method=value["method"],  # type: ignore[arg-type]
            source_basis_kind=value[
                "source_basis_kind"
            ],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )


def compile_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
    source: Gemma3SpectralSource,
    parent: Gemma3GraphWaveletCandidate,
    comparison: Gemma3GraphWaveletGroupedComparisonCandidate,
) -> Gemma3L3L4GraphWaveletSignedG8Candidate:
    """Replay the fit-only signed-g8 construction and freeze its exact plan."""

    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    if not isinstance(parent, Gemma3GraphWaveletCandidate):
        raise TypeError("parent must be a Gemma3GraphWaveletCandidate")
    if not isinstance(
        comparison,
        Gemma3GraphWaveletGroupedComparisonCandidate,
    ):
        raise TypeError(
            "comparison must be a grouped-comparison candidate"
        )
    source.mapping.validate_integrity()
    parent.validate_integrity()
    comparison.validate_integrity()
    if comparison.artifact_sha256 != EXPECTED_COMPARISON_ARTIFACT_SHA256:
        raise ValueError("comparison differs from frozen signed-g8 selection")
    if (
        comparison.source_receipt != _EXPECTED_SOURCE_RECEIPT
        or comparison.parent_receipt != _EXPECTED_PARENT_RECEIPT
        or comparison.conclusions.get("controlled_development_nominee_method")
        != METHOD
        or comparison.protocol.get("control_group_count") != GROUP_COUNT
        or comparison.protocol.get("target_source_rank")
        != TARGET_SOURCE_RANK
    ):
        raise ValueError("comparison provenance or selected method differs")

    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    source_receipt = {
        "tensor_file_sha256": source.file_sha256,
        "report_file_sha256": source.report_file_sha256,
        "report_payload_sha256": source.report_payload_sha256,
        "mapping_artifact_sha256": source.mapping.artifact_sha256,
        "response_artifact_sha256": local.artifact_sha256,
        "source_model_sha256": source.binding.get("source_model_sha256"),
    }
    if not _same_metadata(source_receipt, _EXPECTED_SOURCE_RECEIPT):
        raise ValueError("source receipt differs from frozen signed-g8 input")
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
        source_receipt=source_receipt,
        fft_length=source.mapping.fft_length,
        expected_parent_rank=64,
    )
    if (
        _graph_tensor_sha256(context.q64) != EXPECTED_Q64_SHA256
        or context.parent_subspace.artifact_sha256
        != EXPECTED_PARENT_SUBSPACE_ARTIFACT_SHA256
        or context.response_binding_sha256
        != EXPECTED_RESPONSE_BINDING_SHA256
    ):
        raise ValueError("reconstructed signed-g8 parent context differs")

    topology_top_k = min(TOPOLOGY_TOP_K, context.q64.shape[1] - 1)
    partition = fit_graph_wavelet_topology_partition(
        context.q64,
        context.signed_laplacian,
        group_count=GROUP_COUNT,
        topology_top_k=topology_top_k,
    )
    grouped = fit_graph_wavelet_grouped_basis(
        context.q64,
        context.weighted_fit,
        partition,
        method="wavelet_local_svd",
        fit_origins=FIT_ORIGINS,
        response_binding_sha256=context.response_binding_sha256,
        parent_subspace_artifact_sha256=(
            context.parent_subspace.artifact_sha256
        ),
    )
    basis = grouped.prefix(TARGET_SOURCE_RANK)
    construction_receipt = {
        **_grouped_receipt(
            parent_basis=context.q64,
            partition=partition,
            family=grouped,
            fit_folds=context.fit_folds,
            fit_origins=FIT_ORIGINS,
            response_binding_sha256=context.response_binding_sha256,
            parent_subspace_artifact_sha256=(
                context.parent_subspace.artifact_sha256
            ),
            rank=TARGET_SOURCE_RANK,
            topology_kind="signed",
        ),
        "basis_sha256": _graph_tensor_sha256(basis),
        "basis_mixing": _basis_mixing(context.q64, basis),
    }
    expected_construction = comparison.construction_receipts[METHOD]
    if not _same_metadata(construction_receipt, expected_construction):
        raise ValueError("reconstructed signed-g8 construction receipt differs")
    if (
        partition.artifact_sha256 != EXPECTED_PARTITION_ARTIFACT_SHA256
        or grouped.artifact_sha256
        != EXPECTED_GROUPED_BASIS_ARTIFACT_SHA256
        or grouped.metadata()["basis_sha256"]
        != EXPECTED_GROUPED_FULL_BASIS_SHA256
        or _graph_tensor_sha256(basis)
        != EXPECTED_RANK45_GRAPH_BASIS_SHA256
    ):
        raise ValueError("reconstructed signed-g8 basis identity differs")

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
    selected = _selected_row(comparison)
    if (
        plan.artifact_sha256 != EXPECTED_PLAN_ARTIFACT_SHA256
        or selected.get("plan_artifact_sha256") != plan.artifact_sha256
        or selected.get("source_basis_sha256")
        != EXPECTED_RANK45_GRAPH_BASIS_SHA256
    ):
        raise ValueError("reconstructed signed-g8 plan identity differs")

    candidate = Gemma3L3L4GraphWaveletSignedG8Candidate(
        plan=plan,
        binding=source.binding,
        model=source.model,
        source_receipt=source_receipt,
        parent_receipt=context.parent_receipt,
        comparison_receipt=_EXPECTED_COMPARISON_RECEIPT,
        construction_receipt=construction_receipt,
        selection_receipt=selected,
        protocol={
            "method": METHOD,
            "source_basis_kind": SOURCE_BASIS_KIND,
            "fit_origins": FIT_ORIGINS,
            "source_modes": plan.source_modes,
            "source_rank": plan.source_rank,
            "target_modes": plan.target_modes,
            "target_rank": plan.target_rank,
            "lag_count": plan.lag_count,
            "fft_length": plan.fft_length,
            "source_origin_minimum": FIT_ORIGINS[0],
            "source_origin_maximum": FIT_ORIGINS[-1],
            "supported_shadow_arms": ("identity", "all_on"),
            "candidate_serving_authorized": False,
            "plan_reconstructed_from_fit_origins_only": True,
            "selection_values_used_to_refit_plan": False,
        },
        claims=_CLAIMS,
    )
    candidate.validate_frozen_identity()
    return candidate


def _validate_local_output(output: Path | str) -> Path:
    destination = Path(output)
    if destination.suffix != ".pt":
        raise ValueError("signed-g8 output must use a .pt suffix")
    resolved = destination.resolve()
    local_root = (Path.cwd() / ".local-runs").resolve()
    if local_root != resolved and local_root not in resolved.parents:
        raise ValueError("signed-g8 output must remain under .local-runs")
    return destination


def _publish_candidate(
    candidate: Gemma3L3L4GraphWaveletSignedG8Candidate,
    *,
    output: Path,
) -> dict[str, object]:
    candidate.validate_frozen_identity()
    destination = _validate_local_output(output)
    report_path = destination.with_suffix(".json")
    reservation = _reserve_outputs((destination, report_path))
    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch(candidate.state_dict(), destination)
        tensor_digest = _file_sha256(tensor_stage)
        report: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "candidate": candidate.metadata(),
            "artifact": {
                "tensor_file": str(destination),
                "tensor_file_sha256": tensor_digest,
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
            "scientific_status": dict(candidate.claims),
            "safety": _SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        report_stage = _stage_json(report, report_path)
        reservation.publish((tensor_stage, report_stage))
        return report
    finally:
        reservation.release()
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> Gemma3L3L4GraphWaveletSignedG8Candidate:
    """Strictly authenticate and restore the frozen signed-g8 candidate."""

    source = Path(path)
    actual_file = _file_sha256(source)
    if actual_file != _require_sha256(
        expected_tensor_file_sha256,
        label="expected signed-g8 tensor file",
    ):
        raise ValueError("signed-g8 tensor file hash differs")
    with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    claimed_report = _require_sha256(
        report.get("report_sha256"),
        label="signed-g8 report SHA-256",
    )
    payload = dict(report)
    payload.pop("report_sha256")
    if (
        report.get("schema") != _SCHEMA
        or report.get("format_version") != _FORMAT_VERSION
        or report.get("safety") != _SAFETY
        or claimed_report
        != _require_sha256(
            expected_report_sha256,
            label="expected signed-g8 report",
        )
        or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed_report
        or report.get("artifact", {}).get("tensor_file_sha256")
        != actual_file
    ):
        raise ValueError("signed-g8 report binding differs")
    candidate = Gemma3L3L4GraphWaveletSignedG8Candidate.from_state_dict(
        torch.load(source, map_location="cpu", weights_only=True)
    )
    candidate.validate_frozen_identity()
    if (
        candidate.artifact_sha256
        != _require_sha256(
            expected_artifact_sha256,
            label="expected signed-g8 artifact",
        )
        or report.get("candidate", {}).get("artifact_sha256")
        != candidate.artifact_sha256
        or not _same_metadata(report.get("candidate"), candidate.metadata())
    ):
        raise ValueError("signed-g8 logical artifact differs")
    return candidate


def freeze_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    comparison_artifact_path: Path | str = DEFAULT_COMPARISON_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Strict-load the pinned lineage, reproduce signed-g8, and publish once."""

    destination = _validate_local_output(output)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite frozen signed-g8 artifact")
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
    comparison = load_gemma3_graph_wavelet_grouped_comparison_candidate(
        comparison_artifact_path,
        expected_artifact_sha256=EXPECTED_COMPARISON_ARTIFACT_SHA256,
        expected_tensor_file_sha256=EXPECTED_COMPARISON_TENSOR_FILE_SHA256,
        expected_report_sha256=EXPECTED_COMPARISON_REPORT_SHA256,
    )
    if (
        _file_sha256(Path(comparison_artifact_path).with_suffix(".json"))
        != EXPECTED_COMPARISON_REPORT_FILE_SHA256
    ):
        raise ValueError("grouped-comparison report file hash differs")
    candidate = compile_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        source,
        parent,
        comparison,
    )
    return _publish_candidate(candidate, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the exact Gemma L3/L4 signed-g8 candidate",
    )
    parser.add_argument("--source-artifact", default=DEFAULT_INTERIOR_ARTIFACT)
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument(
        "--comparison-artifact",
        default=DEFAULT_COMPARISON_ARTIFACT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = freeze_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        source_artifact_path=args.source_artifact,
        parent_artifact_path=args.parent_artifact,
        comparison_artifact_path=args.comparison_artifact,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
