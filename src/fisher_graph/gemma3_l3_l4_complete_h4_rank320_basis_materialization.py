"""Reconstruct and publish the exact successful complete-H4 D320 basis.

The successful A16 tail-informed factorial intentionally retained only scalar
receipts.  This rung replays the sixteen fit observations with a streamlined
two-forward/one-backward exact-X4 collector, proves every parent fit sequence
byte-for-byte, reconstructs the same unweighted U320 and tail-informed D320,
and publishes the coefficients in the closed local sidecar format.

This is materialization, not another capacity evaluation.  No projection arm,
transfer arm, validation prompt, serving path, or compression claim is opened.
The scalar JSON report is the commit marker for its atomically published
sidecar.  An exact authenticated orphan sidecar may be recovered after an
interrupted first attempt; no existing report or mismatched sidecar is ever
overwritten.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
    fit_complete_h4_projection_basis,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_sidecar import (
    DEFAULT_OUTPUT as DEFAULT_BASIS_OUTPUT,
    EXPECTED_BASIS_MATRIX_SHA256,
    EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256,
    EXPECTED_RUNTIME_TENSOR_SHA256,
    FIT_TO_PREFIX_LINEAGE_SHA256,
    PARENT_FILE_SHA256,
    PARENT_REPORT_SHA256,
    TAIL_INFORMED_FIT_ARTIFACT_SHA256,
    CompleteH4Rank320BasisSidecar,
    build_complete_h4_rank320_basis_sidecar,
    load_complete_h4_rank320_basis_sidecar,
    save_complete_h4_rank320_basis_sidecar,
)
from .gemma3_l3_l4_complete_h4_tail_informed_projection import (
    CompleteH4TailProjectionTrace,
    fit_complete_h4_tail_informed_projection,
)
from .gemma3_l3_l4_exact_x4_fit_observation import (
    Gemma3L3L4ExactX4FitObservation,
    collect_gemma3_l3_l4_exact_x4_fit_observation,
)
from .gemma3_l3_l4_conditional_spectral_shadow_runtime import (
    Gemma3L3L4ConditionalSpectralShadowRuntime,
)
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
    load_gemma3_graph_organized_svd_candidate,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    Gemma3L3L4OnePassBridge,
    _runtime_tensor_sha256,
)
from .graph_organized_svd import PreparedGraphOrganizedSVD


__all__ = [
    "DEFAULT_BASIS_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_PARENT_FACTORIAL",
    "CompleteH4Rank320LiveContext",
    "load_complete_h4_rank320_basis_materialization_report",
    "prepare_complete_h4_rank320_live_context",
    "run_gemma3_l3_l4_complete_h4_rank320_basis_materialization",
    "validate_complete_h4_rank320_runtime_graph_abi",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_PARENT_FACTORIAL = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-"
    "tail-informed-single-projector-factorial-a-fit16-dev-v1.json"
)
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-"
    "tail-informed-rank320-basis-materialization-a-fit16-dev-v1.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_complete_h4_tail_informed_rank320_basis_"
    "materialization"
)
_FORMAT_VERSION = 1
_ROLE = "reused_calibration_a_exact_rank320_basis_materialization"
_GRAPH_RUNTIME_BINDING_SHA256 = (
    "02e4914a0862eea1211ad9d5716b86546460c178f1b057a6ab7a01accc9551cb"
)
_GRAPH_BRIDGE_BINDING_SHA256 = (
    "dbbcd953566ded17a146f10ec01fb6b663b71929f9facea1ba4a43f9678cc70e"
)
_GRAPH_CARRIER_CONSUMED_BASIS_SHA256S = {
    "basis.x3_mean": (
        "e68859c2cc4a013930556c5a14bbde29cf5bb263a47ffa49a28a0a69e8dcd286"
    ),
    "basis.R3": (
        "2f3299ea2b764dacf9ac281397b3e705f65bd236a4bb7d6ff889f43889675783"
    ),
    "basis.P3": (
        "a6be6f241d1ef6b39dac1040e54b3b92acccf68c059dce1259930ecf1db2dcca"
    ),
}
_GRAPH_CARRIER_PROBE_SHA256S = {
    "source_modes": (
        "746dab80c0f016bf9cd4a9c17233106803ac44d095ef10f4529a8708dcf1fe41"
    ),
    "clamped_y3": (
        "d75ef12107f1b62db2bab212011bdbd54372457391e5847429ada343dd17b746"
    ),
    "source_eligible_mask": (
        "54f056d1b356965b87165a01df0387d6f910ed221e47747ffca2071f47dd9c80"
    ),
    "target_affected_mask": (
        "13aac5ec3d4e2f0e62c1f3f3d311ad7ab05454840d3f66a16d060a73ab8a68f8"
    ),
}
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-rank320-basis-materialization:"
    b"v1\0"
)
_PARENT_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-tail-informed-single-projector-"
    b"factorial:v1\0"
)
_PARENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_complete_h4_tail_informed_single_projector_"
    "factorial_development"
)
_PARENT_ROLE = (
    "reused_calibration_a_truth_leaking_complete_h4_tail_informed_"
    "single_global_projector_factorial"
)
_SELECTED_ARM = "tail_informed.rank320"
_WIDTH = 640
_RANK = 320
_ANCHOR_RANK = 192
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_SUPPORT_ROWS = 819
_EXPECTED_GRAPH_CORE_ROWS = 802
_EXPECTED_TAIL_ROWS = 17
_EXPECTED_GLOBAL_BASIS_ARTIFACT_SHA256 = (
    "418911d08d45577905b9bc964fc84ce86fc20eb6fe919cd7d601b2aef54b9de4"
)
_EXPECTED_GLOBAL_BASIS_MATRIX_SHA256 = (
    "e1b8d9462b2883828a0b4040c66a905e896192e14dce57e8a8b5924b4ec559fb"
)

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
    "basis_coefficients_exist_only_in_local_sidecar": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    return frozen._mapping(value, label=label)


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
            "rank320 materialization report must be JSON under .local-runs "
            "as a lexical repo-relative path without traversal or nesting"
        )
    return destination


def _load_parent_factorial(path: Path | str) -> dict[str, object]:
    source = Path(path)
    file_sha256 = frozen._file_sha256(source)
    if file_sha256 != PARENT_FILE_SHA256:
        raise ValueError("successful tail-informed factorial file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    report = dict(_mapping(raw, label="successful tail-informed factorial"))
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    selection = _mapping(report.get("selection"), label="factorial selection")
    status = _mapping(
        report.get("scientific_status"),
        label="factorial scientific status",
    )
    tail = _mapping(report.get("tail_informed_fit"), label="tail-informed fit")
    fitted = _mapping(report.get("fitted_bases"), label="factorial fitted bases")
    global_basis = _mapping(fitted.get("unweighted"), label="unweighted U320")
    treatment_rows = _mapping(
        tail.get("treatment_basis_rows"),
        label="tail-informed treatment rows",
    )
    global_rows = _mapping(
        global_basis.get("basis_rows"),
        label="unweighted U320 rows",
    )
    if (
        claimed != PARENT_REPORT_SHA256
        or frozen._json_sha256(payload, domain=_PARENT_REPORT_DOMAIN)
        != claimed
        or report.get("schema") != _PARENT_SCHEMA
        or report.get("format_version") != 1
        or report.get("role") != _PARENT_ROLE
        or selection.get("selected_arm") != _SELECTED_ARM
        or selection.get("factorial_capacity_gate_passed") is not True
        or selection.get(
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized"
        )
        is not True
        or status.get("tail_informed_factorial_complete") is not True
        or status.get("all_eight_arms_executed") is not True
        or status.get("live_exact_h4_ceiling_validated") is not True
        or status.get("calibration_b_opened") is not False
        or status.get("validation_opened") is not False
        or status.get("test_opened") is not False
        or tail.get("artifact_sha256") != TAIL_INFORMED_FIT_ARTIFACT_SHA256
        or tail.get("anchor_rank") != _ANCHOR_RANK
        or tail.get("tail_rank") != _EXPECTED_TAIL_ROWS
        or tail.get("max_rank") != _RANK
        or tail.get("source_example_count") != _EXPECTED_EXAMPLES
        or tail.get("source_row_count") != _EXPECTED_SUPPORT_ROWS
        or tail.get("source_tail_row_count") != _EXPECTED_TAIL_ROWS
        or treatment_rows.get("matrix_sha256")
        != EXPECTED_BASIS_MATRIX_SHA256
        or treatment_rows.get("shape") != [_RANK, _WIDTH]
        or global_basis.get("artifact_sha256")
        != _EXPECTED_GLOBAL_BASIS_ARTIFACT_SHA256
        or global_rows.get("matrix_sha256")
        != _EXPECTED_GLOBAL_BASIS_MATRIX_SHA256
        or global_rows.get("shape") != [_RANK, _WIDTH]
    ):
        raise ValueError("successful tail-informed factorial identity differs")
    receipts = report.get("collect_receipts")
    if not isinstance(receipts, list) or len(receipts) != _EXPECTED_EXAMPLES:
        raise ValueError("successful factorial collect receipts differ")
    if len(
        {
            _mapping(row, label="factorial collect receipt").get("example_id")
            for row in receipts
        }
    ) != _EXPECTED_EXAMPLES:
        raise ValueError("factorial collect example ids are not unique")
    receipt = {
        "file": str(source),
        "file_sha256": file_sha256,
        "report_sha256": claimed,
        "report": report,
    }
    return receipt


def _parent_lineage_by_example(
    parent_report: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    fit_manifest = _mapping(
        parent_report.get("fit_manifest"),
        label="parent fit manifest",
    )
    tail = _mapping(
        parent_report.get("tail_informed_fit"),
        label="parent tail-informed fit",
    )
    example_ids = tuple(fit_manifest.get("example_ids", ()))
    family_ids = tuple(fit_manifest.get("family_ids_by_example", ()))
    sequence_sha256s = tuple(fit_manifest.get("sequence_sha256s", ()))
    pair_sha256s = tuple(tail.get("source_pair_sha256s", ()))
    source_mask_sha256s = tuple(
        tail.get("source_graph_core_mask_sha256s", ())
    )
    trace_sha256s = tuple(tail.get("source_trace_sha256s", ()))
    trace_mask_sha256s = tuple(tail.get("graph_core_mask_sha256s", ()))
    groups = (
        example_ids,
        family_ids,
        sequence_sha256s,
        pair_sha256s,
        source_mask_sha256s,
        trace_sha256s,
        trace_mask_sha256s,
    )
    if any(len(group) != _EXPECTED_EXAMPLES for group in groups):
        raise ValueError("parent tail lineage lengths differ")
    raw_receipts = parent_report.get("collect_receipts")
    assert isinstance(raw_receipts, list)
    receipts = {
        str(_mapping(row, label="parent collect receipt")["example_id"]): (
            _mapping(row, label="parent collect receipt")
        )
        for row in raw_receipts
    }
    result: dict[str, dict[str, object]] = {}
    for index, example_id in enumerate(example_ids):
        if not isinstance(example_id, str) or example_id not in receipts:
            raise ValueError("parent fit manifest example differs")
        receipt = receipts[example_id]
        pair = _mapping(receipt.get("pair"), label="parent pair receipt")
        fit_sequence = _mapping(
            receipt.get("fit_sequence"),
            label="parent fit sequence",
        )
        if (
            receipt.get("family_id") != family_ids[index]
            or fit_sequence.get("sequence_sha256") != sequence_sha256s[index]
            or pair.get("artifact_sha256") != pair_sha256s[index]
        ):
            raise ValueError("parent fit/tail lineage is internally inconsistent")
        result[example_id] = {
            "family_id": family_ids[index],
            "sequence_sha256": sequence_sha256s[index],
            "source_pair_sha256": pair_sha256s[index],
            "source_graph_core_mask_sha256": source_mask_sha256s[index],
            "trace_sha256": trace_sha256s[index],
            "trace_graph_core_mask_sha256": trace_mask_sha256s[index],
            "receipt": receipt,
        }
    if len(result) != _EXPECTED_EXAMPLES:
        raise ValueError("parent fit manifest example ids are not unique")
    return result


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def validate_complete_h4_rank320_runtime_graph_abi(
    *,
    fit_runtime: Gemma3L3L4ConditionalSpectralShadowRuntime,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    bridge: Gemma3L3L4OnePassBridge,
) -> dict[str, object]:
    """Prove the D320 fit prefix survives promotion to the real graph ABI.

    The parent D320 fit used the conditional rank-64 measurement runtime.  The
    transfer rung must instead execute the signed-GFA graph runtime used by the
    progressive/iterative workers.  Exact-X4 fitting consumes source modes,
    the L3 clamp, and the source/target masks from its prefix; it deliberately
    replaces X4 and therefore does not admit either carrier's graph prediction.
    This prompt-free probe proves those consumed prefix fields are bitwise
    identical while also rejecting a conditional generator masquerading as a
    graph-organized one-pass bridge.
    """

    if type(fit_runtime) is not Gemma3L3L4ConditionalSpectralShadowRuntime:
        raise TypeError("fit_runtime must be the exact conditional runtime")
    if type(runtime) is not Gemma3L3L4GraphOrganizedSVDShadowRuntime:
        raise TypeError("runtime must be the genuine graph-organized runtime")
    if type(bridge) is not Gemma3L3L4OnePassBridge:
        raise TypeError("bridge must be the genuine graph one-pass bridge")
    fit_runtime.validate_integrity()
    runtime.validate_integrity()
    bridge.validate_integrity()
    if type(runtime._graph) is not PreparedGraphOrganizedSVD or type(
        bridge._graph
    ) is not PreparedGraphOrganizedSVD:
        raise RuntimeError("rank320 transfer lacks the prepared graph ABI")
    if (
        bridge.parent_runtime_binding_sha256
        != runtime.runtime_binding_sha256
        or bridge._graph.plan_sha256 != runtime._graph.plan_sha256
        or bridge._graph.pack_count != runtime._graph.pack_count
        or bridge._graph.pack_count <= 0
    ):
        raise RuntimeError("rank320 graph runtime/bridge lineage differs")
    geometry = (
        fit_runtime.residual_width,
        fit_runtime.source_modes,
        fit_runtime.target_modes,
        fit_runtime._plan.fit_knot_origins,
        fit_runtime._plan.lag_count,
    )
    graph_geometry = (
        bridge.residual_width,
        bridge.source_modes,
        bridge.target_modes,
        bridge.fit_knot_origins,
        bridge.lag_count,
    )
    if geometry != graph_geometry:
        raise RuntimeError("rank320 fit and graph prefix geometries differ")
    if (
        fit_runtime.live_model_sha256 != runtime.live_model_sha256
        or fit_runtime.adapter_execution_sha256
        != runtime.adapter_execution_sha256
    ):
        raise RuntimeError("rank320 fit and graph execution scopes differ")
    fit_tensors = fit_runtime._internal_tensors()
    bridge_tensors = bridge._internal_tensors()
    consumed_basis_names = ("basis.x3_mean", "basis.R3", "basis.P3")
    if any(
        name not in fit_tensors
        or name not in bridge_tensors
        or not _bitwise_equal(fit_tensors[name], bridge_tensors[name])
        for name in consumed_basis_names
    ):
        raise RuntimeError("rank320 fit and graph prefix bases differ")

    origins = tuple(int(value) for value in bridge.fit_knot_origins)
    positions = torch.tensor(
        [
            max(0, origins[0] - 1),
            *origins,
            origins[-1] + 1,
            origins[-1] + bridge.lag_count - 1,
        ],
        dtype=torch.int64,
    ).unique(sorted=True).unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)
    base = torch.linspace(
        -0.25,
        0.25,
        bridge.residual_width,
        dtype=torch.float32,
    )
    x3 = torch.stack(
        tuple(base.roll(shifts=index) for index in range(positions.shape[1])),
        dim=0,
    ).unsqueeze(0).contiguous()
    native_y3 = torch.flip(x3, dims=(-1,)).contiguous()
    zero_x4 = torch.zeros_like(x3)
    fit_probe = fit_runtime.execute_boundary_shadow(
        x3=x3,
        native_y3=native_y3,
        native_x4=zero_x4,
        reference_x4=zero_x4,
        logical_positions=positions,
        valid_mask=valid,
        arm="all_on",
    )
    graph_probe = bridge._prepare_prefix(
        x3=x3,
        native_y3=native_y3,
        logical_positions=positions,
        valid_mask=valid,
    )
    consumed_probe_fields = {
        "source_modes": (fit_probe.source_modes, graph_probe.source_modes),
        "clamped_y3": (fit_probe.clamped_y3, graph_probe.clamped_y3),
        "source_eligible_mask": (
            fit_probe.source_eligible_mask,
            graph_probe.source_eligible_mask,
        ),
        "target_affected_mask": (
            fit_probe.target_affected_mask,
            graph_probe.target_affected_mask,
        ),
    }
    if any(
        not _bitwise_equal(left, right)
        for left, right in consumed_probe_fields.values()
    ):
        raise RuntimeError("rank320 fit prefix changed on the graph carrier")
    fit_runtime.validate_integrity()
    runtime.validate_integrity()
    bridge.validate_integrity()
    receipt = {
        "d320_source_runtime_binding_sha256": (
            fit_runtime.runtime_binding_sha256
        ),
        "graph_target_runtime_binding_sha256": runtime.runtime_binding_sha256,
        "graph_target_bridge_binding_sha256": bridge.bridge_binding_sha256,
        "graph_plan_sha256": bridge._graph.plan_sha256,
        "graph_prepared_type": type(bridge._graph).__name__,
        "graph_pack_count": bridge._graph.pack_count,
        "fit_consumed_prefix_geometry": {
            "residual_width": bridge.residual_width,
            "source_modes": bridge.source_modes,
            "target_modes": bridge.target_modes,
            "fit_knot_origins": bridge.fit_knot_origins,
            "lag_count": bridge.lag_count,
        },
        "d320_source_graph_prediction_rank": fit_runtime._plan.source_rank,
        "graph_target_prediction_rank": bridge.source_rank,
        "prediction_rank_divergence_expected": (
            fit_runtime._plan.source_rank != bridge.source_rank
        ),
        "consumed_basis_tensor_sha256s": {
            name: _runtime_tensor_sha256(bridge_tensors[name])
            for name in consumed_basis_names
        },
        "prompt_free_probe_tensor_sha256s": {
            name: _runtime_tensor_sha256(right)
            for name, (_left, right) in consumed_probe_fields.items()
        },
        "fit_consumed_prefix_fields_bitwise_identical": True,
        "graph_prediction_admitted_to_exact_x4_fit_lineage": False,
        "preflight_model_forward_count": 0,
    }
    if (
        receipt["consumed_basis_tensor_sha256s"]
        != _GRAPH_CARRIER_CONSUMED_BASIS_SHA256S
        or receipt["prompt_free_probe_tensor_sha256s"]
        != _GRAPH_CARRIER_PROBE_SHA256S
    ):
        raise RuntimeError("rank320 graph carrier preflight receipt differs")
    return receipt


@dataclass(slots=True)
class CompleteH4Rank320LiveContext:
    """Authenticated factorized Gemma/A16 context shared by two closed rungs."""

    examples: tuple[Any, ...]
    adapter: Any
    fit_runtime: Gemma3L3L4ConditionalSpectralShadowRuntime
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime
    bridge: Gemma3L3L4OnePassBridge
    carrier_preflight: Mapping[str, object]
    parent: Mapping[str, object]
    panel_receipt: Mapping[str, object]
    max_length: int
    device: torch.device
    _tokenizer: Any
    _tokenizer_integrity_check: Callable[[str], None]
    _switcher: Any
    _tokenization_count: int = 0
    _closed: bool = False

    def tokenize(self, example: Any) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        if self._closed:
            raise RuntimeError("rank320 live context is closed")
        if not any(example is candidate for candidate in self.examples):
            raise ValueError("example does not belong to the authenticated A16 panel")
        self._tokenizer_integrity_check("before")
        result = frozen._tokenize_one(
            self._tokenizer,
            example.prompt,
            max_length=self.max_length,
            model_input_device=self.device,
        )
        self._tokenizer_integrity_check("after")
        self._tokenization_count += 1
        return result

    def validate_immutable_inputs(self) -> None:
        if self._closed:
            raise RuntimeError("rank320 live context is closed")
        self._tokenizer_integrity_check(
            "before" if self._tokenization_count == 0 else "after"
        )
        self.fit_runtime.validate_integrity()
        self.runtime.validate_integrity()
        self.bridge.validate_integrity()
        current_preflight = validate_complete_h4_rank320_runtime_graph_abi(
            fit_runtime=self.fit_runtime,
            runtime=self.runtime,
            bridge=self.bridge,
        )
        if frozen._canonical_json_bytes(current_preflight) != (
            frozen._canonical_json_bytes(self.carrier_preflight)
        ):
            raise RuntimeError("rank320 carrier preflight receipt drifted")
        factorized_model, factorized_execution = frozen._live_factorized_identity(
            self.adapter
        )
        lineage = _mapping(self.parent.get("lineage"), label="parent lineage")
        if (
            factorized_model
            != lineage.get("factorized_live_model_sha256")
            or factorized_execution
            != lineage.get("factorized_adapter_execution_sha256")
        ):
            raise RuntimeError("factorized Gemma identity drifted")

    def close(self) -> None:
        if self._closed:
            return
        self._switcher.close()
        self._closed = True
        if self.adapter.model_fingerprint() != frozen._EXPECTED_RAW_MODEL_SHA256:
            raise RuntimeError("rank320 live context did not restore raw Gemma")


def prepare_complete_h4_rank320_live_context(
    *,
    fit_source_artifact_path: Path | str = frozen.DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = frozen.DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = frozen.DEFAULT_CANDIDATE_ARTIFACT,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = frozen.DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = frozen.DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        frozen.DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    panel_path: Path | str = frozen.DEFAULT_PANEL,
    parent_factorial_path: Path | str = DEFAULT_PARENT_FACTORIAL,
    cache_dir: Path | str | None = None,
    max_length: int = frozen.DEFAULT_MAX_LENGTH,
) -> CompleteH4Rank320LiveContext:
    """Load the parent fit carrier and genuine deployable graph carrier.

    ``fit_runtime`` authenticates the exact conditional prefix lineage that
    produced D320.  ``runtime`` and ``bridge`` are the signed-GFA
    graph-organized pair used by progressive/iterative execution.  A
    prompt-free bitwise preflight binds their fit-consumed prefix semantics;
    conditional bridge export remains prohibited.
    """

    if type(max_length) is not int or max_length != frozen.DEFAULT_MAX_LENGTH:
        raise ValueError(
            f"max_length must equal locked value {frozen.DEFAULT_MAX_LENGTH}"
        )
    parent = _load_parent_factorial(parent_factorial_path)
    parent_report = _mapping(parent["report"], label="parent factorial report")
    examples, panel_receipt = frozen._load_panel(panel_path)
    if frozen._canonical_json_bytes(panel_receipt) != frozen._canonical_json_bytes(
        parent_report.get("panel")
    ):
        raise ValueError("live A16 panel differs from successful factorial")
    fit_source = frozen.load_gemma3_spectral_source(
        fit_source_artifact_path,
        expected_file_sha256=frozen.DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=frozen.DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=frozen.INTERIOR_ORIGINS,
    )
    graph_parent = frozen.load_gemma3_graph_wavelet_candidate(
        parent_artifact_path,
        expected_artifact_sha256=frozen.DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=frozen.DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=frozen.DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = frozen.load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=frozen.DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=frozen.DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=frozen.DEFAULT_FROZEN_REPORT_SHA256,
    )
    basis_package = frozen.load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=frozen.DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    plan, plan_receipt = frozen.build_rank64_global_svd_plan(
        fit_source,
        graph_parent,
    )
    if frozen._canonical_json_bytes(plan_receipt) != frozen._canonical_json_bytes(
        parent_report.get("rank64_x4_plan")
    ):
        raise ValueError("rebuilt rank64 plan differs from successful factorial")
    arm = _mapping(
        parent_report.get("rank64_x4_arm_receipt"),
        label="parent rank64 arm",
    )
    common = _mapping(arm.get("common_binding"), label="rank64 common binding")
    if (
        arm.get("artifact_sha256") != frozen._EXPECTED_RANK64_ARM_SHA256
        or common.get("signed_g8_candidate_artifact_sha256")
        != candidate.artifact_sha256
        or common.get("fit_response_tensor_file_sha256")
        != fit_source.file_sha256
        or common.get("parent_graph_wavelet_artifact_sha256")
        != graph_parent.artifact_sha256
        or common.get("basis_package_payload_sha256")
        != frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        or common.get("panel_file_sha256") != panel_receipt["file_sha256"]
        or common.get("max_length") != max_length
    ):
        raise ValueError("live rank64 arm inputs differ from successful factorial")

    protocol = frozen.default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    protocol_metadata = _mapping(
        protocol.metadata(),
        label="graph-organized shadow protocol",
    )
    graph_binding = _mapping(
        protocol_metadata.get("graph_candidate"),
        label="graph-organized candidate binding",
    )
    graph_basis_binding = _mapping(
        protocol_metadata.get("prompt_blind_basis"),
        label="graph-organized basis binding",
    )
    if (
        graph_basis_binding.get("tensor_file_sha256")
        != frozen.DEFAULT_BASIS_PACKAGE_FILE_SHA256
        or graph_basis_binding.get("logical_payload_sha256")
        != frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ):
        raise ValueError("graph carrier and parent fit basis lineage differ")
    graph_candidate = load_gemma3_graph_organized_svd_candidate(
        graph_candidate_path,
        expected_file_sha256=str(graph_binding["tensor_file_sha256"]),
    )
    tokenizer, tokenizer_contract = (
        frozen._load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    if (
        tokenizer_contract.get("configuration_sha256")
        != frozen._EXPECTED_TOKENIZER_CONFIGURATION_SHA256
        or tokenizer_contract.get("backend_serialized_sha256")
        != frozen._EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256
    ):
        raise ValueError("live tokenizer differs from successful factorial")
    tokenizer_integrity_check = frozen._frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )
    model_metadata = candidate.model
    if model_metadata.get("source_model_sha256") != frozen._EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("candidate raw model lineage differs")
    device = frozen.resolve_torch_device("cpu")
    cache = frozen.resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = frozen._load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = frozen.Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != frozen._EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("live raw Gemma differs from successful factorial")
    catalog = frozen.restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = frozen.PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {frozen._FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(frozen._FACTORIZED_SCOPE)
        factorized_model, factorized_execution = frozen._live_factorized_identity(
            adapter
        )
        fit_runtime = Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis_package,
            candidate_artifact_sha256=frozen._EXPECTED_RANK64_ARM_SHA256,
            candidate_method="global_svd_rank64_capacity_oracle",
            candidate_binding=candidate.binding,
            candidate_model=candidate.model,
            expected_plan_artifact_sha256=frozen._EXPECTED_RANK64_PLAN_SHA256,
            expected_basis_payload_sha256=(
                frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            expected_live_model_sha256=factorized_model,
            expected_adapter_execution_sha256=factorized_execution,
            analysis_device="cpu",
        )
        if frozen._canonical_json_bytes(fit_runtime.metadata()) != (
            frozen._canonical_json_bytes(parent_report.get("runtime_binding"))
        ):
            raise ValueError("live fit runtime differs from successful factorial")
        if (
            factorized_model
            != graph_binding.get("factorized_live_execution_sha256")
            or factorized_execution
            != graph_binding.get("factorized_refit_execution_sha256")
        ):
            raise ValueError("live Gemma differs from graph execution lineage")
        runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            graph_candidate,
            basis_package,
            expected_candidate_artifact_sha256=str(
                graph_binding["logical_artifact_sha256"]
            ),
            expected_basis_payload_sha256=str(
                graph_basis_binding["logical_payload_sha256"]
            ),
            expected_plan_artifact_sha256=str(
                graph_binding["deployment_plan_sha256"]
            ),
            expected_live_model_sha256=factorized_model,
            expected_adapter_execution_sha256=factorized_execution,
            analysis_device="cpu",
        )
        bridge = runtime.export_one_pass_bridge()
        carrier_preflight = validate_complete_h4_rank320_runtime_graph_abi(
            fit_runtime=fit_runtime,
            runtime=runtime,
            bridge=bridge,
        )
        if (
            runtime.runtime_binding_sha256 != _GRAPH_RUNTIME_BINDING_SHA256
            or bridge.bridge_binding_sha256 != _GRAPH_BRIDGE_BINDING_SHA256
        ):
            raise ValueError("live graph runtime/bridge identity differs")
        context = CompleteH4Rank320LiveContext(
            examples=tuple(examples),
            adapter=adapter,
            fit_runtime=fit_runtime,
            runtime=runtime,
            bridge=bridge,
            carrier_preflight=carrier_preflight,
            parent=parent_report,
            panel_receipt=panel_receipt,
            max_length=max_length,
            device=device,
            _tokenizer=tokenizer,
            _tokenizer_integrity_check=tokenizer_integrity_check,
            _switcher=switcher,
        )
        context.validate_immutable_inputs()
        return context
    except BaseException:
        switcher.close()
        raise


@dataclass(frozen=True, slots=True)
class _MaterializationTrace:
    sequence: CompleteH4ProjectionFitSequence
    graph_core_rows: Tensor
    source_pair_sha256: str
    source_graph_core_mask_sha256: str
    expected_trace_sha256: str
    expected_trace_graph_core_mask_sha256: str
    proof: Mapping[str, object]


def _authenticated_tail_traces(
    traces: Sequence[_MaterializationTrace],
) -> tuple[CompleteH4TailProjectionTrace, ...]:
    """Admit frozen legacy lineage only when the rebuilt trace is exact."""

    result: list[CompleteH4TailProjectionTrace] = []
    for trace in traces:
        tail_trace = CompleteH4TailProjectionTrace.from_fit_sequence(
            trace.sequence,
            trace.graph_core_rows,
            source_pair_sha256=trace.source_pair_sha256,
            source_graph_core_mask_sha256=(
                trace.source_graph_core_mask_sha256
            ),
        )
        if (
            tail_trace.trace_sha256 != trace.expected_trace_sha256
            or tail_trace.graph_core_mask_sha256
            != trace.expected_trace_graph_core_mask_sha256
        ):
            raise ValueError("reproduced tail trace differs from parent")
        result.append(tail_trace)
    return tuple(result)


def _row_difference_mask(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("H4 difference tensors differ")
    return (
        left.detach().contiguous().view(torch.uint8)
        != right.detach().contiguous().view(torch.uint8)
    ).reshape(*left.shape[:2], -1).any(dim=-1)


def _require_match(observed: object, expected: object, *, label: str) -> None:
    if observed != expected:
        raise ValueError(f"live exact-X4 {label} differs from successful parent")


def _fit_trace_from_observation(
    observation: Gemma3L3L4ExactX4FitObservation,
    *,
    example: Any,
    model_inputs: Mapping[str, Tensor],
    supervised_indices: Tensor,
    parent_lineage: Mapping[str, object],
) -> _MaterializationTrace:
    observation.validate_integrity()
    receipt = _mapping(parent_lineage.get("receipt"), label="parent collect receipt")
    pair = _mapping(receipt.get("pair"), label="parent pair receipt")
    parent_sequence = _mapping(
        receipt.get("fit_sequence"),
        label="parent fit sequence",
    )
    prompt_sha256 = frozen._prompt_sha256(example.prompt)
    support = observation.complete_h4_support_mask
    target = observation.prefix.target_affected_mask
    if (
        observation.native_h4.shape[0] != 1
        or support.shape[0] != 1
        or target.shape != support.shape
        or bool((target & ~support).any())
    ):
        raise RuntimeError("live exact-X4 support geometry differs")
    support_cpu = support[0].detach().to(device="cpu").contiguous()
    target_cpu = target[0].detach().to(device="cpu").contiguous()
    support_indices = torch.nonzero(
        support_cpu,
        as_tuple=False,
    ).flatten().to(dtype=torch.int64)
    graph_core_rows = target_cpu.index_select(0, support_indices).contiguous()
    device_indices = support_indices.to(observation.native_h4.device)
    # Preserve the exact parent arithmetic: select each live tensor first,
    # cast the operands independently to float64, then subtract.
    native_rows = observation.native_h4[0].index_select(0, device_indices)
    incomplete_rows = observation.incomplete_h4[0].index_select(
        0,
        device_indices,
    )
    residual_rows = native_rows.to(dtype=torch.float64) - incomplete_rows.to(
        dtype=torch.float64
    )
    gradient_rows = observation.h4_gradient[0].index_select(0, device_indices)
    sequence = CompleteH4ProjectionFitSequence(
        example_id=example.example_id,
        family_id=example.family_id,
        residual_rows=residual_rows,
        gradient_rows=gradient_rows,
    )
    current_target_sha256 = _runtime_tensor_sha256(target)
    expected_target_sha256 = str(
        parent_lineage.get("source_graph_core_mask_sha256")
    )
    difference = _row_difference_mask(
        observation.native_h4,
        observation.incomplete_h4,
    )
    expected_values = {
        "family id": (example.family_id, receipt.get("family_id")),
        "prompt hash": (prompt_sha256, receipt.get("prompt_sha256")),
        "tokenized token count": (
            int(model_inputs["input_ids"].shape[1]),
            receipt.get("tokenized_tokens"),
        ),
        "supervised token count": (
            int(supervised_indices.numel()),
            receipt.get("supervised_tokens"),
        ),
        "model-input hash": (
            observation.model_inputs_sha256,
            receipt.get("model_inputs_sha256"),
        ),
        "execution-grid hash": (
            observation.execution_grid_sha256,
            receipt.get("execution_grid_sha256"),
        ),
        "native H4 hash": (
            observation.native_h4_sha256,
            pair.get("native_h4_sha256"),
        ),
        "incomplete H4 hash": (
            observation.incomplete_h4_sha256,
            pair.get("incomplete_h4_sha256"),
        ),
        "H4 gradient hash": (
            observation.h4_gradient_sha256,
            pair.get("h4_gradient_sha256"),
        ),
        "partial exact-X4 logits hash": (
            observation.partial_exact_x4_logits_sha256,
            pair.get("partial_exact_x4_logits_sha256"),
        ),
        "supervised-indices hash": (
            observation.supervised_indices_sha256,
            pair.get("supervised_indices_sha256"),
        ),
        "supervised-targets hash": (
            observation.supervised_targets_sha256,
            pair.get("supervised_targets_sha256"),
        ),
        "NLL objective receipt": (
            observation.objective_receipt_sha256,
            pair.get("objective_receipt_sha256"),
        ),
        "NLL objective mean": (
            observation.objective_mean_nll,
            pair.get("objective_mean_nll"),
        ),
        "NLL objective ignore index": (
            observation.objective_ignore_index,
            pair.get("objective_ignore_index"),
        ),
        "NLL supervised count": (
            observation.supervised_token_count,
            pair.get("supervised_token_count"),
        ),
        "adapter execution hash": (
            observation.adapter_execution_sha256,
            pair.get("adapter_execution_sha256"),
        ),
        "source-mode hash": (
            _runtime_tensor_sha256(observation.prefix.source_modes),
            pair.get("source_modes_sha256"),
        ),
        "complete-H4 support hash": (
            _runtime_tensor_sha256(support),
            pair.get("complete_h4_support_mask_sha256"),
        ),
        "graph-core source-mask hash": (
            current_target_sha256,
            expected_target_sha256,
        ),
        "complete-H4 support rows": (
            int(support.sum()),
            pair.get("complete_h4_support_rows"),
        ),
        "graph-core rows": (
            int(target.sum()),
            pair.get("graph_target_affected_rows"),
        ),
        "support outside graph rows": (
            int((support & ~target).sum()),
            pair.get("complete_h4_support_outside_graph_rows"),
        ),
        "H4 difference rows": (
            int(difference.sum()),
            pair.get("incomplete_h4_difference_rows"),
        ),
        "H4 padding difference rows": (
            int((difference & ~observation.prefix.valid_target_mask).sum()),
            pair.get("incomplete_h4_difference_padding_rows"),
        ),
        "H4 difference outside support rows": (
            int((difference & ~support).sum()),
            pair.get("incomplete_h4_difference_outside_support_rows"),
        ),
    }
    for label, (observed, expected) in expected_values.items():
        _require_match(observed, expected, label=label)
    if pair.get("objective_reduction") != "mean":
        raise ValueError("successful parent objective reduction differs")
    if frozen._canonical_json_bytes(sequence.metadata()) != (
        frozen._canonical_json_bytes(parent_sequence)
    ):
        raise ValueError("live exact-X4 full fit sequence differs from parent")
    _require_match(
        sequence.sequence_sha256,
        parent_lineage.get("sequence_sha256"),
        label="fit sequence lineage",
    )
    proof = {
        "example_id": example.example_id,
        "family_id": example.family_id,
        "prompt_sha256": prompt_sha256,
        "observation_artifact_sha256": observation.artifact_sha256,
        "bridge_binding_sha256": observation.bridge_binding_sha256,
        "prefix_artifact_sha256": observation.prefix.artifact_sha256,
        "native_x4_sha256": observation.native_x4_sha256,
        "native_h4_sha256": observation.native_h4_sha256,
        "incomplete_h4_sha256": observation.incomplete_h4_sha256,
        "h4_gradient_sha256": observation.h4_gradient_sha256,
        "native_logits_sha256": observation.native_logits_sha256,
        "partial_exact_x4_logits_sha256": (
            observation.partial_exact_x4_logits_sha256
        ),
        "objective_receipt_sha256": observation.objective_receipt_sha256,
        "source_modes_sha256": _runtime_tensor_sha256(
            observation.prefix.source_modes
        ),
        "complete_h4_support_mask_sha256": _runtime_tensor_sha256(support),
        "graph_core_mask_sha256": current_target_sha256,
        "residual_rows_matrix_sha256": sequence.residual_rows.matrix_sha256,
        "gradient_rows_matrix_sha256": sequence.gradient_rows.matrix_sha256,
        "fit_sequence_sha256": sequence.sequence_sha256,
        "frozen_parent_pair_sha256": parent_lineage.get("source_pair_sha256"),
        "all_parent_tensor_objective_and_sequence_receipts_matched": True,
        "legacy_shadow_reexecuted": False,
    }
    return _MaterializationTrace(
        sequence=sequence,
        graph_core_rows=graph_core_rows,
        source_pair_sha256=str(parent_lineage["source_pair_sha256"]),
        source_graph_core_mask_sha256=expected_target_sha256,
        expected_trace_sha256=str(parent_lineage["trace_sha256"]),
        expected_trace_graph_core_mask_sha256=str(
            parent_lineage["trace_graph_core_mask_sha256"]
        ),
        proof=proof,
    )


def _existing_sidecar_receipt(
    path: Path | str,
    *,
    logical_artifact_sha256: str,
) -> dict[str, object]:
    source = Path(path)
    absolute = source if source.is_absolute() else Path.cwd() / source
    absolute = absolute.absolute()
    return {
        "file": str(absolute),
        "file_sha256": frozen._file_sha256(absolute),
        "file_bytes": absolute.stat().st_size,
        "logical_artifact_sha256": logical_artifact_sha256,
        "committable": False,
    }


def _publish_or_recover_exact_sidecar(
    d320: Tensor,
    *,
    basis_destination: Path,
    existing_sidecar: CompleteH4Rank320BasisSidecar | None,
) -> tuple[CompleteH4Rank320BasisSidecar, dict[str, object], bool]:
    """Publish D320 or accept only the exact authenticated orphan sidecar."""

    sidecar = build_complete_h4_rank320_basis_sidecar(d320)
    if existing_sidecar is None:
        receipt = save_complete_h4_rank320_basis_sidecar(
            sidecar,
            basis_destination,
        )
        return sidecar, receipt, False
    existing_sidecar.validate_integrity()
    if (
        existing_sidecar.artifact_sha256 != sidecar.artifact_sha256
        or not torch.equal(existing_sidecar.basis_tensor(), d320)
    ):
        raise ValueError("existing rank320 sidecar differs from reproduction")
    receipt = _existing_sidecar_receipt(
        basis_destination,
        logical_artifact_sha256=existing_sidecar.artifact_sha256,
    )
    return sidecar, receipt, True


def load_complete_h4_rank320_basis_materialization_report(
    path: Path | str = DEFAULT_OUTPUT,
    *,
    expected_file_sha256: str | None = None,
    expected_report_sha256: str | None = None,
) -> dict[str, object]:
    """Load the commit marker and reauthenticate its exact local sidecar."""

    source = _validate_output(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    file_sha256 = frozen._file_sha256(source)
    report = dict(_mapping(raw, label="rank320 materialization report"))
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    artifact = _mapping(report.get("artifact"), label="materialization artifact")
    sidecar_receipt = _mapping(
        artifact.get("basis_sidecar"),
        label="materialization basis sidecar",
    )
    status = _mapping(
        report.get("scientific_status"),
        label="materialization scientific status",
    )
    expected_top_level_keys = {
        "schema",
        "format_version",
        "role",
        "lineage",
        "protocol",
        "carrier_promotion",
        "fit_manifest",
        "reproduction",
        "materialized_basis",
        "resource_accounting",
        "publication",
        "scientific_status",
        "artifact",
        "safety",
        "report_sha256",
    }
    if (
        set(report) != expected_top_level_keys
        or set(artifact) != {"file", "basis_sidecar", "committable"}
        or set(sidecar_receipt)
        != {
            "file",
            "file_sha256",
            "file_bytes",
            "logical_artifact_sha256",
            "committable",
        }
        or artifact.get("file") != str(source)
        or artifact.get("committable") is not False
        or
        (expected_file_sha256 is not None and file_sha256 != expected_file_sha256)
        or (
            expected_report_sha256 is not None
            and claimed != expected_report_sha256
        )
        or
        report.get("schema") != _SCHEMA
        or report.get("format_version") != _FORMAT_VERSION
        or report.get("role") != _ROLE
        or not isinstance(claimed, str)
        or frozen._json_sha256(payload, domain=_REPORT_DOMAIN) != claimed
        or status.get("exact_rank320_basis_materialized") is not True
        or status.get("transfer_evaluation_run") is not False
        or sidecar_receipt.get("committable") is not False
    ):
        raise ValueError("rank320 materialization report identity differs")
    sidecar_path = sidecar_receipt.get("file")
    if not isinstance(sidecar_path, str):
        raise ValueError("materialization sidecar file is absent")
    sidecar = load_complete_h4_rank320_basis_sidecar(sidecar_path)
    actual = _existing_sidecar_receipt(
        sidecar_path,
        logical_artifact_sha256=sidecar.artifact_sha256,
    )
    for name in (
        "file_sha256",
        "file_bytes",
        "logical_artifact_sha256",
        "committable",
    ):
        if actual[name] != sidecar_receipt.get(name):
            raise ValueError("materialization sidecar receipt differs")
    lineage = _mapping(report.get("lineage"), label="materialization lineage")
    expected_lineage = {
        "parent_factorial_file_sha256": PARENT_FILE_SHA256,
        "parent_factorial_report_sha256": PARENT_REPORT_SHA256,
        "selected_arm": _SELECTED_ARM,
        "global_fit_basis_artifact_sha256": (
            _EXPECTED_GLOBAL_BASIS_ARTIFACT_SHA256
        ),
        "tail_informed_fit_artifact_sha256": (
            TAIL_INFORMED_FIT_ARTIFACT_SHA256
        ),
        "fit_to_prefix_lineage_sha256": FIT_TO_PREFIX_LINEAGE_SHA256,
        "projection_basis_artifact_sha256": (
            EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256
        ),
    }
    if frozen._canonical_json_bytes(lineage) != frozen._canonical_json_bytes(
        expected_lineage
    ):
        raise ValueError("materialization parent lineage differs")
    parent = _load_parent_factorial(DEFAULT_PARENT_FACTORIAL)
    parent_report = _mapping(parent["report"], label="parent factorial report")
    parent_runtime = _mapping(
        parent_report.get("runtime_binding"),
        label="parent fit runtime binding",
    )
    carrier = _mapping(
        report.get("carrier_promotion"),
        label="materialization carrier promotion",
    )
    carrier_geometry = _mapping(
        carrier.get("fit_consumed_prefix_geometry"),
        label="materialization carrier prefix geometry",
    )
    carrier_basis = _mapping(
        carrier.get("consumed_basis_tensor_sha256s"),
        label="materialization carrier basis tensors",
    )
    carrier_probe = _mapping(
        carrier.get("prompt_free_probe_tensor_sha256s"),
        label="materialization carrier probe tensors",
    )
    carrier_sha_values = tuple(carrier_basis.values()) + tuple(
        carrier_probe.values()
    )
    if (
        set(carrier)
        != {
            "d320_source_runtime_binding_sha256",
            "graph_target_runtime_binding_sha256",
            "graph_target_bridge_binding_sha256",
            "graph_plan_sha256",
            "graph_prepared_type",
            "graph_pack_count",
            "fit_consumed_prefix_geometry",
            "d320_source_graph_prediction_rank",
            "graph_target_prediction_rank",
            "prediction_rank_divergence_expected",
            "consumed_basis_tensor_sha256s",
            "prompt_free_probe_tensor_sha256s",
            "fit_consumed_prefix_fields_bitwise_identical",
            "graph_prediction_admitted_to_exact_x4_fit_lineage",
            "preflight_model_forward_count",
        }
        or carrier.get("d320_source_runtime_binding_sha256")
        != parent_runtime.get("runtime_binding_sha256")
        or carrier.get("graph_target_runtime_binding_sha256")
        != _GRAPH_RUNTIME_BINDING_SHA256
        or carrier.get("graph_target_bridge_binding_sha256")
        != _GRAPH_BRIDGE_BINDING_SHA256
        or carrier.get("graph_plan_sha256")
        != "10299119071f215c97979edf6b02bec4e7e7cde5a6d2c316a662b802a84aa469"
        or carrier.get("graph_prepared_type") != "PreparedGraphOrganizedSVD"
        or carrier.get("graph_pack_count") != 4
        or frozen._canonical_json_bytes(carrier_geometry)
        != frozen._canonical_json_bytes(
            {
                "residual_width": 640,
                "source_modes": 64,
                "target_modes": 64,
                "fit_knot_origins": (8, 24, 40),
                "lag_count": 32,
            }
        )
        or carrier.get("d320_source_graph_prediction_rank") != 64
        or carrier.get("graph_target_prediction_rank") != 45
        or carrier.get("prediction_rank_divergence_expected") is not True
        or frozen._canonical_json_bytes(carrier_basis)
        != frozen._canonical_json_bytes(
            _GRAPH_CARRIER_CONSUMED_BASIS_SHA256S
        )
        or frozen._canonical_json_bytes(carrier_probe)
        != frozen._canonical_json_bytes(_GRAPH_CARRIER_PROBE_SHA256S)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in carrier_sha_values
        )
        or carrier.get("fit_consumed_prefix_fields_bitwise_identical")
        is not True
        or carrier.get("graph_prediction_admitted_to_exact_x4_fit_lineage")
        is not False
        or carrier.get("preflight_model_forward_count") != 0
    ):
        raise ValueError("materialization carrier promotion differs")
    parent_lineage = _parent_lineage_by_example(parent_report)
    if frozen._canonical_json_bytes(report.get("fit_manifest")) != (
        frozen._canonical_json_bytes(parent_report.get("fit_manifest"))
    ):
        raise ValueError("materialization fit manifest differs from parent")
    protocol = _mapping(
        report.get("protocol"),
        label="materialization protocol",
    )
    expected_protocol = {
        "panel": "reused_calibration_a_fit16",
        "collection_per_prompt": (
            "native_boundary_forward_then_clamped_y3_exact_x4_h4_vjp"
        ),
        "model_forwards_per_prompt": 2,
        "backwards_per_prompt": 1,
        "objective": "prompt_mean_next_token_nll",
        "residual_row_arithmetic": (
            "select_then_native_to_float64_minus_incomplete_to_float64"
        ),
        "fit_ordering": "family_id_then_example_id",
        "global_fit": "family_example_macro_unweighted_u320",
        "tail_fit": (
            "u192_then_full_tail_residual_svd_span_then_two_pass_mgs_u320"
        ),
        "legacy_pair_and_graph_mask_lineage_policy": (
            "frozen_parent_receipts_reused_only_after_exact_live_tensor_"
            "objective_sequence_and_mask_proof"
        ),
        "transfer_evaluation": "not_run",
    }
    if frozen._canonical_json_bytes(protocol) != frozen._canonical_json_bytes(
        expected_protocol
    ):
        raise ValueError("materialization protocol differs")
    materialized = _mapping(
        report.get("materialized_basis"),
        label="materialized basis",
    )
    sidecar_metadata = sidecar.metadata()
    if set(materialized) != set(sidecar_metadata) | {
        "sidecar_file_sha256",
        "sidecar_file_bytes",
    } or any(
        materialized.get(name) != value
        for name, value in sidecar_metadata.items()
    ) or (
        materialized.get("sidecar_file_sha256") != actual["file_sha256"]
        or materialized.get("sidecar_file_bytes") != actual["file_bytes"]
    ):
        raise ValueError("materialized D320 metadata differs from sidecar")
    resources = _mapping(
        report.get("resource_accounting"),
        label="materialization resources",
    )
    expected_resources = {
        "model_load_count": 1,
        "tokenizer_load_count": 1,
        "model_forward_count": 32,
        "backward_count": 16,
        "evaluation_model_forward_count": 0,
        "projection_arm_model_forward_count": 0,
        "transfer_model_forward_count": 0,
        "simultaneously_live_full_vocabulary_logit_tensor_peak": 1,
        "full_vocabulary_logit_roles_at_peak": (
            "native_or_partial_exact_x4_never_both",
        ),
        "retained_fit_sequence_count": 16,
        "retained_fit_sequence_residual_and_gradient_bytes": (
            _EXPECTED_SUPPORT_ROWS * _WIDTH * 8 * 2
        ),
        "d320_float64_coefficient_count": _RANK * _WIDTH,
        "d320_float64_matrix_bytes": _RANK * _WIDTH * 8,
        "float64_width_square_eigendecomposition_count": 1,
        "tail_residual_float64_svd_count": 1,
        "inference_projection_macs_executed": 0,
        "basis_fit_linear_algebra_is_offline_only": True,
        "latency_or_speed_claim": False,
        "whole_model_parameter_reduction_claim": False,
    }
    if frozen._canonical_json_bytes(resources) != frozen._canonical_json_bytes(
        expected_resources
    ):
        raise ValueError("materialization resource accounting differs")
    expected_status = {
        "exact_rank320_basis_materialized": True,
        "same_a_fit_only": True,
        "transfer_evaluation_run": False,
        "frozen_basis_one_pass_carrier_transfer_ready": True,
        "candidate_serving_authorized": False,
        "generator_validated": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "next_rung": "fixed_d320_one_pass_carrier_transfer_oracle",
    }
    if frozen._canonical_json_bytes(status) != frozen._canonical_json_bytes(
        expected_status
    ) or frozen._canonical_json_bytes(report.get("safety")) != (
        frozen._canonical_json_bytes(_SAFETY)
    ):
        raise ValueError("materialization status or safety differs")
    publication = _mapping(
        report.get("publication"),
        label="materialization publication protocol",
    )
    if (
        set(publication)
        != {
            "report_is_sidecar_commit_marker",
            "consumer_must_require_report_and_sidecar",
            "sidecar_atomic_no_overwrite_publication",
            "report_atomic_no_overwrite_publication",
            "recovered_existing_authenticated_orphan_sidecar",
            "mismatched_existing_sidecar_accepted",
            "existing_report_overwrite_allowed",
        }
        or publication.get("report_is_sidecar_commit_marker") is not True
        or publication.get("consumer_must_require_report_and_sidecar") is not True
        or publication.get("sidecar_atomic_no_overwrite_publication") is not True
        or publication.get("report_atomic_no_overwrite_publication") is not True
        or type(
            publication.get("recovered_existing_authenticated_orphan_sidecar")
        )
        is not bool
        or publication.get("mismatched_existing_sidecar_accepted") is not False
        or publication.get("existing_report_overwrite_allowed") is not False
    ):
        raise ValueError("materialization publication protocol differs")
    reproduction = _mapping(
        report.get("reproduction"),
        label="materialization reproduction proof",
    )
    proofs = reproduction.get("observation_receipts")
    if (
        set(reproduction)
        != {
            "observation_receipts",
            "all_16_parent_fit_sequences_reproduced_exactly",
            "all_16_parent_tail_traces_reproduced_exactly",
            "unweighted_u320_metadata_reproduced_exactly",
            "tail_informed_fit_metadata_reproduced_exactly",
            "support_rows",
            "graph_core_rows",
            "causal_tail_rows",
            "legacy_three_pass_shadow_reexecuted",
            "frozen_pair_lineage_reused_after_exact_proof",
            "frozen_graph_mask_lineage_reused_after_exact_proof",
        }
        or
        not isinstance(proofs, list)
        or len(proofs) != _EXPECTED_EXAMPLES
        or reproduction.get(
            "all_16_parent_fit_sequences_reproduced_exactly"
        )
        is not True
        or reproduction.get(
            "all_16_parent_tail_traces_reproduced_exactly"
        )
        is not True
        or reproduction.get("unweighted_u320_metadata_reproduced_exactly")
        is not True
        or reproduction.get("tail_informed_fit_metadata_reproduced_exactly")
        is not True
        or reproduction.get("support_rows") != _EXPECTED_SUPPORT_ROWS
        or reproduction.get("graph_core_rows") != _EXPECTED_GRAPH_CORE_ROWS
        or reproduction.get("causal_tail_rows") != _EXPECTED_TAIL_ROWS
        or reproduction.get("legacy_three_pass_shadow_reexecuted") is not False
        or reproduction.get("frozen_pair_lineage_reused_after_exact_proof")
        is not True
        or reproduction.get("frozen_graph_mask_lineage_reused_after_exact_proof")
        is not True
    ):
        raise ValueError("materialization reproduction proof differs")
    expected_ids = tuple(
        _mapping(parent_report.get("fit_manifest"), label="parent fit manifest")
        .get("example_ids", ())
    )
    if tuple(
        _mapping(row, label="materialization observation proof").get("example_id")
        for row in proofs
    ) != expected_ids:
        raise ValueError("materialization observation proof order differs")
    for raw_proof in proofs:
        proof = _mapping(raw_proof, label="materialization observation proof")
        expected_proof_keys = {
            "example_id",
            "family_id",
            "prompt_sha256",
            "observation_artifact_sha256",
            "bridge_binding_sha256",
            "prefix_artifact_sha256",
            "native_x4_sha256",
            "native_h4_sha256",
            "incomplete_h4_sha256",
            "h4_gradient_sha256",
            "native_logits_sha256",
            "partial_exact_x4_logits_sha256",
            "objective_receipt_sha256",
            "source_modes_sha256",
            "complete_h4_support_mask_sha256",
            "graph_core_mask_sha256",
            "residual_rows_matrix_sha256",
            "gradient_rows_matrix_sha256",
            "fit_sequence_sha256",
            "frozen_parent_pair_sha256",
            "all_parent_tensor_objective_and_sequence_receipts_matched",
            "legacy_shadow_reexecuted",
        }
        if set(proof) != expected_proof_keys:
            raise ValueError("materialization observation proof keyset differs")
        example_id = proof.get("example_id")
        if not isinstance(example_id, str) or example_id not in parent_lineage:
            raise ValueError("materialization observation example differs")
        expected = parent_lineage[example_id]
        receipt = _mapping(expected["receipt"], label="parent collect receipt")
        pair = _mapping(receipt.get("pair"), label="parent pair receipt")
        sequence = _mapping(
            receipt.get("fit_sequence"),
            label="parent fit sequence",
        )
        residual = _mapping(
            sequence.get("residual_rows"),
            label="parent residual rows",
        )
        gradient = _mapping(
            sequence.get("gradient_rows"),
            label="parent gradient rows",
        )
        pinned = {
            "family_id": receipt.get("family_id"),
            "prompt_sha256": receipt.get("prompt_sha256"),
            "native_h4_sha256": pair.get("native_h4_sha256"),
            "incomplete_h4_sha256": pair.get("incomplete_h4_sha256"),
            "h4_gradient_sha256": pair.get("h4_gradient_sha256"),
            "partial_exact_x4_logits_sha256": (
                pair.get("partial_exact_x4_logits_sha256")
            ),
            "objective_receipt_sha256": pair.get("objective_receipt_sha256"),
            "source_modes_sha256": pair.get("source_modes_sha256"),
            "complete_h4_support_mask_sha256": (
                pair.get("complete_h4_support_mask_sha256")
            ),
            "graph_core_mask_sha256": expected.get(
                "source_graph_core_mask_sha256"
            ),
            "residual_rows_matrix_sha256": residual.get("matrix_sha256"),
            "gradient_rows_matrix_sha256": gradient.get("matrix_sha256"),
            "fit_sequence_sha256": sequence.get("sequence_sha256"),
            "frozen_parent_pair_sha256": expected.get("source_pair_sha256"),
        }
        if any(proof.get(name) != value for name, value in pinned.items()):
            raise ValueError("materialization observation proof receipt differs")
        if proof.get("bridge_binding_sha256") != _GRAPH_BRIDGE_BINDING_SHA256:
            raise ValueError(
                "materialization observation bridge lineage differs"
            )
        current_only_hashes = (
            "observation_artifact_sha256",
            "bridge_binding_sha256",
            "prefix_artifact_sha256",
            "native_x4_sha256",
            "native_logits_sha256",
        )
        if any(
            not isinstance(proof.get(name), str)
            or len(str(proof[name])) != 64
            or any(character not in "0123456789abcdef" for character in str(proof[name]))
            for name in current_only_hashes
        ) or proof.get(
            "all_parent_tensor_objective_and_sequence_receipts_matched"
        ) is not True or proof.get("legacy_shadow_reexecuted") is not False:
            raise ValueError("materialization current observation proof differs")
    return {
        **report,
        "artifact": {
            **dict(artifact),
            "file_sha256": file_sha256,
            "file_bytes": source.stat().st_size,
        },
    }


def run_gemma3_l3_l4_complete_h4_rank320_basis_materialization(
    *,
    fit_source_artifact_path: Path | str = frozen.DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = frozen.DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = frozen.DEFAULT_CANDIDATE_ARTIFACT,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = frozen.DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = frozen.DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        frozen.DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    panel_path: Path | str = frozen.DEFAULT_PANEL,
    parent_factorial_path: Path | str = DEFAULT_PARENT_FACTORIAL,
    basis_output: Path | str = DEFAULT_BASIS_OUTPUT,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = frozen.DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    """Reproduce all A16 fit sequences and publish the exact selected D320."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite rank320 commit-marker report")
    basis_destination = Path(basis_output)
    existing_sidecar = None
    recovered_existing_sidecar = basis_destination.exists()
    if recovered_existing_sidecar:
        # This is the sole recovery state: an interrupted prior attempt may
        # have durably linked the exact closed sidecar before its report.  It
        # must fully reauthenticate now and is never overwritten.
        existing_sidecar = load_complete_h4_rank320_basis_sidecar(
            basis_destination
        )

    context = prepare_complete_h4_rank320_live_context(
        fit_source_artifact_path=fit_source_artifact_path,
        parent_artifact_path=parent_artifact_path,
        candidate_artifact_path=candidate_artifact_path,
        graph_candidate_path=graph_candidate_path,
        basis_package_path=basis_package_path,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        panel_path=panel_path,
        parent_factorial_path=parent_factorial_path,
        cache_dir=cache_dir,
        max_length=max_length,
    )
    traces: list[_MaterializationTrace] = []
    try:
        parent = context.parent
        carrier_preflight = dict(context.carrier_preflight)
        parent_lineage = _parent_lineage_by_example(parent)
        for example in sorted(context.examples, key=lambda row: row.example_id):
            model_inputs, supervised_indices, supervised_targets = (
                context.tokenize(example)
            )
            grid_indices = frozen._supervised_grid_indices(
                supervised_indices
            )
            targets_cpu = (
                supervised_targets.detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous()
            )
            observation = collect_gemma3_l3_l4_exact_x4_fit_observation(
                context.bridge,
                context.adapter,
                model_inputs,
                supervised_indices=grid_indices,
                supervised_targets=targets_cpu,
                ignore_index=-100,
            )
            lineage = parent_lineage.get(example.example_id)
            if lineage is None:
                raise ValueError("live A16 example is absent from parent lineage")
            traces.append(
                _fit_trace_from_observation(
                    observation,
                    example=example,
                    model_inputs=model_inputs,
                    supervised_indices=supervised_indices,
                    parent_lineage=lineage,
                )
            )
            del (
                model_inputs,
                supervised_indices,
                supervised_targets,
                grid_indices,
                targets_cpu,
                observation,
            )
        context.validate_immutable_inputs()
    finally:
        context.close()

    if (
        len(traces) != _EXPECTED_EXAMPLES
        or len({trace.sequence.example_id for trace in traces})
        != _EXPECTED_EXAMPLES
        or len({trace.sequence.family_id for trace in traces})
        != _EXPECTED_FAMILIES
        or sum(trace.sequence.row_count for trace in traces)
        != _EXPECTED_SUPPORT_ROWS
        or sum(int(trace.graph_core_rows.sum()) for trace in traces)
        != _EXPECTED_GRAPH_CORE_ROWS
        or sum(int((~trace.graph_core_rows).sum()) for trace in traces)
        != _EXPECTED_TAIL_ROWS
    ):
        raise ValueError("reproduced A16 fit support differs from parent")

    sequences = tuple(trace.sequence for trace in traces)
    global_basis = fit_complete_h4_projection_basis(
        sequences,
        max_rank=_RANK,
        fit_weighting="unweighted",
    )
    parent_fitted = _mapping(parent.get("fitted_bases"), label="parent bases")
    parent_global = _mapping(
        parent_fitted.get("unweighted"),
        label="parent unweighted U320",
    )
    if frozen._canonical_json_bytes(global_basis.metadata()) != (
        frozen._canonical_json_bytes(parent_global)
    ):
        raise ValueError("reproduced unweighted U320 differs from parent")
    if (
        global_basis.artifact_sha256
        != _EXPECTED_GLOBAL_BASIS_ARTIFACT_SHA256
        or global_basis.basis_rows.matrix_sha256
        != _EXPECTED_GLOBAL_BASIS_MATRIX_SHA256
    ):
        raise ValueError("reproduced unweighted U320 receipt differs")
    fit_manifest = ladder._fit_manifest_receipt(
        sequences,
        {"unweighted": global_basis},
        weightings=("unweighted",),
    )
    if frozen._canonical_json_bytes(fit_manifest) != frozen._canonical_json_bytes(
        parent.get("fit_manifest")
    ):
        raise ValueError("reproduced A16 fit manifest differs from parent")

    # The legacy pair/mask receipts cannot be regenerated without the
    # discarded three-pass shadow.  They enter only through this exact trace
    # gate, after the live tensor/objective/sequence/mask proof above.
    tail_traces = _authenticated_tail_traces(traces)
    tail_fit = fit_complete_h4_tail_informed_projection(
        tail_traces,
        global_basis,
        anchor_rank=_ANCHOR_RANK,
        max_rank=_RANK,
    )
    tail_fit.validate_integrity()
    parent_tail = _mapping(
        parent.get("tail_informed_fit"),
        label="parent tail-informed fit",
    )
    if frozen._canonical_json_bytes(tail_fit.metadata()) != (
        frozen._canonical_json_bytes(parent_tail)
    ):
        raise ValueError("reproduced tail-informed D320 fit differs from parent")
    if (
        tail_fit.artifact_sha256 != TAIL_INFORMED_FIT_ARTIFACT_SHA256
        or tail_fit.tail_rank != _EXPECTED_TAIL_ROWS
    ):
        raise ValueError("reproduced tail-informed fit receipt differs")
    d320 = tail_fit.basis_tensor(_RANK).contiguous()
    sidecar, sidecar_receipt, recovered_by_publisher = (
        _publish_or_recover_exact_sidecar(
            d320,
            basis_destination=basis_destination,
            existing_sidecar=existing_sidecar,
        )
    )
    if recovered_by_publisher != recovered_existing_sidecar:
        raise RuntimeError("rank320 sidecar recovery state differs")
    if (
        sidecar.basis_matrix_sha256 != EXPECTED_BASIS_MATRIX_SHA256
        or sidecar.runtime_tensor_sha256 != EXPECTED_RUNTIME_TENSOR_SHA256
        or sidecar.projection_basis_artifact_sha256
        != EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256
    ):
        raise ValueError("reproduced closed D320 sidecar differs")

    selected_lineage_rows = parent.get("fit_to_prefix_lineage")
    if not isinstance(selected_lineage_rows, list):
        raise ValueError("parent selected-arm lineage differs")
    selected_rows = [
        _mapping(row, label="parent fit-to-prefix lineage")
        for row in selected_lineage_rows
        if _mapping(row, label="parent fit-to-prefix lineage").get("arm_id")
        == _SELECTED_ARM
    ]
    if len(selected_rows) != 1:
        raise ValueError("parent selected rank320 lineage is not unique")
    selected_lineage = selected_rows[0]
    if (
        selected_lineage.get("fit_to_prefix_lineage_sha256")
        != FIT_TO_PREFIX_LINEAGE_SHA256
        or selected_lineage.get("execution_basis_sha256")
        != EXPECTED_RUNTIME_TENSOR_SHA256
        or selected_lineage.get("execution_basis_artifact_sha256")
        != EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256
    ):
        raise ValueError("parent selected rank320 execution lineage differs")

    fit_sequence_bytes = sum(
        trace.sequence.row_count * trace.sequence.width * 8 * 2
        for trace in traces
    )
    proof_rows = tuple(
        trace.proof
        for trace in sorted(
            traces,
            key=lambda row: (row.sequence.family_id, row.sequence.example_id),
        )
    )
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": _ROLE,
        "lineage": {
            "parent_factorial_file_sha256": PARENT_FILE_SHA256,
            "parent_factorial_report_sha256": PARENT_REPORT_SHA256,
            "selected_arm": _SELECTED_ARM,
            "global_fit_basis_artifact_sha256": (
                global_basis.artifact_sha256
            ),
            "tail_informed_fit_artifact_sha256": tail_fit.artifact_sha256,
            "fit_to_prefix_lineage_sha256": FIT_TO_PREFIX_LINEAGE_SHA256,
            "projection_basis_artifact_sha256": (
                sidecar.projection_basis_artifact_sha256
            ),
        },
        "protocol": {
            "panel": "reused_calibration_a_fit16",
            "collection_per_prompt": (
                "native_boundary_forward_then_clamped_y3_exact_x4_h4_vjp"
            ),
            "model_forwards_per_prompt": 2,
            "backwards_per_prompt": 1,
            "objective": "prompt_mean_next_token_nll",
            "residual_row_arithmetic": (
                "select_then_native_to_float64_minus_incomplete_to_float64"
            ),
            "fit_ordering": "family_id_then_example_id",
            "global_fit": "family_example_macro_unweighted_u320",
            "tail_fit": (
                "u192_then_full_tail_residual_svd_span_then_two_pass_mgs_u320"
            ),
            "legacy_pair_and_graph_mask_lineage_policy": (
                "frozen_parent_receipts_reused_only_after_exact_live_tensor_"
                "objective_sequence_and_mask_proof"
            ),
            "transfer_evaluation": "not_run",
        },
        "carrier_promotion": carrier_preflight,
        "fit_manifest": fit_manifest,
        "reproduction": {
            "observation_receipts": proof_rows,
            "all_16_parent_fit_sequences_reproduced_exactly": True,
            "all_16_parent_tail_traces_reproduced_exactly": True,
            "unweighted_u320_metadata_reproduced_exactly": True,
            "tail_informed_fit_metadata_reproduced_exactly": True,
            "support_rows": _EXPECTED_SUPPORT_ROWS,
            "graph_core_rows": _EXPECTED_GRAPH_CORE_ROWS,
            "causal_tail_rows": _EXPECTED_TAIL_ROWS,
            "legacy_three_pass_shadow_reexecuted": False,
            "frozen_pair_lineage_reused_after_exact_proof": True,
            "frozen_graph_mask_lineage_reused_after_exact_proof": True,
        },
        "materialized_basis": {
            **sidecar.metadata(),
            "sidecar_file_sha256": sidecar_receipt["file_sha256"],
            "sidecar_file_bytes": sidecar_receipt["file_bytes"],
        },
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            "model_forward_count": 32,
            "backward_count": 16,
            "evaluation_model_forward_count": 0,
            "projection_arm_model_forward_count": 0,
            "transfer_model_forward_count": 0,
            "simultaneously_live_full_vocabulary_logit_tensor_peak": 1,
            "full_vocabulary_logit_roles_at_peak": (
                "native_or_partial_exact_x4_never_both",
            ),
            "retained_fit_sequence_count": 16,
            "retained_fit_sequence_residual_and_gradient_bytes": (
                fit_sequence_bytes
            ),
            "d320_float64_coefficient_count": _RANK * _WIDTH,
            "d320_float64_matrix_bytes": _RANK * _WIDTH * 8,
            "float64_width_square_eigendecomposition_count": 1,
            "tail_residual_float64_svd_count": 1,
            "inference_projection_macs_executed": 0,
            "basis_fit_linear_algebra_is_offline_only": True,
            "latency_or_speed_claim": False,
            "whole_model_parameter_reduction_claim": False,
        },
        "publication": {
            "report_is_sidecar_commit_marker": True,
            "consumer_must_require_report_and_sidecar": True,
            "sidecar_atomic_no_overwrite_publication": True,
            "report_atomic_no_overwrite_publication": True,
            "recovered_existing_authenticated_orphan_sidecar": (
                recovered_existing_sidecar
            ),
            "mismatched_existing_sidecar_accepted": False,
            "existing_report_overwrite_allowed": False,
        },
        "scientific_status": {
            "exact_rank320_basis_materialized": True,
            "same_a_fit_only": True,
            "transfer_evaluation_run": False,
            "frozen_basis_one_pass_carrier_transfer_ready": True,
            "candidate_serving_authorized": False,
            "generator_validated": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "next_rung": "fixed_d320_one_pass_carrier_transfer_oracle",
        },
        "artifact": {
            "file": str(destination),
            "basis_sidecar": sidecar_receipt,
            "committable": False,
        },
        "safety": dict(_SAFETY),
    }
    # Publication ordering is intentional.  The exact sidecar is atomic; the
    # report is the atomic commit marker.  If report publication is interrupted,
    # a rerun reauthenticates that exact orphan sidecar and can publish only the
    # missing marker.  A consumer rejects sidecar-without-marker state.
    return ladder._publish(
        report,
        output=destination,
        report_domain=_REPORT_DOMAIN,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the successful A16 complete-H4 D320 in 32F/16B"
        )
    )
    parser.add_argument(
        "--fit-source-artifact",
        default=frozen.DEFAULT_INTERIOR_ARTIFACT,
    )
    parser.add_argument("--parent-artifact", default=frozen.DEFAULT_PARENT_ARTIFACT)
    parser.add_argument(
        "--candidate-artifact",
        default=frozen.DEFAULT_CANDIDATE_ARTIFACT,
    )
    parser.add_argument(
        "--graph-candidate",
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument("--basis-package", default=frozen.DEFAULT_BASIS_PACKAGE)
    parser.add_argument(
        "--base-artifact",
        default=frozen.DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        default=frozen.DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--panel", default=frozen.DEFAULT_PANEL)
    parser.add_argument(
        "--parent-factorial",
        default=DEFAULT_PARENT_FACTORIAL,
    )
    parser.add_argument("--basis-output", default=DEFAULT_BASIS_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-length", type=int, default=frozen.DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_rank320_basis_materialization(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        graph_candidate_path=arguments.graph_candidate,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        parent_factorial_path=arguments.parent_factorial,
        basis_output=arguments.basis_output,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        max_length=arguments.max_length,
    )
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "materialized_basis": report["materialized_basis"],
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
