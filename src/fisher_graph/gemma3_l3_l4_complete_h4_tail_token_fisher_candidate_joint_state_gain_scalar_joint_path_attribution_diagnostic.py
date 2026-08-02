"""Live fixed-GL4 attribution of the realized V6-scalar to V7-joint path.

This V9 diagnostic authenticates the exact pinned V8 result and reproduces
its V7 analytic lineage before doing any new model work.  It then executes,
for each of the sixteen outer-held Calibration-A prompts, one fresh native
teacher, exact held-unit and scalar-endpoint teacher-KL VJPs, the joint
endpoint, and four strictly interior GL4 teacher-KL VJPs along the *realized
cast-once H4 path*

``H(alpha) = H_scalar_actual + alpha * (H_joint_actual - H_scalar_actual)``.

The finite ``KL_joint - KL_scalar`` displacement is compared with the GL4
integral of the H4 gradient contracted against that same realized
displacement.  The held-unit and scalar endpoint VJPs are retained only as
separately labelled first-order context.  Fixed, preregistered GL4 closure
thresholds determine whether quadrature closure is established.  There is no
fitting, selection, step grid, fallback, serving route, or model mutation in
the V9 attribution stage.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from . import complete_h4_tail_candidate_state_gain_field as state_field
from . import gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic as path_diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic as v5diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic as v3diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic as v4diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic as v7diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_finite_diagnostic as v8diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic as v6diag
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from .complete_h4_tail_candidate_gain_refit_v4 import CANDIDATE_GAIN_RANK
from .complete_h4_tail_candidate_joint_state_gain_finite import (
    CandidateConditionedK64ThreeArmGainSupport,
    build_candidate_conditioned_k64_three_arm_gain_support,
    candidate_conditioned_k64_gain_correction_rows,
)
from .complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathAccumulator,
    CandidateJointStatePathAttribution,
    CandidateJointStatePathEvidence,
    candidate_joint_state_finite_kl_delta,
    candidate_joint_state_held_unit_endpoint_tangent_contraction,
    candidate_joint_state_path_integrated_contraction,
    candidate_joint_state_scalar_endpoint_tangent_contraction,
    summarize_candidate_joint_state_path_attribution,
)
from .complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    fit_complete_h4_tail_held_family,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import _load_committed_basis
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _require_sha256,
    _runtime_tensor_sha256,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_V8_REPORT",
    "V8_REPORT_FILE_SHA256",
    "V8_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_scalar_joint_path_attribution_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v8diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v8diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v8diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v8diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v8diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v8diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v8diag.DEFAULT_V6_REPORT
DEFAULT_V7_REPORT = v8diag.DEFAULT_V7_REPORT
DEFAULT_V8_REPORT = v8diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-gain-scalar-joint-path-gl4-"
    "attribution-lofo-a-fit16-dev-v9.json"
)

V8_REPORT_FILE_SHA256 = (
    "9a1a5e925b86592831bbc26164fc4df6a6dd778d2f4b7563733277c0dba231ac"
)
V8_REPORT_SHA256 = (
    "622fa8ce212ea625837754a69e8e24f609b554ad35a8567bd8653f19dd44f305"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_gain_scalar_joint_path_gl4_attribution_lofo.v9"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-scalar-joint-path-attribution:v9\0"
_V8_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-v8-live-binding:v9\0"
_ENDPOINT_PAIR_DOMAIN = b"fisher-graph:complete-h4-k64-scalar-joint-pair:v9\0"
_PATH_PROVIDER_DOMAIN = b"fisher-graph:complete-h4-k64-scalar-joint-path-provider:v9\0"
_PATH_NODE_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-scalar-joint-node:v9\0"
_PATH_OBSERVATION_SET_DOMAIN = b"fisher-graph:complete-h4-k64-scalar-joint-set:v9\0"

_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS = 16
_EXPECTED_PATH_NODES = 4
_EXPECTED_PATH_VJP_FORWARDS = 64
_EXPECTED_LINEAGE_FORWARDS = 112
_EXPECTED_LINEAGE_BACKWARDS = 494
_EXPECTED_FRESH_FORWARDS = 128
_EXPECTED_FRESH_BACKWARDS = 654
_EXPECTED_TOTAL_FORWARDS = 240
_EXPECTED_TOTAL_BACKWARDS = 1148
_OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM = 0.05
_FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM = 0.10
_CLOSURE_COSINE_MINIMUM = 0.99


def _canonical(value: object) -> object:
    return v3diag._canonical(value)


def _load_v8_report(path: Path | str) -> dict[str, object]:
    """Load only the exact inspected V8 finite attribution-failure artifact."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V8_REPORT_FILE_SHA256,
        expected_report_sha256=V8_REPORT_SHA256,
        label="candidate joint state-gain finite v8 attribution anchor",
    )
    resources = report.get("resources")
    if (
        report.get("schema") != v8diag._SCHEMA
        or report.get("classification")
        != "analytic_to_finite_attribution_failure_same_a"
        or report.get("passed") is not False
        or not isinstance(resources, Mapping)
        or resources.get("total_model_forward_count") != 176
        or resources.get("total_backward_call_count") != 494
    ):
        raise RuntimeError("candidate joint state-gain finite V8 differs")
    _index_v8_endpoint_observations(report)
    return report


def _index_v8_endpoint_observations(
    report: Mapping[str, object],
) -> dict[tuple[str, str], dict[str, object]]:
    """Authenticate and index the exact V8 16-by-3 observation grid."""

    raw = report.get("finite_observation_receipts")
    if not isinstance(raw, list) or len(raw) != 48:
        raise ValueError("pinned V8 finite observation grid differs")
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    families_by_example: dict[str, str] = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("pinned V8 finite observation row differs")
        row = dict(value)
        receipt = row.pop("observation_sha256", None)
        expected = token_v1._domain_sha256(row, domain=v8diag._OBSERVATION_DOMAIN)
        if receipt != expected:
            raise RuntimeError("pinned V8 finite observation receipt drifted")
        row["observation_sha256"] = receipt
        example_id = token_v1._identifier(
            row.get("example_id"), label="pinned V8 example_id"
        )
        family_id = token_v1._identifier(
            row.get("family_id"), label="pinned V8 family_id"
        )
        arm = row.get("arm")
        if arm not in v8diag.THREE_ARM_FINITE_NAMES:
            raise ValueError("pinned V8 finite arm differs")
        key = (example_id, str(arm))
        if key in indexed or (
            example_id in families_by_example
            and families_by_example[example_id] != family_id
        ):
            raise ValueError("pinned V8 finite observation identity repeats")
        indexed[key] = row
        families_by_example[example_id] = family_id
    examples = tuple(sorted(families_by_example))
    families = tuple(sorted(set(families_by_example.values())))
    if (
        len(examples) != _EXPECTED_PROMPTS
        or len(families) != _EXPECTED_FAMILIES
        or any(
            sum(families_by_example[example] == family for example in examples) != 2
            for family in families
        )
        or any(
            (example, arm) not in indexed
            for example in examples
            for arm in v8diag.THREE_ARM_FINITE_NAMES
        )
    ):
        raise RuntimeError("pinned V8 finite observation universe differs")
    ordered = tuple(
        str(indexed[(example, arm)]["observation_sha256"])
        for example in examples
        for arm in v8diag.THREE_ARM_FINITE_NAMES
    )
    expected_set = token_v1._domain_sha256(
        ordered, domain=v8diag._OBSERVATION_SET_DOMAIN
    )
    if report.get("finite_observation_set_sha256") != expected_set:
        raise RuntimeError("pinned V8 finite observation set drifted")
    return indexed


def _authenticate_v8_live_lineage(
    *,
    v8_report: Mapping[str, object],
    analytic: v8diag._V8AnalyticPhaseResults,
) -> dict[str, object]:
    """Require the newly reproduced V7/V8 analytic lineage to be canonical."""

    control = v8_report.get("v7_control_binding")
    if not isinstance(control, Mapping):
        raise ValueError("pinned V8 V7 control binding differs")
    comparisons = (
        (
            analytic.v7_binding,
            control.get("live_evidence_reproduction"),
            "V7 live evidence",
        ),
        (analytic.codec_binding, v8_report.get("full_seven_codec_binding"), "codec"),
        (
            analytic.unit_binding,
            v8_report.get("pinned_v4_unit_reference_binding"),
            "unit",
        ),
        (
            analytic.pinned_plus_binding,
            v8_report.get("pinned_v5_plus_availability_binding"),
            "plus",
        ),
    )
    for live, pinned, label in comparisons:
        if not isinstance(pinned, Mapping) or _canonical(live) != _canonical(pinned):
            raise RuntimeError(f"live V9 lineage did not reproduce V8 {label}")
    payload: dict[str, object] = {
        "v8_report_file_sha256": V8_REPORT_FILE_SHA256,
        "v8_report_sha256": V8_REPORT_SHA256,
        "v8_schema": v8_report.get("schema"),
        "v8_classification": v8_report.get("classification"),
        "v8_passed": v8_report.get("passed"),
        "v7_live_binding_canonically_equal": True,
        "full_seven_codec_binding_canonically_equal": True,
        "pinned_unit_binding_canonically_equal": True,
        "pinned_plus_binding_canonically_equal": True,
        "authenticated_before_any_v9_fresh_forward": True,
        "raw_tensors_serialized": False,
    }
    payload["artifact_sha256"] = token_v1._domain_sha256(
        payload, domain=_V8_BINDING_DOMAIN
    )
    return payload


def _realized_support_rows(
    candidate_h4: Tensor, support_indices: Tensor
) -> Tensor:
    """Return rank-2 runtime-dtype H4 rows for one single-prompt execution."""

    if (
        not isinstance(candidate_h4, Tensor)
        or candidate_h4.ndim != 3
        or candidate_h4.shape[0] != 1
        or candidate_h4.shape[-1] != token_v1._WIDTH
        or not candidate_h4.is_floating_point()
        or not isinstance(support_indices, Tensor)
        or support_indices.ndim != 1
        or support_indices.dtype != torch.int64
        or support_indices.numel() <= 0
    ):
        raise ValueError("realized support H4 geometry differs")
    rows = (
        candidate_h4.detach()
        .to(device="cpu")[0]
        .index_select(0, support_indices.detach().to(device="cpu"))
        .clone()
        .contiguous()
    )
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("realized support H4 rows must be finite")
    return rows


def _cast_once_scalar_joint_path_rows(
    *, scalar_h4_rows: Tensor, joint_h4_rows: Tensor, path_fraction: float
) -> Tensor:
    """Interpolate realized endpoints in f64, then cast once to endpoint dtype."""

    if (
        not isinstance(scalar_h4_rows, Tensor)
        or scalar_h4_rows.ndim != 2
        or not scalar_h4_rows.is_floating_point()
        or not isinstance(joint_h4_rows, Tensor)
        or joint_h4_rows.shape != scalar_h4_rows.shape
        or joint_h4_rows.dtype != scalar_h4_rows.dtype
        or not joint_h4_rows.is_floating_point()
        or not bool(torch.isfinite(scalar_h4_rows).all())
        or not bool(torch.isfinite(joint_h4_rows).all())
        or isinstance(path_fraction, bool)
        or not isinstance(path_fraction, (int, float))
        or not math.isfinite(float(path_fraction))
        or not 0.0 <= float(path_fraction) <= 1.0
    ):
        raise ValueError("scalar-to-joint cast-once path geometry differs")
    scalar = scalar_h4_rows.detach().to(device="cpu")
    joint = joint_h4_rows.detach().to(device="cpu")
    alpha = float(path_fraction)
    return (
        scalar.to(torch.float64)
        .add(alpha * (joint.to(torch.float64) - scalar.to(torch.float64)))
        .to(dtype=scalar.dtype)
        .clone()
        .contiguous()
    )


class _AuthenticatedV9ScalarJointPathProvider(Gemma3L3L4CorrectionProvider):
    """Single-use provider for one exact GL4 node between realized endpoints."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "example_id",
        "family_id",
        "endpoint_pair_binding_sha256",
        "model_inputs_sha256",
        "bridge_binding_sha256",
        "prefix_artifact_sha256",
        "node_index",
        "path_fraction_hex",
        "quadrature_weight_hex",
        "base_h4_sha256",
        "support_mask_sha256",
        "support_indices_sha256",
        "scalar_endpoint_h4_rows_sha256",
        "joint_endpoint_h4_rows_sha256",
        "path_node_h4_rows_sha256",
        "correction_sha256",
        "_support",
        "_support_indices",
        "_scalar_h4_rows",
        "_joint_h4_rows",
        "_path_h4_rows",
        "_correction",
        "_used",
    )

    def __init__(
        self,
        *,
        example_id: str,
        family_id: str,
        endpoint_pair_binding_sha256: str,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        node_index: int,
        path_fraction: float,
        quadrature_weight: float,
        base_h4: Tensor,
        support_mask: Tensor,
        support_indices: Tensor,
        scalar_endpoint_h4_rows: Tensor,
        joint_endpoint_h4_rows: Tensor,
    ) -> None:
        if type(node_index) is not int or not 0 <= node_index < 4:
            raise ValueError("V9 path provider node index differs")
        alpha = float(path_fraction)
        weight = float(quadrature_weight)
        if (
            alpha.hex() != GL4_UNIT_INTERVAL_NODES[node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
        ):
            raise ValueError("V9 path provider does not use the exact GL4 rule")
        if (
            not isinstance(base_h4, Tensor)
            or base_h4.ndim != 3
            or base_h4.shape[0] != 1
            or base_h4.shape[-1] != token_v1._WIDTH
            or not base_h4.is_floating_point()
            or not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not isinstance(support_indices, Tensor)
            or support_indices.ndim != 1
            or support_indices.dtype != torch.int64
        ):
            raise ValueError("V9 path provider tensor geometry differs")
        live_base_h4_sha256 = _runtime_tensor_sha256(base_h4.detach())
        base = base_h4.detach().to(device="cpu").clone().contiguous()
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        indices = support_indices.detach().to(device="cpu").clone().contiguous()
        canonical_indices = torch.nonzero(support[0], as_tuple=False).flatten()
        scalar = scalar_endpoint_h4_rows.detach().to(device="cpu").clone().contiguous()
        joint = joint_endpoint_h4_rows.detach().to(device="cpu").clone().contiguous()
        if (
            not torch.equal(indices, canonical_indices)
            or scalar.shape != (int(indices.numel()), token_v1._WIDTH)
            or joint.shape != scalar.shape
            or scalar.dtype != base.dtype
            or joint.dtype != base.dtype
            or not bool(torch.isfinite(scalar).all())
            or not bool(torch.isfinite(joint).all())
        ):
            raise ValueError("V9 path provider endpoint geometry differs")
        path_rows = _cast_once_scalar_joint_path_rows(
            scalar_h4_rows=scalar,
            joint_h4_rows=joint,
            path_fraction=alpha,
        )
        base_rows = base[0].index_select(0, indices)
        correction_rows = (
            path_rows.to(torch.float64) - base_rows.to(torch.float64)
        ).contiguous()
        correction = torch.zeros(base.shape, dtype=torch.float64)
        correction[0].index_copy_(0, indices, correction_rows)
        replay = base.clone()
        replay[support] = (
            base[support].to(torch.float64) + correction[support]
        ).to(base.dtype)
        if (
            not token_v1._bitwise_equal(
                replay[0].index_select(0, indices), path_rows
            )
            or bool((correction[~support] != 0.0).any())
        ):
            raise RuntimeError("V9 path provider cannot replay cast-once node")
        self.site = token_v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.example_id = token_v1._identifier(example_id, label="V9 example_id")
        self.family_id = token_v1._identifier(family_id, label="V9 family_id")
        self.endpoint_pair_binding_sha256 = _require_sha256(
            endpoint_pair_binding_sha256, label="V9 endpoint pair binding"
        )
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="V9 model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="V9 bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="V9 prefix"
        )
        self.node_index = node_index
        self.path_fraction_hex = alpha.hex()
        self.quadrature_weight_hex = weight.hex()
        # The execution binding is device-sensitive.  Keep the live runtime
        # hash; the CPU clone above exists only for deterministic row algebra.
        self.base_h4_sha256 = live_base_h4_sha256
        self.support_mask_sha256 = _runtime_tensor_sha256(support)
        self.support_indices_sha256 = _runtime_tensor_sha256(indices)
        self.scalar_endpoint_h4_rows_sha256 = _runtime_tensor_sha256(scalar)
        self.joint_endpoint_h4_rows_sha256 = _runtime_tensor_sha256(joint)
        self.path_node_h4_rows_sha256 = _runtime_tensor_sha256(path_rows)
        self.correction_sha256 = _runtime_tensor_sha256(correction)
        self._support = support
        self._support_indices = indices
        self._scalar_h4_rows = scalar
        self._joint_h4_rows = joint
        self._path_h4_rows = path_rows
        self._correction = correction
        self._used = False
        self.artifact_sha256 = token_v1._domain_sha256(
            self._payload(), domain=_PATH_PROVIDER_DOMAIN
        )
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.complete_h4_k64_scalar_joint_path_provider.v9",
            "site": self.site,
            "write_scope": self.write_scope,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "endpoint_pair_binding_sha256": self.endpoint_pair_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "quadrature_rule": "gauss_legendre_order_4_on_unit_interval",
            "node_index": self.node_index,
            "path_fraction_hex": self.path_fraction_hex,
            "quadrature_weight_hex": self.quadrature_weight_hex,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "support_indices_sha256": self.support_indices_sha256,
            "scalar_endpoint_h4_rows_sha256": self.scalar_endpoint_h4_rows_sha256,
            "joint_endpoint_h4_rows_sha256": self.joint_endpoint_h4_rows_sha256,
            "path_node_h4_rows_sha256": self.path_node_h4_rows_sha256,
            "correction_sha256": self.correction_sha256,
            "trajectory": (
                "realized_scalar_H4_plus_alpha_times_realized_joint_minus_scalar_"
                "then_one_endpoint_dtype_cast"
            ),
            "single_use": True,
            "same_A_attribution_only": True,
            "serving_authorized": False,
        }

    @property
    def used(self) -> bool:
        return self._used

    @property
    def path_h4_rows(self) -> Tensor:
        return self._path_h4_rows.clone()

    def validate_integrity(self) -> None:
        replay = _cast_once_scalar_joint_path_rows(
            scalar_h4_rows=self._scalar_h4_rows,
            joint_h4_rows=self._joint_h4_rows,
            path_fraction=float.fromhex(self.path_fraction_hex),
        )
        if (
            _runtime_tensor_sha256(self._support) != self.support_mask_sha256
            or _runtime_tensor_sha256(self._support_indices)
            != self.support_indices_sha256
            or _runtime_tensor_sha256(self._scalar_h4_rows)
            != self.scalar_endpoint_h4_rows_sha256
            or _runtime_tensor_sha256(self._joint_h4_rows)
            != self.joint_endpoint_h4_rows_sha256
            or _runtime_tensor_sha256(self._path_h4_rows)
            != self.path_node_h4_rows_sha256
            or not token_v1._bitwise_equal(replay, self._path_h4_rows)
            or _runtime_tensor_sha256(self._correction) != self.correction_sha256
            or bool((self._correction[~self._support] != 0.0).any())
            or token_v1._domain_sha256(self._payload(), domain=_PATH_PROVIDER_DOMAIN)
            != self.artifact_sha256
        ):
            raise RuntimeError("V9 scalar-joint path provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("V9 scalar-joint path provider cannot be reused")
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
            raise RuntimeError("V9 scalar-joint path provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _validate_v9_path_node_full_h4(
    *, trace: object, provider: _AuthenticatedV9ScalarJointPathProvider, execution: object
) -> None:
    """Fail if a path execution differs anywhere in full H4, including off support."""

    v3diag._validate_candidate_execution(
        trace=trace, provider=provider, execution=execution
    )


def _prepare_v4_unit_provider(
    *,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
) -> tuple[v4diag._AuthenticatedCandidateGainProviderV4, Tensor, Tensor, Tensor]:
    """Rebuild the exact pinned V4 final unit-K64 provider."""

    gains = torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
    directions, tail_rows, correction_rows, correction = v3diag._candidate_components(
        trace,
        basis=basis.detach().to(device="cpu", dtype=torch.float64).contiguous(),
        fit=fit,
        gains=gains,
    )
    provider = v4diag._AuthenticatedCandidateGainProviderV4(
        stage="final",
        candidate_variant="unit",
        fold_artifact_sha256=fit.artifact_sha256,
        ordered_directions_sha256=_runtime_tensor_sha256(directions),
        gains=gains,
        mean_refit_artifact_sha256=None,
        selection_artifact_sha256=None,
        step=None,
        model_inputs_sha256=trace.model_inputs_sha256,
        bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
        prefix_artifact_sha256=trace.prefix.artifact_sha256,
        base_h4=trace.base_h4,
        support_mask=trace.prefix.complete_h4_causal_support_mask(),
        correction=correction,
    )
    if (
        provider.candidate_variant != "unit"
        or provider.stage != "final"
        or provider.gains_sha256 != _runtime_tensor_sha256(gains)
        or provider.correction_sha256 != _runtime_tensor_sha256(correction)
    ):
        raise RuntimeError("V9 unit provider did not bind exact V4 algebra")
    return provider, directions, tail_rows, correction_rows


def _expected_v4_unit_execution(
    *,
    trace: object,
    fit: CompleteH4TailHeldFamilyFit,
    provider: v4diag._AuthenticatedCandidateGainProviderV4,
    execution: object,
    correction_rows: Tensor,
    teacher_logits: Tensor,
    endpoint_grid: Tensor,
    token_kl: Tensor,
    unit_observation: Mapping[str, object],
) -> tuple[Tensor, dict[str, object]]:
    """Require the fresh held-unit VJP endpoint to replay pinned V4."""

    actual_rows = (
        execution.candidate_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, trace.support_indices)
        - trace.base_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, trace.support_indices)
    ).contiguous()
    stable = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "arm": "unit_k64",
        "candidate_variant": "unit",
        "rank": CANDIDATE_GAIN_RANK,
        "fold_artifact_sha256": fit.artifact_sha256,
        "mean_refit_artifact_sha256": None,
        "selection_artifact_sha256": None,
        "selected_mean_alpha_hex": None,
        "selected_reverse_beta_hex": None,
        "ordered_directions_sha256": provider.ordered_directions_sha256,
        "gains_sha256": provider.gains_sha256,
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_artifact_sha256": execution.artifact_sha256,
        "teacher_logits_sha256": _runtime_tensor_sha256(teacher_logits),
        "endpoint_supervised_grid_sha256": _runtime_tensor_sha256(endpoint_grid),
        "endpoint_supervised_token_count": int(endpoint_grid.shape[0]),
        "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
        "executed_correction_rows_sha256": _runtime_tensor_sha256(actual_rows),
    }
    changed = tuple(
        key for key, value in stable.items() if unit_observation.get(key) != value
    )
    expected_rows = v8diag._executed_cast_once_delta_rows(
        base_h4=trace.base_h4,
        correction=provider._correction,
        support_mask=provider._support,
        support_indices=trace.support_indices,
    )
    if changed or not torch.equal(actual_rows, expected_rows):
        raise RuntimeError(
            f"V9 held-unit endpoint did not replay pinned V4: {changed}"
        )
    realized = _realized_support_rows(execution.candidate_h4, trace.support_indices)
    receipt: dict[str, object] = {
        **stable,
        "pinned_v4_observation_sha256": unit_observation["observation_sha256"],
        "analytic_correction_rows_sha256": _runtime_tensor_sha256(correction_rows),
        "realized_endpoint_h4_rows_sha256": _runtime_tensor_sha256(realized),
        "realized_endpoint_h4_dtype": str(realized.dtype),
        "realized_endpoint_h4_shape": tuple(int(size) for size in realized.shape),
        "pinned_v4_unit_boundary_fields_replayed_exactly": True,
        "raw_tensors_serialized": False,
    }
    return realized, receipt


def _held_unit_fullfit_convention_tokens(
    *,
    trace: object,
    refit: object,
    codec: object,
    record: object,
    directions: Tensor,
    tail_rows: Tensor,
    unit_gradients: Tensor,
    unit_token_kl: Tensor,
) -> tuple[Tensor, dict[str, object]]:
    """Evaluate V7's derivative convention on the fresh held unit gradient."""

    base_rows = _realized_support_rows(
        trace.base_h4, trace.support_indices
    ).to(torch.float64)
    features = state_field.encode_candidate_conditioned_k64_state_features(
        codec,
        base_h4_support_rows=base_rows,
        ordered_directions=directions,
    )
    mean_gain_delta = v6diag._mean_delta(refit)
    row_scores = state_field.contract_candidate_conditioned_k64_row_direction_scores(
        tail_rows=tail_rows,
        ordered_directions=directions,
        token_h4_gradients=unit_gradients,
        mean_gain_delta=mean_gain_delta,
    )
    static_tangent = row_scores.sum(dim=1).contiguous()
    state_design = (row_scores @ features).contiguous()
    joint_tokens = (
        static_tangent * float(record.full_joint_fit.applied_intercept)
        + state_design
        @ record.full_joint_fit.applied_parameter[1:].detach().to(
            device="cpu", dtype=torch.float64
        )
    ).contiguous()
    scalar_tokens = (
        static_tangent
        * float(record.full_scalar_control_fit.applied_coefficient)
    ).contiguous()
    margin = (joint_tokens - scalar_tokens).contiguous()
    if (
        margin.shape != unit_token_kl.shape
        or not bool(torch.isfinite(margin).all())
    ):
        raise RuntimeError("V9 held-unit full-fit convention geometry differs")
    receipt = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "full_joint_fit_artifact_sha256": record.full_joint_fit.artifact_sha256,
        "full_scalar_control_fit_artifact_sha256": (
            record.full_scalar_control_fit.artifact_sha256
        ),
        "codec_artifact_sha256": codec.artifact_sha256,
        "refit_artifact_sha256": refit.artifact_sha256,
        "unit_gradient_runtime_sha256": _runtime_tensor_sha256(unit_gradients),
        "unit_token_teacher_kl_runtime_sha256": _runtime_tensor_sha256(
            unit_token_kl
        ),
        "standardized_state_features_runtime_sha256": _runtime_tensor_sha256(
            features
        ),
        "token_row_direction_scores_runtime_sha256": _runtime_tensor_sha256(
            row_scores
        ),
        "held_fullfit_joint_minus_scalar_token_derivative_runtime_sha256": (
            _runtime_tensor_sha256(margin)
        ),
        "held_fullfit_joint_minus_scalar_mean_derivative": float(margin.mean()),
        "convention": (
            "V7_full_seven_joint_and_scalar_coefficients_on_fresh_held_unit_gradient"
        ),
        "uses_actual_scalar_joint_displacement": False,
        "raw_tensors_serialized": False,
    }
    return margin, receipt


def _prepare_v8_endpoint_provider(
    *,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    support: CandidateConditionedK64ThreeArmGainSupport,
    arm: str,
) -> tuple[v8diag._AuthenticatedV8GainProvider, Tensor]:
    """Rebuild the exact V8 scalar or joint provider without executing it."""

    if arm not in {"v6_exact_scalar", "v7_joint"}:
        raise ValueError("V9 endpoint arm must be scalar or joint")
    directions = v6diag._ordered_k64(fit)
    support.validate_integrity()
    support_metadata = support.metadata()
    gain_hash_field = {
        "v6_exact_scalar": "exact_scalar_gains_sha256",
        "v7_joint": "joint_row_gains_sha256",
    }[arm]
    gains = support.gains_tensor(arm)
    support_mask = (
        trace.prefix.complete_h4_causal_support_mask()
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    base_support_rows = _realized_support_rows(
        trace.base_h4, trace.support_indices
    ).to(torch.float64)
    directions_sha256 = _runtime_tensor_sha256(directions)
    gains_sha256 = _runtime_tensor_sha256(gains)
    if (
        support.phase != "held"
        or support.example_id != trace.example_id
        or support.family_id != trace.family_id
        or support.held_family_id != trace.family_id
        or support.parent_fold_artifact_sha256 != fit.artifact_sha256
        or support.ordered_directions_codec_sha256
        != state_field._tensor_sha256(directions)
        or support.ordered_directions_refit_sha256 != directions_sha256
        or support.base_h4_support_rows_sha256
        != _runtime_tensor_sha256(base_support_rows)
        or support.support_row_count != int(support_mask.sum())
        or support_metadata.get(gain_hash_field) != gains_sha256
    ):
        raise RuntimeError("V9 endpoint gain-support binding differs")
    residual = trace.endpoint.residual_rows.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    frozen_basis = basis.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    supported_rows = ((residual @ frozen_basis.T) @ frozen_basis).contiguous()
    tail_rows = v3diag.project_complete_h4_tail_rows(
        residual, frozen_basis
    ).contiguous()
    correction_rows = candidate_conditioned_k64_gain_correction_rows(
        supported_rows=supported_rows,
        tail_rows=tail_rows,
        ordered_directions=directions,
        gains=gains,
    )
    replay_directions, replay_tail, replay_rows, _ = v3diag._candidate_components(
        trace, basis=frozen_basis, fit=fit, gains=gains
    )
    if (
        not torch.equal(replay_directions, directions)
        or not torch.equal(replay_tail, tail_rows)
        or not torch.equal(replay_rows, correction_rows)
    ):
        raise RuntimeError("V9 endpoint row algebra does not replay V8")
    correction = torch.zeros(trace.base_h4.shape, dtype=torch.float64)
    correction[0].index_copy_(0, trace.support_indices, correction_rows)
    provider = v8diag._AuthenticatedV8GainProvider(
        arm=arm,
        fold_artifact_sha256=fit.artifact_sha256,
        gain_support_artifact_sha256=support.artifact_sha256,
        ordered_directions_sha256=directions_sha256,
        gains=gains,
        example_id=trace.example_id,
        family_id=trace.family_id,
        model_inputs_sha256=trace.model_inputs_sha256,
        bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
        prefix_artifact_sha256=trace.prefix.artifact_sha256,
        base_h4=trace.base_h4,
        support_mask=support_mask,
        correction=correction,
    )
    if (
        provider.gains_sha256 != gains_sha256
        or provider.gain_support_artifact_sha256 != support.artifact_sha256
        or provider.correction_sha256 != _runtime_tensor_sha256(correction)
        or not torch.equal(
            provider._correction[0].index_select(0, trace.support_indices),
            correction_rows,
        )
    ):
        raise RuntimeError("V9 endpoint provider did not bind exact V8 algebra")
    return provider, correction_rows


def _expected_v8_endpoint_execution(
    *,
    trace: object,
    fit: CompleteH4TailHeldFamilyFit,
    support: CandidateConditionedK64ThreeArmGainSupport,
    codec: object,
    record: object,
    refit: object,
    arm: str,
    provider: v8diag._AuthenticatedV8GainProvider,
    execution: object,
    correction_rows: Tensor,
    teacher_logits: Tensor,
    endpoint_grid: Tensor,
    token_kl: Tensor,
    v8_observation: Mapping[str, object],
) -> tuple[Tensor, dict[str, object]]:
    """Require boundary H4 and teacher-KL hashes to replay exact V8 evidence."""

    actual_rows = (
        execution.candidate_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, trace.support_indices)
        - trace.base_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, trace.support_indices)
    ).contiguous()
    expected_rows = v8diag._executed_cast_once_delta_rows(
        base_h4=trace.base_h4,
        correction=provider._correction,
        support_mask=provider._support,
        support_indices=trace.support_indices,
    )
    stable = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "held_family_id": trace.family_id,
        "arm": arm,
        "fold_artifact_sha256": fit.artifact_sha256,
        "refit_artifact_sha256": refit.artifact_sha256,
        "codec_artifact_sha256": codec.artifact_sha256,
        "scalar_fit_artifact_sha256": record.full_scalar_control_fit.artifact_sha256,
        "joint_fit_artifact_sha256": record.full_joint_fit.artifact_sha256,
        "gain_support_artifact_sha256": support.artifact_sha256,
        "ordered_directions_sha256": provider.ordered_directions_sha256,
        "gains_sha256": provider.gains_sha256,
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_artifact_sha256": execution.artifact_sha256,
        "model_inputs_sha256": provider.model_inputs_sha256,
        "bridge_binding_sha256": provider.bridge_binding_sha256,
        "prefix_artifact_sha256": provider.prefix_artifact_sha256,
        "base_h4_sha256": provider.base_h4_sha256,
        "support_mask_sha256": provider.support_mask_sha256,
        "correction_sha256": provider.correction_sha256,
        "analytic_correction_rows_sha256": _runtime_tensor_sha256(correction_rows),
        "teacher_logits_sha256": _runtime_tensor_sha256(teacher_logits),
        "endpoint_supervised_grid_sha256": _runtime_tensor_sha256(endpoint_grid),
        "endpoint_supervised_token_count": int(endpoint_grid.shape[0]),
        "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
        "executed_correction_rows_sha256": _runtime_tensor_sha256(actual_rows),
    }
    changed = tuple(
        key for key, value in stable.items() if v8_observation.get(key) != value
    )
    if changed or not torch.equal(actual_rows, expected_rows):
        raise RuntimeError(
            f"V9 {arm} endpoint did not replay V8 stable evidence: {changed}"
        )
    realized = _realized_support_rows(
        execution.candidate_h4, trace.support_indices
    )
    receipt: dict[str, object] = {
        **stable,
        "v8_observation_sha256": v8_observation["observation_sha256"],
        "realized_endpoint_h4_rows_sha256": _runtime_tensor_sha256(realized),
        "realized_endpoint_h4_dtype": str(realized.dtype),
        "realized_endpoint_h4_shape": tuple(int(size) for size in realized.shape),
        "stable_boundary_fields_replayed_exactly": True,
        "raw_tensors_serialized": False,
    }
    return realized, receipt


@dataclass(slots=True)
class _PromptPathResult:
    evidence: CandidateJointStatePathEvidence
    endpoint_receipt: Mapping[str, object]
    node_receipts: tuple[Mapping[str, object], ...]
    additive_scalars: Mapping[str, float]
    resources: Mapping[str, int]


def _execute_prompt_path(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    gain_support: CandidateConditionedK64ThreeArmGainSupport,
    codec: object,
    record: object,
    refit: object,
    unit_observation: Mapping[str, object],
    v8_observations: Mapping[tuple[str, str], Mapping[str, object]],
) -> _PromptPathResult:
    """Execute held unit/scalar VJPs, joint endpoint, and four path VJPs."""

    bridge = getattr(context, "bridge")
    adapter = getattr(context, "adapter")
    model_inputs, indices, targets, teacher_logits = v5diag._fresh_native_teacher(
        context=context, trace=trace
    )
    endpoint_indices, _endpoint_targets, endpoint_grid = v5diag._endpoint_indices(
        trace, indices, targets
    )
    teacher_sha = _runtime_tensor_sha256(teacher_logits)
    grid_sha = _runtime_tensor_sha256(endpoint_grid)
    support_mask = (
        trace.prefix.complete_h4_causal_support_mask()
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    resources = {
        "fresh_native_teacher_forward_count": 1,
        "held_unit_endpoint_vjp_forward_count": 0,
        "held_unit_endpoint_vjp_backward_call_count": 0,
        "scalar_endpoint_vjp_forward_count": 0,
        "scalar_endpoint_vjp_backward_call_count": 0,
        "joint_boundary_forward_count": 0,
        "path_teacher_kl_vjp_forward_count": 0,
        "path_teacher_kl_vjp_backward_call_count": 0,
        "path_quadrature_node_count": 0,
    }

    unit_provider, directions, tail_rows, unit_correction_rows = (
        _prepare_v4_unit_provider(trace=trace, basis=basis, fit=fit)
    )
    unit_vjp = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=endpoint_grid,
        vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
        h4_head=unit_provider,
    )
    resources["held_unit_endpoint_vjp_forward_count"] += 1
    resources["held_unit_endpoint_vjp_backward_call_count"] += (
        unit_vjp.backward_call_count
    )
    unit_vjp.validate_integrity()
    v3diag._validate_candidate_execution(
        trace=trace, provider=unit_provider, execution=unit_vjp.execution
    )
    unit_token_kl = unit_vjp.token_kl_divergences.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    unit_independent = teacher_kl._selected_token_teacher_kl(
        teacher_logits, unit_vjp.execution.logits, endpoint_indices
    )
    unit_gradients = (
        unit_vjp.h4_gradients.detach()
        .to(device="cpu", dtype=torch.float64)[:, 0]
        .index_select(1, trace.support_indices)
        .contiguous()
    )
    unit_future_maximum, unit_future_nonzero = (
        path_diag._causal_future_gradient_summary(
            gradient_rows=unit_gradients,
            endpoint_indices=endpoint_indices,
            support_indices=trace.support_indices,
            logical_positions=trace.prefix.logical_positions,
        )
    )
    if (
        unit_vjp.teacher_logits_sha256 != teacher_sha
        or not torch.equal(
            unit_vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
        )
        or not torch.allclose(unit_token_kl, unit_independent, rtol=0.0, atol=1.0e-6)
        or unit_future_maximum != 0.0
        or unit_future_nonzero != 0
    ):
        raise RuntimeError("V9 held-unit endpoint VJP authority differs")
    unit_h4_rows, unit_receipt = _expected_v4_unit_execution(
        trace=trace,
        fit=fit,
        provider=unit_provider,
        execution=unit_vjp.execution,
        correction_rows=unit_correction_rows,
        teacher_logits=teacher_logits,
        endpoint_grid=endpoint_grid,
        token_kl=unit_token_kl,
        unit_observation=unit_observation,
    )
    held_fullfit_tokens, held_fullfit_receipt = (
        _held_unit_fullfit_convention_tokens(
            trace=trace,
            refit=refit,
            codec=codec,
            record=record,
            directions=directions,
            tail_rows=tail_rows,
            unit_gradients=unit_gradients,
            unit_token_kl=unit_token_kl,
        )
    )

    scalar_provider, scalar_correction_rows = _prepare_v8_endpoint_provider(
        trace=trace,
        basis=basis,
        fit=fit,
        support=gain_support,
        arm="v6_exact_scalar",
    )
    scalar_vjp = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=endpoint_grid,
        vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
        h4_head=scalar_provider,
    )
    resources["scalar_endpoint_vjp_forward_count"] += 1
    resources["scalar_endpoint_vjp_backward_call_count"] += (
        scalar_vjp.backward_call_count
    )
    scalar_vjp.validate_integrity()
    v3diag._validate_candidate_execution(
        trace=trace, provider=scalar_provider, execution=scalar_vjp.execution
    )
    scalar_token_kl = scalar_vjp.token_kl_divergences.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    scalar_independent = teacher_kl._selected_token_teacher_kl(
        teacher_logits, scalar_vjp.execution.logits, endpoint_indices
    )
    scalar_gradients = (
        scalar_vjp.h4_gradients.detach()
        .to(device="cpu", dtype=torch.float64)[:, 0]
        .index_select(1, trace.support_indices)
        .contiguous()
    )
    scalar_future_maximum, scalar_future_nonzero = (
        path_diag._causal_future_gradient_summary(
            gradient_rows=scalar_gradients,
            endpoint_indices=endpoint_indices,
            support_indices=trace.support_indices,
            logical_positions=trace.prefix.logical_positions,
        )
    )
    if (
        scalar_vjp.teacher_logits_sha256 != teacher_sha
        or not torch.equal(
            scalar_vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
        )
        or not torch.allclose(
            scalar_token_kl, scalar_independent, rtol=0.0, atol=1.0e-6
        )
        or scalar_future_maximum != 0.0
        or scalar_future_nonzero != 0
    ):
        raise RuntimeError("V9 scalar endpoint VJP authority differs")
    scalar_h4_rows, scalar_receipt = _expected_v8_endpoint_execution(
        trace=trace,
        fit=fit,
        support=gain_support,
        codec=codec,
        record=record,
        refit=refit,
        arm="v6_exact_scalar",
        provider=scalar_provider,
        execution=scalar_vjp.execution,
        correction_rows=scalar_correction_rows,
        teacher_logits=teacher_logits,
        endpoint_grid=endpoint_grid,
        token_kl=scalar_token_kl,
        v8_observation=v8_observations[(trace.example_id, "v6_exact_scalar")],
    )

    joint_provider, joint_correction_rows = _prepare_v8_endpoint_provider(
        trace=trace,
        basis=basis,
        fit=fit,
        support=gain_support,
        arm="v7_joint",
    )
    joint_execution = bridge.execute(adapter, model_inputs, h4_head=joint_provider)
    resources["joint_boundary_forward_count"] += 1
    v3diag._validate_candidate_execution(
        trace=trace, provider=joint_provider, execution=joint_execution
    )
    joint_token_kl = teacher_kl._selected_token_teacher_kl(
        teacher_logits, joint_execution.logits, endpoint_indices
    ).detach().to(device="cpu", dtype=torch.float64).contiguous()
    joint_h4_rows, joint_receipt = _expected_v8_endpoint_execution(
        trace=trace,
        fit=fit,
        support=gain_support,
        codec=codec,
        record=record,
        refit=refit,
        arm="v7_joint",
        provider=joint_provider,
        execution=joint_execution,
        correction_rows=joint_correction_rows,
        teacher_logits=teacher_logits,
        endpoint_grid=endpoint_grid,
        token_kl=joint_token_kl,
        v8_observation=v8_observations[(trace.example_id, "v7_joint")],
    )
    if scalar_h4_rows.dtype != joint_h4_rows.dtype:
        raise RuntimeError("V9 scalar and joint endpoint dtypes differ")
    endpoint_pair: dict[str, object] = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "model_inputs_sha256": trace.model_inputs_sha256,
        "teacher_logits_sha256": teacher_sha,
        "supervised_grid_sha256": grid_sha,
        "support_mask_sha256": _runtime_tensor_sha256(support_mask),
        "support_indices_sha256": _runtime_tensor_sha256(trace.support_indices),
        "held_unit_endpoint": unit_receipt,
        "held_unit_fullfit_convention": held_fullfit_receipt,
        "scalar_endpoint": scalar_receipt,
        "joint_endpoint": joint_receipt,
        "scalar_endpoint_vjp_artifact_sha256": scalar_vjp.artifact_sha256,
        "scalar_endpoint_maximum_future_gradient_abs_hex": (
            scalar_future_maximum.hex()
        ),
        "scalar_endpoint_future_gradient_nonzero_count": scalar_future_nonzero,
        "held_unit_endpoint_vjp_artifact_sha256": unit_vjp.artifact_sha256,
        "held_unit_endpoint_maximum_future_gradient_abs_hex": (
            unit_future_maximum.hex()
        ),
        "held_unit_endpoint_future_gradient_nonzero_count": unit_future_nonzero,
        "actual_cast_once_endpoint_pair": True,
        "pinned_v4_held_unit_token_KL_hash_replayed": True,
        "v8_scalar_and_joint_token_KL_hashes_replayed": True,
        "raw_tensors_serialized": False,
    }
    endpoint_pair["artifact_sha256"] = token_v1._domain_sha256(
        endpoint_pair, domain=_ENDPOINT_PAIR_DOMAIN
    )

    # The core signature is deliberately supplied with all scalar tangent
    # provenance.  Its accumulator retains only authenticated scalar context.
    accumulator = CandidateJointStatePathAccumulator(
        example_id=trace.example_id,
        family_id=trace.family_id,
        scalar_endpoint_h4_rows=scalar_h4_rows,
        joint_endpoint_h4_rows=joint_h4_rows,
        scalar_token_teacher_kl=scalar_token_kl,
        joint_token_teacher_kl=joint_token_kl,
        endpoint_pair_binding_sha256=str(endpoint_pair["artifact_sha256"]),
        scalar_endpoint_execution_artifact_sha256=(
            scalar_vjp.execution.artifact_sha256
        ),
        joint_endpoint_execution_artifact_sha256=joint_execution.artifact_sha256,
        supervised_grid_sha256=grid_sha,
        teacher_logits_sha256=teacher_sha,
        scalar_endpoint_token_h4_gradients=scalar_gradients,
        scalar_tangent_vjp_artifact_sha256=scalar_vjp.artifact_sha256,
        scalar_tangent_provider_artifact_sha256=scalar_provider.artifact_sha256,
        scalar_tangent_execution_artifact_sha256=(
            scalar_vjp.execution.artifact_sha256
        ),
        scalar_tangent_maximum_future_gradient_abs=scalar_future_maximum,
        scalar_tangent_future_gradient_nonzero_count=scalar_future_nonzero,
        held_unit_endpoint_h4_rows=unit_h4_rows,
        held_unit_token_teacher_kl=unit_token_kl,
        held_unit_endpoint_token_h4_gradients=unit_gradients,
        held_unit_tangent_vjp_artifact_sha256=unit_vjp.artifact_sha256,
        held_unit_tangent_provider_artifact_sha256=unit_provider.artifact_sha256,
        held_unit_tangent_execution_artifact_sha256=(
            unit_vjp.execution.artifact_sha256
        ),
        held_unit_tangent_maximum_future_gradient_abs=unit_future_maximum,
        held_unit_tangent_future_gradient_nonzero_count=unit_future_nonzero,
    )

    node_receipts: list[Mapping[str, object]] = []
    for node_index, (alpha, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS, strict=True)
    ):
        provider = _AuthenticatedV9ScalarJointPathProvider(
            example_id=trace.example_id,
            family_id=trace.family_id,
            endpoint_pair_binding_sha256=str(endpoint_pair["artifact_sha256"]),
            model_inputs_sha256=trace.model_inputs_sha256,
            bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
            prefix_artifact_sha256=trace.prefix.artifact_sha256,
            node_index=node_index,
            path_fraction=alpha,
            quadrature_weight=weight,
            base_h4=trace.base_h4,
            support_mask=support_mask,
            support_indices=trace.support_indices,
            scalar_endpoint_h4_rows=scalar_h4_rows,
            joint_endpoint_h4_rows=joint_h4_rows,
        )
        node_vjp = bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=teacher_logits,
            supervised_indices=endpoint_grid,
            vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
            h4_head=provider,
        )
        resources["path_teacher_kl_vjp_forward_count"] += 1
        resources["path_teacher_kl_vjp_backward_call_count"] += (
            node_vjp.backward_call_count
        )
        resources["path_quadrature_node_count"] += 1
        node_vjp.validate_integrity()
        provider.validate_integrity()
        _validate_v9_path_node_full_h4(
            trace=trace,
            provider=provider,
            execution=node_vjp.execution,
        )
        token_kl = node_vjp.token_kl_divergences.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        independent_kl = teacher_kl._selected_token_teacher_kl(
            teacher_logits, node_vjp.execution.logits, endpoint_indices
        )
        gradient_rows = (
            node_vjp.h4_gradients.detach()
            .to(device="cpu", dtype=torch.float64)[:, 0]
            .index_select(1, trace.support_indices)
            .contiguous()
        )
        node_h4_rows = _realized_support_rows(
            node_vjp.execution.candidate_h4, trace.support_indices
        )
        maximum_future, future_nonzero = path_diag._causal_future_gradient_summary(
            gradient_rows=gradient_rows,
            endpoint_indices=endpoint_indices,
            support_indices=trace.support_indices,
            logical_positions=trace.prefix.logical_positions,
        )
        if (
            not provider.used
            or node_vjp.execution.model_inputs_sha256 != trace.model_inputs_sha256
            or node_vjp.execution.bridge_binding_sha256
            != trace.prefix.bridge_binding_sha256
            or node_vjp.execution.prefix.artifact_sha256
            != trace.prefix.artifact_sha256
            or node_vjp.execution.h4_head_sha256 != provider.artifact_sha256
            or node_vjp.teacher_logits_sha256 != teacher_sha
            or _runtime_tensor_sha256(node_vjp.execution.candidate_x4)
            != trace.base_x4_sha256
            or not torch.equal(
                node_vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
            )
            or not torch.allclose(token_kl, independent_kl, rtol=0.0, atol=1.0e-6)
            or not token_v1._bitwise_equal(node_h4_rows, provider.path_h4_rows)
            or maximum_future != 0.0
            or future_nonzero != 0
        ):
            raise RuntimeError("V9 scalar-joint GL4 node execution differs")
        core_receipt = accumulator.add_node(
            node_index=node_index,
            path_fraction=alpha,
            quadrature_weight=weight,
            path_node_h4_rows=node_h4_rows,
            token_h4_gradients=gradient_rows,
            token_teacher_kl=token_kl,
            vjp_artifact_sha256=node_vjp.artifact_sha256,
            provider_artifact_sha256=provider.artifact_sha256,
            execution_artifact_sha256=node_vjp.execution.artifact_sha256,
            maximum_future_gradient_abs=maximum_future,
            future_gradient_nonzero_count=future_nonzero,
        )
        receipt: dict[str, object] = {
            "example_id": trace.example_id,
            "family_id": trace.family_id,
            "node_index": node_index,
            "path_fraction_hex": alpha.hex(),
            "quadrature_weight_hex": weight.hex(),
            "endpoint_pair_binding_sha256": endpoint_pair["artifact_sha256"],
            "core_node_receipt_sha256": core_receipt.artifact_sha256,
            "provider_artifact_sha256": provider.artifact_sha256,
            "execution_artifact_sha256": node_vjp.execution.artifact_sha256,
            "vjp_artifact_sha256": node_vjp.artifact_sha256,
            "candidate_h4_sha256": _runtime_tensor_sha256(
                node_vjp.execution.candidate_h4
            ),
            "path_node_h4_rows_sha256": _runtime_tensor_sha256(node_h4_rows),
            "candidate_logits_sha256": _runtime_tensor_sha256(
                node_vjp.execution.logits
            ),
            "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
            "backward_call_count": node_vjp.backward_call_count,
            "maximum_future_gradient_abs_hex": maximum_future.hex(),
            "future_gradient_nonzero_count": future_nonzero,
            "executed_actual_cast_once_scalar_to_joint_H4": True,
            "raw_gradient_logits_or_H4_serialized": False,
        }
        receipt["receipt_sha256"] = token_v1._domain_sha256(
            receipt, domain=_PATH_NODE_RECEIPT_DOMAIN
        )
        node_receipts.append(receipt)
        del node_vjp, provider, gradient_rows, token_kl, independent_kl, node_h4_rows

    evidence = accumulator.finalize()
    held_unit_actual = (
        candidate_joint_state_held_unit_endpoint_tangent_contraction(evidence)
    )
    scalar_tangent = candidate_joint_state_scalar_endpoint_tangent_contraction(
        evidence
    )
    if held_unit_actual is None or scalar_tangent is None:
        raise RuntimeError("V9 endpoint tangent evidence is incomplete")
    path_integral = candidate_joint_state_path_integrated_contraction(evidence)
    finite_delta = candidate_joint_state_finite_kl_delta(evidence)
    if not (
        held_fullfit_tokens.shape
        == held_unit_actual.shape
        == scalar_tangent.shape
        == path_integral.shape
        == finite_delta.shape
    ):
        raise RuntimeError("V9 additive prompt attribution geometry differs")
    additive_scalars = {
        "held_unit_fullfit_convention": float(held_fullfit_tokens.mean()),
        "held_unit_actual_displacement_tangent": float(held_unit_actual.mean()),
        "scalar_endpoint_tangent": float(scalar_tangent.mean()),
        "GL4_path_integral": float(path_integral.mean()),
        "finite_joint_minus_scalar_teacher_KL": float(finite_delta.mean()),
    }
    endpoint_receipt = {
        **endpoint_pair,
        "core_evidence_artifact_sha256": evidence.artifact_sha256,
        "core_scalar_tangent_receipt_sha256": (
            evidence.scalar_endpoint_tangent_receipt.artifact_sha256
        ),
        "core_held_unit_tangent_receipt_sha256": (
            evidence.held_unit_endpoint_tangent_receipt.artifact_sha256
        ),
        "additive_prompt_scalars": additive_scalars,
        "path_node_receipt_sha256s": tuple(
            str(row["receipt_sha256"]) for row in node_receipts
        ),
    }
    del (
        unit_vjp,
        unit_provider,
        unit_gradients,
        unit_independent,
        scalar_vjp,
        scalar_provider,
        joint_execution,
        joint_provider,
        scalar_gradients,
        scalar_independent,
        teacher_logits,
        model_inputs,
        indices,
        targets,
        held_fullfit_tokens,
        held_unit_actual,
        scalar_tangent,
        path_integral,
        finite_delta,
    )
    return _PromptPathResult(
        evidence=evidence,
        endpoint_receipt=endpoint_receipt,
        node_receipts=tuple(node_receipts),
        additive_scalars=additive_scalars,
        resources=resources,
    )


@dataclass(slots=True)
class _LivePathGridResult:
    evidence: tuple[CandidateJointStatePathEvidence, ...]
    attribution: CandidateJointStatePathAttribution
    endpoint_receipts: tuple[Mapping[str, object], ...]
    node_receipts: tuple[Mapping[str, object], ...]
    prompt_additive_scalars: Mapping[str, Mapping[str, float]]
    observation_set_sha256: str
    resources: Mapping[str, int]


def _execute_live_path_grid(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    analytic: v8diag._V8AnalyticPhaseResults,
    v8_observations: Mapping[tuple[str, str], Mapping[str, object]],
) -> _LivePathGridResult:
    values = tuple(sorted(traces, key=lambda value: value.example_id))
    families = tuple(sorted(fits))
    fold_records = {
        record.outer_held_family_id: record
        for record in analytic.v7_phases.joint_fold_records
    }
    if (
        len(values) != _EXPECTED_PROMPTS
        or len(families) != _EXPECTED_FAMILIES
        or set(analytic.codecs) != set(families)
        or set(fold_records) != set(families)
        or set(analytic.unit_observations)
        != {trace.example_id for trace in values}
        or any(sum(trace.family_id == family for trace in values) != 2 for family in families)
    ):
        raise RuntimeError("V9 held scalar-joint path universe differs")
    totals = {
        "fresh_native_teacher_forward_count": 0,
        "held_unit_endpoint_vjp_forward_count": 0,
        "held_unit_endpoint_vjp_backward_call_count": 0,
        "scalar_endpoint_vjp_forward_count": 0,
        "scalar_endpoint_vjp_backward_call_count": 0,
        "joint_boundary_forward_count": 0,
        "path_teacher_kl_vjp_forward_count": 0,
        "path_teacher_kl_vjp_backward_call_count": 0,
        "path_quadrature_node_count": 0,
    }
    evidence: list[CandidateJointStatePathEvidence] = []
    endpoint_receipts: list[Mapping[str, object]] = []
    node_receipts: list[Mapping[str, object]] = []
    prompt_additive_scalars: dict[str, Mapping[str, float]] = {}
    for trace in values:
        family = trace.family_id
        fit = fits[family]
        record = fold_records[family]
        refit = analytic.v7_phases.v6_phases.row_bank.refits[family]
        codec = analytic.codecs[family]
        directions = v6diag._ordered_k64(fit)
        base_rows = _realized_support_rows(
            trace.base_h4, trace.support_indices
        ).to(torch.float64)
        gain_support = build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            codec,
            record.full_scalar_control_fit,
            record.full_joint_fit,
            phase="held",
            example_id=trace.example_id,
            family_id=trace.family_id,
            base_h4_support_rows=base_rows,
            ordered_directions=directions,
        )
        result = _execute_prompt_path(
            context=context,
            trace=trace,
            basis=basis,
            fit=fit,
            gain_support=gain_support,
            codec=codec,
            record=record,
            refit=refit,
            unit_observation=analytic.unit_observations[trace.example_id],
            v8_observations=v8_observations,
        )
        evidence.append(result.evidence)
        endpoint_receipts.append(result.endpoint_receipt)
        node_receipts.extend(result.node_receipts)
        prompt_additive_scalars[trace.example_id] = result.additive_scalars
        for key, value in result.resources.items():
            totals[key] += value
    if (
        len(evidence) != _EXPECTED_PROMPTS
        or len(endpoint_receipts) != _EXPECTED_PROMPTS
        or len(node_receipts) != _EXPECTED_PATH_VJP_FORWARDS
    ):
        raise RuntimeError("V9 live path receipt grid differs")
    attribution = summarize_candidate_joint_state_path_attribution(evidence)
    ordered_receipts = tuple(
        sorted(
            node_receipts,
            key=lambda row: (str(row["example_id"]), int(row["node_index"])),
        )
    )
    observation_set_sha256 = token_v1._domain_sha256(
        tuple(str(row["receipt_sha256"]) for row in ordered_receipts),
        domain=_PATH_OBSERVATION_SET_DOMAIN,
    )
    return _LivePathGridResult(
        evidence=tuple(evidence),
        attribution=attribution,
        endpoint_receipts=tuple(endpoint_receipts),
        node_receipts=ordered_receipts,
        prompt_additive_scalars=prompt_additive_scalars,
        observation_set_sha256=observation_set_sha256,
        resources=totals,
    )


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation vectors must have equal nontrivial length")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = tuple(value - left_mean for value in left)
    centered_right = tuple(value - right_mean for value in right)
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_left)
        * math.fsum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return None
    return math.fsum(
        x * y for x, y in zip(centered_left, centered_right, strict=True)
    ) / denominator


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return tuple(ranks)


def _association_with_finite_D(
    values: Sequence[float], finite: Sequence[float]
) -> dict[str, object]:
    pearson = _pearson(values, finite)
    spearman = _pearson(_average_ranks(values), _average_ranks(finite))
    agreements = tuple(
        _sign(value) == _sign(target)
        for value, target in zip(values, finite, strict=True)
    )
    return {
        "pearson_correlation_with_finite_D": pearson,
        "spearman_rank_correlation_with_finite_D": spearman,
        "sign_agreement_count_with_finite_D": sum(agreements),
        "sign_agreement_rate_with_finite_D": sum(agreements) / len(agreements),
        "sign_zero_policy": "zero_agrees_only_with_zero",
        "correlation_undefined_on_zero_variance": pearson is None or spearman is None,
    }


def _additive_attribution_ledger(
    *,
    attribution: CandidateJointStatePathAttribution,
    joint_fold_records: Sequence[object],
    prompt_additive_scalars: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Build the exact family-equal U-to-D additive FTC attribution ledger."""

    family_summaries = {
        family.family_id: family for family in attribution.family_summaries
    }
    records = {
        str(record.outer_held_family_id): record for record in joint_fold_records
    }
    if (
        len(family_summaries) != _EXPECTED_FAMILIES
        or set(records) != set(family_summaries)
        or set(prompt_additive_scalars)
        != {
            example
            for family in attribution.family_summaries
            for example in family.example_ids
        }
    ):
        raise RuntimeError("V7/V9 additive attribution universe differs")
    rows: list[dict[str, object]] = []
    for family_id in sorted(family_summaries):
        family = family_summaries[family_id]
        record = records[family_id]
        members = tuple(prompt_additive_scalars[value] for value in family.example_ids)

        def prompt_mean(name: str) -> float:
            values = tuple(float(member[name]) for member in members)
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise RuntimeError("V9 additive prompt family geometry differs")
            return math.fsum(values) / len(values)

        u_inner = float(record.joint_inner_macro_derivative) - float(
            record.scalar_inner_macro_derivative
        )
        u_held_fullfit = prompt_mean("held_unit_fullfit_convention")
        u_held_actual = prompt_mean("held_unit_actual_displacement_tangent")
        t_scalar = prompt_mean("scalar_endpoint_tangent")
        path = prompt_mean("GL4_path_integral")
        finite = prompt_mean("finite_joint_minus_scalar_teacher_KL")
        if (
            family.mean_scalar_endpoint_tangent is None
            or family.mean_held_unit_endpoint_tangent is None
        ):
            raise RuntimeError("V9 core endpoint tangent coverage differs")
        if (
            abs(finite - float(family.mean_finite_kl_delta))
            > 128.0 * torch.finfo(torch.float64).eps
            or abs(path - float(family.mean_path_integral))
            > 128.0 * torch.finfo(torch.float64).eps
            or abs(t_scalar - float(family.mean_scalar_endpoint_tangent))
            > 128.0 * torch.finfo(torch.float64).eps
            or abs(u_held_actual - float(family.mean_held_unit_endpoint_tangent))
            > 128.0 * torch.finfo(torch.float64).eps
        ):
            raise RuntimeError("V9 additive and core family summaries differ")
        stages = {
            "U_inner_cv": u_inner,
            "U_held_fullfit_convention": u_held_fullfit,
            "U_held_actual_displacement": u_held_actual,
            "T_scalar_endpoint": t_scalar,
            "P_GL4_path_integral": path,
            "D_finite_joint_minus_scalar": finite,
        }
        components = {
            "hybrid_analytic_screen_to_held_fullfit_mixed_estimand_shift": (
                u_held_fullfit - u_inner
            ),
            "hybrid_nominal_to_realized_finite_cast_convention_shift": (
                u_held_actual - u_held_fullfit
            ),
            "off_path_unit_to_scalar_gradient_reference_shift": (
                t_scalar - u_held_actual
            ),
            "GL4_estimated_path_gradient_transport": path - t_scalar,
            "path_closure_residual": finite - path,
        }
        target = finite - u_inner
        component_sum = math.fsum(components.values())
        residual = target - component_sum
        tolerance = 512.0 * torch.finfo(torch.float64).eps * max(
            1.0, abs(target), abs(component_sum)
        )
        if abs(residual) > tolerance:
            raise RuntimeError("V9 additive FTC ledger does not close")
        rows.append(
            {
                "family_id": family_id,
                "example_ids": family.example_ids,
                "stages": stages,
                "additive_components": components,
                "target_D_minus_U_inner_cv": target,
                "additive_component_sum": component_sum,
                "additive_identity_residual": residual,
                "additive_identity_tolerance": tolerance,
                "additive_identity_closes": True,
            }
        )

    stage_names = tuple(rows[0]["stages"].keys())  # type: ignore[union-attr]
    component_names = tuple(
        rows[0]["additive_components"].keys()  # type: ignore[union-attr]
    )
    aggregate_stages = {
        name: math.fsum(float(row["stages"][name]) for row in rows) / len(rows)  # type: ignore[index]
        for name in stage_names
    }
    aggregate_components = {
        name: math.fsum(
            float(row["additive_components"][name]) for row in rows  # type: ignore[index]
        )
        / len(rows)
        for name in component_names
    }
    aggregate_target = math.fsum(
        float(row["target_D_minus_U_inner_cv"]) for row in rows
    ) / len(rows)
    aggregate_sum = math.fsum(aggregate_components.values())
    aggregate_residual = aggregate_target - aggregate_sum
    core_aggregate_checks = {
        "U_held_actual": (
            attribution.mean_held_unit_endpoint_tangent
            is not None
            and abs(
                aggregate_stages["U_held_actual_displacement"]
                - float(attribution.mean_held_unit_endpoint_tangent)
            )
            <= 128.0 * torch.finfo(torch.float64).eps
        ),
        "T_scalar": (
            attribution.mean_scalar_endpoint_tangent is not None
            and abs(
                aggregate_stages["T_scalar_endpoint"]
                - float(attribution.mean_scalar_endpoint_tangent)
            )
            <= 128.0 * torch.finfo(torch.float64).eps
        ),
        "P_GL4": abs(
            aggregate_stages["P_GL4_path_integral"]
            - float(attribution.mean_path_integral)
        )
        <= 128.0 * torch.finfo(torch.float64).eps,
        "D_finite": abs(
            aggregate_stages["D_finite_joint_minus_scalar"]
            - float(attribution.mean_finite_kl_delta)
        )
        <= 128.0 * torch.finfo(torch.float64).eps,
    }
    if not all(core_aggregate_checks.values()):
        raise RuntimeError("V9 additive and core aggregate summaries differ")
    finite_values = tuple(
        float(row["stages"]["D_finite_joint_minus_scalar"])  # type: ignore[index]
        for row in rows
    )
    predictor_stage_names = tuple(
        name for name in stage_names if name != "D_finite_joint_minus_scalar"
    )
    associations = {
        name: _association_with_finite_D(
            tuple(float(row["stages"][name]) for row in rows),  # type: ignore[index]
            finite_values,
        )
        for name in predictor_stage_names
    }
    component_associations = {
        name: _association_with_finite_D(
            tuple(
                float(row["additive_components"][name])  # type: ignore[index]
                for row in rows
            ),
            finite_values,
        )
        for name in component_names
    }
    return {
        "FTC_orientation": "joint_minus_scalar_teacher_KL",
        "exact_additive_identity": (
            "D-U_inner_cv=(U_held_fullfit-U_inner_cv)+"
            "(U_held_actual-U_held_fullfit)+(T_scalar-U_held_actual)+"
            "(P-T_scalar)+(D-P)"
        ),
        "family_rows": tuple(rows),
        "family_count": len(rows),
        "family_equal_aggregate": {
            "stages": aggregate_stages,
            "additive_components": aggregate_components,
            "target_D_minus_U_inner_cv": aggregate_target,
            "additive_component_sum": aggregate_sum,
            "additive_identity_residual": aggregate_residual,
            "live_stage_core_aggregation_checks": core_aggregate_checks,
        },
        "family_association_with_finite_D": {
            "stages": associations,
            "additive_components": component_associations,
        },
        "interpretation_boundaries": {
            "U_inner_cv": (
                "V7_nested_inner_six_family_macro_of_V4_unit_H4_objective_"
                "gradients_contracted_with_zero_logit_gain_directions"
            ),
            "U_held_fullfit_convention": (
                "V4_unit_H4_objective_gradient_contracted_with_V7_zero_logit_"
                "gain_direction_using_outer_full_seven_fit_coefficients"
            ),
            "U_held_actual_displacement": (
                "fresh_held_unit_gradient_contracted_with_actual_cast_once_"
                "scalar_to_joint_displacement"
            ),
            "U_inner_and_U_held_fullfit_are_literal_realized_endpoint_derivatives": (
                False
            ),
            "V7_zero_logit_ell_zero_is_V5_plus_carrier": True,
            "hybrid_screen_to_held_fullfit_estimand_shift_mixes": (
                "fit_codec_population_and_prompt"
            ),
            "hybrid_screen_to_held_fullfit_shift_is_unique_cause": False,
            "hybrid_nominal_to_realized_convention_shift_mixes": (
                "unit_H4_zero_logit_analytic_convention_tanh_finite_nominal_"
                "versus_realized_direction_and_cast"
            ),
            "hybrid_nominal_to_realized_shift_is_unique_cause": False,
            "unit_to_scalar_reference_shift_is_off_path": True,
            "held_unit_endpoint_need_not_lie_on_scalar_to_joint_path": True,
            "GL4_estimated_path_gradient_transport_is_P_minus_T_scalar": True,
            "gradient_transport_curvature_interpretation_requires_closure": True,
            "path_closure_residual_is_D_minus_P": True,
            "path_closure_residual_mixes": (
                "quadrature_truncation_node_cast_execution_and_VJP_numerics"
            ),
            "casted_scalar_to_joint_path_is_continuous": False,
            "exact_continuous_FTC_claimed": False,
            "binary_single_cause_claim_forced": False,
        },
        "family_correlation_sample_count": len(rows),
        "correlations_are_descriptive_and_algebraically_coupled": True,
        "raw_tensors_serialized": False,
    }


def _v8_finite_delta_anchor(
    *, v8_report: Mapping[str, object], attribution: CandidateJointStatePathAttribution
) -> dict[str, object]:
    pairwise = v8_report.get("joint_pairwise_control_results")
    if not isinstance(pairwise, Mapping):
        raise ValueError("pinned V8 pairwise result differs")
    scalar = pairwise.get("joint_vs_scalar")
    if not isinstance(scalar, Mapping):
        raise ValueError("pinned V8 scalar pairwise result differs")
    pinned = float(scalar["joint_family_equal_mean_teacher_kl"]) - float(
        scalar["control_family_equal_mean_teacher_kl"]
    )
    live = float(attribution.mean_finite_kl_delta)
    difference = live - pinned
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
        1.0, abs(live), abs(pinned)
    )
    if abs(difference) > tolerance:
        raise RuntimeError("V9 finite scalar-joint delta did not replay V8")
    return {
        "pinned_v8_family_equal_joint_minus_scalar_teacher_KL": pinned,
        "live_v9_family_equal_joint_minus_scalar_teacher_KL": live,
        "live_minus_pinned": difference,
        "fixed_numeric_equality_tolerance": tolerance,
        "within_fixed_numeric_equality_tolerance": True,
        "used_as_integrity_anchor_not_scientific_effect_gate": True,
    }


def _resource_accounting(
    *,
    endpoint_resources: Mapping[str, int],
    gradient_resources: Mapping[str, int],
    live_resources: Mapping[str, int],
    row_bank_candidate_support_row_executions: int,
) -> dict[str, object]:
    """Require actual VJP backward chunks and the exact 240F/1148B ledger."""

    parent_forwards = (
        endpoint_resources["base_forward_count"]
        + endpoint_resources["native_forward_count"]
        + endpoint_resources["endpoint_token_vjp_forward_count"]
    )
    parent_backwards = endpoint_resources[
        "endpoint_token_vjp_backward_call_count"
    ]
    analytic_forwards = (
        gradient_resources["gradient_native_forward_count"]
        + gradient_resources["gradient_candidate_vjp_forward_count"]
    )
    analytic_backwards = gradient_resources[
        "gradient_candidate_vjp_backward_call_count"
    ]
    lineage_forwards = parent_forwards + analytic_forwards
    lineage_backwards = parent_backwards + analytic_backwards
    fresh_forwards = (
        live_resources["fresh_native_teacher_forward_count"]
        + live_resources["held_unit_endpoint_vjp_forward_count"]
        + live_resources["scalar_endpoint_vjp_forward_count"]
        + live_resources["joint_boundary_forward_count"]
        + live_resources["path_teacher_kl_vjp_forward_count"]
    )
    fresh_backwards = (
        live_resources["held_unit_endpoint_vjp_backward_call_count"]
        + live_resources["scalar_endpoint_vjp_backward_call_count"]
        + live_resources["path_teacher_kl_vjp_backward_call_count"]
    )
    total_forwards = lineage_forwards + fresh_forwards
    total_backwards = lineage_backwards + fresh_backwards
    support_rows = endpoint_resources["complete_h4_support_row_count"]
    fresh_candidate_support_row_executions = support_rows * 7
    total_candidate_support_row_executions = (
        row_bank_candidate_support_row_executions
        + fresh_candidate_support_row_executions
    )
    if (
        parent_forwards != 48
        or parent_backwards != 109
        or analytic_forwards != 64
        or analytic_backwards != 385
        or lineage_forwards != _EXPECTED_LINEAGE_FORWARDS
        or lineage_backwards != _EXPECTED_LINEAGE_BACKWARDS
        or live_resources["fresh_native_teacher_forward_count"] != 16
        or live_resources["held_unit_endpoint_vjp_forward_count"] != 16
        or live_resources["held_unit_endpoint_vjp_backward_call_count"] != 109
        or live_resources["scalar_endpoint_vjp_forward_count"] != 16
        or live_resources["scalar_endpoint_vjp_backward_call_count"] != 109
        or live_resources["joint_boundary_forward_count"] != 16
        or live_resources["path_teacher_kl_vjp_forward_count"] != 64
        or live_resources["path_teacher_kl_vjp_backward_call_count"] != 436
        or live_resources["path_quadrature_node_count"] != 64
        or fresh_forwards != _EXPECTED_FRESH_FORWARDS
        or fresh_backwards != _EXPECTED_FRESH_BACKWARDS
        or total_forwards != _EXPECTED_TOTAL_FORWARDS
        or total_backwards != _EXPECTED_TOTAL_BACKWARDS
        or row_bank_candidate_support_row_executions != 2842
        or support_rows != 819
        or fresh_candidate_support_row_executions != 5733
        or total_candidate_support_row_executions != 8575
    ):
        raise RuntimeError("V9 scalar-joint path resource accounting differs")
    return {
        **endpoint_resources,
        **gradient_resources,
        **live_resources,
        "phase_order": (
            "parent_endpoint_recollection",
            "exact_v7_analytic_phase_reproduction_and_v8_authentication",
            "frozen_full_seven_codec_reconstruction",
            "fresh_native_teacher_once_per_outer_held_prompt",
            "held_unit_endpoint_teacher_KL_VJP_and_pinned_V4_replay",
            "scalar_endpoint_teacher_KL_VJP",
            "joint_endpoint_boundary_execution",
            "four_fixed_GL4_interior_teacher_KL_VJPs",
            "family_equal_finite_path_closure_publication",
        ),
        "parent_collection_model_forward_count": parent_forwards,
        "parent_collection_backward_call_count": parent_backwards,
        "shared_analytic_model_forward_count": analytic_forwards,
        "shared_analytic_backward_call_count": analytic_backwards,
        "lineage_model_forward_count": lineage_forwards,
        "lineage_backward_call_count": lineage_backwards,
        "fresh_attribution_model_forward_count": fresh_forwards,
        "fresh_attribution_backward_call_count": fresh_backwards,
        "row_bank_candidate_support_row_executions": (
            row_bank_candidate_support_row_executions
        ),
        "fresh_unit_scalar_joint_and_GL4_candidate_support_row_executions": (
            fresh_candidate_support_row_executions
        ),
        "total_candidate_support_row_executions": (
            total_candidate_support_row_executions
        ),
        "tune_model_forward_count": 0,
        "selection_model_forward_count": 0,
        "step_grid_model_forward_count": 0,
        "fallback_model_forward_count": 0,
        "total_model_forward_count": total_forwards,
        "total_backward_call_count": total_backwards,
        "exact_model_forward_count_is_240": total_forwards == 240,
        "exact_backward_call_count_is_1148": total_backwards == 1148,
        "backward_accounting_uses_actual_vjp_backward_call_count": True,
        "raw_logits_gradients_or_H4_retained_in_report": False,
    }


def _safety_metadata() -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_gain_vectors": False,
        "contains_token_teacher_kl_tensors": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


def _quadrature_closure_gate_results(
    attribution: CandidateJointStatePathAttribution,
) -> dict[str, bool]:
    """Apply the fixed pre-live GL4 closure policy, without tuning it."""

    return {
        "overall_family_prompt_token_closure_relative_RMSE_at_most_0_05": (
            attribution.closure_relative_rmse
            <= _OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
        ),
        "every_family_closure_relative_RMSE_at_most_0_10": all(
            family.closure_relative_rmse
            <= _FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
            for family in attribution.family_summaries
        ),
        "overall_closure_cosine_at_least_0_99": (
            attribution.closure_cosine >= _CLOSURE_COSINE_MINIMUM
        ),
    }


def _classification(*, integrity_passed: bool, closure_established: bool) -> str:
    if not integrity_passed:
        return "integrity_failure"
    if not closure_established:
        return "scalar_joint_path_closure_unresolved_same_a"
    return "scalar_joint_GL4_closure_established_same_a"


def _evidence_grid_has_exact_path_nodes(evidence: Sequence[object]) -> bool:
    """Require four finalized node receipts on every one of the 16 prompts."""

    return len(evidence) == _EXPECTED_PROMPTS and all(
        isinstance(value, CandidateJointStatePathEvidence)
        and len(value.node_receipts) == _EXPECTED_PATH_NODES
        for value in evidence
    )


def _require_integrity_gate_results(gates: Mapping[str, bool]) -> None:
    failures = tuple(sorted(name for name, passed in gates.items() if not passed))
    if failures:
        raise RuntimeError(
            "V9 scalar-joint path integrity failed before publication: "
            + ", ".join(failures)
        )


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_scalar_joint_path_attribution_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    v5_report_path: Path | str = DEFAULT_V5_REPORT,
    v6_report_path: Path | str = DEFAULT_V6_REPORT,
    v7_report_path: Path | str = DEFAULT_V7_REPORT,
    v8_report_path: Path | str = DEFAULT_V8_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the locked same-A V9 scalar-to-joint path attribution."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite candidate joint state-gain V9 path report"
        )
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = v4diag._load_v3_report(v3_report_path)
    v4_report = v5diag._load_v4_report(v4_report_path)
    v5_report = v6diag._load_v5_report(v5_report_path)
    v6_report = v7diag._load_v6_report(v6_report_path)
    v7_report = v8diag._load_v7_report(v7_report_path)
    v8_report = _load_v8_report(v8_report_path)
    v8_observations = _index_v8_endpoint_observations(v8_report)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate joint state-gain V9 rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate joint state-gain V9 rank320 transfer",
    )
    transfer_receipts = expanded._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = token_v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != _EXPECTED_PROMPTS or len(families) != _EXPECTED_FAMILIES:
            raise RuntimeError("candidate joint state-gain V9 A16 panel differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        analytic = v8diag._execute_v8_analytic_phases(
            context=context,
            parent=parent,
            v3_report=v3_report,
            v4_report=v4_report,
            v5_report=v5_report,
            v6_report=v6_report,
            v7_report=v7_report,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        v8_binding = _authenticate_v8_live_lineage(
            v8_report=v8_report, analytic=analytic
        )
        live = _execute_live_path_grid(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            analytic=analytic,
            v8_observations=v8_observations,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    resources = _resource_accounting(
        endpoint_resources=endpoint_resources,
        gradient_resources=analytic.v7_phases.v6_phases.row_bank.resources,
        live_resources=live.resources,
        row_bank_candidate_support_row_executions=(
            v8diag._row_bank_candidate_support_row_executions(
                analytic.v7_phases
            )
        ),
    )
    attribution_metadata = live.attribution.metadata()
    finite_anchor = _v8_finite_delta_anchor(
        v8_report=v8_report, attribution=live.attribution
    )
    additive_ledger = _additive_attribution_ledger(
        attribution=live.attribution,
        joint_fold_records=analytic.v7_phases.joint_fold_records,
        prompt_additive_scalars=live.prompt_additive_scalars,
    )
    endpoint_replay_count = sum(
        bool(receipt["v8_scalar_and_joint_token_KL_hashes_replayed"])
        for receipt in live.endpoint_receipts
    )
    unit_replay_count = sum(
        bool(receipt["pinned_v4_held_unit_token_KL_hash_replayed"])
        for receipt in live.endpoint_receipts
    )
    future_nonzero = sum(
        int(receipt["future_gradient_nonzero_count"])
        for receipt in live.node_receipts
    )
    integrity_gates = {
        "exact_v8_file_and_logical_hash_authenticated": (
            v8_binding["v8_report_file_sha256"] == V8_REPORT_FILE_SHA256
            and v8_binding["v8_report_sha256"] == V8_REPORT_SHA256
        ),
        "live_v7_analytic_lineage_canonically_reproduced_before_v9_work": (
            v8_binding["authenticated_before_any_v9_fresh_forward"] is True
            and v8_binding["v7_live_binding_canonically_equal"] is True
            and v8_binding["full_seven_codec_binding_canonically_equal"] is True
        ),
        "all_16_scalar_and_joint_boundary_token_KL_hashes_replay_v8": (
            endpoint_replay_count == _EXPECTED_PROMPTS
        ),
        "all_16_held_unit_boundary_token_KL_hashes_replay_pinned_v4": (
            unit_replay_count == _EXPECTED_PROMPTS
        ),
        "all_16_prompts_have_exactly_four_fixed_GL4_nodes": (
            _evidence_grid_has_exact_path_nodes(live.evidence)
            and len(live.node_receipts) == _EXPECTED_PATH_VJP_FORWARDS
        ),
        "every_GL4_node_executes_actual_cast_once_scalar_to_joint_H4": all(
            bool(receipt["executed_actual_cast_once_scalar_to_joint_H4"])
            for receipt in live.node_receipts
        ),
        "unit_scalar_and_GL4_token_VJPs_have_zero_future_H4_gradient": (
            future_nonzero == 0
            and all(
                int(receipt["scalar_endpoint_future_gradient_nonzero_count"]) == 0
                and int(receipt["held_unit_endpoint_future_gradient_nonzero_count"])
                == 0
                for receipt in live.endpoint_receipts
            )
        ),
        "family_equal_finite_delta_replays_v8": bool(
            finite_anchor["within_fixed_numeric_equality_tolerance"]
        ),
        "no_v9_fit_selection_grid_damping_fallback_or_serving_route": (
            resources["tune_model_forward_count"] == 0
            and resources["selection_model_forward_count"] == 0
            and resources["step_grid_model_forward_count"] == 0
            and resources["fallback_model_forward_count"] == 0
        ),
        "exact_model_forward_count_is_240": (
            resources["total_model_forward_count"] == _EXPECTED_TOTAL_FORWARDS
        ),
        "exact_backward_call_count_is_1148": (
            resources["total_backward_call_count"] == _EXPECTED_TOTAL_BACKWARDS
        ),
    }
    integrity_passed = all(integrity_gates.values())
    _require_integrity_gate_results(integrity_gates)
    closure_gates = _quadrature_closure_gate_results(live.attribution)
    closure_established = all(closure_gates.values())
    classification = _classification(
        integrity_passed=integrity_passed,
        closure_established=closure_established,
    )
    passed = integrity_passed and closure_established
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_attribution_only",
            "outer_held_prompt_count": _EXPECTED_PROMPTS,
            "outer_family_count": _EXPECTED_FAMILIES,
            "prompts_per_outer_family": 2,
            "scalar_endpoint": "exact_frozen_full_seven_V6_scalar_control",
            "joint_endpoint": "exact_frozen_full_seven_V7_joint_field",
            "held_unit_endpoint": (
                "exact_pinned_V4_unit_K64_reexecuted_as_fresh_held_prompt_VJP"
            ),
            "path": (
                "straight_line_between_realized_cast_once_scalar_and_joint_H4_rows"
            ),
            "quadrature_rule": "fixed_gauss_legendre_order_4_on_unit_interval",
            "quadrature_nodes_hex": tuple(
                value.hex() for value in GL4_UNIT_INTERVAL_NODES
            ),
            "quadrature_weights_hex": tuple(
                value.hex() for value in GL4_UNIT_INTERVAL_WEIGHTS
            ),
            "scalar_endpoint_VJP_role": "separate_first_order_context_only",
            "held_unit_endpoint_VJP_role": (
                "separate_off_path_gradient_reference_and_context_measurement"
            ),
            "primary_closure_target": (
                "finite_KL_joint_minus_KL_scalar_compared_with_fixed_GL4_"
                "gradient_contraction_on_the_casted_scalar_to_joint_path"
            ),
            "casted_path_is_a_dtype_staircase_not_an_exact_continuous_curve": True,
            "exact_continuous_FTC_claimed": False,
            "aggregation": "equal_token_then_equal_prompt_then_equal_family",
            "teacher_KL_primary": True,
            "preregistered_GL4_closure_thresholds": {
                "overall_family_prompt_token_closure_relative_RMSE_maximum": (
                    _OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "every_family_closure_relative_RMSE_maximum": (
                    _FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "overall_closure_cosine_minimum": _CLOSURE_COSINE_MINIMUM,
            },
            "v9_fit_performed": False,
            "arm_selection_performed": False,
            "step_or_damping_grid_searched": False,
            "fallback_or_per_family_routing_allowed": False,
            "posthoc_hyperparameter_search_performed": False,
            "result_can_authorize_serving_or_model_mutation": False,
        },
        "v8_control_binding": {
            "file": str(v8_report_path),
            "file_sha256": V8_REPORT_FILE_SHA256,
            "report_sha256": V8_REPORT_SHA256,
            "schema": v8_report.get("schema"),
            "classification": v8_report.get("classification"),
            "passed": v8_report.get("passed"),
            "live_lineage_reproduction": v8_binding,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": (
                token_v1.MATERIALIZATION_REPORT_FILE_SHA256
            ),
            "materialization_report_sha256": token_v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": token_v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": token_v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "prompt_role_receipt": {
            "artifact_sha256": (
                analytic.v7_phases.v6_phases.roles.artifact_sha256
            ),
            "analytic_fit_example_ids": (
                analytic.v7_phases.v6_phases.roles.fit_example_ids
            ),
            "held_path_example_ids": tuple(
                trace.example_id
                for trace in sorted(traces, key=lambda value: value.example_id)
            ),
            "every_prompt_is_outer_held_for_its_own_family_fold": True,
            "no_prompt_used_to_refit_or_select_an_arm_in_v9": True,
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": v5diag._endpoint_prompt_receipts(traces),
        "full_seven_codec_binding": analytic.codec_binding,
        "endpoint_pair_receipts": live.endpoint_receipts,
        "path_node_receipts": live.node_receipts,
        "path_observation_set_sha256": live.observation_set_sha256,
        "family_equal_scalar_joint_path_attribution": attribution_metadata,
        "pinned_v8_finite_delta_replay": finite_anchor,
        "additive_scalar_joint_attribution_ledger": additive_ledger,
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "quadrature_closure_gate_results": tuple(sorted(closure_gates.items())),
        "outcome_matrix": {
            "outcome": classification,
            "integrity_completed": integrity_passed,
            "quadrature_closure_established": closure_established,
            "contributor_attribution_is_descriptive_and_may_be_mixed": True,
            "fresh_confirmation_authorized_next": False,
            "selection_or_serving_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_attribution_only": True,
            "all_parent_outcomes_previously_inspected": True,
            "native_teacher_logits_fresh_for_every_prompt": True,
            "outer_held_family_excluded_from_refit_codec_scalar_and_joint_fit": True,
            "finite_V8_failure_is_the_pinned_displacement_authority": True,
            "GL4_closure_thresholds_preregistered_before_live_execution": True,
            "GL4_closure_thresholds_selected_from_current_results": False,
            "V7_unit_derivative_is_nested_LOFO_context_not_path_authority": True,
            "held_unit_actual_displacement_tangent_is_fresh_same_prompt_context": True,
            "single_location_or_curvature_cause_forced": False,
            "held_rows_used_to_choose_a_correction": False,
            "fresh_family_disjoint_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "model_mutation_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
        },
        "safety": _safety_metadata(),
    }
    return _publish(report, output=destination)


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


def build_parser() -> argparse.ArgumentParser:
    """Return the deliberately no-knob V9 command-line interface."""

    return argparse.ArgumentParser(
        description=(
            "Run the pinned V9 scalar-to-joint fixed-GL4 H4 path attribution."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_scalar_joint_path_attribution_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
