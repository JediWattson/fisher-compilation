"""Fit-only X4/H4 factorial attribution on the frozen Gemma bridge.

This diagnostic crosses the base graph bridge versus its accepted X4 head
with three H4 states: no H4 head, the matched lag-only ``B`` head, and the
frozen independent-state head.  A direct factorized-model pass is the source
authority for every example.  The collector therefore performs exactly seven
forwards per example and immediately reduces logits and X4/H4 boundary
activations to scalar/hash-only observations.

The command boundary can open only the reusable expanded Calibration-A fit
role and authenticated frozen artifacts.  It has no assessment-role input or
search-knob capability, and its report is explicitly development-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol

import torch
from torch import Tensor

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_experiment import (
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
    load_gemma3_graph_organized_svd_candidate,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
    derive_gemma3_l3_l4_supervised_boundary,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .gemma3_l3_l4_h4_damping_materialization import (
    load_gemma_h4_damping_materialization,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
    measure_gemma_h4_damping_finite_nll_observation,
)
from .gemma3_l3_l4_h4_incremental_signal_diagnostic import (
    _accepted_x4_artifact,
    _canonical_json_bytes,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    _file_sha256,
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    gemma3_l3_l4_progressive_a_tokenizer_contract_sha256,
    load_gemma3_l3_l4_progressive_a_fit_role,
)
from .gemma3_l3_l4_progressive_worker import GemmaProgressivePanel
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaL3L4TwoHeadArtifact,
    _tensor_sha256,
)
from .gemma3_l3_l4_x4_h4_factorial_analysis import (
    GemmaX4H4BoundarySufficientStats,
    GemmaX4H4FactorialBoundaryObservation,
    build_gemma_x4_h4_factorial_report,
    validate_gemma_x4_h4_factorial_report,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)
from .shadow_fidelity import ShadowFidelityExample


__all__ = [
    "FACTORIAL_ARM_IDS",
    "GemmaX4H4FactorialLiveCollection",
    "build_parser",
    "collect_gemma_x4_h4_factorial_live",
    "main",
    "publish_gemma_x4_h4_factorial_report",
    "run_gemma_x4_h4_factorial_diagnostic",
]


BASE_NONE_ARM = "base_none"
BASE_LAG_B_ARM = "base_lag_b"
BASE_INDEPENDENT_STATE_ARM = "base_independent_state"
ACCEPTED_NONE_ARM = "accepted_none"
ACCEPTED_LAG_B_ARM = "accepted_lag_b"
ACCEPTED_INDEPENDENT_STATE_ARM = "accepted_independent_state"
FACTORIAL_ARM_IDS = (
    BASE_NONE_ARM,
    BASE_LAG_B_ARM,
    BASE_INDEPENDENT_STATE_ARM,
    ACCEPTED_NONE_ARM,
    ACCEPTED_LAG_B_ARM,
    ACCEPTED_INDEPENDENT_STATE_ARM,
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_x4_h4_factorial_diagnostic"
_FORMAT_VERSION = 1
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-x4-h4-factorial-collection:v1\0"
)
_BOUNDARY_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-x4-h4-factorial-boundary:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FACTORIZED_SCOPE = "factorized_refit"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_DEFAULT_EXPANDED_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
_DEFAULT_EXPANDED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
_DEFAULT_MATERIALIZATION_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-materialization-v1.report.json"
)
_DEFAULT_ACCEPTED_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-projected-state-v6.campaign.json"
)
_DEFAULT_ACCEPTED_CANDIDATE = (
    _LOCAL_ROOT / "progressive-a-h4-projected-state-v6.campaign.candidate.pt"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-x4-h4-factorial-fit-v1.report.json"
)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _assert_scalar_hash_only(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} contains a tensor")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_scalar_hash_only(nested, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_scalar_hash_only(nested, path=f"{path}[{index}]")


class _ArtifactLike(Protocol):
    artifact_sha256: str
    execution_sha256: str
    runtime_binding_sha256: str
    parent_artifact_sha256: str
    bridge_binding_sha256: str
    live_model_sha256: str
    adapter_execution_sha256: str
    prepared_float_scalar_count: int
    logical_macs_per_token_upper_bound: int

    def validate_integrity(self) -> None: ...

    def head(self, site: str) -> object | None: ...


class _AdapterLike(Protocol):
    def model_fingerprint(self) -> str: ...

    def execution_fingerprint(self) -> str: ...

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
    ) -> object: ...


class _BridgeLike(Protocol):
    bridge_binding_sha256: str
    prepared_float_scalar_count: int
    logical_macs_per_token_upper_bound: int

    def validate_integrity(self) -> None: ...

    def execute(
        self,
        adapter: _AdapterLike,
        model_inputs: Mapping[str, Tensor],
        *,
        x4_head: object | None,
        h4_head: object | None,
    ) -> object: ...


def _head_sha256(head: object | None, *, label: str) -> str:
    if head is None:
        raise ValueError(f"{label} head is missing")
    return _require_sha256(
        getattr(head, "artifact_sha256", None),
        label=f"{label} head",
    )


def _validate_factorial_artifacts(
    *,
    panel: GemmaProgressivePanel,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    accepted_x4: _ArtifactLike,
    matched_alpha0: _ArtifactLike,
    challenger_alpha0_5: _ArtifactLike,
) -> dict[str, tuple[object | None, object | None]]:
    bridge.validate_integrity()
    artifacts = (accepted_x4, matched_alpha0, challenger_alpha0_5)
    for artifact in artifacts:
        artifact.validate_integrity()
    if len({artifact.artifact_sha256 for artifact in artifacts}) != 3:
        raise ValueError("factorial artifacts must be distinct")
    model_sha256 = adapter.model_fingerprint()
    execution_sha256 = adapter.execution_fingerprint()
    if any(
        artifact.bridge_binding_sha256 != bridge.bridge_binding_sha256
        or artifact.live_model_sha256 != model_sha256
        or artifact.adapter_execution_sha256 != execution_sha256
        for artifact in artifacts
    ):
        raise ValueError(
            "factorial artifacts differ from the live factorized runtime"
        )
    if (
        matched_alpha0.parent_artifact_sha256
        != accepted_x4.artifact_sha256
        or challenger_alpha0_5.parent_artifact_sha256
        != accepted_x4.artifact_sha256
    ):
        raise ValueError("materialized H4 arms do not descend from accepted X4")
    accepted_head = accepted_x4.head(_X4_SITE)
    alpha0_x4 = matched_alpha0.head(_X4_SITE)
    independent_x4 = challenger_alpha0_5.head(_X4_SITE)
    accepted_sha256 = _head_sha256(accepted_head, label="accepted X4")
    if (
        _head_sha256(alpha0_x4, label="alpha0 X4") != accepted_sha256
        or _head_sha256(independent_x4, label="independent X4")
        != accepted_sha256
        or accepted_x4.head(_H4_SITE) is not None
    ):
        raise ValueError("factorial arms do not share exact accepted X4")
    lag_b = matched_alpha0.head(_H4_SITE)
    independent = challenger_alpha0_5.head(_H4_SITE)
    _head_sha256(lag_b, label="lag-only B H4")
    _head_sha256(independent, label="independent-state H4")
    if (
        getattr(lag_b, "conditioning", None) != "l3_source_modes"
        or getattr(independent, "conditioning", None)
        != "l3_source_modes_plus_independent_realized_h4_modes_v1"
        or getattr(lag_b, "fit_manifest_sha256", panel.manifest_sha256)
        != panel.manifest_sha256
        or getattr(independent, "fit_manifest_sha256", panel.manifest_sha256)
        != panel.manifest_sha256
    ):
        raise ValueError("factorial H4 semantics or fit binding differ")
    return {
        BASE_NONE_ARM: (None, None),
        BASE_LAG_B_ARM: (None, lag_b),
        BASE_INDEPENDENT_STATE_ARM: (None, independent),
        ACCEPTED_NONE_ARM: (accepted_head, None),
        ACCEPTED_LAG_B_ARM: (accepted_head, lag_b),
        ACCEPTED_INDEPENDENT_STATE_ARM: (accepted_head, independent),
    }


def _gather_logits(logits: Tensor, indices: Tensor) -> Tensor:
    if (
        not isinstance(logits, Tensor)
        or not logits.is_floating_point()
        or logits.ndim != 3
        or logits.shape[0] != 1
    ):
        raise ValueError("live logits must have shape [1, sequence, vocab]")
    return (
        logits[0]
        .index_select(0, indices.to(logits.device))
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(
        torch.equal(
            left.detach().to(device="cpu"),
            right.detach().to(device="cpu"),
        )
    )


def _boundary_stats(
    source: Tensor,
    candidate: Tensor,
    *,
    selected: Tensor,
    label: str,
) -> GemmaX4H4BoundarySufficientStats:
    if (
        not isinstance(source, Tensor)
        or not isinstance(candidate, Tensor)
        or source.shape != candidate.shape
        or source.ndim != 3
        or not source.is_floating_point()
        or not candidate.is_floating_point()
        or not isinstance(selected, Tensor)
        or selected.dtype != torch.bool
        or selected.shape != source.shape[:2]
        or not bool(selected.any())
    ):
        raise ValueError(f"{label} boundary geometry differs")
    source_rows = (
        source[selected.to(source.device)]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    candidate_rows = (
        candidate[selected.to(candidate.device)]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    if (
        not bool(torch.isfinite(source_rows).all())
        or not bool(torch.isfinite(candidate_rows).all())
    ):
        raise ValueError(f"{label} boundary rows must be finite")
    error = candidate_rows - source_rows
    return GemmaX4H4BoundarySufficientStats(
        scalar_count=int(error.numel()),
        squared_error_sum=float(error.square().sum()),
        source_squared_sum=float(source_rows.square().sum()),
        candidate_squared_sum=float(candidate_rows.square().sum()),
        source_candidate_dot=float((source_rows * candidate_rows).sum()),
        max_absolute_error=float(error.abs().max()),
    )


@dataclass(frozen=True, slots=True)
class GemmaX4H4FactorialLiveCollection:
    """The 96 scalar output observations and 96 boundary observations."""

    observations: Mapping[
        str, tuple[GemmaH4DampingFiniteNLLObservation, ...]
    ]
    boundary_observations: tuple[
        GemmaX4H4FactorialBoundaryObservation, ...
    ]
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            tuple(self.observations) != FACTORIAL_ARM_IDS
            or any(
                type(rows) is not tuple
                or len(rows) != 16
                or any(
                    not isinstance(
                        row,
                        GemmaH4DampingFiniteNLLObservation,
                    )
                    for row in rows
                )
                for rows in self.observations.values()
            )
            or len(self.boundary_observations) != 96
            or any(
                not isinstance(
                    row,
                    GemmaX4H4FactorialBoundaryObservation,
                )
                for row in self.boundary_observations
            )
        ):
            raise ValueError("factorial collection geometry differs")
        _assert_scalar_hash_only(self.audit, path="collection audit")


def collect_gemma_x4_h4_factorial_live(
    *,
    panel: GemmaProgressivePanel,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    accepted_x4_artifact: _ArtifactLike,
    matched_alpha0_artifact: _ArtifactLike,
    challenger_alpha0_5_artifact: _ArtifactLike,
) -> GemmaX4H4FactorialLiveCollection:
    """Stream one direct source plus the exact six-cell factorial per prompt."""

    if not isinstance(panel, GemmaProgressivePanel):
        raise TypeError("panel must be a strict GemmaProgressivePanel")
    if (
        panel.role != "calibration_a_fit"
        or len(panel.examples) != 16
        or len(panel.family_ids) != 8
        or any(
            sum(
                example.family_id == family_id
                for example in panel.examples
            )
            != 2
            for family_id in panel.family_ids
        )
    ):
        raise ValueError("factorial requires the reusable 16-by-8 A-fit panel")
    heads = _validate_factorial_artifacts(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        accepted_x4=accepted_x4_artifact,
        matched_alpha0=matched_alpha0_artifact,
        challenger_alpha0_5=challenger_alpha0_5_artifact,
    )
    source_model_sha256 = adapter.model_fingerprint()
    source_execution_sha256 = adapter.execution_fingerprint()
    observations: dict[
        str, list[GemmaH4DampingFiniteNLLObservation]
    ] = {arm_id: [] for arm_id in FACTORIAL_ARM_IDS}
    boundary_observations: list[
        GemmaX4H4FactorialBoundaryObservation
    ] = []
    result_sha256s: dict[str, list[str]] = {
        arm_id: [] for arm_id in FACTORIAL_ARM_IDS
    }
    boundary_receipts: list[str] = []
    model_input_sha256s: list[str] = []

    for example in panel.examples:
        example.validate_integrity()
        model_inputs = example.batch.model_inputs
        with torch.no_grad():
            native = adapter.forward(
                model_inputs,
                capture_sites=(_X4_SITE, _H4_SITE),
            )
        sequence = getattr(native, "sequence", None)
        valid_mask = getattr(sequence, "query_valid_mask", None)
        logical_positions = getattr(sequence, "logical_positions", None)
        activations = getattr(native, "activations", None)
        if not isinstance(activations, Mapping):
            raise ValueError("direct source omitted captured boundaries")
        native_x4 = activations.get(_X4_SITE)
        native_h4 = activations.get(_H4_SITE)
        input_ids = model_inputs.get("input_ids")
        if (
            not isinstance(input_ids, Tensor)
            or not isinstance(valid_mask, Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != input_ids.shape
            or not isinstance(logical_positions, Tensor)
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or logical_positions.shape != valid_mask.shape
            or not isinstance(native_x4, Tensor)
            or not isinstance(native_h4, Tensor)
            or native_x4.shape != native_h4.shape
            or native_x4.shape[:2] != input_ids.shape
        ):
            raise ValueError("direct source authority grid differs")
        indices, targets = derive_gemma3_l3_l4_supervised_boundary(
            input_ids,
            valid_mask,
        )
        expected_targets = torch.full_like(example.batch.targets, -100)
        expected_targets[0, indices.to(expected_targets.device)] = targets.to(
            expected_targets.device
        )
        if (
            not torch.equal(
                example.batch.valid_positions.to(valid_mask.device),
                valid_mask,
            )
            or not torch.equal(example.batch.targets, expected_targets)
        ):
            raise ValueError(
                "factorial calibration targets differ from causal boundary"
            )
        source_logits = _gather_logits(
            getattr(native, "logits", None),
            indices,
        )
        del native, sequence, activations
        x4_hash_by_factor: dict[str, str] = {}
        reference_x4_sha256: str | None = None

        for arm_id in FACTORIAL_ARM_IDS:
            requested_x4, requested_h4 = heads[arm_id]
            with torch.no_grad():
                execution = bridge.execute(
                    adapter,
                    model_inputs,
                    x4_head=requested_x4,
                    h4_head=requested_h4,
                )
            validate_execution = getattr(
                execution,
                "validate_integrity",
                None,
            )
            if not callable(validate_execution):
                raise TypeError(f"{arm_id} execution lacks validation")
            validate_execution()
            prefix = getattr(execution, "prefix", None)
            arm_valid = getattr(prefix, "valid_target_mask", None)
            arm_positions = getattr(prefix, "logical_positions", None)
            active = getattr(prefix, "target_affected_mask", None)
            expected_x4_sha256 = (
                None
                if requested_x4 is None
                else _head_sha256(
                    requested_x4,
                    label=f"{arm_id} requested X4",
                )
            )
            expected_h4_sha256 = (
                None
                if requested_h4 is None
                else _head_sha256(
                    requested_h4,
                    label=f"{arm_id} requested H4",
                )
            )
            result_sha256 = _require_sha256(
                getattr(execution, "artifact_sha256", None),
                label=f"{arm_id} execution result",
            )
            if (
                getattr(execution, "model_forward_count", None) != 1
                or getattr(execution, "model_inputs_sha256", None)
                != example.model_inputs_sha256
                or getattr(execution, "bridge_binding_sha256", None)
                != bridge.bridge_binding_sha256
                or getattr(execution, "x4_head_sha256", None)
                != expected_x4_sha256
                or getattr(execution, "h4_head_sha256", None)
                != expected_h4_sha256
            ):
                raise ValueError(f"{arm_id} execution identity differs")
            if (
                not isinstance(arm_valid, Tensor)
                or not isinstance(arm_positions, Tensor)
                or not isinstance(active, Tensor)
                or active.dtype != torch.bool
                or active.shape != valid_mask.shape
                or not torch.equal(arm_valid, valid_mask)
                or not torch.equal(arm_positions, logical_positions)
            ):
                raise ValueError(f"{arm_id} execution grid differs")
            reference_x4 = getattr(execution, "reference_x4", None)
            candidate_x4 = getattr(execution, "candidate_x4", None)
            candidate_h4 = getattr(execution, "candidate_h4", None)
            if (
                not isinstance(reference_x4, Tensor)
                or reference_x4.shape != native_x4.shape
                or not reference_x4.is_floating_point()
                or not isinstance(candidate_x4, Tensor)
                or not isinstance(candidate_h4, Tensor)
                or candidate_x4.shape != native_x4.shape
                or candidate_h4.shape != native_h4.shape
                or not candidate_x4.is_floating_point()
                or not candidate_h4.is_floating_point()
            ):
                raise ValueError(f"{arm_id} execution omitted boundaries")
            observed_reference_x4_sha256 = _tensor_sha256(reference_x4)
            if reference_x4_sha256 is None:
                reference_x4_sha256 = observed_reference_x4_sha256
            elif reference_x4_sha256 != observed_reference_x4_sha256:
                raise ValueError(
                    f"{arm_id} bridge reference X4 changed across arms"
                )
            x4_selected = (
                candidate_x4[active.to(candidate_x4.device)]
                .detach()
                .to(device="cpu")
                .contiguous()
            )
            x4_receipt = _tensor_sha256(x4_selected)
            x4_factor = (
                "base" if requested_x4 is None else "accepted"
            )
            prior_x4_receipt = x4_hash_by_factor.setdefault(
                x4_factor,
                x4_receipt,
            )
            if prior_x4_receipt != x4_receipt:
                raise ValueError(
                    f"{x4_factor} candidate X4 changed across H4 levels"
                )
            x4_stats = _boundary_stats(
                native_x4,
                candidate_x4,
                selected=active,
                label=f"{arm_id} X4",
            )
            h4_stats = _boundary_stats(
                native_h4,
                candidate_h4,
                selected=active,
                label=f"{arm_id} H4",
            )
            boundary_observations.append(
                GemmaX4H4FactorialBoundaryObservation(
                    arm_id=arm_id,
                    example_id=example.example_id,
                    family_id=example.family_id,
                    x4=x4_stats,
                    h4=h4_stats,
                )
            )
            candidate_logits = _gather_logits(
                getattr(execution, "logits", None),
                indices,
            )
            transient = ShadowFidelityExample(
                example_id=example.example_id,
                family_id=example.family_id,
                source_logits=source_logits,
                candidate_logits=candidate_logits,
                targets=targets,
            )
            measured = measure_gemma_h4_damping_finite_nll_observation(
                transient
            )
            observations[arm_id].append(measured)
            result_sha256s[arm_id].append(result_sha256)
            del (
                execution,
                validate_execution,
                prefix,
                arm_valid,
                arm_positions,
                active,
                reference_x4,
                candidate_x4,
                candidate_h4,
                x4_selected,
                x4_stats,
                h4_stats,
                candidate_logits,
                transient,
                measured,
            )

        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            != example.model_inputs_sha256
        ):
            raise RuntimeError("factorial model inputs changed during forwards")
        boundary_receipts.append(
            _sha256(
                _BOUNDARY_DOMAIN,
                {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "model_inputs_sha256": example.model_inputs_sha256,
                    "indices_sha256": _tensor_sha256(indices),
                    "targets_sha256": _tensor_sha256(targets),
                    "valid_mask_sha256": _tensor_sha256(valid_mask),
                    "logical_positions_sha256": _tensor_sha256(
                        logical_positions
                    ),
                    "native_x4_sha256": _tensor_sha256(native_x4),
                    "native_h4_sha256": _tensor_sha256(native_h4),
                    "bridge_reference_x4_sha256": reference_x4_sha256,
                },
            )
        )
        model_input_sha256s.append(example.model_inputs_sha256)
        del (
            source_logits,
            native_x4,
            native_h4,
            indices,
            targets,
            valid_mask,
            logical_positions,
            x4_hash_by_factor,
        )

    if (
        adapter.model_fingerprint() != source_model_sha256
        or adapter.execution_fingerprint() != source_execution_sha256
    ):
        raise RuntimeError("factorized source changed during factorial")
    frozen_observations = {
        arm_id: tuple(observations[arm_id])
        for arm_id in FACTORIAL_ARM_IDS
    }
    audit: dict[str, object] = {
        "execution_mode": "fit_only_matched_independent_prefill_forwards",
        "example_count": 16,
        "family_count": 8,
        "factorial_arm_ids": FACTORIAL_ARM_IDS,
        "model_forward_count_per_example": 7,
        "total_model_forward_count": 112,
        "native_source_forward_count_per_example": 1,
        "candidate_forward_count_per_arm_per_example": 1,
        "candidate_observation_count": 96,
        "boundary_observation_count": 96,
        "native_source_reused_within_example": True,
        "native_source_retained_across_examples": False,
        "candidate_execution_released_before_next_arm": True,
        "bridge_reference_x4_semantics": (
            "x4_after_compiled_y3_clamp_before_base_delta"
        ),
        "candidate_x4_scored_against_direct_native_x4": True,
        "raw_logits_retained": False,
        "raw_token_ids_retained": False,
        "raw_activations_retained": False,
        "source_model_sha256": source_model_sha256,
        "source_execution_sha256": source_execution_sha256,
        "bridge_binding_sha256": bridge.bridge_binding_sha256,
        "example_receipt_sha256s": tuple(sorted(model_input_sha256s)),
        "boundary_receipt_sha256s": tuple(sorted(boundary_receipts)),
        "arm_result_sha256s": {
            arm_id: tuple(result_sha256s[arm_id])
            for arm_id in FACTORIAL_ARM_IDS
        },
    }
    audit["collection_sha256"] = _sha256(_COLLECTION_DOMAIN, audit)
    return GemmaX4H4FactorialLiveCollection(
        observations=frozen_observations,
        boundary_observations=tuple(boundary_observations),
        audit=audit,
    )


def publish_gemma_x4_h4_factorial_report(
    path: Path | str,
    report: Mapping[str, object],
) -> None:
    """Atomically publish JSON without replacing any existing destination."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite factorial report")
    _assert_scalar_hash_only(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        with stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, destination)
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        stage.unlink(missing_ok=True)


def _source_code_sha256s() -> Mapping[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "gemma3_l3_l4_x4_h4_factorial_diagnostic.py",
        "gemma3_l3_l4_x4_h4_factorial_analysis.py",
        "gemma3_l3_l4_h4_damping_materialization.py",
        "gemma3_l3_l4_h4_damping_selection_runtime.py",
        "gemma3_l3_l4_graph_organized_svd_shadow_runtime.py",
    )
    return {name: _file_sha256(package / name) for name in names}


def run_gemma_x4_h4_factorial_diagnostic(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    materialization_report_path: Path | str = (
        _DEFAULT_MATERIALIZATION_REPORT
    ),
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    accepted_x4_report_path: Path | str = _DEFAULT_ACCEPTED_REPORT,
    accepted_x4_candidate_path: Path | str = _DEFAULT_ACCEPTED_CANDIDATE,
    expected_accepted_x4_candidate_file_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Execute the frozen factorial on reusable expanded-fit examples only."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite factorial report")
    materialization, materialization_report = (
        load_gemma_h4_damping_materialization(
            materialization_report_path,
            expected_report_sha256=(
                expected_materialization_report_sha256
            ),
            expected_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
        )
    )
    recollection = _mapping(
        materialization_report.get("recollection"),
        label="materialization recollection",
    )
    accepted_report, accepted_x4, accepted_provenance = (
        _accepted_x4_artifact(
            report_path=accepted_x4_report_path,
            candidate_path=accepted_x4_candidate_path,
            expected_candidate_file_sha256=(
                expected_accepted_x4_candidate_file_sha256
            ),
        )
    )
    del accepted_report
    if _canonical_json_bytes(accepted_provenance) != _canonical_json_bytes(
        _mapping(
            recollection.get("accepted_x4_provenance"),
            label="materialization accepted X4 provenance",
        )
    ):
        raise ValueError("accepted X4 provenance differs from materialization")
    if (
        materialization.alpha0_artifact.parent_artifact_sha256
        != accepted_x4.artifact_sha256
        or materialization.alpha0_5_artifact.parent_artifact_sha256
        != accepted_x4.artifact_sha256
    ):
        raise ValueError("materialized H4 artifacts have another parent")

    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    metadata = protocol.metadata()
    tokenizer_contract = dict(
        _mapping(metadata["tokenizer"], label="frozen tokenizer")
    )
    corpus, fit_input = load_gemma3_l3_l4_progressive_a_fit_role(
        corpus_artifact_path,
        fit_input_path=fit_input_path,
        expected_artifact_sha256=str(
            recollection["corpus_artifact_sha256"]
        ),
        tokenizer_contract=tokenizer_contract,
    )
    tokenizer, live_tokenizer_contract = (
        _load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    if (
        _canonical_json_bytes(live_tokenizer_contract)
        != _canonical_json_bytes(tokenizer_contract)
        or gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
            tokenizer_contract
        )
        != corpus.tokenizer_contract_sha256
    ):
        raise ValueError("live tokenizer differs from expanded-fit contract")
    fit_panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=fit_input,
        view=corpus.role_view("calibration_a_fit"),
        max_length=int(tokenizer_contract["max_length"]),
        device=torch.device(str(tokenizer_contract["device"])),
        forbidden_manifest_sha256s=(
            corpus.forbidden_assessment_manifest_sha256s
        ),
    )
    if (
        fit_panel.manifest_sha256 != recollection["fit_manifest_sha256"]
        or fit_panel.binding_sha256 != recollection["fit_binding_sha256"]
        or len(fit_panel.examples) != 16
        or len(fit_panel.family_ids) != 8
    ):
        raise ValueError("expanded fit panel differs from materialization")

    model_metadata = _mapping(metadata["model"], label="frozen model")
    graph_binding = _mapping(
        metadata["graph_candidate"],
        label="frozen graph candidate",
    )
    basis_binding = _mapping(
        metadata["prompt_blind_basis"],
        label="frozen basis",
    )
    materialized_files = _mapping(
        materialization_report.get("files"),
        label="materialization files",
    )
    immutable_paths = {
        "corpus_artifact": Path(corpus_artifact_path),
        "fit_input": Path(fit_input_path),
        "materialization_report": Path(materialization_report_path),
        "matched_alpha0_candidate": Path(
            str(
                _mapping(
                    materialized_files["matched_alpha0"],
                    label="matched alpha0 file",
                )["tensor_file"]
            )
        ),
        "challenger_alpha0_5_candidate": Path(
            str(
                _mapping(
                    materialized_files["challenger_alpha0_5"],
                    label="challenger file",
                )["tensor_file"]
            )
        ),
        "accepted_x4_report": Path(accepted_x4_report_path),
        "accepted_x4_candidate": Path(accepted_x4_candidate_path),
        "graph_candidate": Path(graph_candidate_path),
        "basis_package": Path(basis_package_path),
        "base_artifact": Path(base_artifact_path),
        "refit_artifact": Path(refit_artifact_path),
    }
    immutable_before = {
        name: _file_sha256(path) for name, path in immutable_paths.items()
    }
    immutable_expected = {
        "corpus_artifact": _file_sha256(corpus_artifact_path),
        "fit_input": fit_input.source_file_sha256,
        "materialization_report": (
            expected_materialization_report_file_sha256
        ),
        "matched_alpha0_candidate": str(
            _mapping(
                materialized_files["matched_alpha0"],
                label="matched alpha0 file",
            )["tensor_file_sha256"]
        ),
        "challenger_alpha0_5_candidate": str(
            _mapping(
                materialized_files["challenger_alpha0_5"],
                label="challenger file",
            )["tensor_file_sha256"]
        ),
        "accepted_x4_report": str(
            accepted_provenance["report_file_sha256"]
        ),
        "accepted_x4_candidate": (
            expected_accepted_x4_candidate_file_sha256
        ),
        "graph_candidate": str(graph_binding["tensor_file_sha256"]),
        "basis_package": str(basis_binding["tensor_file_sha256"]),
        "base_artifact": str(recollection["base_artifact_file_sha256"]),
        "refit_artifact": str(
            recollection["refit_artifact_file_sha256"]
        ),
    }
    if immutable_before != immutable_expected:
        raise ValueError("factorial immutable input binding differs")
    code_before = _source_code_sha256s()

    device = resolve_torch_device("cpu")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != model_metadata["source_model_sha256"]:
        raise ValueError("live raw Gemma differs from frozen source")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256 = adapter.model_fingerprint()
        factorized_execution_sha256 = adapter.execution_fingerprint()
        if (
            factorized_model_sha256
            != graph_binding["factorized_live_execution_sha256"]
            or factorized_execution_sha256
            != graph_binding["factorized_refit_execution_sha256"]
            or factorized_model_sha256
            != recollection["factorized_model_sha256"]
            or factorized_execution_sha256
            != recollection["factorized_execution_sha256"]
        ):
            raise ValueError("live factorized Gemma differs")
        graph_candidate = load_gemma3_graph_organized_svd_candidate(
            graph_candidate_path,
            expected_file_sha256=str(graph_binding["tensor_file_sha256"]),
        )
        basis = load_gemma3_l3_l4_basis_package(
            basis_package_path,
            expected_file_sha256=str(basis_binding["tensor_file_sha256"]),
            expected_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
        )
        runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            graph_candidate,
            basis,
            expected_candidate_artifact_sha256=str(
                graph_binding["logical_artifact_sha256"]
            ),
            expected_basis_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
            expected_plan_artifact_sha256=str(
                graph_binding["deployment_plan_sha256"]
            ),
            expected_live_model_sha256=str(
                graph_binding["factorized_live_execution_sha256"]
            ),
            expected_adapter_execution_sha256=str(
                graph_binding["factorized_refit_execution_sha256"]
            ),
            analysis_device="cpu",
        )
        if (
            runtime.runtime_binding_sha256
            != recollection["progressive_runtime_binding_sha256"]
        ):
            raise ValueError("progressive runtime binding differs")
        bridge = runtime.export_one_pass_bridge()
        collection = collect_gemma_x4_h4_factorial_live(
            panel=fit_panel,
            adapter=adapter,
            bridge=bridge,
            accepted_x4_artifact=accepted_x4,
            matched_alpha0_artifact=materialization.alpha0_artifact,
            challenger_alpha0_5_artifact=materialization.alpha0_5_artifact,
        )
        report = build_gemma_x4_h4_factorial_report(
            observations=collection.observations,
            boundary_observations=collection.boundary_observations,
            manifest={
                example.example_id: example.family_id
                for example in fit_panel.examples
            },
            lineage={
                "corpus_artifact_sha256": corpus.artifact_sha256,
                "fit_input_file_sha256": fit_input.source_file_sha256,
                "fit_manifest_sha256": fit_panel.manifest_sha256,
                "fit_binding_sha256": fit_panel.binding_sha256,
                "materialization_report_sha256": (
                    materialization_report["report_sha256"]
                ),
                "materialization_report_file_sha256": immutable_before[
                    "materialization_report"
                ],
                **{
                    f"accepted_x4_{name}": str(value)
                    for name, value in accepted_provenance.items()
                    if isinstance(value, str)
                    and _SHA256.fullmatch(value) is not None
                },
            },
            execution=collection.audit,
            resources={
                "bridge_prepared_float_scalar_count": (
                    bridge.prepared_float_scalar_count
                ),
                "bridge_logical_macs_per_token_upper_bound": (
                    bridge.logical_macs_per_token_upper_bound
                ),
                "accepted_x4_head_prepared_float_scalar_count": (
                    int(
                        getattr(
                            accepted_x4.head(_X4_SITE),
                            "prepared_float_scalar_count",
                        )
                    )
                ),
                "accepted_x4_head_logical_macs_per_token_upper_bound": (
                    int(
                        getattr(
                            accepted_x4.head(_X4_SITE),
                            "logical_macs_per_token_upper_bound",
                        )
                    )
                ),
                "lag_b_h4_prepared_float_scalar_count": int(
                    getattr(
                        materialization.alpha0_artifact.head(_H4_SITE),
                        "prepared_float_scalar_count",
                    )
                ),
                "lag_b_h4_logical_macs_per_token_upper_bound": int(
                    getattr(
                        materialization.alpha0_artifact.head(_H4_SITE),
                        "logical_macs_per_token_upper_bound",
                    )
                ),
                "independent_h4_prepared_float_scalar_count": int(
                    getattr(
                        materialization.alpha0_5_artifact.head(_H4_SITE),
                        "prepared_float_scalar_count",
                    )
                ),
                "independent_h4_logical_macs_per_token_upper_bound": int(
                    getattr(
                        materialization.alpha0_5_artifact.head(_H4_SITE),
                        "logical_macs_per_token_upper_bound",
                    )
                ),
            },
            safety=None,
        )
        validate_gemma_x4_h4_factorial_report(report)
        if (
            {
                name: _file_sha256(path)
                for name, path in immutable_paths.items()
            }
            != immutable_before
            or _source_code_sha256s() != code_before
            or adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
        ):
            raise RuntimeError(
                "factorial immutable input, code, or runtime changed"
            )
        publish_gemma_x4_h4_factorial_report(destination, report)
        return report
    finally:
        switcher.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen X4/H4 factorial on reusable expanded A-fit data."
        )
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=_DEFAULT_EXPANDED_CORPUS,
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=_DEFAULT_EXPANDED_FIT_INPUT,
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=_DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--materialization-report-sha256", required=True)
    parser.add_argument(
        "--materialization-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--accepted-x4-report",
        type=Path,
        default=_DEFAULT_ACCEPTED_REPORT,
    )
    parser.add_argument(
        "--accepted-x4-candidate",
        type=Path,
        default=_DEFAULT_ACCEPTED_CANDIDATE,
    )
    parser.add_argument(
        "--accepted-x4-candidate-sha256",
        required=True,
    )
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma_x4_h4_factorial_diagnostic(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        materialization_report_path=args.materialization_report,
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        accepted_x4_report_path=args.accepted_x4_report,
        accepted_x4_candidate_path=args.accepted_x4_candidate,
        expected_accepted_x4_candidate_file_sha256=(
            args.accepted_x4_candidate_sha256
        ),
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
