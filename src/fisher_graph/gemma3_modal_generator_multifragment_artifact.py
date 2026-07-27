"""Strict source-safe artifacts for multi-fragment Gemma modal graphs.

This module is deliberately independent of the live experiment runner.  The
runner supplies already-built, authenticated compiler objects plus small plain
metadata mappings; this module owns their canonical serialization boundary.

The development protocol represented here has three distinct data roles:

* the fit split authenticates the prompt trace, grouped Fisher, and clusters;
* an open-development selection partition fits node curves and graph edges;
* a disjoint open-development assessment partition measures the frozen graph.

No calibration-B, guard, validation, or test split is represented or opened.
The outer digest detects accidental mutation.  It is not a signature, and the
numerical extraction and split membership remain caller-declared and
self-attested.  Nested compiler artifacts retain their own tensor hashes and
lineage checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import torch

from .computational_modes import ComputationalModeRateCurve
from .fisher_prompt_clustering import FisherPromptClusterPlan
from .modal_compiler_pipeline import ModalCompilerPipeline
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_generators import ModalGeneratorRateCurve
from .modal_interaction_fitting import ModalInteractionSelection
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragmentPlan,
)
from .parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPParameterGroupCatalog,
)
from .prompt_mode_tracing import PromptModeTrace


__all__ = [
    "GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION",
    "GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA",
    "Gemma3ModalGeneratorMultifragmentNodeRecord",
    "build_gemma3_modal_generator_multifragment_evaluation",
    "build_gemma3_modal_generator_multifragment_evaluation_from_rung",
    "build_gemma3_modal_generator_multifragment_model_metadata",
    "build_gemma3_modal_generator_multifragment_payload",
    "build_gemma3_modal_generator_multifragment_protocol",
    "build_gemma3_modal_generator_multifragment_report",
    "build_gemma3_modal_generator_multifragment_scientific_status",
    "build_gemma3_modal_generator_multifragment_splits",
    "build_gemma3_modal_generator_multifragment_upstream_metadata",
    "load_gemma3_modal_generator_multifragment_artifact",
    "save_gemma3_modal_generator_multifragment_artifact",
]


GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA = (
    "fisher_graph.gemma3_modal_generator_multifragment_development"
)
GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION = 1

_NODE_KIND = (
    "fisher_graph.gemma3_modal_generator_multifragment_node_record"
)
_NODE_FORMAT_VERSION = 1
_NODE_DOMAIN = (
    b"fisher_graph.gemma3_modal_generator_multifragment.node.v1\0"
)
_PAYLOAD_DOMAIN = (
    b"fisher_graph.gemma3_modal_generator_multifragment.payload.v1\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_modal_generator_multifragment.report.v1\0"
)
_RAW_PARTITION_BINDING_DOMAIN = (
    b"fisher_graph.gemma3_modal_generator_multifragment."
    b"raw_partition_binding.v1\0"
)

_UPSTREAM_SCHEMA = "fisher_graph.gemma3_modal_generator_development"
_UPSTREAM_FORMAT_VERSION = 3

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_RECIPE = (
    "weights",
    "fisher_coupling",
    "parameter_clusters",
    "computational_modes",
    "modal_generators",
    "graph_of_generator_interactions",
    "inference_by_graph_traversal",
)
_FRAGMENT_SELECTION_RULES = frozenset(
    {
        "highest_fisher_mass_same_cluster_causal_pair",
        "predeclared_fit_fisher_multifragment",
    }
)
_INTERACTION_WEIGHTING_RULES = frozenset(
    {
        "native_reference_target_fragment_fisher",
        "mean_normalized_native_reference_fragment_fisher",
        "uniform_valid_token_rows",
    }
)
_INTERACTION_STATE_SOURCE = (
    "edgeless_compiled_gemma_runtime_modal_states"
)
_INTERACTION_TARGET_SOURCE = (
    "native_removed_fragment_at_edgeless_shifted_input"
)

_SAFETY: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_raw_token_rows": False,
    "contains_tokenizer_state": False,
    "contains_generator_weights": True,
    "contains_interaction_weights": True,
    "executable": True,
}
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "text",
        "token_ids",
        "input_ids",
        "targets",
        "score_gradients",
        "activation_rows",
        "gradient_rows",
        "raw_token_rows",
        "raw_fit_rows",
        "raw_eval_rows",
        "source_model_weights",
        "source_parameter_values",
        "model_state_dict",
        "source_state_dict",
        "tokenizer_state",
    }
)

_PAYLOAD_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "model",
    "protocol",
    "splits",
    "upstream_metadata",
    "fit_prompt_trace",
    "parameter_catalog",
    "fisher_coupling",
    "parameter_clusters",
    "parameter_cluster_fragments",
    "node_records",
    "interaction_selection",
    "edgeless_graph",
    "compiler_pipeline",
    "resource_accounting",
    "evaluation",
    "safety",
    "scientific_payload_sha256",
}
_SCIENTIFIC_STATUS_FIELDS = {
    "outcome",
    "compression_claim",
    "heldout_confirmation",
    "clusters_frozen_from_fit_only",
    "node_ranks_predeclared",
    "interaction_selection_split_role",
    "interaction_selection_used_open_development",
    "assessment_reused_for_selection",
    "ready_for_calibration_b",
    "calibration_b_used",
    "guard_used",
    "validation_used",
    "test_used",
    "numerical_extraction_provenance",
    "split_membership_provenance",
    "numerical_extraction_externally_authenticated",
    "split_membership_externally_authenticated",
}
_MODEL_FIELDS = {
    "model_id",
    "requested_revision",
    "resolved_commit",
    "adapter_model_fingerprint",
    "source_whole_model_learned_parameters",
    "local_files_only",
}
_PROTOCOL_FIELDS = {
    "recipe",
    "scope",
    "primary_execution_path",
    "fragment_count",
    "graph_node_count",
    "candidate_interaction_count",
    "selected_interaction_count",
    "graph_traversal_order",
    "fragment_selection_rule",
    "node_rank_selection_rule",
    "interaction_selection_rule",
    "interaction_state_source",
    "interaction_target_source",
    "interaction_weighting",
    "interaction_selection_split_role",
    "assessment_split_role",
    "dense_fused_path_status",
    "local_files_only",
}
_UPSTREAM_FIELDS = {
    "source_schema",
    "source_format_version",
    "source_scientific_payload_sha256",
    "source_evaluation_export_sha256",
    "source_role",
    "fit_prompt_trace_sha256",
    "parameter_catalog_sha256",
    "fisher_coupling_sha256",
    "parameter_clusters_sha256",
    "parameter_cluster_fragments_sha256",
}
_SPLIT_FIELDS = {
    "fit",
    "upstream_evaluation",
    "selection",
    "assessment",
    "partition",
    "provenance",
}
_SPLIT_ENTRY_FIELDS = {
    "role",
    "serialized_sha256",
    "content_sha256",
    "content_count",
}
_SPLIT_PARTITION_FIELDS = {
    "source_evaluation_export_sha256",
    "raw_partition_plan_sha256",
    "selection_partition_sha256",
    "assessment_partition_sha256",
    "raw_partition_binding_sha256",
    "upstream_evaluation_partition_exact",
    "fit_selection_overlap_count",
    "fit_assessment_overlap_count",
    "selection_assessment_overlap_count",
    "all_runtime_partitions_content_disjoint",
}
_SPLIT_PROVENANCE_FIELDS = {
    "export_provenance_assurance",
    "export_provenance_externally_authenticated",
    "split_membership_provenance",
    "split_membership_externally_authenticated",
    "selection_used_for_node_or_edge_fitting",
    "assessment_used_for_node_or_edge_fitting",
    "calibration_b_used",
    "guard_used",
    "validation_used",
    "test_used",
}
_EVALUATION_FIELDS = {
    "assessment_split_sha256",
    "supervised_tokens",
    "logical_valid_tokens",
    "conditions",
    "edgeless_dense_equivalence",
    "selection_split_metrics_stored",
    "assessment_used_for_selection",
}
_CONDITION_NAMES = frozenset(
    {
        "native",
        "interaction_graph",
        "edgeless_graph",
        "matched_deletion",
        "dense_fused_edgeless",
    }
)
_NATIVE_METRIC_FIELDS = {"nll_per_token"}
_CANDIDATE_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_EQUIVALENCE_FIELDS = {
    "compared",
    "scope",
    "maximum_absolute_logit_difference",
    "absolute_tolerance",
    "relative_tolerance",
    "passed",
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _strict_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _update_payload_digest(digest: object, value: object) -> None:
    """Hash JSON-like values and exact tensor bytes without stride aliasing."""

    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a hashlib SHA-256 object")
    if value is None:
        digest.update(b"N;")
    elif isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
    elif type(value) is int:
        digest.update(f"I{value};".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific payload floats must be finite")
        digest.update(f"F{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"S{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("scientific payload tensors must be finite")
        digest.update(b"T")
        _update_payload_digest(digest, str(tensor.dtype))
        _update_payload_digest(digest, tuple(tensor.shape))
        # NumPy serializes logical C-order values and does not require the
        # byte-view stride invariant that fails for size-one tensor axes.
        raw = tensor.numpy().tobytes(order="C")
        digest.update(f"{len(raw)}:".encode("ascii"))
        digest.update(raw)
        digest.update(b";")
    elif isinstance(value, Mapping):
        keys = sorted(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("scientific payload mapping keys must be strings")
        digest.update(f"M{len(keys)}[".encode("ascii"))
        for key in keys:
            _update_payload_digest(digest, key)
            _update_payload_digest(digest, value[key])
        digest.update(b"];")
    elif isinstance(value, tuple):
        digest.update(f"U{len(value)}[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
    elif isinstance(value, list):
        digest.update(f"L{len(value)}[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
    else:
        raise TypeError(
            "scientific payload contains unsupported "
            f"{type(value).__qualname__}"
        )


def _assert_source_safe(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact mapping keys must be strings")
            if key in _FORBIDDEN_KEYS:
                location = ".".join((*path, key))
                raise ValueError(f"artifact contains forbidden field {location}")
            _assert_source_safe(child, path=(*path, key))
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_source_safe(child, path=(*path, str(index)))
        return
    if isinstance(value, str):
        if any(
            character.isspace()
            or unicodedata.category(character).startswith("C")
            for character in value
        ):
            location = ".".join(path) or "<root>"
            raise ValueError(
                f"artifact contains non-machine string at {location}"
            )
        return
    if isinstance(value, torch.Tensor):
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError("artifact contains a non-source-safe scalar")


def _authenticated_copy(value: object, expected_type: type[object]) -> object:
    if not isinstance(value, expected_type):
        raise TypeError(f"artifact must be {expected_type.__name__}")
    validator = getattr(value, "validate_integrity", None)
    if callable(validator):
        validator()
    state_builder = getattr(value, "state_dict", None)
    state_loader = getattr(type(value), "from_state_dict", None)
    if not callable(state_builder) or not callable(state_loader):
        raise TypeError("artifact lacks a strict state roundtrip")
    restored = state_loader(state_builder())
    if getattr(restored, "artifact_sha256", None) != getattr(
        value,
        "artifact_sha256",
        None,
    ):
        raise ValueError("artifact roundtrip changed its hash")
    return restored


@dataclass(frozen=True, slots=True)
class Gemma3ModalGeneratorMultifragmentNodeRecord:
    """One canonical node's two fitted curves and executable lowering."""

    node_name: str
    computational_modes: ComputationalModeRateCurve
    modal_generators: ModalGeneratorRateCurve
    lowering: ModalGeneratorLowering
    artifact_sha256: str = ""
    artifact_kind: str = _NODE_KIND
    format_version: int = _NODE_FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.node_name, label="node_name")
        modes = _authenticated_copy(
            self.computational_modes,
            ComputationalModeRateCurve,
        )
        generators = _authenticated_copy(
            self.modal_generators,
            ModalGeneratorRateCurve,
        )
        lowering = _authenticated_copy(
            self.lowering,
            ModalGeneratorLowering,
        )
        object.__setattr__(self, "computational_modes", modes)
        object.__setattr__(self, "modal_generators", generators)
        object.__setattr__(self, "lowering", lowering)
        selected_basis = modes.selected_basis
        selected_generator = generators.selected_plan
        if selected_basis is None or selected_generator is None:
            raise ValueError("node record requires fixed selected curve points")
        if (
            selected_basis.artifact_sha256
            != lowering.computational_mode_basis.artifact_sha256
            or selected_generator.artifact_sha256
            != lowering.coordinate_generator_plan.artifact_sha256
        ):
            raise ValueError("node curves do not select the saved lowering")
        if (
            selected_basis.binding.parameter_cluster_sha256
            != lowering.selected_fragment_sha256
            or selected_generator.binding.parameter_cluster_fragment_sha256
            != lowering.selected_fragment_sha256
        ):
            raise ValueError("node curves and lowering select different fragments")
        if (
            self.artifact_kind != _NODE_KIND
            or self.format_version != _NODE_FORMAT_VERSION
        ):
            raise ValueError("multifragment node-record header is invalid")
        computed = _json_sha256(self._payload(), domain=_NODE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("multifragment node-record hash mismatch")

    @property
    def fragment_sha256(self) -> str:
        return self.lowering.selected_fragment_sha256

    @property
    def mode_rank(self) -> int:
        return self.lowering.computational_mode_basis.rank

    @property
    def generator_rank(self) -> int:
        return self.lowering.coordinate_generator_plan.rank

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "node_name": self.node_name,
            "fragment_sha256": self.fragment_sha256,
            "computational_modes_sha256": (
                self.computational_modes.artifact_sha256
            ),
            "modal_generators_sha256": (
                self.modal_generators.artifact_sha256
            ),
            "lowering_sha256": self.lowering.artifact_sha256,
            "selected_basis_sha256": (
                self.lowering.computational_mode_basis.artifact_sha256
            ),
            "selected_generator_sha256": (
                self.lowering.coordinate_generator_plan.artifact_sha256
            ),
            "mode_rank": self.mode_rank,
            "generator_rank": self.generator_rank,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "computational_modes": self.computational_modes.state_dict(),
            "modal_generators": self.modal_generators.state_dict(),
            "lowering": self.lowering.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    def validate_integrity(self) -> None:
        modes = ComputationalModeRateCurve.from_state_dict(
            self.computational_modes.state_dict()
        )
        generators = ModalGeneratorRateCurve.from_state_dict(
            self.modal_generators.state_dict()
        )
        lowering = ModalGeneratorLowering.from_state_dict(
            self.lowering.state_dict()
        )
        if (
            modes.artifact_sha256
            != self.computational_modes.artifact_sha256
            or generators.artifact_sha256
            != self.modal_generators.artifact_sha256
            or lowering.artifact_sha256 != self.lowering.artifact_sha256
        ):
            raise ValueError("node-record nested artifact roundtrip drifted")
        if _json_sha256(self._payload(), domain=_NODE_DOMAIN) != (
            self.artifact_sha256
        ):
            raise ValueError("multifragment node-record hash mismatch")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> Gemma3ModalGeneratorMultifragmentNodeRecord:
        fields = {
            "artifact_kind",
            "format_version",
            "node_name",
            "fragment_sha256",
            "computational_modes_sha256",
            "modal_generators_sha256",
            "lowering_sha256",
            "selected_basis_sha256",
            "selected_generator_sha256",
            "mode_rank",
            "generator_rank",
            "computational_modes",
            "modal_generators",
            "lowering",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="multifragment node record")
        modes = ComputationalModeRateCurve.from_state_dict(
            state["computational_modes"]  # type: ignore[arg-type]
        )
        generators = ModalGeneratorRateCurve.from_state_dict(
            state["modal_generators"]  # type: ignore[arg-type]
        )
        lowering = ModalGeneratorLowering.from_state_dict(
            state["lowering"]  # type: ignore[arg-type]
        )
        result = cls(
            node_name=state["node_name"],  # type: ignore[arg-type]
            computational_modes=modes,
            modal_generators=generators,
            lowering=lowering,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        expected = result._payload()
        for field in (
            "fragment_sha256",
            "computational_modes_sha256",
            "modal_generators_sha256",
            "lowering_sha256",
            "selected_basis_sha256",
            "selected_generator_sha256",
            "mode_rank",
            "generator_rank",
        ):
            if state[field] != expected[field]:
                raise ValueError(f"serialized node {field} is inconsistent")
        return result


def build_gemma3_modal_generator_multifragment_scientific_status(
) -> dict[str, object]:
    return {
        "outcome": "development_only_multifragment_modal_generator_measurement",
        "compression_claim": False,
        "heldout_confirmation": False,
        "clusters_frozen_from_fit_only": True,
        "node_ranks_predeclared": True,
        "interaction_selection_split_role": "open_development_selection",
        "interaction_selection_used_open_development": True,
        "assessment_reused_for_selection": False,
        "ready_for_calibration_b": False,
        "calibration_b_used": False,
        "guard_used": False,
        "validation_used": False,
        "test_used": False,
        "numerical_extraction_provenance": "caller_declared_self_attested",
        "split_membership_provenance": "caller_declared_self_attested",
        "numerical_extraction_externally_authenticated": False,
        "split_membership_externally_authenticated": False,
    }


def build_gemma3_modal_generator_multifragment_model_metadata(
    *,
    model_id: str,
    requested_revision: str,
    resolved_commit: str,
    adapter_model_fingerprint: str,
    source_whole_model_learned_parameters: int,
) -> dict[str, object]:
    _require_name(model_id, label="model_id")
    if (
        not isinstance(requested_revision, str)
        or _REVISION.fullmatch(requested_revision) is None
        or not isinstance(resolved_commit, str)
        or _REVISION.fullmatch(resolved_commit) is None
        or requested_revision != resolved_commit
    ):
        raise ValueError("model revisions must be the same exact commit")
    _require_sha256(
        adapter_model_fingerprint,
        label="adapter_model_fingerprint",
    )
    _require_int(
        source_whole_model_learned_parameters,
        label="source_whole_model_learned_parameters",
        minimum=1,
    )
    return {
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_commit": resolved_commit,
        "adapter_model_fingerprint": adapter_model_fingerprint,
        "source_whole_model_learned_parameters": (
            source_whole_model_learned_parameters
        ),
        "local_files_only": True,
    }


def build_gemma3_modal_generator_multifragment_upstream_metadata(
    *,
    source_scientific_payload_sha256: str,
    source_evaluation_export_sha256: str,
    fit_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    fisher_coupling: GroupedVirtualGateFisher,
    parameter_clusters: FisherPromptClusterPlan,
    parameter_cluster_fragments: ParameterClusterLayerFragmentPlan,
) -> dict[str, object]:
    return {
        "source_schema": _UPSTREAM_SCHEMA,
        "source_format_version": _UPSTREAM_FORMAT_VERSION,
        "source_scientific_payload_sha256": _require_sha256(
            source_scientific_payload_sha256,
            label="source_scientific_payload_sha256",
        ),
        "source_evaluation_export_sha256": _require_sha256(
            source_evaluation_export_sha256,
            label="source_evaluation_export_sha256",
        ),
        "source_role": "strict_loaded_v3_fit_fisher_analysis",
        "fit_prompt_trace_sha256": fit_prompt_trace.artifact_sha256,
        "parameter_catalog_sha256": parameter_catalog.artifact_sha256,
        "fisher_coupling_sha256": fisher_coupling.artifact_sha256,
        "parameter_clusters_sha256": parameter_clusters.artifact_sha256,
        "parameter_cluster_fragments_sha256": (
            parameter_cluster_fragments.artifact_sha256
        ),
    }


def _canonical_content_hashes(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(
        _require_sha256(value, label=f"{label} content hash")
        for value in values
    )
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} content hashes must be nonempty and unique")
    return result


def build_gemma3_modal_generator_multifragment_splits(
    *,
    fit_split_sha256: str,
    upstream_evaluation_split_sha256: str,
    selection_split_sha256: str,
    assessment_split_sha256: str,
    source_evaluation_export_sha256: str,
    raw_partition_plan_sha256: str,
    selection_partition_sha256: str,
    assessment_partition_sha256: str,
    fit_content_sha256s: Sequence[str],
    upstream_evaluation_content_sha256s: Sequence[str],
    selection_content_sha256s: Sequence[str],
    assessment_content_sha256s: Sequence[str],
) -> dict[str, object]:
    split_hashes = tuple(
        _require_sha256(value, label=label)
        for value, label in (
            (fit_split_sha256, "fit_split_sha256"),
            (
                upstream_evaluation_split_sha256,
                "upstream_evaluation_split_sha256",
            ),
            (selection_split_sha256, "selection_split_sha256"),
            (assessment_split_sha256, "assessment_split_sha256"),
        )
    )
    if len(split_hashes) != len(set(split_hashes)):
        raise ValueError("fit, upstream, selection, and assessment hashes differ")
    raw_partition_lineage = {
        "source_evaluation_export_sha256": _require_sha256(
            source_evaluation_export_sha256,
            label="source_evaluation_export_sha256",
        ),
        "raw_partition_plan_sha256": _require_sha256(
            raw_partition_plan_sha256,
            label="raw_partition_plan_sha256",
        ),
        "selection_partition_sha256": _require_sha256(
            selection_partition_sha256,
            label="selection_partition_sha256",
        ),
        "assessment_partition_sha256": _require_sha256(
            assessment_partition_sha256,
            label="assessment_partition_sha256",
        ),
    }
    fit = _canonical_content_hashes(
        fit_content_sha256s,
        label="fit",
    )
    upstream = _canonical_content_hashes(
        upstream_evaluation_content_sha256s,
        label="upstream evaluation",
    )
    selection = _canonical_content_hashes(
        selection_content_sha256s,
        label="selection",
    )
    assessment = _canonical_content_hashes(
        assessment_content_sha256s,
        label="assessment",
    )
    if set(selection) & set(assessment):
        raise ValueError("selection and assessment content overlap")
    if set(selection) | set(assessment) != set(upstream):
        raise ValueError(
            "selection and assessment must exactly partition upstream evaluation"
        )
    fit_selection = len(set(fit) & set(selection))
    fit_assessment = len(set(fit) & set(assessment))
    if fit_selection or fit_assessment:
        raise ValueError("fit content overlaps a development partition")

    def entry(
        role: str,
        digest: str,
        content: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "role": role,
            "serialized_sha256": digest,
            "content_sha256": content,
            "content_count": len(content),
        }

    return {
        "fit": entry("fit_fisher_and_node_training", split_hashes[0], fit),
        "upstream_evaluation": entry(
            "upstream_open_development_parent",
            split_hashes[1],
            upstream,
        ),
        "selection": entry(
            "open_development_node_and_edge_selection",
            split_hashes[2],
            selection,
        ),
        "assessment": entry(
            "open_development_post_selection_assessment",
            split_hashes[3],
            assessment,
        ),
        "partition": {
            **raw_partition_lineage,
            "raw_partition_binding_sha256": _json_sha256(
                raw_partition_lineage,
                domain=_RAW_PARTITION_BINDING_DOMAIN,
            ),
            "upstream_evaluation_partition_exact": True,
            "fit_selection_overlap_count": fit_selection,
            "fit_assessment_overlap_count": fit_assessment,
            "selection_assessment_overlap_count": 0,
            "all_runtime_partitions_content_disjoint": True,
        },
        "provenance": {
            "export_provenance_assurance": "declared_self_attested",
            "export_provenance_externally_authenticated": False,
            "split_membership_provenance": "caller_declared_self_attested",
            "split_membership_externally_authenticated": False,
            "selection_used_for_node_or_edge_fitting": True,
            "assessment_used_for_node_or_edge_fitting": False,
            "calibration_b_used": False,
            "guard_used": False,
            "validation_used": False,
            "test_used": False,
        },
    }


def build_gemma3_modal_generator_multifragment_protocol(
    *,
    compiler_pipeline: ModalCompilerPipeline,
    fragment_selection_rule: str,
    interaction_weighting: str,
) -> dict[str, object]:
    if fragment_selection_rule not in _FRAGMENT_SELECTION_RULES:
        raise ValueError("fragment_selection_rule is invalid")
    if interaction_weighting not in _INTERACTION_WEIGHTING_RULES:
        raise ValueError("interaction_weighting is invalid")
    selection = compiler_pipeline.interaction_selection
    if selection is None:
        raise ValueError("multifragment protocol requires edge selection")
    return {
        "recipe": _RECIPE,
        "scope": "fit_fisher_multifragment_interaction_development",
        "primary_execution_path": (
            "incremental_modal_generator_graph_traversal"
        ),
        "fragment_count": len(compiler_pipeline.nodes),
        "graph_node_count": len(compiler_pipeline.graph_plan.nodes),
        "candidate_interaction_count": len(selection.candidate_edges),
        "selected_interaction_count": len(selection.interactions),
        "graph_traversal_order": (
            compiler_pipeline.graph_plan.traversal_order
        ),
        "fragment_selection_rule": fragment_selection_rule,
        "node_rank_selection_rule": "fixed_predeclared",
        "interaction_selection_rule": (
            "greedy_open_development_weighted_nrmse"
        ),
        "interaction_state_source": _INTERACTION_STATE_SOURCE,
        "interaction_target_source": _INTERACTION_TARGET_SOURCE,
        "interaction_weighting": interaction_weighting,
        "interaction_selection_split_role": "open_development_selection",
        "assessment_split_role": (
            "open_development_post_selection_assessment"
        ),
        "dense_fused_path_status": (
            "separate_edgeless_optimization_control"
        ),
        "local_files_only": True,
    }


def _quality_metrics(
    value: Mapping[str, object],
    *,
    native_nll: float,
    native: bool,
    label: str,
) -> dict[str, float]:
    expected = _NATIVE_METRIC_FIELDS if native else _CANDIDATE_METRIC_FIELDS
    _strict_fields(value, expected, label=f"{label} metrics")
    nll = _require_float(
        value["nll_per_token"],
        label=f"{label} nll_per_token",
        minimum=0.0,
    )
    if native:
        return {"nll_per_token": nll}
    delta = _require_float(
        value["delta_nll_per_token"],
        label=f"{label} delta_nll_per_token",
    )
    if not math.isclose(
        delta,
        nll - native_nll,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} NLL delta is inconsistent")
    kl = _require_float(
        value["native_to_candidate_kl_per_token"],
        label=f"{label} KL",
        minimum=0.0,
    )
    agreement = _require_float(
        value["top1_agreement_to_native"],
        label=f"{label} top1 agreement",
        minimum=0.0,
    )
    if agreement > 1.0:
        raise ValueError(f"{label} top1 agreement must be <= 1")
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": agreement,
    }


def build_gemma3_modal_generator_multifragment_evaluation(
    *,
    assessment_split_sha256: str,
    supervised_tokens: int,
    logical_valid_tokens: int,
    conditions: Mapping[str, Mapping[str, object]],
    edgeless_dense_max_abs_logit_difference: float,
    edgeless_dense_absolute_tolerance: float,
    edgeless_dense_relative_tolerance: float = 0.0,
) -> dict[str, object]:
    split = _require_sha256(
        assessment_split_sha256,
        label="assessment_split_sha256",
    )
    supervised = _require_int(
        supervised_tokens,
        label="supervised_tokens",
        minimum=1,
    )
    valid = _require_int(
        logical_valid_tokens,
        label="logical_valid_tokens",
        minimum=1,
    )
    if supervised > valid:
        raise ValueError("supervised_tokens cannot exceed logical_valid_tokens")
    if not isinstance(conditions, Mapping) or set(conditions) != (
        _CONDITION_NAMES
    ):
        raise ValueError("evaluation conditions are invalid")
    native_metrics = _quality_metrics(
        conditions["native"],
        native_nll=0.0,
        native=True,
        label="native",
    )
    native_nll = native_metrics["nll_per_token"]
    canonical_conditions: dict[str, object] = {"native": native_metrics}
    for name in sorted(_CONDITION_NAMES - {"native"}):
        canonical_conditions[name] = _quality_metrics(
            conditions[name],
            native_nll=native_nll,
            native=False,
            label=name,
        )
    difference = _require_float(
        edgeless_dense_max_abs_logit_difference,
        label="edgeless/dense maximum absolute logit difference",
        minimum=0.0,
    )
    tolerance = _require_float(
        edgeless_dense_absolute_tolerance,
        label="edgeless/dense absolute tolerance",
        minimum=0.0,
    )
    relative_tolerance = _require_float(
        edgeless_dense_relative_tolerance,
        label="edgeless/dense relative tolerance",
        minimum=0.0,
    )
    return {
        "assessment_split_sha256": split,
        "supervised_tokens": supervised,
        "logical_valid_tokens": valid,
        "conditions": canonical_conditions,
        "edgeless_dense_equivalence": {
            "compared": True,
            "scope": "supervised_logits",
            "maximum_absolute_logit_difference": difference,
            "absolute_tolerance": tolerance,
            "relative_tolerance": relative_tolerance,
            # The unified evaluator raises before returning when its
            # elementwise atol/rtol comparison fails.
            "passed": True,
        },
        "selection_split_metrics_stored": False,
        "assessment_used_for_selection": False,
    }


def build_gemma3_modal_generator_multifragment_evaluation_from_rung(
    *,
    assessment_split_sha256: str,
    rung_evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Normalize the standalone unified-rung evaluator's source-safe report."""

    required = {
        "execution_path",
        "assessment_role",
        "heldout_confirmation",
        "supervised_tokens",
        "logical_valid_tokens",
        "native",
        "conditions",
        "graph_comparison",
        "resource_accounting",
        "latency_or_kernel_speed_claim",
    }
    _strict_fields(rung_evaluation, required, label="rung_evaluation")
    if (
        rung_evaluation["execution_path"]
        != "unified_modal_generator_graph_rung"
        or rung_evaluation["assessment_role"]
        != "open_development_assessment"
        or rung_evaluation["heldout_confirmation"] is not False
        or rung_evaluation["latency_or_kernel_speed_claim"] is not False
    ):
        raise ValueError("rung evaluation scientific labels are invalid")
    native = rung_evaluation["native"]
    conditions = rung_evaluation["conditions"]
    comparison = rung_evaluation["graph_comparison"]
    if (
        not isinstance(native, Mapping)
        or not isinstance(conditions, Mapping)
        or not isinstance(comparison, Mapping)
    ):
        raise TypeError("rung evaluation nested values must be mappings")
    expected_conditions = {
        "interacting_graph",
        "edgeless_graph",
        "matched_deletion",
        "nodewise_dense_fused",
    }
    if set(conditions) != expected_conditions:
        raise ValueError("rung evaluation must include the dense control")
    if (
        comparison.get("nodewise_dense_supplied") is not True
        or comparison.get("nodewise_dense_agrees_with_edgeless") is not True
        or comparison.get("nodewise_dense_equivalence_scope")
        != "supervised_logits"
    ):
        raise ValueError("rung edgeless/dense equivalence did not pass")
    normalized_conditions = {
        "native": native,
        "interaction_graph": conditions["interacting_graph"],
        "edgeless_graph": conditions["edgeless_graph"],
        "matched_deletion": conditions["matched_deletion"],
        "dense_fused_edgeless": conditions["nodewise_dense_fused"],
    }
    return build_gemma3_modal_generator_multifragment_evaluation(
        assessment_split_sha256=assessment_split_sha256,
        supervised_tokens=rung_evaluation[
            "supervised_tokens"
        ],  # type: ignore[arg-type]
        logical_valid_tokens=rung_evaluation[
            "logical_valid_tokens"
        ],  # type: ignore[arg-type]
        conditions=normalized_conditions,  # type: ignore[arg-type]
        edgeless_dense_max_abs_logit_difference=comparison[
            "nodewise_dense_max_abs_logit_difference"
        ],  # type: ignore[arg-type]
        edgeless_dense_absolute_tolerance=comparison[
            "nodewise_dense_equivalence_atol"
        ],  # type: ignore[arg-type]
        edgeless_dense_relative_tolerance=comparison[
            "nodewise_dense_equivalence_rtol"
        ],  # type: ignore[arg-type]
    )


def _validate_scientific_status(value: Mapping[str, object]) -> None:
    _strict_fields(
        value,
        _SCIENTIFIC_STATUS_FIELDS,
        label="scientific_status",
    )
    if value != build_gemma3_modal_generator_multifragment_scientific_status():
        raise ValueError("scientific_status is invalid or overclaims evidence")


def _validate_model(
    value: Mapping[str, object],
    *,
    model_fingerprint: str,
) -> None:
    _strict_fields(value, _MODEL_FIELDS, label="model")
    expected = build_gemma3_modal_generator_multifragment_model_metadata(
        model_id=value["model_id"],  # type: ignore[arg-type]
        requested_revision=value["requested_revision"],  # type: ignore[arg-type]
        resolved_commit=value["resolved_commit"],  # type: ignore[arg-type]
        adapter_model_fingerprint=value[
            "adapter_model_fingerprint"
        ],  # type: ignore[arg-type]
        source_whole_model_learned_parameters=value[
            "source_whole_model_learned_parameters"
        ],  # type: ignore[arg-type]
    )
    if value != expected or value["adapter_model_fingerprint"] != (
        model_fingerprint
    ):
        raise ValueError("model metadata does not bind the compiler model")


def _validate_upstream(
    value: Mapping[str, object],
    *,
    fit_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    fisher_coupling: GroupedVirtualGateFisher,
    parameter_clusters: FisherPromptClusterPlan,
    parameter_cluster_fragments: ParameterClusterLayerFragmentPlan,
) -> None:
    _strict_fields(value, _UPSTREAM_FIELDS, label="upstream_metadata")
    expected = build_gemma3_modal_generator_multifragment_upstream_metadata(
        source_scientific_payload_sha256=value[
            "source_scientific_payload_sha256"
        ],  # type: ignore[arg-type]
        source_evaluation_export_sha256=value[
            "source_evaluation_export_sha256"
        ],  # type: ignore[arg-type]
        fit_prompt_trace=fit_prompt_trace,
        parameter_catalog=parameter_catalog,
        fisher_coupling=fisher_coupling,
        parameter_clusters=parameter_clusters,
        parameter_cluster_fragments=parameter_cluster_fragments,
    )
    if value != expected:
        raise ValueError("upstream metadata is inconsistent")


def _validate_splits(value: Mapping[str, object]) -> None:
    _strict_fields(value, _SPLIT_FIELDS, label="splits")
    for name in ("fit", "upstream_evaluation", "selection", "assessment"):
        _strict_fields(
            value[name],  # type: ignore[arg-type]
            _SPLIT_ENTRY_FIELDS,
            label=f"splits.{name}",
        )
    _strict_fields(
        value["partition"],  # type: ignore[arg-type]
        _SPLIT_PARTITION_FIELDS,
        label="splits.partition",
    )
    _strict_fields(
        value["provenance"],  # type: ignore[arg-type]
        _SPLIT_PROVENANCE_FIELDS,
        label="splits.provenance",
    )
    rebuilt = build_gemma3_modal_generator_multifragment_splits(
        fit_split_sha256=value["fit"]["serialized_sha256"],  # type: ignore[index]
        upstream_evaluation_split_sha256=value[
            "upstream_evaluation"
        ]["serialized_sha256"],  # type: ignore[index]
        selection_split_sha256=value[
            "selection"
        ]["serialized_sha256"],  # type: ignore[index]
        assessment_split_sha256=value[
            "assessment"
        ]["serialized_sha256"],  # type: ignore[index]
        source_evaluation_export_sha256=value[
            "partition"
        ]["source_evaluation_export_sha256"],  # type: ignore[index]
        raw_partition_plan_sha256=value[
            "partition"
        ]["raw_partition_plan_sha256"],  # type: ignore[index]
        selection_partition_sha256=value[
            "partition"
        ]["selection_partition_sha256"],  # type: ignore[index]
        assessment_partition_sha256=value[
            "partition"
        ]["assessment_partition_sha256"],  # type: ignore[index]
        fit_content_sha256s=value["fit"]["content_sha256"],  # type: ignore[index]
        upstream_evaluation_content_sha256s=value[
            "upstream_evaluation"
        ]["content_sha256"],  # type: ignore[index]
        selection_content_sha256s=value[
            "selection"
        ]["content_sha256"],  # type: ignore[index]
        assessment_content_sha256s=value[
            "assessment"
        ]["content_sha256"],  # type: ignore[index]
    )
    if value != rebuilt:
        raise ValueError("split metadata is inconsistent")


def _validate_protocol(
    value: Mapping[str, object],
    *,
    compiler_pipeline: ModalCompilerPipeline,
) -> None:
    _strict_fields(value, _PROTOCOL_FIELDS, label="protocol")
    expected = build_gemma3_modal_generator_multifragment_protocol(
        compiler_pipeline=compiler_pipeline,
        fragment_selection_rule=value[
            "fragment_selection_rule"
        ],  # type: ignore[arg-type]
        interaction_weighting=value[
            "interaction_weighting"
        ],  # type: ignore[arg-type]
    )
    if value != expected:
        raise ValueError("protocol metadata is inconsistent")


def _validate_evaluation(
    value: Mapping[str, object],
    *,
    assessment_split_sha256: str,
) -> None:
    _strict_fields(value, _EVALUATION_FIELDS, label="evaluation")
    conditions = value["conditions"]
    equivalence = value["edgeless_dense_equivalence"]
    if not isinstance(conditions, Mapping) or not isinstance(
        equivalence,
        Mapping,
    ):
        raise TypeError("evaluation conditions/equivalence must be mappings")
    _strict_fields(
        equivalence,
        _EQUIVALENCE_FIELDS,
        label="evaluation.edgeless_dense_equivalence",
    )
    rebuilt = build_gemma3_modal_generator_multifragment_evaluation(
        assessment_split_sha256=value[
            "assessment_split_sha256"
        ],  # type: ignore[arg-type]
        supervised_tokens=value["supervised_tokens"],  # type: ignore[arg-type]
        logical_valid_tokens=value[
            "logical_valid_tokens"
        ],  # type: ignore[arg-type]
        conditions=conditions,  # type: ignore[arg-type]
        edgeless_dense_max_abs_logit_difference=equivalence[
            "maximum_absolute_logit_difference"
        ],  # type: ignore[arg-type]
        edgeless_dense_absolute_tolerance=equivalence[
            "absolute_tolerance"
        ],  # type: ignore[arg-type]
        edgeless_dense_relative_tolerance=equivalence[
            "relative_tolerance"
        ],  # type: ignore[arg-type]
    )
    if value != rebuilt or value["assessment_split_sha256"] != (
        assessment_split_sha256
    ):
        raise ValueError("evaluation metadata is inconsistent")


def _resource_accounting(
    *,
    model: Mapping[str, object],
    pipeline: ModalCompilerPipeline,
    edgeless_graph: ModalGeneratorGraphPlan,
    node_records: tuple[
        Gemma3ModalGeneratorMultifragmentNodeRecord,
        ...,
    ],
) -> dict[str, object]:
    source_parameters = pipeline.source_parameter_count
    source_macs = pipeline.source_macs_per_token
    if source_parameters is None or source_macs is None:
        raise ValueError("multifragment artifact requires exact source accounting")
    whole_model = _require_int(
        model["source_whole_model_learned_parameters"],
        label="source whole-model parameters",
        minimum=source_parameters,
    )
    dense_parameters = sum(
        record.lowering.fused_residual_plan.parameter_count
        for record in node_records
    )
    dense_macs = sum(
        (
            record.lowering.fused_residual_plan.input_width
            * record.lowering.fused_residual_plan.rank
            + record.lowering.fused_residual_plan.rank
            * record.lowering.fused_residual_plan.output_width
        )
        for record in node_records
    )
    dense_additions = sum(
        (
            0
            if record.lowering.fused_residual_plan.factors.bias is None
            else record.lowering.fused_residual_plan.output_width
        )
        for record in node_records
    )

    def control(
        *,
        replacement_parameters: int,
        matrix_macs_per_token: int,
        elementwise_additions_per_token: int,
    ) -> dict[str, int]:
        return {
            "replacement_learned_parameters": replacement_parameters,
            "candidate_whole_model_learned_parameters": (
                whole_model - source_parameters + replacement_parameters
            ),
            "matrix_macs_per_token": matrix_macs_per_token,
            "elementwise_additions_per_token": (
                elementwise_additions_per_token
            ),
            "net_stored_parameter_savings": (
                source_parameters - replacement_parameters
            ),
            "net_matrix_macs_saved_per_token": (
                source_macs - matrix_macs_per_token
            ),
        }

    return {
        "source": {
            "source_whole_model_learned_parameters": whole_model,
            "native_removed_learned_parameters": source_parameters,
            "native_removed_matrix_macs_per_token": source_macs,
        },
        "interaction_graph": control(
            replacement_parameters=pipeline.graph_plan.parameter_count,
            matrix_macs_per_token=pipeline.graph_plan.macs_per_token,
            elementwise_additions_per_token=(
                pipeline.graph_plan.accounting.elementwise_additions_per_token
            ),
        ),
        "edgeless_graph": control(
            replacement_parameters=edgeless_graph.parameter_count,
            matrix_macs_per_token=edgeless_graph.macs_per_token,
            elementwise_additions_per_token=(
                edgeless_graph.accounting.elementwise_additions_per_token
            ),
        ),
        "matched_deletion": control(
            replacement_parameters=0,
            matrix_macs_per_token=0,
            elementwise_additions_per_token=0,
        ),
        "dense_fused_edgeless": control(
            replacement_parameters=dense_parameters,
            matrix_macs_per_token=dense_macs,
            elementwise_additions_per_token=dense_additions,
        ),
        "definitions": {
            "matrix_mac_policy": "matrix_multiplies_only",
            "additions_reported_separately": True,
            "matched_deletion_storage_scope": (
                "standalone_control_model_no_replacement_stored"
            ),
            "latency_or_kernel_speed_claim": False,
        },
    }


def _validate_cross_lineage(
    *,
    fit_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    fisher_coupling: GroupedVirtualGateFisher,
    parameter_clusters: FisherPromptClusterPlan,
    parameter_cluster_fragments: ParameterClusterLayerFragmentPlan,
    node_records: tuple[
        Gemma3ModalGeneratorMultifragmentNodeRecord,
        ...,
    ],
    interaction_selection: ModalInteractionSelection,
    edgeless_graph: ModalGeneratorGraphPlan,
    compiler_pipeline: ModalCompilerPipeline,
    splits: Mapping[str, object],
    upstream_metadata: Mapping[str, object],
) -> None:
    if (
        fisher_coupling.source_prompt_trace_sha256
        != fit_prompt_trace.artifact_sha256
        or fit_prompt_trace.provenance.source_model_fingerprint
        != parameter_catalog.model_fingerprint
        or fit_prompt_trace.provenance.calibration_split_sha256
        != fisher_coupling.calibration_split_sha256
        or fisher_coupling.catalog.artifact_sha256
        != parameter_catalog.artifact_sha256
        or parameter_clusters.config.source_fisher_coupling_sha256
        != fisher_coupling.artifact_sha256
        or parameter_cluster_fragments.source_cluster_plan_sha256
        != parameter_clusters.artifact_sha256
        or parameter_cluster_fragments.source_fisher_coupling_sha256
        != fisher_coupling.artifact_sha256
    ):
        raise ValueError("fit trace/Fisher/cluster lineage is inconsistent")
    if (
        compiler_pipeline.parameter_catalog.artifact_sha256
        != parameter_catalog.artifact_sha256
        or compiler_pipeline.grouped_fisher.referenced_artifact_sha256
        != fisher_coupling.artifact_sha256
        or compiler_pipeline.fisher_clusters.referenced_artifact_sha256
        != parameter_clusters.artifact_sha256
        or compiler_pipeline.parameter_cluster_fragments.artifact_sha256
        != parameter_cluster_fragments.artifact_sha256
        or compiler_pipeline.interaction_selection is None
        or compiler_pipeline.interaction_selection.artifact_sha256
        != interaction_selection.artifact_sha256
    ):
        raise ValueError("common analysis and compiler pipeline lineage drifted")
    if len(node_records) < 2:
        raise ValueError("multifragment artifact requires at least two nodes")
    pipeline_nodes = compiler_pipeline.nodes
    if tuple(record.node_name for record in node_records) != tuple(
        node.node_name for node in pipeline_nodes
    ):
        raise ValueError("node records are not in compiler traversal order")
    for record, node in zip(node_records, pipeline_nodes):
        if record.lowering.artifact_sha256 != node.lowering.artifact_sha256:
            raise ValueError("node record lowering differs from compiler node")
    graph = compiler_pipeline.graph_plan
    if (
        graph.artifact_sha256 == edgeless_graph.artifact_sha256
        and graph.interactions
    ):
        raise ValueError("interactive and edgeless graphs cannot share a hash")
    if (
        edgeless_graph.interactions
        or edgeless_graph.model_fingerprint != graph.model_fingerprint
        or edgeless_graph.parameter_cluster_plan_sha256
        != graph.parameter_cluster_plan_sha256
        or tuple(
            node.artifact_sha256 for node in edgeless_graph.nodes
        )
        != tuple(node.artifact_sha256 for node in graph.nodes)
    ):
        raise ValueError("edgeless control does not contain the exact graph nodes")
    if tuple(
        edge.artifact_sha256 for edge in interaction_selection.interactions
    ) != tuple(edge.artifact_sha256 for edge in graph.interactions):
        raise ValueError("compiler graph edges differ from interaction selection")
    fit_split = splits["fit"]["serialized_sha256"]  # type: ignore[index]
    selection_split = splits["selection"][
        "serialized_sha256"
    ]  # type: ignore[index]
    if splits["partition"][  # type: ignore[index]
        "source_evaluation_export_sha256"
    ] != upstream_metadata["source_evaluation_export_sha256"]:
        raise ValueError(
            "raw evaluation export and upstream metadata hashes differ"
        )
    if (
        fit_prompt_trace.provenance.calibration_split_sha256 != fit_split
        or compiler_pipeline.fit_split_sha256 != fit_split
        or compiler_pipeline.eval_split_sha256 != selection_split
        or interaction_selection.fit_split_sha256 != fit_split
        or interaction_selection.eval_split_sha256 != selection_split
    ):
        raise ValueError("fit/selection split hashes do not bind the pipeline")


def _coerce_node_records(
    values: Sequence[Gemma3ModalGeneratorMultifragmentNodeRecord],
) -> tuple[Gemma3ModalGeneratorMultifragmentNodeRecord, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("node_records must be a sequence")
    records = tuple(
        Gemma3ModalGeneratorMultifragmentNodeRecord.from_state_dict(
            value.state_dict()
        )
        if isinstance(value, Gemma3ModalGeneratorMultifragmentNodeRecord)
        else None
        for value in values
    )
    if not records or any(value is None for value in records):
        raise TypeError("node_records contain an invalid value")
    result = tuple(records)  # type: ignore[arg-type]
    names = tuple(value.node_name for value in result)
    if len(names) != len(set(names)):
        raise ValueError("node record names must be unique")
    return result


def _validate_and_restore_payload(
    raw: Mapping[str, object],
) -> tuple[
    PromptModeTrace,
    NaturalMLPParameterGroupCatalog,
    GroupedVirtualGateFisher,
    FisherPromptClusterPlan,
    ParameterClusterLayerFragmentPlan,
    tuple[Gemma3ModalGeneratorMultifragmentNodeRecord, ...],
    ModalInteractionSelection,
    ModalGeneratorGraphPlan,
    ModalCompilerPipeline,
]:
    _strict_fields(raw, _PAYLOAD_FIELDS, label="multifragment artifact")
    if (
        raw.get("schema") != GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA
        or raw.get("format_version")
        != GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION
    ):
        raise ValueError("unsupported multifragment artifact")
    if raw.get("safety") != _SAFETY:
        raise ValueError("multifragment artifact safety flags are invalid")
    _assert_source_safe(raw)
    digest = _require_sha256(
        raw.get("scientific_payload_sha256"),
        label="scientific_payload_sha256",
    )
    without_digest = {
        key: value
        for key, value in raw.items()
        if key != "scientific_payload_sha256"
    }
    if _payload_sha256(without_digest) != digest:
        raise ValueError("multifragment scientific payload hash mismatch")

    fit_trace = PromptModeTrace.from_state_dict(
        raw["fit_prompt_trace"]  # type: ignore[arg-type]
    )
    catalog = NaturalMLPParameterGroupCatalog.from_state_dict(
        raw["parameter_catalog"]  # type: ignore[arg-type]
    )
    fisher = GroupedVirtualGateFisher.from_state_dict(
        raw["fisher_coupling"]  # type: ignore[arg-type]
    )
    clusters = FisherPromptClusterPlan.from_state_dict(
        raw["parameter_clusters"]  # type: ignore[arg-type]
    )
    fragments = ParameterClusterLayerFragmentPlan.from_state_dict(
        raw["parameter_cluster_fragments"]  # type: ignore[arg-type]
    )
    raw_records = raw["node_records"]
    if not isinstance(raw_records, tuple):
        raise TypeError("serialized node_records must be a tuple")
    records = tuple(
        Gemma3ModalGeneratorMultifragmentNodeRecord.from_state_dict(value)
        for value in raw_records  # type: ignore[arg-type]
    )
    selection = ModalInteractionSelection.from_state_dict(
        raw["interaction_selection"]  # type: ignore[arg-type]
    )
    edgeless = ModalGeneratorGraphPlan.from_state_dict(
        raw["edgeless_graph"]  # type: ignore[arg-type]
    )
    pipeline = ModalCompilerPipeline.from_state_dict(
        raw["compiler_pipeline"]  # type: ignore[arg-type]
    )

    scientific_status = raw["scientific_status"]
    model = raw["model"]
    protocol = raw["protocol"]
    splits = raw["splits"]
    upstream = raw["upstream_metadata"]
    evaluation = raw["evaluation"]
    for value, label in (
        (scientific_status, "scientific_status"),
        (model, "model"),
        (protocol, "protocol"),
        (splits, "splits"),
        (upstream, "upstream_metadata"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    _validate_scientific_status(scientific_status)
    _validate_model(model, model_fingerprint=catalog.model_fingerprint)
    _validate_splits(splits)
    _validate_upstream(
        upstream,
        fit_prompt_trace=fit_trace,
        parameter_catalog=catalog,
        fisher_coupling=fisher,
        parameter_clusters=clusters,
        parameter_cluster_fragments=fragments,
    )
    _validate_protocol(protocol, compiler_pipeline=pipeline)
    assessment_split = splits["assessment"][
        "serialized_sha256"
    ]  # type: ignore[index]
    _validate_evaluation(
        evaluation,
        assessment_split_sha256=assessment_split,
    )
    _validate_cross_lineage(
        fit_prompt_trace=fit_trace,
        parameter_catalog=catalog,
        fisher_coupling=fisher,
        parameter_clusters=clusters,
        parameter_cluster_fragments=fragments,
        node_records=records,
        interaction_selection=selection,
        edgeless_graph=edgeless,
        compiler_pipeline=pipeline,
        splits=splits,
        upstream_metadata=upstream,
    )
    expected_resources = _resource_accounting(
        model=model,
        pipeline=pipeline,
        edgeless_graph=edgeless,
        node_records=records,
    )
    if raw["resource_accounting"] != expected_resources:
        raise ValueError("saved multifragment resource accounting is inconsistent")
    return (
        fit_trace,
        catalog,
        fisher,
        clusters,
        fragments,
        records,
        selection,
        edgeless,
        pipeline,
    )


def build_gemma3_modal_generator_multifragment_payload(
    *,
    scientific_status: Mapping[str, object],
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    splits: Mapping[str, object],
    upstream_metadata: Mapping[str, object],
    fit_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    fisher_coupling: GroupedVirtualGateFisher,
    parameter_clusters: FisherPromptClusterPlan,
    parameter_cluster_fragments: ParameterClusterLayerFragmentPlan,
    node_records: Sequence[
        Gemma3ModalGeneratorMultifragmentNodeRecord
    ],
    interaction_selection: ModalInteractionSelection,
    edgeless_graph: ModalGeneratorGraphPlan,
    compiler_pipeline: ModalCompilerPipeline,
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Build and fully validate one canonical source-safe payload."""

    fit_trace = _authenticated_copy(
        fit_prompt_trace,
        PromptModeTrace,
    )
    catalog = _authenticated_copy(
        parameter_catalog,
        NaturalMLPParameterGroupCatalog,
    )
    fisher = _authenticated_copy(
        fisher_coupling,
        GroupedVirtualGateFisher,
    )
    clusters = _authenticated_copy(
        parameter_clusters,
        FisherPromptClusterPlan,
    )
    fragments = _authenticated_copy(
        parameter_cluster_fragments,
        ParameterClusterLayerFragmentPlan,
    )
    selection = _authenticated_copy(
        interaction_selection,
        ModalInteractionSelection,
    )
    edgeless = _authenticated_copy(
        edgeless_graph,
        ModalGeneratorGraphPlan,
    )
    pipeline = _authenticated_copy(
        compiler_pipeline,
        ModalCompilerPipeline,
    )
    records = _coerce_node_records(node_records)
    for value, label in (
        (scientific_status, "scientific_status"),
        (model, "model"),
        (protocol, "protocol"),
        (splits, "splits"),
        (upstream_metadata, "upstream_metadata"),
        (evaluation, "evaluation"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping")
    resources = _resource_accounting(
        model=model,
        pipeline=pipeline,
        edgeless_graph=edgeless,
        node_records=records,
    )
    without_digest: dict[str, object] = {
        "schema": GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA,
        "format_version": (
            GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION
        ),
        "scientific_status": dict(scientific_status),
        "model": dict(model),
        "protocol": dict(protocol),
        "splits": dict(splits),
        "upstream_metadata": dict(upstream_metadata),
        "fit_prompt_trace": fit_trace.state_dict(),
        "parameter_catalog": catalog.state_dict(),
        "fisher_coupling": fisher.state_dict(),
        "parameter_clusters": clusters.state_dict(),
        "parameter_cluster_fragments": fragments.state_dict(),
        "node_records": tuple(record.state_dict() for record in records),
        "interaction_selection": selection.state_dict(),
        "edgeless_graph": edgeless.state_dict(),
        "compiler_pipeline": pipeline.state_dict(),
        "resource_accounting": resources,
        "evaluation": dict(evaluation),
        "safety": dict(_SAFETY),
    }
    _assert_source_safe(without_digest)
    payload = {
        **without_digest,
        "scientific_payload_sha256": _payload_sha256(without_digest),
    }
    _validate_and_restore_payload(payload)
    return payload


def build_gemma3_modal_generator_multifragment_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    """Create a compact JSON report from an already strict payload."""

    restored = dict(payload)
    (
        _,
        _,
        _,
        _,
        _,
        records,
        selection,
        edgeless,
        pipeline,
    ) = _validate_and_restore_payload(restored)
    _require_name(tensor_file, label="tensor_file")
    report_without_digest: dict[str, object] = {
        "schema": GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA,
        "format_version": (
            GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION
        ),
        "scientific_status": restored["scientific_status"],
        "model": restored["model"],
        "protocol": restored["protocol"],
        "splits": restored["splits"],
        "upstream_metadata": restored["upstream_metadata"],
        "nodes": tuple(record.metadata() for record in records),
        "interaction_selection": selection.metadata(),
        "interaction_graph": pipeline.graph_plan.metadata(),
        "edgeless_graph": edgeless.metadata(),
        "compiler_pipeline": pipeline.metadata(),
        "resource_accounting": restored["resource_accounting"],
        "evaluation": restored["evaluation"],
        "artifact": {
            "tensor_file": tensor_file,
            "scientific_payload_sha256": restored[
                "scientific_payload_sha256"
            ],
            "safety": dict(_SAFETY),
        },
    }
    _assert_source_safe(report_without_digest)
    return {
        **report_without_digest,
        "report_sha256": _json_sha256(
            report_without_digest,
            domain=_REPORT_DOMAIN,
        ),
    }


def save_gemma3_modal_generator_multifragment_artifact(
    output: Path | str,
    **payload_arguments: object,
) -> dict[str, object]:
    """Build, save without overwrite, and report one development artifact."""

    path = Path(output)
    if path.suffix != ".pt":
        raise ValueError("multifragment artifact output must use .pt")
    report_path = path.with_suffix(".json")
    if path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite multifragment artifact")
    payload = build_gemma3_modal_generator_multifragment_payload(
        **payload_arguments,  # type: ignore[arg-type]
    )
    report = build_gemma3_modal_generator_multifragment_report(
        payload,
        tensor_file=path.name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            torch.save(payload, handle)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        if path.exists() and not report_path.exists():
            path.unlink()
        raise
    return report


def load_gemma3_modal_generator_multifragment_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load and cross-check every nested multifragment artifact."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("multifragment artifact must be a dict")
    _validate_and_restore_payload(raw)
    return raw
