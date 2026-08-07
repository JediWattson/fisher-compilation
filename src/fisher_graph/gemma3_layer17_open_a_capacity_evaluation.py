"""Open-development comparison for two layer-17 candidates.

This runner deliberately opens exactly one already-open data role:
``calibration_a_selection`` from the frozen v8 Calibration-A corpus.  It
compares two *edgeless* layer-17 executors against native Gemma and one
bit-exact matched-deletion control.  A legacy comparison may add capacity.  A
fixed-capacity adaptive comparison instead requires a separately authenticated,
passing family-LOFO report and a challenger fitted on all eight Calibration-A
fit families before this runner is allowed to open selection.  Guard,
Calibration-B, validation, and test are not accepted as inputs and cannot be
opened here.

The emitted JSON contains aggregate scalars, authenticated hashes, and exact
resource counts only.  Prompt text, family identifiers, token ids, logits,
and model/candidate tensors never cross the result boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_FORMAT_VERSION,
    GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_SCHEMA,
    GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION,
    GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA,
    Gemma3L3L4ProgressiveACorpusArtifact,
    Gemma3L3L4ProgressiveARolePrompts,
    gemma3_l3_l4_progressive_a_tokenizer_contract_sha256,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
    DEFAULT_SELECTION_OUTPUT,
    load_gemma3_layer10_v8_corpus_receipt,
)
from .gemma3_layer17_capped_node_fit import (
    GEMMA3_LAYER17_CAPPED_NODE_SCHEMA,
    default_gemma3_layer17_capped_node_output,
    restore_gemma3_layer17_capped_node_runtime,
)
from .gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
)
from .gemma3_layer17_v8_fit_lofo import (
    DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    load_gemma3_layer17_v8_fit_lofo_report,
)
from .gemma3_layer17_v8_all_family_refit import (
    DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT,
    GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
    load_gemma3_layer17_v8_all_family_refit_candidate,
    restore_gemma3_layer17_v8_all_family_refit_runtime,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_state_conditioned_modal_graph_artifact import (
    GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
    _validate_and_restore_payload,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _materialize_role,
    _tokenizer_contract,
)
from .modal_graph_rung_evaluation import (
    _GRAPH_LOGICAL_FIELDS,
    _GRAPH_STATIC_FIELDS,
    _assert_close_logits,
    _candidate_comparison,
    _execution_fields,
    _model_logits,
    _native_nll,
    _selected_logits_and_targets,
    _validate_graph_execution,
)


__all__ = [
    "DEFAULT_LAYER17_OPEN_A_OUTPUT",
    "DEFAULT_RANK16_CANDIDATE",
    "DEFAULT_RANK32_CANDIDATE",
    "DEFAULT_CAPPED_CAP48_CANDIDATE",
    "DEFAULT_CAPPED_CAP64_CANDIDATE",
    "DEFAULT_ADAPTIVE_A_FIT_CANDIDATE",
    "DEFAULT_ADAPTIVE_OPEN_A_OUTPUT",
    "evaluate_gemma3_layer17_open_a_capacity",
    "finalize_gemma3_layer17_open_a_prevalidation_checkpoint",
    "load_gemma3_layer17_open_a_capacity_result",
    "load_gemma3_layer17_open_a_prevalidation_checkpoint",
    "validate_gemma3_layer17_open_a_capacity_result",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_RANK16_CANDIDATE = (
    _LOCAL_ROOT / "state-conditioned-shape-flow-gain-dev-v2.pt"
)
DEFAULT_RANK32_CANDIDATE = (
    _LOCAL_ROOT / "layer17-shape-flow-gen32-open-dev-c-v1.pt"
)
DEFAULT_CAPPED_CAP48_CANDIDATE = default_gemma3_layer17_capped_node_output(
    48, 16
)
DEFAULT_CAPPED_CAP64_CANDIDATE = default_gemma3_layer17_capped_node_output(
    64, 16
)
DEFAULT_ADAPTIVE_A_FIT_CANDIDATE = (
    DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT
)
DEFAULT_LAYER17_OPEN_A_OUTPUT = (
    _LOCAL_ROOT / "layer17-open-a-capacity-evaluation-v1.json"
)
DEFAULT_ADAPTIVE_OPEN_A_OUTPUT = (
    _LOCAL_ROOT / "layer17-open-a-adaptive-refit-c48-r16-dev-v1.json"
)

_SCHEMA = "fisher_graph.gemma3_layer17_open_a_capacity_evaluation"
_FORMAT_VERSION = 1
_RESULT_DOMAIN = b"fisher-graph:gemma3-layer17-open-a-capacity:v1\0"
_PREVALIDATION_CHECKPOINT_DOMAIN = (
    b"fisher-graph:gemma3-layer17-open-a-prevalidation-checkpoint:v1\0"
)
_PROMPT_IDENTITY_DOMAIN = b"fisher-graph:open-a-selection-identities:v1\0"
_REMOVAL_SCOPE_DOMAIN = b"fisher-graph:layer17-removal-scope:v1\0"
_ADAPTIVE_GATE_POLICY_DOMAIN = (
    b"fisher-graph:layer17-fixed-capacity-adaptive-open-a-policy:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_EXPECTED_EXAMPLES = 128
_EXPECTED_FAMILIES = 4
_EXPECTED_MODE_RANK = 32
_VOCABULARY_CHUNK_SIZE = 16384
_MICRO_FAMILY_METRIC_IDENTITY_ULPS = 2
_MACRO_METRIC_IDENTITY_ULPS = 32
_CONDITIONS = ("rank16_edgeless", "rank32_edgeless", "matched_deletion")
_LOFO_PASS_NEXT_ACTION = (
    "freeze_full_eight_family_refit_then_replay_eligible_open_"
    "development_assessment"
)
_FIXED_CAPACITY_COMPARISON_KIND = "fixed_capacity_refit"
_CAPACITY_INCREASE_COMPARISON_KIND = "capacity_increase"
_ADAPTIVE_GATE_POLICY: dict[str, object] = {
    "policy_id": "layer17_fixed_capacity_adaptive_open_a_v1",
    "candidate_maximum_micro_delta_nll_per_token": 0.08,
    "candidate_maximum_equal_family_macro_delta_nll_per_token": 0.08,
    "candidate_maximum_equal_family_macro_native_kl_per_token": 0.09,
    "candidate_minimum_equal_family_macro_top1_agreement": 0.84,
    "candidate_maximum_family_delta_nll_per_token": 0.10,
    "minimum_passing_family_count": 3,
    "required_family_count": 4,
    "require_strict_micro_delta_nll_improvement_over_baseline": True,
    "require_strict_macro_delta_nll_improvement_over_baseline": True,
    "require_macro_native_kl_no_worse_than_baseline": True,
    "require_macro_top1_no_worse_than_baseline": True,
    "required_parameter_delta": 0,
    "required_graph_macs_per_token_delta": 0,
    "decision_role": "already_open_adaptive_development_selection",
    "heldout_confirmation": False,
}
_ROLE_INPUT_FIELDS = {
    "schema",
    "format_version",
    "corpus_id",
    "profile",
    "role",
    "prompts",
    "family_ids",
}
_PHYSICAL_SCOPE_FIELDS = (
    "replacement_scope",
    "replaced_layer_count",
    "fragment_count",
    "removed_mode_count",
    "source_whole_model_learned_parameters",
    "native_removed_learned_parameters",
)
_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_family_ids": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_model_or_candidate_weights": False,
    "fit_opened": False,
    "guard_opened": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "candidate_weights_mutated": False,
    "model_weights_mutated": False,
    "local_files_only": True,
    "source_safe": True,
}
_PREVALIDATION_CHECKPOINT_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_family_ids": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_model_or_candidate_weights": False,
    "source_safe": True,
    "selection_scoring_completed": True,
    "strict_result_validation_completed": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return _sha256_bytes(domain + _canonical_json_bytes(value))


_ADAPTIVE_GATE_POLICY_SHA256 = _domain_sha256(
    _ADAPTIVE_GATE_POLICY_DOMAIN,
    _ADAPTIVE_GATE_POLICY,
)


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _read_canonical_json(path: Path | str, *, label: str) -> tuple[dict[str, object], str]:
    source = Path(path)
    try:
        encoded = source.read_bytes()
        raw = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must contain one JSON object")
    if encoded != _canonical_json_bytes(raw):
        raise ValueError(f"{label} is not canonical JSON")
    return raw, _sha256_bytes(encoded)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish one canonical JSON object without overwriting.

    The temporary file lives beside the destination, so the hard-link commit
    cannot expose a partially written result.  The linked inode and containing
    directory are synchronized before success is reported.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(value)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite {path.name}"
            ) from error
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _CandidateAuthority:
    label: str
    path: Path
    file_sha256: str
    binding: dict[str, object]
    topology: tuple[tuple[object, ...], ...]
    removal_scope: dict[str, object]
    lowerings: tuple[tuple[str, object], ...]
    edgeless_graph: object
    private_metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _SelectionAuthority:
    role: Gemma3L3L4ProgressiveARolePrompts
    binding: dict[str, object]


@dataclass(frozen=True, slots=True)
class _RoleSlice:
    prompts: tuple[str, ...]
    ordered_prompt_sha256s: tuple[str, ...]


def _require_candidate_label(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CANDIDATE_LABEL.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be lowercase snake-case with at most 32 characters"
        )
    if value == "matched_deletion":
        raise ValueError(f"{label} conflicts with the deletion control")
    return value


def _candidate_conditions(
    baseline_label: str,
    challenger_label: str,
) -> tuple[str, str, str]:
    baseline = _require_candidate_label(baseline_label, label="baseline label")
    challenger = _require_candidate_label(
        challenger_label,
        label="challenger label",
    )
    if baseline == challenger:
        raise ValueError("capacity candidate labels must be distinct")
    return (f"{baseline}_edgeless", f"{challenger}_edgeless", "matched_deletion")


def _candidate_authority(
    candidate_path: Path | str,
    *,
    label: str,
    expected_generator_rank: int | None = None,
) -> _CandidateAuthority:
    candidate_label = _require_candidate_label(label, label="candidate label")
    path = Path(candidate_path)
    file_sha256 = _file_sha256(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"{candidate_label} candidate must contain a dict")
    artifact_schema = raw.get("schema")
    if artifact_schema == GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA:
        restored, edgeless, _, _ = _validate_and_restore_payload(raw)
        lowerings_by_node = dict(restored)
        candidate_kind = "legacy_state_conditioned_edgeless_control"
    elif artifact_schema == GEMMA3_LAYER17_CAPPED_NODE_SCHEMA:
        edgeless, lowerings_by_node, _ = (
            restore_gemma3_layer17_capped_node_runtime(raw)
        )
        candidate_kind = "capped_node_edgeless_candidate"
    elif artifact_schema == GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA:
        # Use the public strict loader as the authoritative file boundary, then
        # restore only from its validated in-memory value.
        raw = load_gemma3_layer17_v8_all_family_refit_candidate(path)
        edgeless, lowerings_by_node, _ = (
            restore_gemma3_layer17_v8_all_family_refit_runtime(raw)
        )
        candidate_kind = "v8_all_family_fixed_capacity_refit_candidate"
    else:
        raise ValueError(f"{candidate_label} uses an unsupported candidate schema")
    experiment = raw.get("experiment")
    config = raw.get("config")
    if not isinstance(experiment, Mapping) or not isinstance(config, Mapping):
        raise TypeError(f"{candidate_label} candidate metadata is invalid")
    assert isinstance(experiment, Mapping)
    assert isinstance(config, Mapping)
    if artifact_schema == GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA:
        selection = raw.get("selection")
        fragment = config.get("fragment_selection")
        if not isinstance(selection, Mapping):
            raise TypeError(f"{candidate_label} selection metadata is invalid")
        mode_rank_cap = config.get("selected_mode_rank")
        generator_rank = config.get("selected_generator_rank")
        if (
            selection.get("guard_opened") is not False
            or experiment.get("guard_status") != "sealed_unopened"
        ):
            raise ValueError(f"{candidate_label} legacy candidate opened its guard")
    elif artifact_schema == GEMMA3_LAYER17_CAPPED_NODE_SCHEMA:
        fragment = raw.get("fragment_selection")
        safety = raw.get("safety")
        mode_rank_cap = config.get("mode_rank_cap")
        generator_rank = config.get("generator_rank")
        if (
            config.get("edge_policy") != "edgeless"
            or not isinstance(safety, Mapping)
            or safety.get("guard_used") is not False
            or safety.get("calibration_b_used") is not False
            or safety.get("validation_used") is not False
            or safety.get("test_used") is not False
            or safety.get("heldout_confirmation") is not False
        ):
            raise ValueError(
                f"{candidate_label} capped candidate violates source safety"
            )
    else:
        fragment = raw.get("fragment_selection")
        safety = raw.get("safety")
        mode_rank_cap = config.get("mode_rank_cap")
        generator_rank = config.get("generator_rank")
        if (
            config.get("edge_policy") != "edgeless"
            or config.get("interaction_count") != 0
            or not isinstance(safety, Mapping)
            or safety.get("fit_opened") is not True
            or safety.get("selection_opened") is not False
            or safety.get("guard_opened") is not False
            or safety.get("calibration_b_opened") is not False
            or safety.get("validation_opened") is not False
            or safety.get("test_opened") is not False
            or experiment.get("heldout_confirmation") is not False
            or experiment.get("serving_authorized") is not False
            or experiment.get("assessment_metrics_present") is not False
        ):
            raise ValueError(
                f"{candidate_label} all-family refit violates source safety"
            )
    if not isinstance(fragment, Mapping):
        raise TypeError(f"{candidate_label} fragment selection is unavailable")
    if type(mode_rank_cap) is not int or mode_rank_cap <= 0:
        raise ValueError(f"{candidate_label} mode-rank cap is invalid")
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError(f"{candidate_label} generator rank is invalid")
    if expected_generator_rank is not None and generator_rank != expected_generator_rank:
        raise ValueError(f"{candidate_label} generator rank differs from expectation")
    lowerings = tuple(
        (name, lowerings_by_node[name]) for name in edgeless.traversal_order
    )
    resolved_node_ranks = tuple(
        lowering.computational_mode_basis.rank for _, lowering in lowerings
    )
    resolved_generator_ranks = tuple(
        lowering.coordinate_generator_plan.rank for _, lowering in lowerings
    )
    if (
        fragment.get("layer_ordinal") != 17
        or experiment.get("heldout_confirmation") is not False
        or len(edgeless.nodes) != 4
        or edgeless.interactions
        or tuple(name for name, _ in lowerings) != edgeless.traversal_order
        or len(resolved_node_ranks) != 4
        or any(rank <= 0 or rank > mode_rank_cap for rank in resolved_node_ranks)
        or resolved_generator_ranks != (generator_rank,) * 4
    ):
        raise ValueError(
            f"{candidate_label} is not a frozen edgeless layer-17 candidate"
        )
    declared_resolved = config.get("resolved_node_ranks")
    if artifact_schema in {
        GEMMA3_LAYER17_CAPPED_NODE_SCHEMA,
        GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
    } and tuple(declared_resolved) != resolved_node_ranks:  # type: ignore[arg-type]
        raise ValueError(f"{candidate_label} resolved node ranks drifted")
    revision = experiment.get("requested_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"{candidate_label} requested revision is invalid")
    fragment_ids = fragment.get("fragment_ids")
    execution_hashes = fragment.get("execution_order_sha256s")
    removed_indices = fragment.get("removed_mode_indices")
    if (
        not isinstance(fragment_ids, Sequence)
        or isinstance(fragment_ids, (str, bytes))
        or not isinstance(execution_hashes, Sequence)
        or isinstance(execution_hashes, (str, bytes))
        or not isinstance(removed_indices, Sequence)
        or isinstance(removed_indices, (str, bytes))
        or len(fragment_ids) != 4
        or len(execution_hashes) != 4
        or any(not isinstance(value, str) or not value for value in fragment_ids)
        or any(_SHA256.fullmatch(value) is None for value in execution_hashes)
        or any(type(value) is not int or value < 0 for value in removed_indices)
    ):
        raise ValueError(f"{candidate_label} fragment removal scope is malformed")
    selected_fragment_sha256s = tuple(
        lowering.selected_fragment_sha256 for _, lowering in lowerings
    )
    if selected_fragment_sha256s != tuple(execution_hashes):
        raise ValueError(
            f"{candidate_label} lowerings differ from fragment execution order"
        )
    topology = tuple(
        (node.name, node.causal_order, node.input_boundary, node.output_boundary)
        for node in edgeless.nodes
    )
    removal_scope: dict[str, object] = {
        "layer_ordinal": 17,
        "fragment_ids": tuple(fragment_ids),
        "execution_order_sha256s": tuple(execution_hashes),
        "removed_mode_indices": tuple(removed_indices),
        "source_fragment_plan_sha256": _require_sha256(
            fragment.get("source_fragment_plan_sha256"),
            label=f"{candidate_label} source fragment plan",
        ),
        "source_model_sha256": _require_sha256(
            fragment.get("source_model_sha256"),
            label=f"{candidate_label} source model",
        ),
    }
    binding: dict[str, object] = {
        "candidate_role": candidate_label,
        "candidate_kind": candidate_kind,
        "candidate_artifact_schema": artifact_schema,
        "tensor_file": path.name,
        "tensor_file_sha256": file_sha256,
        "scientific_payload_sha256": _require_sha256(
            raw.get("scientific_payload_sha256"),
            label=f"{candidate_label} scientific payload",
        ),
        "model_id": experiment.get("model_id"),
        "requested_revision": revision,
        "model_fingerprint": _require_sha256(
            experiment.get("adapter_model_fingerprint"),
            label=f"{candidate_label} model fingerprint",
        ),
        "parameter_cluster_plan_sha256": _require_sha256(
            edgeless.parameter_cluster_plan_sha256,
            label=f"{candidate_label} parameter cluster plan",
        ),
        "edgeless_graph_sha256": _require_sha256(
            edgeless.artifact_sha256,
            label=f"{candidate_label} edgeless graph",
        ),
        "mode_rank": mode_rank_cap,
        "mode_rank_cap": mode_rank_cap,
        "resolved_node_ranks": resolved_node_ranks,
        "generator_rank": generator_rank,
        "node_count": len(edgeless.nodes),
        "interaction_count": 0,
        "graph_parameters": edgeless.parameter_count,
        "graph_macs_per_token": edgeless.macs_per_token,
        "removal_scope_sha256": _domain_sha256(
            _REMOVAL_SCOPE_DOMAIN,
            removal_scope,
        ),
    }
    if artifact_schema == GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA:
        lineage = raw.get("lineage")
        fit_receipt = raw.get("fit_receipt")
        if not isinstance(lineage, Mapping) or not isinstance(
            fit_receipt, Mapping
        ):
            raise TypeError("all-family refit provenance is unavailable")
        binding["refit_provenance"] = {
            "lofo_report_file_sha256": _require_sha256(
                lineage.get("lofo_report_file_sha256"),
                label="all-family LOFO report file",
            ),
            "lofo_report_sha256": _require_sha256(
                lineage.get("lofo_report_sha256"),
                label="all-family LOFO report",
            ),
            "lofo_protocol_sha256": _require_sha256(
                lineage.get("lofo_protocol_sha256"),
                label="all-family LOFO protocol",
            ),
            "lofo_authority_sha256": _require_sha256(
                lineage.get("lofo_authority_sha256"),
                label="all-family LOFO authority",
            ),
            "frozen_v9_candidate_file_sha256": _require_sha256(
                lineage.get("frozen_v9_candidate_file_sha256"),
                label="all-family frozen-v9 file",
            ),
            "frozen_v9_candidate_scientific_sha256": _require_sha256(
                lineage.get("frozen_v9_candidate_scientific_sha256"),
                label="all-family frozen-v9 scientific payload",
            ),
            "fit_split_sha256": _require_sha256(
                fit_receipt.get("fit_split_sha256"),
                label="all-family fit split",
            ),
            "diagnostic_split_sha256": _require_sha256(
                fit_receipt.get("diagnostic_split_sha256"),
                label="all-family diagnostic split",
            ),
            "fit_family_count": fit_receipt.get("family_count"),
            "fit_example_count": fit_receipt.get("fit_example_count"),
            "full_eight_family_refit_completed": experiment.get(
                "full_eight_family_refit_completed"
            ),
            "diagnostic_subset_within_fit": fit_receipt.get(
                "diagnostic_subset_within_fit"
            ),
            "diagnostic_used_for_selection": fit_receipt.get(
                "diagnostic_used_for_selection"
            ),
        }
    return _CandidateAuthority(
        label=candidate_label,
        path=path,
        file_sha256=file_sha256,
        binding=binding,
        topology=topology,
        removal_scope=removal_scope,
        lowerings=lowerings,  # type: ignore[arg-type]
        edgeless_graph=edgeless,
        private_metadata={
            "experiment": dict(experiment),
            "splits": (
                dict(raw["splits"])
                if isinstance(raw.get("splits"), Mapping)
                else None
            ),
            "lineage": (
                dict(raw["lineage"])
                if isinstance(raw.get("lineage"), Mapping)
                else None
            ),
            "authority": (
                dict(raw["authority"])
                if isinstance(raw.get("authority"), Mapping)
                else None
            ),
            "protocol": (
                dict(raw["protocol"])
                if isinstance(raw.get("protocol"), Mapping)
                else None
            ),
            "fit_receipt": (
                dict(raw["fit_receipt"])
                if isinstance(raw.get("fit_receipt"), Mapping)
                else None
            ),
            "fit_collection": (
                dict(raw["fit_collection"])
                if isinstance(raw.get("fit_collection"), Mapping)
                else None
            ),
        },
    )


def _validate_candidate_pair(
    baseline: _CandidateAuthority,
    challenger: _CandidateAuthority,
) -> dict[str, object]:
    for field in (
        "model_id",
        "requested_revision",
        "model_fingerprint",
        "parameter_cluster_plan_sha256",
        "node_count",
        "interaction_count",
        "removal_scope_sha256",
    ):
        if baseline.binding.get(field) != challenger.binding.get(field):
            raise ValueError(f"candidate pair differs in {field}")
    if baseline.topology != challenger.topology:
        raise ValueError("candidate pair has different fragment topology")
    if baseline.removal_scope != challenger.removal_scope:
        raise ValueError("candidate pair has different native removal scope")
    parameter_delta = int(challenger.binding["graph_parameters"]) - int(
        baseline.binding["graph_parameters"]
    )
    mac_delta = int(challenger.binding["graph_macs_per_token"]) - int(
        baseline.binding["graph_macs_per_token"]
    )
    if parameter_delta < 0 or mac_delta < 0:
        raise ValueError("challenger cannot reduce only one declared resource")
    if (parameter_delta == 0) != (mac_delta == 0):
        raise ValueError("candidate parameter and MAC deltas must have one class")
    if parameter_delta == 0:
        comparison_kind = _FIXED_CAPACITY_COMPARISON_KIND
        for field in (
            "mode_rank_cap",
            "resolved_node_ranks",
            "generator_rank",
        ):
            if baseline.binding.get(field) != challenger.binding.get(field):
                raise ValueError(
                    f"fixed-capacity candidate pair differs in {field}"
                )
        baseline_file = _require_sha256(
            baseline.binding.get("tensor_file_sha256", baseline.file_sha256),
            label="baseline tensor file",
        )
        challenger_file = _require_sha256(
            challenger.binding.get(
                "tensor_file_sha256", challenger.file_sha256
            ),
            label="challenger tensor file",
        )
        baseline_scientific = _require_sha256(
            baseline.binding.get("scientific_payload_sha256"),
            label="baseline scientific payload",
        )
        challenger_scientific = _require_sha256(
            challenger.binding.get("scientific_payload_sha256"),
            label="challenger scientific payload",
        )
        if baseline_file == challenger_file:
            raise ValueError(
                "fixed-capacity challenger must be a distinct tensor artifact"
            )
        if baseline_scientific == challenger_scientific:
            raise ValueError(
                "fixed-capacity challenger must have distinct scientific content"
            )
    elif parameter_delta > 0 and mac_delta > 0:
        comparison_kind = _CAPACITY_INCREASE_COMPARISON_KIND
    else:  # Defensive: the paired zero/nonzero cases are rejected above.
        raise ValueError("challenger does not define a supported comparison")
    return {
        "baseline_label": baseline.label,
        "challenger_label": challenger.label,
        "comparison_kind": comparison_kind,
        "identical_model": True,
        "identical_fragment_topology": True,
        "identical_native_removal_scope": True,
        "both_graphs_edgeless": True,
        "graph_parameter_delta": parameter_delta,
        "graph_macs_per_token_delta": mac_delta,
        f"{challenger.label}_added_graph_parameters": parameter_delta,
        f"{challenger.label}_added_graph_macs_per_token": mac_delta,
    }


def _authorize_fixed_capacity_adaptive_selection(
    *,
    lofo_report_path: Path | str,
    baseline: _CandidateAuthority,
    challenger: _CandidateAuthority,
) -> dict[str, object]:
    """Authenticate LOFO and the full-A refit before selection can be read."""

    if (
        baseline.binding.get("candidate_artifact_schema")
        != GEMMA3_LAYER17_CAPPED_NODE_SCHEMA
        or challenger.binding.get("candidate_artifact_schema")
        != GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
    ):
        raise ValueError(
            "fixed-capacity adaptive selection requires frozen-v9 baseline "
            "and strict all-family refit challenger schemas"
        )
    report_path = Path(lofo_report_path)
    report = load_gemma3_layer17_v8_fit_lofo_report(report_path)
    decision = report.get("decision")
    protocol = report.get("protocol")
    authority = report.get("authority")
    lineage = report.get("lineage")
    scope = report.get("scope")
    safety = report.get("safety")
    if not all(
        isinstance(value, Mapping)
        for value in (decision, protocol, authority, lineage, scope, safety)
    ):
        raise TypeError("LOFO authorization sections are unavailable")
    assert isinstance(decision, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(authority, Mapping)
    assert isinstance(lineage, Mapping)
    assert isinstance(scope, Mapping)
    assert isinstance(safety, Mapping)
    if (
        decision.get("all_required_gates_pass") is not True
        or decision.get("next_action") != _LOFO_PASS_NEXT_ACTION
        or decision.get("protocol_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or protocol.get("artifact_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or report.get("heldout_confirmation") is not False
        or report.get("serving_authorized") is not False
        or report.get("compression_claim") is not False
        or scope.get("compiled_layer_count") != 1
        or scope.get("compiled_layer_ordinal") != 17
        or scope.get("whole_model_compiled") is not False
        or safety.get("selection_opened") is not False
        or safety.get("guard_opened") is not False
        or safety.get("calibration_b_opened") is not False
        or safety.get("validation_opened") is not False
        or safety.get("test_opened") is not False
    ):
        raise ValueError("LOFO report does not authorize adaptive selection")
    report_sha256 = _require_sha256(
        report.get("report_sha256"),
        label="LOFO report",
    )
    report_file_sha256 = _file_sha256(report_path)
    authority_sha256 = _require_sha256(
        authority.get("authority_sha256"),
        label="LOFO authority",
    )
    baseline_file_sha256 = _require_sha256(
        baseline.binding.get("tensor_file_sha256"),
        label="baseline tensor file",
    )
    baseline_scientific_sha256 = _require_sha256(
        baseline.binding.get("scientific_payload_sha256"),
        label="baseline scientific payload",
    )
    if (
        lineage.get("frozen_v9_candidate_file_sha256")
        != baseline_file_sha256
        or lineage.get("frozen_v9_candidate_scientific_sha256")
        != baseline_scientific_sha256
    ):
        raise ValueError("LOFO report does not bind the supplied frozen baseline")

    metadata = challenger.private_metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("adaptive challenger private metadata is unavailable")
    experiment = metadata.get("experiment")
    challenger_lineage = metadata.get("lineage")
    fit_receipt = metadata.get("fit_receipt")
    challenger_protocol = metadata.get("protocol")
    challenger_authority = metadata.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (
            experiment,
            challenger_lineage,
            fit_receipt,
            challenger_protocol,
            challenger_authority,
        )
    ):
        raise TypeError("adaptive challenger provenance is incomplete")
    assert isinstance(experiment, Mapping)
    assert isinstance(challenger_lineage, Mapping)
    assert isinstance(fit_receipt, Mapping)
    assert isinstance(challenger_protocol, Mapping)
    assert isinstance(challenger_authority, Mapping)
    if (
        experiment.get("experiment_kind")
        != "gemma3_layer17_v8_fit_all_family_refit_v1"
        or experiment.get("scientific_role")
        != "calibration_a_fit_all_family_refit_candidate"
        or experiment.get("full_eight_family_refit_completed") is not True
        or experiment.get("fit_family_count") != 8
        or experiment.get("fit_example_count") != 256
        or experiment.get("selection_opened") is not False
        or experiment.get("heldout_confirmation") is not False
        or experiment.get("assessment_metrics_present") is not False
        or experiment.get("serving_authorized") is not False
        or experiment.get("lofo_report_sha256") != report_sha256
        or experiment.get("lofo_protocol_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or experiment.get("lofo_authority_sha256") != authority_sha256
        or experiment.get("frozen_v9_candidate_file_sha256")
        != baseline_file_sha256
        or experiment.get("frozen_v9_candidate_scientific_sha256")
        != baseline_scientific_sha256
        or challenger_lineage.get("lofo_report_file_sha256")
        != report_file_sha256
        or challenger_lineage.get("lofo_report_sha256") != report_sha256
        or challenger_lineage.get("lofo_protocol_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or challenger_lineage.get("lofo_all_required_gates_pass") is not True
        or challenger_lineage.get("lofo_authorized_next_action")
        != _LOFO_PASS_NEXT_ACTION
        or challenger_lineage.get("lofo_authority_sha256")
        != authority_sha256
        or challenger_lineage.get("frozen_v9_candidate_file_sha256")
        != baseline_file_sha256
        or challenger_lineage.get(
            "frozen_v9_candidate_scientific_sha256"
        )
        != baseline_scientific_sha256
        or challenger_protocol.get("artifact_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or challenger_protocol.get("authorized_by_passing_outer_lofo")
        is not True
        or challenger_authority.get("authority_sha256") != authority_sha256
        or fit_receipt.get("family_count") != 8
        or fit_receipt.get("fit_example_count") != 256
        or fit_receipt.get("all_coefficients_fit_on_all_normalized_rows")
        is not True
        or fit_receipt.get("diagnostic_subset_within_fit") is not True
        or fit_receipt.get("diagnostic_used_for_selection") is not False
        or fit_receipt.get("diagnostic_supports_assessment_claim") is not False
    ):
        raise ValueError(
            "adaptive challenger does not bind the authorized full-A refit"
        )
    fit_split_sha256 = _require_sha256(
        fit_receipt.get("fit_split_sha256"),
        label="all-family fit split",
    )
    diagnostic_split_sha256 = _require_sha256(
        fit_receipt.get("diagnostic_split_sha256"),
        label="all-family diagnostic split",
    )
    challenger_file_sha256 = _require_sha256(
        challenger.binding.get("tensor_file_sha256"),
        label="challenger tensor file",
    )
    challenger_scientific_sha256 = _require_sha256(
        challenger.binding.get("scientific_payload_sha256"),
        label="challenger scientific payload",
    )
    refit_provenance = challenger.binding.get("refit_provenance")
    expected_refit_provenance = {
        "lofo_report_file_sha256": report_file_sha256,
        "lofo_report_sha256": report_sha256,
        "lofo_protocol_sha256": FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
        "lofo_authority_sha256": authority_sha256,
        "frozen_v9_candidate_file_sha256": baseline_file_sha256,
        "frozen_v9_candidate_scientific_sha256": baseline_scientific_sha256,
        "fit_split_sha256": fit_split_sha256,
        "diagnostic_split_sha256": diagnostic_split_sha256,
        "fit_family_count": 8,
        "fit_example_count": 256,
        "full_eight_family_refit_completed": True,
        "diagnostic_subset_within_fit": True,
        "diagnostic_used_for_selection": False,
    }
    if refit_provenance != expected_refit_provenance:
        raise ValueError("public challenger refit provenance does not cross-bind")
    return {
        "authorization_kind": "passing_lofo_then_full_eight_family_refit",
        "selection_access_authorized": True,
        "authorization_completed_before_selection_open": True,
        "lofo_report_file": report_path.name,
        "lofo_report_file_sha256": report_file_sha256,
        "lofo_report_sha256": report_sha256,
        "lofo_protocol_sha256": FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
        "lofo_authority_sha256": authority_sha256,
        "lofo_completed_fold_count": 8,
        "lofo_all_required_gates_pass": True,
        "lofo_authorized_next_action": _LOFO_PASS_NEXT_ACTION,
        "baseline_tensor_file_sha256": baseline_file_sha256,
        "baseline_scientific_payload_sha256": baseline_scientific_sha256,
        "challenger_tensor_file_sha256": challenger_file_sha256,
        "challenger_scientific_payload_sha256": challenger_scientific_sha256,
        "full_eight_family_refit_completed": True,
        "fit_family_count": 8,
        "fit_example_count": 256,
        "fit_split_sha256": fit_split_sha256,
        "diagnostic_split_sha256": diagnostic_split_sha256,
        "diagnostic_subset_within_fit": True,
        "diagnostic_used_for_selection": False,
        "claim_role": "already_open_adaptive_development_selection",
        "heldout_confirmation": False,
        "serving_authorized": False,
        "compression_claim": False,
        "source_safe": True,
    }


def _load_open_selection_authority(
    *,
    corpus_artifact_path: Path | str,
    selection_path: Path | str,
    receipt_path: Path | str,
) -> _SelectionAuthority:
    """Open only the selection role and authenticate it against prompt-free authorities."""

    receipt = load_gemma3_layer10_v8_corpus_receipt(receipt_path)
    artifact_raw, artifact_file_sha256 = _read_canonical_json(
        corpus_artifact_path,
        label="v8 Calibration-A corpus artifact",
    )
    if (
        artifact_raw.get("schema") != GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_SCHEMA
        or artifact_raw.get("format_version")
        != GEMMA3_L3_L4_PROGRESSIVE_A_CORPUS_FORMAT_VERSION
    ):
        raise ValueError("v8 corpus artifact header is invalid")
    artifact = Gemma3L3L4ProgressiveACorpusArtifact.from_dict(artifact_raw)
    expected_tokenizer_sha256 = (
        gemma3_l3_l4_progressive_a_tokenizer_contract_sha256(
            _tokenizer_contract()
        )
    )
    if artifact.tokenizer_contract_sha256 != expected_tokenizer_sha256:
        raise ValueError("v8 selection tokenizer contract drifted")

    role_raw, role_file_sha256 = _read_canonical_json(
        selection_path,
        label="v8 Calibration-A selection role",
    )
    if set(role_raw) != _ROLE_INPUT_FIELDS or (
        role_raw.get("schema") != GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_SCHEMA
        or role_raw.get("format_version")
        != GEMMA3_L3_L4_PROGRESSIVE_A_ROLE_FORMAT_VERSION
        or role_raw.get("corpus_id") != artifact.corpus_id
        or role_raw.get("profile") != artifact.profile
        or role_raw.get("role") != "calibration_a_selection"
    ):
        raise ValueError("v8 Calibration-A selection role header is invalid")
    prompts = role_raw.get("prompts")
    family_ids = role_raw.get("family_ids")
    if not isinstance(prompts, list) or not isinstance(family_ids, list):
        raise TypeError("v8 selection prompts and families must be JSON arrays")
    role = Gemma3L3L4ProgressiveARolePrompts(
        corpus_id=artifact.corpus_id,
        profile=artifact.profile,
        role="calibration_a_selection",
        prompts=tuple(prompts),
        family_ids=tuple(family_ids),
        source_file_sha256=role_file_sha256,
    )
    view = artifact.role_view("calibration_a_selection")
    receipt_corpus = receipt.get("corpus")
    receipt_roles = receipt.get("roles")
    receipt_role = (
        receipt_roles.get("calibration_a_selection")
        if isinstance(receipt_roles, Mapping)
        else None
    )
    if not isinstance(receipt_corpus, Mapping) or not isinstance(
        receipt_role, Mapping
    ):
        raise TypeError("v8 receipt selection bindings are unavailable")
    if (
        len(role.prompts) != _EXPECTED_EXAMPLES
        or len(set(role.family_ids)) != _EXPECTED_FAMILIES
        or view.example_count != _EXPECTED_EXAMPLES
        or len(view.family_ids) != _EXPECTED_FAMILIES
        or role.source_file_sha256 != view.role_input_file_sha256
        or role.ordered_prompt_sha256s != view.ordered_prompt_sha256s
        or role.family_ids != view.ordered_family_ids
        or receipt_corpus.get("artifact_sha256") != artifact.artifact_sha256
        or receipt_corpus.get("artifact_file_sha256") != artifact_file_sha256
        or receipt_corpus.get("tokenizer_contract_sha256")
        != artifact.tokenizer_contract_sha256
        or receipt_role.get("manifest_sha256") != view.manifest_sha256
        or receipt_role.get("role_input_file_sha256") != role_file_sha256
        or receipt_role.get("example_count") != _EXPECTED_EXAMPLES
        or receipt_role.get("family_count") != _EXPECTED_FAMILIES
    ):
        raise ValueError("v8 receipt, corpus, and selection role disagree")
    binding = {
        "corpus_artifact_file": Path(corpus_artifact_path).name,
        "corpus_artifact_file_sha256": artifact_file_sha256,
        "corpus_artifact_sha256": artifact.artifact_sha256,
        "receipt_file": Path(receipt_path).name,
        "receipt_file_sha256": _file_sha256(receipt_path),
        "receipt_sha256": _require_sha256(
            receipt.get("receipt_sha256"), label="v8 receipt"
        ),
        "selection_role_file": Path(selection_path).name,
        "selection_role_file_sha256": role_file_sha256,
        "selection_manifest_sha256": view.manifest_sha256,
        "ordered_membership_sha256": _domain_sha256(
            _PROMPT_IDENTITY_DOMAIN,
            role.ordered_prompt_sha256s,
        ),
        "tokenizer_contract_sha256": artifact.tokenizer_contract_sha256,
        "example_count": _EXPECTED_EXAMPLES,
        "family_count": _EXPECTED_FAMILIES,
        "assessment_role": "already_open_calibration_a_selection",
    }
    return _SelectionAuthority(role=role, binding=binding)


def _materialize_selection_families(
    tokenizer: object,
    authority: _SelectionAuthority,
    *,
    device: torch.device,
    tokenization_batch_size: int,
) -> tuple[
    tuple[tuple[str, tuple[CalibrationBatch, ...]], ...],
    dict[str, object],
]:
    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    role = authority.role
    family_order = tuple(dict.fromkeys(role.family_ids))
    if len(family_order) != _EXPECTED_FAMILIES:
        raise ValueError("selection role must contain exactly four families")
    materialized: list[tuple[str, tuple[CalibrationBatch, ...]]] = []
    observed_ids: list[str] = []
    stream_hashes: list[str] = []
    total_valid = 0
    total_supervised = 0
    max_length = int(_tokenizer_contract()["max_length"])
    for family_index, family_id in enumerate(family_order):
        selected = tuple(
            (prompt, prompt_sha256)
            for prompt, prompt_sha256, observed_family in zip(
                role.prompts,
                role.ordered_prompt_sha256s,
                role.family_ids,
                strict=True,
            )
            if observed_family == family_id
        )
        if not selected:
            raise RuntimeError("selection family is empty")
        family_role = _RoleSlice(
            prompts=tuple(value[0] for value in selected),
            ordered_prompt_sha256s=tuple(value[1] for value in selected),
        )
        batches, stream = _materialize_role(
            tokenizer,
            family_role,  # type: ignore[arg-type]
            split_name=f"layer17_open_a_family_{family_index:02d}",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        batch_ids = tuple(
            example_id
            for batch in batches
            for example_id in (
                batch.example_ids if batch.example_ids is not None else ()
            )
        )
        if any(batch.example_ids is None for batch in batches) or (
            batch_ids != family_role.ordered_prompt_sha256s
        ):
            raise RuntimeError("selection family tokenization membership drifted")
        serialized_sha256 = _require_sha256(
            stream.get("serialized_sha256"),
            label="selection tokenized stream",
        )
        stream_hashes.append(serialized_sha256)
        total_valid += sum(
            int(batch.valid_positions.sum().item()) for batch in batches
        )
        total_supervised += sum(
            int((batch.targets != -100).sum().item()) for batch in batches
        )
        observed_ids.extend(batch_ids)
        materialized.append((f"family_{family_index:02d}", batches))
    if tuple(observed_ids) != tuple(
        identity
        for family_id in family_order
        for identity, observed_family in zip(
            role.ordered_prompt_sha256s,
            role.family_ids,
            strict=True,
        )
        if observed_family == family_id
    ) or set(observed_ids) != set(role.ordered_prompt_sha256s):
        raise RuntimeError("selection family materialization is incomplete")
    tokenization = {
        "family_stream_count": _EXPECTED_FAMILIES,
        "family_stream_catalog_sha256": _domain_sha256(
            b"fisher-graph:layer17-open-a-tokenized-streams:v1\0",
            tuple(stream_hashes),
        ),
        "example_count": _EXPECTED_EXAMPLES,
        "logical_valid_tokens": total_valid,
        "supervised_tokens": total_supervised,
        "max_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "contains_prompt_text": False,
        "contains_prompt_identities": False,
        "contains_token_ids": False,
    }
    return tuple(materialized), tokenization


def _new_metric_accumulator(
    conditions: Sequence[str] = _CONDITIONS,
) -> dict[str, object]:
    names = tuple(conditions)
    if not names or len(names) != len(set(names)):
        raise ValueError("metric condition catalog is invalid")
    return {
        "supervised_tokens": 0,
        "native_nll_sum": 0.0,
        "conditions": {
            name: {
                "nll_sum": 0.0,
                "native_to_candidate_kl_sum": 0.0,
                "top1_matches": 0,
            }
            for name in names
        },
    }


def _add_native(
    accumulator: dict[str, object],
    *,
    nll_sum: float,
    token_count: int,
) -> None:
    accumulator["supervised_tokens"] = int(
        accumulator["supervised_tokens"]
    ) + token_count
    accumulator["native_nll_sum"] = float(
        accumulator["native_nll_sum"]
    ) + nll_sum


def _add_comparison(
    accumulator: dict[str, object],
    name: str,
    comparison: Mapping[str, float | int],
) -> None:
    raw = accumulator.get("conditions")
    if not isinstance(raw, dict) or not isinstance(raw.get(name), dict):
        raise TypeError("condition metric accumulator is unavailable")
    totals = raw[name]
    totals["nll_sum"] = float(totals["nll_sum"]) + float(
        comparison["nll_sum"]
    )
    totals["native_to_candidate_kl_sum"] = float(
        totals["native_to_candidate_kl_sum"]
    ) + float(comparison["native_to_candidate_kl_sum"])
    totals["top1_matches"] = int(totals["top1_matches"]) + int(
        comparison["top1_matches"]
    )


def _finalize_metric_accumulator(
    accumulator: Mapping[str, object],
    *,
    conditions: Sequence[str] = _CONDITIONS,
) -> dict[str, object]:
    names = tuple(conditions)
    tokens = accumulator.get("supervised_tokens")
    raw_conditions = accumulator.get("conditions")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("metric accumulator has no supervised tokens")
    if not isinstance(raw_conditions, Mapping) or set(raw_conditions) != set(
        names
    ):
        raise ValueError("metric accumulator condition catalog differs")
    native_nll = _finite(
        accumulator.get("native_nll_sum"), label="native NLL sum"
    ) / tokens
    conditions: dict[str, object] = {}
    for name in names:
        totals = raw_conditions[name]
        if not isinstance(totals, Mapping):
            raise TypeError(f"{name} metric totals are invalid")
        nll = _finite(totals.get("nll_sum"), label=f"{name} NLL sum")
        kl = _finite(
            totals.get("native_to_candidate_kl_sum"),
            label=f"{name} KL sum",
        )
        top1 = totals.get("top1_matches")
        if type(top1) is not int or not 0 <= top1 <= tokens:
            raise ValueError(f"{name} top-1 total is invalid")
        nll_per_token = nll / tokens
        conditions[name] = {
            "nll_per_token": nll_per_token,
            "delta_nll_per_token": nll_per_token - native_nll,
            "native_to_candidate_kl_per_token": kl / tokens,
            "top1_agreement_to_native": top1 / tokens,
        }
    return {
        "supervised_tokens": tokens,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
    }


def _equal_family_macro(
    families: Mapping[str, Mapping[str, object]],
    *,
    conditions: Sequence[str] = _CONDITIONS,
) -> dict[str, object]:
    condition_names = tuple(conditions)
    if len(families) != _EXPECTED_FAMILIES:
        raise ValueError("equal-family macro requires exactly four families")
    native = sum(
        _finite(
            family["native"]["nll_per_token"],  # type: ignore[index]
            label="family native NLL",
        )
        for family in families.values()
    ) / _EXPECTED_FAMILIES
    metrics = (
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    )
    conditions: dict[str, object] = {}
    for name in condition_names:
        conditions[name] = {
            metric: sum(
                _finite(
                    family["conditions"][name][metric],  # type: ignore[index]
                    label=f"family {name} {metric}",
                )
                for family in families.values()
            )
            / _EXPECTED_FAMILIES
            for metric in metrics
        }
    return {"native": {"nll_per_token": native}, "conditions": conditions}


def _adaptive_gate_row(
    gate: str,
    *,
    observed: bool | int | float,
    operator: str,
    threshold: bool | int | float,
    passed: bool,
) -> dict[str, object]:
    return {
        "gate": gate,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _evaluate_fixed_capacity_adaptive_gates(
    *,
    assessment: Mapping[str, object],
    candidate_pair: Mapping[str, object],
    baseline_label: str,
    challenger_label: str,
) -> dict[str, object]:
    """Replay the outcome-independent adaptive-A policy on open selection.

    These absolute and comparative thresholds are module constants, so the
    policy exists before the selection authority is opened.  The result is an
    open-development decision only and never a heldout or serving claim.
    """

    if candidate_pair.get("comparison_kind") != _FIXED_CAPACITY_COMPARISON_KIND:
        raise ValueError("adaptive gates require a fixed-capacity refit pair")
    conditions = assessment.get("conditions")
    macro = assessment.get("equal_family_macro")
    families = assessment.get("families")
    macro_conditions = macro.get("conditions") if isinstance(macro, Mapping) else None
    if (
        not isinstance(conditions, Mapping)
        or not isinstance(macro_conditions, Mapping)
        or not isinstance(families, Mapping)
        or len(families) != int(_ADAPTIVE_GATE_POLICY["required_family_count"])
    ):
        raise ValueError("adaptive assessment metrics are incomplete")
    baseline_condition, challenger_condition, _ = _candidate_conditions(
        baseline_label,
        challenger_label,
    )
    baseline_micro = conditions.get(baseline_condition)
    challenger_micro = conditions.get(challenger_condition)
    baseline_macro = macro_conditions.get(baseline_condition)
    challenger_macro = macro_conditions.get(challenger_condition)
    if not all(
        isinstance(row, Mapping)
        for row in (
            baseline_micro,
            challenger_micro,
            baseline_macro,
            challenger_macro,
        )
    ):
        raise TypeError("adaptive baseline or challenger metrics are unavailable")
    assert isinstance(baseline_micro, Mapping)
    assert isinstance(challenger_micro, Mapping)
    assert isinstance(baseline_macro, Mapping)
    assert isinstance(challenger_macro, Mapping)
    baseline_micro_delta = _finite(
        baseline_micro.get("delta_nll_per_token"),
        label="adaptive baseline micro delta NLL",
    )
    challenger_micro_delta = _finite(
        challenger_micro.get("delta_nll_per_token"),
        label="adaptive challenger micro delta NLL",
    )
    baseline_macro_delta = _finite(
        baseline_macro.get("delta_nll_per_token"),
        label="adaptive baseline macro delta NLL",
    )
    challenger_macro_delta = _finite(
        challenger_macro.get("delta_nll_per_token"),
        label="adaptive challenger macro delta NLL",
    )
    baseline_macro_kl = _finite(
        baseline_macro.get("native_to_candidate_kl_per_token"),
        label="adaptive baseline macro KL",
    )
    challenger_macro_kl = _finite(
        challenger_macro.get("native_to_candidate_kl_per_token"),
        label="adaptive challenger macro KL",
    )
    baseline_macro_top1 = _finite(
        baseline_macro.get("top1_agreement_to_native"),
        label="adaptive baseline macro top1",
    )
    challenger_macro_top1 = _finite(
        challenger_macro.get("top1_agreement_to_native"),
        label="adaptive challenger macro top1",
    )
    family_limit = float(
        _ADAPTIVE_GATE_POLICY[
            "candidate_maximum_family_delta_nll_per_token"
        ]
    )
    passing_family_count = 0
    for family_name, family in families.items():
        family_conditions = (
            family.get("conditions") if isinstance(family, Mapping) else None
        )
        row = (
            family_conditions.get(challenger_condition)
            if isinstance(family_conditions, Mapping)
            else None
        )
        if not isinstance(row, Mapping):
            raise TypeError(f"{family_name} adaptive challenger metrics are invalid")
        delta = _finite(
            row.get("delta_nll_per_token"),
            label=f"{family_name} adaptive challenger delta NLL",
        )
        passing_family_count += int(delta <= family_limit)
    parameter_delta = candidate_pair.get("graph_parameter_delta")
    mac_delta = candidate_pair.get("graph_macs_per_token_delta")
    if type(parameter_delta) is not int or type(mac_delta) is not int:
        raise ValueError("adaptive resource deltas must be exact integers")

    micro_limit = float(
        _ADAPTIVE_GATE_POLICY[
            "candidate_maximum_micro_delta_nll_per_token"
        ]
    )
    macro_limit = float(
        _ADAPTIVE_GATE_POLICY[
            "candidate_maximum_equal_family_macro_delta_nll_per_token"
        ]
    )
    kl_limit = float(
        _ADAPTIVE_GATE_POLICY[
            "candidate_maximum_equal_family_macro_native_kl_per_token"
        ]
    )
    top1_floor = float(
        _ADAPTIVE_GATE_POLICY[
            "candidate_minimum_equal_family_macro_top1_agreement"
        ]
    )
    family_floor = int(_ADAPTIVE_GATE_POLICY["minimum_passing_family_count"])
    gate_table = (
        _adaptive_gate_row(
            "candidate_micro_delta_nll_per_token",
            observed=challenger_micro_delta,
            operator="<=",
            threshold=micro_limit,
            passed=challenger_micro_delta <= micro_limit,
        ),
        _adaptive_gate_row(
            "candidate_equal_family_macro_delta_nll_per_token",
            observed=challenger_macro_delta,
            operator="<=",
            threshold=macro_limit,
            passed=challenger_macro_delta <= macro_limit,
        ),
        _adaptive_gate_row(
            "candidate_equal_family_macro_native_kl_per_token",
            observed=challenger_macro_kl,
            operator="<=",
            threshold=kl_limit,
            passed=challenger_macro_kl <= kl_limit,
        ),
        _adaptive_gate_row(
            "candidate_equal_family_macro_top1_agreement",
            observed=challenger_macro_top1,
            operator=">=",
            threshold=top1_floor,
            passed=challenger_macro_top1 >= top1_floor,
        ),
        _adaptive_gate_row(
            "candidate_family_delta_nll_pass_count",
            observed=passing_family_count,
            operator=">=",
            threshold=family_floor,
            passed=passing_family_count >= family_floor,
        ),
        _adaptive_gate_row(
            "strict_micro_delta_nll_improvement_over_baseline",
            observed=challenger_micro_delta - baseline_micro_delta,
            operator="<",
            threshold=0.0,
            passed=challenger_micro_delta < baseline_micro_delta,
        ),
        _adaptive_gate_row(
            "strict_macro_delta_nll_improvement_over_baseline",
            observed=challenger_macro_delta - baseline_macro_delta,
            operator="<",
            threshold=0.0,
            passed=challenger_macro_delta < baseline_macro_delta,
        ),
        _adaptive_gate_row(
            "macro_native_kl_no_worse_than_baseline",
            observed=challenger_macro_kl - baseline_macro_kl,
            operator="<=",
            threshold=0.0,
            passed=challenger_macro_kl <= baseline_macro_kl,
        ),
        _adaptive_gate_row(
            "macro_top1_no_worse_than_baseline",
            observed=challenger_macro_top1 - baseline_macro_top1,
            operator=">=",
            threshold=0.0,
            passed=challenger_macro_top1 >= baseline_macro_top1,
        ),
        _adaptive_gate_row(
            "exact_graph_parameter_delta",
            observed=parameter_delta,
            operator="==",
            threshold=0,
            passed=parameter_delta == 0,
        ),
        _adaptive_gate_row(
            "exact_graph_macs_per_token_delta",
            observed=mac_delta,
            operator="==",
            threshold=0,
            passed=mac_delta == 0,
        ),
    )
    passed = all(bool(row["passed"]) for row in gate_table)
    return {
        "assessment_role": "already_open_adaptive_development_selection",
        "heldout_confirmation": False,
        "policy": {
            **_ADAPTIVE_GATE_POLICY,
            "artifact_sha256": _ADAPTIVE_GATE_POLICY_SHA256,
        },
        "passing_family_count": passing_family_count,
        "gate_table": list(gate_table),
        "all_required_gates_pass": passed,
        "adaptive_candidate_selected": passed,
        "next_action": (
            "retain_adaptive_candidate_for_open_development_only"
            if passed
            else "reject_adaptive_candidate_keep_frozen_v9"
        ),
        "heldout_or_serving_authorized": False,
    }


def _record_resources(
    resources: dict[str, dict[str, object]],
    logical_totals: dict[str, dict[str, int]],
    peak_widths: dict[str, int],
    *,
    name: str,
    execution: object,
) -> dict[str, object]:
    static = _execution_fields(execution, _GRAPH_STATIC_FIELDS, label=name)
    prior = resources.setdefault(name, static)
    if prior != static:
        raise RuntimeError(f"{name} static accounting changed by batch")
    totals = logical_totals.setdefault(
        name, {field: 0 for field in _GRAPH_LOGICAL_FIELDS}
    )
    for field in _GRAPH_LOGICAL_FIELDS:
        value = getattr(execution, field, None)
        if type(value) is not int:
            raise ValueError(f"{name} {field} must be an integer")
        totals[field] += value
    peak = getattr(execution, "peak_live_modal_width", None)
    if type(peak) is not int or peak < 0:
        raise ValueError(f"{name} peak width is invalid")
    peak_widths[name] = max(peak_widths.get(name, 0), peak)
    return static


def _require_zero_deletion_work(execution: object, *, label: str) -> None:
    if any(
        getattr(execution, field, None) != 0
        for field in (
            "logical_executed_modal_graph_macs",
            "logical_executed_modal_graph_additions",
            "peak_live_modal_width",
        )
    ):
        raise RuntimeError(f"{label} executed modal graph work")


def _exact_resource_summary(
    *,
    name: str,
    plan: object,
    static: Mapping[str, object],
    totals: Mapping[str, int],
    logical_valid_tokens: int,
    peak_width: int,
) -> dict[str, object]:
    graph_parameters = getattr(plan, "parameter_count", None)
    graph_macs = getattr(plan, "macs_per_token", None)
    accounting = getattr(plan, "accounting", None)
    graph_additions = getattr(accounting, "elementwise_additions_per_token", None)
    if any(type(value) is not int for value in (graph_parameters, graph_macs, graph_additions)):
        raise TypeError(f"{name} graph lacks exact static accounting")
    if (
        static["modal_graph_learned_parameters"] != graph_parameters
        or totals["logical_modal_graph_macs"]
        != graph_macs * logical_valid_tokens
        or totals["logical_modal_graph_additions"]
        != graph_additions * logical_valid_tokens
    ):
        raise RuntimeError(f"{name} exact graph accounting drifted")
    executed_macs = totals["logical_executed_modal_graph_macs"]
    executed_additions = totals["logical_executed_modal_graph_additions"]
    if name == "matched_deletion":
        if executed_macs or executed_additions or peak_width:
            raise RuntimeError("matched deletion has nonzero graph execution")
    elif (
        executed_macs != graph_macs * logical_valid_tokens
        or executed_additions != graph_additions * logical_valid_tokens
    ):
        raise RuntimeError(f"{name} edgeless execution accounting drifted")
    removed = int(static["native_removed_learned_parameters"])
    if totals["logical_linear_macs_native_removed"] != removed * logical_valid_tokens:
        raise RuntimeError(f"{name} native removed MAC accounting drifted")
    expected_net = removed - (0 if name == "matched_deletion" else graph_macs)
    if totals["net_logical_macs_saved"] != expected_net * logical_valid_tokens:
        raise RuntimeError(f"{name} net MAC accounting drifted")
    return {
        **static,
        **totals,
        "executed_peak_live_modal_width": peak_width,
        "graph_macs_per_token": graph_macs,
        "graph_additions_per_token": graph_additions,
        "native_removed_macs_per_token": removed,
        "executed_graph_macs_per_token": (
            0 if name == "matched_deletion" else graph_macs
        ),
        "net_logical_macs_saved_per_token": expected_net,
    }


def _score_capacity_panel_in_transaction(
    *,
    adapter: Gemma3CausalLMAdapter,
    rank16_executor: Gemma3ModalGeneratorGraphExecutor,
    rank32_executor: Gemma3ModalGeneratorGraphExecutor,
    family_batches: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
    baseline_label: str = "rank16",
    challenger_label: str = "rank32",
) -> dict[str, object]:
    condition_names = _candidate_conditions(baseline_label, challenger_label)
    baseline_condition, challenger_condition, deletion_condition = condition_names
    native_model = adapter.module
    if not callable(native_model):
        raise TypeError("adapter does not expose a callable native model")
    rank16_plan = rank16_executor.graph_plan
    rank32_plan = rank32_executor.graph_plan
    if rank16_plan.interactions or rank32_plan.interactions:
        raise ValueError("capacity comparison requires two edgeless graphs")
    if len(family_batches) != _EXPECTED_FAMILIES or tuple(
        name for name, _ in family_batches
    ) != tuple(f"family_{index:02d}" for index in range(_EXPECTED_FAMILIES)):
        raise ValueError("capacity family batch catalog is invalid")
    example_ids = tuple(
        example_id
        for _, batches in family_batches
        for batch in batches
        for example_id in (
            batch.example_ids if batch.example_ids is not None else ()
        )
    )
    if (
        any(batch.example_ids is None for _, batches in family_batches for batch in batches)
        or len(example_ids) != _EXPECTED_EXAMPLES
        or len(set(example_ids)) != _EXPECTED_EXAMPLES
    ):
        raise ValueError("capacity batches must contain 128 unique examples")

    aggregate = _new_metric_accumulator(condition_names)
    family_accumulators = {
        name: _new_metric_accumulator(condition_names)
        for name, _ in family_batches
    }
    resources: dict[str, dict[str, object]] = {}
    logical_totals: dict[str, dict[str, int]] = {}
    peak_widths: dict[str, int] = {}
    logical_valid_tokens = 0
    deletion_max_abs = 0.0

    for family_name, batches in family_batches:
        if not batches or any(not isinstance(batch, CalibrationBatch) for batch in batches):
            raise ValueError(f"{family_name} contains invalid batches")
        family_accumulator = family_accumulators[family_name]
        for batch in batches:
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output), batch
            )
            token_count = targets.numel()
            native_nll_sum = _native_nll(native_logits, targets)
            _add_native(aggregate, nll_sum=native_nll_sum, token_count=token_count)
            _add_native(
                family_accumulator,
                nll_sum=native_nll_sum,
                token_count=token_count,
            )

            generated_executions: dict[str, object] = {}
            generated_static: dict[str, dict[str, object]] = {}
            for name, executor, plan in (
                (baseline_condition, rank16_executor, rank16_plan),
                (challenger_condition, rank32_executor, rank32_plan),
            ):
                with torch.no_grad():
                    execution = executor.run(batch.model_inputs, condition="generated")
                _validate_graph_execution(
                    execution, plan, condition="generated", label=name
                )
                logits, candidate_targets = _selected_logits_and_targets(
                    _model_logits(execution.model_output), batch
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{name} evaluation targets drifted")
                comparison = _candidate_comparison(
                    native_logits,
                    logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                )
                _add_comparison(aggregate, name, comparison)
                _add_comparison(family_accumulator, name, comparison)
                generated_static[name] = _record_resources(
                    resources,
                    logical_totals,
                    peak_widths,
                    name=name,
                    execution=execution,
                )
                generated_executions[name] = execution

            deletion_logits: dict[str, Tensor] = {}
            deletion_executions: dict[str, object] = {}
            for name, executor, plan in (
                (baseline_label, rank16_executor, rank16_plan),
                (challenger_label, rank32_executor, rank32_plan),
            ):
                with torch.no_grad():
                    execution = executor.run(batch.model_inputs, condition="deletion")
                _validate_graph_execution(
                    execution,
                    plan,
                    condition="deletion",
                    label=f"{name} deletion",
                )
                logits, candidate_targets = _selected_logits_and_targets(
                    _model_logits(execution.model_output), batch
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{name} deletion targets drifted")
                _require_zero_deletion_work(execution, label=f"{name} deletion")
                deletion_logits[name] = logits
                deletion_executions[name] = execution

            comparison = _candidate_comparison(
                native_logits,
                deletion_logits[baseline_label],
                targets,
                vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
            )
            _add_comparison(aggregate, deletion_condition, comparison)
            _add_comparison(family_accumulator, deletion_condition, comparison)
            deletion_static = _record_resources(
                resources,
                logical_totals,
                peak_widths,
                name=deletion_condition,
                execution=deletion_executions[baseline_label],
            )
            deletion_max_abs = max(
                deletion_max_abs,
                _assert_close_logits(
                    deletion_logits[baseline_label],
                    deletion_logits[challenger_label],
                    atol=0.0,
                    rtol=0.0,
                    label=(
                        f"{baseline_label}/{challenger_label} matched deletion"
                    ),
                ),
            )
            rank32_deletion_static = _execution_fields(
                deletion_executions[challenger_label],
                _GRAPH_STATIC_FIELDS,
                label=f"{challenger_label} deletion",
            )
            if deletion_static != generated_static[baseline_condition]:
                raise RuntimeError(
                    f"{baseline_label} generated/deletion static accounting differs"
                )
            if rank32_deletion_static != generated_static[challenger_condition]:
                raise RuntimeError(
                    f"{challenger_label} generated/deletion static accounting differs"
                )
            for field in _PHYSICAL_SCOPE_FIELDS:
                if generated_static[baseline_condition][field] != generated_static[
                    challenger_condition
                ][field]:
                    raise RuntimeError(f"candidate physical scope differs in {field}")
            expected_valid = int(batch.valid_positions.sum().item())
            observed_valid = {
                getattr(execution, "valid_tokens", None)
                for execution in (
                    *generated_executions.values(),
                    *deletion_executions.values(),
                )
            }
            if observed_valid != {expected_valid}:
                raise RuntimeError("graph conditions disagree on valid tokens")
            logical_valid_tokens += expected_valid

    micro = _finalize_metric_accumulator(
        aggregate,
        conditions=condition_names,
    )
    families = {
        name: _finalize_metric_accumulator(
            accumulator,
            conditions=condition_names,
        )
        for name, accumulator in family_accumulators.items()
    }
    macro = _equal_family_macro(families, conditions=condition_names)
    if logical_valid_tokens <= 0:
        raise RuntimeError("capacity panel has no logical valid tokens")

    resource_output = {
        baseline_condition: _exact_resource_summary(
            name=baseline_condition,
            plan=rank16_plan,
            static=resources[baseline_condition],
            totals=logical_totals[baseline_condition],
            logical_valid_tokens=logical_valid_tokens,
            peak_width=peak_widths[baseline_condition],
        ),
        challenger_condition: _exact_resource_summary(
            name=challenger_condition,
            plan=rank32_plan,
            static=resources[challenger_condition],
            totals=logical_totals[challenger_condition],
            logical_valid_tokens=logical_valid_tokens,
            peak_width=peak_widths[challenger_condition],
        ),
        deletion_condition: _exact_resource_summary(
            name=deletion_condition,
            plan=rank16_plan,
            static=resources[deletion_condition],
            totals=logical_totals[deletion_condition],
            logical_valid_tokens=logical_valid_tokens,
            peak_width=peak_widths[deletion_condition],
        ),
    }
    rank16_metrics = micro["conditions"][baseline_condition]  # type: ignore[index]
    rank32_metrics = micro["conditions"][challenger_condition]  # type: ignore[index]
    capacity_delta = {
        f"{challenger_label}_minus_{baseline_label}_nll_per_token": float(
            rank32_metrics["nll_per_token"] - rank16_metrics["nll_per_token"]
        ),
        f"{challenger_label}_minus_{baseline_label}_native_kl_per_token": float(
            rank32_metrics["native_to_candidate_kl_per_token"]
            - rank16_metrics["native_to_candidate_kl_per_token"]
        ),
        f"{challenger_label}_minus_{baseline_label}_top1_agreement": float(
            rank32_metrics["top1_agreement_to_native"]
            - rank16_metrics["top1_agreement_to_native"]
        ),
        f"{challenger_label}_added_graph_parameters": int(
            resource_output[challenger_condition]["modal_graph_learned_parameters"]
        )
        - int(resource_output[baseline_condition]["modal_graph_learned_parameters"]),
        f"{challenger_label}_added_graph_macs_per_token": int(
            resource_output[challenger_condition]["graph_macs_per_token"]
        )
        - int(resource_output[baseline_condition]["graph_macs_per_token"]),
    }
    return {
        "execution_path": "paired_layer17_edgeless_modal_graph_executors",
        "assessment_role": "open_development_capacity_comparison",
        "heldout_confirmation": False,
        "example_count": _EXPECTED_EXAMPLES,
        "family_count": _EXPECTED_FAMILIES,
        "supervised_tokens": micro["supervised_tokens"],
        "logical_valid_tokens": logical_valid_tokens,
        "native": micro["native"],
        "conditions": micro["conditions"],
        "equal_family_macro": macro,
        "families": families,
        "capacity_delta": capacity_delta,
        "graph_comparison": {
            f"{baseline_label}_node_count": len(rank16_plan.nodes),
            f"{challenger_label}_node_count": len(rank32_plan.nodes),
            f"{baseline_label}_edge_count": 0,
            f"{challenger_label}_edge_count": 0,
            "deletion_paths_agree": True,
            "deletion_equivalence_atol": 0.0,
            "deletion_equivalence_rtol": 0.0,
            "deletion_max_abs_logit_difference": deletion_max_abs,
        },
        "resource_accounting": resource_output,
        "latency_or_kernel_speed_claim": False,
    }


def _score_capacity_panel(
    *,
    adapter: Gemma3CausalLMAdapter,
    rank16_executor: Gemma3ModalGeneratorGraphExecutor,
    rank32_executor: Gemma3ModalGeneratorGraphExecutor,
    family_batches: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
    baseline_label: str = "rank16",
    challenger_label: str = "rank32",
) -> dict[str, object]:
    if rank16_executor is rank32_executor:
        raise ValueError("capacity executors must be distinct")
    with ExitStack() as stack:
        stack.enter_context(rank16_executor.validated_transaction())
        stack.enter_context(rank32_executor.validated_transaction())
        return _score_capacity_panel_in_transaction(
            adapter=adapter,
            rank16_executor=rank16_executor,
            rank32_executor=rank32_executor,
            family_batches=family_batches,
            baseline_label=baseline_label,
            challenger_label=challenger_label,
        )


def _forbidden_output_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized.startswith("contains_"):
        return False
    return normalized in {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "prompt_sha256",
        "prompt_sha256s",
        "ordered_prompt_sha256s",
        "family_id",
        "family_ids",
        "ordered_family_ids",
        "input_ids",
        "token_ids",
        "tokens",
        "logits",
        "weights",
        "state_dict",
    }


def _reject_forbidden_output_fields(value: object, *, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if _forbidden_output_key(key):
                raise ValueError(f"{path}.{key} is a forbidden source field")
            _reject_forbidden_output_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_forbidden_output_fields(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite scalar")


def _metric_identity_within_ulps(
    actual: float,
    expected: float,
    *,
    operands: Sequence[float],
    maximum_ulps: int,
) -> tuple[bool, float]:
    if type(maximum_ulps) is not int or maximum_ulps < 0:
        raise ValueError("metric identity ULP allowance is invalid")
    if actual == expected:
        return True, 0.0
    scale = max(
        math.ulp(value)
        for value in (actual, expected, *operands)
    )
    tolerance = maximum_ulps * scale
    return abs(actual - expected) <= tolerance, tolerance


def _validate_metric_container(
    value: Mapping[str, object],
    *,
    label: str,
    conditions: Sequence[str] = _CONDITIONS,
    identity_ulps: int = _MICRO_FAMILY_METRIC_IDENTITY_ULPS,
) -> None:
    condition_names = tuple(conditions)
    native = value.get("native")
    conditions = value.get("conditions")
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError(f"{label} native metric is invalid")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        condition_names
    ):
        raise ValueError(f"{label} condition catalog is invalid")
    native_nll = _finite(
        native.get("nll_per_token"),
        label=f"{label} native NLL",
    )
    if native_nll < 0:
        raise ValueError(f"{label} native NLL must be nonnegative")
    expected = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    for name, record in conditions.items():
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError(f"{label} {name} metrics are invalid")
        nll = _finite(record.get("nll_per_token"), label=f"{label} {name} NLL")
        kl = _finite(
            record.get("native_to_candidate_kl_per_token"),
            label=f"{label} {name} KL",
        )
        top1 = _finite(
            record.get("top1_agreement_to_native"),
            label=f"{label} {name} top1",
        )
        delta = _finite(
            record.get("delta_nll_per_token"),
            label=f"{label} {name} delta NLL",
        )
        if nll < 0:
            raise ValueError(f"{label} {name} NLL {nll!r} is negative")
        if kl < 0:
            raise ValueError(f"{label} {name} KL {kl!r} is negative")
        if not 0 <= top1 <= 1:
            raise ValueError(
                f"{label} {name} top1 {top1!r} is outside [0, 1]"
            )
        expected_delta = nll - native_nll
        identity_matches, tolerance = _metric_identity_within_ulps(
            delta,
            expected_delta,
            operands=(nll, native_nll),
            maximum_ulps=identity_ulps,
        )
        if not identity_matches:
            raise ValueError(
                f"{label} {name} delta NLL {delta!r} contradicts "
                f"NLL-native {expected_delta!r} beyond "
                f"{identity_ulps} ULPs ({tolerance!r})"
            )


def _json_native_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Normalize tuple-bearing in-memory payloads through canonical JSON."""

    try:
        normalized = json.loads(_canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON data") from error
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must contain one JSON object")
    return normalized


_PREVALIDATION_CHECKPOINT_SCHEMA = (
    "fisher_graph.gemma3_layer17_open_a_prevalidation_checkpoint"
)
_PREVALIDATION_CHECKPOINT_FORMAT_VERSION = 1

_CHECKPOINT_RESULT_BASE_FIELDS = frozenset(
    {
        "schema",
        "format_version",
        "scientific_role",
        "heldout_confirmation",
        "candidates",
        "candidate_pair",
        "corpus",
        "runtime",
        "tokenization",
        "assessment",
        "candidate_changed",
        "candidate_tensor_file_sha256s_after",
        "selection_opened",
        "fit_opened",
        "guard_opened",
        "calibration_b_opened",
        "validation_opened",
        "test_opened",
        "safety",
        "result_sha256",
    }
)
_CHECKPOINT_RESULT_ADAPTIVE_FIELDS = _CHECKPOINT_RESULT_BASE_FIELDS | {
    "authorization",
    "adaptive_selection",
}
_CHECKPOINT_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_role",
        "candidate_kind",
        "candidate_artifact_schema",
        "tensor_file",
        "tensor_file_sha256",
        "scientific_payload_sha256",
        "model_id",
        "requested_revision",
        "model_fingerprint",
        "parameter_cluster_plan_sha256",
        "edgeless_graph_sha256",
        "mode_rank",
        "mode_rank_cap",
        "resolved_node_ranks",
        "generator_rank",
        "node_count",
        "interaction_count",
        "graph_parameters",
        "graph_macs_per_token",
        "removal_scope_sha256",
        "refit_provenance",
    }
)
_CHECKPOINT_REFIT_PROVENANCE_FIELDS = frozenset(
    {
        "lofo_report_file_sha256",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_authority_sha256",
        "frozen_v9_candidate_file_sha256",
        "frozen_v9_candidate_scientific_sha256",
        "fit_split_sha256",
        "diagnostic_split_sha256",
        "fit_family_count",
        "fit_example_count",
        "full_eight_family_refit_completed",
        "diagnostic_subset_within_fit",
        "diagnostic_used_for_selection",
    }
)
_CHECKPOINT_CORPUS_FIELDS = frozenset(
    {
        "corpus_artifact_file",
        "corpus_artifact_file_sha256",
        "corpus_artifact_sha256",
        "receipt_file",
        "receipt_file_sha256",
        "receipt_sha256",
        "selection_role_file",
        "selection_role_file_sha256",
        "selection_manifest_sha256",
        "ordered_membership_sha256",
        "tokenizer_contract_sha256",
        "example_count",
        "family_count",
        "assessment_role",
        # Accepted by the public legacy result validator and hash-only.
        "source_manifest_chain",
    }
)
_CHECKPOINT_RUNTIME_FIELDS = frozenset(
    {
        "model_id",
        "requested_revision",
        "model_fingerprint",
        "device",
        "dtype",
        "tokenization_batch_size",
        "max_length",
        "vocabulary_chunk_size",
        "local_files_only",
    }
)
_CHECKPOINT_TOKENIZATION_FIELDS = frozenset(
    {
        "family_stream_count",
        "family_stream_catalog_sha256",
        "example_count",
        "logical_valid_tokens",
        "supervised_tokens",
        "max_length",
        "tokenization_batch_size",
        "contains_prompt_text",
        "contains_prompt_identities",
        "contains_token_ids",
    }
)
_CHECKPOINT_ASSESSMENT_FIELDS = frozenset(
    {
        "execution_path",
        "assessment_role",
        "heldout_confirmation",
        "example_count",
        "family_count",
        "supervised_tokens",
        "logical_valid_tokens",
        "native",
        "conditions",
        "equal_family_macro",
        "families",
        "capacity_delta",
        "graph_comparison",
        "resource_accounting",
        "latency_or_kernel_speed_claim",
    }
)
_CHECKPOINT_METRIC_FIELDS = frozenset(
    {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
)
_CHECKPOINT_AUTHORIZATION_FIELDS = frozenset(
    {
        "authorization_kind",
        "selection_access_authorized",
        "authorization_completed_before_selection_open",
        "lofo_report_file",
        "lofo_report_file_sha256",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_authority_sha256",
        "lofo_completed_fold_count",
        "lofo_all_required_gates_pass",
        "lofo_authorized_next_action",
        "baseline_tensor_file_sha256",
        "baseline_scientific_payload_sha256",
        "challenger_tensor_file_sha256",
        "challenger_scientific_payload_sha256",
        "full_eight_family_refit_completed",
        "fit_family_count",
        "fit_example_count",
        "fit_split_sha256",
        "diagnostic_split_sha256",
        "diagnostic_subset_within_fit",
        "diagnostic_used_for_selection",
        "claim_role",
        "heldout_confirmation",
        "serving_authorized",
        "compression_claim",
        "source_safe",
    }
)


def _checkpoint_mapping(
    value: object,
    *,
    label: str,
    allowed: frozenset[str] | set[str],
    required: frozenset[str] | set[str] = frozenset(),
    exact: bool = False,
    nested: frozenset[str] | set[str] = frozenset(),
    scalar_sequences: frozenset[str] | set[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {label} must be a mapping")
    fields = set(value)
    expected = set(allowed)
    if (exact and fields != expected) or not fields <= expected:
        raise ValueError(f"checkpoint {label} fields are invalid")
    if not set(required) <= fields:
        raise ValueError(f"checkpoint {label} fields are incomplete")
    nested_fields = set(nested)
    scalar_sequence_fields = set(scalar_sequences)
    for field, child in value.items():
        if field in nested_fields:
            continue
        if field in scalar_sequence_fields:
            if isinstance(child, (str, bytes)) or not isinstance(
                child, Sequence
            ):
                raise TypeError(
                    f"checkpoint {label} {field} must be a scalar sequence"
                )
            if any(
                isinstance(item, Mapping)
                or (
                    isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes))
                )
                for item in child
            ):
                raise TypeError(
                    f"checkpoint {label} {field} contains nested data"
                )
            continue
        if isinstance(child, Mapping) or (
            isinstance(child, Sequence) and not isinstance(child, (str, bytes))
        ):
            raise TypeError(f"checkpoint {label} {field} must be scalar")
    return value


def _validate_checkpoint_metric_shape(
    value: object,
    *,
    label: str,
    conditions: Sequence[str],
    family: bool,
) -> None:
    expected = {"native", "conditions"}
    if family:
        expected.add("supervised_tokens")
    container = _checkpoint_mapping(
        value,
        label=label,
        allowed=expected,
        exact=True,
        nested={"native", "conditions"},
    )
    _checkpoint_mapping(
        container.get("native"),
        label=f"{label} native",
        allowed={"nll_per_token"},
        exact=True,
    )
    rows = _checkpoint_mapping(
        container.get("conditions"),
        label=f"{label} conditions",
        allowed=set(conditions),
        exact=True,
        nested=set(conditions),
    )
    for name, row in rows.items():
        _checkpoint_mapping(
            row,
            label=f"{label} {name}",
            allowed=_CHECKPOINT_METRIC_FIELDS,
            exact=True,
        )


def _validate_checkpoint_result_shape(result: Mapping[str, object]) -> None:
    """Close every persisted result section without replaying science checks."""

    top_fields = frozenset(result)
    if top_fields not in {
        _CHECKPOINT_RESULT_BASE_FIELDS,
        _CHECKPOINT_RESULT_ADAPTIVE_FIELDS,
    }:
        raise ValueError("checkpoint nested result fields are invalid")
    adaptive = top_fields == _CHECKPOINT_RESULT_ADAPTIVE_FIELDS
    nested_top_fields = {
        "candidates",
        "candidate_pair",
        "corpus",
        "runtime",
        "tokenization",
        "assessment",
        "candidate_tensor_file_sha256s_after",
        "safety",
    }
    if adaptive:
        nested_top_fields.update({"authorization", "adaptive_selection"})
    _checkpoint_mapping(
        result,
        label="nested result",
        allowed=set(top_fields),
        exact=True,
        nested=nested_top_fields,
    )
    if result.get("safety") != _SAFETY:
        raise ValueError("checkpoint nested result safety is invalid")

    candidates = _checkpoint_mapping(
        result.get("candidates"),
        label="candidate catalog",
        allowed=set(result.get("candidates", {})),
        nested=set(result.get("candidates", {})),
    )
    if len(candidates) != 2:
        raise ValueError("checkpoint candidate catalog is invalid")
    for name, raw_candidate in candidates.items():
        candidate = _checkpoint_mapping(
            raw_candidate,
            label=f"candidate {name}",
            allowed=_CHECKPOINT_CANDIDATE_FIELDS,
            required={
                "candidate_role",
                "mode_rank",
                "generator_rank",
                "node_count",
                "interaction_count",
                "tensor_file_sha256",
            },
            nested={"refit_provenance"},
            scalar_sequences={"resolved_node_ranks"},
        )
        provenance = candidate.get("refit_provenance")
        if provenance is not None:
            _checkpoint_mapping(
                provenance,
                label=f"candidate {name} refit provenance",
                allowed=_CHECKPOINT_REFIT_PROVENANCE_FIELDS,
                exact=True,
            )

    pair = _checkpoint_mapping(
        result.get("candidate_pair"),
        label="candidate pair",
        allowed={
            "baseline_label",
            "challenger_label",
            "comparison_kind",
            "identical_model",
            "identical_fragment_topology",
            "identical_native_removal_scope",
            "both_graphs_edgeless",
            "graph_parameter_delta",
            "graph_macs_per_token_delta",
            *(
                f"{label}_added_{resource}"
                for label in candidates
                for resource in ("graph_parameters", "graph_macs_per_token")
            ),
        },
        required={
            "identical_model",
            "identical_fragment_topology",
            "identical_native_removal_scope",
            "both_graphs_edgeless",
        },
    )
    baseline = pair.get("baseline_label")
    challenger = pair.get("challenger_label")
    if baseline is None and challenger is None and set(candidates) == {
        "rank16",
        "rank32",
    }:
        baseline, challenger = "rank16", "rank32"
    baseline_name = _require_candidate_label(
        baseline,
        label="checkpoint baseline label",
    )
    challenger_name = _require_candidate_label(
        challenger,
        label="checkpoint challenger label",
    )
    if set(candidates) != {baseline_name, challenger_name}:
        raise ValueError("checkpoint candidate labels differ")
    condition_names = _candidate_conditions(baseline_name, challenger_name)

    _checkpoint_mapping(
        result.get("corpus"),
        label="corpus",
        allowed=_CHECKPOINT_CORPUS_FIELDS,
        required={"example_count", "family_count", "assessment_role"},
        scalar_sequences={"source_manifest_chain"},
    )
    _checkpoint_mapping(
        result.get("runtime"),
        label="runtime",
        allowed=_CHECKPOINT_RUNTIME_FIELDS,
    )
    _checkpoint_mapping(
        result.get("tokenization"),
        label="tokenization",
        allowed=_CHECKPOINT_TOKENIZATION_FIELDS,
        required={"example_count", "family_stream_count"},
    )
    after = _checkpoint_mapping(
        result.get("candidate_tensor_file_sha256s_after"),
        label="candidate hashes after evaluation",
        allowed=set(candidates),
        exact=True,
    )
    if set(after) != set(candidates):
        raise ValueError("checkpoint candidate hashes after evaluation differ")

    assessment = _checkpoint_mapping(
        result.get("assessment"),
        label="assessment",
        allowed=_CHECKPOINT_ASSESSMENT_FIELDS,
        exact=True,
        nested={
            "native",
            "conditions",
            "equal_family_macro",
            "families",
            "capacity_delta",
            "graph_comparison",
            "resource_accounting",
        },
    )
    _validate_checkpoint_metric_shape(
        {
            "native": assessment.get("native"),
            "conditions": assessment.get("conditions"),
        },
        label="micro metrics",
        conditions=condition_names,
        family=False,
    )
    macro = assessment.get("equal_family_macro")
    _validate_checkpoint_metric_shape(
        macro,
        label="equal-family macro",
        conditions=condition_names,
        family=False,
    )
    family_names = {
        f"family_{index:02d}" for index in range(_EXPECTED_FAMILIES)
    }
    families = _checkpoint_mapping(
        assessment.get("families"),
        label="families",
        allowed=family_names,
        exact=True,
        nested=family_names,
    )
    for name, family_metrics in families.items():
        _validate_checkpoint_metric_shape(
            family_metrics,
            label=name,
            conditions=condition_names,
            family=True,
        )
    _checkpoint_mapping(
        assessment.get("capacity_delta"),
        label="capacity delta",
        allowed={
            f"{challenger_name}_minus_{baseline_name}_nll_per_token",
            f"{challenger_name}_minus_{baseline_name}_native_kl_per_token",
            f"{challenger_name}_minus_{baseline_name}_top1_agreement",
            f"{challenger_name}_added_graph_parameters",
            f"{challenger_name}_added_graph_macs_per_token",
        },
    )
    _checkpoint_mapping(
        assessment.get("graph_comparison"),
        label="graph comparison",
        allowed={
            f"{baseline_name}_node_count",
            f"{challenger_name}_node_count",
            f"{baseline_name}_edge_count",
            f"{challenger_name}_edge_count",
            "deletion_paths_agree",
            "deletion_equivalence_atol",
            "deletion_equivalence_rtol",
            "deletion_max_abs_logit_difference",
        },
    )
    resources = _checkpoint_mapping(
        assessment.get("resource_accounting"),
        label="resource accounting",
        allowed=set(condition_names),
        exact=True,
        nested=set(condition_names),
    )
    resource_fields = set(_GRAPH_STATIC_FIELDS) | set(_GRAPH_LOGICAL_FIELDS) | {
        "executed_peak_live_modal_width",
        "graph_macs_per_token",
        "graph_additions_per_token",
        "native_removed_macs_per_token",
        "executed_graph_macs_per_token",
        "net_logical_macs_saved_per_token",
    }
    for name, resource in resources.items():
        _checkpoint_mapping(
            resource,
            label=f"resource {name}",
            allowed=resource_fields,
        )

    if adaptive:
        _checkpoint_mapping(
            result.get("authorization"),
            label="authorization",
            allowed=_CHECKPOINT_AUTHORIZATION_FIELDS,
            exact=True,
        )
        selection = _checkpoint_mapping(
            result.get("adaptive_selection"),
            label="adaptive selection",
            allowed={
                "assessment_role",
                "heldout_confirmation",
                "policy",
                "passing_family_count",
                "gate_table",
                "all_required_gates_pass",
                "adaptive_candidate_selected",
                "next_action",
                "heldout_or_serving_authorized",
            },
            exact=True,
            nested={"policy", "gate_table"},
        )
        _checkpoint_mapping(
            selection.get("policy"),
            label="adaptive policy",
            allowed=set(_ADAPTIVE_GATE_POLICY) | {"artifact_sha256"},
            exact=True,
        )
        gate_table = selection.get("gate_table")
        if isinstance(gate_table, (str, bytes)) or not isinstance(
            gate_table, Sequence
        ):
            raise TypeError("checkpoint adaptive gate table is invalid")
        for index, gate in enumerate(gate_table):
            _checkpoint_mapping(
                gate,
                label=f"adaptive gate {index}",
                allowed={"gate", "observed", "operator", "threshold", "passed"},
                exact=True,
            )


def _prevalidation_checkpoint_path(final_output: Path | str) -> Path:
    destination = Path(final_output)
    return destination.with_name(
        f"{destination.stem}.prevalidation-checkpoint.json"
    )


def _validate_unvalidated_result_hash(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint unvalidated result must be a mapping")
    result = _json_native_mapping(
        value,
        label="checkpoint unvalidated result",
    )
    supplied = _require_sha256(
        result.get("result_sha256"),
        label="checkpoint result",
    )
    payload = {
        key: child
        for key, child in result.items()
        if key != "result_sha256"
    }
    if supplied != _domain_sha256(_RESULT_DOMAIN, payload):
        raise ValueError("checkpoint result hash mismatch")
    _reject_forbidden_output_fields(result, path="checkpoint.unvalidated_result")
    _validate_checkpoint_result_shape(result)
    return result


def _build_prevalidation_checkpoint(
    result: Mapping[str, object],
    *,
    final_output: Path | str,
) -> dict[str, object]:
    destination = Path(final_output)
    if destination.name in {"", ".", ".."} or destination.suffix != ".json":
        raise ValueError("final output must have a source-safe JSON basename")
    unvalidated = _validate_unvalidated_result_hash(result)
    payload: dict[str, object] = {
        "schema": _PREVALIDATION_CHECKPOINT_SCHEMA,
        "format_version": _PREVALIDATION_CHECKPOINT_FORMAT_VERSION,
        "status": "unvalidated",
        "scientific_role": "source_safe_prevalidation_recovery_only",
        "intended_final_output_file": destination.name,
        "unvalidated_result": unvalidated,
        "safety": dict(_PREVALIDATION_CHECKPOINT_SAFETY),
    }
    _reject_forbidden_output_fields(payload, path="checkpoint")
    checkpoint = {
        **payload,
        "checkpoint_sha256": _domain_sha256(
            _PREVALIDATION_CHECKPOINT_DOMAIN,
            payload,
        ),
    }
    return _validate_prevalidation_checkpoint(checkpoint)


def _validate_prevalidation_checkpoint(
    value: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = _json_native_mapping(
        value,
        label="layer17 open-A prevalidation checkpoint",
    )
    expected_fields = {
        "schema",
        "format_version",
        "status",
        "scientific_role",
        "intended_final_output_file",
        "unvalidated_result",
        "safety",
        "checkpoint_sha256",
    }
    if set(checkpoint) != expected_fields:
        raise ValueError("prevalidation checkpoint fields are invalid")
    supplied = _require_sha256(
        checkpoint.pop("checkpoint_sha256", None),
        label="prevalidation checkpoint",
    )
    if supplied != _domain_sha256(
        _PREVALIDATION_CHECKPOINT_DOMAIN,
        checkpoint,
    ):
        raise ValueError("prevalidation checkpoint hash mismatch")
    final_file = checkpoint.get("intended_final_output_file")
    if (
        checkpoint.get("schema") != _PREVALIDATION_CHECKPOINT_SCHEMA
        or checkpoint.get("format_version")
        != _PREVALIDATION_CHECKPOINT_FORMAT_VERSION
        or checkpoint.get("status") != "unvalidated"
        or checkpoint.get("scientific_role")
        != "source_safe_prevalidation_recovery_only"
        or not isinstance(final_file, str)
        or Path(final_file).name != final_file
        or Path(final_file).suffix != ".json"
        or checkpoint.get("safety") != _PREVALIDATION_CHECKPOINT_SAFETY
    ):
        raise ValueError("prevalidation checkpoint boundary is invalid")
    checkpoint["unvalidated_result"] = _validate_unvalidated_result_hash(
        checkpoint.get("unvalidated_result")
    )
    _reject_forbidden_output_fields(checkpoint, path="checkpoint")
    checkpoint["checkpoint_sha256"] = supplied
    return checkpoint


def load_gemma3_layer17_open_a_prevalidation_checkpoint(
    path: Path | str,
) -> dict[str, object]:
    raw, _ = _read_canonical_json(
        path,
        label="layer17 open-A prevalidation checkpoint",
    )
    return _validate_prevalidation_checkpoint(raw)


def _publish_with_prevalidation_checkpoint(
    result: Mapping[str, object],
    *,
    output: Path | str,
) -> dict[str, object]:
    destination = Path(output)
    if destination.suffix != ".json" or not destination.name:
        raise ValueError("open-A output must have a JSON basename")
    checkpoint_path = _prevalidation_checkpoint_path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination.name}")
    checkpoint = _build_prevalidation_checkpoint(
        result,
        final_output=destination,
    )
    _write_exclusive(checkpoint_path, checkpoint)
    try:
        validated = validate_gemma3_layer17_open_a_capacity_result(result)
        _write_exclusive(destination, validated)
    except BaseException:
        # The source-safe, explicitly unvalidated recovery envelope is the
        # durable handoff for a later validator fix.
        raise
    checkpoint_path.unlink()
    _fsync_directory(checkpoint_path.parent)
    return validated


def finalize_gemma3_layer17_open_a_prevalidation_checkpoint(
    checkpoint_path: Path | str,
    *,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Strict-validate and publish a surviving checkpoint without model I/O."""

    source = Path(checkpoint_path)
    checkpoint = load_gemma3_layer17_open_a_prevalidation_checkpoint(source)
    intended = str(checkpoint["intended_final_output_file"])
    intended_destination = source.with_name(intended)
    if source != _prevalidation_checkpoint_path(intended_destination):
        raise ValueError("recovery checkpoint path differs from intended output")
    destination = intended_destination if output is None else Path(output)
    if destination != intended_destination:
        raise ValueError("recovery output path differs from checkpoint")
    validated = validate_gemma3_layer17_open_a_capacity_result(
        checkpoint["unvalidated_result"]  # type: ignore[arg-type]
    )
    _write_exclusive(destination, validated)
    source.unlink()
    _fsync_directory(source.parent)
    return validated


def _validate_adaptive_authorization_receipt(
    value: object,
    *,
    baseline: Mapping[str, object],
    challenger: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "authorization_kind",
        "selection_access_authorized",
        "authorization_completed_before_selection_open",
        "lofo_report_file",
        "lofo_report_file_sha256",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_authority_sha256",
        "lofo_completed_fold_count",
        "lofo_all_required_gates_pass",
        "lofo_authorized_next_action",
        "baseline_tensor_file_sha256",
        "baseline_scientific_payload_sha256",
        "challenger_tensor_file_sha256",
        "challenger_scientific_payload_sha256",
        "full_eight_family_refit_completed",
        "fit_family_count",
        "fit_example_count",
        "fit_split_sha256",
        "diagnostic_split_sha256",
        "diagnostic_subset_within_fit",
        "diagnostic_used_for_selection",
        "claim_role",
        "heldout_confirmation",
        "serving_authorized",
        "compression_claim",
        "source_safe",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("adaptive authorization receipt fields are invalid")
    for name in (
        "lofo_report_file_sha256",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_authority_sha256",
        "baseline_tensor_file_sha256",
        "baseline_scientific_payload_sha256",
        "challenger_tensor_file_sha256",
        "challenger_scientific_payload_sha256",
        "fit_split_sha256",
        "diagnostic_split_sha256",
    ):
        _require_sha256(value.get(name), label=f"adaptive receipt {name}")
    report_file = value.get("lofo_report_file")
    if (
        not isinstance(report_file, str)
        or Path(report_file).name != report_file
        or Path(report_file).suffix != ".json"
        or value.get("authorization_kind")
        != "passing_lofo_then_full_eight_family_refit"
        or value.get("selection_access_authorized") is not True
        or value.get("authorization_completed_before_selection_open")
        is not True
        or value.get("lofo_protocol_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or value.get("lofo_completed_fold_count") != 8
        or value.get("lofo_all_required_gates_pass") is not True
        or value.get("lofo_authorized_next_action")
        != _LOFO_PASS_NEXT_ACTION
        or value.get("baseline_tensor_file_sha256")
        != baseline.get("tensor_file_sha256")
        or value.get("baseline_scientific_payload_sha256")
        != baseline.get("scientific_payload_sha256")
        or value.get("challenger_tensor_file_sha256")
        != challenger.get("tensor_file_sha256")
        or value.get("challenger_scientific_payload_sha256")
        != challenger.get("scientific_payload_sha256")
        or value.get("full_eight_family_refit_completed") is not True
        or value.get("fit_family_count") != 8
        or value.get("fit_example_count") != 256
        or value.get("fit_split_sha256")
        == value.get("diagnostic_split_sha256")
        or value.get("diagnostic_subset_within_fit") is not True
        or value.get("diagnostic_used_for_selection") is not False
        or value.get("claim_role")
        != "already_open_adaptive_development_selection"
        or value.get("heldout_confirmation") is not False
        or value.get("serving_authorized") is not False
        or value.get("compression_claim") is not False
        or value.get("source_safe") is not True
    ):
        raise ValueError("adaptive authorization receipt does not authorize selection")
    provenance = challenger.get("refit_provenance")
    provenance_fields = {
        "lofo_report_file_sha256",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_authority_sha256",
        "frozen_v9_candidate_file_sha256",
        "frozen_v9_candidate_scientific_sha256",
        "fit_split_sha256",
        "diagnostic_split_sha256",
        "fit_family_count",
        "fit_example_count",
        "full_eight_family_refit_completed",
        "diagnostic_subset_within_fit",
        "diagnostic_used_for_selection",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != provenance_fields:
        raise ValueError("challenger refit provenance fields are invalid")
    if (
        provenance.get("lofo_report_file_sha256")
        != value.get("lofo_report_file_sha256")
        or provenance.get("lofo_report_sha256")
        != value.get("lofo_report_sha256")
        or provenance.get("lofo_protocol_sha256")
        != value.get("lofo_protocol_sha256")
        or provenance.get("lofo_authority_sha256")
        != value.get("lofo_authority_sha256")
        or provenance.get("frozen_v9_candidate_file_sha256")
        != baseline.get("tensor_file_sha256")
        or provenance.get("frozen_v9_candidate_scientific_sha256")
        != baseline.get("scientific_payload_sha256")
        or provenance.get("fit_split_sha256")
        != value.get("fit_split_sha256")
        or provenance.get("diagnostic_split_sha256")
        != value.get("diagnostic_split_sha256")
        or provenance.get("fit_family_count") != value.get("fit_family_count")
        or provenance.get("fit_example_count") != value.get("fit_example_count")
        or provenance.get("full_eight_family_refit_completed") is not True
        or provenance.get("diagnostic_subset_within_fit") is not True
        or provenance.get("diagnostic_used_for_selection") is not False
    ):
        raise ValueError("challenger refit provenance does not bind authorization")
    return value


def validate_gemma3_layer17_open_a_capacity_result(
    value: Mapping[str, object] | Path | str,
) -> dict[str, object]:
    """Strict-validate either an in-memory payload or a canonical JSON file.

    In-memory candidate bindings can contain tuples, while JSON necessarily
    restores them as lists.  Canonical normalization makes those two public
    validation routes identical before schema and digest checks are applied.
    """

    if isinstance(value, (Path, str)):
        normalized, _ = _read_canonical_json(
            value,
            label="layer17 open-A result",
        )
    elif isinstance(value, Mapping):
        normalized = _json_native_mapping(
            value,
            label="layer17 open-A result",
        )
    else:
        raise TypeError(
            "layer17 open-A result must be a mapping or JSON file path"
        )
    value = normalized
    legacy_fields = {
        "schema",
        "format_version",
        "scientific_role",
        "heldout_confirmation",
        "candidates",
        "candidate_pair",
        "corpus",
        "runtime",
        "tokenization",
        "assessment",
        "candidate_changed",
        "candidate_tensor_file_sha256s_after",
        "selection_opened",
        "fit_opened",
        "guard_opened",
        "calibration_b_opened",
        "validation_opened",
        "test_opened",
        "safety",
        "result_sha256",
    }
    adaptive_fields = legacy_fields | {"authorization", "adaptive_selection"}
    actual_fields = frozenset(value) if isinstance(value, Mapping) else frozenset()
    if not isinstance(value, Mapping) or actual_fields not in {
        frozenset(legacy_fields),
        frozenset(adaptive_fields),
    }:
        raise ValueError("layer17 open-A result fields are invalid")
    has_adaptive_extension = actual_fields == frozenset(adaptive_fields)
    supplied = value.get("result_sha256")
    payload = {key: child for key, child in value.items() if key != "result_sha256"}
    if (
        value.get("schema") != _SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("scientific_role") not in {
            "open_development_capacity_comparison",
            "already_open_adaptive_development_fixed_capacity_refit",
        }
        or value.get("heldout_confirmation") is not False
        or supplied != _domain_sha256(_RESULT_DOMAIN, payload)
        or value.get("safety") != _SAFETY
        or value.get("candidate_changed") is not False
        or value.get("selection_opened") is not True
        or any(
            value.get(field) is not False
            for field in (
                "fit_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
            )
        )
    ):
        raise ValueError("layer17 open-A result header, hash, or safety is invalid")
    candidates = value.get("candidates")
    after = value.get("candidate_tensor_file_sha256s_after")
    pair = value.get("candidate_pair")
    if (
        not isinstance(candidates, Mapping)
        or not isinstance(after, Mapping)
        or not isinstance(pair, Mapping)
        or len(candidates) != 2
        or set(after) != set(candidates)
    ):
        raise ValueError("layer17 open-A candidate catalog is invalid")
    baseline_label = pair.get("baseline_label")
    challenger_label = pair.get("challenger_label")
    if baseline_label is None and challenger_label is None and set(candidates) == {
        "rank16",
        "rank32",
    }:
        # Backward-compatible validation for the first frozen result schema.
        baseline_label = "rank16"
        challenger_label = "rank32"
    baseline_label = _require_candidate_label(
        baseline_label,
        label="result baseline label",
    )
    challenger_label = _require_candidate_label(
        challenger_label,
        label="result challenger label",
    )
    condition_names = _candidate_conditions(baseline_label, challenger_label)
    if set(candidates) != {baseline_label, challenger_label}:
        raise ValueError("result candidate labels differ from candidate catalog")
    comparison_kind = pair.get("comparison_kind")
    if comparison_kind is None:
        comparison_kind = _CAPACITY_INCREASE_COMPARISON_KIND
    if comparison_kind not in {
        _CAPACITY_INCREASE_COMPARISON_KIND,
        _FIXED_CAPACITY_COMPARISON_KIND,
    }:
        raise ValueError("result comparison kind is unsupported")
    if comparison_kind == _FIXED_CAPACITY_COMPARISON_KIND:
        expected_pair_fields = {
            "baseline_label",
            "challenger_label",
            "comparison_kind",
            "identical_model",
            "identical_fragment_topology",
            "identical_native_removal_scope",
            "both_graphs_edgeless",
            "graph_parameter_delta",
            "graph_macs_per_token_delta",
            f"{challenger_label}_added_graph_parameters",
            f"{challenger_label}_added_graph_macs_per_token",
        }
        if (
            not has_adaptive_extension
            or set(pair) != expected_pair_fields
            or value.get("scientific_role")
            != "already_open_adaptive_development_fixed_capacity_refit"
            or pair.get("graph_parameter_delta") != 0
            or pair.get("graph_macs_per_token_delta") != 0
            or pair.get(f"{challenger_label}_added_graph_parameters") != 0
            or pair.get(f"{challenger_label}_added_graph_macs_per_token") != 0
        ):
            raise ValueError("fixed-capacity adaptive result binding drifted")
    elif (
        has_adaptive_extension
        or value.get("scientific_role")
        != "open_development_capacity_comparison"
    ):
        raise ValueError("legacy capacity result cannot carry adaptive authority")
    for name in (baseline_label, challenger_label):
        candidate = candidates[name]
        resolved = candidate.get("resolved_node_ranks") if isinstance(
            candidate, Mapping
        ) else None
        mode_rank_cap = candidate.get("mode_rank_cap") if isinstance(
            candidate, Mapping
        ) else None
        if mode_rank_cap is None and isinstance(candidate, Mapping):
            mode_rank_cap = candidate.get("mode_rank")
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("candidate_role") != name
            or type(mode_rank_cap) is not int
            or mode_rank_cap <= 0
            or type(candidate.get("generator_rank")) is not int
            or int(candidate["generator_rank"]) <= 0
            or candidate.get("interaction_count") != 0
            or candidate.get("node_count") != 4
            or candidate.get("tensor_file_sha256") != after[name]
        ):
            raise ValueError(f"{name} candidate binding is invalid")
        if resolved is None:
            if (
                name not in {"rank16", "rank32"}
                or mode_rank_cap != _EXPECTED_MODE_RANK
                or candidate.get("generator_rank")
                != (16 if name == "rank16" else 32)
            ):
                raise ValueError(f"{name} legacy capacity binding is invalid")
        elif (
            isinstance(resolved, (str, bytes))
            or not isinstance(resolved, Sequence)
            or len(resolved) != 4
            or any(
                type(rank) is not int or not 0 < rank <= mode_rank_cap
                for rank in resolved
            )
        ):
            raise ValueError(f"{name} resolved node ranks are invalid")
        artifact_schema = candidate.get("candidate_artifact_schema")
        if artifact_schema is not None and artifact_schema not in {
            GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
            GEMMA3_LAYER17_CAPPED_NODE_SCHEMA,
            GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
        }:
            raise ValueError(f"{name} candidate artifact schema is unsupported")
    baseline_candidate = candidates[baseline_label]
    challenger_candidate = candidates[challenger_label]
    assert isinstance(baseline_candidate, Mapping)
    assert isinstance(challenger_candidate, Mapping)
    if comparison_kind == _FIXED_CAPACITY_COMPARISON_KIND:
        fixed_candidate_fields = {
            "candidate_role",
            "candidate_kind",
            "candidate_artifact_schema",
            "tensor_file",
            "tensor_file_sha256",
            "scientific_payload_sha256",
            "model_id",
            "requested_revision",
            "model_fingerprint",
            "parameter_cluster_plan_sha256",
            "edgeless_graph_sha256",
            "mode_rank",
            "mode_rank_cap",
            "resolved_node_ranks",
            "generator_rank",
            "node_count",
            "interaction_count",
            "graph_parameters",
            "graph_macs_per_token",
            "removal_scope_sha256",
        }
        if (
            set(baseline_candidate) != fixed_candidate_fields
            or set(challenger_candidate)
            != fixed_candidate_fields | {"refit_provenance"}
        ):
            raise ValueError("adaptive candidate binding fields are invalid")
        if (
            baseline_candidate.get("tensor_file_sha256")
            == challenger_candidate.get("tensor_file_sha256")
            or baseline_candidate.get("scientific_payload_sha256")
            == challenger_candidate.get("scientific_payload_sha256")
        ):
            raise ValueError("adaptive candidates must be scientifically distinct")
        for field in (
            "mode_rank_cap",
            "resolved_node_ranks",
            "generator_rank",
            "graph_parameters",
            "graph_macs_per_token",
        ):
            if baseline_candidate.get(field) != challenger_candidate.get(field):
                raise ValueError(f"fixed-capacity result differs in {field}")
        if (
            baseline_candidate.get("candidate_artifact_schema")
            != GEMMA3_LAYER17_CAPPED_NODE_SCHEMA
            or challenger_candidate.get("candidate_artifact_schema")
            != GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
        ):
            raise ValueError("adaptive result candidate schemas are invalid")
        _require_sha256(
            baseline_candidate.get("scientific_payload_sha256"),
            label="adaptive baseline scientific payload",
        )
        _require_sha256(
            challenger_candidate.get("scientific_payload_sha256"),
            label="adaptive challenger scientific payload",
        )
    corpus = value.get("corpus")
    runtime = value.get("runtime")
    tokenization = value.get("tokenization")
    assessment = value.get("assessment")
    if not all(
        isinstance(child, Mapping)
        for child in (corpus, runtime, tokenization, assessment, pair)
    ):
        raise TypeError("layer17 open-A result sections are incomplete")
    assert isinstance(corpus, Mapping)
    assert isinstance(tokenization, Mapping)
    assert isinstance(assessment, Mapping)
    assert isinstance(pair, Mapping)
    if (
        corpus.get("example_count") != _EXPECTED_EXAMPLES
        or corpus.get("family_count") != _EXPECTED_FAMILIES
        or corpus.get("assessment_role")
        != "already_open_calibration_a_selection"
        or tokenization.get("example_count") != _EXPECTED_EXAMPLES
        or tokenization.get("family_stream_count") != _EXPECTED_FAMILIES
        or assessment.get("assessment_role") != value.get("scientific_role")
        or assessment.get("heldout_confirmation") is not False
        or assessment.get("example_count") != _EXPECTED_EXAMPLES
        or assessment.get("family_count") != _EXPECTED_FAMILIES
        or pair.get("identical_model") is not True
        or pair.get("identical_fragment_topology") is not True
        or pair.get("identical_native_removal_scope") is not True
        or pair.get("both_graphs_edgeless") is not True
    ):
        raise ValueError("layer17 open-A role or candidate-pair binding drifted")
    _validate_metric_container(
        assessment,
        label="micro",
        conditions=condition_names,
        identity_ulps=_MICRO_FAMILY_METRIC_IDENTITY_ULPS,
    )
    macro = assessment.get("equal_family_macro")
    families = assessment.get("families")
    if not isinstance(macro, Mapping) or not isinstance(families, Mapping):
        raise TypeError("layer17 open-A family metrics are unavailable")
    _validate_metric_container(
        macro,
        label="equal-family macro",
        conditions=condition_names,
        identity_ulps=_MACRO_METRIC_IDENTITY_ULPS,
    )
    expected_family_keys = {
        f"family_{index:02d}" for index in range(_EXPECTED_FAMILIES)
    }
    if set(families) != expected_family_keys:
        raise ValueError("layer17 open-A opaque family catalog is invalid")
    for family, metrics in families.items():
        if not isinstance(metrics, Mapping):
            raise TypeError(f"{family} metrics are invalid")
        _validate_metric_container(
            metrics,
            label=family,
            conditions=condition_names,
            identity_ulps=_MICRO_FAMILY_METRIC_IDENTITY_ULPS,
        )
    replayed_macro = _equal_family_macro(
        families,  # type: ignore[arg-type]
        conditions=condition_names,
    )
    if _canonical_json_bytes(macro) != _canonical_json_bytes(replayed_macro):
        raise ValueError("equal-family macro does not replay from family rows")
    graph_comparison = assessment.get("graph_comparison")
    resources = assessment.get("resource_accounting")
    if (
        not isinstance(graph_comparison, Mapping)
        or graph_comparison.get("deletion_paths_agree") is not True
        or graph_comparison.get("deletion_equivalence_atol") != 0.0
        or graph_comparison.get("deletion_equivalence_rtol") != 0.0
        or graph_comparison.get("deletion_max_abs_logit_difference") != 0.0
        or not isinstance(resources, Mapping)
        or set(resources) != set(condition_names)
    ):
        raise ValueError("layer17 open-A graph controls or resources are invalid")
    for name, record in resources.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"{name} resource record is invalid")
        for field in (
            "modal_graph_learned_parameters",
            "native_removed_learned_parameters",
            "net_stored_parameter_savings",
            "graph_macs_per_token",
            "native_removed_macs_per_token",
            "executed_graph_macs_per_token",
            "net_logical_macs_saved_per_token",
        ):
            if type(record.get(field)) is not int:
                raise ValueError(f"{name} {field} is not exact integer accounting")
    baseline_condition, challenger_condition, _ = condition_names
    baseline_resources = resources[baseline_condition]
    challenger_resources = resources[challenger_condition]
    assert isinstance(baseline_resources, Mapping)
    assert isinstance(challenger_resources, Mapping)
    if comparison_kind == _FIXED_CAPACITY_COMPARISON_KIND:
        for candidate, record, label in (
            (baseline_candidate, baseline_resources, baseline_label),
            (challenger_candidate, challenger_resources, challenger_label),
        ):
            if (
                type(candidate.get("graph_parameters")) is not int
                or candidate.get("graph_parameters")
                != record.get("modal_graph_learned_parameters")
                or type(candidate.get("graph_macs_per_token")) is not int
                or candidate.get("graph_macs_per_token")
                != record.get("graph_macs_per_token")
            ):
                raise ValueError(
                    f"{label} candidate/runtime resource binding drifted"
                )
        if any(
            baseline_resources.get(field) != challenger_resources.get(field)
            for field in (
                "modal_graph_learned_parameters",
                "graph_macs_per_token",
                "executed_graph_macs_per_token",
            )
        ):
            raise ValueError("fixed-capacity runtime resources differ")
        capacity_delta = assessment.get("capacity_delta")
        if (
            not isinstance(capacity_delta, Mapping)
            or capacity_delta.get(
                f"{challenger_label}_added_graph_parameters"
            )
            != 0
            or capacity_delta.get(
                f"{challenger_label}_added_graph_macs_per_token"
            )
            != 0
        ):
            raise ValueError("fixed-capacity assessment resources drifted")
        _validate_adaptive_authorization_receipt(
            value.get("authorization"),
            baseline=baseline_candidate,
            challenger=challenger_candidate,
        )
        expected_adaptive = _evaluate_fixed_capacity_adaptive_gates(
            assessment=assessment,
            candidate_pair=pair,
            baseline_label=baseline_label,
            challenger_label=challenger_label,
        )
        if _canonical_json_bytes(value.get("adaptive_selection")) != (
            _canonical_json_bytes(expected_adaptive)
        ):
            raise ValueError("adaptive selection decision does not replay")
    elif pair.get("comparison_kind") is not None:
        parameter_delta = pair.get("graph_parameter_delta")
        mac_delta = pair.get("graph_macs_per_token_delta")
        if (
            type(parameter_delta) is not int
            or parameter_delta <= 0
            or type(mac_delta) is not int
            or mac_delta <= 0
        ):
            raise ValueError("capacity-increase result deltas are invalid")
    _reject_forbidden_output_fields(value)
    return dict(value)


def load_gemma3_layer17_open_a_capacity_result(
    path: Path | str,
) -> dict[str, object]:
    return validate_gemma3_layer17_open_a_capacity_result(path)


def evaluate_gemma3_layer17_open_a_capacity(
    *,
    rank16_candidate_path: Path | str = DEFAULT_RANK16_CANDIDATE,
    rank32_candidate_path: Path | str = DEFAULT_RANK32_CANDIDATE,
    baseline_label: str = "rank16",
    challenger_label: str = "rank32",
    lofo_report_path: Path | str | None = DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    selection_path: Path | str = DEFAULT_SELECTION_OUTPUT,
    receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = 4,
    output: Path | str = DEFAULT_LAYER17_OPEN_A_OUTPUT,
) -> dict[str, object]:
    """Run one named edgeless capacity comparison on open selection only."""

    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    destination = Path(output)
    if destination.suffix != ".json" or not destination.name:
        raise ValueError("open-A output must have a JSON basename")
    checkpoint_path = _prevalidation_checkpoint_path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination.name}")
    if checkpoint_path.exists():
        raise FileExistsError(
            f"surviving prevalidation checkpoint requires recovery: "
            f"{checkpoint_path.name}"
        )
    baseline_name, challenger_name = (
        _require_candidate_label(baseline_label, label="baseline label"),
        _require_candidate_label(challenger_label, label="challenger label"),
    )
    _candidate_conditions(baseline_name, challenger_name)
    rank16 = _candidate_authority(
        rank16_candidate_path,
        label=baseline_name,
    )
    rank32 = _candidate_authority(
        rank32_candidate_path,
        label=challenger_name,
    )
    pair = _validate_candidate_pair(rank16, rank32)
    authorization: dict[str, object] | None = None
    if pair["comparison_kind"] == _FIXED_CAPACITY_COMPARISON_KIND:
        if lofo_report_path is None:
            raise ValueError(
                "fixed-capacity adaptive selection requires a LOFO report"
            )
        # This authorization deliberately precedes the only call capable of
        # opening selection prompts.
        authorization = _authorize_fixed_capacity_adaptive_selection(
            lofo_report_path=lofo_report_path,
            baseline=rank16,
            challenger=rank32,
        )
    selection = _load_open_selection_authority(
        corpus_artifact_path=corpus_artifact_path,
        selection_path=selection_path,
        receipt_path=receipt_path,
    )
    model_id = rank16.binding.get("model_id")
    revision = rank16.binding.get("requested_revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("candidate model id is invalid")
    if not isinstance(revision, str) or not revision:
        raise ValueError("candidate revision is invalid")
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != rank16.binding["model_fingerprint"]:
        raise ValueError("live Gemma fingerprint differs from frozen candidates")
    family_batches, tokenization = _materialize_selection_families(
        tokenizer,
        selection,
        device=device,
        tokenization_batch_size=tokenization_batch_size,
    )
    rank16_lowerings = dict(rank16.lowerings)
    rank32_lowerings = dict(rank32.lowerings)
    rank16_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        rank16.edgeless_graph,  # type: ignore[arg-type]
        tuple(
            rank16_lowerings[name]
            for name in rank16.edgeless_graph.traversal_order
        ),
    )
    rank32_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        rank32.edgeless_graph,  # type: ignore[arg-type]
        tuple(
            rank32_lowerings[name]
            for name in rank32.edgeless_graph.traversal_order
        ),
    )
    assessment = _score_capacity_panel(
        adapter=adapter,
        rank16_executor=rank16_executor,
        rank32_executor=rank32_executor,
        family_batches=family_batches,
        baseline_label=baseline_name,
        challenger_label=challenger_name,
    )
    adaptive_selection: dict[str, object] | None = None
    scientific_role = "open_development_capacity_comparison"
    if pair["comparison_kind"] == _FIXED_CAPACITY_COMPARISON_KIND:
        scientific_role = (
            "already_open_adaptive_development_fixed_capacity_refit"
        )
        assessment["assessment_role"] = scientific_role
        adaptive_selection = _evaluate_fixed_capacity_adaptive_gates(
            assessment=assessment,
            candidate_pair=pair,
            baseline_label=baseline_name,
            challenger_label=challenger_name,
        )
    if (
        tokenization["logical_valid_tokens"]
        != assessment["logical_valid_tokens"]
        or tokenization["supervised_tokens"] != assessment["supervised_tokens"]
    ):
        raise RuntimeError("tokenization and scoring totals disagree")
    after = {
        baseline_name: _file_sha256(rank16.path),
        challenger_name: _file_sha256(rank32.path),
    }
    if after != {
        baseline_name: rank16.file_sha256,
        challenger_name: rank32.file_sha256,
    } or adapter.model_fingerprint() != rank16.binding["model_fingerprint"]:
        raise RuntimeError("candidate or source model changed during evaluation")

    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "scientific_role": scientific_role,
        "heldout_confirmation": False,
        "candidates": {
            baseline_name: rank16.binding,
            challenger_name: rank32.binding,
        },
        "candidate_pair": pair,
        "corpus": selection.binding,
        "runtime": {
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": rank16.binding["model_fingerprint"],
            "device": str(device),
            "dtype": dtype,
            "tokenization_batch_size": tokenization_batch_size,
            "max_length": int(_tokenizer_contract()["max_length"]),
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
            "local_files_only": True,
        },
        "tokenization": tokenization,
        "assessment": assessment,
        "candidate_changed": False,
        "candidate_tensor_file_sha256s_after": after,
        "selection_opened": True,
        "fit_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "safety": dict(_SAFETY),
    }
    if authorization is not None and adaptive_selection is not None:
        payload["authorization"] = authorization
        payload["adaptive_selection"] = adaptive_selection
    result = {
        **payload,
        "result_sha256": _domain_sha256(_RESULT_DOMAIN, payload),
    }
    return _publish_with_prevalidation_checkpoint(
        result,
        output=destination,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "compare two named frozen layer17 edgeless candidates on the "
            "already-open v8 Calibration-A selection role"
        )
    )
    parser.add_argument(
        "--baseline-candidate",
        "--rank16-candidate",
        dest="baseline_candidate",
        type=Path,
        default=DEFAULT_RANK16_CANDIDATE,
    )
    parser.add_argument(
        "--challenger-candidate",
        "--rank32-candidate",
        dest="challenger_candidate",
        type=Path,
        default=DEFAULT_RANK32_CANDIDATE,
    )
    parser.add_argument("--baseline-label", default="rank16")
    parser.add_argument("--challenger-label", default="rank32")
    parser.add_argument(
        "--lofo-report",
        type=Path,
        default=DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
        help=(
            "strict passing family-LOFO report required only for a "
            "fixed-capacity refit comparison"
        ),
    )
    parser.add_argument("--corpus-artifact", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_LAYER17_OPEN_A_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_gemma3_layer17_open_a_capacity(
        rank16_candidate_path=args.baseline_candidate,
        rank32_candidate_path=args.challenger_candidate,
        baseline_label=args.baseline_label,
        challenger_label=args.challenger_label,
        lofo_report_path=args.lofo_report,
        corpus_artifact_path=args.corpus_artifact,
        selection_path=args.selection,
        receipt_path=args.receipt,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        tokenization_batch_size=args.tokenization_batch_size,
        output=args.output,
    )
    summary = {
        "output": str(args.output),
        "result_sha256": result["result_sha256"],
        "capacity_delta": result["assessment"]["capacity_delta"],  # type: ignore[index]
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
