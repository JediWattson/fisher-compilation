"""Fit the post-LOFO layer-17 executable on all eight v8 A-fit families.

This is the protocol-authorized rung after a passing family-LOFO diagnostic.
It opens only the authenticated v8 ``calibration_a_fit`` role, captures the
four frozen layer-17 fragment streams once, normalizes raw empirical Fisher
mass equally across all eight families, and fits one cap-48/rank-16 edgeless
candidate on every captured row.

The underlying fixed-rank fitting primitive requires evaluation rows to build
descriptive rate-curve records.  Those rows are an explicit deterministic,
balanced subset of A-fit itself.  They are not disjoint, held out, used for
rank selection, used for early stopping, or reported as an assessment.  The
only subsequent assessment authorized by the protocol is a separate replay
of the already-open Calibration-A selection role.

The tensor artifact contains executable generator parameters but no prompt
text, prompt identities, token ids, logits, or captured activation/gradient
rows.  Its JSON companion is tensor-free.  Selection, guard, Calibration-B,
validation, and test paths are intentionally absent from the API and CLI.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import sys

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
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_capped_node_fit import (
    _coerce_lowering_records,
    _restore_lowering_records,
    _selection_from_lowerings,
    _validate_frozen_selection,
    _validate_pipeline,
    fit_layer17_capped_node_pilots,
)
from .gemma3_layer17_family_lofo_authority import (
    load_gemma3_layer17_family_lofo_authority,
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_authority_metadata,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    validate_v8_layer17_family_lofo_protocol,
)
from .gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_NATIVE_MODE_COUNTS,
    LAYER17_TOPOLOGY_SHA256,
    Layer17NodeRankResourceRow,
    build_layer17_node_rank_resource_row,
    resolve_layer17_node_ranks,
)
from .gemma3_layer17_v8_fit_lofo import (
    DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    _FISHER_NORMALIZATION,
    _equal_family_fisher_concat,
    _family_blocks,
    _load_authenticated_protocol,
    _selected_rows,
    _tensor_sha256,
    _validate_fold_pilots_are_fit_only,
    load_gemma3_layer17_v8_fit_lofo_report,
    partition_aligned_fragment_rows_by_family,
)
from .gemma3_modal_generator_dev_experiment import (
    load_gemma3_modal_generator_dev_artifact,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    DEFAULT_BASE_ARTIFACT,
    _restore_upstream_analysis,
    _validate_upstream_bindings,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    build_edgeless_same_layer_graph,
    select_top_fisher_same_layer_fragments,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _collect_same_layer_native_rows,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_compiler_pipeline import (
    ModalCompilerPipeline,
    ModalRefitFisherAuthority,
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


__all__ = [
    "DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT",
    "DEFAULT_DIAGNOSTIC_OBSERVATIONS_PER_FAMILY",
    "GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_FORMAT_VERSION",
    "GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA",
    "build_gemma3_layer17_v8_all_family_refit_candidate",
    "build_gemma3_layer17_v8_all_family_refit_report",
    "build_layer17_all_family_refit_rows",
    "build_parser",
    "load_gemma3_layer17_v8_all_family_refit_candidate",
    "restore_gemma3_layer17_v8_all_family_refit_runtime",
    "run_gemma3_layer17_v8_all_family_refit",
    "save_gemma3_layer17_v8_all_family_refit_candidate",
    "validate_gemma3_layer17_v8_all_family_refit_candidate",
]


GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA = (
    "fisher_graph.gemma3_layer17_v8_fit_all_family_refit_candidate"
)
GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_FORMAT_VERSION = 1
_REPORT_SCHEMA = "fisher_graph.gemma3_layer17_v8_all_family_refit_report"
_MODE_RANK_CAP = 48
_GENERATOR_RANK = 16
_RIDGE = 0.0
_EXPECTED_FAMILIES = 8
_EXPECTED_EXAMPLES = 256
DEFAULT_DIAGNOSTIC_OBSERVATIONS_PER_FAMILY = 128
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT = _LOCAL_ROOT / (
    "layer17-capped-node-c48-r16-edgeless-a-fit-v8-full-refit-dev-v1.pt"
)

_SCIENTIFIC_DOMAIN = (
    b"fisher-graph:gemma3-layer17-v8-all-family-refit:scientific:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-layer17-v8-all-family-refit:report:v1\0"
)
_FIT_SPLIT_DOMAIN = (
    b"fisher-graph:gemma3-layer17-v8-all-family-refit:fit-split:v1\0"
)
_DIAGNOSTIC_SPLIT_DOMAIN = (
    b"fisher-graph:gemma3-layer17-v8-all-family-refit:diagnostic-split:v1\0"
)
_DIAGNOSTIC_ORDER_DOMAIN = (
    b"fisher-graph:gemma3-layer17-v8-all-family-refit:diagnostic-order:v1\0"
)
_FIT_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-layer17-v8-all-family-refit:fit-receipt:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")

_ROOT_FIELDS = {
    "schema",
    "format_version",
    "experiment",
    "config",
    "authority",
    "protocol",
    "fit_collection",
    "fit_receipt",
    "resources",
    "fragment_selection",
    "lowering_records",
    "edgeless_graph",
    "compiler_pipeline",
    "lineage",
    "safety",
    "scientific_payload_sha256",
}
_FIT_COLLECTION_FIELDS = {
    "materialization",
    "capture_count",
    "captured_examples",
    "captured_observations",
    "captured_sequences",
    "captured_row_key_sha256",
    "family_observations",
    "model_rows_recollected",
}
_FIT_RECEIPT_FIELDS = {
    "fit_receipt_sha256",
    "fit_split_sha256",
    "diagnostic_split_sha256",
    "family_aliases",
    "family_count",
    "fit_example_count",
    "fit_observations",
    "fit_sequences",
    "fisher_normalization",
    "fit_fisher_total_mass",
    "fit_fisher_total_mass_per_family",
    "all_coefficients_fit_on_all_normalized_rows",
    "diagnostic_subset_within_fit",
    "diagnostic_subset_balanced_by_family",
    "diagnostic_observations_per_family",
    "diagnostic_observations",
    "diagnostic_used_for_fixed_rank_curve_construction",
    "diagnostic_used_for_selection",
    "diagnostic_used_for_early_stopping",
    "diagnostic_supports_assessment_claim",
}
_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_semantic_family_identifiers": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_or_gradient_rows": False,
    "contains_source_model_weights": False,
    "contains_executable_generator_weights": True,
    "fit_opened": True,
    "selection_opened": False,
    "guard_opened": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "diagnostic_subset_within_fit": True,
    "diagnostic_subset_used_for_selection": False,
    "source_safe": True,
}


def _progress(message: str) -> None:
    print(
        f"[layer17-v8-all-family-refit] {message}",
        file=sys.stderr,
        flush=True,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite JSON number")
    return float(value)


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _strict_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _reject_source_metadata(value: object, *, path: str = "metadata") -> None:
    forbidden = {
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
        "raw_rows",
        "activation_rows",
        "gradient_rows",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if not normalized.startswith("contains_") and normalized in forbidden:
                raise ValueError(f"{path}.{key} is a forbidden source field")
            _reject_source_metadata(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_source_metadata(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite float")


def _row_split_sha256(
    rows: AlignedFragmentRows,
    *,
    authority_sha256: str,
    materialization_sha256: str,
    lofo_report_sha256: str,
    role: str,
    family_aliases: Sequence[str],
) -> str:
    if role not in ("fit_all_eight", "diagnostic_within_fit"):
        raise ValueError("all-family split role is invalid")
    aliases = tuple(family_aliases)
    if aliases != tuple(sorted(set(aliases))) or not aliases:
        raise ValueError("all-family split aliases must be sorted and unique")
    domain = (
        _FIT_SPLIT_DOMAIN
        if role == "fit_all_eight"
        else _DIAGNOSTIC_SPLIT_DOMAIN
    )
    return _sha256(
        {
            "authority_sha256": _require_sha256(
                authority_sha256,
                label="authority",
            ),
            "materialization_sha256": _require_sha256(
                materialization_sha256,
                label="materialization",
            ),
            "lofo_report_sha256": _require_sha256(
                lofo_report_sha256,
                label="LOFO report",
            ),
            "protocol_sha256": FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
            "role": role,
            "family_aliases": aliases,
            "row_key_sha256": rows.row_key_sha256,
            "observations": rows.observations,
            "sequences": rows.sequences,
            "fisher_normalization": _FISHER_NORMALIZATION,
            "fisher_weight_sha256_by_fragment": {
                fragment_id: _tensor_sha256(fragment.fisher_weights)
                for fragment_id, fragment in sorted(
                    rows.rows_by_fragment.items()
                )
            },
        },
        domain=domain,
    )


def _diagnostic_indices(
    rows: AlignedFragmentRows,
    *,
    family_alias: str,
    count: int,
) -> Tensor:
    if type(count) is not int or count <= 0 or count > rows.observations:
        raise ValueError("diagnostic count is outside the family row range")
    ranked = sorted(
        range(rows.observations),
        key=lambda index: hashlib.sha256(
            _DIAGNOSTIC_ORDER_DOMAIN
            + _canonical_json_bytes(
                {
                    "family_alias": family_alias,
                    "row_key": rows.row_keys[index],
                }
            )
        ).digest(),
    )[:count]
    return torch.tensor(sorted(ranked), dtype=torch.long)


def build_layer17_all_family_refit_rows(
    family_rows: Mapping[str, AlignedFragmentRows],
    *,
    authority_sha256: str,
    materialization_sha256: str,
    lofo_report_sha256: str,
    diagnostic_observations_per_family: int = (
        DEFAULT_DIAGNOSTIC_OBSERVATIONS_PER_FAMILY
    ),
) -> tuple[AlignedFragmentRows, AlignedFragmentRows, dict[str, object]]:
    """Build all-eight fit rows and a balanced in-fit diagnostic subset."""

    aliases = tuple(V8_FAMILY_LOFO_FAMILY_ALIASES)
    if not isinstance(family_rows, Mapping) or set(family_rows) != set(aliases):
        raise ValueError("all-family refit requires the exact eight opaque blocks")
    if (
        type(diagnostic_observations_per_family) is not int
        or diagnostic_observations_per_family <= 0
    ):
        raise ValueError("diagnostic_observations_per_family must be positive")
    minimum = min(family_rows[alias].observations for alias in aliases)
    example_counts = {
        alias: len({example_id for example_id, _ in family_rows[alias].row_keys})
        for alias in aliases
    }
    if (
        any(count != 32 for count in example_counts.values())
        or sum(example_counts.values()) != _EXPECTED_EXAMPLES
        or len(
            {
                example_id
                for alias in aliases
                for example_id, _ in family_rows[alias].row_keys
            }
        )
        != _EXPECTED_EXAMPLES
    ):
        raise ValueError(
            "all-family refit rows must contain 32 unique examples per family"
        )
    resolved_diagnostic_count = min(
        diagnostic_observations_per_family,
        minimum,
    )
    if resolved_diagnostic_count <= 0:
        raise ValueError("family rows contain no diagnostic observations")

    fit_rows = _equal_family_fisher_concat(family_rows, aliases)
    diagnostic_family_rows = {
        alias: _selected_rows(
            family_rows[alias],
            _diagnostic_indices(
                family_rows[alias],
                family_alias=alias,
                count=resolved_diagnostic_count,
            ),
        )
        for alias in aliases
    }
    diagnostic_rows = _equal_family_fisher_concat(
        diagnostic_family_rows,
        aliases,
    )
    if not set(diagnostic_rows.row_keys).issubset(fit_rows.row_keys):
        raise RuntimeError("diagnostic rows escaped the A-fit row set")
    fit_split_sha256 = _row_split_sha256(
        fit_rows,
        authority_sha256=authority_sha256,
        materialization_sha256=materialization_sha256,
        lofo_report_sha256=lofo_report_sha256,
        role="fit_all_eight",
        family_aliases=aliases,
    )
    diagnostic_split_sha256 = _row_split_sha256(
        diagnostic_rows,
        authority_sha256=authority_sha256,
        materialization_sha256=materialization_sha256,
        lofo_report_sha256=lofo_report_sha256,
        role="diagnostic_within_fit",
        family_aliases=aliases,
    )
    if fit_split_sha256 == diagnostic_split_sha256:
        raise RuntimeError("fit and diagnostic split commitments collided")
    receipt_payload = {
        "fit_split_sha256": fit_split_sha256,
        "diagnostic_split_sha256": diagnostic_split_sha256,
        "family_aliases": aliases,
        "family_count": _EXPECTED_FAMILIES,
        "fit_example_count": _EXPECTED_EXAMPLES,
        "fit_observations": fit_rows.observations,
        "fit_sequences": fit_rows.sequences,
        "fisher_normalization": _FISHER_NORMALIZATION,
        "fit_fisher_total_mass": 1.0,
        "fit_fisher_total_mass_per_family": 1.0 / _EXPECTED_FAMILIES,
        "all_coefficients_fit_on_all_normalized_rows": True,
        "diagnostic_subset_within_fit": True,
        "diagnostic_subset_balanced_by_family": True,
        "diagnostic_observations_per_family": resolved_diagnostic_count,
        "diagnostic_observations": diagnostic_rows.observations,
        "diagnostic_used_for_fixed_rank_curve_construction": True,
        "diagnostic_used_for_selection": False,
        "diagnostic_used_for_early_stopping": False,
        "diagnostic_supports_assessment_claim": False,
    }
    receipt = {
        **receipt_payload,
        "fit_receipt_sha256": _sha256(
            receipt_payload,
            domain=_FIT_RECEIPT_DOMAIN,
        ),
    }
    return fit_rows, diagnostic_rows, receipt


def _passing_lofo_lineage(
    report: Mapping[str, object],
    *,
    report_path: Path | str,
) -> dict[str, object]:
    decision = report.get("decision")
    experiment = report.get("experiment")
    protocol = report.get("protocol")
    authority = report.get("authority")
    fit_collection = report.get("fit_collection")
    lineage = report.get("lineage")
    if any(
        not isinstance(value, Mapping)
        for value in (
            decision,
            experiment,
            protocol,
            authority,
            fit_collection,
            lineage,
        )
    ):
        raise TypeError("passing LOFO report bindings are incomplete")
    assert isinstance(decision, Mapping)
    assert isinstance(experiment, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(authority, Mapping)
    assert isinstance(fit_collection, Mapping)
    assert isinstance(lineage, Mapping)
    authority_receipt = authority.get("receipt")
    authority_corpus = authority.get("corpus")
    if not isinstance(authority_receipt, Mapping) or not isinstance(
        authority_corpus,
        Mapping,
    ):
        raise TypeError("passing LOFO authority provenance is incomplete")
    if (
        decision.get("all_required_gates_pass") is not True
        or decision.get("next_action")
        != "freeze_full_eight_family_refit_then_replay_eligible_open_"
        "development_assessment"
        or protocol.get("artifact_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or report.get("heldout_confirmation") is not False
        or report.get("serving_authorized") is not False
        or report.get("compression_claim") is not False
    ):
        raise ValueError("LOFO report does not authorize the all-family refit")
    return {
        "lofo_report_file": Path(report_path).name,
        "lofo_report_file_sha256": _file_sha256(report_path),
        "lofo_report_sha256": _require_sha256(
            report.get("report_sha256"),
            label="LOFO report",
        ),
        "lofo_protocol_sha256": FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
        "lofo_all_required_gates_pass": True,
        "lofo_authorized_next_action": decision["next_action"],
        "lofo_authority_sha256": _require_sha256(
            authority.get("authority_sha256"),
            label="LOFO authority",
        ),
        "lofo_receipt_sha256": _require_sha256(
            authority_receipt.get("receipt_sha256"),
            label="LOFO receipt",
        ),
        "lofo_corpus_artifact_sha256": _require_sha256(
            authority_corpus.get("corpus_artifact_sha256"),
            label="LOFO corpus artifact",
        ),
        "lofo_fit_manifest_sha256": _require_sha256(
            authority_corpus.get("fit_manifest_sha256"),
            label="LOFO fit manifest",
        ),
        "lofo_materialization_sha256": _require_sha256(
            fit_collection.get("materialization_sha256"),
            label="LOFO materialization",
        ),
        "lofo_captured_row_key_sha256": _require_sha256(
            fit_collection.get("captured_row_key_sha256"),
            label="LOFO captured row-key",
        ),
        "lofo_model_fingerprint": _require_sha256(
            experiment.get("adapter_model_fingerprint"),
            label="LOFO model fingerprint",
        ),
        "lofo_requested_revision": experiment.get("requested_revision"),
        "lofo_base_artifact_file_sha256": _require_sha256(
            lineage.get("base_artifact_file_sha256"),
            label="LOFO base artifact",
        ),
        "lofo_fragment_plan_sha256": _require_sha256(
            lineage.get("fragment_plan_sha256"),
            label="LOFO fragment plan",
        ),
        "lofo_fragment_selection_sha256": _require_sha256(
            lineage.get("fragment_selection_sha256"),
            label="LOFO fragment selection",
        ),
        "frozen_v9_candidate_file": lineage.get(
            "frozen_v9_candidate_file"
        ),
        "frozen_v9_candidate_file_sha256": _require_sha256(
            lineage.get("frozen_v9_candidate_file_sha256"),
            label="LOFO frozen v9 candidate file",
        ),
        "frozen_v9_candidate_scientific_sha256": _require_sha256(
            lineage.get("frozen_v9_candidate_scientific_sha256"),
            label="LOFO frozen v9 candidate scientific payload",
        ),
    }


def _embedded_protocol(protocol: Mapping[str, object]) -> dict[str, object]:
    frozen = validate_v8_layer17_family_lofo_protocol(protocol)
    return {
        "artifact_sha256": frozen["artifact_sha256"],
        "corpus_authority": frozen["corpus_authority"],
        "first_arm": frozen["first_arm"],
        "evaluation_contract": frozen["evaluation_contract"],
        "claim_boundary": frozen["claim_boundary"],
        "runtime_fisher_normalization": _FISHER_NORMALIZATION,
        "authorized_by_passing_outer_lofo": True,
    }


def _validate_full_refit_pipeline_authority(
    pipeline: ModalCompilerPipeline,
    *,
    authority: Mapping[str, object],
    fit_collection: Mapping[str, object],
    fit_receipt: Mapping[str, object],
    lofo_lineage: Mapping[str, object],
) -> ModalRefitFisherAuthority:
    refit = pipeline.modal_refit_fisher_authority
    authority_corpus = authority.get("corpus")
    materialization = fit_collection.get("materialization")
    if (
        not isinstance(refit, ModalRefitFisherAuthority)
        or not isinstance(authority_corpus, Mapping)
        or not isinstance(materialization, Mapping)
    ):
        raise ValueError(
            "all-family compiler requires explicit refit-Fisher authority"
        )
    expected = {
        "fit_split_sha256": fit_receipt.get("fit_split_sha256"),
        "eval_split_sha256": fit_receipt.get(
            "diagnostic_split_sha256"
        ),
        "eval_role": "balanced_within_fit_fixed_rank_diagnostic",
        "fisher_normalization": _FISHER_NORMALIZATION,
        "source_model_sha256": pipeline.model_fingerprint,
        "parameter_catalog_sha256": (
            pipeline.parameter_catalog.artifact_sha256
        ),
        "topology_grouped_fisher_sha256": (
            pipeline.grouped_fisher.referenced_artifact_sha256
        ),
        "topology_fisher_calibration_split_sha256": (
            pipeline.grouped_fisher.metadata.get(
                "calibration_split_sha256"
            )
        ),
        "topology_fisher_cluster_plan_sha256": (
            pipeline.fisher_clusters.referenced_artifact_sha256
        ),
        "topology_fragment_plan_sha256": (
            pipeline.parameter_cluster_fragments.artifact_sha256
        ),
        "authorizing_report_sha256": lofo_lineage.get(
            "lofo_report_sha256"
        ),
        "refit_protocol_sha256": lofo_lineage.get(
            "lofo_protocol_sha256"
        ),
        "fit_authority_sha256": authority.get("authority_sha256"),
        "fit_receipt_sha256": fit_receipt.get("fit_receipt_sha256"),
        "fit_corpus_artifact_sha256": authority_corpus.get(
            "corpus_artifact_sha256"
        ),
        "fit_manifest_sha256": authority_corpus.get(
            "fit_manifest_sha256"
        ),
        "fit_materialization_sha256": materialization.get(
            "materialization_sha256"
        ),
        "fit_example_count": _EXPECTED_EXAMPLES,
        "fit_family_count": _EXPECTED_FAMILIES,
        "equal_family_weighting": True,
        "eval_subset_within_fit": True,
        "eval_used_for_selection": False,
    }
    if any(getattr(refit, key) != value for key, value in expected.items()):
        raise ValueError("all-family refit-Fisher authority provenance drifted")
    return refit


def _runtime_lineage(
    *,
    graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    pipeline: ModalCompilerPipeline,
    selection: SameLayerFragmentSelection,
    resources: Layer17NodeRankResourceRow,
    fit_receipt: Mapping[str, object],
    lofo_lineage: Mapping[str, object],
    base_artifact_file: str,
    base_artifact_file_sha256: str,
) -> dict[str, object]:
    fit_hashes = {
        lowering.coordinate_generator_plan.binding.fit_split_sha256
        for lowering in lowerings_by_node.values()
    }
    diagnostic_hashes = {
        lowering.coordinate_generator_plan.binding.eval_split_sha256
        for lowering in lowerings_by_node.values()
    }
    if fit_hashes != {fit_receipt.get("fit_split_sha256")} or (
        diagnostic_hashes
        != {fit_receipt.get("diagnostic_split_sha256")}
    ):
        raise ValueError("all-family lowerings differ from the fit receipt")
    return {
        "topology_sha256": LAYER17_TOPOLOGY_SHA256,
        "model_fingerprint": graph.model_fingerprint,
        "fragment_plan_sha256": graph.parameter_cluster_plan_sha256,
        "fragment_selection_sha256": selection.artifact_sha256,
        "fragment_ids": LAYER17_FRAGMENT_IDS,
        "native_mode_counts": LAYER17_NATIVE_MODE_COUNTS,
        "fit_split_sha256": next(iter(fit_hashes)),
        "diagnostic_split_sha256": next(iter(diagnostic_hashes)),
        "fit_receipt_sha256": _require_sha256(
            fit_receipt.get("fit_receipt_sha256"),
            label="fit receipt",
        ),
        "lowering_sha256s": tuple(
            lowerings_by_node[name].artifact_sha256
            for name in graph.traversal_order
        ),
        "graph_sha256": graph.artifact_sha256,
        "compiler_pipeline_sha256": pipeline.artifact_sha256,
        "modal_refit_fisher_authority_sha256": _require_sha256(
            getattr(
                pipeline.modal_refit_fisher_authority,
                "artifact_sha256",
                None,
            ),
            label="modal refit-Fisher authority",
        ),
        "resource_row_sha256": resources.artifact_sha256,
        "base_artifact_file": base_artifact_file,
        "base_artifact_file_sha256": _require_sha256(
            base_artifact_file_sha256,
            label="base artifact file",
        ),
        **dict(lofo_lineage),
    }


def _scientific_projection(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "schema",
            "format_version",
            "experiment",
            "config",
            "authority",
            "protocol",
            "fit_collection",
            "fit_receipt",
            "resources",
            "fragment_selection",
            "lineage",
            "safety",
        )
    }


def build_gemma3_layer17_v8_all_family_refit_candidate(
    *,
    experiment: Mapping[str, object],
    authority: Mapping[str, object],
    protocol: Mapping[str, object],
    fit_collection: Mapping[str, object],
    fit_receipt: Mapping[str, object],
    lofo_lineage: Mapping[str, object],
    selection: SameLayerFragmentSelection,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    edgeless_graph: ModalGeneratorGraphPlan,
    compiler_pipeline: ModalCompilerPipeline,
    base_artifact_file: str,
    base_artifact_file_sha256: str,
) -> dict[str, object]:
    """Build one strict executable full-eight-family refit payload."""

    _validate_frozen_selection(selection)
    edgeless_graph.validate_integrity()
    if edgeless_graph.interactions:
        raise ValueError("all-family refit graph must remain edgeless")
    resources = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        edge_policy="edgeless",
    )
    if (
        edgeless_graph.parameter_count != resources.graph_parameter_count
        or edgeless_graph.macs_per_token != resources.graph_dense_macs_per_token
    ):
        raise ValueError("all-family graph resources differ from cap48/r16")
    _validate_pipeline(
        compiler_pipeline,
        graph=edgeless_graph,
        lowerings_by_node=lowerings_by_node,
        resource_row=resources,
    )
    _validate_full_refit_pipeline_authority(
        compiler_pipeline,
        authority=authority,
        fit_collection=fit_collection,
        fit_receipt=fit_receipt,
        lofo_lineage=lofo_lineage,
    )
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
        "format_version": (
            GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_FORMAT_VERSION
        ),
        "experiment": dict(experiment),
        "config": {
            "mode_rank_cap": _MODE_RANK_CAP,
            "resolved_node_ranks": resolve_layer17_node_ranks(
                _MODE_RANK_CAP
            ),
            "generator_rank": _GENERATOR_RANK,
            "ridge": _RIDGE,
            "edge_policy": "edgeless",
            "interaction_count": 0,
            "fit_policy": "all_eight_equal_family_normalized",
            "diagnostic_policy": (
                "balanced_deterministic_subset_within_fit_fixed_rank_"
                "curve_only"
            ),
        },
        "authority": dict(authority),
        "protocol": _embedded_protocol(protocol),
        "fit_collection": dict(fit_collection),
        "fit_receipt": dict(fit_receipt),
        "resources": resources.state_dict(),
        "fragment_selection": selection.metadata(),
        "lowering_records": _coerce_lowering_records(
            lowerings_by_node,
            edgeless_graph,
        ),
        "edgeless_graph": edgeless_graph.state_dict(),
        "compiler_pipeline": compiler_pipeline.state_dict(),
        "lineage": _runtime_lineage(
            graph=edgeless_graph,
            lowerings_by_node=lowerings_by_node,
            pipeline=compiler_pipeline,
            selection=selection,
            resources=resources,
            fit_receipt=fit_receipt,
            lofo_lineage=lofo_lineage,
            base_artifact_file=base_artifact_file,
            base_artifact_file_sha256=base_artifact_file_sha256,
        ),
        "safety": dict(_SAFETY),
    }
    _reject_source_metadata(_scientific_projection(payload))
    payload["scientific_payload_sha256"] = _sha256(
        _scientific_projection(payload),
        domain=_SCIENTIFIC_DOMAIN,
    )
    return validate_gemma3_layer17_v8_all_family_refit_candidate(payload)


def _validate_fit_receipt(
    value: object,
    *,
    lowerings: Mapping[str, ModalGeneratorLowering],
) -> Mapping[str, object]:
    receipt = _strict_mapping(
        value,
        fields=_FIT_RECEIPT_FIELDS,
        label="all-family fit receipt",
    )
    aliases = tuple(receipt.get("family_aliases", ()))
    fit_split = _require_sha256(
        receipt.get("fit_split_sha256"),
        label="all-family fit split",
    )
    diagnostic_split = _require_sha256(
        receipt.get("diagnostic_split_sha256"),
        label="all-family diagnostic split",
    )
    supplied_receipt_sha256 = _require_sha256(
        receipt.get("fit_receipt_sha256"),
        label="all-family fit receipt",
    )
    receipt_payload = {
        key: value
        for key, value in receipt.items()
        if key != "fit_receipt_sha256"
    }
    if supplied_receipt_sha256 != _sha256(
        receipt_payload,
        domain=_FIT_RECEIPT_DOMAIN,
    ):
        raise ValueError("all-family fit receipt hash mismatch")
    fit_total_mass = _finite_number(
        receipt.get("fit_fisher_total_mass"),
        label="fit_fisher_total_mass",
    )
    fit_family_mass = _finite_number(
        receipt.get("fit_fisher_total_mass_per_family"),
        label="fit_fisher_total_mass_per_family",
    )
    if (
        aliases != tuple(V8_FAMILY_LOFO_FAMILY_ALIASES)
        or receipt.get("family_count") != _EXPECTED_FAMILIES
        or receipt.get("fit_example_count") != _EXPECTED_EXAMPLES
        or type(receipt.get("fit_observations")) is not int
        or int(receipt["fit_observations"]) <= 0
        or receipt.get("fit_sequences") != _EXPECTED_EXAMPLES
        or receipt.get("fisher_normalization") != _FISHER_NORMALIZATION
        or fit_total_mass != 1.0
        or not math.isclose(
            fit_family_mass,
            1.0 / _EXPECTED_FAMILIES,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or receipt.get("all_coefficients_fit_on_all_normalized_rows")
        is not True
        or receipt.get("diagnostic_subset_within_fit") is not True
        or receipt.get("diagnostic_subset_balanced_by_family") is not True
        or type(receipt.get("diagnostic_observations_per_family")) is not int
        or receipt.get("diagnostic_observations_per_family")
        != DEFAULT_DIAGNOSTIC_OBSERVATIONS_PER_FAMILY
        or receipt.get("diagnostic_observations")
        != int(receipt["diagnostic_observations_per_family"])
        * _EXPECTED_FAMILIES
        or receipt.get("diagnostic_used_for_fixed_rank_curve_construction")
        is not True
        or receipt.get("diagnostic_used_for_selection") is not False
        or receipt.get("diagnostic_used_for_early_stopping") is not False
        or receipt.get("diagnostic_supports_assessment_claim") is not False
        or fit_split == diagnostic_split
        or {
            lowering.coordinate_generator_plan.binding.fit_split_sha256
            for lowering in lowerings.values()
        }
        != {fit_split}
        or {
            lowering.coordinate_generator_plan.binding.eval_split_sha256
            for lowering in lowerings.values()
        }
        != {diagnostic_split}
    ):
        raise ValueError("all-family fit receipt integrity check failed")
    return receipt


def validate_gemma3_layer17_v8_all_family_refit_candidate(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strict-restore and authenticate an executable full-refit payload."""

    root = _strict_mapping(
        value,
        fields=_ROOT_FIELDS,
        label="all-family refit candidate",
    )
    if (
        root.get("schema") != GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
        or root.get("format_version")
        != GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_FORMAT_VERSION
    ):
        raise ValueError("unsupported all-family refit candidate")
    config = _strict_mapping(
        root.get("config"),
        fields={
            "mode_rank_cap",
            "resolved_node_ranks",
            "generator_rank",
            "ridge",
            "edge_policy",
            "interaction_count",
            "fit_policy",
            "diagnostic_policy",
        },
        label="all-family refit config",
    )
    ridge = _finite_number(config.get("ridge"), label="config.ridge")
    if (
        type(config.get("mode_rank_cap")) is not int
        or config.get("mode_rank_cap") != _MODE_RANK_CAP
        or type(config.get("generator_rank")) is not int
        or type(config.get("interaction_count")) is not int
        or tuple(config.get("resolved_node_ranks", ()))
        != resolve_layer17_node_ranks(_MODE_RANK_CAP)
        or config.get("generator_rank") != _GENERATOR_RANK
        or ridge != _RIDGE
        or config.get("edge_policy") != "edgeless"
        or config.get("interaction_count") != 0
        or config.get("fit_policy")
        != "all_eight_equal_family_normalized"
        or config.get("diagnostic_policy")
        != "balanced_deterministic_subset_within_fit_fixed_rank_curve_only"
    ):
        raise ValueError("all-family refit config drifted")

    authority = root.get("authority")
    if not isinstance(authority, Mapping):
        raise TypeError("all-family authority must be a mapping")
    validate_gemma3_layer17_family_lofo_authority_metadata(authority)
    protocol = root.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("artifact_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or protocol.get("authorized_by_passing_outer_lofo") is not True
        or protocol.get("runtime_fisher_normalization")
        != _FISHER_NORMALIZATION
    ):
        raise ValueError("all-family protocol binding drifted")

    fit_collection = _strict_mapping(
        root.get("fit_collection"),
        fields=_FIT_COLLECTION_FIELDS,
        label="all-family fit collection",
    )
    materialization = fit_collection.get("materialization")
    if not isinstance(materialization, Mapping):
        raise TypeError("fit materialization must be a mapping")
    validate_gemma3_layer17_family_lofo_materialization_metadata(
        materialization
    )
    family_observations = fit_collection.get("family_observations")
    if (
        materialization.get("authority_sha256")
        != authority.get("authority_sha256")
        or fit_collection.get("capture_count") != 1
        or fit_collection.get("captured_examples") != _EXPECTED_EXAMPLES
        or type(fit_collection.get("captured_observations")) is not int
        or int(fit_collection["captured_observations"]) <= 0
        or fit_collection.get("captured_sequences") != _EXPECTED_EXAMPLES
        or fit_collection.get("model_rows_recollected") is not False
        or not isinstance(family_observations, Mapping)
        or set(family_observations) != set(V8_FAMILY_LOFO_FAMILY_ALIASES)
        or any(
            type(count) is not int or count <= 0
            for count in family_observations.values()
        )
        or sum(int(count) for count in family_observations.values())
        != fit_collection.get("captured_observations")
    ):
        raise ValueError("all-family fit collection integrity check failed")
    _require_sha256(
        fit_collection.get("captured_row_key_sha256"),
        label="captured row-key",
    )

    graph = ModalGeneratorGraphPlan.from_state_dict(
        root.get("edgeless_graph")  # type: ignore[arg-type]
    )
    if graph.interactions:
        raise ValueError("serialized all-family graph is not edgeless")
    lowerings = _restore_lowering_records(root.get("lowering_records"), graph)
    selection = _selection_from_lowerings(lowerings)
    if root.get("fragment_selection") != selection.metadata():
        raise ValueError("all-family fragment selection drifted")
    resources_raw = root.get("resources")
    if not isinstance(resources_raw, Mapping):
        raise TypeError("all-family resources must be a mapping")
    resources = Layer17NodeRankResourceRow.from_state_dict(resources_raw)
    expected_resources = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        edge_policy="edgeless",
    )
    if (
        resources.state_dict() != expected_resources.state_dict()
        or graph.parameter_count != resources.graph_parameter_count
        or graph.macs_per_token != resources.graph_dense_macs_per_token
        or tuple(
            lowerings[node.name].computational_mode_basis.rank
            for node in graph.nodes
        )
        != resolve_layer17_node_ranks(_MODE_RANK_CAP)
        or tuple(
            lowerings[node.name].coordinate_generator_plan.rank
            for node in graph.nodes
        )
        != (_GENERATOR_RANK,) * 4
    ):
        raise ValueError("all-family executable resources or ranks drifted")
    pipeline = ModalCompilerPipeline.from_state_dict(
        root.get("compiler_pipeline")  # type: ignore[arg-type]
    )
    _validate_pipeline(
        pipeline,
        graph=graph,
        lowerings_by_node=lowerings,
        resource_row=resources,
    )
    fit_receipt = _validate_fit_receipt(
        root.get("fit_receipt"),
        lowerings=lowerings,
    )

    lineage = root.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("all-family lineage must be a mapping")
    required_lineage = {
        "topology_sha256",
        "model_fingerprint",
        "fragment_plan_sha256",
        "fragment_selection_sha256",
        "fragment_ids",
        "native_mode_counts",
        "fit_split_sha256",
        "diagnostic_split_sha256",
        "fit_receipt_sha256",
        "lowering_sha256s",
        "graph_sha256",
        "compiler_pipeline_sha256",
        "modal_refit_fisher_authority_sha256",
        "resource_row_sha256",
        "base_artifact_file",
        "base_artifact_file_sha256",
        "lofo_report_file",
        "lofo_report_file_sha256",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_all_required_gates_pass",
        "lofo_authorized_next_action",
        "lofo_authority_sha256",
        "lofo_receipt_sha256",
        "lofo_corpus_artifact_sha256",
        "lofo_fit_manifest_sha256",
        "lofo_materialization_sha256",
        "lofo_captured_row_key_sha256",
        "lofo_model_fingerprint",
        "lofo_requested_revision",
        "lofo_base_artifact_file_sha256",
        "lofo_fragment_plan_sha256",
        "lofo_fragment_selection_sha256",
        "frozen_v9_candidate_file",
        "frozen_v9_candidate_file_sha256",
        "frozen_v9_candidate_scientific_sha256",
    }
    if set(lineage) != required_lineage:
        raise ValueError("all-family lineage fields are invalid")
    for key in required_lineage:
        if key.endswith("sha256") or key.endswith("fingerprint"):
            _require_sha256(lineage.get(key), label=f"lineage {key}")
    for key in (
        "base_artifact_file",
        "lofo_report_file",
        "frozen_v9_candidate_file",
    ):
        filename = lineage.get(key)
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix not in (".pt", ".json")
        ):
            raise ValueError(f"lineage {key} must be a source-safe basename")
    requested_revision = lineage.get("lofo_requested_revision")
    if (
        not isinstance(requested_revision, str)
        or _REVISION.fullmatch(requested_revision) is None
    ):
        raise ValueError("lineage LOFO revision is invalid")
    expected_lineage = {
        "topology_sha256": LAYER17_TOPOLOGY_SHA256,
        "model_fingerprint": graph.model_fingerprint,
        "fragment_plan_sha256": graph.parameter_cluster_plan_sha256,
        "fragment_selection_sha256": selection.artifact_sha256,
        "fragment_ids": LAYER17_FRAGMENT_IDS,
        "native_mode_counts": LAYER17_NATIVE_MODE_COUNTS,
        "fit_split_sha256": fit_receipt["fit_split_sha256"],
        "diagnostic_split_sha256": fit_receipt[
            "diagnostic_split_sha256"
        ],
        "fit_receipt_sha256": fit_receipt["fit_receipt_sha256"],
        "lowering_sha256s": tuple(
            lowerings[name].artifact_sha256 for name in graph.traversal_order
        ),
        "graph_sha256": graph.artifact_sha256,
        "compiler_pipeline_sha256": pipeline.artifact_sha256,
        "modal_refit_fisher_authority_sha256": (
            pipeline.modal_refit_fisher_authority.artifact_sha256
            if pipeline.modal_refit_fisher_authority is not None
            else None
        ),
        "resource_row_sha256": resources.artifact_sha256,
    }
    if any(
        lineage.get(key) != expected
        for key, expected in expected_lineage.items()
    ):
        raise ValueError("all-family runtime lineage drifted")
    if (
        lineage.get("lofo_protocol_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or lineage.get("lofo_all_required_gates_pass") is not True
        or lineage.get("lofo_authorized_next_action")
        != "freeze_full_eight_family_refit_then_replay_eligible_open_"
        "development_assessment"
        or lineage.get("lofo_authority_sha256")
        != authority.get("authority_sha256")
        or lineage.get("lofo_receipt_sha256")
        != authority.get("receipt", {}).get("receipt_sha256")
        or lineage.get("lofo_corpus_artifact_sha256")
        != authority.get("corpus", {}).get("corpus_artifact_sha256")
        or lineage.get("lofo_fit_manifest_sha256")
        != authority.get("corpus", {}).get("fit_manifest_sha256")
        or lineage.get("lofo_materialization_sha256")
        != materialization.get("materialization_sha256")
        or lineage.get("lofo_captured_row_key_sha256")
        != fit_collection.get("captured_row_key_sha256")
        or lineage.get("lofo_model_fingerprint") != graph.model_fingerprint
        or lineage.get("lofo_fragment_plan_sha256")
        != graph.parameter_cluster_plan_sha256
        or lineage.get("lofo_fragment_selection_sha256")
        != selection.artifact_sha256
        or lineage.get("lofo_base_artifact_file_sha256")
        != lineage.get("base_artifact_file_sha256")
    ):
        raise ValueError("all-family LOFO lineage drifted")
    _validate_full_refit_pipeline_authority(
        pipeline,
        authority=authority,
        fit_collection=fit_collection,
        fit_receipt=fit_receipt,
        lofo_lineage=lineage,
    )
    experiment = root.get("experiment")
    expected_experiment_fields = {
        "experiment_kind",
        "scientific_role",
        "model_id",
        "requested_revision",
        "adapter_model_fingerprint",
        "source_model_unchanged",
        "heldout_confirmation",
        "assessment_metrics_present",
        "serving_authorized",
        "full_eight_family_refit_completed",
        "fit_family_count",
        "fit_example_count",
        "selection_opened",
        "lofo_report_sha256",
        "lofo_protocol_sha256",
        "lofo_authority_sha256",
        "frozen_v9_candidate_file_sha256",
        "frozen_v9_candidate_scientific_sha256",
    }
    if (
        not isinstance(experiment, Mapping)
        or set(experiment) != expected_experiment_fields
        or (
            experiment.get("experiment_kind")
            != "gemma3_layer17_v8_fit_all_family_refit_v1"
            or experiment.get("scientific_role")
            != "calibration_a_fit_all_family_refit_candidate"
            or experiment.get("model_id") != DEFAULT_MODEL_ID
            or experiment.get("adapter_model_fingerprint")
            != graph.model_fingerprint
            or experiment.get("requested_revision")
            != lineage.get("lofo_requested_revision")
            or experiment.get("source_model_unchanged") is not True
            or experiment.get("heldout_confirmation") is not False
            or experiment.get("assessment_metrics_present") is not False
            or experiment.get("serving_authorized") is not False
            or experiment.get("full_eight_family_refit_completed") is not True
            or experiment.get("fit_family_count") != _EXPECTED_FAMILIES
            or experiment.get("fit_example_count") != _EXPECTED_EXAMPLES
            or experiment.get("selection_opened") is not False
            or experiment.get("lofo_report_sha256")
            != lineage.get("lofo_report_sha256")
            or experiment.get("lofo_protocol_sha256")
            != lineage.get("lofo_protocol_sha256")
            or experiment.get("lofo_authority_sha256")
            != lineage.get("lofo_authority_sha256")
            or experiment.get("frozen_v9_candidate_file_sha256")
            != lineage.get("frozen_v9_candidate_file_sha256")
            or experiment.get("frozen_v9_candidate_scientific_sha256")
            != lineage.get("frozen_v9_candidate_scientific_sha256")
        )
    ):
        raise ValueError("all-family experiment contract drifted")
    if root.get("safety") != _SAFETY:
        raise ValueError("all-family safety boundary drifted")
    _reject_source_metadata(_scientific_projection(root))
    supplied = _require_sha256(
        root.get("scientific_payload_sha256"),
        label="scientific payload",
    )
    if supplied != _sha256(
        _scientific_projection(root),
        domain=_SCIENTIFIC_DOMAIN,
    ):
        raise ValueError("all-family scientific payload hash mismatch")
    # The strict graph, lowering, and pipeline restorers above authenticate the
    # tensor-bearing fields.  Only the source-safe scientific projection is
    # JSON-hashed; attempting to JSON-clone the executable payload would reject
    # its tensors by construction.
    return dict(root)


def build_gemma3_layer17_v8_all_family_refit_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    validated = validate_gemma3_layer17_v8_all_family_refit_candidate(payload)
    if (
        not isinstance(tensor_file, str)
        or Path(tensor_file).name != tensor_file
        or Path(tensor_file).suffix != ".pt"
    ):
        raise ValueError("tensor_file must be a source-safe .pt basename")
    graph = ModalGeneratorGraphPlan.from_state_dict(
        validated["edgeless_graph"]  # type: ignore[arg-type]
    )
    without_digest = {
        "schema": _REPORT_SCHEMA,
        "format_version": GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_FORMAT_VERSION,
        "experiment": validated["experiment"],
        "config": validated["config"],
        "authority": validated["authority"],
        "protocol": validated["protocol"],
        "fit_collection": validated["fit_collection"],
        "fit_receipt": validated["fit_receipt"],
        "resources": validated["resources"],
        "fragment_selection": validated["fragment_selection"],
        "lineage": validated["lineage"],
        "graph": {
            "artifact_sha256": graph.artifact_sha256,
            "node_count": len(graph.nodes),
            "interaction_count": len(graph.interactions),
            "traversal_order": graph.traversal_order,
            "parameter_count": graph.parameter_count,
            "macs_per_token": graph.macs_per_token,
        },
        "safety": {
            **dict(_SAFETY),
            "contains_executable_generator_weights": False,
            "contains_tensors": False,
        },
        "artifact": {
            "tensor_file": tensor_file,
            "scientific_payload_sha256": validated[
                "scientific_payload_sha256"
            ],
        },
    }
    _reject_source_metadata(without_digest)
    report = {
        **without_digest,
        "report_sha256": _sha256(without_digest, domain=_REPORT_DOMAIN),
    }
    native = _json_clone(report)
    if not isinstance(native, dict):
        raise AssertionError("all-family report did not canonicalize")
    return native


def save_gemma3_layer17_v8_all_family_refit_candidate(
    output: Path | str,
    **build_arguments: object,
) -> dict[str, object]:
    destination = Path(output)
    report_path = destination.with_suffix(".json")
    if (
        destination.suffix != ".pt"
        or ".local-runs" not in destination.parts
    ):
        raise ValueError("all-family output must be .pt under .local-runs")
    if destination.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite all-family refit")
    payload = build_gemma3_layer17_v8_all_family_refit_candidate(
        **build_arguments  # type: ignore[arg-type]
    )
    report = build_gemma3_layer17_v8_all_family_refit_report(
        payload,
        tensor_file=destination.name,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor_created = False
    report_created = False
    try:
        with destination.open("xb") as handle:
            tensor_created = True
            torch.save(payload, handle)
        # Strict restore before publishing the companion report.
        load_gemma3_layer17_v8_all_family_refit_candidate(destination)
        with report_path.open("x", encoding="utf-8") as handle:
            report_created = True
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except BaseException:
        if report_created and report_path.exists():
            report_path.unlink()
        if tensor_created and destination.exists():
            destination.unlink()
        raise
    return report


def load_gemma3_layer17_v8_all_family_refit_candidate(
    path: Path | str,
) -> dict[str, object]:
    source = Path(path)
    if source.suffix != ".pt" or not source.is_file():
        raise FileNotFoundError(source)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("all-family candidate must contain one dict")
    return validate_gemma3_layer17_v8_all_family_refit_candidate(raw)


def restore_gemma3_layer17_v8_all_family_refit_runtime(
    value: Mapping[str, object] | Path | str,
) -> tuple[
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
    ModalCompilerPipeline,
]:
    raw = (
        load_gemma3_layer17_v8_all_family_refit_candidate(value)
        if isinstance(value, (Path, str))
        else validate_gemma3_layer17_v8_all_family_refit_candidate(value)
    )
    graph = ModalGeneratorGraphPlan.from_state_dict(
        raw["edgeless_graph"]  # type: ignore[arg-type]
    )
    lowerings = _restore_lowering_records(raw["lowering_records"], graph)
    pipeline = ModalCompilerPipeline.from_state_dict(
        raw["compiler_pipeline"]  # type: ignore[arg-type]
    )
    return graph, lowerings, pipeline


def _blocks_to_device(
    blocks: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
    device: torch.device,
) -> tuple[tuple[str, tuple[CalibrationBatch, ...]], ...]:
    return tuple(
        (
            alias,
            tuple(
                CalibrationBatch(
                    model_inputs={
                        name: value.to(device=device)
                        for name, value in batch.model_inputs.items()
                    },
                    targets=batch.targets.to(device=device),
                    valid_positions=batch.valid_positions.to(device=device),
                    shared_input_names=batch.shared_input_names,
                    example_ids=batch.example_ids,
                )
                for batch in batches
            ),
        )
        for alias, batches in blocks
    )


def run_gemma3_layer17_v8_all_family_refit(
    *,
    revision: str,
    output: Path | str = DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT,
    lofo_report_path: Path | str = DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    base_artifact_path: Path | str = DEFAULT_BASE_ARTIFACT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Fit and save the protocol-authorized all-eight-family candidate."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite all-family refit")

    _progress("preflight: authenticate the passing fit-only LOFO decision")
    lofo_report = load_gemma3_layer17_v8_fit_lofo_report(lofo_report_path)
    lofo_lineage = _passing_lofo_lineage(
        lofo_report,
        report_path=lofo_report_path,
    )
    if (
        revision != lofo_lineage["lofo_requested_revision"]
        or model_id != DEFAULT_MODEL_ID
    ):
        raise ValueError("requested model/revision differ from passing LOFO")

    _progress("preflight: authenticate the same A-fit-only authority")
    authority = load_gemma3_layer17_family_lofo_authority(
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    authority_safe = authority.metadata()
    protocol = _load_authenticated_protocol(
        corpus_artifact_path,
        authority_metadata=authority_safe,
    )
    if authority_safe.get("authority_sha256") != lofo_lineage[
        "lofo_authority_sha256"
    ]:
        raise ValueError("current A-fit authority differs from passing LOFO")

    _progress("preflight: restore frozen layer-17 topology")
    upstream = load_gemma3_modal_generator_dev_artifact(base_artifact_path)
    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    selection = select_top_fisher_same_layer_fragments(
        fragment_plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    _validate_frozen_selection(selection)
    base_file_sha256 = _file_sha256(base_artifact_path)
    if (
        base_file_sha256
        != lofo_lineage["lofo_base_artifact_file_sha256"]
        or fragment_plan.artifact_sha256
        != lofo_lineage["lofo_fragment_plan_sha256"]
        or selection.artifact_sha256
        != lofo_lineage["lofo_fragment_selection_sha256"]
    ):
        raise ValueError("current frozen topology differs from passing LOFO")

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load the pinned local Gemma checkpoint")
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
    fingerprint = adapter.model_fingerprint()
    _validate_upstream_bindings(
        upstream,
        model_id=model_id,
        revision=revision,
        model_fingerprint=fingerprint,
    )
    if fingerprint != lofo_lineage["lofo_model_fingerprint"]:
        raise ValueError("live model fingerprint differs from passing LOFO")
    live_specs, leaf_site, _ = _whole_model_layer_specs(adapter)
    if tuple(spec.layer_id for spec in live_specs) != tuple(
        spec.layer_id for spec in fit_trace.layer_specs
    ):
        raise ValueError("live layer catalog differs from frozen topology")

    _progress("tokenize: materialize all eight opaque A-fit family blocks")
    raw_blocks, materialization_safe = materialize_gemma3_layer17_family_lofo(
        authority,
        tokenizer,
    )
    blocks = _blocks_to_device(_family_blocks(raw_blocks), device)
    materialization_sha256 = _require_sha256(
        materialization_safe.get("materialization_sha256"),
        label="A-fit materialization",
    )
    if materialization_sha256 != lofo_lineage[
        "lofo_materialization_sha256"
    ]:
        raise ValueError("current A-fit materialization differs from passing LOFO")
    batches = tuple(batch for _, family_batches in blocks for batch in family_batches)
    family_alias_by_example = {
        example_id: alias
        for alias, family_batches in blocks
        for batch in family_batches
        for example_id in batch.example_ids or ()
    }

    _progress("rows: collect four layer-17 fragment streams exactly once")
    all_rows = _collect_same_layer_native_rows(
        adapter,
        batches,
        selection=selection,
        leaf_activation_site=leaf_site,
    )
    if all_rows.row_key_sha256 != lofo_lineage[
        "lofo_captured_row_key_sha256"
    ]:
        raise ValueError("captured A-fit rows differ from passing LOFO")
    family_rows = partition_aligned_fragment_rows_by_family(
        all_rows,
        family_alias_by_example,
    )
    fit_rows, diagnostic_rows, fit_receipt = (
        build_layer17_all_family_refit_rows(
            family_rows,
            authority_sha256=str(authority_safe["authority_sha256"]),
            materialization_sha256=materialization_sha256,
            lofo_report_sha256=str(lofo_lineage["lofo_report_sha256"]),
            diagnostic_observations_per_family=(
                DEFAULT_DIAGNOSTIC_OBSERVATIONS_PER_FAMILY
            ),
        )
    )

    _progress(
        "fit: all eight equal-family-normalized blocks; cap48/r16/edgeless"
    )
    pilots = fit_layer17_capped_node_pilots(
        fit_rows,
        diagnostic_rows,
        selection=selection,
        source_model_sha256=fingerprint,
        parameter_catalog_sha256=catalog.artifact_sha256,
        fisher_coupling_sha256=fisher.artifact_sha256,
        fragment_plan=fragment_plan,
        fit_split_sha256=str(fit_receipt["fit_split_sha256"]),
        selection_split_sha256=str(
            fit_receipt["diagnostic_split_sha256"]
        ),
        mode_rank_cap=_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        ridge=_RIDGE,
    )
    _validate_fold_pilots_are_fit_only(pilots)
    edgeless = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=fragment_plan,
        lowerings_by_fragment={
            fragment_id: pilot.lowering
            for fragment_id, pilot in pilots.items()
        },
    )
    accounting = build_modal_source_replacement_accounting(
        catalog,
        fragment_plan,
        LAYER17_FRAGMENT_IDS,
    )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=fit_trace,
        parameter_catalog=catalog,
        grouped_fisher=fisher,
        fisher_clusters=clusters,
        parameter_cluster_fragments=fragment_plan,
        lowerings_by_node=edgeless.lowerings_by_node,
        graph_plan=edgeless.graph_plan,
        modal_refit_fisher_authority=ModalRefitFisherAuthority(
            fit_split_sha256=str(fit_receipt["fit_split_sha256"]),
            eval_split_sha256=str(
                fit_receipt["diagnostic_split_sha256"]
            ),
            eval_role="balanced_within_fit_fixed_rank_diagnostic",
            fisher_normalization=_FISHER_NORMALIZATION,
            source_model_sha256=fingerprint,
            parameter_catalog_sha256=catalog.artifact_sha256,
            topology_grouped_fisher_sha256=fisher.artifact_sha256,
            topology_fisher_calibration_split_sha256=(
                fisher.calibration_split_sha256
            ),
            topology_fisher_cluster_plan_sha256=(
                clusters.artifact_sha256
            ),
            topology_fragment_plan_sha256=fragment_plan.artifact_sha256,
            authorizing_report_sha256=str(
                lofo_lineage["lofo_report_sha256"]
            ),
            refit_protocol_sha256=str(
                lofo_lineage["lofo_protocol_sha256"]
            ),
            fit_authority_sha256=str(
                lofo_lineage["lofo_authority_sha256"]
            ),
            fit_receipt_sha256=str(
                fit_receipt["fit_receipt_sha256"]
            ),
            fit_corpus_artifact_sha256=str(
                lofo_lineage["lofo_corpus_artifact_sha256"]
            ),
            fit_manifest_sha256=str(
                lofo_lineage["lofo_fit_manifest_sha256"]
            ),
            fit_materialization_sha256=materialization_sha256,
            fit_example_count=_EXPECTED_EXAMPLES,
            fit_family_count=_EXPECTED_FAMILIES,
            equal_family_weighting=True,
            eval_subset_within_fit=True,
            eval_used_for_selection=False,
        ),
        interaction_selection=None,
        source_replacement_accounting=accounting,
    )
    resources = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=_MODE_RANK_CAP,
        generator_rank=_GENERATOR_RANK,
        edge_policy="edgeless",
    )
    if (
        pipeline.source_parameter_count != resources.source_parameter_count
        or pipeline.graph_parameter_count != resources.graph_parameter_count
        or pipeline.graph_plan.macs_per_token
        != resources.graph_dense_macs_per_token
        or adapter.model_fingerprint() != fingerprint
    ):
        raise RuntimeError("all-family compiler resources or source model drifted")

    fit_collection = {
        "materialization": dict(materialization_safe),
        "capture_count": 1,
        "captured_examples": _EXPECTED_EXAMPLES,
        "captured_observations": all_rows.observations,
        "captured_sequences": all_rows.sequences,
        "captured_row_key_sha256": all_rows.row_key_sha256,
        "family_observations": {
            alias: rows.observations for alias, rows in family_rows.items()
        },
        "model_rows_recollected": False,
    }
    _progress("artifact: save executable .pt and tensor-free JSON report")
    return save_gemma3_layer17_v8_all_family_refit_candidate(
        destination,
        experiment={
            "experiment_kind": "gemma3_layer17_v8_fit_all_family_refit_v1",
            "scientific_role": "calibration_a_fit_all_family_refit_candidate",
            "model_id": model_id,
            "requested_revision": revision,
            "adapter_model_fingerprint": fingerprint,
            "source_model_unchanged": True,
            "heldout_confirmation": False,
            "assessment_metrics_present": False,
            "serving_authorized": False,
            "full_eight_family_refit_completed": True,
            "fit_family_count": _EXPECTED_FAMILIES,
            "fit_example_count": _EXPECTED_EXAMPLES,
            "selection_opened": False,
            "lofo_report_sha256": lofo_lineage["lofo_report_sha256"],
            "lofo_protocol_sha256": lofo_lineage[
                "lofo_protocol_sha256"
            ],
            "lofo_authority_sha256": lofo_lineage[
                "lofo_authority_sha256"
            ],
            "frozen_v9_candidate_file_sha256": lofo_lineage[
                "frozen_v9_candidate_file_sha256"
            ],
            "frozen_v9_candidate_scientific_sha256": lofo_lineage[
                "frozen_v9_candidate_scientific_sha256"
            ],
        },
        authority=authority_safe,
        protocol=protocol,
        fit_collection=fit_collection,
        fit_receipt=fit_receipt,
        lofo_lineage=lofo_lineage,
        selection=selection,
        lowerings_by_node=edgeless.lowerings_by_node,
        edgeless_graph=edgeless.graph_plan,
        compiler_pipeline=pipeline,
        base_artifact_file=Path(base_artifact_path).name,
        base_artifact_file_sha256=base_file_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the passing-LOFO-authorized cap48/r16 layer-17 candidate "
            "on all eight authenticated Calibration-A fit families."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT,
    )
    parser.add_argument(
        "--lofo-report",
        type=Path,
        default=DEFAULT_LAYER17_V8_FIT_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--corpus-receipt",
        type=Path,
        default=DEFAULT_RECEIPT_OUTPUT,
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=DEFAULT_CORPUS_OUTPUT,
    )
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_BASE_ARTIFACT,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_layer17_v8_all_family_refit(
        revision=arguments.revision,
        output=arguments.output,
        lofo_report_path=arguments.lofo_report,
        corpus_receipt_path=arguments.corpus_receipt,
        corpus_artifact_path=arguments.corpus_artifact,
        fit_input_path=arguments.fit_input,
        base_artifact_path=arguments.base_artifact,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
