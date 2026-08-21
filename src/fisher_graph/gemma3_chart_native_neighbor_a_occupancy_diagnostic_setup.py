"""Freeze the A-only K8 chart-occupancy diagnostic protocol.

This module is deliberately standard-library-only and import inert.  It does
not open a panel, import a tensor runtime, load a model, or run the diagnostic.
The diagnostic exists solely to recover the occupancy evidence that the
failed v1 A fit did not retain.  It cannot select a replacement chart count or
open any role after A.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence


SCHEMA = "fisher_graph.gemma3_chart_native_neighbor.a_occupancy_diagnostic.setup.v1"
FORMAT_VERSION = 1
SETUP_ID = "gemma3-chart-native-neighbor-a-k8-occupancy-diagnostic-v1"
DEFAULT_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-chart-native-neighbor-a-k8-occupancy-diagnostic-setup-v1.json"
)
DEFAULT_RESULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-a-k8-occupancy-diagnostic-v1.json"
)

# Filled only after the canonical artifact is generated and independently
# reloaded.  Keeping the public pins in source makes --check fail closed.
PINNED_SETUP_SHA256 = (
    "bbb87b3314ce5dd317afe9d524e116ed6d53cf546f2f21cdd48bab9404608f32"
)
PINNED_SETUP_FILE_SHA256 = (
    "34166b8f16f4f11ce058360e743fc29d3056a96fe942f9c0b01c36519d9e9b96"
)
PINNED_RUNNER_SOURCE_SHA256 = (
    "452703018c17689e71522df126fd8da017019f4be3c7304904c4c59364d33b86"
)
PINNED_NUMERIC_SOURCE_SHA256 = (
    "e414a95485eb5055292d1ae69c60e85295eb9c03f5f8d1bac56e1755023206a1"
)
PINNED_SELECTION_KERNEL_RECEIPT_SHA256 = (
    "133311b1c982ebc42cf98219faa8eed37c8b36b4348385c02f78edceebded9f6"
)

_SETUP_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-a-occupancy-diagnostic-setup:v1\0"
)

V1_SETUP_PATH = Path(
    "artifacts/research/gemma3-chart-native-neighbor-setup-v1.json"
)
V1_SETUP_SHA256 = (
    "5ebc02aa5800add5d16f9b3f506224029adcf1fa257ac80a7d46155ce0a1e484"
)
V1_SETUP_FILE_SHA256 = (
    "d597c9210f0fddae71a514b1a1738612eda22e99978f94d109ef61b489fee042"
)
V1_RUNTIME_QUALIFICATION_PATH = Path(
    "artifacts/research/"
    "gemma3-chart-native-neighbor-runtime-qualification-v1.json"
)
V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256 = (
    "d430889be9f91fdbc76b8c59d4ced252c860b9fc3d7ea6c655f43b25c8f693ca"
)
V1_PRODUCTION_EXECUTION_CONTRACT_SHA256 = (
    "639d619ca6148f74576974e65bbabeb485e4e3bd6f495cc9a959e45abe4db95d"
)
V1_A_ROLE_PANEL_BINDING_SHA256 = (
    "bc11c01c66d25daffc96fa69778c2ab34e8d482bcec1de4b9d6bf1b6235b1432"
)
V1_A_ORDERED_ROW_CATALOG_SHA256 = (
    "4d26d2a4c28712f6fc98c9e9c913bbaac77cad3b1aaa0ba8e09758922669d4d9"
)

FIT_ROLE = "A_chart_transport_fit"
FORBIDDEN_ROLES = (
    "B_neighbor_discovery",
    "C_neighbor_confirmation",
    "D_response_validation",
    "reserve_unopened",
)
CHART_COUNT = 8
TANGENT_RANK = 16
HIDDEN_WIDTH = 640
MODE_COUNT = 38
FAMILY_COUNT = 6
CANONICAL_MODE_IDS = tuple(
    sorted(
        (
            "mode.lexical.ingress",
            *(f"mode.mutation.{index}" for index in range(1, 37)),
            "mode.lexical.egress",
        )
    )
)
CANONICAL_FAMILY_IDS = (
    "fresh-v1/counterfactual-token-exchange",
    "fresh-v1/cyclic-beacon-state",
    "fresh-v1/nonce-lexicon-binding",
    "fresh-v1/route-parity-resolution",
    "fresh-v1/temporal-slot-placement",
    "fresh-v1/witness-reliability-ranking",
)
ROWS_PER_MODE = 24_732
TOTAL_POOL_ROWS = MODE_COUNT * ROWS_PER_MODE
MINIMUM_HARD_ROWS = TANGENT_RANK + 1
MINIMUM_BALANCED_HARD_OCCUPANCY = 0.005


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def setup_sha256(value: Mapping[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("setup_sha256", None)
    return _digest(_SETUP_DOMAIN, unsigned)


def _build_state() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "predecessor_v1": {
            "setup_path": str(V1_SETUP_PATH),
            "setup_sha256": V1_SETUP_SHA256,
            "setup_file_sha256": V1_SETUP_FILE_SHA256,
            "runtime_qualification_path": str(
                V1_RUNTIME_QUALIFICATION_PATH
            ),
            "runtime_qualification_receipt_sha256": (
                V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256
            ),
            "production_execution_contract_sha256": (
                V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
            ),
            "retained_failure_facts": {
                "role": FIT_ROLE,
                "terminal_phase": "final_balanced_hard_occupancy_gate",
                "exception": (
                    "a chart failed the minimum balanced hard occupancy gate"
                ),
                "all_8_charts_had_at_least_17_hard_rows": True,
                "at_least_one_balanced_hard_mass_below_0_005": True,
                "required_numeric_outcome": (
                    "BALANCED_HARD_MASS_BELOW_MINIMUM"
                ),
                "required_numeric_terminal_phase": "final_hard_mass_gate",
                "exact_count_and_mass_vectors_were_not_retained": True,
                "atlas_frozen": False,
                "transports_fitted": False,
                "B_semantically_opened": False,
                "C_semantically_opened": False,
                "D_semantically_opened": False,
                "reserve_semantically_opened": False,
            },
            "terminal_journal_evidence": {
                "dispatch_timestamp_UTC": "2026-08-10T11:22:01.376Z",
                "dispatch_record_jsonl_SHA256": (
                    "2862739f786553412c003a305dd505e54afe720090ae3a7f5523224a592d93fa"
                ),
                "terminal_timestamp_UTC": "2026-08-10T11:30:39.349Z",
                "terminal_record_jsonl_SHA256": (
                    "3eab4751d8f72183c6f4420289405856f212ddf07388954e5a2c25e9594116bc"
                ),
                "unified_session_id": 2893,
                "wait_call_id": "call_KsAF1L8jlIUcuP8TZeIHhFFp",
                "exit_code": 1,
                "traceback_source_line": 1662,
                "external_journal_path_is_not_an_executable_dependency": True,
            },
            "diagnostic_is_post_A_informed": True,
            "v1_result_remains_terminal_no_graph": True,
        },
        "scope": {
            "allowed_role": FIT_ROLE,
            "forbidden_roles": list(FORBIDDEN_ROLES),
            "purpose": (
                "recover_complete_K8_hard_occupancy_evidence_without_"
                "selecting_or_fitting_a_replacement_atlas"
            ),
            "A_captured_exactly_once": True,
            "A_raw_spools_reused_read_only_for_diagnostic": True,
            "chart_count_selection_performed": False,
            "atlas_publication_performed": False,
            "transport_fit_performed": False,
            "neighbor_graph_performed": False,
            "alternate_valid_numeric_outcome_is_a_freeze_integrity_"
            "disagreement": True,
            "freeze_integrity_disagreement_forbids_result_publication": True,
        },
        "frozen_K8_config": {
            "hidden_width": HIDDEN_WIDTH,
            "chart_count": CHART_COUNT,
            "tangent_rank": TANGENT_RANK,
            "source_feature_width": CHART_COUNT * (TANGENT_RANK + 1),
            "lloyd_iteration_count": 12,
            "membership_temperature": 1.0,
            "radius_floor_fraction": 1.0e-8,
            "variance_floor_fraction": 1.0e-8,
            "minimum_hard_rows_per_chart": MINIMUM_HARD_ROWS,
            "minimum_balanced_hard_occupancy_weight": (
                MINIMUM_BALANCED_HARD_OCCUPANCY
            ),
            "eigenvalue_tie_relative_tolerance": 1.0e-12,
            "transport_relative_ridge": 1.0e-3,
            "ridge_scale_floor": 1.0e-12,
            "transport_fisher_energy_floor": 1.0e-12,
            "only_diagnostic_API": (
                "diagnose_shared_chart_atlas_occupancy_streaming"
            ),
            "descending_K_selector_forbidden": True,
            "alternate_seed_forbidden": True,
            "threshold_relaxation_forbidden": True,
            "rank_change_forbidden": True,
            "continuing_after_first_failed_gate_forbidden": True,
        },
        "frozen_streaming_resources": {
            "row_chunk_size": 256,
            "atlas_chunk_size": 256,
            "max_resident_bytes": 8 * 1024**3,
            "max_artifact_bytes": 4 * 1024**3,
            "exact_payload_bound_by_traced_selection_kernel_receipt": True,
            "resource_drift_aborts_before_model_runtime": True,
        },
        "fit_pool": {
            "mode_count": MODE_COUNT,
            "canonical_mode_ids": list(CANONICAL_MODE_IDS),
            "canonical_mode_axis_source": (
                "lexicographically_sorted_pinned_v1_setup.logical_mode_"
                "ontology.ordered_modes"
            ),
            "family_count": FAMILY_COUNT,
            "canonical_family_ids": list(CANONICAL_FAMILY_IDS),
            "canonical_family_axis_source": (
                "pinned_A_fresh_role_corpus.family_ids_sorted"
            ),
            "rows_per_mode": ROWS_PER_MODE,
            "total_pool_rows": TOTAL_POOL_ROWS,
            "balanced_mass_total": 1.0,
            "role_panel_binding_sha256": V1_A_ROLE_PANEL_BINDING_SHA256,
            "ordered_objective_row_catalog_sha256": (
                V1_A_ORDERED_ROW_CATALOG_SHA256
            ),
            "weighting": (
                "equal_mode_then_equal_family_then_equal_example_then_"
                "equal_row_within_example"
            ),
            "diagnostic_reads_context_values_only": True,
            "generalized_contributions_not_read_by_diagnostic": True,
            "comparison_values_not_read_by_diagnostic": True,
        },
        "required_numeric_receipt": {
            "candidate_chart_count": CHART_COUNT,
            "candidate_feature_width": CHART_COUNT * (TANGENT_RANK + 1),
            "required_candidate_outcome": (
                "BALANCED_HARD_MASS_BELOW_MINIMUM"
            ),
            "required_candidate_terminal_phase": "final_hard_mass_gate",
            "ADMISSIBLE_or_other_typed_failure_is_terminal_disagreement": (
                True
            ),
            "top_level_required_fields": [
                "schema",
                "config",
                "config_sha256",
                "resources",
                "resource_config_receipt_sha256",
                "kernel_receipt_sha256",
                "selection_kernel_receipt_sha256",
                "fit_role",
                "fit_panel_binding_sha256",
                "fit_row_catalog_sha256",
                "fit_context_catalog_sha256",
                "fit_balanced_row_weight_catalog_sha256",
                "fit_mode_ids",
                "fit_family_ids",
                "fit_row_count_by_family",
                "spool_artifact_sha256s",
                "spool_mode_identity_sha256s",
                "spool_tensor_catalog_sha256s",
                "mode_count",
                "row_count_per_mode",
                "total_row_count",
                "total_balanced_mass_binary64",
                "compute_device",
                "compute_dtype",
                "candidate",
                "receipt_sha256",
            ],
            "candidate_required_fields": [
                "schema",
                "chart_count",
                "source_feature_width",
                "total_row_count",
                "mode_count",
                "family_count",
                "mode_ids",
                "family_ids",
                "fit_row_count_by_family",
                "total_balanced_mass_binary64",
                "config_sha256",
                "seed_receipts",
                "initialized_centers_sha256",
                "initialized_center_sha256s",
                "lloyd_pass_receipts",
                "final_sort_permutation",
                "final_pre_sort_center_sha256s",
                "sorted_centers_sha256",
                "sorted_center_sha256s",
                "final_hard_occupancy_count",
                "final_hard_occupancy_mass_binary64",
                "final_hard_count_by_chart_and_mode",
                "final_hard_mass_binary64_by_chart_and_mode",
                "final_hard_count_by_chart_and_family",
                "final_hard_mass_binary64_by_chart_and_family",
                "composition_catalog_sha256",
                "radius_squared_sha256",
                "soft_occupancy_mass_binary64",
                "covariance_sha256s",
                "leading_eigenvalues_sha256s",
                "tangent_bases_sha256",
                "tangent_variance_sha256",
                "orthogonal_alpha_sha256",
                "atlas_sha256",
                "fit_solver_receipt_sha256",
                "gate_evidence",
                "minimum_required_hard_count",
                "minimum_hard_occupancy_mass_binary64",
                "minimum_observed_hard_mass_binary64",
                "minimum_hard_mass_margin_binary64",
                "minimum_hard_mass_chart_ordinals",
                "outcome",
                "terminal_phase",
                "failing_chart_ordinals",
                "receipt_sha256",
            ],
            "seed_receipt_required_fields": [
                "schema",
                "seed_ordinal",
                "selected_global_row_ordinal",
                "selected_mode_ordinal",
                "selected_row_ordinal",
                "mode_id",
                "family_id",
                "example_id",
                "prediction_query_position",
                "target_label_position",
                "mode_identity_sha256",
                "row_identity_sha256",
                "weighted_farthest_score_binary64",
                "selected_center_sha256",
                "receipt_sha256",
            ],
            "lloyd_pass_receipt_required_fields": [
                "schema",
                "pass_ordinal",
                "centers_before_assignment_sha256",
                "centers_before_assignment_sha256s",
                "hard_occupancy_count",
                "hard_occupancy_mass_binary64",
                "empty_chart_ordinals",
                "updated_centers_sha256",
                "updated_centers_sha256s",
                "receipt_sha256",
            ],
            "gate_evidence_required_fields": [
                "schema",
                "gate_id",
                "predicate_id",
                "status",
                "chart_ordinal",
                "component_ordinal",
                "global_row_ordinal",
                "observed_binary64",
                "bound_binary64",
                "signed_margin_binary64",
                "receipt_sha256",
            ],
            "fit_mode_axis_count": MODE_COUNT,
            "fit_family_axis_count": FAMILY_COUNT,
            "candidate_axes_must_exactly_equal_fit_axes": True,
            "every_nested_receipt_requires_exact_field_equality": True,
            "seed_receipts_bind_global_row_ordinal_mode_and_row_identity_"
            "hashes_score_bits_and_center_hash": True,
            "all_12_pass_receipts_bind_counts_masses_and_updated_center_"
            "hashes": True,
            "full_vectors_required_not_only_minimum": True,
            "first_failed_gate_must_be_explicit": True,
            "chart_by_mode_and_chart_by_family_composition_values_required": (
                True
            ),
            "tensor_values_forbidden": True,
            "tensor_digests_allowed": True,
        },
        "execution_firewall": {
            "static_v1_preflight_precedes_model_runtime": True,
            "pinned_v1_runtime_qualification_reused": True,
            "A_dense_lexical_parity_required": True,
            "A_capture_batch_count": 45,
            "A_example_count": 48,
            "B_C_D_reserve_role_open_calls_forbidden": True,
            "panel_opened_roles_must_equal": [FIT_ROLE],
            "panel_freeze_receipts_must_remain_empty": True,
            "runtime_closed_before_result_seal": True,
            "workspace_removed_before_result_seal": True,
            "raw_spool_count_after_cleanup": 0,
            "output_and_temp_directories_must_be_distinct_regular_roots": (
                True
            ),
            "preexisting_links_and_special_files_forbidden": True,
            "before_catalog_binds_root_identity_modes_file_metadata_and_"
            "content_SHA256": True,
            "prepublication_catalog_must_exactly_equal_before": True,
            "success_catalog_delta_is_exactly_one_regular_result": True,
            "all_new_entries_are_transactionally_unlinked_and_parent_"
            "directories_fsynced_on_failure": True,
            "failure_catalog_must_exactly_equal_before": True,
            "preexisting_staging_entries_forbidden": True,
            "result_is_no_clobber_canonical_JSON": True,
        },
        "implementation_sources": {
            "setup_module_is_reviewed_non_self_hashed_trust_anchor": True,
            "source_hashes_checked_before_model_runtime": True,
            "sources": {
                "diagnostic_runner": {
                    "path": (
                        "src/fisher_graph/gemma3_chart_native_neighbor_a_"
                        "occupancy_diagnostic.py"
                    ),
                    "file_sha256": PINNED_RUNNER_SOURCE_SHA256,
                },
                "streaming_atlas_diagnostic": {
                    "path": (
                        "src/fisher_graph/"
                        "streaming_shared_chart_atlas_selection_v1.py"
                    ),
                    "file_sha256": PINNED_NUMERIC_SOURCE_SHA256,
                },
            },
            "v1_capture_dependency_closure_authenticated_by_v1_preflight": (
                True
            ),
            "traced_selection_kernel": {
                "function": (
                    "streaming_shared_chart_atlas_selection_kernel_receipt_v1"
                ),
                "receipt_sha256": (
                    PINNED_SELECTION_KERNEL_RECEIPT_SHA256
                ),
                "exact_receipt_required_in_diagnostic_result": True,
            },
        },
        "cli": {
            "module": (
                "fisher_graph.gemma3_chart_native_neighbor_a_occupancy_"
                "diagnostic"
            ),
            "required_exactly_one_action": [
                "--preflight-only",
                "--run-a-only",
                "--check",
            ],
            "default_model_execution": False,
            "default_result_path": str(DEFAULT_RESULT_OUTPUT),
        },
        "summary_safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_contexts": False,
            "contains_raw_comparisons": False,
            "contains_raw_generalized_contributions": False,
            "contains_centers": False,
            "contains_covariances": False,
            "contains_tangent_bases": False,
        },
        "claim_boundary": {
            "occupancy_diagnostic_measured": False,
            "atlas_validity_claimed": False,
            "chart_count_selected": False,
            "neighbor_edge_claimed": False,
            "stable_geometry_claimed": False,
            "response_validation_claimed": False,
            "compression_claimed": False,
            "weights_alone_determine_neighbors_claimed": False,
        },
        "setup_receipt": {
            "prompt_data_opened": False,
            "tensor_library_imported": False,
            "model_weights_loaded": False,
            "activations_captured": False,
            "diagnostic_executed": False,
            "scientific_result_present": False,
        },
    }


def build_setup() -> dict[str, object]:
    state = _build_state()
    state["setup_sha256"] = setup_sha256(state)
    return state


def validate_setup(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("A-only occupancy diagnostic setup must be a mapping")
    try:
        observed = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("diagnostic setup is not canonical JSON") from error
    if canonical_json_bytes(observed) != canonical_json_bytes(build_setup()):
        raise ValueError("diagnostic setup differs from frozen source state")
    return observed


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )


def _read_regular(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("diagnostic setup must be a regular file")
    if before.st_size > 4 * 1024 * 1024:
        raise ValueError("diagnostic setup exceeds byte cap")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("diagnostic setup changed during open")
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, opened.st_size - len(raw)))
            if not chunk:
                raise ValueError("diagnostic setup was truncated")
            raw += chunk
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError("diagnostic setup changed during read")
    return raw


def load_setup(source: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(source)
    raw = _read_regular(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("diagnostic setup JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("diagnostic setup must be a JSON object")
    return validate_setup(value)


def write_setup(destination: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = build_setup()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = os.lstat(temporary)
        os.link(temporary, path)
        published = True
        published_stat = path.lstat()
        if (
            stat.S_ISLNK(published_stat.st_mode)
            or not stat.S_ISREG(published_stat.st_mode)
            or (published_stat.st_dev, published_stat.st_ino)
            != (temporary_stat.st_dev, temporary_stat.st_ino)
        ):
            raise RuntimeError("diagnostic setup publication identity drifted")
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if load_setup(path) != payload:
            raise RuntimeError("diagnostic setup strict reload drifted")
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        if published:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            else:
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        raise
    return payload


def check_setup(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    path = Path(source)
    value = load_setup(path)
    if value["setup_sha256"] != PINNED_SETUP_SHA256:
        raise ValueError("diagnostic setup semantic pin drifted")
    if hashlib.sha256(_read_regular(path)).hexdigest() != PINNED_SETUP_FILE_SHA256:
        raise ValueError("diagnostic setup file pin drifted")
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    sources = value["implementation_sources"]["sources"]
    for name, receipt in sources.items():
        relative = Path(receipt["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("diagnostic implementation source path is unsafe")
        observed = hashlib.sha256(_read_regular(root / relative)).hexdigest()
        if observed != receipt["file_sha256"]:
            raise ValueError(f"diagnostic {name} source hash drifted")
    return value


@dataclass(frozen=True, slots=True)
class Gemma3ChartNativeNeighborAOccupancyDiagnosticSetup:
    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3ChartNativeNeighborAOccupancyDiagnosticSetup":
        return cls(build_setup())

    def to_dict(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self.state))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.check:
        check_setup(arguments.output)
    else:
        write_setup(arguments.output)
    return 0


__all__ = [
    "CANONICAL_FAMILY_IDS",
    "CANONICAL_MODE_IDS",
    "CHART_COUNT",
    "DEFAULT_OUTPUT",
    "DEFAULT_RESULT_OUTPUT",
    "FIT_ROLE",
    "FAMILY_COUNT",
    "Gemma3ChartNativeNeighborAOccupancyDiagnosticSetup",
    "HIDDEN_WIDTH",
    "MINIMUM_BALANCED_HARD_OCCUPANCY",
    "MINIMUM_HARD_ROWS",
    "MODE_COUNT",
    "PINNED_SETUP_FILE_SHA256",
    "PINNED_SETUP_SHA256",
    "PINNED_NUMERIC_SOURCE_SHA256",
    "PINNED_RUNNER_SOURCE_SHA256",
    "PINNED_SELECTION_KERNEL_RECEIPT_SHA256",
    "ROWS_PER_MODE",
    "SCHEMA",
    "SETUP_ID",
    "TANGENT_RANK",
    "TOTAL_POOL_ROWS",
    "V1_A_ORDERED_ROW_CATALOG_SHA256",
    "V1_A_ROLE_PANEL_BINDING_SHA256",
    "V1_PRODUCTION_EXECUTION_CONTRACT_SHA256",
    "V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256",
    "V1_SETUP_SHA256",
    "build_setup",
    "canonical_json_bytes",
    "check_setup",
    "load_setup",
    "main",
    "setup_sha256",
    "validate_setup",
    "write_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
