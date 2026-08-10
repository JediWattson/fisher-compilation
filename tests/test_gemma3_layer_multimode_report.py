from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

import fisher_graph.gemma3_layer_multimode_report as report_io
import fisher_graph.gemma3_layer_multimode_fresh_panels as panel_module
import fisher_graph.gemma3_layer_multimode_implementation_qualification as qualified
import fisher_graph.gemma3_layer_multimode_runtime as runtime_module
import fisher_graph.gemma3_layer_multimode_setup as multimode_setup
from fisher_graph.gemma3_layer_multimode_compiler import (
    Gemma3LayerMultimodeCompilationReceipt,
    LayerMultimodeLayerAccounting,
)
from fisher_graph.gemma3_layer_multimode_evaluation import (
    CompiledCandidateIdentityReceipt,
    D3NoNativeMLPTraceReceipt,
    D3ResourceComparisonReceipt,
    FullStackExampleAggregate,
    LayerTeacherForcedAdditiveEvidence,
    LayerTeacherForcedFamilyAggregate,
    LayerTeacherForcedModeAggregate,
    evaluate_b3_teacher_forced,
    evaluate_c3_full_stack,
    evaluate_d3_final,
    heldout_evaluation_policy_sha256,
)
from fisher_graph.gemma3_layer_multimode_materialization import (
    LayerMLPFlowMaterializationInventoryReceipt,
    LayerMLPFlowMaterializationReceipt,
)
from fisher_graph.gemma3_layer_multimode_selection import (
    _grant_live_candidate_evaluation_authority,
    select_layer_multimode_candidates,
)


_ROOT = Path(__file__).resolve().parents[1]
_FAMILIES = tuple(sorted(panel_module.FAMILY_IDS))


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(raw).hexdigest()


def _selection_helpers() -> object:
    path = _ROOT / "tests/test_gemma3_layer_multimode_selection.py"
    spec = importlib.util.spec_from_file_location("_multimode_selection_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel_helpers() -> object:
    path = _ROOT / "tests/test_gemma3_layer_multimode_fresh_panels.py"
    spec = importlib.util.spec_from_file_location(
        "_multimode_report_panel_helpers",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_regular(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _live_authorization_root(tmp_path: Path) -> tuple[Path, object]:
    panel_receipt_bytes = (_ROOT / panel_module.DEFAULT_RECEIPT).read_bytes()
    panel_receipt = json.loads(panel_receipt_bytes)
    panel_bundle = panel_module.AuthoredFreshPanelBundle(
        manifest_bytes=(_ROOT / panel_module.DEFAULT_MANIFEST).read_bytes(),
        role_corpus_items=tuple(
            (
                role,
                (_ROOT / panel_module.DEFAULT_ROLE_CORPORA[role]).read_bytes(),
            )
            for role in panel_module.ORDERED_ROLES
        ),
        receipt_bytes=panel_receipt_bytes,
        bundle_sha256=panel_receipt["bundle_sha256"],
    )
    root = tmp_path / "repository"
    for relative in qualified._discover_source_closure(_ROOT).values():
        _copy_regular(_ROOT / relative, root / relative)
    for relative in qualified._TEST_PATHS:
        _copy_regular(_ROOT / relative, root / relative)
    for relative in (
        multimode_setup.PREDECESSOR_V2_SETUP,
        multimode_setup.PREDECESSOR_V2_REPORT,
    ):
        _copy_regular(_ROOT / relative, root / relative)
    setup_path = root / multimode_setup.DEFAULT_OUTPUT
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_bytes(multimode_setup.render_setup_bytes())
    setup_path.chmod(0o600)
    panel_path = root / panel_module.DEFAULT_RECEIPT
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_bytes(panel_bundle.receipt_bytes)
    manifest_path = root / panel_module.DEFAULT_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(panel_bundle.manifest_bytes)
    for role, raw in panel_bundle.role_corpus_items:
        role_path = root / panel_module.DEFAULT_ROLE_CORPORA[role]
        role_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.write_bytes(raw)
    qualification_path = root / qualified.DEFAULT_OUTPUT
    qualification_path.parent.mkdir(parents=True, exist_ok=True)
    qualified.write_qualification(qualification_path, repository_root=root)
    return root, panel_bundle


def _selection(source_model_sha256: str | None = None) -> object:
    helpers = _selection_helpers()
    grid = helpers._grid(production=True)
    if source_model_sha256 is not None:
        grid = tuple(
            replace(
                value,
                source_model_sha256=source_model_sha256,
                artifact_sha256="",
            )
            for value in grid
        )
    grid = tuple(
        _grant_live_candidate_evaluation_authority(value) for value in grid
    )
    return select_layer_multimode_candidates(grid)


def _materialization(
    source_model_sha256: str,
    *,
    source_execution_fingerprint_sha256: str | None = None,
) -> object:
    shared = {
        "source_full_A_bridge_receipt_sha256": _sha("bridge"),
        "a_development_partition_receipt_sha256": _sha("partition"),
        "fit_partition_sha256": _sha("fit-partition"),
        "validation_partition_sha256": _sha("validation-partition"),
        "source_role_capture_receipt_sha256": _sha("A-capture"),
        "source_runtime_session_binding_sha256": _sha("session"),
        "predecessor_preflight_receipt_sha256": _sha("old-preflight"),
        "predecessor_runtime_qualification_receipt_sha256": _sha("old-qual"),
        "source_model_fingerprint_sha256": source_model_sha256,
        "source_execution_fingerprint_sha256": (
            source_execution_fingerprint_sha256 or _sha("source-execution")
        ),
        "compiled_execution_fingerprint_sha256": _sha("old-compiled-execution"),
        "tied_readout_binding_receipt_sha256": _sha("tied-readout"),
        "source_role_panel_binding_sha256": _sha("A-panel"),
        "source_spool_row_catalog_sha256": _sha("row-catalog"),
    }
    layers = []
    for layer in range(18):
        context = _sha(f"context-{layer}")
        comparison = _sha(f"comparison-{layer}")
        chart = _sha(f"chart-{layer}")
        velocity = _sha(f"velocity-{layer}")
        layers.append(
            LayerMLPFlowMaterializationReceipt(
                layer_ordinal=layer,
                mode_id=f"mode.mutation.{2 + 2 * layer}",
                **shared,
                source_spool_artifact_sha256=_sha(f"spool-{layer}"),
                source_spool_tensor_catalog_sha256=_sha(f"tensors-{layer}"),
                source_spool_kernel_receipt_sha256=_sha("spool-kernel"),
                source_spool_context_float64_semantic_sha256=context,
                source_spool_comparison_float64_semantic_sha256=comparison,
                verified_context_float64_semantic_sha256=context,
                verified_comparison_float64_semantic_sha256=comparison,
                materialized_chart_state_float32_sha256=chart,
                materialized_teacher_velocity_float32_sha256=velocity,
                verified_cast_chart_state_float32_sha256=chart,
                verified_cast_teacher_velocity_float32_sha256=velocity,
                flow_rows_receipt_sha256=_sha(f"rows-{layer}"),
                row_count=24_732,
                hidden_width=640,
                verification_chunk_size=1_024,
                verification_chunk_count=math.ceil(24_732 / 1_024),
            )
        )
    return LayerMLPFlowMaterializationInventoryReceipt(
        layer_receipts=tuple(layers),
        shared_spool_row_catalog_sha256=shared[
            "source_spool_row_catalog_sha256"
        ],
        **{
            key: value
            for key, value in shared.items()
            if key != "source_spool_row_catalog_sha256"
        },
    )


def _compilation(selection: object, materialization: object) -> object:
    candidates = {
        value.artifact_sha256: value for value in selection.candidates
    }
    rows = []
    for selected in selection.layers:
        candidate = candidates[selected.selected_candidate_sha256]
        h = 640
        intermediate = 2_048
        c = candidate.chart_count
        m = candidate.dynamic_mode_count
        rank = candidate.tangent_rank
        owned = c * h + c * h * rank + 1 + c * m * (rank + 1)
        owned += c * m * (rank * rank + rank)
        native = 2 * h + 3 * h * intermediate
        compiled = h + owned
        rows.append(
            LayerMultimodeLayerAccounting(
                layer_ordinal=selected.layer_ordinal,
                hidden_width=h,
                source_intermediate_width=intermediate,
                chart_count=c,
                modes_per_chart=m,
                tangent_rank=rank,
                retained_input_rmsnorm_parameters=h,
                removed_output_rmsnorm_parameters=h,
                removed_native_mlp_parameters=3 * h * intermediate,
                replacement_owned_parameters=owned,
                native_boundary_parameters=native,
                compiled_boundary_parameters=compiled,
                boundary_parameter_savings=native - compiled,
                replacement_float32_bytes=4 * owned,
                coefficient_sha256=_sha(f"coefficient-{selected.layer_ordinal}"),
            )
        )
    source_parameters = 100_000_000
    compiled_parameters = source_parameters + sum(
        row.compiled_boundary_parameters - row.native_boundary_parameters
        for row in rows
    )
    return Gemma3LayerMultimodeCompilationReceipt(
        source_model_sha256=selection.source_model_sha256,
        recomputed_source_model_sha256=selection.source_model_sha256,
        source_execution_fingerprint_sha256=(
            materialization.source_execution_fingerprint_sha256
        ),
        recomputed_source_execution_fingerprint_sha256=(
            materialization.source_execution_fingerprint_sha256
        ),
        compiled_execution_fingerprint_sha256=_sha("compiled-execution"),
        selection_receipt_sha256=selection.artifact_sha256,
        selected_inventory_sha256=_sha("inventory"),
        source_state_sha256=_sha("source-state"),
        protected_architecture_sha256=_sha("architecture"),
        protected_compiled_state_sha256=_sha("protected-state"),
        compiled_state_sha256=_sha("compiled-state"),
        layer_accounting=tuple(rows),
        source_parameter_count=source_parameters,
        compiled_parameter_count=compiled_parameters,
        source_parameter_storage_bytes=source_parameters * 4,
        compiled_parameter_storage_bytes=compiled_parameters * 4,
        source_persistent_storage_bytes=source_parameters * 4,
        compiled_persistent_storage_bytes=compiled_parameters * 4,
        source_state_tensor_payload_bytes=source_parameters * 4,
        compiled_state_tensor_payload_bytes=compiled_parameters * 4,
        tied_embedding_parameter_identity=True,
        production_shape_enforced=True,
        production_provenance_authorized=True,
        replacement_count=18,
        removed_output_rmsnorm_count=18,
        native_mlp_fallback_present=False,
        source_parameter_alias_present=False,
    )


def _b3(
    selection: object,
    *,
    failing: bool,
    panel_manifest_sha256: str | None = None,
    role_capture_receipt_sha256: str | None = None,
) -> object:
    selected = {
        value.artifact_sha256: value for value in selection.candidates
    }
    evidence = []
    rows_per_family = sum(
        value - 1 for value in panel_module.TARGET_TOKEN_COUNTS
    )
    total_rows = rows_per_family * len(_FAMILIES)
    for layer in selection.layers:
        candidate = selected[layer.selected_candidate_sha256]
        nrmse = 0.50 if failing and layer.layer_ordinal == 0 else 0.20
        energy = 1_000.0
        families = tuple(
            LayerTeacherForcedFamilyAggregate(
                family_id=family,
                scalar_count=rows_per_family * multimode_setup.HIDDEN_WIDTH,
                prediction_sse=nrmse * nrmse * energy,
                teacher_energy=energy,
                prediction_energy=energy,
                prediction_teacher_dot=0.97 * energy,
            )
            for family in _FAMILIES
        )
        chart_weight = float(total_rows) / candidate.chart_count
        mode_aggregates = []
        for chart in range(candidate.chart_count):
            masses = tuple(
                1.0 / candidate.dynamic_mode_count
                for _ in range(candidate.dynamic_mode_count)
            )
            entropy = -sum(value * math.log(value) for value in masses)
            router_entropy = (
                0.0
                if candidate.dynamic_mode_count == 1
                else entropy - 0.10
            )
            mode_aggregates.append(
                LayerTeacherForcedModeAggregate(
                    chart_ordinal=chart,
                    chart_weight_sum=chart_weight,
                    chart_mode_weight_sums=tuple(
                        chart_weight * value for value in masses
                    ),
                    weighted_router_entropy_sum_nats=(
                        chart_weight * router_entropy
                    ),
                )
            )
        evidence.append(
            LayerTeacherForcedAdditiveEvidence(
                layer_ordinal=layer.layer_ordinal,
                chart_count=candidate.chart_count,
                dynamic_mode_count=candidate.dynamic_mode_count,
                tangent_rank=candidate.tangent_rank,
                selected_candidate_sha256=candidate.artifact_sha256,
                role_capture_receipt_sha256=(
                    role_capture_receipt_sha256 or _sha("B3-capture")
                ),
                family_aggregates=families,
                mode_aggregates=tuple(mode_aggregates),
                row_weight_sum=float(total_rows),
            )
        )
    return evaluate_b3_teacher_forced(
        evidence,
        source_model_sha256=selection.source_model_sha256,
        selection_receipt_sha256=selection.artifact_sha256,
        panel_manifest_sha256=panel_manifest_sha256 or _sha("panels"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
    )


def _preflight(
    source_model_sha256: str,
    *,
    setup_sha256: str | None = None,
    setup_file_sha256: str | None = None,
    panel_manifest_sha256: str | None = None,
    panel_manifest_file_sha256: str | None = None,
    panel_receipt_sha256: str | None = None,
    panel_receipt_file_sha256: str | None = None,
    panel_bundle_sha256: str | None = None,
    qualification_sha256: str | None = None,
    qualification_file_sha256: str | None = None,
    predecessor_v2_setup_sha256: str | None = None,
    predecessor_v2_report_sha256: str | None = None,
    predecessor_A_capture_receipt_sha256: str | None = None,
    predecessor_runtime_execution_receipt_sha256: str | None = None,
) -> dict[str, object]:
    gib = 1024**3
    resource: dict[str, object] = {
        "schema": "fisher_graph.gemma3_layer_multimode.resource_preflight.v1",
        "format_version": 1,
        "workspace_parent_device": 1,
        "workspace_parent_inode": 1,
        "output_parent_devices": [1, 1, 1],
        "validated_free_bytes_floor_by_device": [[1, 24 * gib]],
        "physical_memory_bytes": 16 * gib,
        "A_recapture_payload_estimate_bytes": 2 * 18 * 24_732 * 640 * 4,
        "A_materialization_payload_estimate_bytes": 2 * 18 * 24_732 * 640 * 4,
        "full_grid_fit_peak_estimate_bytes": 8 * gib,
        "model_and_atomic_staging_estimate_bytes": 4 * gib,
        "required_free_disk_bytes": 24 * gib,
        "required_physical_memory_bytes": 16 * gib,
        "workspace_path_safe": True,
        "output_paths_safe": True,
        "test_only_nonproduction": False,
    }
    resource["receipt_sha256"] = report_io._digest(
        b"fisher_graph:layer-multimode:resource-preflight:v1\0",
        resource,
    )
    value: dict[str, object] = {
        "schema": "fisher_graph.gemma3_layer_multimode.execution_preflight.v1",
        "format_version": 1,
        "setup_sha256": setup_sha256 or _sha("setup"),
        "setup_file_sha256": setup_file_sha256 or _sha("setup-file"),
        "panel_manifest_sha256": panel_manifest_sha256 or _sha("panels"),
        "panel_manifest_file_sha256": (
            panel_manifest_file_sha256 or _sha("panel-file")
        ),
        "panel_receipt_sha256": panel_receipt_sha256 or _sha("panel-receipt"),
        "panel_receipt_file_sha256": (
            panel_receipt_file_sha256 or _sha("panel-receipt-file")
        ),
        "panel_bundle_sha256": panel_bundle_sha256 or _sha("panel-bundle"),
        "implementation_qualification_sha256": (
            qualification_sha256 or _sha("qualification")
        ),
        "implementation_qualification_file_sha256": (
            qualification_file_sha256 or _sha("qualification-file")
        ),
        "predecessor_v2_setup_sha256": (
            predecessor_v2_setup_sha256 or _sha("v2-setup")
        ),
        "predecessor_v2_report_sha256": (
            predecessor_v2_report_sha256 or _sha("v2-report")
        ),
        "predecessor_A_capture_receipt_sha256": (
            predecessor_A_capture_receipt_sha256 or _sha("A-capture")
        ),
        "predecessor_runtime_execution_receipt_sha256": (
            predecessor_runtime_execution_receipt_sha256
            or _sha("predecessor-runtime")
        ),
        "source_model_sha256": source_model_sha256,
        "evaluation_protocol_sha256": heldout_evaluation_policy_sha256(),
        "resource_preflight_receipt": resource,
        "test_only_nonproduction": False,
        "exposed_predecessor_A_recapture_only": True,
        "fresh_role_order": [
            "B3_teacher_forced_layerwise",
            "C3_full_stack_source_free",
            "D3_final_heldout_validation",
            "reserve_v3_unopened",
        ],
        "full_grid_candidate_count": 324,
        "production_runtime_ready": True,
    }
    value["receipt_sha256"] = report_io._digest(
        b"fisher_graph:layer-multimode:preflight:v1\0",
        value,
    )
    return value


@pytest.fixture(scope="module")
def base_evidence() -> tuple[object, object, object, bytes, dict[str, object]]:
    selection = _selection()
    materialization = _materialization(selection.source_model_sha256)
    compilation = _compilation(selection, materialization)
    bundle = b"strict-test-layer-multimode-bundle"
    bundle_receipt = {
        "bundle_sha256": _sha(bundle),
        "byte_count": len(bundle),
        "manifest_sha256": _sha("bundle-manifest"),
        "compilation_receipt_sha256": compilation.receipt_sha256,
    }
    return selection, materialization, compilation, bundle, bundle_receipt


def _candidate_identity(
    selection: object,
    compilation: object,
    bundle_receipt: dict[str, object],
) -> CompiledCandidateIdentityReceipt:
    return CompiledCandidateIdentityReceipt(
        selection_receipt_sha256=selection.artifact_sha256,
        compiler_protocol_sha256=hashlib.sha256(
            b"fisher_graph:layer-multimode:compiler-source:v1\0"
            + (
                _ROOT
                / "src/fisher_graph/gemma3_layer_multimode_compiler.py"
            ).read_bytes()
        ).hexdigest(),
        compile_receipt_sha256=compilation.receipt_sha256,
        bundle_semantic_sha256=bundle_receipt["manifest_sha256"],
        compiled_model_state_sha256=compilation.compiled_state_sha256,
    )


def _full_stack_records(
    role: str,
    prefix: str,
    *,
    failing: bool,
) -> tuple[FullStackExampleAggregate, ...]:
    records = []
    for family in _FAMILIES:
        for target in panel_module.TARGET_TOKEN_COUNTS:
            rows = target - 1
            hard_long = target >= panel_module.HARD_LONG_MINIMUM_TOKEN_COUNT
            delta = 0.08 if failing else 0.005
            kl = 0.08 if failing else 0.004
            errors = max(1, rows // (5 if failing else 100))
            identity = _sha(f"{prefix}-{family}-{target}")
            records.append(
                FullStackExampleAggregate(
                    role=role,
                    example_identity_sha256=identity,
                    family_id=family,
                    hard_long=hard_long,
                    predicted_token_count=rows,
                    delta_nll_sum_nats=delta * rows,
                    native_to_candidate_kl_sum_nats=kl * rows,
                    top1_agreement_count=rows - errors,
                    example_execution_receipt_sha256=_sha(
                        f"execution-{identity}"
                    ),
                )
            )
    return tuple(sorted(records, key=lambda value: value.example_identity_sha256))


def _worker_receipt(domain: bytes, payload: dict[str, object]) -> dict[str, object]:
    state = dict(payload)
    state["receipt_sha256"] = report_io._digest(domain, state)
    return state


def _artifact_receipt(
    domain: bytes,
    payload: dict[str, object],
) -> dict[str, object]:
    state = dict(payload)
    state["artifact_sha256"] = report_io._digest(domain, state)
    return state


def _predecessor_A_runtime_receipt(
    *,
    preflight: dict[str, object],
    predecessor_preflight_sha256: str,
    predecessor_runtime_qualification_sha256: str,
    materialization: object,
    source_model_sha256: str,
) -> dict[str, object]:
    return _worker_receipt(
        b"fisher_graph:layer-multimode:predecessor-A-runtime:v1\0",
        {
            "schema": (
                "fisher_graph.gemma3_layer_multimode."
                "predecessor_A_runtime_execution.v1"
            ),
            "format_version": 1,
            "execution_preflight_receipt_sha256": preflight["receipt_sha256"],
            "predecessor_preflight_receipt_sha256": (
                predecessor_preflight_sha256
            ),
            "predecessor_runtime_qualification_receipt_sha256": (
                predecessor_runtime_qualification_sha256
            ),
            "source_capture_receipt_sha256": (
                materialization.source_role_capture_receipt_sha256
            ),
            "runtime_session_binding_sha256": (
                materialization.source_runtime_session_binding_sha256
            ),
            "source_model_fingerprint_before_sha256": source_model_sha256,
            "source_model_fingerprint_after_sha256": source_model_sha256,
            "source_execution_fingerprint_before_sha256": (
                materialization.source_execution_fingerprint_sha256
            ),
            "source_execution_fingerprint_after_sha256": (
                materialization.source_execution_fingerprint_sha256
            ),
            "role_batch_counts": [
                ["A_chart_transport_fit", 45],
                ["B_neighbor_discovery", 0],
                ["C_neighbor_confirmation", 0],
            ],
            "exact_control_forward_count": 45,
            "exact_control_vjp_count": 45,
            "dense_parity_smoke_count": 1,
            "dense_parity_forward_count": 1,
            "dense_parity_vjp_count": 1,
            "total_backbone_forward_count": 46,
            "total_terminal_cotangent_vjp_count": 46,
            "captured_example_count": 48,
            "supplemental_route_receipt_count": 864,
            "mode_raw_artifact_count": 38,
            "lexical_nll_receipt_count": 45,
            "execution_receipt_count": 45,
            "example_capture_receipt_count": 48,
            "canonical_gradient_targets_per_vjp": 73,
            "auxiliary_gradient_targets_per_vjp": 19,
            "gradient_targets_per_vjp": 92,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_activations": False,
            "contains_raw_gradients": False,
            "contains_logits": False,
        },
    )


def _predecessor_A_comparison_receipt(
    *,
    preflight: dict[str, object],
    materialization: object,
    predecessor_runtime_session_sha256: str,
) -> dict[str, object]:
    return _artifact_receipt(
        b"fisher_graph:layer-multimode:predecessor-A-comparison:v1\0",
        {
            "schema": (
                "fisher_graph.gemma3_layer_multimode."
                "predecessor_A_recapture_comparison.v1"
            ),
            "format_version": 1,
            "predecessor_capture_receipt_sha256": preflight[
                "predecessor_A_capture_receipt_sha256"
            ],
            "recaptured_capture_receipt_sha256": (
                materialization.source_role_capture_receipt_sha256
            ),
            "predecessor_runtime_execution_receipt_sha256": preflight[
                "predecessor_runtime_execution_receipt_sha256"
            ],
            "predecessor_runtime_session_binding_sha256": (
                predecessor_runtime_session_sha256
            ),
            "recaptured_runtime_session_binding_sha256": (
                materialization.source_runtime_session_binding_sha256
            ),
            "role": "A_chart_transport_fit",
            "batch_count": 45,
            "row_count": 24_732,
            "role_panel_binding_sha256": _sha("A-role-panel"),
            "ordered_objective_row_catalog_sha256": _sha("A-row-catalog"),
            "score_objective_receipt_sha256": _sha("A-score-objective"),
            "dense_lexical_parity_receipt_sha256": _sha("A-dense-parity"),
            "mode_raw_artifact_catalog_sha256": _sha("A-mode-raw"),
            "mode_semantic_projection_catalog_sha256": _sha(
                "A-mode-semantic"
            ),
            "lexical_nll_receipt_catalog_sha256": _sha("A-lexical-nll"),
            "supplemental_route_projection_catalog_sha256": _sha(
                "A-supplemental-route"
            ),
            "compared_field_inventory": [
                "batch_count",
                "dense_lexical_parity_receipt_sha256",
                "lexical_nll_receipt_sha256s",
                "mode_raw_artifacts",
                "mode_semantic_projection",
                "ordered_objective_row_catalog_sha256",
                "role",
                "role_panel_binding_sha256",
                "row_count",
                "score_objective_receipt_sha256",
                "supplemental_route_projection",
            ],
            "intentionally_session_dependent_field_inventory": [
                "batch_axis_catalog_sha256",
                "example_capture_receipt_sha256s",
                "execution_receipt_sha256s",
                "receipt_sha256",
                "runtime_session_binding_sha256",
                "supplemental_route_receipt_sha256s",
            ],
            "runtime_sessions_expected_different": True,
            "all_comparable_fields_matched": True,
        },
    )


def _fresh_role_capture_receipt(
    *,
    role: str,
    stage_kind: str,
    preflight: dict[str, object],
    selection_receipt_sha256: str,
    frozen_candidate_identity_sha256: str,
    runtime_session_binding_sha256: str,
    panel_manifest: dict[str, object] | None,
) -> dict[str, object]:
    if panel_manifest is None:
        role_file_sha256 = _sha(f"{role}-file")
        example_catalog_sha256 = _sha(f"{role}-example-catalog")
        row_catalog_sha256 = _sha(f"{role}-row-catalog")
        family_axis_sha256 = _sha("family-axis")
        batch_axis_sha256 = _sha("batch-axis")
    else:
        descriptor = panel_manifest["roles"][role]
        role_file_sha256 = descriptor["file_sha256"]
        (
            example_catalog_sha256,
            row_catalog_sha256,
            family_axis_sha256,
            batch_axis_sha256,
        ) = runtime_module._fresh_role_catalog_sha256s(
            manifest=panel_manifest,
            role=role,
        )
    return _worker_receipt(
        b"fisher_graph:layer-multimode:fresh-role-capture:v1\0",
        {
            "schema": "fisher_graph.gemma3_layer_multimode.fresh_role_capture.v1",
            "format_version": 1,
            "role": role,
            "stage_kind": stage_kind,
            "panel_manifest_sha256": preflight["panel_manifest_sha256"],
            "panel_manifest_file_sha256": preflight[
                "panel_manifest_file_sha256"
            ],
            "panel_receipt_sha256": preflight["panel_receipt_sha256"],
            "panel_receipt_file_sha256": preflight[
                "panel_receipt_file_sha256"
            ],
            "panel_bundle_sha256": preflight["panel_bundle_sha256"],
            "role_file_sha256": role_file_sha256,
            "selection_receipt_sha256": selection_receipt_sha256,
            "frozen_candidate_identity_sha256": (
                frozen_candidate_identity_sha256
            ),
            "runtime_session_binding_sha256": runtime_session_binding_sha256,
            "ordered_tokenization_receipt_sha256": _sha(
                f"{role}-tokenization"
            ),
            "ordered_example_identity_catalog_sha256": (
                example_catalog_sha256
            ),
            "ordered_row_identity_catalog_sha256": row_catalog_sha256,
            "family_axis_sha256": family_axis_sha256,
            "batch_axis_sha256": batch_axis_sha256,
            "example_count": 48,
            "batch_count": 8,
            "predicted_next_token_position_count": 24_732,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_raw_activations": False,
            "contains_raw_gradients": False,
        },
    )


def _terminal_payload(
    base: tuple[object, object, object, bytes, dict[str, object]],
    terminal: str,
    *,
    d3_failing: bool = False,
    preflight: dict[str, object] | None = None,
    panel_manifest_sha256: str | None = None,
    panel_manifest: dict[str, object] | None = None,
    predecessor_report: dict[str, object] | None = None,
    runtime_implementation_sha256: str | None = None,
) -> dict[str, object]:
    selection, materialization, compilation, _, bundle_receipt = base
    panel_sha = panel_manifest_sha256 or _sha("panels")
    preflight_state = preflight or _preflight(selection.source_model_sha256)
    opened = {
        report_io.TERMINAL_B3_FAILED: 1,
        report_io.TERMINAL_C3_FAILED: 2,
        report_io.TERMINAL_D3_COMPLETED: 3,
    }[terminal]
    roles = [
        "B3_teacher_forced_layerwise",
        "C3_full_stack_source_free",
        "D3_final_heldout_validation",
    ][:opened]
    stage_kinds = [
        "B3_teacher_forced_evaluation",
        "C3_full_stack_source_free_evaluation",
        "D3_final_heldout_evaluation",
    ][:opened]
    implementation = runtime_implementation_sha256 or _sha(
        "runtime-implementation"
    )
    environment = _sha("execution-environment")
    tokenizer = _sha("tokenizer")
    predecessor_preflight_sha = (
        predecessor_report["v2_preflight"]["receipt_sha256"]
        if predecessor_report is not None
        else _sha("predecessor-preflight")
    )
    predecessor_runtime_qualification_sha = (
        predecessor_report["runtime_qualification_receipt"][
            "qualification_receipt_sha256"
        ]
        if predecessor_report is not None
        else _sha("predecessor-runtime-qualification")
    )
    predecessor_runtime_session_sha = (
        predecessor_report["runtime_execution_receipt"][
            "runtime_session_binding_sha256"
        ]
        if predecessor_report is not None
        else _sha("predecessor-runtime-session")
    )
    if (
        predecessor_runtime_session_sha
        == materialization.source_runtime_session_binding_sha256
    ):
        predecessor_runtime_session_sha = _sha(
            f"different-{predecessor_runtime_session_sha}"
        )
    runtime_qualification = _worker_receipt(
        b"fisher_graph:layer-multimode:runtime-qualification:v1\0",
        {
            "schema": (
                "fisher_graph.gemma3_layer_multimode.runtime_qualification.v1"
            ),
            "format_version": 1,
            "execution_preflight_receipt_sha256": preflight_state[
                "receipt_sha256"
            ],
            "implementation_qualification_sha256": preflight_state[
                "implementation_qualification_sha256"
            ],
            "implementation_qualification_file_sha256": preflight_state[
                "implementation_qualification_file_sha256"
            ],
            "runtime_implementation_sha256": implementation,
            "qualified_runtime_source_sha256": implementation,
            "predecessor_preflight_receipt_sha256": predecessor_preflight_sha,
            "predecessor_runtime_qualification_receipt_sha256": (
                predecessor_runtime_qualification_sha
            ),
            "source_model_fingerprint_before_sha256": (
                selection.source_model_sha256
            ),
            "source_model_fingerprint_after_sha256": (
                selection.source_model_sha256
            ),
            "source_execution_fingerprint_before_sha256": (
                materialization.source_execution_fingerprint_sha256
            ),
            "source_execution_fingerprint_after_sha256": (
                materialization.source_execution_fingerprint_sha256
            ),
            "tokenizer_asset_catalog_sha256": tokenizer,
            "evaluation_protocol_sha256": heldout_evaluation_policy_sha256(),
            "source_model_mutated": False,
            "full_vocab_accumulator_dtype": "float64",
            "top1_tie_break": "lowest_vocabulary_index",
            "predecessor_A_exact_control_batch_count": 45,
            "predecessor_A_dense_parity_count": 1,
            "fresh_batches_per_opened_role": 8,
            "logits_persisted": False,
            "token_ids_persisted": False,
            "prompt_text_persisted": False,
        },
    )
    session_payload = {
        "runtime_implementation_sha256": implementation,
        "implementation_qualification_sha256": preflight_state[
            "implementation_qualification_sha256"
        ],
        "execution_environment_sha256": environment,
        "tokenizer_fingerprint_sha256": tokenizer,
        "runtime_qualification_receipt_sha256": runtime_qualification[
            "receipt_sha256"
        ],
        "source_model_fingerprint_before_sha256": selection.source_model_sha256,
        "source_model_fingerprint_after_sha256": selection.source_model_sha256,
        "source_execution_fingerprint_before_sha256": (
            materialization.source_execution_fingerprint_sha256
        ),
        "source_execution_fingerprint_after_sha256": (
            materialization.source_execution_fingerprint_sha256
        ),
        "compiled_model_fingerprint_before_sha256": (
            compilation.compiled_state_sha256
        ),
        "compiled_model_fingerprint_after_sha256": (
            compilation.compiled_state_sha256
        ),
        "compiled_execution_fingerprint_before_sha256": (
            compilation.compiled_execution_fingerprint_sha256
        ),
        "compiled_execution_fingerprint_after_sha256": (
            compilation.compiled_execution_fingerprint_sha256
        ),
        "compiled_candidate_mutated": False,
    }
    runtime_session_sha = report_io._digest(
        b"fisher_graph:layer-multimode:runtime-session:v1\0",
        session_payload,
    )
    candidate = _candidate_identity(selection, compilation, bundle_receipt)
    fresh_role_captures = [
        _fresh_role_capture_receipt(
            role=role,
            stage_kind=stage_kind,
            preflight=preflight_state,
            selection_receipt_sha256=selection.artifact_sha256,
            frozen_candidate_identity_sha256=candidate.artifact_sha256,
            runtime_session_binding_sha256=runtime_session_sha,
            panel_manifest=panel_manifest,
        )
        for role, stage_kind in zip(roles, stage_kinds, strict=True)
    ]
    B3 = _b3(
        selection,
        failing=terminal == report_io.TERMINAL_B3_FAILED,
        panel_manifest_sha256=panel_sha,
        role_capture_receipt_sha256=fresh_role_captures[0]["receipt_sha256"],
    )
    C3 = None
    D3 = None
    if terminal != report_io.TERMINAL_B3_FAILED:
        C3 = evaluate_c3_full_stack(
            _full_stack_records(
                "C3_full_stack_source_free",
                "C3",
                failing=terminal == report_io.TERMINAL_C3_FAILED,
            ),
            b3_evaluation=B3,
            source_model_sha256=selection.source_model_sha256,
            panel_manifest_sha256=panel_sha,
            role_capture_receipt_sha256=fresh_role_captures[1][
                "receipt_sha256"
            ],
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=candidate,
        )
        if terminal == report_io.TERMINAL_D3_COMPLETED:
            records = _full_stack_records(
                "D3_final_heldout_validation",
                "D3",
                failing=d3_failing,
            )
            native_boundary = sum(
                row.native_boundary_parameters
                for row in compilation.layer_accounting
            )
            compiled_boundary = sum(
                row.compiled_boundary_parameters
                for row in compilation.layer_accounting
            )
            resources = D3ResourceComparisonReceipt(
                candidate_identity_sha256=candidate.artifact_sha256,
                parameter_formula_receipt_sha256=_sha("parameter-formula"),
                serialization_protocol_sha256=_sha("serialization"),
                native_model_file_sha256=_sha("native-file"),
                compiled_model_file_sha256=_sha("compiled-comparison-file"),
                native_full_model_parameter_count=(
                    compilation.source_parameter_count
                ),
                native_replaced_boundary_parameter_count=native_boundary,
                compiled_replacement_boundary_parameter_count=compiled_boundary,
                expected_compiled_full_model_parameter_count=(
                    compilation.compiled_parameter_count
                ),
                measured_compiled_full_model_parameter_count=(
                    compilation.compiled_parameter_count
                ),
                native_serialized_model_bytes=2_500,
                compiled_serialized_model_bytes=2_000,
                formula_parameter_reduction_passed=True,
                measured_parameter_reduction_passed=True,
                serialized_byte_reduction_passed=True,
                passed=True,
            )
            trace = D3NoNativeMLPTraceReceipt(
                candidate_identity_sha256=candidate.artifact_sha256,
                role_capture_receipt_sha256=fresh_role_captures[2][
                    "receipt_sha256"
                ],
                trace_protocol_sha256=_sha("trace-protocol"),
                module_inventory_sha256=_sha("module-inventory"),
                trace_event_stream_sha256=_sha("trace-events"),
                proof_backend=(
                    "qualified-module-inventory-plus-all-replacement-"
                    "forward-hooks-v1"
                ),
                traced_forward_call_count=8,
                traced_example_count=len(records),
                traced_predicted_token_count=sum(
                    value.predicted_token_count for value in records
                ),
                compiled_layer_call_counts=tuple(8 for _ in range(18)),
                native_mlp_module_inventory_count=0,
                native_mlp_projection_inventory_count=0,
                native_mlp_fallback_inventory_count=0,
                native_mlp_parameter_storage_alias_count=0,
                passed=True,
            )
            D3 = evaluate_d3_final(
                records,
                c3_evaluation=C3,
                source_model_sha256=selection.source_model_sha256,
                panel_manifest_sha256=panel_sha,
                role_capture_receipt_sha256=fresh_role_captures[2][
                    "receipt_sha256"
                ],
                evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
                candidate_identity=candidate,
                resource_comparison=resources,
                no_native_mlp_trace=trace,
            )
    teardown = _worker_receipt(
        b"fisher_graph:layer-multimode:teardown:v1\0",
        {
            "schema": (
                "fisher_graph.gemma3_layer_multimode.development_teardown.v1"
            ),
            "format_version": 1,
            "source_model_sha256": selection.source_model_sha256,
            "source_execution_fingerprint_sha256": (
                materialization.source_execution_fingerprint_sha256
            ),
            "compiled_candidate_present": True,
            "compilation_receipt_sha256": compilation.receipt_sha256,
            "compiled_model_fingerprint_sha256": (
                compilation.compiled_state_sha256
            ),
            "source_compilation_aliases_released": True,
            "native_baseline_retained_for_heldout_evaluation": True,
            "A_activation_rows_released": True,
            "A_fit_models_released": True,
            "A_optimizer_state_released": True,
            "source_compilation_native_mlp_alias_count": 0,
            "native_baseline_instance_count": 1,
        },
    )
    counters = _worker_receipt(
        b"fisher_graph:layer-multimode:counters:v1\0",
        {
            "schema": "fisher_graph.gemma3_layer_multimode.experiment_counters.v1",
            "format_version": 1,
            "terminal_status": terminal,
            "execution_preflight_receipt_sha256": preflight_state[
                "receipt_sha256"
            ],
            "selection_receipt_sha256": selection.artifact_sha256,
            "strict_preflight_count": 1,
            "runtime_factory_count": 1,
            "exposed_predecessor_A_recapture_count": 1,
            "fresh_v3_A_capture_count": 0,
            "materialization_count": 18,
            "atlas_fit_count": 108,
            "candidate_fit_count": 324,
            "split_refit_count": 216,
            "candidate_evaluation_count": 324,
            "selection_attempt_count": 1,
            "compilation_count": 1,
            "candidate_freeze_count": 1,
            "development_teardown_count": 1,
            "B3_open_count": 1,
            "B3_evaluation_count": 1,
            "C3_open_count": 1 if opened >= 2 else 0,
            "C3_evaluation_count": 1 if opened >= 2 else 0,
            "D3_open_count": 1 if opened >= 3 else 0,
            "D3_evaluation_count": 1 if opened >= 3 else 0,
            "reserve_open_count": 0,
            "native_baseline_evaluator_count": 1,
            "native_baseline_eval_mode": True,
            "native_baseline_parameter_mutation_count": 0,
            "native_baseline_optimizer_reference_count": 0,
            "compiled_candidate_native_mlp_module_count": 0,
        },
    )
    evaluations = [B3, *(item for item in (C3, D3) if item is not None)]
    fresh_panel = _worker_receipt(
        b"fisher_graph:layer-multimode:fresh-panel-execution:v1\0",
        {
            "schema": (
                "fisher_graph.gemma3_layer_multimode.fresh_panel_execution.v1"
            ),
            "format_version": 1,
            "terminal_status": terminal,
            "panel_manifest_sha256": preflight_state["panel_manifest_sha256"],
            "panel_manifest_file_sha256": preflight_state[
                "panel_manifest_file_sha256"
            ],
            "panel_receipt_sha256": preflight_state["panel_receipt_sha256"],
            "panel_receipt_file_sha256": preflight_state[
                "panel_receipt_file_sha256"
            ],
            "panel_bundle_sha256": preflight_state["panel_bundle_sha256"],
            "selection_receipt_sha256": selection.artifact_sha256,
            "opened_roles": roles,
            "ordered_freeze_receipts": [
                {
                    "role": role,
                    "stage_kind": stage_kind,
                    "stage_receipt_sha256": evaluation.artifact_sha256,
                    "terminal_passed": evaluation.passed,
                }
                for role, stage_kind, evaluation in zip(
                    roles,
                    stage_kinds,
                    evaluations,
                    strict=True,
                )
            ],
            "terminal_state": (
                f"terminal_{roles[-1].split('_', 1)[0]}_"
                f"{'passed' if evaluations[-1].passed else 'failed'}"
            ),
            "reserve_open_count": 0,
        },
    )
    predecessor_A_runtime = _predecessor_A_runtime_receipt(
        preflight=preflight_state,
        predecessor_preflight_sha256=predecessor_preflight_sha,
        predecessor_runtime_qualification_sha256=(
            predecessor_runtime_qualification_sha
        ),
        materialization=materialization,
        source_model_sha256=selection.source_model_sha256,
    )
    predecessor_A_comparison = _predecessor_A_comparison_receipt(
        preflight=preflight_state,
        materialization=materialization,
        predecessor_runtime_session_sha256=predecessor_runtime_session_sha,
    )
    runtime = _worker_receipt(
        b"fisher_graph:layer-multimode:runtime-execution:v1\0",
        {
            "schema": "fisher_graph.gemma3_layer_multimode.runtime_execution.v1",
            "format_version": 1,
            "terminal_status": terminal,
            **session_payload,
            "runtime_qualification_receipt": runtime_qualification,
            "predecessor_A_capture_receipt_sha256": preflight_state[
                "predecessor_A_capture_receipt_sha256"
            ],
            "predecessor_runtime_execution_receipt_sha256": preflight_state[
                "predecessor_runtime_execution_receipt_sha256"
            ],
            "predecessor_A_runtime_receipt": predecessor_A_runtime,
            "predecessor_A_runtime_receipt_sha256": predecessor_A_runtime[
                "receipt_sha256"
            ],
            "predecessor_A_lineage_comparison_receipt": (
                predecessor_A_comparison
            ),
            "predecessor_A_lineage_comparison_receipt_sha256": (
                predecessor_A_comparison["artifact_sha256"]
            ),
            "predecessor_A_recaptured_capture_receipt_sha256": (
                materialization.source_role_capture_receipt_sha256
            ),
            "predecessor_A_current_runtime_session_binding_sha256": (
                materialization.source_runtime_session_binding_sha256
            ),
            "source_model_mutated": False,
            "source_execution_mutated": False,
            "runtime_session_binding_sha256": runtime_session_sha,
            "fresh_role_capture_receipts": fresh_role_captures,
            "predecessor_A_batch_count": 45,
            "predecessor_A_exact_control_forward_count": 45,
            "predecessor_A_exact_control_vjp_count": 45,
            "predecessor_A_dense_parity_forward_count": 1,
            "predecessor_A_dense_parity_vjp_count": 1,
            "B3_batch_count": 8,
            "B3_native_forward_count": 8,
            "B3_candidate_layer_velocity_forward_count": 144,
            "C3_batch_count": 8 if opened >= 2 else 0,
            "C3_native_forward_count": 8 if opened >= 2 else 0,
            "C3_compiled_forward_count": 8 if opened >= 2 else 0,
            "D3_batch_count": 8 if opened >= 3 else 0,
            "D3_native_forward_count": 8 if opened >= 3 else 0,
            "D3_compiled_forward_count": 8 if opened >= 3 else 0,
            "fresh_tokenization_authentication_count": opened,
            "fresh_example_count": 48 * opened,
            "fresh_next_token_position_count": 24_732 * opened,
            "native_baseline_evaluator_count": 1,
            "native_baseline_eval_mode": True,
            "native_baseline_parameter_mutation_count": 0,
            "native_baseline_optimizer_reference_count": 0,
            "compiled_candidate_native_mlp_module_count": 0,
            "reserve_open_count": 0,
        },
    )
    cleanup = _worker_receipt(
        b"fisher_graph:layer-multimode:post-terminal-cleanup:v1\0",
        {
            "schema": (
                "fisher_graph.gemma3_layer_multimode.post_terminal_cleanup.v1"
            ),
            "format_version": 1,
            "terminal_status": terminal,
            "runtime_execution_receipt_sha256": runtime["receipt_sha256"],
            "runtime_session_binding_sha256": runtime[
                "runtime_session_binding_sha256"
            ],
            "fresh_panel_execution_receipt_sha256": fresh_panel[
                "receipt_sha256"
            ],
            "runtime_closed": True,
            "native_baseline_released": True,
            "native_baseline_instance_count_after_close": 0,
            "compiled_candidate_released": True,
            "model_reference_count_after_close": 0,
            "tokenizer_reference_count_after_close": 0,
            "temporary_A_spools_removed": True,
            "temporary_materializations_removed": True,
            "temporary_fit_state_removed": True,
            "frozen_bundle_retained_for_publication": True,
            "reserve_open_count": 0,
        },
    )
    return {
        "terminal_status": terminal,
        "preflight_receipt": preflight_state,
        "A_materialization_receipt": materialization.state_dict(),
        "selection_receipt": selection.state_dict(),
        "compilation_receipt": compilation.state_dict(),
        "bundle_receipt": dict(bundle_receipt),
        "development_teardown_receipt": teardown,
        "experiment_counters_receipt": counters,
        "fresh_panel_execution_receipt": fresh_panel,
        "runtime_execution_receipt": runtime,
        "post_terminal_cleanup_receipt": cleanup,
        "B3_evaluation": B3.state_dict(),
        "C3_evaluation": None if C3 is None else C3.state_dict(),
        "D3_evaluation": None if D3 is None else D3.state_dict(),
        "reserve_disclosure": {
            "role": "reserve_v3_unopened",
            "semantically_opened": False,
            "capture_receipt_sha256": None,
            "evaluation_receipt_sha256": None,
        },
    }


@pytest.fixture(scope="module")
def sealed_report(
    base_evidence: tuple[object, object, object, bytes, dict[str, object]],
) -> tuple[dict[str, object], bytes]:
    payload = _terminal_payload(
        base_evidence,
        report_io.TERMINAL_B3_FAILED,
    )
    return (
        report_io.seal_layer_multimode_experiment_report(payload),
        base_evidence[3],
    )


def test_seal_strict_replay_and_claim_boundary(
    sealed_report: tuple[dict[str, object], bytes],
) -> None:
    report, _ = sealed_report
    raw = report_io.canonical_json_bytes(report)
    assert report_io.strict_load_layer_multimode_experiment_report_bytes(raw) == report
    assert report["claims"]["scientific_result_is_negative"] is True
    assert report["claims"]["latency_improvement_claimed"] is False
    assert report["privacy"]["contains_activation_rows"] is False
    summary = json.loads(report_io.render_layer_multimode_summary_bytes(report))
    assert summary["report_sha256"] == report["report_sha256"]
    assert summary["summary_sha256"] == report["summary_sha256"]


@pytest.mark.parametrize(
    ("terminal", "d3_failing", "expected_negative"),
    (
        (report_io.TERMINAL_B3_FAILED, False, True),
        (report_io.TERMINAL_C3_FAILED, False, True),
        (report_io.TERMINAL_D3_COMPLETED, False, False),
        (report_io.TERMINAL_D3_COMPLETED, True, True),
    ),
)
def test_all_terminal_receipt_shapes_strict_replay(
    base_evidence: tuple[object, object, object, bytes, dict[str, object]],
    terminal: str,
    d3_failing: bool,
    expected_negative: bool,
) -> None:
    report = report_io.seal_layer_multimode_experiment_report(
        _terminal_payload(
            base_evidence,
            terminal,
            d3_failing=d3_failing,
        )
    )
    restored = report_io.strict_load_layer_multimode_experiment_report_bytes(
        report_io.canonical_json_bytes(report)
    )
    assert restored["terminal_status"] == terminal
    assert restored["claims"]["scientific_result_is_negative"] is expected_negative
    if terminal == report_io.TERMINAL_D3_COMPLETED and not d3_failing:
        assert restored["claims"]["exact_parameter_reduction_claimed"] is True
        assert restored["claims"]["MAC_reduction_claimed"] is False
        assert restored["claims"]["generic_operation_reduction_claimed"] is False
        assert restored["claims"]["matrix_MAC_formula_accounting_reported"] is True
        assert restored["claims"]["latency_improvement_claimed"] is False


def test_strict_replay_rejects_duplicate_raw_and_provenance_tamper(
    sealed_report: tuple[dict[str, object], bytes],
) -> None:
    report, _ = sealed_report
    raw = report_io.canonical_json_bytes(report)
    with pytest.raises(ValueError, match="invalid"):
        report_io.strict_load_layer_multimode_experiment_report_bytes(
            raw[:-1] + b',"schema":"duplicate"}'
        )
    changed = json.loads(raw)
    changed["B3_evaluation"]["source_model_sha256"] = _sha("foreign")
    changed["B3_evaluation"]["artifact_sha256"] = ""
    changed["report_sha256"] = ""
    with pytest.raises(ValueError):
        report_io.seal_layer_multimode_experiment_report(
            {key: changed[key] for key in report_io._REPORT_PAYLOAD_FIELDS}
        )


def test_strict_replay_rejects_nonfinite_bool_int_and_phase_shape_attacks(
    sealed_report: tuple[dict[str, object], bytes],
    base_evidence: tuple[object, object, object, bytes, dict[str, object]],
) -> None:
    report, _ = sealed_report
    nonfinite = json.loads(report_io.canonical_json_bytes(report))
    nonfinite["B3_evaluation"]["layer_evidence"][0][
        "family_aggregates"
    ][0]["prediction_sse"] = float("nan")
    raw = json.dumps(
        nonfinite,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode()
    with pytest.raises(ValueError, match="invalid"):
        report_io.strict_load_layer_multimode_experiment_report_bytes(raw)

    payload = _terminal_payload(base_evidence, report_io.TERMINAL_B3_FAILED)
    preflight = dict(payload["preflight_receipt"])
    preflight["format_version"] = True
    preflight.pop("receipt_sha256")
    preflight["receipt_sha256"] = report_io._digest(
        b"fisher_graph:layer-multimode:preflight:v1\0",
        preflight,
    )
    payload["preflight_receipt"] = preflight
    with pytest.raises(ValueError, match="header"):
        report_io.seal_layer_multimode_experiment_report(payload)

    wrong_phase = _terminal_payload(
        base_evidence,
        report_io.TERMINAL_C3_FAILED,
    )
    wrong_phase["C3_evaluation"] = None
    with pytest.raises(ValueError, match="sequence"):
        report_io.seal_layer_multimode_experiment_report(wrong_phase)


@pytest.mark.parametrize(
    ("receipt_name", "field", "domain"),
    (
        (
            "development_teardown_receipt",
            "source_model_sha256",
            b"fisher_graph:layer-multimode:teardown:v1\0",
        ),
        (
            "experiment_counters_receipt",
            "selection_receipt_sha256",
            b"fisher_graph:layer-multimode:counters:v1\0",
        ),
        (
            "fresh_panel_execution_receipt",
            "selection_receipt_sha256",
            b"fisher_graph:layer-multimode:fresh-panel-execution:v1\0",
        ),
        (
            "runtime_execution_receipt",
            "predecessor_A_capture_receipt_sha256",
            b"fisher_graph:layer-multimode:runtime-execution:v1\0",
        ),
        (
            "post_terminal_cleanup_receipt",
            "fresh_panel_execution_receipt_sha256",
            b"fisher_graph:layer-multimode:post-terminal-cleanup:v1\0",
        ),
    ),
)
def test_each_execution_receipt_is_crossbound_to_the_run(
    base_evidence: tuple[object, object, object, bytes, dict[str, object]],
    receipt_name: str,
    field: str,
    domain: bytes,
) -> None:
    payload = _terminal_payload(base_evidence, report_io.TERMINAL_B3_FAILED)
    receipt = dict(payload[receipt_name])
    receipt[field] = _sha(f"foreign-{receipt_name}-{field}")
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = report_io._digest(domain, receipt)
    payload[receipt_name] = receipt
    with pytest.raises(ValueError):
        report_io.seal_layer_multimode_experiment_report(payload)


def test_runtime_qualification_is_crossbound_to_v3_preflight(
    base_evidence: tuple[object, object, object, bytes, dict[str, object]],
) -> None:
    payload = _terminal_payload(base_evidence, report_io.TERMINAL_B3_FAILED)
    runtime = dict(payload["runtime_execution_receipt"])
    nested = dict(runtime["runtime_qualification_receipt"])
    nested["execution_preflight_receipt_sha256"] = _sha("foreign-preflight")
    nested.pop("receipt_sha256")
    nested["receipt_sha256"] = report_io._digest(
        b"fisher_graph:layer-multimode:runtime-qualification:v1\0",
        nested,
    )
    runtime["runtime_qualification_receipt"] = nested
    runtime["runtime_qualification_receipt_sha256"] = nested["receipt_sha256"]
    session_fields = (
        "runtime_implementation_sha256",
        "implementation_qualification_sha256",
        "execution_environment_sha256",
        "tokenizer_fingerprint_sha256",
        "runtime_qualification_receipt_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
        "compiled_model_fingerprint_before_sha256",
        "compiled_model_fingerprint_after_sha256",
        "compiled_execution_fingerprint_before_sha256",
        "compiled_execution_fingerprint_after_sha256",
        "compiled_candidate_mutated",
    )
    runtime["runtime_session_binding_sha256"] = report_io._digest(
        b"fisher_graph:layer-multimode:runtime-session:v1\0",
        {name: runtime[name] for name in session_fields},
    )
    runtime.pop("receipt_sha256")
    runtime["receipt_sha256"] = report_io._digest(
        b"fisher_graph:layer-multimode:runtime-execution:v1\0",
        runtime,
    )
    payload["runtime_execution_receipt"] = runtime
    with pytest.raises(
        ValueError,
        match="qualification differs|fresh-role capture common/runtime lineage",
    ):
        report_io.seal_layer_multimode_experiment_report(payload)


def test_publication_is_report_last_no_clobber_and_capability_rollback(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    links: list[str] = []
    real_link = report_io.os.link

    def observed_link(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        links.append(Path(destination).name)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(report_io.os, "link", observed_link)
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        report_path=report_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
        verify_live_authorization=False,
    )
    assert links[-1] == report_io.DEFAULT_REPORT_OUTPUT.name
    assert report_io.strict_load_layer_multimode_experiment_report(
        report_path,
        verify_live_authorization=False,
    ) == report
    with pytest.raises(FileExistsError):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            report_path=report_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
            verify_live_authorization=False,
        )
    forged = replace(publication)
    with pytest.raises(PermissionError):
        report_io.rollback_published_layer_multimode_experiment(forged)
    report_io.rollback_published_layer_multimode_experiment(publication)
    assert not any(path.exists() for path in (report_path, bundle_path, summary_path))


@pytest.mark.parametrize("failed_link", (1, 2, 3))
def test_every_publication_link_failure_rolls_back_all_outputs(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_link: int,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_link = report_io.os.link
    calls = 0

    def injected_link(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_link:
            raise OSError(f"injected link {failed_link}")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(report_io.os, "link", injected_link)
    with pytest.raises(OSError, match="injected link"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert not any(path.exists() for path in destinations)
    assert list(output.glob(".*.tmp")) == []


def test_publication_link_race_does_not_delete_unowned_target(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    sentinel = b"concurrent unowned publication target"

    def race_link(
        _source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        del args
        descriptor = report_io.os.open(
            target,
            report_io.os.O_WRONLY | report_io.os.O_CREAT | report_io.os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        try:
            report_io.os.write(descriptor, sentinel)
        finally:
            report_io.os.close(descriptor)
        raise FileExistsError(target)

    monkeypatch.setattr(report_io.os, "link", race_link)
    with pytest.raises(FileExistsError):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=bundle_path,
            summary_path=summary_path,
            report_path=report_path,
            verify_live_authorization=False,
        )
    assert bundle_path.read_bytes() == sentinel
    assert not summary_path.exists()
    assert not report_path.exists()
    assert list(output.glob(".*.tmp")) == []


def test_publication_post_link_fault_rolls_back_owned_side_effect(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_link = report_io.os.link

    def link_then_fail(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        real_link(source, destination, *args, **kwargs)
        raise OSError("injected post-link fault")

    monkeypatch.setattr(report_io.os, "link", link_then_fail)
    with pytest.raises(OSError, match="post-link fault"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert not any(path.exists() for path in destinations)
    assert list(output.glob(".*.tmp")) == []


def test_post_link_stat_fault_rolls_back_registered_target(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_link = report_io.os.link
    real_stat_at = report_io._stat_at
    fault_name: str | None = None

    def arm_after_link(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal fault_name
        real_link(source, destination, *args, **kwargs)
        fault_name = str(destination)

    def fail_post_link_stat(descriptor: int, name: str) -> object:
        nonlocal fault_name
        if name == fault_name:
            fault_name = None
            raise OSError("injected post-link stat fault")
        return real_stat_at(descriptor, name)

    monkeypatch.setattr(report_io.os, "link", arm_after_link)
    monkeypatch.setattr(report_io, "_stat_at", fail_post_link_stat)
    with pytest.raises(OSError, match="post-link stat fault"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert not any(path.exists() for path in destinations)
    assert list(output.glob(".*.tmp")) == []


def test_temporary_replacement_is_preserved_and_owned_target_rolls_back(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    replacement = b"unowned replacement temporary"
    real_link = report_io.os.link
    swapped = False

    def replace_source_after_link(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        real_link(source, destination, *args, **kwargs)
        if swapped:
            return
        swapped = True
        source_dir_fd = kwargs["src_dir_fd"]
        report_io.os.unlink(source, dir_fd=source_dir_fd)
        descriptor = report_io.os.open(
            source,
            report_io.os.O_WRONLY
            | report_io.os.O_CREAT
            | report_io.os.O_EXCL,
            0o600,
            dir_fd=source_dir_fd,
        )
        try:
            report_io.os.write(descriptor, replacement)
        finally:
            report_io.os.close(descriptor)

    monkeypatch.setattr(report_io.os, "link", replace_source_after_link)
    with pytest.raises(RuntimeError, match="identity drifted"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert not any(path.exists() for path in destinations)
    preserved = list(output.glob(".*.tmp"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == replacement
    preserved[0].unlink()


def test_fstat_fault_after_temporary_create_uses_descriptor_fallback(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_open = report_io.os.open
    real_fstat = report_io.os.fstat
    fail_descriptor: int | None = None

    def observe_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal fail_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & report_io.os.O_CREAT and dir_fd is not None:
            fail_descriptor = descriptor
        return descriptor

    def fail_first_created_fstat(descriptor: int) -> object:
        nonlocal fail_descriptor
        if descriptor == fail_descriptor:
            fail_descriptor = None
            raise OSError("injected temporary fstat fault")
        return real_fstat(descriptor)

    monkeypatch.setattr(report_io.os, "open", observe_open)
    monkeypatch.setattr(report_io.os, "fstat", fail_first_created_fstat)
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        bundle_path=destinations[0],
        summary_path=destinations[1],
        report_path=destinations[2],
        verify_live_authorization=False,
    )
    assert list(output.glob(".*.tmp")) == []
    report_io.rollback_published_layer_multimode_experiment(publication)


@pytest.mark.parametrize("failed_fsync", (1, 2))
def test_every_publication_directory_fsync_failure_rolls_back_all_outputs(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_fsync: int,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_fsync = report_io.os.fsync
    calls = 0

    def injected_fsync(descriptor: int) -> None:
        nonlocal calls
        if report_io.stat.S_ISDIR(report_io.os.fstat(descriptor).st_mode):
            calls += 1
            if calls == failed_fsync:
                raise OSError(f"injected fsync {failed_fsync}")
        real_fsync(descriptor)

    monkeypatch.setattr(report_io.os, "fsync", injected_fsync)
    with pytest.raises(OSError, match="injected fsync"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert not any(path.exists() for path in destinations)


def test_publication_preserves_primary_when_temporary_cleanup_also_fails(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_unlink = report_io.os.unlink

    def fail_temporary_unlink(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(path).name.endswith(".tmp"):
            raise OSError("persistent temporary unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(report_io.os, "unlink", fail_temporary_unlink)
    with pytest.raises(OSError, match="persistent temporary unlink failure") as caught:
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    assert not any(path.exists() for path in destinations)
    for temporary in output.glob(".*.tmp"):
        real_unlink(temporary)


def test_publication_preserves_link_failure_through_persistent_fsync_faults(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )

    def fail_link(
        _source: object,
        _destination: object,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise OSError("primary publication link failure")

    real_fsync = report_io.os.fsync

    def fail_fsync(descriptor: int) -> None:
        if report_io.stat.S_ISDIR(report_io.os.fstat(descriptor).st_mode):
            raise OSError("persistent temporary-parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(report_io.os, "link", fail_link)
    monkeypatch.setattr(report_io.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="primary publication link failure") as caught:
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert not any(path.exists() for path in destinations)
    assert list(output.glob(".*.tmp")) == []


def test_publication_rejects_symlink_in_parent_ancestry(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
) -> None:
    report, bundle = sealed_report
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real_parent, target_is_directory=True)
    output = redirected / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    with pytest.raises(ValueError, match="ancestry"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
            summary_path=output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
            report_path=output / report_io.DEFAULT_REPORT_OUTPUT.name,
            verify_live_authorization=False,
        )
    assert list(output.iterdir()) == []


def test_publication_ancestor_swap_rolls_back_through_pinned_dirfd(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    moved = tmp_path / "moved-original-output"
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    destinations = (
        output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
        output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
        output / report_io.DEFAULT_REPORT_OUTPUT.name,
    )
    real_link = report_io.os.link
    swapped = False

    def swap_parent_before_link(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            output.rename(moved)
            output.mkdir()
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(report_io.os, "link", swap_parent_before_link)
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            bundle_path=destinations[0],
            summary_path=destinations[1],
            report_path=destinations[2],
            verify_live_authorization=False,
        )
    assert list(output.iterdir()) == []
    assert list(moved.iterdir()) == []


def test_strict_loader_rejects_symlinked_output_ancestor(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        bundle_path=bundle_path,
        summary_path=summary_path,
        report_path=report_path,
        verify_live_authorization=False,
    )
    moved = tmp_path / "moved-outputs"
    output.rename(moved)
    output.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestry"):
        report_io.strict_load_layer_multimode_experiment_report(
            report_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
            verify_live_authorization=False,
        )
    output.unlink()
    moved.rename(output)
    report_io.rollback_published_layer_multimode_experiment(publication)


def test_post_commit_reload_failure_removes_report_before_sidecars(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    unlinks: list[str] = []
    real_unlink = report_io.os.unlink

    def observed_unlink(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        name = Path(path).name
        if name in {path.name for path in (report_path, bundle_path, summary_path)}:
            unlinks.append(name)
        real_unlink(path, *args, **kwargs)

    def fail_reload(*args: object, **kwargs: object) -> object:
        raise ValueError("injected strict reload")

    monkeypatch.setattr(report_io.os, "unlink", observed_unlink)
    monkeypatch.setattr(
        report_io,
        "strict_load_layer_multimode_experiment_report",
        fail_reload,
    )
    with pytest.raises(ValueError, match="injected strict reload"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            report_path=report_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
            verify_live_authorization=False,
        )
    assert unlinks == [report_path.name, summary_path.name, bundle_path.name]
    assert not any(path.exists() for path in (report_path, bundle_path, summary_path))


def test_strict_reload_target_replacement_is_preserved_while_owned_peers_rollback(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    real_open = report_io.os.open
    real_unlink = report_io.os.unlink
    replacement = b"unowned summary replacement"
    replaced = False

    def replace_summary_when_reload_opens(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == summary_path.name and not replaced:
            replaced = True
            parent_descriptor = kwargs["dir_fd"]
            real_unlink(path, dir_fd=parent_descriptor)
            replacement_descriptor = real_open(
                path,
                report_io.os.O_WRONLY
                | report_io.os.O_CREAT
                | report_io.os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                report_io.os.write(replacement_descriptor, replacement)
            finally:
                report_io.os.close(replacement_descriptor)
        return descriptor

    monkeypatch.setattr(report_io.os, "open", replace_summary_when_reload_opens)
    with pytest.raises(ValueError, match="published layer-multimode summary"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            report_path=report_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
            verify_live_authorization=False,
        )
    assert not report_path.exists()
    assert not bundle_path.exists()
    assert summary_path.read_bytes() == replacement
    real_unlink(summary_path)


def test_external_rollback_is_best_effort_after_unlink_side_effect_fault(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        report_path=report_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
        verify_live_authorization=False,
    )
    real_unlink = report_io.os.unlink
    injected = False

    def unlink_then_fail(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        real_unlink(path, *args, **kwargs)
        if Path(path).name == report_path.name and not injected:
            injected = True
            raise OSError("injected rollback unlink side-effect fault")

    monkeypatch.setattr(report_io.os, "unlink", unlink_then_fail)
    with pytest.raises(OSError, match="unlink side-effect fault"):
        report_io.rollback_published_layer_multimode_experiment(publication)
    assert not any(path.exists() for path in (report_path, summary_path, bundle_path))
    monkeypatch.setattr(report_io.os, "unlink", real_unlink)
    report_io.rollback_published_layer_multimode_experiment(publication)
    with pytest.raises(PermissionError):
        report_io.rollback_published_layer_multimode_experiment(publication)


def test_wrong_paths_and_bundle_bytes_fail_before_publication(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle + b"tamper")
    with pytest.raises(ValueError, match="staged bundle"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            report_path=output / report_io.DEFAULT_REPORT_OUTPUT.name,
            bundle_path=output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
            summary_path=output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
            verify_live_authorization=False,
        )
    with pytest.raises(ValueError, match="frozen contract"):
        report_io.publish_layer_multimode_experiment_outputs(
            report=report,
            bundle_source_path=staged,
            report_path=output / "wrong-report.json",
            bundle_path=output / report_io.DEFAULT_BUNDLE_OUTPUT.name,
            summary_path=output / report_io.DEFAULT_SUMMARY_OUTPUT.name,
            verify_live_authorization=False,
        )
    assert list(output.iterdir()) == []


def test_finalize_and_target_authentication_are_exact_object_capabilities(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        report_path=report_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
        verify_live_authorization=False,
    )
    with pytest.raises(PermissionError):
        report_io.finalize_published_layer_multimode_experiment(
            replace(publication)
        )
    report_io.finalize_published_layer_multimode_experiment(publication)
    with pytest.raises(PermissionError):
        report_io.rollback_published_layer_multimode_experiment(publication)
    assert all(path.exists() for path in (report_path, bundle_path, summary_path))


def test_rollback_authenticates_every_target_before_deleting_any(
    sealed_report: tuple[dict[str, object], bytes],
    tmp_path: Path,
) -> None:
    report, bundle = sealed_report
    output = tmp_path / "outputs"
    output.mkdir()
    staged = tmp_path / "staged.pt"
    staged.write_bytes(bundle)
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        report_path=report_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
        verify_live_authorization=False,
    )
    summary_path.unlink()
    summary_path.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="identity drifted"):
        report_io.rollback_published_layer_multimode_experiment(publication)
    assert all(path.exists() for path in (report_path, bundle_path, summary_path))


def test_live_setup_qualification_and_panel_rehash_are_all_enforced(
    base_evidence: tuple[object, object, object, bytes, dict[str, object]],
    tmp_path: Path,
) -> None:
    root, panel_bundle = _live_authorization_root(tmp_path)
    setup_provenance = multimode_setup.build_setup()["methodology_provenance"]
    selection = _selection(
        setup_provenance["predecessor_v2_source_model_fingerprint_sha256"]
    )
    materialization = _materialization(
        selection.source_model_sha256,
        source_execution_fingerprint_sha256=setup_provenance[
            "predecessor_v2_source_execution_fingerprint_sha256"
        ],
    )
    compilation = _compilation(selection, materialization)
    live_bundle_receipt = {
        "bundle_sha256": _sha(base_evidence[3]),
        "byte_count": len(base_evidence[3]),
        "manifest_sha256": _sha("live-bundle-manifest"),
        "compilation_receipt_sha256": compilation.receipt_sha256,
    }
    live_evidence = (
        selection,
        materialization,
        compilation,
        base_evidence[3],
        live_bundle_receipt,
    )
    setup_path = root / multimode_setup.DEFAULT_OUTPUT
    panel_path = root / panel_module.DEFAULT_RECEIPT
    qualification_path = root / qualified.DEFAULT_OUTPUT
    setup_raw = setup_path.read_bytes()
    panel_raw = panel_path.read_bytes()
    qualification_raw = qualification_path.read_bytes()
    qualification_state = qualified.load_qualification(
        qualification_path,
        repository_root=root,
    )
    panel_manifest = json.loads(panel_bundle.manifest_bytes)
    panel_receipt = json.loads(panel_raw)
    predecessor_report = {
        "report_sha256": setup_provenance["predecessor_v2_report_sha256"],
        "A_fit": {
            "capture_receipt": {
                "receipt_sha256": setup_provenance[
                    "predecessor_v2_A_capture_receipt_sha256"
                ]
            }
        },
        "v2_preflight": {
            "receipt_sha256": setup_provenance[
                "predecessor_v2_preflight_receipt_sha256"
            ]
        },
        "runtime_qualification_receipt": {
            "qualification_receipt_sha256": setup_provenance[
                "predecessor_v2_runtime_qualification_receipt_sha256"
            ]
        },
        "runtime_execution_receipt": {
            "runtime_receipt_sha256": setup_provenance[
                "predecessor_v2_runtime_execution_receipt_sha256"
            ],
            "runtime_session_binding_sha256": setup_provenance[
                "predecessor_v2_runtime_session_binding_sha256"
            ],
        },
    }
    panel_sha = panel_manifest["manifest_sha256"]
    preflight = _preflight(
        selection.source_model_sha256,
        setup_sha256=multimode_setup.build_setup()["setup_sha256"],
        setup_file_sha256=_sha(setup_raw),
        panel_manifest_sha256=panel_sha,
        panel_manifest_file_sha256=_sha(panel_bundle.manifest_bytes),
        panel_receipt_sha256=panel_receipt["receipt_sha256"],
        panel_receipt_file_sha256=_sha(panel_raw),
        panel_bundle_sha256=panel_receipt["bundle_sha256"],
        qualification_sha256=qualification_state["qualification_sha256"],
        qualification_file_sha256=_sha(qualification_raw),
        predecessor_v2_setup_sha256=(
            multimode_setup.PREDECESSOR_V2_SETUP_SHA256
        ),
        predecessor_v2_report_sha256=predecessor_report["report_sha256"],
        predecessor_A_capture_receipt_sha256=(
            predecessor_report["A_fit"]["capture_receipt"]["receipt_sha256"]
        ),
        predecessor_runtime_execution_receipt_sha256=(
            predecessor_report["runtime_execution_receipt"][
                "runtime_receipt_sha256"
            ]
        ),
    )
    wrong_capture_axes = report_io.seal_layer_multimode_experiment_report(
        _terminal_payload(
            live_evidence,
            report_io.TERMINAL_B3_FAILED,
            preflight=preflight,
            panel_manifest_sha256=panel_sha,
            panel_manifest=None,
            predecessor_report=predecessor_report,
            runtime_implementation_sha256=_sha(
                (
                    root
                    / "src/fisher_graph/gemma3_layer_multimode_runtime.py"
                ).read_bytes()
            ),
        )
    )
    with pytest.raises(ValueError, match="live frozen panel axes"):
        report_io._strict_live_rehash(
            wrong_capture_axes,
            repository_root=root,
        )
    wrong_example_axes = report_io.seal_layer_multimode_experiment_report(
        _terminal_payload(
            live_evidence,
            report_io.TERMINAL_C3_FAILED,
            preflight=preflight,
            panel_manifest_sha256=panel_sha,
            panel_manifest=panel_manifest,
            predecessor_report=predecessor_report,
            runtime_implementation_sha256=_sha(
                (
                    root
                    / "src/fisher_graph/gemma3_layer_multimode_runtime.py"
                ).read_bytes()
            ),
        )
    )
    with pytest.raises(ValueError, match="live frozen example axis"):
        report_io._strict_live_rehash(
            wrong_example_axes,
            repository_root=root,
        )
    report = report_io.seal_layer_multimode_experiment_report(
        _terminal_payload(
            live_evidence,
            report_io.TERMINAL_B3_FAILED,
            preflight=preflight,
            panel_manifest_sha256=panel_sha,
            panel_manifest=panel_manifest,
            predecessor_report=predecessor_report,
            runtime_implementation_sha256=_sha(
                (
                    root
                    / "src/fisher_graph/gemma3_layer_multimode_runtime.py"
                ).read_bytes()
            ),
        )
    )
    output = root / ".local-runs/google--gemma-3-270m"
    output.mkdir(parents=True, exist_ok=True)
    staged = root / "staged.pt"
    staged.write_bytes(base_evidence[3])
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    bundle_path = output / report_io.DEFAULT_BUNDLE_OUTPUT.name
    summary_path = output / report_io.DEFAULT_SUMMARY_OUTPUT.name
    publication = report_io.publish_layer_multimode_experiment_outputs(
        report=report,
        bundle_source_path=staged,
        report_path=report_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
        repository_root=root,
        verify_live_authorization=True,
    )
    assert report_io.strict_load_layer_multimode_experiment_report(
        report_path,
        repository_root=root,
    ) == report

    for path, raw, label in (
        (setup_path, setup_raw, "setup"),
        (qualification_path, qualification_raw, "qualification"),
        (panel_path, panel_raw, "panel"),
        (
            root / multimode_setup.PREDECESSOR_V2_SETUP,
            (root / multimode_setup.PREDECESSOR_V2_SETUP).read_bytes(),
            "predecessor",
        ),
    ):
        path.write_bytes(raw + b"\n")
        with pytest.raises(ValueError, match=label):
            report_io.strict_load_layer_multimode_experiment_report(
                report_path,
                repository_root=root,
            )
        path.write_bytes(raw)
        assert report_io.strict_load_layer_multimode_experiment_report(
            report_path,
            repository_root=root,
        ) == report
    predecessor_report_path = root / multimode_setup.PREDECESSOR_V2_REPORT
    predecessor_report_size = predecessor_report_path.stat().st_size
    with predecessor_report_path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="predecessor"):
        report_io.strict_load_layer_multimode_experiment_report(
            report_path,
            repository_root=root,
        )
    with predecessor_report_path.open("r+b") as stream:
        stream.truncate(predecessor_report_size)
    assert report_io.strict_load_layer_multimode_experiment_report(
        report_path,
        repository_root=root,
    ) == report
    report_io.rollback_published_layer_multimode_experiment(publication)
