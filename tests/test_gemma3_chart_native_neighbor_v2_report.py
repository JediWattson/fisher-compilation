from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fisher_graph import gemma3_chart_native_neighbor_v2_report as report_io
from fisher_graph import gemma3_chart_native_neighbor_v2_setup as v2_setup


_SHA0 = "0" * 64
_SHA1 = "1" * 64
_SHA2 = "2" * 64


def _load_v1_completed_harness_module() -> object:
    module_name = "_fisher_graph_v1_completed_report_harness_for_v2"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = (
        Path(__file__).resolve().parent
        / "test_gemma3_chart_native_neighbor_experiment_adversarial.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the model-free v1 report harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_selector_numeric_harness_module() -> object:
    module_name = "_fisher_graph_selector_numeric_harness_for_v2"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = (
        Path(__file__).resolve().parent
        / "test_streaming_shared_chart_atlas_selection_v1.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the selector numeric harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reseal_report(value: dict[str, object]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("report_sha256", None)
    value["report_sha256"] = report_io._digest(
        report_io._REPORT_DOMAIN, unsigned
    )
    return report_io.canonical_json_bytes(value)


def _minimal_top_level_report(*, format_version: object = 2) -> dict[str, object]:
    return {
        "schema": report_io.REPORT_SCHEMA,
        "format_version": format_version,
        "report_sha256": _SHA0,
        "v2_preflight": {},
        "preflight": {},
        "A_fit": {},
        "B_discovery": {},
        "C_confirmation": {},
        "D_response_validation": {},
        "capture_call_ledger": {},
        "claims": {},
        "confirmed_edges": [],
        "final_resource_observation": {},
        "mode_axis": [],
        "neighbor_svg_sha256": _SHA0,
        "phase_cleanup_resource_observations": [],
        "protocol": {},
        "runtime_execution_receipt": {},
        "runtime_qualification_receipt": {},
        "methodology": {},
    }


def _valid_v2_preflight_payload() -> dict[str, object]:
    prior = {
        "result_path": str(report_io.DEFAULT_A_DIAGNOSTIC_RESULT_PATH),
        "result_file_sha256": _SHA0,
        "result_sha256": _SHA0,
        "diagnosis_receipt_sha256": _SHA0,
        "candidate_receipt_sha256": _SHA0,
        "semantic_projection_sha256": _SHA0,
    }
    unsigned = {
        "schema": report_io.V2_PREFLIGHT_SCHEMA,
        "v1_preflight_receipt_sha256": _SHA0,
        "v2_setup_sha256": v2_setup.PINNED_SETUP_SHA256,
        "v2_setup_file_sha256": v2_setup.PINNED_SETUP_FILE_SHA256,
        "implementation_qualification_sha256": _SHA0,
        "implementation_qualification_file_sha256": _SHA0,
        "required_K8_diagnostic": prior,
        "model_runtime_touched_before_v1_static_preflight": False,
        "numeric_selector_touched_before_v1_static_preflight": False,
    }
    return {
        **unsigned,
        "receipt_sha256": report_io._digest(
            report_io._PREFLIGHT_DOMAIN, unsigned
        ),
    }


def _minimal_A(*, chart_count: object = 4, evaluations: object = 2) -> dict[str, object]:
    source_feature_width: object = (
        17 * chart_count if type(chart_count) is int else 17
    )
    return {
        "atlas_artifact_file_sha256": _SHA0,
        "atlas_sha256": _SHA0,
        "capture_receipt": {},
        "mode_lineage_catalog_sha256": _SHA0,
        "published_atlas_file_name": report_io.DEFAULT_ATLAS_OUTPUT.name,
        "resource_observations": [],
        "stage_receipt": {},
        "streaming_kernel_receipt_sha256": _SHA0,
        "transport_artifacts": [],
        "transport_catalog_sha256": _SHA0,
        "selection_receipt": {},
        "preselection_K8_diagnosis": {},
        "K8_agreement_receipt": {},
        "selected_resource_receipt": {},
        "selected_chart_count": chart_count,
        "selected_source_feature_width": source_feature_width,
        "selected_tangent_rank": 16,
        "claim_tier": "selected",
        "K8_numeric_evaluation_count": evaluations,
    }


def _publication_receipt(
    directory: Path,
    *,
    report_raw: bytes,
    svg_raw: bytes,
    atlas_raw: bytes,
) -> report_io.PublishedV2ExperimentReceipt:
    payload = {
        "atlas_path": str(directory / report_io.DEFAULT_ATLAS_OUTPUT.name),
        "report_path": str(directory / report_io.DEFAULT_REPORT_OUTPUT.name),
        "svg_path": str(directory / report_io.DEFAULT_SVG_OUTPUT.name),
        "atlas_file_sha256": hashlib.sha256(atlas_raw).hexdigest(),
        "atlas_sha256": _SHA0,
        "report_file_sha256": hashlib.sha256(report_raw).hexdigest(),
        "svg_file_sha256": hashlib.sha256(svg_raw).hexdigest(),
        "report_sha256": _SHA1,
    }
    return report_io.PublishedV2ExperimentReceipt(
        **payload,
        publication_receipt_sha256=report_io._digest(
            report_io._PUBLICATION_DOMAIN,
            {
                "schema": report_io.PUBLICATION_SCHEMA,
                "format_version": report_io.PUBLICATION_FORMAT_VERSION,
                **payload,
            },
        ),
    )


def _reseal_publication(
    publication: report_io.PublishedV2ExperimentReceipt,
    **changes: object,
) -> report_io.PublishedV2ExperimentReceipt:
    values = {
        field: getattr(publication, field)
        for field in publication.__dataclass_fields__
        if field != "publication_receipt_sha256"
    }
    values.update(changes)
    unsigned = {
        "schema": report_io.PUBLICATION_SCHEMA,
        "format_version": report_io.PUBLICATION_FORMAT_VERSION,
        **values,
    }
    return report_io.PublishedV2ExperimentReceipt(
        **values,
        publication_receipt_sha256=report_io._digest(
            report_io._PUBLICATION_DOMAIN, unsigned
        ),
    )


def _write_publication_files(
    publication: report_io.PublishedV2ExperimentReceipt,
    *,
    report_raw: bytes,
    svg_raw: bytes,
    atlas_raw: bytes,
) -> None:
    Path(publication.report_path).write_bytes(report_raw)
    Path(publication.svg_path).write_bytes(svg_raw)
    Path(publication.atlas_path).write_bytes(atlas_raw)
    paths = (
        Path(publication.report_path),
        Path(publication.svg_path),
        Path(publication.atlas_path),
    )
    report_io._register_publication_authority(
        publication,
        target_identities={
            path: report_io._file_identity(path.lstat()) for path in paths
        },
    )


def _patch_publication_science(
    monkeypatch: pytest.MonkeyPatch,
    *,
    atlas_source: Path,
    svg_raw: bytes,
    strict_reload_error: BaseException | None = None,
) -> tuple[dict[str, object], object]:
    atlas_raw = atlas_source.read_bytes()
    atlas_sha = _SHA2
    loaded = {
        "report_sha256": _SHA1,
        "mode_axis": ["mode"],
        "confirmed_edges": [],
        "neighbor_svg_sha256": hashlib.sha256(svg_raw).hexdigest(),
        "A_fit": {
            "published_atlas_file_name": report_io.DEFAULT_ATLAS_OUTPUT.name,
            "atlas_artifact_file_sha256": hashlib.sha256(atlas_raw).hexdigest(),
            "atlas_sha256": atlas_sha,
        },
    }
    atlas = SimpleNamespace(atlas_sha256=atlas_sha)
    from fisher_graph import shared_chart_energy_transport as shared

    monkeypatch.setattr(
        report_io,
        "strict_load_v2_experiment_report_bytes",
        lambda raw: loaded,
    )
    monkeypatch.setattr(report_io.v1, "_validate_svg_bytes", lambda raw: None)
    monkeypatch.setattr(
        report_io.v1,
        "render_neighbor_svg",
        lambda **kwargs: svg_raw,
    )
    monkeypatch.setattr(shared, "load_shared_chart_atlas", lambda path: atlas)
    monkeypatch.setattr(report_io, "_validate_selected_atlas", lambda *a, **k: None)
    if strict_reload_error is None:
        monkeypatch.setattr(
            report_io,
            "strict_load_v2_experiment_report",
            lambda *args, **kwargs: loaded,
        )
    else:
        def fail_reload(*args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            raise strict_reload_error

        monkeypatch.setattr(
            report_io, "strict_load_v2_experiment_report", fail_reload
        )
    return loaded, atlas


def _rehash_capture_with_mode_artifacts(
    capture: object,
    mode_raw_artifacts: tuple[tuple[str, str, str, str], ...],
) -> object:
    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1_worker

    values = {
        field.name: getattr(capture, field.name)
        for field in fields(type(capture))
        if field.name != "receipt_sha256"
    }
    values["mode_raw_artifacts"] = mode_raw_artifacts
    provisional = object.__new__(type(capture))
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", _SHA0)
    return type(capture)(
        **values,
        receipt_sha256=v1_worker._digest(
            v1_worker._CAPTURE_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


def _stage_payload(receipt: object) -> dict[str, object]:
    return {
        **receipt.payload_without_sha256(),
        "stage_receipt_sha256": receipt.stage_receipt_sha256,
    }


def _rebuild_stage_with_prior(
    stage_payload: object,
    *,
    prior_stage_receipt_sha256: str | None,
    artifact_overrides: dict[str, str] | None = None,
) -> object:
    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1_worker

    stage = v1_worker._parse_frozen_stage_payload(stage_payload)
    artifacts = dict(stage.artifact_sha256s)
    artifacts.update(artifact_overrides or {})
    return v1_worker.build_frozen_stage_receipt(
        role=stage.role,
        preflight_receipt_sha256=stage.preflight_receipt_sha256,
        role_panel_binding_sha256_value=stage.role_panel_binding_sha256,
        ordered_objective_row_catalog_sha256=(
            stage.ordered_objective_row_catalog_sha256
        ),
        score_objective_receipt_sha256=stage.score_objective_receipt_sha256,
        prior_stage_receipt_sha256=prior_stage_receipt_sha256,
        artifact_sha256s=artifacts,
        primary_eligible_example_catalog_sha256=(
            stage.primary_eligible_example_catalog_sha256
        ),
    )


def _selected_resource_receipt(chart_count: int) -> dict[str, object]:
    from fisher_graph import gemma3_chart_native_neighbor_v2_experiment as worker
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        estimate_streaming_chart_resources,
    )

    frozen = v2_setup.build_setup()
    intrinsic = frozen["unchanged_intrinsic_atlas_contract"]
    dynamic = frozen["dynamic_transport_contract"]
    resources = dynamic["selection_streaming_resource_config"]
    row_count = intrinsic["fit_pool_rows_per_mode"]
    mode_count = intrinsic["fit_pool_total_rows"] // row_count
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=row_count,
        mode_count=mode_count,
        config=SharedChartNumericsConfig(
            hidden_width=intrinsic["hidden_width"],
            chart_count=chart_count,
            tangent_rank=intrinsic["tangent_rank"],
        ),
        resources=StreamingChartResourceConfig(
            row_chunk_size=resources["row_chunk_size"],
            atlas_chunk_size=resources["atlas_chunk_size"],
            max_resident_bytes=resources["max_resident_bytes"],
            max_artifact_bytes=resources["max_artifact_bytes"],
        ),
    )
    return worker._selected_resource_receipt(estimate=estimate)


def _seal_private_receipt(
    cls: type,
    domain: bytes,
    values: dict[str, object],
    *,
    digest_function: object,
) -> object:
    provisional = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", _SHA0)
    digest = digest_function(domain, provisional._payload_without_hash())
    return cls(**values, receipt_sha256=digest)


def _authentic_production_selection_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    preselection: object,
    selected_atlas_sha256: str,
) -> tuple[object, object]:
    """Lift a real K7 numeric trace onto the sealed production A lineage."""

    import torch

    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as selector
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig

    numeric_harness = _load_selector_numeric_harness_module()
    numeric_root = tmp_path / "authentic-selector"
    numeric_root.mkdir()
    resources = numeric_harness._resources(atlas_chunk_size=43)
    spools = numeric_harness._write_spools(
        numeric_root,
        numeric_harness._well_conditioned_batches(),
        resources,
        prefix="descending",
    )
    real_eigh = torch.linalg.eigh
    calls = 0

    def first_tied(value: object) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        eigenvalues, eigenvectors = real_eigh(value)
        if calls == 1:
            eigenvalues = eigenvalues.clone()
            eigenvalues[-1] = eigenvalues[-2]
        return eigenvalues, eigenvectors

    with monkeypatch.context() as numeric_patch:
        numeric_patch.setattr(torch.linalg, "eigh", first_tied)
        numeric = selector.select_and_fit_shared_chart_atlas_streaming(
            spools,
            config=SharedChartNumericsConfig(hidden_width=20),
            resources=resources,
        )
    assert numeric.receipt.attempted_chart_counts == (8, 7)
    numeric_candidate = numeric.receipt.attempted_diagnoses[-1].candidate
    chart_count = 7
    mode_ids = preselection.fit_mode_ids
    family_ids = preselection.fit_family_ids
    fit_family_counts = preselection.fit_row_count_by_family
    mode_count = preselection.mode_count
    rows_per_mode = preselection.row_count_per_mode
    total_rows = preselection.total_row_count

    # Build positive integer margins with exact production row marginals.
    mode_base, mode_remainder = divmod(rows_per_mode, chart_count)
    assert mode_remainder == 1 and mode_count == 38
    extra_by_mode = (
        (0,) * 6
        + (1,) * 6
        + (2,) * 6
        + (3,) * 5
        + (4,) * 5
        + (5,) * 5
        + (6,) * 5
    )
    count_by_mode = [
        [mode_base for _ in range(mode_count)] for _ in range(chart_count)
    ]
    for mode, chart in enumerate(extra_by_mode):
        count_by_mode[chart][mode] += 1
    hard_counts = tuple(sum(row) for row in count_by_mode)

    family_extra_sets = (
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 2, 5),
        (0, 3, 4, 6),
        (1, 3, 5, 6),
        (2, 4, 5, 6),
    )
    count_by_family = [
        [0 for _ in family_ids] for _ in range(chart_count)
    ]
    for family, family_rows in enumerate(fit_family_counts):
        total_family_rows = mode_count * family_rows
        base, remainder = divmod(total_family_rows, chart_count)
        assert remainder == len(family_extra_sets[family])
        for chart in range(chart_count):
            count_by_family[chart][family] = base + int(
                chart in family_extra_sets[family]
            )
    assert tuple(sum(row) for row in count_by_family) == hard_counts
    assert sum(hard_counts) == total_rows

    hard_mass = 1.0 / chart_count
    hard_mass_bits = (selector._binary64(hard_mass),) * chart_count
    mass_by_mode = tuple(
        tuple(
            selector._binary64(1.0 / (chart_count * mode_count))
            for _ in mode_ids
        )
        for _ in range(chart_count)
    )
    mass_by_family = tuple(
        tuple(
            selector._binary64(1.0 / (chart_count * len(family_ids)))
            for _ in family_ids
        )
        for _ in range(chart_count)
    )

    seed_receipts = []
    for seed in numeric_candidate.seed_receipts:
        mode_ordinal = seed.seed_ordinal
        row_ordinal = seed.seed_ordinal
        mode_id = mode_ids[mode_ordinal]
        family_id = family_ids[seed.seed_ordinal % len(family_ids)]
        global_row = mode_ordinal * rows_per_mode + row_ordinal
        mode_identity = (
            selector.streaming_shared_chart_atlas_mode_identity_sha256_v1(
                mode_id, mode_ordinal
            )
        )
        row_key = [
            family_id,
            f"model-free-authentic-seed-{seed.seed_ordinal}",
            row_ordinal,
            row_ordinal + 1,
        ]
        values = {
            "seed_ordinal": seed.seed_ordinal,
            "selected_global_row_ordinal": global_row,
            "selected_mode_ordinal": mode_ordinal,
            "selected_row_ordinal": row_ordinal,
            "mode_id": mode_id,
            "family_id": family_id,
            "example_id": row_key[1],
            "prediction_query_position": row_ordinal,
            "target_label_position": row_ordinal + 1,
            "mode_identity_sha256": mode_identity,
            "row_identity_sha256": selector._digest(
                selector._ROW_IDENTITY_DOMAIN,
                {
                    "global_row_ordinal": global_row,
                    "mode_identity_sha256": mode_identity,
                    "row_key": row_key,
                    "row_ordinal": row_ordinal,
                },
            ),
            "weighted_farthest_score_binary64": (
                seed.weighted_farthest_score_binary64
            ),
            "selected_center_sha256": seed.selected_center_sha256,
        }
        seed_receipts.append(
            _seal_private_receipt(
                selector.StreamingSharedChartAtlasSeedReceiptV1,
                selector._SEED_RECEIPT_DOMAIN,
                values,
                digest_function=selector._digest,
            )
        )

    pass_receipts = []
    for item in numeric_candidate.lloyd_pass_receipts:
        values = {
            field.name: getattr(item, field.name)
            for field in fields(type(item))
            if field.name != "receipt_sha256"
        }
        values.update(
            {
                "hard_occupancy_count": hard_counts,
                "hard_occupancy_mass_binary64": hard_mass_bits,
            }
        )
        pass_receipts.append(
            _seal_private_receipt(
                selector.StreamingSharedChartAtlasPassReceiptV1,
                selector._PASS_RECEIPT_DOMAIN,
                values,
                digest_function=selector._digest,
            )
        )

    config = replace(preselection.config, chart_count=chart_count)
    config_sha256 = selector._config_sha256(config)
    solver_sha256 = selector._expected_fit_solver_receipt_sha256(
        config=config,
        resources=preselection.resources,
        row_count_per_mode=rows_per_mode,
        mode_count=mode_count,
        fit_context_catalog_sha256=preselection.fit_context_catalog_sha256,
        fit_balanced_row_weight_catalog_sha256=(
            preselection.fit_balanced_row_weight_catalog_sha256
        ),
    )
    count_mode_tuple = tuple(tuple(row) for row in count_by_mode)
    count_family_tuple = tuple(tuple(row) for row in count_by_family)
    composition_sha256 = selector._composition_catalog_sha256(
        chart_count=chart_count,
        mode_count=mode_count,
        family_count=len(family_ids),
        mode_ids=mode_ids,
        family_ids=family_ids,
        fit_row_count_by_family=fit_family_counts,
        count_by_chart_and_mode=count_mode_tuple,
        mass_by_chart_and_mode=mass_by_mode,
        count_by_chart_and_family=count_family_tuple,
        mass_by_chart_and_family=mass_by_family,
    )
    selected_candidate = selector._revise_candidate(
        numeric_candidate,
        total_row_count=total_rows,
        total_balanced_mass_binary64=(
            preselection.total_balanced_mass_binary64
        ),
        mode_count=mode_count,
        family_count=len(family_ids),
        mode_ids=mode_ids,
        family_ids=family_ids,
        fit_row_count_by_family=fit_family_counts,
        config_sha256=config_sha256,
        seed_receipts=tuple(seed_receipts),
        lloyd_pass_receipts=tuple(pass_receipts),
        final_hard_occupancy_count=hard_counts,
        final_hard_occupancy_mass_binary64=hard_mass_bits,
        final_hard_count_by_chart_and_mode=count_mode_tuple,
        final_hard_mass_binary64_by_chart_and_mode=mass_by_mode,
        final_hard_count_by_chart_and_family=count_family_tuple,
        final_hard_mass_binary64_by_chart_and_family=mass_by_family,
        composition_catalog_sha256=composition_sha256,
        minimum_observed_hard_mass_binary64=selector._binary64(hard_mass),
        minimum_hard_mass_margin_binary64=selector._binary64(
            hard_mass
            - selector.STREAMING_SHARED_CHART_ATLAS_MINIMUM_HARD_MASS_V1
        ),
        minimum_hard_mass_chart_ordinals=tuple(range(chart_count)),
        atlas_sha256=selected_atlas_sha256,
        fit_solver_receipt_sha256=solver_sha256,
    )
    diagnosis_values = {
        field.name: getattr(preselection, field.name)
        for field in fields(type(preselection))
        if field.name != "receipt_sha256"
    }
    diagnosis_values.update(
        {
            "config": config,
            "config_sha256": config_sha256,
            "candidate": selected_candidate,
        }
    )
    selected_diagnosis = _seal_private_receipt(
        selector.StreamingSharedChartAtlasOccupancyDiagnosisV1,
        selector._DIAGNOSIS_RECEIPT_DOMAIN,
        diagnosis_values,
        digest_function=selector._digest,
    )
    selection_values = {
        "selection_outcome": selector.A_ATLAS_SELECTED,
        "selection_kernel_receipt_sha256": (
            preselection.selection_kernel_receipt_sha256
        ),
        "candidate_order": selector.STREAMING_SHARED_CHART_ATLAS_CANDIDATE_ORDER_V1,
        "attempted_diagnoses": (preselection, selected_diagnosis),
        "attempted_chart_counts": (8, chart_count),
        "unattempted_chart_counts": (6, 5, 4, 3, 2, 1),
        "selected_chart_count": chart_count,
        "selected_source_feature_width": 17 * chart_count,
        "selected_atlas_sha256": selected_candidate.atlas_sha256,
        "selected_fit_solver_receipt_sha256": solver_sha256,
        "selection_policy_id": (
            "largest_intrinsic_atlas_admissible_k_descending_8_to_1_v1"
        ),
        "claim_tier": "selected_shared_rank16_chart_transport_geometry",
    }
    selection = _seal_private_receipt(
        selector.StreamingSharedChartAtlasSelectionReceiptV1,
        selector._SELECTION_RECEIPT_DOMAIN,
        selection_values,
        digest_function=selector._digest,
    )
    reparsed = selector.parse_streaming_shared_chart_atlas_selection_receipt_v1(
        selection.state_dict()
    )
    assert reparsed == selection
    return selection, selected_diagnosis


def _build_complete_model_free_v2_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], bytes, object]:
    """Adapt the real production-cardinality outer harness to the v2 schema."""

    harness_module = _load_v1_completed_harness_module()
    harness = harness_module._ModelFreeCompletedReportHarness(monkeypatch)
    base = harness.build_report()

    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1_worker
    from fisher_graph import gemma3_chart_native_neighbor_v2_experiment as worker
    from fisher_graph import (
        gemma3_chart_native_neighbor_a_occupancy_diagnostic as diagnostic,
    )
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as selector

    root = Path(__file__).resolve().parents[1]
    from fisher_graph import (
        gemma3_chart_native_neighbor_v2_implementation_qualification as qualification,
    )

    qualification_path = root / report_io.DEFAULT_IMPLEMENTATION_QUALIFICATION_PATH
    qualification_state = qualification.load_qualification(
        qualification_path,
        repository_root=root,
    )
    qualification_raw = qualification_path.read_bytes()
    qualification_sha256 = qualification_state["qualification_sha256"]
    qualification_file_sha256 = hashlib.sha256(qualification_raw).hexdigest()
    diagnostic_result = diagnostic.strict_load_result(
        root / worker.DEFAULT_A_DIAGNOSTIC_RESULT_PATH,
        diagnostic_setup_path=root / worker.DEFAULT_A_DIAGNOSTIC_SETUP_PATH,
    )
    real_parse_diagnosis = (
        selector.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1
    )
    preselection = real_parse_diagnosis(
        diagnostic_result["K8_occupancy_diagnosis"]
    )
    prior = worker.load_required_K8_diagnostic(repository_root=root)
    agreement = worker.compare_K8_diagnostic_semantics(
        prior=prior,
        live_diagnosis=preselection,
    )

    atlas_sha256 = base["A_fit"]["atlas_sha256"]
    selection, selected = _authentic_production_selection_receipt(
        monkeypatch,
        tmp_path,
        preselection=preselection,
        selected_atlas_sha256=atlas_sha256,
    )
    chart_count = selection.selected_chart_count
    assert type(chart_count) is int
    solver_sha256 = selection.selected_fit_solver_receipt_sha256
    assert type(solver_sha256) is str

    base_capture = v1_worker._parse_role_capture_payload(
        base["A_fit"]["capture_receipt"]
    )
    spool_by_mode = {
        mode_id: (artifact_sha256, tensor_sha256)
        for mode_id, artifact_sha256, tensor_sha256 in zip(
            preselection.fit_mode_ids,
            preselection.spool_artifact_sha256s,
            preselection.spool_tensor_catalog_sha256s,
        )
    }
    aligned_capture = _rehash_capture_with_mode_artifacts(
        base_capture,
        tuple(
            (
                mode_id,
                spool_by_mode[mode_id][0],
                preselection.fit_row_catalog_sha256,
                spool_by_mode[mode_id][1],
            )
            for mode_id, _artifact, _rows, _tensors in (
                base_capture.mode_raw_artifacts
            )
        ),
    )

    preflight_unsigned = {
        "schema": report_io.V2_PREFLIGHT_SCHEMA,
        "v1_preflight_receipt_sha256": base["preflight"]["receipt_sha256"],
        "v2_setup_sha256": v2_setup.PINNED_SETUP_SHA256,
        "v2_setup_file_sha256": v2_setup.PINNED_SETUP_FILE_SHA256,
        "implementation_qualification_sha256": qualification_sha256,
        "implementation_qualification_file_sha256": qualification_file_sha256,
        "required_K8_diagnostic": prior.safe_payload(),
        "model_runtime_touched_before_v1_static_preflight": False,
        "numeric_selector_touched_before_v1_static_preflight": False,
    }
    v2_preflight = {
        **preflight_unsigned,
        "receipt_sha256": report_io._digest(
            report_io._PREFLIGHT_DOMAIN, preflight_unsigned
        ),
    }
    selected_resource = _selected_resource_receipt(chart_count)
    atlas_raw = b"model-free-selected-v2-atlas"
    atlas_file_sha256 = hashlib.sha256(atlas_raw).hexdigest()
    A_artifacts = {
        "A_capture_receipt": aligned_capture.receipt_sha256,
        "A_mode_lineage_catalog": base["A_fit"][
            "mode_lineage_catalog_sha256"
        ],
        "A_v2_preflight": v2_preflight["receipt_sha256"],
        "A_v2_implementation_qualification": qualification_sha256,
        "A_prior_K8_result": prior.result_sha256,
        "A_prior_K8_result_file": prior.result_file_sha256,
        "A_prior_K8_diagnosis": prior.diagnosis_receipt_sha256,
        "A_prior_K8_candidate": prior.candidate_receipt_sha256,
        "A_live_preselection_K8_diagnosis": preselection.receipt_sha256,
        "A_live_selector_K8_diagnosis": preselection.receipt_sha256,
        "A_K8_agreement": agreement.receipt_sha256,
        "A_selection_receipt": selection.receipt_sha256,
        "A_selection_kernel": selection.selection_kernel_receipt_sha256,
        "A_selected_resource": selected_resource["receipt_sha256"],
        "A_shared_chart_atlas": atlas_sha256,
        "A_shared_chart_atlas_file": atlas_file_sha256,
        "A_streaming_kernel": base["A_fit"][
            "streaming_kernel_receipt_sha256"
        ],
        "A_transport_catalog": base["A_fit"]["transport_catalog_sha256"],
    }
    A_stage = v1_worker.build_frozen_stage_receipt(
        role=v1_worker.FIT_ROLE,
        preflight_receipt_sha256=base["preflight"]["receipt_sha256"],
        role_panel_binding_sha256_value=(
            aligned_capture.role_panel_binding_sha256
        ),
        ordered_objective_row_catalog_sha256=(
            aligned_capture.ordered_objective_row_catalog_sha256
        ),
        score_objective_receipt_sha256=(
            aligned_capture.score_objective_receipt_sha256
        ),
        prior_stage_receipt_sha256=None,
        artifact_sha256s=A_artifacts,
        primary_eligible_example_catalog_sha256=None,
    )
    B_stage = _rebuild_stage_with_prior(
        base["B_discovery"]["stage_receipt"],
        prior_stage_receipt_sha256=A_stage.stage_receipt_sha256,
    )
    C_stage = _rebuild_stage_with_prior(
        base["C_confirmation"]["stage_receipt"],
        prior_stage_receipt_sha256=B_stage.stage_receipt_sha256,
    )

    A = {
        **base["A_fit"],
        "atlas_artifact_file_sha256": atlas_file_sha256,
        "published_atlas_file_name": report_io.DEFAULT_ATLAS_OUTPUT.name,
        "capture_receipt": {
            **aligned_capture.payload_without_sha256(),
            "receipt_sha256": aligned_capture.receipt_sha256,
        },
        "stage_receipt": _stage_payload(A_stage),
        "selection_receipt": selection.state_dict(),
        "preselection_K8_diagnosis": preselection.state_dict(),
        "K8_agreement_receipt": agreement.state_dict(),
        "selected_resource_receipt": selected_resource,
        "selected_chart_count": chart_count,
        "selected_source_feature_width": 17 * chart_count,
        "selected_tangent_rank": 16,
        "claim_tier": selection.claim_tier,
        "K8_numeric_evaluation_count": 2,
    }
    B = deepcopy(base["B_discovery"])
    C = deepcopy(base["C_confirmation"])
    B["stage_receipt"] = _stage_payload(B_stage)
    C["stage_receipt"] = _stage_payload(C_stage)
    payload = {
        key: deepcopy(value)
        for key, value in base.items()
        if key not in {"schema", "format_version", "report_sha256"}
    }
    payload.update(
        {
            "v2_preflight": v2_preflight,
            "A_fit": A,
            "B_discovery": B,
            "C_confirmation": C,
            "claims": report_io.v2_claims_payload(chart_count),
            "protocol": report_io.v2_protocol_payload(chart_count),
            "methodology": report_io.v2_methodology_payload(),
        }
    )
    report = report_io.seal_v2_experiment_report(payload)
    atlas = SimpleNamespace(
        config=selected.config,
        atlas_sha256=atlas_sha256,
        fit_solver_receipt_sha256=solver_sha256,
        fit_role=selected.fit_role,
        fit_panel_binding_sha256=selected.fit_panel_binding_sha256,
        fit_row_catalog_sha256=selected.fit_row_catalog_sha256,
        fit_mode_ids=selected.fit_mode_ids,
        fit_context_catalog_sha256=selected.fit_context_catalog_sha256,
        fit_balanced_row_weight_catalog_sha256=(
            selected.fit_balanced_row_weight_catalog_sha256
        ),
    )
    return report, atlas_raw, atlas


def test_report_module_does_not_reuse_v1_report_schema_validator_or_publisher() -> None:
    source = Path(report_io.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "EXPERIMENT_REPORT_SCHEMA",
        "_validate_completed_experiment_report",
        "strict_load_experiment_report",
        "publish_experiment_outputs",
    }
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert forbidden.isdisjoint(referenced)
    assert "_strict_rehash_frozen_repository_inputs" in referenced


def test_seal_and_strict_bytes_are_canonical_and_digest_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        report_io,
        "_validate_completed_v2_report",
        lambda value: observed.append(dict(value)),
    )
    sealed = report_io.seal_v2_experiment_report(
        {"payload": {"flag": False, "count": 2}}
    )
    raw = report_io.canonical_json_bytes(sealed)
    assert sealed["report_sha256"] == report_io._digest(
        report_io._REPORT_DOMAIN,
        {key: value for key, value in sealed.items() if key != "report_sha256"},
    )
    assert report_io.strict_load_v2_experiment_report_bytes(raw) == sealed
    assert observed == [sealed, sealed]


def test_strict_bytes_reject_duplicate_noncanonical_and_digest_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_io, "_validate_completed_v2_report", lambda value: None)
    sealed = report_io.seal_v2_experiment_report({"payload": 1})
    with pytest.raises(ValueError, match="invalid"):
        report_io.strict_load_v2_experiment_report_bytes(
            b'{"schema":"first","schema":"second"}'
        )
    with pytest.raises(ValueError, match="canonical"):
        report_io.strict_load_v2_experiment_report_bytes(
            report_io.canonical_json_bytes(sealed) + b"\n"
        )
    tampered = deepcopy(sealed)
    tampered["payload"] = 2
    with pytest.raises(ValueError, match="digest"):
        report_io.strict_load_v2_experiment_report_bytes(
            report_io.canonical_json_bytes(tampered)
        )


def test_coherently_rehashed_bool_for_format_version_is_rejected() -> None:
    report = _minimal_top_level_report(format_version=True)
    with pytest.raises(ValueError, match="envelope"):
        report_io.strict_load_v2_experiment_report_bytes(_reseal_report(report))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_runtime_touched_before_v1_static_preflight", 0),
        ("numeric_selector_touched_before_v1_static_preflight", 0),
    ],
)
def test_coherently_rehashed_v2_preflight_bool_int_attacks_fail(
    field: str,
    replacement: object,
) -> None:
    payload = _valid_v2_preflight_payload()
    assert report_io._validate_v2_preflight(payload) == payload
    payload[field] = replacement
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    payload["receipt_sha256"] = report_io._digest(
        report_io._PREFLIGHT_DOMAIN, unsigned
    )
    with pytest.raises(ValueError, match="preflight"):
        report_io._validate_v2_preflight(payload)


@pytest.mark.parametrize(
    ("chart_count", "evaluations"),
    [(True, 2), (4, 2.0)],
)
def test_A_contract_rejects_bool_K_and_float_evaluation_count(
    chart_count: object,
    evaluations: object,
) -> None:
    report = {"A_fit": _minimal_A(chart_count=chart_count, evaluations=evaluations)}
    with pytest.raises(ValueError, match="selected-K output contract"):
        report_io._validate_A_fit(
            report,
            preflight={},
            v2_preflight={},
            manifest={},
            runtime=object(),
            expected_streaming_kernel=_SHA0,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        (
            "methodology",
            "whole_protocol_preregistered_from_scratch_claimed",
            0,
        ),
        ("claims", "weights_alone_determine_neighbors", 0),
        ("protocol", "selected_chart_count", 4.0),
    ],
)
def test_disclosure_comparison_is_type_sensitive(
    section: str,
    field: str,
    replacement: object,
) -> None:
    disclosures = {
        "methodology": report_io.v2_methodology_payload(),
        "claims": report_io.v2_claims_payload(4),
        "protocol": report_io.v2_protocol_payload(4),
    }
    report_io._validate_disclosures(disclosures, selected_chart_count=4)
    disclosures[section][field] = replacement
    with pytest.raises(ValueError, match="methodology|claim|protocol"):
        report_io._validate_disclosures(disclosures, selected_chart_count=4)


def test_disclosures_state_post_A_scope_and_no_overclaim_boundary() -> None:
    methodology = report_io.v2_methodology_payload()
    claims = report_io.v2_claims_payload(4)
    protocol = report_io.v2_protocol_payload(4)
    assert methodology == {
        "A_is_fit_development_not_fresh_confirmation": True,
        "B_C_are_prospective_to_selected_K": True,
        "K8_failure_interpretation": (
            "K8_failed_this_A_balanced_solver_gate_not_K8_is_universally_impossible"
        ),
        "K8_prior_is_nonselecting": True,
        "descending_selector_stopped_at_first_full_intrinsic_pass": True,
        "one_A_model_capture": True,
        "two_deterministic_CPU_K8_center_evaluations": True,
        "v2_is_post_A_informed_after_terminal_v1_K8_failure": True,
        "whole_protocol_preregistered_from_scratch_claimed": False,
    }
    assert protocol["selected_tangent_rank"] == 16
    assert protocol["selected_chart_count"] == 4
    assert protocol["selected_source_feature_width"] == 68
    assert all(
        claims[field] is False
        for field in (
            "stable_atlas_geometry_claimed",
            "causal_interchangeability_claimed",
            "parameter_sharing_claimed",
            "merge_or_deletion_ready",
            "compression_ready",
            "downstream_response_supported_chart_neighbor_claimed",
            "weights_alone_determine_neighbors",
        )
    )
    with pytest.raises(ValueError, match="selected K"):
        report_io.v2_claims_payload(True)


def test_selected_atlas_closes_full_config_solver_and_fit_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as selector

    config = SharedChartNumericsConfig(chart_count=4)
    candidate = SimpleNamespace(
        atlas_sha256=_SHA0,
        fit_solver_receipt_sha256=_SHA1,
    )
    selected = SimpleNamespace(
        config=config,
        candidate=candidate,
        fit_role="A_fit",
        fit_panel_binding_sha256=_SHA0,
        fit_row_catalog_sha256=_SHA0,
        fit_mode_ids=("mode",),
        fit_context_catalog_sha256=_SHA0,
        fit_balanced_row_weight_catalog_sha256=_SHA0,
    )
    selection = SimpleNamespace(
        attempted_diagnoses=(selected,),
        selected_atlas_sha256=_SHA0,
        selected_fit_solver_receipt_sha256=_SHA1,
    )
    monkeypatch.setattr(
        selector,
        "parse_streaming_shared_chart_atlas_selection_receipt_v1",
        lambda value: selection,
    )
    atlas = SimpleNamespace(
        config=config,
        atlas_sha256=_SHA0,
        fit_solver_receipt_sha256=_SHA1,
        fit_role=selected.fit_role,
        fit_panel_binding_sha256=selected.fit_panel_binding_sha256,
        fit_row_catalog_sha256=selected.fit_row_catalog_sha256,
        fit_mode_ids=selected.fit_mode_ids,
        fit_context_catalog_sha256=selected.fit_context_catalog_sha256,
        fit_balanced_row_weight_catalog_sha256=(
            selected.fit_balanced_row_weight_catalog_sha256
        ),
    )
    A = {
        "selection_receipt": {},
        "selected_chart_count": 4,
        "selected_source_feature_width": 68,
        "selected_tangent_rank": 16,
        "atlas_sha256": _SHA0,
    }
    report_io._validate_selected_atlas(atlas, A=A)

    atlas.config = replace(config, lloyd_iteration_count=13)
    with pytest.raises(ValueError, match="atlas closure"):
        report_io._validate_selected_atlas(atlas, A=A)
    atlas.config = config
    atlas.fit_solver_receipt_sha256 = _SHA2
    with pytest.raises(ValueError, match="atlas closure"):
        report_io._validate_selected_atlas(atlas, A=A)


@pytest.mark.parametrize("case", ["report_name", "svg_name", "different_parent"])
def test_public_strict_loader_rejects_noncanonical_output_paths_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    if case == "report_name":
        report_path = tmp_path / "renamed.json"
    elif case == "svg_name":
        svg_path = tmp_path / "renamed.svg"
    else:
        other = tmp_path / "other"
        other.mkdir()
        atlas_path = other / report_io.DEFAULT_ATLAS_OUTPUT.name
    monkeypatch.setattr(
        report_io,
        "_read_regular",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("file read occurred before path validation")
        ),
    )
    with pytest.raises(ValueError, match="frozen names|one directory"):
        report_io.strict_load_v2_experiment_report(
            report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )


def test_live_qualification_replay_validates_the_already_read_bytes() -> None:
    source = Path(report_io.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_strict_live_rehash"
    )
    calls = {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }
    assert "qualification.validate_qualification" in calls
    assert "qualification.load_qualification" not in calls


def test_publisher_is_report_last_and_returns_typed_bound_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    loaded, atlas = _patch_publication_science(
        monkeypatch, atlas_source=atlas_source, svg_raw=svg_raw
    )
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    events: list[str] = []

    def prelink_rehash(value: object, *, repository_root: Path) -> None:
        del value, repository_root
        assert svg_path.is_file()
        assert atlas_path.is_file()
        assert not report_path.exists()
        events.append("live_rehash_before_report")

    monkeypatch.setattr(report_io, "_strict_live_rehash", prelink_rehash)
    publication = report_io.publish_v2_experiment_outputs(
        report={"opaque": "report"},
        svg_bytes=svg_raw,
        atlas_source_path=atlas_source,
        report_path=report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
        repository_root=tmp_path,
    )
    assert events == ["live_rehash_before_report"]
    assert isinstance(publication, report_io.PublishedV2ExperimentReceipt)
    publication.validate_receipt()
    assert report_path.read_bytes() == report_io.canonical_json_bytes(
        {"opaque": "report"}
    )
    assert svg_path.read_bytes() == svg_raw
    assert atlas_path.read_bytes() == b"atlas"
    assert publication.report_sha256 == loaded["report_sha256"]
    assert publication.atlas_sha256 == atlas.atlas_sha256
    assert not list(tmp_path.glob(".*.tmp"))
    report_io.finalize_published_v2_experiment(publication)
    with pytest.raises(PermissionError, match="no live in-process authority"):
        report_io.rollback_published_v2_experiment(publication)


def test_publisher_no_clobber_and_exact_destination_contract(tmp_path: Path) -> None:
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    report_path.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        report_io.publish_v2_experiment_outputs(
            report={},
            svg_bytes=b"svg",
            atlas_source_path=tmp_path / "source",
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    with pytest.raises(ValueError, match="destinations"):
        report_io.publish_v2_experiment_outputs(
            report={},
            svg_bytes=b"svg",
            atlas_source_path=tmp_path / "source",
            report_path=tmp_path / "renamed.json",
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )


def test_publication_reload_failure_unlinks_and_fsyncs_report_before_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch,
        atlas_source=atlas_source,
        svg_raw=svg_raw,
        strict_reload_error=RuntimeError("injected strict reload failure"),
    )
    monkeypatch.setattr(report_io, "_strict_live_rehash", lambda *a, **k: None)
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    states: list[tuple[bool, bool, bool]] = []

    def record_fsync(paths: object) -> None:
        del paths
        states.append(
            (report_path.exists(), svg_path.exists(), atlas_path.exists())
        )

    monkeypatch.setattr(report_io, "_fsync_directories", record_fsync)
    with pytest.raises(RuntimeError, match="injected strict reload failure"):
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    assert (False, True, True) in states
    assert states[-1] == (False, False, False)
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("missing_kind", ["report", "svg", "atlas"])
def test_internal_rollback_tolerates_an_already_missing_published_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_kind: str,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch,
        atlas_source=atlas_source,
        svg_raw=svg_raw,
    )
    monkeypatch.setattr(report_io, "_strict_live_rehash", lambda *a, **k: None)
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    missing = {
        "report": report_path,
        "svg": svg_path,
        "atlas": atlas_path,
    }[missing_kind]

    def remove_one_then_fail(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        assert report_path.is_file()
        assert svg_path.is_file()
        assert atlas_path.is_file()
        missing.unlink()
        raise RuntimeError("original strict reload failure after missing target")

    monkeypatch.setattr(
        report_io,
        "strict_load_v2_experiment_report",
        remove_one_then_fail,
    )
    with pytest.raises(
        RuntimeError,
        match="original strict reload failure after missing target",
    ):
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_first_post_link_identity_failure_still_removes_every_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch,
        atlas_source=atlas_source,
        svg_raw=svg_raw,
    )
    monkeypatch.setattr(report_io, "_strict_live_rehash", lambda *a, **k: None)
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    real_lstat = Path.lstat
    real_link = report_io.os.link
    linked_svg = False
    failed = False

    def record_link(source: Path | str, destination: Path | str) -> None:
        nonlocal linked_svg
        real_link(source, destination)
        if Path(destination) == svg_path:
            linked_svg = True

    def fail_first_post_link_lstat(path: Path) -> os.stat_result:
        nonlocal failed
        if path == svg_path and linked_svg and not failed:
            failed = True
            raise OSError("injected first post-link identity failure")
        return real_lstat(path)

    monkeypatch.setattr(report_io.os, "link", record_link)
    monkeypatch.setattr(Path, "lstat", fail_first_post_link_lstat)
    with pytest.raises(OSError, match="first post-link identity failure"):
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    assert failed is True
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("target_kind", ["svg", "report"])
def test_post_temp_unlink_identity_failure_uses_retained_inode_for_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch,
        atlas_source=atlas_source,
        svg_raw=svg_raw,
    )
    monkeypatch.setattr(report_io, "_strict_live_rehash", lambda *a, **k: None)
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    target = svg_path if target_kind == "svg" else report_path
    real_lstat = Path.lstat
    real_link = report_io.os.link
    real_unlink = report_io.os.unlink
    linked_temporary_destinations: dict[Path, Path] = {}
    just_unlinked_target_temporary = False
    failed = False

    def record_link(source: Path | str, destination: Path | str) -> None:
        real_link(source, destination)
        linked_temporary_destinations[Path(source)] = Path(destination)

    def record_unlink(path: Path | str) -> None:
        nonlocal just_unlinked_target_temporary
        candidate = Path(path)
        real_unlink(candidate)
        if linked_temporary_destinations.get(candidate) == target:
            just_unlinked_target_temporary = True

    def fail_post_unlink_lstat(path: Path) -> os.stat_result:
        nonlocal failed
        if path == target and just_unlinked_target_temporary and not failed:
            failed = True
            raise OSError("injected post-temp-unlink identity failure")
        return real_lstat(path)

    monkeypatch.setattr(report_io.os, "link", record_link)
    monkeypatch.setattr(report_io.os, "unlink", record_unlink)
    monkeypatch.setattr(Path, "lstat", fail_post_unlink_lstat)
    with pytest.raises(OSError, match="post-temp-unlink identity failure"):
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    assert failed is True
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_replaced_temporary_is_never_unlinked_as_publisher_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch,
        atlas_source=atlas_source,
        svg_raw=svg_raw,
    )
    monkeypatch.setattr(report_io, "_strict_live_rehash", lambda *a, **k: None)
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    real_link = report_io.os.link
    real_unlink = report_io.os.unlink
    replaced_temporary: Path | None = None

    def replace_before_link(source: Path | str, destination: Path | str) -> None:
        nonlocal replaced_temporary
        source_path = Path(source)
        if replaced_temporary is None:
            real_unlink(source_path)
            source_path.write_bytes(b"attacker replacement temporary")
            replaced_temporary = source_path
        real_link(source_path, destination)

    monkeypatch.setattr(report_io.os, "link", replace_before_link)
    with pytest.raises(RuntimeError, match="temporary changed while linking") as error:
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    assert any(
        "temporary cleanup also failed" in note
        for note in getattr(error.value, "__notes__", [])
    )
    assert replaced_temporary is not None
    assert replaced_temporary.read_bytes() == b"attacker replacement temporary"
    assert not report_path.exists()
    # The raced source inode was never publisher-owned, so both names are
    # preserved rather than turning cleanup into an arbitrary unlink primitive.
    assert svg_path.read_bytes() == b"attacker replacement temporary"
    assert not atlas_path.exists()


@pytest.mark.parametrize("cleanup_fault", ["report_unlink", "report_fsync"])
def test_internal_publication_rollback_continues_and_preserves_original_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fault: str,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch,
        atlas_source=atlas_source,
        svg_raw=svg_raw,
        strict_reload_error=RuntimeError("original strict reload failure"),
    )
    monkeypatch.setattr(report_io, "_strict_live_rehash", lambda *a, **k: None)
    report_path = tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = tmp_path / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name
    real_unlink = report_io.os.unlink
    real_fsync_directories = report_io._fsync_directories
    fsync_call_count = 0

    def maybe_fail_unlink(path: Path | str) -> None:
        if cleanup_fault == "report_unlink" and Path(path) == report_path:
            raise OSError("injected report unlink cleanup failure")
        real_unlink(path)

    def maybe_fail_fsync(paths: object) -> None:
        nonlocal fsync_call_count
        fsync_call_count += 1
        if (
            cleanup_fault == "report_fsync"
            and fsync_call_count == 3
        ):
            raise OSError("injected report fsync cleanup failure")
        real_fsync_directories(paths)

    monkeypatch.setattr(report_io.os, "unlink", maybe_fail_unlink)
    monkeypatch.setattr(report_io, "_fsync_directories", maybe_fail_fsync)
    with pytest.raises(RuntimeError, match="original strict reload failure") as error:
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
            repository_root=tmp_path,
        )
    assert isinstance(error.value.__cause__, BaseExceptionGroup)
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert report_path.exists() is (cleanup_fault == "report_unlink")
    assert not list(tmp_path.glob(".*.tmp"))


def test_publisher_detects_source_atlas_change_across_byte_and_object_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas_source = tmp_path / "selected-atlas-source.bundle"
    atlas_source.write_bytes(b"atlas")
    svg_raw = b"<svg/>"
    _patch_publication_science(
        monkeypatch, atlas_source=atlas_source, svg_raw=svg_raw
    )
    from fisher_graph import shared_chart_energy_transport as shared

    def mutate_source(path: Path) -> object:
        path.write_bytes(b"changed-atlas")
        return SimpleNamespace(atlas_sha256=_SHA2)

    monkeypatch.setattr(shared, "load_shared_chart_atlas", mutate_source)
    with pytest.raises(RuntimeError, match="changed across replay"):
        report_io.publish_v2_experiment_outputs(
            report={"opaque": "report"},
            svg_bytes=svg_raw,
            atlas_source_path=atlas_source,
            report_path=tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name,
            svg_path=tmp_path / report_io.DEFAULT_SVG_OUTPUT.name,
            atlas_path=tmp_path / report_io.DEFAULT_ATLAS_OUTPUT.name,
            repository_root=tmp_path,
        )
    assert not (tmp_path / report_io.DEFAULT_REPORT_OUTPUT.name).exists()


def test_public_rollback_authenticates_all_targets_before_any_unlink(
    tmp_path: Path,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    Path(publication.atlas_path).write_bytes(b"attacker replacement")
    with pytest.raises(RuntimeError, match="identity"):
        report_io.rollback_published_v2_experiment(publication)
    assert Path(publication.report_path).read_bytes() == report_raw
    assert Path(publication.svg_path).read_bytes() == svg_raw
    assert Path(publication.atlas_path).read_bytes() == b"attacker replacement"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink"])
def test_public_rollback_rejects_non_single_link_targets_without_deleting_any(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    atlas_path = Path(publication.atlas_path)
    external = tmp_path / "external-atlas"
    external.write_bytes(atlas_raw)
    atlas_path.unlink()
    if unsafe_kind == "symlink":
        atlas_path.symlink_to(external)
    else:
        atlas_path.hardlink_to(external)
    with pytest.raises(ValueError, match="single-link|opened safely"):
        report_io.rollback_published_v2_experiment(publication)
    assert Path(publication.report_path).exists()
    assert Path(publication.svg_path).exists()
    assert atlas_path.exists()


@pytest.mark.parametrize("forgery", ["wrong_name", "different_parent"])
def test_self_rehashed_forged_publication_paths_are_rejected(
    tmp_path: Path,
    forgery: str,
) -> None:
    publication = _publication_receipt(
        tmp_path,
        report_raw=b"report",
        svg_raw=b"svg",
        atlas_raw=b"atlas",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"keep")
    if forgery == "wrong_name":
        forged = _reseal_publication(
            publication, report_path=str(unrelated)
        )
    else:
        other = tmp_path / "other"
        other.mkdir()
        forged = _reseal_publication(
            publication,
            report_path=str(other / report_io.DEFAULT_REPORT_OUTPUT.name),
        )
    with pytest.raises(ValueError, match="publication paths"):
        report_io.rollback_published_v2_experiment(forged)
    assert unrelated.read_bytes() == b"keep"


def test_unissued_self_rehashed_canonical_receipt_cannot_delete_matching_files(
    tmp_path: Path,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    forged = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    Path(forged.report_path).write_bytes(report_raw)
    Path(forged.svg_path).write_bytes(svg_raw)
    Path(forged.atlas_path).write_bytes(atlas_raw)
    with pytest.raises(PermissionError, match="no live in-process authority"):
        report_io.rollback_published_v2_experiment(forged)
    assert Path(forged.report_path).read_bytes() == report_raw
    assert Path(forged.svg_path).read_bytes() == svg_raw
    assert Path(forged.atlas_path).read_bytes() == atlas_raw


def test_reconstructed_receipt_has_no_authority_but_exact_issued_object_does(
    tmp_path: Path,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    reconstructed = _reseal_publication(publication)
    assert reconstructed == publication
    assert reconstructed is not publication
    with pytest.raises(PermissionError, match="no live in-process authority"):
        report_io.rollback_published_v2_experiment(reconstructed)
    report_io.rollback_published_v2_experiment(publication)
    assert not Path(publication.report_path).exists()
    assert not Path(publication.svg_path).exists()
    assert not Path(publication.atlas_path).exists()


def test_finalize_authenticates_then_retires_exact_rollback_authority(
    tmp_path: Path,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    report_io.finalize_published_v2_experiment(publication)
    assert Path(publication.report_path).read_bytes() == report_raw
    assert Path(publication.svg_path).read_bytes() == svg_raw
    assert Path(publication.atlas_path).read_bytes() == atlas_raw
    with pytest.raises(PermissionError, match="no live in-process authority"):
        report_io.rollback_published_v2_experiment(publication)


def test_finalize_drift_leaves_authority_live_but_refuses_replacement_deletion(
    tmp_path: Path,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    Path(publication.atlas_path).write_bytes(b"attacker replacement")
    with pytest.raises(RuntimeError, match="identity"):
        report_io.finalize_published_v2_experiment(publication)
    with pytest.raises(RuntimeError, match="identity"):
        report_io.rollback_published_v2_experiment(publication)
    assert Path(publication.report_path).read_bytes() == report_raw
    assert Path(publication.svg_path).read_bytes() == svg_raw
    assert Path(publication.atlas_path).read_bytes() == b"attacker replacement"


def test_public_rollback_removes_report_then_fsyncs_before_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    events: list[str] = []
    real_unlink = report_io.os.unlink

    def record_unlink(path: Path | str) -> None:
        events.append(f"unlink:{Path(path).name}")
        real_unlink(path)

    def record_fsync(paths: object) -> None:
        del paths
        events.append("fsync")

    monkeypatch.setattr(report_io.os, "unlink", record_unlink)
    monkeypatch.setattr(report_io, "_fsync_directories", record_fsync)
    report_io.rollback_published_v2_experiment(publication)
    assert events == [
        f"unlink:{report_io.DEFAULT_REPORT_OUTPUT.name}",
        "fsync",
        f"unlink:{report_io.DEFAULT_SVG_OUTPUT.name}",
        f"unlink:{report_io.DEFAULT_ATLAS_OUTPUT.name}",
        "fsync",
    ]


def test_public_rollback_attempts_both_sidecars_after_report_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_raw, svg_raw, atlas_raw = b"report", b"svg", b"atlas"
    publication = _publication_receipt(
        tmp_path,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    _write_publication_files(
        publication,
        report_raw=report_raw,
        svg_raw=svg_raw,
        atlas_raw=atlas_raw,
    )
    report_path = Path(publication.report_path)
    real_unlink = report_io.os.unlink

    def fail_report_unlink(path: Path | str) -> None:
        if Path(path) == report_path:
            raise OSError("injected report cleanup failure")
        real_unlink(path)

    monkeypatch.setattr(report_io.os, "unlink", fail_report_unlink)
    with pytest.raises(OSError, match="injected report cleanup failure"):
        report_io.rollback_published_v2_experiment(publication)
    assert report_path.exists()
    assert not Path(publication.svg_path).exists()
    assert not Path(publication.atlas_path).exists()
    # Partial cleanup retains exact-object authority for a bounded retry.
    monkeypatch.setattr(report_io.os, "unlink", real_unlink)
    report_io.rollback_published_v2_experiment(publication)
    assert not report_path.exists()
    with pytest.raises(PermissionError, match="no live in-process authority"):
        report_io.rollback_published_v2_experiment(publication)


def test_complete_production_cardinality_v2_report_seals_and_publicly_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real v2 top-level validator over the full outer lineage."""

    report, atlas_raw, atlas = _build_complete_model_free_v2_report(
        monkeypatch, tmp_path
    )
    # Do not stub `_validate_completed_v2_report`: both seal and this bytes
    # replay traverse it with 38 modes, 48 examples, 1,406 A transports per
    # role binding, exact-seven B/C, all four diagnostic subsets, and A->B->C.
    assert report_io.strict_load_v2_experiment_report_bytes(
        report_io.canonical_json_bytes(report)
    ) == report

    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1_worker
    from fisher_graph import shared_chart_energy_transport as shared
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as selector

    capture = v1_worker._parse_role_capture_payload(
        report["A_fit"]["capture_receipt"]
    )
    diagnosis = selector.parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
        report["A_fit"]["preselection_K8_diagnosis"]
    )
    selection = selector.parse_streaming_shared_chart_atlas_selection_receipt_v1(
        report["A_fit"]["selection_receipt"]
    )
    assert isinstance(
        selection,
        selector.StreamingSharedChartAtlasSelectionReceiptV1,
    )
    assert selection.attempted_chart_counts == (8, 7)
    assert selection.selected_chart_count == 7
    capture_axis = tuple(row[0] for row in capture.mode_raw_artifacts)
    # The capture receipt preserves event order; selector spools are frozen in
    # lexical order.  The passing replay proves mode-key projection, not a
    # false same-tuple-order assumption.
    assert capture_axis != diagnosis.fit_mode_ids
    assert tuple(sorted(capture_axis)) == diagnosis.fit_mode_ids
    assert sum(diagnosis.fit_row_count_by_family) == 24_732

    output = tmp_path / "outputs"
    output.mkdir()
    report_path = output / report_io.DEFAULT_REPORT_OUTPUT.name
    svg_path = output / report_io.DEFAULT_SVG_OUTPUT.name
    atlas_path = output / report_io.DEFAULT_ATLAS_OUTPUT.name
    svg_raw = v1_worker.render_neighbor_svg(
        mode_ids=report["mode_axis"],
        confirmed_edges=report["confirmed_edges"],
    )
    report_path.write_bytes(report_io.canonical_json_bytes(report))
    svg_path.write_bytes(svg_raw)
    atlas_path.write_bytes(atlas_raw)
    # Keep only the model-free atlas decoder seam.  The public loader executes
    # the real `_strict_live_rehash` over the issued setup, qualification,
    # source, panel, runtime, and prior closure.
    monkeypatch.setattr(shared, "load_shared_chart_atlas", lambda path: atlas)
    assert report_io.strict_load_v2_experiment_report(
        report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
        repository_root=Path(__file__).resolve().parents[1],
    ) == report


def test_coordinated_capture_selection_swap_fails_after_full_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _atlas_raw, _atlas = _build_complete_model_free_v2_report(
        monkeypatch, tmp_path
    )
    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1_worker

    attacked = deepcopy(report)
    capture = attacked["A_fit"]["capture_receipt"]
    capture["mode_raw_artifacts"][0]["artifact_sha256"] = _SHA2
    capture_unsigned = dict(capture)
    capture_unsigned.pop("receipt_sha256")
    capture["receipt_sha256"] = v1_worker._digest(
        v1_worker._CAPTURE_DOMAIN, capture_unsigned
    )
    A_stage = _rebuild_stage_with_prior(
        attacked["A_fit"]["stage_receipt"],
        prior_stage_receipt_sha256=None,
        artifact_overrides={"A_capture_receipt": capture["receipt_sha256"]},
    )
    B_stage = _rebuild_stage_with_prior(
        attacked["B_discovery"]["stage_receipt"],
        prior_stage_receipt_sha256=A_stage.stage_receipt_sha256,
    )
    C_stage = _rebuild_stage_with_prior(
        attacked["C_confirmation"]["stage_receipt"],
        prior_stage_receipt_sha256=B_stage.stage_receipt_sha256,
    )
    attacked["A_fit"]["stage_receipt"] = _stage_payload(A_stage)
    attacked["B_discovery"]["stage_receipt"] = _stage_payload(B_stage)
    attacked["C_confirmation"]["stage_receipt"] = _stage_payload(C_stage)
    with pytest.raises(ValueError, match="selector/capture spool lineage"):
        report_io.strict_load_v2_experiment_report_bytes(
            _reseal_report(attacked)
        )
