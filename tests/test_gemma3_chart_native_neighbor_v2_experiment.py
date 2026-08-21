from __future__ import annotations

import ast
import copy
from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import gemma3_chart_native_neighbor_v2_experiment as worker
from fisher_graph import gemma3_chart_native_neighbor_v2_setup as setup


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _prior_and_typed_diagnosis():
    from fisher_graph import (
        gemma3_chart_native_neighbor_a_occupancy_diagnostic as diagnostic,
    )
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1,
    )

    root = Path(__file__).resolve().parents[1]
    prior = worker.load_required_K8_diagnostic(repository_root=root)
    result = diagnostic.strict_load_result(
        root / worker.DEFAULT_A_DIAGNOSTIC_RESULT_PATH,
        diagnostic_setup_path=root / worker.DEFAULT_A_DIAGNOSTIC_SETUP_PATH,
    )
    typed = parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
        result["K8_occupancy_diagnosis"]
    )
    return prior, typed


def _production_capture_receipt_fixture():
    mode_ids = (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )
    example_ids = tuple(f"example.{index:02d}" for index in range(48))
    root = Path(__file__).resolve().parents[1]
    fresh_manifest = json.loads(
        (root / "examples/gemma3_chart_native_neighbor_fresh_manifest_v1.json")
        .read_text()
    )
    role_manifest = fresh_manifest["roles"][worker.FIT_ROLE]
    prior, _ = _prior_and_typed_diagnosis()
    frozen_raw_row_catalog = json.loads(prior.semantic_projection_bytes)[
        "fit_row_catalog_sha256"
    ]
    assert frozen_raw_row_catalog != role_manifest[
        "ordered_objective_row_catalog_sha256"
    ]
    panel = worker.v1.role_panel_binding_sha256(
        fresh_manifest,
        worker.FIT_ROLE,
    )
    spools = tuple(
        SimpleNamespace(
            mode_id=mode_id,
            artifact_sha256=_sha(f"artifact:{mode_id}"),
            row_catalog_sha256=frozen_raw_row_catalog,
            tensor_catalog_sha256=_sha(f"tensor:{mode_id}"),
        )
        for mode_id in mode_ids
    )
    receipt = worker.v1._build_role_capture_receipt(
        role=worker.FIT_ROLE,
        runtime_session_binding_sha256=_sha("runtime-session"),
        panel_binding_sha256=panel,
        role_manifest=role_manifest,
        spools=spools,
        batch_example_ids=tuple(
            (example_ids[index % len(example_ids)],) for index in range(45)
        ),
        lexical_receipts=tuple(_sha(f"lexical:{index}") for index in range(45)),
        execution_receipts=tuple(
            _sha(f"execution:{index}") for index in range(45)
        ),
        example_receipts=tuple(
            (example_id, _sha(f"capture:{example_id}"))
            for example_id in example_ids
        ),
        route_evidence=tuple(
            (
                f"mode.mutation.{mutation}",
                example_id,
                _sha(f"route-catalog:{mutation}:{example_id}"),
                _sha(f"route-receipt:{mutation}:{example_id}"),
            )
            for mutation in range(1, 37, 2)
            for example_id in example_ids
        ),
        dense_lexical_parity_receipt_sha256=_sha("dense-parity"),
        row_count=24_732,
    )
    return receipt, spools, fresh_manifest


@pytest.fixture(scope="module")
def no_admissible_terminal_evidence(
    tmp_path_factory: pytest.TempPathFactory,
):
    import torch

    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as selection
    from fisher_graph.shared_chart_energy_transport import (
        FIT_ROLE,
        ChartRowKey,
        SharedChartNumericsConfig,
        SharedChartRowBatch,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        write_raw_mode_row_spool,
    )

    destination = tmp_path_factory.mktemp("v2-no-admissible")
    keys = tuple(
        ChartRowKey("family.0", "example.0", row, row + 1)
        for row in range(68)
    )
    resources = StreamingChartResourceConfig(
        row_chunk_size=17,
        atlas_chunk_size=53,
        max_resident_bytes=512 * 1024 * 1024,
        max_artifact_bytes=128 * 1024 * 1024,
    )
    spools = tuple(
        write_raw_mode_row_spool(
            SharedChartRowBatch(
                role=FIT_ROLE,
                panel_binding_sha256=_sha("no-admissible-panel"),
                mode_id=f"mode.{mode:02d}",
                row_keys=keys,
                context_values=torch.zeros((len(keys), 20)),
                comparison_values=torch.ones((len(keys), 20)),
                generalized_contributions=torch.ones((len(keys), 20)),
            ),
            destination / f"mode-{mode:02d}.fgstream",
            resources=resources,
        )
        for mode in range(2)
    )
    result = selection.select_and_fit_shared_chart_atlas_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    assert result.atlas is None
    first = result.receipt.attempted_diagnoses[0]
    projection = worker.K8_semantic_projection(first)
    prior = worker.RequiredK8Diagnostic(
        result_path="model-free-test-result.json",
        result_file_sha256=_sha("model-free-result-file"),
        result_sha256=_sha("model-free-result"),
        diagnosis_receipt_sha256=first.receipt_sha256,
        candidate_receipt_sha256=first.candidate.receipt_sha256,
        semantic_projection_bytes=worker.canonical_json_bytes(projection),
        semantic_projection_sha256=worker._digest(
            worker._K8_PROJECTION_DOMAIN, projection
        ),
    )
    agreement = worker.compare_K8_diagnostic_semantics(
        prior=prior,
        live_diagnosis=first,
    )
    return result.receipt, agreement


def test_required_prior_is_canonical_single_read_and_agreement_is_exhaustive() -> None:
    prior, diagnosis = _prior_and_typed_diagnosis()
    prior.validate_snapshot()
    receipt = worker.compare_K8_diagnostic_semantics(
        prior=prior, live_diagnosis=diagnosis
    )
    assert receipt.compared_top_level_fields == (
        worker._K8_COMPARED_TOP_LEVEL_FIELDS
    )
    assert receipt.excluded_top_level_fields == (
        worker._K8_EXCLUDED_TOP_LEVEL_FIELDS
    )
    assert set(receipt.compared_top_level_fields) | set(
        receipt.excluded_top_level_fields
    ) == worker._K8_ALL_TOP_LEVEL_FIELDS
    assert receipt.common_semantic_projection_sha256 == (
        prior.semantic_projection_sha256
    )
    receipt.validate_receipt()


def test_required_prior_snapshot_rejects_path_sha_and_projection_forges() -> None:
    prior, _ = _prior_and_typed_diagnosis()
    with pytest.raises(ValueError, match="path drifted"):
        replace(prior, result_path=str(Path("/tmp") / prior.result_path)).validate_snapshot()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(prior, result_sha256="not-a-sha").validate_snapshot()
    projection = json.loads(prior.semantic_projection_bytes)
    projection["candidate"]["receipt_sha256"] = _sha("forged-candidate")
    projection_bytes = worker.canonical_json_bytes(projection)
    forged = replace(
        prior,
        semantic_projection_bytes=projection_bytes,
        semantic_projection_sha256=worker._digest(
            worker._K8_PROJECTION_DOMAIN,
            projection,
        ),
    )
    with pytest.raises(ValueError, match="semantic snapshot drifted"):
        forged.validate_snapshot()


def test_K8_projection_excludes_only_run_specific_spool_receipts() -> None:
    _, diagnosis = _prior_and_typed_diagnosis()
    original = diagnosis.state_dict()
    changed = copy.deepcopy(original)
    changed["receipt_sha256"] = "0" * 64
    changed["spool_artifact_sha256s"] = ["1" * 64] * len(
        changed["spool_artifact_sha256s"]
    )
    changed["spool_tensor_catalog_sha256s"] = ["2" * 64] * len(
        changed["spool_tensor_catalog_sha256s"]
    )
    assert worker.K8_semantic_projection(changed) == (
        worker.K8_semantic_projection(original)
    )
    changed_candidate = copy.deepcopy(changed)
    changed_candidate["candidate"]["final_sort_permutation"][0] = 99
    assert worker.K8_semantic_projection(changed_candidate) != (
        worker.K8_semantic_projection(original)
    )


def test_K8_projection_rejects_field_omission_and_boolean_K() -> None:
    _, diagnosis = _prior_and_typed_diagnosis()
    state = diagnosis.state_dict()
    state.pop("fit_context_catalog_sha256")
    with pytest.raises(ValueError, match="field inventory"):
        worker.K8_semantic_projection(state)
    state = diagnosis.state_dict()
    state["candidate"]["chart_count"] = True
    with pytest.raises(ValueError, match="chart_count=8"):
        worker.K8_semantic_projection(state)


def test_K8_comparator_requires_a_typed_authenticated_live_receipt() -> None:
    prior, diagnosis = _prior_and_typed_diagnosis()
    with pytest.raises(TypeError, match="typed validated"):
        worker.compare_K8_diagnostic_semantics(
            prior=prior, live_diagnosis=diagnosis.state_dict()
        )


def test_required_prior_rejects_noncanonical_pretty_rewrite(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    setup_target = tmp_path / worker.DEFAULT_A_DIAGNOSTIC_SETUP_PATH
    result_target = tmp_path / worker.DEFAULT_A_DIAGNOSTIC_RESULT_PATH
    setup_target.parent.mkdir(parents=True)
    result_target.parent.mkdir(parents=True)
    setup_target.write_bytes(
        (root / worker.DEFAULT_A_DIAGNOSTIC_SETUP_PATH).read_bytes()
    )
    parsed = json.loads(
        (root / worker.DEFAULT_A_DIAGNOSTIC_RESULT_PATH).read_text()
    )
    result_target.write_text(json.dumps(parsed, sort_keys=True))
    with pytest.raises(ValueError, match="noncanonical"):
        worker.load_required_K8_diagnostic(repository_root=tmp_path)


@pytest.mark.parametrize("chart_count", range(1, 9))
def test_selected_resource_receipt_replays_dynamic_K_exactly(
    chart_count: int,
) -> None:
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        estimate_streaming_chart_resources,
    )

    frozen = setup.build_setup()
    intrinsic = frozen["unchanged_intrinsic_atlas_contract"]
    dynamic = frozen["dynamic_transport_contract"]
    resources = dynamic["selection_streaming_resource_config"]
    rows = intrinsic["fit_pool_rows_per_mode"]
    modes = intrinsic["fit_pool_total_rows"] // rows
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=rows,
        mode_count=modes,
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
    receipt = worker._selected_resource_receipt(estimate=estimate)
    F = chart_count * (intrinsic["tangent_rank"] + 1)
    assert receipt["source_feature_width"] == F
    assert receipt["all_ordered_coefficient_bytes"] == (
        dynamic["ordered_pair_count"] * F * intrinsic["hidden_width"] * 8
    )


def test_selected_resource_receipt_rejects_payload_duck_typing() -> None:
    with pytest.raises(TypeError, match="StreamingChartResourceEstimate"):
        worker._selected_resource_receipt(
            estimate=SimpleNamespace(payload=lambda: {})
        )


@pytest.mark.parametrize(
    ("section", "field", "delta"),
    [
        ("intrinsic", "fit_pool_rows_per_mode", -12_366),
        ("intrinsic", "fit_pool_total_rows", 24_732),
        ("intrinsic", "hidden_width", 1),
        ("intrinsic", "tangent_rank", 1),
        ("resources", "row_chunk_size", 1),
        ("resources", "atlas_chunk_size", 1),
        ("resources", "max_resident_bytes", 4_096),
        ("resources", "max_artifact_bytes", 4_096),
    ],
)
def test_selected_resource_receipt_rejects_setup_derived_shape_and_budget_drift(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    delta: int,
) -> None:
    from fisher_graph.shared_chart_energy_transport import SharedChartNumericsConfig
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        estimate_streaming_chart_resources,
    )

    frozen = setup.build_setup()
    intrinsic = frozen["unchanged_intrinsic_atlas_contract"]
    dynamic = frozen["dynamic_transport_contract"]
    resources = dynamic["selection_streaming_resource_config"]
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=intrinsic["fit_pool_rows_per_mode"],
        mode_count=(
            intrinsic["fit_pool_total_rows"]
            // intrinsic["fit_pool_rows_per_mode"]
        ),
        config=SharedChartNumericsConfig(
            hidden_width=intrinsic["hidden_width"],
            chart_count=4,
            tangent_rank=intrinsic["tangent_rank"],
        ),
        resources=StreamingChartResourceConfig(
            row_chunk_size=resources["row_chunk_size"],
            atlas_chunk_size=resources["atlas_chunk_size"],
            max_resident_bytes=resources["max_resident_bytes"],
            max_artifact_bytes=resources["max_artifact_bytes"],
        ),
    )
    changed = copy.deepcopy(frozen)
    target = (
        changed["unchanged_intrinsic_atlas_contract"]
        if section == "intrinsic"
        else changed["dynamic_transport_contract"]
        ["selection_streaming_resource_config"]
    )
    target[field] += delta
    monkeypatch.setattr(worker.v2_setup, "build_setup", lambda: changed)
    with pytest.raises(RuntimeError, match="resource estimate drifted"):
        worker._selected_resource_receipt(estimate=estimate)


def test_A_capture_cross_binding_uses_distinct_frozen_raw_and_objective_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisher_graph import streaming_shared_chart_energy_transport as streaming

    receipt, duck_spools, fresh_manifest = _production_capture_receipt_fixture()
    preflight = SimpleNamespace(
        v1_preflight=SimpleNamespace(fresh_manifest=fresh_manifest)
    )
    with pytest.raises(ValueError, match="inventory/panel binding"):
        worker._validated_A_capture_spools(
            preflight=preflight,
            captured=worker.v1.CapturedRole(raw_spools=(), receipt=receipt),
            max_artifact_bytes=128 * 1024 * 1024,
        )
    with pytest.raises(TypeError, match="typed raw mode spools"):
        worker._validated_A_capture_spools(
            preflight=preflight,
            captured=worker.v1.CapturedRole(
                raw_spools=duck_spools,
                receipt=receipt,
            ),
            max_artifact_bytes=128 * 1024 * 1024,
        )
    shared_rows = (None,) * receipt.row_count
    typed_spools = []
    for ordinal, row in enumerate(duck_spools):
        spool = object.__new__(streaming.RawModeRowSpool)
        object.__setattr__(
            spool,
            "_store",
            SimpleNamespace(
                path=Path(f"/virtual/frozen-A-{ordinal}.fgstream"),
                artifact_sha256=row.artifact_sha256,
                row_keys=shared_rows,
            ),
        )
        object.__setattr__(spool, "role", worker.FIT_ROLE)
        object.__setattr__(
            spool,
            "panel_binding_sha256",
            receipt.role_panel_binding_sha256,
        )
        object.__setattr__(spool, "mode_id", row.mode_id)
        object.__setattr__(
            spool,
            "row_catalog_sha256",
            row.row_catalog_sha256,
        )
        object.__setattr__(
            spool,
            "tensor_catalog_sha256",
            row.tensor_catalog_sha256,
        )
        object.__setattr__(spool, "hidden_width", 640)
        object.__setattr__(spool, "row_chunk_size", 256)
        object.__setattr__(
            spool,
            "kernel_receipt_sha256",
            _sha("frozen-streaming-kernel"),
        )
        typed_spools.append(spool)
    by_path = {spool.path: spool for spool in typed_spools}
    monkeypatch.setattr(
        streaming,
        "open_raw_mode_row_spool",
        lambda source, **_kwargs: by_path[Path(source)],
    )
    reopened = worker._validated_A_capture_spools(
        preflight=preflight,
        captured=worker.v1.CapturedRole(
            raw_spools=tuple(typed_spools),
            receipt=receipt,
        ),
        max_artifact_bytes=128 * 1024 * 1024,
    )
    assert reopened == tuple(typed_spools)
    assert all(
        spool.row_catalog_sha256
        != receipt.ordered_objective_row_catalog_sha256
        for spool in reopened
    )


def test_source_orders_K8_agreement_before_descending_selector_and_B_freeze() -> None:
    source = Path(worker.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "select_fit_A_chart_atlas_and_transports_v2"
    )
    calls: dict[str, int] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if name in {
                "diagnose_shared_chart_atlas_occupancy_streaming",
                "compare_K8_diagnostic_semantics",
                "select_and_fit_shared_chart_atlas_streaming",
                "revalidate_v2_authorization",
                "build_frozen_stage_receipt",
            }:
                calls.setdefault(name, node.lineno)
    assert calls["diagnose_shared_chart_atlas_occupancy_streaming"] < calls[
        "compare_K8_diagnostic_semantics"
    ] < calls["select_and_fit_shared_chart_atlas_streaming"]
    assert calls["revalidate_v2_authorization"] < calls[
        "build_frozen_stage_receipt"
    ]


def test_orchestration_revalidates_before_heldouts_and_keeps_close_retry() -> None:
    source = Path(worker.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_chart_native_neighbor_v2_experiment"
    )
    calls = sorted(
        [
        (
            node.lineno,
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else "",
            node,
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        ],
        key=lambda item: (item[0], item[1]),
    )
    B_open = next(
        line
        for line, name, node in calls
        if name == "open_role"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "DISCOVERY_ROLE"
    )
    C_open = next(
        line
        for line, name, node in calls
        if name == "open_role"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "CONFIRMATION_ROLE"
    )
    revalidations = [
        line for line, name, _ in calls if name == "revalidate_v2_authorization"
    ]
    assert len([line for line in revalidations if line < B_open]) >= 2
    assert any(B_open < line < C_open for line in revalidations)
    runtime_close = next(
        line
        for line, name, node in calls
        if name == "close"
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "runtime"
    )
    report_import = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        and node.module == "fisher_graph"
        and any(
            alias.name == "gemma3_chart_native_neighbor_v2_report"
            for alias in node.names
        )
        and node.lineno > runtime_close
    )
    assert any(line > report_import for line in revalidations)
    runtime_none_assignments = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "runtime" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    assert any(line > runtime_close for line in runtime_none_assignments)


def test_no_admissible_terminal_carries_full_receipts_without_artifact(
    no_admissible_terminal_evidence,
) -> None:
    selection, agreement = no_admissible_terminal_evidence
    error = worker.NoAdmissibleAtlasError(
        selection_receipt=selection, K8_agreement=agreement
    )
    assert error.selection_receipt["attempted_chart_counts"] == list(
        range(8, 0, -1)
    )
    assert error.K8_agreement == agreement.state_dict()
    assert error.safe_summary()["B_semantically_opened"] is False


def test_no_admissible_terminal_rejects_coordinated_receipt_forges(
    no_admissible_terminal_evidence,
) -> None:
    selection, agreement = no_admissible_terminal_evidence
    forged_selection = object.__new__(type(selection))
    for field in fields(type(selection)):
        object.__setattr__(
            forged_selection,
            field.name,
            getattr(selection, field.name),
        )
    object.__setattr__(
        forged_selection,
        "receipt_sha256",
        _sha("forged-selection"),
    )
    with pytest.raises(ValueError, match="selection receipt hash"):
        worker.NoAdmissibleAtlasError(
            selection_receipt=forged_selection,
            K8_agreement=agreement,
        )

    forged_agreement = replace(
        agreement,
        live_diagnosis_receipt_sha256=_sha("unrelated-live-diagnosis"),
        live_candidate_receipt_sha256=_sha("unrelated-live-candidate"),
        receipt_sha256="0" * 64,
    )
    forged_agreement = replace(
        forged_agreement,
        receipt_sha256=worker._digest(
            worker._K8_AGREEMENT_DOMAIN,
            forged_agreement.payload_without_sha256(),
        ),
    )
    forged_agreement.validate_receipt()
    with pytest.raises(ValueError, match="agreed live K8 lineage"):
        worker.NoAdmissibleAtlasError(
            selection_receipt=selection,
            K8_agreement=forged_agreement,
        )


def test_cli_requires_an_explicit_action() -> None:
    with pytest.raises(SystemExit) as error:
        worker.main([])
    assert error.value.code == 2


def test_cli_preflight_dispatch_cannot_construct_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binding = SimpleNamespace(repository_root=tmp_path)
    monkeypatch.setattr(
        worker.v1, "build_default_execution_binding", lambda **_: binding
    )
    receipt = SimpleNamespace(
        receipt_sha256=_sha("preflight-cli"),
        safe_payload_without_sha256=lambda: {"schema": "preflight.test"},
    )
    monkeypatch.setattr(worker, "static_preflight_v2", lambda **_: receipt)
    monkeypatch.setattr(
        worker.v1,
        "_build_default_runtime_after_preflight",
        lambda *_args, **_kwargs: pytest.fail("preflight-only constructed runtime"),
    )
    assert worker.main(
        ["--preflight-only", "--repository-root", str(tmp_path)]
    ) == 0
    assert json.loads(capsys.readouterr().out)["receipt_sha256"] == (
        receipt.receipt_sha256
    )


def test_cli_no_admissible_emits_full_canonical_receipts_without_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    no_admissible_terminal_evidence,
) -> None:
    selection, agreement = no_admissible_terminal_evidence
    terminal = worker.NoAdmissibleAtlasError(
        selection_receipt=selection,
        K8_agreement=agreement,
    )
    monkeypatch.setattr(
        worker.v1,
        "build_default_execution_binding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def fail_without_model_run(**_kwargs: object):
        raise terminal

    monkeypatch.setattr(
        worker,
        "run_chart_native_neighbor_v2_experiment",
        fail_without_model_run,
    )
    report = tmp_path / "outputs" / worker.DEFAULT_REPORT_OUTPUT.name
    svg = report.with_name(worker.DEFAULT_SVG_OUTPUT.name)
    atlas = report.with_name(worker.DEFAULT_ATLAS_OUTPUT.name)
    assert worker.main(
        [
            "--run-v2",
            "--repository-root",
            str(tmp_path),
            "--report",
            str(report),
            "--svg",
            str(svg),
            "--atlas",
            str(atlas),
        ]
    ) == 3
    captured = capsys.readouterr()
    summary = json.loads(captured.err)
    assert summary["selection_receipt"] == selection.state_dict()
    assert summary["K8_agreement_receipt"] == agreement.state_dict()
    assert captured.err.strip().encode("utf-8") == worker.canonical_json_bytes(
        summary
    )
    assert not report.exists()
    assert not svg.exists()
    assert not atlas.exists()


def _fake_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    A_terminal: BaseException | None = None,
    close_failures: int = 0,
):
    from fisher_graph import gemma3_chart_native_neighbor_fresh_panels as panels
    from fisher_graph import gemma3_chart_native_neighbor_v2_report as report_io

    events: list[str] = []
    output = tmp_path / "out"
    temporary = tmp_path / "tmp"
    output.mkdir()
    temporary.mkdir()
    binding = SimpleNamespace(
        repository_root=tmp_path,
        output_directory=output,
        temp_directory=temporary,
        test_only_allow_nonproduction_shape=True,
    )
    base = SimpleNamespace(
        binding=binding,
        receipt_sha256=_sha("base-preflight"),
        fresh_manifest={
            "roles": {
                role: {"examples": []}
                for role in (
                    worker.FIT_ROLE,
                    worker.DISCOVERY_ROLE,
                    worker.CONFIRMATION_ROLE,
                )
            }
        },
        fresh_receipt={},
        opaque_panel_receipt=object(),
        safe_payload_without_sha256=lambda: {"schema": "base.test"},
    )
    outer = SimpleNamespace(
        v1_preflight=base,
        receipt_sha256=_sha("v2-preflight"),
        safe_payload_without_sha256=lambda: {"schema": "v2.test"},
    )
    monkeypatch.setattr(
        worker,
        "static_preflight_v2",
        lambda **_kwargs: (events.append("preflight"), outer)[1],
    )
    monkeypatch.setattr(
        worker,
        "revalidate_v2_authorization",
        lambda _receipt: events.append("revalidate"),
    )

    class FakePanel:
        def __init__(self, **_kwargs: object) -> None:
            self.opened_roles: tuple[str, ...] = ()

        def open_role(self, role: str):
            events.append(f"open:{role}")
            self.opened_roles = (*self.opened_roles, role)
            return {"examples": [{}]}

        def freeze_current(self, kind: str, sha: str) -> None:
            events.append(f"freeze:{kind}:{sha[:4]}")

    monkeypatch.setattr(panels, "SequentialPanelSource", FakePanel)
    mode_ids = ("mode.a", "mode.b")
    lineage = tuple(SimpleNamespace(mode_id=value) for value in mode_ids)
    qualification = SimpleNamespace(
        qualification_receipt_sha256=_sha("qualification"),
        source_model_fingerprint_after_compilation_sha256=_sha("model"),
        source_execution_fingerprint_after_compilation_sha256=_sha("execution"),
        compiled_execution_fingerprint_sha256=_sha("compiled"),
        tied_readout_binding_receipt_sha256=_sha("readout"),
        mode_lineage=lineage,
        payload_without_sha256=lambda: {"schema": "qualification.test"},
    )
    session = _sha("session")

    class FakeRuntimeReceipt:
        runtime_qualification_receipt_sha256 = (
            qualification.qualification_receipt_sha256
        )
        runtime_session_binding_sha256 = session
        mode_lineage = lineage
        runtime_receipt_sha256 = _sha("runtime-receipt")
        observed_role_batch_counts = tuple(
            (role, 45)
            for role in (
                worker.FIT_ROLE,
                worker.DISCOVERY_ROLE,
                worker.CONFIRMATION_ROLE,
            )
        )
        observed_chart_capture_backbone_forward_count = 135
        observed_chart_capture_terminal_cotangent_VJP_count = 135

        def payload_without_sha256(self):
            return {"schema": "runtime.test"}

    monkeypatch.setattr(worker.v1, "RuntimeExecutionReceipt", FakeRuntimeReceipt)

    class FakeRuntime:
        mode_lineage = lineage
        runtime_session_binding_sha256 = session

        def __init__(self) -> None:
            self.close_failures = close_failures
            self.close_count = 0

        def runtime_qualification_receipt(self):
            return qualification

        def run_dense_lexical_parity_smoke(self, **_kwargs: object) -> str:
            events.append("parity")
            return _sha("parity")

        def runtime_execution_receipt(self):
            return FakeRuntimeReceipt()

        def close(self) -> None:
            self.close_count += 1
            events.append("close")
            if self.close_failures:
                self.close_failures -= 1
                raise RuntimeError("injected close failure")

    runtime = FakeRuntime()
    monkeypatch.setattr(worker.v1, "validate_runtime_qualification", lambda **_: None)
    monkeypatch.setattr(
        worker.v1, "derive_runtime_session_binding_sha256", lambda **_: session
    )
    monkeypatch.setattr(
        worker.v1,
        "build_core_mode_specs",
        lambda **_: tuple(SimpleNamespace(mode_id=value) for value in mode_ids),
    )

    def stage(role: str, ordinal: int):
        return SimpleNamespace(
            stage_kind=f"stage-{role}",
            stage_receipt_sha256=_sha(f"stage:{ordinal}"),
            payload_without_sha256=lambda: {"role": role},
        )

    def capture(**kwargs: object):
        role = kwargs["role"]
        events.append(f"capture:{role}")
        return SimpleNamespace(
            receipt=SimpleNamespace(
                role=role,
                batch_count=45,
                runtime_session_binding_sha256=session,
                receipt_sha256=_sha(f"capture:{role}"),
                payload_without_sha256=lambda: {"role": role},
            )
        )

    monkeypatch.setattr(worker.v1, "capture_role_to_raw_spools", capture)

    selection = SimpleNamespace(
        selected_chart_count=3,
        selected_source_feature_width=51,
        claim_tier="selected_shared_rank16_chart_transport_geometry",
        state_dict=lambda: {"schema": "selection.test"},
    )
    atlas_path = temporary / "fake-atlas.bundle"
    atlas_path.write_bytes(b"atlas")

    def fit(**kwargs: object):
        events.append("fit:A")
        if A_terminal is not None:
            raise A_terminal
        destination = Path(kwargs["destination_directory"])
        (destination / "applied").mkdir()
        return SimpleNamespace(
            stage_receipt=stage(worker.FIT_ROLE, 1),
            selection_receipt=selection,
            atlas=SimpleNamespace(
                atlas_sha256=_sha("atlas"),
                config=SimpleNamespace(tangent_rank=16),
            ),
            atlas_artifact_path=str(atlas_path),
            atlas_artifact_file_sha256=_sha("atlas-file"),
            mode_lineage_catalog_sha256=_sha("mode-lineage"),
            streaming_kernel_receipt_sha256=_sha("streaming"),
            transport_artifacts=(),
            transport_catalog_sha256=_sha("transports"),
            resource_observations=(),
            preselection_K8_diagnosis=SimpleNamespace(
                state_dict=lambda: {"schema": "K8.test"}
            ),
            K8_agreement=SimpleNamespace(
                state_dict=lambda: {"schema": "agreement.test"}
            ),
            selected_resource_receipt={"receipt_sha256": _sha("resource")},
        )

    monkeypatch.setattr(worker, "select_fit_A_chart_atlas_and_transports_v2", fit)
    monkeypatch.setattr(worker.v1, "evaluate_heldout_role_transports", lambda **_: object())
    monkeypatch.setattr(worker.v1, "build_role_graph_inputs", lambda **_: object())
    phase_ordinals = {worker.DISCOVERY_ROLE: 2, worker.CONFIRMATION_ROLE: 3}

    def graph_phase(**kwargs: object):
        role = kwargs["captured"].receipt.role
        return SimpleNamespace(
            stage_receipt=stage(role, phase_ordinals[role]),
            primary_receipt=SimpleNamespace(analysis={"mode_axis": list(mode_ids)}),
            candidate_or_graph_catalog_sha256=_sha(f"graph:{role}"),
            confirmed_edges=(("mode.a", "mode.b"),) if role == worker.CONFIRMATION_ROLE else (),
            role=role,
        )

    monkeypatch.setattr(worker.v1, "build_role_graph_phase", graph_phase)
    monkeypatch.setattr(
        worker.v1,
        "_role_phase_safe_payload",
        lambda phase: {"role": phase.role},
    )
    monkeypatch.setattr(
        worker.v1,
        "observe_phase_resources",
        lambda *_args, phase_id, **_kwargs: SimpleNamespace(
            receipt_sha256=_sha(f"resource:{phase_id}"),
            payload_without_sha256=lambda: {"phase_id": phase_id},
        ),
    )
    ledger = SimpleNamespace(
        total_chart_forward_count=135,
        total_summed_shifted_nll_backward_count=135,
        ledger_sha256=_sha("ledger"),
        payload_without_sha256=lambda: {"schema": "ledger.test"},
    )
    monkeypatch.setattr(worker.v1, "build_capture_call_ledger", lambda: ledger)
    monkeypatch.setattr(worker.v1, "render_neighbor_svg", lambda **_: b"<svg/>")
    monkeypatch.setattr(report_io, "seal_v2_experiment_report", lambda payload: payload, raising=False)
    publication = SimpleNamespace(
        atlas_path=str(output / "result.atlas"),
        report_path=str(output / "result.json"),
        svg_path=str(output / "result.svg"),
        report_sha256=_sha("report"),
    )
    monkeypatch.setattr(
        report_io,
        "publish_v2_experiment_outputs",
        lambda **_: (events.append("publish"), publication)[1],
        raising=False,
    )
    monkeypatch.setattr(
        report_io,
        "finalize_published_v2_experiment",
        lambda _receipt: events.append("finalize"),
        raising=False,
    )
    monkeypatch.setattr(
        report_io,
        "rollback_published_v2_experiment",
        lambda _receipt: events.append("rollback"),
        raising=False,
    )
    return events, binding, runtime, output, temporary


@pytest.mark.parametrize("occupied", ["report", "svg", "atlas"])
def test_run_rejects_preexisting_output_before_preflight_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    occupied: str,
) -> None:
    destinations = {
        "report": tmp_path / "result.json",
        "svg": tmp_path / "result.svg",
        "atlas": tmp_path / "result.atlas",
    }
    destinations[occupied].write_bytes(b"occupied")
    monkeypatch.setattr(
        worker,
        "static_preflight_v2",
        lambda **_kwargs: pytest.fail("preflight ran with occupied output"),
    )
    with pytest.raises(FileExistsError, match="already exists before preflight"):
        worker.run_chart_native_neighbor_v2_experiment(
            binding=SimpleNamespace(test_only_allow_nonproduction_shape=True),
            runtime_factory=lambda _preflight: pytest.fail(
                "runtime ran with occupied output"
            ),
            report_path=destinations["report"],
            svg_path=destinations["svg"],
            atlas_path=destinations["atlas"],
        )


def test_fake_end_to_end_orchestration_opens_only_A_then_B_then_C(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events, binding, runtime, output, temporary = _fake_orchestration(
        monkeypatch, tmp_path
    )
    result = worker.run_chart_native_neighbor_v2_experiment(
        binding=binding,
        runtime_factory=lambda _preflight: runtime,
        report_path=output / "result.json",
        svg_path=output / "result.svg",
        atlas_path=output / "result.atlas",
    )
    assert [event for event in events if event.startswith("open:")] == [
        f"open:{worker.FIT_ROLE}",
        f"open:{worker.DISCOVERY_ROLE}",
        f"open:{worker.CONFIRMATION_ROLE}",
    ]
    assert events.index("fit:A") < events.index(f"open:{worker.DISCOVERY_ROLE}")
    C_open = events.index(f"open:{worker.CONFIRMATION_ROLE}")
    assert events[C_open - 1] == "revalidate"
    assert events.index("close") < events.index("publish")
    assert events.index("publish") < events.index("finalize")
    assert runtime.close_count == 1
    assert result.selected_chart_count == 3
    assert not (temporary / worker._WORKSPACE_NAME).exists()


def test_fake_no_admissible_cleans_and_never_opens_B_or_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_admissible_terminal_evidence,
) -> None:
    selection, agreement = no_admissible_terminal_evidence
    terminal = worker.NoAdmissibleAtlasError(
        selection_receipt=selection,
        K8_agreement=agreement,
    )
    events, binding, runtime, output, temporary = _fake_orchestration(
        monkeypatch, tmp_path, A_terminal=terminal
    )
    with pytest.raises(worker.NoAdmissibleAtlasError):
        worker.run_chart_native_neighbor_v2_experiment(
            binding=binding,
            runtime_factory=lambda _preflight: runtime,
            report_path=output / "result.json",
            svg_path=output / "result.svg",
            atlas_path=output / "result.atlas",
        )
    assert f"open:{worker.DISCOVERY_ROLE}" not in events
    assert "publish" not in events
    assert runtime.close_count == 1
    assert not (temporary / worker._WORKSPACE_NAME).exists()


def test_fake_close_failure_is_retried_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events, binding, runtime, output, temporary = _fake_orchestration(
        monkeypatch, tmp_path, close_failures=1
    )
    with pytest.raises(RuntimeError, match="injected close failure"):
        worker.run_chart_native_neighbor_v2_experiment(
            binding=binding,
            runtime_factory=lambda _preflight: runtime,
            report_path=output / "result.json",
            svg_path=output / "result.svg",
            atlas_path=output / "result.atlas",
        )
    assert runtime.close_count == 2
    assert "publish" not in events
    assert not (temporary / worker._WORKSPACE_NAME).exists()


def test_failure_cleanup_attempts_rollback_workspace_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_v2_report as report_io

    events, binding, runtime, output, temporary = _fake_orchestration(
        monkeypatch,
        tmp_path,
    )
    workspace = temporary / worker._WORKSPACE_NAME
    real_safe_remove = worker.v1._safe_remove_tree
    workspace_attempts = 0

    def fail_workspace(path: Path | str, *, allowed_parent: Path | str) -> None:
        nonlocal workspace_attempts
        if Path(path) == workspace:
            workspace_attempts += 1
            if workspace_attempts == 1:
                raise RuntimeError("primary workspace cleanup failure")
            raise OSError("retry workspace cleanup failure")
        real_safe_remove(path, allowed_parent=allowed_parent)

    rollback_attempts = 0

    def fail_rollback(_publication: object) -> None:
        nonlocal rollback_attempts
        rollback_attempts += 1
        raise PermissionError("injected rollback cleanup failure")

    monkeypatch.setattr(worker.v1, "_safe_remove_tree", fail_workspace)
    monkeypatch.setattr(
        report_io,
        "rollback_published_v2_experiment",
        fail_rollback,
    )
    with pytest.raises(
        RuntimeError,
        match="primary workspace cleanup failure",
    ) as caught:
        worker.run_chart_native_neighbor_v2_experiment(
            binding=binding,
            runtime_factory=lambda _preflight: runtime,
            report_path=output / "result.json",
            svg_path=output / "result.svg",
            atlas_path=output / "result.atlas",
        )
    assert "publish" in events
    assert rollback_attempts == 1
    assert workspace_attempts == 2
    assert runtime.close_count == 1
    notes = getattr(caught.value, "__notes__", ())
    assert any("published v2 output rollback also failed" in note for note in notes)
    assert any("v2 workspace cleanup also failed" in note for note in notes)


def test_finalize_failure_keeps_issuance_authority_for_outer_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_v2_report as report_io

    events, binding, runtime, output, temporary = _fake_orchestration(
        monkeypatch,
        tmp_path,
    )

    def fail_finalize(_publication: object) -> None:
        events.append("finalize")
        raise RuntimeError("injected finalize failure")

    monkeypatch.setattr(
        report_io,
        "finalize_published_v2_experiment",
        fail_finalize,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="injected finalize failure"):
        worker.run_chart_native_neighbor_v2_experiment(
            binding=binding,
            runtime_factory=lambda _preflight: runtime,
            report_path=output / "result.json",
            svg_path=output / "result.svg",
            atlas_path=output / "result.atlas",
        )
    assert events.index("publish") < events.index("finalize") < events.index(
        "rollback"
    )
    assert runtime.close_count == 1
    assert not (temporary / worker._WORKSPACE_NAME).exists()


def test_failure_cleanup_preserves_first_close_error_when_retry_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events, binding, runtime, output, temporary = _fake_orchestration(
        monkeypatch,
        tmp_path,
        close_failures=2,
    )
    with pytest.raises(RuntimeError, match="injected close failure") as caught:
        worker.run_chart_native_neighbor_v2_experiment(
            binding=binding,
            runtime_factory=lambda _preflight: runtime,
            report_path=output / "result.json",
            svg_path=output / "result.svg",
            atlas_path=output / "result.atlas",
        )
    assert runtime.close_count == 2
    assert "publish" not in events
    assert not (temporary / worker._WORKSPACE_NAME).exists()
    assert any(
        "v2 runtime close also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
