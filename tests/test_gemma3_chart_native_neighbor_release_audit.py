from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from fisher_graph import gemma3_chart_native_neighbor_experiment as worker
from fisher_graph import gemma3_chart_native_neighbor_fresh_panels as fresh


def _sha(label: str) -> str:
    return worker.hashlib.sha256(label.encode("utf-8")).hexdigest()


def _first_party_module_path(
    package_root: Path,
    module_name: str,
) -> Path | None:
    if module_name == "fisher_graph":
        candidate = package_root / "__init__.py"
    elif module_name.startswith("fisher_graph."):
        relative = Path(*module_name.split(".")[1:])
        candidate = (package_root / relative).with_suffix(".py")
        if not candidate.is_file():
            candidate = package_root / relative / "__init__.py"
    else:
        return None
    return candidate if candidate.is_file() else None


def _conservative_first_party_source_closure(
    repository_root: Path,
) -> frozenset[str]:
    """Derive the experiment closure without expanding eager root exports."""

    package_root = repository_root / "src/fisher_graph"
    pending = ["fisher_graph.gemma3_chart_native_neighbor_runtime"]
    observed: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in observed:
            continue
        source = _first_party_module_path(package_root, module_name)
        if source is None:
            continue
        observed.add(module_name)
        # The root file is itself part of the declared boundary, while its
        # broad eager re-export fanout is explicitly outside the experiment's
        # conservative functional-source claim.
        if module_name == "fisher_graph":
            continue
        package_name = (
            module_name
            if source.name == "__init__.py"
            else module_name.rsplit(".", 1)[0]
        )
        parsed = ast.parse(source.read_bytes(), filename=str(source))
        imported: set[str] = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name
                    for alias in node.names
                    if _first_party_module_path(package_root, alias.name)
                    is not None
                )
                continue
            if isinstance(node, ast.ImportFrom):
                try:
                    base = (
                        importlib.util.resolve_name(
                            "." * node.level + (node.module or ""),
                            package_name,
                        )
                        if node.level
                        else (node.module or "")
                    )
                except (ImportError, ValueError):
                    continue
                candidates = [
                    base,
                    *(
                        f"{base}.{alias.name}" if base else alias.name
                        for alias in node.names
                    ),
                ]
                imported.update(
                    candidate
                    for candidate in candidates
                    if _first_party_module_path(package_root, candidate)
                    is not None
                )
                continue
            if not isinstance(node, ast.Call):
                continue
            callback = node.func
            literal_import = (
                isinstance(callback, ast.Attribute)
                and isinstance(callback.value, ast.Name)
                and callback.value.id == "importlib"
                and callback.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            )
            if not literal_import:
                continue
            target = node.args[0].value
            try:
                resolved = (
                    importlib.util.resolve_name(target, package_name)
                    if target.startswith(".")
                    else target
                )
            except (ImportError, ValueError):
                continue
            if _first_party_module_path(package_root, resolved) is not None:
                imported.add(resolved)
        pending.extend(imported - observed)
    return frozenset(
        str(
            _first_party_module_path(package_root, module_name).relative_to(
                repository_root
            )
        )
        for module_name in observed
    )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


class _SafeReceipt:
    def __init__(self, label: str, **attributes: object) -> None:
        self.receipt_sha256 = _sha(label)
        self.__dict__.update(attributes)

    def payload_without_sha256(self) -> dict[str, object]:
        return {"summary_receipt_kind": "model_free_release_audit"}


class _SafeStage:
    def __init__(self, role: str, stage_kind: str) -> None:
        self.role = role
        self.stage_kind = stage_kind
        self.stage_receipt_sha256 = _sha(f"stage:{role}")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "role": self.role,
            "stage_kind": self.stage_kind,
            "summary_only": True,
        }


def _runtime_lineage() -> tuple[worker.RuntimeModeLineage, ...]:
    mode_ids = (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )
    return tuple(
        worker.RuntimeModeLineage(
            mode_id=mode_id,
            state_fingerprint_sha256=_sha(f"state:{mode_id}"),
            site_placement_sha256=_sha(f"site:{mode_id}"),
            comparison_adapter_receipt_sha256=_sha(f"adapter:{mode_id}"),
        )
        for mode_id in mode_ids
    )


def _qualification(
    lineage: tuple[worker.RuntimeModeLineage, ...],
) -> worker.RuntimeQualificationReceipt:
    return worker.build_runtime_qualification_receipt(
        schema=(
            "fisher_graph.gemma3_chart_native_neighbor."
            "runtime_qualification.v1"
        ),
        format_version=1,
        model_id="google/gemma-3-270m",
        model_revision="release-audit-model-revision",
        source_model_fingerprint_before_compilation_sha256=_sha("source-model"),
        source_model_fingerprint_after_compilation_sha256=_sha("source-model"),
        source_execution_fingerprint_before_compilation_sha256=_sha(
            "source-execution"
        ),
        source_execution_fingerprint_after_compilation_sha256=_sha(
            "source-execution"
        ),
        source_model_mutated_during_compilation=False,
        source_parameter_count=268_098_176,
        hidden_width=640,
        compiled_execution_fingerprint_sha256=_sha("compiled-execution"),
        compiler_receipt_sha256=_sha("compiler"),
        tied_readout_binding_receipt_sha256=_sha("tied-readout"),
        tied_lexical_lineage_sha256=_sha("tied-lexical"),
        final_logit_softcap=4.0,
        mode_lineage=lineage,
        mode_lineage_catalog_sha256=(
            worker.runtime_mode_lineage_catalog_sha256(lineage)
        ),
        lexical_kernel_receipt_sha256=_sha("lexical-kernel"),
        streaming_kernel_receipt_sha256=_sha("streaming-kernel"),
        factorized_route_kernel_receipt_sha256=_sha("route-kernel"),
        contains_prompt_text=False,
        contains_token_ids=False,
        contains_raw_activations=False,
        contains_raw_gradients=False,
        contains_logits=False,
    )


def _execution_receipt(
    qualification: worker.RuntimeQualificationReceipt,
    session_sha256: str,
) -> worker.RuntimeExecutionReceipt:
    return worker.build_runtime_execution_receipt(
        model_id=qualification.model_id,
        model_revision=qualification.model_revision,
        runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        runtime_session_binding_sha256=session_sha256,
        source_model_fingerprint_before_sha256=(
            qualification.source_model_fingerprint_before_compilation_sha256
        ),
        source_model_fingerprint_after_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_before_sha256=(
            qualification.source_execution_fingerprint_before_compilation_sha256
        ),
        source_execution_fingerprint_after_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        source_model_mutated=False,
        source_parameter_count=268_098_176,
        hidden_width=640,
        mode_lineage=qualification.mode_lineage,
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        tied_lexical_lineage_sha256=qualification.tied_lexical_lineage_sha256,
        final_logit_softcap=qualification.final_logit_softcap,
        compiler_receipt_sha256=qualification.compiler_receipt_sha256,
        lexical_kernel_receipt_sha256=(
            qualification.lexical_kernel_receipt_sha256
        ),
        streaming_kernel_receipt_sha256=(
            qualification.streaming_kernel_receipt_sha256
        ),
        factorized_route_kernel_receipt_sha256=(
            qualification.factorized_route_kernel_receipt_sha256
        ),
        expected_chart_role_count=3,
        expected_batches_per_chart_role=45,
        expected_chart_capture_backbone_forward_count=135,
        expected_chart_capture_terminal_cotangent_VJP_count=135,
        expected_dense_parity_backbone_forward_count=1,
        expected_dense_parity_terminal_cotangent_VJP_count=1,
        expected_total_backbone_forward_count=136,
        expected_total_terminal_cotangent_VJP_count=136,
        expected_gradient_target_count_per_VJP=92,
        observed_role_batch_counts=(
            (worker.FIT_ROLE, 45),
            (worker.DISCOVERY_ROLE, 45),
            (worker.CONFIRMATION_ROLE, 45),
        ),
        observed_chart_capture_backbone_forward_count=135,
        observed_chart_capture_terminal_cotangent_VJP_count=135,
        observed_dense_parity_backbone_forward_count=1,
        observed_dense_parity_terminal_cotangent_VJP_count=1,
        observed_total_backbone_forward_count=136,
        observed_total_terminal_cotangent_VJP_count=136,
        observed_dense_lexical_parity_smoke_count=1,
        observed_captured_example_count=144,
        observed_supplemental_route_receipt_count=3 * 18 * 48,
        contains_prompt_text=False,
        contains_token_ids=False,
        contains_raw_activations=False,
        contains_raw_gradients=False,
        contains_logits=False,
    )


def test_model_free_orchestrator_uses_real_A_B_C_firewall_and_commits_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    manifest = fresh.load_manifest(repository_root / fresh.DEFAULT_MANIFEST)
    receipt = fresh.load_receipt(repository_root / fresh.DEFAULT_RECEIPT)
    opaque = fresh.opaque_preflight(
        repository_root=repository_root,
        manifest=manifest,
        receipt=receipt,
    )
    temp_directory = tmp_path / "temp"
    output_directory = tmp_path / "out"
    temp_directory.mkdir()
    output_directory.mkdir()
    binding = SimpleNamespace(
        repository_root=repository_root,
        temp_directory=temp_directory,
        output_directory=output_directory,
        test_only_allow_nonproduction_shape=True,
    )
    preflight_sha = _sha("preflight")
    preflight = SimpleNamespace(
        binding=binding,
        fresh_manifest=manifest,
        fresh_receipt=receipt,
        opaque_panel_receipt=opaque,
        receipt_sha256=preflight_sha,
        validate_receipt=lambda: None,
        safe_payload_without_sha256=lambda: {
            "summary_preflight_kind": "model_free_release_audit"
        },
    )
    events: list[str] = []

    real_panel_source = fresh.SequentialPanelSource

    class RecordingPanelSource(real_panel_source):
        def open_role(self, role: str) -> dict[str, object]:
            events.append(f"open:{role}")
            return super().open_role(role)

        def freeze_current(self, stage_kind: str, stage_sha256: str) -> None:
            events.append(f"freeze:{self._current_role}:{stage_kind}")
            super().freeze_current(stage_kind, stage_sha256)

    monkeypatch.setattr(fresh, "SequentialPanelSource", RecordingPanelSource)
    monkeypatch.setattr(
        worker,
        "static_preflight",
        lambda _binding: events.append("static-preflight") or preflight,
    )
    monkeypatch.setattr(
        worker,
        "validate_runtime_qualification",
        lambda **_kwargs: events.append("qualification:validated"),
    )
    monkeypatch.setattr(
        worker,
        "build_core_mode_specs",
        lambda **_kwargs: (SimpleNamespace(mode_id="model-free-mode"),),
    )

    lineage = _runtime_lineage()
    qualification = _qualification(lineage)
    session_sha = worker.derive_runtime_session_binding_sha256(
        preflight_receipt_sha256=preflight_sha,
        runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        mode_lineage=lineage,
    )
    execution = _execution_receipt(qualification, session_sha)

    class FakeRuntime:
        mode_lineage = lineage
        runtime_session_binding_sha256 = session_sha
        hidden_width = 640
        closed = False

        def runtime_qualification_receipt(self) -> worker.RuntimeQualificationReceipt:
            events.append("qualification:read")
            return qualification

        def run_dense_lexical_parity_smoke(self, **_kwargs: object) -> str:
            events.append("dense-parity:A")
            return _sha("dense-parity")

        def runtime_execution_receipt(self) -> worker.RuntimeExecutionReceipt:
            events.append("ledger:136/136")
            return execution

        def close(self) -> None:
            events.append("runtime:close")
            self.closed = True

    runtime = FakeRuntime()

    def runtime_factory(_preflight: object) -> FakeRuntime:
        events.append("runtime:create")
        return runtime

    def capture_role(**values: object) -> SimpleNamespace:
        role = str(values["role"])
        events.append(f"capture:{role}")
        capture_receipt = _SafeReceipt(
            f"capture:{role}",
            role=role,
            batch_count=45,
            runtime_session_binding_sha256=session_sha,
        )
        return SimpleNamespace(receipt=capture_receipt)

    monkeypatch.setattr(worker, "capture_role_to_raw_spools", capture_role)

    A_stage = _SafeStage(worker.FIT_ROLE, "chart_atlas_transport_fit")

    def fit_A(**values: object) -> SimpleNamespace:
        events.append("phase:A-fit")
        destination = Path(values["destination_directory"])
        (destination / "applied").mkdir()
        return SimpleNamespace(
            stage_receipt=A_stage,
            atlas=SimpleNamespace(atlas_sha256=_sha("atlas")),
            atlas_artifact_file_sha256=_sha("atlas-file"),
            atlas_artifact_path=destination / "atlas.bundle",
            resource_observations=(),
            streaming_kernel_receipt_sha256=_sha("streaming-kernel"),
            mode_lineage_catalog_sha256=_sha("mode-lineage-catalog"),
            transport_catalog_sha256=_sha("transport-catalog"),
            transport_artifacts=(),
        )

    monkeypatch.setattr(worker, "fit_A_chart_atlas_and_transports", fit_A)

    def evaluate(**values: object) -> SimpleNamespace:
        role = values["captured"].receipt.role if hasattr(
            values["captured"].receipt, "role"
        ) else "heldout"
        events.append(f"evaluate:{role}")
        return SimpleNamespace(role=role)

    monkeypatch.setattr(worker, "evaluate_heldout_role_transports", evaluate)
    monkeypatch.setattr(
        worker,
        "build_role_graph_inputs",
        lambda **values: SimpleNamespace(role=values["evaluation"].role),
    )

    mode_specs = tuple(
        SimpleNamespace(mode_id=row.mode_id)
        for row in sorted(lineage, key=lambda item: item.mode_id)
    )

    def graph_phase(**values: object) -> SimpleNamespace:
        role = values["captured"].receipt.role
        events.append(f"graph:{role}")
        if role == worker.DISCOVERY_ROLE:
            stage_kind = "neighbor_candidate_catalog"
            frozen_edges = ((lineage[0].mode_id, lineage[1].mode_id),)
            confirmed_edges: tuple[tuple[str, str], ...] = ()
        else:
            assert values["frozen_B_phase"] is not None
            stage_kind = "chart_neighbor_graph_and_gates"
            frozen_edges = values["frozen_B_phase"].frozen_candidate_edges
            confirmed_edges = frozen_edges
        return SimpleNamespace(
            role=role,
            all_eight_diagnostic_source=SimpleNamespace(
                mode_specs=mode_specs,
                canonical_payload=lambda: {"summary_only": True},
            ),
            candidate_or_graph_catalog_sha256=_sha(f"catalog:{role}"),
            capture_receipt=values["captured"].receipt,
            confirmation_core_edges=confirmed_edges,
            confirmed_edges=confirmed_edges,
            diagnostic_receipts=(),
            exploratory_C_edges=(),
            frozen_candidate_edges=frozen_edges,
            hard_long_veto_edges=confirmed_edges,
                primary_receipt=SimpleNamespace(
                    analysis={
                        "mode_axis": sorted(row.mode_id for row in lineage)
                    },
                input_sha256=_sha(f"input:{role}"),
                receipt_sha256=_sha(f"primary:{role}"),
            ),
            evaluation_catalog_sha256=_sha(f"evaluation:{role}"),
            evaluation_resource_observations=(),
            stage_receipt=_SafeStage(role, stage_kind),
        )

    monkeypatch.setattr(worker, "build_role_graph_phase", graph_phase)

    resource_ordinal = 0

    def observe_resources(_binding: object, **values: object) -> _SafeReceipt:
        nonlocal resource_ordinal
        resource_ordinal += 1
        return _SafeReceipt(
            f"resource:{resource_ordinal}",
            phase_id=values["phase_id"],
            payload_without_sha256=lambda: {},
        )

    monkeypatch.setattr(worker, "observe_phase_resources", observe_resources)

    report_path = output_directory / "report.json"
    svg_path = output_directory / "graph.svg"
    atlas_path = output_directory / worker.DEFAULT_ATLAS_OUTPUT.name

    def publish(**values: object) -> worker.PublishedExperimentReceipt:
        events.append("publication:report-last")
        assert runtime.closed
        report = values["report"]
        assert report["capture_call_ledger"]["observed_live_capture_count"] == 135
        assert report["runtime_execution_receipt"][
            "observed_total_backbone_forward_count"
        ] == 136
        assert report["runtime_execution_receipt"][
            "observed_total_terminal_cotangent_VJP_count"
        ] == 136
        return worker.PublishedExperimentReceipt(
            atlas_path=str(atlas_path),
            report_path=str(report_path),
            svg_path=str(svg_path),
            atlas_file_sha256=_sha("atlas-file"),
            atlas_sha256=_sha("atlas"),
            report_file_sha256=_sha("report-file"),
            svg_file_sha256=_sha("svg-file"),
            report_sha256=report["report_sha256"],
            publication_receipt_sha256=_sha("publication"),
        )

    monkeypatch.setattr(worker, "publish_experiment_outputs", publish)

    result = worker.run_chart_native_neighbor_experiment(
        binding=binding,
        runtime_factory=runtime_factory,
        report_path=report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    )

    assert result.temp_workspace_removed is True
    assert not (temp_directory / "chart-native-neighbor-work-v1").exists()
    assert [event for event in events if event.startswith("open:")] == [
        f"open:{worker.FIT_ROLE}",
        f"open:{worker.DISCOVERY_ROLE}",
        f"open:{worker.CONFIRMATION_ROLE}",
    ]
    assert all(worker.RESPONSE_ROLE not in event for event in events)
    assert all(worker.RESERVE_ROLE not in event for event in events)
    assert events.index("static-preflight") < events.index("runtime:create")
    assert events.index("runtime:create") < events.index(f"open:{worker.FIT_ROLE}")
    assert events.index("qualification:validated") < events.index(
        f"open:{worker.FIT_ROLE}"
    )
    assert events.index("runtime:close") < events.index(
        "publication:report-last"
    )
    assert "ledger:136/136" in events


def test_default_model_factory_hook_is_reached_only_after_static_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    preflight = SimpleNamespace(
        binding=SimpleNamespace(
            storage_budget=SimpleNamespace(maximum_resident_memory_bytes=1)
        )
    )

    def preflight_first(_binding: object) -> object:
        events.append("static-preflight")
        return preflight

    def stop_at_default_factory(
        observed_preflight: object,
        *,
        cache_dir: object,
    ) -> object:
        assert observed_preflight is preflight
        assert cache_dir is None
        events.append("default-model-factory")
        raise RuntimeError("model-free stop before runtime import")

    monkeypatch.setattr(worker, "static_preflight", preflight_first)
    monkeypatch.setattr(
        worker,
        "_build_default_runtime_after_preflight",
        stop_at_default_factory,
    )
    with pytest.raises(RuntimeError, match="model-free stop"):
        worker.qualify_pinned_runtime(
            binding=SimpleNamespace(),
            destination=tmp_path / "qualification.json",
        )
    assert events == ["static-preflight", "default-model-factory"]


def test_conservative_functional_source_catalog_is_exact_transitive_closure(
    tmp_path: Path,
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_setup as setup_module

    repository_root = Path(__file__).resolve().parents[1]
    closure = _conservative_first_party_source_closure(repository_root)
    trust_anchor = (
        "src/fisher_graph/gemma3_chart_native_neighbor_setup.py"
    )
    root_boundary = "src/fisher_graph/__init__.py"
    predecessor_only = (
        "src/fisher_graph/gemma3_unified_event_chart_sidecar.py"
    )
    catalog = setup_module.PINNED_WORKER_IMPLEMENTATION_SOURCES
    pinned_paths = frozenset(str(receipt["path"]) for receipt in catalog.values())

    assert len(closure) == 61
    assert trust_anchor in closure
    assert len(catalog) == len(pinned_paths) == 60
    assert pinned_paths == closure - {trust_anchor}
    assert root_boundary in pinned_paths
    assert predecessor_only not in closure
    assert predecessor_only not in pinned_paths
    assert trust_anchor not in pinned_paths
    mismatched_live_hashes = {
        name: {
            "expected": str(receipt["file_sha256"]),
            "observed": hashlib.sha256(
                (repository_root / receipt["path"]).read_bytes()
            ).hexdigest(),
        }
        for name, receipt in catalog.items()
        if hashlib.sha256(
            (repository_root / receipt["path"]).read_bytes()
        ).hexdigest()
        != receipt["file_sha256"]
    }
    assert mismatched_live_hashes == {}

    binding = worker.build_default_execution_binding(
        repository_root=repository_root,
        output_directory=tmp_path / "out",
        temp_directory=tmp_path / "temp",
    )
    assert tuple(pin.name for pin in binding.source_pins) == tuple(
        sorted(catalog)
    )
    assert frozenset(pin.path for pin in binding.source_pins) == pinned_paths


def test_nonworker_closure_byte_mutation_fails_preflight_and_strict_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_setup as setup_module

    repository_root = Path(__file__).resolve().parents[1]
    catalog = setup_module.PINNED_WORKER_IMPLEMENTATION_SOURCES
    for receipt in catalog.values():
        relative = Path(str(receipt["path"]))
        _copy_file(repository_root / relative, tmp_path / relative)

    manifest = fresh.load_manifest(repository_root / fresh.DEFAULT_MANIFEST)
    frozen_inputs = {
        Path(worker.DEFAULT_SETUP_PATH),
        Path(fresh.DEFAULT_MANIFEST),
        Path(fresh.DEFAULT_RECEIPT),
        *(
            Path(str(manifest["roles"][role]["path"]))
            for role in worker.ORDERED_ROLES
        ),
    }
    for relative in frozen_inputs:
        _copy_file(repository_root / relative, tmp_path / relative)

    output_directory = tmp_path / "out"
    temp_directory = tmp_path / "temp"
    output_directory.mkdir()
    temp_directory.mkdir()
    source_pins = tuple(
        worker.ExecutionSourcePin(
            name=name,
            path=str(receipt["path"]),
            file_sha256=str(receipt["file_sha256"]),
        )
        for name, receipt in sorted(catalog.items())
    )
    binding = worker.ChartNativeExecutionBinding(
        repository_root=tmp_path,
        setup_path=str(worker.DEFAULT_SETUP_PATH),
        source_pins=source_pins,
        schedule=worker.LengthBucketBatchSchedule(),
        storage_budget=worker.ExecutionStorageBudget(),
        output_directory=output_directory,
        temp_directory=temp_directory,
        test_only_allow_nonproduction_shape=True,
    )
    # Other collected model-free tests may already have imported the concrete
    # module.  Its ordering guard has separate coverage; isolate byte closure.
    monkeypatch.setattr(worker, "_FORBIDDEN_BEFORE_PREFLIGHT", ())
    preflight = worker.static_preflight(binding)
    report = {
        "preflight": {
            "setup_sha256": preflight.setup["setup_sha256"],
            "source_file_sha256s": dict(preflight.source_file_sha256s),
        }
    }
    monkeypatch.setattr(
        worker,
        "_load_frozen_runtime_qualification_from_setup",
        lambda **_kwargs: None,
    )
    worker._strict_rehash_frozen_repository_inputs(
        report=report,
        repository_root=tmp_path,
    )

    target = tmp_path / catalog["activations"]["path"]
    target.write_bytes(target.read_bytes() + b"\n# adversarial-byte-change\n")
    with pytest.raises(ValueError, match="source activations.*execution pin"):
        worker.static_preflight(binding)
    with pytest.raises(
        ValueError,
        match="implementation source activations drifted",
    ):
        worker._strict_rehash_frozen_repository_inputs(
            report=report,
            repository_root=tmp_path,
        )


def test_primary_exact7_row_shuffle_excludes_514_and_rejects_tamper() -> None:
    import torch

    from fisher_graph.shared_chart_energy_transport import (
        FIT_ROLE as SHARED_FIT_ROLE,
        ChartRowKey,
        RowShuffleCatalog,
        SharedChartNumericsConfig,
        SharedChartRowBatch,
        apply_shared_chart_atlas,
        balanced_atlas_row_weights,
        build_within_family_row_shuffle,
        evaluate_directed_chart_transport,
        fit_directed_chart_transport,
        fit_shared_chart_atlas,
    )

    repository_root = Path(__file__).resolve().parents[1]
    manifest = fresh.load_manifest(repository_root / fresh.DEFAULT_MANIFEST)
    primary_keys_by_role: dict[str, tuple[ChartRowKey, ...]] = {}
    for role in (worker.DISCOVERY_ROLE, worker.CONFIRMATION_ROLE):
        members = tuple(manifest["roles"][role]["ordered_members"])
        primary = tuple(
            member
            for member in members
            if member["eligible_for_pooled_point_topology"] is True
        )
        transition_ids = {
            str(member["example_id"])
            for member in members
            if member["target_token_count"] == 514
        }
        assert len(primary) == 6 * 7 == 42
        assert len(transition_ids) == 6
        assert all(member["target_token_count"] != 514 for member in primary)
        keys = tuple(
            ChartRowKey(
                str(member["family_id"]),
                str(member["example_id"]),
                query_position,
                query_position + 1,
            )
            for member in primary
            for query_position in range(
                int(member["prediction_query_position_start"]),
                int(member["prediction_query_position_stop_exclusive"]),
            )
        )
        assert len(keys) == 21_654
        shuffle = build_within_family_row_shuffle(keys, role=role)
        source_indices = shuffle.permutation.tolist()
        assert all(
            keys[target_index].example_id not in transition_ids
            and keys[source_index].example_id not in transition_ids
            and keys[target_index].family_id == keys[source_index].family_id
            and keys[target_index].example_id == keys[source_index].example_id
            and target_index != source_index
            for target_index, source_index in enumerate(source_indices)
        )
        groups = sorted({(key.family_id, key.example_id) for key in keys})
        assert len(groups) == 42
        for family_id, example_id in groups:
            positions = [
                index
                for index, key in enumerate(keys)
                if (key.family_id, key.example_id)
                == (family_id, example_id)
            ]
            local_ordinal = {
                absolute: ordinal for ordinal, absolute in enumerate(positions)
            }
            offsets = {
                (local_ordinal[source_indices[target]] - local_ordinal[target])
                % len(positions)
                for target in positions
            }
            assert len(offsets) == 1
            assert next(iter(offsets)) != 0
        primary_keys_by_role[role] = keys

    # Exercise the production evaluator's authenticated reconstruction on a
    # small slice of the actual exact-seven B row identity catalog.
    B_keys = primary_keys_by_role[worker.DISCOVERY_ROLE]
    first_examples = []
    for key in B_keys:
        identity = (key.family_id, key.example_id)
        if identity not in first_examples:
            first_examples.append(identity)
        if len(first_examples) == 2:
            break
    tiny_keys = tuple(
        key
        for key in B_keys
        if (key.family_id, key.example_id) in first_examples
        and key.prediction_query_position < 4
    )
    assert len(tiny_keys) == 8

    def batches(role: str, label: str) -> tuple[SharedChartRowBatch, ...]:
        result = []
        for mode_index in range(2):
            generator = torch.Generator().manual_seed(
                91_000 + sum(label.encode("utf-8")) + mode_index
            )
            context = torch.randn(
                len(tiny_keys),
                4,
                generator=generator,
                dtype=torch.float64,
            )
            context[: len(tiny_keys) // 2, 0] -= 3.0
            context[len(tiny_keys) // 2 :, 0] += 3.0
            comparison = 0.25 * context + float(mode_index + 1)
            score = torch.randn(
                len(tiny_keys),
                4,
                generator=generator,
                dtype=torch.float64,
            )
            result.append(
                SharedChartRowBatch(
                    role=role,
                    panel_binding_sha256=_sha(label),
                    mode_id=f"mode.{mode_index}",
                    row_keys=tiny_keys,
                    context_values=context,
                    comparison_values=comparison,
                    generalized_contributions=comparison * score,
                )
            )
        return tuple(result)

    fit_batches = batches(SHARED_FIT_ROLE, "exact7-fit")
    atlas = fit_shared_chart_atlas(
        fit_batches,
        config=SharedChartNumericsConfig(
            hidden_width=4,
            chart_count=2,
            tangent_rank=2,
        ),
    )
    fit_rows = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in fit_batches
    )
    transport = fit_directed_chart_transport(
        atlas,
        fit_rows[0],
        fit_rows[1],
        row_weights=balanced_atlas_row_weights(fit_batches)["mode.0"],
    )
    heldout_batches = batches(worker.DISCOVERY_ROLE, "exact7-heldout")
    heldout_rows = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout_batches
    )
    shuffle = build_within_family_row_shuffle(
        tiny_keys,
        role=worker.DISCOVERY_ROLE,
    )
    tampered_permutation = shuffle.permutation.clone()
    first = tampered_permutation[0].clone()
    tampered_permutation[0] = tampered_permutation[1]
    tampered_permutation[1] = first
    tampered = RowShuffleCatalog(
        role=shuffle.role,
        row_catalog_sha256=shuffle.row_catalog_sha256,
        seed=shuffle.seed,
        permutation=tampered_permutation,
        catalog_sha256=shuffle.catalog_sha256,
    )
    with pytest.raises(ValueError, match="deterministic within-example control"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_rows[0],
            heldout_rows[1],
            row_shuffle=tampered,
        )


def _real_parser_graph_input(
    family_example_ids: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    config: object | None = None,
) -> object:
    """Build a small numeric input over the real frozen B example catalog."""

    from fisher_graph.chart_energy_neighbor_graph import (
        DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL,
        TOKEN_LOCAL_NATIVE_ABI,
        ChartEnergyNeighborConfig,
        ChartEnergyNeighborInput,
        ChartExampleStats,
        ChartModeSpec,
        ChartWithinFamilyBootstrapConfig,
        DirectedChartTransportStats,
        NativeChartABI,
        canonical_carrier_comparison_abi,
        canonicalize_chart_energy_neighbor_input,
    )
    from fisher_graph.shared_chart_energy_transport import (
        ChartRowKey,
        build_within_example_row_shuffle,
    )

    manifest = fresh.load_manifest()
    role_manifest = manifest["roles"][worker.DISCOVERY_ROLE]
    panel_binding = worker.role_panel_binding_sha256(
        manifest,
        worker.DISCOVERY_ROLE,
    )
    row_contract_by_example: dict[str, tuple[str, int, str]] = {}
    all_role_row_keys: list[ChartRowKey] = []
    for member in role_manifest["ordered_members"]:
        family_id = str(member["family_id"])
        example_id = str(member["example_id"])
        row_count = int(member["prediction_row_count"])
        row_keys = tuple(
            ChartRowKey(family_id, example_id, query, query + 1)
            for query in range(row_count)
        )
        all_role_row_keys.extend(row_keys)
        row_contract_by_example[example_id] = (
            family_id,
            row_count,
            worker._example_row_catalog_sha256(
                role=worker.DISCOVERY_ROLE,
                row_keys=row_keys,
            ),
        )
    comparison = canonical_carrier_comparison_abi()
    native = NativeChartABI(
        abi_kind=TOKEN_LOCAL_NATIVE_ABI,
        abi_name="mlp.residual_h640.token_local.v1",
        abi_version=1,
        input_signature="release.audit.input",
        output_signature="release.audit.output",
        context_signature="release.audit.context",
    )
    mode_specs = tuple(
        ChartModeSpec(
            mode_id=f"mode.release.audit.{ordinal}",
            mode_kind="mlp",
            state_fingerprint_sha256=_sha(f"audit-state:{ordinal}"),
            event_id=f"release.audit.event.{ordinal}",
            event_index=ordinal,
            context_site_id=f"release.audit.context.{ordinal}",
            contribution_site_id=f"release.audit.contribution.{ordinal}",
            site_placement_sha256=_sha(f"audit-site:{ordinal}"),
            native_abi=native,
            comparison_abi=comparison,
            comparison_adapter_kind=(
                DIRECT_H640_MUTATION_RESIDUAL_ADAPTER
            ),
            comparison_adapter_receipt_sha256=_sha(
                f"audit-adapter:{ordinal}"
            ),
        )
        for ordinal in range(3)
    )
    chart_atlas = _sha("audit-chart-atlas")
    fit_catalog = _sha("audit-fit-catalog")
    split_receipt = _sha("audit-split-receipt")
    shuffle_catalog = build_within_example_row_shuffle(
        all_role_row_keys,
        role=worker.DISCOVERY_ROLE,
    ).catalog_sha256
    example_stats: list[ChartExampleStats] = []
    chart_by_key: dict[tuple[str, str, str], ChartExampleStats] = {}
    for family_id, example_ids in family_example_ids:
        for example_id in example_ids:
            (
                expected_family,
                row_count,
                row_catalog,
            ) = row_contract_by_example[example_id]
            assert expected_family == family_id
            for ordinal, mode in enumerate(mode_specs, start=1):
                energy = float(ordinal * ordinal)
                stats = ChartExampleStats(
                    family_id=family_id,
                    example_id=example_id,
                    mode_id=mode.mode_id,
                    comparison_abi_sha256=comparison.sha256,
                    chart_atlas_sha256=chart_atlas,
                    aligned_row_catalog_sha256=row_catalog,
                    chart_energy_vector_catalog_sha256=_sha(
                        f"audit-vectors:{family_id}:{example_id}:"
                        f"{mode.mode_id}"
                    ),
                    row_count=row_count,
                    occupied_row_count=row_count,
                    minimum_row_fisher_energy=0.0,
                    weighted_activation_energy=2.0 * energy,
                    weighted_chart_tangent_energy=0.75 * energy,
                    weighted_chart_orthogonal_energy=0.25 * energy,
                    weighted_fisher_energy=energy,
                )
                example_stats.append(stats)
                chart_by_key[(family_id, example_id, mode.mode_id)] = stats

    directed: list[DirectedChartTransportStats] = []
    for family_id, example_ids in family_example_ids:
        for example_id in example_ids:
            (
                expected_family,
                row_count,
                row_catalog,
            ) = row_contract_by_example[example_id]
            assert expected_family == family_id
            for source_ordinal, source in enumerate(mode_specs, start=1):
                for target_ordinal, target in enumerate(mode_specs, start=1):
                    if source.mode_id == target.mode_id:
                        continue
                    predicted = float(source_ordinal * source_ordinal)
                    target_energy = float(target_ordinal * target_ordinal)
                    cross = float(source_ordinal * target_ordinal)
                    residual = predicted + target_energy - 2.0 * cross
                    directed.append(
                        DirectedChartTransportStats(
                            family_id=family_id,
                            example_id=example_id,
                            source_mode_id=source.mode_id,
                            target_mode_id=target.mode_id,
                            comparison_abi_sha256=comparison.sha256,
                            aligned_row_catalog_sha256=row_catalog,
                            source_chart_stats_sha256=chart_by_key[
                                (family_id, example_id, source.mode_id)
                            ].sha256,
                            target_chart_stats_sha256=chart_by_key[
                                (family_id, example_id, target.mode_id)
                            ].sha256,
                            row_count=row_count,
                            transported_row_count=row_count,
                            transport_fit_role=worker.FIT_ROLE,
                            transport_evaluation_role=(
                                worker.DISCOVERY_ROLE
                            ),
                            transport_fit_catalog_sha256=fit_catalog,
                            transport_evaluation_catalog_sha256=row_catalog,
                            transport_evaluation_panel_sha256=panel_binding,
                            transport_split_receipt_sha256=split_receipt,
                            transport_fit_receipt_sha256=_sha(
                                f"audit-fit:{source.mode_id}:"
                                f"{target.mode_id}"
                            ),
                            row_shuffle_catalog_sha256=shuffle_catalog,
                            transport_protocol=(
                                FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL
                            ),
                            weighted_transport_residual_energy=residual,
                            weighted_predicted_fisher_energy=predicted,
                            weighted_target_fisher_energy=target_energy,
                            weighted_predicted_target_cross_inner_product=(
                                cross
                            ),
                            weighted_zero_prediction_residual_energy=(
                                target_energy
                            ),
                            weighted_row_shuffled_transport_residual_energy=(
                                residual + target_energy
                            ),
                        )
                    )
    return canonicalize_chart_energy_neighbor_input(
        ChartEnergyNeighborInput(
            panel_role=worker.DISCOVERY_ROLE,
            panel_binding_sha256=panel_binding,
            mode_specs=mode_specs,
            example_stats=tuple(example_stats),
            directed_transport_stats=tuple(directed),
            within_family_bootstrap_config=(
                ChartWithinFamilyBootstrapConfig(
                    seed=7_901,
                    resample_count=10_000,
                    draws_per_family=len(family_example_ids[0][1]),
                    maximum_degenerate_fraction=0.01,
                    family_example_ids=family_example_ids,
                )
            ),
            config=(config or ChartEnergyNeighborConfig()),
        )
    )


def test_frozen_graph_policy_rejects_coordinated_input_rehash_attacks() -> None:
    from fisher_graph import chart_energy_neighbor_graph as core
    from fisher_graph.gemma3_chart_native_neighbor_setup import build_setup

    manifest = fresh.load_manifest()
    role_manifest = manifest["roles"][worker.DISCOVERY_ROLE]
    eligibility = worker.build_role_eligibility_catalog(
        manifest,
        role=worker.DISCOVERY_ROLE,
    )
    setup = build_setup()

    def parse_and_hash(value: object) -> tuple[object, str]:
        payload = value.canonical_payload()
        # Model a coordinated attacker who updates the exact production input
        # digest after changing policy rather than leaving a stale checksum.
        attacker_digest = core._domain_digest(core._INPUT_DOMAIN, payload)
        parsed = core.parse_chart_energy_neighbor_input_payload(payload)
        assert attacker_digest == core._domain_digest(
            core._INPUT_DOMAIN,
            parsed.canonical_payload(),
        )
        return parsed, attacker_digest

    all_eight, all_eight_sha = parse_and_hash(
        _real_parser_graph_input(eligibility.all_family_example_ids)
    )
    primary, primary_sha = parse_and_hash(
        _real_parser_graph_input(eligibility.primary_family_example_ids)
    )
    assert all_eight.within_family_bootstrap_config.resample_count == 10_000
    assert primary.within_family_bootstrap_config.resample_count == 10_000
    assert all_eight.within_family_bootstrap_config.draws_per_family == 8
    assert primary.within_family_bootstrap_config.draws_per_family == 7
    worker._validate_role_graph_input_contract(
        setup=setup,
        role_manifest=role_manifest,
        eligibility=eligibility,
        all_eight_source=all_eight,
        primary_source=primary,
        capture_row_count=None,
    )

    config_attacks = (
        ("k", replace(all_eight.config, k=1)),
        (
            "rank boundary tolerance",
            replace(
                all_eight.config,
                rank_boundary_tie_absolute_tolerance=2.0e-10,
            ),
        ),
        (
            "Fisher epsilon",
            replace(all_eight.config, fisher_energy_epsilon=2.0e-12),
        ),
        (
            "maximum directional residual",
            replace(
                all_eight.config,
                maximum_directional_normalized_residual=0.99,
            ),
        ),
        (
            "bootstrap minimum frequency",
            replace(
                all_eight.config,
                bootstrap_minimum_mutual_frequency=0.79,
            ),
        ),
        (
            "LOFO minimum count",
            replace(all_eight.config, lofo_minimum_mutual_count=4),
        ),
    )
    bootstrap_attacks = (
        (
            "bootstrap seed",
            replace(all_eight.within_family_bootstrap_config, seed=7_902),
        ),
        (
            "bootstrap resample count",
            replace(
                all_eight.within_family_bootstrap_config,
                resample_count=9_999,
            ),
        ),
        (
            "bootstrap maximum degenerate fraction",
            replace(
                all_eight.within_family_bootstrap_config,
                maximum_degenerate_fraction=0.02,
            ),
        ),
    )
    attacks: list[tuple[str, object, object]] = [
        (
            label,
            replace(all_eight, config=attacked_config),
            primary,
        )
        for label, attacked_config in config_attacks
    ]
    attacks.extend(
        (
            label,
            replace(
                all_eight,
                within_family_bootstrap_config=attacked_bootstrap,
            ),
            primary,
        )
        for label, attacked_bootstrap in bootstrap_attacks
    )

    reversed_all_catalog = tuple(
        reversed(eligibility.all_family_example_ids)
    )
    wrong_primary_catalog = tuple(
        (family_id, example_ids[:-1])
        for family_id, example_ids in eligibility.all_family_example_ids
    )
    reordered_all, _ = parse_and_hash(
        _real_parser_graph_input(reversed_all_catalog)
    )
    wrong_primary, _ = parse_and_hash(
        _real_parser_graph_input(wrong_primary_catalog)
    )
    attacks.extend(
        (
            (
                "all-eight draw count and catalog replaced by exact-seven",
                primary,
                primary,
            ),
            (
                "exact-seven draw count and catalog replaced by all-eight",
                all_eight,
                all_eight,
            ),
            (
                "all-eight ordered family catalog",
                reordered_all,
                primary,
            ),
            (
                "exact-seven membership catalog",
                all_eight,
                wrong_primary,
            ),
        )
    )

    for label, attacked_all, attacked_primary in attacks:
        parsed_all, attacked_all_sha = parse_and_hash(attacked_all)
        parsed_primary, attacked_primary_sha = parse_and_hash(attacked_primary)
        assert (attacked_all_sha, attacked_primary_sha) != (
            all_eight_sha,
            primary_sha,
        ), label
        with pytest.raises(
            ValueError,
            match="role graph policy/bootstrap differs from frozen setup",
        ):
            worker._validate_role_graph_input_contract(
                setup=setup,
                role_manifest=role_manifest,
                eligibility=eligibility,
                all_eight_source=parsed_all,
                primary_source=parsed_primary,
                capture_row_count=None,
            )

    # Draw-count and domain changes are rejected by the real core parser even
    # before the outer frozen binder; recomputing the input digest cannot help.
    parser_attacks: list[tuple[str, dict[str, object]]] = []
    all_payload = deepcopy(all_eight.canonical_payload())
    all_payload["within_family_bootstrap_config"]["draws_per_family"] = 7
    parser_attacks.append(("all-eight draws", all_payload))
    primary_payload = deepcopy(primary.canonical_payload())
    primary_payload["within_family_bootstrap_config"]["draws_per_family"] = 8
    parser_attacks.append(("exact-seven draws", primary_payload))
    domain_payload = deepcopy(all_eight.canonical_payload())
    domain_payload["within_family_bootstrap_config"][
        "draw_protocol_domain_hex"
    ] = b"fisher_graph:attacker-bootstrap-domain:v1\0".hex()
    parser_attacks.append(("bootstrap draw domain", domain_payload))
    for label, attacked_payload in parser_attacks:
        attacker_digest = core._domain_digest(
            core._INPUT_DOMAIN,
            attacked_payload,
        )
        assert attacker_digest != all_eight_sha, label
        with pytest.raises(
            ValueError,
            match=(
                "draws_per_family must equal examples per family|"
                "draw_protocol_domain_hex is unsupported"
            ),
        ):
            core.parse_chart_energy_neighbor_input_payload(attacked_payload)
