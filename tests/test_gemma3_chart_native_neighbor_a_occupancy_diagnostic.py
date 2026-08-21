from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic as diagnostic,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _counter_payload() -> dict[str, object]:
    return {
        "role_batch_counts": [
            [diagnostic.FIT_ROLE, 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ],
        "chart_backbone_forward_count": 45,
        "chart_terminal_cotangent_vjp_count": 45,
        "dense_parity_smoke_count": 1,
        "parity_backbone_forward_count": 1,
        "parity_terminal_cotangent_vjp_count": 1,
        "captured_example_count": 48,
        "supplemental_route_receipt_count": 864,
        "canonical_gradient_targets_per_vjp": 73,
        "auxiliary_gradient_targets_per_vjp": 19,
        "gradient_targets_per_vjp": 92,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_activations": False,
        "contains_raw_gradients": False,
        "contains_logits": False,
    }


def _safe_result_payload() -> dict[str, object]:
    counters = _counter_payload()
    return {
        "setup": {
            "path": str(diagnostic.DEFAULT_SETUP_PATH),
            "setup_sha256": diagnostic.PINNED_SETUP_SHA256,
            "setup_file_sha256": diagnostic.PINNED_SETUP_FILE_SHA256,
        },
        "preflight": {
            "execution_contract_sha256": (
                diagnostic.V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
            ),
            "preflight_receipt_sha256": _sha("preflight"),
            "v1_setup_sha256": diagnostic.V1_SETUP_SHA256,
            "diagnostic_source_file_sha256s": {
                "diagnostic_runner": diagnostic.PINNED_RUNNER_SOURCE_SHA256,
                "streaming_atlas_diagnostic": (
                    diagnostic.PINNED_NUMERIC_SOURCE_SHA256
                ),
            },
        },
        "runtime": {
            "qualification_receipt_sha256": (
                diagnostic.V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256
            ),
            "runtime_session_binding_sha256": _sha("session"),
            "runtime_counters": counters,
            "runtime_counters_receipt_sha256": diagnostic._digest(
                diagnostic._COUNTER_DOMAIN, counters
            ),
            "runtime_closed_before_result_seal": True,
        },
        "A_capture": {
            "capture_receipt_sha256": _sha("capture"),
            "role_panel_binding_sha256": (
                diagnostic.V1_A_ROLE_PANEL_BINDING_SHA256
            ),
            "ordered_objective_row_catalog_sha256": (
                diagnostic.V1_A_ORDERED_ROW_CATALOG_SHA256
            ),
            "batch_count": 45,
            "example_count": 48,
            "mode_count": 38,
            "rows_per_mode": 24_732,
            "total_pool_rows": 939_816,
        },
        "K8_occupancy_diagnosis": {
            "schema": "test.numeric.diagnosis",
            "diagnostic_receipt_sha256": _sha("diagnosis"),
        },
        "firewall": {
            "opened_roles": [diagnostic.FIT_ROLE],
            "freeze_receipt_count": 0,
            "A_semantically_opened": True,
            "B_semantically_opened": False,
            "C_semantically_opened": False,
            "D_semantically_opened": False,
            "reserve_semantically_opened": False,
        },
        "cleanup": {
            "workspace_removed_before_result_seal": True,
            "raw_spool_count_after_cleanup": 0,
            "atlas_artifact_count": 0,
            "transport_artifact_count": 0,
            "graph_artifact_count": 0,
        },
        "filesystem": {
            "before_catalog_sha256": _sha("filesystem"),
            "prepublication_catalog_sha256": _sha("filesystem"),
            "prepublication_equals_before": True,
            "staging_entry_count_before": 0,
            "success_allows_exactly_one_new_regular_result": True,
            "failure_requires_exact_catalog_equality": True,
            "postpublication_exact_delta_verified": True,
        },
        "claims": {
            "K8_occupancy_diagnostic_measured": True,
            "replacement_chart_count_selected": False,
            "atlas_frozen": False,
            "transport_fitted": False,
            "neighbor_graph_measured": False,
            "response_validation_measured": False,
            "weights_alone_determine_neighbors": False,
        },
    }


def test_result_seal_binds_firewall_cleanup_and_no_claims() -> None:
    result = diagnostic.seal_result(_safe_result_payload())
    diagnostic._validate_result(result, replay_numeric=False)
    assert result["firewall"]["opened_roles"] == [diagnostic.FIT_ROLE]
    assert result["cleanup"]["raw_spool_count_after_cleanup"] == 0
    assert result["claims"]["replacement_chart_count_selected"] is False


def test_numeric_summary_field_guards_match_public_state_dict_types() -> None:
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as numeric

    guards = {
        diagnostic._K8_DIAGNOSIS_FIELDS: (
            numeric.StreamingSharedChartAtlasOccupancyDiagnosisV1
        ),
        diagnostic._K8_CANDIDATE_FIELDS: (
            numeric.StreamingSharedChartAtlasCandidateOutcomeV1
        ),
        diagnostic._K8_SEED_FIELDS: (
            numeric.StreamingSharedChartAtlasSeedReceiptV1
        ),
        diagnostic._K8_LLOYD_PASS_FIELDS: (
            numeric.StreamingSharedChartAtlasPassReceiptV1
        ),
        diagnostic._K8_GATE_EVIDENCE_FIELDS: (
            numeric.StreamingSharedChartAtlasGateEvidenceV1
        ),
    }
    for guarded_fields, public_type in guards.items():
        assert set(guarded_fields) == {
            member.name for member in fields(public_type)
        } | {"schema"}


def test_numeric_summary_rejects_valid_coordinated_alternate_outcome(
    monkeypatch, tmp_path: Path
) -> None:
    import torch

    from fisher_graph.shared_chart_energy_transport import (
        FIT_ROLE,
        ChartRowKey,
        SharedChartNumericsConfig,
        SharedChartRowBatch,
    )
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        ADMISSIBLE_OUTCOME,
        diagnose_shared_chart_atlas_occupancy_streaming,
        parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        write_raw_mode_row_spool,
    )

    keys = tuple(
        ChartRowKey(
            family_id=f"family.{family}",
            example_id=f"example.{family}.{example}",
            prediction_query_position=row,
            target_label_position=row + 1,
        )
        for family in range(2)
        for example in range(2)
        for row in range(48)
    )
    vertices = tuple(
        (
            -12.0 if chart & 1 else 12.0,
            -12.0 if chart & 2 else 12.0,
            -12.0 if chart & 4 else 12.0,
        )
        for chart in range(8)
    )
    batches = []
    for mode in range(2):
        generator = torch.Generator().manual_seed(72_901 + mode)
        context = torch.randn(
            len(keys), 20, generator=generator, dtype=torch.float64
        )
        for row in range(len(keys)):
            context[row, :3] += torch.tensor(vertices[row % 8])
        batches.append(
            SharedChartRowBatch(
                role=FIT_ROLE,
                panel_binding_sha256=_sha("alternate-panel"),
                mode_id=f"mode.{mode:02d}",
                row_keys=keys,
                context_values=context.to(torch.float32),
                comparison_values=(0.25 * context).to(torch.float32),
                generalized_contributions=context.abs().to(torch.float32),
            )
        )
    resources = StreamingChartResourceConfig()
    spools = tuple(
        write_raw_mode_row_spool(
            batch,
            tmp_path / f"alternate-{index}.fgstream",
            resources=resources,
        )
        for index, batch in enumerate(batches)
    )
    diagnosis = diagnose_shared_chart_atlas_occupancy_streaming(
        spools,
        config=SharedChartNumericsConfig(hidden_width=20),
        resources=resources,
    )
    state = diagnosis.state_dict()
    assert diagnosis.candidate.outcome == ADMISSIBLE_OUTCOME
    assert parse_streaming_shared_chart_atlas_occupancy_diagnosis_v1(
        state
    ) == diagnosis

    monkeypatch.setattr(diagnostic, "HIDDEN_WIDTH", 20)
    monkeypatch.setattr(diagnostic, "MODE_COUNT", 2)
    monkeypatch.setattr(diagnostic, "FAMILY_COUNT", 2)
    monkeypatch.setattr(diagnostic, "ROWS_PER_MODE", len(keys))
    monkeypatch.setattr(diagnostic, "TOTAL_POOL_ROWS", 2 * len(keys))
    monkeypatch.setattr(
        diagnostic, "CANONICAL_MODE_IDS", ("mode.00", "mode.01")
    )
    monkeypatch.setattr(
        diagnostic, "CANONICAL_FAMILY_IDS", ("family.0", "family.1")
    )
    with pytest.raises(ValueError, match="frozen v1 hard-mass terminal"):
        diagnostic._validate_K8_numeric_summary(state)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"prompt_text": "secret"},
        {"token_ids": [1, 2]},
        {"context_values": [[1.0]]},
        {"generalized_contributions": [[1.0]]},
        {"centers": [[1.0]]},
    ],
)
def test_result_seal_rejects_raw_material(forbidden: dict[str, object]) -> None:
    payload = _safe_result_payload()
    payload["K8_occupancy_diagnosis"] = forbidden
    with pytest.raises(ValueError, match="forbidden raw material"):
        diagnostic.seal_result(payload)


def test_result_validation_rejects_later_role_or_replacement_selection() -> None:
    payload = _safe_result_payload()
    payload["firewall"]["B_semantically_opened"] = True
    with pytest.raises(ValueError, match="firewall drifted"):
        diagnostic.seal_result(payload)
    payload = _safe_result_payload()
    payload["claims"]["replacement_chart_count_selected"] = True
    with pytest.raises(ValueError, match="claims drifted"):
        diagnostic.seal_result(payload)


def test_result_validation_rejects_substituted_source_lineage() -> None:
    payload = _safe_result_payload()
    payload["preflight"]["diagnostic_source_file_sha256s"][
        "diagnostic_runner"
    ] = _sha("substituted-runner")
    with pytest.raises(ValueError, match="source inventory drifted"):
        diagnostic.seal_result(payload)
    payload = _safe_result_payload()
    payload["runtime"]["qualification_receipt_sha256"] = _sha(
        "substituted-qualification"
    )
    with pytest.raises(ValueError, match="qualification binding drifted"):
        diagnostic.seal_result(payload)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        (
            "preflight",
            "execution_contract_sha256",
            _sha("substituted-execution-contract"),
            "execution contract binding drifted",
        ),
        (
            "preflight",
            "v1_setup_sha256",
            _sha("substituted-v1-setup"),
            "predecessor setup binding drifted",
        ),
        (
            "runtime",
            "qualification_receipt_sha256",
            _sha("substituted-runtime-qualification"),
            "qualification binding drifted",
        ),
    ],
)
def test_strict_loader_rejects_coordinated_rehashed_lineage_substitution(
    monkeypatch,
    tmp_path: Path,
    section: str,
    field: str,
    replacement: str,
    message: str,
) -> None:
    result = diagnostic.seal_result(_safe_result_payload())
    result[section][field] = replacement
    unsigned = dict(result)
    unsigned.pop("result_sha256")
    result["result_sha256"] = diagnostic._digest(
        diagnostic._RESULT_DOMAIN, unsigned
    )
    source = tmp_path / "coordinated-rehash.json"
    source.write_bytes(
        diagnostic.json.dumps(
            result, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    monkeypatch.setattr(
        diagnostic,
        "check_setup",
        lambda _: {
            "setup_sha256": diagnostic.PINNED_SETUP_SHA256,
            "predecessor_v1": {
                "setup_sha256": diagnostic.V1_SETUP_SHA256,
            },
        },
    )
    with pytest.raises(ValueError, match=message):
        diagnostic.strict_load_result(
            source, diagnostic_setup_path=tmp_path / "setup.json"
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("A_capture", "batch_count", 45.0, "capture inventory drifted"),
        ("A_capture", "example_count", True, "capture inventory drifted"),
        (
            "cleanup",
            "raw_spool_count_after_cleanup",
            False,
            "cleanup receipt drifted",
        ),
    ],
)
def test_result_validation_rejects_bool_float_inventory_substitution(
    section: str, field: str, replacement: object, message: str
) -> None:
    payload = _safe_result_payload()
    payload[section][field] = replacement
    with pytest.raises(ValueError, match=message):
        diagnostic.seal_result(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("chart_backbone_forward_count", 45.0),
        ("dense_parity_smoke_count", True),
        (
            "role_batch_counts",
            [
                [diagnostic.FIT_ROLE, 45.0],
                ["B_neighbor_discovery", 0],
                ["C_neighbor_confirmation", 0],
            ],
        ),
        (
            "role_batch_counts",
            (
                [diagnostic.FIT_ROLE, 45],
                ["B_neighbor_discovery", 0],
                ["C_neighbor_confirmation", 0],
            ),
        ),
    ],
)
def test_result_validation_rejects_rehashed_counter_type_substitution(
    field: str, replacement: object
) -> None:
    payload = _safe_result_payload()
    counters = payload["runtime"]["runtime_counters"]
    counters[field] = replacement
    payload["runtime"]["runtime_counters_receipt_sha256"] = diagnostic._digest(
        diagnostic._COUNTER_DOMAIN, counters
    )
    with pytest.raises(ValueError, match="exactly one A capture"):
        diagnostic.seal_result(payload)


def test_result_validation_rejects_rehashed_boolean_format_version() -> None:
    result = diagnostic.seal_result(_safe_result_payload())
    result["format_version"] = True
    unsigned = dict(result)
    unsigned.pop("result_sha256")
    result["result_sha256"] = diagnostic._digest(
        diagnostic._RESULT_DOMAIN, unsigned
    )
    with pytest.raises(ValueError, match="envelope drifted"):
        diagnostic._validate_result(result, replay_numeric=False)


def test_cli_requires_explicit_action() -> None:
    with pytest.raises(SystemExit) as error:
        diagnostic.main([])
    assert error.value.code == 2


def test_selection_kernel_preflight_requires_exact_frozen_pin(
    monkeypatch,
) -> None:
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as numeric

    monkeypatch.setattr(
        numeric,
        "streaming_shared_chart_atlas_selection_kernel_receipt_v1",
        lambda _resources: {"receipt_sha256": _sha("substituted-kernel")},
    )
    binding = SimpleNamespace(
        storage_budget=SimpleNamespace(
            maximum_resident_memory_bytes=8 * 1024**3
        )
    )
    with pytest.raises(ValueError, match="selection-kernel pin drifted"):
        diagnostic._validated_selection_resources(binding)


def test_capture_lineage_rejects_coordinated_renamed_mode_axis() -> None:
    from fisher_graph.streaming_shared_chart_atlas_selection_v1 import (
        streaming_shared_chart_atlas_mode_identity_sha256_v1,
    )

    spools = []
    receipt_rows = []
    kernel_receipt = _sha("streaming-kernel")
    for mode_id in diagnostic.CANONICAL_MODE_IDS:
        artifact = _sha(f"artifact:{mode_id}")
        row_catalog = _sha("rows")
        tensor_catalog = _sha(f"tensors:{mode_id}")
        spools.append(
            SimpleNamespace(
                mode_id=mode_id,
                artifact_sha256=artifact,
                row_catalog_sha256=row_catalog,
                tensor_catalog_sha256=tensor_catalog,
                panel_binding_sha256=(
                    diagnostic.V1_A_ROLE_PANEL_BINDING_SHA256
                ),
                kernel_receipt_sha256=kernel_receipt,
                row_count=diagnostic.ROWS_PER_MODE,
            )
        )
        receipt_rows.append(
            (mode_id, artifact, row_catalog, tensor_catalog)
        )
    captured = SimpleNamespace(
        raw_spools=tuple(spools),
        receipt=SimpleNamespace(
            mode_raw_artifacts=tuple(receipt_rows),
            role_panel_binding_sha256=(
                diagnostic.V1_A_ROLE_PANEL_BINDING_SHA256
            ),
            ordered_objective_row_catalog_sha256=(
                diagnostic.V1_A_ORDERED_ROW_CATALOG_SHA256
            ),
        ),
    )
    diagnosis = {
        "fit_role": diagnostic.FIT_ROLE,
        "fit_mode_ids": list(diagnostic.CANONICAL_MODE_IDS),
        "fit_family_ids": list(diagnostic.CANONICAL_FAMILY_IDS),
        "fit_row_count_by_family": [4_122] * diagnostic.FAMILY_COUNT,
        "fit_panel_binding_sha256": (
            diagnostic.V1_A_ROLE_PANEL_BINDING_SHA256
        ),
        "fit_row_catalog_sha256": _sha("rows"),
        "kernel_receipt_sha256": kernel_receipt,
        "spool_mode_identity_sha256s": [
            streaming_shared_chart_atlas_mode_identity_sha256_v1(
                mode_id, ordinal
            )
            for ordinal, mode_id in enumerate(diagnostic.CANONICAL_MODE_IDS)
        ],
        "spool_artifact_sha256s": [row[1] for row in receipt_rows],
        "spool_tensor_catalog_sha256s": [row[3] for row in receipt_rows],
        "mode_count": diagnostic.MODE_COUNT,
        "row_count_per_mode": diagnostic.ROWS_PER_MODE,
        "total_row_count": diagnostic.TOTAL_POOL_ROWS,
    }
    diagnostic._validate_capture_and_diagnosis_lineage(
        captured=captured, diagnosis=diagnosis
    )
    original_mode_identity = diagnosis["spool_mode_identity_sha256s"][0]
    diagnosis["spool_mode_identity_sha256s"][0] = _sha("substituted-mode")
    with pytest.raises(RuntimeError, match="just-captured A spools"):
        diagnostic._validate_capture_and_diagnosis_lineage(
            captured=captured, diagnosis=diagnosis
        )
    diagnosis["spool_mode_identity_sha256s"][0] = original_mode_identity
    diagnosis["fit_mode_ids"][0] = "mode.renamed"
    with pytest.raises(RuntimeError, match="just-captured A spools"):
        diagnostic._validate_capture_and_diagnosis_lineage(
            captured=captured, diagnosis=diagnosis
        )


class _FakePanelSource:
    last: "_FakePanelSource | None" = None

    def __init__(self, **_: object) -> None:
        self.opened_roles: tuple[str, ...] = ()
        self.current_role: str | None = None
        self.freeze_receipts: tuple[object, ...] = ()
        self.next_role = diagnostic.FIT_ROLE
        _FakePanelSource.last = self

    def open_role(self, role: str) -> dict[str, object]:
        assert role == diagnostic.FIT_ROLE
        self.opened_roles = (role,)
        self.current_role = role
        return {"examples": [{"example_id": "A0"}]}


class _FakeCounters:
    def payload(self) -> dict[str, object]:
        return _counter_payload()


class _FakeRuntime:
    def __init__(self, qualification: object) -> None:
        self.mode_lineage = qualification.mode_lineage
        self.runtime_session_binding_sha256 = _sha("expected-session")
        self.hidden_width = 640
        self.counters = _FakeCounters()
        self._qualification = qualification
        self.closed = False
        self.close_calls = 0
        self.fail_first_close = False

    def runtime_qualification_receipt(self) -> object:
        return self._qualification

    def run_dense_lexical_parity_smoke(self, **_: object) -> str:
        return _sha("parity")

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_first_close and self.close_calls == 1:
            raise RuntimeError("injected close failure")
        self.closed = True


class _FakeDiagnosis:
    def state_dict(self) -> dict[str, object]:
        return {"diagnostic_receipt_sha256": _sha("numeric")}


def _install_orchestration_fakes(monkeypatch, tmp_path: Path):
    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1
    from fisher_graph import gemma3_chart_native_neighbor_fresh_panels as panels
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as numeric

    qualification = SimpleNamespace(
        qualification_receipt_sha256=(
            diagnostic.V1_RUNTIME_QUALIFICATION_RECEIPT_SHA256
        ),
        source_model_fingerprint_after_compilation_sha256=_sha("model"),
        source_execution_fingerprint_after_compilation_sha256=_sha("execution"),
        compiled_execution_fingerprint_sha256=_sha("compiled"),
        tied_readout_binding_receipt_sha256=_sha("readout"),
        mode_lineage=tuple(range(38)),
    )
    preflight = SimpleNamespace(
        receipt_sha256=_sha("preflight"),
        setup={"setup_sha256": diagnostic.V1_SETUP_SHA256},
        fresh_manifest={
            "roles": {diagnostic.FIT_ROLE: {"role": diagnostic.FIT_ROLE}}
        },
        fresh_receipt={},
        opaque_panel_receipt=object(),
        validate_receipt=lambda: None,
    )
    monkeypatch.setattr(diagnostic, "check_setup", lambda _: {
        "setup_sha256": diagnostic.PINNED_SETUP_SHA256
    })
    monkeypatch.setattr(
        diagnostic,
        "_validate_local_source_pins",
        lambda *_args, **_kwargs: {
            "diagnostic_runner": diagnostic.PINNED_RUNNER_SOURCE_SHA256,
            "streaming_atlas_diagnostic": (
                diagnostic.PINNED_NUMERIC_SOURCE_SHA256
            ),
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_validated_selection_resources",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(v1, "static_preflight", lambda _: preflight)
    monkeypatch.setattr(v1, "validate_runtime_qualification", lambda **_: None)
    monkeypatch.setattr(
        v1,
        "derive_runtime_session_binding_sha256",
        lambda **_: _sha("expected-session"),
    )
    monkeypatch.setattr(v1, "build_core_mode_specs", lambda **_: tuple(range(38)))
    captured = SimpleNamespace(
        raw_spools=(object(),),
        receipt=SimpleNamespace(
            receipt_sha256=_sha("capture"),
            role_panel_binding_sha256=(
                diagnostic.V1_A_ROLE_PANEL_BINDING_SHA256
            ),
            ordered_objective_row_catalog_sha256=(
                diagnostic.V1_A_ORDERED_ROW_CATALOG_SHA256
            ),
            batch_count=45,
            row_count=24_732,
        ),
    )
    monkeypatch.setattr(v1, "capture_role_to_raw_spools", lambda **_: captured)
    monkeypatch.setattr(panels, "SequentialPanelSource", _FakePanelSource)
    monkeypatch.setattr(
        diagnostic,
        "_numeric_diagnosis_state",
        lambda _: {"diagnostic_receipt_sha256": _sha("numeric")},
    )
    monkeypatch.setattr(
        diagnostic,
        "_validate_capture_and_diagnosis_lineage",
        lambda **_: None,
    )
    monkeypatch.setattr(
        numeric,
        "validate_streaming_shared_chart_atlas_occupancy_diagnosis_"
        "against_raw_spools_v1",
        lambda *_args, **_kwargs: None,
    )
    def fake_write_result(result, destination):
        Path(destination).write_bytes(b"authenticated diagnostic test result\n")
        return dict(result)

    monkeypatch.setattr(diagnostic, "write_result", fake_write_result)
    temp = tmp_path / "temp"
    temp.mkdir()
    binding = SimpleNamespace(
        repository_root=tmp_path,
        output_directory=tmp_path,
        temp_directory=temp,
        test_only_allow_nonproduction_shape=True,
        execution_contract_sha256=(
            diagnostic.V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
        ),
        storage_budget=SimpleNamespace(maximum_resident_memory_bytes=8 * 1024**3),
    )
    runtime = _FakeRuntime(qualification)
    return binding, runtime


def test_A_only_orchestration_closes_cleans_and_never_opens_later_roles(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    result = diagnostic.run_A_only_K8_occupancy_diagnostic(
        binding=binding,
        runtime_factory=lambda _: runtime,
        output_path=tmp_path / "result.json",
        diagnostic_setup_path=tmp_path / "setup.json",
        diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
    )
    assert runtime.closed
    assert _FakePanelSource.last is not None
    assert _FakePanelSource.last.opened_roles == (diagnostic.FIT_ROLE,)
    assert result["firewall"]["B_semantically_opened"] is False
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()


def test_preflight_only_authenticates_sources_and_kernel_without_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    receipt = diagnostic.preflight_only(
        binding=binding,
        diagnostic_setup_path=tmp_path / "setup.json",
    )
    assert receipt == {
        "diagnostic_setup_sha256": diagnostic.PINNED_SETUP_SHA256,
        "execution_contract_sha256": (
            diagnostic.V1_PRODUCTION_EXECUTION_CONTRACT_SHA256
        ),
        "preflight_receipt_sha256": _sha("preflight"),
        "diagnostic_source_file_sha256s": {
            "diagnostic_runner": diagnostic.PINNED_RUNNER_SOURCE_SHA256,
            "streaming_atlas_diagnostic": (
                diagnostic.PINNED_NUMERIC_SOURCE_SHA256
            ),
        },
        "selection_kernel_receipt_sha256": (
            diagnostic.PINNED_SELECTION_KERNEL_RECEIPT_SHA256
        ),
        "v1_setup_sha256": diagnostic.V1_SETUP_SHA256,
    }
    assert runtime.close_calls == 0
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()


def test_A_only_failure_closes_runtime_and_removes_spools(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("diagnosis failed")

    with pytest.raises(RuntimeError, match="diagnosis failed"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=fail,
        )
    assert runtime.closed
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()
    assert not (tmp_path / "result.json").exists()


def test_A_only_panel_source_initialization_failure_removes_workspace(
    monkeypatch, tmp_path: Path
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_fresh_panels as panels

    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    baseline = diagnostic._filesystem_catalog(binding)

    def fail_panel_source(**_kwargs):
        raise RuntimeError("injected panel-source initialization failure")

    monkeypatch.setattr(panels, "SequentialPanelSource", fail_panel_source)
    with pytest.raises(RuntimeError, match="panel-source initialization failure"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert runtime.close_calls == 0
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()
    assert diagnostic._filesystem_catalog(binding) == baseline


def test_A_only_preflight_failure_rolls_back_new_filesystem_entry(
    monkeypatch, tmp_path: Path
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1

    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    baseline = diagnostic._filesystem_catalog(binding)
    escaped = binding.temp_directory / "preflight-side-artifact.bin"

    def fail_preflight(_binding):
        escaped.write_bytes(b"unexpected\n")
        raise RuntimeError("injected preflight failure")

    monkeypatch.setattr(v1, "static_preflight", fail_preflight)
    with pytest.raises(RuntimeError, match="preflight failure"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert runtime.close_calls == 0
    assert not escaped.exists()
    assert diagnostic._filesystem_catalog(binding) == baseline


def test_A_only_failure_removes_new_temp_side_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    baseline = diagnostic._filesystem_catalog(binding)
    escaped = binding.temp_directory / "unexpected-side-artifact.bin"

    def fail(*_args, **_kwargs):
        escaped.write_bytes(b"unexpected\n")
        raise RuntimeError("diagnosis failed with side artifact")

    with pytest.raises(RuntimeError, match="diagnosis failed with side artifact"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=fail,
        )
    assert runtime.closed
    assert not escaped.exists()
    assert diagnostic._filesystem_catalog(binding) == baseline


def test_A_only_workspace_remover_failure_still_runs_catalog_rollback(
    monkeypatch, tmp_path: Path
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_experiment as v1

    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    baseline = diagnostic._filesystem_catalog(binding)

    def fail_workspace_removal(*_args, **_kwargs):
        raise RuntimeError("injected workspace remover failure")

    monkeypatch.setattr(v1, "_safe_remove_tree", fail_workspace_removal)
    with pytest.raises(RuntimeError, match="workspace remover failure"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert runtime.closed
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()
    assert not (tmp_path / "result.json").exists()
    assert diagnostic._filesystem_catalog(binding) == baseline


def test_A_only_rejects_diagnosis_not_bound_to_captured_spools(
    monkeypatch, tmp_path: Path
) -> None:
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as numeric

    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)

    def reject_swapped(*_args, **_kwargs):
        raise ValueError("diagnosis/spool lineage differs")

    monkeypatch.setattr(
        numeric,
        "validate_streaming_shared_chart_atlas_occupancy_diagnosis_"
        "against_raw_spools_v1",
        reject_swapped,
    )
    with pytest.raises(ValueError, match="diagnosis/spool lineage differs"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert runtime.closed
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("injection", ["diagnose", "runtime"])
def test_live_binding_rejects_injected_execution_callables(
    monkeypatch, tmp_path: Path, injection: str
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    binding.test_only_allow_nonproduction_shape = False
    arguments = {
        "binding": binding,
        "output_path": tmp_path / "result.json",
        "diagnostic_setup_path": tmp_path / "setup.json",
    }
    if injection == "diagnose":
        arguments["diagnose"] = lambda *_args, **_kwargs: _FakeDiagnosis()
    else:
        arguments["runtime_factory"] = lambda _: runtime
    with pytest.raises(ValueError, match="live diagnostic forbids"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(**arguments)
    assert runtime.close_calls == 0


def test_A_only_close_failure_is_retried_and_rolls_back(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    runtime.fail_first_close = True

    with pytest.raises(RuntimeError, match="injected close failure"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert runtime.close_calls == 2
    assert runtime.closed
    assert not (binding.temp_directory / diagnostic._WORKSPACE_NAME).exists()
    assert not (tmp_path / "result.json").exists()


def test_A_only_extra_publication_entry_fails_and_is_rolled_back(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    baseline = diagnostic._filesystem_catalog(binding)
    extra = tmp_path / "escaped-side-artifact.bin"

    def write_with_extra(result, destination):
        Path(destination).write_bytes(b"result\n")
        extra.write_bytes(b"escape\n")
        return dict(result)

    monkeypatch.setattr(diagnostic, "write_result", write_with_extra)
    with pytest.raises(RuntimeError, match="beyond its result"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=lambda _: runtime,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert runtime.closed
    assert not (tmp_path / "result.json").exists()
    assert not extra.exists()
    assert diagnostic._filesystem_catalog(binding) == baseline


def test_A_only_rejects_preexisting_staging_before_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    staging = tmp_path / ".result.json.injected.tmp"
    staging.write_bytes(b"preexisting\n")
    factory_called = False

    def runtime_factory(_):
        nonlocal factory_called
        factory_called = True
        return runtime

    with pytest.raises(FileExistsError, match="staging entry"):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=runtime_factory,
            output_path=tmp_path / "result.json",
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert factory_called is False
    assert runtime.close_calls == 0


def test_A_only_rejects_symlink_destination_before_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    binding, runtime = _install_orchestration_fakes(monkeypatch, tmp_path)
    target = tmp_path / "unrelated.json"
    target.write_bytes(b"preserve\n")
    destination = tmp_path / "result.json"
    destination.symlink_to(target)
    factory_called = False

    def runtime_factory(_):
        nonlocal factory_called
        factory_called = True
        return runtime

    with pytest.raises(FileExistsError):
        diagnostic.run_A_only_K8_occupancy_diagnostic(
            binding=binding,
            runtime_factory=runtime_factory,
            output_path=destination,
            diagnostic_setup_path=tmp_path / "setup.json",
            diagnose=lambda *_args, **_kwargs: _FakeDiagnosis(),
        )
    assert factory_called is False
    assert target.read_bytes() == b"preserve\n"


def test_write_result_post_read_failure_unlinks_and_fsyncs(
    monkeypatch, tmp_path: Path
) -> None:
    result = diagnostic.seal_result(_safe_result_payload())
    monkeypatch.setattr(
        diagnostic, "_validate_result", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        diagnostic,
        "strict_load_result",
        lambda _: (_ for _ in ()).throw(RuntimeError("post-read fault")),
    )
    fsynced: list[Path] = []
    real_fsync_directory = diagnostic._fsync_directory

    def record_fsync(path: Path) -> None:
        fsynced.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(diagnostic, "_fsync_directory", record_fsync)
    destination = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="post-read fault"):
        diagnostic.write_result(result, destination)
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".result.json.*.tmp"))
    assert fsynced == [tmp_path, tmp_path]
