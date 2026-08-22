from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_attention_transport_artifact as artifact_module
import fisher_graph.gemma3_attention_transport_experiment as experiment
from fisher_graph.gemma3_attention_transport_experiment import (
    EXPECTED_BATCHES_PER_ROLE,
    EXPECTED_INVOCATIONS_PER_BATCH,
    EXPECTED_ROWS_PER_ROLE,
    NO_PASS_REPORT_SCHEMA,
    ROWS_PER_BATCH_AND_INVOCATION,
    _IMPLEMENTATION_SOURCES,
    _NO_PASS_DOMAIN,
    _canonical_json_bytes,
    _capture_role,
    _check_output_publication_headroom,
    _check_resident_set_budget,
    _fixed_protocol_after_preflight,
    _json_sha256,
    _publish_no_pass_report,
    _strict_no_pass_report,
    check_attention_transport_outputs,
)


def _noop_publication_guard(_phase_id: str) -> None:
    return None


def test_cli_module_defers_transport_runtime_and_compiler_imports() -> None:
    code = """
import sys
import fisher_graph.gemma3_attention_transport_experiment
forbidden = (
    'fisher_graph.gemma3_attention_transport_compiler',
    'fisher_graph.gemma3_attention_transport_runtime',
    'fisher_graph.gemma3_attention_transport_experiment_core',
    'fisher_graph.gemma3_attention_transport_artifact',
    'fisher_graph.gemma3_chart_native_neighbor_runtime',
    'fisher_graph.gemma3_experiment',
    'transformers',
)
assert not [name for name in forbidden if name in sys.modules]
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_fixed_all_length_canary_protocol_is_not_cli_tunable() -> None:
    configs, gates = _fixed_protocol_after_preflight()

    assert tuple(config.candidate_id for config in configs) == (
        "k4_r8_top2",
        "k8_r8_top2",
        "k8_r16_top2",
        "k8_r16_top4",
    )
    assert gates.minimum_aggregate_rows == EXPECTED_ROWS_PER_ROLE == 12_960
    assert gates.max_relative_fisher_loss == 0.10
    assert gates.oracle_gap_numerical_tolerance == 1.0e-10
    assert gates.max_postnorm_normalized_rmse == 0.02
    assert gates.min_postnorm_cosine == 0.999
    assert gates.require_parameter_compression is True


def test_capture_role_requires_45_by_18_sampled_row_ledger() -> None:
    receipt_counter = 0
    sink_state: dict[str, object] = {
        "active_core_role": None,
        "records": [],
    }

    class FakePanelSource:
        def open_role(self, role: str) -> dict[str, object]:
            assert role == "runtime_A"
            return {"examples": ({"id": "first"},)}

    class FakeRuntime:
        def run_dense_lexical_parity_smoke(self, **_kwargs: object) -> str:
            return "a" * 64

        def capture_batch(self, **_kwargs: object) -> SimpleNamespace:
            nonlocal receipt_counter
            receipt_counter += 1
            receipt = f"{receipt_counter:064x}"
            records = sink_state["records"]
            assert isinstance(records, list)
            records.extend(
                SimpleNamespace(
                    base_capture_receipt_sha256=receipt,
                    capture=SimpleNamespace(
                        row_count=ROWS_PER_BATCH_AND_INVOCATION,
                    ),
                )
                for _ in range(EXPECTED_INVOCATIONS_PER_BATCH)
            )
            return SimpleNamespace(execution_receipt_sha256=receipt)

    progress: list[tuple[str, int, int]] = []
    resource_phases: list[str] = []
    records, parity = _capture_role(
        runtime=FakeRuntime(),
        preflight=SimpleNamespace(
            fresh_manifest={"roles": {"runtime_A": {}}},
        ),
        panel_source=FakePanelSource(),
        runtime_role="runtime_A",
        core_role="A",
        sink_state=sink_state,
        capture_batch_plan=lambda _corpus: tuple(
            (object(),) for _ in range(EXPECTED_BATCHES_PER_ROLE)
        ),
        run_dense_parity=True,
        progress=lambda role, complete, total: progress.append(
            (role, complete, total)
        ),
        resource_guard=resource_phases.append,
    )

    assert parity == "a" * 64
    assert len(records) == EXPECTED_BATCHES_PER_ROLE * 18
    assert sum(record.capture.row_count for record in records) == 12_960
    assert progress[0] == ("A", 1, 45)
    assert progress[-1] == ("A", 45, 45)
    assert resource_phases[0] == "A_capture_batch_01"
    assert resource_phases[-1] == "A_capture_batch_45"
    assert len(resource_phases) == EXPECTED_BATCHES_PER_ROLE
    assert sink_state == {"active_core_role": None, "records": []}


def test_resident_set_guard_enforces_frozen_budget() -> None:
    binding = SimpleNamespace(
        storage_budget=SimpleNamespace(maximum_resident_memory_bytes=1_024),
    )

    assert (
        _check_resident_set_budget(
            binding,
            lambda: 1_024,
            phase_id="fit_complete",
        )
        == 1_024
    )
    with pytest.raises(RuntimeError, match="fit_complete.*resident-set cap"):
        _check_resident_set_budget(
            binding,
            lambda: 1_025,
            phase_id="fit_complete",
        )


def test_output_headroom_guard_covers_a_separate_output_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = SimpleNamespace(
        output_directory=tmp_path,
        storage_budget=SimpleNamespace(
            atomic_publication_headroom_bytes=1_024,
        ),
        test_only_allow_nonproduction_shape=False,
    )
    monkeypatch.setattr(
        experiment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1_024),
    )
    assert (
        _check_output_publication_headroom(
            binding,
            phase_id="artifact_temporary_sealed",
        )
        == 1_024
    )

    monkeypatch.setattr(
        experiment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1_023),
    )
    with pytest.raises(RuntimeError, match="lost frozen output.*headroom"):
        _check_output_publication_headroom(
            binding,
            phase_id="artifact_temporary_sealed",
        )


def _valid_no_pass_report() -> dict[str, object]:
    candidate_id = "candidate_0"
    decision = {
        "candidate_id": candidate_id,
        "candidate_artifact_sha256": "7" * 64,
        "evaluation_artifact_sha256": "8" * 64,
        "relative_fisher_pass": True,
        "oracle_gap_pass": True,
        "postnorm_nrmse_pass": True,
        "postnorm_cosine_pass": True,
        "sparse_renormalization_pass": True,
        "effective_support_pass": True,
        "global_support_budget_pass": False,
        "resource_pass": True,
        "minimum_rows_pass": True,
        "passed": False,
        "evaluation": {"role": "B"},
    }
    without_hash = {
        "schema": NO_PASS_REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "operator_boundary_only": True,
            "behavioral_nll_measured": False,
            "logical_speedup_claimed": False,
            "b_candidate_passed": False,
            "c_opened": False,
            "executable_artifact_published": False,
        },
        "model": {
            "model_id": "google/gemma-3-4b-it",
            "model_revision": "revision_0",
            "runtime_qualification_receipt_sha256": "1" * 64,
        },
        "protocol": {
            "protocol_id": "all_length_uniform_row_canary_v1",
            "rows_per_batch_and_invocation": ROWS_PER_BATCH_AND_INVOCATION,
            "expected_rows_per_role": EXPECTED_ROWS_PER_ROLE,
            "candidate_configs": [{"candidate_id": candidate_id}],
            "gates": {"gates_artifact_sha256": "2" * 64},
            "implementation_source_sha256s": {
                item: "3" * 64 for item in _IMPLEMENTATION_SOURCES
            },
            "preflight_receipt_sha256": "4" * 64,
        },
        "splits": {
            "A_split_artifact_sha256": "5" * 64,
            "B_split_artifact_sha256": "6" * 64,
            "A_receipt_sha256s": ["a" * 64],
            "B_receipt_sha256s": ["b" * 64],
        },
        "selection": {
            "A_development_artifact_sha256": "c" * 64,
            "B_selection_artifact_sha256": "d" * 64,
            "decisions": [decision],
        },
        "runtime_counters": {
            "role_batch_counts": [["A_chart_transport_fit", 45]],
        },
        "runtime_resources": {
            "prepublication_peak_resident_set_bytes": 1,
            "resident_set_cap_bytes": 2,
            "publication_guarded": True,
            "logical_speedup_claimed": False,
        },
    }
    return {
        **without_hash,
        "report_sha256": _json_sha256(_NO_PASS_DOMAIN, without_hash),
    }


def test_no_pass_report_canonicalizes_configs_to_selection_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = (
        SimpleNamespace(candidate_id="k8_r8_top2"),
        SimpleNamespace(candidate_id="k8_r16_top2"),
    )
    decisions = tuple(
        SimpleNamespace(candidate_id=value)
        for value in ("k8_r16_top2", "k8_r8_top2")
    )
    monkeypatch.setattr(
        experiment,
        "_config_payload",
        lambda value: {"candidate_id": value.candidate_id},
    )
    monkeypatch.setattr(
        experiment,
        "_decision_payload",
        lambda value: {"candidate_id": value.candidate_id},
    )
    monkeypatch.setattr(experiment, "_gate_payload", lambda _value: {})

    report = experiment._no_pass_report(
        preflight=SimpleNamespace(receipt_sha256="0" * 64),
        qualification=SimpleNamespace(
            model_id="google/gemma-3-4b-it",
            model_revision="revision_0",
            qualification_receipt_sha256="1" * 64,
        ),
        development=SimpleNamespace(
            a_split_artifact_sha256="2" * 64,
            a_receipt_sha256s=("3" * 64,),
            artifact_sha256="4" * 64,
        ),
        selection=SimpleNamespace(
            b_split_artifact_sha256="5" * 64,
            b_receipt_sha256s=("6" * 64,),
            artifact_sha256="7" * 64,
            decisions=decisions,
        ),
        configs=configs,
        gates=SimpleNamespace(),
        implementation_sources={},
        runtime_counters={},
        peak_resident_set_bytes=1,
        resident_set_cap_bytes=2,
    )

    assert [
        value["candidate_id"]
        for value in report["protocol"]["candidate_configs"]
    ] == [value.candidate_id for value in decisions]


def test_no_b_pass_report_is_canonical_and_checkable(tmp_path: Path) -> None:
    report = _valid_no_pass_report()
    report_path = tmp_path / "attention.json"
    artifact_path = tmp_path / "attention.pt"

    _publish_no_pass_report(
        report_path,
        report,
        artifact_path=artifact_path,
        publication_guard=_noop_publication_guard,
    )
    checked = check_attention_transport_outputs(
        artifact_path=artifact_path,
        report_path=report_path,
    )

    assert checked["status"] == "no_B_candidate_passed"
    assert checked["artifact_published"] is False
    assert json.loads(report_path.read_bytes()) == report


def test_success_check_surfaces_confirmation_gate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "attention.pt"
    report_path = tmp_path / "attention.json"
    artifact_path.write_bytes(b"artifact-present")
    monkeypatch.setattr(
        artifact_module,
        "strict_load_attention_transport_report",
        lambda *_args, **_kwargs: {
            "selection": {"C_passed": False},
            "report_sha256": "a" * 64,
        },
    )

    checked = check_attention_transport_outputs(
        artifact_path=artifact_path,
        report_path=report_path,
    )

    assert checked == {
        "status": "confirmation_complete",
        "c_passed": False,
        "artifact_published": True,
        "report_sha256": "a" * 64,
    }


def test_no_pass_report_rejects_minimal_and_contradictory_status(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "attention.json"
    minimal_unsigned = {
        "schema": NO_PASS_REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {"b_candidate_passed": False},
    }
    minimal = {
        **minimal_unsigned,
        "report_sha256": _json_sha256(_NO_PASS_DOMAIN, minimal_unsigned),
    }
    report_path.write_bytes(_canonical_json_bytes(minimal))
    with pytest.raises(ValueError, match="fields are invalid"):
        _strict_no_pass_report(report_path)

    contradictory = json.loads(_canonical_json_bytes(_valid_no_pass_report()))
    contradictory["scientific_status"]["b_candidate_passed"] = True
    unsigned = dict(contradictory)
    unsigned.pop("report_sha256")
    contradictory["report_sha256"] = _json_sha256(_NO_PASS_DOMAIN, unsigned)
    report_path.write_bytes(_canonical_json_bytes(contradictory))
    with pytest.raises(ValueError, match="terminal status is invalid"):
        _strict_no_pass_report(report_path)


def test_no_pass_report_rejects_report_and_parent_symlinks(
    tmp_path: Path,
) -> None:
    report = _valid_no_pass_report()
    target = tmp_path / "target.json"
    target.write_bytes(_canonical_json_bytes(report))
    report_link = tmp_path / "report.json"
    report_link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        _strict_no_pass_report(report_link)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent must be a real directory"):
        _publish_no_pass_report(
            linked_parent / "attention.json",
            report,
            artifact_path=linked_parent / "attention.pt",
            publication_guard=_noop_publication_guard,
        )


def test_no_pass_paths_reject_dangling_artifact_symlink(
    tmp_path: Path,
) -> None:
    report = _valid_no_pass_report()
    report_path = tmp_path / "attention.json"
    artifact_path = tmp_path / "attention.pt"
    artifact_path.symlink_to(tmp_path / "missing.pt")

    with pytest.raises(FileExistsError, match="artifact.*occupied"):
        _publish_no_pass_report(
            report_path,
            report,
            artifact_path=artifact_path,
            publication_guard=_noop_publication_guard,
        )
    assert not report_path.exists()

    artifact_path.unlink()
    _publish_no_pass_report(
        report_path,
        report,
        artifact_path=artifact_path,
        publication_guard=_noop_publication_guard,
    )
    artifact_path.symlink_to(tmp_path / "still-missing.pt")
    with pytest.raises(ValueError, match="occupied by a non-file"):
        check_attention_transport_outputs(
            artifact_path=artifact_path,
            report_path=report_path,
        )


def test_no_pass_publication_rolls_back_if_artifact_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "attention.json"
    artifact_path = tmp_path / "attention.pt"
    original_link = experiment.os.link

    def racing_link(*args: object, **kwargs: object) -> None:
        original_link(*args, **kwargs)
        artifact_path.write_bytes(b"foreign artifact")

    monkeypatch.setattr(experiment.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="artifact.*occupied"):
        _publish_no_pass_report(
            report_path,
            _valid_no_pass_report(),
            artifact_path=artifact_path,
            publication_guard=_noop_publication_guard,
        )

    assert not report_path.exists()
    assert artifact_path.read_bytes() == b"foreign artifact"


def test_no_pass_publication_guard_failure_rolls_back_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "attention.json"
    artifact_path = tmp_path / "attention.pt"
    phases: list[str] = []

    def guard(phase_id: str) -> None:
        phases.append(phase_id)
        if phase_id == "no_pass_publication_verified":
            raise RuntimeError("resident_cap_crossed")

    with pytest.raises(RuntimeError, match="resident_cap_crossed"):
        _publish_no_pass_report(
            report_path,
            _valid_no_pass_report(),
            artifact_path=artifact_path,
            publication_guard=guard,
        )

    assert phases == [
        "no_pass_temporary_sealed",
        "no_pass_publication_verified",
    ]
    assert not report_path.exists()
    assert not artifact_path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_no_pass_check_rejects_artifact_appearing_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "attention.json"
    artifact_path = tmp_path / "attention.pt"
    report_path.write_bytes(_canonical_json_bytes(_valid_no_pass_report()))
    original_load = experiment._strict_no_pass_report

    def racing_load(path: Path) -> dict[str, object]:
        loaded = original_load(path)
        artifact_path.write_bytes(b"foreign artifact")
        return loaded

    monkeypatch.setattr(experiment, "_strict_no_pass_report", racing_load)
    with pytest.raises(RuntimeError, match="appeared during no-pass check"):
        check_attention_transport_outputs(
            artifact_path=artifact_path,
            report_path=report_path,
        )
