from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_attention_transport_global_rank1_coupling_diagnostic as runner
from fisher_graph.gemma3_attention_transport_global_rank1_coupling_diagnostic import (
    REPORT_SCHEMA,
    _publish_report,
    _seal_report,
    _source_sha256s,
    _strict_report,
    check_global_rank1_coupling_A_diagnostic,
)


_GLOBAL_IDS = ("mutation.11", "mutation.23", "mutation.35")
_INVOCATION_IDS = tuple(
    sorted(f"mutation.{2 * index + 1}" for index in range(18))
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _metric(rows: int, relative: float, solver: str) -> dict[str, object]:
    zero = 10.0
    fisher = zero * relative
    return {
        "row_count": rows,
        "fisher_loss": fisher,
        "directional_mse": fisher / 2.0,
        "euclidean_mse": fisher / (2.0 * 0.64),
        "normalized_rmse": relative,
        "cosine": 0.9,
        "zero_predictor_fisher_loss": zero,
        "relative_fisher_loss": relative,
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "solver_kind": solver,
    }


def _fit_metric(
    rows: int,
    invocation_ids: tuple[str, ...],
    relative: float,
) -> dict[str, object]:
    zero = 10.0
    fisher = zero * relative
    return {
        "row_count": rows,
        "fisher_loss": fisher,
        "directional_mse": fisher,
        "euclidean_mse": 0.0,
        "normalized_rmse": relative,
        "cosine": 0.9,
        "zero_predictor_fisher_loss": zero,
        "relative_fisher_loss": relative,
        "active_count_limit": 2,
        "coefficient_ridge": 1.0e-8,
        "per_invocation_relative_fisher_loss": [
            [invocation_id, relative] for invocation_id in invocation_ids
        ],
    }


def _diagnostics() -> dict[str, object]:
    relative_by_solver = {
        "baseline": (0.4, 0.6, 0.8),
        "tied": (0.5, 0.3, 0.7),
        "fixed_support_untied": (0.2, 0.2, 0.4),
        "reselected_support_untied": (0.1, 0.15, 0.3),
    }
    per_site = [
        {
            "invocation_id": invocation_id,
            **{
                name: _metric(144, relative_by_solver[name][index], solver)
                for name, solver in runner._SOLVERS.items()
            },
        }
        for index, invocation_id in enumerate(_GLOBAL_IDS)
    ]
    aggregate_relative = {
        name: sum(values) / len(values)
        for name, values in relative_by_solver.items()
    }
    aggregate = {
        name: _metric(432, aggregate_relative[name], solver)
        for name, solver in runner._SOLVERS.items()
    }
    leave_one = [
        [
            invocation_id,
            sum(
                per_site[site_index][
                    "baseline" if site_index == index else "tied"
                ]["fisher_loss"]
                for site_index in range(len(per_site))
            )
            / sum(site["tied"]["zero_predictor_fisher_loss"] for site in per_site),
        ]
        for index, invocation_id in enumerate(_GLOBAL_IDS)
    ]
    baseline = aggregate_relative["baseline"]
    tied = aggregate_relative["tied"]
    fixed = aggregate_relative["fixed_support_untied"]
    reselected = aggregate_relative["reselected_support_untied"]
    tolerance = 1.0e-10
    return {
        "global_attention_invocation_ids": list(_GLOBAL_IDS),
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "ridge_parameterization": runner._RIDGE_PARAMETERIZATION,
        "aggregate": aggregate,
        "leave_one_site_residual_off_relative_fisher_losses": leave_one,
        "per_site": per_site,
        "summary": {
            "numerical_tolerance": tolerance,
            "baseline_minus_tied_relative_fisher_loss": baseline - tied,
            "tied_minus_fixed_support_untied_relative_fisher_loss": tied - fixed,
            "tied_minus_reselected_support_untied_relative_fisher_loss": (
                tied - reselected
            ),
            "reselected_minus_fixed_support_untied_relative_fisher_loss": (
                reselected - fixed
            ),
            "fixed_support_untied_not_worse_than_tied": fixed <= tied + tolerance,
            "reselected_support_untied_not_worse_than_fixed": (
                reselected <= fixed + tolerance
            ),
            "reselected_support_untied_not_worse_than_baseline": (
                reselected <= baseline + tolerance
            ),
            "tied_nonworsening_at_every_global_site": all(
                site["tied"]["relative_fisher_loss"]
                <= site["baseline"]["relative_fisher_loss"] + tolerance
                for site in per_site
            ),
            "reselected_untied_strictly_better_than_tied_site_ids": [
                site["invocation_id"]
                for site in per_site
                if site["reselected_support_untied"]["relative_fisher_loss"]
                < site["tied"]["relative_fisher_loss"] - tolerance
            ],
        },
        "retention": {
            "contains_oracle_coefficients": False,
            "contains_targets": False,
            "contains_score_gradients": False,
            "contains_raw_activations": False,
        },
    }


def _receipts() -> tuple[list[str], list[str]]:
    values = sorted(
        [_sha(f"receipt-{index}") for index in range(45)],
        key=lambda value: hashlib.sha256(
            runner._PARTITION_DOMAIN + value.encode("ascii")
        ).hexdigest(),
    )
    return values[:36], values[36:]


def _runtime_counters() -> dict[str, object]:
    return {
        "role_batch_counts": [
            ["A_chart_transport_fit", 45],
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


def _unsealed_report(root: Path) -> dict[str, object]:
    fit_receipts, dev_receipts = _receipts()
    bank_sha = _sha("rank1-bank")
    return {
        "schema": REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "A_only_development_diagnostic": True,
            "A_fitted": True,
            "B_previously_exposed": True,
            "B_influenced_architecture_design": True,
            "B_opened_by_this_run": False,
            "B_used_for_training": False,
            "C_ever_opened": False,
            "confirmation_claimed": False,
            "behavioral_nll_measured": False,
            "operator_boundary_only": True,
            "global_attention_rank1_enforced": True,
            "global_top2_support_enforced": True,
            "untied_oracle_is_deployable_router": False,
            "executable_artifact_published": False,
            "logical_speedup_claimed": False,
        },
        "exposure_binding": {
            "prior_v1_report_sha256": runner._PRIOR_V1_REPORT_SHA256,
            "prior_v1_report_file_sha256": runner._PRIOR_V1_REPORT_FILE_SHA256,
            "prior_v1_report_path": str(runner._PRIOR_V1_REPORT_PATH),
            "prior_sparse_A_report_sha256": runner._PRIOR_SPARSE_A_REPORT_SHA256,
            "prior_sparse_A_report_file_sha256": (
                runner._PRIOR_SPARSE_A_REPORT_FILE_SHA256
            ),
            "prior_sparse_A_report_path": str(runner._PRIOR_SPARSE_A_REPORT_PATH),
            "prior_global_B_report_sha256": runner._PRIOR_GLOBAL_B_REPORT_SHA256,
            "prior_global_B_report_file_sha256": (
                runner._PRIOR_GLOBAL_B_REPORT_FILE_SHA256
            ),
            "prior_global_B_report_path": str(runner._PRIOR_GLOBAL_B_REPORT_PATH),
            "prior_hierarchical_B_report_sha256": (
                runner._PRIOR_HIERARCHICAL_B_REPORT_SHA256
            ),
            "prior_hierarchical_B_report_file_sha256": (
                runner._PRIOR_HIERARCHICAL_B_REPORT_FILE_SHA256
            ),
            "prior_hierarchical_B_report_path": str(
                runner._PRIOR_HIERARCHICAL_B_REPORT_PATH
            ),
            "prior_hierarchical_bank_artifact_sha256": (
                runner._PRIOR_HIERARCHICAL_BANK_SHA256
            ),
            "lineage_reads_bound_reports_only": True,
            "B_or_C_corpus_opened_by_this_run": False,
            "interpretation": (
                "architecture_informed_A_only_structural_diagnostic_not_confirmation"
            ),
        },
        "model": {
            "model_id": "google/gemma-3-270m",
            "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "hidden_width": 640,
            "runtime_qualification_receipt_sha256": _sha("qualification"),
        },
        "protocol": {
            "protocol_id": "A_only_global_rank1_coupling_36_fit_9_dev_v1",
            "fit_receipt_count": 36,
            "dev_receipt_count": 9,
            "fit_row_count": 10_368,
            "dev_row_count": 2_592,
            "global_dev_row_count": 432,
            "B_and_C_capture_forbidden": True,
            "partition_policy": (
                "domain_hash_sorted_whole_A_capture_receipts_36_fit_9_dev"
            ),
            "bank_fit": {
                "generator_count": 8,
                "generator_rank": 8,
                "active_count": 2,
                "dense_warm_start_steps": 96,
                "sparse_alternating_rounds": 4,
                "sparse_steps_per_round": 32,
                "hierarchy_alternating_rounds": 4,
                "global_rank_one_alternating_rounds": 4,
                "global_rank_one_steps_per_round": 48,
                "learning_rate": 0.01,
                "euclidean_floor": 0.001,
                "fit_oracle_coefficient_ridge": 1.0e-8,
                "hierarchy_ridge": 1.0e-8,
                "global_rank_one_residual_ridge": 1.0e-8,
                "gradient_clip_norm": 10.0,
                "gauge_floor": 1.0e-12,
                "acceptance_metric": (
                    "candidate_round_requires_strict_all_A_and_"
                    "global_attention_improvement"
                ),
                "learned_objects_fit_role": "A_fit_only",
            },
            "diagnostic_oracles": {
                "evaluation_role": "A_dev_only",
                "global_attention_invocation_ids": list(_GLOBAL_IDS),
                "active_generator_count_limit": 2,
                "coefficient_ridge": 0.0,
                "ridge_parameterization": runner._RIDGE_PARAMETERIZATION,
                "support_semantics": (
                    "shared_generator_top2_with_separate_base_and_"
                    "residual_amplitudes"
                ),
                "coefficients_retained": False,
                "oracle_ceiling_only": True,
            },
            "preflight_receipt_sha256": _sha("preflight"),
            "dense_lexical_parity_receipt_sha256": _sha("dense-parity"),
            "implementation_source_sha256s": _source_sha256s(root),
        },
        "A_partition": {
            "A_split_artifact_sha256": _sha("A-split"),
            "partition_policy": (
                "domain_hash_sorted_whole_A_capture_receipts_36_fit_9_dev"
            ),
            "fit_receipt_sha256s": fit_receipts,
            "dev_receipt_sha256s": dev_receipts,
            "fit_row_count": 10_368,
            "dev_row_count": 2_592,
            "fit_and_dev_disjoint": True,
            "final_firewall_state": {
                "opened_roles": ["A_chart_transport_fit"],
                "current_role": "A_chart_transport_fit",
                "next_role": "A_chart_transport_fit",
                "freeze_receipt_count": 0,
                "B_opened_by_this_run": False,
                "C_ever_opened": False,
            },
        },
        "candidate": {
            "candidate_id": "k8_r8_global_attention_rank1_A_fit_coupling_diagnostic",
            "bank_artifact_sha256": bank_sha,
            "pre_diagnostic_bank_artifact_sha256": bank_sha,
            "post_diagnostic_bank_artifact_sha256": bank_sha,
            "hierarchical_bank_artifact_sha256": _sha("hierarchical-bank"),
            "dense_warm_start_bank_artifact_sha256": _sha("dense-bank"),
            "generator_bank_parameters": 93_248,
            "hierarchical_bank_parameters": 83_008,
            "global_rank_one_parameters": 10_240,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
            "diagnostic_contains_oracle_coefficients": False,
            "diagnostic_contains_targets_or_gradients": False,
            "A_fit": {
                "initial_all_A_sparse_oracle": _fit_metric(
                    10_368, _INVOCATION_IDS, 0.6
                ),
                "final_all_A_sparse_oracle": _fit_metric(
                    10_368, _INVOCATION_IDS, 0.4
                ),
                "initial_global_attention_sparse_oracle": _fit_metric(
                    1_728, _GLOBAL_IDS, 0.7
                ),
                "final_global_attention_sparse_oracle": _fit_metric(
                    1_728, _GLOBAL_IDS, 0.5
                ),
                "round_all_A_relative_fisher_losses": [
                    0.6,
                    0.55,
                    0.5,
                    0.45,
                    0.4,
                ],
                "round_global_attention_relative_fisher_losses": [
                    0.7,
                    0.65,
                    0.6,
                    0.55,
                    0.5,
                ],
            },
            "executable_artifact_published": False,
        },
        "diagnostics": _diagnostics(),
        "runtime_counters": _runtime_counters(),
        "runtime_resources": {
            "prepublication_peak_resident_set_bytes": 1024,
            "resident_set_cap_bytes": 8 * 1024**3,
            "publication_guarded": True,
        },
    }


def _noop_guard(_phase_id: str) -> None:
    return None


def test_cli_import_is_standard_library_only() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/gemma3_attention_transport_global_rank1_coupling_diagnostic.py"
    )
    code = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('coupling_preflight', {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = (
    'torch',
    'numpy',
    'transformers',
    'fisher_graph.gemma3_attention_transport_compiler',
    'fisher_graph.gemma3_attention_transport_runtime',
    'fisher_graph.gemma3_attention_transport_experiment',
    'fisher_graph.gemma3_attention_transport_experiment_core',
    'fisher_graph.gemma3_attention_transport_global_rank1',
    'fisher_graph.gemma3_chart_native_neighbor_experiment',
    'fisher_graph.gemma3_chart_native_neighbor_fresh_panels',
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


def test_public_surface_has_exactly_one_A_capture_and_no_role_advance() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/gemma3_attention_transport_global_rank1_coupling_diagnostic.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    public = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    capture_roles: list[str] = []
    called_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        called_names.append(called)
        if called == "_capture_role":
            role = next(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "runtime_role"
            )
            assert isinstance(role, ast.Name)
            capture_roles.append(role.id)

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert public == {
        "run_global_rank1_coupling_A_diagnostic",
        "check_global_rank1_coupling_A_diagnostic",
        "main",
    }
    assert capture_roles == ["FIT_ROLE"]
    assert "DISCOVERY_ROLE" not in names
    assert "CONFIRMATION_ROLE" not in names
    assert "freeze_current" not in called_names


def test_report_is_canonical_closed_source_free_and_never_overwrites(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "coupling.json"
    _publish_report(output, report, publication_guard=_noop_guard)

    assert output.read_bytes() == runner._canonical_json_bytes(report)
    assert _strict_report(output) == report
    checked = check_global_rank1_coupling_A_diagnostic(
        output, repository_root=root
    )
    assert checked["report_sha256"] == report["report_sha256"]
    assert checked["B_opened_by_this_run"] is False
    assert checked["C_ever_opened"] is False
    candidate = report["candidate"]
    assert candidate["bank_artifact_sha256"] == candidate[
        "pre_diagnostic_bank_artifact_sha256"
    ] == candidate["post_diagnostic_bank_artifact_sha256"]
    assert all(
        value is False
        for value in report["diagnostics"]["retention"].values()
    )
    with pytest.raises(FileExistsError):
        _publish_report(output, report, publication_guard=_noop_guard)


def _set_relative(metric: dict[str, object], relative: float) -> None:
    metric["relative_fisher_loss"] = relative
    metric["fisher_loss"] = 10.0 * relative
    metric["directional_mse"] = 5.0 * relative
    metric["euclidean_mse"] = 5.0 * relative / 0.64


def _shift_aggregate_components(value: dict[str, object]) -> None:
    metric = value["diagnostics"]["aggregate"]["baseline"]
    metric["directional_mse"] += 0.064
    metric["euclidean_mse"] -= 0.1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["scientific_status"].__setitem__(
                "B_opened_by_this_run", True
            ),
            "scientific status",
        ),
        (
            lambda value: value["candidate"].__setitem__(
                "post_diagnostic_bank_artifact_sha256", _sha("mutated-bank")
            ),
            "candidate lineage",
        ),
        (
            lambda value: value["candidate"].__setitem__(
                "diagnostic_contains_oracle_coefficients", True
            ),
            "candidate lineage",
        ),
        (
            lambda value: value["diagnostics"]["retention"].__setitem__(
                "contains_targets", True
            ),
            "catalog",
        ),
        (
            lambda value: value["protocol"]["diagnostic_oracles"].__setitem__(
                "support_semantics", "scalar_channel_top2"
            ),
            "oracle protocol",
        ),
        (
            lambda value: value["diagnostics"]["per_site"].reverse(),
            "per-site catalog",
        ),
        (
            lambda value: _set_relative(
                value["diagnostics"]["per_site"][0][
                    "fixed_support_untied"
                ],
                0.6,
            ),
            "nested oracle order",
        ),
        (
            lambda value: _set_relative(
                value["diagnostics"]["aggregate"]["baseline"], 0.61
            ),
            "aggregate.*closure",
        ),
        (
            _shift_aggregate_components,
            "aggregate metric closure",
        ),
        (
            lambda value: value["diagnostics"][
                "leave_one_site_residual_off_relative_fisher_losses"
            ][0].__setitem__(1, 0.99),
            "leave-one-site closure",
        ),
    ],
)
def test_report_rejects_lineage_catalog_nesting_and_closure_tampering(
    mutate: object,
    message: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    value = _unsealed_report(root)
    mutate(value)
    with pytest.raises(ValueError, match=message):
        _seal_report(value)


def test_check_rejects_source_drift(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    value = _unsealed_report(root)
    first = next(iter(value["protocol"]["implementation_source_sha256s"]))
    value["protocol"]["implementation_source_sha256s"][first] = "0" * 64
    report = _seal_report(value)
    output = tmp_path / "source-drift.json"
    _publish_report(output, report, publication_guard=_noop_guard)

    with pytest.raises(ValueError, match="sources drifted"):
        check_global_rank1_coupling_A_diagnostic(output, repository_root=root)


def test_strict_reader_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    source = tmp_path / "source.json"
    source.write_bytes(runner._canonical_json_bytes(report))

    symlink = tmp_path / "symlink.json"
    os.symlink(source, symlink)
    with pytest.raises(ValueError, match="regular file"):
        _strict_report(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="identity or size"):
        _strict_report(source)
    with pytest.raises(ValueError, match="identity or size"):
        _strict_report(hardlink)


def test_publication_rejects_symlink_parent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    os.symlink(real, linked)

    with pytest.raises(ValueError, match="real directory"):
        _publish_report(
            linked / "report.json", report, publication_guard=_noop_guard
        )


def test_publication_race_preserves_foreign_replacement(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "report.json"
    owned = tmp_path / "owned.json"

    def guard(phase_id: str) -> None:
        if phase_id == "global_rank1_coupling_publication_verified":
            output.rename(owned)
            output.write_bytes(b"foreign replacement")

    with pytest.raises((RuntimeError, ValueError)):
        _publish_report(output, report, publication_guard=guard)

    assert output.read_bytes() == b"foreign replacement"
    assert owned.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publication_rejects_temporary_hardlink_without_removing_it(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "report.json"
    foreign = tmp_path / "foreign-hardlink.json"

    def guard(phase_id: str) -> None:
        if phase_id == "global_rank1_coupling_temporary_sealed":
            temporary = next(tmp_path.glob(".*.tmp"))
            os.link(temporary, foreign)

    with pytest.raises(RuntimeError, match="temporary identity"):
        _publish_report(output, report, publication_guard=guard)

    assert not output.exists()
    assert foreign.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_run_captures_and_partitions_A_without_advancing_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fisher_graph.gemma3_attention_transport_experiment as experiment
    import fisher_graph.gemma3_attention_transport_experiment_core as core
    import fisher_graph.gemma3_attention_transport_global_rank1 as rank_one
    import fisher_graph.gemma3_attention_transport_runtime as runtime_module
    import fisher_graph.gemma3_chart_native_neighbor_experiment as neighbor
    import fisher_graph.gemma3_chart_native_neighbor_fresh_panels as panels

    root = Path(__file__).resolve().parents[1]
    fit_role = neighbor.FIT_ROLE
    events: list[str] = []
    panel_instances: list[object] = []
    captured_report: dict[str, object] = {}
    fit_receipts, dev_receipts = _receipts()
    all_receipts = fit_receipts + dev_receipts
    records = tuple(
        SimpleNamespace(
            base_capture_receipt_sha256=receipt,
            capture=SimpleNamespace(invocation_id=invocation_id, row_count=16),
        )
        for receipt in all_receipts
        for invocation_id in _INVOCATION_IDS
    )

    class FakePanelSource:
        def __init__(self, **_kwargs: object) -> None:
            self.opened_roles: tuple[str, ...] = ()
            self.current_role: str | None = None
            self.next_role: str | None = fit_role
            self.freeze_receipts: dict[str, object] = {}
            panel_instances.append(self)

    class FakeRuntime:
        def __init__(self) -> None:
            self.counters = SimpleNamespace(payload=_runtime_counters)

        def runtime_qualification_receipt(self) -> object:
            return SimpleNamespace(
                model_id="google/gemma-3-270m",
                model_revision="9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
                hidden_width=640,
                qualification_receipt_sha256=_sha("qualification"),
            )

        def close(self) -> None:
            events.append("runtime-close")

    preflight = SimpleNamespace(
        binding=SimpleNamespace(
            repository_root=root,
            storage_budget=SimpleNamespace(
                maximum_resident_memory_bytes=8 * 1024**3
            ),
        ),
        fresh_manifest={},
        fresh_receipt=object(),
        opaque_panel_receipt=object(),
        receipt_sha256=_sha("preflight"),
        validate_receipt=lambda: events.append("preflight"),
    )

    def fake_capture_role(**kwargs: object) -> tuple[tuple[object, ...], str]:
        panel = kwargs["panel_source"]
        assert kwargs["runtime_role"] == fit_role
        assert kwargs["core_role"] == "A"
        assert panel.opened_roles == ()
        panel.opened_roles = (fit_role,)
        panel.current_role = fit_role
        panel.next_role = fit_role
        events.append("capture-A")
        return records, _sha("dense-parity")

    bank_sha = _sha("rank1-bank")
    fake_bank = SimpleNamespace(
        artifact_sha256=bank_sha,
        hierarchical_bank_sha256=_sha("hierarchical-bank"),
        stored_parameter_count=93_248,
        global_rank_one_parameter_count=10_240,
        state_dict=lambda: {
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
        },
    )
    initial_all = SimpleNamespace(**_fit_metric(10_368, _INVOCATION_IDS, 0.6))
    final_all = SimpleNamespace(**_fit_metric(10_368, _INVOCATION_IDS, 0.4))
    initial_global = SimpleNamespace(**_fit_metric(1_728, _GLOBAL_IDS, 0.7))
    final_global = SimpleNamespace(**_fit_metric(1_728, _GLOBAL_IDS, 0.5))
    for value in (initial_all, final_all, initial_global, final_global):
        value.output_metrics = SimpleNamespace(
            row_count=value.row_count,
            fisher_loss=value.fisher_loss,
            directional_mse=value.directional_mse,
            euclidean_mse=value.euclidean_mse,
            normalized_rmse=value.normalized_rmse,
            cosine=value.cosine,
        )
    fake_fit = SimpleNamespace(
        bank=fake_bank,
        hierarchical_fit=SimpleNamespace(
            bank=SimpleNamespace(stored_parameter_count=83_008),
            dense_warm_start=SimpleNamespace(
                bank=SimpleNamespace(artifact_sha256=_sha("dense-bank"))
            ),
        ),
        initial_fit_oracle=initial_all,
        final_fit_oracle=final_all,
        initial_global_fit_oracle=initial_global,
        final_global_fit_oracle=final_global,
        fit_round_relative_fisher_losses=(0.6, 0.55, 0.5, 0.45, 0.4),
        fit_round_global_relative_fisher_losses=(0.7, 0.65, 0.6, 0.55, 0.5),
    )

    def fake_fit_bank(captures: tuple[object, ...], **_kwargs: object) -> object:
        assert len(captures) == 36 * 18
        assert events[-1] == "capture-A"
        events.append("fit-A")
        return fake_fit

    def fake_diagnostic(
        _bank: object, captures: tuple[object, ...], **_kwargs: object
    ) -> object:
        assert len(captures) == 9 * 18
        assert events[-1] == "fit-A"
        events.append("diagnose-A-dev")
        return object()

    def fake_publish(
        _path: Path,
        report: dict[str, object],
        *,
        publication_guard: object,
    ) -> None:
        publication_guard("global_rank1_coupling_temporary_sealed")
        publication_guard("global_rank1_coupling_publication_verified")
        captured_report.update(report)
        events.append("publish")

    monkeypatch.setattr(neighbor, "static_preflight", lambda _binding: preflight)
    monkeypatch.setattr(
        neighbor, "validate_runtime_qualification", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        neighbor,
        "observe_phase_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            peak_resident_set_bytes=1024
        ),
    )
    monkeypatch.setattr(experiment, "_capture_role", fake_capture_role)
    monkeypatch.setattr(
        experiment, "_check_output_publication_headroom", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        experiment,
        "_check_resident_set_budget",
        lambda *_args, **_kwargs: 1024,
    )
    monkeypatch.setattr(
        core,
        "_open_split_panel",
        lambda _records, *, role: SimpleNamespace(
            artifact_sha256=_sha(f"{role}-split")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_pinned_gemma3_attention_transport_capture_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(panels, "SequentialPanelSource", FakePanelSource)
    monkeypatch.setattr(
        rank_one,
        "fit_global_rank_one_attention_transport_generator_bank",
        fake_fit_bank,
    )
    monkeypatch.setattr(
        rank_one, "evaluate_global_rank_one_coupling_diagnostic", fake_diagnostic
    )
    monkeypatch.setattr(runner, "_validate_prior_reports", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_source_sha256s",
        lambda _root: {path: _sha(path) for path in runner._SOURCE_PATHS},
    )
    monkeypatch.setattr(
        runner, "_diagnostic_payload", lambda _value: deepcopy(_diagnostics())
    )
    monkeypatch.setattr(runner, "_publish_report", fake_publish)

    result = runner.run_global_rank1_coupling_A_diagnostic(
        binding=object(), report_path=tmp_path / "report.json"
    )

    assert result["B_opened_by_this_run"] is False
    assert result["C_ever_opened"] is False
    assert events.count("capture-A") == 1
    assert "diagnose-A-dev" in events
    assert not any("capture-B" in event or "capture-C" in event for event in events)
    assert events.index("capture-A") < events.index("fit-A")
    assert events.index("fit-A") < events.index("diagnose-A-dev")
    assert events.index("diagnose-A-dev") < events.index("publish")
    assert len(panel_instances) == 1
    assert panel_instances[0].opened_roles == (fit_role,)
    assert panel_instances[0].current_role == fit_role
    assert panel_instances[0].next_role == fit_role
    assert panel_instances[0].freeze_receipts == {}
    assert len(captured_report["A_partition"]["fit_receipt_sha256s"]) == 36
    assert len(captured_report["A_partition"]["dev_receipt_sha256s"]) == 9
    assert captured_report["candidate"]["bank_artifact_sha256"] == (
        captured_report["candidate"]["pre_diagnostic_bank_artifact_sha256"]
    ) == captured_report["candidate"]["post_diagnostic_bank_artifact_sha256"]
