from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_attention_transport_targeted_site_rank2_a_diagnostic as runner


_GLOBAL_IDS = ("mutation.11", "mutation.23", "mutation.35")
_TARGET_IDS = _GLOBAL_IDS[:2]
_LEXICAL_IDS = tuple(
    sorted(f"mutation.{2 * index + 1}" for index in range(18))
)
_NUMERIC_IDS = tuple(f"mutation.{2 * index + 1}" for index in range(18))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _output_metrics(rows: int, fisher: float) -> SimpleNamespace:
    return SimpleNamespace(
        row_count=rows,
        fisher_loss=fisher,
        directional_mse=fisher / 2.0,
        euclidean_mse=fisher / 3.0,
        normalized_rmse=max(fisher, 0.0),
        cosine=0.8,
    )


def _sparse_evaluation(
    rows: int,
    fisher: float,
    zero: float,
    invocation_ids: tuple[str, ...],
    *,
    active_count: int = 2,
    ridge: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_metrics=_output_metrics(rows, fisher),
        zero_predictor_fisher_loss=zero,
        relative_fisher_loss=fisher / zero,
        active_count_limit=active_count,
        coefficient_ridge=ridge,
        per_invocation_relative_fisher_loss=tuple(
            (invocation_id, fisher / zero) for invocation_id in invocation_ids
        ),
    )


def _grouped_evaluation(
    rows: int,
    relative: float,
    zero: float,
    solver_kind: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_metrics=_output_metrics(rows, zero * relative),
        zero_predictor_fisher_loss=zero,
        relative_fisher_loss=relative,
        active_generator_count_limit=2,
        coefficient_ridge=0.0,
        solver_kind=solver_kind,
        row_count=rows,
        fisher_loss=zero * relative,
    )


def _all_layer_bank(
    branch_index: int,
) -> SimpleNamespace:
    per_invocation = []
    for index, invocation_id in enumerate(_NUMERIC_IDS):
        zero = float(index + 1)
        base_relative = 0.05 + 0.005 * index
        if invocation_id in _TARGET_IDS:
            relative = base_relative + 0.04 * (3 - branch_index)
        else:
            # Every global-only residual bank is structurally identical at all
            # local sites, so the synthetic payload preserves that invariant.
            relative = base_relative
        per_invocation.append(
            (
                invocation_id,
                _sparse_evaluation(144, zero * relative, zero, (invocation_id,)),
            )
        )

    def aggregate(selected: list[tuple[str, SimpleNamespace]]) -> SimpleNamespace:
        rows = sum(item.output_metrics.row_count for _key, item in selected)
        fisher = sum(
            item.output_metrics.row_count * item.output_metrics.fisher_loss
            for _key, item in selected
        ) / rows
        zero = sum(
            item.output_metrics.row_count * item.zero_predictor_fisher_loss
            for _key, item in selected
        ) / rows
        return _sparse_evaluation(
            rows,
            fisher,
            zero,
            tuple(sorted(key for key, _item in selected)),
        )

    local = [item for item in per_invocation if item[0] not in _GLOBAL_IDS]
    global_values = [item for item in per_invocation if item[0] in _GLOBAL_IDS]
    return SimpleNamespace(
        all_layers=aggregate(per_invocation),
        local_attention=aggregate(local),
        global_attention=aggregate(global_values),
        # Deliberately numeric, matching the hierarchy rather than the runner's
        # lexical report order.  The payload must normalize this safely.
        per_invocation=tuple(per_invocation),
    )


def _all_layer_payload() -> dict[str, object]:
    return runner._all_layer_payload(
        {
            variant: _all_layer_bank(index)
            for index, variant in enumerate(runner._ALL_LAYER_VARIANTS)
        }
    )


def _coupling_branch(
    tied_values: tuple[float, float, float],
) -> SimpleNamespace:
    sites = []
    for index, invocation_id in enumerate(_GLOBAL_IDS):
        zero = float(index + 1)
        tied = tied_values[index]
        sites.append(
            SimpleNamespace(
                invocation_id=invocation_id,
                tied=_grouped_evaluation(
                    144,
                    tied,
                    zero,
                    "exact_generator_support_topk_tied_base_plus_residual_v1",
                ),
                fixed_support_untied=_grouped_evaluation(
                    144,
                    tied - 0.02,
                    zero,
                    "exact_untied_amplitudes_on_tied_generator_support_v1",
                ),
                reselected_support_untied=_grouped_evaluation(
                    144,
                    tied - 0.03,
                    zero,
                    "exact_grouped_generator_support_topk_untied_amplitudes_v1",
                ),
            )
        )

    def aggregate(attribute: str, solver_kind: str) -> SimpleNamespace:
        fisher = sum(
            site.__getattribute__(attribute).fisher_loss * 144 for site in sites
        ) / 432
        zero = sum(
            site.__getattribute__(attribute).zero_predictor_fisher_loss * 144
            for site in sites
        ) / 432
        return _grouped_evaluation(432, fisher / zero, zero, solver_kind)

    return SimpleNamespace(
        global_attention_invocation_ids=_GLOBAL_IDS,
        tied=aggregate(
            "tied",
            "exact_generator_support_topk_tied_base_plus_residual_v1",
        ),
        fixed_support_untied=aggregate(
            "fixed_support_untied",
            "exact_untied_amplitudes_on_tied_generator_support_v1",
        ),
        reselected_support_untied=aggregate(
            "reselected_support_untied",
            "exact_grouped_generator_support_topk_untied_amplitudes_v1",
        ),
        per_site=tuple(sites),
    )


def _fit_evaluation(
    rows: int,
    invocation_ids: tuple[str, ...],
    relative: float,
) -> SimpleNamespace:
    zero = 10.0
    return SimpleNamespace(
        output_metrics=_output_metrics(rows, zero * relative),
        zero_predictor_fisher_loss=zero,
        relative_fisher_loss=relative,
        active_count_limit=2,
        coefficient_ridge=1.0e-8,
        per_invocation_relative_fisher_loss=tuple(
            (invocation_id, relative) for invocation_id in invocation_ids
        ),
    )


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


def test_module_import_is_stdlib_only_and_public_surface_is_A_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2_a_diagnostic.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            assert node.module is not None
            assert node.module.split(".")[0] in sys.stdlib_module_names

    public = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    capture_roles = []
    called_names = []
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
            runtime_role = next(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "runtime_role"
            )
            assert isinstance(runtime_role, ast.Name)
            capture_roles.append(runtime_role.id)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert public == {
        "run_targeted_site_rank2_A_diagnostic",
        "check_targeted_site_rank2_A_diagnostic",
        "main",
    }
    assert capture_roles == ["FIT_ROLE"]
    assert "DISCOVERY_ROLE" not in names
    assert "CONFIRMATION_ROLE" not in names
    assert "freeze_current" not in called_names
    assert "advance" not in called_names

    code = """
import sys
import fisher_graph
baseline = set(sys.modules)
import fisher_graph.gemma3_attention_transport_targeted_site_rank2_a_diagnostic
new_modules = set(sys.modules) - baseline
forbidden_heavy = (
    'torch',
    'mlx',
    'transformers',
)
forbidden_experiment = (
    'fisher_graph.gemma3_attention_transport_targeted_site_rank2',
    'fisher_graph.gemma3_chart_native_neighbor_fresh_panels',
)
assert not [name for name in forbidden_heavy if name in new_modules]
assert not [name for name in forbidden_experiment if name in sys.modules]
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_exact_predecessor_report_bank_split_and_source_pins() -> None:
    root = Path(__file__).resolve().parents[1]
    report = runner._validate_prior_site_rank1_report(root)
    raw = (root / runner._PRIOR_SITE_RANK1_REPORT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == runner._PRIOR_SITE_RANK1_REPORT_FILE_SHA256
    assert report["report_sha256"] == runner._PRIOR_SITE_RANK1_REPORT_SHA256
    assert report["candidate"]["bank_artifact_sha256"] == runner._PRIOR_SITE_RANK1_BANK_SHA256
    assert report["A_partition"]["A_split_artifact_sha256"] == runner._PRIOR_SITE_RANK1_A_SPLIT_SHA256
    for relative, expected in (
        (runner._PRIOR_SITE_RANK1_CORE_SOURCE_PATH, runner._PRIOR_SITE_RANK1_CORE_SOURCE_SHA256),
        (runner._PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH, runner._PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256),
    ):
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


def test_whole_receipt_partition_is_deterministic_disjoint_and_run_scoped() -> None:
    fit_receipts, dev_receipts = _receipts()
    records = tuple(
        SimpleNamespace(
            base_capture_receipt_sha256=receipt,
            capture=SimpleNamespace(row_count=16),
        )
        for receipt in reversed(fit_receipts + dev_receipts)
        for _invocation_id in _LEXICAL_IDS
    )
    fit, dev, observed_fit, observed_dev = runner._partition_records(records)
    assert observed_fit == tuple(fit_receipts)
    assert observed_dev == tuple(dev_receipts)
    assert len(fit) == 36 * 18
    assert len(dev) == 9 * 18
    assert sum(record.capture.row_count for record in fit) == 10_368
    assert sum(record.capture.row_count for record in dev) == 2_592
    assert set(observed_fit).isdisjoint(observed_dev)


def test_all_layer_payload_normalizes_order_closes_burden_and_keeps_global_gate() -> None:
    payload = _all_layer_payload()
    runner._validate_all_layer(payload)
    assert [layer["invocation_id"] for layer in payload["per_layer"]] == list(_LEXICAL_IDS)
    assert payload["summary"]["primary_fidelity_gate_scope"] == (
        "global_attention_targeted_tied_oracle_reported_separately"
    )
    ranking = payload["absolute_burden_ranking"]
    masses = [entry["weighted_absolute_fisher_burden"] for entry in ranking]
    assert masses == sorted(masses, reverse=True)
    assert sum(entry["absolute_fisher_burden_fraction"] for entry in ranking) == pytest.approx(1.0)
    assert all(
        payload["aggregate"]["local_attention"][variant]["fisher_loss"]
        == payload["aggregate"]["local_attention"]["site_rank_one_parent"]["fisher_loss"]
        for variant in runner._ALL_LAYER_VARIANTS
    )

    tampered = deepcopy(payload)
    tampered["absolute_burden_ranking"].reverse()
    with pytest.raises(ValueError, match="burden ranking"):
        runner._validate_all_layer(tampered)
    diluted = deepcopy(payload)
    diluted["summary"]["primary_fidelity_gate_scope"] = "all_layers"
    with pytest.raises(ValueError, match="closure"):
        runner._validate_all_layer(diluted)


def test_global_coupling_uses_total_residual_top2_generator_support_and_closes() -> None:
    payload = runner._coupling_payload(
        _coupling_branch((0.22, 0.50, 0.12)),
        _coupling_branch((0.21, 0.48, 0.12)),
        _coupling_branch((0.15, 0.30, 0.12)),
    )
    runner._validate_coupling(payload)
    assert payload["untied_amplitude_semantics"] == (
        "separate_hierarchical_base_and_total_site_residual_amplitudes_"
        "top2_counted_by_generator_identity_v1"
    )
    assert payload["active_generator_count_limit"] == 2
    assert payload["coefficient_ridge"] == 0.0
    assert payload["summary"]["continued_minus_targeted_tied_capacity_delta"] > 0.0
    target_tied = payload["aggregate"]["targeted_rank_2_2_1_tied"]["fisher_loss"]
    target_fixed = payload["aggregate"]["targeted_rank_2_2_1_fixed_support_untied"]["fisher_loss"]
    target_reselected = payload["aggregate"]["targeted_rank_2_2_1_reselected_support_untied"]["fisher_loss"]
    assert target_reselected <= target_fixed <= target_tied

    tampered = deepcopy(payload)
    tampered["retention"]["contains_oracle_coefficients"] = True
    with pytest.raises(ValueError, match="closure"):
        runner._validate_coupling(tampered)


def test_warmup_provenance_payload_is_complete_and_strictly_validated() -> None:
    fit = SimpleNamespace(
        initial_fit_oracle=_fit_evaluation(10_368, _LEXICAL_IDS, 0.3),
        final_fit_oracle=_fit_evaluation(10_368, _LEXICAL_IDS, 0.2),
        initial_global_fit_oracle=_fit_evaluation(1_728, _GLOBAL_IDS, 0.4),
        final_global_fit_oracle=_fit_evaluation(1_728, _GLOBAL_IDS, 0.25),
        fit_round_relative_fisher_losses=(0.3, 0.27, 0.24, 0.21, 0.2),
        fit_round_global_relative_fisher_losses=(0.4, 0.36, 0.32, 0.28, 0.25),
        fit_round_target_site_relative_fisher_losses=tuple(
            (key, (0.4, 0.36, 0.32, 0.28, 0.25)) for key in _TARGET_IDS
        ),
        warmup_fixed_support_relative_decreases=(
            (_TARGET_IDS[0], 0.1),
            (_TARGET_IDS[1], 0.2),
        ),
        warmup_accepted_learning_rates=(
            (_TARGET_IDS[0], 0.005),
            (_TARGET_IDS[1], 0.01),
        ),
        warmup_backtrack_counts=((_TARGET_IDS[0], 1), (_TARGET_IDS[1], 0)),
        activation_warmup_steps_per_target_site=8,
        joint_optimizer_steps_per_target_site_per_round=48,
        optimizer_steps_per_target_site_by_round=(56, 48, 48, 48),
    )
    payload = runner._branch_fit_history_payload(fit, include_warmup=True)
    runner._validate_branch_fit_history(payload, rounds=4, include_warmup=True)
    assert payload["warmup_accepted_learning_rates"] == [
        [_TARGET_IDS[0], 0.005],
        [_TARGET_IDS[1], 0.01],
    ]
    assert payload["warmup_backtrack_counts"] == [
        [_TARGET_IDS[0], 1],
        [_TARGET_IDS[1], 0],
    ]
    assert payload["retained_optimizer_steps_per_target_site_by_round"] == [
        56,
        48,
        48,
        48,
    ]

    wrong_rate = deepcopy(payload)
    wrong_rate["warmup_accepted_learning_rates"][0][1] = 0.004
    with pytest.raises(ValueError, match="warmup history"):
        runner._validate_branch_fit_history(
            wrong_rate,
            rounds=4,
            include_warmup=True,
        )
    too_many_trials = deepcopy(payload)
    too_many_trials["warmup_backtrack_counts"][0][1] = 25
    with pytest.raises(ValueError, match="warmup history"):
        runner._validate_branch_fit_history(
            too_many_trials,
            rounds=4,
            include_warmup=True,
        )


def test_publication_is_no_overwrite_and_preserves_foreign_race_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_validate_report", lambda value: value)
    monkeypatch.setattr(
        runner,
        "_strict_report",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    report = {"schema": "test", "report_sha256": _sha("report")}
    output = tmp_path / "report.json"
    runner._publish_report(output, report, publication_guard=lambda _phase: None)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        runner._publish_report(output, report, publication_guard=lambda _phase: None)

    raced = tmp_path / "raced.json"
    owned = tmp_path / "owned.json"

    def replacement_guard(phase_id: str) -> None:
        if phase_id == "targeted_site_rank2_publication_verified":
            raced.rename(owned)
            raced.write_bytes(b"foreign replacement")

    with pytest.raises((RuntimeError, ValueError, json.JSONDecodeError)):
        runner._publish_report(raced, report, publication_guard=replacement_guard)
    assert raced.read_bytes() == b"foreign replacement"
    assert owned.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_strict_reader_rejects_duplicate_nonfinite_and_linked_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="duplicate JSON field"):
        runner._strict_json_loads(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="non-finite"):
        runner._strict_json_loads(b'{"value":NaN}')

    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"test"}')
    symlink = tmp_path / "symlink.json"
    os.symlink(source, symlink)
    with pytest.raises(ValueError, match="regular file"):
        runner._strict_report(symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="identity or size"):
        runner._read_no_follow_regular_bytes(source)


def test_mocked_run_captures_A_once_shares_partition_and_never_opens_B_or_C(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fisher_graph.gemma3_attention_transport_experiment as experiment
    import fisher_graph.gemma3_attention_transport_experiment_core as experiment_core
    import fisher_graph.gemma3_attention_transport_global_site_rank1 as site_rank_one
    import fisher_graph.gemma3_attention_transport_runtime as runtime_module
    import fisher_graph.gemma3_attention_transport_targeted_site_rank2 as targeted
    import fisher_graph.gemma3_chart_native_neighbor_experiment as neighbor
    import fisher_graph.gemma3_chart_native_neighbor_fresh_panels as panels

    fit_role = neighbor.FIT_ROLE
    fit_receipts, dev_receipts = _receipts()
    records = tuple(
        SimpleNamespace(
            base_capture_receipt_sha256=receipt,
            capture=SimpleNamespace(invocation_id=invocation_id, row_count=16),
        )
        for receipt in fit_receipts + dev_receipts
        for invocation_id in _LEXICAL_IDS
    )
    events: list[str] = []
    panels_seen: list[object] = []
    published: dict[str, object] = {}
    root = Path(__file__).resolve().parents[1]

    class FakePanelSource:
        def __init__(self, **_kwargs: object) -> None:
            self.opened_roles: tuple[str, ...] = ()
            self.current_role: str | None = None
            self.next_role: str | None = fit_role
            self.freeze_receipts: dict[str, object] = {}
            panels_seen.append(self)

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
            storage_budget=SimpleNamespace(maximum_resident_memory_bytes=8 * 1024**3),
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
        panel.opened_roles = (fit_role,)
        panel.current_role = fit_role
        panel.next_role = fit_role
        events.append("capture-A")
        return records, _sha("dense-parity")

    hierarchy_bank = SimpleNamespace(
        artifact_sha256=_sha("hierarchy"),
        stored_parameter_count=83_008,
    )
    shared_bank = SimpleNamespace(
        artifact_sha256=_sha("shared"),
        stored_parameter_count=93_248,
    )
    parent_bank = SimpleNamespace(
        artifact_sha256=_sha("parent"),
        stored_parameter_count=113_728,
        global_site_rank_one_parameter_count=30_720,
    )
    control_bank = SimpleNamespace(
        artifact_sha256=_sha("control"),
        stored_parameter_count=113_728,
    )
    treatment_bank = SimpleNamespace(
        artifact_sha256=_sha("treatment"),
        stored_parameter_count=134_208,
        second_residual_parameter_count=20_480,
        targeted_residual_parameter_count=51_200,
        initial_site_rank_one_bank_sha256=parent_bank.artifact_sha256,
        parent_site_rank_one_sha256=_sha("current-first"),
        state_dict=lambda: {
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
        },
    )
    parent_initial = _fit_evaluation(10_368, _LEXICAL_IDS, 0.4)
    parent_final = _fit_evaluation(10_368, _LEXICAL_IDS, 0.3)
    parent_global_initial = _fit_evaluation(1_728, _GLOBAL_IDS, 0.5)
    parent_global_final = _fit_evaluation(1_728, _GLOBAL_IDS, 0.4)
    parent_fit = SimpleNamespace(
        bank=parent_bank,
        shared_rank_one_fit=SimpleNamespace(
            bank=shared_bank,
            hierarchical_fit=SimpleNamespace(
                bank=hierarchy_bank,
                dense_warm_start=SimpleNamespace(
                    bank=SimpleNamespace(artifact_sha256=_sha("dense"))
                ),
            ),
        ),
        initial_fit_oracle=parent_initial,
        final_fit_oracle=parent_final,
        initial_global_fit_oracle=parent_global_initial,
        final_global_fit_oracle=parent_global_final,
        fit_round_relative_fisher_losses=(0.4, 0.35, 0.33, 0.31, 0.3),
        fit_round_global_relative_fisher_losses=(0.5, 0.47, 0.44, 0.42, 0.4),
        fit_round_site_relative_fisher_losses=tuple(
            (key, (0.5, 0.47, 0.44, 0.42, 0.4)) for key in _GLOBAL_IDS
        ),
    )
    control_fit = SimpleNamespace(
        bank=control_bank,
        initial_site_rank_one_bank_sha256=parent_bank.artifact_sha256,
        initial_fit_oracle=parent_final,
        final_fit_oracle=_fit_evaluation(10_368, _LEXICAL_IDS, 0.29),
        initial_global_fit_oracle=parent_global_final,
        final_global_fit_oracle=_fit_evaluation(1_728, _GLOBAL_IDS, 0.39),
        fit_round_relative_fisher_losses=(0.3, 0.295, 0.292, 0.29, 0.29),
        fit_round_global_relative_fisher_losses=(0.4, 0.395, 0.392, 0.39, 0.39),
        fit_round_target_site_relative_fisher_losses=tuple(
            (key, (0.4, 0.395, 0.392, 0.39, 0.39)) for key in _TARGET_IDS
        ),
        optimizer_steps_per_target_site_by_round=(56, 48, 48, 48),
    )
    fitted = SimpleNamespace(
        bank=treatment_bank,
        site_rank_one_fit=parent_fit,
        continued_site_rank_one_control=control_fit,
        initial_fit_oracle=parent_final,
        final_fit_oracle=_fit_evaluation(10_368, _LEXICAL_IDS, 0.2),
        initial_global_fit_oracle=parent_global_final,
        final_global_fit_oracle=_fit_evaluation(1_728, _GLOBAL_IDS, 0.25),
        fit_round_relative_fisher_losses=(0.3, 0.27, 0.24, 0.21, 0.2),
        fit_round_global_relative_fisher_losses=(0.4, 0.36, 0.32, 0.28, 0.25),
        fit_round_target_site_relative_fisher_losses=tuple(
            (key, (0.4, 0.36, 0.32, 0.28, 0.25)) for key in _TARGET_IDS
        ),
        warmup_fixed_support_relative_decreases=((_TARGET_IDS[0], 0.1), (_TARGET_IDS[1], 0.1)),
        warmup_accepted_learning_rates=((_TARGET_IDS[0], 0.005), (_TARGET_IDS[1], 0.01)),
        warmup_backtrack_counts=((_TARGET_IDS[0], 1), (_TARGET_IDS[1], 0)),
        activation_warmup_steps_per_target_site=8,
        joint_optimizer_steps_per_target_site_per_round=48,
        optimizer_steps_per_target_site_by_round=(56, 48, 48, 48),
    )

    def fake_fit(captures: tuple[object, ...], **_kwargs: object) -> object:
        assert len(captures) == 36 * 18
        events.append("fit-A")
        return fitted

    def fake_all_layer(_fit: object, captures: tuple[object, ...], **_kwargs: object) -> object:
        assert _fit is fitted
        assert len(captures) == 9 * 18
        assert events[-1] == "fit-A"
        events.append("all-layer-A-dev")
        return SimpleNamespace(
            hierarchical=_all_layer_bank(0),
            shared_rank_one=_all_layer_bank(1),
            site_rank_one=_all_layer_bank(2),
            continued_site_rank_one=_all_layer_bank(3),
            targeted_site_rank_two=_all_layer_bank(4),
        )

    def fake_coupling(bank: object, captures: tuple[object, ...], **_kwargs: object) -> object:
        assert len(captures) == 9 * 18
        assert events[-1] in {"all-layer-A-dev", "coupling-A-dev"}
        events.append("coupling-A-dev")
        if bank is parent_bank:
            return _coupling_branch((0.22, 0.50, 0.12))
        if bank is control_bank:
            return _coupling_branch((0.21, 0.48, 0.12))
        assert bank is treatment_bank
        return _coupling_branch((0.15, 0.30, 0.12))

    def fake_publish(
        _path: Path,
        report: dict[str, object],
        *,
        publication_guard: object,
    ) -> None:
        publication_guard("targeted_site_rank2_temporary_sealed")
        publication_guard("targeted_site_rank2_publication_verified")
        published.update(report)
        events.append("publish")

    monkeypatch.setattr(neighbor, "static_preflight", lambda _binding: preflight)
    monkeypatch.setattr(neighbor, "validate_runtime_qualification", lambda **_kwargs: None)
    monkeypatch.setattr(
        neighbor,
        "observe_phase_resources",
        lambda *_args, **_kwargs: SimpleNamespace(peak_resident_set_bytes=1024),
    )
    monkeypatch.setattr(experiment, "_capture_role", fake_capture_role)
    monkeypatch.setattr(experiment, "_check_output_publication_headroom", lambda *_a, **_k: None)
    monkeypatch.setattr(experiment, "_check_resident_set_budget", lambda *_a, **_k: 1024)
    monkeypatch.setattr(
        experiment_core,
        "_open_split_panel",
        lambda _records, *, role: SimpleNamespace(artifact_sha256=_sha(f"split-{role}")),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_pinned_gemma3_attention_transport_capture_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(panels, "SequentialPanelSource", FakePanelSource)
    monkeypatch.setattr(targeted, "fit_targeted_site_rank_two_attention_transport_generator_bank", fake_fit)
    monkeypatch.setattr(targeted, "evaluate_all_layer_attention_transport_diagnostic", fake_all_layer)
    monkeypatch.setattr(targeted, "evaluate_targeted_site_rank_two_coupling_diagnostic", fake_coupling)
    monkeypatch.setattr(site_rank_one, "evaluate_global_site_rank_one_coupling_diagnostic", fake_coupling)
    monkeypatch.setattr(runner, "_validate_prior_site_rank1_report", lambda _root: {})

    def fake_source_hashes(_root: Path) -> dict[str, str]:
        result = {path: _sha(path) for path in runner._SOURCE_PATHS}
        result[runner._PRIOR_SITE_RANK1_CORE_SOURCE_PATH] = (
            runner._PRIOR_SITE_RANK1_CORE_SOURCE_SHA256
        )
        result[runner._PRIOR_SITE_RANK1_RUNNER_SOURCE_PATH] = (
            runner._PRIOR_SITE_RANK1_RUNNER_SOURCE_SHA256
        )
        return result

    monkeypatch.setattr(runner, "_source_sha256s", fake_source_hashes)
    monkeypatch.setattr(runner, "_publish_report", fake_publish)

    result = runner.run_targeted_site_rank2_A_diagnostic(
        binding=object(),
        report_path=tmp_path / "report.json",
    )

    assert result["B_opened_by_this_run"] is False
    assert result["C_ever_opened"] is False
    assert result["primary_gate_scope"] == "global_attention_targeted_tied_oracle"
    assert events.count("capture-A") == 1
    assert not any("capture-B" in event or "capture-C" in event for event in events)
    assert events.index("capture-A") < events.index("fit-A") < events.index("all-layer-A-dev")
    assert events[-1] == "publish"
    assert len(panels_seen) == 1
    panel = panels_seen[0]
    assert panel.opened_roles == (fit_role,)
    assert panel.current_role == fit_role
    assert panel.next_role == fit_role
    assert panel.freeze_receipts == {}
    assert published["A_partition"]["control_and_treatment_share_partition"] is True
    assert len(published["A_partition"]["fit_receipt_sha256s"]) == 36
    assert len(published["A_partition"]["dev_receipt_sha256s"]) == 9
    candidate = published["candidate"]
    assert candidate["generator_bank_parameters"] == 134_208
    assert candidate["second_residual_parameters"] == 20_480
    assert candidate["targeted_total_residual_parameters"] == 51_200
    assert candidate["hypothetical_total_parameters"] == 186_049
    closure = candidate["fit_initialization_closure"]
    assert closure["same_target_site_retained_optimizer_step_budget"] is True
    assert closure["same_target_site_compute_budget"] is False
    assert closure["same_target_site_warmup_search_budget"] is False
    assert closure["rejected_warmup_trials_enter_targeted_fitted_state"] is False
    treatment_history = candidate["targeted_rank_2_2_1_A_fit"]
    assert treatment_history["warmup_accepted_learning_rates"] == [
        [_TARGET_IDS[0], 0.005],
        [_TARGET_IDS[1], 0.01],
    ]
    assert treatment_history["warmup_backtrack_counts"] == [
        [_TARGET_IDS[0], 1],
        [_TARGET_IDS[1], 0],
    ]
    bank_fit = published["protocol"]["bank_fit"]
    assert bank_fit["warmup_max_backtracks"] == 24
    assert bank_fit["warmup_backtrack_factor"] == 0.5
    assert bank_fit["total_retained_optimizer_steps_per_target_site_per_branch"] == 200
    assert bank_fit["warmup_semantics"] == targeted.TARGETED_SITE_RANK_TWO_WARMUP_SEMANTICS
    assert bank_fit["control_matching_semantics"] == (
        "same_parent_same_retained_optimizer_step_budget_not_compute_or_search_matched_v1"
    )
    assert bank_fit["rejected_warmup_trial_state_semantics"] == (
        "rejected_backtracking_trials_discarded_no_optimizer_or_parameter_state_retained_v1"
    )
    assert published["scientific_status"]["component1_frozen_during_joint_fit"] is False
    assert published["scientific_status"][
        "retained_optimizer_step_matched_continued_rank1_control_fit"
    ] is True
    assert published["scientific_status"]["control_and_treatment_compute_matched"] is False
    assert published["scientific_status"][
        "control_and_treatment_warmup_search_matched"
    ] is False
    assert published["scientific_status"][
        "rejected_warmup_trials_enter_fitted_state"
    ] is False
    assert published["scientific_status"]["B_opened_by_this_run"] is False
    assert published["scientific_status"]["C_ever_opened"] is False
