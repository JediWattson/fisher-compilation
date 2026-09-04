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

import fisher_graph.gemma3_attention_transport_hierarchical_edge_experiment as runner
from fisher_graph.gemma3_attention_transport_hierarchical_edge_experiment import (
    REPORT_SCHEMA,
    _B_CORPUS_PATH,
    _B_CORPUS_SHA256,
    _B_MANIFEST_FILE_SHA256,
    _B_MANIFEST_KEY,
    _B_MANIFEST_PATH,
    _PRIOR_B_SPLIT_SHA256,
    _PRIOR_GLOBAL_B_EVALUATION_SHA256,
    _PRIOR_GLOBAL_B_REPORT_FILE_SHA256,
    _PRIOR_GLOBAL_B_REPORT_PATH,
    _PRIOR_GLOBAL_B_REPORT_SHA256,
    _PRIOR_GLOBAL_B_SPLIT_SHA256,
    _PRIOR_SPARSE_A_REPORT_FILE_SHA256,
    _PRIOR_SPARSE_A_REPORT_PATH,
    _PRIOR_SPARSE_A_REPORT_SHA256,
    _PRIOR_V1_REPORT_FILE_SHA256,
    _PRIOR_V1_REPORT_PATH,
    _PRIOR_V1_REPORT_SHA256,
    _publish_report,
    _seal_report,
    _source_sha256s,
    _strict_report,
    _validate_prior_exposure_files,
    check_hierarchical_edge_B_replay,
)


_LAYOUT_SHA256 = (
    "0547169009cec9e730c5a967dd9f4ab1aa5c12181c0121c050fb76f8b981adae"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _resources() -> dict[str, object]:
    return {
        "native_v_o_parameters": 14_745_600,
        "generator_bank_parameters": 83_008,
        "chart_parameters": 23_041,
        "router_parameters": 28_800,
        "compiled_parameters": 134_849,
        "parameter_reduction_fraction": 1.0 - 134_849 / 14_745_600,
        "parameter_compression_eligible": True,
        "native_qk_retained": True,
        "native_output_rmsnorm_retained": True,
        "logical_speedup_claimed": False,
    }


def _hierarchy_protocol() -> dict[str, object]:
    return {
        "layout_kind": "crossed_3x6_zero_sum_signed_scale_contrasts",
        "layout_artifact_sha256": _LAYOUT_SHA256,
        "contrast_kind": "helmert_zero_sum_v1",
        "macroblock_count": 3,
        "repeated_slot_count": 6,
        "macroblock_contrast_count": 2,
        "slot_contrast_count": 5,
        "generator_count": 8,
        "signed_scale_contrast_parameters": 448,
        "router_context_width": 16,
        "router_feature_width": 24,
        "global_active_count": 2,
        "hierarchy_alternating_rounds": 4,
        "hierarchy_ridge": 1.0e-8,
        "normal_equation_jitter": 1.0e-12,
        "signed_feature_floor": 1.0e-12,
        "edge_feature_semantics": (
            "log1p_context_rms_and_scale_latent_cosine"
        ),
        "support_semantics": "membership_mix_then_single_global_topk",
        "learned_objects_fit_role": "A_only",
    }


def _unsealed_report(root: Path) -> dict[str, object]:
    # The pinned predecessor is a useful authentic B fixture.  Rebind only the
    # new A-fitted candidate lineage; its held-out B metrics remain internally
    # closed and the hierarchy checker still validates every field afresh.
    value = json.loads((root / _PRIOR_GLOBAL_B_REPORT_PATH).read_bytes())
    value.pop("report_sha256")
    passed = value["scientific_status"]["B_replay_gates_passed"]
    value["schema"] = REPORT_SCHEMA
    value["scientific_status"] = {
        "A_fitted": True,
        "B_previously_exposed_replay": True,
        "B_influenced_architecture_design": True,
        "B_opened": True,
        "B_used_for_training": False,
        "B_replay_gates_passed": passed,
        "C_staging_eligible": passed,
        "C_open_authorized": False,
        "C_opened": False,
        "confirmation_claimed": False,
        "behavioral_nll_measured": False,
        "global_top2_support_enforced": True,
        "hierarchical_edge_conditioning_enforced": True,
        "operator_boundary_only": True,
        "executable_artifact_published": False,
        "logical_speedup_claimed": False,
    }
    b_split = value["splits"]["B_split_artifact_sha256"]
    value["exposure_binding"] = {
        "prior_v1_report_sha256": _PRIOR_V1_REPORT_SHA256,
        "prior_v1_report_file_sha256": _PRIOR_V1_REPORT_FILE_SHA256,
        "prior_v1_report_path": str(_PRIOR_V1_REPORT_PATH),
        "prior_B_split_artifact_sha256": _PRIOR_B_SPLIT_SHA256,
        "prior_sparse_A_report_sha256": _PRIOR_SPARSE_A_REPORT_SHA256,
        "prior_sparse_A_report_file_sha256": (
            _PRIOR_SPARSE_A_REPORT_FILE_SHA256
        ),
        "prior_sparse_A_report_path": str(_PRIOR_SPARSE_A_REPORT_PATH),
        "prior_global_B_report_sha256": _PRIOR_GLOBAL_B_REPORT_SHA256,
        "prior_global_B_report_file_sha256": (
            _PRIOR_GLOBAL_B_REPORT_FILE_SHA256
        ),
        "prior_global_B_report_path": str(_PRIOR_GLOBAL_B_REPORT_PATH),
        "prior_global_B_split_artifact_sha256": _PRIOR_GLOBAL_B_SPLIT_SHA256,
        "prior_global_B_evaluation_artifact_sha256": (
            _PRIOR_GLOBAL_B_EVALUATION_SHA256
        ),
        "B_manifest_path": str(_B_MANIFEST_PATH),
        "B_manifest_file_sha256": _B_MANIFEST_FILE_SHA256,
        "B_manifest_key": _B_MANIFEST_KEY,
        "B_corpus_path": str(_B_CORPUS_PATH),
        "B_corpus_file_sha256": _B_CORPUS_SHA256,
        "current_B_split_matches_prior_v1": b_split == _PRIOR_B_SPLIT_SHA256,
        "current_B_split_matches_prior_global": (
            b_split == _PRIOR_GLOBAL_B_SPLIT_SHA256
        ),
        "fresh_confirmation": False,
        "interpretation": (
            "architecture_informed_development_replay_on_previously_"
            "exposed_B_not_confirmation"
        ),
    }
    protocol = value["protocol"]
    protocol["protocol_id"] = (
        "hierarchical_edge_k8_r8_global_top2_exposed_B_replay_v1"
    )
    protocol["hierarchy"] = _hierarchy_protocol()
    protocol["implementation_source_sha256s"] = _source_sha256s(root)

    candidate = value["candidate"]
    candidate["candidate_id"] = "k8_r8_hierarchical_edge_global_top2"
    candidate["pre_B_candidate_artifact_sha256"] = candidate[
        "candidate_artifact_sha256"
    ]
    candidate["post_B_candidate_artifact_sha256"] = candidate[
        "candidate_artifact_sha256"
    ]
    candidate["shared_bank_artifact_sha256"] = _sha("shared-bank")
    candidate["hierarchy_layout_artifact_sha256"] = _LAYOUT_SHA256
    candidate["hierarchy_parameter_count"] = 448
    candidate["contains_native_value_or_output_weights"] = False
    candidate["contains_oracle_outputs_or_coefficients"] = False
    candidate["contains_raw_teacher_outputs_or_gradients"] = False
    candidate["A_hierarchical_fit"] = candidate.pop("A_sparse_fit")
    candidate["resources"] = _resources()
    value["B_replay"]["evaluation"]["resources"] = _resources()
    decision = value["B_replay"]["decision"]
    decision["candidate_id"] = candidate["candidate_id"]
    decision["candidate_artifact_sha256"] = candidate[
        "candidate_artifact_sha256"
    ]
    return value


def _noop_guard(_phase_id: str) -> None:
    return None


def test_cli_import_is_standard_library_only() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/gemma3_attention_transport_hierarchical_edge_experiment.py"
    )
    code = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('hierarchical_edge_preflight', {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = (
    'torch',
    'numpy',
    'transformers',
    'fisher_graph.gemma3_attention_transport_compiler',
    'fisher_graph.gemma3_attention_transport_runtime',
    'fisher_graph.gemma3_attention_transport_experiment_core',
    'fisher_graph.gemma3_attention_transport_sparse_v2',
    'fisher_graph.gemma3_attention_transport_hierarchical_edge',
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


def test_public_surface_and_capture_calls_have_no_C_capable_entrypoint() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/gemma3_attention_transport_hierarchical_edge_experiment.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    capture_roles: list[str] = []
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
        assert "confirm" not in called.lower()
        if called == "_capture_role":
            runtime_role = next(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "runtime_role"
            )
            assert isinstance(runtime_role, ast.Name)
            capture_roles.append(runtime_role.id)

    assert public_functions == {
        "run_hierarchical_edge_B_replay",
        "check_hierarchical_edge_B_replay",
        "main",
    }
    assert capture_roles == ["FIT_ROLE", "DISCOVERY_ROLE"]


def test_prior_chain_and_source_catalog_are_fully_pinned() -> None:
    root = Path(__file__).resolve().parents[1]

    _validate_prior_exposure_files(root)
    sources = _source_sha256s(root)

    assert _PRIOR_V1_REPORT_SHA256 == (
        "6117879689a3449defd9b6b2a9ac0bdedd336064560d83fc811a96c09a97a48f"
    )
    assert _B_CORPUS_SHA256 == (
        "f0c0b60e2e42b69703a4c81fc9a35c906a49d7a3e604b186c8d3fbc9c7755cc1"
    )
    assert _PRIOR_GLOBAL_B_EVALUATION_SHA256 == (
        "6445299e291e2613485bbdf82d380466dabd542c49ef6ad15129612bc7babe52"
    )
    assert (
        "src/fisher_graph/gemma3_attention_transport_hierarchical_edge.py"
        in sources
    )
    assert (
        "src/fisher_graph/gemma3_chart_native_neighbor_experiment.py"
        in sources
    )
    assert not any("global_router.py" in path for path in sources)


def test_report_is_canonical_source_bound_and_never_overwrites(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"
    _publish_report(output, report, publication_guard=_noop_guard)

    assert output.read_bytes() == runner._canonical_json_bytes(report)
    assert _strict_report(output) == report
    checked = check_hierarchical_edge_B_replay(
        output,
        repository_root=root,
    )
    assert checked["report_sha256"] == report["report_sha256"]
    assert checked["C_opened"] is False
    with pytest.raises(FileExistsError):
        _publish_report(output, report, publication_guard=_noop_guard)

    drifted = _unsealed_report(root)
    first_source = next(iter(drifted["protocol"]["implementation_source_sha256s"]))
    drifted["protocol"]["implementation_source_sha256s"][first_source] = "0" * 64
    drifted_report = _seal_report(drifted)
    drifted_output = tmp_path / "drifted.json"
    _publish_report(drifted_output, drifted_report, publication_guard=_noop_guard)
    with pytest.raises(ValueError, match="sources drifted"):
        check_hierarchical_edge_B_replay(
            drifted_output,
            repository_root=root,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["scientific_status"].__setitem__(
                "B_used_for_training", True
            ),
            "scientific status",
        ),
        (
            lambda value: value["scientific_status"].__setitem__(
                "C_opened", True
            ),
            "scientific status",
        ),
        (
            lambda value: value["B_replay"].__setitem__(
                "C_open_authorized", True
            ),
            "replay lineage",
        ),
        (
            lambda value: value["candidate"].__setitem__(
                "post_B_candidate_artifact_sha256", _sha("B-fit")
            ),
            "candidate lineage",
        ),
        (
            lambda value: value["candidate"].__setitem__(
                "contains_raw_teacher_outputs_or_gradients", True
            ),
            "candidate lineage",
        ),
        (
            lambda value: value["splits"]["freeze_receipts"][
                "A_chart_transport_fit"
            ].__setitem__("stage_receipt_sha256", _sha("foreign-A")),
            "freeze receipt",
        ),
        (
            lambda value: value["splits"]["final_firewall_state"].__setitem__(
                "next_role", None
            ),
            "firewall",
        ),
    ],
)
def test_report_rejects_B_fit_C_open_and_frozen_lineage_tampering(
    mutate: object,
    message: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    value = _unsealed_report(root)
    mutate(value)
    with pytest.raises(ValueError, match=message):
        _seal_report(value)


def test_report_rejects_metric_gate_resource_and_counter_tampering() -> None:
    root = Path(__file__).resolve().parents[1]

    metric = _unsealed_report(root)
    metric["B_replay"]["evaluation"]["relative_fisher_loss"] = 0.01
    with pytest.raises(ValueError, match="Fisher equations"):
        _seal_report(metric)

    support = _unsealed_report(root)
    support["B_replay"]["evaluation"]["effective_support_count"][
        "maximum"
    ] = 3
    with pytest.raises(ValueError, match="support"):
        _seal_report(support)

    gate = _unsealed_report(root)
    gate["gates"]["max_effective_support_fraction"] = 0.5
    with pytest.raises(ValueError, match="gates"):
        _seal_report(gate)

    resource = _unsealed_report(root)
    resource["candidate"]["resources"]["router_parameters"] = 28_801
    with pytest.raises(ValueError, match="resource"):
        _seal_report(resource)

    counter = _unsealed_report(root)
    counter["runtime_counters"]["role_batch_counts"][2][1] = 1
    with pytest.raises(ValueError, match="counters"):
        _seal_report(counter)


def test_prior_files_require_exact_canonical_regular_bytes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    pinned = (
        _PRIOR_V1_REPORT_PATH,
        _PRIOR_SPARSE_A_REPORT_PATH,
        _PRIOR_GLOBAL_B_REPORT_PATH,
        _B_MANIFEST_PATH,
        _B_CORPUS_PATH,
    )
    for relative in pinned:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())

    _validate_prior_exposure_files(tmp_path)
    prior = tmp_path / _PRIOR_SPARSE_A_REPORT_PATH
    reauthored = json.loads(prior.read_bytes())
    prior.write_text(json.dumps(reauthored, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match=r"sparse.*A"):
        _validate_prior_exposure_files(tmp_path)


def test_strict_reader_rejects_duplicate_fields_and_symlinks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    canonical = runner._canonical_json_bytes(report)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(canonical.replace(b"{", b'{"schema":"duplicate",', 1))
    with pytest.raises(ValueError, match="JSON"):
        _strict_report(duplicate)

    source = tmp_path / "source.json"
    source.write_bytes(canonical)
    linked = tmp_path / "linked.json"
    os.symlink(source, linked)
    with pytest.raises(ValueError, match="regular file"):
        _strict_report(linked)


def test_publication_rejects_symlink_parent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    os.symlink(real, linked)

    with pytest.raises(ValueError, match="real directory"):
        _publish_report(
            linked / "replay.json",
            report,
            publication_guard=_noop_guard,
        )


def test_publication_rolls_back_only_the_owned_inode(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"
    displaced = tmp_path / "owned.json"
    phases: list[str] = []

    def guard(phase_id: str) -> None:
        phases.append(phase_id)
        if phase_id == "hierarchical_edge_publication_verified":
            output.rename(displaced)
            output.write_bytes(b"foreign replacement")
            raise RuntimeError("resident_cap_crossed")

    with pytest.raises(RuntimeError, match="resident_cap_crossed"):
        _publish_report(output, report, publication_guard=guard)

    assert phases == [
        "hierarchical_edge_temporary_sealed",
        "hierarchical_edge_publication_verified",
    ]
    assert output.read_bytes() == b"foreign replacement"
    assert displaced.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publication_preserves_a_foreign_replacement_at_temporary_path(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"
    displaced = tmp_path / "owned-temporary.json"
    foreign = b"foreign temporary replacement"

    def guard(phase_id: str) -> None:
        if phase_id == "hierarchical_edge_temporary_sealed":
            temporary = next(tmp_path.glob(".*.tmp"))
            temporary.rename(displaced)
            temporary.write_bytes(foreign)

    with pytest.raises(RuntimeError, match="temporary report identity"):
        _publish_report(output, report, publication_guard=guard)

    remaining = list(tmp_path.glob(".*.tmp"))
    assert len(remaining) == 1
    assert remaining[0].read_bytes() == foreign
    assert displaced.exists()
    assert not output.exists()


def test_publication_removes_owned_same_inode_corruption(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"

    def guard(phase_id: str) -> None:
        if phase_id == "hierarchical_edge_publication_verified":
            with output.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"corrupt")
                handle.flush()
                os.fsync(handle.fileno())

    with pytest.raises((RuntimeError, ValueError)):
        _publish_report(output, report, publication_guard=guard)

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_run_fits_only_on_A_then_freezes_A_captures_B_and_stops_before_C(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fisher_graph.gemma3_attention_transport_compiler as compiler
    import fisher_graph.gemma3_attention_transport_experiment as experiment
    import fisher_graph.gemma3_attention_transport_experiment_core as core
    import fisher_graph.gemma3_attention_transport_hierarchical_edge as hierarchy
    import fisher_graph.gemma3_attention_transport_runtime as runtime_module
    import fisher_graph.gemma3_chart_native_neighbor_experiment as neighbor
    import fisher_graph.gemma3_chart_native_neighbor_fresh_panels as panels

    root = Path(__file__).resolve().parents[1]
    fit_role = neighbor.FIT_ROLE
    discovery_role = neighbor.DISCOVERY_ROLE
    invocation_ids = tuple(f"mutation.{2 * index + 1}" for index in range(18))
    events: list[str] = []
    captured_report: dict[str, object] = {}
    fixture = _unsealed_report(root)

    class FakePanelSource:
        def __init__(self, **_kwargs: object) -> None:
            self.opened_roles: tuple[str, ...] = ()
            self.current_role: str | None = None
            self.next_role: str | None = fit_role
            self.freeze_receipts: dict[str, dict[str, str]] = {}

        def freeze_current(self, stage_kind: str, receipt: str) -> None:
            assert self.current_role is not None
            role = self.current_role
            events.append(f"freeze-{role}")
            self.freeze_receipts[role] = {
                "stage_kind": stage_kind,
                "stage_receipt_sha256": receipt,
            }
            self.current_role = None
            self.next_role = (
                discovery_role
                if role == fit_role
                else "C_neighbor_confirmation"
            )

    class FakeRuntime:
        def __init__(self) -> None:
            self.counters = SimpleNamespace(
                payload=lambda: deepcopy(fixture["runtime_counters"])
            )

        def runtime_qualification_receipt(self) -> object:
            return SimpleNamespace(
                model_id="google/gemma-3-270m",
                model_revision=(
                    "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
                ),
                hidden_width=640,
                qualification_receipt_sha256=_sha("qualification"),
            )

        def close(self) -> None:
            events.append("runtime-close")

    preflight_binding = SimpleNamespace(
        repository_root=root,
        storage_budget=SimpleNamespace(
            maximum_resident_memory_bytes=8 * 1024**3
        ),
    )
    preflight = SimpleNamespace(
        binding=preflight_binding,
        fresh_manifest={
            "roles": {
                discovery_role: {
                    "file_sha256": _B_CORPUS_SHA256,
                    "path": str(_B_CORPUS_PATH),
                }
            }
        },
        fresh_receipt=object(),
        opaque_panel_receipt=object(),
        receipt_sha256=_sha("preflight"),
        validate_receipt=lambda: None,
    )
    a_captures = {
        invocation_id: SimpleNamespace(
            invocation_id=invocation_id,
            row_count=1,
        )
        for invocation_id in invocation_ids
    }
    b_captures = {
        invocation_id: SimpleNamespace(
            invocation_id=invocation_id,
            row_count=1,
        )
        for invocation_id in invocation_ids
    }
    a_records = tuple(
        SimpleNamespace(capture=a_captures[invocation_id])
        for invocation_id in invocation_ids
    )
    b_records = tuple(
        SimpleNamespace(capture=b_captures[invocation_id])
        for invocation_id in invocation_ids
    )

    class FakeSplitPanel:
        def __init__(self, role: str) -> None:
            records = a_records if role == "A" else b_records
            self._records = {
                record.capture.invocation_id: (record,) for record in records
            }
            self.invocation_ids = invocation_ids
            self.signatures = tuple(
                SimpleNamespace(native_v_o_parameter_count=819_200)
                for _ in invocation_ids
            )
            self.artifact_sha256 = (
                _sha("A-split")
                if role == "A"
                else _PRIOR_GLOBAL_B_SPLIT_SHA256
            )
            self.receipt_sha256s = tuple(
                _sha(f"{role}-{index}") for index in range(18)
            )

        def records_for_invocation(self, invocation_id: str) -> tuple[object, ...]:
            return self._records[invocation_id]

    def fake_capture_role(**kwargs: object) -> tuple[tuple[object, ...], str | None]:
        panel_source = kwargs["panel_source"]
        runtime_role = kwargs["runtime_role"]
        core_role = kwargs["core_role"]
        assert isinstance(panel_source, FakePanelSource)
        assert runtime_role == panel_source.next_role
        assert core_role in {"A", "B"}
        panel_source.opened_roles += (runtime_role,)
        panel_source.current_role = runtime_role
        events.append(f"capture-{core_role}")
        if core_role == "A":
            return a_records, _sha("dense-parity")
        return b_records, None

    layout = hierarchy.AttentionTransportHierarchyLayout.gemma3_270m_attention()
    fake_bank = SimpleNamespace(
        hierarchy=layout,
        hierarchy_parameter_count=448,
        artifact_sha256=_sha("hierarchical-bank"),
        shared_bank_sha256=_sha("shared-bank"),
        stored_parameter_count=83_008,
        state_dict=lambda: {
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
        },
    )
    fake_sparse_fit = SimpleNamespace(
        bank=fake_bank,
        hierarchy_layout_sha256=layout.artifact_sha256,
        initial_fit_oracle=SimpleNamespace(output_metrics=object()),
        final_fit_oracle=SimpleNamespace(output_metrics=object()),
        fit_round_relative_fisher_losses=(0.5, 0.4, 0.3, 0.2, 0.1),
        dense_warm_start=SimpleNamespace(
            bank=SimpleNamespace(artifact_sha256=_sha("dense-warm-bank"))
        ),
    )

    def fake_fit_bank(*_args: object, **_kwargs: object) -> object:
        assert "capture-B" not in events
        events.append("fit-bank")
        return fake_sparse_fit

    fake_atlas = SimpleNamespace(artifact_sha256=_sha("atlas"))

    def fake_fit_atlas(*_args: object, **_kwargs: object) -> object:
        assert "capture-B" not in events
        events.append("fit-atlas")
        return fake_atlas

    def fake_fit_router(
        _bank: object,
        captures: tuple[object, ...],
        _atlas: object,
        **_kwargs: object,
    ) -> object:
        assert "capture-B" not in events
        invocation_id = captures[0].invocation_id
        events.append(f"fit-router-{invocation_id}")
        fake_router = SimpleNamespace(
            invocation_id=invocation_id,
            artifact_sha256=_sha(f"router-{invocation_id}"),
            state_dict=lambda: {
                "contains_oracle_outputs_or_coefficients": False,
                "contains_teacher_outputs_or_gradients": False,
            },
        )
        return SimpleNamespace(router=fake_router)

    fake_resources = SimpleNamespace(**_resources())

    def fake_candidate(**kwargs: object) -> object:
        return SimpleNamespace(
            config=kwargs["config"],
            artifact_sha256=_sha("candidate"),
        )

    def fake_development(**_kwargs: object) -> object:
        return SimpleNamespace(artifact_sha256=_sha("development"))

    decision = SimpleNamespace(
        passed=False,
        evaluation=object(),
    )

    def fake_select(*_args: object, **_kwargs: object) -> object:
        assert events[-1] == "capture-B"
        events.append("select-B")
        return SimpleNamespace(
            decisions=(decision,),
            artifact_sha256=_sha("selection"),
        )

    def fake_seal(value: dict[str, object]) -> dict[str, object]:
        captured_report.update(value)
        return {**value, "report_sha256": _sha("report")}

    def fake_publish(
        _path: Path,
        _report: dict[str, object],
        *,
        publication_guard: object,
    ) -> None:
        assert captured_report["scientific_status"]["B_used_for_training"] is False
        assert captured_report["scientific_status"]["C_opened"] is False
        assert captured_report["splits"]["final_firewall_state"]["C_opened"] is False
        events.append("publish")

    monkeypatch.setattr(neighbor, "static_preflight", lambda _binding: preflight)
    monkeypatch.setattr(neighbor, "validate_runtime_qualification", lambda **_kwargs: None)
    monkeypatch.setattr(
        neighbor,
        "observe_phase_resources",
        lambda *_args, **_kwargs: SimpleNamespace(peak_resident_set_bytes=1024),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_pinned_gemma3_attention_transport_capture_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(experiment, "_capture_role", fake_capture_role)
    monkeypatch.setattr(
        experiment, "_check_output_publication_headroom", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        experiment,
        "_check_resident_set_budget",
        lambda _binding, _reader, *, phase_id: 1024,
    )
    monkeypatch.setattr(compiler, "fit_frozen_attention_chart_atlas", fake_fit_atlas)
    monkeypatch.setattr(
        compiler,
        "AttentionTransportBankFit",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        hierarchy,
        "fit_hierarchical_sparse_attention_transport_generator_bank",
        fake_fit_bank,
    )
    monkeypatch.setattr(
        hierarchy,
        "fit_hierarchical_edge_attention_transport_router",
        fake_fit_router,
    )
    monkeypatch.setattr(panels, "SequentialPanelSource", FakePanelSource)
    monkeypatch.setattr(
        core,
        "_open_split_panel",
        lambda _records, *, role: FakeSplitPanel(role),
    )
    monkeypatch.setattr(core, "_resource_accounting", lambda *_args: fake_resources)
    monkeypatch.setattr(core, "AttentionTransportFittedCandidate", fake_candidate)
    monkeypatch.setattr(core, "AttentionTransportADevelopment", fake_development)
    monkeypatch.setattr(
        core, "select_attention_transport_candidate_on_b", fake_select
    )
    monkeypatch.setattr(runner, "_validate_prior_exposure_files", lambda _root: None)
    monkeypatch.setattr(
        runner,
        "_source_sha256s",
        lambda _root: {path: _sha(path) for path in runner._SOURCE_PATHS},
    )
    monkeypatch.setattr(
        runner,
        "_aggregate_evaluation_payload",
        lambda _value: deepcopy(fixture["B_replay"]["evaluation"]),
    )
    monkeypatch.setattr(
        runner,
        "_sparse_fit_metric_payload",
        lambda value: deepcopy(
            fixture["candidate"]["A_hierarchical_fit"][
                "initial_sparse_oracle"
                if value is fake_sparse_fit.initial_fit_oracle
                else "final_sparse_oracle"
            ]
        ),
    )
    monkeypatch.setattr(runner, "_resource_payload", lambda _value: _resources())
    monkeypatch.setattr(
        runner,
        "_decision_payload",
        lambda _value: deepcopy(fixture["B_replay"]["decision"]),
    )
    monkeypatch.setattr(
        runner,
        "_gate_payload",
        lambda _value: deepcopy(fixture["gates"]),
    )
    monkeypatch.setattr(runner, "_seal_report", fake_seal)
    monkeypatch.setattr(runner, "_publish_report", fake_publish)

    result = runner.run_hierarchical_edge_B_replay(
        binding=object(),
        report_path=tmp_path / "replay.json",
    )

    assert result["C_opened"] is False
    assert events.index(f"freeze-{fit_role}") < events.index("capture-B")
    assert events.index("capture-B") < events.index(f"freeze-{discovery_role}")
    assert not any(
        event.startswith("fit-")
        for event in events[events.index("capture-B") + 1 :]
    )
    assert events.count("capture-A") == 1
    assert events.count("capture-B") == 1
    assert not any("C_neighbor_confirmation" in event for event in events)
    assert captured_report["candidate"]["pre_B_candidate_artifact_sha256"] == (
        captured_report["candidate"]["post_B_candidate_artifact_sha256"]
    )
