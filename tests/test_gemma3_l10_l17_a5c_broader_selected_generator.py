from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
import inspect
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_l10_l17_a5c_broader_selected_generator as runner
from fisher_graph.gemma3_a5_same_shape_locality import (
    audit_same_shape_off_row_locality,
)
from fisher_graph.gemma3_l10_l17_trajectory_correction_lofo import (
    _EXPECTED_CONDITION_RESOURCES,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _a5b_failure_report() -> dict[str, object]:
    return {
        "report_sha256": runner._EXPECTED_A5B_REPORT_SHA256,
        "conclusion": {
            "learned_generator_improves_frozen_kl": False,
            "learned_generator_improves_frozen_delta_nll": False,
            "learned_generator_improves_frozen_top1": False,
        },
        "evaluation": {
            "conditions": {
                "trajectory_corrected_composition": dict(
                    runner._A5B_LEARNED_COMPOSITION
                )
            }
        },
    }


class _TokenLocalHead:
    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        score = hidden_states[..., 0] + hidden_states[..., -1]
        return torch.stack((score, -score, score.square()), dim=-1)


def _locality_receipt(*, offset: float = 0.0) -> dict[str, object]:
    absolute = 1.0e-6
    relative = 2.0e-6
    chunks: list[dict[str, object]] = []
    for chunk_index in range(2):
        rows = torch.tensor(
            (
                (0.4 + offset + chunk_index, -0.1, 1.0),
                (-0.6, 0.2 + offset, -0.75 - chunk_index),
            ),
            dtype=torch.float32,
        )
        teacher = audit_same_shape_off_row_locality(
            adapter=_TokenLocalHead(),  # type: ignore[arg-type]
            rows=rows,
            probe_name="native_teacher",
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            solver_authorization=True,
        ).receipt
        a4 = audit_same_shape_off_row_locality(
            adapter=_TokenLocalHead(),  # type: ignore[arg-type]
            rows=(rows + 0.125).contiguous(),
            probe_name="a4_euclidean_baseline",
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            solver_authorization=True,
        ).receipt
        locality = {
            "policy": "same_shape_directed_off_row_v2",
            "method": "same_shape_directed_off_row_native_and_a4",
            "probe_states": ["native_teacher", "a4_euclidean_baseline"],
            "row_count": 2,
            "nontrivial_multirow_probe": True,
            "absolute_tolerance": absolute,
            "relative_tolerance": relative,
            "teacher": dict(teacher),
            "a4_baseline": dict(a4),
            "directed_receipt_sha256_by_probe": {
                "teacher": teacher["receipt_sha256"],
                "a4_baseline": a4["receipt_sha256"],
            },
            "native_teacher_baseline_logits_reused_by_solver": True,
            "changes_solver_authorization": True,
            "passed": True,
        }
        chunks.append(
            {
                "chunk_index": chunk_index,
                "row_start": 2 * chunk_index,
                "row_stop": 2 * (chunk_index + 1),
                "row_count": 2,
                "token_locality": locality,
            }
        )

    return {
        "schema": "fisher_graph.a5c.fake_target",
        "receipt_sha256": _SHA_A,
        "row_count": 4,
        "row_chunk_size": 2,
        "batching": {
            "token_locality_audited_on_native_and_a4_states": True,
            "token_locality_absolute_tolerance": absolute,
            "token_locality_relative_tolerance": relative,
            "token_locality_policy": "same_shape_directed_off_row_v2",
        },
        "chunk_receipts": chunks,
    }


def _metric(*, nll: float = 2.0, native_nll: float = 1.0) -> dict[str, float]:
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": nll - native_nll,
        "native_to_candidate_kl_per_token": 0.5,
        "top1_agreement_to_native": 0.75,
    }


def _raw_evaluation(*, selected_nll: float = 2.0) -> dict[str, object]:
    resources = {
        name: {
            **values,
            "executed_peak_live_modal_width": (
                0 if name == "matched_double_deletion" else 1
            ),
        }
        for name, values in _EXPECTED_CONDITION_RESOURCES.items()
    }
    return {
        "execution_path": "full_model_logits_fixed_capacity_a3_trajectory_lofo",
        "assessment_role": "calibration_a_fit_family_blocked_development",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": 3,
        "logical_valid_tokens": 4,
        "native": {"nll_per_token": 1.0},
        "conditions": {
            "layer10_only": _metric(),
            "trajectory_corrected_layer17_only": _metric(),
            "frozen_uncorrected_composition": _metric(),
            "trajectory_corrected_composition": _metric(nll=selected_nll),
            "matched_double_deletion": _metric(),
        },
        "resource_accounting": resources,
        "exact_resources_match_protocol": True,
        "latency_or_kernel_speed_claim": False,
    }


def _executable(*, kind: str) -> runner._FrozenA5cExecutable:
    selected_graph = _SHA_B if kind == "learned_correction" else _SHA_A
    selected_composition = _SHA_C if kind == "learned_correction" else _SHA_A
    selected = {
        "layer17_graph_sha256": selected_graph,
        "composition_graph_sha256": selected_composition,
    }
    frozen = {
        "layer17_graph_sha256": _SHA_A,
        "composition_graph_sha256": _SHA_A,
    }
    return runner._FrozenA5cExecutable(
        kind=kind,
        selected_ridge=1.0e-4 if kind == "learned_correction" else None,
        layer17_graph=None,  # type: ignore[arg-type]
        layer17_lowerings_by_node={},
        composition_graph=None,  # type: ignore[arg-type]
        composition_lowerings=(),
        selected_descriptor=selected,
        frozen_descriptor=frozen,
        lineage={},
        selection_freeze_sha256=_SHA_A,
    )


def test_authenticates_only_the_exact_canonical_a5b_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _a5b_failure_report()
    monkeypatch.setattr(runner, "load_a5b_generator_microcanary_report", lambda _: report)
    monkeypatch.setattr(runner, "_file_sha256", lambda _: runner._EXPECTED_A5B_FILE_SHA256)

    authenticated, comparison = runner._authenticate_canonical_a5b_failure(
        Path("unused.json")
    )

    assert authenticated is report
    assert comparison["a5b_learned_composition"] == runner._A5B_LEARNED_COMPOSITION
    report["conclusion"]["learned_generator_improves_frozen_kl"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical A5b v1 failure"):
        runner._authenticate_canonical_a5b_failure(Path("unused.json"))


def test_token_locality_lineage_binds_exact_audited_chunk_catalog() -> None:
    receipt = _locality_receipt()
    baseline = runner.a5c_target_token_locality_lineage_sha256(receipt)

    changed = _locality_receipt(offset=0.25)
    assert runner.a5c_target_token_locality_lineage_sha256(changed) != baseline

    broken = _locality_receipt()
    broken["chunk_receipts"][1]["row_start"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="catalog drifted"):
        runner.a5c_target_token_locality_lineage_sha256(broken)


def test_source_scorer_hashes_bind_metrics_and_resources_separately() -> None:
    raw = _raw_evaluation()
    baseline_evaluation = runner.a5c_source_scorer_evaluation_sha256(raw)
    baseline_resources = runner.a5c_resource_accounting_sha256(
        raw["resource_accounting"]
    )

    changed_metric = _raw_evaluation(selected_nll=2.25)
    assert (
        runner.a5c_source_scorer_evaluation_sha256(changed_metric)
        != baseline_evaluation
    )
    assert runner.a5c_resource_accounting_sha256(
        changed_metric["resource_accounting"]
    ) == baseline_resources

    changed_resources = _raw_evaluation()
    changed_resources["resource_accounting"]["layer10_only"][  # type: ignore[index]
        "executed_peak_live_modal_width"
    ] = 2
    assert runner.a5c_resource_accounting_sha256(
        changed_resources["resource_accounting"]
    ) != baseline_resources


def test_outer_evaluation_explicitly_renames_conditions_and_binds_graphs() -> None:
    executable = _executable(kind="learned_correction")
    adapted = runner._adapt_outer_evaluation(
        _raw_evaluation(selected_nll=2.25),
        outer_fold_index=0,
        layer10_graph_sha256="d" * 64,
        executable=executable,
    )

    assert set(adapted["conditions"]) == {
        "layer10_only",
        "selected_layer17_only",
        "frozen_uncorrected_composition",
        "selected_composition",
        "matched_double_deletion",
    }
    assert adapted["conditions"]["layer10_only"]["graph_sha256"] == "d" * 64
    assert adapted["conditions"]["selected_layer17_only"]["graph_sha256"] == _SHA_B
    assert adapted["conditions"]["selected_composition"]["graph_sha256"] == _SHA_C
    assert adapted["conditions"]["matched_double_deletion"]["graph_sha256"] == _SHA_C


def test_frozen_fallback_requires_metric_and_hash_identical_composition() -> None:
    executable = _executable(kind="frozen_source_fallback")
    adapted = runner._adapt_outer_evaluation(
        _raw_evaluation(),
        outer_fold_index=0,
        layer10_graph_sha256="d" * 64,
        executable=executable,
    )
    assert adapted["conditions"]["selected_composition"] == adapted[
        "conditions"
    ]["frozen_uncorrected_composition"]

    with pytest.raises(RuntimeError, match="metric/hash-identical"):
        runner._adapt_outer_evaluation(
            _raw_evaluation(selected_nll=2.25),
            outer_fold_index=0,
            layer10_graph_sha256="d" * 64,
            executable=executable,
        )


class _HeldBlocks(Mapping[str, object]):
    def __init__(self) -> None:
        self.selected = False

    def __getitem__(self, key: str) -> object:
        assert key == "held"
        self.selected = True
        return "held-block"

    def __iter__(self):
        yield "held"

    def __len__(self) -> int:
        return 1


class _Graph:
    def __init__(self, identity: str, order: tuple[str, ...]) -> None:
        self.artifact_sha256 = identity * 64
        self.traversal_order = order
        self.parameter_count = 163_094 if len(order) == 4 else 1
        self.macs_per_token = 160_352 if len(order) == 4 else 1

    def validate_integrity(self) -> None:
        return None


class _Lowering:
    def __init__(self, identity: str) -> None:
        self.artifact_sha256 = identity * 64

    def validate_integrity(self) -> None:
        return None


class _Executor:
    def __init__(
        self,
        adapter: object,
        graph_plan: object,
        lowerings: object,
        *,
        post_feedforward_delta_layer_ordinals=(),
    ) -> None:
        self.graph_plan = graph_plan
        self.post_feedforward_delta_layer_ordinals = tuple(
            post_feedforward_delta_layer_ordinals
        )


def test_held_scoring_batch_is_not_selected_until_freeze_reauthenticates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(kind="frozen_source_fallback")
    blocks = _HeldBlocks()
    monkeypatch.setattr(runner, "a5c_selection_freeze_sha256", lambda **_: _SHA_A)
    monkeypatch.setattr(runner, "_take_first_examples", lambda value, count: (value, count))

    assert runner._select_held_scoring_batch_after_freeze(
        blocks=blocks,
        held_family_alias="held",
        executable=executable,
    ) == ("held-block", 1)
    assert blocks.selected is True

    blocks = _HeldBlocks()
    monkeypatch.setattr(runner, "a5c_selection_freeze_sha256", lambda **_: _SHA_B)
    with pytest.raises(RuntimeError, match="not frozen"):
        runner._select_held_scoring_batch_after_freeze(
            blocks=blocks,
            held_family_alias="held",
            executable=executable,
        )
    assert blocks.selected is False


def test_scoring_executor_post_delta_semantics_are_preheld_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "Gemma3ModalGeneratorGraphExecutor", _Executor)
    layer10 = _Graph("1", ("l10",))
    layer17 = _Graph("2", ("a", "b", "c", "d"))
    composition = _Graph("3", ("composition",))
    layer10_lowerings = {"l10": _Lowering("4")}
    layer17_lowerings = {
        name: _Lowering(str(index + 5))
        for index, name in enumerate(layer17.traversal_order)
    }
    executable = runner._FrozenA5cExecutable(
        kind="learned_correction",
        selected_ridge=1.0e-4,
        layer17_graph=layer17,  # type: ignore[arg-type]
        layer17_lowerings_by_node=layer17_lowerings,  # type: ignore[arg-type]
        composition_graph=composition,  # type: ignore[arg-type]
        composition_lowerings=(),
        selected_descriptor={
            "layer17_graph_sha256": layer17.artifact_sha256,
            "composition_graph_sha256": composition.artifact_sha256,
            "layer17_post_feedforward_delta_layer_ordinals": [17],
            "composition_post_feedforward_delta_layer_ordinals": [17],
        },
        frozen_descriptor={
            "layer17_graph_sha256": _SHA_A,
            "composition_graph_sha256": composition.artifact_sha256,
        },
        lineage={},
        selection_freeze_sha256=_SHA_A,
    )

    executors = runner._build_scoring_executors(
        adapter=object(),  # type: ignore[arg-type]
        layer10_graph=layer10,  # type: ignore[arg-type]
        layer10_lowerings_by_node=layer10_lowerings,  # type: ignore[arg-type]
        source_composition_graph=composition,  # type: ignore[arg-type]
        source_composition_lowerings=(),
        executable=executable,
    )

    assert len({id(executor) for executor in executors}) == 4
    assert executors[0].post_feedforward_delta_layer_ordinals == ()
    assert executors[1].post_feedforward_delta_layer_ordinals == (17,)
    assert executors[2].post_feedforward_delta_layer_ordinals == ()
    assert executors[3].post_feedforward_delta_layer_ordinals == (17,)


def test_invalid_runtime_fails_before_any_model_or_artifact_access() -> None:
    with pytest.raises(ValueError, match="canonical pinned CPU"):
        runner.run_gemma3_l10_l17_a5c_broader_selected_generator(
            revision="not-a-revision"
        )


def test_surviving_prepublication_bundle_replays_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "a5c.json"
    bundle = runner.default_a5c_prepublication_bundle_path(destination)
    bundle.write_text("survivor", encoding="utf-8")
    expected = {"report_sha256": _SHA_A}
    observed: dict[str, object] = {}

    def finalize(path: Path, *, output: Path) -> dict[str, object]:
        observed.update({"path": path, "output": output})
        return expected

    monkeypatch.setattr(runner, "finalize_a5c_prepublication_bundle", finalize)
    monkeypatch.setattr(
        runner,
        "_authenticate_canonical_a5b_failure",
        lambda *_: (_ for _ in ()).throw(AssertionError("source accessed")),
    )

    result = runner.run_gemma3_l10_l17_a5c_broader_selected_generator(
        revision=runner._EXPECTED_MODEL_REVISION,
        output=destination,
    )

    assert result is expected
    assert observed == {"path": bundle, "output": destination}


def test_continuation_workspace_is_frozen_and_dispatch_is_strict() -> None:
    workspace = object.__new__(runner.A5cBroaderTrainingWorkspace)
    observed: list[object] = []
    expected = {"report_sha256": _SHA_A}

    result = runner._dispatch_a5c_training_continuation(
        workspace,
        lambda value: observed.append(value) or expected,
    )

    assert result is expected
    assert observed == [workspace]
    assert runner.A5cBroaderTrainingWorkspace.__dataclass_params__.frozen
    workspace_fields = {value.name for value in fields(workspace)}
    assert {
        "adapter",
        "model",
        "blocks",
        "held_family_alias",
        "source_composition_graph",
        "source_composition_lowerings",
        "layer10_graph",
        "layer17_graph",
        "layer10_lowerings_by_node",
        "layer17_lowerings_by_node",
        "bases_by_node",
        "fragment_id_by_node",
        "native_block_states",
        "frozen_compiled_block_states",
        "compiled_correction_base_states",
        "target_solution",
        "target_report_receipt",
        "target_compact",
        "bridge",
        "bridge_receipt",
        "coordinate_row_bank_compact",
        "breadth",
        "breadth_receipt",
        "breadth_compact",
        "capture_metadata",
        "capture_audit",
        "capture_sha256",
        "capture_audit_sha256",
        "target_token_locality_lineage_sha256",
        "runtime_metadata",
        "source_bindings",
        "comparison_to_a5b",
        "configuration_metadata",
    } <= workspace_fields
    with pytest.raises(FrozenInstanceError):
        workspace.output_path = Path("mutated")
    with pytest.raises(TypeError, match="report dictionary"):
        runner._dispatch_a5c_training_continuation(
            workspace,
            lambda _: None,  # type: ignore[arg-type,return-value]
        )


def test_continuation_dispatch_is_post_breadth_pre_cv_pre_held_access() -> None:
    source = inspect.getsource(
        runner.run_gemma3_l10_l17_a5c_broader_selected_generator
    )

    breadth = source.index("breadth_receipt = breadth.receipt()")
    dispatch = source.index("_dispatch_a5c_training_continuation(")
    selector = source.index("cv_selection = select_a5c_family_disjoint_ridge(")
    held = source.index("held_batches = _select_held_scoring_batch_after_freeze(")
    assert breadth < dispatch < selector < held


def test_continuation_skips_surviving_a5c_prepublication_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "a5d.json"
    bundle = runner.default_a5c_prepublication_bundle_path(destination)
    bundle.write_text("survivor", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "finalize_a5c_prepublication_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("A5c prepublication recovery ran")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_canonical_a5b_failure",
        lambda *_: (_ for _ in ()).throw(RuntimeError("continued-preflight")),
    )

    with pytest.raises(RuntimeError, match="continued-preflight"):
        runner.run_gemma3_l10_l17_a5c_broader_selected_generator(
            revision=runner._EXPECTED_MODEL_REVISION,
            output=destination,
            _continuation=lambda _: {},
        )
