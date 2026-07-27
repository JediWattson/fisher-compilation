from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from fisher_graph import (
    gemma3_structured_mlp_pseudo_unit_a_experiment as pseudo_unit_a_experiment,
)
from fisher_graph.adapters import StructuredOperatorSites

from fisher_graph.gemma3_full_width_single_layer_experiment import (
    FAMILY_STATUS,
    PROMPT_STATUS,
    PromptFamilyManifest,
)
from fisher_graph.gemma3_stability_experiment import Gemma3PromptSplits
from fisher_graph.gemma3_structured_mlp_pseudo_unit_a_experiment import (
    DEFAULT_DOWN_RIDGE,
    DEFAULT_GENERATOR_GRADIENT_CLIP_NORM,
    DEFAULT_GENERATOR_LEARNING_RATE,
    DEFAULT_GENERATOR_MINIBATCH_ROWS,
    DEFAULT_GENERATOR_STEPS,
    DEFAULT_JACOBIAN_FLOOR_FRACTION,
    DEFAULT_LAYER_INDEX,
    DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS,
    DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS,
    _build_deletion_baseline,
    _execute_fit_then_guard,
    _frozen_generator_protocol,
    _frozen_partition_token_contract,
    _guard_gate_report,
    _publish_artifact_pair,
    _standard_thresholds,
    _validate_artifact_candidate_fingerprint_bindings,
    _write_failed_guard_diagnostic,
    validate_v9_calibration_a_partitions,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)

from test_structured_mlp_compression import (
    _executor,
    _provenance,
    _score_batch,
    _targets,
)


def test_partition_token_contract_matches_frozen_v9_halves() -> None:
    assert DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS == 45_000
    assert DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS == 3
    assert _frozen_partition_token_contract() == {
        "minimum_supervised_tokens": 45_000,
        "minimum_length_buckets": 3,
        "applies_equally_to_fit_and_guard": True,
        "frozen_before_v9_tokenizer_or_model_access": True,
        "origin": "preregistered_for_fresh_structured_strong_v9",
    }


def test_generator_and_refit_protocol_is_frozen() -> None:
    assert _frozen_generator_protocol() == {
        "steps": DEFAULT_GENERATOR_STEPS,
        "learning_rate": DEFAULT_GENERATOR_LEARNING_RATE,
        "minibatch_rows": DEFAULT_GENERATOR_MINIBATCH_ROWS,
        "gradient_clip_norm": DEFAULT_GENERATOR_GRADIENT_CLIP_NORM,
        "jacobian_floor_fraction": DEFAULT_JACOBIAN_FLOOR_FRACTION,
        "down_ridge_for_diagnostic_variants": DEFAULT_DOWN_RIDGE,
        "fixed_before_guard": True,
        "checkpoint_selection": "final_fixed_step",
        "early_stopping": False,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("layer_index", DEFAULT_LAYER_INDEX + 1),
        ("generator_steps", DEFAULT_GENERATOR_STEPS + 1),
        (
            "generator_learning_rate",
            DEFAULT_GENERATOR_LEARNING_RATE * 2,
        ),
        (
            "generator_minibatch_rows",
            DEFAULT_GENERATOR_MINIBATCH_ROWS // 2,
        ),
        (
            "generator_gradient_clip_norm",
            DEFAULT_GENERATOR_GRADIENT_CLIP_NORM / 2,
        ),
        (
            "jacobian_floor_fraction",
            DEFAULT_JACOBIAN_FLOOR_FRACTION * 2,
        ),
        ("down_ridge", DEFAULT_DOWN_RIDGE * 2),
    ),
)
def test_run_rejects_tunable_layer_generator_or_refit_configuration(
    tmp_path,
    name: str,
    value: int | float,
) -> None:
    kwargs = {
        "parent_artifact_path": tmp_path / "parent.pt",
        "prompt_splits_path": tmp_path / "prompts.json",
        "family_manifest_path": tmp_path / "families.json",
        "corpus_audit_path": tmp_path / "audit.json",
        "revision": "a" * 40,
        "output": tmp_path / "candidate.pt",
        name: value,
    }

    with pytest.raises(ValueError, match="frozen"):
        pseudo_unit_a_experiment.run_gemma3_structured_mlp_pseudo_unit_a_experiment(
            **kwargs,
        )


def test_module_entrypoint_exposes_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisher_graph.gemma3_structured_mlp_pseudo_unit_a_experiment",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
    assert "--parent-artifact" in completed.stdout


def _index_digest(indices: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _fixture() -> tuple[
    Gemma3PromptSplits,
    PromptFamilyManifest,
    dict[str, object],
]:
    calibration_a = tuple(f"calibration-a-{index}" for index in range(512))
    heldout = {
        role: tuple(f"{role}-{index}" for index in range(96))
        for role in ("calibration_b", "validation", "test")
    }
    prompts = Gemma3PromptSplits(
        calibration_a=calibration_a,
        calibration_b=heldout["calibration_b"],
        validation=heldout["validation"],
        test=heldout["test"],
        scientific_status=PROMPT_STATUS,
    )
    a_families = tuple(f"a-family-{index % 16}" for index in range(512))
    heldout_families = {
        role: tuple(
            f"{role}-family-{index % 8}"
            for index in range(96)
        )
        for role in ("calibration_b", "validation", "test")
    }
    families = PromptFamilyManifest(
        calibration_a=a_families,
        calibration_b=heldout_families["calibration_b"],
        validation=heldout_families["validation"],
        test=heldout_families["test"],
        scientific_status=FAMILY_STATUS,
    )
    fit_family_ids = tuple(f"a-family-{index}" for index in range(0, 16, 2))
    guard_family_ids = tuple(
        f"a-family-{index}" for index in range(1, 16, 2)
    )

    def partition(family_ids: tuple[str, ...]) -> dict[str, object]:
        indices = [
            index
            for index, family in enumerate(a_families)
            if family in set(family_ids)
        ]
        return {
            "family_ids": list(family_ids),
            "family_count": 8,
            "prompt_count": 256,
            "prompt_indices": indices,
            "prompt_index_sha256": _index_digest(indices),
            "band_counts": {
                "micro": 8,
                "compact": 8,
                "medium": 16,
                "long": 224,
            },
        }

    audit = {
        "format_version": 4,
        "corpus_id": "structured-strong-v9",
        "purpose": "mode_bundling_fit_guard_and_frozen_evaluation",
        "calibration_a_policy": (
            "family_disjoint_fit_guard_development_only"
        ),
        "calibration_a_fit_may_train_candidate": True,
        "calibration_a_guard_may_change_candidate": False,
        "calibration_b_reuse_allowed": False,
        "heldout_splits_evaluated": False,
        "heldout_splits_tokenized": False,
        "heldout_splits_unevaluated": True,
        "heldout_splits_untokenized": True,
        "calibration_b_model_evaluated": False,
        "validation_model_evaluated": False,
        "test_model_evaluated": False,
        "tokenizer_or_model_accessed": False,
        "corpus_frozen_before_model_load": True,
        "prior_local_exact_prompt_overlap_count": 0,
        "prior_raw_prompt_overlap_count": 0,
        "prior_normalized_prompt_overlap_count": 0,
        "prior_domain_slug_overlap_count": 0,
        "prior_template_marker_overlap_count": 0,
        "prior_template_signature_overlap_count": 0,
        "prior_5_6_7_8_word_ngram_overlap_count": 0,
        "calibration_a_family_partitions": {
            "family_disjoint": True,
            "union_covers_calibration_a": True,
            "fit": partition(fit_family_ids),
            "guard": partition(guard_family_ids),
        },
    }
    return prompts, families, audit


def test_v9_fit_guard_partition_is_family_disjoint_and_complete() -> None:
    prompts, families, audit = _fixture()

    fit, guard = validate_v9_calibration_a_partitions(
        prompts,
        families,
        audit,
    )

    assert len(fit.prompts) == len(guard.prompts) == 256
    assert set(fit.family_ids).isdisjoint(guard.family_ids)
    assert set(fit.prompt_indices).isdisjoint(guard.prompt_indices)
    assert set(fit.prompt_indices) | set(guard.prompt_indices) == set(
        range(512)
    )


def test_v9_partition_rejects_guard_training_or_digest_drift() -> None:
    prompts, families, audit = _fixture()
    audit["calibration_a_guard_may_change_candidate"] = True
    with pytest.raises(ValueError, match="policy"):
        validate_v9_calibration_a_partitions(prompts, families, audit)

    prompts, families, audit = _fixture()
    partitions = audit["calibration_a_family_partitions"]
    assert isinstance(partitions, dict)
    fit = partitions["fit"]
    assert isinstance(fit, dict)
    fit["prompt_index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="partition"):
        validate_v9_calibration_a_partitions(prompts, families, audit)


def test_fit_then_guard_ordering_freezes_before_guard_access() -> None:
    events: list[str] = []
    state = {
        "pseudo_unit_direct": "1" * 64,
        "pseudo_unit_down_refit": "2" * 64,
        "fisher_deletion_down_refit": "3" * 64,
    }

    def materialize_fit() -> object:
        events.append("fit_materialized")
        return "fit"

    def build_from_fit(value: object) -> object:
        assert value == "fit"
        events.append("fit_built")
        return state

    def fingerprints(value: object) -> dict[str, str]:
        assert value is state
        events.append("fingerprints")
        return dict(state)

    def materialize_guard() -> object:
        events.append("guard_materialized")
        return "guard"

    def evaluate_guard(candidate: object, guard: object) -> object:
        assert candidate is state
        assert guard == "guard"
        events.append("guard_evaluated")
        return "evaluation"

    fit, candidates, guard, result, frozen = _execute_fit_then_guard(
        materialize_fit=materialize_fit,
        build_from_fit=build_from_fit,
        candidate_fingerprints=fingerprints,
        materialize_guard=materialize_guard,
        evaluate_guard=evaluate_guard,
    )

    assert fit == "fit"
    assert candidates is state
    assert guard == "guard"
    assert result == "evaluation"
    assert frozen == state
    assert events == [
        "fit_materialized",
        "fit_built",
        "fingerprints",
        "guard_materialized",
        "guard_evaluated",
        "fingerprints",
    ]


def test_fit_then_guard_rejects_candidate_mutation() -> None:
    state = {
        "pseudo_unit_direct": "1" * 64,
        "pseudo_unit_down_refit": "2" * 64,
        "fisher_deletion_down_refit": "3" * 64,
    }

    def evaluate(candidate: object, _guard: object) -> None:
        assert candidate is state
        state["pseudo_unit_direct"] = "4" * 64

    with pytest.raises(RuntimeError, match="mutated"):
        _execute_fit_then_guard(
            materialize_fit=lambda: "fit",
            build_from_fit=lambda _fit: state,
            candidate_fingerprints=lambda value: dict(value),  # type: ignore[arg-type]
            materialize_guard=lambda: "guard",
            evaluate_guard=evaluate,
        )


def test_deletion_refit_uses_its_actual_runtime_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_width = 4
    retained_width = 3
    parent_without_sites = _executor(
        source_width,
        projection_bias=False,
        seed=10_301,
    )
    layer_id = "layer.0"
    operator_sites = StructuredOperatorSites(
        attention_query_projection=f"{layer_id}.attention.query_projection",
        attention_query_normalized=f"{layer_id}.attention.query_normalized",
        attention_key_projection=f"{layer_id}.attention.key_projection",
        attention_key_normalized=f"{layer_id}.attention.key_normalized",
        attention_value_projection=f"{layer_id}.attention.value_projection",
        attention_context=f"{layer_id}.attention.context",
        feed_forward_gate_projection=f"{layer_id}.mlp.gate_projection",
        feed_forward_up_projection=f"{layer_id}.mlp.up_projection",
        feed_forward_down_input=f"{layer_id}.mlp.down_input",
    )
    parent = StructuredTransformerLayerExecutor(
        replace(
            parent_without_sites.config,
            transformer=replace(
                parent_without_sites.config.transformer,
                operator_sites=operator_sites,
            ),
        )
    )
    parent.load_state_dict(parent_without_sites.state_dict(), strict=True)
    parent.eval()
    native = _executor(source_width, projection_bias=False, seed=10_302)
    provenance = _provenance("d")
    hidden = torch.randn(
        2,
        4,
        parent.width,
        generator=torch.Generator().manual_seed(10_303),
    )
    valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    target = _targets(native, hidden, valid, provenance)
    native_features = target.feed_forward_projection_input
    assert native_features is not None
    score = _score_batch(
        batch_id="actual-runtime-deletion-refit",
        activations=native_features.detach().clone(),
        gradients=torch.arange(
            1,
            source_width + 1,
            dtype=native_features.dtype,
        )
        .view(1, 1, source_width)
        .expand_as(native_features)
        .clone(),
        mask=valid,
        provenance=provenance,
    )
    calibration_sha256 = "a" * 64
    sites = parent.config.transformer.operator_sites
    assert sites is not None
    selection = (
        pseudo_unit_a_experiment.select_fisher_taylor_mlp_units(
            (score,),
            calibration_split_sha256=calibration_sha256,
            activation_site=sites.feed_forward_down_input,
            parent_executor_fingerprint=parent.execution_fingerprint(),
            retained_width=retained_width,
            expected_source_width=source_width,
        )
    )
    native_selected = native_features.index_select(
        -1,
        torch.tensor(selection.selected_indices),
    )
    real_refit = (
        pseudo_unit_a_experiment
        .refit_structured_mlp_down_projection_from_targets_
    )
    actual_binding_observed = False

    def checking_refit(
        executor,
        batches,
        *,
        calibration_split_sha256: str,
        ridge: float,
    ) -> dict[str, object]:
        nonlocal actual_binding_observed
        supplied = batches[0].feed_forward_projection_input
        assert supplied is not None
        with torch.no_grad():
            actual = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            )
        torch.testing.assert_close(supplied, actual, rtol=0, atol=0)
        assert not torch.equal(supplied[valid], native_selected[valid])
        actual_binding_observed = True
        return real_refit(
            executor,
            batches,
            calibration_split_sha256=calibration_split_sha256,
            ridge=ridge,
        )

    monkeypatch.setattr(
        pseudo_unit_a_experiment,
        "GEMMA_MLP_PSEUDO_UNIT_SOURCE_WIDTH",
        source_width,
    )
    monkeypatch.setattr(
        pseudo_unit_a_experiment,
        "GEMMA_MLP_PSEUDO_UNIT_RETAINED_WIDTH",
        retained_width,
    )
    monkeypatch.setattr(
        pseudo_unit_a_experiment,
        "refit_structured_mlp_down_projection_from_targets_",
        checking_refit,
    )

    candidate, report = _build_deletion_baseline(
        parent,
        (target,),
        (score,),
        calibration_split_sha256=calibration_sha256,
        down_ridge=1e-5,
    )

    assert actual_binding_observed
    assert report["diagnostic_only"] is True
    refit_targets = report["refit_targets"]
    assert refit_targets["actual_runtime_features_used_for_down_refit"] is True
    assert (
        refit_targets[
            "native_selected_projection_features_used_for_down_refit"
        ]
        is False
    )
    assert report["execution_fingerprint"] == candidate.execution_fingerprint()


def _artifact_fingerprint_binding_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    str,
]:
    direct = "1" * 64
    refit = "2" * 64
    deletion_fingerprint = "3" * 64
    fingerprints = {
        "pseudo_unit_direct": direct,
        "pseudo_unit_down_refit": refit,
        "fisher_deletion_down_refit": deletion_fingerprint,
    }
    fit = {
        "candidate_fingerprints_frozen_before_guard": dict(fingerprints),
    }
    guard = {
        "candidate_fingerprints_before": dict(fingerprints),
        "candidate_fingerprints_after": dict(fingerprints),
    }
    pipeline = {
        "variants": {
            "direct": {"execution_fingerprint": direct},
            "global_down_refit": {"execution_fingerprint": refit},
        },
    }
    deletion = {
        "terminal_projection_refit": {
            "executor_fingerprint_before": "4" * 64,
            "executor_fingerprint_after": deletion_fingerprint,
        },
        "refit_targets": {
            "candidate_execution_fingerprint_before_refit": "4" * 64,
            "actual_runtime_features_used_for_down_refit": True,
            "native_selected_projection_features_used_for_down_refit": False,
        },
        "execution_fingerprint": deletion_fingerprint,
    }
    return fit, guard, pipeline, deletion, direct


def test_artifact_candidate_fingerprints_accept_exact_fit_guard_binding() -> None:
    fit, guard, pipeline, deletion, direct = (
        _artifact_fingerprint_binding_fixture()
    )

    _validate_artifact_candidate_fingerprint_bindings(
        fit=fit,
        guard=guard,
        pipeline=pipeline,
        deletion=deletion,
        direct_fingerprint=direct,
    )


@pytest.mark.parametrize(
    "drift",
    (
        "missing_fit_map",
        "fit_guard_mismatch",
        "missing_refit_key",
        "malformed_deletion_hash",
        "pipeline_refit_mismatch",
        "deletion_report_mismatch",
    ),
)
def test_artifact_candidate_fingerprints_reject_incomplete_or_drifted_binding(
    drift: str,
) -> None:
    fit, guard, pipeline, deletion, direct = (
        _artifact_fingerprint_binding_fixture()
    )
    if drift == "missing_fit_map":
        del fit["candidate_fingerprints_frozen_before_guard"]
    elif drift == "fit_guard_mismatch":
        fit_fingerprints = fit[
            "candidate_fingerprints_frozen_before_guard"
        ]
        assert isinstance(fit_fingerprints, dict)
        fit_fingerprints["pseudo_unit_direct"] = "4" * 64
    elif drift == "missing_refit_key":
        before = guard["candidate_fingerprints_before"]
        after = guard["candidate_fingerprints_after"]
        assert isinstance(before, dict)
        assert isinstance(after, dict)
        del before["pseudo_unit_down_refit"]
        del after["pseudo_unit_down_refit"]
    elif drift == "malformed_deletion_hash":
        before = guard["candidate_fingerprints_before"]
        after = guard["candidate_fingerprints_after"]
        fit_fingerprints = fit[
            "candidate_fingerprints_frozen_before_guard"
        ]
        assert isinstance(before, dict)
        assert isinstance(after, dict)
        assert isinstance(fit_fingerprints, dict)
        before["fisher_deletion_down_refit"] = "not-a-sha256"
        after["fisher_deletion_down_refit"] = "not-a-sha256"
        fit_fingerprints["fisher_deletion_down_refit"] = "not-a-sha256"
    elif drift == "pipeline_refit_mismatch":
        variants = pipeline["variants"]
        assert isinstance(variants, dict)
        refit_variant = variants["global_down_refit"]
        assert isinstance(refit_variant, dict)
        refit_variant["execution_fingerprint"] = "4" * 64
    elif drift == "deletion_report_mismatch":
        terminal = deletion["terminal_projection_refit"]
        assert isinstance(terminal, dict)
        terminal["executor_fingerprint_after"] = "4" * 64
    else:  # pragma: no cover
        raise AssertionError(f"unexpected drift fixture {drift!r}")

    with pytest.raises(ValueError, match="fingerprint"):
        _validate_artifact_candidate_fingerprint_bindings(
            fit=fit,
            guard=guard,
            pipeline=pipeline,
            deletion=deletion,
            direct_fingerprint=direct,
        )


def _publication_fixture() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {"executor": {"model_state_dict": {"weight": torch.ones(1)}}},
        {"artifact": {"main_artifact_written": True}},
    )


def test_success_publication_exposes_only_complete_pair(tmp_path) -> None:
    output = tmp_path / "candidate.pt"
    report_path = output.with_suffix(".json")
    artifact, report = _publication_fixture()

    result = _publish_artifact_pair(
        output,
        artifact,
        report,
        load_published=lambda: "validated",
    )

    assert result == "validated"
    loaded = torch.load(output, weights_only=True)
    assert loaded.keys() == artifact.keys()
    torch.testing.assert_close(
        loaded["executor"]["model_state_dict"]["weight"],
        artifact["executor"]["model_state_dict"]["weight"],
        rtol=0,
        atol=0,
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert set(tmp_path.iterdir()) == {output, report_path}


@pytest.mark.parametrize("collision_suffix", (".pt", ".json"))
def test_success_publication_refuses_late_collision_without_overwrite(
    tmp_path,
    collision_suffix: str,
) -> None:
    output = tmp_path / "candidate.pt"
    collision = output.with_suffix(collision_suffix)
    collision.write_bytes(b"preexisting")
    artifact, report = _publication_fixture()

    with pytest.raises(FileExistsError, match="overwrite"):
        _publish_artifact_pair(output, artifact, report)

    assert collision.read_bytes() == b"preexisting"
    other = output.with_suffix(
        ".json" if collision_suffix == ".pt" else ".pt"
    )
    assert not other.exists()
    assert set(tmp_path.iterdir()) == {collision}


@pytest.mark.parametrize("collision_suffix", (".pt", ".json"))
def test_success_publication_rolls_back_on_collision_created_after_staging(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    collision_suffix: str,
) -> None:
    output = tmp_path / "candidate.pt"
    report_path = output.with_suffix(".json")
    collision = output.with_suffix(collision_suffix)
    artifact, report = _publication_fixture()
    if collision_suffix == ".pt":
        real_save = torch.save

        def save_then_collide(value: object, path: object) -> None:
            real_save(value, path)
            collision.write_bytes(b"late collision")

        monkeypatch.setattr(
            pseudo_unit_a_experiment.torch,
            "save",
            save_then_collide,
        )
    else:
        real_write_json = pseudo_unit_a_experiment._write_json

        def write_then_collide(path, value) -> None:
            real_write_json(path, value)
            collision.write_bytes(b"late collision")

        monkeypatch.setattr(
            pseudo_unit_a_experiment,
            "_write_json",
            write_then_collide,
        )

    with pytest.raises(FileExistsError):
        _publish_artifact_pair(output, artifact, report)

    assert collision.read_bytes() == b"late collision"
    other = report_path if collision_suffix == ".pt" else output
    assert not other.exists()
    assert set(tmp_path.iterdir()) == {collision}


def test_success_publication_rolls_back_pair_when_validation_fails(
    tmp_path,
) -> None:
    output = tmp_path / "candidate.pt"
    report_path = output.with_suffix(".json")
    artifact, report = _publication_fixture()

    def reject_published_pair() -> object:
        assert output.exists()
        assert report_path.exists()
        raise RuntimeError("injected validation failure")

    with pytest.raises(RuntimeError, match="injected validation failure"):
        _publish_artifact_pair(
            output,
            artifact,
            report,
            load_published=reject_published_pair,
        )

    assert not output.exists()
    assert not report_path.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_success_publication_cleans_staging_when_json_serialization_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidate.pt"
    artifact, report = _publication_fixture()

    def fail_json_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected JSON failure")

    monkeypatch.setattr(
        pseudo_unit_a_experiment,
        "_write_json",
        fail_json_write,
    )
    with pytest.raises(OSError, match="injected JSON failure"):
        _publish_artifact_pair(output, artifact, report)

    assert tuple(tmp_path.iterdir()) == ()


def test_failed_guard_writes_json_only(tmp_path) -> None:
    output = tmp_path / "candidate.pt"
    payload = {
        "scientific_status": {"primary_passed": False},
        "model": {},
        "protocol": {},
        "parent": {},
        "calibration_a_fit": {},
        "calibration_a_guard": {},
        "pipeline": {},
        "deletion_baseline": {},
        "resource_report": {},
    }

    report = _write_failed_guard_diagnostic(output, payload)

    assert not output.exists()
    assert output.with_suffix(".json").exists()
    assert report["artifact"]["main_artifact_written"] is False


def test_failed_guard_refuses_existing_json_without_overwrite(tmp_path) -> None:
    output = tmp_path / "candidate.pt"
    report_path = output.with_suffix(".json")
    report_path.write_bytes(b"preexisting diagnostic")
    payload = {
        "scientific_status": {"primary_passed": False},
        "model": {},
        "protocol": {},
        "parent": {},
        "calibration_a_fit": {},
        "calibration_a_guard": {},
        "pipeline": {},
        "deletion_baseline": {},
        "resource_report": {},
    }

    with pytest.raises(FileExistsError, match="overwrite"):
        _write_failed_guard_diagnostic(output, payload)

    assert not output.exists()
    assert report_path.read_bytes() == b"preexisting diagnostic"


def test_refit_cannot_promote_direct_bundle_after_margin_failure() -> None:
    names = (
        "pseudo_unit_direct",
        "pseudo_unit_down_refit",
        "fisher_deletion_down_refit",
    )
    behavior = {
        name: {
            "delta_nll_per_token": 0.0,
            "top1_agreement_to_baseline": 1.0,
            "teacher_kl_per_token": 0.0,
            "per_example_delta_nll_per_token": {
                "p90_absolute": 0.0,
            },
            "per_example_top1_agreement": {"p10": 1.0},
        }
        for name in names
    }
    direct = {
        name: {
            "block_delta_nrmse": (
                0.016 if name == "pseudo_unit_direct" else 0.001
            ),
            "block_delta_cosine": 1.0,
        }
        for name in names
    }
    branch = {
        "attention_delta": {
            "block_delta_nrmse": 0.0,
            "block_delta_cosine": 1.0,
        },
        "feed_forward_delta": {
            "block_delta_nrmse": 0.0,
            "block_delta_cosine": 1.0,
        },
    }
    evaluation = {
        "behavior": behavior,
        "direct": direct,
        "branches": {name: branch for name in names},
        "execution_audits": {
            name: {"passed": True}
            for name in names
        },
        "ordinary_vs_segmented_native": {"passed": True},
        "native_boundary_replay": {"passed": True},
    }

    gates = _guard_gate_report(
        evaluation,
        thresholds=_standard_thresholds(),
    )

    assert gates["primary_passed"] is False
    candidates = gates["candidates"]
    assert candidates["pseudo_unit_direct"]["standard_passed"] is True
    assert (
        candidates["pseudo_unit_direct"][
            "primary_direct_block_nrmse_margin"
        ]
        is False
    )
    assert candidates["pseudo_unit_down_refit"]["standard_passed"] is True
    assert candidates["pseudo_unit_down_refit"]["authorizes_b"] is False
