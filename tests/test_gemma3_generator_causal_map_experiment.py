from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.generator_causal_fingerprints import (
    GeneratorCausalFingerprintProvenance,
    GeneratorInterventionLogitBatch,
    collect_generator_causal_fingerprints,
)
from fisher_graph.gemma3_generator_causal_fingerprint_experiment import (
    _causal_analysis_lineage,
    _catalog_sha256,
    _generator_causal_summaries,
    _generator_fit_lineage,
    _pairwise_similarities,
)
from fisher_graph.gemma3_generator_causal_map_experiment import (
    _active_generator_output_mapping,
    _authenticate_fingerprint_source,
    _directed_edges,
    _generator_cohort_affinities,
    _generator_nodes,
    _joint_interactions,
    _prompt_cohorts,
    _verify_replayed_singleton_identity,
    build_parser,
    run_gemma3_generator_causal_map_experiment,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _small_runtime(layer_count: int = 2, prompt_count: int = 3) -> object:
    source_fits = tuple(_sha(f"source-fit:{index}") for index in range(layer_count))
    deployed_fits = tuple(
        _sha(f"deployed-fit:{index}") for index in range(layer_count)
    )
    plans = tuple(_sha(f"plan:{index}") for index in range(layer_count))
    content = tuple(_sha(f"content:{index}") for index in range(prompt_count))
    return SimpleNamespace(
        replacements=tuple(range(layer_count)),
        refit_start_layer=1,
        layer_lineage=tuple(
            {"layer_ordinal": index, "layer_id": f"layer.{index}"}
            for index in range(layer_count)
        ),
        source_fit_sha256s=source_fits,
        deployed_fit_sha256s=deployed_fits,
        generator_plan_sha256s=plans,
        source_model_sha256=_sha("model"),
        base_artifact_file_sha256=_sha("base-file"),
        base_scientific_payload_sha256=_sha("base-scientific"),
        refit_artifact_file_sha256=_sha("refit-file"),
        refit_scientific_payload_sha256=_sha("refit-scientific"),
        analysis_split={
            "serialized_sha256": _sha("split"),
            "content_sha256": content,
            "example_count": prompt_count,
            "logical_valid_tokens": prompt_count,
            "supervised_tokens": prompt_count,
        },
    )


def _fingerprint_analysis(
    runtime: object,
    generator_ids: tuple[str, ...],
    example_ids: tuple[str, ...],
) -> object:
    prompt_count = len(example_ids)
    baseline = torch.linspace(
        2.0,
        -1.0,
        8,
        dtype=torch.float64,
    ).reshape(1, 1, 8).repeat(prompt_count, 1, 1)
    muted: dict[str, torch.Tensor] = {}
    for ordinal, generator_id in enumerate(generator_ids):
        value = baseline.clone()
        value[:, :, (ordinal + 1) % 8] -= (
            torch.arange(1, prompt_count + 1, dtype=torch.float64)[:, None]
            * 0.01
            * (ordinal + 1)
        )
        muted[generator_id] = value
    return collect_generator_causal_fingerprints(
        (
            GeneratorInterventionLogitBatch(
                example_ids=example_ids,
                baseline_logits=baseline,
                muted_logits_by_generator=muted,
                targets=torch.zeros((prompt_count, 1), dtype=torch.int64),
                supervised_mask=torch.ones(
                    (prompt_count, 1),
                    dtype=torch.bool,
                ),
            ),
        ),
        generator_ids=generator_ids,
        provenance=GeneratorCausalFingerprintProvenance(
            source_model_sha256=runtime.source_model_sha256,
            generator_catalog_sha256=_catalog_sha256(runtime),
            evaluation_split_sha256=runtime.analysis_split[
                "serialized_sha256"
            ],
            objective_sha256=_sha("objective"),
        ),
        anchor_count=2,
        top_importance_count=1,
    )


def _fingerprint_payload(
    runtime: object,
    analysis: object,
    example_ids: tuple[str, ...],
    *,
    model_id: str = "google/gemma-3-270m",
    revision: str = "1" * 40,
) -> dict[str, object]:
    summaries = _generator_causal_summaries(
        analysis,
        runtime,
        prompt_content_sha256s=runtime.analysis_split["content_sha256"],
        example_ids=example_ids,
    )
    return {
        "model": {
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_commit": revision,
            "adapter_model_fingerprint": runtime.source_model_sha256,
            "local_files_only": True,
        },
        "frozen_sources": {
            "base_full_stack": {
                "artifact_file_sha256": runtime.base_artifact_file_sha256,
                "scientific_payload_sha256": (
                    runtime.base_scientific_payload_sha256
                ),
                "frozen_before_analysis": True,
            },
            "sequential_refit": {
                "artifact_file_sha256": runtime.refit_artifact_file_sha256,
                "scientific_payload_sha256": (
                    runtime.refit_scientific_payload_sha256
                ),
                "frozen_before_analysis": True,
            },
        },
        "analysis_split": {
            "role": "adaptive_open_development_generator_family_discovery",
            **runtime.analysis_split,
            "membership_exact": True,
            "assurance": "caller_declared_self_attested",
            "externally_authenticated": False,
            "heldout_confirmation": False,
            "used_for_adaptive_analysis": True,
            "used_for_generator_fit": False,
            "used_for_generator_selection": False,
        },
        "causal_analysis_lineage": _causal_analysis_lineage(analysis),
        "deployed_generator_plan_sha256s": runtime.generator_plan_sha256s,
        "generator_fit_lineage": _generator_fit_lineage(runtime),
        "generator_causal_summaries": summaries,
        "pairwise_similarities": _pairwise_similarities(
            analysis,
            runtime,
            summaries,
        ),
        "scientific_payload_sha256": _sha("fingerprint-scientific"),
    }


def test_converts_ephemeral_traces_to_exact_active_mapping() -> None:
    traces = (
        torch.ones((2, 3, 4)),
        None,
        torch.full((2, 3, 4), 2.0),
    )
    execution = SimpleNamespace(
        generated_residuals=traces,
        suppressed_layer_ordinals=(1,),
    )
    result = _active_generator_output_mapping(
        execution,
        ("g0", "g1", "g2"),
    )
    assert tuple(result) == ("g0", "g2")
    assert result["g0"] is traces[0]
    assert result["g2"] is traces[2]

    execution.generated_residuals = (traces[0], traces[0], traces[2])
    with pytest.raises(ValueError, match="suppressed generator"):
        _active_generator_output_mapping(
            execution,
            ("g0", "g1", "g2"),
        )


def test_authenticates_and_exactly_replays_frozen_singleton_identity() -> None:
    runtime = _small_runtime()
    generator_ids = ("g0", "g1")
    example_ids = tuple(_sha(f"example:{index}") for index in range(3))
    analysis = _fingerprint_analysis(runtime, generator_ids, example_ids)
    fingerprint = _fingerprint_payload(runtime, analysis, example_ids)

    _authenticate_fingerprint_source(
        fingerprint,
        runtime,
        model_id="google/gemma-3-270m",
        revision="1" * 40,
    )
    _verify_replayed_singleton_identity(
        analysis,
        json.loads(json.dumps(fingerprint)),
        runtime,
        prompt_content_sha256s=runtime.analysis_split["content_sha256"],
        example_ids=example_ids,
    )

    fingerprint["causal_analysis_lineage"] = {
        **fingerprint["causal_analysis_lineage"],
        "artifact_sha256": _sha("wrong"),
    }
    with pytest.raises(ValueError, match="differs from fingerprint"):
        _verify_replayed_singleton_identity(
            analysis,
            fingerprint,
            runtime,
            prompt_content_sha256s=runtime.analysis_split["content_sha256"],
            example_ids=example_ids,
        )


def _interaction_analysis() -> object:
    return SimpleNamespace(
        generator_ids=("g0", "g1"),
        pair_catalog=(("g0", "g1"),),
        directed_edge_catalog=(("g0", "g1"),),
        prompt_nll_second_differences=torch.tensor(
            [[-0.2, 0.0, 0.4]],
            dtype=torch.float64,
        ),
        prompt_joint_baseline_to_condition_kls=torch.tensor(
            [[0.1, 0.2, 0.3]],
            dtype=torch.float64,
        ),
        prompt_joint_top1_agreements=torch.tensor(
            [[1.0, 0.5, 0.0]],
            dtype=torch.float64,
        ),
        prompt_centered_anchor_interaction_residual_rms=torch.tensor(
            [[0.3, 0.6, 0.9]],
            dtype=torch.float64,
        ),
        prompt_relative_interaction_denominator_rms=torch.tensor(
            [[1.0, 0.0, 2.0]],
            dtype=torch.float64,
        ),
        prompt_relative_interaction_ratios=torch.tensor(
            [[0.3, 0.0, 0.45]],
            dtype=torch.float64,
        ),
        prompt_relative_interaction_defined=torch.tensor(
            [[True, False, True]],
            dtype=torch.bool,
        ),
        prompt_directed_response_rms=torch.tensor(
            [[0.2, 0.4, 0.6]],
            dtype=torch.float64,
        ),
        prompt_directed_baseline_output_rms=torch.tensor(
            [[1.0, 0.0, 2.0]],
            dtype=torch.float64,
        ),
        prompt_directed_response_cosines=torch.tensor(
            [[0.5, 0.0, -0.5]],
            dtype=torch.float64,
        ),
        prompt_directed_response_cosine_defined=torch.tensor(
            [[True, False, True]],
            dtype=torch.bool,
        ),
        prompt_directed_response_ratios=torch.tensor(
            [[0.2, 0.0, 0.3]],
            dtype=torch.float64,
        ),
        prompt_directed_response_ratio_defined=torch.tensor(
            [[True, False, True]],
            dtype=torch.bool,
        ),
    )


def test_builds_ordered_nodes_joint_rows_and_directed_rows() -> None:
    runtime = _small_runtime()
    example_ids = tuple(_sha(f"example:{index}") for index in range(3))
    singleton = _fingerprint_analysis(runtime, ("g0", "g1"), example_ids)
    fingerprint = _fingerprint_payload(runtime, singleton, example_ids)
    nodes = _generator_nodes(
        fingerprint,
        generator_ids=singleton.generator_ids,
    )
    assert tuple(row["layer_ordinal"] for row in nodes) == (0, 1)
    assert nodes[0]["authorizes_mutation"] is False

    interaction = _interaction_analysis()
    joints = _joint_interactions(
        interaction,
        runtime,
        fingerprint,
        prompt_content_sha256s=runtime.analysis_split["content_sha256"],
    )
    assert len(joints) == 1
    assert joints[0]["mean_nll_second_difference_per_token"] == pytest.approx(
        0.2 / 3.0
    )
    assert joints[0][
        "mean_relative_interaction_ratio_over_defined"
    ] == pytest.approx(0.375)
    assert joints[0]["authorizes_merge"] is False

    directed = _directed_edges(
        interaction,
        runtime,
        fingerprint,
        prompt_content_sha256s=runtime.analysis_split["content_sha256"],
    )
    assert len(directed) == 1
    assert directed[0]["mean_directed_response_rms"] == pytest.approx(0.4)
    assert directed[0][
        "mean_directed_response_cosine_over_defined"
    ] == pytest.approx(0.0)
    assert directed[0]["strict_upstream_invariance_confirmed"] is True


def test_hashes_declared_family_cohorts_and_builds_affinities() -> None:
    runtime = _small_runtime(prompt_count=4)
    example_ids = tuple(_sha(f"example:{index}") for index in range(4))
    singleton = _fingerprint_analysis(runtime, ("g0", "g1"), example_ids)
    fingerprint = _fingerprint_payload(runtime, singleton, example_ids)
    assessment = SimpleNamespace(
        prompt_sha256s=example_ids,
        family_ids=("private-family-a", "private-family-b") * 2,
        artifact_sha256=_sha("partition"),
    )
    cohorts = _prompt_cohorts(
        assessment,
        prompt_content_sha256s=runtime.analysis_split["content_sha256"],
    )
    assert len(cohorts) == 2
    assert tuple(row["prompt_count"] for row in cohorts) == (2, 2)
    assert "private-family-a" not in repr(cohorts)
    assert all(row["membership_exact"] is True for row in cohorts)

    affinities = _generator_cohort_affinities(
        fingerprint,
        runtime,
        cohorts,
    )
    assert len(affinities) == 4
    assert all(row["authorizes_mutation"] is False for row in affinities)
    assert all(row["prompt_count"] == 2 for row in affinities)


def test_parser_exposes_ignored_json_output_and_fingerprint_source() -> None:
    parser = build_parser()
    arguments = parser.parse_args(["--revision", "a" * 40])
    assert arguments.output.as_posix().startswith(".local-runs/")
    assert arguments.output.suffix == ".json"
    assert arguments.fingerprint_artifact.suffix == ".json"


def test_full_runner_visits_all_172_conditions_and_publishes_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisher_graph.gemma3_generator_causal_map_experiment as module

    layer_count = 18
    prompt_count = 20
    runtime = _small_runtime(layer_count=layer_count, prompt_count=prompt_count)
    revision = "b" * 40
    runtime.model_metadata = {
        "model_id": "google/gemma-3-270m",
        "requested_revision": revision,
        "resolved_commit": revision,
    }
    family_counts = (4, 4, 3, 3, 2, 2, 1, 1)
    families = tuple(
        family
        for family, count in enumerate(family_counts)
        for _ in range(count)
    )
    example_ids = tuple(_sha(f"raw-prompt:{index}") for index in range(20))
    assessment = SimpleNamespace(
        prompts=tuple(f"prompt {index}" for index in range(20)),
        prompt_sha256s=example_ids,
        family_ids=tuple(f"family.{value}" for value in families),
        artifact_sha256=_sha("assessment-partition"),
        source_export_sha256=_sha("source-export"),
        source_fit_prompt_index_sha256=_sha("fit-prompt-index"),
        role="open_development_assessment",
        prompt_count=20,
    )
    selection_metadata = {
        "partition_salt": "test-partition",
    }
    runtime.partition_metadata = {
        "selection_prompt_count": 20,
        "expected_prompt_count": 40,
        "selection": selection_metadata,
    }
    partition = SimpleNamespace(
        assessment=assessment,
        artifact_sha256=_sha("partition-plan"),
        metadata=lambda: runtime.partition_metadata,
    )
    generator_ids = tuple(f"g{index}" for index in range(layer_count))
    baseline = torch.linspace(
        3.0,
        -2.0,
        20,
        dtype=torch.float64,
    ).reshape(1, 1, 20).repeat(prompt_count, 1, 1)
    prompt_scale = torch.linspace(
        0.01,
        0.02,
        prompt_count,
        dtype=torch.float64,
    )[:, None]
    singleton_logits: list[torch.Tensor] = []
    for ordinal in range(layer_count):
        value = baseline.clone()
        value[:, :, (ordinal + 1) % 20] -= prompt_scale * (ordinal + 1)
        singleton_logits.append(value)
    singleton_analysis = collect_generator_causal_fingerprints(
        (
            GeneratorInterventionLogitBatch(
                example_ids=example_ids,
                baseline_logits=baseline,
                muted_logits_by_generator={
                    generator_id: singleton_logits[ordinal]
                    for ordinal, generator_id in enumerate(generator_ids)
                },
                targets=torch.zeros((prompt_count, 1), dtype=torch.int64),
                supervised_mask=torch.ones(
                    (prompt_count, 1),
                    dtype=torch.bool,
                ),
            ),
        ),
        generator_ids=generator_ids,
        provenance=GeneratorCausalFingerprintProvenance(
            source_model_sha256=runtime.source_model_sha256,
            generator_catalog_sha256=_catalog_sha256(runtime),
            evaluation_split_sha256=runtime.analysis_split[
                "serialized_sha256"
            ],
            objective_sha256=_sha("objective"),
        ),
        anchor_count=2,
        top_importance_count=5,
    )
    fingerprint = _fingerprint_payload(
        runtime,
        singleton_analysis,
        example_ids,
        revision=revision,
    )
    batch = CalibrationBatch(
        model_inputs={
            "input_ids": torch.zeros(
                (prompt_count, 1),
                dtype=torch.int64,
            )
        },
        targets=torch.zeros((prompt_count, 1), dtype=torch.int64),
        valid_positions=torch.ones(
            (prompt_count, 1),
            dtype=torch.bool,
        ),
    )
    safe_stream = {
        "serialized_sha256": runtime.analysis_split["serialized_sha256"],
        "sequences": prompt_count,
        "valid_tokens": {"total": prompt_count},
        "supervised_positions": {"total": prompt_count},
        "content_sha256": runtime.analysis_split["content_sha256"],
    }

    class FakeModel:
        def eval(self) -> None:
            return None

        def requires_grad_(self, _enabled: bool) -> FakeModel:
            return self

    class FakeAdapter:
        def __init__(self, _model: object) -> None:
            self.module = _model

        def model_fingerprint(self) -> str:
            return runtime.source_model_sha256

    visits: list[tuple[int, ...]] = []

    class FakeInterventionExecutor:
        def __init__(self, _full: object) -> None:
            self.generator_ids = generator_ids
            self.generator_plan_sha256s = runtime.generator_plan_sha256s
            self.layer_count = layer_count

        def visit_generator_interaction_map(
            self,
            _model_inputs: object,
            *,
            visitor: object,
            joint_pairs: object,
        ) -> None:
            schedule = (
                (),
                *((ordinal,) for ordinal in range(layer_count)),
                *joint_pairs,
            )
            base_outputs = tuple(
                torch.full(
                    (prompt_count, 1, 2),
                    float(ordinal + 1),
                    dtype=torch.float64,
                )
                for ordinal in range(layer_count)
            )
            for suppressed in schedule:
                visits.append(suppressed)
                if not suppressed:
                    logits = baseline
                    traces = base_outputs
                elif len(suppressed) == 1:
                    muted = suppressed[0]
                    logits = singleton_logits[muted]
                    traces = tuple(
                        None
                        if ordinal == muted
                        else (
                            base_outputs[ordinal]
                            if ordinal < muted
                            else base_outputs[ordinal] + 0.001 * (muted + 1)
                        )
                        for ordinal in range(layer_count)
                    )
                else:
                    left, right = suppressed
                    logits = (
                        singleton_logits[left]
                        + singleton_logits[right]
                        - baseline
                    )
                    traces = None
                visitor(
                    SimpleNamespace(
                        suppressed_layer_ordinals=suppressed,
                        generator_plan_sha256s=runtime.generator_plan_sha256s,
                        valid_tokens=prompt_count,
                        observational_only=True,
                        mutation_authority=False,
                        model_output=SimpleNamespace(logits=logits),
                        generated_residuals=traces,
                    )
                )

    captured: dict[str, object] = {}

    def save(_path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"published": True}

    for path in (
        tmp_path / "base.pt",
        tmp_path / "refit.pt",
        tmp_path / "fingerprint.json",
    ):
        path.write_bytes(b"fixture")
    monkeypatch.setattr(
        module,
        "restore_gemma3_full_mlp_stack_refit_runtime",
        lambda *_args: runtime,
    )
    monkeypatch.setattr(
        module,
        "load_gemma3_generator_causal_fingerprint_artifact",
        lambda _path: fingerprint,
    )
    monkeypatch.setattr(
        module,
        "load_development_prompt_export",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        module,
        "partition_development_export_for_interactions",
        lambda *_args, **_kwargs: partition,
    )
    monkeypatch.setattr(module, "resolve_torch_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(
        module,
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(
        module,
        "load_gemma3",
        lambda **_kwargs: (object(), FakeModel()),
    )
    monkeypatch.setattr(module, "Gemma3CausalLMAdapter", FakeAdapter)
    monkeypatch.setattr(
        module,
        "_materialize_split",
        lambda *_args, **_kwargs: ((batch,), object()),
    )
    monkeypatch.setattr(
        module,
        "_safe_tokenized_stream_metadata",
        lambda _stream: safe_stream,
    )
    monkeypatch.setattr(
        module,
        "_stream_content_sha256s",
        lambda *_args, **_kwargs: runtime.analysis_split["content_sha256"],
    )
    monkeypatch.setattr(
        module,
        "Gemma3FullMLPStackExecutor",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        module,
        "FrozenGemma3GeneratorCausalInterventionExecutor",
        FakeInterventionExecutor,
    )
    monkeypatch.setattr(
        module,
        "save_gemma3_generator_causal_map_artifact",
        save,
    )

    result = run_gemma3_generator_causal_map_experiment(
        eval_export_path=tmp_path / "prompts.json",
        revision=revision,
        base_artifact_path=tmp_path / "base.pt",
        refit_artifact_path=tmp_path / "refit.pt",
        fingerprint_artifact_path=tmp_path / "fingerprint.json",
        output=tmp_path / "map.json",
        tokenization_batch_size=20,
    )

    assert result == {"published": True}
    assert len(visits) == 1 + 18 + 153
    assert len(captured["generator_nodes"]) == 18
    assert len(captured["copied_pairwise_similarities"]) == 153
    assert len(captured["joint_interactions"]) == 153
    assert len(captured["directed_edges"]) == 153
    assert len(captured["prompt_cohorts"]) == 8
    assert len(captured["generator_cohort_affinities"]) == 18 * 8
    assert captured["cohort_partition_lineage"] == {
        "partition_plan_sha256": partition.artifact_sha256,
        "assessment_partition_sha256": assessment.artifact_sha256,
        "source_export_sha256": assessment.source_export_sha256,
        "source_fit_prompt_index_sha256": (
            assessment.source_fit_prompt_index_sha256
        ),
        "role": "open_development_assessment",
        "assessment_status": "open_development_not_closed_guard",
        "membership_provenance": "caller_declared_self_attested",
        "membership_externally_authenticated": False,
        "serialized_contains_prompt_text": False,
        "prompt_count": 20,
        "family_count": 8,
        "family_id_storage": "domain_separated_sha256_only",
        "exact_declared_family_membership": True,
    }
