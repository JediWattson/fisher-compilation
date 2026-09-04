from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fisher_graph.fisher_local_chart import (
    ChartBaselineSet,
    EuclideanChartBasis,
    FrozenAtlas,
    LocalChart,
    LowRankFisherChartBasis,
    NeighborhoodAssignments,
    Rank64FisherCarrier,
)
from fisher_graph.fisher_local_chart_protocol import (
    freeze_fisher_local_chart_candidate,
    load_fisher_local_chart_frozen_candidate,
)
from fisher_graph.gemma3_fisher_local_chart_capture import make_mlp_probe_partition
from fisher_graph import gemma3_fisher_local_chart_probe as probe


_DIGESTS = {
    "protocol": "1" * 64,
    "split": "2" * 64,
    "carrier": "3" * 64,
    "corpus": "4" * 64,
    "guard": "5" * 64,
    "model_config": "6" * 64,
    "model": "7" * 64,
    "execution": "8" * 64,
}

_EXPECTED_FINAL_ARTIFACT_NAMES = {
    "exact_layer4_boundary_design.md",
    "experiment_manifest.json",
    "sequence_split_manifest.json",
    "subregion_definitions.json",
    "mlp_partitions.json",
    "chart_rank_tables.json",
    "tangent_capture_tables.json",
    "affine_generator_tables.json",
    "attention_context_comparison.json",
    "teacher_composed_comparison.json",
    "chart_support_tables.json",
    "fisher_similarity_matrices.json",
    "typed_transition_matrices.json",
    "chart_transition.dot",
    "full_sequence_fidelity.json",
    "resource_runtime.json",
    "final_report.md",
    "commands.md",
    "guard_result.json",
}


def _small_carrier() -> Rank64FisherCarrier:
    return Rank64FisherCarrier(
        q=torch.eye(4, dtype=torch.float64)[:, :2],
        eigenvalues=torch.tensor([2.0, 1.0], dtype=torch.float64),
    )


def _chart_for_arm(
    arm: str,
    carrier: Rank64FisherCarrier,
) -> LocalChart:
    center = torch.zeros(4, dtype=torch.float64)
    if arm == "B4":
        basis = LowRankFisherChartBasis(
            center=center,
            carrier=carrier,
            carrier_directions=torch.eye(2, dtype=torch.float64),
            regularization=1.0e-8,
        )
    else:
        basis = EuclideanChartBasis(
            center=center,
            directions=torch.eye(4, dtype=torch.float64)[:, :2],
        )
    return LocalChart(
        basis=basis,
        coordinate_mean=torch.zeros(2, dtype=torch.float64),
        coordinate_precision=torch.eye(2, dtype=torch.float64),
        normal_scale_squared=1.0,
        tangential_radius_squared=4.0,
        tangential_weight=1.0,
    )


def _small_baselines() -> dict[str, ChartBaselineSet]:
    carrier = _small_carrier()
    assignments = torch.zeros(3, dtype=torch.long)
    neighborhoods = NeighborhoodAssignments(
        centers=torch.zeros((1, 4), dtype=torch.float64),
        assignments=assignments,
        seed=23,
        iterations=1,
        inertia=0.0,
    )
    atlases = {
        arm: FrozenAtlas(
            baseline=arm,
            charts=(_chart_for_arm(arm, carrier),),
            initial_assignments=assignments,
        )
        for arm in probe.BASELINE_IDS
    }
    baseline_set = ChartBaselineSet(neighborhoods=neighborhoods, atlases=atlases)
    return {scheme: baseline_set for scheme in probe.PARTITION_SCHEMES}


def _runner_bundle() -> dict[str, object]:
    baselines = _small_baselines()
    assignment_sha256 = probe._shared_assignment_hash(baselines)
    generators = {
        scheme: {
            arm: {
                "scheme": scheme,
                "ridge": 1.0e-4,
                "targets": {
                    "sentinel": {
                        "charts": [
                            {
                                "weight": torch.eye(2),
                                "bias": torch.zeros(2),
                                "mean": torch.zeros(2),
                            }
                        ]
                    }
                },
            }
            for arm in probe.BASELINE_IDS
        }
        for scheme in probe.PARTITION_SCHEMES
    }
    routers = {
        scheme: {
            arm: {
                stage: {
                    "models": [
                        {
                            "weight": torch.ones((1, 2)),
                            "bias": torch.tensor([float(index)]),
                        }
                    ],
                    "global_frequency": torch.ones(1),
                    "source_frequencies": [torch.ones(1)],
                }
                for stage in ("attention", "mlp")
            }
            for index, arm in enumerate(probe.BASELINE_IDS)
        }
        for scheme in probe.PARTITION_SCHEMES
    }
    config = probe.ProbeConfig(
        chart_count=1,
        coordinate_rank=2,
        ridge_grid=(1.0e-4,),
        support_coverages=(0.95,),
        chart_seed=23,
        partition_seed=29,
        bootstrap_replicates=10,
    )
    return probe._bundle_from_fit(
        baselines=baselines,
        generators=generators,
        routers=routers,
        evaluations={"roles": {"A_fit": {}, "A_selection": {}}},
        ridge_transcript={"selected_ridge": 1.0e-4},
        support_transcript={"selected_coverage": 0.95},
        selected_ridge=1.0e-4,
        selected_coverage=0.95,
        selected_scheme="interleaved",
        attention_affine_control_arm="B3",
        mlp_affine_control_arm="B3",
        assignment_sha256=assignment_sha256,
        protocol_sha256=_DIGESTS["protocol"],
        split_manifest_sha256=_DIGESTS["split"],
        carrier_sha256=_DIGESTS["carrier"],
        corpus_artifact_sha256=_DIGESTS["corpus"],
        guard_manifest_sha256=_DIGESTS["guard"],
        model_config_sha256=_DIGESTS["model_config"],
        source_model_sha256=_DIGESTS["model"],
        source_execution_sha256=_DIGESTS["execution"],
        config=config,
    )


def test_probe_config_normalizes_finite_grids() -> None:
    config = probe.ProbeConfig(
        chart_count=2,
        coordinate_rank=3,
        ridge_grid=(1, 0.25),
        support_coverages=(0.8,),
        chart_seed=0,
        partition_seed=0,
        bootstrap_replicates=1,
    )
    assert config.ridge_grid == (1.0, 0.25)
    assert config.support_coverages == (0.8,)
    assert config.to_dict()["accumulation_dtype"] == "float64"


@pytest.mark.parametrize(
    "changes",
    [
        {"chart_count": 0},
        {"chart_count": True},
        {"coordinate_rank": 0},
        {"coordinate_rank": 65},
        {"ridge_grid": ()},
        {"ridge_grid": (0.0,)},
        {"ridge_grid": (float("nan"),)},
        {"ridge_grid": (1.0e-4, 1.0e-4)},
        {"support_coverages": ()},
        {"support_coverages": (0.0,)},
        {"support_coverages": (1.01,)},
        {"support_coverages": (0.9, 0.95)},
        {"support_coverages": (0.9, 0.9)},
        {"chart_seed": -1},
        {"partition_seed": -1},
        {"bootstrap_replicates": 0},
        {"bootstrap_replicates": True},
    ],
)
def test_probe_config_rejects_invalid_or_ambiguous_values(
    changes: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "chart_count": 2,
        "coordinate_rank": 2,
        "ridge_grid": (1.0e-4,),
        "support_coverages": (0.95,),
        "chart_seed": 1,
        "partition_seed": 2,
        "bootstrap_replicates": 5,
    }
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError)):
        probe.ProbeConfig(**arguments)  # type: ignore[arg-type]


def test_config_from_args_uses_same_fail_closed_validation() -> None:
    args = argparse.Namespace(
        chart_count=4,
        coordinate_rank=0,
        ridge_grid=[1.0e-4],
        support_coverages=[0.95],
        chart_seed=1,
        partition_seed=2,
        bootstrap_replicates=5,
    )
    with pytest.raises(ValueError, match="coordinate_rank"):
        probe._config_from_args(args)


def test_protocol_identity_rejects_live_source_snapshot_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe,
        "_protocol_source_sha256s",
        lambda: {"unexpected-live-source.py": "0" * 64},
    )

    with pytest.raises(RuntimeError, match="source files changed after.*imported"):
        probe._protocol_sha256(
            probe.ProbeConfig(),
            split_manifest_sha256=_DIGESTS["split"],
            carrier_sha256=_DIGESTS["carrier"],
        )


def test_fit_select_checks_protocol_before_capture_and_aborts_pre_freeze_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    protocol_calls = 0

    def protocol_identity(*_args: object, **_kwargs: object) -> str:
        nonlocal protocol_calls
        protocol_calls += 1
        events.append(f"protocol-{protocol_calls}")
        return ("a" if protocol_calls < 3 else "b") * 64

    carrier_artifact = SimpleNamespace(
        carrier=_small_carrier(),
        carrier_sha256=_DIGESTS["carrier"],
        config_sha256=_DIGESTS["model_config"],
    )
    corpus = SimpleNamespace(
        guard_opened=False,
        guard_consumed=False,
        artifact=SimpleNamespace(artifact_sha256=_DIGESTS["corpus"]),
        open_development_role=lambda role: role,
        preclaim_view=lambda _role: SimpleNamespace(
            manifest_sha256=_DIGESTS["guard"]
        ),
    )
    runtime = SimpleNamespace(
        carrier_artifact=carrier_artifact,
        split_manifest=SimpleNamespace(manifest_sha256=_DIGESTS["split"]),
        adapter=SimpleNamespace(
            model_fingerprint=lambda: _DIGESTS["model"],
            execution_fingerprint=lambda: _DIGESTS["execution"],
        ),
        corpus=corpus,
    )
    baselines = _small_baselines()
    generators = {
        scheme: {arm: {} for arm in probe.BASELINE_IDS}
        for scheme in probe.PARTITION_SCHEMES
    }

    def capture(*_args: object, **_kwargs: object) -> object:
        events.append("capture")
        return SimpleNamespace()

    arm_evaluation = {
        "stage_outputs": {
            "composed": {"fisher_relative_squared_error": 0.0},
        },
        "affine_generators": {
            "component_sums": {
                name: {"affine": {"fisher_relative_squared_error": 0.0}}
                for name in ("attention_causal_context", "mlp_teacher_forced")
            }
        },
    }

    def bundle_from_fit(**_kwargs: object) -> dict[str, object]:
        events.append("bundle")
        return {"ready": True}

    def freeze_candidate(**_kwargs: object) -> object:
        events.append("freeze")
        raise AssertionError("candidate freeze must not run after protocol drift")

    monkeypatch.setattr(probe, "_runtime_context", lambda **_kwargs: runtime)
    monkeypatch.setattr(probe, "_protocol_sha256", protocol_identity)
    monkeypatch.setattr(probe, "_materialize_panel", lambda *_args: object())
    monkeypatch.setattr(probe, "_capture_panel", capture)
    monkeypatch.setattr(probe, "_fit_baselines", lambda *_args, **_kwargs: baselines)
    monkeypatch.setattr(
        probe,
        "_fit_generator_grid",
        lambda *_args, **_kwargs: (1.0e-4, generators, {}),
    )
    monkeypatch.setattr(
        probe,
        "_select_support_coverage",
        lambda *_args, **_kwargs: (0.95, baselines, {}),
    )
    monkeypatch.setattr(probe, "_role_assignments", lambda *_args: {})
    monkeypatch.setattr(probe, "_fit_routers", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        probe,
        "_evaluate_arm",
        lambda **_kwargs: arm_evaluation,
    )
    monkeypatch.setattr(
        probe,
        "_fisher_similarity_matrix",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(probe, "_bundle_from_fit", bundle_from_fit)
    monkeypatch.setattr(
        probe,
        "freeze_fisher_local_chart_candidate",
        freeze_candidate,
    )
    args = SimpleNamespace(
        candidate=tmp_path / "candidate.pt",
        output_dir=tmp_path,
        corpus_artifact=tmp_path / "corpus.json",
        fit_input=tmp_path / "fit.json",
        selection_input=tmp_path / "selection.json",
        guard_input=tmp_path / "guard.json",
        carrier=tmp_path / "carrier.pt",
        cache_dir=None,
    )

    with pytest.raises(RuntimeError, match="protocol identity changed before candidate freeze"):
        probe._run_fit_select(args, config=probe.ProbeConfig())

    assert events == [
        "protocol-1",
        "capture",
        "capture",
        "protocol-2",
        "bundle",
        "protocol-3",
    ]


def test_support_coverage_is_preregistered_not_outcome_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = probe.ProbeConfig(
        chart_count=1,
        coordinate_rank=2,
        ridge_grid=(1.0e-4,),
        support_coverages=(0.9,),
        bootstrap_replicates=2,
    )
    baseline_set = SimpleNamespace(atlases={"B4": object()})
    baselines = {scheme: baseline_set for scheme in probe.PARTITION_SCHEMES}
    selection = SimpleNamespace(
        h_A=torch.zeros((2, 4), dtype=torch.float64),
        h_plus=torch.zeros((2, 4), dtype=torch.float64),
    )
    observed: list[float] = []

    def fake_calibrate(*args: object, coverage: float, **kwargs: object) -> object:
        del args, kwargs
        observed.append(coverage)
        return baselines

    monkeypatch.setattr(probe, "_calibrate_support", fake_calibrate)
    monkeypatch.setattr(
        probe,
        "_composed_prediction",
        lambda *_: {"composed_h_plus": selection.h_plus},
    )
    monkeypatch.setattr(
        probe,
        "_role_assignments",
        lambda *_: {"h_A": torch.zeros(2, dtype=torch.long)},
    )
    monkeypatch.setattr(
        probe,
        "_assigned_support",
        lambda *_: (torch.tensor([True, False]), {}),
    )
    selected, calibrated, transcript = probe._select_support_coverage(
        baselines,
        {scheme: {"B4": {}} for scheme in probe.PARTITION_SCHEMES},
        selection,
        _small_carrier(),
        config=config,
    )
    assert selected == 0.9
    assert calibrated is baselines
    assert observed == [0.9]
    assert transcript["preregistered"] is True
    assert "scores" not in transcript


def test_shared_assignment_receipt_rejects_cross_scheme_center_drift() -> None:
    baselines = _small_baselines()
    interleaved = baselines["interleaved"]
    assignments = interleaved.neighborhoods.assignments
    baselines["interleaved"] = ChartBaselineSet(
        neighborhoods=NeighborhoodAssignments(
            centers=torch.ones((1, 4), dtype=torch.float64),
            assignments=assignments,
            seed=interleaved.neighborhoods.seed,
            iterations=interleaved.neighborhoods.iterations,
            inertia=interleaved.neighborhoods.inertia,
        ),
        atlases=interleaved.atlases,
    )

    with pytest.raises(ValueError, match="center"):
        probe._shared_assignment_hash(baselines)


def test_runner_candidate_bundle_survives_strict_weights_only_roundtrip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.pt"
    candidate = freeze_fisher_local_chart_candidate(
        bundle=_runner_bundle(),
        protocol_sha256=_DIGESTS["protocol"],
        split_manifest_sha256=_DIGESTS["split"],
        carrier_sha256=_DIGESTS["carrier"],
        path=path,
    )
    restored = load_fisher_local_chart_frozen_candidate(path)

    assert restored.candidate_sha256 == candidate.candidate_sha256
    assert restored.candidate_file_sha256 == candidate.candidate_file_sha256
    assert restored.bundle["bindings"]["source_execution_sha256"] == _DIGESTS[  # type: ignore[index]
        "execution"
    ]
    selected = restored.bundle["selected_hyperparameters"]
    assert selected["selected_baseline"] == "B4"  # type: ignore[index]
    assert selected["selected_partition_scheme"] == "interleaved"  # type: ignore[index]
    arm, scheme, atlas, centers, generators, routers = probe._selected_candidate_parts(
        restored, _small_carrier()
    )
    assert (arm, scheme, atlas.baseline) == ("B4", "interleaved", "B4")
    assert torch.equal(centers, torch.zeros((1, 4), dtype=torch.float64))
    assert generators["scheme"] == "interleaved"
    assert torch.equal(
        routers["attention"]["models"][0]["bias"],  # type: ignore[index]
        torch.tensor([4.0]),
    )
    assert isinstance(atlas.charts[0].basis, LowRankFisherChartBasis)


def test_candidate_reload_rejects_tampered_assignment_receipt(
    tmp_path: Path,
) -> None:
    bundle = _runner_bundle()
    baselines = bundle["baselines"]
    for arm in probe.BASELINE_IDS:
        baselines[arm]["assignment_sha256"] = "0" * 64  # type: ignore[index]
    candidate = freeze_fisher_local_chart_candidate(
        bundle=bundle,
        protocol_sha256=_DIGESTS["protocol"],
        split_manifest_sha256=_DIGESTS["split"],
        carrier_sha256=_DIGESTS["carrier"],
        path=tmp_path / "tampered-assignment-receipt.pt",
    )
    restored = load_fisher_local_chart_frozen_candidate(candidate.path)

    with pytest.raises(ValueError, match="fixed-geometry receipt"):
        probe._candidate_shared_geometry(restored.bundle)


def test_candidate_reload_rejects_cross_scheme_center_inconsistency(
    tmp_path: Path,
) -> None:
    bundle = _runner_bundle()
    baselines = bundle["baselines"]
    for arm in probe.BASELINE_IDS:
        interleaved = baselines[arm]["schemes"]["interleaved"]  # type: ignore[index]
        interleaved["fixed_neighborhood_centers"] = torch.ones(  # type: ignore[index]
            (1, 4), dtype=torch.float64
        )
        for chart in interleaved["charts"]:  # type: ignore[index]
            chart["center"] = torch.ones(4, dtype=torch.float64)
    candidate = freeze_fisher_local_chart_candidate(
        bundle=bundle,
        protocol_sha256=_DIGESTS["protocol"],
        split_manifest_sha256=_DIGESTS["split"],
        carrier_sha256=_DIGESTS["carrier"],
        path=tmp_path / "inconsistent-centers.pt",
    )
    restored = load_fisher_local_chart_frozen_candidate(candidate.path)

    with pytest.raises(ValueError, match="center"):
        probe._candidate_shared_geometry(restored.bundle)


def test_capacity_signature_is_derived_from_all_serialized_model_families() -> None:
    bundle = _runner_bundle()
    baselines = bundle["baselines"]
    signature = baselines["B0"]["capacity_signature"]  # type: ignore[index]
    assert signature["generator_affine_weight_elements"] == 4  # type: ignore[index]
    assert signature["generator_affine_bias_elements"] == 2  # type: ignore[index]
    assert signature["generator_constant_mean_elements"] == 2  # type: ignore[index]
    assert signature["router_affine_weight_elements"] == 4  # type: ignore[index]
    assert signature["router_affine_bias_elements"] == 2  # type: ignore[index]
    assert signature["router_global_frequency_elements"] == 2  # type: ignore[index]
    assert signature["router_source_frequency_elements"] == 2  # type: ignore[index]
    assert signature["total_serialized_model_elements"] == 18  # type: ignore[index]
    assert all(
        baselines[arm]["capacity_signature"] == signature  # type: ignore[index]
        for arm in probe.BASELINE_IDS
    )

    generators = bundle["generators"]
    routers = bundle["routers"]
    generators["contiguous"]["B0"]["targets"]["sentinel"]["charts"][0][  # type: ignore[index]
        "weight"
    ] = torch.ones((2, 3))
    with pytest.raises(ValueError, match="equal actual capacity"):
        probe._equal_capacity_signature(generators, routers)  # type: ignore[arg-type]


def test_teacher_forced_and_composed_predictions_use_distinct_mlp_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = SimpleNamespace(
        h=torch.zeros((2, 3), dtype=torch.float64),
        h_A=torch.full((2, 3), 10.0, dtype=torch.float64),
    )
    baseline_set = SimpleNamespace(
        neighborhoods=SimpleNamespace(
            centers=torch.zeros((1, 3), dtype=torch.float64)
        )
    )
    assignments = {
        "h": torch.zeros(2, dtype=torch.long),
        "h_A": torch.zeros(2, dtype=torch.long),
        "h_plus": torch.zeros(2, dtype=torch.long),
    }
    monkeypatch.setattr(probe, "_role_assignments", lambda *_: assignments)

    def fake_components(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        del args, kwargs
        return {
            "attention_components": torch.full((2, 4, 3), 0.25),
            "attention_sum": torch.ones((2, 3), dtype=torch.float64),
            "mlp_components": torch.full((2, 4, 3), 0.5),
            "mlp_sum": torch.full((2, 3), 2.0, dtype=torch.float64),
        }

    monkeypatch.setattr(probe, "_predict_components", fake_components)
    composed_sources: list[torch.Tensor] = []

    def fake_generator(
        atlas: object,
        model: object,
        source_states: torch.Tensor,
        assignments: torch.Tensor,
        *,
        extra_features: torch.Tensor | None,
        constant: bool = False,
    ) -> torch.Tensor:
        del atlas, model, assignments, extra_features, constant
        composed_sources.append(source_states.clone())
        return torch.full_like(source_states, float(len(composed_sources)))

    monkeypatch.setattr(probe, "_predict_generator", fake_generator)
    models = {
        "scheme": "contiguous",
        "targets": {f"mlp_block_{index}": {} for index in range(4)},
    }
    result = probe._composed_prediction(
        SimpleNamespace(), models, rows, baseline_set
    )

    # Teacher forcing starts the MLP residual at the captured h_A=10 and uses
    # the MLP prediction made there.  Composition instead predicts h_A=1,
    # reroutes that state, and performs four fresh MLP calls at that state.
    assert torch.equal(result["teacher_h_plus"], torch.full((2, 3), 12.0))
    assert torch.equal(result["predicted_h_A"], torch.ones((2, 3)))
    assert torch.equal(result["composed_h_plus"], torch.full((2, 3), 11.0))
    assert len(composed_sources) == 4
    assert all(torch.equal(source, result["predicted_h_A"]) for source in composed_sources)
    assert not torch.equal(result["teacher_h_plus"], result["composed_h_plus"])


def test_transition_entropy_averages_only_populated_source_charts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = {
        "h": torch.tensor([0, 0]),
        "h_A": torch.tensor([0, 1]),
        "h_plus": torch.tensor([2, 2]),
    }
    monkeypatch.setattr(
        probe,
        "_router_probabilities",
        lambda *_args, **_kwargs: torch.full(
            (2, 3), 1.0 / 3.0, dtype=torch.float64
        ),
    )
    result = probe._evaluate_router(
        SimpleNamespace(chart_count=3),
        {},
        SimpleNamespace(row_count=2),
        assignments,
    )
    attention = result["attention"]["transition_structure"]  # type: ignore[index]
    assert attention["populated_source_chart_count"] == 1
    assert attention["empty_source_chart_count"] == 2
    assert attention["mean_source_transition_entropy"] == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )


def test_partition_stability_includes_both_router_stages() -> None:
    def arm(attention_ce: float, mlp_ce: float) -> dict[str, object]:
        return {
            "tangent_capture": {
                f"mlp_probe_block_{block}": {"fisher_captured_fraction": 0.5}
                for block in range(4)
            },
            "affine_generators": {
                "component_sums": {
                    "mlp_teacher_forced": {
                        "affine": {"fisher_relative_squared_error": 0.25}
                    }
                }
            },
            "router": {
                "attention": {"affine": {"cross_entropy": attention_ce}},
                "mlp": {"affine": {"cross_entropy": mlp_ce}},
            },
        }

    result = probe._partition_stability(
        {
            "contiguous": {"B4": arm(0.4, 0.2)},
            "interleaved": {"B4": arm(0.4, 0.4)},
        }
    )
    assert result["metrics"]["graph_attention_cross_entropy"] == {  # type: ignore[index]
        "contiguous": 0.4,
        "interleaved": 0.4,
    }
    assert result["metrics"]["graph_mlp_cross_entropy"] == {  # type: ignore[index]
        "contiguous": 0.2,
        "interleaved": 0.4,
    }
    assert result["partition_sensitive"] is True


def test_verdict_bootstraps_match_each_reported_ratio_and_drive_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    affine_errors = {
        "B0": (0.5, 0.6),
        "B1": (0.6, 0.7),
        "B2": (0.7, 0.8),
        "B3": (0.8, 0.9),
        "B4": (0.2, 0.3),
    }

    def evaluated_arm(arm: str) -> dict[str, object]:
        capture = 0.8 if arm == "B4" else 0.5 if arm == "B3" else 0.4
        attention_error, mlp_error = affine_errors[arm]
        tangent = {
            name: {"fisher_captured_fraction": capture}
            for name in [f"attention_head_{index}" for index in range(4)]
            + [f"mlp_probe_block_{index}" for index in range(4)]
        }
        return {
            "tangent_capture": tangent,
            "affine_generators": {
                "component_sums": {
                    "attention_causal_context": {
                        "affine": {
                            "fisher_relative_squared_error": attention_error
                        },
                        "constant": {"fisher_relative_squared_error": 0.9},
                    },
                    "attention_source_only": {
                        "affine": {"fisher_relative_squared_error": 0.4},
                        "constant": {"fisher_relative_squared_error": 0.9},
                    },
                    "mlp_teacher_forced": {
                        "affine": {"fisher_relative_squared_error": mlp_error},
                        "constant": {"fisher_relative_squared_error": 0.9},
                    },
                }
            },
            "stage_outputs": {
                "attention": {"fisher_relative_squared_error": 0.1},
                "mlp_teacher_forced": {"fisher_relative_squared_error": 0.2},
                "composed": {"fisher_relative_squared_error": 0.25},
            },
            "support": {
                "composed_support_drift": {"support_drift_rate": 0.0}
            },
            "router": {
                stage: {
                    "delta_vs_source_frequency": {
                        "cross_entropy_reduction": 0.1,
                        "top1_improvement": 0.1,
                    }
                }
                for stage in ("attention", "mlp")
            },
        }

    candidate = SimpleNamespace(
        bundle={
                "selected_hyperparameters": {
                    "selected_partition_scheme": "contiguous",
                    "attention_affine_control_arm": "B0",
                    "mlp_affine_control_arm": "B0",
                },
            "generators": {
                "contiguous": {
                    arm: {"arm": arm} for arm in probe.BASELINE_IDS
                }
            },
        }
    )
    baseline_set = SimpleNamespace(
        atlases={arm: arm for arm in probe.BASELINE_IDS}
    )
    monkeypatch.setattr(
        probe, "_baseline_set_for_candidate", lambda *_args, **_kwargs: baseline_set
    )
    monkeypatch.setattr(probe, "_role_assignments", lambda *_: {})

    def component_energy(
        atlas: str, *_args: object, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.ones((4, 8), dtype=torch.float64)
        ratio = 0.2 if atlas == "B4" else 0.5
        return target, torch.full_like(target, ratio)

    def generator_energy(
        _atlas: object, models: dict[str, str], *_args: object, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        attention, mlp = affine_errors[models["arm"]]
        target = torch.ones(4, dtype=torch.float64)
        return {
            "attention_target": target,
            "attention_residual": torch.full_like(target, attention),
            "mlp_target": target,
            "mlp_residual": torch.full_like(target, mlp),
        }

    monkeypatch.setattr(probe, "_component_fisher_energy_rows", component_energy)
    monkeypatch.setattr(probe, "_generator_fisher_energy_rows", generator_energy)
    evaluations = {
        "contiguous": {
            arm: evaluated_arm(arm) for arm in probe.BASELINE_IDS
        }
    }
    verdicts, bootstrap = probe._verdicts_and_bootstrap(
        candidate,
        _small_carrier(),
        SimpleNamespace(sequence_ids=("a", "a", "b", "b")),
        evaluations,
        {
            "gates": {"passed": True},
            "aggregate": {
                "delta_nll_per_token": 0.0,
                "source_to_candidate_kl_per_token": 0.0,
                "top1_agreement_to_source": 1.0,
            },
        },
        config=probe.ProbeConfig(
            chart_count=1,
            coordinate_rank=2,
            ridge_grid=(1.0e-4,),
            support_coverages=(0.95,),
            bootstrap_replicates=20,
        ),
    )
    tangent = verdicts["fisher_tangent_capture"]
    attention = verdicts["affine_generators"]["attention"]  # type: ignore[index]
    mlp = verdicts["affine_generators"]["mlp_teacher_forced"]  # type: ignore[index]
    assert tangent["delta_tangent"] == pytest.approx(0.3)  # type: ignore[index]
    assert tangent["bootstrap_95"]["point"] == pytest.approx(0.3)  # type: ignore[index]
    assert tangent["status"] == "positive"  # type: ignore[index]
    assert attention["delta_affine"] == pytest.approx(0.3)
    assert attention["bootstrap_95"]["point"] == pytest.approx(0.3)
    assert attention["best_euclidean_control_arm"] == "B0"
    assert attention["status"] == "positive"
    assert mlp["delta_affine"] == pytest.approx(0.3)
    assert mlp["bootstrap_95"]["point"] == pytest.approx(0.3)
    assert mlp["best_euclidean_control_arm"] == "B0"
    assert mlp["status"] == "positive"
    assert set(bootstrap) == {
        "tangent_B4_vs_B3_mean_component_capture_delta",
        "affine_attention_B4_vs_best_euclidean_control_error_delta",
        "affine_mlp_teacher_forced_B4_vs_best_euclidean_control_error_delta",
    }
    assert probe._ci_evidence_status({"lower_95": -0.1, "upper_95": 0.2}) == (
        "inconclusive"
    )


def test_fisher_rank_curve_full_rank_matches_the_chart_projector() -> None:
    carrier = _small_carrier()
    chart = _chart_for_arm("B4", carrier)
    atlas = FrozenAtlas(
        baseline="B4",
        charts=(chart,),
        initial_assignments=torch.zeros(3, dtype=torch.long),
    )
    values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.0, -1.0]],
        dtype=torch.float64,
    )
    assignments = torch.zeros(values.shape[0], dtype=torch.long)

    curve = probe._rank_curve(atlas, values, assignments, carrier)
    direct = probe._flow_metrics(atlas, values, assignments, carrier)

    assert curve[-1]["rank"] == atlas.coordinate_rank
    assert curve[-1]["euclidean_captured_fraction"] == pytest.approx(
        direct["euclidean_captured_fraction"]
    )
    assert curve[-1]["fisher_captured_fraction"] == pytest.approx(
        direct["fisher_captured_fraction"]
    )


def test_geometry_flows_are_attached_only_to_their_actual_source_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 2
    width = 4
    h = torch.zeros((rows, width))
    h_a = torch.ones((rows, width))
    h_plus = torch.full((rows, width), 3.0)
    v_a = h_a - h
    v_m = h_plus - h_a
    attention = torch.arange(rows * 4 * width, dtype=torch.float32).reshape(
        rows, 4, width
    )
    mlp = {
        scheme: attention + (100.0 if scheme == "contiguous" else 200.0)
        for scheme in probe.PARTITION_SCHEMES
    }
    fit = SimpleNamespace(
        h=h,
        h_A=h_a,
        h_plus=h_plus,
        v_A=v_a,
        v_M=v_m,
        v_total=h_plus - h,
        attention_components=attention,
        mlp_components=mlp,
        state_cloud=torch.cat((h, h_a, h_plus)),
        row_count=rows,
    )
    observed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def fake_fit(*, total_flows: torch.Tensor, component_flows: torch.Tensor, **kwargs: object) -> object:
        del kwargs
        scheme = probe.PARTITION_SCHEMES[len(observed)]
        observed[scheme] = (total_flows.clone(), component_flows.clone())
        return SimpleNamespace(
            neighborhoods=SimpleNamespace(centers=torch.zeros((1, width)))
        )

    monkeypatch.setattr(probe, "fit_equal_capacity_chart_baselines", fake_fit)
    monkeypatch.setattr(probe, "_shared_assignment_hash", lambda *_: "receipt")
    probe._fit_baselines(
        fit,
        _small_carrier(),
        config=probe.ProbeConfig(
            chart_count=1,
            coordinate_rank=2,
            ridge_grid=(1.0e-4,),
            support_coverages=(0.95,),
        ),
    )

    for scheme, (totals, components) in observed.items():
        assert torch.equal(totals[:rows, 0], v_a)
        assert torch.equal(totals[:rows, 1], fit.v_total)
        assert torch.count_nonzero(totals[:rows, 2]) == 0
        assert torch.equal(totals[rows : 2 * rows, 0], v_m)
        assert torch.count_nonzero(totals[rows : 2 * rows, 1:]) == 0
        assert torch.count_nonzero(totals[2 * rows :]) == 0
        assert torch.equal(components[:rows, :4], attention)
        assert torch.count_nonzero(components[:rows, 4:]) == 0
        assert torch.count_nonzero(components[rows : 2 * rows, :4]) == 0
        assert torch.equal(components[rows : 2 * rows, 4:], mlp[scheme])
        assert torch.count_nonzero(components[2 * rows :]) == 0


def test_final_artifact_name_contract_is_exact() -> None:
    assert set(probe.FINAL_ARTIFACT_NAMES) == _EXPECTED_FINAL_ARTIFACT_NAMES
    assert len(probe.FINAL_ARTIFACT_NAMES) == len(_EXPECTED_FINAL_ARTIFACT_NAMES)


def test_final_artifact_preparation_emits_every_prerendered_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"candidate")
    candidate = SimpleNamespace(
        path=candidate_path,
        candidate_sha256="a" * 64,
        candidate_file_sha256="b" * 64,
        protocol_sha256="c" * 64,
        bundle={
            "selected_hyperparameters": {
                "selected_baseline": "B4",
                "selected_partition_scheme": "contiguous",
            },
            "generators": {"contiguous": {}, "interleaved": {}},
            "routers": {"contiguous": {}, "interleaved": {}},
            "baselines": {},
            "selection_transcript": {
                "evaluations": {"fisher_similarity_matrices": {}}
            },
            "bindings": {
                "source_model_sha256": "d" * 64,
                "source_execution_sha256": "e" * 64,
            },
        },
    )
    carrier_artifact = SimpleNamespace(
        carrier=object(),
        metadata=lambda: {"carrier_sha256": "e" * 64},
    )
    runtime = SimpleNamespace(
        carrier_artifact=carrier_artifact,
        split_manifest=SimpleNamespace(to_dict=lambda: {"split": "manifest"}),
    )
    claim = SimpleNamespace(metadata=lambda: {"claim_sha256": "f" * 64})
    guard_rows = SimpleNamespace(
        teacher_forward_count=8,
        capture_seconds=1.0,
        sequence_ids=("guard-a", "guard-a", "guard-b"),
    )
    preclaim_selection_rows = SimpleNamespace(
        teacher_forward_count=8,
        capture_seconds=1.5,
    )
    fit_report = {
        "capture": {
            "A_fit": {
                "sequence_count": 8,
                "teacher_forward_count": 16,
                "seconds": 2.0,
            },
            "A_selection": {
                "sequence_count": 4,
                "teacher_forward_count": 8,
                "seconds": 1.0,
            },
        },
        "fit_seconds": 3.0,
        "evaluation_seconds": 4.0,
    }
    verdicts = {
        "fisher_tangent_capture": {},
        "affine_generators": {},
        "attention_context_sufficiency": {},
        "composition": {},
    }
    monkeypatch.setattr(probe, "_capture_invariant_summary", lambda *_: {})
    monkeypatch.setattr(
        probe, "_transition_artifact", lambda *_: ({"typed": "transition"}, "digraph{}\n")
    )
    monkeypatch.setattr(probe, "_fisher_similarity_matrix", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(probe, "_evaluations_section", lambda *_: {})
    monkeypatch.setattr(probe, "_next_experiment", lambda *_: "next")
    monkeypatch.setattr(probe, "_markdown_report", lambda **_: "# report\n")
    monkeypatch.setattr(probe, "_commands_markdown", lambda *_: "# commands\n")

    _summary, encoded = probe._prepare_final_artifacts(
        args=SimpleNamespace(),
        config=probe.ProbeConfig(bootstrap_replicates=2),
        runtime=runtime,
        candidate=candidate,
        claim=claim,
        fit_report=fit_report,
        roles={"A_guard": {}},
        guard_rows=guard_rows,
        preclaim_selection_rows=preclaim_selection_rows,
        fidelity={},
        fidelity_forward_count=8,
        fidelity_seconds=1.0,
        guard_evaluation_seconds=1.0,
        verdicts=verdicts,
        bootstrap={},
        partition_stability={},
    )

    assert set(encoded) == _EXPECTED_FINAL_ARTIFACT_NAMES - {"guard_result.json"}
    assert all(isinstance(payload, bytes) and payload for payload in encoded.values())
    design = encoded["exact_layer4_boundary_design.md"].decode("utf-8")
    assert "layer.4.input" in design
    assert "layer.4.output" in design
    assert "shared" in design.lower()
    resources = json.loads(encoded["resource_runtime.json"])
    assert resources["teacher_capture_forward_count"] == 40
    assert resources["capture_seconds"]["A_selection_preclaim_dry_run"] == 1.5
    assert resources["sequence_counts"] == {
        "A_fit": 8,
        "A_selection": 4,
        "A_guard": 2,
    }


def test_markdown_report_uses_observed_corpus_sequence_counts() -> None:
    candidate = SimpleNamespace(
        candidate_sha256="a" * 64,
        bundle={
            "selected_hyperparameters": {
                "selected_baseline": "B4",
                "selected_partition_scheme": "contiguous",
                "chart_count": 4,
                "coordinate_rank": 8,
                "ridge": 1.0e-4,
                "support_coverage": 0.95,
            }
        },
    )
    verdicts = {
        "fisher_tangent_capture": {
            "status": "inconclusive",
            "interpretation": "interval overlaps zero",
            "B4_mean_component_capture": 0.5,
            "B3_mean_component_capture": 0.5,
            "delta_tangent": 0.0,
        },
        "affine_generators": {
            "attention": {
                "status": "inconclusive",
                "B4_fisher_relative_squared_error": 0.5,
            },
            "mlp_teacher_forced": {
                "status": "inconclusive",
                "B4_fisher_relative_squared_error": 0.5,
            },
        },
        "attention_context_sufficiency": {
            "status": "source_state_sufficient",
            "source_only_error": 0.5,
            "causal_context_error": 0.5,
        },
        "graph_routing": {"status": "mixed_or_negative"},
        "composition": {
            "status": "preserved",
            "interpretation": "preserved",
            "composed_output_error": 0.5,
            "support_drift_rate": 0.0,
        },
        "full_sequence_fidelity": {"status": "diagnostic"},
    }
    report = probe._markdown_report(
        candidate=candidate,
        carrier_metadata={"retained_trace_fraction": 0.75},
        guard_evaluations={"contiguous": {"B4": {}, "B3": {}}},
        fidelity={
            "aggregate": {
                "delta_nll_per_token": 0.0,
                "source_to_candidate_kl_per_token": 0.0,
                "top1_agreement_to_source": 1.0,
            }
        },
        verdicts=verdicts,
        partition_stability={"verdict": "stable"},
        invariant_summary={
            "absolute_tolerance": 1.0e-4,
            "relative_tolerance": 1.0e-5,
            "maximum_absolute_error": 0.0,
            "maximum_relative_l2_error": 0.0,
        },
        sequence_counts={"A_fit": 12, "A_selection": 7, "A_guard": 4},
        next_experiment="next",
    )
    assert "23 sequences (12 fit, 7 selection, 4 one-shot guard)" in report
    assert "Sixteen sequences" not in report


def test_bootstrap_and_mlp_partition_controls_are_deterministic() -> None:
    values = torch.tensor([0.0, 2.0, 10.0, 12.0], dtype=torch.float64)
    sequence_ids = ("seq-a", "seq-a", "seq-b", "seq-b")
    first = probe.sequence_bootstrap(
        values, sequence_ids, replicates=50, confidence=0.8, seed=91
    )
    second = probe.sequence_bootstrap(
        values, sequence_ids, replicates=50, confidence=0.8, seed=91
    )
    assert torch.equal(first.replicates, second.replicates)
    # Whole-sequence resampling can draw AA, AB/BA, or BB.  Row-wise
    # resampling would create many additional means.
    assert set(first.replicates.tolist()) <= {1.0, 6.0, 11.0}

    contiguous = make_mlp_probe_partition(
        17, scheme="contiguous", block_count=4, seed=13
    )
    interleaved = make_mlp_probe_partition(
        17, scheme="interleaved", block_count=4, seed=13
    )
    repeated = make_mlp_probe_partition(
        17, scheme="interleaved", block_count=4, seed=13
    )
    changed_seed = make_mlp_probe_partition(
        17, scheme="interleaved", block_count=4, seed=14
    )
    assert contiguous.channel_indices == (
        tuple(range(0, 5)),
        tuple(range(5, 9)),
        tuple(range(9, 13)),
        tuple(range(13, 17)),
    )
    assert interleaved.channel_indices == repeated.channel_indices
    assert interleaved.channel_indices != changed_seed.channel_indices
    assert sorted(
        index for block in interleaved.channel_indices for index in block
    ) == list(range(17))
    assert max(interleaved.block_sizes) - min(interleaved.block_sizes) == 1


def test_nondefault_output_directory_rebinds_default_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Path] = {}

    def fake_inspect(args: argparse.Namespace) -> dict[str, object]:
        seen["candidate"] = args.candidate
        return {"phase": "inspect"}

    monkeypatch.setattr(probe, "_run_inspect", fake_inspect)
    assert probe.main(["inspect", "--output-dir", str(tmp_path)]) == 0
    assert seen["candidate"] == tmp_path / "frozen_candidate.pt"
    assert '"phase": "inspect"' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["guard", "run"])
def test_guard_claiming_commands_require_explicit_acknowledgement(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(probe, "_run_guard", unexpected)
    monkeypatch.setattr(probe, "_run_extract_carrier", unexpected)
    with pytest.raises(SystemExit, match="2"):
        probe.main([command])
    assert called is False


def test_acknowledged_guard_reaches_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_guard(args: argparse.Namespace, *, config: probe.ProbeConfig) -> dict[str, object]:
        seen["acknowledged"] = args.acknowledge_one_shot_guard
        seen["coverage"] = config.support_coverages
        return {"phase": "guard"}

    monkeypatch.setattr(probe, "_run_guard", fake_guard)
    assert probe.main(["guard", "--acknowledge-one-shot-guard"]) == 0
    assert seen == {"acknowledged": True, "coverage": (0.95,)}
    assert '"phase": "guard"' in capsys.readouterr().out


def test_reproduction_commands_include_every_protocol_flag_and_guard_ack() -> None:
    args = probe.build_parser().parse_args(
        [
            "guard",
            "--cache-dir",
            "/tmp/model cache",
            "--chart-count",
            "3",
            "--coordinate-rank",
            "7",
            "--ridge-grid",
            "0.001",
            "0.02",
            "--support-coverages",
            "0.8",
            "--chart-seed",
            "11",
            "--partition-seed",
            "13",
            "--bootstrap-replicates",
            "17",
            "--acknowledge-one-shot-guard",
        ]
    )
    markdown = probe._commands_markdown(args)
    commands = [
        shlex.split(line)
        for line in markdown.splitlines()
        if "fisher_graph.gemma3_fisher_local_chart_probe" in line
    ]
    guard = next(tokens for tokens in commands if "guard" in tokens)
    fit = next(tokens for tokens in commands if "fit-select" in tokens)
    for tokens in (guard, fit):
        assert tokens[tokens.index("--cache-dir") + 1] == "/tmp/model cache"
        assert tokens[tokens.index("--chart-count") + 1] == "3"
        assert tokens[tokens.index("--coordinate-rank") + 1] == "7"
        assert tokens[tokens.index("--ridge-grid") + 1 : tokens.index("--support-coverages")] == [
            "0.001",
            "0.02",
        ]
        assert tokens[tokens.index("--support-coverages") + 1] == "0.80000000000000004"
        assert tokens[tokens.index("--chart-seed") + 1] == "11"
        assert tokens[tokens.index("--partition-seed") + 1] == "13"
        assert tokens[tokens.index("--bootstrap-replicates") + 1] == "17"
    assert "--acknowledge-one-shot-guard" in guard
    assert "--acknowledge-one-shot-guard" not in fit


def _selection_grid(value: float = 1.0) -> dict[str, object]:
    return {
        scheme: {
            arm: {"cell": f"{scheme}.{arm}", "metric": value}
            for arm in probe.BASELINE_IDS
        }
        for scheme in probe.PARTITION_SCHEMES
    }


def test_complete_candidate_evaluation_executes_every_scheme_and_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    candidate = SimpleNamespace(
        bundle={
            "generators": {
                scheme: {arm: {} for arm in probe.BASELINE_IDS}
                for scheme in probe.PARTITION_SCHEMES
            },
            "routers": {
                scheme: {arm: {} for arm in probe.BASELINE_IDS}
                for scheme in probe.PARTITION_SCHEMES
            },
        }
    )

    def fake_baselines(*_args: object, scheme: str, **_kwargs: object) -> object:
        return SimpleNamespace(
            atlases={arm: f"{scheme}.{arm}.atlas" for arm in probe.BASELINE_IDS}
        )

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        scheme = str(kwargs["scheme"])
        arm = str(kwargs["arm"])
        calls.append((scheme, arm))
        return {"cell": f"{scheme}.{arm}"}

    monkeypatch.setattr(probe, "_baseline_set_for_candidate", fake_baselines)
    monkeypatch.setattr(probe, "_evaluate_arm", fake_evaluate)
    result = probe._evaluate_candidate_role(
        candidate, SimpleNamespace(), SimpleNamespace()
    )

    assert calls == [
        (scheme, arm)
        for scheme in probe.PARTITION_SCHEMES
        for arm in probe.BASELINE_IDS
    ]
    assert set(result) == set(probe.PARTITION_SCHEMES)
    assert all(
        set(result[scheme]) == set(probe.BASELINE_IDS)  # type: ignore[arg-type]
        for scheme in probe.PARTITION_SCHEMES
    )


def test_preclaim_dry_run_requires_exact_frozen_selection_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _selection_grid()
    candidate = SimpleNamespace(
        bundle={
            "selection_transcript": {
                "evaluations": {"roles": {"A_selection": expected}}
            }
        }
    )
    monkeypatch.setattr(probe, "_evaluate_candidate_role", lambda *_: expected)
    assert probe._candidate_model_free_dry_run(
        candidate, SimpleNamespace(), SimpleNamespace()
    ) == expected

    changed = _selection_grid()
    changed["contiguous"]["B3"]["metric"] = 2.0  # type: ignore[index]
    monkeypatch.setattr(probe, "_evaluate_candidate_role", lambda *_: changed)
    with pytest.raises(ValueError, match="differ from the frozen transcript"):
        probe._candidate_model_free_dry_run(
            candidate, SimpleNamespace(), SimpleNamespace()
        )


def test_preclaim_dry_run_rejects_incomplete_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _selection_grid()
    candidate = SimpleNamespace(
        bundle={
            "selection_transcript": {
                "evaluations": {"roles": {"A_selection": expected}}
            }
        }
    )
    incomplete = _selection_grid()
    del incomplete["interleaved"]["B4"]  # type: ignore[index]
    monkeypatch.setattr(probe, "_evaluate_candidate_role", lambda *_: incomplete)
    with pytest.raises(ValueError, match="omitted a scheme/baseline cell"):
        probe._candidate_model_free_dry_run(
            candidate, SimpleNamespace(), SimpleNamespace()
        )


def _freeze_runner_candidate(tmp_path: Path) -> Any:
    return freeze_fisher_local_chart_candidate(
        bundle=_runner_bundle(),
        protocol_sha256=_DIGESTS["protocol"],
        split_manifest_sha256=_DIGESTS["split"],
        carrier_sha256=_DIGESTS["carrier"],
        path=tmp_path / "candidate.pt",
    )


def _fit_report_for(candidate: Any) -> dict[str, object]:
    transcript = candidate.bundle["selection_transcript"]
    return {
        "schema": probe.SCHEMA,
        "phase": "fit-select",
        "candidate": {
            "path": str(candidate.path),
            "manifest_path": str(candidate.manifest_path),
            "candidate_sha256": candidate.candidate_sha256,
            "candidate_file_sha256": candidate.candidate_file_sha256,
            "protocol_sha256": candidate.protocol_sha256,
            "split_manifest_sha256": candidate.split_manifest_sha256,
            "carrier_sha256": candidate.carrier_sha256,
        },
        "bindings": dict(candidate.bundle["bindings"]),
        "selected": dict(candidate.bundle["selected_hyperparameters"]),
        "evaluations": dict(transcript["evaluations"]),
        "guard_opened": False,
        "guard_consumed": False,
    }


def test_fit_report_authenticates_logical_file_and_execution_bindings(
    tmp_path: Path,
) -> None:
    candidate = _freeze_runner_candidate(tmp_path)
    path = tmp_path / "fit_selection_report.json"
    probe._write_json(path, _fit_report_for(candidate))

    restored, report_sha256 = probe._load_fit_report_with_sha256(path, candidate)
    receipt = restored["candidate"]
    bindings = restored["bindings"]
    assert receipt["candidate_sha256"] == candidate.candidate_sha256  # type: ignore[index]
    assert receipt["candidate_file_sha256"] == candidate.candidate_file_sha256  # type: ignore[index]
    assert bindings["source_execution_sha256"] == _DIGESTS["execution"]  # type: ignore[index]
    assert len(report_sha256) == 64


@pytest.mark.parametrize(
    "tamper",
    ("candidate_file_sha256", "source_execution_sha256", "evaluations"),
)
def test_fit_report_rejects_candidate_binding_or_transcript_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    candidate = _freeze_runner_candidate(tmp_path)
    report = _fit_report_for(candidate)
    if tamper == "candidate_file_sha256":
        report["candidate"]["candidate_file_sha256"] = "0" * 64  # type: ignore[index]
    elif tamper == "source_execution_sha256":
        report["bindings"]["source_execution_sha256"] = "0" * 64  # type: ignore[index]
    else:
        report["evaluations"] = {"changed": True}
    path = tmp_path / "fit_selection_report.json"
    probe._write_json(path, report)
    with pytest.raises(ValueError):
        probe._load_fit_report_with_sha256(path, candidate)


def test_candidate_identity_includes_serialized_file_sha() -> None:
    common = {
        "protocol_sha256": "1" * 64,
        "split_manifest_sha256": "2" * 64,
        "carrier_sha256": "3" * 64,
        "candidate_sha256": "4" * 64,
    }
    first = SimpleNamespace(**common, candidate_file_sha256="5" * 64)
    second = SimpleNamespace(**common, candidate_file_sha256="6" * 64)
    assert probe._candidate_identity(first) != probe._candidate_identity(second)


def test_selection_preclaim_rehearses_fidelity_verdict_transition_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"candidate")
    selection_evaluations = _selection_grid()
    candidate = SimpleNamespace(
        path=candidate_path,
        candidate_sha256="a" * 64,
        bundle={
            "selection_transcript": {
                "evaluations": {
                    "roles": {"A_fit": {}, "A_selection": selection_evaluations}
                }
            }
        },
    )
    runtime = SimpleNamespace(
        carrier_artifact=SimpleNamespace(carrier=SimpleNamespace())
    )

    def fake_fidelity(*_args: object) -> tuple[dict[str, object], int, float]:
        calls.append("fidelity")
        return {"aggregate": {"ok": True}}, 8, 0.25

    def fake_verdicts(
        *_args: object, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        calls.append("verdicts")
        return {"verdict": "selection-rehearsal"}, {"bootstrap": True}

    def fake_partition(*_args: object) -> dict[str, object]:
        calls.append("partition")
        return {"stable": True}

    expected_encoded = {
        name: name.encode("ascii")
        for name in probe.FINAL_ARTIFACT_NAMES
        if name != "guard_result.json"
    }

    def fake_prepare(**kwargs: object) -> tuple[dict[str, object], dict[str, bytes]]:
        calls.append("report")
        assert kwargs["roles"]["A_guard"] == selection_evaluations  # type: ignore[index]
        claim = kwargs["claim"]
        assert claim.metadata()["guard_claimed"] is False  # type: ignore[attr-defined]
        return {"transition_artifact": {"prepared": True}}, expected_encoded

    monkeypatch.setattr(probe, "_full_sequence_fidelity", fake_fidelity)
    monkeypatch.setattr(probe, "_verdicts_and_bootstrap", fake_verdicts)
    monkeypatch.setattr(probe, "_partition_stability", fake_partition)
    monkeypatch.setattr(probe, "_prepare_final_artifacts", fake_prepare)
    receipt = probe._selection_preclaim_rehearsal(
        args=SimpleNamespace(),
        config=probe.ProbeConfig(bootstrap_replicates=2),
        runtime=runtime,
        candidate=candidate,
        fit_report={},
        selection_panel=SimpleNamespace(),
        selection_rows=SimpleNamespace(),
        selection_evaluations=selection_evaluations,
    )

    assert calls == ["fidelity", "verdicts", "partition", "report"]
    assert receipt["mode"] == "A_selection_only_no_guard_authority"
    assert receipt["full_sequence_forward_count"] == 8
    assert set(receipt["rendered_artifact_sha256s"]) == set(expected_encoded)  # type: ignore[arg-type]


def test_guard_rejects_candidate_file_swap_before_claim_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fit_selection_report.json").write_text("placeholder")
    bindings = {
        "source_model_sha256": "m" * 64,
        "source_execution_sha256": "x" * 64,
    }
    common = {
        "protocol_sha256": "p" * 64,
        "split_manifest_sha256": "s" * 64,
        "carrier_sha256": "c" * 64,
        "candidate_sha256": "d" * 64,
        "bundle": {"bindings": bindings},
    }
    preflight = SimpleNamespace(**common, candidate_file_sha256="f" * 64)
    swapped = SimpleNamespace(**common, candidate_file_sha256="z" * 64)
    adapter = SimpleNamespace(
        model_fingerprint=lambda: "m" * 64,
        execution_fingerprint=lambda: "x" * 64,
    )
    corpus = SimpleNamespace(
        guard_opened=False,
        guard_consumed=False,
        open_development_role=lambda _role: SimpleNamespace(),
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        corpus=corpus,
        carrier_artifact=SimpleNamespace(
            carrier=SimpleNamespace(), carrier_sha256="c" * 64
        ),
        split_manifest=SimpleNamespace(manifest_sha256="s" * 64),
    )
    monkeypatch.setattr(probe, "_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(probe, "_protocol_sha256", lambda *_args, **_kwargs: "p" * 64)
    monkeypatch.setattr(
        probe, "load_fisher_local_chart_frozen_candidate", lambda *_: preflight
    )
    monkeypatch.setattr(
        probe,
        "_load_fit_report_with_sha256",
        lambda *_: ({"bound": True}, "r" * 64),
    )
    monkeypatch.setattr(probe, "_materialize_panel", lambda *_: SimpleNamespace())
    monkeypatch.setattr(
        probe, "_capture_panel", lambda *_args, **_kwargs: SimpleNamespace()
    )

    claim_attempted = False

    def fake_claim(**kwargs: object) -> object:
        nonlocal claim_attempted
        dry_run = kwargs["dry_run"]
        dry_run(swapped)  # type: ignore[operator]
        claim_attempted = True
        raise AssertionError("claim must not be attempted")

    monkeypatch.setattr(probe, "claim_and_open_fisher_local_chart_guard", fake_claim)
    args = SimpleNamespace(
        output_dir=output_dir,
        candidate=tmp_path / "candidate.pt",
        corpus_artifact=tmp_path / "corpus.json",
        fit_input=tmp_path / "fit.json",
        selection_input=tmp_path / "selection.json",
        guard_input=tmp_path / "guard.json",
        carrier=tmp_path / "carrier.pt",
        cache_dir=None,
        acknowledge_one_shot_guard=True,
    )
    with pytest.raises(ValueError, match="changed between preflight and claim callback"):
        probe._run_guard(args, config=probe.ProbeConfig(bootstrap_replicates=2))
    assert claim_attempted is False


def test_guard_rejects_execution_fingerprint_mismatch_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "fit_selection_report.json").write_text("placeholder")
    candidate = SimpleNamespace(
        protocol_sha256="p" * 64,
        split_manifest_sha256="s" * 64,
        carrier_sha256="c" * 64,
        bundle={
            "bindings": {
                "source_model_sha256": "m" * 64,
                "source_execution_sha256": "expected-execution",
            }
        },
    )
    runtime = SimpleNamespace(
        adapter=SimpleNamespace(
            model_fingerprint=lambda: "m" * 64,
            execution_fingerprint=lambda: "different-execution",
        ),
        corpus=SimpleNamespace(),
        carrier_artifact=SimpleNamespace(
            carrier=SimpleNamespace(), carrier_sha256="c" * 64
        ),
        split_manifest=SimpleNamespace(manifest_sha256="s" * 64),
    )
    monkeypatch.setattr(probe, "_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(probe, "_protocol_sha256", lambda *_args, **_kwargs: "p" * 64)
    monkeypatch.setattr(
        probe, "load_fisher_local_chart_frozen_candidate", lambda *_: candidate
    )
    args = SimpleNamespace(
        output_dir=output_dir,
        candidate=tmp_path / "candidate.pt",
        corpus_artifact=tmp_path / "corpus.json",
        fit_input=tmp_path / "fit.json",
        selection_input=tmp_path / "selection.json",
        guard_input=tmp_path / "guard.json",
        carrier=tmp_path / "carrier.pt",
        cache_dir=None,
        acknowledge_one_shot_guard=True,
    )
    with pytest.raises(ValueError, match="execution semantics differ"):
        probe._run_guard(args, config=probe.ProbeConfig(bootstrap_replicates=2))
