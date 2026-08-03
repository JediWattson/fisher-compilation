from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_l3_l4_complete_h4_soft_polarity_held_radius_diagnostic as runner


FAMILIES = tuple(f"development_family_{index}" for index in range(8))
EXPECTED_RADII = (
    0.0,
    1.0 / 128.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
    2.0,
    4.0,
    8.0,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _radius_key(radius: float) -> str:
    return str(float(radius))


def _fold(
    family: str,
    *,
    objectives: dict[float, float],
    selected_radius: float,
    health_passed: bool = True,
) -> dict[str, object]:
    return {
        "outer_held_family_id": family,
        "radius_order": EXPECTED_RADII,
        "held_objective_by_radius": {
            _radius_key(radius): float(objectives[radius])
            for radius in EXPECTED_RADII
        },
        "v20g_selected_radius": float(selected_radius),
        "v20g_selected_objective": float(objectives[selected_radius]),
        "v20g_base_objective": float(objectives[0.0]),
        "v20g_fixed_plus_objective": float(objectives[0.0] + 1.0),
        "precommitted_cvar2_radius": float(selected_radius),
        "precommitted_cvar2_objective": float(objectives[selected_radius]),
        "diagnosis": "test_diagnostic",
        "tau_zero_exact_V20g_base_output_anchor": True,
        "V20g_selected_tau_exact_soft_output_anchor": True,
        "health_passed": health_passed,
    }


def _diagnostic_folds() -> dict[str, dict[str, object]]:
    folds: dict[str, dict[str, object]] = {}
    for index, family in enumerate(FAMILIES):
        objectives = {radius: 1.5 + radius for radius in EXPECTED_RADII}
        objectives[0.0] = 1.0 + 0.1 * index

        if index < 7:
            # The two best radii deliberately tie. The diagnostic oracle and
            # the global selector must resolve that tie toward the smaller
            # radius, independent of mapping insertion order.
            objectives[1.0 / 64.0] = 0.8 + 0.1 * index
            objectives[1.0 / 128.0] = 0.8 + 0.1 * index
            selected_radius = 1.0 / 64.0
        else:
            # One fitted direction is not rescued by any positive radius.
            for radius in EXPECTED_RADII[1:]:
                objectives[radius] = objectives[0.0] + radius
            objectives[1.0 / 128.0] = objectives[0.0] + 0.05
            objectives[1.0 / 64.0] = objectives[0.0] + 0.10
            selected_radius = 1.0 / 64.0

        folds[family] = _fold(
            family,
            objectives=objectives,
            selected_radius=selected_radius,
        )
    return folds


def test_parser_defaults_to_separate_v20h_output_and_exact_radius_ladder() -> None:
    arguments = runner.build_parser().parse_args([])

    assert runner._RADII == EXPECTED_RADII
    assert len(set(runner._RADII)) == len(runner._RADII)
    assert tuple(sorted(runner._RADII)) == runner._RADII
    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.output != runner._v20g.DEFAULT_OUTPUT
    assert runner.DEFAULT_OUTPUT.name.endswith("v20h.json")
    assert arguments.cache_dir is None


def test_output_and_fold_paths_cannot_overwrite_v20g_authority() -> None:
    with pytest.raises(ValueError, match="preserve immutable prerequisite"):
        runner._validate_output(runner._v20g.DEFAULT_OUTPUT)
    with pytest.raises(ValueError, match="under .local-runs"):
        runner._validate_output(Path("outside.json"))

    family = FAMILIES[0]
    v20h_fold = runner._fold_path(runner.DEFAULT_OUTPUT, family)
    v20g_fold = runner._v20g._fold_path(runner._v20g.DEFAULT_OUTPUT, family)

    assert v20h_fold != v20g_fold
    assert v20h_fold.parent == runner.DEFAULT_OUTPUT.resolve(strict=False).parent
    assert "v20h" in v20h_fold.name


def test_aggregate_diagnostic_reports_ties_regret_and_rescuability() -> None:
    folds = _diagnostic_folds()

    aggregate = runner._aggregate_diagnostic(folds)

    assert aggregate["family_ids"] == FAMILIES
    assert aggregate["integrity_passed"] is True
    assert aggregate["diagnostic_global_best_radius"] == 1.0 / 128.0
    assert aggregate["diagnostic_oracle_radius_by_family"] == {
        **{family: 1.0 / 128.0 for family in FAMILIES[:-1]},
        FAMILIES[-1]: 0.0,
    }
    assert aggregate["direction_rescuable_family_count"] == 7
    assert aggregate["direction_unrescuable_family_ids"] == (FAMILIES[-1],)

    expected_selected_macro = sum(
        float(fold["v20g_selected_objective"]) for fold in folds.values()
    ) / len(folds)
    expected_oracle_macro = sum(
        min(
            (float(value), float(radius))
            for radius, value in (
                (float(key), objective)
                for key, objective in fold["held_objective_by_radius"].items()
            )
        )[0]
        for fold in folds.values()
    ) / len(folds)

    assert aggregate["v20g_selected_schedule_macro_objective"] == pytest.approx(
        expected_selected_macro
    )
    assert aggregate["diagnostic_oracle_macro_objective"] == pytest.approx(
        expected_oracle_macro
    )
    assert aggregate["v20g_selected_vs_base_win_count"] == 7
    for family in FAMILIES[:-1]:
        assert aggregate["v20g_selected_regret_by_family"][family] == pytest.approx(
            0.0
        )
    assert aggregate["v20g_selected_regret_by_family"][FAMILIES[-1]] == (
        pytest.approx(0.10)
    )

    macro = aggregate["macro_objective_by_radius"]
    assert macro[_radius_key(1.0 / 128.0)] == pytest.approx(
        sum(
            fold["held_objective_by_radius"][_radius_key(1.0 / 128.0)]
            for fold in folds.values()
        )
        / len(folds)
    )
    assert aggregate["family_objectives_by_radius"][FAMILIES[0]] == folds[
        FAMILIES[0]
    ]["held_objective_by_radius"]
    assert aggregate["delta_from_base_by_family_and_radius"][FAMILIES[0]][
        _radius_key(1.0 / 128.0)
    ] == pytest.approx(-0.2)


def test_aggregate_requires_all_folds_same_ladder_and_healthy() -> None:
    folds = _diagnostic_folds()
    folds[FAMILIES[0]]["health_passed"] = False

    assert runner._aggregate_diagnostic(folds)["integrity_passed"] is False

    folds = _diagnostic_folds()
    folds[FAMILIES[0]]["radius_order"] = tuple(reversed(EXPECTED_RADII))
    with pytest.raises(ValueError, match="held curve differs"):
        runner._aggregate_diagnostic(folds)


def test_build_report_is_diagnostic_only_even_when_every_radius_looks_good(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_folds = _diagnostic_folds()
    aggregate = runner._aggregate_diagnostic(raw_folds)
    folds = {
        family: {
            "fragment_sha256": _sha(f"fragment:{family}"),
            "diagnostic_receipt": diagnostic,
        }
        for family, diagnostic in raw_folds.items()
    }
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))

    report = runner._build_report(
        output=tmp_path / "v20h.json",
        source={"artifact_sha256": _sha("source")},
        v20g_report={
            "report_sha256": _sha("v20g-report"),
            "classification": "soft_polarity_trust_region_oof_failed_rollback_to_base",
            "passed": False,
            "rollback_to_base": True,
            "final_refit": None,
        },
        panel_receipt={"artifact_sha256": _sha("panel")},
        bridge_binding_sha256=_sha("bridge"),
        fold_fragments=folds,
        diagnostic=aggregate,
    )

    assert report["passed"] is False
    assert report["full_refit_authorized"] is False
    assert report["final_refit"] is None
    assert report["final_provider_frozen"] is False
    assert report["calibration_b_eligibility_gate_passed"] is False
    assert report["calibration_b_eligible"] is False
    assert report["calibration_b_authorized"] is False
    assert report["calibration_b_opened"] is False
    assert report["rollback_to_base"] is True
    assert report["held_oracle_used_for_selection"] is False
    assert report["integrity"]["held_oracle_is_descriptive_only"] is True
    assert "diagnostic" in report["classification"]


def test_scalar_fragment_mode_and_hash_detect_tampering(tmp_path: Path) -> None:
    path = tmp_path / "v20h.fold.json"
    runner._v20g._v20b._publish_scalar_fragment(
        {"value": 1.0},
        path=path,
        domain=runner._FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="test V20h fold fragment",
    )

    assert path.stat().st_mode & 0o777 == 0o600
    raw = path.read_text().replace("1.0", "2.0")
    path.write_text(raw)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="hash drifted"):
        runner._v20g._v20b._load_scalar_fragment(
            path=path,
            domain=runner._FOLD_DOMAIN,
            hash_key="fragment_sha256",
            label="test V20h fold fragment",
        )


def test_v20g_fragment_cannot_be_promoted_by_v20h_rehash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "v20h.json"
    path = tmp_path / "v20h.fold.json"
    family = FAMILIES[0]
    monkeypatch.setattr(runner, "_fold_path", lambda *args, **kwargs: path)
    v20g_payload = {
        "schema": runner._v20g._FOLD_SCHEMA,
        "format_version": runner._v20g._FORMAT_VERSION,
        "target_output": output.resolve(strict=False).as_posix(),
        "runner_protocol_sha256": runner._v20g._RUNNER_PROTOCOL_SHA256,
        "source_artifact_sha256": _sha("source"),
        "panel_receipt_sha256": _sha("panel"),
        "bridge_binding_sha256": _sha("bridge"),
        "outer_held_family_id": family,
        "endpoint_receipt": {},
        "endpoint_evidence": {},
        "fit_receipt": {},
        "fit_training_evidence": {},
        "provider_manifest": {},
        "held_evidence": {},
        "fold_receipt": {},
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
    }
    persisted = runner._publish_fold_fragment(
        v20g_payload,
        output=output,
        outer_family_id=family,
    )

    assert persisted["fragment_sha256"] == runner._v14._sha256(
        v20g_payload,
        domain=runner._FOLD_DOMAIN,
    )
    with pytest.raises(ValueError, match="V20h fold fragment key set differs"):
        runner._load_fold_fragment(
            output=output,
            source={"artifact_sha256": _sha("source")},
            panel_receipt={"artifact_sha256": _sha("panel")},
            outer_family_id=family,
            bridge_binding_sha256=_sha("bridge"),
            authenticated_v20g_fold={"fragment_sha256": _sha("v20g-fold")},
        )


def test_all_radius_providers_freeze_before_held_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = FAMILIES[0]
    endpoint = SimpleNamespace(
        receipt={"artifact_sha256": _sha("endpoint")},
        evidence={"artifact_sha256": _sha("endpoint-evidence")},
    )
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=outer,
                example_id=f"{outer}:{index}",
            )
        )
        for index in range(2)
    )
    order: list[str] = []

    monkeypatch.setattr(
        runner._v20g,
        "_outer_endpoint",
        lambda *args, **kwargs: endpoint,
    )

    def freeze(*args, **kwargs):
        order.append("freeze")
        providers = {
            radius: SimpleNamespace(artifact_sha256=_sha(f"provider:{radius}"))
            for radius in EXPECTED_RADII
        }
        manifest = {
            "artifact_sha256": _sha("manifest"),
            "all_twelve_providers_frozen_before_held_capability": True,
            "all_twelve_traces_frozen_before_held_capability": True,
            "held_capability_count_at_freeze": 0,
            "held_objectives_or_teacher_rows_used": False,
        }
        traces = {
            radius: {"artifact_sha256": _sha(f"trace:{radius}")}
            for radius in EXPECTED_RADII
        }
        return providers, manifest, traces

    monkeypatch.setattr(runner, "_freeze_radius_providers", freeze)

    class HeldOpened(RuntimeError):
        pass

    class Vault:
        def capability(self, *args, **kwargs):
            order.append("capability")
            raise HeldOpened

    with pytest.raises(HeldOpened):
        runner._execute_outer_fold(
            object(),
            records,
            Vault(),
            family_ids=FAMILIES,
            outer_family_id=outer,
            panel_receipt={},
            authenticated_v20a_fold={},
            authenticated_v20g_fold={
                "endpoint_receipt": endpoint.receipt,
                "endpoint_evidence": endpoint.evidence,
            },
        )

    assert order == ["freeze", "capability"]


def test_existing_report_fast_path_is_model_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "existing-v20h.json"
    output.write_text("{}")
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_load_prerequisites",
        lambda: (
            {
                "nested_panel_receipt": {"artifact_sha256": _sha("panel")},
                "authenticated_bridge_binding_sha256": _sha("bridge"),
            },
            {},
            {"report_sha256": _sha("v20g-report")},
            {},
            {"artifact_sha256": _sha("source")},
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("existing V20h report constructed Gemma"),
    )
    monkeypatch.setattr(
        runner,
        "_load_existing_report",
        lambda *args, **kwargs: {"authenticated": True, "passed": False},
    )

    assert runner.run_gemma3_l3_l4_complete_h4_soft_polarity_held_radius_diagnostic(
        output=output
    ) == {"authenticated": True, "passed": False}
