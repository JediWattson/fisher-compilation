from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import gemma3_l3_l4_complete_h4_tangent_response_smoke as smoke


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_v20d_file_hash_fails_before_json_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(smoke._v20b, "_secure_stat", lambda *_a, **_k: None)
    monkeypatch.setattr(smoke._v14, "_file_sha256", lambda _path: "0" * 64)

    def forbidden_read(_self: Path) -> bytes:
        events.append("json")
        raise AssertionError("V20d JSON must not be read after file-hash failure")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(RuntimeError, match="V20d report file hash drifted"):
        smoke._load_authenticated_v20d_source()
    assert events == []


def test_existing_report_fast_path_never_constructs_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing-v20e.json"
    output.touch()
    expected = {"report_sha256": "a" * 64}
    monkeypatch.setattr(smoke, "_validate_output", lambda _value: output)
    monkeypatch.setattr(smoke, "_load_existing_report", lambda _path: expected)

    def forbidden_model(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("existing V20e output must reload without a model")

    monkeypatch.setattr(
        smoke, "prepare_complete_h4_rank320_live_context", forbidden_model
    )
    assert smoke.run_gemma3_l3_l4_complete_h4_tangent_response_smoke(
        output=output
    ) == expected


def test_inner_cv_capability_does_not_forbid_its_validation_fit_family() -> None:
    family = "fit-family"
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=family,
                example_id=f"{family}/{index}",
            )
        )
        for index in range(2)
    )

    class _Vault:
        requested_ids: tuple[str, ...] | None = None
        held_family_id: str | None = "not-called"

        def capability(
            self, example_ids: tuple[str, ...], *, held_family_id: str | None
        ) -> object:
            self.requested_ids = tuple(example_ids)
            self.held_family_id = held_family_id
            raise RuntimeError("stop after CV capability request")

    vault = _Vault()
    bank = SimpleNamespace(training_records=records)
    manifest = SimpleNamespace(
        directions={"signed_log": {family: {}}},
        rays={"signed_log": {family: {}}},
        providers={"signed_log": {family: {}}},
        receipt={"artifact_sha256": _sha("manifest")},
    )
    with pytest.raises(RuntimeError, match="stop after CV capability request"):
        smoke._score_cv_fold(
            SimpleNamespace(),
            vault,
            law="signed_log",
            validation_family_id=family,
            bank=bank,
            manifest=manifest,
        )
    assert vault.requested_ids == tuple(
        record.sequence.example_id for record in records
    )
    assert vault.held_family_id is None


def test_exact_full_cv_and_preflight_work_accounting() -> None:
    full = smoke._stage_work(held_scoring_executed=True, terminal_stage=None)
    cv_terminal = smoke._stage_work(
        held_scoring_executed=False, terminal_stage=None
    )
    early = smoke._stage_work(
        held_scoring_executed=False, terminal_stage="cv_direction_preflight"
    )
    all_six = smoke._stage_work(
        held_scoring_executed=False,
        terminal_stage="all_six_direction_preflight",
    )
    planned = full["planned_full_budget"]
    assert full["observed_stage_counters"] == planned
    assert planned["full_model_forward_count"] == 312
    assert planned["full_suffix_backward_traversal_count"] == 52
    assert planned["local_head_autograd_contraction_count"] == 36
    assert planned["total_autograd_grad_call_count"] == 88
    assert planned["unique_empirical_fisher_gradient_row_count"] == 24
    assert planned["empirical_fisher_outer_product_evaluation_count"] == 24
    assert planned["teacher_capability_access_count"] == 280
    assert planned["provider_only_runtime_trace_count"] == 136
    assert planned["total_capability_count_including_endpoint_reconstruction"] == 17
    observed = cv_terminal["observed_stage_counters"]
    assert observed["full_model_forward_count"] == 284
    assert observed["teacher_capability_access_count"] == 252
    assert observed["provider_only_runtime_trace_count"] == 122
    assert observed["total_capability_count_including_endpoint_reconstruction"] == 15
    early_observed = early["observed_stage_counters"]
    assert early_observed["full_model_forward_count"] == 68
    assert early_observed["cv_validation_forward_count"] == 0
    assert early_observed["cv_positive_fraction_score_count"] == 0
    assert early_observed["cv_prompt_score_forward_count"] == 0
    assert early_observed["logical_cv_candidate_count"] == 0
    assert early_observed["teacher_capability_access_count"] == 36
    assert early_observed["provider_only_runtime_trace_count"] == 0
    assert early_observed["unique_empirical_fisher_gradient_row_count"] == 24
    assert early_observed["empirical_fisher_outer_product_evaluation_count"] == 24
    all_six_observed = all_six["observed_stage_counters"]
    assert all_six_observed["full_model_forward_count"] == 284
    assert all_six_observed["provider_only_runtime_trace_count"] == 120


def _manifest() -> dict[str, object]:
    folds = tuple(f"family-{index}" for index in range(6))
    slots: dict[str, dict[str, dict[str, object]]] = {}
    directions: dict[str, dict[str, str]] = {}
    rays: dict[str, dict[str, str]] = {}
    zero_by_law: dict[str, str] = {}
    for law in smoke._LAWS:
        slots[law] = {}
        directions[law] = {}
        rays[law] = {}
        zero_by_law[law] = _sha(f"{law}-zero-provider")
        for fold in folds:
            directions[law][fold] = _sha(f"{law}-{fold}-direction")
            rays[law][fold] = _sha(f"{law}-{fold}-ray")
            slots[law][fold] = {}
            for fraction in smoke._FRACTIONS:
                weights = smoke._INITIAL_WEIGHTS
                provider = (
                    zero_by_law[law]
                    if fraction == 0.0
                    else _sha(f"{law}-{fold}-{fraction}-provider")
                )
                slots[law][fold][smoke._fraction_key(fraction)] = {
                    "fraction": fraction,
                    "weights": weights,
                    "weight_tensor_sha256": smoke._provider_tensor_sha256(
                        smoke._weight_tensor(weights)
                    ),
                    "box_certificate": smoke._core.bilinear_box_certificate(weights),
                    "provider_artifact_sha256": provider,
                    "provider_transfer_evidence_sha256": (
                        _sha(f"{law}-zero-evidence")
                        if fraction == 0.0
                        else smoke._v14._sha256(
                            {
                                "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
                                "source_artifact_sha256": _sha("source"),
                                "response_law": law,
                                "validation_family_id": fold,
                                "direction_artifact_sha256": directions[law][fold],
                                "ray_artifact_sha256": rays[law][fold],
                                "fraction": fraction,
                                "weights": weights,
                                "radial_projection_used": False,
                                "held_rows_used": False,
                            },
                            domain=smoke._PROVIDER_SEED_DOMAIN,
                        )
                    ),
                    "direction_artifact_sha256": directions[law][fold],
                    "ray_artifact_sha256": rays[law][fold],
                    "radial_projection_used": False,
                }
    return smoke._hashed(
        {
            "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": smoke._core.TANGENT_RESPONSE_PROTOCOL_SHA256,
            "source_artifact_sha256": _sha("source"),
            "law_order": smoke._LAWS,
            "fold_order": folds,
            "fraction_order": smoke._FRACTIONS,
            "direction_artifact_sha256s_by_law_and_fold": directions,
            "ray_artifact_sha256s_by_law_and_fold": rays,
            "provider_slots_by_law_and_fold": slots,
            "logical_provider_slot_count": 120,
            "positive_provider_artifact_count": 108,
            "positive_provider_hashes_unique": True,
            "beta_zero_provider_artifact_sha256s_by_law": zero_by_law,
            "beta_zero_provider_reused_across_six_folds_per_law": True,
            "all_slots_frozen_before_first_cv_capability": True,
            "cv_capability_count_at_freeze": 0,
            "radial_projection_used": False,
            "held_data_or_objectives_used": False,
            "raw_tensors_or_provider_sidecars_serialized": False,
        },
        domain=smoke._MANIFEST_DOMAIN,
    )


def test_manifest_binds_120_slots_and_shared_beta_zero() -> None:
    receipt = smoke._validate_manifest(_manifest())
    assert receipt["logical_provider_slot_count"] == 120
    assert receipt["positive_provider_artifact_count"] == 108
    assert receipt["beta_zero_provider_reused_across_six_folds_per_law"] is True
    assert receipt["radial_projection_used"] is False


def test_manifest_replay_uses_the_live_torch_box_certificate_at_one_ulp() -> None:
    receipt = _manifest()
    law = "signed_log"
    family = "family-0"
    fraction = 0.5
    weights = (0.3355384267614559, 0.5, -0.16446157323854405)
    runtime_certificate = smoke.fisher_continuous_bilinear_box_max_abs(
        smoke._weight_tensor(weights)
    )
    assert runtime_certificate == 0.9999999999999999
    assert smoke._core.bilinear_box_certificate(weights) == 1.0
    row = receipt["provider_slots_by_law_and_fold"][law][family][
        smoke._fraction_key(fraction)
    ]
    row["weights"] = weights
    row["weight_tensor_sha256"] = smoke._provider_tensor_sha256(
        smoke._weight_tensor(weights)
    )
    row["box_certificate"] = runtime_certificate
    row["provider_transfer_evidence_sha256"] = smoke._v14._sha256(
        {
            "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
            "source_artifact_sha256": receipt["source_artifact_sha256"],
            "response_law": law,
            "validation_family_id": family,
            "direction_artifact_sha256": row["direction_artifact_sha256"],
            "ray_artifact_sha256": row["ray_artifact_sha256"],
            "fraction": fraction,
            "weights": weights,
            "radial_projection_used": False,
            "held_rows_used": False,
        },
        domain=smoke._PROVIDER_SEED_DOMAIN,
    )
    payload = {key: value for key, value in receipt.items() if key != "artifact_sha256"}
    receipt["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._MANIFEST_DOMAIN
    )
    validated = smoke._validate_manifest(receipt)
    assert validated["artifact_sha256"] == receipt["artifact_sha256"]

    smoke._validate_cv_candidate_manifest_weight_binding(
        {
            "weights": weights,
            "box_certificate": smoke._core.bilinear_box_certificate(weights),
        },
        {
            "weights": weights,
            "box_certificate": runtime_certificate,
        },
    )
    with pytest.raises(ValueError, match="weight binding differs"):
        smoke._validate_cv_candidate_manifest_weight_binding(
            {
                "weights": weights,
                "box_certificate": smoke._core.bilinear_box_certificate(weights),
            },
            {
                "weights": (weights[0], weights[1], weights[2] + 0.01),
                "box_certificate": runtime_certificate,
            },
        )


def test_manifest_rejects_positive_provider_reuse_even_after_rehash() -> None:
    receipt = _manifest()
    slots = receipt["provider_slots_by_law_and_fold"]
    first = slots["signed_log"]["family-0"][smoke._fraction_key(1 / 256)]
    second = slots["linear"]["family-5"][smoke._fraction_key(1.0)]
    second["provider_artifact_sha256"] = first["provider_artifact_sha256"]
    payload = {key: value for key, value in receipt.items() if key != "artifact_sha256"}
    receipt["artifact_sha256"] = smoke._v14._sha256(
        payload, domain=smoke._MANIFEST_DOMAIN
    )
    with pytest.raises(ValueError, match="provider uniqueness"):
        smoke._validate_manifest(receipt)


def test_degenerate_fold_is_controlled_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = tuple(
        sorted((*smoke._v20c._FROZEN_EXCLUDED, *(f"family-{i}" for i in range(6))))
    )
    fit_families = tuple(
        family for family in families if family not in smoke._v20c._FROZEN_EXCLUDED
    )
    banks = {
        law: SimpleNamespace(
            gradients_by_family={
                family: {f"{family}/0": (1.0, 0.0, 0.0)}
                for family in fit_families
            },
            gradient_bank_receipt={"law": law},
            evidence={"artifact_sha256": _sha(f"{law}-bank")},
        )
        for law in smoke._LAWS
    }
    counter = {"providers": 0}

    def direction(**kwargs: object) -> dict[str, object]:
        return {
            "artifact_sha256": _sha(
                f"{kwargs['gradient_bank_receipt']['law']}-"
                f"{kwargs['validation_family_id']}"
            ),
            "strict_descent_direction": False,
        }

    def ray(*, direction_receipt: object) -> dict[str, object]:
        return {
            "artifact_sha256": _sha(f"ray-{direction_receipt['artifact_sha256']}"),
            "direction_degenerate": True,
        }

    def forbidden_provider(*_args: object, **_kwargs: object) -> object:
        counter["providers"] += 1
        raise AssertionError("degenerate preflight must not construct CV providers")

    monkeypatch.setattr(
        smoke._core,
        "build_tangent_response_direction_from_gradient_bank_receipt",
        direction,
    )
    monkeypatch.setattr(smoke._core, "build_tangent_response_ray_receipt", ray)
    monkeypatch.setattr(smoke, "_build_response_provider", forbidden_provider)
    workspace = SimpleNamespace(
        base_provider=SimpleNamespace(artifact_sha256=_sha("base")),
        proposal_provider=SimpleNamespace(artifact_sha256=_sha("proposal")),
    )
    source = {
        "artifact_sha256": _sha("source"),
        "report_logical_sha256": _sha("logical"),
        "report_file_sha256": _sha("file"),
        "v20d_two_fit_bundle_artifact_sha256": _sha("bundle"),
        "v20b_pair_fragment_sha256": _sha("pair"),
    }
    with pytest.raises(smoke._ControlledTerminal) as caught:
        smoke._build_cv_provider_manifest(
            workspace,
            source=source,
            family_ids=families,
            banks=banks,
        )
    assert caught.value.stage == "cv_direction_preflight"
    assert caught.value.evidence["cv_capability_count"] == 0
    assert counter["providers"] == 0


def test_output_publication_is_0600_and_immutable(tmp_path: Path) -> None:
    output = tmp_path / "v20e.json"
    smoke._v20b._publish_scalar_fragment(
        {"schema": "fixture", "passed": False},
        path=output,
        domain=smoke._REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20e fixture",
    )
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        smoke._v20b._publish_scalar_fragment(
            {"schema": "replacement", "passed": True},
            path=output,
            domain=smoke._REPORT_DOMAIN,
            hash_key="report_sha256",
            label="V20e fixture",
        )


def _install_report_roundtrip_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    panel: dict[str, object],
    endpoint_fit: dict[str, object],
    cv: dict[str, dict[str, object]],
    manifest: dict[str, object],
    selection: dict[str, object],
    fit_bundle: dict[str, object] | None,
    provider_receipts: dict[str, dict[str, object]],
    provider_bundle: dict[str, object] | None,
    qualification: dict[str, object] | None,
) -> None:
    monkeypatch.setattr(
        smoke._v20b._core,
        "validate_nested_microstep_panel_receipt",
        lambda _value: panel,
    )
    monkeypatch.setattr(
        smoke._v20b._core,
        "validate_nested_microstep_fit_receipt",
        lambda _value, *, panel_receipt: endpoint_fit,
    )
    monkeypatch.setattr(
        smoke._v20b, "_validate_fit_training_evidence", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        smoke,
        "_validate_initial_evidence",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        smoke,
        "_validate_cv_stage",
        lambda *_a, **_k: (manifest, cv, selection),
    )
    monkeypatch.setattr(smoke, "_validate_cv_panel_lineage", lambda **_k: None)
    monkeypatch.setattr(
        smoke,
        "_validate_cv_direction_terminal",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        smoke,
        "_validate_all_six_terminal",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        smoke._core,
        "validate_tangent_response_fit_receipt",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        smoke,
        "_validate_all_six_bank_against_cv",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        smoke, "_validate_final_trace_evidence", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        smoke,
        "_initial_gain_hashes_from_fold_evidence",
        lambda _rows: {},
    )
    if fit_bundle is not None:
        monkeypatch.setattr(
            smoke._core,
            "validate_tangent_response_two_fit_bundle_receipt",
            lambda _value: fit_bundle,
        )
    monkeypatch.setattr(
        smoke,
        "_validate_provider_bundle",
        lambda *_a, **_k: (provider_receipts, provider_bundle),
    )
    monkeypatch.setattr(
        smoke._core,
        "validate_tangent_response_held_role_receipt",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        smoke, "_validate_role_evidence", lambda value, **_kwargs: dict(value)
    )
    if qualification is not None:
        monkeypatch.setattr(
            smoke, "_pair_qualification", lambda **_kwargs: qualification
        )


@pytest.mark.parametrize(
    "stage",
    ("cv_direction_terminal", "all_six_terminal", "fit_only", "full_held"),
)
def test_json_report_roundtrips_cover_every_terminal_shape(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / f"v20e-{stage}.json"
    fit_families = tuple(f"family-{index}" for index in range(6))
    panel_families = tuple(sorted((*smoke._v20c._FROZEN_EXCLUDED, *fit_families)))
    example_families = {
        f"{family}/example-{index}": family
        for family in fit_families
        for index in range(2)
    }
    sequences = {example: _sha(f"sequence-{example}") for example in example_families}
    source = smoke._hashed(
        {
            "fixture": stage,
            "report_logical_sha256": _sha("v20d-logical"),
            "report_file_sha256": _sha("v20d-file"),
            "v20d_two_fit_bundle_artifact_sha256": _sha("v20d-fit-bundle"),
            "v20b_pair_fragment_sha256": _sha("v20b-pair"),
            "authenticated_before_model_work": True,
        },
        domain=smoke._SOURCE_DOMAIN,
    )
    panel = {
        "family_prompt_sha256s": {
            family: _sha(f"panel-{family}") for family in panel_families
        }
    }
    endpoint_fit = {
        "artifact_sha256": _sha("endpoint-fit"),
        "base_provider_artifact_sha256": _sha("base-provider"),
        "proposal_provider_artifact_sha256": _sha("proposal-provider"),
    }
    fit_training_evidence = {"example_family_ids": example_families}
    coordinate_trace = {
        "artifact_sha256": _sha("coordinate-trace"),
        "sequence_rows": tuple(
            {
                "example_id": example,
                "family_id": example_families[example],
                "sequence_artifact_sha256": sequences[example],
            }
            for example in sorted(example_families)
        ),
    }
    source_pair = {"artifact_sha256": _sha("source-pair")}
    initial = {
        law: {
            "artifact_sha256": _sha(f"{law}-initial-bank"),
            "provider_artifact_sha256": _sha(f"{law}-initial-provider"),
            "training_family_ids": fit_families,
            "training_example_family_ids": example_families,
            "training_sequence_sha256s": sequences,
        }
        for law in smoke._LAWS
    }
    banks = {law: SimpleNamespace(evidence=initial[law]) for law in smoke._LAWS}
    manifest = {"artifact_sha256": _sha("manifest")}
    selection = {"artifact_sha256": _sha("selection")}
    cv = {
        law: {
            "fit_family_ids": fit_families,
            "artifact_sha256": _sha(f"{law}-cv"),
        }
        for law in smoke._LAWS
    }
    cv_live = {
        law: SimpleNamespace(
            receipt=cv[law], fold_evidence=({"law": law, "fold": 0},)
        )
        for law in smoke._LAWS
    }
    finals: dict[str, SimpleNamespace] = {}
    fit_bundle: dict[str, object] | None = None
    if stage in {"fit_only", "full_held"}:
        for law in smoke._LAWS:
            direction = {
                "source_artifact_sha256s": smoke._source_sha256s(source),
                "base_provider_artifact_sha256": endpoint_fit[
                    "base_provider_artifact_sha256"
                ],
                "proposal_provider_artifact_sha256": endpoint_fit[
                    "proposal_provider_artifact_sha256"
                ],
                "gradient_evidence_sha256": initial[law]["artifact_sha256"],
            }
            ray = {"artifact_sha256": _sha(f"{law}-final-ray")}
            trace = {"artifact_sha256": _sha(f"{law}-final-trace")}
            fit = {
                "artifact_sha256": _sha(f"{law}-fit"),
                "final_direction_receipt": direction,
                "final_ray_receipt": ray,
                "cv_receipt": cv[law],
            }
            finals[law] = SimpleNamespace(
                direction=direction, ray=ray, trace_evidence=trace, fit=fit
            )
        fit_bundle = {
            "artifact_sha256": _sha("fit-bundle"),
            "fit_receipts_by_law": {
                law: finals[law].fit for law in smoke._LAWS
            },
            "held_score_authorized": stage == "full_held",
        }
    provider_receipts = {
        arm: {"provider_artifact_sha256": _sha(f"{arm}-provider")}
        for arm in smoke._ARMS
    }
    provider_bundle = (
        {"artifact_sha256": _sha("provider-bundle")}
        if stage == "full_held"
        else None
    )
    roles = (
        [
            {
                "artifact_sha256": _sha(f"role-{index}"),
                "outer_held_family_id": outer,
            }
            for index, outer in enumerate(smoke._v20c._FROZEN_EXCLUDED)
        ]
        if stage == "full_held"
        else []
    )
    role_evidence = (
        [{"artifact_sha256": _sha(f"role-evidence-{index}")} for index in range(2)]
        if stage == "full_held"
        else []
    )
    qualification = (
        {"artifact_sha256": _sha("qualification"), "passed": False}
        if stage == "full_held"
        else None
    )
    authenticated_v20d = {
        "source_pair_diagnostic": source_pair,
        "panel_receipt": panel,
        "shared_fit_receipt": endpoint_fit,
        "fit_training_evidence": fit_training_evidence,
        "coordinate_trace_receipt": coordinate_trace,
        "role_execution_evidence": tuple(
            {
                "outer_held_family_id": role["outer_held_family_id"],
                "arm_execution_evidence": {
                    "base": {
                        "post_cast_h4_sha256s": {
                            f"held-{index}/0": _sha(f"held-{index}-h4")
                        }
                    }
                },
            }
            for index, role in enumerate(roles)
        ),
    }
    manifest_live = (
        None
        if stage == "cv_direction_terminal"
        else SimpleNamespace(receipt=manifest)
    )
    terminal = (
        SimpleNamespace(
            stage=(
                "cv_direction_preflight"
                if stage == "cv_direction_terminal"
                else "all_six_direction_preflight"
            ),
            evidence={"artifact_sha256": _sha(f"{stage}-terminal")},
        )
        if stage in {"cv_direction_terminal", "all_six_terminal"}
        else None
    )
    monkeypatch.setattr(
        smoke,
        "_fit_stage_authorized",
        lambda *_a, **_k: stage == "full_held",
    )
    report = smoke._build_report(
        output=output,
        source=source,
        v20d_report=authenticated_v20d,
        workspace=SimpleNamespace(
            fit_receipt=endpoint_fit,
            fit_training_evidence=fit_training_evidence,
        ),
        coordinate_trace=coordinate_trace,
        banks=banks,
        manifest=manifest_live,
        cv={} if stage == "cv_direction_terminal" else cv_live,
        selection_bundle=(
            None if stage == "cv_direction_terminal" else selection
        ),
        finals=finals,
        fit_bundle=fit_bundle,
        provider_receipts=(provider_receipts if stage == "full_held" else None),
        provider_bundle=provider_bundle,
        roles=roles,
        role_evidence=role_evidence,
        qualification=qualification,
        terminal=terminal,
    )
    report["report_sha256"] = smoke._v14._sha256(
        report, domain=smoke._REPORT_DOMAIN
    )
    serialized = json.loads(json.dumps(report, sort_keys=True))
    _install_report_roundtrip_stubs(
        monkeypatch,
        panel=panel,
        endpoint_fit=endpoint_fit,
        cv=cv,
        manifest=manifest,
        selection=selection,
        fit_bundle=fit_bundle,
        provider_receipts=provider_receipts,
        provider_bundle=provider_bundle,
        qualification=qualification,
    )
    validated = smoke._validate_report(
        serialized,
        output=output,
        authenticated_source=source,
        authenticated_v20d=authenticated_v20d,
    )
    assert validated["classification"] == report["classification"]

    output.write_text(json.dumps(serialized, sort_keys=True) + "\n")
    output.chmod(0o600)
    monkeypatch.setattr(
        smoke,
        "_load_authenticated_v20d_source",
        lambda: (source, authenticated_v20d, {}),
    )
    loaded = smoke._load_existing_report(output)
    assert loaded["classification"] == report["classification"]
    assert loaded["report_sha256"] == report["report_sha256"]


def _real_early_terminal_fixture(
    output: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    fit_families = tuple(f"family-{index}" for index in range(6))
    panel_families = tuple(
        sorted((*smoke._v20c._FROZEN_EXCLUDED, *fit_families))
    )
    example_families = {
        f"{family}/example-{index}": family
        for family in fit_families
        for index in range(2)
    }
    example_ids = tuple(sorted(example_families))
    sequences = {
        example: _sha(f"real-sequence-{example}") for example in example_ids
    }
    source = smoke._hashed(
        {
            "report_logical_sha256": _sha("real-v20d-logical"),
            "report_file_sha256": _sha("real-v20d-file"),
            "v20d_two_fit_bundle_artifact_sha256": _sha(
                "real-v20d-fit-bundle"
            ),
            "v20b_pair_fragment_sha256": _sha("real-v20b-pair"),
            "authenticated_before_model_work": True,
        },
        domain=smoke._SOURCE_DOMAIN,
    )
    endpoint_fit = {
        "artifact_sha256": _sha("real-endpoint-fit"),
        "base_provider_artifact_sha256": _sha("real-base-provider"),
        "proposal_provider_artifact_sha256": _sha("real-proposal-provider"),
    }
    panel = {
        "family_prompt_sha256s": {
            family: _sha(f"real-panel-{family}") for family in panel_families
        }
    }
    fit_training_evidence = {"example_family_ids": example_families}
    coordinate_trace = {
        "artifact_sha256": _sha("real-coordinate-trace"),
        "sequence_rows": tuple(
            {
                "example_id": example,
                "family_id": example_families[example],
                "sequence_artifact_sha256": sequences[example],
            }
            for example in example_ids
        ),
    }
    source_pair = {"artifact_sha256": _sha("real-source-pair")}
    directions: dict[str, dict[str, dict[str, object]]] = {}
    rays: dict[str, dict[str, dict[str, object]]] = {}
    initial: dict[str, dict[str, object]] = {}
    for law in smoke._LAWS:
        provider_sha = _sha(f"real-{law}-initial-provider")
        provider_seed = smoke._v14._sha256(
            {
                "runner_protocol_sha256": smoke._RUNNER_PROTOCOL_SHA256,
                "source_artifact_sha256": source["artifact_sha256"],
                "endpoint_fit_artifact_sha256": endpoint_fit[
                    "artifact_sha256"
                ],
                "response_law": law,
                "weights": smoke._INITIAL_WEIGHTS,
                "scope": "all_six_shared_beta_zero",
                "held_rows_used": False,
            },
            domain=smoke._PROVIDER_SEED_DOMAIN,
        )
        gradients = {
            family: {
                f"{family}/example-{index}": (0.0, 0.0, 0.0)
                for index in range(2)
            }
            for family in fit_families
        }
        bank = smoke._core.build_tangent_response_gradient_bank_receipt(
            source_artifact_sha256s=smoke._source_sha256s(source),
            family_ids=panel_families,
            excluded_family_ids=smoke._v20c._FROZEN_EXCLUDED,
            fit_gradients_by_family=gradients,
            base_provider_artifact_sha256=endpoint_fit[
                "base_provider_artifact_sha256"
            ],
            proposal_provider_artifact_sha256=endpoint_fit[
                "proposal_provider_artifact_sha256"
            ],
            response_law=law,
        )
        core_rows = {
            str(example): str(row_hash)
            for summary in bank["family_gradient_summaries_by_family"].values()
            for example, row_hash in summary[
                "example_gradient_sha256s"
            ].items()
        }
        objectives = {
            family: {
                f"{family}/example-{index}": 1.0 for index in range(2)
            }
            for family in fit_families
        }
        h4 = {
            example: _sha(f"real-{law}-{example}-h4")
            for example in example_ids
        }
        logits = {
            example: _sha(f"real-{law}-{example}-logits")
            for example in example_ids
        }
        executions = {
            example: smoke._initial_execution_sha256(
                law=law,
                provider_sha256=provider_sha,
                example_id=example,
                family_id=example_families[example],
                objective=1.0,
                h4_sha256=h4[example],
                logits_sha256=logits[example],
            )
            for example in example_ids
        }
        capability = {
            "artifact_sha256": _sha(f"real-{law}-capability"),
            "held_family_id": None,
            "authorized_example_count": len(example_ids),
            "authorized_family_count": 6,
            "access_count": len(example_ids),
            "per_example_access_counts": {
                example: 1 for example in example_ids
            },
            "held_family_capability_excluded": False,
            "teacher_rows_consumed_only_through_capability": True,
        }
        initial[law] = smoke._hashed(
            {
                "response_law": law,
                "source_artifact_sha256": source["artifact_sha256"],
                "endpoint_fit_artifact_sha256": endpoint_fit[
                    "artifact_sha256"
                ],
                "provider_artifact_sha256": provider_sha,
                "provider_transfer_evidence_sha256": provider_seed,
                "initial_weights": smoke._INITIAL_WEIGHTS,
                "initial_weight_tensor_sha256": smoke._provider_tensor_sha256(
                    smoke._weight_tensor(smoke._INITIAL_WEIGHTS)
                ),
                "training_family_ids": fit_families,
                "training_sequence_sha256s": sequences,
                "training_example_family_ids": example_families,
                "initial_objectives_by_family": objectives,
                "gradient_sha256s": {
                    example: _sha(f"real-{law}-{example}-gradient-tensor")
                    for example in example_ids
                },
                "core_gradient_row_sha256s": core_rows,
                "gradient_bank_receipt": bank,
                "gradient_bank_frozen_before_direction_solves": True,
                "direction_solve_count_at_gradient_bank_freeze": 0,
                "gradient_tensor_and_core_row_hashes_collected_from_same_live_tensor": True,
                "unique_empirical_fisher_gradient_row_count": len(example_ids),
                "empirical_fisher_outer_product_evaluation_count": len(
                    example_ids
                ),
                "post_cast_h4_sha256s": h4,
                "supervised_full_vocab_logits_sha256s": logits,
                "execution_receipt_sha256s": executions,
                "capability_receipt": capability,
                "full_suffix_vjp_count": len(example_ids),
                "local_response_autograd_contraction_count": len(example_ids),
                "beta_zero_exact_execution_reusable_by_all_six_folds": True,
                "all_initial_executions_finite": True,
                "all_initial_executions_exact": True,
                "held_family_ids": smoke._v20c._FROZEN_EXCLUDED,
                "held_data_or_objectives_used": False,
                "raw_gradients_h4_logits_targets_or_tensors_serialized": False,
            },
            domain=smoke._INITIAL_DOMAIN,
        )
        directions[law] = {}
        rays[law] = {}
        for family in fit_families:
            direction = (
                smoke._core.build_tangent_response_direction_from_gradient_bank_receipt(
                    gradient_bank_receipt=bank,
                    gradient_evidence_sha256=str(
                        initial[law]["artifact_sha256"]
                    ),
                    validation_family_id=family,
                )
            )
            directions[law][family] = direction
            rays[law][family] = smoke._core.build_tangent_response_ray_receipt(
                direction_receipt=direction
            )
    degenerate_rows = tuple(
        {
            "response_law": law,
            "validation_family_id": family,
            "direction_artifact_sha256": directions[law][family][
                "artifact_sha256"
            ],
            "ray_artifact_sha256": rays[law][family]["artifact_sha256"],
            "strict_descent_direction": directions[law][family][
                "strict_descent_direction"
            ],
            "direction_degenerate": rays[law][family]["direction_degenerate"],
        }
        for law in smoke._LAWS
        for family in fit_families
    )
    terminal_evidence = smoke._hashed(
        {
            "source_artifact_sha256": source["artifact_sha256"],
            "direction_receipts_by_law_and_fold": directions,
            "ray_receipts_by_law_and_fold": rays,
            "degenerate_rows": degenerate_rows,
            "provider_manifest_created": False,
            "cv_capability_count": 0,
            "held_capability_count": 0,
            "radial_projection_used": False,
            "classification": "tangent_direction_preflight_failed",
        },
        domain=smoke._MANIFEST_DOMAIN,
    )
    authenticated_v20d = {
        "source_pair_diagnostic": source_pair,
        "panel_receipt": panel,
        "shared_fit_receipt": endpoint_fit,
        "fit_training_evidence": fit_training_evidence,
        "coordinate_trace_receipt": coordinate_trace,
    }
    report = smoke._build_report(
        output=output,
        source=source,
        v20d_report=authenticated_v20d,
        workspace=SimpleNamespace(
            fit_receipt=endpoint_fit,
            fit_training_evidence=fit_training_evidence,
        ),
        coordinate_trace=coordinate_trace,
        banks={
            law: SimpleNamespace(evidence=initial[law]) for law in smoke._LAWS
        },
        manifest=None,
        cv={},
        selection_bundle=None,
        finals={},
        fit_bundle=None,
        provider_receipts=None,
        provider_bundle=None,
        roles=(),
        role_evidence=(),
        qualification=None,
        terminal=SimpleNamespace(
            stage="cv_direction_preflight", evidence=terminal_evidence
        ),
    )
    return report, source, authenticated_v20d


def _install_upstream_endpoint_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    authenticated_v20d: dict[str, object],
) -> None:
    monkeypatch.setattr(
        smoke._v20b._core,
        "validate_nested_microstep_panel_receipt",
        lambda _value: authenticated_v20d["panel_receipt"],
    )
    monkeypatch.setattr(
        smoke._v20b._core,
        "validate_nested_microstep_fit_receipt",
        lambda _value, *, panel_receipt: authenticated_v20d[
            "shared_fit_receipt"
        ],
    )
    monkeypatch.setattr(
        smoke._v20b, "_validate_fit_training_evidence", lambda *_a, **_k: None
    )


def _seal_report(report: dict[str, object]) -> dict[str, object]:
    sealed = copy.deepcopy(report)
    sealed["report_sha256"] = smoke._v14._sha256(
        sealed, domain=smoke._REPORT_DOMAIN
    )
    return json.loads(json.dumps(sealed, sort_keys=True))


def test_real_early_terminal_report_roundtrips_through_nested_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v20e-real-early-terminal.json"
    report, source, authenticated_v20d = _real_early_terminal_fixture(output)
    _install_upstream_endpoint_stubs(
        monkeypatch, authenticated_v20d=authenticated_v20d
    )
    serialized = _seal_report(report)
    validated = smoke._validate_report(
        serialized,
        output=output,
        authenticated_source=source,
        authenticated_v20d=authenticated_v20d,
    )
    assert validated["classification"] == "tangent_direction_preflight_failed"

    smoke._v20b._publish_scalar_fragment(
        report,
        path=output,
        domain=smoke._REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20e real early-terminal fixture",
    )
    monkeypatch.setattr(
        smoke,
        "_load_authenticated_v20d_source",
        lambda: (source, authenticated_v20d, {}),
    )
    loaded = smoke._load_existing_report(output)
    assert loaded["report_sha256"] == serialized["report_sha256"]
    assert output.stat().st_mode & 0o777 == 0o600


def test_real_early_terminal_rejects_rehashed_gradient_bank_stat_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v20e-real-bank-tamper.json"
    report, source, authenticated_v20d = _real_early_terminal_fixture(output)
    _install_upstream_endpoint_stubs(
        monkeypatch, authenticated_v20d=authenticated_v20d
    )
    forged = _seal_report(report)
    initial = forged["initial_vjp_evidence_by_law"]["signed_log"]
    bank = initial["gradient_bank_receipt"]
    summaries = copy.deepcopy(bank["family_gradient_summaries_by_family"])
    family = sorted(summaries)[0]
    summary = summaries[family]
    summaries[family] = smoke._core._build_family_gradient_summary_from_statistics(
        response_law=summary["response_law"],
        family_id=summary["family_id"],
        example_ids=summary["example_ids"],
        example_gradient_sha256s=summary["example_gradient_sha256s"],
        gradient_mean=(0.125, 0.0, 0.0),
        empirical_fisher=summary["empirical_fisher"],
    )
    initial["gradient_bank_receipt"] = (
        smoke._core._build_gradient_bank_from_summaries(
            source_artifact_sha256s=bank["source_artifact_sha256s"],
            family_ids=bank["family_ids"],
            excluded_family_ids=bank["excluded_family_ids"],
            base_provider_artifact_sha256=bank[
                "base_provider_artifact_sha256"
            ],
            proposal_provider_artifact_sha256=bank[
                "proposal_provider_artifact_sha256"
            ],
            response_law=bank["response_law"],
            family_gradient_summaries_by_family=summaries,
            held_objectives_or_gradients_used=False,
        )
    )
    initial_payload = {
        key: value for key, value in initial.items() if key != "artifact_sha256"
    }
    initial["artifact_sha256"] = smoke._v14._sha256(
        initial_payload, domain=smoke._INITIAL_DOMAIN
    )
    report_payload = {
        key: value for key, value in forged.items() if key != "report_sha256"
    }
    forged["report_sha256"] = smoke._v14._sha256(
        report_payload, domain=smoke._REPORT_DOMAIN
    )
    with pytest.raises(ValueError, match="frozen initial bank"):
        smoke._validate_report(
            forged,
            output=output,
            authenticated_source=source,
            authenticated_v20d=authenticated_v20d,
        )


def test_real_report_rejects_fully_rehashed_gradient_bank_family_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v20e-real-bank-family-swap.json"
    report, source, authenticated_v20d = _real_early_terminal_fixture(output)
    _install_upstream_endpoint_stubs(
        monkeypatch, authenticated_v20d=authenticated_v20d
    )
    forged = _seal_report(report)
    initial = forged["initial_vjp_evidence_by_law"]["signed_log"]
    bank = initial["gradient_bank_receipt"]
    summaries = copy.deepcopy(bank["family_gradient_summaries_by_family"])
    first_family, second_family = sorted(summaries)[:2]
    first = summaries[first_family]
    second = summaries[second_family]
    first_ids = tuple(first["example_ids"])
    second_ids = tuple(second["example_ids"])
    first_moved = first_ids[0]
    second_moved = second_ids[0]
    first_hashes = dict(first["example_gradient_sha256s"])
    second_hashes = dict(second["example_gradient_sha256s"])
    summaries[first_family] = (
        smoke._core._build_family_gradient_summary_from_statistics(
            response_law=first["response_law"],
            family_id=first_family,
            example_ids=(second_moved, first_ids[1]),
            example_gradient_sha256s={
                second_moved: second_hashes[second_moved],
                first_ids[1]: first_hashes[first_ids[1]],
            },
            gradient_mean=first["gradient_mean"],
            empirical_fisher=first["empirical_fisher"],
        )
    )
    summaries[second_family] = (
        smoke._core._build_family_gradient_summary_from_statistics(
            response_law=second["response_law"],
            family_id=second_family,
            example_ids=(first_moved, second_ids[1]),
            example_gradient_sha256s={
                first_moved: first_hashes[first_moved],
                second_ids[1]: second_hashes[second_ids[1]],
            },
            gradient_mean=second["gradient_mean"],
            empirical_fisher=second["empirical_fisher"],
        )
    )
    initial["gradient_bank_receipt"] = (
        smoke._core._build_gradient_bank_from_summaries(
            source_artifact_sha256s=bank["source_artifact_sha256s"],
            family_ids=bank["family_ids"],
            excluded_family_ids=bank["excluded_family_ids"],
            base_provider_artifact_sha256=bank[
                "base_provider_artifact_sha256"
            ],
            proposal_provider_artifact_sha256=bank[
                "proposal_provider_artifact_sha256"
            ],
            response_law=bank["response_law"],
            family_gradient_summaries_by_family=summaries,
            held_objectives_or_gradients_used=False,
        )
    )
    objectives = initial["initial_objectives_by_family"]
    first_objective = objectives[first_family].pop(first_moved)
    second_objective = objectives[second_family].pop(second_moved)
    objectives[first_family][second_moved] = second_objective
    objectives[second_family][first_moved] = first_objective
    initial_payload = {
        key: value for key, value in initial.items() if key != "artifact_sha256"
    }
    initial["artifact_sha256"] = smoke._v14._sha256(
        initial_payload, domain=smoke._INITIAL_DOMAIN
    )
    report_payload = {
        key: value for key, value in forged.items() if key != "report_sha256"
    }
    forged["report_sha256"] = smoke._v14._sha256(
        report_payload, domain=smoke._REPORT_DOMAIN
    )
    with pytest.raises(ValueError, match="objective family grouping"):
        smoke._validate_report(
            forged,
            output=output,
            authenticated_source=source,
            authenticated_v20d=authenticated_v20d,
        )
