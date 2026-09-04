from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from fisher_graph import gemma3_l3_l4_progressive_guard_ledger as ledger
from fisher_graph.fisher_local_chart_protocol import (
    FisherLocalChartSequenceSplitManifest,
    claim_and_open_fisher_local_chart_guard,
    freeze_fisher_local_chart_candidate,
    load_fisher_local_chart_frozen_candidate,
    sequence_split_manifest_from_corpus,
    validate_calibration_a_role,
    write_fisher_local_chart_guard_result,
)
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveAGuardClosedError,
    build_gemma3_l3_l4_progressive_a_corpus_artifact,
    load_gemma3_l3_l4_progressive_a_corpus,
    write_gemma3_l3_l4_progressive_a_corpus_artifact,
    write_gemma3_l3_l4_progressive_a_role_input,
)


def _sha(character: str) -> str:
    return character * 64


def _corpus(tmp_path: Path):
    roles = {
        "calibration_a_fit": ("fit text", "fit-family"),
        "calibration_a_selection": ("selection text", "selection-family"),
        "calibration_a_guard": ("guard text", "guard-family"),
    }
    paths = {}
    for role, (prompt, family) in roles.items():
        path = tmp_path / f"{role}.json"
        write_gemma3_l3_l4_progressive_a_role_input(
            path,
            corpus_id="local-chart-test",
            profile="pilot",
            role=role,
            prompts=(prompt,),
            family_ids=(family,),
        )
        paths[role] = path
    artifact = build_gemma3_l3_l4_progressive_a_corpus_artifact(
        corpus_id="local-chart-test",
        profile="pilot",
        tokenizer_contract={"tokenizer": "test", "max_length": 16},
        role_input_paths=paths,
    )
    artifact_path = tmp_path / "corpus.json"
    write_gemma3_l3_l4_progressive_a_corpus_artifact(
        artifact_path,
        artifact,
    )
    corpus = load_gemma3_l3_l4_progressive_a_corpus(
        artifact_path,
        role_input_paths=paths,
    )
    return corpus, paths


def _candidate_bundle(
    *,
    split: str,
    corpus: str = _sha("d"),
    guard: str = _sha("e"),
) -> dict[str, object]:
    schemes = ("contiguous", "interleaved")
    baselines_ids = ("B0", "B1", "B2", "B3", "B4")
    capacity = {
        "chart_count": 2,
        "coordinate_rank": 4,
        "affine_parameter_budget": 128,
    }
    baselines = {
        baseline: {
            "assignment_sha256": _sha("f"),
            "capacity_signature": dict(capacity),
            "schemes": {
                scheme: {
                    "coefficient": torch.tensor(
                        [[float(index), float(index + 1)]],
                        dtype=torch.float64,
                    )
                }
                for scheme in schemes
            },
        }
        for index, baseline in enumerate(baselines_ids)
    }
    return {
        "baselines": baselines,
        "selected_hyperparameters": {
            "selected_baseline": "B4",
            "selected_partition_scheme": "interleaved",
            "attention_affine_control_arm": "B3",
            "mlp_affine_control_arm": "B3",
            "fisher_regularization": 1.0e-6,
            "generator_ridge": 1.0e-4,
            "router_ridge": 1.0e-3,
        },
        "generators": {
            scheme: {
                baseline: {
                    "coefficient": torch.tensor(
                        [float(index + 1)], dtype=torch.float64
                    )
                }
                for index, baseline in enumerate(baselines_ids)
            }
            for scheme in schemes
        },
        "routers": {
            scheme: {
                baseline: {"coefficient": torch.tensor([float(index + 3)])}
                for index, baseline in enumerate(baselines_ids)
            }
            for scheme in schemes
        },
        "support": {
            scheme: {
                baseline: {
                    "coverage": 0.95,
                    "tangential_threshold": float(index + 1),
                }
                for index, baseline in enumerate(baselines_ids)
            }
            for scheme in schemes
        },
        "selection_transcript": [
            {"selected_baseline": "B4", "selection_score": 0.25}
        ],
        "bindings": {
            "protocol_sha256": _sha("a"),
            "split_manifest_sha256": split,
            "carrier_sha256": _sha("c"),
            "corpus_artifact_sha256": corpus,
            "guard_manifest_sha256": guard,
            "model_config_sha256": _sha("1"),
            "source_model_sha256": _sha("2"),
            "source_execution_sha256": _sha("3"),
        },
        "report_schema": {
            "schema": "fisher_graph.fisher_local_chart.report.v1",
            "format_version": 1,
        },
    }


def _freeze(
    tmp_path: Path,
    *,
    split: str,
    corpus: str = _sha("d"),
    guard: str = _sha("e"),
    bundle: dict[str, object] | None = None,
):
    return freeze_fisher_local_chart_candidate(
        bundle=(
            _candidate_bundle(split=split, corpus=corpus, guard=guard)
            if bundle is None
            else bundle
        ),
        protocol_sha256=_sha("a"),
        split_manifest_sha256=split,
        carrier_sha256=_sha("c"),
        path=tmp_path / "candidate.pt",
    )


def test_sequence_split_is_fixed_before_token_expansion(tmp_path: Path) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)

    sets = [set(manifest.sequence_ids[role]) for role in manifest.sequence_ids]
    assert all(
        not left.intersection(right)
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    )
    family_sets = [
        set(manifest.family_ids[role]) for role in manifest.family_ids
    ]
    assert all(
        not left.intersection(right)
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    )
    manifest.assert_row_provenance(
        "A_fit",
        [manifest.sequence_ids["A_fit"][0]] * 5,
    )
    with pytest.raises(ValueError, match="escaped"):
        manifest.assert_row_provenance(
            "A_fit",
            [manifest.sequence_ids["A_selection"][0]],
        )


def test_manual_split_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        FisherLocalChartSequenceSplitManifest(
            corpus_artifact_sha256=_sha("0"),
            tokenizer_contract_sha256=_sha("1"),
            role_manifest_sha256s={
                "A_fit": _sha("2"),
                "A_selection": _sha("3"),
                "A_guard": _sha("4"),
            },
            role_input_file_sha256s={
                "A_fit": _sha("5"),
                "A_selection": _sha("6"),
                "A_guard": _sha("7"),
            },
            sequence_ids={
                "A_fit": ("shared",),
                "A_selection": ("selection",),
                "A_guard": ("shared",),
            },
            family_ids={
                "A_fit": ("fit",),
                "A_selection": ("selection",),
                "A_guard": ("guard",),
            },
            forbidden_assessment_manifest_sha256s=(_sha("8"),),
        )


def test_manual_family_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="family disjoint"):
        FisherLocalChartSequenceSplitManifest(
            corpus_artifact_sha256=_sha("0"),
            tokenizer_contract_sha256=_sha("1"),
            role_manifest_sha256s={
                "A_fit": _sha("2"),
                "A_selection": _sha("3"),
                "A_guard": _sha("4"),
            },
            role_input_file_sha256s={
                "A_fit": _sha("5"),
                "A_selection": _sha("6"),
                "A_guard": _sha("7"),
            },
            sequence_ids={
                "A_fit": ("fit",),
                "A_selection": ("selection",),
                "A_guard": ("guard",),
            },
            family_ids={
                "A_fit": ("shared-family",),
                "A_selection": ("selection-family",),
                "A_guard": ("shared-family",),
            },
            forbidden_assessment_manifest_sha256s=(_sha("8"),),
        )


@pytest.mark.parametrize(
    "role", ("calibration_b", "B_guard", "B-anything", "test")
)
def test_calibration_b_and_non_a_roles_are_prohibited(role: str) -> None:
    expected = "Calibration B" if role != "test" else "role must"
    with pytest.raises(ValueError, match=expected):
        validate_calibration_a_role(role)


def test_candidate_is_content_addressed_and_strictly_reloaded(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path, split=_sha("b"))
    restored = load_fisher_local_chart_frozen_candidate(frozen.path)
    assert restored.candidate_sha256 == frozen.candidate_sha256
    assert torch.equal(
        restored.bundle["generators"]["interleaved"]["B4"][  # type: ignore[index]
            "coefficient"
        ],
        torch.tensor([5.0], dtype=torch.float64),
    )
    assert set(restored.bundle["baselines"]) == {  # type: ignore[arg-type]
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
    }
    manifest = json.loads(frozen.manifest_path.read_text())
    assert manifest["schema"] == "fisher_graph.fisher_local_chart.frozen_candidate.v2"
    assert manifest["format_version"] == 2
    assert manifest["bindings"] == restored.bundle["bindings"]
    with pytest.raises(FileExistsError, match="overwrite"):
        _freeze(tmp_path, split=_sha("b"))

    manifest["carrier_sha256"] = _sha("d")
    frozen.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        load_fisher_local_chart_frozen_candidate(frozen.path)


def test_candidate_manifest_authenticates_complete_model_lineage(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path, split=_sha("b"))
    manifest = json.loads(frozen.manifest_path.read_text())
    manifest["bindings"]["source_execution_sha256"] = _sha("9")
    frozen.manifest_path.write_bytes(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )

    with pytest.raises(ValueError, match="manifest"):
        load_fisher_local_chart_frozen_candidate(frozen.path)


def test_candidate_tensor_tampering_breaks_strict_reload(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path, split=_sha("b"))
    raw = torch.load(frozen.path, map_location="cpu", weights_only=True)
    raw["bundle"]["generators"]["interleaved"]["B4"]["coefficient"].add_(
        1.0
    )
    torch.save(raw, frozen.path)

    with pytest.raises(ValueError, match="logical|manifest"):
        load_fisher_local_chart_frozen_candidate(frozen.path)


def test_pre_guard_bundle_rejects_guard_and_b_payloads(tmp_path: Path) -> None:
    common = dict(
        protocol_sha256=_sha("a"),
        split_manifest_sha256=_sha("b"),
        carrier_sha256=_sha("c"),
        path=tmp_path / "candidate.pt",
    )
    guard_bundle = _candidate_bundle(split=_sha("b"))
    guard_bundle["selection_transcript"][0]["nested"] = {  # type: ignore[index]
        "guard_prompts": ["hidden"]
    }
    with pytest.raises(ValueError, match="guard"):
        freeze_fisher_local_chart_candidate(
            bundle=guard_bundle,
            **common,
        )
    b_bundle = _candidate_bundle(split=_sha("b"))
    b_bundle["selection_transcript"][0]["nested"] = {  # type: ignore[index]
        "role": "calibration_b_selection"
    }
    with pytest.raises(ValueError, match="Calibration B"):
        freeze_fisher_local_chart_candidate(
            bundle=b_bundle,
            **common,
        )


def test_candidate_requires_complete_equal_capacity_bundle(
    tmp_path: Path,
) -> None:
    incomplete = _candidate_bundle(split=_sha("b"))
    del incomplete["routers"]
    with pytest.raises(ValueError, match="exactly baselines"):
        freeze_fisher_local_chart_candidate(
            bundle=incomplete,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / "incomplete.pt",
        )

    unequal = _candidate_bundle(split=_sha("b"))
    unequal["baselines"]["B3"]["capacity_signature"][  # type: ignore[index]
        "coordinate_rank"
    ] = 5
    with pytest.raises(ValueError, match="equal capacity"):
        freeze_fisher_local_chart_candidate(
            bundle=unequal,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / "unequal.pt",
        )


@pytest.mark.parametrize("section", ("generators", "routers", "support"))
def test_candidate_requires_exact_partition_and_baseline_grids(
    tmp_path: Path,
    section: str,
) -> None:
    missing_scheme = _candidate_bundle(split=_sha("b"))
    del missing_scheme[section]["interleaved"]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly contiguous and interleaved"):
        freeze_fisher_local_chart_candidate(
            bundle=missing_scheme,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / f"{section}-missing-scheme.pt",
        )

    extra_scheme = _candidate_bundle(split=_sha("b"))
    extra_scheme[section]["hashed"] = copy.deepcopy(  # type: ignore[index]
        extra_scheme[section]["contiguous"]  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="exactly contiguous and interleaved"):
        freeze_fisher_local_chart_candidate(
            bundle=extra_scheme,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / f"{section}-extra-scheme.pt",
        )

    missing_baseline = _candidate_bundle(split=_sha("b"))
    del missing_baseline[section]["contiguous"]["B2"]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly B0-B4"):
        freeze_fisher_local_chart_candidate(
            bundle=missing_baseline,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / f"{section}-missing-baseline.pt",
        )

    extra_baseline = _candidate_bundle(split=_sha("b"))
    extra_baseline[section]["contiguous"]["B5"] = {  # type: ignore[index]
        "coefficient": torch.tensor([9.0])
    }
    with pytest.raises(ValueError, match="exactly B0-B4"):
        freeze_fisher_local_chart_candidate(
            bundle=extra_baseline,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / f"{section}-extra-baseline.pt",
        )


def test_candidate_requires_exact_baseline_partition_states(tmp_path: Path) -> None:
    bundle = _candidate_bundle(split=_sha("b"))
    del bundle["baselines"]["B3"]["schemes"]["contiguous"]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly contiguous and interleaved"):
        freeze_fisher_local_chart_candidate(
            bundle=bundle,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / "missing-baseline-partition.pt",
        )


@pytest.mark.parametrize(
    "binding",
    (
        "model_config_sha256",
        "source_model_sha256",
        "source_execution_sha256",
    ),
)
def test_candidate_requires_model_lineage_bindings(
    tmp_path: Path,
    binding: str,
) -> None:
    bundle = _candidate_bundle(split=_sha("b"))
    del bundle["bindings"][binding]  # type: ignore[index]
    with pytest.raises(ValueError, match="model config.*source model/execution hashes"):
        freeze_fisher_local_chart_candidate(
            bundle=bundle,
            protocol_sha256=_sha("a"),
            split_manifest_sha256=_sha("b"),
            carrier_sha256=_sha("c"),
            path=tmp_path / f"missing-{binding}.pt",
        )


def test_guard_cannot_claim_or_open_before_frozen_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    events = []

    def impossible_claim(**_kwargs):
        events.append("claim")
        raise AssertionError("claim should not run")

    with pytest.raises(
        Gemma3L3L4ProgressiveAGuardClosedError,
        match="requires",
    ):
        corpus.open_development_role("calibration_a_guard")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=tmp_path / "absent.pt",
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: events.append("dry-run"),
            claim_function=impossible_claim,
        )
    assert events == []
    assert corpus.guard_opened is False
    assert corpus.guard_consumed is False


def test_guard_preflight_binds_candidate_to_exact_corpus(
    tmp_path: Path,
) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    frozen = _freeze(
        tmp_path,
        split=manifest.manifest_sha256,
        corpus=_sha("d"),
        guard=manifest.role_manifest_sha256s["A_guard"],
    )
    events: list[str] = []

    with pytest.raises(ValueError, match="supplied A corpus"):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: events.append("dry-run"),
            claim_function=lambda **_kwargs: events.append("claim"),  # type: ignore[arg-type,return-value]
        )
    assert events == []
    assert corpus.guard_opened is False
    assert corpus.guard_consumed is False


def test_dry_run_may_not_mutate_frozen_candidate(
    tmp_path: Path,
) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    frozen = _freeze(
        tmp_path,
        split=manifest.manifest_sha256,
        corpus=corpus.artifact.artifact_sha256,
        guard=manifest.role_manifest_sha256s["A_guard"],
    )
    events: list[str] = []

    def mutate(candidate) -> None:
        events.append("dry-run")
        coefficient = candidate.bundle["generators"]["interleaved"]["B4"][  # type: ignore[index]
            "coefficient"
        ]
        coefficient.add_(1.0)

    with pytest.raises(ValueError, match="dry run mutated"):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=mutate,
            claim_function=lambda **_kwargs: events.append("claim"),  # type: ignore[arg-type,return-value]
        )
    assert events == ["dry-run"]
    assert corpus.guard_opened is False
    assert corpus.guard_consumed is False


def test_foreign_claim_receipt_never_opens_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    frozen = _freeze(
        tmp_path,
        split=manifest.manifest_sha256,
        corpus=corpus.artifact.artifact_sha256,
        guard=manifest.role_manifest_sha256s["A_guard"],
    )
    monkeypatch.setattr(ledger, "_FROZEN_LEDGER_ROOT", tmp_path / "ledger")

    def foreign_claim(**kwargs):
        return ledger.claim_gemma3_l3_l4_progressive_guard(
            protocol_sha256=kwargs["protocol_sha256"],
            guard_manifest_sha256=kwargs["guard_manifest_sha256"],
            challenger_receipt_sha256=_sha("d"),
        )

    with pytest.raises(ValueError, match="does not bind"):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: None,
            claim_function=foreign_claim,
        )
    assert corpus.guard_opened is False
    assert corpus.guard_consumed is False
    with pytest.raises(Exception, match="consumed|claimed"):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: None,
            claim_function=ledger.claim_gemma3_l3_l4_progressive_guard,
        )


def test_guard_order_is_reload_dry_run_claim_then_open_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    frozen = _freeze(
        tmp_path,
        split=manifest.manifest_sha256,
        corpus=corpus.artifact.artifact_sha256,
        guard=manifest.role_manifest_sha256s["A_guard"],
    )
    monkeypatch.setattr(ledger, "_FROZEN_LEDGER_ROOT", tmp_path / "ledger")
    events = []

    def claim(**kwargs):
        events.append("claim")
        return ledger.claim_gemma3_l3_l4_progressive_guard(**kwargs)

    _frozen, receipt, guard = claim_and_open_fisher_local_chart_guard(
        corpus=corpus,
        candidate_path=frozen.path,
        expected_protocol_sha256=_sha("a"),
        expected_split_manifest_sha256=manifest.manifest_sha256,
        expected_carrier_sha256=_sha("c"),
        dry_run=lambda _candidate: events.append("dry-run"),
        claim_function=claim,
    )
    events.append("open-returned")
    assert events == ["dry-run", "claim", "open-returned"]
    assert guard.role == "calibration_a_guard"
    assert corpus.guard_consumed is True
    assert receipt.guard_manifest_sha256 == (
        manifest.role_manifest_sha256s["A_guard"]
    )
    with pytest.raises(Exception, match="already|consumed"):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: None,
            claim_function=claim,
        )


def test_failure_after_claim_burns_guard(tmp_path: Path, monkeypatch) -> None:
    corpus, paths = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    frozen = _freeze(
        tmp_path,
        split=manifest.manifest_sha256,
        corpus=corpus.artifact.artifact_sha256,
        guard=manifest.role_manifest_sha256s["A_guard"],
    )
    monkeypatch.setattr(ledger, "_FROZEN_LEDGER_ROOT", tmp_path / "ledger")
    # Corrupt the still-unopened guard after the prompt-free artifact exists.
    paths["calibration_a_guard"].write_text("{corrupt", encoding="utf-8")

    with pytest.raises(Exception):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: None,
            claim_function=ledger.claim_gemma3_l3_l4_progressive_guard,
        )
    assert corpus.guard_consumed is True
    claim_path = (
        tmp_path
        / "ledger"
        / f"{manifest.role_manifest_sha256s['A_guard']}.claim.json"
    )
    assert claim_path.is_file()
    with pytest.raises(Exception, match="already|consumed"):
        claim_and_open_fisher_local_chart_guard(
            corpus=corpus,
            candidate_path=frozen.path,
            expected_protocol_sha256=_sha("a"),
            expected_split_manifest_sha256=manifest.manifest_sha256,
            expected_carrier_sha256=_sha("c"),
            dry_run=lambda _candidate: None,
            claim_function=ledger.claim_gemma3_l3_l4_progressive_guard,
        )


def test_guard_result_is_lineage_bound_b_free_and_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _ = _corpus(tmp_path)
    manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    frozen = _freeze(
        tmp_path,
        split=manifest.manifest_sha256,
        corpus=corpus.artifact.artifact_sha256,
        guard=manifest.role_manifest_sha256s["A_guard"],
    )
    monkeypatch.setattr(ledger, "_FROZEN_LEDGER_ROOT", tmp_path / "ledger")
    candidate, claim, _guard = claim_and_open_fisher_local_chart_guard(
        corpus=corpus,
        candidate_path=frozen.path,
        expected_protocol_sha256=_sha("a"),
        expected_split_manifest_sha256=manifest.manifest_sha256,
        expected_carrier_sha256=_sha("c"),
        dry_run=lambda _candidate: None,
        claim_function=ledger.claim_gemma3_l3_l4_progressive_guard,
    )
    result_path = tmp_path / "result.json"
    metrics = {
        "guard_accuracy": 0.75,
        "verdicts": {"fisher_tangent_capture": "pass"},
    }
    result = write_fisher_local_chart_guard_result(
        path=result_path,
        candidate=candidate,
        claim=claim,
        metrics=metrics,
    )
    assert result["evaluation_count"] == 1
    assert result["corpus_artifact_sha256"] == corpus.artifact.artifact_sha256
    assert json.loads(result_path.read_text(encoding="ascii")) == result
    with pytest.raises(FileExistsError):
        write_fisher_local_chart_guard_result(
            path=result_path,
            candidate=candidate,
            claim=claim,
            metrics=metrics,
        )

    with pytest.raises(ValueError, match="Calibration B"):
        write_fisher_local_chart_guard_result(
            path=tmp_path / "b-result.json",
            candidate=candidate,
            claim=claim,
            metrics={"source_role": "calibration_b_validation"},
        )
    assert not (tmp_path / "b-result.json").exists()

    other_bundle = copy.deepcopy(
        _candidate_bundle(
            split=manifest.manifest_sha256,
            corpus=corpus.artifact.artifact_sha256,
            guard=manifest.role_manifest_sha256s["A_guard"],
        )
    )
    other_bundle["generators"]["interleaved"]["B4"][  # type: ignore[index]
        "coefficient"
    ].add_(1.0)
    other_candidate = _freeze(
        tmp_path / "other",
        split=manifest.manifest_sha256,
        corpus=corpus.artifact.artifact_sha256,
        guard=manifest.role_manifest_sha256s["A_guard"],
        bundle=other_bundle,
    )
    with pytest.raises(ValueError, match="does not bind"):
        write_fisher_local_chart_guard_result(
            path=tmp_path / "foreign-result.json",
            candidate=other_candidate,
            claim=claim,
            metrics=metrics,
        )
