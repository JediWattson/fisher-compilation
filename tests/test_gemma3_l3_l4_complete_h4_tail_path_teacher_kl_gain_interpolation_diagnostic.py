from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_path_teacher_kl_gain_interpolation_diagnostic as diagnostic,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic as path_v1,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as endpoint,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)


def _prefix() -> Gemma3L3L4OnePassPrefix:
    return Gemma3L3L4OnePassPrefix(
        source_modes=torch.zeros((1, 3, 2), dtype=torch.float64),
        clamped_y3=torch.zeros((1, 3, 640), dtype=torch.float32),
        predicted_target_modal_delta=torch.zeros((1, 3, 2), dtype=torch.float64),
        decoded_base_x4_delta=torch.zeros((1, 3, 640), dtype=torch.float64),
        logical_positions=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        valid_target_mask=torch.tensor([[True, True, True]]),
        source_eligible_mask=torch.tensor([[False, True, False]]),
        target_affected_mask=torch.tensor([[False, True, False]]),
        bridge_binding_sha256="c" * 64,
    )


def test_protocol_constants_are_fixed_and_cli_has_no_beta_selection_override() -> None:
    assert diagnostic.GAIN_INTERPOLATION_BETAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert diagnostic.PATH_V1_REPORT_FILE_SHA256 == (
        "1031cf3c11354e7e59a4bb6adf616eea9ea92f39e4b3619f015bcf4bb7bd91a2"
    )
    assert diagnostic.PATH_V1_REPORT_SHA256 == (
        "4fbb5b8106f4048753d484eb319db813236cb5b31289db864359ab6a48de1dc2"
    )
    args = diagnostic.build_parser().parse_args([])
    assert args.path_v1_report == diagnostic.DEFAULT_PATH_V1_REPORT
    assert args.path_v1_report_file_sha256 == diagnostic.PATH_V1_REPORT_FILE_SHA256
    assert args.path_v1_report_sha256 == diagnostic.PATH_V1_REPORT_SHA256
    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert not any(
        action.dest in {"betas", "beta", "rank", "quadrature_order"}
        for action in diagnostic.build_parser()._actions
    )


def test_gain_interpolation_moves_every_gain_to_one_on_the_fixed_grid() -> None:
    fit = SimpleNamespace(
        rank=64,
        gains=tuple(torch.linspace(0.0, 2.0, 64, dtype=torch.float64).tolist()),
        validate_integrity=lambda: None,
    )
    original = torch.tensor(fit.gains, dtype=torch.float64)
    assert torch.equal(
        diagnostic._interpolated_gains(fit, beta=0.0), original
    )
    assert torch.equal(
        diagnostic._interpolated_gains(fit, beta=1.0), torch.ones(64, dtype=torch.float64)
    )
    assert torch.allclose(
        diagnostic._interpolated_gains(fit, beta=0.25),
        original + 0.25 * (1.0 - original),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="predeclared"):
        diagnostic._interpolated_gains(fit, beta=0.1)
    short = SimpleNamespace(rank=63, gains=(1.0,) * 63, validate_integrity=lambda: None)
    with pytest.raises(ValueError, match="exact K64"):
        diagnostic._interpolated_gains(short, beta=0.0)


def test_gain_provider_binds_beta_gain_hash_and_exact_sentinel() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base, dtype=torch.float64)
    correction[support] = 0.5
    gains = torch.linspace(0.25, 1.25, 64, dtype=torch.float64)
    provider = diagnostic._AuthenticatedGainInterpolationFiniteProvider(
        rank=64,
        beta=0.25,
        interpolated_gains=gains,
        fold_artifact_sha256="f" * 64,
        model_inputs_sha256="a" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        support_mask=support,
        correction=correction,
    )
    metadata = provider.metadata()
    assert metadata["variant_id"] == "beta_0_25"
    assert metadata["beta_index"] == 1
    assert metadata["beta_hex"] == (0.25).hex()
    assert metadata["interpolated_gains_sha256"] == diagnostic._runtime_tensor_sha256(
        gains
    )
    assert torch.equal(provider.correction(prefix, base), correction)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)

    sentinel = diagnostic._AuthenticatedGainInterpolationFiniteProvider(
        rank=320,
        beta=None,
        interpolated_gains=None,
        fold_artifact_sha256="f" * 64,
        model_inputs_sha256="a" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        support_mask=support,
        correction=correction,
    )
    assert sentinel.metadata()["variant_id"] == "shared_exact_sentinel"
    assert sentinel.metadata()["interpolated_gains_sha256"] is None


def _observation(example: int, variant_id: str) -> dict[str, object]:
    if variant_id == "shared_exact_sentinel":
        rank = 320
        beta_index = None
        beta_hex = None
        method = "shared_exact_sentinel"
        gain_formula = None
        gains_sha256 = None
    else:
        beta_index = diagnostic._BETA_IDS.index(variant_id)
        rank = 64
        beta_hex = diagnostic.GAIN_INTERPOLATION_BETAS[beta_index].hex()
        method = "gain_interpolated_path_signed_K64"
        gain_formula = "g_plus_beta_times_one_minus_g"
        gains_sha256 = "a" * 64
    row: dict[str, object] = {
        "example_id": f"e{example}",
        "family_id": f"f{example // 2}",
        "method": method,
        "variant_id": variant_id,
        "rank": rank,
        "requested_rank": rank,
        "effective_direction_count": rank,
        "beta_index": beta_index,
        "beta_hex": beta_hex,
        "gain_formula": gain_formula,
        "interpolated_gains_sha256": gains_sha256,
    }
    row["observation_sha256"] = token_v1._domain_sha256(
        row, domain=diagnostic._OBSERVATION_DOMAIN
    )
    return row


def test_observation_receipt_authenticates_all_96_predeclared_cells() -> None:
    rows = [
        _observation(example, variant)
        for example in range(16)
        for variant in diagnostic._BETA_IDS + ("shared_exact_sentinel",)
    ]
    receipt = diagnostic._finite_observation_set_sha256(rows)
    assert len(receipt) == 64
    tampered = deepcopy(rows)
    tampered[0]["beta_hex"] = (0.25).hex()
    with pytest.raises(ValueError, match="protocol differs"):
        diagnostic._finite_observation_set_sha256(tampered)
    missing = rows[:-1]
    with pytest.raises(ValueError, match="count differs"):
        diagnostic._finite_observation_set_sha256(missing)


def test_exact_replay_fails_closed_before_finite_when_a_fit_receipt_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = ({"prompt": "receipt"},)
    monkeypatch.setattr(path_v1, "_prompt_report_receipts", lambda *args: prompts)
    closure = SimpleNamespace(metadata=lambda: {"artifact_sha256": "a" * 64})
    fit = SimpleNamespace(
        rank=64,
        artifact_sha256="b" * 64,
        metadata=lambda: {"artifact_sha256": "b" * 64},
    )
    parent = {
        "prompt_path_receipts": prompts,
        "path_signed_joint_folds": ({"artifact_sha256": "c" * 64},),
        "FTC_closure": closure.metadata(),
        "FTC_closure_gate_results": (("gate", True),),
        "FTC_closure_diagnostics": {"diagnostic": 0.0},
        "resources": {"count": 1},
    }
    evidence = (SimpleNamespace(artifact_sha256="d" * 64),)
    with pytest.raises(RuntimeError, match="signed_joint_K64_folds_exact"):
        diagnostic._exact_path_v1_replay_receipt(
            parent_report=parent,
            traces=(object(),),
            evidence=evidence,
            closure=closure,
            closure_gates={"gate": True},
            closure_diagnostics={"diagnostic": 0.0},
            collection_resources={"count": 1},
            signed_fits={"held": fit},
        )


def test_resource_ledger_closes_at_112_collection_112_finite_224_total() -> None:
    collection = {
        "base_forward_count": 16,
        "native_teacher_forward_count": 16,
        "d320_boundary_forward_count": 16,
        "path_teacher_kl_vjp_forward_count": 64,
        "path_teacher_kl_vjp_backward_call_count": 436,
    }
    finite = {
        "finite_native_forward_count": 16,
        "finite_candidate_forward_count": 96,
        "finite_gain_interpolation_forward_count": 80,
        "finite_shared_exact_sentinel_forward_count": 16,
    }
    traces = tuple(
        SimpleNamespace(
            endpoint=SimpleNamespace(
                residual_rows=torch.zeros((1, 640)), supervised_tokens=1
            )
        )
        for _ in range(16)
    )
    result = diagnostic._resource_accounting(
        collection=collection,
        finite=finite,
        traces=traces,
        path_v1_resources={
            "signed_joint_streamed_coordinate_transform_logical_macs": 1,
            "signed_joint_streamed_operator_and_direction_score_logical_macs": 2,
            "signed_joint_low_rank_U_factor_deflation_logical_macs": 3,
            "signed_joint_symmetric_eigh_320_by_320_call_count": 512,
        },
    )
    assert result["collection_model_forward_count"] == 112
    assert result["finite_evaluation_model_forward_count"] == 112
    assert result["total_model_forward_count"] == 224
    assert result["PCA_control_fit_or_forward_count"] == 0
    assert result["signed_joint_K64_fit_replay_count"] == 8


def test_real_path_v1_parent_pins_and_observation_set_authenticate() -> None:
    if not diagnostic.DEFAULT_PATH_V1_REPORT.exists():
        pytest.skip("local path-v1 report is intentionally not committed")
    report = diagnostic._load_path_v1_report(
        diagnostic.DEFAULT_PATH_V1_REPORT,
        expected_file_sha256=diagnostic.PATH_V1_REPORT_FILE_SHA256,
        expected_report_sha256=diagnostic.PATH_V1_REPORT_SHA256,
    )
    assert report["schema"] == path_v1._SCHEMA
    assert report["report_sha256"] == diagnostic.PATH_V1_REPORT_SHA256


def test_beta_zero_comparison_uses_exact_parent_geometry_semantics() -> None:
    if not diagnostic.DEFAULT_PATH_V1_REPORT.exists():
        pytest.skip("local path-v1 report is intentionally not committed")
    parent = diagnostic._load_path_v1_report(
        diagnostic.DEFAULT_PATH_V1_REPORT,
        expected_file_sha256=diagnostic.PATH_V1_REPORT_FILE_SHA256,
        expected_report_sha256=diagnostic.PATH_V1_REPORT_SHA256,
    )
    observations = []
    for raw in parent["finite_observation_receipts"]:
        if raw["method"] == "signed_joint" and raw["rank"] == 64:
            row = dict(raw)
            row["variant_id"] = "beta_0"
            observations.append(row)
    parent_behavior = parent[
        "established_behavioral_fidelity_by_method_rank"
    ]["signed_joint"]["64"]
    parent_geometry = parent[
        "executed_cast_once_geometry_by_method_rank"
    ]["signed_joint"]["64"]
    assert parent_geometry["semantics"]["candidate"] == (
        "actual_cast_once_d320_plus_signed_joint_teacher_kl_tail_k64"
    )
    comparison = diagnostic._stable_parent_beta_zero_comparison(
        parent_report=parent,
        observations=observations,
        ladders={"beta_0": parent["finite_ladder_by_method"]["signed_joint"]},
        behavioral={"beta_0": parent_behavior},
        geometry={"beta_0": parent_geometry},
    )
    assert comparison["all_checks_passed"] is True


def test_path_v1_loader_rejects_semantic_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not diagnostic.DEFAULT_PATH_V1_REPORT.exists():
        pytest.skip("local path-v1 report is intentionally not committed")
    real = diagnostic._load_path_v1_report(
        diagnostic.DEFAULT_PATH_V1_REPORT,
        expected_file_sha256=diagnostic.PATH_V1_REPORT_FILE_SHA256,
        expected_report_sha256=diagnostic.PATH_V1_REPORT_SHA256,
    )
    tampered = deepcopy(real)
    tampered["protocol"]["selection_performed"] = True
    tampered["protocol"]["split"] = "held_family_selected"
    monkeypatch.setattr(token_v1, "_load_pinned_report", lambda *args, **kwargs: tampered)
    with pytest.raises(ValueError, match="semantics differ"):
        diagnostic._load_path_v1_report(
            "parent.json",
            expected_file_sha256="a" * 64,
            expected_report_sha256="b" * 64,
        )
