from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_jvp_diagnostic as diagnostic,
)


def _sha(value: object) -> str:
    return hashlib.sha256(repr(value).encode("ascii")).hexdigest()


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _mock_v10_report() -> dict[str, object]:
    endpoints: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    evidence_ids: list[str] = []
    evidence_hashes: list[str] = []
    families: list[str] = []
    folds: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    for family_index in range(8):
        family = f"family-{family_index}"
        families.append(family)
        folds.append({"family_id": family, "fold": family_index})
    for index in range(16):
        example = f"example-{index:02d}"
        family = families[index // 2]
        evidence = _sha(("evidence", example))
        evidence_ids.append(example)
        evidence_hashes.append(evidence)
        prompts.append({"example_id": example, "family_id": family})
        node_hashes: list[str] = []
        node_bindings: list[dict[str, object]] = []
        for node_index, (alpha, weight) in enumerate(
            zip(
                diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES,
                diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS,
                strict=True,
            )
        ):
            binding = {
                "node_index": node_index,
                "path_fraction_hex": alpha.hex(),
                "quadrature_weight_hex": weight.hex(),
                "captured_equals_independent_bitwise": True,
                "core_node_receipt_sha256": _sha(("core", example, node_index)),
                "token_teacher_kl_f64_sha256": _sha(("kl64", example, node_index)),
                "vjp_artifact_sha256": _sha(("vjp", example, node_index)),
            }
            node_bindings.append(binding)
            node: dict[str, object] = {
                "example_id": example,
                "family_id": family,
                "node_index": node_index,
                "path_fraction_hex": alpha.hex(),
                "quadrature_weight_hex": weight.hex(),
                "objective_dtype": "torch.float64",
                "captured_equals_independent_bitwise": True,
                "legacy_V9_node_replayed_exactly": True,
                "core_node_receipt_sha256": binding["core_node_receipt_sha256"],
                "token_teacher_kl_f64_sha256": binding[
                    "token_teacher_kl_f64_sha256"
                ],
                "vjp_artifact_sha256": binding["vjp_artifact_sha256"],
            }
            node_receipt = diagnostic.v10diag.token_v1._domain_sha256(
                node, domain=diagnostic.v10diag._NODE_RECEIPT_DOMAIN
            )
            node["receipt_sha256"] = node_receipt
            nodes.append(node)
            node_hashes.append(node_receipt)
        f32: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "objective_dtype": "torch.float32",
        }
        f32["artifact_sha256"] = diagnostic.v10diag.token_v1._domain_sha256(
            f32, domain=diagnostic.v10diag._F32_OBJECTIVE_DOMAIN
        )
        f64: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "objective_dtype": "torch.float64",
            "captured_objectives_equal_independent_bitwise": True,
            "path_f64_node_bindings": node_bindings,
        }
        f64["artifact_sha256"] = diagnostic.v10diag.token_v1._domain_sha256(
            f64, domain=diagnostic.v10diag._F64_OBJECTIVE_DOMAIN
        )
        base: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "legacy_V9_endpoint_receipts_replayed_exactly": True,
        }
        endpoint = {
            **base,
            "artifact_sha256": diagnostic.v10diag.token_v1._domain_sha256(
                base, domain=diagnostic.v10diag._ENDPOINT_REPLAY_DOMAIN
            ),
            "precision_evidence_artifact_sha256": evidence,
            "f32_objective_binding": f32,
            "f64_objective_binding": f64,
            "direct_f64_delta_mean": index * 1.0e-7,
            "legacy_f32_delta_sha256": _sha(("d32", example)),
            "legacy_f32_delta_mean": index * 2.0e-7,
            "pinned_V9_f32_delta_mean": index * 2.0e-7,
            "legacy_V9_D32_hash_and_scalar_replayed": True,
            "GL4_node_receipt_sha256s": node_hashes,
        }
        endpoints.append(endpoint)
    ordered = tuple(
        str(value["receipt_sha256"])
        for value in sorted(
            nodes,
            key=lambda value: (str(value["example_id"]), int(value["node_index"])),
        )
    )
    comparison = {
        "artifact_sha256": _sha("comparison"),
        "evidence_example_ids": evidence_ids,
        "evidence_artifact_sha256s": evidence_hashes,
        "closure_relative_rmse": 0.08673388652083237,
        "closure_rmse": 1.3509587725610062e-6,
        "closure_cosine": 0.99626247056314,
        "transport_relative_rmse_to_finite_f64": 0.0006221383476886703,
        "transport_rmse": 9.690367770556959e-9,
        "finite_precision_relative_rmse_to_finite_f64": 0.02003952674974365,
        "finite_precision_rmse": 3.121337639359058e-7,
        "finite_delta_f64_rms": 1.5575904951941974e-5,
    }
    return {
        "schema": diagnostic.v10diag._SCHEMA,
        "classification": (
            "small_path_transport_live_dtype_or_finite_rounding_supported_same_a"
        ),
        "passed": False,
        "resources": {
            "total_model_forward_count": 224,
            "total_backward_call_count": 1039,
            "total_candidate_support_row_executions": 7756,
            "phase_order": ["v10"],
        },
        "endpoint_precision_receipts": endpoints,
        "path_precision_node_receipts": nodes,
        "path_precision_node_receipt_set_sha256": (
            diagnostic.v10diag.token_v1._domain_sha256(
                ordered, domain=diagnostic.v10diag._NODE_SET_DOMAIN
            )
        ),
        "family_equal_objective_precision_comparison": comparison,
        "folds": folds,
        "prompt_receipts": prompts,
    }


def test_v10_loader_uses_exact_file_and_logical_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    report = _mock_v10_report()

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(diagnostic.v10diag.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v10_report("v10.json") is report
    assert captured["expected_file_sha256"] == diagnostic.V10_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V10_REPORT_SHA256

    invalid = {**report, "classification": "unexpected"}
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: invalid,
    )
    with pytest.raises(RuntimeError, match="V10 anchor differs"):
        diagnostic._load_v10_report("v10.json")


def test_v10_index_authenticates_exact_16_by_4_receipt_grid() -> None:
    report = _mock_v10_report()
    pinned = diagnostic._index_v10_receipts(report)
    assert len(pinned.endpoints) == 16
    assert len(pinned.nodes) == 64
    assert len(pinned.evidence_artifacts) == 16

    changed_node = _mock_v10_report()
    changed_node["path_precision_node_receipts"][0][  # type: ignore[index]
        "token_teacher_kl_f64_sha256"
    ] = _sha("changed")
    with pytest.raises(RuntimeError, match="path node receipt drifted"):
        diagnostic._index_v10_receipts(changed_node)

    changed_chain = _mock_v10_report()
    changed_chain["endpoint_precision_receipts"][0][  # type: ignore[index]
        "GL4_node_receipt_sha256s"
    ][0] = _sha("changed-chain")
    with pytest.raises(RuntimeError, match="endpoint node chain"):
        diagnostic._index_v10_receipts(changed_chain)

    changed_binding = _mock_v10_report()
    changed_binding["endpoint_precision_receipts"][0][  # type: ignore[index]
        "f64_objective_binding"
    ]["path_f64_node_bindings"][0]["node_index"] = 3
    with pytest.raises(RuntimeError, match="objective binding drifted"):
        diagnostic._index_v10_receipts(changed_binding)


def test_full_path_point_uses_float64_cast_once_geometry() -> None:
    scalar = torch.tensor(
        [[[1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))]]],
        dtype=torch.float32,
    ).contiguous()
    joint = torch.tensor([[[2.0, -3.0]]], dtype=torch.float32).contiguous()
    alpha = diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[1]
    point, direction = diagnostic._full_path_point_f64(scalar, joint, alpha)
    expected_direction = joint.double() - scalar.double()
    expected = scalar.double() + alpha * expected_direction
    assert point.dtype == torch.float64
    assert direction.dtype == torch.float64
    assert torch.equal(point, expected)
    assert torch.equal(direction, expected_direction)
    assert torch.equal(point.float(), expected.float())

    with pytest.raises(ValueError, match="geometry differs"):
        diagnostic._full_path_point_f64(scalar, joint, -0.1)


def test_discrete_receipt_separates_true_cast_collisions_from_static_coordinates() -> None:
    scalar = torch.tensor([[[1.0, 0.0, -2.0, 3.0]]], dtype=torch.float32)
    joint = scalar.clone()
    joint[0, 0, 0] = torch.nextafter(
        joint[0, 0, 0], torch.tensor(float("inf"), dtype=torch.float32)
    )
    joint[0, 0, 1] = 2.0
    fractions = (
        0.0,
        *diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES,
        1.0,
    )
    ideal = tuple(
        diagnostic._full_path_point_f64(scalar, joint, fraction)[0]
        for fraction in fractions
    )
    live = tuple(value.float().contiguous() for value in ideal)
    scalar_kl = torch.tensor([0.25, 0.5], dtype=torch.float64)
    joint_kl = torch.tensor([0.5, 0.125], dtype=torch.float64)
    kl_points = tuple(
        (scalar_kl + fraction * (joint_kl - scalar_kl)).contiguous()
        for fraction in fractions
    )
    state = diagnostic._ObservedPrompt(
        example_id="example",
        family_id="family",
        scalar_h4=live[0],
        scalar_token_kl_f64=kl_points[0],
        joint_h4=live[-1],
        joint_token_kl_f64=kl_points[-1],
        direction_h4_f64=joint.double() - scalar.double(),
        runtime=None,  # type: ignore[arg-type]
        live_node_h4={index: live[index + 1] for index in range(4)},
        node_token_kl_f64={index: kl_points[index + 1] for index in range(4)},
    )
    receipt = diagnostic._SuffixJVPCollector(
        context=object()
    )._discrete_cast_receipt(state)
    assert receipt["interval_count"] == 5
    assert receipt["cast_collision_excludes_static_coordinates"] is True
    assert receipt["cast_collision_coordinate_interval_count"] > 0
    assert receipt["static_coordinate_interval_count"] > 0
    assert receipt["preserved_change_coordinate_interval_count"] > 0
    assert (
        receipt["preserved_change_coordinate_interval_count"]
        + receipt["cast_collision_coordinate_interval_count"]
        == receipt["ideal_changed_coordinate_interval_count"]
    )
    assert not _contains_tensor(receipt)
    for interval in receipt["adjacent_intervals"]:
        assert (
            interval["preserved_change_coordinate_count"]
            + interval["cast_collision_coordinate_count"]
            == interval["ideal_changed_coordinate_count"]
        )
        assert (
            interval["cast_collision_coordinate_count"]
            + interval["static_coordinate_count"]
            == interval["unchanged_live_coordinate_count"]
        )


def test_node_bridge_uses_wrapped_core_hashes_not_runtime_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = SimpleNamespace(
        artifact_sha256=_sha("core-artifact"),
        token_teacher_kl_sha256=_sha("core-token-kl"),
        provider_artifact_sha256=_sha("core-provider"),
        execution_artifact_sha256=_sha("core-execution"),
        path_node_h4_sha256=_sha("core-path-h4"),
    )
    path = SimpleNamespace(
        node_receipts=(core,),
        supervised_grid_sha256=_sha("core-grid"),
        endpoint_pair_binding_sha256=_sha("core-endpoint"),
    )
    precision = SimpleNamespace(path_evidence=path)
    observed = diagnostic._ObservedNode(
        node_index=0,
        path_fraction=diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0],
        quadrature_weight=diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[0],
        directional_token_teacher_kl_f64=torch.tensor(
            [1.0], dtype=torch.float64
        ),
        core_node_receipt=core,
        suffix_runtime_receipt_sha256=_sha("runtime-receipt"),
    )
    captured: dict[str, object] = {}

    def construct(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        diagnostic, "CandidateJointStateSuffixJVPNodeEvidence", construct
    )
    diagnostic._build_suffix_node_evidence(
        precision=precision, observed=observed, node_index=0
    )
    assert captured["pinned_v10_node_receipt_artifact_sha256"] == core.artifact_sha256
    assert captured["primal_token_teacher_kl_sha256"] == core.token_teacher_kl_sha256
    assert captured["path_h4_sha256"] == core.path_node_h4_sha256
    assert captured["suffix_runtime_receipt_sha256"] == _sha("runtime-receipt")
    assert captured["primal_token_teacher_kl_sha256"] != _sha("runtime-primal")

    observed.core_node_receipt = SimpleNamespace(**vars(core))
    with pytest.raises(RuntimeError, match="ownership differs"):
        diagnostic._build_suffix_node_evidence(
            precision=precision, observed=observed, node_index=0
        )


def test_forward_mode_and_corrected_collision_semantics_are_report_gated() -> None:
    runtime = [
        {
            "suffix_runtime_receipt": {
                "ad_mechanism": "torch.func.jvp.forward_mode",
                "jvp_strict": True,
                "jvp_has_aux": True,
                "h4_dtype_cast_count": 1,
            }
        }
        for _ in range(64)
    ]
    assert diagnostic._strict_forward_mode_receipts_valid(runtime)
    runtime[0]["suffix_runtime_receipt"]["ad_mechanism"] = "reverse-over-reverse"
    assert not diagnostic._strict_forward_mode_receipts_valid(runtime)

    discrete = [
        {
            "interval_count": 5,
            "all_six_points_and_five_adjacent_intervals_authenticated": True,
            "descriptive_only_no_fit_or_correction": True,
            "cast_collision_excludes_static_coordinates": True,
        }
        for _ in range(16)
    ]
    assert diagnostic._discrete_cast_receipts_valid(discrete)
    discrete[0]["cast_collision_excludes_static_coordinates"] = False
    assert not diagnostic._discrete_cast_receipts_valid(discrete)


def test_optional_v10_observer_defaults_to_none_and_exact_replay_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        inspect.signature(diagnostic.v10diag._execute_prompt_precision)
        .parameters["path_node_observer"]
        .default
        is None
    )
    assert (
        inspect.signature(diagnostic.v10diag._execute_live_precision_grid)
        .parameters["path_node_observer"]
        .default
        is None
    )
    report = _mock_v10_report()
    pinned = diagnostic._index_v10_receipts(report)
    comparison = SimpleNamespace(
        artifact_sha256=pinned.comparison_artifact_sha256,
        metadata=lambda: report["family_equal_objective_precision_comparison"],
    )
    evidence = tuple(
        SimpleNamespace(example_id=example, artifact_sha256=artifact)
        for example, artifact in sorted(pinned.evidence_artifacts.items())
    )
    live = SimpleNamespace(
        endpoint_receipts=tuple(report["endpoint_precision_receipts"]),
        node_receipts=tuple(report["path_precision_node_receipts"]),
        node_receipt_set_sha256=report["path_precision_node_receipt_set_sha256"],
        comparison=comparison,
        evidence=evidence,
    )
    fits = {
        value["family_id"]: SimpleNamespace(
            metadata=lambda value=value: dict(value)
        )
        for value in report["folds"]
    }
    traces = tuple(SimpleNamespace(example_id=value["example_id"]) for value in report["prompt_receipts"])
    monkeypatch.setattr(
        diagnostic.v10diag.v9diag.v5diag,
        "_endpoint_prompt_receipts",
        lambda values: tuple(report["prompt_receipts"]),
    )
    replay = diagnostic._authenticate_live_v10_replay(
        report=report,
        pinned=pinned,
        live=live,
        fits=fits,
        traces=traces,
    )
    assert replay["optional_observer_did_not_change_default_V10_serialization"]

    live.node_receipts = (*live.node_receipts[:-1], {"changed": True})
    with pytest.raises(RuntimeError, match="serialization or evidence drifted"):
        diagnostic._authenticate_live_v10_replay(
            report=report,
            pinned=pinned,
            live=live,
            fits=fits,
            traces=traces,
        )


def _suffix_resources() -> dict[str, int]:
    return {
        "suffix_jvp_evaluation_count": 64,
        "suffix_differentiated_token_KL_evaluation_count": 64,
        "suffix_full_token_KL_crosscheck_evaluation_count": 64,
        "suffix_forward_structural_segment_call_count": 832,
        "suffix_logit_projection_call_count": 64,
        "suffix_h4_dtype_cast_count": 64,
        "suffix_directional_token_KL_element_count": 3212,
        "suffix_primal_token_KL_element_count": 3212,
        "suffix_full_crosscheck_token_KL_element_count": 3212,
        "suffix_full_H4_row_evaluation_count": 3788,
        "candidate_support_row_observation_count": 3276,
        "suffix_outside_candidate_support_H4_row_evaluation_count": 512,
        "suffix_forward_structural_row_layer_call_count": 49244,
        "candidate_support_row_layer_observation_count": 42588,
        "suffix_outside_candidate_support_row_layer_call_count": 6656,
        "discrete_cast_interval_count": 80,
        "discrete_cast_endpoint_validation_cast_count": 160,
    }


def test_exact_v11_resource_ledger_is_structural_not_a_flop_claim() -> None:
    report = _mock_v10_report()
    resources = diagnostic._resource_accounting(
        v10_resources=report["resources"],
        suffix_resources=_suffix_resources(),
    )
    assert resources["total_model_forward_count"] == 224
    assert resources["total_backward_call_count"] == 1039
    assert resources["suffix_jvp_evaluation_count"] == 64
    assert resources["suffix_forward_structural_segment_call_count"] == 832
    assert resources["suffix_forward_structural_row_layer_call_count"] == 49244
    assert resources["suffix_full_H4_row_evaluation_count"] == 3788
    assert resources["suffix_directional_token_KL_element_count"] == 3212
    assert resources["suffix_primal_token_KL_element_count"] == 3212
    assert resources["suffix_full_crosscheck_token_KL_element_count"] == 3212
    assert resources["candidate_support_row_observation_count"] == 3276
    assert resources["candidate_support_row_layer_observation_count"] == 42588
    assert (
        resources["suffix_outside_candidate_support_H4_row_evaluation_count"]
        == 512
    )
    assert (
        resources["suffix_outside_candidate_support_row_layer_call_count"]
        == 6656
    )
    assert resources["suffix_full_H4_row_evaluation_count"] == (
        resources["candidate_support_row_observation_count"]
        + resources["suffix_outside_candidate_support_H4_row_evaluation_count"]
    )
    assert resources["suffix_forward_structural_row_layer_call_count"] == (
        resources["candidate_support_row_layer_observation_count"]
        + resources["suffix_outside_candidate_support_row_layer_call_count"]
    )
    assert (
        resources["combined_candidate_and_suffix_h4_support_evaluations"]
        == 11032
    )
    assert resources["FLOP_or_total_compute_claim"] is False
    assert resources[
        "suffix_segment_calls_are_forward_structural_calls_not_AD_sweep_count"
    ]

    changed = {**_suffix_resources(), "suffix_h4_dtype_cast_count": 63}
    with pytest.raises(RuntimeError, match="resource accounting differs"):
        diagnostic._resource_accounting(
            v10_resources=report["resources"], suffix_resources=changed
        )

    # H4 support rows (819 x 4) and supervised token-KL elements
    # (803 x 4) are distinct units and must never be interchanged.
    conflated = {
        **_suffix_resources(),
        "candidate_support_row_observation_count": 3212,
        "candidate_support_row_layer_observation_count": 41756,
    }
    with pytest.raises(RuntimeError, match="resource accounting differs"):
        diagnostic._resource_accounting(
            v10_resources=report["resources"], suffix_resources=conflated
        )


def test_integrity_fails_closed_but_scientific_miss_is_not_an_integrity_error(
    tmp_path: Path,
) -> None:
    output = tmp_path / diagnostic.DEFAULT_OUTPUT.name
    assert not output.exists()
    diagnostic._require_integrity_gate_results(
        {"V10_authenticated": True, "suffix_primal_replayed": True}
    )
    with pytest.raises(RuntimeError, match="before publication"):
        diagnostic._require_integrity_gate_results(
            {"V10_authenticated": True, "suffix_primal_replayed": False}
        )
    assert not output.exists()


def test_no_knob_cli_output_suffix_and_pyproject_entry() -> None:
    assert vars(diagnostic.build_parser().parse_args([])) == {}
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(["--output", "elsewhere.json"])
    assert diagnostic.DEFAULT_OUTPUT.name.endswith(
        "candidate-joint-state-suffix-jvp-gl4-lofo-a-fit16-dev-v11.json"
    )
    pyproject = Path("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-candidate-"
        "joint-state-suffix-jvp-gl4-v11-a-dev = "
        '"fisher_graph.gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_'
        'joint_state_suffix_jvp_diagnostic:main"'
    ) in pyproject
