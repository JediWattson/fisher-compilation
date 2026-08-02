from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic as diagnostic,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_contraction_precision import (
    CandidateJointStateContractionPrecisionAccumulator,
)
from tests.test_complete_h4_tail_candidate_joint_state_contraction_precision import (
    _fixture as _core_contraction_fixture,
)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _mock_v11_report() -> dict[str, object]:
    return {
        "schema": diagnostic.v11diag._SCHEMA,
        "classification": "suffix_adjoint_ambiguity_same_a",
        "passed": False,
        "family_equal_suffix_jvp_comparison": {
            "artifact_sha256": diagnostic.V11_COMPARISON_ARTIFACT_SHA256,
            "adjoint_relative_rmse": 0.0002564458592054882,
            "jvp_closure_relative_rmse": 0.08675453755642996,
            "vjp_closure_relative_rmse": 0.08673388652083237,
        },
        "suffix_runtime_node_receipt_set_sha256": (
            diagnostic.V11_SUFFIX_RUNTIME_SET_SHA256
        ),
        "discrete_cast_prompt_receipt_set_sha256": (
            diagnostic.V11_DISCRETE_CAST_SET_SHA256
        ),
        "suffix_jvp_evidence": [{} for _ in range(16)],
        "suffix_runtime_node_receipts": [{} for _ in range(64)],
        "discrete_cast_prompt_receipts": [{} for _ in range(16)],
        "resources": {
            "total_model_forward_count": 224,
            "total_backward_call_count": 1039,
            "suffix_jvp_evaluation_count": 64,
            "combined_candidate_and_suffix_h4_support_evaluations": 11032,
        },
    }


def _valid_v12_receipt_grid() -> tuple[
    tuple[dict[str, object], ...],
    str,
    tuple[dict[str, object], ...],
    str,
    dict[str, object],
]:
    environment = diagnostic._typed_reduction_environment()
    environment_hash = str(environment["artifact_sha256"])
    raw: list[dict[str, object]] = []
    casts: list[dict[str, object]] = []
    digest_index = 1

    def digest() -> str:
        nonlocal digest_index
        value = f"{digest_index:064x}"
        digest_index += 1
        return value

    for prompt in range(16):
        example = f"example-{prompt:02d}"
        family = f"family-{prompt // 2}"
        for node in range(4):
            payload: dict[str, object] = {
                "example_id": example,
                "family_id": family,
                "node_index": node,
                "path_fraction_hex": diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[
                    node
                ].hex(),
                "quadrature_weight_hex": (
                    diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[node].hex()
                ),
                "raw_full_gradient_shape": (2, 1, 3, 4),
                "raw_full_gradient_dtype": "torch.float32",
                "raw_full_gradient_f32_sha256": digest(),
                "path_vjp_artifact_sha256": digest(),
                "selected_batch_index": 0,
                "support_indices_sha256": digest(),
                "selected_support_gradient_shape": (2, 2, 4),
                "selected_support_gradient_f32_sha256": digest(),
                "selected_then_promoted_gradient_f64_sha256": digest(),
                "pinned_v10_promoted_gradient_f64_sha256": digest(),
                "contraction_node_evidence_artifact_sha256": digest(),
                "pinned_v10_core_node_receipt_artifact_sha256": digest(),
                "raw_float32_selected_before_float64_promotion": True,
                "selected_promoted_equals_V10_bank_bitwise": True,
                "transient_gradient_retained_after_callback": False,
                "raw_tensors_serialized": False,
            }
            payload["artifact_sha256"] = (
                diagnostic.v10diag.token_v1._domain_sha256(
                    payload, domain=diagnostic._RAW_GRADIENT_DOMAIN
                )
            )
            raw.append(payload)
        cast_payload: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "node_index": 0,
            "path_fraction_hex": (
                diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0].hex()
            ),
            "ad_mechanism": "torch.func.jvp.forward_mode",
            "jvp_strict": True,
            "function": "isolated_float64_to_float32_cast_only",
            "device_type": "cpu",
            "input_dtype": "torch.float64",
            "output_dtype": "torch.float32",
            "shape": (1, 3, 4),
            "element_count": 12,
            "point_f64_sha256": digest(),
            "direction_f64_sha256": digest(),
            "primal_f32_sha256": digest(),
            "tangent_f32_sha256": digest(),
            "live_v10_full_h4_f32_sha256": digest(),
            "primal_matches_direct_cast_bitwise": True,
            "primal_matches_live_V10_full_H4_bitwise": True,
            "tangent_matches_direct_cast_bitwise": True,
            "typed_reduction_environment_sha256": environment_hash,
            "model_or_suffix_segment_evaluated": False,
            "raw_tensors_serialized": False,
        }
        cast_payload["artifact_sha256"] = (
            diagnostic.v10diag.token_v1._domain_sha256(
                cast_payload, domain=diagnostic._CAST_ONLY_JVP_DOMAIN
            )
        )
        casts.append(cast_payload)
    raw_values = tuple(raw)
    cast_values = tuple(casts)
    raw_set = diagnostic.v10diag.token_v1._domain_sha256(
        tuple(str(value["artifact_sha256"]) for value in raw_values),
        domain=diagnostic._RAW_GRADIENT_SET_DOMAIN,
    )
    cast_set = diagnostic.v10diag.token_v1._domain_sha256(
        tuple(str(value["artifact_sha256"]) for value in cast_values),
        domain=diagnostic._CAST_ONLY_JVP_SET_DOMAIN,
    )
    return raw_values, raw_set, cast_values, cast_set, environment


def test_v11_loader_uses_exact_file_and_logical_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    report = _mock_v11_report()

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(diagnostic.v10diag.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v11_report("v11.json") is report
    assert captured["expected_file_sha256"] == diagnostic.V11_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V11_REPORT_SHA256

    changed = {**report, "classification": "unexpected"}
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: changed,
    )
    with pytest.raises(RuntimeError, match="V11 anchor differs"):
        diagnostic._load_v11_report("v11.json")


def test_typed_reduction_environment_binds_public_torch_settings() -> None:
    receipt = diagnostic._typed_reduction_environment()
    assert receipt["torch_version"] == str(torch.__version__)
    assert receipt["reduction_device_type"] == "cpu"
    assert receipt["cpu_default_dtype"] == str(torch.get_default_dtype())
    assert receipt["torch_num_threads"] == torch.get_num_threads()
    assert receipt["torch_num_interop_threads"] == torch.get_num_interop_threads()
    assert receipt["deterministic_algorithms_enabled"] == (
        torch.are_deterministic_algorithms_enabled()
    )
    assert receipt["P_live_product_and_reduction_dtype"] == "torch.float32"
    assert receipt["internal_torch_sum_kernel_schedule_claimed"] is False
    assert diagnostic._typed_reduction_environment() == receipt
    assert not _contains_tensor(receipt)


def test_receipt_rehash_rejects_payload_set_order_duplicate_and_environment_drift() -> None:
    raw, raw_set, casts, cast_set, environment = _valid_v12_receipt_grid()
    authentication = diagnostic._validate_contraction_receipts(
        raw_receipts=raw,
        raw_receipt_set_sha256=raw_set,
        cast_receipts=casts,
        cast_receipt_set_sha256=cast_set,
        reduction_environment=environment,
    )
    assert authentication["raw_gradient_receipt_count"] == 64
    assert authentication["cast_only_jvp_receipt_count"] == 16
    assert authentication[
        "all_receipts_payload_set_order_and_ownership_rehashed"
    ] is True

    changed_raw = [dict(value) for value in raw]
    changed_raw[0]["raw_full_gradient_dtype"] = "torch.float64"
    with pytest.raises(RuntimeError, match="raw gradient receipt drifted"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=changed_raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=environment,
        )

    changed_cast = [dict(value) for value in casts]
    changed_cast[0]["primal_matches_live_V10_full_H4_bitwise"] = False
    with pytest.raises(RuntimeError, match="cast-only JVP receipt drifted"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=changed_cast,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=environment,
        )

    with pytest.raises(RuntimeError, match="raw gradient receipt set drifted"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256="a" * 64,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=environment,
        )

    with pytest.raises(RuntimeError, match="cast-only JVP receipt set drifted"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256="c" * 64,
            reduction_environment=environment,
        )

    reordered = list(raw)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(RuntimeError, match="order or ownership"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=reordered,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=environment,
        )

    duplicated = list(raw)
    duplicated[-1] = duplicated[0]
    with pytest.raises(RuntimeError, match="order or ownership"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=duplicated,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=environment,
        )

    reordered_casts = list(casts)
    reordered_casts[0], reordered_casts[1] = reordered_casts[1], reordered_casts[0]
    with pytest.raises(RuntimeError, match="order or ownership"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=reordered_casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=environment,
        )

    changed_environment = dict(environment)
    changed_environment["torch_num_threads"] = int(
        changed_environment["torch_num_threads"]
    ) + 1
    with pytest.raises(RuntimeError, match="environment receipt drifted"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=changed_environment,
        )

    changed_environment_hash = dict(environment)
    changed_environment_hash["artifact_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="environment receipt drifted"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=changed_environment_hash,
        )

    rehashed_changed_environment = dict(environment)
    rehashed_changed_environment["torch_num_threads"] = int(
        rehashed_changed_environment["torch_num_threads"]
    ) + 1
    environment_payload = dict(rehashed_changed_environment)
    environment_payload.pop("artifact_sha256")
    rehashed_changed_environment["artifact_sha256"] = (
        diagnostic.v10diag.token_v1._domain_sha256(
            environment_payload,
            domain=diagnostic._REDUCTION_ENVIRONMENT_DOMAIN,
        )
    )
    with pytest.raises(RuntimeError, match="environment semantics differ"):
        diagnostic._validate_contraction_receipts(
            raw_receipts=raw,
            raw_receipt_set_sha256=raw_set,
            cast_receipts=casts,
            cast_receipt_set_sha256=cast_set,
            reduction_environment=rehashed_changed_environment,
        )


def test_isolated_cast_only_jvp_matches_direct_cast_and_serializes_no_tensor() -> None:
    point = torch.tensor(
        [[[1.0, -2.0], [2.0**-40, 3.0]]], dtype=torch.float64
    )
    direction = torch.tensor(
        [[[2.0**-30, 1.25], [-4.0, 2.0**-50]]], dtype=torch.float64
    )
    environment = diagnostic._typed_reduction_environment()
    receipt = diagnostic._cast_only_jvp_receipt(
        example_id="example",
        family_id="family",
        point_f64=point,
        direction_f64=direction,
        live_h4_f32=point.float(),
        reduction_environment_sha256=str(environment["artifact_sha256"]),
    )
    assert receipt["ad_mechanism"] == "torch.func.jvp.forward_mode"
    assert receipt["jvp_strict"] is True
    assert receipt["primal_matches_direct_cast_bitwise"] is True
    assert receipt["primal_matches_live_V10_full_H4_bitwise"] is True
    assert receipt["tangent_matches_direct_cast_bitwise"] is True
    assert receipt["element_count"] == point.numel()
    assert receipt["model_or_suffix_segment_evaluated"] is False
    assert not _contains_tensor(receipt)

    with pytest.raises(RuntimeError, match="float64 primal"):
        diagnostic._cast_only_jvp_receipt(
            example_id="example",
            family_id="family",
            point_f64=point.float(),
            direction_f64=direction,
            live_h4_f32=point.float(),
            reduction_environment_sha256=str(environment["artifact_sha256"]),
        )

    changed_live = point.float().clone()
    changed_live[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="live V10 H4"):
        diagnostic._cast_only_jvp_receipt(
            example_id="example",
            family_id="family",
            point_f64=point,
            direction_f64=direction,
            live_h4_f32=changed_live,
            reduction_environment_sha256=str(environment["artifact_sha256"]),
        )


_TOKENS_SUPPORT = (
    (35, 36),
    (37, 38),
    (34, 35),
    (37, 38),
    (42, 43),
    (37, 38),
    (34, 35),
    (34, 35),
    (61, 62),
    (65, 66),
    (60, 61),
    (67, 68),
    (72, 73),
    (65, 66),
    (62, 63),
    (61, 62),
)


def _resource_fixture() -> tuple[tuple[object, ...], tuple[dict[str, int], ...], int]:
    values: list[object] = []
    casts: list[dict[str, int]] = []
    raw_full = 0
    for tokens, support in _TOKENS_SUPPORT:
        full = support + 8
        width = 640
        node_elements = 4 * tokens * support * width
        resources = {
            "quadrature_node_count": 4,
            "supervised_token_count": tokens,
            "full_h4_row_count": full,
            "support_h4_row_count": support,
            "outside_support_h4_row_count": 8,
            "gradient_f64_to_f32_roundtrip_validation_element_count": (
                node_elements
            ),
            "full_direction_cast_validation_element_count": full * width,
            "outside_support_zero_validation_element_count": 2 * 8 * width,
            "v10_gradient_weighted_add_element_count": node_elements,
            "v10_final_contraction_product_count": tokens * support * width,
            "nodewise_contraction_coordinate_observation_count_per_stage": (
                node_elements
            ),
            "nodewise_contraction_coordinate_observation_count_total": (
                4 * node_elements
            ),
            "actual_coordinate_product_bank_count": 3,
            "actual_coordinate_product_count_total": 3 * node_elements,
            "nodewise_GL4_token_weight_application_count": 4 * 4 * tokens,
        }
        values.append(SimpleNamespace(resource_accounting=resources, h4_width=width))
        casts.append({"element_count": full * width})
        raw_full += 4 * tokens * full * width
    return tuple(values), tuple(casts), raw_full


def test_exact_v12_resource_ledger_uses_distinct_units_and_is_not_flops() -> None:
    evidence, casts, raw_full = _resource_fixture()
    resources = diagnostic._contraction_resource_accounting(
        evidence=evidence,  # type: ignore[arg-type]
        raw_full_gradient_element_count=raw_full,
        cast_receipts=casts,
    )
    assert resources["contraction_quadrature_node_count"] == 64
    assert resources["contraction_supervised_token_count"] == 803
    assert resources["contraction_full_H4_row_count"] == 947
    assert resources["contraction_support_H4_row_count"] == 819
    assert resources["raw_f32_full_gradient_observation_element_count"] == 130_048_000
    assert resources["raw_f32_support_gradient_selection_element_count"] == 113_602_560
    assert resources["v10_final_contraction_product_count"] == 28_400_640
    assert (
        resources["nodewise_contraction_coordinate_observation_count_total"]
        == 454_410_240
    )
    assert resources["actual_coordinate_product_bank_count_per_prompt"] == 3
    assert resources["actual_coordinate_product_bank_observation_count_total"] == 48
    assert resources["actual_coordinate_product_count_total"] == 340_807_680
    assert resources["P_prod_and_P_live_share_one_f32_product_bank"] is True
    assert resources["cast_only_jvp_tangent_element_count"] == 606_080
    assert resources["retained_integrated_gradient_bank_count_after_node_three"] == 0

    bad_casts = list(casts)
    bad_casts[0] = {"element_count": bad_casts[0]["element_count"] + 1}
    with pytest.raises(RuntimeError, match="resource accounting differs"):
        diagnostic._contraction_resource_accounting(
            evidence=evidence,  # type: ignore[arg-type]
            raw_full_gradient_element_count=raw_full,
            cast_receipts=bad_casts,
        )


def test_composite_observer_validates_raw_f32_then_runs_unchanged_v11(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class FakeSuffix:
        def __init__(self, *, context: object) -> None:
            self.context = context

        def __call__(self, **observation: object) -> None:
            order.append("v11")

    class FakeAccumulator:
        def __init__(self, **kwargs: object) -> None:
            self.node_count = 0
            self.added: list[torch.Tensor] = []

        def add_node(self, **kwargs: object) -> object:
            order.append("v12")
            value = kwargs["token_support_h4_gradients_f64"]
            assert isinstance(value, torch.Tensor)
            self.added.append(value)
            node = SimpleNamespace(
                node_index=self.node_count,
                artifact_sha256="a" * 64,
            )
            self.node_count += 1
            return node

    monkeypatch.setattr(diagnostic.v11diag, "_SuffixJVPCollector", FakeSuffix)
    monkeypatch.setattr(
        diagnostic,
        "CandidateJointStateContractionPrecisionAccumulator",
        FakeAccumulator,
    )
    monkeypatch.setattr(
        diagnostic,
        "_cast_only_jvp_receipt",
        lambda **kwargs: {
            "example_id": kwargs["example_id"],
            "artifact_sha256": "b" * 64,
        },
    )
    context = object()
    collector = diagnostic._CompositeContractionPrecisionCollector(context=context)
    scalar = torch.zeros((1, 4, 3), dtype=torch.float32)
    joint = scalar.clone()
    joint[0, 0] = 1.0
    joint[0, 2] = -2.0
    raw = torch.arange(2 * 1 * 4 * 3, dtype=torch.float32).reshape(2, 1, 4, 3)
    support = torch.tensor([0, 2], dtype=torch.int64)
    promoted = raw[:, 0].index_select(1, support).double().contiguous()
    core = SimpleNamespace(
        artifact_sha256="c" * 64,
        node_index=0,
        path_fraction=diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0],
        quadrature_weight=diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[0],
        vjp_artifact_sha256="d" * 64,
    )
    validated: list[bool] = []
    path_vjp = SimpleNamespace(
        h4_gradients=raw,
        execution=SimpleNamespace(candidate_h4=(
            scalar.double()
            + diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0]
            * (joint.double() - scalar.double())
        ).float()),
        artifact_sha256="d" * 64,
        validate_integrity=lambda: validated.append(True),
    )
    collector(
        context=context,
        trace=SimpleNamespace(
            example_id="example", family_id="family", support_indices=support
        ),
        scalar_execution=SimpleNamespace(candidate_h4=scalar),
        joint_execution=SimpleNamespace(candidate_h4=joint),
        path_vjp=path_vjp,
        path_token_h4_gradients_f64=promoted,
        core_node_receipt=core,
        node_index=0,
        path_fraction=diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0],
        quadrature_weight=diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[0],
    )
    assert order == ["v12", "v11"]
    assert validated == [True]
    state = collector._prompts["example"]
    assert torch.equal(state.accumulator.added[0], promoted)  # type: ignore[attr-defined]
    receipt = state.raw_gradient_receipts[0]
    assert receipt["raw_full_gradient_dtype"] == "torch.float32"
    assert isinstance(receipt["raw_full_gradient_f32_sha256"], str)
    assert receipt["path_vjp_artifact_sha256"] == "d" * 64
    assert receipt["raw_float32_selected_before_float64_promotion"] is True
    assert receipt["selected_promoted_equals_V10_bank_bitwise"] is True
    assert not _contains_tensor(receipt)

    changed = promoted.clone()
    changed[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="does not equal V10 promoted"):
        collector(
            context=context,
            trace=SimpleNamespace(
                example_id="other", family_id="family", support_indices=support
            ),
            scalar_execution=SimpleNamespace(candidate_h4=scalar),
            joint_execution=SimpleNamespace(candidate_h4=joint),
            path_vjp=SimpleNamespace(
                h4_gradients=raw,
                execution=path_vjp.execution,
                artifact_sha256="d" * 64,
                validate_integrity=lambda: None,
            ),
            path_token_h4_gradients_f64=changed,
            core_node_receipt=core,
            node_index=0,
            path_fraction=diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0],
            quadrature_weight=diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[0],
        )


def test_observer_rejects_coordinate_drift_before_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSuffix:
        def __init__(self, *, context: object) -> None:
            pass

        def __call__(self, **observation: object) -> None:
            raise AssertionError("V11 must not run after coordinate drift")

    monkeypatch.setattr(diagnostic.v11diag, "_SuffixJVPCollector", FakeSuffix)
    context = object()
    collector = diagnostic._CompositeContractionPrecisionCollector(context=context)
    scalar = torch.zeros((1, 2, 2), dtype=torch.float32)
    raw = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    support = torch.tensor([0], dtype=torch.int64)
    alpha = diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[0]
    weight = diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[0]
    path = SimpleNamespace(
        h4_gradients=raw,
        execution=SimpleNamespace(candidate_h4=scalar),
        artifact_sha256="f" * 64,
        validate_integrity=lambda: None,
    )
    core = SimpleNamespace(
        artifact_sha256="a" * 64,
        node_index=0,
        path_fraction=alpha,
        quadrature_weight=weight,
        vjp_artifact_sha256="f" * 64,
    )
    with pytest.raises(RuntimeError, match="core node coordinates"):
        collector(
            context=context,
            trace=SimpleNamespace(
                example_id="example", family_id="family", support_indices=support
            ),
            scalar_execution=SimpleNamespace(candidate_h4=scalar),
            joint_execution=SimpleNamespace(candidate_h4=scalar),
            path_vjp=path,
            path_token_h4_gradients_f64=raw[:, 0, :, :].double(),
            core_node_receipt=core,
            node_index=0,
            path_fraction=alpha,
            quadrature_weight=weight + 1.0e-12,
        )


def test_collector_finalize_seals_a_real_four_node_core_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _built,
        suffix,
        support,
        displacement,
        tangent,
        gradients,
        _expected,
    ) = _core_contraction_fixture("collector-finalize")
    accumulator = CandidateJointStateContractionPrecisionAccumulator(
        support_indices=support,
        full_displacement_f64=displacement,
        full_cast_tangent_f32=tangent,
    )
    for receipt, gradient in zip(
        suffix.precision_evidence.path_evidence.node_receipts,
        gradients,
        strict=True,
    ):
        accumulator.add_node(
            node_receipt=receipt,
            token_support_h4_gradients_f64=gradient,
        )
    assert accumulator.node_count == 4

    collector = diagnostic._CompositeContractionPrecisionCollector(context=object())
    collector._suffix = SimpleNamespace(
        finalize=lambda _values: SimpleNamespace(evidence=(suffix,))
    )
    collector._prompts = {
        suffix.example_id: diagnostic._ObservedContractionPrompt(
            example_id=suffix.example_id,
            family_id=suffix.family_id,
            support_indices=support,
            full_displacement_f64=displacement,
            full_cast_tangent_f32=tangent,
            accumulator=accumulator,
            raw_gradient_receipts={
                index: {
                    "example_id": suffix.example_id,
                    "node_index": index,
                    "artifact_sha256": f"{index + 1:064x}",
                }
                for index in range(4)
            },
            cast_only_jvp_receipt={
                "example_id": suffix.example_id,
                "artifact_sha256": "f" * 64,
            },
        )
    }
    monkeypatch.setattr(diagnostic, "_EXPECTED_PROMPTS", 1)
    monkeypatch.setattr(
        diagnostic,
        "_contraction_resource_accounting",
        lambda **kwargs: {"collector_finalize_regression": 1},
    )
    result = collector.finalize((suffix.precision_evidence,))
    assert len(result.evidence) == 1
    assert result.evidence[0].example_id == suffix.example_id
    assert getattr(accumulator, "_integrated_gradient64") is None
    assert getattr(accumulator, "_token_p_v10_f64") is None
    with pytest.raises(RuntimeError, match="sealed"):
        accumulator.finalize(suffix_jvp_evidence=suffix)


def test_live_v11_section_authentication_is_exact_and_fail_closed() -> None:
    keys = (
        "input_binding",
        "folds",
        "prompt_receipts",
        "replayed_v10_endpoint_precision_receipts",
        "replayed_v10_path_precision_node_receipts",
        "replayed_v10_path_precision_node_receipt_set_sha256",
        "suffix_jvp_evidence",
        "suffix_runtime_node_receipts",
        "suffix_runtime_node_receipt_set_sha256",
        "discrete_cast_prompt_receipts",
        "discrete_cast_prompt_receipt_set_sha256",
        "family_equal_suffix_jvp_comparison",
        "resources",
    )
    report: dict[str, object] = {key: {"key": key} for key in keys}
    report["v10_control_binding"] = {
        "live_reproduction": {"v10": True},
        "v9_live_lineage_reproduction": {"v9": True},
    }
    live = {
        **{key: report[key] for key in keys},
        "v10_live_reproduction": {"v10": True},
        "v9_live_lineage_reproduction": {"v9": True},
    }
    receipt = diagnostic._authenticate_live_v11_sections(
        report=report, live_sections=live
    )
    assert receipt["all_live_sections_canonically_equal"] is True
    assert receipt["section_count"] == 15

    live["resources"] = {"changed": True}
    with pytest.raises(RuntimeError, match="resources"):
        diagnostic._authenticate_live_v11_sections(
            report=report, live_sections=live
        )


def test_combined_resources_preserve_v11_model_counts_and_make_no_flop_claim() -> None:
    evidence, casts, raw_full = _resource_fixture()
    contraction = diagnostic._contraction_resource_accounting(
        evidence=evidence,  # type: ignore[arg-type]
        raw_full_gradient_element_count=raw_full,
        cast_receipts=casts,
    )
    resources = diagnostic._resource_accounting(
        v11_resources={
            "total_model_forward_count": 224,
            "total_backward_call_count": 1039,
            "suffix_jvp_evaluation_count": 64,
            "phase_order": ("v11",),
        },
        contraction_resources=contraction,
    )
    assert resources["total_model_forward_count"] == 224
    assert resources["total_backward_call_count"] == 1039
    assert resources["suffix_jvp_evaluation_count"] == 64
    assert resources["V12_additional_model_forward_count_is_zero"] is True
    assert resources["V12_reuses_existing_V10_gradient_banks"] is True
    assert resources["FLOP_or_total_compute_claim"] is False


def test_run_authenticates_v11_before_context_and_again_after_model_work() -> None:
    source = inspect.getsource(
        diagnostic.run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic
    )
    first_load = source.index("v11_report_before = _load_v11_report")
    context = source.index("prepare_complete_h4_rank320_live_context")
    second_load = source.index("v11_report_after = _load_v11_report")
    assert first_load < context < second_load
    assert "path_node_observer=collector" in source
    assert "_authenticate_live_v11_sections" in source


def test_no_knob_cli_output_suffix_and_pyproject_entry() -> None:
    assert vars(diagnostic.build_parser().parse_args([])) == {}
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(["--output", "elsewhere.json"])
    assert diagnostic.DEFAULT_OUTPUT.name.endswith(
        "candidate-joint-state-contraction-precision-gl4-lofo-a-fit16-dev-v12.json"
    )
    pyproject = Path("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-candidate-"
        "joint-state-contraction-precision-gl4-v12-a-dev = "
        '"fisher_graph.gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_'
        'joint_state_contraction_precision_diagnostic:main"'
    ) in pyproject


def test_integrity_gate_fails_closed_and_safety_is_tensor_free() -> None:
    diagnostic._require_integrity_gate_results(
        {"V11_exact": True, "raw_gradients_exact": True}
    )
    with pytest.raises(RuntimeError, match="before publication"):
        diagnostic._require_integrity_gate_results(
            {"V11_exact": True, "raw_gradients_exact": False}
        )
    safety = diagnostic._safety_metadata()
    assert safety["contains_gradient_tensors"] is False
    assert safety["artifact_must_remain_outside_git"] is True
    assert not _contains_tensor(safety)
