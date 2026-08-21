from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_exact_edge_sidecar_experiment as edge_experiment
from fisher_graph.gemma3_exact_edge_sidecar_experiment import (
    ARM_ORDER,
    EDGE_ARM_ORDER,
    EXPECTED_PROTOCOL_SHA256,
    JVP_DIRECTION_COUNT_PER_EDGE,
    NEGATIVE_CONTROL_ARM_IDS,
    EdgeSidecarCausalEdgeMetric,
    EdgeSidecarCausalRequest,
    EdgeSidecarPreparedBatch,
    build_edge_sidecar_causal_receipt,
    capture_exact_edge_sidecar_role,
    evaluate_exact_edge_sidecar_predictive,
    fit_exact_edge_sidecar_catalog,
    load_pinned_edge_sidecar_split,
    validate_exact_edge_sidecar_experiment_report,
)
from fisher_graph.gemma3_exact_edge_sidecar_protocol import (
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_PANEL_SHA256,
    PINNED_SPLIT_ID_SHA256S,
    build_exact_edge_sidecar_protocol,
)
from test_gemma3_exact_edge_sidecar_observer import _exact_executor, _sequence


_SOURCE_FINGERPRINT = "a" * 64


def _prepared_batch(
    role: str,
) -> tuple[EdgeSidecarPreparedBatch, dict[str, str]]:
    split = load_pinned_edge_sidecar_split()
    examples = split.examples_for(role)
    valid = torch.ones(len(examples), 2, dtype=torch.bool)
    input_ids = torch.tensor(
        [
            [index % 7, (index + 1) % 7]
            for index in range(len(examples))
        ],
        dtype=torch.long,
    )
    row_keys = tuple(
        (example.example_id, position)
        for example in examples
        for position in range(2)
    )
    return (
        EdgeSidecarPreparedBatch(
            input_ids=input_ids,
            sequence=_sequence(valid=valid),
            valid_mask=valid,
            row_keys=row_keys,
        ),
        {example.example_id: example.family_id for example in examples},
    )


def _A_prepared_batch() -> tuple[EdgeSidecarPreparedBatch, dict[str, str]]:
    return _prepared_batch("A_fit")


def _role_capture(
    role: str,
    *,
    unique_site_designs: bool = False,
):
    batch, families = _prepared_batch(role)
    result = capture_exact_edge_sidecar_role(
        executor=_exact_executor(),
        batches=(batch,),
        role=role,
        source_fingerprint=_SOURCE_FINGERPRINT,
        example_ids_sha256=PINNED_SPLIT_ID_SHA256S[role],
        family_by_example=families,
    )
    if not unique_site_designs:
        return result
    offsets = torch.arange(
        1,
        result.capture.hidden_width + 1,
        dtype=torch.float32,
    )
    observations = tuple(
        replace(
            item,
            values=(item.values + (index + 1) * 1.0e-4 * offsets)
            .contiguous(),
        )
        for index, item in enumerate(result.capture.observations)
    )
    return replace(
        result,
        capture=replace(result.capture, observations=observations),
    )


@pytest.fixture()
def toy_catalog(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(edge_experiment, "PINNED_HIDDEN_WIDTH", 4)
    fit_capture = _role_capture("A_fit", unique_site_designs=True)
    catalog = fit_exact_edge_sidecar_catalog(fit_capture)
    return fit_capture, catalog


def test_pinned_split_binds_real_A_B_C_ids_without_exporting_prompts() -> None:
    split = load_pinned_edge_sidecar_split()

    assert split.panel_sha256 == PINNED_PANEL_SHA256
    assert len(split.A_fit) == 36
    assert len(split.B_predictive) == 12
    assert len(split.C_intervention) == 12
    assert not (
        {item.example_id for item in split.A_fit}
        & {item.example_id for item in split.B_predictive}
    )
    assert not (
        {item.example_id for item in split.A_fit}
        & {item.example_id for item in split.C_intervention}
    )
    metadata = split.metadata()
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_token_ids"] is False
    assert metadata["roles"]["A_fit"]["example_ids_sha256"] == (
        PINNED_SPLIT_ID_SHA256S["A_fit"]
    )


def test_role_capture_observes_all_sites_and_charges_every_exact_pass() -> None:
    executor = _exact_executor()
    batch, families = _A_prepared_batch()
    before = {
        name: value.detach().clone()
        for name, value in executor.state_dict().items()
    }

    result = capture_exact_edge_sidecar_role(
        executor=executor,
        batches=(batch,),
        role="A_fit",
        source_fingerprint=_SOURCE_FINGERPRINT,
        example_ids_sha256=PINNED_SPLIT_ID_SHA256S["A_fit"],
        family_by_example=families,
    )

    assert result.role == "A_fit"
    assert len(result.capture.observations) == 73
    assert len(result.capture.row_keys) == 72
    assert len(result.score_gradients) == 73
    assert result.exact_forward_count == 2
    assert result.native_forward_count == 0
    assert result.backward_count == 1
    assert result.exact_gate["passed"] is True
    assert result.exact_gate["event_order_equal"] is True
    assert result.exact_gate[
        "every_mutation_chart_input_byte_identical"
    ] is True
    assert result.exact_gate["every_mutation_delta_byte_identical"] is True
    assert result.exact_gate["final_carrier_byte_identical"] is True
    assert result.exact_gate["normalized_output_byte_identical"] is True
    assert result.exact_gate["logits_byte_identical"] is True
    assert result.exact_gate["logits_byte_identical"] is True
    assert result.metadata()["contains_activations"] is False
    assert result.metadata()["contains_gradients"] is False
    weights = result.fisher_row_weights("mutation.3.output")
    assert weights.shape == (72,)
    assert torch.isfinite(weights).all()
    assert bool((weights > 0).all()) is True
    for name, expected in before.items():
        assert torch.equal(executor.state_dict()[name], expected), name


def test_native_provider_is_a_separately_counted_literal_exact_gate() -> None:
    executor = _exact_executor()
    batch, families = _A_prepared_batch()

    result = capture_exact_edge_sidecar_role(
        executor=executor,
        batches=(batch,),
        role="A_fit",
        source_fingerprint=_SOURCE_FINGERPRINT,
        example_ids_sha256=PINNED_SPLIT_ID_SHA256S["A_fit"],
        family_by_example=families,
        native_logits_provider=lambda item: executor.forward_components(
            item.input_ids,
            item.sequence,
        ).logits,
    )

    native = result.exact_gate["native_exact_control"]
    assert native["run"] is True
    assert native["literal_valid_query_logit_parity"] is True
    assert native["delta_nll_per_token"] == 0.0
    assert native["native_to_candidate_kl_per_query"] == 0.0
    assert native["top1_agreement_to_native"] == 1.0
    assert native["maximum_absolute_logit_error"] == 0.0
    assert result.exact_forward_count == 2
    assert result.native_forward_count == 1
    assert result.backward_count == 1


def test_native_exact_gate_is_literal_on_valid_queries_not_padding() -> None:
    executor = _exact_executor()
    split = load_pinned_edge_sidecar_split()
    examples = split.A_fit
    valid = torch.tensor([[False, True, True]]).expand(
        len(examples),
        -1,
    ).clone()
    input_ids = torch.tensor(
        [
            [0, index % 7, (index + 1) % 7]
            for index in range(len(examples))
        ],
        dtype=torch.long,
    )
    batch = EdgeSidecarPreparedBatch(
        input_ids=input_ids,
        sequence=_sequence(valid=valid),
        valid_mask=valid,
        row_keys=tuple(
            (example.example_id, position)
            for example in examples
            for position in range(2)
        ),
    )
    families = {
        example.example_id: example.family_id for example in examples
    }

    def native_with_padding_only_difference(
        item: EdgeSidecarPreparedBatch,
    ) -> torch.Tensor:
        logits = executor.forward_components(
            item.input_ids,
            item.sequence,
        ).logits.clone()
        logits[~item.valid_mask] += 1.0
        return logits

    result = capture_exact_edge_sidecar_role(
        executor=executor,
        batches=(batch,),
        role="A_fit",
        source_fingerprint=_SOURCE_FINGERPRINT,
        example_ids_sha256=PINNED_SPLIT_ID_SHA256S["A_fit"],
        family_by_example=families,
        native_logits_provider=native_with_padding_only_difference,
    )

    native = result.exact_gate["native_exact_control"]
    assert result.exact_gate["passed"] is True
    assert native["literal_valid_query_logit_parity"] is True
    assert native["maximum_absolute_logit_error"] == 0.0


def test_role_capture_rejects_row_role_spoofing_before_model_execution() -> None:
    executor = _exact_executor()
    batch, families = _A_prepared_batch()
    split = load_pinned_edge_sidecar_split()
    spoof_id = split.B_predictive[0].example_id
    spoofed_rows = ((spoof_id, 0), (spoof_id, 1), *batch.row_keys[2:])
    spoofed_families = dict(families)
    spoofed_families.pop(batch.row_keys[0][0])
    spoofed_families[spoof_id] = split.B_predictive[0].family_id

    with pytest.raises(ValueError, match="pinned role identities"):
        capture_exact_edge_sidecar_role(
            executor=executor,
            batches=(replace(batch, row_keys=spoofed_rows),),
            role="A_fit",
            source_fingerprint=_SOURCE_FINGERPRINT,
            example_ids_sha256=PINNED_SPLIT_ID_SHA256S["A_fit"],
            family_by_example=spoofed_families,
        )
    noncontiguous = (
        (batch.row_keys[0][0], 1),
        (batch.row_keys[0][0], 2),
        *batch.row_keys[2:],
    )
    with pytest.raises(ValueError, match="positions must be contiguous"):
        capture_exact_edge_sidecar_role(
            executor=executor,
            batches=(replace(batch, row_keys=noncontiguous),),
            role="A_fit",
            source_fingerprint=_SOURCE_FINGERPRINT,
            example_ids_sha256=PINNED_SPLIT_ID_SHA256S["A_fit"],
            family_by_example=families,
        )


def test_family_shuffle_handles_variable_length_prompt_rows() -> None:
    row_keys = tuple(
        (f"{family}.{example}", position)
        for family, lengths in (
            ("a", (3,)),
            ("b", (3,)),
            ("c", (2,)),
            ("d", (2,)),
            ("e", (1,)),
            ("f", (1,)),
        )
        for example, length in enumerate(lengths)
        for position in range(length)
    )
    families = {
        example_id: example_id.split(".", maxsplit=1)[0]
        for example_id, _ in row_keys
    }

    permutation = edge_experiment._family_shuffle_permutation(
        row_keys,
        families,
    )

    assert sorted(permutation.tolist()) == list(range(len(row_keys)))
    assert sorted(position for _, position in row_keys) == sorted(
        row_keys[index][1] for index in permutation.tolist()
    )
    assert all(
        families[row_keys[index][0]] != families[row_keys[donor][0]]
        for index, donor in enumerate(permutation.tolist())
    )


def test_discovery_fails_closed_when_only_provenance_can_break_a_design_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_capture = _role_capture("A_fit")
    monkeypatch.setattr(edge_experiment, "PINNED_HIDDEN_WIDTH", 4)

    with pytest.raises(
        RuntimeError,
        match="provenance-blind indistinguishable design tie",
    ):
        fit_exact_edge_sidecar_catalog(fit_capture)


def test_A_only_catalog_is_dense_forward_cross_block_and_control_matched(
    toy_catalog,
) -> None:
    _, catalog = toy_catalog
    real = catalog.arm("real_forward_cross_block_edges")

    assert tuple(arm.arm_id for arm in catalog.arms) == EDGE_ARM_ORDER
    assert real.edges
    assert catalog.provenance_ineligible_target_count == 2
    assert catalog.fit_dense_solve_count == (
        catalog.fit_candidate_count + 3 * len(real.edges)
    )
    assert catalog.fit_design_mac_estimate > 0
    assert catalog.fit_dense_solve_flop_lower_bound > 0
    assert catalog.rank_decomposition_flop_lower_bound > 0
    assert all(arm.topology == real.topology for arm in catalog.arms)
    assert all(
        artifact.edge.metadata()["strictly_forward"] is True
        and artifact.edge.metadata()["cross_block"] is True
        and artifact.edge.metadata()["cross_family"] is True
        and artifact.edge.coefficient.shape == (4, 4)
        and artifact.edge.metadata()[
            "event_or_layer_features_in_numerical_design"
        ]
        is False
        for arm in catalog.arms
        for artifact in arm.edges
    )
    provenance = catalog.arm("source_provenance_permutation_control")
    assert all(
        artifact.claimed_source_site_id != artifact.packet_source_site_id
        for artifact in provenance.edges
    )
    metadata = catalog.metadata()
    assert metadata["B_or_C_used_for_fit_or_selection"] is False
    assert metadata["low_rank_factorization_used"] is False
    assert metadata["model_mutated"] is False
    assert metadata["compression"] is False


def test_frozen_catalog_detects_dense_and_local_tensor_mutation(
    toy_catalog,
) -> None:
    _, catalog = toy_catalog
    artifact = catalog.arm("real_forward_cross_block_edges").edges[0]
    coefficient = artifact.edge.coefficient.detach().clone()
    local_weight = catalog.local_baselines[0].weight.detach().clone()

    with torch.no_grad():
        artifact.edge.coefficient[0, 0].add_(1.0)
    with pytest.raises(RuntimeError, match="coefficient changed"):
        catalog.assert_integrity()
    with torch.no_grad():
        artifact.edge.coefficient.copy_(coefficient)
        catalog.local_baselines[0].weight[0, 0].add_(1.0)
    with pytest.raises(RuntimeError, match="catalog tensors changed"):
        catalog.assert_integrity()
    with torch.no_grad():
        catalog.local_baselines[0].weight.copy_(local_weight)
    catalog.assert_integrity()


def test_predictive_arm_uses_frozen_A_catalog_on_disjoint_B_without_refit(
    toy_catalog,
) -> None:
    fit_capture, catalog = toy_catalog
    heldout = _role_capture("B_predictive", unique_site_designs=True)
    digest_before = catalog.artifact_sha256
    coefficients_before = tuple(
        artifact.edge.coefficient.detach().clone()
        for arm in catalog.arms
        for artifact in arm.edges
    )

    receipt = evaluate_exact_edge_sidecar_predictive(
        catalog=catalog,
        heldout_capture=heldout,
    )

    assert receipt.parent_edge_catalog_sha256 == digest_before
    assert receipt.B_row_identity_sha256 == heldout.capture.row_identity_sha256
    assert tuple(receipt.metrics_by_arm) == ARM_ORDER
    assert tuple(receipt.margins) == NEGATIVE_CONTROL_ARM_IDS
    assert receipt.metrics_by_arm["zero_edge_control"][
        "relative_SSE_gain_vs_local"
    ] == 0.0
    assert receipt.metadata()["heldout_refit_performed"] is False
    assert receipt.metadata()["edge_catalog_pruned_or_refitted_on_B"] is False
    assert all(
        baseline.fit_row_identity_sha256
        != heldout.capture.row_identity_sha256
        for baseline in catalog.local_baselines
    )
    assert catalog.artifact_sha256 == digest_before
    assert all(
        torch.equal(artifact.edge.coefficient, expected)
        for artifact, expected in zip(
            (
                artifact
                for arm in catalog.arms
                for artifact in arm.edges
            ),
            coefficients_before,
            strict=True,
        )
    )

    reused_A_as_B = replace(
        fit_capture,
        role="B_predictive",
        example_ids_sha256=PINNED_SPLIT_ID_SHA256S["B_predictive"],
    )
    with pytest.raises(ValueError, match="reused fit"):
        evaluate_exact_edge_sidecar_predictive(
            catalog=catalog,
            heldout_capture=reused_A_as_B,
        )


def _causal_metric(
    *,
    source_site_id: str,
    target_event_id: str,
    finite_cosine: float = 0.90,
) -> EdgeSidecarCausalEdgeMetric:
    batch_count = 1
    jvp_count = 4 * JVP_DIRECTION_COUNT_PER_EDGE * batch_count
    finite_count = (
        4 * 2 * JVP_DIRECTION_COUNT_PER_EDGE * 3 * batch_count
    )
    forward_count = (
        4 * (2 + JVP_DIRECTION_COUNT_PER_EDGE + 2 * JVP_DIRECTION_COUNT_PER_EDGE * 3)
        * batch_count
    )
    controls = tuple(
        zip(
            NEGATIVE_CONTROL_ARM_IDS,
            (0.0, 0.70, 0.72, 0.74),
            strict=True,
        )
    )
    return EdgeSidecarCausalEdgeMetric(
        source_site_id=source_site_id,
        target_event_id=target_event_id,
        C_batch_count=batch_count,
        minimum_real_jvp_cosine=0.95,
        maximum_real_jvp_relative_norm_error=0.10,
        minimum_real_finite_cosine=finite_cosine,
        maximum_real_finite_relative_norm_error=0.20,
        maximum_adjacent_epsilon_derivative_disagreement=0.05,
        control_minimum_jvp_cosines=controls,
        control_minimum_finite_cosines=controls,
        exact_forward_count=forward_count,
        exact_jvp_count=jvp_count,
        backward_or_jvp_count=jvp_count,
        finite_intervention_forward_count=finite_count,
    )


def _artifact_digest(payload: dict[str, object], key: str) -> None:
    payload.pop(key, None)
    payload[key] = edge_experiment._json_sha256(
        payload,
        domain=edge_experiment._ARTIFACT_HASH_DOMAIN,
    )


def _report_digest(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = edge_experiment._report_sha256(report)


def _valid_strict_report(*, causal: bool = False) -> dict[str, object]:
    protocol = build_exact_edge_sidecar_protocol()
    split = load_pinned_edge_sidecar_split()
    protocol_receipt = {
        "schema": protocol["schema"],
        "format_version": protocol["format_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "artifact_path": "unit-fixture.json",
        "authenticated_against_current_builder": True,
    }
    site_receipts = protocol["unified_exact_chart_contract"]["sites"]
    sites = tuple(
        SimpleNamespace(site_id=item["site_id"])
        for item in site_receipts
    )
    split_metadata = split.metadata()

    def exact_gate() -> dict[str, object]:
        return {
            "passed": True,
            "event_order_equal": True,
            "every_mutation_chart_input_byte_identical": True,
            "every_mutation_delta_byte_identical": True,
            "final_carrier_byte_identical": True,
            "normalized_output_byte_identical": True,
            "logits_byte_identical": True,
            "source_state_fingerprint_unchanged": True,
            "native_exact_control": {
                "run": True,
                "literal_valid_query_logit_parity": True,
                "delta_nll_per_token": 0.0,
                "native_to_candidate_kl_per_query": 0.0,
                "top1_agreement_to_native": 1.0,
                "maximum_absolute_logit_error": 0.0,
            },
        }

    class CaptureReceipt:
        def __init__(self, role: str, row_hash: str) -> None:
            self.role = role
            self.exact_gate = exact_gate()
            self.exact_forward_count = 2
            self.native_forward_count = 1
            self.backward_count = 1
            self.capture = SimpleNamespace(
                observations=sites,
                row_identity_sha256=row_hash,
            )

        def metadata(self) -> dict[str, object]:
            return {
                "role": self.role,
                "example_ids_sha256": PINNED_SPLIT_ID_SHA256S[self.role],
                "family_assignment_sha256": split_metadata["roles"][
                    self.role
                ]["family_assignment_sha256"],
                "row_identity_sha256": self.capture.row_identity_sha256,
                "row_count": 2,
                "site_count": 73,
                "score_gradient_site_count": 73,
                "hidden_width": 640,
                "exact_gate": copy.deepcopy(self.exact_gate),
                "exact_forward_count": self.exact_forward_count,
                "native_forward_count": self.native_forward_count,
                "backward_count": self.backward_count,
                "contains_prompt_text": False,
                "contains_token_ids": False,
                "contains_activations": False,
                "contains_gradients": False,
            }

    captures = {
        role: CaptureReceipt(role, f"{index + 1:064x}")
        for index, role in enumerate(edge_experiment.ROLE_ORDER)
    }
    edge = {
        "source_site_id": "lexical.ingress.output",
        "claimed_source_site_id": "lexical.ingress.output",
        "packet_source_site_id": "lexical.ingress.output",
        "target_event_id": "mutation.3",
        "strictly_forward": True,
        "cross_block": True,
        "cross_family": True,
        "coefficient_shape": [640, 640],
        "edge_low_rank_factorization_used": False,
        "dense_640_by_640": True,
        "factorized": False,
        "coefficient_sha256": "a" * 64,
        "fit_design_sha256": "b" * 64,
        "realized_numerical_rank": 1,
        "coefficient_serialized_in_report": False,
    }
    arms = []
    for arm_id in EDGE_ARM_ORDER:
        arms.append(
            {
                "arm_id": arm_id,
                "edge_count": 1,
                "hard_zero": arm_id == "zero_edge_control",
                "fit_permutation_sha256": (
                    None
                    if arm_id
                    in ("real_forward_cross_block_edges", "zero_edge_control")
                    else "c" * 64
                ),
                "maximum_incoming_edges_per_target": 1,
                "stored_scalar_count": (
                    0 if arm_id == "zero_edge_control" else 640 * 640
                ),
                "edges": [copy.deepcopy(edge)],
                "model_weights_mutated": False,
                "model_carrier_mutated_by_sidecar": False,
                "model_logits_mutated_by_sidecar": False,
                "compression": False,
            }
        )
    catalog_metadata: dict[str, object] = {
        "schema": "fisher_graph.gemma3_exact_edge_sidecar.edge_arm_catalog.v1",
        "local_baseline_count": 36,
        "local_baseline_scalar_count": 36 * (640 * 640 + 640),
        "local_baselines": [],
        "arm_order": list(EDGE_ARM_ORDER),
        "arms": arms,
        "real_edge_count": 1,
        "sidecar_edge_scalar_count": 4 * 640 * 640,
        "fit_candidate_count": 10,
        "fit_dense_solve_count": 13,
        "fit_design_mac_estimate": 100,
        "fit_dense_solve_flop_lower_bound": 200,
        "rank_decomposition_flop_lower_bound": 300,
        "provenance_ineligible_target_count": 2,
        "provenance_ineligible_target_reason": (
            "fewer_than_two_distinct_eligible_source_packets"
        ),
        "cost_estimates_are_backend_independent_lower_bounds": True,
        "full_width_coordinates": True,
        "dense_640_by_640_edges": True,
        "low_rank_factorization_used": False,
        "maximum_real_incoming_edges_per_target": 1,
        "B_or_C_used_for_fit_or_selection": False,
        "installed_in_exact_runtime": False,
        "model_mutated": False,
        "compression": False,
    }
    _artifact_digest(catalog_metadata, "artifact_sha256")

    class CatalogReceipt:
        local_baseline_scalar_count = catalog_metadata[
            "local_baseline_scalar_count"
        ]
        sidecar_edge_scalar_count = catalog_metadata[
            "sidecar_edge_scalar_count"
        ]
        fit_design_mac_estimate = catalog_metadata["fit_design_mac_estimate"]
        fit_dense_solve_flop_lower_bound = catalog_metadata[
            "fit_dense_solve_flop_lower_bound"
        ]
        rank_decomposition_flop_lower_bound = catalog_metadata[
            "rank_decomposition_flop_lower_bound"
        ]

        def metadata(self) -> dict[str, object]:
            return copy.deepcopy(catalog_metadata)

    gains = {
        "local_target_baseline": 0.0,
        "real_forward_cross_block_edges": 0.20,
        "zero_edge_control": 0.0,
        "row_shuffled_control": 0.05,
        "family_shuffled_control": 0.04,
        "source_provenance_permutation_control": 0.03,
    }
    predictive_metadata: dict[str, object] = {
        "schema": "fisher_graph.gemma3_exact_edge_sidecar.predictive_receipt.v1",
        "parent_edge_catalog_sha256": catalog_metadata["artifact_sha256"],
        "B_row_identity_sha256": captures[
            "B_predictive"
        ].capture.row_identity_sha256,
        "arm_order": list(ARM_ORDER),
        "arm_metrics": {
            arm_id: {"relative_SSE_gain_vs_local": gain}
            for arm_id, gain in gains.items()
        },
        "per_target_metrics": [{"real_gain_positive": True}],
        "real_gain_margin_over_controls": {
            arm_id: gains["real_forward_cross_block_edges"] - gains[arm_id]
            for arm_id in NEGATIVE_CONTROL_ARM_IDS
        },
        "paired_family_bootstrap_lower_bound": 0.10,
        "bootstrap_seed": edge_experiment.DETERMINISTIC_SEED,
        "bootstrap_resamples": edge_experiment.BOOTSTRAP_RESAMPLES,
        "prediction_mac_estimate": 400,
        "minimum_real_gain_vs_local": 0.02,
        "minimum_real_gain_margin_over_each_negative_control": 0.01,
        "edge_catalog_pruned_or_refitted_on_B": False,
        "heldout_refit_performed": False,
        "passed": True,
    }
    _artifact_digest(predictive_metadata, "receipt_sha256")

    class PredictiveReceipt:
        passed = True
        prediction_mac_estimate = predictive_metadata[
            "prediction_mac_estimate"
        ]

        def metadata(self) -> dict[str, object]:
            return copy.deepcopy(predictive_metadata)

    causal_receipt = None
    if causal:
        metric = _causal_metric(
            source_site_id="lexical.ingress.output",
            target_event_id="mutation.3",
        )
        causal_metadata: dict[str, object] = {
            "schema": "fisher_graph.gemma3_exact_edge_sidecar.causal_receipt.v1",
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "parent_edge_catalog_sha256": catalog_metadata["artifact_sha256"],
            "parent_predictive_receipt_sha256": predictive_metadata[
                "receipt_sha256"
            ],
            "C_row_identity_sha256": captures[
                "C_intervention"
            ].capture.row_identity_sha256,
            "source_fingerprint_before": "d" * 64,
            "source_fingerprint_after": "d" * 64,
            "source_state_fingerprint_restored_after_each_probe": True,
            "direction_count_per_edge_per_batch": JVP_DIRECTION_COUNT_PER_EDGE,
            "direction_normalization": (
                edge_experiment.DIRECTION_IMPLEMENTATION_RECEIPT
            ),
            "protocol_direction_spec": edge_experiment.DIRECTION_PROTOCOL_LITERAL,
            "finite_displacement_scale": (
                edge_experiment.FINITE_DISPLACEMENT_SCALE_LITERAL
            ),
            "epsilon_times_source_RMS": list(
                edge_experiment.INTERVENTION_EPSILON_LADDER
            ),
            "edge_metrics": [metric.metadata()],
            "edge_count": 1,
            "exact_forward_count": metric.exact_forward_count,
            "backward_or_JVP_count": metric.backward_or_jvp_count,
            "finite_intervention_forward_count": (
                metric.finite_intervention_forward_count
            ),
            "peak_resident_bytes": 1,
            "JVP_gate_passed": True,
            "finite_intervention_gate_passed": True,
            "negative_control_margin_evaluated_per_edge": True,
            "baseline_model_output_replaced": False,
            "model_weights_mutated": False,
            "shadow_execution_only": True,
            "passed": True,
        }
        _artifact_digest(causal_metadata, "receipt_sha256")

        class CausalReceipt:
            edge_metrics = (metric,)
            exact_forward_count = metric.exact_forward_count
            backward_or_jvp_count = metric.backward_or_jvp_count
            finite_intervention_forward_count = (
                metric.finite_intervention_forward_count
            )
            peak_resident_bytes = 1

            def metadata(self) -> dict[str, object]:
                return copy.deepcopy(causal_metadata)

        causal_receipt = CausalReceipt()

    class CompilationReceipt:
        def metadata(self) -> dict[str, object]:
            return {
                "compilation_kind": "exact_39_event_control",
                "exact_parameter_inventory_parity": True,
                "source_model_parameter_count": (
                    edge_experiment.PINNED_SOURCE_PARAMETER_COUNT
                ),
                "executor_manifest": {
                    "exact_control_arm": True,
                    "event_count": 39,
                    "attention_mutation_count": 18,
                    "mlp_mutation_count": 18,
                    "direct_compiled_attention_mutation_count": 18,
                    "transport_edge_count": 0,
                    "tied_lexical_ingress_and_egress_present": True,
                },
            }

    return edge_experiment._assemble_report(
        protocol_receipt=protocol_receipt,
        protocol=protocol,
        split=split,
        model_metadata={
            "model_id": PINNED_MODEL_ID,
            "requested_revision": PINNED_MODEL_REVISION,
        },
        compilation=CompilationReceipt(),
        captures=captures,
        catalog=CatalogReceipt(),
        predictive=PredictiveReceipt(),
        causal=causal_receipt,
        source_fingerprint_before="d" * 64,
        source_fingerprint_after="d" * 64,
        batch_size=4,
        peak_resident_bytes=1,
    )


def test_causal_receipt_is_bound_to_frozen_C_topology_and_call_ledger(
    toy_catalog,
) -> None:
    _, catalog = toy_catalog
    predictive_capture = _role_capture(
        "B_predictive",
        unique_site_designs=True,
    )
    predictive = evaluate_exact_edge_sidecar_predictive(
        catalog=catalog,
        heldout_capture=predictive_capture,
    )
    C_batch, _ = _prepared_batch("C_intervention")
    C_capture = _role_capture("C_intervention", unique_site_designs=True)
    request = EdgeSidecarCausalRequest(
        protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        source_fingerprint=_SOURCE_FINGERPRINT,
        executor=_exact_executor(),
        catalog=catalog,
        predictive_receipt=predictive,
        C_capture=C_capture,
        batches=(C_batch,),
    )
    real = catalog.arm("real_forward_cross_block_edges")
    metrics = tuple(
        _causal_metric(
            source_site_id=artifact.claimed_source_site_id,
            target_event_id=artifact.edge.spec.target_event_id,
        )
        for artifact in real.edges
    )

    receipt = build_edge_sidecar_causal_receipt(
        request=request,
        edge_metrics=metrics,
        source_fingerprint_after=_SOURCE_FINGERPRINT,
        peak_resident_bytes=1,
    )

    assert receipt.passed is True
    assert receipt.C_row_identity_sha256 == C_capture.capture.row_identity_sha256
    assert receipt.parent_edge_catalog_sha256 == catalog.artifact_sha256
    assert receipt.parent_predictive_receipt_sha256 == predictive.receipt_sha256
    assert receipt.exact_forward_count == sum(
        item.exact_forward_count for item in metrics
    )
    assert receipt.backward_or_jvp_count == sum(
        item.backward_or_jvp_count for item in metrics
    )
    assert receipt.finite_intervention_forward_count == sum(
        item.finite_intervention_forward_count for item in metrics
    )

    with pytest.raises(ValueError, match="inventory differs"):
        build_edge_sidecar_causal_receipt(
            request=request,
            edge_metrics=tuple(reversed(metrics)),
            source_fingerprint_after=_SOURCE_FINGERPRINT,
            peak_resident_bytes=1,
        )
    with pytest.raises(ValueError, match="changed the exact source"):
        build_edge_sidecar_causal_receipt(
            request=request,
            edge_metrics=metrics,
            source_fingerprint_after="b" * 64,
            peak_resident_bytes=1,
        )


def _rebind_catalog_descendants(report: dict[str, object]) -> None:
    catalog = report["edge_catalog"]
    predictive = report["heldout_predictive_receipt"]
    assert isinstance(catalog, dict)
    assert isinstance(predictive, dict)
    _artifact_digest(catalog, "artifact_sha256")
    predictive["parent_edge_catalog_sha256"] = catalog["artifact_sha256"]
    _artifact_digest(predictive, "receipt_sha256")
    causal = report["heldout_causal_receipt"]
    if isinstance(causal, dict):
        causal["parent_edge_catalog_sha256"] = catalog["artifact_sha256"]
        causal["parent_predictive_receipt_sha256"] = predictive[
            "receipt_sha256"
        ]
        _artifact_digest(causal, "receipt_sha256")
    _report_digest(report)


def test_strict_loader_binds_disjoint_A_B_C_row_hashes_to_receipts() -> None:
    report = _valid_strict_report(causal=True)
    validate_exact_edge_sidecar_experiment_report(report)

    B_tamper = copy.deepcopy(report)
    B_capture = B_tamper["captures"]["B_predictive"]
    A_capture = B_tamper["captures"]["A_fit"]
    predictive = B_tamper["heldout_predictive_receipt"]
    B_capture["row_identity_sha256"] = A_capture["row_identity_sha256"]
    predictive["B_row_identity_sha256"] = A_capture["row_identity_sha256"]
    _artifact_digest(predictive, "receipt_sha256")
    causal = B_tamper["heldout_causal_receipt"]
    causal["parent_predictive_receipt_sha256"] = predictive["receipt_sha256"]
    _artifact_digest(causal, "receipt_sha256")
    _report_digest(B_tamper)
    with pytest.raises(ValueError, match="row identities must be disjoint"):
        validate_exact_edge_sidecar_experiment_report(B_tamper)

    C_tamper = copy.deepcopy(report)
    causal = C_tamper["heldout_causal_receipt"]
    causal["C_row_identity_sha256"] = "f" * 64
    _artifact_digest(causal, "receipt_sha256")
    _report_digest(C_tamper)
    with pytest.raises(ValueError, match="causal receipt lineage"):
        validate_exact_edge_sidecar_experiment_report(C_tamper)


def test_strict_loader_accepts_canonical_sorted_JSON_roundtrip() -> None:
    report = _valid_strict_report(causal=True)
    roundtripped = json.loads(json.dumps(report, sort_keys=True))

    assert validate_exact_edge_sidecar_experiment_report(roundtripped) == (
        roundtripped
    )


def test_strict_loader_recomputes_every_role_exact_and_native_gate() -> None:
    report = _valid_strict_report()
    role_gate = report["captures"]["A_fit"]["exact_gate"]
    role_gate["every_mutation_delta_byte_identical"] = False
    aggregate_gate = report["compilation"]["exact_39_event_gate"]
    aggregate_gate["role_gates"]["A_fit"] = copy.deepcopy(role_gate)
    _report_digest(report)

    with pytest.raises(ValueError, match="exact/native literal gate"):
        validate_exact_edge_sidecar_experiment_report(report)


def test_strict_loader_closes_all_added_analysis_costs() -> None:
    report = _valid_strict_report()
    validate_exact_edge_sidecar_experiment_report(report)
    report["analysis_cost"]["fit_design_MAC_estimate"] += 1
    _report_digest(report)

    with pytest.raises(ValueError, match="analysis cost ledger"):
        validate_exact_edge_sidecar_experiment_report(report)


def test_strict_loader_rejects_rehashed_dense_rank_and_tensor_hash_drift() -> None:
    rank_report = _valid_strict_report()
    rank_report["edge_catalog"]["arms"][0]["edges"][0][
        "realized_numerical_rank"
    ] = 641
    _rebind_catalog_descendants(rank_report)
    with pytest.raises(ValueError, match="dense edge metadata"):
        validate_exact_edge_sidecar_experiment_report(rank_report)

    hash_report = _valid_strict_report()
    hash_report["edge_catalog"]["arms"][0]["edges"][0][
        "fit_design_sha256"
    ] = "not-a-sha"
    _rebind_catalog_descendants(hash_report)
    with pytest.raises(ValueError, match="dense edge metadata"):
        validate_exact_edge_sidecar_experiment_report(hash_report)
