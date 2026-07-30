from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

import fisher_graph.gemma3_l3_l4_progressive_compilation as progressive_binding
from fisher_graph.compiler.progressive import (
    DevelopmentCorpus,
    ProgressiveBehavioralFidelity,
    ProgressiveFidelity,
    ProgressiveResourceBudget,
    ProgressiveResourceFootprint,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from fisher_graph.gemma3_l3_l4_progressive_compilation import (
    current_gemma3_l3_l4_progressive_seed,
    gemma3_l3_l4_legacy_progressive_binding_metadata,
    gemma3_l3_l4_progressive_fidelity_targets,
    make_gemma3_l3_l4_progressive_protocol,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _corpus() -> DevelopmentCorpus:
    return DevelopmentCorpus(
        corpus_id="gemma-progressive-a-v1",
        fit_manifest_sha256=_sha(1),
        selection_manifest_sha256=_sha(2),
        guard_manifest_sha256=_sha(3),
        fit_example_count=48,
        selection_example_count=24,
        guard_example_count=24,
        fit_family_ids=("fit-physics", "fit-reasoning"),
        selection_family_ids=("selection-code", "selection-prose"),
        guard_family_ids=("guard-math", "guard-science"),
    )


def _budget() -> ProgressiveResourceBudget:
    return ProgressiveResourceBudget(
        parameter_scope="gemma3.full-model.parameters",
        compute_scope="gemma3.full-model.macs-per-token",
        runtime_id="torch.cpu",
        runtime_dtype="float32",
        sequence_scope_sha256=_sha(20),
        source_learned_parameters=1000,
        source_runtime_parameter_bytes=2000,
        source_logical_macs_per_token=3000,
        max_total_parameter_fraction=0.8,
        max_total_parameter_byte_fraction=0.8,
        max_total_mac_fraction=0.8,
        max_retained_source_parameter_fraction=0.2,
        max_retained_source_parameter_byte_fraction=0.2,
        max_retained_source_mac_fraction=0.2,
    )


def _resources() -> ProgressiveResourceFootprint:
    return ProgressiveResourceFootprint(
        candidate_execution_sha256=(
            "911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9"
        ),
        accounting_artifact_sha256=_sha(21),
        parameter_scope="gemma3.full-model.parameters",
        compute_scope="gemma3.full-model.macs-per-token",
        runtime_id="torch.cpu",
        runtime_dtype="float32",
        sequence_scope_sha256=_sha(20),
        compiled_learned_parameters=500,
        retained_source_learned_parameters=100,
        support_learned_parameters=10,
        compiled_runtime_parameter_bytes=1000,
        retained_source_runtime_parameter_bytes=200,
        support_runtime_parameter_bytes=20,
        compiled_logical_macs_per_token=1800,
        retained_source_logical_macs_per_token=300,
        support_logical_macs_per_token=100,
        cost_complete=True,
    )


def test_factory_binds_exact_legacy_seed_and_forbids_b_manifest() -> None:
    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    legacy_metadata = legacy.metadata()
    resources = _resources()
    progressive = make_gemma3_l3_l4_progressive_protocol(
        corpus=_corpus(),
        seed_runtime_binding_sha256=legacy_metadata[
            "runtime_binding_contract"
        ]["artifact_sha256"],
        fit_panel_binding_sha256=_sha(30),
        selection_panel_binding_sha256=_sha(31),
        guard_preclaim_binding_sha256=_sha(32),
        resource_budget=_budget(),
        seed_resources=resources,
    )
    seed = current_gemma3_l3_l4_progressive_seed(
        resources=resources,
        runtime_binding_sha256=(
            progressive.seed_runtime_binding_sha256
        ),
    )

    assert progressive.source_model_sha256 == (
        legacy_metadata["model"]["source_model_sha256"]
    )
    assert progressive.seed_candidate_artifact_sha256 == (
        legacy_metadata["graph_candidate"]["logical_artifact_sha256"]
    )
    assert progressive.seed_candidate_execution_sha256 == (
        legacy_metadata["graph_candidate"][
            "factorized_refit_execution_sha256"
        ]
    )
    assert progressive.seed_runtime_binding_sha256 == (
        legacy_metadata["runtime_binding_contract"]["artifact_sha256"]
    )
    assert progressive.seed_resource_receipt_sha256 == (
        resources.receipt_sha256
    )
    assert (
        legacy_metadata["graph_candidate"]["deployment_plan_sha256"]
        in progressive.seed_lineage_sha256s
    )
    assert (
        legacy_metadata["prompt_blind_basis"]["logical_payload_sha256"]
        in progressive.seed_lineage_sha256s
    )
    assert progressive.forbidden_assessment_manifest_sha256s == (
        legacy_metadata["corpus"]["calibration_b_manifest"][
            "artifact_sha256"
        ],
    )
    assert seed.artifact_sha256 == (
        progressive.seed_candidate_artifact_sha256
    )
    assert seed.execution_sha256 == (
        progressive.seed_candidate_execution_sha256
    )
    assert seed.runtime_binding_sha256 == (
        progressive.seed_runtime_binding_sha256
    )


def test_factory_rejects_a_runtime_binding_for_another_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrupt_metadata = deepcopy(
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol().metadata()
    )
    corrupt_metadata["runtime_binding_contract"][
        "adapter_execution_fingerprint"
    ] = _sha(999)

    class CorruptLegacyProtocol:
        def validate_integrity(self) -> None:
            return None

        def metadata(self):
            return corrupt_metadata

        artifact_sha256 = corrupt_metadata["artifact_sha256"]

    monkeypatch.setattr(
        progressive_binding,
        "default_gemma3_l3_l4_graph_organized_svd_shadow_protocol",
        CorruptLegacyProtocol,
    )

    with pytest.raises(
        ValueError,
        match="runtime binding does not match",
    ):
        make_gemma3_l3_l4_progressive_protocol(
            corpus=_corpus(),
            seed_runtime_binding_sha256=_sha(40),
            fit_panel_binding_sha256=_sha(30),
            selection_panel_binding_sha256=_sha(31),
            guard_preclaim_binding_sha256=_sha(32),
            resource_budget=_budget(),
            seed_resources=_resources(),
        )


def test_factory_rejects_resource_accounting_for_another_execution() -> None:
    resources = replace(
        _resources(),
        candidate_execution_sha256=_sha(999),
    )

    with pytest.raises(
        ValueError,
        match="resource accounting does not bind",
    ):
        make_gemma3_l3_l4_progressive_protocol(
            corpus=_corpus(),
            seed_runtime_binding_sha256=_sha(40),
            fit_panel_binding_sha256=_sha(30),
            selection_panel_binding_sha256=_sha(31),
            guard_preclaim_binding_sha256=_sha(32),
            resource_budget=_budget(),
            seed_resources=resources,
        )


def test_factory_reuses_frozen_behavioral_and_boundary_gates() -> None:
    legacy = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    metadata = legacy.metadata()

    targets = gemma3_l3_l4_progressive_fidelity_targets()

    assert targets.candidate_behavior.absolute_delta_nll_per_token_max == (
        metadata["behavioral_gates"][
            "absolute_delta_nll_per_token_max"
        ]
    )
    assert targets.candidate_behavior.top1_agreement_to_source_min == (
        metadata["behavioral_gates"][
            "top1_agreement_to_source_min"
        ]
    )
    assert targets.boundary_relative_error_max == (
        metadata["boundary_gates"][
            "pooled_target_modal_relative_error_max"
        ]
    )
    assert targets.boundary_cosine_min == (
        metadata["boundary_gates"][
            "pooled_target_modal_cosine_min"
        ]
    )
    assert targets.projection_full_width_relative_error_max == (
        metadata["projection_capacity_gates"][
            "pooled_full_width_delta_relative_error_max"
        ]
    )
    assert targets.projection_full_width_cosine_min == (
        metadata["projection_capacity_gates"][
            "pooled_full_width_delta_cosine_min"
        ]
    )
    assert (
        targets.carrier_oracle_behavior.top1_agreement_to_source_min
        == metadata["carrier_completeness_gates"][
            "exact_full_width_x4_on_clamped_reference_carrier"
        ]["top1_agreement_to_source_min"]
    )


def test_operator_gate_can_be_frozen_more_strictly() -> None:
    targets = gemma3_l3_l4_progressive_fidelity_targets(
        operator_nrmse_max=0.125,
    )
    assert targets.operator_nrmse_max == 0.125


def test_operator_default_uses_the_full_width_projection_gate() -> None:
    targets = gemma3_l3_l4_progressive_fidelity_targets()
    assert targets.operator_nrmse_max == 0.05


def test_immutable_projection_and_carrier_failures_remain_diagnostic() -> None:
    targets = gemma3_l3_l4_progressive_fidelity_targets()
    passing_behavior = ProgressiveBehavioralFidelity(
        absolute_delta_nll_per_token=0.0,
        source_to_candidate_kl_per_token=0.0,
        top1_agreement_to_source=1.0,
        per_prompt_p90_absolute_delta_nll_per_token=0.0,
        per_prompt_p10_top1_agreement_to_source=1.0,
    )
    failing_carrier = ProgressiveBehavioralFidelity(
        absolute_delta_nll_per_token=2.0121,
        source_to_candidate_kl_per_token=2.5847,
        top1_agreement_to_source=0.4651,
        per_prompt_p90_absolute_delta_nll_per_token=2.0121,
        per_prompt_p10_top1_agreement_to_source=0.4651,
    )
    known_failure = ProgressiveFidelity(
        candidate_behavior=passing_behavior,
        projection_oracle_behavior=passing_behavior,
        carrier_oracle_behavior=failing_carrier,
        operator_nrmse=0.9741,
        boundary_relative_error=0.0,
        boundary_cosine=1.0,
        valid_target_coverage=1.0,
        worst_family_boundary_relative_error=0.0,
        worst_family_boundary_cosine=1.0,
        minimum_family_source_modal_signal_l2_norm=1.0,
        projection_full_width_relative_error=0.9741,
        projection_full_width_cosine=0.2261,
        worst_family_projection_relative_error=0.9741,
        worst_family_projection_cosine=0.2261,
        minimum_family_source_full_width_signal_l2_norm=1.0,
    )

    assert targets.passes(known_failure) is True
    ratios = targets.normalized_ratios(known_failure)
    assert ratios["projection.full_width_relative_error"] > 1.0
    assert (
        ratios["carrier_oracle_behavior.absolute_delta_nll_per_token"]
        > 1.0
    )
    assert set(targets.execution_fidelity_ratios(known_failure)) == {
        "candidate_behavior.absolute_delta_nll_per_token",
        "candidate_behavior.per_prompt_p10_top1_agreement_to_source",
        "candidate_behavior.per_prompt_p90_absolute_delta_nll_per_token",
        "candidate_behavior.source_to_candidate_kl_per_token",
        "candidate_behavior.top1_agreement_to_source",
    }
    assert (
        "projection.full_width_relative_error"
        in targets.structural_diagnostic_ratios(known_failure)
    )


def test_development_corpus_cannot_reuse_registered_b_manifest() -> None:
    metadata = gemma3_l3_l4_legacy_progressive_binding_metadata()
    b_manifest = metadata["calibration_b_manifest_sha256"]
    corpus = replace(_corpus(), guard_manifest_sha256=b_manifest)

    with pytest.raises(ValueError, match="assessment manifest"):
        make_gemma3_l3_l4_progressive_protocol(
            corpus=corpus,
            seed_runtime_binding_sha256=_sha(40),
            fit_panel_binding_sha256=_sha(30),
            selection_panel_binding_sha256=_sha(31),
            guard_preclaim_binding_sha256=_sha(32),
            resource_budget=_budget(),
            seed_resources=_resources(),
        )


def test_binding_explicitly_requires_a_future_candidate_bound_handoff() -> None:
    metadata = gemma3_l3_l4_legacy_progressive_binding_metadata()

    assert metadata["calibration_b_access"] == (
        "forbidden_identity_only"
    )
    assert metadata[
        "legacy_one_shot_accepts_new_progressive_winner"
    ] is False
    assert metadata["required_next_boundary"] == (
        "candidate_bound_shadow_protocol_and_runtime_v2"
    )
    assert metadata["legacy_shadow_protocol_sha256"] == (
        "7a79087fe4ea90b383bd98f787bede4131c457533a9eef91e6903b2e9c5ea3c8"
    )
