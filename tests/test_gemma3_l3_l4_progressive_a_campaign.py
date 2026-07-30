from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.gemma3_l3_l4_progressive_a_campaign import (
    _adaptive_parent_lineage,
    _evaluation_qualification,
    _h4_conditioning_contract,
    build_gemma3_l3_l4_progressive_resource_envelope,
    gemma3_l3_l4_guard_preclaim_binding_sha256,
    materialize_gemma3_l3_l4_progressive_panel,
)
from fisher_graph.gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveARolePreclaimView,
    Gemma3L3L4ProgressiveARolePrompts,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


class _Tokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, prompts, **_kwargs):
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            rows.append(
                [
                    2,
                    int.from_bytes(digest[:2], "big") + 3,
                    int.from_bytes(digest[2:4], "big") + 3,
                    1,
                ]
            )
        return {
            "input_ids": torch.tensor(rows, dtype=torch.int64),
            "attention_mask": torch.ones(
                (len(rows), 4),
                dtype=torch.int64,
            ),
        }


def _role(
    role: str,
) -> tuple[
    Gemma3L3L4ProgressiveARolePrompts,
    Gemma3L3L4ProgressiveARolePreclaimView,
]:
    prompts = (
        f"{role} alpha prompt with enough tokens.",
        f"{role} beta prompt with enough tokens.",
    )
    prompt_sha256s = tuple(
        gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
        for prompt in prompts
    )
    families = (f"{role}.family-a", f"{role}.family-b")
    opened = Gemma3L3L4ProgressiveARolePrompts(
        corpus_id="progressive-a-test",
        profile="pilot",
        role=role,  # type: ignore[arg-type]
        prompts=prompts,
        family_ids=families,
        source_file_sha256=_sha(10),
    )
    view = Gemma3L3L4ProgressiveARolePreclaimView(
        role=role,  # type: ignore[arg-type]
        manifest_sha256=_sha(20),
        role_input_file_sha256=_sha(10),
        example_count=2,
        family_ids=tuple(sorted(families)),
        ordered_prompt_sha256s=prompt_sha256s,
        ordered_family_ids=families,
    )
    return opened, view


def test_panel_uses_frozen_prompt_hashes_as_cross_role_example_ids() -> None:
    opened, view = _role("calibration_a_selection")
    panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=_Tokenizer(),
        role_input=opened,
        view=view,
        max_length=32,
        device=torch.device("cpu"),
        forbidden_manifest_sha256s=(_sha(99),),
    )

    assert tuple(
        example.example_id for example in panel.examples
    ) == view.ordered_prompt_sha256s
    assert panel.family_ids == view.family_ids
    assert panel.model_inputs_receipt_sha256 != (
        panel.membership_receipt_sha256
    )


def test_guard_preclaim_binds_hash_membership_without_prompt_payload() -> None:
    _opened, view = _role("calibration_a_guard")

    first = gemma3_l3_l4_guard_preclaim_binding_sha256(
        corpus_artifact_sha256=_sha(30),
        tokenizer_contract_sha256=_sha(31),
        view=view,
    )
    second = gemma3_l3_l4_guard_preclaim_binding_sha256(
        corpus_artifact_sha256=_sha(30),
        tokenizer_contract_sha256=_sha(32),
        view=view,
    )

    assert len(first) == 64
    assert first != second


def test_resource_envelope_charges_carrier_bridge_and_two_heads() -> None:
    arguments = {
        "candidate_execution_sha256": _sha(1),
        "sequence_scope_sha256": _sha(2),
        "raw_model_resources": (1000, 4000, 1000),
        "factorized_model_resources": (700, 2800, 700),
        "bridge_float_count": 20,
        "bridge_integer_count": 2,
        "bridge_runtime_bytes": 168,
        "bridge_logical_macs_per_token": 15,
        "residual_width": 10,
        "source_modes": 4,
        "head_rank": 3,
        "max_residual_directions": 2,
        "lag_count": 5,
    }
    seed, budget = build_gemma3_l3_l4_progressive_resource_envelope(
        **arguments,
    )
    _conditioned_seed, conditioned_budget = (
        build_gemma3_l3_l4_progressive_resource_envelope(
            **arguments,
            h4_conditioning=(
                "l3_source_modes_plus_realized_h4_decoder_modes_v1"
            ),
        )
    )

    # Two heads each contain R * (W + L*S) = 2 * (10 + 5*4) = 60
    # floating scalars.
    assert seed.total_learned_parameters == 720
    assert seed.total_runtime_parameter_bytes == 2968
    assert seed.total_logical_macs_per_token == 715
    assert budget.source_learned_parameters == 1000
    assert budget.max_total_parameter_fraction > 0.84
    assert budget.max_total_parameter_fraction < 0.841
    assert budget.allows(seed) is False
    assert budget.violations(seed) == ("cost_incomplete",)
    # Rank is min(3, 2, 10) = 2. The conditioned H4 head adds R^2 = 4
    # floats/parameters, 32 bytes, and W*R + R^2 = 24 MACs/token.
    assert conditioned_budget.max_total_parameter_fraction > 0.844
    assert conditioned_budget.max_total_parameter_fraction < 0.845
    assert (
        conditioned_budget.max_total_parameter_byte_fraction
        > budget.max_total_parameter_byte_fraction
    )
    assert (
        conditioned_budget.max_total_mac_fraction
        > budget.max_total_mac_fraction
    )


def test_conditioning_contract_reports_effective_state_rank() -> None:
    contract = _h4_conditioning_contract(
        h4_conditioning=(
            "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        ),
        residual_width=10,
        head_rank=3,
        max_residual_directions=2,
    )

    assert contract["effective_state_rank"] == 2
    assert contract["additional_state_kernel_parameters"] == 4


def test_conditioning_parent_lineage_enforces_fixed_controls(
    tmp_path,
) -> None:
    controls = {
        "corpus_artifact_sha256": _sha(80),
        "maximum_sequence_length": 256,
        "device": "cpu",
        "dtype": "float32",
        "head_rank": 8,
        "lag_count": 32,
        "ridge": 1.0e-6,
        "h4_fit_objective": "candidate_nll_vjp_metric_ridge_v1",
        "maximum_residual_directions": 8,
        "maximum_iterations": 3,
        "candidate_schedule": "forced_x4_then_remeasure_then_h4",
        "guard_policy": "deferred_after_frozen_selection",
    }
    report = {
        "campaign_spec": controls,
        "campaign_spec_sha256": _sha(81),
        "protocol_sha256": _sha(82),
        "report_sha256": _sha(83),
        "transcript_sha256": _sha(84),
        "safety": {
            "calibration_b_opened": False,
            "guard_opened": False,
            "guard_consumed": False,
        },
    }
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    lineage = _adaptive_parent_lineage(
        path,
        expected_fixed_controls=controls,
        expected_parent_conditioning="l3_source_modes",
    )

    assert lineage is not None
    assert lineage["campaign_spec_sha256"] == _sha(81)
    assert lineage["parent_h4_conditioning"] == "l3_source_modes"
    assert lineage["fixed_control_snapshot"] == controls

    changed = {
        **report,
        "campaign_spec": {
            **controls,
            "head_rank": 16,
        },
    }
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed controls differ"):
        _adaptive_parent_lineage(
            path,
            expected_fixed_controls=controls,
            expected_parent_conditioning="l3_source_modes",
        )


def test_qualification_report_uses_candidate_evaluation_role_field() -> None:
    class _Targets:
        def execution_fidelity_ratios(self, _fidelity):
            return {"candidate_behavior.test": 0.5}

        def structural_diagnostic_ratios(self, _fidelity):
            return {"projection.test": 2.0}

        def passes_execution_fidelity(self, _fidelity):
            return True

    evaluation = SimpleNamespace(
        candidate_receipt_sha256=_sha(70),
        receipt_sha256=_sha(71),
        development_role="calibration_a_selection",
        fidelity=object(),
    )
    report = _evaluation_qualification(
        protocol=SimpleNamespace(fidelity_targets=_Targets()),
        evaluation=evaluation,
    )

    assert report["role"] == "calibration_a_selection"
    assert report["execution_fidelity_passed"] is True
    assert report["structural_failed_axes"] == ("projection.test",)
