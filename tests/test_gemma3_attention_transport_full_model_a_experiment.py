import json

import torch

from fisher_graph.gemma3_attention_transport_full_model_a_experiment import (
    REPORT_SCHEMA,
    _BehaviorAccumulator,
    _report_sha256,
    check_full_model_cross_layer_A_experiment,
)


def test_chunked_behavior_metrics_close_on_identical_terminal_states() -> None:
    torch.manual_seed(12)
    states = torch.randn(2, 5, 4)
    input_ids = torch.randint(0, 11, (2, 5))
    tied_weight = torch.randn(11, 4)
    accumulator = _BehaviorAccumulator()
    accumulator.update(
        teacher_states=states,
        candidate_states=states.clone(),
        input_ids=input_ids,
        tied_weight=tied_weight,
        logit_softcap=30.0,
        row_chunk=3,
    )

    payload = accumulator.payload()
    assert payload["supervised_token_count"] == 8
    assert payload["delta_nll_per_token"] == 0.0
    assert abs(payload["teacher_to_candidate_kl_per_token"]) < 1.0e-7
    assert payload["top1_agreement"] == 1.0
    assert payload["logit_rmse"] == 0.0


def test_report_checker_authenticates_hash_and_firewall(tmp_path) -> None:
    payload = {
        "schema": REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
            "confirmation_claimed": False,
            "seam_parity_passed": True,
        },
        "candidate": {"artifact_sha256": "a" * 64},
        "gate_decision": {
            "passed": False,
            "failed_gates": ["held_block_relative_fisher_loss"],
        },
        "implementation_source_sha256s": {},
    }
    report = {**payload, "report_sha256": _report_sha256(payload)}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    checked = check_full_model_cross_layer_A_experiment(path)

    assert checked["A_screen_passed"] is False
    assert checked["failed_gates"] == ["held_block_relative_fisher_loss"]
    assert checked["B_opened"] is False
    assert checked["C_opened"] is False
