from __future__ import annotations

import copy
import json

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_v2_protocol as protocol,
)
from fisher_graph.token_loss_fisher_generator_innovation_adaptive_v2 import (
    AdaptiveGeneratorInnovationEligibilityReceipt,
    AdaptiveGeneratorInnovationV2Protocol,
)


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


def _signs(count: int) -> dict[str, dict[str, int]]:
    negative = count // 2
    return {
        channel: {
            "negative": negative,
            "zero": 0,
            "positive": count - negative,
        }
        for channel in protocol.GENERATOR_INNOVATION_V2_CHANNEL_ORDER
    }


def _quantiles(base: float) -> dict[str, dict[str, float]]:
    return {
        "real": {"q50": base, "q90": base + 1.0, "q99": base + 2.0},
        "imag": {
            "q50": base + 0.5,
            "q90": base + 1.5,
            "q99": base + 2.5,
        },
    }


def _raw_source(
    *,
    source_id: str,
    active_count: int,
    base: float,
    trace_index: int,
) -> dict[str, object]:
    counts = (1, 3, 12, active_count - 16)
    buckets: dict[str, object] = {}
    for bucket_index, (bucket_id, count) in enumerate(
        zip(
            protocol.GENERATOR_INNOVATION_V2_AGE_BUCKET_ORDER,
            counts,
            strict=True,
        )
    ):
        buckets[bucket_id] = {
            "active_count": count,
            "abs_raw_quantiles_by_channel": _quantiles(
                base + bucket_index
            ),
            "sign_counts_by_channel": _signs(count),
        }
    half_life = (
        None if source_id == "current_only" else int(source_id[2:])
    )
    return {
        "source_id": source_id,
        "half_life_active_positions": half_life,
        "active_count": active_count,
        "abs_raw_quantiles_by_channel": _quantiles(base),
        "sign_counts_by_channel": _signs(active_count),
        "age_buckets": buckets,
        "raw_trace_sha256": _sha(trace_index),
        "prior_kind": (
            "none_current_only"
            if source_id == "current_only"
            else "ew_prior_before_current_update"
        ),
        "whole_sequence_equals_two_chunks": True,
        "padding_updates_state": False,
    }


def _summaries() -> dict[str, object]:
    result: dict[str, object] = {}
    for example_index in range(16):
        example_id = f"{1000 + example_index:064x}"
        active_count = 20
        raw = {
            source_id: _raw_source(
                source_id=source_id,
                active_count=active_count,
                base=10.0 + source_index * 10.0 + example_index,
                trace_index=2000 + source_index * 100 + example_index,
            )
            for source_index, source_id in enumerate(
                protocol.GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER
            )
        }
        health: dict[str, object] = {}
        for candidate_index, candidate_id in enumerate(
            protocol.GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
        ):
            if candidate_id == "exact_v1_ew16_tau1":
                q90 = (0.98, 0.98)
                central = (0.10, 0.10)
            elif candidate_id == "ew64_scale_x2":
                q90 = (0.96, 0.70)
                central = (0.70, 0.70)
            else:
                q90 = (0.70, 0.75)
                central = (0.70, 0.65)
            health[candidate_id] = {
                "candidate_id": candidate_id,
                "active_count": active_count,
                "q90_absolute_bounded_by_channel": q90,
                "central_fraction_by_channel": central,
                "bounded_trace_sha256": _sha(
                    4000 + candidate_index * 100 + example_index
                ),
                "candidate_feature_receipt_sha256": _sha(
                    5000 + candidate_index * 100 + example_index
                ),
            }
        result[example_id] = {
            "example_id": example_id,
            "family_id": f"family-{example_index // 2}",
            "active_count": active_count,
            "top_mode_indices": (0, 1),
            "top_mode_norms": (9.0, 4.0),
            "parent_modal_trace_sha256": _sha(6000 + example_index),
            "raw_by_source": raw,
            "candidate_health_by_id": health,
            "audit": {
                "accepted_parent_only": True,
                "candidate_output_read": False,
                "compensation_target_read": False,
                "prompt_or_family_outcome_read": False,
                "raw_feature_rows_retained": False,
                "raw_modal_rows_retained": False,
                "target_blind": True,
                "token_gradient_read": False,
                "token_loss_read": False,
            },
        }
    return result


def _lineage() -> dict[str, str]:
    return {
        key: _sha(index + 7000)
        for index, key in enumerate(sorted(protocol._PRIOR_LINEAGE_FIELDS))
    }


def _scale_receipt() -> dict[str, object]:
    return protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=_summaries(),
        prior_lineage=_lineage(),
    )


def _source_hashes() -> dict[str, object]:
    return {
        "v2_scale_receipt_sha256": _sha(8000),
        "v2_scale_receipt_file_sha256": _sha(8001),
        "v1_plan_sha256": _sha(8002),
        "v1_plan_file_sha256": _sha(8003),
        "v1_development_report_sha256": _sha(8004),
        "v1_development_report_file_sha256": _sha(8005),
        "v1_panel_receipt_sha256": _sha(8006),
        "v1_panel_receipt_file_sha256": _sha(8007),
        "basis_sha256": _sha(8008),
        "collection_manifest_sha256": _sha(8009),
        "collection_membership_receipt_sha256": _sha(8010),
        "collection_role_input_file_sha256": _sha(8011),
    }


def test_frozen_candidate_and_simplicity_orders_are_exact() -> None:
    assert protocol.GENERATOR_INNOVATION_V2_CANDIDATE_ORDER == (
        "exact_v1_ew16_tau1",
        "current_only_scale_x0p5",
        "current_only_scale_x1",
        "current_only_scale_x2",
        "ew04_scale_x0p5",
        "ew04_scale_x1",
        "ew04_scale_x2",
        "ew16_scale_x0p5",
        "ew16_scale_x1",
        "ew16_scale_x2",
        "ew64_scale_x0p5",
        "ew64_scale_x1",
        "ew64_scale_x2",
    )
    assert set(protocol.GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER) == (
        set(protocol.GENERATOR_INNOVATION_V2_CANDIDATE_ORDER)
    )
    assert (
        protocol.GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER[:4]
        == (
            "exact_v1_ew16_tau1",
            "ew16_scale_x1",
            "ew16_scale_x2",
            "ew16_scale_x0p5",
        )
    )


def test_scale_receipt_build_validate_and_replay() -> None:
    summaries = _summaries()
    lineage = _lineage()
    receipt = (
        protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
            per_example_raw_summaries=summaries,
            prior_lineage=lineage,
        )
    )
    protocol.validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        receipt
    )
    assert receipt["schema"] == protocol.GENERATOR_INNOVATION_V2_SCALE_SCHEMA
    assert receipt["scale_by_source"]["current_only"][  # type: ignore[index]
        "prompt_balanced_median_absolute_raw_by_channel"
    ] == pytest.approx((17.5, 18.0))
    assert receipt["scale_by_source"]["ew16"][  # type: ignore[index]
        "prompt_balanced_median_absolute_raw_by_channel"
    ] == pytest.approx((37.5, 38.0))
    assert receipt["audit"]["target_blind"] is True  # type: ignore[index]
    assert receipt["audit"]["raw_rows_retained"] is False  # type: ignore[index]
    protocol.replay_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=summaries,
        prior_lineage=lineage,
        expected_receipt=receipt,
    )


def test_scale_receipt_survives_canonical_json_key_sorting() -> None:
    receipt = _scale_receipt()
    serialized = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    loaded = json.loads(serialized)
    protocol.validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        loaded
    )


def test_scale_receipt_rejects_target_read_or_summary_tamper() -> None:
    summaries = _summaries()
    first = next(iter(summaries))
    summaries[first]["audit"]["token_loss_read"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="not target blind"):
        protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
            per_example_raw_summaries=summaries,
            prior_lineage=_lineage(),
        )

    receipt = _scale_receipt()
    tampered = copy.deepcopy(receipt)
    tampered["scale_by_source"]["ew16"][  # type: ignore[index]
        "prompt_balanced_median_absolute_raw_by_channel"
    ] = (38.5, 38.0)
    with pytest.raises(ValueError, match="replay differs"):
        protocol.validate_gemma_iterative_generator_innovation_v2_scale_receipt(
            tampered
        )


def test_scale_receipt_accepts_null_quantiles_only_for_empty_age_bucket() -> None:
    summaries = _summaries()
    first = next(iter(summaries))
    source = summaries[first]["raw_by_source"]["current_only"]  # type: ignore[index]
    source["age_buckets"]["5_16"] = {  # type: ignore[index]
        "active_count": 16,
        "abs_raw_quantiles_by_channel": _quantiles(13.0),
        "sign_counts_by_channel": _signs(16),
    }
    source["age_buckets"]["17_plus"] = {  # type: ignore[index]
        "active_count": 0,
        "abs_raw_quantiles_by_channel": {
            channel: {"q50": None, "q90": None, "q99": None}
            for channel in protocol.GENERATOR_INNOVATION_V2_CHANNEL_ORDER
        },
        "sign_counts_by_channel": _signs(0),
    }
    protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=summaries,
        prior_lineage=_lineage(),
    )
    source["age_buckets"]["17_plus"][  # type: ignore[index]
        "abs_raw_quantiles_by_channel"
    ]["real"]["q50"] = 0.0
    with pytest.raises(ValueError, match="empty quantiles must be null"):
        protocol.build_gemma_iterative_generator_innovation_v2_scale_receipt(
            per_example_raw_summaries=summaries,
            prior_lineage=_lineage(),
        )


def test_candidate_specs_include_current_scale_and_exact_v1() -> None:
    receipt = _scale_receipt()
    specs = protocol.generator_innovation_v2_candidate_specs(receipt)
    assert tuple(row["candidate_id"] for row in specs) == (
        protocol.GENERATOR_INNOVATION_V2_CANDIDATE_ORDER
    )
    assert specs[0]["temperatures"] == (1.0, 1.0)
    assert specs[1]["temperatures"] == pytest.approx((8.75, 9.0))
    assert specs[2]["temperatures"] == pytest.approx((17.5, 18.0))
    ew16_x2 = next(
        row for row in specs if row["candidate_id"] == "ew16_scale_x2"
    )
    assert ew16_x2["half_life_active_positions"] == 16
    assert ew16_x2["temperatures"] == pytest.approx((75.0, 76.0))


def test_candidate_plan_binds_analyzer_protocol_eligibility_and_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scale = _scale_receipt()
    monkeypatch.setattr(
        protocol,
        "_authenticate_candidate_sources",
        lambda **_kwargs: _source_hashes(),
    )
    dummy = {"unused": True}
    plan = protocol.build_gemma_iterative_generator_innovation_v2_candidate_plan(
        scale_receipt=scale,
        scale_receipt_file_sha256=_sha(9000),
        v1_plan=dummy,
        v1_plan_file_sha256=_sha(9001),
        v1_development_report=dummy,
        v1_development_report_file_sha256=_sha(9002),
        v1_panel_receipt=dummy,
        v1_panel_receipt_file_sha256=_sha(9003),
    )
    protocol.validate_gemma_iterative_generator_innovation_v2_candidate_plan(
        plan
    )
    bank = plan["candidate_bank"]
    assert bank["total_coordinate_count"] == 28
    assert bank["static_reference_candidate_id"] == "exact_v1_ew16_tau1"
    nested = plan["nested_evaluator"]
    analyzer = AdaptiveGeneratorInnovationV2Protocol.from_dict(
        nested["adaptive_analyzer_protocol"]
    )
    eligibility = AdaptiveGeneratorInnovationEligibilityReceipt.from_dict(
        nested["activation_only_eligibility_receipt"]
    )
    assert analyzer.candidate_simplicity_order == (
        protocol.GENERATOR_INNOVATION_V2_CANDIDATE_SIMPLICITY_ORDER
    )
    assert analyzer.static_reference_candidate_id == "exact_v1_ew16_tau1"
    assert "exact_v1_ew16_tau1" not in eligibility.eligible_candidate_ids
    assert "ew64_scale_x2" not in eligibility.eligible_candidate_ids
    assert nested["arms"]["scaled_l16"] == (
        "ew16_scale_x1",
        "ew16_scale_x2",
        "ew16_scale_x0p5",
    )
    assert plan["claim_boundary"]["finite_displacement_authorized"] is False
    assert plan["claim_boundary"]["compression_claim_authorized"] is False
    protocol.replay_gemma_iterative_generator_innovation_v2_candidate_plan(
        scale_receipt=scale,
        scale_receipt_file_sha256=_sha(9000),
        v1_plan=dummy,
        v1_plan_file_sha256=_sha(9001),
        v1_development_report=dummy,
        v1_development_report_file_sha256=_sha(9002),
        v1_panel_receipt=dummy,
        v1_panel_receipt_file_sha256=_sha(9003),
        expected_plan=plan,
    )


def test_candidate_plan_rejects_widened_claim_even_with_rehashed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol,
        "_authenticate_candidate_sources",
        lambda **_kwargs: _source_hashes(),
    )
    dummy = {"unused": True}
    plan = protocol.build_gemma_iterative_generator_innovation_v2_candidate_plan(
        scale_receipt=_scale_receipt(),
        scale_receipt_file_sha256=_sha(9100),
        v1_plan=dummy,
        v1_plan_file_sha256=_sha(9101),
        v1_development_report=dummy,
        v1_development_report_file_sha256=_sha(9102),
        v1_panel_receipt=dummy,
        v1_panel_receipt_file_sha256=_sha(9103),
    )
    tampered = copy.deepcopy(plan)
    tampered["claim_boundary"]["provider_compilation_authorized"] = True
    payload = {
        key: value for key, value in tampered.items() if key != "plan_sha256"
    }
    tampered["plan_sha256"] = protocol._sha256(
        protocol._PLAN_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="claim boundary"):
        protocol.validate_gemma_iterative_generator_innovation_v2_candidate_plan(
            tampered
        )


def test_candidate_plan_rejects_rehashed_candidate_spec_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol,
        "_authenticate_candidate_sources",
        lambda **_kwargs: _source_hashes(),
    )
    dummy = {"unused": True}
    plan = protocol.build_gemma_iterative_generator_innovation_v2_candidate_plan(
        scale_receipt=_scale_receipt(),
        scale_receipt_file_sha256=_sha(9200),
        v1_plan=dummy,
        v1_plan_file_sha256=_sha(9201),
        v1_development_report=dummy,
        v1_development_report_file_sha256=_sha(9202),
        v1_panel_receipt=dummy,
        v1_panel_receipt_file_sha256=_sha(9203),
    )
    tampered = copy.deepcopy(plan)
    tampered["candidate_bank"]["candidate_specs"][2][
        "temperatures"
    ] = (999.0, 999.0)
    payload = {
        key: value for key, value in tampered.items() if key != "plan_sha256"
    }
    tampered["plan_sha256"] = protocol._sha256(
        protocol._PLAN_DOMAIN,
        payload,
    )
    with pytest.raises(ValueError, match="adaptive analyzer binding"):
        protocol.validate_gemma_iterative_generator_innovation_v2_candidate_plan(
            tampered
        )
