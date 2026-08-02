from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from fisher_graph.token_loss_fisher import (
    build_token_loss_fisher_prompt_record,
)
from fisher_graph.token_loss_fisher_generator_innovation_adaptive_v2 import (
    AdaptiveGeneratorInnovationCandidateSpec,
    AdaptiveGeneratorInnovationEligibilityReceipt,
    AdaptiveGeneratorInnovationPortfolioSpec,
    AdaptiveGeneratorInnovationV2Protocol,
    build_generator_innovation_adaptive_v2_report,
    replay_generator_innovation_adaptive_v2_report,
    validate_generator_innovation_adaptive_v2_report,
)


_NAMES = (
    "generator_real_shared",
    "generator_imag_shared",
    "generator_real_innovation",
    "generator_imag_innovation",
)
_LEGACY_NAMES = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
)
_CANDIDATES = (
    "v1_l16_tau1",
    "scaled_l16_a0p5",
    "scaled_l16_a1",
    "temporal_l4_a1",
    "current_only",
)
_BASIS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _protocol() -> AdaptiveGeneratorInnovationV2Protocol:
    return AdaptiveGeneratorInnovationV2Protocol(
        candidate_specs=tuple(
            AdaptiveGeneratorInnovationCandidateSpec(
                candidate_id=name,
                family=(
                    "v1"
                    if name.startswith("v1")
                    else (
                        "scaled_l16"
                        if name.startswith("scaled")
                        else (
                            "full_temporal"
                            if name.startswith("temporal")
                            else "current_only"
                        )
                    )
                ),
                metadata=(("collector_owned", True),),
            )
            for name in _CANDIDATES
        ),
        candidate_simplicity_order=_CANDIDATES,
        portfolio_specs=(
            AdaptiveGeneratorInnovationPortfolioSpec(
                portfolio_id="scaled_l16",
                candidate_ids=_CANDIDATES[:3],
            ),
            AdaptiveGeneratorInnovationPortfolioSpec(
                portfolio_id="full_temporal",
                candidate_ids=_CANDIDATES[:4],
            ),
            AdaptiveGeneratorInnovationPortfolioSpec(
                portfolio_id="current_only",
                candidate_ids=(_CANDIDATES[4],),
            ),
        ),
        static_reference_candidate_id=_CANDIDATES[0],
        v1_candidate_id=_CANDIDATES[0],
        required_prompts_per_family=2,
    )


def _eligibility(
    protocol: AdaptiveGeneratorInnovationV2Protocol,
) -> AdaptiveGeneratorInnovationEligibilityReceipt:
    # v1 and current-only remain scored controls, but are not portfolio-eligible.
    eligible = (_CANDIDATES[1], _CANDIDATES[2], _CANDIDATES[3])
    return AdaptiveGeneratorInnovationEligibilityReceipt(
        protocol_sha256=protocol.protocol_sha256,
        scale_receipt_sha256=_digest("activation-only-scale"),
        eligible_candidate_ids=eligible,
        feature_health_receipt_sha256_by_candidate=tuple(
            (name, _digest(f"feature-health:{name}"))
            for name in _CANDIDATES
        ),
    )


def _bank() -> tuple[dict[str, tuple[object, ...]], tuple[object, ...]]:
    generator = torch.Generator().manual_seed(811)
    result: dict[str, list[object]] = {
        name: [] for name in _CANDIDATES
    }
    legacy_records = []
    strengths = {
        _CANDIDATES[0]: 0.15,
        _CANDIDATES[1]: 0.75,
        _CANDIDATES[2]: 1.0,
        _CANDIDATES[3]: 0.55,
        _CANDIDATES[4]: 0.05,
    }
    for index in range(16):
        count = 48
        shared = torch.randn(
            count,
            2,
            generator=generator,
            dtype=torch.float64,
        )
        latent = torch.randn(
            count,
            2,
            generator=generator,
            dtype=torch.float64,
        )
        noise = torch.randn(
            count,
            generator=generator,
            dtype=torch.float64,
        )
        target = (
            shared
            @ torch.tensor((0.025, -0.02), dtype=torch.float64)
            + latent
            @ torch.tensor((0.04, -0.035), dtype=torch.float64)
            + 0.01 * noise
        )
        legacy_records.append(
            build_token_loss_fisher_prompt_record(
                example_id=f"example-{index:02d}",
                family_id=f"family-{index % 8}",
                coordinate_names=_LEGACY_NAMES,
                token_scores=torch.randn(
                    count,
                    6,
                    generator=generator,
                    dtype=torch.float64,
                ),
                compensation_target=target,
            )
        )
        for candidate_index, candidate_id in enumerate(_CANDIDATES):
            local_noise = torch.randn(
                count,
                2,
                generator=generator,
                dtype=torch.float64,
            )
            innovation = (
                strengths[candidate_id] * latent
                + (1.0 - strengths[candidate_id])
                * local_noise
                + 0.002 * candidate_index
            )
            scores = torch.cat((shared, innovation), dim=1)
            result[candidate_id].append(
                build_token_loss_fisher_prompt_record(
                    example_id=f"example-{index:02d}",
                    family_id=f"family-{index % 8}",
                    coordinate_names=_NAMES,
                    token_scores=scores,
                    compensation_target=target,
                )
            )
    return (
        {name: tuple(rows) for name, rows in result.items()},
        tuple(legacy_records),
    )


def test_adaptive_bank_scores_every_arm_and_exactly_replays() -> None:
    protocol = _protocol()
    eligibility = _eligibility(protocol)
    assert protocol.ordered_fit_candidate_ids == (
        protocol.static_candidate_id,
        *(
            f"{candidate_id}__conditional_ridge_{ridge.replace('.', 'p')}"
            for ridge in protocol.ridge_simplicity_order
            for candidate_id in protocol.candidate_simplicity_order
        ),
    )
    bank, legacy = _bank()
    report = build_generator_innovation_adaptive_v2_report(
        bank,
        legacy_records=legacy,
        fixed_basis=_BASIS,
        protocol=protocol,
        eligibility=eligibility,
    )
    validate_generator_innovation_adaptive_v2_report(report)
    assert replay_generator_innovation_adaptive_v2_report(report) == report
    assert report["ordered_candidate_ids"] == _CANDIDATES
    assert len(report["ordered_fit_candidate_ids"]) == 16
    assert len(report["folds"]) == 8
    assert set(report["metrics"]["portfolio_metrics"]) == {
        "scaled_l16",
        "full_temporal",
        "current_only",
    }
    assert report["audit"]["token_used_as_independent_split_unit"] is False
    assert report["decision"]["held_family_used_for_selection"] is False

    allowed = set(eligibility.eligible_candidate_ids)
    for fold in report["folds"]:
        assert fold["held_family_id"] not in fold["train_family_ids"]
        for portfolio in fold["portfolio_selections"]:
            selection = portfolio["selection"]
            assert selection["ordered_candidate_ids"] == (
                protocol.static_candidate_id,
                *(
                    (
                        f"{candidate_id}__conditional_ridge_"
                        f"{ridge.replace('.', 'p')}"
                    )
                    for ridge in protocol.ridge_simplicity_order
                    for candidate_id in portfolio[
                        "eligible_feature_candidate_ids"
                    ]
                ),
            )
            if not selection["selected_static"]:
                assert selection["selected_feature_candidate_id"] in allowed
        v1 = next(
            row
            for row in fold["variant_ridge_selections"]
            if row["feature_candidate_id"] == protocol.v1_candidate_id
        )
        assert v1["selection"]["scope_id"].startswith("feature:")


def test_outer_held_candidate_rows_cannot_change_inner_selection() -> None:
    protocol = _protocol()
    eligibility = _eligibility(protocol)
    bank, legacy = _bank()
    first = build_generator_innovation_adaptive_v2_report(
        bank,
        legacy_records=legacy,
        fixed_basis=_BASIS,
        protocol=protocol,
        eligibility=eligibility,
    )
    changed = {name: list(rows) for name, rows in bank.items()}
    candidate = _CANDIDATES[2]
    source = changed[candidate][0]
    # Recreate an aligned record with the exact shared columns but unrelated
    # held-family conditional columns. Its target receipt must remain exact.
    # Prompt records do not retain token rows, so use the original static
    # sufficient statistics and alter only conditional sufficient statistics.
    payload = source.to_dict()
    payload["fisher_second_moment"] = [
        list(row) for row in payload["fisher_second_moment"]
    ]
    payload["target_cross_moment"] = list(payload["target_cross_moment"])
    payload["mean_score"] = list(payload["mean_score"])
    payload["fisher_second_moment"][2][2] += 0.25
    payload["fisher_second_moment"][3][3] += 0.25
    payload["target_cross_moment"][2] = 0.0
    payload["target_cross_moment"][3] = 0.0
    payload["mean_score"][2] = 0.0
    payload["mean_score"][3] = 0.0
    del payload["prompt_record_sha256"]
    changed[candidate][0] = type(source)(**payload)
    second = build_generator_innovation_adaptive_v2_report(
        {name: tuple(rows) for name, rows in changed.items()},
        legacy_records=legacy,
        fixed_basis=_BASIS,
        protocol=protocol,
        eligibility=eligibility,
    )
    first_fold = next(
        fold for fold in first["folds"]
        if fold["held_family_id"] == "family-0"
    )
    second_fold = next(
        fold for fold in second["folds"]
        if fold["held_family_id"] == "family-0"
    )
    assert tuple(
        row["selection"]["selected_fit_candidate_id"]
        for row in first_fold["portfolio_selections"]
    ) == tuple(
        row["selection"]["selected_fit_candidate_id"]
        for row in second_fold["portfolio_selections"]
    )


def test_alignment_eligibility_and_report_tampering_fail_closed() -> None:
    protocol = _protocol()
    eligibility = _eligibility(protocol)
    bank, legacy = _bank()

    bad_eligibility = AdaptiveGeneratorInnovationEligibilityReceipt(
        protocol_sha256=protocol.protocol_sha256,
        scale_receipt_sha256=_digest("activation-only-scale"),
        eligible_candidate_ids=tuple(reversed(eligibility.eligible_candidate_ids)),
        feature_health_receipt_sha256_by_candidate=(
            eligibility.feature_health_receipt_sha256_by_candidate
        ),
    )
    with pytest.raises(ValueError, match="eligibility"):
        build_generator_innovation_adaptive_v2_report(
            bank,
            legacy_records=legacy,
            fixed_basis=_BASIS,
            protocol=protocol,
            eligibility=bad_eligibility,
        )

    misaligned = {name: list(rows) for name, rows in bank.items()}
    source = misaligned[_CANDIDATES[1]][0]
    payload = source.to_dict()
    payload["family_id"] = "wrong-family"
    del payload["prompt_record_sha256"]
    misaligned[_CANDIDATES[1]][0] = type(source)(**payload)
    with pytest.raises(ValueError, match="static-U shared columns"):
        build_generator_innovation_adaptive_v2_report(
            {name: tuple(rows) for name, rows in misaligned.items()},
            legacy_records=legacy,
            fixed_basis=_BASIS,
            protocol=protocol,
            eligibility=eligibility,
        )

    report = build_generator_innovation_adaptive_v2_report(
        bank,
        legacy_records=legacy,
        fixed_basis=_BASIS,
        protocol=protocol,
        eligibility=eligibility,
    )
    changed = copy.deepcopy(report)
    changed["audit"]["token_used_as_independent_split_unit"] = True
    with pytest.raises(ValueError, match="audit|hash"):
        validate_generator_innovation_adaptive_v2_report(changed)
