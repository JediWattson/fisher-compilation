from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_fisher_corrective_development as development,
)
from fisher_graph.token_loss_fisher import (
    build_token_loss_fisher_prompt_record,
)


_COMBINED_NAMES = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
    "ew_occupancy_contrast_real",
    "ew_occupancy_contrast_imag",
)


def _prompt_records() -> tuple[object, ...]:
    generator = torch.Generator().manual_seed(23)
    scores = torch.randn(
        40,
        8,
        generator=generator,
        dtype=torch.float64,
    )
    target = scores[:, :2] @ torch.tensor(
        (0.04, -0.03), dtype=torch.float64
    )
    return tuple(
        build_token_loss_fisher_prompt_record(
            example_id=f"example-{family}",
            family_id=f"family-{family}",
            coordinate_names=_COMBINED_NAMES,
            token_scores=scores,
            compensation_target=target,
        )
        for family in range(4)
    )


def _graph(*, arm: str) -> dict[str, object]:
    occupancy = f"{arm}_occupancy_contrast"
    names = (
        "shared_real",
        "shared_imag",
        "balance_contrast_real",
        "balance_contrast_imag",
        f"{occupancy}_real",
        f"{occupancy}_imag",
    )
    pairs = (
        (names[0], names[2]),
        (names[0], names[4]),
        (names[2], names[4]),
        (names[1], names[3]),
        (names[1], names[5]),
        (names[3], names[5]),
    )
    return {
        "coordinate_names": names,
        "fisher_coupling_is_symmetric": True,
        "causal_direction_inferred": False,
        "stable_edge_count": 6,
        "edges": tuple(
            {
                "left_coordinate": left,
                "right_coordinate": right,
                "global_correlation": 0.5,
                "stable_fold_count": 8,
                "stable": True,
            }
            for left, right in pairs
        ),
    }


def _upstream() -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph.gemma3_l3_l4."
            "iterative_token_fisher_development.v1"
        ),
        "report_sha256": "a" * 64,
        "prompt_fisher_records": tuple(
            row.to_dict() for row in _prompt_records()
        ),
        "analysis": {
            "cumulative_coupling_graph": _graph(arm="cumulative"),
            "ew_coupling_graph": _graph(arm="ew"),
        },
    }


def test_build_validate_and_replay_adaptive_corrective_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_gemma_iterative_token_fisher_development_report",
        lambda _report: None,
    )
    upstream = _upstream()
    report = (
        development
        .build_gemma_iterative_fisher_corrective_development_report(
            token_fisher_report=upstream,
            token_fisher_report_file_sha256="b" * 64,
        )
    )

    development.validate_gemma_iterative_fisher_corrective_development_report(
        report
    )
    assert report["primary_arm"] == "cumulative"
    assert report["sensitivity_arm"] == "ew"
    assert report["decision"]["provider_compiled"] is False
    assert report["decision"]["runtime_claim_authorized"] is False
    assert report["audit"]["family_blocked_inner_lofo"] is True
    assert report["resources"]["source_model_forwards"] == 0
    assert (
        development
        .replay_gemma_iterative_fisher_corrective_development_report(
            token_fisher_report=upstream,
            token_fisher_report_file_sha256="b" * 64,
            report=report,
        )
        == report
    )


def test_topology_cannot_claim_causal_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_gemma_iterative_token_fisher_development_report",
        lambda _report: None,
    )
    upstream = _upstream()
    upstream["analysis"]["cumulative_coupling_graph"][
        "causal_direction_inferred"
    ] = True

    with pytest.raises(ValueError, match="symmetric stable couplings"):
        (
            development
            .build_gemma_iterative_fisher_corrective_development_report(
                token_fisher_report=upstream,
                token_fisher_report_file_sha256="b" * 64,
            )
        )


def test_outer_report_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        development,
        "validate_gemma_iterative_token_fisher_development_report",
        lambda _report: None,
    )
    report = (
        development
        .build_gemma_iterative_fisher_corrective_development_report(
            token_fisher_report=_upstream(),
            token_fisher_report_file_sha256="b" * 64,
        )
    )
    changed = copy.deepcopy(report)
    changed["resources"]["source_model_forwards"] = 1

    with pytest.raises(ValueError, match="resource receipt differs"):
        (
            development
            .validate_gemma_iterative_fisher_corrective_development_report(
                changed
            )
        )
