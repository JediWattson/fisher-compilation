from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fisher_graph.gemma3_l3_l4_mixed_mode_falsification_experiment as exp


def _synthetic_panel(
    *,
    interaction_strength: float = 0.0,
    interaction_pair: int | None = None,
    interaction_parity: str = "constant",
    candidate_interaction: float = 0.0,
    interaction_radius_power: float | None = None,
    high_radius_interaction_sign: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    protocol = exp.default_mixed_mode_protocol()
    singleton = torch.zeros(
        (2, 16, 2, 1, 32, 64),
        dtype=torch.float64,
    )
    for radius_ordinal, radius in enumerate(protocol.radii):
        for mode_ordinal, mode in enumerate(protocol.unique_modes):
            amplitude = radius * (1.0 + mode_ordinal / 32.0)
            singleton[radius_ordinal, mode_ordinal, 0, 0, 0, mode] = (
                amplitude
            )
            singleton[radius_ordinal, mode_ordinal, 1, 0, 0, mode] = (
                -amplitude
            )
    mode_ordinal = {
        mode: ordinal for ordinal, mode in enumerate(protocol.unique_modes)
    }
    mixed_truth = torch.zeros(
        (2, 24, 4, 1, 32, 64),
        dtype=torch.float64,
    )
    mixed_prediction = torch.zeros_like(mixed_truth)
    for radius_ordinal in range(2):
        for pair_ordinal, (left, right) in enumerate(protocol.pairs):
            for sign_ordinal, (_label, left_sign, right_sign) in enumerate(
                protocol.sign_rows
            ):
                left_row = 0 if left_sign == 1 else 1
                right_row = 0 if right_sign == 1 else 1
                additive = (
                    singleton[radius_ordinal, mode_ordinal[left], left_row]
                    + singleton[
                        radius_ordinal,
                        mode_ordinal[right],
                        right_row,
                    ]
                )
                mixed_prediction[
                    radius_ordinal, pair_ordinal, sign_ordinal
                ] = additive
                mixed_truth[
                    radius_ordinal, pair_ordinal, sign_ordinal
                ] = additive
            if (
                interaction_strength > 0.0
                and (
                    interaction_pair is None
                    or pair_ordinal == interaction_pair
                )
            ):
                response_scale = float(
                    mixed_prediction[radius_ordinal, pair_ordinal]
                    .square()
                    .mean()
                    .sqrt()
                )
                value = interaction_strength * (
                    response_scale
                    if interaction_radius_power is None
                    else protocol.radii[radius_ordinal]
                    ** interaction_radius_power
                )
                if radius_ordinal == 1:
                    value *= high_radius_interaction_sign
                for sign_ordinal, (
                    _label,
                    left_sign,
                    right_sign,
                ) in enumerate(protocol.sign_rows):
                    parity = (
                        left_sign * right_sign
                        if interaction_parity == "odd_odd"
                        else 1
                    )
                    mixed_truth[
                        radius_ordinal,
                        pair_ordinal,
                        sign_ordinal,
                        0,
                        1,
                        0,
                    ] += parity * value
    if candidate_interaction:
        mixed_prediction[:, 0, :, 0, 2, 1] += candidate_interaction
    zeros_mixed = torch.zeros_like(mixed_prediction)
    zeros_singleton = torch.zeros_like(singleton)
    tensors = {
        "mixed_truth": mixed_truth,
        "mixed_linear_prediction": mixed_prediction,
        "mixed_quadratic_prediction": zeros_mixed,
        "singleton_truth": singleton,
        "singleton_linear_prediction": singleton.clone(),
        "singleton_quadratic_prediction": zeros_singleton,
        "zero_sentinel": torch.zeros((1, 32, 64), dtype=torch.float64),
        "repeat_sentinel_first": torch.ones(
            (1, 32, 64),
            dtype=torch.float64,
        ),
        "repeat_sentinel_second": torch.ones(
            (1, 32, 64),
            dtype=torch.float64,
        ),
    }
    sentinels = {
        "empirical_noise_floor_l2": (
            torch.finfo(torch.float64).eps * (32 * 64) ** 0.5
        )
    }
    return tensors, sentinels


def test_protocol_freezes_balanced_panel_before_inference() -> None:
    protocol = exp.default_mixed_mode_protocol()
    assert protocol.origin == 28
    assert protocol.radii == (0.5, 1.0)
    assert tuple(label for label, _left, _right in protocol.sign_rows) == (
        "++",
        "+-",
        "-+",
        "--",
    )
    assert len(protocol.pairs) == 24
    assert len(protocol.unique_modes) == 16
    assert all(left < right for left, right in protocol.pairs)
    assert protocol.metadata()["response_measurement_used_for_protocol_selection"] is False
    assert protocol.metadata()["component_normalization"].endswith(
        "two_mode_standardized_radial_norm=rho"
    )
    assert protocol.artifact_sha256 == exp.default_mixed_mode_protocol().artifact_sha256

    state = deepcopy(protocol)
    with pytest.raises(ValueError, match="frozen panel"):
        exp.FrozenMixedModeProtocol(
            origin=state.origin + 1,
            sequence_length=state.sequence_length,
            modal_rank=state.modal_rank,
            max_lag=state.max_lag,
            radii=state.radii,
            sign_rows=state.sign_rows,
            pair_families=state.pair_families,
            unique_modes=state.unique_modes,
            gates=state.gates,
        )


def test_stress_modes_are_exact_quadratic_source_leverage_top8() -> None:
    basis = torch.zeros((64, 4), dtype=torch.float64)
    descending = (0, 2, 1, 15, 7, 43, 42, 28)
    for ordinal, mode in enumerate(descending):
        basis[mode, 0] = 8.0 - ordinal * 0.5
    candidate = SimpleNamespace(
        quadratic_plan=SimpleNamespace(source_basis=basis)
    )
    result = exp._validate_stress_modes_from_quadratic_leverage(
        candidate,  # type: ignore[arg-type]
        exp.default_mixed_mode_protocol(),
    )
    assert result["top8_descending_leverage_order"] == descending
    assert result["set_equality_verified_before_live_response_measurement"]
    assert result["stress_pair_order_is_exact_frozen_ring_not_score_order"]


def test_repeat_sentinel_brackets_the_complete_measurement_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = exp.default_mixed_mode_protocol()
    basis = torch.zeros((64, 4), dtype=torch.float64)
    for ordinal, mode in enumerate((0, 2, 1, 15, 7, 43, 42, 28)):
        basis[mode, 0] = 8.0 - ordinal * 0.5
    scales = torch.ones(64, dtype=torch.float64)
    candidate = SimpleNamespace(
        linear_plan=SimpleNamespace(source_scales=scales),
        quadratic_plan=SimpleNamespace(
            source_scales=scales,
            source_basis=basis,
        ),
    )

    class FakeAccounting:
        def metadata(self) -> dict[str, object]:
            return {"factorized_linear_macs": 1}

    class FakeRuntime(nn.Module):
        def forward(
            self,
            source_modes: torch.Tensor,
            **_kwargs: object,
        ) -> torch.Tensor:
            return torch.zeros_like(source_modes)

        def execution_accounting(self, **_kwargs: object) -> FakeAccounting:
            return FakeAccounting()

    monkeypatch.setattr(
        exp,
        "Gemma3ConditionalSpectralCandidate",
        SimpleNamespace,
    )
    monkeypatch.setattr(
        exp,
        "_prepare_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    calls: list[torch.Tensor] = []

    def structural_map(source: torch.Tensor) -> torch.Tensor:
        calls.append(source.detach().clone())
        return source.clone()

    positions = torch.arange(60, dtype=torch.long).unsqueeze(0)
    valid = torch.ones((1, 60), dtype=torch.bool)
    _tensors, sentinels = exp.measure_mixed_mode_responses(
        structural_map,
        candidate=candidate,  # type: ignore[arg-type]
        source_sigmas=scales,
        logical_positions=positions,
        valid_mask=valid,
        protocol=protocol,
    )
    assert len(calls) == 259
    assert not bool(calls[0].any())
    torch.testing.assert_close(calls[1], calls[-1])
    assert not torch.equal(calls[1], calls[2])
    assert sentinels["repeat_sentinel_brackets_all_panel_measurements"]
    assert sentinels["fixed_reference_function_evaluation_count"] == 259


def test_additive_panel_passes_support_and_generic_math_crosscheck() -> None:
    tensors, sentinels = _synthetic_panel()
    protocol = exp.default_mixed_mode_protocol()
    metrics, derived = exp.evaluate_mixed_mode_falsification(
        tensors,
        sentinels=sentinels,
        protocol=protocol,
    )
    assert metrics["decision"] == "panel_supports_additive_diagonal_executor"
    assert all(metrics["support_checks"].values())
    assert not any(metrics["material_failure_checks"].values())
    assert metrics["diagnostic_checks"]["candidate_interaction_numerically_zero"]
    assert (
        metrics["by_radius"][1]["pooled"]["nonadditivity_relative_norm"]
        == 0.0
    )

    scales = torch.ones(64, dtype=torch.float64)
    candidate = SimpleNamespace(
        artifact_sha256="a" * 64,
        linear_plan=SimpleNamespace(source_scales=scales),
    )
    generic, crosscheck = exp._generic_interaction_crosscheck(
        tensors,
        derived,
        candidate=candidate,  # type: ignore[arg-type]
        protocol=protocol,
    )
    assert generic.origin == 28
    assert generic.pair_count == 24
    assert crosscheck["crosscheck"][
        "singleton_nonadditivity_matches_runner"
    ]


def test_singleton_subtraction_catches_c00_interaction_that_c11_misses() -> None:
    tensors, sentinels = _synthetic_panel(
        interaction_strength=10.0,
        interaction_pair=0,
        interaction_parity="constant",
    )
    metrics, _derived = exp.evaluate_mixed_mode_falsification(
        tensors,
        sentinels=sentinels,
        protocol=exp.default_mixed_mode_protocol(),
    )
    operating = metrics["by_radius"][1]
    pair = operating["pairs"][0]
    assert pair["nonadditivity_relative_norm"] >= 0.20
    assert pair["interaction_parity_c11_energy_fraction"] == 0.0
    assert metrics["material_failure_checks"]["reliable_pair_nonadditivity"]
    assert metrics["decision"] == "material_cross_interaction_failure"


def test_tiny_pure_c11_can_be_harmless_despite_pi11_near_one() -> None:
    tensors, sentinels = _synthetic_panel(
        interaction_strength=0.001,
        interaction_parity="odd_odd",
    )
    metrics, _derived = exp.evaluate_mixed_mode_falsification(
        tensors,
        sentinels=sentinels,
        protocol=exp.default_mixed_mode_protocol(),
    )
    pooled = metrics["by_radius"][1]["pooled"]
    assert pooled["interaction_parity_c11_energy_fraction"] == pytest.approx(
        1.0
    )
    assert pooled["c11_full_response_energy_fraction"] < 0.05
    assert metrics["diagnostic_checks"]["c11_full_response_energy_fraction"]
    assert not metrics["bilinear_branch_suitability"]["suitable"]


def test_material_quadratically_scaling_c11_supports_bilinear_branch() -> None:
    tensors, sentinels = _synthetic_panel(
        interaction_strength=1.0,
        interaction_parity="odd_odd",
        interaction_radius_power=2.0,
    )
    metrics, _derived = exp.evaluate_mixed_mode_falsification(
        tensors,
        sentinels=sentinels,
        protocol=exp.default_mixed_mode_protocol(),
    )
    suitability = metrics["bilinear_branch_suitability"]
    assert suitability["interaction_magnitude_is_material"]
    assert suitability["interaction_parity_c11_gate_passes"]
    assert suitability["c11_hessian_scaling_cosine_gate_passes"]
    assert suitability[
        "c11_hessian_scaling_relative_error_gate_passes"
    ]
    assert suitability["suitable"]


def test_material_c11_with_bad_radius_scaling_rejects_bilinear_branch() -> None:
    tensors, sentinels = _synthetic_panel(
        interaction_strength=1.0,
        interaction_parity="odd_odd",
        interaction_radius_power=1.0,
        high_radius_interaction_sign=-1,
    )
    metrics, _derived = exp.evaluate_mixed_mode_falsification(
        tensors,
        sentinels=sentinels,
        protocol=exp.default_mixed_mode_protocol(),
    )
    suitability = metrics["bilinear_branch_suitability"]
    assert suitability["interaction_magnitude_is_material"]
    assert suitability["interaction_parity_c11_gate_passes"]
    assert not suitability["c11_hessian_scaling_cosine_gate_passes"]
    assert not suitability[
        "c11_hessian_scaling_relative_error_gate_passes"
    ]
    assert not suitability["suitable"]


def test_oracle_metrics_fail_closed_if_frozen_candidate_is_not_additive() -> None:
    tensors, sentinels = _synthetic_panel(candidate_interaction=1e-2)
    with pytest.raises(ValueError, match="not numerically additive"):
        exp.evaluate_mixed_mode_falsification(
            tensors,
            sentinels=sentinels,
            protocol=exp.default_mixed_mode_protocol(),
        )


def test_publish_is_no_overwrite_and_json_has_no_response_tensors(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mixed.pt"
    report = exp._publish(
        {"response_tensors": {"truth": torch.ones(2)}},
        {
            "schema": "fixture",
            "safety": exp._SAFETY,
            "tensor_manifest": {
                "truth": {
                    "shape": (2,),
                    "dtype": "float64",
                    "sha256": "b" * 64,
                }
            },
        },
        output=output,
    )
    assert output.exists()
    report_path = output.with_suffix(".json")
    assert report_path.exists()
    source_safe = json.loads(report_path.read_text(encoding="utf-8"))
    assert "response_tensors" not in source_safe
    assert source_safe["artifact"]["json_report_source_safe"] is True
    assert source_safe["report_sha256"] == report["report_sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        exp._publish({}, {}, output=output)


def test_runner_has_no_fitting_api_or_prompt_surface() -> None:
    source = inspect.getsource(exp)
    assert "fit_conditional_spectral_generator" not in source
    assert "AutoTokenizer" not in source
    assert exp._SAFETY["contains_prompt_text"] is False
    assert exp._SAFETY["contains_token_ids"] is False
    assert exp._claim_boundaries()["candidate_was_refit_or_modified"] is False
    assert exp._claim_boundaries()["nll_claim"] is False
