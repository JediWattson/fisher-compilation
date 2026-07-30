from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from fisher_graph.gemma3_l3_l4_h4_incremental_signal_diagnostic import (
    _parent_cell_from_damping_cell,
    _resource_cost,
    analyze_gemma_h4_damping,
    analyze_gemma_h4_incremental_signal,
    build_damping_parser,
    build_parser,
    derive_gemma_h4_damping_recipe_tensors,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaTwoHeadFitSequence,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _incremental_signal_sequences() -> tuple[GemmaTwoHeadFitSequence, ...]:
    """Make H4 useful only through a direction outside the output decoder."""

    width = 3
    length = 8
    decoder_direction = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    hidden_state_direction = torch.tensor(
        [0.0, 1.0, 0.0],
        dtype=torch.float64,
    )
    result: list[GemmaTwoHeadFitSequence] = []
    for index, family in enumerate(
        ("family-a", "family-b", "family-c", "family-d")
    ):
        # Distinct magnitudes keep the rank-one H4 feature deterministic while
        # every outer fold still sees both signs and a nonzero residual.
        values = (
            torch.linspace(-1.5, 1.5, length, dtype=torch.float64)
            + 0.125 * index
        )
        candidate_h4 = (
            values.unsqueeze(1) * hidden_state_direction.unsqueeze(0)
        )
        residual = (
            (1.25 * values).unsqueeze(1)
            * decoder_direction.unsqueeze(0)
        )
        gradient = decoder_direction.repeat(length, 1)
        mask = torch.ones(length, dtype=torch.bool)
        zeros = torch.zeros(length, width, dtype=torch.float64)
        result.append(
            GemmaTwoHeadFitSequence(
                example_id=f"example-{index}",
                family_id=family,
                model_inputs_sha256=_sha(10 + index),
                runtime_binding_sha256=_sha(20),
                source_modes=torch.zeros(
                    length,
                    1,
                    dtype=torch.float64,
                ),
                logical_positions=torch.arange(
                    length,
                    dtype=torch.int64,
                ),
                valid_target_mask=mask,
                source_eligible_mask=mask.clone(),
                target_affected_mask=mask.clone(),
                native_x4=zeros,
                candidate_x4=zeros,
                native_h4=candidate_h4 + residual,
                candidate_h4=candidate_h4,
                x4_loss_gradient=gradient,
                h4_loss_gradient=gradient,
                candidate_h4_loss_gradient=gradient,
            )
        )
    return tuple(result)


def _assert_tensor_free(value: object) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_tensor_free(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_tensor_free(nested)
    else:
        assert not isinstance(value, torch.Tensor)


def test_parser_has_no_selection_or_guard_input_capability() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--fit-input" in options
    assert "--selection-input" not in options
    assert "--guard-input" not in options

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--accepted-x4-candidate-sha256",
                _sha(1),
                "--selection-input",
                "selection.json",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--accepted-x4-candidate-sha256",
                _sha(1),
                "--guard-input",
                "guard.json",
            ]
        )


def test_independent_h4_encoder_is_identifiable_deterministic_and_audited(
) -> None:
    sequences = _incremental_signal_sequences()
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    first = analyze_gemma_h4_incremental_signal(
        sequences=sequences,
        output_decoder=decoder,
        lag_counts=(1,),
        input_ranks=(1,),
        ridge=1.0e-6,
    )
    repeated = analyze_gemma_h4_incremental_signal(
        sequences=tuple(reversed(sequences)),
        output_decoder=decoder,
        lag_counts=(1,),
        input_ranks=(1,),
        ridge=1.0e-6,
    )

    assert repeated == first
    assert first["analysis_sha256"] == repeated["analysis_sha256"]
    assert first["contract"]["development_role"] == (
        "calibration_a_fit_only"
    )
    assert first["selection"]["selection_panel_authorized"] is False
    assert first["selection"]["guard_authorized"] is False
    assert first["safety"] == {
        "fit_only": True,
        "selection_input_capability_present": False,
        "guard_input_capability_present": False,
        "prompt_text_in_report": False,
        "token_ids_in_report": False,
        "activation_rows_in_report": False,
        "gradient_rows_in_report": False,
        "coefficient_tensors_in_report": False,
        "model_weights_in_report": False,
        "compression_claim": False,
        "latency_claim": False,
    }

    cells = {
        cell["encoder_kind"]: cell
        for cell in first["conditioned_cells"]
    }
    reused = cells["reused_output_decoder"]
    independent = cells["independent_crossfit_h4_svd"]

    assert reused["minimum_incremental_numerical_rank"] == 0
    assert reused["eligible"] is False
    assert independent["minimum_incremental_numerical_rank"] == 1
    assert independent["eligible"] is True
    assert independent["family_linearized_win_count"] == 4
    assert (
        independent["macro_relative_improvement"][
            "linearized_nll_residual_rmse"
        ]
        > 0.99
    )
    assert first["selection"]["status"] == (
        "crossfit_incremental_signal_identified"
    )
    selected = first["selection"]["selected_cell"]
    assert selected["encoder_kind"] == "independent_crossfit_h4_svd"
    recipe = first["selection"]["winning_recipe"]
    assert recipe["residualizer_folded_into_lag_kernel"] is True
    assert set(recipe) >= {
        "decoder_sha256",
        "state_encoder_sha256",
        "state_kernel_sha256",
        "stored_lag_coefficients_sha256",
        "recipe_sha256",
    }

    # W=3, S=1, Rout=Rin=L=1. Both variants execute eight MACs, but
    # the reused encoder stores only its 1x1 state kernel while the independent
    # encoder additionally stores its 1x3 projector.
    assert reused["resources"] == {
        "head_parameters": 5,
        "head_runtime_parameter_bytes": 40,
        "head_logical_macs_per_token": 8,
        "conditioning_parameters": 1,
        "conditioning_runtime_parameter_bytes": 8,
        "conditioning_logical_macs_per_token": 4,
    }
    assert independent["resources"] == {
        "head_parameters": 8,
        "head_runtime_parameter_bytes": 64,
        "head_logical_macs_per_token": 8,
        "conditioning_parameters": 4,
        "conditioning_runtime_parameter_bytes": 32,
        "conditioning_logical_macs_per_token": 4,
    }

    _assert_tensor_free(first)
    encoded = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert "example-0" not in encoded
    assert "family-a" in encoded


def test_damping_recipe_tensor_materialization_matches_scalar_recipe() -> None:
    sequences = _incremental_signal_sequences()
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    tensors, recipe = derive_gemma_h4_damping_recipe_tensors(
        sequences=tuple(reversed(sequences)),
        output_decoder=decoder,
        lag_count=1,
        input_rank=1,
        state_scale=0.5,
        ridge=1.0e-6,
    )
    repeated, repeated_recipe = derive_gemma_h4_damping_recipe_tensors(
        sequences=sequences,
        output_decoder=decoder,
        lag_count=1,
        input_rank=1,
        state_scale=0.5,
        ridge=1.0e-6,
    )

    assert repeated_recipe == recipe
    torch.testing.assert_close(repeated.decoder, tensors.decoder)
    torch.testing.assert_close(
        repeated.state_encoder,
        tensors.state_encoder,
    )
    torch.testing.assert_close(repeated.state_kernel, tensors.state_kernel)
    torch.testing.assert_close(
        repeated.stored_lag_coefficients,
        tensors.stored_lag_coefficients,
    )
    torch.testing.assert_close(
        repeated.baseline_lag_coefficients,
        tensors.baseline_lag_coefficients,
    )
    assert tensors.lag_kernel.shape == (1, 1, 1)
    assert tensors.baseline_lag_kernel.shape == (1, 1, 1)
    assert recipe["state_scale"] == 0.5
    assert recipe["state_encoder_sha256"] != recipe["decoder_sha256"]
    assert recipe["state_kernel_sha256"] != (
        recipe["stored_lag_coefficients_sha256"]
    )


def test_full_row_rank_l3_design_rejects_apparent_h4_improvement() -> None:
    sequences: list[GemmaTwoHeadFitSequence] = []
    for family_index, sequence in enumerate(
        _incremental_signal_sequences()
    ):
        source_modes = torch.zeros(8, 32, dtype=torch.float64)
        source_modes[
            :,
            family_index * 8 : (family_index + 1) * 8,
        ] = torch.eye(8, dtype=torch.float64)
        sequences.append(replace(sequence, source_modes=source_modes))

    result = analyze_gemma_h4_incremental_signal(
        sequences=sequences,
        output_decoder=torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        lag_counts=(1,),
        input_ranks=(1,),
        ridge=1.0e-6,
    )

    independent = next(
        cell
        for cell in result["conditioned_cells"]
        if cell["encoder_kind"] == "independent_crossfit_h4_svd"
    )
    assert (
        independent["macro_relative_improvement"][
            "linearized_nll_residual_rmse"
        ]
        > 0.99
    )
    assert independent["minimum_incremental_numerical_rank"] == 0
    assert independent["gate"]["all_input_ranks_identifiable"] is False
    assert independent["eligible"] is False
    assert result["selection"]["status"] == (
        "no_crossfit_incremental_signal"
    )
    for fold in result["design_spectra"][0]["outer_crossfit_folds"]:
        assert fold["numerical_rank"] == fold["row_count"]
        assert fold["available_row_dimension"] == 0


@pytest.mark.parametrize(
    ("input_rank", "conditioning", "head_total"),
    (
        (8, 5_184, 12_352),
        (16, 10_368, 17_536),
        (32, 20_736, 27_904),
    ),
)
def test_independent_encoder_resource_accounting_at_gemma_width(
    input_rank: int,
    conditioning: int,
    head_total: int,
) -> None:
    resources = _resource_cost(
        lag_count=4,
        source_rank=64,
        width=640,
        output_rank=8,
        encoder_kind="independent_crossfit_h4_svd",
        input_rank=input_rank,
    )

    assert resources == {
        "head_parameters": head_total,
        "head_runtime_parameter_bytes": 8 * head_total,
        "head_logical_macs_per_token": head_total,
        "conditioning_parameters": conditioning,
        "conditioning_runtime_parameter_bytes": 8 * conditioning,
        "conditioning_logical_macs_per_token": conditioning,
    }


def test_eight_family_gate_requires_six_wins_and_three_inner_folds() -> None:
    base = _incremental_signal_sequences()
    sequences = tuple(
        replace(
            base[index % len(base)],
            example_id=f"expanded-example-{index}",
            family_id=f"expanded-family-{index}",
            model_inputs_sha256=_sha(100 + index),
        )
        for index in range(8)
    )

    result = analyze_gemma_h4_incremental_signal(
        sequences=sequences,
        output_decoder=torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        lag_counts=(1,),
        input_ranks=(1,),
        ridge=1.0e-6,
    )

    assert result["contract"]["gate"]["minimum_family_win_fraction"] == 0.75
    assert result["contract"]["gate"]["minimum_family_win_count"] == 6
    independent = next(
        cell
        for cell in result["conditioned_cells"]
        if cell["encoder_kind"] == "independent_crossfit_h4_svd"
    )
    assert independent["minimum_family_win_count"] == 6
    assert independent["family_linearized_win_count"] == 8
    assert independent["gate"]["minimum_family_win_count_met"] is True
    for fold in result["design_spectra"][0]["outer_crossfit_folds"]:
        assert len(fold["inner_training_folds"]) == 3
        held = {
            family
            for inner in fold["inner_training_folds"]
            for family in inner["held_families"]
        }
        assert len(held) == 7


def test_fixed_head_damping_is_deterministic_and_selects_smallest_eligible_alpha(
) -> None:
    sequences = _incremental_signal_sequences()
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    alphas = (0.25, 0.5, 0.75, 1.0)

    first = analyze_gemma_h4_damping(
        sequences=sequences,
        output_decoder=decoder,
        lag_count=1,
        input_rank=1,
        damping_alphas=alphas,
        ridge=1.0e-6,
    )
    repeated = analyze_gemma_h4_damping(
        sequences=tuple(reversed(sequences)),
        output_decoder=decoder,
        lag_count=1,
        input_rank=1,
        damping_alphas=alphas,
        ridge=1.0e-6,
    )

    assert repeated == first
    assert repeated["analysis_sha256"] == first["analysis_sha256"]
    assert all(cell["eligible"] for cell in first["alpha_cells"])
    assert first["contract"]["selection_rule"] == (
        "smallest_alpha_passing_all_gates"
    )
    selected = first["selection"]["selected_alpha_cell"]
    assert selected["alpha"] == 0.25
    assert first["selection"]["winning_recipe"]["state_scale"] == 0.25
    _assert_tensor_free(first)
    json.dumps(first, allow_nan=False)


def test_alpha_one_exactly_reproduces_incremental_signal_parent_cell() -> None:
    sequences = _incremental_signal_sequences()
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    damping = analyze_gemma_h4_damping(
        sequences=sequences,
        output_decoder=decoder,
        lag_count=1,
        input_rank=1,
        damping_alphas=(0.25, 1.0),
        ridge=1.0e-6,
    )
    incremental = analyze_gemma_h4_incremental_signal(
        sequences=sequences,
        output_decoder=decoder,
        lag_counts=(1,),
        input_ranks=(1,),
        ridge=1.0e-6,
    )

    alpha_one = next(
        cell for cell in damping["alpha_cells"] if cell["alpha"] == 1.0
    )
    normalized = _parent_cell_from_damping_cell(
        alpha_one,
        lag_count=1,
        source_rank=1,
        input_rank=1,
        output_rank=1,
    )
    parent = next(
        cell
        for cell in incremental["conditioned_cells"]
        if cell["encoder_kind"] == "independent_crossfit_h4_svd"
    )

    assert normalized == parent


def test_damping_fails_closed_before_selection_on_parent_mismatch() -> None:
    sequences = _incremental_signal_sequences()
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    incremental = analyze_gemma_h4_incremental_signal(
        sequences=sequences,
        output_decoder=decoder,
        lag_counts=(1,),
        input_ranks=(1,),
        ridge=1.0e-6,
    )
    parent = next(
        cell
        for cell in incremental["conditioned_cells"]
        if cell["encoder_kind"] == "independent_crossfit_h4_svd"
    )

    matched = analyze_gemma_h4_damping(
        sequences=sequences,
        output_decoder=decoder,
        lag_count=1,
        input_rank=1,
        damping_alphas=(0.25, 1.0),
        ridge=1.0e-6,
        hypothesis_cell=parent,
    )
    assert matched["selection"]["selected_alpha_cell"]["alpha"] == 0.25

    tampered = {
        **parent,
        "worst_family_linearized_improvement": -0.5,
    }
    with pytest.raises(
        RuntimeError,
        match="alpha=1 hypothesis reproduction mismatch",
    ):
        analyze_gemma_h4_damping(
            sequences=sequences,
            output_decoder=decoder,
            lag_count=1,
            input_rank=1,
            damping_alphas=(0.25, 1.0),
            ridge=1.0e-6,
            hypothesis_cell=tampered,
        )


def test_fixed_head_hashes_are_undamped_across_alpha() -> None:
    result = analyze_gemma_h4_damping(
        sequences=_incremental_signal_sequences(),
        output_decoder=torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        lag_count=1,
        input_rank=1,
        damping_alphas=(0.25, 0.5, 1.0),
        ridge=1.0e-6,
    )
    by_alpha = {
        cell["alpha"]: {
            family["held_family"]: family
            for family in cell["per_family"]
        }
        for cell in result["alpha_cells"]
    }

    for held_family, alpha_one in by_alpha[1.0].items():
        undamped_hashes = {
            families[held_family]["undamped_state_kernel_sha256"]
            for families in by_alpha.values()
        }
        encoder_hashes = {
            families[held_family]["encoder_sha256"]
            for families in by_alpha.values()
        }
        assert len(undamped_hashes) == 1
        assert len(encoder_hashes) == 1
        assert (
            alpha_one["damped_state_kernel_sha256"]
            == alpha_one["undamped_state_kernel_sha256"]
        )


def test_damping_l16_rank32_gemma_resource_accounting_is_exact() -> None:
    resources = _resource_cost(
        lag_count=16,
        source_rank=64,
        width=640,
        output_rank=8,
        encoder_kind="independent_crossfit_h4_svd",
        input_rank=32,
    )

    assert resources == {
        "head_parameters": 34_048,
        "head_runtime_parameter_bytes": 272_384,
        "head_logical_macs_per_token": 34_048,
        "conditioning_parameters": 20_736,
        "conditioning_runtime_parameter_bytes": 165_888,
        "conditioning_logical_macs_per_token": 20_736,
    }


@pytest.mark.parametrize(
    "alphas",
    (
        (),
        (True,),
        (0.0, 1.0),
        (0.5, 0.5, 1.0),
        (0.5, 0.25, 1.0),
        (0.25, 1.25),
        (0.25, float("nan")),
    ),
)
def test_damping_rejects_invalid_alpha_grids(
    alphas: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="damping_alphas"):
        analyze_gemma_h4_damping(
            sequences=_incremental_signal_sequences(),
            output_decoder=torch.tensor(
                [[1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
            lag_count=1,
            input_rank=1,
            damping_alphas=alphas,  # type: ignore[arg-type]
            ridge=1.0e-6,
        )


def test_damping_parser_locks_structure_and_exposes_hypothesis_hashes() -> None:
    parser = build_damping_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--hypothesis-report" in options
    assert "--hypothesis-report-sha256" in options
    assert "--hypothesis-report-file-sha256" in options
    forbidden = {
        "--lag-count",
        "--lag-counts",
        "--input-rank",
        "--input-ranks",
        "--selection-input",
        "--guard-input",
        "--damping-alpha",
        "--damping-alphas",
        "--device",
        "--dtype",
        "--expected-fit-example-count",
        "--expected-fit-family-count",
        "--output-rank",
        "--ridge",
    }
    assert options.isdisjoint(forbidden)

    required = [
        "--hypothesis-report-sha256",
        _sha(1),
        "--hypothesis-report-file-sha256",
        _sha(2),
        "--accepted-x4-candidate-sha256",
        _sha(3),
    ]
    parsed = parser.parse_args(required)
    assert parsed.hypothesis_report_sha256 == _sha(1)
    assert parsed.hypothesis_report_file_sha256 == _sha(2)

    for option in sorted(forbidden):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, option, "1"])
