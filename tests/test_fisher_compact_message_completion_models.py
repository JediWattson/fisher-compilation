from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from fisher_graph.fisher_compact_message_completion_capture import (
    decode_safetensors,
    encode_safetensors,
)
from fisher_graph.fisher_compact_message_completion_models import (
    FULL_DIAGNOSTIC_TRAINING_CONFIG,
    FULL_INITIAL_TRAINING_CONFIG,
    FULL_REFIT_CONFIG,
    build_weights_only_bundles,
    bootstrap_interaction_statistics,
    evaluate_bank,
    evaluate_interaction_mean,
    fit_initial_model_banks,
    fit_refit_model_banks,
    freeze_epsilon_down,
    freeze_epsilon_down_from_fit_train,
    load_weights_only_bundles,
    RAW_EXECUTION_PROTOCOL_SHA256,
    self_test_weights_only_bundles,
    trained_j4_operator_aggregate,
)
from fisher_graph.fisher_compact_message_completion_qualification import (
    QualificationError,
    evaluate_o2,
    fit_selection_geometry,
    prepare_completion_panel,
)
from fisher_graph.fisher_compact_message_completion_science import (
    CompletionTrainingConfig,
    J4Model,
    OperatorObservation,
    aggregate_operator_pair_summaries,
    hierarchical_row_weights,
    stable_rank,
    summarize_operator_pairs,
)
from fisher_graph.fisher_compact_message_completion_synthetic import (
    build_synthetic_campaign,
)


FAST_TRAINING = CompletionTrainingConfig(
    learning_rate=1e-3,
    warmup_steps=1,
    batch_size=16,
    maximum_steps=1,
    minimum_steps=1,
    evaluation_interval=1,
    early_stopping_patience=1,
)
FAST_REFIT = CompletionTrainingConfig(
    learning_rate=1e-3,
    warmup_steps=1,
    batch_size=16,
    maximum_steps=1,
    minimum_steps=1,
    evaluation_interval=1,
    early_stopping_patience=1,
    early_stopping=False,
)


def test_interaction_bootstrap_uses_three_aligned_loss_arms() -> None:
    unit_ids = (b"a" * 32, b"b" * 32)
    result = bootstrap_interaction_statistics(
        (3.0, 3.0),
        (2.0, 2.0),
        (2.5, 2.5),
        ("one", "one"),
        unit_ids=unit_ids,
        advantage_floor=3e-6,
        replicates=20,
    )
    assert result.paired_effect_interval == (0.5, 0.5)
    assert result.gamma_interval == (0.5, 0.5)
    assert result.valid_gamma_replicate_fraction == 1.0
    undefined = bootstrap_interaction_statistics(
        (2.0, 2.0),
        (2.0, 2.0),
        (2.5, 2.5),
        ("one", "one"),
        unit_ids=unit_ids,
        advantage_floor=2e-6,
        replicates=20,
    )
    assert undefined.gamma_interval is None
    assert undefined.valid_gamma_replicate_fraction == 0.0


def _qualified_fixture():
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    _, selection = evaluate_o2(campaign, geometry)
    assert selection.selected_codec_id is not None
    return campaign, geometry, selection.selected_codec_id


def test_frozen_model_budget_configs_match_the_final_amendment() -> None:
    assert FULL_INITIAL_TRAINING_CONFIG.maximum_steps == 2_000
    assert FULL_INITIAL_TRAINING_CONFIG.minimum_steps == 200
    assert FULL_INITIAL_TRAINING_CONFIG.early_stopping
    assert FULL_DIAGNOSTIC_TRAINING_CONFIG.maximum_steps == 1_000
    assert FULL_DIAGNOSTIC_TRAINING_CONFIG.minimum_steps == 200
    assert FULL_DIAGNOSTIC_TRAINING_CONFIG.early_stopping
    assert FULL_REFIT_CONFIG.maximum_steps == 1_000
    assert FULL_REFIT_CONFIG.minimum_steps == 1_000
    assert not FULL_REFIT_CONFIG.early_stopping


def test_initial_and_refit_banks_cover_the_frozen_family_contract() -> None:
    campaign, geometry, codec_id = _qualified_fixture()
    initial = fit_initial_model_banks(
        campaign,
        geometry,
        codec_id,
        config=FAST_TRAINING,
        diagnostic_config=FAST_TRAINING,
        include_optional_j2=True,
    )
    assert set(initial.fits) == {"J0", "J1", "J2N", "J2", "J2+", "J3", "J4"}
    assert initial.optimizer_updates == 14
    assert initial.representation_sha256 == geometry.identity_sha256
    assert initial.fits["J1"].restart_results == ()
    assert all(
        len(value.weights_sha256) == 64 and len(value.architecture_sha256) == 64
        for value in initial.fits.values()
    )
    fisher = torch.diag(
        torch.from_numpy(campaign.source.h4_fisher_diagonal)
    ).to(torch.float64)
    assert freeze_epsilon_down_from_fit_train(campaign) == freeze_epsilon_down(
        initial.train_panel, fisher
    )

    refit = fit_refit_model_banks(
        campaign,
        geometry,
        codec_id,
        config=FAST_REFIT,
    )
    assert set(refit.fits) == {"J2N", "J2+", "J3", "J4"}
    assert refit.optimizer_updates == 4
    assert refit.interaction_means.values.shape == (4, 4, 8)
    assert len(refit.interaction_means.identity_sha256) == 64
    for chart in refit.fits["J4"].bank.chart_ids:
        mask = refit.all_fit_panel.chart_ids == chart
        keys = tuple(
            key
            for key, keep in zip(refit.all_fit_panel.row_keys, mask, strict=True)
            if bool(keep)
        )
        weights = hierarchical_row_weights(keys).to(torch.float32)
        z = refit.all_fit_panel.z[mask].to(torch.float32)
        for head in range(campaign.config.head_count):
            cell = refit.fits["J4"].bank.cell(chart, head)
            assert isinstance(cell, J4Model)
            expected = torch.einsum(
                "n,nb->b", weights, z @ cell.activation_interaction.T
            )
            torch.testing.assert_close(
                refit.interaction_means.values[chart, head], expected
            )
    assert all(value.selected_step == 1 for value in refit.fits.values())
    assert all(
        result.final_step == 1 and not result.stopped_early
        for value in refit.fits.values()
        for result in value.restart_results
    )


def test_development_evaluation_ablation_and_weight_bundles_are_finite() -> None:
    campaign, geometry, codec_id = _qualified_fixture()
    refit = fit_refit_model_banks(
        campaign,
        geometry,
        codec_id,
        config=FAST_REFIT,
    )
    development = prepare_completion_panel(
        campaign.development_confirm, campaign, geometry, codec_id
    )
    fisher = torch.diag(
        torch.from_numpy(campaign.source.h4_fisher_diagonal)
    ).to(torch.float64)
    epsilon_down = freeze_epsilon_down(refit.all_fit_panel, fisher)
    j2n = evaluate_bank(
        refit.fits["J2N"],
        development,
        geometry.codec(codec_id),
        campaign,
        epsilon_down=epsilon_down,
    )
    j4 = evaluate_bank(
        refit.fits["J4"],
        development,
        geometry.codec(codec_id),
        campaign,
        epsilon_down=epsilon_down,
    )
    assert j2n.fidelity is not None
    assert j4.fidelity is not None
    assert torch.isfinite(torch.tensor(j2n.downstream.mean_loss))
    assert torch.isfinite(torch.tensor(j4.downstream.mean_loss))

    operator = trained_j4_operator_aggregate(
        refit.fits["J4"],
        development,
        geometry,
        campaign,
        codec_id=codec_id,
        tau_r=1e-8,
    )
    assert (
        operator.supported_chart_head_pair_count
        + operator.excluded_chart_head_pair_count
        == geometry.locator.chart_count * campaign.config.head_count
    )
    assert operator.supported_chart_head_pair_count > 0
    assert operator.positive_ambient_pair_count > 0
    assert operator.aggregate_median_ambient_operator_norm > 0.0
    assert operator.aggregate_median_h4_surrogate_operator_norm > 0.0
    assert operator.aggregate_median_ambient_stable_rank >= 0.0

    with pytest.raises(QualificationError, match="strictly positive"):
        trained_j4_operator_aggregate(
            refit.fits["J4"],
            development,
            geometry,
            campaign,
            codec_id=codec_id,
            tau_r=0.0,
        )
    incomplete_chart_universe = replace(
        development,
        active_chart_ids=development.active_chart_ids[:-1],
    )
    with pytest.raises(QualificationError, match="source universes differ"):
        trained_j4_operator_aggregate(
            refit.fits["J4"],
            incomplete_chart_universe,
            geometry,
            campaign,
            codec_id=codec_id,
            tau_r=1e-8,
        )

    manual_observations: list[OperatorObservation] = []
    codec = geometry.codec(codec_id)
    for row_index, row_key in enumerate(development.row_keys):
        if not bool(development.support_mask[row_index]):
            continue
        chart_id = int(development.chart_ids[row_index])
        z = development.z[row_index].to(torch.float64)
        for head_id in range(campaign.config.head_count):
            q_codec = codec.q_codecs[chart_id]
            message = geometry.message_sketch.representations[head_id]
            basis = message.basis.to(torch.float64)
            scale = message.coordinate_scale.to(torch.float64)
            output_weight = torch.from_numpy(
                np.asarray(campaign.source.output_weights[head_id]).copy()
            ).to(torch.float64)
            analytic = (
                q_codec.encoder.to(torch.float64)
                @ output_weight
                @ basis
                @ torch.diag(scale)
            )
            cell = refit.fits["J4"].bank.cell(chart_id, head_id)
            assert isinstance(cell, J4Model)
            activation = cell.activation_interaction.detach().to(torch.float64) @ z
            standardized = (
                analytic
                + cell.message_weight.detach().to(torch.float64)
                + (
                    cell.interaction_output.detach().to(torch.float64)
                    * activation.unsqueeze(0)
                )
                @ cell.message_interaction.detach().to(torch.float64)
            )
            ambient = (
                q_codec.decoder.to(torch.float64)
                @ standardized
                @ torch.diag(torch.reciprocal(scale))
                @ basis.T
            )
            h4_surrogate = (
                q_codec.carrier_eigenvalues.to(torch.float64).sqrt().unsqueeze(1)
                * (q_codec.carrier_vectors.to(torch.float64).T @ ambient)
            )
            manual_observations.append(
                OperatorObservation(
                    row_key,
                    chart_id,
                    head_id,
                    float(torch.linalg.matrix_norm(ambient, ord=2)),
                    float(torch.linalg.matrix_norm(h4_surrogate, ord=2)),
                    stable_rank(ambient, tolerance=1e-8),
                )
            )
    expected_operator = aggregate_operator_pair_summaries(
        summarize_operator_pairs(
            manual_observations,
            expected_pairs=tuple(
                (chart_id, head_id)
                for chart_id in range(geometry.locator.chart_count)
                for head_id in range(campaign.config.head_count)
            ),
        )
    )
    assert operator.aggregate_median_ambient_operator_norm == pytest.approx(
        expected_operator.aggregate_median_ambient_operator_norm
    )
    assert operator.aggregate_median_h4_surrogate_operator_norm == pytest.approx(
        expected_operator.aggregate_median_h4_surrogate_operator_norm
    )
    assert operator.aggregate_median_ambient_stable_rank == pytest.approx(
        expected_operator.aggregate_median_ambient_stable_rank
    )
    assert [
        (row.chart_id, row.head_id, row.analysis_unit_count, row.status)
        for row in operator.pair_summaries
    ] == [
        (row.chart_id, row.head_id, row.analysis_unit_count, row.status)
        for row in expected_operator.pair_summaries
    ]
    for observed, expected in zip(
        operator.pair_summaries,
        expected_operator.pair_summaries,
        strict=True,
    ):
        assert observed.median_ambient_operator_norm == pytest.approx(
            expected.median_ambient_operator_norm
        )
        assert observed.median_h4_surrogate_operator_norm == pytest.approx(
            expected.median_h4_surrogate_operator_norm
        )
        assert observed.median_ambient_operator_stable_rank == pytest.approx(
            expected.median_ambient_operator_stable_rank
        )

    family_by_unit = {
        item.analysis_unit_commitment: item.family_id
        for item in campaign.development_confirm.metadata
    }
    strata = [
        family_by_unit[item.analysis_unit_commitment.hex()]
        for item in j4.downstream.units
    ]
    ablation = evaluate_interaction_mean(
        refit.fits["J4"],
        j2n,
        j4,
        development,
        refit.interaction_means,
        geometry.codec(codec_id),
        campaign,
        epsilon_down=epsilon_down,
        strata=strata,
        replicates=100,
    )
    assert 0.0 <= ablation.valid_fraction <= 1.0
    assert ablation.gamma_interval is None or len(ablation.gamma_interval) == 2

    first = build_weights_only_bundles(
        refit.fits,
        baseline_id="J2N",
        geometry=geometry,
        codec_id=codec_id,
        interaction_means=refit.interaction_means,
        campaign=campaign,
    )
    second = build_weights_only_bundles(
        refit.fits,
        baseline_id="J2N",
        geometry=geometry,
        codec_id=codec_id,
        interaction_means=refit.interaction_means,
        campaign=campaign,
    )
    assert first == second
    assert all(blob[:8] != b"\x80\x04" for blob in first)
    assert all(len(hashlib.sha256(blob).hexdigest()) == 64 for blob in first)
    names = []
    for blob in first:
        header_length = int.from_bytes(blob[:8], "little")
        names.append(set(json.loads(blob[8 : 8 + header_length])))
    candidate_names, comparator_names = names
    for prefix in (
        "representation.",
        "support.",
        "message_sketch.",
        "codec.",
        "interaction_mean.",
        "source.attention_output.",
        "source.post_attention_norm.",
        "source.mlp.",
        "source.post_mlp_norm.",
        "source.lm_head.",
        "metadata.execution_protocol_",
        "J4.",
    ):
        assert any(name.startswith(prefix) for name in candidate_names)
    for prefix in ("J1.", "J2N.", "J2+.", "J3."):
        assert any(name.startswith(prefix) for name in comparator_names)

    self_test = self_test_weights_only_bundles(*first)
    assert self_test.passed
    assert self_test.row_count == 4
    assert self_test.model_max_abs_errors == (
        ("J4", 0.0),
        ("J1", 0.0),
        ("J2N", 0.0),
        ("J2+", 0.0),
        ("J3", 0.0),
    )
    assert self_test.output_sha256 == self_test_weights_only_bundles(
        *second
    ).output_sha256

    loaded = load_weights_only_bundles(*first)
    assert loaded.baseline_id in {"J1", "J2N", "J2+", "J3"}
    assert loaded.codec_id == codec_id
    assert loaded.execution_protocol_sha256 == RAW_EXECUTION_PROTOCOL_SHA256
    assert set(loaded.banks) == {"J1", "J2N", "J2+", "J3", "J4"}
    candidate_arrays = decode_safetensors(first[0])
    raw_execution = loaded.execute_raw(
        "J4",
        candidate_arrays["execution.self_test.raw_h"],
        candidate_arrays["execution.self_test.raw_message"],
    )
    assert raw_execution.h_plus.shape == (
        geometry.locator.chart_count,
        campaign.config.hidden_dimension,
    )
    assert raw_execution.logits.shape == (
        geometry.locator.chart_count,
        campaign.config.vocabulary_size,
    )
    assert torch.equal(
        raw_execution.chart_ids,
        torch.arange(geometry.locator.chart_count, dtype=torch.int64),
    )
    np.testing.assert_array_equal(
        raw_execution.h_plus.numpy(),
        candidate_arrays["execution.self_test.expected.h_plus.J4"],
    )

    weight_name = next(
        name
        for name, value in candidate_arrays.items()
        if name.startswith("J4.cells.") and np.issubdtype(value.dtype, np.floating)
    )
    changed = candidate_arrays[weight_name].copy()
    changed.flat[0] += np.float32(0.125)
    candidate_arrays[weight_name] = changed
    corrupted = encode_safetensors(candidate_arrays)
    with pytest.raises(QualificationError, match="weights identity"):
        load_weights_only_bundles(corrupted, first[1])


def test_raw_bundle_self_test_catches_source_and_sketch_mutation() -> None:
    campaign, geometry, codec_id = _qualified_fixture()
    refit = fit_refit_model_banks(
        campaign,
        geometry,
        codec_id,
        config=FAST_REFIT,
    )
    initial = fit_initial_model_banks(
        campaign,
        geometry,
        codec_id,
        config=FAST_TRAINING,
        diagnostic_config=FAST_TRAINING,
    )
    fits = dict(initial.fits)
    fits.update(refit.fits)
    candidate, comparator = build_weights_only_bundles(
        fits,
        baseline_id="J1",
        geometry=geometry,
        codec_id=codec_id,
        interaction_means=refit.interaction_means,
        campaign=campaign,
    )
    arrays = decode_safetensors(candidate)
    changed = arrays["source.post_attention_norm.weight"].copy()
    changed.flat[0] += np.asarray(0.125, dtype=changed.dtype)
    arrays["source.post_attention_norm.weight"] = changed
    corrupted = encode_safetensors(arrays)
    with pytest.raises(QualificationError, match="identity differs"):
        load_weights_only_bundles(corrupted, comparator)

    # Even a coherently rebound internal source-state identity cannot hide a
    # changed raw execution path: the independently frozen expected outputs
    # make the deterministic self-test fail.
    source_arrays = {
        name: value
        for name, value in arrays.items()
        if name.startswith("source.")
    }
    rebound = hashlib.sha256(
        b"fisher-cmc/synthetic-source-execution-state/v1\0"
        + encode_safetensors(source_arrays)
    ).digest()
    arrays["metadata.source_execution_identity_raw32"] = np.frombuffer(
        rebound, dtype=np.uint8
    ).copy()
    comparator_arrays = decode_safetensors(comparator)
    comparator_arrays["metadata.source_execution_identity_raw32"] = np.frombuffer(
        rebound, dtype=np.uint8
    ).copy()
    rebound_candidate = encode_safetensors(arrays)
    rebound_comparator = encode_safetensors(comparator_arrays)
    load_weights_only_bundles(rebound_candidate, rebound_comparator)
    result = self_test_weights_only_bundles(
        rebound_candidate, rebound_comparator
    )
    assert not result.passed
    assert all(error > 0.0 for _, error in result.model_max_abs_errors)
