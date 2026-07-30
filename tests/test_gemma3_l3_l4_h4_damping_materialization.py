from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_l3_l4_h4_damping_materialization import (
    build_gemma_h4_damping_materialization,
    build_parser,
    load_gemma_h4_damping_materialization,
    publish_gemma_h4_damping_materialization,
)
from fisher_graph.gemma3_l3_l4_h4_incremental_signal_diagnostic import (
    _DAMPING_ANALYSIS_DOMAIN,
    _DAMPING_REPORT_DOMAIN,
    _canonical_json_bytes,
    derive_gemma_h4_damping_recipe_tensors,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaTwoHeadFitSequence,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    GemmaL3L4TwoHeadArtifact,
    _tensor_sha256,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _sequences() -> tuple[GemmaTwoHeadFitSequence, ...]:
    width = 40
    length = 14
    source_rank = 2
    result: list[GemmaTwoHeadFitSequence] = []
    for index, family in enumerate(
        ("material-a", "material-b", "material-c", "material-d")
    ):
        generator = torch.Generator().manual_seed(8100 + index)
        source = torch.randn(
            length,
            source_rank,
            generator=generator,
            dtype=torch.float64,
        )
        candidate_h4 = torch.randn(
            length,
            width,
            generator=generator,
            dtype=torch.float64,
        )
        residual = torch.randn(
            length,
            width,
            generator=generator,
            dtype=torch.float64,
        )
        residual[:, :8] += 0.2 * candidate_h4[:, 8:16]
        gradient = torch.randn(
            length,
            width,
            generator=generator,
            dtype=torch.float64,
        )
        gradient[:, :8] = 0.0
        mask = torch.ones(length, dtype=torch.bool)
        zeros = torch.zeros(length, width, dtype=torch.float64)
        result.append(
            GemmaTwoHeadFitSequence(
                example_id=f"material-example-{index}",
                family_id=family,
                model_inputs_sha256=_sha(100 + index),
                runtime_binding_sha256=_sha(20),
                source_modes=source,
                logical_positions=torch.arange(length, dtype=torch.int64),
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


def _accepted_x4(
    sequences: tuple[GemmaTwoHeadFitSequence, ...],
) -> GemmaL3L4TwoHeadArtifact:
    decoder = torch.eye(40, dtype=torch.float64)[:2]
    head = GemmaCausalResidualHead(
        site="layer.4.mlp.normalized_input",
        parent_runtime_binding_sha256=_sha(19),
        residual_map_sha256=_sha(21),
        analysis_artifact_sha256=_sha(22),
        fit_manifest_sha256=_sha(23),
        bridge_binding_sha256=_sha(24),
        decoder=decoder,
        lag_kernel=torch.zeros(1, 2, 2, dtype=torch.float64),
        state_kernel=torch.empty(0, 0, dtype=torch.float64),
        conditioning="l3_source_modes",
        ridge=1.0e-6,
        fit_row_count=56,
        family_ids=tuple(sorted({value.family_id for value in sequences})),
        fit_sequence_sha256s=tuple(
            sorted(value.artifact_sha256 for value in sequences)
        ),
        fit_objective="hidden_residual_ridge",
        weighted_residual_rmse=0.0,
        normalized_nll_direction_rmse=0.0,
        linearized_nll_residual_rmse=0.0,
    )
    return GemmaL3L4TwoHeadArtifact(
        parent_artifact_sha256=_sha(30),
        parent_receipt_sha256=_sha(31),
        residual_map_sha256=_sha(32),
        analysis_artifact_sha256=_sha(33),
        bridge_binding_sha256=_sha(24),
        live_model_sha256=_sha(34),
        adapter_execution_sha256=_sha(35),
        heads=(head,),
        recipe_sha256=_sha(36),
    )


def _damping_report(
    *,
    sequences: tuple[GemmaTwoHeadFitSequence, ...],
    parent: GemmaL3L4TwoHeadArtifact,
    decoder: torch.Tensor,
) -> dict[str, object]:
    _tensors, recipe = derive_gemma_h4_damping_recipe_tensors(
        sequences=sequences,
        output_decoder=decoder,
        lag_count=16,
        input_rank=32,
        state_scale=0.5,
        ridge=1.0e-6,
    )
    diagnostic: dict[str, object] = {
        "input": {
            "fit_sequence_sha256s": tuple(
                value.artifact_sha256 for value in sequences
            ),
            "family_ids": tuple(
                sorted({value.family_id for value in sequences})
            ),
            "affected_row_count": sum(
                value.affected_rows for value in sequences
            ),
            "output_decoder_sha256": _tensor_sha256(decoder),
        },
        "selection": {
            "status": "fit_only_damping_recipe_frozen",
            "winning_recipe": recipe,
        },
    }
    diagnostic["analysis_sha256"] = _domain_sha256(
        _DAMPING_ANALYSIS_DOMAIN,
        diagnostic,
    )
    report: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_l3_l4_h4_incremental_signal_damping"
        ),
        "format_version": 1,
        "spec": {
            "fixed_head": {
                "encoder_kind": "independent_crossfit_h4_svd",
                "input_rank": 32,
                "lag_count": 16,
                "output_rank": 8,
            },
            "ridge": 1.0e-6,
            "fit_manifest_sha256": _sha(23),
            "selection_input_accepted": False,
            "guard_input_accepted": False,
        },
        "accepted_x4_provenance": {
            "candidate_artifact_sha256": parent.artifact_sha256,
            "candidate_execution_sha256": parent.execution_sha256,
            "candidate_runtime_binding_sha256": (
                parent.runtime_binding_sha256
            ),
        },
        "diagnostic": diagnostic,
        "safety": {
            "fit_role_opened": True,
            "selection_role_opened": False,
            "guard_role_opened": False,
            "calibration_b_opened": False,
        },
    }
    report["report_sha256"] = _domain_sha256(
        _DAMPING_REPORT_DOMAIN,
        report,
    )
    return report


@pytest.fixture(scope="module")
def materialization_inputs() -> tuple[
    tuple[GemmaTwoHeadFitSequence, ...],
    torch.Tensor,
    GemmaL3L4TwoHeadArtifact,
    dict[str, object],
]:
    sequences = _sequences()
    decoder = torch.eye(40, dtype=torch.float64)[:8]
    parent = _accepted_x4(sequences)
    sequences = tuple(
        replace(
            value,
            runtime_binding_sha256=parent.runtime_binding_sha256,
        )
        for value in sequences
    )
    return sequences, decoder, parent, _damping_report(
        sequences=sequences,
        parent=parent,
        decoder=decoder,
    )


def test_materialization_reproduces_fixed_recipe_and_distinct_arms(
    materialization_inputs: tuple[
        tuple[GemmaTwoHeadFitSequence, ...],
        torch.Tensor,
        GemmaL3L4TwoHeadArtifact,
        dict[str, object],
    ],
) -> None:
    sequences, decoder, parent, report = materialization_inputs
    result = build_gemma_h4_damping_materialization(
        sequences=tuple(reversed(sequences)),
        output_decoder=decoder,
        accepted_x4_artifact=parent,
        accepted_x4_receipt_sha256=_sha(40),
        damping_report=report,
        expected_damping_report_sha256=str(report["report_sha256"]),
    )

    alpha0 = result.alpha0_artifact.head("layer.4.output")
    challenger = result.alpha0_5_artifact.head("layer.4.output")
    assert alpha0 is not None
    assert challenger is not None
    assert alpha0.conditioning == "l3_source_modes"
    assert alpha0.state_encoder is None
    assert challenger.conditioning == (
        "l3_source_modes_plus_independent_realized_h4_modes_v1"
    )
    assert challenger.state_encoder is not None
    assert challenger.state_encoder.shape == (32, 40)
    assert challenger.state_kernel.shape == (32, 8)
    assert challenger.lag_kernel.shape == (16, 2, 8)
    assert alpha0.lag_kernel.shape == (16, 2, 8)
    assert (
        result.alpha0_artifact.head(
            "layer.4.mlp.normalized_input"
        ).artifact_sha256
        == parent.head(
            "layer.4.mlp.normalized_input"
        ).artifact_sha256
    )
    assert result.report_payload["coefficient_reproduction"]["status"] == (
        "exact_hash_match"
    )
    assert result.report_payload["safety"][
        "alternate_alpha_fallback_present"
    ] is False
    assert "material-example" not in json.dumps(
        result.report_payload,
        sort_keys=True,
    )


def test_materialization_fails_closed_on_frozen_hash_drift(
    materialization_inputs: tuple[
        tuple[GemmaTwoHeadFitSequence, ...],
        torch.Tensor,
        GemmaL3L4TwoHeadArtifact,
        dict[str, object],
    ],
) -> None:
    sequences, decoder, parent, report = materialization_inputs
    tampered = json.loads(json.dumps(report))
    tampered["diagnostic"]["selection"]["winning_recipe"][
        "state_kernel_sha256"
    ] = _sha(999)
    diagnostic = tampered["diagnostic"]
    diagnostic.pop("analysis_sha256")
    diagnostic["analysis_sha256"] = _domain_sha256(
        _DAMPING_ANALYSIS_DOMAIN,
        diagnostic,
    )
    tampered.pop("report_sha256")
    tampered["report_sha256"] = _domain_sha256(
        _DAMPING_REPORT_DOMAIN,
        tampered,
    )

    with pytest.raises(RuntimeError, match="did not reproduce"):
        build_gemma_h4_damping_materialization(
            sequences=sequences,
            output_decoder=decoder,
            accepted_x4_artifact=parent,
            accepted_x4_receipt_sha256=_sha(40),
            damping_report=tampered,
        )


def test_materialization_publication_round_trips_tensor_only_artifacts(
    tmp_path,
    materialization_inputs: tuple[
        tuple[GemmaTwoHeadFitSequence, ...],
        torch.Tensor,
        GemmaL3L4TwoHeadArtifact,
        dict[str, object],
    ],
) -> None:
    sequences, decoder, parent, report = materialization_inputs
    result = build_gemma_h4_damping_materialization(
        sequences=sequences,
        output_decoder=decoder,
        accepted_x4_artifact=parent,
        accepted_x4_receipt_sha256=_sha(40),
        damping_report=report,
    )
    output = tmp_path / "materialization.report.json"
    published = publish_gemma_h4_damping_materialization(
        result,
        output=output,
    )

    assert output.exists()
    assert output.with_suffix(".alpha0.candidate.pt").exists()
    assert output.with_suffix(".alpha0_5.candidate.pt").exists()
    assert published["files"]["contains_model_weights"] is False
    assert published["safety"]["coefficient_tensors_in_report"] is False
    loaded, loaded_report = load_gemma_h4_damping_materialization(
        output,
        expected_report_sha256=str(published["report_sha256"]),
        expected_report_file_sha256=hashlib.sha256(
            output.read_bytes()
        ).hexdigest(),
    )
    assert (
        loaded.alpha0_artifact.artifact_sha256
        == result.alpha0_artifact.artifact_sha256
    )
    assert (
        loaded.alpha0_5_artifact.artifact_sha256
        == result.alpha0_5_artifact.artifact_sha256
    )
    assert loaded_report == published
    with pytest.raises(FileExistsError):
        publish_gemma_h4_damping_materialization(result, output=output)


def test_materialization_publication_rejects_report_mutation(
    tmp_path,
    materialization_inputs: tuple[
        tuple[GemmaTwoHeadFitSequence, ...],
        torch.Tensor,
        GemmaL3L4TwoHeadArtifact,
        dict[str, object],
    ],
) -> None:
    sequences, decoder, parent, report = materialization_inputs
    result = build_gemma_h4_damping_materialization(
        sequences=sequences,
        output_decoder=decoder,
        accepted_x4_artifact=parent,
        accepted_x4_receipt_sha256=_sha(40),
        damping_report=report,
    )
    result.report_payload["artifacts"]["matched_alpha0"][  # type: ignore[index]
        "h4_prepared_float_scalar_count"
    ] = 1

    with pytest.raises(RuntimeError, match="payload drifted"):
        publish_gemma_h4_damping_materialization(
            result,
            output=tmp_path / "mutated.report.json",
        )


def test_materialization_parser_exposes_no_assessment_capability() -> None:
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--fit-input" in options
    assert "--selection-input" not in options
    assert "--guard-input" not in options
    assert "--calibration-b" not in options
