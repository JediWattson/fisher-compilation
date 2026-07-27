from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.prompt_mode_tracing import (
    PromptModeTrace,
    PromptModeTraceProvenance,
    collect_prompt_mode_trace,
    prompt_example_id_sha256,
)
from fisher_graph.parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPLayerParameterSpec,
    build_grouped_virtual_gate_fisher_from_trace,
    build_natural_mlp_parameter_group_catalog,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
)


def _provenance() -> PromptModeTraceProvenance:
    return PromptModeTraceProvenance(
        source_model_fingerprint="a" * 64,
        calibration_split_sha256="b" * 64,
        objective_sha256="c" * 64,
        score_reduction="sum_log_probability",
        normalizer="independent_prompt",
    )


def _specs() -> tuple[CrossBlockLayerSpec, ...]:
    return (
        CrossBlockLayerSpec(
            layer_id="layer.0",
            layer_ordinal=0,
            activation_site="layer.0.mlp.down_input",
            width=2,
        ),
        CrossBlockLayerSpec(
            layer_id="layer.1",
            layer_ordinal=1,
            activation_site="layer.1.mlp.down_input",
            width=1,
        ),
    )


def _row(
    *,
    example_id: str | None,
    first_activation: torch.Tensor,
    first_gradient: torch.Tensor,
    second_activation: torch.Tensor,
    second_gradient: torch.Tensor,
    loss: float,
) -> ActivationScoreGradientRows:
    observations = first_activation.shape[0]
    assert second_activation.shape[0] == observations
    # Reverse mapping insertion order to verify that the explicit catalog,
    # rather than mapping iteration, fixes the global mode coordinates.
    return ActivationScoreGradientRows(
        activations={
            "layer.1.mlp.down_input": second_activation,
            "layer.0.mlp.down_input": first_activation,
        },
        score_gradients={
            "layer.1.mlp.down_input": second_gradient,
            "layer.0.mlp.down_input": first_gradient,
        },
        logical_positions=torch.arange(observations, dtype=torch.int64),
        loss=loss,
        example_id=example_id,
    )


def _rows() -> tuple[ActivationScoreGradientRows, ...]:
    return (
        _row(
            example_id="fit/prompt-alpha",
            first_activation=torch.tensor(
                [[1.0, -2.0], [3.0, 0.0]],
                dtype=torch.float32,
            ),
            first_gradient=torch.tensor(
                [[2.0, 1.0], [-1.0, 4.0]],
                dtype=torch.float32,
            ),
            second_activation=torch.tensor(
                [[-1.0], [2.0]],
                dtype=torch.float32,
            ),
            second_gradient=torch.tensor(
                [[3.0], [-2.0]],
                dtype=torch.float32,
            ),
            loss=1.25,
        ),
        _row(
            example_id="fit/prompt-beta",
            first_activation=torch.tensor(
                [[-2.0, 4.0]],
                dtype=torch.float64,
            ),
            first_gradient=torch.tensor(
                [[0.5, -0.25]],
                dtype=torch.float64,
            ),
            second_activation=torch.tensor(
                [[3.0]],
                dtype=torch.float64,
            ),
            second_gradient=torch.tensor(
                [[2.0]],
                dtype=torch.float64,
            ),
            loss=0.75,
        ),
    )


def _trace() -> PromptModeTrace:
    return collect_prompt_mode_trace(
        _rows(),
        layer_specs=_specs(),
        provenance=_provenance(),
    )


def _parameter_catalog():
    return build_natural_mlp_parameter_group_catalog(
        model_fingerprint=_provenance().source_model_fingerprint,
        layer_specs=tuple(
            NaturalMLPLayerParameterSpec.from_cross_block_layer_spec(
                spec,
                input_width=4,
                output_width=4,
                parameter_prefix=f"{spec.layer_id}.mlp",
            )
            for spec in _specs()
        ),
    )


def _all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            string
            for key, item in value.items()
            for string in (*_all_strings(key), *_all_strings(item))
        ]
    if isinstance(value, (tuple, list)):
        return [
            string
            for item in value
            for string in _all_strings(item)
        ]
    return []


def test_exact_prompt_conditioned_summaries_and_catalog_order() -> None:
    trace = _trace()

    assert trace.example_count == 2
    assert trace.mode_count == 3
    assert trace.layer_specs == _specs()
    assert trace.site_slice("layer.0.mlp.down_input") == slice(0, 2)
    assert trace.site_slice("layer.1.mlp.down_input") == slice(2, 3)
    with pytest.raises(KeyError, match="unknown activation site"):
        trace.site_slice("missing.site")

    torch.testing.assert_close(
        trace.prompt_effects,
        torch.tensor(
            [[-1.0, -2.0, -7.0], [-1.0, -1.0, 6.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        trace.additive_fisher_need,
        torch.tensor(
            [[13.0, 4.0, 25.0], [1.0, 1.0, 36.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        trace.activation_means,
        torch.tensor(
            [[2.0, -1.0, 0.5], [-2.0, 4.0, 3.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        trace.activation_rms,
        torch.tensor(
            [
                [5.0**0.5, 2.0**0.5, 2.5**0.5],
                [2.0, 4.0, 3.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        trace.activation_positive_fractions,
        torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    assert trace.valid_row_counts.tolist() == [2, 1]
    torch.testing.assert_close(
        trace.losses,
        torch.tensor([1.25, 0.75], dtype=torch.float64),
    )
    assert trace.mean_loss == 1.0
    assert all(
        tensor.dtype == torch.float64
        for tensor in (
            trace.prompt_effects,
            trace.additive_fisher_need,
            trace.activation_means,
            trace.activation_rms,
            trace.activation_positive_fractions,
            trace.losses,
        )
    )


def test_grouped_fisher_constructor_authenticates_source_trace() -> None:
    trace = _trace()
    fisher = build_grouped_virtual_gate_fisher_from_trace(
        trace,
        catalog=_parameter_catalog(),
    )

    assert fisher.source_prompt_trace_sha256 == trace.artifact_sha256
    assert fisher.metadata()["source_trace_authenticated"] is True
    assert (
        GroupedVirtualGateFisher.from_state_dict(fisher.state_dict())
        .artifact_sha256
        == fisher.artifact_sha256
    )

    wrong_catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint="d" * 64,
        layer_specs=_parameter_catalog().layer_specs,
    )
    with pytest.raises(ValueError, match="do not align"):
        build_grouped_virtual_gate_fisher_from_trace(
            trace,
            catalog=wrong_catalog,
        )


def test_coherent_gate_effect_and_token_additive_need_are_distinct() -> None:
    trace = _trace()

    # Coordinate zero has token effects +2 and -3.  The virtual gate
    # derivative is their coherent sum (-1), while the additive need is
    # 2^2 + (-3)^2 = 13.
    assert trace.prompt_effects[0, 0].item() == -1.0
    assert trace.prompt_effects[0, 0].square().item() == 1.0
    assert trace.additive_fisher_need[0, 0].item() == 13.0

    # Coordinate one has token effects -2 and 0: no cross-position
    # cancellation, so the two definitions happen to agree after squaring.
    assert trace.prompt_effects[0, 1].square().item() == 4.0
    assert trace.additive_fisher_need[0, 1].item() == 4.0


def test_artifact_round_trip_is_deterministic_and_source_safe() -> None:
    first = _trace()
    second = _trace()

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.metadata() == second.metadata()
    assert first.example_id_sha256s == (
        prompt_example_id_sha256("fit/prompt-alpha"),
        prompt_example_id_sha256("fit/prompt-beta"),
    )
    restored = PromptModeTrace.from_state_dict(first.state_dict())
    assert restored.artifact_sha256 == first.artifact_sha256
    assert restored.metadata() == first.metadata()
    for field in (
        "prompt_effects",
        "additive_fisher_need",
        "activation_means",
        "activation_rms",
        "activation_positive_fractions",
        "valid_row_counts",
        "losses",
    ):
        torch.testing.assert_close(
            getattr(restored, field),
            getattr(first, field),
        )

    metadata = first.metadata()
    assert metadata["contains_source_model_weights"] is False
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_raw_example_ids"] is False
    assert metadata["contains_token_ids"] is False
    assert metadata["contains_decoded_tokens"] is False
    assert metadata["contains_context_windows"] is False
    assert metadata["contains_token_row_activations"] is False
    assert metadata["contains_token_row_gradients"] is False
    assert metadata["analysis_only"] is True
    assert metadata["authorizes_intervention"] is False
    assert metadata["authorizes_compilation"] is False
    assert metadata["authorizes_execution"] is False

    strings = _all_strings(first.state_dict())
    assert "fit/prompt-alpha" not in strings
    assert "fit/prompt-beta" not in strings
    assert not any("decoded" in value for value in strings if value.startswith("fit/"))

    # Serialized tensor values are defensive copies.
    state = first.state_dict()
    state["prompt_effects"][0, 0] = 123.0
    assert first.prompt_effects[0, 0].item() == -1.0


def test_artifact_rejects_tensor_metadata_and_hash_poisoning() -> None:
    trace = _trace()

    changed_tensor = copy.deepcopy(trace.state_dict())
    changed_tensor["prompt_effects"][0, 0] += 0.5
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        PromptModeTrace.from_state_dict(changed_tensor)

    changed_id = copy.deepcopy(trace.state_dict())
    changed_id["example_id_sha256s"] = (
        "d" * 64,
        changed_id["example_id_sha256s"][1],
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        PromptModeTrace.from_state_dict(changed_id)

    changed_hash = copy.deepcopy(trace.state_dict())
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        PromptModeTrace.from_state_dict(changed_hash)

    changed_safety = copy.deepcopy(trace.state_dict())
    changed_safety["contains_prompt_text"] = True
    with pytest.raises(ValueError, match="safety metadata"):
        PromptModeTrace.from_state_dict(changed_safety)

    changed_dtype = copy.deepcopy(trace.state_dict())
    changed_dtype["activation_means"] = changed_dtype[
        "activation_means"
    ].to(torch.float32)
    with pytest.raises(ValueError, match="activation_means"):
        PromptModeTrace.from_state_dict(changed_dtype)

    boolean_version = copy.deepcopy(trace.state_dict())
    boolean_version["format_version"] = True
    with pytest.raises(ValueError, match="safety metadata"):
        PromptModeTrace.from_state_dict(boolean_version)

    coerced_spec = copy.deepcopy(trace.state_dict())
    coerced_spec["layer_specs"][0]["width"] = "2"
    with pytest.raises(TypeError, match="layer spec types"):
        PromptModeTrace.from_state_dict(coerced_spec)

    unexpected = copy.deepcopy(trace.state_dict())
    unexpected["prompt_text"] = "must never be accepted"
    with pytest.raises(ValueError, match="fields are invalid"):
        PromptModeTrace.from_state_dict(unexpected)


def test_collection_rejects_empty_duplicate_and_missing_example_ids() -> None:
    with pytest.raises(ValueError, match="empty prompt row stream"):
        collect_prompt_mode_trace(
            (),
            layer_specs=_specs(),
            provenance=_provenance(),
        )

    missing = _rows()[0]
    missing = _row(
        example_id=None,
        first_activation=missing.activations[
            "layer.0.mlp.down_input"
        ],
        first_gradient=missing.score_gradients[
            "layer.0.mlp.down_input"
        ],
        second_activation=missing.activations[
            "layer.1.mlp.down_input"
        ],
        second_gradient=missing.score_gradients[
            "layer.1.mlp.down_input"
        ],
        loss=missing.loss,
    )
    with pytest.raises(ValueError, match="missing an example_id"):
        collect_prompt_mode_trace(
            (missing,),
            layer_specs=_specs(),
            provenance=_provenance(),
        )

    duplicate = _row(
        example_id="fit/prompt-alpha",
        first_activation=_rows()[1].activations[
            "layer.0.mlp.down_input"
        ],
        first_gradient=_rows()[1].score_gradients[
            "layer.0.mlp.down_input"
        ],
        second_activation=_rows()[1].activations[
            "layer.1.mlp.down_input"
        ],
        second_gradient=_rows()[1].score_gradients[
            "layer.1.mlp.down_input"
        ],
        loss=0.5,
    )
    with pytest.raises(ValueError, match="duplicate example_id"):
        collect_prompt_mode_trace(
            (_rows()[0], duplicate),
            layer_specs=_specs(),
            provenance=_provenance(),
        )


def test_collection_rejects_site_width_and_catalog_drift() -> None:
    only_first = ActivationScoreGradientRows(
        activations={
            "layer.0.mlp.down_input": torch.ones(
                2,
                2,
                dtype=torch.float64,
            )
        },
        score_gradients={
            "layer.0.mlp.down_input": torch.ones(
                2,
                2,
                dtype=torch.float64,
            )
        },
        logical_positions=torch.arange(2, dtype=torch.int64),
        loss=0.0,
        example_id="fit/missing-site",
    )
    with pytest.raises(ValueError, match="sites do not match"):
        collect_prompt_mode_trace(
            (only_first,),
            layer_specs=_specs(),
            provenance=_provenance(),
        )

    wrong_width = (
        _specs()[0],
        CrossBlockLayerSpec(
            layer_id="layer.1",
            layer_ordinal=1,
            activation_site="layer.1.mlp.down_input",
            width=2,
        ),
    )
    with pytest.raises(ValueError, match="does not match catalog width"):
        collect_prompt_mode_trace(
            (_rows()[0],),
            layer_specs=wrong_width,
            provenance=_provenance(),
        )

    with pytest.raises(ValueError, match="canonical layer/site order"):
        collect_prompt_mode_trace(
            (_rows()[0],),
            layer_specs=tuple(reversed(_specs())),
            provenance=_provenance(),
        )

    duplicate_site = (
        _specs()[0],
        CrossBlockLayerSpec(
            layer_id="layer.0",
            layer_ordinal=0,
            activation_site="layer.0.mlp.down_input",
            width=2,
        ),
    )
    with pytest.raises(ValueError, match="activation sites must be unique"):
        collect_prompt_mode_trace(
            (_rows()[0],),
            layer_specs=duplicate_site,
            provenance=_provenance(),
        )


def test_catalog_allows_multiple_ordered_sites_in_one_layer() -> None:
    specs = (
        CrossBlockLayerSpec(
            layer_id="layer.0",
            layer_ordinal=0,
            activation_site="layer.0.mlp.gate",
            width=1,
        ),
        CrossBlockLayerSpec(
            layer_id="layer.0",
            layer_ordinal=0,
            activation_site="layer.0.mlp.up",
            width=1,
        ),
    )
    row = ActivationScoreGradientRows(
        activations={
            "layer.0.mlp.up": torch.tensor([[2.0]], dtype=torch.float64),
            "layer.0.mlp.gate": torch.tensor([[3.0]], dtype=torch.float64),
        },
        score_gradients={
            "layer.0.mlp.up": torch.tensor([[5.0]], dtype=torch.float64),
            "layer.0.mlp.gate": torch.tensor([[7.0]], dtype=torch.float64),
        },
        logical_positions=torch.tensor([0], dtype=torch.int64),
        loss=0.0,
        example_id="fit/two-sites",
    )

    trace = collect_prompt_mode_trace(
        (row,),
        layer_specs=specs,
        provenance=_provenance(),
    )
    torch.testing.assert_close(
        trace.prompt_effects,
        torch.tensor([[21.0, 10.0]], dtype=torch.float64),
    )
