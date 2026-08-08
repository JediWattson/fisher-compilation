import pytest
import torch

from fisher_graph.complete_block_residual_forms import (
    CompleteBlockResidualForm,
    ResidualForm,
)
from test_structured_transformer_layer_executor import _executor, _sequence


def _forms() -> dict[ResidualForm, CompleteBlockResidualForm]:
    source = _executor()
    state = source.state_dict()
    result = {}
    for form in ResidualForm:
        engine = _executor()
        engine.load_state_dict(state, strict=True)
        result[form] = CompleteBlockResidualForm(engine, form).eval()
    return result


def test_explicit_and_direct_forms_match_and_authenticate_identities() -> None:
    forms = _forms()
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    sequence = _sequence(mask)
    torch.manual_seed(611)
    hidden = torch.randn(2, 5, 8)

    explicit = forms[ResidualForm.EXPLICIT].forward_components(
        hidden,
        sequence,
    )
    direct = forms[ResidualForm.DIRECT_OUTPUT].forward_components(
        hidden,
        sequence,
    )

    torch.testing.assert_close(explicit.output, direct.output, rtol=0, atol=0)
    torch.testing.assert_close(
        explicit.post_attention,
        hidden + explicit.attention_delta,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        explicit.output,
        explicit.post_attention + explicit.feed_forward_delta,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        direct.output,
        hidden + direct.attention_delta + direct.feed_forward_delta,
        rtol=0,
        atol=0,
    )
    assert explicit.attention_delta[mask].norm() > 0
    assert explicit.feed_forward_delta[mask].norm() > 0


def test_direct_form_does_not_delegate_to_explicit_executor_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    form = _forms()[ResidualForm.DIRECT_OUTPUT]

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("direct form delegated to explicit flow")

    monkeypatch.setattr(form.executor, "forward_components", forbidden)
    mask = torch.ones(1, 4, dtype=torch.bool)
    output = form(torch.randn(1, 4, 8), _sequence(mask))
    assert output.shape == (1, 4, 8)


def test_identity_and_branch_deletion_controls_are_nonvacuous() -> None:
    forms = _forms()
    mask = torch.ones(2, 6, dtype=torch.bool)
    sequence = _sequence(mask)
    torch.manual_seed(991)
    hidden = torch.randn(2, 6, 8)
    exact = forms[ResidualForm.DIRECT_OUTPUT].forward_components(
        hidden,
        sequence,
    )

    controls = {
        form: candidate.forward_components(hidden, sequence)
        for form, candidate in forms.items()
        if form not in (ResidualForm.EXPLICIT, ResidualForm.DIRECT_OUTPUT)
    }
    for execution in controls.values():
        assert not torch.allclose(execution.output[mask], exact.output[mask])

    drop_block = controls[ResidualForm.DROP_BLOCK_IDENTITY]
    torch.testing.assert_close(
        drop_block.output,
        drop_block.attention_delta + drop_block.feed_forward_delta,
    )
    drop_first = controls[ResidualForm.DROP_ATTENTION_IDENTITY]
    torch.testing.assert_close(
        drop_first.post_attention,
        drop_first.attention_delta,
    )
    drop_second = controls[ResidualForm.DROP_FEED_FORWARD_IDENTITY]
    torch.testing.assert_close(
        drop_second.output,
        drop_second.feed_forward_delta,
    )


def test_every_control_preserves_invalid_padding_rows() -> None:
    forms = _forms()
    mask = torch.tensor([[True, True, True, False, False]])
    sequence = _sequence(mask)
    hidden = torch.randn(1, 5, 8)

    for graph in forms.values():
        output = graph(hidden, sequence)
        torch.testing.assert_close(output[~mask], hidden[~mask], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, True, True, False, False]]),
        torch.tensor([[False, False, True, True, True]]),
    ],
)
@pytest.mark.parametrize(
    "form",
    [ResidualForm.EXPLICIT, ResidualForm.DIRECT_OUTPUT],
)
def test_padding_values_cannot_poison_valid_rows(
    mask: torch.Tensor,
    form: ResidualForm,
) -> None:
    graph = _forms()[form]
    sequence = _sequence(mask)
    hidden = torch.randn(1, 5, 8)
    poisoned = hidden.clone()
    poisoned[~mask] = 1.0e4 * torch.randn_like(poisoned[~mask])

    baseline = graph(hidden, sequence)
    changed = graph(poisoned, sequence)
    torch.testing.assert_close(baseline[mask], changed[mask], rtol=0, atol=0)


@pytest.mark.parametrize(
    "form",
    [ResidualForm.EXPLICIT, ResidualForm.DIRECT_OUTPUT],
)
def test_exact_forms_are_causal_and_padding_rows_are_identity(
    form: ResidualForm,
) -> None:
    graph = _forms()[form]
    mask = torch.tensor([[True, True, True, True, False, False]])
    sequence = _sequence(mask)
    torch.manual_seed(1201)
    hidden = torch.randn(1, 6, 8)
    changed = hidden.clone()
    changed[:, 3] += 100.0
    baseline = graph(hidden, sequence)
    perturbed = graph(changed, sequence)

    torch.testing.assert_close(
        baseline[:, :3],
        perturbed[:, :3],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(baseline[~mask], hidden[~mask], rtol=0, atol=0)


def test_manifests_distinguish_public_fusion_from_arithmetic_removal() -> None:
    forms = _forms()
    explicit = forms[ResidualForm.EXPLICIT].graph_manifest()
    direct = forms[ResidualForm.DIRECT_OUTPUT].graph_manifest()

    assert explicit["public_standalone_residual_add_nodes"] == 2
    assert direct["public_standalone_residual_add_nodes"] == 0
    assert direct["embedded_identity_combine_count"] == 2
    assert direct["all_native_identity_edges_retained"] is True
    assert direct["identity_arithmetic_removed"] is False
    assert direct["compression_attempted"] is False
    assert forms[
        ResidualForm.ZERO_FEED_FORWARD_BRANCH
    ].graph_manifest()["embedded_identity_combine_count"] == 1
    assert hasattr(forms[ResidualForm.EXPLICIT], "attention_residual_add")
    assert not hasattr(
        forms[ResidualForm.DIRECT_OUTPUT],
        "attention_residual_add",
    )
    assert hasattr(
        forms[ResidualForm.DIRECT_OUTPUT],
        "attention_state_generator",
    )
    assert forms[ResidualForm.EXPLICIT].learned_parameter_count == forms[
        ResidualForm.DIRECT_OUTPUT
    ].learned_parameter_count


def test_invalid_form_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported complete-block"):
        CompleteBlockResidualForm(_executor(), "not-a-form")
