from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph import gemma3_joint_cross_block_chart_executor as joint
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)
from fisher_graph.gemma3_chart_mode_surfaces import (
    build_attention_postnorm_bridge_receipt,
)


_WIDTH = 4
_VOCABULARY = 7


class _CopyNorm(nn.Module):
    """A parameter-free identity that still creates a distinct autograd node."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 1.0


class _CausalToyOperator(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        sequence: SequenceContext,
    ) -> torch.Tensor:
        valid = sequence.query_valid_mask.unsqueeze(-1)
        prefix = value.masked_fill(~valid, 0).cumsum(dim=1)
        denominator = torch.arange(
            1,
            value.shape[1] + 1,
            dtype=value.dtype,
            device=value.device,
        ).view(1, -1, 1)
        return value + 0.01 * prefix / denominator


class _SmallDelta(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 0.01 * torch.tanh(value)


def _sequence(*, valid: torch.Tensor | None = None) -> SequenceContext:
    if valid is None:
        valid = torch.ones(2, 3, dtype=torch.bool)
    positions = torch.arange(valid.shape[1]).expand(valid.shape[0], -1)
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions.clone(),
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _exact_executor(
    *,
    final_logit_softcap: float | None = None,
) -> joint.Gemma3JointCrossBlockChartExecutor:
    lexical_weight = torch.arange(
        _VOCABULARY * _WIDTH,
        dtype=torch.float32,
    ).reshape(_VOCABULARY, _WIDTH) / 100.0
    lexical = SharedTiedLexicalFactor(exact_weight=lexical_weight)
    mutations: list[joint.JointGraphGeneratorMutation] = []
    for event in joint.build_joint_cross_block_event_schedule()[1:-2]:
        generator = joint.ExactTokenLocalChartGenerator(
            operator=_SmallDelta(),
            hidden_width=_WIDTH,
            chart_count=1,
            generator_rank=_WIDTH,
        )
        mutations.append(
            joint.JointGraphGeneratorMutation(
                event_spec=event,
                input_norm=_CopyNorm(),
                chart_generator=generator,
                output_norm=_CopyNorm(),
                sequence_operator=(
                    _CausalToyOperator()
                    if event.event_kind == "attention_mutation"
                    else None
                ),
                source_checkpoint_coefficients_retained=True,
            )
        )
    return joint.Gemma3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=tuple(mutations),
        transports=(),
        terminal_norm=_CopyNorm(),
        final_logit_softcap=final_logit_softcap,
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )


def _expected_site_ids() -> tuple[str, ...]:
    return (
        "lexical.ingress.output",
        *(
            site_id
            for index in range(1, joint.PINNED_GEMMA3_MUTATION_COUNT + 1)
            for site_id in (
                f"mutation.{index}.input",
                f"mutation.{index}.output",
            )
        ),
    )


def _state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_state_unchanged(
    before: Mapping[str, torch.Tensor],
    module: nn.Module,
) -> None:
    after = module.state_dict()
    assert tuple(after) == tuple(before)
    for name, expected in before.items():
        assert torch.equal(after[name], expected), name


def _assert_same_execution(
    left: joint.JointGraphWholeModelExecution,
    right: joint.JointGraphWholeModelExecution,
) -> None:
    assert torch.equal(left.output, right.output)
    assert torch.equal(left.normalized_output, right.normalized_output)
    assert torch.equal(left.logits, right.logits)
    assert left.event_ids == right.event_ids
    assert left.mutation_receipts == right.mutation_receipts
    assert left.operator_receipts == right.operator_receipts
    assert left.carrier_version == right.carrier_version == 36
    assert left.transport_edge_count == right.transport_edge_count == 0
    assert left.peak_live_transport_packets == right.peak_live_transport_packets
    assert (
        left.peak_live_transport_packet_bytes
        == right.peak_live_transport_packet_bytes
    )


def test_observer_exports_exactly_73_detached_sites_without_a_write_path() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    model_before = _state_snapshot(executor)
    baseline = executor.forward_components(input_ids, sequence)
    records: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []

    def observer(
        *, event_id: str, role: str, values: torch.Tensor
    ) -> object:
        assert values.requires_grad is False
        assert values.grad_fn is None
        records.append((event_id, role, values.clone(), values))
        values.fill_(float("nan"))
        return {"attempted_replacement": torch.full_like(values, 123.0)}

    observed = executor.forward_components_with_observer(
        input_ids,
        sequence,
        observer=observer,
    )

    _assert_same_execution(baseline, observed.execution)
    _assert_state_unchanged(model_before, executor)
    assert observed.gradient_tape is None
    assert len(records) == 73
    assert tuple(f"{event_id}.{role}" for event_id, role, _, _ in records) == (
        _expected_site_ids()
    )
    assert len({snapshot.data_ptr() for _, _, _, snapshot in records}) == 73
    assert all(clean.shape == (2, 3, _WIDTH) for _, _, clean, _ in records)
    assert torch.isfinite(
        torch.stack([clean.sum() for _, _, clean, _ in records])
    ).all()
    assert torch.isfinite(observed.execution.logits).all()


def test_observer_abort_does_not_poison_a_subsequent_exact_execution() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    expected = executor.forward_components(input_ids, sequence)
    before = _state_snapshot(executor)

    def reject(**_: object) -> None:
        raise RuntimeError("observer rejected packet")

    with pytest.raises(RuntimeError, match="observer rejected packet"):
        executor.forward_components_with_observer(
            input_ids,
            sequence,
            observer=reject,
        )

    recovered = executor.forward_components(input_ids, sequence)
    _assert_same_execution(expected, recovered)
    _assert_state_unchanged(before, executor)


def test_gradient_tape_scores_all_sites_once_without_exposing_live_values() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    callbacks: list[torch.Tensor] = []

    observed = executor.forward_components_with_observer(
        input_ids,
        _sequence(),
        observer=lambda **payload: callbacks.append(payload["values"]),
        retain_gradient_tape=True,
    )

    assert len(callbacks) == 73
    assert all(not value.requires_grad and value.grad_fn is None for value in callbacks)
    tape = observed.gradient_tape
    assert tape is not None
    assert tape.site_ids == _expected_site_ids()
    objective = observed.execution.logits.float().square().mean()
    gradients = tape.score_gradients(objective)

    assert tuple(gradients) == _expected_site_ids()
    assert len(gradients) == 73
    assert all(value.device.type == "cpu" for value in gradients.values())
    assert all(value.dtype == torch.float32 for value in gradients.values())
    assert all(value.shape == (2, 3, _WIDTH) for value in gradients.values())
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert tape.consumed is True
    with pytest.raises(RuntimeError, match="already consumed"):
        tape.score_gradients(objective)


def test_gradient_tape_scores_terminal_and_all_attention_auxiliaries_once() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    observed = executor.forward_components_with_observer(
        input_ids,
        sequence,
        observer=lambda **_: None,
        retain_gradient_tape=True,
    )
    tape = observed.gradient_tape
    assert tape is not None
    assert tape.auxiliary_site_ids == (
        *(f"mutation.{index}.operator_output" for index in range(1, 37, 2)),
        "terminal.rmsnorm.output",
    )
    bundle = tape.score_gradients_with_auxiliary(
        observed.execution.logits.sum()
    )
    assert tuple(bundle.site_gradients) == _expected_site_ids()
    assert tuple(bundle.attention_operator_output_gradients) == tuple(
        f"mutation.{index}.operator_output" for index in range(1, 37, 2)
    )
    assert len(bundle.site_gradients) == 73
    assert len(bundle.auxiliary_gradients) == 19
    expected_terminal = executor.lexical.exact_weight.sum(dim=0).view(1, 1, -1)
    expected_terminal = expected_terminal.expand(2, 3, -1)
    torch.testing.assert_close(
        bundle.terminal_normalized_output_gradient,
        expected_terminal,
    )
    assert all(
        value.shape == (2, 3, _WIDTH)
        and value.dtype == torch.float32
        and value.device.type == "cpu"
        and bool(torch.isfinite(value).all())
        for value in bundle.auxiliary_gradients.values()
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        tape.score_gradients_with_auxiliary(observed.execution.logits.sum())


def test_logits_free_backbone_terminal_cotangent_matches_full_logit_vjp() -> None:
    executor = _exact_executor(final_logit_softcap=0.75)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    full = executor.forward_components_with_observer(
        input_ids,
        sequence,
        observer=lambda **_: None,
        retain_gradient_tape=True,
    )
    assert full.gradient_tape is not None
    full_loss = F.cross_entropy(
        full.execution.logits[:, :-1].reshape(-1, _VOCABULARY),
        input_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    full_bundle = full.gradient_tape.score_gradients_with_auxiliary(full_loss)

    original_egress = executor.lexical.egress

    def reject_egress(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("backbone execution crossed lexical egress")

    executor.lexical.egress = reject_egress  # type: ignore[method-assign]
    observed_values: dict[str, torch.Tensor] = {}
    backbone = executor.forward_backbone_with_observer(
        input_ids,
        sequence,
        observer=lambda **payload: observed_values.__setitem__(
            f"{payload['event_id']}.{payload['role']}",
            payload["values"],
        ),
    )
    assert isinstance(backbone.execution, joint.JointGraphBackboneExecution)
    assert len(backbone.execution.event_ids) == 38
    assert backbone.execution.event_ids[-1] == "terminal.rmsnorm"
    assert not hasattr(backbone.execution, "logits")
    assert tuple(observed_values) == _expected_site_ids()
    operator_snapshots = (
        backbone.execution.attention_operator_output_snapshots
    )
    operator_outputs = backbone.execution.attention_operator_outputs
    expected_operator_sites = backbone.gradient_tape.auxiliary_site_ids[:-1]
    assert tuple(snapshot.site_id for snapshot in operator_snapshots) == (
        expected_operator_sites
    )
    assert tuple(operator_outputs) == expected_operator_sites
    assert tuple(
        backbone.execution.attention_operator_output_sha256s
    ) == expected_operator_sites
    assert tuple(
        backbone.execution.attention_operator_output_snapshot_sha256s
    ) == expected_operator_sites
    assert len(
        backbone.execution.attention_operator_output_catalog_sha256
    ) == 64
    assert all(
        snapshot.block_ordinal == ordinal
        and snapshot.values.shape == (2, 3, _WIDTH)
        and snapshot.values.device.type == "cpu"
        and snapshot.values.requires_grad is False
        and snapshot.metadata()["snapshot_sha256"] == snapshot.sha256
        for ordinal, snapshot in enumerate(operator_snapshots)
    )
    with pytest.raises(ValueError, match="snapshot inventory drifted"):
        replace(
            backbone.execution,
            attention_operator_output_snapshots=tuple(
                reversed(operator_snapshots)
            ),
        )

    manual_logits = original_egress(backbone.execution.normalized_output)
    cap = executor.final_logit_softcap
    assert cap is not None
    manual_logits = torch.tanh(manual_logits / cap) * cap
    manual_loss = F.cross_entropy(
        manual_logits[:, :-1].reshape(-1, _VOCABULARY),
        input_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    (terminal_gradient,) = torch.autograd.grad(
        manual_loss,
        backbone.execution.normalized_output,
        retain_graph=True,
    )
    cotangent_bundle = (
        backbone.gradient_tape.score_gradients_from_terminal_normalized_output(
            terminal_gradient
        )
    )
    assert tuple(cotangent_bundle.site_gradients) == tuple(
        full_bundle.site_gradients
    )
    assert tuple(cotangent_bundle.auxiliary_gradients) == tuple(
        full_bundle.auxiliary_gradients
    )
    assert tuple(operator_outputs) == tuple(
        cotangent_bundle.attention_operator_output_gradients
    )
    first_operator_site = expected_operator_sites[0]
    first_event_id = first_operator_site.removesuffix(".operator_output")
    bridge = build_attention_postnorm_bridge_receipt(
        operator_output=operator_outputs[first_operator_site],
        operator_output_score_gradient=(
            cotangent_bundle.attention_operator_output_gradients[
                first_operator_site
            ]
        ),
        postnorm_delta=observed_values[f"{first_event_id}.output"],
        postnorm_state_sha256="1" * 64,
        postnorm_config_sha256="2" * 64,
        execution_fingerprint_sha256="3" * 64,
    )
    assert len(bridge.operator_output_sha256) == 64
    assert len(bridge.operator_output_score_gradient_sha256) == 64
    for site_id in full_bundle.site_gradients:
        torch.testing.assert_close(
            cotangent_bundle.site_gradients[site_id],
            full_bundle.site_gradients[site_id],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    for site_id in full_bundle.auxiliary_gradients:
        torch.testing.assert_close(
            cotangent_bundle.auxiliary_gradients[site_id],
            full_bundle.auxiliary_gradients[site_id],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    normalized_before = backbone.execution.normalized_output.detach().clone()
    catalog_before = (
        backbone.execution.attention_operator_output_catalog_sha256
    )
    operator_snapshots[0].values.add_(1.0)
    assert (
        backbone.execution.attention_operator_output_catalog_sha256
        != catalog_before
    )
    assert torch.equal(
        backbone.execution.normalized_output,
        normalized_before,
    )
    assert backbone.gradient_tape.consumed is True


def test_terminal_cotangent_validation_is_fail_closed_and_one_use() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    backbone = executor.forward_backbone_with_observer(
        input_ids,
        _sequence(),
        observer=lambda **_: None,
    )
    tape = backbone.gradient_tape
    shape = backbone.execution.normalized_output.shape
    for invalid in (
        torch.zeros(shape[:-1]),
        torch.zeros(shape, dtype=torch.int64),
        torch.full(shape, float("nan")),
    ):
        with pytest.raises(ValueError, match="terminal gradient"):
            tape.score_gradients_from_terminal_normalized_output(invalid)
        assert tape.consumed is False
    bundle = tape.score_gradients_from_terminal_normalized_output(
        torch.zeros(shape, dtype=torch.float64)
    )
    assert all(
        not bool(value.any())
        for value in (
            *bundle.site_gradients.values(),
            *bundle.auxiliary_gradients.values(),
        )
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        tape.score_gradients_from_terminal_normalized_output(
            torch.zeros(shape)
        )


def test_gradient_tape_rejects_inference_mode_and_invalid_objectives() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    with torch.inference_mode(), pytest.raises(
        RuntimeError,
        match="cannot be captured in inference mode",
    ):
        executor.forward_components_with_observer(
            input_ids,
            sequence,
            observer=lambda **_: None,
            retain_gradient_tape=True,
        )

    observed = executor.forward_components_with_observer(
        input_ids,
        sequence,
        observer=lambda **_: None,
        retain_gradient_tape=True,
    )
    assert observed.gradient_tape is not None
    with pytest.raises(ValueError, match="differentiable scalar"):
        observed.gradient_tape.score_gradients(torch.ones(2))
    assert observed.gradient_tape.consumed is False
    objective = observed.execution.logits.sum()
    assert len(observed.gradient_tape.score_gradients(objective)) == 73


def test_observer_arguments_fail_closed_before_execution() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    before = _state_snapshot(executor)
    with pytest.raises(TypeError, match="observer must be callable"):
        executor.forward_components_with_observer(
            input_ids,
            sequence,
            observer=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="retain_gradient_tape must be bool"):
        executor.forward_components_with_observer(
            input_ids,
            sequence,
            observer=lambda **_: None,
            retain_gradient_tape=1,  # type: ignore[arg-type]
        )
    _assert_state_unchanged(before, executor)


@pytest.mark.parametrize(
    "source_site_id",
    ("lexical.ingress.output", "mutation.2.input", "mutation.2.output"),
)
def test_shadow_intervention_is_ephemeral_and_supports_every_source_role(
    source_site_id: str,
) -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    baseline = executor.forward_components(input_ids, sequence)
    state_before = _state_snapshot(executor)
    perturbation = torch.full((2, 3, _WIDTH), 0.05)

    shadow = executor.forward_components_with_shadow_intervention(
        input_ids,
        sequence,
        source_site_id=source_site_id,
        target_event_id="mutation.3",
        perturbation=perturbation,
    )

    assert shadow.shadow_source_site_id == source_site_id
    assert shadow.shadow_target_site_id == "mutation.3.output"
    assert shadow.shadow_target_flow is not None
    assert shadow.shadow_target_flow.shape == (2, 3, _WIDTH)
    assert torch.isfinite(shadow.shadow_target_flow).all()
    assert not torch.equal(shadow.execution.logits, baseline.logits)
    _assert_state_unchanged(state_before, executor)
    recovered = executor.forward_components(input_ids, sequence)
    _assert_same_execution(baseline, recovered)


def test_shadow_zero_is_exact_and_jvp_matches_symmetric_finite_displacement() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    sequence = _sequence()
    baseline = executor.forward_components(input_ids, sequence)
    zero = torch.zeros(2, 3, _WIDTH)
    direction = torch.linspace(-1.0, 1.0, zero.numel()).reshape_as(zero)
    direction = direction / direction.norm()

    zero_shadow = executor.forward_components_with_shadow_intervention(
        input_ids,
        sequence,
        source_site_id="mutation.2.output",
        target_event_id="mutation.3",
        perturbation=zero,
    )
    _assert_same_execution(baseline, zero_shadow.execution)

    def target_flow(perturbation: torch.Tensor) -> torch.Tensor:
        result = executor.forward_components_with_shadow_intervention(
            input_ids,
            sequence,
            source_site_id="mutation.2.output",
            target_event_id="mutation.3",
            perturbation=perturbation,
        )
        assert result.shadow_target_flow is not None
        return result.shadow_target_flow

    _, jvp = torch.autograd.functional.jvp(
        target_flow,
        zero,
        direction,
        create_graph=False,
        strict=True,
    )
    epsilon = 1.0e-3
    plus = target_flow(epsilon * direction)
    minus = target_flow(-epsilon * direction)
    finite = (plus - minus) / (2.0 * epsilon)

    assert jvp.norm() > 0
    assert torch.allclose(jvp, finite, atol=2.0e-6, rtol=2.0e-4)
    assert torch.equal(
        executor.forward_components(input_ids, sequence).logits,
        baseline.logits,
    )


def test_shadow_intervention_masks_invalid_rows_and_rejects_bad_topology() -> None:
    executor = _exact_executor()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    sequence = _sequence(valid=valid)
    baseline = executor.forward_components(input_ids, sequence)
    invalid_only = torch.zeros(2, 3, _WIDTH)
    invalid_only[~valid] = 100.0

    shadow = executor.forward_components_with_shadow_intervention(
        input_ids,
        sequence,
        source_site_id="mutation.2.input",
        target_event_id="mutation.3",
        perturbation=invalid_only,
    )
    _assert_same_execution(baseline, shadow.execution)

    zero = torch.zeros_like(invalid_only)
    invalid_edges = (
        ("terminal.rmsnorm.output", "mutation.3"),
        ("mutation.3.output", "mutation.2"),
        ("lexical.ingress.input", "mutation.3"),
        ("mutation.2.output", "lexical.egress"),
    )
    for source_site_id, target_event_id in invalid_edges:
        with pytest.raises(ValueError, match="shadow intervention requires"):
            executor.forward_components_with_shadow_intervention(
                input_ids,
                sequence,
                source_site_id=source_site_id,
                target_event_id=target_event_id,
                perturbation=zero,
            )
    with pytest.raises(ValueError, match="must match the exact residual carrier"):
        executor.forward_components_with_shadow_intervention(
            input_ids,
            sequence,
            source_site_id="mutation.2.output",
            target_event_id="mutation.3",
            perturbation=torch.zeros(1, 3, _WIDTH),
        )
