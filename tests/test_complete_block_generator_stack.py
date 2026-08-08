from dataclasses import fields, replace

import pytest
import torch

from fisher_graph.complete_block_generator_stack import (
    CompleteBlockGeneratorStackExecutor,
)
from fisher_graph.complete_block_residual_forms import (
    CompleteBlockResidualForm,
    ResidualForm,
)
from fisher_graph.residual_carrier import (
    DenseResidualCarrierFactory,
    ResidualCarrierSession,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)
from test_structured_transformer_layer_executor import _executor, _sequence


def _rename_dataclass_sites(value: object, layer_id: str) -> object:
    updates = {
        field.name: (
            getattr(value, field.name).replace("layer.0", layer_id)
            if isinstance(getattr(value, field.name), str)
            else getattr(value, field.name)
        )
        for field in fields(value)  # type: ignore[arg-type]
    }
    return replace(value, **updates)  # type: ignore[arg-type]


def _block(
    ordinal: int,
    form: ResidualForm = ResidualForm.DIRECT_OUTPUT,
    *,
    global_attention: bool = False,
) -> CompleteBlockResidualForm:
    parent = _executor(
        attention_kind=(
            "global_causal" if global_attention else "sliding_causal"
        ),
        window_size=None if global_attention else 3,
    )
    layer_id = f"layer.{ordinal}"
    transformer = parent.config.transformer
    stages = tuple(
        _rename_dataclass_sites(stage, layer_id)
        for stage in transformer.stages
    )
    operator_sites = transformer.operator_sites
    if operator_sites is not None:
        operator_sites = _rename_dataclass_sites(operator_sites, layer_id)
    config = replace(
        parent.config,
        transformer=replace(
            transformer,
            stages=stages,
            operator_sites=operator_sites,
        ),
    )
    with torch.random.fork_rng(devices=()):
        torch.manual_seed(9100 + ordinal)
        executor = StructuredTransformerLayerExecutor(config)
    executor.eval()
    return CompleteBlockResidualForm(executor, form).eval()


def _stack(
    *,
    mark_source_weights: bool = False,
    carrier_factory: object | None = None,
) -> CompleteBlockGeneratorStackExecutor:
    blocks = (
        _block(0, ResidualForm.DIRECT_OUTPUT),
        _block(1, ResidualForm.EXPLICIT, global_attention=True),
        _block(2, ResidualForm.DIRECT_OUTPUT),
    )
    if mark_source_weights:
        with torch.no_grad():
            for block in blocks:
                block.executor._weight_origin.fill_(1)
    return CompleteBlockGeneratorStackExecutor(
        ("layer.0", "layer.1", "layer.2"),
        blocks,
        carrier_factory=carrier_factory,
    )


def _manual(
    stack: CompleteBlockGeneratorStackExecutor,
    hidden: torch.Tensor,
    sequence: object,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    current = hidden
    outputs = []
    for block in stack.blocks:
        current = block(current, sequence, prefix=None)  # type: ignore[arg-type]
        outputs.append(current)
    return current, tuple(outputs)


def test_three_layer_stack_matches_sequential_exact_blocks_without_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack()
    mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [False, False, True, True, True, True],
        ]
    )
    sequence = _sequence(mask)
    torch.manual_seed(2201)
    hidden = torch.randn(2, 6, stack.width)
    expected, expected_layers = _manual(stack, hidden, sequence)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("stack delegated to complete-block forward")

    for block in stack.blocks:
        monkeypatch.setattr(block, "forward_components", forbidden)
        monkeypatch.setattr(block, "forward", forbidden)

    execution = stack.forward_components(
        hidden,
        sequence,
        capture_layer_outputs=True,
    )
    torch.testing.assert_close(execution.output, expected, rtol=0, atol=0)
    assert len(execution.layer_outputs) == 3
    for actual, expected_layer in zip(
        execution.layer_outputs,
        expected_layers,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected_layer, rtol=0, atol=0)
    assert stack.forward_components(hidden, sequence).layer_outputs == ()


def test_mutation_plan_and_receipts_are_strictly_ordered_and_versioned() -> None:
    stack = _stack()
    hidden = torch.randn(1, 5, stack.width)
    execution = stack.forward_components(
        hidden,
        _sequence(torch.ones(1, 5, dtype=torch.bool)),
    )
    expected_ids = (
        "layer.0.attention",
        "layer.0.feed_forward",
        "layer.1.attention",
        "layer.1.feed_forward",
        "layer.2.attention",
        "layer.2.feed_forward",
    )
    assert stack.expected_mutation_ids == expected_ids
    assert tuple(step.mutation_id for step in stack.mutation_plan) == expected_ids
    assert tuple(
        (step.version_before, step.version_after)
        for step in stack.mutation_plan
    ) == tuple((index, index + 1) for index in range(6))
    assert tuple(
        receipt.mutation_id for receipt in execution.mutation_receipts
    ) == expected_ids
    assert tuple(
        (receipt.version_before, receipt.version_after)
        for receipt in execution.mutation_receipts
    ) == tuple((index, index + 1) for index in range(6))
    assert execution.carrier_version == 6


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[True, True, True, True, False, False]]),
        torch.tensor([[False, False, True, True, True, True]]),
    ],
)
def test_padding_is_identity_and_causal_prefixes_ignore_future_changes(
    mask: torch.Tensor,
) -> None:
    stack = _stack()
    sequence = _sequence(mask)
    torch.manual_seed(3301)
    hidden = torch.randn(1, 6, stack.width)
    baseline = stack(hidden, sequence)
    torch.testing.assert_close(baseline[~mask], hidden[~mask], rtol=0, atol=0)

    poisoned_padding = hidden.clone()
    poisoned_padding[~mask] = 1.0e4 * torch.randn_like(
        poisoned_padding[~mask]
    )
    poisoned_output = stack(poisoned_padding, sequence)
    torch.testing.assert_close(
        baseline[mask],
        poisoned_output[mask],
        rtol=0,
        atol=0,
    )

    valid_indices = torch.nonzero(mask[0], as_tuple=False).flatten()
    changed_position = int(valid_indices[-1].item())
    earlier = valid_indices[:-1]
    changed = hidden.clone()
    changed[:, changed_position] += 100.0
    perturbed = stack(changed, sequence)
    torch.testing.assert_close(
        baseline[:, earlier],
        perturbed[:, earlier],
        rtol=0,
        atol=0,
    )


def test_failure_aborts_transaction_and_same_executor_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack()
    sequence = _sequence(torch.ones(1, 5, dtype=torch.bool))
    hidden = torch.randn(1, 5, stack.width)
    hidden_snapshot = hidden.clone()
    expected, _ = _manual(stack, hidden, sequence)

    aborted_metadata: list[dict[str, object]] = []
    original_abort = ResidualCarrierSession.abort

    def tracking_abort(session: ResidualCarrierSession) -> None:
        original_abort(session)
        aborted_metadata.append(session.metadata)

    monkeypatch.setattr(ResidualCarrierSession, "abort", tracking_abort)
    projection = stack.blocks[1].executor.feed_forward.project

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected leaf failure")

    monkeypatch.setattr(
        stack.blocks[1].executor.feed_forward,
        "project",
        fail,
    )
    with pytest.raises(RuntimeError, match="injected leaf failure"):
        stack(hidden, sequence)
    assert len(aborted_metadata) == 1
    assert aborted_metadata[0]["status"] == "aborted"
    assert aborted_metadata[0]["active"] is False
    assert aborted_metadata[0]["live"] is False
    torch.testing.assert_close(hidden, hidden_snapshot, rtol=0, atol=0)

    monkeypatch.setattr(
        stack.blocks[1].executor.feed_forward,
        "project",
        projection,
    )
    retry = stack(hidden, sequence)
    torch.testing.assert_close(retry, expected, rtol=0, atol=0)
    assert not any("session" in name or "carrier" in name for name in stack._modules)


class _CountingFactory:
    def __init__(self) -> None:
        self.created = 0
        self.delegate = DenseResidualCarrierFactory()

    def create(self, *args: object, **kwargs: object) -> object:
        self.created += 1
        return self.delegate.create(*args, **kwargs)  # type: ignore[arg-type]


class _MaterializationSpyCarrier:
    def __init__(self, delegate: object, counter: list[int]) -> None:
        self.delegate = delegate
        self.counter = counter

    @property
    def backend_kind(self) -> str:
        return self.delegate.backend_kind  # type: ignore[union-attr]

    @property
    def wire_id(self) -> str:
        return self.delegate.wire_id  # type: ignore[union-attr]

    @property
    def shape(self) -> torch.Size:
        return self.delegate.shape  # type: ignore[union-attr]

    @property
    def dtype(self) -> torch.dtype:
        return self.delegate.dtype  # type: ignore[union-attr]

    @property
    def device(self) -> torch.device:
        return self.delegate.device  # type: ignore[union-attr]

    @property
    def version(self) -> int:
        return self.delegate.version  # type: ignore[union-attr]

    def normalized_view(self, normalizer: object) -> torch.Tensor:
        return self.delegate.normalized_view(  # type: ignore[union-attr]
            normalizer
        )

    def apply_mutation(
        self,
        delta: torch.Tensor,
        combiner: object | None = None,
    ) -> object:
        return _MaterializationSpyCarrier(
            self.delegate.apply_mutation(  # type: ignore[union-attr]
                delta,
                combiner,
            ),
            self.counter,
        )

    def materialize(self) -> torch.Tensor:
        self.counter[0] += 1
        return self.delegate.materialize()  # type: ignore[union-attr]

    def metadata(self) -> object:
        return self.delegate.metadata()  # type: ignore[union-attr]


class _MaterializationSpyFactory:
    def __init__(self) -> None:
        self.delegate = DenseResidualCarrierFactory()
        self.materializations = [0]

    def create(self, *args: object, **kwargs: object) -> object:
        carrier = self.delegate.create(*args, **kwargs)  # type: ignore[arg-type]
        return _MaterializationSpyCarrier(carrier, self.materializations)


def test_each_forward_owns_and_closes_a_fresh_carrier_resource() -> None:
    factory = _CountingFactory()
    stack = _stack(carrier_factory=factory)
    sequence = _sequence(torch.ones(1, 4, dtype=torch.bool))
    hidden = torch.randn(1, 4, stack.width)
    first = stack.forward_components(hidden, sequence)
    second = stack.forward_components(hidden, sequence)

    assert factory.created == 2
    assert first.carrier_version == second.carrier_version == 6
    assert len(first.mutation_receipts) == len(second.mutation_receipts) == 6
    assert not hasattr(stack, "_session")
    assert not hasattr(stack, "_carrier")
    torch.testing.assert_close(first.output, second.output, rtol=0, atol=0)


def test_non_capture_path_materializes_only_the_final_state() -> None:
    factory = _MaterializationSpyFactory()
    stack = _stack(carrier_factory=factory)
    sequence = _sequence(torch.ones(1, 4, dtype=torch.bool))
    hidden = torch.randn(1, 4, stack.width)

    stack.forward_components(hidden, sequence, capture_layer_outputs=False)
    assert factory.materializations[0] == 1

    stack.forward_components(hidden, sequence, capture_layer_outputs=True)
    assert factory.materializations[0] == 5


def test_accounting_and_manifest_make_no_compression_or_source_free_claim() -> None:
    stack = _stack(mark_source_weights=True)
    sequence = _sequence(
        torch.tensor([[True, True, True, False, False]])
    )
    accounting = stack.logical_accounting(sequence)
    manifest = stack.graph_manifest()

    assert stack.owns_source_model_weights is True
    assert stack.executor_local_source_free is False
    assert stack.compression_attempted is False
    assert stack.physical_kernel_fusion_measured is False
    assert accounting.source_parameter_count == stack.learned_parameter_count
    assert accounting.candidate_parameter_count == stack.learned_parameter_count
    assert accounting.removed_parameter_count == 0
    assert accounting.source_logical_macs == accounting.candidate_logical_macs
    assert accounting.removed_logical_macs == 0
    assert manifest["contains_source_model_weights"] is True
    assert manifest["executor_local_source_free"] is False
    assert manifest["compression_attempted"] is False
    assert manifest["parameter_reduction"] is False
    assert manifest["logical_mac_reduction"] is False
    assert manifest["physical_kernel_fusion_measured"] is False
    assert manifest["ordered_mutation_count"] == 6
    assert len(stack.execution_fingerprint()) == 64


def test_constructor_rejects_nonexact_order_dtype_and_storage_aliases() -> None:
    block0 = _block(0)
    block1 = _block(1)
    with pytest.raises(ValueError, match="contiguous"):
        CompleteBlockGeneratorStackExecutor(
            ("layer.0", "layer.2"),
            (block0, _block(2)),
        )
    with pytest.raises(ValueError, match="exact residual forms"):
        CompleteBlockGeneratorStackExecutor(
            ("layer.0",),
            (_block(0, ResidualForm.DROP_BLOCK_IDENTITY),),
        )
    with pytest.raises(ValueError, match="declared layer order"):
        CompleteBlockGeneratorStackExecutor(
            ("layer.0", "layer.1"),
            (block1, block0),
        )
    with pytest.raises(ValueError, match="alias tensor storage"):
        CompleteBlockGeneratorStackExecutor(
            ("layer.0", "layer.1"),
            (block0, block0),
        )
    block1.double()
    with pytest.raises(ValueError, match="share width, dtype, and device"):
        CompleteBlockGeneratorStackExecutor(
            ("layer.0", "layer.1"),
            (block0, block1),
        )
