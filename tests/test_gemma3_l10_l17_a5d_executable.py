from __future__ import annotations

from collections.abc import Mapping
import copy

import pytest
import torch

from fisher_graph import gemma3_l10_l17_a5d_executable as executable
from fisher_graph import gemma3_l10_l17_a5d_report as report
from fisher_graph.gemma3_l10_l17_a5d_family_residual_cv import (
    A5dFamilyResidualCvSelection,
    fit_zero_mean_residual_generators,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    LayerFragmentRows,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)
from test_gemma3_l10_l17_a5d_report import (
    _digest,
    _lineage as _canonical_lineage,
    _resource,
    _source_owner as _canonical_source_owner,
)
from test_gemma3_l10_l17_full_block_closure_bundle import (
    POST_FEEDFORWARD_DELTA,
    RAW_OUTPUT,
    _runtime,
)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(child) for child in value)
    return False


def _report_hash_fixture(*, positive: bool) -> dict[str, object]:
    owner = _canonical_source_owner()
    additive = None
    if positive:
        additive = {
            "graph_sha256": _digest(701),
            "lowering_sha256_by_node": {
                name: _digest(710 + index)
                for index, name in enumerate(
                    report._EXPECTED_LAYER17_LOWERING_SHA256_BY_NODE
                )
            },
            "application_layer_ordinal": 17,
            "basis_means_exactly_zero": True,
            "source_decoders_reused": True,
            "source_affine_means_reinjected": False,
            "resources": _resource(
                nodes=4,
                interactions=0,
                parameters=160_534,
                macs=160_352,
                additions=620,
                peak=48,
            ),
        }
    additive_resources = None if additive is None else additive["resources"]
    return {
        "kind": report._ADDITIVE if positive else report._FROZEN,
        "selected_alpha": 0.25 if positive else 0.0,
        "selected_ridge": 1.0e-4 if positive else None,
        "lineage": _canonical_lineage(),
        "source_owner": owner,
        "additive_residual": additive,
        "selected_resources": {
            "layer17_scope": report._sum_resource(
                owner["layer17_resources"], additive_resources
            ),
            "composition_scope": report._sum_resource(
                owner["composition_resources"], additive_resources
            ),
        },
    }


@pytest.mark.parametrize("positive", [False, True])
def test_freeze_hash_is_byte_identical_to_publication_report(positive: bool) -> None:
    values = _report_hash_fixture(positive=positive)
    ours = executable.a5d_executable_freeze_sha256(**values)
    expected = report.a5d_selection_freeze_sha256(
        application_boundary=report._OUTPUT_BOUNDARY,
        application_order=report._APPLICATION_ORDER,
        source_ownership_preserved=True,
        source_affine_means_reinjected=False,
        **values,
    )
    assert ours == expected


def _real_residual_fit():
    source_graph, source_lowerings = _runtime(output_boundary=RAW_OUTPUT)
    generator = torch.Generator().manual_seed(4_817)
    observations = 20
    inputs = torch.randn(
        observations, 640, generator=generator, dtype=torch.float64
    )
    rows = AlignedFragmentRows(
        rows_by_fragment={
            lowering.mode_set_id: LayerFragmentRows(
                inputs=inputs,
                contributions=(
                    torch.randn(
                        observations,
                        lowering.computational_mode_basis.rank,
                        generator=generator,
                        dtype=torch.float64,
                    )
                    @ lowering.computational_mode_basis.decoder_basis
                ),
                fisher_weights=torch.linspace(
                    1.0, 2.0, observations, dtype=torch.float64
                ),
                sequences=observations,
            )
            for lowering in source_lowerings.values()
        },
        row_keys=tuple((f"example-{index}", 0) for index in range(observations)),
    )

    def subset(start: int, stop: int) -> AlignedFragmentRows:
        return AlignedFragmentRows(
            rows_by_fragment={
                name: LayerFragmentRows(
                    inputs=value.inputs[start:stop],
                    contributions=value.contributions[start:stop],
                    fisher_weights=value.fisher_weights[start:stop],
                    sequences=stop - start,
                )
                for name, value in rows.rows_by_fragment.items()
            },
            row_keys=rows.row_keys[start:stop],
        )

    fit = fit_zero_mean_residual_generators(
        subset(0, 16),
        subset(16, 20),
        source_graph=source_graph,
        source_lowerings_by_node=source_lowerings,
        fit_split_sha256="8" * 64,
        eval_split_sha256="9" * 64,
        generator_rank=16,
        ridge=1.0e-4,
        output_boundary=POST_FEEDFORWARD_DELTA,
    )
    return source_graph, source_lowerings, fit


def _selection(
    *,
    source_graph,
    source_lowerings,
    residual_fit,
    positive: bool,
) -> A5dFamilyResidualCvSelection:
    alpha = 0.25 if positive else 0.0
    ridge = 1.0e-4 if positive else None
    final_fit = None
    if positive:
        final_fit = {
            "graph_sha256": residual_fit.graph_plan.artifact_sha256,
            "lowering_sha256_by_node": {
                name: residual_fit.lowerings_by_node[name].artifact_sha256
                for name in residual_fit.graph_plan.traversal_order
            },
            "zero_mean_sha256_by_node": {
                name: residual_fit.lowerings_by_node[
                    name
                ].computational_mode_basis.mean_bias_sha256
                for name in residual_fit.graph_plan.traversal_order
            },
            "decoder_sha256_by_node": {
                name: source_lowerings[
                    name
                ].computational_mode_basis.decoder_basis_sha256
                for name in source_graph.traversal_order
            },
            "parameter_count": residual_fit.graph_plan.parameter_count,
            "macs_per_token": residual_fit.graph_plan.macs_per_token,
        }
    receipt = {
        "receipt_sha256": "1" * 64,
        "source": {
            "bridge_receipt_sha256": "2" * 64,
            "residual_target_receipt_sha256": "3" * 64,
            "source_graph_sha256": source_graph.artifact_sha256,
            "source_graph_parameter_count": source_graph.parameter_count,
            "source_graph_macs_per_token": source_graph.macs_per_token,
            "source_model_sha256": source_graph.model_fingerprint,
            "source_lowering_sha256_by_node": {
                name: source_lowerings[name].artifact_sha256
                for name in source_graph.traversal_order
            },
            "source_mean_sha256_by_node": {
                name: source_lowerings[
                    name
                ].computational_mode_basis.mean_bias_sha256
                for name in source_graph.traversal_order
            },
            "source_decoder_sha256_by_node": {
                name: source_lowerings[
                    name
                ].computational_mode_basis.decoder_basis_sha256
                for name in source_graph.traversal_order
            },
        },
        "selection": {
            "selected_alpha": alpha,
            "selected_ridge": ridge,
            "use_frozen_fallback": not positive,
        },
        "final_refit": {"fit": final_fit},
        "configuration": {"output_boundary": POST_FEEDFORWARD_DELTA},
    }
    selection = object.__new__(A5dFamilyResidualCvSelection)
    object.__setattr__(selection, "selected_alpha", alpha)
    object.__setattr__(selection, "selected_ridge", ridge)
    object.__setattr__(selection, "use_frozen_fallback", not positive)
    object.__setattr__(selection, "residual_fit", residual_fit if positive else None)
    object.__setattr__(selection, "node_order", source_graph.traversal_order)
    object.__setattr__(selection, "residual_width", 640)
    object.__setattr__(selection, "_receipt", receipt)
    return selection


def _lineage(*, source_graph, source_lowerings) -> dict[str, object]:
    return {
        "a5c_report_sha256": "a" * 64,
        "capture_sha256": "b" * 64,
        "target_solve_receipt_sha256": "c" * 64,
        "coordinate_row_bank_receipt_sha256": "2" * 64,
        "breadth_split_receipt_sha256": "d" * 64,
        "source_anchored_residual_receipt_sha256": "3" * 64,
        "residual_cv_receipt_sha256": "1" * 64,
        "layer10_graph_sha256": "e" * 64,
        "layer10_lowering_sha256_by_node": {
            name: source_lowerings[name].artifact_sha256
            for name in source_graph.traversal_order
        },
        "matched_double_deletion_graph_sha256": source_graph.artifact_sha256,
    }


@pytest.mark.parametrize("positive", [False, True])
def test_freeze_keeps_source_owner_and_only_attaches_zero_mean_residual(
    monkeypatch: pytest.MonkeyPatch, positive: bool
) -> None:
    source_graph, source_lowerings, fit = _real_residual_fit()
    selection = _selection(
        source_graph=source_graph,
        source_lowerings=source_lowerings,
        residual_fit=fit,
        positive=positive,
    )
    monkeypatch.setattr(
        executable,
        "validate_a5d_family_residual_cv_receipt",
        lambda value: copy.deepcopy(value),
    )
    # This focused fixture reuses Layer17 as its tiny composition; production
    # composition ownership is separately authenticated by the unpatched helper.
    monkeypatch.setattr(
        executable, "_validate_composition_ownership", lambda **_: None
    )

    frozen = executable.freeze_a5d_executable(
        selection=selection,
        source_layer17_graph=source_graph,
        source_layer17_lowerings_by_node=source_lowerings,
        source_composition_graph=source_graph,
        source_composition_lowerings=tuple(source_lowerings.values()),
        lineage=_lineage(
            source_graph=source_graph, source_lowerings=source_lowerings
        ),
    )
    section = frozen.report_section()

    assert frozen.source_layer17_graph.artifact_sha256 == source_graph.artifact_sha256
    assert (
        frozen.source_composition_graph.artifact_sha256
        == source_graph.artifact_sha256
    )
    assert section["source_ownership_preserved"] is True
    assert (
        section["source_owner"]["layer17_graph_sha256"]
        == source_graph.artifact_sha256
    )
    assert not _contains_tensor(section)
    if positive:
        assert frozen.kind == "additive_zero_mean_residual"
        assert frozen.additive_residual_graph is not None
        assert frozen.additive_residual_graph.interactions == ()
        assert section["additive_residual"]["resources"]["parameter_count"] == 160_534
        assert all(
            not bool(torch.count_nonzero(value.computational_mode_basis.mean_bias))
            for value in frozen.additive_residual_lowerings_by_node.values()
        )
    else:
        assert frozen.kind == "frozen_source_fallback"
        assert frozen.additive_residual_graph is None
        assert frozen.additive_residual_lowerings_by_node == {}
        assert section["additive_residual"] is None
        assert section["selected_resources"]["layer17_scope"] == section[
            "source_owner"
        ]["layer17_resources"]

    # Roundtrip copies make the freeze independent of subsequent caller-owned
    # tensor mutation.
    name = source_graph.traversal_order[0]
    frozen_mean = frozen.source_layer17_lowerings_by_node[
        name
    ].computational_mode_basis.mean_bias.clone()
    source_lowerings[name].computational_mode_basis.mean_bias.add_(1.0)
    assert torch.equal(
        frozen.source_layer17_lowerings_by_node[
            name
        ].computational_mode_basis.mean_bias,
        frozen_mean,
    )


class _Graph:
    def __init__(self, digest: str, order: tuple[str, ...]) -> None:
        self.artifact_sha256 = digest
        self.traversal_order = order

    def validate_integrity(self) -> None:
        return None


class _Lowering:
    def __init__(self, digest: str) -> None:
        self.artifact_sha256 = digest


class _Executor:
    calls: list[dict[str, object]] = []

    def __init__(self, adapter, graph_plan, lowerings, **kwargs) -> None:
        del adapter
        self.graph_plan = graph_plan
        self.lowerings = tuple(lowerings)
        self.kwargs = dict(kwargs)
        self.post_feedforward_delta_layer_ordinals = tuple(
            kwargs.get("post_feedforward_delta_layer_ordinals", ())
        )
        self.calls.append(
            {"graph_plan": graph_plan, "lowerings": self.lowerings, **kwargs}
        )


def _manual_executable(*, positive: bool) -> executable.A5dExecutableFreeze:
    layer17 = _Graph("3" * 64, ("l17-a", "l17-b"))
    composition = _Graph("4" * 64, ("l10", "l17-a", "l17-b"))
    additive = _Graph("5" * 64, layer17.traversal_order) if positive else None
    value = object.__new__(executable.A5dExecutableFreeze)
    fields = {
        "kind": "additive_zero_mean_residual" if positive else "frozen_source_fallback",
        "selected_alpha": 0.25 if positive else 0.0,
        "selected_ridge": 1.0e-4 if positive else None,
        "source_layer17_graph": layer17,
        "source_layer17_lowerings_by_node": {
            "l17-a": _Lowering("6" * 64),
            "l17-b": _Lowering("7" * 64),
        },
        "source_composition_graph": composition,
        "source_composition_lowerings": (
            _Lowering("8" * 64),
            _Lowering("6" * 64),
            _Lowering("7" * 64),
        ),
        "additive_residual_graph": additive,
        "additive_residual_lowerings_by_node": (
            {}
            if additive is None
            else {"l17-a": _Lowering("9" * 64), "l17-b": _Lowering("a" * 64)}
        ),
        "lineage": {
            **{
                name: "b" * 64
                for name in executable._LINEAGE_FIELDS
                if name != "layer10_lowering_sha256_by_node"
            },
            "layer10_graph_sha256": "1" * 64,
            "layer10_lowering_sha256_by_node": {"l10": "2" * 64},
        },
        "source_owner": {},
        "additive_residual": None,
        "selected_resources": {},
        "selection_freeze_sha256": "c" * 64,
    }
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    return value


@pytest.mark.parametrize("positive", [False, True])
def test_scoring_executors_never_replace_the_owner_or_use_primary_post_delta(
    monkeypatch: pytest.MonkeyPatch, positive: bool
) -> None:
    frozen = _manual_executable(positive=positive)
    layer10 = _Graph("1" * 64, ("l10",))
    layer10_lowerings = {"l10": _Lowering("2" * 64)}
    _Executor.calls = []
    monkeypatch.setattr(
        executable.A5dExecutableFreeze, "validate_integrity", lambda self: None
    )
    monkeypatch.setattr(
        executable, "Gemma3ModalGeneratorGraphExecutor", _Executor
    )

    result = executable.build_a5d_scoring_executors(
        object(), layer10, layer10_lowerings, frozen  # type: ignore[arg-type]
    )

    assert len({id(value) for value in result}) == 4
    assert all(
        "post_feedforward_delta_layer_ordinals" not in call
        for call in _Executor.calls
    )
    assert _Executor.calls[1]["graph_plan"] is frozen.source_layer17_graph
    assert _Executor.calls[2]["graph_plan"] is frozen.source_composition_graph
    assert _Executor.calls[3]["graph_plan"] is frozen.source_composition_graph
    additive_calls = [
        call
        for call in _Executor.calls
        if "additive_post_feedforward_graph_plan" in call
    ]
    assert len(additive_calls) == (2 if positive else 0)
    if positive:
        assert all(
            call["additive_post_feedforward_graph_plan"]
            is frozen.additive_residual_graph
            and call["additive_post_feedforward_scale"] == 0.25
            for call in additive_calls
        )
    else:
        assert all(
            call.keys() == {"graph_plan", "lowerings"}
            for call in _Executor.calls
        )


class _HeldBlocks(Mapping[str, object]):
    def __init__(self) -> None:
        self.accessed = False

    def __getitem__(self, key: str) -> object:
        self.accessed = True
        return (key,)

    def __iter__(self):
        self.accessed = True
        yield "held"

    def __len__(self) -> int:
        self.accessed = True
        return 1


def test_held_batch_is_inaccessible_until_freeze_hash_recomputes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _manual_executable(positive=False)
    blocks = _HeldBlocks()
    monkeypatch.setattr(
        executable,
        "recompute_a5d_executable_freeze_sha256",
        lambda _: "d" * 64,
    )
    with pytest.raises(RuntimeError, match="not frozen"):
        executable.select_a5d_held_scoring_batch_after_freeze(
            blocks=blocks, held_family_alias="held", executable=frozen
        )
    assert blocks.accessed is False
