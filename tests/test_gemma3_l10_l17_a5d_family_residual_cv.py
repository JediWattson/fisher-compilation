from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from fisher_graph import gemma3_l10_l17_a5d_family_residual_cv as a5d
from fisher_graph.computational_modes import ComputationalModeBasis
from fisher_graph.gemma3_l10_l17_a5d_family_residual_cv import (
    A5D_FIXED_GENERATOR_RANK,
    fit_zero_mean_residual_generators,
    select_a5d_family_disjoint_residual,
    validate_a5d_family_residual_cv_receipt,
)
from fisher_graph.gemma3_l10_l17_a5d_source_anchored_residual import (
    build_a5d_source_anchored_residual_targets,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    LayerFragmentRows,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)
from test_gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    FRAGMENTS,
    NODE_ORDER,
    _bases,
    _build,
    _rows,
)
from test_gemma3_l10_l17_full_block_closure_bundle import (
    _runtime as _real_rank16_runtime,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Node:
    name: str
    input_width: int = 7
    output_width: int = 182
    causal_order: int = 0
    input_boundary: str = "layer.17.mlp.input"


class _Graph:
    def __init__(self, identity: str, *, model: str = "a" * 64) -> None:
        self.nodes = tuple(
            _Node(name, causal_order=index)
            for index, name in enumerate(NODE_ORDER)
        )
        self.traversal_order = NODE_ORDER
        self.interactions = ()
        self.model_fingerprint = model
        self.parameter_cluster_plan_sha256 = "c" * 64
        self.parameter_count = 91_337
        self.macs_per_token = 87_221
        self.artifact_sha256 = _sha(identity)

    def validate_integrity(self) -> None:
        return None


class _SourceLowering:
    def __init__(self, name: str, basis: ComputationalModeBasis) -> None:
        self.artifact_sha256 = _sha(f"source:{name}")
        self.computational_mode_basis = basis


class _FittedLowering:
    def __init__(
        self,
        name: str,
        ridge: float,
        gain: float,
        source_basis: ComputationalModeBasis,
    ) -> None:
        self.artifact_sha256 = _sha(
            f"fit:{name}:{ridge.hex()}:{gain.hex()}"
        )
        self.coordinate_generator_plan = SimpleNamespace(
            rank=A5D_FIXED_GENERATOR_RANK
        )
        self.fused_residual_plan = SimpleNamespace(
            gain=gain / len(NODE_ORDER)
        )
        self.computational_mode_basis = ComputationalModeBasis(
            binding=source_basis.binding,
            config=source_basis.config,
            rank=source_basis.rank,
            mean_bias=torch.zeros_like(source_basis.mean_bias),
            encoder_basis=source_basis.encoder_basis,
        )


class _Fit:
    def __init__(
        self,
        ridge: float,
        gain: float,
        call: int,
        source_bases: dict[str, ComputationalModeBasis],
    ) -> None:
        self.graph_plan = _Graph(
            f"fit-graph:{ridge.hex()}:{gain.hex()}:{call}"
        )
        self.lowerings_by_node = {
            name: _FittedLowering(name, ridge, gain, source_bases[name])
            for name in NODE_ORDER
        }


class _HeadAdapter:
    def __init__(self) -> None:
        self.maximum_rows = 0

    def model_fingerprint(self) -> str:
        return "a" * 64

    def project_logits(self, states: Tensor, sequence: object) -> Tensor:
        assert states.shape[:2] == (1, sequence.query_length)
        self.maximum_rows = max(self.maximum_rows, states.shape[1])
        first = states[..., :3]
        fourth = -first.sum(dim=-1, keepdim=True)
        return torch.cat((first, fourth), dim=-1)


def _desired_residual(inputs: Tensor) -> Tensor:
    result = torch.zeros(inputs.shape[0], 182, dtype=torch.float32)
    result[:, :3] = inputs[:, :3].to(dtype=torch.float32) / 20.0
    return result


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(child) for child in value)
    return False


def _fixture(*, compiled_inputs: Tensor | None = None):
    bases = _bases()
    bridge = _build(bases=bases, compiled_inputs=compiled_inputs)
    inputs = next(iter(bridge.all_rows.rows_by_fragment.values())).inputs
    desired = _desired_residual(inputs)
    frozen = 0.2 * desired
    native = frozen + desired
    targets = build_a5d_source_anchored_residual_targets(
        frozen_compiled_block_states=frozen.to(dtype=torch.float64),
        compiled_correction_base_states=torch.zeros_like(
            frozen, dtype=torch.float64
        ),
        oracle_rows=bridge.all_rows,
        bases_by_node=bases,
        node_order=NODE_ORDER,
        fragment_id_by_node=FRAGMENTS,
    )
    source_lowerings = {
        name: _SourceLowering(name, bases[name]) for name in NODE_ORDER
    }
    return bridge, targets, bases, source_lowerings, inputs, native, frozen


def _install_fake_fitter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gains: dict[float, float],
    bases: dict[str, ComputationalModeBasis],
) -> list[tuple[object, object, float]]:
    calls: list[tuple[object, object, float]] = []
    real_authenticate = a5d._authenticate_lowering_catalog

    def fake_fit(fit_rows, eval_rows, **kwargs):
        ridge = float(kwargs["ridge"])
        calls.append((fit_rows, eval_rows, ridge))
        assert kwargs["generator_rank"] == A5D_FIXED_GENERATOR_RANK
        assert not (set(fit_rows.row_keys) & set(eval_rows.row_keys))
        return _Fit(ridge, gains[ridge], len(calls), bases)

    def fake_apply(inputs: Tensor, plan: object) -> Tensor:
        assert inputs.dtype == torch.float32
        return _desired_residual(inputs) * plan.gain

    def authenticate(values, node_order, *, label):
        if all(
            isinstance(value, (_SourceLowering, _FittedLowering))
            for value in values.values()
        ):
            return dict(values)
        return real_authenticate(values, node_order, label=label)

    monkeypatch.setattr(a5d, "fit_zero_mean_residual_generators", fake_fit)
    monkeypatch.setattr(a5d, "apply_modal_generator", fake_apply)
    monkeypatch.setattr(a5d, "_authenticate_lowering_catalog", authenticate)
    return calls


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gains: dict[float, float],
    alpha_grid=(0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
    compiled_inputs: Tensor | None = None,
):
    (
        bridge,
        targets,
        bases,
        source_lowerings,
        inputs,
        native,
        frozen,
    ) = _fixture(compiled_inputs=compiled_inputs)
    calls = _install_fake_fitter(monkeypatch, gains=gains, bases=bases)
    adapter = _HeadAdapter()
    result = select_a5d_family_disjoint_residual(
        bridge=bridge,
        targets=targets,
        source_graph=_Graph("source"),
        source_lowerings_by_node=source_lowerings,
        adapter=adapter,
        native_block_states=native,
        frozen_compiled_block_states=frozen,
        output_boundary="layer.17.mlp.delta",
        ridge_grid=tuple(gains),
        alpha_grid=alpha_grid,
        final_head_chunk_rows=2,
        final_head_token_locality_lineage_sha256="f" * 64,
    )
    return result, calls, adapter, inputs, native, frozen, source_lowerings


def test_joint_cv_selects_interior_alpha_and_refits_only_winning_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, adapter, inputs, native, frozen, _ = _run(
        monkeypatch,
        gains={0.0: 2.0, 1.0: 0.1},
    )

    assert result.use_frozen_fallback is False
    assert result.selected_ridge == 0.0
    assert result.selected_alpha == 0.5
    assert len(calls) == 2 * 7 + 1
    assert calls[-1][2] == 0.0
    assert calls[-1][0].observations == 28
    assert calls[-1][0].sequences == 14
    candidate = result.candidate_block_states(frozen, inputs)
    torch.testing.assert_close(candidate, native, rtol=1e-6, atol=1e-7)

    receipt = result.receipt()
    assert receipt["configuration"]["fit_reuse_policy"] == (
        "fit_once_per_ridge_and_family_fold_then_score_all_alphas"
    )
    assert all(candidate["fit_count"] == 7 for candidate in receipt["candidates"])
    assert receipt["final_refit"]["performed"] is True
    assert receipt["final_refit"]["fit_observations"] == 28
    assert adapter.maximum_rows <= 2
    assert not _contains_tensor(receipt)
    assert validate_a5d_family_residual_cv_receipt(receipt) == receipt


def test_alpha_zero_fallback_is_exact_and_retains_no_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, _adapter, _inputs, _native, frozen, _ = _run(
        monkeypatch,
        gains={0.0: 0.0, 1.0: 0.0},
    )

    assert result.use_frozen_fallback is True
    assert result.selected_alpha == 0.0
    assert result.selected_ridge is None
    assert result.residual_fit is None
    assert len(calls) == 2 * 7
    monkeypatch.setattr(
        a5d,
        "apply_modal_generator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("alpha zero executed residual arithmetic")
        ),
    )
    observed = result.candidate_block_states(frozen, inputs=None)
    assert torch.equal(observed, frozen)
    assert observed.data_ptr() != frozen.data_ptr()
    with pytest.raises(RuntimeError, match="alpha zero"):
        result.predict_residual(torch.ones(1, 7))
    receipt = result.receipt()
    assert receipt["selection"]["winner_alpha_before_fallback"] == 0.0
    # All alpha-zero ridge entries tie; the declared policy chooses stronger
    # ridge before collapsing the executable choice to ridge=None.
    assert receipt["selection"]["winner_ridge_before_fallback"] == 1.0
    assert receipt["final_refit"]["performed"] is False


def test_outer_held_family_in_ownership_fails_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bridge,
        targets,
        bases,
        source_lowerings,
        _inputs,
        native,
        frozen,
    ) = _fixture()

    class LeakingBridge:
        def __init__(self) -> None:
            for name in (
                "all_rows",
                "audit_rows",
                "node_order",
                "training_family_aliases",
                "held_family_alias",
                "receipt_sha256",
            ):
                setattr(self, name, getattr(bridge, name))
            self.family_alias_by_example = dict(
                bridge.family_alias_by_example
            )
            first = next(iter(self.family_alias_by_example))
            self.family_alias_by_example[first] = self.held_family_alias

        def receipt(self):
            return bridge.receipt()

    calls = _install_fake_fitter(
        monkeypatch, gains={0.0: 2.0}, bases=bases
    )
    with pytest.raises(ValueError, match="outer-held"):
        select_a5d_family_disjoint_residual(
            bridge=LeakingBridge(),
            targets=targets,
            source_graph=_Graph("source"),
            source_lowerings_by_node=source_lowerings,
            adapter=_HeadAdapter(),
            native_block_states=native,
            frozen_compiled_block_states=frozen,
            output_boundary="layer.17.mlp.delta",
            ridge_grid=(0.0,),
            alpha_grid=(0.0, 0.5),
            final_head_chunk_rows=2,
            final_head_token_locality_lineage_sha256="f" * 64,
        )
    assert calls == []


def test_equal_positive_scores_choose_smaller_alpha_then_stronger_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, *_ = _run(
        monkeypatch,
        gains={0.0: 4.0, 1.0: 4.0},
    )

    assert result.selected_alpha == 0.25
    assert result.selected_ridge == 1.0
    assert calls[-1][2] == 1.0
    selection = result.receipt()["selection"]
    assert selection["winner_alpha_before_fallback"] == 0.25
    assert selection["winner_ridge_before_fallback"] == 1.0


def test_signature_purge_is_preserved_for_residual_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _coordinates, _fisher, row_keys, ownership = _rows()
    modified = inputs.clone()
    seen: set[str] = set()
    for index, (example_id, _) in enumerate(row_keys):
        alias = ownership[example_id]
        if alias not in seen:
            modified[index] = 0.0
            seen.add(alias)

    result, calls, *_ = _run(
        monkeypatch,
        gains={0.0: 2.0},
        compiled_inputs=modified.contiguous(),
    )

    assert {
        fold["fit_rows_removed_for_signature_overlap"]
        for fold in result.receipt()["candidates"][0]["folds"]
    } == {6}
    for fit_rows, eval_rows, _ridge in calls[:-1]:
        assert not (
            set(a5d._input_row_signatures(fit_rows))
            & set(a5d._input_row_signatures(eval_rows))
        )


def test_receipt_rejects_rehashed_selection_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, *_ = _run(monkeypatch, gains={0.0: 2.0})
    receipt = result.receipt()
    receipt["selection"]["selected_alpha"] = 0.25
    receipt["selection"]["selected_alpha_hex"] = float(0.25).hex()
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        a5d._REPORT_DOMAIN
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="selection contradicts"):
        validate_a5d_family_residual_cv_receipt(receipt)


def test_fit_receipts_pin_zero_means_decoders_and_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _calls, _adapter, _inputs, _native, _frozen, sources = _run(
        monkeypatch,
        gains={0.0: 2.0},
    )
    receipt = result.receipt()
    source_decoders = {
        name: sources[name].computational_mode_basis.decoder_basis_sha256
        for name in NODE_ORDER
    }
    evidence = receipt["candidates"][0]["folds"][0]["fit"]

    assert evidence["all_residual_basis_means_exactly_zero"] is True
    assert evidence["all_residual_decoders_byte_identical_to_source"] is True
    assert evidence["decoder_sha256_by_node"] == source_decoders
    assert evidence["parameters_match_zero_mean_source_expectation"] is True
    assert evidence["macs_equal_source"] is True
    assert evidence["source_parameter_count"] == receipt["source"][
        "source_graph_parameter_count"
    ]
    assert evidence["parameter_count"] == (
        evidence["source_parameter_count"]
        - evidence["removed_source_affine_mean_parameters"]
    )
    assert evidence["macs_per_token"] == receipt["source"][
        "source_graph_macs_per_token"
    ]
    for lowering in result.residual_fit.lowerings_by_node.values():
        assert not bool(
            torch.count_nonzero(
                lowering.computational_mode_basis.mean_bias
            )
        )


def test_validator_rejects_rehashed_resource_or_decoder_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, *_ = _run(monkeypatch, gains={0.0: 2.0})
    for field, value in (
        ("parameter_count", 1),
        ("all_residual_decoders_byte_identical_to_source", False),
    ):
        receipt = copy.deepcopy(result.receipt())
        receipt["candidates"][0]["folds"][0]["fit"][field] = value
        payload = dict(receipt)
        payload.pop("receipt_sha256")
        receipt["receipt_sha256"] = hashlib.sha256(
            a5d._REPORT_DOMAIN
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        with pytest.raises(ValueError, match="invariant"):
            validate_a5d_family_residual_cv_receipt(receipt)


def test_real_fitter_builds_zero_mean_byte_identical_decoder_runtime() -> None:
    graph, lowerings = _real_rank16_runtime()
    generator = torch.Generator().manual_seed(92_501)
    observations = 24
    inputs = torch.randn(
        observations, 640, generator=generator, dtype=torch.float64
    )
    all_rows = AlignedFragmentRows(
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
            for lowering in lowerings.values()
        },
        row_keys=tuple((f"example-{index}", 0) for index in range(observations)),
    )

    def subset(start: int, stop: int) -> AlignedFragmentRows:
        return AlignedFragmentRows(
            rows_by_fragment={
                name: LayerFragmentRows(
                    inputs=rows.inputs[start:stop],
                    contributions=rows.contributions[start:stop],
                    fisher_weights=rows.fisher_weights[start:stop],
                    sequences=stop - start,
                )
                for name, rows in all_rows.rows_by_fragment.items()
            },
            row_keys=all_rows.row_keys[start:stop],
        )

    fitted = fit_zero_mean_residual_generators(
        subset(0, 18),
        subset(18, 24),
        source_graph=graph,
        source_lowerings_by_node=lowerings,
        fit_split_sha256="8" * 64,
        eval_split_sha256="9" * 64,
        generator_rank=16,
        ridge=1.0e-4,
    )

    removed_means = sum(
        node.weights.output_bias.numel()
        for node in graph.nodes
        if node.weights.output_bias is not None
    )
    assert fitted.graph_plan.parameter_count == (
        graph.parameter_count - removed_means
    )
    assert fitted.graph_plan.macs_per_token == graph.macs_per_token
    for name in graph.traversal_order:
        source_basis = lowerings[name].computational_mode_basis
        residual_basis = fitted.lowerings_by_node[
            name
        ].computational_mode_basis
        assert not bool(torch.count_nonzero(residual_basis.mean_bias))
        assert residual_basis.decoder_basis_sha256 == (
            source_basis.decoder_basis_sha256
        )
        assert torch.equal(
            residual_basis.decoder_basis, source_basis.decoder_basis
        )
