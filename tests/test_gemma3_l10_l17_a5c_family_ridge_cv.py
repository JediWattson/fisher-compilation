from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from fisher_graph import gemma3_l10_l17_a5c_family_ridge_cv as a5c
from fisher_graph.gemma3_l10_l17_a5c_family_ridge_cv import (
    A5C_FIXED_GENERATOR_RANK,
    select_a5c_family_disjoint_ridge,
    validate_a5c_family_ridge_cv_receipt,
)
from fisher_graph.gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    a5b_tensor_sha256,
)
from test_gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    NODE_ORDER,
    _build,
    _rows,
)
from test_gemma3_l10_l17_full_block_closure_bundle import (
    _runtime as _real_rank16_runtime,
)
from test_modal_generator_lowering import _lowering as _real_lowering


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Node:
    name: str
    input_width: int = 7
    output_width: int = 182


class _Graph:
    def __init__(self, identity: str, *, model: str = "a" * 64) -> None:
        self.nodes = tuple(_Node(name) for name in NODE_ORDER)
        self.traversal_order = NODE_ORDER
        self.interactions = ()
        self.model_fingerprint = model
        self.artifact_sha256 = _sha(identity)

    def validate_integrity(self) -> None:
        return None


class _SourceLowering:
    def __init__(self, name: str) -> None:
        self.artifact_sha256 = _sha(f"source:{name}")

    def validate_integrity(self) -> None:
        return None


class _FittedLowering:
    def __init__(self, name: str, ridge: float, gain: float) -> None:
        self.artifact_sha256 = _sha(f"fit:{name}:{ridge.hex()}:{gain.hex()}")
        self.coordinate_generator_plan = SimpleNamespace(
            rank=A5C_FIXED_GENERATOR_RANK
        )
        self.fused_residual_plan = SimpleNamespace(gain=gain / len(NODE_ORDER))

    def validate_integrity(self) -> None:
        return None


class _Fit:
    def __init__(self, ridge: float, gain: float, call: int) -> None:
        self.graph_plan = _Graph(f"fit-graph:{ridge.hex()}:{gain.hex()}:{call}")
        self.lowerings_by_node = {
            name: _FittedLowering(name, ridge, gain) for name in NODE_ORDER
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


class _RowBankProxy:
    """Structural row bank with no inheritance from the A5b bridge type."""

    def __init__(self, source: object) -> None:
        for name in (
            "all_rows",
            "audit_rows",
            "node_order",
            "family_alias_by_example",
            "training_family_aliases",
            "held_family_alias",
            "receipt_sha256",
        ):
            setattr(self, name, getattr(source, name))
        self._source = source

    def receipt(self) -> dict[str, object]:
        return self._source.receipt()


class _BreadthReceiptProxy(_RowBankProxy):
    """Expose the same executable rows through the A5c breadth receipt seam."""

    def __init__(self, source: object) -> None:
        super().__init__(source)
        authentication = source.receipt()["authentication"]
        self.receipt_sha256 = "9" * 64
        self._receipt = {
            "receipt_sha256": self.receipt_sha256,
            "source": {
                "bridge_compiled_inputs_sha256": authentication[
                    "compiled_inputs_sha256"
                ]
            },
        }

    def receipt(self) -> dict[str, object]:
        return json.loads(json.dumps(self._receipt))


def _desired_states(inputs: Tensor) -> Tensor:
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


def _install_fake_fitter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gains: dict[float, float],
) -> list[tuple[object, object, float]]:
    calls: list[tuple[object, object, float]] = []
    real_authenticate_catalog = a5c._authenticate_lowering_catalog

    def fake_fit(fit_rows, eval_rows, **kwargs):
        ridge = float(kwargs["ridge"])
        calls.append((fit_rows, eval_rows, ridge))
        assert kwargs["generator_rank"] == A5C_FIXED_GENERATOR_RANK
        assert not (set(fit_rows.row_keys) & set(eval_rows.row_keys))
        return _Fit(ridge, gains[ridge], len(calls))

    def fake_apply(inputs: Tensor, plan: object) -> Tensor:
        # CV and the returned executable must both exercise the same float32
        # input/factor arithmetic path as Gemma3ModalGeneratorGraphExecutor.
        assert inputs.dtype == torch.float32
        return _desired_states(inputs).to(dtype=inputs.dtype) * plan.gain

    def authenticate_test_catalog(values, node_order, *, label):
        if all(
            isinstance(value, (_SourceLowering, _FittedLowering))
            for value in values.values()
        ):
            return dict(values)
        return real_authenticate_catalog(values, node_order, label=label)

    monkeypatch.setattr(a5c, "fit_frozen_basis_coordinate_generators", fake_fit)
    monkeypatch.setattr(a5c, "apply_modal_generator", fake_apply)
    monkeypatch.setattr(
        a5c, "_authenticate_lowering_catalog", authenticate_test_catalog
    )
    return calls


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bridge=None,
    gains: dict[float, float] | None = None,
    source_lowerings=None,
):
    actual_bridge = _build() if bridge is None else bridge
    inputs = next(iter(actual_bridge.all_rows.rows_by_fragment.values())).inputs
    native = _desired_states(inputs)
    correction_base = 0.2 * native
    # The source/frozen graph contributes another 0.5 * native.  Candidate
    # fits must replace that contribution from the correction base, not add to
    # the already-corrected frozen state.
    frozen = correction_base + 0.5 * native
    actual_gains = (
        {0.0: 0.4, 0.1: 0.8, 1.0: 1.2} if gains is None else gains
    )
    calls = _install_fake_fitter(monkeypatch, gains=actual_gains)
    adapter = _HeadAdapter()
    source_graph = _Graph("source-graph")
    actual_source_lowerings = (
        {name: _SourceLowering(name) for name in NODE_ORDER}
        if source_lowerings is None
        else source_lowerings
    )
    result = select_a5c_family_disjoint_ridge(
        bridge=actual_bridge,
        source_graph=source_graph,
        source_lowerings_by_node=actual_source_lowerings,
        adapter=adapter,
        native_block_states=native,
        frozen_compiled_block_states=frozen,
        compiled_correction_base_states=correction_base,
        output_boundary="layer.17.mlp.delta",
        ridge_grid=tuple(actual_gains),
        final_head_chunk_rows=2,
        final_head_token_locality_lineage_sha256="f" * 64,
    )
    return result, calls, adapter, inputs, native, correction_base, frozen


def test_cv_startup_authenticates_real_modal_generator_lowerings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_lowering, _inputs = _real_lowering()
    lowerings = {name: real_lowering for name in NODE_ORDER}

    result, *_ = _run(monkeypatch, source_lowerings=lowerings)

    authenticated = a5c._authenticate_lowering_catalog(
        lowerings,
        NODE_ORDER,
        label="regression source lowering",
    )
    expected_hashes = {
        name: real_lowering.artifact_sha256 for name in NODE_ORDER
    }
    assert {name: value.artifact_sha256 for name, value in authenticated.items()} == (
        expected_hashes
    )
    assert result.receipt()["source"]["source_lowering_sha256_by_node"] == (
        expected_hashes
    )
    assert all(
        isinstance(value, a5c.ModalGeneratorLowering)
        for value in authenticated.values()
    )
    assert all(value is not real_lowering for value in authenticated.values())


def test_fit_hashes_authenticate_real_rank16_lowerings() -> None:
    graph, lowerings = _real_rank16_runtime()
    fit = SimpleNamespace(graph_plan=graph, lowerings_by_node=lowerings)

    graph_sha256, lowering_hashes = a5c._fit_hashes(
        fit, graph.traversal_order
    )

    assert graph_sha256 == graph.artifact_sha256
    assert lowering_hashes == {
        name: lowerings[name].artifact_sha256
        for name in graph.traversal_order
    }


def test_real_lowering_authentication_rejects_fake_and_nested_tensor_drift() -> None:
    with pytest.raises(TypeError, match="must be a ModalGeneratorLowering"):
        a5c._authenticate_modal_generator_lowering(
            _SourceLowering("fake"), label="fake source"
        )

    real_lowering, _inputs = _real_lowering()
    real_lowering.coordinate_generator_plan.factors.input_factor[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="does not match tensor"):
        a5c._authenticate_modal_generator_lowering(
            real_lowering, label="mutated source"
        )


def test_nested_cv_selects_ridge_and_refits_all_examples_without_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, adapter, inputs, native, correction_base, frozen = _run(
        monkeypatch
    )

    assert result.use_frozen_fallback is False
    assert result.selected_ridge == 0.1
    assert len(calls) == 3 * 7 + 1
    final_fit_rows, final_eval_rows, final_ridge = calls[-1]
    assert final_ridge == 0.1
    assert final_fit_rows.observations == 28
    assert final_fit_rows.sequences == 14
    assert final_eval_rows.observations == 14
    assert not (set(final_fit_rows.row_keys) & set(final_eval_rows.row_keys))
    for rows in final_fit_rows.rows_by_fragment.values():
        assert float(rows.fisher_weights.sum().item()) == pytest.approx(1.0)

    correction = result.predict_correction(inputs)
    assert correction.dtype == torch.float32
    candidate_state = correction_base.to(dtype=correction.dtype) + correction
    torch.testing.assert_close(
        candidate_state,
        native.to(dtype=candidate_state.dtype),
        rtol=1e-6,
        atol=1e-7,
    )
    # This is the regression lock: adding the selected correction to the
    # frozen state would double-count the old graph and miss the native state.
    assert not torch.allclose(
        frozen.to(dtype=correction.dtype) + correction,
        native.to(dtype=correction.dtype),
    )
    receipt = result.receipt()
    assert receipt["selection"]["winner_family_equal_final_head_kl"] < 1e-12
    assert receipt["final_refit"]["fit_uses_all_outer_training_examples"] is True
    assert receipt["final_refit"]["descriptive_eval_is_independent"] is False
    assert receipt["final_refit"]["descriptive_eval_used_for_selection"] is False
    assert receipt["configuration"]["final_head_chunk_rows"] == 2
    assert receipt["source"]["bridge_compiled_inputs_sha256"] == (
        a5b_tensor_sha256(inputs)
    )
    assert receipt["source"]["bridge_compiled_inputs_sha256"] != receipt[
        "source"
    ]["all_rows_input_sha256"]
    assert (
        receipt["configuration"]["candidate_generator_execution_dtype"]
        == "torch.float32"
    )
    cast_audit = receipt["source"]["candidate_runtime_input_cast"]
    assert cast_audit["source_dtype"] == "torch.float64"
    assert cast_audit["runtime_dtype"] == "torch.float32"
    assert cast_audit["passed"] is True
    assert len(cast_audit["lineage_sha256"]) == 64
    assert adapter.maximum_rows <= 2
    assert not _contains_tensor(receipt)
    assert validate_a5c_family_ridge_cv_receipt(receipt) == receipt


def test_structural_authenticated_row_bank_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = _RowBankProxy(_build())
    assert isinstance(proxy, a5c.A5cAuthenticatedRowBank)

    result, *_ = _run(monkeypatch, bridge=proxy)

    assert result.selected_ridge == 0.1
    assert result.receipt()["source"]["bridge_receipt_sha256"] == (
        proxy.receipt_sha256
    )


def test_breadth_receipt_propagates_bridge_domain_input_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build()
    proxy = _BreadthReceiptProxy(source)

    result, *_ = _run(monkeypatch, bridge=proxy)

    receipt = result.receipt()
    expected = source.receipt()["authentication"]["compiled_inputs_sha256"]
    assert receipt["source"]["bridge_compiled_inputs_sha256"] == expected
    assert receipt["source"]["bridge_compiled_inputs_sha256"] != receipt[
        "source"
    ]["all_rows_input_sha256"]


def test_chunked_final_head_kl_is_chunk_size_invariant() -> None:
    generator = torch.Generator().manual_seed(8_507)
    native = torch.randn(17, 182, generator=generator)
    candidate = native + 0.05 * torch.randn(
        17, 182, generator=generator
    )
    weights = torch.linspace(1.0, 2.0, 17, dtype=torch.float64)
    weights /= weights.sum()

    rowwise_adapter = _HeadAdapter()
    rowwise = a5c._chunked_final_head_kl(
        rowwise_adapter, native, candidate, weights, chunk_rows=1
    )
    batched_adapter = _HeadAdapter()
    batched = a5c._chunked_final_head_kl(
        batched_adapter, native, candidate, weights, chunk_rows=8
    )

    assert rowwise == pytest.approx(batched, rel=1e-12, abs=1e-15)
    assert rowwise_adapter.maximum_rows == 1
    assert batched_adapter.maximum_rows == 8


def test_cross_family_input_signatures_are_removed_from_fit_and_reported(
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
    bridge = _build(compiled_inputs=modified.contiguous())

    result, calls, *_ = _run(monkeypatch, bridge=bridge)

    first_candidate = result.receipt()["candidates"][0]
    assert {
        fold["fit_rows_removed_for_signature_overlap"]
        for fold in first_candidate["folds"]
    } == {6}
    assert all(
        fold["input_signature_overlap_count"] == 0
        and fold["input_signatures_disjoint"] is True
        for fold in first_candidate["folds"]
    )
    for fit_rows, eval_rows, _ridge in calls[:-1]:
        fit_signatures = set(a5c._input_row_signatures(fit_rows))
        eval_signatures = set(a5c._input_row_signatures(eval_rows))
        assert not (fit_signatures & eval_signatures)


def test_signature_purge_fails_if_the_bounded_bank_leaves_no_fit_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _coordinates, _fisher, _row_keys, _ownership = _rows()
    bridge = _build(compiled_inputs=torch.zeros_like(inputs).contiguous())
    _install_fake_fitter(monkeypatch, gains={0.0: 0.4})
    source_graph = _Graph("source-graph")
    source_lowerings = {
        name: _SourceLowering(name) for name in NODE_ORDER
    }
    native = _desired_states(inputs)
    with pytest.raises(ValueError, match="row subset is empty"):
        select_a5c_family_disjoint_ridge(
            bridge=bridge,
            source_graph=source_graph,
            source_lowerings_by_node=source_lowerings,
            adapter=_HeadAdapter(),
            native_block_states=native,
            frozen_compiled_block_states=native,
            compiled_correction_base_states=torch.zeros_like(native),
            output_boundary="layer.17.mlp.delta",
            ridge_grid=(0.0,),
            final_head_chunk_rows=2,
            final_head_token_locality_lineage_sha256="f" * 64,
        )


def test_frozen_fallback_is_explicit_and_prediction_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, _adapter, inputs, *_ = _run(
        monkeypatch,
        gains={0.0: 0.0, 0.1: 0.1},
    )

    assert result.use_frozen_fallback is True
    assert result.selected_ridge is None
    assert result.correction_fit is None
    assert len(calls) == 2 * 7
    with pytest.raises(RuntimeError, match="execute the source graph"):
        result.predict_correction(inputs)
    receipt = result.receipt()
    assert receipt["final_refit"]["performed"] is False
    assert receipt["selection"]["selected_ridge"] is None


def test_self_rehashed_selection_contradiction_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, *_ = _run(monkeypatch)
    receipt = result.receipt()
    receipt["selection"]["use_frozen_fallback"] = True
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        a5c._REPORT_DOMAIN
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="selection contradicts"):
        validate_a5c_family_ridge_cv_receipt(receipt)
