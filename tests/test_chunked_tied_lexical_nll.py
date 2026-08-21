from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fisher_graph.chunked_tied_lexical_nll import (
    BatchObjectiveAxis,
    BatchObjectiveExample,
    ChunkedTiedLexicalNLLConfig,
    DENSE_PARITY_ABSOLUTE_TOLERANCE,
    DENSE_PARITY_RELATIVE_TOLERANCE,
    DENSE_PARITY_SELECTED_PREDICTION_ROW_COUNT,
    DENSE_PARITY_SELECTOR_ID,
    FixtureChunkedTiedLexicalNLLConfig,
    PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE,
    PRODUCTION_VOCABULARY_CHUNK_SIZE,
    bind_tied_readout_weight,
    chunked_tied_lexical_nll_kernel_receipt,
    compute_chunked_tied_lexical_nll_vjp,
    fixture_chunked_softcap_nll_vjp_rows,
    objective_row_stream_sha256,
    parse_batch_objective_axis_payload,
    parse_chunked_tied_lexical_nll_config_payload,
    parse_chunked_tied_lexical_nll_kernel_receipt_payload,
    parse_chunked_tied_lexical_nll_receipt_payload,
    parse_dense_parity_smoke_receipt_payload,
    parse_tied_readout_weight_receipt_payload,
    run_dense_full_logit_parity_smoke,
    tensor_stream_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _axis(
    *,
    batch: int,
    token_count: int,
    role: str = "A_chart_transport_fit",
    suffix: str = "x",
) -> BatchObjectiveAxis:
    examples = tuple(
        BatchObjectiveExample(
            family_id=f"family-{index}",
            example_id=f"{role}-{suffix}-{index}",
            target_token_count=token_count,
            frozen_objective_row_stream_sha256=objective_row_stream_sha256(
                role=role,
                family_id=f"family-{index}",
                example_id=f"{role}-{suffix}-{index}",
                token_count=token_count,
            ),
        )
        for index in range(batch)
    )
    return BatchObjectiveAxis(
        role=role,
        examples=examples,
        role_score_objective_receipt_sha256=_sha(f"{role}-objective"),
        role_ordered_objective_row_catalog_sha256=_sha(f"{role}-catalog"),
    )


def _binding(weight: torch.Tensor, *, suffix: str = "x"):
    return bind_tied_readout_weight(
        weight,
        readout_state_sha256=_sha(f"readout-{suffix}"),
        tied_lexical_lineage_sha256=_sha(f"lineage-{suffix}"),
    )


def _compute(
    normalized: torch.Tensor,
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    cap: float | None,
    *,
    axis: BatchObjectiveAxis | None = None,
    mask: torch.Tensor | None = None,
    binding=None,
):
    axis = axis or _axis(
        batch=int(normalized.shape[0]),
        token_count=int(normalized.shape[1]),
    )
    mask = mask if mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
    binding = binding or _binding(weight)
    result = compute_chunked_tied_lexical_nll_vjp(
        terminal_normalized_output=normalized,
        input_token_ids=input_ids,
        valid_token_mask=mask,
        batch_axis=axis,
        tied_readout_weight=binding,
        logit_softcap=cap,
    )
    return result, axis, mask, binding


def _dense_reference(
    normalized: torch.Tensor,
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    cap: float | None,
) -> tuple[float, torch.Tensor]:
    live = normalized.detach().clone().requires_grad_(True)
    raw = live @ weight.T
    logits = raw if cap is None else torch.tanh(raw / cap) * cap
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, weight.shape[0]),
        input_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    (gradient,) = torch.autograd.grad(loss, live)
    return float(loss.detach()), gradient.detach()


@pytest.mark.parametrize("cap", [None, 3.25])
def test_production_chunked_nll_and_vjp_match_dense_full_logits(
    cap: float | None,
) -> None:
    generator = torch.Generator().manual_seed(18)
    normalized = torch.randn(2, 6, 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(13, 5, generator=generator, dtype=torch.float64)
    input_ids = torch.randint(0, 13, (2, 6), generator=generator)
    expected_nll, expected_gradient = _dense_reference(
        normalized,
        input_ids,
        weight,
        cap,
    )
    result, axis, _mask, binding = _compute(
        normalized,
        input_ids,
        weight,
        cap,
    )
    assert result.receipt.summed_next_token_nll == pytest.approx(
        expected_nll,
        rel=1e-13,
        abs=1e-13,
    )
    assert torch.allclose(
        result.terminal_gradient,
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not bool(result.terminal_gradient[:, -1].count_nonzero())
    assert result.receipt.batch_objective_axis_sha256 == axis.sha256
    assert (
        result.receipt.tied_readout_binding_sha256
        == binding.receipt.receipt_sha256
    )


def test_default_production_chunks_cross_both_frozen_boundaries() -> None:
    generator = torch.Generator().manual_seed(91)
    vocabulary = PRODUCTION_VOCABULARY_CHUNK_SIZE + 17
    normalized = torch.randn(2, 21, 3, generator=generator, dtype=torch.float64)
    weight = torch.randn(vocabulary, 3, generator=generator, dtype=torch.float64)
    input_ids = torch.randint(0, vocabulary, (2, 21), generator=generator)
    expected_nll, expected_gradient = _dense_reference(
        normalized,
        input_ids,
        weight,
        5.0,
    )
    result, _axis_value, _mask, _binding_value = _compute(
        normalized,
        input_ids,
        weight,
        5.0,
    )
    assert result.receipt.prediction_row_count == 40
    assert result.receipt.summed_next_token_nll == pytest.approx(
        expected_nll,
        rel=1e-12,
        abs=1e-12,
    )
    assert torch.allclose(
        result.terminal_gradient,
        expected_gradient,
        rtol=1e-11,
        atol=1e-11,
    )
    assert result.receipt.full_prediction_by_vocabulary_matrix_materialized is False
    assert result.receipt.maximum_single_logit_matrix_scalar_count == (
        PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE
        * PRODUCTION_VOCABULARY_CHUNK_SIZE
    )


def test_fixture_numeric_kernel_is_explicitly_separate() -> None:
    generator = torch.Generator().manual_seed(22)
    queries = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    weight = torch.randn(19, 4, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 19, (8,), generator=generator)
    config = FixtureChunkedTiedLexicalNLLConfig(
        prediction_position_chunk_size=3,
        vocabulary_chunk_size=5,
    )
    loss, gradient = fixture_chunked_softcap_nll_vjp_rows(
        queries,
        labels,
        weight,
        logit_softcap=4.0,
        config=config,
    )
    live = queries.detach().clone().requires_grad_(True)
    dense_loss = F.cross_entropy(
        torch.tanh((live @ weight.T) / 4.0) * 4.0,
        labels,
        reduction="sum",
    )
    expected = torch.autograd.grad(dense_loss, live)[0]
    assert loss == pytest.approx(
        float(dense_loss.detach()),
        rel=1e-13,
        abs=1e-13,
    )
    assert torch.allclose(gradient, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(TypeError, match="frozen production config"):
        compute_chunked_tied_lexical_nll_vjp(
            terminal_normalized_output=torch.zeros(1, 3, 4),
            input_token_ids=torch.tensor([[0, 1, 2]]),
            valid_token_mask=torch.ones(1, 3, dtype=torch.bool),
            batch_axis=_axis(batch=1, token_count=3),
            tied_readout_weight=_binding(torch.ones(19, 4)),
            logit_softcap=4.0,
            config=config,  # type: ignore[arg-type]
        )


def test_float32_terminal_seed_matches_float64_reference() -> None:
    generator = torch.Generator().manual_seed(3)
    normalized = torch.randn(1, 8, 7, generator=generator)
    weight = torch.randn(31, 7, generator=generator)
    input_ids = torch.randint(0, 31, (1, 8), generator=generator)
    expected_nll, expected_gradient = _dense_reference(
        normalized.to(torch.float64),
        input_ids,
        weight.to(torch.float64),
        30.0,
    )
    result, *_ = _compute(normalized, input_ids, weight, 30.0)
    assert result.terminal_gradient.dtype == torch.float32
    assert result.receipt.summed_next_token_nll == pytest.approx(
        expected_nll,
        rel=2e-6,
        abs=2e-6,
    )
    assert torch.allclose(
        result.terminal_gradient.to(torch.float64),
        expected_gradient,
        rtol=2e-6,
        atol=2e-6,
    )


def test_batch_axis_matches_a_real_frozen_example_row_stream() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "examples/gemma3_chart_native_neighbor_fresh_manifest_v1.json")
        .read_text(encoding="utf-8")
    )
    role = "A_chart_transport_fit"
    frozen_role = manifest["roles"][role]
    member = frozen_role["ordered_members"][0]
    example = BatchObjectiveExample(
        family_id=member["family_id"],
        example_id=member["example_id"],
        target_token_count=member["target_token_count"],
        frozen_objective_row_stream_sha256=member[
            "objective_row_stream_sha256"
        ],
    )
    axis = BatchObjectiveAxis(
        role=role,
        examples=(example,),
        role_score_objective_receipt_sha256=frozen_role[
            "score_objective_receipt_sha256"
        ],
        role_ordered_objective_row_catalog_sha256=frozen_role[
            "ordered_objective_row_catalog_sha256"
        ],
    )
    assert (
        objective_row_stream_sha256(
            role=role,
            family_id=example.family_id,
            example_id=example.example_id,
            token_count=example.target_token_count,
        )
        == example.frozen_objective_row_stream_sha256
        == axis.batch_ordered_objective_row_catalog_sha256
    )
    assert parse_batch_objective_axis_payload(axis.canonical_payload()) == axis


def test_padding_mixed_lengths_and_row_catalog_drift_fail_closed() -> None:
    normalized = torch.zeros(1, 4, 2)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    weight = torch.ones(4, 2)
    mask = torch.tensor([[True, True, True, False]])
    with pytest.raises(ValueError, match="padding rows are forbidden"):
        _compute(normalized, input_ids, weight, 30.0, mask=mask)

    first = _axis(batch=1, token_count=3).examples[0]
    second = BatchObjectiveExample(
        family_id="family-1",
        example_id="different-length",
        target_token_count=4,
        frozen_objective_row_stream_sha256=objective_row_stream_sha256(
            role="A_chart_transport_fit",
            family_id="family-1",
            example_id="different-length",
            token_count=4,
        ),
    )
    with pytest.raises(ValueError, match="equal exact per-example lengths"):
        BatchObjectiveAxis(
            role="A_chart_transport_fit",
            examples=(first, second),
            role_score_objective_receipt_sha256=_sha("objective"),
            role_ordered_objective_row_catalog_sha256=_sha("catalog"),
        )

    with pytest.raises(ValueError, match="row stream drifted"):
        BatchObjectiveAxis(
            role="A_chart_transport_fit",
            examples=(replace(first, frozen_objective_row_stream_sha256=_sha("bad")),),
            role_score_objective_receipt_sha256=_sha("objective"),
            role_ordered_objective_row_catalog_sha256=_sha("catalog"),
        )


def test_bound_weight_detects_normal_and_alias_mutation_and_rejects_views() -> None:
    normalized = torch.zeros(1, 3, 2)
    input_ids = torch.tensor([[0, 1, 2]])

    weight = torch.ones(4, 2)
    binding = _binding(weight, suffix="normal")
    weight.add_(1.0)
    with pytest.raises(RuntimeError, match="changed after binding"):
        _compute(normalized, input_ids, weight, 30.0, binding=binding)

    alias_weight = torch.ones(4, 2)
    alias_binding = _binding(alias_weight, suffix="alias")
    alias_weight.data[0, 0] = 9.0
    with pytest.raises(RuntimeError, match="changed after binding"):
        alias_binding.validate_unchanged()

    backing = torch.ones(5, 2)
    with pytest.raises(ValueError, match="dedicated storage"):
        _binding(backing[:4], suffix="view")
    with pytest.raises(TypeError, match="bound tied-readout"):
        compute_chunked_tied_lexical_nll_vjp(
            terminal_normalized_output=normalized,
            input_token_ids=input_ids,
            valid_token_mask=torch.ones_like(input_ids, dtype=torch.bool),
            batch_axis=_axis(batch=1, token_count=3),
            tied_readout_weight=torch.ones(4, 2),  # type: ignore[arg-type]
            logit_softcap=30.0,
        )


def test_production_config_kernel_binding_and_nll_parsers_are_strict() -> None:
    config = ChunkedTiedLexicalNLLConfig()
    assert parse_chunked_tied_lexical_nll_config_payload(config.payload()) == config
    with pytest.raises(ValueError, match="must remain 32"):
        ChunkedTiedLexicalNLLConfig(prediction_position_chunk_size=31)
    with pytest.raises(ValueError, match="must remain 4096"):
        ChunkedTiedLexicalNLLConfig(vocabulary_chunk_size=4095)

    kernel = chunked_tied_lexical_nll_kernel_receipt()
    assert parse_chunked_tied_lexical_nll_kernel_receipt_payload(kernel) == kernel
    assert kernel["receipt_sha256"] == (
        "e79c953bd99483d87e8c09406388deecf2474a9dc20c468c0fd68c38d9deffdb"
    )
    assert kernel["maximum_single_logit_matrix_scalar_count"] == 32 * 4096

    normalized = torch.zeros(1, 3, 2, dtype=torch.float64)
    input_ids = torch.tensor([[0, 1, 2]])
    weight = torch.ones(4, 2, dtype=torch.float64)
    result, axis, _mask, binding = _compute(
        normalized,
        input_ids,
        weight,
        30.0,
    )
    assert (
        parse_tied_readout_weight_receipt_payload(
            binding.receipt.canonical_payload()
        )
        == binding.receipt
    )
    assert (
        parse_chunked_tied_lexical_nll_receipt_payload(
            result.receipt.canonical_payload(),
            expected_axis=axis,
            expected_weight=binding,
        )
        == result.receipt
    )
    tampered = result.receipt.canonical_payload()
    tampered["all_tokens_valid"] = False
    with pytest.raises(ValueError):
        parse_chunked_tied_lexical_nll_receipt_payload(tampered)


def test_dense_parity_is_fixed_and_bound_to_real_source_batch() -> None:
    generator = torch.Generator().manual_seed(44)
    normalized = torch.randn(1, 5, 4, generator=generator)
    weight = torch.randn(17, 4, generator=generator)
    input_ids = torch.randint(0, 17, (1, 5), generator=generator)
    result, axis, mask, binding = _compute(
        normalized,
        input_ids,
        weight,
        7.0,
    )
    receipt = run_dense_full_logit_parity_smoke(
        source_result=result,
        terminal_normalized_output=normalized,
        input_token_ids=input_ids,
        valid_token_mask=mask,
        batch_axis=axis,
        tied_readout_weight=binding,
        logit_softcap=7.0,
    )
    assert receipt.selector_id == DENSE_PARITY_SELECTOR_ID
    assert (
        receipt.selected_prediction_row_count
        == DENSE_PARITY_SELECTED_PREDICTION_ROW_COUNT
    )
    assert receipt.absolute_tolerance == DENSE_PARITY_ABSOLUTE_TOLERANCE
    assert receipt.relative_tolerance == DENSE_PARITY_RELATIVE_TOLERANCE
    assert (
        parse_dense_parity_smoke_receipt_payload(
            receipt.canonical_payload(),
            expected_source=result.receipt,
            expected_axis=axis,
            expected_weight=binding,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="digest"):
        replace(receipt, selected_labels_sha256=_sha("other-labels"))

    other_axis = _axis(batch=1, token_count=5, suffix="other")
    with pytest.raises(ValueError, match="batch axis"):
        parse_dense_parity_smoke_receipt_payload(
            receipt.canonical_payload(),
            expected_axis=other_axis,
        )
    other_binding = _binding(weight.detach().clone(), suffix="other")
    with pytest.raises(ValueError, match="tied weight"):
        parse_dense_parity_smoke_receipt_payload(
            receipt.canonical_payload(),
            expected_weight=other_binding,
        )


def test_full_gemma_vocabulary_float32_input_requires_float64_dense_reference() -> None:
    vocabulary = 262_144
    hidden_width = 8
    generator = torch.Generator().manual_seed(270_008)
    normalized = torch.randn(
        1,
        3,
        hidden_width,
        generator=generator,
        dtype=torch.float32,
    )
    weight = torch.randn(
        vocabulary,
        hidden_width,
        generator=generator,
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[2, 7, vocabulary - 11]], dtype=torch.int64)
    result, axis, mask, binding = _compute(
        normalized,
        input_ids,
        weight,
        None,
    )

    # Reproduce the rejected reference: at Gemma's full vocabulary, a
    # float32 dense reduction can differ materially from the frozen float64
    # production kernel even with only eight hidden coordinates.
    old_rows = normalized[:, :-1, :].reshape(-1, hidden_width).detach()
    old_rows = old_rows.clone().requires_grad_(True)
    old_loss = F.cross_entropy(
        old_rows @ weight.T,
        input_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    (old_gradient,) = torch.autograd.grad(old_loss, old_rows)
    chunk_gradient = result.terminal_gradient[:, :-1, :].reshape(
        -1,
        hidden_width,
    )
    assert not torch.allclose(
        chunk_gradient,
        old_gradient,
        rtol=DENSE_PARITY_RELATIVE_TOLERANCE,
        atol=DENSE_PARITY_ABSOLUTE_TOLERANCE,
    )

    receipt = run_dense_full_logit_parity_smoke(
        source_result=result,
        terminal_normalized_output=normalized,
        input_token_ids=input_ids,
        valid_token_mask=mask,
        batch_axis=axis,
        tied_readout_weight=binding,
        logit_softcap=None,
    )
    assert receipt.dense_reference_dtype == "torch.float64"
    assert receipt.loss_absolute_error < 1.0e-12
    assert receipt.gradient_max_absolute_error < 1.0e-12
    assert receipt.gradient_normalized_rmse < 1.0e-12


def test_swapped_same_shape_terminal_or_axis_cannot_reuse_source_receipt() -> None:
    generator = torch.Generator().manual_seed(71)
    first = torch.randn(1, 4, 3, generator=generator)
    second = torch.randn(1, 4, 3, generator=generator)
    weight = torch.randn(11, 3, generator=generator)
    ids = torch.randint(0, 11, (1, 4), generator=generator)
    result, axis, mask, binding = _compute(first, ids, weight, 5.0)
    with pytest.raises(ValueError, match="source batch"):
        run_dense_full_logit_parity_smoke(
            source_result=result,
            terminal_normalized_output=second,
            input_token_ids=ids,
            valid_token_mask=mask,
            batch_axis=axis,
            tied_readout_weight=binding,
            logit_softcap=5.0,
        )
    swapped_axis = _axis(batch=1, token_count=4, suffix="swapped")
    with pytest.raises(ValueError, match="expected batch axis"):
        parse_chunked_tied_lexical_nll_receipt_payload(
            result.receipt.canonical_payload(),
            expected_axis=swapped_axis,
        )


def test_tensor_hash_finiteness_scan_is_row_chunk_bounded(monkeypatch) -> None:
    value = torch.ones(101, 7)
    observed_counts: list[int] = []
    original = torch.isfinite

    def recording_isfinite(chunk: torch.Tensor) -> torch.Tensor:
        observed_counts.append(int(chunk.numel()))
        return original(chunk)

    monkeypatch.setattr(torch, "isfinite", recording_isfinite)
    digest = tensor_stream_sha256(value, row_chunk_size=11)
    assert len(digest) == 64
    assert observed_counts
    assert max(observed_counts) <= 11 * 7


def test_terminal_cotangent_pulls_back_like_full_logits() -> None:
    generator = torch.Generator().manual_seed(108)
    base = torch.randn(1, 5, 4, generator=generator, requires_grad=True)
    terminal = torch.sin(base) * 1.25
    weight = torch.randn(23, 4, generator=generator)
    ids = torch.randint(0, 23, (1, 5), generator=generator)
    binding = _binding(weight, suffix="vjp")
    result, _axis_value, _mask, _binding_value = _compute(
        terminal,
        ids,
        weight,
        9.0,
        binding=binding,
    )
    analytic_base_gradient = torch.autograd.grad(
        terminal,
        base,
        grad_outputs=result.terminal_gradient,
        retain_graph=True,
    )[0]
    logits = torch.tanh((terminal @ weight.T) / 9.0) * 9.0
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, weight.shape[0]),
        ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    dense_base_gradient = torch.autograd.grad(loss, base)[0]
    assert torch.allclose(
        analytic_base_gradient,
        dense_base_gradient,
        rtol=3e-6,
        atol=3e-6,
    )


def test_receipts_are_prompt_free_and_module_source_is_torch_inert() -> None:
    normalized = torch.zeros(1, 3, 2)
    ids = torch.tensor([[0, 1, 2]])
    weight = torch.ones(4, 2)
    result, _axis_value, _mask, _binding_value = _compute(
        normalized,
        ids,
        weight,
        3.0,
    )
    payload = result.receipt.canonical_payload()
    assert not any(isinstance(value, (list, tuple, torch.Tensor)) for value in payload.values())
    encoded = json.dumps(payload, sort_keys=True)
    assert "[[0, 1, 2]]" not in encoded
    assert "prompt" not in encoded.lower()

    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fisher_graph"
        / "chunked_tied_lexical_nll.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "torch" for alias in node.names)
        )
        for node in top_level_imports
    )
