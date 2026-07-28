from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)
from fisher_graph.prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)

from test_gemma3_full_mlp_stack_executor import _full_replacements
from test_gemma3_modal_generator_executor import _adapter, _batch, _plan


def _candidate_catalog(
    adapter: object,
    *,
    label: str,
    scale: float,
) -> tuple[Gemma3ModalGeneratorReplacement, ...]:
    source_sha256 = adapter.model_fingerprint()
    replacements = []
    for ordinal, layer in enumerate(adapter.layers):
        transformer = layer.transformer
        assert transformer is not None
        stage = next(
            value
            for value in transformer.stages
            if value.kind == "feed_forward"
        )
        input_factor = scale * torch.tensor(
            (
                (0.5, -0.2),
                (0.1, 0.4),
                (-0.3, 0.25),
                (0.2, 0.15),
            ),
            dtype=torch.float64,
        )
        output_factor = torch.tensor(
            (
                (0.4, -0.1, 0.3, 0.2),
                (-0.2, 0.5, 0.1, -0.4),
            ),
            dtype=torch.float64,
        )
        bias = torch.full(
            (4,),
            scale * (ordinal + 1) / 20,
            dtype=torch.float64,
        )
        replacements.append(
            Gemma3ModalGeneratorReplacement(
                layer_ordinal=ordinal,
                removed_mode_indices=tuple(range(6)),
                generator_plan=_plan(
                    source_model_sha256=source_sha256,
                    generator_id=f"{label}.layer.{ordinal}",
                    input_site=stage.normalized_input_site,
                    output_site=stage.operator_output_site,
                    input_factor=input_factor,
                    output_factor=output_factor,
                    bias=bias,
                ),
            )
        )
    return tuple(replacements)


def _full_rank_candidate_catalog(
    adapter: object,
) -> tuple[Gemma3ModalGeneratorReplacement, ...]:
    source_sha256 = adapter.model_fingerprint()
    replacements = []
    for ordinal, layer in enumerate(adapter.layers):
        transformer = layer.transformer
        assert transformer is not None
        stage = next(
            value
            for value in transformer.stages
            if value.kind == "feed_forward"
        )
        input_factor = torch.eye(4, dtype=torch.float64)
        input_factor[0, 1] = 0.15 * (ordinal + 1)
        output_factor = torch.tensor(
            (
                (0.8, -0.1, 0.2, 0.0),
                (0.1, 0.7, -0.2, 0.1),
                (-0.1, 0.2, 0.9, -0.2),
                (0.0, -0.1, 0.3, 0.6),
            ),
            dtype=torch.float64,
        )
        replacements.append(
            Gemma3ModalGeneratorReplacement(
                layer_ordinal=ordinal,
                removed_mode_indices=tuple(range(6)),
                generator_plan=_plan(
                    source_model_sha256=source_sha256,
                    generator_id=f"full-rank.layer.{ordinal}",
                    input_site=stage.normalized_input_site,
                    output_site=stage.operator_output_site,
                    input_factor=input_factor,
                    output_factor=output_factor,
                    bias=torch.full(
                        (4,),
                        (ordinal + 1) / 50,
                        dtype=torch.float64,
                    ),
                ),
            )
        )
    return tuple(replacements)


def _switcher() -> tuple[object, PreparedGemma3FullMLPStackSwitcher]:
    adapter = _adapter()
    return adapter, PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {
            "compact-a": _candidate_catalog(
                adapter,
                label="compact-a",
                scale=0.75,
            ),
            "compact-b": _candidate_catalog(
                adapter,
                label="compact-b",
                scale=1.25,
            ),
        },
    )


def test_switches_complete_named_stacks_and_hot_forward_never_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    batch = _batch()
    native_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    native_logits = adapter.module(**batch.model_inputs).logits.detach().clone()
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {
            "compact-a": _candidate_catalog(
                adapter,
                label="compact-a",
                scale=0.75,
            ),
            "compact-b": _candidate_catalog(
                adapter,
                label="compact-b",
                scale=1.25,
            ),
        },
    )

    def fail_hash() -> str:
        raise AssertionError("hot path attempted to hash the model")

    monkeypatch.setattr(adapter, "model_fingerprint", fail_hash)

    assert switcher.scopes == ("native", "compact-a", "compact-b")
    assert switcher.active_scope == "native"
    switcher.switch("compact-a")
    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == tuple(switcher.prepared_stacks[0])
    first = switcher(**batch.model_inputs).logits.detach().clone()

    switcher.switch("compact-b")
    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == tuple(switcher.prepared_stacks[1])
    second = switcher(**batch.model_inputs).logits.detach().clone()
    assert not torch.equal(first, second)

    switcher.switch("native")
    restored = switcher(**batch.model_inputs).logits.detach()
    torch.testing.assert_close(restored, native_logits, rtol=0.0, atol=0.0)
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        native_mlps
    )


def test_fused_full_rank_scope_matches_factorized_and_reduces_work() -> None:
    adapter = _adapter()
    batch = _batch()
    native_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {"factorized-full": _full_rank_candidate_catalog(adapter)},
        fused_variants={"fused-full": "factorized-full"},
    )

    assert switcher.scopes == (
        "native",
        "factorized-full",
        "fused-full",
    )
    assert switcher.factorized_candidate_names == ("factorized-full",)
    assert switcher.fused_variant_names == ("fused-full",)
    assert dict(switcher.fused_variants) == {
        "fused-full": "factorized-full"
    }
    assert all(
        isinstance(module, torch.nn.Linear)
        for module in switcher.prepared_stacks[1]
    )

    switcher.switch("factorized-full")
    factorized = switcher(**batch.model_inputs).logits.detach().clone()
    switcher.switch("fused-full")
    fused = switcher(**batch.model_inputs).logits.detach().clone()
    torch.testing.assert_close(fused, factorized, rtol=2e-6, atol=2e-6)

    factorized_accounting = switcher.scope_accounting["factorized-full"]
    fused_accounting = switcher.scope_accounting["fused-full"]
    assert factorized_accounting.implementation == "factorized"
    assert fused_accounting.implementation == "fused"
    assert factorized_accounting.learned_parameter_count == 3 * (32 + 4)
    assert fused_accounting.learned_parameter_count == 3 * (16 + 4)
    assert factorized_accounting.linear_macs_per_token == 3 * 32
    assert fused_accounting.linear_macs_per_token == 3 * 16
    assert switcher.scope_parameter_counts["fused-full"] < (
        switcher.scope_parameter_counts["factorized-full"]
    )
    assert switcher.scope_macs_per_token["fused-full"] < (
        switcher.scope_macs_per_token["factorized-full"]
    )

    switcher.close()
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        native_mlps
    )


def test_context_exit_and_close_restore_native_after_failure() -> None:
    adapter, switcher = _switcher()
    native_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)

    with pytest.raises(RuntimeError, match="sentinel benchmark failure"):
        with switcher as prepared:
            prepared.switch("compact-a")
            assert prepared.active_scope == "compact-a"
            raise RuntimeError("sentinel benchmark failure")

    assert switcher.closed is True
    assert switcher.active_scope == "native"
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        native_mlps
    )
    switcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        switcher.switch("compact-a")
    with pytest.raises(RuntimeError, match="closed"):
        switcher(**_batch().model_inputs)


def test_forward_switch_and_context_reentrancy_are_rejected() -> None:
    adapter, switcher = _switcher()
    batch = _batch()
    switcher.switch("compact-a")

    def recurse(
        _module: object,
        _arguments: tuple[object, ...],
        _output: object,
    ) -> None:
        switcher(**batch.model_inputs)

    handle = adapter.module.model.embed_tokens.register_forward_hook(recurse)
    try:
        with pytest.raises(RuntimeError, match="not reentrant"):
            switcher(**batch.model_inputs)
    finally:
        handle.remove()
    assert switcher(**batch.model_inputs).logits.shape == (2, 4, 17)

    def switch_during_forward(
        _module: object,
        _arguments: tuple[object, ...],
        _output: object,
    ) -> None:
        switcher.switch("native")

    handle = adapter.module.model.embed_tokens.register_forward_hook(
        switch_during_forward
    )
    try:
        with pytest.raises(RuntimeError, match="during a forward"):
            switcher(**batch.model_inputs)
    finally:
        handle.remove()
    assert switcher.active_scope == "compact-a"

    with switcher:
        with pytest.raises(RuntimeError, match="context is not reentrant"):
            switcher.__enter__()
    assert switcher.closed is True


def test_rejects_invalid_catalogs_scopes_and_live_scope_drift() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="nonempty mapping"):
        PreparedGemma3FullMLPStackSwitcher(adapter, {})
    with pytest.raises(ValueError, match="reserved"):
        PreparedGemma3FullMLPStackSwitcher(
            adapter,
            {"native": _full_replacements(adapter)},
        )
    with pytest.raises(ValueError, match="one replacement per Gemma layer"):
        PreparedGemma3FullMLPStackSwitcher(
            adapter,
            {"partial": _full_replacements(adapter)[:-1]},
        )
    with pytest.raises(ValueError, match="factorized candidate"):
        PreparedGemma3FullMLPStackSwitcher(
            adapter,
            {"factorized": _full_replacements(adapter)},
            fused_variants={"fused": "missing"},
        )
    with pytest.raises(ValueError, match="collide"):
        PreparedGemma3FullMLPStackSwitcher(
            adapter,
            {"factorized": _full_replacements(adapter)},
            fused_variants={"factorized": "factorized"},
        )

    drift_adapter, switcher = _switcher()
    native_mlps = tuple(
        layer.mlp for layer in drift_adapter.module.model.layers
    )
    with pytest.raises(ValueError, match="unknown"):
        switcher.switch("missing")
    with pytest.raises(TypeError, match="string"):
        switcher.switch(7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frozen eval"):
        switcher.train()

    switcher.switch("compact-a")
    drift_adapter.module.model.layers[0].mlp = native_mlps[0]
    with pytest.raises(RuntimeError, match="identity drifted"):
        switcher.switch("compact-b")
    switcher.close()
    assert tuple(
        layer.mlp for layer in drift_adapter.module.model.layers
    ) == native_mlps
