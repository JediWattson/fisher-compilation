from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.generator_causal_fingerprints import (
    GeneratorCausalFingerprintProvenance,
    GeneratorInterventionLogitBatch,
    collect_generator_causal_fingerprints,
)
from fisher_graph.gemma3_generator_causal_fingerprint_experiment import (
    _causal_analysis_lineage,
    _generator_causal_summaries,
    _generator_fit_lineage,
    _pairwise_similarities,
    _prompts_in_frozen_order,
    _live_split_matches_normalized_refit,
)
from fisher_graph.gemma3_generator_causal_fingerprint_artifact import (
    build_gemma3_generator_causal_fingerprint_payload,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime() -> object:
    source_model = _sha("model")
    source_fits = tuple(_sha(f"source-fit:{index}") for index in range(18))
    deployed_fits = tuple(
        source_fits[index] if index < 10 else _sha(f"refit-fit:{index}")
        for index in range(18)
    )
    plans = tuple(_sha(f"plan:{index}") for index in range(18))
    prompts = tuple(_sha(f"prompt:{index}") for index in range(5))
    return SimpleNamespace(
        replacements=tuple(range(18)),
        refit_start_layer=10,
        layer_lineage=tuple(
            {
                "layer_ordinal": index,
                "layer_id": f"layer.{index}",
            }
            for index in range(18)
        ),
        source_fit_sha256s=source_fits,
        deployed_fit_sha256s=deployed_fits,
        generator_plan_sha256s=plans,
        base_scientific_payload_sha256=_sha("base-scientific"),
        refit_scientific_payload_sha256=_sha("refit-scientific"),
        analysis_split={
            "serialized_sha256": _sha("split"),
            "content_sha256": prompts,
            "example_count": 5,
            "logical_valid_tokens": 10,
            "supervised_tokens": 10,
        },
    )


def _analysis(runtime: object) -> object:
    generator_ids = tuple(f"generator:{index}" for index in range(18))
    prompts = runtime.analysis_split["content_sha256"]
    baseline_row = torch.linspace(3.0, -1.5, 10, dtype=torch.float64)
    baseline = baseline_row.repeat(5, 1, 1)
    muted = {}
    for index, generator_id in enumerate(generator_ids):
        value = baseline.clone()
        value[:, :, index % 9] -= (index + 1) * torch.tensor(
            [0.01, 0.02, 0.03, 0.04, 0.05],
            dtype=torch.float64,
        )[:, None]
        muted[generator_id] = value
    batch = GeneratorInterventionLogitBatch(
        example_ids=prompts,
        baseline_logits=baseline,
        muted_logits_by_generator=muted,
        targets=torch.zeros((5, 1), dtype=torch.int64),
        supervised_mask=torch.ones((5, 1), dtype=torch.bool),
    )
    return collect_generator_causal_fingerprints(
        (batch,),
        generator_ids=generator_ids,
        provenance=GeneratorCausalFingerprintProvenance(
            source_model_sha256=_sha("model"),
            generator_catalog_sha256=_sha("catalog"),
            evaluation_split_sha256=_sha("split"),
            objective_sha256=_sha("objective"),
        ),
        anchor_count=8,
        top_importance_count=5,
    )


def test_selects_exact_frozen_prompt_membership_in_declared_order() -> None:
    export = SimpleNamespace(
        prompts=("alpha", "beta", "gamma"),
        prompt_sha256s=("a", "b", "c"),
    )
    assert _prompts_in_frozen_order(export, ("c", "a")) == (
        "gamma",
        "alpha",
    )
    with pytest.raises(ValueError, match="absent"):
        _prompts_in_frozen_order(export, ("missing",))


def test_matches_live_tokenization_to_normalized_refit_split() -> None:
    content = (_sha("content:0"), _sha("content:1"))
    live = {
        "serialized_sha256": _sha("serialized"),
        "sequences": 2,
        "valid_tokens": {"total": 20},
        "supervised_positions": {"total": 18},
        "content_sha256": content,
    }
    frozen = {
        "serialized_sha256": _sha("serialized"),
        "example_count": 2,
        "logical_valid_tokens": 20,
        "supervised_tokens": 18,
        "content_sha256": content,
    }
    _live_split_matches_normalized_refit(live, frozen)
    frozen["supervised_tokens"] = 17
    with pytest.raises(ValueError, match="normalized frozen"):
        _live_split_matches_normalized_refit(live, frozen)


def test_translates_core_analysis_to_strict_tensor_free_artifact_rows() -> None:
    runtime = _runtime()
    analysis = _analysis(runtime)
    lineage = _generator_fit_lineage(runtime)
    summaries = _generator_causal_summaries(
        analysis,
        runtime,
        prompt_content_sha256s=runtime.analysis_split["content_sha256"],
        example_ids=runtime.analysis_split["content_sha256"],
    )
    pairs = _pairwise_similarities(analysis, runtime, summaries)
    split = {
        "role": "adaptive_open_development_generator_family_discovery",
        **runtime.analysis_split,
        "membership_exact": True,
        "assurance": "caller_declared_self_attested",
        "externally_authenticated": False,
        "heldout_confirmation": False,
        "used_for_adaptive_analysis": True,
        "used_for_generator_fit": False,
        "used_for_generator_selection": False,
    }
    payload = build_gemma3_generator_causal_fingerprint_payload(
        model={
            "model_id": "google/gemma-3-270m",
            "requested_revision": "1" * 40,
            "resolved_commit": "1" * 40,
            "adapter_model_fingerprint": _sha("model"),
            "local_files_only": True,
        },
        frozen_sources={
            "base_full_stack": {
                "schema": (
                    "fisher_graph."
                    "gemma3_full_native_mlp_stack_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("base-file"),
                "scientific_payload_sha256": _sha("base-scientific"),
                "frozen_before_analysis": True,
            },
            "sequential_refit": {
                "schema": (
                    "fisher_graph."
                    "gemma3_sequential_full_mlp_stack_refit_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("refit-file"),
                "scientific_payload_sha256": _sha("refit-scientific"),
                "frozen_before_analysis": True,
            },
        },
        analysis_split=split,
        causal_analysis_lineage=_causal_analysis_lineage(analysis),
        generator_fit_lineage=lineage,
        generator_causal_summaries=summaries,
        pairwise_similarities=pairs,
    )

    assert len(payload["generator_causal_summaries"]) == 18
    assert len(payload["pairwise_similarities"]) == 153
    assert payload["scientific_status"]["authorizes_model_mutation"] is False
    assert payload["safety"]["contains_logits"] is False
