from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph.gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)

from test_gemma3_modal_generator_executor import _plan


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _FakeFit:
    def __init__(
        self,
        *,
        ordinal: int,
        kind: str,
        source_model_sha256: str,
    ) -> None:
        self.superfragment = SimpleNamespace(
            layer_ordinal=ordinal,
            channel_indices=(0, 1, 2),
            source_model_sha256=source_model_sha256,
        )
        self.executable_plan = _plan(
            source_model_sha256=source_model_sha256,
            generator_id=f"layer.{ordinal}.{kind}",
            input_site=f"layer.{ordinal}.mlp.normalized_input",
            output_site=f"layer.{ordinal}.mlp.operator_output",
        )
        self.artifact_sha256 = _sha(f"fit:{kind}:{ordinal}")

    def validate_integrity(self) -> None:
        return None


def _fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, _FakeFit],
]:
    source_model = _sha("source-model")
    source_fits = tuple(
        _FakeFit(
            ordinal=ordinal,
            kind="source",
            source_model_sha256=source_model,
        )
        for ordinal in range(18)
    )
    refit_fits = tuple(
        _FakeFit(
            ordinal=ordinal,
            kind="refit",
            source_model_sha256=source_model,
        )
        for ordinal in range(10, 18)
    )
    lookup = {
        f"source:{ordinal}": source_fits[ordinal]
        for ordinal in range(18)
    }
    lookup.update(
        {
            f"refit:{ordinal}": refit_fits[ordinal - 10]
            for ordinal in range(10, 18)
        }
    )
    model = {
        "model_id": "google/gemma-3-270m",
        "resolved_commit": "1" * 40,
        "adapter_model_fingerprint": source_model,
    }
    base_scientific = _sha("base-scientific")
    base_file = _sha("base-file")
    base = {
        "scientific_payload_sha256": base_scientific,
        "model": model,
        "splits": {
            "partition": {
                "artifact_sha256": _sha("partition"),
                "selection_prompt_count": 2,
                "assessment_prompt_count": 3,
                "expected_prompt_count": 5,
            }
        },
        "generator_fits": tuple(
            {"key": f"source:{ordinal}"} for ordinal in range(18)
        ),
    }
    summaries = tuple(
        {
            "layer_ordinal": ordinal,
            "layer_id": f"model.layers.{ordinal}",
            "source_model_sha256": source_model,
            "source_fit_sha256": source_fits[ordinal].artifact_sha256,
            "dense_plan_sha256": (
                source_fits[ordinal].executable_plan.artifact_sha256
            ),
        }
        for ordinal in range(18)
    )
    refit = {
        "scientific_payload_sha256": _sha("refit-scientific"),
        "model": dict(model),
        "frozen_sources": {
            "full_stack": {
                "artifact_file_sha256": base_file,
                "scientific_payload_sha256": base_scientific,
            }
        },
        "source_layer_summaries": summaries,
        "splits": {
            "assessment": {
                "role": "open_development_assessment",
                "serialized_sha256": _sha("assessment-split"),
                "content_sha256": tuple(
                    _sha(f"assessment:{index}") for index in range(3)
                ),
                "example_count": 3,
                "logical_valid_tokens": 30,
                "supervised_tokens": 27,
            }
        },
        "evaluation": {
            "conditions": {
                "sequential_refit_full_stack": {
                    "nll_per_token": 2.5,
                    "delta_nll_per_token": 0.1,
                    "native_to_candidate_kl_per_token": 0.2,
                    "top1_agreement_to_native": 0.8,
                }
            }
        },
        "resource_accounting": {"logical_candidate_learned_parameters": 123},
        "refit_generator_fits": tuple(
            {"key": f"refit:{ordinal}"} for ordinal in range(10, 18)
        ),
        "layer_refits": tuple(
            {
                "layer_ordinal": ordinal,
                "source_fit_sha256": source_fits[ordinal].artifact_sha256,
                "refit_fit_sha256": refit_fits[
                    ordinal - 10
                ].artifact_sha256,
            }
            for ordinal in range(10, 18)
        ),
    }
    return base, refit, lookup


def test_restores_exact_prefix_source_and_suffix_refit_catalog() -> None:
    base, refit, lookup = _fixture()

    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        "base.pt",
        "refit.pt",
        load_base=lambda _path: copy.deepcopy(base),
        load_refit=lambda _path: copy.deepcopy(refit),
        restore_fit=lambda state: lookup[state["key"]],
        file_sha256=lambda path: _sha(
            "base-file" if path == Path("base.pt") else "refit-file"
        ),
    )

    assert len(catalog.replacements) == 18
    assert tuple(value.layer_ordinal for value in catalog.replacements) == tuple(
        range(18)
    )
    assert tuple(
        row["deployment_kind"] for row in catalog.layer_lineage
    ) == (
        *("frozen_source" for _ in range(10)),
        *("sequential_refit" for _ in range(8)),
    )
    assert catalog.generator_plan_sha256s[:10] == tuple(
        lookup[f"source:{ordinal}"].executable_plan.artifact_sha256
        for ordinal in range(10)
    )
    assert catalog.generator_plan_sha256s[10:] == tuple(
        lookup[f"refit:{ordinal}"].executable_plan.artifact_sha256
        for ordinal in range(10, 18)
    )
    assert catalog.deployed_fit_sha256s[:10] == catalog.source_fit_sha256s[:10]
    assert all(
        deployed != source
        for deployed, source in zip(
            catalog.deployed_fit_sha256s[10:],
            catalog.source_fit_sha256s[10:],
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("base_file", "does not bind"),
        ("model", "model bindings differ"),
        ("source_lineage", "source lineage differs"),
        ("refit_fit", "fit lineage differs"),
    ),
)
def test_rejects_cross_artifact_lineage_drift(
    mutation: str,
    message: str,
) -> None:
    base, refit, lookup = _fixture()
    if mutation == "base_file":
        refit["frozen_sources"]["full_stack"][  # type: ignore[index]
            "artifact_file_sha256"
        ] = _sha("wrong")
    elif mutation == "model":
        refit["model"]["resolved_commit"] = "2" * 40  # type: ignore[index]
    elif mutation == "source_lineage":
        refit["layer_refits"][0]["source_fit_sha256"] = _sha("wrong")  # type: ignore[index]
    elif mutation == "refit_fit":
        refit["layer_refits"][0]["refit_fit_sha256"] = _sha("wrong")  # type: ignore[index]
    else:
        raise AssertionError("unknown mutation")

    with pytest.raises(ValueError, match=message):
        restore_gemma3_full_mlp_stack_refit_runtime(
            "base.pt",
            "refit.pt",
            load_base=lambda _path: copy.deepcopy(base),
            load_refit=lambda _path: copy.deepcopy(refit),
            restore_fit=lambda state: lookup[state["key"]],
            file_sha256=lambda path: _sha(
                "base-file" if path == Path("base.pt") else "refit-file"
            ),
        )
