from __future__ import annotations

from collections.abc import Iterator, Mapping
import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from fisher_graph.fisher_local_chart_carrier import (
    DEFAULT_CARRIER_SIDECAR,
    DEFAULT_FISHER_CARRIER_PATH,
    DEFAULT_FISHER_SOURCE,
    DEFAULT_WEIGHTED_JACOBIAN_PATH,
    FisherLocalChartCarrierArtifact,
    _source_factors,
    extract_fisher_local_chart_carrier,
    load_fisher_local_chart_carrier,
)


_COMMIT = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_CONFIG = "588ffd9caad89c7bcb29c981e6836a16812c7b5533ffc4d8e0448e1fab3e1924"
_SCIENCE = "1" * 64


def _source_payload() -> dict[str, Any]:
    q = torch.eye(4, dtype=torch.float64)
    eigenvalues = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    return {
        "schema": "fisher_graph.gemma3_weighted_jacobian",
        "format_version": 2,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "model": {
            "model_id": "google/gemma-3-270m",
            "resolved_commit": _COMMIT,
            "config_sha256": _CONFIG,
            "hidden_size": 4,
            "weights_in_artifact": False,
        },
        "protocol": {
            "residual_width": 4,
            "fisher_scope": "width_pooled",
            "normalizer": "valid_activation_positions",
            "score": "summed_hard_target_next_token_nll",
            "fit_policy": "calibration_a_only",
            "canonical_boundaries": ("layer.4.input", "layer.4.output"),
            "gated_output_sites": ("layer.4.output",),
            "extraction_rank": 4,
        },
        "calibration_a": {
            "fisher": {
                "format_version": 1,
                "sequences": 3,
                "bases": {
                    "layer.4.output": {
                        "activation_name": "layer.4.output",
                        # This mean must not be copied into the sidecar.
                        "mean": torch.full((4,), 991.0, dtype=torch.float64),
                        "sequences": 3,
                        "fisher": {
                            "activation_name": "layer.4.output",
                            "width": 4,
                            "modes": 4,
                            "observations": 6,
                            "nonzero_observations": 4,
                            "rows_seen": 6,
                            "requested_rank": 4,
                            "sketch_rows": 5,
                            "squared_gradient_norm_sum": 60.0,
                            "fisher_trace": 10.0,
                            "retained_trace": 10.0,
                            "retained_trace_fraction": 1.0,
                            "retained_trace_semantics": "unused-in-test",
                            "accumulation_dtype": "float64",
                            "scope": "width_pooled",
                            "score_reduction": "sum",
                            "normalizer": "valid_activation_positions",
                            "algorithm": "frequent_directions",
                            "algorithm_version": 1,
                            "eigenvalues": eigenvalues,
                            "vectors": q,
                        },
                    }
                },
            },
            "codecs": {"forbidden-codec-sentinel": torch.tensor([12345.0])},
        },
        "calibration_b": {
            "selection": "FORBIDDEN-CALIBRATION-B-SELECTION-SENTINEL",
            "codec": torch.tensor([54321.0]),
        },
        "scientific_payload_sha256": _SCIENCE,
    }


def _extract(source: Path, sidecar: Path) -> FisherLocalChartCarrierArtifact:
    return extract_fisher_local_chart_carrier(
        source,
        sidecar,
        rank=2,
        expected_hidden_width=4,
    )


def _load(sidecar: Path) -> FisherLocalChartCarrierArtifact:
    return load_fisher_local_chart_carrier(
        sidecar,
        expected_rank=2,
        expected_hidden_width=4,
    )


def test_runner_facing_default_path_aliases_are_stable() -> None:
    assert DEFAULT_FISHER_SOURCE == DEFAULT_WEIGHTED_JACOBIAN_PATH
    assert DEFAULT_CARRIER_SIDECAR == DEFAULT_FISHER_CARRIER_PATH


def test_extracts_only_authorized_a_output_fisher_and_roundtrips(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    sidecar = tmp_path / "carrier.pt"
    torch.save(_source_payload(), source)

    artifact = _extract(source, sidecar)
    loaded = _load(sidecar)

    assert isinstance(artifact, FisherLocalChartCarrierArtifact)
    assert artifact.carrier.rank == 2
    assert artifact.carrier.hidden_width == 4
    assert torch.equal(artifact.q, torch.eye(4, dtype=torch.float64)[:, :2])
    assert torch.equal(artifact.eigenvalues, torch.tensor([4.0, 3.0]))
    assert artifact.retained_rank_trace == 7.0
    assert artifact.retained_trace_fraction == 0.7
    assert loaded.carrier_sha256 == artifact.carrier_sha256
    assert loaded.source_file_sha256 == artifact.source_file_sha256
    assert loaded.metadata()["semantic_qualification"] == {
        "metric": "shared_output_empirical_fisher_vjp",
        "metric_site": "layer.4.output",
        "shared_post_branch_metric": True,
        "exact_local_fisher_at_attention_site": False,
        "exact_local_fisher_at_mlp_site": False,
        "claim": (
            "shared output-Fisher sensitivity metric only; it is not an exact "
            "local Fisher at either earlier branch site"
        ),
    }


def test_sidecar_is_minimal_safe_and_weights_only_loadable(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    sidecar = tmp_path / "carrier.pt"
    torch.save(_source_payload(), source)
    _extract(source, sidecar)

    raw = torch.load(sidecar, map_location="cpu", weights_only=True)
    serialized = repr(raw)
    assert "FORBIDDEN-CALIBRATION-B-SELECTION-SENTINEL" not in serialized
    assert "forbidden-codec-sentinel" not in serialized
    assert "991" not in serialized
    assert set(raw["tensors"]) == {"q", "eigenvalues"}
    assert raw["safety"] == {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_tokenizer_state": False,
        "contains_model_weights": False,
        "contains_calibration_b": False,
    }


def test_publication_is_exclusive_and_does_not_replace_existing_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    sidecar = tmp_path / "carrier.pt"
    torch.save(_source_payload(), source)
    _extract(source, sidecar)
    original = sidecar.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _extract(source, sidecar)
    assert sidecar.read_bytes() == original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema="wrong"), "schema mismatch"),
        (
            lambda value: value.update(contains_prompt_text=True),
            "unsafe weighted-Jacobian source flag",
        ),
        (
            lambda value: value["model"].update(resolved_commit="wrong"),
            "resolved model commit mismatch",
        ),
        (
            lambda value: value["calibration_a"]["fisher"]["bases"][
                "layer.4.output"
            ]["fisher"].update(scope="per_token"),
            "unexpected scope",
        ),
        (
            lambda value: value["calibration_a"]["fisher"]["bases"][
                "layer.4.output"
            ]["fisher"]["eigenvalues"].__setitem__(1, 0.0),
            "include a zero mode",
        ),
        (
            lambda value: value["calibration_a"]["fisher"]["bases"][
                "layer.4.output"
            ]["fisher"]["vectors"].__setitem__((0, 1), 1.0),
            "orthonormal",
        ),
    ],
)
def test_source_contract_rejects_schema_metadata_and_factor_drift(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    source = tmp_path / "source.pt"
    sidecar = tmp_path / "carrier.pt"
    payload = _source_payload()
    mutation(payload)
    torch.save(payload, source)
    with pytest.raises(ValueError, match=message):
        _extract(source, sidecar)
    assert not sidecar.exists()


@pytest.mark.parametrize("tamper", ["tensor", "metadata", "extra-key"])
def test_strict_loader_rejects_tampering(tmp_path: Path, tamper: str) -> None:
    source = tmp_path / "source.pt"
    sidecar = tmp_path / "carrier.pt"
    changed = tmp_path / "changed.pt"
    torch.save(_source_payload(), source)
    _extract(source, sidecar)
    raw = torch.load(sidecar, map_location="cpu", weights_only=True)
    if tamper == "tensor":
        raw["tensors"]["q"][0, 0] += 0.01
    elif tamper == "metadata":
        raw["estimator"]["observations"] += 1
    else:
        raw["calibration_b"] = {"selection": "leak"}
    torch.save(raw, changed)

    with pytest.raises(ValueError):
        _load(changed)


def test_source_reader_never_touches_calibration_b() -> None:
    class ExplodingMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise AssertionError("Calibration B was consulted")

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("Calibration B was consulted")

        def __len__(self) -> int:
            raise AssertionError("Calibration B was consulted")

    payload = copy.deepcopy(_source_payload())
    payload["calibration_b"] = ExplodingMapping()
    carrier, source, estimator = _source_factors(
        payload,
        rank=2,
        expected_hidden_width=4,
        expected_model_id="google/gemma-3-270m",
        expected_resolved_commit=_COMMIT,
        expected_config_sha256=_CONFIG,
    )
    assert carrier.rank == 2
    assert source["source_role"] == "calibration_a"
    assert estimator["retained_rank"] == 2


def test_loader_can_bind_expected_source_file_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    sidecar = tmp_path / "carrier.pt"
    torch.save(_source_payload(), source)
    artifact = _extract(source, sidecar)

    bound = load_fisher_local_chart_carrier(
        sidecar,
        expected_rank=2,
        expected_hidden_width=4,
        expected_source_file_sha256=artifact.source_file_sha256,
    )
    assert bound.carrier_sha256 == artifact.carrier_sha256
    with pytest.raises(ValueError, match="source file SHA-256 mismatch"):
        load_fisher_local_chart_carrier(
            sidecar,
            expected_rank=2,
            expected_hidden_width=4,
            expected_source_file_sha256="0" * 64,
        )
