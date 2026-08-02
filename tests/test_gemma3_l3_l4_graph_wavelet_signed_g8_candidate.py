from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_candidate as frozen
from fisher_graph.gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    INTERIOR_ORIGINS,
    load_gemma3_spectral_source,
)
from fisher_graph.gemma3_l3_l4_graph_wavelet_experiment import (
    load_gemma3_graph_wavelet_candidate,
)
from fisher_graph.gemma3_l3_l4_graph_wavelet_grouped_comparison_experiment import (
    load_gemma3_graph_wavelet_grouped_comparison_candidate,
)
from fisher_graph.gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
)


@pytest.fixture(scope="module")
def candidate() -> frozen.Gemma3L3L4GraphWaveletSignedG8Candidate:
    required = (
        DEFAULT_INTERIOR_ARTIFACT,
        DEFAULT_INTERIOR_ARTIFACT.with_suffix(".json"),
        DEFAULT_PARENT_ARTIFACT,
        DEFAULT_PARENT_ARTIFACT.with_suffix(".json"),
        frozen.DEFAULT_COMPARISON_ARTIFACT,
        frozen.DEFAULT_COMPARISON_ARTIFACT.with_suffix(".json"),
    )
    if any(not path.exists() for path in required):
        pytest.skip("pinned local signed-g8 source artifacts are unavailable")
    source = load_gemma3_spectral_source(
        DEFAULT_INTERIOR_ARTIFACT,
        expected_file_sha256=DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        DEFAULT_PARENT_ARTIFACT,
        expected_artifact_sha256=DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_PARENT_REPORT_SHA256,
    )
    comparison = load_gemma3_graph_wavelet_grouped_comparison_candidate(
        frozen.DEFAULT_COMPARISON_ARTIFACT,
        expected_artifact_sha256=(
            frozen.EXPECTED_COMPARISON_ARTIFACT_SHA256
        ),
        expected_tensor_file_sha256=(
            frozen.EXPECTED_COMPARISON_TENSOR_FILE_SHA256
        ),
        expected_report_sha256=frozen.EXPECTED_COMPARISON_REPORT_SHA256,
    )
    return frozen.compile_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        source,
        parent,
        comparison,
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_reconstruction_matches_every_frozen_execution_identity(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
) -> None:
    candidate.validate_frozen_identity()

    assert candidate.artifact_sha256 == frozen.DEFAULT_FROZEN_ARTIFACT_SHA256
    assert candidate.method == "signed_local_svd_g8"
    assert candidate.source_basis_kind == (
        "fit_only_graph_wavelet_local_block_svd"
    )
    assert candidate.plan.artifact_sha256 == frozen.EXPECTED_PLAN_ARTIFACT_SHA256
    assert candidate.plan.metadata()["tensor_sha256s"] == (
        frozen.EXPECTED_PLAN_TENSOR_SHA256S
    )
    assert candidate.plan.source_rank == 45
    assert candidate.plan.target_rank == 64
    assert candidate.plan.accounting().prepared_storage_bytes == 2_268_184
    assert candidate.binding["source_model_sha256"] == (
        frozen.EXPECTED_SOURCE_MODEL_SHA256
    )
    assert candidate.construction_receipt["partition"][  # type: ignore[index]
        "artifact_sha256"
    ] == frozen.EXPECTED_PARTITION_ARTIFACT_SHA256


def test_state_roundtrip_serializes_only_the_plan_tensors(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
) -> None:
    state = candidate.state_dict()
    restored = frozen.Gemma3L3L4GraphWaveletSignedG8Candidate.from_state_dict(
        state
    )

    assert restored.metadata() == candidate.metadata()
    restored.validate_frozen_identity()
    for key, value in state.items():
        if key == "plan":
            assert _contains_tensor(value)
        else:
            assert not _contains_tensor(value)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("method", "signed_local_svd_g4", "plan ABI"),
        ("source_basis_kind", "fixed_orthonormal_control", "plan ABI"),
    ),
)
def test_method_or_basis_kind_substitution_fails_closed(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
    field: str,
    replacement: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(
            candidate,
            **{field: replacement, "artifact_sha256": ""},
        )


def test_plan_tensor_tamper_fails_before_outer_candidate_load(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
) -> None:
    state = deepcopy(candidate.state_dict())
    state["plan"]["knot_cores"][0, 0, 0, 0] += 1.0  # type: ignore[index]

    with pytest.raises(ValueError, match="hash or shape mismatch"):
        frozen.Gemma3L3L4GraphWaveletSignedG8Candidate.from_state_dict(state)


def test_self_consistent_plan_substitution_fails_frozen_identity(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
) -> None:
    alternate_plan = replace(
        candidate.plan,
        knot_cores=-candidate.plan.knot_cores,
        artifact_sha256="",
    )
    selection = deepcopy(dict(candidate.selection_receipt))
    selection["plan_artifact_sha256"] = alternate_plan.artifact_sha256
    selection["fit_evaluation"][  # type: ignore[index]
        "plan_sha256"
    ] = alternate_plan.artifact_sha256
    selection["heldout_evaluation"][  # type: ignore[index]
        "plan_sha256"
    ] = alternate_plan.artifact_sha256
    substituted = replace(
        candidate,
        plan=alternate_plan,
        selection_receipt=selection,
        artifact_sha256="",
    )
    substituted.validate_integrity()

    with pytest.raises(ValueError, match="frozen identity"):
        substituted.validate_frozen_identity()


def test_publish_and_strict_load_roundtrip_under_local_runs(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path(".local-runs").mkdir()
    output = Path(".local-runs/frozen.pt")

    report = frozen._publish_candidate(candidate, output=output)
    restored = frozen.load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        output,
        expected_artifact_sha256=candidate.artifact_sha256,
        expected_tensor_file_sha256=report["artifact"][  # type: ignore[index]
            "tensor_file_sha256"
        ],
        expected_report_sha256=report["report_sha256"],  # type: ignore[arg-type]
    )

    assert restored.metadata() == candidate.metadata()
    with pytest.raises(ValueError, match="tensor file hash differs"):
        frozen.load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
            output,
            expected_artifact_sha256=candidate.artifact_sha256,
            expected_tensor_file_sha256="0" * 64,
            expected_report_sha256=report[
                "report_sha256"
            ],  # type: ignore[arg-type]
        )


def test_publisher_rejects_output_outside_local_runs(
    candidate: frozen.Gemma3L3L4GraphWaveletSignedG8Candidate,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="under .local-runs"):
        frozen._publish_candidate(candidate, output=Path("frozen.pt"))
