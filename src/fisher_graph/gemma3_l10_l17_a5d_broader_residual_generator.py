"""Run the one-fold A5d source-anchored residual-generator assessment.

The expensive capture, downstream target solve, authenticated row bridge, and
breadth construction remain owned by the A5c runner.  This module enters its
post-breadth/pre-held continuation seam, fits only the zero-mean residual,
freezes the complete dual-graph executable, and touches the outer-held family
only after that freeze has been reauthenticated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from .gemma3_l10_l17_a5c_broader_selected_generator import (
    A5cBroaderTrainingWorkspace,
    run_gemma3_l10_l17_a5c_broader_selected_generator,
)
from .gemma3_l10_l17_a5c_report import (
    DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT,
    load_gemma3_l10_l17_a5c_report,
)
from .gemma3_l10_l17_a5d_evaluation import (
    score_a5d_source_anchored_residual_fold,
)
from .gemma3_l10_l17_a5d_executable import (
    build_a5d_scoring_executors,
    freeze_a5d_executable,
    select_a5d_held_scoring_batch_after_freeze,
)
from .gemma3_l10_l17_a5d_family_residual_cv import (
    A5D_ALPHA_GRID,
    A5D_FIXED_GENERATOR_RANK,
    A5D_RIDGE_GRID,
    select_a5d_family_disjoint_residual,
)
from .gemma3_l10_l17_a5d_prepublication_bundle import (
    default_a5d_prepublication_bundle_path,
    finalize_a5d_prepublication_bundle,
    publish_a5d_report_with_prepublication_bundle,
)
from .gemma3_l10_l17_a5d_report import (
    DEFAULT_GEMMA3_L10_L17_A5D_REPORT_OUTPUT,
    a5d_outer_evaluation_sha256,
    compact_a5d_family_residual_cv_receipt,
    compact_a5d_source_anchored_residual_receipt,
)
from .gemma3_l10_l17_a5d_source_anchored_residual import (
    build_a5d_source_anchored_residual_targets,
)
from .gemma3_experiment import DEFAULT_MODEL_ID


__all__ = [
    "run_gemma3_l10_l17_a5d_broader_residual_generator",
]


_EXPECTED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_EXPECTED_A5C_FILE_SHA256 = (
    "10a0389f6c6a91893697fcb915edd1917ba4966a4e30b8f9870caed10f43a840"
)
_EXPECTED_A5C_REPORT_SHA256 = (
    "94238d934e9c5db4e6ddbb67eb9a4426c70cb1d12684c4ffc25777b62f437fe7"
)
_EXPECTED_A5C_FROZEN_COMPOSITION = {
    "nll_per_token": 7.300289344787598,
    "delta_nll_per_token": 0.14230638231549975,
    "native_to_candidate_kl_per_token": 0.10553961873443381,
    "top1_agreement_to_native": 0.7714285714285715,
}
_OUTPUT_BOUNDARY = "layer.17.mlp.delta"
_FINAL_HEAD_CHUNK_ROWS = 8
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _progress(message: str) -> None:
    print(f"[a5d-residual] {message}", flush=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _receipt_sha256(value: Mapping[str, object], *, label: str) -> str:
    digest = value.get("receipt_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} lacks a receipt SHA-256")
    return digest


def _a5c_frozen_metric(value: object) -> dict[str, object]:
    condition = _mapping(value, label="canonical A5c frozen composition")
    fields = set(_EXPECTED_A5C_FROZEN_COMPOSITION)
    if not fields.issubset(condition):
        raise ValueError("canonical A5c frozen metric is incomplete")
    return {name: condition[name] for name in fields}


def _authenticate_canonical_a5c_file(path: Path | str) -> Path:
    """Bind only the immutable A5c file bytes before fresh held scoring."""

    source = Path(path)
    if _file_sha256(source) != _EXPECTED_A5C_FILE_SHA256:
        raise ValueError("A5d requires the canonical A5c report file")
    return source


def _authenticate_canonical_a5c_after_outer_scoring(
    path: Path | str,
    *,
    outer_evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Parse A5c only after fresh scoring and prove the held reference matches."""

    source = Path(path)
    if _file_sha256(source) != _EXPECTED_A5C_FILE_SHA256:
        raise ValueError("canonical A5c report file changed after preflight")
    value = load_gemma3_l10_l17_a5c_report(source)
    evaluation = _mapping(
        value.get("outer_evaluation"), label="canonical A5c evaluation"
    )
    conditions = _mapping(
        evaluation.get("conditions"), label="canonical A5c conditions"
    )
    frozen = _a5c_frozen_metric(
        conditions.get("frozen_uncorrected_composition")
    )
    fresh_conditions = _mapping(
        outer_evaluation.get("conditions"), label="fresh A5d conditions"
    )
    fresh_frozen = _a5c_frozen_metric(
        fresh_conditions.get("frozen_uncorrected_composition")
    )
    if (
        value.get("report_sha256") != _EXPECTED_A5C_REPORT_SHA256
        or frozen != _EXPECTED_A5C_FROZEN_COMPOSITION
        or fresh_frozen != frozen
        or evaluation.get("outer_fold_index") != 0
        or outer_evaluation.get("outer_fold_index") != 0
        or evaluation.get("logical_valid_tokens")
        != outer_evaluation.get("logical_valid_tokens")
        or evaluation.get("supervised_tokens")
        != outer_evaluation.get("supervised_tokens")
    ):
        raise ValueError(
            "fresh A5d frozen control does not match canonical A5c held scoring"
        )
    return value


def _layer10_lowering_hashes(
    workspace: A5cBroaderTrainingWorkspace,
) -> dict[str, str]:
    return {
        name: workspace.layer10_lowerings_by_node[name].artifact_sha256
        for name in workspace.layer10_graph.traversal_order
    }


def _capture_summary(
    workspace: A5cBroaderTrainingWorkspace,
) -> dict[str, object]:
    observations = int(workspace.compiled_inputs.shape[0])
    return {
        "capture_sha256": workspace.capture_sha256,
        "capture_audit_sha256": workspace.capture_audit_sha256,
        # This is the capture receipt's row-key identity, which is the
        # canonical A5c report binding.  ``workspace.row_catalog_sha256`` is a
        # separate split-construction hash with a different domain.
        "source_row_catalog_sha256": (
            workspace.capture.native_rows.row_key_sha256
        ),
        "training_family_count": len(workspace.training_family_aliases),
        "training_example_count": len(workspace.training_example_ids),
        "captured_observation_count": observations,
        "outer_held_family_rows_present": False,
        "all_required_capture_audits_pass": True,
    }


def _configuration(
    workspace: A5cBroaderTrainingWorkspace,
) -> dict[str, object]:
    return {
        "outer_fold_index": 0,
        "training_family_count": len(workspace.training_family_aliases),
        "training_examples_per_family": (
            len(workspace.training_example_ids)
            // len(workspace.training_family_aliases)
        ),
        "row_selection_policy": "all_valid_captured_rows_per_example",
        "generator_rank": A5D_FIXED_GENERATOR_RANK,
        "ridge_grid": list(A5D_RIDGE_GRID),
        "alpha_grid": list(A5D_ALPHA_GRID),
        "held_examples_scored": 1,
        "output_boundary": _OUTPUT_BOUNDARY,
        "final_head_chunk_rows": _FINAL_HEAD_CHUNK_ROWS,
    }


def _comparison_to_a5c(
    canonical_a5c: Mapping[str, object],
    *,
    outer_evaluation: Mapping[str, object],
) -> dict[str, object]:
    evaluation = _mapping(
        canonical_a5c.get("outer_evaluation"), label="canonical A5c evaluation"
    )
    conditions = _mapping(
        evaluation.get("conditions"), label="canonical A5c conditions"
    )
    frozen = _a5c_frozen_metric(
        conditions.get("frozen_uncorrected_composition")
    )
    fresh_conditions = _mapping(
        outer_evaluation.get("conditions"), label="fresh A5d conditions"
    )
    fresh_frozen = _a5c_frozen_metric(
        fresh_conditions.get("frozen_uncorrected_composition")
    )
    if (
        frozen != _EXPECTED_A5C_FROZEN_COMPOSITION
        or fresh_frozen != frozen
    ):
        raise ValueError("canonical A5c frozen comparison metric drifted")
    return {
        "a5c_file_sha256": _EXPECTED_A5C_FILE_SHA256,
        "a5c_report_sha256": _EXPECTED_A5C_REPORT_SHA256,
        "same_outer_fold": True,
        "same_held_example_policy": True,
        "a5c_frozen_composition": dict(_EXPECTED_A5C_FROZEN_COMPOSITION),
    }


def _run_a5d_training_continuation(
    workspace: A5cBroaderTrainingWorkspace,
    *,
    canonical_a5c_path: Path,
    destination: Path,
) -> dict[str, object]:
    """Consume the post-breadth workspace without early held-family access."""

    _progress("target: source-anchored zero-mean residual over all breadth rows")
    residual_targets = build_a5d_source_anchored_residual_targets(
        frozen_compiled_block_states=(
            workspace.frozen_compiled_block_states
        ),
        compiled_correction_base_states=(
            workspace.compiled_correction_base_states
        ),
        oracle_rows=workspace.breadth.all_rows,
        bases_by_node=workspace.bases_by_node,
        node_order=workspace.layer17_graph.traversal_order,
        fragment_id_by_node=workspace.fragment_id_by_node,
    )
    residual_target_receipt = residual_targets.receipt()

    _progress("select: seven-family residual ridge/alpha CV")
    selection = select_a5d_family_disjoint_residual(
        bridge=workspace.bridge,
        targets=residual_targets,
        source_graph=workspace.layer17_graph,
        source_lowerings_by_node=workspace.layer17_lowerings_by_node,
        adapter=workspace.adapter,
        native_block_states=workspace.native_block_states,
        frozen_compiled_block_states=(
            workspace.frozen_compiled_block_states
        ),
        output_boundary=_OUTPUT_BOUNDARY,
        ridge_grid=A5D_RIDGE_GRID,
        alpha_grid=A5D_ALPHA_GRID,
        final_head_chunk_rows=_FINAL_HEAD_CHUNK_ROWS,
        final_head_token_locality_lineage_sha256=(
            workspace.target_token_locality_lineage_sha256
        ),
    )
    cv_receipt = selection.receipt()

    target_report_receipt = _mapping(
        workspace.target_report_receipt, label="A5d target-solve receipt"
    )
    bridge_receipt = _mapping(
        workspace.bridge_receipt, label="A5d bridge receipt"
    )
    breadth_receipt = _mapping(
        workspace.breadth_receipt, label="A5d breadth receipt"
    )
    lineage = {
        "a5c_report_sha256": _EXPECTED_A5C_REPORT_SHA256,
        "capture_sha256": workspace.capture_sha256,
        "target_solve_receipt_sha256": _receipt_sha256(
            target_report_receipt, label="A5d target solve"
        ),
        "coordinate_row_bank_receipt_sha256": _receipt_sha256(
            bridge_receipt, label="A5d coordinate row bank"
        ),
        "breadth_split_receipt_sha256": _receipt_sha256(
            breadth_receipt, label="A5d breadth split"
        ),
        "source_anchored_residual_receipt_sha256": _receipt_sha256(
            residual_target_receipt, label="A5d residual target"
        ),
        "residual_cv_receipt_sha256": _receipt_sha256(
            cv_receipt, label="A5d residual CV"
        ),
        "layer10_graph_sha256": workspace.layer10_graph.artifact_sha256,
        "layer10_lowering_sha256_by_node": _layer10_lowering_hashes(
            workspace
        ),
        "matched_double_deletion_graph_sha256": (
            workspace.source_composition_graph.artifact_sha256
        ),
    }

    _progress("freeze: source owner plus optional additive residual")
    executable = freeze_a5d_executable(
        selection=selection,
        source_layer17_graph=workspace.layer17_graph,
        source_layer17_lowerings_by_node=(
            workspace.layer17_lowerings_by_node
        ),
        source_composition_graph=workspace.source_composition_graph,
        source_composition_lowerings=(
            workspace.source_composition_lowerings
        ),
        lineage=lineage,
    )
    executable_section = executable.report_section()

    _progress("prepare: authenticate four distinct scoring executors")
    (
        layer10_executor,
        selected_layer17_executor,
        frozen_composition_executor,
        selected_composition_executor,
    ) = build_a5d_scoring_executors(
        workspace.adapter,
        workspace.layer10_graph,
        workspace.layer10_lowerings_by_node,
        executable,
    )

    _progress("held: select first outer-family example after freeze")
    held_batches = select_a5d_held_scoring_batch_after_freeze(
        blocks=workspace.blocks,
        held_family_alias=workspace.held_family_alias,
        executable=executable,
    )
    _progress("evaluate: full-model held logits with matched deletion")
    outer_evaluation = score_a5d_source_anchored_residual_fold(
        adapter=workspace.adapter,
        layer10_executor=layer10_executor,
        selected_layer17_executor=selected_layer17_executor,
        frozen_composition_executor=frozen_composition_executor,
        selected_composition_executor=selected_composition_executor,
        batches=held_batches,
    )

    # The canonical A5c report contains the outer-held result.  Its contents
    # remain unavailable throughout target construction, CV, freeze, held
    # selection, and fresh scoring.  Parse it only now, then require the fresh
    # frozen control to reproduce the same fold and example policy exactly.
    _progress("compare: authenticate canonical A5c after fresh outer scoring")
    canonical_a5c = _authenticate_canonical_a5c_after_outer_scoring(
        canonical_a5c_path,
        outer_evaluation=outer_evaluation,
    )

    residual_target_compact = (
        compact_a5d_source_anchored_residual_receipt(
            residual_target_receipt
        )
    )
    residual_cv_compact = compact_a5d_family_residual_cv_receipt(cv_receipt)
    chronology = {
        "residual_cv_completed_event": 1,
        "executable_frozen_event": 2,
        "outer_held_batch_selected_event": 3,
        "outer_held_model_evaluated_event": 4,
        "outer_held_batch_selected_or_scored_before_freeze": False,
        "executable_frozen_before_outer_held_batch_selection": True,
        "executable_frozen_before_outer_held_model_evaluation": True,
        "residual_cv_receipt_sha256": residual_cv_compact[
            "receipt_sha256"
        ],
        "selection_freeze_sha256": executable_section[
            "selection_freeze_sha256"
        ],
        "outer_evaluation_sha256": a5d_outer_evaluation_sha256(
            outer_evaluation
        ),
    }
    report_inputs = {
        "source_bindings": {
            "a5c_file_sha256": _EXPECTED_A5C_FILE_SHA256,
            "a5c_report_sha256": _EXPECTED_A5C_REPORT_SHA256,
        },
        "runtime": dict(workspace.runtime_metadata),
        "configuration": _configuration(workspace),
        "capture": _capture_summary(workspace),
        "source_anchored_residual": residual_target_compact,
        "residual_cv": residual_cv_compact,
        "evidence_receipts": {
            "source_anchored_residual": residual_target_receipt,
            "residual_cv": cv_receipt,
        },
        "selected_executable": executable_section,
        "chronology": chronology,
        "outer_evaluation": outer_evaluation,
        "comparison_to_a5c": _comparison_to_a5c(
            canonical_a5c,
            outer_evaluation=outer_evaluation,
        ),
    }
    saved = publish_a5d_report_with_prepublication_bundle(
        output=destination,
        report_inputs=report_inputs,
    )
    _progress(f"published: {destination}")
    del residual_targets, selection, executable
    gc.collect()
    return saved


def run_gemma3_l10_l17_a5d_broader_residual_generator(
    *,
    revision: str,
    output: Path | str = DEFAULT_GEMMA3_L10_L17_A5D_REPORT_OUTPUT,
    a5c_path: Path | str = DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run A5d or recover its tensor-free publication checkpoint."""

    if (
        not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or revision != _EXPECTED_MODEL_REVISION
        or model_id != DEFAULT_MODEL_ID
        or device_name != "cpu"
        or dtype != "float32"
    ):
        raise ValueError("A5d must replay the canonical pinned CPU float32 runtime")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5d report")
    prepublication = default_a5d_prepublication_bundle_path(destination)
    if prepublication.exists() or prepublication.is_symlink():
        _progress("recover: publish surviving tensor-free prepublication bundle")
        return finalize_a5d_prepublication_bundle(
            prepublication, output=destination
        )

    _progress("preflight: authenticate canonical A5c source report")
    canonical_a5c_path = _authenticate_canonical_a5c_file(a5c_path)

    def continuation(workspace: A5cBroaderTrainingWorkspace) -> dict[str, object]:
        return _run_a5d_training_continuation(
            workspace,
            canonical_a5c_path=canonical_a5c_path,
            destination=destination,
        )

    return run_gemma3_l10_l17_a5c_broader_selected_generator(
        revision=revision,
        output=destination,
        model_id=model_id,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
        _continuation=continuation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_GEMMA3_L10_L17_A5D_REPORT_OUTPUT
    )
    parser.add_argument(
        "--a5c-path",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A5C_REPORT_OUTPUT,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_l10_l17_a5d_broader_residual_generator(
        revision=args.revision,
        output=args.output,
        a5c_path=args.a5c_path,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
