"""Collect exact token-loss Fisher evidence on reusable Gemma A-fit only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_l3_l4_basis_package import DEFAULT_BASIS_PACKAGE
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
)
from .gemma3_l3_l4_iterative_residual_campaign import (
    _gather_logits,
    _observation,
    _panel_manifest,
    _source_authority,
    _validate_execution,
    _validate_parent,
)
from .gemma3_l3_l4_iterative_residual_diagnostic import (
    _GemmaDevelopmentCollectionRecipe,
    build_parser as build_residual_parser,
    run_gemma_iterative_residual_diagnostic,
)
from .gemma3_l3_l4_iterative_token_fisher_development import (
    build_gemma_iterative_token_fisher_development_report,
    publish_gemma_iterative_token_fisher_development_report,
    validate_gemma_iterative_token_fisher_development_report,
)
from .gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
    build_gemma_iterative_token_occupancy_tangent_record,
)
from .token_loss_fisher import (
    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
    build_token_loss_fisher_prompt_record,
)


__all__ = [
    "CLI_NAME",
    "DEFAULT_OUTPUT",
    "TOKEN_FISHER_PARENT_LOSS_AUTHORITY_TOLERANCE",
    "TOKEN_FISHER_VJP_CHUNK_SIZE",
    "build_parser",
    "main",
    "run_gemma_iterative_token_fisher_development_diagnostic",
]


CLI_NAME = "fisher-graph-gemma-l3-l4-iterative-token-fisher-dev"
TOKEN_FISHER_VJP_CHUNK_SIZE = 8
TOKEN_FISHER_PARENT_LOSS_AUTHORITY_TOLERANCE = 5.0e-4
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_DEFAULT_EXPANDED_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
_DEFAULT_EXPANDED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
_DEFAULT_MATERIALIZATION_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-materialization-v1.report.json"
)
_DEFAULT_FACTORIAL_REPORT = (
    _LOCAL_ROOT / "progressive-a-x4-h4-factorial-fit-v1.report.json"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-token-loss-fisher-dev-v1.report.json"
)


def _collect_token_fisher(
    *,
    panel: object,
    adapter: object,
    bridge: object,
    parent_artifact: object,
    parent_h4: object,
    x4_head: object,
    lineage: Mapping[str, object],
) -> Mapping[str, object]:
    """Run 16 source forwards and 16 retained token-VJP forwards."""

    manifest = _panel_manifest(panel)  # type: ignore[arg-type]
    validated_x4, validated_h4 = _validate_parent(
        panel=panel,  # type: ignore[arg-type]
        adapter=adapter,  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
        parent=parent_artifact,  # type: ignore[arg-type]
    )
    if (
        getattr(validated_x4, "artifact_sha256", None)
        != getattr(x4_head, "artifact_sha256", None)
        or getattr(validated_h4, "artifact_sha256", None)
        != getattr(parent_h4, "artifact_sha256", None)
    ):
        raise ValueError("token Fisher live parent heads differ")
    if (
        TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER
        != COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES
    ):
        raise RuntimeError("token Fisher coordinate declarations drifted")

    tangent_records: list[object] = []
    prompt_records: list[object] = []
    vjp_receipts: dict[str, str] = {}
    backward_calls = 0
    examples = getattr(panel, "examples", None)
    if not isinstance(examples, tuple):
        raise TypeError("token Fisher panel examples must be a tuple")
    for example in examples:
        example.validate_integrity()
        (
            source_execution,
            source_logits,
            supervised_indices,
            targets,
            logical_positions,
        ) = _source_authority(
            adapter=adapter,  # type: ignore[arg-type]
            example=example,
        )
        token_vjp = bridge.execute_h4_token_nll_vjps(
            adapter,
            example.batch.model_inputs,
            targets=example.batch.targets,
            vjp_chunk_size=TOKEN_FISHER_VJP_CHUNK_SIZE,
            x4_head=x4_head,
            h4_head=parent_h4,
        )
        token_vjp.validate_integrity()
        execution = token_vjp.execution
        _validate_execution(
            execution,
            example_model_inputs_sha256=example.model_inputs_sha256,
            bridge_binding_sha256=bridge.bridge_binding_sha256,
            x4_head=x4_head,
            h4_head=parent_h4,
            label="exact token-loss parent VJP",
        )
        expected_token_grid = torch.stack(
            (
                torch.zeros_like(supervised_indices),
                supervised_indices,
            ),
            dim=1,
        ).to(
            device=token_vjp.supervised_indices.device,
            dtype=torch.int64,
        )
        if not torch.equal(
            token_vjp.supervised_indices,
            expected_token_grid,
        ):
            raise ValueError(
                "token Fisher supervised-token order differs from authority"
            )
        parent_logits = _gather_logits(
            getattr(execution, "logits", None),
            supervised_indices,
        )
        parent_observation = _observation(
            example=example,
            source_logits=source_logits,
            candidate_logits=parent_logits,
            targets=targets,
        )
        target_device = source_logits.device
        source_token_nll = F.cross_entropy(
            source_logits,
            targets.to(target_device),
            reduction="none",
        ).to(torch.float64)
        parent_token_nll = F.cross_entropy(
            parent_logits,
            targets.to(parent_logits.device),
            reduction="none",
        ).to(torch.float64)
        if not torch.allclose(
            token_vjp.token_losses.detach()
            .to(device="cpu", dtype=torch.float64),
            parent_token_nll.to(device="cpu"),
            rtol=0.0,
            atol=TOKEN_FISHER_PARENT_LOSS_AUTHORITY_TOLERANCE,
        ):
            maximum_difference = float(
                (
                    token_vjp.token_losses.detach()
                    .to(device="cpu", dtype=torch.float64)
                    - parent_token_nll.to(device="cpu")
                )
                .abs()
                .max()
            )
            raise RuntimeError(
                "token VJP losses differ from parent finite-NLL authority; "
                f"maximum absolute difference={maximum_difference:.9g}"
            )
        supervised_logical_positions = (
            logical_positions[0]
            .index_select(
                0,
                supervised_indices.to(logical_positions.device),
            )
            .detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
        )
        tangent = build_gemma_iterative_token_occupancy_tangent_record(
            example=example,
            parent_execution=execution,
            token_loss_gradients=token_vjp.h4_gradients,
            supervised_token_logical_positions=(
                supervised_logical_positions
            ),
            parent_h4=parent_h4,  # type: ignore[arg-type]
            parent_observation=parent_observation,
        )
        token_scores = torch.tensor(
            tuple(
                row.tangent_by_combined_occupancy_coordinate
                for row in tangent.rows
            ),
            dtype=torch.float64,
        )
        compensation_target = (
            source_token_nll.to(device="cpu")
            - parent_token_nll.to(device="cpu")
        ).contiguous()
        prompt = build_token_loss_fisher_prompt_record(
            example_id=example.example_id,
            family_id=example.family_id,
            coordinate_names=(
                COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES
            ),
            token_scores=token_scores,
            compensation_target=compensation_target,
        )
        if (
            tangent.example_id != prompt.example_id
            or tangent.family_id != prompt.family_id
            or tangent.supervised_token_count
            != prompt.supervised_tokens
            or manifest[example.example_id] != example.family_id
        ):
            raise RuntimeError("token Fisher prompt binding differs")
        tangent_records.append(tangent)
        prompt_records.append(prompt)
        vjp_receipts[example.example_id] = token_vjp.artifact_sha256
        backward_calls += token_vjp.backward_call_count
        del (
            source_execution,
            source_logits,
            supervised_indices,
            targets,
            logical_positions,
            token_vjp,
            execution,
            expected_token_grid,
            parent_logits,
            parent_observation,
            source_token_nll,
            parent_token_nll,
            supervised_logical_positions,
            tangent,
            token_scores,
            compensation_target,
            prompt,
        )

    return build_gemma_iterative_token_fisher_development_report(
        token_tangent_records=tangent_records,
        prompt_records=prompt_records,
        lineage=lineage,
        token_vjp_artifact_sha256_by_example=vjp_receipts,
        total_backward_call_count=backward_calls,
        vjp_chunk_size=TOKEN_FISHER_VJP_CHUNK_SIZE,
    )


def run_gemma_iterative_token_fisher_development_diagnostic(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    materialization_report_path: Path | str = (
        _DEFAULT_MATERIALIZATION_REPORT
    ),
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    factorial_report_path: Path | str = _DEFAULT_FACTORIAL_REPORT,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Collect exact token Fisher evidence and stop before fresh validation."""

    recipe = _GemmaDevelopmentCollectionRecipe(
        collect=_collect_token_fisher,
        validate_report=(
            validate_gemma_iterative_token_fisher_development_report
        ),
        publish_report=(
            publish_gemma_iterative_token_fisher_development_report
        ),
        report_label="exact token-loss Fisher development",
        source_code_files=(
            "gemma3_l3_l4_iterative_token_fisher_diagnostic.py",
            "gemma3_l3_l4_iterative_token_fisher_development.py",
            "gemma3_l3_l4_iterative_token_fisher_edges.py",
            "gemma3_l3_l4_iterative_occupancy_route.py",
            "gemma3_l3_l4_iterative_state_router.py",
            "token_loss_fisher.py",
        ),
    )
    report = run_gemma_iterative_residual_diagnostic(
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=(
            expected_materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            expected_materialization_report_file_sha256
        ),
        factorial_report_path=factorial_report_path,
        expected_factorial_report_sha256=(
            expected_factorial_report_sha256
        ),
        expected_factorial_report_file_sha256=(
            expected_factorial_report_file_sha256
        ),
        graph_candidate_path=graph_candidate_path,
        basis_package_path=basis_package_path,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        output=output,
        cache_dir=cache_dir,
        _diagnostic_recipe=recipe,
    )
    validate_gemma_iterative_token_fisher_development_report(report)
    with Path(output).open("r", encoding="utf-8") as handle:
        replay = json.load(handle)
    validate_gemma_iterative_token_fisher_development_report(replay)
    if _canonical_report(replay) != _canonical_report(report):
        raise RuntimeError("published token Fisher report differs")
    return report


def _canonical_report(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_residual_parser()
    parser.description = (
        "Collect exact supervised-token Fisher edge evidence on reusable "
        "A-fit and stop before every selection/fresh-panel boundary."
    )
    parser.set_defaults(output=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma_iterative_token_fisher_development_diagnostic(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        materialization_report_path=args.materialization_report,
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        factorial_report_path=args.factorial_report,
        expected_factorial_report_sha256=(
            args.factorial_report_sha256
        ),
        expected_factorial_report_file_sha256=(
            args.factorial_report_file_sha256
        ),
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
