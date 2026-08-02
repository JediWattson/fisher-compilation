"""Prepare and collect the fixed Fisher-generator innovation development rung.

Preparation authenticates the already-frozen generator plan before publishing
the private 16-by-8 role input and its prompt-free receipt.  Collection keeps
the original expanded A-fit panel solely as the authority for the retained
parent heads, then evaluates the new family-disjoint panel with exactly one
source forward and one retained-parent token-VJP forward per example.

Only prompt sufficient statistics and non-reconstructive feature, tangent,
and VJP receipts cross the collection boundary.  Prompt text, token ids,
logits, activation rows, gradient rows, and token-score rows remain transient.
No finite-displacement candidate or runtime provider is constructed here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile

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
from .gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
)
from .gemma3_l3_l4_iterative_generator_innovation_development import (
    build_gemma_iterative_generator_innovation_development_report,
    replay_gemma_iterative_generator_innovation_development_report,
    validate_gemma_iterative_generator_innovation_development_report,
)
from .gemma3_l3_l4_iterative_generator_innovation_edges import (
    GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
    build_gemma_generator_innovation_token_scores,
)
from .gemma3_l3_l4_iterative_generator_innovation_panel import (
    DEFAULT_EXPANDED_FIT_CORPUS,
    DEFAULT_GENERATOR_INNOVATION_PLAN,
    DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT,
    DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    DEFAULT_PRIOR_OCCUPANCY_PANEL,
    FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256,
    FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    Gemma3L3L4GeneratorInnovationPanelReceipt,
    load_gemma3_l3_l4_generator_innovation_panel_receipt,
    materialize_gemma3_l3_l4_generator_innovation_panel,
    prepare_gemma3_l3_l4_generator_innovation_panel,
)
from .gemma3_l3_l4_iterative_generator_innovation_plan import (
    validate_gemma_iterative_generator_innovation_plan,
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
from .token_loss_fisher import build_token_loss_fisher_prompt_record


__all__ = [
    "CLI_NAME",
    "DEFAULT_OUTPUT",
    "GENERATOR_INNOVATION_PARENT_LOSS_AUTHORITY_TOLERANCE",
    "GENERATOR_INNOVATION_VJP_CHUNK_SIZE",
    "PREPARATION_CLI_NAME",
    "build_parser",
    "main",
    "preparation_main",
    "preparation_parser",
    "run_gemma_iterative_generator_innovation_development_diagnostic",
    "run_gemma_iterative_generator_innovation_panel_preparation",
]


CLI_NAME = "fisher-graph-gemma-l3-l4-generator-innovation-dev"
PREPARATION_CLI_NAME = (
    "fisher-graph-gemma-l3-l4-generator-innovation-panel-prepare"
)
GENERATOR_INNOVATION_VJP_CHUNK_SIZE = 8
GENERATOR_INNOVATION_PARENT_LOSS_AUTHORITY_TOLERANCE = 5.0e-4
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
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
    / "progressive-a-iterative-generator-innovation-dev-v1.report.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _publish_generator_innovation_report_once(
    destination: Path,
    report: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    plan_file_sha256: str,
) -> None:
    """Replay and install one report without a check-then-replace race."""

    validate_gemma_iterative_generator_innovation_development_report(report)
    replay = replay_gemma_iterative_generator_innovation_development_report(
        report=report,
        plan=plan,
        plan_file_sha256=plan_file_sha256,
    )
    if _canonical(replay) != _canonical(report):
        raise RuntimeError("generator innovation report replay differs")
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite generator innovation development report"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_raw = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(temporary_raw, Mapping):
            raise TypeError(
                "temporary generator innovation report must be an object"
            )
        validate_gemma_iterative_generator_innovation_development_report(
            temporary_raw
        )
        if _canonical(temporary_raw) != _canonical(report):
            raise RuntimeError(
                "temporary generator innovation development report differs"
            )
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite generator innovation development "
                "report"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_plan(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
) -> dict[str, object]:
    encoded = path.read_bytes()
    raw = json.loads(encoded)
    if not isinstance(raw, Mapping):
        raise TypeError("generator innovation plan must contain one object")
    plan = dict(raw)
    validate_gemma_iterative_generator_innovation_plan(plan)
    if (
        plan.get("plan_sha256") != expected_plan_sha256
        or hashlib.sha256(encoded).hexdigest()
        != expected_plan_file_sha256
    ):
        raise ValueError("generator innovation plan identity differs")
    return plan


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _materialize_collection_panel(
    *,
    receipt: Gemma3L3L4GeneratorInnovationPanelReceipt,
    role_input_path: Path,
    tokenizer: object,
    tokenizer_contract: Mapping[str, object],
    parent_corpus: object,
    parent_fit_panel: object,
) -> object:
    """Materialize only the receipt-bound new panel for collection."""

    if (
        getattr(parent_corpus, "artifact_sha256", None)
        != receipt.expanded_fit_corpus_artifact_sha256
        or getattr(parent_fit_panel, "manifest_sha256", None)
        == receipt.manifest_sha256
    ):
        raise ValueError(
            "generator collection parent corpus/panel lineage differs"
        )
    max_length = tokenizer_contract.get("max_length")
    device = tokenizer_contract.get("device")
    if type(max_length) is not int or max_length <= 0 or not isinstance(
        device, str
    ):
        raise ValueError("generator collection tokenizer contract differs")
    panel = materialize_gemma3_l3_l4_generator_innovation_panel(
        tokenizer=tokenizer,
        receipt=receipt,
        role_input_path=role_input_path,
        max_length=max_length,
        device=torch.device(device),
    )
    if (
        getattr(panel, "manifest_sha256", None) != receipt.manifest_sha256
        or getattr(panel, "membership_receipt_sha256", None)
        != receipt.membership_receipt_sha256
    ):
        raise RuntimeError(
            "generator collection materialization receipt differs"
        )
    return panel


def _collect_generator_innovation(
    *,
    plan: Mapping[str, object],
    plan_file_sha256: str,
    collection_lineage: Mapping[str, object],
    panel: object,
    parent_fit_panel: object,
    adapter: object,
    bridge: object,
    parent_artifact: object,
    parent_h4: object,
    x4_head: object,
    lineage: Mapping[str, object],
) -> Mapping[str, object]:
    """Collect exact Q6/R4 prompt moments on the new 16-by-8 panel."""

    manifest = _panel_manifest(panel)  # type: ignore[arg-type]
    validated_x4, validated_h4 = _validate_parent(
        panel=parent_fit_panel,  # type: ignore[arg-type]
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
        raise ValueError("generator innovation live parent heads differ")

    frozen_basis = _mapping(
        plan.get("frozen_generator_basis"),
        label="frozen generator basis",
    )
    basis = frozen_basis.get(
        "basis_matrix_source_coordinates_by_generator"
    )
    if basis is None:
        raise ValueError("generator innovation plan omitted its fixed basis")

    legacy_records: list[object] = []
    generator_records: list[object] = []
    feature_receipts: dict[str, object] = {}
    top_mode_receipts: dict[str, object] = {}
    vjp_receipts: dict[str, str] = {}
    tangent_receipts: dict[str, str] = {}
    backward_calls = 0
    examples = getattr(panel, "examples", None)
    if not isinstance(examples, tuple) or len(examples) != 16:
        raise TypeError(
            "generator innovation collection examples must be a 16-tuple"
        )
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
            vjp_chunk_size=GENERATOR_INNOVATION_VJP_CHUNK_SIZE,
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
            label="generator innovation retained-parent token VJP",
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
                "generator innovation supervised-token order differs"
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
        source_token_nll = F.cross_entropy(
            source_logits,
            targets.to(source_logits.device),
            reduction="none",
        ).to(torch.float64)
        parent_token_nll = F.cross_entropy(
            parent_logits,
            targets.to(parent_logits.device),
            reduction="none",
        ).to(torch.float64)
        retained_vjp_losses = (
            token_vjp.token_losses.detach()
            .to(device="cpu", dtype=torch.float64)
        )
        if not torch.allclose(
            retained_vjp_losses,
            parent_token_nll.to(device="cpu"),
            rtol=0.0,
            atol=GENERATOR_INNOVATION_PARENT_LOSS_AUTHORITY_TOLERANCE,
        ):
            maximum_difference = float(
                (
                    retained_vjp_losses
                    - parent_token_nll.to(device="cpu")
                )
                .abs()
                .max()
            )
            raise RuntimeError(
                "generator token-VJP losses differ from parent authority; "
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
        token_scores = build_gemma_generator_innovation_token_scores(
            example=example,
            parent_execution=execution,
            token_loss_gradients=token_vjp.h4_gradients,
            supervised_token_logical_positions=(
                supervised_logical_positions
            ),
            parent_h4=parent_h4,  # type: ignore[arg-type]
            parent_observation=parent_observation,
            fixed_generator_basis=basis,  # type: ignore[arg-type]
        )
        compensation_target = (
            source_token_nll.to(device="cpu")
            - parent_token_nll.to(device="cpu")
        ).contiguous()
        legacy = build_token_loss_fisher_prompt_record(
            example_id=example.example_id,
            family_id=example.family_id,
            coordinate_names=(
                GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER
            ),
            token_scores=token_scores.legacy_cumulative_token_scores,
            compensation_target=compensation_target,
        )
        generator = build_token_loss_fisher_prompt_record(
            example_id=example.example_id,
            family_id=example.family_id,
            coordinate_names=GENERATOR_INNOVATION_TANGENT_ORDER,
            token_scores=(
                token_scores.generator_innovation_token_scores
            ),
            compensation_target=compensation_target,
        )
        tangent = token_scores.source_tangent_record
        if (
            manifest[example.example_id] != example.family_id
            or legacy.example_id != generator.example_id
            or legacy.family_id != generator.family_id
            or legacy.supervised_tokens != generator.supervised_tokens
            or legacy.compensation_target_sha256
            != generator.compensation_target_sha256
            or legacy.target_second_moment
            != generator.target_second_moment
            or tangent.example_id != example.example_id
            or tangent.family_id != example.family_id
            or tangent.supervised_token_count
            != legacy.supervised_tokens
        ):
            raise RuntimeError(
                "generator innovation prompt/target binding differs"
            )
        legacy_records.append(legacy)
        generator_records.append(generator)
        feature_receipts[example.example_id] = dict(
            token_scores.feature_summary
        )
        top_mode_receipts[example.example_id] = {
            "top_mode_indices": token_scores.top_mode_indices,
            "top_mode_norms": token_scores.top_mode_norms,
        }
        vjp_receipts[example.example_id] = token_vjp.artifact_sha256
        tangent_receipts[example.example_id] = (
            tangent.token_tangent_record_sha256
        )
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
            retained_vjp_losses,
            supervised_logical_positions,
            token_scores,
            compensation_target,
            legacy,
            generator,
            tangent,
        )

    return build_gemma_iterative_generator_innovation_development_report(
        legacy_records=legacy_records,
        generator_records=generator_records,
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        feature_summary_by_example=feature_receipts,
        top_mode_receipt_by_example=top_mode_receipts,
        token_vjp_artifact_sha256_by_example=vjp_receipts,
        source_tangent_record_sha256_by_example=tangent_receipts,
        total_backward_call_count=backward_calls,
        vjp_chunk_size=GENERATOR_INNOVATION_VJP_CHUNK_SIZE,
        lineage=lineage,
        collection_lineage=collection_lineage,
    )


def run_gemma_iterative_generator_innovation_panel_preparation(
    *,
    plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expected_plan_sha256: str = FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    expected_plan_file_sha256: str = (
        FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ),
    expanded_fit_corpus_path: Path | str = DEFAULT_EXPANDED_FIT_CORPUS,
    prior_occupancy_panel_path: Path | str = DEFAULT_PRIOR_OCCUPANCY_PANEL,
    private_output: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT
    ),
    receipt_output: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT
    ),
) -> dict[str, object]:
    """Publish the exact private role and prompt-free receipt once."""

    return prepare_gemma3_l3_l4_generator_innovation_panel(
        plan_path=plan_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_plan_file_sha256=expected_plan_file_sha256,
        expanded_fit_corpus_path=expanded_fit_corpus_path,
        prior_occupancy_panel_path=prior_occupancy_panel_path,
        private_output=private_output,
        receipt_output=receipt_output,
    )


def run_gemma_iterative_generator_innovation_development_diagnostic(
    *,
    plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expected_plan_sha256: str = FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    expected_plan_file_sha256: str = (
        FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ),
    panel_receipt_path: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT
    ),
    expected_panel_receipt_sha256: str,
    expected_panel_receipt_file_sha256: str,
    private_role_input_path: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT
    ),
    expected_private_role_input_file_sha256: str,
    prior_occupancy_panel_path: Path | str = DEFAULT_PRIOR_OCCUPANCY_PANEL,
    corpus_artifact_path: Path | str = DEFAULT_EXPANDED_FIT_CORPUS,
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
    """Authenticate the fixed rung, collect exact derivatives, and stop."""

    plan_path = Path(plan_path)
    panel_receipt_path = Path(panel_receipt_path)
    private_role_input_path = Path(private_role_input_path)
    prior_occupancy_panel_path = Path(prior_occupancy_panel_path)
    corpus_artifact_path = Path(corpus_artifact_path)
    plan = _load_plan(
        plan_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_plan_file_sha256=expected_plan_file_sha256,
    )
    receipt = load_gemma3_l3_l4_generator_innovation_panel_receipt(
        panel_receipt_path,
        plan_path=plan_path,
        expanded_fit_corpus_path=corpus_artifact_path,
        prior_occupancy_panel_path=prior_occupancy_panel_path,
    )
    actual_receipt_file_sha256 = _file_sha256(panel_receipt_path)
    actual_private_file_sha256 = _file_sha256(private_role_input_path)
    if (
        receipt.receipt_sha256 != expected_panel_receipt_sha256
        or actual_receipt_file_sha256
        != expected_panel_receipt_file_sha256
        or receipt.role_input_file_sha256
        != expected_private_role_input_file_sha256
        or actual_private_file_sha256
        != expected_private_role_input_file_sha256
        or receipt.plan_sha256 != expected_plan_sha256
        or receipt.plan_file_sha256 != expected_plan_file_sha256
    ):
        raise ValueError(
            "generator innovation plan/panel/private identity differs"
        )

    frozen_basis = _mapping(
        plan.get("frozen_generator_basis"),
        label="frozen generator basis",
    )
    basis_sha256 = frozen_basis.get("basis_sha256")
    if not isinstance(basis_sha256, str):
        raise ValueError("generator innovation basis receipt differs")
    plan_lineage = _mapping(
        plan.get("lineage"),
        label="generator innovation plan lineage",
    )
    parent_lineage = _mapping(
        plan_lineage.get("token_fisher_model_and_parent_lineage"),
        label="generator innovation parent lineage",
    )
    collection_lineage: dict[str, object] = {
        "plan_sha256": expected_plan_sha256,
        "plan_file_sha256": expected_plan_file_sha256,
        "basis_sha256": basis_sha256,
        "collection_role_input_file_sha256": (
            expected_private_role_input_file_sha256
        ),
        "collection_manifest_sha256": receipt.manifest_sha256,
        "collection_membership_receipt_sha256": (
            receipt.membership_receipt_sha256
        ),
        "prompt_free_panel_artifact_receipt_sha256": (
            receipt.receipt_sha256
        ),
    }
    extra_lineage = {
        key: value
        for key, value in collection_lineage.items()
        if isinstance(value, str)
    }
    recipe = _GemmaDevelopmentCollectionRecipe(
        collect=partial(
            _collect_generator_innovation,
            plan=plan,
            plan_file_sha256=expected_plan_file_sha256,
            collection_lineage=collection_lineage,
        ),
        validate_report=(
            validate_gemma_iterative_generator_innovation_development_report
        ),
        publish_report=partial(
            _publish_generator_innovation_report_once,
            plan=plan,
            plan_file_sha256=expected_plan_file_sha256,
        ),
        report_label="fixed-basis generator innovation development",
        expected_parent_lineage={
            str(key): str(value) for key, value in parent_lineage.items()
        },
        extra_lineage=extra_lineage,
        extra_immutable_inputs=(
            (
                "generator_innovation_plan",
                plan_path,
                expected_plan_file_sha256,
            ),
            (
                "generator_innovation_panel_receipt",
                panel_receipt_path,
                expected_panel_receipt_file_sha256,
            ),
            (
                "generator_innovation_private_role",
                private_role_input_path,
                expected_private_role_input_file_sha256,
            ),
            (
                "generator_innovation_prior_occupancy_panel",
                prior_occupancy_panel_path,
                receipt.prior_occupancy_panel_file_sha256,
            ),
        ),
        source_code_files=(
            "gemma3_l3_l4_iterative_generator_innovation_diagnostic.py",
            "gemma3_l3_l4_iterative_generator_innovation_development.py",
            "gemma3_l3_l4_iterative_generator_innovation_edges.py",
            "gemma3_l3_l4_iterative_generator_innovation.py",
            "gemma3_l3_l4_iterative_generator_innovation_panel.py",
            "gemma3_l3_l4_iterative_generator_innovation_plan.py",
            "gemma3_l3_l4_iterative_residual_diagnostic.py",
            "gemma3_l3_l4_iterative_occupancy_route.py",
            "gemma3_l3_l4_iterative_state_router.py",
            "gemma3_l3_l4_iterative_token_fisher_edges.py",
            "token_loss_fisher.py",
            "token_loss_fisher_generator_innovation.py",
        ),
        collection_panel_factory=partial(
            _materialize_collection_panel,
            receipt=receipt,
            role_input_path=private_role_input_path,
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
    validate_gemma_iterative_generator_innovation_development_report(report)
    published_raw = json.loads(Path(output).read_text(encoding="utf-8"))
    if not isinstance(published_raw, Mapping):
        raise TypeError(
            "published generator innovation report must contain one object"
        )
    published = dict(published_raw)
    validate_gemma_iterative_generator_innovation_development_report(
        published
    )
    replay = replay_gemma_iterative_generator_innovation_development_report(
        report=published,
        plan=plan,
        plan_file_sha256=expected_plan_file_sha256,
    )
    if _canonical(replay) != _canonical(report):
        raise RuntimeError(
            "published generator innovation development report differs"
        )
    return report


def preparation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PREPARATION_CLI_NAME,
        description=(
            "Authenticate the frozen generator plan, then publish the "
            "family-disjoint private fit role and prompt-free receipt once."
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PLAN,
    )
    parser.add_argument(
        "--plan-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    )
    parser.add_argument(
        "--plan-file-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256,
    )
    parser.add_argument(
        "--expanded-fit-corpus",
        type=Path,
        default=DEFAULT_EXPANDED_FIT_CORPUS,
    )
    parser.add_argument(
        "--prior-occupancy-panel",
        type=Path,
        default=DEFAULT_PRIOR_OCCUPANCY_PANEL,
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT,
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    )
    return parser


def preparation_main(argv: Sequence[str] | None = None) -> int:
    args = preparation_parser().parse_args(argv)
    receipt = run_gemma_iterative_generator_innovation_panel_preparation(
        plan_path=args.plan,
        expected_plan_sha256=args.plan_sha256,
        expected_plan_file_sha256=args.plan_file_sha256,
        expanded_fit_corpus_path=args.expanded_fit_corpus,
        prior_occupancy_panel_path=args.prior_occupancy_panel,
        private_output=args.private_output,
        receipt_output=args.receipt_output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = build_residual_parser()
    parser.prog = CLI_NAME
    parser.description = (
        "Collect exact fixed-basis generator-innovation token derivatives "
        "on the new family-disjoint A-fit panel and stop before finite "
        "displacement, selection, or provider compilation."
    )
    parser.set_defaults(output=DEFAULT_OUTPUT)
    parser.add_argument(
        "--generator-plan",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PLAN,
    )
    parser.add_argument(
        "--generator-plan-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    )
    parser.add_argument(
        "--generator-plan-file-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256,
    )
    parser.add_argument(
        "--generator-panel-receipt",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    )
    parser.add_argument(
        "--generator-panel-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--generator-panel-receipt-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--generator-private-role-input",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT,
    )
    parser.add_argument(
        "--generator-private-role-input-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-occupancy-panel",
        type=Path,
        default=DEFAULT_PRIOR_OCCUPANCY_PANEL,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = (
        run_gemma_iterative_generator_innovation_development_diagnostic(
            plan_path=args.generator_plan,
            expected_plan_sha256=args.generator_plan_sha256,
            expected_plan_file_sha256=(
                args.generator_plan_file_sha256
            ),
            panel_receipt_path=args.generator_panel_receipt,
            expected_panel_receipt_sha256=(
                args.generator_panel_receipt_sha256
            ),
            expected_panel_receipt_file_sha256=(
                args.generator_panel_receipt_file_sha256
            ),
            private_role_input_path=args.generator_private_role_input,
            expected_private_role_input_file_sha256=(
                args.generator_private_role_input_file_sha256
            ),
            prior_occupancy_panel_path=args.prior_occupancy_panel,
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
