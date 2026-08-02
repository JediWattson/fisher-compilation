"""Live, two-pass orchestration for adaptive generator innovation v2.

``scale`` executes the accepted parent exactly once for each member of the
published v1 16-by-8 development panel.  It never opens targets, token losses,
or gradients.  Raw traces remain transient; a standalone protocol receipt and
a live scale-development wrapper are published once.

``plan`` is model-free.  It authenticates the failed v1 development rung and
the standalone scale receipt, then freezes the 13-candidate adaptive plan.

``target`` rematerializes the exact same panel and spends one source-authority
forward plus one retained-parent token-VJP forward per prompt.  It verifies
the pre-target feature receipt before the single gradient contraction, reduces
Q6 and all 13 R4 banks to prompt sufficient statistics, runs the nested
family analyzer, and stops before any finite displacement or provider.
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
from .gemma3_l3_l4_iterative_generator_innovation_development import (
    validate_gemma_iterative_generator_innovation_development_report,
)
from .gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
)
from .gemma3_l3_l4_iterative_generator_innovation_diagnostic import (
    GENERATOR_INNOVATION_PARENT_LOSS_AUTHORITY_TOLERANCE,
    GENERATOR_INNOVATION_VJP_CHUNK_SIZE,
    _load_plan,
    _materialize_collection_panel,
)
from .gemma3_l3_l4_iterative_generator_innovation_panel import (
    DEFAULT_EXPANDED_FIT_CORPUS,
    DEFAULT_GENERATOR_INNOVATION_PLAN,
    DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT,
    DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    DEFAULT_PRIOR_OCCUPANCY_PANEL,
    FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256,
    FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    load_gemma3_l3_l4_generator_innovation_panel_receipt,
)
from .gemma3_l3_l4_iterative_generator_innovation_v2_development import (
    build_gemma_iterative_generator_innovation_v2_scale_development_report,
    build_gemma_iterative_generator_innovation_v2_target_development_report,
    replay_gemma_iterative_generator_innovation_v2_scale_development_report,
    replay_gemma_iterative_generator_innovation_v2_target_development_report,
    validate_gemma_iterative_generator_innovation_v2_scale_development_report,
    validate_gemma_iterative_generator_innovation_v2_target_development_report,
)
from .gemma3_l3_l4_iterative_generator_innovation_v2_edges import (
    build_gemma_generator_innovation_v2_token_scores,
    extract_gemma_generator_innovation_v2_activations,
)
from .gemma3_l3_l4_iterative_generator_innovation_v2_protocol import (
    GENERATOR_INNOVATION_V2_CANDIDATE_ORDER,
    GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER,
    build_gemma_iterative_generator_innovation_v2_candidate_plan,
    build_gemma_iterative_generator_innovation_v2_scale_receipt,
    generator_innovation_v2_candidate_specs,
    replay_gemma_iterative_generator_innovation_v2_candidate_plan,
    validate_gemma_iterative_generator_innovation_v2_candidate_plan,
    validate_gemma_iterative_generator_innovation_v2_scale_receipt,
)
from .gemma3_l3_l4_iterative_generator_innovation_edges import (
    GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
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
    "DEFAULT_CANDIDATE_PLAN_OUTPUT",
    "DEFAULT_SCALE_OUTPUT",
    "DEFAULT_SCALE_RECEIPT_OUTPUT",
    "DEFAULT_TARGET_OUTPUT",
    "candidate_plan_main",
    "candidate_plan_parser",
    "main",
    "run_gemma_iterative_generator_innovation_v2_candidate_plan_preparation",
    "run_gemma_iterative_generator_innovation_v2_scale_diagnostic",
    "run_gemma_iterative_generator_innovation_v2_target_diagnostic",
    "scale_main",
    "scale_parser",
    "target_main",
    "target_parser",
]


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
_DEFAULT_V1_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-dev-v1.report.json"
)
DEFAULT_SCALE_RECEIPT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-v2-scale-v1.receipt.json"
)
DEFAULT_SCALE_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-v2-scale-v1.report.json"
)
DEFAULT_CANDIDATE_PLAN_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-v2-candidate-plan-v1.json"
)
DEFAULT_TARGET_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-v2-target-dev-v1.report.json"
)

_SOURCE_CODE_FILES = (
    "gemma3_l3_l4_iterative_generator_innovation_v2_diagnostic.py",
    "gemma3_l3_l4_iterative_generator_innovation_v2_development.py",
    "gemma3_l3_l4_iterative_generator_innovation_v2_protocol.py",
    "gemma3_l3_l4_iterative_generator_innovation_v2_edges.py",
    "gemma3_l3_l4_iterative_generator_innovation_v2.py",
    "gemma3_l3_l4_iterative_generator_innovation_diagnostic.py",
    "gemma3_l3_l4_iterative_generator_innovation_development.py",
    "gemma3_l3_l4_iterative_generator_innovation_panel.py",
    "gemma3_l3_l4_iterative_generator_innovation_plan.py",
    "gemma3_l3_l4_iterative_residual_diagnostic.py",
    "gemma3_l3_l4_iterative_residual_campaign.py",
    "gemma3_l3_l4_iterative_occupancy_route.py",
    "gemma3_l3_l4_iterative_state_router.py",
    "gemma3_l3_l4_iterative_token_fisher_edges.py",
    "token_loss_fisher.py",
    "token_loss_fisher_generator_innovation.py",
    "token_loss_fisher_generator_innovation_adaptive_v2.py",
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serialized(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load_mapping(path: Path | str, *, label: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain one object")
    return dict(value)


def _source_code_sha256s() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _SOURCE_CODE_FILES
    }


def _write_once(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_serialized(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_scale_once(
    destination: Path,
    report: Mapping[str, object],
    *,
    scale_receipt_destination: Path,
) -> None:
    validate_gemma_iterative_generator_innovation_v2_scale_development_report(
        report
    )
    replay_gemma_iterative_generator_innovation_v2_scale_development_report(
        report
    )
    receipt = _mapping(report["scale_receipt"], label="scale receipt")
    expected_file = _mapping(report["lineage"], label="scale lineage")[
        "scale_receipt_file_sha256"
    ]
    if hashlib.sha256(_serialized(receipt)).hexdigest() != expected_file:
        raise RuntimeError("standalone scale-receipt serialization differs")
    if destination.exists() or scale_receipt_destination.exists():
        raise FileExistsError("refusing to overwrite v2 scale artifacts")
    _write_once(scale_receipt_destination, receipt)
    try:
        _write_once(destination, report)
    except BaseException:
        scale_receipt_destination.unlink(missing_ok=True)
        raise


def _publish_target_once(
    destination: Path,
    report: Mapping[str, object],
    *,
    candidate_plan: Mapping[str, object],
    candidate_plan_file_sha256: str,
    scale_receipt_file_sha256: str,
    scale_development_report: Mapping[str, object],
    scale_development_report_file_sha256: str,
) -> None:
    validate_gemma_iterative_generator_innovation_v2_target_development_report(
        report
    )
    replay = (
        replay_gemma_iterative_generator_innovation_v2_target_development_report(
            report=report,
            candidate_plan=candidate_plan,
            candidate_plan_file_sha256=candidate_plan_file_sha256,
            scale_receipt_file_sha256=scale_receipt_file_sha256,
            scale_development_report=scale_development_report,
            scale_development_report_file_sha256=(
                scale_development_report_file_sha256
            ),
        )
    )
    if _canonical(replay) != _canonical(report):
        raise RuntimeError("v2 target report replay differs")
    _write_once(destination, report)


def _transpose_pairs(value: object) -> dict[str, dict[str, object]]:
    rows = _mapping(value, label="two-channel summary")
    result = {"real": {}, "imag": {}}
    for key in ("q50", "q90", "q99"):
        pair = rows[key]
        if pair is None:
            result["real"][key] = None
            result["imag"][key] = None
            continue
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("two-channel quantile pair differs")
        result["real"][key] = pair[0]
        result["imag"][key] = pair[1]
    return result


def _transpose_counts(value: object) -> dict[str, dict[str, int]]:
    rows = _mapping(value, label="two-channel sign counts")
    result = {"real": {}, "imag": {}}
    for key in ("negative", "zero", "positive"):
        pair = rows[key]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("two-channel sign-count pair differs")
        result["real"][key] = int(pair[0])
        result["imag"][key] = int(pair[1])
    return result


def _raw_source_row(
    trace: Mapping[str, object],
    *,
    source_id: str,
) -> dict[str, object]:
    buckets: dict[str, object] = {}
    for raw_bucket in trace["age_buckets"]:  # type: ignore[union-attr]
        bucket = _mapping(raw_bucket, label="raw age bucket")
        bucket_id = str(bucket["age_bucket"])
        buckets[bucket_id] = {
            "active_count": int(bucket["active_activation_row_count"]),
            "abs_raw_quantiles_by_channel": _transpose_pairs(
                bucket["absolute_raw_quantiles_by_channel"]
            ),
            "sign_counts_by_channel": _transpose_counts(
                bucket["sign_counts_by_channel"]
            ),
        }
    half_life = trace["half_life"]
    return {
        "source_id": source_id,
        "half_life_active_positions": (
            None if half_life is None else int(float(half_life))
        ),
        "active_count": int(trace["active_activation_row_count"]),
        "abs_raw_quantiles_by_channel": _transpose_pairs(
            trace["absolute_raw_quantiles_by_channel"]
        ),
        "sign_counts_by_channel": _transpose_counts(
            trace["sign_counts_by_channel"]
        ),
        "age_buckets": buckets,
        "raw_trace_sha256": trace["selected_raw_trace_sha256"],
        "prior_kind": (
            "none_current_only"
            if source_id == "current_only"
            else "ew_prior_before_current_update"
        ),
        "whole_sequence_equals_two_chunks": True,
        "padding_updates_state": False,
    }


def _scale_example_summary(
    extraction: object,
    *,
    candidate_feature_receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    raw = _mapping(
        getattr(extraction, "raw_trace_receipt"),
        label="raw trace receipt",
    )
    temporal = {
        f"ew{int(float(_mapping(row, label='temporal trace')['half_life'])):02d}":
        _mapping(row, label="temporal trace")
        for row in raw["temporal_traces"]  # type: ignore[union-attr]
    }
    sources = {
        "current_only": _raw_source_row(
            _mapping(raw["current_only_trace"], label="current trace"),
            source_id="current_only",
        ),
        **{
            source_id: _raw_source_row(
                temporal[source_id],
                source_id=source_id,
            )
            for source_id in GENERATOR_INNOVATION_V2_RAW_SOURCE_ORDER[1:]
        },
    }
    candidate_hash = (
        "0" * 64
        if candidate_feature_receipt is None
        else candidate_feature_receipt[
            "candidate_feature_receipt_sha256"
        ]
    )
    health: dict[str, object] = {}
    summaries = (
        ()
        if candidate_feature_receipt is None
        else candidate_feature_receipt["candidate_summaries"]
    )
    by_id = {
        str(_mapping(row, label="candidate summary")["candidate_id"]):
        _mapping(row, label="candidate summary")
        for row in summaries  # type: ignore[union-attr]
    }
    active_count = int(
        _mapping(sources["current_only"], label="current source")[
            "active_count"
        ]
    )
    for candidate_id in GENERATOR_INNOVATION_V2_CANDIDATE_ORDER:
        row = by_id.get(candidate_id)
        health[candidate_id] = {
            "candidate_id": candidate_id,
            "active_count": active_count,
            "q90_absolute_bounded_by_channel": (
                (0.0, 0.0)
                if row is None
                else row["q90_absolute_feature_by_channel"]
            ),
            "central_fraction_by_channel": (
                (0.0, 0.0)
                if row is None
                else row["central_fraction_by_channel"]
            ),
            "bounded_trace_sha256": (
                "0" * 64
                if row is None
                else row["selected_bounded_feature_sha256"]
            ),
            "candidate_feature_receipt_sha256": candidate_hash,
        }
    return {
        "example_id": raw["example_id"],
        "family_id": raw["family_id"],
        "active_count": active_count,
        "top_mode_indices": raw["top_mode_indices"],
        "top_mode_norms": raw["top_mode_norms"],
        "parent_modal_trace_sha256": raw["raw_trace_receipt_sha256"],
        "raw_by_source": sources,
        "candidate_health_by_id": health,
        "audit": {
            "accepted_parent_only": True,
            "candidate_output_read": False,
            "compensation_target_read": False,
            "prompt_or_family_outcome_read": False,
            "raw_feature_rows_retained": False,
            "raw_modal_rows_retained": False,
            "target_blind": True,
            "token_gradient_read": False,
            "token_loss_read": False,
        },
    }


def _validate_live_parent(
    *,
    parent_fit_panel: object,
    adapter: object,
    bridge: object,
    parent_artifact: object,
    parent_h4: object,
    x4_head: object,
) -> None:
    validated_x4, validated_h4 = _validate_parent(
        panel=parent_fit_panel,
        adapter=adapter,
        bridge=bridge,
        parent=parent_artifact,
    )
    if (
        getattr(validated_x4, "artifact_sha256", None)
        != getattr(x4_head, "artifact_sha256", None)
        or getattr(validated_h4, "artifact_sha256", None)
        != getattr(parent_h4, "artifact_sha256", None)
    ):
        raise ValueError("v2 live parent heads differ from accepted parent")


def _collect_scale(
    *,
    prior_lineage: Mapping[str, object],
    panel: object,
    parent_fit_panel: object,
    adapter: object,
    bridge: object,
    parent_artifact: object,
    parent_h4: object,
    x4_head: object,
    lineage: Mapping[str, object],
) -> Mapping[str, object]:
    _validate_live_parent(
        parent_fit_panel=parent_fit_panel,
        adapter=adapter,
        bridge=bridge,
        parent_artifact=parent_artifact,
        parent_h4=parent_h4,
        x4_head=x4_head,
    )
    if dict(lineage) != dict(prior_lineage):
        raise ValueError("v2 scale live lineage differs")
    examples = getattr(panel, "examples", None)
    if not isinstance(examples, tuple) or len(examples) != 16:
        raise TypeError("v2 scale panel must contain exactly 16 examples")
    extractions: dict[str, object] = {}
    preliminary: dict[str, object] = {}
    with torch.no_grad():
        for example in examples:
            example.validate_integrity()
            execution = bridge.execute(
                adapter,
                example.batch.model_inputs,
                x4_head=x4_head,
                h4_head=parent_h4,
            )
            _validate_execution(
                execution,
                example_model_inputs_sha256=example.model_inputs_sha256,
                bridge_binding_sha256=bridge.bridge_binding_sha256,
                x4_head=x4_head,
                h4_head=parent_h4,
                label="generator innovation v2 target-blind scale",
            )
            extraction = extract_gemma_generator_innovation_v2_activations(
                example=example,
                parent_execution=execution,
                parent_h4=parent_h4,
            )
            extractions[example.example_id] = extraction
            preliminary[example.example_id] = _scale_example_summary(
                extraction,
                candidate_feature_receipt=None,
            )
    preliminary = dict(sorted(preliminary.items()))
    provisional = build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=preliminary,
        prior_lineage=prior_lineage,
    )
    candidate_specs = generator_innovation_v2_candidate_specs(provisional)
    final_rows: dict[str, object] = {}
    feature_hashes: dict[str, object] = {}
    raw_hashes: dict[str, object] = {}
    for example_id, extraction in sorted(extractions.items()):
        feature = extraction.build_candidate_feature_receipt(candidate_specs)
        final_rows[example_id] = _scale_example_summary(
            extraction,
            candidate_feature_receipt=feature,
        )
        feature_hashes[example_id] = feature[
            "candidate_feature_receipt_sha256"
        ]
        raw_hashes[example_id] = extraction.raw_trace_receipt[
            "raw_trace_receipt_sha256"
        ]
    scale_receipt = build_gemma_iterative_generator_innovation_v2_scale_receipt(
        per_example_raw_summaries=final_rows,
        prior_lineage=prior_lineage,
    )
    scale_file_sha256 = hashlib.sha256(
        _serialized(scale_receipt)
    ).hexdigest()
    report = (
        build_gemma_iterative_generator_innovation_v2_scale_development_report(
            scale_receipt=scale_receipt,
            scale_receipt_file_sha256=scale_file_sha256,
            candidate_feature_receipt_sha256_by_example_id=feature_hashes,
            raw_trace_receipt_sha256_by_example_id=raw_hashes,
            source_code_sha256_by_file=_source_code_sha256s(),
        )
    )
    del extractions, preliminary, provisional, final_rows
    return report


def _collect_target(
    *,
    candidate_plan: Mapping[str, object],
    candidate_plan_file_sha256: str,
    scale_receipt_file_sha256: str,
    scale_development_report: Mapping[str, object],
    scale_development_report_file_sha256: str,
    fixed_basis: Sequence[Sequence[float]],
    panel: object,
    parent_fit_panel: object,
    adapter: object,
    bridge: object,
    parent_artifact: object,
    parent_h4: object,
    x4_head: object,
    lineage: Mapping[str, object],
) -> Mapping[str, object]:
    manifest = _panel_manifest(panel)
    _validate_live_parent(
        parent_fit_panel=parent_fit_panel,
        adapter=adapter,
        bridge=bridge,
        parent_artifact=parent_artifact,
        parent_h4=parent_h4,
        x4_head=x4_head,
    )
    bank = _mapping(candidate_plan["candidate_bank"], label="candidate bank")
    candidate_specs = tuple(bank["candidate_specs"])  # type: ignore[arg-type]
    candidate_ids = tuple(
        str(_mapping(row, label="candidate spec")["candidate_id"])
        for row in candidate_specs
    )
    scale_binding = _mapping(
        scale_development_report["pre_target_feature_binding"],
        label="scale feature binding",
    )
    expected_features = _mapping(
        scale_binding["candidate_feature_receipt_sha256_by_example_id"],
        label="scale feature hashes",
    )
    examples = getattr(panel, "examples", None)
    if not isinstance(examples, tuple) or len(examples) != 16:
        raise TypeError("v2 target panel must contain exactly 16 examples")
    legacy_records: list[object] = []
    record_bank: dict[str, list[object]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    feature_hashes: dict[str, object] = {}
    raw_hashes: dict[str, object] = {}
    score_hashes: dict[str, object] = {}
    vjp_hashes: dict[str, object] = {}
    backward_calls = 0
    for example in examples:
        example.validate_integrity()
        (
            source_execution,
            source_logits,
            supervised_indices,
            targets,
            logical_positions,
        ) = _source_authority(adapter=adapter, example=example)
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
            label="generator innovation v2 retained-parent token VJP",
        )
        expected_grid = torch.stack(
            (torch.zeros_like(supervised_indices), supervised_indices),
            dim=1,
        ).to(token_vjp.supervised_indices.device, dtype=torch.int64)
        if not torch.equal(token_vjp.supervised_indices, expected_grid):
            raise ValueError("v2 supervised-token order differs")
        parent_logits = _gather_logits(execution.logits, supervised_indices)
        observation = _observation(
            example=example,
            source_logits=source_logits,
            candidate_logits=parent_logits,
            targets=targets,
        )
        source_nll = F.cross_entropy(
            source_logits,
            targets.to(source_logits.device),
            reduction="none",
        ).to(torch.float64)
        parent_nll = F.cross_entropy(
            parent_logits,
            targets.to(parent_logits.device),
            reduction="none",
        ).to(torch.float64)
        if not torch.allclose(
            token_vjp.token_losses.detach().to("cpu", torch.float64),
            parent_nll.to("cpu"),
            rtol=0.0,
            atol=GENERATOR_INNOVATION_PARENT_LOSS_AUTHORITY_TOLERANCE,
        ):
            raise RuntimeError("v2 token-VJP losses differ from parent NLL")
        supervised_positions = (
            logical_positions[0]
            .index_select(0, supervised_indices.to(logical_positions.device))
            .detach()
            .to("cpu", torch.int64)
            .contiguous()
        )
        scores = build_gemma_generator_innovation_v2_token_scores(
            example=example,
            parent_execution=execution,
            token_loss_gradients=token_vjp.h4_gradients,
            supervised_token_logical_positions=supervised_positions,
            parent_h4=parent_h4,
            parent_observation=observation,
            fixed_generator_basis=fixed_basis,
            candidate_specs=candidate_specs,
            expected_candidate_feature_receipt=str(
                expected_features[example.example_id]
            ),
        )
        target = (
            source_nll.to("cpu") - parent_nll.to("cpu")
        ).contiguous()
        legacy = build_token_loss_fisher_prompt_record(
            example_id=example.example_id,
            family_id=example.family_id,
            coordinate_names=GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
            token_scores=scores.legacy_q6_token_scores,
            compensation_target=target,
        )
        if manifest[example.example_id] != example.family_id:
            raise RuntimeError("v2 target panel membership differs")
        legacy_records.append(legacy)
        for candidate_id in candidate_ids:
            record = _build_candidate_prompt_record(
                example_id=example.example_id,
                family_id=example.family_id,
                token_scores=scores.candidate_r4_token_scores[candidate_id],
                compensation_target=target,
            )
            if (
                record.supervised_tokens != legacy.supervised_tokens
                or record.compensation_target_sha256
                != legacy.compensation_target_sha256
            ):
                raise RuntimeError("v2 Q6/R4 target binding differs")
            record_bank[candidate_id].append(record)
        feature_hashes[example.example_id] = (
            scores.candidate_feature_receipt[
                "candidate_feature_receipt_sha256"
            ]
        )
        raw_hashes[example.example_id] = scores.raw_trace_receipt[
            "raw_trace_receipt_sha256"
        ]
        score_hashes[example.example_id] = scores.score_receipt[
            "score_receipt_sha256"
        ]
        vjp_hashes[example.example_id] = token_vjp.artifact_sha256
        backward_calls += token_vjp.backward_call_count
        del (
            source_execution,
            source_logits,
            supervised_indices,
            targets,
            logical_positions,
            token_vjp,
            execution,
            expected_grid,
            parent_logits,
            observation,
            source_nll,
            parent_nll,
            supervised_positions,
            scores,
            target,
            legacy,
        )
    return build_gemma_iterative_generator_innovation_v2_target_development_report(
        record_bank=record_bank,
        legacy_records=legacy_records,
        fixed_basis=fixed_basis,
        candidate_plan=candidate_plan,
        candidate_plan_file_sha256=candidate_plan_file_sha256,
        scale_receipt_file_sha256=scale_receipt_file_sha256,
        scale_development_report=scale_development_report,
        scale_development_report_file_sha256=(
            scale_development_report_file_sha256
        ),
        live_lineage=lineage,
        candidate_feature_receipt_sha256_by_example_id=feature_hashes,
        raw_trace_receipt_sha256_by_example_id=raw_hashes,
        score_receipt_sha256_by_example_id=score_hashes,
        token_vjp_artifact_sha256_by_example_id=vjp_hashes,
        total_backward_call_count=backward_calls,
        vjp_chunk_size=GENERATOR_INNOVATION_VJP_CHUNK_SIZE,
        source_code_sha256_by_file=_source_code_sha256s(),
    )


def _build_candidate_prompt_record(
    *,
    example_id: str,
    family_id: str,
    token_scores: torch.Tensor,
    compensation_target: torch.Tensor,
) -> object:
    """Reduce one bank member in the analyzer's shared R4 coordinates."""

    return build_token_loss_fisher_prompt_record(
        example_id=example_id,
        family_id=family_id,
        coordinate_names=GENERATOR_INNOVATION_TANGENT_ORDER,
        token_scores=token_scores,
        compensation_target=compensation_target,
    )


def _authenticate_v1(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    panel_receipt_path: Path,
    expected_panel_receipt_sha256: str,
    expected_panel_receipt_file_sha256: str,
    private_role_input_path: Path,
    expected_private_role_input_file_sha256: str,
    v1_report_path: Path,
    expected_v1_report_sha256: str,
    expected_v1_report_file_sha256: str,
    corpus_artifact_path: Path,
    prior_occupancy_panel_path: Path,
) -> tuple[
    dict[str, object],
    object,
    dict[str, object],
    dict[str, object],
    dict[str, str],
]:
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
    report = _load_mapping(v1_report_path, label="v1 development report")
    validate_gemma_iterative_generator_innovation_development_report(report)
    if (
        receipt.receipt_sha256 != expected_panel_receipt_sha256
        or _file_sha256(panel_receipt_path)
        != expected_panel_receipt_file_sha256
        or receipt.role_input_file_sha256
        != expected_private_role_input_file_sha256
        or _file_sha256(private_role_input_path)
        != expected_private_role_input_file_sha256
        or report.get("report_sha256") != expected_v1_report_sha256
        or _file_sha256(v1_report_path) != expected_v1_report_file_sha256
    ):
        raise ValueError("v1 plan/panel/private/report identity differs")
    frozen_basis = _mapping(
        plan["frozen_generator_basis"],
        label="v1 frozen generator basis",
    )
    basis_sha256 = str(frozen_basis["basis_sha256"])
    report_lineage = _mapping(report["lineage"], label="v1 report lineage")
    report_plan = _mapping(
        report_lineage["plan"],
        label="v1 report plan binding",
    )
    collection = _mapping(
        report_lineage["collection"],
        label="v1 collection lineage",
    )
    if (
        report_plan.get("plan_sha256") != expected_plan_sha256
        or report_plan.get("plan_file_sha256")
        != expected_plan_file_sha256
        or report_plan.get("basis_sha256") != basis_sha256
        or collection.get("collection_manifest_sha256")
        != receipt.manifest_sha256
        or collection.get("collection_membership_receipt_sha256")
        != receipt.membership_receipt_sha256
        or collection.get("collection_role_input_file_sha256")
        != expected_private_role_input_file_sha256
    ):
        raise ValueError("v1 report and panel collection lineage differ")
    prior = {
        **{
            key: str(value)
            for key, value in _mapping(
                report_lineage["live_lineage"],
                label="v1 live lineage",
            ).items()
            if key
            in {
                "accepted_x4_head_sha256",
                "adapter_execution_sha256",
                "basis_sha256",
                "bridge_binding_sha256",
                "collection_manifest_sha256",
                "collection_membership_receipt_sha256",
                "collection_role_input_file_sha256",
                "factorial_report_file_sha256",
                "factorial_report_sha256",
                "fit_manifest_sha256",
                "model_sha256",
                "parent_artifact_sha256",
                "parent_h4_head_sha256",
            }
        },
        "v1_plan_sha256": expected_plan_sha256,
        "v1_plan_file_sha256": expected_plan_file_sha256,
        "v1_development_report_sha256": expected_v1_report_sha256,
        "v1_development_report_file_sha256": (
            expected_v1_report_file_sha256
        ),
        "v1_panel_receipt_sha256": expected_panel_receipt_sha256,
        "v1_panel_receipt_file_sha256": (
            expected_panel_receipt_file_sha256
        ),
    }
    return plan, receipt, report, dict(frozen_basis), prior


def _base_run_kwargs(
    *,
    corpus_artifact_path: Path,
    fit_input_path: Path,
    materialization_report_path: Path,
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    factorial_report_path: Path,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    graph_candidate_path: Path,
    basis_package_path: Path,
    base_artifact_path: Path,
    refit_artifact_path: Path,
    output: Path,
    cache_dir: Path | None,
    recipe: _GemmaDevelopmentCollectionRecipe,
) -> dict[str, object]:
    return {
        "corpus_artifact_path": corpus_artifact_path,
        "fit_input_path": fit_input_path,
        "materialization_report_path": materialization_report_path,
        "expected_materialization_report_sha256": (
            expected_materialization_report_sha256
        ),
        "expected_materialization_report_file_sha256": (
            expected_materialization_report_file_sha256
        ),
        "factorial_report_path": factorial_report_path,
        "expected_factorial_report_sha256": (
            expected_factorial_report_sha256
        ),
        "expected_factorial_report_file_sha256": (
            expected_factorial_report_file_sha256
        ),
        "graph_candidate_path": graph_candidate_path,
        "basis_package_path": basis_package_path,
        "base_artifact_path": base_artifact_path,
        "refit_artifact_path": refit_artifact_path,
        "output": output,
        "cache_dir": cache_dir,
        "_diagnostic_recipe": recipe,
    }


def run_gemma_iterative_generator_innovation_v2_scale_diagnostic(
    *,
    expected_panel_receipt_sha256: str,
    expected_panel_receipt_file_sha256: str,
    expected_private_role_input_file_sha256: str,
    expected_v1_report_sha256: str,
    expected_v1_report_file_sha256: str,
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expected_plan_sha256: str = FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    expected_plan_file_sha256: str = (
        FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ),
    panel_receipt_path: Path | str = DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    private_role_input_path: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT
    ),
    v1_report_path: Path | str = _DEFAULT_V1_REPORT,
    prior_occupancy_panel_path: Path | str = DEFAULT_PRIOR_OCCUPANCY_PANEL,
    corpus_artifact_path: Path | str = DEFAULT_EXPANDED_FIT_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    materialization_report_path: Path | str = _DEFAULT_MATERIALIZATION_REPORT,
    factorial_report_path: Path | str = _DEFAULT_FACTORIAL_REPORT,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    scale_receipt_output: Path | str = DEFAULT_SCALE_RECEIPT_OUTPUT,
    output: Path | str = DEFAULT_SCALE_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run exactly 16 target-blind accepted-parent scale forwards."""

    paths = {
        "plan": Path(plan_path),
        "panel": Path(panel_receipt_path),
        "private": Path(private_role_input_path),
        "v1_report": Path(v1_report_path),
        "prior": Path(prior_occupancy_panel_path),
        "corpus": Path(corpus_artifact_path),
    }
    plan, receipt, _v1_report, _basis, prior = _authenticate_v1(
        plan_path=paths["plan"],
        expected_plan_sha256=expected_plan_sha256,
        expected_plan_file_sha256=expected_plan_file_sha256,
        panel_receipt_path=paths["panel"],
        expected_panel_receipt_sha256=expected_panel_receipt_sha256,
        expected_panel_receipt_file_sha256=(
            expected_panel_receipt_file_sha256
        ),
        private_role_input_path=paths["private"],
        expected_private_role_input_file_sha256=(
            expected_private_role_input_file_sha256
        ),
        v1_report_path=paths["v1_report"],
        expected_v1_report_sha256=expected_v1_report_sha256,
        expected_v1_report_file_sha256=expected_v1_report_file_sha256,
        corpus_artifact_path=paths["corpus"],
        prior_occupancy_panel_path=paths["prior"],
    )
    parent_lineage = _mapping(
        _mapping(plan["lineage"], label="v1 plan lineage")[
            "token_fisher_model_and_parent_lineage"
        ],
        label="v1 planned parent lineage",
    )
    base_parent_keys = set(parent_lineage)
    extra_lineage = {
        key: value for key, value in prior.items() if key not in base_parent_keys
    }
    recipe = _GemmaDevelopmentCollectionRecipe(
        collect=partial(
            _collect_scale,
            prior_lineage=prior,
        ),
        validate_report=(
            validate_gemma_iterative_generator_innovation_v2_scale_development_report
        ),
        publish_report=partial(
            _publish_scale_once,
            scale_receipt_destination=Path(scale_receipt_output),
        ),
        report_label="generator innovation v2 target-blind scale",
        expected_parent_lineage={
            str(key): str(value) for key, value in parent_lineage.items()
        },
        extra_lineage=extra_lineage,
        extra_immutable_inputs=(
            ("generator_v1_plan", paths["plan"], expected_plan_file_sha256),
            (
                "generator_v1_panel",
                paths["panel"],
                expected_panel_receipt_file_sha256,
            ),
            (
                "generator_v1_private",
                paths["private"],
                expected_private_role_input_file_sha256,
            ),
            (
                "generator_v1_report",
                paths["v1_report"],
                expected_v1_report_file_sha256,
            ),
            (
                "generator_v1_prior_panel",
                paths["prior"],
                receipt.prior_occupancy_panel_file_sha256,
            ),
        ),
        source_code_files=_SOURCE_CODE_FILES,
        collection_panel_factory=partial(
            _materialize_collection_panel,
            receipt=receipt,
            role_input_path=paths["private"],
        ),
    )
    report = run_gemma_iterative_residual_diagnostic(
        **_base_run_kwargs(
            corpus_artifact_path=paths["corpus"],
            fit_input_path=Path(fit_input_path),
            materialization_report_path=Path(materialization_report_path),
            expected_materialization_report_sha256=(
                expected_materialization_report_sha256
            ),
            expected_materialization_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
            factorial_report_path=Path(factorial_report_path),
            expected_factorial_report_sha256=(
                expected_factorial_report_sha256
            ),
            expected_factorial_report_file_sha256=(
                expected_factorial_report_file_sha256
            ),
            graph_candidate_path=Path(graph_candidate_path),
            basis_package_path=Path(basis_package_path),
            base_artifact_path=Path(base_artifact_path),
            refit_artifact_path=Path(refit_artifact_path),
            output=Path(output),
            cache_dir=None if cache_dir is None else Path(cache_dir),
            recipe=recipe,
        )
    )
    validate_gemma_iterative_generator_innovation_v2_scale_development_report(
        report
    )
    return dict(report)


def run_gemma_iterative_generator_innovation_v2_candidate_plan_preparation(
    *,
    scale_receipt_path: Path | str = DEFAULT_SCALE_RECEIPT_OUTPUT,
    expected_scale_receipt_sha256: str,
    expected_scale_receipt_file_sha256: str,
    scale_development_report_path: Path | str = DEFAULT_SCALE_OUTPUT,
    expected_scale_development_report_sha256: str,
    expected_scale_development_report_file_sha256: str,
    v1_plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expected_v1_plan_sha256: str = FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    expected_v1_plan_file_sha256: str = (
        FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ),
    v1_report_path: Path | str = _DEFAULT_V1_REPORT,
    expected_v1_report_sha256: str,
    expected_v1_report_file_sha256: str,
    v1_panel_receipt_path: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT
    ),
    expected_v1_panel_receipt_sha256: str,
    expected_v1_panel_receipt_file_sha256: str,
    output: Path | str = DEFAULT_CANDIDATE_PLAN_OUTPUT,
) -> dict[str, object]:
    """Freeze the candidate plan without opening the model or target data."""

    scale_path = Path(scale_receipt_path)
    scale_report_path = Path(scale_development_report_path)
    plan_path = Path(v1_plan_path)
    report_path = Path(v1_report_path)
    panel_path = Path(v1_panel_receipt_path)
    scale = _load_mapping(scale_path, label="standalone scale receipt")
    scale_report = _load_mapping(
        scale_report_path,
        label="scale development report",
    )
    v1_plan = _load_mapping(plan_path, label="v1 plan")
    v1_report = _load_mapping(report_path, label="v1 development report")
    v1_panel = _load_mapping(panel_path, label="v1 panel receipt")
    validate_gemma_iterative_generator_innovation_v2_scale_receipt(scale)
    validate_gemma_iterative_generator_innovation_v2_scale_development_report(
        scale_report
    )
    if (
        scale.get("receipt_sha256") != expected_scale_receipt_sha256
        or _file_sha256(scale_path) != expected_scale_receipt_file_sha256
        or scale_report.get("report_sha256")
        != expected_scale_development_report_sha256
        or _file_sha256(scale_report_path)
        != expected_scale_development_report_file_sha256
        or not _canonical(scale_report["scale_receipt"])
        == _canonical(scale)
    ):
        raise ValueError("scale receipt/development report identity differs")
    candidate_plan = (
        build_gemma_iterative_generator_innovation_v2_candidate_plan(
            scale_receipt=scale,
            scale_receipt_file_sha256=expected_scale_receipt_file_sha256,
            v1_plan=v1_plan,
            v1_plan_file_sha256=expected_v1_plan_file_sha256,
            v1_development_report=v1_report,
            v1_development_report_file_sha256=(
                expected_v1_report_file_sha256
            ),
            v1_panel_receipt=v1_panel,
            v1_panel_receipt_file_sha256=(
                expected_v1_panel_receipt_file_sha256
            ),
        )
    )
    if (
        v1_plan.get("plan_sha256") != expected_v1_plan_sha256
        or _file_sha256(plan_path) != expected_v1_plan_file_sha256
        or v1_report.get("report_sha256") != expected_v1_report_sha256
        or _file_sha256(report_path) != expected_v1_report_file_sha256
        or v1_panel.get("receipt_sha256")
        != expected_v1_panel_receipt_sha256
        or _file_sha256(panel_path) != expected_v1_panel_receipt_file_sha256
    ):
        raise ValueError("v1 candidate-plan source identity differs")
    replay = replay_gemma_iterative_generator_innovation_v2_candidate_plan(
        scale_receipt=scale,
        scale_receipt_file_sha256=expected_scale_receipt_file_sha256,
        v1_plan=v1_plan,
        v1_plan_file_sha256=expected_v1_plan_file_sha256,
        v1_development_report=v1_report,
        v1_development_report_file_sha256=expected_v1_report_file_sha256,
        v1_panel_receipt=v1_panel,
        v1_panel_receipt_file_sha256=(
            expected_v1_panel_receipt_file_sha256
        ),
        expected_plan=candidate_plan,
    )
    if _canonical(replay) != _canonical(candidate_plan):
        raise RuntimeError("v2 candidate-plan replay differs")
    _write_once(Path(output), candidate_plan)
    return candidate_plan


def run_gemma_iterative_generator_innovation_v2_target_diagnostic(
    *,
    expected_candidate_plan_sha256: str,
    expected_candidate_plan_file_sha256: str,
    expected_scale_receipt_sha256: str,
    expected_scale_receipt_file_sha256: str,
    expected_scale_development_report_sha256: str,
    expected_scale_development_report_file_sha256: str,
    expected_panel_receipt_sha256: str,
    expected_panel_receipt_file_sha256: str,
    expected_private_role_input_file_sha256: str,
    expected_v1_report_sha256: str,
    expected_v1_report_file_sha256: str,
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    candidate_plan_path: Path | str = DEFAULT_CANDIDATE_PLAN_OUTPUT,
    scale_receipt_path: Path | str = DEFAULT_SCALE_RECEIPT_OUTPUT,
    scale_development_report_path: Path | str = DEFAULT_SCALE_OUTPUT,
    v1_plan_path: Path | str = DEFAULT_GENERATOR_INNOVATION_PLAN,
    expected_v1_plan_sha256: str = FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    expected_v1_plan_file_sha256: str = (
        FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256
    ),
    panel_receipt_path: Path | str = DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    private_role_input_path: Path | str = (
        DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT
    ),
    v1_report_path: Path | str = _DEFAULT_V1_REPORT,
    prior_occupancy_panel_path: Path | str = DEFAULT_PRIOR_OCCUPANCY_PANEL,
    corpus_artifact_path: Path | str = DEFAULT_EXPANDED_FIT_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    materialization_report_path: Path | str = _DEFAULT_MATERIALIZATION_REPORT,
    factorial_report_path: Path | str = _DEFAULT_FACTORIAL_REPORT,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    output: Path | str = DEFAULT_TARGET_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run exact Q6 plus 13-R4 adaptive development on the frozen panel."""

    candidate_path = Path(candidate_plan_path)
    scale_path = Path(scale_receipt_path)
    scale_report_path = Path(scale_development_report_path)
    candidate_plan = _load_mapping(candidate_path, label="candidate plan")
    scale_receipt = _load_mapping(scale_path, label="scale receipt")
    scale_report = _load_mapping(
        scale_report_path,
        label="scale development report",
    )
    validate_gemma_iterative_generator_innovation_v2_candidate_plan(
        candidate_plan
    )
    validate_gemma_iterative_generator_innovation_v2_scale_receipt(
        scale_receipt
    )
    validate_gemma_iterative_generator_innovation_v2_scale_development_report(
        scale_report
    )
    if (
        candidate_plan.get("plan_sha256") != expected_candidate_plan_sha256
        or _file_sha256(candidate_path)
        != expected_candidate_plan_file_sha256
        or scale_receipt.get("receipt_sha256")
        != expected_scale_receipt_sha256
        or _file_sha256(scale_path) != expected_scale_receipt_file_sha256
        or scale_report.get("report_sha256")
        != expected_scale_development_report_sha256
        or _file_sha256(scale_report_path)
        != expected_scale_development_report_file_sha256
        or _canonical(scale_report["scale_receipt"])
        != _canonical(scale_receipt)
    ):
        raise ValueError("v2 candidate/scale artifact identity differs")
    paths = {
        "plan": Path(v1_plan_path),
        "panel": Path(panel_receipt_path),
        "private": Path(private_role_input_path),
        "v1_report": Path(v1_report_path),
        "prior": Path(prior_occupancy_panel_path),
        "corpus": Path(corpus_artifact_path),
    }
    v1_plan, receipt, _v1_report, basis, prior = _authenticate_v1(
        plan_path=paths["plan"],
        expected_plan_sha256=expected_v1_plan_sha256,
        expected_plan_file_sha256=expected_v1_plan_file_sha256,
        panel_receipt_path=paths["panel"],
        expected_panel_receipt_sha256=expected_panel_receipt_sha256,
        expected_panel_receipt_file_sha256=(
            expected_panel_receipt_file_sha256
        ),
        private_role_input_path=paths["private"],
        expected_private_role_input_file_sha256=(
            expected_private_role_input_file_sha256
        ),
        v1_report_path=paths["v1_report"],
        expected_v1_report_sha256=expected_v1_report_sha256,
        expected_v1_report_file_sha256=expected_v1_report_file_sha256,
        corpus_artifact_path=paths["corpus"],
        prior_occupancy_panel_path=paths["prior"],
    )
    if _mapping(candidate_plan["lineage"], label="candidate-plan lineage").get(
        "v2_scale_receipt_file_sha256"
    ) != expected_scale_receipt_file_sha256:
        raise ValueError("candidate plan does not bind standalone scale file")
    fixed_basis = basis["basis_matrix_source_coordinates_by_generator"]
    parent_lineage = _mapping(
        _mapping(v1_plan["lineage"], label="v1 plan lineage")[
            "token_fisher_model_and_parent_lineage"
        ],
        label="v1 planned parent lineage",
    )
    extra_lineage = {
        **{
            key: value
            for key, value in prior.items()
            if key not in set(parent_lineage)
        },
        "v2_scale_receipt_sha256": expected_scale_receipt_sha256,
        "v2_scale_receipt_file_sha256": expected_scale_receipt_file_sha256,
        "v2_scale_development_report_sha256": (
            expected_scale_development_report_sha256
        ),
        "v2_scale_development_report_file_sha256": (
            expected_scale_development_report_file_sha256
        ),
        "v2_candidate_plan_sha256": expected_candidate_plan_sha256,
        "v2_candidate_plan_file_sha256": (
            expected_candidate_plan_file_sha256
        ),
    }
    recipe = _GemmaDevelopmentCollectionRecipe(
        collect=partial(
            _collect_target,
            candidate_plan=candidate_plan,
            candidate_plan_file_sha256=(
                expected_candidate_plan_file_sha256
            ),
            scale_receipt_file_sha256=expected_scale_receipt_file_sha256,
            scale_development_report=scale_report,
            scale_development_report_file_sha256=(
                expected_scale_development_report_file_sha256
            ),
            fixed_basis=fixed_basis,
        ),
        validate_report=(
            validate_gemma_iterative_generator_innovation_v2_target_development_report
        ),
        publish_report=partial(
            _publish_target_once,
            candidate_plan=candidate_plan,
            candidate_plan_file_sha256=(
                expected_candidate_plan_file_sha256
            ),
            scale_receipt_file_sha256=expected_scale_receipt_file_sha256,
            scale_development_report=scale_report,
            scale_development_report_file_sha256=(
                expected_scale_development_report_file_sha256
            ),
        ),
        report_label="generator innovation v2 adaptive target development",
        expected_parent_lineage={
            str(key): str(value) for key, value in parent_lineage.items()
        },
        extra_lineage=extra_lineage,
        extra_immutable_inputs=(
            (
                "generator_v2_candidate_plan",
                candidate_path,
                expected_candidate_plan_file_sha256,
            ),
            (
                "generator_v2_scale_receipt",
                scale_path,
                expected_scale_receipt_file_sha256,
            ),
            (
                "generator_v2_scale_report",
                scale_report_path,
                expected_scale_development_report_file_sha256,
            ),
            (
                "generator_v1_plan",
                paths["plan"],
                expected_v1_plan_file_sha256,
            ),
            (
                "generator_v1_panel",
                paths["panel"],
                expected_panel_receipt_file_sha256,
            ),
            (
                "generator_v1_private",
                paths["private"],
                expected_private_role_input_file_sha256,
            ),
            (
                "generator_v1_report",
                paths["v1_report"],
                expected_v1_report_file_sha256,
            ),
            (
                "generator_v1_prior_panel",
                paths["prior"],
                receipt.prior_occupancy_panel_file_sha256,
            ),
        ),
        source_code_files=_SOURCE_CODE_FILES,
        collection_panel_factory=partial(
            _materialize_collection_panel,
            receipt=receipt,
            role_input_path=paths["private"],
        ),
    )
    report = run_gemma_iterative_residual_diagnostic(
        **_base_run_kwargs(
            corpus_artifact_path=paths["corpus"],
            fit_input_path=Path(fit_input_path),
            materialization_report_path=Path(materialization_report_path),
            expected_materialization_report_sha256=(
                expected_materialization_report_sha256
            ),
            expected_materialization_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
            factorial_report_path=Path(factorial_report_path),
            expected_factorial_report_sha256=(
                expected_factorial_report_sha256
            ),
            expected_factorial_report_file_sha256=(
                expected_factorial_report_file_sha256
            ),
            graph_candidate_path=Path(graph_candidate_path),
            basis_package_path=Path(basis_package_path),
            base_artifact_path=Path(base_artifact_path),
            refit_artifact_path=Path(refit_artifact_path),
            output=Path(output),
            cache_dir=None if cache_dir is None else Path(cache_dir),
            recipe=recipe,
        )
    )
    validate_gemma_iterative_generator_innovation_v2_target_development_report(
        report
    )
    return dict(report)


def _add_v1_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--v1-plan",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PLAN,
    )
    parser.add_argument(
        "--v1-plan-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    )
    parser.add_argument(
        "--v1-plan-file-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256,
    )
    parser.add_argument(
        "--v1-panel-receipt",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    )
    parser.add_argument("--v1-panel-receipt-sha256", required=True)
    parser.add_argument("--v1-panel-receipt-file-sha256", required=True)
    parser.add_argument(
        "--v1-private-role-input",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PRIVATE_OUTPUT,
    )
    parser.add_argument("--v1-private-role-input-file-sha256", required=True)
    parser.add_argument(
        "--v1-development-report",
        type=Path,
        default=_DEFAULT_V1_REPORT,
    )
    parser.add_argument("--v1-development-report-sha256", required=True)
    parser.add_argument(
        "--v1-development-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-occupancy-panel",
        type=Path,
        default=DEFAULT_PRIOR_OCCUPANCY_PANEL,
    )


def scale_parser() -> argparse.ArgumentParser:
    parser = build_residual_parser()
    parser.prog = "fisher-graph-gemma-generator-innovation-v2-scale"
    parser.description = (
        "Run the 16-forward target-blind generator-innovation v2 scale pass."
    )
    parser.set_defaults(output=DEFAULT_SCALE_OUTPUT)
    _add_v1_live_arguments(parser)
    parser.add_argument(
        "--scale-receipt-output",
        type=Path,
        default=DEFAULT_SCALE_RECEIPT_OUTPUT,
    )
    return parser


def scale_main(argv: Sequence[str] | None = None) -> int:
    args = scale_parser().parse_args(argv)
    report = run_gemma_iterative_generator_innovation_v2_scale_diagnostic(
        expected_panel_receipt_sha256=args.v1_panel_receipt_sha256,
        expected_panel_receipt_file_sha256=(
            args.v1_panel_receipt_file_sha256
        ),
        expected_private_role_input_file_sha256=(
            args.v1_private_role_input_file_sha256
        ),
        expected_v1_report_sha256=args.v1_development_report_sha256,
        expected_v1_report_file_sha256=(
            args.v1_development_report_file_sha256
        ),
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        expected_factorial_report_sha256=args.factorial_report_sha256,
        expected_factorial_report_file_sha256=(
            args.factorial_report_file_sha256
        ),
        plan_path=args.v1_plan,
        expected_plan_sha256=args.v1_plan_sha256,
        expected_plan_file_sha256=args.v1_plan_file_sha256,
        panel_receipt_path=args.v1_panel_receipt,
        private_role_input_path=args.v1_private_role_input,
        v1_report_path=args.v1_development_report,
        prior_occupancy_panel_path=args.prior_occupancy_panel,
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        materialization_report_path=args.materialization_report,
        factorial_report_path=args.factorial_report,
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        scale_receipt_output=args.scale_receipt_output,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def candidate_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fisher-graph-gemma-generator-innovation-v2-plan",
        description="Freeze the target-blind v2 candidate plan.",
    )
    parser.add_argument(
        "--scale-receipt",
        type=Path,
        default=DEFAULT_SCALE_RECEIPT_OUTPUT,
    )
    parser.add_argument("--scale-receipt-sha256", required=True)
    parser.add_argument("--scale-receipt-file-sha256", required=True)
    parser.add_argument(
        "--scale-development-report",
        type=Path,
        default=DEFAULT_SCALE_OUTPUT,
    )
    parser.add_argument("--scale-development-report-sha256", required=True)
    parser.add_argument(
        "--scale-development-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--v1-plan",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_PLAN,
    )
    parser.add_argument(
        "--v1-plan-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_SHA256,
    )
    parser.add_argument(
        "--v1-plan-file-sha256",
        default=FROZEN_GENERATOR_INNOVATION_PLAN_FILE_SHA256,
    )
    parser.add_argument(
        "--v1-development-report",
        type=Path,
        default=_DEFAULT_V1_REPORT,
    )
    parser.add_argument("--v1-development-report-sha256", required=True)
    parser.add_argument(
        "--v1-development-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--v1-panel-receipt",
        type=Path,
        default=DEFAULT_GENERATOR_INNOVATION_RECEIPT_OUTPUT,
    )
    parser.add_argument("--v1-panel-receipt-sha256", required=True)
    parser.add_argument("--v1-panel-receipt-file-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CANDIDATE_PLAN_OUTPUT,
    )
    return parser


def candidate_plan_main(argv: Sequence[str] | None = None) -> int:
    args = candidate_plan_parser().parse_args(argv)
    plan = (
        run_gemma_iterative_generator_innovation_v2_candidate_plan_preparation(
            scale_receipt_path=args.scale_receipt,
            expected_scale_receipt_sha256=args.scale_receipt_sha256,
            expected_scale_receipt_file_sha256=(
                args.scale_receipt_file_sha256
            ),
            scale_development_report_path=args.scale_development_report,
            expected_scale_development_report_sha256=(
                args.scale_development_report_sha256
            ),
            expected_scale_development_report_file_sha256=(
                args.scale_development_report_file_sha256
            ),
            v1_plan_path=args.v1_plan,
            expected_v1_plan_sha256=args.v1_plan_sha256,
            expected_v1_plan_file_sha256=args.v1_plan_file_sha256,
            v1_report_path=args.v1_development_report,
            expected_v1_report_sha256=(
                args.v1_development_report_sha256
            ),
            expected_v1_report_file_sha256=(
                args.v1_development_report_file_sha256
            ),
            v1_panel_receipt_path=args.v1_panel_receipt,
            expected_v1_panel_receipt_sha256=(
                args.v1_panel_receipt_sha256
            ),
            expected_v1_panel_receipt_file_sha256=(
                args.v1_panel_receipt_file_sha256
            ),
            output=args.output,
        )
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def target_parser() -> argparse.ArgumentParser:
    parser = build_residual_parser()
    parser.prog = "fisher-graph-gemma-generator-innovation-v2-target"
    parser.description = (
        "Run exact Q6 plus 13-R4 adaptive development after scale freeze."
    )
    parser.set_defaults(output=DEFAULT_TARGET_OUTPUT)
    _add_v1_live_arguments(parser)
    parser.add_argument(
        "--candidate-plan",
        type=Path,
        default=DEFAULT_CANDIDATE_PLAN_OUTPUT,
    )
    parser.add_argument("--candidate-plan-sha256", required=True)
    parser.add_argument("--candidate-plan-file-sha256", required=True)
    parser.add_argument(
        "--scale-receipt",
        type=Path,
        default=DEFAULT_SCALE_RECEIPT_OUTPUT,
    )
    parser.add_argument("--scale-receipt-sha256", required=True)
    parser.add_argument("--scale-receipt-file-sha256", required=True)
    parser.add_argument(
        "--scale-development-report",
        type=Path,
        default=DEFAULT_SCALE_OUTPUT,
    )
    parser.add_argument("--scale-development-report-sha256", required=True)
    parser.add_argument(
        "--scale-development-report-file-sha256",
        required=True,
    )
    return parser


def target_main(argv: Sequence[str] | None = None) -> int:
    args = target_parser().parse_args(argv)
    report = run_gemma_iterative_generator_innovation_v2_target_diagnostic(
        expected_candidate_plan_sha256=args.candidate_plan_sha256,
        expected_candidate_plan_file_sha256=args.candidate_plan_file_sha256,
        expected_scale_receipt_sha256=args.scale_receipt_sha256,
        expected_scale_receipt_file_sha256=args.scale_receipt_file_sha256,
        expected_scale_development_report_sha256=(
            args.scale_development_report_sha256
        ),
        expected_scale_development_report_file_sha256=(
            args.scale_development_report_file_sha256
        ),
        expected_panel_receipt_sha256=args.v1_panel_receipt_sha256,
        expected_panel_receipt_file_sha256=(
            args.v1_panel_receipt_file_sha256
        ),
        expected_private_role_input_file_sha256=(
            args.v1_private_role_input_file_sha256
        ),
        expected_v1_report_sha256=args.v1_development_report_sha256,
        expected_v1_report_file_sha256=(
            args.v1_development_report_file_sha256
        ),
        expected_materialization_report_sha256=(
            args.materialization_report_sha256
        ),
        expected_materialization_report_file_sha256=(
            args.materialization_report_file_sha256
        ),
        expected_factorial_report_sha256=args.factorial_report_sha256,
        expected_factorial_report_file_sha256=(
            args.factorial_report_file_sha256
        ),
        candidate_plan_path=args.candidate_plan,
        scale_receipt_path=args.scale_receipt,
        scale_development_report_path=args.scale_development_report,
        v1_plan_path=args.v1_plan,
        expected_v1_plan_sha256=args.v1_plan_sha256,
        expected_v1_plan_file_sha256=args.v1_plan_file_sha256,
        panel_receipt_path=args.v1_panel_receipt,
        private_role_input_path=args.v1_private_role_input,
        v1_report_path=args.v1_development_report,
        prior_occupancy_panel_path=args.prior_occupancy_panel,
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        materialization_report_path=args.materialization_report,
        factorial_report_path=args.factorial_report,
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else None
    if values is None:
        import sys

        values = sys.argv[1:]
    if not values or values[0] not in {"scale", "plan", "target"}:
        raise SystemExit("usage: ... {scale|plan|target} [arguments]")
    command, rest = values[0], values[1:]
    if command == "scale":
        return scale_main(rest)
    if command == "plan":
        return candidate_plan_main(rest)
    return target_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
