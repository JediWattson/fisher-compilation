"""Run the isolated A-only stable-split global-gated residual diagnostic.

Importing this module is standard-library-only.  The runner captures the pinned
A corpus exactly once, binds each runtime capture to a stable batch identity,
and uses fixed 36/9 outer plus nested 30/6 inner partitions.  Allocation is
selected only on the inner panels: while inner validation global-attention
relative Fisher loss exceeds 0.10, only global sites are eligible; after that
gate passes, any of the 18 sites may compete on the all-layer objective.  The
site sequence is frozen, replayed from a fresh parent fit on all 36 outer-fit
batches, and read once on outer A-dev.  Because all A evidence was previously
exposed, the final read is descriptive development evidence, not confirmation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile


DEFAULT_REPORT_OUTPUT = Path(
    "artifacts/gemma3-attention-transport-global-gated-residual-spectrum-a-diagnostic.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-global-gated-residual-spectrum-a-diagnostic-temp"
)
REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport."
    "global_gated_residual_spectrum_a_diagnostic.v1"
)
_REPORT_DOMAIN = b"fisher_graph:global-gated-residual-spectrum-a-diagnostic:v1\0"
_PROFILE_DOMAIN = b"fisher_graph:global-gated-residual-spectrum-frozen-profile:v1\0"
_CANDIDATE_ROUNDS_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-candidate-rounds:v1\0"
)
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_PINNED_MODEL_ID = "google/gemma-3-270m"
_PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_PINNED_A_CORPUS_FILE_SHA256 = (
    "112023d40793f1ff1c9952cc8b44e7d02b1f24a3473a72c941c4a494c2d6d0bf"
)
_PINNED_A_ROLE_PANEL_BINDING_SHA256 = (
    "bc11c01c66d25daffc96fa69778c2ab34e8d482bcec1de4b9d6bf1b6235b1432"
)
_PINNED_A_BATCH_PLAN_SHA256 = (
    "7983848a72046f598c0e2a439b7fb3e97802f7de95b48af7eddf178a5dd215b1"
)
_PINNED_A_OUTER_SPLIT_SHA256 = (
    "70fe80a40639a9312253b66064afb942683ffc41327d32f29b1a317b483d90c9"
)
_PINNED_A_INNER_SPLIT_SHA256 = (
    "10896b3a92a30d624b7c3647a9d3b3950aa2711f08153aaaeb90118dac66d9a8"
)
_RUNTIME_CAPTURE_BINDING_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-a-runtime-capture-binding:v1\0"
)
_OUTER_FIT_RECEIPT_COUNT = 36
_OUTER_DEV_RECEIPT_COUNT = 9
_DIRECTION_TRAIN_RECEIPT_COUNT = 30
_ALLOCATION_VALIDATION_RECEIPT_COUNT = 6
_ROWS_PER_RECEIPT = 18 * 16
_OUTER_FIT_ROWS = _OUTER_FIT_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_OUTER_DEV_ROWS = _OUTER_DEV_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_DIRECTION_TRAIN_ROWS = _DIRECTION_TRAIN_RECEIPT_COUNT * _ROWS_PER_RECEIPT
_ALLOCATION_VALIDATION_ROWS = (
    _ALLOCATION_VALIDATION_RECEIPT_COUNT * _ROWS_PER_RECEIPT
)
_GLOBAL_DIRECTION_TRAIN_ROWS = _DIRECTION_TRAIN_RECEIPT_COUNT * 3 * 16
_GLOBAL_ALLOCATION_VALIDATION_ROWS = (
    _ALLOCATION_VALIDATION_RECEIPT_COUNT * 3 * 16
)
_ALLOCATION_BUDGET = 2
_GLOBAL_RELATIVE_FISHER_THRESHOLD = 0.10
_RESIDUAL_COMPONENT_PARAMETERS = 10_240
_MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS = (
    _ALLOCATION_BUDGET * _RESIDUAL_COMPONENT_PARAMETERS
)
_NUMERIC_INVOCATION_IDS = tuple(f"mutation.{2 * index + 1}" for index in range(18))
_INVOCATION_IDS = tuple(sorted(_NUMERIC_INVOCATION_IDS))
_GLOBAL_ATTENTION_INVOCATION_IDS = ("mutation.11", "mutation.23", "mutation.35")

_PREDECESSOR_REPORT_PATH = Path(
    "artifacts/gemma3-attention-transport-all-layer-residual-spectrum-a-diagnostic.json"
)
_PREDECESSOR_REPORT_FILE_SHA256 = (
    "2a00d94ee73a8ea831e2df367dd62c1816da4f81d90564875b0129be85c01275"
)
_PREDECESSOR_REPORT_SHA256 = (
    "cda9b644d7c9a3ed7ed0d59f24da36031b7fffa58a2ea43393f6279ffd57f95d"
)
_PREDECESSOR_CORE_SOURCE_PATH = (
    "src/fisher_graph/gemma3_attention_transport_all_layer_residual_spectrum.py"
)
_PREDECESSOR_RUNNER_SOURCE_PATH = (
    "src/fisher_graph/"
    "gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic.py"
)
_PREDECESSOR_CORE_SOURCE_SHA256 = (
    "e0439489d166a6b06ad6b923d9d10258b5645905ccfd4f9cdfc1dd0db60c1fbd"
)
_PREDECESSOR_RUNNER_SOURCE_SHA256 = (
    "8f5cd8826cccba6f30e0a637c703c34a66920b8e3995d176b9f35bb35e0b7b0d"
)

_NEW_CORE_SOURCE_PATH = (
    "src/fisher_graph/gemma3_attention_transport_global_gated_residual_spectrum.py"
)
_PARTITION_SOURCE_PATH = (
    "src/fisher_graph/"
    "gemma3_attention_transport_global_gated_residual_spectrum_partition.py"
)
_RUNNER_SOURCE_PATH = (
    "src/fisher_graph/"
    "gemma3_attention_transport_global_gated_residual_spectrum_a_diagnostic.py"
)
_SOURCE_PATHS = (
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_experiment.py",
    "src/fisher_graph/gemma3_attention_transport_experiment_core.py",
    "src/fisher_graph/gemma3_attention_transport_sparse_v2.py",
    "src/fisher_graph/gemma3_attention_transport_hierarchical_edge.py",
    "src/fisher_graph/gemma3_attention_transport_global_rank1.py",
    "src/fisher_graph/gemma3_attention_transport_global_site_rank1.py",
    "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2.py",
    "src/fisher_graph/gemma3_attention_transport_targeted_site_rank2_a_diagnostic.py",
    _PREDECESSOR_CORE_SOURCE_PATH,
    _PREDECESSOR_RUNNER_SOURCE_PATH,
    _NEW_CORE_SOURCE_PATH,
    _PARTITION_SOURCE_PATH,
    _RUNNER_SOURCE_PATH,
    "src/fisher_graph/gemma3_chart_native_neighbor_experiment.py",
    "src/fisher_graph/gemma3_chart_native_neighbor_fresh_panels.py",
    "pyproject.toml",
)
_TOP_LEVEL_FIELDS = {
    "schema",
    "format_version",
    "scientific_status",
    "exposure_binding",
    "model",
    "protocol",
    "A_partition",
    "selection",
    "candidate",
    "outer_A_dev_diagnostic",
    "runtime_counters",
    "runtime_resources",
    "report_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object, *, minimum: float | None = None) -> bool:
    return (
        type(value) is float
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def _close(left: object, right: object) -> bool:
    return (
        _finite_float(left)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=1.0e-11, abs_tol=1.0e-12)
    )


def _sealed_runner() -> object:
    from fisher_graph import (
        gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic,
    )

    return gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic


def _read_no_follow_regular_bytes(path: Path) -> bytes:
    return _sealed_runner()._read_no_follow_regular_bytes(path)


def _require_real_directory(path: Path) -> None:
    _sealed_runner()._require_real_directory(path)


def _entry_exists(path: Path) -> bool:
    return bool(_sealed_runner()._entry_exists(path))


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _unlink_if_owned(path: Path, owner: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if (metadata.st_dev, metadata.st_ino) != owner:
        return False
    path.unlink()
    return True


def _strict_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _strict_json_loads(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )


def _source_sha256s(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_PATHS:
        try:
            raw = _read_no_follow_regular_bytes(repository_root / relative)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"global-gated source is not stable: {relative}") from error
        result[relative] = hashlib.sha256(raw).hexdigest()
    return result


def _validate_predecessor(repository_root: Path) -> Mapping[str, object]:
    for relative, expected in (
        (_PREDECESSOR_CORE_SOURCE_PATH, _PREDECESSOR_CORE_SOURCE_SHA256),
        (_PREDECESSOR_RUNNER_SOURCE_PATH, _PREDECESSOR_RUNNER_SOURCE_SHA256),
    ):
        observed = hashlib.sha256(
            _read_no_follow_regular_bytes(repository_root / relative)
        ).hexdigest()
        if observed != expected:
            raise ValueError(f"sealed all-layer predecessor drifted: {relative}")
    path = repository_root / _PREDECESSOR_REPORT_PATH
    raw = _read_no_follow_regular_bytes(path)
    if hashlib.sha256(raw).hexdigest() != _PREDECESSOR_REPORT_FILE_SHA256:
        raise ValueError("sealed all-layer predecessor report file drifted")
    sealed = _sealed_runner()
    report = sealed._strict_report(path)
    sealed.check_all_layer_residual_spectrum_A_diagnostic(
        path,
        repository_root=repository_root,
    )
    status = report.get("scientific_status")
    summary = report.get("outer_A_dev_diagnostic", {}).get("summary")
    if (
        report.get("report_sha256") != _PREDECESSOR_REPORT_SHA256
        or not isinstance(status, Mapping)
        or status.get("A_only_development_diagnostic") is not True
        or status.get("B_opened_by_this_run") is not False
        or status.get("C_ever_opened") is not False
        or status.get("confirmation_claimed") is not False
        or status.get("deployment_router_built") is not False
        or report.get("candidate", {}).get("allocation_profile")
        != ["mutation.15", "mutation.5"]
        or not isinstance(summary, Mapping)
        or not _close(
            float(summary.get("global_parent_minus_allocated_relative_fisher_loss")),
            0.0,
        )
    ):
        raise ValueError("sealed all-layer predecessor lineage drifted")
    return report


def _profile_artifact_sha256(
    sequence: Sequence[str],
    allocated_rank_profile: Sequence[object],
    total_rank_profile: Sequence[object],
    candidate_rounds_sha256: str,
) -> str:
    return hashlib.sha256(
        _PROFILE_DOMAIN
        + _canonical_json_bytes(
            {
                "allocated_rank_profile": list(allocated_rank_profile),
                "candidate_rounds_artifact_sha256": candidate_rounds_sha256,
                "frozen_allocation_sequence": list(sequence),
                "total_rank_profile": list(total_rank_profile),
            }
        )
    ).hexdigest()


def _candidate_rounds_artifact_sha256(
    candidate_rounds: Sequence[Mapping[str, object]],
) -> str:
    return hashlib.sha256(
        _CANDIDATE_ROUNDS_DOMAIN + _canonical_json_bytes(list(candidate_rounds))
    ).hexdigest()


def _metric_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._sparse_metric_payload(value)


def _fit_metric_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._fit_metric_payload(value)


def _bank_summary_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._bank_summary_payload(value)


def _marginal_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._marginal_payload(value)


def _first_spectrum_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._first_spectrum_payload(value)


def _parent_fit_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._site_rank_one_parent_fit_payload(value)


def _outer_diagnostic_payload(value: object) -> dict[str, object]:
    return _sealed_runner()._outer_diagnostic_payload(
        value,
        current_run_outer_panel_previously_exposed_exactly=False,
    )


def _allocation_round_payload(value: object) -> dict[str, object]:
    current_sha = str(getattr(value, "current_bank_sha256"))
    objective_mode = str(getattr(value, "objective_mode"))
    global_gains = dict(getattr(value, "global_normalized_fisher_mass_gains"))
    all_gains = dict(getattr(value, "all_layer_normalized_fisher_mass_gains"))
    marginals = {
        str(getattr(item, "invocation_id")): item
        for item in getattr(value, "marginals")
    }
    if (
        objective_mode not in ("global_attention", "all_layers")
        or set(global_gains) != set(_NUMERIC_INVOCATION_IDS)
        or set(all_gains) != set(_NUMERIC_INVOCATION_IDS)
        or set(marginals) != set(_NUMERIC_INVOCATION_IDS)
    ):
        raise RuntimeError("global-gated allocation-round catalog drifted")
    floor = float(getattr(value, "minimum_relative_fisher_gain"))
    selected = getattr(value, "selected_invocation_id")
    candidates = []
    for invocation_id in _NUMERIC_INVOCATION_IDS:
        marginal = marginals[invocation_id]
        global_gain = float(global_gains[invocation_id])
        all_gain = float(all_gains[invocation_id])
        objective_gain = global_gain if objective_mode == "global_attention" else all_gain
        in_scope = (
            invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
            if objective_mode == "global_attention"
            else True
        )
        strictly = bool(getattr(marginal, "strictly_improves"))
        source_sha = str(getattr(marginal, "source_bank_sha256"))
        candidates.append(
            {
                "invocation_id": invocation_id,
                "attention_kind": (
                    "global"
                    if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                    else "local"
                ),
                "source_bank_sha256": source_sha,
                "used_cached_off_site_marginal": source_sha != current_sha,
                "strictly_improves_all_protocol_gates": strictly,
                "in_objective_scope": in_scope,
                "global_normalized_fisher_mass_gain": global_gain,
                "all_layer_normalized_fisher_mass_gain": all_gain,
                "selection_objective_normalized_fisher_mass_gain": objective_gain,
                "eligible_for_selection": strictly and in_scope and objective_gain > floor,
                "marginal": _marginal_payload(marginal),
            }
        )
    return {
        "allocation_index": int(getattr(value, "allocation_index")),
        "current_bank_sha256": current_sha,
        "objective_mode": objective_mode,
        "global_gate_active": objective_mode == "global_attention",
        "global_relative_fisher_threshold": float(
            getattr(value, "global_relative_fisher_threshold")
        ),
        "global_relative_fisher_loss_before": float(
            getattr(value, "global_relative_fisher_loss_before")
        ),
        "all_layer_relative_fisher_loss_before": float(
            getattr(value, "all_layer_relative_fisher_loss_before")
        ),
        "global_evaluation_row_count": int(
            getattr(value, "global_evaluation_row_count")
        ),
        "global_evaluation_zero_predictor_fisher_loss": float(
            getattr(value, "global_evaluation_zero_predictor_fisher_loss")
        ),
        "all_layer_evaluation_row_count": int(
            getattr(value, "all_layer_evaluation_row_count")
        ),
        "all_layer_evaluation_zero_predictor_fisher_loss": float(
            getattr(value, "all_layer_evaluation_zero_predictor_fisher_loss")
        ),
        "minimum_relative_fisher_gain": floor,
        "tie_tolerance": float(getattr(value, "tie_tolerance")),
        "candidates": candidates,
        "selected_invocation_id": None if selected is None else str(selected),
        "early_stop_no_eligible_candidate": selected is None,
    }


def _allocation_step_payload(value: object) -> dict[str, object]:
    return {
        "allocation_index": int(getattr(value, "allocation_index")),
        "invocation_id": str(getattr(value, "invocation_id")),
        "objective_mode": str(getattr(value, "objective_mode")),
        "source_bank_sha256": str(getattr(value, "source_bank_sha256")),
        "result_bank_sha256": str(getattr(value, "result_bank_sha256")),
        "marginal_source_bank_sha256": str(
            getattr(value, "marginal_source_bank_sha256")
        ),
        "used_cached_off_site_marginal": bool(
            getattr(value, "used_cached_off_site_marginal")
        ),
        "selection_objective_normalized_fisher_gain": float(
            getattr(value, "selection_objective_normalized_fisher_gain")
        ),
        "selection_global_normalized_fisher_gain": float(
            getattr(value, "selection_global_normalized_fisher_gain")
        ),
        "selection_all_layer_normalized_fisher_gain": float(
            getattr(value, "selection_all_layer_normalized_fisher_gain")
        ),
        "marginal": _marginal_payload(getattr(value, "marginal")),
        "direction_train_before": _fit_metric_payload(
            getattr(value, "proposal_fit_before")
        ),
        "direction_train_after": _fit_metric_payload(
            getattr(value, "proposal_fit_after")
        ),
        "allocation_validation_before": _fit_metric_payload(
            getattr(value, "evaluation_before")
        ),
        "allocation_validation_after": _fit_metric_payload(
            getattr(value, "evaluation_after")
        ),
        "global_allocation_validation_before": _fit_metric_payload(
            getattr(value, "global_evaluation_before")
        ),
        "global_allocation_validation_after": _fit_metric_payload(
            getattr(value, "global_evaluation_after")
        ),
    }


def _global_gated_allocation_payload(value: object) -> dict[str, object]:
    final_bank = getattr(value, "final_bank")
    bank_summary = _bank_summary_payload(final_bank)
    sequence = [str(item) for item in getattr(value, "allocation_profile")]
    rounds = [_allocation_round_payload(item) for item in getattr(value, "candidate_rounds")]
    rounds_sha = _candidate_rounds_artifact_sha256(rounds)
    profile_sha = _profile_artifact_sha256(
        sequence,
        bank_summary["allocated_rank_profile"],
        bank_summary["total_rank_profile"],
        rounds_sha,
    )
    return {
        "initial_bank_sha256": str(getattr(value, "initial_bank_sha256")),
        "final_bank": bank_summary,
        "allocation_budget": int(getattr(value, "allocation_budget")),
        "budget_exhausted": bool(getattr(value, "budget_exhausted")),
        "global_relative_fisher_threshold": float(
            getattr(value, "global_relative_fisher_threshold")
        ),
        "global_gate_passed": bool(getattr(value, "global_gate_passed")),
        "frozen_allocation_sequence": sequence,
        "frozen_allocated_rank_profile": bank_summary["allocated_rank_profile"],
        "frozen_total_rank_profile": bank_summary["total_rank_profile"],
        "frozen_profile_artifact_sha256": profile_sha,
        "first_spectrum": _first_spectrum_payload(getattr(value, "first_spectrum")),
        "steps": [_allocation_step_payload(item) for item in getattr(value, "steps")],
        "candidate_rounds": rounds,
        "candidate_rounds_artifact_sha256": rounds_sha,
        "initial_direction_train_oracle": _fit_metric_payload(
            getattr(value, "initial_proposal_fit_oracle")
        ),
        "final_direction_train_oracle": _fit_metric_payload(
            getattr(value, "final_proposal_fit_oracle")
        ),
        "initial_allocation_validation_oracle": _fit_metric_payload(
            getattr(value, "initial_evaluation_oracle")
        ),
        "final_allocation_validation_oracle": _fit_metric_payload(
            getattr(value, "final_evaluation_oracle")
        ),
        "initial_global_allocation_validation_oracle": _fit_metric_payload(
            getattr(value, "initial_global_evaluation_oracle")
        ),
        "final_global_allocation_validation_oracle": _fit_metric_payload(
            getattr(value, "final_global_evaluation_oracle")
        ),
        "direction_train_relative_fisher_loss_history": [
            float(item)
            for item in getattr(value, "proposal_fit_relative_fisher_loss_history")
        ],
        "allocation_validation_relative_fisher_loss_history": [
            float(item)
            for item in getattr(value, "evaluation_relative_fisher_loss_history")
        ],
        "global_allocation_validation_relative_fisher_loss_history": [
            float(item)
            for item in getattr(value, "global_evaluation_relative_fisher_loss_history")
        ],
    }


def _replay_step_payload(value: object) -> dict[str, object]:
    return {
        "allocation_index": int(getattr(value, "allocation_index")),
        "invocation_id": str(getattr(value, "invocation_id")),
        "source_bank_sha256": str(getattr(value, "source_bank_sha256")),
        "result_bank_sha256": str(getattr(value, "result_bank_sha256")),
        "marginal_source_bank_sha256": str(
            getattr(value, "marginal_source_bank_sha256")
        ),
        "used_cached_off_site_marginal": bool(
            getattr(value, "used_cached_off_site_marginal")
        ),
        "all_layer_relative_fisher_gain": float(
            getattr(value, "selection_all_layer_normalized_fisher_gain")
        ),
        "marginal": _marginal_payload(getattr(value, "marginal")),
        "full_fit_before": _fit_metric_payload(getattr(value, "proposal_fit_before")),
        "full_fit_after": _fit_metric_payload(getattr(value, "proposal_fit_after")),
    }


def _fixed_profile_replay_payload(
    value: object,
    *,
    frozen_profile_artifact_sha256: str,
    selection_candidate_rounds_artifact_sha256: str,
) -> dict[str, object]:
    final_bank = getattr(value, "final_bank")
    summary = _bank_summary_payload(final_bank)
    sequence = [str(item) for item in getattr(value, "allocation_profile")]
    replay_profile_sha = _profile_artifact_sha256(
        sequence,
        summary["allocated_rank_profile"],
        summary["total_rank_profile"],
        selection_candidate_rounds_artifact_sha256,
    )
    return {
        "initial_bank_sha256": str(getattr(value, "initial_bank_sha256")),
        "final_bank": summary,
        "allocation_profile": sequence,
        "allocated_component_count": int(getattr(value, "allocated_component_count")),
        "selection_candidate_rounds_artifact_sha256": (
            selection_candidate_rounds_artifact_sha256
        ),
        "frozen_profile_artifact_sha256": frozen_profile_artifact_sha256,
        "replay_profile_artifact_sha256": replay_profile_sha,
        "replay_profile_matches_frozen_selection": (
            replay_profile_sha == frozen_profile_artifact_sha256
        ),
        "steps": [_replay_step_payload(item) for item in getattr(value, "steps")],
        "initial_full_fit_oracle": _fit_metric_payload(
            getattr(value, "initial_fit_oracle")
        ),
        "final_full_fit_oracle": _fit_metric_payload(
            getattr(value, "final_fit_oracle")
        ),
        "full_fit_relative_fisher_loss_history": [
            float(item) for item in getattr(value, "fit_relative_fisher_loss_history")
        ],
    }


def _partition_payload(value: object) -> dict[str, object]:
    catalog = getattr(value, "catalog")
    capture_batches = tuple(getattr(value, "capture_batches"))
    batches = [
        {
            "batch_ordinal": int(getattr(batch, "batch_ordinal")),
            "batch_identity_sha256": str(getattr(batch, "batch_identity_sha256")),
            "ordered_context_fingerprint_sha256s": [
                str(item)
                for item in getattr(batch, "ordered_context_fingerprint_sha256s")
            ],
            "ordered_objective_row_stream_sha256s": [
                str(item)
                for item in getattr(batch, "ordered_objective_row_stream_sha256s")
            ],
        }
        for batch in getattr(catalog, "batches")
    ]
    runtime_batches = [
        {
            "batch_ordinal": int(getattr(item.identity, "batch_ordinal")),
            "batch_identity_sha256": str(
                getattr(item.identity, "batch_identity_sha256")
            ),
            "execution_receipt_sha256": str(
                getattr(item, "execution_receipt_sha256")
            ),
        }
        for item in capture_batches
    ]
    receipt_by_ordinal = {
        int(item["batch_ordinal"]): str(item["execution_receipt_sha256"])
        for item in runtime_batches
    }

    def split(
        *,
        ordinals: Sequence[int],
        identities: Sequence[str],
        row_count: int,
    ) -> dict[str, object]:
        return {
            "batch_ordinals": [int(item) for item in ordinals],
            "batch_identity_sha256s": [str(item) for item in identities],
            "execution_receipt_sha256s": [
                receipt_by_ordinal[int(item)] for item in ordinals
            ],
            "row_count": row_count,
        }

    return {
        "stable_batch_catalog": {
            "role_panel_binding_sha256": str(
                getattr(catalog, "role_panel_binding_sha256")
            ),
            "batch_plan_artifact_sha256": str(
                getattr(catalog, "batch_plan_artifact_sha256")
            ),
            "batches": batches,
            "contains_prompt_text": False,
            "contains_token_ids": False,
        },
        "runtime_capture_binding": {
            "runtime_capture_binding_sha256": str(
                getattr(value, "runtime_capture_binding_sha256")
            ),
            "batches": runtime_batches,
            "execution_receipts_drive_partition": False,
        },
        "outer": {
            "outer_split_artifact_sha256": str(
                getattr(value, "outer_split_artifact_sha256")
            ),
            "split_identity_stable_across_execution_receipt_drift": True,
            "outer_fit": split(
                ordinals=getattr(value, "outer_fit_batch_ordinals"),
                identities=getattr(value, "outer_fit_batch_identity_sha256s"),
                row_count=_OUTER_FIT_ROWS,
            ),
            "outer_dev": split(
                ordinals=getattr(value, "outer_dev_batch_ordinals"),
                identities=getattr(value, "outer_dev_batch_identity_sha256s"),
                row_count=_OUTER_DEV_ROWS,
            ),
        },
        "inner": {
            "inner_split_artifact_sha256": str(
                getattr(value, "inner_split_artifact_sha256")
            ),
            "direction_train": split(
                ordinals=getattr(value, "direction_train_batch_ordinals"),
                identities=getattr(value, "direction_train_batch_identity_sha256s"),
                row_count=_DIRECTION_TRAIN_ROWS,
            ),
            "allocation_validation": split(
                ordinals=getattr(value, "allocation_validation_batch_ordinals"),
                identities=getattr(
                    value,
                    "allocation_validation_batch_identity_sha256s",
                ),
                row_count=_ALLOCATION_VALIDATION_ROWS,
            ),
            "inner_is_strict_partition_of_outer_fit": True,
            "outer_dev_excluded_from_inner_partition": True,
        },
    }


def _runtime_capture_binding_sha256(
    *,
    batch_plan_artifact_sha256: str,
    batches: Sequence[Mapping[str, object]],
) -> str:
    return hashlib.sha256(
        _RUNTIME_CAPTURE_BINDING_DOMAIN
        + _canonical_json_bytes(
            {
                "batch_plan_artifact_sha256": batch_plan_artifact_sha256,
                "batches": [dict(item) for item in batches],
            }
        )
    ).hexdigest()


def _validate_partition(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "stable_batch_catalog",
        "runtime_capture_binding",
        "outer",
        "inner",
        "final_firewall_state",
    }:
        raise ValueError("global-gated A partition schema drifted")
    catalog_payload = value["stable_batch_catalog"]
    runtime_payload = value["runtime_capture_binding"]
    outer = value["outer"]
    inner = value["inner"]
    if (
        not isinstance(catalog_payload, Mapping)
        or set(catalog_payload)
        != {
            "role_panel_binding_sha256",
            "batch_plan_artifact_sha256",
            "batches",
            "contains_prompt_text",
            "contains_token_ids",
        }
        or not isinstance(runtime_payload, Mapping)
        or set(runtime_payload)
        != {
            "runtime_capture_binding_sha256",
            "batches",
            "execution_receipts_drive_partition",
        }
        or not isinstance(outer, Mapping)
        or set(outer)
        != {
            "outer_split_artifact_sha256",
            "split_identity_stable_across_execution_receipt_drift",
            "outer_fit",
            "outer_dev",
        }
        or not isinstance(inner, Mapping)
        or set(inner)
        != {
            "inner_split_artifact_sha256",
            "direction_train",
            "allocation_validation",
            "inner_is_strict_partition_of_outer_fit",
            "outer_dev_excluded_from_inner_partition",
        }
    ):
        raise ValueError("global-gated A partition section drifted")
    from fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum_partition import (
        ALLOCATION_VALIDATION_BATCH_ORDINALS,
        DIRECTION_TRAIN_BATCH_ORDINALS,
        OUTER_DEV_BATCH_ORDINALS,
        OUTER_FIT_BATCH_ORDINALS,
        StableABatchCatalog,
        StableABatchIdentity,
        inner_split_artifact_sha256,
        outer_split_artifact_sha256,
    )

    raw_batches = catalog_payload["batches"]
    if not isinstance(raw_batches, list) or len(raw_batches) != 45:
        raise ValueError("global-gated stable batch catalog count drifted")
    batches = []
    for ordinal, item in enumerate(raw_batches):
        if not isinstance(item, Mapping) or set(item) != {
            "batch_ordinal",
            "batch_identity_sha256",
            "ordered_context_fingerprint_sha256s",
            "ordered_objective_row_stream_sha256s",
        }:
            raise ValueError("global-gated stable batch identity schema drifted")
        if item["batch_ordinal"] != ordinal:
            raise ValueError("global-gated stable batch ordinal drifted")
        batches.append(
            StableABatchIdentity(
                batch_ordinal=ordinal,
                batch_identity_sha256=str(item["batch_identity_sha256"]),
                ordered_context_fingerprint_sha256s=tuple(
                    item["ordered_context_fingerprint_sha256s"]
                ),
                ordered_objective_row_stream_sha256s=tuple(
                    item["ordered_objective_row_stream_sha256s"]
                ),
            )
        )
    catalog = StableABatchCatalog(
        role_panel_binding_sha256=str(
            catalog_payload["role_panel_binding_sha256"]
        ),
        batch_plan_artifact_sha256=str(
            catalog_payload["batch_plan_artifact_sha256"]
        ),
        batches=tuple(batches),
    )
    if (
        catalog.role_panel_binding_sha256 != _PINNED_A_ROLE_PANEL_BINDING_SHA256
        or catalog.batch_plan_artifact_sha256 != _PINNED_A_BATCH_PLAN_SHA256
        or outer_split_artifact_sha256(catalog) != _PINNED_A_OUTER_SPLIT_SHA256
        or inner_split_artifact_sha256(catalog) != _PINNED_A_INNER_SPLIT_SHA256
        or outer["outer_split_artifact_sha256"] != _PINNED_A_OUTER_SPLIT_SHA256
        or inner["inner_split_artifact_sha256"] != _PINNED_A_INNER_SPLIT_SHA256
        or catalog_payload["contains_prompt_text"] is not False
        or catalog_payload["contains_token_ids"] is not False
        or runtime_payload["execution_receipts_drive_partition"] is not False
        or outer["split_identity_stable_across_execution_receipt_drift"] is not True
        or inner["inner_is_strict_partition_of_outer_fit"] is not True
        or inner["outer_dev_excluded_from_inner_partition"] is not True
    ):
        raise ValueError("global-gated stable split binding drifted")
    runtime_batches = runtime_payload["batches"]
    if (
        not isinstance(runtime_batches, list)
        or len(runtime_batches) != 45
        or any(
            not isinstance(item, Mapping)
            or set(item)
            != {
                "batch_ordinal",
                "batch_identity_sha256",
                "execution_receipt_sha256",
            }
            for item in runtime_batches
        )
        or [item["batch_ordinal"] for item in runtime_batches] != list(range(45))
        or [item["batch_identity_sha256"] for item in runtime_batches]
        != [item.batch_identity_sha256 for item in catalog.batches]
        or any(not _is_sha256(item["execution_receipt_sha256"]) for item in runtime_batches)
        or len({item["execution_receipt_sha256"] for item in runtime_batches}) != 45
        or runtime_payload["runtime_capture_binding_sha256"]
        != _runtime_capture_binding_sha256(
            batch_plan_artifact_sha256=catalog.batch_plan_artifact_sha256,
            batches=runtime_batches,
        )
    ):
        raise ValueError("global-gated runtime capture binding drifted")
    receipt_by_ordinal = {
        int(item["batch_ordinal"]): item["execution_receipt_sha256"]
        for item in runtime_batches
    }
    identity_by_ordinal = {
        item.batch_ordinal: item.batch_identity_sha256 for item in catalog.batches
    }

    def validate_split(
        payload: object,
        *,
        ordinals: Sequence[int],
        row_count: int,
    ) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "batch_ordinals",
            "batch_identity_sha256s",
            "execution_receipt_sha256s",
            "row_count",
        }:
            raise ValueError("global-gated split payload schema drifted")
        if (
            payload["batch_ordinals"] != list(ordinals)
            or payload["batch_identity_sha256s"]
            != [identity_by_ordinal[item] for item in ordinals]
            or payload["execution_receipt_sha256s"]
            != [receipt_by_ordinal[item] for item in ordinals]
            or payload["row_count"] != row_count
        ):
            raise ValueError("global-gated split payload values drifted")

    validate_split(
        outer["outer_fit"],
        ordinals=OUTER_FIT_BATCH_ORDINALS,
        row_count=_OUTER_FIT_ROWS,
    )
    validate_split(
        outer["outer_dev"],
        ordinals=OUTER_DEV_BATCH_ORDINALS,
        row_count=_OUTER_DEV_ROWS,
    )
    validate_split(
        inner["direction_train"],
        ordinals=DIRECTION_TRAIN_BATCH_ORDINALS,
        row_count=_DIRECTION_TRAIN_ROWS,
    )
    validate_split(
        inner["allocation_validation"],
        ordinals=ALLOCATION_VALIDATION_BATCH_ORDINALS,
        row_count=_ALLOCATION_VALIDATION_ROWS,
    )
    if value["final_firewall_state"] != {
        "opened_roles": ["A_chart_transport_fit"],
        "current_role": "A_chart_transport_fit",
        "next_role": "A_chart_transport_fit",
        "freeze_receipt_count": 0,
        "B_opened_by_this_run": False,
        "C_ever_opened": False,
    }:
        raise ValueError("global-gated final firewall state drifted")
    return value


def _validate_fit_metric(
    value: object,
    *,
    expected_rows: int,
    expected_invocation_ids: Sequence[str],
    expected_ridge: float = 0.0,
) -> Mapping[str, object]:
    return _sealed_runner()._validate_fit_metric(
        value,
        expected_rows=expected_rows,
        expected_invocation_ids=expected_invocation_ids,
        expected_ridge=expected_ridge,
    )


def _validate_bank_summary(
    value: object,
    *,
    expected_parent_sha256: str,
    expected_sequence: Sequence[str],
) -> Mapping[str, object]:
    return _sealed_runner()._validate_bank_summary(
        value,
        expected_parent_sha256=expected_parent_sha256,
        expected_sequence=expected_sequence,
    )


def _site_metric_aggregate_closes(
    metrics_by_id: Mapping[str, Mapping[str, object]],
    *,
    invocation_ids: Sequence[str],
    aggregate: Mapping[str, object],
) -> bool:
    """Close a serialized aggregate against its row-weighted site metrics."""
    aggregate_fields = {
        "row_count",
        "fisher_loss",
        "zero_predictor_fisher_loss",
        "relative_fisher_loss",
    }
    # The production validator above requires these fields.  The guard keeps this
    # arithmetic helper usable with intentionally skeletal validator test doubles.
    if not aggregate_fields.issubset(aggregate):
        return True
    if any(invocation_id not in metrics_by_id for invocation_id in invocation_ids):
        return False
    row_count = sum(
        int(metrics_by_id[invocation_id]["row_count"])
        for invocation_id in invocation_ids
    )
    if row_count <= 0:
        return False
    fisher_loss = sum(
        int(metrics_by_id[invocation_id]["row_count"])
        * float(metrics_by_id[invocation_id]["fisher_loss"])
        for invocation_id in invocation_ids
    ) / row_count
    zero_predictor_fisher_loss = sum(
        int(metrics_by_id[invocation_id]["row_count"])
        * float(
            metrics_by_id[invocation_id]["zero_predictor_fisher_loss"]
        )
        for invocation_id in invocation_ids
    ) / row_count
    return (
        aggregate["row_count"] == row_count
        and zero_predictor_fisher_loss > 0.0
        and _close(aggregate["fisher_loss"], fisher_loss)
        and _close(
            aggregate["zero_predictor_fisher_loss"],
            zero_predictor_fisher_loss,
        )
        and _close(
            aggregate["relative_fisher_loss"],
            fisher_loss / zero_predictor_fisher_loss,
        )
    )


def _validate_marginal(
    value: object,
    *,
    expected_source_sha256: str | None = None,
    expected_parent_sha256: str | None = None,
    expected_proposal_sequence: Sequence[str] | None = None,
) -> Mapping[str, object]:
    return _sealed_runner()._validate_marginal(
        value,
        proposal_rows=_DIRECTION_TRAIN_RECEIPT_COUNT * 16,
        evaluation_rows=_ALLOCATION_VALIDATION_RECEIPT_COUNT * 16,
        expected_source_sha256=expected_source_sha256,
        expected_parent_sha256=expected_parent_sha256,
        expected_proposal_sequence=expected_proposal_sequence,
    )


def _mutation_ordinal(invocation_id: str) -> int:
    if invocation_id not in _NUMERIC_INVOCATION_IDS:
        raise ValueError("global-gated invocation ID is invalid")
    return int(invocation_id.split(".", 1)[1])


def _validate_global_gated_allocation(value: object) -> Mapping[str, object]:
    fields = {
        "initial_bank_sha256",
        "final_bank",
        "allocation_budget",
        "budget_exhausted",
        "global_relative_fisher_threshold",
        "global_gate_passed",
        "frozen_allocation_sequence",
        "frozen_allocated_rank_profile",
        "frozen_total_rank_profile",
        "frozen_profile_artifact_sha256",
        "first_spectrum",
        "steps",
        "candidate_rounds",
        "candidate_rounds_artifact_sha256",
        "initial_direction_train_oracle",
        "final_direction_train_oracle",
        "initial_allocation_validation_oracle",
        "final_allocation_validation_oracle",
        "initial_global_allocation_validation_oracle",
        "final_global_allocation_validation_oracle",
        "direction_train_relative_fisher_loss_history",
        "allocation_validation_relative_fisher_loss_history",
        "global_allocation_validation_relative_fisher_loss_history",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("global-gated allocation schema drifted")
    sequence = value["frozen_allocation_sequence"]
    rounds = value["candidate_rounds"]
    steps = value["steps"]
    initial_sha = value["initial_bank_sha256"]
    threshold = value["global_relative_fisher_threshold"]
    if (
        not _is_sha256(initial_sha)
        or not isinstance(sequence, list)
        or len(sequence) > _ALLOCATION_BUDGET
        or any(item not in _NUMERIC_INVOCATION_IDS for item in sequence)
        or value["allocation_budget"] != _ALLOCATION_BUDGET
        or type(value["budget_exhausted"]) is not bool
        or value["budget_exhausted"] is not (len(sequence) == _ALLOCATION_BUDGET)
        or not _close(threshold, _GLOBAL_RELATIVE_FISHER_THRESHOLD)
        or type(value["global_gate_passed"]) is not bool
        or not isinstance(rounds, list)
        or not isinstance(steps, list)
        or len(steps) != len(sequence)
        or value["candidate_rounds_artifact_sha256"]
        != _candidate_rounds_artifact_sha256(rounds)
    ):
        raise ValueError("global-gated allocation profile drifted")
    if tuple(
        step.get("invocation_id") if isinstance(step, Mapping) else None
        for step in steps
    ) != tuple(sequence):
        raise ValueError("global-gated frozen allocation sequence drifted")
    expected_rounds = len(steps) if len(steps) == _ALLOCATION_BUDGET else len(steps) + 1
    if len(rounds) != expected_rounds:
        raise ValueError("global-gated allocation round count drifted")
    bank = value["final_bank"]
    if not isinstance(bank, Mapping):
        raise ValueError("global-gated final bank is absent")
    parent_sha = bank.get("parent_site_rank_one_bank_sha256")
    if not _is_sha256(parent_sha):
        raise ValueError("global-gated parent bank binding drifted")
    _validate_bank_summary(
        bank,
        expected_parent_sha256=str(parent_sha),
        expected_sequence=sequence,
    )
    if (
        bank["allocated_rank_profile"] != value["frozen_allocated_rank_profile"]
        or bank["total_rank_profile"] != value["frozen_total_rank_profile"]
        or value["frozen_profile_artifact_sha256"]
        != _profile_artifact_sha256(
            sequence,
            bank["allocated_rank_profile"],
            bank["total_rank_profile"],
            value["candidate_rounds_artifact_sha256"],
        )
    ):
        raise ValueError("global-gated frozen profile binding drifted")
    first = _sealed_runner()._validate_first_spectrum(
        value["first_spectrum"],
        expected_parent_sha256=parent_sha,
    )
    if first["source_bank_sha256"] != initial_sha:
        raise ValueError("global-gated first spectrum lineage drifted")
    first_marginal_by_id = {
        str(item["invocation_id"]): item for item in first["per_site"]
    }

    initial_fit = _validate_fit_metric(
        value["initial_direction_train_oracle"],
        expected_rows=_DIRECTION_TRAIN_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
    )
    final_fit = _validate_fit_metric(
        value["final_direction_train_oracle"],
        expected_rows=_DIRECTION_TRAIN_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
    )
    initial_all = _validate_fit_metric(
        value["initial_allocation_validation_oracle"],
        expected_rows=_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
    )
    final_all = _validate_fit_metric(
        value["final_allocation_validation_oracle"],
        expected_rows=_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
    )
    initial_global = _validate_fit_metric(
        value["initial_global_allocation_validation_oracle"],
        expected_rows=_GLOBAL_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    final_global = _validate_fit_metric(
        value["final_global_allocation_validation_oracle"],
        expected_rows=_GLOBAL_ALLOCATION_VALIDATION_ROWS,
        expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
    )
    if (
        initial_fit != first["direction_train_current_oracle"]
        or initial_all != first["allocation_validation_current_oracle"]
        or value["global_gate_passed"]
        is not (
            float(final_global["relative_fisher_loss"])
            <= _GLOBAL_RELATIVE_FISHER_THRESHOLD
        )
    ):
        raise ValueError("global-gated initial/final oracle binding drifted")
    histories = (
        value["direction_train_relative_fisher_loss_history"],
        value["allocation_validation_relative_fisher_loss_history"],
        value["global_allocation_validation_relative_fisher_loss_history"],
    )
    endpoints = ((initial_fit, final_fit), (initial_all, final_all), (initial_global, final_global))
    for history, (initial, final) in zip(histories, endpoints, strict=True):
        if (
            not isinstance(history, list)
            or len(history) != len(sequence) + 1
            or any(not _finite_float(item, minimum=0.0) for item in history)
            or not _close(history[0], initial["relative_fisher_loss"])
            or not _close(history[-1], final["relative_fisher_loss"])
            or any(right > left + 1.0e-12 for left, right in zip(history, history[1:]))
        ):
            raise ValueError("global-gated loss history drifted")

    prior_round_by_id: dict[str, Mapping[str, object]] | None = None
    current_sha = initial_sha
    state_profile_by_sha: dict[str, tuple[str, ...]] = {initial_sha: ()}
    expected_before_fit: Mapping[str, object] = initial_fit
    expected_before_all: Mapping[str, object] = initial_all
    expected_before_global: Mapping[str, object] = initial_global
    for round_index, round_value in enumerate(rounds):
        round_fields = {
            "allocation_index",
            "current_bank_sha256",
            "objective_mode",
            "global_gate_active",
            "global_relative_fisher_threshold",
            "global_relative_fisher_loss_before",
            "all_layer_relative_fisher_loss_before",
            "global_evaluation_row_count",
            "global_evaluation_zero_predictor_fisher_loss",
            "all_layer_evaluation_row_count",
            "all_layer_evaluation_zero_predictor_fisher_loss",
            "minimum_relative_fisher_gain",
            "tie_tolerance",
            "candidates",
            "selected_invocation_id",
            "early_stop_no_eligible_candidate",
        }
        if not isinstance(round_value, Mapping) or set(round_value) != round_fields:
            raise ValueError("global-gated candidate-round schema drifted")
        global_loss = round_value["global_relative_fisher_loss_before"]
        all_loss = round_value["all_layer_relative_fisher_loss_before"]
        expected_mode = (
            "global_attention"
            if float(global_loss) > _GLOBAL_RELATIVE_FISHER_THRESHOLD
            else "all_layers"
        )
        candidates = round_value["candidates"]
        if (
            round_value["allocation_index"] != round_index
            or round_value["current_bank_sha256"] != current_sha
            or round_value["objective_mode"] != expected_mode
            or round_value["global_gate_active"] is not (expected_mode == "global_attention")
            or not _close(
                round_value["global_relative_fisher_threshold"],
                _GLOBAL_RELATIVE_FISHER_THRESHOLD,
            )
            or not _close(global_loss, histories[2][round_index])
            or not _close(all_loss, histories[1][round_index])
            or round_value["global_evaluation_row_count"]
            != _GLOBAL_ALLOCATION_VALIDATION_ROWS
            or round_value["all_layer_evaluation_row_count"]
            != _ALLOCATION_VALIDATION_ROWS
            or not _finite_float(
                round_value["global_evaluation_zero_predictor_fisher_loss"],
                minimum=0.0,
            )
            or float(round_value["global_evaluation_zero_predictor_fisher_loss"])
            <= 0.0
            or not _finite_float(
                round_value["all_layer_evaluation_zero_predictor_fisher_loss"],
                minimum=0.0,
            )
            or float(round_value["all_layer_evaluation_zero_predictor_fisher_loss"])
            <= 0.0
            or (
                "zero_predictor_fisher_loss" in expected_before_all
                and not _close(
                    round_value[
                        "all_layer_evaluation_zero_predictor_fisher_loss"
                    ],
                    expected_before_all["zero_predictor_fisher_loss"],
                )
            )
            or (
                "zero_predictor_fisher_loss" in expected_before_global
                and not _close(
                    round_value[
                        "global_evaluation_zero_predictor_fisher_loss"
                    ],
                    expected_before_global["zero_predictor_fisher_loss"],
                )
            )
            or not _close(round_value["minimum_relative_fisher_gain"], 1.0e-12)
            or not _close(round_value["tie_tolerance"], 1.0e-15)
            or not isinstance(candidates, list)
            or len(candidates) != 18
            or [item.get("invocation_id") for item in candidates if isinstance(item, Mapping)]
            != list(_NUMERIC_INVOCATION_IDS)
        ):
            raise ValueError("global-gated candidate-round binding drifted")
        by_id: dict[str, Mapping[str, object]] = {}
        eligible: list[Mapping[str, object]] = []
        current_metrics_by_id: dict[str, Mapping[str, object]] = {}
        current_fit_metrics_by_id: dict[str, Mapping[str, object]] = {}
        current_profile = tuple(sequence[:round_index])
        if state_profile_by_sha.get(current_sha) != current_profile:
            raise ValueError("global-gated current state/profile chain drifted")
        for candidate in candidates:
            candidate_fields = {
                "invocation_id",
                "attention_kind",
                "source_bank_sha256",
                "used_cached_off_site_marginal",
                "strictly_improves_all_protocol_gates",
                "in_objective_scope",
                "global_normalized_fisher_mass_gain",
                "all_layer_normalized_fisher_mass_gain",
                "selection_objective_normalized_fisher_mass_gain",
                "eligible_for_selection",
                "marginal",
            }
            if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
                raise ValueError("global-gated candidate schema drifted")
            invocation_id = candidate["invocation_id"]
            in_scope = (
                invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                if expected_mode == "global_attention"
                else invocation_id in _NUMERIC_INVOCATION_IDS
            )
            expected_kind = (
                "global" if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS else "local"
            )
            candidate_source_sha = candidate["source_bank_sha256"]
            source_profile = state_profile_by_sha.get(str(candidate_source_sha))
            if source_profile is None:
                raise ValueError("global-gated candidate source is not a prior state")
            if current_profile[: len(source_profile)] != source_profile:
                raise ValueError(
                    "global-gated candidate source is not on the current state chain"
                )
            if (
                candidate_source_sha != current_sha
                and invocation_id in current_profile[len(source_profile) :]
            ):
                raise ValueError(
                    "global-gated cached marginal crossed a same-site update"
                )
            marginal = _validate_marginal(
                candidate["marginal"],
                expected_source_sha256=str(candidate_source_sha),
                expected_parent_sha256=parent_sha,
                expected_proposal_sequence=source_profile + (str(invocation_id),),
            )
            current_metrics_by_id[str(invocation_id)] = marginal[
                "allocation_evaluation_current_oracle"
            ]
            if "proposal_fit_current_oracle" in marginal:
                current_fit_metrics_by_id[str(invocation_id)] = marginal[
                    "proposal_fit_current_oracle"
                ]
            raw_mass_gain = int(
                marginal["allocation_evaluation_current_oracle"]["row_count"]
            ) * (
                float(marginal["allocation_evaluation_current_oracle"]["fisher_loss"])
                - float(marginal["allocation_evaluation_candidate_oracle"]["fisher_loss"])
            )
            expected_all_gain = raw_mass_gain / float(
                round_value["all_layer_evaluation_row_count"]
            ) / float(round_value["all_layer_evaluation_zero_predictor_fisher_loss"])
            expected_global_gain = (
                raw_mass_gain
                / float(round_value["global_evaluation_row_count"])
                / float(round_value["global_evaluation_zero_predictor_fisher_loss"])
                if invocation_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                else 0.0
            )
            objective_gain = (
                expected_global_gain if expected_mode == "global_attention" else expected_all_gain
            )
            strictly = marginal["strictly_improves_all_protocol_gates"]
            is_eligible = (
                strictly
                and in_scope
                and objective_gain > float(round_value["minimum_relative_fisher_gain"])
            )
            if (
                invocation_id not in _NUMERIC_INVOCATION_IDS
                or candidate["attention_kind"] != expected_kind
                or candidate["source_bank_sha256"]
                != marginal["source_bank_sha256"]
                or candidate["used_cached_off_site_marginal"]
                is not (candidate["source_bank_sha256"] != current_sha)
                or candidate["strictly_improves_all_protocol_gates"] is not strictly
                or candidate["in_objective_scope"] is not in_scope
                or not _close(
                    candidate["global_normalized_fisher_mass_gain"],
                    expected_global_gain,
                )
                or not _close(
                    candidate["all_layer_normalized_fisher_mass_gain"],
                    expected_all_gain,
                )
                or not _close(
                    candidate["selection_objective_normalized_fisher_mass_gain"],
                    objective_gain,
                )
                or candidate["eligible_for_selection"] is not is_eligible
            ):
                raise ValueError("global-gated candidate objective closure drifted")
            by_id[str(invocation_id)] = candidate
            if is_eligible:
                eligible.append(candidate)
        all_rows = sum(
            int(metric["row_count"]) for metric in current_metrics_by_id.values()
        )
        global_rows = sum(
            int(current_metrics_by_id[key]["row_count"])
            for key in _GLOBAL_ATTENTION_INVOCATION_IDS
        )
        all_zero = sum(
            int(metric["row_count"])
            * float(metric["zero_predictor_fisher_loss"])
            for metric in current_metrics_by_id.values()
        ) / all_rows
        global_zero = sum(
            int(current_metrics_by_id[key]["row_count"])
            * float(current_metrics_by_id[key]["zero_predictor_fisher_loss"])
            for key in _GLOBAL_ATTENTION_INVOCATION_IDS
        ) / global_rows
        all_fisher = sum(
            int(metric["row_count"]) * float(metric["fisher_loss"])
            for metric in current_metrics_by_id.values()
        ) / all_rows
        global_fisher = sum(
            int(current_metrics_by_id[key]["row_count"])
            * float(current_metrics_by_id[key]["fisher_loss"])
            for key in _GLOBAL_ATTENTION_INVOCATION_IDS
        ) / global_rows
        if (
            all_rows != round_value["all_layer_evaluation_row_count"]
            or global_rows != round_value["global_evaluation_row_count"]
            or not _close(
                round_value["all_layer_evaluation_zero_predictor_fisher_loss"],
                all_zero,
            )
            or not _close(
                round_value["global_evaluation_zero_predictor_fisher_loss"],
                global_zero,
            )
            or not _close(
                round_value["all_layer_relative_fisher_loss_before"],
                all_fisher / all_zero,
            )
            or not _close(
                round_value["global_relative_fisher_loss_before"],
                global_fisher / global_zero,
            )
        ):
            raise ValueError("global-gated round aggregate baseline does not close")
        if current_fit_metrics_by_id and (
            set(current_fit_metrics_by_id) != set(_NUMERIC_INVOCATION_IDS)
            or not _site_metric_aggregate_closes(
                current_fit_metrics_by_id,
                invocation_ids=_NUMERIC_INVOCATION_IDS,
                aggregate=expected_before_fit,
            )
        ):
            raise ValueError(
                "global-gated direction-train aggregate baseline does not close"
            )
        selected = round_value["selected_invocation_id"]
        expected_selected: str | None = None
        if eligible:
            best = max(
                float(item["selection_objective_normalized_fisher_mass_gain"])
                for item in eligible
            )
            tied = [
                item
                for item in eligible
                if float(item["selection_objective_normalized_fisher_mass_gain"])
                >= best - float(round_value["tie_tolerance"])
            ]
            expected_selected = str(
                min(tied, key=lambda item: _mutation_ordinal(str(item["invocation_id"])))
                ["invocation_id"]
            )
        if (
            selected != expected_selected
            or round_value["early_stop_no_eligible_candidate"]
            is not (expected_selected is None)
        ):
            raise ValueError("global-gated argmax, tie, or early-stop drifted")
        if prior_round_by_id is None:
            if (
                any(item["source_bank_sha256"] != initial_sha for item in candidates)
                or any(
                    by_id[invocation_id]["marginal"]
                    != first_marginal_by_id[invocation_id]
                    for invocation_id in _NUMERIC_INVOCATION_IDS
                )
            ):
                raise ValueError("global-gated first-round source lineage drifted")
        else:
            prior_selected = sequence[round_index - 1]
            for invocation_id, candidate in by_id.items():
                prior_candidate = prior_round_by_id[invocation_id]
                if invocation_id == prior_selected:
                    if (
                        candidate["source_bank_sha256"] != current_sha
                        or candidate["used_cached_off_site_marginal"] is not False
                    ):
                        raise ValueError(
                            "global-gated selected-site marginal was not recomputed"
                        )
                elif (
                    candidate["source_bank_sha256"]
                    != prior_candidate["source_bank_sha256"]
                    or candidate["marginal"] != prior_candidate["marginal"]
                ):
                    raise ValueError("global-gated off-site cached candidate drifted")
        prior_round_by_id = by_id
        if round_index < len(steps):
            step = steps[round_index]
            if not isinstance(step, Mapping):
                raise ValueError("global-gated allocation step is absent")
            if selected != sequence[round_index]:
                raise ValueError(
                    "global-gated accepted round/sequence binding drifted"
                )
            step_fields = {
                "allocation_index",
                "invocation_id",
                "objective_mode",
                "source_bank_sha256",
                "result_bank_sha256",
                "marginal_source_bank_sha256",
                "used_cached_off_site_marginal",
                "selection_objective_normalized_fisher_gain",
                "selection_global_normalized_fisher_gain",
                "selection_all_layer_normalized_fisher_gain",
                "marginal",
                "direction_train_before",
                "direction_train_after",
                "allocation_validation_before",
                "allocation_validation_after",
                "global_allocation_validation_before",
                "global_allocation_validation_after",
            }
            selected_candidate = by_id[str(selected)]
            selected_marginal = selected_candidate["marginal"]
            selected_is_fresh = (
                selected_candidate["source_bank_sha256"] == current_sha
            )
            if (
                selected_is_fresh
                and "proposal_bank" in selected_marginal
                and selected_marginal["proposal_bank"]["artifact_sha256"]
                != step.get("result_bank_sha256")
            ):
                raise ValueError(
                    "global-gated fresh proposal/result bank identity drifted"
                )
            if (
                not selected_is_fresh
                and str(selected) in current_profile[
                    len(
                        state_profile_by_sha[
                            str(selected_candidate["source_bank_sha256"])
                        ]
                    ) :
                ]
            ):
                raise ValueError(
                    "global-gated cached selection crossed a same-site update"
                )
            if (
                set(step) != step_fields
                or step["allocation_index"] != round_index
                or step["invocation_id"] != selected
                or step["objective_mode"] != expected_mode
                or step["source_bank_sha256"] != current_sha
                or step["marginal_source_bank_sha256"]
                != selected_candidate["source_bank_sha256"]
                or step["used_cached_off_site_marginal"]
                is not selected_candidate["used_cached_off_site_marginal"]
                or step["marginal"] != selected_candidate["marginal"]
                or not _is_sha256(step["result_bank_sha256"])
                or step["result_bank_sha256"] == current_sha
                or step["result_bank_sha256"] in state_profile_by_sha
                or not _close(
                    step["selection_objective_normalized_fisher_gain"],
                    selected_candidate[
                        "selection_objective_normalized_fisher_mass_gain"
                    ],
                )
                or not _close(
                    step["selection_global_normalized_fisher_gain"],
                    selected_candidate["global_normalized_fisher_mass_gain"],
                )
                or not _close(
                    step["selection_all_layer_normalized_fisher_gain"],
                    selected_candidate["all_layer_normalized_fisher_mass_gain"],
                )
            ):
                raise ValueError("global-gated allocation step lineage drifted")
            before_fit = _validate_fit_metric(
                step["direction_train_before"],
                expected_rows=_DIRECTION_TRAIN_ROWS,
                expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            )
            after_fit = _validate_fit_metric(
                step["direction_train_after"],
                expected_rows=_DIRECTION_TRAIN_ROWS,
                expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            )
            before_all = _validate_fit_metric(
                step["allocation_validation_before"],
                expected_rows=_ALLOCATION_VALIDATION_ROWS,
                expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            )
            after_all = _validate_fit_metric(
                step["allocation_validation_after"],
                expected_rows=_ALLOCATION_VALIDATION_ROWS,
                expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
            )
            before_global = _validate_fit_metric(
                step["global_allocation_validation_before"],
                expected_rows=_GLOBAL_ALLOCATION_VALIDATION_ROWS,
                expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
            )
            after_global = _validate_fit_metric(
                step["global_allocation_validation_after"],
                expected_rows=_GLOBAL_ALLOCATION_VALIDATION_ROWS,
                expected_invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
            )
            before_fit_by_id = dict(
                before_fit.get("per_invocation_relative_fisher_loss", ())
            )
            after_fit_by_id = dict(
                after_fit.get("per_invocation_relative_fisher_loss", ())
            )
            before_all_by_id = dict(
                before_all.get("per_invocation_relative_fisher_loss", ())
            )
            after_all_by_id = dict(
                after_all.get("per_invocation_relative_fisher_loss", ())
            )
            before_global_by_id = dict(
                before_global.get("per_invocation_relative_fisher_loss", ())
            )
            after_global_by_id = dict(
                after_global.get("per_invocation_relative_fisher_loss", ())
            )
            has_fit_per_invocation = (
                set(before_fit_by_id)
                == set(after_fit_by_id)
                == set(_NUMERIC_INVOCATION_IDS)
            )
            has_all_per_invocation = (
                set(before_all_by_id)
                == set(after_all_by_id)
                == set(_NUMERIC_INVOCATION_IDS)
            )
            has_global_per_invocation = (
                set(before_global_by_id)
                == set(after_global_by_id)
                == set(_GLOBAL_ATTENTION_INVOCATION_IDS)
            )
            selected_id = str(selected)
            marginal_fit_current = None
            marginal_fit_candidate = None
            if (
                "proposal_fit_current_oracle" in selected_marginal
                and "proposal_fit_candidate_oracle" in selected_marginal
            ):
                marginal_fit_current = selected_marginal[
                    "proposal_fit_current_oracle"
                ]["relative_fisher_loss"]
                marginal_fit_candidate = selected_marginal[
                    "proposal_fit_candidate_oracle"
                ]["relative_fisher_loss"]
            marginal_evaluation_current = selected_marginal[
                "allocation_evaluation_current_oracle"
            ]["relative_fisher_loss"]
            marginal_evaluation_candidate = selected_marginal[
                "allocation_evaluation_candidate_oracle"
            ]["relative_fisher_loss"]
            post_fit_metrics_by_id: dict[
                str, Mapping[str, object]
            ] | None = None
            if current_fit_metrics_by_id:
                candidate_fit_metric = selected_marginal.get(
                    "proposal_fit_candidate_oracle"
                )
                if not isinstance(candidate_fit_metric, Mapping):
                    raise ValueError(
                        "global-gated selected direction metric is absent"
                    )
                post_fit_metrics_by_id = dict(current_fit_metrics_by_id)
                post_fit_metrics_by_id[selected_id] = candidate_fit_metric
            candidate_evaluation_metric = selected_marginal[
                "allocation_evaluation_candidate_oracle"
            ]
            if not isinstance(candidate_evaluation_metric, Mapping):
                raise ValueError(
                    "global-gated selected allocation metric is absent"
                )
            post_evaluation_metrics_by_id = dict(current_metrics_by_id)
            post_evaluation_metrics_by_id[selected_id] = (
                candidate_evaluation_metric
            )
            if (
                before_fit != expected_before_fit
                or before_all != expected_before_all
                or before_global != expected_before_global
                or (
                    "zero_predictor_fisher_loss" in before_fit
                    and "zero_predictor_fisher_loss" in after_fit
                    and not _close(
                        before_fit["zero_predictor_fisher_loss"],
                        after_fit["zero_predictor_fisher_loss"],
                    )
                )
                or (
                    "zero_predictor_fisher_loss" in before_all
                    and "zero_predictor_fisher_loss" in after_all
                    and not _close(
                        before_all["zero_predictor_fisher_loss"],
                        after_all["zero_predictor_fisher_loss"],
                    )
                )
                or (
                    "zero_predictor_fisher_loss" in before_global
                    and "zero_predictor_fisher_loss" in after_global
                    and not _close(
                        before_global["zero_predictor_fisher_loss"],
                        after_global["zero_predictor_fisher_loss"],
                    )
                )
                or (
                    post_fit_metrics_by_id is not None
                    and not _site_metric_aggregate_closes(
                        post_fit_metrics_by_id,
                        invocation_ids=_NUMERIC_INVOCATION_IDS,
                        aggregate=after_fit,
                    )
                )
                or not _site_metric_aggregate_closes(
                    post_evaluation_metrics_by_id,
                    invocation_ids=_NUMERIC_INVOCATION_IDS,
                    aggregate=after_all,
                )
                or not _site_metric_aggregate_closes(
                    post_evaluation_metrics_by_id,
                    invocation_ids=_GLOBAL_ATTENTION_INVOCATION_IDS,
                    aggregate=after_global,
                )
                or not _close(
                    before_fit["relative_fisher_loss"],
                    histories[0][round_index],
                )
                or not _close(after_fit["relative_fisher_loss"], histories[0][round_index + 1])
                or not _close(before_all["relative_fisher_loss"], histories[1][round_index])
                or not _close(after_all["relative_fisher_loss"], histories[1][round_index + 1])
                or not _close(before_global["relative_fisher_loss"], histories[2][round_index])
                or not _close(after_global["relative_fisher_loss"], histories[2][round_index + 1])
                or not _close(
                    step["selection_all_layer_normalized_fisher_gain"],
                    float(before_all["relative_fisher_loss"])
                    - float(after_all["relative_fisher_loss"]),
                )
                or not _close(
                    step["selection_global_normalized_fisher_gain"],
                    float(before_global["relative_fisher_loss"])
                    - float(after_global["relative_fisher_loss"]),
                )
                or (
                    marginal_fit_current is not None
                    and has_fit_per_invocation
                    and not _close(
                        marginal_fit_current,
                        before_fit_by_id[selected_id],
                    )
                )
                or (
                    marginal_fit_candidate is not None
                    and has_fit_per_invocation
                    and not _close(
                        marginal_fit_candidate,
                        after_fit_by_id[selected_id],
                    )
                )
                or (
                    has_all_per_invocation
                    and not _close(
                        marginal_evaluation_current,
                        before_all_by_id[selected_id],
                    )
                )
                or (
                    has_all_per_invocation
                    and not _close(
                        marginal_evaluation_candidate,
                        after_all_by_id[selected_id],
                    )
                )
                or (
                    len(current_fit_metrics_by_id) == len(_NUMERIC_INVOCATION_IDS)
                    and has_fit_per_invocation
                    and any(
                        not _close(
                            current_fit_metrics_by_id[key]["relative_fisher_loss"],
                            before_fit_by_id[key],
                        )
                        for key in _NUMERIC_INVOCATION_IDS
                    )
                )
                or (
                    has_all_per_invocation
                    and any(
                        not _close(
                            current_metrics_by_id[key]["relative_fisher_loss"],
                            before_all_by_id[key],
                        )
                        for key in _NUMERIC_INVOCATION_IDS
                    )
                )
                or (
                    has_global_per_invocation
                    and any(
                        not _close(
                            current_metrics_by_id[key]["relative_fisher_loss"],
                            before_global_by_id[key],
                        )
                        for key in _GLOBAL_ATTENTION_INVOCATION_IDS
                    )
                )
                or (
                    has_fit_per_invocation
                    and any(
                        not _close(before_fit_by_id[key], after_fit_by_id[key])
                        for key in _NUMERIC_INVOCATION_IDS
                        if key != selected_id
                    )
                )
                or (
                    has_all_per_invocation
                    and any(
                        not _close(before_all_by_id[key], after_all_by_id[key])
                        for key in _NUMERIC_INVOCATION_IDS
                        if key != selected_id
                    )
                )
                or (
                    has_global_per_invocation
                    and any(
                        not _close(
                            before_global_by_id[key],
                            after_global_by_id[key],
                        )
                        for key in _GLOBAL_ATTENTION_INVOCATION_IDS
                        if key != selected_id
                    )
                )
                or (
                    selected_id in _GLOBAL_ATTENTION_INVOCATION_IDS
                    and has_global_per_invocation
                    and (
                        not _close(
                            marginal_evaluation_current,
                            before_global_by_id[selected_id],
                        )
                        or not _close(
                            marginal_evaluation_candidate,
                            after_global_by_id[selected_id],
                        )
                    )
                )
            ):
                raise ValueError("global-gated allocation step metrics drifted")
            current_sha = str(step["result_bank_sha256"])
            state_profile_by_sha[current_sha] = tuple(sequence[: round_index + 1])
            expected_before_fit = after_fit
            expected_before_all = after_all
            expected_before_global = after_global
    if current_sha != bank["artifact_sha256"]:
        raise ValueError("global-gated final bank lineage drifted")
    if (
        expected_before_fit != final_fit
        or expected_before_all != final_all
        or expected_before_global != final_global
    ):
        raise ValueError("global-gated final aggregate state drifted")
    return value


def _validate_fixed_profile_replay(
    value: object,
    *,
    frozen_sequence: Sequence[str],
    frozen_profile_sha256: str,
    frozen_candidate_rounds_sha256: str,
) -> Mapping[str, object]:
    fields = {
        "initial_bank_sha256",
        "final_bank",
        "allocation_profile",
        "allocated_component_count",
        "selection_candidate_rounds_artifact_sha256",
        "frozen_profile_artifact_sha256",
        "replay_profile_artifact_sha256",
        "replay_profile_matches_frozen_selection",
        "steps",
        "initial_full_fit_oracle",
        "final_full_fit_oracle",
        "full_fit_relative_fisher_loss_history",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("global-gated fixed replay schema drifted")
    sequence = list(frozen_sequence)
    initial_sha = value["initial_bank_sha256"]
    bank = value["final_bank"]
    if not isinstance(bank, Mapping) or not _is_sha256(initial_sha):
        raise ValueError("global-gated fixed replay bank binding drifted")
    parent_sha = bank.get("parent_site_rank_one_bank_sha256")
    if not _is_sha256(parent_sha):
        raise ValueError("global-gated fixed replay parent binding drifted")
    _validate_bank_summary(
        bank,
        expected_parent_sha256=str(parent_sha),
        expected_sequence=sequence,
    )
    expected_profile = _profile_artifact_sha256(
        sequence,
        bank["allocated_rank_profile"],
        bank["total_rank_profile"],
        frozen_candidate_rounds_sha256,
    )
    if (
        value["allocation_profile"] != sequence
        or value["allocated_component_count"] != len(sequence)
        or value["selection_candidate_rounds_artifact_sha256"]
        != frozen_candidate_rounds_sha256
        or value["frozen_profile_artifact_sha256"] != frozen_profile_sha256
        or value["replay_profile_artifact_sha256"] != expected_profile
        or expected_profile != frozen_profile_sha256
        or value["replay_profile_matches_frozen_selection"] is not True
    ):
        raise ValueError("global-gated fixed replay profile drifted")
    initial = _validate_fit_metric(
        value["initial_full_fit_oracle"],
        expected_rows=_OUTER_FIT_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
    )
    final = _validate_fit_metric(
        value["final_full_fit_oracle"],
        expected_rows=_OUTER_FIT_ROWS,
        expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
    )
    history = value["full_fit_relative_fisher_loss_history"]
    steps = value["steps"]
    if (
        not isinstance(history, list)
        or len(history) != len(sequence) + 1
        or any(not _finite_float(item, minimum=0.0) for item in history)
        or not _close(history[0], initial["relative_fisher_loss"])
        or not _close(history[-1], final["relative_fisher_loss"])
        or not _close(
            initial["zero_predictor_fisher_loss"],
            final["zero_predictor_fisher_loss"],
        )
        or any(right >= left for left, right in zip(history, history[1:]))
        or not isinstance(steps, list)
        or len(steps) != len(sequence)
    ):
        raise ValueError("global-gated fixed replay history drifted")
    source_sha = initial_sha
    seen_state_sha256s = {initial_sha}
    expected_before: Mapping[str, object] = initial
    for index, step in enumerate(steps):
        step_fields = {
            "allocation_index",
            "invocation_id",
            "source_bank_sha256",
            "result_bank_sha256",
            "marginal_source_bank_sha256",
            "used_cached_off_site_marginal",
            "all_layer_relative_fisher_gain",
            "marginal",
            "full_fit_before",
            "full_fit_after",
        }
        if (
            not isinstance(step, Mapping)
            or set(step) != step_fields
            or step["allocation_index"] != index
            or step["invocation_id"] != sequence[index]
            or step["source_bank_sha256"] != source_sha
            or step["marginal_source_bank_sha256"] != source_sha
            or step["used_cached_off_site_marginal"] is not False
            or not _is_sha256(step["result_bank_sha256"])
            or step["result_bank_sha256"] in seen_state_sha256s
            or not _finite_float(
                step["all_layer_relative_fisher_gain"], minimum=0.0
            )
            or float(step["all_layer_relative_fisher_gain"]) <= 0.0
        ):
            raise ValueError("global-gated fixed replay step lineage drifted")
        marginal = _sealed_runner()._validate_marginal(
            step["marginal"],
            proposal_rows=_OUTER_FIT_RECEIPT_COUNT * 16,
            evaluation_rows=_OUTER_FIT_RECEIPT_COUNT * 16,
            expected_source_sha256=source_sha,
            expected_parent_sha256=parent_sha,
            expected_proposal_sequence=sequence[: index + 1],
        )
        before = _validate_fit_metric(
            step["full_fit_before"],
            expected_rows=_OUTER_FIT_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        )
        after = _validate_fit_metric(
            step["full_fit_after"],
            expected_rows=_OUTER_FIT_ROWS,
            expected_invocation_ids=_NUMERIC_INVOCATION_IDS,
        )
        before_by_id = dict(before["per_invocation_relative_fisher_loss"])
        after_by_id = dict(after["per_invocation_relative_fisher_loss"])
        selected = str(step["invocation_id"])
        evaluation_current = marginal[
            "allocation_evaluation_current_oracle"
        ]
        evaluation_candidate = marginal[
            "allocation_evaluation_candidate_oracle"
        ]
        site_mass_gain = (
            int(evaluation_current["row_count"])
            * (
                float(evaluation_current["fisher_loss"])
                - float(evaluation_candidate["fisher_loss"])
            )
            / _OUTER_FIT_ROWS
            / float(before["zero_predictor_fisher_loss"])
        )
        if (
            before != expected_before
            or not _close(before["relative_fisher_loss"], history[index])
            or not _close(after["relative_fisher_loss"], history[index + 1])
            or not _close(
                before["zero_predictor_fisher_loss"],
                after["zero_predictor_fisher_loss"],
            )
            or not _close(
                step["all_layer_relative_fisher_gain"],
                float(before["relative_fisher_loss"])
                - float(after["relative_fisher_loss"]),
            )
            or not _close(
                step["all_layer_relative_fisher_gain"],
                site_mass_gain,
            )
            or not _close(
                marginal["proposal_fit_current_oracle"]["relative_fisher_loss"],
                before_by_id[selected],
            )
            or not _close(
                marginal["proposal_fit_candidate_oracle"]["relative_fisher_loss"],
                after_by_id[selected],
            )
            or not _close(
                evaluation_current["relative_fisher_loss"],
                before_by_id[selected],
            )
            or not _close(
                evaluation_candidate["relative_fisher_loss"],
                after_by_id[selected],
            )
            or any(
                not _close(before_by_id[key], after_by_id[key])
                for key in _NUMERIC_INVOCATION_IDS
                if key != selected
            )
            or marginal["proposal_bank"]["artifact_sha256"]
            != step["result_bank_sha256"]
        ):
            raise ValueError("global-gated fixed replay step metrics drifted")
        source_sha = str(step["result_bank_sha256"])
        seen_state_sha256s.add(source_sha)
        expected_before = after
    if (
        source_sha != bank["artifact_sha256"]
        or expected_before != final
    ):
        raise ValueError("global-gated fixed replay final lineage drifted")
    return value


def _expected_scientific_status() -> dict[str, object]:
    return {
        "A_only_development_diagnostic": True,
        "A_fitted": True,
        "A_corpus_previously_exposed": True,
        "prior_A_dev_evidence_influenced_architecture_design": True,
        "stable_cross_run_A_split_identity_enforced": True,
        "execution_receipts_drive_partition": False,
        "inner_allocation_validation_used_for_selection": True,
        "global_gate_uses_inner_allocation_validation_only": True,
        "global_gate_threshold_fixed_before_run": True,
        "global_sites_only_while_gate_active": True,
        "all_layer_objective_only_after_global_gate_passes": True,
        "outer_A_dev_readout_descriptive_replay_only": True,
        "outer_A_dev_used_for_current_run_selection_or_fitting": False,
        "outer_A_dev_evaluated_once_after_profile_freeze": True,
        "allocation_profile_frozen_before_full_fit_replay": True,
        "fresh_inner_parent_fit": True,
        "fresh_full_fit_parent_fit": True,
        "sealed_predecessor_artifact_reused": False,
        "B_previously_exposed": True,
        "B_influenced_architecture_design": True,
        "B_opened_by_this_run": False,
        "B_used_for_training": False,
        "C_ever_opened": False,
        "confirmation_claimed": False,
        "behavioral_nll_measured": False,
        "operator_boundary_only": True,
        "all_18_layer_residual_spectrum_measured": True,
        "allocation_budget_two_enforced": True,
        "global_top2_oracle_support_limit_enforced": True,
        "gate_candidate_rounds_ledger_retained": True,
        "gate_argmax_tie_switch_and_early_stop_recomputed": True,
        "deployment_router_built": False,
        "deployment_parameter_count_hypothetical": True,
        "executable_artifact_published": False,
        "logical_speedup_claimed": False,
    }


def _expected_exposure_binding() -> dict[str, object]:
    return {
        "predecessor_report_path": str(_PREDECESSOR_REPORT_PATH),
        "predecessor_report_file_sha256": _PREDECESSOR_REPORT_FILE_SHA256,
        "predecessor_report_sha256": _PREDECESSOR_REPORT_SHA256,
        "predecessor_core_source_sha256": _PREDECESSOR_CORE_SOURCE_SHA256,
        "predecessor_runner_source_sha256": _PREDECESSOR_RUNNER_SOURCE_SHA256,
        "predecessor_report_status_validated": True,
        "predecessor_report_source_catalog_validated": True,
        "predecessor_global_outer_gain_was_zero": True,
        "predecessor_artifact_used_for_initialization": False,
        "A_corpus_previously_exposed": True,
        "B_or_C_corpus_opened_by_this_run": False,
        "outer_A_dev_is_fresh_holdout": False,
        "outer_A_dev_is_confirmation": False,
        "interpretation": (
            "sealed_all_layer_failure_binds_lineage_and_motivates_a_stable_split_"
            "inner_global_gate_outer_A_dev_remains_descriptive_only"
        ),
    }


def _validate_report(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_FIELDS:
        raise ValueError("global-gated report top-level schema drifted")
    if value["schema"] != REPORT_SCHEMA or value["format_version"] != 1:
        raise ValueError("global-gated report version drifted")
    if value["scientific_status"] != _expected_scientific_status():
        raise ValueError("global-gated scientific status drifted")
    if value["exposure_binding"] != _expected_exposure_binding():
        raise ValueError("global-gated predecessor exposure binding drifted")
    model = value["model"]
    if (
        not isinstance(model, Mapping)
        or set(model)
        != {
            "model_id",
            "model_revision",
            "hidden_width",
            "runtime_qualification_receipt_sha256",
        }
        or model["model_id"] != _PINNED_MODEL_ID
        or model["model_revision"] != _PINNED_MODEL_REVISION
        or model["hidden_width"] != 640
        or not _is_sha256(model["runtime_qualification_receipt_sha256"])
    ):
        raise ValueError("global-gated model binding drifted")

    protocol = value["protocol"]
    protocol_fields = {
        "protocol_id",
        "outer_fit_batch_count",
        "outer_dev_batch_count",
        "direction_train_batch_count",
        "allocation_validation_batch_count",
        "outer_fit_row_count",
        "outer_dev_row_count",
        "direction_train_row_count",
        "allocation_validation_row_count",
        "global_direction_train_row_count",
        "global_allocation_validation_row_count",
        "allocation_budget",
        "global_relative_fisher_threshold",
        "B_and_C_capture_forbidden",
        "A_corpus_file_sha256",
        "A_role_panel_binding_sha256",
        "A_batch_plan_artifact_sha256",
        "A_outer_split_artifact_sha256",
        "A_inner_split_artifact_sha256",
        "outer_partition_policy",
        "inner_partition_policy",
        "parent_bank_fit",
        "global_gated_residual_spectrum",
        "outer_evaluation",
        "preflight_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
        "implementation_source_sha256s",
    }
    if not isinstance(protocol, Mapping) or set(protocol) != protocol_fields:
        raise ValueError("global-gated protocol schema drifted")
    expected_parent_fit = {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "site_rank_one_alternating_rounds": 4,
        "site_rank_one_steps_per_site_per_round": 48,
        "fit_oracle_coefficient_ridge": 1.0e-8,
        "fresh_fit_for_inner_and_full_panels": True,
        "learned_objects_fit_role": "A_only",
    }
    expected_gate = {
        "generator_count": 8,
        "generator_rank": 8,
        "active_count": 2,
        "euclidean_floor": 0.001,
        "coefficient_ridge": 0.0,
        "amplitude_ridge": 0.0,
        "power_iterations": 16,
        "deterministic_start_count": 4,
        "allocation_budget": 2,
        "residual_component_parameters": _RESIDUAL_COMPONENT_PARAMETERS,
        "maximum_allocated_residual_parameters": (
            _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
        ),
        "gauge_floor": 1.0e-12,
        "minimum_relative_fisher_gain": 1.0e-12,
        "tie_tolerance": 1.0e-15,
        "global_relative_fisher_threshold": _GLOBAL_RELATIVE_FISHER_THRESHOLD,
        "gate_decision_panel": "inner_allocation_validation_only",
        "global_candidate_invocation_ids": list(_GLOBAL_ATTENTION_INVOCATION_IDS),
        "objective_semantics": (
            "ratio_of_summed_fisher_mass_global_until_0_10_then_all_layer_v1"
        ),
        "greedy_semantics": (
            "budget_two_global_only_above_gate_cached_off_site_numeric_tiebreak_v1"
        ),
        "ledger_semantics": (
            "all_18_global_and_all_layer_normalized_marginals_v1"
        ),
        "allocation_can_stop_before_budget": True,
        "candidate_rounds_bound_into_frozen_profile": True,
        "candidate_rounds_contain_factor_tensors": False,
    }
    expected_outer = {
        "evaluation_role": "stable_outer_A_dev_descriptive_replay_only",
        "A_corpus_previously_exposed": True,
        "used_for_current_run_selection_or_fitting": False,
        "fresh_holdout": False,
        "confirmation_readout": False,
        "evaluation_after_profile_freeze_and_full_fit_replay": True,
        "parent_and_allocated_evaluated_on_same_outer_rows": True,
        "all_local_global_and_per_layer_metrics": True,
        "coefficient_ridge": 0.0,
        "oracle_ceiling_only": True,
        "evaluation_call_count": 1,
    }
    if (
        protocol["protocol_id"]
        != "A_only_stable_split_global_gated_residual_nested_30_6_outer_36_9_v1"
        or protocol["outer_fit_batch_count"] != _OUTER_FIT_RECEIPT_COUNT
        or protocol["outer_dev_batch_count"] != _OUTER_DEV_RECEIPT_COUNT
        or protocol["direction_train_batch_count"] != _DIRECTION_TRAIN_RECEIPT_COUNT
        or protocol["allocation_validation_batch_count"]
        != _ALLOCATION_VALIDATION_RECEIPT_COUNT
        or protocol["outer_fit_row_count"] != _OUTER_FIT_ROWS
        or protocol["outer_dev_row_count"] != _OUTER_DEV_ROWS
        or protocol["direction_train_row_count"] != _DIRECTION_TRAIN_ROWS
        or protocol["allocation_validation_row_count"]
        != _ALLOCATION_VALIDATION_ROWS
        or protocol["global_direction_train_row_count"]
        != _GLOBAL_DIRECTION_TRAIN_ROWS
        or protocol["global_allocation_validation_row_count"]
        != _GLOBAL_ALLOCATION_VALIDATION_ROWS
        or protocol["allocation_budget"] != _ALLOCATION_BUDGET
        or not _close(
            protocol["global_relative_fisher_threshold"],
            _GLOBAL_RELATIVE_FISHER_THRESHOLD,
        )
        or protocol["B_and_C_capture_forbidden"] is not True
        or protocol["A_corpus_file_sha256"] != _PINNED_A_CORPUS_FILE_SHA256
        or protocol["A_role_panel_binding_sha256"]
        != _PINNED_A_ROLE_PANEL_BINDING_SHA256
        or protocol["A_batch_plan_artifact_sha256"]
        != _PINNED_A_BATCH_PLAN_SHA256
        or protocol["A_outer_split_artifact_sha256"]
        != _PINNED_A_OUTER_SPLIT_SHA256
        or protocol["A_inner_split_artifact_sha256"]
        != _PINNED_A_INNER_SPLIT_SHA256
        or protocol["outer_partition_policy"]
        != "fixed_manifest_batch_identity_ordinals_36_fit_9_dev_v1"
        or protocol["inner_partition_policy"]
        != "fixed_outer_fit_batch_identity_ordinals_30_direction_6_validation_v1"
        or protocol["parent_bank_fit"] != expected_parent_fit
        or protocol["global_gated_residual_spectrum"] != expected_gate
        or protocol["outer_evaluation"] != expected_outer
        or not _is_sha256(protocol["preflight_receipt_sha256"])
        or not _is_sha256(protocol["dense_lexical_parity_receipt_sha256"])
    ):
        raise ValueError("global-gated protocol binding drifted")
    sources = protocol["implementation_source_sha256s"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(_SOURCE_PATHS)
        or any(not _is_sha256(item) for item in sources.values())
        or sources[_PREDECESSOR_CORE_SOURCE_PATH]
        != _PREDECESSOR_CORE_SOURCE_SHA256
        or sources[_PREDECESSOR_RUNNER_SOURCE_PATH]
        != _PREDECESSOR_RUNNER_SOURCE_SHA256
    ):
        raise ValueError("global-gated source catalog drifted")

    partition = _validate_partition(value["A_partition"])
    catalog = partition["stable_batch_catalog"]
    if (
        catalog["role_panel_binding_sha256"]
        != protocol["A_role_panel_binding_sha256"]
        or catalog["batch_plan_artifact_sha256"]
        != protocol["A_batch_plan_artifact_sha256"]
        or partition["outer"]["outer_split_artifact_sha256"]
        != protocol["A_outer_split_artifact_sha256"]
        or partition["inner"]["inner_split_artifact_sha256"]
        != protocol["A_inner_split_artifact_sha256"]
    ):
        raise ValueError("global-gated protocol/partition lineage drifted")

    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "selection_id",
        "inner_parent_A_fit",
        "global_gated_allocation",
        "candidate_rounds_artifact_sha256",
        "candidate_rounds_bound_into_frozen_profile",
        "profile_frozen_before_full_fit_replay",
        "outer_A_dev_observed_during_selection",
        "contains_residual_factor_tensors",
    }:
        raise ValueError("global-gated selection schema drifted")
    inner_parent = _sealed_runner()._validate_parent_fit(
        selection["inner_parent_A_fit"],
        receipt_count=_DIRECTION_TRAIN_RECEIPT_COUNT,
    )
    allocation = _validate_global_gated_allocation(
        selection["global_gated_allocation"]
    )
    if (
        selection["selection_id"]
        != "stable_inner30_direction_inner6_global_gated_budget2_v1"
        or allocation["final_bank"]["parent_site_rank_one_bank_sha256"]
        != inner_parent["bank_artifact_sha256"]
        or selection["candidate_rounds_artifact_sha256"]
        != allocation["candidate_rounds_artifact_sha256"]
        or selection["candidate_rounds_bound_into_frozen_profile"] is not True
        or selection["profile_frozen_before_full_fit_replay"] is not True
        or selection["outer_A_dev_observed_during_selection"] is not False
        or selection["contains_residual_factor_tensors"] is not False
    ):
        raise ValueError("global-gated selection binding drifted")

    candidate = value["candidate"]
    candidate_fields = {
        "candidate_id",
        "full_fit_parent_A_fit",
        "full_fit_parent_bank_artifact_sha256",
        "bank_artifact_sha256",
        "pre_outer_diagnostic_bank_artifact_sha256",
        "post_outer_diagnostic_bank_artifact_sha256",
        "allocation_profile",
        "selection_candidate_rounds_artifact_sha256",
        "frozen_profile_artifact_sha256",
        "fixed_profile_replay",
        "parent_bank_parameters",
        "allocated_residual_parameters",
        "maximum_allocated_residual_parameters",
        "generator_bank_parameters",
        "hypothetical_atlas_parameters",
        "hypothetical_router_parameters",
        "hypothetical_total_parameters",
        "native_attention_parameters",
        "hypothetical_parameter_reduction_fraction",
        "deployment_router_built",
        "contains_native_value_or_output_weights",
        "contains_oracle_outputs_or_coefficients",
        "contains_raw_teacher_outputs_or_gradients",
        "contains_residual_factor_tensors",
        "executable_artifact_published",
    }
    if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
        raise ValueError("global-gated candidate schema drifted")
    full_parent = _sealed_runner()._validate_parent_fit(
        candidate["full_fit_parent_A_fit"],
        receipt_count=_OUTER_FIT_RECEIPT_COUNT,
    )
    sequence = allocation["frozen_allocation_sequence"]
    replay = _validate_fixed_profile_replay(
        candidate["fixed_profile_replay"],
        frozen_sequence=sequence,
        frozen_profile_sha256=allocation["frozen_profile_artifact_sha256"],
        frozen_candidate_rounds_sha256=allocation[
            "candidate_rounds_artifact_sha256"
        ],
    )
    component_count = len(sequence)
    bank_parameters = 113_728 + component_count * _RESIDUAL_COMPONENT_PARAMETERS
    hypothetical_total = 23_041 + 28_800 + bank_parameters
    if (
        candidate["candidate_id"]
        != "k8_r8_stable_split_global_gated_residual_budget2_A_fit_diagnostic"
        or candidate["full_fit_parent_bank_artifact_sha256"]
        != full_parent["bank_artifact_sha256"]
        or replay["final_bank"]["parent_site_rank_one_bank_sha256"]
        != full_parent["bank_artifact_sha256"]
        or candidate["bank_artifact_sha256"]
        != replay["final_bank"]["artifact_sha256"]
        or candidate["pre_outer_diagnostic_bank_artifact_sha256"]
        != candidate["bank_artifact_sha256"]
        or candidate["post_outer_diagnostic_bank_artifact_sha256"]
        != candidate["bank_artifact_sha256"]
        or candidate["allocation_profile"] != sequence
        or candidate["selection_candidate_rounds_artifact_sha256"]
        != allocation["candidate_rounds_artifact_sha256"]
        or candidate["frozen_profile_artifact_sha256"]
        != allocation["frozen_profile_artifact_sha256"]
        or candidate["parent_bank_parameters"] != 113_728
        or candidate["allocated_residual_parameters"]
        != component_count * _RESIDUAL_COMPONENT_PARAMETERS
        or candidate["maximum_allocated_residual_parameters"]
        != _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
        or candidate["generator_bank_parameters"] != bank_parameters
        or candidate["hypothetical_atlas_parameters"] != 23_041
        or candidate["hypothetical_router_parameters"] != 28_800
        or candidate["hypothetical_total_parameters"] != hypothetical_total
        or candidate["native_attention_parameters"] != 14_745_600
        or not _close(
            candidate["hypothetical_parameter_reduction_fraction"],
            1.0 - hypothetical_total / 14_745_600,
        )
        or any(
            candidate[key] is not False
            for key in (
                "deployment_router_built",
                "contains_native_value_or_output_weights",
                "contains_oracle_outputs_or_coefficients",
                "contains_raw_teacher_outputs_or_gradients",
                "contains_residual_factor_tensors",
                "executable_artifact_published",
            )
        )
    ):
        raise ValueError("global-gated candidate binding drifted")
    outer = _sealed_runner()._validate_outer_diagnostic(
        value["outer_A_dev_diagnostic"]
    )
    if (
        outer["parent_bank_sha256"]
        != candidate["full_fit_parent_bank_artifact_sha256"]
        or outer["allocated_bank_sha256"] != candidate["bank_artifact_sha256"]
        or outer["allocation_profile"] != sequence
        or outer["summary"]["current_run_outer_panel_previously_exposed_exactly"]
        is not False
    ):
        raise ValueError("global-gated outer diagnostic lineage drifted")

    counters = value["runtime_counters"]
    counter_fields = {
        "role_batch_counts",
        "chart_backbone_forward_count",
        "chart_terminal_cotangent_vjp_count",
        "dense_parity_smoke_count",
        "parity_backbone_forward_count",
        "parity_terminal_cotangent_vjp_count",
        "captured_example_count",
        "supplemental_route_receipt_count",
        "canonical_gradient_targets_per_vjp",
        "auxiliary_gradient_targets_per_vjp",
        "gradient_targets_per_vjp",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations",
        "contains_raw_gradients",
        "contains_logits",
    }
    if (
        not isinstance(counters, Mapping)
        or set(counters) != counter_fields
        or counters["role_batch_counts"]
        != [
            ["A_chart_transport_fit", 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ]
        or counters["chart_backbone_forward_count"] != 45
        or counters["chart_terminal_cotangent_vjp_count"] != 45
        or counters["dense_parity_smoke_count"] != 1
        or counters["parity_backbone_forward_count"] != 1
        or counters["parity_terminal_cotangent_vjp_count"] != 1
        or counters["captured_example_count"] != 48
        or counters["supplemental_route_receipt_count"] != 864
        or counters["canonical_gradient_targets_per_vjp"] != 73
        or counters["auxiliary_gradient_targets_per_vjp"] != 19
        or counters["gradient_targets_per_vjp"] != 92
        or any(
            counters[key] is not False
            for key in (
                "contains_prompt_text",
                "contains_token_ids",
                "contains_raw_activations",
                "contains_raw_gradients",
                "contains_logits",
            )
        )
    ):
        raise ValueError("global-gated runtime counters drifted")
    resources = value["runtime_resources"]
    if (
        not isinstance(resources, Mapping)
        or set(resources)
        != {
            "prepublication_peak_resident_set_bytes",
            "resident_set_cap_bytes",
            "publication_guarded",
        }
        or type(resources["prepublication_peak_resident_set_bytes"]) is not int
        or resources["prepublication_peak_resident_set_bytes"] <= 0
        or resources["resident_set_cap_bytes"] != 8 * 1024**3
        or resources["prepublication_peak_resident_set_bytes"]
        > resources["resident_set_cap_bytes"]
        or resources["publication_guarded"] is not True
    ):
        raise ValueError("global-gated runtime resource binding drifted")
    without_hash = dict(value)
    report_sha = without_hash.pop("report_sha256")
    if not _is_sha256(report_sha) or report_sha != _digest(without_hash):
        raise ValueError("global-gated report digest drifted")
    return value


def _seal_report(value: Mapping[str, object]) -> dict[str, object]:
    report = dict(value)
    if "report_sha256" in report:
        raise ValueError("unsealed global-gated report already has a digest")
    report["report_sha256"] = _digest(report)
    return _validate_report(report)


def _strict_report(path: Path) -> dict[str, object]:
    raw = _read_no_follow_regular_bytes(path)
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("global-gated report JSON is invalid") from error
    if _canonical_json_bytes(value) != raw:
        raise ValueError("global-gated report must use canonical JSON")
    return _validate_report(value)


def _publish_report(
    path: Path,
    report: Mapping[str, object],
    *,
    publication_guard: Callable[[str], None],
) -> None:
    _validate_report(dict(report))
    if path.suffix != ".json":
        raise ValueError("global-gated output must use .json")
    if not callable(publication_guard):
        raise TypeError("publication_guard must be callable")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_real_directory(path.parent)
    if _entry_exists(path):
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    metadata = os.fstat(descriptor)
    owner = (metadata.st_dev, metadata.st_ino)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        publication_guard("global_gated_temporary_sealed")
        source = temporary.lstat()
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
            or (source.st_dev, source.st_ino) != owner
        ):
            raise RuntimeError("global-gated temporary identity drifted")
        os.link(temporary, path, follow_symlinks=False)
        published = True
        target = path.lstat()
        if (
            stat.S_ISLNK(target.st_mode)
            or not stat.S_ISREG(target.st_mode)
            or (target.st_dev, target.st_ino) != owner
        ):
            raise RuntimeError("global-gated publication identity drifted")
        if not _unlink_if_owned(temporary, owner):
            raise RuntimeError("global-gated temporary ownership drifted")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _strict_report(path) != dict(report):
            raise RuntimeError("global-gated strict publication replay drifted")
        verified = path.lstat()
        if (
            stat.S_ISLNK(verified.st_mode)
            or not stat.S_ISREG(verified.st_mode)
            or verified.st_nlink != 1
            or (verified.st_dev, verified.st_ino) != owner
        ):
            raise RuntimeError("global-gated verified identity drifted")
        verified_identity = _file_identity(verified)
        publication_guard("global_gated_publication_verified")
        if _strict_report(path) != dict(report):
            raise RuntimeError("global-gated final publication replay drifted")
        final = path.lstat()
        if (
            stat.S_ISLNK(final.st_mode)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != owner
            or _file_identity(final) != verified_identity
        ):
            raise RuntimeError("global-gated report changed during final guard")
    except BaseException:
        _unlink_if_owned(temporary, owner)
        removed_owned = False
        if published:
            try:
                target = path.lstat()
                if (target.st_dev, target.st_ino) == owner:
                    path.unlink()
                    removed_owned = True
            except FileNotFoundError:
                pass
        if removed_owned:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        raise


def run_global_gated_residual_spectrum_A_diagnostic(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run one fail-closed stable-split, A-only global-gated diagnostic."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("global-gated output must use .json")
    _require_real_directory(output.parent)
    if _entry_exists(output):
        raise FileExistsError(output)

    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        FIT_ROLE,
        _capture_batch_plan,
        observe_phase_resources,
        role_panel_binding_sha256,
        static_preflight,
        validate_runtime_qualification,
    )

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    _validate_predecessor(preflight.binding.repository_root)
    source_hashes = _source_sha256s(preflight.binding.repository_root)

    from fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum import (
        ALL_LAYER_RESIDUAL_ACCEPTANCE_SEMANTICS,
        ALL_LAYER_RESIDUAL_AMPLITUDE_SEMANTICS,
        ALL_LAYER_RESIDUAL_PROPOSAL_SEMANTICS,
        AllLayerResidualSpectrumConfig,
        evaluate_allocated_residual_bank_all_layers,
        replay_fixed_residual_allocation_profile,
    )
    from fisher_graph.gemma3_attention_transport_experiment import (
        ROWS_PER_BATCH_AND_INVOCATION,
        _capture_role,
        _check_output_publication_headroom,
        _check_resident_set_budget,
    )
    from fisher_graph.gemma3_attention_transport_experiment_core import (
        _open_split_panel,
    )
    from fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum import (
        DEFAULT_GLOBAL_RELATIVE_FISHER_THRESHOLD,
        GLOBAL_GATED_RESIDUAL_GREEDY_SEMANTICS,
        GLOBAL_GATED_RESIDUAL_LEDGER_SEMANTICS,
        GLOBAL_GATED_RESIDUAL_OBJECTIVE_SEMANTICS,
        greedily_allocate_global_gated_residual_spectrum,
    )
    from fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum_partition import (
        build_stable_A_batch_catalog,
        inner_split_artifact_sha256,
        outer_split_artifact_sha256,
        partition_stable_A_capture_records,
    )
    from fisher_graph.gemma3_attention_transport_global_site_rank1 import (
        GlobalSiteRankOneAttentionTransportFitConfig,
        fit_global_site_rank_one_attention_transport_generator_bank,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    parent_config = GlobalSiteRankOneAttentionTransportFitConfig.fixed_k8_r8_top2()
    spectrum_config = AllLayerResidualSpectrumConfig()
    if (
        spectrum_config.generator_count != 8
        or spectrum_config.generator_rank != 8
        or spectrum_config.active_count != 2
        or spectrum_config.allocation_budget != _ALLOCATION_BUDGET
        or spectrum_config.coefficient_ridge != 0.0
        or spectrum_config.amplitude_ridge != 0.0
        or not math.isclose(
            DEFAULT_GLOBAL_RELATIVE_FISHER_THRESHOLD,
            _GLOBAL_RELATIVE_FISHER_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError("global-gated fixed config drifted")
    role_manifest = preflight.fresh_manifest["roles"][FIT_ROLE]
    if role_manifest.get("file_sha256") != _PINNED_A_CORPUS_FILE_SHA256:
        raise RuntimeError("pinned A corpus file identity drifted")
    panel_binding = role_panel_binding_sha256(preflight.fresh_manifest, FIT_ROLE)
    if panel_binding != _PINNED_A_ROLE_PANEL_BINDING_SHA256:
        raise RuntimeError("pinned A role panel binding drifted")
    catalog = build_stable_A_batch_catalog(
        role_panel_binding_sha256=panel_binding,
        ordered_members=role_manifest["ordered_members"],
    )
    if (
        catalog.batch_plan_artifact_sha256 != _PINNED_A_BATCH_PLAN_SHA256
        or outer_split_artifact_sha256(catalog) != _PINNED_A_OUTER_SPLIT_SHA256
        or inner_split_artifact_sha256(catalog) != _PINNED_A_INNER_SPLIT_SHA256
    ):
        raise RuntimeError("pinned stable A split drifted")

    panel_source = SequentialPanelSource(
        repository_root=preflight.binding.repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    sink_state: dict[str, object] = {"active_core_role": None, "records": []}

    def sink(captured: object) -> None:
        records = sink_state["records"]
        if sink_state["active_core_role"] != "A" or not isinstance(records, list):
            raise RuntimeError("global-gated sink was called outside A")
        sampled = sample_attention_transport_captured_invocation(
            captured,
            maximum_rows=ROWS_PER_BATCH_AND_INVOCATION,
        )
        records.append(replace(sampled, role="A"))

    runtime: object | None = None
    peak_resident_set_bytes = 0

    def guard_resources(phase_id: str) -> None:
        nonlocal peak_resident_set_bytes
        observation = observe_phase_resources(
            preflight.binding,
            phase_id=f"attention_transport_global_gated_{phase_id}",
        )
        _check_output_publication_headroom(preflight.binding, phase_id=phase_id)
        peak_resident_set_bytes = max(
            peak_resident_set_bytes,
            _check_resident_set_budget(
                preflight.binding,
                lambda: int(observation.peak_resident_set_bytes),
                phase_id=phase_id,
            ),
        )

    def capture_progress(role: str, completed: int, total: int) -> None:
        if progress is not None:
            progress(
                {
                    "phase": f"{role}_capture",
                    "completed": completed,
                    "total": total,
                }
            )

    def spectrum_progress(
        phase: str,
        completed: int,
        total: int,
        invocation_id: str | None,
    ) -> None:
        guard_resources(f"{phase}_{completed:02d}")
        if progress is not None:
            progress(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "invocation_id": invocation_id,
                }
            )

    def parent_round_observer(
        phase: str,
    ) -> Callable[[int, int, object, object], None]:
        def observe(
            round_index: int,
            total: int,
            evaluation: object,
            global_evaluation: object,
        ) -> None:
            guard_resources(f"{phase}_round_{round_index:02d}")
            if progress is not None:
                progress(
                    {
                        "phase": phase,
                        "completed": round_index,
                        "total": total,
                        "all_A_relative_fisher_loss": float(
                            getattr(evaluation, "relative_fisher_loss")
                        ),
                        "global_attention_relative_fisher_loss": float(
                            getattr(global_evaluation, "relative_fisher_loss")
                        ),
                    }
                )

        return observe

    try:
        runtime = build_pinned_gemma3_attention_transport_capture_runtime(
            preflight,
            sink=sink,
            cache_dir=cache_dir,
        )
        qualification = runtime.runtime_qualification_receipt()
        validate_runtime_qualification(preflight=preflight, receipt=qualification)
        guard_resources("runtime_qualified")
        records, dense_parity = _capture_role(
            runtime=runtime,
            preflight=preflight,
            panel_source=panel_source,
            runtime_role=FIT_ROLE,
            core_role="A",
            sink_state=sink_state,
            capture_batch_plan=_capture_batch_plan,
            run_dense_parity=True,
            progress=capture_progress,
            resource_guard=guard_resources,
        )
        if dense_parity is None:
            raise RuntimeError("global-gated dense parity receipt is absent")
        panel = _open_split_panel(records, role="A")
        if not _is_sha256(panel.artifact_sha256):
            raise RuntimeError("global-gated opened A panel binding is invalid")
        partition = partition_stable_A_capture_records(records, catalog=catalog)
        if (
            partition.outer_split_artifact_sha256
            != _PINNED_A_OUTER_SPLIT_SHA256
            or partition.inner_split_artifact_sha256
            != _PINNED_A_INNER_SPLIT_SHA256
        ):
            raise RuntimeError("runtime stable A partition drifted")
        outer_fit_captures = tuple(
            record.capture for record in partition.outer_fit_records
        )
        outer_dev_captures = tuple(
            record.capture for record in partition.outer_dev_records
        )
        direction_captures = tuple(
            record.capture for record in partition.direction_train_records
        )
        allocation_validation_captures = tuple(
            record.capture for record in partition.allocation_validation_records
        )

        guard_resources("before_inner_parent_fit")
        inner_parent_fit = fit_global_site_rank_one_attention_transport_generator_bank(
            direction_captures,
            config=parent_config,
            round_observer=parent_round_observer(
                "global_gated_inner_parent_fit"
            ),
        )
        guard_resources("inner_parent_fit_complete")
        allocation = greedily_allocate_global_gated_residual_spectrum(
            inner_parent_fit.bank,
            direction_captures,
            allocation_validation_captures,
            config=spectrum_config,
            global_relative_fisher_threshold=_GLOBAL_RELATIVE_FISHER_THRESHOLD,
            progress_callback=spectrum_progress,
        )
        selection_payload = _global_gated_allocation_payload(allocation)
        frozen_sequence = tuple(allocation.allocation_profile)
        frozen_profile_sha = selection_payload["frozen_profile_artifact_sha256"]
        guard_resources("inner_allocation_profile_frozen")

        full_parent_fit = fit_global_site_rank_one_attention_transport_generator_bank(
            outer_fit_captures,
            config=parent_config,
            round_observer=parent_round_observer(
                "global_gated_full_parent_fit"
            ),
        )
        guard_resources("full_fit_parent_complete")
        replay = replay_fixed_residual_allocation_profile(
            full_parent_fit.bank,
            outer_fit_captures,
            frozen_sequence,
            config=spectrum_config,
            progress_callback=spectrum_progress,
        )
        replay_payload = _fixed_profile_replay_payload(
            replay,
            frozen_profile_artifact_sha256=str(frozen_profile_sha),
            selection_candidate_rounds_artifact_sha256=str(
                selection_payload["candidate_rounds_artifact_sha256"]
            ),
        )
        bank = replay.final_bank
        state = bank.state_dict()
        if (
            tuple(bank.allocated_invocation_ids) != frozen_sequence
            or state.get("contains_native_value_or_output_weights") is not False
            or state.get("contains_oracle_outputs_or_coefficients") is not False
            or state.get("contains_raw_teacher_outputs_or_gradients") is not False
        ):
            raise RuntimeError("global-gated replay retention or lineage drifted")
        pre_outer_bank_sha = bank.artifact_sha256
        full_parent_sha = full_parent_fit.bank.artifact_sha256
        outer_diagnostic = evaluate_allocated_residual_bank_all_layers(
            full_parent_fit.bank,
            bank,
            outer_dev_captures,
            config=spectrum_config,
            progress_callback=spectrum_progress,
        )
        outer_payload = _outer_diagnostic_payload(outer_diagnostic)
        post_outer_bank_sha = bank.artifact_sha256
        if (
            post_outer_bank_sha != pre_outer_bank_sha
            or full_parent_fit.bank.artifact_sha256 != full_parent_sha
        ):
            raise RuntimeError("global-gated outer diagnostic mutated a fitted bank")
        guard_resources("outer_A_dev_evaluation_complete")
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.next_role != FIT_ROLE
            or panel_source.freeze_receipts
        ):
            raise RuntimeError("global-gated diagnostic advanced beyond A")
        counters = runtime.counters.payload()
        runtime.close()
        runtime = None
        guard_resources("runtime_closed")

        def guard_publication(phase_id: str) -> None:
            guard_resources(phase_id)
            if _source_sha256s(preflight.binding.repository_root) != source_hashes:
                raise RuntimeError("global-gated implementation sources changed")
            _validate_predecessor(preflight.binding.repository_root)

        guard_publication("before_publication")
        sealed_peak = peak_resident_set_bytes
        component_count = len(frozen_sequence)
        bank_parameters = bank.stored_parameter_count
        hypothetical_total = 23_041 + 28_800 + bank_parameters
        partition_report = _partition_payload(partition)
        partition_report["final_firewall_state"] = {
            "opened_roles": [FIT_ROLE],
            "current_role": FIT_ROLE,
            "next_role": FIT_ROLE,
            "freeze_receipt_count": 0,
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
        }
        report = _seal_report(
            {
                "schema": REPORT_SCHEMA,
                "format_version": 1,
                "scientific_status": _expected_scientific_status(),
                "exposure_binding": _expected_exposure_binding(),
                "model": {
                    "model_id": str(getattr(qualification, "model_id")),
                    "model_revision": str(
                        getattr(qualification, "model_revision")
                    ),
                    "hidden_width": int(getattr(qualification, "hidden_width")),
                    "runtime_qualification_receipt_sha256": str(
                        getattr(qualification, "qualification_receipt_sha256")
                    ),
                },
                "protocol": {
                    "protocol_id": (
                        "A_only_stable_split_global_gated_residual_nested_30_6_"
                        "outer_36_9_v1"
                    ),
                    "outer_fit_batch_count": _OUTER_FIT_RECEIPT_COUNT,
                    "outer_dev_batch_count": _OUTER_DEV_RECEIPT_COUNT,
                    "direction_train_batch_count": _DIRECTION_TRAIN_RECEIPT_COUNT,
                    "allocation_validation_batch_count": (
                        _ALLOCATION_VALIDATION_RECEIPT_COUNT
                    ),
                    "outer_fit_row_count": _OUTER_FIT_ROWS,
                    "outer_dev_row_count": _OUTER_DEV_ROWS,
                    "direction_train_row_count": _DIRECTION_TRAIN_ROWS,
                    "allocation_validation_row_count": (
                        _ALLOCATION_VALIDATION_ROWS
                    ),
                    "global_direction_train_row_count": (
                        _GLOBAL_DIRECTION_TRAIN_ROWS
                    ),
                    "global_allocation_validation_row_count": (
                        _GLOBAL_ALLOCATION_VALIDATION_ROWS
                    ),
                    "allocation_budget": _ALLOCATION_BUDGET,
                    "global_relative_fisher_threshold": (
                        _GLOBAL_RELATIVE_FISHER_THRESHOLD
                    ),
                    "B_and_C_capture_forbidden": True,
                    "A_corpus_file_sha256": _PINNED_A_CORPUS_FILE_SHA256,
                    "A_role_panel_binding_sha256": (
                        _PINNED_A_ROLE_PANEL_BINDING_SHA256
                    ),
                    "A_batch_plan_artifact_sha256": (
                        _PINNED_A_BATCH_PLAN_SHA256
                    ),
                    "A_outer_split_artifact_sha256": (
                        _PINNED_A_OUTER_SPLIT_SHA256
                    ),
                    "A_inner_split_artifact_sha256": (
                        _PINNED_A_INNER_SPLIT_SHA256
                    ),
                    "outer_partition_policy": (
                        "fixed_manifest_batch_identity_ordinals_36_fit_9_dev_v1"
                    ),
                    "inner_partition_policy": (
                        "fixed_outer_fit_batch_identity_ordinals_30_direction_6_"
                        "validation_v1"
                    ),
                    "parent_bank_fit": {
                        "generator_count": 8,
                        "generator_rank": 8,
                        "active_count": 2,
                        "site_rank_one_alternating_rounds": (
                            parent_config.alternating_rounds
                        ),
                        "site_rank_one_steps_per_site_per_round": (
                            parent_config.steps_per_site_per_round
                        ),
                        "fit_oracle_coefficient_ridge": 1.0e-8,
                        "fresh_fit_for_inner_and_full_panels": True,
                        "learned_objects_fit_role": "A_only",
                    },
                    "global_gated_residual_spectrum": {
                        "generator_count": spectrum_config.generator_count,
                        "generator_rank": spectrum_config.generator_rank,
                        "active_count": spectrum_config.active_count,
                        "euclidean_floor": float(spectrum_config.euclidean_floor),
                        "coefficient_ridge": float(
                            spectrum_config.coefficient_ridge
                        ),
                        "amplitude_ridge": float(spectrum_config.amplitude_ridge),
                        "power_iterations": spectrum_config.power_iterations,
                        "deterministic_start_count": (
                            spectrum_config.deterministic_start_count
                        ),
                        "allocation_budget": spectrum_config.allocation_budget,
                        "residual_component_parameters": (
                            _RESIDUAL_COMPONENT_PARAMETERS
                        ),
                        "maximum_allocated_residual_parameters": (
                            _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
                        ),
                        "gauge_floor": float(spectrum_config.gauge_floor),
                        "minimum_relative_fisher_gain": float(
                            spectrum_config.minimum_relative_fisher_gain
                        ),
                        "tie_tolerance": float(spectrum_config.tie_tolerance),
                        "global_relative_fisher_threshold": (
                            _GLOBAL_RELATIVE_FISHER_THRESHOLD
                        ),
                        "gate_decision_panel": (
                            "inner_allocation_validation_only"
                        ),
                        "global_candidate_invocation_ids": list(
                            _GLOBAL_ATTENTION_INVOCATION_IDS
                        ),
                        "objective_semantics": (
                            GLOBAL_GATED_RESIDUAL_OBJECTIVE_SEMANTICS
                        ),
                        "greedy_semantics": (
                            GLOBAL_GATED_RESIDUAL_GREEDY_SEMANTICS
                        ),
                        "ledger_semantics": (
                            GLOBAL_GATED_RESIDUAL_LEDGER_SEMANTICS
                        ),
                        "allocation_can_stop_before_budget": True,
                        "candidate_rounds_bound_into_frozen_profile": True,
                        "candidate_rounds_contain_factor_tensors": False,
                    },
                    "outer_evaluation": {
                        "evaluation_role": (
                            "stable_outer_A_dev_descriptive_replay_only"
                        ),
                        "A_corpus_previously_exposed": True,
                        "used_for_current_run_selection_or_fitting": False,
                        "fresh_holdout": False,
                        "confirmation_readout": False,
                        "evaluation_after_profile_freeze_and_full_fit_replay": True,
                        "parent_and_allocated_evaluated_on_same_outer_rows": True,
                        "all_local_global_and_per_layer_metrics": True,
                        "coefficient_ridge": 0.0,
                        "oracle_ceiling_only": True,
                        "evaluation_call_count": 1,
                    },
                    "preflight_receipt_sha256": str(preflight.receipt_sha256),
                    "dense_lexical_parity_receipt_sha256": dense_parity,
                    "implementation_source_sha256s": source_hashes,
                },
                "A_partition": partition_report,
                "selection": {
                    "selection_id": (
                        "stable_inner30_direction_inner6_global_gated_budget2_v1"
                    ),
                    "inner_parent_A_fit": _parent_fit_payload(inner_parent_fit),
                    "global_gated_allocation": selection_payload,
                    "candidate_rounds_artifact_sha256": selection_payload[
                        "candidate_rounds_artifact_sha256"
                    ],
                    "candidate_rounds_bound_into_frozen_profile": True,
                    "profile_frozen_before_full_fit_replay": True,
                    "outer_A_dev_observed_during_selection": False,
                    "contains_residual_factor_tensors": False,
                },
                "candidate": {
                    "candidate_id": (
                        "k8_r8_stable_split_global_gated_residual_budget2_A_fit_"
                        "diagnostic"
                    ),
                    "full_fit_parent_A_fit": _parent_fit_payload(full_parent_fit),
                    "full_fit_parent_bank_artifact_sha256": full_parent_sha,
                    "bank_artifact_sha256": bank.artifact_sha256,
                    "pre_outer_diagnostic_bank_artifact_sha256": (
                        pre_outer_bank_sha
                    ),
                    "post_outer_diagnostic_bank_artifact_sha256": (
                        post_outer_bank_sha
                    ),
                    "allocation_profile": list(frozen_sequence),
                    "selection_candidate_rounds_artifact_sha256": (
                        selection_payload["candidate_rounds_artifact_sha256"]
                    ),
                    "frozen_profile_artifact_sha256": frozen_profile_sha,
                    "fixed_profile_replay": replay_payload,
                    "parent_bank_parameters": 113_728,
                    "allocated_residual_parameters": (
                        component_count * _RESIDUAL_COMPONENT_PARAMETERS
                    ),
                    "maximum_allocated_residual_parameters": (
                        _MAXIMUM_ALLOCATED_RESIDUAL_PARAMETERS
                    ),
                    "generator_bank_parameters": bank_parameters,
                    "hypothetical_atlas_parameters": 23_041,
                    "hypothetical_router_parameters": 28_800,
                    "hypothetical_total_parameters": hypothetical_total,
                    "native_attention_parameters": 14_745_600,
                    "hypothetical_parameter_reduction_fraction": (
                        1.0 - hypothetical_total / 14_745_600
                    ),
                    "deployment_router_built": False,
                    "contains_native_value_or_output_weights": False,
                    "contains_oracle_outputs_or_coefficients": False,
                    "contains_raw_teacher_outputs_or_gradients": False,
                    "contains_residual_factor_tensors": False,
                    "executable_artifact_published": False,
                },
                "outer_A_dev_diagnostic": outer_payload,
                "runtime_counters": counters,
                "runtime_resources": {
                    "prepublication_peak_resident_set_bytes": sealed_peak,
                    "resident_set_cap_bytes": int(
                        preflight.binding.storage_budget.maximum_resident_memory_bytes
                    ),
                    "publication_guarded": True,
                },
            }
        )
        guard_publication("report_sealed")
        _publish_report(
            output,
            report,
            publication_guard=guard_publication,
        )
        summary = report["outer_A_dev_diagnostic"]["summary"]
        selected = report["selection"]["global_gated_allocation"]
        return {
            "status": "A_only_stable_split_global_gated_residual_completed",
            "B_opened_by_this_run": False,
            "C_ever_opened": False,
            "confirmation_claimed": False,
            "report_sha256": report["report_sha256"],
            "allocation_profile": report["candidate"]["allocation_profile"],
            "global_gate_passed_on_inner_validation": selected[
                "global_gate_passed"
            ],
            "final_inner_global_relative_fisher_loss": selected[
                "final_global_allocation_validation_oracle"
            ]["relative_fisher_loss"],
            "outer_global_parent_minus_allocated_relative_fisher_loss": summary[
                "global_parent_minus_allocated_relative_fisher_loss"
            ],
        }
    finally:
        if runtime is not None:
            runtime.close()


def check_global_gated_residual_spectrum_A_diagnostic(
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    """Independently verify a published stable-split global-gated report."""

    report = _strict_report(Path(report_path))
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    _validate_predecessor(root)
    if _source_sha256s(root) != report["protocol"][
        "implementation_source_sha256s"
    ]:
        raise ValueError("global-gated sources drifted from report")
    allocation = report["selection"]["global_gated_allocation"]
    summary = report["outer_A_dev_diagnostic"]["summary"]
    return {
        "status": "A_only_stable_split_global_gated_residual_verified",
        "B_opened_by_this_run": False,
        "C_ever_opened": False,
        "stable_cross_run_A_split_identity_enforced": True,
        "execution_receipts_drive_partition": False,
        "outer_A_dev_readout_descriptive_replay_only": True,
        "fresh_holdout_claimed": False,
        "confirmation_claimed": False,
        "report_sha256": report["report_sha256"],
        "allocation_profile": report["candidate"]["allocation_profile"],
        "allocation_budget_exhausted": allocation["budget_exhausted"],
        "global_gate_passed_on_inner_validation": allocation[
            "global_gate_passed"
        ],
        "final_inner_global_relative_fisher_loss": allocation[
            "final_global_allocation_validation_oracle"
        ]["relative_fisher_loss"],
        "outer_all_layer_parent_minus_allocated_relative_fisher_loss": summary[
            "all_layer_parent_minus_allocated_relative_fisher_loss"
        ],
        "outer_local_parent_minus_allocated_relative_fisher_loss": summary[
            "local_parent_minus_allocated_relative_fisher_loss"
        ],
        "outer_global_parent_minus_allocated_relative_fisher_loss": summary[
            "global_parent_minus_allocated_relative_fisher_loss"
        ],
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the A-only stable-split global-gated residual diagnostic"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    report = _resolve(root, arguments.report)
    if arguments.check:
        print(
            json.dumps(
                check_global_gated_residual_spectrum_A_diagnostic(
                    report,
                    repository_root=root,
                ),
                sort_keys=True,
            )
        )
        return 0

    temporary = _resolve(
        root,
        arguments.temp_directory
        if arguments.temp_directory is not None
        else DEFAULT_TEMP_DIRECTORY,
    )
    report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        build_default_execution_binding,
    )

    binding = build_default_execution_binding(
        repository_root=root,
        output_directory=report.parent,
        temp_directory=temporary,
    )

    def progress(value: Mapping[str, object]) -> None:
        print(json.dumps(dict(value), sort_keys=True), flush=True)

    result = run_global_gated_residual_spectrum_A_diagnostic(
        binding=binding,
        report_path=report,
        cache_dir=arguments.cache_dir,
        progress=progress,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
