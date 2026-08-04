"""Render the partial V20q nested token-VJP validation as deterministic SVG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_SUMMARY = Path("artifacts/research/v20q_partial_validation_v1.json")
DEFAULT_OUTPUT = Path("docs/images/v20q-partial-validation.svg")


@dataclass(frozen=True)
class V20qFoldProgress:
    short_label: str
    outer_family_id: str
    source_path: str
    source_file_sha256: str
    fragment_sha256: str
    incumbent_feature_id: str
    incumbent_b: float
    incumbent_a: float
    selected_candidate_id: str
    selected_candidate_role: str
    selected_feature_id: str
    selected_b: float
    selected_a: float
    selected_inner_oof_kl: float
    incumbent_inner_oof_kl: float
    inner_delta_kl: float
    strict_inner_family_wins: int
    outer_candidate_kl: float
    outer_incumbent_kl: float
    outer_delta_kl: float
    continuous_inner_better: bool
    exact_output_differs: bool
    strict_outer_win: bool


@dataclass(frozen=True)
class V20qProgressData:
    as_of: str
    status: str
    required_fold_count: int
    inner_fold_count: int
    candidate_count: int
    fit_candidate_count: int
    feature_ids: tuple[str, ...]
    completed_fold_count: int
    remaining_fold_count: int
    continuous_count: int
    continuous_required: int
    continuous_needed: int
    different_count: int
    different_required: int
    different_needed: int
    outer_win_count: int
    outer_win_required: int
    outer_wins_needed: int
    cumulative_outer_delta_kl: float
    runtime_parameter_delta: int
    runtime_mac_delta_per_token: int
    folds: tuple[V20qFoldProgress, ...]
    interpretation: str
    next_rung: str


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _number(value: object, path: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{path} must be {qualifier}")
    return result


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    number = _number(value, path)
    result = int(number)
    if result != number or result < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return result


def _sha256(value: object, path: str) -> str:
    result = _string(value, path)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _same_float(left: float, right: float) -> bool:
    return left.hex() == right.hex()


def extract_v20q_progress_data(summary: Mapping[str, object]) -> V20qProgressData:
    """Validate the source-safe partial-campaign summary used by the figure."""

    if summary.get("format_version") != 1:
        raise ValueError("summary.format_version must be 1")
    if summary.get("schema") != "fisher_graph.v20q_partial_validation_summary":
        raise ValueError(
            "summary.schema must be fisher_graph.v20q_partial_validation_summary"
        )
    protocol = _object(summary.get("protocol"), "summary.protocol")
    ledger = _object(summary.get("gate_ledger"), "summary.gate_ledger")
    claims = _object(summary.get("claim_boundary"), "summary.claim_boundary")

    required = _integer(
        protocol.get("required_outer_fold_count"),
        "summary.protocol.required_outer_fold_count",
        minimum=1,
    )
    inner_folds = _integer(
        protocol.get("inner_family_fold_count_per_outer"),
        "summary.protocol.inner_family_fold_count_per_outer",
        minimum=1,
    )
    candidate_count = _integer(
        protocol.get("candidate_count_per_inner_fold"),
        "summary.protocol.candidate_count_per_inner_fold",
        minimum=1,
    )
    fit_candidate_count = _integer(
        protocol.get("fit_candidate_count_per_inner_fold"),
        "summary.protocol.fit_candidate_count_per_inner_fold",
        minimum=1,
    )
    feature_ids = tuple(
        _string(value, f"summary.protocol.feature_ids[{index}]")
        for index, value in enumerate(
            _array(protocol.get("feature_ids"), "summary.protocol.feature_ids")
        )
    )
    if (
        required != 8
        or inner_folds != 7
        or candidate_count != 174
        or fit_candidate_count != 168
        or feature_ids != ("c1", "c2", "c1_times_c2", "source_z")
        or protocol.get("compiler_scope")
        != "fit_only_token_vjp_refit_into_unchanged_v20p_runtime"
        or protocol.get("exact_float64_full_vocabulary_teacher_kl") is not True
        or protocol.get("outer_family_used_for_fit_or_selection") is not False
    ):
        raise ValueError("summary.protocol differs from the frozen V20q screen")

    runtime_parameter_delta = _integer(
        protocol.get("runtime_parameter_delta_vs_v20p"),
        "summary.protocol.runtime_parameter_delta_vs_v20p",
    )
    runtime_mac_delta = _integer(
        protocol.get("runtime_mac_delta_per_token_vs_v20p"),
        "summary.protocol.runtime_mac_delta_per_token_vs_v20p",
    )
    if runtime_parameter_delta != 0 or runtime_mac_delta != 0:
        raise ValueError("V20q must preserve the unchanged V20p runtime")

    folds: list[V20qFoldProgress] = []
    seen_labels: set[str] = set()
    seen_families: set[str] = set()
    for index, value in enumerate(_array(summary.get("folds"), "summary.folds")):
        row = _object(value, f"summary.folds[{index}]")
        short_label = _string(row.get("short_label"), f"summary.folds[{index}].short_label")
        family = _string(row.get("outer_family_id"), f"summary.folds[{index}].outer_family_id")
        if short_label in seen_labels or family in seen_families:
            raise ValueError("summary.folds labels and families must be unique")
        seen_labels.add(short_label)
        seen_families.add(family)
        inner_delta = _number(
            row.get("inner_candidate_minus_incumbent_kl"),
            f"summary.folds[{index}].inner_candidate_minus_incumbent_kl",
            nonnegative=False,
        )
        outer_delta = _number(
            row.get("outer_candidate_minus_incumbent_kl"),
            f"summary.folds[{index}].outer_candidate_minus_incumbent_kl",
            nonnegative=False,
        )
        inner_candidate = _number(
            row.get("selected_inner_oof_kl"),
            f"summary.folds[{index}].selected_inner_oof_kl",
        )
        inner_incumbent = _number(
            row.get("incumbent_inner_oof_kl"),
            f"summary.folds[{index}].incumbent_inner_oof_kl",
        )
        outer_candidate = _number(
            row.get("outer_candidate_kl"),
            f"summary.folds[{index}].outer_candidate_kl",
        )
        outer_incumbent = _number(
            row.get("outer_incumbent_kl"),
            f"summary.folds[{index}].outer_incumbent_kl",
        )
        continuous = _boolean(
            row.get("continuous_inner_better"),
            f"summary.folds[{index}].continuous_inner_better",
        )
        differs = _boolean(
            row.get("exact_output_differs"),
            f"summary.folds[{index}].exact_output_differs",
        )
        outer_win = _boolean(
            row.get("strict_outer_win"),
            f"summary.folds[{index}].strict_outer_win",
        )
        role = _string(
            row.get("selected_candidate_role"),
            f"summary.folds[{index}].selected_candidate_role",
        )
        selected_feature_id = _string(
            row.get("selected_feature_id"),
            f"summary.folds[{index}].selected_feature_id",
        )
        incumbent_feature_id = _string(
            row.get("incumbent_feature_id"),
            f"summary.folds[{index}].incumbent_feature_id",
        )
        strict_inner_family_wins = _integer(
            row.get("strict_inner_family_wins"),
            f"summary.folds[{index}].strict_inner_family_wins",
        )
        if (
            not _same_float(inner_delta, inner_candidate - inner_incumbent)
            or not _same_float(outer_delta, outer_candidate - outer_incumbent)
            or continuous != (role == "token_vjp_fit" and inner_delta < 0.0)
            or outer_win != (outer_delta < 0.0)
            or (
                role == "v20p_incumbent"
                and (inner_delta != 0.0 or differs)
            )
            or role
            not in {
                "v20p_incumbent",
                "exact_anchor",
                "smooth_seed",
                "token_vjp_fit",
            }
            or selected_feature_id not in feature_ids
            or incumbent_feature_id not in feature_ids
            or strict_inner_family_wins > inner_folds
        ):
            raise ValueError(f"summary.folds[{index}] decision arithmetic differs")
        folds.append(
            V20qFoldProgress(
                short_label=short_label,
                outer_family_id=family,
                source_path=_string(row.get("source_path"), f"summary.folds[{index}].source_path"),
                source_file_sha256=_sha256(
                    row.get("source_file_sha256"),
                    f"summary.folds[{index}].source_file_sha256",
                ),
                fragment_sha256=_sha256(
                    row.get("fragment_sha256"),
                    f"summary.folds[{index}].fragment_sha256",
                ),
                incumbent_feature_id=incumbent_feature_id,
                incumbent_b=_number(
                    row.get("incumbent_b"),
                    f"summary.folds[{index}].incumbent_b",
                    nonnegative=False,
                ),
                incumbent_a=_number(
                    row.get("incumbent_a"),
                    f"summary.folds[{index}].incumbent_a",
                    nonnegative=False,
                ),
                selected_candidate_id=_string(
                    row.get("selected_candidate_id"),
                    f"summary.folds[{index}].selected_candidate_id",
                ),
                selected_candidate_role=role,
                selected_feature_id=selected_feature_id,
                selected_b=_number(
                    row.get("selected_b"),
                    f"summary.folds[{index}].selected_b",
                    nonnegative=False,
                ),
                selected_a=_number(
                    row.get("selected_a"),
                    f"summary.folds[{index}].selected_a",
                    nonnegative=False,
                ),
                selected_inner_oof_kl=inner_candidate,
                incumbent_inner_oof_kl=inner_incumbent,
                inner_delta_kl=inner_delta,
                strict_inner_family_wins=strict_inner_family_wins,
                outer_candidate_kl=outer_candidate,
                outer_incumbent_kl=outer_incumbent,
                outer_delta_kl=outer_delta,
                continuous_inner_better=continuous,
                exact_output_differs=differs,
                strict_outer_win=outer_win,
            )
        )

    completed = _integer(
        ledger.get("completed_outer_fold_count"),
        "summary.gate_ledger.completed_outer_fold_count",
    )
    remaining = _integer(
        ledger.get("remaining_outer_fold_count"),
        "summary.gate_ledger.remaining_outer_fold_count",
    )
    continuous_count = _integer(
        ledger.get("continuous_inner_better_count"),
        "summary.gate_ledger.continuous_inner_better_count",
    )
    continuous_required = _integer(
        ledger.get("continuous_inner_better_required"),
        "summary.gate_ledger.continuous_inner_better_required",
    )
    continuous_needed = _integer(
        ledger.get("continuous_inner_better_needed_from_remaining"),
        "summary.gate_ledger.continuous_inner_better_needed_from_remaining",
    )
    different_count = _integer(
        ledger.get("exact_output_difference_count"),
        "summary.gate_ledger.exact_output_difference_count",
    )
    different_required = _integer(
        ledger.get("exact_output_difference_required"),
        "summary.gate_ledger.exact_output_difference_required",
    )
    different_needed = _integer(
        ledger.get("exact_output_difference_needed_from_remaining"),
        "summary.gate_ledger.exact_output_difference_needed_from_remaining",
    )
    outer_win_count = _integer(
        ledger.get("strict_outer_win_count"),
        "summary.gate_ledger.strict_outer_win_count",
    )
    outer_win_required = _integer(
        ledger.get("strict_outer_win_required"),
        "summary.gate_ledger.strict_outer_win_required",
    )
    outer_wins_needed = _integer(
        ledger.get("strict_outer_wins_needed_from_remaining"),
        "summary.gate_ledger.strict_outer_wins_needed_from_remaining",
    )
    cumulative_delta = _number(
        ledger.get("cumulative_outer_candidate_minus_incumbent_kl"),
        "summary.gate_ledger.cumulative_outer_candidate_minus_incumbent_kl",
        nonnegative=False,
    )
    expected_continuous = sum(fold.continuous_inner_better for fold in folds)
    expected_different = sum(fold.exact_output_differs for fold in folds)
    expected_outer_wins = sum(fold.strict_outer_win for fold in folds)
    expected_delta = math.fsum(fold.outer_delta_kl for fold in folds)
    one_fallback_closes_both = (
        completed < required
        and continuous_needed == remaining
        and different_needed == remaining
    )
    if (
        completed != len(folds)
        or completed + remaining != required
        or continuous_count != expected_continuous
        or different_count != expected_different
        or outer_win_count != expected_outer_wins
        or continuous_needed != max(0, continuous_required - continuous_count)
        or different_needed != max(0, different_required - different_count)
        or outer_wins_needed != max(0, outer_win_required - outer_win_count)
        or not _same_float(cumulative_delta, expected_delta)
        or ledger.get("macro_outer_kl_gate")
        != "final_eight_fold_mean_must_be_strictly_negative"
        or ledger.get(
            "one_additional_fallback_makes_continuous_and_output_gates_impossible"
        )
        is not one_fallback_closes_both
    ):
        raise ValueError("summary.gate_ledger does not replay from completed folds")
    structurally_reachable = (
        continuous_needed <= remaining
        and different_needed <= remaining
        and outer_wins_needed <= remaining
    )
    complete = completed == required
    development_passed = (
        complete
        and continuous_count >= continuous_required
        and different_count >= different_required
        and outer_win_count >= outer_win_required
        and cumulative_delta < 0.0
    )
    if complete:
        expected_status = (
            "complete_development_gate_passed"
            if development_passed
            else "complete_development_gate_failed"
        )
    else:
        expected_status = (
            "incomplete_still_mathematically_reachable"
            if structurally_reachable
            else "incomplete_gate_unreachable"
        )
    if summary.get("status") != expected_status:
        raise ValueError("summary.status differs from the replayed campaign state")
    if (
        claims.get("campaign_complete") is not complete
        or claims.get("development_gate_decided") is not complete
        or any(
            claims.get(key) is not False
            for key in (
                "fresh_validation_claim_authorized",
                "serving_claim_authorized",
                "compression_claim_authorized",
                "speed_claim_authorized",
                "model_weights_or_prompt_text_serialized",
            )
        )
    ):
        raise ValueError("summary.claim_boundary must remain closed")

    return V20qProgressData(
        as_of=_string(summary.get("as_of"), "summary.as_of"),
        status=_string(summary.get("status"), "summary.status"),
        required_fold_count=required,
        inner_fold_count=inner_folds,
        candidate_count=candidate_count,
        fit_candidate_count=fit_candidate_count,
        feature_ids=feature_ids,
        completed_fold_count=completed,
        remaining_fold_count=remaining,
        continuous_count=continuous_count,
        continuous_required=continuous_required,
        continuous_needed=continuous_needed,
        different_count=different_count,
        different_required=different_required,
        different_needed=different_needed,
        outer_win_count=outer_win_count,
        outer_win_required=outer_win_required,
        outer_wins_needed=outer_wins_needed,
        cumulative_outer_delta_kl=cumulative_delta,
        runtime_parameter_delta=runtime_parameter_delta,
        runtime_mac_delta_per_token=runtime_mac_delta,
        folds=tuple(folds),
        interpretation=_string(summary.get("interpretation"), "summary.interpretation"),
        next_rung=_string(
            summary.get("next_rung_if_gate_becomes_unreachable"),
            "summary.next_rung_if_gate_becomes_unreachable",
        ),
    )


def verify_available_fold_sources(
    data: V20qProgressData, *, source_root: Path
) -> tuple[str, ...]:
    """Replay summary fields from ignored fold files when they are available."""

    verified: list[str] = []
    for fold in data.folds:
        relative = Path(fold.source_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("fold source path must stay below source_root")
        path = source_root / relative
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != fold.source_file_sha256:
            raise ValueError(f"fold source file hash differs: {fold.source_path}")
        value = _object(json.loads(payload), f"fold source {fold.source_path}")
        if value.get("fragment_sha256") != fold.fragment_sha256:
            raise ValueError(f"fold fragment hash differs: {fold.source_path}")
        fold_receipt = _object(
            value.get("fold_receipt"),
            f"fold source {fold.source_path}.fold_receipt",
        )
        candidate = _object(
            value.get("candidate"),
            f"fold source {fold.source_path}.candidate",
        )
        incumbent = _object(
            value.get("inherited_v20p_field_selection_receipt"),
            f"fold source {fold.source_path}.inherited_v20p_field_selection_receipt",
        )
        selection = _object(
            value.get("selection_receipt"),
            f"fold source {fold.source_path}.selection_receipt",
        )
        aggregates = _object(
            selection.get("aggregate_by_candidate"),
            f"fold source {fold.source_path}.selection_receipt.aggregate_by_candidate",
        )
        selected_aggregate = _object(
            aggregates.get(fold.selected_candidate_id),
            f"fold source {fold.source_path}.selected_aggregate",
        )
        incumbent_aggregate = _object(
            aggregates.get("v20q_v20p_incumbent"),
            f"fold source {fold.source_path}.incumbent_aggregate",
        )
        held = _object(
            value.get("held_evidence"),
            f"fold source {fold.source_path}.held_evidence",
        )

        exact_fields = (
            (value.get("outer_held_family_id"), fold.outer_family_id),
            (fold_receipt.get("outer_held_family_id"), fold.outer_family_id),
            (held.get("outer_held_family_id"), fold.outer_family_id),
            (incumbent.get("selected_feature_id"), fold.incumbent_feature_id),
            (candidate.get("candidate_id"), fold.selected_candidate_id),
            (fold_receipt.get("selected_candidate_id"), fold.selected_candidate_id),
            (selection.get("selected_candidate_id"), fold.selected_candidate_id),
            (held.get("selected_candidate_id"), fold.selected_candidate_id),
            (candidate.get("candidate_role"), fold.selected_candidate_role),
            (fold_receipt.get("selected_candidate_role"), fold.selected_candidate_role),
            (candidate.get("feature_id"), fold.selected_feature_id),
            (fold_receipt.get("feature_id"), fold.selected_feature_id),
            (
                selected_aggregate.get("strict_inner_family_wins_over_incumbent"),
                fold.strict_inner_family_wins,
            ),
            (
                fold_receipt.get("selected_nonzero_continuous_candidate") is True
                and fold_receipt.get("selected_inner_oof_mean_beats_incumbent")
                is True,
                fold.continuous_inner_better,
            ),
            (
                fold_receipt.get(
                    "candidate_exact_output_differs_from_v20p_incumbent"
                ),
                fold.exact_output_differs,
            ),
            (
                held.get("candidate_exact_output_differs_from_v20p_incumbent"),
                fold.exact_output_differs,
            ),
            (
                fold_receipt.get("candidate_strictly_beats_v20p_incumbent"),
                fold.strict_outer_win,
            ),
            (
                held.get("candidate_strictly_beats_v20p_incumbent"),
                fold.strict_outer_win,
            ),
        )
        float_fields = (
            (incumbent.get("selected_b"), fold.incumbent_b),
            (incumbent.get("selected_a"), fold.incumbent_a),
            (candidate.get("b"), fold.selected_b),
            (candidate.get("a"), fold.selected_a),
            (fold_receipt.get("b"), fold.selected_b),
            (fold_receipt.get("a"), fold.selected_a),
            (
                selected_aggregate.get("family_equal_exact_kl"),
                fold.selected_inner_oof_kl,
            ),
            (
                incumbent_aggregate.get("family_equal_exact_kl"),
                fold.incumbent_inner_oof_kl,
            ),
            (
                fold_receipt.get("selected_inner_oof_mean"),
                fold.selected_inner_oof_kl,
            ),
            (
                fold_receipt.get("incumbent_inner_oof_mean"),
                fold.incumbent_inner_oof_kl,
            ),
            (fold_receipt.get("candidate_objective"), fold.outer_candidate_kl),
            (held.get("candidate_objective"), fold.outer_candidate_kl),
            (held.get("v20p_incumbent_objective"), fold.outer_incumbent_kl),
        )
        if any(actual != expected for actual, expected in exact_fields) or any(
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not _same_float(float(actual), expected)
            for actual, expected in float_fields
        ):
            raise ValueError(
                f"fold summary differs from authenticated source: {fold.source_path}"
            )
        verified.append(fold.source_path)
    return tuple(verified)


def _micro(value: float) -> str:
    scaled = value * 1_000_000.0
    if scaled == 0.0:
        return "0.000"
    sign = "−" if scaled < 0.0 else "+"
    return f"{sign}{abs(scaled):.3f}"


def render_v20q_progress(
    data: V20qProgressData, *, source_sha256: str, source_label: str
) -> str:
    """Return an accessible SVG summarizing partial fold results and open gates."""

    width = 1200
    height = 760
    chart_left = 300.0
    chart_right = 610.0
    zero_x = (chart_left + chart_right) / 2.0
    maximum_micro = max(
        (
            10.0,
            *(abs(fold.inner_delta_kl) * 1_000_000.0 for fold in data.folds),
            *(abs(fold.outer_delta_kl) * 1_000_000.0 for fold in data.folds),
        )
    )

    def delta_x(delta: float) -> float:
        micro = delta * 1_000_000.0
        return zero_x + (micro / maximum_micro) * (
            (chart_right - chart_left) / 2.0
        )

    def compact(value: float) -> str:
        return f"{value:.5g}".replace("-", "−")

    continuous_folds = tuple(
        fold for fold in data.folds if fold.continuous_inner_better
    )
    rollback_folds = tuple(
        fold
        for fold in data.folds
        if fold.selected_candidate_role == "v20p_incumbent"
    )
    continuous_names = ", ".join(fold.short_label for fold in continuous_folds)
    rollback_names = ", ".join(fold.short_label for fold in rollback_folds)
    description = (
        f"{data.completed_fold_count} of {data.required_fold_count} outer folds "
        f"are complete. {data.continuous_count} selected a continuous inner-better "
        f"fit, {data.different_count} changed exact output, and "
        f"{data.outer_win_count} strictly improved outer KL."
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">V20q partial nested token-VJP validation</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f'<!-- Generated from {escape(source_label)} sha256={escape(source_sha256)} -->',
        "<style>",
        ".bg{fill:#f8fafc}.panel{fill:#fff;stroke:#dbe3ec;stroke-width:1.5}.title{font:700 28px ui-sans-serif,system-ui,sans-serif;fill:#0f172a}.subtitle{font:400 15px ui-sans-serif,system-ui,sans-serif;fill:#475569}.section{font:700 17px ui-sans-serif,system-ui,sans-serif;fill:#0f172a}.label{font:600 14px ui-sans-serif,system-ui,sans-serif;fill:#1e293b}.small{font:400 12px ui-sans-serif,system-ui,sans-serif;fill:#64748b}.value{font:600 13px ui-monospace,SFMono-Regular,monospace;fill:#1e293b}.axis{stroke:#94a3b8;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.inner{fill:#60a5fa}.outer{fill:#22c55e}.tie{fill:#94a3b8}.future{fill:#fef3c7;stroke:#f59e0b;stroke-width:1}.miss{fill:#e2e8f0}.hit{fill:#22c55e}.footer{font:500 14px ui-sans-serif,system-ui,sans-serif;fill:#334155}.warning{font:600 14px ui-sans-serif,system-ui,sans-serif;fill:#9a3412}.source{font:400 10px ui-monospace,SFMono-Regular,monospace;fill:#64748b}",
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}"/>',
        f'<text class="title" x="48" y="54">V20q token-VJP validation: partial {data.completed_fold_count}/{data.required_fold_count}</text>',
        '<text class="subtitle" x="48" y="82">Compiler-only refit into the unchanged V20p runtime · exact full-vocabulary teacher KL</text>',
        f'<text class="subtitle" x="1152" y="54" text-anchor="end">as of {escape(data.as_of)}</text>',
        '<rect class="panel" x="32" y="108" width="670" height="400" rx="14"/>',
        '<text class="section" x="56" y="142">Candidate − incumbent KL by completed outer fold</text>',
        '<text class="small" x="56" y="166">µKL (×10⁻⁶); negative is better</text>',
        '<rect class="inner" x="405" y="153" width="14" height="8" rx="2"/>',
        '<text class="small" x="425" y="161">inner OOF</text>',
        '<rect class="outer" x="500" y="153" width="14" height="8" rx="2"/>',
        '<text class="small" x="520" y="161">outer held</text>',
    ]
    for tick_index in range(-2, 3):
        tick = maximum_micro * tick_index / 2.0
        x = zero_x + (tick / maximum_micro) * (
            (chart_right - chart_left) / 2.0
        )
        tick_label = f"{tick:.0f}" if tick.is_integer() else f"{tick:.1f}"
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="184" x2="{x:.1f}" y2="455"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="478" text-anchor="middle">{tick_label}</text>')
    lines.append(f'<line class="axis" x1="{zero_x:.1f}" y1="180" x2="{zero_x:.1f}" y2="458"/>')

    if not data.folds:
        row_y = ()
    elif len(data.folds) == 1:
        row_y = (320.0,)
    else:
        row_step = min(90.0, 230.0 / (len(data.folds) - 1))
        row_y = tuple(215.0 + index * row_step for index in range(len(data.folds)))
    for fold, y in zip(data.folds, row_y, strict=True):
        lines.append(f'<text class="label" x="56" y="{y + 4:.1f}">{escape(fold.short_label)}</text>')
        outcome = (
            f'{fold.selected_feature_id} fit · {fold.strict_inner_family_wins}/{data.inner_fold_count}'
            if fold.continuous_inner_better
            else (
                "rollback"
                if fold.selected_candidate_role == "v20p_incumbent"
                else fold.selected_candidate_role.replace("_", " ")
            )
        )
        lines.append(f'<text class="small" x="118" y="{y + 4:.1f}">{escape(outcome)}</text>')
        inner_x = delta_x(fold.inner_delta_kl)
        outer_x = delta_x(fold.outer_delta_kl)
        inner_width = abs(zero_x - inner_x)
        outer_width = abs(zero_x - outer_x)
        if inner_width > 0.0:
            lines.append(
                f'<rect class="inner" x="{min(inner_x, zero_x):.1f}" y="{y - 12:.1f}" width="{inner_width:.1f}" height="10" rx="3"/>'
            )
        else:
            lines.append(f'<circle class="tie" cx="{zero_x:.1f}" cy="{y - 7:.1f}" r="4"/>')
        if outer_width > 0.0:
            lines.append(
                f'<rect class="outer" x="{min(outer_x, zero_x):.1f}" y="{y + 3:.1f}" width="{outer_width:.1f}" height="10" rx="3"/>'
            )
        else:
            lines.append(f'<circle class="tie" cx="{zero_x:.1f}" cy="{y + 8:.1f}" r="4"/>')
        lines.append(
            f'<text class="value" x="688" y="{y - 2:.1f}" text-anchor="end">{_micro(fold.inner_delta_kl)}</text>'
        )
        lines.append(
            f'<text class="value" x="688" y="{y + 13:.1f}" text-anchor="end">{_micro(fold.outer_delta_kl)}</text>'
        )

    lines.extend(
        [
            '<rect class="panel" x="722" y="108" width="446" height="400" rx="14"/>',
            f'<text class="section" x="746" y="142">Gate ledger after {data.completed_fold_count}/{data.required_fold_count} folds</text>',
            '<text class="small" x="746" y="166">Each square is one outer family; yellow folds are unrun</text>',
        ]
    )
    gates = (
        (
            "Continuous + inner-better",
            tuple(fold.continuous_inner_better for fold in data.folds),
            data.continuous_count,
            data.continuous_required,
            data.continuous_needed,
        ),
        (
            "Exact output difference",
            tuple(fold.exact_output_differs for fold in data.folds),
            data.different_count,
            data.different_required,
            data.different_needed,
        ),
        (
            "Strict outer KL win",
            tuple(fold.strict_outer_win for fold in data.folds),
            data.outer_win_count,
            data.outer_win_required,
            data.outer_wins_needed,
        ),
    )
    gate_row_y = (230.0, 320.0, 410.0)
    for (label, outcomes, observed, required_count, needed), y in zip(
        gates, gate_row_y, strict=True
    ):
        lines.append(f'<text class="label" x="746" y="{y - 24:.1f}">{escape(label)}</text>')
        for index in range(data.required_fold_count):
            x = 746.0 + index * 43.0
            css = "future"
            if index < len(outcomes):
                css = "hit" if outcomes[index] else "miss"
            lines.append(f'<rect class="{css}" x="{x:.1f}" y="{y - 6:.1f}" width="32" height="24" rx="4"/>')
            lines.append(f'<text class="small" x="{x + 16:.1f}" y="{y + 11:.1f}" text-anchor="middle">{index + 1}</text>')
        lines.append(
            f'<text class="value" x="746" y="{y + 42:.1f}">{observed}/{required_count} observed · {needed}/{data.remaining_fold_count} remaining required</text>'
        )

    cumulative = _micro(data.cumulative_outer_delta_kl)
    if len(continuous_folds) == 1:
        winning_fold = continuous_folds[0]
        continuous_summary = (
            f"{winning_fold.short_label}: {winning_fold.selected_feature_id} field "
            f"(b={compact(winning_fold.selected_b)}, "
            f"a={compact(winning_fold.selected_a)}), "
            f"{winning_fold.strict_inner_family_wins}/{data.inner_fold_count} "
            f"inner wins, outer delta {_micro(winning_fold.outer_delta_kl)} µKL."
        )
    else:
        continuous_summary = (
            f"Continuous inner-better selections: {len(continuous_folds)}"
            + (f" ({continuous_names})." if continuous_names else ".")
        )
    rollback_summary = (
        f"Exact V20p rollbacks: {len(rollback_folds)}"
        + (f" ({rollback_names})." if rollback_names else ".")
    )
    if data.remaining_fold_count:
        warning = (
            f"Incomplete: {data.remaining_fold_count} folds remain; need "
            f"{data.continuous_needed} continuous inner-better, "
            f"{data.different_needed} output-distinct, and "
            f"{data.outer_wins_needed} strict outer wins."
        )
    else:
        warning = "Complete development ledger; see the authenticated report for the decision."
    lines.extend(
        [
            '<line class="grid" x1="48" y1="540" x2="1152" y2="540"/>',
            '<text class="section" x="48" y="578">What the completed folds establish</text>',
            f'<text class="footer" x="48" y="610">{escape(continuous_summary)}</text>',
            f'<text class="footer" x="48" y="638">{escape(rollback_summary)}</text>',
            f'<text class="footer" x="48" y="666">Cumulative outer delta: {cumulative} µKL. Serving delta vs V20p: 0 parameters and 0 MACs/token.</text>',
            f'<text class="warning" x="48" y="702">{escape(warning)}</text>',
            f'<text class="source" x="48" y="735">source: {escape(source_label)} · sha256 {escape(source_sha256)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def render_summary_file(
    summary_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    verify_sources: bool = True,
) -> None:
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if not isinstance(summary, Mapping):
        raise ValueError("V20q progress summary must be an object")
    data = extract_v20q_progress_data(summary)
    if verify_sources:
        verify_available_fold_sources(data, source_root=source_root)
    source_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    try:
        source_label = summary_path.relative_to(source_root).as_posix()
    except ValueError:
        source_label = summary_path.as_posix()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_v20q_progress(
            data,
            source_sha256=source_sha256,
            source_label=source_label,
        ),
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the partial V20q nested token-VJP validation SVG"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--skip-source-verification", action="store_true")
    arguments = parser.parse_args(argv)
    render_summary_file(
        arguments.input,
        arguments.output,
        source_root=arguments.source_root,
        verify_sources=not arguments.skip_source_verification,
    )
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
