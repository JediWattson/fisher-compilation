from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as runner,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_informed_factorial as factorial,
)
from fisher_graph.gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionBasis,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
)


_TAIL_RANK = 17
_MAX_RANK = 320
_WIDTH = 320
_A16_COMPLETE_H4_SUPPORT_ROWS = 819


def _receipt(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _global_unweighted_u320() -> CompleteH4ProjectionBasis:
    eigenvalues = tuple(float(_MAX_RANK - index) for index in range(_MAX_RANK))
    return CompleteH4ProjectionBasis(
        width=_WIDTH,
        max_rank=_MAX_RANK,
        basis_rows=torch.eye(_WIDTH, dtype=torch.float64),
        residual_eigenvalues=eigenvalues,
        residual_energy_fractions=tuple(
            1.0 / _MAX_RANK for _ in range(_MAX_RANK)
        ),
        directional_residual_variance=eigenvalues,
        next_residual_eigenvalue=0.0,
        cutoff_spectral_gap=1.0,
        source_example_ids=("synthetic-a16",),
        source_family_ids=("synthetic-family",),
        source_sequence_sha256s=(_receipt("synthetic-a16-sequence"),),
        fit_weighting="unweighted",
    )


class _FakeTailInformedFit:
    """Minimal immutable-fit protocol used by the runner integration tests."""

    tail_rank = _TAIL_RANK
    max_rank = _MAX_RANK
    artifact_sha256 = _receipt("fake-tail-informed-fit")

    def __init__(self) -> None:
        self._basis = torch.eye(_WIDTH, dtype=torch.float64).contiguous()

    def validate_integrity(self) -> bool:
        assert self._basis.dtype == torch.float64
        assert self._basis.device.type == "cpu"
        assert self._basis.is_contiguous()
        return True

    def basis_tensor(self, rank: int) -> torch.Tensor:
        if type(rank) is not int or not 1 <= rank <= self.max_rank:
            raise ValueError("rank lies outside the fake treatment fit")
        return self._basis[:rank].clone().contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "test.fake_complete_h4_tail_informed_fit",
            "artifact_sha256": self.artifact_sha256,
            "tail_rank": self.tail_rank,
            "max_rank": self.max_rank,
            "ordering": COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
        }

    def lineage(
        self,
        rank: int,
        execution_basis_artifact_sha256: str,
    ) -> dict[str, object]:
        return {
            "fit_artifact_sha256": self.artifact_sha256,
            "rank": rank,
            "tail_rank": self.tail_rank,
            "ordering": COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
            "prefix_artifact_sha256": _receipt(f"fake-prefix-{rank}"),
            "execution_basis_artifact_sha256": (
                execution_basis_artifact_sha256
            ),
            "lineage_sha256": _receipt(
                f"fake-lineage-{rank}-{execution_basis_artifact_sha256}"
            ),
        }


def _factorial_comparisons(
    *,
    global_passes: tuple[bool, bool, bool, bool],
    treatment_passes: tuple[bool, bool, bool, bool],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    branches = (
        ("unweighted", (192, 224, 256, 320), global_passes),
        (
            "tail_informed",
            (192 + _TAIL_RANK, 224, 256, 320),
            treatment_passes,
        ),
    )
    for prefix, ranks, pass_pattern in branches:
        for rank, passed in zip(ranks, pass_pattern, strict=True):
            rows[f"{prefix}.rank{rank}"] = {
                "rank": rank,
                "factorial_capacity_gate_passed": passed,
            }
    return rows


def _u192_correction_receipt(
    example_id: str,
    *,
    containing_fit: str,
) -> dict[str, object]:
    prefix = {
        "schema": "fisher_graph.complete_h4_fit_to_prefix_lineage",
        "format_version": 1,
        "arm_id": "unweighted.rank192",
        "fit_basis_artifact_sha256": _receipt(
            f"{containing_fit}-fit-artifact"
        ),
        "fit_basis_matrix_sha256": _receipt(
            f"{containing_fit}-fit-matrix"
        ),
        "fit_weighting": "unweighted",
        "fit_max_rank": 192 if containing_fit == "u192" else 320,
        "residual_width": 640,
        "prefix_rank": 192,
        "prefix_definition": (
            "first_rank_rows_of_fit_basis_in_residual_eigenvalue_order"
        ),
        "execution_ordering": "descending_unweighted_residual_eigenvalue",
        "execution_basis_sha256": _receipt("u192-basis-tensor"),
        "execution_basis_artifact_sha256": (
            runner._EXPECTED_UNWEIGHTED_RANK192_EXECUTION_BASIS_ARTIFACT_SHA256
        ),
        "fit_to_prefix_lineage_sha256": _receipt(
            f"{containing_fit}-{example_id}-prefix-lineage"
        ),
    }
    arm = {
        "role": "projection_oracle",
        "execution_mode": "authenticated_complete_h4_correction_arm",
        "projection_rank": 192,
        "metrics_only": True,
        "serving_authorized": False,
        "model_forward_count": 1,
        "injected_h4_sha256": _receipt(f"{example_id}-injected"),
        "native_h4_sha256": _receipt(f"{example_id}-native"),
        "incomplete_h4_sha256": _receipt(f"{example_id}-incomplete"),
        "projected_delta_sha256": _receipt(f"{example_id}-delta"),
        "projection_basis_sha256": _receipt("u192-basis-tensor"),
        "projection_basis_artifact_sha256": (
            runner._EXPECTED_UNWEIGHTED_RANK192_EXECUTION_BASIS_ARTIFACT_SHA256
        ),
        "projection_fit_basis_artifact_sha256": _receipt(
            f"{containing_fit}-fit-artifact"
        ),
        "projection_ordering": "descending_unweighted_residual_eigenvalue",
        "projection_definition": (
            "cpu_float64_residual_matmul_D_transpose_matmul_D_cast_once"
        ),
        "projection_basis_orthonormal_max_abs_error": 3.5e-15,
        "complete_h4_pair_artifact_sha256": _receipt(f"{example_id}-pair"),
        "shadow_result_artifact_sha256": _receipt(f"{example_id}-shadow"),
        "runtime_binding_sha256": _receipt("runtime"),
        "model_inputs_sha256": _receipt(f"{example_id}-inputs"),
        "execution_grid_sha256": _receipt(f"{example_id}-grid"),
        "adapter_execution_sha256": _receipt("adapter"),
        "complete_h4_support_mask_sha256": _receipt(
            f"{example_id}-support"
        ),
        "boundary_callback_order": (
            "complete_h4_correction.y3",
            "complete_h4_correction.x4",
            "complete_h4_correction.h4",
        ),
        "logits_bitwise_authoritative": False,
        "max_abs_authoritative_logit_error": 1.25,
        "logits_sha256": _receipt(f"{example_id}-logits"),
        "artifact_sha256": _receipt(
            f"{containing_fit}-{example_id}-arm-artifact"
        ),
    }
    example_number = int(example_id.rsplit("-", 1)[1])
    return {
        "example_id": example_id,
        "family_id": f"family-{example_number // 2}",
        "prompt_sha256": _receipt(f"{example_id}-prompt"),
        "model_inputs_sha256": _receipt(f"{example_id}-inputs"),
        "execution_grid_sha256": _receipt(f"{example_id}-grid"),
        "complete_h4_support_rows": 32,
        "complete_h4_padding_write_rows": 0,
        "fit_to_prefix_before": dict(prefix),
        "fit_to_prefix_after": dict(prefix),
        "arm": arm,
        "receipt_sha256": _receipt(
            f"{containing_fit}-{example_id}-receipt"
        ),
    }


def _u192_receipt_panels(
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    parent = [
        _u192_correction_receipt(
            f"a-fit-{index:03d}",
            containing_fit="u192",
        )
        for index in range(16)
    ]
    live = [
        _u192_correction_receipt(
            f"a-fit-{index:03d}",
            containing_fit="u320",
        )
        for index in range(16)
    ]
    return parent, live


def test_builds_locked_global_and_tail_informed_eight_arm_factorial() -> None:
    global_specs = runner._build_projection_arm_specs(
        {"unweighted": _global_unweighted_u320()},
        ranks=(192, 224, 256, 320),
        weightings=("unweighted",),
    )
    tail_fit = _FakeTailInformedFit()
    treatment_specs = runner._build_tail_informed_projection_arm_specs(
        tail_fit,
        ranks=(192 + tail_fit.tail_rank, 224, 256, 320),
    )
    specs = (*global_specs, *treatment_specs)

    assert tuple(spec.arm_id for spec in specs) == (
        "unweighted.rank192",
        "unweighted.rank224",
        "unweighted.rank256",
        "unweighted.rank320",
        "tail_informed.rank209",
        "tail_informed.rank224",
        "tail_informed.rank256",
        "tail_informed.rank320",
    )
    assert len(specs) == len({spec.arm_id for spec in specs}) == 8
    assert all(spec.fit_weighting == "unweighted" for spec in specs)
    assert all(
        spec.execution_ordering
        == COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
        for spec in treatment_specs
    )
    assert all(
        spec.projection_fit_artifact_sha256 == tail_fit.artifact_sha256
        for spec in treatment_specs
    )

    rows = torch.arange(2 * _WIDTH, dtype=torch.float64).reshape(2, _WIDTH)
    for spec in specs:
        lineage = runner._validate_any_projection_arm_spec(spec)
        assert lineage["prefix_rank"] == spec.rank
        projected = runner._project_projection_arm_rows(rows, spec)
        expected = (rows @ spec.execution_basis.T) @ spec.execution_basis
        assert torch.equal(projected, expected)


def test_locked_rank_sum_and_a16_projection_mac_count() -> None:
    ranks = (
        192,
        224,
        256,
        320,
        192 + _TAIL_RANK,
        224,
        256,
        320,
    )

    assert sum(ranks) == 2001
    assert runner._WIDTH == 640
    logical_projection_macs = (
        2 * runner._WIDTH * _A16_COMPLETE_H4_SUPPORT_ROWS * sum(ranks)
    )
    assert logical_projection_macs == 2_097_688_320

    resources = runner._expected_resources(prompt_count=16, arm_count=8)
    assert resources == {
        "collect_model_forward_count": 80,
        "evaluation_shadow_model_forward_count": 48,
        "projection_arm_model_forward_count": 128,
        "exact_h4_ceiling_model_forward_count": 16,
        "evaluation_model_forward_count": 192,
        "total_model_forward_count": 272,
        "backward_count": 16,
    }


def test_u192_parent_receipts_allow_only_containing_fit_lineage_drift() -> None:
    parent, live = _u192_receipt_panels()

    regression = runner._validate_u192_parent_invariant_receipts(
        parent_receipts=parent,
        live_receipts=list(reversed(live)),
    )

    assert regression["matched"] is True
    assert regression["prompt_count"] == 16
    assert len(regression["invariant_receipts_sha256"]) == 64
    excluded = regression[
        "excluded_only_because_u192_fit_became_u320_fit"
    ]
    assert excluded == {
        "top_level": ("receipt_sha256",),
        "fit_to_prefix": (
            "fit_basis_artifact_sha256",
            "fit_basis_matrix_sha256",
            "fit_max_rank",
            "fit_to_prefix_lineage_sha256",
        ),
        "arm": (
            "projection_fit_basis_artifact_sha256",
            "artifact_sha256",
        ),
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("complete_h4_support_rows",), 31),
        (("fit_to_prefix_before", "execution_basis_sha256"), "00" * 32),
        (("arm", "projected_delta_sha256"), "11" * 32),
        (("arm", "max_abs_authoritative_logit_error"), 1.5),
        (("arm", "logits_sha256"), "22" * 32),
    ),
)
def test_u192_parent_receipts_reject_any_invariant_prompt_tamper(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    parent, live = _u192_receipt_panels()
    changed = copy.deepcopy(live)
    target: dict[str, object] = changed[0]
    for name in path[:-1]:
        nested = target[name]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = replacement

    with pytest.raises(RuntimeError, match="per-prompt invariant receipts differ"):
        runner._validate_u192_parent_invariant_receipts(
            parent_receipts=parent,
            live_receipts=changed,
        )


def test_u192_parent_receipts_reject_unclassified_schema_drift() -> None:
    parent, live = _u192_receipt_panels()
    live[0]["unexpected_field"] = True

    with pytest.raises(ValueError, match="closed U192 policy"):
        runner._validate_u192_parent_invariant_receipts(
            parent_receipts=parent,
            live_receipts=live,
        )


def test_factorial_arm_is_only_a_transfer_candidate_until_stable_selection() -> None:
    raw = {
        "unweighted.rank192": {
            "rank": 192,
            "later_lofo_fitting_authorized": True,
        }
    }

    normalized = runner._normalize_tail_informed_factorial_comparisons(raw)
    arm = normalized["unweighted.rank192"]

    assert raw["unweighted.rank192"]["later_lofo_fitting_authorized"] is True
    assert arm["factorial_capacity_gate_passed"] is True
    assert (
        arm[
            "frozen_basis_one_pass_carrier_transfer_candidate_if_stable_suffix"
        ]
        is True
    )
    assert arm["later_lofo_fitting_authorized"] is False
    assert "frozen_basis_one_pass_carrier_transfer_oracle_authorized" not in arm


def test_factorial_selection_requires_stable_suffix_and_is_order_invariant() -> None:
    comparisons = _factorial_comparisons(
        # The isolated rank-192 pass is invalidated by rank 224.  The global
        # branch therefore stabilizes only at 256, while treatment stabilizes
        # at 224 and wins despite failing its smaller rank-209 arm.
        global_passes=(True, False, True, True),
        treatment_passes=(False, True, True, True),
    )

    forward = runner._select_tail_informed_factorial(
        comparisons,
        tail_rank=_TAIL_RANK,
    )
    reverse = runner._select_tail_informed_factorial(
        dict(reversed(comparisons.items())),
        tail_rank=_TAIL_RANK,
    )

    assert forward == reverse
    assert forward["per_branch_smallest_stable_passing_arm"] == {
        "global_unweighted": "unweighted.rank256",
        "tail_informed": "tail_informed.rank224",
    }
    assert forward["overall_stable_passing_rank"] == 224
    assert forward["selected_arm"] == "tail_informed.rank224"
    assert forward["factorial_capacity_gate_passed"] is True
    assert (
        forward[
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized"
        ]
        is True
    )
    assert forward["later_lofo_fitting_authorized"] is False
    assert forward["serving_authorized"] is False
    assert forward["compression_claim"] is False


def test_factorial_selection_prefers_global_unweighted_on_equal_rank() -> None:
    selection = runner._select_tail_informed_factorial(
        _factorial_comparisons(
            global_passes=(False, True, True, True),
            treatment_passes=(False, True, True, True),
        ),
        tail_rank=_TAIL_RANK,
    )

    assert selection["overall_stable_passing_rank"] == 224
    assert selection["selected_arm"] == "unweighted.rank224"
    assert selection["selection_rule"] == (
        "smallest_total_rank_with_all_larger_same_branch_ranks_passing_"
        "then_global_unweighted_then_lexical_arm_id"
    )


def test_factorial_failure_opens_no_transfer_or_learned_lofo_authority() -> None:
    selection = runner._select_tail_informed_factorial(
        _factorial_comparisons(
            global_passes=(False, False, False, False),
            treatment_passes=(False, False, False, False),
        ),
        tail_rank=_TAIL_RANK,
    )

    assert selection["selected_arm"] is None
    assert selection["factorial_capacity_gate_passed"] is False
    assert (
        selection[
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized"
        ]
        is False
    )
    assert selection["later_lofo_fitting_authorized"] is False


def test_factorial_rejects_any_tail_rank_other_than_locked_a16_rT17() -> None:
    comparisons = _factorial_comparisons(
        global_passes=(False, False, False, False),
        treatment_passes=(False, False, False, False),
    )
    with pytest.raises(ValueError, match="requires numerical tail rank rT=17"):
        runner._select_tail_informed_factorial(comparisons, tail_rank=16)

    invalid_fit = _FakeTailInformedFit()
    invalid_fit.tail_rank = 16
    with pytest.raises(ValueError, match="requires rT=17"):
        runner._build_tail_informed_projection_arm_specs(
            invalid_fit,
            ranks=(208, 224, 256, 320),
        )


def test_factorial_cli_has_locked_local_defaults_and_parent_lineage_flag() -> None:
    parser = factorial.build_parser()
    arguments = parser.parse_args([])

    assert arguments.output == factorial.DEFAULT_OUTPUT
    assert runner._validate_output(arguments.output) == factorial.DEFAULT_OUTPUT
    assert arguments.parent_basis_rank_ladder == factorial.DEFAULT_PARENT_LADDER
    help_text = parser.format_help()
    assert "tail-informed single-projector factorial" in help_text
    assert "--output" in help_text
    assert "--parent-basis-rank-ladder" in help_text


def test_public_runner_locks_factorial_report_and_parent_contract() -> None:
    sentinel_report = {
        "schema": factorial._SCHEMA,
        "format_version": factorial._FORMAT_VERSION,
        "role": factorial._ROLE,
        "report_sha256": "ab" * 32,
    }
    parent = Path(".local-runs/synthetic/parent-ladder.json")
    output = Path(".local-runs/synthetic/tail-factorial.json")

    with patch.object(
        factorial.ladder,
        "run_gemma3_l3_l4_complete_h4_projection_basis_rank_ladder",
        return_value=sentinel_report,
    ) as inherited_runner:
        actual = (
            factorial.run_gemma3_l3_l4_complete_h4_tail_informed_factorial(
                parent_ladder_path=parent,
                output=output,
            )
        )

    assert actual is sentinel_report
    call = inherited_runner.call_args
    assert call is not None
    assert call.kwargs["output"] == output
    config = call.kwargs["_tail_informed_factorial"]
    assert config.schema == factorial._SCHEMA
    assert config.format_version == factorial._FORMAT_VERSION
    assert config.report_domain == factorial._REPORT_DOMAIN
    assert config.role == factorial._ROLE
    assert config.parent_ladder_path == parent
    assert config.parent_ladder_file_sha256 == factorial._PARENT_FILE_SHA256
    assert config.parent_ladder_report_sha256 == factorial._PARENT_REPORT_SHA256


def test_cli_emits_the_public_report_summary(capsys) -> None:
    report = {
        "report_sha256": "cd" * 32,
        "artifact": {
            "file": ".local-runs/synthetic/tail-factorial.json",
            "committable": False,
        },
        "selection": {
            "selected_arm": "tail_informed.rank224",
            "serving_authorized": False,
        },
        "scientific_status": {
            "tail_informed_factorial_complete": True,
            "compression_claim": False,
        },
    }
    with patch.object(
        factorial,
        "run_gemma3_l3_l4_complete_h4_tail_informed_factorial",
        return_value=report,
    ) as public_runner:
        exit_code = factorial.main(
            [
                "--output",
                ".local-runs/synthetic/tail-factorial.json",
                "--parent-basis-rank-ladder",
                ".local-runs/synthetic/parent-ladder.json",
            ]
        )

    assert exit_code == 0
    assert public_runner.call_args is not None
    assert public_runner.call_args.kwargs["parent_ladder_path"] == (
        ".local-runs/synthetic/parent-ladder.json"
    )
    assert json.loads(capsys.readouterr().out) == report
