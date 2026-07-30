"""Fit-only identification of incremental realized-H4 signal.

This diagnostic never accepts selection or guard inputs.  It authenticates an
already accepted X4-only executor, collects its private A-fit traces once, and
uses nested family cross-fitting to ask whether realized post-X4/pre-H4 state
adds out-of-family information beyond a matched causal L3 lag design.

No activation rows, gradients, prompt text, token IDs, model weights, or fitted
coefficient tensors are written to the report.  Only scalar summaries, hashes,
the frozen gate, and (when one exists) a tensor-hash-only winning recipe escape.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .conditional_quadratic_edge import (
    build_causal_lagged_modal_design,
)
from .gemma3_experiment import (
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
    load_gemma3_graph_organized_svd_candidate,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    _REPORT_DOMAIN as _CAMPAIGN_REPORT_DOMAIN,
    _canonical_json_bytes,
    _domain_sha256,
    _file_sha256,
    _mapping,
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpusArtifact,
    gemma3_l3_l4_progressive_a_fit_replacement_lineage,
    load_gemma3_l3_l4_progressive_a_fit_role,
)
from .gemma3_l3_l4_progressive_worker import (
    GemmaTwoHeadFitSequence,
    LegacyRank64GemmaProgressiveExecutable,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GemmaL3L4TwoHeadArtifact,
    GemmaL3L4TwoHeadExecutable,
    _canonical_sign_rows,
    _normalized_nll_directions,
    _overflow_safe_weighted_rms,
    _tensor_sha256,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)
from .radial_finite_displacement_correction import (
    family_balanced_row_weights,
)


__all__ = [
    "DEFAULT_H4_DAMPING_OUTPUT",
    "DEFAULT_H4_INCREMENTAL_SIGNAL_OUTPUT",
    "GemmaH4DampingRecipeTensors",
    "analyze_gemma_h4_damping",
    "analyze_gemma_h4_incremental_signal",
    "build_damping_parser",
    "build_parser",
    "damping_main",
    "derive_candidate_h4_output_decoder",
    "derive_gemma_h4_damping_recipe_tensors",
    "main",
    "run_gemma3_l3_l4_h4_damping_diagnostic",
    "run_gemma3_l3_l4_h4_incremental_signal_diagnostic",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_h4_incremental_signal"
_FORMAT_VERSION = 2
_ANALYSIS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-incremental-signal-analysis:v2\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-incremental-signal-report:v2\0"
)
_RECIPE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-incremental-signal-recipe:v2\0"
)
_DAMPING_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_h4_incremental_signal_damping"
)
_DAMPING_FORMAT_VERSION = 1
_DAMPING_ANALYSIS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-analysis:v1\0"
)
_DAMPING_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-report:v1\0"
)
_DAMPING_RECIPE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-recipe:v1\0"
)
_DAMPING_HYPOTHESIS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-h4-damping-hypothesis:v1\0"
)
_FACTORIZED_SCOPE = "factorized_refit"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_DEFAULT_CORPUS = _LOCAL_ROOT / "progressive-a-loss-v3.corpus.json"
_DEFAULT_ACCEPTED_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-projected-state-v6.campaign.json"
)
_DEFAULT_ACCEPTED_CANDIDATE = (
    _LOCAL_ROOT
    / "progressive-a-h4-projected-state-v6.campaign.candidate.pt"
)
DEFAULT_H4_INCREMENTAL_SIGNAL_OUTPUT = (
    _LOCAL_ROOT / "progressive-a-h4-incremental-signal-fit-v2.report.json"
)
DEFAULT_H4_DAMPING_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-h4-incremental-signal-damping-fit-v1.report.json"
)
_DEFAULT_DAMPING_HYPOTHESIS_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-h4-incremental-signal-fit-expanded-v2.report.json"
)
_DEFAULT_EXPANDED_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
_DEFAULT_EXPANDED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
_DEFAULT_DAMPING_ALPHAS = (0.25, 0.5, 0.75, 1.0)
_DAMPING_LAG_COUNT = 16
_DAMPING_INPUT_RANK = 32
_DAMPING_OUTPUT_RANK = 8
_DAMPING_FIT_EXAMPLE_COUNT = 16
_DAMPING_FIT_FAMILY_COUNT = 8
_DAMPING_FIT_ROW_COUNT = 1_008
_DEFAULT_LAG_COUNTS = (1, 2, 4, 8, 16, 32)
_DEFAULT_INPUT_RANKS = (8, 16, 32)
_LINEARIZED_IMPROVEMENT_MIN = 0.02
_WORST_FAMILY_REGRESSION_MAX = 0.02
_SECONDARY_REGRESSION_MAX = 0.02
_RESIDUAL_ENERGY_FRACTION_MIN = 1.0e-6
_MINIMUM_FAMILY_WIN_FRACTION = 0.75
_MAXIMUM_INNER_FAMILY_FOLDS = 3
_CG_RELATIVE_RESIDUAL_TOLERANCE = 1.0e-7
_CG_MAXIMUM_ITERATIONS = 1_024
_CG_RESTART_INTERVAL = 128


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _read_json_mapping(path: Path | str, *, label: str) -> Mapping[str, object]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return raw


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _canonical_positive_ints(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise ValueError(f"{label} must be a nonempty sequence")
    result = tuple(_positive_int(value, label=label) for value in values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _canonical_damping_alphas(
    values: Sequence[float],
) -> tuple[float, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise ValueError("damping_alphas must be a nonempty sequence")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError(
            "damping_alphas must be sorted unique values in (0, 1]"
        )
    result = tuple(float(value) for value in values)
    if (
        any(
            not math.isfinite(value)
            or value <= 0.0
            or value > 1.0
            for value in result
        )
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            "damping_alphas must be sorted unique values in (0, 1]"
        )
    return result


def _finite_float(value: Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("diagnostic metric became nonfinite")
    return result


def _source_code_sha256s() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "gemma3_l3_l4_h4_incremental_signal_diagnostic.py",
        "gemma3_l3_l4_progressive_a_corpus.py",
        "gemma3_l3_l4_progressive_worker.py",
        "gemma3_l3_l4_two_head_lowerer.py",
        "gemma3_l3_l4_graph_organized_svd_shadow_runtime.py",
    )
    return {
        name: _file_sha256(package / name)
        for name in names
    }


@dataclass(frozen=True, slots=True)
class _LagRows:
    design: Tensor
    realized_h4: Tensor
    target_modal: Tensor
    full_residual: Tensor
    loss_gradient: Tensor
    family_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    source_rank: int
    width: int
    output_rank: int

    def __post_init__(self) -> None:
        rows = int(self.design.shape[0])
        if (
            self.design.ndim != 2
            or rows <= 0
            or self.realized_h4.shape != (rows, self.width)
            or self.target_modal.shape != (rows, self.output_rank)
            or self.full_residual.shape != (rows, self.width)
            or self.loss_gradient.shape != (rows, self.width)
            or len(self.family_ids) != rows
            or len(self.example_ids) != rows
            or any(
                not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                for value in (
                    self.design,
                    self.realized_h4,
                    self.target_modal,
                    self.full_residual,
                    self.loss_gradient,
                )
            )
        ):
            raise ValueError("incremental-signal row geometry is invalid")


@dataclass(frozen=True, slots=True)
class GemmaH4DampingRecipeTensors:
    """Transient coefficient bundle reproduced from the frozen fit traces."""

    decoder: Tensor
    state_encoder: Tensor
    state_kernel: Tensor
    stored_lag_coefficients: Tensor
    baseline_lag_coefficients: Tensor
    source_rank: int
    lag_count: int
    ridge: float
    state_scale: float

    def __post_init__(self) -> None:
        output_rank, width = self.decoder.shape
        input_rank = int(self.state_encoder.shape[0])
        expected_lag_shape = (
            self.lag_count * self.source_rank,
            output_rank,
        )
        if (
            self.decoder.ndim != 2
            or output_rank <= 0
            or width <= 0
            or self.state_encoder.shape != (input_rank, width)
            or input_rank <= 0
            or self.state_kernel.shape != (input_rank, output_rank)
            or self.stored_lag_coefficients.shape != expected_lag_shape
            or self.baseline_lag_coefficients.shape != expected_lag_shape
            or any(
                not value.is_floating_point()
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or not bool(torch.isfinite(value).all())
                for value in (
                    self.decoder,
                    self.state_encoder,
                    self.state_kernel,
                    self.stored_lag_coefficients,
                    self.baseline_lag_coefficients,
                )
            )
            or self.source_rank <= 0
            or self.lag_count <= 0
            or not math.isfinite(self.ridge)
            or self.ridge <= 0.0
            or not math.isfinite(self.state_scale)
            or self.state_scale <= 0.0
            or self.state_scale > 1.0
        ):
            raise ValueError("damping recipe tensor geometry is invalid")
        for value in (self.decoder, self.state_encoder):
            identity = torch.eye(value.shape[0], dtype=torch.float64)
            if not torch.allclose(
                value @ value.T,
                identity,
                atol=1.0e-10,
                rtol=1.0e-10,
            ):
                raise ValueError("damping recipe encoders are not orthonormal")

    @property
    def input_rank(self) -> int:
        return int(self.state_encoder.shape[0])

    @property
    def output_rank(self) -> int:
        return int(self.decoder.shape[0])

    @property
    def width(self) -> int:
        return int(self.decoder.shape[1])

    @property
    def lag_kernel(self) -> Tensor:
        return self.stored_lag_coefficients.reshape(
            self.lag_count,
            self.source_rank,
            self.output_rank,
        ).contiguous()

    @property
    def baseline_lag_kernel(self) -> Tensor:
        return self.baseline_lag_coefficients.reshape(
            self.lag_count,
            self.source_rank,
            self.output_rank,
        ).contiguous()


def _prepare_lag_rows(
    sequences: Sequence[GemmaTwoHeadFitSequence],
    *,
    decoder: Tensor,
    lag_count: int,
) -> _LagRows:
    ordered = tuple(
        sorted(
            (sequence.detached_copy() for sequence in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    if not ordered:
        raise ValueError("incremental-signal analysis requires fit sequences")
    design_rows: list[Tensor] = []
    h4_rows: list[Tensor] = []
    modal_rows: list[Tensor] = []
    residual_rows: list[Tensor] = []
    gradient_rows: list[Tensor] = []
    families: list[str] = []
    examples: list[str] = []
    width = ordered[0].width
    source_rank = ordered[0].source_rank
    for sequence in ordered:
        if (
            sequence.width != width
            or sequence.source_rank != source_rank
            or sequence.candidate_h4_loss_gradient is None
        ):
            raise ValueError(
                "fit sequences lack common candidate-H4 geometry"
            )
        selected = sequence.target_affected_mask
        design = build_causal_lagged_modal_design(
            sequence.source_modes,
            logical_positions=sequence.logical_positions,
            valid_mask=sequence.valid_target_mask,
            lag_count=lag_count,
        )
        residual = sequence.h4_residual_rows.to(torch.float64)
        design_rows.append(design[selected].to(torch.float64))
        h4_rows.append(sequence.candidate_h4[selected].to(torch.float64))
        modal_rows.append(residual @ decoder.T)
        residual_rows.append(residual)
        gradient_rows.append(
            sequence.candidate_h4_loss_gradient[selected].to(torch.float64)
        )
        families.extend([sequence.family_id] * sequence.affected_rows)
        examples.extend([sequence.example_id] * sequence.affected_rows)
    return _LagRows(
        design=torch.cat(design_rows, dim=0).contiguous(),
        realized_h4=torch.cat(h4_rows, dim=0).contiguous(),
        target_modal=torch.cat(modal_rows, dim=0).contiguous(),
        full_residual=torch.cat(residual_rows, dim=0).contiguous(),
        loss_gradient=torch.cat(gradient_rows, dim=0).contiguous(),
        family_ids=tuple(families),
        example_ids=tuple(examples),
        source_rank=source_rank,
        width=width,
        output_rank=int(decoder.shape[0]),
    )


def _row_mask(
    family_ids: Sequence[str],
    included: set[str],
) -> Tensor:
    return torch.tensor(
        [family in included for family in family_ids],
        dtype=torch.bool,
    )


def _family_partitions(
    family_ids: set[str],
    *,
    maximum_fold_count: int,
) -> tuple[tuple[str, ...], ...]:
    ordered = tuple(sorted(family_ids))
    fold_count = min(maximum_fold_count, len(ordered))
    return tuple(
        tuple(ordered[index::fold_count])
        for index in range(fold_count)
    )


def _selected_weights(
    rows: _LagRows,
    selected: Tensor,
) -> Tensor:
    indices = selected.nonzero(as_tuple=False).flatten().tolist()
    return family_balanced_row_weights(
        tuple(rows.family_ids[index] for index in indices),
        tuple(rows.example_ids[index] for index in indices),
    ).to(torch.float64)


def _ridge_coefficients(
    design: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    ridge: float,
) -> Tensor:
    rms = _overflow_safe_weighted_rms(design, weights)
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    standardized = design / scales
    root = weights.sqrt().unsqueeze(1)
    weighted_design = standardized * root
    weighted_target = target * root
    rows, columns = weighted_design.shape
    if columns <= rows:
        coefficient = torch.linalg.solve(
            weighted_design.T @ weighted_design
            + ridge * torch.eye(columns, dtype=torch.float64),
            weighted_design.T @ weighted_target,
        )
    else:
        dual = torch.linalg.solve(
            weighted_design @ weighted_design.T
            + ridge * torch.eye(rows, dtype=torch.float64),
            weighted_target,
        )
        coefficient = weighted_design.T @ dual
    result = (coefficient / scales.unsqueeze(1)).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("weighted ridge nuisance fit became nonfinite")
    return result


def _ridge_predictions(
    train_design: Tensor,
    train_target: Tensor,
    train_weights: Tensor,
    evaluation_design: Tensor,
    *,
    ridge: float,
) -> Tensor:
    """Predict a wide nuisance target without materializing ``P x W``."""

    rms = _overflow_safe_weighted_rms(train_design, train_weights)
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    standardized_train = train_design / scales
    standardized_evaluation = evaluation_design / scales
    root = train_weights.sqrt().unsqueeze(1)
    weighted_design = standardized_train * root
    weighted_target = train_target * root
    rows, columns = weighted_design.shape
    if columns <= rows:
        coefficient = torch.linalg.solve(
            weighted_design.T @ weighted_design
            + ridge * torch.eye(columns, dtype=torch.float64),
            weighted_design.T @ weighted_target,
        )
        prediction = standardized_evaluation @ coefficient
    else:
        dual = torch.linalg.solve(
            weighted_design @ weighted_design.T
            + ridge * torch.eye(rows, dtype=torch.float64),
            weighted_target,
        )
        prediction = (
            standardized_evaluation @ weighted_design.T
        ) @ dual
    result = prediction.contiguous()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("weighted ridge nuisance prediction became nonfinite")
    return result


def _candidate_metric_coefficients(
    *,
    design: Tensor,
    target_modal: Tensor,
    full_residual: Tensor,
    loss_gradient: Tensor,
    decoder: Tensor,
    weights: Tensor,
    ridge: float,
) -> Tensor:
    rms = _overflow_safe_weighted_rms(design, weights)
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    standardized = design / scales
    root = weights.sqrt().unsqueeze(1)
    weighted_design = standardized * root
    weighted_target = target_modal * root
    rows, columns = weighted_design.shape
    if columns <= rows:
        initial = torch.linalg.solve(
            weighted_design.T @ weighted_design
            + ridge * torch.eye(columns, dtype=torch.float64),
            weighted_design.T @ weighted_target,
        )
    else:
        dual = torch.linalg.solve(
            weighted_design @ weighted_design.T
            + ridge * torch.eye(rows, dtype=torch.float64),
            weighted_target,
        )
        initial = weighted_design.T @ dual
    fitted = _solve_diagnostic_candidate_metric_ridge(
        standardized_design=standardized,
        weighted_design=weighted_design,
        weights=weights,
        target=target_modal,
        residual=full_residual,
        normalized_gradient=_normalized_nll_directions(loss_gradient),
        decoder=decoder,
        ridge=ridge,
        initial=initial,
    )
    result = (fitted / scales.unsqueeze(1)).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("candidate metric fit became nonfinite")
    return result


def _solve_diagnostic_candidate_metric_ridge(
    *,
    standardized_design: Tensor,
    weighted_design: Tensor,
    weights: Tensor,
    target: Tensor,
    residual: Tensor,
    normalized_gradient: Tensor,
    decoder: Tensor,
    ridge: float,
    initial: Tensor,
) -> Tensor:
    """Solve the bounded VJP metric to an audited diagnostic tolerance."""

    projected_gradient = normalized_gradient @ decoder.T
    directional_target = (
        normalized_gradient * residual
    ).sum(dim=1)
    if not bool(projected_gradient.abs().any()):
        return initial.contiguous()

    weighted_target = target * weights.sqrt().unsqueeze(1)
    right_hand_side = weighted_design.T @ weighted_target
    right_hand_side = right_hand_side + standardized_design.T @ (
        (weights * directional_target).unsqueeze(1)
        * projected_gradient
    )

    def operator(value: Tensor) -> Tensor:
        prediction = standardized_design @ value
        directional_prediction = (
            prediction * projected_gradient
        ).sum(dim=1)
        return (
            weighted_design.T @ (weighted_design @ value)
            + ridge * value
            + standardized_design.T
            @ (
                (weights * directional_prediction).unsqueeze(1)
                * projected_gradient
            )
        )

    diagonal = (
        weighted_design.square().sum(dim=0).unsqueeze(1)
        + ridge
        + standardized_design.square().T
        @ (weights.unsqueeze(1) * projected_gradient.square())
    )
    if (
        not bool(torch.isfinite(right_hand_side).all())
        or not bool(torch.isfinite(diagonal).all())
        or bool((diagonal <= 0.0).any())
    ):
        raise RuntimeError("diagnostic loss-metric ridge system is invalid")

    solution = initial.detach().clone()
    residual_vector = right_hand_side - operator(solution)
    right_norm = max(
        float(torch.linalg.vector_norm(right_hand_side)),
        1.0,
    )
    tolerance = _CG_RELATIVE_RESIDUAL_TOLERANCE * right_norm
    if float(torch.linalg.vector_norm(residual_vector)) <= tolerance:
        return solution.contiguous()
    preconditioned = residual_vector / diagonal
    search = preconditioned.clone()
    residual_dot = (residual_vector * preconditioned).sum()
    for iteration in range(_CG_MAXIMUM_ITERATIONS):
        operated = operator(search)
        denominator = (search * operated).sum()
        if (
            not bool(torch.isfinite(denominator))
            or float(denominator) <= 0.0
        ):
            raise RuntimeError(
                "diagnostic loss-metric CG curvature is invalid"
            )
        step = residual_dot / denominator
        solution = solution + step * search
        residual_vector = residual_vector - step * operated
        if (iteration + 1) % _CG_RESTART_INTERVAL == 0:
            residual_vector = right_hand_side - operator(solution)
        if float(torch.linalg.vector_norm(residual_vector)) <= tolerance:
            return solution.contiguous()
        preconditioned = residual_vector / diagonal
        next_residual_dot = (
            residual_vector * preconditioned
        ).sum()
        if (
            not bool(torch.isfinite(next_residual_dot))
            or float(next_residual_dot) < 0.0
        ):
            raise RuntimeError(
                "diagnostic loss-metric CG residual is invalid"
            )
        if (iteration + 1) % _CG_RESTART_INTERVAL == 0:
            search = preconditioned.clone()
        else:
            search = (
                preconditioned
                + (next_residual_dot / residual_dot) * search
            )
        residual_dot = next_residual_dot
    relative_residual = float(
        torch.linalg.vector_norm(
            right_hand_side - operator(solution)
        )
    ) / right_norm
    raise RuntimeError(
        "diagnostic loss-metric CG did not converge: "
        f"relative residual {relative_residual:.3e}"
    )


def _energy_rank(singular_values: Tensor, fraction: float) -> int:
    energy = singular_values.square()
    total = float(energy.sum())
    if total == 0.0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(
        torch.searchsorted(
            cumulative,
            torch.tensor(fraction, dtype=torch.float64),
        )
    ) + 1


def _spectrum(
    design: Tensor,
    weights: Tensor,
    *,
    ridge: float,
) -> tuple[dict[str, object], Tensor]:
    rms = _overflow_safe_weighted_rms(design, weights)
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    weighted = design / scales * weights.sqrt().unsqueeze(1)
    left, singular_values, _right = torch.linalg.svd(
        weighted,
        full_matrices=False,
    )
    maximum = float(singular_values.max()) if singular_values.numel() else 0.0
    tolerance = (
        max(weighted.shape)
        * torch.finfo(torch.float64).eps
        * maximum
    )
    support = singular_values > tolerance
    rank = int(support.sum())
    supported = singular_values[support]
    minimum = float(supported.min()) if supported.numel() else 0.0
    squares = singular_values.square()
    total = float(squares.sum())
    fourth = float(squares.square().sum())
    summary = {
        "row_count": int(design.shape[0]),
        "column_count": int(design.shape[1]),
        "numerical_rank": rank,
        "available_row_dimension": int(design.shape[0]) - rank,
        "rank_tolerance": _finite_float(tolerance),
        "sigma_max": _finite_float(maximum),
        "sigma_min_supported": _finite_float(minimum),
        "condition_number_supported": (
            0.0 if minimum == 0.0 else _finite_float(maximum / minimum)
        ),
        "stable_rank": (
            0.0 if maximum == 0.0 else _finite_float(total / maximum**2)
        ),
        "participation_ratio": (
            0.0 if fourth == 0.0 else _finite_float(total**2 / fourth)
        ),
        "ridge_effective_degrees_of_freedom": _finite_float(
            (squares / (squares + ridge)).sum()
        ),
        "energy_rank_90": _energy_rank(singular_values, 0.90),
        "energy_rank_95": _energy_rank(singular_values, 0.95),
        "energy_rank_99": _energy_rank(singular_values, 0.99),
        "singular_values_sha256": _tensor_sha256(singular_values),
    }
    return summary, left[:, support].contiguous()


def _incremental_rank(
    design: Tensor,
    state_features: Tensor,
    weights: Tensor,
    *,
    design_basis: Tensor,
    design_sigma_max: float,
) -> int:
    rms = _overflow_safe_weighted_rms(state_features, weights)
    floor = math.sqrt(torch.finfo(torch.float64).eps)
    scales = torch.where(rms > floor, rms, torch.ones_like(rms))
    weighted = state_features / scales * weights.sqrt().unsqueeze(1)
    if design_basis.numel():
        weighted = weighted - design_basis @ (
            design_basis.T @ weighted
        )
    singular_values = torch.linalg.svdvals(weighted)
    residual_maximum = (
        float(singular_values.max()) if singular_values.numel() else 0.0
    )
    maximum = max(design_sigma_max, residual_maximum)
    tolerance = (
        max(design.shape[0], design.shape[1] + state_features.shape[1])
        * torch.finfo(torch.float64).eps
        * maximum
    )
    return int((singular_values > tolerance).sum())


def _metrics(
    *,
    prediction_modal: Tensor,
    target_modal: Tensor,
    full_residual: Tensor,
    loss_gradient: Tensor,
    decoder: Tensor,
    weights: Tensor,
) -> dict[str, float]:
    modal_error = target_modal - prediction_modal
    full_error = full_residual - prediction_modal @ decoder
    normalized = _normalized_nll_directions(loss_gradient)
    projected = torch.sqrt(
        (
            modal_error.square().sum(dim=1) * weights
        ).sum()
        / target_modal.shape[1]
    )
    normalized_rmse = torch.sqrt(
        (
            (normalized * full_error).sum(dim=1).square()
            * weights
        ).sum()
    )
    linearized = torch.sqrt(
        (
            (loss_gradient * full_error).sum(dim=1).square()
            * weights
        ).sum()
    )
    return {
        "projected_residual_rmse": _finite_float(projected),
        "normalized_nll_direction_rmse": _finite_float(normalized_rmse),
        "linearized_nll_residual_rmse": _finite_float(linearized),
        "bounded_candidate_metric_rmse": _finite_float(
            torch.sqrt(projected.square() + normalized_rmse.square())
        ),
    }


def _mean_metrics(
    values: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    names = tuple(values[0])
    return {
        name: _finite_float(
            sum(float(value[name]) for value in values) / len(values)
        )
        for name in names
    }


def _relative_improvement(new: float, baseline: float) -> float:
    if baseline == 0.0:
        # Keep the scalar-only JSON artifact finite while still making a
        # newly introduced error fail every preregistered improvement gate.
        return 0.0 if new == 0.0 else -1.0
    return _finite_float(1.0 - new / baseline)


def _partial_r2(new: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if new == 0.0 else -1.0
    return _finite_float(1.0 - (new / baseline) ** 2)


def _canonical_encoder(value: Tensor) -> Tensor:
    return _canonical_sign_rows(value.to(torch.float64))


def _encoder_stability(encoders: Sequence[Tensor]) -> dict[str, float]:
    minimum_cosines: list[float] = []
    overlaps: list[float] = []
    for left_index, left in enumerate(encoders):
        for right in encoders[left_index + 1 :]:
            singular_values = torch.linalg.svdvals(left @ right.T).clamp(
                min=0.0,
                max=1.0,
            )
            minimum_cosines.append(float(singular_values.min()))
            overlaps.append(
                float(singular_values.square().mean())
            )
    return {
        "minimum_principal_cosine": _finite_float(min(minimum_cosines)),
        "mean_squared_subspace_overlap": _finite_float(
            sum(overlaps) / len(overlaps)
        ),
    }


@dataclass(slots=True)
class _OuterFold:
    held_family: str
    train_mask: Tensor
    test_mask: Tensor
    baseline_coefficients: Tensor
    test_h4_residual: Tensor
    cross_design: Tensor
    cross_h4_residual: Tensor
    cross_modal_residual: Tensor
    cross_full_residual: Tensor
    cross_gradient: Tensor
    cross_family_ids: tuple[str, ...]
    cross_example_ids: tuple[str, ...]
    cross_weights: Tensor
    cross_design_basis: Tensor
    cross_design_spectrum: dict[str, object]
    inner_fold_audits: tuple[dict[str, object], ...]
    independent_encoder_32: Tensor
    baseline_metrics: dict[str, float]


def _outer_fold(
    rows: _LagRows,
    *,
    decoder: Tensor,
    held_family: str,
    maximum_input_rank: int,
    ridge: float,
) -> _OuterFold:
    all_families = set(rows.family_ids)
    train_families = all_families - {held_family}
    train_mask = _row_mask(rows.family_ids, train_families)
    test_mask = ~train_mask
    train_weights = _selected_weights(rows, train_mask)
    baseline = _candidate_metric_coefficients(
        design=rows.design[train_mask],
        target_modal=rows.target_modal[train_mask],
        full_residual=rows.full_residual[train_mask],
        loss_gradient=rows.loss_gradient[train_mask],
        decoder=decoder,
        weights=train_weights,
        ridge=ridge,
    )
    test_h4_residual = (
        rows.realized_h4[test_mask]
        - _ridge_predictions(
            rows.design[train_mask],
            rows.realized_h4[train_mask],
            train_weights,
            rows.design[test_mask],
            ridge=ridge,
        )
    )
    test_weights = _selected_weights(rows, test_mask)
    baseline_metrics = _metrics(
        prediction_modal=rows.design[test_mask] @ baseline,
        target_modal=rows.target_modal[test_mask],
        full_residual=rows.full_residual[test_mask],
        loss_gradient=rows.loss_gradient[test_mask],
        decoder=decoder,
        weights=test_weights,
    )

    cross_design: list[Tensor] = []
    cross_h4: list[Tensor] = []
    cross_modal: list[Tensor] = []
    cross_full: list[Tensor] = []
    cross_gradient: list[Tensor] = []
    cross_families: list[str] = []
    cross_examples: list[str] = []
    inner_fold_audits: list[dict[str, object]] = []
    for inner_held_families in _family_partitions(
        train_families,
        maximum_fold_count=_MAXIMUM_INNER_FAMILY_FOLDS,
    ):
        inner_held = set(inner_held_families)
        inner_train_families = train_families - inner_held
        inner_train = _row_mask(rows.family_ids, inner_train_families)
        inner_test = _row_mask(rows.family_ids, inner_held)
        inner_weights = _selected_weights(rows, inner_train)
        inner_fold_audits.append(
            {
                "held_families": inner_held_families,
                "train_family_count": len(inner_train_families),
                "train_row_count": int(inner_train.sum()),
                "held_row_count": int(inner_test.sum()),
                "column_count": int(rows.design.shape[1]),
            }
        )
        inner_baseline = _candidate_metric_coefficients(
            design=rows.design[inner_train],
            target_modal=rows.target_modal[inner_train],
            full_residual=rows.full_residual[inner_train],
            loss_gradient=rows.loss_gradient[inner_train],
            decoder=decoder,
            weights=inner_weights,
            ridge=ridge,
        )
        indices = inner_test.nonzero(as_tuple=False).flatten().tolist()
        predicted_modal = rows.design[inner_test] @ inner_baseline
        cross_design.append(rows.design[inner_test])
        cross_h4.append(
            rows.realized_h4[inner_test]
            - _ridge_predictions(
                rows.design[inner_train],
                rows.realized_h4[inner_train],
                inner_weights,
                rows.design[inner_test],
                ridge=ridge,
            )
        )
        cross_modal.append(
            rows.target_modal[inner_test] - predicted_modal
        )
        cross_full.append(
            rows.full_residual[inner_test] - predicted_modal @ decoder
        )
        cross_gradient.append(rows.loss_gradient[inner_test])
        cross_families.extend(rows.family_ids[index] for index in indices)
        cross_examples.extend(rows.example_ids[index] for index in indices)
    design = torch.cat(cross_design, dim=0).contiguous()
    h4_residual = torch.cat(cross_h4, dim=0).contiguous()
    modal_residual = torch.cat(cross_modal, dim=0).contiguous()
    full_residual = torch.cat(cross_full, dim=0).contiguous()
    gradient = torch.cat(cross_gradient, dim=0).contiguous()
    weights = family_balanced_row_weights(
        tuple(cross_families),
        tuple(cross_examples),
    ).to(torch.float64)
    spectrum, basis = _spectrum(design, weights, ridge=ridge)
    weighted_h4 = h4_residual * weights.sqrt().unsqueeze(1)
    _left, _singular, right = torch.linalg.svd(
        weighted_h4,
        full_matrices=False,
    )
    count = min(maximum_input_rank, int(right.shape[0]))
    if count < maximum_input_rank:
        raise ValueError("outer fold has too few rows for the input rank grid")
    encoder = _canonical_encoder(right[:maximum_input_rank])
    return _OuterFold(
        held_family=held_family,
        train_mask=train_mask,
        test_mask=test_mask,
        baseline_coefficients=baseline,
        test_h4_residual=test_h4_residual,
        cross_design=design,
        cross_h4_residual=h4_residual,
        cross_modal_residual=modal_residual,
        cross_full_residual=full_residual,
        cross_gradient=gradient,
        cross_family_ids=tuple(cross_families),
        cross_example_ids=tuple(cross_examples),
        cross_weights=weights,
        cross_design_basis=basis,
        cross_design_spectrum=spectrum,
        inner_fold_audits=tuple(inner_fold_audits),
        independent_encoder_32=encoder,
        baseline_metrics=baseline_metrics,
    )


def _resource_cost(
    *,
    lag_count: int,
    source_rank: int,
    width: int,
    output_rank: int,
    encoder_kind: str,
    input_rank: int,
) -> dict[str, int]:
    baseline = output_rank * (
        width + lag_count * source_rank
    )
    if encoder_kind == "reused_output_decoder":
        extra_parameters = input_rank * output_rank
        extra_macs = width * input_rank + input_rank * output_rank
    else:
        extra_parameters = (
            width * input_rank + input_rank * output_rank
        )
        extra_macs = extra_parameters
    return {
        "head_parameters": baseline + extra_parameters,
        "head_runtime_parameter_bytes": 8 * (
            baseline + extra_parameters
        ),
        "head_logical_macs_per_token": baseline + extra_macs,
        "conditioning_parameters": extra_parameters,
        "conditioning_runtime_parameter_bytes": 8 * extra_parameters,
        "conditioning_logical_macs_per_token": extra_macs,
    }


def _cell(
    rows: _LagRows,
    folds: Sequence[_OuterFold],
    *,
    decoder: Tensor,
    lag_count: int,
    encoder_kind: str,
    input_rank: int,
    minimum_family_win_count: int,
    ridge: float,
) -> tuple[dict[str, object], tuple[Tensor, ...]]:
    per_family: list[dict[str, object]] = []
    encoders: list[Tensor] = []
    residual_energy_fractions: list[float] = []
    encoder_capture_fractions: list[float] = []
    for fold in folds:
        encoder = (
            decoder
            if encoder_kind == "reused_output_decoder"
            else fold.independent_encoder_32[:input_rank]
        ).contiguous()
        encoders.append(encoder)
        cross_state = fold.cross_h4_residual @ encoder.T
        state_kernel = _candidate_metric_coefficients(
            design=cross_state,
            target_modal=fold.cross_modal_residual,
            full_residual=fold.cross_full_residual,
            loss_gradient=fold.cross_gradient,
            decoder=decoder,
            weights=fold.cross_weights,
            ridge=ridge,
        )
        test_state = fold.test_h4_residual @ encoder.T
        prediction = (
            rows.design[fold.test_mask] @ fold.baseline_coefficients
            + test_state @ state_kernel
        )
        test_weights = _selected_weights(rows, fold.test_mask)
        metrics = _metrics(
            prediction_modal=prediction,
            target_modal=rows.target_modal[fold.test_mask],
            full_residual=rows.full_residual[fold.test_mask],
            loss_gradient=rows.loss_gradient[fold.test_mask],
            decoder=decoder,
            weights=test_weights,
        )
        baseline = fold.baseline_metrics
        improvements = {
            name: _relative_improvement(metrics[name], baseline[name])
            for name in metrics
        }
        h4_energy = float(
            (
                rows.realized_h4[fold.train_mask].square().sum(dim=1)
                * _selected_weights(rows, fold.train_mask)
            ).sum()
        )
        residual_energy = float(
            (
                fold.cross_h4_residual.square().sum(dim=1)
                * fold.cross_weights
            ).sum()
        )
        state_energy = float(
            (
                cross_state.square().sum(dim=1)
                * fold.cross_weights
            ).sum()
        )
        residual_fraction = (
            0.0 if h4_energy == 0.0 else residual_energy / h4_energy
        )
        capture_fraction = (
            0.0 if residual_energy == 0.0 else state_energy / residual_energy
        )
        residual_energy_fractions.append(_finite_float(residual_fraction))
        encoder_capture_fractions.append(_finite_float(capture_fraction))
        added_rank = _incremental_rank(
            fold.cross_design,
            cross_state,
            fold.cross_weights,
            design_basis=fold.cross_design_basis,
            design_sigma_max=float(
                fold.cross_design_spectrum["sigma_max"]
            ),
        )
        overlap = float(
            (encoder @ decoder.T).square().sum() / input_rank
        )
        per_family.append(
            {
                "held_family": fold.held_family,
                "baseline": baseline,
                "conditioned": metrics,
                "relative_improvement": improvements,
                "incremental_numerical_rank": added_rank,
                "residualized_h4_energy_fraction": _finite_float(
                    residual_fraction
                ),
                "encoder_capture_fraction": _finite_float(capture_fraction),
                "encoder_output_decoder_overlap": _finite_float(overlap),
                "encoder_sha256": _tensor_sha256(encoder),
                "state_kernel_sha256": _tensor_sha256(state_kernel),
            }
        )
    macro_baseline = _mean_metrics(
        tuple(fold.baseline_metrics for fold in folds)
    )
    macro_conditioned = _mean_metrics(
        tuple(
            value["conditioned"]  # type: ignore[arg-type]
            for value in per_family
        )
    )
    macro_improvement = {
        name: _relative_improvement(
            macro_conditioned[name],
            macro_baseline[name],
        )
        for name in macro_baseline
    }
    linearized_changes = tuple(
        float(
            _mapping(
                value["relative_improvement"],
                label="family improvement",
            )["linearized_nll_residual_rmse"]
        )
        for value in per_family
    )
    partial_r2 = tuple(
        _partial_r2(
            float(
                _mapping(
                    value["conditioned"],
                    label="conditioned metrics",
                )["projected_residual_rmse"]
            ),
            float(
                _mapping(
                    value["baseline"],
                    label="baseline metrics",
                )["projected_residual_rmse"]
            ),
        )
        for value in per_family
    )
    minimum_incremental_rank = min(
        int(value["incremental_numerical_rank"])
        for value in per_family
    )
    gate = {
        "all_input_ranks_identifiable": (
            minimum_incremental_rank == input_rank
        ),
        "macro_linearized_improvement_at_least_2pct": (
            macro_improvement["linearized_nll_residual_rmse"]
            >= _LINEARIZED_IMPROVEMENT_MIN
        ),
        "minimum_family_win_count_met": (
            sum(change > 0.0 for change in linearized_changes)
            >= minimum_family_win_count
        ),
        "worst_family_regression_at_most_2pct": (
            min(linearized_changes) >= -_WORST_FAMILY_REGRESSION_MAX
        ),
        "secondary_metrics_regress_at_most_2pct": (
            macro_improvement["normalized_nll_direction_rmse"]
            >= -_SECONDARY_REGRESSION_MAX
            and macro_improvement["projected_residual_rmse"]
            >= -_SECONDARY_REGRESSION_MAX
        ),
        "residualized_h4_energy_is_nontrivial": (
            min(residual_energy_fractions)
            >= _RESIDUAL_ENERGY_FRACTION_MIN
        ),
    }
    passed = all(gate.values())
    result = {
        "lag_count": lag_count,
        "l3_column_count": lag_count * rows.source_rank,
        "encoder_kind": encoder_kind,
        "input_rank": input_rank,
        "output_rank": rows.output_rank,
        "per_family": tuple(per_family),
        "macro_baseline": macro_baseline,
        "macro_conditioned": macro_conditioned,
        "macro_relative_improvement": macro_improvement,
        "family_linearized_win_count": sum(
            change > 0.0 for change in linearized_changes
        ),
        "minimum_family_win_count": minimum_family_win_count,
        "worst_family_linearized_improvement": _finite_float(
            min(linearized_changes)
        ),
        "macro_partial_r2": _finite_float(
            sum(partial_r2) / len(partial_r2)
        ),
        "minimum_incremental_numerical_rank": (
            minimum_incremental_rank
        ),
        "mean_residualized_h4_energy_fraction": _finite_float(
            sum(residual_energy_fractions)
            / len(residual_energy_fractions)
        ),
        "mean_encoder_capture_fraction": _finite_float(
            sum(encoder_capture_fractions)
            / len(encoder_capture_fractions)
        ),
        "encoder_stability": _encoder_stability(encoders),
        "resources": _resource_cost(
            lag_count=lag_count,
            source_rank=rows.source_rank,
            width=rows.width,
            output_rank=rows.output_rank,
            encoder_kind=encoder_kind,
            input_rank=input_rank,
        ),
        "gate": gate,
        "eligible": passed,
    }
    return result, tuple(encoders)


def _derive_final_recipe_tensors(
    rows: _LagRows,
    folds: Sequence[_OuterFold],
    *,
    decoder: Tensor,
    cell: Mapping[str, object],
    ridge: float,
    state_scale: float = 1.0,
) -> GemmaH4DampingRecipeTensors:
    if (
        not math.isfinite(state_scale)
        or state_scale <= 0.0
        or state_scale > 1.0
    ):
        raise ValueError("state_scale must be finite and in (0, 1]")
    encoder_kind = str(cell["encoder_kind"])
    input_rank = int(cell["input_rank"])
    cross_h4: list[Tensor] = []
    cross_modal: list[Tensor] = []
    cross_full: list[Tensor] = []
    cross_gradient: list[Tensor] = []
    cross_families: list[str] = []
    cross_examples: list[str] = []
    for fold in folds:
        indices = fold.test_mask.nonzero(as_tuple=False).flatten().tolist()
        predicted = (
            rows.design[fold.test_mask] @ fold.baseline_coefficients
        )
        cross_h4.append(
            fold.test_h4_residual
        )
        cross_modal.append(
            rows.target_modal[fold.test_mask] - predicted
        )
        cross_full.append(
            rows.full_residual[fold.test_mask] - predicted @ decoder
        )
        cross_gradient.append(rows.loss_gradient[fold.test_mask])
        cross_families.extend(rows.family_ids[index] for index in indices)
        cross_examples.extend(rows.example_ids[index] for index in indices)
    h4_residual = torch.cat(cross_h4, dim=0).contiguous()
    modal_residual = torch.cat(cross_modal, dim=0).contiguous()
    full_residual = torch.cat(cross_full, dim=0).contiguous()
    gradient = torch.cat(cross_gradient, dim=0).contiguous()
    weights = family_balanced_row_weights(
        tuple(cross_families),
        tuple(cross_examples),
    ).to(torch.float64)
    if encoder_kind == "reused_output_decoder":
        encoder = decoder
    else:
        weighted = h4_residual * weights.sqrt().unsqueeze(1)
        _left, _singular, right = torch.linalg.svd(
            weighted,
            full_matrices=False,
        )
        encoder = _canonical_encoder(right[:input_rank])
    state = h4_residual @ encoder.T
    state_kernel = _candidate_metric_coefficients(
        design=state,
        target_modal=modal_residual,
        full_residual=full_residual,
        loss_gradient=gradient,
        decoder=decoder,
        weights=weights,
        ridge=ridge,
    ) * state_scale
    all_rows = torch.ones(len(rows.family_ids), dtype=torch.bool)
    all_weights = _selected_weights(rows, all_rows)
    baseline = _candidate_metric_coefficients(
        design=rows.design,
        target_modal=rows.target_modal,
        full_residual=rows.full_residual,
        loss_gradient=rows.loss_gradient,
        decoder=decoder,
        weights=all_weights,
        ridge=ridge,
    )
    nuisance_state = _ridge_coefficients(
        rows.design,
        rows.realized_h4 @ encoder.T,
        all_weights,
        ridge=ridge,
    )
    stored_lag = (
        baseline
        - nuisance_state @ state_kernel
    ).contiguous()
    return GemmaH4DampingRecipeTensors(
        decoder=decoder.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous(),
        state_encoder=encoder.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous(),
        state_kernel=state_kernel.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous(),
        stored_lag_coefficients=stored_lag.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous(),
        baseline_lag_coefficients=baseline.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous(),
        source_rank=rows.source_rank,
        lag_count=int(cell["lag_count"]),
        ridge=ridge,
        state_scale=state_scale,
    )


def _recipe_from_tensors(
    tensors: GemmaH4DampingRecipeTensors,
    *,
    encoder_kind: str,
    resources: object,
    recipe_domain: bytes,
) -> dict[str, object]:
    recipe = {
        "lag_count": tensors.lag_count,
        "encoder_kind": encoder_kind,
        "input_rank": tensors.input_rank,
        "output_rank": tensors.output_rank,
        "state_scale": tensors.state_scale,
        "residualizer_folded_into_lag_kernel": True,
        "decoder_sha256": _tensor_sha256(tensors.decoder),
        "state_encoder_sha256": _tensor_sha256(
            tensors.state_encoder
        ),
        "state_kernel_sha256": _tensor_sha256(tensors.state_kernel),
        "stored_lag_coefficients_sha256": _tensor_sha256(
            tensors.stored_lag_coefficients
        ),
        "runtime_formula": (
            "(lagged_l3 @ stored_lag + realized_h4 @ "
            "state_encoder.T @ state_kernel) @ decoder"
        ),
        "resources": resources,
    }
    recipe["recipe_sha256"] = _sha256(recipe_domain, recipe)
    return recipe


def _final_recipe(
    rows: _LagRows,
    folds: Sequence[_OuterFold],
    *,
    decoder: Tensor,
    cell: Mapping[str, object],
    ridge: float,
    state_scale: float = 1.0,
    recipe_domain: bytes = _RECIPE_DOMAIN,
) -> dict[str, object]:
    tensors = _derive_final_recipe_tensors(
        rows,
        folds,
        decoder=decoder,
        cell=cell,
        ridge=ridge,
        state_scale=state_scale,
    )
    return _recipe_from_tensors(
        tensors,
        encoder_kind=str(cell["encoder_kind"]),
        resources=cell["resources"],
        recipe_domain=recipe_domain,
    )


def _refit_final_recipe(
    sequences: Sequence[GemmaTwoHeadFitSequence],
    *,
    decoder: Tensor,
    cell: Mapping[str, object],
    maximum_input_rank: int,
    ridge: float,
    state_scale: float = 1.0,
    recipe_domain: bytes = _RECIPE_DOMAIN,
) -> dict[str, object]:
    """Recompute only the selected lag instead of retaining every fit tensor."""

    lag_count = int(cell["lag_count"])
    rows = _prepare_lag_rows(
        sequences,
        decoder=decoder,
        lag_count=lag_count,
    )
    families = tuple(sorted(set(rows.family_ids)))
    folds = tuple(
        _outer_fold(
            rows,
            decoder=decoder,
            held_family=family,
            maximum_input_rank=maximum_input_rank,
            ridge=ridge,
        )
        for family in families
    )
    return _final_recipe(
        rows,
        folds,
        decoder=decoder,
        cell=cell,
        ridge=ridge,
        state_scale=state_scale,
        recipe_domain=recipe_domain,
    )


def derive_gemma_h4_damping_recipe_tensors(
    *,
    sequences: Sequence[GemmaTwoHeadFitSequence],
    output_decoder: Tensor,
    lag_count: int = _DAMPING_LAG_COUNT,
    input_rank: int = _DAMPING_INPUT_RANK,
    state_scale: float = 0.5,
    ridge: float = 1.0e-6,
) -> tuple[GemmaH4DampingRecipeTensors, dict[str, object]]:
    """Reproduce one fixed damping recipe and retain its tensors transiently."""

    lag_count = _positive_int(lag_count, label="lag_count")
    input_rank = _positive_int(input_rank, label="input_rank")
    ridge = _positive_finite(ridge, label="ridge")
    decoder = _canonical_encoder(output_decoder)
    if (
        decoder.ndim != 2
        or decoder.shape[0] <= 0
        or input_rank < decoder.shape[0]
        or input_rank > decoder.shape[1]
        or not torch.allclose(
            decoder @ decoder.T,
            torch.eye(decoder.shape[0], dtype=torch.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    ):
        raise ValueError("fixed damping materialization geometry is invalid")
    ordered = tuple(
        sorted(
            (sequence.detached_copy() for sequence in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    if not ordered:
        raise ValueError("fixed damping materialization requires fit sequences")
    rows = _prepare_lag_rows(
        ordered,
        decoder=decoder,
        lag_count=lag_count,
    )
    families = tuple(sorted(set(rows.family_ids)))
    if len(families) < 4:
        raise ValueError("fixed damping materialization requires four families")
    folds = tuple(
        _outer_fold(
            rows,
            decoder=decoder,
            held_family=family,
            maximum_input_rank=input_rank,
            ridge=ridge,
        )
        for family in families
    )
    resources = _resource_cost(
        lag_count=lag_count,
        source_rank=rows.source_rank,
        width=rows.width,
        output_rank=rows.output_rank,
        encoder_kind="independent_crossfit_h4_svd",
        input_rank=input_rank,
    )
    cell = {
        "lag_count": lag_count,
        "encoder_kind": "independent_crossfit_h4_svd",
        "input_rank": input_rank,
        "resources": resources,
    }
    tensors = _derive_final_recipe_tensors(
        rows,
        folds,
        decoder=decoder,
        cell=cell,
        ridge=ridge,
        state_scale=state_scale,
    )
    recipe = _recipe_from_tensors(
        tensors,
        encoder_kind="independent_crossfit_h4_svd",
        resources=resources,
        recipe_domain=_DAMPING_RECIPE_DOMAIN,
    )
    return tensors, recipe


def analyze_gemma_h4_incremental_signal(
    *,
    sequences: Sequence[GemmaTwoHeadFitSequence],
    output_decoder: Tensor,
    lag_counts: Sequence[int] = _DEFAULT_LAG_COUNTS,
    input_ranks: Sequence[int] = _DEFAULT_INPUT_RANKS,
    ridge: float = 1.0e-6,
) -> dict[str, object]:
    """Run the tensor-only nested family-cross-fitted identification grid."""

    lags = _canonical_positive_ints(lag_counts, label="lag_counts")
    ranks = _canonical_positive_ints(input_ranks, label="input_ranks")
    ridge = _positive_finite(ridge, label="ridge")
    decoder = _canonical_encoder(output_decoder)
    if (
        decoder.ndim != 2
        or decoder.shape[0] <= 0
        or decoder.shape[1] <= 0
        or ranks[0] < decoder.shape[0]
        or ranks[-1] > decoder.shape[1]
    ):
        raise ValueError("output decoder or input-rank grid is invalid")
    identity = torch.eye(decoder.shape[0], dtype=torch.float64)
    if not torch.allclose(
        decoder @ decoder.T,
        identity,
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise ValueError("output decoder rows must be orthonormal")
    ordered = tuple(
        sorted(
            (sequence.detached_copy() for sequence in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    families = tuple(sorted({value.family_id for value in ordered}))
    if len(families) < 4:
        raise ValueError("incremental-signal analysis requires four families")
    if any(value.candidate_h4_loss_gradient is None for value in ordered):
        raise ValueError("candidate-H4 VJPs are required")
    minimum_family_win_count = math.ceil(
        _MINIMUM_FAMILY_WIN_FRACTION * len(families)
    )

    design_spectra: list[dict[str, object]] = []
    baselines: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    for lag_count in lags:
        rows = _prepare_lag_rows(
            ordered,
            decoder=decoder,
            lag_count=lag_count,
        )
        full_weights = family_balanced_row_weights(
            rows.family_ids,
            rows.example_ids,
        ).to(torch.float64)
        spectrum, _basis = _spectrum(
            rows.design,
            full_weights,
            ridge=ridge,
        )
        design_spectra.append(
            {
                "lag_count": lag_count,
                "source_rank": rows.source_rank,
                **spectrum,
            }
        )
        folds = tuple(
            _outer_fold(
                rows,
                decoder=decoder,
                held_family=family,
                maximum_input_rank=ranks[-1],
                ridge=ridge,
            )
            for family in families
        )
        design_spectra[-1]["outer_crossfit_folds"] = tuple(
            {
                "held_family": fold.held_family,
                **fold.cross_design_spectrum,
                "inner_training_folds": fold.inner_fold_audits,
            }
            for fold in folds
        )
        macro_baseline = _mean_metrics(
            tuple(fold.baseline_metrics for fold in folds)
        )
        baselines.append(
            {
                "lag_count": lag_count,
                "l3_column_count": lag_count * rows.source_rank,
                "macro": macro_baseline,
                "per_family": tuple(
                    {
                        "held_family": fold.held_family,
                        **fold.baseline_metrics,
                    }
                    for fold in folds
                ),
                "resources": {
                    "head_parameters": rows.output_rank
                    * (
                        rows.width
                        + lag_count * rows.source_rank
                    ),
                    "head_runtime_parameter_bytes": 8
                    * rows.output_rank
                    * (
                        rows.width
                        + lag_count * rows.source_rank
                    ),
                    "head_logical_macs_per_token": rows.output_rank
                    * (
                        rows.width
                        + lag_count * rows.source_rank
                    ),
                },
            }
        )
        reused, _encoders = _cell(
            rows,
            folds,
            decoder=decoder,
            lag_count=lag_count,
            encoder_kind="reused_output_decoder",
            input_rank=int(decoder.shape[0]),
            minimum_family_win_count=minimum_family_win_count,
            ridge=ridge,
        )
        cells.append(reused)
        for input_rank in ranks:
            independent, _encoders = _cell(
                rows,
                folds,
                decoder=decoder,
                lag_count=lag_count,
                encoder_kind="independent_crossfit_h4_svd",
                input_rank=input_rank,
                minimum_family_win_count=minimum_family_win_count,
                ridge=ridge,
            )
            cells.append(independent)

    eligible = tuple(
        cell for cell in cells if bool(cell["eligible"])
    )
    selected = (
        None
        if not eligible
        else min(
            eligible,
            key=lambda cell: (
                int(
                    _mapping(
                        cell["resources"],
                        label="cell resources",
                    )["head_logical_macs_per_token"]
                ),
                -float(cell["worst_family_linearized_improvement"]),
                -float(
                    _mapping(
                        cell["macro_relative_improvement"],
                        label="macro improvement",
                    )["linearized_nll_residual_rmse"]
                ),
                int(cell["lag_count"]),
                str(cell["encoder_kind"]),
                int(cell["input_rank"]),
            ),
        )
    )
    recipe = (
        None
        if selected is None
        else _refit_final_recipe(
            ordered,
            decoder=decoder,
            cell=selected,
            maximum_input_rank=ranks[-1],
            ridge=ridge,
        )
    )
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "contract": {
            "development_role": "calibration_a_fit_only",
            "outer_validation": "leave_one_family_out",
            "inner_encoder_and_state_fit": (
                "family_blocked_crossfit"
            ),
            "maximum_inner_family_fold_count": (
                _MAXIMUM_INNER_FAMILY_FOLDS
            ),
            "weighting": "equal_family_example_row_mass",
            "l3_standardization": "weighted_rms",
            "h4_centering": False,
            "output_rank_fixed": int(decoder.shape[0]),
            "objective": "candidate_nll_vjp_metric_ridge_v1",
            "ridge": ridge,
            "solver": {
                "algorithm": "matrix_free_preconditioned_cg",
                "relative_residual_tolerance": (
                    _CG_RELATIVE_RESIDUAL_TOLERANCE
                ),
                "maximum_iterations": _CG_MAXIMUM_ITERATIONS,
                "restart_interval": _CG_RESTART_INTERVAL,
            },
            "lag_counts": lags,
            "input_ranks": ranks,
            "encoder_kinds": (
                "reused_output_decoder",
                "independent_crossfit_h4_svd",
            ),
            "gate": {
                "macro_linearized_improvement_min": (
                    _LINEARIZED_IMPROVEMENT_MIN
                ),
                "minimum_family_win_fraction": (
                    _MINIMUM_FAMILY_WIN_FRACTION
                ),
                "minimum_family_win_count": minimum_family_win_count,
                "worst_family_regression_max": (
                    _WORST_FAMILY_REGRESSION_MAX
                ),
                "secondary_metric_regression_max": (
                    _SECONDARY_REGRESSION_MAX
                ),
                "residual_energy_fraction_min": (
                    _RESIDUAL_ENERGY_FRACTION_MIN
                ),
                "all_input_ranks_identifiable": True,
            },
            "selection_rule": (
                "lowest_head_macs_then_best_worst_family_then_best_macro"
            ),
        },
        "input": {
            "fit_sequence_count": len(ordered),
            "fit_sequence_sha256s": tuple(
                value.artifact_sha256 for value in ordered
            ),
            "family_ids": families,
            "affected_row_count": sum(
                value.affected_rows for value in ordered
            ),
            "source_rank": ordered[0].source_rank,
            "width": ordered[0].width,
            "output_decoder_sha256": _tensor_sha256(decoder),
        },
        "design_spectra": tuple(design_spectra),
        "matched_l3_baselines": tuple(baselines),
        "conditioned_cells": tuple(cells),
        "selection": {
            "status": (
                "no_crossfit_incremental_signal"
                if selected is None
                else "crossfit_incremental_signal_identified"
            ),
            "eligible_cell_count": len(eligible),
            "selected_cell": selected,
            "winning_recipe": recipe,
            "selection_panel_authorized": False,
            "guard_authorized": False,
        },
        "safety": {
            "fit_only": True,
            "selection_input_capability_present": False,
            "guard_input_capability_present": False,
            "prompt_text_in_report": False,
            "token_ids_in_report": False,
            "activation_rows_in_report": False,
            "gradient_rows_in_report": False,
            "coefficient_tensors_in_report": False,
            "model_weights_in_report": False,
            "compression_claim": False,
            "latency_claim": False,
        },
    }
    payload["analysis_sha256"] = _sha256(_ANALYSIS_DOMAIN, payload)
    return payload


@dataclass(slots=True)
class _DampingFold:
    held_family: str
    baseline_prediction: Tensor
    incremental_prediction: Tensor
    target_modal: Tensor
    full_residual: Tensor
    loss_gradient: Tensor
    weights: Tensor
    baseline_metrics: dict[str, float]
    incremental_numerical_rank: int
    residualized_h4_energy_fraction: float
    encoder_capture_fraction: float
    encoder_output_decoder_overlap: float
    encoder: Tensor
    state_kernel: Tensor


def _damping_cell(
    folds: Sequence[_DampingFold],
    *,
    alpha: float,
    decoder: Tensor,
    resources: Mapping[str, int],
    minimum_family_win_count: int,
) -> dict[str, object]:
    per_family: list[dict[str, object]] = []
    residual_energy_fractions: list[float] = []
    encoder_capture_fractions: list[float] = []
    for fold in folds:
        metrics = _metrics(
            prediction_modal=(
                fold.baseline_prediction
                + alpha * fold.incremental_prediction
            ),
            target_modal=fold.target_modal,
            full_residual=fold.full_residual,
            loss_gradient=fold.loss_gradient,
            decoder=decoder,
            weights=fold.weights,
        )
        improvements = {
            name: _relative_improvement(
                metrics[name],
                fold.baseline_metrics[name],
            )
            for name in metrics
        }
        residual_energy_fractions.append(
            fold.residualized_h4_energy_fraction
        )
        encoder_capture_fractions.append(
            fold.encoder_capture_fraction
        )
        per_family.append(
            {
                "held_family": fold.held_family,
                "baseline": fold.baseline_metrics,
                "damped": metrics,
                "relative_improvement": improvements,
                "incremental_numerical_rank": (
                    fold.incremental_numerical_rank
                ),
                "residualized_h4_energy_fraction": (
                    fold.residualized_h4_energy_fraction
                ),
                "encoder_capture_fraction": (
                    fold.encoder_capture_fraction
                ),
                "encoder_output_decoder_overlap": (
                    fold.encoder_output_decoder_overlap
                ),
                "encoder_sha256": _tensor_sha256(fold.encoder),
                "undamped_state_kernel_sha256": _tensor_sha256(
                    fold.state_kernel
                ),
                "damped_state_kernel_sha256": _tensor_sha256(
                    fold.state_kernel * alpha
                ),
            }
        )
    macro_baseline = _mean_metrics(
        tuple(fold.baseline_metrics for fold in folds)
    )
    macro_damped = _mean_metrics(
        tuple(
            _mapping(
                value["damped"],
                label="damped metrics",
            )  # type: ignore[arg-type]
            for value in per_family
        )
    )
    macro_improvement = {
        name: _relative_improvement(
            macro_damped[name],
            macro_baseline[name],
        )
        for name in macro_baseline
    }
    linearized_changes = tuple(
        float(
            _mapping(
                value["relative_improvement"],
                label="family improvement",
            )["linearized_nll_residual_rmse"]
        )
        for value in per_family
    )
    partial_r2 = tuple(
        _partial_r2(
            float(
                _mapping(
                    value["damped"],
                    label="damped metrics",
                )["projected_residual_rmse"]
            ),
            float(
                _mapping(
                    value["baseline"],
                    label="baseline metrics",
                )["projected_residual_rmse"]
            ),
        )
        for value in per_family
    )
    minimum_incremental_rank = min(
        fold.incremental_numerical_rank for fold in folds
    )
    family_win_count = sum(
        change > 0.0 for change in linearized_changes
    )
    gate = {
        "all_input_ranks_identifiable": (
            minimum_incremental_rank
            == int(folds[0].encoder.shape[0])
        ),
        "macro_linearized_improvement_at_least_2pct": (
            macro_improvement["linearized_nll_residual_rmse"]
            >= _LINEARIZED_IMPROVEMENT_MIN
        ),
        "minimum_family_win_count_met": (
            family_win_count >= minimum_family_win_count
        ),
        "worst_family_regression_at_most_2pct": (
            min(linearized_changes) >= -_WORST_FAMILY_REGRESSION_MAX
        ),
        "secondary_metrics_regress_at_most_2pct": (
            macro_improvement["normalized_nll_direction_rmse"]
            >= -_SECONDARY_REGRESSION_MAX
            and macro_improvement["projected_residual_rmse"]
            >= -_SECONDARY_REGRESSION_MAX
        ),
        "residualized_h4_energy_is_nontrivial": (
            min(residual_energy_fractions)
            >= _RESIDUAL_ENERGY_FRACTION_MIN
        ),
    }
    return {
        "alpha": alpha,
        "per_family": tuple(per_family),
        "macro_baseline": macro_baseline,
        "macro_damped": macro_damped,
        "macro_relative_improvement": macro_improvement,
        "family_linearized_win_count": family_win_count,
        "minimum_family_win_count": minimum_family_win_count,
        "worst_family_linearized_improvement": _finite_float(
            min(linearized_changes)
        ),
        "macro_partial_r2": _finite_float(
            sum(partial_r2) / len(partial_r2)
        ),
        "minimum_incremental_numerical_rank": (
            minimum_incremental_rank
        ),
        "mean_residualized_h4_energy_fraction": _finite_float(
            sum(residual_energy_fractions)
            / len(residual_energy_fractions)
        ),
        "mean_encoder_capture_fraction": _finite_float(
            sum(encoder_capture_fractions)
            / len(encoder_capture_fractions)
        ),
        "encoder_stability": _encoder_stability(
            tuple(fold.encoder for fold in folds)
        ),
        "resources": dict(resources),
        "gate": gate,
        "eligible": all(gate.values()),
    }


def analyze_gemma_h4_damping(
    *,
    sequences: Sequence[GemmaTwoHeadFitSequence],
    output_decoder: Tensor,
    lag_count: int = 16,
    input_rank: int = 32,
    damping_alphas: Sequence[float] = _DEFAULT_DAMPING_ALPHAS,
    ridge: float = 1.0e-6,
    hypothesis_cell: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate a fixed independent H4 head under scalar interpolation."""

    lag_count = _positive_int(lag_count, label="lag_count")
    input_rank = _positive_int(input_rank, label="input_rank")
    alphas = _canonical_damping_alphas(damping_alphas)
    ridge = _positive_finite(ridge, label="ridge")
    decoder = _canonical_encoder(output_decoder)
    if (
        decoder.ndim != 2
        or decoder.shape[0] <= 0
        or input_rank < decoder.shape[0]
        or input_rank > decoder.shape[1]
        or not torch.allclose(
            decoder @ decoder.T,
            torch.eye(decoder.shape[0], dtype=torch.float64),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
    ):
        raise ValueError("fixed damping geometry is invalid")
    ordered = tuple(
        sorted(
            (sequence.detached_copy() for sequence in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    families = tuple(sorted({value.family_id for value in ordered}))
    if len(families) < 4:
        raise ValueError("H4 damping requires at least four families")
    if any(value.candidate_h4_loss_gradient is None for value in ordered):
        raise ValueError("candidate-H4 VJPs are required")
    minimum_family_win_count = math.ceil(
        _MINIMUM_FAMILY_WIN_FRACTION * len(families)
    )
    rows = _prepare_lag_rows(
        ordered,
        decoder=decoder,
        lag_count=lag_count,
    )
    full_weights = family_balanced_row_weights(
        rows.family_ids,
        rows.example_ids,
    ).to(torch.float64)
    full_spectrum, _full_basis = _spectrum(
        rows.design,
        full_weights,
        ridge=ridge,
    )
    outer_folds = tuple(
        _outer_fold(
            rows,
            decoder=decoder,
            held_family=family,
            maximum_input_rank=input_rank,
            ridge=ridge,
        )
        for family in families
    )
    damping_folds: list[_DampingFold] = []
    for fold in outer_folds:
        encoder = fold.independent_encoder_32[:input_rank].contiguous()
        cross_state = fold.cross_h4_residual @ encoder.T
        state_kernel = _candidate_metric_coefficients(
            design=cross_state,
            target_modal=fold.cross_modal_residual,
            full_residual=fold.cross_full_residual,
            loss_gradient=fold.cross_gradient,
            decoder=decoder,
            weights=fold.cross_weights,
            ridge=ridge,
        )
        test_state = fold.test_h4_residual @ encoder.T
        test_weights = _selected_weights(rows, fold.test_mask)
        h4_energy = float(
            (
                rows.realized_h4[fold.train_mask].square().sum(dim=1)
                * _selected_weights(rows, fold.train_mask)
            ).sum()
        )
        residual_energy = float(
            (
                fold.cross_h4_residual.square().sum(dim=1)
                * fold.cross_weights
            ).sum()
        )
        state_energy = float(
            (
                cross_state.square().sum(dim=1)
                * fold.cross_weights
            ).sum()
        )
        damping_folds.append(
            _DampingFold(
                held_family=fold.held_family,
                baseline_prediction=(
                    rows.design[fold.test_mask]
                    @ fold.baseline_coefficients
                ),
                incremental_prediction=test_state @ state_kernel,
                target_modal=rows.target_modal[fold.test_mask],
                full_residual=rows.full_residual[fold.test_mask],
                loss_gradient=rows.loss_gradient[fold.test_mask],
                weights=test_weights,
                baseline_metrics=fold.baseline_metrics,
                incremental_numerical_rank=_incremental_rank(
                    fold.cross_design,
                    cross_state,
                    fold.cross_weights,
                    design_basis=fold.cross_design_basis,
                    design_sigma_max=float(
                        fold.cross_design_spectrum["sigma_max"]
                    ),
                ),
                residualized_h4_energy_fraction=_finite_float(
                    0.0
                    if h4_energy == 0.0
                    else residual_energy / h4_energy
                ),
                encoder_capture_fraction=_finite_float(
                    0.0
                    if residual_energy == 0.0
                    else state_energy / residual_energy
                ),
                encoder_output_decoder_overlap=_finite_float(
                    (encoder @ decoder.T).square().sum() / input_rank
                ),
                encoder=encoder,
                state_kernel=state_kernel,
            )
        )
    resources = _resource_cost(
        lag_count=lag_count,
        source_rank=rows.source_rank,
        width=rows.width,
        output_rank=rows.output_rank,
        encoder_kind="independent_crossfit_h4_svd",
        input_rank=input_rank,
    )
    alpha_cells = tuple(
        _damping_cell(
            damping_folds,
            alpha=alpha,
            decoder=decoder,
            resources=resources,
            minimum_family_win_count=minimum_family_win_count,
        )
        for alpha in alphas
    )
    if hypothesis_cell is not None:
        _assert_alpha_one_reproduces_hypothesis(
            {
                "contract": {
                    "lag_count": lag_count,
                    "input_rank": input_rank,
                    "output_rank": rows.output_rank,
                },
                "input": {"source_rank": rows.source_rank},
                "alpha_cells": alpha_cells,
            },
            hypothesis_cell,
        )
    eligible = tuple(
        cell for cell in alpha_cells if bool(cell["eligible"])
    )
    selected = None if not eligible else min(
        eligible,
        key=lambda cell: float(cell["alpha"]),
    )
    recipe = (
        None
        if selected is None
        else _refit_final_recipe(
            ordered,
            decoder=decoder,
            cell={
                **dict(selected),
                "lag_count": lag_count,
                "encoder_kind": "independent_crossfit_h4_svd",
                "input_rank": input_rank,
            },
            maximum_input_rank=input_rank,
            ridge=ridge,
            state_scale=float(selected["alpha"]),
            recipe_domain=_DAMPING_RECIPE_DOMAIN,
        )
    )
    payload: dict[str, object] = {
        "schema": _DAMPING_SCHEMA,
        "format_version": _DAMPING_FORMAT_VERSION,
        "contract": {
            "development_role": "calibration_a_fit_only",
            "hypothesis": "fixed_lag16_independent_h4_rank32",
            "outer_validation": "leave_one_family_out",
            "inner_encoder_and_state_fit": "family_blocked_crossfit",
            "maximum_inner_family_fold_count": (
                _MAXIMUM_INNER_FAMILY_FOLDS
            ),
            "weighting": "equal_family_example_row_mass",
            "objective": "candidate_nll_vjp_metric_ridge_v1",
            "ridge": ridge,
            "lag_count": lag_count,
            "input_rank": input_rank,
            "output_rank": rows.output_rank,
            "damping_alphas": alphas,
            "baseline_alpha": 0.0,
            "selection_rule": "smallest_alpha_passing_all_gates",
            "gate": {
                "macro_linearized_improvement_min": (
                    _LINEARIZED_IMPROVEMENT_MIN
                ),
                "minimum_family_win_fraction": (
                    _MINIMUM_FAMILY_WIN_FRACTION
                ),
                "minimum_family_win_count": minimum_family_win_count,
                "worst_family_regression_max": (
                    _WORST_FAMILY_REGRESSION_MAX
                ),
                "secondary_metric_regression_max": (
                    _SECONDARY_REGRESSION_MAX
                ),
                "residual_energy_fraction_min": (
                    _RESIDUAL_ENERGY_FRACTION_MIN
                ),
            },
        },
        "input": {
            "fit_sequence_count": len(ordered),
            "fit_sequence_sha256s": tuple(
                value.artifact_sha256 for value in ordered
            ),
            "family_ids": families,
            "affected_row_count": sum(
                value.affected_rows for value in ordered
            ),
            "source_rank": rows.source_rank,
            "width": rows.width,
            "output_decoder_sha256": _tensor_sha256(decoder),
        },
        "design_spectrum": {
            "lag_count": lag_count,
            "source_rank": rows.source_rank,
            **full_spectrum,
            "outer_crossfit_folds": tuple(
                {
                    "held_family": fold.held_family,
                    **fold.cross_design_spectrum,
                    "inner_training_folds": fold.inner_fold_audits,
                }
                for fold in outer_folds
            ),
        },
        "alpha_cells": alpha_cells,
        "selection": {
            "status": (
                "no_robust_damping_coefficient"
                if selected is None
                else "fit_only_damping_recipe_frozen"
            ),
            "eligible_alpha_count": len(eligible),
            "selected_alpha_cell": selected,
            "winning_recipe": recipe,
            "selection_panel_authorized": False,
            "guard_authorized": False,
        },
        "safety": {
            "fit_only": True,
            "selection_input_capability_present": False,
            "guard_input_capability_present": False,
            "prompt_text_in_report": False,
            "token_ids_in_report": False,
            "activation_rows_in_report": False,
            "gradient_rows_in_report": False,
            "coefficient_tensors_in_report": False,
            "model_weights_in_report": False,
            "compression_claim": False,
            "latency_claim": False,
        },
    }
    payload["analysis_sha256"] = _sha256(
        _DAMPING_ANALYSIS_DOMAIN,
        payload,
    )
    return payload


def _parent_cell_from_damping_cell(
    cell: Mapping[str, object],
    *,
    lag_count: int,
    source_rank: int,
    input_rank: int,
    output_rank: int,
) -> dict[str, object]:
    """Normalize alpha=1 into the exact v2 conditioned-cell shape."""

    per_family: list[dict[str, object]] = []
    for raw_family in cell["per_family"]:  # type: ignore[union-attr]
        family = _mapping(raw_family, label="damping family cell")
        per_family.append(
            {
                "held_family": family["held_family"],
                "baseline": family["baseline"],
                "conditioned": family["damped"],
                "relative_improvement": family["relative_improvement"],
                "incremental_numerical_rank": (
                    family["incremental_numerical_rank"]
                ),
                "residualized_h4_energy_fraction": (
                    family["residualized_h4_energy_fraction"]
                ),
                "encoder_capture_fraction": (
                    family["encoder_capture_fraction"]
                ),
                "encoder_output_decoder_overlap": (
                    family["encoder_output_decoder_overlap"]
                ),
                "encoder_sha256": family["encoder_sha256"],
                "state_kernel_sha256": (
                    family["undamped_state_kernel_sha256"]
                ),
            }
        )
    return {
        "lag_count": lag_count,
        "l3_column_count": lag_count * source_rank,
        "encoder_kind": "independent_crossfit_h4_svd",
        "input_rank": input_rank,
        "output_rank": output_rank,
        "per_family": tuple(per_family),
        "macro_baseline": cell["macro_baseline"],
        "macro_conditioned": cell["macro_damped"],
        "macro_relative_improvement": cell["macro_relative_improvement"],
        "family_linearized_win_count": (
            cell["family_linearized_win_count"]
        ),
        "minimum_family_win_count": cell["minimum_family_win_count"],
        "worst_family_linearized_improvement": (
            cell["worst_family_linearized_improvement"]
        ),
        "macro_partial_r2": cell["macro_partial_r2"],
        "minimum_incremental_numerical_rank": (
            cell["minimum_incremental_numerical_rank"]
        ),
        "mean_residualized_h4_energy_fraction": (
            cell["mean_residualized_h4_energy_fraction"]
        ),
        "mean_encoder_capture_fraction": (
            cell["mean_encoder_capture_fraction"]
        ),
        "encoder_stability": cell["encoder_stability"],
        "resources": cell["resources"],
        "gate": cell["gate"],
        "eligible": cell["eligible"],
    }


def _assert_alpha_one_reproduces_hypothesis(
    diagnostic: Mapping[str, object],
    hypothesis_cell: Mapping[str, object],
) -> dict[str, object]:
    cells = tuple(diagnostic["alpha_cells"])  # type: ignore[arg-type]
    alpha_one = tuple(
        _mapping(value, label="alpha cell")
        for value in cells
        if float(_mapping(value, label="alpha cell")["alpha"]) == 1.0
    )
    if len(alpha_one) != 1:
        raise RuntimeError(
            "fixed damping grid lacks a unique alpha=1 control"
        )
    cell = alpha_one[0]
    for raw_family in cell["per_family"]:  # type: ignore[union-attr]
        family = _mapping(raw_family, label="alpha=1 family cell")
        if (
            family["undamped_state_kernel_sha256"]
            != family["damped_state_kernel_sha256"]
        ):
            raise RuntimeError("alpha=1 changed a state-kernel hash")
    contract = _mapping(
        diagnostic["contract"],
        label="damping diagnostic contract",
    )
    input_metadata = _mapping(
        diagnostic["input"],
        label="damping diagnostic input",
    )
    normalized = _parent_cell_from_damping_cell(
        cell,
        lag_count=int(contract["lag_count"]),
        source_rank=int(input_metadata["source_rank"]),
        input_rank=int(contract["input_rank"]),
        output_rank=int(contract["output_rank"]),
    )
    if _canonical_json_bytes(normalized) != _canonical_json_bytes(
        hypothesis_cell
    ):
        raise RuntimeError("alpha=1 hypothesis reproduction mismatch")
    return {
        "status": "exact_match",
        "alpha": 1.0,
        "source_cell_sha256": _sha256(
            _DAMPING_HYPOTHESIS_DOMAIN,
            hypothesis_cell,
        ),
        "recomputed_cell_sha256": _sha256(
            _DAMPING_HYPOTHESIS_DOMAIN,
            normalized,
        ),
        "per_family_encoder_and_state_kernel_hashes_matched": True,
        "scalar_cell_mapping_matched": True,
    }


def _load_damping_hypothesis(
    report_path: Path | str,
    *,
    expected_report_sha256: str,
    expected_report_file_sha256: str,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Authenticate the expanded fit-only result as a hypothesis declaration."""

    report = _read_json_mapping(
        report_path,
        label="damping hypothesis report",
    )
    if _file_sha256(report_path) != expected_report_file_sha256:
        raise ValueError("damping hypothesis report file hash differs")
    bound_report = dict(report)
    observed_report_sha256 = bound_report.pop("report_sha256", None)
    if (
        report.get("schema") != _SCHEMA
        or report.get("format_version") != _FORMAT_VERSION
        or observed_report_sha256 != expected_report_sha256
        or observed_report_sha256
        != _sha256(_REPORT_DOMAIN, bound_report)
    ):
        raise ValueError("damping hypothesis report integrity differs")

    diagnostic = _mapping(
        report.get("diagnostic"),
        label="damping hypothesis diagnostic",
    )
    bound_diagnostic = dict(diagnostic)
    observed_analysis_sha256 = bound_diagnostic.pop(
        "analysis_sha256",
        None,
    )
    if (
        diagnostic.get("schema") != _SCHEMA
        or diagnostic.get("format_version") != _FORMAT_VERSION
        or observed_analysis_sha256
        != _sha256(_ANALYSIS_DOMAIN, bound_diagnostic)
    ):
        raise ValueError("damping hypothesis analysis integrity differs")

    top_safety = _mapping(
        report.get("safety"),
        label="damping hypothesis safety",
    )
    required_true = ("development_only", "fit_role_opened")
    required_false = (
        "selection_input_capability_present",
        "selection_role_opened",
        "guard_input_capability_present",
        "guard_role_opened",
        "calibration_b_loader_present",
        "calibration_b_opened",
        "prompt_text_in_report",
        "token_ids_in_report",
        "activation_rows_in_report",
        "gradient_rows_in_report",
        "fit_sequences_in_report",
        "coefficient_tensors_in_report",
        "model_weights_in_report",
        "compression_claim",
        "latency_claim",
    )
    if (
        any(not bool(top_safety.get(name)) for name in required_true)
        or any(bool(top_safety.get(name)) for name in required_false)
    ):
        raise ValueError("damping hypothesis safety contract differs")
    diagnostic_safety = _mapping(
        diagnostic.get("safety"),
        label="damping hypothesis diagnostic safety",
    )
    if (
        not bool(diagnostic_safety.get("fit_only"))
        or any(
            bool(diagnostic_safety.get(name))
            for name in (
                "selection_input_capability_present",
                "guard_input_capability_present",
                "prompt_text_in_report",
                "token_ids_in_report",
                "activation_rows_in_report",
                "gradient_rows_in_report",
                "coefficient_tensors_in_report",
                "model_weights_in_report",
                "compression_claim",
                "latency_claim",
            )
        )
    ):
        raise ValueError("damping hypothesis diagnostic is not fit-only")

    spec = _mapping(report.get("spec"), label="damping hypothesis spec")
    expected_spec = {
        "corpus_artifact_sha256": (
            "e4804338dbc3e76a84bf0483526ac9bab4e5f8aeaa86a32283832fed25f4b766"
        ),
        "fit_manifest_sha256": (
            "75ae2045fd16de3128e3eef3a0177422bf2bad87a7bbd83ffeeb80ba2c1aac1d"
        ),
        "fit_binding_sha256": (
            "7e71e7abfee83cdcbc94e84eb12bc7947733cb9202ba1adeeb325f268ec2cab5"
        ),
        "maximum_sequence_length": 256,
        "device": "cpu",
        "dtype": "float32",
        "output_rank": _DAMPING_OUTPUT_RANK,
        "ridge": 1.0e-6,
        "expected_fit_example_count": _DAMPING_FIT_EXAMPLE_COUNT,
        "expected_fit_family_count": _DAMPING_FIT_FAMILY_COUNT,
        "observed_fit_example_count": _DAMPING_FIT_EXAMPLE_COUNT,
        "observed_fit_family_count": _DAMPING_FIT_FAMILY_COUNT,
        "selection_input_accepted": False,
        "guard_input_accepted": False,
    }
    if any(spec.get(name) != value for name, value in expected_spec.items()):
        raise ValueError("damping hypothesis fit specification differs")
    if (
        tuple(spec.get("lag_counts", ())) != _DEFAULT_LAG_COUNTS
        or tuple(spec.get("input_ranks", ())) != _DEFAULT_INPUT_RANKS
    ):
        raise ValueError("damping hypothesis search grid differs")

    lineage = _mapping(
        report.get("fit_corpus_lineage"),
        label="damping hypothesis fit lineage",
    )
    expected_lineage = {
        "kind": "fit_role_only_replacement",
        "parent_corpus_artifact_sha256": (
            "55015297b5f06006ac0a03fbb3fa38a15b4d6815625fe35b76ad4a52e3aa066b"
        ),
        "parent_fit_manifest_sha256": (
            "0406ba16f1070f9fb1b9030d9e9ed8adedb50785cd9b798bac0b82a1427751bf"
        ),
        "replacement_corpus_artifact_sha256": (
            "e4804338dbc3e76a84bf0483526ac9bab4e5f8aeaa86a32283832fed25f4b766"
        ),
        "replacement_fit_manifest_sha256": (
            "75ae2045fd16de3128e3eef3a0177422bf2bad87a7bbd83ffeeb80ba2c1aac1d"
        ),
        "preserved_role_manifest_sha256s": {
            "calibration_a_selection": (
                "f0d561339b6255b6a942ee657580fa54da7f4d1ec96f502b70b4d8c7a7c29f4e"
            ),
            "calibration_a_guard": (
                "53dc29646354beed29ceec9969a820f661d0499c2aad25b5bbcea361f1bbc3b9"
            ),
        },
        "preserved_role_input_file_sha256s": {
            "calibration_a_selection": (
                "7177855ef7e6a21660bce2dd66171e174706e4185e322dbb258f8bd4a4a5b5b9"
            ),
            "calibration_a_guard": (
                "fa6e0885de5df15eda9964e0ab6f3a3931a62696c6da3ed12d07ff997e5786e2"
            ),
        },
    }
    if _canonical_json_bytes(lineage) != _canonical_json_bytes(
        expected_lineage
    ):
        raise ValueError("damping hypothesis fit lineage differs")

    selection = _mapping(
        diagnostic.get("selection"),
        label="damping hypothesis selection",
    )
    if selection != {
        "status": "no_crossfit_incremental_signal",
        "eligible_cell_count": 0,
        "selected_cell": None,
        "winning_recipe": None,
        "selection_panel_authorized": False,
        "guard_authorized": False,
    }:
        raise ValueError("damping hypothesis selection state differs")

    input_metadata = _mapping(
        diagnostic.get("input"),
        label="damping hypothesis input",
    )
    if (
        input_metadata.get("fit_sequence_count")
        != _DAMPING_FIT_EXAMPLE_COUNT
        or len(tuple(input_metadata.get("fit_sequence_sha256s", ())))
        != _DAMPING_FIT_EXAMPLE_COUNT
        or len(tuple(input_metadata.get("family_ids", ())))
        != _DAMPING_FIT_FAMILY_COUNT
        or input_metadata.get("affected_row_count")
        != _DAMPING_FIT_ROW_COUNT
        or input_metadata.get("source_rank") != 64
        or input_metadata.get("width") != 640
        or input_metadata.get("output_decoder_sha256")
        != "05bd5a372ab7fb4cddce10650f91817699a0a49af64fbba4e087ba649ba1ee79"
    ):
        raise ValueError("damping hypothesis trace geometry differs")

    candidates = tuple(
        _mapping(value, label="hypothesis conditioned cell")
        for value in diagnostic.get("conditioned_cells", ())  # type: ignore[arg-type]
        if (
            int(_mapping(value, label="hypothesis conditioned cell")[
                "lag_count"
            ])
            == _DAMPING_LAG_COUNT
            and str(
                _mapping(value, label="hypothesis conditioned cell")[
                    "encoder_kind"
                ]
            )
            == "independent_crossfit_h4_svd"
            and int(
                _mapping(value, label="hypothesis conditioned cell")[
                    "input_rank"
                ]
            )
            == _DAMPING_INPUT_RANK
        )
    )
    if len(candidates) != 1:
        raise ValueError("damping hypothesis cell is not unique")
    cell = dict(candidates[0])
    expected_gate = {
        "all_input_ranks_identifiable": True,
        "macro_linearized_improvement_at_least_2pct": True,
        "minimum_family_win_count_met": True,
        "worst_family_regression_at_most_2pct": False,
        "secondary_metrics_regress_at_most_2pct": True,
        "residualized_h4_energy_is_nontrivial": True,
    }
    if (
        cell.get("output_rank") != _DAMPING_OUTPUT_RANK
        or cell.get("eligible") is not False
        or cell.get("family_linearized_win_count") != 7
        or cell.get("minimum_family_win_count") != 6
        or cell.get("worst_family_linearized_improvement")
        != -0.030000716249646953
        or _mapping(
            cell.get("macro_relative_improvement"),
            label="hypothesis macro improvement",
        ).get("linearized_nll_residual_rmse")
        != 0.04379309187029412
        or dict(
            _mapping(cell.get("gate"), label="hypothesis cell gate")
        )
        != expected_gate
        or dict(
            _mapping(cell.get("resources"), label="hypothesis resources")
        )
        != {
            "head_parameters": 34_048,
            "head_runtime_parameter_bytes": 272_384,
            "head_logical_macs_per_token": 34_048,
            "conditioning_parameters": 20_736,
            "conditioning_runtime_parameter_bytes": 165_888,
            "conditioning_logical_macs_per_token": 20_736,
        }
    ):
        raise ValueError("damping hypothesis cell evidence differs")

    decoder_audit = _mapping(
        report.get("output_decoder"),
        label="damping hypothesis decoder",
    )
    if (
        decoder_audit.get("output_rank") != _DAMPING_OUTPUT_RANK
        or decoder_audit.get("decoder_sha256")
        != input_metadata["output_decoder_sha256"]
    ):
        raise ValueError("damping hypothesis decoder binding differs")
    hypothesis_snapshot = {
        "source_report_file_sha256": expected_report_file_sha256,
        "source_report_sha256": expected_report_sha256,
        "source_analysis_sha256": observed_analysis_sha256,
        "source_cell_sha256": _sha256(
            _DAMPING_HYPOTHESIS_DOMAIN,
            cell,
        ),
        "lag_count": _DAMPING_LAG_COUNT,
        "encoder_kind": "independent_crossfit_h4_svd",
        "input_rank": _DAMPING_INPUT_RANK,
        "output_rank": _DAMPING_OUTPUT_RANK,
        "damping_alphas": _DEFAULT_DAMPING_ALPHAS,
        "selection_rule": "smallest_alpha_passing_all_gates",
    }
    provenance = {
        **hypothesis_snapshot,
        "source_schema": _SCHEMA,
        "source_format_version": _FORMAT_VERSION,
        "source_selection_status": selection["status"],
        "source_corpus_artifact_sha256": spec[
            "corpus_artifact_sha256"
        ],
        "source_fit_manifest_sha256": spec["fit_manifest_sha256"],
        "source_fit_binding_sha256": spec["fit_binding_sha256"],
        "source_fit_sequence_sha256s": tuple(
            input_metadata["fit_sequence_sha256s"]  # type: ignore[arg-type]
        ),
        "source_family_ids": tuple(
            input_metadata["family_ids"]  # type: ignore[arg-type]
        ),
        "source_affected_row_count": input_metadata[
            "affected_row_count"
        ],
        "source_output_decoder_sha256": input_metadata[
            "output_decoder_sha256"
        ],
        "source_accepted_x4_provenance_sha256": _sha256(
            _DAMPING_HYPOTHESIS_DOMAIN,
            report["accepted_x4_provenance"],
        ),
        "source_cell_snapshot": {
            "macro_linearized_improvement": 0.04379309187029412,
            "family_linearized_win_count": 7,
            "worst_family_linearized_improvement": (
                -0.030000716249646953
            ),
            "failed_gate_names": (
                "worst_family_regression_at_most_2pct",
            ),
            "resources": cell["resources"],
        },
        "use": "hypothesis_selection_only",
        "source_trace_rows_reused": False,
        "source_coefficients_reused": False,
        "source_metrics_used_for_scoring": False,
        "hypothesis_sha256": _sha256(
            _DAMPING_HYPOTHESIS_DOMAIN,
            hypothesis_snapshot,
        ),
    }
    return cell, provenance


def derive_candidate_h4_output_decoder(
    sequences: Sequence[GemmaTwoHeadFitSequence],
    *,
    output_rank: int,
) -> tuple[Tensor, dict[str, object]]:
    """Reproduce the candidate-H4 family-macro residual map direction rule."""

    output_rank = _positive_int(output_rank, label="output_rank")
    ordered = tuple(
        sorted(
            (sequence.detached_copy() for sequence in sequences),
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    if not ordered:
        raise ValueError("candidate-H4 residual map requires fit sequences")
    family_covariances: dict[str, Tensor] = {}
    family_fishers: dict[str, Tensor] = {}
    family_examples: dict[str, int] = {}
    width = ordered[0].width
    for sequence in ordered:
        gradient_source = sequence.candidate_h4_loss_gradient
        if gradient_source is None or sequence.width != width:
            raise ValueError("candidate-H4 map requires common VJP geometry")
        residual = sequence.h4_residual_rows.to(torch.float64)
        gradient = gradient_source[
            sequence.target_affected_mask
        ].to(torch.float64)
        residual_square = residual.square().sum(dim=1)
        gradient_square = gradient.square().sum(dim=1)
        alignment = (residual * gradient).sum(dim=1).square() / (
            residual_square * gradient_square + 1.0e-30
        )
        weighted = residual * (
            1.0 + alignment.clamp(min=0.0, max=1.0)
        ).sqrt().unsqueeze(1)
        covariance = weighted.T @ weighted / residual.shape[0]
        fisher = gradient.T @ gradient / residual.shape[0]
        family = sequence.family_id
        if family not in family_covariances:
            family_covariances[family] = covariance
            family_fishers[family] = fisher
            family_examples[family] = 1
        else:
            family_covariances[family] += covariance
            family_fishers[family] += fisher
            family_examples[family] += 1
    macro_covariance = sum(
        family_covariances[family] / family_examples[family]
        for family in sorted(family_covariances)
    ) / len(family_covariances)
    macro_fisher = sum(
        family_fishers[family] / family_examples[family]
        for family in sorted(family_fishers)
    ) / len(family_fishers)
    macro_covariance = (
        (macro_covariance + macro_covariance.T) * 0.5
    ).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(macro_covariance)
    order = torch.argsort(eigenvalues, descending=True)
    count = min(
        output_rank,
        width,
        int((eigenvalues > 0.0).sum()),
    )
    if count != output_rank:
        raise ValueError("candidate-H4 residual map lacks requested rank")
    selected_values = eigenvalues.index_select(
        0,
        order[:count],
    ).clamp_min(0.0)
    directions = _canonical_encoder(
        eigenvectors.index_select(1, order[:count]).T
    )
    loss_couplings = torch.einsum(
        "kw,wx,kx->k",
        directions,
        macro_fisher,
        directions,
    ).clamp_min(0.0)
    return directions, {
        "output_rank": count,
        "decoder_sha256": _tensor_sha256(directions),
        "selected_residual_eigenvalues": tuple(
            _finite_float(value) for value in selected_values
        ),
        "selected_loss_couplings": tuple(
            _finite_float(value) for value in loss_couplings
        ),
        "total_residual_energy": _finite_float(
            eigenvalues.clamp_min(0.0).sum()
        ),
    }


def _accepted_x4_artifact(
    *,
    report_path: Path | str,
    candidate_path: Path | str,
    expected_candidate_file_sha256: str,
) -> tuple[
    Mapping[str, object],
    GemmaL3L4TwoHeadArtifact,
    dict[str, object],
]:
    report = _read_json_mapping(report_path, label="accepted X4 report")
    bound_report = dict(report)
    observed_report_sha256 = bound_report.pop("report_sha256", None)
    if (
        report.get("schema")
        != "fisher_graph.gemma3_l3_l4_progressive_a_campaign"
        or report.get("format_version") != 4
        or observed_report_sha256
        != _domain_sha256(_CAMPAIGN_REPORT_DOMAIN, bound_report)
    ):
        raise ValueError("accepted X4 report integrity differs")
    safety = _mapping(report.get("safety"), label="accepted report safety")
    if (
        bool(safety.get("guard_opened"))
        or bool(safety.get("guard_consumed"))
        or bool(safety.get("calibration_b_opened"))
        or not bool(safety.get("guard_deferred"))
    ):
        raise ValueError("accepted X4 report is not a deferred safe lineage")
    artifact_metadata = _mapping(
        report.get("artifact"),
        label="accepted report artifact",
    )
    candidate_sha256 = _file_sha256(candidate_path)
    if (
        candidate_sha256 != expected_candidate_file_sha256
        or artifact_metadata.get("candidate_tensor_file_sha256")
        != candidate_sha256
        or bool(artifact_metadata.get("contains_model_weights"))
    ):
        raise ValueError("accepted X4 candidate file binding differs")
    raw = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("accepted X4 candidate state must be a mapping")
    artifact = GemmaL3L4TwoHeadArtifact.from_state_dict(raw)
    result = _mapping(report.get("result"), label="accepted report result")
    final_candidate = _mapping(
        result.get("final_candidate"),
        label="accepted final candidate",
    )
    if (
        final_candidate.get("artifact_sha256")
        != artifact.artifact_sha256
        or final_candidate.get("execution_sha256")
        != artifact.execution_sha256
        or final_candidate.get("runtime_binding_sha256")
        != artifact.runtime_binding_sha256
        or artifact.head(_X4_SITE) is None
        or artifact.head(_H4_SITE) is not None
    ):
        raise ValueError("accepted artifact is not the report's X4-only winner")
    provenance = {
        "report_file_sha256": _file_sha256(report_path),
        "report_sha256": str(report["report_sha256"]),
        "campaign_spec_sha256": str(report["campaign_spec_sha256"]),
        "protocol_sha256": str(report["protocol_sha256"]),
        "transcript_sha256": str(report["transcript_sha256"]),
        "candidate_file_sha256": candidate_sha256,
        "candidate_artifact_sha256": artifact.artifact_sha256,
        "candidate_execution_sha256": artifact.execution_sha256,
        "candidate_runtime_binding_sha256": (
            artifact.runtime_binding_sha256
        ),
    }
    return report, artifact, provenance


def _publish_report(path: Path | str, report: Mapping[str, object]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite diagnostic output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        .encode("utf-8")
        + b"\n"
    )
    stage: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            stage = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, destination)
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_h4_incremental_signal_diagnostic(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_CORPUS,
    fit_input_path: Path | str = DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    accepted_x4_report_path: Path | str = _DEFAULT_ACCEPTED_REPORT,
    accepted_x4_candidate_path: Path | str = _DEFAULT_ACCEPTED_CANDIDATE,
    expected_accepted_x4_candidate_file_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_H4_INCREMENTAL_SIGNAL_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    output_rank: int = 8,
    lag_counts: Sequence[int] = _DEFAULT_LAG_COUNTS,
    input_ranks: Sequence[int] = _DEFAULT_INPUT_RANKS,
    ridge: float = 1.0e-6,
    expected_fit_example_count: int | None = None,
    expected_fit_family_count: int | None = None,
    _analysis_mode: str = "incremental_grid",
    _damping_hypothesis_cell: Mapping[str, object] | None = None,
    _damping_hypothesis_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Collect accepted-X4 A-fit traces and publish a scalar-only diagnostic."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite diagnostic output")
    if _analysis_mode not in ("incremental_grid", "fixed_head_damping"):
        raise ValueError("unknown incremental-signal analysis mode")
    damping_mode = _analysis_mode == "fixed_head_damping"
    output_rank = _positive_int(output_rank, label="output_rank")
    lags = _canonical_positive_ints(lag_counts, label="lag_counts")
    ranks = _canonical_positive_ints(input_ranks, label="input_ranks")
    ridge = _positive_finite(ridge, label="ridge")
    if expected_fit_example_count is not None:
        expected_fit_example_count = _positive_int(
            expected_fit_example_count,
            label="expected_fit_example_count",
        )
    if expected_fit_family_count is not None:
        expected_fit_family_count = _positive_int(
            expected_fit_family_count,
            label="expected_fit_family_count",
        )
    if damping_mode:
        if (
            output_rank != _DAMPING_OUTPUT_RANK
            or lags != (_DAMPING_LAG_COUNT,)
            or ranks != (_DAMPING_INPUT_RANK,)
            or ridge != 1.0e-6
            or device_name != "cpu"
            or dtype != "float32"
            or expected_fit_example_count
            != _DAMPING_FIT_EXAMPLE_COUNT
            or expected_fit_family_count
            != _DAMPING_FIT_FAMILY_COUNT
            or _damping_hypothesis_cell is None
            or _damping_hypothesis_provenance is None
        ):
            raise ValueError("fixed damping preregistration differs")
        report_schema = _DAMPING_SCHEMA
        report_format_version = _DAMPING_FORMAT_VERSION
        report_domain = _DAMPING_REPORT_DOMAIN
    else:
        if (
            _damping_hypothesis_cell is not None
            or _damping_hypothesis_provenance is not None
        ):
            raise ValueError(
                "incremental grid cannot accept a damping hypothesis"
            )
        report_schema = _SCHEMA
        report_format_version = _FORMAT_VERSION
        report_domain = _REPORT_DOMAIN
    accepted_report, accepted_artifact, accepted_provenance = (
        _accepted_x4_artifact(
            report_path=accepted_x4_report_path,
            candidate_path=accepted_x4_candidate_path,
            expected_candidate_file_sha256=(
                expected_accepted_x4_candidate_file_sha256
            ),
        )
    )
    legacy_protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    legacy_protocol.validate_integrity()
    legacy_metadata = legacy_protocol.metadata()
    tokenizer_contract = dict(
        _mapping(legacy_metadata["tokenizer"], label="frozen tokenizer")
    )
    max_length = int(tokenizer_contract["max_length"])
    corpus, fit_input = load_gemma3_l3_l4_progressive_a_fit_role(
        corpus_artifact_path,
        fit_input_path=fit_input_path,
        tokenizer_contract=tokenizer_contract,
    )
    accepted_corpus = _mapping(
        accepted_report.get("corpus"),
        label="accepted report corpus",
    )
    accepted_corpus_artifact = (
        Gemma3L3L4ProgressiveACorpusArtifact.from_dict(accepted_corpus)
    )
    if accepted_corpus_artifact.artifact_sha256 == corpus.artifact_sha256:
        fit_corpus_lineage: dict[str, object] = {
            "kind": "accepted_x4_corpus",
            "parent_corpus_artifact_sha256": (
                accepted_corpus_artifact.artifact_sha256
            ),
            "replacement_corpus_artifact_sha256": corpus.artifact_sha256,
        }
    else:
        fit_corpus_lineage = (
            gemma3_l3_l4_progressive_a_fit_replacement_lineage(
                accepted_corpus_artifact,
                corpus,
            )
        )
    tokenizer, live_contract = _load_and_validate_frozen_local_tokenizer(
        protocol=legacy_protocol,
    )
    if _canonical_json_bytes(live_contract) != _canonical_json_bytes(
        tokenizer_contract
    ):
        raise ValueError("live tokenizer differs from diagnostic contract")
    token_device = torch.device(str(tokenizer_contract["device"]))
    fit_panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=fit_input,
        view=corpus.role_view("calibration_a_fit"),
        max_length=max_length,
        device=token_device,
        forbidden_manifest_sha256s=(
            corpus.forbidden_assessment_manifest_sha256s
        ),
    )
    observed_fit_families = {
        example.family_id for example in fit_panel.examples
    }
    if (
        expected_fit_example_count is not None
        and len(fit_panel.examples) != expected_fit_example_count
    ):
        raise ValueError("fit example count differs from preregistration")
    if (
        expected_fit_family_count is not None
        and len(observed_fit_families) != expected_fit_family_count
    ):
        raise ValueError("fit family count differs from preregistration")
    model_metadata = _mapping(
        legacy_metadata["model"],
        label="legacy model",
    )
    graph_binding = _mapping(
        legacy_metadata["graph_candidate"],
        label="legacy graph candidate",
    )
    basis_binding = _mapping(
        legacy_metadata["prompt_blind_basis"],
        label="legacy basis",
    )
    code_before = _source_code_sha256s()
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype=dtype,
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != str(
        model_metadata["source_model_sha256"]
    ):
        raise ValueError("live raw Gemma differs from the frozen source")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256 = adapter.model_fingerprint()
        factorized_execution_sha256 = adapter.execution_fingerprint()
        if (
            factorized_model_sha256
            != str(graph_binding["factorized_live_execution_sha256"])
            or factorized_execution_sha256
            != str(graph_binding["factorized_refit_execution_sha256"])
        ):
            raise ValueError(
                "live factorized Gemma differs from frozen execution"
            )
        graph_path = Path(graph_candidate_path)
        basis_path = Path(basis_package_path)
        graph_candidate = load_gemma3_graph_organized_svd_candidate(
            graph_path,
            expected_file_sha256=str(
                graph_binding["tensor_file_sha256"]
            ),
        )
        basis = load_gemma3_l3_l4_basis_package(
            basis_path,
            expected_file_sha256=str(
                basis_binding["tensor_file_sha256"]
            ),
            expected_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
        )
        runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            graph_candidate,
            basis,
            expected_candidate_artifact_sha256=str(
                graph_binding["logical_artifact_sha256"]
            ),
            expected_basis_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
            expected_plan_artifact_sha256=str(
                graph_binding["deployment_plan_sha256"]
            ),
            expected_live_model_sha256=str(
                graph_binding["factorized_live_execution_sha256"]
            ),
            expected_adapter_execution_sha256=str(
                graph_binding["factorized_refit_execution_sha256"]
            ),
            analysis_device="cpu",
        )
        source_probe = LegacyRank64GemmaProgressiveExecutable(
            adapter=adapter,
            runtime=runtime,
            candidate_execution_sha256=factorized_execution_sha256,
        )
        bridge = runtime.export_one_pass_bridge()
        executable = GemmaL3L4TwoHeadExecutable(
            adapter=adapter,
            shadow_runtime=runtime,
            bridge=bridge,
            source_probe=source_probe,
            artifact=accepted_artifact,
        )
        sequences: list[GemmaTwoHeadFitSequence] = []
        for example in fit_panel.examples:
            observation = executable.observe(
                example,
                collect_carrier_fisher=True,
            )
            sequence = observation.two_head_fit_sequence
            if sequence is None:
                raise RuntimeError("accepted X4 omitted its private fit trace")
            sequences.append(sequence.detached_copy())
        output_decoder, decoder_audit = (
            derive_candidate_h4_output_decoder(
                sequences,
                output_rank=output_rank,
            )
        )
        if damping_mode:
            diagnostic = analyze_gemma_h4_damping(
                sequences=sequences,
                output_decoder=output_decoder,
                lag_count=_DAMPING_LAG_COUNT,
                input_rank=_DAMPING_INPUT_RANK,
                damping_alphas=_DEFAULT_DAMPING_ALPHAS,
                ridge=ridge,
                hypothesis_cell=_damping_hypothesis_cell,
            )
            assert _damping_hypothesis_cell is not None
            assert _damping_hypothesis_provenance is not None
            reproduction = _assert_alpha_one_reproduces_hypothesis(
                diagnostic,
                _damping_hypothesis_cell,
            )
            diagnostic_input = _mapping(
                diagnostic["input"],
                label="damping diagnostic input",
            )
            live_binding = {
                "corpus_artifact_sha256": corpus.artifact_sha256,
                "fit_manifest_sha256": fit_panel.manifest_sha256,
                "fit_binding_sha256": fit_panel.binding_sha256,
                "fit_sequence_sha256s": tuple(
                    diagnostic_input["fit_sequence_sha256s"]  # type: ignore[arg-type]
                ),
                "family_ids": tuple(
                    diagnostic_input["family_ids"]  # type: ignore[arg-type]
                ),
                "affected_row_count": diagnostic_input[
                    "affected_row_count"
                ],
                "output_decoder_sha256": diagnostic_input[
                    "output_decoder_sha256"
                ],
                "accepted_x4_provenance_sha256": _sha256(
                    _DAMPING_HYPOTHESIS_DOMAIN,
                    accepted_provenance,
                ),
            }
            expected_binding = {
                "corpus_artifact_sha256": (
                    _damping_hypothesis_provenance[
                        "source_corpus_artifact_sha256"
                    ]
                ),
                "fit_manifest_sha256": (
                    _damping_hypothesis_provenance[
                        "source_fit_manifest_sha256"
                    ]
                ),
                "fit_binding_sha256": (
                    _damping_hypothesis_provenance[
                        "source_fit_binding_sha256"
                    ]
                ),
                "fit_sequence_sha256s": tuple(
                    _damping_hypothesis_provenance[
                        "source_fit_sequence_sha256s"
                    ]  # type: ignore[arg-type]
                ),
                "family_ids": tuple(
                    _damping_hypothesis_provenance[
                        "source_family_ids"
                    ]  # type: ignore[arg-type]
                ),
                "affected_row_count": (
                    _damping_hypothesis_provenance[
                        "source_affected_row_count"
                    ]
                ),
                "output_decoder_sha256": (
                    _damping_hypothesis_provenance[
                        "source_output_decoder_sha256"
                    ]
                ),
                "accepted_x4_provenance_sha256": (
                    _damping_hypothesis_provenance[
                        "source_accepted_x4_provenance_sha256"
                    ]
                ),
            }
            if _canonical_json_bytes(live_binding) != _canonical_json_bytes(
                expected_binding
            ):
                raise RuntimeError(
                    "damping trace recollection differs from hypothesis"
                )
            trace_recollection: dict[str, object] | None = {
                "collection_kind": "live_model_recollection",
                **live_binding,
                "source_report_trace_payload_reused": False,
                "source_report_coefficient_tensors_reused": False,
                "alpha_one_reproduction": reproduction,
            }
        else:
            diagnostic = analyze_gemma_h4_incremental_signal(
                sequences=sequences,
                output_decoder=output_decoder,
                lag_counts=lags,
                input_ranks=ranks,
                ridge=ridge,
            )
            reproduction = None
            trace_recollection = None
        code_after = _source_code_sha256s()
        if code_after != code_before:
            raise RuntimeError("diagnostic source code changed during run")
        if (
            adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
            or _file_sha256(graph_path)
            != str(graph_binding["tensor_file_sha256"])
            or _file_sha256(basis_path)
            != str(basis_binding["tensor_file_sha256"])
        ):
            raise RuntimeError(
                "diagnostic model or frozen artifacts changed during run"
            )
        report: dict[str, object] = {
            "schema": report_schema,
            "format_version": report_format_version,
            "spec": {
                "analysis_mode": _analysis_mode,
                "corpus_artifact_sha256": corpus.artifact_sha256,
                "fit_manifest_sha256": fit_panel.manifest_sha256,
                "fit_binding_sha256": fit_panel.binding_sha256,
                "maximum_sequence_length": max_length,
                "device": device_name,
                "dtype": dtype,
                "output_rank": output_rank,
                "lag_counts": lags,
                "input_ranks": ranks,
                "ridge": ridge,
                "expected_fit_example_count": expected_fit_example_count,
                "expected_fit_family_count": expected_fit_family_count,
                "observed_fit_example_count": len(fit_panel.examples),
                "observed_fit_family_count": len(observed_fit_families),
                "selection_input_accepted": False,
                "guard_input_accepted": False,
            },
            "accepted_x4_provenance": accepted_provenance,
            "fit_corpus_lineage": fit_corpus_lineage,
            "output_decoder": decoder_audit,
            "diagnostic": diagnostic,
            "bindings": {
                "raw_model_sha256": str(
                    model_metadata["source_model_sha256"]
                ),
                "factorized_model_sha256": factorized_model_sha256,
                "factorized_execution_sha256": (
                    factorized_execution_sha256
                ),
                "progressive_runtime_binding_sha256": (
                    runtime.runtime_binding_sha256
                ),
                "graph_candidate_file_sha256": _file_sha256(graph_path),
                "basis_file_sha256": _file_sha256(basis_path),
                "base_artifact_file_sha256": _file_sha256(
                    base_artifact_path
                ),
                "refit_artifact_file_sha256": _file_sha256(
                    refit_artifact_path
                ),
                "source_code_sha256s": code_before,
            },
            "safety": {
                "development_only": True,
                "fit_role_opened": True,
                "selection_input_capability_present": False,
                "selection_role_opened": False,
                "guard_input_capability_present": False,
                "guard_role_opened": False,
                "calibration_b_loader_present": False,
                "calibration_b_opened": False,
                "prompt_text_in_report": False,
                "token_ids_in_report": False,
                "activation_rows_in_report": False,
                "gradient_rows_in_report": False,
                "fit_sequences_in_report": False,
                "coefficient_tensors_in_report": False,
                "model_weights_in_report": False,
                "compression_claim": False,
                "latency_claim": False,
            },
        }
        if damping_mode:
            report_spec = _mapping(report["spec"], label="report spec")
            report["spec"] = {
                **dict(report_spec),
                "fixed_head": {
                    "lag_count": _DAMPING_LAG_COUNT,
                    "encoder_kind": "independent_crossfit_h4_svd",
                    "input_rank": _DAMPING_INPUT_RANK,
                    "output_rank": _DAMPING_OUTPUT_RANK,
                },
                "damping_alphas": _DEFAULT_DAMPING_ALPHAS,
                "baseline_alpha": 0.0,
                "selection_rule": "smallest_alpha_passing_all_gates",
            }
            report["hypothesis_provenance"] = dict(
                _damping_hypothesis_provenance
            )
            report["trace_recollection"] = trace_recollection
        report["report_sha256"] = _sha256(report_domain, report)
        _publish_report(destination, report)
        return report
    finally:
        switcher.close()


def run_gemma3_l3_l4_h4_damping_diagnostic(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    hypothesis_report_path: Path | str = (
        _DEFAULT_DAMPING_HYPOTHESIS_REPORT
    ),
    expected_hypothesis_report_sha256: str,
    expected_hypothesis_report_file_sha256: str,
    accepted_x4_report_path: Path | str = _DEFAULT_ACCEPTED_REPORT,
    accepted_x4_candidate_path: Path | str = _DEFAULT_ACCEPTED_CANDIDATE,
    expected_accepted_x4_candidate_file_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_H4_DAMPING_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    ridge: float = 1.0e-6,
) -> dict[str, object]:
    """Run the single preregistered L16/r32 fixed-head damping rung."""

    hypothesis_cell, hypothesis_provenance = _load_damping_hypothesis(
        hypothesis_report_path,
        expected_report_sha256=expected_hypothesis_report_sha256,
        expected_report_file_sha256=(
            expected_hypothesis_report_file_sha256
        ),
    )
    return run_gemma3_l3_l4_h4_incremental_signal_diagnostic(
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
        accepted_x4_report_path=accepted_x4_report_path,
        accepted_x4_candidate_path=accepted_x4_candidate_path,
        expected_accepted_x4_candidate_file_sha256=(
            expected_accepted_x4_candidate_file_sha256
        ),
        graph_candidate_path=graph_candidate_path,
        basis_package_path=basis_package_path,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        output=output,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
        output_rank=_DAMPING_OUTPUT_RANK,
        lag_counts=(_DAMPING_LAG_COUNT,),
        input_ranks=(_DAMPING_INPUT_RANK,),
        ridge=ridge,
        expected_fit_example_count=_DAMPING_FIT_EXAMPLE_COUNT,
        expected_fit_family_count=_DAMPING_FIT_FAMILY_COUNT,
        _analysis_mode="fixed_head_damping",
        _damping_hypothesis_cell=hypothesis_cell,
        _damping_hypothesis_provenance=hypothesis_provenance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fit-only family-cross-fitted realized-H4 signal "
            "diagnostic; selection and guard inputs are intentionally absent."
        )
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=_DEFAULT_CORPUS,
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    )
    parser.add_argument(
        "--accepted-x4-report",
        type=Path,
        default=_DEFAULT_ACCEPTED_REPORT,
    )
    parser.add_argument(
        "--accepted-x4-candidate",
        type=Path,
        default=_DEFAULT_ACCEPTED_CANDIDATE,
    )
    parser.add_argument(
        "--accepted-x4-candidate-sha256",
        required=True,
    )
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_H4_INCREMENTAL_SIGNAL_OUTPUT,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-rank", type=int, default=8)
    parser.add_argument(
        "--lag-counts",
        type=int,
        nargs="+",
        default=_DEFAULT_LAG_COUNTS,
    )
    parser.add_argument(
        "--input-ranks",
        type=int,
        nargs="+",
        default=_DEFAULT_INPUT_RANKS,
    )
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--expected-fit-example-count", type=int)
    parser.add_argument("--expected-fit-family-count", type=int)
    return parser


def build_damping_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the hash-locked fit-only L16/r32 fixed-head damping rung; "
            "structural search, selection, and guard inputs are absent."
        )
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=_DEFAULT_EXPANDED_CORPUS,
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=_DEFAULT_EXPANDED_FIT_INPUT,
    )
    parser.add_argument(
        "--hypothesis-report",
        type=Path,
        default=_DEFAULT_DAMPING_HYPOTHESIS_REPORT,
    )
    parser.add_argument("--hypothesis-report-sha256", required=True)
    parser.add_argument(
        "--hypothesis-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--accepted-x4-report",
        type=Path,
        default=_DEFAULT_ACCEPTED_REPORT,
    )
    parser.add_argument(
        "--accepted-x4-candidate",
        type=Path,
        default=_DEFAULT_ACCEPTED_CANDIDATE,
    )
    parser.add_argument(
        "--accepted-x4-candidate-sha256",
        required=True,
    )
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_H4_DAMPING_OUTPUT,
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_h4_incremental_signal_diagnostic(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        accepted_x4_report_path=args.accepted_x4_report,
        accepted_x4_candidate_path=args.accepted_x4_candidate,
        expected_accepted_x4_candidate_file_sha256=(
            args.accepted_x4_candidate_sha256
        ),
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        output_rank=args.output_rank,
        lag_counts=args.lag_counts,
        input_ranks=args.input_ranks,
        ridge=args.ridge,
        expected_fit_example_count=args.expected_fit_example_count,
        expected_fit_family_count=args.expected_fit_family_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def damping_main(argv: Sequence[str] | None = None) -> int:
    args = build_damping_parser().parse_args(argv)
    report = run_gemma3_l3_l4_h4_damping_diagnostic(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        hypothesis_report_path=args.hypothesis_report,
        expected_hypothesis_report_sha256=(
            args.hypothesis_report_sha256
        ),
        expected_hypothesis_report_file_sha256=(
            args.hypothesis_report_file_sha256
        ),
        accepted_x4_report_path=args.accepted_x4_report,
        accepted_x4_candidate_path=args.accepted_x4_candidate,
        expected_accepted_x4_candidate_file_sha256=(
            args.accepted_x4_candidate_sha256
        ),
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
        device_name="cpu",
        dtype="float32",
        ridge=1.0e-6,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
