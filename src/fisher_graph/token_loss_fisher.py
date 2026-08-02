"""Exact token-loss Fisher sufficient statistics and family-held-out fits.

This module operates *after* a model-specific collector has produced the
token-loss directional score matrix

``Q[t, k] = d loss_t / d edge_coefficient_k``.

For one prompt with ``N`` supervised tokens and compensation target ``z``, it
retains only the scalar sufficient statistics

``A = Q.T @ Q / N``, ``b = Q.T @ z / N``, ``c = z.T @ z / N``, and
``mu = Q.sum(0) / N``.

``A`` is the empirical Fisher pulled back into the declared edge-coordinate
basis.  It is not inferred from activation-position rows of one summed-loss
VJP.  Its off-diagonal entries are symmetric co-sensitivity evidence; edge
direction must already be declared by each coordinate's causal tangent.
Model inputs, token IDs, activations, gradients, ``Q``, and ``z`` are not
retained.

Family-held-out fits are ridge-free standardized minimum-norm least-squares
solutions.  Weighting is hierarchical: families receive equal mass, prompts
within a family receive equal mass, and the prompt statistics already give
each supervised token equal mass within its prompt.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES",
    "CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES",
    "EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES",
    "TOKEN_LOSS_FISHER_DEFAULT_GATE_CONFIG",
    "TOKEN_LOSS_FISHER_SCHEMA",
    "TokenLossFisherFoldFit",
    "TokenLossFisherGateConfig",
    "TokenLossFisherLOFOReport",
    "TokenLossFisherPromptRecord",
    "analyze_cumulative_occupancy_token_loss_fisher_lofo",
    "analyze_ew_occupancy_token_loss_fisher_lofo",
    "analyze_token_loss_fisher_lofo",
    "build_token_loss_fisher_prompt_record",
    "fit_token_loss_fisher_fold",
    "token_loss_fisher_prompt_record_from_dict",
]


TOKEN_LOSS_FISHER_SCHEMA = "fisher_graph.token_loss_fisher.v1"

COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
    "ew_occupancy_contrast_real",
    "ew_occupancy_contrast_imag",
)
CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES = (0, 1, 2, 3, 4, 5)
EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES = (0, 1, 2, 3, 6, 7)

_PROMPT_DOMAIN = b"fisher-graph:token-loss-fisher-prompt:v1\0"
_FOLD_DOMAIN = b"fisher-graph:token-loss-fisher-fold:v1\0"
_REPORT_DOMAIN = b"fisher-graph:token-loss-fisher-lofo:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:token-loss-fisher-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SUPPORT_EPSILON = 1.0e-12
_RANK_ABSOLUTE_TOLERANCE = 1.0e-12
_RANK_RELATIVE_TOLERANCE = 1.0e-10
_PSD_RELATIVE_TOLERANCE = 1.0e-10


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _coordinate_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("coordinate_names must be a nonempty sequence")
    result = tuple(
        _identifier(name, label=f"coordinate_names[{index}]")
        for index, name in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError("coordinate_names must be unique")
    return result


def _float_tuple(
    value: object,
    *,
    count: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} scalars")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _float_matrix(
    value: object,
    *,
    width: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != width:
        raise ValueError(f"{label} must be a square {width}-matrix")
    return tuple(
        _float_tuple(row, count=width, label=f"{label}[{index}]")
        for index, row in enumerate(value)
    )


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise TypeError("token Fisher hash inputs must be materialized tensors")
    canonical = (
        value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(_canonical_json_bytes(tuple(int(x) for x in canonical.shape)))
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _symmetric_eigenvalues(value: Tensor) -> Tensor:
    return torch.linalg.eigvalsh((value + value.T) * 0.5)


def _psd_tolerance(value: Tensor) -> float:
    scale = max(float(value.abs().max()), 1.0)
    return _PSD_RELATIVE_TOLERANCE * scale


def _validate_moments(
    fisher: Tensor,
    cross: Tensor,
    target_second_moment: float,
    mean_score: Tensor,
) -> None:
    width = int(cross.numel())
    if (
        fisher.shape != (width, width)
        or mean_score.shape != (width,)
        or not bool(torch.isfinite(fisher).all())
        or not bool(torch.isfinite(cross).all())
        or not bool(torch.isfinite(mean_score).all())
        or not math.isfinite(target_second_moment)
        or target_second_moment < 0.0
    ):
        raise ValueError("token Fisher sufficient statistics are invalid")
    tolerance = _psd_tolerance(fisher)
    if float((fisher - fisher.T).abs().max()) > tolerance:
        raise ValueError("token Fisher matrix must be symmetric")
    if float(_symmetric_eigenvalues(fisher).min()) < -tolerance:
        raise ValueError("token Fisher matrix must be positive semidefinite")

    covariance = fisher - torch.outer(mean_score, mean_score)
    if float(_symmetric_eigenvalues(covariance).min()) < -_psd_tolerance(
        covariance
    ):
        raise ValueError("token score mean is incompatible with its Fisher")

    joint = torch.empty((width + 1, width + 1), dtype=torch.float64)
    joint[:width, :width] = fisher
    joint[:width, width] = cross
    joint[width, :width] = cross
    joint[width, width] = target_second_moment
    if float(_symmetric_eigenvalues(joint).min()) < -_psd_tolerance(joint):
        raise ValueError("token target cross moment is incompatible")


@dataclass(frozen=True, slots=True)
class TokenLossFisherPromptRecord:
    """Prompt-local scalar moments; raw token rows are deliberately absent."""

    example_id: str
    family_id: str
    coordinate_names: tuple[str, ...]
    supervised_tokens: int
    token_score_matrix_sha256: str
    compensation_target_sha256: str
    fisher_second_moment: tuple[tuple[float, ...], ...]
    target_cross_moment: tuple[float, ...]
    target_second_moment: float
    mean_score: tuple[float, ...]
    prompt_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="example_id")
        _identifier(self.family_id, label="family_id")
        names = _coordinate_names(self.coordinate_names)
        object.__setattr__(self, "coordinate_names", names)
        if type(self.supervised_tokens) is not int or self.supervised_tokens <= 0:
            raise ValueError("supervised_tokens must be a positive integer")
        _require_sha256(
            self.token_score_matrix_sha256,
            label="token score matrix",
        )
        _require_sha256(
            self.compensation_target_sha256,
            label="compensation target",
        )
        width = len(names)
        fisher = _float_matrix(
            self.fisher_second_moment,
            width=width,
            label="fisher_second_moment",
        )
        cross = _float_tuple(
            self.target_cross_moment,
            count=width,
            label="target_cross_moment",
        )
        mean = _float_tuple(
            self.mean_score,
            count=width,
            label="mean_score",
        )
        target_second = _finite(
            self.target_second_moment,
            label="target_second_moment",
        )
        _validate_moments(
            torch.tensor(fisher, dtype=torch.float64),
            torch.tensor(cross, dtype=torch.float64),
            target_second,
            torch.tensor(mean, dtype=torch.float64),
        )
        object.__setattr__(self, "fisher_second_moment", fisher)
        object.__setattr__(self, "target_cross_moment", cross)
        object.__setattr__(self, "target_second_moment", target_second)
        object.__setattr__(self, "mean_score", mean)
        object.__setattr__(
            self,
            "prompt_record_sha256",
            _sha256(_PROMPT_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "prompt_record_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "prompt_record_sha256": self.prompt_record_sha256,
        }

    def validate_integrity(self) -> None:
        if (
            _sha256(_PROMPT_DOMAIN, self._payload())
            != self.prompt_record_sha256
        ):
            raise RuntimeError("token Fisher prompt record drifted")


def build_token_loss_fisher_prompt_record(
    *,
    example_id: str,
    family_id: str,
    coordinate_names: Sequence[str],
    token_scores: Tensor,
    compensation_target: Tensor,
) -> TokenLossFisherPromptRecord:
    """Reduce exact token directional scores to prompt-local moments."""

    names = _coordinate_names(tuple(coordinate_names))
    if (
        not isinstance(token_scores, Tensor)
        or token_scores.ndim != 2
        or token_scores.shape[0] <= 0
        or token_scores.shape[1] != len(names)
        or not token_scores.is_floating_point()
        or not isinstance(compensation_target, Tensor)
        or compensation_target.ndim != 1
        or compensation_target.shape[0] != token_scores.shape[0]
        or not compensation_target.is_floating_point()
    ):
        raise ValueError("token score matrix and target geometry differ")
    scores = (
        token_scores.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )
    target = (
        compensation_target.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    if (
        not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("token score matrix and target must be finite")
    count = int(scores.shape[0])
    fisher = (scores.T @ scores) / count
    fisher = ((fisher + fisher.T) * 0.5).contiguous()
    cross = scores.T @ target / count
    target_second = float(target.square().sum() / count)
    mean = scores.sum(dim=0) / count
    return TokenLossFisherPromptRecord(
        example_id=_identifier(example_id, label="example_id"),
        family_id=_identifier(family_id, label="family_id"),
        coordinate_names=names,
        supervised_tokens=count,
        token_score_matrix_sha256=_tensor_sha256(scores),
        compensation_target_sha256=_tensor_sha256(target),
        fisher_second_moment=tuple(
            tuple(float(item) for item in row) for row in fisher
        ),
        target_cross_moment=tuple(float(item) for item in cross),
        target_second_moment=target_second,
        mean_score=tuple(float(item) for item in mean),
    )


def token_loss_fisher_prompt_record_from_dict(
    value: Mapping[str, object],
) -> TokenLossFisherPromptRecord:
    expected = set(TokenLossFisherPromptRecord.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("serialized token Fisher prompt fields differ")
    payload = dict(value)
    receipt = payload.pop("prompt_record_sha256")
    result = TokenLossFisherPromptRecord(**payload)  # type: ignore[arg-type]
    if result.prompt_record_sha256 != receipt:
        raise ValueError("token Fisher prompt-record hash mismatch")
    return result


def _prompt_record(value: object) -> TokenLossFisherPromptRecord:
    if isinstance(value, TokenLossFisherPromptRecord):
        value.validate_integrity()
        return value
    if not isinstance(value, Mapping):
        raise TypeError("token Fisher records must be prompt records or mappings")
    return token_loss_fisher_prompt_record_from_dict(value)


@dataclass(frozen=True, slots=True)
class _Moments:
    fisher: Tensor
    cross: Tensor
    target_second: float


def _mean_moments(values: Sequence[_Moments]) -> _Moments:
    if not values:
        raise ValueError("cannot average empty token Fisher moments")
    fisher = sum((value.fisher for value in values), torch.zeros_like(values[0].fisher))
    cross = sum((value.cross for value in values), torch.zeros_like(values[0].cross))
    count = len(values)
    result = _Moments(
        fisher=((fisher / count + (fisher / count).T) * 0.5).contiguous(),
        cross=(cross / count).contiguous(),
        target_second=math.fsum(value.target_second for value in values)
        / count,
    )
    _validate_moments(
        result.fisher,
        result.cross,
        result.target_second,
        torch.zeros_like(result.cross),
    )
    return result


def _record_moments(
    record: TokenLossFisherPromptRecord,
    indices: tuple[int, ...],
) -> _Moments:
    fisher = torch.tensor(record.fisher_second_moment, dtype=torch.float64)
    cross = torch.tensor(record.target_cross_moment, dtype=torch.float64)
    selected = torch.tensor(indices, dtype=torch.int64)
    return _Moments(
        fisher=fisher.index_select(0, selected)
        .index_select(1, selected)
        .contiguous(),
        cross=cross.index_select(0, selected).contiguous(),
        target_second=record.target_second_moment,
    )


def _family_moments(
    records: Sequence[TokenLossFisherPromptRecord],
    indices: tuple[int, ...],
) -> dict[str, _Moments]:
    grouped: dict[str, list[_Moments]] = defaultdict(list)
    for record in records:
        grouped[record.family_id].append(_record_moments(record, indices))
    return {
        family: _mean_moments(grouped[family])
        for family in sorted(grouped)
    }


def _canonical_records(
    records: Sequence[object],
) -> tuple[TokenLossFisherPromptRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("token Fisher records must be a sequence")
    selected = tuple(
        sorted(
            (_prompt_record(value) for value in records),
            key=lambda row: (row.family_id, row.example_id),
        )
    )
    if not selected:
        raise ValueError("token Fisher records must be nonempty")
    if len({row.example_id for row in selected}) != len(selected):
        raise ValueError("token Fisher example IDs must be globally unique")
    if len({row.coordinate_names for row in selected}) != 1:
        raise ValueError("token Fisher coordinate systems differ")
    return selected


def _coordinate_view(
    coordinate_count: int,
    coordinate_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if coordinate_indices is None:
        return tuple(range(coordinate_count))
    if (
        isinstance(coordinate_indices, (str, bytes))
        or not isinstance(coordinate_indices, Sequence)
        or not coordinate_indices
    ):
        raise ValueError("coordinate_indices must be a nonempty sequence")
    result = tuple(coordinate_indices)
    if (
        any(
            type(index) is not int or not 0 <= index < coordinate_count
            for index in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError("coordinate_indices contain invalid or duplicate values")
    return result


def _rank_and_condition(value: Tensor) -> tuple[int, float, Tensor, float]:
    eigenvalues, eigenvectors = torch.linalg.eigh((value + value.T) * 0.5)
    maximum = max(float(eigenvalues.max()), 0.0) if eigenvalues.numel() else 0.0
    tolerance = max(
        _RANK_ABSOLUTE_TOLERANCE,
        _RANK_RELATIVE_TOLERANCE * maximum,
    )
    positive = eigenvalues > tolerance
    rank = int(positive.sum())
    condition = (
        0.0
        if rank == 0
        else float(eigenvalues[positive].max() / eigenvalues[positive].min())
    )
    return rank, condition, eigenvectors[:, positive], tolerance


def _standardized_minimum_norm_fit(
    moments: _Moments,
) -> tuple[Tensor, tuple[float, ...], int, int, float]:
    fisher = moments.fisher
    width = int(fisher.shape[0])
    raw_rank, _raw_condition, _vectors, _tolerance = _rank_and_condition(
        fisher
    )
    scales = torch.sqrt(torch.clamp(torch.diag(fisher), min=0.0))
    supported = tuple(
        index
        for index, scale in enumerate(scales)
        if float(scale) > _SUPPORT_EPSILON
    )
    coefficients = torch.zeros(width, dtype=torch.float64)
    standardized_rank = 0
    condition = 0.0
    if supported:
        selected = torch.tensor(supported, dtype=torch.int64)
        chosen_scales = scales.index_select(0, selected)
        chosen_fisher = fisher.index_select(0, selected).index_select(
            1, selected
        )
        standardized = chosen_fisher / torch.outer(
            chosen_scales, chosen_scales
        )
        standardized = ((standardized + standardized.T) * 0.5).contiguous()
        eigenvalues, eigenvectors = torch.linalg.eigh(standardized)
        maximum = max(float(eigenvalues.max()), 0.0)
        tolerance = max(
            _RANK_ABSOLUTE_TOLERANCE,
            _RANK_RELATIVE_TOLERANCE * maximum,
        )
        positive = eigenvalues > tolerance
        standardized_rank = int(positive.sum())
        if standardized_rank:
            selected_values = eigenvalues[positive]
            selected_vectors = eigenvectors[:, positive]
            condition = float(
                selected_values.max() / selected_values.min()
            )
            standardized_cross = moments.cross.index_select(
                0, selected
            ) / chosen_scales
            standardized_coefficients = selected_vectors @ (
                (selected_vectors.T @ standardized_cross) / selected_values
            )
            coefficients[selected] = standardized_coefficients / chosen_scales
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("token Fisher least-squares fit became nonfinite")
    return (
        coefficients,
        tuple(float(value) for value in scales),
        raw_rank,
        standardized_rank,
        condition,
    )


def _residual_rmse(moments: _Moments, coefficients: Tensor) -> float:
    squared = float(
        moments.target_second
        - 2.0 * torch.dot(coefficients, moments.cross)
        + torch.dot(coefficients, moments.fisher @ coefficients)
    )
    tolerance = 1.0e-10 * max(moments.target_second, 1.0)
    if squared < -tolerance:
        raise RuntimeError("token Fisher residual moment became negative")
    return math.sqrt(max(squared, 0.0))


def _relative_improvement(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0 if after == 0.0 else -1.0
    return 1.0 - after / before


def _incremental_energy_fractions(
    fisher: Tensor,
    base_coordinate_count: int | None,
) -> tuple[float, ...]:
    width = int(fisher.shape[0])
    if base_coordinate_count is None:
        return ()
    if (
        type(base_coordinate_count) is not int
        or not 0 < base_coordinate_count < width
    ):
        raise ValueError(
            "base_coordinate_count must split the selected coordinate view"
        )
    base = fisher[:base_coordinate_count, :base_coordinate_count]
    cross = fisher[:base_coordinate_count, base_coordinate_count:]
    added = fisher[base_coordinate_count:, base_coordinate_count:]
    projection = torch.linalg.pinv(
        (base + base.T) * 0.5,
        rtol=_RANK_RELATIVE_TOLERANCE,
        atol=_RANK_ABSOLUTE_TOLERANCE,
        hermitian=True,
    )
    schur = added - cross.T @ projection @ cross
    schur = ((schur + schur.T) * 0.5).contiguous()
    tolerance = _psd_tolerance(added)
    if float(_symmetric_eigenvalues(schur).min()) < -tolerance:
        raise RuntimeError("incremental token Fisher Schur complement is not PSD")
    result = []
    for residual, total in zip(torch.diag(schur), torch.diag(added), strict=True):
        denominator = float(total)
        value = (
            0.0
            if denominator <= _SUPPORT_EPSILON
            else float(residual) / denominator
        )
        result.append(min(max(value, 0.0), 1.0))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TokenLossFisherFoldFit:
    """One ridge-free whole-family-held-out fit and exact held score."""

    held_family_id: str
    coordinate_indices: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    base_coordinate_count: int | None
    train_family_ids: tuple[str, ...]
    train_example_ids: tuple[str, ...]
    held_example_ids: tuple[str, ...]
    train_prompt_record_sha256s: tuple[str, ...]
    held_prompt_record_sha256s: tuple[str, ...]
    coefficients: tuple[float, ...]
    column_scales: tuple[float, ...]
    raw_normal_rank: int
    standardized_normal_rank: int
    standardized_positive_spectrum_condition_number: float
    incremental_energy_fraction_by_added_coordinate: tuple[float, ...]
    train_rmse_before: float
    train_rmse_after: float
    held_rmse_before: float
    held_rmse_after: float
    held_relative_rmse_improvement: float
    fold_fit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.held_family_id, label="held_family_id")
        names = _coordinate_names(self.coordinate_names)
        width = len(names)
        indices = _coordinate_view(
            max(self.coordinate_indices) + 1,
            self.coordinate_indices,
        )
        if len(indices) != width:
            raise ValueError("fold coordinate names and indices differ")
        object.__setattr__(self, "coordinate_indices", indices)
        object.__setattr__(self, "coordinate_names", names)
        for label, values in (
            ("train_family_ids", self.train_family_ids),
            ("train_example_ids", self.train_example_ids),
            ("held_example_ids", self.held_example_ids),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or values != tuple(sorted(values))
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{label} must be canonical and nonempty")
            for value in values:
                _identifier(value, label=label)
        if self.held_family_id in self.train_family_ids:
            raise ValueError("held family leaked into token Fisher training")
        for label, values, expected_count in (
            (
                "train_prompt_record_sha256s",
                self.train_prompt_record_sha256s,
                len(self.train_example_ids),
            ),
            (
                "held_prompt_record_sha256s",
                self.held_prompt_record_sha256s,
                len(self.held_example_ids),
            ),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != expected_count
                or values != tuple(sorted(values))
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{label} must be canonical")
            for value in values:
                _require_sha256(value, label=label)
        coefficients = _float_tuple(
            self.coefficients, count=width, label="coefficients"
        )
        scales = _float_tuple(
            self.column_scales, count=width, label="column_scales"
        )
        if any(value < 0.0 for value in scales):
            raise ValueError("token Fisher column scales must be nonnegative")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "column_scales", scales)
        for label, value in (
            ("raw_normal_rank", self.raw_normal_rank),
            ("standardized_normal_rank", self.standardized_normal_rank),
        ):
            if type(value) is not int or not 0 <= value <= width:
                raise ValueError(f"{label} is invalid")
        condition = _finite(
            self.standardized_positive_spectrum_condition_number,
            label="standardized condition",
        )
        if condition < 0.0:
            raise ValueError("standardized condition must be nonnegative")
        object.__setattr__(
            self,
            "standardized_positive_spectrum_condition_number",
            condition,
        )
        if self.base_coordinate_count is None:
            if self.incremental_energy_fraction_by_added_coordinate:
                raise ValueError("incremental energy requires a base split")
        else:
            if (
                type(self.base_coordinate_count) is not int
                or not 0 < self.base_coordinate_count < width
            ):
                raise ValueError("fold base coordinate count is invalid")
            energy = _float_tuple(
                self.incremental_energy_fraction_by_added_coordinate,
                count=width - self.base_coordinate_count,
                label="incremental energy",
            )
            if any(not 0.0 <= value <= 1.0 for value in energy):
                raise ValueError("incremental energy must lie in [0, 1]")
            object.__setattr__(
                self,
                "incremental_energy_fraction_by_added_coordinate",
                energy,
            )
        for name in (
            "train_rmse_before",
            "train_rmse_after",
            "held_rmse_before",
            "held_rmse_after",
        ):
            value = _finite(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        improvement = _finite(
            self.held_relative_rmse_improvement,
            label="held relative RMSE improvement",
        )
        object.__setattr__(
            self, "held_relative_rmse_improvement", improvement
        )
        object.__setattr__(
            self,
            "fold_fit_sha256",
            _sha256(_FOLD_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "fold_fit_sha256"
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fold_fit_sha256": self.fold_fit_sha256}

    def validate_integrity(self) -> None:
        if _sha256(_FOLD_DOMAIN, self._payload()) != self.fold_fit_sha256:
            raise RuntimeError("token Fisher fold fit drifted")


def fit_token_loss_fisher_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
    coordinate_indices: Sequence[int] | None = None,
    base_coordinate_count: int | None = None,
) -> TokenLossFisherFoldFit:
    """Fit all non-held families and score the held family from exact moments."""

    held = _identifier(held_family_id, label="held_family_id")
    selected = _canonical_records(records)
    names = selected[0].coordinate_names
    view = _coordinate_view(len(names), coordinate_indices)
    view_names = tuple(names[index] for index in view)
    family_stats = _family_moments(selected, view)
    if held not in family_stats:
        raise ValueError("held family is absent from token Fisher records")
    train_families = tuple(family for family in family_stats if family != held)
    if not train_families:
        raise ValueError("token Fisher LOFO requires at least two families")
    train = _mean_moments(
        tuple(family_stats[family] for family in train_families)
    )
    held_moments = family_stats[held]
    coefficients, scales, raw_rank, standardized_rank, condition = (
        _standardized_minimum_norm_fit(train)
    )
    train_before = _residual_rmse(
        train, torch.zeros_like(coefficients)
    )
    train_after = _residual_rmse(train, coefficients)
    held_before = _residual_rmse(
        held_moments, torch.zeros_like(coefficients)
    )
    held_after = _residual_rmse(held_moments, coefficients)
    train_records = tuple(row for row in selected if row.family_id != held)
    held_records = tuple(row for row in selected if row.family_id == held)
    return TokenLossFisherFoldFit(
        held_family_id=held,
        coordinate_indices=view,
        coordinate_names=view_names,
        base_coordinate_count=base_coordinate_count,
        train_family_ids=train_families,
        train_example_ids=tuple(sorted(row.example_id for row in train_records)),
        held_example_ids=tuple(sorted(row.example_id for row in held_records)),
        train_prompt_record_sha256s=tuple(
            sorted(row.prompt_record_sha256 for row in train_records)
        ),
        held_prompt_record_sha256s=tuple(
            sorted(row.prompt_record_sha256 for row in held_records)
        ),
        coefficients=tuple(float(value) for value in coefficients),
        column_scales=scales,
        raw_normal_rank=raw_rank,
        standardized_normal_rank=standardized_rank,
        standardized_positive_spectrum_condition_number=condition,
        incremental_energy_fraction_by_added_coordinate=(
            _incremental_energy_fractions(
                train.fisher, base_coordinate_count
            )
        ),
        train_rmse_before=train_before,
        train_rmse_after=train_after,
        held_rmse_before=held_before,
        held_rmse_after=held_after,
        held_relative_rmse_improvement=_relative_improvement(
            held_before, held_after
        ),
    )


@dataclass(frozen=True, slots=True)
class TokenLossFisherGateConfig:
    maximum_median_standardized_condition: float = 100.0
    minimum_mean_pairwise_coefficient_cosine: float = 0.90
    minimum_family_macro_relative_rmse_improvement: float = 0.02
    minimum_family_win_fraction: float = 0.75
    maximum_worst_family_relative_rmse_regression: float = 0.02
    minimum_incremental_energy_fraction: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _finite(getattr(self, name), label=name)
            object.__setattr__(self, name, value)
        if (
            self.maximum_median_standardized_condition <= 0.0
            or not 0.0
            <= self.minimum_mean_pairwise_coefficient_cosine
            <= 1.0
            or not -1.0
            <= self.minimum_family_macro_relative_rmse_improvement
            <= 1.0
            or not 0.0 <= self.minimum_family_win_fraction <= 1.0
            or self.maximum_worst_family_relative_rmse_regression < 0.0
            or not 0.0 <= self.minimum_incremental_energy_fraction <= 1.0
        ):
            raise ValueError("token Fisher gate configuration is invalid")

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


TOKEN_LOSS_FISHER_DEFAULT_GATE_CONFIG = TokenLossFisherGateConfig()


def _coefficient_cosine(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    first = torch.tensor(tuple(left), dtype=torch.float64)
    second = torch.tensor(tuple(right), dtype=torch.float64)
    first_norm = float(torch.linalg.vector_norm(first))
    second_norm = float(torch.linalg.vector_norm(second))
    if first_norm == 0.0 and second_norm == 0.0:
        return 1.0
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return float(torch.dot(first, second) / (first_norm * second_norm))


def _median(values: Sequence[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("median requires at least one value")
    center = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[center]
    return (ordered[center - 1] + ordered[center]) / 2.0


@dataclass(frozen=True, slots=True)
class TokenLossFisherLOFOReport:
    """Canonical prompt-free LOFO metrics and preregistered gate outcomes."""

    schema: str
    coordinate_indices: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    base_coordinate_count: int | None
    family_ids: tuple[str, ...]
    prompt_record_sha256s: tuple[str, ...]
    folds: tuple[TokenLossFisherFoldFit, ...]
    gate_config: TokenLossFisherGateConfig
    family_macro_rmse_before: float
    family_macro_rmse_after: float
    family_macro_relative_rmse_improvement: float
    family_win_count: int
    minimum_family_win_count: int
    worst_family_relative_rmse_improvement: float
    median_fold_standardized_condition: float
    mean_pairwise_fold_coefficient_cosine: float
    minimum_fold_incremental_energy_fraction: float | None
    gate_results: tuple[tuple[str, bool], ...]
    passed: bool
    report_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != TOKEN_LOSS_FISHER_SCHEMA:
            raise ValueError("token Fisher report schema differs")
        names = _coordinate_names(self.coordinate_names)
        indices = _coordinate_view(
            max(self.coordinate_indices) + 1,
            self.coordinate_indices,
        )
        if len(names) != len(indices):
            raise ValueError("report coordinate names and indices differ")
        object.__setattr__(self, "coordinate_names", names)
        object.__setattr__(self, "coordinate_indices", indices)
        if (
            not isinstance(self.family_ids, tuple)
            or len(self.family_ids) < 2
            or self.family_ids != tuple(sorted(self.family_ids))
            or len(set(self.family_ids)) != len(self.family_ids)
        ):
            raise ValueError("report family IDs must be canonical")
        for family in self.family_ids:
            _identifier(family, label="report family")
        if (
            not isinstance(self.prompt_record_sha256s, tuple)
            or not self.prompt_record_sha256s
            or self.prompt_record_sha256s
            != tuple(sorted(self.prompt_record_sha256s))
            or len(set(self.prompt_record_sha256s))
            != len(self.prompt_record_sha256s)
        ):
            raise ValueError("report prompt receipts must be canonical")
        for receipt in self.prompt_record_sha256s:
            _require_sha256(receipt, label="report prompt receipt")
        if (
            not isinstance(self.folds, tuple)
            or len(self.folds) != len(self.family_ids)
            or tuple(fold.held_family_id for fold in self.folds)
            != self.family_ids
        ):
            raise ValueError("report folds do not match canonical families")
        for fold in self.folds:
            fold.validate_integrity()
            if (
                fold.coordinate_indices != indices
                or fold.coordinate_names != names
                or fold.base_coordinate_count != self.base_coordinate_count
            ):
                raise ValueError("report fold coordinate systems differ")
        if not isinstance(self.gate_config, TokenLossFisherGateConfig):
            raise TypeError("report gate config has the wrong type")
        for name in (
            "family_macro_rmse_before",
            "family_macro_rmse_after",
            "median_fold_standardized_condition",
        ):
            value = _finite(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in (
            "family_macro_relative_rmse_improvement",
            "worst_family_relative_rmse_improvement",
            "mean_pairwise_fold_coefficient_cosine",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), label=name)
            )
        if not -1.0 <= self.mean_pairwise_fold_coefficient_cosine <= 1.0:
            raise ValueError("report fold coefficient cosine is invalid")
        if self.minimum_fold_incremental_energy_fraction is not None:
            value = _finite(
                self.minimum_fold_incremental_energy_fraction,
                label="minimum fold incremental energy",
            )
            if not 0.0 <= value <= 1.0:
                raise ValueError("minimum fold incremental energy is invalid")
            object.__setattr__(
                self, "minimum_fold_incremental_energy_fraction", value
            )
        elif self.base_coordinate_count is not None:
            raise ValueError("base split report omitted incremental energy")
        for name, value in (
            ("family_win_count", self.family_win_count),
            ("minimum_family_win_count", self.minimum_family_win_count),
        ):
            if type(value) is not int or not 0 <= value <= len(self.family_ids):
                raise ValueError(f"{name} is invalid")
        if (
            not isinstance(self.gate_results, tuple)
            or not self.gate_results
            or tuple(name for name, _value in self.gate_results)
            != tuple(sorted(name for name, _value in self.gate_results))
            or len({name for name, _value in self.gate_results})
            != len(self.gate_results)
            or any(type(value) is not bool for _name, value in self.gate_results)
        ):
            raise ValueError("report gate results must be canonical")
        expected_passed = all(value for _name, value in self.gate_results)
        if type(self.passed) is not bool or self.passed is not expected_passed:
            raise ValueError("report passed flag differs from its gates")
        object.__setattr__(
            self,
            "report_sha256",
            _sha256(_REPORT_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "coordinate_indices": self.coordinate_indices,
            "coordinate_names": self.coordinate_names,
            "base_coordinate_count": self.base_coordinate_count,
            "family_ids": self.family_ids,
            "prompt_record_sha256s": self.prompt_record_sha256s,
            "folds": tuple(fold.to_dict() for fold in self.folds),
            "gate_config": self.gate_config.to_dict(),
            "family_macro_rmse_before": self.family_macro_rmse_before,
            "family_macro_rmse_after": self.family_macro_rmse_after,
            "family_macro_relative_rmse_improvement": (
                self.family_macro_relative_rmse_improvement
            ),
            "family_win_count": self.family_win_count,
            "minimum_family_win_count": self.minimum_family_win_count,
            "worst_family_relative_rmse_improvement": (
                self.worst_family_relative_rmse_improvement
            ),
            "median_fold_standardized_condition": (
                self.median_fold_standardized_condition
            ),
            "mean_pairwise_fold_coefficient_cosine": (
                self.mean_pairwise_fold_coefficient_cosine
            ),
            "minimum_fold_incremental_energy_fraction": (
                self.minimum_fold_incremental_energy_fraction
            ),
            "gate_results": self.gate_results,
            "passed": self.passed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "report_sha256": self.report_sha256}

    def validate_integrity(self) -> None:
        for fold in self.folds:
            fold.validate_integrity()
        if _sha256(_REPORT_DOMAIN, self._payload()) != self.report_sha256:
            raise RuntimeError("token Fisher LOFO report drifted")


def analyze_token_loss_fisher_lofo(
    records: Sequence[object],
    *,
    coordinate_indices: Sequence[int] | None = None,
    base_coordinate_count: int | None = None,
    gate_config: TokenLossFisherGateConfig = (
        TOKEN_LOSS_FISHER_DEFAULT_GATE_CONFIG
    ),
) -> TokenLossFisherLOFOReport:
    """Run deterministic whole-family-held-out token Fisher analysis."""

    if not isinstance(gate_config, TokenLossFisherGateConfig):
        raise TypeError("gate_config must be TokenLossFisherGateConfig")
    selected = _canonical_records(records)
    names = selected[0].coordinate_names
    view = _coordinate_view(len(names), coordinate_indices)
    view_names = tuple(names[index] for index in view)
    families = tuple(sorted({row.family_id for row in selected}))
    if len(families) < 2:
        raise ValueError("token Fisher LOFO requires at least two families")
    folds = tuple(
        fit_token_loss_fisher_fold(
            selected,
            held_family_id=family,
            coordinate_indices=view,
            base_coordinate_count=base_coordinate_count,
        )
        for family in families
    )
    before = math.fsum(fold.held_rmse_before for fold in folds) / len(folds)
    after = math.fsum(fold.held_rmse_after for fold in folds) / len(folds)
    relative = _relative_improvement(before, after)
    family_improvements = tuple(
        fold.held_relative_rmse_improvement for fold in folds
    )
    wins = sum(value > 0.0 for value in family_improvements)
    minimum_wins = math.ceil(
        gate_config.minimum_family_win_fraction * len(families)
    )
    conditions = tuple(
        fold.standardized_positive_spectrum_condition_number for fold in folds
    )
    pairwise = tuple(
        _coefficient_cosine(folds[left].coefficients, folds[right].coefficients)
        for left in range(len(folds))
        for right in range(left + 1, len(folds))
    )
    mean_cosine = (
        1.0 if not pairwise else math.fsum(pairwise) / len(pairwise)
    )
    minimum_incremental: float | None = None
    if base_coordinate_count is not None:
        minimum_incremental = min(
            value
            for fold in folds
            for value in fold.incremental_energy_fraction_by_added_coordinate
        )
    gate_results = {
        "all_fold_raw_normal_ranks_full": all(
            fold.raw_normal_rank == len(view) for fold in folds
        ),
        "all_fold_standardized_normal_ranks_full": all(
            fold.standardized_normal_rank == len(view) for fold in folds
        ),
        "family_macro_relative_rmse_improvement_at_least_minimum": (
            relative
            >= gate_config.minimum_family_macro_relative_rmse_improvement
        ),
        "family_win_count_at_least_minimum": wins >= minimum_wins,
        "mean_pairwise_fold_coefficient_cosine_at_least_minimum": (
            mean_cosine
            >= gate_config.minimum_mean_pairwise_coefficient_cosine
        ),
        "median_fold_standardized_condition_at_most_maximum": (
            _median(conditions)
            <= gate_config.maximum_median_standardized_condition
        ),
        "worst_family_relative_rmse_regression_at_most_maximum": (
            min(family_improvements)
            >= -gate_config.maximum_worst_family_relative_rmse_regression
        ),
    }
    if minimum_incremental is not None:
        gate_results[
            "all_fold_incremental_energy_fractions_at_least_minimum"
        ] = (
            minimum_incremental
            >= gate_config.minimum_incremental_energy_fraction
        )
    canonical_gates = tuple(sorted(gate_results.items()))
    return TokenLossFisherLOFOReport(
        schema=TOKEN_LOSS_FISHER_SCHEMA,
        coordinate_indices=view,
        coordinate_names=view_names,
        base_coordinate_count=base_coordinate_count,
        family_ids=families,
        prompt_record_sha256s=tuple(
            sorted(row.prompt_record_sha256 for row in selected)
        ),
        folds=folds,
        gate_config=gate_config,
        family_macro_rmse_before=before,
        family_macro_rmse_after=after,
        family_macro_relative_rmse_improvement=relative,
        family_win_count=wins,
        minimum_family_win_count=minimum_wins,
        worst_family_relative_rmse_improvement=min(family_improvements),
        median_fold_standardized_condition=_median(conditions),
        mean_pairwise_fold_coefficient_cosine=mean_cosine,
        minimum_fold_incremental_energy_fraction=minimum_incremental,
        gate_results=canonical_gates,
        passed=all(value for _name, value in canonical_gates),
    )


def _require_combined_occupancy_coordinates(
    records: Sequence[object],
) -> tuple[TokenLossFisherPromptRecord, ...]:
    selected = _canonical_records(records)
    if (
        selected[0].coordinate_names
        != COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES
    ):
        raise ValueError("records do not use the combined occupancy coordinates")
    return selected


def analyze_cumulative_occupancy_token_loss_fisher_lofo(
    records: Sequence[object],
    *,
    gate_config: TokenLossFisherGateConfig = (
        TOKEN_LOSS_FISHER_DEFAULT_GATE_CONFIG
    ),
) -> TokenLossFisherLOFOReport:
    return analyze_token_loss_fisher_lofo(
        _require_combined_occupancy_coordinates(records),
        coordinate_indices=(
            CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES
        ),
        base_coordinate_count=4,
        gate_config=gate_config,
    )


def analyze_ew_occupancy_token_loss_fisher_lofo(
    records: Sequence[object],
    *,
    gate_config: TokenLossFisherGateConfig = (
        TOKEN_LOSS_FISHER_DEFAULT_GATE_CONFIG
    ),
) -> TokenLossFisherLOFOReport:
    return analyze_token_loss_fisher_lofo(
        _require_combined_occupancy_coordinates(records),
        coordinate_indices=EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
        base_coordinate_count=4,
        gate_config=gate_config,
    )
