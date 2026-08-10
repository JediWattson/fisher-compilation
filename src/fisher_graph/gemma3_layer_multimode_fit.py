"""Deterministic A-only fitting for Gemma 3 layer-local multimode flows.

This module is the numerical compiler for the first, MLP-only rung.  It has no
API accepting fresh B3/C3/D3 rows: callers must provide authenticated
``A_exposed_development`` rows, and the production policy fixes the exact
fit/validation target lengths.  Atlas construction and optimizer updates read
fit rows only.  Validation rows are opened only after the fitted parameters
have been frozen.

Charts partition state and modes partition chart-local dynamics.  In
particular, mode regularizers are conditional on each chart; they never
balance or harden chart membership ``q``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Final
from weakref import WeakValueDictionary

import torch
from torch import Tensor

from .gemma3_layer_multimode_capture import (
    A_DEVELOPMENT_ROLE,
    A_FIT_TARGET_TOKEN_COUNTS,
    A_VALIDATION_TARGET_TOKEN_COUNTS,
    DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE,
    LayerMLPFlowRows,
    PREDECESSOR_A_ROLE,
    PRODUCTION_FULL_A_CAPTURE_SCOPE,
    family_example_balanced_row_weights,
    fixed_A_development_split_masks,
)
from .gemma3_layer_multimode_flow import (
    AffineLocalLayerMultimodeFlow,
    FixedLayerChartAtlas,
    LayerMultimodeFlowConfig,
    LayerMultimodeFlowResult,
    layer_multimode_flow_loss,
)
from .gemma3_layer_multimode_selection import (
    NATIVE_MLP_MATRIX_MACS_PER_TOKEN,
    NATIVE_MLP_MATRIX_PARAMETER_COUNT,
    PRODUCTION_FULL_A_CAPTURE_SCOPE as SELECTION_PRODUCTION_FULL_A_CAPTURE_SCOPE,
    PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256,
    LayerMultimodeCandidateEvaluationReceipt,
    LayerMultimodeFamilyFidelityReceipt,
    _grant_live_candidate_evaluation_authority,
    candidate_matrix_macs_per_token,
    candidate_serialized_parameter_count,
)


__all__ = [
    "AtlasSupportError",
    "LayerMultimodeAtlasFit",
    "LayerMultimodeAtlasFitReceipt",
    "LayerMultimodeCandidateFit",
    "LayerMultimodeCandidateFitReceipt",
    "LayerMultimodeFitPolicy",
    "LayerMultimodeSplitRefitEvidence",
    "LayerMultimodeValidationEvidence",
    "PRODUCTION_LAYER_MULTIMODE_FIT_POLICY",
    "evaluate_layer_multimode_split_refit",
    "fit_layer_chart_atlas",
    "fit_layer_multimode_candidate",
    "layer_multimode_candidate_evaluation_from_fit",
    "make_layer_multimode_test_policy",
]


_FORMAT_VERSION: Final = 1
_SCHEMA: Final = "fisher_graph.gemma3_layer_multimode_fit.v1"
_ATLAS_INIT_ID: Final = (
    "weighted_mean_plus_first_argmax_weighted_squared_distance.v1"
)
_ATLAS_FIT_ID: Final = "twelve_weighted_lloyd_then_weighted_state_pca.v1"
_OPTIMIZER_ID: Final = (
    "torch_adam_cpu_float64_fixed_minibatch_separate_tau_clip.v2"
)
_GRADIENT_CLIP_GROUPING_ID: Final = (
    "non_tau_global_l2_and_chart_temperature_scalar_independent.v1"
)
_SPLIT_ID: Final = "family_length_checkerboard_fit_only.v1"
_HARD_ROUTE_ID: Final = "argmax_flattened_q_times_pi_label_c_times_m_plus_k.v1"
_ARI_ID: Final = "hubert_arabie_contingency_label_permutation_invariant.v1"
_VALIDATION_WEIGHTING_ID: Final = (
    "family_example_balanced_normalized_row_weights_times_validation_N.v1"
)
_POLICY_DOMAIN = b"fisher_graph:gemma3-layer-multimode-fit-policy:v1\0"
_ATLAS_RECEIPT_DOMAIN = b"fisher_graph:gemma3-layer-multimode-atlas-receipt:v1\0"
_SUBSET_DOMAIN = b"fisher_graph:gemma3-layer-multimode-subset:v1\0"
_EVIDENCE_DOMAIN = b"fisher_graph:gemma3-layer-multimode-evidence:v1\0"
_FIT_RECEIPT_DOMAIN = b"fisher_graph:gemma3-layer-multimode-fit-receipt:v1\0"
_SPLIT_DOMAIN = b"fisher_graph:gemma3-layer-multimode-split-refit:v1\0"
_SELECTION_ADAPTER_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode-selection-adapter:v1\0"
)
_TENSOR_DOMAIN = b"fisher_graph:gemma3-layer-multimode-fit-tensor:v1\0"
_MASK_DOMAIN = b"fisher_graph:gemma3-layer-multimode-fit-mask:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MASS_ATOL = 1.0e-10
_LIVE_ATLAS_FITTER_AUTHORITY = object()
_LIVE_FITTER_AUTHORITY = object()
_LIVE_SPLIT_EVALUATOR_AUTHORITY = object()
_LIVE_ATLAS_FITS: WeakValueDictionary[int, object] = WeakValueDictionary()
_LIVE_CANDIDATE_FITS: WeakValueDictionary[int, object] = WeakValueDictionary()
_LIVE_SPLIT_EVIDENCE: WeakValueDictionary[int, object] = WeakValueDictionary()


class AtlasSupportError(RuntimeError):
    """Raised before training when a requested chart/rank lacks support."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{label} must be an exact integer >= {minimum}")
    return value


def _strict_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be an exact finite float")
    if positive and value <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _strict_fields(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _tuple_of_ints(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    nonempty: bool = True,
) -> tuple[int, ...]:
    if type(value) not in (tuple, list):
        raise TypeError(f"{label} must be a tuple/list")
    result = tuple(
        _strict_int(item, label=f"{label} item", minimum=minimum)
        for item in value
    )
    if nonempty and not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _tuple_of_floats(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    nonempty: bool = True,
) -> tuple[float, ...]:
    if type(value) not in (tuple, list):
        raise TypeError(f"{label} must be a tuple/list")
    result = tuple(
        _strict_float(
            item,
            label=f"{label} item",
            minimum=minimum,
        )
        for item in value
    )
    if nonempty and not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _mask_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.bool
        or value.ndim != 1
        or value.requires_grad
        or not value.is_contiguous()
    ):
        raise ValueError("mask must be detached contiguous CPU bool [N]")
    return _digest(
        _MASK_DOMAIN,
        {
            "row_count": int(value.numel()),
            "selected_count": int(value.sum()),
            "bytes_sha256": hashlib.sha256(
                value.numpy().tobytes(order="C")
            ).hexdigest(),
        },
    )


def _safety_payload() -> dict[str, bool]:
    return {
        "contains_activation_rows": False,
        "contains_gradient_rows": False,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_raw_tensors": False,
        "contains_token_ids": False,
        "uses_A_fit_rows_only_for_updates": True,
        "uses_B3_C3_D3_or_reserve": False,
    }


@dataclass(frozen=True, slots=True)
class LayerMultimodeFitPolicy:
    """Receipt-bound numerical policy; production fields are not overridable."""

    policy_name: str
    fit_target_token_counts: tuple[int, ...]
    validation_target_token_counts: tuple[int, ...]
    lloyd_passes: int
    optimizer_steps: int
    minibatch_size: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip_norm: float
    fidelity_coefficient: float
    load_balance_coefficient: float
    sparsity_coefficient: float
    distinctness_coefficient: float
    mode_temperature: float
    covariance_relative_tolerance: float
    covariance_absolute_floor: float
    ridge_relative_coefficient: float
    chart_temperature_bound_factor: float
    validation_minimum_chart_soft_mass: float
    seed_base: int
    test_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name:
            raise ValueError("policy_name must be nonempty")
        fit_lengths = _tuple_of_ints(
            self.fit_target_token_counts,
            label="fit_target_token_counts",
            minimum=2,
        )
        validation_lengths = _tuple_of_ints(
            self.validation_target_token_counts,
            label="validation_target_token_counts",
            minimum=2,
        )
        if (
            len(set(fit_lengths)) != len(fit_lengths)
            or len(set(validation_lengths)) != len(validation_lengths)
            or set(fit_lengths) & set(validation_lengths)
        ):
            raise ValueError("policy target lengths must be unique and disjoint")
        object.__setattr__(self, "fit_target_token_counts", fit_lengths)
        object.__setattr__(
            self,
            "validation_target_token_counts",
            validation_lengths,
        )
        _strict_int(self.lloyd_passes, label="lloyd_passes", minimum=1)
        _strict_int(self.optimizer_steps, label="optimizer_steps", minimum=1)
        _strict_int(self.minibatch_size, label="minibatch_size", minimum=1)
        for label in (
            "learning_rate",
            "adam_epsilon",
            "gradient_clip_norm",
            "fidelity_coefficient",
            "mode_temperature",
            "covariance_relative_tolerance",
            "covariance_absolute_floor",
            "ridge_relative_coefficient",
            "chart_temperature_bound_factor",
        ):
            _strict_float(
                getattr(self, label),
                label=label,
                positive=True,
            )
        for label in (
            "load_balance_coefficient",
            "sparsity_coefficient",
            "distinctness_coefficient",
        ):
            _strict_float(
                getattr(self, label),
                label=label,
                minimum=0.0,
            )
        _strict_float(
            self.adam_beta1,
            label="adam_beta1",
            minimum=0.0,
            maximum=1.0,
        )
        _strict_float(
            self.adam_beta2,
            label="adam_beta2",
            minimum=0.0,
            maximum=1.0,
        )
        if self.adam_beta1 >= 1.0 or self.adam_beta2 >= 1.0:
            raise ValueError("Adam betas must be < 1")
        _strict_float(
            self.validation_minimum_chart_soft_mass,
            label="validation_minimum_chart_soft_mass",
            minimum=0.0,
            maximum=0.5,
        )
        if self.chart_temperature_bound_factor <= 1.0:
            raise ValueError("chart_temperature_bound_factor must exceed one")
        seed = _strict_int(self.seed_base, label="seed_base")
        if seed >= 2**63:
            raise ValueError("seed_base must fit signed int64")
        if type(self.test_only) is not bool:
            raise TypeError("test_only must be bool")

    def payload(self) -> dict[str, object]:
        return {
            name: (
                list(value) if isinstance(value, tuple) else value
            )
            for name, value in (
                ("adam_beta1", self.adam_beta1),
                ("adam_beta2", self.adam_beta2),
                ("adam_epsilon", self.adam_epsilon),
                (
                    "chart_temperature_bound_factor",
                    self.chart_temperature_bound_factor,
                ),
                (
                    "covariance_absolute_floor",
                    self.covariance_absolute_floor,
                ),
                (
                    "covariance_relative_tolerance",
                    self.covariance_relative_tolerance,
                ),
                ("distinctness_coefficient", self.distinctness_coefficient),
                ("fidelity_coefficient", self.fidelity_coefficient),
                ("fit_target_token_counts", self.fit_target_token_counts),
                ("gradient_clip_norm", self.gradient_clip_norm),
                ("gradient_clip_grouping_id", _GRADIENT_CLIP_GROUPING_ID),
                ("learning_rate", self.learning_rate),
                ("lloyd_passes", self.lloyd_passes),
                ("load_balance_coefficient", self.load_balance_coefficient),
                ("minibatch_size", self.minibatch_size),
                ("mode_temperature", self.mode_temperature),
                ("optimizer_steps", self.optimizer_steps),
                ("optimizer_id", _OPTIMIZER_ID),
                ("policy_name", self.policy_name),
                ("ridge_relative_coefficient", self.ridge_relative_coefficient),
                ("seed_base", self.seed_base),
                ("sparsity_coefficient", self.sparsity_coefficient),
                ("test_only", self.test_only),
                (
                    "validation_minimum_chart_soft_mass",
                    self.validation_minimum_chart_soft_mass,
                ),
                (
                    "validation_sufficient_statistics_weighting_id",
                    _VALIDATION_WEIGHTING_ID,
                ),
                (
                    "validation_target_token_counts",
                    self.validation_target_token_counts,
                ),
            )
        }

    @property
    def policy_sha256(self) -> str:
        return _digest(_POLICY_DOMAIN, self.payload())


PRODUCTION_LAYER_MULTIMODE_FIT_POLICY = LayerMultimodeFitPolicy(
    policy_name="gemma3_layer_multimode_A_only_production_v1",
    fit_target_token_counts=A_FIT_TARGET_TOKEN_COUNTS,
    validation_target_token_counts=A_VALIDATION_TARGET_TOKEN_COUNTS,
    lloyd_passes=12,
    optimizer_steps=256,
    minibatch_size=512,
    learning_rate=0.003,
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_epsilon=1.0e-8,
    gradient_clip_norm=5.0,
    fidelity_coefficient=1.0,
    load_balance_coefficient=0.02,
    sparsity_coefficient=0.001,
    distinctness_coefficient=0.01,
    mode_temperature=1.0,
    covariance_relative_tolerance=1.0e-10,
    covariance_absolute_floor=1.0e-12,
    ridge_relative_coefficient=1.0e-6,
    chart_temperature_bound_factor=16.0,
    validation_minimum_chart_soft_mass=0.05,
    seed_base=2_026_081_000,
)
if (
    PRODUCTION_LAYER_MULTIMODE_FIT_POLICY.policy_sha256
    != PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
):
    raise RuntimeError("production fit-policy digest drifted from selector ABI")
if PRODUCTION_FULL_A_CAPTURE_SCOPE != SELECTION_PRODUCTION_FULL_A_CAPTURE_SCOPE:
    raise RuntimeError("production full-A capture scope drifted from selector ABI")


def make_layer_multimode_test_policy(
    *,
    fit_target_token_counts: tuple[int, ...] = (3, 5),
    validation_target_token_counts: tuple[int, ...] = (4, 6),
    optimizer_steps: int = 8,
    minibatch_size: int = 16,
) -> LayerMultimodeFitPolicy:
    """Return an explicit, non-authorizing small-data policy for unit tests."""

    return LayerMultimodeFitPolicy(
        policy_name="TEST_ONLY_gemma3_layer_multimode_fit_v1",
        fit_target_token_counts=fit_target_token_counts,
        validation_target_token_counts=validation_target_token_counts,
        lloyd_passes=12,
        optimizer_steps=optimizer_steps,
        minibatch_size=minibatch_size,
        learning_rate=0.01,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1.0e-8,
        gradient_clip_norm=5.0,
        fidelity_coefficient=1.0,
        load_balance_coefficient=0.02,
        sparsity_coefficient=0.001,
        distinctness_coefficient=0.01,
        mode_temperature=1.0,
        covariance_relative_tolerance=1.0e-10,
        covariance_absolute_floor=1.0e-12,
        ridge_relative_coefficient=1.0e-6,
        chart_temperature_bound_factor=16.0,
        validation_minimum_chart_soft_mass=0.0,
        seed_base=9_001_337,
        test_only=True,
    )


@dataclass(frozen=True, slots=True)
class LayerMultimodeAtlasFitReceipt:
    layer_ordinal: int
    source_rows_receipt_sha256: str
    split_receipt_sha256: str
    policy_sha256: str
    fit_source_sha256: str
    fit_row_count: int
    hidden_width: int
    chart_count: int
    tangent_rank: int
    hard_chart_counts: tuple[int, ...]
    covariance_numerical_ranks: tuple[int, ...]
    covariance_selected_min_eigenvalues: tuple[float, ...]
    covariance_max_eigenvalues: tuple[float, ...]
    atlas_sha256: str
    initial_chart_temperature: float
    chart_temperature_lower_bound: float
    chart_temperature_upper_bound: float
    initialization_id: str = _ATLAS_INIT_ID
    atlas_fit_id: str = _ATLAS_FIT_ID
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _strict_int(self.layer_ordinal, label="layer_ordinal")
        if self.layer_ordinal >= 18:
            raise ValueError("layer_ordinal must be in [0,17]")
        for label in (
            "source_rows_receipt_sha256",
            "split_receipt_sha256",
            "policy_sha256",
            "fit_source_sha256",
            "atlas_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        _strict_int(self.fit_row_count, label="fit_row_count", minimum=1)
        _strict_int(self.hidden_width, label="hidden_width", minimum=1)
        _strict_int(self.chart_count, label="chart_count", minimum=1)
        _strict_int(self.tangent_rank, label="tangent_rank", minimum=1)
        if self.chart_count not in (1, 2) or self.tangent_rank > self.hidden_width:
            raise ValueError("atlas candidate dimensions are invalid")
        hard = _tuple_of_ints(
            self.hard_chart_counts,
            label="hard_chart_counts",
            minimum=1,
        )
        ranks = _tuple_of_ints(
            self.covariance_numerical_ranks,
            label="covariance_numerical_ranks",
            minimum=1,
        )
        selected = _tuple_of_floats(
            self.covariance_selected_min_eigenvalues,
            label="covariance_selected_min_eigenvalues",
            minimum=0.0,
        )
        maxima = _tuple_of_floats(
            self.covariance_max_eigenvalues,
            label="covariance_max_eigenvalues",
            minimum=0.0,
        )
        if not all(
            len(value) == self.chart_count
            for value in (hard, ranks, selected, maxima)
        ):
            raise ValueError("atlas per-chart evidence has wrong dimensions")
        if sum(hard) != self.fit_row_count:
            raise ValueError("hard chart counts do not close to fit rows")
        if any(count < self.tangent_rank + 1 for count in hard):
            raise ValueError("hard chart support is below R+1")
        if any(rank < self.tangent_rank for rank in ranks):
            raise ValueError("chart covariance rank is below requested R")
        if any(value <= 0.0 for value in selected):
            raise ValueError("selected chart covariance support must be positive")
        object.__setattr__(self, "hard_chart_counts", hard)
        object.__setattr__(self, "covariance_numerical_ranks", ranks)
        object.__setattr__(
            self,
            "covariance_selected_min_eigenvalues",
            selected,
        )
        object.__setattr__(self, "covariance_max_eigenvalues", maxima)
        tau = _strict_float(
            self.initial_chart_temperature,
            label="initial_chart_temperature",
            positive=True,
        )
        lower = _strict_float(
            self.chart_temperature_lower_bound,
            label="chart_temperature_lower_bound",
            positive=True,
        )
        upper = _strict_float(
            self.chart_temperature_upper_bound,
            label="chart_temperature_upper_bound",
            positive=True,
        )
        if self.chart_count == 1:
            if (tau, lower, upper) != (1.0, 1.0, 1.0):
                raise ValueError("C=1 chart temperature must be exactly frozen one")
        elif not lower < tau < upper:
            raise ValueError("C=2 chart temperature bounds must bracket init")
        if self.initialization_id != _ATLAS_INIT_ID:
            raise ValueError("atlas initialization algorithm drifted")
        if self.atlas_fit_id != _ATLAS_FIT_ID:
            raise ValueError("atlas fit algorithm drifted")
        expected = _digest(_ATLAS_RECEIPT_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="receipt_sha256",
        ) != expected:
            raise ValueError("atlas fit receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "atlas_fit_id": self.atlas_fit_id,
            "atlas_sha256": self.atlas_sha256,
            "chart_count": self.chart_count,
            "chart_temperature_lower_bound": self.chart_temperature_lower_bound,
            "chart_temperature_upper_bound": self.chart_temperature_upper_bound,
            "covariance_max_eigenvalues": list(
                self.covariance_max_eigenvalues
            ),
            "covariance_numerical_ranks": list(
                self.covariance_numerical_ranks
            ),
            "covariance_selected_min_eigenvalues": list(
                self.covariance_selected_min_eigenvalues
            ),
            "fit_row_count": self.fit_row_count,
            "fit_source_sha256": self.fit_source_sha256,
            "format_version": _FORMAT_VERSION,
            "hard_chart_counts": list(self.hard_chart_counts),
            "hidden_width": self.hidden_width,
            "initial_chart_temperature": self.initial_chart_temperature,
            "initialization_id": self.initialization_id,
            "layer_ordinal": self.layer_ordinal,
            "lloyd_passes": 12,
            "policy_sha256": self.policy_sha256,
            "schema": _SCHEMA,
            "source_rows_receipt_sha256": self.source_rows_receipt_sha256,
            "split_receipt_sha256": self.split_receipt_sha256,
            "tangent_rank": self.tangent_rank,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "receipt_sha256": self.receipt_sha256,
            **_safety_payload(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeAtlasFitReceipt:
        expected = set(cls._state_fields())
        _strict_fields(state, expected=expected, label="atlas fit receipt")
        _validate_common_state(state)
        if state["lloyd_passes"] != 12:
            raise ValueError("serialized atlas must disclose 12 Lloyd passes")
        for name in (
            "hard_chart_counts",
            "covariance_numerical_ranks",
            "covariance_selected_min_eigenvalues",
            "covariance_max_eigenvalues",
        ):
            if type(state[name]) is not list:
                raise TypeError(f"serialized {name} must be a JSON list")
        return cls(
            layer_ordinal=state["layer_ordinal"],
            source_rows_receipt_sha256=state[
                "source_rows_receipt_sha256"
            ],
            split_receipt_sha256=state["split_receipt_sha256"],
            policy_sha256=state["policy_sha256"],
            fit_source_sha256=state["fit_source_sha256"],
            fit_row_count=state["fit_row_count"],
            hidden_width=state["hidden_width"],
            chart_count=state["chart_count"],
            tangent_rank=state["tangent_rank"],
            hard_chart_counts=_tuple_of_ints(
                state["hard_chart_counts"],
                label="hard_chart_counts",
                minimum=1,
            ),
            covariance_numerical_ranks=_tuple_of_ints(
                state["covariance_numerical_ranks"],
                label="covariance_numerical_ranks",
                minimum=1,
            ),
            covariance_selected_min_eigenvalues=_tuple_of_floats(
                state["covariance_selected_min_eigenvalues"],
                label="covariance_selected_min_eigenvalues",
                minimum=0.0,
            ),
            covariance_max_eigenvalues=_tuple_of_floats(
                state["covariance_max_eigenvalues"],
                label="covariance_max_eigenvalues",
                minimum=0.0,
            ),
            atlas_sha256=state["atlas_sha256"],
            initial_chart_temperature=state["initial_chart_temperature"],
            chart_temperature_lower_bound=state[
                "chart_temperature_lower_bound"
            ],
            chart_temperature_upper_bound=state[
                "chart_temperature_upper_bound"
            ],
            initialization_id=state["initialization_id"],
            atlas_fit_id=state["atlas_fit_id"],
            receipt_sha256=state["receipt_sha256"],
        )

    @staticmethod
    def _state_fields() -> tuple[str, ...]:
        return (
            "atlas_fit_id",
            "atlas_sha256",
            "chart_count",
            "chart_temperature_lower_bound",
            "chart_temperature_upper_bound",
            "covariance_max_eigenvalues",
            "covariance_numerical_ranks",
            "covariance_selected_min_eigenvalues",
            "fit_row_count",
            "fit_source_sha256",
            "format_version",
            "hard_chart_counts",
            "hidden_width",
            "initial_chart_temperature",
            "initialization_id",
            "layer_ordinal",
            "lloyd_passes",
            "policy_sha256",
            "receipt_sha256",
            "schema",
            "source_rows_receipt_sha256",
            "split_receipt_sha256",
            "tangent_rank",
            *_safety_payload().keys(),
        )


def _validate_common_state(state: Mapping[str, object]) -> None:
    if (
        state["schema"] != _SCHEMA
        or type(state["format_version"]) is not int
        or state["format_version"] != _FORMAT_VERSION
    ):
        raise ValueError("fit receipt schema/version is invalid")
    for name, expected in _safety_payload().items():
        if type(state[name]) is not bool or state[name] is not expected:
            raise ValueError(f"fit receipt safety field {name} is invalid")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMultimodeAtlasFit:
    atlas: FixedLayerChartAtlas
    receipt: LayerMultimodeAtlasFitReceipt
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.atlas) is not FixedLayerChartAtlas:
            raise TypeError("atlas must be a FixedLayerChartAtlas")
        if type(self.receipt) is not LayerMultimodeAtlasFitReceipt:
            raise TypeError("receipt must be a LayerMultimodeAtlasFitReceipt")
        self.assert_integrity()

    def assert_integrity(self) -> None:
        self.atlas.assert_integrity()
        if self.atlas.atlas_sha256 != self.receipt.atlas_sha256:
            raise ValueError("atlas tensors drifted from fit receipt")
        if (
            self.atlas.chart_count != self.receipt.chart_count
            or self.atlas.hidden_width != self.receipt.hidden_width
            or self.atlas.tangent_rank != self.receipt.tangent_rank
            or self.atlas.source_fit_sha256 != self.receipt.fit_source_sha256
        ):
            raise ValueError("atlas dimensions/source drifted from fit receipt")

    @property
    def live_atlas_fitter_authorized(self) -> bool:
        """True only for the original atlas returned by deterministic fitting."""

        return (
            self._construction_authority is _LIVE_ATLAS_FITTER_AUTHORITY
            and _LIVE_ATLAS_FITS.get(id(self)) is self
        )


def _family_stats(
    value: object,
) -> tuple[tuple[str, int, float, float, float, float, float, float], ...]:
    if type(value) not in (tuple, list) or not value:
        raise TypeError("family_fidelity_sufficient_statistics must be nonempty")
    result: list[tuple[str, int, float, float, float, float, float, float]] = []
    for entry in value:
        if type(entry) not in (tuple, list) or len(entry) != 8:
            raise TypeError("family sufficient-stat entries need eight fields")
        family, count, sse, teacher, nrmse, prediction, dot, cosine = entry
        if not isinstance(family, str) or not family:
            raise ValueError("family ID must be nonempty")
        count = _strict_int(count, label="family scalar_count", minimum=1)
        sse = _strict_float(sse, label="family SSE", minimum=0.0)
        teacher = _strict_float(
            teacher,
            label="family teacher energy",
            positive=True,
        )
        nrmse = _strict_float(nrmse, label="family NRMSE", minimum=0.0)
        prediction = _strict_float(
            prediction,
            label="family prediction energy",
            positive=True,
        )
        dot = _strict_float(dot, label="family prediction-teacher dot")
        cosine = _strict_float(
            cosine,
            label="family cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        if not math.isclose(
            nrmse,
            math.sqrt(sse / teacher),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("family NRMSE does not recompute")
        expected_cosine = dot / math.sqrt(prediction * teacher)
        if not math.isclose(
            cosine,
            expected_cosine,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("family cosine does not recompute")
        result.append(
            (family, count, sse, teacher, nrmse, prediction, dot, cosine)
        )
    if tuple(entry[0] for entry in result) != tuple(
        sorted(entry[0] for entry in result)
    ) or len({entry[0] for entry in result}) != len(result):
        raise ValueError("family sufficient statistics must be sorted and unique")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LayerMultimodeValidationEvidence:
    layer_ordinal: int
    chart_count: int
    modes_per_chart: int
    row_count: int
    validation_source_sha256: str
    velocity_nrmse: float
    aggregate_velocity_cosine: float
    velocity_sse: float
    validation_scalar_count: int
    validation_prediction_energy: float
    validation_teacher_energy: float
    validation_prediction_teacher_dot: float
    teacher_velocity_rms: float
    family_velocity_nrmse: tuple[tuple[str, float], ...]
    family_fidelity_sufficient_statistics: tuple[
        tuple[str, int, float, float, float, float, float, float], ...
    ]
    chart_soft_masses: tuple[float, ...]
    conditional_mode_masses: tuple[tuple[float, ...], ...]
    per_chart_mean_row_router_entropy_nats: tuple[float, ...]
    per_chart_state_mode_mutual_information_nats: tuple[float, ...]
    per_chart_minimum_pairwise_velocity_separation_rms: tuple[float, ...]
    per_chart_minimum_pairwise_velocity_separation_ratio: tuple[float, ...]
    hard_joint_route_counts: tuple[int, ...]
    sufficient_statistics_weighting_id: str = _VALIDATION_WEIGHTING_ID
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        _strict_int(self.layer_ordinal, label="layer_ordinal")
        if self.layer_ordinal >= 18:
            raise ValueError("layer_ordinal must be in [0,17]")
        _strict_int(self.chart_count, label="chart_count", minimum=1)
        _strict_int(self.modes_per_chart, label="modes_per_chart", minimum=1)
        _strict_int(self.row_count, label="row_count", minimum=1)
        _require_sha256(
            self.validation_source_sha256,
            label="validation_source_sha256",
        )
        _strict_float(
            self.velocity_nrmse,
            label="velocity_nrmse",
            minimum=0.0,
        )
        _strict_float(
            self.aggregate_velocity_cosine,
            label="aggregate_velocity_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        _strict_float(self.velocity_sse, label="velocity_sse", minimum=0.0)
        _strict_int(
            self.validation_scalar_count,
            label="validation_scalar_count",
            minimum=1,
        )
        for label in (
            "validation_prediction_energy",
            "validation_teacher_energy",
        ):
            _strict_float(
                getattr(self, label),
                label=label,
                positive=True,
            )
        _strict_float(
            self.validation_prediction_teacher_dot,
            label="validation_prediction_teacher_dot",
        )
        if self.validation_scalar_count != self.row_count * 640:
            raise ValueError("validation scalar count must equal N*H640")
        if not math.isclose(
            self.velocity_nrmse,
            math.sqrt(self.velocity_sse / self.validation_teacher_energy),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("aggregate validation NRMSE does not recompute")
        expected_cosine = self.validation_prediction_teacher_dot / math.sqrt(
            self.validation_prediction_energy * self.validation_teacher_energy
        )
        if not math.isclose(
            self.aggregate_velocity_cosine,
            expected_cosine,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("aggregate validation cosine does not recompute")
        _strict_float(
            self.teacher_velocity_rms,
            label="teacher_velocity_rms",
            positive=True,
        )
        if not math.isclose(
            self.teacher_velocity_rms,
            math.sqrt(
                self.validation_teacher_energy / self.validation_scalar_count
            ),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("teacher velocity RMS does not recompute")
        if (
            type(self.family_velocity_nrmse) is not tuple
            or not self.family_velocity_nrmse
        ):
            raise TypeError("family_velocity_nrmse must be a nonempty tuple")
        family_names: list[str] = []
        for pair in self.family_velocity_nrmse:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("family velocity entries must be exact pairs")
            family, value = pair
            if not isinstance(family, str) or not family:
                raise ValueError("family ID must be nonempty")
            _strict_float(value, label="family velocity NRMSE", minimum=0.0)
            family_names.append(family)
        if (
            tuple(family_names) != tuple(sorted(family_names))
            or len(set(family_names)) != len(family_names)
        ):
            raise ValueError("family velocity entries must be sorted and unique")
        family_stats = _family_stats(
            self.family_fidelity_sufficient_statistics
        )
        if tuple(family_names) != tuple(entry[0] for entry in family_stats):
            raise ValueError("family metric catalogs do not align")
        if any(
            not math.isclose(pair[1], entry[4], rel_tol=1e-12, abs_tol=1e-12)
            for pair, entry in zip(
                self.family_velocity_nrmse,
                family_stats,
                strict=True,
            )
        ):
            raise ValueError("family NRMSE catalogs do not close")
        if (
            sum(entry[1] for entry in family_stats)
            != self.validation_scalar_count
            or not math.isclose(
                math.fsum(entry[2] for entry in family_stats),
                self.velocity_sse,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                math.fsum(entry[3] for entry in family_stats),
                self.validation_teacher_energy,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                math.fsum(entry[5] for entry in family_stats),
                self.validation_prediction_energy,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                math.fsum(entry[6] for entry in family_stats),
                self.validation_prediction_teacher_dot,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
        ):
            raise ValueError("family sufficient statistics do not aggregate")
        object.__setattr__(
            self,
            "family_fidelity_sufficient_statistics",
            family_stats,
        )
        chart_masses = _tuple_of_floats(
            self.chart_soft_masses,
            label="chart_soft_masses",
            minimum=0.0,
        )
        if len(chart_masses) != self.chart_count or any(
            value <= 0.0 for value in chart_masses
        ):
            raise ValueError("every chart must have positive soft mass")
        if not math.isclose(sum(chart_masses), 1.0, abs_tol=_MASS_ATOL):
            raise ValueError("chart soft masses do not sum to one")
        if self.chart_count == 1 and chart_masses != (1.0,):
            raise ValueError("C=1 chart mass must be exactly one")
        if (
            type(self.conditional_mode_masses) is not tuple
            or len(self.conditional_mode_masses) != self.chart_count
        ):
            raise ValueError("conditional mode masses need one row per chart")
        conditional: list[tuple[float, ...]] = []
        for chart_ordinal, values in enumerate(self.conditional_mode_masses):
            row = _tuple_of_floats(
                values,
                label=f"conditional_mode_masses[{chart_ordinal}]",
                minimum=0.0,
            )
            if len(row) != self.modes_per_chart or any(v <= 0.0 for v in row):
                raise ValueError("every chart-local mode must have positive mass")
            if not math.isclose(sum(row), 1.0, abs_tol=_MASS_ATOL):
                raise ValueError("conditional mode masses do not sum to one")
            conditional.append(row)
        object.__setattr__(self, "chart_soft_masses", chart_masses)
        object.__setattr__(self, "conditional_mode_masses", tuple(conditional))
        mean_entropy = _tuple_of_floats(
            self.per_chart_mean_row_router_entropy_nats,
            label="per_chart_mean_row_router_entropy_nats",
            minimum=0.0,
        )
        mi = _tuple_of_floats(
            self.per_chart_state_mode_mutual_information_nats,
            label="per_chart_state_mode_mutual_information_nats",
            minimum=0.0,
        )
        separation_rms = _tuple_of_floats(
            self.per_chart_minimum_pairwise_velocity_separation_rms,
            label="per_chart_minimum_pairwise_velocity_separation_rms",
            minimum=0.0,
        )
        separation_ratio = _tuple_of_floats(
            self.per_chart_minimum_pairwise_velocity_separation_ratio,
            label="per_chart_minimum_pairwise_velocity_separation_ratio",
            minimum=0.0,
        )
        if any(
            len(values) != self.chart_count
            for values in (mean_entropy, mi, separation_rms, separation_ratio)
        ):
            raise ValueError("per-chart dynamic evidence has wrong dimensions")
        for chart_ordinal, masses in enumerate(conditional):
            mass_entropy = -math.fsum(
                value * math.log(value) for value in masses if value > 0.0
            )
            if not math.isclose(
                mi[chart_ordinal],
                mass_entropy - mean_entropy[chart_ordinal],
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            ):
                raise ValueError("per-chart state/mode MI does not recompute")
            if not math.isclose(
                separation_ratio[chart_ordinal],
                separation_rms[chart_ordinal] / self.teacher_velocity_rms,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise ValueError("per-chart separation ratio does not recompute")
        if self.modes_per_chart == 1 and (
            mean_entropy != (0.0,) * self.chart_count
            or mi != (0.0,) * self.chart_count
            or separation_rms != (0.0,) * self.chart_count
            or separation_ratio != (0.0,) * self.chart_count
            or any(row != (1.0,) for row in conditional)
        ):
            raise ValueError("M=1 evidence must be exactly degenerate and safe")
        object.__setattr__(
            self,
            "per_chart_mean_row_router_entropy_nats",
            mean_entropy,
        )
        object.__setattr__(
            self,
            "per_chart_state_mode_mutual_information_nats",
            mi,
        )
        object.__setattr__(
            self,
            "per_chart_minimum_pairwise_velocity_separation_rms",
            separation_rms,
        )
        object.__setattr__(
            self,
            "per_chart_minimum_pairwise_velocity_separation_ratio",
            separation_ratio,
        )
        counts = _tuple_of_ints(
            self.hard_joint_route_counts,
            label="hard_joint_route_counts",
        )
        if (
            len(counts) != self.chart_count * self.modes_per_chart
            or sum(counts) != self.row_count
        ):
            raise ValueError("hard joint-route counts do not close")
        object.__setattr__(self, "hard_joint_route_counts", counts)
        if self.sufficient_statistics_weighting_id != _VALIDATION_WEIGHTING_ID:
            raise ValueError("validation sufficient-stat weighting drifted")
        expected = _digest(_EVIDENCE_DOMAIN, self._payload())
        if self.evidence_sha256 == "":
            object.__setattr__(self, "evidence_sha256", expected)
        elif _require_sha256(
            self.evidence_sha256,
            label="evidence_sha256",
        ) != expected:
            raise ValueError("validation evidence hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "aggregate_velocity_cosine": self.aggregate_velocity_cosine,
            "chart_count": self.chart_count,
            "chart_soft_masses": list(self.chart_soft_masses),
            "conditional_mode_masses": [
                list(row) for row in self.conditional_mode_masses
            ],
            "family_velocity_nrmse": [
                list(pair) for pair in self.family_velocity_nrmse
            ],
            "family_fidelity_sufficient_statistics": [
                list(entry)
                for entry in self.family_fidelity_sufficient_statistics
            ],
            "format_version": _FORMAT_VERSION,
            "hard_joint_route_counts": list(self.hard_joint_route_counts),
            "layer_ordinal": self.layer_ordinal,
            "modes_per_chart": self.modes_per_chart,
            "per_chart_mean_row_router_entropy_nats": list(
                self.per_chart_mean_row_router_entropy_nats
            ),
            "per_chart_minimum_pairwise_velocity_separation_ratio": list(
                self.per_chart_minimum_pairwise_velocity_separation_ratio
            ),
            "per_chart_minimum_pairwise_velocity_separation_rms": list(
                self.per_chart_minimum_pairwise_velocity_separation_rms
            ),
            "per_chart_state_mode_mutual_information_nats": list(
                self.per_chart_state_mode_mutual_information_nats
            ),
            "row_count": self.row_count,
            "schema": _SCHEMA,
            "sufficient_statistics_weighting_id": (
                self.sufficient_statistics_weighting_id
            ),
            "teacher_velocity_rms": self.teacher_velocity_rms,
            "validation_prediction_energy": self.validation_prediction_energy,
            "validation_prediction_teacher_dot": (
                self.validation_prediction_teacher_dot
            ),
            "validation_scalar_count": self.validation_scalar_count,
            "validation_source_sha256": self.validation_source_sha256,
            "validation_teacher_energy": self.validation_teacher_energy,
            "velocity_nrmse": self.velocity_nrmse,
            "velocity_sse": self.velocity_sse,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "evidence_sha256": self.evidence_sha256,
            **_safety_payload(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeValidationEvidence:
        expected = {
            *cls._payload_field_names(),
            "evidence_sha256",
            *_safety_payload().keys(),
        }
        _strict_fields(state, expected=expected, label="validation evidence")
        _validate_common_state(state)
        raw_family = state["family_velocity_nrmse"]
        if type(raw_family) is not list:
            raise TypeError("serialized family metrics must be a list")
        family: list[tuple[str, float]] = []
        for value in raw_family:
            if type(value) is not list or len(value) != 2:
                raise TypeError("serialized family metric must be a list pair")
            family.append((value[0], value[1]))
        raw_conditional = state["conditional_mode_masses"]
        if type(raw_conditional) is not list:
            raise TypeError("serialized conditional masses must be a list")
        for row in raw_conditional:
            if type(row) is not list:
                raise TypeError("serialized conditional mass rows must be lists")
        for name in (
            "chart_soft_masses",
            "family_fidelity_sufficient_statistics",
            "hard_joint_route_counts",
            "per_chart_mean_row_router_entropy_nats",
            "per_chart_state_mode_mutual_information_nats",
            "per_chart_minimum_pairwise_velocity_separation_rms",
            "per_chart_minimum_pairwise_velocity_separation_ratio",
        ):
            if type(state[name]) is not list:
                raise TypeError(f"serialized {name} must be a JSON list")
        for entry in state["family_fidelity_sufficient_statistics"]:
            if type(entry) is not list:
                raise TypeError("serialized family sufficient stats must be lists")
        return cls(
            layer_ordinal=state["layer_ordinal"],
            chart_count=state["chart_count"],
            modes_per_chart=state["modes_per_chart"],
            row_count=state["row_count"],
            validation_source_sha256=state["validation_source_sha256"],
            velocity_nrmse=state["velocity_nrmse"],
            aggregate_velocity_cosine=state["aggregate_velocity_cosine"],
            velocity_sse=state["velocity_sse"],
            validation_scalar_count=state["validation_scalar_count"],
            validation_prediction_energy=state[
                "validation_prediction_energy"
            ],
            validation_teacher_energy=state["validation_teacher_energy"],
            validation_prediction_teacher_dot=state[
                "validation_prediction_teacher_dot"
            ],
            teacher_velocity_rms=state["teacher_velocity_rms"],
            family_velocity_nrmse=tuple(family),
            family_fidelity_sufficient_statistics=_family_stats(
                state["family_fidelity_sufficient_statistics"]
            ),
            chart_soft_masses=_tuple_of_floats(
                state["chart_soft_masses"],
                label="chart_soft_masses",
                minimum=0.0,
            ),
            conditional_mode_masses=tuple(
                _tuple_of_floats(
                    row,
                    label="conditional mode row",
                    minimum=0.0,
                )
                for row in raw_conditional
            ),
            per_chart_mean_row_router_entropy_nats=_tuple_of_floats(
                state["per_chart_mean_row_router_entropy_nats"],
                label="per_chart_mean_row_router_entropy_nats",
                minimum=0.0,
            ),
            per_chart_state_mode_mutual_information_nats=_tuple_of_floats(
                state["per_chart_state_mode_mutual_information_nats"],
                label="per_chart_state_mode_mutual_information_nats",
                minimum=0.0,
            ),
            per_chart_minimum_pairwise_velocity_separation_rms=_tuple_of_floats(
                state["per_chart_minimum_pairwise_velocity_separation_rms"],
                label="per_chart_minimum_pairwise_velocity_separation_rms",
                minimum=0.0,
            ),
            per_chart_minimum_pairwise_velocity_separation_ratio=_tuple_of_floats(
                state["per_chart_minimum_pairwise_velocity_separation_ratio"],
                label="per_chart_minimum_pairwise_velocity_separation_ratio",
                minimum=0.0,
            ),
            hard_joint_route_counts=_tuple_of_ints(
                state["hard_joint_route_counts"],
                label="hard_joint_route_counts",
            ),
            sufficient_statistics_weighting_id=state[
                "sufficient_statistics_weighting_id"
            ],
            evidence_sha256=state["evidence_sha256"],
        )

    @staticmethod
    def _payload_field_names() -> tuple[str, ...]:
        return (
            "aggregate_velocity_cosine",
            "chart_count",
            "chart_soft_masses",
            "conditional_mode_masses",
            "family_fidelity_sufficient_statistics",
            "family_velocity_nrmse",
            "format_version",
            "hard_joint_route_counts",
            "layer_ordinal",
            "modes_per_chart",
            "per_chart_mean_row_router_entropy_nats",
            "per_chart_minimum_pairwise_velocity_separation_ratio",
            "per_chart_minimum_pairwise_velocity_separation_rms",
            "per_chart_state_mode_mutual_information_nats",
            "row_count",
            "schema",
            "sufficient_statistics_weighting_id",
            "teacher_velocity_rms",
            "validation_prediction_energy",
            "validation_prediction_teacher_dot",
            "validation_scalar_count",
            "validation_source_sha256",
            "validation_teacher_energy",
            "velocity_nrmse",
            "velocity_sse",
        )


def _loss_terms(value: object, *, label: str) -> tuple[tuple[str, float], ...]:
    names = ("distinctness", "fidelity", "load_balance", "sparsity", "total")
    if type(value) not in (tuple, list) or len(value) != len(names):
        raise ValueError(f"{label} must contain the five canonical terms")
    result: list[tuple[str, float]] = []
    for expected_name, pair in zip(names, value, strict=True):
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise TypeError(f"{label} entries must be pairs")
        name, term = pair
        if name != expected_name:
            raise ValueError(f"{label} term order/name drifted")
        result.append(
            (name, _strict_float(term, label=f"{label} {name}", minimum=0.0))
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LayerMultimodeCandidateFitReceipt:
    layer_ordinal: int
    chart_count: int
    modes_per_chart: int
    tangent_rank: int
    source_rows_receipt_sha256: str
    source_capture_scope: str
    split_receipt_sha256: str
    policy_sha256: str
    fit_policy_test_only: bool
    production_fit_authorized: bool
    fit_source_sha256: str
    validation_source_sha256: str
    initialization_seed: int
    fit_row_count: int
    validation_row_count: int
    optimizer_steps_completed: int
    atlas_fit: LayerMultimodeAtlasFitReceipt
    model_parameter_sha256: str
    resource_receipt_sha256: str
    candidate_total_parameter_count: int
    candidate_matrix_macs_per_token: int
    retained_rmsnorm_vector_count: int
    final_chart_temperature: float
    chart_temperature_fitted_on_A: bool
    mode_temperature: float
    output_rmsnorm_folded_into_teacher_velocity: bool
    initial_fit_loss_terms: tuple[tuple[str, float], ...]
    final_fit_loss_terms: tuple[tuple[str, float], ...]
    validation: LayerMultimodeValidationEvidence
    optimizer_id: str = _OPTIMIZER_ID
    gradient_clip_grouping_id: str = _GRADIENT_CLIP_GROUPING_ID
    fit_role: str = A_DEVELOPMENT_ROLE
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for label, value, minimum in (
            ("layer_ordinal", self.layer_ordinal, 0),
            ("chart_count", self.chart_count, 1),
            ("modes_per_chart", self.modes_per_chart, 1),
            ("tangent_rank", self.tangent_rank, 1),
            ("initialization_seed", self.initialization_seed, 0),
            ("fit_row_count", self.fit_row_count, 1),
            ("validation_row_count", self.validation_row_count, 1),
            (
                "optimizer_steps_completed",
                self.optimizer_steps_completed,
                1,
            ),
            (
                "candidate_total_parameter_count",
                self.candidate_total_parameter_count,
                1,
            ),
            (
                "candidate_matrix_macs_per_token",
                self.candidate_matrix_macs_per_token,
                1,
            ),
            (
                "retained_rmsnorm_vector_count",
                self.retained_rmsnorm_vector_count,
                0,
            ),
        ):
            _strict_int(value, label=label, minimum=minimum)
        if self.layer_ordinal >= 18:
            raise ValueError("layer_ordinal must be in [0,17]")
        if self.chart_count not in (1, 2) or self.modes_per_chart not in (1, 2, 4):
            raise ValueError("candidate C/M is outside the frozen ladder")
        if self.retained_rmsnorm_vector_count != 1:
            raise ValueError("compiled MLP retains only its input RMSNorm")
        if self.source_capture_scope not in (
            PRODUCTION_FULL_A_CAPTURE_SCOPE,
            DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE,
        ) or type(self.source_capture_scope) is not str:
            raise ValueError("source_capture_scope is outside the frozen boundary")
        if type(self.fit_policy_test_only) is not bool:
            raise TypeError("fit_policy_test_only must be bool")
        if type(self.production_fit_authorized) is not bool:
            raise TypeError("production_fit_authorized must be bool")
        if self.fit_policy_test_only is (
            self.policy_sha256 == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
        ):
            raise ValueError(
                "fit policy test-only status conflicts with the production hash"
            )
        expected_production_authority = (
            not self.fit_policy_test_only
            and self.policy_sha256
            == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
            and self.source_capture_scope == PRODUCTION_FULL_A_CAPTURE_SCOPE
        )
        if self.production_fit_authorized is not expected_production_authority:
            raise ValueError(
                "production_fit_authorized does not recompute from policy/scope"
            )
        if type(self.atlas_fit) is not LayerMultimodeAtlasFitReceipt:
            raise TypeError("atlas_fit must be an atlas fit receipt")
        if type(self.validation) is not LayerMultimodeValidationEvidence:
            raise TypeError("validation must be validation evidence")
        final_tau = _strict_float(
            self.final_chart_temperature,
            label="final_chart_temperature",
            positive=True,
        )
        if type(self.chart_temperature_fitted_on_A) is not bool:
            raise TypeError("chart_temperature_fitted_on_A must be bool")
        if type(self.output_rmsnorm_folded_into_teacher_velocity) is not bool:
            raise TypeError(
                "output_rmsnorm_folded_into_teacher_velocity must be bool"
            )
        if not self.output_rmsnorm_folded_into_teacher_velocity:
            raise ValueError("postnorm teacher target must fold/remove output RMSNorm")
        if (
            _strict_float(
                self.mode_temperature,
                label="mode_temperature",
                positive=True,
            )
            != 1.0
        ):
            raise ValueError("mode_temperature must be frozen exactly 1.0")
        if self.chart_count == 1:
            if final_tau != 1.0 or self.chart_temperature_fitted_on_A:
                raise ValueError("C=1 tau must be exactly one and frozen")
        elif (
            not self.chart_temperature_fitted_on_A
            or final_tau < self.atlas_fit.chart_temperature_lower_bound
            or final_tau > self.atlas_fit.chart_temperature_upper_bound
        ):
            raise ValueError("C=2 tau must be A-fitted within frozen bounds")
        for label in (
            "source_rows_receipt_sha256",
            "split_receipt_sha256",
            "policy_sha256",
            "fit_source_sha256",
            "validation_source_sha256",
            "model_parameter_sha256",
            "resource_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        dimensions = (
            self.layer_ordinal,
            self.chart_count,
            self.tangent_rank,
        )
        if dimensions != (
            self.atlas_fit.layer_ordinal,
            self.atlas_fit.chart_count,
            self.atlas_fit.tangent_rank,
        ):
            raise ValueError("candidate and atlas dimensions drifted")
        if (
            self.fit_source_sha256 != self.atlas_fit.fit_source_sha256
            or self.source_rows_receipt_sha256
            != self.atlas_fit.source_rows_receipt_sha256
            or self.split_receipt_sha256 != self.atlas_fit.split_receipt_sha256
            or self.policy_sha256 != self.atlas_fit.policy_sha256
            or self.fit_row_count != self.atlas_fit.fit_row_count
        ):
            raise ValueError("candidate and atlas sources drifted")
        if (
            self.validation.layer_ordinal != self.layer_ordinal
            or self.validation.chart_count != self.chart_count
            or self.validation.modes_per_chart != self.modes_per_chart
            or self.validation.row_count != self.validation_row_count
            or self.validation.validation_source_sha256
            != self.validation_source_sha256
        ):
            raise ValueError("candidate and validation evidence drifted")
        object.__setattr__(
            self,
            "initial_fit_loss_terms",
            _loss_terms(self.initial_fit_loss_terms, label="initial_fit_loss_terms"),
        )
        object.__setattr__(
            self,
            "final_fit_loss_terms",
            _loss_terms(self.final_fit_loss_terms, label="final_fit_loss_terms"),
        )
        if self.optimizer_id != _OPTIMIZER_ID:
            raise ValueError("optimizer algorithm drifted")
        if self.gradient_clip_grouping_id != _GRADIENT_CLIP_GROUPING_ID:
            raise ValueError("gradient clip grouping algorithm drifted")
        if self.fit_role != A_DEVELOPMENT_ROLE:
            raise ValueError("only A exposed development rows may be fitted")
        expected = _digest(_FIT_RECEIPT_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="receipt_sha256",
        ) != expected:
            raise ValueError("candidate fit receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "atlas_fit": self.atlas_fit.state_dict(),
            "candidate_matrix_macs_per_token": (
                self.candidate_matrix_macs_per_token
            ),
            "candidate_total_parameter_count": (
                self.candidate_total_parameter_count
            ),
            "chart_count": self.chart_count,
            "chart_temperature_fitted_on_A": (
                self.chart_temperature_fitted_on_A
            ),
            "final_fit_loss_terms": [
                list(pair) for pair in self.final_fit_loss_terms
            ],
            "fit_role": self.fit_role,
            "fit_row_count": self.fit_row_count,
            "fit_policy_test_only": self.fit_policy_test_only,
            "fit_source_sha256": self.fit_source_sha256,
            "final_chart_temperature": self.final_chart_temperature,
            "format_version": _FORMAT_VERSION,
            "gradient_clip_grouping_id": self.gradient_clip_grouping_id,
            "initial_fit_loss_terms": [
                list(pair) for pair in self.initial_fit_loss_terms
            ],
            "initialization_seed": self.initialization_seed,
            "layer_ordinal": self.layer_ordinal,
            "model_parameter_sha256": self.model_parameter_sha256,
            "modes_per_chart": self.modes_per_chart,
            "mode_temperature": self.mode_temperature,
            "optimizer_id": self.optimizer_id,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "output_rmsnorm_folded_into_teacher_velocity": (
                self.output_rmsnorm_folded_into_teacher_velocity
            ),
            "policy_sha256": self.policy_sha256,
            "production_fit_authorized": self.production_fit_authorized,
            "resource_receipt_sha256": self.resource_receipt_sha256,
            "retained_rmsnorm_vector_count": (
                self.retained_rmsnorm_vector_count
            ),
            "schema": _SCHEMA,
            "source_rows_receipt_sha256": self.source_rows_receipt_sha256,
            "source_capture_scope": self.source_capture_scope,
            "split_receipt_sha256": self.split_receipt_sha256,
            "tangent_rank": self.tangent_rank,
            "validation": self.validation.state_dict(),
            "validation_row_count": self.validation_row_count,
            "validation_source_sha256": self.validation_source_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "receipt_sha256": self.receipt_sha256,
            **_safety_payload(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeCandidateFitReceipt:
        expected = {
            *cls._payload_field_names(),
            "receipt_sha256",
            *_safety_payload().keys(),
        }
        _strict_fields(state, expected=expected, label="candidate fit receipt")
        _validate_common_state(state)
        if not isinstance(state["atlas_fit"], Mapping) or not isinstance(
            state["validation"], Mapping
        ):
            raise TypeError("candidate nested receipts must be mappings")
        for name in ("initial_fit_loss_terms", "final_fit_loss_terms"):
            if type(state[name]) is not list or any(
                type(value) is not list for value in state[name]
            ):
                raise TypeError(f"serialized {name} must be JSON list pairs")
        return cls(
            layer_ordinal=state["layer_ordinal"],
            chart_count=state["chart_count"],
            modes_per_chart=state["modes_per_chart"],
            tangent_rank=state["tangent_rank"],
            source_rows_receipt_sha256=state[
                "source_rows_receipt_sha256"
            ],
            source_capture_scope=state["source_capture_scope"],
            split_receipt_sha256=state["split_receipt_sha256"],
            policy_sha256=state["policy_sha256"],
            fit_policy_test_only=state["fit_policy_test_only"],
            production_fit_authorized=state["production_fit_authorized"],
            fit_source_sha256=state["fit_source_sha256"],
            validation_source_sha256=state["validation_source_sha256"],
            initialization_seed=state["initialization_seed"],
            fit_row_count=state["fit_row_count"],
            validation_row_count=state["validation_row_count"],
            optimizer_steps_completed=state["optimizer_steps_completed"],
            atlas_fit=LayerMultimodeAtlasFitReceipt.from_state_dict(
                state["atlas_fit"]
            ),
            model_parameter_sha256=state["model_parameter_sha256"],
            resource_receipt_sha256=state["resource_receipt_sha256"],
            candidate_total_parameter_count=state[
                "candidate_total_parameter_count"
            ],
            candidate_matrix_macs_per_token=state[
                "candidate_matrix_macs_per_token"
            ],
            retained_rmsnorm_vector_count=state[
                "retained_rmsnorm_vector_count"
            ],
            final_chart_temperature=state["final_chart_temperature"],
            chart_temperature_fitted_on_A=state[
                "chart_temperature_fitted_on_A"
            ],
            mode_temperature=state["mode_temperature"],
            output_rmsnorm_folded_into_teacher_velocity=state[
                "output_rmsnorm_folded_into_teacher_velocity"
            ],
            initial_fit_loss_terms=_loss_terms(
                state["initial_fit_loss_terms"],
                label="initial_fit_loss_terms",
            ),
            final_fit_loss_terms=_loss_terms(
                state["final_fit_loss_terms"],
                label="final_fit_loss_terms",
            ),
            validation=LayerMultimodeValidationEvidence.from_state_dict(
                state["validation"]
            ),
            optimizer_id=state["optimizer_id"],
            gradient_clip_grouping_id=state["gradient_clip_grouping_id"],
            fit_role=state["fit_role"],
            receipt_sha256=state["receipt_sha256"],
        )

    @staticmethod
    def _payload_field_names() -> tuple[str, ...]:
        return (
            "atlas_fit",
            "candidate_matrix_macs_per_token",
            "candidate_total_parameter_count",
            "chart_count",
            "chart_temperature_fitted_on_A",
            "final_fit_loss_terms",
            "fit_role",
            "fit_row_count",
            "fit_policy_test_only",
            "fit_source_sha256",
            "final_chart_temperature",
            "format_version",
            "gradient_clip_grouping_id",
            "initial_fit_loss_terms",
            "initialization_seed",
            "layer_ordinal",
            "model_parameter_sha256",
            "modes_per_chart",
            "mode_temperature",
            "optimizer_id",
            "optimizer_steps_completed",
            "output_rmsnorm_folded_into_teacher_velocity",
            "policy_sha256",
            "production_fit_authorized",
            "resource_receipt_sha256",
            "retained_rmsnorm_vector_count",
            "schema",
            "source_rows_receipt_sha256",
            "source_capture_scope",
            "split_receipt_sha256",
            "tangent_rank",
            "validation",
            "validation_row_count",
            "validation_source_sha256",
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMultimodeCandidateFit:
    model: AffineLocalLayerMultimodeFlow
    atlas_fit: LayerMultimodeAtlasFit
    receipt: LayerMultimodeCandidateFitReceipt
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.model) is not AffineLocalLayerMultimodeFlow:
            raise TypeError("model must be an AffineLocalLayerMultimodeFlow")
        if type(self.atlas_fit) is not LayerMultimodeAtlasFit:
            raise TypeError("atlas_fit must be a LayerMultimodeAtlasFit")
        if type(self.receipt) is not LayerMultimodeCandidateFitReceipt:
            raise TypeError("receipt must be a candidate fit receipt")
        self.assert_integrity()

    def assert_integrity(self) -> None:
        self.atlas_fit.assert_integrity()
        if self.model.model_parameter_sha256 != self.receipt.model_parameter_sha256:
            raise ValueError("fitted model parameters drifted from receipt")
        if self.model.atlas.atlas_sha256 != self.atlas_fit.atlas.atlas_sha256:
            raise ValueError("fitted model atlas drifted")
        resources = self.model.resource_accounting
        if (
            resources.receipt_sha256 != self.receipt.resource_receipt_sha256
            or resources.candidate_total_parameter_count
            != self.receipt.candidate_total_parameter_count
            or resources.candidate_matrix_and_mixture_macs_per_token
            != self.receipt.candidate_matrix_macs_per_token
            or self.model.config.retained_rmsnorm_vector_count != 1
        ):
            raise ValueError("fitted model resource ABI drifted")

    @property
    def live_fitter_authorized(self) -> bool:
        """True only for the original result returned by the live fitter."""

        return (
            self._construction_authority is _LIVE_FITTER_AUTHORITY
            and _LIVE_CANDIDATE_FITS.get(id(self)) is self
            and self.atlas_fit.live_atlas_fitter_authorized
        )


def _validate_fit_request(
    rows: LayerMLPFlowRows,
    *,
    chart_count: int,
    modes_per_chart: int,
    tangent_rank: int,
    policy: LayerMultimodeFitPolicy,
    allow_test_policy: bool,
) -> None:
    if type(rows) is not LayerMLPFlowRows or rows.role != A_DEVELOPMENT_ROLE:
        raise ValueError("fitting requires authenticated A development rows")
    if rows.predecessor_source_role != PREDECESSOR_A_ROLE:
        raise ValueError("fitting requires the predecessor A role witness")
    if type(policy) is not LayerMultimodeFitPolicy:
        raise TypeError("policy must be a LayerMultimodeFitPolicy")
    if type(allow_test_policy) is not bool:
        raise TypeError("allow_test_policy must be bool")
    if policy.test_only:
        if not allow_test_policy:
            raise ValueError("test-only fit policy requires explicit opt-in")
    else:
        if (
            policy.policy_sha256
            != PRODUCTION_LAYER_MULTIMODE_FIT_POLICY.policy_sha256
        ):
            raise ValueError(
                "non-test fit policy must equal the frozen production policy"
            )
        if rows.source_capture_scope != PRODUCTION_FULL_A_CAPTURE_SCOPE:
            raise ValueError(
                "production fitting requires the authenticated full-A spool bridge"
            )
    if policy.lloyd_passes != 12:
        raise ValueError("the fit protocol requires exactly 12 Lloyd passes")
    for label, value in (
        ("chart_count", chart_count),
        ("modes_per_chart", modes_per_chart),
        ("tangent_rank", tangent_rank),
    ):
        _strict_int(value, label=label, minimum=1)
    if chart_count not in (1, 2) or modes_per_chart not in (1, 2, 4):
        raise ValueError("candidate C/M is outside the frozen ladder")
    if tangent_rank > 640:
        raise ValueError("tangent rank exceeds H640")
    if not policy.test_only and tangent_rank not in (16, 32, 64):
        raise ValueError("production tangent rank is outside R=(64,32,16)")


def _development_masks(
    rows: LayerMLPFlowRows,
    policy: LayerMultimodeFitPolicy,
) -> tuple[Tensor, Tensor, str]:
    if not policy.test_only:
        fit, validation, _ = fixed_A_development_split_masks(
            rows
        )
    else:
        fit = torch.zeros(rows.row_count, dtype=torch.bool)
        validation = torch.zeros_like(fit)
        by_family: dict[str, set[int]] = {}
        for family, length, (start, end) in zip(
            rows.family_ids,
            rows.token_counts,
            rows.example_row_ranges,
            strict=True,
        ):
            seen = by_family.setdefault(family, set())
            if length in seen:
                raise ValueError("test split repeats a length within one family")
            seen.add(length)
            if length in policy.fit_target_token_counts:
                fit[start:end] = True
            elif length in policy.validation_target_token_counts:
                validation[start:end] = True
            else:
                raise ValueError("test split encountered an unregistered length")
        expected = set(
            (*policy.fit_target_token_counts, *policy.validation_target_token_counts)
        )
        if any(value != expected for value in by_family.values()):
            raise ValueError("every family must cover every test-policy length")
    if (
        fit.shape != validation.shape
        or fit.shape != (rows.row_count,)
        or bool((fit & validation).any())
        or not bool((fit | validation).all())
        or not bool(fit.any())
        or not bool(validation.any())
    ):
        raise ValueError("A fit/validation masks do not close")
    receipt = _digest(
        _MASK_DOMAIN,
        {
            "fit_mask_sha256": _mask_sha256(fit.contiguous()),
            "fit_target_token_counts": list(policy.fit_target_token_counts),
            "layer_ordinal": rows.layer_ordinal,
            "mode_id": rows.mode_id,
            "policy_sha256": policy.policy_sha256,
            "row_axis_sha256": _digest(
                _MASK_DOMAIN,
                {
                    "example_ids": list(rows.example_ids),
                    "family_ids": list(rows.family_ids),
                    "row_keys": [list(value) for value in rows.row_keys],
                    "token_counts": list(rows.token_counts),
                },
            ),
            "validation_mask_sha256": _mask_sha256(validation.contiguous()),
            "validation_target_token_counts": list(
                policy.validation_target_token_counts
            ),
        },
    )
    return fit.contiguous(), validation.contiguous(), receipt


def _subset_rows(
    rows: LayerMLPFlowRows,
    mask: Tensor,
    all_weights: Tensor,
    *,
    subset_role: str,
    split_receipt_sha256: str,
) -> tuple[Tensor, Tensor, Tensor, str]:
    if mask.shape != (rows.row_count,) or mask.dtype != torch.bool:
        raise ValueError("subset mask does not align with flow rows")
    if all_weights.shape != (rows.row_count,) or all_weights.dtype != torch.float64:
        raise ValueError("row weights do not align with flow rows")
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("fit subset cannot be empty")
    x = rows.chart_states[indices].to(dtype=torch.float64).contiguous()
    v = rows.teacher_velocities[indices].to(dtype=torch.float64).contiguous()
    weights = all_weights[indices].clone().contiguous()
    weights /= weights.sum()
    source = _digest(
        _SUBSET_DOMAIN,
        {
            "chart_state_sha256": _tensor_sha256(x),
            "layer_ordinal": rows.layer_ordinal,
            "mask_sha256": _mask_sha256(mask.contiguous()),
            "mode_id": rows.mode_id,
            "row_count": int(indices.numel()),
            "row_indices_sha256": _tensor_sha256(indices.to(torch.int64)),
            "selected_row_keys_sha256": _digest(
                _SUBSET_DOMAIN,
                [list(rows.row_keys[index]) for index in indices.tolist()],
            ),
            "selected_source_capture_receipts": [
                [example, receipt]
                for example, receipt, (start, end) in zip(
                    rows.example_ids,
                    rows.source_capture_receipt_sha256s,
                    rows.example_row_ranges,
                    strict=True,
                )
                if bool(mask[start:end].any())
            ],
            "row_weights_sha256": _tensor_sha256(weights),
            "split_receipt_sha256": split_receipt_sha256,
            "subset_role": subset_role,
            "teacher_velocity_sha256": _tensor_sha256(v),
        },
    )
    return x, v, weights, source


def _canonicalize_basis_signs(basis: Tensor) -> Tensor:
    result = basis.clone()
    for column in range(result.shape[1]):
        pivot = int(result[:, column].abs().argmax())
        if float(result[pivot, column]) < 0.0:
            result[:, column].neg_()
    return result.contiguous()


def _weighted_median(values: Tensor, weights: Tensor) -> float:
    order = torch.argsort(values, stable=True)
    ordered_values = values[order]
    ordered_weights = weights[order]
    index = int(
        torch.searchsorted(
            ordered_weights.cumsum(0),
            torch.tensor(0.5, dtype=torch.float64),
            right=False,
        ).clamp_max(len(values) - 1)
    )
    return float(ordered_values[index])


def _fit_atlas_from_subset(
    *,
    rows: LayerMLPFlowRows,
    fit_x: Tensor,
    fit_weights: Tensor,
    fit_source_sha256: str,
    split_receipt_sha256: str,
    chart_count: int,
    tangent_rank: int,
    policy: LayerMultimodeFitPolicy,
) -> LayerMultimodeAtlasFit:
    hidden = int(fit_x.shape[1])
    weighted_mean = (fit_weights[:, None] * fit_x).sum(dim=0)
    if chart_count == 1:
        centers = weighted_mean[None, :]
        assignments = torch.zeros(len(fit_x), dtype=torch.long)
    else:
        distance_squared = (fit_x - weighted_mean).square().sum(dim=1)
        second_index = int((fit_weights * distance_squared).argmax())
        centers = torch.stack((weighted_mean, fit_x[second_index].clone()))
        for _ in range(policy.lloyd_passes):
            distances = (fit_x[:, None, :] - centers[None, :, :]).square().sum(
                dim=-1
            )
            assignments = distances.argmin(dim=1)
            updated: list[Tensor] = []
            for chart_ordinal in range(2):
                selected = assignments == chart_ordinal
                if not bool(selected.any()):
                    raise AtlasSupportError(
                        "deterministic C=2 Lloyd fit produced an empty chart"
                    )
                local_weights = fit_weights[selected]
                updated.append(
                    (
                        local_weights[:, None] * fit_x[selected]
                    ).sum(dim=0)
                    / local_weights.sum()
                )
            centers = torch.stack(updated)
        distances = (fit_x[:, None, :] - centers[None, :, :]).square().sum(
            dim=-1
        )
        assignments = distances.argmin(dim=1)
        # A hash order is a canonical label choice and has no geometric effect.
        order = sorted(range(2), key=lambda index: _tensor_sha256(centers[index]))
        if order != [0, 1]:
            centers = centers[torch.tensor(order)]
            assignments = 1 - assignments

    bases: list[Tensor] = []
    hard_counts: list[int] = []
    numerical_ranks: list[int] = []
    selected_minima: list[float] = []
    covariance_maxima: list[float] = []
    for chart_ordinal in range(chart_count):
        selected = assignments == chart_ordinal
        count = int(selected.sum())
        hard_counts.append(count)
        if count < tangent_rank + 1:
            raise AtlasSupportError(
                f"chart {chart_ordinal} has {count} rows, below R+1={tangent_rank + 1}"
            )
        local_weights = fit_weights[selected]
        local_weights = local_weights / local_weights.sum()
        centered = fit_x[selected] - centers[chart_ordinal]
        scaled = local_weights.sqrt()[:, None] * centered
        if count < hidden:
            gram = scaled @ scaled.T
            eigenvalues, left_eigenvectors = torch.linalg.eigh(gram)
            chosen_left = left_eigenvectors[:, -tangent_rank:].flip(dims=(1,))
            chosen_for_basis = eigenvalues[-tangent_rank:].flip(dims=(0,))
            eigenvectors_descending = (
                scaled.T @ chosen_left
            ) / chosen_for_basis.sqrt()[None, :]
        else:
            covariance = scaled.T @ scaled
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvectors_descending = eigenvectors[
                :, -tangent_rank:
            ].flip(dims=(1,))
        maximum = float(eigenvalues[-1])
        tolerance = max(
            policy.covariance_absolute_floor,
            policy.covariance_relative_tolerance * max(maximum, 0.0),
        )
        numerical_rank = int((eigenvalues > tolerance).sum())
        if numerical_rank < tangent_rank:
            raise AtlasSupportError(
                f"chart {chart_ordinal} covariance rank {numerical_rank} "
                f"is below R={tangent_rank}"
            )
        chosen_values = eigenvalues[-tangent_rank:]
        selected_minimum = float(chosen_values[0])
        if not math.isfinite(selected_minimum) or selected_minimum <= tolerance:
            raise AtlasSupportError("selected chart covariance support is nonpositive")
        basis = eigenvectors_descending
        bases.append(_canonicalize_basis_signs(basis))
        numerical_ranks.append(numerical_rank)
        selected_minima.append(selected_minimum)
        covariance_maxima.append(maximum)

    centers = centers.contiguous()
    basis_tensor = torch.stack(bases).contiguous()
    atlas = FixedLayerChartAtlas(
        centers=centers,
        bases=basis_tensor,
        source_fit_sha256=fit_source_sha256,
    )
    if chart_count == 1:
        initial_tau = lower_tau = upper_tau = 1.0
    else:
        centered = fit_x[:, None, :] - centers[None, :, :]
        z = torch.einsum("nch,chr->ncr", centered, basis_tensor)
        normal_squared = (
            centered.square().sum(dim=-1) - z.square().sum(dim=-1)
        ).clamp_min(0.0)
        difference = (normal_squared[:, 0] - normal_squared[:, 1]).abs()
        scale = float(
            (fit_weights * (fit_x - weighted_mean).square().sum(dim=1)).sum()
        )
        floor = torch.finfo(torch.float64).eps * max(1.0, scale)
        initial_tau = max(_weighted_median(difference, fit_weights), floor)
        lower_tau = initial_tau / policy.chart_temperature_bound_factor
        upper_tau = initial_tau * policy.chart_temperature_bound_factor

    receipt = LayerMultimodeAtlasFitReceipt(
        layer_ordinal=rows.layer_ordinal,
        source_rows_receipt_sha256=rows.receipt_sha256,
        split_receipt_sha256=split_receipt_sha256,
        policy_sha256=policy.policy_sha256,
        fit_source_sha256=fit_source_sha256,
        fit_row_count=len(fit_x),
        hidden_width=hidden,
        chart_count=chart_count,
        tangent_rank=tangent_rank,
        hard_chart_counts=tuple(hard_counts),
        covariance_numerical_ranks=tuple(numerical_ranks),
        covariance_selected_min_eigenvalues=tuple(selected_minima),
        covariance_max_eigenvalues=tuple(covariance_maxima),
        atlas_sha256=atlas.atlas_sha256,
        initial_chart_temperature=float(initial_tau),
        chart_temperature_lower_bound=float(lower_tau),
        chart_temperature_upper_bound=float(upper_tau),
    )
    fitted_atlas = LayerMultimodeAtlasFit(atlas=atlas, receipt=receipt)
    fitted_atlas.assert_integrity()
    object.__setattr__(
        fitted_atlas,
        "_construction_authority",
        _LIVE_ATLAS_FITTER_AUTHORITY,
    )
    _LIVE_ATLAS_FITS[id(fitted_atlas)] = fitted_atlas
    return fitted_atlas


def fit_layer_chart_atlas(
    *,
    rows: LayerMLPFlowRows,
    chart_count: int,
    tangent_rank: int,
    policy: LayerMultimodeFitPolicy = PRODUCTION_LAYER_MULTIMODE_FIT_POLICY,
    allow_test_policy: bool = False,
) -> LayerMultimodeAtlasFit:
    """Fit a cacheable chart atlas using only the frozen A fit partition."""

    _validate_fit_request(
        rows,
        chart_count=chart_count,
        modes_per_chart=1,
        tangent_rank=tangent_rank,
        policy=policy,
        allow_test_policy=allow_test_policy,
    )
    fit_mask, _, split_receipt = _development_masks(rows, policy)
    all_weights = family_example_balanced_row_weights(rows)
    fit_x, _, fit_weights, fit_source = _subset_rows(
        rows,
        fit_mask,
        all_weights,
        subset_role="A_full_fit_atlas_and_optimizer",
        split_receipt_sha256=split_receipt,
    )
    return _fit_atlas_from_subset(
        rows=rows,
        fit_x=fit_x,
        fit_weights=fit_weights,
        fit_source_sha256=fit_source,
        split_receipt_sha256=split_receipt,
        chart_count=chart_count,
        tangent_rank=tangent_rank,
        policy=policy,
    )


def _chart_q(x: Tensor, atlas: FixedLayerChartAtlas, tau: float) -> Tensor:
    centered = x[:, None, :] - atlas.centers[None, :, :]
    z = torch.einsum("nch,chr->ncr", centered, atlas.bases)
    normal_squared = (
        centered.square().sum(dim=-1) - z.square().sum(dim=-1)
    ).clamp_min(0.0)
    if atlas.chart_count == 1:
        return torch.ones(len(x), 1, dtype=torch.float64)
    return torch.softmax(-normal_squared / tau, dim=-1)


def _initialize_affine_generators(
    model: AffineLocalLayerMultimodeFlow,
    *,
    fit_x: Tensor,
    fit_v: Tensor,
    fit_weights: Tensor,
    policy: LayerMultimodeFitPolicy,
    seed: int,
) -> None:
    atlas = model.atlas
    centered = fit_x[:, None, :] - atlas.centers[None, :, :]
    z = torch.einsum("nch,chr->ncr", centered, atlas.bases)
    target_coordinates = torch.einsum("nh,chr->ncr", fit_v, atlas.bases)
    q = _chart_q(
        fit_x,
        atlas,
        model.config.chart_temperature,
    )
    base_weight = torch.empty_like(model.generator_weight[:, 0])
    base_bias = torch.empty_like(model.generator_bias[:, 0])
    for chart_ordinal in range(atlas.chart_count):
        design = torch.cat(
            (
                z[:, chart_ordinal],
                torch.ones(len(z), 1, dtype=torch.float64),
            ),
            dim=1,
        )
        local_weights = fit_weights * q[:, chart_ordinal]
        local_weights /= local_weights.sum()
        gram = design.T @ (local_weights[:, None] * design)
        trace_scale = float(torch.trace(gram)) / int(gram.shape[0])
        ridge = policy.ridge_relative_coefficient * max(trace_scale, 1.0e-12)
        regularized = gram + ridge * torch.eye(
            gram.shape[0],
            dtype=torch.float64,
        )
        right = design.T @ (
            local_weights[:, None] * target_coordinates[:, chart_ordinal]
        )
        coefficients = torch.linalg.solve(regularized, right)
        base_weight[chart_ordinal] = coefficients[:-1].T
        base_bias[chart_ordinal] = coefficients[-1]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    modes = model.config.modes_per_chart
    rank = atlas.tangent_rank
    with torch.no_grad():
        if modes == 1:
            model.router_weight.zero_()
            model.router_bias.zero_()
            model.generator_weight[:, 0].copy_(base_weight)
            model.generator_bias[:, 0].copy_(base_bias)
            return
        router = torch.randn(
            model.router_weight.shape,
            dtype=torch.float64,
            generator=generator,
        )
        router -= router.mean(dim=1, keepdim=True)
        model.router_weight.copy_(router * (0.01 / math.sqrt(rank)))
        model.router_bias.zero_()
        weight_noise = torch.randn(
            model.generator_weight.shape,
            dtype=torch.float64,
            generator=generator,
        )
        weight_noise -= weight_noise.mean(dim=1, keepdim=True)
        bias_noise = torch.randn(
            model.generator_bias.shape,
            dtype=torch.float64,
            generator=generator,
        )
        bias_noise -= bias_noise.mean(dim=1, keepdim=True)
        weight_scale = max(float(base_weight.square().mean().sqrt()), 1.0e-3)
        bias_scale = max(float(base_bias.square().mean().sqrt()), 1.0e-3)
        model.generator_weight.copy_(
            base_weight[:, None] + 1.0e-3 * weight_scale * weight_noise
        )
        model.generator_bias.copy_(
            base_bias[:, None] + 1.0e-3 * bias_scale * bias_noise
        )


def _loss_term_tuple(loss: object) -> tuple[tuple[str, float], ...]:
    return tuple(
        (name, float(getattr(loss, name).detach()))
        for name in ("distinctness", "fidelity", "load_balance", "sparsity", "total")
    )


def _flow_loss(
    model: AffineLocalLayerMultimodeFlow,
    *,
    x: Tensor,
    teacher: Tensor,
    weights: Tensor,
    source_sha256: str,
    policy: LayerMultimodeFitPolicy,
) -> tuple[LayerMultimodeFlowResult, object]:
    result = model(
        chart_inputs=x,
        residual_states=torch.zeros_like(x),
        source_capture_sha256=source_sha256,
    )
    loss = layer_multimode_flow_loss(
        result=result,
        teacher_velocity=teacher,
        row_weights=weights,
        fidelity_coefficient=policy.fidelity_coefficient,
        load_balance_coefficient=policy.load_balance_coefficient,
        sparsity_coefficient=policy.sparsity_coefficient,
        distinctness_coefficient=policy.distinctness_coefficient,
    )
    return result, loss


def _validation_family_ids(rows: LayerMLPFlowRows, mask: Tensor) -> tuple[str, ...]:
    selected = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    return tuple(rows.row_keys[index][0] for index in selected)


def _validation_evidence(
    *,
    rows: LayerMLPFlowRows,
    validation_mask: Tensor,
    validation_v: Tensor,
    validation_weights: Tensor,
    validation_source_sha256: str,
    result: LayerMultimodeFlowResult,
) -> LayerMultimodeValidationEvidence:
    prediction = result.predicted_velocity.detach()
    teacher = validation_v
    residual = prediction - teacher
    # Scale normalized family/example-balanced row weights by N.  This keeps
    # the pooled sufficient statistics additive and gives them the same scale
    # as ordinary scalar sums when rows happen to be uniformly weighted.
    sufficient_row_weights = len(teacher) * validation_weights
    sse = float((sufficient_row_weights * residual.square().sum(dim=-1)).sum())
    teacher_energy = float(
        (sufficient_row_weights * teacher.square().sum(dim=-1)).sum()
    )
    prediction_energy = float(
        (sufficient_row_weights * prediction.square().sum(dim=-1)).sum()
    )
    dot = float(
        (sufficient_row_weights * (prediction * teacher).sum(dim=-1)).sum()
    )
    if teacher_energy <= 0.0 or prediction_energy <= 0.0:
        raise ValueError("validation teacher/prediction energy must be positive")
    nrmse = math.sqrt(sse / teacher_energy)
    cosine = dot / math.sqrt(prediction_energy * teacher_energy)
    scalar_count = int(teacher.numel())
    teacher_rms = math.sqrt(teacher_energy / scalar_count)

    family_ids = _validation_family_ids(rows, validation_mask)
    family_stats: list[
        tuple[str, int, float, float, float, float, float, float]
    ] = []
    for family in sorted(set(family_ids)):
        family_mask = torch.tensor(
            [value == family for value in family_ids],
            dtype=torch.bool,
        )
        family_prediction = prediction[family_mask]
        family_teacher = teacher[family_mask]
        family_residual = family_prediction - family_teacher
        family_weights = sufficient_row_weights[family_mask]
        family_sse = float(
            (family_weights * family_residual.square().sum(dim=-1)).sum()
        )
        family_teacher_energy = float(
            (family_weights * family_teacher.square().sum(dim=-1)).sum()
        )
        family_prediction_energy = float(
            (family_weights * family_prediction.square().sum(dim=-1)).sum()
        )
        family_dot = float(
            (
                family_weights
                * (family_prediction * family_teacher).sum(dim=-1)
            ).sum()
        )
        if family_teacher_energy <= 0.0 or family_prediction_energy <= 0.0:
            raise ValueError("family validation energy must be positive")
        family_stats.append(
            (
                family,
                int(family_teacher.numel()),
                family_sse,
                family_teacher_energy,
                math.sqrt(family_sse / family_teacher_energy),
                family_prediction_energy,
                family_dot,
                family_dot
                / math.sqrt(family_prediction_energy * family_teacher_energy),
            )
        )

    # Canonicalize pooled values from the family receipts themselves.  This
    # makes the selector's stricter additive closure independent of reduction
    # order while preserving the exact same weighted sufficient statistics.
    sse = math.fsum(value[2] for value in family_stats)
    teacher_energy = math.fsum(value[3] for value in family_stats)
    prediction_energy = math.fsum(value[5] for value in family_stats)
    dot = math.fsum(value[6] for value in family_stats)
    nrmse = math.sqrt(sse / teacher_energy)
    cosine = dot / math.sqrt(prediction_energy * teacher_energy)
    teacher_rms = math.sqrt(teacher_energy / scalar_count)

    q = result.q.detach()
    pi = result.pi.detach()
    chart_masses_tensor = (validation_weights[:, None] * q).sum(dim=0)
    chart_masses = tuple(float(value) for value in chart_masses_tensor)
    if result.receipt.chart_count == 1:
        chart_masses = (1.0,)
    conditional_mass_tensor = (
        validation_weights[:, None, None] * q[:, :, None] * pi
    ).sum(dim=0) / chart_masses_tensor[:, None]
    conditional_masses = tuple(
        tuple(float(value) for value in row)
        for row in conditional_mass_tensor
    )
    modes = result.receipt.modes_per_chart
    if modes == 1:
        conditional_masses = ((1.0,),) * result.receipt.chart_count
        mean_entropy = (0.0,) * result.receipt.chart_count
        mutual_information = (0.0,) * result.receipt.chart_count
        separation_rms = (0.0,) * result.receipt.chart_count
        separation_ratio = (0.0,) * result.receipt.chart_count
    else:
        safe_pi = pi.clamp_min(torch.finfo(torch.float64).tiny)
        per_row_entropy = -(pi * safe_pi.log()).sum(dim=-1)
        mean_entropy_tensor = (
            validation_weights[:, None] * q * per_row_entropy
        ).sum(dim=0) / chart_masses_tensor
        mean_entropy = tuple(float(value) for value in mean_entropy_tensor)
        mass_entropy = tuple(
            -math.fsum(value * math.log(value) for value in row if value > 0.0)
            for row in conditional_masses
        )
        mutual_information = tuple(
            max(0.0, mass_entropy[index] - mean_entropy[index])
            for index in range(result.receipt.chart_count)
        )
        separation_values: list[float] = []
        coordinates = result.per_mode_coordinates.detach()
        hidden = result.receipt.hidden_width
        for chart_ordinal in range(result.receipt.chart_count):
            conditional_row_weights = (
                validation_weights * q[:, chart_ordinal]
                / chart_masses_tensor[chart_ordinal]
            )
            pair_values: list[float] = []
            for first in range(modes):
                for second in range(first + 1, modes):
                    delta = (
                        coordinates[:, chart_ordinal, first]
                        - coordinates[:, chart_ordinal, second]
                    )
                    pair_values.append(
                        math.sqrt(
                            float(
                                (
                                    conditional_row_weights
                                    * delta.square().sum(dim=-1)
                                ).sum()
                            )
                            / hidden
                        )
                    )
            separation_values.append(min(pair_values))
        separation_rms = tuple(separation_values)
        separation_ratio = tuple(value / teacher_rms for value in separation_rms)

    joint = q[:, :, None] * pi
    hard = joint.reshape(len(joint), -1).argmax(dim=1)
    counts = tuple(
        int(value)
        for value in torch.bincount(
            hard,
            minlength=result.receipt.chart_count * modes,
        )
    )
    family_stats_tuple = tuple(family_stats)
    return LayerMultimodeValidationEvidence(
        layer_ordinal=rows.layer_ordinal,
        chart_count=result.receipt.chart_count,
        modes_per_chart=modes,
        row_count=len(teacher),
        validation_source_sha256=validation_source_sha256,
        velocity_nrmse=float(nrmse),
        aggregate_velocity_cosine=float(cosine),
        velocity_sse=float(sse),
        validation_scalar_count=scalar_count,
        validation_prediction_energy=float(prediction_energy),
        validation_teacher_energy=float(teacher_energy),
        validation_prediction_teacher_dot=float(dot),
        teacher_velocity_rms=float(teacher_rms),
        family_velocity_nrmse=tuple(
            (entry[0], entry[4]) for entry in family_stats_tuple
        ),
        family_fidelity_sufficient_statistics=family_stats_tuple,
        chart_soft_masses=chart_masses,
        conditional_mode_masses=conditional_masses,
        per_chart_mean_row_router_entropy_nats=mean_entropy,
        per_chart_state_mode_mutual_information_nats=mutual_information,
        per_chart_minimum_pairwise_velocity_separation_rms=separation_rms,
        per_chart_minimum_pairwise_velocity_separation_ratio=separation_ratio,
        hard_joint_route_counts=counts,
    )


def _candidate_seed(
    policy: LayerMultimodeFitPolicy,
    *,
    layer_ordinal: int,
    chart_count: int,
    modes_per_chart: int,
    tangent_rank: int,
    partition_ordinal: int,
) -> int:
    seed = (
        policy.seed_base
        + layer_ordinal * 1_000_000
        + chart_count * 100_000
        + modes_per_chart * 10_000
        + tangent_rank * 10
        + partition_ordinal
    )
    if seed >= 2**63:
        raise ValueError("derived candidate seed exceeds signed int64")
    return seed


def _clip_multimode_parameter_gradients(
    model: AffineLocalLayerMultimodeFlow,
    trainable: list[Tensor],
    *,
    max_norm: float,
) -> tuple[Tensor, Tensor]:
    """Clip dynamics and chart-temperature gradients in independent groups.

    Chart-local regularizers are constructed with detached chart memberships,
    so their direct derivative with respect to ``log_chart_temperature`` is
    zero.  Keeping tau outside the global dynamics norm is also necessary:
    otherwise large router/generator regularizer gradients would indirectly
    rescale the fidelity-only tau update through the shared clip coefficient.
    """

    if type(model) is not AffineLocalLayerMultimodeFlow:
        raise TypeError("model must be an AffineLocalLayerMultimodeFlow")
    if type(trainable) is not list or any(
        not isinstance(parameter, Tensor) for parameter in trainable
    ):
        raise TypeError("trainable must be a list of tensors")
    _strict_float(max_norm, label="max_norm", positive=True)
    tau = model.log_chart_temperature
    non_tau = [parameter for parameter in trainable if parameter is not tau]
    if not non_tau:
        raise ValueError("non-tau trainable inventory must not be empty")
    non_tau_norm = torch.nn.utils.clip_grad_norm_(
        non_tau,
        max_norm=max_norm,
        error_if_nonfinite=True,
    )
    if any(parameter is tau for parameter in trainable):
        if not tau.requires_grad:
            raise ValueError("frozen chart temperature entered trainable inventory")
        tau_norm = torch.nn.utils.clip_grad_norm_(
            [tau],
            max_norm=max_norm,
            error_if_nonfinite=True,
        )
    else:
        if tau.requires_grad:
            raise ValueError("trainable chart temperature is missing from inventory")
        tau_norm = torch.zeros((), dtype=tau.dtype, device=tau.device)
    if not bool(torch.isfinite(non_tau_norm)) or not bool(
        torch.isfinite(tau_norm)
    ):
        raise RuntimeError("multimode optimizer produced a nonfinite gradient")
    return non_tau_norm, tau_norm


def _fit_candidate_from_masks(
    *,
    rows: LayerMLPFlowRows,
    chart_count: int,
    modes_per_chart: int,
    tangent_rank: int,
    policy: LayerMultimodeFitPolicy,
    allow_test_policy: bool,
    fit_mask: Tensor,
    validation_mask: Tensor,
    training_split_receipt_sha256: str,
    validation_split_receipt_sha256: str,
    partition_ordinal: int,
    atlas_fit: LayerMultimodeAtlasFit | None,
) -> LayerMultimodeCandidateFit:
    _validate_fit_request(
        rows,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        policy=policy,
        allow_test_policy=allow_test_policy,
    )
    all_weights = family_example_balanced_row_weights(rows)
    fit_x, fit_v, fit_weights, fit_source = _subset_rows(
        rows,
        fit_mask,
        all_weights,
        subset_role=(
            "A_full_fit_atlas_and_optimizer"
            if partition_ordinal == 0
            else f"A_split_refit_partition_{partition_ordinal}"
        ),
        split_receipt_sha256=training_split_receipt_sha256,
    )
    if atlas_fit is None:
        atlas_fit = _fit_atlas_from_subset(
            rows=rows,
            fit_x=fit_x,
            fit_weights=fit_weights,
            fit_source_sha256=fit_source,
            split_receipt_sha256=training_split_receipt_sha256,
            chart_count=chart_count,
            tangent_rank=tangent_rank,
            policy=policy,
        )
    else:
        if type(atlas_fit) is not LayerMultimodeAtlasFit:
            raise TypeError("atlas_fit must be a LayerMultimodeAtlasFit")
        atlas_fit.assert_integrity()
        if not atlas_fit.live_atlas_fitter_authorized:
            raise ValueError(
                "supplied atlas_fit must be the original live deterministic "
                "atlas-fitter result"
            )
        receipt = atlas_fit.receipt
        if (
            receipt.source_rows_receipt_sha256 != rows.receipt_sha256
            or receipt.split_receipt_sha256
            != training_split_receipt_sha256
            or receipt.policy_sha256 != policy.policy_sha256
            or receipt.fit_source_sha256 != fit_source
            or receipt.chart_count != chart_count
            or receipt.tangent_rank != tangent_rank
        ):
            raise ValueError("prefit atlas does not match this fit-only request")

    seed = _candidate_seed(
        policy,
        layer_ordinal=rows.layer_ordinal,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        partition_ordinal=partition_ordinal,
    )
    config = LayerMultimodeFlowConfig(
        layer_ordinal=rows.layer_ordinal,
        modes_per_chart=modes_per_chart,
        chart_temperature=atlas_fit.receipt.initial_chart_temperature,
        mode_temperature=1.0,
        source_mlp_intermediate_width=2048,
        retained_rmsnorm_vector_count=1,
        initialization_seed=seed,
        fit_chart_temperature=chart_count == 2,
    )
    model = AffineLocalLayerMultimodeFlow(
        atlas=atlas_fit.atlas,
        config=config,
    )
    _initialize_affine_generators(
        model,
        fit_x=fit_x,
        fit_v=fit_v,
        fit_weights=fit_weights,
        policy=policy,
        seed=seed,
    )
    with torch.no_grad():
        _, initial_loss = _flow_loss(
            model,
            x=fit_x,
            teacher=fit_v,
            weights=fit_weights,
            source_sha256=fit_source,
            policy=policy,
        )
    initial_terms = _loss_term_tuple(initial_loss)
    del initial_loss

    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=policy.learning_rate,
        betas=(policy.adam_beta1, policy.adam_beta2),
        eps=policy.adam_epsilon,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    row_count = len(fit_x)
    batch_size = min(policy.minibatch_size, row_count)
    permutation = torch.randperm(row_count, generator=generator)
    cursor = 0
    model.train()
    for step in range(policy.optimizer_steps):
        if cursor + batch_size <= row_count:
            batch = permutation[cursor : cursor + batch_size]
            cursor += batch_size
            if cursor == row_count:
                permutation = torch.randperm(row_count, generator=generator)
                cursor = 0
        else:
            tail = permutation[cursor:]
            permutation = torch.randperm(row_count, generator=generator)
            needed = batch_size - len(tail)
            batch = torch.cat((tail, permutation[:needed]))
            cursor = needed
        batch_source = _digest(
            _SUBSET_DOMAIN,
            {
                "batch_indices_sha256": _tensor_sha256(batch.to(torch.int64)),
                "fit_source_sha256": fit_source,
                "optimizer_step": step,
            },
        )
        _, loss = _flow_loss(
            model,
            x=fit_x[batch].contiguous(),
            teacher=fit_v[batch].contiguous(),
            weights=fit_weights[batch].contiguous(),
            source_sha256=batch_source,
            policy=policy,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        _clip_multimode_parameter_gradients(
            model,
            trainable,
            max_norm=policy.gradient_clip_norm,
        )
        optimizer.step()
        with torch.no_grad():
            if chart_count == 2:
                model.log_chart_temperature.clamp_(
                    min=math.log(
                        atlas_fit.receipt.chart_temperature_lower_bound
                    ),
                    max=math.log(
                        atlas_fit.receipt.chart_temperature_upper_bound
                    ),
                )
            else:
                model.log_chart_temperature.fill_(0.0)
            if modes_per_chart == 1:
                model.router_weight.zero_()
                model.router_bias.zero_()
            for parameter in model.parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError(
                        "multimode optimizer produced nonfinite parameters"
                    )

    model.eval()
    with torch.no_grad():
        _, final_loss = _flow_loss(
            model,
            x=fit_x,
            teacher=fit_v,
            weights=fit_weights,
            source_sha256=fit_source,
            policy=policy,
        )
    final_terms = _loss_term_tuple(final_loss)
    # This is the first read of validation tensors by the numerical fitter.
    validation_x, validation_v, validation_weights, validation_source = (
        _subset_rows(
            rows,
            validation_mask,
            all_weights,
            subset_role="A_frozen_validation_evaluation_only",
            split_receipt_sha256=validation_split_receipt_sha256,
        )
    )
    with torch.no_grad():
        validation_result = model(
            chart_inputs=validation_x,
            residual_states=torch.zeros_like(validation_x),
            source_capture_sha256=validation_source,
        )
    validation = _validation_evidence(
        rows=rows,
        validation_mask=validation_mask,
        validation_v=validation_v,
        validation_weights=validation_weights,
        validation_source_sha256=validation_source,
        result=validation_result,
    )
    resources = model.resource_accounting
    final_tau = float(model.log_chart_temperature.detach().exp())
    receipt = LayerMultimodeCandidateFitReceipt(
        layer_ordinal=rows.layer_ordinal,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        source_rows_receipt_sha256=rows.receipt_sha256,
        source_capture_scope=rows.source_capture_scope,
        split_receipt_sha256=training_split_receipt_sha256,
        policy_sha256=policy.policy_sha256,
        fit_policy_test_only=policy.test_only,
        production_fit_authorized=(
            not policy.test_only
            and policy.policy_sha256
            == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256
            and rows.source_capture_scope == PRODUCTION_FULL_A_CAPTURE_SCOPE
        ),
        fit_source_sha256=fit_source,
        validation_source_sha256=validation_source,
        initialization_seed=seed,
        fit_row_count=len(fit_x),
        validation_row_count=len(validation_x),
        optimizer_steps_completed=policy.optimizer_steps,
        atlas_fit=atlas_fit.receipt,
        model_parameter_sha256=model.model_parameter_sha256,
        resource_receipt_sha256=resources.receipt_sha256,
        candidate_total_parameter_count=resources.candidate_total_parameter_count,
        candidate_matrix_macs_per_token=(
            resources.candidate_matrix_and_mixture_macs_per_token
        ),
        retained_rmsnorm_vector_count=1,
        final_chart_temperature=final_tau,
        chart_temperature_fitted_on_A=chart_count == 2,
        mode_temperature=1.0,
        output_rmsnorm_folded_into_teacher_velocity=True,
        initial_fit_loss_terms=initial_terms,
        final_fit_loss_terms=final_terms,
        validation=validation,
    )
    fitted = LayerMultimodeCandidateFit(
        model=model,
        atlas_fit=atlas_fit,
        receipt=receipt,
    )
    fitted.assert_integrity()
    object.__setattr__(
        fitted,
        "_construction_authority",
        _LIVE_FITTER_AUTHORITY,
    )
    _LIVE_CANDIDATE_FITS[id(fitted)] = fitted
    return fitted


def fit_layer_multimode_candidate(
    *,
    rows: LayerMLPFlowRows,
    chart_count: int,
    modes_per_chart: int,
    tangent_rank: int,
    policy: LayerMultimodeFitPolicy = PRODUCTION_LAYER_MULTIMODE_FIT_POLICY,
    allow_test_policy: bool = False,
    atlas_fit: LayerMultimodeAtlasFit | None = None,
) -> LayerMultimodeCandidateFit:
    """Fit one candidate on A-fit and evaluate it once on A-validation."""

    _validate_fit_request(
        rows,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        policy=policy,
        allow_test_policy=allow_test_policy,
    )
    fit_mask, validation_mask, split_receipt = _development_masks(rows, policy)
    return _fit_candidate_from_masks(
        rows=rows,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        policy=policy,
        allow_test_policy=allow_test_policy,
        fit_mask=fit_mask,
        validation_mask=validation_mask,
        training_split_receipt_sha256=split_receipt,
        validation_split_receipt_sha256=split_receipt,
        partition_ordinal=0,
        atlas_fit=atlas_fit,
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMultimodeSplitRefitEvidence:
    """Scalar closure for the two deterministic checkerboard A-fit refits.

    The counts and mask digests are derived by the builder.  A strict
    production replayer must still regenerate the checkerboard masks from the
    frozen scalar A partition manifest; parser-only equality of hashes is not
    presented as proof of raw-mask disjointness or union.
    """

    layer_ordinal: int
    chart_count: int
    modes_per_chart: int
    tangent_rank: int
    source_rows_receipt_sha256: str
    policy_sha256: str
    base_split_receipt_sha256: str
    base_fit_mask_sha256: str
    partition_union_mask_sha256: str
    base_fit_row_count: int
    partition_row_counts: tuple[int, int]
    base_fit_example_count: int
    partition_example_counts: tuple[int, int]
    intersection_row_count: int
    union_equals_base_fit: bool
    partition_mask_sha256s: tuple[str, str]
    partition_example_catalog_sha256s: tuple[str, str]
    partition_fit_receipt_sha256s: tuple[str, str]
    partition_model_parameter_sha256s: tuple[str, str]
    validation_source_sha256: str
    matched_prediction_cosine: float
    hard_route_adjusted_rand_index: float
    split_id: str = _SPLIT_ID
    hard_route_id: str = _HARD_ROUTE_ID
    adjusted_rand_id: str = _ARI_ID
    evidence_sha256: str = ""
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for label, value in (
            ("layer_ordinal", self.layer_ordinal),
            ("chart_count", self.chart_count),
            ("modes_per_chart", self.modes_per_chart),
            ("tangent_rank", self.tangent_rank),
        ):
            _strict_int(
                value,
                label=label,
                minimum=0 if label == "layer_ordinal" else 1,
            )
        if self.layer_ordinal >= 18:
            raise ValueError("layer_ordinal must be in [0,17]")
        if (
            self.chart_count not in (1, 2)
            or self.modes_per_chart not in (1, 2, 4)
        ):
            raise ValueError("split-refit C/M is outside the frozen ladder")
        for label in (
            "source_rows_receipt_sha256",
            "policy_sha256",
            "base_split_receipt_sha256",
            "base_fit_mask_sha256",
            "partition_union_mask_sha256",
            "validation_source_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        for label in (
            "partition_mask_sha256s",
            "partition_example_catalog_sha256s",
            "partition_fit_receipt_sha256s",
            "partition_model_parameter_sha256s",
        ):
            value = getattr(self, label)
            if type(value) is not tuple or len(value) != 2:
                raise TypeError(f"{label} must be an exact pair")
            for digest in value:
                _require_sha256(digest, label=f"{label} item")
            if value[0] == value[1]:
                raise ValueError(f"{label} split halves must be distinct")
        _strict_int(
            self.base_fit_row_count,
            label="base_fit_row_count",
            minimum=1,
        )
        _strict_int(
            self.base_fit_example_count,
            label="base_fit_example_count",
            minimum=1,
        )
        for label, value in (
            ("partition_row_counts", self.partition_row_counts),
            ("partition_example_counts", self.partition_example_counts),
        ):
            if type(value) is not tuple or len(value) != 2:
                raise TypeError(f"{label} must be an exact pair")
            for count in value:
                _strict_int(count, label=f"{label} item", minimum=1)
        if sum(self.partition_row_counts) != self.base_fit_row_count:
            raise ValueError("partition row counts do not close over base fit")
        if sum(self.partition_example_counts) != self.base_fit_example_count:
            raise ValueError("partition example counts do not close over base fit")
        _strict_int(
            self.intersection_row_count,
            label="intersection_row_count",
            minimum=0,
        )
        if self.intersection_row_count != 0:
            raise ValueError("split-refit partition intersection must be empty")
        if type(self.union_equals_base_fit) is not bool:
            raise TypeError("union_equals_base_fit must be bool")
        if not self.union_equals_base_fit:
            raise ValueError("split-refit partition union must equal base fit")
        if self.partition_union_mask_sha256 != self.base_fit_mask_sha256:
            raise ValueError("partition union mask digest must equal base fit mask")
        if self.policy_sha256 == PRODUCTION_LAYER_MULTIMODE_FIT_POLICY_SHA256:
            if (
                self.base_fit_row_count != 10_548
                or self.partition_row_counts != (5_274, 5_274)
                or self.base_fit_example_count != 24
                or self.partition_example_counts != (12, 12)
            ):
                raise ValueError("production split-refit counts drifted")
        _strict_float(
            self.matched_prediction_cosine,
            label="matched_prediction_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        _strict_float(
            self.hard_route_adjusted_rand_index,
            label="hard_route_adjusted_rand_index",
            minimum=-1.0,
            maximum=1.0,
        )
        if (
            self.split_id != _SPLIT_ID
            or self.hard_route_id != _HARD_ROUTE_ID
            or self.adjusted_rand_id != _ARI_ID
        ):
            raise ValueError("split-refit algorithm IDs drifted")
        expected = _digest(_SPLIT_DOMAIN, self._payload())
        if self.evidence_sha256 == "":
            object.__setattr__(self, "evidence_sha256", expected)
        elif _require_sha256(
            self.evidence_sha256,
            label="evidence_sha256",
        ) != expected:
            raise ValueError("split-refit evidence hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "adjusted_rand_id": self.adjusted_rand_id,
            "base_fit_example_count": self.base_fit_example_count,
            "base_fit_mask_sha256": self.base_fit_mask_sha256,
            "base_fit_row_count": self.base_fit_row_count,
            "base_split_receipt_sha256": self.base_split_receipt_sha256,
            "chart_count": self.chart_count,
            "format_version": _FORMAT_VERSION,
            "hard_route_adjusted_rand_index": (
                self.hard_route_adjusted_rand_index
            ),
            "hard_route_id": self.hard_route_id,
            "intersection_row_count": self.intersection_row_count,
            "layer_ordinal": self.layer_ordinal,
            "matched_prediction_cosine": self.matched_prediction_cosine,
            "modes_per_chart": self.modes_per_chart,
            "partition_example_catalog_sha256s": list(
                self.partition_example_catalog_sha256s
            ),
            "partition_example_counts": list(self.partition_example_counts),
            "partition_fit_receipt_sha256s": list(
                self.partition_fit_receipt_sha256s
            ),
            "partition_mask_sha256s": list(self.partition_mask_sha256s),
            "partition_model_parameter_sha256s": list(
                self.partition_model_parameter_sha256s
            ),
            "partition_row_counts": list(self.partition_row_counts),
            "partition_union_mask_sha256": self.partition_union_mask_sha256,
            "policy_sha256": self.policy_sha256,
            "schema": _SCHEMA,
            "source_rows_receipt_sha256": self.source_rows_receipt_sha256,
            "split_id": self.split_id,
            "tangent_rank": self.tangent_rank,
            "union_equals_base_fit": self.union_equals_base_fit,
            "validation_source_sha256": self.validation_source_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "evidence_sha256": self.evidence_sha256,
            **_safety_payload(),
        }

    @property
    def live_split_evaluator_authorized(self) -> bool:
        """True only for live M>1 evidence returned by the split evaluator."""

        return (
            self.modes_per_chart > 1
            and self._construction_authority
            is _LIVE_SPLIT_EVALUATOR_AUTHORITY
            and _LIVE_SPLIT_EVIDENCE.get(id(self)) is self
        )

    def assert_integrity(self) -> None:
        if _digest(_SPLIT_DOMAIN, self._payload()) != self.evidence_sha256:
            raise ValueError("split-refit evidence hash mismatch")

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerMultimodeSplitRefitEvidence:
        expected = {
            *cls._payload_field_names(),
            "evidence_sha256",
            *_safety_payload().keys(),
        }
        _strict_fields(state, expected=expected, label="split-refit evidence")
        _validate_common_state(state)
        pair_fields: dict[str, tuple[str, str]] = {}
        for label in (
            "partition_mask_sha256s",
            "partition_example_catalog_sha256s",
            "partition_fit_receipt_sha256s",
            "partition_model_parameter_sha256s",
        ):
            value = state[label]
            if type(value) is not list or len(value) != 2:
                raise TypeError(f"serialized {label} must be a list pair")
            pair_fields[label] = (value[0], value[1])
        count_pair_fields: dict[str, tuple[int, int]] = {}
        for label in ("partition_row_counts", "partition_example_counts"):
            value = state[label]
            if type(value) is not list or len(value) != 2:
                raise TypeError(f"serialized {label} must be a list pair")
            count_pair_fields[label] = (value[0], value[1])
        return cls(
            layer_ordinal=state["layer_ordinal"],
            chart_count=state["chart_count"],
            modes_per_chart=state["modes_per_chart"],
            tangent_rank=state["tangent_rank"],
            source_rows_receipt_sha256=state[
                "source_rows_receipt_sha256"
            ],
            policy_sha256=state["policy_sha256"],
            base_split_receipt_sha256=state[
                "base_split_receipt_sha256"
            ],
            base_fit_mask_sha256=state["base_fit_mask_sha256"],
            partition_union_mask_sha256=state[
                "partition_union_mask_sha256"
            ],
            base_fit_row_count=state["base_fit_row_count"],
            partition_row_counts=count_pair_fields["partition_row_counts"],
            base_fit_example_count=state["base_fit_example_count"],
            partition_example_counts=count_pair_fields[
                "partition_example_counts"
            ],
            intersection_row_count=state["intersection_row_count"],
            union_equals_base_fit=state["union_equals_base_fit"],
            partition_mask_sha256s=pair_fields[
                "partition_mask_sha256s"
            ],
            partition_example_catalog_sha256s=pair_fields[
                "partition_example_catalog_sha256s"
            ],
            partition_fit_receipt_sha256s=pair_fields[
                "partition_fit_receipt_sha256s"
            ],
            partition_model_parameter_sha256s=pair_fields[
                "partition_model_parameter_sha256s"
            ],
            validation_source_sha256=state["validation_source_sha256"],
            matched_prediction_cosine=state["matched_prediction_cosine"],
            hard_route_adjusted_rand_index=state[
                "hard_route_adjusted_rand_index"
            ],
            split_id=state["split_id"],
            hard_route_id=state["hard_route_id"],
            adjusted_rand_id=state["adjusted_rand_id"],
            evidence_sha256=state["evidence_sha256"],
        )

    @staticmethod
    def _payload_field_names() -> tuple[str, ...]:
        return (
            "adjusted_rand_id",
            "base_fit_example_count",
            "base_fit_mask_sha256",
            "base_fit_row_count",
            "base_split_receipt_sha256",
            "chart_count",
            "format_version",
            "hard_route_adjusted_rand_index",
            "hard_route_id",
            "intersection_row_count",
            "layer_ordinal",
            "matched_prediction_cosine",
            "modes_per_chart",
            "partition_example_catalog_sha256s",
            "partition_example_counts",
            "partition_fit_receipt_sha256s",
            "partition_mask_sha256s",
            "partition_model_parameter_sha256s",
            "partition_row_counts",
            "partition_union_mask_sha256",
            "policy_sha256",
            "schema",
            "source_rows_receipt_sha256",
            "split_id",
            "tangent_rank",
            "union_equals_base_fit",
            "validation_source_sha256",
        )


def _checkerboard_fit_partitions(
    rows: LayerMLPFlowRows,
    fit_mask: Tensor,
    policy: LayerMultimodeFitPolicy,
    base_split_receipt_sha256: str,
) -> tuple[tuple[Tensor, str, str], tuple[Tensor, str, str]]:
    families = tuple(sorted(set(rows.family_ids)))
    family_ordinals = {value: index for index, value in enumerate(families)}
    lengths = tuple(sorted(policy.fit_target_token_counts))
    length_ordinals = {value: index for index, value in enumerate(lengths)}
    masks = (
        torch.zeros(rows.row_count, dtype=torch.bool),
        torch.zeros(rows.row_count, dtype=torch.bool),
    )
    examples: tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]] = (
        [],
        [],
    )
    for family, example, length, (start, end) in zip(
        rows.family_ids,
        rows.example_ids,
        rows.token_counts,
        rows.example_row_ranges,
        strict=True,
    ):
        if length not in length_ordinals:
            continue
        partition = (family_ordinals[family] + length_ordinals[length]) % 2
        masks[partition][start:end] = True
        examples[partition].append((family, example, length))
    if (
        bool((masks[0] & masks[1]).any())
        or not torch.equal(masks[0] | masks[1], fit_mask)
        or not bool(masks[0].any())
        or not bool(masks[1].any())
    ):
        raise ValueError("checkerboard fit-only partitions do not close")
    result: list[tuple[Tensor, str, str]] = []
    for ordinal in range(2):
        mask = masks[ordinal].contiguous()
        example_digest = _digest(
            _SPLIT_DOMAIN,
            {
                "examples": [list(value) for value in examples[ordinal]],
                "partition_ordinal": ordinal + 1,
            },
        )
        split_receipt = _digest(
            _SPLIT_DOMAIN,
            {
                "base_split_receipt_sha256": base_split_receipt_sha256,
                "checkerboard_formula": (
                    "(sorted_family_ordinal+sorted_fit_length_ordinal)%2"
                ),
                "example_catalog_sha256": example_digest,
                "mask_sha256": _mask_sha256(mask),
                "partition_ordinal": ordinal + 1,
                "split_id": _SPLIT_ID,
            },
        )
        result.append((mask, split_receipt, example_digest))
    return result[0], result[1]


def _hard_joint_routes(result: LayerMultimodeFlowResult) -> Tensor:
    return result.joint_routing.reshape(result.receipt.row_count, -1).argmax(
        dim=1
    )


def _whole_example_count(rows: LayerMLPFlowRows, mask: Tensor) -> int:
    if mask.dtype != torch.bool or mask.shape != (rows.row_count,):
        raise ValueError("example-count mask must be aligned bool rows")
    count = 0
    for start, end in rows.example_row_ranges:
        selected = mask[start:end]
        if bool(selected.any()) and not bool(selected.all()):
            raise ValueError("split-refit mask selects a partial example")
        count += int(bool(selected.all()))
    return count


def _adjusted_rand_index(first: Tensor, second: Tensor) -> float:
    if (
        first.dtype != torch.long
        or second.dtype != torch.long
        or first.ndim != 1
        or first.shape != second.shape
        or len(first) == 0
    ):
        raise ValueError("ARI labels must be aligned nonempty int64 vectors")
    _, first_inverse = torch.unique(first, sorted=True, return_inverse=True)
    _, second_inverse = torch.unique(second, sorted=True, return_inverse=True)
    second_count = int(second_inverse.max()) + 1
    joint_index = first_inverse * second_count + second_inverse
    contingency = torch.bincount(joint_index).to(torch.float64)
    first_counts = torch.bincount(first_inverse).to(torch.float64)
    second_counts = torch.bincount(second_inverse).to(torch.float64)

    def combinations_two(values: Tensor) -> float:
        return float((values * (values - 1.0) / 2.0).sum())

    cells = combinations_two(contingency)
    rows = combinations_two(first_counts)
    columns = combinations_two(second_counts)
    total_pairs = len(first) * (len(first) - 1.0) / 2.0
    if total_pairs == 0.0:
        return 1.0
    expected = rows * columns / total_pairs
    maximum = 0.5 * (rows + columns)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0
    return max(-1.0, min(1.0, (cells - expected) / denominator))


def evaluate_layer_multimode_split_refit(
    *,
    rows: LayerMLPFlowRows,
    chart_count: int,
    modes_per_chart: int,
    tangent_rank: int,
    policy: LayerMultimodeFitPolicy = PRODUCTION_LAYER_MULTIMODE_FIT_POLICY,
    allow_test_policy: bool = False,
) -> LayerMultimodeSplitRefitEvidence:
    """Fit checkerboard A halves and compare untouched validation behavior.

    Hard route labels encode ``chart * M + mode``.  The Hubert-Arabie
    contingency ARI is invariant to both chart-label and within-chart
    mode-label permutations, so no potentially ambiguous manual matching is
    performed.
    """

    _validate_fit_request(
        rows,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        policy=policy,
        allow_test_policy=allow_test_policy,
    )
    fit_mask, validation_mask, base_split = _development_masks(rows, policy)
    partitions = _checkerboard_fit_partitions(
        rows,
        fit_mask,
        policy,
        base_split,
    )
    fits: list[LayerMultimodeCandidateFit] = []
    for partition_ordinal, (mask, split_receipt, _) in enumerate(
        partitions,
        start=1,
    ):
        fits.append(
            _fit_candidate_from_masks(
                rows=rows,
                chart_count=chart_count,
                modes_per_chart=modes_per_chart,
                tangent_rank=tangent_rank,
                policy=policy,
                allow_test_policy=allow_test_policy,
                fit_mask=mask,
                validation_mask=validation_mask,
                training_split_receipt_sha256=split_receipt,
                validation_split_receipt_sha256=base_split,
                partition_ordinal=partition_ordinal,
                atlas_fit=None,
            )
        )
    all_weights = family_example_balanced_row_weights(rows)
    validation_x, _, _, validation_source = _subset_rows(
        rows,
        validation_mask,
        all_weights,
        subset_role="A_frozen_validation_evaluation_only",
        split_receipt_sha256=base_split,
    )
    results: list[LayerMultimodeFlowResult] = []
    with torch.no_grad():
        for fit in fits:
            results.append(
                fit.model(
                    chart_inputs=validation_x,
                    residual_states=torch.zeros_like(validation_x),
                    source_capture_sha256=validation_source,
                )
            )
    first_prediction = results[0].predicted_velocity
    second_prediction = results[1].predicted_velocity
    denominator = float(first_prediction.square().sum()) * float(
        second_prediction.square().sum()
    )
    if denominator <= 0.0:
        raise ValueError("split-refit predictions need positive energy")
    cosine = float((first_prediction * second_prediction).sum()) / math.sqrt(
        denominator
    )
    ari = _adjusted_rand_index(
        _hard_joint_routes(results[0]),
        _hard_joint_routes(results[1]),
    )
    first_mask = partitions[0][0]
    second_mask = partitions[1][0]
    union_mask = (first_mask | second_mask).contiguous()
    intersection_row_count = int((first_mask & second_mask).sum())
    union_equals_base_fit = torch.equal(union_mask, fit_mask)
    evidence = LayerMultimodeSplitRefitEvidence(
        layer_ordinal=rows.layer_ordinal,
        chart_count=chart_count,
        modes_per_chart=modes_per_chart,
        tangent_rank=tangent_rank,
        source_rows_receipt_sha256=rows.receipt_sha256,
        policy_sha256=policy.policy_sha256,
        base_split_receipt_sha256=base_split,
        base_fit_mask_sha256=_mask_sha256(fit_mask.contiguous()),
        partition_union_mask_sha256=_mask_sha256(union_mask),
        base_fit_row_count=int(fit_mask.sum()),
        partition_row_counts=(
            int(first_mask.sum()),
            int(second_mask.sum()),
        ),
        base_fit_example_count=_whole_example_count(rows, fit_mask),
        partition_example_counts=(
            _whole_example_count(rows, first_mask),
            _whole_example_count(rows, second_mask),
        ),
        intersection_row_count=intersection_row_count,
        union_equals_base_fit=union_equals_base_fit,
        partition_mask_sha256s=(
            _mask_sha256(first_mask),
            _mask_sha256(second_mask),
        ),
        partition_example_catalog_sha256s=(
            partitions[0][2],
            partitions[1][2],
        ),
        partition_fit_receipt_sha256s=(
            fits[0].receipt.receipt_sha256,
            fits[1].receipt.receipt_sha256,
        ),
        partition_model_parameter_sha256s=(
            fits[0].receipt.model_parameter_sha256,
            fits[1].receipt.model_parameter_sha256,
        ),
        validation_source_sha256=validation_source,
        matched_prediction_cosine=max(-1.0, min(1.0, cosine)),
        hard_route_adjusted_rand_index=ari,
    )
    evidence.assert_integrity()
    if modes_per_chart > 1:
        object.__setattr__(
            evidence,
            "_construction_authority",
            _LIVE_SPLIT_EVALUATOR_AUTHORITY,
        )
        _LIVE_SPLIT_EVIDENCE[id(evidence)] = evidence
    return evidence


def layer_multimode_candidate_evaluation_from_fit(
    candidate_fit: LayerMultimodeCandidateFit,
    *,
    source_model_sha256: str,
    fit_a_partition_sha256: str,
    validation_a_partition_sha256: str,
    evaluation_protocol_sha256: str,
    split_refit: LayerMultimodeSplitRefitEvidence | None = None,
    allow_test_only_nonlive_authority: bool = False,
) -> LayerMultimodeCandidateEvaluationReceipt:
    """Strictly lower fitter receipts into the selector's scalar ABI.

    ``M=1`` has no mode-specialization claim and therefore requires no
    split-refit input; the selector's exact ``1.0/1.0`` sentinels are emitted.
    Every ``M>1`` candidate must supply independently authenticated,
    dimension/source/policy-matched checkerboard split-refit evidence.  The
    two A partition digests identify the shared example/row masks across all
    eighteen layers; layer-specific tensor lineage remains in the fit receipt.
    Live fitter/evaluator authority is mandatory by default.  The explicit
    nonlive opt-in is restricted to test-policy receipts and can never lower a
    production-authorized fit.
    """

    if type(candidate_fit) is not LayerMultimodeCandidateFit:
        raise TypeError("candidate_fit must be a LayerMultimodeCandidateFit")
    if type(allow_test_only_nonlive_authority) is not bool:
        raise TypeError("allow_test_only_nonlive_authority must be bool")
    candidate_fit.assert_integrity()
    _require_sha256(source_model_sha256, label="source_model_sha256")
    _require_sha256(
        fit_a_partition_sha256,
        label="fit_a_partition_sha256",
    )
    _require_sha256(
        validation_a_partition_sha256,
        label="validation_a_partition_sha256",
    )
    if fit_a_partition_sha256 == validation_a_partition_sha256:
        raise ValueError("fit and validation A partitions must be distinct")
    _require_sha256(
        evaluation_protocol_sha256,
        label="evaluation_protocol_sha256",
    )
    fit_receipt = candidate_fit.receipt
    if not candidate_fit.live_fitter_authorized:
        if fit_receipt.production_fit_authorized:
            raise ValueError(
                "production evaluation requires original live fitter authority"
            )
        if not allow_test_only_nonlive_authority or not fit_receipt.fit_policy_test_only:
            raise ValueError(
                "nonlive test fit requires explicit test-only authority opt-in"
            )
    validation = fit_receipt.validation
    if fit_receipt.modes_per_chart == 1:
        if split_refit is not None:
            raise ValueError("M=1 requires no split-refit evidence")
        split_cosine = 1.0
        split_ari = 1.0
        split_sha256: str | None = None
    else:
        if type(split_refit) is not LayerMultimodeSplitRefitEvidence:
            raise ValueError("M>1 requires split-refit evidence")
        if not split_refit.live_split_evaluator_authorized:
            if fit_receipt.production_fit_authorized:
                raise ValueError(
                    "production M>1 evaluation requires original live split-"
                    "evaluator authority"
                )
            if not allow_test_only_nonlive_authority:
                raise ValueError(
                    "nonlive test split evidence requires explicit test-only "
                    "authority opt-in"
                )
        expected_dimensions = (
            fit_receipt.layer_ordinal,
            fit_receipt.chart_count,
            fit_receipt.modes_per_chart,
            fit_receipt.tangent_rank,
        )
        actual_dimensions = (
            split_refit.layer_ordinal,
            split_refit.chart_count,
            split_refit.modes_per_chart,
            split_refit.tangent_rank,
        )
        if actual_dimensions != expected_dimensions:
            raise ValueError("split-refit candidate dimensions do not match fit")
        if (
            split_refit.source_rows_receipt_sha256
            != fit_receipt.source_rows_receipt_sha256
            or split_refit.policy_sha256 != fit_receipt.policy_sha256
            or split_refit.base_split_receipt_sha256
            != fit_receipt.split_receipt_sha256
            or split_refit.validation_source_sha256
            != fit_receipt.validation_source_sha256
        ):
            raise ValueError("split-refit sources/policy do not match fit")
        # Reconstructing the dataclass through its strict parser catches a
        # coordinated field edit that forgot any derived closure.
        LayerMultimodeSplitRefitEvidence.from_state_dict(
            split_refit.state_dict()
        )
        split_cosine = split_refit.matched_prediction_cosine
        split_ari = split_refit.hard_route_adjusted_rand_index
        split_sha256 = split_refit.evidence_sha256

    expected_parameters = candidate_serialized_parameter_count(
        fit_receipt.chart_count,
        fit_receipt.modes_per_chart,
        fit_receipt.tangent_rank,
    )
    expected_macs = candidate_matrix_macs_per_token(
        fit_receipt.chart_count,
        fit_receipt.modes_per_chart,
        fit_receipt.tangent_rank,
    )
    if (
        fit_receipt.candidate_total_parameter_count != expected_parameters
        or fit_receipt.candidate_matrix_macs_per_token != expected_macs
    ):
        raise ValueError("fitter resource receipt does not match selector ABI")

    family_fidelity = tuple(
        LayerMultimodeFamilyFidelityReceipt(
            family_id=family,
            scalar_count=scalar_count,
            prediction_sse=prediction_sse,
            teacher_energy=teacher_energy,
            velocity_nrmse=velocity_nrmse,
            prediction_energy=prediction_energy,
            prediction_teacher_dot=prediction_teacher_dot,
            velocity_cosine=velocity_cosine,
        )
        for (
            family,
            scalar_count,
            prediction_sse,
            teacher_energy,
            velocity_nrmse,
            prediction_energy,
            prediction_teacher_dot,
            velocity_cosine,
        ) in validation.family_fidelity_sufficient_statistics
    )
    if len(family_fidelity) != 6:
        raise ValueError("selector adapter requires the frozen six A families")

    training_receipt_sha256 = _digest(
        _SELECTION_ADAPTER_DOMAIN,
        {
            "candidate_fit_receipt_sha256": fit_receipt.receipt_sha256,
            "fit_a_partition_sha256": fit_a_partition_sha256,
            "validation_a_partition_sha256": validation_a_partition_sha256,
            "fit_policy_sha256": fit_receipt.policy_sha256,
            "fit_policy_test_only": fit_receipt.fit_policy_test_only,
            "model_parameter_sha256": fit_receipt.model_parameter_sha256,
            "output_rmsnorm_folded_into_teacher_velocity": (
                fit_receipt.output_rmsnorm_folded_into_teacher_velocity
            ),
            "retained_rmsnorm_vector_count": (
                fit_receipt.retained_rmsnorm_vector_count
            ),
            "source_capture_scope": fit_receipt.source_capture_scope,
            "production_fit_authorized": (
                fit_receipt.production_fit_authorized
            ),
            "split_refit_evidence_sha256": split_sha256,
            "validation_weighting_id": (
                validation.sufficient_statistics_weighting_id
            ),
        },
    )
    resource_pass = (
        expected_parameters < NATIVE_MLP_MATRIX_PARAMETER_COUNT
        and expected_macs < NATIVE_MLP_MATRIX_MACS_PER_TOKEN
    )
    evaluation = LayerMultimodeCandidateEvaluationReceipt(
        layer_ordinal=fit_receipt.layer_ordinal,
        chart_count=fit_receipt.chart_count,
        dynamic_mode_count=fit_receipt.modes_per_chart,
        tangent_rank=fit_receipt.tangent_rank,
        chart_temperature=fit_receipt.final_chart_temperature,
        chart_temperature_fitted_on_a=(
            fit_receipt.chart_temperature_fitted_on_A
        ),
        source_model_sha256=source_model_sha256,
        source_rows_receipt_sha256=(
            fit_receipt.source_rows_receipt_sha256
        ),
        fit_policy_sha256=fit_receipt.policy_sha256,
        fit_policy_test_only=fit_receipt.fit_policy_test_only,
        source_capture_scope=fit_receipt.source_capture_scope,
        production_fit_authorized=fit_receipt.production_fit_authorized,
        fit_source_sha256=fit_receipt.fit_source_sha256,
        validation_source_sha256=fit_receipt.validation_source_sha256,
        atlas_sha256=fit_receipt.atlas_fit.atlas_sha256,
        fit_a_sha256=fit_a_partition_sha256,
        validation_a_sha256=validation_a_partition_sha256,
        training_receipt_sha256=training_receipt_sha256,
        evaluation_protocol_sha256=evaluation_protocol_sha256,
        resource_receipt_sha256=fit_receipt.resource_receipt_sha256,
        serialized_parameter_count=expected_parameters,
        matrix_macs_per_token=expected_macs,
        validation_scalar_count=validation.validation_scalar_count,
        validation_prediction_sse=validation.velocity_sse,
        validation_teacher_energy=validation.validation_teacher_energy,
        validation_velocity_nrmse=validation.velocity_nrmse,
        validation_prediction_energy=validation.validation_prediction_energy,
        validation_prediction_teacher_dot=(
            validation.validation_prediction_teacher_dot
        ),
        aggregate_velocity_cosine=validation.aggregate_velocity_cosine,
        family_fidelity=family_fidelity,
        chart_soft_masses=validation.chart_soft_masses,
        conditional_mode_masses=validation.conditional_mode_masses,
        per_chart_mean_row_router_entropy_nats=(
            validation.per_chart_mean_row_router_entropy_nats
        ),
        per_chart_state_mode_mutual_information_nats=(
            validation.per_chart_state_mode_mutual_information_nats
        ),
        per_chart_minimum_pairwise_velocity_separation_rms=(
            validation.per_chart_minimum_pairwise_velocity_separation_rms
        ),
        per_chart_minimum_pairwise_velocity_separation_ratio=(
            validation.per_chart_minimum_pairwise_velocity_separation_ratio
        ),
        minimum_chart_soft_mass=min(validation.chart_soft_masses),
        minimum_conditional_mode_mass=min(
            min(values) for values in validation.conditional_mode_masses
        ),
        minimum_per_chart_state_mode_mutual_information_nats=min(
            validation.per_chart_state_mode_mutual_information_nats
        ),
        minimum_per_chart_velocity_separation_ratio=min(
            validation.per_chart_minimum_pairwise_velocity_separation_ratio
        ),
        split_refit_matched_prediction_cosine=split_cosine,
        split_refit_hard_route_ari=split_ari,
        resource_pass=resource_pass,
    )
    return _grant_live_candidate_evaluation_authority(evaluation)
