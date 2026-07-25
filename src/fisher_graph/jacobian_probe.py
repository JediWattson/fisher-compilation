"""Bounded forward-JVP diagnostics for causal modal graph edges.

The full token-by-token Jacobian of a transformer block is too large to
materialize at realistic sequence lengths and residual widths.  This module
provides a deliberately bounded reference path: it excites a small number of
caller-supplied codec directions at one valid input position at a time,
projects the resulting block JVP into output codec coordinates, and pools
signed mean/RMS edge statistics by exact logical lag.

The result is a forward Jacobian diagnostic, not the reverse score-gradient
regression implemented by :mod:`fisher_graph.streaming_causal_transport`.
It never treats elementwise RMS values as an executable signed operator.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .adapters import LayerBlockBoundaryPlan, ModelAdapter, SegmentSpec
from .compiler.calibration import CalibrationBatch
from .linear_codec import LinearActivationCodec


_FORMAT_VERSION = 2
_ALGORITHM = "exact_forward_jvp_logical_lag_pool"
_ALGORITHM_VERSION = 1


def _finite_nonnegative(value: float, *, label: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative float")


@dataclass(frozen=True, slots=True)
class CausalLagJacobianStatistics:
    """Signed and RMS modal-Jacobian edges pooled by exact logical lag.

    ``mean`` and ``rms`` have axes
    ``[lag, output_mode, input_mode]``.  A lag of zero is a same-position
    edge; positive lag means the input precedes the output.  Future-to-prefix
    responses are never folded into lag zero and are reported separately as
    causal leakage energy.
    """

    input_activation: str
    output_activation: str
    input_modes: int
    output_modes: int
    max_lag: int
    sequences: int
    jvp_calls: int
    lag_pair_counts: tuple[int, ...]
    mean: Tensor
    rms: Tensor
    captured_squared_sum: float
    omitted_past_squared_sum: float
    causal_leakage_squared_sum: float
    total_squared_sum: float
    algorithm: str = _ALGORITHM
    algorithm_version: int = _ALGORITHM_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("input_activation", self.input_activation),
            ("output_activation", self.output_activation),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a nonempty string")
        if self.input_activation == self.output_activation:
            raise ValueError("Jacobian endpoints must be distinct")
        for label, value in (
            ("input_modes", self.input_modes),
            ("output_modes", self.output_modes),
            ("sequences", self.sequences),
            ("jvp_calls", self.jvp_calls),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if type(self.max_lag) is not int or self.max_lag < 0:
            raise ValueError("max_lag must be nonnegative")
        if (
            not isinstance(self.lag_pair_counts, tuple)
            or len(self.lag_pair_counts) != self.max_lag + 1
            or any(type(count) is not int or count < 0 for count in self.lag_pair_counts)
        ):
            raise ValueError("lag_pair_counts are invalid")
        expected = (
            self.max_lag + 1,
            self.output_modes,
            self.input_modes,
        )
        for label, tensor in (("mean", self.mean), ("rms", self.rms)):
            if not isinstance(tensor, Tensor) or tensor.shape != expected:
                raise ValueError(f"{label} must have shape {list(expected)}")
            if tensor.device.type != "cpu" or tensor.dtype != torch.float64:
                raise ValueError(f"{label} must be a CPU float64 tensor")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{label} must be finite")
        if (self.rms < 0).any():
            raise ValueError("rms cannot be negative")
        moment_scale = max(
            float(self.mean.abs().max().item()),
            float(self.rms.abs().max().item()),
            1.0,
        )
        tolerance = (
            512
            * torch.finfo(torch.float64).eps
            * moment_scale
            * max(self.jvp_calls, 1)
        )
        if (self.mean.abs() > self.rms + tolerance).any():
            raise ValueError("absolute mean cannot exceed RMS")
        for label, value in (
            ("captured_squared_sum", self.captured_squared_sum),
            ("omitted_past_squared_sum", self.omitted_past_squared_sum),
            ("causal_leakage_squared_sum", self.causal_leakage_squared_sum),
            ("total_squared_sum", self.total_squared_sum),
        ):
            _finite_nonnegative(value, label=label)
        accounted = (
            self.captured_squared_sum
            + self.omitted_past_squared_sum
            + self.causal_leakage_squared_sum
        )
        accounting_tolerance = (
            4096
            * torch.finfo(torch.float64).eps
            * max(self.total_squared_sum, 1.0)
            * max(self.jvp_calls * self.output_modes, 1)
        )
        if abs(accounted - self.total_squared_sum) > accounting_tolerance:
            raise ValueError("Jacobian energy accounting is inconsistent")
        captured_from_moments = float(
            (
                self.rms.square()
                * torch.tensor(
                    self.lag_pair_counts,
                    dtype=torch.float64,
                )[:, None, None]
            )
            .sum()
            .item()
        )
        if (
            abs(captured_from_moments - self.captured_squared_sum)
            > accounting_tolerance
        ):
            raise ValueError(
                "Jacobian lag moments do not match captured energy"
            )
        if self.algorithm != _ALGORITHM:
            raise ValueError("unsupported Jacobian probe algorithm")
        if self.algorithm_version != _ALGORITHM_VERSION:
            raise ValueError("unsupported Jacobian probe algorithm version")

    @property
    def captured_energy_fraction(self) -> float:
        if self.total_squared_sum == 0:
            return 1.0
        return self.captured_squared_sum / self.total_squared_sum

    @property
    def causal_leakage_fraction(self) -> float:
        if self.total_squared_sum == 0:
            return 0.0
        return self.causal_leakage_squared_sum / self.total_squared_sum

    @property
    def stationary_mean_squared_sum(self) -> float:
        """Energy of the best constant signed edge at each logical lag."""

        counts = torch.tensor(
            self.lag_pair_counts,
            dtype=torch.float64,
        )[:, None, None]
        return float((self.mean.square() * counts).sum().item())

    @property
    def within_lag_variation_squared_sum(self) -> float:
        """Captured edge energy not explained by the constant lag mean.

        This is the pooled sum of per-edge Jacobian variance
        ``E[J^2] - E[J]^2``.  A large value is the concrete signal that one
        stationary signed edge map is insufficient and a context-conditioned
        or gated executor is worth testing.
        """

        return max(
            self.captured_squared_sum
            - self.stationary_mean_squared_sum,
            0.0,
        )

    @property
    def stationary_mean_energy_fraction_of_captured(self) -> float:
        if self.captured_squared_sum == 0:
            return 1.0
        return min(
            self.stationary_mean_squared_sum
            / self.captured_squared_sum,
            1.0,
        )

    @property
    def regime_variation_fraction_of_captured(self) -> float:
        if self.captured_squared_sum == 0:
            return 0.0
        return min(
            self.within_lag_variation_squared_sum
            / self.captured_squared_sum,
            1.0,
        )

    @property
    def captured_squared_sum_by_lag(self) -> tuple[float, ...]:
        """Observed projected-Jacobian energy for each retained lag."""

        counts = torch.tensor(
            self.lag_pair_counts,
            dtype=torch.float64,
        )[:, None, None]
        values = (self.rms.square() * counts).sum(dim=(1, 2))
        return tuple(float(value.item()) for value in values)

    @property
    def stationary_mean_squared_sum_by_lag(self) -> tuple[float, ...]:
        """Energy of each lag's best constant signed modal edge."""

        counts = torch.tensor(
            self.lag_pair_counts,
            dtype=torch.float64,
        )[:, None, None]
        values = (self.mean.square() * counts).sum(dim=(1, 2))
        return tuple(float(value.item()) for value in values)

    @property
    def within_lag_variation_squared_sum_by_lag(
        self,
    ) -> tuple[float, ...]:
        """Context/position variation energy for each retained lag."""

        return tuple(
            max(captured - stationary, 0.0)
            for captured, stationary in zip(
                self.captured_squared_sum_by_lag,
                self.stationary_mean_squared_sum_by_lag,
                strict=True,
            )
        )

    @property
    def stationary_mean_energy_fraction_by_lag(
        self,
    ) -> tuple[float, ...]:
        """Per-lag constant-mean fraction; zero-energy lags return one."""

        return tuple(
            (
                1.0
                if captured == 0.0
                else min(stationary / captured, 1.0)
            )
            for captured, stationary in zip(
                self.captured_squared_sum_by_lag,
                self.stationary_mean_squared_sum_by_lag,
                strict=True,
            )
        )

    @property
    def regime_variation_fraction_by_lag(self) -> tuple[float, ...]:
        """Per-lag varying fraction; zero-energy lags return zero."""

        return tuple(
            (
                0.0
                if captured == 0.0
                else min(variation / captured, 1.0)
            )
            for captured, variation in zip(
                self.captured_squared_sum_by_lag,
                self.within_lag_variation_squared_sum_by_lag,
                strict=True,
            )
        )

    @property
    def positive_lag_stationary_mean_energy_fraction(self) -> float:
        """Constant-mean fraction after excluding same-position lag zero."""

        captured = sum(self.captured_squared_sum_by_lag[1:])
        if captured == 0.0:
            return 1.0
        stationary = sum(self.stationary_mean_squared_sum_by_lag[1:])
        return min(stationary / captured, 1.0)

    @property
    def positive_lag_regime_variation_fraction(self) -> float:
        """Varying fraction after excluding same-position lag zero."""

        captured = sum(self.captured_squared_sum_by_lag[1:])
        if captured == 0.0:
            return 0.0
        variation = sum(
            self.within_lag_variation_squared_sum_by_lag[1:]
        )
        return min(variation / captured, 1.0)

    @property
    def lag_matrices(self) -> Tensor:
        """Return executable row-map blocks ``[lag, input_mode, output_mode]``."""

        return self.mean.transpose(1, 2).contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "input_activation": self.input_activation,
            "output_activation": self.output_activation,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "max_lag": self.max_lag,
            "sequences": self.sequences,
            "jvp_calls": self.jvp_calls,
            "lag_pair_counts": self.lag_pair_counts,
            "captured_squared_sum": self.captured_squared_sum,
            "omitted_past_squared_sum": self.omitted_past_squared_sum,
            "causal_leakage_squared_sum": self.causal_leakage_squared_sum,
            "total_squared_sum": self.total_squared_sum,
            "captured_energy_fraction": self.captured_energy_fraction,
            "causal_leakage_fraction": self.causal_leakage_fraction,
            "stationary_mean_squared_sum": (
                self.stationary_mean_squared_sum
            ),
            "within_lag_variation_squared_sum": (
                self.within_lag_variation_squared_sum
            ),
            "stationary_mean_energy_fraction_of_captured": (
                self.stationary_mean_energy_fraction_of_captured
            ),
            "regime_variation_fraction_of_captured": (
                self.regime_variation_fraction_of_captured
            ),
            "captured_squared_sum_by_lag": (
                self.captured_squared_sum_by_lag
            ),
            "stationary_mean_squared_sum_by_lag": (
                self.stationary_mean_squared_sum_by_lag
            ),
            "within_lag_variation_squared_sum_by_lag": (
                self.within_lag_variation_squared_sum_by_lag
            ),
            "stationary_mean_energy_fraction_by_lag": (
                self.stationary_mean_energy_fraction_by_lag
            ),
            "regime_variation_fraction_by_lag": (
                self.regime_variation_fraction_by_lag
            ),
            "positive_lag_stationary_mean_energy_fraction": (
                self.positive_lag_stationary_mean_energy_fraction
            ),
            "positive_lag_regime_variation_fraction": (
                self.positive_lag_regime_variation_fraction
            ),
            "coordinate_scope": "projected_modal_slice",
            "projected_modal_slice_shape": (
                self.input_modes,
                self.output_modes,
            ),
            "full_residual_jacobian_energy_claim": False,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            **self.metadata(),
            "mean": self.mean.clone(),
            "rms": self.rms.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CausalLagJacobianStatistics:
        expected = {
            "format_version",
            "input_activation",
            "output_activation",
            "input_modes",
            "output_modes",
            "max_lag",
            "sequences",
            "jvp_calls",
            "lag_pair_counts",
            "captured_squared_sum",
            "omitted_past_squared_sum",
            "causal_leakage_squared_sum",
            "total_squared_sum",
            "captured_energy_fraction",
            "causal_leakage_fraction",
            "stationary_mean_squared_sum",
            "within_lag_variation_squared_sum",
            "stationary_mean_energy_fraction_of_captured",
            "regime_variation_fraction_of_captured",
            "captured_squared_sum_by_lag",
            "stationary_mean_squared_sum_by_lag",
            "within_lag_variation_squared_sum_by_lag",
            "stationary_mean_energy_fraction_by_lag",
            "regime_variation_fraction_by_lag",
            "positive_lag_stationary_mean_energy_fraction",
            "positive_lag_regime_variation_fraction",
            "coordinate_scope",
            "projected_modal_slice_shape",
            "full_residual_jacobian_energy_claim",
            "algorithm",
            "algorithm_version",
            "mean",
            "rms",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Jacobian probe state fields are invalid")
        if state["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported Jacobian probe state format")
        for name in (
            "input_activation",
            "output_activation",
            "algorithm",
        ):
            if type(state[name]) is not str:
                raise TypeError(f"{name} must be a string")
        for name in (
            "input_modes",
            "output_modes",
            "max_lag",
            "sequences",
            "jvp_calls",
            "algorithm_version",
        ):
            if type(state[name]) is not int:
                raise TypeError(f"{name} must be an integer")
        for name in (
            "captured_squared_sum",
            "omitted_past_squared_sum",
            "causal_leakage_squared_sum",
            "total_squared_sum",
            "captured_energy_fraction",
            "causal_leakage_fraction",
            "stationary_mean_squared_sum",
            "within_lag_variation_squared_sum",
            "stationary_mean_energy_fraction_of_captured",
            "regime_variation_fraction_of_captured",
        ):
            if type(state[name]) is not float:
                raise TypeError(f"{name} must be a float")
        if not isinstance(state["mean"], Tensor) or not isinstance(
            state["rms"], Tensor
        ):
            raise TypeError("Jacobian mean and RMS must be Tensors")
        counts = state["lag_pair_counts"]
        if not isinstance(counts, tuple):
            raise TypeError("lag_pair_counts must be a tuple")
        result = cls(
            input_activation=state["input_activation"],
            output_activation=state["output_activation"],
            input_modes=state["input_modes"],
            output_modes=state["output_modes"],
            max_lag=state["max_lag"],
            sequences=state["sequences"],
            jvp_calls=state["jvp_calls"],
            lag_pair_counts=counts,
            mean=state["mean"],
            rms=state["rms"],
            captured_squared_sum=state["captured_squared_sum"],
            omitted_past_squared_sum=state[
                "omitted_past_squared_sum"
            ],
            causal_leakage_squared_sum=state[
                "causal_leakage_squared_sum"
            ],
            total_squared_sum=state["total_squared_sum"],
            algorithm=state["algorithm"],
            algorithm_version=state["algorithm_version"],
        )
        expected_metadata = result.metadata()
        for name in (
            "captured_energy_fraction",
            "causal_leakage_fraction",
            "stationary_mean_squared_sum",
            "within_lag_variation_squared_sum",
            "stationary_mean_energy_fraction_of_captured",
            "regime_variation_fraction_of_captured",
            "captured_squared_sum_by_lag",
            "stationary_mean_squared_sum_by_lag",
            "within_lag_variation_squared_sum_by_lag",
            "stationary_mean_energy_fraction_by_lag",
            "regime_variation_fraction_by_lag",
            "positive_lag_stationary_mean_energy_fraction",
            "positive_lag_regime_variation_fraction",
        ):
            actual = state[name]
            expected_value = expected_metadata[name]
            if isinstance(expected_value, tuple):
                valid = (
                    isinstance(actual, tuple)
                    and len(actual) == len(expected_value)
                    and all(
                        isinstance(left, float)
                        and math.isfinite(left)
                        and math.isclose(
                            left,
                            right,
                            rel_tol=1e-12,
                            abs_tol=1e-15,
                        )
                        for left, right in zip(
                            actual,
                            expected_value,
                            strict=True,
                        )
                    )
                )
            else:
                valid = (
                    isinstance(actual, float)
                    and math.isfinite(actual)
                    and math.isclose(
                        actual,
                        float(expected_value),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )
            if not valid:
                raise ValueError("Jacobian derived metadata is invalid")
        for name in (
            "coordinate_scope",
            "projected_modal_slice_shape",
            "full_residual_jacobian_energy_claim",
        ):
            if state[name] != expected_metadata[name]:
                raise ValueError("Jacobian coordinate scope is invalid")
        return result


def _single_layer_segments(
    adapter: ModelAdapter,
    plan: LayerBlockBoundaryPlan,
) -> tuple[SegmentSpec, ...]:
    by_layer: dict[str, SegmentSpec] = {}
    for segment in adapter.segments:
        if len(segment.layer_ids) != 1:
            continue
        layer_id = segment.layer_ids[0]
        if layer_id in by_layer:
            raise ValueError(
                f"adapter has multiple single-layer segments for {layer_id!r}"
            )
        by_layer[layer_id] = segment
    missing = set(plan.layer_ids) - set(by_layer)
    if missing:
        raise ValueError(
            "Jacobian block requires one native segment per layer: "
            f"{sorted(missing)}"
        )
    segments = tuple(by_layer[layer_id] for layer_id in plan.layer_ids)
    for segment, (source, target) in zip(
        segments,
        plan.transitions,
        strict=True,
    ):
        if segment.output_site != target:
            raise ValueError(
                "single-layer segment output does not match the canonical "
                "block boundary"
            )
        if segment.input_site == source:
            continue
        input_site = adapter.activation_site(segment.input_site)
        if input_site.alias_of != source:
            raise ValueError(
                "single-layer segment input does not alias the canonical "
                "block boundary"
            )
    return segments


def collect_block_causal_lag_jacobian(
    adapter: ModelAdapter,
    plan: LayerBlockBoundaryPlan,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    input_codec: LinearActivationCodec,
    output_codec: LinearActivationCodec,
    input_modes: int,
    output_modes: int,
    max_lag: int,
    max_sequences: int | None = None,
) -> CausalLagJacobianStatistics:
    """Collect a bounded exact-JVP diagnostic through a contiguous block.

    Every calibration example is executed independently.  The routine
    supports masks and nonzero logical positions, but query and key grids must
    describe the same prefill residual sequence.  Only valid input/output
    positions are probed and scored.
    """

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    if not isinstance(plan, LayerBlockBoundaryPlan):
        raise TypeError("plan must be a LayerBlockBoundaryPlan")
    if not isinstance(input_codec, LinearActivationCodec) or not isinstance(
        output_codec, LinearActivationCodec
    ):
        raise TypeError("input_codec and output_codec must be linear codecs")
    if input_codec.activation_name != plan.activation_sites[0]:
        raise ValueError("input codec does not match the block leaf boundary")
    if output_codec.activation_name != plan.activation_sites[-1]:
        raise ValueError("output codec does not match the block output boundary")
    if input_codec.width != plan.widths[0]:
        raise ValueError("input codec width does not match the block")
    if output_codec.width != plan.widths[-1]:
        raise ValueError("output codec width does not match the block")
    if type(input_modes) is not int or not 1 <= input_modes <= input_codec.width:
        raise ValueError("input_modes is outside the input codec")
    if (
        type(output_modes) is not int
        or not 1 <= output_modes <= output_codec.width
    ):
        raise ValueError("output_modes is outside the output codec")
    if type(max_lag) is not int or max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    if max_sequences is not None and (
        type(max_sequences) is not int or max_sequences <= 0
    ):
        raise ValueError("max_sequences must be positive when provided")
    if any(parameter.requires_grad for parameter in adapter.module.parameters()):
        raise ValueError("Jacobian collection requires frozen source weights")

    segments = _single_layer_segments(adapter, plan)
    sums = torch.zeros(
        max_lag + 1,
        output_modes,
        input_modes,
        dtype=torch.float64,
    )
    square_sums = torch.zeros_like(sums)
    pair_counts = [0] * (max_lag + 1)
    captured_energy = 0.0
    omitted_energy = 0.0
    leakage_energy = 0.0
    total_energy = 0.0
    sequences = 0
    jvp_calls = 0
    seen_ids: set[str] = set()

    module = adapter.module
    was_training = module.training
    module.eval()
    try:
        for batch in calibration_batches:
            if not isinstance(batch, CalibrationBatch):
                raise TypeError(
                    "calibration_batches must contain CalibrationBatch values"
                )
            for batch_index in range(batch.batch_size):
                if max_sequences is not None and sequences >= max_sequences:
                    break
                sample = batch.sample(batch_index)
                if sample.example_ids is not None:
                    example_id = sample.example_ids[0]
                    if example_id in seen_ids:
                        raise ValueError(
                            f"duplicate calibration example_id: {example_id!r}"
                        )
                    seen_ids.add(example_id)
                with torch.no_grad():
                    source_run = adapter.forward(
                        sample.model_inputs,
                        capture_sites=(plan.leaf_activation_name,),
                        retain_gradients=False,
                    )
                    hidden_states = source_run.activations[
                        plan.leaf_activation_name
                    ].detach()
                sequence = source_run.sequence
                del source_run
                if (
                    sequence.phase != "prefill"
                    or sequence.cache_state is not None
                    or sequence.batch_size != 1
                    or sequence.query_length != hidden_states.shape[1]
                    or sequence.key_length != hidden_states.shape[1]
                ):
                    raise ValueError(
                        "Jacobian collection requires one prefill residual grid"
                    )
                if not torch.equal(
                    sequence.query_valid_mask,
                    sequence.key_valid_mask,
                ) or not torch.equal(
                    sequence.logical_positions,
                    sequence.key_logical_positions,
                ):
                    raise ValueError(
                        "Jacobian collection requires aligned query/key grids"
                    )
                valid = sequence.query_valid_mask[0]
                valid_indices = valid.nonzero(as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    raise ValueError(
                        "Jacobian calibration sequence has no valid positions"
                    )
                positions = sequence.logical_positions[0]
                input_directions = input_codec.decoder[
                    :, :input_modes
                ].to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                output_encoder = output_codec.encoder[
                    :, :output_modes
                ].to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

                def block_function(value: Tensor) -> Tensor:
                    result = value
                    for segment in segments:
                        result = adapter.run_segment(
                            segment,
                            result,
                            sequence,
                        ).hidden_states
                    return result

                for input_index_tensor in valid_indices:
                    input_index = int(input_index_tensor.item())
                    input_position = int(positions[input_index].item())
                    for input_mode in range(input_modes):
                        tangent = torch.zeros_like(hidden_states)
                        tangent[0, input_index] = input_directions[:, input_mode]
                        with torch.enable_grad():
                            _, output_tangent = torch.autograd.functional.jvp(
                                block_function,
                                hidden_states,
                                tangent,
                                create_graph=False,
                                strict=True,
                            )
                        projected = (
                            output_tangent[0] @ output_encoder
                        ).detach().to(device="cpu", dtype=torch.float64)
                        if not torch.isfinite(projected[valid]).all():
                            raise ValueError(
                                "block JVP produced non-finite modal edges"
                            )
                        for output_index_tensor in valid_indices:
                            output_index = int(output_index_tensor.item())
                            output_position = int(
                                positions[output_index].item()
                            )
                            lag = output_position - input_position
                            values = projected[output_index]
                            energy = float(values.square().sum().item())
                            total_energy += energy
                            if lag < 0:
                                leakage_energy += energy
                            elif lag > max_lag:
                                omitted_energy += energy
                            else:
                                sums[lag, :, input_mode] += values
                                square_sums[lag, :, input_mode] += values.square()
                                captured_energy += energy
                        jvp_calls += 1
                sequences += 1
                for source_index_tensor in valid_indices:
                    source_index = int(source_index_tensor.item())
                    source_position = int(positions[source_index].item())
                    for target_index_tensor in valid_indices:
                        target_index = int(target_index_tensor.item())
                        lag = int(positions[target_index].item()) - source_position
                        if 0 <= lag <= max_lag:
                            pair_counts[lag] += 1
            if max_sequences is not None and sequences >= max_sequences:
                break
    finally:
        module.train(was_training)

    if sequences == 0:
        raise ValueError("cannot collect Jacobians from an empty stream")
    mean = torch.zeros_like(sums)
    rms = torch.zeros_like(sums)
    for lag, count in enumerate(pair_counts):
        if count == 0:
            continue
        mean[lag] = sums[lag] / count
        rms[lag] = (square_sums[lag] / count).sqrt()
    return CausalLagJacobianStatistics(
        input_activation=input_codec.activation_name,
        output_activation=output_codec.activation_name,
        input_modes=input_modes,
        output_modes=output_modes,
        max_lag=max_lag,
        sequences=sequences,
        jvp_calls=jvp_calls,
        lag_pair_counts=tuple(pair_counts),
        mean=mean,
        rms=rms,
        captured_squared_sum=float(captured_energy),
        omitted_past_squared_sum=float(omitted_energy),
        causal_leakage_squared_sum=float(leakage_energy),
        total_squared_sum=float(total_energy),
    )


__all__ = [
    "CausalLagJacobianStatistics",
    "collect_block_causal_lag_jacobian",
]
