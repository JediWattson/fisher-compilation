"""Fit-only native replacement oracle for cross-block Gemma MLP modes.

This module answers a deliberately narrow question: can one earlier native
MLP ``down_proj`` input coordinate substitute for one later coordinate inside
the frozen source model?  It fits a single through-origin carry scale on
calibration-A fit data, then compares native execution, consumer ablation,
correct same-token replacement, and a same-position cross-example shuffle.

The result is diagnostic evidence only.  It does not resolve a proposal into
an executable artifact, authorize compilation, open a guard or calibration-B
split, or contain source weights, corpus rows, activation rows, or gradients.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .adapters import ModelAdapter, module_state_fingerprint
from .compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
)
from .structured_mlp_cross_block_plan import (
    StructuredMLPCrossBlockPlan,
    UnresolvedCrossBlockCarryProposal,
)
from .structured_mlp_cross_block_replacement import (
    CrossBlockReplacementConditionMetric,
    CrossBlockReplacementFitRows,
    CrossBlockReplacementOracleProvenance,
    CrossBlockReplacementOracleResult,
    CrossBlockReplacementProvenance,
    CrossBlockScalarReplacementEvidence,
    aggregate_cross_block_replacement_conditions,
    fit_cross_block_scalar_replacement,
)


GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_SCHEMA = (
    "fisher_graph.gemma3_cross_block_replacement_oracle"
)
GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_FORMAT_VERSION = 1

ScaleKind = Literal["fisher_weighted", "unweighted"]
_SCALE_KINDS = frozenset(("fisher_weighted", "unweighted"))
_CONDITION_NAMES = (
    "native",
    "consumer_ablation",
    "carried_replacement",
    "shuffled_negative_control",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_replacement_oracle.v1\0"
)
_STREAM_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_replacement_oracle.stream.v1\0"
)
_EXAMPLE_CONTENT_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_replacement_oracle."
    b"normalized_example.v1\0"
)
_CONTENT_SET_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_replacement_oracle.content_set.v1\0"
)
_SAFETY = {
    "contains_source_model_weights": False,
    "contains_executable_weights": False,
    "contains_optimizer_state": False,
    "contains_corpus_rows": False,
    "contains_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_prompt_text": False,
    "contains_tokenizer_state": False,
    "fit_only_diagnostic": True,
    "resolves_proposal": False,
    "authorizes_further_intervention": False,
    "authorizes_compilation": False,
    "authorizes_executor_construction": False,
    "authorizes_execution": False,
    "authorizes_calibration_a_guard": False,
    "authorizes_guard": False,
    "authorizes_b": False,
    "authorizes_calibration_b": False,
    "authorizes_validation": False,
    "authorizes_test": False,
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes = _ARTIFACT_DOMAIN) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_json_bytes(value))
    return digest.hexdigest()


def _assert_json_safe(value: object, *, path: str = "value") -> None:
    if isinstance(value, Tensor):
        raise RuntimeError(f"{path} contains a Tensor")
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"{path} contains an invalid mapping key")
            _assert_json_safe(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_json_safe(nested, path=f"{path}[{index}]")
        return
    raise RuntimeError(f"{path} contains unsupported type {type(value)!r}")


def _tensor_digest_update(digest: object, tensor: Tensor) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a SHA-256 object")
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _materialized_fit_stream_sha256(
    batches: Sequence[CalibrationBatch],
) -> str:
    digest = hashlib.sha256()
    digest.update(_STREAM_DOMAIN)
    for batch_index, batch in enumerate(batches):
        digest.update(f"batch:{batch_index};".encode("ascii"))
        assert batch.example_ids is not None
        for example_id in batch.example_ids:
            encoded = example_id.encode("utf-8")
            digest.update(f"id:{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
        for name in sorted(batch.model_inputs):
            digest.update(f"input:{name};".encode("utf-8"))
            _tensor_digest_update(digest, batch.model_inputs[name])
        digest.update(b"targets;")
        _tensor_digest_update(digest, batch.targets)
        digest.update(b"valid_positions;")
        _tensor_digest_update(digest, batch.valid_positions)
        for name in sorted(batch.shared_input_names):
            digest.update(f"shared:{name};".encode("utf-8"))
    return digest.hexdigest()


def _example_content_sha256(
    batch: CalibrationBatch,
    index: int,
) -> str:
    """Hash one logical tokenized example without padding or its identifier."""

    sample = batch.sample(index)
    valid = sample.valid_positions[0].detach().to(device="cpu")
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    sequence_length = int(valid.shape[0])

    def logical_tensor(value: Tensor, *, batched: bool) -> Tensor:
        """Remove the singleton batch axis and all padded sequence axes."""

        canonical = value.detach().to(device="cpu")
        if batched and canonical.ndim > 0:
            if canonical.shape[0] != 1:
                raise RuntimeError(
                    "a materialized calibration sample must have batch size 1"
                )
            canonical = canonical[0]
        for dimension in range(canonical.ndim):
            if canonical.shape[dimension] == sequence_length:
                canonical = canonical.index_select(
                    dimension,
                    valid_indices,
                )
        return canonical

    digest = hashlib.sha256()
    digest.update(_EXAMPLE_CONTENT_DOMAIN)
    digest.update(
        f"logical_positions:{int(valid_indices.numel())};".encode("ascii")
    )
    for name in sorted(sample.model_inputs):
        digest.update(f"input:{name};".encode("utf-8"))
        _tensor_digest_update(
            digest,
            logical_tensor(
                sample.model_inputs[name],
                batched=name not in sample.shared_input_names,
            ),
        )
    digest.update(b"targets;")
    _tensor_digest_update(
        digest,
        logical_tensor(sample.targets, batched=True),
    )
    for name in sorted(sample.shared_input_names):
        digest.update(f"shared:{name};".encode("utf-8"))
    return digest.hexdigest()


def _materialized_example_content_sha256s(
    batches: Sequence[CalibrationBatch],
) -> tuple[str, ...]:
    return tuple(
        _example_content_sha256(batch, index)
        for batch in batches
        for index in range(batch.batch_size)
    )


def _content_set_sha256(values: Sequence[str]) -> str:
    return _json_sha256(
        tuple(sorted(values)),
        domain=_CONTENT_SET_DOMAIN,
    )


def _materialized_fit_batches(
    values: Sequence[CalibrationBatch],
) -> tuple[tuple[CalibrationBatch, ...], tuple[str, ...]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("calibration_fit_batches must be a sequence")
    batches = tuple(values)
    if not batches or any(
        not isinstance(batch, CalibrationBatch) for batch in batches
    ):
        raise ValueError(
            "calibration_fit_batches must be a nonempty sequence of "
            "CalibrationBatch values"
        )
    example_ids: list[str] = []
    for batch in batches:
        if batch.example_ids is None:
            raise ValueError(
                "fit-only replacement requires explicit example_ids"
            )
        example_ids.extend(batch.example_ids)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("fit example_ids must be globally unique")
    return batches, tuple(example_ids)


class _FrozenSourceGuard:
    """Detect source tensor replacement, mutation, gradients, or mode drift."""

    def __init__(self, adapter: ModelAdapter) -> None:
        model = adapter.module
        if model.training:
            raise ValueError("cross-block replacement requires model.eval()")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(
                "cross-block replacement requires frozen source parameters"
            )
        dirty = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        )
        if dirty:
            raise ValueError(
                "cross-block replacement requires source parameter gradients "
                "to be empty"
            )
        self._adapter = adapter
        self._model = model
        self._state_sha256 = module_state_fingerprint(model)
        self._model_fingerprint = adapter.model_fingerprint()
        self._execution_fingerprint = adapter.execution_fingerprint()
        self._parameters = self._snapshot(
            model.named_parameters(remove_duplicate=False)
        )
        self._buffers = self._snapshot(
            model.named_buffers(remove_duplicate=False)
        )

    @staticmethod
    def _snapshot(
        values: Sequence[tuple[str, Tensor]] | object,
    ) -> dict[str, tuple[Tensor, int, int, bool]]:
        return {
            name: (
                tensor,
                int(tensor._version),
                int(tensor.untyped_storage().data_ptr()),
                bool(tensor.requires_grad),
            )
            for name, tensor in values  # type: ignore[union-attr]
        }

    @staticmethod
    def _assert_snapshot(
        *,
        label: str,
        expected: Mapping[str, tuple[Tensor, int, int, bool]],
        actual: object,
    ) -> None:
        current = dict(actual)  # type: ignore[arg-type]
        if set(current) != set(expected):
            raise RuntimeError(f"source-model {label} names changed")
        for name, (original, version, pointer, requires_grad) in expected.items():
            tensor = current[name]
            if tensor is not original:
                raise RuntimeError(
                    f"source-model {label} object {name!r} was replaced"
                )
            if int(tensor._version) != version:
                raise RuntimeError(
                    f"source-model {label} {name!r} was mutated in place"
                )
            if int(tensor.untyped_storage().data_ptr()) != pointer:
                raise RuntimeError(
                    f"source-model {label} {name!r} storage was replaced"
                )
            if bool(tensor.requires_grad) != requires_grad:
                raise RuntimeError(
                    f"source-model {label} {name!r} requires_grad changed"
                )

    def assert_unchanged(self) -> dict[str, object]:
        if self._model.training:
            raise RuntimeError("source model did not remain in eval mode")
        dirty = tuple(
            name
            for name, parameter in self._model.named_parameters()
            if parameter.grad is not None
        )
        if dirty:
            raise RuntimeError(
                "source parameter gradients appeared during replacement"
            )
        self._assert_snapshot(
            label="parameter",
            expected=self._parameters,
            actual=self._model.named_parameters(remove_duplicate=False),
        )
        self._assert_snapshot(
            label="buffer",
            expected=self._buffers,
            actual=self._model.named_buffers(remove_duplicate=False),
        )
        state_after = module_state_fingerprint(self._model)
        model_after = self._adapter.model_fingerprint()
        execution_after = self._adapter.execution_fingerprint()
        if (
            state_after != self._state_sha256
            or model_after != self._model_fingerprint
            or execution_after != self._execution_fingerprint
        ):
            raise RuntimeError(
                "source model fingerprint changed during replacement"
            )
        return {
            "source_state_sha256_before": self._state_sha256,
            "source_state_sha256_after": state_after,
            "source_model_fingerprint_before": self._model_fingerprint,
            "source_model_fingerprint_after": model_after,
            "source_execution_fingerprint_before": (
                self._execution_fingerprint
            ),
            "source_execution_fingerprint_after": execution_after,
            "source_parameter_objects_preserved": True,
            "source_parameter_storages_preserved": True,
            "source_parameter_versions_preserved": True,
            "source_parameter_requires_grad_preserved": True,
            "source_parameter_gradients_observed": False,
            "source_parameters_frozen": True,
            "source_eval_mode_preserved": True,
        }


def _proposal_for_id(
    plan: StructuredMLPCrossBlockPlan,
    proposal_id: str,
) -> UnresolvedCrossBlockCarryProposal:
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("proposal_id must be a nonempty string")
    matches = tuple(
        proposal
        for proposal in plan.proposals
        if proposal.proposal_id == proposal_id
    )
    if len(matches) != 1:
        raise ValueError(
            "proposal_id must identify exactly one proposal in the plan"
        )
    return matches[0]


def _validate_plan_binding(
    adapter: ModelAdapter,
    plan: StructuredMLPCrossBlockPlan,
    proposal: UnresolvedCrossBlockCarryProposal,
    *,
    expected_discovery_artifact_sha256: str,
    expected_plan_artifact_sha256: str,
) -> tuple[str, str]:
    if not isinstance(plan, StructuredMLPCrossBlockPlan):
        raise TypeError("plan must be a StructuredMLPCrossBlockPlan")
    discovery_sha256 = _require_sha256(
        expected_discovery_artifact_sha256,
        label="expected_discovery_artifact_sha256",
    )
    plan_sha256 = _require_sha256(
        expected_plan_artifact_sha256,
        label="expected_plan_artifact_sha256",
    )
    if discovery_sha256 != plan.source_discovery_artifact_sha256:
        raise ValueError("expected discovery artifact does not match plan")
    if plan_sha256 != plan.artifact_sha256:
        raise ValueError("expected plan artifact does not match plan")
    if (
        not plan.discovery_only
        or plan.authorizes_intervention
        or plan.authorizes_compilation
        or plan.authorizes_execution
        or plan.authorizes_guard
        or plan.authorizes_b
    ):
        raise ValueError("source plan safety metadata is invalid")
    model_fingerprint = adapter.model_fingerprint()
    if model_fingerprint != plan.source_model_fingerprint:
        raise ValueError("source model fingerprint does not match plan")

    specs = {
        spec.layer_ordinal: spec for spec in plan.source_layer_specs
    }
    for endpoint in (proposal.anchor, proposal.consumer):
        try:
            layer = adapter.layer(endpoint.layer_id)
            spec = specs[endpoint.layer_ordinal]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "proposal endpoint is outside the adapter or plan"
            ) from error
        transformer = layer.transformer
        if transformer is None or transformer.operator_sites is None:
            raise ValueError(
                "proposal endpoint layer lacks structured MLP sites"
            )
        down_input = transformer.operator_sites.feed_forward_down_input
        if (
            layer.ordinal != endpoint.layer_ordinal
            or spec.layer_id != endpoint.layer_id
            or spec.activation_site != endpoint.activation_site
            or down_input != endpoint.activation_site
            or endpoint.mode_index >= spec.width
            or endpoint.mode_index
            >= transformer.feed_forward.intermediate_width
        ):
            raise ValueError(
                "proposal endpoint does not match native MLP coordinate"
            )
        site = adapter.activation_site(endpoint.activation_site)
        if not site.intervenable:
            raise ValueError(
                f"MLP down-input site {site.id!r} is not intervenable"
            )
    return model_fingerprint, adapter.execution_fingerprint()


def _valid_mask(batch: CalibrationBatch, run: object) -> Tensor:
    sequence = getattr(run, "sequence", None)
    query_valid = getattr(sequence, "query_valid_mask", None)
    if not isinstance(query_valid, Tensor):
        raise TypeError("adapter run does not expose a query-valid mask")
    valid = batch.valid_positions.to(device=query_valid.device)
    if valid.shape != query_valid.shape:
        raise ValueError(
            "calibration valid positions do not match adapter query positions"
        )
    return valid & query_valid


@dataclass(slots=True)
class _ErrorTotals:
    values: int = 0
    square_error: float = 0.0
    reference_square: float = 0.0
    maximum_absolute_error: float = 0.0

    def update(
        self,
        reference: Tensor,
        candidate: Tensor,
        position_mask: Tensor,
    ) -> None:
        if reference.shape != candidate.shape:
            raise ValueError("paired metric tensors disagree on shape")
        if reference.shape[:2] != position_mask.shape:
            raise ValueError("paired metric mask disagrees with tensor shape")
        left = reference[position_mask].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        right = candidate[position_mask].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not left.numel():
            return
        difference = right - left
        if not torch.isfinite(difference).all():
            raise ValueError("paired metric observed a non-finite value")
        self.values += left.numel()
        self.square_error += float(
            difference.square().sum().item()
        )
        self.reference_square += float(
            left.square().sum().item()
        )
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(difference.abs().max().item()),
        )

    def update_chunked_features(
        self,
        reference: Tensor,
        candidate: Tensor,
        position_mask: Tensor,
        *,
        feature_chunk_size: int = 4096,
    ) -> None:
        """Accumulate a wide surface without materializing all features."""

        if (
            type(feature_chunk_size) is not int
            or feature_chunk_size <= 0
        ):
            raise ValueError("feature_chunk_size must be positive")
        if reference.shape != candidate.shape or reference.ndim != 3:
            raise ValueError("chunked paired tensors must be aligned 3D")
        if reference.shape[:2] != position_mask.shape:
            raise ValueError("chunked paired mask disagrees with tensors")
        for start in range(0, reference.shape[-1], feature_chunk_size):
            stop = min(start + feature_chunk_size, reference.shape[-1])
            self.update(
                reference[..., start:stop],
                candidate[..., start:stop],
                position_mask,
            )

    def finish(self) -> dict[str, object]:
        if self.values <= 0:
            raise ValueError("paired metric observed no values")
        floor = torch.finfo(torch.float64).eps
        return {
            "value_count": self.values,
            "rmse": math.sqrt(self.square_error / self.values),
            "nrmse": math.sqrt(
                self.square_error / max(self.reference_square, floor)
            ),
            "maximum_absolute_error": self.maximum_absolute_error,
            "square_error": self.square_error,
            "reference_square": self.reference_square,
        }


def _streaming_logit_sums(
    native: Tensor,
    candidate: Tensor,
    targets: Tensor,
) -> tuple[float, float, float, int]:
    """Reduce full-vocabulary metrics one token row at a time."""

    if (
        native.ndim != 2
        or candidate.shape != native.shape
        or targets.shape != (native.shape[0],)
    ):
        raise ValueError("streaming logit metric shapes are invalid")
    native_nll = 0.0
    candidate_nll = 0.0
    teacher_kl = 0.0
    matches = 0
    for index in range(native.shape[0]):
        left = native[index].float()
        right = candidate[index].float()
        target = int(targets[index].item())
        left_log_z = torch.logsumexp(left, dim=0)
        right_log_z = torch.logsumexp(right, dim=0)
        native_nll += float((left_log_z - left[target]).item())
        candidate_nll += float((right_log_z - right[target]).item())
        probability = torch.softmax(left, dim=0)
        row_kl = (
            (probability * (left - right)).sum()
            + right_log_z
            - left_log_z
        )
        teacher_kl += max(float(row_kl.item()), 0.0)
        matches += int(left.argmax().item() == right.argmax().item())
    return native_nll, candidate_nll, teacher_kl, matches


class _ConditionTotals:
    def __init__(self) -> None:
        self.batch_count = 0
        self.intervened_positions = 0
        self.supervised_tokens = 0
        self.native_nll_sum = 0.0
        self.candidate_nll_sum = 0.0
        self.teacher_kl_sum = 0.0
        self.top1_matches = 0
        self.surfaces = {
            "consumer_down_coordinate": _ErrorTotals(),
            "anchor_layer_output": _ErrorTotals(),
            "consumer_layer_input": _ErrorTotals(),
            "consumer_mlp_output": _ErrorTotals(),
            "window_output": _ErrorTotals(),
            "final_logits": _ErrorTotals(),
        }

    def update(
        self,
        *,
        native_logits: Tensor,
        candidate_logits: Tensor,
        batch: CalibrationBatch,
        valid_mask: Tensor,
        intervention_mask: Tensor,
        consumer_metric_mask: Tensor | None = None,
        native_consumer: Tensor,
        candidate_consumer: Tensor,
        native_captures: Mapping[str, Tensor],
        candidate_captures: Mapping[str, Tensor],
        capture_sites: Mapping[str, str],
        native_metric_rows: Sequence[
            CrossBlockReplacementConditionMetric
        ],
        candidate_metric_rows: Sequence[
            CrossBlockReplacementConditionMetric
        ],
        ignore_index: int,
    ) -> None:
        if native_logits.shape != candidate_logits.shape:
            raise ValueError("paired logits disagree on shape")
        targets = batch.targets.to(device=native_logits.device)
        supervised = targets != ignore_index
        if supervised.shape != valid_mask.shape or (
            supervised & ~valid_mask
        ).any():
            raise ValueError(
                "supervised targets must lie at valid query positions"
            )
        if not supervised.any():
            raise ValueError("oracle batch has no supervised targets")
        native_rows = tuple(native_metric_rows)
        candidate_rows = tuple(candidate_metric_rows)
        assert batch.example_ids is not None
        if (
            len(native_rows) != batch.batch_size
            or len(candidate_rows) != batch.batch_size
            or tuple(row.example_id for row in native_rows)
            != batch.example_ids
            or tuple(row.example_id for row in candidate_rows)
            != batch.example_ids
        ):
            raise ValueError(
                "condition metric rows do not match the paired batch"
            )
        for native_row, candidate_row in zip(
            native_rows,
            candidate_rows,
            strict=True,
        ):
            if (
                native_row.supervised_tokens
                != candidate_row.supervised_tokens
                or native_row.family_id != candidate_row.family_id
            ):
                raise ValueError(
                    "condition metric rows are not strictly paired"
                )
        supervised_tokens = sum(
            row.supervised_tokens for row in candidate_rows
        )
        if supervised_tokens != int(supervised.sum().item()):
            raise ValueError(
                "condition metric token count disagrees with targets"
            )
        native_nll = sum(row.summed_nll for row in native_rows)
        candidate_nll = sum(
            row.summed_nll for row in candidate_rows
        )
        teacher_kl = sum(
            row.teacher_kl_sum_to_native
            for row in candidate_rows
        )
        matches = sum(
            row.top1_matches_to_native
            for row in candidate_rows
        )
        self.batch_count += 1
        self.intervened_positions += int(intervention_mask.sum().item())
        self.supervised_tokens += supervised_tokens
        self.native_nll_sum += native_nll
        self.candidate_nll_sum += candidate_nll
        self.teacher_kl_sum += teacher_kl
        self.top1_matches += matches

        self.surfaces["consumer_down_coordinate"].update(
            native_consumer.unsqueeze(-1),
            candidate_consumer.unsqueeze(-1),
            (
                intervention_mask
                if consumer_metric_mask is None
                else consumer_metric_mask
            ),
        )
        for metric_name in (
            "anchor_layer_output",
            "consumer_layer_input",
            "consumer_mlp_output",
            "window_output",
        ):
            site = capture_sites[metric_name]
            self.surfaces[metric_name].update(
                native_captures[site],
                candidate_captures[site],
                valid_mask,
            )
        self.surfaces["final_logits"].update_chunked_features(
            native_logits,
            candidate_logits,
            supervised,
        )

    def finish(self) -> dict[str, object]:
        if self.batch_count <= 0 or self.supervised_tokens <= 0:
            raise ValueError("condition metric observed no paired batches")
        return {
            "available": True,
            "evaluated_batch_count": self.batch_count,
            "intervened_valid_position_count": self.intervened_positions,
            "supervised_token_count": self.supervised_tokens,
            "native_mean_nll": self.native_nll_sum
            / self.supervised_tokens,
            "candidate_mean_nll": self.candidate_nll_sum
            / self.supervised_tokens,
            "delta_mean_nll": (
                self.candidate_nll_sum - self.native_nll_sum
            )
            / self.supervised_tokens,
            "teacher_kl_per_supervised_token": self.teacher_kl_sum
            / self.supervised_tokens,
            "top1_agreement": self.top1_matches
            / self.supervised_tokens,
            "surfaces": {
                name: totals.finish()
                for name, totals in self.surfaces.items()
            },
        }


def _capture_sites(
    adapter: ModelAdapter,
    proposal: UnresolvedCrossBlockCarryProposal,
) -> dict[str, str]:
    anchor_layer = adapter.layer(proposal.anchor.layer_id)
    consumer_layer = adapter.layer(proposal.consumer.layer_id)
    transformer = consumer_layer.transformer
    if transformer is None:
        raise ValueError("consumer layer lacks transformer semantics")
    feed_forward_stages = tuple(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    if len(feed_forward_stages) != 1:
        raise ValueError(
            "consumer layer must expose exactly one feed-forward stage"
        )
    feed_forward = feed_forward_stages[0]
    return {
        "anchor_layer_output": anchor_layer.output_site,
        "consumer_layer_input": consumer_layer.input_site,
        "consumer_mlp_output": feed_forward.operator_output_site,
        "window_output": consumer_layer.output_site,
    }


def _run_with_coordinate_intervention(
    adapter: ModelAdapter,
    batch: CalibrationBatch,
    proposal: UnresolvedCrossBlockCarryProposal,
    *,
    condition: str,
    carry_scale: float,
    capture_sites: Mapping[str, str],
    shuffle_donor_indices: tuple[int, ...] | None = None,
) -> tuple[object, Tensor, Tensor, Tensor, dict[str, float]]:
    if condition not in _CONDITION_NAMES:
        raise ValueError(f"unknown oracle condition: {condition!r}")
    anchor_scalar: Tensor | None = None
    consumer_original: Tensor | None = None
    consumer_applied: Tensor | None = None
    intervention_mask: Tensor | None = None
    audit = {
        "anchor_observer_maximum_absolute_error": 0.0,
        "nonconsumer_coordinates_maximum_absolute_error": 0.0,
        "nonintervened_consumer_positions_maximum_absolute_error": 0.0,
    }
    prepared = adapter.prepare_sequence(batch.model_inputs)
    batch_valid = batch.valid_positions.to(
        device=prepared.query_valid_mask.device,
        dtype=torch.bool,
    )
    if batch_valid.shape != prepared.query_valid_mask.shape:
        raise ValueError(
            "calibration valid positions do not match adapter query positions"
        )
    coordinate_valid = batch_valid & prepared.query_valid_mask

    def observe_anchor(values: Tensor) -> Tensor:
        nonlocal anchor_scalar
        if anchor_scalar is not None:
            raise RuntimeError("anchor MLP site executed more than once")
        anchor_scalar = values[..., proposal.anchor_source_index]
        return values

    def replace_consumer(values: Tensor) -> Tensor:
        nonlocal consumer_original, consumer_applied, intervention_mask
        if consumer_original is not None:
            raise RuntimeError("consumer MLP site executed more than once")
        if anchor_scalar is None:
            raise RuntimeError("consumer MLP site executed before anchor site")
        if anchor_scalar.shape != values.shape[:2]:
            raise ValueError("anchor and consumer token layouts disagree")
        sequence_mask = coordinate_valid.to(
            device=values.device,
            dtype=torch.bool,
        )
        source_mask = sequence_mask
        target_mask = sequence_mask
        if condition == "shuffled_negative_control":
            if (
                values.shape[0] < 2
                or shuffle_donor_indices is None
                or len(shuffle_donor_indices) != values.shape[0]
                or set(shuffle_donor_indices)
                != set(range(values.shape[0]))
                or any(
                    donor == target
                    for target, donor in enumerate(
                        shuffle_donor_indices
                    )
                )
            ):
                raise ValueError(
                    "shuffled negative control requires a batch derangement"
                )
            replacement = torch.zeros_like(anchor_scalar)
            source_mask = torch.zeros_like(sequence_mask)
            logical_positions = prepared.logical_positions.to(
                device=values.device,
            )
            for target, donor in enumerate(shuffle_donor_indices):
                target_positions = logical_positions[target]
                donor_positions = logical_positions[donor]
                matches = (
                    target_positions[:, None]
                    == donor_positions[None, :]
                )
                matches = (
                    matches
                    & sequence_mask[target, :, None]
                    & sequence_mask[donor, None, :]
                )
                if (matches.sum(dim=1) > 1).any():
                    raise ValueError(
                        "logical positions are not unique within a sequence"
                    )
                available = matches.any(dim=1)
                donor_columns = matches.to(torch.int64).argmax(dim=1)
                replacement[target, available] = anchor_scalar[
                    donor,
                    donor_columns[available],
                ]
                source_mask[target] = available
            replacement = replacement * carry_scale
        elif condition == "consumer_ablation":
            replacement = torch.zeros_like(anchor_scalar)
        else:
            replacement = anchor_scalar * carry_scale
        apply = source_mask & target_mask
        consumer_original = values[..., proposal.consumer_source_index]
        updated = values.clone()
        if condition != "native":
            updated_coordinate = torch.where(
                apply,
                replacement.to(dtype=values.dtype, device=values.device),
                consumer_original,
            )
            updated[..., proposal.consumer_source_index] = updated_coordinate
        consumer_applied = updated[..., proposal.consumer_source_index]
        intervention_mask = (
            torch.zeros_like(apply) if condition == "native" else apply
        )

        if values.shape[-1] > 1:
            retained = torch.arange(
                values.shape[-1],
                device=values.device,
            )
            retained = retained[
                retained != proposal.consumer_source_index
            ]
            audit["nonconsumer_coordinates_maximum_absolute_error"] = float(
                (
                    updated.index_select(-1, retained)
                    - values.index_select(-1, retained)
                )
                .abs()
                .max()
                .item()
            )
        unchanged = ~apply
        if unchanged.any():
            audit[
                "nonintervened_consumer_positions_maximum_absolute_error"
            ] = float(
                (
                    consumer_applied[unchanged]
                    - consumer_original[unchanged]
                )
                .abs()
                .max()
                .item()
            )
        return updated

    run = adapter.forward(
        batch.model_inputs,
        capture_sites=tuple(dict.fromkeys(capture_sites.values())),
        interventions={
            proposal.anchor.activation_site: observe_anchor,
            proposal.consumer.activation_site: replace_consumer,
        },
    )
    if (
        anchor_scalar is None
        or consumer_original is None
        or consumer_applied is None
        or intervention_mask is None
    ):
        raise RuntimeError("oracle intervention sites were not executed")
    return (
        run,
        consumer_original,
        consumer_applied,
        intervention_mask,
        audit,
    )


def _fit_replacement_evidence(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    proposal: UnresolvedCrossBlockCarryProposal,
    *,
    family_by_example: Mapping[str, str],
    family_fold_assignment: Mapping[str, int],
    fold_count: int,
    model_fingerprint: str,
    fit_split_sha256: str,
    objective_sha256: str,
    proposal_artifact_sha256: str,
    scale_kind: ScaleKind,
    ignore_index: int,
) -> tuple[float, CrossBlockScalarReplacementEvidence]:
    objective = CausalLanguageModelNLL(ignore_index=ignore_index)
    rows: list[CrossBlockReplacementFitRows] = []
    for batch in batches:
        anchor_scalar: Tensor | None = None
        consumer_values: Tensor | None = None

        def observe_anchor(values: Tensor) -> Tensor:
            nonlocal anchor_scalar
            if anchor_scalar is not None:
                raise RuntimeError("anchor MLP site executed more than once")
            anchor_scalar = values[..., proposal.anchor_source_index]
            return values

        def expose_consumer_leaf(values: Tensor) -> Tensor:
            nonlocal consumer_values
            if consumer_values is not None:
                raise RuntimeError("consumer MLP site executed more than once")
            consumer_values = values.detach().requires_grad_(True)
            return consumer_values

        with torch.enable_grad():
            run = adapter.forward(
                batch.model_inputs,
                interventions={
                    proposal.anchor.activation_site: observe_anchor,
                    proposal.consumer.activation_site: expose_consumer_leaf,
                },
            )
            if anchor_scalar is None or consumer_values is None:
                raise RuntimeError("scale-fit intervention sites did not run")
            mask = _valid_mask(batch, run)
            loss = objective(run, batch)
            gradient = torch.autograd.grad(
                loss,
                consumer_values,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
        assert batch.example_ids is not None
        for index, example_id in enumerate(batch.example_ids):
            selected = mask[index]
            logical_positions = run.sequence.logical_positions[
                index
            ][selected]
            rows.append(
                CrossBlockReplacementFitRows(
                    example_id=example_id,
                    family_id=family_by_example[example_id],
                    logical_positions=logical_positions.detach().to(
                        device="cpu",
                        dtype=torch.int64,
                    ),
                    anchor_values=anchor_scalar[index][selected]
                    .detach()
                    .to(device="cpu", dtype=torch.float64),
                    consumer_values=consumer_values[
                        index, ..., proposal.consumer_source_index
                    ][selected]
                    .detach()
                    .to(device="cpu", dtype=torch.float64),
                    consumer_score_gradients=gradient[
                        index, ..., proposal.consumer_source_index
                    ][selected]
                    .detach()
                    .to(device="cpu", dtype=torch.float64),
                )
            )
    evidence = fit_cross_block_scalar_replacement(
        rows,
        provenance=CrossBlockReplacementProvenance(
            model_fingerprint=model_fingerprint,
            fit_split_sha256=fit_split_sha256,
            objective_sha256=objective_sha256,
            proposal_artifact_sha256=proposal_artifact_sha256,
        ),
        anchor=proposal.anchor,
        consumer=proposal.consumer,
        family_fold_assignment=family_fold_assignment,
        fold_count=fold_count,
    )
    scale = (
        evidence.fisher_weighted.scale
        if scale_kind == "fisher_weighted"
        else evidence.unweighted.scale
    )
    return scale, evidence


def _recovery_fraction(
    candidate: Mapping[str, object],
    ablation: Mapping[str, object],
    *,
    surface: str,
) -> float:
    candidate_surfaces = candidate["surfaces"]
    ablation_surfaces = ablation["surfaces"]
    if not isinstance(candidate_surfaces, Mapping) or not isinstance(
        ablation_surfaces,
        Mapping,
    ):
        raise TypeError("condition surface metrics are invalid")
    candidate_metric = candidate_surfaces[surface]
    ablation_metric = ablation_surfaces[surface]
    if not isinstance(candidate_metric, Mapping) or not isinstance(
        ablation_metric,
        Mapping,
    ):
        raise TypeError("condition error metrics are invalid")
    candidate_error = float(candidate_metric["square_error"])
    ablation_error = float(ablation_metric["square_error"])
    candidate_count = int(candidate_metric["value_count"])
    ablation_count = int(ablation_metric["value_count"])
    if candidate_count <= 0 or ablation_count <= 0:
        raise ValueError("condition error metric has no values")
    candidate_mse = candidate_error / candidate_count
    ablation_mse = ablation_error / ablation_count
    if ablation_mse <= torch.finfo(torch.float64).eps:
        return 0.0
    return 1.0 - candidate_mse / ablation_mse


def _validated_family_bindings(
    *,
    scale_example_ids: Sequence[str],
    evaluation_example_ids: Sequence[str],
    family_by_example: Mapping[str, str],
    family_fold_assignment: Mapping[str, int],
    fold_count: int,
) -> tuple[dict[str, str], dict[str, int]]:
    if not isinstance(family_by_example, Mapping):
        raise TypeError("family_by_example must be a mapping")
    required_ids = set(scale_example_ids) | set(evaluation_example_ids)
    if set(family_by_example) != required_ids:
        raise ValueError(
            "family_by_example must cover exactly the scale-fit and "
            "evaluation-fit examples"
        )
    normalized_families: dict[str, str] = {}
    for example_id in sorted(required_ids):
        family_id = family_by_example[example_id]
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("family ids must be nonempty strings")
        normalized_families[example_id] = family_id
    scale_families = {
        normalized_families[example_id]
        for example_id in scale_example_ids
    }
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if not isinstance(family_fold_assignment, Mapping) or set(
        family_fold_assignment
    ) != scale_families:
        raise ValueError(
            "family_fold_assignment must cover exactly the scale-fit "
            "families"
        )
    normalized_folds: dict[str, int] = {}
    for family_id in sorted(scale_families):
        fold = family_fold_assignment[family_id]
        if type(fold) is not int or not 0 <= fold < fold_count:
            raise ValueError("family fold indices are out of range")
        normalized_folds[family_id] = fold
    if set(normalized_folds.values()) != set(range(fold_count)):
        raise ValueError("every scale-fit family fold must be represented")
    return normalized_families, normalized_folds


def _family_derangement(
    example_ids: tuple[str, ...],
    family_by_example: Mapping[str, str],
) -> tuple[int, ...] | None:
    """Return a deterministic example- and family-disjoint donor matching."""

    size = len(example_ids)
    if size < 2:
        return None
    donor_to_target: dict[int, int] = {}

    def assign(target: int, seen: set[int]) -> bool:
        target_family = family_by_example[example_ids[target]]
        for donor in range(size):
            if (
                donor == target
                or donor in seen
                or family_by_example[example_ids[donor]] == target_family
            ):
                continue
            seen.add(donor)
            previous = donor_to_target.get(donor)
            if previous is None or assign(previous, seen):
                donor_to_target[donor] = target
                return True
        return False

    for target in range(size):
        if not assign(target, set()):
            return None
    target_to_donor = [-1] * size
    for donor, target in donor_to_target.items():
        target_to_donor[target] = donor
    if any(donor < 0 for donor in target_to_donor):
        return None
    return tuple(target_to_donor)


def _per_example_condition_metrics(
    *,
    native_logits: Tensor,
    candidate_logits: Tensor,
    batch: CalibrationBatch,
    family_by_example: Mapping[str, str],
    condition: str,
    ignore_index: int,
) -> tuple[CrossBlockReplacementConditionMetric, ...]:
    if condition not in ("native", "ablation", "replacement", "shuffled"):
        raise ValueError("unknown core replacement condition")
    if native_logits.shape != candidate_logits.shape:
        raise ValueError("paired condition logits disagree on shape")
    assert batch.example_ids is not None
    rows: list[CrossBlockReplacementConditionMetric] = []
    for index, example_id in enumerate(batch.example_ids):
        targets = batch.targets[index].to(device=native_logits.device)
        valid = batch.valid_positions[index].to(device=native_logits.device)
        supervised = targets != ignore_index
        if (supervised & ~valid).any() or not supervised.any():
            raise ValueError(
                "each replacement example needs supervised valid targets"
            )
        native = native_logits[index][supervised]
        candidate = candidate_logits[index][supervised]
        selected_targets = targets[supervised]
        _, summed_nll, teacher_kl, matches = _streaming_logit_sums(
            native,
            candidate,
            selected_targets,
        )
        if condition == "native":
            teacher_kl = 0.0
            matches = int(supervised.sum().item())
        rows.append(
            CrossBlockReplacementConditionMetric(
                example_id=example_id,
                family_id=family_by_example[example_id],
                condition=condition,
                supervised_tokens=int(supervised.sum().item()),
                summed_nll=summed_nll,
                teacher_kl_sum_to_native=teacher_kl,
                top1_matches_to_native=matches,
            )
        )
    return tuple(rows)


def _state_from_json(value: object) -> object:
    """Restore tuple-valued core states after a JSON round trip."""

    if isinstance(value, Mapping):
        return {
            key: _state_from_json(nested)
            for key, nested in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_state_from_json(nested) for nested in value)
    return value


def _validate_outer_conditions_against_core(
    conditions: Mapping[str, object],
    core: CrossBlockReplacementOracleResult,
) -> None:
    """Cross-check duplicated runner aggregates against the core artifact."""

    core_by_name = {
        condition.condition: condition for condition in core.conditions
    }
    native_core = core_by_name["native"]
    name_pairs = (
        ("native", "native"),
        ("consumer_ablation", "ablation"),
        ("carried_replacement", "replacement"),
        ("shuffled_negative_control", "shuffled"),
    )
    numeric_fields = (
        ("native_mean_nll", native_core.nll_per_token),
        ("candidate_mean_nll", "nll_per_token"),
        ("delta_mean_nll", "delta_nll_per_token_to_native"),
        (
            "teacher_kl_per_supervised_token",
            "teacher_kl_per_token_to_native",
        ),
        ("top1_agreement", "top1_agreement_to_native"),
    )
    for outer_name, core_name in name_pairs:
        outer = conditions[outer_name]
        if not isinstance(outer, Mapping) or outer.get("available") is not True:
            raise ValueError(
                "replacement outer/core condition availability is "
                "inconsistent"
            )
        core_condition = core_by_name[core_name]
        if (
            type(outer.get("supervised_token_count")) is not int
            or outer["supervised_token_count"]
            != core_condition.supervised_tokens
        ):
            raise ValueError(
                "replacement outer/core condition token counts are "
                "inconsistent"
            )
        for outer_field, expected in numeric_fields:
            expected_value = (
                float(expected)
                if isinstance(expected, float)
                else float(getattr(core_condition, expected))
            )
            outer_value = outer.get(outer_field)
            if (
                isinstance(outer_value, bool)
                or not isinstance(outer_value, (int, float))
                or not math.isfinite(float(outer_value))
                or not math.isclose(
                    float(outer_value),
                    expected_value,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "replacement outer/core condition aggregates are "
                    "inconsistent"
                )


def validate_gemma3_cross_block_replacement_oracle_artifact(
    artifact: Mapping[str, object],
) -> None:
    """Validate the strict JSON artifact schema and its content hash."""

    expected_fields = {
        "schema",
        "format_version",
        "binding",
        "proposal",
        "protocol",
        "fit",
        "conditions",
        "comparisons",
        "source_audit",
        "safety",
        "artifact_sha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_fields:
        raise ValueError("replacement oracle artifact fields are invalid")
    if (
        artifact["schema"]
        != GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_SCHEMA
        or artifact["format_version"]
        != GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_FORMAT_VERSION
    ):
        raise ValueError("replacement oracle artifact schema is invalid")
    _assert_json_safe(artifact, path="artifact")
    payload = dict(artifact)
    supplied_hash = _require_sha256(
        payload.pop("artifact_sha256"),
        label="artifact_sha256",
    )
    if supplied_hash != _json_sha256(payload):
        raise ValueError("replacement oracle artifact hash mismatch")
    if artifact["safety"] != _SAFETY:
        raise ValueError("replacement oracle artifact safety is invalid")
    binding = artifact["binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        "source_model_fingerprint",
        "source_execution_fingerprint",
        "source_discovery_artifact_sha256",
        "source_sketch_artifact_sha256",
        "source_plan_artifact_sha256",
        "calibration_scale_fit_split_sha256",
        "materialized_scale_fit_stream_sha256",
        "scale_fit_example_content_set_sha256",
        "calibration_evaluation_fit_split_sha256",
        "materialized_evaluation_fit_stream_sha256",
        "evaluation_fit_example_content_set_sha256",
    }:
        raise ValueError("replacement oracle binding is invalid")
    for label, value in binding.items():
        _require_sha256(value, label=label)
    if (
        binding["calibration_scale_fit_split_sha256"]
        == binding["calibration_evaluation_fit_split_sha256"]
        or binding["materialized_scale_fit_stream_sha256"]
        == binding["materialized_evaluation_fit_stream_sha256"]
        or binding["scale_fit_example_content_set_sha256"]
        == binding["evaluation_fit_example_content_set_sha256"]
    ):
        raise ValueError(
            "replacement oracle fit and evaluation bindings are not "
            "distinct"
        )
    conditions = artifact["conditions"]
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        _CONDITION_NAMES
    ):
        raise ValueError("replacement oracle conditions are invalid")
    for name in _CONDITION_NAMES:
        if not isinstance(conditions[name], Mapping):
            raise ValueError("replacement oracle condition is invalid")
    for name in ("native", "consumer_ablation", "carried_replacement"):
        if conditions[name].get("available") is not True:
            raise ValueError(f"{name} condition must be available")

    protocol = artifact["protocol"]
    fit = artifact["fit"]
    proposal = artifact["proposal"]
    comparisons = artifact["comparisons"]
    source_audit = artifact["source_audit"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            protocol,
            fit,
            proposal,
            comparisons,
            source_audit,
        )
    ):
        raise ValueError("replacement oracle nested metadata is invalid")
    if (
        protocol.get("objective_sha256") is None
        or protocol.get("shuffle_plan_sha256") is None
    ):
        raise ValueError("replacement oracle protocol hashes are missing")
    objective_sha256 = _require_sha256(
        protocol["objective_sha256"],
        label="objective_sha256",
    )
    shuffle_plan_sha256 = _require_sha256(
        protocol["shuffle_plan_sha256"],
        label="shuffle_plan_sha256",
    )
    if (
        protocol.get(
            "scale_fit_and_evaluation_fit_examples_disjoint"
        )
        is not True
        or protocol.get(
            "scale_fit_and_evaluation_fit_content_overlap_count"
        )
        != 0
        or protocol.get(
            "shuffled_control_requires_full_logical_position_coverage"
        )
        is not True
    ):
        raise ValueError("replacement oracle split/control protocol is invalid")

    if set(fit) != {
        "core_evidence",
        "selected_scale_kind",
        "selected_scale",
        "default_scale_is_fisher_weighted",
        "scale_frozen_before_evaluation_fit",
    } or not isinstance(fit["core_evidence"], Mapping):
        raise ValueError("replacement oracle fit metadata is invalid")
    evidence_state = _state_from_json(fit["core_evidence"])
    if not isinstance(evidence_state, Mapping):
        raise ValueError("replacement core evidence state is invalid")
    evidence = CrossBlockScalarReplacementEvidence.from_state_dict(
        evidence_state
    )
    selected_scale_kind = fit["selected_scale_kind"]
    if selected_scale_kind not in _SCALE_KINDS:
        raise ValueError("replacement oracle selected scale kind is invalid")
    expected_scale = (
        evidence.fisher_weighted.scale
        if selected_scale_kind == "fisher_weighted"
        else evidence.unweighted.scale
    )
    selected_scale = float(fit["selected_scale"])
    if (
        not math.isclose(
            selected_scale,
            expected_scale,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or fit["default_scale_is_fisher_weighted"] is not True
        or fit["scale_frozen_before_evaluation_fit"] is not True
    ):
        raise ValueError("replacement oracle selected scale is inconsistent")
    if (
        evidence.provenance.model_fingerprint
        != binding["source_model_fingerprint"]
        or evidence.provenance.fit_split_sha256
        != binding["calibration_scale_fit_split_sha256"]
        or evidence.provenance.objective_sha256 != objective_sha256
        or evidence.provenance.proposal_artifact_sha256
        != binding["source_plan_artifact_sha256"]
        or proposal.get("anchor") != evidence.anchor.metadata()
        or proposal.get("consumer") != evidence.consumer.metadata()
        or proposal.get("anchor_source_index")
        != evidence.anchor.mode_index
        or proposal.get("consumer_source_index")
        != evidence.consumer.mode_index
        or proposal.get("consumer_decoder_scale_resolved_for_execution")
        is not False
    ):
        raise ValueError("replacement oracle fit/proposal binding is invalid")

    core_state = comparisons.get("core_paired_oracle")
    shuffled_available = (
        conditions["shuffled_negative_control"].get("available") is True
    )
    if shuffled_available != (core_state is not None):
        raise ValueError(
            "replacement oracle shuffled/core availability is inconsistent"
        )
    if core_state is not None:
        normalized_core = _state_from_json(core_state)
        if not isinstance(normalized_core, Mapping):
            raise ValueError("replacement core oracle state is invalid")
        core = CrossBlockReplacementOracleResult.from_state_dict(
            normalized_core
        )
        if (
            core.provenance.model_fingerprint
            != binding["source_model_fingerprint"]
            or core.provenance.evaluation_fit_split_sha256
            != binding["calibration_evaluation_fit_split_sha256"]
            or core.provenance.objective_sha256 != objective_sha256
            or core.provenance.replacement_evidence_sha256
            != evidence.artifact_sha256
            or core.provenance.shuffle_plan_sha256
            != shuffle_plan_sha256
        ):
            raise ValueError(
                "replacement core oracle binding is inconsistent"
            )
        _validate_outer_conditions_against_core(conditions, core)

    ablation = conditions["consumer_ablation"]
    replacement = conditions["carried_replacement"]
    expected_recovery = {
        surface: _recovery_fraction(
            replacement,
            ablation,
            surface=surface,
        )
        for surface in (
            "consumer_down_coordinate",
            "consumer_mlp_output",
            "window_output",
            "final_logits",
        )
    }
    if comparisons.get(
        "replacement_recovery_fraction_vs_ablation"
    ) != expected_recovery:
        raise ValueError("replacement oracle recovery metrics are invalid")
    expected_shuffled_recovery = None
    if shuffled_available:
        shuffled = conditions["shuffled_negative_control"]
        expected_shuffled_recovery = {
            surface: _recovery_fraction(
                shuffled,
                ablation,
                surface=surface,
            )
            for surface in (
                "consumer_down_coordinate",
                "consumer_mlp_output",
                "window_output",
                "final_logits",
            )
        }
    if comparisons.get(
        "shuffled_recovery_fraction_vs_ablation"
    ) != expected_shuffled_recovery:
        raise ValueError(
            "replacement shuffled-control recovery metrics are invalid"
        )
    coordinate_audit = source_audit.get("coordinate_write_audit")
    if (
        source_audit.get("only_consumer_coordinate_writable") is not True
        or source_audit.get("invalid_positions_preserved") is not True
        or source_audit.get("source_parameters_frozen") is not True
        or not isinstance(coordinate_audit, Mapping)
        or set(coordinate_audit) != set(_CONDITION_NAMES)
    ):
        raise ValueError("replacement oracle source audit is invalid")
    for values in coordinate_audit.values():
        if (
            not isinstance(values, Mapping)
            or set(values)
            != {
                "anchor_observer_maximum_absolute_error",
                "nonconsumer_coordinates_maximum_absolute_error",
                "nonintervened_consumer_positions_maximum_absolute_error",
            }
            or any(float(value) != 0.0 for value in values.values())
        ):
            raise ValueError(
                "replacement oracle coordinate write audit is invalid"
            )


def run_gemma3_cross_block_replacement_oracle(
    adapter: ModelAdapter,
    calibration_fit_batches: Sequence[CalibrationBatch],
    *,
    calibration_fit_split_sha256: str,
    evaluation_fit_batches: Sequence[CalibrationBatch],
    evaluation_fit_split_sha256: str,
    family_by_example: Mapping[str, str],
    family_fold_assignment: Mapping[str, int],
    fold_count: int,
    plan: StructuredMLPCrossBlockPlan,
    proposal_id: str,
    expected_discovery_artifact_sha256: str,
    expected_plan_artifact_sha256: str,
    scale_kind: ScaleKind = "fisher_weighted",
    ignore_index: int = -100,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit on one A-fit stream and evaluate on a disjoint A-fit stream.

    The scale-fit and evaluation-fit streams are both development data, but
    their example IDs and split hashes must be disjoint.  This freezes the
    scalar before the paired native intervention replay without consuming a
    guard, calibration-B, validation, or test split.
    """

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    scale_split_sha256 = _require_sha256(
        calibration_fit_split_sha256,
        label="calibration_fit_split_sha256",
    )
    evaluation_split_sha256 = _require_sha256(
        evaluation_fit_split_sha256,
        label="evaluation_fit_split_sha256",
    )
    if scale_split_sha256 == evaluation_split_sha256:
        raise ValueError(
            "scale-fit and evaluation-fit split digests must be distinct"
        )
    if scale_kind not in _SCALE_KINDS:
        raise ValueError(
            "scale_kind must be 'fisher_weighted' or 'unweighted'"
        )
    if type(ignore_index) is not int:
        raise TypeError("ignore_index must be an integer")
    if not isinstance(plan, StructuredMLPCrossBlockPlan):
        raise TypeError("plan must be a StructuredMLPCrossBlockPlan")
    proposal = _proposal_for_id(plan, proposal_id)
    model_fingerprint, execution_fingerprint = _validate_plan_binding(
        adapter,
        plan,
        proposal,
        expected_discovery_artifact_sha256=(
            expected_discovery_artifact_sha256
        ),
        expected_plan_artifact_sha256=expected_plan_artifact_sha256,
    )
    scale_batches, scale_ids = _materialized_fit_batches(
        calibration_fit_batches
    )
    evaluation_batches, evaluation_ids = _materialized_fit_batches(
        evaluation_fit_batches
    )
    if set(scale_ids) & set(evaluation_ids):
        raise ValueError(
            "scale-fit and evaluation-fit example_ids must be disjoint"
        )
    scale_content_sha256s = _materialized_example_content_sha256s(
        scale_batches
    )
    evaluation_content_sha256s = (
        _materialized_example_content_sha256s(evaluation_batches)
    )
    content_overlap = set(scale_content_sha256s) & set(
        evaluation_content_sha256s
    )
    if content_overlap:
        raise ValueError(
            "scale-fit and evaluation-fit tokenized example content "
            "must be disjoint"
        )
    scale_content_set_sha256 = _content_set_sha256(
        scale_content_sha256s
    )
    evaluation_content_set_sha256 = _content_set_sha256(
        evaluation_content_sha256s
    )
    families, folds = _validated_family_bindings(
        scale_example_ids=scale_ids,
        evaluation_example_ids=evaluation_ids,
        family_by_example=family_by_example,
        family_fold_assignment=family_fold_assignment,
        fold_count=fold_count,
    )
    scale_stream_sha256 = _materialized_fit_stream_sha256(scale_batches)
    evaluation_stream_sha256 = _materialized_fit_stream_sha256(
        evaluation_batches
    )
    objective_descriptor = {
        "name": "causal_language_model_nll",
        "target_kind": "hard_ground_truth_tokens",
        "reduction": "sum",
        "ignore_index": ignore_index,
        "consumer_fisher_weight": (
            "squared_gradient_of_summed_nll_wrt_consumer_coordinate"
        ),
    }
    objective_sha256 = _json_sha256(objective_descriptor)
    source_guard = _FrozenSourceGuard(adapter)
    carry_scale, fit_evidence = _fit_replacement_evidence(
        adapter,
        scale_batches,
        proposal,
        family_by_example=families,
        family_fold_assignment=folds,
        fold_count=fold_count,
        model_fingerprint=model_fingerprint,
        fit_split_sha256=scale_split_sha256,
        objective_sha256=objective_sha256,
        proposal_artifact_sha256=plan.artifact_sha256,
        scale_kind=scale_kind,
        ignore_index=ignore_index,
    )
    fit_report = {
        "core_evidence": fit_evidence.state_dict(),
        "selected_scale_kind": scale_kind,
        "selected_scale": carry_scale,
        "default_scale_is_fisher_weighted": True,
        "scale_frozen_before_evaluation_fit": True,
    }

    capture_sites = _capture_sites(adapter, proposal)
    totals = {
        name: _ConditionTotals()
        for name in _CONDITION_NAMES
        if name != "shuffled_negative_control"
    }
    shuffled_totals = _ConditionTotals()
    shuffle_full_batch_count = 0
    shuffle_target_position_count = 0
    shuffle_matched_position_count = 0
    core_rows: list[CrossBlockReplacementConditionMetric] = []
    coordinate_audit = {
        name: {
            "anchor_observer_maximum_absolute_error": 0.0,
            "nonconsumer_coordinates_maximum_absolute_error": 0.0,
            "nonintervened_consumer_positions_maximum_absolute_error": 0.0,
        }
        for name in _CONDITION_NAMES
    }
    shuffle_donors_by_batch = tuple(
        _family_derangement(batch.example_ids, families)
        for batch in evaluation_batches
        if batch.example_ids is not None
    )
    if len(shuffle_donors_by_batch) != len(evaluation_batches):
        raise RuntimeError("evaluation batches lost their example IDs")

    def update_audit(condition: str, values: Mapping[str, float]) -> None:
        for key, value in values.items():
            coordinate_audit[condition][key] = max(
                coordinate_audit[condition][key],
                value,
            )

    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_batches):
            (
                native_run,
                native_consumer,
                native_applied,
                _,
                native_audit,
            ) = _run_with_coordinate_intervention(
                adapter,
                batch,
                proposal,
                condition="native",
                carry_scale=carry_scale,
                capture_sites=capture_sites,
            )
            valid = _valid_mask(batch, native_run)
            shuffle_target_position_count += int(valid.sum().item())
            native_core_batch = _per_example_condition_metrics(
                native_logits=native_run.logits,
                candidate_logits=native_run.logits,
                batch=batch,
                family_by_example=families,
                condition="native",
                ignore_index=ignore_index,
            )
            totals["native"].update(
                native_logits=native_run.logits,
                candidate_logits=native_run.logits,
                batch=batch,
                valid_mask=valid,
                intervention_mask=torch.zeros_like(valid),
                consumer_metric_mask=valid,
                native_consumer=native_consumer,
                candidate_consumer=native_applied,
                native_captures=native_run.activations,
                candidate_captures=native_run.activations,
                capture_sites=capture_sites,
                native_metric_rows=native_core_batch,
                candidate_metric_rows=native_core_batch,
                ignore_index=ignore_index,
            )
            core_rows.extend(native_core_batch)
            update_audit("native", native_audit)

            for public_name, core_name in (
                ("consumer_ablation", "ablation"),
                ("carried_replacement", "replacement"),
            ):
                (
                    candidate_run,
                    _,
                    candidate_consumer,
                    applied_mask,
                    candidate_audit,
                ) = _run_with_coordinate_intervention(
                    adapter,
                    batch,
                    proposal,
                    condition=public_name,
                    carry_scale=carry_scale,
                    capture_sites=capture_sites,
                )
                candidate_core_batch = _per_example_condition_metrics(
                    native_logits=native_run.logits,
                    candidate_logits=candidate_run.logits,
                    batch=batch,
                    family_by_example=families,
                    condition=core_name,
                    ignore_index=ignore_index,
                )
                totals[public_name].update(
                    native_logits=native_run.logits,
                    candidate_logits=candidate_run.logits,
                    batch=batch,
                    valid_mask=valid,
                    intervention_mask=applied_mask,
                    native_consumer=native_consumer,
                    candidate_consumer=candidate_consumer,
                    native_captures=native_run.activations,
                    candidate_captures=candidate_run.activations,
                    capture_sites=capture_sites,
                    native_metric_rows=native_core_batch,
                    candidate_metric_rows=candidate_core_batch,
                    ignore_index=ignore_index,
                )
                core_rows.extend(candidate_core_batch)
                update_audit(public_name, candidate_audit)

            shuffle_donors = shuffle_donors_by_batch[batch_index]
            if shuffle_donors is not None:
                (
                    shuffled_run,
                    _,
                    shuffled_consumer,
                    shuffled_mask,
                    shuffled_audit,
                ) = _run_with_coordinate_intervention(
                    adapter,
                    batch,
                    proposal,
                    condition="shuffled_negative_control",
                    carry_scale=carry_scale,
                    capture_sites=capture_sites,
                    shuffle_donor_indices=shuffle_donors,
                )
                shuffle_matched_position_count += int(
                    shuffled_mask.sum().item()
                )
                if torch.equal(shuffled_mask, valid):
                    shuffle_full_batch_count += 1
                    shuffled_core_batch = (
                        _per_example_condition_metrics(
                            native_logits=native_run.logits,
                            candidate_logits=shuffled_run.logits,
                            batch=batch,
                            family_by_example=families,
                            condition="shuffled",
                            ignore_index=ignore_index,
                        )
                    )
                    shuffled_totals.update(
                        native_logits=native_run.logits,
                        candidate_logits=shuffled_run.logits,
                        batch=batch,
                        valid_mask=valid,
                        intervention_mask=shuffled_mask,
                        native_consumer=native_consumer,
                        candidate_consumer=shuffled_consumer,
                        native_captures=native_run.activations,
                        candidate_captures=shuffled_run.activations,
                        capture_sites=capture_sites,
                        native_metric_rows=native_core_batch,
                        candidate_metric_rows=shuffled_core_batch,
                        ignore_index=ignore_index,
                    )
                    core_rows.extend(shuffled_core_batch)
                    update_audit(
                        "shuffled_negative_control",
                        shuffled_audit,
                    )

    complete_shuffle = (
        shuffle_full_batch_count == len(evaluation_batches)
        and shuffle_matched_position_count
        == shuffle_target_position_count
    )
    shuffle_coverage = (
        shuffle_matched_position_count / shuffle_target_position_count
    )
    conditions: dict[str, object] = {
        "native": totals["native"].finish(),
        "consumer_ablation": totals["consumer_ablation"].finish(),
        "carried_replacement": totals["carried_replacement"].finish(),
        "shuffled_negative_control": (
            shuffled_totals.finish()
            if complete_shuffle
            else {
                "available": False,
                "reason": (
                    "every evaluation-fit batch must admit a family "
                    "derangement covering every valid logical position"
                ),
                "fully_covered_batch_count": shuffle_full_batch_count,
                "evaluation_batch_count": len(evaluation_batches),
                "matched_valid_position_count": (
                    shuffle_matched_position_count
                ),
                "target_valid_position_count": (
                    shuffle_target_position_count
                ),
                "logical_position_coverage": shuffle_coverage,
                "partial_metrics_discarded": (
                    shuffle_matched_position_count > 0
                ),
            }
        ),
    }
    ablation_metrics = conditions["consumer_ablation"]
    replacement_metrics = conditions["carried_replacement"]
    assert isinstance(ablation_metrics, Mapping)
    assert isinstance(replacement_metrics, Mapping)
    replacement_recovery = {
        surface: _recovery_fraction(
            replacement_metrics,
            ablation_metrics,
            surface=surface,
        )
        for surface in (
            "consumer_down_coordinate",
            "consumer_mlp_output",
            "window_output",
            "final_logits",
        )
    }
    shuffled_recovery = None
    if complete_shuffle:
        shuffled_metrics = conditions["shuffled_negative_control"]
        assert isinstance(shuffled_metrics, Mapping)
        shuffled_recovery = {
            surface: _recovery_fraction(
                shuffled_metrics,
                ablation_metrics,
                surface=surface,
            )
            for surface in (
                "consumer_down_coordinate",
                "consumer_mlp_output",
                "window_output",
                "final_logits",
            )
        }

    shuffle_plan = tuple(
        {
            "example_ids": batch.example_ids,
            "donor_example_ids": (
                None
                if donors is None
                else tuple(
                    batch.example_ids[index]
                    for index in donors
                )
            ),
            "policy": (
                "family_derangement_joined_by_logical_position"
            ),
        }
        for batch, donors in zip(
            evaluation_batches,
            shuffle_donors_by_batch,
            strict=True,
        )
    )
    shuffle_plan_sha256 = _json_sha256(shuffle_plan)
    core_oracle = None
    if complete_shuffle:
        core_oracle = aggregate_cross_block_replacement_conditions(
            core_rows,
            provenance=CrossBlockReplacementOracleProvenance(
                model_fingerprint=model_fingerprint,
                evaluation_fit_split_sha256=(
                    evaluation_split_sha256
                ),
                objective_sha256=objective_sha256,
                replacement_evidence_sha256=fit_evidence.artifact_sha256,
                shuffle_plan_sha256=shuffle_plan_sha256,
                shuffle_policy=(
                    "family_derangement_joined_by_logical_position"
                ),
            ),
        )
    comparisons = {
        "replacement_recovery_fraction_vs_ablation": (
            replacement_recovery
        ),
        "shuffled_recovery_fraction_vs_ablation": shuffled_recovery,
        "core_paired_oracle": (
            None if core_oracle is None else core_oracle.state_dict()
        ),
        "core_paired_oracle_unavailable_reason": (
            None
            if core_oracle is not None
            else (
                "a complete paired quartet requires every evaluation-fit "
                "batch to support same-position cross-example shuffling"
            )
        ),
        "interpretation": (
            "positive recovery means the condition reduced mean squared "
            "error relative to deleting the consumer coordinate"
        ),
        "thresholds_frozen": False,
        "pass_fail_gate_applied": False,
        "causal_substitutability_proven": False,
    }
    source_audit = source_guard.assert_unchanged()
    source_audit["coordinate_write_audit"] = coordinate_audit
    source_audit["only_consumer_coordinate_writable"] = True
    source_audit["invalid_positions_preserved"] = True

    binding = {
        "source_model_fingerprint": model_fingerprint,
        "source_execution_fingerprint": execution_fingerprint,
        "source_discovery_artifact_sha256": (
            plan.source_discovery_artifact_sha256
        ),
        "source_sketch_artifact_sha256": (
            plan.source_sketch_artifact_sha256
        ),
        "source_plan_artifact_sha256": plan.artifact_sha256,
        "calibration_scale_fit_split_sha256": scale_split_sha256,
        "materialized_scale_fit_stream_sha256": scale_stream_sha256,
        "scale_fit_example_content_set_sha256": (
            scale_content_set_sha256
        ),
        "calibration_evaluation_fit_split_sha256": (
            evaluation_split_sha256
        ),
        "materialized_evaluation_fit_stream_sha256": (
            evaluation_stream_sha256
        ),
        "evaluation_fit_example_content_set_sha256": (
            evaluation_content_set_sha256
        ),
    }
    protocol = {
        "calibration_role": "calibration_a_fit_only",
        "scale_fit_role": "calibration_a_scale_fit",
        "evaluation_fit_role": "calibration_a_unseen_fit_evaluation",
        "scale_fit_batch_count": len(scale_batches),
        "scale_fit_independent_sequence_count": len(scale_ids),
        "evaluation_fit_batch_count": len(evaluation_batches),
        "evaluation_fit_independent_sequence_count": len(evaluation_ids),
        "scale_fit_and_evaluation_fit_examples_disjoint": True,
        "scale_fit_and_evaluation_fit_content_overlap_count": 0,
        "conditions": _CONDITION_NAMES,
        "scale_fit_passes": 1,
        "paired_evaluation_passes_per_batch": (
            4 if complete_shuffle else 3
        ),
        "shuffled_negative_control": (
            "carry_scale_times_family_deranged_anchor_joined_by_logical_position"
        ),
        "shuffled_control_requires_full_logical_position_coverage": True,
        "shuffled_control_fully_covered_batch_count": (
            shuffle_full_batch_count
        ),
        "shuffled_control_target_valid_position_count": (
            shuffle_target_position_count
        ),
        "shuffled_control_matched_valid_position_count": (
            shuffle_matched_position_count
        ),
        "shuffled_control_logical_position_coverage": shuffle_coverage,
        "shuffle_changes_time_position": False,
        "consumer_write_scope": (
            "one_native_coordinate_where_source_and_target_are_valid"
        ),
        "objective": objective_descriptor,
        "objective_sha256": objective_sha256,
        "shuffle_plan_sha256": shuffle_plan_sha256,
        "guard_or_b_data_consumed": False,
        "heldout_or_test_data_consumed": False,
    }
    proposal_metadata = {
        "proposal_id": proposal.proposal_id,
        "anchor": proposal.anchor.metadata(),
        "consumer": proposal.consumer.metadata(),
        "anchor_source_index": proposal.anchor_source_index,
        "consumer_source_index": proposal.consumer_source_index,
        "inclusive_layer_interval": proposal.inclusive_interval,
        "consumer_decoder_scale_resolved_for_execution": False,
    }
    payload: dict[str, object] = {
        "schema": GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_SCHEMA,
        "format_version": (
            GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_FORMAT_VERSION
        ),
        "binding": binding,
        "proposal": proposal_metadata,
        "protocol": protocol,
        "fit": fit_report,
        "conditions": conditions,
        "comparisons": comparisons,
        "source_audit": source_audit,
        "safety": dict(_SAFETY),
    }
    artifact = {**payload, "artifact_sha256": _json_sha256(payload)}
    validate_gemma3_cross_block_replacement_oracle_artifact(artifact)
    report: dict[str, object] = {
        "schema": GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_SCHEMA,
        "format_version": (
            GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_FORMAT_VERSION
        ),
        "scientific_status": {
            "outcome": (
                "disjoint_fit_only_native_cross_block_replacement_completed"
            ),
            "scale_fit_completed": True,
            "unseen_fit_evaluation_completed": True,
            "shuffled_negative_control_available": complete_shuffle,
            "scientific_compression_success": False,
            "executable_candidate_built": False,
            "proposal_resolved": False,
            "calibration_a_guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "authorizes_execution": False,
            "authorizes_guard": False,
            "authorizes_b": False,
        },
        "binding": binding,
        "proposal": proposal_metadata,
        "protocol": protocol,
        "fit": fit_report,
        "conditions": conditions,
        "comparisons": comparisons,
        "source_audit": source_audit,
        "artifact": {
            "artifact_sha256": artifact["artifact_sha256"],
            **dict(_SAFETY),
        },
    }
    _assert_json_safe(report, path="report")
    return artifact, report


__all__ = [
    "GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_FORMAT_VERSION",
    "GEMMA3_CROSS_BLOCK_REPLACEMENT_ORACLE_SCHEMA",
    "ScaleKind",
    "run_gemma3_cross_block_replacement_oracle",
    "validate_gemma3_cross_block_replacement_oracle_artifact",
]
