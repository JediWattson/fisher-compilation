"""Run the first executable Gemma layer-agnostic whole-model rung.

This worker deliberately implements the preregistered
``compressed_lexical_exact_body_ablation`` arm, not the unrestricted primary
arm.  It compiles one standalone 39-event executor:

``lexical ingress -> 36 residual mutations -> final RMSNorm -> lexical egress``.

The transformer body is the existing exact cloned residual-carrier control.
Compression in this first runnable rung comes only from one low-rank lexical
factor shared physically by token ingress and vocabulary egress.  Consequently
the report may claim tied-lexical parameter reduction, but never body-MAC,
latency, generator-sharing, or unrestricted layer-agnostic compression wins.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_body_residual_carrier_experiment import (
    _compile_complete_body,
)
from .gemma3_full_width_single_layer_experiment import (
    _assert_source_independence,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    DEFAULT_PANEL_PATH,
    DEFAULT_REVISION,
    _load_prompt_split,
    _tokenize_batches,
)
from .gemma3_one_block_residual_form_experiment import (
    _LogitMetricAccumulator,
    _native_nll_sum,
)
from .gemma3_layer_agnostic_whole_model_compression_protocol import (
    DEFAULT_PROTOCOL_OUTPUT,
    PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
    PINNED_HIDDEN_WIDTH,
    PINNED_LAYER_COUNT,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
    PINNED_TIED_LEXICAL_PARAMETER_COUNT,
    PINNED_TRANSFORMER_BODY_PARAMETER_COUNT,
    PINNED_VOCABULARY_SIZE,
    build_layer_agnostic_whole_model_compression_protocol,
    load_layer_agnostic_whole_model_compression_protocol,
)


GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_SCHEMA = (
    "fisher_graph.gemma3_layer_agnostic_whole_model_compression.worker.v1"
)
GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_FORMAT_VERSION = 1
PROTOCOL_ARM_ID = "compressed_lexical_exact_body_ablation"
LEXICAL_RANK_LADDER = (640, 512, 384, 256)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer-agnostic-compressed-lexical-exact-body-r512-v1.json"
)
DEFAULT_LEXICAL_RANK = 512
DEFAULT_GRAM_CHUNK_ROWS = 4096
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-layer-agnostic-whole-model-worker:v1\0"
)


def _progress(message: str) -> None:
    print(f"[layer-agnostic-worker] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _storage_pointers(module: nn.Module) -> set[int]:
    pointers = {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if isinstance(value, Tensor) and value.numel()
    }
    pointers.discard(0)
    return pointers


class SharedTiedLexicalFactor(nn.Module):
    """One physical lexical representation reused in both directions.

    Rank ``hidden_width`` stores one exact copy of the tied checkpoint matrix.
    Lower ranks store ``codebook [vocabulary, rank]`` and
    ``basis [hidden, rank]`` exactly once.  Ingress and egress reference those
    same tensors; a second language-model-head copy is never materialized.
    """

    def __init__(
        self,
        *,
        exact_weight: Tensor | None = None,
        codebook: Tensor | None = None,
        basis: Tensor | None = None,
        embed_scale: float = 1.0,
    ) -> None:
        super().__init__()
        exact = exact_weight is not None
        factorized = codebook is not None or basis is not None
        if exact == factorized:
            raise ValueError(
                "provide either exact_weight or both codebook and basis"
            )
        if not isinstance(embed_scale, (int, float)) or isinstance(
            embed_scale, bool
        ) or not math.isfinite(float(embed_scale)) or embed_scale <= 0:
            raise ValueError("embed_scale must be finite and positive")
        if exact:
            assert isinstance(exact_weight, Tensor)
            if (
                exact_weight.ndim != 2
                or not exact_weight.is_floating_point()
                or not bool(torch.isfinite(exact_weight).all())
            ):
                raise ValueError("exact lexical weight is invalid")
            self.exact_weight = nn.Parameter(
                exact_weight.detach().clone(), requires_grad=False
            )
            self.register_parameter("packed_factor", None)
            vocabulary, hidden = exact_weight.shape
            rank = hidden
            self._codebook_elements = 0
        else:
            if not isinstance(codebook, Tensor) or not isinstance(basis, Tensor):
                raise TypeError("factorized lexical tensors must be Tensors")
            if (
                codebook.ndim != 2
                or basis.ndim != 2
                or codebook.shape[1] != basis.shape[1]
                or basis.shape[0] < basis.shape[1]
                or codebook.dtype != basis.dtype
                or codebook.device != basis.device
                or not codebook.is_floating_point()
                or not bool(torch.isfinite(codebook).all())
                or not bool(torch.isfinite(basis).all())
            ):
                raise ValueError("factorized lexical tensors are invalid")
            self.register_parameter("exact_weight", None)
            packed = torch.cat(
                (
                    codebook.detach().reshape(-1),
                    basis.detach().reshape(-1),
                )
            )
            self.packed_factor = nn.Parameter(packed, requires_grad=False)
            vocabulary = codebook.shape[0]
            hidden, rank = basis.shape
            self._codebook_elements = int(codebook.numel())
        self._vocabulary_size = int(vocabulary)
        self._hidden_width = int(hidden)
        self._rank = int(rank)
        self.register_buffer(
            "embed_scale",
            torch.tensor(float(embed_scale), dtype=torch.float32),
            persistent=False,
        )
        self.eval().requires_grad_(False)

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    @property
    def hidden_width(self) -> int:
        return self._hidden_width

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_exact(self) -> bool:
        return self.exact_weight is not None

    @property
    def codebook(self) -> Tensor | None:
        if self.packed_factor is None:
            return None
        return self.packed_factor[: self._codebook_elements].view(
            self.vocabulary_size,
            self.rank,
        )

    @property
    def basis(self) -> Tensor | None:
        if self.packed_factor is None:
            return None
        return self.packed_factor[self._codebook_elements :].view(
            self.hidden_width,
            self.rank,
        )

    @property
    def stored_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def physical_storage_count(self) -> int:
        return len(_storage_pointers(self))

    def ingress(self, input_ids: Tensor) -> Tensor:
        if (
            not isinstance(input_ids, Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("input_ids must be an integer [batch, sequence] tensor")
        scale = self.embed_scale.to(
            device=input_ids.device,
            dtype=(
                self.exact_weight.dtype
                if self.exact_weight is not None
                else self.codebook.dtype  # type: ignore[union-attr]
            ),
        )
        if self.exact_weight is not None:
            return F.embedding(input_ids, self.exact_weight) * scale
        assert self.codebook is not None and self.basis is not None
        coordinates = F.embedding(input_ids, self.codebook)
        return (coordinates @ self.basis.transpose(0, 1)) * scale

    def egress(self, hidden_states: Tensor) -> Tensor:
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.ndim != 3
            or hidden_states.shape[-1] != self.hidden_width
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                "hidden_states must be floating [batch, sequence, hidden]"
            )
        if self.exact_weight is not None:
            return F.linear(hidden_states, self.exact_weight)
        assert self.codebook is not None and self.basis is not None
        coordinates = hidden_states @ self.basis
        return coordinates @ self.codebook.transpose(0, 1)

    def forward(self, value: Tensor, *, direction: str) -> Tensor:
        if direction == "ingress":
            return self.ingress(value)
        if direction == "egress":
            return self.egress(value)
        raise ValueError("direction must be 'ingress' or 'egress'")

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "shared_tied_lexical_factor",
            "rank": self.rank,
            "hidden_width": self.hidden_width,
            "vocabulary_size": self.vocabulary_size,
            "is_exact_copied_matrix": self.is_exact,
            "stored_scalar_count": self.stored_scalar_count,
            "physical_storage_count": self.physical_storage_count,
            "ingress_and_egress_share_the_same_storage": True,
            "second_head_copy_present": False,
        }


def compute_tied_lexical_gram(
    weight: Tensor,
    *,
    chunk_rows: int = DEFAULT_GRAM_CHUNK_ROWS,
) -> Tensor:
    """Compute ``E.T @ E`` without a full float64 copy of ``E``."""

    if (
        not isinstance(weight, Tensor)
        or weight.ndim != 2
        or not weight.is_floating_point()
    ):
        raise ValueError("lexical weight must be a floating matrix")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    hidden = weight.shape[1]
    gram = torch.zeros(
        hidden,
        hidden,
        dtype=torch.float64,
        device="cpu",
    )
    source = weight.detach()
    for start in range(0, source.shape[0], chunk_rows):
        chunk = source[start : start + chunk_rows].to(
            device="cpu", dtype=torch.float64
        )
        gram.addmm_(chunk.transpose(0, 1), chunk)
    if not bool(torch.isfinite(gram).all()):
        raise RuntimeError("lexical Gram matrix contains non-finite values")
    return gram


def _deterministic_basis_sign_(basis: Tensor) -> None:
    for column in range(basis.shape[1]):
        vector = basis[:, column]
        pivot = int(vector.abs().argmax())
        if float(vector[pivot]) < 0:
            vector.mul_(-1)


def fit_shared_tied_lexical_factor(
    weight: Tensor,
    rank: int,
    *,
    embed_scale: float = 1.0,
    gram: Tensor | None = None,
    chunk_rows: int = DEFAULT_GRAM_CHUNK_ROWS,
) -> SharedTiedLexicalFactor:
    """Fit one tied lexical factor using a chunked right singular basis."""

    if (
        not isinstance(weight, Tensor)
        or weight.ndim != 2
        or not weight.is_floating_point()
    ):
        raise ValueError("lexical weight must be a floating matrix")
    vocabulary, hidden = weight.shape
    if type(rank) is not int or rank <= 0 or rank > hidden:
        raise ValueError("lexical rank is outside the hidden width")
    if rank == hidden:
        return SharedTiedLexicalFactor(
            exact_weight=weight,
            embed_scale=embed_scale,
        )
    resolved_gram = (
        compute_tied_lexical_gram(weight, chunk_rows=chunk_rows)
        if gram is None
        else gram
    )
    if (
        not isinstance(resolved_gram, Tensor)
        or resolved_gram.shape != (hidden, hidden)
        or resolved_gram.device.type != "cpu"
        or resolved_gram.dtype != torch.float64
        or not bool(torch.isfinite(resolved_gram).all())
    ):
        raise ValueError("lexical Gram matrix is invalid")
    eigenvalues, eigenvectors = torch.linalg.eigh(resolved_gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    basis64 = eigenvectors.index_select(1, order).contiguous()
    _deterministic_basis_sign_(basis64)
    basis = basis64.to(device=weight.device, dtype=weight.dtype)
    codebook = torch.empty(
        vocabulary,
        rank,
        dtype=weight.dtype,
        device=weight.device,
    )
    with torch.no_grad():
        for start in range(0, vocabulary, chunk_rows):
            stop = min(start + chunk_rows, vocabulary)
            codebook[start:stop].copy_(weight[start:stop] @ basis)
    return SharedTiedLexicalFactor(
        codebook=codebook,
        basis=basis,
        embed_scale=embed_scale,
    )


@dataclass(frozen=True, slots=True)
class WholeModelGraphExecution:
    """Inspectable closure receipt for the standalone 39-event executor."""

    output: Tensor
    normalized_output: Tensor
    logits: Tensor
    event_ids: tuple[str, ...]
    mutation_receipts: tuple[object, ...]

    def __post_init__(self) -> None:
        if len(self.event_ids) != 39:
            raise ValueError("whole-model graph must close exactly 39 events")
        if self.event_ids[0] != "lexical.ingress" or self.event_ids[-2:] != (
            "terminal.rmsnorm",
            "lexical.egress",
        ):
            raise ValueError("whole-model boundary event order is invalid")
        if len(self.mutation_receipts) != 36:
            raise ValueError("whole-model graph must execute 36 mutations")


class Gemma3LayerAgnosticWholeModelExecutor(nn.Module):
    """Standalone first-rung executor with no live teacher reference."""

    def __init__(
        self,
        *,
        lexical: SharedTiedLexicalFactor,
        body: nn.Module,
        terminal_norm: nn.Module,
        final_logit_softcap: float | None,
    ) -> None:
        super().__init__()
        if not isinstance(lexical, SharedTiedLexicalFactor):
            raise TypeError("lexical must be SharedTiedLexicalFactor")
        if not isinstance(body, nn.Module) or not callable(
            getattr(body, "forward_components", None)
        ):
            raise TypeError("body must expose forward_components")
        if not isinstance(terminal_norm, nn.Module):
            raise TypeError("terminal_norm must be an nn.Module")
        if final_logit_softcap is not None and (
            not isinstance(final_logit_softcap, (int, float))
            or isinstance(final_logit_softcap, bool)
            or not math.isfinite(float(final_logit_softcap))
            or final_logit_softcap <= 0
        ):
            raise ValueError("final_logit_softcap must be positive when set")
        self.lexical = lexical
        self.body = body
        self.terminal_norm = terminal_norm
        self.final_logit_softcap = (
            None
            if final_logit_softcap is None
            else float(final_logit_softcap)
        )
        self.eval().requires_grad_(False)

    @property
    def ordered_event_ids(self) -> tuple[str, ...]:
        mutation_plan = getattr(self.body, "mutation_plan", None)
        if isinstance(mutation_plan, (tuple, list)):
            mutation_ids = tuple(
                str(getattr(step, "mutation_id")) for step in mutation_plan
            )
        else:
            expected = getattr(self.body, "expected_mutation_ids", None)
            if not isinstance(expected, (tuple, list)):
                raise TypeError("body omitted its mutation schedule")
            mutation_ids = tuple(str(value) for value in expected)
        if len(mutation_ids) != 36 or len(set(mutation_ids)) != 36:
            raise RuntimeError("body omitted the pinned 36 source mutations")
        result = (
            "lexical.ingress",
            *(f"mutation.{index}" for index in range(1, 37)),
            "terminal.rmsnorm",
            "lexical.egress",
        )
        if len(result) != 39 or len(set(result)) != 39:
            raise RuntimeError("executor event schedule is not the pinned 39 events")
        return result

    @property
    def deployed_scalar_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_components(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
    ) -> WholeModelGraphExecution:
        hidden = self.lexical.ingress(input_ids)
        body_execution = self.body.forward_components(
            hidden,
            sequence,
            capture_layer_outputs=False,
        )
        output = getattr(body_execution, "output", None)
        receipts = getattr(body_execution, "mutation_receipts", None)
        if not isinstance(output, Tensor) or not isinstance(
            receipts, (tuple, list)
        ):
            raise TypeError("compiled body returned an invalid execution receipt")
        normalized = self.terminal_norm(output)
        logits = self.lexical.egress(normalized)
        if self.final_logit_softcap is not None:
            cap = self.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        result = WholeModelGraphExecution(
            output=output,
            normalized_output=normalized,
            logits=logits,
            event_ids=self.ordered_event_ids,
            mutation_receipts=tuple(receipts),
        )
        source_receipt_ids = tuple(
            str(getattr(receipt, "mutation_id", ""))
            for receipt in result.mutation_receipts
        )
        source_plan = getattr(self.body, "expected_mutation_ids", None)
        if source_plan is not None and source_receipt_ids != tuple(source_plan):
            raise RuntimeError("body receipt order disagrees with source plan")
        return result

    execute = forward_components

    def forward(self, input_ids: Tensor, sequence: SequenceContext) -> Tensor:
        return self.forward_components(input_ids, sequence).logits

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "gemma3_layer_agnostic_whole_model_executor",
            "protocol_arm_id": PROTOCOL_ARM_ID,
            "event_ids": self.ordered_event_ids,
            "source_mutation_ids_are_provenance_only": True,
            "event_count": 39,
            "ordered_body_mutation_count": 36,
            "lexical": self.lexical.metadata(),
            "body": (
                self.body.architecture_manifest()
                if callable(getattr(self.body, "architecture_manifest", None))
                else {"kind": type(self.body).__qualname__}
            ),
            "terminal_rmsnorm_compiled": True,
            "final_logit_softcap": self.final_logit_softcap,
            "live_teacher_reference_present": False,
            "unrestricted_primary_implemented": False,
            "body_parameter_or_mac_reduction_claimed": False,
        }


def _source_lexical_modules(
    adapter: Gemma3CausalLMAdapter,
) -> tuple[nn.Module, nn.Module, Tensor, float]:
    model = adapter.module
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    input_module = get_input() if callable(get_input) else None
    output_module = get_output() if callable(get_output) else None
    if not isinstance(input_module, nn.Module) or not isinstance(
        output_module, nn.Module
    ):
        raise TypeError("Gemma must expose input and output embeddings")
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if not isinstance(input_weight, Tensor) or not isinstance(
        output_weight, Tensor
    ):
        raise TypeError("Gemma lexical modules must expose weights")
    physically_tied = (
        input_weight.untyped_storage().data_ptr()
        == output_weight.untyped_storage().data_ptr()
    )
    if not physically_tied or input_weight.shape != output_weight.shape:
        raise RuntimeError("Gemma input embedding and output head are not tied")
    raw_scale = getattr(input_module, "embed_scale", 1.0)
    scale = float(raw_scale.item() if isinstance(raw_scale, Tensor) else raw_scale)
    return input_module, output_module, input_weight, scale


def compile_gemma3_layer_agnostic_whole_model_executor(
    adapter: Gemma3CausalLMAdapter,
    *,
    lexical_rank: int,
    gram: Tensor | None = None,
    gram_chunk_rows: int = DEFAULT_GRAM_CHUNK_ROWS,
) -> Gemma3LayerAgnosticWholeModelExecutor:
    """Compile one source-independent runtime for the registered ablation."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be Gemma3CausalLMAdapter")
    if len(adapter.layers) != PINNED_LAYER_COUNT:
        raise RuntimeError("worker requires the pinned 18-block Gemma")
    _, _, weight, embed_scale = _source_lexical_modules(adapter)
    lexical = fit_shared_tied_lexical_factor(
        weight,
        lexical_rank,
        embed_scale=embed_scale,
        gram=gram,
        chunk_rows=gram_chunk_rows,
    )
    body = _compile_complete_body(
        adapter,
        dtype=torch.float32,
        device=weight.device,
    )
    backbone = getattr(adapter.module, "model", None)
    source_norm = getattr(backbone, "norm", None)
    if not isinstance(source_norm, nn.Module):
        raise TypeError("Gemma backbone omitted terminal RMSNorm")
    terminal_norm = copy.deepcopy(source_norm).eval().requires_grad_(False)
    softcap = getattr(adapter.module.config, "final_logit_softcapping", None)
    executor = Gemma3LayerAgnosticWholeModelExecutor(
        lexical=lexical,
        body=body,
        terminal_norm=terminal_norm,
        final_logit_softcap=softcap,
    )
    _assert_source_independence(
        adapter.module,
        {"layer_agnostic_whole_model_executor": executor},
    )
    return executor


def _resource_report(
    executor: Gemma3LayerAgnosticWholeModelExecutor,
    *,
    source_unique_parameters: int = PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
    source_tied_lexical_parameters: int = PINNED_TIED_LEXICAL_PARAMETER_COUNT,
    source_body_parameters: int = PINNED_TRANSFORMER_BODY_PARAMETER_COUNT,
    source_terminal_norm_parameters: int = PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
    serialized_scalar_bytes: int = 4,
) -> dict[str, object]:
    if not isinstance(executor, Gemma3LayerAgnosticWholeModelExecutor):
        raise TypeError("executor has the wrong type")
    lexical = executor.lexical.stored_scalar_count
    body = sum(parameter.numel() for parameter in executor.body.parameters())
    terminal = sum(
        parameter.numel() for parameter in executor.terminal_norm.parameters()
    )
    candidate = lexical + body + terminal
    expected_lexical = (
        executor.lexical.vocabulary_size * executor.lexical.hidden_width
        if executor.lexical.is_exact
        else executor.lexical.rank
        * (
            executor.lexical.vocabulary_size
            + executor.lexical.hidden_width
        )
    )
    if lexical != expected_lexical:
        raise RuntimeError("candidate lexical factor count does not close")
    if body != source_body_parameters:
        raise RuntimeError("candidate exact body parameter count drifted")
    if terminal != source_terminal_norm_parameters:
        raise RuntimeError("candidate terminal norm parameter count drifted")
    if candidate != executor.deployed_scalar_count:
        raise RuntimeError("candidate resource inventory does not close")
    source_closes = (
        source_tied_lexical_parameters
        + source_body_parameters
        + source_terminal_norm_parameters
        == source_unique_parameters
    )
    if not source_closes:
        raise ValueError("source parameter inventory does not close")
    if type(serialized_scalar_bytes) is not int or serialized_scalar_bytes <= 0:
        raise ValueError("serialized_scalar_bytes must be positive")
    reduction = source_unique_parameters - candidate
    hidden = executor.lexical.hidden_width
    vocabulary = executor.lexical.vocabulary_size
    rank = executor.lexical.rank
    source_lexical_macs_per_token = hidden * vocabulary
    if executor.lexical.is_exact:
        candidate_ingress_macs_per_token = 0
        candidate_egress_macs_per_token = source_lexical_macs_per_token
    else:
        candidate_ingress_macs_per_token = rank * hidden
        candidate_egress_macs_per_token = rank * (
            hidden + vocabulary
        )
    candidate_lexical_macs_per_token = (
        candidate_ingress_macs_per_token
        + candidate_egress_macs_per_token
    )
    return {
        "source_unique_parameters": source_unique_parameters,
        "source_tied_lexical_parameters_counted_once": (
            source_tied_lexical_parameters
        ),
        "source_transformer_body_parameters": source_body_parameters,
        "source_terminal_rmsnorm_parameters": (
            source_terminal_norm_parameters
        ),
        "source_inventory_closes": True,
        "candidate_deployed_parameters": candidate,
        "candidate_tied_lexical_parameters_counted_once": lexical,
        "candidate_transformer_body_parameters": body,
        "candidate_terminal_rmsnorm_parameters": terminal,
        "candidate_inventory_closes": True,
        "removed_parameters": reduction,
        "parameter_reduction_fraction": reduction / source_unique_parameters,
        "serialized_scalar_bytes": serialized_scalar_bytes,
        "serialized_bytes_kind": "projected_float32_coefficient_bytes",
        "serialized_artifact_written": False,
        "measured_serialized_artifact_bytes": None,
        "source_serialized_bytes": (
            source_unique_parameters * serialized_scalar_bytes
        ),
        "candidate_serialized_bytes": candidate * serialized_scalar_bytes,
        "removed_serialized_bytes": reduction * serialized_scalar_bytes,
        "serialized_byte_reduction_fraction": (
            reduction / source_unique_parameters
        ),
        "tied_lexical_reduction_parameters": (
            source_tied_lexical_parameters - lexical
        ),
        "body_parameter_reduction": source_body_parameters - body,
        "terminal_norm_parameter_reduction": (
            source_terminal_norm_parameters - terminal
        ),
        "logical_body_mac_reduction_claimed": False,
        "lexical_logical_macs_per_valid_token": {
            "source_ingress_embedding_lookup_macs": 0,
            "source_egress_macs": source_lexical_macs_per_token,
            "source_total_macs": source_lexical_macs_per_token,
            "candidate_ingress_decode_macs": (
                candidate_ingress_macs_per_token
            ),
            "candidate_egress_macs": candidate_egress_macs_per_token,
            "candidate_total_macs": candidate_lexical_macs_per_token,
            "removed_macs": (
                source_lexical_macs_per_token
                - candidate_lexical_macs_per_token
            ),
            "reduction_fraction": (
                1.0
                - candidate_lexical_macs_per_token
                / source_lexical_macs_per_token
            ),
            "embedding_lookup_addressing_and_memory_traffic_excluded": True,
            "softcap_and_terminal_norm_excluded": True,
        },
        "whole_model_logical_mac_reduction_claimed": False,
        "latency_reduction_claimed": False,
        "physical_storage_count": executor.lexical.physical_storage_count,
        "second_lexical_head_copy_present": False,
        "copied_exact_body_checkpoint_coefficients_counted": True,
        "source_free_deployable_artifact_established": False,
    }


def _protocol_receipt(protocol_path: Path | str) -> dict[str, object]:
    protocol = load_layer_agnostic_whole_model_compression_protocol(
        Path(protocol_path)
    )
    arms = protocol.get("arms")
    if not isinstance(arms, list):
        raise ValueError("layer-agnostic protocol omitted its arm catalog")
    matching = [
        arm
        for arm in arms
        if isinstance(arm, Mapping) and arm.get("arm_id") == PROTOCOL_ARM_ID
    ]
    if len(matching) != 1:
        raise ValueError("protocol does not authenticate the requested arm")
    return {
        "protocol_schema": protocol.get("schema"),
        "protocol_format_version": protocol.get("format_version"),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "protocol_arm_id": PROTOCOL_ARM_ID,
        "protocol_arm": dict(matching[0]),
    }


def _source_call_modules(
    adapter: Gemma3CausalLMAdapter,
) -> dict[str, nn.Module]:
    input_module, output_module, _, _ = _source_lexical_modules(adapter)
    backbone = getattr(adapter.module, "model", None)
    norm = getattr(backbone, "norm", None)
    if not isinstance(norm, nn.Module):
        raise TypeError("Gemma terminal norm is unavailable")
    modules: dict[str, nn.Module] = {
        "lexical_ingress": input_module,
        "terminal_rmsnorm": norm,
        "lexical_egress": output_module,
    }
    modules.update(
        {
            f"source_{layer.id}": adapter.source_module(layer.id)
            for layer in adapter.layers
        }
    )
    return modules


def _evaluate_executor(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3LayerAgnosticWholeModelExecutor,
    batches: Sequence[Mapping[str, Tensor]],
) -> tuple[dict[str, object], dict[str, object]]:
    if not batches:
        raise ValueError("worker evaluation requires tokenized batches")
    modules = _source_call_modules(adapter)
    native_calls = {name: 0 for name in modules}
    compiled_calls = {name: 0 for name in modules}
    phase = "idle"
    handles: list[Any] = []

    def count_source(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
        *,
        name: str,
    ) -> None:
        if phase == "native":
            native_calls[name] += 1
        elif phase == "compiled":
            compiled_calls[name] += 1
        else:
            raise RuntimeError("source module executed outside an audited phase")

    for name, module in modules.items():
        handles.append(
            module.register_forward_hook(
                lambda module, args, output, *, name=name: count_source(
                    module,
                    args,
                    output,
                    name=name,
                )
            )
        )

    accumulator = _LogitMetricAccumulator()
    native_nll_sum = 0.0
    supervised_tokens = 0
    event_receipts: list[list[str]] = []
    source_mutation_receipts: list[list[str]] = []
    try:
        for batch in batches:
            phase = "native"
            with torch.inference_mode():
                native = adapter.forward(batch)
            batch_nll, batch_tokens = _native_nll_sum(native.logits, batch)
            native_nll_sum += batch_nll
            supervised_tokens += batch_tokens

            phase = "compiled"
            with torch.inference_mode():
                sequence = adapter.prepare_sequence(batch)
                execution = executor.execute(batch["input_ids"], sequence)
            phase = "idle"
            accumulator.update(native.logits, execution.logits, batch)
            event_receipts.append(list(execution.event_ids))
            source_mutation_receipts.append(
                [
                    str(getattr(receipt, "mutation_id", ""))
                    for receipt in execution.mutation_receipts
                ]
            )
    finally:
        phase = "idle"
        for handle in reversed(handles):
            handle.remove()
    if supervised_tokens <= 0:
        raise RuntimeError("worker evaluation produced no supervised tokens")
    if any(value != 0 for value in compiled_calls.values()):
        raise RuntimeError("compiled whole-model executor called the teacher")
    if any(receipt != list(executor.ordered_event_ids) for receipt in event_receipts):
        raise RuntimeError("compiled whole-model event schedule drifted")
    native_nll = native_nll_sum / supervised_tokens
    metrics = {
        "native": {
            "supervised_tokens": supervised_tokens,
            "nll_per_token": native_nll,
        },
        "candidate": accumulator.result(native_nll_per_token=native_nll),
    }
    audit = {
        "batch_count": len(batches),
        "native_source_calls": native_calls,
        "compiled_source_calls": compiled_calls,
        "compiled_source_call_total": sum(compiled_calls.values()),
        "zero_live_teacher_calls": True,
        "event_count_per_execution": 39,
        "body_mutation_count_per_execution": 36,
        "event_receipts": event_receipts,
        "source_mutation_receipts_provenance_only": source_mutation_receipts,
        "all_event_receipts_match_executor_schedule": True,
    }
    return metrics, audit


def _require_exact_control_parity(metrics: Mapping[str, object]) -> None:
    candidate = metrics.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("exact control metrics are incomplete")
    exact_values = {
        "delta_nll_per_token": 0.0,
        "native_to_candidate_kl_per_query": 0.0,
        "top1_agreement_to_native": 1.0,
        "logit_nrmse": 0.0,
        "maximum_absolute_logit_error": 0.0,
    }
    if any(candidate.get(name) != value for name, value in exact_values.items()):
        raise RuntimeError("rank-640 exact whole-model control failed parity")


def _build_qualification(
    candidate_metrics: Mapping[str, object],
    candidate_resources: Mapping[str, object],
    *,
    protocol_sha256: str,
) -> dict[str, object]:
    """Recompute the preregistered gate screen without authorizing a claim."""

    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise ValueError("qualification requires the protocol SHA-256")
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    if protocol.get("protocol_sha256") != protocol_sha256:
        raise ValueError("qualification protocol SHA-256 is not current")
    acceptance = protocol.get("acceptance_gates")
    if not isinstance(acceptance, Mapping):
        raise RuntimeError("protocol acceptance gates are unavailable")
    fidelity_thresholds = acceptance.get("candidate_fidelity")
    compression_thresholds = acceptance.get("compression")
    if not isinstance(fidelity_thresholds, Mapping) or not isinstance(
        compression_thresholds, Mapping
    ):
        raise RuntimeError("protocol candidate gates are unavailable")

    def numeric(name: str, source: Mapping[str, object]) -> float | None:
        value = source.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    delta_nll = numeric("delta_nll_per_token", candidate_metrics)
    candidate_kl = numeric(
        "native_to_candidate_kl_per_query", candidate_metrics
    )
    top1 = numeric("top1_agreement_to_native", candidate_metrics)
    downstream = numeric(
        "mean_downstream_accuracy_retained", candidate_metrics
    )
    worst_family = numeric(
        "worst_family_bootstrap_lower_bound", candidate_metrics
    )
    parameter_reduction = numeric(
        "parameter_reduction_fraction", candidate_resources
    )
    byte_reduction = numeric(
        "serialized_byte_reduction_fraction", candidate_resources
    )

    delta_max = float(
        fidelity_thresholds["delta_nll_per_token_maximum"]
    )
    kl_max = float(
        fidelity_thresholds[
            "native_to_candidate_kl_per_token_maximum"
        ]
    )
    top1_min = float(fidelity_thresholds["top1_agreement_minimum"])
    downstream_min = float(
        fidelity_thresholds[
            "mean_downstream_accuracy_retained_minimum"
        ]
    )
    worst_family_min = float(
        fidelity_thresholds[
            "worst_family_bootstrap_lower_bound_minimum"
        ]
    )
    parameter_min = float(
        compression_thresholds[
            "whole_model_unique_parameter_reduction_minimum"
        ]
    )
    byte_min = float(
        compression_thresholds["serialized_byte_reduction_minimum"]
    )

    fidelity_gates = {
        "delta_nll_per_token": {
            "observed": delta_nll,
            "maximum": delta_max,
            "passed": delta_nll is not None and delta_nll <= delta_max,
        },
        "native_to_candidate_kl_per_token_position": {
            "observed": candidate_kl,
            "maximum": kl_max,
            "source_metric": "native_to_candidate_kl_per_query",
            "passed": candidate_kl is not None and candidate_kl <= kl_max,
        },
        "top1_agreement": {
            "observed": top1,
            "minimum": top1_min,
            "passed": top1 is not None and top1 >= top1_min,
        },
        "downstream_accuracy_retained": {
            "observed": downstream,
            "minimum": downstream_min,
            "passed": downstream is not None and downstream >= downstream_min,
        },
        "worst_family_bootstrap_lower_bound": {
            "observed": worst_family,
            "minimum": worst_family_min,
            "passed": (
                worst_family is not None and worst_family >= worst_family_min
            ),
        },
    }
    missing_fidelity = [
        name
        for name, gate in fidelity_gates.items()
        if gate.get("observed") is None
    ]
    fidelity_passed = all(
        gate.get("passed") is True for gate in fidelity_gates.values()
    )

    parameter_gate_passed = (
        parameter_reduction is not None and parameter_reduction >= parameter_min
    )
    projected_byte_numeric_passed = (
        byte_reduction is not None and byte_reduction >= byte_min
    )
    serialized_artifact_written = (
        candidate_resources.get("serialized_artifact_written") is True
    )
    projected_byte_authoritative = False
    primary_evidence = {
        "matched_or_beat_svd_and_deletion_at_same_budget": False,
        "real_transport_gain_positive": False,
        "shuffled_gain_fraction_within_limit": False,
        "whole_model_logical_mac_reduction_measured": False,
    }
    compression_passed = (
        parameter_gate_passed
        and projected_byte_numeric_passed
        and projected_byte_authoritative
        and all(primary_evidence.values())
    )
    source_free_passed = (
        serialized_artifact_written
        and candidate_resources.get(
            "source_free_deployable_artifact_established"
        )
        is True
        and candidate_resources.get(
            "copied_exact_body_checkpoint_coefficients_counted"
        )
        is False
    )

    reasons: list[str] = []
    for name, gate in fidelity_gates.items():
        if gate["passed"] is not True:
            reasons.append(f"fidelity:{name}")
    if not parameter_gate_passed:
        reasons.append("compression:whole_model_parameter_reduction")
    if not projected_byte_numeric_passed:
        reasons.append("compression:projected_serialized_byte_reduction")
    reasons.append("compression:projected_byte_gate_not_authoritative")
    for name, passed in primary_evidence.items():
        if not passed:
            reasons.append(f"compression:{name}")
    if not serialized_artifact_written:
        reasons.append("source_free:serialized_candidate_artifact_not_written")
    if candidate_resources.get(
        "copied_exact_body_checkpoint_coefficients_counted"
    ) is True:
        reasons.append("source_free:copied_exact_body_coefficients_retained")
    if not source_free_passed:
        reasons.append("source_free:deployable_artifact_not_established")

    qualified = fidelity_passed and compression_passed and source_free_passed
    return {
        "gate_version": 1,
        "protocol_sha256": protocol_sha256,
        "thresholds_from_protocol": {
            "candidate_fidelity": dict(fidelity_thresholds),
            "compression": dict(compression_thresholds),
        },
        "compression": {
            "whole_model_unique_parameter_reduction": {
                "observed": parameter_reduction,
                "minimum": parameter_min,
                "passed": parameter_gate_passed,
            },
            "serialized_byte_reduction": {
                "observed_projected_float32_coefficient_fraction": (
                    byte_reduction
                ),
                "minimum": byte_min,
                "numeric_screen_passed": projected_byte_numeric_passed,
                "serialized_artifact_written": serialized_artifact_written,
                "projected_byte_gate_authoritative": (
                    projected_byte_authoritative
                ),
                "passed": False,
            },
            "primary_protocol_evidence": primary_evidence,
            "passed": compression_passed,
        },
        "fidelity": {
            "per_gate": fidelity_gates,
            "missing_required_metrics": missing_fidelity,
            "passed": fidelity_passed,
        },
        "source_free": {
            "serialized_artifact_written": serialized_artifact_written,
            "copied_exact_body_coefficients_counted": (
                candidate_resources.get(
                    "copied_exact_body_checkpoint_coefficients_counted"
                )
                is True
            ),
            "deployable_source_free_artifact_established": (
                candidate_resources.get(
                    "source_free_deployable_artifact_established"
                )
                is True
            ),
            "passed": source_free_passed,
        },
        "passed": qualified,
        "qualified": qualified,
        "compression_claim_authorized": qualified,
        "compression_success_authorized": qualified,
        "compression_success_claimed": False,
        "failure_reasons": reasons,
    }


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if (
        report.get("schema")
        != GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_SCHEMA
        or report.get("format_version")
        != GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_FORMAT_VERSION
    ):
        raise ValueError("unsupported layer-agnostic worker report")
    if report.get("contains_model_weights") is not False or report.get(
        "contains_prompt_text"
    ) is not False:
        raise ValueError("worker report contains forbidden material")
    protocol = report.get("protocol")
    status = report.get("scientific_status")
    resources = report.get("resources")
    audit = report.get("execution_audit")
    compilation = report.get("compilation")
    metrics = report.get("metrics")
    qualification = report.get("qualification")
    if not all(
        isinstance(value, Mapping)
        for value in (
            protocol,
            status,
            resources,
            audit,
            compilation,
            metrics,
            qualification,
        )
    ):
        raise ValueError("worker report sections are incomplete")
    assert isinstance(protocol, Mapping)
    assert isinstance(status, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(audit, Mapping)
    assert isinstance(compilation, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(qualification, Mapping)
    expected_protocol = build_layer_agnostic_whole_model_compression_protocol()
    exact_audit = audit.get("rank_640_exact_control")
    candidate_audit = audit.get("requested_candidate")
    integrity = audit.get("executor_integrity")
    exact_resources = resources.get("rank_640_exact_control")
    candidate_resources = resources.get("requested_candidate")
    exact_metrics = metrics.get("rank_640_exact_control")
    if not all(
        isinstance(value, Mapping)
        for value in (
            exact_audit,
            candidate_audit,
            exact_resources,
            candidate_resources,
            exact_metrics,
            integrity,
        )
    ):
        raise ValueError("worker control/candidate sections are incomplete")
    assert isinstance(exact_audit, Mapping)
    assert isinstance(candidate_audit, Mapping)
    assert isinstance(exact_resources, Mapping)
    assert isinstance(candidate_resources, Mapping)
    assert isinstance(exact_metrics, Mapping)
    assert isinstance(integrity, Mapping)
    if (
        protocol.get("protocol_arm_id") != PROTOCOL_ARM_ID
        or protocol.get("protocol_sha256")
        != expected_protocol.get("protocol_sha256")
        or compilation.get("protocol_arm_id") != PROTOCOL_ARM_ID
        or status.get("unrestricted_layer_agnostic_primary_executed") is not False
        or status.get("exact_transformer_body_retained") is not True
        or status.get("shared_tied_lexical_factor_executed") is not True
        or status.get("compression_success_claimed") is not False
        or status.get("compression_claim_authorized") is not False
        or status.get("compression_success_authorized") is not False
        or status.get("qualification_passed") is not False
        or any(
            condition.get("compiled_source_call_total") != 0
            or condition.get("zero_live_teacher_calls") is not True
            or condition.get("event_count_per_execution") != 39
            or condition.get("body_mutation_count_per_execution") != 36
            for condition in (exact_audit, candidate_audit)
        )
        or any(
            condition.get("second_lexical_head_copy_present") is not False
            or condition.get("logical_body_mac_reduction_claimed") is not False
            or condition.get("whole_model_logical_mac_reduction_claimed")
            is not False
            or condition.get("latency_reduction_claimed") is not False
            for condition in (exact_resources, candidate_resources)
        )
        or integrity.get("rank_640_parameter_count_unchanged") is not True
        or integrity.get("requested_parameter_count_unchanged") is not True
        or integrity.get("rank_640_parameters_before")
        != integrity.get("rank_640_parameters_after")
        or integrity.get("requested_parameters_before")
        != integrity.get("requested_parameters_after")
    ):
        raise ValueError("worker report claim boundary is invalid")

    exact_candidate_metrics = exact_metrics.get("candidate")
    if not isinstance(exact_candidate_metrics, Mapping) or any(
        exact_candidate_metrics.get(name) != value
        for name, value in {
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_query": 0.0,
            "top1_agreement_to_native": 1.0,
            "logit_nrmse": 0.0,
            "maximum_absolute_logit_error": 0.0,
        }.items()
    ):
        raise ValueError("worker rank-640 exact control parity is invalid")

    requested_rank = compilation.get("lexical_rank")
    if type(requested_rank) is not int or requested_rank not in LEXICAL_RANK_LADDER:
        raise ValueError("worker requested lexical rank is invalid")

    def validate_resources(value: Mapping[str, object], rank: int) -> None:
        lexical = (
            PINNED_VOCABULARY_SIZE * PINNED_HIDDEN_WIDTH
            if rank == PINNED_HIDDEN_WIDTH
            else rank * (PINNED_VOCABULARY_SIZE + PINNED_HIDDEN_WIDTH)
        )
        candidate = (
            lexical
            + PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
            + PINNED_FINAL_RMSNORM_PARAMETER_COUNT
        )
        removed = PINNED_SOURCE_UNIQUE_PARAMETER_COUNT - candidate
        expected = {
            "source_unique_parameters": PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
            "source_tied_lexical_parameters_counted_once": (
                PINNED_TIED_LEXICAL_PARAMETER_COUNT
            ),
            "source_transformer_body_parameters": (
                PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
            ),
            "source_terminal_rmsnorm_parameters": (
                PINNED_FINAL_RMSNORM_PARAMETER_COUNT
            ),
            "candidate_deployed_parameters": candidate,
            "candidate_tied_lexical_parameters_counted_once": lexical,
            "candidate_transformer_body_parameters": (
                PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
            ),
            "candidate_terminal_rmsnorm_parameters": (
                PINNED_FINAL_RMSNORM_PARAMETER_COUNT
            ),
            "removed_parameters": removed,
            "parameter_reduction_fraction": (
                removed / PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            "serialized_scalar_bytes": 4,
            "serialized_bytes_kind": (
                "projected_float32_coefficient_bytes"
            ),
            "serialized_artifact_written": False,
            "measured_serialized_artifact_bytes": None,
            "source_serialized_bytes": (
                4 * PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            "candidate_serialized_bytes": 4 * candidate,
            "removed_serialized_bytes": 4 * removed,
            "serialized_byte_reduction_fraction": (
                removed / PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
            ),
            "body_parameter_reduction": 0,
            "terminal_norm_parameter_reduction": 0,
            "physical_storage_count": 1,
            "copied_exact_body_checkpoint_coefficients_counted": True,
            "source_free_deployable_artifact_established": False,
        }
        if any(
            value.get(name) != expected_value
            for name, expected_value in expected.items()
        ):
            raise ValueError("worker resource ledger was not recomputed")
        lexical_macs = value.get("lexical_logical_macs_per_valid_token")
        if not isinstance(lexical_macs, Mapping):
            raise ValueError("worker lexical MAC ledger is incomplete")
        source_macs = PINNED_HIDDEN_WIDTH * PINNED_VOCABULARY_SIZE
        ingress_macs = (
            0
            if rank == PINNED_HIDDEN_WIDTH
            else rank * PINNED_HIDDEN_WIDTH
        )
        egress_macs = (
            source_macs
            if rank == PINNED_HIDDEN_WIDTH
            else rank * (PINNED_HIDDEN_WIDTH + PINNED_VOCABULARY_SIZE)
        )
        candidate_macs = ingress_macs + egress_macs
        expected_macs = {
            "source_ingress_embedding_lookup_macs": 0,
            "source_egress_macs": source_macs,
            "source_total_macs": source_macs,
            "candidate_ingress_decode_macs": ingress_macs,
            "candidate_egress_macs": egress_macs,
            "candidate_total_macs": candidate_macs,
            "removed_macs": source_macs - candidate_macs,
            "reduction_fraction": 1.0 - candidate_macs / source_macs,
        }
        if any(
            lexical_macs.get(name) != expected_value
            for name, expected_value in expected_macs.items()
        ):
            raise ValueError("worker lexical MAC ledger was not recomputed")

    validate_resources(exact_resources, PINNED_HIDDEN_WIDTH)
    validate_resources(candidate_resources, requested_rank)

    requested_metrics = metrics.get("requested_candidate")
    if not isinstance(requested_metrics, Mapping):
        raise ValueError("worker requested-candidate metrics are incomplete")
    requested_candidate_metrics = requested_metrics.get("candidate")
    if not isinstance(requested_candidate_metrics, Mapping):
        raise ValueError("worker candidate metric payload is incomplete")
    protocol_sha256 = protocol.get("protocol_sha256")
    if not isinstance(protocol_sha256, str):
        raise ValueError("worker protocol receipt omitted its SHA-256")
    recomputed_qualification = _build_qualification(
        requested_candidate_metrics,
        candidate_resources,
        protocol_sha256=protocol_sha256,
    )
    if dict(qualification) != recomputed_qualification:
        raise ValueError("worker qualification was not recomputed")
    if (
        qualification.get("qualified") is not False
        or qualification.get("passed") is not False
        or qualification.get("compression_claim_authorized") is not False
        or qualification.get("compression_success_authorized") is not False
        or qualification.get("compression_success_claimed") is not False
    ):
        raise ValueError("worker qualification claim boundary is invalid")

    expected_events = [
        "lexical.ingress",
        *(f"mutation.{index}" for index in range(1, 37)),
        "terminal.rmsnorm",
        "lexical.egress",
    ]
    for condition in (exact_audit, candidate_audit):
        receipts = condition.get("event_receipts")
        if (
            not isinstance(receipts, list)
            or len(receipts) != condition.get("batch_count")
            or any(receipt != expected_events for receipt in receipts)
        ):
            raise ValueError("worker event receipts are invalid")
    digest = report.get("report_sha256")
    without_digest = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    if not isinstance(digest, str) or digest != _report_sha256(without_digest):
        raise ValueError("worker report SHA-256 is invalid")


def load_gemma3_layer_agnostic_worker_report(
    source: Path | str,
) -> dict[str, object]:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("layer-agnostic worker report is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("layer-agnostic worker report must be an object")
    _validate_report_semantics(value)
    return copy.deepcopy(dict(value))


def run_gemma3_layer_agnostic_whole_model_compression_worker(
    *,
    revision: str = DEFAULT_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    protocol_path: Path | str = DEFAULT_PROTOCOL_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
    lexical_rank: int = DEFAULT_LEXICAL_RANK,
    gram_chunk_rows: int = DEFAULT_GRAM_CHUNK_ROWS,
) -> dict[str, object]:
    """Fit, execute, and report one lexical-rank candidate."""

    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("worker output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a worker report")
    if revision != PINNED_MODEL_REVISION or model_id != PINNED_MODEL_ID:
        raise ValueError("worker requires the pinned Gemma checkpoint")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("worker requires deterministic CPU float32")
    if lexical_rank not in LEXICAL_RANK_LADDER:
        raise ValueError("lexical_rank is outside the frozen protocol ladder")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(gram_chunk_rows) is not int or gram_chunk_rows <= 0:
        raise ValueError("gram_chunk_rows must be positive")

    protocol_receipt = _protocol_receipt(protocol_path)
    _, evaluation_prompts, split = _load_prompt_split(Path(panel_path))
    if not split.get("prompt_disjoint") or not evaluation_prompts:
        raise RuntimeError("prompt-disjoint development evaluation is unavailable")
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma teacher")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    source_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if source_parameter_count != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT:
        raise RuntimeError("pinned source parameter inventory drifted")
    source_fingerprint_before = adapter.model_fingerprint()
    source_execution_before = adapter.execution_fingerprint()
    guard = _FrozenModelTensorGuard(model)
    batches = _tokenize_batches(
        tokenizer,
        evaluation_prompts,
        batch_size=batch_size,
        device=device,
    )

    _progress("compile rank-640 exact 39-event whole-model control")
    exact_executor = compile_gemma3_layer_agnostic_whole_model_executor(
        adapter,
        lexical_rank=PINNED_HIDDEN_WIDTH,
        gram_chunk_rows=gram_chunk_rows,
    )
    exact_resources = _resource_report(exact_executor)
    exact_parameters_before = exact_executor.deployed_scalar_count
    exact_manifest = exact_executor.architecture_manifest()
    _progress("evaluate and gate rank-640 exact control")
    exact_metrics, exact_audit = _evaluate_executor(
        adapter=adapter,
        executor=exact_executor,
        batches=batches,
    )
    _require_exact_control_parity(exact_metrics)
    exact_parameters_after = exact_executor.deployed_scalar_count
    if exact_parameters_after != exact_parameters_before:
        raise RuntimeError("rank-640 exact executor changed during evaluation")

    if lexical_rank == PINNED_HIDDEN_WIDTH:
        executor = exact_executor
        resources = exact_resources
        metrics = exact_metrics
        audit = exact_audit
        candidate_parameters_before = exact_parameters_before
        candidate_parameters_after = exact_parameters_after
        requested_reused_exact_control = True
    else:
        body = exact_executor.body
        terminal_norm = exact_executor.terminal_norm
        final_logit_softcap = exact_executor.final_logit_softcap
        del exact_executor
        gc.collect()
        _, _, lexical_weight, embed_scale = _source_lexical_modules(adapter)
        _progress(
            f"fit one packed shared tied lexical factor at rank {lexical_rank}"
        )
        lexical = fit_shared_tied_lexical_factor(
            lexical_weight,
            lexical_rank,
            embed_scale=embed_scale,
            chunk_rows=gram_chunk_rows,
        )
        executor = Gemma3LayerAgnosticWholeModelExecutor(
            lexical=lexical,
            body=body,
            terminal_norm=terminal_norm,
            final_logit_softcap=final_logit_softcap,
        )
        _assert_source_independence(
            model,
            {"compressed_lexical_exact_body": executor},
        )
        resources = _resource_report(executor)
        candidate_parameters_before = executor.deployed_scalar_count
        _progress("evaluate requested compressed lexical candidate")
        metrics, audit = _evaluate_executor(
            adapter=adapter,
            executor=executor,
            batches=batches,
        )
        candidate_parameters_after = executor.deployed_scalar_count
        if candidate_parameters_after != candidate_parameters_before:
            raise RuntimeError("requested executor changed during evaluation")
        requested_reused_exact_control = False

    guard.assert_unchanged()
    source_fingerprint_after = adapter.model_fingerprint()
    source_execution_after = adapter.execution_fingerprint()
    source_restored = (
        source_fingerprint_after == source_fingerprint_before
        and source_execution_after == source_execution_before
    )
    if not source_restored:
        raise RuntimeError("source Gemma changed during worker execution")
    candidate_metric_payload = metrics.get("candidate")
    if not isinstance(candidate_metric_payload, Mapping):
        raise RuntimeError("requested candidate metrics are incomplete")
    protocol_sha256 = protocol_receipt.get("protocol_sha256")
    if not isinstance(protocol_sha256, str):
        raise RuntimeError("protocol receipt omitted its SHA-256")
    qualification = _build_qualification(
        candidate_metric_payload,
        resources,
        protocol_sha256=protocol_sha256,
    )

    report: dict[str, object] = {
        "schema": GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_SCHEMA,
        "format_version": GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "protocol": protocol_receipt,
        "scientific_status": {
            "development_only": True,
            "prompt_disjoint_evaluation": True,
            "family_disjoint_evaluation": bool(split.get("family_disjoint")),
            "heldout_confirmation": False,
            "protocol_arm_executed": PROTOCOL_ARM_ID,
            "shared_tied_lexical_factor_executed": True,
            "exact_transformer_body_retained": True,
            "exact_terminal_rmsnorm_retained": True,
            "unrestricted_layer_agnostic_primary_executed": False,
            "cross_provenance_generator_sharing_executed": False,
            "body_compression_success_claimed": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "compression_success_authorized": False,
            "qualification_passed": False,
            "logical_body_mac_reduction_claimed": False,
            "latency_or_kernel_speed_claimed": False,
        },
        "model": _model_provenance(
            model,
            model_id=model_id,
            requested_revision=revision,
        ),
        "data": split,
        "compilation": {
            "protocol_arm_id": PROTOCOL_ARM_ID,
            "lexical_rank": lexical_rank,
            "lexical_rank_ladder": list(LEXICAL_RANK_LADDER),
            "lexical_factorization": "right_singular_basis_from_chunked_gram",
            "gram_chunk_rows": gram_chunk_rows,
            "embedding_scale_preserved": True,
            "tied_ingress_egress_storage": True,
            "exact_body_ordered_mutations": 36,
            "whole_model_ordered_events": 39,
            "runtime_native_layer_objects_present": False,
            "source_layer_ids_are_provenance_only": True,
            "unrestricted_primary_implemented": False,
            "rank_640_exact_control_manifest": exact_manifest,
            "requested_candidate_manifest": executor.architecture_manifest(),
            "requested_candidate_reused_exact_control": (
                requested_reused_exact_control
            ),
        },
        "metrics": {
            "rank_640_exact_control": exact_metrics,
            "requested_candidate": metrics,
        },
        "execution_audit": {
            "rank_640_exact_control": exact_audit,
            "requested_candidate": audit,
            "executor_integrity": {
                "rank_640_parameters_before": exact_parameters_before,
                "rank_640_parameters_after": exact_parameters_after,
                "rank_640_parameter_count_unchanged": True,
                "requested_parameters_before": candidate_parameters_before,
                "requested_parameters_after": candidate_parameters_after,
                "requested_parameter_count_unchanged": True,
            },
        },
        "resources": {
            "rank_640_exact_control": exact_resources,
            "requested_candidate": resources,
        },
        "qualification": qualification,
        "source_restoration": {
            "model_fingerprint_before": source_fingerprint_before,
            "model_fingerprint_after": source_fingerprint_after,
            "execution_fingerprint_before": source_execution_before,
            "execution_fingerprint_after": source_execution_after,
            "source_restored_exactly": True,
        },
    }
    report["report_sha256"] = _report_sha256(report)
    _validate_report_semantics(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    authenticated_report = load_gemma3_layer_agnostic_worker_report(
        destination
    )
    if _canonical_json_bytes(authenticated_report) != _canonical_json_bytes(
        report
    ):
        raise RuntimeError(
            "serialized worker report does not match the authenticated report"
        )
    _progress(f"published: {destination}")
    return authenticated_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered compressed-lexical/exact-body Gemma "
            "whole-model worker rung."
        )
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protocol-path", type=Path, default=DEFAULT_PROTOCOL_OUTPUT
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--lexical-rank",
        type=int,
        choices=LEXICAL_RANK_LADDER,
        default=DEFAULT_LEXICAL_RANK,
    )
    parser.add_argument(
        "--gram-chunk-rows", type=int, default=DEFAULT_GRAM_CHUNK_ROWS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_layer_agnostic_whole_model_compression_worker(
        revision=args.revision,
        output=args.output,
        protocol_path=args.protocol_path,
        panel_path=args.panel_path,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        lexical_rank=args.lexical_rank,
        gram_chunk_rows=args.gram_chunk_rows,
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_FORMAT_VERSION",
    "GEMMA3_LAYER_AGNOSTIC_WHOLE_MODEL_WORKER_SCHEMA",
    "Gemma3LayerAgnosticWholeModelExecutor",
    "LEXICAL_RANK_LADDER",
    "PROTOCOL_ARM_ID",
    "SharedTiedLexicalFactor",
    "WholeModelGraphExecution",
    "compile_gemma3_layer_agnostic_whole_model_executor",
    "compute_tied_lexical_gram",
    "fit_shared_tied_lexical_factor",
    "load_gemma3_layer_agnostic_worker_report",
    "main",
    "run_gemma3_layer_agnostic_whole_model_compression_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
