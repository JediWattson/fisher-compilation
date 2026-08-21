"""Concrete CPU runtime for the chart-native Gemma mode-neighbor experiment.

The orchestration module owns the sequential A/B/C firewall and all disk
artifacts.  This module is the deliberately narrow model-facing half: it
loads (or, in tests, receives) one exact Gemma control, tokenizes one already
opened batch, captures the 73 canonical event sites plus 19 auxiliary sites,
and returns only prompt-free common-carrier rows and authenticated hashes.

No full ``[B,L,V]`` logit tensor is formed by :meth:`capture_batch`.  The
full-vocabulary shifted NLL and terminal cotangent are evaluated by the frozen
chunked tied-readout kernel, then a one-use executor tape performs exactly one
VJP to all 92 requested sites.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext, SequenceInputOrigin
from .chart_energy_neighbor_graph import (
    CAUSAL_SEQUENCE_NATIVE_ABI,
    LEXICAL_INGRESS_NATIVE_ABI,
    LEXICAL_READOUT_NATIVE_ABI,
    TOKEN_LOCAL_NATIVE_ABI,
    NativeChartABI,
    canonical_carrier_comparison_abi,
)
from .chunked_tied_lexical_nll import (
    BatchObjectiveAxis,
    BatchObjectiveExample,
    ChunkedTiedLexicalNLLConfig,
    TiedReadoutWeightBinding,
    bind_tied_readout_weight,
    chunked_tied_lexical_nll_kernel_receipt,
    compute_chunked_tied_lexical_nll_vjp,
    run_dense_full_logit_parity_smoke,
    tensor_stream_sha256,
)
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_chart_mode_surfaces import (
    AttentionRouteStreamContext,
    NativeRoPESpec,
    TiedLexicalStateLineage,
    build_attention_postnorm_bridge_receipt,
    build_factorized_attention_route_aggregates,
    build_lexical_egress_from_terminal_score_gradient,
    build_lexical_ingress_chart_surface,
    build_mutation_carrier_chart_surface,
    factorized_attention_route_kernel_receipt,
)
from .gemma3_chart_native_neighbor_experiment import (
    CONFIRMATION_ROLE,
    DISCOVERY_ROLE,
    FIT_ROLE,
    CapturedBatch,
    CapturedExampleRows,
    CapturedModeRows,
    ExecutionPreflightReceipt,
    RuntimeExecutionReceipt,
    RuntimeQualificationReceipt,
    build_runtime_execution_receipt,
    build_runtime_qualification_receipt,
    derive_runtime_mode_lineage,
    derive_runtime_session_binding_sha256,
    runtime_mode_lineage_catalog_sha256,
)
from .gemma3_joint_cross_block_chart_executor import (
    Gemma3JointCrossBlockChartExecutor,
    JointGraphScoreGradientBundle,
)
from .gemma3_joint_cross_block_compiler import (
    Gemma3JointGraphCompilation,
    compile_gemma3_joint_exact_control,
)
from .streaming_shared_chart_energy_transport import (
    StreamingChartResourceConfig,
    streaming_chart_kernel_receipt,
)


RUNTIME_SCHEMA = "fisher_graph.gemma3_chart_native_neighbor.runtime.v1"
_CAPTURE_DOMAIN = b"fisher_graph:chart-native-runtime:capture:v1\0"
_EXECUTION_DOMAIN = b"fisher_graph:chart-native-runtime:execution:v1\0"
_MODE_STATE_DOMAIN = b"fisher_graph:chart-native-runtime:mode-state:v1\0"
_COMPILER_DOMAIN = b"fisher_graph:chart-native-runtime:compiler:v1\0"
_TOKEN_IDS_DOMAIN = b"fisher-graph:chart-neighbor-fresh:token-ids:v1\0"
_CHART_ROLES = (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE)
_EXPECTED_CANONICAL_GRADIENT_COUNT = 73
_EXPECTED_AUXILIARY_GRADIENT_COUNT = 19
_EXPECTED_MODE_COUNT = 38


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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _token_ids_sha256(ids: Sequence[int]) -> str:
    encoded = b"".join(int(value).to_bytes(4, "big") for value in ids)
    return hashlib.sha256(_TOKEN_IDS_DOMAIN + encoded).hexdigest()


def _module_state_sha256(module: nn.Module, *, identity: str) -> str:
    entries: list[dict[str, object]] = []
    for kind, named in (
        ("parameter", module.named_parameters(remove_duplicate=False)),
        ("buffer", module.named_buffers(remove_duplicate=False)),
    ):
        for name, value in named:
            hash_value = value.detach()
            entries.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(hash_value.shape),
                    # The shared streaming helper intentionally hashes row
                    # tensors; represent a scalar buffer as one authenticated
                    # row while retaining its original shape above.
                    "tensor_sha256": tensor_stream_sha256(
                        hash_value if hash_value.ndim else hash_value.reshape(1)
                    ),
                }
            )
    return _digest(
        _MODE_STATE_DOMAIN,
        {
            "identity": identity,
            "module_type": type(module).__qualname__,
            "tensors": entries,
        },
    )


def _adapter_fingerprint(adapter: object, method_name: str) -> str:
    callback = getattr(adapter, method_name, None)
    if callable(callback):
        return _require_sha256(callback(), label=method_name)
    module = getattr(adapter, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError(f"adapter omitted {method_name}() and module")
    return _module_state_sha256(module, identity=method_name)


def _adapter_parameter_count(adapter: object) -> int:
    module = getattr(adapter, "module", None)
    if not isinstance(module, nn.Module):
        raise TypeError("adapter omitted its source module")
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _compiler_runtime_payload(
    compilation: Gemma3JointGraphCompilation,
) -> dict[str, object]:
    executor = compilation.executor
    metadata_callback = getattr(executor.lexical, "metadata", None)
    lexical_metadata = (
        metadata_callback() if callable(metadata_callback) else None
    )
    if not isinstance(lexical_metadata, Mapping):
        raise TypeError("exact lexical factor omitted authenticated metadata")
    return {
        "compilation_metadata": compilation.metadata(),
        "runtime_lexical_contract": {
            "ingress_and_egress_share_the_same_storage": lexical_metadata.get(
                "ingress_and_egress_share_the_same_storage"
            ),
            "is_exact_copied_matrix": lexical_metadata.get(
                "is_exact_copied_matrix"
            ),
            "readout_bias_present": (
                getattr(executor.lexical, "bias", None) is not None
            ),
            "second_head_copy_present": lexical_metadata.get(
                "second_head_copy_present"
            ),
            "final_logit_softcap": executor.final_logit_softcap,
        },
    }


def _prepare_sequence(adapter: object, input_ids: Tensor) -> SequenceContext:
    callback = getattr(adapter, "prepare_sequence", None)
    if callable(callback):
        sequence = callback(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                "position_ids": torch.arange(
                    input_ids.shape[1], dtype=torch.long
                )[None, :].expand(input_ids.shape[0], -1),
            }
        )
        if not isinstance(sequence, SequenceContext):
            raise TypeError("adapter returned an invalid SequenceContext")
        return sequence
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    positions = torch.arange(input_ids.shape[1], dtype=torch.long)[None, :]
    positions = positions.expand(input_ids.shape[0], -1).contiguous()
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
    )


def _encode_ids(tokenizer: object, prompt: str) -> tuple[int, ...]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    encoder = getattr(backend if backend is not None else tokenizer, "encode", None)
    if not callable(encoder):
        raise TypeError("tokenizer must expose an exact encode() surface")
    encoded = encoder(prompt, add_special_tokens=True)
    ids = getattr(encoded, "ids", encoded)
    if not isinstance(ids, (list, tuple)) or not ids or any(
        type(value) is not int or value < 0 for value in ids
    ):
        raise TypeError("tokenizer encode() returned invalid token IDs")
    return tuple(ids)


def _native_abi_for_mode(mode: Mapping[str, object]) -> NativeChartABI:
    kind_by_mode = {
        "attention": CAUSAL_SEQUENCE_NATIVE_ABI,
        "mlp": TOKEN_LOCAL_NATIVE_ABI,
        "lexical_ingress": LEXICAL_INGRESS_NATIVE_ABI,
        "lexical_egress": LEXICAL_READOUT_NATIVE_ABI,
    }
    mode_kind = str(mode["mode_kind"])
    try:
        abi_kind = kind_by_mode[mode_kind]
    except KeyError as error:
        raise ValueError("runtime setup contains an unsupported mode kind") from error
    return NativeChartABI(
        abi_kind=abi_kind,
        abi_name=str(mode["native_ABI"]),
        abi_version=1,
        input_signature=f"{abi_kind}.input",
        output_signature=f"{abi_kind}.output",
        context_signature=f"{abi_kind}.context",
    )


@dataclass(frozen=True, slots=True)
class ChartNativeRuntimeCounters:
    """Prompt-free live counters; production authority uses the sealed receipt."""

    role_batch_counts: tuple[tuple[str, int], ...]
    chart_backbone_forward_count: int
    chart_terminal_cotangent_vjp_count: int
    dense_parity_smoke_count: int
    parity_backbone_forward_count: int
    parity_terminal_cotangent_vjp_count: int
    captured_example_count: int
    supplemental_route_receipt_count: int
    canonical_gradient_targets_per_vjp: int
    auxiliary_gradient_targets_per_vjp: int

    @property
    def gradient_targets_per_vjp(self) -> int:
        return (
            self.canonical_gradient_targets_per_vjp
            + self.auxiliary_gradient_targets_per_vjp
        )

    def payload(self) -> dict[str, object]:
        return {
            "role_batch_counts": [list(row) for row in self.role_batch_counts],
            "chart_backbone_forward_count": self.chart_backbone_forward_count,
            "chart_terminal_cotangent_vjp_count": (
                self.chart_terminal_cotangent_vjp_count
            ),
            "dense_parity_smoke_count": self.dense_parity_smoke_count,
            "parity_backbone_forward_count": self.parity_backbone_forward_count,
            "parity_terminal_cotangent_vjp_count": (
                self.parity_terminal_cotangent_vjp_count
            ),
            "captured_example_count": self.captured_example_count,
            "supplemental_route_receipt_count": (
                self.supplemental_route_receipt_count
            ),
            "canonical_gradient_targets_per_vjp": (
                self.canonical_gradient_targets_per_vjp
            ),
            "auxiliary_gradient_targets_per_vjp": (
                self.auxiliary_gradient_targets_per_vjp
            ),
            "gradient_targets_per_vjp": self.gradient_targets_per_vjp,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_activations": False,
            "contains_raw_gradients": False,
            "contains_logits": False,
        }


class Gemma3ChartNativeCaptureRuntime:
    """Concrete exact-control implementation of ``ChartNativeCaptureRuntime``."""

    def __init__(
        self,
        preflight: ExecutionPreflightReceipt | object,
        *,
        adapter: Gemma3CausalLMAdapter | object,
        tokenizer: object,
        compilation: Gemma3JointGraphCompilation | None = None,
    ) -> None:
        validate = getattr(preflight, "validate_receipt", None)
        if callable(validate):
            validate()
        binding = getattr(preflight, "binding", None)
        test_only = bool(
            getattr(binding, "test_only_allow_nonproduction_shape", False)
        )
        if compilation is not None and not test_only:
            raise ValueError(
                "injected exact compilation is restricted to nonproduction tests"
            )
        setup = getattr(preflight, "setup", None)
        if not isinstance(setup, Mapping):
            raise TypeError("runtime preflight omitted the frozen setup")
        source_model = setup.get("source_model")
        ontology = setup.get("logical_mode_ontology")
        if not isinstance(source_model, Mapping) or not isinstance(
            ontology, Mapping
        ):
            raise ValueError("runtime setup omitted model or mode ontology")
        setup_modes_raw = ontology.get("ordered_modes")
        if not isinstance(setup_modes_raw, list):
            raise ValueError("runtime setup mode catalog is invalid")
        self._setup_modes = tuple(setup_modes_raw)
        if (
            len(self._setup_modes) != _EXPECTED_MODE_COUNT
            or tuple(str(row["mode_id"]) for row in self._setup_modes)
            != (
                "mode.lexical.ingress",
                *(f"mode.mutation.{index}" for index in range(1, 37)),
                "mode.lexical.egress",
            )
        ):
            raise ValueError("runtime requires the exact ordered 38-mode ontology")

        self._preflight = preflight
        self._adapter = adapter
        self._tokenizer = tokenizer
        source_module = getattr(adapter, "module", None)
        if not isinstance(source_module, nn.Module):
            raise TypeError("runtime adapter must expose a source nn.Module")
        source_module.eval().requires_grad_(False)
        self._source_guard = _FrozenModelTensorGuard(source_module)
        self._source_model_before = _adapter_fingerprint(
            adapter, "model_fingerprint"
        )
        self._source_execution_before = _adapter_fingerprint(
            adapter, "execution_fingerprint"
        )
        self._source_parameter_count = _adapter_parameter_count(adapter)
        if not test_only:
            expected = {
                "model": source_model.get("source_model_fingerprint_sha256"),
                "execution": source_model.get(
                    "source_execution_fingerprint_sha256"
                ),
                "parameters": source_model.get("parameter_count"),
                "hidden": source_model.get("hidden_width"),
            }
            if (
                self._source_model_before != expected["model"]
                or self._source_execution_before != expected["execution"]
                or self._source_parameter_count != expected["parameters"]
                or expected["hidden"] != 640
            ):
                raise ValueError("loaded source model differs from frozen setup")

        if compilation is None:
            if not isinstance(adapter, Gemma3CausalLMAdapter):
                raise TypeError("production compilation requires Gemma3CausalLMAdapter")
            compilation = compile_gemma3_joint_exact_control(
                adapter,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        if (
            not isinstance(compilation, Gemma3JointGraphCompilation)
            or compilation.compilation_kind != "exact_39_event_control"
        ):
            raise TypeError("runtime requires an exact 39-event compilation")
        executor = compilation.executor
        if not isinstance(executor, Gemma3JointCrossBlockChartExecutor):
            raise TypeError("runtime compilation omitted the joint executor")
        if executor.hidden_width != int(source_model["hidden_width"]):
            raise ValueError("compiled hidden width differs from setup")
        for value in (*executor.parameters(), *executor.buffers()):
            if value.device.type != "cpu" or (
                value.is_floating_point() and value.dtype != torch.float32
            ):
                raise ValueError("runtime requires exact CPU float32 execution")
        executor.eval().requires_grad_(False)
        compiler_payload = _compiler_runtime_payload(compilation)
        lexical_contract = compiler_payload["runtime_lexical_contract"]
        if lexical_contract != {
            "ingress_and_egress_share_the_same_storage": True,
            "is_exact_copied_matrix": True,
            "readout_bias_present": False,
            "second_head_copy_present": False,
            "final_logit_softcap": executor.final_logit_softcap,
        }:
            raise ValueError("exact tied, bias-free lexical contract drifted")
        compilation_metadata = compiler_payload["compilation_metadata"]
        if not isinstance(compilation_metadata, Mapping):
            raise TypeError("exact compiler metadata is invalid")
        if not test_only and (
            compilation.source_model_parameter_count
            != self._source_parameter_count
            or compilation.source_model_fingerprint
            != self._source_model_before
            or compilation.source_independence_audit is None
            or compilation_metadata["exact_parameter_inventory_parity"]
            is not True
        ):
            raise RuntimeError("production exact-control compilation audit drifted")
        self._source_guard.assert_unchanged()
        self._source_model_after_compilation = _adapter_fingerprint(
            adapter, "model_fingerprint"
        )
        self._source_execution_after_compilation = _adapter_fingerprint(
            adapter, "execution_fingerprint"
        )
        if (
            self._source_model_after_compilation != self._source_model_before
            or self._source_execution_after_compilation
            != self._source_execution_before
        ):
            raise RuntimeError("exact compilation mutated the source model")
        self._compilation = compilation
        self._executor = executor
        self._executor_guard = _FrozenModelTensorGuard(executor)
        self.hidden_width = executor.hidden_width
        self._compiled_execution_fingerprint = _module_state_sha256(
            executor,
            identity="compiled_exact_39_event_control",
        )
        self._compiler_receipt_sha256 = _digest(
            _COMPILER_DOMAIN,
            compiler_payload,
        )

        lexical_weight = getattr(executor.lexical, "exact_weight", None)
        if (
            not isinstance(lexical_weight, Tensor)
            or lexical_weight.device.type != "cpu"
            or lexical_weight.dtype != torch.float32
            or not lexical_weight.is_contiguous()
            or lexical_weight.requires_grad
        ):
            raise ValueError("exact runtime omitted its CPU float32 tied weight")
        tied_state_sha = tensor_stream_sha256(lexical_weight)
        embed_scale_tensor = getattr(executor.lexical, "embed_scale", None)
        if (
            not isinstance(embed_scale_tensor, Tensor)
            or embed_scale_tensor.numel() != 1
        ):
            raise TypeError("exact lexical factor omitted embedding scale")
        embed_scale = float(embed_scale_tensor)
        self._lexical_lineage = TiedLexicalStateLineage(
            tied_state_sha256=tied_state_sha,
            vocabulary_size=int(lexical_weight.shape[0]),
            hidden_width=int(lexical_weight.shape[1]),
            embedding_scale=embed_scale,
        )
        self._tied_weight: TiedReadoutWeightBinding = bind_tied_readout_weight(
            lexical_weight,
            readout_state_sha256=tied_state_sha,
            tied_lexical_lineage_sha256=self._lexical_lineage.sha256,
        )
        self._terminal_norm_state_sha256 = _module_state_sha256(
            executor.terminal_norm,
            identity="terminal.rmsnorm",
        )

        native_abis: dict[str, NativeChartABI] = {}
        state_fingerprints_by_mode: dict[str, str] = {}
        mutation_by_id = {
            mutation.event_spec.event_id: mutation
            for mutation in executor.mutations
        }
        for mode in self._setup_modes:
            mode_id = str(mode["mode_id"])
            mode_kind = str(mode["mode_kind"])
            event_id = str(mode["event_id"])
            native = _native_abi_for_mode(mode)
            native_abis[mode_id] = native
            if mode_kind.startswith("lexical_"):
                state_sha = tied_state_sha
            else:
                mutation = mutation_by_id.get(event_id)
                if mutation is None:
                    raise ValueError("compiled mutation inventory differs from setup")
                state_sha = _module_state_sha256(mutation, identity=event_id)
            state_fingerprints_by_mode[mode_id] = state_sha
        self.mode_lineage = derive_runtime_mode_lineage(
            setup_modes=self._setup_modes,
            state_fingerprints_by_mode=state_fingerprints_by_mode,
            hidden_width=self.hidden_width,
        )
        self._native_abis = native_abis
        self._lineage_by_mode = {row.mode_id: row for row in self.mode_lineage}
        resources = StreamingChartResourceConfig(
            max_resident_bytes=(
                preflight.binding.storage_budget.maximum_resident_memory_bytes
            )
        )
        self._lexical_kernel_receipt_sha256 = str(
            chunked_tied_lexical_nll_kernel_receipt()["receipt_sha256"]
        )
        self._streaming_kernel_receipt_sha256 = str(
            streaming_chart_kernel_receipt(resources)["receipt_sha256"]
        )
        self._factorized_route_kernel_receipt_sha256 = str(
            factorized_attention_route_kernel_receipt()["receipt_sha256"]
        )
        self._runtime_qualification = build_runtime_qualification_receipt(
            schema=(
                "fisher_graph.gemma3_chart_native_neighbor."
                "runtime_qualification.v1"
            ),
            format_version=1,
            model_id=str(source_model["model_id"]),
            model_revision=str(source_model["model_revision"]),
            source_model_fingerprint_before_compilation_sha256=(
                self._source_model_before
            ),
            source_model_fingerprint_after_compilation_sha256=(
                self._source_model_after_compilation
            ),
            source_execution_fingerprint_before_compilation_sha256=(
                self._source_execution_before
            ),
            source_execution_fingerprint_after_compilation_sha256=(
                self._source_execution_after_compilation
            ),
            source_model_mutated_during_compilation=False,
            source_parameter_count=self._source_parameter_count,
            hidden_width=self.hidden_width,
            compiled_execution_fingerprint_sha256=(
                self._compiled_execution_fingerprint
            ),
            compiler_receipt_sha256=self._compiler_receipt_sha256,
            tied_readout_binding_receipt_sha256=(
                self._tied_weight.receipt.receipt_sha256
            ),
            tied_lexical_lineage_sha256=self._lexical_lineage.sha256,
            final_logit_softcap=self._executor.final_logit_softcap,
            mode_lineage=self.mode_lineage,
            mode_lineage_catalog_sha256=(
                runtime_mode_lineage_catalog_sha256(self.mode_lineage)
            ),
            lexical_kernel_receipt_sha256=(
                self._lexical_kernel_receipt_sha256
            ),
            streaming_kernel_receipt_sha256=(
                self._streaming_kernel_receipt_sha256
            ),
            factorized_route_kernel_receipt_sha256=(
                self._factorized_route_kernel_receipt_sha256
            ),
            contains_prompt_text=False,
            contains_token_ids=False,
            contains_raw_activations=False,
            contains_raw_gradients=False,
            contains_logits=False,
        )
        preflight_sha = getattr(preflight, "receipt_sha256", "0" * 64)
        _require_sha256(preflight_sha, label="preflight receipt")
        self.runtime_session_binding_sha256 = (
            derive_runtime_session_binding_sha256(
                preflight_receipt_sha256=preflight_sha,
                runtime_qualification_receipt_sha256=(
                    self._runtime_qualification.qualification_receipt_sha256
                ),
                source_model_fingerprint_sha256=(
                    self._source_model_after_compilation
                ),
                source_execution_fingerprint_sha256=(
                    self._source_execution_after_compilation
                ),
                compiled_execution_fingerprint_sha256=(
                    self._compiled_execution_fingerprint
                ),
                tied_readout_binding_receipt_sha256=(
                    self._tied_weight.receipt.receipt_sha256
                ),
                mode_lineage=self.mode_lineage,
            )
        )
        self._role_batches = {role: 0 for role in _CHART_ROLES}
        self._chart_forwards = 0
        self._chart_vjps = 0
        self._parity_smokes = 0
        self._parity_forwards = 0
        self._parity_vjps = 0
        self._captured_examples = 0
        self._route_receipts = 0
        self._dense_parity_receipt_sha256: str | None = None
        self._execution_started = False
        self._closed = False
        self._test_only = test_only
        self._assert_open_and_unchanged()

    @property
    def counters(self) -> ChartNativeRuntimeCounters:
        return ChartNativeRuntimeCounters(
            role_batch_counts=tuple(
                (role, self._role_batches[role]) for role in _CHART_ROLES
            ),
            chart_backbone_forward_count=self._chart_forwards,
            chart_terminal_cotangent_vjp_count=self._chart_vjps,
            dense_parity_smoke_count=self._parity_smokes,
            parity_backbone_forward_count=self._parity_forwards,
            parity_terminal_cotangent_vjp_count=self._parity_vjps,
            captured_example_count=self._captured_examples,
            supplemental_route_receipt_count=self._route_receipts,
            canonical_gradient_targets_per_vjp=(
                _EXPECTED_CANONICAL_GRADIENT_COUNT
            ),
            auxiliary_gradient_targets_per_vjp=(
                _EXPECTED_AUXILIARY_GRADIENT_COUNT
            ),
        )

    def _assert_open_and_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("chart-native runtime is closed")
        self._source_guard.assert_unchanged()
        self._executor_guard.assert_unchanged()
        self._tied_weight.validate_unchanged()
        if (
            _adapter_fingerprint(self._adapter, "execution_fingerprint")
            != self._source_execution_before
        ):
            raise RuntimeError("source execution fingerprint changed during runtime")
        if (
            _digest(
                _COMPILER_DOMAIN,
                _compiler_runtime_payload(self._compilation),
            )
            != self._compiler_receipt_sha256
        ):
            raise RuntimeError("compiled execution semantics changed during runtime")

    def _members_and_tokens(
        self,
        *,
        role: str,
        raw_examples: Sequence[Mapping[str, object]],
        role_manifest: Mapping[str, object],
    ) -> tuple[tuple[Mapping[str, object], ...], Tensor]:
        if role not in _CHART_ROLES:
            raise ValueError("chart capture role must be A, B, or C")
        if role_manifest.get("role") != role:
            raise ValueError("role manifest identity drifted")
        ordered = role_manifest.get("ordered_members")
        if not isinstance(ordered, list):
            raise TypeError("role manifest ordered_members is invalid")
        by_id = {
            str(member["example_id"]): member
            for member in ordered
            if isinstance(member, Mapping)
        }
        examples = tuple(raw_examples)
        if not examples:
            raise ValueError("capture batch cannot be empty")
        members: list[Mapping[str, object]] = []
        encoded_rows: list[tuple[int, ...]] = []
        length: int | None = None
        for raw in examples:
            if not isinstance(raw, Mapping):
                raise TypeError("raw example must be a mapping")
            example_id = str(raw.get("example_id", ""))
            member = by_id.get(example_id)
            if member is None:
                raise ValueError("raw example is absent from the frozen role")
            if (
                raw.get("family_id") != member.get("family_id")
                or raw.get("target_token_count")
                != member.get("target_token_count")
            ):
                raise ValueError("raw example identity differs from manifest")
            prompt = raw.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("opened example prompt is invalid")
            ids = _encode_ids(self._tokenizer, prompt)
            target = int(member["target_token_count"])
            if len(ids) != target or ids[0] != 2:
                raise ValueError("tokenizer length or pinned BOS drifted")
            if _token_ids_sha256(ids) != member.get("token_ids_sha256"):
                raise ValueError("tokenizer ID stream differs from frozen manifest")
            if length is None:
                length = target
            elif target != length:
                raise ValueError("capture batch must have one exact token length")
            members.append(member)
            encoded_rows.append(ids)
        return tuple(members), torch.tensor(encoded_rows, dtype=torch.long)

    def runtime_qualification_receipt(self) -> RuntimeQualificationReceipt:
        """Return the immutable prompt-free lineage before any panel work."""

        self._assert_open_and_unchanged()
        if self._execution_started:
            raise RuntimeError(
                "runtime qualification is unavailable after panel execution starts"
            )
        return self._runtime_qualification

    @staticmethod
    def _objective_axis(
        *,
        role: str,
        members: Sequence[Mapping[str, object]],
        role_manifest: Mapping[str, object],
    ) -> BatchObjectiveAxis:
        return BatchObjectiveAxis(
            role=role,
            examples=tuple(
                BatchObjectiveExample(
                    family_id=str(member["family_id"]),
                    example_id=str(member["example_id"]),
                    target_token_count=int(member["target_token_count"]),
                    frozen_objective_row_stream_sha256=str(
                        member["objective_row_stream_sha256"]
                    ),
                )
                for member in members
            ),
            role_score_objective_receipt_sha256=str(
                role_manifest["score_objective_receipt_sha256"]
            ),
            role_ordered_objective_row_catalog_sha256=str(
                role_manifest["ordered_objective_row_catalog_sha256"]
            ),
        )

    def _run_backbone(
        self,
        *,
        input_ids: Tensor,
        capture_routes: bool,
    ) -> tuple[
        object,
        dict[str, Tensor],
        dict[str, object],
        SequenceContext,
    ]:
        observations: dict[str, Tensor] = {}
        routes: dict[str, object] = {}

        def observer(*, event_id: str, role: str, values: Tensor) -> None:
            site_id = f"{event_id}.{role}"
            if site_id in observations:
                raise RuntimeError("executor emitted one chart site twice")
            observations[site_id] = values

        def route_observer(
            *, event_id: str, block_ordinal: int, snapshot: object
        ) -> None:
            if event_id in routes or block_ordinal != len(routes):
                raise RuntimeError("attention route factor order drifted")
            routes[event_id] = snapshot

        sequence = _prepare_sequence(self._adapter, input_ids)
        if not bool(sequence.query_valid_mask.all()) or not bool(
            sequence.key_valid_mask.all()
        ):
            raise ValueError("chart batches must be exact-length and all-valid")
        with torch.enable_grad():
            observed = self._executor.forward_backbone_with_observer(
                input_ids,
                sequence,
                observer=observer,
                attention_route_factor_observer=(
                    route_observer if capture_routes else None
                ),
            )
        if len(observations) != _EXPECTED_CANONICAL_GRADIENT_COUNT:
            raise RuntimeError("executor did not expose all 73 canonical sites")
        if capture_routes and len(routes) != 18:
            raise RuntimeError("executor did not expose all 18 attention factors")
        return observed, observations, routes, sequence

    @staticmethod
    def _validate_gradient_bundle(bundle: JointGraphScoreGradientBundle) -> None:
        if (
            not isinstance(bundle, JointGraphScoreGradientBundle)
            or len(bundle.site_gradients) != _EXPECTED_CANONICAL_GRADIENT_COUNT
            or len(bundle.auxiliary_gradients)
            != _EXPECTED_AUXILIARY_GRADIENT_COUNT
            or len(bundle.attention_operator_output_gradients) != 18
        ):
            raise RuntimeError("exact score-gradient inventory did not close")

    def run_dense_lexical_parity_smoke(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        raw_example: Mapping[str, object],
        role_manifest: Mapping[str, object],
    ) -> str:
        self._assert_open_and_unchanged()
        if self._parity_smokes:
            raise RuntimeError("dense lexical parity smoke may run only once")
        if preflight is not self._preflight:
            raise ValueError("runtime preflight object changed")
        self._execution_started = True
        members, input_ids = self._members_and_tokens(
            role=FIT_ROLE,
            raw_examples=(raw_example,),
            role_manifest=role_manifest,
        )
        axis = self._objective_axis(
            role=FIT_ROLE,
            members=members,
            role_manifest=role_manifest,
        )
        observed, _values, _routes, sequence = self._run_backbone(
            input_ids=input_ids,
            capture_routes=False,
        )
        result = compute_chunked_tied_lexical_nll_vjp(
            terminal_normalized_output=observed.execution.normalized_output,
            input_token_ids=input_ids,
            valid_token_mask=sequence.query_valid_mask,
            batch_axis=axis,
            tied_readout_weight=self._tied_weight,
            logit_softcap=self._executor.final_logit_softcap,
            config=ChunkedTiedLexicalNLLConfig(),
        )
        with torch.enable_grad():
            parity = run_dense_full_logit_parity_smoke(
                source_result=result,
                terminal_normalized_output=(
                    observed.execution.normalized_output
                ),
                input_token_ids=input_ids,
                valid_token_mask=sequence.query_valid_mask,
                batch_axis=axis,
                tied_readout_weight=self._tied_weight,
                logit_softcap=self._executor.final_logit_softcap,
                config=ChunkedTiedLexicalNLLConfig(),
            )
        bundle = observed.gradient_tape.score_gradients_from_terminal_normalized_output(
            result.terminal_gradient
        )
        self._validate_gradient_bundle(bundle)
        self._parity_smokes = 1
        self._parity_forwards = 1
        self._parity_vjps = 1
        self._dense_parity_receipt_sha256 = parity.receipt_sha256
        self._assert_open_and_unchanged()
        return parity.receipt_sha256

    def capture_batch(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        role: str,
        raw_examples: Sequence[Mapping[str, object]],
        role_manifest: Mapping[str, object],
    ) -> CapturedBatch:
        self._assert_open_and_unchanged()
        if preflight is not self._preflight:
            raise ValueError("runtime preflight object changed")
        if self._dense_parity_receipt_sha256 is None:
            raise RuntimeError("A dense lexical parity smoke must run first")
        role_index = _CHART_ROLES.index(role) if role in _CHART_ROLES else -1
        if role_index < 0:
            raise ValueError("chart capture role must be A, B, or C")
        if (
            any(
                self._role_batches[item] != 45
                for item in _CHART_ROLES[:role_index]
            )
            or any(
                self._role_batches[item] != 0
                for item in _CHART_ROLES[role_index + 1 :]
            )
            or self._role_batches[role] >= 45
        ):
            raise RuntimeError(
                "A/B/C runtime capture order or 45-batch ledger drifted"
            )
        members, input_ids = self._members_and_tokens(
            role=role,
            raw_examples=raw_examples,
            role_manifest=role_manifest,
        )
        if not self._test_only:
            expected_batch_size = self._preflight.binding.schedule.batch_size_for(
                int(input_ids.shape[1])
            )
            if int(input_ids.shape[0]) != expected_batch_size:
                raise ValueError(
                    "capture batch differs from the frozen length schedule"
                )
        axis = self._objective_axis(
            role=role,
            members=members,
            role_manifest=role_manifest,
        )
        observed, observations, route_factors, sequence = self._run_backbone(
            input_ids=input_ids,
            capture_routes=True,
        )
        nll = compute_chunked_tied_lexical_nll_vjp(
            terminal_normalized_output=observed.execution.normalized_output,
            input_token_ids=input_ids,
            valid_token_mask=sequence.query_valid_mask,
            batch_axis=axis,
            tied_readout_weight=self._tied_weight,
            logit_softcap=self._executor.final_logit_softcap,
            config=ChunkedTiedLexicalNLLConfig(),
        )
        bundle = observed.gradient_tape.score_gradients_from_terminal_normalized_output(
            nll.terminal_gradient
        )
        self._validate_gradient_bundle(bundle)
        execution_sha = _digest(
            _EXECUTION_DOMAIN,
            {
                "runtime_session_binding_sha256": (
                    self.runtime_session_binding_sha256
                ),
                "batch_objective_axis_sha256": axis.sha256,
                "compiled_execution_fingerprint_sha256": (
                    self._compiled_execution_fingerprint
                ),
                "event_ids": list(observed.execution.event_ids),
                "mutation_chart_input_sha256s": list(
                    observed.execution.mutation_chart_input_sha256s
                ),
                "mutation_delta_sha256s": list(
                    observed.execution.mutation_delta_sha256s
                ),
                "attention_operator_output_catalog_sha256": (
                    observed.execution.attention_operator_output_catalog_sha256
                ),
                "lexical_nll_receipt_sha256": nll.receipt.receipt_sha256,
                "canonical_gradient_count": len(bundle.site_gradients),
                "auxiliary_gradient_count": len(bundle.auxiliary_gradients),
                "full_B_L_V_logits_materialized": False,
            },
        )
        examples = tuple(str(member["example_id"]) for member in members)
        contexts = tuple(
            str(member["canonical_prompt_sha256"]) for member in members
        )
        query_valid = sequence.query_valid_mask.detach().clone()
        query_valid[:, -1] = False
        key_valid = sequence.key_valid_mask
        positions = sequence.logical_positions
        comparison_sha = canonical_carrier_comparison_abi().sha256
        score_sha = str(role_manifest["score_objective_receipt_sha256"])

        surfaces: dict[str, object] = {}
        ingress_lineage = self._lineage_by_mode["mode.lexical.ingress"]
        ingress = build_lexical_ingress_chart_surface(
            native_abi_sha256=self._native_abis[
                "mode.lexical.ingress"
            ].sha256,
            comparison_abi_sha256=comparison_sha,
            static_comparison_adapter_receipt_sha256=(
                ingress_lineage.comparison_adapter_receipt_sha256
            ),
            execution_fingerprint_sha256=execution_sha,
            score_objective_receipt_sha256=score_sha,
            ingress_lookup_receipt_sha256=(
                self._tied_weight.receipt.receipt_sha256
            ),
            lineage=self._lexical_lineage,
            example_ids=examples,
            context_fingerprints=contexts,
            valid_mask=query_valid,
            logical_positions=positions,
            ingress_carrier_values=observations["lexical.ingress.output"],
            ingress_carrier_score_gradients=bundle.site_gradients[
                "lexical.ingress.output"
            ],
        )
        surfaces["mode.lexical.ingress"] = ingress

        route_by_mode_and_example: dict[
            tuple[str, str],
            tuple[str, str, str],
        ] = {}
        for mutation_index in range(1, 37):
            event_id = f"mutation.{mutation_index}"
            mode_id = f"mode.{event_id}"
            lineage = self._lineage_by_mode[mode_id]
            surface = build_mutation_carrier_chart_surface(
                mode_id=mode_id,
                native_abi_sha256=self._native_abis[mode_id].sha256,
                comparison_abi_sha256=comparison_sha,
                static_comparison_adapter_receipt_sha256=(
                    lineage.comparison_adapter_receipt_sha256
                ),
                source_state_sha256=lineage.state_fingerprint_sha256,
                execution_fingerprint_sha256=execution_sha,
                score_objective_receipt_sha256=score_sha,
                context_site_id=f"{event_id}.input",
                contribution_site_id=f"{event_id}.output",
                example_ids=examples,
                context_fingerprints=contexts,
                valid_mask=query_valid,
                logical_positions=positions,
                mutation_context_values=observations[f"{event_id}.input"],
                mutation_output_values=observations[f"{event_id}.output"],
                mutation_output_score_gradients=bundle.site_gradients[
                    f"{event_id}.output"
                ],
            )
            surfaces[mode_id] = surface
            if mutation_index % 2:
                operator_site = f"{event_id}.operator_output"
                operator_output = observed.execution.attention_operator_outputs[
                    operator_site
                ]
                operator_gradient = (
                    bundle.attention_operator_output_gradients[operator_site]
                )
                bridge = build_attention_postnorm_bridge_receipt(
                    operator_output=operator_output,
                    operator_output_score_gradient=operator_gradient,
                    postnorm_delta=observations[f"{event_id}.output"],
                    postnorm_state_sha256=lineage.state_fingerprint_sha256,
                    postnorm_config_sha256=lineage.site_placement_sha256,
                    execution_fingerprint_sha256=execution_sha,
                )
                snapshot = route_factors[event_id]
                route_context = AttentionRouteStreamContext(
                    example_ids=examples,
                    context_fingerprints=contexts,
                    query_valid_mask=query_valid,
                    key_valid_mask=key_valid,
                    query_logical_positions=positions,
                    key_logical_positions=sequence.key_logical_positions,
                    query_head_count=int(snapshot.query_head_count),
                    window_size=snapshot.window_size,
                    rope=NativeRoPESpec(
                        theta=float(snapshot.rope_theta),
                        rotary_dimension=int(snapshot.head_dimension),
                        scaling_kind=str(snapshot.rope_scaling_kind),
                        scaling_factor=snapshot.rope_scaling_factor,
                    ),
                )
                receipts = build_factorized_attention_route_aggregates(
                    carrier=surface.carrier,
                    block_ordinal=(mutation_index - 1) // 2,
                    context=route_context,
                    postnorm_bridge=bridge,
                    route_factors=snapshot,
                    attention_operator_output=operator_output,
                    attention_operator_output_score_gradients=(
                        operator_gradient
                    ),
                )
                if tuple(receipt.example_id for receipt in receipts) != examples:
                    raise RuntimeError("route receipt example order drifted")
                for receipt in receipts:
                    catalog_sha, receipt_sha = receipt.supplemental_evidence()
                    route_by_mode_and_example[(mode_id, receipt.example_id)] = (
                        catalog_sha,
                        receipt_sha,
                        receipt.probability_simplex_audit_sha256,
                    )

        egress_lineage = self._lineage_by_mode["mode.lexical.egress"]
        egress = build_lexical_egress_from_terminal_score_gradient(
            native_abi_sha256=self._native_abis["mode.lexical.egress"].sha256,
            comparison_abi_sha256=comparison_sha,
            static_comparison_adapter_receipt_sha256=(
                egress_lineage.comparison_adapter_receipt_sha256
            ),
            execution_fingerprint_sha256=execution_sha,
            lineage=self._lexical_lineage,
            example_ids=examples,
            context_fingerprints=contexts,
            valid_mask=query_valid,
            logical_positions=positions,
            terminal_carrier=observed.execution.normalized_output,
            terminal_carrier_score_gradient=(
                bundle.terminal_normalized_output_gradient
            ),
            objective_receipt_sha256=score_sha,
            terminal_norm_state_sha256=self._terminal_norm_state_sha256,
            readout_state_sha256=self._tied_weight.receipt.tensor_sha256,
            readout_mapping_receipt_sha256=(
                self._tied_weight.receipt.receipt_sha256
            ),
            terminal_score_gradient_receipt_sha256=(
                nll.receipt.receipt_sha256
            ),
            analytic_parity_receipt_sha256=(
                self._dense_parity_receipt_sha256
            ),
            final_logit_softcap=self._executor.final_logit_softcap,
        )
        surfaces["mode.lexical.egress"] = egress
        mode_ids = tuple(row.mode_id for row in self.mode_lineage)
        if tuple(surfaces) != mode_ids:
            raise RuntimeError("captured surface order differs from 38-mode ontology")

        captured_examples: list[CapturedExampleRows] = []
        row_start = 0
        for member in members:
            example_id = str(member["example_id"])
            family_id = str(member["family_id"])
            token_count = int(member["target_token_count"])
            row_stop = row_start + token_count - 1
            captured_modes: list[CapturedModeRows] = []
            surface_receipts: list[dict[str, object]] = []
            for mode_id in mode_ids:
                surface = surfaces[mode_id]
                carrier = surface.carrier
                if carrier.row_keys[row_start:row_stop] != tuple(
                    (example_id, query) for query in range(token_count - 1)
                ):
                    raise RuntimeError("surface q/t carrier row order drifted")
                context_values = getattr(
                    surface, "context_values", carrier.carrier_values
                )[row_start:row_stop].contiguous()
                route = route_by_mode_and_example.get((mode_id, example_id))
                captured_modes.append(
                    CapturedModeRows(
                        mode_id=mode_id,
                        context_values=context_values,
                        comparison_values=carrier.carrier_values[
                            row_start:row_stop
                        ].contiguous(),
                        generalized_contributions=carrier.coordinates[
                            row_start:row_stop
                        ].contiguous(),
                        supplemental_route_catalog_sha256=(
                            None if route is None else route[0]
                        ),
                        supplemental_route_receipt_sha256=(
                            None if route is None else route[1]
                        ),
                        supplemental_route_simplex_audit_sha256=(
                            None if route is None else route[2]
                        ),
                    )
                )
                surface_receipts.append(
                    {
                        "mode_id": mode_id,
                        "surface_sha256": surface.sha256,
                        "supplemental_route_catalog_sha256": (
                            None if route is None else route[0]
                        ),
                        "supplemental_route_receipt_sha256": (
                            None if route is None else route[1]
                        ),
                        "supplemental_route_simplex_audit_sha256": (
                            None if route is None else route[2]
                        ),
                    }
                )
            capture_sha = _digest(
                _CAPTURE_DOMAIN,
                {
                    "runtime_session_binding_sha256": (
                        self.runtime_session_binding_sha256
                    ),
                    "role": role,
                    "family_id": family_id,
                    "example_id": example_id,
                    "token_count": token_count,
                    "objective_row_stream_sha256": member[
                        "objective_row_stream_sha256"
                    ],
                    "context_fingerprint_sha256": member[
                        "canonical_prompt_sha256"
                    ],
                    "execution_receipt_sha256": execution_sha,
                    "lexical_nll_receipt_sha256": (
                        nll.receipt.receipt_sha256
                    ),
                    "surfaces": surface_receipts,
                    "contains_prompt_text": False,
                    "contains_token_ids": False,
                    "contains_raw_outputs": False,
                },
            )
            captured_examples.append(
                CapturedExampleRows(
                    family_id=family_id,
                    example_id=example_id,
                    token_count=token_count,
                    row_keys=tuple(
                        (family_id, example_id, query, query + 1)
                        for query in range(token_count - 1)
                    ),
                    modes=tuple(captured_modes),
                    context_fingerprint_sha256=str(
                        member["canonical_prompt_sha256"]
                    ),
                    capture_receipt_sha256=capture_sha,
                )
            )
            row_start = row_stop
        if row_start != axis.prediction_row_count:
            raise RuntimeError("captured example row partition did not close")

        self._role_batches[role] += 1
        self._chart_forwards += 1
        self._chart_vjps += 1
        self._captured_examples += len(captured_examples)
        self._route_receipts += len(route_by_mode_and_example)
        self._assert_open_and_unchanged()
        return CapturedBatch(
            role=role,
            examples=tuple(captured_examples),
            lexical_nll_receipt_sha256=nll.receipt.receipt_sha256,
            execution_receipt_sha256=execution_sha,
        )

    def response_validate(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        raw_corpus: Mapping[str, object],
        role_manifest: Mapping[str, object],
        confirmed_edges: Sequence[Sequence[str]],
    ) -> Mapping[str, object]:
        del preflight, raw_corpus, role_manifest, confirmed_edges
        raise RuntimeError(
            "D response validation is unsupported in chart-native v1; "
            "no frozen exact response protocol is available"
        )

    def runtime_execution_receipt(self) -> RuntimeExecutionReceipt:
        """Seal 135 chart captures plus the one parity execution after C."""

        self._assert_open_and_unchanged()
        if self._test_only:
            raise RuntimeError(
                "production RuntimeExecutionReceipt is unavailable to test-only shapes"
            )
        source_model_after = _adapter_fingerprint(
            self._adapter, "model_fingerprint"
        )
        source_execution_after = _adapter_fingerprint(
            self._adapter, "execution_fingerprint"
        )
        source_model = self._preflight.setup["source_model"]
        return build_runtime_execution_receipt(
            model_id=str(source_model["model_id"]),
            model_revision=str(source_model["model_revision"]),
            runtime_qualification_receipt_sha256=(
                self._runtime_qualification.qualification_receipt_sha256
            ),
            runtime_session_binding_sha256=self.runtime_session_binding_sha256,
            source_model_fingerprint_before_sha256=self._source_model_before,
            source_model_fingerprint_after_sha256=source_model_after,
            source_execution_fingerprint_before_sha256=(
                self._source_execution_before
            ),
            source_execution_fingerprint_after_sha256=(
                source_execution_after
            ),
            compiled_execution_fingerprint_sha256=(
                self._compiled_execution_fingerprint
            ),
            source_model_mutated=bool(
                source_model_after != self._source_model_before
                or source_execution_after != self._source_execution_before
            ),
            source_parameter_count=self._source_parameter_count,
            hidden_width=self.hidden_width,
            mode_lineage=self.mode_lineage,
            tied_readout_binding_receipt_sha256=(
                self._tied_weight.receipt.receipt_sha256
            ),
            tied_lexical_lineage_sha256=self._lexical_lineage.sha256,
            final_logit_softcap=self._executor.final_logit_softcap,
            compiler_receipt_sha256=self._compiler_receipt_sha256,
            lexical_kernel_receipt_sha256=(
                self._lexical_kernel_receipt_sha256
            ),
            streaming_kernel_receipt_sha256=(
                self._streaming_kernel_receipt_sha256
            ),
            factorized_route_kernel_receipt_sha256=(
                self._factorized_route_kernel_receipt_sha256
            ),
            expected_chart_role_count=3,
            expected_batches_per_chart_role=45,
            expected_chart_capture_backbone_forward_count=135,
            expected_chart_capture_terminal_cotangent_VJP_count=135,
            expected_dense_parity_backbone_forward_count=1,
            expected_dense_parity_terminal_cotangent_VJP_count=1,
            expected_total_backbone_forward_count=136,
            expected_total_terminal_cotangent_VJP_count=136,
            expected_gradient_target_count_per_VJP=92,
            observed_role_batch_counts=tuple(
                (role, self._role_batches[role]) for role in _CHART_ROLES
            ),
            observed_chart_capture_backbone_forward_count=(
                self._chart_forwards
            ),
            observed_chart_capture_terminal_cotangent_VJP_count=(
                self._chart_vjps
            ),
            observed_dense_parity_backbone_forward_count=(
                self._parity_forwards
            ),
            observed_dense_parity_terminal_cotangent_VJP_count=(
                self._parity_vjps
            ),
            observed_total_backbone_forward_count=(
                self._chart_forwards + self._parity_forwards
            ),
            observed_total_terminal_cotangent_VJP_count=(
                self._chart_vjps + self._parity_vjps
            ),
            observed_dense_lexical_parity_smoke_count=self._parity_smokes,
            observed_captured_example_count=self._captured_examples,
            observed_supplemental_route_receipt_count=self._route_receipts,
            contains_prompt_text=False,
            contains_token_ids=False,
            contains_raw_activations=False,
            contains_raw_gradients=False,
            contains_logits=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._assert_open_and_unchanged()
        if (
            _adapter_fingerprint(self._adapter, "model_fingerprint")
            != self._source_model_before
        ):
            raise RuntimeError("source model fingerprint changed during runtime")
        self._closed = True


def build_pinned_gemma3_chart_native_capture_runtime(
    preflight: ExecutionPreflightReceipt,
    *,
    cache_dir: Path | str | None = None,
) -> Gemma3ChartNativeCaptureRuntime:
    """Load and compile the setup-pinned Gemma checkpoint on CPU float32."""

    preflight.validate_receipt()
    test_only = preflight.binding.test_only_allow_nonproduction_shape
    if test_only:
        raise ValueError(
            "test-only runtimes must use the explicit injected constructor"
        )
    source = preflight.setup["source_model"]
    model_id = str(source["model_id"])
    revision = str(source["model_revision"])
    tokenizer_revision = str(source.get("tokenizer_revision", ""))
    from .gemma3_chart_native_neighbor_setup import (
        PINNED_MODEL_ID,
        PINNED_MODEL_REVISION,
    )

    if (
        model_id != PINNED_MODEL_ID
        or revision != PINNED_MODEL_REVISION
        or tokenizer_revision != PINNED_MODEL_REVISION
    ):
        raise ValueError("runtime setup model/tokenizer pin drifted")
    from .gemma3_experiment import (
        load_gemma3,
        resolve_gemma3_huggingface_paths,
    )
    from .gemma3_chart_native_neighbor_fresh_panels import (
        authenticate_tokenizer_assets,
    )

    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    snapshot = (
        cache
        / "models--google--gemma-3-270m"
        / "snapshots"
        / revision
    )
    authenticate_tokenizer_assets(snapshot)
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=torch.device("cpu"),
        dtype="float32",
        local_files_only=True,
    )
    if not isinstance(model, nn.Module):
        raise TypeError("pinned Gemma loader returned an invalid model")
    model.to(device="cpu", dtype=torch.float32)
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    return Gemma3ChartNativeCaptureRuntime(
        preflight,
        adapter=adapter,
        tokenizer=tokenizer,
    )


__all__ = [
    "ChartNativeRuntimeCounters",
    "Gemma3ChartNativeCaptureRuntime",
    "RUNTIME_SCHEMA",
    "build_pinned_gemma3_chart_native_capture_runtime",
]
