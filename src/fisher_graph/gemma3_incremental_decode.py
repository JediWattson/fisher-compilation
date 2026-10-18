"""Incremental causal decode for the joint cross-block chart executor.

The audited forward path (``Gemma3JointCrossBlockChartExecutor.execute``) is
teacher-forced: every call replays all 39 events over the full sequence, so
naive autoregressive generation is O(T^2).  Causality makes that redundant —
each mutation is declared ``causal_prefix_mixer`` or ``token_local`` in its
event spec, so once a position's state has passed an event it can never
change again.  This module keeps a persistent decode session that caches
exactly the cross-position state attention needs:

* ``lexical.ingress``/``egress`` and ``terminal_norm`` are token-local.
* ``mlp_mutation`` events are token-local — they run through the unchanged
  ``JointGraphGeneratorMutation.execute_with_incoming`` on a one-token
  sequence context.
* ``attention_mutation`` events mix over the causal prefix.  The cacheable
  quantity is the per-position ``(qk_normed + rotary) key`` and ``value``
  derived inside ``Gemma3AttentionGenerator._execute_coordinates``; both are
  immutable once computed.  Prefill derives them from the transported input
  packet (identical expressions to the audited path), and each decode step
  appends one position.
* Transport edges are ``same_token`` by construction, so emitted generator
  latents never cross positions.  A fresh ``_JointGraphMessageSession`` per
  token reproduces the audited publish/consume semantics with zero
  cross-step state.

The result is math-exact incremental decode: prefill + per-token steps
recompute nothing for positions already processed.  ``JointGraphDecodeSession``
is a runtime mode, not an approximation — parity against teacher-forced
execution is verified by ``tests/test_incremental_decode.py``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .adapters.base import SequenceContext, SequenceInputOrigin
from .gemma3_attention_generator import Gemma3AttentionGenerator
from .gemma3_joint_cross_block_chart_executor import (
    Gemma3JointCrossBlockChartExecutor,
    JointGraphGeneratorMutation,
    _JointGraphMessageSession,
)


class JointGraphDecodeSession:
    """Persistent incremental-decode state for one executor batch.

    Instances are created by :meth:`prefill`, which runs the event schedule
    over the prompt (batched, causal — same asymptotics as the audited
    forward) and fills the per-attention-event key/value caches.  Each
    :meth:`step` then runs the schedule on a single new token: token-local
    mutations run unchanged while attention scores the new query against the
    cached prefix keys and values.
    """

    def __init__(
        self,
        executor: Gemma3JointCrossBlockChartExecutor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(
            executor, Gemma3JointCrossBlockChartExecutor
        ):
            raise TypeError("decode session requires the joint chart executor")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._executor = executor
        self._batch_size = batch_size
        self._device = device
        self._dtype = dtype
        self._length = 0
        self._key_valid: Tensor | None = None
        self._attention_kv: dict[str, dict[str, Tensor]] = {}
        self._attention_mutations = [
            mutation
            for mutation in executor.mutations
            if mutation.event_spec.event_kind == "attention_mutation"
        ]
        self._mutation_by_id = {
            mutation.event_spec.event_id: mutation
            for mutation in executor.mutations
        }
        self._event_indexes = {
            event.event_id: event.event_index
            for event in executor._schedule[1:-2]
        }
        for mutation in self._attention_mutations:
            generator = mutation.compiled_operator
            if not isinstance(generator, Gemma3AttentionGenerator):
                raise ValueError(
                    "incremental decode requires Gemma3AttentionGenerator "
                    "attention endpoints"
                )

    @property
    def length(self) -> int:
        """Number of positions processed so far (prefill + steps)."""

        return self._length

    @classmethod
    def prefill(
        cls,
        executor: Gemma3JointCrossBlockChartExecutor,
        input_ids: Tensor,
        sequence: SequenceContext,
    ) -> tuple["JointGraphDecodeSession", Tensor]:
        """Run the event schedule over a prompt and return (session, logits).

        ``sequence`` must describe the same batch/length as ``input_ids``.
        Rows may be fully valid or left-padded (validity nondecreasing):
        masked prefix positions never publish transport state and are never
        attended, so pad state stays confined to its own slots.
        """

        if (
            not isinstance(input_ids, Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.shape
            != (sequence.batch_size, sequence.query_length)
        ):
            raise ValueError("input_ids must match the sequence context")
        # Left-padded batches are supported: each row's validity must be a
        # nondecreasing mask (pad* then real*), so pad state stays confined
        # to masked prefix slots and never reaches real queries.
        left_pad_ok = bool(
            (
                sequence.query_valid_mask[:, 1:].long()
                >= sequence.query_valid_mask[:, :-1].long()
            ).all()
        )
        if not left_pad_ok:
            raise ValueError(
                "incremental decode requires fully valid or left-padded "
                "prefill rows (attention_mask must be nondecreasing)"
            )
        session = cls(
            executor,
            batch_size=sequence.batch_size,
            device=sequence.device,
            dtype=executor.lexical.exact_weight.dtype
            if executor.lexical.exact_weight is not None
            else executor.lexical.codebook.dtype,
        )
        state = executor.lexical.ingress(input_ids)
        if state.shape != (
            *input_ids.shape,
            executor.hidden_width,
        ):
            raise RuntimeError("lexical ingress returned an invalid carrier")
        state = session._run_events(state, sequence)
        normalized = executor.terminal_norm(state)
        logits = executor.lexical.egress(normalized)
        if executor.final_logit_softcap is not None:
            cap = executor.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        session._length = input_ids.shape[1]
        session._key_valid = sequence.query_valid_mask.clone()
        return session, logits

    def step(self, token_ids: Tensor) -> Tensor:
        """Append one token per batch row and return ``[batch, 1, vocab]``."""

        if self._length == 0:
            raise RuntimeError("decode sessions must prefill before stepping")
        if (
            not isinstance(token_ids, Tensor)
            or token_ids.shape != (self._batch_size, 1)
            or token_ids.dtype not in (torch.int32, torch.int64)
            or token_ids.device != self._device
        ):
            raise ValueError("token_ids must be integer [batch, 1]")
        batch = self._batch_size
        new_key_valid = torch.cat(
            (
                self._key_valid,
                torch.ones(batch, 1, dtype=torch.bool, device=self._device),
            ),
            dim=1,
        )
        # Per-row semantic positions derive from the validity mask: real
        # tokens count 0..len_i-1 while left-pad slots sit at 0 (masked).
        key_logical = (new_key_valid.long().cumsum(1) - 1).clamp(min=0)
        context = SequenceContext(
            query_valid_mask=torch.ones(
                batch, 1, dtype=torch.bool, device=self._device
            ),
            key_valid_mask=new_key_valid,
            logical_positions=key_logical[:, -1:],
            key_logical_positions=key_logical,
            cache_positions=None,
            phase="decode",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=True,
                position_ids_supplied=True,
                cache_positions_supplied=False,
            ),
        )
        state = self._executor.lexical.ingress(token_ids)
        state = self._run_events(state, context)
        normalized = self._executor.terminal_norm(state)
        logits = self._executor.lexical.egress(normalized)
        if self._executor.final_logit_softcap is not None:
            cap = self._executor.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        self._key_valid = new_key_valid
        self._length += 1
        return logits

    def _new_message_session(self, valid_positions: Tensor):
        return _JointGraphMessageSession(
            mutations=self._mutation_by_id,
            edges=tuple(self._executor.transports),
            valid_positions=valid_positions,
            event_indexes=self._event_indexes,
            dtype=self._dtype,
            device=self._device,
        )

    def _run_events(self, state: Tensor, sequence: SequenceContext) -> Tensor:
        """Replay the mutation schedule; attention branches on phase."""

        executor = self._executor
        message_session = self._new_message_session(
            sequence.query_valid_mask
        )
        for mutation in executor.mutations:
            incoming = message_session.incoming_for(mutation)
            if (
                mutation.event_spec.event_kind == "attention_mutation"
                and sequence.phase == "decode"
            ):
                delta, emitted = self._attention_decode(
                    mutation, state, sequence, incoming
                )
            elif (
                mutation.event_spec.event_kind == "attention_mutation"
            ):
                delta, emitted = self._attention_prefill(
                    mutation, state, sequence, incoming
                )
            else:
                execution = mutation.execute_with_incoming(
                    state,
                    sequence,
                    incoming,
                )
                delta = execution.delta
                emitted = execution.emitted_generator_latents
            state = state + delta
            message_session.publish(mutation, emitted)
        return state

    def _attention_prefill(
        self,
        mutation: JointGraphGeneratorMutation,
        state: Tensor,
        sequence: SequenceContext,
        incoming: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Batched causal attention + key/value cache fill.

        Runs the generator's own ``encode_input``/``forward_from_coordinates``
        so the forward stays on the audited code path, then derives the
        immutable rotated keys and values from the same transported packet for
        the decode cache.
        """

        generator = mutation.compiled_operator
        normalized, packet = generator.encode_input(state, sequence)
        combined = packet.values + incoming.squeeze(-2)
        combined = combined.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1), 0
        )
        keys, values = self._derive_attention_kv(
            generator, combined, sequence.key_logical_positions
        )
        self._attention_kv[mutation.event_spec.event_id] = {
            "k": keys,
            "v": values,
        }
        transported = packet.__class__(
            values=combined,
            valid_token_mask=packet.valid_token_mask,
            coordinate_system_sha256=packet.coordinate_system_sha256,
            semantic_role="transported_input",
            source_invocation_id=packet.source_invocation_id,
        )
        execution = generator.forward_from_coordinates(
            transported, sequence
        )
        emitted = execution.emitted_generator_latents
        return execution.delta, emitted

    def _derive_attention_kv(
        self,
        generator: Gemma3AttentionGenerator,
        combined: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Per-position rotated keys and values, exactly as the audited path."""

        attention = generator.config.attention
        batch, length = combined.shape[:2]
        head_dimension = attention.head_dimension
        key = F.linear(
            combined, generator.k_weight, generator.k_bias
        ).view(
            batch, length, attention.key_value_heads, head_dimension
        ).transpose(1, 2)
        value = F.linear(
            combined, generator.v_weight, generator.v_bias
        ).view(
            batch, length, attention.key_value_heads, head_dimension
        ).transpose(1, 2)
        if attention.qk_norm:
            assert generator.k_norm_weight is not None
            assert generator.config.qk_norm is not None
            key = generator._rms_norm(
                key, generator.k_norm_weight, generator.config.qk_norm
            )
        cos, sin = generator._rotary_cos_sin(
            positions, dtype=key.dtype
        )
        key = (
            key * cos.unsqueeze(1)
            + generator._rotate_half(key) * sin.unsqueeze(1)
        )
        return key, value

    def _attention_decode(
        self,
        mutation: JointGraphGeneratorMutation,
        state: Tensor,
        sequence: SequenceContext,
        incoming: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single-query attention against the cached prefix.

        Replicates ``_execute_coordinates`` exactly: same encode, same
        transported packet, same projections — but the score/value mix reads
        the persistent cache instead of recomputing prefix positions.
        """

        generator = mutation.compiled_operator
        attention = generator.config.attention
        batch = sequence.batch_size
        head_dimension = attention.head_dimension

        # encode_input math inlined: its _validate_sequence gate is
        # prefill-only, but the computation itself is token-local.
        normalized = generator._rms_norm(
            state, generator.input_norm_weight, generator.config.input_norm
        )
        coordinates = (
            normalized
            if generator.input_basis is None
            else torch.matmul(normalized, generator.input_basis)
        )
        coordinates = coordinates.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1), 0
        )
        combined = coordinates + incoming.squeeze(-2)
        combined = combined.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1), 0
        )
        # The new token's own key/value use the same derivation as prefill.
        query = F.linear(
            combined, generator.q_weight, generator.q_bias
        ).view(
            batch, 1, attention.query_heads, head_dimension
        ).transpose(1, 2)
        key_new, value_new = self._derive_attention_kv(
            generator, combined, sequence.logical_positions
        )
        if attention.qk_norm:
            assert generator.q_norm_weight is not None
            query = generator._rms_norm(
                query, generator.q_norm_weight, generator.config.qk_norm
            )
        cos, sin = generator._rotary_cos_sin(
            sequence.logical_positions, dtype=query.dtype
        )
        query = (
            query * cos.unsqueeze(1)
            + generator._rotate_half(query) * sin.unsqueeze(1)
        )

        cache = self._attention_kv.get(mutation.event_spec.event_id)
        if cache is None:
            raise RuntimeError("attention cache missing before decode step")
        keys = torch.cat((cache["k"], key_new), dim=2)
        values = torch.cat((cache["v"], value_new), dim=2)
        cache["k"] = keys
        cache["v"] = values

        repeats = attention.query_heads // attention.key_value_heads
        repeated_key = generator._repeat_key_value(keys, repeats)
        repeated_value = generator._repeat_key_value(values, repeats)
        scores = torch.matmul(
            query, repeated_key.transpose(-2, -1)
        ) * float(attention.query_scale)
        if generator.config.logit_softcap is not None:
            cap = float(generator.config.logit_softcap)
            scores = torch.tanh(scores / cap) * cap
        # Decode-step causal rule: every cached position precedes the new
        # token, so causality holds by construction; key_valid_mask drops
        # left-pad slots and a sliding window drops distant positions.
        allowed = sequence.key_valid_mask.unsqueeze(-2)
        if attention.window_size is not None:
            distance = (
                sequence.logical_positions.unsqueeze(-1)
                - sequence.key_logical_positions.unsqueeze(-2)
            )
            allowed = allowed & (distance < attention.window_size)
        scores = scores.masked_fill(
            ~allowed.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        probabilities = F.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(dtype=query.dtype)
        context = torch.matmul(probabilities, repeated_value)
        context = context.transpose(1, 2).contiguous().view(
            batch, 1, generator.config.query_width
        )
        if generator.output_encoder_weight is None:
            output_coordinates = context
        else:
            output_coordinates = F.linear(
                context, generator.output_encoder_weight
            )
        operator_output = F.linear(
            output_coordinates,
            generator.output_decoder_weight,
            generator.o_bias,
        )
        delta = generator._rms_norm(
            operator_output,
            generator.output_norm_weight,
            generator.config.output_norm,
        )
        delta = delta.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1), 0
        )
        output_coordinates = output_coordinates.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1), 0
        )
        emitted = output_coordinates.unsqueeze(-2)
        return delta, emitted


def batched_greedy_generate(
    executor: Gemma3JointCrossBlockChartExecutor,
    adapter: object,
    tokenizer: object,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
) -> tuple[list[list[int]], JointGraphDecodeSession]:
    """Greedy-decode a ragged batch of prompts in one decode session.

    Prompts are left-padded to a common length (pad state stays confined to
    masked slots, which real queries never attend), then all rows advance in
    lockstep — each ``step`` call produces one token per row for the same
    dispatch cost as a single-row step.  Rows that emit ``eos_id`` keep
    stepping internally but stop recording.

    Returns ``(per_prompt_new_tokens, session)``.
    """

    if not prompts:
        raise ValueError("prompts must be nonempty")
    encoded = [
        tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        for prompt in prompts
    ]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    width = max(int(ids.shape[0]) for ids in encoded)
    input_ids = torch.stack(
        [
            torch.cat(
                (
                    torch.full(
                        (width - int(ids.shape[0]),),
                        int(pad_id),
                        dtype=torch.long,
                    ),
                    ids.long(),
                )
            )
            for ids in encoded
        ]
    )
    attention_mask = torch.stack(
        [
            torch.cat(
                (
                    torch.zeros(
                        width - int(ids.shape[0]), dtype=torch.bool
                    ),
                    torch.ones(int(ids.shape[0]), dtype=torch.bool),
                )
            )
            for ids in encoded
        ]
    )
    position_ids = (attention_mask.long().cumsum(1) - 1).clamp(min=0)
    sequence = adapter.prepare_sequence(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask.long(),
            "position_ids": position_ids,
        }
    )
    session, logits = JointGraphDecodeSession.prefill(
        executor, input_ids, sequence
    )
    new_tokens: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    current = logits
    for _ in range(max_new_tokens):
        next_ids = current[:, -1].argmax(dim=-1)
        for row, token in enumerate(next_ids.tolist()):
            if finished[row]:
                continue
            new_tokens[row].append(int(token))
            if eos_id is not None and int(token) == eos_id:
                finished[row] = True
        if eos_id is not None and all(finished):
            break
        current = session.step(next_ids.view(-1, 1))
    return new_tokens, session
