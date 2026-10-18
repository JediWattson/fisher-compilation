"""Parity tests for incremental decode vs teacher-forced execution.

The decode session reorders the same computation (per-position work once,
cached across steps) rather than approximating it, so incremental logits
must match the audited forward path to within float32 matmul-association
tolerance — and argmax must match exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fisher_graph.gemma3_experiment import (
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.gemma3_incremental_decode import JointGraphDecodeSession

CACHE_DIR = Path(".local-runs/google--gemma-3-270m")
EXECUTOR_PATH = (
    CACHE_DIR
    / "ewc-interactive-executor-39-event-fused640-terminalrefit-v2.pt"
)
FALLBACK_EXECUTOR_PATH = (
    CACHE_DIR / "ewc-interactive-executor-39-event-fused640-v1.pt"
)
REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PROMPT = "The capital of France is"
DECODE_STEPS = 6


@pytest.fixture(scope="module")
def executor_and_sequence():
    path = EXECUTOR_PATH if EXECUTOR_PATH.exists() else FALLBACK_EXECUTOR_PATH
    if not path.exists():
        pytest.skip("compiled executor artifact not present")
    device = resolve_torch_device("cpu")
    cache = resolve_gemma3_huggingface_paths(None)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id="google/gemma-3-270m",
        revision=REVISION,
        cache_dir=cache,
        device=device,
        dtype="float32",
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    executor = torch.load(path, map_location="cpu", weights_only=False)
    executor.eval().requires_grad_(False)
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    sequence = adapter.prepare_sequence(batch)
    return executor, adapter, input_ids, sequence


def test_prefill_logits_match_teacher_forced(executor_and_sequence):
    executor, adapter, input_ids, sequence = executor_and_sequence
    with torch.inference_mode():
        _, session_logits = JointGraphDecodeSession.prefill(
            executor, input_ids, sequence
        )
        reference = executor.execute(input_ids, sequence)
    if hasattr(reference, "logits"):
        reference = reference.logits
    torch.testing.assert_close(
        session_logits, reference, rtol=1e-4, atol=1e-5
    )


def test_decode_steps_match_teacher_forced(executor_and_sequence):
    executor, adapter, input_ids, sequence = executor_and_sequence
    generated = input_ids.clone()
    step_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        session, prefill_logits = JointGraphDecodeSession.prefill(
            executor, input_ids, sequence
        )
        step_logits.append(prefill_logits[:, -1:])
        for _ in range(DECODE_STEPS):
            next_id = int(step_logits[-1][0, -1].argmax().item())
            token = torch.tensor(
                [[next_id]], dtype=input_ids.dtype, device=input_ids.device
            )
            step_logits.append(session.step(token))
            generated = torch.cat((generated, token), dim=1)

        # Teacher-forced reference over the same tokens.
        ref_batch = {
            "input_ids": generated,
            "attention_mask": torch.ones_like(generated),
        }
        ref_sequence = adapter.prepare_sequence(ref_batch)
        reference = executor.execute(generated, ref_sequence)
        if hasattr(reference, "logits"):
            reference = reference.logits

    prompt_len = input_ids.shape[1]
    for step_index, step in enumerate(step_logits):
        position = prompt_len - 1 + step_index
        torch.testing.assert_close(
            step[:, -1],
            reference[:, position],
            rtol=1e-3,
            atol=5e-3,
            msg=f"decode step {step_index} drifted from teacher-forced",
        )
        assert int(step[0, -1].argmax()) == int(
            reference[0, position].argmax()
        ), f"decode step {step_index} changed argmax"
