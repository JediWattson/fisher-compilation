from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from fisher_graph import gemma3_layer_multimode_runtime as runtime


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _qualification() -> runtime.LayerMultimodeRuntimeQualificationReceipt:
    return runtime.LayerMultimodeRuntimeQualificationReceipt(
        execution_preflight_receipt_sha256=_sha("preflight"),
        implementation_qualification_sha256=_sha("implementation"),
        implementation_qualification_file_sha256=_sha("implementation-file"),
        runtime_implementation_sha256=_sha("runtime"),
        qualified_runtime_source_sha256=_sha("runtime"),
        predecessor_preflight_receipt_sha256=_sha("predecessor-preflight"),
        predecessor_runtime_qualification_receipt_sha256=(
            _sha("predecessor-runtime-qualification")
        ),
        source_model_fingerprint_before_sha256=_sha("model"),
        source_model_fingerprint_after_sha256=_sha("model"),
        source_execution_fingerprint_before_sha256=_sha("execution"),
        source_execution_fingerprint_after_sha256=_sha("execution"),
        tokenizer_asset_catalog_sha256=_sha("tokenizer-assets"),
        evaluation_protocol_sha256=_sha("evaluation"),
        source_model_mutated=False,
    )


def test_import_is_model_and_tensor_runtime_inert(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import json,sys; "
        "import fisher_graph.gemma3_layer_multimode_runtime; "
        "print(json.dumps({"
        "'transformers': 'transformers' in sys.modules, "
        "'model_loader': 'fisher_graph.gemma3_experiment' in sys.modules, "
        "'predecessor_runtime': "
        "'fisher_graph.gemma3_chart_native_neighbor_runtime' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env={"PYTHONPATH": str(source_root)},
        check=False,
        capture_output=True,
        text=True,
    )
    # -I ignores PYTHONPATH; use a source insertion while keeping site isolation.
    if completed.returncode != 0:
        script = (
            f"import sys;sys.path.insert(0,{str(source_root)!r});" + script
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
    assert json.loads(completed.stdout) == {
        "transformers": False,
        "model_loader": False,
        "predecessor_runtime": False,
    }
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "torch" not in imported_roots


def test_runtime_qualification_is_self_hashed_and_immutable() -> None:
    receipt = _qualification()
    assert receipt.state_dict()["receipt_sha256"] == receipt.receipt_sha256
    with pytest.raises(ValueError, match="immutable source"):
        runtime.LayerMultimodeRuntimeQualificationReceipt(
            **{
                **receipt._payload(),
                "source_model_mutated": True,
            }
        )


@dataclass(frozen=True)
class _Recaptured:
    receipt_sha256: str
    payload: dict[str, object]

    def payload_without_sha256(self) -> dict[str, object]:
        return dict(self.payload)


def _capture_state() -> dict[str, object]:
    mode_ids = (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )
    return {
        "role": "A_chart_transport_fit",
        "runtime_session_binding_sha256": _sha("old-session"),
        "role_panel_binding_sha256": _sha("panel"),
        "ordered_objective_row_catalog_sha256": _sha("rows"),
        "score_objective_receipt_sha256": _sha("objective"),
        "batch_axis_catalog_sha256": _sha("batch-axis"),
        "mode_raw_artifacts": [
            [
                mode,
                _sha(f"artifact:{index}"),
                _sha(f"rows:{index}"),
                _sha(f"tensors:{index}"),
            ]
            for index, mode in enumerate(mode_ids)
        ],
        "lexical_nll_receipt_sha256s": [_sha("nll")],
        "execution_receipt_sha256s": [_sha("execution")],
        "example_capture_receipt_sha256s": {"example": _sha("old-example")},
        "supplemental_route_evidence": sorted(
            [
                [
                    f"mode.mutation.{1 + 2 * layer}",
                    f"example.{example:02d}",
                    _sha(f"route-catalog:{layer}:{example}"),
                    _sha(f"route-receipt:{layer}:{example}"),
                ]
                for layer in range(18)
                for example in range(48)
            ]
        ),
        "dense_lexical_parity_receipt_sha256": _sha("parity"),
        "batch_count": 45,
        "row_count": 24_732,
        "receipt_sha256": _sha("old-capture"),
    }


def test_A_comparison_allows_new_session_but_requires_comparable_lineage() -> None:
    frozen = _capture_state()
    current = dict(frozen)
    current.pop("receipt_sha256")
    current["runtime_session_binding_sha256"] = _sha("new-session")
    current["execution_receipt_sha256s"] = [_sha("new-execution")]
    current["batch_axis_catalog_sha256"] = _sha("new-batch-axis")
    current["supplemental_route_evidence"] = [
        [mode, example, catalog, _sha(f"new-route-receipt:{index}")]
        for index, (mode, example, catalog, _receipt) in enumerate(
            current["supplemental_route_evidence"]
        )
    ]
    current["example_capture_receipt_sha256s"] = {
        "example": _sha("new-example")
    }
    recaptured = _Recaptured(_sha("new-capture"), current)
    receipt = runtime._compare_predecessor_A(
        frozen=frozen,
        recaptured=recaptured,
        predecessor_runtime_execution_receipt_sha256=_sha("predecessor-runtime"),
    )
    assert receipt.all_comparable_fields_matched
    assert receipt.predecessor_capture_receipt_sha256 != (
        receipt.recaptured_capture_receipt_sha256
    )

    drifted = dict(current)
    drifted["row_count"] = 24_731
    with pytest.raises(RuntimeError, match="row_count"):
        runtime._compare_predecessor_A(
            frozen=frozen,
            recaptured=_Recaptured(_sha("drifted"), drifted),
            predecessor_runtime_execution_receipt_sha256=(
                _sha("predecessor-runtime")
            ),
        )


def test_fresh_kernel_surface_requires_typed_candidate_bound_capture() -> None:
    expected = {
        "evaluate_b3_teacher_forced": {
            "panel",
            "candidate",
            "fresh_role_capture_receipt",
        },
        "evaluate_c3_full_stack": {
            "panel",
            "candidate",
            "b3_evaluation",
            "fresh_role_capture_receipt",
        },
        "evaluate_d3_final": {
            "panel",
            "candidate",
            "c3_evaluation",
            "fresh_role_capture_receipt",
        },
    }
    for name, parameters in expected.items():
        method = getattr(runtime.Gemma3LayerMultimodeRuntime, name)
        assert set(inspect.signature(method).parameters) == {
            "self",
            *parameters,
        }
        assert "not yet frozen" not in tuple(
            value for value in method.__code__.co_consts if isinstance(value, str)
        )


@pytest.mark.parametrize("failed_name", ("native00.pt", "compile0.pt"))
def test_D3_cleanup_attempts_every_file_after_persistent_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_name: str,
) -> None:
    directory = tmp_path / "resource"
    directory.mkdir(mode=0o700)
    workspace = runtime._open_D3_resource_directory(directory)
    first = directory / "native00.pt"
    second = directory / "compile0.pt"
    for name, payload in ((first.name, b"native"), (second.name, b"compiled")):
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=workspace.directory_descriptor,
        )
        try:
            os.write(descriptor, payload)
            observed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        workspace.artifacts.append(
            runtime._D3TemporaryArtifact(
                name=name,
                device=observed.st_dev,
                inode=observed.st_ino,
            )
        )
    original_unlink = os.unlink
    attempts: list[str] = []

    def fail_one(path, *args, **kwargs) -> None:
        name = os.fsdecode(path)
        logical_name = next(
            value
            for value in ("native00.pt", "compile0.pt")
            if name.startswith(f".{value}.")
        )
        attempts.append(logical_name)
        if logical_name == failed_name:
            raise OSError(f"persistent {failed_name} unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_one)
    errors = runtime._cleanup_D3_resource_directory(workspace)
    assert attempts == ["native00.pt", "compile0.pt"]
    assert not first.exists()
    assert not second.exists()
    preserved = tuple(directory.iterdir())
    assert len(preserved) == 1
    assert preserved[0].name.startswith(f".{failed_name}.")
    assert len(errors) == 2
    assert str(errors[0]) == f"persistent {failed_name} unlink failure"
    assert str(errors[1]) == (
        "D3 resource workspace retained after cleanup failure"
    )
    assert directory.exists()


def test_D3_cleanup_preserves_replaced_artifact_and_attempts_other_targets(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "resource"
    directory.mkdir(mode=0o700)
    workspace = runtime._open_D3_resource_directory(directory)
    for name in ("native00.pt", "compile0.pt"):
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=workspace.directory_descriptor,
        )
        try:
            os.write(descriptor, name.encode("ascii"))
            observed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        workspace.artifacts.append(
            runtime._D3TemporaryArtifact(
                name=name,
                device=observed.st_dev,
                inode=observed.st_ino,
            )
        )
    os.unlink("native00.pt", dir_fd=workspace.directory_descriptor)
    os.symlink(
        "replacement-target",
        "native00.pt",
        dir_fd=workspace.directory_descriptor,
    )

    errors = runtime._cleanup_D3_resource_directory(workspace)

    assert len(errors) == 2
    assert "preserved replaced artifact" in str(errors[0])
    assert (directory / "native00.pt").is_symlink()
    assert not (directory / "compile0.pt").exists()
    (directory / "native00.pt").unlink()
    directory.rmdir()


def test_D3_cleanup_preserves_swap_between_identity_check_and_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "resource"
    directory.mkdir(mode=0o700)
    workspace = runtime._open_D3_resource_directory(directory)
    for name in ("native00.pt", "compile0.pt"):
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=workspace.directory_descriptor,
        )
        try:
            os.write(descriptor, name.encode("ascii"))
            observed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        workspace.artifacts.append(
            runtime._D3TemporaryArtifact(
                name=name,
                device=observed.st_dev,
                inode=observed.st_ino,
            )
        )
    original_rename = os.rename
    injected = False

    def swap_then_rename(source, destination, *args, **kwargs):
        nonlocal injected
        if source == "native00.pt" and not injected:
            injected = True
            source_directory = kwargs["src_dir_fd"]
            os.unlink(source, dir_fd=source_directory)
            os.symlink(
                "replacement-target",
                source,
                dir_fd=source_directory,
            )
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "rename", swap_then_rename)
    errors = runtime._cleanup_D3_resource_directory(workspace)

    assert injected
    assert len(errors) == 2
    assert "quarantined a replaced artifact" in str(errors[0])
    preserved = tuple(directory.iterdir())
    assert len(preserved) == 1
    assert preserved[0].is_symlink()
    assert os.readlink(preserved[0]) == "replacement-target"


def test_D3_cleanup_preserves_workspace_swap_before_rmdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "resource"
    directory.mkdir(mode=0o700)
    workspace = runtime._open_D3_resource_directory(directory)
    original_rename = os.rename
    injected = False

    def swap_then_rename(source, destination, *args, **kwargs):
        nonlocal injected
        if source == directory.name and not injected:
            injected = True
            parent_descriptor = kwargs["src_dir_fd"]
            os.rmdir(source, dir_fd=parent_descriptor)
            os.mkdir(source, mode=0o700, dir_fd=parent_descriptor)
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "rename", swap_then_rename)
    errors = runtime._cleanup_D3_resource_directory(workspace)

    assert injected
    assert len(errors) == 1
    assert "quarantined a replaced workspace" in str(errors[0])
    preserved = tuple(tmp_path.iterdir())
    assert len(preserved) == 1
    assert preserved[0].is_dir()
    preserved[0].rmdir()


def test_D3_serialization_uses_exclusive_anchored_regular_file(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "resource"
    directory.mkdir(mode=0o700)
    workspace = runtime._open_D3_resource_directory(directory)
    model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.arange(4, dtype=torch.float32)}
    )

    digest, byte_count = runtime._serialize_model_state_same_protocol(
        model,
        workspace,
        "native00.pt",
    )

    raw = (directory / "native00.pt").read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert byte_count == len(raw) > 0
    assert len(workspace.artifacts) == 1
    assert runtime._cleanup_D3_resource_directory(workspace) == ()
    assert not directory.exists()


class _WeakObject:
    pass


class _ClosingOwner:
    def __init__(self, *values: object) -> None:
        self.values = list(values)

    def close(self) -> None:
        self.values.clear()


def _terminal_close_runtime(
    tmp_path: Path,
    *,
    retain_compiled_alias: bool,
) -> tuple[runtime.Gemma3LayerMultimodeRuntime, object | None]:
    value = object.__new__(runtime.Gemma3LayerMultimodeRuntime)
    native_model = _WeakObject()
    tokenizer = _WeakObject()
    compiled_model = _WeakObject()
    bundle = tmp_path / "candidate.pt"
    bundle.write_bytes(b"candidate bundle")
    frozen = SimpleNamespace(
        compiled_model=compiled_model,
        bundle_path=str(bundle),
        bundle_file_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
    )
    value._closed = False
    value._development_teardown_confirmed = True
    value._issued_runtime_execution_receipt = SimpleNamespace(
        terminal_status=runtime.experiment.TERMINAL_B3_FAILED,
        receipt_sha256=_sha("runtime-execution"),
        runtime_session_binding_sha256=_sha("runtime-session"),
    )
    value._capture_runtime = None
    value._A_raw_directory = None
    value._frozen_candidate = frozen
    value._native_baseline = _ClosingOwner(native_model, tokenizer)
    value._compiled_evaluator = _ClosingOwner(compiled_model)
    value._native_model_weakref = weakref.ref(native_model)
    value._native_tokenizer_weakref = weakref.ref(tokenizer)
    value._compiled_model_weakref = weakref.ref(compiled_model)
    value._captured_A = None
    value._bridge = None
    value._partition = None
    value._predecessor_qualification = None
    value._runtime_qualification = None
    value._A_runtime_receipt = None
    value._A_comparison_receipt = None
    value._pending_development_release = None
    value._tokenization_receipts = {}
    value._authenticated_panel_ids = {}
    value._fresh_manifests = {}
    value._fresh_role_capture_receipts = {}
    value._selected_candidate_sha256s = None
    value._full_stack_example_aggregates = {}
    value._D3_trace_events = []
    value._execution_environment_sha256 = _sha("environment")
    value._runtime_session_binding_sha256 = _sha("runtime-session")
    value._compiled_model_fingerprint_before_sha256 = _sha("compiled-model")
    value._compiled_execution_fingerprint_before_sha256 = _sha(
        "compiled-execution"
    )
    value.predecessor_preflight = None
    value.predecessor_report = {}
    value.implementation_qualification = {}
    leaked = compiled_model if retain_compiled_alias else None
    del native_model, tokenizer, compiled_model, frozen
    return value, leaked


def test_terminal_close_proves_all_model_and_tokenizer_weakrefs_are_dead(
    tmp_path: Path,
) -> None:
    value, leaked = _terminal_close_runtime(
        tmp_path,
        retain_compiled_alias=False,
    )
    assert leaked is None
    receipt = value.close(
        terminal_status=runtime.experiment.TERMINAL_B3_FAILED,
        runtime_execution_receipt_sha256=_sha("runtime-execution"),
        runtime_session_binding_sha256=_sha("runtime-session"),
        fresh_panel_execution_receipt_sha256=_sha("fresh-execution"),
    )
    assert receipt.model_reference_count_after_close == 0
    assert receipt.tokenizer_reference_count_after_close == 0


def test_terminal_close_refuses_an_external_compiled_model_alias(
    tmp_path: Path,
) -> None:
    value, leaked = _terminal_close_runtime(
        tmp_path,
        retain_compiled_alias=True,
    )
    assert leaked is not None
    with pytest.raises(RuntimeError, match="external model/tokenizer alias"):
        value.close(
            terminal_status=runtime.experiment.TERMINAL_B3_FAILED,
            runtime_execution_receipt_sha256=_sha("runtime-execution"),
            runtime_session_binding_sha256=_sha("runtime-session"),
            fresh_panel_execution_receipt_sha256=_sha("fresh-execution"),
        )


def test_chunked_full_vocab_metrics_match_dense_and_are_example_additive() -> None:
    generator = torch.Generator().manual_seed(20260810)
    native_hidden = torch.randn(7, 5, generator=generator, dtype=torch.float32)
    candidate_hidden = torch.randn(7, 5, generator=generator, dtype=torch.float32)
    native_weight = torch.randn(13, 5, generator=generator, dtype=torch.float32)
    candidate_weight = torch.randn(13, 5, generator=generator, dtype=torch.float32)
    targets = torch.tensor([0, 12, 4, 5, 3, 9, 1], dtype=torch.long)

    observed = runtime._chunked_full_vocab_pair_metric_sums(
        native_hidden=native_hidden,
        candidate_hidden=candidate_hidden,
        native_weight=native_weight,
        candidate_weight=candidate_weight,
        target_token_ids=targets,
        example_row_counts=(2, 3, 2),
        native_softcap=4.0,
        candidate_softcap=4.0,
        vocabulary_chunk_size=3,
    )
    native_logits = torch.tanh(
        (native_hidden.double() @ native_weight.double().T) / 4.0
    ) * 4.0
    candidate_logits = torch.tanh(
        (candidate_hidden.double() @ candidate_weight.double().T) / 4.0
    ) * 4.0
    native_logp = torch.log_softmax(native_logits, dim=-1)
    candidate_logp = torch.log_softmax(candidate_logits, dim=-1)
    rows = torch.arange(7)
    dense_delta = -candidate_logp[rows, targets] + native_logp[rows, targets]
    dense_kl = (
        native_logp.exp() * (native_logp - candidate_logp)
    ).sum(dim=-1)
    dense_agreement = native_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
    expected = []
    offset = 0
    for count in (2, 3, 2):
        end = offset + count
        expected.append(
            (
                float(dense_delta[offset:end].sum()),
                float(dense_kl[offset:end].sum()),
                int(dense_agreement[offset:end].sum()),
            )
        )
        offset = end
    for actual, wanted in zip(observed, expected, strict=True):
        assert actual[0] == pytest.approx(wanted[0], abs=1e-12)
        assert actual[1] == pytest.approx(wanted[1], abs=1e-12)
        assert actual[2] == wanted[2]
    pooled = tuple(sum(row[index] for row in observed) for index in range(3))
    assert pooled[0] == pytest.approx(float(dense_delta.sum()), abs=1e-12)
    assert pooled[1] == pytest.approx(float(dense_kl.sum()), abs=1e-12)
    assert pooled[2] == int(dense_agreement.sum())


def test_chunked_top1_ties_use_lowest_vocabulary_index() -> None:
    hidden = torch.ones(2, 1, dtype=torch.float32)
    weight = torch.zeros(5, 1, dtype=torch.float32)
    observed = runtime._chunked_full_vocab_pair_metric_sums(
        native_hidden=hidden,
        candidate_hidden=hidden,
        native_weight=weight,
        candidate_weight=weight,
        target_token_ids=torch.tensor([1, 4]),
        example_row_counts=(1, 1),
        native_softcap=None,
        candidate_softcap=None,
        vocabulary_chunk_size=2,
    )
    assert observed == ((0.0, 0.0, 1), (0.0, 0.0, 1))
