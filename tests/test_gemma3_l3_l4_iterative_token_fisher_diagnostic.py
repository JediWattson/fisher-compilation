from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_iterative_token_fisher_diagnostic as diagnostic,
)
from fisher_graph.gemma3_l3_l4_iterative_residual_diagnostic import (
    _GemmaDevelopmentCollectionRecipe,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Example:
    def __init__(self, index: int) -> None:
        self.example_id = f"example-{index:02d}"
        self.family_id = f"family-{index % 8}"
        self.model_inputs_sha256 = _hash(f"inputs-{index}")
        input_ids = torch.tensor([[1, 2]], dtype=torch.int64)
        targets = torch.tensor([[1, 2]], dtype=torch.int64)
        self.batch = SimpleNamespace(
            model_inputs={"input_ids": input_ids},
            targets=targets,
        )

    def validate_integrity(self) -> None:
        return None


class _TokenVJP:
    def __init__(
        self,
        *,
        example: _Example,
        execution: object,
        token_losses: torch.Tensor,
    ) -> None:
        self.execution = execution
        self.supervised_indices = torch.tensor(
            [[0, 0], [0, 1]],
            dtype=torch.int64,
        )
        self.token_losses = token_losses
        self.h4_gradients = torch.ones(
            (2, 1, 2, 3),
            dtype=torch.float64,
        )
        self.artifact_sha256 = _hash(f"token-vjp-{example.example_id}")
        self.backward_call_count = 1

    def validate_integrity(self) -> None:
        return None


class _Bridge:
    bridge_binding_sha256 = _hash("bridge")

    def __init__(self, *, x4_head: object, h4_head: object) -> None:
        self.x4_head = x4_head
        self.h4_head = h4_head
        self.parent_forward_count = 0
        self.calls: list[dict[str, object]] = []

    def execute_h4_token_nll_vjps(
        self,
        _adapter: object,
        model_inputs: object,
        *,
        targets: torch.Tensor,
        vjp_chunk_size: int,
        x4_head: object,
        h4_head: object,
    ) -> _TokenVJP:
        assert x4_head is self.x4_head
        assert h4_head is self.h4_head
        assert vjp_chunk_size == diagnostic.TOKEN_FISHER_VJP_CHUNK_SIZE
        assert isinstance(model_inputs, dict)
        assert targets.shape == (1, 2)
        self.parent_forward_count += 1
        example = self._example_by_inputs[id(model_inputs)]
        parent_logits = torch.tensor(
            [[[0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]]],
            dtype=torch.float64,
        )
        token_losses = torch.nn.functional.cross_entropy(
            parent_logits[0],
            targets[0],
            reduction="none",
        )
        execution = SimpleNamespace(logits=parent_logits)
        self.calls.append(
            {
                "example": example,
                "model_inputs": model_inputs,
                "targets": targets,
                "x4_head": x4_head,
                "h4_head": h4_head,
                "execution": execution,
            }
        )
        return _TokenVJP(
            example=example,
            execution=execution,
            token_losses=token_losses,
        )

    def bind_examples(self, examples: tuple[_Example, ...]) -> None:
        self._example_by_inputs = {
            id(example.batch.model_inputs): example for example in examples
        }


def test_fake_collector_binds_source_and_parent_with_exact_32_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = tuple(_Example(index) for index in range(16))
    panel = SimpleNamespace(examples=examples)
    manifest = {
        example.example_id: example.family_id for example in examples
    }
    x4_head = SimpleNamespace(artifact_sha256=_hash("x4"))
    h4_head = SimpleNamespace(artifact_sha256=_hash("h4"))
    parent = object()
    adapter = object()
    bridge = _Bridge(x4_head=x4_head, h4_head=h4_head)
    bridge.bind_examples(examples)
    lineage = {
        "parent_artifact_sha256": _hash("parent"),
        "parent_h4_head_sha256": h4_head.artifact_sha256,
        "accepted_x4_head_sha256": x4_head.artifact_sha256,
        "bridge_binding_sha256": bridge.bridge_binding_sha256,
    }
    observed: dict[str, Any] = {
        "source_forward_count": 0,
        "execution_bindings": [],
        "tangent_bindings": [],
    }

    monkeypatch.setattr(
        diagnostic,
        "_panel_manifest",
        lambda value: manifest if value is panel else None,
    )
    monkeypatch.setattr(
        diagnostic,
        "_validate_parent",
        lambda **kwargs: (
            x4_head,
            h4_head,
        )
        if kwargs
        == {
            "panel": panel,
            "adapter": adapter,
            "bridge": bridge,
            "parent": parent,
        }
        else (_ for _ in ()).throw(AssertionError("parent binding differs")),
    )

    def source_authority(*, adapter: object, example: _Example):
        assert adapter is globals_adapter
        observed["source_forward_count"] += 1
        source_logits = torch.tensor(
            [[0.0, 1.0, -1.0], [-1.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        return (
            SimpleNamespace(source=True),
            source_logits,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([[10, 11]], dtype=torch.int64),
        )

    globals_adapter = adapter
    monkeypatch.setattr(diagnostic, "_source_authority", source_authority)

    def validate_execution(
        execution: object,
        *,
        example_model_inputs_sha256: str,
        bridge_binding_sha256: str,
        x4_head: object,
        h4_head: object,
        label: str,
    ) -> None:
        assert bridge_binding_sha256 == bridge.bridge_binding_sha256
        assert x4_head is globals_x4
        assert h4_head is globals_h4
        assert label == "exact token-loss parent VJP"
        observed["execution_bindings"].append(
            (execution, example_model_inputs_sha256)
        )

    globals_x4 = x4_head
    globals_h4 = h4_head
    monkeypatch.setattr(
        diagnostic,
        "_validate_execution",
        validate_execution,
    )
    monkeypatch.setattr(
        diagnostic,
        "_observation",
        lambda **kwargs: SimpleNamespace(
            observation_sha256=_hash(
                f"observation-{len(observed['tangent_bindings'])}"
            )
        ),
    )

    coordinate_count = len(
        diagnostic.TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER
    )

    def build_tangent(
        *,
        example: _Example,
        parent_execution: object,
        token_loss_gradients: torch.Tensor,
        supervised_token_logical_positions: torch.Tensor,
        parent_h4: object,
        parent_observation: object,
    ) -> object:
        assert parent_execution is bridge.calls[-1]["execution"]
        assert token_loss_gradients.shape == (2, 1, 2, 3)
        assert torch.equal(
            supervised_token_logical_positions,
            torch.tensor([10, 11], dtype=torch.int64),
        )
        assert parent_h4 is h4_head
        assert parent_observation.observation_sha256
        observed["tangent_bindings"].append(example.example_id)
        rows = tuple(
            SimpleNamespace(
                tangent_by_combined_occupancy_coordinate=(
                    tuple(float(row + 1) for _ in range(coordinate_count))
                )
            )
            for row in range(2)
        )
        return SimpleNamespace(
            example_id=example.example_id,
            family_id=example.family_id,
            supervised_token_count=2,
            rows=rows,
        )

    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_token_occupancy_tangent_record",
        build_tangent,
    )

    def build_prompt(
        *,
        example_id: str,
        family_id: str,
        coordinate_names: object,
        token_scores: torch.Tensor,
        compensation_target: torch.Tensor,
    ) -> object:
        assert tuple(coordinate_names) == (
            diagnostic.TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER
        )
        assert token_scores.shape == (2, coordinate_count)
        assert compensation_target.shape == (2,)
        return SimpleNamespace(
            example_id=example_id,
            family_id=family_id,
            supervised_tokens=2,
        )

    monkeypatch.setattr(
        diagnostic,
        "build_token_loss_fisher_prompt_record",
        build_prompt,
    )

    def build_report(
        *,
        token_tangent_records: object,
        prompt_records: object,
        lineage: object,
        token_vjp_artifact_sha256_by_example: object,
        total_backward_call_count: int,
        vjp_chunk_size: int,
    ) -> dict[str, object]:
        assert len(token_tangent_records) == 16
        assert len(prompt_records) == 16
        assert lineage == globals_lineage
        assert len(token_vjp_artifact_sha256_by_example) == 16
        assert total_backward_call_count == 16
        assert vjp_chunk_size == diagnostic.TOKEN_FISHER_VJP_CHUNK_SIZE
        return {
            "schema": "test.token-fisher-development",
            "resources": {
                "source_forward_count": 16,
                "parent_token_vjp_forward_count": 16,
                "total_model_forward_count": 32,
                "candidate_forward_count": 0,
                "fresh_forward_count": 0,
            },
            "audit": {
                "selection_panel_referenced": False,
                "selection_panel_opened": False,
                "selection_claim_created": False,
            },
        }

    globals_lineage = lineage
    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_token_fisher_development_report",
        build_report,
    )

    report = diagnostic._collect_token_fisher(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        parent_artifact=parent,
        parent_h4=h4_head,
        x4_head=x4_head,
        lineage=lineage,
    )

    assert observed["source_forward_count"] == 16
    assert bridge.parent_forward_count == 16
    assert len(observed["execution_bindings"]) == 16
    assert {
        binding for _execution, binding in observed["execution_bindings"]
    } == {example.model_inputs_sha256 for example in examples}
    assert observed["tangent_bindings"] == [
        example.example_id for example in examples
    ]
    assert report["resources"] == {
        "source_forward_count": 16,
        "parent_token_vjp_forward_count": 16,
        "total_model_forward_count": 32,
        "candidate_forward_count": 0,
        "fresh_forward_count": 0,
    }
    assert report["audit"] == {
        "selection_panel_referenced": False,
        "selection_panel_opened": False,
        "selection_claim_created": False,
    }


def test_public_diagnostic_has_no_fresh_inputs_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "token-fisher.report.json"
    report = {
        "schema": "test.token-fisher-development",
        "resources": {"total_model_forward_count": 32},
        "audit": {
            "selection_panel_referenced": False,
            "selection_panel_opened": False,
            "selection_claim_created": False,
        },
    }
    observed: dict[str, Any] = {
        "validation_count": 0,
        "publication_count": 0,
    }

    def validate(value: object) -> None:
        assert value == report
        observed["validation_count"] += 1

    def publish(path: Path, value: object) -> None:
        assert path == destination
        assert value == report
        assert not path.exists()
        observed["publication_count"] += 1
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_token_fisher_development_report",
        validate,
    )
    monkeypatch.setattr(
        diagnostic,
        "publish_gemma_iterative_token_fisher_development_report",
        publish,
    )

    def base_driver(**kwargs: object) -> dict[str, object]:
        protected = (
            "selection",
            "private",
            "fresh",
            "claim",
            "guard",
            "assessment",
            "calibration_b",
        )
        assert not any(
            marker in name for marker in protected for name in kwargs
        )
        recipe = kwargs["_diagnostic_recipe"]
        assert isinstance(recipe, _GemmaDevelopmentCollectionRecipe)
        assert recipe.collect is diagnostic._collect_token_fisher
        assert not any(
            marker in name
            for marker in protected
            for name in recipe.source_code_files
        )
        assert kwargs["output"] == destination
        recipe.validate_report(report)
        recipe.publish_report(destination, report)
        return report

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_residual_diagnostic",
        base_driver,
    )

    result = (
        diagnostic
        .run_gemma_iterative_token_fisher_development_diagnostic(
            expected_materialization_report_sha256="1" * 64,
            expected_materialization_report_file_sha256="2" * 64,
            expected_factorial_report_sha256="3" * 64,
            expected_factorial_report_file_sha256="4" * 64,
            output=destination,
        )
    )

    assert result == report
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert observed == {
        "validation_count": 3,
        "publication_count": 1,
    }
    signature = inspect.signature(
        diagnostic.run_gemma_iterative_token_fisher_development_diagnostic
    )
    protected = (
        "selection",
        "private",
        "fresh",
        "claim",
        "guard",
        "assessment",
        "calibration_b",
    )
    assert not any(
        marker in name
        for marker in protected
        for name in signature.parameters
    )
    parser_destinations = {
        action.dest
        for action in diagnostic.build_parser()._actions  # noqa: SLF001
    }
    assert not any(
        marker in name
        for marker in protected
        for name in parser_destinations
    )
