from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_VOCABULARY_CHUNK_SIZE,
    EXPECTED_LAYER_COUNT,
    REFIT_LAYER_ORDINALS,
    REFIT_START_LAYER,
    _collect_layer_rows_under_prefix,
    _live_split_matches_frozen,
    _prefix_catalog_sha256,
    _publish_verified_refit,
    _sequentially_refit_layers,
    _validate_frozen_source_bindings,
    _validate_runner_preflight,
    build_parser,
)
from fisher_graph.gemma3_full_mlp_stack_rows import FullMLPStackLayerRows
from fisher_graph.streaming_analysis import ActivationScoreGradientRows


_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"


def _h(value: int) -> str:
    return f"{value:064x}"


def _metrics(nll: float = 2.0) -> dict[str, float]:
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": nll - 2.0,
        "native_to_candidate_kl_per_token": max(nll - 2.0, 0.0),
        "top1_agreement_to_native": 1.0 if nll == 2.0 else 0.9,
    }


def _batch(batch_size: int, *, start: int = 0) -> CalibrationBatch:
    return CalibrationBatch(
        model_inputs={
            "input_ids": torch.arange(
                start,
                start + 2 * batch_size,
                dtype=torch.long,
            ).reshape(batch_size, 2)
        },
        targets=torch.zeros((batch_size, 2), dtype=torch.long),
        valid_positions=torch.ones((batch_size, 2), dtype=torch.bool),
        example_ids=tuple(
            f"example-{start + index}" for index in range(batch_size)
        ),
    )


def _rows(
    ordinal: int,
    *,
    row_hash: str,
    sequences: int = 2,
) -> FullMLPStackLayerRows:
    return FullMLPStackLayerRows(
        layer_ordinal=ordinal,
        layer_id=f"layer.{ordinal}",
        input_site=f"layer.{ordinal}.input",
        activation_site=f"layer.{ordinal}.activation",
        output_site=f"layer.{ordinal}.output",
        intermediate_width=2,
        fragment_ids=(f"fragment.{ordinal}",),
        fragment_sha256s=(_h(1000 + ordinal),),
        inputs=torch.ones((2, 2), dtype=torch.float64),
        contributions=torch.ones((2, 2), dtype=torch.float64),
        fisher_weights=torch.ones(2, dtype=torch.float64),
        row_keys=((f"{row_hash}-a", 0), (f"{row_hash}-b", 1)),
        sequences=sequences,
    )


def _preflight(tmp_path: Path, **overrides: object) -> None:
    values: dict[str, object] = {
        "revision": _REVISION,
        "output": tmp_path / "refit.pt",
        "base_artifact_path": tmp_path / "source.pt",
        "trajectory_artifact_path": tmp_path / "trajectory.json",
        "model_id": "google/gemma-3-270m",
        "device_name": "cpu",
        "dtype": "float32",
        "max_length": 256,
        "tokenization_batch_size": 1,
        "vocabulary_chunk_size": DEFAULT_VOCABULARY_CHUNK_SIZE,
        "refit_start_layer": REFIT_START_LAYER,
    }
    values.update(overrides)
    _validate_runner_preflight(**values)


def test_preflight_accepts_fixed_refit_protocol(tmp_path: Path) -> None:
    _preflight(tmp_path)


def test_live_split_comparison_ignores_separately_bound_annotations() -> None:
    live = {
        "schema": "fisher_graph.tokenized_calibration_stream",
        "serialized_sha256": _h(1),
        "content_sha256": (_h(2),),
    }
    frozen = {
        **live,
        "role": "generator_fit",
    }

    _live_split_matches_frozen(live, frozen, label="fit")

    with pytest.raises(ValueError, match="live fit tokenization differs"):
        _live_split_matches_frozen(
            {**live, "serialized_sha256": _h(3)},
            frozen,
            label="fit",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"revision": "main"}, "exact lowercase commit"),
        ({"base_artifact_path": "source.json"}, "must use .pt"),
        ({"trajectory_artifact_path": "trajectory.pt"}, "must use .json"),
        ({"output": "refit.json"}, "must use .pt"),
        ({"refit_start_layer": 9}, "layers 10 through 17"),
        ({"tokenization_batch_size": 0}, "must be positive"),
        ({"vocabulary_chunk_size": 0}, "must be positive"),
    ),
)
def test_preflight_rejects_protocol_drift(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _preflight(tmp_path, **override)


def test_preflight_refuses_tensor_or_sidecar_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "refit.pt"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _preflight(tmp_path, output=output)
    output.unlink()
    output.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _preflight(tmp_path, output=output)


def _frozen_bindings() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    fit_export = {"artifact_sha256": _h(1)}
    eval_export = {"artifact_sha256": _h(2)}
    partition = {
        "artifact_sha256": _h(3),
        "selection_prompt_count": 20,
        "expected_prompt_count": 40,
    }
    assessment = {
        "role": "open_development_assessment",
        "serialized_sha256": _h(4),
        "content_sha256": (_h(5), _h(6)),
        "sequences": 2,
        "valid_tokens": {"total": 20},
        "supervised_positions": {"total": 18},
    }
    source = {
        "schema": "source.schema",
        "format_version": 1,
        "scientific_payload_sha256": _h(7),
        "model": {
            "model_id": "google/gemma-3-270m",
            "requested_revision": _REVISION,
            "resolved_commit": _REVISION,
            "adapter_model_fingerprint": _h(8),
            "source_whole_model_learned_parameters": 1000,
            "local_files_only": True,
        },
        "protocol": {
            "scope": "full_native_mlp_stack_replacement",
            "transformer_layer_count": 18,
            "mode_ranks": (640,),
            "selected_mode_rank": 640,
            "generator_ranks": (640,),
            "selected_generator_rank": 640,
            "local_files_only": True,
        },
        "splits": {
            "fit_export": fit_export,
            "eval_export": eval_export,
            "partition": partition,
            "assessment": assessment,
        },
        "evaluation": {
            "conditions": {
                "native": _metrics(),
                "generated_full_stack": _metrics(2.3),
                "matched_deletion": _metrics(9.0),
            }
        },
    }
    prefix = tuple(
        {
            "depth": depth,
            "metrics": (
                _metrics(2.3)
                if depth == 18
                else _metrics(2.0 + depth / 100)
            ),
            "resources": {
                "replaced_layer_ordinals": tuple(range(depth)),
            },
        }
        for depth in range(1, 19)
    )
    trajectory = {
        "frozen_source_artifact": {
            "source_schema": "source.schema",
            "source_format_version": 1,
            "source_scope": "full_native_mlp_stack_replacement",
            "artifact_file_sha256": _h(9),
            "scientific_payload_sha256": _h(7),
            "frozen_before_trajectory": True,
        },
        "model": {
            **source["model"],
            "native_mlp_stack_learned_parameters": 500,
            "native_mlp_stack_linear_macs_per_token": 500,
        },
        "protocol": {
            "scope": "frozen_full_native_mlp_stack_trajectory_ladder",
            "transformer_layer_count": 18,
            "generators_frozen": True,
            "generator_refit_performed": False,
            "generator_rank_selection_performed": False,
        },
        "splits": {
            "assessment": {
                "role": "open_development_assessment",
                "serialized_sha256": _h(4),
                "content_sha256": (_h(5), _h(6)),
                "example_count": 2,
                "logical_valid_tokens": 20,
                "supervised_tokens": 18,
            }
        },
        "evaluation": {
            "native": _metrics(),
            "prefix_ladder": prefix,
        },
    }
    return source, trajectory, fit_export, eval_export


def test_frozen_bindings_authenticate_source_trajectory_and_breakpoint() -> None:
    source, trajectory, fit_export, eval_export = _frozen_bindings()
    result = _validate_frozen_source_bindings(
        source,
        trajectory,
        source_file_sha256=_h(9),
        revision=_REVISION,
        model_id="google/gemma-3-270m",
        fit_export_metadata=fit_export,
        eval_export_metadata=eval_export,
        partition_metadata=source["splits"]["partition"],  # type: ignore[index]
    )
    assert result[4]["depth"] == 10
    assert result[4]["resources"]["replaced_layer_ordinals"] == tuple(  # type: ignore[index]
        range(10)
    )


def test_frozen_bindings_reject_source_file_or_breakpoint_drift() -> None:
    source, trajectory, fit_export, eval_export = _frozen_bindings()
    with pytest.raises(ValueError, match="exact frozen source"):
        _validate_frozen_source_bindings(
            source,
            trajectory,
            source_file_sha256=_h(10),
            revision=_REVISION,
            model_id="google/gemma-3-270m",
            fit_export_metadata=fit_export,
            eval_export_metadata=eval_export,
            partition_metadata=source["splits"]["partition"],  # type: ignore[index]
        )
    trajectory["evaluation"]["prefix_ladder"][9]["resources"][  # type: ignore[index]
        "replaced_layer_ordinals"
    ] = tuple(range(9))
    with pytest.raises(ValueError, match="layer-10 breakpoint"):
        _validate_frozen_source_bindings(
            source,
            trajectory,
            source_file_sha256=_h(9),
            revision=_REVISION,
            model_id="google/gemma-3-270m",
            fit_export_metadata=fit_export,
            eval_export_metadata=eval_export,
            partition_metadata=source["splits"]["partition"],  # type: ignore[index]
        )


def test_frozen_bindings_reject_trajectory_full_endpoint_drift() -> None:
    source, trajectory, fit_export, eval_export = _frozen_bindings()
    trajectory["evaluation"]["prefix_ladder"][-1]["metrics"] = _metrics(  # type: ignore[index]
        2.31
    )
    with pytest.raises(
        ValueError,
        match="trajectory full-stack endpoint .* differs",
    ):
        _validate_frozen_source_bindings(
            source,
            trajectory,
            source_file_sha256=_h(9),
            revision=_REVISION,
            model_id="google/gemma-3-270m",
            fit_export_metadata=fit_export,
            eval_export_metadata=eval_export,
            partition_metadata=source["splits"]["partition"],  # type: ignore[index]
        )


def test_collect_rows_consumes_one_overlay_and_counts_samples() -> None:
    fragments = (
        SimpleNamespace(
            layer_ordinal=10,
            input_site="layer.10.input",
            activation_site="layer.10.activation",
        ),
    )
    batches = (_batch(2), _batch(1, start=2))
    calls: dict[str, object] = {}

    class Executor:
        def run_with_subset_overlay(
            self,
            *,
            generated_layer_ordinals: object,
            callback: object,
            expected_forward_calls: int,
        ) -> object:
            calls["prefix"] = generated_layer_ordinals
            calls["expected"] = expected_forward_calls
            calls["callbacks"] = int(calls.get("callbacks", 0)) + 1
            assert callable(callback)
            return callback()

    def row_factory(
        _adapter: object,
        materialized: object,
        **kwargs: object,
    ) -> object:
        calls["materialized"] = materialized
        calls["activation_names"] = kwargs["activation_names"]
        calls["leaf"] = kwargs["leaf_activation_name"]
        names = tuple(kwargs["activation_names"])  # type: ignore[arg-type]
        return (
            ActivationScoreGradientRows(
                activations={
                    name: torch.ones((2, 2), dtype=torch.float64)
                    for name in names
                },
                score_gradients={
                    name: torch.ones((2, 2), dtype=torch.float64)
                    for name in names
                },
                logical_positions=torch.tensor([0, 1]),
                loss=1.0,
                example_id="example",
            ),
        )

    def collector(rows: object, **_kwargs: object) -> FullMLPStackLayerRows:
        selected = tuple(rows)  # type: ignore[arg-type]
        assert set(selected[0].activations) == {
            "layer.10.input",
            "layer.10.activation",
        }
        return _rows(10, row_hash="collected")

    result = _collect_layer_rows_under_prefix(
        object(),  # type: ignore[arg-type]
        batches,
        fragments=fragments,  # type: ignore[arg-type]
        down_projection_weight=torch.eye(2),
        executor=Executor(),  # type: ignore[arg-type]
        generated_layer_ordinals=tuple(range(10)),
        row_factory=row_factory,  # type: ignore[arg-type]
        row_collector=collector,
    )
    assert result.layer_ordinal == 10
    assert calls["callbacks"] == 1
    assert calls["expected"] == 3
    assert calls["prefix"] == tuple(range(10))
    assert calls["activation_names"] == (
        "layer.10.input",
        "layer.10.activation",
    )
    assert calls["leaf"] == "layer.10.input"


def test_sequential_refit_freezes_each_new_plan_into_next_prefix() -> None:
    resources = {
        "native_mlp_parameter_count": 300,
        "dense_fused_parameter_count": 100,
        "dense_fused_macs_per_token": 90,
        "extra_authenticated_resource": 7,
    }

    def plan(ordinal: int, *, refit: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            artifact_sha256=_h(10_000 + ordinal + (100 if refit else 0)),
            rank=640,
            parameter_count=100,
            macs_per_token=90,
        )

    replacements = tuple(
        SimpleNamespace(layer_ordinal=ordinal, generator_plan=plan(ordinal))
        for ordinal in range(EXPECTED_LAYER_COUNT)
    )
    records = tuple(
        {
            "layer_ordinal": ordinal,
            "source_fit_sha256": _h(20_000 + ordinal),
            "dense_plan_sha256": plan(ordinal).artifact_sha256,
            "selected_mode_rank": 640,
            "selected_generator_rank": 640,
            "native_mlp_parameter_count": 300,
            "dense_fused_parameter_count": 100,
            "dense_fused_macs_per_token": 90,
        }
        for ordinal in range(EXPECTED_LAYER_COUNT)
    )

    class FragmentPlan:
        def for_layer(self, ordinal: int) -> tuple[SimpleNamespace, ...]:
            return (SimpleNamespace(layer_ordinal=ordinal),)

    class SuperfragmentPlan:
        layer_count = EXPECTED_LAYER_COUNT
        artifact_sha256 = _h(30_000)

        def for_layer(self, ordinal: int) -> SimpleNamespace:
            return SimpleNamespace(
                layer_ordinal=ordinal,
                channel_indices=(0, 1),
            )

    observed_prefixes: list[tuple[int, tuple[int, ...]]] = []
    executors: list[tuple[str, ...]] = []

    def executor_factory(
        _adapter: object,
        current: tuple[SimpleNamespace, ...],
    ) -> SimpleNamespace:
        executors.append(
            tuple(value.generator_plan.artifact_sha256 for value in current)
        )
        return SimpleNamespace()

    def collect_rows(
        _adapter: object,
        _batches: object,
        *,
        fragments: tuple[SimpleNamespace, ...],
        generated_layer_ordinals: tuple[int, ...],
        **_kwargs: object,
    ) -> FullMLPStackLayerRows:
        ordinal = fragments[0].layer_ordinal
        observed_prefixes.append((ordinal, generated_layer_ordinals))
        return _rows(
            ordinal,
            row_hash=f"{ordinal}-{len(observed_prefixes)}",
        )

    def fit_layer(
        _fit_rows: FullMLPStackLayerRows,
        _selection_rows: FullMLPStackLayerRows,
        *,
        superfragment: SimpleNamespace,
        **_kwargs: object,
    ) -> SimpleNamespace:
        ordinal = superfragment.layer_ordinal
        return SimpleNamespace(
            superfragment=superfragment,
            selected_basis=SimpleNamespace(rank=640),
            executable_plan=plan(ordinal, refit=True),
            artifact_sha256=_h(40_000 + ordinal),
            resource_metadata=resources,
        )

    def replacement_factory(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def plan_metrics(
        rows: FullMLPStackLayerRows,
        candidate: SimpleNamespace,
    ) -> dict[str, object]:
        return {
            "observations": rows.observations,
            "plan_sha256": candidate.artifact_sha256,
        }

    final, fits, layer_refits = _sequentially_refit_layers(
        object(),  # type: ignore[arg-type]
        (_batch(1),),
        (_batch(1, start=1),),
        fragment_plan=FragmentPlan(),  # type: ignore[arg-type]
        superfragment_plan=SuperfragmentPlan(),  # type: ignore[arg-type]
        down_projection_weights={
            ordinal: torch.eye(2)
            for ordinal in range(EXPECTED_LAYER_COUNT)
        },
        source_replacements=replacements,  # type: ignore[arg-type]
        source_fit_records=records,
        source_model_sha256=_h(1),
        parameter_catalog_sha256=_h(2),
        fisher_coupling_sha256=_h(3),
        fit_split_sha256=_h(4),
        selection_split_sha256=_h(5),
        mode_ranks=(640,),
        selected_mode_rank=640,
        generator_ranks=(640,),
        selected_generator_rank=640,
        generator_ridge=1e-6,
        trajectory_executor_factory=executor_factory,
        collect_rows=collect_rows,
        fit_layer=fit_layer,  # type: ignore[arg-type]
        plan_metrics=plan_metrics,  # type: ignore[arg-type]
        replacement_factory=replacement_factory,  # type: ignore[arg-type]
    )
    assert len(fits) == len(REFIT_LAYER_ORDINALS)
    assert tuple(row["layer_ordinal"] for row in layer_refits) == (
        REFIT_LAYER_ORDINALS
    )
    assert all(
        prefix == tuple(range(ordinal))
        for ordinal, prefix in observed_prefixes
    )
    assert final[9].generator_plan.artifact_sha256 == plan(9).artifact_sha256
    assert final[10].generator_plan.artifact_sha256 == plan(
        10,
        refit=True,
    ).artifact_sha256
    assert executors[1][10] == plan(10, refit=True).artifact_sha256
    assert layer_refits[1]["generated_prefix_plan_sha256s"][10] == plan(  # type: ignore[index]
        10,
        refit=True,
    ).artifact_sha256
    assert layer_refits[0]["generated_prefix_catalog_sha256"] == (
        _prefix_catalog_sha256(
            tuple(plan(index).artifact_sha256 for index in range(10))
        )
    )
    assert layer_refits[0]["old_plan_fit_metrics"]["plan_sha256"] == plan(  # type: ignore[index]
        10
    ).artifact_sha256
    assert layer_refits[0]["refit_plan_fit_metrics"][  # type: ignore[index]
        "plan_sha256"
    ] == plan(10, refit=True).artifact_sha256


def test_sequential_refit_rejects_resource_drift() -> None:
    replacements = tuple(
        SimpleNamespace(
            layer_ordinal=ordinal,
            generator_plan=SimpleNamespace(
                artifact_sha256=_h(100 + ordinal),
                rank=640,
                parameter_count=100,
                macs_per_token=90,
            ),
        )
        for ordinal in range(18)
    )
    records = tuple(
        {
            "layer_ordinal": ordinal,
            "source_fit_sha256": _h(200 + ordinal),
            "dense_plan_sha256": _h(100 + ordinal),
            "selected_mode_rank": 640,
            "selected_generator_rank": 640,
            "native_mlp_parameter_count": 300,
            "dense_fused_parameter_count": 100,
            "dense_fused_macs_per_token": 90,
        }
        for ordinal in range(18)
    )
    fragment_plan = SimpleNamespace(
        for_layer=lambda ordinal: (SimpleNamespace(layer_ordinal=ordinal),)
    )
    superfragment_plan = SimpleNamespace(
        layer_count=18,
        artifact_sha256=_h(300),
        for_layer=lambda ordinal: SimpleNamespace(
            layer_ordinal=ordinal,
            channel_indices=(0, 1),
        ),
    )
    fake_rows = lambda *_args, fragments, **_kwargs: _rows(  # noqa: E731
        fragments[0].layer_ordinal,
        row_hash=f"rows-{fragments[0].layer_ordinal}-{id(_kwargs)}",
    )

    def drifting_fit(
        _fit: object,
        _selection: object,
        *,
        superfragment: SimpleNamespace,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            superfragment=superfragment,
            selected_basis=SimpleNamespace(rank=640),
            executable_plan=SimpleNamespace(
                artifact_sha256=_h(400 + superfragment.layer_ordinal),
                rank=640,
                parameter_count=101,
                macs_per_token=90,
            ),
            artifact_sha256=_h(500 + superfragment.layer_ordinal),
            resource_metadata={
                "native_mlp_parameter_count": 300,
                "dense_fused_parameter_count": 101,
                "dense_fused_macs_per_token": 90,
            },
        )

    with pytest.raises(ValueError, match="changed layer rank"):
        _sequentially_refit_layers(
            object(),  # type: ignore[arg-type]
            (_batch(1),),
            (_batch(1, start=1),),
            fragment_plan=fragment_plan,  # type: ignore[arg-type]
            superfragment_plan=superfragment_plan,  # type: ignore[arg-type]
            down_projection_weights={
                ordinal: torch.eye(2) for ordinal in range(18)
            },
            source_replacements=replacements,  # type: ignore[arg-type]
            source_fit_records=records,
            source_model_sha256=_h(1),
            parameter_catalog_sha256=_h(2),
            fisher_coupling_sha256=_h(3),
            fit_split_sha256=_h(4),
            selection_split_sha256=_h(5),
            mode_ranks=(640,),
            selected_mode_rank=640,
            generator_ranks=(640,),
            selected_generator_rank=640,
            generator_ridge=1e-6,
            trajectory_executor_factory=lambda *_args: object(),
            collect_rows=fake_rows,  # type: ignore[arg-type]
            fit_layer=drifting_fit,  # type: ignore[arg-type]
            plan_metrics=lambda *_args: {"observations": 2},
            replacement_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        )


@pytest.mark.parametrize("drift", ("model", "source", "trajectory"))
def test_publish_removes_owned_outputs_on_postcondition_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    source = tmp_path / "source.pt"
    trajectory = tmp_path / "trajectory.json"
    source.write_bytes(b"source")
    trajectory.write_bytes(b"trajectory")
    source_sha256 = __import__("hashlib").sha256(b"source").hexdigest()
    trajectory_sha256 = __import__("hashlib").sha256(
        b"trajectory"
    ).hexdigest()
    output = tmp_path / "refit.pt"

    class Adapter:
        calls = 0

        def model_fingerprint(self) -> str:
            self.calls += 1
            return _h(99) if drift == "model" else _h(1)

    def save(path: Path | str, **_kwargs: object) -> dict[str, object]:
        Path(path).write_bytes(b"refit")
        Path(path).with_suffix(".json").write_text("{}", encoding="utf-8")
        if drift == "source":
            source.write_bytes(b"changed")
        if drift == "trajectory":
            trajectory.write_bytes(b"changed")
        return {"saved": True}

    with pytest.raises(RuntimeError, match="mutated|changed"):
        _publish_verified_refit(
            output=output,
            adapter=Adapter(),  # type: ignore[arg-type]
            expected_model_fingerprint=_h(1),
            source_path=source,
            expected_source_file_sha256=source_sha256,
            trajectory_path=trajectory,
            expected_trajectory_file_sha256=trajectory_sha256,
            save=save,
            load=lambda _path: {},
            save_kwargs={},
        )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_parser_exposes_fixed_refit_defaults() -> None:
    arguments = build_parser().parse_args(["--revision", _REVISION])
    assert arguments.dtype == "float32"
    assert arguments.device == "cpu"
    assert arguments.vocabulary_chunk_size == DEFAULT_VOCABULARY_CHUNK_SIZE
