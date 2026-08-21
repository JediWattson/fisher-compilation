from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_chart_native_neighbor_experiment as worker
from fisher_graph.gemma3_chart_native_neighbor_experiment import (
    CONFIRMATION_ROLE,
    DISCOVERY_ROLE,
    FIT_ROLE,
    RESPONSE_ROLE,
    ChartNativeExecutionBinding,
    ExecutionSourcePin,
    ExecutionStorageBudget,
    LengthBucketBatchSchedule,
    build_capture_call_ledger,
    build_frozen_stage_receipt,
    build_role_eligibility_catalog,
    freeze_C_confirmed_edge_intersection,
    publish_experiment_outputs,
    render_neighbor_svg,
    seal_experiment_report,
    strict_load_experiment_report_bytes,
)
from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import load_manifest
from fisher_graph.chunked_tied_lexical_nll import (
    FixtureChunkedTiedLexicalNLLConfig,
    fixture_chunked_softcap_nll_vjp_rows,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_small_shared_atlas(path: Path):
    from fisher_graph.shared_chart_energy_transport import (
        FIT_ROLE as SHARED_FIT_ROLE,
        ChartRowKey,
        SharedChartNumericsConfig,
        SharedChartRowBatch,
        fit_shared_chart_atlas,
        save_shared_chart_atlas,
    )

    keys = tuple(
        ChartRowKey(
            f"family.{family}",
            f"example.{family}.{example}",
            query,
            query + 1,
        )
        for family in range(2)
        for example in range(2)
        for query in range(3)
    )
    batches = []
    for mode in range(3):
        generator = torch.Generator().manual_seed(700 + mode)
        context = torch.randn(len(keys), 4, generator=generator, dtype=torch.float64)
        context[: len(keys) // 2, 0] -= 3.0
        context[len(keys) // 2 :, 0] += 3.0
        comparison = 0.25 * context + float(mode + 1)
        contribution = comparison * torch.randn(
            len(keys), 4, generator=generator, dtype=torch.float64
        )
        batches.append(
            SharedChartRowBatch(
                role=SHARED_FIT_ROLE,
                panel_binding_sha256=_sha("small-atlas-panel"),
                mode_id=f"mode.{mode}",
                row_keys=keys,
                context_values=context,
                comparison_values=comparison,
                generalized_contributions=contribution,
            )
        )
    atlas = fit_shared_chart_atlas(
        tuple(batches),
        config=SharedChartNumericsConfig(
            hidden_width=4,
            chart_count=2,
            tangent_rank=2,
        ),
    )
    save_shared_chart_atlas(atlas, path)
    return atlas


def _source_pins() -> tuple[ExecutionSourcePin, ...]:
    return tuple(
        ExecutionSourcePin(
            name=name,
            path=f"src/{name}.py",
            file_sha256=_sha(name),
        )
        for name in sorted(worker._REQUIRED_SOURCE_NAMES)
    )


def test_live_schedule_and_resource_budget_fit_the_16gib_host() -> None:
    schedule = LengthBucketBatchSchedule()
    assert schedule.buckets == (
        (96, 2),
        (511, 1),
        (512, 1),
        (513, 1),
        (514, 1),
        (576, 1),
        (640, 1),
        (768, 1),
    )
    budget = ExecutionStorageBudget()
    assert budget.maximum_resident_memory_bytes == 8 * 1024**3
    assert budget.minimum_physical_memory_bytes == 12 * 1024**3
    assert budget.minimum_free_bytes == 24 * 1024**3
    assert budget.fit_tensor_temp_bytes == 20 * 1024**3
    assert budget.transport_artifact_bytes == 979_025_920
    ledger = build_capture_call_ledger(schedule=schedule)
    assert ledger.per_role_length_batch_counts == (
        (96, 3),
        (511, 6),
        (512, 6),
        (513, 6),
        (514, 6),
        (576, 6),
        (640, 6),
        (768, 6),
    )
    assert ledger.chart_batches_per_role == 45
    assert ledger.total_chart_forward_count == 135
    assert ledger.total_summed_shifted_nll_backward_count == 135
    assert ledger.gradient_target_count_per_backward == 73 + 19 == 92
    assert ledger.D_chart_capture_count == 0
    from fisher_graph.streaming_shared_chart_energy_transport import (
        estimate_streaming_chart_resources,
    )

    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=24_732,
        mode_count=38,
    )
    assert estimate.raw_spool_payload_bytes_per_role == 7_217_786_880
    assert estimate.applied_store_payload_bytes_per_role == 3_518_671_104
    assert estimate.maximum_phase_payload_and_atomic_temp_bytes == 10_829_054_592
    assert estimate.all_ordered_coefficient_bytes == 979_025_920
    assert estimate.atlas_peak_resident_bytes < budget.maximum_resident_memory_bytes


def test_static_preflight_rejects_an_already_imported_model_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = ChartNativeExecutionBinding(
        repository_root=tmp_path,
        setup_path=str(worker.DEFAULT_SETUP_PATH),
        source_pins=_source_pins(),
        schedule=LengthBucketBatchSchedule(),
        storage_budget=ExecutionStorageBudget(),
        output_directory=tmp_path,
        temp_directory=tmp_path,
        test_only_allow_nonproduction_shape=True,
    )
    monkeypatch.setitem(
        worker.sys.modules,
        "fisher_graph.gemma3_chart_native_neighbor_runtime",
        ModuleType("fisher_graph.gemma3_chart_native_neighbor_runtime"),
    )
    with pytest.raises(RuntimeError, match="before static preflight"):
        worker.static_preflight(binding)


def test_first_A_capture_canary_hard_stops_before_spools_and_later_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manifest = load_manifest()
    preflight = SimpleNamespace(
        receipt_sha256=_sha("canary-preflight"),
        fresh_manifest=manifest,
        fresh_receipt={},
        opaque_panel_receipt=object(),
        validate_receipt=lambda: events.append("preflight:validate"),
    )
    monkeypatch.setattr(
        worker,
        "static_preflight",
        lambda _binding: (
            events.append("preflight:static"),
            preflight,
        )[1],
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    original_open = SequentialPanelSource.open_role

    def tracked_open(self: object, role: str) -> dict[str, object]:
        events.append(f"panel:open:{role}")
        return original_open(self, role)  # type: ignore[arg-type]

    monkeypatch.setattr(SequentialPanelSource, "open_role", tracked_open)
    mode_ids = (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )
    lineage = tuple(
        worker.RuntimeModeLineage(
            mode_id=mode_id,
            state_fingerprint_sha256=_sha(f"state:{mode_id}"),
            site_placement_sha256=_sha(f"site:{mode_id}"),
            comparison_adapter_receipt_sha256=_sha(f"adapter:{mode_id}"),
        )
        for mode_id in mode_ids
    )
    qualification = SimpleNamespace(
        qualification_receipt_sha256=_sha("canary-qualification"),
        source_model_fingerprint_after_compilation_sha256=_sha("model"),
        source_execution_fingerprint_after_compilation_sha256=_sha("execution"),
        compiled_execution_fingerprint_sha256=_sha("compiled"),
        tied_readout_binding_receipt_sha256=_sha("readout"),
        mode_lineage=lineage,
    )
    session_sha = worker.derive_runtime_session_binding_sha256(
        preflight_receipt_sha256=preflight.receipt_sha256,
        runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        mode_lineage=lineage,
    )
    monkeypatch.setattr(
        worker,
        "validate_runtime_qualification",
        lambda **_kwargs: events.append("qualification:validate"),
    )
    monkeypatch.setattr(
        worker,
        "build_core_mode_specs",
        lambda **_kwargs: tuple(
            SimpleNamespace(mode_id=mode_id) for mode_id in mode_ids
        ),
    )

    class FakeRuntime:
        hidden_width = 640
        mode_lineage = lineage
        runtime_session_binding_sha256 = session_sha
        counter_overrides: dict[str, object] = {}
        tamper_row_keys = False
        artifact_path: Path | None = None
        close_failures_remaining = 0

        @property
        def counters(self) -> object:
            payload = {
                "role_batch_counts": [
                    [FIT_ROLE, 1],
                    [DISCOVERY_ROLE, 0],
                    [CONFIRMATION_ROLE, 0],
                ],
                "chart_backbone_forward_count": 1,
                "chart_terminal_cotangent_vjp_count": 1,
                "dense_parity_smoke_count": 1,
                "parity_backbone_forward_count": 1,
                "parity_terminal_cotangent_vjp_count": 1,
                "captured_example_count": 2,
                "supplemental_route_receipt_count": 36,
                "canonical_gradient_targets_per_vjp": 73,
                "auxiliary_gradient_targets_per_vjp": 19,
                "gradient_targets_per_vjp": 92,
                "contains_prompt_text": False,
                "contains_token_ids": False,
                "contains_raw_activations": False,
                "contains_raw_gradients": False,
                "contains_logits": False,
            }
            payload.update(type(self).counter_overrides)
            return SimpleNamespace(
                payload=lambda: payload,
            )

        def runtime_qualification_receipt(self) -> object:
            events.append("qualification:read")
            return qualification

        def run_dense_lexical_parity_smoke(self, **_kwargs: object) -> str:
            events.append("parity:A0")
            return _sha("canary-parity")

        def capture_batch(
            self,
            *,
            role: str,
            raw_examples: object,
            **_kwargs: object,
        ) -> worker.CapturedBatch:
            rows = tuple(raw_examples)  # type: ignore[arg-type]
            assert role == FIT_ROLE
            assert len(rows) == 2
            assert all(row["target_token_count"] == 96 for row in rows)
            events.append("capture:A:96:b2")
            if type(self).artifact_path is not None:
                type(self).artifact_path.touch()
            tensor = torch.zeros(95, 640, dtype=torch.float32)
            examples = []
            for row in rows:
                family_id = str(row["family_id"])
                example_id = str(row["example_id"])
                modes = tuple(
                    worker.CapturedModeRows(
                        mode_id=mode_id,
                        context_values=tensor,
                        comparison_values=tensor,
                        generalized_contributions=tensor,
                        supplemental_route_catalog_sha256=(
                            _sha(
                                "catalog:"
                                + (
                                    "global"
                                    if mode_id
                                    in {
                                        "mode.mutation.11",
                                        "mode.mutation.23",
                                        "mode.mutation.35",
                                    }
                                    else "sliding_512"
                                )
                                + f":{example_id}"
                            )
                            if mode_id.startswith("mode.mutation.")
                            and int(mode_id.rsplit(".", 1)[1]) % 2
                            else None
                        ),
                        supplemental_route_receipt_sha256=(
                            _sha(f"receipt:{mode_id}:{example_id}")
                            if mode_id.startswith("mode.mutation.")
                            and int(mode_id.rsplit(".", 1)[1]) % 2
                            else None
                        ),
                        supplemental_route_simplex_audit_sha256=(
                            _sha(f"audit:{mode_id}:{example_id}")
                            if mode_id.startswith("mode.mutation.")
                            and int(mode_id.rsplit(".", 1)[1]) % 2
                            else None
                        ),
                    )
                    for mode_id in mode_ids
                )
                examples.append(
                    worker.CapturedExampleRows(
                        family_id=family_id,
                        example_id=example_id,
                        token_count=96,
                        row_keys=tuple(
                            (family_id, example_id, query, query + 1)
                            for query in range(95)
                        ),
                        modes=modes,
                        context_fingerprint_sha256=_sha(f"context:{example_id}"),
                        capture_receipt_sha256=_sha(f"capture:{example_id}"),
                    )
                )
            if type(self).tamper_row_keys:
                object.__setattr__(
                    examples[0],
                    "row_keys",
                    (
                        (
                            examples[0].family_id,
                            examples[0].example_id,
                            0,
                            2,
                        ),
                        *examples[0].row_keys[1:],
                    ),
                )
            return worker.CapturedBatch(
                role=FIT_ROLE,
                examples=tuple(examples),
                lexical_nll_receipt_sha256=_sha("canary-nll"),
                execution_receipt_sha256=_sha("canary-execution"),
            )

        def close(self) -> None:
            events.append("runtime:close")
            if type(self).close_failures_remaining:
                type(self).close_failures_remaining -= 1
                raise RuntimeError("injected canary close failure")

    original_builder = worker.build_first_A_capture_canary_receipt

    def tracked_builder(**values: object) -> worker.FirstACaptureCanaryReceipt:
        events.append("canary:seal")
        return original_builder(**values)

    monkeypatch.setattr(
        worker,
        "build_first_A_capture_canary_receipt",
        tracked_builder,
    )
    output_directory = tmp_path / "output"
    temp_directory = tmp_path / "temp"
    output_directory.mkdir()
    temp_directory.mkdir()
    binding = SimpleNamespace(
        repository_root=Path.cwd(),
        output_directory=output_directory,
        temp_directory=temp_directory,
    )
    before = tuple(tmp_path.iterdir())
    receipt = worker.run_first_A_capture_canary(
        binding=binding,  # type: ignore[arg-type]
        runtime_factory=lambda _preflight: (
            events.append("runtime:create"),
            FakeRuntime(),
        )[1],
    )
    assert events == [
        "preflight:static",
        "preflight:validate",
        "runtime:create",
        "qualification:read",
        "qualification:validate",
        f"panel:open:{FIT_ROLE}",
        "parity:A0",
        "capture:A:96:b2",
        "runtime:close",
        "canary:seal",
    ]
    assert tuple(tmp_path.iterdir()) == before
    assert receipt.total_backbone_forward_count == 2
    assert receipt.total_terminal_cotangent_VJP_count == 2
    assert receipt.supplemental_route_receipt_count == 36
    assert receipt.B_semantically_opened is False
    assert receipt.C_semantically_opened is False
    assert receipt.D_semantically_opened is False
    assert receipt.reserve_semantically_opened is False
    assert receipt.published_output_count == 0
    assert receipt.contains_prompt_text is False
    assert receipt.contains_token_ids is False
    assert receipt.contains_raw_tensors is False
    assert receipt.state_dict()["canary_receipt_sha256"] == (
        receipt.canary_receipt_sha256
    )
    with pytest.raises(TypeError, match="exact integers"):
        worker.build_first_A_capture_canary_receipt(
            **{
                **{
                    field_name: getattr(receipt, field_name)
                    for field_name in receipt.__dataclass_fields__
                    if field_name != "canary_receipt_sha256"
                },
                "chart_backbone_forward_count": True,
            }
        )
    receipt_values = {
        field_name: getattr(receipt, field_name)
        for field_name in receipt.__dataclass_fields__
        if field_name != "canary_receipt_sha256"
    }
    within_group_forge = list(receipt.route_evidence)
    mode_id, example_id, _, route_sha, audit_sha = within_group_forge[1]
    within_group_forge[1] = (
        mode_id,
        example_id,
        _sha("forged-within-group-catalog"),
        route_sha,
        audit_sha,
    )
    with pytest.raises(ValueError, match="route inventory drifted"):
        worker.build_first_A_capture_canary_receipt(
            **{
                **receipt_values,
                "route_evidence": tuple(within_group_forge),
            }
        )
    cross_group_forge = list(receipt.route_evidence)
    first_sliding_catalog = cross_group_forge[0][2]
    first_example_id = cross_group_forge[0][1]
    for index, row in enumerate(cross_group_forge):
        mode_id, example_id, _, route_sha, audit_sha = row
        if example_id == first_example_id and mode_id in {
            "mode.mutation.11",
            "mode.mutation.23",
            "mode.mutation.35",
        }:
            cross_group_forge[index] = (
                mode_id,
                example_id,
                first_sliding_catalog,
                route_sha,
                audit_sha,
            )
    with pytest.raises(ValueError, match="route inventory drifted"):
        worker.build_first_A_capture_canary_receipt(
            **{
                **receipt_values,
                "route_evidence": tuple(cross_group_forge),
            }
        )

    FakeRuntime.counter_overrides = {"chart_backbone_forward_count": 2}
    with pytest.raises(ValueError, match="observed runtime ledger"):
        worker.run_first_A_capture_canary(
            binding=binding,  # type: ignore[arg-type]
            runtime_factory=lambda _preflight: FakeRuntime(),
        )
    FakeRuntime.counter_overrides = {}

    FakeRuntime.tamper_row_keys = True
    with pytest.raises(RuntimeError, match="mode rows drifted"):
        worker.run_first_A_capture_canary(
            binding=binding,  # type: ignore[arg-type]
            runtime_factory=lambda _preflight: FakeRuntime(),
        )
    FakeRuntime.tamper_row_keys = False

    injected_artifact = temp_directory / "chart-native-neighbor-work-v1"
    FakeRuntime.artifact_path = injected_artifact
    with pytest.raises(RuntimeError, match="failure path changed an artifact"):
        worker.run_first_A_capture_canary(
            binding=binding,  # type: ignore[arg-type]
            runtime_factory=lambda _preflight: FakeRuntime(),
        )
    injected_artifact.unlink()
    FakeRuntime.artifact_path = None

    close_events_before = events.count("runtime:close")
    FakeRuntime.close_failures_remaining = 1
    with pytest.raises(RuntimeError, match="injected canary close failure"):
        worker.run_first_A_capture_canary(
            binding=binding,  # type: ignore[arg-type]
            runtime_factory=lambda _preflight: FakeRuntime(),
        )
    assert events.count("runtime:close") == close_events_before + 2
    with pytest.raises(ValueError, match="execution ledger"):
        replace(receipt, total_backbone_forward_count=3)


@pytest.mark.parametrize("softcap", [None, 4.0])
def test_chunked_softcap_nll_vjp_matches_dense_float64_autograd(
    softcap: float | None,
) -> None:
    generator = torch.Generator().manual_seed(81_020_026)
    queries = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(13, 5, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 12, 3, 8, 5, 1, 9], dtype=torch.int64)
    config = FixtureChunkedTiedLexicalNLLConfig(
        prediction_position_chunk_size=2,
        vocabulary_chunk_size=4,
    )
    loss, gradient = fixture_chunked_softcap_nll_vjp_rows(
        queries,
        labels,
        weight,
        logit_softcap=softcap,
        config=config,
    )
    dense_queries = queries.clone().requires_grad_(True)
    raw = dense_queries @ weight.T
    logits = raw if softcap is None else softcap * torch.tanh(raw / softcap)
    expected_loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="sum",
    )
    expected_gradient = torch.autograd.grad(expected_loss, dense_queries)[0]
    assert loss == pytest.approx(
        float(expected_loss.detach()),
        rel=1e-13,
        abs=1e-13,
    )
    torch.testing.assert_close(
        gradient,
        expected_gradient,
        rtol=1e-12,
        atol=1e-12,
    )


def test_live_binding_rejects_relaxed_schedule_and_rss_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="length schedule"):
        ChartNativeExecutionBinding(
            repository_root=tmp_path,
            setup_path="setup.json",
            source_pins=_source_pins(),
            schedule=LengthBucketBatchSchedule(buckets=((96, 4),)),
            storage_budget=ExecutionStorageBudget(),
            output_directory=tmp_path,
            temp_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="budget"):
        ChartNativeExecutionBinding(
            repository_root=tmp_path,
            setup_path="setup.json",
            source_pins=_source_pins(),
            schedule=LengthBucketBatchSchedule(),
            storage_budget=replace(
                ExecutionStorageBudget(),
                maximum_resident_memory_bytes=9 * 1024**3,
            ),
            output_directory=tmp_path,
            temp_directory=tmp_path,
        )


def test_manifest_drives_exact_seven_primary_and_diagnostic_subsets() -> None:
    manifest = load_manifest()
    catalog = build_role_eligibility_catalog(
        manifest,
        role=DISCOVERY_ROLE,
    )
    assert all(
        len(examples) == 8
        for examples in catalog.mapping("all").values()
    )
    assert all(
        len(examples) == 7
        for examples in catalog.mapping("primary").values()
    )
    assert all(
        len(examples) == 1
        for examples in catalog.mapping("transition").values()
    )
    assert all(
        example.endswith("-514")
        for examples in catalog.mapping("transition").values()
        for example in examples
    )
    assert all(
        len(examples) == 3
        for examples in catalog.mapping("hard_long").values()
    )
    assert len(catalog.catalog_sha256) == 64
    with pytest.raises(ValueError, match="digest"):
        replace(catalog, catalog_sha256=_sha("tampered"))

    fit = build_role_eligibility_catalog(manifest, role=FIT_ROLE)
    assert all(len(rows) == 8 for rows in fit.mapping("fit").values())
    assert all(len(rows) == 0 for rows in fit.mapping("primary").values())
    assert all(len(rows) == 0 for rows in fit.mapping("response").values())

    response = build_role_eligibility_catalog(manifest, role=RESPONSE_ROLE)
    assert all(len(rows) == 8 for rows in response.mapping("response").values())
    assert all(len(rows) == 0 for rows in response.mapping("fit").values())
    assert all(len(rows) == 0 for rows in response.mapping("primary").values())


def test_capture_batch_plan_is_exact_three_short_pairs_then_42_singletons() -> None:
    examples = []
    for family in range(6):
        for length, _ in worker._EXPECTED_PRODUCTION_SCHEDULE:
            examples.append(
                {
                    "example_id": f"f{family}-{length}",
                    "family_id": f"family-{family}",
                    "target_token_count": length,
                }
            )
    plan = worker._capture_batch_plan({"examples": examples})
    assert len(plan) == 45
    assert tuple(len(batch) for batch in plan[:3]) == (2, 2, 2)
    assert all(len(batch) == 1 for batch in plan[3:])
    assert {
        int(example["target_token_count"])
        for batch in plan[:3]
        for example in batch
    } == {96}
    assert [int(batch[0]["target_token_count"]) for batch in plan[3:]] == [
        length
        for length, _ in worker._EXPECTED_PRODUCTION_SCHEDULE[1:]
        for _ in range(6)
    ]


def test_C_confirmation_is_strict_B_core_and_hard_long_intersection() -> None:
    confirmed, exploratory = freeze_C_confirmed_edge_intersection(
        frozen_B_candidates=(
            ("mode.a", "mode.b"),
            ("mode.c", "mode.d"),
        ),
        C_confirmation_core=(
            ("mode.a", "mode.b"),
            ("mode.c", "mode.d"),
            ("mode.e", "mode.f"),
        ),
        C_hard_long_veto_pass=(
            ("mode.a", "mode.b"),
            ("mode.e", "mode.f"),
        ),
    )
    assert confirmed == (("mode.a", "mode.b"),)
    assert exploratory == (("mode.e", "mode.f"),)


def test_frozen_stage_receipt_is_typed_chained_and_tamper_evident() -> None:
    receipt = build_frozen_stage_receipt(
        role=CONFIRMATION_ROLE,
        preflight_receipt_sha256=_sha("preflight"),
        role_panel_binding_sha256_value=_sha("panel-C"),
        ordered_objective_row_catalog_sha256=_sha("q-t-C"),
        score_objective_receipt_sha256=_sha("objective-C"),
        prior_stage_receipt_sha256=_sha("B-stage"),
        artifact_sha256s={
            "confirmed_graph": _sha("graph"),
            "hard_long_veto": _sha("long"),
        },
        primary_eligible_example_catalog_sha256=_sha("eligible-C"),
    )
    assert receipt.stage_kind == "chart_neighbor_graph_and_gates"
    assert tuple(name for name, _ in receipt.artifact_sha256s) == (
        "confirmed_graph",
        "hard_long_veto",
    )
    with pytest.raises(ValueError, match="does not authenticate"):
        replace(receipt, prior_stage_receipt_sha256=_sha("other-B"))


def test_report_guard_rejects_private_rows_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="leak private row data"):
        seal_experiment_report({"token_ids": [1, 2, 3]})
    with pytest.raises(ValueError, match="non-finite"):
        seal_experiment_report({"metric": float("nan")})
    with pytest.raises(ValueError, match="raw tensor"):
        seal_experiment_report({"metric": object()})


def test_report_svg_and_atlas_publish_atomically_without_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas = _write_small_shared_atlas(atlas_source)
    atlas_file_sha = hashlib.sha256(atlas_source.read_bytes()).hexdigest()
    svg = render_neighbor_svg(
        mode_ids=("mode.a", "mode.b", "mode.c"),
        confirmed_edges=(("mode.a", "mode.b"),),
    )
    report = seal_experiment_report(
        {
            "A_fit": {
                "atlas_artifact_file_sha256": atlas_file_sha,
                "atlas_sha256": atlas.atlas_sha256,
                "published_atlas_file_name": worker.DEFAULT_ATLAS_OUTPUT.name,
            },
            "confirmed_edges": [["mode.a", "mode.b"]],
            "mode_axis": ["mode.a", "mode.b", "mode.c"],
            "neighbor_svg_sha256": hashlib.sha256(svg).hexdigest(),
        }
    )
    encoded = worker._canonical_json_bytes(report)
    assert worker._strict_load_experiment_report_envelope_bytes(encoded) == report
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda value: None,
    )
    monkeypatch.setattr(
        worker,
        "_strict_rehash_frozen_repository_inputs",
        lambda **values: None,
    )
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name
    receipt = publish_experiment_outputs(
        report=report,
        svg_bytes=svg,
        atlas_source_path=atlas_source,
        report_path=report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    )
    assert receipt.report_sha256 == report["report_sha256"]
    assert receipt.atlas_sha256 == atlas.atlas_sha256
    assert worker.strict_load_experiment_report(
        report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    ) == report
    assert svg_path.read_bytes() == svg
    assert atlas_path.read_bytes() == atlas_source.read_bytes()
    with pytest.raises(FileExistsError):
        publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )


@pytest.mark.parametrize("failure_link", (2, 3))
def test_triple_publication_rolls_back_if_sibling_or_commit_link_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_link: int,
) -> None:
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas = _write_small_shared_atlas(atlas_source)
    svg = render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    report = seal_experiment_report(
        {
            "A_fit": {
                "atlas_artifact_file_sha256": hashlib.sha256(
                    atlas_source.read_bytes()
                ).hexdigest(),
                "atlas_sha256": atlas.atlas_sha256,
                "published_atlas_file_name": worker.DEFAULT_ATLAS_OUTPUT.name,
            },
            "confirmed_edges": [],
            "mode_axis": ["mode.a"],
            "neighbor_svg_sha256": hashlib.sha256(svg).hexdigest(),
        }
    )
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda value: None,
    )
    monkeypatch.setattr(
        worker,
        "_strict_rehash_frozen_repository_inputs",
        lambda **values: None,
    )
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name
    real_link = worker.os.link
    calls = 0

    def fail_publication_link(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_link:
            raise OSError("injected publication failure")
        real_link(source, destination)

    monkeypatch.setattr(worker.os, "link", fail_publication_link)
    with pytest.raises(OSError, match="injected"):
        publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()


def test_explicit_run_workspace_cleanup_removes_only_direct_child(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "one-run"
    nested = workspace / "nested"
    nested.mkdir(parents=True)
    (nested / "receipt.bin").write_bytes(b"sealed")
    worker._safe_remove_tree(workspace, allowed_parent=tmp_path)
    assert not workspace.exists()
    outside = tmp_path.parent / "outside"
    with pytest.raises(ValueError, match="direct run workspace"):
        worker._safe_remove_tree(outside, allowed_parent=tmp_path)


def test_report_cap_covers_production_B_C_directed_example_rows() -> None:
    directed_rows_per_role = 38 * 37 * 48
    assert directed_rows_per_role == 67_488
    # The strict cap leaves 64 MiB for graph/bootstrap/LOFO receipts while
    # budgeting 3 KiB for every B and C directed example row.
    conservative = 2 * directed_rows_per_role * 3_072 + 64 * 1024**2
    assert conservative == 481_755_136
    assert conservative < worker._MAX_REPORT_BYTES == 512 * 1024**2
