from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import fisher_graph.gemma3_chart_native_neighbor_experiment as worker
from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
    PINNED_ROLE_FILE_SHA256S,
    load_manifest,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash_report(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = worker._digest(worker._REPORT_DOMAIN, report)


def _rehash_resource(resource: dict[str, object]) -> None:
    resource.pop("receipt_sha256", None)
    resource["receipt_sha256"] = worker._digest(
        worker._RESOURCE_DOMAIN,
        resource,
    )


def _partial_publication_report(
    *,
    svg: bytes,
    atlas_raw: bytes,
    atlas_sha256: str,
) -> dict[str, object]:
    return worker.seal_experiment_report(
        {
            "A_fit": {
                "atlas_artifact_file_sha256": hashlib.sha256(
                    atlas_raw
                ).hexdigest(),
                "atlas_sha256": atlas_sha256,
                "published_atlas_file_name": worker.DEFAULT_ATLAS_OUTPUT.name,
            },
            "confirmed_edges": [],
            "mode_axis": ["mode.a"],
            "neighbor_svg_sha256": hashlib.sha256(svg).hexdigest(),
        }
    )


def _patch_fake_atlas_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    atlas_sha256: str,
) -> None:
    from fisher_graph import shared_chart_energy_transport as shared

    monkeypatch.setattr(
        shared,
        "load_shared_chart_atlas",
        lambda _path: SimpleNamespace(atlas_sha256=atlas_sha256),
    )


def _write_tiny_strict_atlas_bundle(destination: Path) -> object:
    import torch

    from fisher_graph.shared_chart_energy_transport import (
        FIT_ROLE,
        ChartRowKey,
        SharedChartNumericsConfig,
        SharedChartRowBatch,
        fit_shared_chart_atlas,
        save_shared_chart_atlas,
    )

    row_keys = tuple(
        ChartRowKey(
            f"family.{family}",
            f"example.{family}.{example}",
            query,
            query + 1,
        )
        for family in range(2)
        for example in range(2)
        for query in range(2)
    )
    batches = []
    for mode in range(3):
        generator = torch.Generator().manual_seed(8_100 + mode)
        context = torch.randn(
            len(row_keys),
            4,
            generator=generator,
            dtype=torch.float64,
        )
        comparison = 0.25 * context + float(mode + 1)
        score = torch.randn(
            len(row_keys),
            4,
            generator=generator,
            dtype=torch.float64,
        )
        batches.append(
            SharedChartRowBatch(
                role=FIT_ROLE,
                panel_binding_sha256=_sha("tiny-atlas-panel"),
                mode_id=f"mode.{mode:02d}",
                row_keys=row_keys,
                context_values=context,
                comparison_values=comparison,
                generalized_contributions=comparison * score,
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
    save_shared_chart_atlas(atlas, destination)
    return atlas


class _TinyCapturedTensor:
    live = 0
    maximum_live = 0

    def __init__(self) -> None:
        type(self).live += 1
        type(self).maximum_live = max(
            type(self).maximum_live,
            type(self).live,
        )

    def __del__(self) -> None:
        type(self).live -= 1

    def __getitem__(self, _item: object) -> _TinyCapturedTensor:
        return self

    def contiguous(self) -> _TinyCapturedTensor:
        return self

    def to(self, **_kwargs: object) -> _TinyCapturedTensor:
        return self


class _ModelFreeCaptureRuntime:
    hidden_width = 640

    def __init__(
        self,
        *,
        setup: Mapping[str, object],
        manifest: Mapping[str, object],
        role: str,
    ) -> None:
        self.runtime_session_binding_sha256 = _sha("capture-runtime-session")
        self.mode_lineage = tuple(
            worker.RuntimeModeLineage(
                mode_id=str(row["mode_id"]),
                state_fingerprint_sha256=_sha(
                    f"capture-state:{row['mode_id']}"
                ),
                site_placement_sha256=_sha(
                    f"capture-site:{row['mode_id']}"
                ),
                comparison_adapter_receipt_sha256=_sha(
                    f"capture-adapter:{row['mode_id']}"
                ),
            )
            for row in setup["logical_mode_ontology"]["ordered_modes"]
        )
        self._role = role
        self._members = {
            str(row["example_id"]): row
            for row in manifest["roles"][role]["ordered_members"]
        }
        self._batch_index = 0

    def capture_batch(
        self,
        *,
        preflight: object,
        role: str,
        raw_examples: object,
        role_manifest: object,
    ) -> worker.CapturedBatch:
        del preflight, role_manifest
        assert role == self._role
        gc.collect()
        rows: list[worker.CapturedExampleRows] = []
        for raw in raw_examples:
            example_id = str(raw["example_id"])
            member = self._members[example_id]
            family_id = str(member["family_id"])
            token_count = int(member["target_token_count"])
            marker = _TinyCapturedTensor()
            modes = tuple(
                worker.CapturedModeRows(
                    mode_id=mode_id,
                    context_values=marker,
                    comparison_values=marker,
                    generalized_contributions=marker,
                    supplemental_route_catalog_sha256=(
                        _sha(f"capture-route-catalog:{mode_id}:{example_id}")
                        if mode_id in _ATTENTION_MODE_IDS
                        else None
                    ),
                    supplemental_route_receipt_sha256=(
                        _sha(f"capture-route-receipt:{mode_id}:{example_id}")
                        if mode_id in _ATTENTION_MODE_IDS
                        else None
                    ),
                )
                for mode_id in _MODE_IDS
            )
            rows.append(
                worker.CapturedExampleRows(
                    family_id=family_id,
                    example_id=example_id,
                    token_count=token_count,
                    row_keys=tuple(
                        (family_id, example_id, query, query + 1)
                        for query in range(token_count - 1)
                    ),
                    modes=modes,
                    context_fingerprint_sha256=_sha(
                        f"capture-context:{example_id}"
                    ),
                    capture_receipt_sha256=_sha(
                        f"capture-example:{example_id}"
                    ),
                )
            )
        batch_index = self._batch_index
        self._batch_index += 1
        return worker.CapturedBatch(
            role=role,
            examples=tuple(rows),
            lexical_nll_receipt_sha256=_sha(f"capture-nll:{batch_index}"),
            execution_receipt_sha256=_sha(
                f"capture-execution:{batch_index}"
            ),
        )


def _exercise_production_capture_without_model(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reject_oversized_append: bool,
) -> tuple[object, list[int], list[object]]:
    from fisher_graph.gemma3_chart_native_neighbor_setup import build_setup
    from fisher_graph import streaming_shared_chart_energy_transport as streaming

    setup = build_setup()
    manifest = load_manifest()
    gc.collect()
    _TinyCapturedTensor.live = 0
    _TinyCapturedTensor.maximum_live = 0
    role = worker.DISCOVERY_ROLE
    runtime = _ModelFreeCaptureRuntime(
        setup=setup,
        manifest=manifest,
        role=role,
    )
    append_sizes: list[int] = []
    first_mode_keys: list[object] = []
    sealed_spools: dict[str, object] = {}

    class FakeDiskTensor:
        def __getitem__(self, _item: object) -> FakeDiskTensor:
            return self

    class FakeSpool:
        def __init__(
            self,
            *,
            path: Path,
            role: str,
            panel_binding_sha256: str,
            mode_id: str,
            row_keys: tuple[object, ...],
        ) -> None:
            self.path = path
            self.role = role
            self.panel_binding_sha256 = panel_binding_sha256
            self.mode_id = mode_id
            self.row_keys = row_keys
            self.artifact_sha256 = _sha(f"capture-spool:{path}")
            self.row_catalog_sha256 = _sha(f"capture-spool-rows:{path}")
            self.tensor_catalog_sha256 = _sha(
                f"capture-spool-tensors:{path}"
            )

        def iter_float64_chunks(
            self,
            *,
            chunk_size: int,
        ) -> object:
            for start in range(0, len(self.row_keys), chunk_size):
                keys = self.row_keys[start : start + chunk_size]
                yield SimpleNamespace(
                    row_keys=keys,
                    tensors={
                        "context_values": FakeDiskTensor(),
                        "comparison_values": FakeDiskTensor(),
                        "generalized_contributions": FakeDiskTensor(),
                    },
                )

    class FakeWriter:
        def __init__(
            self,
            *,
            path: Path,
            role: str,
            panel_binding_sha256: str,
            mode_id: str,
            row_keys: tuple[object, ...],
        ) -> None:
            self.path = path
            self.role = role
            self.panel_binding_sha256 = panel_binding_sha256
            self.mode_id = mode_id
            self.row_keys = row_keys
            self.appended_row_keys: list[object] = []
            self.is_fragment = path.parent.name == "sealed-example-fragments"

        @classmethod
        def create(cls, path: object, **kwargs: object) -> FakeWriter:
            resolved = Path(path)
            resolved.touch(exist_ok=False)
            return cls(
                path=resolved,
                role=str(kwargs["role"]),
                panel_binding_sha256=str(
                    kwargs["panel_binding_sha256"]
                ),
                mode_id=str(kwargs["mode_id"]),
                row_keys=tuple(kwargs["row_keys"]),
            )

        def append(
            self,
            row_keys: object,
            _context: object,
            _comparison: object,
            _contribution: object,
        ) -> None:
            materialized = tuple(row_keys)
            append_sizes.append(len(materialized))
            if reject_oversized_append and len(materialized) > 256:
                raise ValueError("raw spool append exceeds fixed 256-row chunk")
            self.appended_row_keys.extend(materialized)
            if not self.is_fragment and self.mode_id == _MODE_IDS[0]:
                first_mode_keys.extend(materialized)

        def seal(self) -> object:
            if tuple(self.appended_row_keys) != self.row_keys:
                raise ValueError("fake raw spool is incomplete or out of order")
            spool = FakeSpool(
                path=self.path,
                role=self.role,
                panel_binding_sha256=self.panel_binding_sha256,
                mode_id=self.mode_id,
                row_keys=self.row_keys,
            )
            sealed_spools[str(self.path)] = spool
            return spool

        def abort(self) -> None:
            return None

    monkeypatch.setattr(streaming.RawModeRowSpoolWriter, "create", FakeWriter.create)
    monkeypatch.setattr(
        streaming,
        "open_raw_mode_row_spool",
        lambda path, **_kwargs: sealed_spools[str(Path(path))],
    )
    monkeypatch.setattr(worker, "_validate_captured_mode_tensor", lambda *a, **k: None)
    destination = tmp_path / "raw"
    destination.mkdir()
    storage_budget = worker.ExecutionStorageBudget()
    binding = SimpleNamespace(
        storage_budget=storage_budget,
        test_only_allow_nonproduction_shape=False,
    )
    preflight = SimpleNamespace(
        binding=binding,
        fresh_manifest=manifest,
        setup=setup,
        validate_receipt=lambda: None,
    )
    raw_corpus = {
        "examples": [
            {
                "example_id": row["example_id"],
                "family_id": row["family_id"],
                "target_token_count": row["target_token_count"],
            }
            for row in manifest["roles"][role]["ordered_members"]
        ]
    }
    result = worker.capture_role_to_raw_spools(
        preflight=preflight,
        runtime=runtime,
        role=role,
        raw_corpus=raw_corpus,
        destination_directory=destination,
        dense_lexical_parity_receipt_sha256=None,
    )
    return result, append_sizes, first_mode_keys


def _capture_receipt(
    role: str,
    *,
    route_evidence: tuple[tuple[str, str, str, str], ...] | None = None,
) -> worker.RoleCaptureReceipt:
    modes = (
        ("mode.lexical.ingress",)
        + tuple(f"mode.mutation.{index}" for index in range(1, 37))
        + ("mode.lexical.egress",)
    )
    attention_modes = tuple(
        f"mode.mutation.{index}" for index in range(1, 37, 2)
    )
    examples = tuple(f"example-{index:02d}" for index in range(48))
    routes = route_evidence or tuple(
        (
            mode_id,
            example_id,
            _sha(f"route-catalog:{mode_id}:{example_id}"),
            _sha(f"route-receipt:{mode_id}:{example_id}"),
        )
        for mode_id in attention_modes
        for example_id in examples
    )
    lexical_receipts = tuple(
        _sha(f"nll:{role}:{index}") for index in range(45)
    )
    execution_receipts = tuple(
        _sha(f"execution:{role}:{index}") for index in range(45)
    )
    generic_batches = (
        (examples[0], examples[1]),
        (examples[2], examples[3]),
        (examples[4], examples[5]),
        *((example_id,) for example_id in examples[6:]),
    )
    runtime_binding_field = (
        "runtime_session_binding_sha256"
        if "runtime_session_binding_sha256"
        in worker.RoleCaptureReceipt.__dataclass_fields__
        else "runtime_receipt_sha256"
    )
    values = {
        "role": role,
        runtime_binding_field: _sha("runtime"),
        "role_panel_binding_sha256": _sha(f"panel:{role}"),
        "ordered_objective_row_catalog_sha256": _sha(f"q-t:{role}"),
        "score_objective_receipt_sha256": _sha(f"objective:{role}"),
        "mode_raw_artifacts": tuple(
            (
                mode_id,
                _sha(f"artifact:{role}:{mode_id}"),
                _sha(f"rows:{role}:{mode_id}"),
                _sha(f"tensors:{role}:{mode_id}"),
            )
            for mode_id in modes
        ),
        "batch_axis_catalog_sha256": worker._capture_batch_axis_catalog_sha256(
            role=role,
            batch_example_ids=generic_batches,
            lexical_nll_receipt_sha256s=lexical_receipts,
            execution_receipt_sha256s=execution_receipts,
            ordered_objective_row_catalog_sha256=_sha(f"q-t:{role}"),
            score_objective_receipt_sha256=_sha(f"objective:{role}"),
        ),
        "lexical_nll_receipt_sha256s": lexical_receipts,
        "execution_receipt_sha256s": execution_receipts,
        "example_capture_receipt_sha256s": tuple(
            (example_id, _sha(f"example:{role}:{example_id}"))
            for example_id in examples
        ),
        "supplemental_route_evidence": routes,
        "dense_lexical_parity_receipt_sha256": (
            _sha("dense-parity") if role == worker.FIT_ROLE else None
        ),
        "batch_count": 45,
        "row_count": 24_732,
    }
    provisional = object.__new__(worker.RoleCaptureReceipt)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return worker.RoleCaptureReceipt(
        **values,
        receipt_sha256=worker._digest(
            worker._CAPTURE_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


_MODE_IDS = (
    ("mode.lexical.ingress",)
    + tuple(f"mode.mutation.{index}" for index in range(1, 37))
    + ("mode.lexical.egress",)
)
_ATTENTION_MODE_IDS = tuple(
    f"mode.mutation.{index}" for index in range(1, 37, 2)
)


def _payload_with_digest(receipt: object, digest_field: str) -> dict[str, object]:
    payload = dict(receipt.payload_without_sha256())
    payload[digest_field] = getattr(receipt, digest_field)
    return payload


def _resource_receipt(
    phase_id: str,
    *,
    completed: int = 0,
    total: int = 0,
    ordinal: int = 1,
) -> dict[str, object]:
    payload = {
        "completed_pair_count": completed,
        "observed_monotonic_ns": ordinal,
        "peak_resident_set_bytes": 1,
        "phase_id": phase_id,
        "temp_free_bytes": 32 * 1024**3,
        "temp_unique_regular_bytes": 1,
        "total_pair_count": total,
    }
    return {
        **payload,
        "receipt_sha256": worker._digest(worker._RESOURCE_DOMAIN, payload),
    }


class _FakeSubsetReceipt:
    def __init__(
        self,
        *,
        role: str,
        source_key: str,
        subset_id: str,
        point_edges: list[list[str]],
    ) -> None:
        self.analysis = {
            "authorizes_edges": False,
            "point_gated_edges": point_edges,
        }
        self.receipt_sha256 = _sha(
            f"diagnostic:{role}:{source_key}:{subset_id}:{point_edges}"
        )
        self._payload = {
            "analysis": self.analysis,
            "authorizes_edges": False,
            "receipt_sha256": self.receipt_sha256,
            "source_key": source_key,
            "subset_id": subset_id,
        }

    def canonical_payload(self) -> dict[str, object]:
        return self._payload


class _ModelFreeCompletedReportHarness:
    """Production-cardinality outer receipt with numeric core calls stubbed.

    The graph core has its own exact parser/rebuild adversarial suite.  This
    harness isolates the worker's public strict loader and keeps the expensive
    model and 10,000-replicate bootstrap out of report-lineage tests while
    preserving 38 modes, 48 examples, 67,488 directed rows per heldout role,
    all four diagnostic subsets, and every outer digest/stage chain.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.manifest = load_manifest()
        self.setup = json.loads(Path(worker.DEFAULT_SETUP_PATH).read_text())
        self._source_cache: dict[bytes, object] = {}
        model_free_source_pins = dict(worker.PINNED_SOURCE_SHA256S)
        model_free_source_pins["chart_native_neighbor_worker"] = (
            hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()
        )
        monkeypatch.setattr(
            worker,
            "PINNED_SOURCE_SHA256S",
            model_free_source_pins,
        )
        monkeypatch.setattr(
            worker,
            "_REQUIRED_SOURCE_NAMES",
            frozenset(model_free_source_pins),
        )

        from fisher_graph import chart_energy_neighbor_graph as core

        monkeypatch.setattr(core, "_parse_input", self.parse_source)
        monkeypatch.setattr(
            core,
            "parse_chart_energy_neighbor_input_payload",
            self.parse_source,
        )
        monkeypatch.setattr(
            worker,
            "_primary_input_from_all_eight",
            self.primary_from_source,
        )
        # This legacy outer-envelope harness deliberately uses lightweight
        # SimpleNamespace graph rows.  The frozen policy/q-t/shuffle contract
        # is exercised against the real core parser in the dedicated
        # real-parser audit; keep this harness focused on outer report lineage.
        monkeypatch.setattr(
            worker,
            "_validate_role_graph_input_contract",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            core,
            "build_chart_energy_neighbor_receipt",
            self.build_primary,
        )
        monkeypatch.setattr(
            core,
            "build_chart_energy_diagnostic_subset_receipt",
            self.build_subset,
        )
        monkeypatch.setattr(
            core,
            "parse_chart_energy_diagnostic_subset_receipt_payload",
            self.parse_subset,
        )
        self.frozen_runtime_qualification = self.runtime_qualification()
        monkeypatch.setattr(
            worker,
            "_load_frozen_runtime_qualification_from_setup",
            lambda **_kwargs: self.frozen_runtime_qualification,
        )

    def role_examples(self, role: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(row["family_id"]), str(row["example_id"]))
            for row in self.manifest["roles"][role]["ordered_members"]
        )

    def source_payload(
        self,
        role: str,
        *,
        atlas_sha256: str,
        transport_catalog_sha256: str,
        panel_binding_sha256: str,
        transport_receipt_nonce: str = "A",
        mode_state_nonce: str = "baseline",
        mode_site_nonce: str = "baseline",
        mode_adapter_nonce: str = "baseline",
        mode_native_abi_nonce: str = "baseline",
        mode_comparison_abi_nonce: str = "baseline",
        route_nonce: str = "baseline",
    ) -> dict[str, object]:
        return {
            "atlas_sha256": atlas_sha256,
            "mode_adapter_nonce": mode_adapter_nonce,
            "mode_native_abi_nonce": mode_native_abi_nonce,
            "mode_comparison_abi_nonce": mode_comparison_abi_nonce,
            "mode_site_nonce": mode_site_nonce,
            "mode_state_nonce": mode_state_nonce,
            "panel_binding_sha256": panel_binding_sha256,
            "role": role,
            "route_nonce": route_nonce,
            "transport_catalog_sha256": transport_catalog_sha256,
            "transport_receipt_nonce": transport_receipt_nonce,
        }

    def runtime_mode_lineage(
        self,
        *,
        state_nonce: str,
        site_nonce: str,
        adapter_nonce: str,
    ) -> tuple[worker.RuntimeModeLineage, ...]:
        derived = worker.derive_runtime_mode_lineage(
            setup_modes=self.setup["logical_mode_ontology"]["ordered_modes"],
            state_fingerprints_by_mode={
                mode_id: _sha(f"state:{state_nonce}:{mode_id}")
                for mode_id in _MODE_IDS
            },
            hidden_width=int(self.setup["source_model"]["hidden_width"]),
        )
        return tuple(
            worker.RuntimeModeLineage(
                mode_id=row.mode_id,
                state_fingerprint_sha256=row.state_fingerprint_sha256,
                site_placement_sha256=(
                    row.site_placement_sha256
                    if site_nonce == "baseline"
                    else _sha(f"site:{site_nonce}:{row.mode_id}")
                ),
                comparison_adapter_receipt_sha256=(
                    row.comparison_adapter_receipt_sha256
                    if adapter_nonce == "baseline"
                    else _sha(f"adapter:{adapter_nonce}:{row.mode_id}")
                ),
            )
            for row in derived
        )

    def runtime_qualification(
        self,
        *,
        mode_lineage: tuple[worker.RuntimeModeLineage, ...] | None = None,
        compiled_execution_fingerprint_sha256: str | None = None,
        tied_readout_binding_receipt_sha256: str | None = None,
        tied_lexical_lineage_sha256: str | None = None,
        compiler_receipt_sha256: str | None = None,
        final_logit_softcap: float | None = None,
        source_model_fingerprint_sha256: str | None = None,
        source_execution_fingerprint_sha256: str | None = None,
    ) -> worker.RuntimeQualificationReceipt:
        from fisher_graph.chunked_tied_lexical_nll import (
            chunked_tied_lexical_nll_kernel_receipt,
        )
        from fisher_graph.gemma3_chart_mode_surfaces import (
            factorized_attention_route_kernel_receipt,
        )
        from fisher_graph.streaming_shared_chart_energy_transport import (
            StreamingChartResourceConfig,
            streaming_chart_kernel_receipt,
        )

        source = self.setup["source_model"]
        modes = mode_lineage or self.runtime_mode_lineage(
            state_nonce="baseline",
            site_nonce="baseline",
            adapter_nonce="baseline",
        )
        model_fingerprint = (
            source_model_fingerprint_sha256
            or source["source_model_fingerprint_sha256"]
        )
        execution_fingerprint = (
            source_execution_fingerprint_sha256
            or source["source_execution_fingerprint_sha256"]
        )
        return worker.build_runtime_qualification_receipt(
            schema=(
                "fisher_graph.gemma3_chart_native_neighbor."
                "runtime_qualification.v1"
            ),
            format_version=1,
            model_id=source["model_id"],
            model_revision=source["model_revision"],
            source_model_fingerprint_before_compilation_sha256=(
                model_fingerprint
            ),
            source_model_fingerprint_after_compilation_sha256=(
                model_fingerprint
            ),
            source_execution_fingerprint_before_compilation_sha256=(
                execution_fingerprint
            ),
            source_execution_fingerprint_after_compilation_sha256=(
                execution_fingerprint
            ),
            source_model_mutated_during_compilation=False,
            source_parameter_count=source["parameter_count"],
            hidden_width=source["hidden_width"],
            compiled_execution_fingerprint_sha256=(
                compiled_execution_fingerprint_sha256
                or _sha("compiled-execution")
            ),
            compiler_receipt_sha256=(
                compiler_receipt_sha256 or _sha("compiler")
            ),
            tied_readout_binding_receipt_sha256=(
                tied_readout_binding_receipt_sha256
                or _sha("tied-readout")
            ),
            tied_lexical_lineage_sha256=(
                tied_lexical_lineage_sha256
                or _sha("tied-lexical-lineage")
            ),
            final_logit_softcap=final_logit_softcap,
            mode_lineage=modes,
            mode_lineage_catalog_sha256=(
                worker.runtime_mode_lineage_catalog_sha256(modes)
            ),
            lexical_kernel_receipt_sha256=(
                chunked_tied_lexical_nll_kernel_receipt()["receipt_sha256"]
            ),
            streaming_kernel_receipt_sha256=(
                streaming_chart_kernel_receipt(
                    StreamingChartResourceConfig(
                        max_resident_bytes=8 * 1024**3
                    )
                )["receipt_sha256"]
            ),
            factorized_route_kernel_receipt_sha256=(
                factorized_attention_route_kernel_receipt()[
                    "receipt_sha256"
                ]
            ),
            contains_prompt_text=False,
            contains_token_ids=False,
            contains_raw_activations=False,
            contains_raw_gradients=False,
            contains_logits=False,
        )

    def parse_source(self, value: object) -> object:
        if not isinstance(value, Mapping):
            raise ValueError("fake all-eight source must be a mapping")
        raw = dict(value)
        key = _canonical(raw)
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        role = str(raw["role"])
        runtime_rows = self.runtime_mode_lineage(
            state_nonce=str(raw["mode_state_nonce"]),
            site_nonce=str(raw["mode_site_nonce"]),
            adapter_nonce=str(raw["mode_adapter_nonce"]),
        )
        modes = worker.build_core_mode_specs(
            preflight=SimpleNamespace(
                setup=self.setup,
                validate_receipt=lambda: None,
            ),
            runtime=SimpleNamespace(mode_lineage=runtime_rows),
        )
        native_abi_nonce = str(raw["mode_native_abi_nonce"])
        if native_abi_nonce != "baseline":
            modes = tuple(
                replace(
                    mode,
                    native_abi=replace(
                        mode.native_abi,
                        input_signature=(
                            f"{mode.native_abi.abi_kind}.input."
                            f"{native_abi_nonce}"
                        ),
                    ),
                )
                for mode in modes
            )
        comparison_abi_nonce = str(raw["mode_comparison_abi_nonce"])
        if comparison_abi_nonce != "baseline":
            modes = tuple(
                replace(
                    mode,
                    comparison_abi=replace(
                        mode.comparison_abi,
                        row_semantics=(
                            f"carrier_chart_energy.rows."
                            f"{comparison_abi_nonce}"
                        ),
                    ),
                )
                for mode in modes
            )
        # Core graph payloads canonicalize the mode axis lexicographically;
        # runtime qualification remains in setup/event order above.
        modes = tuple(sorted(modes, key=lambda mode: mode.mode_id))
        examples: list[object] = []
        for family_id, example_id in self.role_examples(role):
            for mode_id in _MODE_IDS:
                route_catalog = (
                    _sha(
                        f"route-catalog:{raw['route_nonce']}:{role}:"
                        f"{mode_id}:{example_id}"
                    )
                    if mode_id in _ATTENTION_MODE_IDS
                    else None
                )
                route_receipt = (
                    _sha(
                        f"route-receipt:{raw['route_nonce']}:{role}:"
                        f"{mode_id}:{example_id}"
                    )
                    if mode_id in _ATTENTION_MODE_IDS
                    else None
                )
                examples.append(
                    SimpleNamespace(
                        family_id=family_id,
                        example_id=example_id,
                        mode_id=mode_id,
                        chart_atlas_sha256=raw["atlas_sha256"],
                        supplemental_route_catalog_sha256=route_catalog,
                        supplemental_route_receipt_sha256=route_receipt,
                        sha256=_sha(
                            f"example-stat:{key.hex()}:{family_id}:"
                            f"{example_id}:{mode_id}"
                        ),
                    )
                )
        directed: list[object] = []
        for family_id, example_id in self.role_examples(role):
            for target in _MODE_IDS:
                for source in _MODE_IDS:
                    if source == target:
                        continue
                    fit_receipt = _sha(
                        f"transport:{raw['transport_receipt_nonce']}:"
                        f"{source}:{target}"
                    )
                    directed.append(
                        SimpleNamespace(
                            family_id=family_id,
                            example_id=example_id,
                            source_mode_id=source,
                            target_mode_id=target,
                            transport_fit_catalog_sha256=raw[
                                "transport_catalog_sha256"
                            ],
                            transport_fit_receipt_sha256=fit_receipt,
                            row_shuffle_catalog_sha256=_sha(
                                f"shuffle:{role}"
                            ),
                            sha256=_sha(
                                f"directed:{key.hex()}:{family_id}:"
                                f"{example_id}:{source}:{target}"
                            ),
                        )
                    )
        from fisher_graph.chart_energy_neighbor_graph import (
            ChartEnergyNeighborConfig,
            ChartWithinFamilyBootstrapConfig,
        )

        eligibility = worker.build_role_eligibility_catalog(
            self.manifest,
            role=role,
        )
        bootstrap_contract = self.setup["fresh_confirmation_gates"][
            "within_family_bootstrap"
        ]
        source = SimpleNamespace(
            panel_role=role,
            panel_binding_sha256=raw["panel_binding_sha256"],
            mode_specs=modes,
            example_stats=tuple(examples),
            directed_transport_stats=tuple(directed),
            within_family_bootstrap_config=ChartWithinFamilyBootstrapConfig(
                seed=int(bootstrap_contract["seed"]),
                resample_count=int(bootstrap_contract["resample_count"]),
                draws_per_family=8,
                maximum_degenerate_fraction=float(
                    bootstrap_contract["maximum_degenerate_fraction"]
                ),
                family_example_ids=eligibility.all_family_example_ids,
            ),
            config=ChartEnergyNeighborConfig(),
            _fake_source_payload=raw,
            _fake_source_key=key.hex(),
        )
        self._source_cache[key] = source
        return source

    @staticmethod
    def primary_from_source(source: object, eligibility: object) -> object:
        del eligibility
        return SimpleNamespace(
            panel_role=source.panel_role,
            example_stats=(None,) * (38 * 42),
            directed_transport_stats=(None,) * (38 * 37 * 42),
            _fake_source=source,
        )

    @staticmethod
    def build_primary(primary: object) -> object:
        source = primary._fake_source
        role = source.panel_role
        edge = [[_MODE_IDS[0], _MODE_IDS[1]]]
        analysis = (
            {"point_gated_candidate_edges": edge}
            if role == worker.DISCOVERY_ROLE
            else {"confirmation_core_edges": edge}
        )
        return SimpleNamespace(
            analysis=analysis,
            input_sha256=_sha(f"primary-input:{source._fake_source_key}"),
            receipt_sha256=_sha(f"primary-receipt:{source._fake_source_key}"),
        )

    @staticmethod
    def build_subset(
        source: object,
        *,
        subset_id: str,
        family_example_ids: object,
    ) -> _FakeSubsetReceipt:
        del family_example_ids
        long_edges = (
            [[_MODE_IDS[0], _MODE_IDS[1]]]
            if subset_id == "hard_long_576_640_768_veto"
            else []
        )
        return _FakeSubsetReceipt(
            role=source.panel_role,
            source_key=source._fake_source_key,
            subset_id=subset_id,
            point_edges=long_edges,
        )

    def parse_subset(
        self,
        value: object,
        *,
        source_graph_input: object,
    ) -> _FakeSubsetReceipt:
        if not isinstance(value, Mapping) or not isinstance(
            value.get("subset_id"), str
        ):
            raise ValueError("fake diagnostic payload drifted")
        expected = self.build_subset(
            source_graph_input,
            subset_id=str(value["subset_id"]),
            family_example_ids=(),
        )
        if dict(value) != expected.canonical_payload():
            raise ValueError("fake diagnostic payload does not replay")
        return expected

    def capture(
        self,
        role: str,
        *,
        runtime_binding: str,
        panel_binding: str,
        route_nonce: str,
        ordered_objective_row_catalog_sha256: str | None = None,
        score_objective_receipt_sha256: str | None = None,
        example_ids_override: tuple[str, ...] | None = None,
        reverse_batch_axis: bool = False,
    ) -> worker.RoleCaptureReceipt:
        role_manifest = self.manifest["roles"][role]
        manifest_example_ids = tuple(
            str(row["example_id"]) for row in role_manifest["ordered_members"]
        )
        example_ids = example_ids_override or manifest_example_ids
        if len(example_ids) != 48:
            raise ValueError("capture test fixture requires exactly 48 examples")
        routes = tuple(
            sorted(
                (
                    mode_id,
                    example_id,
                    _sha(
                        f"route-catalog:{route_nonce}:{role}:"
                        f"{mode_id}:{example_id}"
                    ),
                    _sha(
                        f"route-receipt:{route_nonce}:{role}:"
                        f"{mode_id}:{example_id}"
                    ),
                )
                for mode_id in _ATTENTION_MODE_IDS
                for example_id in example_ids
            )
        )
        runtime_field = (
            "runtime_session_binding_sha256"
            if "runtime_session_binding_sha256"
            in worker.RoleCaptureReceipt.__dataclass_fields__
            else "runtime_receipt_sha256"
        )
        values = {
            "role": role,
            runtime_field: runtime_binding,
            "role_panel_binding_sha256": panel_binding,
            "ordered_objective_row_catalog_sha256": (
                ordered_objective_row_catalog_sha256
                or role_manifest["ordered_objective_row_catalog_sha256"]
            ),
            "score_objective_receipt_sha256": (
                score_objective_receipt_sha256
                or role_manifest["score_objective_receipt_sha256"]
            ),
            "mode_raw_artifacts": tuple(
                (
                    mode_id,
                    _sha(f"raw:{role}:{mode_id}"),
                    _sha(f"rows:{role}:{mode_id}"),
                    _sha(f"tensors:{role}:{mode_id}"),
                )
                for mode_id in _MODE_IDS
            ),
            "lexical_nll_receipt_sha256s": tuple(
                _sha(f"nll:{role}:{index}") for index in range(45)
            ),
            "execution_receipt_sha256s": tuple(
                _sha(f"execution:{role}:{index}") for index in range(45)
            ),
            "example_capture_receipt_sha256s": tuple(
                (example_id, _sha(f"capture:{role}:{example_id}"))
                for example_id in example_ids
            ),
            "supplemental_route_evidence": routes,
            "dense_lexical_parity_receipt_sha256": (
                _sha("A-dense-parity") if role == worker.FIT_ROLE else None
            ),
            "batch_count": 45,
            "row_count": 24_732,
        }
        if example_ids_override is None:
            plan_ids = tuple(
                tuple(str(row["example_id"]) for row in batch)
                for batch in worker._capture_batch_plan(
                    {
                        "examples": [
                            {
                                "example_id": row["example_id"],
                                "target_token_count": row[
                                    "target_token_count"
                                ],
                            }
                            for row in role_manifest["ordered_members"]
                        ]
                    }
                )
            )
        else:
            plan_ids = (
                (example_ids[0], example_ids[1]),
                (example_ids[2], example_ids[3]),
                (example_ids[4], example_ids[5]),
                *((example_id,) for example_id in example_ids[6:]),
            )
        if reverse_batch_axis:
            plan_ids = tuple(reversed(plan_ids))
        values["batch_axis_catalog_sha256"] = (
            worker._capture_batch_axis_catalog_sha256(
                role=role,
                batch_example_ids=plan_ids,
                lexical_nll_receipt_sha256s=values[
                    "lexical_nll_receipt_sha256s"
                ],
                execution_receipt_sha256s=values[
                    "execution_receipt_sha256s"
                ],
                ordered_objective_row_catalog_sha256=values[
                    "ordered_objective_row_catalog_sha256"
                ],
                score_objective_receipt_sha256=values[
                    "score_objective_receipt_sha256"
                ],
            )
        )
        provisional = object.__new__(worker.RoleCaptureReceipt)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "receipt_sha256", "0" * 64)
        return worker.RoleCaptureReceipt(
            **values,
            receipt_sha256=worker._digest(
                worker._CAPTURE_DOMAIN,
                provisional.payload_without_sha256(),
            ),
        )

    def build_report(
        self,
        *,
        B_transport_catalog: str | None = None,
        C_transport_catalog: str | None = None,
        B_transport_receipt_nonce: str = "A",
        C_transport_receipt_nonce: str = "A",
        B_panel_binding: str | None = None,
        C_panel_binding: str | None = None,
        B_mode_state_nonce: str = "baseline",
        C_mode_state_nonce: str = "baseline",
        B_mode_site_nonce: str = "baseline",
        C_mode_site_nonce: str = "baseline",
        B_mode_adapter_nonce: str = "baseline",
        C_mode_adapter_nonce: str = "baseline",
        B_mode_native_abi_nonce: str = "baseline",
        C_mode_native_abi_nonce: str = "baseline",
        B_mode_comparison_abi_nonce: str = "baseline",
        C_mode_comparison_abi_nonce: str = "baseline",
        B_route_nonce: str = "baseline",
        C_route_nonce: str = "baseline",
        B_capture_route_nonce: str | None = None,
        C_capture_route_nonce: str | None = None,
        A_objective_row_catalog_sha256: str | None = None,
        B_objective_row_catalog_sha256: str | None = None,
        C_objective_row_catalog_sha256: str | None = None,
        A_score_objective_receipt_sha256: str | None = None,
        B_score_objective_receipt_sha256: str | None = None,
        C_score_objective_receipt_sha256: str | None = None,
        runtime_binding_sha256: str | None = None,
        A_example_ids_override: tuple[str, ...] | None = None,
        A_reverse_batch_axis: bool = False,
        stage_preflight_receipt_sha256: str | None = None,
        execution_binding_sha256: str | None = None,
        execution_contract_attack: str | None = None,
        streaming_kernel_sha256: str | None = None,
        compiled_execution_fingerprint_sha256: str | None = None,
        tied_readout_binding_receipt_sha256: str | None = None,
        tied_lexical_lineage_sha256: str | None = None,
        compiler_receipt_sha256: str | None = None,
        final_logit_softcap: float | None = None,
        source_model_fingerprint_sha256: str | None = None,
        source_execution_fingerprint_sha256: str | None = None,
        source_pin_overrides: Mapping[str, str] | None = None,
        D_measured: object = False,
        D_measurement: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        from fisher_graph.streaming_shared_chart_energy_transport import (
            StreamingChartResourceConfig,
            streaming_chart_kernel_receipt,
        )
        from fisher_graph.chunked_tied_lexical_nll import (
            chunked_tied_lexical_nll_kernel_receipt,
        )
        from fisher_graph.gemma3_chart_mode_surfaces import (
            factorized_attention_route_kernel_receipt,
        )

        atlas_sha = _sha("shared-atlas")
        transport_rows = [
            {
                "byte_count": 1,
                "source_mode_id": source,
                "target_mode_id": target,
                "transport_sha256": _sha(f"transport:A:{source}:{target}"),
            }
            for target in sorted(_MODE_IDS)
            for source in sorted(_MODE_IDS)
            if source != target
        ]
        A_transport_catalog = worker._digest(
            worker._TRANSPORT_CATALOG_DOMAIN,
            transport_rows,
        )
        B_fit_catalog = B_transport_catalog or A_transport_catalog
        C_fit_catalog = C_transport_catalog or A_transport_catalog
        panel_bindings = {
            role: worker.role_panel_binding_sha256(self.manifest, role)
            for role in (
                worker.FIT_ROLE,
                worker.DISCOVERY_ROLE,
                worker.CONFIRMATION_ROLE,
                worker.RESPONSE_ROLE,
            )
        }
        B_panel = B_panel_binding or panel_bindings[worker.DISCOVERY_ROLE]
        C_panel = C_panel_binding or panel_bindings[worker.CONFIRMATION_ROLE]
        execution_contract = deepcopy(
            worker.production_execution_contract_payload()
        )
        effective_source_pin_overrides = dict(source_pin_overrides or {})
        if execution_contract_attack == "schedule":
            attacked_schedule = deepcopy(execution_contract["schedule"])
            attacked_schedule[0][1] += 1
            execution_contract["schedule"] = attacked_schedule
            execution_contract["schedule_sha256"] = worker._digest(
                b"fisher_graph:chart-native-batch-schedule:v1\0",
                attacked_schedule,
            )
        elif execution_contract_attack == "storage-budget":
            attacked_budget = deepcopy(execution_contract["storage_budget"])
            attacked_budget["maximum_resident_memory_bytes"] //= 2
            execution_contract["storage_budget"] = attacked_budget
        elif execution_contract_attack == "test-only":
            execution_contract["test_only_allow_nonproduction_shape"] = True
        elif execution_contract_attack == "source-pin":
            attacked_source_pins = deepcopy(execution_contract["source_pins"])
            target_pin = next(
                pin
                for pin in attacked_source_pins
                if pin["name"] != "chart_native_neighbor_worker"
            )
            substituted_sha256 = _sha("attacker-live-source-bytes")
            target_pin["file_sha256"] = substituted_sha256
            effective_source_pin_overrides[target_pin["name"]] = (
                substituted_sha256
            )
            execution_contract["source_pins"] = attacked_source_pins
        elif execution_contract_attack is not None:
            raise AssertionError(
                f"unknown execution-contract attack {execution_contract_attack!r}"
            )
        preflight_unsigned = {
            "binding_sha256": (
                execution_binding_sha256 or _sha("execution-binding")
            ),
            "execution_contract": execution_contract,
            "execution_contract_sha256": worker._digest(
                worker._EXECUTION_CONTRACT_DOMAIN,
                execution_contract,
            ),
            "format_version": 1,
            "free_bytes_by_device": [[1, 32 * 1024**3]],
            "fresh_manifest_sha256": (
                "fe045222c8d4f7ec21b14d158da63878c3462a1c38c6b9f9dcecf2b4d1d5e57f"
            ),
            "fresh_receipt_sha256": (
                "a1586d6601323390573e750e9732bc7db4e415eefc0f15bf0a2fa8ef820cf1f0"
            ),
            "opaque_role_file_sha256s": dict(PINNED_ROLE_FILE_SHA256S),
            "physical_memory_bytes": 16 * 1024**3,
            "reserve_semantically_decoded": False,
            "schema": worker.EXECUTION_PREFLIGHT_SCHEMA,
            "setup_sha256": worker.PINNED_SETUP_SHA256,
            "source_file_sha256s": {
                **worker.PINNED_SOURCE_SHA256S,
                "chart_native_neighbor_worker": hashlib.sha256(
                    Path(worker.__file__).read_bytes()
                ).hexdigest(),
                **effective_source_pin_overrides,
            },
            "transformers_or_model_runtime_touched_before_preflight": False,
        }
        preflight = {
            **preflight_unsigned,
            "receipt_sha256": worker._digest(
                worker._PREFLIGHT_DOMAIN,
                preflight_unsigned,
            ),
        }
        default_kernel = streaming_chart_kernel_receipt(
            StreamingChartResourceConfig(max_resident_bytes=8 * 1024**3)
        )["receipt_sha256"]
        kernel_sha = streaming_kernel_sha256 or default_kernel
        runtime_modes = self.runtime_mode_lineage(
            state_nonce=B_mode_state_nonce,
            site_nonce=B_mode_site_nonce,
            adapter_nonce=B_mode_adapter_nonce,
        )
        runtime_qualification = self.runtime_qualification(
            mode_lineage=runtime_modes,
            compiled_execution_fingerprint_sha256=(
                compiled_execution_fingerprint_sha256
            ),
            tied_readout_binding_receipt_sha256=(
                tied_readout_binding_receipt_sha256
            ),
            tied_lexical_lineage_sha256=tied_lexical_lineage_sha256,
            compiler_receipt_sha256=compiler_receipt_sha256,
            final_logit_softcap=final_logit_softcap,
            source_model_fingerprint_sha256=(
                source_model_fingerprint_sha256
            ),
            source_execution_fingerprint_sha256=(
                source_execution_fingerprint_sha256
            ),
        )
        runtime_binding = runtime_binding_sha256 or (
            worker.derive_runtime_session_binding_sha256(
                preflight_receipt_sha256=preflight["receipt_sha256"],
                runtime_qualification_receipt_sha256=(
                    runtime_qualification.qualification_receipt_sha256
                ),
                source_model_fingerprint_sha256=(
                    runtime_qualification
                    .source_model_fingerprint_after_compilation_sha256
                ),
                source_execution_fingerprint_sha256=(
                    runtime_qualification
                    .source_execution_fingerprint_after_compilation_sha256
                ),
                compiled_execution_fingerprint_sha256=(
                    runtime_qualification
                    .compiled_execution_fingerprint_sha256
                ),
                tied_readout_binding_receipt_sha256=(
                    runtime_qualification
                    .tied_readout_binding_receipt_sha256
                ),
                mode_lineage=runtime_qualification.mode_lineage,
            )
        )
        A_capture = self.capture(
            worker.FIT_ROLE,
            runtime_binding=runtime_binding,
            panel_binding=panel_bindings[worker.FIT_ROLE],
            route_nonce="baseline",
            ordered_objective_row_catalog_sha256=(
                A_objective_row_catalog_sha256
            ),
            score_objective_receipt_sha256=(
                A_score_objective_receipt_sha256
            ),
            example_ids_override=A_example_ids_override,
            reverse_batch_axis=A_reverse_batch_axis,
        )
        B_capture = self.capture(
            worker.DISCOVERY_ROLE,
            runtime_binding=runtime_binding,
            panel_binding=B_panel,
            route_nonce=B_capture_route_nonce or B_route_nonce,
            ordered_objective_row_catalog_sha256=(
                B_objective_row_catalog_sha256
            ),
            score_objective_receipt_sha256=(
                B_score_objective_receipt_sha256
            ),
        )
        C_capture = self.capture(
            worker.CONFIRMATION_ROLE,
            runtime_binding=runtime_binding,
            panel_binding=C_panel,
            route_nonce=C_capture_route_nonce or C_route_nonce,
            ordered_objective_row_catalog_sha256=(
                C_objective_row_catalog_sha256
            ),
            score_objective_receipt_sha256=(
                C_score_objective_receipt_sha256
            ),
        )
        source_model = self.setup["source_model"]
        source_model_fingerprint = (
            source_model_fingerprint_sha256
            or source_model["source_model_fingerprint_sha256"]
        )
        source_execution_fingerprint = (
            source_execution_fingerprint_sha256
            or source_model["source_execution_fingerprint_sha256"]
        )
        runtime_receipt = worker.build_runtime_execution_receipt(
            model_id=source_model["model_id"],
            model_revision=source_model["model_revision"],
            runtime_qualification_receipt_sha256=(
                runtime_qualification.qualification_receipt_sha256
            ),
            runtime_session_binding_sha256=runtime_binding,
            source_model_fingerprint_before_sha256=source_model_fingerprint,
            source_model_fingerprint_after_sha256=source_model_fingerprint,
            source_execution_fingerprint_before_sha256=(
                source_execution_fingerprint
            ),
            source_execution_fingerprint_after_sha256=(
                source_execution_fingerprint
            ),
            compiled_execution_fingerprint_sha256=(
                compiled_execution_fingerprint_sha256
                or _sha("compiled-execution")
            ),
            source_model_mutated=False,
            source_parameter_count=source_model["parameter_count"],
            hidden_width=source_model["hidden_width"],
            mode_lineage=runtime_modes,
            tied_readout_binding_receipt_sha256=(
                tied_readout_binding_receipt_sha256
                or _sha("tied-readout")
            ),
            tied_lexical_lineage_sha256=(
                tied_lexical_lineage_sha256
                or _sha("tied-lexical-lineage")
            ),
            final_logit_softcap=final_logit_softcap,
            compiler_receipt_sha256=(
                compiler_receipt_sha256 or _sha("compiler")
            ),
            lexical_kernel_receipt_sha256=(
                chunked_tied_lexical_nll_kernel_receipt()["receipt_sha256"]
            ),
            streaming_kernel_receipt_sha256=default_kernel,
            factorized_route_kernel_receipt_sha256=(
                factorized_attention_route_kernel_receipt()[
                    "receipt_sha256"
                ]
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
            observed_role_batch_counts=(
                (worker.FIT_ROLE, 45),
                (worker.DISCOVERY_ROLE, 45),
                (worker.CONFIRMATION_ROLE, 45),
            ),
            observed_chart_capture_backbone_forward_count=135,
            observed_chart_capture_terminal_cotangent_VJP_count=135,
            observed_dense_parity_backbone_forward_count=1,
            observed_dense_parity_terminal_cotangent_VJP_count=1,
            observed_total_backbone_forward_count=136,
            observed_total_terminal_cotangent_VJP_count=136,
            observed_dense_lexical_parity_smoke_count=1,
            observed_captured_example_count=144,
            observed_supplemental_route_receipt_count=3 * 18 * 48,
            contains_prompt_text=False,
            contains_token_ids=False,
            contains_raw_activations=False,
            contains_raw_gradients=False,
            contains_logits=False,
        )

        stage_preflight = (
            stage_preflight_receipt_sha256 or preflight["receipt_sha256"]
        )
        B_mode_specs = worker.build_core_mode_specs(
            preflight=SimpleNamespace(
                setup=self.setup,
                validate_receipt=lambda: None,
            ),
            runtime=SimpleNamespace(mode_lineage=runtime_modes),
        )
        if B_mode_native_abi_nonce != "baseline":
            B_mode_specs = tuple(
                replace(
                    mode,
                    native_abi=replace(
                        mode.native_abi,
                        input_signature=(
                            f"{mode.native_abi.abi_kind}.input."
                            f"{B_mode_native_abi_nonce}"
                        ),
                    ),
                )
                for mode in B_mode_specs
            )
        if B_mode_comparison_abi_nonce != "baseline":
            B_mode_specs = tuple(
                replace(
                    mode,
                    comparison_abi=replace(
                        mode.comparison_abi,
                        row_semantics=(
                            f"carrier_chart_energy.rows."
                            f"{B_mode_comparison_abi_nonce}"
                        ),
                    ),
                )
                for mode in B_mode_specs
            )
        mode_lineage_catalog_sha = worker._core_mode_lineage_catalog_sha256(
            B_mode_specs
        )
        A_stage = worker.build_frozen_stage_receipt(
            role=worker.FIT_ROLE,
            preflight_receipt_sha256=stage_preflight,
            role_panel_binding_sha256_value=panel_bindings[worker.FIT_ROLE],
            ordered_objective_row_catalog_sha256=A_capture.ordered_objective_row_catalog_sha256,
            score_objective_receipt_sha256=A_capture.score_objective_receipt_sha256,
            prior_stage_receipt_sha256=None,
            artifact_sha256s={
                "A_capture_receipt": A_capture.receipt_sha256,
                "A_mode_lineage_catalog": mode_lineage_catalog_sha,
                "A_shared_chart_atlas": atlas_sha,
                "A_shared_chart_atlas_file": _sha("atlas-file"),
                "A_streaming_kernel": kernel_sha,
                "A_transport_catalog": A_transport_catalog,
            },
            primary_eligible_example_catalog_sha256=None,
        )

        source_payloads = {
            worker.DISCOVERY_ROLE: self.source_payload(
                worker.DISCOVERY_ROLE,
                atlas_sha256=atlas_sha,
                transport_catalog_sha256=B_fit_catalog,
                panel_binding_sha256=B_panel,
                transport_receipt_nonce=B_transport_receipt_nonce,
                mode_state_nonce=B_mode_state_nonce,
                mode_site_nonce=B_mode_site_nonce,
                mode_adapter_nonce=B_mode_adapter_nonce,
                mode_native_abi_nonce=B_mode_native_abi_nonce,
                mode_comparison_abi_nonce=B_mode_comparison_abi_nonce,
                route_nonce=B_route_nonce,
            ),
            worker.CONFIRMATION_ROLE: self.source_payload(
                worker.CONFIRMATION_ROLE,
                atlas_sha256=atlas_sha,
                transport_catalog_sha256=C_fit_catalog,
                panel_binding_sha256=C_panel,
                transport_receipt_nonce=C_transport_receipt_nonce,
                mode_state_nonce=C_mode_state_nonce,
                mode_site_nonce=C_mode_site_nonce,
                mode_adapter_nonce=C_mode_adapter_nonce,
                mode_native_abi_nonce=C_mode_native_abi_nonce,
                mode_comparison_abi_nonce=C_mode_comparison_abi_nonce,
                route_nonce=C_route_nonce,
            ),
        }
        captures = {
            worker.DISCOVERY_ROLE: B_capture,
            worker.CONFIRMATION_ROLE: C_capture,
        }
        role_parts: dict[str, dict[str, Any]] = {}
        prior = A_stage.stage_receipt_sha256
        B_candidates: tuple[tuple[str, str], ...] = ()
        B_catalog = ""
        for role in (worker.DISCOVERY_ROLE, worker.CONFIRMATION_ROLE):
            source = self.parse_source(source_payloads[role])
            eligibility = worker.build_role_eligibility_catalog(
                self.manifest,
                role=role,
            )
            primary = self.build_primary(
                self.primary_from_source(source, eligibility)
            )
            diagnostics = {
                subset_id: self.build_subset(
                    source,
                    subset_id=subset_id,
                    family_example_ids=rows,
                )
                for subset_id, rows in (
                    ("short_96", eligibility.short_family_example_ids),
                    (
                        "boundary_511_512_513",
                        eligibility.boundary_family_example_ids,
                    ),
                    (
                        "boundary_transition_514_diagnostic_only",
                        eligibility.transition_family_example_ids,
                    ),
                    (
                        "hard_long_576_640_768_veto",
                        eligibility.hard_long_family_example_ids,
                    ),
                )
            }
            evaluation = worker._recompute_role_evaluation_catalog(
                source=source,
                capture=captures[role],
            )
            if role == worker.DISCOVERY_ROLE:
                B_candidates = tuple(
                    tuple(edge)
                    for edge in primary.analysis["point_gated_candidate_edges"]
                )
                B_catalog = worker._digest(
                    worker._CANDIDATE_DOMAIN,
                    {
                        "B_primary_input_sha256": primary.input_sha256,
                        "B_primary_receipt_sha256": primary.receipt_sha256,
                        "candidate_edges": [list(edge) for edge in B_candidates],
                        "eligibility_catalog_sha256": eligibility.catalog_sha256,
                    },
                )
                catalog = B_catalog
                core_edges: tuple[tuple[str, str], ...] = ()
                long_edges: tuple[tuple[str, str], ...] = ()
                confirmed: tuple[tuple[str, str], ...] = ()
                exploratory: tuple[tuple[str, str], ...] = ()
            else:
                core_edges = tuple(
                    tuple(edge)
                    for edge in primary.analysis["confirmation_core_edges"]
                )
                long_edges = tuple(
                    tuple(edge)
                    for edge in diagnostics[
                        "hard_long_576_640_768_veto"
                    ].analysis["point_gated_edges"]
                )
                confirmed, exploratory = worker.freeze_C_confirmed_edge_intersection(
                    frozen_B_candidates=B_candidates,
                    C_confirmation_core=core_edges,
                    C_hard_long_veto_pass=long_edges,
                )
                catalog = worker._digest(
                    worker._GRAPH_DOMAIN,
                    {
                        "C_confirmation_core_edges": [
                            list(edge) for edge in core_edges
                        ],
                        "C_exploratory_edges_not_in_B": [
                            list(edge) for edge in exploratory
                        ],
                        "C_hard_long_veto_pass_edges": [
                            list(edge) for edge in long_edges
                        ],
                        "C_primary_input_sha256": primary.input_sha256,
                        "C_primary_receipt_sha256": primary.receipt_sha256,
                        "confirmed_edges": [list(edge) for edge in confirmed],
                        "frozen_B_candidate_catalog_sha256": B_catalog,
                        "frozen_B_candidate_edges": [
                            list(edge) for edge in B_candidates
                        ],
                        "hard_long_diagnostic_receipt_sha256": diagnostics[
                            "hard_long_576_640_768_veto"
                        ].receipt_sha256,
                    },
                )
            artifacts = {
                f"{role}_capture_receipt": captures[role].receipt_sha256,
                f"{role}_evaluation_catalog": evaluation,
                f"{role}_primary_graph_input": primary.input_sha256,
                f"{role}_primary_graph_receipt": primary.receipt_sha256,
                f"{role}_candidate_or_graph_catalog": catalog,
                **{
                    f"{role}_diagnostic_{subset_id}": receipt.receipt_sha256
                    for subset_id, receipt in diagnostics.items()
                },
            }
            stage = worker.build_frozen_stage_receipt(
                role=role,
                preflight_receipt_sha256=stage_preflight,
                role_panel_binding_sha256_value=captures[
                    role
                ].role_panel_binding_sha256,
                ordered_objective_row_catalog_sha256=captures[
                    role
                ].ordered_objective_row_catalog_sha256,
                score_objective_receipt_sha256=captures[
                    role
                ].score_objective_receipt_sha256,
                prior_stage_receipt_sha256=prior,
                artifact_sha256s=artifacts,
                primary_eligible_example_catalog_sha256=eligibility.catalog_sha256,
            )
            prior = stage.stage_receipt_sha256
            role_parts[role] = {
                "all_eight_diagnostic_source": source_payloads[role],
                "candidate_or_graph_catalog_sha256": catalog,
                "capture_receipt": _payload_with_digest(
                    captures[role], "receipt_sha256"
                ),
                "confirmation_core_edges": [list(edge) for edge in core_edges],
                "confirmed_edges": [list(edge) for edge in confirmed],
                "diagnostic_receipts": {
                    subset_id: receipt.canonical_payload()
                    for subset_id, receipt in diagnostics.items()
                },
                "evaluation_catalog_sha256": evaluation,
                "evaluation_resource_observations": [
                    _resource_receipt(
                        f"{role}_applied_chart_stores_frozen",
                        ordinal=(
                            7 if role == worker.DISCOVERY_ROLE else 11
                        ),
                    ),
                    _resource_receipt(
                        f"{role}_all_directional_evaluations_frozen",
                        completed=38 * 37,
                        total=38 * 37,
                        ordinal=(
                            8 if role == worker.DISCOVERY_ROLE else 12
                        ),
                    ),
                ],
                "exploratory_C_edges": [list(edge) for edge in exploratory],
                "frozen_candidate_edges": [
                    list(edge) for edge in B_candidates
                ],
                "hard_long_veto_edges": [list(edge) for edge in long_edges],
                "primary_receipt": {
                    "analysis": primary.analysis,
                    "input_sha256": primary.input_sha256,
                    "receipt_sha256": primary.receipt_sha256,
                },
                "role": role,
                "stage_receipt": _payload_with_digest(
                    stage, "stage_receipt_sha256"
                ),
            }

        C_part = role_parts[worker.CONFIRMATION_ROLE]
        if D_measured:
            measurement = dict(
                D_measurement
                or {
                    "D_chart_capture_count": 0,
                    "D_may_change_graph": False,
                    "D_semantically_opened": True,
                }
            )
            response_sha = worker._digest(worker._D_RESPONSE_DOMAIN, measurement)
            D_manifest = self.manifest["roles"][worker.RESPONSE_ROLE]
            D_stage = worker.build_frozen_stage_receipt(
                role=worker.RESPONSE_ROLE,
                preflight_receipt_sha256=stage_preflight,
                role_panel_binding_sha256_value=panel_bindings[
                    worker.RESPONSE_ROLE
                ],
                ordered_objective_row_catalog_sha256=D_manifest[
                    "ordered_objective_row_catalog_sha256"
                ],
                score_objective_receipt_sha256=D_manifest[
                    "score_objective_receipt_sha256"
                ],
                prior_stage_receipt_sha256=prior,
                artifact_sha256s={
                    "D_response_validation": response_sha,
                    "frozen_C_graph": C_part[
                        "candidate_or_graph_catalog_sha256"
                    ],
                },
                primary_eligible_example_catalog_sha256=None,
            )
            D_part = {
                "measurement": measurement,
                "stage_receipt": _payload_with_digest(
                    D_stage, "stage_receipt_sha256"
                ),
            }
        else:
            D_part = {
                "measurement": {
                    "D_chart_capture_count": 0,
                    "D_may_change_graph": False,
                    "D_semantically_opened": False,
                    "skip_receipt_sha256": worker._digest(
                        worker._D_SKIP_DOMAIN,
                        {
                            "C_graph_catalog_sha256": C_part[
                                "candidate_or_graph_catalog_sha256"
                            ],
                            "reason": "response_validation_not_requested",
                        },
                    ),
                },
                "stage_receipt": None,
            }
        ledger = worker.build_capture_call_ledger()
        A_resources = [
            _resource_receipt("A_raw_capture_sealed", ordinal=1),
            _resource_receipt("A_shared_atlas_frozen", ordinal=2),
            _resource_receipt("A_applied_chart_stores_frozen", ordinal=3),
            _resource_receipt(
                "A_all_directional_transports_frozen",
                completed=38 * 37,
                total=38 * 37,
                ordinal=4,
            ),
        ]
        payload = {
            "A_fit": {
                "atlas_artifact_file_sha256": _sha("atlas-file"),
                "atlas_sha256": atlas_sha,
                "capture_receipt": _payload_with_digest(
                    A_capture, "receipt_sha256"
                ),
                "mode_lineage_catalog_sha256": mode_lineage_catalog_sha,
                "published_atlas_file_name": (
                    worker.DEFAULT_ATLAS_OUTPUT.name
                ),
                "resource_observations": A_resources,
                "stage_receipt": _payload_with_digest(
                    A_stage, "stage_receipt_sha256"
                ),
                "streaming_kernel_receipt_sha256": kernel_sha,
                "transport_artifacts": transport_rows,
                "transport_catalog_sha256": A_transport_catalog,
            },
            "B_discovery": role_parts[worker.DISCOVERY_ROLE],
            "C_confirmation": C_part,
            "D_response_validation": D_part,
            "capture_call_ledger": {
                **ledger.payload_without_sha256(),
                "ledger_sha256": ledger.ledger_sha256,
                "observed_live_capture_count": 135,
            },
            "claims": {
                "D_response_validation_measured": D_measured,
                "chart_native_neighbor_measured": True,
                "compression_ready": False,
                "fresh_reserve_opened": False,
                "new_C_edges_authorize_neighbors": False,
                "response_measurements_changed_graph": False,
            },
            "confirmed_edges": C_part["confirmed_edges"],
            "neighbor_svg_sha256": hashlib.sha256(
                worker.render_neighbor_svg(
                    mode_ids=tuple(sorted(_MODE_IDS)),
                    confirmed_edges=C_part["confirmed_edges"],
                )
            ).hexdigest(),
            "final_resource_observation": _resource_receipt(
                "C_graph_frozen_before_publication",
                completed=38 * 37 * 3,
                total=38 * 37 * 3,
                ordinal=15,
            ),
            "phase_cleanup_resource_observations": [
                _resource_receipt(name, ordinal=ordinal)
                for name, ordinal in (
                    ("A_before_raw_and_applied_cleanup", 5),
                    ("A_after_raw_and_applied_cleanup", 6),
                    ("B_before_raw_and_applied_cleanup", 9),
                    ("B_after_raw_and_applied_cleanup", 10),
                    ("C_before_raw_and_applied_cleanup", 13),
                    ("C_after_raw_and_applied_cleanup", 14),
                )
            ],
            "mode_axis": sorted(_MODE_IDS),
            "preflight": preflight,
            "protocol": {
                "B_candidates_are_point_gated_only": True,
                "C_confirmation_is_frozen_B_intersect_C_core_intersect_hard_long_veto": True,
                "common_comparison_ABI": "carrier_chart_energy.h640.v1",
                "response_validation_is_non_authorizing": True,
            },
            "runtime_execution_receipt": {
                **runtime_receipt.payload_without_sha256(),
                "runtime_receipt_sha256": (
                    runtime_receipt.runtime_receipt_sha256
                ),
            },
            "runtime_qualification_receipt": {
                **runtime_qualification.payload_without_sha256(),
                "qualification_receipt_sha256": (
                    runtime_qualification.qualification_receipt_sha256
                ),
            },
        }
        return worker.seal_experiment_report(payload)


@pytest.fixture
def completed_report_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> _ModelFreeCompletedReportHarness:
    return _ModelFreeCompletedReportHarness(monkeypatch)


def test_production_cardinality_model_free_report_replays_through_public_loader(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    report = completed_report_harness.build_report()
    assert worker.strict_load_experiment_report_bytes(_canonical(report)) == report


def test_runtime_call_accounting_includes_dense_parity_in_overall_totals(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    report = completed_report_harness.build_report()
    runtime = report["runtime_execution_receipt"]
    ledger = report["capture_call_ledger"]
    parity = runtime["observed_dense_lexical_parity_smoke_count"]
    assert parity == 1
    assert runtime["observed_total_backbone_forward_count"] == (
        ledger["observed_live_capture_count"] + parity
    ) == 136
    assert runtime["observed_total_terminal_cotangent_VJP_count"] == (
        ledger["total_summed_shifted_nll_backward_count"] + parity
    ) == 136


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_total_backbone_forward_count", True),
        ("observed_total_backbone_forward_count", 135),
        ("observed_captured_example_count", 145),
        ("source_model_mutated", 0),
        ("contains_token_ids", 0),
        ("contains_raw_activations", True),
        (
            "observed_role_batch_counts",
            [
                [worker.FIT_ROLE, True],
                [worker.DISCOVERY_ROLE, 45],
                [worker.CONFIRMATION_ROLE, 45],
            ],
        ),
    ],
    ids=(
        "bool-forward-count",
        "wrong-forward-count",
        "wrong-example-count",
        "int-source-mutation-flag",
        "int-token-content-flag",
        "raw-activation-content-claim",
        "bool-role-batch-count",
    ),
)
def test_runtime_counts_and_content_flags_reject_bool_or_rehashed_drift(
    completed_report_harness: _ModelFreeCompletedReportHarness,
    field: str,
    value: object,
) -> None:
    attacked = completed_report_harness.build_report()
    runtime = attacked["runtime_execution_receipt"]
    runtime[field] = value
    runtime.pop("runtime_receipt_sha256")
    runtime["runtime_receipt_sha256"] = worker._digest(
        worker._RUNTIME_RECEIPT_DOMAIN,
        runtime,
    )
    _rehash_report(attacked)
    with pytest.raises(
        (TypeError, ValueError),
        match="runtime|integer|inventory|role|count|bool",
    ):
        worker.strict_load_experiment_report_bytes(_canonical(attacked))


@pytest.mark.parametrize(
    "changes",
    [
        {"streaming_kernel_sha256": _sha("substituted-streaming-kernel")},
        {"B_transport_catalog": _sha("substituted-B-fit-catalog")},
        {"C_transport_catalog": _sha("substituted-C-fit-catalog")},
        {"B_transport_receipt_nonce": "substituted"},
        {"C_transport_receipt_nonce": "substituted"},
        {"B_panel_binding": _sha("substituted-B-panel")},
        {"C_panel_binding": _sha("substituted-C-panel")},
        {
            "A_objective_row_catalog_sha256": _sha(
                "substituted-A-objective-rows"
            )
        },
        {
            "B_objective_row_catalog_sha256": _sha(
                "substituted-B-objective-rows"
            )
        },
        {
            "C_score_objective_receipt_sha256": _sha(
                "substituted-C-score-objective"
            )
        },
        {
            "stage_preflight_receipt_sha256": _sha(
                "substituted-stage-preflight"
            )
        },
        {
            "execution_binding_sha256": _sha(
                "attacker-test-only-or-budget-binding"
            ),
            "execution_contract_attack": "schedule",
        },
        {
            "execution_binding_sha256": _sha(
                "attacker-storage-budget-binding"
            ),
            "execution_contract_attack": "storage-budget",
        },
        {
            "execution_binding_sha256": _sha("attacker-test-only-binding"),
            "execution_contract_attack": "test-only",
        },
        {
            "execution_binding_sha256": _sha("attacker-source-pin-binding"),
            "execution_contract_attack": "source-pin",
        },
        {
            "runtime_binding_sha256": _sha(
                "attacker-owned-runtime-session"
            )
        },
        {
            "compiled_execution_fingerprint_sha256": _sha(
                "attacker-compiled-execution"
            )
        },
        {
            "tied_readout_binding_receipt_sha256": _sha(
                "attacker-tied-readout"
            )
        },
        {
            "tied_lexical_lineage_sha256": _sha(
                "attacker-tied-lexical-lineage"
            )
        },
        {
            "compiler_receipt_sha256": _sha(
                "attacker-compiler-receipt"
            )
        },
        {"final_logit_softcap": 30.0},
        {
            "source_model_fingerprint_sha256": _sha(
                "attacker-source-model"
            )
        },
        {
            "source_execution_fingerprint_sha256": _sha(
                "attacker-source-execution"
            )
        },
        {
            "source_pin_overrides": {
                "chart_native_neighbor_worker": _sha(
                    "attacker-self-pinned-worker"
                )
            }
        },
        {
            "A_example_ids_override": tuple(
                f"forged-A-example-{index:02d}" for index in range(48)
            )
        },
        {"A_reverse_batch_axis": True},
        {
            "B_mode_state_nonce": "substituted",
            "C_mode_state_nonce": "substituted",
        },
        {
            "B_mode_site_nonce": "substituted",
            "C_mode_site_nonce": "substituted",
        },
        {
            "B_mode_adapter_nonce": "substituted",
            "C_mode_adapter_nonce": "substituted",
        },
        {
            "B_mode_native_abi_nonce": "substituted",
            "C_mode_native_abi_nonce": "substituted",
        },
        {
            "B_mode_comparison_abi_nonce": "substituted",
            "C_mode_comparison_abi_nonce": "substituted",
        },
        {
            "B_capture_route_nonce": "capture-only-substitution",
        },
        {
            "C_capture_route_nonce": "capture-only-substitution",
        },
    ],
    ids=(
        "streaming-kernel",
        "B-fit-catalog",
        "C-fit-catalog",
        "B-fit-receipts",
        "C-fit-receipts",
        "B-panel-binding",
        "C-panel-binding",
        "A-q-t-catalog",
        "B-q-t-catalog",
        "C-score-objective",
        "stage-preflight",
        "execution-contract-schedule",
        "execution-contract-storage-budget",
        "execution-contract-test-only",
        "execution-contract-source-pin",
        "runtime-session",
        "compiled-execution",
        "tied-readout",
        "tied-lexical-lineage",
        "compiler-receipt",
        "final-logit-softcap",
        "source-model-fingerprint",
        "source-execution-fingerprint",
        "worker-self-pin",
        "A-example-inventory",
        "A-batch-axis",
        "mode-state",
        "mode-site",
        "mode-static-adapter",
        "mode-native-ABI",
        "mode-comparison-ABI",
        "B-route-cross-binding",
        "C-route-cross-binding",
    ),
)
def test_coordinated_rehash_cannot_replace_frozen_runtime_or_measurement_lineage(
    completed_report_harness: _ModelFreeCompletedReportHarness,
    changes: dict[str, object],
) -> None:
    attacked = completed_report_harness.build_report(**changes)
    with pytest.raises(
        ValueError,
        match=(
            "runtime|source|kernel|transport|panel|objective|stage|mode|route|"
            "lineage|frozen|binding|capture|batch|inventory|execution|contract"
        ),
    ):
        worker.strict_load_experiment_report_bytes(_canonical(attacked))


@pytest.mark.parametrize(
    "D_measured,D_measurement",
    [
        (
            1,
            {
                "D_chart_capture_count": 0,
                "D_may_change_graph": False,
                "D_semantically_opened": True,
            },
        ),
        (
            True,
            {
                "D_chart_capture_count": 7,
                "D_may_change_graph": True,
                "D_semantically_opened": True,
                "fresh_reserve_opened": True,
            },
        ),
    ],
    ids=("bool-as-int", "arbitrary-authorizing-D-payload"),
)
def test_D_reporting_cannot_alias_boolean_or_claim_graph_authority(
    completed_report_harness: _ModelFreeCompletedReportHarness,
    D_measured: object,
    D_measurement: Mapping[str, object],
) -> None:
    attacked = completed_report_harness.build_report(
        D_measured=D_measured,
        D_measurement=D_measurement,
    )
    with pytest.raises(ValueError, match="D|response|claim|bool|graph|reserve"):
        worker.strict_load_experiment_report_bytes(_canonical(attacked))


def test_all8_primary_and_diagnostic_payloads_are_not_substitutable(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    baseline = completed_report_harness.build_report()
    attacks: list[dict[str, object]] = []

    swapped_source = deepcopy(baseline)
    swapped_source["B_discovery"]["all_eight_diagnostic_source"] = deepcopy(
        swapped_source["C_confirmation"]["all_eight_diagnostic_source"]
    )
    _rehash_report(swapped_source)
    attacks.append(swapped_source)

    changed_primary = deepcopy(baseline)
    changed_primary["B_discovery"]["primary_receipt"]["input_sha256"] = _sha(
        "substituted-primary-seven"
    )
    _rehash_report(changed_primary)
    attacks.append(changed_primary)

    swapped_subset = deepcopy(baseline)
    swapped_subset["C_confirmation"]["diagnostic_receipts"]["short_96"] = (
        deepcopy(
            swapped_subset["C_confirmation"]["diagnostic_receipts"][
                "boundary_transition_514_diagnostic_only"
            ]
        )
    )
    _rehash_report(swapped_subset)
    attacks.append(swapped_subset)

    for attacked in attacks:
        with pytest.raises(
            ValueError,
            match="role|source|cardinality|primary|diagnostic|replay",
        ):
            worker.strict_load_experiment_report_bytes(_canonical(attacked))


def test_514_or_new_C_edges_cannot_enter_authorizing_graph_catalogs(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    report = completed_report_harness.build_report()
    report["C_confirmation"]["confirmed_edges"].append(
        ["mode.mutation.2", "mode.mutation.3"]
    )
    report["confirmed_edges"] = deepcopy(
        report["C_confirmation"]["confirmed_edges"]
    )
    _rehash_report(report)
    with pytest.raises(ValueError, match="C confirmed graph|catalog|edge"):
        worker.strict_load_experiment_report_bytes(_canonical(report))


def test_hard_long_veto_cannot_be_erased_to_nominate_an_edge(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    baseline = completed_report_harness.build_report()

    erased_catalog = deepcopy(baseline)
    erased_catalog["C_confirmation"]["hard_long_veto_edges"] = []
    _rehash_report(erased_catalog)

    erased_receipt = deepcopy(baseline)
    hard_long = erased_receipt["C_confirmation"]["diagnostic_receipts"][
        "hard_long_576_640_768_veto"
    ]
    hard_long["analysis"]["point_gated_edges"] = []
    hard_long["receipt_sha256"] = _sha("attacker-erased-hard-long-veto")
    _rehash_report(erased_receipt)

    for attacked in (erased_catalog, erased_receipt):
        with pytest.raises(
            ValueError,
            match="hard|long|diagnostic|receipt|graph|replay|catalog",
        ):
            worker.strict_load_experiment_report_bytes(_canonical(attacked))


def test_reserve_semantics_and_claims_remain_fail_closed_after_rehash(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    baseline = completed_report_harness.build_report()
    attacks: list[dict[str, object]] = []

    opened_preflight = deepcopy(baseline)
    preflight = opened_preflight["preflight"]
    preflight["reserve_semantically_decoded"] = True
    preflight.pop("receipt_sha256")
    preflight["receipt_sha256"] = worker._digest(
        worker._PREFLIGHT_DOMAIN,
        preflight,
    )
    _rehash_report(opened_preflight)
    attacks.append(opened_preflight)

    substituted_reserve = deepcopy(baseline)
    preflight = substituted_reserve["preflight"]
    preflight["opaque_role_file_sha256s"][worker.RESERVE_ROLE] = _sha(
        "attacker-opened-reserve-file"
    )
    preflight.pop("receipt_sha256")
    preflight["receipt_sha256"] = worker._digest(
        worker._PREFLIGHT_DOMAIN,
        preflight,
    )
    _rehash_report(substituted_reserve)
    attacks.append(substituted_reserve)

    opened_claim = deepcopy(baseline)
    opened_claim["claims"]["fresh_reserve_opened"] = True
    _rehash_report(opened_claim)
    attacks.append(opened_claim)

    for attacked in attacks:
        with pytest.raises(
            ValueError,
            match="reserve|opaque|preflight|claim|frozen|lineage",
        ):
            worker.strict_load_experiment_report_bytes(_canonical(attacked))


def test_resource_lifecycle_receipts_cannot_be_relabelled_and_rehashed(
    completed_report_harness: _ModelFreeCompletedReportHarness,
) -> None:
    report = completed_report_harness.build_report()
    first = report["phase_cleanup_resource_observations"][0]
    first["phase_id"] = "C_graph_frozen_before_publication"
    _rehash_resource(first)
    _rehash_report(report)
    with pytest.raises(ValueError, match="resource|phase|lifecycle|cleanup"):
        worker.strict_load_experiment_report_bytes(_canonical(report))


def test_production_capture_releases_each_batch_instead_of_retaining_7gib_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, _append_sizes, _keys = _exercise_production_capture_without_model(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        reject_oversized_append=False,
    )
    gc.collect()
    assert captured.receipt.row_count == 24_732
    assert _TinyCapturedTensor.maximum_live <= 2
    assert _TinyCapturedTensor.live == 0


def test_full_length_capture_appends_fixed_256_row_slices_in_canonical_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, append_sizes, first_mode_keys = (
        _exercise_production_capture_without_model(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            reject_oversized_append=True,
        )
    )
    assert captured.receipt.row_count == 24_732
    assert append_sizes
    assert max(append_sizes) <= 256
    observed = tuple(
        (
            row.family_id,
            row.example_id,
            row.prediction_query_position,
            row.target_label_position,
        )
        for row in first_mode_keys
    )
    manifest = load_manifest()
    assert observed == worker._canonical_role_row_keys(
        manifest["roles"][worker.DISCOVERY_ROLE]
    )


def test_strict_report_loader_rejects_noncanonical_duplicate_and_nan_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test isolates the byte-level loader. Completed-report semantics are
    # exercised separately against a production-shaped model-free receipt.
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    report = worker.seal_experiment_report({"confirmed_edges": []})
    canonical = worker._canonical_json_bytes(report)
    assert worker.strict_load_experiment_report_bytes(canonical) == report

    with pytest.raises(ValueError, match="canonical"):
        worker.strict_load_experiment_report_bytes(
            json.dumps(report, sort_keys=False).encode("utf-8")
        )

    duplicate = canonical[:-1] + b',"schema":"duplicate"}'
    with pytest.raises(ValueError, match="strict JSON"):
        worker.strict_load_experiment_report_bytes(duplicate)

    with pytest.raises(ValueError, match="strict JSON"):
        worker.strict_load_experiment_report_bytes(
            canonical.replace(b'"confirmed_edges":[]', b'"metric":NaN')
        )


def test_publication_rolls_back_all_three_outputs_when_strict_reread_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    atlas_raw = b"strict-fake-shared-chart-atlas"
    atlas_sha = _sha("strict-fake-shared-chart-atlas")
    report = _partial_publication_report(
        svg=svg,
        atlas_raw=atlas_raw,
        atlas_sha256=atlas_sha,
    )
    _patch_fake_atlas_loader(monkeypatch, atlas_sha256=atlas_sha)
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas_source.write_bytes(atlas_raw)
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name
    real_loader = worker.strict_load_experiment_report_bytes
    calls = 0

    def fail_second(raw: bytes) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected strict report reread failure")
        return real_loader(raw)

    monkeypatch.setattr(worker, "strict_load_experiment_report_bytes", fail_second)
    with pytest.raises(ValueError, match="injected strict report reread"):
        worker.publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
    assert calls == 2
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert atlas_source.read_bytes() == atlas_raw
    assert set(tmp_path.iterdir()) == {atlas_source}


@pytest.mark.parametrize("failed_link", [2, 3], ids=("atlas", "report"))
def test_triple_publication_rolls_back_when_second_or_commit_link_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_link: int,
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    atlas_raw = b"strict-fake-shared-chart-atlas"
    atlas_sha = _sha("strict-fake-shared-chart-atlas")
    report = _partial_publication_report(
        svg=svg,
        atlas_raw=atlas_raw,
        atlas_sha256=atlas_sha,
    )
    _patch_fake_atlas_loader(monkeypatch, atlas_sha256=atlas_sha)
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas_source.write_bytes(atlas_raw)
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name
    real_link = worker.os.link
    link_count = 0

    def fail_selected_link(source: object, destination: object) -> None:
        nonlocal link_count
        link_count += 1
        if link_count == failed_link:
            raise OSError(f"injected publication link {failed_link} failure")
        real_link(source, destination)

    monkeypatch.setattr(worker.os, "link", fail_selected_link)
    with pytest.raises(OSError, match="injected publication link"):
        worker.publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
    assert link_count == failed_link
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert atlas_source.read_bytes() == atlas_raw
    assert set(tmp_path.iterdir()) == {atlas_source}


def test_triple_publication_durability_barriers_precede_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    atlas_raw = b"strict-fake-shared-chart-atlas"
    atlas_sha = _sha("strict-fake-shared-chart-atlas")
    report = _partial_publication_report(
        svg=svg,
        atlas_raw=atlas_raw,
        atlas_sha256=atlas_sha,
    )
    _patch_fake_atlas_loader(monkeypatch, atlas_sha256=atlas_sha)
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas_source.write_bytes(atlas_raw)
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name
    monkeypatch.setattr(
        worker,
        "strict_load_experiment_report",
        lambda *_args, **_kwargs: report,
    )
    real_open = worker.os.open
    real_fsync = worker.os.fsync
    real_link = worker.os.link
    directory_descriptors: dict[int, Path] = {}
    events: list[str] = []

    def tracked_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        candidate = Path(path)
        if candidate.is_dir():
            directory_descriptors[descriptor] = candidate
        return descriptor

    def tracked_fsync(descriptor: int) -> None:
        directory = directory_descriptors.get(descriptor)
        if directory is not None:
            events.append(f"fsync-directory:{directory}")
        real_fsync(descriptor)

    def tracked_link(source: object, destination: object) -> None:
        events.append(f"link:{Path(destination).name}")
        real_link(source, destination)

    monkeypatch.setattr(worker.os, "open", tracked_open)
    monkeypatch.setattr(worker.os, "fsync", tracked_fsync)
    monkeypatch.setattr(worker.os, "link", tracked_link)
    worker.publish_experiment_outputs(
        report=report,
        svg_bytes=svg,
        atlas_source_path=atlas_source,
        report_path=report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    )
    assert events == [
        f"link:{svg_path.name}",
        f"link:{atlas_path.name}",
        f"fsync-directory:{tmp_path}",
        f"link:{report_path.name}",
        f"fsync-directory:{tmp_path}",
    ]


@pytest.mark.parametrize(
    "failed_directory_barrier,expected_link_count",
    [(1, 2), (2, 3)],
    ids=("sidecar-barrier", "commit-marker-barrier"),
)
def test_triple_publication_rolls_back_on_directory_barrier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_directory_barrier: int,
    expected_link_count: int,
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    atlas_raw = b"strict-fake-shared-chart-atlas"
    atlas_sha = _sha("strict-fake-shared-chart-atlas")
    report = _partial_publication_report(
        svg=svg,
        atlas_raw=atlas_raw,
        atlas_sha256=atlas_sha,
    )
    _patch_fake_atlas_loader(monkeypatch, atlas_sha256=atlas_sha)
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas_source.write_bytes(atlas_raw)
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name
    monkeypatch.setattr(
        worker,
        "strict_load_experiment_report",
        lambda *_args, **_kwargs: report,
    )
    real_open = worker.os.open
    real_fsync = worker.os.fsync
    real_link = worker.os.link
    directory_descriptors: set[int] = set()
    directory_barriers = 0
    link_count = 0

    def tracked_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path).is_dir():
            directory_descriptors.add(descriptor)
        return descriptor

    def fail_selected_fsync(descriptor: int) -> None:
        nonlocal directory_barriers
        if descriptor in directory_descriptors:
            directory_barriers += 1
            if directory_barriers == failed_directory_barrier:
                raise OSError(
                    f"injected directory barrier {failed_directory_barrier} failure"
                )
        real_fsync(descriptor)

    def tracked_link(source: object, destination: object) -> None:
        nonlocal link_count
        link_count += 1
        real_link(source, destination)

    monkeypatch.setattr(worker.os, "open", tracked_open)
    monkeypatch.setattr(worker.os, "fsync", fail_selected_fsync)
    monkeypatch.setattr(worker.os, "link", tracked_link)
    with pytest.raises(OSError, match="injected directory barrier"):
        worker.publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
    assert link_count == expected_link_count
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert set(tmp_path.iterdir()) == {atlas_source}


def test_durable_rollback_unlinks_and_fsyncs_commit_marker_before_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_directory = tmp_path / "report-parent"
    svg_directory = tmp_path / "svg-parent"
    atlas_directory = tmp_path / "atlas-parent"
    for directory in (report_directory, svg_directory, atlas_directory):
        directory.mkdir()
    report_path = report_directory / "report.json"
    svg_path = svg_directory / "graph.svg"
    atlas_path = atlas_directory / worker.DEFAULT_ATLAS_OUTPUT.name
    for path in (report_path, svg_path, atlas_path):
        path.write_bytes(b"published")
    publication = worker.PublishedExperimentReceipt(
        atlas_path=str(atlas_path),
        report_path=str(report_path),
        svg_path=str(svg_path),
        atlas_file_sha256=_sha("atlas-file"),
        atlas_sha256=_sha("atlas-semantic"),
        report_file_sha256=_sha("report-file"),
        svg_file_sha256=_sha("svg-file"),
        report_sha256=_sha("report-semantic"),
        publication_receipt_sha256=_sha("publication"),
    )
    real_open = worker.os.open
    real_fsync = worker.os.fsync
    real_unlink = worker.os.unlink
    directory_descriptors: dict[int, Path] = {}
    events: list[str] = []

    def tracked_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        candidate = Path(path)
        if candidate.is_dir():
            directory_descriptors[descriptor] = candidate
        return descriptor

    def tracked_fsync(descriptor: int) -> None:
        directory = directory_descriptors.get(descriptor)
        if directory is not None:
            events.append(f"fsync-directory:{directory.name}")
        real_fsync(descriptor)

    def tracked_unlink(path: object) -> None:
        events.append(f"unlink:{Path(path).name}")
        real_unlink(path)

    monkeypatch.setattr(worker.os, "open", tracked_open)
    monkeypatch.setattr(worker.os, "fsync", tracked_fsync)
    monkeypatch.setattr(worker.os, "unlink", tracked_unlink)
    worker._rollback_published_experiment(publication)
    assert events == [
        f"unlink:{report_path.name}",
        f"fsync-directory:{report_directory.name}",
        f"unlink:{svg_path.name}",
        f"unlink:{atlas_path.name}",
        f"fsync-directory:{atlas_directory.name}",
        f"fsync-directory:{svg_directory.name}",
    ]
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()


@pytest.mark.parametrize(
    "wrong_file_digest,wrong_semantic_digest",
    [(True, False), (False, True)],
    ids=("file-digest", "semantic-digest"),
)
def test_triple_publication_rejects_wrong_atlas_without_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_file_digest: bool,
    wrong_semantic_digest: bool,
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    atlas_raw = b"strict-fake-shared-chart-atlas"
    atlas_sha = _sha("strict-fake-shared-chart-atlas")
    report = _partial_publication_report(
        svg=svg,
        atlas_raw=(b"different-atlas-bytes" if wrong_file_digest else atlas_raw),
        atlas_sha256=atlas_sha,
    )
    _patch_fake_atlas_loader(
        monkeypatch,
        atlas_sha256=(
            _sha("wrong-atlas-semantics")
            if wrong_semantic_digest
            else atlas_sha
        ),
    )
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas_source.write_bytes(atlas_raw)
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name

    with pytest.raises(ValueError, match="atlas|digest|semantic"):
        worker.publish_experiment_outputs(
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
    assert set(tmp_path.iterdir()) == {atlas_source}


def test_triple_publication_strict_atlas_roundtrip_and_tamper_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    monkeypatch.setattr(
        worker,
        "_strict_rehash_frozen_repository_inputs",
        lambda **_kwargs: None,
    )
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas = _write_tiny_strict_atlas_bundle(atlas_source)
    atlas_raw = atlas_source.read_bytes()
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    report = _partial_publication_report(
        svg=svg,
        atlas_raw=atlas_raw,
        atlas_sha256=atlas.atlas_sha256,
    )
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name

    receipt = worker.publish_experiment_outputs(
        report=report,
        svg_bytes=svg,
        atlas_source_path=atlas_source,
        report_path=report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    )
    assert receipt.atlas_file_sha256 == hashlib.sha256(atlas_raw).hexdigest()
    assert receipt.atlas_sha256 == atlas.atlas_sha256
    assert worker.strict_load_experiment_report(
        report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    ) == report

    tampered = bytearray(atlas_path.read_bytes())
    tampered[-1] ^= 1
    atlas_path.write_bytes(tampered)
    with pytest.raises(ValueError, match="atlas|digest"):
        worker.strict_load_experiment_report(
            report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )


def test_completed_report_schema_cannot_downgrade_to_digest_only_envelope() -> None:
    downgraded = worker.seal_experiment_report(
        {
            "confirmed_edges": [],
            "mode_axis": ["mode.lexical.ingress"],
        }
    )
    with pytest.raises(ValueError, match="completed report"):
        worker.strict_load_experiment_report_bytes(_canonical(downgraded))


def test_publication_rejects_unrelated_safe_svg_and_leaves_no_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep the test focused on the report-to-visualization binding while the
    # expensive scientific receipt replay is covered by the fixture below.
    monkeypatch.setattr(
        worker,
        "_validate_completed_experiment_report",
        lambda _report: None,
    )
    correct = worker.render_neighbor_svg(
        mode_ids=("mode.a",),
        confirmed_edges=(),
    )
    atlas_raw = b"strict-fake-shared-chart-atlas"
    atlas_sha = _sha("strict-fake-shared-chart-atlas")
    report = _partial_publication_report(
        svg=correct,
        atlas_raw=atlas_raw,
        atlas_sha256=atlas_sha,
    )
    atlas_source = tmp_path / "source-atlas.bundle"
    atlas_source.write_bytes(atlas_raw)
    unrelated = worker.render_neighbor_svg(
        mode_ids=("mode.unrelated",),
        confirmed_edges=(),
    )
    report_path = tmp_path / "report.json"
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name

    with pytest.raises(ValueError, match="SVG|visualization|render"):
        worker.publish_experiment_outputs(
            report=report,
            svg_bytes=unrelated,
            atlas_source_path=atlas_source,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
    assert not report_path.exists()
    assert not svg_path.exists()
    assert not atlas_path.exists()
    assert atlas_source.read_bytes() == atlas_raw
    assert set(tmp_path.iterdir()) == {atlas_source}


def test_publication_rejects_symlink_destination_without_touching_peer(
    tmp_path: Path,
) -> None:
    report = worker.seal_experiment_report({"confirmed_edges": []})
    svg = worker.render_neighbor_svg(mode_ids=("mode.a",), confirmed_edges=())
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"do-not-clobber")
    report_path = tmp_path / "report.json"
    report_path.symlink_to(existing)
    svg_path = tmp_path / "graph.svg"
    atlas_path = tmp_path / worker.DEFAULT_ATLAS_OUTPUT.name

    with pytest.raises(FileExistsError):
        worker.publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=tmp_path / "unread-atlas-source.bundle",
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
    assert existing.read_bytes() == b"do-not-clobber"
    assert report_path.is_symlink()
    assert not svg_path.exists()
    assert not atlas_path.exists()


def test_public_strict_loader_rehashes_every_setup_listed_live_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fisher_graph import gemma3_chart_native_neighbor_setup as setup_module

    def write_frozen(relative: str, raw: bytes) -> dict[str, str]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "path": relative,
        }

    setup_receipt = write_frozen(
        "artifacts/research/fake-chart-native-setup.json",
        b'{"authenticated":"setup"}',
    )
    implementations = {
        "chart_native_neighbor_worker": write_frozen(
            "src/fisher_graph/fake_worker.py",
            b"worker-v1\n",
        ),
        "chart_native_neighbor_runtime": write_frozen(
            "src/fisher_graph/fake_runtime.py",
            b"runtime-v1\n",
        ),
        "streaming_shared_chart_energy_transport": write_frozen(
            "src/fisher_graph/fake_streaming.py",
            b"streaming-v1\n",
        ),
    }
    manifest = write_frozen("examples/fake_manifest.json", b"manifest-v1\n")
    receipt = write_frozen("examples/fake_receipt.json", b"receipt-v1\n")
    role = write_frozen("examples/fake_A.json", b"role-A-v1\n")
    setup_sha256 = _sha("fake-authenticated-setup")
    setup = {
        "frozen_fresh_panel_lineage": {
            "manifest": manifest,
            "receipt": receipt,
            "role_files": {"A": role},
        },
        "frozen_worker_implementation_sources": implementations,
        "setup_sha256": setup_sha256,
    }
    report = {
        "preflight": {
            "setup_sha256": setup_sha256,
            "source_file_sha256s": {
                name: source["file_sha256"]
                for name, source in implementations.items()
            },
        }
    }
    monkeypatch.setattr(
        worker,
        "DEFAULT_SETUP_PATH",
        Path(setup_receipt["path"]),
    )
    monkeypatch.setattr(
        worker,
        "PINNED_SETUP_FILE_SHA256",
        setup_receipt["file_sha256"],
    )
    monkeypatch.setattr(setup_module, "check_setup", lambda _path: setup)
    monkeypatch.setattr(
        worker,
        "_load_frozen_runtime_qualification_from_setup",
        lambda **_kwargs: object(),
    )

    # Establish that the synthetic catalog and all of its live bytes agree.
    worker._strict_rehash_frozen_repository_inputs(
        report=report,
        repository_root=tmp_path,
    )

    # Alter a non-worker implementation without changing either authenticated
    # catalog.  Exercise the public path (the semantic JSON parse is isolated)
    # so merely checking catalog constants cannot make this report load.
    runtime_path = tmp_path / implementations[
        "chart_native_neighbor_runtime"
    ]["path"]
    runtime_path.write_bytes(b"runtime-v2-attacker\n")
    report_path = tmp_path / "completed-report.json"
    report_path.write_bytes(b"{}")
    monkeypatch.setattr(
        worker,
        "strict_load_experiment_report_bytes",
        lambda _raw: report,
    )
    monkeypatch.setattr(
        worker,
        "__file__",
        str(tmp_path / "src/fisher_graph/fake_worker_module.py"),
    )
    with pytest.raises(ValueError, match="runtime|implementation|source|drift"):
        worker.strict_load_experiment_report(report_path)


@pytest.mark.parametrize("attack", ["missing", "symlink", "byte-tamper"])
def test_frozen_runtime_qualification_artifact_is_regular_present_and_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attack: str,
) -> None:
    real_loader = worker._load_frozen_runtime_qualification_from_setup
    harness = _ModelFreeCompletedReportHarness(monkeypatch)
    receipt = harness.frozen_runtime_qualification
    raw = _canonical(
        {
            **receipt.payload_without_sha256(),
            "qualification_receipt_sha256": (
                receipt.qualification_receipt_sha256
            ),
        }
    )
    qualification_path = tmp_path / "runtime-qualification.json"
    qualification_path.write_bytes(raw)
    setup = {
        "frozen_runtime_qualification": {
            "authenticated_present": True,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "path": qualification_path.name,
            "qualification_receipt_sha256": (
                receipt.qualification_receipt_sha256
            ),
        }
    }
    assert real_loader(setup=setup, repository_root=tmp_path) == receipt

    if attack == "missing":
        qualification_path.unlink()
    elif attack == "symlink":
        target = tmp_path / "attacker-qualification-target.json"
        target.write_bytes(raw)
        qualification_path.unlink()
        qualification_path.symlink_to(target.name)
    else:
        qualification_path.write_bytes(raw + b"\n")

    with pytest.raises(ValueError, match="qualification|regular|bytes|drift"):
        real_loader(setup=setup, repository_root=tmp_path)


@pytest.mark.parametrize(
    "bad_route",
    [
        ("mode.lexical.ingress", "example-00"),
        ("mode.mutation.1", "not-a-panel-example"),
    ],
)
def test_capture_receipt_rejects_replayed_or_out_of_inventory_route_cells(
    bad_route: tuple[str, str],
) -> None:
    valid = _capture_receipt(worker.DISCOVERY_ROLE)
    routes = list(valid.supplemental_route_evidence)
    routes[-1] = (
        bad_route[0],
        bad_route[1],
        _sha(f"bad-route-catalog:{bad_route}"),
        _sha(f"bad-route-receipt:{bad_route}"),
    )
    # Keep all byte counts and the outer receipt digest internally valid. The
    # route cell must still fail semantic inventory validation.
    with pytest.raises(ValueError, match="route|supplemental|inventory"):
        _capture_receipt(
            worker.DISCOVERY_ROLE,
            route_evidence=tuple(routes),
        )
