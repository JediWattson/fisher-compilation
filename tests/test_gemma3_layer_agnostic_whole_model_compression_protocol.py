from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fisher_graph.gemma3_layer_agnostic_whole_model_compression_protocol import (
    DEFAULT_PROTOCOL_OUTPUT,
    LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_FORMAT_VERSION,
    LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_SCHEMA,
    PINNED_FINAL_RMSNORM_PARAMETER_COUNT,
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
    PINNED_TIED_LEXICAL_PARAMETER_COUNT,
    PINNED_TRANSFORMER_BODY_PARAMETER_COUNT,
    LayerAgnosticWholeModelCompressionProtocol,
    build_layer_agnostic_whole_model_compression_protocol,
    load_layer_agnostic_whole_model_compression_protocol,
    main,
    validate_layer_agnostic_whole_model_compression_protocol,
    write_layer_agnostic_whole_model_compression_protocol,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / (
    "src/fisher_graph/"
    "gemma3_layer_agnostic_whole_model_compression_protocol.py"
)


def test_protocol_is_setup_only_and_covers_the_whole_model_boundary() -> None:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    assert (
        protocol["schema"]
        == LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_SCHEMA
    )
    assert (
        protocol["format_version"]
        == LAYER_AGNOSTIC_WHOLE_MODEL_COMPRESSION_PROTOCOL_FORMAT_VERSION
    )
    assert protocol["artifact_role"] == "preregistered_setup_only"
    assert protocol["setup_receipt"] == {
        "compression_success_claimed": False,
        "experiment_executed": False,
        "fitting_performed": False,
        "metrics_present": False,
        "model_weights_loaded": False,
        "prompt_data_opened": False,
        "tokenization_performed": False,
    }
    graph = protocol["whole_model_boundary_graph"]
    assert graph["lexical_ingress"]["mutation_index"] == 0
    assert len(graph["mutation_invocations"]) == 36
    attention = graph["attention_runtime_contracts"]
    assert attention["shared"]["query_heads"] == 4
    assert attention["shared"]["key_value_heads"] == 1
    assert attention["shared"]["head_dimension"] == 256
    assert attention["sliding_512"] == {
        "rope_theta": 10_000.0,
        "window": 512,
    }
    assert attention["global"] == {
        "rope_theta": 1_000_000.0,
        "window": None,
    }
    assert attention["global_source_layer_ordinals"] == [5, 11, 17]
    assert graph["terminal_rmsnorm"]["mutation_index"] == 37
    assert graph["lexical_egress"]["mutation_index"] == 38
    assert graph["candidate_requires_carrier_native_chart_mutations"] is True
    assert json.loads(json.dumps(protocol)) == protocol


def test_source_parameter_inventory_closes_and_tied_storage_counts_once() -> None:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    inventory = protocol["source_parameter_inventory"]
    assert inventory["inventory_closes_exactly"] is True
    assert inventory["whole_model_unique_parameters"] == (
        PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    )
    assert inventory["tied_embedding_and_head_counted_once"] == (
        PINNED_TIED_LEXICAL_PARAMETER_COUNT
    )
    assert inventory["transformer_body"] == (
        PINNED_TRANSFORMER_BODY_PARAMETER_COUNT
    )
    assert inventory["final_rmsnorm"] == (
        PINNED_FINAL_RMSNORM_PARAMETER_COUNT
    )
    lexical = protocol["shared_lexical_boundary"]
    assert lexical["physical_storage_multiplicity"] == 1
    assert lexical["physical_alias_audit_required"] is True
    assert lexical["equal_but_distinct_copies_count_as_tied"] is False
    assert lexical["factor_storage_layout"] == (
        "single_packed_C_then_D_backing_span"
    )
    assert lexical["physical_span_receipt_covers_C_and_D"] is True
    assert lexical["separate_C_and_D_storage_allowed_in_v1"] is False
    assert lexical["terminal_rmsnorm_preserved_exactly"] is True


def test_layer_labels_are_provenance_only_and_forbidden_as_features() -> None:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    ontology = protocol["runtime_ontology"]
    assert ontology["native_layers_are_execution_objects"] is False
    assert ontology["native_layer_ids_retained_as_provenance_only"] is True
    assert ontology["generator_family_is_parameter_identity"] is True
    assert ontology["generator_invocation_is_temporal_identity"] is True
    assert ontology["same_family_invocations_are_not_self_edges"] is True
    discovery = protocol["layer_agnostic_discovery_policy"]
    assert discovery["candidate_pair_scope"] == (
        "any_shape_compatible_source_modes"
    )
    forbidden = set(discovery["forbidden_clustering_and_routing_features"])
    assert {
        "layer_id",
        "layer_ordinal",
        "layer_distance",
        "same_layer",
        "adjacent_layer",
    }.issubset(forbidden)
    for occurrence in protocol["whole_model_boundary_graph"][
        "mutation_invocations"
    ]:
        assert "source_layer_ordinal" not in {
            key for key in occurrence if key != "provenance_only"
        }
        assert set(occurrence["provenance_only"]) == {
            "source_layer_ordinal",
            "source_mutation_id",
            "source_stage",
        }


def test_protocol_preregisters_compression_and_control_ladders() -> None:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    ladders = protocol["frozen_ladders"]
    assert ladders["residual_rank"] == [640, 576, 512, 448, 384]
    assert ladders["lexical_rank"] == [640, 512, 384, 256]
    assert ladders["atlas_region_count"] == [1, 4, 8, 16]
    assert ladders["relative_ridge"] == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    assert ladders["chart_temperature"] == [0.5, 1.0, 2.0]
    assert ladders["full_cartesian_product_exhausted"] is False
    points = ladders["ordered_resource_points"]
    assert [point["resource_point_id"] for point in points] == [
        "conservative",
        "balanced",
        "compact",
        "aggressive",
    ]
    assert all(
        point["active_charts_per_token"] <= point["atlas_regions"]
        for point in points
    )
    assert ladders["resource_shortfall_stops_lower_ranks"] is False
    numerics = protocol["frozen_numerics"]
    assert numerics["deterministic_seed"] == 1729
    assert numerics["ridge_solve_dtype"] == "float64"
    assert numerics["block_coordinate_sweeps"] == 8
    arm_ids = {arm["arm_id"] for arm in protocol["arms"]}
    assert {
        "native",
        "exact_whole_model_graph",
        "full_rank_layer_agnostic_identity",
        "global_residual_basis_control",
        "boundary_local_basis_control",
        "budget_matched_modular_factorization",
        "unrestricted_layer_agnostic_primary",
        "matched_naive_svd",
        "matched_fisher_deletion",
        "primary_zero_transport_ablation",
        "primary_family_shuffled_negative_control",
        "exact_lexical_compressed_body_ablation",
        "compressed_lexical_exact_body_ablation",
    } == arm_ids


def test_protocol_preregisters_firewall_causality_and_rate_curve_gates() -> None:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    firewall = protocol["fit_and_validation_firewall"]
    assert firewall["required_manifests"] == [
        "A_fit",
        "A_select",
        "B_guard",
        "C_assessment",
    ]
    assert firewall["row_prompt_and_family_overlap_allowed"] is False
    assert firewall["A_select_may_refit_coefficients_or_topology"] is False
    assert firewall["A_fit_inner_holdout_disjoint_from_coefficient_fit_rows"]
    assert firewall["A_fit_inner_holdout_may_touch_A_select_B_or_C"] is False
    assert firewall["pareto_frontier_frozen_before_B_guard"] is True
    assert firewall["existing_a5e_panel_role"] == "development_smoke_only"
    causal = protocol["causal_and_runtime_invariants"]
    assert causal["future_token_edges_forbidden"] is True
    assert causal["zero_live_source_calls_required"] is True
    assert causal["unsupported_chart_state_may_call_native_fallback"] is False
    compression = protocol["acceptance_gates"]["compression"]
    assert compression["whole_model_unique_parameter_reduction_minimum"] == 0.20
    assert compression["serialized_byte_reduction_minimum"] == 0.20
    assert compression["logical_mac_reduction_for_compute_claim_minimum"] == 0.10


def test_protocol_resource_rules_reject_fake_savings() -> None:
    protocol = build_layer_agnostic_whole_model_compression_protocol()
    accounting = protocol["resource_accounting"]
    assert accounting["count_all_transports_corrections_and_sparse_indices"]
    assert accounting["deduplicate_only_proven_physical_aliases"]
    assert accounting["equal_values_without_alias_are_deduplicated"] is False
    assert accounting["negative_savings_are_preserved_not_clamped"]
    assert accounting["primary_stored_scalar_footprint"] == (
        "unique_learned_plus_fixed_inference_scalars"
    )
    assert accounting[
        "causal_pair_workload_keyed_by_sequence_operator_contract"
    ]
    assert accounting["mixed_global_and_sliding_attention_priced_separately"]
    assert accounting["deployment_artifact_component_spans_must_close_exactly"]
    assert accounting["serialized_bytes_come_from_audited_artifact_spans"]
    assert accounting["counted_exact_components_disqualify_partial_compression"] is False
    assert accounting["unaccounted_teacher_duplicate_disqualifies_compression"]
    assert accounting["reachable_native_fallback_disqualifies_compression"]
    assert (
        accounting["generic_resource_eligibility_authorizes_gemma_compression"]
        is False
    )
    assert accounting["gemma_bridge_and_family_disjoint_guard_required"]
    assert accounting["logical_macs_and_executed_kernel_costs_separate"]
    assert accounting["latency_inferred_from_macs"] is False
    references = protocol["external_rate_references"]
    assert references["primary_graph_candidate_storage_encoding"] == "float32"
    assert references["primary_graph_quantized_storage_allowed"] is False


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["runtime_ontology"].__setitem__(
            "native_layers_are_execution_objects", True
        ),
        lambda value: value["layer_agnostic_discovery_policy"].__setitem__(
            "candidate_pair_scope", "same_layer"
        ),
        lambda value: value["shared_lexical_boundary"].__setitem__(
            "physical_storage_multiplicity", 2
        ),
        lambda value: value["causal_and_runtime_invariants"].__setitem__(
            "future_token_edges_forbidden", False
        ),
        lambda value: value["resource_accounting"].__setitem__(
            "count_unaccounted_or_live_fallbacks", False
        ),
        lambda value: value["setup_receipt"].__setitem__(
            "experiment_executed", True
        ),
        lambda value: value["claim_boundary"].__setitem__(
            "compression_success_claimed", True
        ),
        lambda value: value["shared_lexical_boundary"].__setitem__(
            "terminal_rmsnorm_preserved_exactly", False
        ),
    ),
)
def test_strict_validator_rejects_semantic_tampering_even_if_rehashed(
    mutate,
) -> None:
    value = deepcopy(build_layer_agnostic_whole_model_compression_protocol())
    mutate(value)
    value["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen v1"):
        validate_layer_agnostic_whole_model_compression_protocol(value)


def test_protocol_module_has_no_tensor_or_model_runtime_dependency() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    allowed_roots = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= allowed_roots
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "load_gemma3" not in source
    assert "torch.load" not in source
    assert "AutoModel" not in source


def test_setup_cli_writes_once_without_runtime_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "protocol.json"
    assert main(["--output", str(destination)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["setup_only"] is True
    assert receipt["model_weights_loaded"] is False
    assert receipt["experiment_executed"] is False
    loaded = load_layer_agnostic_whole_model_compression_protocol(destination)
    assert loaded == build_layer_agnostic_whole_model_compression_protocol()
    assert main(["--check", "--output", str(destination)]) == 0
    check_receipt = json.loads(capsys.readouterr().out)
    assert check_receipt["action"] == "checked"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_layer_agnostic_whole_model_compression_protocol(destination)
    with pytest.raises(SystemExit):
        main(["--revision", "not-allowed", "--output", str(tmp_path / "x")])


def test_setup_entrypoint_check_never_imports_tensor_or_model_runtimes(
    tmp_path: Path,
) -> None:
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "fisher-graph-gemma-layer-agnostic-compression-setup = "
        '"fisher_graph_layer_agnostic_compression_setup:main"'
    ) in pyproject
    destination = tmp_path / "protocol.json"
    write_layer_agnostic_whole_model_compression_protocol(destination)
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        """
import builtins

_original_import = builtins.__import__
_blocked_roots = {"mlx", "torch", "transformers"}

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in _blocked_roots:
        raise RuntimeError(f"blocked runtime import: {name}")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = _guarded_import
""".lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    python_path = [str(tmp_path), str(_REPO_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisher_graph_layer_agnostic_compression_setup",
            "--check",
            "--output",
            str(destination),
        ],
        cwd=_REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt == {
        "action": "checked",
        "experiment_executed": False,
        "model_weights_loaded": False,
        "output": str(destination),
        "protocol_sha256": (
            build_layer_agnostic_whole_model_compression_protocol()[
                "protocol_sha256"
            ]
        ),
        "setup_only": True,
    }


def test_checked_in_protocol_matches_builder() -> None:
    source = _REPO_ROOT / DEFAULT_PROTOCOL_OUTPUT
    assert load_layer_agnostic_whole_model_compression_protocol(source) == (
        build_layer_agnostic_whole_model_compression_protocol()
    )


def test_invalid_custom_ladders_fail_closed() -> None:
    with pytest.raises(ValueError, match="residual_rank_ladder"):
        LayerAgnosticWholeModelCompressionProtocol(
            residual_rank_ladder=(640, 384, 512)
        )
    with pytest.raises(ValueError, match="community budget"):
        LayerAgnosticWholeModelCompressionProtocol(
            generator_community_budget_ratio_ladder=(1.0, 1.2)
        )
