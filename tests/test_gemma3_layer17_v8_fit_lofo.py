from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_layer17_family_lofo_authority as authority_module
import fisher_graph.gemma3_layer17_v8_fit_lofo as lofo
from fisher_graph.gemma3_layer17_family_lofo_authority import (
    GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA,
    GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA,
    Gemma3Layer17FamilyLOFOAuthority,
)
from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    build_default_v8_layer17_family_lofo_protocol,
)
from fisher_graph.gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_TOPOLOGY_SHA256,
    build_layer17_node_rank_resource_row,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    LayerFragmentRows,
    fit_layer_cluster_modal_generator,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from fisher_graph.parameter_fisher_coupling import (
    natural_mlp_input_catalog_sha256,
)


MODEL_SHA = "a" * 64
CLUSTER_SHA = "b" * 64
FISHER_SHA = "c" * 64
CATALOG_SHA = "d" * 64
AUTHORITY_SHA = "e" * 64


def _synthetic_aligned_rows() -> tuple[
    AlignedFragmentRows,
    dict[str, str],
]:
    row_keys: list[tuple[str, int]] = []
    family_by_example: dict[str, str] = {}
    for alias in V8_FAMILY_LOFO_FAMILY_ALIASES:
        for example_index in range(2):
            example_id = f"opaque-{alias}-{example_index}"
            family_by_example[example_id] = alias
            row_keys.extend(((example_id, 0), (example_id, 1)))
    observations = len(row_keys)
    base = torch.arange(observations, dtype=torch.float64)
    rows = {
        fragment_id: LayerFragmentRows(
            inputs=torch.stack(
                (base + index, base.square() / 100.0 + 1.0, base * 0.1),
                dim=1,
            ),
            contributions=torch.stack(
                (
                    base * (index + 1),
                    torch.sin(base + index),
                    torch.cos(base - index),
                    base.square() / (index + 2),
                ),
                dim=1,
            ),
            fisher_weights=base + index + 1.0,
            sequences=len(family_by_example),
        )
        for index, fragment_id in enumerate(LAYER17_FRAGMENT_IDS)
    }
    return AlignedFragmentRows(
        rows_by_fragment=rows,
        row_keys=tuple(row_keys),
    ), family_by_example


def _fold_metric(
    index: int,
    *,
    deletion_delta: float = 0.2,
) -> dict[str, object]:
    native_nll = 5.0 + index * 0.001
    resources = _resources()
    logical_valid_tokens = 128
    source_whole_model_parameters = 100_000_000
    native_removed_macs = (
        logical_valid_tokens * resources["source_macs_per_token"]
    )
    graph_macs = (
        logical_valid_tokens * resources["graph_dense_macs_per_token"]
    )
    graph_additions = logical_valid_tokens * 4

    def condition(
        delta: float,
        *,
        kl: float,
        top1: float,
    ) -> dict[str, float]:
        return {
            "nll_per_token": native_nll + delta,
            "delta_nll_per_token": delta,
            "native_to_candidate_kl_per_token": kl,
            "top1_agreement_to_native": top1,
        }

    return {
        "execution_path": "paired_layer17_lofo_and_frozen_edgeless_graphs",
        "supervised_tokens": 120,
        "native": {"nll_per_token": native_nll},
        "conditions": {
            "lofo_refit": condition(0.05, kl=0.03, top1=0.9),
            "frozen_v9_cap48": condition(0.09, kl=0.07, top1=0.82),
            "matched_deletion": condition(
                deletion_delta,
                kl=0.15,
                top1=0.65,
            ),
        },
        "logical_valid_tokens": logical_valid_tokens,
        "graph": {
            "node_count": 4,
            "interaction_count": 0,
            "traversal_order": list(lofo._LAYER17_NODE_NAMES),
        },
        "resource_accounting": {
            "replacement_scope": "partial_native_mlp_mode_replacement",
            "replaced_layer_count": 1,
            "graph_node_count": 4,
            "fragment_count": 4,
            "removed_mode_count": 230,
            "source_whole_model_learned_parameters": (
                source_whole_model_parameters
            ),
            "candidate_whole_model_learned_parameters": (
                source_whole_model_parameters
                - resources["net_parameter_savings"]
            ),
            "native_removed_learned_parameters": resources[
                "source_parameter_count"
            ],
            "modal_graph_learned_parameters": resources[
                "graph_parameter_count"
            ],
            "net_stored_parameter_savings": resources[
                "net_parameter_savings"
            ],
            "graph_runtime_storage": (
                "registered_copied_device_local_graph_parameters"
            ),
            "lofo_generated": {
                "logical_linear_macs_native_removed": native_removed_macs,
                "logical_modal_graph_macs": graph_macs,
                "logical_executed_modal_graph_macs": graph_macs,
                "logical_modal_graph_additions": graph_additions,
                "logical_executed_modal_graph_additions": graph_additions,
                "net_logical_macs_saved": native_removed_macs - graph_macs,
            },
            "matched_deletion_executed_graph_macs": 0,
            "latency_or_kernel_speed_claim": False,
        },
    }


def _resources() -> dict[str, object]:
    return build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=48,
        generator_rank=16,
        edge_policy="edgeless",
    ).state_dict()


def _authority_metadata() -> dict[str, object]:
    binding = authority_module._public_protocol_binding(
        build_default_v8_layer17_family_lofo_protocol()
    )
    protocol = {
        key: binding[key]
        for key in (
            "protocol_artifact_sha256",
            "fit_membership_sha256",
            "family_alias_mapping_sha256",
            "fold_count",
            "folds",
        )
    }
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA,
        "format_version": 1,
        "scientific_role": "open_development_calibration_a_fit_family_lofo",
        "heldout_confirmation": False,
        "receipt": {
            "receipt_sha256": "1" * 64,
            "receipt_file_sha256": "2" * 64,
        },
        "protocol": protocol,
        "corpus": {
            "corpus_artifact_sha256": binding["corpus_artifact_sha256"],
            "corpus_artifact_file_sha256": "3" * 64,
            "tokenizer_contract_sha256": binding[
                "tokenizer_contract_sha256"
            ],
            "fit_manifest_sha256": binding["fit_manifest_sha256"],
            "fit_role_file_sha256": binding["fit_role_file_sha256"],
            "example_count": 256,
            "block_count": 8,
            "examples_per_block": 32,
            "block_labels": list(V8_FAMILY_LOFO_FAMILY_ALIASES),
        },
        "access": dict(authority_module._AUTHORITY_ACCESS),
        "safety": dict(authority_module._PUBLIC_SAFETY),
    }
    result = {
        **payload,
        "authority_sha256": authority_module._domain_sha256(
            authority_module._AUTHORITY_DOMAIN,
            payload,
        ),
    }
    authority_module.validate_gemma3_layer17_family_lofo_authority_metadata(
        result
    )
    return result


def _fit_collection(authority: dict[str, object]) -> dict[str, object]:
    blocks = {
        alias: {
            "example_count": 32,
            "batch_count": 4,
            "logical_valid_tokens": 128,
            "supervised_tokens": 120,
        }
        for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
    }
    tokenization = {
        "block_count": 8,
        "block_labels": list(V8_FAMILY_LOFO_FAMILY_ALIASES),
        "example_count": 256,
        "examples_per_block": 32,
        "batch_count": 32,
        "logical_valid_tokens": 1024,
        "supervised_tokens": 960,
        "max_length": 128,
        "tokenization_batch_size": 8,
        "device": "cpu",
        "stream_catalog_sha256": "4" * 64,
        "blocks": blocks,
    }
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA,
        "format_version": 1,
        "scientific_role": "open_development_calibration_a_fit_family_lofo",
        "heldout_confirmation": False,
        "authority_sha256": authority["authority_sha256"],
        "tokenization": tokenization,
        "access": {
            "fit_opened": True,
            "fit_tokenized": True,
            "selection_opened": False,
            "guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "model_loaded": False,
            "model_evaluated": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_family_ids": False,
            "contains_token_ids": False,
            "contains_tokenized_content_identities": False,
            "contains_logits": False,
            "contains_model_or_candidate_weights": False,
            "source_safe": True,
        },
    }
    materialization_sha256 = authority_module._domain_sha256(
        authority_module._MATERIALIZATION_DOMAIN,
        payload,
    )
    return {
        **payload,
        "materialization_sha256": materialization_sha256,
        "capture_count": 1,
        "captured_examples": 256,
        "captured_observations": 2048,
        "captured_sequences": 256,
        "captured_row_key_sha256": "5" * 64,
        "family_observations": {
            alias: 256 for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
        },
        "model_rows_recollected_per_fold": False,
    }


def _report() -> dict[str, object]:
    protocol = build_default_v8_layer17_family_lofo_protocol()
    folds = lofo.build_layer17_family_folds(
        V8_FAMILY_LOFO_FAMILY_ALIASES,
        protocol=protocol,
    )
    evaluations = tuple(_fold_metric(index) for index in range(8))
    fold_rows = [
        {
            **fold.metadata(),
            "fit_split_sha256": hashlib.sha256(
                f"fit/{index}".encode()
            ).hexdigest(),
            "held_split_sha256": hashlib.sha256(
                f"held/{index}".encode()
            ).hexdigest(),
            "fit_observations": 1792,
            "held_observations": 256,
            "fit_sequences": 224,
            "held_sequences": 32,
            "fisher_normalization": lofo._FISHER_NORMALIZATION,
            "fit_fisher_total_mass_per_training_family": 1.0 / 7.0,
            "held_fisher_total_mass": 1.0,
            "held_family_excluded_from_center_basis_fisher_and_generator": True,
            "held_rows_used_only_for_fixed_rank_descriptive_metrics": True,
            "lowering_sha256_by_node": {
                node_name: hashlib.sha256(
                    f"{index}/{node_name}".encode()
                ).hexdigest()
                for node_name in evaluations[index]["graph"][
                    "traversal_order"
                ]
            },
            "evaluation": evaluations[index],
        }
        for index, fold in enumerate(folds)
    ]
    resources = _resources()
    aggregate = lofo.aggregate_layer17_lofo_fold_metrics(evaluations)
    decision = lofo.evaluate_layer17_lofo_protocol_gates(
        protocol=protocol,
        fold_metrics=evaluations,
        aggregate=aggregate,
        resources=resources,
    )
    authority = _authority_metadata()
    return lofo.build_layer17_v8_fit_lofo_report(
        protocol=protocol,
        authority=authority,
        experiment={
            "experiment_kind": "gemma3_layer17_v8_fit_family_lofo_v1",
            "scientific_role": "calibration_a_fit_cross_fitted_diagnostic",
            "model_id": "google/gemma-3-270m",
            "requested_revision": "6" * 40,
            "adapter_model_fingerprint": MODEL_SHA,
            "source_model_unchanged": True,
        },
        fit_collection=_fit_collection(authority),
        folds=fold_rows,
        aggregate=aggregate,
        resources=resources,
        decision=decision,
        lineage={
            "frozen_v9_candidate_file": Path(
                lofo.DEFAULT_FROZEN_CAP48_CANDIDATE
            ).name,
            "frozen_v9_candidate_file_sha256": "7" * 64,
            "frozen_v9_candidate_scientific_sha256": "8" * 64,
            "base_artifact_file": Path(lofo.DEFAULT_BASE_ARTIFACT).name,
            "base_artifact_file_sha256": "9" * 64,
            "frozen_topology_sha256": LAYER17_TOPOLOGY_SHA256,
            "fragment_plan_sha256": "a" * 64,
            "fragment_selection_sha256": "b" * 64,
        },
    )


def _small_fragment_plan() -> tuple[
    ParameterClusterLayerFragmentPlan,
    ParameterClusterLayerFragment,
]:
    input_width = 16
    output_width = 48
    input_site = "model.layers.17.mlp.input"
    input_catalog_sha256 = natural_mlp_input_catalog_sha256(
        source_model_sha256=MODEL_SHA,
        input_site=input_site,
        input_width=input_width,
    )
    fragment = ParameterClusterLayerFragment(
        cluster_id=0,
        layer_ordinal=17,
        layer_id="model.layers.17",
        activation_site="model.layers.17.mlp.down_input",
        input_site=input_site,
        output_site="model.layers.17.mlp.residual_delta",
        input_catalog_sha256=input_catalog_sha256,
        input_width=input_width,
        output_width=output_width,
        group_indices=tuple(range(48)),
        channel_indices=tuple(range(48)),
        fisher_ranks=tuple(range(48)),
        axial_orientations=(1,) * 48,
        native_parameter_count=48 * (2 * input_width + output_width),
        fisher_mass=1.0,
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        source_model_sha256=MODEL_SHA,
    )
    plan = ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        source_model_sha256=MODEL_SHA,
        cluster_count=1,
        source_group_count=48,
        assigned_group_count=48,
        assigned_native_parameter_count=fragment.native_parameter_count,
        fragments=(fragment,),
    )
    return plan, fragment


def _fit_pilot(
    fit_rows: LayerFragmentRows,
    eval_rows: LayerFragmentRows,
    *,
    fit_sha: str,
    eval_sha: str,
):
    plan, fragment = _small_fragment_plan()
    return fit_layer_cluster_modal_generator(
        fit_rows,
        eval_rows,
        selection=fragment,
        source_model_sha256=MODEL_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        fisher_coupling_sha256=FISHER_SHA,
        fragment_plan=plan,
        fit_split_sha256=fit_sha,
        eval_split_sha256=eval_sha,
        input_site=fragment.input_site,
        output_site=fragment.output_site,
        mode_ranks=(48,),
        selected_mode_rank=48,
        generator_ranks=(16,),
        selected_generator_rank=16,
        ridge=0.0,
    )


def _numeric_fit_tensors(pilot) -> tuple[torch.Tensor, ...]:
    basis = pilot.computational_modes.selected_basis
    generator = pilot.modal_generators.selected_plan
    assert basis is not None
    assert generator is not None
    assert generator.factors.bias is not None
    graph_weights = pilot.lowering.graph_weights
    values = (
        basis.mean_bias,
        basis.encoder_basis,
        generator.factors.input_factor,
        generator.factors.output_factor,
        generator.factors.bias,
        graph_weights.input_factor,
        graph_weights.state_factor,
        graph_weights.output_factor,
        graph_weights.latent_bias,
        graph_weights.output_bias,
    )
    return tuple(value for value in values if value is not None)


def test_fold_partition_is_disjoint_exhaustive_and_equal_family_normalized():
    rows, family_by_example = _synthetic_aligned_rows()
    family_rows = lofo.partition_aligned_fragment_rows_by_family(
        rows,
        family_by_example,
    )
    protocol = build_default_v8_layer17_family_lofo_protocol()
    folds = lofo.build_layer17_family_folds(
        tuple(family_rows),
        protocol=protocol,
    )

    assert len(folds) == 8
    split_hashes: set[str] = set()
    for fold in folds:
        train, held, receipt = lofo.make_layer17_lofo_fold_rows(
            family_rows,
            fold,
            authority_sha256=AUTHORITY_SHA,
        )
        assert set(train.row_keys).isdisjoint(held.row_keys)
        assert set(train.row_keys) | set(held.row_keys) == set(rows.row_keys)
        assert train.sequences == 14
        assert held.sequences == 2
        split_hashes.update(
            (receipt["fit_split_sha256"], receipt["held_split_sha256"])
        )
        for fragment in train.rows_by_fragment.values():
            assert fragment.fisher_weights.sum().item() == pytest.approx(1.0)
            offset = 0
            for alias in fold.training_family_aliases:
                count = family_rows[alias].observations
                family_mass = fragment.fisher_weights[offset : offset + count].sum()
                assert family_mass.item() == pytest.approx(1.0 / 7.0)
                offset += count
        for fragment in held.rows_by_fragment.values():
            assert fragment.fisher_weights.sum().item() == pytest.approx(1.0)
    assert len(split_hashes) == 16


def test_protocol_gates_pass_and_invalid_deletion_denominator_fails_closed():
    protocol = build_default_v8_layer17_family_lofo_protocol()
    passing = tuple(_fold_metric(index) for index in range(8))
    aggregate = lofo.aggregate_layer17_lofo_fold_metrics(passing)
    decision = lofo.evaluate_layer17_lofo_protocol_gates(
        protocol=protocol,
        fold_metrics=passing,
        aggregate=aggregate,
        resources=_resources(),
    )
    assert decision["all_required_gates_pass"] is True
    assert decision["deletion_nll_recovery_denominator_valid"] is True
    assert decision["deletion_nll_recovery_invalid_denominator_count"] == 0
    assert tuple(
        decision["deletion_nll_recovery_fraction_by_family_alias"].values()
    ) == pytest.approx((0.75,) * 8)

    invalid = list(copy.deepcopy(passing))
    invalid[3] = _fold_metric(3, deletion_delta=0.0)
    invalid_aggregate = lofo.aggregate_layer17_lofo_fold_metrics(invalid)
    invalid_decision = lofo.evaluate_layer17_lofo_protocol_gates(
        protocol=protocol,
        fold_metrics=invalid,
        aggregate=invalid_aggregate,
        resources=_resources(),
    )
    assert invalid_decision["all_required_gates_pass"] is False
    assert invalid_decision["deletion_nll_recovery_denominator_valid"] is False
    assert invalid_decision["deletion_nll_recovery_invalid_denominator_count"] == 1
    assert invalid_decision[
        "deletion_nll_recovery_fraction_by_family_alias"
    ]["family_03"] is None
    recovery_gates = {
        row["gate_id"]: row
        for row in invalid_decision["gate_table"]
        if "deletion_nll_recovery" in row["gate_id"]
    }
    assert all(row["passed"] is False for row in recovery_gates.values())

    bad_folds = copy.deepcopy(passing)
    for row in bad_folds:
        native_nll = row["native"]["nll_per_token"]
        row["conditions"]["lofo_refit"] = {
            "nll_per_token": native_nll + 0.5,
            "delta_nll_per_token": 0.5,
            "native_to_candidate_kl_per_token": 0.5,
            "top1_agreement_to_native": 0.1,
        }
        row["conditions"]["matched_deletion"] = {
            "nll_per_token": native_nll + 2.0,
            "delta_nll_per_token": 2.0,
            "native_to_candidate_kl_per_token": 0.8,
            "top1_agreement_to_native": 0.05,
        }
    with pytest.raises(ValueError, match="aggregate differs"):
        lofo.evaluate_layer17_lofo_protocol_gates(
            protocol=protocol,
            fold_metrics=bad_folds,
            aggregate=aggregate,
            resources=_resources(),
        )

    forged_resources = copy.deepcopy(_resources())
    forged_resources["net_parameter_savings"] += 1
    with pytest.raises(ValueError, match="resource"):
        lofo.evaluate_layer17_lofo_protocol_gates(
            protocol=protocol,
            fold_metrics=passing,
            aggregate=aggregate,
            resources=forged_resources,
        )


def test_report_roundtrips_and_recomputed_hash_cannot_hide_decision_tamper(
    tmp_path: Path,
):
    report = _report()
    destination = tmp_path / ".local-runs" / "lofo.json"
    saved = lofo.save_gemma3_layer17_v8_fit_lofo_report(
        destination,
        report,
    )
    assert saved == report
    assert lofo.load_gemma3_layer17_v8_fit_lofo_report(destination) == report

    corrupted = copy.deepcopy(report)
    corrupted["decision"]["all_required_gates_pass"] = False
    payload = dict(corrupted)
    payload.pop("report_sha256")
    corrupted["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="decision"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(corrupted)

    renamed_lowering = copy.deepcopy(report)
    lowering_catalog = renamed_lowering["folds"][0][
        "lowering_sha256_by_node"
    ]
    digest = lowering_catalog.pop(lofo._LAYER17_NODE_NAMES[0])
    lowering_catalog["unbound-node"] = digest
    payload = dict(renamed_lowering)
    payload.pop("report_sha256")
    renamed_lowering["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="fitting receipt"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(renamed_lowering)

    unsafe_node_name = copy.deepcopy(report)
    unsafe_graph = unsafe_node_name["folds"][0]["evaluation"]["graph"]
    old_name = unsafe_graph["traversal_order"][0]
    unsafe_graph["traversal_order"][0] = "raw prompt text could be here"
    unsafe_lowerings = unsafe_node_name["folds"][0][
        "lowering_sha256_by_node"
    ]
    unsafe_lowerings["raw prompt text could be here"] = unsafe_lowerings.pop(
        old_name
    )
    payload = dict(unsafe_node_name)
    payload.pop("report_sha256")
    unsafe_node_name["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="evaluation contract"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(unsafe_node_name)

    contradictory_resources = copy.deepcopy(report)
    contradictory_resources["folds"][0]["evaluation"][
        "resource_accounting"
    ]["lofo_generated"]["logical_executed_modal_graph_macs"] += 1
    payload = dict(contradictory_resources)
    payload.pop("report_sha256")
    contradictory_resources["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="logical resources"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(
            contradictory_resources
        )

    string_metric = copy.deepcopy(report)
    string_metric["folds"][0]["evaluation"]["conditions"]["lofo_refit"][
        "delta_nll_per_token"
    ] = "0.05"
    payload = dict(string_metric)
    payload.pop("report_sha256")
    string_metric["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="JSON numbers"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(string_metric)

    unsafe = copy.deepcopy(report)
    unsafe["authority"]["prompts"] = ["must not serialize"]
    payload = dict(unsafe)
    payload.pop("report_sha256")
    unsafe["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="forbidden source field"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(unsafe)

    boolean_version = copy.deepcopy(report)
    boolean_version["format_version"] = True
    payload = dict(boolean_version)
    payload.pop("report_sha256")
    boolean_version["report_sha256"] = lofo._sha256(
        payload,
        domain=lofo._REPORT_DOMAIN,
    )
    with pytest.raises(ValueError, match="unsupported"):
        lofo.validate_gemma3_layer17_v8_fit_lofo_report(boolean_version)


def test_fixed_rank_fit_is_numerically_independent_of_held_rows():
    generator = torch.Generator().manual_seed(20260806)
    fit_inputs = torch.randn(80, 16, generator=generator, dtype=torch.float64)
    teacher = torch.randn(16, 48, generator=generator, dtype=torch.float64)
    fit_contributions = fit_inputs @ teacher + 0.03 * torch.randn(
        80,
        48,
        generator=generator,
        dtype=torch.float64,
    )
    fit_rows = LayerFragmentRows(
        inputs=fit_inputs,
        contributions=fit_contributions,
        fisher_weights=torch.rand(80, generator=generator, dtype=torch.float64)
        + 0.25,
        sequences=70,
    )
    eval_inputs = torch.randn(24, 16, generator=generator, dtype=torch.float64)
    eval_rows_a = LayerFragmentRows(
        inputs=eval_inputs,
        contributions=eval_inputs @ teacher,
        fisher_weights=torch.ones(24, dtype=torch.float64),
        sequences=10,
    )
    eval_rows_b = LayerFragmentRows(
        inputs=eval_inputs * -13.0 + 7.0,
        contributions=torch.randn(
            24,
            48,
            generator=generator,
            dtype=torch.float64,
        )
        * 100.0,
        fisher_weights=torch.linspace(0.1, 10.0, 24, dtype=torch.float64),
        sequences=10,
    )

    pilot_a = _fit_pilot(
        fit_rows,
        eval_rows_a,
        fit_sha="1" * 64,
        eval_sha="2" * 64,
    )
    pilot_b = _fit_pilot(
        fit_rows,
        eval_rows_b,
        fit_sha="1" * 64,
        eval_sha="3" * 64,
    )
    assert pilot_a.lowering.artifact_sha256 != pilot_b.lowering.artifact_sha256
    for left, right in zip(
        _numeric_fit_tensors(pilot_a),
        _numeric_fit_tensors(pilot_b),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    changed_fit = LayerFragmentRows(
        inputs=fit_inputs,
        contributions=fit_contributions
        + 0.2
        * torch.randn(80, 48, generator=generator, dtype=torch.float64),
        fisher_weights=fit_rows.fisher_weights,
        sequences=fit_rows.sequences,
    )
    pilot_changed = _fit_pilot(
        changed_fit,
        eval_rows_a,
        fit_sha="4" * 64,
        eval_sha="2" * 64,
    )
    assert any(
        not torch.allclose(left, right, rtol=1e-10, atol=1e-12)
        for left, right in zip(
            _numeric_fit_tensors(pilot_a),
            _numeric_fit_tensors(pilot_changed),
            strict=True,
        )
    )


def test_cli_and_runner_expose_no_protected_role_path_or_selector():
    parser = lofo.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {"--corpus-receipt", "--corpus-artifact", "--fit-input"}.issubset(
        option_strings
    )
    forbidden = ("selection", "guard", "calibration-b", "validation", "test")
    assert all(
        token not in option
        for option in option_strings
        for token in forbidden
    )
    parameters = inspect.signature(lofo.run_gemma3_layer17_v8_fit_lofo).parameters
    assert all(
        token not in name.replace("_", "-")
        for name in parameters
        for token in forbidden
    )


def test_restored_lowering_map_is_ordered_for_executor_boundary():
    graph = SimpleNamespace(
        nodes=(SimpleNamespace(name="node-b"), SimpleNamespace(name="node-a"))
    )
    lowering_a = object()
    lowering_b = object()
    assert lofo._ordered_restored_lowerings(
        graph,
        {"node-a": lowering_a, "node-b": lowering_b},
    ) == (lowering_b, lowering_a)
    with pytest.raises(ValueError, match="catalog differ"):
        lofo._ordered_restored_lowerings(graph, {"node-a": lowering_a})


def test_graph_topology_signature_uses_live_latent_width_contract():
    node = SimpleNamespace(
        name="node-a",
        input_boundary="layer.17.input",
        output_boundary="layer.17.output",
        input_width=640,
        latent_width=48,
        output_width=640,
    )
    plan = SimpleNamespace(nodes=(node,))
    assert lofo._graph_node_topology_signature(plan) == (
        (
            "node-a",
            "layer.17.input",
            "layer.17.output",
            640,
            48,
            640,
        ),
    )
