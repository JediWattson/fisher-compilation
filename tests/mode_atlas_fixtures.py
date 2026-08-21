"""Reusable model-free fixtures for mode-atlas protocol/worker tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from fisher_graph.gemma3_mlp_mode_similarity_atlas_setup import (
    build_gemma3_mlp_mode_similarity_atlas_setup,
)
from fisher_graph.mode_similarity_atlas import (
    AtlasConfig,
    DirectedFamilyExampleStats,
    DirectedFamilyStats,
    ModeABI,
    ModeSimilarityAtlasReceipt,
    ModeSpec,
    WithinFamilyBootstrapConfig,
    build_mode_similarity_atlas_receipt,
    parse_atlas_config_payload,
    parse_within_family_bootstrap_config_payload,
)


@dataclass(frozen=True, slots=True)
class ProductionScaleSyntheticModeAtlas:
    setup: Mapping[str, object]
    abi: ModeABI
    config: AtlasConfig
    bootstrap_config: WithinFamilyBootstrapConfig
    modes: tuple[ModeSpec, ...]
    directed_stats: tuple[DirectedFamilyStats, ...]
    directed_example_stats: tuple[DirectedFamilyExampleStats, ...]
    receipt: ModeSimilarityAtlasReceipt


def build_production_scale_synthetic_mode_atlas(
) -> ProductionScaleSyntheticModeAtlas:
    """Build the exact 18x6x10/10k shape without loading Torch or Gemma."""

    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    abi = ModeABI(**setup["comparison_abi"])
    config = parse_atlas_config_payload(setup["atlas_config"])
    bootstrap_config = parse_within_family_bootstrap_config_payload(
        setup["within_family_bootstrap_config"]
    )
    scope = setup["scope_freeze"]
    assert isinstance(scope, Mapping)
    catalog = scope["mode_catalog"]
    assert isinstance(catalog, list)
    mode_kind_by_id = {
        item["mode_id"]: item["generic_core_mode_kind"]
        for item in catalog
        if isinstance(item, Mapping)
    }
    ordered_mode_ids = tuple(scope["ordered_mode_ids"])
    positions = {
        mode_id: float(index * index + index / 11.0)
        for index, mode_id in enumerate(ordered_mode_ids)
    }
    modes = tuple(
        ModeSpec(
            mode_id=mode_id,
            mode_kind=str(mode_kind_by_id[mode_id]),
            state_fingerprint_sha256=hashlib.sha256(
                f"synthetic-state:{mode_id}".encode("utf-8")
            ).hexdigest(),
            compatibility_abi=abi,
        )
        for mode_id in ordered_mode_ids
    )

    directed_stats: list[DirectedFamilyStats] = []
    directed_example_stats: list[DirectedFamilyExampleStats] = []
    for family_index, (family_id, example_ids) in enumerate(
        bootstrap_config.family_example_cluster_ids
    ):
        unnormalized_shares = tuple(1.0 + index / 10.0 for index in range(10))
        share_total = sum(unnormalized_shares)
        for reference_mode_id, reference_position in positions.items():
            for candidate_mode_id, candidate_position in positions.items():
                if reference_mode_id == candidate_mode_id:
                    continue
                weighted_sse = (
                    (1.0 + family_index / 20.0)
                    * (reference_position - candidate_position) ** 2
                )
                directed_stats.append(
                    DirectedFamilyStats(
                        family_id=family_id,
                        reference_mode_id=reference_mode_id,
                        candidate_mode_id=candidate_mode_id,
                        compatibility_abi_sha256=abi.sha256,
                        weighted_sse=weighted_sse,
                        weighted_reference_energy=1.0,
                        weighted_dot=1.0,
                        weighted_candidate_energy=1.0 + weighted_sse,
                        row_count=20,
                    )
                )
                for example_index, example_id in enumerate(example_ids):
                    share = unnormalized_shares[example_index] / share_total
                    directed_example_stats.append(
                        DirectedFamilyExampleStats(
                            family_id=family_id,
                            example_cluster_id=example_id,
                            reference_mode_id=reference_mode_id,
                            candidate_mode_id=candidate_mode_id,
                            compatibility_abi_sha256=abi.sha256,
                            weighted_sse=share * weighted_sse,
                            weighted_reference_energy=share,
                            weighted_dot=share,
                            weighted_candidate_energy=(
                                share * (1.0 + weighted_sse)
                            ),
                            row_count=2,
                        )
                    )
    receipt = build_mode_similarity_atlas_receipt(
        modes,
        directed_stats,
        config=config,
        directed_example_stats=directed_example_stats,
        within_family_bootstrap_config=bootstrap_config,
    )
    return ProductionScaleSyntheticModeAtlas(
        setup=setup,
        abi=abi,
        config=config,
        bootstrap_config=bootstrap_config,
        modes=modes,
        directed_stats=tuple(directed_stats),
        directed_example_stats=tuple(directed_example_stats),
        receipt=receipt,
    )
