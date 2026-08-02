from __future__ import annotations

from collections import Counter
import hashlib
import json

import pytest
import torch

from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_panel import (
    FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
    Gemma3L3L4H4DampingExpandedFitLineage,
    Gemma3L3L4H4DampingSelectionPanelArtifact,
    _prompt_blind_forbidden_binding,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_panel import (
    GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE,
    ITERATIVE_OCCUPANCY_SELECTION_FAMILIES,
    ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE,
    ITERATIVE_OCCUPANCY_SELECTION_PROMPTS,
    Gemma3L3L4IterativeOccupancySelectionClosedError,
    Gemma3L3L4IterativeOccupancySelectionIntegrityError,
    Gemma3L3L4IterativeOccupancySelectionPanelSource,
    claim_gemma3_l3_l4_iterative_occupancy_selection_panel,
    freeze_gemma3_l3_l4_iterative_occupancy_selection_panel,
    load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact,
    materialize_gemma3_l3_l4_iterative_occupancy_selection_panel,
    write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact,
    write_gemma3_l3_l4_iterative_occupancy_selection_role_input,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _lineage(
    *,
    extra_prompt_sha256s: tuple[str, ...] = (),
    extra_family_ids: tuple[str, ...] = (),
) -> Gemma3L3L4H4DampingExpandedFitLineage:
    fit_prompts = tuple(_sha(100 + index) for index in range(16))
    fit_families = tuple(
        f"fixture-expanded-family-{index}" for index in range(8)
    )
    forbidden_manifest, _prompts, _families = (
        _prompt_blind_forbidden_binding()
    )
    fit_manifest = _sha(20)
    return Gemma3L3L4H4DampingExpandedFitLineage(
        expanded_corpus_artifact_sha256=_sha(1),
        tokenizer_contract_sha256=_sha(2),
        fit_manifest_sha256=fit_manifest,
        fit_role_input_file_sha256=_sha(3),
        fit_binding_sha256=_sha(4),
        fit_example_count=16,
        fit_family_ids=fit_families,
        ordered_fit_prompt_sha256s=fit_prompts,
        ordered_fit_family_ids=fit_families + fit_families,
        occupied_development_manifest_sha256s=(fit_manifest,),
        occupied_development_prompt_sha256s=tuple(
            sorted((*fit_prompts, *extra_prompt_sha256s))
        ),
        occupied_development_family_ids=tuple(
            sorted((*fit_families, *extra_family_ids))
        ),
        forbidden_assessment_manifest_sha256s=(forbidden_manifest,),
    )


def _prior(
    lineage: Gemma3L3L4H4DampingExpandedFitLineage,
    *,
    prompt_sha256s: tuple[str, ...] | None = None,
) -> Gemma3L3L4H4DampingSelectionPanelArtifact:
    return Gemma3L3L4H4DampingSelectionPanelArtifact(
        expanded_fit_lineage=lineage,
        selection_role_input_file_sha256=_sha(30),
        ordered_prompt_sha256s=(
            tuple(_sha(300 + index) for index in range(16))
            if prompt_sha256s is None
            else prompt_sha256s
        ),
        ordered_family_ids=FRESH_DAMPING_SELECTION_FAMILY_SCHEDULE,
    )


def _freeze(tmp_path):
    lineage = _lineage()
    prior = _prior(lineage)
    private_path = tmp_path / "selection.private.json"
    write_gemma3_l3_l4_iterative_occupancy_selection_role_input(
        private_path
    )
    artifact = freeze_gemma3_l3_l4_iterative_occupancy_selection_panel(
        expanded_fit_lineage=lineage,
        prior_selection_panel=prior,
        selection_plan_sha256=_sha(900),
        role_input_path=private_path,
    )
    return artifact, private_path


class _Tokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, prompts, **_kwargs):
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            rows.append(
                [
                    2,
                    int.from_bytes(digest[:2], "big") + 3,
                    int.from_bytes(digest[2:4], "big") + 3,
                    1,
                ]
            )
        return {
            "input_ids": torch.tensor(rows, dtype=torch.int64),
            "attention_mask": torch.ones(
                (len(rows), 4),
                dtype=torch.int64,
            ),
        }


def test_frozen_panel_has_broad_balanced_unique_geometry() -> None:
    assert len(ITERATIVE_OCCUPANCY_SELECTION_PROMPTS) == 16
    assert len(ITERATIVE_OCCUPANCY_SELECTION_FAMILIES) == 8
    assert tuple(sorted(ITERATIVE_OCCUPANCY_SELECTION_FAMILIES)) == (
        ITERATIVE_OCCUPANCY_SELECTION_FAMILIES
    )
    assert Counter(ITERATIVE_OCCUPANCY_SELECTION_FAMILY_SCHEDULE) == {
        family_id: 2
        for family_id in ITERATIVE_OCCUPANCY_SELECTION_FAMILIES
    }
    prompt_hashes = tuple(
        gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
        for prompt in ITERATIVE_OCCUPANCY_SELECTION_PROMPTS
    )
    assert len(set(prompt_hashes)) == 16
    forbidden_words = (
        "occupancy",
        "signed state",
        "modes",
        "routing",
        "experts",
    )
    assert all(
        word not in prompt.lower()
        for prompt in ITERATIVE_OCCUPANCY_SELECTION_PROMPTS
        for word in forbidden_words
    )


def test_public_artifact_is_hash_only_roundtrips_and_refuses_overwrite(
    tmp_path,
) -> None:
    artifact, _private_path = _freeze(tmp_path)
    artifact_path = tmp_path / "selection.panel.json"
    write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
        artifact_path,
        artifact,
    )

    encoded = artifact_path.read_text()
    assert '"prompts"' not in encoded
    assert '"token_ids"' not in encoded
    assert all(
        prompt not in encoded
        for prompt in ITERATIVE_OCCUPANCY_SELECTION_PROMPTS
    )
    assert json.loads(encoded)["role"] == (
        GEMMA3_L3_L4_ITERATIVE_OCCUPANCY_SELECTION_ROLE
    )
    restored = (
        load_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
            artifact_path,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
    )
    assert restored == artifact
    with pytest.raises(FileExistsError):
        write_gemma3_l3_l4_iterative_occupancy_selection_panel_artifact(
            artifact_path,
            artifact,
        )


def test_freeze_rejects_expanded_and_prior_prompt_reuse(tmp_path) -> None:
    current_hashes = tuple(
        gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt)
        for prompt in ITERATIVE_OCCUPANCY_SELECTION_PROMPTS
    )
    private_path = tmp_path / "selection.private.json"
    write_gemma3_l3_l4_iterative_occupancy_selection_role_input(
        private_path
    )
    expanded_overlap = _lineage(
        extra_prompt_sha256s=(current_hashes[0],)
    )
    with pytest.raises(ValueError, match="prompts overlap"):
        freeze_gemma3_l3_l4_iterative_occupancy_selection_panel(
            expanded_fit_lineage=expanded_overlap,
            prior_selection_panel=_prior(expanded_overlap),
            selection_plan_sha256=_sha(900),
            role_input_path=private_path,
        )

    lineage = _lineage()
    prior_hashes = (
        current_hashes[0],
        *(tuple(_sha(400 + index) for index in range(15))),
    )
    with pytest.raises(ValueError, match="prompts overlap"):
        freeze_gemma3_l3_l4_iterative_occupancy_selection_panel(
            expanded_fit_lineage=lineage,
            prior_selection_panel=_prior(
                lineage,
                prompt_sha256s=prior_hashes,
            ),
            selection_plan_sha256=_sha(900),
            role_input_path=private_path,
        )


def test_durable_claim_gates_fail_closed_one_use_open(tmp_path) -> None:
    artifact, private_path = _freeze(tmp_path)
    claim_path = tmp_path / "selection.claim.json"
    claim = claim_gemma3_l3_l4_iterative_occupancy_selection_panel(
        claim_path,
        artifact=artifact,
    )
    with pytest.raises(FileExistsError):
        claim_gemma3_l3_l4_iterative_occupancy_selection_panel(
            claim_path,
            artifact=artifact,
        )
    source = Gemma3L3L4IterativeOccupancySelectionPanelSource(
        artifact=artifact,
        role_input_path=private_path,
    )

    opened = source.open_once(claim=claim)

    assert opened.prompts == ITERATIVE_OCCUPANCY_SELECTION_PROMPTS
    assert source.consumed is True
    assert source.opened is True
    with pytest.raises(Gemma3L3L4IterativeOccupancySelectionClosedError):
        source.open_once(claim=claim)


def test_materialization_reuses_progressive_tokenizer_after_claim(
    tmp_path,
) -> None:
    artifact, private_path = _freeze(tmp_path)
    claim = claim_gemma3_l3_l4_iterative_occupancy_selection_panel(
        tmp_path / "selection.claim.json",
        artifact=artifact,
    )
    source = Gemma3L3L4IterativeOccupancySelectionPanelSource(
        artifact=artifact,
        role_input_path=private_path,
    )

    panel = materialize_gemma3_l3_l4_iterative_occupancy_selection_panel(
        source=source,
        claim=claim,
        tokenizer=_Tokenizer(),
        max_length=64,
        device=torch.device("cpu"),
    )

    assert panel.role == "calibration_a_selection"
    assert panel.manifest_sha256 == artifact.manifest_sha256
    assert panel.membership_receipt_sha256 == (
        artifact.membership_receipt_sha256
    )
    assert len(panel.examples) == 16


def test_changed_claim_is_rejected_before_source_consumption(tmp_path) -> None:
    artifact, private_path = _freeze(tmp_path)
    claim_path = tmp_path / "selection.claim.json"
    claim = claim_gemma3_l3_l4_iterative_occupancy_selection_panel(
        claim_path,
        artifact=artifact,
    )
    raw = json.loads(claim_path.read_text())
    raw["selection_plan_sha256"] = _sha(901)
    claim_path.write_text(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    source = Gemma3L3L4IterativeOccupancySelectionPanelSource(
        artifact=artifact,
        role_input_path=private_path,
    )

    with pytest.raises(
        Gemma3L3L4IterativeOccupancySelectionIntegrityError
    ):
        source.open_once(claim=claim)
    assert source.consumed is False
