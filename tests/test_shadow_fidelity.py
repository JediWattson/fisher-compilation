import math

import pytest
import torch
import torch.nn.functional as F

from fisher_graph.shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    ShadowFidelityGates,
    SourceAuthoritativeShadowFidelityAccumulator,
    evaluate_source_authoritative_shadow,
)


def _example(
    example_id: str,
    family_id: str,
    source: list[list[float]],
    candidate: list[list[float]] | None = None,
    targets: list[int] | None = None,
) -> ShadowFidelityExample:
    source_logits = torch.tensor(source, dtype=torch.float64)
    return ShadowFidelityExample(
        example_id=example_id,
        family_id=family_id,
        source_logits=source_logits,
        candidate_logits=(
            source_logits.clone()
            if candidate is None
            else torch.tensor(candidate, dtype=torch.float64)
        ),
        targets=torch.tensor(
            [0] * len(source) if targets is None else targets,
            dtype=torch.int64,
        ),
    )


def test_identical_shadow_passes_and_is_explicitly_source_authoritative() -> None:
    examples = [
        _example("a.1", "family-a", [[4.0, 1.0], [1.0, 4.0]], targets=[0, 1]),
        _example("a.2", "family-a", [[2.0, 1.0]], targets=[0]),
        _example("b.1", "family-b", [[0.0, 3.0]], targets=[1]),
        _example("b.2", "family-b", [[1.0, 0.0], [0.0, 1.0]], targets=[0, 1]),
    ]
    report = evaluate_source_authoritative_shadow(
        examples,
        expected_family_by_example={
            "a.1": "family-a",
            "a.2": "family-a",
            "b.1": "family-b",
            "b.2": "family-b",
        },
        vocab_chunk_size=1,
    )

    assert report["semantics"] == {
        "execution_mode": "shadow",
        "authoritative_path": "source",
        "source_outputs_authoritative": True,
        "candidate_outputs_authoritative": False,
        "candidate_logits_used_for_metrics_only": True,
        "candidate_outputs_must_not_be_served": True,
    }
    assert report["manifest"] == {
        "strict_example_membership": True,
        "strict_family_membership": True,
        "expected_examples": 4,
        "observed_examples": 4,
        "complete": True,
        "family_count": 2,
    }
    aggregate = report["aggregate"]
    assert aggregate["example_count"] == 4
    assert aggregate["supervised_tokens"] == 6
    assert aggregate["delta_nll_per_token"] == pytest.approx(0.0)
    assert aggregate["source_to_candidate_kl_per_token"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert aggregate["top1_agreement_to_source"] == 1.0
    assert report["per_prompt"] == {
        "absolute_delta_nll_per_token": {
            "p90": pytest.approx(0.0),
            "worst": pytest.approx(0.0),
        },
        "top1_agreement_to_source": {"p10": 1.0, "worst": 1.0},
    }
    family_summary = report["family_summary"]
    assert [
        row["family_id"] for row in family_summary["families"]
    ] == ["family-a", "family-b"]
    assert family_summary["macro"]["absolute_delta_nll_per_token"] == 0.0
    assert family_summary["worst"]["top1_agreement_to_source"] == 1.0
    assert report["gates"] == {
        "absolute_delta_nll_per_token": True,
        "aggregate_top1_agreement": True,
        "source_to_candidate_kl_per_token": True,
        "per_prompt_p90_absolute_delta_nll": True,
        "per_prompt_p10_top1_agreement": True,
        "passed": True,
    }


def test_metrics_are_token_weighted_and_family_summaries_are_macro_and_worst() -> None:
    examples = [
        _example(
            "a.short",
            "family-a",
            [[3.0, 0.0]],
            [[0.0, 3.0]],
            [0],
        ),
        _example(
            "b.long",
            "family-b",
            [[2.0, 0.0], [0.0, 2.0], [1.5, -0.5]],
            [[1.5, 0.5], [0.0, 2.0], [1.0, 0.0]],
            [0, 1, 0],
        ),
    ]
    report = evaluate_source_authoritative_shadow(
        examples,
        expected_family_by_example={
            "a.short": "family-a",
            "b.long": "family-b",
        },
        gates=ShadowFidelityGates(
            absolute_delta_nll_per_token_max=10.0,
            top1_agreement_to_source_min=0.0,
            source_to_candidate_kl_per_token_max=10.0,
            per_prompt_p90_absolute_delta_nll_per_token_max=10.0,
            per_prompt_p10_top1_agreement_to_source_min=0.0,
        ),
        vocab_chunk_size=1,
    )

    source = torch.cat([example.source_logits for example in examples])
    candidate = torch.cat([example.candidate_logits for example in examples])
    targets = torch.cat([example.targets for example in examples])
    expected_source_nll = float(
        F.cross_entropy(source, targets, reduction="sum").item()
    )
    expected_candidate_nll = float(
        F.cross_entropy(candidate, targets, reduction="sum").item()
    )
    source_log_probability = F.log_softmax(source, dim=-1)
    candidate_log_probability = F.log_softmax(candidate, dim=-1)
    expected_kl = float(
        (
            source_log_probability.exp()
            * (source_log_probability - candidate_log_probability)
        )
        .sum()
        .item()
    )
    aggregate = report["aggregate"]
    assert aggregate["source_summed_nll"] == pytest.approx(expected_source_nll)
    assert aggregate["candidate_summed_nll"] == pytest.approx(
        expected_candidate_nll
    )
    assert aggregate["delta_nll_per_token"] == pytest.approx(
        (expected_candidate_nll - expected_source_nll) / 4
    )
    assert aggregate["source_to_candidate_kl_per_token"] == pytest.approx(
        expected_kl / 4
    )
    assert aggregate["top1_agreement_to_source"] == 3 / 4

    families = report["family_summary"]["families"]
    assert families[0]["example_count"] == 1
    assert families[0]["supervised_tokens"] == 1
    assert families[0]["top1_agreement_to_source"] == 0.0
    assert families[1]["supervised_tokens"] == 3
    assert families[1]["top1_agreement_to_source"] == 1.0
    macro = report["family_summary"]["macro"]
    assert macro["top1_agreement_to_source"] == 0.5
    assert macro["delta_nll_per_token"] == pytest.approx(
        (
            families[0]["delta_nll_per_token"]
            + families[1]["delta_nll_per_token"]
        )
        / 2
    )
    worst = report["family_summary"]["worst"]
    assert worst["top1_agreement_to_source"] == 0.0
    assert worst["top1_agreement_to_source_family_ids"] == ["family-a"]
    assert worst["absolute_delta_nll_per_token"] == pytest.approx(
        max(row["absolute_delta_nll_per_token"] for row in families)
    )
    assert report["per_prompt"]["absolute_delta_nll_per_token"]["p90"] == (
        pytest.approx(
            max(abs(row["delta_nll_per_token"]) for row in families)
        )
    )
    assert report["per_prompt"]["top1_agreement_to_source"]["p10"] == 0.0
    assert report["gates"]["passed"] is True


def test_established_gate_thresholds_are_inclusive_and_joint() -> None:
    gates = ESTABLISHED_SHADOW_FIDELITY_GATES
    assert gates.metadata() == {
        "absolute_delta_nll_per_token_max": 0.05,
        "top1_agreement_to_source_min": 0.95,
        "source_to_candidate_kl_per_token_max": 0.05,
        "per_prompt_p90_absolute_delta_nll_per_token_max": 0.10,
        "per_prompt_p10_top1_agreement_to_source_min": 0.90,
    }
    boundary = gates.evaluate(
        delta_nll_per_token=-0.05,
        top1_agreement_to_source=0.95,
        source_to_candidate_kl_per_token=0.05,
        per_prompt_p90_absolute_delta_nll_per_token=0.10,
        per_prompt_p10_top1_agreement_to_source=0.90,
    )
    assert boundary["passed"] is True

    failed = gates.evaluate(
        delta_nll_per_token=math.nextafter(0.05, math.inf),
        top1_agreement_to_source=math.nextafter(0.95, -math.inf),
        source_to_candidate_kl_per_token=math.nextafter(0.05, math.inf),
        per_prompt_p90_absolute_delta_nll_per_token=math.nextafter(
            0.10,
            math.inf,
        ),
        per_prompt_p10_top1_agreement_to_source=math.nextafter(
            0.90,
            -math.inf,
        ),
    )
    assert failed == {
        "absolute_delta_nll_per_token": False,
        "aggregate_top1_agreement": False,
        "source_to_candidate_kl_per_token": False,
        "per_prompt_p90_absolute_delta_nll": False,
        "per_prompt_p10_top1_agreement": False,
        "passed": False,
    }


def test_manifest_rejects_undeclared_relabelled_duplicate_and_missing_examples() -> None:
    manifest = {"a": "family-a", "b": "family-b"}
    accumulator = SourceAuthoritativeShadowFidelityAccumulator(manifest)
    with pytest.raises(ValueError, match="undeclared"):
        accumulator.add(_example("c", "family-c", [[1.0, 0.0]]))
    with pytest.raises(ValueError, match="belongs to family"):
        accumulator.add(_example("a", "family-b", [[1.0, 0.0]]))

    accumulator.add(_example("a", "family-a", [[1.0, 0.0]]))
    with pytest.raises(ValueError, match="duplicate"):
        accumulator.add(_example("a", "family-a", [[1.0, 0.0]]))
    with pytest.raises(ValueError, match=r"missing examples: b"):
        accumulator.finalize()


@pytest.mark.parametrize(
    ("example", "message"),
    [
        (
            ShadowFidelityExample(
                "a",
                "family-a",
                torch.ones(2, 3),
                torch.ones(2, 4),
                torch.tensor([0, 1]),
            ),
            "equal shape",
        ),
        (
            ShadowFidelityExample(
                "a",
                "family-a",
                torch.tensor([[float("nan"), 0.0]]),
                torch.ones(1, 2),
                torch.tensor([0]),
            ),
            "source_logits must be finite",
        ),
        (
            ShadowFidelityExample(
                "a",
                "family-a",
                torch.ones(1, 2),
                torch.ones(1, 2),
                torch.tensor([2]),
            ),
            "index the logits vocabulary",
        ),
        (
            ShadowFidelityExample(
                "a",
                "family-a",
                torch.ones(1, 2),
                torch.ones(1, 2),
                torch.tensor([0.0]),
            ),
            "integer dtype",
        ),
    ],
)
def test_invalid_supervised_tensors_fail_closed(
    example: ShadowFidelityExample,
    message: str,
) -> None:
    accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        {"a": "family-a"}
    )
    with pytest.raises((TypeError, ValueError), match=message):
        accumulator.add(example)


def test_mapping_input_and_insertion_order_are_deterministic() -> None:
    first = _example("a", "family-a", [[1.0, 0.0]])
    second = _example("b", "family-b", [[0.0, 1.0]], [[0.25, 0.75]])
    manifest = {"a": "family-a", "b": "family-b"}
    forward = evaluate_source_authoritative_shadow(
        [first, second],
        expected_family_by_example=manifest,
    )
    reverse = evaluate_source_authoritative_shadow(
        [
            {
                "example_id": second.example_id,
                "family_id": second.family_id,
                "source_logits": second.source_logits,
                "candidate_logits": second.candidate_logits,
                "targets": second.targets,
            },
            first,
        ],
        expected_family_by_example=dict(reversed(list(manifest.items()))),
    )
    assert forward == reverse
