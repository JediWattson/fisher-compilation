from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_PANEL = Path("examples/gemma3_attention_edge_fidelity_B_v1.json")
_RETROSPECTIVE_A = Path("examples/gemma3_downstream_retention_v1.json")
_PANEL_ID = "gemma3-attention-edge-fidelity-B-v1"
_PANEL_SHA256 = "1d8383f6c158fa2ea7e300c4722c5ae558667321fb091c79694d1e3918365364"
_PANEL_SCHEMA = "fisher_graph.gemma3_downstream_retention_panel"
_FAMILIES = {
    "integer_addition",
    "integer_subtraction",
    "integer_comparison",
    "lexical_antonym",
    "category_completion",
    "world_capitals",
}
_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}
_FACT_ANSWERS = {
    "lexical_antonym": {
        "wet": "dry",
        "early": "late",
        "clean": "dirty",
        "near": "far",
        "alive": "dead",
        "left": "right",
        "noisy": "quiet",
        "push": "pull",
        "enter": "leave",
        "always": "never",
    },
    "category_completion": {
        "sparrow": "bird",
        "trout": "fish",
        "maple": "tree",
        "wrench": "tool",
        "pear": "fruit",
        "iron": "metal",
        "venus": "planet",
        "french": "language",
        "bee": "insect",
        "train": "vehicle",
    },
    "world_capitals": {
        "portugal": "lisbon",
        "greece": "athens",
        "norway": "oslo",
        "sweden": "stockholm",
        "finland": "helsinki",
        "poland": "warsaw",
        "austria": "vienna",
        "peru": "lima",
        "ireland": "dublin",
        "belgium": "brussels",
    },
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_load(path: Path) -> dict[str, Any]:
    raw = json.loads(
        path.read_bytes().decode("utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
    )
    assert isinstance(raw, dict)
    return raw


def _examples(panel: dict[str, Any]) -> list[dict[str, Any]]:
    assert set(panel) == {
        "schema",
        "format_version",
        "panel_id",
        "choice_count",
        "examples",
    }
    assert panel["schema"] == _PANEL_SCHEMA
    assert panel["format_version"] == 1
    assert panel["choice_count"] == 4
    examples = panel["examples"]
    assert isinstance(examples, list)
    for example in examples:
        assert isinstance(example, dict)
        assert set(example) == {
            "example_id",
            "family_id",
            "prompt",
            "choices",
            "correct_choice",
        }
        assert isinstance(example["example_id"], str)
        assert isinstance(example["family_id"], str)
        assert isinstance(example["prompt"], str)
        assert example["prompt"] == example["prompt"].strip()
        assert isinstance(example["choices"], list)
        assert len(example["choices"]) == 4
        assert len(set(example["choices"])) == 4
        assert all(
            isinstance(choice, str) and choice == choice.strip()
            for choice in example["choices"]
        )
        assert type(example["correct_choice"]) is int
        assert 0 <= example["correct_choice"] < 4
    return examples


def _content_key(example: dict[str, Any]) -> object:
    family = example["family_id"]
    prompt = example["prompt"]
    assert isinstance(family, str)
    assert isinstance(prompt, str)
    if family == "integer_addition":
        return tuple(sorted(int(value) for value in re.findall(r"\d+", prompt)))
    if family == "integer_subtraction":
        return tuple(int(value) for value in re.findall(r"\d+", prompt))
    if family == "integer_comparison":
        return tuple(sorted(int(value) for value in re.findall(r"\d+", prompt)))
    if family == "lexical_antonym":
        match = re.search(r"The opposite of ([A-Za-z]+) is$", prompt)
    elif family == "category_completion":
        match = re.search(
            r"category fact: (?:A |An )?([A-Za-z]+) is a kind of$",
            prompt,
        )
    elif family == "world_capitals":
        match = re.search(r"The capital of ([A-Za-z]+) is$", prompt)
    else:  # pragma: no cover - the frozen family catalog excludes this branch.
        raise AssertionError(f"unexpected family: {family}")
    assert match is not None
    return match.group(1).casefold()


def _expected_answer(example: dict[str, Any]) -> str:
    family = example["family_id"]
    prompt = example["prompt"]
    assert isinstance(family, str)
    assert isinstance(prompt, str)
    numbers = tuple(int(value) for value in re.findall(r"\d+", prompt))
    if family == "integer_addition":
        return _NUMBER_WORDS[numbers[0] + numbers[1]]
    if family == "integer_subtraction":
        return _NUMBER_WORDS[numbers[0] - numbers[1]]
    if family == "integer_comparison":
        return _NUMBER_WORDS[max(numbers)]
    content = _content_key(example)
    assert isinstance(content, str)
    return _FACT_ANSWERS[family][content]


def test_attention_edge_fidelity_b_is_frozen_balanced_and_strict() -> None:
    assert hashlib.sha256(_PANEL.read_bytes()).hexdigest() == _PANEL_SHA256

    raw = _strict_load(_PANEL)
    examples = _examples(raw)

    assert raw["panel_id"] == _PANEL_ID
    assert len(examples) == 60
    assert Counter(example["family_id"] for example in examples) == {
        family: 10 for family in _FAMILIES
    }
    assert len({example["example_id"] for example in examples}) == 60
    assert all(
        example["example_id"].startswith("fidelity-b-") for example in examples
    )
    assert len({example["prompt"] for example in examples}) == 60
    assert Counter(example["correct_choice"] for example in examples) == {
        0: 15,
        1: 15,
        2: 15,
        3: 15,
    }
    assert all(
        re.fullmatch(
            r"[A-Za-z]+",
            example["choices"][example["correct_choice"]],
        )
        for example in examples
    )
    assert all(
        example["choices"][example["correct_choice"]].casefold()
        == _expected_answer(example)
        for example in examples
    )


def test_attention_edge_fidelity_b_content_is_disjoint_from_retrospective_a() -> None:
    retrospective_a = _examples(_strict_load(_RETROSPECTIVE_A))
    fidelity_b = _examples(_strict_load(_PANEL))

    assert {example["example_id"] for example in retrospective_a}.isdisjoint(
        example["example_id"] for example in fidelity_b
    )
    assert {example["prompt"] for example in retrospective_a}.isdisjoint(
        example["prompt"] for example in fidelity_b
    )
    for family in _FAMILIES:
        a_content = {
            _content_key(example)
            for example in retrospective_a
            if example["family_id"] == family
        }
        b_content = {
            _content_key(example)
            for example in fidelity_b
            if example["family_id"] == family
        }
        assert len(a_content) == 10
        assert len(b_content) == 10
        assert a_content.isdisjoint(b_content)
