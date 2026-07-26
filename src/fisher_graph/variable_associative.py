"""Variable-layout associative recall for conditional-compute experiments.

The fixed associative-recall task has one eight-token layout, so valid
sequence length, query position, and token role are perfectly confounded.
This module keeps the same two-pair lookup semantics while rendering every
semantic context through controlled layout variants:

* valid lengths span eight through twelve tokens;
* examples are right padded and carry an explicit attention mask;
* both queried keys and both pair presentation orders are present;
* neutral fillers occur both before and after the supervised answer marker;
* all renderings of one key/value mapping stay in the same data split.

The suffix-filler layouts are intentional.  They create examples with an
identical causal prefix through the supervised position but different valid
lengths.  This makes sequence length distinct from supervised position and
provides direct fixtures for future-invariance checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from itertools import combinations, permutations
from typing import Iterator

import torch
from torch import Tensor

from .compiler.calibration import CalibrationBatch
from .config import TransformerConfig


_CONTEXT_HASH_DOMAIN = b"fisher_graph.variable_associative.context.v1\0"
_EXAMPLE_HASH_DOMAIN = b"fisher_graph.variable_associative.example.v1\0"
_PREFIX_HASH_DOMAIN = b"fisher_graph.variable_associative.prefix.v1\0"
_CONFIG_HASH_DOMAIN = b"fisher_graph.variable_associative.config.v1\0"
_DATASET_HASH_DOMAIN = b"fisher_graph.variable_associative.dataset.v1\0"


def _portable_sha256(domain: bytes, value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VariableAssociativeLayout:
    """One deterministic placement of neutral fillers around the task roles."""

    name: str
    prefix_fillers: int = 0
    between_pair_fillers: int = 0
    pre_query_fillers: int = 0
    suffix_fillers: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("layout name must be a nonempty string")
        for field_name in (
            "prefix_fillers",
            "between_pair_fillers",
            "pre_query_fillers",
            "suffix_fillers",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"layout {field_name} must be a nonnegative integer"
                )

    @property
    def pre_answer_fillers(self) -> int:
        return (
            self.prefix_fillers
            + self.between_pair_fillers
            + self.pre_query_fillers
        )

    @property
    def filler_count(self) -> int:
        return self.pre_answer_fillers + self.suffix_fillers

    @property
    def valid_length(self) -> int:
        return 8 + self.filler_count

    @property
    def supervised_position(self) -> int:
        return 7 + self.pre_answer_fillers

    @property
    def causal_layout_key(self) -> tuple[int, int, int]:
        """Return the placement signature visible through the answer marker."""

        return (
            self.prefix_fillers,
            self.between_pair_fillers,
            self.pre_query_fillers,
        )


DEFAULT_VARIABLE_ASSOCIATIVE_LAYOUTS = (
    VariableAssociativeLayout("compact"),
    VariableAssociativeLayout("suffix_1", suffix_fillers=1),
    VariableAssociativeLayout("suffix_2", suffix_fillers=2),
    VariableAssociativeLayout("prefix_1", prefix_fillers=1),
    VariableAssociativeLayout(
        "prefix_1_suffix_2",
        prefix_fillers=1,
        suffix_fillers=2,
    ),
    VariableAssociativeLayout(
        "pair_gap_pre_query_suffix",
        between_pair_fillers=1,
        pre_query_fillers=1,
        suffix_fillers=1,
    ),
    VariableAssociativeLayout(
        "prefix_pre_query_suffix_2",
        prefix_fillers=1,
        pre_query_fillers=1,
        suffix_fillers=2,
    ),
    VariableAssociativeLayout(
        "distributed_4",
        prefix_fillers=1,
        between_pair_fillers=1,
        pre_query_fillers=2,
    ),
)


class VariableAssociativeTokenRole(IntEnum):
    """Stable role IDs for token-role and future-position controls."""

    PAD = 0
    BOS = 1
    KEY = 2
    VALUE = 3
    FILLER = 4
    QUERY_MARKER = 5
    QUERY_KEY = 6
    ANSWER_MARKER = 7


@dataclass(frozen=True, slots=True)
class VariableAssociativeRecallTaskConfig:
    """Vocabulary, layouts, and semantic-context split configuration."""

    n_keys: int = 8
    n_values: int = 8
    n_filler_tokens: int = 4
    split_seed: int = 26_071
    train_fraction: float = 0.8
    ignore_index: int = -100
    layouts: tuple[VariableAssociativeLayout, ...] = (
        DEFAULT_VARIABLE_ASSOCIATIVE_LAYOUTS
    )

    def __post_init__(self) -> None:
        if type(self.n_keys) is not int or self.n_keys < 2:
            raise ValueError("n_keys must be at least 2")
        if type(self.n_values) is not int or self.n_values < 2:
            raise ValueError("n_values must be at least 2")
        if type(self.n_filler_tokens) is not int or self.n_filler_tokens < 1:
            raise ValueError("n_filler_tokens must be positive")
        if type(self.split_seed) is not int:
            raise ValueError("split_seed must be an integer")
        if not isinstance(self.train_fraction, float) or not (
            0.0 < self.train_fraction < 1.0
        ):
            raise ValueError("train_fraction must be strictly between 0 and 1")
        if type(self.ignore_index) is not int:
            raise ValueError("ignore_index must be an integer")
        if 0 <= self.ignore_index < self.vocab_size:
            raise ValueError("ignore_index must not be a task token ID")
        if (
            type(self.layouts) is not tuple
            or len(self.layouts) < 2
            or any(
                not isinstance(layout, VariableAssociativeLayout)
                for layout in self.layouts
            )
        ):
            raise ValueError("layouts must contain at least two layout specs")
        names = tuple(layout.name for layout in self.layouts)
        if len(set(names)) != len(names):
            raise ValueError("layout names must be unique")
        lengths = tuple(layout.valid_length for layout in self.layouts)
        if min(lengths) < 8 or max(lengths) > 12:
            raise ValueError("layout valid lengths must stay between 8 and 12")
        if set(lengths) != set(range(8, 13)):
            raise ValueError("layouts must cover every valid length from 8 to 12")

        positions_by_length: dict[int, set[int]] = {}
        lengths_by_position: dict[int, set[int]] = {}
        for layout in self.layouts:
            positions_by_length.setdefault(layout.valid_length, set()).add(
                layout.supervised_position
            )
            lengths_by_position.setdefault(
                layout.supervised_position, set()
            ).add(layout.valid_length)
        if not any(len(values) > 1 for values in positions_by_length.values()):
            raise ValueError(
                "at least one valid length must have multiple supervised positions"
            )
        if not any(len(values) > 1 for values in lengths_by_position.values()):
            raise ValueError(
                "at least one supervised position must occur at multiple lengths"
            )

        suffixes_by_prefix: dict[tuple[int, int, int], set[int]] = {}
        for layout in self.layouts:
            suffixes_by_prefix.setdefault(layout.causal_layout_key, set()).add(
                layout.suffix_fillers
            )
        if not any(len(values) > 1 for values in suffixes_by_prefix.values()):
            raise ValueError(
                "layouts must include causal-prefix-matched suffix variants"
            )

    @property
    def value_offset(self) -> int:
        return self.n_keys

    @property
    def bos_token_id(self) -> int:
        return self.n_keys + self.n_values

    @property
    def query_token_id(self) -> int:
        return self.bos_token_id + 1

    @property
    def answer_token_id(self) -> int:
        return self.bos_token_id + 2

    @property
    def pad_token_id(self) -> int:
        return self.bos_token_id + 3

    @property
    def filler_offset(self) -> int:
        return self.bos_token_id + 4

    @property
    def vocab_size(self) -> int:
        return self.n_keys + self.n_values + 4 + self.n_filler_tokens

    @property
    def maximum_sequence_length(self) -> int:
        return max(layout.valid_length for layout in self.layouts)

    @property
    def minimum_sequence_length(self) -> int:
        return min(layout.valid_length for layout in self.layouts)

    @property
    def semantic_context_count(self) -> int:
        return (
            math.comb(self.n_keys, 2)
            * self.n_values
            * (self.n_values - 1)
        )

    @property
    def variants_per_context(self) -> int:
        return 2 * 2 * len(self.layouts)

    @property
    def fingerprint(self) -> str:
        return _portable_sha256(
            _CONFIG_HASH_DOMAIN,
            {
                **asdict(self),
                "vocabulary": {
                    "value_offset": self.value_offset,
                    "bos_token_id": self.bos_token_id,
                    "query_token_id": self.query_token_id,
                    "answer_token_id": self.answer_token_id,
                    "pad_token_id": self.pad_token_id,
                    "filler_offset": self.filler_offset,
                    "vocab_size": self.vocab_size,
                },
            },
        )


@dataclass(frozen=True, slots=True)
class VariableAssociativeTokenMetadata:
    """Row-major metadata aligned with all valid token activation rows."""

    selected_flat_indices: Tensor
    example_indices: Tensor
    semantic_context_indices: Tensor
    logical_positions: Tensor
    valid_lengths: Tensor
    token_ids: Tensor
    token_role_ids: Tensor
    supervised_positions: Tensor
    is_supervised: Tensor
    is_future: Tensor
    query_slots: Tensor
    pair_orders: Tensor
    rendered_query_slots: Tensor
    layout_indices: Tensor

    @property
    def observations(self) -> int:
        return self.selected_flat_indices.numel()


@dataclass(frozen=True, slots=True)
class VariableAssociativeRecallSplit:
    """One semantic-context-grouped variable-layout data partition."""

    name: str
    input_ids: Tensor
    attention_mask: Tensor
    targets: Tensor
    token_role_ids: Tensor
    supervised_positions: Tensor
    valid_lengths: Tensor
    query_slots: Tensor
    pair_orders: Tensor
    rendered_query_slots: Tensor
    layout_indices: Tensor
    queried_key_ids: Tensor
    answer_value_indices: Tensor
    example_context_indices: Tensor
    semantic_contexts: Tensor
    context_ids: Tensor
    semantic_context_hashes: tuple[str, ...]
    example_ids: tuple[str, ...]
    example_hashes: tuple[str, ...]
    causal_prefix_hashes: tuple[str, ...]
    n_values: int
    ignore_index: int
    pad_token_id: int
    content_sha256: str

    @property
    def samples(self) -> int:
        return self.input_ids.shape[0]

    @property
    def contexts(self) -> int:
        return self.context_ids.shape[0]

    @property
    def maximum_sequence_length(self) -> int:
        return self.input_ids.shape[1]

    @property
    def answer_token_ids(self) -> Tensor:
        rows = torch.arange(self.samples)
        return self.targets[rows, self.supervised_positions]

    def valid_token_metadata(self) -> VariableAssociativeTokenMetadata:
        """Flatten valid-token controls in activation-collection row order."""

        flat_valid = self.attention_mask.reshape(-1)
        selected = flat_valid.nonzero(as_tuple=False).flatten()
        sequence_length = self.maximum_sequence_length
        examples = (
            torch.arange(self.samples)
            .unsqueeze(1)
            .expand(-1, sequence_length)
            .reshape(-1)[flat_valid]
        )
        positions = (
            torch.arange(sequence_length)
            .unsqueeze(0)
            .expand(self.samples, -1)
            .reshape(-1)[flat_valid]
        )

        def repeat_rows(values: Tensor) -> Tensor:
            return (
                values.unsqueeze(1)
                .expand(-1, sequence_length)
                .reshape(-1)[flat_valid]
            )

        supervised = repeat_rows(self.supervised_positions)
        return VariableAssociativeTokenMetadata(
            selected_flat_indices=selected,
            example_indices=examples,
            semantic_context_indices=repeat_rows(
                self.example_context_indices
            ),
            logical_positions=positions,
            valid_lengths=repeat_rows(self.valid_lengths),
            token_ids=self.input_ids.reshape(-1)[flat_valid],
            token_role_ids=self.token_role_ids.reshape(-1)[flat_valid],
            supervised_positions=supervised,
            is_supervised=positions == supervised,
            is_future=positions > supervised,
            query_slots=repeat_rows(self.query_slots),
            pair_orders=repeat_rows(self.pair_orders),
            rendered_query_slots=repeat_rows(self.rendered_query_slots),
            layout_indices=repeat_rows(self.layout_indices),
        )


@dataclass(frozen=True, slots=True)
class VariableAssociativeRecallSplits:
    """Train, validation, and untouched test semantic-context partitions."""

    task_config: VariableAssociativeRecallTaskConfig
    train: VariableAssociativeRecallSplit
    validation: VariableAssociativeRecallSplit
    test: VariableAssociativeRecallSplit
    dataset_sha256: str


def variable_associative_recall_model_config(
    task_config: VariableAssociativeRecallTaskConfig | None = None,
) -> TransformerConfig:
    """Return an instrumentable toy model sized for the variable task."""

    task = task_config or VariableAssociativeRecallTaskConfig()
    return TransformerConfig(
        vocab_size=task.vocab_size,
        max_sequence_length=task.maximum_sequence_length,
        d_model=32,
        n_heads=4,
        n_layers=3,
        d_ff=64,
        dropout=0.0,
        tie_embeddings=False,
    )


def _enumerate_semantic_contexts(
    config: VariableAssociativeRecallTaskConfig,
) -> Tensor:
    contexts = [
        (key0, key1, value0, value1)
        for key0, key1 in combinations(range(config.n_keys), 2)
        for value0, value1 in permutations(range(config.n_values), 2)
    ]
    return torch.tensor(contexts, dtype=torch.long)


def _semantic_context_hash(context: tuple[int, int, int, int]) -> str:
    key0, key1, value0, value1 = context
    return _portable_sha256(
        _CONTEXT_HASH_DOMAIN,
        {
            "mapping": (
                {"key": key0, "value_index": value0},
                {"key": key1, "value_index": value1},
            )
        },
    )


def _filler_tokens(
    config: VariableAssociativeRecallTaskConfig,
    *,
    region: int,
    count: int,
) -> list[int]:
    return [
        config.filler_offset
        + ((region * 2 + ordinal) % config.n_filler_tokens)
        for ordinal in range(count)
    ]


def _extend_with_fillers(
    tokens: list[int],
    roles: list[int],
    config: VariableAssociativeRecallTaskConfig,
    *,
    region: int,
    count: int,
) -> None:
    fillers = _filler_tokens(config, region=region, count=count)
    tokens.extend(fillers)
    roles.extend([int(VariableAssociativeTokenRole.FILLER)] * count)


def _render_example(
    *,
    context: tuple[int, int, int, int],
    query_slot: int,
    pair_order: int,
    layout: VariableAssociativeLayout,
    config: VariableAssociativeRecallTaskConfig,
) -> tuple[list[int], list[int], int, int, int]:
    key0, key1, value0, value1 = context
    keys = (key0, key1)
    values = (value0, value1)
    rendered_slots = (0, 1) if pair_order == 0 else (1, 0)
    rendered_query_slot = rendered_slots.index(query_slot)

    tokens = [config.bos_token_id]
    roles = [int(VariableAssociativeTokenRole.BOS)]
    _extend_with_fillers(
        tokens,
        roles,
        config,
        region=0,
        count=layout.prefix_fillers,
    )
    for rendered_index, semantic_slot in enumerate(rendered_slots):
        tokens.extend(
            (
                keys[semantic_slot],
                config.value_offset + values[semantic_slot],
            )
        )
        roles.extend(
            (
                int(VariableAssociativeTokenRole.KEY),
                int(VariableAssociativeTokenRole.VALUE),
            )
        )
        if rendered_index == 0:
            _extend_with_fillers(
                tokens,
                roles,
                config,
                region=1,
                count=layout.between_pair_fillers,
            )
    _extend_with_fillers(
        tokens,
        roles,
        config,
        region=2,
        count=layout.pre_query_fillers,
    )
    tokens.extend(
        (
            config.query_token_id,
            keys[query_slot],
            config.answer_token_id,
        )
    )
    roles.extend(
        (
            int(VariableAssociativeTokenRole.QUERY_MARKER),
            int(VariableAssociativeTokenRole.QUERY_KEY),
            int(VariableAssociativeTokenRole.ANSWER_MARKER),
        )
    )
    supervised_position = len(tokens) - 1
    _extend_with_fillers(
        tokens,
        roles,
        config,
        region=3,
        count=layout.suffix_fillers,
    )
    if len(tokens) != layout.valid_length:
        raise RuntimeError("rendered layout length disagrees with its contract")
    return (
        tokens,
        roles,
        supervised_position,
        rendered_query_slot,
        values[query_slot],
    )


def _make_split(
    *,
    name: str,
    all_contexts: Tensor,
    context_ids: Tensor,
    config: VariableAssociativeRecallTaskConfig,
) -> VariableAssociativeRecallSplit:
    selected_contexts = all_contexts.index_select(0, context_ids)
    context_count = selected_contexts.shape[0]
    sample_count = context_count * config.variants_per_context
    maximum_length = config.maximum_sequence_length

    input_ids = torch.full(
        (sample_count, maximum_length),
        config.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (sample_count, maximum_length),
        dtype=torch.bool,
    )
    targets = torch.full_like(input_ids, config.ignore_index)
    token_role_ids = torch.full_like(
        input_ids,
        int(VariableAssociativeTokenRole.PAD),
    )

    supervised_positions = torch.empty(sample_count, dtype=torch.long)
    valid_lengths = torch.empty(sample_count, dtype=torch.long)
    query_slots = torch.empty(sample_count, dtype=torch.long)
    pair_orders = torch.empty(sample_count, dtype=torch.long)
    rendered_query_slots = torch.empty(sample_count, dtype=torch.long)
    layout_indices = torch.empty(sample_count, dtype=torch.long)
    queried_key_ids = torch.empty(sample_count, dtype=torch.long)
    answer_value_indices = torch.empty(sample_count, dtype=torch.long)
    example_context_indices = torch.empty(sample_count, dtype=torch.long)

    semantic_context_hashes = tuple(
        _semantic_context_hash(tuple(int(value) for value in context.tolist()))
        for context in selected_contexts
    )
    example_ids: list[str] = []
    example_hashes: list[str] = []
    causal_prefix_hashes: list[str] = []

    row = 0
    for local_context_index, context_tensor in enumerate(selected_contexts):
        context = tuple(int(value) for value in context_tensor.tolist())
        context_hash = semantic_context_hashes[local_context_index]
        for query_slot in range(2):
            for pair_order in range(2):
                for layout_index, layout in enumerate(config.layouts):
                    (
                        tokens,
                        roles,
                        supervised_position,
                        rendered_query_slot,
                        answer_value_index,
                    ) = _render_example(
                        context=context,
                        query_slot=query_slot,
                        pair_order=pair_order,
                        layout=layout,
                        config=config,
                    )
                    valid_length = len(tokens)
                    input_ids[row, :valid_length] = torch.tensor(tokens)
                    attention_mask[row, :valid_length] = True
                    token_role_ids[row, :valid_length] = torch.tensor(roles)
                    targets[
                        row, supervised_position
                    ] = config.value_offset + answer_value_index
                    supervised_positions[row] = supervised_position
                    valid_lengths[row] = valid_length
                    query_slots[row] = query_slot
                    pair_orders[row] = pair_order
                    rendered_query_slots[row] = rendered_query_slot
                    layout_indices[row] = layout_index
                    queried_key_ids[row] = context[query_slot]
                    answer_value_indices[row] = answer_value_index
                    example_context_indices[row] = local_context_index

                    example_id = (
                        f"ctx-{context_hash[:16]}.q{query_slot}."
                        f"o{pair_order}.l-{layout.name}"
                    )
                    example_payload = {
                        "semantic_context_sha256": context_hash,
                        "query_slot": query_slot,
                        "pair_order": pair_order,
                        "layout": asdict(layout),
                        "input_ids": tokens,
                        "target_position": supervised_position,
                        "target_token_id": (
                            config.value_offset + answer_value_index
                        ),
                    }
                    prefix_payload = {
                        "semantic_context_sha256": context_hash,
                        "query_slot": query_slot,
                        "pair_order": pair_order,
                        "prefix_input_ids": tokens[
                            : supervised_position + 1
                        ],
                    }
                    example_ids.append(example_id)
                    example_hashes.append(
                        _portable_sha256(
                            _EXAMPLE_HASH_DOMAIN,
                            example_payload,
                        )
                    )
                    causal_prefix_hashes.append(
                        _portable_sha256(
                            _PREFIX_HASH_DOMAIN,
                            prefix_payload,
                        )
                    )
                    row += 1

    if row != sample_count:
        raise RuntimeError("variable associative split rendered the wrong size")
    content_sha256 = _portable_sha256(
        _DATASET_HASH_DOMAIN,
        {
            "name": name,
            "task_config_sha256": config.fingerprint,
            "context_ids": context_ids.tolist(),
            "semantic_context_hashes": semantic_context_hashes,
            "example_hashes": example_hashes,
        },
    )
    return VariableAssociativeRecallSplit(
        name=name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        targets=targets,
        token_role_ids=token_role_ids,
        supervised_positions=supervised_positions,
        valid_lengths=valid_lengths,
        query_slots=query_slots,
        pair_orders=pair_orders,
        rendered_query_slots=rendered_query_slots,
        layout_indices=layout_indices,
        queried_key_ids=queried_key_ids,
        answer_value_indices=answer_value_indices,
        example_context_indices=example_context_indices,
        semantic_contexts=selected_contexts.clone(),
        context_ids=context_ids.clone(),
        semantic_context_hashes=semantic_context_hashes,
        example_ids=tuple(example_ids),
        example_hashes=tuple(example_hashes),
        causal_prefix_hashes=tuple(causal_prefix_hashes),
        n_values=config.n_values,
        ignore_index=config.ignore_index,
        pad_token_id=config.pad_token_id,
        content_sha256=content_sha256,
    )


def build_variable_associative_recall_splits(
    config: VariableAssociativeRecallTaskConfig | None = None,
) -> VariableAssociativeRecallSplits:
    """Build deterministic splits grouped by canonical key/value mapping."""

    task = config or VariableAssociativeRecallTaskConfig()
    contexts = _enumerate_semantic_contexts(task)
    generator = torch.Generator(device="cpu").manual_seed(task.split_seed)
    shuffled_ids = torch.randperm(contexts.shape[0], generator=generator)

    train_contexts = int(contexts.shape[0] * task.train_fraction)
    remaining = contexts.shape[0] - train_contexts
    validation_contexts = remaining // 2
    if train_contexts == 0 or validation_contexts == 0:
        raise ValueError("the configured task is too small for three nonempty splits")
    train_end = train_contexts
    validation_end = train_end + validation_contexts

    train = _make_split(
        name="train",
        all_contexts=contexts,
        context_ids=shuffled_ids[:train_end],
        config=task,
    )
    validation = _make_split(
        name="validation",
        all_contexts=contexts,
        context_ids=shuffled_ids[train_end:validation_end],
        config=task,
    )
    test = _make_split(
        name="test",
        all_contexts=contexts,
        context_ids=shuffled_ids[validation_end:],
        config=task,
    )
    dataset_sha256 = _portable_sha256(
        _DATASET_HASH_DOMAIN,
        {
            "task_config_sha256": task.fingerprint,
            "splits": {
                "train": train.content_sha256,
                "validation": validation.content_sha256,
                "test": test.content_sha256,
            },
        },
    )
    return VariableAssociativeRecallSplits(
        task_config=task,
        train=train,
        validation=validation,
        test=test,
        dataset_sha256=dataset_sha256,
    )


def subset_variable_associative_recall_split(
    split: VariableAssociativeRecallSplit,
    *,
    context_rows: Tensor,
    name: str,
) -> VariableAssociativeRecallSplit:
    """Select complete semantic contexts and remap local context row IDs."""

    if not isinstance(split, VariableAssociativeRecallSplit):
        raise TypeError("split must be a VariableAssociativeRecallSplit")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    if (
        not isinstance(context_rows, Tensor)
        or context_rows.dtype not in (torch.int32, torch.int64)
        or context_rows.ndim != 1
        or context_rows.numel() == 0
    ):
        raise ValueError("context_rows must be a nonempty integer vector")
    selected_context_rows = context_rows.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    if (
        int(selected_context_rows.min().item()) < 0
        or int(selected_context_rows.max().item()) >= split.contexts
    ):
        raise ValueError("context_rows exceed the source split")
    if selected_context_rows.unique().numel() != selected_context_rows.numel():
        raise ValueError("context_rows must be unique")

    example_chunks = [
        (split.example_context_indices == int(context_row)).nonzero(
            as_tuple=False
        ).flatten()
        for context_row in selected_context_rows.tolist()
    ]
    if any(chunk.numel() == 0 for chunk in example_chunks):
        raise RuntimeError("a selected semantic context has no examples")
    example_rows = torch.cat(example_chunks)
    remapped_contexts = torch.cat(
        [
            torch.full_like(chunk, local_row)
            for local_row, chunk in enumerate(example_chunks)
        ]
    )

    def select_examples(values: Tensor) -> Tensor:
        return values.index_select(0, example_rows)

    selected_context_ids = split.context_ids.index_select(
        0,
        selected_context_rows,
    )
    semantic_hashes = tuple(
        split.semantic_context_hashes[index]
        for index in selected_context_rows.tolist()
    )
    example_ids = tuple(split.example_ids[index] for index in example_rows.tolist())
    example_hashes = tuple(
        split.example_hashes[index] for index in example_rows.tolist()
    )
    causal_prefix_hashes = tuple(
        split.causal_prefix_hashes[index] for index in example_rows.tolist()
    )
    content_sha256 = _portable_sha256(
        _DATASET_HASH_DOMAIN,
        {
            "name": name,
            "source_content_sha256": split.content_sha256,
            "context_ids": selected_context_ids.tolist(),
            "semantic_context_hashes": semantic_hashes,
            "example_hashes": example_hashes,
        },
    )
    return VariableAssociativeRecallSplit(
        name=name,
        input_ids=select_examples(split.input_ids),
        attention_mask=select_examples(split.attention_mask),
        targets=select_examples(split.targets),
        token_role_ids=select_examples(split.token_role_ids),
        supervised_positions=select_examples(split.supervised_positions),
        valid_lengths=select_examples(split.valid_lengths),
        query_slots=select_examples(split.query_slots),
        pair_orders=select_examples(split.pair_orders),
        rendered_query_slots=select_examples(split.rendered_query_slots),
        layout_indices=select_examples(split.layout_indices),
        queried_key_ids=select_examples(split.queried_key_ids),
        answer_value_indices=select_examples(split.answer_value_indices),
        example_context_indices=remapped_contexts,
        semantic_contexts=split.semantic_contexts.index_select(
            0,
            selected_context_rows,
        ),
        context_ids=selected_context_ids,
        semantic_context_hashes=semantic_hashes,
        example_ids=example_ids,
        example_hashes=example_hashes,
        causal_prefix_hashes=causal_prefix_hashes,
        n_values=split.n_values,
        ignore_index=split.ignore_index,
        pad_token_id=split.pad_token_id,
        content_sha256=content_sha256,
    )


def iter_variable_associative_calibration_batches(
    split: VariableAssociativeRecallSplit,
    *,
    batch_size: int,
) -> Iterator[CalibrationBatch]:
    """Yield replayable compiler batches with stable per-example identities."""

    if not isinstance(split, VariableAssociativeRecallSplit):
        raise TypeError("split must be a VariableAssociativeRecallSplit")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, split.samples, batch_size):
        stop = min(start + batch_size, split.samples)
        yield CalibrationBatch(
            model_inputs={
                "input_ids": split.input_ids[start:stop],
                "attention_mask": split.attention_mask[start:stop],
            },
            targets=split.targets[start:stop],
            valid_positions=split.attention_mask[start:stop],
            example_ids=split.example_ids[start:stop],
        )


__all__ = [
    "DEFAULT_VARIABLE_ASSOCIATIVE_LAYOUTS",
    "VariableAssociativeLayout",
    "VariableAssociativeRecallSplit",
    "VariableAssociativeRecallSplits",
    "VariableAssociativeRecallTaskConfig",
    "VariableAssociativeTokenMetadata",
    "VariableAssociativeTokenRole",
    "build_variable_associative_recall_splits",
    "iter_variable_associative_calibration_batches",
    "subset_variable_associative_recall_split",
    "variable_associative_recall_model_config",
]
