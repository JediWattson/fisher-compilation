"""Pure logical accounting for native and conditional full-span execution.

The functions in this module count mathematical multiply-accumulates (MACs)
and stored scalar coefficients.  They do not estimate kernel launches,
memory traffic, activation functions, softmax, gathers, masking, or wall-clock
latency.

Native transformer span
-----------------------

For ``T`` valid rows, ``P`` allowed causal query/key pairs, residual width
``d``, feed-forward width ``d_ff``, and ``L`` layers, the logical MAC count is

``L * (4*T*d^2 + 2*P*d + 2*T*d*d_ff)``.

The three terms are the four attention projections, the QK and AV pairwise
products, and the two feed-forward projections.  Padded row and pair counts
are accepted separately so padding overhead is never confused with logical
work.

Conditional causal graph
------------------------

The conditional graph assumed here matches a shared exponential-state trunk
with ``C`` state channels and hidden width ``H``, a causal-hidden affine
router with ``R`` routes, and a hard-routed output head plus width-``d``
decoder.  For ``T`` valid key/prefix rows, ``Q`` demanded query rows, and
``A = sum_q active_rank[q]`` active modal applications, its ideal MAC count is

``T*C*d*H + Q*C*H^2 + Q*H*R + A*H + A*d``.

The recurrent decay/update arithmetic is reported separately instead of
quietly treating it as free or folding it into a matrix-MAC count.

``stored_output_modes`` can structurally prune the output head and decoder to
``m <= d`` columns.  A router-free static comparator sets
``include_router=False`` and normally applies the same ``m`` stored columns to
every demanded query, making its tail cost ``Q*m*(H+d)`` without charging
router coefficients or route masks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _padded_count(
    logical: int,
    padded: int | None,
    *,
    logical_name: str,
    padded_name: str,
) -> int:
    if padded is None:
        return logical
    result = _nonnegative_int(padded, name=padded_name)
    if result < logical:
        raise ValueError(f"{padded_name} cannot be smaller than {logical_name}")
    return result


@dataclass(frozen=True, slots=True)
class NativeTransformerSpanAccounting:
    """Logical and padded accounting for a standard pre-norm span.

    Parameter counts match the repository's toy transformer block: biased QKV
    and attention-output linears, two biased feed-forward linears, and two
    LayerNorms with learned scale and bias.  ``linear_weight_parameter_count``
    is also exposed separately for architectures with different bias or norm
    conventions.
    """

    valid_rows: int
    padded_rows: int
    logical_causal_pairs: int
    padded_causal_pairs: int
    width: int
    feed_forward_width: int
    layer_count: int
    qkv_projection_macs: int
    attention_output_projection_macs: int
    attention_score_macs: int
    attention_value_macs: int
    feed_forward_input_macs: int
    feed_forward_output_macs: int
    logical_total_macs: int
    padded_total_macs: int
    linear_weight_parameter_count: int
    affine_bias_parameter_count: int
    normalization_parameter_count: int
    total_parameter_count: int

    @property
    def logical_attention_projection_macs(self) -> int:
        return (
            self.qkv_projection_macs
            + self.attention_output_projection_macs
        )

    @property
    def logical_attention_pair_macs(self) -> int:
        return self.attention_score_macs + self.attention_value_macs

    @property
    def logical_feed_forward_macs(self) -> int:
        return (
            self.feed_forward_input_macs
            + self.feed_forward_output_macs
        )

    @property
    def padding_row_count(self) -> int:
        return self.padded_rows - self.valid_rows

    @property
    def padding_pair_count(self) -> int:
        return self.padded_causal_pairs - self.logical_causal_pairs

    @property
    def padding_mac_overhead(self) -> int:
        return self.padded_total_macs - self.logical_total_macs

    @property
    def logical_to_padded_mac_ratio(self) -> float:
        if self.padded_total_macs == 0:
            return 0.0
        return self.logical_total_macs / self.padded_total_macs


def native_transformer_span_accounting(
    *,
    valid_rows: int,
    causal_pairs: int,
    width: int,
    feed_forward_width: int,
    layer_count: int,
    padded_rows: int | None = None,
    padded_causal_pairs: int | None = None,
) -> NativeTransformerSpanAccounting:
    """Return exact logical-MAC and toy-block parameter formulas.

    Bias applications, layer normalization, residual additions, attention
    scaling, masking, softmax, and activation functions are not MACs.  They
    remain relevant to actual runtime.
    """

    tokens = _nonnegative_int(valid_rows, name="valid_rows")
    pairs = _nonnegative_int(causal_pairs, name="causal_pairs")
    d_model = _nonnegative_int(width, name="width")
    d_ff = _nonnegative_int(
        feed_forward_width,
        name="feed_forward_width",
    )
    layers = _nonnegative_int(layer_count, name="layer_count")
    padded_tokens = _padded_count(
        tokens,
        padded_rows,
        logical_name="valid_rows",
        padded_name="padded_rows",
    )
    padded_pairs = _padded_count(
        pairs,
        padded_causal_pairs,
        logical_name="causal_pairs",
        padded_name="padded_causal_pairs",
    )

    qkv = layers * 3 * tokens * d_model * d_model
    attention_output = layers * tokens * d_model * d_model
    attention_scores = layers * pairs * d_model
    attention_values = layers * pairs * d_model
    feed_forward_input = layers * tokens * d_model * d_ff
    feed_forward_output = layers * tokens * d_ff * d_model
    logical_total = (
        qkv
        + attention_output
        + attention_scores
        + attention_values
        + feed_forward_input
        + feed_forward_output
    )
    padded_total = layers * (
        4 * padded_tokens * d_model * d_model
        + 2 * padded_pairs * d_model
        + 2 * padded_tokens * d_model * d_ff
    )

    linear_weights = layers * (
        4 * d_model * d_model + 2 * d_model * d_ff
    )
    # QKV + attention output + both MLP biases: 5*d + d_ff.
    affine_biases = layers * (5 * d_model + d_ff)
    # Two LayerNorms, each with a width-d scale and bias.
    normalization = layers * 4 * d_model
    return NativeTransformerSpanAccounting(
        valid_rows=tokens,
        padded_rows=padded_tokens,
        logical_causal_pairs=pairs,
        padded_causal_pairs=padded_pairs,
        width=d_model,
        feed_forward_width=d_ff,
        layer_count=layers,
        qkv_projection_macs=qkv,
        attention_output_projection_macs=attention_output,
        attention_score_macs=attention_scores,
        attention_value_macs=attention_values,
        feed_forward_input_macs=feed_forward_input,
        feed_forward_output_macs=feed_forward_output,
        logical_total_macs=logical_total,
        padded_total_macs=padded_total,
        linear_weight_parameter_count=linear_weights,
        affine_bias_parameter_count=affine_biases,
        normalization_parameter_count=normalization,
        total_parameter_count=(
            linear_weights + affine_biases + normalization
        ),
    )


@dataclass(frozen=True, slots=True)
class ConditionalCausalGraphAccounting:
    """Ideal logical accounting for a source-independent conditional graph.

    The stored-coefficient scope is execution state, not fitting diagnostics.
    It includes the trainable graph, fitted router normalization and affine
    coefficients, fixed output codec, and dense boolean route masks.  It
    excludes Fisher centroids and other analysis-only statistics.
    """

    logical_key_rows: int
    padded_key_rows: int
    routed_query_rows: int
    padded_query_rows: int
    logical_causal_pairs: int | None
    padded_causal_pairs: int | None
    width: int
    state_channels: int
    hidden_width: int
    routes: int
    include_router: bool
    stored_output_modes: int
    active_rank_applications: int
    state_input_macs: int
    hidden_projection_macs: int
    shared_trunk_macs: int
    router_macs: int
    output_head_macs: int
    decoder_macs: int
    route_tail_macs: int
    ideal_total_macs: int
    dense_route_tail_macs: int
    recurrence_decay_argument_multiplications: int
    recurrence_exponential_evaluations: int
    recurrence_state_multiplications: int
    recurrence_state_additions: int
    recurrence_elementwise_arithmetic_ops: int
    state_input_parameter_count: int
    decay_parameter_count: int
    hidden_weight_parameter_count: int
    hidden_bias_parameter_count: int
    shared_trunk_parameter_count: int
    output_head_weight_parameter_count: int
    output_head_bias_parameter_count: int
    graph_trainable_parameter_count: int
    router_weight_coefficient_count: int
    router_bias_coefficient_count: int
    router_normalization_coefficient_count: int
    router_stored_coefficient_count: int
    decoder_coefficient_count: int
    delta_mean_coefficient_count: int
    codec_fixed_coefficient_count: int
    total_learned_coefficient_count: int
    total_floating_runtime_coefficient_count: int
    route_mask_boolean_count: int
    total_runtime_scalar_count: int

    @property
    def average_active_rank(self) -> float:
        if self.routed_query_rows == 0:
            return 0.0
        return self.active_rank_applications / self.routed_query_rows

    @property
    def padding_key_rows(self) -> int:
        return self.padded_key_rows - self.logical_key_rows

    @property
    def padding_query_rows(self) -> int:
        return self.padded_query_rows - self.routed_query_rows

    @property
    def padding_pair_count(self) -> int | None:
        if (
            self.logical_causal_pairs is None
            or self.padded_causal_pairs is None
        ):
            return None
        return self.padded_causal_pairs - self.logical_causal_pairs

    @property
    def route_tail_to_dense_ratio(self) -> float:
        if self.dense_route_tail_macs == 0:
            return 0.0
        return self.route_tail_macs / self.dense_route_tail_macs


def _active_rank_total(
    *,
    routed_query_rows: int,
    stored_output_modes: int,
    active_ranks: Sequence[int] | None,
    active_rank_applications: int | None,
) -> int:
    if active_ranks is None and active_rank_applications is None:
        raise ValueError(
            "provide active_ranks or active_rank_applications"
        )
    ranks_total: int | None = None
    if active_ranks is not None:
        if isinstance(active_ranks, (str, bytes)) or not isinstance(
            active_ranks,
            Sequence,
        ):
            raise TypeError("active_ranks must be an integer sequence")
        ranks = tuple(active_ranks)
        if len(ranks) != routed_query_rows:
            raise ValueError(
                "active_ranks must contain one rank per routed query row"
            )
        if any(
            type(rank) is not int
            or not 0 <= rank <= stored_output_modes
            for rank in ranks
        ):
            raise ValueError(
                "active ranks must be integers between zero and "
                "stored_output_modes"
            )
        ranks_total = sum(ranks)
    if active_rank_applications is not None:
        aggregate = _nonnegative_int(
            active_rank_applications,
            name="active_rank_applications",
        )
        if aggregate > routed_query_rows * stored_output_modes:
            raise ValueError(
                "active_rank_applications exceeds query_rows * "
                "stored_output_modes"
            )
        if ranks_total is not None and aggregate != ranks_total:
            raise ValueError(
                "active rank sequence and aggregate disagree"
            )
        ranks_total = aggregate
    assert ranks_total is not None
    return ranks_total


def conditional_causal_graph_accounting(
    *,
    key_rows: int,
    query_rows: int,
    width: int,
    state_channels: int,
    hidden_width: int,
    routes: int,
    active_ranks: Sequence[int] | None = None,
    active_rank_applications: int | None = None,
    include_router: bool = True,
    stored_output_modes: int | None = None,
    padded_key_rows: int | None = None,
    padded_query_rows: int | None = None,
    logical_causal_pairs: int | None = None,
    padded_causal_pairs: int | None = None,
) -> ConditionalCausalGraphAccounting:
    """Return ideal logical MACs and execution-state coefficient counts.

    The router is applied to the causal hidden width ``H``.  Matrix MACs are
    counted only on logical key/query rows; supplied padded counts are retained
    as provenance and do not inflate the ideal graph.  Pair counts are also
    provenance: the recurrent trunk scales with ``T`` rather than materializing
    ``P`` causal edges.

    Recurrence arithmetic assumes one decay argument/exponential per
    key-row/channel, followed by one state multiplication and one state
    addition per key-row/channel/hidden element.
    """

    keys = _nonnegative_int(key_rows, name="key_rows")
    queries = _nonnegative_int(query_rows, name="query_rows")
    d_model = _nonnegative_int(width, name="width")
    channels = _nonnegative_int(
        state_channels,
        name="state_channels",
    )
    hidden = _nonnegative_int(hidden_width, name="hidden_width")
    route_count = _nonnegative_int(routes, name="routes")
    if type(include_router) is not bool:
        raise ValueError("include_router must be a bool")
    if stored_output_modes is None:
        stored_modes = d_model
    else:
        stored_modes = _nonnegative_int(
            stored_output_modes,
            name="stored_output_modes",
        )
        if stored_modes > d_model:
            raise ValueError(
                "stored_output_modes cannot exceed width"
            )
    padded_keys = _padded_count(
        keys,
        padded_key_rows,
        logical_name="key_rows",
        padded_name="padded_key_rows",
    )
    padded_queries = _padded_count(
        queries,
        padded_query_rows,
        logical_name="query_rows",
        padded_name="padded_query_rows",
    )
    if (logical_causal_pairs is None) != (padded_causal_pairs is None):
        raise ValueError(
            "logical and padded causal pair counts must be supplied together"
        )
    if logical_causal_pairs is None:
        logical_pairs = None
        padded_pairs = None
    else:
        logical_pairs = _nonnegative_int(
            logical_causal_pairs,
            name="logical_causal_pairs",
        )
        padded_pairs = _padded_count(
            logical_pairs,
            padded_causal_pairs,
            logical_name="logical_causal_pairs",
            padded_name="padded_causal_pairs",
        )

    active = _active_rank_total(
        routed_query_rows=queries,
        stored_output_modes=stored_modes,
        active_ranks=active_ranks,
        active_rank_applications=active_rank_applications,
    )

    state_input_macs = keys * channels * d_model * hidden
    hidden_projection_macs = queries * channels * hidden * hidden
    shared_trunk_macs = state_input_macs + hidden_projection_macs
    router_macs = (
        queries * hidden * route_count
        if include_router
        else 0
    )
    output_head_macs = active * hidden
    decoder_macs = active * d_model
    route_tail_macs = output_head_macs + decoder_macs

    decay_arguments = keys * channels
    recurrence_state_elements = keys * channels * hidden
    recurrence_arithmetic = (
        decay_arguments
        + recurrence_state_elements
        + recurrence_state_elements
    )

    state_input_parameters = channels * d_model * hidden
    decay_parameters = channels
    hidden_weight_parameters = channels * hidden * hidden
    hidden_bias_parameters = hidden
    shared_trunk_parameters = (
        state_input_parameters
        + decay_parameters
        + hidden_weight_parameters
        + hidden_bias_parameters
    )
    output_head_weights = hidden * stored_modes
    output_head_biases = stored_modes
    graph_parameters = (
        shared_trunk_parameters
        + output_head_weights
        + output_head_biases
    )

    router_weights = (
        hidden * route_count
        if include_router
        else 0
    )
    router_biases = route_count if include_router else 0
    router_normalization = 2 * hidden if include_router else 0
    router_stored = (
        router_weights + router_biases + router_normalization
    )
    decoder_coefficients = d_model * stored_modes
    delta_mean_coefficients = d_model
    codec_coefficients = decoder_coefficients + delta_mean_coefficients
    learned_coefficients = (
        graph_parameters + router_weights + router_biases
    )
    floating_coefficients = (
        graph_parameters + router_stored + codec_coefficients
    )
    route_mask_booleans = (
        route_count * stored_modes
        if include_router
        else 0
    )

    return ConditionalCausalGraphAccounting(
        logical_key_rows=keys,
        padded_key_rows=padded_keys,
        routed_query_rows=queries,
        padded_query_rows=padded_queries,
        logical_causal_pairs=logical_pairs,
        padded_causal_pairs=padded_pairs,
        width=d_model,
        state_channels=channels,
        hidden_width=hidden,
        routes=route_count,
        include_router=include_router,
        stored_output_modes=stored_modes,
        active_rank_applications=active,
        state_input_macs=state_input_macs,
        hidden_projection_macs=hidden_projection_macs,
        shared_trunk_macs=shared_trunk_macs,
        router_macs=router_macs,
        output_head_macs=output_head_macs,
        decoder_macs=decoder_macs,
        route_tail_macs=route_tail_macs,
        ideal_total_macs=(
            shared_trunk_macs + router_macs + route_tail_macs
        ),
        dense_route_tail_macs=queries
        * stored_modes
        * (hidden + d_model),
        recurrence_decay_argument_multiplications=decay_arguments,
        recurrence_exponential_evaluations=decay_arguments,
        recurrence_state_multiplications=recurrence_state_elements,
        recurrence_state_additions=recurrence_state_elements,
        recurrence_elementwise_arithmetic_ops=recurrence_arithmetic,
        state_input_parameter_count=state_input_parameters,
        decay_parameter_count=decay_parameters,
        hidden_weight_parameter_count=hidden_weight_parameters,
        hidden_bias_parameter_count=hidden_bias_parameters,
        shared_trunk_parameter_count=shared_trunk_parameters,
        output_head_weight_parameter_count=output_head_weights,
        output_head_bias_parameter_count=output_head_biases,
        graph_trainable_parameter_count=graph_parameters,
        router_weight_coefficient_count=router_weights,
        router_bias_coefficient_count=router_biases,
        router_normalization_coefficient_count=router_normalization,
        router_stored_coefficient_count=router_stored,
        decoder_coefficient_count=decoder_coefficients,
        delta_mean_coefficient_count=delta_mean_coefficients,
        codec_fixed_coefficient_count=codec_coefficients,
        total_learned_coefficient_count=learned_coefficients,
        total_floating_runtime_coefficient_count=floating_coefficients,
        route_mask_boolean_count=route_mask_booleans,
        total_runtime_scalar_count=(
            floating_coefficients + route_mask_booleans
        ),
    )


__all__ = [
    "ConditionalCausalGraphAccounting",
    "NativeTransformerSpanAccounting",
    "conditional_causal_graph_accounting",
    "native_transformer_span_accounting",
]
