import unittest

import torch

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.compiler.capabilities import (
    CapabilityValues,
    LengthDomain,
    MatchStatus,
    SequenceCapabilitySet,
    capabilities_from_manifest_v1,
    match_capabilities,
    overlay_capabilities,
    request_from_context,
)
from fisher_graph.compiler.manifest import SequenceSpec


def _context(
    mask: torch.Tensor,
    *,
    offset: int = 0,
    attention_mask_supplied: bool = True,
    position_ids_supplied: bool = False,
) -> SequenceContext:
    batch, length = mask.shape
    positions = (
        torch.arange(length, dtype=torch.long) + offset
    ).unsqueeze(0).expand(batch, -1)
    return SequenceContext(
        query_valid_mask=mask.to(torch.bool),
        key_valid_mask=mask.to(torch.bool),
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=attention_mask_supplied,
            position_ids_supplied=position_ids_supplied,
            cache_positions_supplied=False,
        ),
    )


def _known_capabilities(
    *,
    mask_origins: tuple[str, ...] = ("omitted", "provided"),
    position_origins: tuple[str, ...] = ("omitted", "provided"),
) -> SequenceCapabilitySet:
    return SequenceCapabilitySet(
        length=LengthDomain(1, 32),
        executions=CapabilityValues.known("prefill"),
        qk_relations=CapabilityValues.known("equal"),
        position_relations=CapabilityValues.known(
            "equal",
            "query_suffix",
            "arbitrary",
        ),
        mask_origins=CapabilityValues.known(*mask_origins),
        mask_patterns=CapabilityValues.known(
            "all_valid",
            "left_padded",
            "right_padded",
            "mixed_padded",
            "sparse",
        ),
        mask_representations=CapabilityValues.known("boolean_valid"),
        visibility_families=CapabilityValues.known("global_causal"),
        position_origins=CapabilityValues.known(*position_origins),
        position_domains=CapabilityValues.known(
            "zero_contiguous",
            "offset_contiguous",
            "arbitrary",
        ),
        cache_kinds=CapabilityValues.known("none"),
        dtypes=CapabilityValues.known("float32"),
        devices=CapabilityValues.known("cpu"),
        layouts=CapabilityValues.known("contiguous"),
    )


def _request(context: SequenceContext):
    hidden = torch.randn(
        context.batch_size,
        context.query_length,
        4,
    )
    return request_from_context(
        context,
        hidden,
        mask_representation="boolean_valid",
        visibility_family="global_causal",
        cache_kind="none",
    )


class RuntimeCapabilityTests(unittest.TestCase):
    def test_origin_is_matched_separately_from_normalized_mask(self) -> None:
        mask = torch.ones(1, 3, dtype=torch.bool)
        omitted = _context(mask, attention_mask_supplied=False)
        explicit = _context(mask, attention_mask_supplied=True)
        self.assertTrue(
            torch.equal(
                omitted.query_valid_mask,
                explicit.query_valid_mask,
            )
        )

        omitted_only = _known_capabilities(mask_origins=("omitted",))
        self.assertEqual(
            match_capabilities(omitted_only, _request(omitted)).status,
            MatchStatus.MATCH,
        )
        explicit_match = match_capabilities(
            omitted_only,
            _request(explicit),
        )
        self.assertEqual(explicit_match.status, MatchStatus.MISMATCH)
        self.assertIn("mask_origin", explicit_match.reasons[0])

    def test_position_origin_and_semantic_domain_are_independent(self) -> None:
        mask = torch.ones(1, 4, dtype=torch.bool)
        generated = _context(mask, offset=9)
        supplied = _context(
            mask,
            offset=9,
            position_ids_supplied=True,
        )
        self.assertEqual(
            _request(generated).position_domain,
            "offset_contiguous",
        )
        self.assertEqual(
            _request(supplied).position_domain,
            "offset_contiguous",
        )
        generated_only = _known_capabilities(
            position_origins=("omitted",),
        )
        self.assertEqual(
            match_capabilities(
                generated_only,
                _request(generated),
            ).status,
            MatchStatus.MATCH,
        )
        self.assertEqual(
            match_capabilities(
                generated_only,
                _request(supplied),
            ).status,
            MatchStatus.MISMATCH,
        )

    def test_negative_positions_are_invalid_not_supported_offsets(self) -> None:
        context = _context(
            torch.ones(1, 3, dtype=torch.bool),
            offset=-2,
        )
        request = _request(context)

        self.assertEqual(request.position_domain, "invalid")
        result = match_capabilities(_known_capabilities(), request)
        self.assertEqual(result.status, MatchStatus.MISMATCH)
        self.assertTrue(
            any("position_domain" in reason for reason in result.reasons)
        )

    def test_position_relation_is_distinct_from_query_key_lengths(self) -> None:
        mask = torch.ones(1, 2, dtype=torch.bool)
        base = _context(mask, offset=10)
        distinct = SequenceContext(
            query_valid_mask=base.query_valid_mask,
            key_valid_mask=base.key_valid_mask,
            logical_positions=base.logical_positions,
            key_logical_positions=torch.tensor([[20, 21]]),
            cache_positions=None,
            phase="prefill",
            input_origin=base.input_origin,
        )
        request = _request(distinct)

        self.assertEqual(request.qk_relation, "equal")
        self.assertEqual(request.position_relation, "arbitrary")
        equal_positions_only = _known_capabilities()
        equal_positions_only = SequenceCapabilitySet(
            length=equal_positions_only.length,
            executions=equal_positions_only.executions,
            qk_relations=equal_positions_only.qk_relations,
            position_relations=CapabilityValues.known("equal"),
            mask_origins=equal_positions_only.mask_origins,
            mask_patterns=equal_positions_only.mask_patterns,
            mask_representations=(
                equal_positions_only.mask_representations
            ),
            visibility_families=equal_positions_only.visibility_families,
            position_origins=equal_positions_only.position_origins,
            position_domains=equal_positions_only.position_domains,
            cache_kinds=equal_positions_only.cache_kinds,
            dtypes=equal_positions_only.dtypes,
            devices=equal_positions_only.devices,
            layouts=equal_positions_only.layouts,
        )
        result = match_capabilities(equal_positions_only, request)
        self.assertEqual(result.status, MatchStatus.MISMATCH)
        self.assertTrue(
            any("position_relation" in reason for reason in result.reasons)
        )

    def test_mask_patterns_are_derived_from_normalized_semantics(self) -> None:
        cases = (
            ("all_valid", [[1, 1, 1, 1]]),
            ("right_padded", [[1, 1, 0, 0]]),
            ("left_padded", [[0, 0, 1, 1]]),
            ("mixed_padded", [[1, 1, 0, 0], [0, 0, 1, 1]]),
            ("sparse", [[1, 0, 1, 0]]),
        )
        for expected, values in cases:
            with self.subTest(expected=expected):
                context = _context(torch.tensor(values, dtype=torch.bool))
                self.assertEqual(_request(context).mask_pattern, expected)

    def test_manifest_v1_unknowns_fail_closed_until_overlayed(self) -> None:
        manifest = SequenceSpec(
            policy="fixed",
            minimum_length=3,
            maximum_length=3,
            causal=True,
            attention_mask="optional_all_true",
            padding="none",
            position_ids="unsupported",
            cache="none",
        )
        request = _request(
            _context(
                torch.ones(1, 3, dtype=torch.bool),
                attention_mask_supplied=False,
            )
        )
        incomplete = capabilities_from_manifest_v1(manifest)
        unknown = match_capabilities(incomplete, request)
        self.assertEqual(unknown.status, MatchStatus.UNKNOWN)
        self.assertTrue(
            any("manifest_v1_unexpressed" in item for item in unknown.reasons)
        )

        complete = overlay_capabilities(
            incomplete,
            _known_capabilities(),
        )
        self.assertEqual(
            match_capabilities(complete, request).status,
            MatchStatus.MATCH,
        )
        self.assertEqual(complete.length, LengthDomain(3, 3))
        self.assertEqual(
            complete.position_origins.values,
            frozenset({"omitted"}),
        )

    def test_manifest_input_form_guard_cannot_be_widened_by_overlay(self) -> None:
        manifest = SequenceSpec(
            policy="bounded_dynamic",
            minimum_length=1,
            maximum_length=8,
            causal=True,
            attention_mask="unsupported",
            padding="none",
            position_ids="required",
            cache="none",
        )
        complete = overlay_capabilities(
            capabilities_from_manifest_v1(manifest),
            _known_capabilities(),
        )
        explicit_mask = _request(
            _context(
                torch.ones(1, 3, dtype=torch.bool),
                attention_mask_supplied=True,
                position_ids_supplied=True,
            )
        )
        result = match_capabilities(complete, explicit_mask)
        self.assertEqual(result.status, MatchStatus.MISMATCH)
        self.assertTrue(
            any("mask_origin" in reason for reason in result.reasons)
        )


if __name__ == "__main__":
    unittest.main()
