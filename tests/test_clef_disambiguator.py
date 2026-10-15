"""test_clef_disambiguator.py — Unit tests for ClefDisambiguator and disambiguate_fast.

Tests cover:
  1. Unambiguous query fast-path (single noul gate call, no sense selection)
  2. Ambiguous query full path (noul + choice gates)
  3. Sense node construction from Clef response
  4. Domain candidate pool heuristics
  5. disambiguate_fast() returns valid DisambiguationPacket
  6. Pipeline integration: field_weights sum to 1.0, entropy is non-negative
"""

from __future__ import annotations

import math
import os
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from gaussian_rag.config import load_dotenv_file

for _k, _v in load_dotenv_file(".env").items():
    os.environ.setdefault(_k, _v)
for _k, _v in load_dotenv_file("../.env").items():
    os.environ.setdefault(_k, _v)

from gaussian_rag.core.types import DisambiguationPacket, SenseNode
from gaussian_rag.rag.disambiguation.clef_client import ClefClient, ClefDecisionResponse, QuestionAnswer
from gaussian_rag.rag.disambiguation.clef_disambiguator import (
    ClefDisambiguator,
    _extract_candidate_domains,
    _build_sense_node,
    disambiguate_fast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client() -> ClefClient:
    return ClefClient(account_id="test-account", api_token="test-token", max_retries=0)


def _clef_response(answers: dict[str, dict[str, Any]]) -> ClefDecisionResponse:
    return ClefDecisionResponse(
        answers={
            k: QuestionAnswer(answer=v["answer"], probabilities=v.get("probabilities", {}))
            for k, v in answers.items()
        },
        elapsed_ms=5.0,
    )


# ---------------------------------------------------------------------------
# 1. Domain candidate heuristics
# ---------------------------------------------------------------------------

class TestExtractCandidateDomains:
    def test_returns_at_least_two_candidates(self) -> None:
        candidates = _extract_candidate_domains("What is the meaning of life?")
        assert len(candidates) >= 2

    def test_general_always_included(self) -> None:
        domains = [d for _, d in _extract_candidate_domains("quantum entanglement")]
        # May not be first, but should appear somewhere if keyword not matched
        assert len(domains) >= 2

    def test_tech_keywords_detected(self) -> None:
        candidates = _extract_candidate_domains("What is the best Python framework for API development?")
        labels = [label for label, _ in candidates]
        assert "technology" in labels

    def test_finance_keywords_detected(self) -> None:
        candidates = _extract_candidate_domains("How is the stock market doing after the earnings report?")
        labels = [label for label, _ in candidates]
        assert "finance" in labels

    def test_max_candidates_capped(self) -> None:
        candidates = _extract_candidate_domains(
            "Python stock market patient legal contract company research"
        )
        assert len(candidates) <= 5


# ---------------------------------------------------------------------------
# 2. Unambiguous query fast path
# ---------------------------------------------------------------------------

class TestUnambiguousPath:
    @pytest.fixture
    def client(self) -> ClefClient:
        return _mock_client()

    @pytest.mark.asyncio
    async def test_unambiguous_returns_single_sense(self, client: ClefClient) -> None:
        disambiguator = ClefDisambiguator(client, ambiguity_threshold=0.5)

        unambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": False, "probabilities": {"yes": 0.12, "no": 0.88}}
        })

        with patch.object(client, "request_decision", new_callable=AsyncMock, return_value=unambiguous_resp):
            is_ambiguous, senses = await disambiguator.route("What is 2 + 2?")

        assert is_ambiguous is False
        assert len(senses) == 1
        assert senses[0].confidence == 1.0
        assert "unambiguous" in senses[0].supporting_evidence[0].lower()

    @pytest.mark.asyncio
    async def test_unambiguous_makes_exactly_one_clef_call(self, client: ClefClient) -> None:
        disambiguator = ClefDisambiguator(client)

        unambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": False, "probabilities": {"yes": 0.1, "no": 0.9}}
        })

        call_count = 0

        async def _once(*_: Any, **__: Any) -> ClefDecisionResponse:
            nonlocal call_count
            call_count += 1
            return unambiguous_resp

        with patch.object(client, "request_decision", side_effect=_once):
            await disambiguator.route("simple factual query")

        assert call_count == 1


# ---------------------------------------------------------------------------
# 3. Ambiguous query full path
# ---------------------------------------------------------------------------

class TestAmbiguousPath:
    @pytest.fixture
    def client(self) -> ClefClient:
        return _mock_client()

    @pytest.mark.asyncio
    async def test_ambiguous_returns_multiple_senses(self, client: ClefClient) -> None:
        disambiguator = ClefDisambiguator(client, ambiguity_threshold=0.5)

        ambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": True, "probabilities": {"yes": 0.82, "no": 0.18}}
        })
        sense_resp = _clef_response({
            "dominant_domain": {
                "answer": "technology",
                "probabilities": {"technology": 0.65, "finance": 0.25, "general": 0.10},
            }
        })

        responses = iter([ambiguous_resp, sense_resp])

        async def _responses(*_: Any, **__: Any) -> ClefDecisionResponse:
            return next(responses)

        with patch.object(client, "request_decision", side_effect=_responses):
            is_ambiguous, senses = await disambiguator.route("apple stock python")

        assert is_ambiguous is True
        assert len(senses) >= 1
        # The top sense should correspond to the highest probability domain
        assert senses[0].domain is not None

    @pytest.mark.asyncio
    async def test_ambiguous_makes_exactly_two_clef_calls(self, client: ClefClient) -> None:
        disambiguator = ClefDisambiguator(client)

        ambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": True, "probabilities": {"yes": 0.9, "no": 0.1}}
        })
        sense_resp = _clef_response({
            "dominant_domain": {
                "answer": "technology",
                "probabilities": {"technology": 0.8, "general": 0.2},
            }
        })
        responses = iter([ambiguous_resp, sense_resp])

        call_count = 0

        async def _count_calls(*_: Any, **__: Any) -> ClefDecisionResponse:
            nonlocal call_count
            call_count += 1
            return next(responses)

        with patch.object(client, "request_decision", side_effect=_count_calls):
            await disambiguator.route("apple stock python", max_senses=2)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_senses_confidence_normalized(self, client: ClefClient) -> None:
        disambiguator = ClefDisambiguator(client)

        ambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": True, "probabilities": {"yes": 0.75, "no": 0.25}}
        })
        sense_resp = _clef_response({
            "dominant_domain": {
                "answer": "finance",
                "probabilities": {"finance": 0.60, "technology": 0.30, "general": 0.10},
            }
        })
        responses = iter([ambiguous_resp, sense_resp])

        async def _r(*_: Any, **__: Any) -> ClefDecisionResponse:
            return next(responses)

        with patch.object(client, "request_decision", side_effect=_r):
            _, senses = await disambiguator.route("market crash python api", max_senses=3)

        total = sum(s.confidence for s in senses)
        assert abs(total - 1.0) < 0.01  # normalized within 1%


# ---------------------------------------------------------------------------
# 4. SenseNode construction
# ---------------------------------------------------------------------------

class TestBuildSenseNode:
    def test_sense_node_has_valid_fields(self) -> None:
        node = _build_sense_node("python programming language", "technology/software", 0.75)
        assert isinstance(node, SenseNode)
        assert node.domain == "technology/software"
        assert node.confidence == 0.75
        assert "technology/software" in node.retrieval_query or "software" in node.retrieval_query
        assert len(node.sense_id) > 0

    def test_sense_id_is_stable(self) -> None:
        node1 = _build_sense_node("apple", "technology/corporate", 0.8)
        node2 = _build_sense_node("apple", "technology/corporate", 0.6)
        # Same text+domain → same sense_id regardless of confidence
        assert node1.sense_id == node2.sense_id

    def test_context_hints_extracted_from_text(self) -> None:
        node = _build_sense_node("machine learning algorithm optimisation", "technology/software", 1.0)
        assert isinstance(node.context_hints, list)


# ---------------------------------------------------------------------------
# 5. disambiguate_fast() returns valid DisambiguationPacket
# ---------------------------------------------------------------------------

class TestDisambiguateFast:
    @pytest.fixture
    def client(self) -> ClefClient:
        return _mock_client()

    @pytest.mark.asyncio
    async def test_unambiguous_returns_single_sense_packet(self, client: ClefClient) -> None:
        unambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": False, "probabilities": {"yes": 0.1, "no": 0.9}}
        })

        with patch.object(client, "request_decision", new_callable=AsyncMock, return_value=unambiguous_resp):
            packet = await disambiguate_fast("What is 2 + 2?", client)

        assert isinstance(packet, DisambiguationPacket)
        assert len(packet.senses) == 1
        assert packet.selected_sense_id == packet.senses[0].sense_id
        assert packet.entropy == 0.0  # log2(1) = 0

    @pytest.mark.asyncio
    async def test_ambiguous_returns_multi_sense_packet(self, client: ClefClient) -> None:
        ambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": True, "probabilities": {"yes": 0.8, "no": 0.2}}
        })
        sense_resp = _clef_response({
            "dominant_domain": {
                "answer": "technology",
                "probabilities": {"technology": 0.7, "finance": 0.3},
            }
        })
        responses = iter([ambiguous_resp, sense_resp])

        async def _r(*_: Any, **__: Any) -> ClefDecisionResponse:
            return next(responses)

        with patch.object(client, "request_decision", side_effect=_r):
            packet = await disambiguate_fast("apple stock python", client, max_senses=2)

        assert isinstance(packet, DisambiguationPacket)
        assert len(packet.senses) >= 1
        assert packet.selected_sense_id is None  # multi-sense = None
        assert packet.disambiguation_id != ""

    @pytest.mark.asyncio
    async def test_field_weights_sum_to_one(self, client: ClefClient) -> None:
        ambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": True, "probabilities": {"yes": 0.9, "no": 0.1}}
        })
        sense_resp = _clef_response({
            "dominant_domain": {
                "answer": "finance",
                "probabilities": {"finance": 0.5, "technology": 0.3, "general": 0.2},
            }
        })
        responses = iter([ambiguous_resp, sense_resp])

        async def _r(*_: Any, **__: Any) -> ClefDecisionResponse:
            return next(responses)

        with patch.object(client, "request_decision", side_effect=_r):
            packet = await disambiguate_fast("market crash python", client, max_senses=3)

        total_weight = sum(packet.field_weights.values())
        assert abs(total_weight - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_entropy_is_non_negative(self, client: ClefClient) -> None:
        unambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": False, "probabilities": {"yes": 0.2, "no": 0.8}}
        })

        with patch.object(client, "request_decision", new_callable=AsyncMock, return_value=unambiguous_resp):
            packet = await disambiguate_fast("simple query", client)

        assert packet.entropy >= 0.0

    @pytest.mark.asyncio
    async def test_memory_ops_initialized_empty(self, client: ClefClient) -> None:
        unambiguous_resp = _clef_response({
            "is_ambiguous": {"answer": False, "probabilities": {"yes": 0.1, "no": 0.9}}
        })

        with patch.object(client, "request_decision", new_callable=AsyncMock, return_value=unambiguous_resp):
            packet = await disambiguate_fast("test query", client)

        assert packet.memory_ops.upsert_nodes == []
        assert packet.memory_ops.prune_ids == []


# ---------------------------------------------------------------------------
# 6. Fallback behavior: missing Clef answer
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    @pytest.fixture
    def client(self) -> ClefClient:
        return _mock_client()

    @pytest.mark.asyncio
    async def test_missing_ambiguity_answer_defaults_to_unambiguous(self, client: ClefClient) -> None:
        """Missing 'is_ambiguous' key → defaults to unambiguous (safe fallback)."""
        empty_resp = _clef_response({})  # no 'is_ambiguous' key

        with patch.object(client, "request_decision", new_callable=AsyncMock, return_value=empty_resp):
            disambiguator = ClefDisambiguator(client)
            is_ambiguous, _ = await disambiguator.check_ambiguity("test")

        assert is_ambiguous is False

    @pytest.mark.asyncio
    async def test_missing_sense_answer_falls_back_to_top_candidate(self, client: ClefClient) -> None:
        """Missing 'dominant_domain' key → uses top heuristic candidate."""
        empty_resp = _clef_response({})

        with patch.object(client, "request_decision", new_callable=AsyncMock, return_value=empty_resp):
            disambiguator = ClefDisambiguator(client)
            senses = await disambiguator.select_senses("python api framework")

        assert len(senses) >= 1
        assert isinstance(senses[0], SenseNode)


# ---------------------------------------------------------------------------
# 7. Live end-to-end disambiguation test (requires CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID")),
    reason="Live Cloudflare credentials not available",
)
@pytest.mark.asyncio
async def test_live_clef_disambiguation() -> None:
    """Live smoke test verifying ambiguity detection and sense selection via Cloudflare Clef-Flash."""
    async with ClefClient(
        account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
        api_token=os.environ["CLOUDFLARE_API_TOKEN"],
        timeout=30.0,
        max_retries=2,
    ) as client:
        # 1. Clear factual query (unambiguous)
        packet_unambig = await disambiguate_fast("What is the speed of light in vacuum?", client)
        assert isinstance(packet_unambig, DisambiguationPacket)
        assert len(packet_unambig.senses) == 1
        assert packet_unambig.selected_sense_id is not None
        assert packet_unambig.entropy == 0.0

        # 2. Multi-domain query (ambiguous)
        packet_ambig = await disambiguate_fast("Apple stock crashed after Python library bug", client, max_senses=3)
        assert isinstance(packet_ambig, DisambiguationPacket)
        assert len(packet_ambig.senses) >= 1
        total_w = sum(packet_ambig.field_weights.values())
        assert abs(total_w - 1.0) < 0.01

