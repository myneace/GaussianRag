"""test_clef_client.py — Comprehensive unit and integration tests for ClefClient.

Tests cover:
  1. Question schema serialisation (choice, noul, score)
  2. Successful response parsing with mocked HTTP 200 payload
  3. Retry logic with simulated 429 → 200 recovery
  4. Exception raising when HTTP 500 persists past max_retries
  5. Transport error retry and exhaustion
  6. Live integration test (skipped when CLOUDFLARE_API_TOKEN is absent)
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Any

import pytest
import httpx

from gaussian_rag.config import load_dotenv_file

# Populate os.environ from .env if present so live tests run with real credentials
for _k, _v in load_dotenv_file(".env").items():
    os.environ.setdefault(_k, _v)
for _k, _v in load_dotenv_file("../.env").items():
    os.environ.setdefault(_k, _v)

from gaussian_rag.rag.disambiguation.clef_client import (
    ChoiceQuestion,
    ClefClient,
    ClefDecisionResponse,
    ClefRequestError,
    ClefRetryExhaustedError,
    NoulQuestion,
    QuestionAnswer,
    ScoreQuestion,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_client(**kwargs: Any) -> ClefClient:
    """Return a ClefClient with dummy credentials and a tiny retry budget."""
    return ClefClient(
        account_id="test-account-id",
        api_token="test-api-token",
        max_retries=kwargs.pop("max_retries", 3),
        **kwargs,
    )


def _mock_response(status: int, body: dict[str, Any]) -> MagicMock:
    """Construct a minimal mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


# ---------------------------------------------------------------------------
# 1. Question schema serialisation
# ---------------------------------------------------------------------------

class TestQuestionSchemas:
    def test_choice_question_wire_format(self) -> None:
        q = ChoiceQuestion(
            instructions="Route to the right team",
            criteria={"billing": "Payment issues", "tech": "Software bugs"},
        )
        wire = q.to_wire()
        assert wire["type"] == "choice"
        assert wire["instructions"] == "Route to the right team"
        assert wire["criteria"] == {"billing": "Payment issues", "tech": "Software bugs"}

    def test_noul_question_wire_format(self) -> None:
        q = NoulQuestion(instructions="Is this urgent?")
        wire = q.to_wire()
        assert wire["type"] == "noul"
        assert wire["instructions"] == "Is this urgent?"
        assert "criteria" not in wire

    def test_score_question_with_criteria(self) -> None:
        q = ScoreQuestion(
            instructions="Rate frustration level",
            criteria=["Calm", "Upset", "Furious"],
        )
        wire = q.to_wire()
        assert wire["type"] == "score"
        assert wire["criteria"] == ["Calm", "Upset", "Furious"]

    def test_score_question_without_criteria(self) -> None:
        q = ScoreQuestion(instructions="Rate satisfaction")
        wire = q.to_wire()
        assert wire["type"] == "score"
        assert "criteria" in wire
        assert len(wire["criteria"]) > 0

    def test_choice_question_default_type_immutable(self) -> None:
        q = ChoiceQuestion(instructions="x", criteria={"a": "b"})
        assert q.type == "choice"

    def test_noul_question_default_type_immutable(self) -> None:
        assert NoulQuestion(instructions="x").type == "noul"

    def test_score_question_default_type_immutable(self) -> None:
        assert ScoreQuestion(instructions="x").type == "score"


# ---------------------------------------------------------------------------
# 2. Successful response parsing (mocked HTTP 200)
# ---------------------------------------------------------------------------

class TestSuccessfulParsing:
    @pytest.fixture
    def mock_answer_body(self) -> dict[str, Any]:
        return {
            "answers": {
                "department": {
                    "answer": "billing",
                    "probabilities": {"billing": 0.82, "tech": 0.18},
                },
                "is_urgent": {
                    "answer": True,
                    "probabilities": {"yes": 0.91, "no": 0.09},
                },
                "frustration": {
                    "answer": 0.65,
                    "probabilities": {"low": 0.15, "medium": 0.35, "high": 0.50},
                },
            }
        }

    @pytest.mark.asyncio
    async def test_parses_answers_dict(self, mock_answer_body: dict) -> None:
        client = _make_client()
        mock_resp = _mock_response(200, mock_answer_body)

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.request_decision(
                state="Customer complaint about billing",
                questions={
                    "department": ChoiceQuestion(
                        instructions="Route?",
                        criteria={"billing": "Payments", "tech": "Bugs"},
                    ),
                    "is_urgent": NoulQuestion(instructions="Urgent?"),
                    "frustration": ScoreQuestion(instructions="Rate frustration"),
                },
            )

        assert isinstance(result, ClefDecisionResponse)
        assert result.answers["department"].answer == "billing"
        assert result.answers["department"].probabilities["billing"] == pytest.approx(0.82)
        assert result.answers["is_urgent"].answer is True
        assert result.answers["frustration"].answer == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_elapsed_ms_populated(self, mock_answer_body: dict) -> None:
        client = _make_client()
        mock_resp = _mock_response(200, mock_answer_body)
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.request_decision(state="x", questions={})
        assert result.elapsed_ms >= 0.0

    @pytest.mark.asyncio
    async def test_cloudflare_wrapped_result_format(self) -> None:
        """Cloudflare API wraps answers in {'result': {...}, 'success': True}."""
        client = _make_client()
        body = {
            "result": {
                "answers": {
                    "check": {"answer": True, "probabilities": {"yes": 1.0, "no": 0.0}}
                }
            },
            "success": True,
        }
        mock_resp = _mock_response(200, body)
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.request_decision(state="ctx", questions={})
        assert result.answers["check"].answer is True

    @pytest.mark.asyncio
    async def test_raw_dict_questions_passed_through(self) -> None:
        """Raw dicts should be forwarded to the wire without modification."""
        client = _make_client()
        body = {"answers": {"q": {"answer": "x", "probabilities": {}}}}
        captured: list[dict] = []

        async def _capture(url: str, **kwargs: Any) -> MagicMock:
            captured.append(kwargs.get("json", {}))
            return _mock_response(200, body)

        with patch.object(client._client, "post", side_effect=_capture):
            await client.request_decision(
                state="s",
                questions={"q": {"type": "noul", "instructions": "ok?"}},
            )

        assert captured[0]["questions"]["q"]["type"] == "noul"


# ---------------------------------------------------------------------------
# 3. Retry logic: 429 → 200 recovery
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self) -> None:
        client = _make_client(max_retries=3)
        success_body = {"answers": {"q": {"answer": True, "probabilities": {}}}}
        responses = [
            _mock_response(429, {"error": "rate limited"}),
            _mock_response(200, success_body),
        ]
        response_iter = iter(responses)

        async def _side_effect(*_: Any, **__: Any) -> MagicMock:
            return next(response_iter)

        with (
            patch.object(client._client, "post", side_effect=_side_effect),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client.request_decision(state="x", questions={})

        assert result.answers["q"].answer is True

    @pytest.mark.asyncio
    async def test_retry_sleep_uses_exponential_backoff(self) -> None:
        """Sleep durations must be 0.4 * 2^attempt."""
        client = _make_client(max_retries=2)
        success_body = {"answers": {}}
        responses = [
            _mock_response(429, {}),
            _mock_response(429, {}),
            _mock_response(200, success_body),
        ]
        response_iter = iter(responses)

        async def _side_effect(*_: Any, **__: Any) -> MagicMock:
            return next(response_iter)

        sleep_calls: list[float] = []

        async def _mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with (
            patch.object(client._client, "post", side_effect=_side_effect),
            patch("asyncio.sleep", side_effect=_mock_sleep),
        ):
            await client.request_decision(state="x", questions={})

        # Attempt 0 → sleep 0.4*1 = 0.4, attempt 1 → sleep 0.4*2 = 0.8
        assert sleep_calls == pytest.approx([0.4, 0.8])


# ---------------------------------------------------------------------------
# 4. HTTP 500 persisting past max_retries raises ClefRetryExhaustedError
# ---------------------------------------------------------------------------

class TestRetryExhaustion:
    @pytest.mark.asyncio
    async def test_raises_after_max_retries_on_500(self) -> None:
        client = _make_client(max_retries=2)

        async def _always_500(*_: Any, **__: Any) -> MagicMock:
            return _mock_response(500, {"error": "server error"})

        with (
            patch.object(client._client, "post", side_effect=_always_500),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(ClefRetryExhaustedError):
                await client.request_decision(state="x", questions={})

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retryable_4xx(self) -> None:
        client = _make_client(max_retries=3)

        async def _bad_request(*_: Any, **__: Any) -> MagicMock:
            return _mock_response(400, {"error": "bad request"})

        with patch.object(client._client, "post", side_effect=_bad_request):
            with pytest.raises(ClefRequestError) as exc_info:
                await client.request_decision(state="x", questions={})
        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_immediately_on_401(self) -> None:
        client = _make_client(max_retries=3)

        async def _unauthorized(*_: Any, **__: Any) -> MagicMock:
            return _mock_response(401, {"error": "unauthorized"})

        with patch.object(client._client, "post", side_effect=_unauthorized):
            with pytest.raises(ClefRequestError):
                await client.request_decision(state="x", questions={})


# ---------------------------------------------------------------------------
# 5. Transport error retry and exhaustion
# ---------------------------------------------------------------------------

class TestTransportErrors:
    @pytest.mark.asyncio
    async def test_retries_on_connection_error_then_succeeds(self) -> None:
        client = _make_client(max_retries=2)
        success_body = {"answers": {"q": {"answer": "ok", "probabilities": {}}}}
        call_count = 0

        async def _side_effect(*_: Any, **__: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("connection refused")
            return _mock_response(200, success_body)

        with (
            patch.object(client._client, "post", side_effect=_side_effect),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client.request_decision(state="x", questions={})

        assert result.answers["q"].answer == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_transport_error(self) -> None:
        client = _make_client(max_retries=2)

        async def _always_fail(*_: Any, **__: Any) -> MagicMock:
            raise httpx.ConnectError("network unreachable")

        with (
            patch.object(client._client, "post", side_effect=_always_fail),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(ClefRetryExhaustedError) as exc_info:
                await client.request_decision(state="x", questions={})
        assert "retries exhausted" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 6. Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_raises_on_empty_account_id(self) -> None:
        with pytest.raises(ValueError, match="account_id"):
            ClefClient(account_id="", api_token="tok")

    def test_raises_on_empty_api_token(self) -> None:
        with pytest.raises(ValueError, match="api_token"):
            ClefClient(account_id="acct", api_token="")

    def test_url_format_contains_account_id(self) -> None:
        c = _make_client()
        assert "test-account-id" in c._url
        assert "@cf/cloudflare/clef-flash" in c._url


# ---------------------------------------------------------------------------
# 7. Batch request
# ---------------------------------------------------------------------------

class TestBatchRequest:
    @pytest.mark.asyncio
    async def test_batch_issues_concurrent_requests(self) -> None:
        client = _make_client()
        body = {"answers": {"q": {"answer": True, "probabilities": {}}}}
        mock_resp = _mock_response(200, body)

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            results = await client.batch_request(
                state="ctx",
                question_batches=[
                    {"q": NoulQuestion(instructions="A?")},
                    {"q": NoulQuestion(instructions="B?")},
                ],
            )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, ClefDecisionResponse)


# ---------------------------------------------------------------------------
# 8. Async context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_calls_close(self) -> None:
        async with ClefClient(account_id="a", api_token="b") as client:
            assert isinstance(client, ClefClient)
        # After exiting, the internal httpx client should be closed (no assertion on internal
        # state — we just verify no exception is raised during close)


# ---------------------------------------------------------------------------
# 9. Live integration test (requires CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID")),
    reason="Live Cloudflare credentials not available — set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID to enable",
)
@pytest.mark.asyncio
async def test_live_clef_decision() -> None:
    """End-to-end smoke test against real Cloudflare Workers AI Clef-Flash endpoint."""
    async with ClefClient(
        account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
        api_token=os.environ["CLOUDFLARE_API_TOKEN"],
        timeout=30.0,
        max_retries=2,
    ) as client:
        result = await client.request_decision(
            state="Customer upgraded to Pro but console still shows Free plan.",
            questions={
                "department": ChoiceQuestion(
                    instructions="Which team handles this?",
                    criteria={
                        "billing": "Payment or subscription issues",
                        "technical": "Product bugs or feature failures",
                    },
                ),
                "is_urgent": NoulQuestion(instructions="Does the customer express urgency?"),
            },
        )
    assert isinstance(result, ClefDecisionResponse)
    assert "department" in result.answers
    assert "is_urgent" in result.answers
    assert result.elapsed_ms > 0
