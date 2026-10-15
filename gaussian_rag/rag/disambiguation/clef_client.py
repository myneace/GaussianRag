"""clef_client.py — Async Cloudflare Workers AI Clef-Flash decision client.

Wire format (identical to the Jev decision protocol observed in jev-rag/jev_test.py):

  POST /accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash
  Authorization: Bearer {api_token}
  Content-Type: application/json

  Body:
    {
      "state": <str | dict | list>,
      "questions": {
        "<key>": {
          "type": "choice" | "noul" | "score",
          "instructions": "<str>",
          // choice only:
          "criteria": {"<option>": "<description>", ...}
          // score only (optional):
          "criteria": ["<level_description>", ...]
        }
      }
    }

  Response:
    {"answers": {"<key>": {"answer": ..., "probabilities": {...}}}}
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question schemas (Pydantic for validation and serialisation)
# ---------------------------------------------------------------------------

class ChoiceQuestion(BaseModel):
    """Non-autoregressive multi-class routing question."""

    type: str = Field(default="choice", frozen=True)
    instructions: str
    criteria: dict[str, str]  # {"option_key": "human-readable description", ...}

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


class NoulQuestion(BaseModel):
    """Binary yes/no question returning a probability score in [0, 1]."""

    type: str = Field(default="noul", frozen=True)
    instructions: str

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions}


class ScoreQuestion(BaseModel):
    """Ordinal scoring question; criteria describe ascending intensity levels."""

    type: str = Field(default="score", frozen=True)
    instructions: str
    criteria: list[str] = Field(
        default_factory=lambda: ["1 - Low", "2 - Medium", "3 - High"]
    )

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


# Union alias for type hints
AnyQuestion = ChoiceQuestion | NoulQuestion | ScoreQuestion


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class QuestionAnswer(BaseModel):
    """Per-question result returned by Clef-Flash."""

    answer: Any = None  # str for choice, bool for noul, float for score
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    type: str | None = None
    score: float | None = None
    choice: str | None = None
    noul: float | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_wire_data(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return {"answer": data}
        data = dict(data)
        # If 'answer' already provided (e.g. from tests or prior format), keep it
        if "answer" not in data or data["answer"] is None:
            if "choice" in data:
                data["answer"] = data["choice"]
            elif "noul" in data:
                noul_val = data["noul"]
                data["answer"] = bool(noul_val >= 0.5) if isinstance(noul_val, (int, float)) else noul_val
            elif "score" in data:
                data["answer"] = data["score"]

        # If noul question without probabilities dict, construct calibrated yes/no dict
        if "noul" in data and not data.get("probabilities"):
            noul_val = data["noul"]
            if isinstance(noul_val, (int, float)):
                data["probabilities"] = {
                    "yes": round(float(noul_val), 4),
                    "no": round(1.0 - float(noul_val), 4),
                }

        return data


class ClefDecisionResponse(BaseModel):
    """Full response from a Clef decision request."""

    answers: dict[str, QuestionAnswer]
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_CF_BASE = "https://api.cloudflare.com/client/v4"
_CLEF_FLASH_MODEL = "@cf/cloudflare/clef-flash"

# HTTP status codes that are safe to retry
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ClefClient:
    """Async Cloudflare Workers AI client for the Clef-Flash decision model.

    Instantiate once and reuse across requests — the underlying
    ``httpx.AsyncClient`` maintains a keep-alive connection pool.

    Args:
        account_id: Cloudflare account ID (``CLOUDFLARE_ACCOUNT_ID``).
        api_token:  Cloudflare API token (``CLOUDFLARE_API_TOKEN``).
        model:      Clef model identifier.  Defaults to ``@cf/cloudflare/clef-flash``.
        timeout:    Per-request timeout in seconds.  Defaults to 15.0.
        max_retries: Maximum retry attempts for transient failures.  Defaults to 3.
    """

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str = _CLEF_FLASH_MODEL,
        timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        if not account_id:
            raise ValueError("account_id must be a non-empty string")
        if not api_token:
            raise ValueError("api_token must be a non-empty string")

        self._account_id = account_id
        self._model = model
        self._max_retries = max_retries

        self._url = (
            f"{_CF_BASE}/accounts/{account_id}/ai/run/{model}"
        )
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gaussian-rag-clef-client/1.0",
        }
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0,
            ),
            headers=self._headers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def request_decision(
        self,
        state: Any,
        questions: dict[str, AnyQuestion | dict[str, Any]],
    ) -> ClefDecisionResponse:
        """Send a batch of typed decision questions to Clef-Flash.

        Args:
            state:     The context string, dict, or list describing the situation
                       that the model should reason about.
            questions: A mapping of question keys to typed question objects
                       (or pre-serialised wire-format dicts).

        Returns:
            :class:`ClefDecisionResponse` with per-question answers and
            calibrated probability distributions.

        Raises:
            ClefRequestError: On non-retryable HTTP errors.
            ClefRetryExhaustedError: When all retry attempts are exhausted.
        """
        wire_questions: dict[str, Any] = {}
        for key, q in questions.items():
            if isinstance(q, (ChoiceQuestion, NoulQuestion, ScoreQuestion)):
                wire_questions[key] = q.to_wire()
            else:
                wire_questions[key] = q  # pass raw dict through

        payload = {"state": state, "questions": wire_questions}

        t0 = time.perf_counter()
        raw = await self._post_with_retry(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        answers_raw = raw.get("answers") or raw.get("result", {}).get("answers", {})
        if not isinstance(answers_raw, dict):
            raise ClefRequestError(
                f"Unexpected response shape — 'answers' not a dict: {raw!r}"
            )

        answers = {
            k: QuestionAnswer.model_validate(v) for k, v in answers_raw.items()
        }
        return ClefDecisionResponse(answers=answers, elapsed_ms=round(elapsed_ms, 2))

    async def batch_request(
        self,
        state: Any,
        question_batches: list[dict[str, AnyQuestion | dict[str, Any]]],
    ) -> list[ClefDecisionResponse]:
        """Dispatch multiple question-dicts concurrently over the same connection pool.

        Useful when independent question groups should not share a single JSON
        payload (e.g., for isolation or size reasons).
        """
        tasks = [self.request_decision(state, batch) for batch in question_batches]
        return list(await asyncio.gather(*tasks))

    async def close(self) -> None:
        """Gracefully shut down the underlying HTTP connection pool."""
        await self._client.aclose()

    # Async context manager support
    async def __aenter__(self) -> "ClefClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to the Clef endpoint with exponential-backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self._url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    # Cloudflare wraps in {"result": ..., "success": true}
                    if "result" in data and isinstance(data["result"], dict):
                        return data["result"]
                    return data
                if response.status_code in _RETRYABLE_STATUS:
                    wait = 0.4 * (2 ** attempt)
                    last_exc = ClefRequestError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                    if attempt < self._max_retries:
                        logger.warning(
                            "ClefClient: HTTP %s on attempt %d/%d — retrying in %.1fs",
                            response.status_code,
                            attempt + 1,
                            self._max_retries,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    # Retries exhausted on a retryable status
                    raise ClefRetryExhaustedError(
                        f"All {self._max_retries} retries exhausted on HTTP {response.status_code}"
                    ) from last_exc
                # Non-retryable HTTP error — fail immediately
                raise ClefRequestError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = 0.4 * (2 ** attempt)
                    logger.warning(
                        "ClefClient: transport error on attempt %d/%d — retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise ClefRetryExhaustedError(
                    f"All {self._max_retries} retries exhausted: {exc}"
                ) from exc

        # Exhausted retries from a retryable HTTP status
        assert last_exc is not None
        raise ClefRetryExhaustedError(
            f"All {self._max_retries} retries exhausted"
        ) from last_exc


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ClefClientError(RuntimeError):
    """Base exception for ClefClient errors."""


class ClefRequestError(ClefClientError):
    """Raised on non-retryable HTTP errors or malformed responses."""


class ClefRetryExhaustedError(ClefClientError):
    """Raised when all retry attempts for a transient failure are exhausted."""
