"""OpenAI-compatible AI assistant for text normalization and corpus structuring.

Assists with ambiguous raw text, corrupted OCR characters, scrambled unicode,
mojibake, and difficult document boundary detection using any OpenAI-compatible
LLM endpoint (e.g., OpenAI, Mistral, Gemini, Ollama, vLLM, LM Studio).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenAICorpusAssistant:
    """AI Assistant leveraging any OpenAI-compatible `/chat/completions` endpoint.

    Automatically resolves credentials from arguments or environment variables:
      1. Explicit ``api_key``, ``base_url``, ``model``
      2. ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL``
      3. ``MISTRAL_API_KEY``, ``MISTRAL_BASE_URL``, ``MISTRAL_MODEL``
      4. ``GEMINI_API_KEY`` (using Google's OpenAI-compatible endpoint)

    Gracefully falls back if endpoints are unreachable or unconfigured.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        # Only fall back to env-var discovery when api_key is *not* explicitly
        # supplied (None). An explicit empty string means "no key" and should
        # NOT trigger automatic env-var lookup — this lets tests assert that
        # an unconfigured instance behaves as a no-op without env var leakage.
        if api_key is None:
            self.api_key = (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("MISTRAL_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or ""
            )
        else:
            self.api_key = api_key
        
        # Resolve base_url
        default_base_url = "https://api.openai.com/v1"
        if os.getenv("MISTRAL_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            default_base_url = "https://api.mistral.ai/v1"
        elif os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

        raw_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("MISTRAL_BASE_URL") or default_base_url
        self.base_url = self._normalize_base_url(raw_base_url)

        # Resolve model
        default_model = "gpt-4o-mini"
        if "mistral.ai" in self.base_url:
            default_model = "mistral-small-latest"
        elif "generativelanguage.googleapis.com" in self.base_url:
            default_model = "gemini-1.5-flash"

        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("MISTRAL_MODEL") or default_model
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        url = url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[:-len("/chat/completions")]
        return url

    @property
    def is_configured(self) -> bool:
        """Returns True if an API key or local base_url is present."""
        return bool(self.api_key or "localhost" in self.base_url or "127.0.0.1" in self.base_url)

    def _call_chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI-compatible /chat/completions endpoint."""
        if not self.is_configured:
            raise ValueError("OpenAICorpusAssistant is not configured with an API key or endpoint.")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("Response missing 'choices'")
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()

    def repair_ambiguous_text(self, text: str) -> str:
        """Repair scrambled characters, OCR errors, broken line-breaks, and mojibake.

        Falls back to returning the input string if the endpoint fails.
        """
        if not text or not text.strip():
            return ""

        if not self.is_configured:
            logger.debug("OpenAICorpusAssistant not configured — skipping AI text repair.")
            return text

        system_prompt = (
            "You are an expert text normalizer and OCR repair specialist. "
            "Clean and repair the following messy or corrupted user-pasted text:\n"
            "1. Fix OCR errors (e.g., 'rn' misread as 'm', '0' misread as 'O', split words, accidental punctuation).\n"
            "2. Fix scrambled characters, encoding artifacts, and mojibake.\n"
            "3. Unwrap artificial line-breaks while preserving genuine paragraph divisions.\n"
            "4. Remove leftover UI/boilerplate artifacts, broken headers, and page counters.\n"
            "5. DO NOT summarize, hallucinate, omit important information, or add commentary. Return ONLY the repaired clean text."
        )

        try:
            cleaned = self._call_chat_completions(system_prompt, text)
            return cleaned if cleaned else text
        except Exception as exc:
            logger.warning("AI text repair call failed, falling back to rule-based output: %s", exc)
            return text

    def structure_into_sections(
        self, text: str, default_title: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Use the AI to detect semantic sections and return structured JSON records.

        Returns a list of dicts with keys: 'id', 'title', 'section_number', 'text', 'metadata'.
        Returns None if AI call fails or is unconfigured.
        """
        if not text or not text.strip():
            return None

        if not self.is_configured:
            return None

        system_prompt = (
            "You are a document structure parser for RAG systems. "
            "Analyze the text and partition it into coherent semantic sections (articles, laws, topics, or QA pairs). "
            "Return a strictly valid JSON array of objects with the following schema:\n"
            "[\n"
            "  {\n"
            "    \"id\": \"sec-1\",\n"
            "    \"title\": \"Section or Topic Title\",\n"
            "    \"section_number\": 1,\n"
            "    \"text\": \"Clean, normalized text content for this section.\",\n"
            "    \"metadata\": {\"detected_type\": \"law|section|topic|qa|other\"}\n"
            "  }\n"
            "]\n"
            "Do not output markdown code blocks (```json). Output ONLY the raw JSON string."
        )

        user_prompt = f"Document Title Context: {default_title or 'Untitled Document'}\n\nContent:\n{text}"

        try:
            raw_json = self._call_chat_completions(system_prompt, user_prompt)
            # Strip markdown code blocks if the model wrapped it
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json.strip())
            raw_json = re.sub(r"\s*```$", "", raw_json.strip())
            data = json.loads(raw_json)
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception as exc:
            logger.warning("AI structure extraction failed (%s); falling back to rule-based chunker.", exc)

        return None
