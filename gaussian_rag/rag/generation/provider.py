from __future__ import annotations

from dataclasses import dataclass

import httpx

from gaussian_rag.config import ProviderEnvConfig


@dataclass(slots=True)
class MistralChatClient:
    provider_env: ProviderEnvConfig
    timeout_seconds: float = 30.0

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.provider_env["model"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.provider_env['api_key']}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            self.provider_env["base_url"],
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Provider response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider response missing text content")
        return content.strip()
