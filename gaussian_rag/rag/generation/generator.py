from __future__ import annotations

from dataclasses import dataclass

from gaussian_rag.config import GenerationConfig, ProviderEnvConfig
from gaussian_rag.core.types import RetrievedChunk

from .context_builder import build_context
from .provider import MistralChatClient
from .prompt_templates import uncertainty_prompt


@dataclass(slots=True)
class SimpleGenerator:
    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> dict[str, str]:
        context = build_context(query, retrieved)
        prompt = uncertainty_prompt(context, query)
        answer_lines = [
            f"Query: {query}",
            "Evidence summary:",
        ]
        for item in retrieved:
            answer_lines.append(f"- ({item.uncertainty_label}) {item.knowledge.text}")
        return {"context": context, "prompt": prompt, "answer": "\n".join(answer_lines)}


@dataclass(slots=True)
class ProviderBackedGenerator:
    generation_config: GenerationConfig
    provider_env: ProviderEnvConfig

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> dict[str, str]:
        context = build_context(query, retrieved)
        prompt = uncertainty_prompt(context, query)
        client = MistralChatClient(
            self.provider_env,
            timeout_seconds=self.generation_config["timeout_seconds"],
        )
        answer = client.generate(prompt)
        return {"context": context, "prompt": prompt, "answer": answer}
