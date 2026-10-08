from __future__ import annotations

import json
import logging
import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Any

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.creator.chunker import split_into_chunks

logger = logging.getLogger(__name__)


def _strip_markdown_json(text: str) -> str:
    """Strip leading/trailing markdown code fences from an LLM JSON response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

LLM_AI_CHUNKER_PROMPT = """
ROLE: Semantic Proposition Extractor.
TASK: Analyze the input text chunk and break it down into an exhaustive list of atomic, self-contained propositions or facts.
This implements a combination of heuristic boundaries and AI-driven expansion into the Gaussian space. 
For each distinct fact, idea, or subject-predicate-object relationship, generate a clean, self-contained sentence. 
Resolve all pronouns (e.g., replace "He" with the actual name) so each proposition can stand alone in the vector space.

INPUT TEXT: "{chunk_text}"

OUTPUT SCHEMA:
{{
  "propositions": [
    "<self-contained fact 1>",
    "<self-contained fact 2>"
  ]
}}
"""

LLM_DISSOLVER_PROMPT = """
ROLE: High-Resolution Semantic Manifold Dissolver.
TASK: Analyze the input text for semantic complexity, polysemy, and latent dimensions. 
Project the text into an exhaustive set of distinct contextual interpretations.

GUIDELINES:
- **Exhaustive Mapping**: Generate as many meaningfully distinct interpretations as necessary to fully capture the semantic variance and multi-dimensional nature of the text. 
- **Higher Precision**: Do not limit the output to a specific number. The goal is a high-resolution expansion of the semantic field. If the text has 10 potential domain intersections, generate 10 interpretations.
- **Unique Value**: Ensure each interpretation represents a unique semantic "view" or contextual domain.
- **Literal Text**: For simple, absolute facts, you may still provide fewer interpretations to maintain efficiency, but always lean toward higher resolution if any nuance exists.

INPUT TEXT: "{chunk_text}"

OUTPUT SCHEMA:
{{
  "ambiguity_score": <int 1-10, where 1 is literal and 10 is highly polysemous>,
  "reasoning": "<short explanation of semantic complexity>",
  "interpretations": [
    {{
      "domain": "<domain_name>",
      "text": "<domain-specific rewrite>"
    }}
  ]
}}
"""


def _dissolve_chunk(text: str, llm_caller: Callable[[str], str]) -> dict[str, Any]:
    prompt = LLM_DISSOLVER_PROMPT.format(chunk_text=text)
    try:
        return json.loads(_strip_markdown_json(llm_caller(prompt)))
    except Exception as e:
        logger.warning("Agent Dissolver failed: %s", e)
        return {}


def _ai_chunk_text(text: str, llm_caller: Callable[[str], str]) -> list[str]:
    prompt = LLM_AI_CHUNKER_PROMPT.format(chunk_text=text)
    try:
        data = json.loads(_strip_markdown_json(llm_caller(prompt)))
        return data.get("propositions", [])
    except Exception as e:
        logger.warning("AI Chunker failed: %s", e)
        return []


@dataclass(slots=True)
class IngestionPipeline:
    embedder: GaussianEmbedder
    llm_caller: Callable[[str], str] | None = None
    max_workers: int = 4
    extract_entities: bool = False
    ai_chunking: bool = False
    on_event: Callable[[str, dict[str, Any]], Any] | None = None

    def _emit(self, event_type: str, data: dict[str, Any]):
        if self.on_event:
            self.on_event(event_type, data)

    def _extract_entities(self, text: str) -> list[str]:
        import re
        stop_words = {
            "the", "a", "an", "of", "and", "or", "in", "on", "at", "to",
            "for", "with", "by", "as", "is", "are", "was", "were", "be",
            "been", "being", "it", "its", "this", "that", "these", "those",
        }
        tokens = re.findall(r"\b[A-Za-z][a-z]*\b", text)
        entities = [
            t for t in tokens
            if t[0].isupper() and t.lower() not in stop_words
        ]
        # De-duplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            lower_e = e.lower()
            if lower_e not in seen:
                seen.add(lower_e)
                unique_entities.append(e)
        return unique_entities

    def ingest_document(
        self, text: str, *, document_id: str, metadata: dict[str, Any] | None = None
    ) -> list[GaussianKnowledge]:
        self._emit("ingestion_started", {"document_id": document_id})
        
        chunks = split_into_chunks(text)
        self._emit("chunking_complete", {"count": len(chunks)})

        # Determine if we should use the dissolver based on embedder's sigma_mode
        use_dissolver = (
            self.embedder.sigma_mode == "dissolver" and self.llm_caller is not None
        )

        if use_dissolver:
            self._emit("dissolver_started", {"count": len(chunks)})
            llm_caller = self.llm_caller
            assert llm_caller is not None
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    futures.append(executor.submit(_dissolve_chunk, chunk, llm_caller))

                results = []
                for i, f in enumerate(futures):
                    res = f.result()
                    results.append(res)
                    self._emit("agent_dissolved", {"chunk_index": i, "interpretations": len(res.get("interpretations", []))})

                dissolved_data = []
                for res in results:
                    interps = [
                        item.get("text", "")
                        for item in res.get("interpretations", [])
                        if item.get("text")
                    ]
                    dissolved_data.append(
                        {
                            "interpretations": interps,
                            "ambiguity_score": res.get("ambiguity_score", 1),
                            "reasoning": res.get("reasoning", ""),
                        }
                    )
        else:
            dissolved_data = [
                {"interpretations": [], "ambiguity_score": 0, "reasoning": ""}
                for _ in chunks
            ]

        # Determine if we should use the AI chunker to expand the space
        use_ai_chunking = self.ai_chunking and self.llm_caller is not None
        if use_ai_chunking:
            self._emit("ai_chunking_started", {"count": len(chunks)})
            llm_caller = self.llm_caller
            assert llm_caller is not None
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                ai_chunks_futures = [
                    executor.submit(_ai_chunk_text, chunk, llm_caller)
                    for chunk in chunks
                ]
                ai_chunks_results = []
                for i, f in enumerate(ai_chunks_futures):
                    res = f.result()
                    ai_chunks_results.append(res)
                    self._emit("agent_proposition_extracted", {"chunk_index": i, "count": len(res)})
        else:
            ai_chunks_results = [[] for _ in chunks]

        all_nodes: list[GaussianKnowledge] = []
        for index, (chunk, data, ai_props) in enumerate(zip(chunks, dissolved_data, ai_chunks_results)):
            interpretations = data["interpretations"]
            anchor_id = f"{document_id}-chunk-{index}"
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": index,
                "node_type": "anchor",
                "ambiguity_score": data["ambiguity_score"],
                "dissolver_reasoning": data["reasoning"],
                **(metadata or {}),
            }

            if self.extract_entities:
                entities = self._extract_entities(chunk)
                chunk_metadata["entities"] = entities
                if entities:
                    self._emit("entities_extracted", {"chunk_index": index, "entities": entities})

            # 1. Create the Anchor Node (the semantic field)
            anchor_node = self.embedder.encode(
                chunk,
                item_id=anchor_id,
                metadata=chunk_metadata,
                interpretations=interpretations,
            )
            all_nodes.append(anchor_node)
            self._emit("node_created", {"id": anchor_id, "type": "anchor"})

            # 2. Create Interpretation Nodes (the dissolved points)
            if use_dissolver and interpretations:
                original_mode = self.embedder.sigma_mode
                self.embedder.sigma_mode = "heuristic"
                try:
                    for i_idx, interp_text in enumerate(interpretations):
                        interp_id = f"{anchor_id}-interp-{i_idx}"
                        interp_metadata = {
                            **chunk_metadata,
                            "node_type": "interpretation",
                            "anchor_id": anchor_id,
                            "interpretation_index": i_idx,
                        }
                        node = self.embedder.encode(
                            interp_text, item_id=interp_id, metadata=interp_metadata
                        )
                        all_nodes.append(node)
                        self._emit("node_created", {"id": interp_id, "type": "interpretation"})
                finally:
                    self.embedder.sigma_mode = original_mode

            # 3. Create AI Chunker Nodes (the expanded atomic propositions)
            if use_ai_chunking and ai_props:
                original_mode = self.embedder.sigma_mode
                self.embedder.sigma_mode = "heuristic"
                try:
                    for p_idx, prop_text in enumerate(ai_props):
                        prop_id = f"{anchor_id}-aiprop-{p_idx}"
                        prop_metadata = {
                            **chunk_metadata,
                            "node_type": "ai_proposition",
                            "anchor_id": anchor_id,
                            "proposition_index": p_idx,
                        }
                        node = self.embedder.encode(
                            prop_text, item_id=prop_id, metadata=prop_metadata
                        )
                        all_nodes.append(node)
                        self._emit("node_created", {"id": prop_id, "type": "ai_proposition"})
                finally:
                    self.embedder.sigma_mode = original_mode

        self._emit("ingestion_complete", {"document_id": document_id, "total_nodes": len(all_nodes)})
        return all_nodes
