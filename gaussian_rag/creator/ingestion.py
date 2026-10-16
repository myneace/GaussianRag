from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Any

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.creator.chunker import split_into_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOTE: IngestionPipeline is the legacy per-document ingestion helper.
# For new code, prefer UniversalIngestor (gaussian_rag.creator.universal_ingestor)
# which provides format-agnostic ingestion and lockstep dual-store indexing
# (KnowledgeStore + SQLite FTS5 LexicalStore).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IngestionPipeline:
    """Legacy per-document ingestion pipeline.

    Chunks a single text document, embeds each chunk via ``GaussianEmbedder``,
    and returns the resulting ``GaussianKnowledge`` nodes.

    For format-agnostic ingestion and automatic dual-store synchronisation, use
    :class:`~gaussian_rag.creator.universal_ingestor.UniversalIngestor` instead.
    """

    embedder: GaussianEmbedder
    extract_entities: bool = False
    on_event: Callable[[str, dict[str, Any]], Any] | None = None

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
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
        seen: set[str] = set()
        unique_entities: list[str] = []
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

        all_nodes: list[GaussianKnowledge] = []
        for index, chunk in enumerate(chunks):
            anchor_id = f"{document_id}-chunk-{index}"
            chunk_metadata: dict[str, Any] = {
                "document_id": document_id,
                "chunk_index": index,
                "node_type": "anchor",
                **(metadata or {}),
            }

            if self.extract_entities:
                entities = self._extract_entities(chunk)
                chunk_metadata["entities"] = entities
                if entities:
                    self._emit("entities_extracted", {"chunk_index": index, "entities": entities})

            anchor_node = self.embedder.encode(
                chunk,
                item_id=anchor_id,
                metadata=chunk_metadata,
            )
            all_nodes.append(anchor_node)
            self._emit("node_created", {"id": anchor_id, "type": "anchor"})

        self._emit("ingestion_complete", {"document_id": document_id, "total_nodes": len(all_nodes)})
        return all_nodes
