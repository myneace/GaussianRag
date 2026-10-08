from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaussian_rag.core.types import SenseNode


@dataclass(slots=True)
class SenseCache:
    """Session-scoped store for SenseNode objects with TTL-based expiry.

    Mirrors the KnowledgeStore interface pattern but operates on SenseNodes
    rather than GaussianKnowledge items.  Upsert is idempotent by sense_id.
    """

    items: dict[str, SenseNode] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, nodes: list[SenseNode]) -> None:
        """Add or overwrite nodes by sense_id."""
        for node in nodes:
            existing = self.items.get(node.sense_id)
            if existing is None or node.confidence >= existing.confidence:
                self.items[node.sense_id] = node

    def boost_confidence(self, sense_id: str, delta: float = 0.15) -> None:
        """Boost an existing node's confidence (context continuity signal)."""
        node = self.items.get(sense_id)
        if node is not None:
            # slots=True means we cannot set directly — rebuild via dict
            self.items[sense_id] = SenseNode(
                sense_id=node.sense_id,
                term=node.term,
                domain=node.domain,
                confidence=min(node.confidence + delta, 0.95),
                context_hints=node.context_hints,
                supporting_evidence=node.supporting_evidence,
                retrieval_query=node.retrieval_query,
                mu_anchor=node.mu_anchor,
                sigma_anchor=node.sigma_anchor,
                sigma_L_anchor=node.sigma_L_anchor,
                ttl_hours=node.ttl_hours,
                version=node.version,
                created_at=node.created_at,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_term(self, term: str) -> list[SenseNode]:
        """Return all active (non-expired) nodes for a given term."""
        return [
            node
            for node in self.items.values()
            if node.term == term and not node.is_expired()
        ]

    def context_window(self) -> list[str]:
        """Return sense_ids of all active (non-expired) nodes."""
        return [node.sense_id for node in self.items.values() if not node.is_expired()]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        """Remove expired nodes. Returns count removed."""
        expired = [sid for sid, node in self.items.items() if node.is_expired()]
        for sid in expired:
            del self.items[sid]
        return len(expired)

    def prune_by_ids(self, sense_ids: list[str]) -> int:
        """Explicitly remove nodes by ID. Returns count removed."""
        removed = 0
        for sid in sense_ids:
            if sid in self.items:
                del self.items[sid]
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = []
        for node in self.items.values():
            payload.append(
                {
                    "sense_id": node.sense_id,
                    "term": node.term,
                    "domain": node.domain,
                    "confidence": node.confidence,
                    "context_hints": node.context_hints,
                    "supporting_evidence": node.supporting_evidence,
                    "retrieval_query": node.retrieval_query,
                    "mu_anchor": node.mu_anchor.tolist()
                    if node.mu_anchor is not None
                    else None,
                    "sigma_anchor": node.sigma_anchor.tolist()
                    if node.sigma_anchor is not None
                    else None,
                    "sigma_L_anchor": node.sigma_L_anchor.tolist()
                    if node.sigma_L_anchor is not None
                    else None,
                    "ttl_hours": node.ttl_hours,
                    "version": node.version,
                    "created_at": node.created_at,
                }
            )
        (path / "sense_cache.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "SenseCache":
        path = Path(directory) / "sense_cache.json"
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache = cls()
        nodes = [
            SenseNode(
                sense_id=item["sense_id"],
                term=item["term"],
                domain=item["domain"],
                confidence=item["confidence"],
                context_hints=item["context_hints"],
                supporting_evidence=item["supporting_evidence"],
                retrieval_query=item["retrieval_query"],
                mu_anchor=np.asarray(item["mu_anchor"], dtype=float)
                if item.get("mu_anchor") is not None
                else None,
                sigma_anchor=np.asarray(item["sigma_anchor"], dtype=float)
                if item.get("sigma_anchor") is not None
                else None,
                sigma_L_anchor=np.asarray(item["sigma_L_anchor"], dtype=float)
                if item.get("sigma_L_anchor") is not None
                else None,
                ttl_hours=item.get("ttl_hours", 24),
                version=item.get("version", 1),
                created_at=item.get("created_at", time.time()),
            )
            for item in payload
        ]
        cache.upsert(nodes)
        cache.prune_expired()
        return cache
