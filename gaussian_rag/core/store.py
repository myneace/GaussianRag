from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.core.lexical_store import LexicalStore


@dataclass(slots=True)
class KnowledgeStore:
    items: dict[str, GaussianKnowledge] = field(default_factory=dict)
    lexical: LexicalStore | None = field(default=None)

    def add(self, chunks: list[GaussianKnowledge]) -> None:
        for chunk in chunks:
            self.items[chunk.id] = chunk

    def get(self, ids: list[str]) -> list[GaussianKnowledge]:
        return [self.items[item_id] for item_id in ids if item_id in self.items]

    def values(self) -> list[GaussianKnowledge]:
        return list(self.items.values())

    def search_lexical(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.lexical is not None:
            return self.lexical.search(query, top_k=top_k)
        return []

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = []
        for item in self.items.values():
            payload.append(
                {
                    "id": item.id,
                    "text": item.text,
                    "mu": item.mu.tolist(),
                    "sigma_diag": item.sigma_diag.tolist(),
                    "sigma_L": item.sigma_L.tolist(),
                    "metadata": item.metadata,
                }
            )
        (path / "knowledge.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        if self.lexical is not None:
            self.lexical.close()

        # Sync embedded SQLite FTS5 index
        db_path = path / "lexical.db"
        lex = LexicalStore(db_path)
        passages = [
            (
                item.id,
                str(item.metadata.get("document_id", item.metadata.get("source_path", "default"))),
                item.text,
            )
            for item in self.items.values()
        ]
        lex.add_passages(passages)
        self.lexical = lex

    def close(self) -> None:
        """Close SQLite lexical store connection if open."""
        if self.lexical is not None:
            self.lexical.close()
            self.lexical = None

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeStore":
        path = Path(directory) / "knowledge.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        store.add(
            [
                GaussianKnowledge(
                    id=item["id"],
                    text=item["text"],
                    mu=np.asarray(item["mu"], dtype=float),
                    sigma_diag=np.asarray(item["sigma_diag"], dtype=float),
                    sigma_L=np.asarray(item["sigma_L"], dtype=float),
                    metadata=item.get("metadata", {}),
                )
                for item in payload
            ]
        )
        db_path = Path(directory) / "lexical.db"
        if db_path.exists():
            store.lexical = LexicalStore(db_path)
        return store
