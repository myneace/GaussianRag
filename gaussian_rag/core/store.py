from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.core.lexical_store import LexicalStore


@dataclass(slots=True)
class KnowledgeStore:
    items: dict[str, GaussianKnowledge] = field(default_factory=dict)
    lexical: LexicalStore | None = field(default_factory=lambda: LexicalStore(":memory:"))

    def add(self, chunks: list[GaussianKnowledge]) -> None:
        passages = []
        for chunk in chunks:
            self.items[chunk.id] = chunk
            doc_id = (
                str(chunk.metadata.get("document_id", chunk.metadata.get("source_path", "default")))
                if chunk.metadata
                else "default"
            )
            passages.append((chunk.id, doc_id, chunk.text))
        if self.lexical is not None and passages:
            self.lexical.add_passages(passages)

    def get(self, ids: list[str]) -> list[GaussianKnowledge]:
        return [self.items[item_id] for item_id in ids if item_id in self.items]

    def values(self) -> list[GaussianKnowledge]:
        return list(self.items.values())

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items.values())

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

    def count_lexical(self) -> int:
        """Return total count of indexed passages in the active lexical store."""
        if self.lexical is not None:
            return self.lexical.count()
        return 0

    def close(self) -> None:
        """Close SQLite lexical store connection if open."""
        if self.lexical is not None:
            self.lexical.close()

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeStore":
        path = Path(directory)
        json_path = path / "knowledge.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        db_path = path / "lexical.db"
        if db_path.exists():
            lex = LexicalStore(db_path)
        else:
            lex = LexicalStore(":memory:")
        store = cls(lexical=lex)

        for item in payload:
            store.items[item["id"]] = GaussianKnowledge(
                id=item["id"],
                text=item["text"],
                mu=np.asarray(item.get("mu", []), dtype=float),
                sigma_diag=np.asarray(item.get("sigma_diag", []), dtype=float),
                sigma_L=np.asarray(item.get("sigma_L", []), dtype=float),
                metadata=item.get("metadata", {}),
            )

        if not db_path.exists() and store.items:
            passages = [
                (
                    item.id,
                    str(item.metadata.get("document_id", item.metadata.get("source_path", "default"))),
                    item.text,
                )
                for item in store.items.values()
            ]
            store.lexical.add_passages(passages)
        return store
