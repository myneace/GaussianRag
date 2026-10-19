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

    def search_lexical(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.lexical is not None:
            return self.lexical.search(query, top_k=top_k)
        return []

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        items_list = list(self.items.values())
        payload = []
        ids = []
        mus = []
        sigma_diags = []
        sigma_Ls = []

        for item in items_list:
            payload.append(
                {
                    "id": item.id,
                    "text": item.text,
                    "metadata": item.metadata,
                }
            )
            ids.append(item.id)
            mus.append(item.mu)
            sigma_diags.append(item.sigma_diag)
            sigma_Ls.append(item.sigma_L)

        # Save lightweight knowledge.json
        (path / "knowledge.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        # Save high-dimensional embeddings in binary NumPy format
        if items_list:
            np.savez_compressed(
                path / "embeddings.npz",
                ids=np.array(ids),
                mu=np.stack(mus),
                sigma_diag=np.stack(sigma_diags),
                sigma_L=np.stack(sigma_Ls),
            )

        if self.lexical is not None:
            self.lexical.close()

        # Sync embedded SQLite FTS5 index
        db_path = path / "lexical.db"
        lex = LexicalStore(db_path)
        lex.clear()
        passages = [
            (
                item.id,
                str(item.metadata.get("document_id", item.metadata.get("source_path", "default"))),
                item.text,
            )
            for item in items_list
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
            self.lexical = None

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeStore":
        path = Path(directory)
        json_path = path / "knowledge.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        npz_path = path / "embeddings.npz"
        npz_data = None
        if npz_path.exists():
            npz_data = np.load(npz_path, allow_pickle=False)

        db_path = path / "lexical.db"
        if db_path.exists():
            lex = LexicalStore(db_path)
        else:
            lex = LexicalStore(":memory:")
        store = cls(lexical=lex)

        if npz_data is not None:
            ids_arr = list(npz_data["ids"])
            mu_arr = npz_data["mu"]
            sigma_diag_arr = npz_data["sigma_diag"]
            sigma_L_arr = npz_data["sigma_L"]
            id_to_idx = {str(cid): idx for idx, cid in enumerate(ids_arr)}

            for item in payload:
                item_id = item["id"]
                idx = id_to_idx.get(item_id)
                if idx is not None:
                    mu = np.asarray(mu_arr[idx], dtype=float)
                    sigma_diag = np.asarray(sigma_diag_arr[idx], dtype=float)
                    sigma_L = np.asarray(sigma_L_arr[idx], dtype=float)
                else:
                    # Fallback to item JSON keys if missing in npz
                    mu = np.asarray(item.get("mu", []), dtype=float)
                    sigma_diag = np.asarray(item.get("sigma_diag", []), dtype=float)
                    sigma_L = np.asarray(item.get("sigma_L", []), dtype=float)

                store.items[item_id] = GaussianKnowledge(
                    id=item_id,
                    text=item["text"],
                    mu=mu,
                    sigma_diag=sigma_diag,
                    sigma_L=sigma_L,
                    metadata=item.get("metadata", {}),
                )
        else:
            # Legacy knowledge.json format where mu, sigma_diag, sigma_L are embedded in JSON
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
