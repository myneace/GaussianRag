"""lexical_store.py — Embedded SQLite FTS5 lexical indexing for sub-2ms BM25 candidate retrieval.

Designed for lockstep dual-store ingestion alongside Gaussian KnowledgeStore:
- Every ingested passage is indexed with its exact chunk_id.
- Operates in WAL mode with robust busy timeouts for concurrent read/write.
- BM25 rank evaluation returns top-k candidate IDs in < 2ms.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_fts5_query(query: str) -> str:
    """Sanitize user query string for SQLite FTS5 MATCH syntax.

    Removes special operators and punctuation that could cause FTS5 syntax errors,
    keeping alphanumeric words and joining with whitespace (implicit AND or phrase).
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query).strip()
    words = [w for w in cleaned.split() if len(w) > 0]
    if not words:
        return '""'
    # Format each term as a safe literal token
    return ' '.join(f'"{w}"' for w in words)


class LexicalStore:
    """Embedded SQLite FTS5 lexical passage index.

    Args:
        db_path: Path to the SQLite database file, or ':memory:' for transient index.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
        else:
            self._conn = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                timeout=10.0,
            )

        self._conn.row_factory = sqlite3.Row
        self._init_pragmas_and_schema()

    def _init_pragmas_and_schema(self) -> None:
        cursor = self._conn.cursor()
        if self.db_path != ":memory:":
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")

        # Create FTS5 virtual table
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                text,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        self._conn.commit()

    def add_passages(self, passages: list[tuple[str, str, str]]) -> int:
        """Add a batch of passages to the FTS5 index.

        Args:
            passages: List of tuples (chunk_id, document_id, text).

        Returns:
            Number of passages inserted.
        """
        if not passages:
            return 0

        cursor = self._conn.cursor()
        cursor.executemany(
            "INSERT INTO passages_fts(chunk_id, document_id, text) VALUES (?, ?, ?);",
            passages,
        )
        self._conn.commit()
        return len(passages)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Perform sub-2ms BM25 ranking against the FTS5 index.

        Args:
            query: User search query.
            top_k: Maximum number of candidates to return.

        Returns:
            List of (chunk_id, bm25_score) tuples, ordered from lowest (best) to highest score.
        """
        sanitized = sanitize_fts5_query(query)
        if sanitized == '""':
            return []

        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """
                SELECT chunk_id, bm25(passages_fts) as rank
                FROM passages_fts
                WHERE passages_fts MATCH ?
                ORDER BY rank
                LIMIT ?;
                """,
                (sanitized, max(1, top_k)),
            )
            return [(row["chunk_id"], float(row["rank"])) for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 match failed for query %r: %s", query, e)
            return []

    def get_passage_text(self, chunk_id: str) -> str | None:
        """Retrieve the raw text for a specific chunk ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT text FROM passages_fts WHERE chunk_id = ? LIMIT 1;", (chunk_id,)
        )
        row = cursor.fetchone()
        return row["text"] if row else None

    def count(self) -> int:
        """Return total number of indexed passages."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT count(*) as cnt FROM passages_fts;")
        return int(cursor.fetchone()["cnt"])

    def close(self) -> None:
        """Close SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "LexicalStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
