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

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "around", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "manage", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def sanitize_fts5_query(query: str) -> str:
    """Sanitize user query string for SQLite FTS5 MATCH syntax.

    Preserves Boolean operators (OR, AND, NOT) while sanitizing individual
    search tokens and wrapping them in quotes so punctuation or special symbols
    cannot trigger SQLite syntax errors.
    """
    if not query or not query.strip():
        return '""'

    boolean_ops = {"OR", "AND", "NOT"}
    raw_tokens = query.strip().split()
    formatted_tokens: list[str] = []

    for token in raw_tokens:
        clean_token = token.strip()
        if not clean_token:
            continue
        if clean_token in boolean_ops:
            # Only append operator if preceded by a valid token and not duplicating
            if formatted_tokens and formatted_tokens[-1] not in boolean_ops:
                formatted_tokens.append(clean_token)
        else:
            sub_words = re.findall(r"\w+", clean_token)
            for w in sub_words:
                if w:
                    formatted_tokens.append(f'"{w}"')

    # Remove trailing operators
    while formatted_tokens and formatted_tokens[-1] in boolean_ops:
        formatted_tokens.pop()

    # Remove leading operators
    while formatted_tokens and formatted_tokens[0] in boolean_ops:
        formatted_tokens.pop(0)

    if not formatted_tokens:
        return '""'

    return " ".join(formatted_tokens)


def extract_query_keywords(query: str, max_terms: int = 10) -> str:
    """Extract informative keywords from a conversational query for FTS5 search.

    Filters conversational stopwords and punctuation, prioritizing proper nouns and
    salient domain terms, then joins them with the FTS5 Boolean OR operator.
    """
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
    filtered = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 1]

    # Prioritize capitalized/proper nouns, then longer words
    prioritized = sorted(
        filtered,
        key=lambda w: (1 if w[0].isupper() else 0, len(w)),
        reverse=True,
    )

    # Deduplicate while preserving priority order
    seen: set[str] = set()
    selected: list[str] = []
    for w in prioritized:
        norm = w.lower()
        if norm not in seen:
            seen.add(norm)
            selected.append(w)
            if len(selected) >= max_terms:
                break

    if not selected:
        selected = [w for w in words[:max_terms] if len(w) > 1]

    return " OR ".join(selected) if selected else '""'


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
