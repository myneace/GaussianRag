"""Universal format-agnostic text ingestion engine for Gaussian RAG.

Supports arbitrary text formats:
  - Raw strings / lists of strings
  - Markdown documents (with heading hierarchy as metadata breadcrumbs)
  - JSON / JSONL structured records
  - CSV tabular data
  - Plain-text files
  - Directory trees (recursive, auto-detecting format by extension)

Every ingested chunk produces a ``GaussianKnowledge`` node that is
simultaneously indexed in the dense ``KnowledgeStore`` AND the SQLite
FTS5 lexical store so that the hybrid retriever never drops items.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from io import StringIO
from pathlib import Path
from typing import Any

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import GaussianKnowledge
from gaussian_rag.creator.chunker import split_into_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _parse_markdown_sections(
    markdown_text: str, document_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """Split a Markdown document into (text, metadata) pairs.

    Each section inherits a ``breadcrumb`` built from the heading hierarchy
    leading to it.  The section text is everything between two headings.

    Returns a list of ``(chunk_text, extra_metadata)`` tuples ready for
    Gaussian embedding.
    """
    lines = markdown_text.splitlines(keepends=True)
    sections: list[tuple[str, dict[str, Any]]] = []

    # Track heading stack: list of (level, title) from h1 down
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading_meta: dict[str, Any] = {}

    def _flush(lines_buf: list[str], meta: dict[str, Any]) -> None:
        text = "".join(lines_buf).strip()
        if text:
            sections.append((text, meta.copy()))

    for line in lines:
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m:
            # Flush what we have so far
            _flush(current_lines, current_heading_meta)
            current_lines = []

            level = len(m.group(1))
            title = m.group(2).strip()

            # Trim the stack to current depth
            heading_stack = [(l, t) for l, t in heading_stack if l < level]
            heading_stack.append((level, title))

            breadcrumb = " > ".join(t for _, t in heading_stack)
            current_heading_meta = {
                "heading_level": level,
                "heading_title": title,
                "breadcrumb": breadcrumb,
                "document_id": document_id,
            }
        else:
            current_lines.append(line)

    _flush(current_lines, current_heading_meta)
    return sections


def _records_to_texts(
    records: list[dict[str, Any]],
    text_field: str,
    id_field: str,
    document_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Extract text + metadata from a list of dictionaries."""
    result: list[tuple[str, dict[str, Any]]] = []
    for i, rec in enumerate(records):
        text = str(rec.get(text_field, "")).strip()
        if not text:
            continue
        meta = {k: v for k, v in rec.items() if k != text_field}
        meta["document_id"] = str(rec.get(id_field, f"{document_id}-{i}"))
        meta["record_index"] = i
        result.append((text, meta))
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UniversalIngestor:
    """Format-agnostic ingestion engine.

    Parameters
    ----------
    embedder:
        A ``GaussianEmbedder`` used to encode each chunk into Gaussian space.
    store:
        The ``KnowledgeStore`` to write chunks into (both dense + lexical).
    chunk_size:
        Target maximum chunk size in whitespace-separated tokens (passed to
        ``split_into_chunks``).
    chunk_overlap:
        Token overlap between consecutive chunks.

    Example
    -------
    >>> from gaussian_rag import GaussianRAG
    >>> rag = GaussianRAG.create()
    >>> rag.add_texts(["Law 1: Never outshine the master."])
    >>> results = rag.search("How to act around superiors?")
    """

    def __init__(
        self,
        embedder: GaussianEmbedder,
        store: KnowledgeStore,
        *,
        chunk_size: int = 128,
        chunk_overlap: int = 16,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Low-level: chunk → embed → store
    # ------------------------------------------------------------------

    def _encode_and_store(
        self,
        text: str,
        document_id: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> list[GaussianKnowledge]:
        """Chunk *text*, embed each chunk, and add to ``self.store``."""
        chunks = split_into_chunks(
            text, max_tokens=self.chunk_size, overlap=self.chunk_overlap
        )
        nodes: list[GaussianKnowledge] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-chunk-{i}"
            metadata: dict[str, Any] = {
                "document_id": document_id,
                "chunk_index": i,
                **(extra_meta or {}),
            }
            node = self.embedder.encode(
                chunk,
                item_id=chunk_id,
                metadata=metadata,
            )
            nodes.append(node)

        self.store.add(nodes)
        logger.debug(
            "UniversalIngestor: stored %d chunks from document '%s'",
            len(nodes),
            document_id,
        )
        return nodes

    # ------------------------------------------------------------------
    # Public ingestion methods
    # ------------------------------------------------------------------

    def ingest_texts(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        *,
        document_id_prefix: str = "text",
    ) -> list[GaussianKnowledge]:
        """Ingest a list of raw text strings.

        Parameters
        ----------
        texts:
            List of plain-text strings.
        metadata:
            Optional per-text metadata dicts (must be same length as *texts*).
        document_id_prefix:
            Prefix for generated document IDs.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created across all texts.
        """
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata must have the same length as texts")

        all_nodes: list[GaussianKnowledge] = []
        for i, text in enumerate(texts):
            doc_id = f"{document_id_prefix}-{i}"
            extra = metadata[i] if metadata else None
            nodes = self._encode_and_store(text, doc_id, extra)
            all_nodes.extend(nodes)
        return all_nodes

    def ingest_markdown(
        self,
        markdown_text: str,
        document_id: str = "doc",
    ) -> list[GaussianKnowledge]:
        """Ingest a Markdown document, preserving heading hierarchy as metadata.

        Each section under a heading becomes a separate embedded passage, with
        ``breadcrumb``, ``heading_title``, and ``heading_level`` stored in the
        node metadata.

        Parameters
        ----------
        markdown_text:
            Raw Markdown string.
        document_id:
            Logical document identifier (used as an ID prefix).

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        sections = _parse_markdown_sections(markdown_text, document_id)
        if not sections:
            # No headings — treat entire document as a single passage
            sections = [(markdown_text, {"document_id": document_id})]

        all_nodes: list[GaussianKnowledge] = []
        for sec_idx, (section_text, meta) in enumerate(sections):
            heading_slug = re.sub(r"\W+", "-", meta.get("heading_title", f"sec{sec_idx}")).strip("-").lower()
            doc_id = f"{document_id}-{heading_slug or sec_idx}"
            nodes = self._encode_and_store(section_text, doc_id, meta)
            all_nodes.extend(nodes)
        return all_nodes

    def ingest_records(
        self,
        records: list[dict[str, Any]],
        *,
        text_field: str = "text",
        id_field: str = "id",
        document_id: str = "records",
    ) -> list[GaussianKnowledge]:
        """Ingest structured records (JSON objects / dicts).

        Each record must contain a text field (``text_field``) which is the
        passage to embed.  All other fields are preserved in node metadata.

        Parameters
        ----------
        records:
            List of dicts representing structured data.
        text_field:
            Key whose value contains the passage text.
        id_field:
            Key to use as the record's document ID.
        document_id:
            Fallback document ID prefix when *id_field* is absent.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        pairs = _records_to_texts(records, text_field, id_field, document_id)
        all_nodes: list[GaussianKnowledge] = []
        for text, meta in pairs:
            doc_id = str(meta.get("document_id", document_id))
            nodes = self._encode_and_store(text, doc_id, meta)
            all_nodes.extend(nodes)
        return all_nodes

    def ingest_file(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> list[GaussianKnowledge]:
        """Ingest a file, auto-detecting format by extension.

        Supported extensions:

        =========  ===========================================================
        Extension  Behaviour
        =========  ===========================================================
        ``.md``    Parsed as Markdown (``ingest_markdown``)
        ``.txt``   Treated as plain text (``ingest_texts`` with one string)
        ``.json``  Parsed as a JSON array of records (``ingest_records``)
        ``.jsonl``  Parsed as JSON Lines records (``ingest_records``)
        ``.csv``   Parsed as CSV with header row (``ingest_records``)
        other      Read as plain UTF-8 text
        =========  ===========================================================

        Additional keyword arguments are forwarded to the appropriate
        lower-level method.

        Parameters
        ----------
        file_path:
            Path to the file.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        document_id = kwargs.pop("document_id", path.stem)

        if suffix == ".md":
            text = path.read_text(encoding="utf-8")
            return self.ingest_markdown(text, document_id=document_id)

        if suffix in {".txt", ""}:
            text = path.read_text(encoding="utf-8")
            return self.ingest_texts([text], document_id_prefix=document_id)

        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                records = raw
            elif isinstance(raw, dict):
                # Try to find a list value
                records = next(
                    (v for v in raw.values() if isinstance(v, list)), [raw]
                )
            else:
                records = [{"text": str(raw)}]
            text_field = kwargs.get("text_field", "text")
            id_field = kwargs.get("id_field", "id")
            return self.ingest_records(
                records,
                text_field=text_field,
                id_field=id_field,
                document_id=document_id,
            )

        if suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            records = [
                json.loads(line) for line in lines if line.strip()
            ]
            text_field = kwargs.get("text_field", "text")
            id_field = kwargs.get("id_field", "id")
            return self.ingest_records(
                records,
                text_field=text_field,
                id_field=id_field,
                document_id=document_id,
            )

        if suffix == ".csv":
            raw = path.read_text(encoding="utf-8")
            reader = csv.DictReader(StringIO(raw))
            records_csv: list[dict[str, Any]] = [dict(row) for row in reader]
            text_field = kwargs.get("text_field", "text")
            id_field = kwargs.get("id_field", "id")
            return self.ingest_records(
                records_csv,
                text_field=text_field,
                id_field=id_field,
                document_id=document_id,
            )

        # Unknown extension — treat as plain text
        logger.warning(
            "UniversalIngestor: unknown extension '%s', reading as plain text", suffix
        )
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_texts([text], document_id_prefix=document_id)

    def ingest_directory(
        self,
        dir_path: str | Path,
        *,
        recursive: bool = True,
        glob_pattern: str = "*",
    ) -> list[GaussianKnowledge]:
        """Ingest all matching files inside *dir_path*.

        Parameters
        ----------
        dir_path:
            Directory to walk.
        recursive:
            If ``True`` (default), walk subdirectories recursively.
        glob_pattern:
            Glob pattern to filter files (default: ``"*"`` — all files).

        Returns
        -------
        All ``GaussianKnowledge`` nodes created across all files.
        """
        base = Path(dir_path)
        if not base.is_dir():
            raise NotADirectoryError(f"Not a directory: {base}")

        searcher = base.rglob if recursive else base.glob
        all_nodes: list[GaussianKnowledge] = []
        for fp in sorted(searcher(glob_pattern)):
            if not fp.is_file():
                continue
            try:
                nodes = self.ingest_file(fp)
                all_nodes.extend(nodes)
                logger.info("UniversalIngestor: ingested '%s' (%d nodes)", fp, len(nodes))
            except Exception as exc:  # noqa: BLE001
                logger.warning("UniversalIngestor: skipping '%s': %s", fp, exc)
        return all_nodes
