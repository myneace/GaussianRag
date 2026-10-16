"""Adaptive, flexible corpus generator for arbitrary raw text.

Converts messy, unformatted, or unorganized pasted text into clean, structured
corpora ready for Gaussian-Clef RAG indexing, evaluation, and search.
Supports both deterministic rule-based structuring and OpenAI-compatible
LLM-assisted semantic partitioning and character repair.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from gaussian_rag.creator.chunker import split_into_chunks
from gaussian_rag.creator.llm_assistant import OpenAICorpusAssistant
from gaussian_rag.creator.text_normalizer import TextNormalizer

if TYPE_CHECKING:
    from gaussian_rag.core.types import GaussianKnowledge
    from gaussian_rag.creator.corpus_cache import CorpusCache
    from gaussian_rag.creator.universal_ingestor import UniversalIngestor
    from gaussian_rag.rag.facade import GaussianRAG


@dataclass
class CorpusDocument:
    """A standardized structured document or passage in a generated corpus."""

    id: str
    title: str
    text: str
    section_number: int | None = None
    structure_type: str = "semantic_chunk"  # "numbered_section", "qa_pair", "dialogue", "markdown_section", "semantic_chunk"
    metadata: dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_estimate:
            self.token_estimate = max(1, len(self.text.split()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorpusGenerator:
    """Flexible corpus generator for converting arbitrary text into structured corpora.

    Parameters
    ----------
    normalizer:
        A ``TextNormalizer`` instance for pre-cleaning input.
    llm_assistant:
        Optional ``OpenAICorpusAssistant`` or boolean True to auto-enable OpenAI-compatible
        LLM assistance for repairing ambiguous text and extracting complex sections.
    chunk_size:
        Target token size for fallback sliding-window chunking.
    chunk_overlap:
        Token overlap between consecutive fallback chunks.
    """

    def __init__(
        self,
        *,
        normalizer: TextNormalizer | None = None,
        llm_assistant: OpenAICorpusAssistant | bool = False,
        chunk_size: int = 150,
        chunk_overlap: int = 20,
        cache: "CorpusCache | bool" = True,
    ) -> None:
        self.normalizer = normalizer or TextNormalizer()
        if isinstance(llm_assistant, bool):
            self.llm_assistant = OpenAICorpusAssistant() if llm_assistant else None
        else:
            self.llm_assistant = llm_assistant

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Cache is enabled by default; pass False to disable or a CorpusCache instance.
        if isinstance(cache, bool):
            if cache:
                from gaussian_rag.creator.corpus_cache import CorpusCache  # noqa: PLC0415
                self._cache: "CorpusCache | None" = CorpusCache()
            else:
                self._cache = None
        else:
            self._cache = cache

    # ----------------------------------------------------------------------
    # Core Generation
    # ----------------------------------------------------------------------

    def generate(
        self,
        raw_text: str,
        *,
        title: str | None = None,
        doc_id_prefix: str = "doc",
        use_llm: bool | None = None,
        use_cache: bool = True,
    ) -> list[CorpusDocument]:
        """Generate structured ``CorpusDocument``s from *raw_text*.

        Parameters
        ----------
        raw_text:
            Arbitrary, potentially messy pasted text.
        title:
            Optional title or topic name for the corpus.
        doc_id_prefix:
            ID prefix for created documents.
        use_llm:
            If True, uses the OpenAI-compatible assistant for text repair
            and structure extraction. If None, auto-uses if LLM is configured.
        use_cache:
            If True (default) and a ``CorpusCache`` was supplied at construction
            time, a cache lookup is performed first and the result is returned
            immediately on a hit -- skipping normalization and any AI API calls.
            Set to False to force full re-processing (e.g. after source changes).
        """
        if not raw_text or not raw_text.strip():
            return []

        # --- Cache lookup (idempotency guard) ---
        if use_cache and self._cache is not None:
            hit, cached_docs = self._cache.lookup(raw_text)
            if hit and cached_docs is not None:
                return cached_docs

        # 1. Tier 1 Rule-based cleaning (+ optional LLM character repair)
        cleaned_text = self.normalizer.clean(raw_text, use_llm=use_llm)
        if not cleaned_text:
            return []

        doc_title = title or self._infer_title(cleaned_text) or "Document"

        # 2. Check if LLM-assisted sectioning is enabled and succeeds
        should_use_llm = (use_llm is True) or (use_llm is None and self.llm_assistant and self.llm_assistant.is_configured)
        if should_use_llm and self.llm_assistant:
            llm_sections = self.llm_assistant.structure_into_sections(cleaned_text, default_title=doc_title)
            if llm_sections:
                docs = self._build_from_llm_records(llm_sections, doc_id_prefix, doc_title)
                if use_cache and self._cache is not None:
                    self._cache.store(raw_text, docs)
                return docs

        # 3. Rule-based structure detection
        # Attempt A: Numbered sections (Laws, Sections, Articles, Rules, Chapters)
        numbered_docs = self._try_parse_numbered_sections(cleaned_text, doc_id_prefix, doc_title)
        if numbered_docs:
            if use_cache and self._cache is not None:
                self._cache.store(raw_text, numbered_docs)
            return numbered_docs

        # Attempt B: Q&A / FAQ pairs
        qa_docs = self._try_parse_qa_pairs(cleaned_text, doc_id_prefix, doc_title)
        if qa_docs:
            if use_cache and self._cache is not None:
                self._cache.store(raw_text, qa_docs)
            return qa_docs

        # Attempt C: Markdown headings
        md_docs = self._try_parse_markdown_headings(cleaned_text, doc_id_prefix, doc_title)
        if md_docs:
            if use_cache and self._cache is not None:
                self._cache.store(raw_text, md_docs)
            return md_docs

        # Attempt D: Fallback to semantic paragraph windowed chunking
        fallback_docs = self._fallback_chunking(cleaned_text, doc_id_prefix, doc_title)
        if use_cache and self._cache is not None:
            self._cache.store(raw_text, fallback_docs)
        return fallback_docs

    # ----------------------------------------------------------------------
    # Structural parsers
    # ----------------------------------------------------------------------

    @staticmethod
    def _infer_title(text: str) -> str | None:
        """Infer an overarching title from the first lines of text."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        first = lines[0]
        # Match markdown header or short uppercase/title line
        m = re.match(r"^#+\s+(.+)$", first)
        if m:
            return m.group(1).strip()
        if len(first) < 80 and not first.endswith((".", ":", ";")):
            return first
        return None

    def _build_from_llm_records(
        self, records: list[dict[str, Any]], doc_id_prefix: str, doc_title: str
    ) -> list[CorpusDocument]:
        """Convert LLM-extracted section dicts into CorpusDocument objects."""
        docs: list[CorpusDocument] = []
        for i, rec in enumerate(records):
            sec_id = rec.get("id") or f"{doc_id_prefix}-{i + 1}"
            sec_title = rec.get("title") or f"{doc_title} Part {i + 1}"
            sec_text = rec.get("text", "").strip()
            if not sec_text:
                continue
            sec_num = rec.get("section_number")
            meta = rec.get("metadata", {})
            meta["corpus_title"] = doc_title
            meta["source"] = "llm_structured"
            docs.append(
                CorpusDocument(
                    id=str(sec_id),
                    title=str(sec_title),
                    text=sec_text,
                    section_number=int(sec_num) if sec_num is not None else None,
                    structure_type="llm_structured",
                    metadata=meta,
                )
            )
        return docs

    def _try_parse_numbered_sections(
        self, text: str, prefix: str, doc_title: str
    ) -> list[CorpusDocument] | None:
        """Parse documents split by 'Law 1:', 'Section 1:', 'Article IV:', etc."""
        # Match e.g. "Law 1: Never Outshine the Master" or "Section 2. Title"
        pattern = re.compile(
            r"(?m)^(?:#{1,3}\s*)?(?:(Law|Rule|Section|Chapter|Article|Item)\s+([0-9IVXLCDM]+)(?:[:.\s-]+([^\n]+))?)"
        )
        matches = list(pattern.finditer(text))
        if len(matches) < 2:
            return None

        docs: list[CorpusDocument] = []
        for i, m in enumerate(matches):
            kind = m.group(1)
            raw_num = m.group(2)
            sub_title = (m.group(3) or "").strip()
            
            # Numeric conversion if possible
            try:
                num_val: int | None = int(raw_num)
            except ValueError:
                num_val = i + 1

            start_idx = m.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_content = text[start_idx:end_idx].strip()
            if not sec_content:
                continue

            full_title = f"{kind} {raw_num}: {sub_title}" if sub_title else f"{kind} {raw_num}"
            docs.append(
                CorpusDocument(
                    id=f"{prefix}-{raw_num.lower()}",
                    title=full_title,
                    text=f"{full_title}\n\n{sec_content}",
                    section_number=num_val,
                    structure_type="numbered_section",
                    metadata={
                        "corpus_title": doc_title,
                        "section_type": kind.lower(),
                        "section_raw_number": raw_num,
                    },
                )
            )
        return docs if docs else None

    def _try_parse_qa_pairs(
        self, text: str, prefix: str, doc_title: str
    ) -> list[CorpusDocument] | None:
        """Parse text structured as Question & Answer pairs."""
        qa_pattern = re.compile(
            r"(?is)(?:^|\n+)(?:Q(?:uestion)?[:.]\s*(.*?))\s*\n+(?:A(?:nswer)?[:.]\s*(.*?))(?=(?:\n+Q(?:uestion)?[:.]|\Z))"
        )
        matches = list(qa_pattern.finditer(text))
        if len(matches) < 2:
            return None

        docs: list[CorpusDocument] = []
        for i, m in enumerate(matches):
            q_text = m.group(1).strip()
            a_text = m.group(2).strip()
            if not q_text or not a_text:
                continue
            title = q_text if len(q_text) <= 80 else q_text[:77] + "..."
            combined = f"Question: {q_text}\nAnswer: {a_text}"
            docs.append(
                CorpusDocument(
                    id=f"{prefix}-qa-{i + 1}",
                    title=title,
                    text=combined,
                    section_number=i + 1,
                    structure_type="qa_pair",
                    metadata={
                        "corpus_title": doc_title,
                        "question": q_text,
                    },
                )
            )
        return docs if docs else None

    def _try_parse_markdown_headings(
        self, text: str, prefix: str, doc_title: str
    ) -> list[CorpusDocument] | None:
        """Parse documents split by Markdown headings (## or ###)."""
        heading_re = re.compile(r"(?m)^(#{1,4})\s+(.+)$")
        matches = list(heading_re.finditer(text))
        if len(matches) < 2:
            return None

        docs: list[CorpusDocument] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            sec_title = m.group(2).strip()
            start_idx = m.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_idx:end_idx].strip()
            if not content:
                continue

            slug = re.sub(r"\W+", "-", sec_title).strip("-").lower()
            docs.append(
                CorpusDocument(
                    id=f"{prefix}-{slug or (i + 1)}",
                    title=sec_title,
                    text=f"{sec_title}\n\n{content}",
                    section_number=i + 1,
                    structure_type="markdown_section",
                    metadata={
                        "corpus_title": doc_title,
                        "heading_level": level,
                    },
                )
            )
        return docs if docs else None

    def _fallback_chunking(
        self, text: str, prefix: str, doc_title: str
    ) -> list[CorpusDocument]:
        """Fallback for unformatted text blobs: split into semantic paragraphs/chunks."""
        raw_chunks = split_into_chunks(
            text, max_tokens=self.chunk_size, overlap=self.chunk_overlap
        )
        docs: list[CorpusDocument] = []
        for i, chunk in enumerate(raw_chunks):
            # Extract first sentence or 8 words as chunk title
            first_sentence = re.split(r"[.!?\n]", chunk)[0].strip()
            chunk_title = first_sentence if len(first_sentence) < 60 else " ".join(chunk.split()[:8]) + "..."
            docs.append(
                CorpusDocument(
                    id=f"{prefix}-part-{i + 1}",
                    title=chunk_title or f"{doc_title} Part {i + 1}",
                    text=chunk,
                    section_number=i + 1,
                    structure_type="semantic_chunk",
                    metadata={
                        "corpus_title": doc_title,
                        "chunk_index": i,
                    },
                )
            )
        return docs

    # ----------------------------------------------------------------------
    # Exporters & Serializers
    # ----------------------------------------------------------------------

    @staticmethod
    def to_jsonl(documents: list[CorpusDocument], output_path: str | Path) -> Path:
        """Export documents to JSON Lines format (one JSON object per line)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def to_markdown(
        documents: list[CorpusDocument], output_path: str | Path, title: str | None = None
    ) -> Path:
        """Export documents to a single clean, structured Markdown file with headings."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        main_title = title or (documents[0].metadata.get("corpus_title") if documents else "Corpus")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {main_title}\n\n")
            for doc in documents:
                f.write(f"## {doc.title}\n\n")
                f.write(f"{doc.text}\n\n")
                f.write("---\n\n")
        return path

    @staticmethod
    def to_text(documents: list[CorpusDocument], output_path: str | Path) -> Path:
        """Export documents to plain text with standardized section separators."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(f"=== {doc.title} ===\n\n")
                f.write(f"{doc.text}\n\n")
        return path

    # ----------------------------------------------------------------------
    # Direct Ingestion into Gaussian RAG
    # ----------------------------------------------------------------------

    @staticmethod
    def load_into_rag(
        documents: list[CorpusDocument], rag_instance: GaussianRAG
    ) -> list[GaussianKnowledge]:
        """Directly index the generated corpus documents into a GaussianRAG instance.

        Each document is ingested into both the dense covariance KnowledgeStore
        and the SQLite FTS5 lexical index in lockstep.
        """
        records = [
            {
                "id": doc.id,
                "text": doc.text,
                "title": doc.title,
                "section_number": doc.section_number,
                "structure_type": doc.structure_type,
                **doc.metadata,
            }
            for doc in documents
        ]
        return rag_instance.add_records(records, text_field="text", id_field="id")
