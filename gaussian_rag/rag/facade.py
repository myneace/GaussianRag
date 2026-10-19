"""High-level GaussianRAG facade â€” unified public API for the library.

This module provides a single entry-point class ``GaussianRAG`` that wires
together all internal sub-systems:

* Gaussian embedding (``GaussianEmbedder``)
* Knowledge store + SQLite FTS5 lexical index (``KnowledgeStore``)
* Hybrid retriever with taxonomy routing (``GaussianRetriever``)
* Optional Cloudflare Clef passage-gating (``ClefPassageGate``)
* Optional LLM answer generation (``ProviderBackedGenerator``)
* Format-agnostic text ingestion (``UniversalIngestor``)

Quick-start (3 lines)::

    from gaussian_rag import GaussianRAG

    rag = GaussianRAG.create()
    rag.add_texts(["Never outshine the master.", "Always say less than necessary."])
    results = rag.search("How should I behave around powerful people?")

Works standalone â€” no cloud tokens required.  Activate Cloudflare Clef and
LLM generation by passing ``use_clef=True`` and supplying API credentials.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import (
    GaussianKnowledge,
    QueryRepresentation,
    RetrievedChunk,
)
from gaussian_rag.creator.universal_ingestor import UniversalIngestor
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.retrieval.taxonomy_router import TaxonomyRouter

logger = logging.getLogger(__name__)

_ST_MODEL_CACHE: dict[str, Any] = {}


def _get_st_model(model_name: str) -> Any:
    if model_name not in _ST_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name]


# Lazy imports so cloud dependencies remain optional
_CLEF_CLIENT_CLS: type | None = None
_CLEF_GATE_CLS: type | None = None
_GENERATOR_CLS: type | None = None


def _get_clef_client():
    global _CLEF_CLIENT_CLS
    if _CLEF_CLIENT_CLS is None:
        from gaussian_rag.rag.disambiguation.clef_client import ClefClient  # type: ignore[import]
        _CLEF_CLIENT_CLS = ClefClient
    return _CLEF_CLIENT_CLS


def _get_clef_gate():
    global _CLEF_GATE_CLS
    if _CLEF_GATE_CLS is None:
        from gaussian_rag.rag.retrieval.clef_passage_gate import ClefPassageGate  # type: ignore[import]
        _CLEF_GATE_CLS = ClefPassageGate
    return _CLEF_GATE_CLS


class GaussianRAG:
    """Unified high-level facade for Gaussian RAG.

    Instantiate via the factory methods ``GaussianRAG.create()`` or
    ``GaussianRAG.load()``.  Do **not** call ``__init__`` directly.
    """

    def __init__(
        self,
        embedder: GaussianEmbedder,
        store: KnowledgeStore,
        retriever: GaussianRetriever,
        ingestor: UniversalIngestor,
        *,
        clef_client: Any | None = None,
        passage_gate: Any | None = None,
        generator: Any | None = None,
        save_dir: Path | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._retriever = retriever
        self._ingestor = ingestor
        self._clef_client = clef_client
        self._passage_gate = passage_gate
        self._generator = generator
        self._save_dir = save_dir

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        dimension: int = 384,
        save_dir: str | Path | None = None,
        use_clef: bool = False,
        cloudflare_account_id: str | None = None,
        cloudflare_api_token: str | None = None,
        use_generator: bool = False,
        generator_api_key: str | None = None,
        generator_base_url: str | None = None,
        generator_model: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        sigma_mode: str = "heuristic",
        chunk_size: int = 128,
        chunk_overlap: int = 16,
    ) -> "GaussianRAG":
        """Create a new GaussianRAG instance with an empty knowledge store.

        Parameters
        ----------
        dimension:
            Embedding dimension.  Default ``384`` (matches ``all-MiniLM-L6-v2``).
        save_dir:
            Optional directory for automatic ``save()`` after ingestion.
        use_clef:
            Enable Cloudflare Clef passage-gating.  Requires
            ``cloudflare_account_id`` and ``cloudflare_api_token``.
        cloudflare_account_id:
            Cloudflare account ID (or set ``CLOUDFLARE_ACCOUNT_ID`` env var).
        cloudflare_api_token:
            Cloudflare API token (or set ``CLOUDFLARE_API_TOKEN`` env var).
        use_generator:
            Enable LLM answer generation in ``rag.query()``.
        generator_api_key:
            API key for the LLM generation endpoint.
        generator_base_url:
            Base URL for an OpenAI-compatible generation endpoint.
        generator_model:
            Model name for LLM generation.
        embedding_model:
            SentenceTransformer model name (ignored when a cloud embedding
            model is configured via environment variables).
        sigma_mode:
            Covariance estimation mode: ``"heuristic"``, ``"dissolver"``,
            or ``"learned"``.
        chunk_size:
            Token budget per chunk for ingestion.
        chunk_overlap:
            Token overlap between consecutive chunks.

        Returns
        -------
        GaussianRAG
        """
        st_model = _get_st_model(embedding_model)
        embedder = GaussianEmbedder(model=st_model, sigma_mode=sigma_mode)

        store = KnowledgeStore()
        retriever = GaussianRetriever(store=store)
        ingestor = UniversalIngestor(
            embedder, store, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Optional Clef
        clef_client = None
        passage_gate = None
        if use_clef:
            import os
            acct = cloudflare_account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
            token = cloudflare_api_token or os.getenv("CLOUDFLARE_API_TOKEN", "")
            if acct and token:
                try:
                    ClefClient = _get_clef_client()
                    clef_client = ClefClient(account_id=acct, api_token=token)
                    ClefPassageGate = _get_clef_gate()
                    passage_gate = ClefPassageGate(clef_client=clef_client)
                    retriever.passage_gate = passage_gate
                    logger.info("GaussianRAG: Cloudflare Clef passage-gating enabled")
                except Exception as exc:
                    logger.warning("GaussianRAG: Clef activation failed: %s", exc)
            else:
                logger.warning(
                    "GaussianRAG: use_clef=True but credentials are missing; "
                    "set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN"
                )

        # Optional generator â€” pass a pre-built generator instance via
        # rag._generator = <generator> after creation, or configure via
        # the existing pipeline in gaussian_rag.rag.generation.
        generator = None
        if use_generator:
            logger.info(
                "GaussianRAG: use_generator=True; assign a generator via "
                "rag._generator after calling GaussianRAG.create()"
            )

        return cls(
            embedder=embedder,
            store=store,
            retriever=retriever,
            ingestor=ingestor,
            clef_client=clef_client,
            passage_gate=passage_gate,
            generator=generator,
            save_dir=Path(save_dir) if save_dir else None,
        )

    @classmethod
    def load(cls, directory: str | Path, **kwargs: Any) -> "GaussianRAG":
        """Load a previously saved GaussianRAG instance from *directory*.

        The saved directory must contain the files produced by ``rag.save()``:
        ``knowledge.json`` and optionally ``lexical.db``.

        Additional keyword arguments are passed to ``GaussianRAG.create()``
        to reconstruct the embedder and optional cloud components.

        Parameters
        ----------
        directory:
            Path to the saved store directory.

        Returns
        -------
        GaussianRAG
        """
        path = Path(directory)
        rag = cls.create(**kwargs)
        loaded_store = KnowledgeStore.load(path)
        # Merge loaded items into the new empty store
        rag._store.items = loaded_store.items
        rag._store.lexical = loaded_store.lexical
        # Rebuild ANN index
        rag._retriever.refresh()
        rag._save_dir = path
        logger.info(
            "GaussianRAG: loaded %d items from '%s'",
            len(rag._store.items),
            directory,
        )
        return rag

    # ------------------------------------------------------------------
    # Ingestion API
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[GaussianKnowledge]:
        """Index a list of raw text strings.

        Parameters
        ----------
        texts:
            Passages to index.
        metadata:
            Optional per-passage metadata dicts.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        nodes = self._ingestor.ingest_texts(texts, metadata)
        self._retriever.refresh()
        return nodes

    def add_raw_text(
        self,
        raw_text: str,
        *,
        title: str | None = None,
        use_llm: bool | None = None,
        doc_id_prefix: str = "raw",
    ) -> list[GaussianKnowledge]:
        """Clean, normalize, structure, and index arbitrary raw or messy pasted text.

        Automatically runs the ``TextNormalizer`` (stripping HTML, repairing OCR
        hyphenation, unwrapping line breaks, removing timestamps/artifacts),
        and partitions the text into structured ``CorpusDocument``s using rule-based
        or optional OpenAI-compatible LLM assistance.

        Parameters
        ----------
        raw_text:
            Unformatted, messy, or arbitrary text string.
        title:
            Optional title or topic name for the corpus.
        use_llm:
            Enable or force OpenAI-compatible LLM assistance for repairing ambiguous
            corrupted characters or extracting complex document structures.
        doc_id_prefix:
            Document ID prefix.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created and indexed into KnowledgeStore and LexicalStore.
        """
        from gaussian_rag.creator.corpus_generator import CorpusGenerator

        generator = CorpusGenerator()
        documents = generator.generate(
            raw_text, title=title, doc_id_prefix=doc_id_prefix, use_llm=use_llm
        )
        nodes = CorpusGenerator.load_into_rag(documents, self)
        self._retriever.refresh()
        return nodes

    def add_markdown(
        self,
        markdown_text: str,
        document_id: str | None = None,
    ) -> list[GaussianKnowledge]:
        """Index a Markdown document, preserving heading metadata.

        Parameters
        ----------
        markdown_text:
            Raw Markdown string.
        document_id:
            Logical document name (auto-generated if omitted).

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        doc_id = document_id or "doc"
        nodes = self._ingestor.ingest_markdown(markdown_text, document_id=doc_id)
        self._retriever.refresh()
        return nodes

    def add_records(
        self,
        records: list[dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id",
    ) -> list[GaussianKnowledge]:
        """Index structured records (dicts / JSON objects).

        Parameters
        ----------
        records:
            List of dicts.  Each must have a key matching *text_field*.
        text_field:
            Key containing the passage text.
        id_field:
            Key to use as the record document ID.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        nodes = self._ingestor.ingest_records(
            records, text_field=text_field, id_field=id_field
        )
        self._retriever.refresh()
        return nodes

    def add_file(self, path: str | Path, **kwargs: Any) -> list[GaussianKnowledge]:
        """Index a file, auto-detecting format by extension.

        Supported: ``.md``, ``.txt``, ``.json``, ``.jsonl``, ``.csv``.

        Parameters
        ----------
        path:
            Path to the file to ingest.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        nodes = self._ingestor.ingest_file(path, **kwargs)
        self._retriever.refresh()
        return nodes

    def add_directory(
        self,
        dir_path: str | Path,
        *,
        recursive: bool = True,
        glob_pattern: str = "*",
    ) -> list[GaussianKnowledge]:
        """Index all matching files in *dir_path*.

        Parameters
        ----------
        dir_path:
            Directory to walk.
        recursive:
            Walk subdirectories recursively (default: ``True``).
        glob_pattern:
            Glob pattern to filter files.

        Returns
        -------
        All ``GaussianKnowledge`` nodes created.
        """
        nodes = self._ingestor.ingest_directory(
            dir_path, recursive=recursive, glob_pattern=glob_pattern
        )
        self._retriever.refresh()
        return nodes

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    def build_taxonomy(
        self,
        top_branches: int = 12,
        leaf_size: int = 64,
        seed: int = 17,
    ) -> TaxonomyRouter:
        """Build hierarchical taxonomy clustering over all ingested passages.

        Prunes 60–80% of candidate search space during retrieval by routing queries
        to relevant topic branches.

        Parameters
        ----------
        top_branches:
            Target number of top-level topic branches.
        leaf_size:
            Target number of items per leaf cluster.
        seed:
            Random seed for deterministic clustering.

        Returns
        -------
        Constructed TaxonomyRouter instance attached to retriever.
        """
        router = TaxonomyRouter.build_from_store(
            self._store,
            top_branches=top_branches,
            leaf_size=leaf_size,
            seed=seed,
        )
        self._retriever.taxonomy = router
        return router

    def search(
        self,
        query: str,
        top_k: int = 5,
        metric: str = "elk",
        with_covariance: bool = False,
    ) -> list[RetrievedChunk]:
        """Pure local hybrid retrieval â€” no LLM or cloud required.

        Uses dense Gaussian ANN search combined with SQLite FTS5 lexical
        BM25 scoring for robust candidate selection, then ranks by the
        specified manifold geometry metric.

        Parameters
        ----------
        query:
            Natural language query string.
        top_k:
            Maximum number of chunks to return.
        metric:
            Ranking metric: ``"elk"`` (ELK score), ``"w2"`` (Wasserstein-2),
            or ``"fr"`` (Fisher-Rao).

        Returns
        -------
        Ranked list of ``RetrievedChunk`` objects.
        """
        if not self._store.items:
            return []

        qr: QueryRepresentation = self._embedder.encode_query(query, with_covariance=with_covariance)
        results = self._retriever.retrieve(
            qr,
            top_k=top_k,
            metric=metric,
            query_text=query,
        )
        return results

    async def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """End-to-end RAG pipeline: retrieve â†’ (optionally) gate â†’ (optionally) generate.

        Works fully locally when no cloud components are configured.  Activates
        Cloudflare Clef gating and LLM answer generation when those are set up.

        Parameters
        ----------
        question:
            Natural language question.
        top_k:
            Number of passages to retrieve.

        Returns
        -------
        A dict with keys:
        - ``"question"`` â€” the original question.
        - ``"retrieved_chunks"`` â€” list of ``RetrievedChunk`` objects.
        - ``"context"`` â€” concatenated passage texts used for generation.
        - ``"answer"`` â€” LLM-generated answer (empty string if no generator).
        """
        chunks = self.search(question, top_k=top_k)

        # Optional Clef passage gating
        if self._passage_gate is not None and chunks:
            try:
                gated_chunks = await self._passage_gate.gate(question, chunks)
                if gated_chunks:
                    chunks = gated_chunks
            except Exception as exc:
                logger.warning("GaussianRAG.query: Clef gating error: %s", exc)

        context = "\n\n".join(c.knowledge.text for c in chunks)
        answer = ""

        if self._generator is not None:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._generator.generate(question, chunks),  # type: ignore[union-attr]
                )
                if isinstance(result, dict):
                    answer = result.get("answer", "")
                else:
                    answer = str(result)
            except Exception as exc:
                logger.warning("GaussianRAG.query: generation error: %s", exc)

        return {
            "question": question,
            "retrieved_chunks": chunks,
            "context": context,
            "answer": answer,
        }

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist the knowledge store (dense + lexical) to disk.

        Parameters
        ----------
        directory:
            Save directory.  Defaults to ``save_dir`` set at creation time.
            Raises ``ValueError`` if neither is provided.

        Returns
        -------
        The directory ``Path`` where the store was saved.
        """
        target = Path(directory) if directory else self._save_dir
        if target is None:
            raise ValueError(
                "No save directory specified. Pass 'directory' or set 'save_dir' in GaussianRAG.create()."
            )
        self._store.save(target)
        if self._retriever.taxonomy is not None and not self._retriever.taxonomy.is_empty:
            self._retriever.taxonomy.save(target)
        self._save_dir = target
        logger.info("GaussianRAG: saved %d items to '%s'", len(self._store.items), target)
        return target

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def item_count(self) -> int:
        """Number of indexed passages in the knowledge store."""
        return len(self._store.items)

    @property
    def embedder(self) -> GaussianEmbedder:
        """The underlying Gaussian embedder."""
        return self._embedder

    @property
    def store(self) -> KnowledgeStore:
        """The underlying knowledge store."""
        return self._store

    @property
    def retriever(self) -> GaussianRetriever:
        """The underlying hybrid retriever."""
        return self._retriever

    @property
    def ingestor(self) -> UniversalIngestor:
        """The underlying universal ingestor."""
        return self._ingestor

    @property
    def taxonomy(self) -> TaxonomyRouter | None:
        """The underlying hierarchical taxonomy router, if built or configured."""
        return self._retriever.taxonomy

    def close(self) -> None:
        """Close open resources (e.g. SQLite lexical database)."""
        if self._store is not None:
            self._store.close()

    def __enter__(self) -> "GaussianRAG":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GaussianRAG(items={self.item_count}, "
            f"clef={'enabled' if self._clef_client else 'disabled'}, "
            f"generator={'enabled' if self._generator else 'disabled'})"
        )
