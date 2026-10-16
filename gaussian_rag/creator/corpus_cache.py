from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaussian_rag.creator.corpus_generator import CorpusDocument

logger = logging.getLogger(__name__)
CACHE_FORMAT_VERSION: str = "1"

_DEFAULT_CACHE_ROOT: Path = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "gaussian_rag_corpus"
    if os.name == "nt"
    else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "gaussian_rag_corpus"
)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CorpusCache:
    """Content-addressed on-disk cache for generated CorpusDocument lists."""

    def __init__(self, cache_dir=None, ttl_seconds=None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_ROOT
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def lookup(self, source_text):
        fp = _fingerprint(source_text)
        entry_dir = self.cache_dir / fp
        if not entry_dir.is_dir():
            logger.debug("Cache miss (no entry): %s", fp[:12])
            return False, None
        meta_path = entry_dir / "meta.json"
        corpus_path = entry_dir / "corpus.json"
        if not meta_path.exists() or not corpus_path.exists():
            logger.debug("Cache miss (incomplete): %s", fp[:12])
            self._evict(entry_dir)
            return False, None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache meta read error %s: %s", fp[:12], exc)
            self._evict(entry_dir)
            return False, None
        if meta.get("format_version") != CACHE_FORMAT_VERSION:
            logger.debug("Cache miss (stale schema): %s", fp[:12])
            self._evict(entry_dir)
            return False, None
        if self.ttl_seconds is not None:
            age = time.time() - meta.get("stored_at", 0.0)
            if age > self.ttl_seconds:
                logger.debug("Cache miss (expired %.0fs): %s", age, fp[:12])
                self._evict(entry_dir)
                return False, None
        try:
            raw_docs = json.loads(corpus_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache corpus read error %s: %s", fp[:12], exc)
            self._evict(entry_dir)
            return False, None
        docs = self._deserialize_docs(raw_docs)
        logger.info("Cache hit: reusing %d docs for %s", len(docs), fp[:12])
        return True, docs

    def store(self, source_text, docs):
        fp = _fingerprint(source_text)
        entry_dir = self.cache_dir / fp
        entry_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "fingerprint": fp,
            "format_version": CACHE_FORMAT_VERSION,
            "stored_at": time.time(),
            "doc_count": len(docs),
            "char_count": len(source_text),
        }
        try:
            (entry_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (entry_dir / "corpus.json").write_text(json.dumps([d.to_dict() for d in docs], indent=2), encoding="utf-8")
            logger.debug("Cached %d docs under %s", len(docs), fp[:12])
        except OSError as exc:
            logger.warning("Cache write error %s: %s", fp[:12], exc)

    def invalidate(self, source_text):
        fp = _fingerprint(source_text)
        entry_dir = self.cache_dir / fp
        if entry_dir.is_dir():
            self._evict(entry_dir)
            logger.info("Invalidated cache: %s", fp[:12])
            return True
        return False

    def clear(self):
        count = 0
        for child in self.cache_dir.iterdir():
            if child.is_dir():
                self._evict(child)
                count += 1
        logger.info("Cleared %d cache entries", count)
        return count

    def stats(self):
        entries = [c for c in self.cache_dir.iterdir() if c.is_dir()]
        total_bytes = sum(f.stat().st_size for e in entries for f in e.iterdir() if f.is_file())
        return {"cache_dir": str(self.cache_dir), "entry_count": len(entries), "total_bytes": total_bytes, "format_version": CACHE_FORMAT_VERSION}

    @staticmethod
    def _evict(entry_dir):
        try:
            for f in entry_dir.iterdir():
                f.unlink(missing_ok=True)
            entry_dir.rmdir()
        except OSError as exc:
            logger.debug("Evict error %s: %s", entry_dir.name, exc)

    @staticmethod
    def _deserialize_docs(raw):
        from gaussian_rag.creator.corpus_generator import CorpusDocument
        docs = []
        for rec in raw:
            try:
                docs.append(CorpusDocument(
                    id=str(rec["id"]),
                    title=str(rec["title"]),
                    text=str(rec["text"]),
                    section_number=rec.get("section_number"),
                    structure_type=str(rec.get("structure_type", "semantic_chunk")),
                    metadata=rec.get("metadata") or {},
                    char_count=int(rec.get("char_count", 0)),
                    token_estimate=int(rec.get("token_estimate", 0)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed cache record: %s", exc)
        return docs
