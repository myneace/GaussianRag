"""Unit and integration tests for TextNormalizer, CorpusGenerator, and CLI."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaussian_rag.creator.corpus_cache import CorpusCache, CACHE_FORMAT_VERSION
from gaussian_rag.creator.corpus_generator import CorpusDocument, CorpusGenerator
from gaussian_rag.creator.llm_assistant import OpenAICorpusAssistant
from gaussian_rag.creator.text_normalizer import TextNormalizer
from gaussian_rag.rag.facade import GaussianRAG


# ---------------------------------------------------------------------------
# CorpusCache tests
# ---------------------------------------------------------------------------

def test_corpus_cache_miss_then_hit(tmp_path: Path) -> None:
    """Storing docs then looking up the same text returns a cache hit."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    raw = "Law 1: Never Outshine the Master\nAlways make those above you feel superior."

    # Initial lookup: miss
    hit, docs = cache.lookup(raw)
    assert not hit
    assert docs is None

    # Store
    source_docs = [
        CorpusDocument(id="d1", title="Law 1", text="Never outshine the master.", section_number=1)
    ]
    cache.store(raw, source_docs)

    # Second lookup: hit
    hit, docs = cache.lookup(raw)
    assert hit
    assert docs is not None
    assert len(docs) == 1
    assert docs[0].id == "d1"
    assert docs[0].title == "Law 1"
    assert docs[0].section_number == 1


def test_corpus_cache_different_text_gives_miss(tmp_path: Path) -> None:
    """Different raw text yields a cache miss even when similar."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    raw_a = "Text A: some unique content."
    raw_b = "Text B: different content."
    cache.store(raw_a, [CorpusDocument(id="a", title="A", text="Text A")])

    hit, _ = cache.lookup(raw_b)
    assert not hit


def test_corpus_cache_stale_schema_invalidates(tmp_path: Path) -> None:
    """Cache entries written with an old format_version are discarded on lookup."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    raw = "Law 2: Never trust friends."
    cache.store(raw, [CorpusDocument(id="d2", title="Law 2", text="Trust no one.")])

    # Corrupt the schema version in meta.json
    import hashlib as _hashlib, json as _json
    fp = _hashlib.sha256(raw.encode()).hexdigest()
    meta_path = cache.cache_dir / fp / "meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = "999"  # bogus future version
    meta_path.write_text(_json.dumps(meta), encoding="utf-8")

    hit, docs = cache.lookup(raw)
    assert not hit
    assert docs is None
    # Entry directory should have been evicted
    assert not (cache.cache_dir / fp).exists()


def test_corpus_cache_ttl_expiry(tmp_path: Path) -> None:
    """Cache entries older than ttl_seconds are treated as expired."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache", ttl_seconds=1)
    raw = "Ephemeral text for TTL test."
    cache.store(raw, [CorpusDocument(id="e1", title="E", text="Ephemeral.")])

    # Immediate lookup: hit
    hit, _ = cache.lookup(raw)
    assert hit

    # Simulate expiry by backdating stored_at in meta.json
    import hashlib as _hashlib, json as _json
    fp = _hashlib.sha256(raw.encode()).hexdigest()
    meta_path = cache.cache_dir / fp / "meta.json"
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stored_at"] = time.time() - 10  # 10s ago, well past ttl of 1s
    meta_path.write_text(_json.dumps(meta), encoding="utf-8")

    hit, docs = cache.lookup(raw)
    assert not hit
    assert docs is None


def test_corpus_cache_invalidate(tmp_path: Path) -> None:
    """invalidate() removes a specific cache entry."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    raw = "Law 3: Conceal your intentions."
    cache.store(raw, [CorpusDocument(id="d3", title="Law 3", text="Never reveal.")])

    removed = cache.invalidate(raw)
    assert removed is True

    hit, _ = cache.lookup(raw)
    assert not hit

    # Double invalidate returns False
    assert cache.invalidate(raw) is False


def test_corpus_cache_clear(tmp_path: Path) -> None:
    """clear() removes all cache entries and returns count."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    for i in range(3):
        cache.store(f"text_{i}", [CorpusDocument(id=f"d{i}", title=f"T{i}", text=f"Body {i}")])

    s = cache.stats()
    assert s["entry_count"] == 3

    removed = cache.clear()
    assert removed == 3
    assert cache.stats()["entry_count"] == 0


def test_corpus_cache_stats(tmp_path: Path) -> None:
    """stats() reports correct entry_count and format_version."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    assert cache.stats()["entry_count"] == 0
    cache.store("some text", [CorpusDocument(id="x", title="X", text="X body.")])
    s = cache.stats()
    assert s["entry_count"] == 1
    assert s["total_bytes"] > 0
    assert s["format_version"] == CACHE_FORMAT_VERSION


def test_corpus_generator_uses_cache(tmp_path: Path) -> None:
    """CorpusGenerator.generate() short-circuits on cache hit."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    generator = CorpusGenerator(cache=cache)
    raw = (
        "Law 1: Never Outshine the Master\n"
        "Always make those above you feel comfortably superior.\n\n"
        "Law 2: Never Put Too Much Trust in Friends\n"
        "Be wary of friends."
    )

    # First call: cache miss — full processing
    docs1 = generator.generate(raw, title="Laws")
    assert len(docs1) >= 1

    # Verify stored in cache
    hit, cached = cache.lookup(raw)
    assert hit
    assert cached is not None and len(cached) == len(docs1)

    # Second call: cache hit — normalizer should NOT be invoked again
    with patch.object(generator.normalizer, "clean", wraps=generator.normalizer.clean) as mock_clean:
        docs2 = generator.generate(raw, title="Laws")
        mock_clean.assert_not_called()  # clean() skipped entirely on cache hit
    assert len(docs2) == len(docs1)


def test_corpus_generator_no_cache_forces_reprocess(tmp_path: Path) -> None:
    """use_cache=False forces full reprocessing even when an entry exists."""
    cache = CorpusCache(cache_dir=tmp_path / "corpus_cache")
    generator = CorpusGenerator(cache=cache)
    raw = "Law 5: Reputation is cornerstone of power."

    docs1 = generator.generate(raw)
    assert len(docs1) >= 1

    with patch.object(generator.normalizer, "clean", wraps=generator.normalizer.clean) as mock_clean:
        docs2 = generator.generate(raw, use_cache=False)
        mock_clean.assert_called_once()  # full pipeline runs again
    assert len(docs2) >= 1


# ---------------------------------------------------------------------------
# TextNormalizer tests
# ---------------------------------------------------------------------------

def test_text_normalizer_html_and_entities() -> None:
    normalizer = TextNormalizer()
    dirty = (
        "<div><script>alert('bad');</script>"
        "<p>This is &quot;quoted&quot; text &amp; a &nbsp; space.<br/>Next line.</p></div>"
    )
    clean = normalizer.clean(dirty)
    assert "alert" not in clean
    assert "<script>" not in clean
    assert "<p>" not in clean
    assert '"quoted"' in clean
    assert "&" in clean
    assert "Next line" in clean


def test_text_normalizer_ocr_hyphenation() -> None:
    normalizer = TextNormalizer()
    dirty = "This is a demon-\nstration of infor-\nmation retrieval."
    clean = normalizer.clean(dirty)
    assert "demonstration" in clean
    assert "information" in clean
    assert "demon-" not in clean
    assert "infor-" not in clean


def test_text_normalizer_soft_linebreaks_and_paragraphs() -> None:
    normalizer = TextNormalizer()
    dirty = (
        "This is a single sentence\n"
        "broken across lines by\n"
        "a PDF reader.\n\n"
        "This is paragraph two\n"
        "with more flowing text."
    )
    clean = normalizer.clean(dirty)
    paras = clean.split("\n\n")
    assert len(paras) == 2
    assert paras[0] == "This is a single sentence broken across lines by a PDF reader."
    assert paras[1] == "This is paragraph two with more flowing text."


def test_text_normalizer_page_artifacts_and_timestamps() -> None:
    normalizer = TextNormalizer()
    dirty = (
        "Page 14 of 95\n"
        "[10:30:15] Alice: We should proceed with the deployment.\n"
        "- 15 -\n"
        "[10:31] Bob: Agreed, the tests are green."
    )
    clean = normalizer.clean(dirty)
    assert "Page 14 of 95" not in clean
    assert "- 15 -" not in clean
    assert "[10:30:15]" not in clean
    assert "[10:31]" not in clean
    assert "Alice: We should proceed with the deployment." in clean
    assert "Bob: Agreed, the tests are green." in clean


def test_text_normalizer_unicode_and_smart_quotes() -> None:
    normalizer = TextNormalizer()
    # Zero-width spaces, smart quotes, em-dashes
    dirty = "Hello\u200bWorld! \u201cSmart Quotes\u201d \u2014 and dashes\u00a0here."
    clean = normalizer.clean(dirty)
    assert "\u200b" not in clean
    assert '"Smart Quotes"' in clean
    assert "-- and dashes here." in clean or "- and dashes here." in clean


# ---------------------------------------------------------------------------
# OpenAICorpusAssistant tests
# ---------------------------------------------------------------------------

def test_llm_assistant_unconfigured_fallback() -> None:
    assistant = OpenAICorpusAssistant(api_key="", base_url="http://dummy")
    assert assistant.is_configured is False
    # Graceful fallback without crashing
    text = "Some text with broken char"
    assert assistant.repair_ambiguous_text(text) == text
    assert assistant.structure_into_sections(text) is None


@patch.object(OpenAICorpusAssistant, "_call_chat_completions")
def test_llm_assistant_repair_and_structure(mock_chat: MagicMock) -> None:
    assistant = OpenAICorpusAssistant(api_key="test-key", base_url="https://api.test.com/v1")
    assert assistant.is_configured is True

    # Test repair
    mock_chat.return_value = "Repaired clean prose."
    repaired = assistant.repair_ambiguous_text("Dirty text...")
    assert repaired == "Repaired clean prose."

    # Test structure
    mock_chat.return_value = json.dumps([
        {"id": "s1", "title": "Rule 1", "section_number": 1, "text": "Content of rule 1", "metadata": {}}
    ])
    sections = assistant.structure_into_sections("Rule 1 Content...")
    assert sections is not None
    assert len(sections) == 1
    assert sections[0]["title"] == "Rule 1"


# ---------------------------------------------------------------------------
# CorpusGenerator tests
# ---------------------------------------------------------------------------

def test_corpus_generator_numbered_sections() -> None:
    generator = CorpusGenerator()
    raw = (
        "Law 1: Never Outshine the Master\n"
        "Always make those above you feel comfortably superior.\n\n"
        "Law 2: Never Put Too Much Trust in Friends\n"
        "Be wary of friends—they will betray you quickly."
    )
    docs = generator.generate(raw, title="Greene's Laws")
    assert len(docs) == 2
    assert docs[0].section_number == 1
    assert "Never Outshine the Master" in docs[0].title
    assert docs[1].section_number == 2
    assert "Never Put Too Much Trust in Friends" in docs[1].title
    assert docs[0].structure_type == "numbered_section"


def test_corpus_generator_qa_pairs() -> None:
    generator = CorpusGenerator()
    raw = (
        "Q: What is Gaussian RAG?\n"
        "A: It is a probabilistic retrieval framework representing knowledge as multivariate distributions.\n\n"
        "Question: How does Clef accelerate retrieval?\n"
        "Answer: It replaces slow autoregressive LLM calls with non-autoregressive decision questions."
    )
    docs = generator.generate(raw, title="Gaussian RAG FAQ")
    assert len(docs) == 2
    assert docs[0].structure_type == "qa_pair"
    assert "Gaussian RAG" in docs[0].title
    assert docs[1].structure_type == "qa_pair"


def test_corpus_generator_fallback_chunking() -> None:
    generator = CorpusGenerator(chunk_size=30, chunk_overlap=5)
    # Long unbroken prose text
    raw = (
        "The quick brown fox jumps over the lazy dog. " * 25
    )
    docs = generator.generate(raw, title="Dense Prose")
    assert len(docs) >= 2
    assert docs[0].structure_type == "semantic_chunk"
    for d in docs:
        assert d.token_estimate > 0


def test_corpus_generator_export_jsonl_markdown_text(tmp_path: Path) -> None:
    generator = CorpusGenerator()
    docs = [
        CorpusDocument(id="doc-1", title="First Law", text="Always obey the rules.", section_number=1),
        CorpusDocument(id="doc-2", title="Second Law", text="Question everything.", section_number=2),
    ]

    # JSONL
    jsonl_path = tmp_path / "corpus.jsonl"
    generator.to_jsonl(docs, jsonl_path)
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    data0 = json.loads(lines[0])
    assert data0["id"] == "doc-1"
    assert data0["title"] == "First Law"

    # Markdown
    md_path = tmp_path / "corpus.md"
    generator.to_markdown(docs, md_path, title="Custom Laws")
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "# Custom Laws" in md_text
    assert "## First Law" in md_text

    # Text
    txt_path = tmp_path / "corpus.txt"
    generator.to_text(docs, txt_path)
    assert txt_path.exists()
    txt_text = txt_path.read_text(encoding="utf-8")
    assert "=== First Law ===" in txt_text


# ---------------------------------------------------------------------------
# Facade integration test
# ---------------------------------------------------------------------------

def test_gaussian_rag_add_raw_text() -> None:
    rag = GaussianRAG.create()
    dirty_text = (
        "<p>Law 1: Never Outshine the Master<br/>"
        "Always make those above you feel superior.</p>\n\n"
        "<p>Law 2: Never Put Too Much Trust in Friends<br/>"
        "Former enemies are often more loyal than friends.</p>"
    )
    nodes = rag.add_raw_text(dirty_text, title="48 Laws Excerpt")
    assert len(nodes) >= 2

    # Verify search finds the cleaned content
    results = rag.search("How to behave around your master?")
    assert len(results) > 0
    top_hit = results[0]
    assert "Never Outshine the Master" in top_hit.knowledge.text or "superior" in top_hit.knowledge.text


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

def test_cli_preset_48_laws(tmp_path: Path) -> None:
    out_file = tmp_path / "test_48_laws.txt"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "data" / "generate_corpus.py"),
        "--preset", "48-laws",
        "-o", str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "The 48 Laws of Power by Robert Greene" in content
    assert "Law 1: Never Outshine the Master" in content


def test_cli_raw_text_argument(tmp_path: Path) -> None:
    out_file = tmp_path / "cli_corpus.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "data" / "generate_corpus.py"),
        "-t", "Law 1: Be Authentic\nAlways be true to yourself.",
        "-o", str(out_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    data = json.loads(lines[0])
    assert "Be Authentic" in data["title"] or "Be Authentic" in data["text"]
