"""test_binary_embeddings.py — Verify binary NumPy storage for high-dimensional embeddings.

Validates SCALE-01:
  - KnowledgeStore.save() exports numerical vectors into binary format (embeddings.npz)
  - knowledge.json stores passage text and chunk metadata without string-serialized float matrices
  - Loading from embeddings.npz preserves exact IEEE 754 precision
  - Backwards-compatibility loading legacy knowledge.json files with embedded float lists
  - File size reduction and serialization benchmarks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.core.types import GaussianKnowledge


def _create_test_store(count: int = 10, dim: int = 64, rank: int = 4) -> KnowledgeStore:
    store = KnowledgeStore()
    rng = np.random.default_rng(42)
    chunks = []
    for i in range(count):
        mu = rng.standard_normal(dim)
        sigma_diag = np.abs(rng.standard_normal(dim)) + 0.01
        sigma_L = rng.standard_normal((dim, rank))
        chunks.append(
            GaussianKnowledge(
                id=f"chunk-{i}",
                text=f"Passage content number {i} with specific technical details.",
                mu=mu,
                sigma_diag=sigma_diag,
                sigma_L=sigma_L,
                metadata={"index": i, "category": "science"},
            )
        )
    store.add(chunks)
    return store


def test_binary_save_creates_npz_and_lightweight_json(tmp_path: Path):
    """Verify save creates embeddings.npz and knowledge.json without float lists."""
    store = _create_test_store(count=5)
    save_dir = tmp_path / "binary_store"
    store.save(save_dir)

    assert (save_dir / "embeddings.npz").exists()
    assert (save_dir / "knowledge.json").exists()

    # Inspect knowledge.json content
    raw_json = json.loads((save_dir / "knowledge.json").read_text(encoding="utf-8"))
    assert len(raw_json) == 5
    for item in raw_json:
        assert "id" in item
        assert "text" in item
        assert "metadata" in item
        assert "mu" not in item, "knowledge.json should not contain string-serialized mu"
        assert "sigma_diag" not in item, "knowledge.json should not contain string-serialized sigma_diag"
        assert "sigma_L" not in item, "knowledge.json should not contain string-serialized sigma_L"


def test_roundtrip_exact_ieee754_precision(tmp_path: Path):
    """Verify roundtrip load restores exact floating point arrays."""
    store = _create_test_store(count=10)
    save_dir = tmp_path / "precision_store"
    store.save(save_dir)

    loaded = KnowledgeStore.load(save_dir)
    assert len(loaded.items) == len(store.items)

    for item_id, orig_item in store.items.items():
        loaded_item = loaded.items[item_id]
        assert loaded_item.id == orig_item.id
        assert loaded_item.text == orig_item.text
        assert loaded_item.metadata == orig_item.metadata
        # Exact array equality down to the last bit
        np.testing.assert_array_equal(loaded_item.mu, orig_item.mu)
        np.testing.assert_array_equal(loaded_item.sigma_diag, orig_item.sigma_diag)
        np.testing.assert_array_equal(loaded_item.sigma_L, orig_item.sigma_L)


def test_legacy_knowledge_json_backwards_compatibility(tmp_path: Path):
    """Verify loading legacy store with mu/sigma_diag/sigma_L inside knowledge.json works."""
    legacy_dir = tmp_path / "legacy_store"
    legacy_dir.mkdir(parents=True)

    legacy_payload = [
        {
            "id": "legacy-1",
            "text": "Legacy passage text without embeddings.npz",
            "mu": [0.1, 0.2, 0.3, 0.4],
            "sigma_diag": [0.01, 0.02, 0.03, 0.04],
            "sigma_L": [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
            "metadata": {"legacy": True},
        }
    ]
    (legacy_dir / "knowledge.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    loaded = KnowledgeStore.load(legacy_dir)
    assert "legacy-1" in loaded.items
    item = loaded.items["legacy-1"]
    assert item.text == "Legacy passage text without embeddings.npz"
    assert item.metadata == {"legacy": True}
    np.testing.assert_allclose(item.mu, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(item.sigma_diag, [0.01, 0.02, 0.03, 0.04])
    assert item.sigma_L.shape == (4, 2)


def test_file_size_reduction_benchmark(tmp_path: Path):
    """Verify binary .npz storage achieves significant disk footprint reduction."""
    count = 200
    dim = 128
    rank = 4
    store = _create_test_store(count=count, dim=dim, rank=rank)

    # 1. Binary save
    bin_dir = tmp_path / "bench_bin"
    store.save(bin_dir)
    bin_total_bytes = (bin_dir / "knowledge.json").stat().st_size + (bin_dir / "embeddings.npz").stat().st_size

    # 2. Legacy save simulation
    legacy_payload = []
    for item in store.items.values():
        legacy_payload.append({
            "id": item.id,
            "text": item.text,
            "mu": item.mu.tolist(),
            "sigma_diag": item.sigma_diag.tolist(),
            "sigma_L": item.sigma_L.tolist(),
            "metadata": item.metadata,
        })
    legacy_bytes = len(json.dumps(legacy_payload, indent=2).encode("utf-8"))

    # Assert > 50% size reduction compared to formatted JSON floats
    reduction = 1.0 - (bin_total_bytes / legacy_bytes)
    assert reduction > 0.50, f"Expected >50% disk footprint reduction, got {reduction * 100:.1f}%"
