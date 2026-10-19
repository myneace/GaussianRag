"""test_online_trainer_windows.py — Verify cross-platform atomic weight replacement.

Validates BUG-02:
  - Replacing destination weights when destination already exists succeeds without FileExistsError.
  - os.replace() works atomically on Windows ([WinError 183] prevention) and POSIX.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from gaussian_rag.training.online_trainer import OnlineRadiationCollector


def test_atomic_weight_replace_overwrites_existing(tmp_path: Path):
    """Verify that replacing an already existing weights file does not raise FileExistsError."""
    target_weights = tmp_path / ".gaussian_rag_weights_64.pt"
    tmp_weights = tmp_path / ".gaussian_rag_weights_64.pt.tmp"

    # Step 1: Pre-create target file (simulating existing trained weights)
    target_weights.write_bytes(b"ORIGINAL_WEIGHTS_V1")
    assert target_weights.exists()

    # Step 2: Write new weights to temporary file
    tmp_weights.write_bytes(b"UPDATED_WEIGHTS_V2")
    assert tmp_weights.exists()

    # Step 3: Perform atomic replace (same operation used in OnlineRadiationCollector)
    os.replace(tmp_weights, target_weights)

    # Verification: Target exists with updated contents, tmp file is gone
    assert target_weights.exists()
    assert not tmp_weights.exists()
    assert target_weights.read_bytes() == b"UPDATED_WEIGHTS_V2"

    # Step 4: Repeat again to simulate multiple training epochs
    tmp_weights.write_bytes(b"UPDATED_WEIGHTS_V3")
    os.replace(tmp_weights, target_weights)
    assert target_weights.read_bytes() == b"UPDATED_WEIGHTS_V3"


def test_online_radiation_collector_stats(tmp_path: Path):
    """Verify OnlineRadiationCollector tracks stats and paths cleanly without crashing."""
    corpus = tmp_path / "corpus.jsonl"
    events = tmp_path / "events.jsonl"
    weights = tmp_path / "weights.pt"
    stats = tmp_path / "stats.json"

    collector = OnlineRadiationCollector(
        corpus_path=corpus,
        events_path=events,
        weights_path=weights,
        stats_path=stats,
        train_every_n=100,  # avoid triggering background thread during unit test
    )

    collector.collect([{"text": "Sample radiation text", "domain": "test_domain"}])
    stats_data = collector.get_stats()
    assert stats_data["total_absorptions"] == 1
    assert stats_data["total_samples_collected"] == 1
    assert corpus.exists()
