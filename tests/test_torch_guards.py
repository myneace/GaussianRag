"""
test_torch_guards.py
Phase 21: Safe PyTorch Import Guards for Background Radiation Training.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from unittest.mock import patch

import pytest


def test_online_trainer_imports_without_crash():
    """online_trainer must be importable in any environment."""
    import gaussian_rag.training.online_trainer as ot
    assert hasattr(ot, "OnlineRadiationCollector")
    assert hasattr(ot, "TORCH_AVAILABLE")


def test_trainer_module_imports_without_crash():
    """trainer.py must be importable without crashing."""
    import gaussian_rag.training.trainer as tr
    assert hasattr(tr, "SigmaTrainer")
    assert hasattr(tr, "DomainTextDataset")
    assert hasattr(tr, "_TORCH_AVAILABLE")


def test_sigma_head_imports_without_crash():
    """sigma_head.py must be importable without crashing."""
    import gaussian_rag.training.sigma_head as sh
    assert hasattr(sh, "SigmaHead")
    assert hasattr(sh, "_TORCH_AVAILABLE")


def test_losses_imports_without_crash():
    """losses.py must be importable without crashing."""
    import gaussian_rag.training.losses as lo
    assert hasattr(lo, "_TORCH_AVAILABLE")


def test_torch_available_is_bool():
    """TORCH_AVAILABLE must be a plain bool."""
    from gaussian_rag.training.online_trainer import TORCH_AVAILABLE
    assert isinstance(TORCH_AVAILABLE, bool)


def test_background_train_skips_without_torch(tmp_path: Path, caplog):
    """When TORCH_AVAILABLE is False, _background_train must return immediately with a warning."""
    import gaussian_rag.training.online_trainer as ot

    collector = ot.OnlineRadiationCollector(
        corpus_path=tmp_path / "corpus.jsonl",
        events_path=tmp_path / "events.jsonl",
        stats_path=tmp_path / "stats.json",
        train_every_n=999,
    )

    emitted = []
    collector._emit_fn = lambda event_type, data: emitted.append({"type": event_type, "data": data})

    with patch.object(ot, "TORCH_AVAILABLE", False):
        with caplog.at_level(logging.WARNING, logger="gaussian_rag.training.online_trainer"):
            collector._background_train()

    assert any("PyTorch" in r.message for r in caplog.records)
    system_logs = [e for e in emitted if e["type"] == "system_log"]
    assert len(system_logs) >= 1
    assert "PyTorch" in system_logs[0]["data"]["message"]


def test_collect_does_not_crash_without_torch(tmp_path: Path):
    """collect() must succeed even when TORCH_AVAILABLE is False."""
    import gaussian_rag.training.online_trainer as ot

    collector = ot.OnlineRadiationCollector(
        corpus_path=tmp_path / "corpus.jsonl",
        events_path=tmp_path / "events.jsonl",
        stats_path=tmp_path / "stats.json",
        train_every_n=1,
    )

    background_train_calls = []

    def fake_bg_train():
        background_train_calls.append("called")

    collector._background_train = fake_bg_train

    with patch.object(ot, "TORCH_AVAILABLE", False):
        collector.collect([{"text": "The manifold is curved.", "domain": "geometry"}])

    stats = collector.get_stats()
    assert stats["total_samples_collected"] == 1
    assert stats["total_absorptions"] == 1


def test_collector_instantiation_needs_no_torch(tmp_path: Path):
    """OnlineRadiationCollector must instantiate cleanly when TORCH_AVAILABLE is False."""
    import gaussian_rag.training.online_trainer as ot

    with patch.object(ot, "TORCH_AVAILABLE", False):
        collector = ot.OnlineRadiationCollector(
            corpus_path=tmp_path / "corpus.jsonl",
            events_path=tmp_path / "events.jsonl",
            stats_path=tmp_path / "stats.json",
        )
    assert collector is not None
    stats = collector.get_stats()
    assert stats["total_absorptions"] == 0
