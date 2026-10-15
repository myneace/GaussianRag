"""
online_trainer.py
─────────────────
Continual-learning loop for the SigmaHead.

Every time the query pipeline runs it produces structured radiation events
plus derived {text, domain} samples from sense-disambiguation and manifold
retrieval. This module:

1. Appends those samples to robust_corpus.jsonl  (append-only, safe)
2. When N new samples have accumulated, spawns a background thread that
   fine-tunes the SigmaHead for 1-2 epochs on the *full* corpus.
3. Atomically swaps the weights file on disk.
   The next /api/query call will load the improved model transparently.

Thread-safety strategy
────────────────────────
• A single threading.Lock serialises JSONL writes.
• A threading.Event flags when training is running so a second query
  arriving mid-training simply skips triggering another run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path("data/training/robust_corpus.jsonl")
DEFAULT_EVENTS_PATH = Path("data/training/radiation_events.jsonl")
DEFAULT_WEIGHTS_PATH = Path(".gaussian_rag_weights_{dimension}.pt")
DEFAULT_STATS_PATH = Path("data/training/radiation_stats.json")
DEFAULT_TRIGGER_N = 5  # new samples before a background retrain
FINE_TUNE_EPOCHS = 2
FINE_TUNE_BATCH = 4


class OnlineRadiationCollector:
    def __init__(
        self,
        corpus_path: Path = DEFAULT_CORPUS_PATH,
        events_path: Path = DEFAULT_EVENTS_PATH,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        stats_path: Path = DEFAULT_STATS_PATH,
        train_every_n: int = DEFAULT_TRIGGER_N,
        emit_fn: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.events_path = Path(events_path)
        self.weights_path = Path(weights_path)
        self.stats_path = Path(stats_path)
        self.train_every_n = train_every_n
        self._emit_fn = emit_fn

        self._write_lock = threading.Lock()
        self._is_training = threading.Event()
        self._pending = 0

        # Load or initialize persistent stats
        self._stats = self._load_stats()

    def _emit(self, event_type: str, data: dict) -> None:
        """Fire an SSE event if a callback was provided (thread-safe)."""
        if self._emit_fn:
            try:
                self._emit_fn(event_type, data)
            except Exception:
                pass  # Never let SSE errors break the training pipeline

    # ──────────────────────────────────────────────────────────────────────
    # Stats persistence
    # ──────────────────────────────────────────────────────────────────────

    def _load_stats(self) -> dict:
        """Load stats from disk, or return a fresh baseline."""
        if self.stats_path.exists():
            try:
                return json.loads(self.stats_path.read_text())
            except Exception:
                pass
        return {
            "total_absorptions": 0,  # how many times collect() was called
            "total_samples_collected": 0,  # all-time unique samples ingested
            "total_training_runs": 0,  # completed background fine-tunes
            "total_training_triggered": 0,  # times training was triggered (may differ if skipped)
            "total_radiation_events": 0,
            "corpus_size_at_last_train": 0,
            "last_absorbed_at": None,
            "last_trained_at": None,
            "history": [],  # list of {timestamp, samples, elapsed_s, corpus_size}
        }

    def _save_stats(self) -> None:
        """Persist stats to disk (must be called inside _write_lock)."""
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.stats_path.write_text(json.dumps(self._stats, indent=2))
        except Exception as exc:
            logger.warning("[OnlineTrainer] Could not save stats: %s", exc)

    def get_stats(self) -> dict:
        """Return a copy of the current stats dict (thread-safe)."""
        with self._write_lock:
            return dict(self._stats)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def collect(self, samples: list[dict]) -> None:
        """
        Append *samples* to the corpus and conditionally trigger retraining.

        Parameters
        ----------
        samples : list of {"text": str, "domain": str}
        """
        if not samples:
            return

        # Deduplicate to avoid obvious duplicates (exact text match)
        unique = []
        seen: set[str] = set()
        for s in samples:
            text = s.get("text", "").strip()
            if text and text not in seen:
                seen.add(text)
                unique.append({"text": text, "domain": s.get("domain", "unknown")})

        with self._write_lock:
            self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
            with self.corpus_path.open("a") as fh:
                for sample in unique:
                    fh.write(json.dumps(sample) + "\n")
            self._pending += len(unique)

            # Update persistent stats
            self._stats["total_absorptions"] += 1
            self._stats["total_samples_collected"] += len(unique)
            self._stats["last_absorbed_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )

            should_train = (
                self._pending >= self.train_every_n and not self._is_training.is_set()
            )
            if should_train:
                self._pending = 0
                self._stats["total_training_triggered"] += 1

            self._save_stats()

        remaining = max(0, self.train_every_n - self._pending)
        stats_snap = self.get_stats()
        absorption_n = stats_snap["total_absorptions"]
        total_collected = stats_snap["total_samples_collected"]

        log_msg = (
            f"🔬 Absorption #{absorption_n} — {len(unique)} samples ingested "
            f"(total: {total_collected}). "
            + (
                "Triggering background refinement…"
                if should_train
                else f"{remaining} more until next refinement."
            )
        )
        logger.info("[OnlineTrainer] %s", log_msg)
        self._emit("system_log", {"message": log_msg})
        self._emit("radiation_stats", stats_snap)

        if should_train:
            t = threading.Thread(
                target=self._background_train,
                name="sigma-head-online-trainer",
                daemon=True,
            )
            t.start()

    def record_event(self, event: dict) -> None:
        event_payload = dict(event)
        event_payload.setdefault(
            "recorded_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

        with self._write_lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event_payload) + "\n")
            self._stats["total_radiation_events"] += 1
            self._save_stats()

        self._emit(
            "radiation_collected",
            {
                "query": event_payload.get("query_text", ""),
                "event_index": self.get_stats().get("total_radiation_events", 0),
                "feedback_signal": event_payload.get(
                    "feedback_signal", "implicit_generation"
                ),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────

    def _background_train(self) -> None:
        """Fine-tune the SigmaHead on the accumulated corpus (background thread)."""
        if self._is_training.is_set():
            logger.debug("[OnlineTrainer] Training already running — skipping.")
            return

        self._is_training.set()
        start = time.monotonic()
        msg = "⚙️ Background SigmaHead refinement started…"
        logger.info("[OnlineTrainer] %s", msg)
        self._emit("system_log", {"message": msg})

        try:
            from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
            from gaussian_rag.training.trainer import SigmaTrainer, DomainTextDataset
            from gaussian_rag.config import load_project_config

            cfg = load_project_config()
            enc_cfg = cfg["encoder"]
            embedder = GaussianEmbedder(
                dimension=enc_cfg["dimension"],
                sigma_mode="learned",
                rank=enc_cfg["low_rank"],
                embedding_backend=enc_cfg.get("embedding_backend", "local"),
                on_mismatch=enc_cfg.get("on_mismatch", "error"),
            )
            trainer = SigmaTrainer(embedder)

            dataset = DomainTextDataset.from_jsonl(self.corpus_path, embedder)
            n = len(dataset)
            logger.info(
                "[OnlineTrainer] Fine-tuning on %d samples for %d epoch(s)…",
                n,
                FINE_TUNE_EPOCHS,
            )
            self._emit(
                "system_log",
                {
                    "message": f"🧠 Fine-tuning SigmaHead on {n} accumulated samples ({FINE_TUNE_EPOCHS} epochs)…"
                },
            )

            trainer.fit(dataset, epochs=FINE_TUNE_EPOCHS, batch_size=FINE_TUNE_BATCH)

            # Atomic write: temp file → rename
            # Use dimension-specific filename
            actual_weights_path = Path(f".gaussian_rag_weights_{embedder.dimension}.pt")
            tmp = actual_weights_path.with_suffix(".pt.tmp")
            trainer.save_model(str(tmp))
            tmp.rename(actual_weights_path)

            elapsed = time.monotonic() - start

            # Persist the completed run into stats
            run_record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "corpus_size": n,
                "elapsed_s": round(elapsed, 2),
            }
            with self._write_lock:
                self._stats["total_training_runs"] += 1
                self._stats["corpus_size_at_last_train"] = n
                self._stats["last_trained_at"] = run_record["timestamp"]
                self._stats["history"].append(run_record)
                self._save_stats()

            msg = (
                f"✅ SigmaHead run #{self._stats['total_training_runs']} complete "
                f"in {elapsed:.1f}s — trained on {n} samples. Next query uses improved model!"
            )
            logger.info("[OnlineTrainer] %s", msg)
            self._emit("system_log", {"message": msg})
            self._emit("radiation_stats", self.get_stats())

        except Exception as exc:
            msg = f"❌ Background SigmaHead training failed: {exc}"
            logger.error("[OnlineTrainer] %s", msg, exc_info=True)
            self._emit("system_log", {"message": msg})
        finally:
            self._is_training.clear()
