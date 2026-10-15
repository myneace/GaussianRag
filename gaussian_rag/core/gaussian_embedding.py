from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, cast

import numpy as np

from .covariance import (
    HeuristicSigmaEstimator,
    LearnedSigmaEstimator,
    DissolverSigmaEstimator,
    EntityAwareSigmaEstimator,
)
from .types import GaussianKnowledge, QueryRepresentation, SenseNode


class SentenceTransformerLike(Protocol):
    def encode(
        self,
        text: str,
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray: ...

    def get_embedding_dimension(self) -> int: ...


class _GeminiEmbeddingModel:
    """Thin wrapper that makes the Gemini Embedding 2 API look like SentenceTransformerLike.

    Uses Google's OpenAI-compatible endpoint so no extra SDK is required—only `openai`.
    """

    def __init__(self, api_key: str, model: str, base_url: str, dimensions: int | None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions  # None → let model return its default (3072)
        self._dim: int | None = None  # resolved lazily on first encode

    # ------------------------------------------------------------------
    # SentenceTransformerLike interface
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        import time
        import logging
        max_retries = 5
        base_delay = 1.0
        
        kwargs: dict[str, Any] = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
            
        for attempt in range(max_retries):
            try:
                response = self._client.embeddings.create(**kwargs)
                vec = np.array(response.data[0].embedding, dtype=float)
                if self._dim is None:
                    self._dim = vec.shape[0]
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                return vec
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                err_msg = str(e).lower()
                if "429" in err_msg or "rate limit" in err_msg or "timeout" in err_msg or "503" in err_msg or "500" in err_msg:
                    delay = base_delay * (2 ** attempt)
                    logging.getLogger(__name__).warning(f"Gemini API rate limit or transient error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    def get_embedding_dimension(self) -> int:
        if self._dim is None:
            # Trigger a dummy encode to discover the real dimension
            self.encode("dimension check", convert_to_numpy=True, normalize_embeddings=False)
        return self._dim or (self._dimensions or 3072)


@dataclass(slots=True)
class GaussianEmbedder:
    dimension: int = 384  # MiniLM default; overridden after model init
    sigma_mode: str = "heuristic"
    rank: int = 8
    entity_weight: float = 0.5  # only used when sigma_mode='entity_aware'
    # "local" (default) or "gemini_cloud"
    embedding_backend: str = "local"
    on_mismatch: str = "error"  # "error", "heuristic", "retrain"
    _heuristic: HeuristicSigmaEstimator = field(init=False, repr=False)
    _learned: LearnedSigmaEstimator = field(init=False, repr=False)
    _dissolver: DissolverSigmaEstimator = field(init=False, repr=False)
    _entity_aware: EntityAwareSigmaEstimator = field(init=False, repr=False)
    _model: SentenceTransformerLike = field(init=False, repr=False)
    on_event: Any = None

    def __post_init__(self) -> None:
        if self.embedding_backend == "gemini_cloud":
            self._init_gemini_model()
        else:
            self._init_local_model()


    # ------------------------------------------------------------------
    # Backend initializers
    # ------------------------------------------------------------------

    def _init_local_model(self) -> None:
        """Load all-MiniLM-L6-v2 locally via sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = cast(
                SentenceTransformerLike,
                SentenceTransformer("all-MiniLM-L6-v2"),
            )
            object.__setattr__(self, "dimension", self._model.get_embedding_dimension())
        except ImportError:
            raise ImportError("Please run: uv add sentence-transformers")
        self._finish_init()

    def _init_gemini_model(self) -> None:
        """Load Gemini Embedding 2 via Google's OpenAI-compatible endpoint."""
        import os
        from pathlib import Path
        from gaussian_rag.config import load_dotenv_file, load_project_config

        # Resolve API key: env > .env file
        api_key = os.environ.get("GEMINI_API_KEY") or load_dotenv_file().get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini Cloud embedding requires GEMINI_API_KEY. "
                "Set it in your environment or .env file. "
                "Get a free key at https://aistudio.google.com"
            )

        # Read cloud config from project config (with sensible defaults)
        try:
            proj_cfg = load_project_config()
            enc = proj_cfg.get("encoder", {})
            cloud_cfg = enc.get("gemini_cloud", {})
        except Exception:
            cloud_cfg = {}

        model = cloud_cfg.get("model", "gemini-embedding-2-preview")
        base_url = cloud_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
        dimensions = cloud_cfg.get("dimensions")  # None → model default

        try:
            import openai as _  # noqa: F401  – verify installed
        except ImportError:
            raise ImportError(
                "Gemini Cloud embedding requires the openai package. "
                "Run: uv add openai"
            )

        self._model = _GeminiEmbeddingModel(
            api_key=api_key,
            model=model,
            base_url=base_url,
            dimensions=dimensions,
        )
        object.__setattr__(self, "dimension", self._model.get_embedding_dimension())
        self._finish_init()

    def _finish_init(self) -> None:
        """Shared sigma-estimator setup. Called after the model is loaded."""
        self._heuristic = HeuristicSigmaEstimator(self._encode_vector, rank=self.rank)
        self._learned = LearnedSigmaEstimator(self.dimension, rank=self.rank)

        if self.sigma_mode == "learned":
            try:
                from gaussian_rag.training.sigma_head import SigmaHead
                import torch
                from pathlib import Path

                head = SigmaHead(dimension=self.dimension, rank=self.rank)
                weights_path = Path(f".gaussian_rag_weights_{self.dimension}.pt")
                if weights_path.exists():
                    try:
                        head.load_state_dict(torch.load(weights_path, weights_only=True))
                        import logging
                        msg = f"✅ Successfully loaded Learned Sigma Head weights from {weights_path}"
                        logging.getLogger(__name__).info(msg)
                        if self.on_event:
                            self.on_event("system_log", {"message": msg})
                    except RuntimeError as e:
                        if "size mismatch" in str(e):
                            import logging
                            if self.on_mismatch == "heuristic":
                                msg = f"⚠️ Dimension mismatch with {weights_path}. Configuration on_mismatch='heuristic' -> switching to heuristic mode."
                                logging.getLogger(__name__).warning(msg)
                                if self.on_event:
                                    self.on_event("system_log", {"message": msg})
                                self.sigma_mode = "heuristic"
                            elif self.on_mismatch == "retrain":
                                msg = f"⚠️ Dimension mismatch with {weights_path}. Configuration on_mismatch='retrain' -> initializing fresh weights."
                                logging.getLogger(__name__).warning(msg)
                                if self.on_event:
                                    self.on_event("system_log", {"message": msg})
                            else:
                                raise ValueError(f"Dimension mismatch loading {weights_path}. Set encoder.on_mismatch to 'heuristic' or 'retrain' in config.yaml to handle this automatically.") from e
                        else:
                            raise
                else:
                    import logging
                    msg = f"⚠️  sigma_mode is 'learned' but {weights_path} not found. Using untrained weights!"
                    logging.getLogger(__name__).warning(msg)
                    if self.on_event:
                        self.on_event("system_log", {"message": msg})
                
                self._learned.head = head
            except ImportError:
                import logging
                logging.getLogger(__name__).warning("torch not found. Learned mode requires torch.")

        self._dissolver = DissolverSigmaEstimator(self._encode_vector)
        self._entity_aware = EntityAwareSigmaEstimator(
            self._encode_vector, rank=self.rank, entity_weight=self.entity_weight
        )


    def _encode_vector(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.dimension, dtype=float)

        # sentence-transformers returns L2-normalized vectors by default for all-MiniLM
        # but we force normalization just to be safe for Wasserstein distance.
        embedding = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.astype(float)

    def _encode_vectors_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encodes a batch of texts concurrently to speed up processing."""
        unique_texts = list(set(texts))
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            def _enc(t):
                if not t.strip():
                    return np.zeros(self.dimension, dtype=float)
                return self._model.encode(t, convert_to_numpy=True, normalize_embeddings=True).astype(float)
            unique_embeddings = list(executor.map(_enc, unique_texts))
            
        mapping = dict(zip(unique_texts, unique_embeddings))
        return [mapping[t] for t in texts]

    def encode(
        self,
        text: str,
        *,
        item_id: str,
        metadata: dict[str, object] | None = None,
        interpretations: Sequence[str] | None = None,
    ) -> GaussianKnowledge:
        mu = self._encode_vector(text)
        if self.sigma_mode == "heuristic":
            sigma_diag, sigma_l = self._heuristic.estimate(text, mu)
        elif self.sigma_mode == "learned":
            sigma_diag, sigma_l = self._learned.estimate(mu)
        elif self.sigma_mode == "dissolver":
            if interpretations:
                sigma_diag, sigma_l = self._dissolver.estimate(text, interpretations)
            else:
                sigma_diag, sigma_l = self._heuristic.estimate(text, mu)
        elif self.sigma_mode == "entity_aware":
            sigma_diag, sigma_l = self._entity_aware.estimate(text, mu)
        else:
            raise ValueError(f"Unsupported sigma_mode: {self.sigma_mode}")
        return GaussianKnowledge(
            id=item_id,
            text=text,
            mu=mu,
            sigma_diag=sigma_diag,
            sigma_L=sigma_l,
            metadata=metadata or {},
        )

    def encode_batch(self, texts: list[str]) -> list[GaussianKnowledge]:
        return [
            self.encode(text, item_id=f"chunk-{index}")
            for index, text in enumerate(texts)
        ]

    def encode_query(
        self, query: str, *, with_covariance: bool = False
    ) -> QueryRepresentation:
        mu = self._encode_vector(query)
        if not with_covariance:
            return QueryRepresentation(text=query, mu=mu)
        if self.sigma_mode == "heuristic":
            sigma_diag, sigma_l = self._heuristic.estimate(query, mu)
        elif self.sigma_mode == "learned":
            sigma_diag, sigma_l = self._learned.estimate(mu)
        elif self.sigma_mode == "entity_aware":
            # Entity density is meaningful for queries — a polysemous query
            # ("apple bank python") should receive larger sigma than a precise one
            # ("mitral valve regurgitation"). Using the heuristic here was the
            # primary cause of ELK score collapse across all experiments.
            # estimate_query() uses token diversity × length saturation instead
            # of augmentation variance, which anti-correlates with query length.
            sigma_diag, sigma_l = self._entity_aware.estimate_query(query, mu)
        elif self.sigma_mode == "dissolver":
            # Dissolver requires explicit interpretation lists, which are not
            # available at query time — heuristic augmentation is the correct
            # fallback here.
            sigma_diag, sigma_l = self._heuristic.estimate(query, mu)
        else:
            raise ValueError(f"Unsupported sigma_mode: {self.sigma_mode}")
        return QueryRepresentation(
            text=query, mu=mu, sigma_diag=sigma_diag, sigma_L=sigma_l
        )

    def fit_sigma_head(self, texts: list[str]) -> None:
        hidden_states = [self._encode_vector(text) for text in texts if text.strip()]
        self._learned.fit(hidden_states)

    def encode_sense(self, sense: SenseNode) -> GaussianKnowledge:
        """Embed a SenseNode and scale Σ by (1 / (1 + confidence)).

        High-confidence senses → small Σ → tight Wasserstein ball → precise retrieval.
        Low-confidence senses  → large Σ → diffuse field → wide retrieval net.
        """
        base = self.encode(
            sense.retrieval_query,
            item_id=sense.sense_id,
            metadata={
                "domain": sense.domain,
                "term": sense.term,
                "sense_id": sense.sense_id,
            },
        )
        scale = 1.0 / (1.0 + max(sense.confidence, 0.0))
        return GaussianKnowledge(
            id=base.id,
            text=base.text,
            mu=base.mu,
            sigma_diag=base.sigma_diag * scale,
            sigma_L=base.sigma_L * np.sqrt(scale),
            metadata=base.metadata,
        )
