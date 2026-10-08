from __future__ import annotations

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder

from ..training.augmentation import generate_augmentations
from ..training.trainer import SigmaTrainer


def run(text: str) -> float:
    trainer = SigmaTrainer(GaussianEmbedder(sigma_mode="learned"))
    trainer.fit(generate_augmentations(text))
    return trainer.score_example(text)
