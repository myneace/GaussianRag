import sys
import logging
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.config import load_project_config
from gaussian_rag.training.trainer import SigmaTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    print("Initializing GaussianEmbedder in 'learned' mode...")
    config = load_project_config()
    encoder_config = config["encoder"]
    embedder = GaussianEmbedder(
        dimension=encoder_config["dimension"],
        sigma_mode="learned",
        rank=encoder_config["low_rank"],
        embedding_backend=encoder_config.get("embedding_backend", "local"),
        on_mismatch=encoder_config.get("on_mismatch", "error"),
    )
    trainer = SigmaTrainer(embedder)

    import json
    from pathlib import Path
    from gaussian_rag.training.trainer import DomainTextDataset

    jsonl_path = Path("data/training/robust_corpus.jsonl")

    if not jsonl_path.exists():
        print(f"Error: Robust training corpus not found at {jsonl_path}")
        print("Please create it with JSONL structure: {'text': '...', 'domain': '...'}")
        sys.exit(1)

    print(f"Loading robust training corpus from {jsonl_path}...")
    dataset = DomainTextDataset.from_jsonl(jsonl_path, embedder)

    print(f"Starting self-supervised training loop...")
    trainer.fit(dataset, epochs=10, batch_size=4)

    weights_path = f".gaussian_rag_weights_{embedder.dimension}.pt"
    trainer.save_model(weights_path)
    print(f"\nTraining complete! Weights saved to {weights_path}")


if __name__ == "__main__":
    main()
