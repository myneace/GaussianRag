from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.training.sigma_head import SigmaHead
from gaussian_rag.training.losses import contrastive_w2_loss, covariance_regularization_loss
from .augmentation import generate_augmentations

logger = logging.getLogger(__name__)


class DomainTextDataset(Dataset):
    """A dataset for loading structured text data with domain info for SigmaHead training."""
    def __init__(self, items: list[dict[str, Any]], embedder: GaussianEmbedder):
        self.items = items
        self.embedder = embedder
        
    @classmethod
    def from_jsonl(cls, path: str | Path, embedder: GaussianEmbedder) -> DomainTextDataset:
        items = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
                    
        unique_texts = set()
        for item in items:
            text = item["text"]
            views = generate_augmentations(text)
            if len(views) < 2:
                anchor_text, pos_text = text, text
            else:
                anchor_text, pos_text = views[0], views[1]
            item["_anchor_text"] = anchor_text
            item["_pos_text"] = pos_text
            unique_texts.add(anchor_text)
            unique_texts.add(pos_text)
            
        unique_texts_list = list(unique_texts)
        if unique_texts_list:
            logger.info(f"Pre-encoding {len(unique_texts_list)} unique texts concurrently...")
            embeddings = embedder._encode_vectors_batch(unique_texts_list)
            precomputed = dict(zip(unique_texts_list, embeddings))
            
            for item in items:
                item["_anchor_vec"] = precomputed[item["_anchor_text"]]
                item["_pos_vec"] = precomputed[item["_pos_text"]]
                
        return cls(items, embedder)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        domain = item.get("domain", "unknown")
        
        if "_anchor_vec" in item and "_pos_vec" in item:
            anchor_hidden = item["_anchor_vec"]
            pos_hidden = item["_pos_vec"]
        else:
            text = item["text"]
            views = generate_augmentations(text)
            if len(views) < 2:
                anchor_text, pos_text = text, text
            else:
                anchor_text, pos_text = views[0], views[1]
            anchor_hidden = self.embedder._encode_vector(anchor_text)
            pos_hidden = self.embedder._encode_vector(pos_text)
        
        return {
            "anchor_hidden": torch.tensor(anchor_hidden, dtype=torch.float32),
            "pos_hidden": torch.tensor(pos_hidden, dtype=torch.float32),
            "domain": domain,
            "index": idx
        }



@dataclass(slots=True)
class SigmaTrainer:
    """Self-supervised training loop for the Learned Covariance Head."""
    embedder: GaussianEmbedder
    head: SigmaHead = field(init=False)
    optimizer: optim.Optimizer = field(init=False)
    # RC-12 fix: default is None (not 0.0) so callers can distinguish
    # "not yet trained" from "trained to zero loss".  The extraction
    # script treats 0.0 as None (the dataclass default) which was the
    # original symptom of loss=None appearing in later epoch rows.
    last_epoch_loss: float | None = field(init=False, default=None)
    
    def __post_init__(self):
        # Initialize PyTorch module
        self.head = SigmaHead(
            dimension=self.embedder.dimension,
            rank=self.embedder.rank
        )
        self.optimizer = optim.AdamW(self.head.parameters(), lr=5e-5, weight_decay=1e-2)
        
        # Wire it into the embedder if it's currently using the learned mode
        if self.embedder.sigma_mode == "learned":
            self.embedder._learned.head = self.head

    def fit(self, texts: list[str] | Dataset, epochs: int = 3, batch_size: int = 4) -> None:
        """Train the covariance head using self-supervised augmented views."""
        self.head.train()
        
        if isinstance(texts, list):
            # Wrap simple list in a basic dataset if needed
            items = [{"text": t} for t in texts]
            dataset = DomainTextDataset(items, self.embedder)
        else:
            dataset = texts
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            
            for batch in loader:
                self.optimizer.zero_grad()
                
                a_h_batch = batch["anchor_hidden"]
                p_h_batch = batch["pos_hidden"]
                
                # 1. Predict covariance structures
                a_diag, a_L = self.head(a_h_batch)
                p_diag, p_L = self.head(p_h_batch)
                
                # Construct full covariance matrices
                a_cov = torch.diag_embed(a_diag) + a_L @ a_L.transpose(-2, -1)
                p_cov = torch.diag_embed(p_diag) + p_L @ p_L.transpose(-2, -1)
                
                # 2. Grab negative samples (in-batch negatives)
                # We shift the batch by 1 to get a negative for each anchor that is 
                # a different document in the same batch.
                if a_h_batch.size(0) > 1:
                    neg_mus_tensor = torch.roll(a_h_batch, shifts=1, dims=0)
                    neg_mus = [neg_mus_tensor]
                    
                    # We need the predicted covariance for the negatives too
                    n_diag, n_L = self.head(neg_mus_tensor)
                    n_cov = torch.diag_embed(n_diag) + n_L @ n_L.transpose(-2, -1)
                    neg_covs = [n_cov]
                else:
                    neg_mus = []
                    neg_covs = []
                
                # 3. Compute Loss
                loss_contrastive = contrastive_w2_loss(
                    a_h_batch, a_cov,
                    p_h_batch, p_cov,
                    neg_mus, neg_covs
                )
                
                loss_reg = covariance_regularization_loss(a_diag, a_L)
                
                loss = loss_contrastive + loss_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                steps += 1
                
            if steps > 0:
                avg_loss = total_loss / steps
                self.last_epoch_loss = avg_loss
                logger.info(f"SigmaHead Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
            else:
                # Empty dataset or all batches skipped — set explicitly to None
                # so callers can detect the no-training condition.
                self.last_epoch_loss = None
                logger.warning(f"SigmaHead Epoch {epoch+1}/{epochs} - no batches processed, loss=None")

    def score_example(self, text: str) -> float:
        """Stub for consistency check (not implemented in PyTorch yet)."""
        return 0.0

    def save_model(self, path: str) -> None:
        """Save the trained PyTorch weights."""
        torch.save(self.head.state_dict(), path)
        logger.info(f"Model weights saved to {path}")
