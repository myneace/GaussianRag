from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SigmaHead(nn.Module):
    """
    Neural network head that predicts the Gaussian covariance components
    (diagonal and low-rank matrix) from sentence-transformer hidden states.
    """
    def __init__(self, dimension: int, rank: int = 8, floor: float = 1e-6):
        super().__init__()
        self.dimension = dimension
        self.rank = rank
        self.floor = floor

        # MLP architecture
        self.fc_shared = nn.Linear(dimension, dimension)
        
        # Predicts the diagonal variance vector (positive)
        self.fc_diag = nn.Linear(dimension, dimension)
        
        # Predicts the low-rank covariance structure L (dimension x rank)
        self.fc_low_rank = nn.Linear(dimension, dimension * rank)

        # XRC-C Fix: Initialize bias so that initial variance is near 1.0.
        # softplus(0.5) ≈ 0.97. Starting with unit variance balances the ELK kernel 
        # terms (Mahalanobis vs Log-Det) before training begins.
        nn.init.constant_(self.fc_diag.bias, 0.5)
        # Initialize weights small to avoid random variance spikes
        nn.init.normal_(self.fc_diag.weight, std=0.001)
        nn.init.normal_(self.fc_low_rank.weight, std=0.001)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: (batch_size, dimension) tensor
        Returns:
            diag: (batch_size, dimension)
            low_rank: (batch_size, dimension, rank)
        """
        x = F.relu(self.fc_shared(hidden_state))
        
        # Diagonal must be strictly positive (enforced via softplus + floor)
        # Pass through softplus and clip to avoid extreme values
        diag = torch.clamp(F.softplus(self.fc_diag(x)), min=1e-4, max=1e2)
        
        # Low rank matrix can be any real numbers
        low_rank = self.fc_low_rank(x)
        low_rank = low_rank.view(-1, self.dimension, self.rank)
        low_rank = torch.clamp(low_rank, min=-10.0, max=10.0)
        
        return diag, low_rank

    def predict(self, hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """NumPy compatibility method for GaussianEmbedder inference."""
        self.eval()
        with torch.no_grad():
            tensor_hidden = torch.from_numpy(np.asarray(hidden_state, dtype=np.float32))
            if tensor_hidden.dim() == 1:
                tensor_hidden = tensor_hidden.unsqueeze(0)
            
            diag, low_rank = self.forward(tensor_hidden)
            
            # Return as squeezed numpy arrays for the embedding pipeline
            return diag.squeeze(0).numpy(), low_rank.squeeze(0).numpy()
