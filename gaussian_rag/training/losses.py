from __future__ import annotations

import torch


def _sqrtm_spd(matrix: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    """Differentiable matrix square root for symmetric positive-definite matrices."""
    # Ensure symmetry
    matrix = (matrix + matrix.transpose(-2, -1)) / 2.0
    
    # Use float64 for better precision during eigh to prevent NaNs
    original_dtype = matrix.dtype
    matrix_f64 = matrix.double()
    
    # Add asymmetric jitter to the diagonal to ensure strict positive-definiteness
    # Scale jitter by matrix magnitude to avoid it being lost to fp truncation
    dim = matrix_f64.size(-1)
    mag = torch.clamp(torch.abs(matrix_f64.diagonal(dim1=-2, dim2=-1).mean(-1)), min=1.0)
    # create shape for broadcasting
    mag = mag.unsqueeze(-1).unsqueeze(-1)
    
    jitter_vals = torch.linspace(floor, floor * 10, dim, device=matrix.device, dtype=torch.float64)
    jitter = torch.diag_embed(jitter_vals).unsqueeze(0) # (1, dim, dim)
    
    # Broadcast jitter across batch and scale by matrix magnitude
    matrix_f64 = matrix_f64 + (jitter * mag)
    
    L, V = torch.linalg.eigh(matrix_f64)
    # Clamp eigenvalues to prevent NaN gradients from sqrt(0)
    L = torch.clamp(L, min=floor)
    
    sqrt_L = torch.diag_embed(torch.sqrt(L))
    res = V @ sqrt_L @ V.transpose(-2, -1)
    
    # Return to original precision
    return res.to(original_dtype)



def torch_wasserstein2(
    mu1: torch.Tensor, cov1: torch.Tensor, 
    mu2: torch.Tensor, cov2: torch.Tensor
) -> torch.Tensor:
    """
    Differentiable Wasserstein-2 distance between two Gaussians.
    Computes: ||mu1 - mu2||^2 + Tr(C1) + Tr(C2) - 2 * Tr((C1^{1/2} C2 C1^{1/2})^{1/2})
    """
    mean_term = torch.sum((mu1 - mu2) ** 2, dim=-1)
    
    sqrt_c1 = _sqrtm_spd(cov1)
    middle = sqrt_c1 @ cov2 @ sqrt_c1
    
    trace_c1 = torch.diagonal(cov1, dim1=-2, dim2=-1).sum(-1)
    trace_c2 = torch.diagonal(cov2, dim1=-2, dim2=-1).sum(-1)
    trace_middle = torch.diagonal(_sqrtm_spd(middle), dim1=-2, dim2=-1).sum(-1)
    
    trace_term = trace_c1 + trace_c2 - 2.0 * trace_middle
    
    # Ensure strict positivity before sqrt to prevent NaNs
    return torch.sqrt(torch.clamp(mean_term + trace_term, min=1e-9))


def contrastive_w2_loss(
    anchor_mu: torch.Tensor, anchor_cov: torch.Tensor,
    positive_mu: torch.Tensor, positive_cov: torch.Tensor,
    negative_mus: list[torch.Tensor], negative_covs: list[torch.Tensor],
    margin: float = 0.5,
) -> torch.Tensor:
    """Contrastive loss pulling positive views together and pushing negatives apart."""
    positive_distance = torch_wasserstein2(anchor_mu, anchor_cov, positive_mu, positive_cov)
    
    if not negative_mus:
        return positive_distance.mean()
        
    penalties = []
    for n_mu, n_cov in zip(negative_mus, negative_covs):
        neg_dist = torch_wasserstein2(anchor_mu, anchor_cov, n_mu, n_cov)
        # Margin ranking loss component
        penalty = torch.clamp(margin + positive_distance - neg_dist, min=0.0)
        penalties.append(penalty)
        
    avg_penalty = torch.stack(penalties).mean(dim=0)
    return (positive_distance + avg_penalty).mean()


def covariance_regularization_loss(
    sigma_diag: torch.Tensor,
    sigma_L: torch.Tensor,
    lambda1: float = 0.01,
    lambda2: float = 0.001,
) -> torch.Tensor:
    """L1/L2 regularization to prevent covariance from exploding."""
    return lambda1 * torch.sum(sigma_diag) + lambda2 * torch.sum(sigma_L * sigma_L)
