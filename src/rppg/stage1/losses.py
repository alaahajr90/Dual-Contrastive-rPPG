"""Symmetric NT-Xent with 2N-1 negatives and no margin/incorrect denominators.

The supplied historical UBFC code applied NT-Xent per frame with a 0.3 negative
margin and an unreachable return. This publication-facing implementation uses
the standard *clip-level* NT-Xent described by the manuscript. See AUDIT.md.
"""
import torch
from torch.nn import functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature=0.07):
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("Expected matching [batch, projection_dim] tensors")
    batch = z1.size(0)
    if batch < 2:
        raise ValueError("Contrastive loss needs at least two independent clips")
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    positives = torch.cat([torch.arange(batch, 2 * batch, device=z.device),
                           torch.arange(batch, device=z.device)])
    return F.cross_entropy(logits, positives)
