"""Multi-head memory attention with RoPE applied to queries and keys.

PyTorch attention is implemented explicitly here so rotary embeddings affect
Q/K before the attention dot products, not input embeddings only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k):
    # q/k: [B,heads,T,head_dim]
    t, d = q.size(-2), q.size(-1)
    if d % 2:
        raise ValueError("RoPE requires even head dimension")
    positions = torch.arange(t, device=q.device, dtype=torch.float32)
    inverse = torch.arange(0, d, 2, device=q.device, dtype=torch.float32)
    inverse = torch.pow(10000.0, -inverse / d)
    angles = torch.outer(positions, inverse)
    emb = torch.repeat_interleave(angles, 2, dim=-1).to(q.dtype)
    sin, cos = emb.sin()[None, None], emb.cos()[None, None]
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class RotarySelfAttention(nn.Module):
    def __init__(self, dim=16, heads=8, dropout=0.1, use_rope=True):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention dimension must be divisible by heads")
        self.dim, self.heads = dim, heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = apply_rope(q, k)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        return self.out(attn.transpose(1, 2).reshape(b, t, c))


class MemoryBlock(nn.Module):
    def __init__(self, dim=16, heads=8, dropout=0.1, use_rope=True):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = RotarySelfAttention(dim, heads, dropout, use_rope)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 8), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 8, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MemoryAttention(nn.Module):
    def __init__(self, depth=4, dim=16, heads=8, dropout=0.1, use_rope=True):
        super().__init__()
        self.layers = nn.ModuleList([MemoryBlock(dim, heads, dropout, use_rope)
                                     for _ in range(depth)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x
