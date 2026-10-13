"""Four-branch rPPG decoder with paper-specified 4+4 attention depth.

The model returns four waveform predictions [P5,P6,P7,Pout]. Ablation flags
are architectural, not ad-hoc copies of source files. No checkpoints ship.
"""
from __future__ import annotations
import torch
from torch import nn
from .attention import MemoryAttention
from ..stage1.model import Encoder


def normalize_waveform(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, unbiased=False) + 1e-5)


class WaveformHead(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(32, 1))

    def forward(self, seq):
        return normalize_waveform(self.net(seq).squeeze(-1))


class RPPGModel(nn.Module):
    def __init__(self, encoder=None, memory_blocks=4, long_range_blocks=4,
                 use_rope=True, use_memory=True, use_long_range=True):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.use_memory = use_memory
        self.use_long_range = use_long_range
        self.memory = nn.ModuleList([MemoryAttention(memory_blocks, use_rope=use_rope)
                                     for _ in range(4)])
        self.long_range = MemoryAttention(long_range_blocks, use_rope=use_rope)
        self.heads = nn.ModuleList([WaveformHead() for _ in range(4)])

    def forward(self, video):
        sequence = self.encoder(video)
        if sequence.shape[-1] != 64:
            raise ValueError("Stage 2 requires the shared 64-channel Stage-1 encoder")
        branches = list(sequence.chunk(4, dim=-1))
        result = []
        for i, features in enumerate(branches):
            if self.use_memory:
                features = self.memory[i](features)
            if i == 3 and self.use_long_range:
                features = self.long_range(features)
            result.append(self.heads[i](features))
        return result
