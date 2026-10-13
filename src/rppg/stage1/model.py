"""Original UBFC 3D-CNN encoder structure, cleaned for shared two-stage imports.

Four 16-channel feature groups are reused as P5, P6, P7 and Pout in Stage 2.
These groups originate from channels of the final 64-channel sequence in the
provided code; the manuscript calls them hierarchical representations, which
is not literally true for the available historical UBFC implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(channels, 16, (1, 5, 5), padding=(0, 2, 2)),
                                   nn.BatchNorm3d(16), nn.ReLU(inplace=True),
                                   nn.AvgPool3d((1, 2, 2)))
        self.convblock1 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1),
                                        nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                        nn.Conv3d(32, 32, 3, padding=1),
                                        nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                        nn.AvgPool3d((1, 2, 2)))
        self.convblock2 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1),
                                        nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                        nn.Conv3d(64, 64, 3, padding=1),
                                        nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                        nn.AvgPool3d((1, 2, 2)))
        self.convblock3 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1),
                                        nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                        nn.Conv3d(64, 64, 3, padding=1),
                                        nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("Expected [batch, 3, time, height, width]")
        x = self.pool(self.convblock3(self.convblock2(self.convblock1(self.conv1(x)))))
        return x.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B,T,64]


class ProjectionHead(nn.Module):
    """Two-layer nonlinear head for clip-level contrastive representation learning."""
    def __init__(self, input_dim=64, hidden_dim=512, output_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, seq):
        return F.normalize(self.net(seq.mean(dim=1)), dim=-1)


class ContrastiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        return self.projector(self.encoder(x))
