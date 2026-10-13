"""Single clip-loading implementation reused by both training stages and all datasets."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from .augmentation import VideoAugment


class ClipDataset(Dataset):
    def __init__(self, rows, stage: int, augment: bool = False):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.rows = list(rows)
        self.stage = stage
        self.augmentation = VideoAugment() if augment and stage == 1 else None
        if not self.rows:
            raise ValueError("Empty dataset partition")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        video = np.load(row["video"], mmap_mode="r")
        if video.shape != (128, 64, 64, 3):
            raise ValueError(f"Incorrect ROI shape: {video.shape}: {row['video']}")
        if self.stage == 1:
            aug = self.augmentation or (lambda x: np.asarray(x))
            first = aug(video)
            second = aug(video)
            return self._tensor(first), self._tensor(second)
        y = np.asarray(np.load(row["bvp"]), dtype=np.float32).reshape(-1)
        if y.shape != (128,) or not np.isfinite(y).all():
            raise ValueError(f"Invalid BVP waveform for {row['clip_id']}")
        # The ground-truth normalization is applied once, identically across datasets.
        y = (y - y.mean()) / (y.std() + 1e-5)
        return self._tensor(video), torch.from_numpy(y.copy()), row["subject"], row["session"], float(row["fps"])

    @staticmethod
    def _tensor(video):
        array = np.asarray(video, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array.transpose(3, 0, 1, 2)))


def select_rows(rows, subjects):
    allowed = set(subjects)
    result = [r for r in rows if r["subject"] in allowed]
    if not result:
        raise ValueError(f"No clips for requested subjects: {sorted(allowed)}")
    return result
