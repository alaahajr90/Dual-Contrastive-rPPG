"""Contrastive video views: clip-consistent spatial perturbations and optional temporal change.

Unlike the legacy UBFC script, no NumPy repeat axis bug is present, and JPEG
encoding uses correct RGB/BGR conversion. All randomness is seeded by workers.
"""
from __future__ import annotations
import random
import cv2
import numpy as np


class VideoAugment:
    def __init__(self, temporal=True):
        self.temporal = temporal

    def __call__(self, input_clip):
        clip = np.asarray(input_clip, dtype=np.float32).copy()
        t, h, w, c = clip.shape
        if c != 3:
            raise ValueError("Expected [T,H,W,3]")
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.6, 0.9)
        dx, dy = random.uniform(-0.02, 0.02) * w, random.uniform(-0.02, 0.02) * h
        transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0 / scale)
        transform[:, 2] += [dx, dy]
        clip = np.stack([cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT_101) for frame in clip])
        if random.random() < 0.5:
            clip = clip[:, :, ::-1, :]
        brightness = random.uniform(-0.2, 0.2) * 255
        contrast = random.uniform(0.8, 1.2)
        tone = np.asarray([random.uniform(0.9, 1.1) for _ in range(3)])
        clip = np.clip(clip * contrast * tone + brightness, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            clip = np.stack([cv2.GaussianBlur(frame, (3, 3), 0) for frame in clip])
        if random.random() < 0.3:
            quality = random.randint(30, 60)
            encoded = []
            for rgb in clip:
                ok, raw = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise ValueError("JPEG compression simulation failed")
                encoded.append(cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            clip = np.stack(encoded)
        if self.temporal:
            method = random.choice(("none", "reverse", "cycle", "reorder"))
            if method == "reverse":
                clip = clip[::-1]
            elif method == "cycle":
                clip = np.roll(clip, shift=random.randint(1, t - 1), axis=0)
            elif method == "reorder":
                blocks = np.array_split(np.arange(t), 8)
                blocks = random.sample(blocks, len(blocks))
                clip = clip[np.concatenate(blocks)]
        return np.ascontiguousarray(clip)
