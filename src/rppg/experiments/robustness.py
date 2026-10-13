"""PURE condition mapping and inference-only, mean-filled synthetic occlusion."""
import numpy as np

PURE_CONDITIONS = {"01": "steady", "02": "talking", "03": "slow_translation",
                   "04": "fast_translation", "05": "small_rotation", "06": "medium_rotation"}


def condition_from_session(name):
    parts = str(name).replace("_", "-").split("-")
    if len(parts) < 2 or parts[1] not in PURE_CONDITIONS:
        raise ValueError(f"Unrecognized PURE condition naming: {name}")
    return PURE_CONDITIONS[parts[1]]


def apply_occlusion(video, fraction, location="center"):
    """Fill central square pixels with their per-clip, per-channel time/space mean.

    Input is [B,C,T,H,W] (or a compatible array); fraction represents masked
    *area* rather than width. At fraction=1, the entire ROI is suppressed.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("Occlusion fraction must be in [0,1]")
    x = video.clone() if hasattr(video, "clone") else np.asarray(video).copy()
    h, w = x.shape[-2:]
    side = int(round(np.sqrt(fraction) * min(h, w)))
    if not side:
        return x
    top = (h - side) // 2 if location == "center" else 0
    left = (w - side) // 2 if location == "center" else 0
    mean = x.mean(axis=(-3, -2, -1), keepdims=True) if isinstance(x, np.ndarray) else x.mean(dim=(-3, -2, -1), keepdim=True)
    x[..., top:top + side, left:left + side] = mean
    return x
