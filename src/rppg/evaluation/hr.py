"""Welch + band-pass + parabolic interpolation; session-level HR metrics."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def estimate_hr(signal, fps, low_hz=0.7, high_hz=3.0, nperseg=128, nfft=256):
    """Estimate bpm for a 128-frame predicted or reference clip.

    Welch uses a Hann window, constant detrending, 50% overlap, 256 FFT bins,
    and three-point interpolation of the dominant in-band spectral peak.
    """
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 128 or not np.isfinite(signal).all() or signal.std() < 1e-5:
        return float("nan")
    if high_hz >= fps / 2:
        raise ValueError("HR high cutoff must be below Nyquist")
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    sos = butter(3, [low_hz, high_hz], fs=fps, btype="bandpass", output="sos")
    signal = sosfiltfilt(sos, signal)
    freq, power = welch(signal, fs=fps, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=nfft, detrend="constant")
    selected = np.flatnonzero((freq >= low_hz) & (freq <= high_hz))
    if not selected.size:
        return float("nan")
    peak = int(selected[np.argmax(power[selected])])
    refined = float(freq[peak])
    if 0 < peak < len(power) - 1:
        l, c, r = power[peak - 1:peak + 2]
        denominator = l - 2 * c + r
        if abs(denominator) > 1e-15:
            offset = np.clip(0.5 * (l - r) / denominator, -1, 1)
            refined += float(offset) * fps / nfft
    return refined * 60


def session_estimates(clip_records):
    """Average clip-wise HR per session. Each record has pred_hr and gt_hr.

    A session is excluded explicitly if *no* valid clip HR pair exists.
    """
    grouped = defaultdict(list)
    for item in clip_records:
        if np.isfinite(item["pred_hr"]) and np.isfinite(item["gt_hr"]):
            grouped[(item["subject"], item["session"])].append(item)
    result = []
    for (subject, session), clips in sorted(grouped.items()):
        result.append({"subject": subject, "session": session,
                       "pred_hr": float(np.mean([c["pred_hr"] for c in clips])),
                       "gt_hr": float(np.mean([c["gt_hr"] for c in clips])),
                       "n_valid_clips": len(clips)})
    if not result:
        raise ValueError("No valid session HR estimates")
    return result


def metrics(records):
    true = np.array([s["gt_hr"] for s in records], dtype=float)
    pred = np.array([s["pred_hr"] for s in records], dtype=float)
    if true.size < 2:
        raise ValueError("At least two sessions required for Pearson correlation")
    if not (np.isfinite(true).all() and np.isfinite(pred).all()):
        raise ValueError("Nonfinite HR values; filter at clip level")
    err = pred - true
    r = float(np.corrcoef(pred, true)[0, 1]) if true.std() > 0 and pred.std() > 0 else float("nan")
    return {"mae_bpm": float(np.mean(np.abs(err))), "rmse_bpm": float(np.sqrt(np.mean(err ** 2))),
            "pearson_r": r, "n_sessions": len(records),
            "n_subjects": len({s["subject"] for s in records})}
