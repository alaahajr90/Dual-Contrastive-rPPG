"""Manuscript Table 3: dataset-specific SmoothL1, FFT and optional regularizers.

The pure/UBFC/COHFACE spectral *families* and coefficients match the text.
Unpublished details (windowing, smoothing finite-difference definition) are
implementation decisions, not a claim of bitwise historical reproduction.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def standardize(x, eps=1e-6):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, unbiased=False,
                                                          keepdim=True) + eps)


def spectral_loss(pred, truth, *, mode="normalized_magnitude", fps=30.0,
                  low_hz=0.7, high_hz=4.0, eps=1e-6):
    pred, truth = standardize(pred), standardize(truth)
    spec1 = torch.fft.rfft(pred, dim=-1).abs()
    spec2 = torch.fft.rfft(truth, dim=-1).abs()
    if mode == "normalized_magnitude":
        spec1 = spec1 / (spec1.sum(dim=-1, keepdim=True) + eps)
        spec2 = spec2 / (spec2.sum(dim=-1, keepdim=True) + eps)
    elif mode == "log_magnitude":
        spec1 = torch.log(spec1 + eps)
        spec2 = torch.log(spec2 + eps)
    elif mode == "band_log_normalized":
        freq = torch.fft.rfftfreq(pred.size(-1), d=1.0 / fps).to(pred.device)
        band = (freq >= low_hz) & (freq <= high_hz)
        if not band.any():
            raise ValueError("No FFT bins in physiological HR band")
        spec1 = torch.log(spec1[..., band] + eps)
        spec2 = torch.log(spec2[..., band] + eps)
        spec1 = (spec1 - spec1.mean(-1, keepdim=True)) / (spec1.std(-1, keepdim=True, unbiased=False) + eps)
        spec2 = (spec2 - spec2.mean(-1, keepdim=True)) / (spec2.std(-1, keepdim=True, unbiased=False) + eps)
    else:
        raise ValueError(f"Unsupported spectral mode: {mode}")
    return F.l1_loss(spec1, spec2)


def signal_statistics_loss(pred, truth):
    return F.mse_loss(pred.std(-1, unbiased=False), truth.std(-1, unbiased=False))


def temporal_smoothness_loss(pred):
    return (pred[..., 1:] - pred[..., :-1]).abs().mean()


class CompositeLoss:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, outputs, truth):
        if len(outputs) != 4:
            raise ValueError("Expected P5, P6, P7 and Pout predictions")
        parts = [F.smooth_l1_loss(out, truth, beta=self.cfg.beta) for out in outputs]
        base = parts[3] + self.cfg.alpha * sum(parts[:3])
        fft = spectral_loss(outputs[3], truth, mode=self.cfg.spectral_mode,
                            fps=self.cfg.fps, low_hz=self.cfg.hr_low_hz,
                            high_hz=self.cfg.hr_high_hz) if self.cfg.use_fft else base.new_zeros(())
        stat = signal_statistics_loss(outputs[3], truth) if self.cfg.stat_weight else base.new_zeros(())
        smooth = temporal_smoothness_loss(outputs[3]) if self.cfg.smooth_weight else base.new_zeros(())
        total = base + self.cfg.fft_weight * fft + self.cfg.stat_weight * stat + self.cfg.smooth_weight * smooth
        return total, {"base": base.detach(), "fft": fft.detach(), "stat": stat.detach(),
                       "smooth": smooth.detach(), "total": total.detach()}


def signed_skewness(batch, eps=1e-8):
    mean = batch.mean(-1, keepdim=True)
    center = batch - mean
    std = (center.square().mean(-1) + eps).sqrt()
    return center.pow(3).mean(-1) / (std.pow(3) + eps)


def quality_train_mask(targets, threshold):
    """Historical signed criterion: exclude samples with skewness > threshold ONLY in training."""
    return signed_skewness(targets) <= threshold
