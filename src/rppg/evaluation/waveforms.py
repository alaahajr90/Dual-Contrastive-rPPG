"""Paired ground-truth/predicted waveform and spectral figures from real test clips.

The selected test clips are visual examples, not independent evaluation units.
No processed RGB frames or participant-identifying imagery is exported here.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.signal import welch
from torch.utils.data import DataLoader
from .hr import estimate_hr
from ..preprocessing.dataset import ClipDataset, select_rows
from ..stage2.model import RPPGModel


def generate_waveform_figures(cfg, manifest_rows, split, checkpoint, destination, examples=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if examples < 1:
        raise ValueError("examples must be positive")
    subject_rows = select_rows(manifest_rows, split["test"])
    selected = []
    # Select different sessions where possible instead of neighboring overlapping clips.
    unique_sessions = set()
    for row in subject_rows:
        identity = (row["subject"], row["session"])
        if identity not in unique_sessions:
            selected.append(row)
            unique_sessions.add(identity)
        if len(selected) >= examples:
            break
    if not selected:
        raise ValueError("No held-out samples")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model = RPPGModel(memory_blocks=cfg.memory_blocks, long_range_blocks=cfg.long_range_blocks,
                      use_rope=cfg.use_rope, use_memory=cfg.use_memory,
                      use_long_range=cfg.use_long_range)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    record = []
    with torch.no_grad():
        loader = DataLoader(ClipDataset(selected, stage=2), batch_size=1, shuffle=False)
        for idx, (video, truth, subjects, sessions, fps) in enumerate(loader):
            predicted = model(video)[-1][0].numpy()
            true = truth[0].numpy()
            sampling = float(fps[0])
            time = np.arange(len(predicted)) / sampling
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(time, true, label="Reference BVP")
            ax.plot(time, predicted, label="Predicted rPPG", alpha=0.8)
            ax.set(title=f"Waveform pair {idx + 1}", xlabel="Time (s)", ylabel="Normalized amplitude")
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / f"waveform_example_{idx + 1}.pdf")
            plt.close(fig)
            freq_true, psd_true = welch(true, sampling, nperseg=128, nfft=256)
            freq_pred, psd_pred = welch(predicted, sampling, nperseg=128, nfft=256)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(freq_true, psd_true, label="Reference PSD")
            ax.plot(freq_pred, psd_pred, label="Predicted PSD", alpha=0.8)
            ax.set_xlim(cfg.hr_low_hz, cfg.hr_high_hz)
            ax.set(title=f"Spectral pair {idx + 1}", xlabel="Frequency (Hz)", ylabel="Power/Hz")
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / f"spectrum_example_{idx + 1}.pdf")
            plt.close(fig)
            record.append({"subject": subjects[0], "session": sessions[0],
                           "pred_hr": estimate_hr(predicted, sampling, cfg.hr_low_hz, cfg.hr_high_hz),
                           "gt_hr": estimate_hr(true, sampling, cfg.hr_low_hz, cfg.hr_high_hz)})
    return record
