"""Central, validated configuration and reproducibility controls.

The manuscript is the source for reported hyperparameters. Dataset-specific
values not disclosed by it are explicit choices, never claimed to be historical.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import numpy as np
import torch
import yaml


@dataclass
class Settings:
    dataset: str
    raw_root: str = "data/raw"
    processed_root: str = "data/processed"
    output_root: str = "outputs"
    predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"
    clip_length: int = 128
    stride: int = 8
    roi_size: int = 64
    test_fraction: float = 0.4
    cv_folds: int = 5
    seed: int = 42
    stage1_epochs: int = 100
    stage1_batch_size: int = 16
    stage1_lr: float = 0.0003
    temperature: float = 0.07
    stage2_epochs: int = 185
    stage2_batch_size: int = 8
    stage2_lr: float = 0.0001
    stage2_patience: int = 3
    alpha: float = 0.5
    beta: float = 0.5
    fft_weight: float = 1.0
    stat_weight: float = 0.0
    smooth_weight: float = 0.0
    skew_threshold: float = 1.15
    fps: float = 30.0
    hr_low_hz: float = 0.7
    hr_high_hz: float = 3.0
    spectral_mode: str = "normalized_magnitude"
    memory_blocks: int = 4
    long_range_blocks: int = 4
    use_ssl: bool = True
    use_rope: bool = True
    use_memory: bool = True
    use_long_range: bool = True
    use_fft: bool = True
    amp: bool = False
    num_workers: int = 0
    split_path: str = ""
    clip_manifest: str = ""
    pretrained_encoder: str = ""
    # Used only if matching archived provenance exists. This is not a claim that
    # the manuscript discloses exact training-learning rate or stop rule for all data.
    info: dict = field(default_factory=dict)

    def validate(self):
        if self.dataset not in {"pure", "cohface", "ubfc"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if (self.clip_length, self.stride, self.roi_size) != (128, 8, 64):
            raise ValueError("Manuscript protocol requires 128 frames, stride 8, 64x64 ROI")
        if self.test_fraction != 0.4 or self.cv_folds != 5:
            raise ValueError("Protocol requires 60% development / 40% test and five folds")
        if self.beta <= 0 or self.stage1_batch_size < 2 or self.stage2_batch_size < 1:
            raise ValueError("Invalid batch size or loss beta")
        if self.memory_blocks < 1 or self.long_range_blocks < 1:
            raise ValueError("Invalid attention depth")
        if self.spectral_mode not in {"normalized_magnitude", "log_magnitude", "band_log_normalized"}:
            raise ValueError("Invalid dataset-specific FFT mode")
        if not (0 < self.hr_low_hz < self.hr_high_hz < self.fps / 2):
            raise ValueError("Invalid HR frequency band relative to FPS")
        return self


def load_settings(path: str | Path) -> Settings:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return Settings(**payload).validate()


def save_settings(settings: Settings, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(settings), indent=2) + "\n")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
