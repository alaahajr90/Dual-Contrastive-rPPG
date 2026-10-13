import numpy as np
from rppg.evaluation.hr import estimate_hr, metrics, session_estimates
from rppg.evaluation.statistics import subject_bootstrap
from rppg.preprocessing.augmentation import VideoAugment


def test_welch_hr_with_sinusoid():
    fps = 30
    signal = np.sin(2 * np.pi * 1.25 * np.arange(128) / fps)
    assert abs(estimate_hr(signal, fps) - 75.0) < 3.0


def test_session_average_not_clip_independence():
    clips = [{"subject": "a", "session": "s1", "pred_hr": 70, "gt_hr": 72},
             {"subject": "a", "session": "s1", "pred_hr": 74, "gt_hr": 72},
             {"subject": "b", "session": "s2", "pred_hr": 90, "gt_hr": 91}]
    session = session_estimates(clips)
    assert len(session) == 2
    assert session[0]["pred_hr"] == 72
    assert metrics(session)["mae_bpm"] == 0.5
    report = subject_bootstrap(session, repetitions=100, seed=7)
    assert report["n_bootstrap"] == 100
    assert "mae_bpm" in report["confidence_intervals_95"]


def test_augment_keeps_frames_and_color_channels():
    source = np.zeros((16, 32, 32, 3), dtype=np.uint8)
    source[..., 0] = 90
    result = VideoAugment()(source)
    assert result.shape == source.shape
    assert result.dtype == np.uint8


def test_occlusion_paper_25_to_100_percent_and_mean_fill():
    import torch
    from rppg.experiments.robustness import apply_occlusion
    video = torch.rand(1, 3, 8, 64, 64)
    full = apply_occlusion(video, 1.0)
    channel_mean = video.mean(dim=(2, 3, 4), keepdim=True)
    assert torch.allclose(full, channel_mean.expand_as(full))
    assert apply_occlusion(video, 0.25).shape == video.shape
