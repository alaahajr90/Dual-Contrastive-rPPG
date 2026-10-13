from dataclasses import replace
import torch
from rppg.config import Settings
from rppg.stage1.losses import nt_xent
from rppg.stage2.losses import CompositeLoss, quality_train_mask, spectral_loss


def test_ntxent_positive_pair_gradient():
    a = torch.randn(4, 16, requires_grad=True)
    b = torch.randn(4, 16, requires_grad=True)
    loss = nt_xent(a, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_three_spectral_branches_differentiable():
    prediction = torch.randn(2, 128, requires_grad=True)
    reference = torch.randn(2, 128)
    for mode, fps in [("normalized_magnitude", 30), ("log_magnitude", 30),
                      ("band_log_normalized", 20)]:
        loss = spectral_loss(prediction, reference, mode=mode, fps=fps)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    assert torch.isfinite(prediction.grad).all()


def test_config_coefficients_and_ablations():
    cfg = Settings(dataset="pure")
    from rppg.experiments.ablations import ablation_config
    no_fft = ablation_config(cfg, "no_fft")
    assert cfg.use_fft and not no_fft.use_fft
    p = torch.randn(2, 128, requires_grad=True)
    target = torch.randn(2, 128)
    total, parts = CompositeLoss(replace(cfg, fft_weight=0.0))([p, p, p, p], target)
    assert torch.allclose(total, 2.5 * torch.nn.functional.smooth_l1_loss(p, target, beta=0.5))


def test_skew_training_only_signed():
    targets = torch.tensor([[0., 0., 0., 0., 10.], [-10., 0., 0., 0., 0.]])
    keep = quality_train_mask(targets, 1.15)
    assert keep.tolist() == [False, True]
