"""Shared optimizers, early stopping, safe weight transfers and experiment artifacts."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import device_auto, set_seed, save_settings
from .preprocessing.dataset import ClipDataset, select_rows
from .stage1.model import ContrastiveNetwork, Encoder
from .stage1.losses import nt_xent
from .stage2.model import RPPGModel
from .stage2.losses import CompositeLoss, quality_train_mask
from .evaluation.hr import estimate_hr, session_estimates, metrics


def _loader(rows, stage, bs, *, shuffle=False, num_workers=0):
    return DataLoader(ClipDataset(rows, stage, augment=(stage == 1 and shuffle)),
                      batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=(stage == 1 and shuffle))


def _save_weights(path, state):
    # Only newly trained weights are written to the user's private outputs.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_stage1(cfg, rows, split, *, out=None):
    """Train SSL without any BVP loading, including during validation."""
    set_seed(cfg.seed)
    device = device_auto()
    run = Path(out or cfg.output_root) / cfg.dataset / "stage1"
    run.mkdir(parents=True, exist_ok=True)
    save_settings(cfg, run / "config.json")
    model = ContrastiveNetwork().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.stage1_lr)
    train = _loader(select_rows(rows, split["train"]), 1, cfg.stage1_batch_size,
                    shuffle=True, num_workers=cfg.num_workers)
    val = _loader(select_rows(rows, split["val"]), 1, cfg.stage1_batch_size,
                  num_workers=cfg.num_workers)
    best, history = float("inf"), []
    for epoch in range(1, cfg.stage1_epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for view1, view2 in train:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer_loss = nt_xent(model(view1), model(view2), cfg.temperature)
            optimizer_loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            loss_sum += optimizer_loss.item() * len(view1)
            n += len(view1)
        if not n:
            raise RuntimeError("Training set too small for Stage-1 contrastive batch size")
        model.eval()
        val_sum, v = 0.0, 0
        with torch.no_grad():
            for a, b in val:
                if len(a) < 2:
                    continue
                loss = nt_xent(model(a.to(device)), model(b.to(device)), cfg.temperature)
                val_sum += loss.item() * len(a)
                v += len(a)
        if not v:
            raise RuntimeError("Validation has no valid contrastive minibatch")
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_sum / v}
        history.append(row)
        print(json.dumps({"stage": 1, **row}), flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            _save_weights(run / "encoder_best.pth", model.encoder.state_dict())
    _write_rows(history, run / "history.csv")
    return run / "encoder_best.pth"


def _model(cfg, *, pretrained_encoder=None):
    encoder = Encoder()
    if cfg.use_ssl:
        if not pretrained_encoder:
            raise ValueError("Stage 2 requires an actual Stage-1 encoder unless --ablation no_ssl")
        state = torch.load(pretrained_encoder, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=True)
    model = RPPGModel(encoder, memory_blocks=cfg.memory_blocks,
                      long_range_blocks=cfg.long_range_blocks, use_rope=cfg.use_rope,
                      use_memory=cfg.use_memory, use_long_range=cfg.use_long_range)
    return model


def _batch_loss(model, batch, cfg, criterion, device, *, training):
    videos, targets = batch[0].to(device), batch[1].to(device)
    if training:
        keep = quality_train_mask(targets, cfg.skew_threshold)
        if not keep.any():
            return None, int(len(keep)), 0
        videos, targets = videos[keep], targets[keep]
        discarded = int((~keep).sum().item())
    else:
        discarded = 0  # Important: validation and held-out test remain unfiltered.
    output = model(videos)
    loss, _ = criterion(output, targets)
    return loss, discarded, len(videos)


def train_stage2(cfg, rows, split, *, pretrained_encoder=None, out=None):
    set_seed(cfg.seed)
    device = device_auto()
    run = Path(out or cfg.output_root) / cfg.dataset / "stage2" / f"seed_{cfg.seed}"
    run.mkdir(parents=True, exist_ok=True)
    save_settings(cfg, run / "config.json")
    model = _model(cfg, pretrained_encoder=pretrained_encoder).to(device)
    criterion = CompositeLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.stage2_lr)
    train = _loader(select_rows(rows, split["train"]), 2, cfg.stage2_batch_size,
                    shuffle=True, num_workers=cfg.num_workers)
    val = _loader(select_rows(rows, split["val"]), 2, cfg.stage2_batch_size,
                  num_workers=cfg.num_workers)
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, cfg.stage2_epochs + 1):
        totals = {}
        model.train()
        loss_sum, samples, excluded = 0.0, 0, 0
        for batch in train:
            loss, dropped, retained = _batch_loss(model, batch, cfg, criterion, device, training=True)
            excluded += dropped
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * retained
            samples += retained
        if not samples:
            raise RuntimeError("Training-only BVP quality filter excluded all training clips")
        model.eval()
        val_loss_sum, val_samples = 0.0, 0
        with torch.no_grad():
            for batch in val:
                loss, _, retained = _batch_loss(model, batch, cfg, criterion, device, training=False)
                val_loss_sum += float(loss.item()) * retained
                val_samples += retained
        current = val_loss_sum / val_samples
        status = {"epoch": epoch, "train_loss": loss_sum / samples,
                  "val_loss": current, "excluded_train_clips": excluded}
        history.append(status)
        print(json.dumps({"stage": 2, **status}), flush=True)
        if current < best:
            best, stale = current, 0
            _save_weights(run / "best_model.pth", model.state_dict())
        else:
            stale += 1
            if stale >= cfg.stage2_patience:
                break
    _write_rows(history, run / "history.csv")
    return run / "best_model.pth"


def evaluate(cfg, rows, split, checkpoint, *, out=None, partition="test", model_cfg=None, video_transform=None):
    """Explicit evaluation only: test data is never passed to training/selection."""
    if partition not in ("val", "test"):
        raise ValueError("Evaluation partition must be val or test")
    model = _model(model_cfg or cfg, pretrained_encoder=None) if not cfg.use_ssl else None
    if model is None:
        # Checkpoint contains the full model; no external encoder is needed here.
        from dataclasses import replace
        model = _model(replace(model_cfg or cfg, use_ssl=False))
    device = device_auto()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    loader = _loader(select_rows(rows, split[partition]), 2, cfg.stage2_batch_size,
                     num_workers=cfg.num_workers)
    clip_records = []
    with torch.no_grad():
        for video, truth, subjects, sessions, fps in loader:
            if video_transform is not None:
                video = video_transform(video)
            pred = model(video.to(device))[-1].cpu().numpy()
            truth = truth.numpy()
            for k in range(len(pred)):
                frequency = float(fps[k])
                clip_records.append({"subject": subjects[k], "session": sessions[k],
                    "pred_hr": estimate_hr(pred[k], frequency, cfg.hr_low_hz, cfg.hr_high_hz),
                    "gt_hr": estimate_hr(truth[k], frequency, cfg.hr_low_hz, cfg.hr_high_hz)})
    sessions_data = session_estimates(clip_records)
    result = metrics(sessions_data)
    result.update({"dataset": cfg.dataset, "partition": partition,
                   "n_processed_clips": len(clip_records)})
    if out:
        run = Path(out)
        run.mkdir(parents=True, exist_ok=True)
        (run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        _write_rows(sessions_data, run / "session_estimates.csv")
        from .evaluation.plots import plot_hr_figures
        plot_hr_figures(sessions_data, run / "figures")
    return result, sessions_data


def _write_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
