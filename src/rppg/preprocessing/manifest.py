"""Build/check flat paired NumPy clip manifests; all splits use subject identities."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
import numpy as np
from .readers import REGISTRY
from .roi import FacialROI

FIELDS = ("dataset", "subject", "session", "clip_id", "video", "bvp", "fps", "start_frame")


def read_manifest(path):
    with open(path, newline="", encoding="utf8") as f:
        records = list(csv.DictReader(f))
    for row in records:
        if not all(row.get(key) for key in FIELDS):
            raise ValueError(f"Missing manifest field(s): {row}")
        if not Path(row["video"]).exists():
            raise FileNotFoundError(row["video"])
        if not Path(row["bvp"]).exists():
            raise FileNotFoundError(row["bvp"])
    if len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("Duplicate clip ID")
    return records


def _align_reference(recording):
    t = recording.video_times
    st = recording.bvp_times
    if len(t) != len(recording.frames):
        raise ValueError("Frame timestamp count mismatch")
    if not (np.all(np.diff(t) > 0) and np.all(np.diff(st) > 0)):
        raise ValueError("Non-monotonic acquisition timestamps")
    if t[0] < st[0] - 0.1 or t[-1] > st[-1] + 0.1:
        raise ValueError("BVP/video duration mismatch; do not extrapolate outside reference")
    return np.interp(t, st, recording.bvp).astype(np.float32)


def preprocess_dataset(config, *, limit_sessions=None):
    dataset = config.dataset
    discover, load = REGISTRY[dataset]
    root = Path(config.raw_root) / dataset
    if not root.is_dir():
        raise FileNotFoundError(f"Provide dataset raw directory: {root}")
    target = Path(config.processed_root) / dataset
    target.mkdir(parents=True, exist_ok=True)
    extract_roi = FacialROI(config.predictor_path, config.roi_size)
    rows, metadata = [], []
    for number, (session, subject, paths) in enumerate(discover(root)):
        if limit_sessions is not None and number >= limit_sessions:
            break
        recording = load(session, subject, paths)
        if abs(recording.fps - config.fps) > 2.0:
            raise ValueError(f"Unexpected FPS {recording.fps} for {session}; review acquisition")
        reference = _align_reference(recording)
        extract_roi.previous_rect = None
        rois = [extract_roi(frame) for frame in recording.frames]
        rois = np.stack(rois)
        safe_session = re.sub(r"[^A-Za-z0-9_-]", "_", session)
        for index, start in enumerate(range(0, len(rois) - config.clip_length + 1, config.stride)):
            name = f"{safe_session}_clip{index}"
            image_path, bvp_path = target / f"{name}.npy", target / f"{name}_bvp.npy"
            np.save(image_path, rois[start:start + config.clip_length])
            np.save(bvp_path, reference[start:start + config.clip_length])
            rows.append(dict(dataset=dataset, subject=str(subject), session=session, clip_id=name,
                             video=str(image_path.resolve()), bvp=str(bvp_path.resolve()),
                             fps=f"{recording.fps:.8f}", start_frame=str(start)))
        metadata.append(dict(subject=str(subject), session=session, alignment=recording.alignment_note,
                             fps=recording.fps, frames=len(recording.frames)))
    if not rows:
        raise RuntimeError("No clips were generated; verify raw layout and recording lengths")
    manifest_path = target / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (target / "acquisition_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest_path
