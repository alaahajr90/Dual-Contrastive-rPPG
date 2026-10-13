"""Dataset-specific acquisition adapters; the output interface is always RGB frames + BVP.

Reference signals are interpolated to video-frame timestamps *before* clip generation.
Unverifiable assumptions (e.g. missing recording timestamps) are stated in metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import h5py
import numpy as np


@dataclass
class Recording:
    dataset: str
    subject: str
    session: str
    frames: np.ndarray  # [frames, H, W, 3] RGB, uint8
    video_times: np.ndarray  # seconds from the start of the recording
    bvp: np.ndarray  # 1D
    bvp_times: np.ndarray  # seconds from the start of the recording
    fps: float
    alignment_note: str


def _read_video(path: Path, nominal_fps: float):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 1:
        fps = nominal_fps
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No video frames: {path}")
    return np.stack(frames), np.arange(len(frames), dtype=float) / fps, fps


def _canonicalize_signal(arr):
    sig = np.asarray(arr, dtype=np.float64).reshape(-1)
    if sig.size < 128 or not np.isfinite(sig).all():
        raise ValueError("BVP must be finite and at least 128 samples long")
    return sig


def _read_pure_json(path: Path):
    payload = json.loads(path.read_text())
    packages = payload.get("FullPackage")
    if not isinstance(packages, list) or not packages:
        raise ValueError(f"PURE expects FullPackage list in {path}")
    values, times = [], []
    for package in packages:
        data = package.get("Value", {})
        waveform = data.get("waveform") if isinstance(data, dict) else None
        stamp = package.get("Timestamp", package.get("timestamp"))
        if waveform is None or stamp is None:
            continue
        try:
            values.append(float(waveform))
            times.append(float(stamp))
        except (TypeError, ValueError):
            continue
    if len(values) < 128:
        raise ValueError(f"PURE waveform or timestamp missing: {path}")
    times = np.asarray(times, dtype=float)
    # PURE timestamps commonly encode seconds; identify milliseconds by cadence.
    dt = float(np.median(np.diff(times)))
    if dt > 0.5:
        times = times / 1000.0
    times -= times[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"PURE non-monotonic timestamps: {path}")
    return _canonicalize_signal(values), times


def discover_pure(root: Path):
    """Expected: <root>/<session>/*.png and <root>/<session>.json or session/*.json."""
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        pngs = list(session_dir.glob("*.png"))
        if not pngs:
            continue
        candidate = [session_dir.with_suffix(".json"), session_dir / f"{session_dir.name}.json"]
        matches = [p for p in candidate if p.exists()]
        if len(matches) != 1:
            raise ValueError(f"PURE session {session_dir} needs exactly one corresponding JSON")
        session = session_dir.name
        yield session, session.split("-")[0], (session_dir, matches[0])


def load_pure(session, subject, paths):
    frames_dir, json_path = paths
    paths_png = sorted(frames_dir.glob("*.png"), key=lambda p: float(p.stem))
    frames = [cv2.imread(str(p)) for p in paths_png]
    if not frames or any(frame is None for frame in frames):
        raise OSError(f"Unreadable PURE PNG sequence: {frames_dir}")
    frames = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frames])
    timestamps = np.asarray([float(p.stem) for p in paths_png], dtype=float)
    dt = float(np.median(np.diff(timestamps)))
    if dt > 0.5:
        timestamps /= 1000.0
    timestamps -= timestamps[0]
    # Reading sample-rate annotations from the signal rather than assuming JSON index rate.
    signal, stimes = _read_pure_json(json_path)
    return Recording("pure", subject, session, frames, timestamps, signal, stimes,
                     float(1.0 / np.median(np.diff(timestamps))),
                     "frame filename timestamps and PURE FullPackage timestamps")


def discover_cohface(root: Path):
    """Typical layout: <root>/<subject>/<session>/data.avi + data.hdf5."""
    for avi in sorted(root.rglob("*.avi")):
        matches = [p for p in (avi.with_suffix(".hdf5"), avi.with_suffix(".h5")) if p.exists()]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous COHFACE HDF5 for {avi}")
        rel = avi.relative_to(root)
        if len(rel.parts) < 3:
            raise ValueError(f"COHFACE expected subject/session/data.avi: {avi}")
        session = "_".join(rel.parts[:-1])
        yield session, rel.parts[0], (avi, matches[0])


def load_cohface(session, subject, paths):
    avi, hdf5 = paths
    frames, ftimes, fps = _read_video(avi, 20.0)
    with h5py.File(hdf5, "r") as h:
        candidates = ["pulse", "bvp", "BVP"]
        keys = [name for name in candidates if name in h]
        if len(keys) != 1:
            raise ValueError(f"COHFACE HDF5 requires one pulse/BVP field, found {list(h)}")
        signal = _canonicalize_signal(h[keys[0]][()])
    stimes = np.arange(len(signal), dtype=float) / 256.0
    return Recording("cohface", subject, session, frames, ftimes, signal, stimes, fps,
                     "nominal video FPS and manuscript COHFACE reference sampling at 256 Hz")


def discover_ubfc(root: Path):
    """Expected: <root>/subjectN/vid.avi + ground_truth.txt."""
    for video in sorted(root.rglob("vid.avi")):
        truth = video.parent / "ground_truth.txt"
        if not truth.exists():
            raise FileNotFoundError(truth)
        session = video.parent.relative_to(root).as_posix().replace("/", "_")
        yield session, video.parent.name, (video, truth)


def load_ubfc(session, subject, paths):
    video, ground_truth = paths
    frames, ftimes, fps = _read_video(video, 30.0)
    lines = [x for x in ground_truth.read_text().splitlines() if x.strip()]
    if not lines:
        raise ValueError("UBFC ground_truth.txt is empty")
    # UBFC first row is waveform, second HR; third commonly holds timestamps.
    signal = _canonicalize_signal(np.fromstring(lines[0], sep=" "))
    stimes = np.arange(len(signal), dtype=float) / fps
    note = "uniform video-FPS waveform sampling assumed"
    if len(lines) >= 3:
        candidate = np.fromstring(lines[2], sep=" ")
        if len(candidate) == len(signal) and np.all(np.diff(candidate) > 0):
            stimes = candidate - candidate[0]
            note = "UBFC supplied waveform timestamps"
    return Recording("ubfc", subject, session, frames, ftimes, signal, stimes, fps, note)


REGISTRY = {"pure": (discover_pure, load_pure),
            "cohface": (discover_cohface, load_cohface),
            "ubfc": (discover_ubfc, load_ubfc)}
