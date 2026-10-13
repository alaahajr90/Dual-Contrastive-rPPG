"""Dlib 68-landmark facial ROI with forehead and cheeks only.

Landmark polygon boundaries are an explicit implementation choice because the
manuscript does not publish numeric ROI coordinates. No identity labels, full
frames, or facial landmarks are stored in the exported clip dataset.
"""
from __future__ import annotations
import cv2
import numpy as np


class FacialROI:
    def __init__(self, predictor_path: str, size: int = 64):
        try:
            import dlib
        except ImportError as exc:
            raise ImportError("Install the optional dlib package to extract 68-point ROIs") from exc
        from pathlib import Path
        if not Path(predictor_path).is_file():
            raise FileNotFoundError(f"Dlib landmark weights missing: {predictor_path}")
        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.size = size
        self.previous_rect = None

    def __call__(self, rgb: np.ndarray):
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected an RGB frame")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 0)
        if faces:
            rect = max(faces, key=lambda r: r.width() * r.height())
            self.previous_rect = rect
        elif self.previous_rect is not None:
            rect = self.previous_rect
        else:
            raise RuntimeError("Face not detected and no prior bounding box available")
        shape = self.predictor(gray, rect)
        p = np.asarray([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        h, w = rgb.shape[:2]
        brow = p[17:27]
        brow_height = max(1, int(np.linalg.norm(p[27] - p[8]) * 0.32))
        forehead = np.concatenate([brow[:5], (brow[:5] - [0, brow_height])[::-1]], axis=0)
        right_cheek = p[[0, 1, 2, 3, 31, 36, 39, 27]]
        left_cheek = p[[16, 15, 14, 13, 35, 45, 42, 27]]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in (forehead, right_cheek, left_cheek):
            cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
        x1, y1 = np.minimum(p.min(0), [w - 1, h - 1]).astype(int)
        x2, y2 = np.maximum(p.max(0), [0, 0]).astype(int)
        x1, y1 = max(0, x1), max(0, y1 - brow_height)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid facial crop")
        region = rgb[y1:y2 + 1, x1:x2 + 1]
        m = mask[y1:y2 + 1, x1:x2 + 1]
        # Preserve full skin ROI pixel colors, suppress eyes/nose/mouth.
        region = cv2.bitwise_and(region, region, mask=m)
        return cv2.resize(region, (self.size, self.size), interpolation=cv2.INTER_AREA)
