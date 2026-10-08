"""Traffic metrics on top of detections: per-class counts, congestion level, line-crossing counter."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from . import CLASSES

# Congestion thresholds on vehicles per frame, picked from the dataset's quartiles (7 / 12 / 17 per frame).
LEVELS = [(7, "Low"), (17, "Moderate"), (10**9, "Heavy")]


def frame_stats(dets: np.ndarray, shape: tuple[int, int]) -> dict:
    """dets Nx6 [x1, y1, x2, y2, conf, cls]."""
    counts = Counter(CLASSES[int(c)] for c in dets[:, 5])
    n = len(dets)
    area = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])).sum() / (shape[0] * shape[1]) if n else 0.0
    heavy = counts["bus"] + counts["truck"]
    return {
        "vehicles": n,
        **{c: counts[c] for c in CLASSES},
        "heavy_ratio": heavy / n if n else 0.0,
        "occupancy": float(min(area, 1.0)),
        "level": next(name for t, name in LEVELS if n <= t),
    }


class LineCounter:
    """Counts tracked vehicles whose centre crosses a horizontal line at y = frac * height."""

    def __init__(self, frac: float = 0.6):
        self.frac = frac
        self.last_y: dict[int, float] = {}
        self.counted: set[int] = set()
        self.counts = {"down": Counter(), "up": Counter()}

    def update(self, tracks: np.ndarray, height: int) -> None:
        line = self.frac * height
        for x1, y1, x2, y2, tid, _, c in tracks:
            tid, cy = int(tid), (y1 + y2) / 2
            prev = self.last_y.get(tid)
            self.last_y[tid] = cy
            if prev is None or tid in self.counted:
                continue
            if prev < line <= cy or prev > line >= cy:
                self.counted.add(tid)
                self.counts["down" if cy > prev else "up"][CLASSES[int(c)]] += 1

    def total(self) -> int:
        return sum(sum(c.values()) for c in self.counts.values())

    def draw(self, img: np.ndarray) -> np.ndarray:
        y = int(self.frac * img.shape[0])
        cv2.line(img, (0, y), (img.shape[1], y), (0, 220, 255), 2)
        txt = f"down {sum(self.counts['down'].values())}  up {sum(self.counts['up'].values())}"
        cv2.putText(img, txt, (10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
        return img
