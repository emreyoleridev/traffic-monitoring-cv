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

    def per_class(self) -> dict[str, int]:
        return {c: self.counts["down"][c] + self.counts["up"][c] for c in CLASSES}

    def draw(self, img: np.ndarray, in_view: int | None = None) -> np.ndarray:
        """Counting line + a running-count panel in the top-left corner."""
        y = int(self.frac * img.shape[0])
        cv2.line(img, (0, y), (img.shape[1], y), (0, 220, 255), 2)
        lines = [f"Crossed: {self.total()}  (down {sum(self.counts['down'].values())}"
                 f" / up {sum(self.counts['up'].values())})",
                 *(f"  {c}: {n}" for c, n in self.per_class().items())]
        if in_view is not None:
            lines.append(f"In view: {in_view}")
        h = 22 * len(lines) + 12
        panel = img[8:8 + h, 8:300]
        panel[:] = (panel * 0.35).astype(np.uint8)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (16, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 255) if i == 0 else (255, 255, 255), 1 if i else 2, cv2.LINE_AA)
        return img
