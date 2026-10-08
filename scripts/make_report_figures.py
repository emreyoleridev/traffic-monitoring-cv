"""Qualitative figure for the report: COCO baseline vs. fine-tuned YOLO11n on held-out sample frames.

    python scripts/make_report_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trafficmon.analytics import frame_stats  # noqa: E402
from trafficmon.detect import Detector, draw_dets  # noqa: E402

SAMPLES = ["pagi_16112021", "adit", "malam_04112021"]
OUT = ROOT / "results" / "figures_report"


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, img.shape[0] - 34), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    models = {"COCO": Detector(str(ROOT / "models" / "yolo11n.pt")),
              "Fine-tuned": Detector(str(ROOT / "models" / "y11n.pt"))}
    rows = []  # one row per model, one column per sample frame
    for name, det in models.items():
        cols = []
        for s in SAMPLES:
            img = cv2.imread(str(ROOT / "samples" / f"{s}.jpg"))
            d, _ = det(img, conf=0.3)
            st = frame_stats(d, img.shape[:2])
            cols.append(label(draw_dets(img, d), f"{name}: {st['vehicles']} vehicles, {st['level']}"))
            cols.append(np.full((img.shape[0], 8, 3), 255, np.uint8))
        rows.append(np.hstack(cols[:-1]))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.vstack([x for r in rows for x in (r, gap)][:-1])
    cv2.imwrite(str(OUT / "qualitative.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", OUT / "qualitative.jpg")
