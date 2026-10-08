"""YOLO wrapper that always returns detections in our car/bus/truck taxonomy, plus drawing helpers."""
from __future__ import annotations

import time

import cv2
import numpy as np
from ultralytics import YOLO

from . import CLASSES, COCO_TO_OURS

# BGR, one per class in CLASSES
COLORS = [(214, 120, 42), (57, 123, 224), (106, 163, 122)]


class Detector:
    """`weights` is either our fine-tuned model or a COCO model (then car/bus/truck are remapped)."""

    def __init__(self, weights: str, device: str | None = None):
        self.model = YOLO(weights)
        self.device = device
        self.is_coco = len(self.model.names) == 80
        self.keep = list(COCO_TO_OURS) if self.is_coco else None

    def _to_ours(self, data: np.ndarray) -> np.ndarray:
        if self.is_coco and len(data):
            data[:, 5] = [COCO_TO_OURS[int(c)] for c in data[:, 5]]
        return data

    def __call__(self, img: np.ndarray, conf: float = 0.3, iou: float = 0.5, imgsz: int = 640
                 ) -> tuple[np.ndarray, float]:
        """Returns (detections Nx6 [x1, y1, x2, y2, conf, cls], latency ms)."""
        t0 = time.perf_counter()
        r = self.model.predict(img, conf=conf, iou=iou, imgsz=imgsz, classes=self.keep,
                               device=self.device, verbose=False)[0]
        dets = r.boxes.data.cpu().numpy() if len(r.boxes) else np.zeros((0, 6), np.float32)
        return self._to_ours(dets), (time.perf_counter() - t0) * 1000

    def track(self, img: np.ndarray, conf: float = 0.3, iou: float = 0.5, imgsz: int = 640) -> np.ndarray:
        """ByteTrack over successive calls. Returns Nx7 [x1, y1, x2, y2, track_id, conf, cls]."""
        r = self.model.track(img, conf=conf, iou=iou, imgsz=imgsz, classes=self.keep, device=self.device,
                             persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is None:
            return np.zeros((0, 7), np.float32)
        b = r.boxes
        out = np.column_stack([b.xyxy.cpu().numpy(), b.id.cpu().numpy(), b.conf.cpu().numpy(),
                               b.cls.cpu().numpy()])
        if self.is_coco:
            out[:, 6] = [COCO_TO_OURS[int(c)] for c in out[:, 6]]
        return out

    def reset_tracker(self) -> None:
        for p in getattr(self.model.predictor, "trackers", None) or []:
            p.reset()


def draw(img: np.ndarray, boxes: np.ndarray, labels: list[str], classes: np.ndarray,
         thickness: int = 2) -> np.ndarray:
    out = img.copy()
    fs = max(0.4, 0.5 * max(out.shape[:2]) / 1000)
    for (x1, y1, x2, y2), label, c in zip(boxes, labels, classes):
        col = COLORS[int(c)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, col, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y = max(p1[1], th + 4)
        cv2.rectangle(out, (p1[0], y - th - 4), (p1[0] + tw + 4, y), col, -1)
        cv2.putText(out, label, (p1[0] + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_dets(img: np.ndarray, dets: np.ndarray) -> np.ndarray:
    return draw(img, dets[:, :4], [f"{CLASSES[int(c)]} {s:.2f}" for s, c in dets[:, 4:6]], dets[:, 5])
