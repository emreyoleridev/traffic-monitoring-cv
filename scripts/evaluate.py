"""Evaluate the fine-tuned model vs. the COCO-pretrained baseline on the held-out time blocks,
and build short demo clips (consecutive test frames) for the Streamlit video tab.

    python scripts/evaluate.py --weights models/y11n.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

from trafficmon import CLASSES  # noqa: E402
from trafficmon.analytics import frame_stats  # noqa: E402
from trafficmon.detect import Detector  # noqa: E402

DATA = ROOT / "data" / "yolo"
RES = ROOT / "results"


def coco_yaml() -> Path:
    """Same test set, labels written with COCO ids so the 80-class model can be scored directly."""
    out = ROOT / "data" / "yolo_coco_ids"
    coco_id = {0: 2, 1: 5, 2: 7}
    for sub in ("images", "labels"):
        (out / "test" / sub).mkdir(parents=True, exist_ok=True)
    for img in (DATA / "test" / "images").glob("*.jpg"):
        dst = out / "test" / "images" / img.name
        if not dst.exists():
            dst.symlink_to(img)
        rows = [l.split() for l in (DATA / "test" / "labels" / f"{img.stem}.txt").read_text().splitlines() if l]
        (out / "test" / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(" ".join([str(coco_id[int(r[0])])] + r[1:]) for r in rows))
    names = YOLO(str(ROOT / "models" / "yolo11n.pt")).names
    y = out / "data.yaml"
    y.write_text(yaml.safe_dump(dict(path=str(out), train="test/images", val="test/images", names=names)))
    return y


def score(weights: str, data: Path, classes: list[int] | None, tag: str) -> dict:
    m = YOLO(weights).val(data=str(data), split="val", imgsz=640, batch=16, conf=0.001, iou=0.6,
                         classes=classes, plots=tag == "finetuned", project=str(RES / "val"), name=tag,
                         exist_ok=True, verbose=False)
    ids = classes or list(range(len(CLASSES)))
    per_cls = {CLASSES[i]: round(float(m.box.maps[c]), 4) for i, c in enumerate(ids)}
    return dict(model=tag, mAP50=round(m.box.map50, 4), mAP50_95=round(m.box.map, 4),
                precision=round(m.box.mp, 4), recall=round(m.box.mr, 4),
                ms_per_img=round(sum(m.speed.values()), 2), **{f"AP50-95 {k}": v for k, v in per_cls.items()})


def count_error(weights: str) -> dict:
    """How well does 'vehicles in frame' (the monitoring signal) match ground truth?"""
    det = Detector(weights)
    rows = []
    for img in sorted((DATA / "test" / "images").glob("*.jpg")):
        gt = len([l for l in (DATA / "test" / "labels" / f"{img.stem}.txt").read_text().splitlines() if l])
        d, _ = det(cv2.imread(str(img)), conf=0.3)
        rows.append((gt, len(d)))
    a = np.array(rows)
    return dict(count_MAE=round(float(np.abs(a[:, 0] - a[:, 1]).mean()), 2),
                count_bias=round(float((a[:, 1] - a[:, 0]).mean()), 2),
                level_acc=round(float(np.mean([frame_stats(np.zeros((g, 6)), (1, 1))["level"]
                                               == frame_stats(np.zeros((p, 6)), (1, 1))["level"]
                                               for g, p in a])), 3))


def demo_clips(n: int = 60) -> None:
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    imgs = sorted((DATA / "test" / "images").glob("*.jpg"))
    by_video: dict[str, list[Path]] = {}
    for p in imgs:
        by_video.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    for video, frames in by_video.items():
        w = cv2.VideoWriter(str(out / f"{video}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 6, (640, 640))
        for p in frames[:n]:
            w.write(cv2.imread(str(p)))
        w.release()
        cv2.imwrite(str(out / f"{video}.jpg"), cv2.imread(str(frames[len(frames) // 2])))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "models" / "y11n.pt"))
    a = p.parse_args()

    test_yaml = RES / "test.yaml"
    test_yaml.write_text(yaml.safe_dump(dict(path=str(DATA), train="test/images", val="test/images",
                                             names=dict(enumerate(CLASSES)))))
    base = str(ROOT / "models" / "yolo11n.pt")
    rows = [score(base, coco_yaml(), [2, 5, 7], "coco_pretrained") | count_error(base),
            score(a.weights, test_yaml, None, "finetuned") | count_error(a.weights)]
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(RES / "metrics.csv")
    (RES / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(df.T)
    demo_clips()
