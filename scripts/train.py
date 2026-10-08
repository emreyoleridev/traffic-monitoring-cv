"""Fine-tune a COCO-pretrained YOLO11 on the car/bus/truck dataset. Runs on CUDA (Colab), Apple MPS or CPU.

    python scripts/train.py --data data/yolo/data.yaml --model yolo11s.pt --epochs 40 --name y11s
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "yolo" / "data.yaml"))
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="y11s")
    p.add_argument("--device", default=None)
    p.add_argument("--time", type=float, default=None, help="hour budget; overrides epochs")
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device or pick_device(),
                project=str(ROOT / "runs"), name=a.name, exist_ok=True, patience=15, seed=0,
                time=a.time, cos_lr=True, close_mosaic=10, plots=True, workers=4)
    best = ROOT / "runs" / a.name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / f"{a.name}.pt")
    print("saved", ROOT / "models" / f"{a.name}.pt")
