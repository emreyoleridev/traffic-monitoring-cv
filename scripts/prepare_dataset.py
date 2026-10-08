"""Build the YOLO dataset + EDA from the raw RF100 "vehicles" export.

The raw Roboflow split is random over frames of 5 CCTV videos, so near-identical consecutive
frames land in train *and* test. We re-split by contiguous time blocks inside each video
(70 / 15 / 15) with a gap of GAP frames between blocks, and fold 12 labels into car/bus/truck.

    python scripts/prepare_dataset.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trafficmon import CLASSES, RAW_TO_OURS  # noqa: E402

RAW, OUT, RES = ROOT / "data" / "raw", ROOT / "data" / "yolo", ROOT / "results" / "eda"
GAP = 10
FRAME_RE = re.compile(r"^(?P<video>.+)_mp4-(?P<frame>\d+)_jpg")


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_names = yaml.safe_load((RAW / "data.yaml").read_text())["names"]
    frames, boxes = [], []
    for img in sorted(RAW.glob("*/images/*.jpg")):
        m = FRAME_RE.match(img.name)
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        rows = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
        frames.append(dict(image=str(img), video=m["video"], frame=int(m["frame"]),
                           raw_split=img.parent.parent.name, n_boxes=len(rows)))
        for c, x, y, w, h in rows:
            boxes.append(dict(image=str(img), video=m["video"], raw_class=raw_names[int(c)],
                              cls=CLASSES[RAW_TO_OURS[raw_names[int(c)]]], x=x, y=y, w=w, h=h))
    return pd.DataFrame(frames), pd.DataFrame(boxes)


def time_block_split(frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in frames.sort_values("frame").groupby("video"):
        n = len(g)
        a, b = int(n * 0.70), int(n * 0.85)
        split = np.array(["train"] * a + ["val"] * (b - a) + ["test"] * (n - b), dtype=object)
        # drop frames right before a block boundary so neighbours don't straddle splits
        idx = np.arange(n)
        split[((idx >= a - GAP) & (idx < a)) | ((idx >= b - GAP) & (idx < b))] = "gap"
        parts.append(g.assign(split=split))
    return pd.concat(parts)


def write_yolo(frames: pd.DataFrame, boxes: pd.DataFrame) -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    by_img = {k: g for k, g in boxes.groupby("image")}
    for r in frames[frames.split != "gap"].itertuples():
        (OUT / r.split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / r.split / "labels").mkdir(parents=True, exist_ok=True)
        stem = f"{r.video}_{r.frame:05d}"
        shutil.copy(r.image, OUT / r.split / "images" / f"{stem}.jpg")
        g = by_img.get(r.image)
        lines = [] if g is None else [f"{CLASSES.index(b.cls)} {b.x:.6f} {b.y:.6f} {b.w:.6f} {b.h:.6f}"
                                      for b in g.itertuples()]
        (OUT / r.split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        dict(path=str(OUT), train="train/images", val="val/images", test="test/images",
             names=dict(enumerate(CLASSES))), sort_keys=False))


def eda(frames: pd.DataFrame, boxes: pd.DataFrame) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    boxes = boxes.merge(frames[["image", "split"]], on="image")
    boxes["area_px"] = boxes.w * boxes.h * 640 * 640

    split_tbl = frames.pivot_table(index="video", columns="split", values="image", aggfunc="count", fill_value=0)
    split_tbl.to_csv(RES / "split_by_video.csv")
    cls_tbl = boxes[boxes.split != "gap"].pivot_table(index="cls", columns="split", values="x",
                                                       aggfunc="count", fill_value=0).loc[CLASSES]
    cls_tbl.to_csv(RES / "class_counts.csv")
    boxes.groupby(["raw_class", "cls"]).size().rename("boxes").to_csv(RES / "raw_class_mapping.csv")
    per_img = frames.n_boxes.describe().round(2)
    per_img.to_csv(RES / "boxes_per_image.csv")
    print(split_tbl, cls_tbl, per_img, sep="\n\n")

    # near-duplicate check on the *original* random split: how many test frames have a
    # train frame from the same video within ±2 frames?
    tr = set(zip(frames[frames.raw_split == "train"].video, frames[frames.raw_split == "train"].frame))
    te = frames[frames.raw_split == "test"]
    leak = np.mean([any((v, f + d) in tr for d in (-2, -1, 1, 2)) for v, f in zip(te.video, te.frame)])
    (RES / "leakage.txt").write_text(f"raw roboflow test frames with a train neighbour within ±2 frames: {leak:.1%}\n")
    print(f"\nleakage in original split: {leak:.1%}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cls_tbl.plot.bar(ax=ax[0], rot=0, color=["#2a78d6", "#c3c2c0", "#e07b39", "#7aa36a"][:cls_tbl.shape[1]])
    ax[0].set(title="Boxes per class and split", ylabel="boxes", xlabel="")
    ax[0].set_yscale("log")
    ax[1].hist(frames.n_boxes, bins=40, color="#2a78d6")
    ax[1].set(title="Vehicles per image", xlabel="boxes", ylabel="images")
    for c, col in zip(CLASSES, ["#2a78d6", "#e07b39", "#7aa36a"]):
        ax[2].hist(np.sqrt(boxes[boxes.cls == c].area_px), bins=50, alpha=0.6, label=c, color=col)
    ax[2].set(title="Box size (sqrt area, px @640)", xlabel="px", ylabel="boxes")
    ax[2].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "eda.png", dpi=130)

    # where vehicles appear in the frame (useful to place counting lines)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist2d(boxes.x, boxes.y, bins=64, range=[[0, 1], [0, 1]], cmap="magma")
    ax.invert_yaxis()
    ax.set(title="Vehicle centre heatmap", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(RES / "centre_heatmap.png", dpi=130)


if __name__ == "__main__":
    frames, boxes = load_frames()
    frames = time_block_split(frames)
    write_yolo(frames, boxes)
    eda(frames, boxes)
    frames.to_csv(RES / "frames.csv", index=False)
    print("wrote", OUT / "data.yaml")
