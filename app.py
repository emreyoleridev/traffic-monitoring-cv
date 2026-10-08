"""Streamlit demo: upload a road image or a short clip, get vehicles, counts and congestion level.

    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from trafficmon import CLASSES
from trafficmon.analytics import LineCounter, frame_stats
from trafficmon.detect import Detector, draw, draw_dets

ROOT = Path(__file__).resolve().parent
MODELS = {"Fine-tuned YOLO11n (car/bus/truck)": ROOT / "models" / "y11n.pt",
          "COCO-pretrained YOLO11n (baseline)": ROOT / "models" / "yolo11n.pt"}
SAMPLES = ROOT / "samples"

st.set_page_config(page_title="Traffic Monitoring", page_icon="🚦", layout="wide")


@st.cache_resource
def load(path: str) -> Detector:
    return Detector(path)


with st.sidebar:
    st.header("Settings")
    avail = {k: v for k, v in MODELS.items() if v.exists()}
    model_name = st.selectbox("Model", list(avail))
    conf = st.slider("Confidence", 0.05, 0.9, 0.3, 0.05)
    iou = st.slider("NMS IoU", 0.3, 0.9, 0.5, 0.05)
    st.caption("Classes: " + ", ".join(CLASSES))

det = load(str(avail[model_name]))
st.title("🚦 Traffic Monitoring")
tab_img, tab_vid, tab_metrics = st.tabs(["Image", "Video / counting", "Model metrics"])

with tab_img:
    src = st.radio("Source", ["Sample", "Upload"], horizontal=True)
    img = None
    if src == "Upload":
        f = st.file_uploader("Road image", type=["jpg", "jpeg", "png"])
        if f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        samples = sorted(SAMPLES.glob("*.jpg"))
        if samples:
            pick = st.selectbox("Sample frame", samples, format_func=lambda p: p.stem)
            img = cv2.imread(str(pick))
    if img is not None:
        dets, ms = det(img, conf=conf, iou=iou)
        s = frame_stats(dets, img.shape[:2])
        c1, c2 = st.columns([3, 2])
        c1.image(cv2.cvtColor(draw_dets(img, dets), cv2.COLOR_BGR2RGB), width="stretch")
        with c2:
            st.metric("Congestion", s["level"])
            a, b = st.columns(2)
            a.metric("Vehicles", s["vehicles"])
            b.metric("Latency", f"{ms:.0f} ms")
            a.metric("Heavy-vehicle share", f"{s['heavy_ratio']:.0%}")
            b.metric("Road occupancy", f"{s['occupancy']:.0%}")
            st.bar_chart(pd.Series({c: s[c] for c in CLASSES}, name="count"))
            st.dataframe(pd.DataFrame(dets, columns=["x1", "y1", "x2", "y2", "conf", "cls"])
                         .assign(cls=lambda d: d.cls.map(lambda c: CLASSES[int(c)])).round(2),
                         width="stretch", height=200)

with tab_vid:
    st.write("Vehicles are tracked with ByteTrack and counted when they cross the yellow line.")
    vsrc = st.radio("Clip", ["Sample", "Upload"], horizontal=True)
    path = None
    if vsrc == "Upload":
        f = st.file_uploader("Video", type=["mp4", "mov", "avi"])
        if f:
            tmp = tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix, delete=False)
            tmp.write(f.read())
            path = tmp.name
    else:
        clips = sorted(SAMPLES.glob("*.mp4"))
        if clips:
            path = str(st.selectbox("Sample clip", clips, format_func=lambda p: p.stem))
    line = st.slider("Counting line (fraction of height)", 0.2, 0.9, 0.6, 0.05)
    max_frames = st.slider("Max frames", 10, 300, 60, 10)
    if path and st.button("Run", type="primary"):
        cap = cv2.VideoCapture(path)
        counter, view, prog, rows = LineCounter(line), st.empty(), st.progress(0.0), []
        det.reset_tracker()
        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            tr = det.track(frame, conf=conf, iou=iou)
            counter.update(tr, frame.shape[0])
            out = draw(frame, tr[:, :4], [f"#{int(t)} {CLASSES[int(c)]}" for t, c in tr[:, [4, 6]]], tr[:, 6])
            view.image(cv2.cvtColor(counter.draw(out), cv2.COLOR_BGR2RGB), width="stretch")
            rows.append(dict(frame=i, in_view=len(tr), crossed=counter.total()))
            prog.progress((i + 1) / max_frames)
        cap.release()
        st.success(f"{counter.total()} vehicles crossed the line")
        st.dataframe(pd.DataFrame({d: dict(c) for d, c in counter.counts.items()}).reindex(CLASSES).fillna(0)
                     .astype(int))
        st.line_chart(pd.DataFrame(rows).set_index("frame"))

with tab_metrics:
    m = ROOT / "results" / "metrics.csv"
    if m.exists():
        st.dataframe(pd.read_csv(m).set_index("model").T, width="stretch")
    for p in ["results/val/finetuned/confusion_matrix_normalized.png", "results/eda/eda.png"]:
        if (ROOT / p).exists():
            st.image(str(ROOT / p), caption=p)
