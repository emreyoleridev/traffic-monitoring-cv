"""Streamlit demo: upload a road image or a short clip, get vehicles, counts and congestion level.

    streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
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


MAX_FRAMES = 1800  # ~1 min at 30 fps; keeps uploads from running forever on a shared CPU


@st.cache_resource
def load(path: str) -> Detector:
    return Detector(path)


def to_mp4(frames: list[np.ndarray], fps: float) -> bytes:
    """H.264 encode RGB frames so the browser's <video> can play them (OpenCV's mp4v can't)."""
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        writer = imageio_ffmpeg.write_frames(f.name, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                             macro_block_size=1)
        writer.send(None)
        for fr in frames:
            writer.send(np.ascontiguousarray(fr))
        writer.close()
        return Path(f.name).read_bytes()


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
    if path and st.button("▶ Play & count", type="primary"):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES, MAX_FRAMES)
        counter, rows, frames_out = LineCounter(line), [], []
        left, right = st.columns([3, 1])
        view = left.empty()
        prog = left.progress(0.0)
        live = right.empty()
        det.reset_tracker()
        t_next = time.perf_counter()
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            tr = det.track(frame, conf=conf, iou=iou)
            counter.update(tr, frame.shape[0])
            out = draw(frame, tr[:, :4], [f"#{int(t)} {CLASSES[int(c)]}" for t, c in tr[:, [4, 6]]], tr[:, 6])
            out = cv2.cvtColor(counter.draw(out, in_view=len(tr)), cv2.COLOR_BGR2RGB)
            frames_out.append(out)
            # play at the clip's own speed: wait until this frame's slot (never faster than real time)
            t_next += 1 / fps
            time.sleep(max(0.0, t_next - time.perf_counter()))
            view.image(out, width="stretch")
            with live.container():
                st.metric("Crossed the line", counter.total())
                for c, n in counter.per_class().items():
                    st.metric(c, n)
                st.metric("In view now", len(tr))
                st.caption(f"t = {i / fps:.1f} s / {n_frames / fps:.1f} s")
            rows.append(dict(second=round(i / fps, 2), in_view=len(tr), crossed=counter.total()))
            prog.progress((i + 1) / n_frames)
        cap.release()
        st.session_state.video_result = dict(
            src=path, mp4=to_mp4(frames_out, fps), total=counter.total(), rows=rows,
            table=pd.DataFrame({d: dict(c) for d, c in counter.counts.items()}).reindex(CLASSES).fillna(0).astype(int))

    res = st.session_state.get("video_result")
    if res and res["src"] == path:
        st.success(f"{res['total']} vehicles crossed the line")
        st.subheader("Replay with counts")
        st.video(res["mp4"])
        st.download_button("Download annotated video", res["mp4"], "traffic_counted.mp4", "video/mp4")
        c1, c2 = st.columns([1, 2])
        c1.dataframe(res["table"])
        c2.line_chart(pd.DataFrame(res["rows"]).set_index("second"))

with tab_metrics:
    m = ROOT / "results" / "metrics.csv"
    if m.exists():
        st.dataframe(pd.read_csv(m).set_index("model").T, width="stretch")
    for p in ["results/val/finetuned/confusion_matrix_normalized.png", "results/eda/eda.png"]:
        if (ROOT / p).exists():
            st.image(str(ROOT / p), caption=p)
