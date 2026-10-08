# Traffic Monitoring — Computer Vision

Vehicle detection (car / bus / truck), counting and congestion analytics on road-camera images,
with a Streamlit app to try it on your own images and clips.

- **Model:** YOLO11n (PyTorch / Ultralytics), COCO-pretrained, fine-tuned on the
  [RF100 vehicles](https://huggingface.co/datasets/LibreYOLO/vehicles-q0x2v) highway-CCTV set (CC BY 4.0).
- **Data prep:** 12 inconsistent raw labels folded into 3 classes; re-split by contiguous time blocks per video
  because the original random split leaks (79% of test frames have a train neighbour within ±2 frames).
- **Analytics:** per-class counts, heavy-vehicle share, road occupancy, congestion level,
  ByteTrack line-crossing counter for video.

**Live demo:** [https://emreyoleridev-traffic-monitoring-cv-app-fldhll.streamlit.app/](https://emreyoleridev-traffic-monitoring-cv-app-fldhll.streamlit.app/)

## Results (held-out time blocks, 611 images)

| Metric | COCO YOLO11n (baseline) | Fine-tuned YOLO11n |
| --- | --- | --- |
| mAP50 | 0.349 | **0.479** |
| mAP50-95 | 0.220 | **0.288** |
| AP50-95 truck | 0.202 | **0.412** |
| AP50-95 bus | **0.098** | 0.046 |
| Count MAE (vehicles/frame) | **4.59** | 9.20 |

The fine-tuned model was trained for a single epoch (~5 min CPU budget), so bus and counting are not yet reliable.
Longer GPU training (40–80 epochs at 640 px) is the obvious next step.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
streamlit run app.py
```

Reproduce training and evaluation:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('LibreYOLO/vehicles-q0x2v', repo_type='dataset', local_dir='data/raw')"
python scripts/prepare_dataset.py          # YOLO dataset + EDA in results/eda/
python scripts/train.py --model models/yolo11n.pt --time 0.07 --imgsz 480 --device cpu --name y11n
python scripts/evaluate.py                 # results/metrics.csv + demo clips in samples/
```
