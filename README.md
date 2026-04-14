# 🧵 Fabric Defect Detector — YOLO26 ONNX Inference Toolkit

A complete, production-ready toolkit for running your `yolo26n.onnx` fabric defect detection
model in **six different modes**: CLI, Streamlit app, Gradio app, REST API, video inference,
real-time webcam, and tile-based inference for large images.

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `inference_core.py` | Shared ONNX inference engine — imported by all scripts |
| `cli_inference.py` | Command-line tool (image / benchmark) |
| `video_inference.py` | Video file inference with annotated output |
| `webcam_inference.py` | Real-time webcam / RTSP stream inference |
| `batch_inference.py` | Batch folder processing + CSV report |
| `app_streamlit.py` | Interactive Streamlit web app |
| `app_gradio.py` | Gradio web app (Hugging Face Spaces-ready) |
| `api_server.py` | FastAPI REST API server |
| `tile_inference.py` | Sliding-window tile inference for large images |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Setup

```bash
# 1. Clone / download this folder
cd fabric_detector

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your model
cp /path/to/yolo26n.onnx .
```

> **GPU acceleration**: replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`
> and make sure CUDA 11.8+ / cuDNN 8+ is installed.

---

## 🚀 Usage

### 1 — `inference_core.py` (Python API)

The core engine you can import anywhere:

```python
from inference_core import FabricDetector

detector = FabricDetector("yolo26n.onnx", conf_threshold=0.3)
result = detector.predict("fabric.jpg")

for det in result.detections:
    print(det.class_name, det.confidence, det.xyxy)

# Return annotated image as BGR numpy array
result, annotated_bgr = detector.predict("fabric.jpg", return_annotated=True)

# Benchmark
bm = detector.benchmark(100)
print(f"{bm['fps']} FPS  |  {bm['mean_ms']} ms avg")
```

**Override class names** (if your model uses different labels):
```python
detector = FabricDetector(
    "yolo26n.onnx",
    class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
)
```

---

### 2 — `cli_inference.py` (Command Line)

```bash
# Single image
python cli_inference.py image --source fabric.jpg

# Whole folder
python cli_inference.py image --source ./images/ --save --out-dir ./results/

# Custom threshold + show window
python cli_inference.py image --source fabric.jpg --conf 0.35 --show

# JSON output (for piping to other tools)
python cli_inference.py image --source fabric.jpg --json | jq .

# Speed benchmark
python cli_inference.py benchmark --runs 200

# Override class names
python cli_inference.py image --source img.jpg --classes hole stain tear oil
```

---

### 3 — `video_inference.py` (Video File)

```bash
# Basic: reads input.mp4, writes input_pred.mp4
python video_inference.py --source input.mp4

# Custom output + live preview
python video_inference.py --source input.mp4 --out annotated.mp4 --show

# Process every 3rd frame (faster)
python video_inference.py --source input.mp4 --skip 3

# Save per-frame detection log
python video_inference.py --source input.mp4 --log frame_log.json

# Stop after 500 frames (for quick testing)
python video_inference.py --source input.mp4 --max-frames 500
```

---

### 4 — `webcam_inference.py` (Real-Time Camera)

```bash
# Default webcam
python webcam_inference.py

# Second camera
python webcam_inference.py --source 1

# RTSP stream
python webcam_inference.py --source rtsp://192.168.1.100:554/stream

# Record output
python webcam_inference.py --record session.mp4

# Mirror + high confidence
python webcam_inference.py --flip --conf 0.45
```

**Keyboard shortcuts while running:**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save snapshot |
| `P` | Pause / resume |
| `+` / `-` | Increase / decrease confidence threshold |
| `F` | Toggle fullscreen |
| `H` | Toggle HUD |

---

### 5 — `batch_inference.py` (Folder Processing)

```bash
# Process all images in a folder
python batch_inference.py --source ./fabric_samples/ --save

# Generate CSV report only (no saved images)
python batch_inference.py --source ./images/ --no-save --csv report.csv

# Per-detection detail CSV
python batch_inference.py --source ./images/ --detail-csv detections.csv

# Parallel processing (4 threads)
python batch_inference.py --source ./images/ --workers 4

# Recursive (search sub-folders)
python batch_inference.py --source ./dataset/ --recursive
```

Output CSV columns: `filename, width, height, num_defects, has_defect, class_counts_json, inference_ms`

---

### 6 — `app_streamlit.py` (Streamlit Web App)

```bash
pip install streamlit
streamlit run app_streamlit.py

# With options
streamlit run app_streamlit.py -- --model yolo26n.onnx --conf 0.3
```

Open `http://localhost:8501` in your browser.

Features:
- Drag & drop image upload
- Side-by-side original vs annotated view
- Detection table with colour-coded class labels
- Class distribution bar chart
- Download annotated image
- Webcam capture tab
- Speed benchmark tab

---

### 7 — `app_gradio.py` (Gradio Web App)

```bash
pip install gradio
python app_gradio.py

# Create a public share link (Hugging Face tunnel)
python app_gradio.py --share
```

Open `http://localhost:7860`. Shareable link printed to terminal when `--share` is used.

---

### 8 — `api_server.py` (FastAPI REST API)

```bash
pip install fastapi uvicorn python-multipart
python api_server.py --model yolo26n.onnx --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Model info |
| `GET` | `/health` | Health check |
| `GET` | `/classes` | List class names |
| `GET` | `/benchmark` | Speed benchmark |
| `POST` | `/predict` | Upload image → JSON detections |
| `POST` | `/predict/image` | Upload image → annotated JPEG/PNG |
| `POST` | `/predict/base64` | Upload image → JSON + base64 annotated image |
| `POST` | `/batch` | Upload multiple images → JSON |

**Example curl:**
```bash
# JSON detections
curl -X POST http://localhost:8000/predict \
  -F "file=@fabric.jpg" \
  -F "conf=0.3"

# Annotated image
curl -X POST http://localhost:8000/predict/image \
  -F "file=@fabric.jpg" \
  -o annotated.jpg

# Benchmark
curl http://localhost:8000/benchmark?runs=100
```

**Python requests example:**
```python
import requests

url  = "http://localhost:8000/predict"
resp = requests.post(url, files={"file": open("fabric.jpg", "rb")},
                     data={"conf": 0.3})
data = resp.json()
print(f"Found {data['count']} defects")
for det in data["detections"]:
    print(det["class_name"], det["confidence"])
```

---

### 9 — `tile_inference.py` (Large Image / Sliding Window)

For high-resolution scanner or microscopy images where defects are small:

```bash
# Basic tile inference
python tile_inference.py --source large_scan.jpg

# Smaller tiles with more overlap
python tile_inference.py --source scan.jpg --tile-size 800 --overlap 0.3

# Show tile grid overlay + save output
python tile_inference.py --source scan.jpg --show --save --show-grid

# Custom merge IoU
python tile_inference.py --source scan.jpg --merge-iou 0.4
```

---

## 🔧 Class Name Configuration

By default the scripts use these class names (matching the Roboflow dataset):

```
broken_yarn, crack, cut, hole, knot, oil_stain, stain, thread_error
```

Override them with `--classes` on the CLI, or pass `class_names=[...]` in Python:

```bash
python cli_inference.py image --source img.jpg \
  --classes defect_a defect_b defect_c
```

Or edit `DEFAULT_CLASSES` in `inference_core.py`.

---

## 🏎️ Performance Tips

| Tip | Speedup |
|-----|---------|
| Install `onnxruntime-gpu` | 5–20× on GPU |
| Use `--imgsz 640` instead of 800 | ~2× faster, slightly lower accuracy |
| Use `--skip 2` in video mode | 2× fewer inference calls |
| Use `--workers 4` in batch mode | Near-linear speedup (CPU, up to 4 cores) |
| Set `batch=8` in API batch endpoint | Amortises Python overhead |

---

## 📐 Output Format Reference

All scripts use `Detection` dataclass from `inference_core.py`:

```python
@dataclass
class Detection:
    x1, y1, x2, y2: float   # pixel coordinates (top-left, bottom-right)
    confidence: float         # 0.0 – 1.0
    class_id: int
    class_name: str
    # Properties: .xyxy, .width, .height, .area, .center, .to_dict()
```
