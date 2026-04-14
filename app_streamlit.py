"""
app_streamlit.py
────────────────
Streamlit web app for YOLO26 fabric defect detection.

Usage
-----
pip install streamlit
streamlit run app_streamlit.py
streamlit run app_streamlit.py -- --model yolo26n.onnx --conf 0.3
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Parse optional CLI args before Streamlit takes over
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default="yolo26n.onnx")
_parser.add_argument("--conf",  type=float, default=0.25)
_parser.add_argument("--iou",   type=float, default=0.45)
_parser.add_argument("--imgsz", type=int,   default=640)
_cli, _ = _parser.parse_known_args()

from inference_core import FabricDetector, PALETTE

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fabric Defect Detector",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #f0f4f8 0%, #c8d6e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8899aa;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #1a2332;
        border: 1px solid #2a3a52;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #7eb3f5;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #778899;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .det-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: #1e2d40;
        border-left: 4px solid;
    }

    .det-class  { font-weight: 600; font-size: 0.9rem; }
    .det-conf   { font-size: 0.8rem; color: #aabbcc; }
    .det-coords { font-size: 0.75rem; color: #667788; font-family: monospace; }

    div[data-testid="stSidebar"] {
        background: #111b27;
        border-right: 1px solid #1e2d40;
    }

    .stSlider > div > div > div { background: #2a3a52 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO26 model…")
def load_detector(model_path: str, imgsz: int) -> FabricDetector:
    det = FabricDetector(model_path, input_size=imgsz, class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"])
    det.warmup(3)
    return det


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_path = st.text_input("Model path", value=_cli.model)
    imgsz = st.selectbox("Input size", [416, 512, 640, 768, 1024], index=2)

    st.markdown("---")
    conf = st.slider("Confidence threshold", 0.05, 0.95, _cli.conf, 0.05)
    iou  = st.slider("IoU threshold (NMS)", 0.10, 0.95, _cli.iou,  0.05)

    st.markdown("---")
    show_labels = st.toggle("Show labels",       value=True)
    show_conf   = st.toggle("Show confidence",   value=True)
    line_thick  = st.slider("Box line thickness", 1, 5, 2)

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Powered by [YOLO26](https://docs.ultralytics.com/models/yolo26/) "
        "via ONNX Runtime."
    )


# ─── Load model ───────────────────────────────────────────────────────────────
try:
    detector = load_detector(model_path, imgsz)
    detector.conf_threshold = conf
    detector.iou_threshold  = iou
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("Make sure `yolo26n.onnx` is in the same directory as this script.")
    st.stop()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧵 Fabric Defect Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">YOLO26 · ONNX Runtime · Real-time bounding box detection</p>',
    unsafe_allow_html=True,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_img, tab_cam, tab_bench = st.tabs(["📷 Image Upload", "🎥 Webcam", "⚡ Benchmark"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Image Upload
# ════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded = st.file_uploader(
        "Drop a fabric image here",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        # Decode
        pil_img = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Run inference
        with st.spinner("Running inference…"):
            result, annotated = detector.predict(img_bgr, return_annotated=True)

        # Display
        col_orig, col_pred = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)

        with col_pred:
            st.markdown("**Detected Defects**")
            pred_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(pred_rgb, use_container_width=True)

        # ── Metrics ───────────────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Defects found", result.count)
        m2.metric("Inference", f"{result.inference_ms:.1f} ms")
        m3.metric("Total time", f"{result.total_ms:.1f} ms")
        m4.metric("Image size", f"{pil_img.width}×{pil_img.height}")

        # ── Detection table ───────────────────────────────────────────────
        if result.count:
            st.markdown("#### Detections")
            for i, det in enumerate(sorted(result.detections,
                                            key=lambda d: -d.confidence)):
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    *PALETTE[det.class_id % len(PALETTE)][::-1]
                )
                st.markdown(
                    f'<div class="det-row" style="border-color:{hex_color}">'
                    f'<span class="det-class">{det.class_name}</span>'
                    f'<span class="det-conf">conf: {det.confidence:.3f}</span>'
                    f'<span class="det-coords">'
                    f'[{det.x1:.0f},{det.y1:.0f} → {det.x2:.0f},{det.y2:.0f}]  '
                    f'{det.width:.0f}×{det.height:.0f}px'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Class counts bar
            if len(result.class_counts()) > 1:
                import pandas as pd
                df = pd.DataFrame(
                    list(result.class_counts().items()),
                    columns=["Defect class", "Count"],
                ).set_index("Defect class")
                st.bar_chart(df)
        else:
            st.success("✅ No defects detected above the confidence threshold.")

        # ── Download button ────────────────────────────────────────────────
        pred_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pred_pil.save(buf, format="JPEG", quality=92)
        st.download_button(
            "⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name=f"{Path(uploaded.name).stem}_pred.jpg",
            mime="image/jpeg",
        )
    else:
        st.info("👆 Upload a fabric image to start detection.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Webcam / RTSP
# ════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.info(
        "ℹ️ Browser-based webcam access is limited in Streamlit. "
        "For real-time camera inference, use **`webcam_inference.py`** instead."
    )

    cam_source = st.text_input("Camera source (index or RTSP URL)", value="0")
    n_frames = st.slider("Capture N frames for demo", 1, 20, 5)

    if st.button("📸 Capture & Detect"):
        cap = cv2.VideoCapture(
            int(cam_source) if cam_source.isdigit() else cam_source
        )
        if not cap.isOpened():
            st.error(f"Cannot open camera source: {cam_source}")
        else:
            frames_shown = 0
            cols = st.columns(min(n_frames, 4))
            with st.spinner("Capturing…"):
                for _ in range(n_frames * 3):   # skip initial dark frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frames_shown >= n_frames:
                        break

                    result, annotated = detector.predict(frame, return_annotated=True)
                    col = cols[frames_shown % len(cols)]
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    col.image(rgb, caption=f"Frame — {result.count} defect(s)",
                               use_container_width=True)
                    frames_shown += 1
            cap.release()


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Benchmark
# ════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### ⚡ Inference Speed Benchmark")
    st.markdown("Runs the model on a blank 640×640 image N times and reports latency.")

    bm_runs = st.slider("Benchmark iterations", 10, 500, 100, 10)

    if st.button("▶ Run Benchmark"):
        progress = st.progress(0, text="Warming up…")
        detector.warmup(5)
        progress.progress(10, text="Running benchmark…")

        t0 = time.perf_counter()
        bm = detector.benchmark(bm_runs)
        wall = time.perf_counter() - t0
        progress.progress(100, text="Done!")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Mean latency",   f"{bm['mean_ms']} ms")
        b2.metric("Std dev",        f"{bm['std_ms']} ms")
        b3.metric("Min latency",    f"{bm['min_ms']} ms")
        b4.metric("Max latency",    f"{bm['max_ms']} ms")
        b5.metric("Throughput",     f"{bm['fps']} FPS")

        st.markdown("---")
        st.json(bm)

    # Model info
    st.markdown("### ℹ️ Model Info")
    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown(f"**Path:** `{model_path}`")
        st.markdown(f"**Input size:** {imgsz}×{imgsz}")
        st.markdown(f"**Provider:** {detector.providers[0]}")
    with info_cols[1]:
        st.markdown(f"**Classes ({len(detector.class_names)}):**")
        for i, cls in enumerate(detector.class_names):
            hex_c = "#{:02x}{:02x}{:02x}".format(*PALETTE[i % len(PALETTE)][::-1])
            st.markdown(
                f'<span style="color:{hex_c}">● </span>**{i}** {cls}',
                unsafe_allow_html=True,
            )
