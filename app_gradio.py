"""
app_gradio.py
─────────────
Gradio web app for YOLO26 fabric defect detection.
Great for quick sharing/demos — can be deployed to Hugging Face Spaces.

Usage
-----
pip install gradio
python app_gradio.py
python app_gradio.py --model yolo26n.onnx --share   # public share link
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio not installed. Run: pip install gradio")

from inference_core import FabricDetector

# ─── Parse CLI args ───────────────────────────────────────────────────────────
p = argparse.ArgumentParser(add_help=False)
p.add_argument("--model",   default="yolo26n.onnx")
p.add_argument("--conf",    type=float, default=0.25)
p.add_argument("--iou",     type=float, default=0.45)
p.add_argument("--imgsz",   type=int,   default=640)
p.add_argument("--port",    type=int,   default=7860)
p.add_argument("--share",   action="store_true")
args, _ = p.parse_known_args()

# ─── Load model ───────────────────────────────────────────────────────────────
detector = FabricDetector(
    args.model,
    conf_threshold=args.conf,
    iou_threshold=args.iou,
    input_size=args.imgsz,
    class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
)
detector.warmup(3)


# ─── Inference function ───────────────────────────────────────────────────────

def detect(image: np.ndarray, conf: float, iou: float) -> tuple:
    """
    image: RGB numpy array (from Gradio)
    Returns: (annotated_image_rgb, json_results_str)
    """
    if image is None:
        return None, "No image provided."

    # Gradio passes RGB; convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detector.conf_threshold = conf
    detector.iou_threshold  = iou
    result, annotated = detector.predict(img_bgr, return_annotated=True)

    # Back to RGB for Gradio
    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Build summary text
    lines = [
        f"⏱  Inference: {result.inference_ms:.1f} ms  |  Total: {result.total_ms:.1f} ms",
        f"🔍  Defects found: {result.count}",
    ]
    if result.count:
        lines.append("")
        lines.append("Detections:")
        for d in sorted(result.detections, key=lambda x: -x.confidence):
            lines.append(
                f"  • {d.class_name:18s}  conf={d.confidence:.3f}  "
                f"bbox=[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]"
            )
        if result.class_counts():
            lines.append("")
            lines.append("Class breakdown: " + "  ".join(
                f"{k}: {v}" for k, v in sorted(result.class_counts().items())
            ))
    else:
        lines.append("✅  No defects detected.")

    return out_rgb, "\n".join(lines)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

css = """
#title { text-align: center; font-size: 2rem; font-weight: 700; }
#subtitle { text-align: center; color: #888; margin-bottom: 1rem; }
"""

with gr.Blocks(title="Fabric Defect Detector", theme=gr.themes.Soft(), css=css) as demo:
    gr.HTML('<p id="title">🧵 Fabric Defect Detector</p>')
    gr.HTML(
        '<p id="subtitle">YOLO26 · ONNX Runtime · Real-time defect localisation</p>'
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                sources=["upload", "clipboard"],
            )
            with gr.Accordion("⚙️ Settings", open=True):
                conf_slider = gr.Slider(
                    0.05, 0.95, value=args.conf, step=0.05,
                    label="Confidence threshold"
                )
                iou_slider = gr.Slider(
                    0.10, 0.95, value=args.iou, step=0.05,
                    label="IoU threshold (NMS)"
                )
            run_btn = gr.Button("🔍 Detect Defects", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Detected Defects", type="numpy")
            output_text  = gr.Textbox(
                label="Detection Results",
                lines=12,
                max_lines=20,
            )

    # ── Model info strip ──────────────────────────────────────────────────
    with gr.Accordion("ℹ️ Model Info", open=False):
        gr.Markdown(
            f"**Model:** `{args.model}`  |  "
            f"**Provider:** {detector.providers[0]}  |  "
            f"**Input size:** {args.imgsz}×{args.imgsz}  |  "
            f"**Classes:** {', '.join(detector.class_names)}"
        )

    # ── Example images (if any .jpg in current dir) ───────────────────────
    example_imgs = sorted(Path(".").glob("*.jpg"))[:4]
    if example_imgs:
        gr.Examples(
            examples=[[str(p)] for p in example_imgs],
            inputs=[input_image],
            label="Example Images",
        )

    # ── Event bindings ────────────────────────────────────────────────────
    run_btn.click(
        fn=detect,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, output_text],
    )
    input_image.change(
        fn=detect,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, output_text],
    )


if __name__ == "__main__":
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
