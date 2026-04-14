"""
api_server.py
─────────────
FastAPI REST API server for YOLO26 fabric defect detection.
Accepts image uploads and returns JSON detection results.

Usage
-----
pip install fastapi uvicorn python-multipart
python api_server.py

Endpoints
---------
GET  /              Health check + model info
GET  /health        Liveness probe
POST /predict       Single image detection (returns JSON)
POST /predict/image Single image detection (returns annotated image)
POST /batch         Multiple images detection
GET  /classes       List class names
GET  /benchmark     Run speed benchmark
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI not installed.\n"
        "Run: pip install fastapi uvicorn python-multipart"
    )

from inference_core import FabricDetector

# ─── Global detector instance ─────────────────────────────────────────────────
_detector: Optional[FabricDetector] = None

MODEL_PATH   = "yolo26n.onnx"
CONF_DEFAULT = 0.25
IOU_DEFAULT  = 0.45


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    _detector = FabricDetector(
        MODEL_PATH,
        conf_threshold=CONF_DEFAULT,
        iou_threshold=IOU_DEFAULT,
        class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
    )
    _detector.warmup(3)
    print(f"[API] Model ready: {MODEL_PATH}")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Fabric Defect Detection API",
    description="YOLO26 ONNX-based fabric defect detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ──────────────────────────────────────────────────────────

class DetectionItem(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str


class PredictResponse(BaseModel):
    status: str = "ok"
    count: int
    detections: List[DetectionItem]
    class_counts: dict
    image_shape: List[int]
    timing_ms: dict
    model: str


class ModelInfo(BaseModel):
    model: str
    classes: List[str]
    num_classes: int
    conf_threshold: float
    iou_threshold: float
    input_size: int
    providers: List[str]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def decode_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image file.")
    return img


def result_to_response(result, model_path: str) -> dict:
    return {
        "status": "ok",
        "count": result.count,
        "detections": [d.to_dict() for d in result.detections],
        "class_counts": result.class_counts(),
        "image_shape": list(result.image_shape),
        "timing_ms": {
            "preprocess": round(result.preprocess_ms, 2),
            "inference":  round(result.inference_ms, 2),
            "postprocess": round(result.postprocess_ms, 2),
            "total": round(result.total_ms, 2),
        },
        "model": Path(model_path).name,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_model=ModelInfo)
def root():
    """Model info and health check."""
    return {
        "model": MODEL_PATH,
        "classes": _detector.class_names,
        "num_classes": len(_detector.class_names),
        "conf_threshold": _detector.conf_threshold,
        "iou_threshold": _detector.iou_threshold,
        "input_size": _detector.input_size,
        "providers": _detector.providers,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH, "timestamp": time.time()}


@app.get("/classes")
def get_classes():
    return {"classes": _detector.class_names, "count": len(_detector.class_names)}


@app.post("/predict", response_model=PredictResponse)
async def predict_json(
    file: UploadFile = File(...),
    conf: float = Form(CONF_DEFAULT),
    iou:  float = Form(IOU_DEFAULT),
):
    """
    Detect defects in an uploaded image.
    Returns JSON with bounding boxes, class names, and confidence scores.

    - **file**: image file (JPEG, PNG, BMP, …)
    - **conf**: confidence threshold (0–1)
    - **iou**: IoU threshold for NMS
    """
    img = decode_upload(file)
    _detector.conf_threshold = conf
    _detector.iou_threshold  = iou
    result = _detector.predict(img)
    return result_to_response(result, MODEL_PATH)


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(CONF_DEFAULT),
    iou:  float = Form(IOU_DEFAULT),
    format: str = Form("jpeg"),
):
    """
    Detect defects and return the annotated image as JPEG/PNG bytes.

    - **format**: output format — `jpeg` or `png`
    """
    img = decode_upload(file)
    _detector.conf_threshold = conf
    _detector.iou_threshold  = iou
    result, annotated = _detector.predict(img, return_annotated=True)

    fmt  = "jpeg" if format.lower() in ("jpg", "jpeg") else "png"
    enc  = ".jpg" if fmt == "jpeg" else ".png"
    ok, buf = cv2.imencode(enc, annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    headers = {
        "X-Defect-Count":    str(result.count),
        "X-Inference-Ms":    f"{result.inference_ms:.1f}",
        "X-Class-Counts":    str(result.class_counts()),
    }
    return Response(
        content=buf.tobytes(),
        media_type=f"image/{fmt}",
        headers=headers,
    )


@app.post("/predict/base64")
async def predict_base64(
    file: UploadFile = File(...),
    conf: float = Form(CONF_DEFAULT),
    iou:  float = Form(IOU_DEFAULT),
):
    """
    Detect defects and return the annotated image as a base64-encoded JPEG
    alongside the JSON detections — useful for frontend apps.
    """
    img = decode_upload(file)
    _detector.conf_threshold = conf
    _detector.iou_threshold  = iou
    result, annotated = _detector.predict(img, return_annotated=True)

    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    resp = result_to_response(result, MODEL_PATH)
    resp["annotated_image_b64"] = f"data:image/jpeg;base64,{b64}"
    return resp


@app.post("/batch")
async def batch_predict(
    files: List[UploadFile] = File(...),
    conf:  float = Form(CONF_DEFAULT),
    iou:   float = Form(IOU_DEFAULT),
):
    """
    Detect defects in multiple uploaded images at once.
    Returns a list of detection results in the same order as the uploaded files.
    """
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Max 32 images per batch.")

    _detector.conf_threshold = conf
    _detector.iou_threshold  = iou

    results = []
    for upload in files:
        img    = decode_upload(upload)
        result = _detector.predict(img)
        rec    = result_to_response(result, MODEL_PATH)
        rec["filename"] = upload.filename
        results.append(rec)

    total_defects = sum(r["count"] for r in results)
    return {
        "status": "ok",
        "total_images": len(files),
        "total_defects": total_defects,
        "results": results,
    }


@app.get("/benchmark")
def run_benchmark(runs: int = 100):
    """Run a speed benchmark (dummy input, `runs` iterations)."""
    if runs > 500:
        raise HTTPException(status_code=400, detail="Max 500 benchmark runs.")
    bm = _detector.benchmark(runs)
    return bm


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="Fabric Defect Detection API Server")
    p.add_argument("--model",  default="yolo26n.onnx", help="ONNX model path")
    p.add_argument("--host",   default="0.0.0.0",      help="Host to bind")
    p.add_argument("--port",   type=int, default=8000,  help="Port to bind")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--reload", action="store_true",     help="Enable hot-reload")
    args = p.parse_args()

    MODEL_PATH   = args.model
    CONF_DEFAULT = args.conf
    IOU_DEFAULT  = args.iou

    print(f"[API] Starting server on http://{args.host}:{args.port}")
    print(f"[API] Interactive docs: http://localhost:{args.port}/docs")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
