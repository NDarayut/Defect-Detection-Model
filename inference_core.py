"""
inference_core.py
─────────────────
Shared ONNX inference engine for YOLO26 fabric defect detection.
Works with any YOLO ONNX export (NMS-free end-to-end or raw grid output).

Usage (standalone):
    from inference_core import FabricDetector
    detector = FabricDetector("yolo26n.onnx")
    detections = detector.predict("fabric.jpg")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Optional provider imports ──────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

# ── Default fabric defect class names (override via FabricDetector(class_names=...)) ─
DEFAULT_CLASSES = [
    "seam", "Thread", "Warp_Weft", "hole", "Stain",
]

# ── Colour palette (BGR for OpenCV) ───────────────────────────────────────────
PALETTE = [
    (0,   114, 255),  # orange-blue
    (0,   200, 100),  # green
    (0,   80,  255),  # red
    (200, 180,   0),  # cyan
    (255, 80,  200),  # purple
    (30,  200, 255),  # yellow
    (255, 150,  30),  # blue
    (100, 255, 200),  # mint
]


@dataclass
class Detection:
    """Single bounding-box detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass
class InferenceResult:
    """Full result for a single image inference."""
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)   # (H, W)
    input_shape: Tuple[int, int] = (640, 640)

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def count(self) -> int:
        return len(self.detections)

    def class_counts(self) -> dict:
        counts: dict = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": self.count,
            "class_counts": self.class_counts(),
            "timing_ms": {
                "preprocess": round(self.preprocess_ms, 2),
                "inference": round(self.inference_ms, 2),
                "postprocess": round(self.postprocess_ms, 2),
                "total": round(self.total_ms, 2),
            },
            "image_shape": list(self.image_shape),
        }


class FabricDetector:
    """
    ONNX-based YOLO26 fabric defect detector.

    Handles both:
      • End-to-end NMS-free output: (batch, max_det, 6)  [x1,y1,x2,y2,conf,cls]
      • Raw grid output:            (batch, 4+nc, anchors) — applies NMS internally
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = str(model_path)
        self.class_names = class_names or DEFAULT_CLASSES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # ── Provider selection ─────────────────────────────────────────────
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

        # ── Load session ───────────────────────────────────────────────────
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 0   # use all available cores
        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_opts, providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,640,640]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.output_shapes = [o.shape for o in self.session.get_outputs()]

        # Detect output format
        self._e2e = self._is_end_to_end()

        print(f"[FabricDetector] Loaded: {Path(model_path).name}")
        print(f"  Provider   : {self.session.get_providers()[0]}")
        print(f"  Input      : {self.input_name} {self.input_shape}")
        print(f"  Outputs    : {list(zip(self.output_names, self.output_shapes))}")
        print(f"  Mode       : {'end-to-end (NMS-free)' if self._e2e else 'raw (NMS applied)'}")
        print(f"  Classes    : {len(self.class_names)} → {self.class_names}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        image: "str | Path | np.ndarray",
        return_annotated: bool = False,
    ) -> "InferenceResult | Tuple[InferenceResult, np.ndarray]":
        """
        Run detection on a single image.

        Args:
            image: file path (str/Path) or BGR numpy array (H,W,3).
            return_annotated: if True, also returns the drawn image.

        Returns:
            InferenceResult  (or tuple with annotated BGR image).
        """
        img_bgr = self._load(image)
        result = self._infer(img_bgr)
        if return_annotated:
            annotated = self.draw(img_bgr, result)
            return result, annotated
        return result

    def predict_batch(
        self,
        images: List["str | Path | np.ndarray"],
    ) -> List[InferenceResult]:
        return [self.predict(img) for img in images]

    def draw(
        self,
        image: np.ndarray,
        result: InferenceResult,
        line_thickness: int = 2,
        font_scale: float = 0.55,
        show_conf: bool = True,
        show_label: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes on a copy of the image (BGR)."""
        out = image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.xyxy
            color = PALETTE[det.class_id % len(PALETTE)]

            cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

            if show_label:
                label = det.class_name
                if show_conf:
                    label += f" {det.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(
                    out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        # ── Stats overlay ──────────────────────────────────────────────────
        h, w = out.shape[:2]
        overlay_txt = f"{result.count} defect(s) | {result.inference_ms:.1f}ms"
        cv2.putText(
            out, overlay_txt, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, overlay_txt, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
        )
        return out

    def warmup(self, n: int = 3) -> None:
        """Run a few dummy inferences to warm up the runtime."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(n):
            self._infer(dummy)
        print(f"[FabricDetector] Warmup done ({n} passes)")

    def benchmark(self, n: int = 100) -> dict:
        """Measure average inference speed over n runs."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.warmup()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self._infer(dummy)
            times.append((time.perf_counter() - t0) * 1000)
        return {
            "runs": n,
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
            "fps": round(1000 / float(np.mean(times)), 1),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _is_end_to_end(self) -> bool:
        """
        Heuristic: if the first output has 3 dims and the last dim is small
        (e.g. 6 for [x1,y1,x2,y2,conf,cls]) → NMS-free end-to-end format.
        """
        shape = self.output_shapes[0]
        if len(shape) == 3 and (shape[-1] in (6, 7)):
            return True
        return False

    @staticmethod
    def _load(image: "str | Path | np.ndarray") -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            return img
        return image.copy()

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Letterbox resize → normalise → NCHW float32.
        Returns (blob, scale_x, scale_y, pad_left, pad_top).
        """
        h0, w0 = img_bgr.shape[:2]
        size = self.input_size

        # Scale
        scale = min(size / h0, size / w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))

        # Resize
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_top  = (size - nh) // 2
        pad_left = (size - nw) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

        # BGR → RGB, HWC → NCHW, uint8 → float32 [0,1]
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis])

        scale_x = w0 / nw
        scale_y = h0 / nh
        return blob, scale_x, scale_y, pad_left, pad_top

    def _infer(self, img_bgr: np.ndarray) -> InferenceResult:
        h0, w0 = img_bgr.shape[:2]

        # ── Pre-process ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        blob, sx, sy, pl, pt = self._preprocess(img_bgr)
        pre_ms = (time.perf_counter() - t0) * 1000

        # ── Run ONNX ───────────────────────────────────────────────────────
        t1 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        inf_ms = (time.perf_counter() - t1) * 1000

        # ── Post-process ───────────────────────────────────────────────────
        t2 = time.perf_counter()
        if self._e2e:
            dets = self._postprocess_e2e(outputs, sx, sy, pl, pt, w0, h0)
        else:
            dets = self._postprocess_raw(outputs, sx, sy, pl, pt, w0, h0)
        post_ms = (time.perf_counter() - t2) * 1000

        return InferenceResult(
            detections=dets,
            inference_ms=inf_ms,
            preprocess_ms=pre_ms,
            postprocess_ms=post_ms,
            image_shape=(h0, w0),
            input_shape=(self.input_size, self.input_size),
        )

    def _to_detection(self, x1, y1, x2, y2, conf, cls_id,
                       sx, sy, pl, pt, w0, h0) -> Optional[Detection]:
        # Un-pad then un-scale
        x1 = (x1 - pl) * sx
        y1 = (y1 - pt) * sy
        x2 = (x2 - pl) * sx
        y2 = (y2 - pt) * sy
        # Clip
        x1 = max(0.0, min(float(x1), w0))
        y1 = max(0.0, min(float(y1), h0))
        x2 = max(0.0, min(float(x2), w0))
        y2 = max(0.0, min(float(y2), h0))
        if x2 <= x1 or y2 <= y1:
            return None
        cls_id = int(cls_id)
        cls_name = (
            self.class_names[cls_id] if cls_id < len(self.class_names)
            else f"class_{cls_id}"
        )
        return Detection(x1, y1, x2, y2, float(conf), cls_id, cls_name)

    def _postprocess_e2e(self, outputs, sx, sy, pl, pt, w0, h0) -> List[Detection]:
        """
        Parse NMS-free end-to-end output.
        Expected shape: (1, N, 6)  →  [x1, y1, x2, y2, conf, cls_id]
        or              (1, N, 7)  →  [batch, x1, y1, x2, y2, conf, cls_id]
        """
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]   # (N, 6) or (N, 7)

        dets = []
        for row in raw:
            if row.shape[0] == 7:
                _, x1, y1, x2, y2, conf, cls_id = row
            else:
                x1, y1, x2, y2, conf, cls_id = row

            if float(conf) < self.conf_threshold:
                continue
            d = self._to_detection(x1, y1, x2, y2, conf, int(cls_id),
                                   sx, sy, pl, pt, w0, h0)
            if d:
                dets.append(d)
        return dets

    def _postprocess_raw(self, outputs, sx, sy, pl, pt, w0, h0) -> List[Detection]:
        """
        Parse raw grid output and apply NMS manually.
        Expected shape: (1, 4+nc, num_anchors)
        """
        raw = outputs[0]
        # Handle transposed layout
        if raw.ndim == 3 and raw.shape[1] > raw.shape[2]:
            raw = raw.transpose(0, 2, 1)   # → (1, anchors, 4+nc)
        raw = raw[0]   # (anchors, 4+nc)

        nc = len(self.class_names)
        boxes_xywh = raw[:, :4]
        scores     = raw[:, 4:]
        if scores.shape[1] == 1:
            # single objectness score
            obj_conf   = scores[:, 0]
            cls_conf   = np.ones((len(raw), nc), dtype=np.float32) / nc
            class_ids  = np.zeros(len(raw), dtype=np.int32)
            confs      = obj_conf
        else:
            cls_conf   = scores[:, :nc] if scores.shape[1] > nc else scores
            confs      = cls_conf.max(axis=1)
            class_ids  = cls_conf.argmax(axis=1)

        mask = confs >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confs      = confs[mask]
        class_ids  = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh → xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # cv2 NMS expects (x,y,w,h) and returns indices
        boxes_cv = boxes_xyxy.copy()
        boxes_cv[:, 2] -= boxes_cv[:, 0]
        boxes_cv[:, 3] -= boxes_cv[:, 1]
        indices = cv2.dnn.NMSBoxes(
            boxes_cv.tolist(),
            confs.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        dets = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            d = self._to_detection(x1, y1, x2, y2, confs[i], class_ids[i],
                                   sx, sy, pl, pt, w0, h0)
            if d:
                dets.append(d)
        return dets


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.onnx"
    img_path   = sys.argv[2] if len(sys.argv) > 2 else None

    det = FabricDetector(model_path)
    det.warmup()

    if img_path:
        result, annotated = det.predict(img_path, return_annotated=True)
        print(f"\nDetections ({result.count}):")
        for d in result.detections:
            print(f"  {d.class_name:15s} conf={d.confidence:.3f}  "
                  f"bbox=[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")
        out_path = "annotated_output.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\nAnnotated image saved to: {out_path}")
    else:
        print("\nRunning benchmark (100 iterations)...")
        bm = det.benchmark(100)
        print(f"  Mean : {bm['mean_ms']} ms")
        print(f"  Std  : {bm['std_ms']} ms")
        print(f"  FPS  : {bm['fps']}")
