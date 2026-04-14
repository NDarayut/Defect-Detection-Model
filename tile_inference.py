"""
tile_inference.py
─────────────────
Sliding-window (tile) inference for high-resolution fabric images.
Runs the detector on overlapping tiles, then merges boxes with NMS.
Ideal for large scanner images where defects are small relative to the image.

Usage
-----
python tile_inference.py --source large_fabric.jpg --model yolo26n.onnx
python tile_inference.py --source fabric.jpg --tile-size 640 --overlap 0.25
python tile_inference.py --source fabric.jpg --show --save
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from inference_core import FabricDetector, Detection, InferenceResult, PALETTE


# ─────────────────────────────────────────────────────────────────────────────

def tile_image(
    image: np.ndarray,
    tile_size: int,
    overlap: float,
) -> list[tuple[np.ndarray, int, int]]:
    """
    Split image into overlapping tiles.
    Returns list of (tile, x_offset, y_offset).
    """
    h, w = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    step = max(step, 1)

    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(x2 - tile_size, 0)
            y1 = max(y2 - tile_size, 0)
            tile = image[y1:y2, x1:x2]
            tiles.append((tile, x1, y1))
    return tiles


def global_nms(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Apply NMS across all detections from all tiles."""
    if not detections:
        return []

    boxes  = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
    scores = np.array([d.confidence for d in detections])

    # Convert to (x,y,w,h) for cv2
    boxes_cv = boxes.copy()
    boxes_cv[:, 2] -= boxes_cv[:, 0]
    boxes_cv[:, 3] -= boxes_cv[:, 1]

    indices = cv2.dnn.NMSBoxes(
        boxes_cv.tolist(),
        scores.tolist(),
        score_threshold=0.0,   # already filtered upstream
        nms_threshold=iou_threshold,
    )
    if len(indices) == 0:
        return []
    return [detections[i] for i in indices.flatten()]


def run_tile_inference(
    image: np.ndarray,
    detector: FabricDetector,
    tile_size: int = 640,
    overlap: float = 0.20,
    merge_iou: float = 0.50,
) -> tuple[list[Detection], dict]:
    """
    Full tile inference pipeline.
    Returns (merged_detections, stats_dict).
    """
    h, w = image.shape[:2]
    tiles = tile_image(image, tile_size, overlap)

    all_dets: list[Detection] = []
    total_inf_ms = 0.0

    for tile, ox, oy in tiles:
        result = detector.predict(tile)
        total_inf_ms += result.inference_ms

        # Translate bounding boxes back to original image coordinates
        for det in result.detections:
            translated = Detection(
                x1=det.x1 + ox,
                y1=det.y1 + oy,
                x2=det.x2 + ox,
                y2=det.y2 + oy,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            # Clip to image bounds
            translated = Detection(
                x1=max(0, min(translated.x1, w)),
                y1=max(0, min(translated.y1, h)),
                x2=max(0, min(translated.x2, w)),
                y2=max(0, min(translated.y2, h)),
                confidence=translated.confidence,
                class_id=translated.class_id,
                class_name=translated.class_name,
            )
            if translated.x2 > translated.x1 and translated.y2 > translated.y1:
                all_dets.append(translated)

    merged = global_nms(all_dets, merge_iou)
    stats  = {
        "num_tiles":        len(tiles),
        "raw_detections":   len(all_dets),
        "merged_detections": len(merged),
        "total_infer_ms":   round(total_inf_ms, 2),
        "avg_infer_ms":     round(total_inf_ms / max(len(tiles), 1), 2),
        "image_shape":      (h, w),
        "tile_size":        tile_size,
        "overlap":          overlap,
    }
    return merged, stats


def draw_tile_grid(
    image: np.ndarray,
    tile_size: int,
    overlap: float,
    alpha: float = 0.15,
) -> np.ndarray:
    """Visualise tile grid boundaries on the image."""
    out = image.copy()
    h, w = out.shape[:2]
    step = int(tile_size * (1 - overlap))

    overlay = out.copy()
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (180, 180, 180), 1)
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (180, 180, 180), 1)

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def draw_detections_full(
    image: np.ndarray,
    detections: list[Detection],
    class_names: list[str],
    line_thickness: int = 2,
) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = PALETTE[det.class_id % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Tile-based inference for high-resolution fabric images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",    required=True,     help="Input image path")
    p.add_argument("--model",     default="yolo26n.onnx")
    p.add_argument("--conf",      type=float, default=0.25)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--imgsz",     type=int,   default=640,
                   help="Tile size (model input size)")
    p.add_argument("--tile-size", type=int,   default=None,
                   help="Override tile size (default: same as --imgsz)")
    p.add_argument("--overlap",   type=float, default=0.20,
                   help="Tile overlap fraction (0.0–0.5)")
    p.add_argument("--merge-iou", type=float, default=0.50,
                   help="IoU threshold for merging boxes across tiles")
    p.add_argument("--show",      action="store_true")
    p.add_argument("--save",      action="store_true")
    p.add_argument("--show-grid", action="store_true",
                   help="Overlay tile grid on output image")
    p.add_argument("--out",       default=None)
    p.add_argument("--classes",   nargs="+")
    args = p.parse_args()

    src = Path(args.source)
    image = cv2.imread(str(src))
    if image is None:
        print(f"[!] Cannot read image: {src}")
        return

    h, w = image.shape[:2]
    tile_size = args.tile_size or args.imgsz
    print(f"[INFO] Image    : {src.name}  ({w}×{h})")
    print(f"[INFO] Tile size: {tile_size}px  Overlap: {args.overlap:.0%}")

    detector = FabricDetector(
        args.model,
        class_names=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.imgsz,
        class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
    )
    detector.warmup(2)

    detections, stats = run_tile_inference(
        image, detector,
        tile_size=tile_size,
        overlap=args.overlap,
        merge_iou=args.merge_iou,
    )

    print(f"\n{'─'*50}")
    print(f"  Tiles processed   : {stats['num_tiles']}")
    print(f"  Raw detections    : {stats['raw_detections']}")
    print(f"  After NMS merge   : {stats['merged_detections']}")
    print(f"  Total infer time  : {stats['total_infer_ms']} ms")
    print(f"  Avg per tile      : {stats['avg_infer_ms']} ms")
    for d in sorted(detections, key=lambda x: -x.confidence):
        print(f"    [{d.class_name:15s}] conf={d.confidence:.3f}  "
              f"[{d.x1:.0f},{d.y1:.0f}→{d.x2:.0f},{d.y2:.0f}]")
    print(f"{'─'*50}")

    # Visualise
    out_img = image.copy()
    if args.show_grid:
        out_img = draw_tile_grid(out_img, tile_size, args.overlap)
    out_img = draw_detections_full(out_img, detections, detector.class_names)

    if args.save or args.out:
        out_path = Path(args.out) if args.out else src.parent / f"{src.stem}_tiled.jpg"
        cv2.imwrite(str(out_path), out_img)
        print(f"[INFO] Saved: {out_path}")

    if args.show:
        # Scale down for display if very large
        disp = out_img
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            disp = cv2.resize(disp, (int(w * scale), int(h * scale)))
        cv2.imshow("Tile Inference", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
