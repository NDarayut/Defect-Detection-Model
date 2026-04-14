"""
video_inference.py
──────────────────
Run YOLO26 fabric defect detection on a video file.
Writes an annotated output video and an optional per-frame JSON log.

Usage
-----
python video_inference.py --source input.mp4 --model yolo26n.onnx
python video_inference.py --source input.mp4 --out output.mp4 --conf 0.30 --skip 2
python video_inference.py --source input.mp4 --show            # live preview
python video_inference.py --source input.mp4 --log results.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from inference_core import FabricDetector, InferenceResult

# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO26 Video Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--model",  default="yolo26n.onnx", help="ONNX model path")
    p.add_argument("--out",    default=None,
                   help="Output video path (default: <source>_pred.mp4)")
    p.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou",    type=float, default=0.45, help="IoU threshold")
    p.add_argument("--imgsz",  type=int,   default=640,  help="Input image size")
    p.add_argument("--skip",   type=int,   default=1,
                   help="Process every Nth frame (1 = all frames)")
    p.add_argument("--show",   action="store_true", help="Show live preview window")
    p.add_argument("--log",    default=None,
                   help="Save per-frame detection log as JSON")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write output video (useful with --show only)")
    p.add_argument("--classes", nargs="+",
                   help="Override class names (space-separated)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (useful for testing)")
    return p


def draw_hud(
    frame: np.ndarray,
    result: InferenceResult,
    frame_no: int,
    fps_actual: float,
    stats: dict,
) -> np.ndarray:
    """Overlay HUD (frame counter, FPS, detection count, per-class history)."""
    h, w = frame.shape[:2]

    # ── Top bar background ─────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Text line 1 ────────────────────────────────────────────────────────
    cv2.putText(
        frame,
        f"Frame {frame_no:06d}   FPS: {fps_actual:.1f}   "
        f"Infer: {result.inference_ms:.1f}ms   "
        f"Defects: {result.count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # ── Text line 2: per-class counts ──────────────────────────────────────
    summary = "  ".join(
        f"{cls}: {cnt}" for cls, cnt in sorted(stats["total_by_class"].items())
    )
    cv2.putText(
        frame,
        f"Total detections: {stats['total_detections']}   {summary}",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 255), 1, cv2.LINE_AA,
    )

    return frame


def run_video(args: argparse.Namespace) -> None:
    source  = Path(args.source)
    if not source.exists():
        print(f"[!] Video not found: {args.source}")
        sys.exit(1)

    # ── Output path ────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = source.parent / f"{source.stem}_pred.mp4"

    # ── Detector ───────────────────────────────────────────────────────────
    detector = FabricDetector(
        args.model,
        class_names=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.imgsz,
        class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
    )
    detector.warmup(3)

    # ── Open video ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[!] Cannot open video: {args.source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[INFO] Source  : {source.name}")
    print(f"       Size    : {frame_w}×{frame_h}  FPS: {src_fps:.2f}  "
          f"Frames: {total_frames}")
    if not args.no_save:
        print(f"       Output  : {out_path}")

    # ── Video writer ───────────────────────────────────────────────────────
    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path), fourcc, src_fps, (frame_w, frame_h)
        )

    # ── Stats ──────────────────────────────────────────────────────────────
    stats: dict = {
        "total_detections": 0,
        "total_by_class": defaultdict(int),
        "frames_processed": 0,
        "inference_times_ms": [],
    }
    frame_log = []
    frame_no  = 0
    t_start   = time.perf_counter()
    fps_actual = src_fps

    if args.show:
        win = "YOLO26 Fabric Defect — Press Q to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(frame_w, 1280), min(frame_h, 720))

    print("\n[INFO] Processing...")
    last_result: InferenceResult | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            if args.max_frames and frame_no > args.max_frames:
                break

            # ── Skip frames ────────────────────────────────────────────────
            if (frame_no - 1) % args.skip == 0:
                last_result = detector.predict(frame)
                stats["frames_processed"] += 1
                stats["total_detections"] += last_result.count
                stats["inference_times_ms"].append(last_result.inference_ms)
                for cls, cnt in last_result.class_counts().items():
                    stats["total_by_class"][cls] += cnt

                if args.log:
                    frame_log.append({
                        "frame": frame_no,
                        **last_result.to_dict(),
                    })

            # Annotate using the last result (even on skipped frames)
            if last_result is not None:
                frame = detector.draw(frame, last_result)

            # ── Compute rolling FPS ────────────────────────────────────────
            elapsed = time.perf_counter() - t_start
            if elapsed > 0:
                fps_actual = frame_no / elapsed

            # ── HUD ────────────────────────────────────────────────────────
            if last_result is not None:
                frame = draw_hud(frame, last_result, frame_no, fps_actual, stats)

            if writer:
                writer.write(frame)

            if args.show:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[INFO] Quit key pressed.")
                    break

            # ── Progress ───────────────────────────────────────────────────
            if frame_no % 50 == 0:
                pct = (frame_no / total_frames * 100) if total_frames > 0 else 0
                inf_arr = np.array(stats["inference_times_ms"])
                print(
                    f"  Frame {frame_no:6d}/{total_frames}  "
                    f"({pct:.0f}%)  "
                    f"avg_infer={inf_arr.mean():.1f}ms  "
                    f"fps={fps_actual:.1f}  "
                    f"detections={stats['total_detections']}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # ── Final summary ──────────────────────────────────────────────────────
    wall = time.perf_counter() - t_start
    inf_arr = np.array(stats["inference_times_ms"])

    print(f"\n{'='*55}")
    print(f"  Frames processed : {stats['frames_processed']}")
    print(f"  Total detections : {stats['total_detections']}")
    print(f"  Class breakdown  : {dict(stats['total_by_class'])}")
    if len(inf_arr):
        print(f"  Infer avg / std  : {inf_arr.mean():.1f}ms / {inf_arr.std():.1f}ms")
        print(f"  Infer FPS (est.) : {1000/inf_arr.mean():.1f}")
    print(f"  Wall time        : {wall:.1f}s  ({frame_no/wall:.1f} frames/s)")
    if not args.no_save:
        print(f"  Output video     : {out_path}")
    print(f"{'='*55}")

    # ── Write JSON log ─────────────────────────────────────────────────────
    if args.log:
        log_data = {
            "source": str(source),
            "model": args.model,
            "conf": args.conf,
            "total_frames": frame_no,
            "summary": {
                "total_detections": stats["total_detections"],
                "by_class": dict(stats["total_by_class"]),
            },
            "frames": frame_log,
        }
        with open(args.log, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"  JSON log         : {args.log}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_video(build_parser().parse_args())
