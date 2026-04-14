#!/usr/bin/env python3
"""
cli_inference.py
────────────────
Command-line interface for YOLO26 fabric defect detection.

Examples
--------
# Single image
python cli_inference.py image --source fabric.jpg --model yolo26n.onnx

# Folder of images
python cli_inference.py image --source ./fabric_images/ --model yolo26n.onnx --save

# Save annotated output with custom threshold
python cli_inference.py image --source fabric.jpg --conf 0.35 --save --out-dir ./results/

# Run speed benchmark
python cli_inference.py benchmark --model yolo26n.onnx --runs 200

# Print JSON output
python cli_inference.py image --source fabric.jpg --json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from inference_core import FabricDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        imgs = sorted([f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS])
        if not imgs:
            print(f"[!] No images found in {source}")
            sys.exit(1)
        return imgs
    print(f"[!] Source not found: {source}")
    sys.exit(1)


def print_result_table(path: Path, result) -> None:
    print(f"\n{'─'*60}")
    print(f"  Image  : {path.name}  ({result.image_shape[1]}×{result.image_shape[0]})")
    print(f"  Found  : {result.count} defect(s)")
    if result.count:
        print(f"  Classes: {result.class_counts()}")
    print(f"  Timing : pre={result.preprocess_ms:.1f}ms  "
          f"infer={result.inference_ms:.1f}ms  "
          f"post={result.postprocess_ms:.1f}ms  "
          f"total={result.total_ms:.1f}ms")
    for d in result.detections:
        print(f"    [{d.class_name:15s}] conf={d.confidence:.3f}  "
              f"box=[{d.x1:.0f},{d.y1:.0f}→{d.x2:.0f},{d.y2:.0f}]  "
              f"size={d.width:.0f}×{d.height:.0f}px")


def build_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="yolo26n.onnx", help="Path to ONNX model")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou",  type=float, default=0.45, help="IoU threshold (NMS)")
    p.add_argument("--imgsz", type=int, default=640,   help="Input image size")
    p.add_argument(
        "--classes", nargs="+",
        help="Override class names (space-separated, in class-id order)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_image(args: argparse.Namespace) -> None:
    detector = FabricDetector(
        args.model,
        class_names=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.imgsz,
        class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
    )
    detector.warmup(2)

    images = collect_images(args.source)
    print(f"\n[INFO] Running on {len(images)} image(s)...")

    out_dir = Path(args.out_dir) if args.save else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_wall = time.perf_counter()

    for img_path in images:
        result, annotated = detector.predict(img_path, return_annotated=True)

        if args.json:
            rec = result.to_dict()
            rec["image"] = str(img_path)
            all_results.append(rec)
        else:
            print_result_table(img_path, result)

        if args.save and out_dir:
            out_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
            cv2.imwrite(str(out_path), annotated)
            if not args.json:
                print(f"  → Saved: {out_path}")

        if args.show:
            win = "Fabric Defect Detection  [q to quit]"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, annotated)
            key = cv2.waitKey(0 if len(images) == 1 else 500)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    wall_s = time.perf_counter() - t_wall
    if args.json:
        summary = {
            "model": args.model,
            "total_images": len(images),
            "wall_time_s": round(wall_s, 3),
            "results": all_results,
        }
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Done — {len(images)} image(s) in {wall_s:.2f}s "
              f"({len(images)/wall_s:.1f} img/s)")
        print(f"{'='*60}")

    if args.show:
        cv2.destroyAllWindows()


def cmd_benchmark(args: argparse.Namespace) -> None:
    detector = FabricDetector(
        args.model,
        class_names=args.classes,
        conf_threshold=args.conf,
        input_size=args.imgsz,
    )
    print(f"\n[INFO] Benchmarking — {args.runs} iterations...")
    bm = detector.benchmark(args.runs)
    print(f"\n{'─'*40}")
    print(f"  Model  : {args.model}")
    print(f"  Runs   : {bm['runs']}")
    print(f"  Mean   : {bm['mean_ms']} ms")
    print(f"  Std    : {bm['std_ms']} ms")
    print(f"  Min    : {bm['min_ms']} ms")
    print(f"  Max    : {bm['max_ms']} ms")
    print(f"  FPS    : {bm['fps']}")
    print(f"{'─'*40}")
    if args.json:
        print(json.dumps(bm, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fabric-detect",
        description="YOLO26 Fabric Defect Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── image sub-command ──────────────────────────────────────────────────
    p_img = sub.add_parser("image", help="Detect defects in image(s)")
    build_common_args(p_img)
    p_img.add_argument("--source", required=True, help="Image path or folder")
    p_img.add_argument("--save",   action="store_true", help="Save annotated output")
    p_img.add_argument("--show",   action="store_true", help="Display result window")
    p_img.add_argument("--out-dir", default="./cli_output", help="Output directory")
    p_img.add_argument("--json",   action="store_true", help="Print JSON output")

    # ── benchmark sub-command ──────────────────────────────────────────────
    p_bm = sub.add_parser("benchmark", help="Measure inference speed")
    build_common_args(p_bm)
    p_bm.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    p_bm.add_argument("--json", action="store_true", help="Print JSON output")

    args = parser.parse_args()

    if args.command == "image":
        cmd_image(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
