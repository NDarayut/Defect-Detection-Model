"""
batch_inference.py
──────────────────
Process an entire folder of fabric images, save annotated results,
and produce a detailed CSV + summary report.

Usage
-----
python batch_inference.py --source ./fabric_images/ --model yolo26n.onnx
python batch_inference.py --source ./images/ --out-dir ./batch_out/ --workers 4
python batch_inference.py --source ./images/ --no-save --csv report.csv
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from inference_core import FabricDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    detector: FabricDetector,
    out_dir: Path | None,
    annotate: bool,
) -> dict:
    """Process a single image. Returns a summary dict for the CSV."""
    try:
        result, annotated = detector.predict(img_path, return_annotated=True)

        if out_dir and annotate:
            out_path = out_dir / f"{img_path.stem}_pred.jpg"
            cv2.imwrite(str(out_path), annotated)

        row = {
            "filename":         img_path.name,
            "path":             str(img_path),
            "width":            result.image_shape[1],
            "height":           result.image_shape[0],
            "num_defects":      result.count,
            "inference_ms":     round(result.inference_ms, 2),
            "total_ms":         round(result.total_ms, 2),
            "has_defect":       int(result.count > 0),
            "class_counts_json": json.dumps(result.class_counts()),
        }

        # Add per-detection rows for the detail CSV
        row["detections"] = result.detections
        return row

    except Exception as e:
        return {
            "filename": img_path.name,
            "path": str(img_path),
            "error": str(e),
            "num_defects": -1,
            "has_defect": -1,
        }


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    """Write per-image summary CSV."""
    fieldnames = [
        "filename", "path", "width", "height",
        "num_defects", "has_defect",
        "class_counts_json",
        "inference_ms", "total_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_detail_csv(rows: list[dict], csv_path: Path) -> None:
    """Write one row per bounding box detection."""
    fieldnames = [
        "filename", "class_name", "class_id", "confidence",
        "x1", "y1", "x2", "y2", "width_px", "height_px",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for det in row.get("detections", []):
                writer.writerow({
                    "filename":    row["filename"],
                    "class_name":  det.class_name,
                    "class_id":    det.class_id,
                    "confidence":  round(det.confidence, 4),
                    "x1": round(det.x1, 1), "y1": round(det.y1, 1),
                    "x2": round(det.x2, 1), "y2": round(det.y2, 1),
                    "width_px":    round(det.width, 1),
                    "height_px":   round(det.height, 1),
                })


def print_report(rows: list[dict], wall_s: float, out_dir: Path | None) -> None:
    valid = [r for r in rows if r.get("num_defects", -1) >= 0]
    errors = [r for r in rows if "error" in r]

    total_imgs    = len(rows)
    defect_imgs   = sum(1 for r in valid if r["has_defect"] == 1)
    clean_imgs    = sum(1 for r in valid if r["has_defect"] == 0)
    total_defects = sum(r.get("num_defects", 0) for r in valid)

    all_classes: dict[str, int] = {}
    for r in valid:
        for cls, cnt in json.loads(r.get("class_counts_json", "{}")).items():
            all_classes[cls] = all_classes.get(cls, 0) + cnt

    inf_times = [r["inference_ms"] for r in valid if "inference_ms" in r]
    avg_inf = np.mean(inf_times) if inf_times else 0

    print(f"\n{'═'*58}")
    print(f"  BATCH INFERENCE REPORT")
    print(f"{'═'*58}")
    print(f"  Images processed : {total_imgs}")
    print(f"  With defects     : {defect_imgs} ({defect_imgs/max(total_imgs,1)*100:.1f}%)")
    print(f"  Clean            : {clean_imgs} ({clean_imgs/max(total_imgs,1)*100:.1f}%)")
    print(f"  Errors           : {len(errors)}")
    print(f"  Total defects    : {total_defects}")
    print(f"  Avg per image    : {total_defects/max(len(valid),1):.2f}")
    if all_classes:
        print(f"\n  Defects by class:")
        for cls, cnt in sorted(all_classes.items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt // max(max(all_classes.values())//20, 1), 30)
            print(f"    {cls:18s} {cnt:5d}  {bar}")
    print(f"\n  Avg inference    : {avg_inf:.1f} ms")
    print(f"  Throughput       : {total_imgs/wall_s:.1f} img/s")
    print(f"  Wall time        : {wall_s:.2f}s")
    if out_dir:
        print(f"\n  Output dir       : {out_dir}")
    print(f"{'═'*58}\n")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="YOLO26 batch image inference with CSV reporting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",  required=True, help="Folder of images")
    p.add_argument("--model",   default="yolo26n.onnx")
    p.add_argument("--out-dir", default="./batch_output",
                   help="Output directory for annotated images")
    p.add_argument("--csv",     default="batch_results.csv",
                   help="Summary CSV output path")
    p.add_argument("--detail-csv", default=None,
                   help="Per-detection detail CSV (optional)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving annotated images (only generate CSV)")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--workers", type=int,   default=1,
                   help="Number of parallel threads (1 = sequential)")
    p.add_argument("--recursive", action="store_true",
                   help="Search images recursively in sub-folders")
    p.add_argument("--classes", nargs="+",
                   help="Override class names (space-separated)")
    args = p.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"[!] Not a directory: {args.source}")
        return

    # Collect images
    glob_fn = source.rglob if args.recursive else source.glob
    images = sorted([
        f for f in glob_fn("*")
        if f.suffix.lower() in IMG_EXTS
    ])
    if not images:
        print(f"[!] No images found in {source}")
        return
    print(f"[INFO] Found {len(images)} image(s) in {source}")

    # Output dir
    out_dir = None
    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Detector (shared across threads — ONNX Runtime is thread-safe for inference)
    detector = FabricDetector(
        args.model,
        class_names=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.imgsz,
        class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
    )
    detector.warmup(2)

    # ── Run ────────────────────────────────────────────────────────────────
    rows: list[dict] = [None] * len(images)
    t_start = time.perf_counter()

    if args.workers <= 1:
        for i, img_path in enumerate(images, 1):
            rows[i - 1] = process_image(img_path, detector, out_dir, not args.no_save)
            if i % 20 == 0 or i == len(images):
                print(f"  [{i:4d}/{len(images)}] {img_path.name}  "
                      f"defects={rows[i-1].get('num_defects', '?')}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_image, img, detector, out_dir, not args.no_save
                ): idx
                for idx, img in enumerate(images)
            }
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                rows[idx] = future.result()
                done += 1
                if done % 20 == 0 or done == len(images):
                    print(f"  [{done:4d}/{len(images)}]")

    wall_s = time.perf_counter() - t_start

    # ── Write CSVs ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    write_summary_csv(rows, csv_path)
    print(f"[INFO] Summary CSV saved: {csv_path}")

    if args.detail_csv:
        write_detail_csv(rows, Path(args.detail_csv))
        print(f"[INFO] Detail CSV saved : {args.detail_csv}")

    print_report(rows, wall_s, out_dir)


if __name__ == "__main__":
    main()
