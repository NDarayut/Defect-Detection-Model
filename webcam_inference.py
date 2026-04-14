"""
webcam_inference.py
───────────────────
Real-time YOLO26 fabric defect detection from webcam or RTSP stream.

Usage
-----
python webcam_inference.py                          # default webcam (index 0)
python webcam_inference.py --source 1               # webcam index 1
python webcam_inference.py --source rtsp://...      # RTSP stream
python webcam_inference.py --record output.mp4      # record output
python webcam_inference.py --conf 0.40 --flip       # higher conf + mirror flip

Controls (press while window is focused)
-----------------------------------------
  Q / ESC  : quit
  S        : save snapshot
  P        : pause / resume
  +/-      : increase / decrease confidence threshold live
  F        : toggle fullscreen
  H        : toggle HUD
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from inference_core import FabricDetector, PALETTE


# ─────────────────────────────────────────────────────────────────────────────

class WebcamDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.detector = FabricDetector(
            args.model,
            class_names=args.classes,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            input_size=args.imgsz,
            class_names=["seam", "Thread", "Warp_Weft", "hole", "Stain"]
        )
        self.detector.warmup(5)

        self.flip        = args.flip
        self.record_path = args.record
        self.snapshot_dir = Path(args.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state
        self.conf         = args.conf
        self.paused       = False
        self.show_hud     = True
        self.fullscreen   = False
        self.frame_no     = 0
        self.fps_ring     = []   # rolling FPS buffer
        self.writer: cv2.VideoWriter | None = None

        # ── Open source ────────────────────────────────────────────────────
        src = args.source
        if src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[WebcamDetector] Source: {args.source}  ({self.w}×{self.h})")

        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.record_path, fourcc, 25, (self.w, self.h)
            )
            print(f"[WebcamDetector] Recording to: {self.record_path}")

    def run(self) -> None:
        win = "YOLO26 Fabric Defect Detector"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(self.w, 1280), min(self.h, 720))

        last_result = None
        last_frame  = None
        t_last = time.perf_counter()

        print("[INFO] Running. Press Q/ESC to quit, S to snapshot, P to pause.\n")

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("[INFO] Stream ended or camera disconnected.")
                    break

                if self.flip:
                    frame = cv2.flip(frame, 1)

                # ── Update confidence if changed via key ───────────────────
                self.detector.conf_threshold = self.conf

                last_result = self.detector.predict(frame)
                annotated = self.detector.draw(frame, last_result)

                # ── FPS ────────────────────────────────────────────────────
                now = time.perf_counter()
                inst_fps = 1.0 / max(now - t_last, 1e-6)
                t_last = now
                self.fps_ring.append(inst_fps)
                if len(self.fps_ring) > 30:
                    self.fps_ring.pop(0)
                avg_fps = np.mean(self.fps_ring)

                if self.show_hud:
                    annotated = self._draw_hud(annotated, last_result, avg_fps)

                last_frame = annotated
                self.frame_no += 1

                if self.writer:
                    self.writer.write(annotated)
            else:
                # Paused: re-draw last frame with PAUSED banner
                if last_frame is not None:
                    annotated = last_frame.copy()
                    cv2.putText(
                        annotated, "PAUSED", (self.w//2 - 70, self.h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA
                    )
                else:
                    annotated = np.zeros((self.h, self.w, 3), dtype=np.uint8)

            cv2.imshow(win, annotated)
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key, last_frame, win)
            if key in (ord("q"), 27):
                break

        self._cleanup()

    def _draw_hud(self, frame: np.ndarray, result, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # Semi-transparent top strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Line 1
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}   Infer: {result.inference_ms:.1f}ms   "
            f"Conf: {self.conf:.2f}   Frame: {self.frame_no}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Line 2: defect counts per class
        cls_counts = result.class_counts()
        if cls_counts:
            txt = "  ".join(f"{k}: {v}" for k, v in sorted(cls_counts.items()))
        else:
            txt = "No defects detected"
        cv2.putText(
            frame, txt, (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 220, 120), 1, cv2.LINE_AA,
        )

        # ── Mini class-count bar (bottom-left) ────────────────────────────
        if cls_counts:
            y_base = h - 10
            for i, (cls_name, cnt) in enumerate(sorted(cls_counts.items())):
                color = PALETTE[i % len(PALETTE)]
                bar_w = min(cnt * 20, 150)
                cv2.rectangle(frame, (10, y_base - 16), (10 + bar_w, y_base),
                               color, -1)
                cv2.putText(frame, f"{cls_name} ({cnt})",
                             (14, y_base - 3),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                y_base -= 24

        return frame

    def _handle_key(self, key: int, last_frame, win: str) -> None:
        if key == ord("s") and last_frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(path), last_frame)
            print(f"[Snapshot] Saved: {path}")

        elif key == ord("p"):
            self.paused = not self.paused
            state = "Paused" if self.paused else "Resumed"
            print(f"[INFO] {state}")

        elif key == ord("+") or key == ord("="):
            self.conf = min(0.95, round(self.conf + 0.05, 2))
            print(f"[INFO] Confidence threshold: {self.conf:.2f}")

        elif key == ord("-"):
            self.conf = max(0.05, round(self.conf - 0.05, 2))
            print(f"[INFO] Confidence threshold: {self.conf:.2f}")

        elif key == ord("h"):
            self.show_hud = not self.show_hud

        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            flag = cv2.WND_PROP_FULLSCREEN
            mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, flag, mode)

    def _cleanup(self) -> None:
        self.cap.release()
        if self.writer:
            self.writer.release()
            print(f"[INFO] Recording saved: {self.record_path}")
        cv2.destroyAllWindows()
        print(f"[INFO] Done. Processed {self.frame_no} frames.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-time YOLO26 webcam/stream inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="Webcam index (0,1,...) or RTSP/HTTP URL")
    p.add_argument("--model", default="yolo26n.onnx")
    p.add_argument("--conf",  type=float, default=0.25)
    p.add_argument("--iou",   type=float, default=0.45)
    p.add_argument("--imgsz", type=int,   default=640)
    p.add_argument("--flip",  action="store_true", help="Mirror the frame horizontally")
    p.add_argument("--record", default=None, metavar="FILE",
                   help="Record annotated stream to a video file")
    p.add_argument("--snapshot-dir", default="./snapshots",
                   help="Directory for saved snapshots (S key)")
    p.add_argument("--classes", nargs="+",
                   help="Override class names (space-separated)")
    args = p.parse_args()

    detector = WebcamDetector(args)
    detector.run()


if __name__ == "__main__":
    main()
