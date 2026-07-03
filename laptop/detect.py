#!/usr/bin/env python3
"""
Local YOLO detection with the trained grocery model (ultralytics).

Runs training/runs/grocery/weights/best.pt (or any ultralytics .pt) on the
laptop so the full multi-camera pipeline is testable without a Pi or a
compiled .hef. ultralytics/torch are lazy-imported inside YoloDetector so
this module still imports on machines that only view Pi streams.

Headless sanity check (no UI):
  python3 detect.py --source demo/camA.mp4 --max-frames 100
"""
import argparse
import time
from pathlib import Path

import cv2

from sources import Detection

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent
                      / "training" / "runs" / "grocery" / "weights" / "best.pt")


class YoloDetector:
    """Wraps an ultralytics YOLO .pt; .detect(frame_bgr) -> list[Detection].
    Class ids/names come from the checkpoint itself (== data.yaml order)."""

    def __init__(self, weights=None, imgsz=640, conf=0.35, iou=0.5,
                 device="auto"):
        from ultralytics import YOLO   # lazy — local mode only
        import torch
        if device == "auto":
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.model = YOLO(weights or DEFAULT_WEIGHTS)
        self.names = self.model.names          # {cls_id: label}

    def detect(self, frame_bgr):
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                                 iou=self.iou, device=self.device,
                                 verbose=False)[0]
        h, w = frame_bgr.shape[:2]
        dets = []
        for xyxy, score, cls in zip(res.boxes.xyxy.tolist(),
                                    res.boxes.conf.tolist(),
                                    res.boxes.cls.tolist()):
            cls_id = int(cls)
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            dets.append(Detection(
                label=self.names.get(cls_id, str(cls_id)), cls_id=cls_id,
                score=float(score),
                x1=max(0, x1), y1=max(0, y1),
                x2=min(w - 1, x2), y2=min(h - 1, y2)))
        return dets


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--source", required=True,
                    help="video path or webcam index")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = run to EOF)")
    args = ap.parse_args()

    spec = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    det = YoloDetector(args.weights, conf=args.conf, device=args.device)

    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = det.detect(frame)
            n += 1
            names = ", ".join(f"{d.label}:{d.score:.2f}" for d in dets) or "-"
            print(f"[{n:04d}] {len(dets):2d} det  {names}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    fps = n / max(1e-3, time.time() - t0)
    print(f"[detect] {n} frames, {fps:.1f} FPS on {det.device}")
    cap.release()


if __name__ == "__main__":
    main()
