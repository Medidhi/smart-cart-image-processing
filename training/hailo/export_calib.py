#!/usr/bin/env python3
"""
Export a calibration image set for the Hailo Dataflow Compiler.

Quantization needs a few hundred REPRESENTATIVE images (what the cameras will
actually see). This dumps them as 640x640 letterboxed JPEGs — the same
preprocessing pi/detector.py applies at runtime.

Sources, pick one:
  python3 export_calib.py --from-dataset          # datasets/grocery train+val
  python3 export_calib.py --images ~/shelf_pics   # your own captured frames

datasets/grocery is gitignored; regenerate it first with
`python3 ../prepare_grocery_dataset.py` if it is missing. Best results come
from real frames captured by the actual Pi cameras in place.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-dataset", action="store_true",
                     help="sample from datasets/grocery images/{train,val}")
    src.add_argument("--images", help="directory of your own .jpg/.png frames")
    ap.add_argument("--out", default=str(HERE / "calib"))
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.from_dataset:
        roots = [TRAINING / "datasets/grocery/images/train",
                 TRAINING / "datasets/grocery/images/val"]
        if not any(r.is_dir() for r in roots):
            raise SystemExit(
                "datasets/grocery not found — run "
                "`python3 ../prepare_grocery_dataset.py` first, or use --images")
    else:
        roots = [Path(args.images).expanduser()]

    paths = [p for r in roots if r.is_dir()
             for p in r.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not paths:
        raise SystemExit(f"no images found under {[str(r) for r in roots]}")
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.count]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(out / f"calib_{n:04d}.jpg"),
                    letterbox(img, args.imgsz),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n += 1
    print(f"[calib] wrote {n} letterboxed {args.imgsz}x{args.imgsz} JPEGs "
          f"to {out}")
    print("[calib] next: see training/HAILO.md for the hailomz compile step")


if __name__ == "__main__":
    main()
