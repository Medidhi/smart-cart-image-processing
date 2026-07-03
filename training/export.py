#!/usr/bin/env python3
"""
Export trained weights to ONNX, ready for the Hailo Dataflow Compiler.

This is the bridge from training (this module) to the Pi's Hailo runtime. The ONNX it
produces is compiled to a .hef separately with `hailomz compile` / DFC (that toolchain
runs in Hailo's x86 SDK, not here), then dropped into pi/models/ — see README Phase 2.

    python3 export.py --weights runs/grocery/weights/best.pt
    python3 export.py --weights runs/grocery/weights/best.pt --imgsz 640 --opset 11

Notes:
- opset 11 and a fixed imgsz (no --dynamic) match what the Hailo parser expects.
- NMS is left OFF in the ONNX graph on purpose: the Hailo compiler adds on-chip
  NMS-by-class, which is the output format pi/detector.py already decodes.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="path to trained .pt (e.g. best.pt)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="export resolution (keep matching training, default 640)")
    ap.add_argument("--opset", type=int, default=11, help="ONNX opset (Hailo-friendly)")
    return ap.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    print(f"[export] loading {weights}")
    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
        simplify=True,
        nms=False,
    )
    print(f"\n[export] ONNX written: {out}")
    print("[export] next (on the Hailo x86 SDK, not this machine):")
    print("         hailomz compile --ckpt <this.onnx> --hw-arch hailo8 \\")
    print("             --calib-path <images/> --classes <N> --yaml <cfg>")
    print("[export] then copy the resulting .hef into pi/models/ and run:")
    print("         python3 server.py --hef models/<your_model>.hef")


if __name__ == "__main__":
    main()
