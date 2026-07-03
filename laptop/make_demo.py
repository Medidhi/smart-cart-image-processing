#!/usr/bin/env python3
"""
Render two overlapping "camera" videos from a folder of grocery item photos,
for testing app.py without cameras or a Pi.

A wide virtual scene is built; camA records the left crop and camB the right
crop with a shared central overlap band (like two fixed cameras aimed at one
shelf). One item glides across the whole scene and exits — a SMOOTH handoff
from camA to camB through the overlap. After a gap longer than the handoff
window, a CLONE of the same item pops into camB with no transition —
app.py should tag it NEW OBJ instead of linking it.

  python3 make_demo.py --images <dir of item .jpg> --out demo/
  python3 app.py --source demo/camA.mp4 --source demo/camB.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

SCENE_W, SCENE_H = 1920, 720
CAM_W = 1152                              # each camera sees 60% of the scene
CAM_A_X0 = 0
CAM_B_X0 = SCENE_W - CAM_W                # overlap = x in [768, 1152)


def load_tile(path, width):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    scale = width / img.shape[1]
    return cv2.resize(img, (width, max(1, int(img.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def paste(scene, tile, cx, cy):
    """Paste tile centered at (cx, cy), clipped to the scene."""
    th, tw = tile.shape[:2]
    x1, y1 = int(cx - tw / 2), int(cy - th / 2)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(SCENE_W, x1 + tw), min(SCENE_H, y1 + th)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    scene[sy1:sy2, sx1:sx2] = tile[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="dir of grocery item photos (.jpg)")
    ap.add_argument("--mover-image", default=None,
                    help="specific photo for the item that walks A->B "
                         "(pick one the model classifies stably; default: "
                         "first .jpg in --images)")
    ap.add_argument("--static-image", default=None,
                    help="specific photo for the static camA item "
                         "(default: third .jpg in --images)")
    ap.add_argument("--out", default="demo", help="output dir")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    paths = sorted(Path(args.images).glob("*.jpg"))
    if len(paths) < 3 and not (args.mover_image and args.static_image):
        raise SystemExit(f"need >=3 .jpg images in {args.images}, "
                         f"found {len(paths)}")
    # Big tiles (~40% of a camera's width): the model was trained with
    # full-frame weak labels, so small pasted items detect poorly.
    mover = load_tile(args.mover_image or paths[0], 440)   # walks A -> B
    static = load_tile(args.static_image or paths[2], 380)  # camA only

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wa = cv2.VideoWriter(str(out_dir / "camA.mp4"), fourcc, args.fps,
                         (CAM_W, SCENE_H))
    wb = cv2.VideoWriter(str(out_dir / "camB.mp4"), fourcc, args.fps,
                         (CAM_W, SCENE_H))

    n_frames = int(args.seconds * args.fps)
    t_move_end = 0.60                          # mover exits scene by 60%
    t_clone = 0.78                             # clone appears at 78% (> gap)
    for k in range(n_frames):
        t = k / max(1, n_frames - 1)
        scene = np.full((SCENE_H, SCENE_W, 3), 235, dtype=np.uint8)
        for y in (120, 360, 600):              # faint shelf lines
            cv2.line(scene, (0, y), (SCENE_W, y), (210, 210, 210), 2)
        paste(scene, static, 200, 360)         # camA-exclusive static item

        if t <= t_move_end:                    # smooth glide, exits right
            frac = t / t_move_end
            cx = 560 + frac * (SCENE_W + 280 - 560)   # starts camA-only
            paste(scene, mover, cx, 360)
        if t >= t_clone:                       # teleporting clone: NEW OBJ
            paste(scene, mover, 1500, 360)     # camB-exclusive, no handoff

        wa.write(scene[:, CAM_A_X0:CAM_A_X0 + CAM_W])
        wb.write(scene[:, CAM_B_X0:CAM_B_X0 + CAM_W])

    wa.release()
    wb.release()
    print(f"[demo] wrote {out_dir/'camA.mp4'} and {out_dir/'camB.mp4'} "
          f"({n_frames} frames @ {args.fps} fps, overlap 384px)")
    print("[demo] timeline: item glides A->B (smooth handoff), exits, then "
          "an identical clone appears in camB -> expect NEW OBJ tag")


if __name__ == "__main__":
    main()
