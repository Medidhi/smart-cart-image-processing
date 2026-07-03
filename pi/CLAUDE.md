# CLAUDE.md — pi/ (edge capture + Hailo inference)

Code that runs **on the Raspberry Pi 5 + Hailo-8 AI HAT+**. Deployed to
`~/grocery-detect` by `../deploy.sh` (rsync of this dir only). See root `CLAUDE.md` for
project-wide conventions.

## Files

| File | Role |
|------|------|
| `server.py` | Entry point. Capture → detect → `filter_grocery` → annotate → JPEG → TCP broadcast to all connected clients. argparse: `--source {picamera,webcam}`, `--hef`, `--thresh`, `--host/--port`, `--jpeg-quality`, `--preview`. |
| `detector.py` | `HailoObjectDetector`: loads a `.hef`, letterboxes to 640×640, runs YOLOv8, decodes on-chip **NMS-by-class** output into `Detection` dataclasses in original-frame pixels. Also `draw()`. |
| `camera.py` | Frame sources. `Picamera2Source` (AI Camera, lazy-imports picamera2) and `WebcamSource` (UVC, for off-Pi testing); `make_source(kind, ...)`. `.read()` returns BGR uint8 or None. |
| `grocery.py` | Grocery subset of COCO + `is_grocery`, `filter_grocery`, `color_for`/`category_of` (stable per-category BGR colors). |
| `coco_labels.py` | `COCO_CLASSES`, 80 names index-aligned to the `.hef` NMS output. |
| `protocol.py` | Length-prefixed TCP frame protocol (`send_frame`/`recv_frame`). **Duplicate of `laptop/protocol.py` — keep in sync.** |

## Runtime environment (important)

- `hailo_platform` and `picamera2` are **system-provided on the Pi**; there is no
  requirements file here and they are NOT pip-installable off-Pi.
- Keep those imports lazy or module-top only where already done: `detector.py` imports
  `hailo_platform` at top (only ever imported on the Pi), `camera.py` imports
  `picamera2` lazily inside `Picamera2Source.__init__` so the module imports anywhere.
- `.hef` files live in `models/` on the Pi (gitignored, symlinked from
  `/usr/share/hailo-models` by `deploy.sh`). Default `models/yolov8s_h8.hef`.

## Detection pipeline invariants

- Detector expects HxWx3 **BGR uint8**; it converts to RGB and letterboxes internally.
- Output decode assumes **HAILO_NMS_BY_CLASS**: `results[out][0]` is a length-80 list of
  `(n,5)` arrays `[ymin,xmin,ymax,xmax,score]` normalized to the 640 canvas. A custom
  model must compile to this same format so nothing here changes.
- `COCO_CLASSES` order == the `.hef` class order. For a custom model, swap this labels
  file and `grocery.py`'s class set together (see root `CLAUDE.md` invariant).

## Editing notes

- `server.py` broadcasts to a thread-safe `Broadcaster` set; a dead client is dropped on
  send failure. Accept loop runs in a daemon thread.
- Test the full pipeline off-Pi with `--source webcam` (still needs Hailo for detect;
  the detector itself only runs on the Pi).
