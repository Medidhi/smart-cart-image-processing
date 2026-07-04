# CLAUDE.md — pi/ (edge capture + Hailo inference)

Code that runs **on the Raspberry Pi 5 + Hailo-8 AI HAT+**. Deployed to
`~/grocery-detect` by `../deploy.sh` (rsync of this dir only). See root `CLAUDE.md` for
project-wide conventions.

## Files

| File | Role |
|------|------|
| `server.py` | Entry point. Capture → detect → `filter_grocery` → annotate → JPEG → TCP broadcast to all connected clients. argparse: `--source {picamera,webcam}`, `--index` (CSI camera number 0/1, or /dev/videoN), `--hef`, `--thresh`, `--host/--port`, `--jpeg-quality`, `--preview`, `--camera-id/--camera-name` (multi-cam identity), `--labels {coco,grocery}`, `--no-annotate` (stream the CLEAN frame for laptop/app.py — it draws boxes itself and cuts re-ID crops from clean pixels), `--width/--height` (capture size), `--max-fps` (cap the loop; low-power fallback), `--shared-device` (share the Hailo across processes — see below). |
| `detector.py` | `HailoObjectDetector(hef, score_thresh, labels=None, shared=False)`: loads a `.hef`, letterboxes to 640×640, runs YOLOv8, decodes on-chip **NMS-by-class** output into `Detection` dataclasses (now with `cls_id`) in original-frame pixels. `labels` defaults to COCO; pass `GROCERY_NAMES` for the custom model. `shared=True` opens the Hailo via the multi-process `hailort` service with the round-robin scheduler (the scheduler owns activation, so `detect()` skips the manual `activate()`). Also `draw()`. |
| `camera.py` | Frame sources. `Picamera2Source(width, height, camera_num)` (Pi 5 has two CSI ports 0/1, lazy-imports picamera2) and `WebcamSource` (UVC, for off-Pi testing); `make_source(kind, index, ...)` — `index` = CSI camera number or /dev/videoN. `.read()` returns BGR uint8 or None. |
| `grocery.py` | Grocery classes for BOTH models: COCO subset + the custom 43 (all grocery); `is_grocery`, `filter_grocery`, `color_for`/`category_of` (stable per-category BGR colors, incl. fruit/dairy/vegetable). |
| `coco_labels.py` | `COCO_CLASSES`, 80 names index-aligned to the stock `.hef` NMS output. |
| `grocery_labels.py` | `GROCERY_NAMES`, 43 names index-aligned to `training/data.yaml` and the custom grocery `.hef` (see `training/HAILO.md`). Regenerate together with data.yaml. |
| `protocol.py` | Length-prefixed TCP frame protocol (`send_frame`/`recv_frame`). **Duplicate of `laptop/protocol.py` — keep in sync.** |

## Runtime environment (important)

- `hailo_platform` and `picamera2` are **system-provided on the Pi**; there is no
  requirements file here and they are NOT pip-installable off-Pi.
- Keep those imports lazy or module-top only where already done: `detector.py` imports
  `hailo_platform` at top (only ever imported on the Pi), `camera.py` imports
  `picamera2` lazily inside `Picamera2Source.__init__` so the module imports anywhere.
- `.hef` files live in `models/` on the Pi (gitignored, symlinked from
  `/usr/share/hailo-models` by `deploy.sh`). Default `models/yolov8s_h8.hef`.

## Two cameras on ONE Pi (shared Hailo)

- A Pi 5 has two CSI ports. Run **one `server.py` per camera** with distinct
  `--index` / `--port` / `--camera-id`. Both must pass `--shared-device`, and
  the `hailort` service must be running (`systemctl status hailort` — enabled by
  default) so the single Hailo is time-shared between the two processes.
  Without it the second process dies with `HAILO_OUT_OF_PHYSICAL_DEVICES`.
- Launch each in its **own** ssh session, detached: `setsid nohup python3 -u
  server.py … >log 2>&1 </dev/null &`. Do NOT put `pkill -f server.py` and a
  start in the same remote command — `pkill -f` matches the ssh command line's
  own text and kills the new process. Use the `[s]erver.py` bracket trick to
  stop: `pkill -f "[s]erver.py"`.
- **Power:** Pi 5 + Hailo + two cameras needs the official 27 W PSU + a
  USB-C-to-C cable. On a weak supply the board browns out and reboots when
  inference runs; the low-power fallback is `--max-fps 6 --width 640
  --height 360 --jpeg-quality 50` plus `usb_max_current_enable=1` in
  `/boot/firmware/config.txt`.

## Detection pipeline invariants

- Detector expects HxWx3 **BGR uint8**; it converts to RGB and letterboxes internally.
- Output decode assumes **HAILO_NMS_BY_CLASS**: `results[out][0]` is a length-80 list of
  `(n,5)` arrays `[ymin,xmin,ymax,xmax,score]` normalized to the 640 canvas. A custom
  model must compile to this same format so nothing here changes.
- The labels list passed to the detector must be index-aligned with the `.hef` class
  order: `COCO_CLASSES` for the stock models, `GROCERY_NAMES` for the custom grocery
  model (selected at runtime via `server.py --labels {coco,grocery}`). The detector
  warns once if the `.hef` class count and the labels list disagree (see root
  `CLAUDE.md` invariant and `training/HAILO.md`).

## Editing notes

- `server.py` broadcasts to a thread-safe `Broadcaster` set; a dead client is dropped on
  send failure. Accept loop runs in a daemon thread.
- Test the full pipeline off-Pi with `--source webcam` (still needs Hailo for detect;
  the detector itself only runs on the Pi).
