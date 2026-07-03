# CLAUDE.md — laptop/ (PyQt6 viewer)

Native desktop client that runs **on the laptop** and displays the Pi's stream. See
root `CLAUDE.md` for project-wide conventions.

## Files

| File | Role |
|------|------|
| `client.py` | Legacy single-camera viewer (untouched). `StreamThread` (QThread) connects to the Pi, reads frames via `recv_frame`, decodes JPEG, emits `QImage` + meta. argparse: `--host`, `--port`. Auto-reconnects every 2s. |
| `app.py` | **Multi-camera viewer.** N panes, one `SourceThread` per `--source` (repeatable spec: `tcp://host:port`, `webcam:N`, or a video path). Click a box → SelectionState; the matcher re-IDs it in the other cameras. All selection/match state mutates only on the GUI thread. |
| `sources.py` | `Detection`/`FramePacket` dataclasses + `DetectionSource`s: `PiStreamSource` (TCP, production) and `LocalYoloSource` (VideoCapture + local YOLO, testing); `make_source(spec, …)`. Frames are always CLEAN (never annotated). |
| `detect.py` | `YoloDetector` wrapping an ultralytics `.pt` (lazy torch import; default `training/runs/grocery/weights/best.pt`). Headless `main()` for sanity checks. |
| `tracker.py` | `IouTracker`: per-camera SORT-style IoU tracker (label-gated, constant-velocity, lap.lapjv w/ greedy fallback, min_hits; max_age is SECONDS so ghost tracks die frame-rate-independently). Stable `track_id`s. |
| `embedding.py` | Re-ID appearance embeddings from clean crops: `DeepEmbedder` (torchvision mobilenet_v3_small 576-d, batched, thread-safe) and torch-free `HistEmbedder`; `cosine_distance`. |
| `reid.py` | `CrossCameraMatcher`: single-query matching of the selected object with three gates — class, appearance (auto-calibrated tau + runner-up margin), and **smooth-transition continuity** (overlapping cameras: co-visible or reappearing within `handoff_window`; otherwise tagged `different` = NEW OBJ). Re-anchors when the object changes cameras. |
| `overlay.py` | Drawing: per-class colored boxes + selection/match/new-object rings. Always draw on a COPY (embeddings come from clean pixels first). |
| `make_demo.py` | Renders two overlapping demo "camera" videos from item photos (smooth A→B glide + a teleporting clone that must be tagged NEW OBJ). |
| `protocol.py` | Length-prefixed TCP frame protocol. **Duplicate of `pi/protocol.py` — keep in sync.** |
| `requirements.txt` | `PyQt6`, `opencv-python`, `numpy` + local-mode/re-ID extras: `ultralytics`, `torch`, `torchvision`, `lap` (pinned to verified cp314 arm64 wheels). |

## Setup / run

```bash
cd laptop
python3 -m venv venv && source venv/bin/activate   # Homebrew Python needs a venv
pip install -r requirements.txt
python3 client.py --host <pi-ip>
```

## Editing notes

- All socket I/O and inference happens in QThreads (`StreamThread` in client.py,
  `SourceThread` in app.py); never block the Qt main thread. UI updates go through
  signals; selection/match state mutates ONLY on the GUI thread.
- Incoming JSON meta shape (produced by `pi/server.py`):
  `{"fps", "camera_id", "camera_name", "annotated", "counts", "detections":
  [{"label","cls_id","score","box":[x1,y1,x2,y2],"center":[cx,cy]}]}` — the
  camera/cls fields are optional additive (older senders omit them).
  If you change these fields, update `pi/server.py` too.
- `QImage` is `.copy()`'d off the numpy buffer before emit to avoid use-after-free.
- Re-ID invariant: embeddings are computed from the CLEAN frame BEFORE any drawing
  (Pis should run `--no-annotate`); `crop_box`'s 8% inset is the backstop.
- One shared `DeepEmbedder` across all SourceThreads (internal lock); one
  `YoloDetector` per local source (ultralytics models are not share-safe).
- This dir is NOT deployed to the Pi (`deploy.sh` only syncs `pi/`), which is why
  `protocol.py` is duplicated rather than imported.
