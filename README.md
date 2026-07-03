# Grocery Detection — Raspberry Pi 5 + Hailo-8 AI HAT+ → Laptop

Real-time grocery object detection. The Pi captures from the Raspberry Pi AI Camera,
runs YOLOv8 on the Hailo-8 AI HAT+, filters to grocery items (produce, prepared food,
containers, utensils), and streams the annotated video + detections to a native
desktop app on your laptop.

```
[Pi AI Camera] → Picamera2 → Hailo-8 YOLOv8 → grocery filter → annotate/JPEG
      → TCP (server.py :8765) → laptop client.py (PyQt6): video + item table
```

## Detected classes (off-the-shelf COCO)
- **Produce:** banana, apple, orange, broccoli, carrot
- **Prepared food:** sandwich, hot dog, pizza, donut, cake
- **Containers:** bottle, wine glass, cup, bowl
- **Utensils:** fork, knife, spoon

> COCO only covers these grocery classes. See **Phase 2** for full produce coverage.

## Run

### 1. Deploy to the Pi
```bash
./deploy.sh          # sync pi/ and link the Hailo model files
```

### 2. Start the server on the Pi
```bash
ssh admin@192.168.68.62
cd ~/grocery-detect
python3 server.py                 # AI Camera + yolov8s, listens on :8765
# options: --hef models/yolov8m_h10.hef --thresh 0.35 --preview --source webcam
```

### 3. Run the viewer on the laptop
```bash
cd laptop
python3 -m venv venv          # first time only (Homebrew Python needs a venv)
source venv/bin/activate
pip install -r requirements.txt
python3 client.py --host 192.168.68.62
```
On later runs just `source venv/bin/activate` before launching the client.

Point the camera at fruit / bottles / packaged food — boxes appear on the video and
the table fills with item names, counts, and confidence. Expect ~15–30 FPS on yolov8s.

## Multi-camera: select an item, track it across cameras

`laptop/app.py` shows N cameras side by side. Click an item in one camera and
the app tracks it there and **re-identifies it in the other camera** (class +
appearance embedding). The cameras are mounted with **overlapping views**, so
a real handoff is a *smooth transition* — co-visible in both, or reappearing
within `--handoff-window` seconds. Anything that merely looks the same but
appears without that smooth transition is tagged **NEW OBJ** instead of linked.

### Try it now — no Pi, no .hef (local mode runs `best.pt` on the laptop)
```bash
cd laptop
python3 -m venv venv                 # first time only
source venv/bin/activate && pip install -r requirements.txt
python3 make_demo.py --images <dir of item photos> --out demo   # two demo cams
python3 app.py --source demo/camA.mp4 --source demo/camB.mp4
# or mix a real webcam in:  --source webcam:0 --source demo/camB.mp4
```
Click the gliding item: cyan ★ SELECTED; the other pane shows a cyan MATCH
ring during the handoff, and the "teleporting" clone at the end gets the
orange NEW OBJ tag.

### Production — two Pi 5 + Hailo units, same app
Compile the custom model to `.hef` first (`training/HAILO.md`), then on each Pi:
```bash
python3 server.py --hef models/grocery_yolov8n.hef --labels grocery \
    --no-annotate --camera-id 0 --camera-name front     # 1 on the other Pi
```
and on the laptop:
```bash
python3 app.py --source tcp://<pi-A>:8765 --name front \
               --source tcp://<pi-B>:8765 --name side
```

## Files
| Path | Role |
|------|------|
| `pi/server.py` | capture → Hailo detect → filter → annotate → TCP stream |
| `pi/detector.py` | Hailo-8 YOLOv8 inference (reused from `depth_detect`) |
| `pi/camera.py` | Picamera2 (AI Camera) + webcam fallback |
| `pi/grocery.py` | grocery class subset, filter, colors |
| `pi/coco_labels.py` | 80 COCO labels |
| `pi/grocery_labels.py` | 43 custom grocery labels (index-aligned to `training/data.yaml`) |
| `pi/protocol.py` | length-prefixed TCP frame protocol (shared) |
| `laptop/client.py` | PyQt6 native viewer (single camera) |
| `laptop/app.py` | multi-camera viewer: click-to-select + cross-camera re-ID |
| `laptop/sources.py` … | multi-cam pipeline: sources / detect / tracker / embedding / reid / overlay |
| `training/HAILO.md` | compile `best.onnx` → `grocery_yolov8n.hef` walkthrough |

## Phase 2 — custom grocery model (broader coverage)
COCO misses most real grocery items (potato, tomato, onion, peppers, leafy greens,
cans, boxes, bagged goods). The `training/` module implements this — see
`training/README.md`. To go further:
1. **Dataset:** Freiburg Groceries, Fruits-360, or RPC checkout; or label your own
   with Roboflow.
2. **Train:** fine-tune `yolov8s`/`yolov8n` (Ultralytics) on the grocery classes.
3. **Compile to Hailo:** export ONNX (`export.py`) → compile with the Hailo
   Dataflow Compiler — full walkthrough in `training/HAILO.md` (calibration set,
   `hailomz compile … --classes 43`, NMS-by-class so `detector.py` needs no changes).
4. Drop the `.hef` in `pi/models/`, `./deploy.sh`, and run with
   `--hef models/grocery_yolov8n.hef --labels grocery` (labels come from
   `pi/grocery_labels.py`, index-aligned to `training/data.yaml`).
```
