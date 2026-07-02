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

## Files
| Path | Role |
|------|------|
| `pi/server.py` | capture → Hailo detect → filter → annotate → TCP stream |
| `pi/detector.py` | Hailo-8 YOLOv8 inference (reused from `depth_detect`) |
| `pi/camera.py` | Picamera2 (AI Camera) + webcam fallback |
| `pi/grocery.py` | grocery class subset, filter, colors |
| `pi/coco_labels.py` | 80 COCO labels |
| `pi/protocol.py` | length-prefixed TCP frame protocol (shared) |
| `laptop/client.py` | PyQt6 native viewer |

## Phase 2 — custom grocery model (broader coverage)
COCO misses most real grocery items (potato, tomato, onion, peppers, leafy greens,
cans, boxes, bagged goods). To go further:
1. **Dataset:** Freiburg Groceries, Fruits-360, or RPC checkout; or label your own
   with Roboflow.
2. **Train:** fine-tune `yolov8s`/`yolov8n` (Ultralytics) on the grocery classes.
3. **Compile to Hailo:** export ONNX → optimize/compile with the Hailo Dataflow
   Compiler (`hailomz compile` / DFC) to a `.hef`, using the same NMS-by-class output
   so `detector.py` needs no changes.
4. Drop the `.hef` in `pi/models/`, update `grocery.py`'s class set + `coco_labels.py`
   (or a new labels file), and run with `--hef models/<your_model>.hef`.
```
