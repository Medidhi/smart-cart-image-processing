# Compiling the custom grocery model to a Hailo-8 `.hef`

Turns `runs/grocery/weights/best.onnx` (YOLOv8n, **43 classes**, 640x640,
exported by `export.py` with `nms=False`) into `grocery_yolov8n.hef` that
`pi/detector.py` can run unchanged.

The Hailo Dataflow Compiler / Model Zoo runs on an **x86-64 Linux machine**
(Hailo's SDK — not macOS, not the Pi). Install the Hailo AI SW Suite or the
`hailo_model_zoo` + DFC pip packages from the Hailo Developer Zone, then:

## 1. Calibration set (on this machine)

```bash
cd training/hailo
python3 export_calib.py --from-dataset          # or --images <real frames dir>
```

~300 letterboxed 640x640 JPEGs land in `training/hailo/calib/`. Real frames
captured by the actual Pi cameras beat dataset images — swap them in when you
have some.

## 2. Compile (on the Hailo x86 SDK machine)

Copy `best.onnx` + `calib/` over, then:

```bash
hailomz compile yolov8n \
  --ckpt best.onnx \
  --hw-arch hailo8 \
  --calib-path calib/ \
  --classes 43 \
  --performance
```

- `yolov8n` selects the stock Model Zoo network config; `--classes 43`
  adapts the heads and the on-chip NMS to the custom class count.
- **The output MUST stay HAILO_NMS_BY_CLASS** (the stock yolov8n config's
  default postprocess): `pi/detector.py` decodes `results[out][0]` as a
  length-43 list of `(n,5)` arrays `[ymin,xmin,ymax,xmax,score]`. If you
  customize the config yaml, keep the NMS postprocess section intact.
- If your Model Zoo version wants a yaml instead of flags, copy its stock
  `cfg/networks/yolov8n.yaml`, change only `classes` and the ckpt path, and
  pass it with `--yaml`.

## 3. Deploy to the Pi(s)

```bash
cp grocery_yolov8n.hef <repo>/pi/models/     # gitignored; rsync carries it
./deploy.sh                                  # syncs pi/ incl. models/*.hef
ssh admin@<pi> 'cd ~/grocery-detect && python3 server.py \
    --hef models/grocery_yolov8n.hef --labels grocery --no-annotate \
    --camera-id 0 --camera-name front'
```

Second camera: same command on the other Pi with `--camera-id 1
--camera-name side`. Then on the laptop:

```bash
cd laptop && source venv/bin/activate
python3 app.py --source tcp://<pi-A>:8765 --name front \
               --source tcp://<pi-B>:8765 --name side
```

## Class-index invariant (do not skip)

`training/data.yaml names` → compiled `.hef` output order →
`pi/grocery_labels.py` → `pi/grocery.py` groups must stay **index-aligned**.
`pi/grocery_labels.py` is a verbatim copy of the 43 `data.yaml` names; if you
retrain with different classes, regenerate BOTH together and recompile the
`.hef`. A mismatch does not error — it silently mislabels every detection.
