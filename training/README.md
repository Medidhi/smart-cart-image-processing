# Training — custom grocery YOLO model

Self-contained module that fine-tunes a YOLO model on your own grocery dataset and
exports it toward a Hailo `.hef` for the Pi (README Phase 2). Runs on your
laptop/workstation, **not** on the Pi.

```
config.yaml  → train.py → runs/<name>/weights/best.pt → export.py → best.onnx → (Hailo DFC) → .hef → pi/models/
```

## Switching models

Change **one line** in `config.yaml`:

```yaml
model: yolov8n     # -> yolov8s, yolov8m, yolo11n, ...  (keys defined in train.py MODELS)
```

or override per-run without editing anything:

```bash
python3 train.py --model yolov8s
```

`yolov8n` = smallest/fastest (best starting point, smallest `.hef`); `yolov8s` = more
accurate. To add a new family later (e.g. YOLO11), add a row to `MODELS` in `train.py`.

## Setup (once)

```bash
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run on Google Colab (A100)

The notebook `train_grocery.ipynb` is Colab-aware. You only need the **code** files on
Colab — the dataset is downloaded there by `prepare_grocery_dataset.py`.

1. Colab → **Runtime → Change runtime type → A100 GPU**.
2. Get the code onto Colab (pick one):
   - **Git (repeatable):** push `training/` to GitHub, then just run the notebook — its
     bootstrap cell clones the repo and `cd`s into `training/`. (Private repo → use a
     token URL or upload manually.)
   - **Manual upload:** File → Upload notebook → `train_grocery.ipynb`; then upload
     `train.py`, `config.yaml`, `export.py`, `prepare_grocery_dataset.py`,
     `requirements.txt` to the working dir. The bootstrap skips cloning if `train.py`
     is already present.
3. Run top-to-bottom. The data cell git-clones the GroceryStore dataset (~250 MB) and
   converts it each session; batch auto-scales to 64 on the A100; step 8 downloads
   `best.pt` + ONNX (Colab storage is ephemeral).

## Quickstart with a sample dataset

Don't have labeled data yet? Pull the [GroceryStore dataset](https://github.com/marcusklasson/GroceryStoreDataset)
and convert it to YOLO format in one step:

```bash
python3 prepare_grocery_dataset.py            # all splits, 43 coarse classes
python3 prepare_grocery_dataset.py --limit 40 # quick smaller sample
python3 prepare_grocery_dataset.py --fine     # 81 fine-grained classes instead
```

This clones the repo, writes `datasets/grocery/images|labels/{train,val,test}`, and
regenerates `data.yaml`. **Caveat:** that dataset has no bounding boxes, so each image
gets one full-frame box (weak label). It's great for exercising the pipeline end-to-end,
but for real detection accuracy relabel with actual boxes (e.g. Roboflow).

## Prepare your own dataset

Put data in YOLO detection format under `datasets/grocery/`:

```
datasets/grocery/
  images/train/*.jpg    labels/train/*.txt
  images/val/*.jpg      labels/val/*.txt
  images/test/*.jpg     labels/test/*.txt   (optional)
```

Each `labels/**/<img>.txt` has one box per line: `class_id x_center y_center w h`
(normalized 0–1). Then edit `names:` in `data.yaml` so it matches your classes.

Easiest sources: label your own in [Roboflow](https://roboflow.com) and "Export →
YOLOv8", or start from Freiburg Groceries / Fruits-360 / RPC.

> **Class order is a contract.** The order in `data.yaml` defines each `class_id` and
> must stay fixed once labeling starts — reordering silently corrupts every label. It
> must also match the label set used on the Pi after compilation (`pi/grocery.py`,
> `pi/coco_labels.py`).

## Train

```bash
python3 train.py                         # uses config.yaml
python3 train.py --epochs 50 --batch 8   # override anything
python3 train.py --device mps            # Apple Silicon;  "0" = CUDA GPU,  "cpu"
python3 train.py --resume                # continue last interrupted run
```

Outputs go to `runs/<name>/`: `weights/best.pt`, `weights/last.pt`, `results.png`,
`confusion_matrix.png`, PR/F1 curves, and val prediction previews.

## Export (toward Hailo)

```bash
python3 export.py --weights runs/grocery/weights/best.pt
```

Produces `best.onnx` (opset 11, fixed 640, NMS off — Hailo adds NMS-by-class on-chip so
`pi/detector.py` needs no changes). Compile it to `.hef` with the Hailo Dataflow
Compiler on Hailo's x86 SDK (`hailomz compile ... --hw-arch hailo8`), then:

```bash
cp your_model.hef ../pi/models/
# update pi/grocery.py class set + labels to match data.yaml, then on the Pi:
python3 server.py --hef models/your_model.hef
```

## What to check — before, during, after

**Before training**
- `data.yaml` `names` matches your actual classes, in the intended fixed order.
- Sanity-check a few labels visually — the #1 cause of bad models is mislabeled or
  misnormalized boxes. `python3 -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('data.yaml')"`
- Train/val split is a real split (no leakage of the same photo into both).
- Class balance isn't wildly skewed; each class has enough (≳100+) examples.
- GPU is actually being used: `python3 -c "import torch; print(torch.cuda.is_available())"`
  (or expect slow CPU training). On Apple Silicon use `--device mps`.

**During training** (watch the console table / `results.png`)
- Box/cls/dfl losses trend **down** and don't diverge (NaN = lower LR or batch).
- `val` metrics `mAP50` and `mAP50-95` trend **up**; if train loss keeps falling but
  val mAP stalls/drops → overfitting (add data/augmentation, lower epochs, `patience`
  will early-stop).
- No CUDA OOM — if it happens, lower `--batch` (16 → 8 → 4) or `--imgsz`.

**After training**
- Check `runs/<name>/`: `results.png` (curves), `confusion_matrix.png` (which classes
  get confused), `val_batch*_pred.jpg` (predictions on real images).
- Target rough guide: `mAP50 ≥ 0.7` is usable; below that, revisit data quality/quantity.
- Run inference on held-out images: `yolo predict model=runs/grocery/weights/best.pt source=some/img.jpg`
- If a class is weak, add more/harder examples for it rather than just training longer.

**After export / on the Pi**
- ONNX input is `1x3x640x640`, opset 11, static shape (Hailo parser requirement).
- Class count + order in the compiled model matches `data.yaml` and the Pi's label file.
- Post-compile accuracy drops slightly (int8 quantization) — use a representative
  calibration image set during DFC compile to minimize it.
- Confirm live FPS and detections on the Pi with `server.py --preview`.
```
