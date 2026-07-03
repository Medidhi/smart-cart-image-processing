# CLAUDE.md — training/ (custom YOLO model)

Fine-tunes a custom grocery YOLO model on a laptop/workstation and exports it toward a
Hailo `.hef` for `pi/`. Implements README Phase 2. See root `CLAUDE.md` for conventions
and the full step-by-step + checklist in `training/README.md`.

## Files

| File | Role |
|------|------|
| `train.py` | Entry point. Loads `config.yaml`, applies CLI overrides, resolves model via the `MODELS` registry, calls `configure_datasets_dir()`, runs `YOLO(...).train(...)`. Output → `runs/<name>/weights/best.pt`. |
| `prepare_grocery_dataset.py` | Downloads the GroceryStore dataset → converts to YOLO format under `datasets/grocery/` (full-frame weak boxes, coarse classes) and regenerates `data.yaml`. Quickstart data. |
| `train_grocery.ipynb` | Interactive notebook mirroring the CLI flow; reuses `config.yaml`/`data.yaml`/`MODELS`. |
| `export.py` | `YOLO(...).export(format="onnx", opset=11, imgsz=640, nms=False)` → `best.onnx`, ready for the Hailo DFC (compiled separately on Hailo's x86 SDK). |
| `config.yaml` | Training profile. **`model:` is the one-line architecture switch.** Hyperparams, device, output dir. |
| `data.yaml` | Ultralytics dataset spec: `path`/`train`/`val`/`test` + ordered `names`. |
| `requirements.txt` | `ultralytics`, `onnx`, `onnxslim` (own venv, separate from `pi/`/`laptop/`). |

## Switching models (the key requirement)

- Change `model:` in `config.yaml`, OR pass `--model yolov8s`.
- Valid keys live in the `MODELS` dict in `train.py`: `yolov8n/s/m/l/x`, `yolo11n/s/m`.
- To add a new architecture/family later, add ONE row to `MODELS` (name → `.pt`
  filename); Ultralytics auto-downloads the checkpoint. No other code changes.
- Default is `yolov8n` (smallest/fastest, smallest `.hef`); `yolov8s` = more accurate.

## Conventions specific to this module

- Paths in `config.yaml`/`data.yaml` are relative and resolved against **this directory**
  (`HERE = Path(__file__).parent` in `train.py`), so training works from any cwd.
- **Ultralytics gotcha:** it resolves a relative `path:` in `data.yaml` against its global
  `datasets_dir` setting, not the yaml's location — causing a doubled path
  (`datasets/datasets/grocery`) and `FileNotFoundError`. `configure_datasets_dir()` in
  `train.py` fixes this by setting `datasets_dir` to `HERE`; call it before any
  train/val/`check_det_dataset` (the notebook does this in the "Pick the model" cell).
- Unlike the rest of the repo (argparse-only), YAML config is used here because
  Ultralytics inherently needs `data.yaml`. Keep per-run knobs available as CLI flags too.
- Export settings are deliberately Hailo-friendly: opset 11, fixed 640, `dynamic=False`,
  `nms=False` (Hailo adds NMS-by-class on-chip — matches `pi/detector.py`).

## Critical invariant — class order

`data.yaml names` defines each `class_id` and is a contract: (1) it must stay fixed once
labeling starts (reordering corrupts every label file), and (2) after compilation it must
match the Pi's label file/order (`pi/coco_labels.py`) and `pi/grocery.py`. Change all
together. See root `CLAUDE.md`.

## Not in git

`venv/`, `runs/`, `datasets/`, `*.pt`, `*.onnx` (all gitignored — large/regenerable).
The placeholder `names` in `data.yaml` must be replaced with the real dataset classes
before training.
