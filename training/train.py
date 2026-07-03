#!/usr/bin/env python3
"""
Fine-tune a YOLO model on the custom grocery dataset.

Reads training/config.yaml for defaults; any value can be overridden via CLI flags.
Switching architecture is a one-liner: change `model:` in config.yaml, or pass
`--model yolov8s`. Anything in MODELS below is valid.

    python3 train.py                          # use config.yaml as-is
    python3 train.py --model yolov8s          # switch nano -> small
    python3 train.py --epochs 50 --batch 8    # override hyperparameters
    python3 train.py --resume                 # continue the last interrupted run

Output (weights, plots, metrics) lands in runs/<name>/ .
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

# Friendly name -> pretrained checkpoint. Ultralytics downloads the .pt on first use.
# To adopt a newer family later, just add a row (e.g. "yolo11n": "yolo11n.pt") and set
# `model:` in config.yaml — no other code changes needed.
MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_weights(model_key):
    if model_key not in MODELS:
        raise SystemExit(
            f"Unknown model '{model_key}'. Choose one of: {', '.join(MODELS)}\n"
            f"(or add it to MODELS in {__file__})"
        )
    return MODELS[model_key]


def configure_datasets_dir():
    """Point Ultralytics at training/ so a relative `path:` in data.yaml resolves here.

    Ultralytics resolves a relative dataset `path` against its global `datasets_dir`
    setting, NOT against data.yaml's location. Without this, `path: datasets/grocery`
    becomes `<datasets_dir>/datasets/grocery` (a wrong, doubled path). Call this before
    training/validation. Also used by the notebook.
    """
    from ultralytics import settings
    settings.update({"datasets_dir": str(HERE)})


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"),
                    help="training profile (default: training/config.yaml)")
    ap.add_argument("--model", choices=list(MODELS),
                    help="architecture to fine-tune (overrides config)")
    ap.add_argument("--data", help="dataset spec yaml (overrides config)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", help='"" auto, "0" GPU, "cpu", "mps"')
    ap.add_argument("--name", help="run name under project dir")
    ap.add_argument("--resume", action="store_true",
                    help="resume the most recent interrupted run")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides win over config.yaml.
    for key in ("model", "data", "epochs", "imgsz", "batch", "device", "name"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    model_key = cfg.get("model", "yolov8n")
    weights = resolve_weights(model_key)

    configure_datasets_dir()

    # Make relative paths (data, project) resolve against training/ regardless of cwd.
    data_path = (HERE / cfg["data"]).resolve()
    project_path = (HERE / cfg.get("project", "runs")).resolve()

    print(f"[train] model={model_key} ({weights})")
    print(f"[train] data={data_path}")
    print(f"[train] epochs={cfg.get('epochs')} imgsz={cfg.get('imgsz')} "
          f"batch={cfg.get('batch')} device={cfg.get('device') or 'auto'}")

    model = YOLO(weights)
    results = model.train(
        data=str(data_path),
        epochs=cfg.get("epochs", 100),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        patience=cfg.get("patience", 25),
        seed=cfg.get("seed", 0),
        device=cfg.get("device", "") or None,
        workers=cfg.get("workers", 8),
        project=str(project_path),
        name=cfg.get("name", "grocery"),
        resume=args.resume,
        exist_ok=False,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print("\n[train] done.")
    print(f"[train] best weights: {best}")
    print(f"[train] metrics/plots: {save_dir}")
    print(f"[train] next: python3 export.py --weights {best}")


if __name__ == "__main__":
    main()
