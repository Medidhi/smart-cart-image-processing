# CLAUDE.md — smart-cart-image-processing

Guidance for AI agents working in this repo. Read this first, then the `CLAUDE.md`
in whichever subdirectory you're editing.

## What this project is

Real-time grocery object detection. A Raspberry Pi 5 + Hailo-8 AI HAT+ captures from
the AI Camera, runs YOLOv8 on the Hailo, filters to grocery classes, annotates frames,
and streams them over TCP to a native PyQt6 viewer on a laptop.

```
[Pi AI Camera] → Picamera2 → Hailo-8 YOLOv8 → grocery filter → annotate/JPEG
      → TCP (pi/server.py :8765) → laptop/client.py (PyQt6): video + item table
```

## Three independent areas (each stands alone, own deps)

| Dir         | Runs on            | Role | Details |
|-------------|--------------------|------|---------|
| `pi/`       | Raspberry Pi 5     | capture → Hailo detect → filter → annotate → TCP stream | `pi/CLAUDE.md` |
| `laptop/`   | your laptop        | PyQt6 viewer that receives the stream | `laptop/CLAUDE.md` |
| `training/` | laptop/workstation | fine-tune a custom YOLO model → export toward `.hef` | `training/CLAUDE.md` |

`deploy.sh` rsyncs **only** `pi/` to `~/grocery-detect` on the Pi and symlinks the
Hailo `.hef` models from `/usr/share/hailo-models`.

## Conventions (follow these when adding code)

- **Flat layout.** No `src/`, no packages, no `__init__.py`. Modules are sibling scripts
  imported by bare name (`from detector import ...`), relying on the script's own dir
  being on `sys.path`.
- Every module opens with a triple-quoted docstring stating its role.
- Runnable scripts use `#!/usr/bin/env python3` + a `main()` under `if __name__ == "__main__"`.
- Script knobs go through `argparse`, not config files (exception: `training/` needs
  YAML because Ultralytics requires it).
- Short lowercase module names; `@dataclass` for records; factory funcs (`make_source`).
- Lazy-import platform-specific heavy deps inside functions/`__init__` (e.g. Picamera2,
  hailo_platform) so modules still import off-device.

## Cross-cutting invariant — class index alignment

The class list is a contract across the whole pipeline and must stay index-aligned:
`training/data.yaml names` → compiled `.hef` output order → `pi/coco_labels.py` →
`pi/grocery.py` filter set. Reordering any one of these silently breaks detections.
If you change classes, update all of them together.

## What is NOT in git

`__pycache__`, venvs, `pi/models/*.hef`, and all `training/` artifacts (`runs/`,
`datasets/`, `*.pt`, `*.onnx`). Don't try to read model/dataset binaries — they're
generated, not committed.

## Gotchas

- `protocol.py` is intentionally **duplicated verbatim** in `pi/` and `laptop/` because
  `deploy.sh` only syncs `pi/`. Edit BOTH copies if you change the wire format.
- The Pi's deps (`hailo_platform`, `picamera2`) are system-provided; there is no
  `pi/requirements.txt`. Don't add hailo/picamera imports at module top level.
