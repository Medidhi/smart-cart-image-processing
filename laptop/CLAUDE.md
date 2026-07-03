# CLAUDE.md — laptop/ (PyQt6 viewer)

Native desktop client that runs **on the laptop** and displays the Pi's stream. See
root `CLAUDE.md` for project-wide conventions.

## Files

| File | Role |
|------|------|
| `client.py` | Entry point. `StreamThread` (QThread) connects to the Pi, reads frames via `recv_frame`, decodes JPEG, emits `QImage` + meta. `MainWindow` shows video (left) + item table (class \| count \| max score) and an FPS/count header (right). argparse: `--host`, `--port` (default 8765). Auto-reconnects every 2s on disconnect. |
| `protocol.py` | Length-prefixed TCP frame protocol. **Duplicate of `pi/protocol.py` — keep in sync.** |
| `requirements.txt` | `PyQt6`, `opencv-python`, `numpy`. |

## Setup / run

```bash
cd laptop
python3 -m venv venv && source venv/bin/activate   # Homebrew Python needs a venv
pip install -r requirements.txt
python3 client.py --host <pi-ip>
```

## Editing notes

- All socket I/O happens in `StreamThread.run`; never block the Qt main thread. UI
  updates go through the `frame_ready` / `status` signals.
- Incoming JSON meta shape (produced by `pi/server.py`):
  `{"fps": float, "counts": {label: n}, "detections": [{"label","score","box":[x1,y1,x2,y2],"center":[cx,cy]}]}`.
  If you change these fields, update `pi/server.py` too.
- `QImage` is `.copy()`'d off the numpy buffer before emit to avoid use-after-free.
- This dir is NOT deployed to the Pi (`deploy.sh` only syncs `pi/`), which is why
  `protocol.py` is duplicated rather than imported.
