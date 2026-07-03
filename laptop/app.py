#!/usr/bin/env python3
"""
Multi-camera grocery viewer — click an item in one camera; the app tracks it
there and re-identifies it in the other camera(s).

Two source types share one pipeline (sources.py): Pi TCP streams for
production, local YOLO on webcams/video files for testing without a Pi:

  # local test (best.pt on two overlapping demo videos — see make_demo.py):
  python3 app.py --source demo/camA.mp4 --source demo/camB.mp4
  # webcam + file mix:
  python3 app.py --source webcam:0 --source demo/camB.mp4
  # production (two Pis running server.py --no-annotate):
  python3 app.py --source tcp://192.168.68.62:8765 --name front \\
                 --source tcp://192.168.68.63:8765 --name side

Click a box to select (cyan * SELECTED). The matcher re-identifies it in the
other camera: a cyan MATCH ring after a smooth handoff through the camera
overlap; an orange NEW OBJ tag when something merely looks the same but
appeared without a smooth transition (see reid.py). Esc clears the selection.
"""
import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from embedding import make_embedder
from overlay import (DIFFERENT_COLOR, MATCH_COLOR, SELECT_COLOR, draw_ring,
                     draw_tracks)
from reid import CrossCameraMatcher, SelectionState
from sources import make_source
from tracker import IouTracker

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent
                      / "training" / "runs" / "grocery" / "weights" / "best.pt")


@dataclass
class TrackView:
    """Immutable snapshot of a Track, safe to hand across threads."""
    track_id: int
    cls_id: int
    label: str
    score: float
    box: list
    embedding: np.ndarray          # unit vector or None
    born_ts: float

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) // 2,
                (self.box[1] + self.box[3]) // 2)


@dataclass
class CameraState:
    camera_id: int
    camera_name: str
    frame_bgr: np.ndarray
    tracks: list
    fps: float
    ts: float


class SourceThread(QThread):
    """Owns one DetectionSource + its per-camera tracker; embeds confirmed
    tracks from the CLEAN frame, then emits an immutable CameraState."""
    state_ready = pyqtSignal(int, object)      # camera_id, CameraState
    status = pyqtSignal(int, str)

    def __init__(self, camera_id, camera_name, source, tracker, embedder):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = source
        self.tracker = tracker
        self.embedder = embedder
        self._running = True
        # backpressure: at most ONE un-consumed CameraState may sit in the
        # GUI event queue per camera; extra frames are dropped for display
        # (the tracker/embedder still see every frame)
        self._pending = threading.Event()

    def run(self):
        while self._running:
            try:
                pkt = self.source.read()
            except Exception as e:
                self.status.emit(self.camera_id, f"source error: {e}")
                self.msleep(500)
                continue
            if pkt is None:
                if self._running:
                    self.status.emit(self.camera_id, "waiting for frames …")
                    self.msleep(50)
                continue

            try:
                tracks = self.tracker.update(pkt.detections)
                embs = (self.embedder.embed_batch(pkt.frame_bgr,
                                                  [t.box for t in tracks])
                        if tracks else [])
                views = []
                for t, e in zip(tracks, embs):
                    if e is not None:          # EMA on the live Track
                        if t.embedding is None:
                            t.embedding = e
                        else:
                            v = 0.7 * t.embedding + 0.3 * e
                            t.embedding = (v / (np.linalg.norm(v) + 1e-9)
                                           ).astype(np.float32)
                    views.append(TrackView(
                        t.track_id, t.cls_id, t.label, float(t.score),
                        [int(v) for v in t.box],
                        None if t.embedding is None else t.embedding.copy(),
                        t.born_ts))
                state = CameraState(self.camera_id, self.camera_name,
                                    pkt.frame_bgr.copy(), views,
                                    pkt.fps, pkt.ts)
            except Exception as e:
                # keep the thread alive across transient torch/MPS/lap errors
                self.status.emit(self.camera_id, f"pipeline error: {e}")
                self.msleep(100)
                continue

            if self._pending.is_set():
                continue                       # GUI still busy; drop frame
            self._pending.set()
            self.state_ready.emit(self.camera_id, state)

        # the worker owns the source: close it only after the loop exits so
        # cv2/socket teardown never races a concurrent read (segfault risk)
        try:
            self.source.close()
        except Exception:
            pass

    def clear_pending(self):
        self._pending.clear()

    def stop(self):
        self._running = False
        # PiStreamSource.interrupt() is the only cross-thread-safe unblock;
        # local captures just finish their (time-bounded) read
        interrupt = getattr(self.source, "interrupt", None)
        if interrupt is not None:
            try:
                interrupt()
            except Exception:
                pass
        if not self.wait(7000):
            self.wait()   # reads are time-bounded; never destroy a live QThread


class VideoWidget(QLabel):
    """One camera pane. Maps clicks back to source-frame pixel coordinates
    (inverts the KeepAspectRatio scale + letterbox offset)."""
    clicked = pyqtSignal(int, int, int)        # camera_id, x, y (src pixels)

    def __init__(self, camera_id, title):
        super().__init__(f"{title}: waiting …")
        self.camera_id = camera_id
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background:#111; color:#888;")
        self._src_size = None
        self._pix_size = None

    def set_frame(self, img: QImage):
        self._src_size = (img.width(), img.height())
        pix = QPixmap.fromImage(img).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._pix_size = (pix.width(), pix.height())
        self.setPixmap(pix)

    def mousePressEvent(self, ev):
        if self._src_size is None or self._pix_size is None:
            return
        pw, ph = self._pix_size
        px = ev.position().x() - (self.width() - pw) / 2
        py = ev.position().y() - (self.height() - ph) / 2
        if not (0 <= px < pw and 0 <= py < ph):
            return
        sw, sh = self._src_size
        self.clicked.emit(self.camera_id,
                          int(px * sw / pw), int(py * sh / ph))


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Grocery multi-cam — select & cross-camera track")
        self.resize(1500, 800)

        self.matcher = CrossCameraMatcher(
            thresh=(None if args.reid_thresh == "auto"
                    else float(args.reid_thresh)),
            margin=args.reid_margin,
            confirm_frames=args.confirm_frames,
            release_frames=args.release_frames,
            handoff_window=args.handoff_window)
        self.selection = None
        self.states = {}
        self.results = {}
        self._last_table = 0.0

        print("[app] loading detector + re-ID weights "
              "(first run downloads to TORCH_HOME) …", flush=True)
        embedder = make_embedder(args.embedder, args.reid_backbone,
                                 args.device)

        grid = QGridLayout()
        self.videos = {}
        self.threads = []
        names = args.name or []
        for cam_id, spec in enumerate(args.source):
            name = names[cam_id] if cam_id < len(names) else f"cam{cam_id}"
            source = make_source(spec, camera_id=cam_id, camera_name=name,
                                 weights=args.weights, device=args.device,
                                 conf=args.conf, iou=args.iou,
                                 imgsz=args.imgsz)
            video = VideoWidget(cam_id, name)
            video.clicked.connect(self.on_click)
            self.videos[cam_id] = video
            grid.addWidget(video, 0, cam_id)
            tracker = IouTracker(iou_min=args.track_iou,
                                 max_age=args.track_max_age,
                                 min_hits=args.track_min_hits)
            thread = SourceThread(cam_id, name, source, tracker, embedder)
            thread.state_ready.connect(self.on_state)
            thread.status.connect(self.on_status)
            self.threads.append(thread)

        self.header = QLabel("—")
        self.header.setStyleSheet(
            "font-size:15px; font-weight:bold; padding:6px;")

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Cam", "Item", "ID", "Score", "State"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        clear_btn = QPushButton("Clear selection (Esc)")
        clear_btn.clicked.connect(self.clear_selection)

        self.status_lbl = QLabel("Starting …")
        self.status_lbl.setStyleSheet("color:#666; padding:4px;")

        right = QVBoxLayout()
        right.addWidget(self.header)
        right.addWidget(self.table)
        right.addWidget(clear_btn)
        right.addWidget(self.status_lbl)
        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setMaximumWidth(360)

        videos_w = QWidget()
        videos_w.setLayout(grid)
        root = QHBoxLayout()
        root.addWidget(videos_w, stretch=1)
        root.addWidget(right_w)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        for thread in self.threads:
            thread.start()

    # ── pipeline callbacks (GUI thread only) ────────────────────────────────
    def on_state(self, cam_id, state):
        self.threads[cam_id].clear_pending()   # allow the next frame through
        self.states[cam_id] = state
        self.matcher.observe_negatives(state)
        self.results = (self.matcher.update(self.selection, self.states)
                        if self.selection else {})

        frame = state.frame_bgr.copy()
        draw_tracks(frame, state.tracks)
        if self.selection:
            anchor = None
            if cam_id == self.selection.anchor_cam:
                anchor = next(
                    (t for t in state.tracks
                     if t.track_id == self.selection.anchor_track_id), None)
            if anchor is not None:
                draw_ring(frame, anchor.box, SELECT_COLOR,
                          f"* SELECTED {anchor.label}")
            else:
                # non-anchor cameras — and the anchor camera itself once its
                # track has died (the matcher then searches it too)
                r = self.results.get(cam_id)
                if r is not None and r.track_id >= 0:
                    t = next((t for t in state.tracks
                              if t.track_id == r.track_id), None)
                    if t is not None:
                        if r.status == "linked":
                            draw_ring(frame, t.box, MATCH_COLOR,
                                      f"MATCH d={r.distance:.2f}")
                        elif r.status == "different":
                            draw_ring(frame, t.box, DIFFERENT_COLOR,
                                      "NEW OBJ (no handoff)")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w,
                     QImage.Format.Format_RGB888).copy()
        self.videos[cam_id].set_frame(img)

        now = time.monotonic()
        if now - self._last_table > 0.25:
            self._last_table = now
            self._refresh_table()
            self._refresh_header()

    def on_status(self, cam_id, msg):
        self.status_lbl.setText(f"cam{cam_id}: {msg}")

    def on_click(self, cam_id, x, y):
        state = self.states.get(cam_id)
        if state is None:
            return
        containing = [t for t in state.tracks
                      if t.box[0] <= x <= t.box[2]
                      and t.box[1] <= y <= t.box[3]]
        if containing:
            target = min(containing,
                         key=lambda t: (t.box[2] - t.box[0])
                                       * (t.box[3] - t.box[1]))
        else:
            def d2(t):
                cx, cy = t.center
                return (cx - x) ** 2 + (cy - y) ** 2
            near = [t for t in state.tracks if d2(t) < 60 ** 2]
            target = min(near, key=d2) if near else None

        if target is None:
            self.clear_selection()
            return
        if target.embedding is None:
            self.status_lbl.setText(
                "that track has no appearance yet — click again in a moment")
            return
        self.matcher.reset()
        self.selection = SelectionState(
            anchor_cam=cam_id, anchor_track_id=target.track_id,
            cls_id=target.cls_id, label=target.label,
            template=target.embedding.copy(), last_seen_ts=state.ts)
        self.status_lbl.setText(
            f"selected {target.label} #{target.track_id} in "
            f"{state.camera_name}")

    def clear_selection(self):
        self.selection = None
        self.results = {}
        self.matcher.reset()
        self.status_lbl.setText("selection cleared")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self.clear_selection()
        else:
            super().keyPressEvent(ev)

    # ── side panel ──────────────────────────────────────────────────────────
    def _refresh_table(self):
        rows = []
        for cam_id in sorted(self.states):
            st = self.states[cam_id]
            for t in st.tracks:
                state_txt = ""
                if self.selection:
                    if (cam_id == self.selection.anchor_cam
                            and t.track_id == self.selection.anchor_track_id):
                        state_txt = "SELECTED"
                    else:
                        r = self.results.get(cam_id)
                        if r is not None and r.track_id == t.track_id:
                            state_txt = {
                                "linked": f"MATCH {r.distance:.2f}",
                                "pending": "match?",
                                "different": "NEW OBJ",
                            }.get(r.status, "")
                rows.append((st.camera_name, t.label, t.track_id,
                             t.score, state_txt))
        self.table.setRowCount(len(rows))
        for r, (cam, label, tid, score, st_txt) in enumerate(rows):
            for c, val in enumerate(
                    [cam, label, f"#{tid}", f"{score:.2f}", st_txt]):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))

    def _refresh_header(self):
        fps = "   ".join(
            f"{self.states[c].camera_name} {self.states[c].fps:.1f}fps"
            for c in sorted(self.states))
        if self.selection:
            tau = self.matcher.tau(self.selection.label)
            sel = (f"tracking {self.selection.label} "
                   f"(anchor cam{self.selection.anchor_cam}, tau {tau:.2f})")
        else:
            sel = "click an item to select"
        self.header.setText(f"{fps}\n{sel}")

    def closeEvent(self, ev):
        for thread in self.threads:
            thread.stop()
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="repeatable: tcp://HOST[:PORT] | webcam:N | "
                         "video file path")
    ap.add_argument("--name", action="append",
                    help="camera name, one per --source (default camN)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="ultralytics .pt for local sources")
    ap.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"],
                    default="auto")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--embedder", choices=["deep", "hist"], default="deep")
    ap.add_argument("--reid-backbone", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "mobilenet_v3_large",
                             "resnet50"])
    ap.add_argument("--reid-thresh", default="auto",
                    help='"auto" (self-calibrated) or a fixed cosine distance')
    ap.add_argument("--reid-margin", type=float, default=0.06)
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--track-min-hits", type=int, default=3)
    ap.add_argument("--track-max-age", type=float, default=1.5,
                    help="seconds a lost track survives before deletion")
    ap.add_argument("--confirm-frames", type=int, default=3)
    ap.add_argument("--release-frames", type=int, default=10)
    ap.add_argument("--handoff-window", type=float, default=2.0,
                    help="max seconds between leaving one camera and "
                         "appearing in the other for a SMOOTH handoff; "
                         "anything later is tagged a NEW OBJECT")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
