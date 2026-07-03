"""
Frame + detection sources for the multi-camera viewer (app.py).

Each DetectionSource yields FramePacket objects: a CLEAN BGR frame (never
annotated — re-ID crops are cut from it), the detections for that frame, and
the camera identity. Everything downstream (tracker, embedder, matcher, UI)
is identical for every source type; only where the boxes come from differs.

- PiStreamSource:  production — consumes a pi/server.py TCP stream (run the
                   Pi with --no-annotate so the JPEG is clean).
- LocalYoloSource: local testing — cv2.VideoCapture (webcam index or video
                   file) + ultralytics YOLO on this machine (see detect.py),
                   so the full pipeline runs with no Pi and no .hef. Video
                   files are paced to their native FPS and loop forever.
- make_source():   factory parsing a --source spec:
                     tcp://HOST[:PORT]   -> PiStreamSource
                     webcam:N  (or "N")  -> LocalYoloSource on camera N
                     <path>              -> LocalYoloSource on a video file
"""
import socket
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from protocol import recv_frame

DEFAULT_PORT = 8765


@dataclass
class Detection:
    label: str
    cls_id: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def box(self):
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class FramePacket:
    camera_id: int
    camera_name: str
    frame_bgr: np.ndarray          # clean frame — never annotated
    detections: list
    fps: float
    seq: int
    ts: float


class PiStreamSource:
    """TCP client for pi/server.py. read() returns None while disconnected
    and reconnects with a ~2 s backoff WITHOUT blocking the caller for long
    (so the owning thread can re-check its stop flag). interrupt() unblocks
    a pending recv/connect for shutdown; it is the only method that is safe
    to call from another thread."""

    def __init__(self, host, port=DEFAULT_PORT, camera_id=0, camera_name=None):
        self.host, self.port = host, port
        self.camera_id = camera_id
        self.camera_name = camera_name or f"cam{camera_id}"
        self._sock = None
        self._seq = 0
        self._stop = threading.Event()
        self._retry_at = 0.0
        self._warned_annotated = False

    def _connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=3)
        sock.settimeout(5)          # a live server streams continuously
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def interrupt(self):
        """Thread-safe: wake a blocked read() so the owner can shut down."""
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def read(self):
        if self._stop.is_set():
            return None
        now = time.monotonic()
        if now < self._retry_at:                 # reconnect backoff
            time.sleep(min(0.1, self._retry_at - now))
            return None
        if self._sock is None:
            try:
                self._sock = self._connect()
            except OSError:
                self._retry_at = time.monotonic() + 2.0
                return None
        try:
            meta, jpeg = recv_frame(self._sock)
        except (OSError, ConnectionError):
            self.close()
            self._retry_at = time.monotonic() + 2.0
            return None
        if meta.get("annotated") and not self._warned_annotated:
            self._warned_annotated = True
            print(f"[source:{self.camera_name}] WARNING: server sends "
                  "ANNOTATED frames; run pi/server.py with --no-annotate so "
                  "re-ID crops come from clean pixels", flush=True)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        dets = []
        for d in meta.get("detections", []):
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            dets.append(Detection(
                label=d["label"], cls_id=int(d.get("cls_id", -1)),
                score=float(d["score"]), x1=x1, y1=y1, x2=x2, y2=y2))
        self._seq += 1
        return FramePacket(
            camera_id=int(meta.get("camera_id", self.camera_id)),
            camera_name=meta.get("camera_name", self.camera_name),
            frame_bgr=frame, detections=dets,
            fps=float(meta.get("fps", 0.0)), seq=self._seq, ts=time.time())

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class LocalYoloSource:
    """cv2.VideoCapture + local YOLO (detect.py). Video files are paced to
    their native FPS (so tracking behaves like a live camera) and loop.
    A dead capture (USB webcam dropout, failed rewind) is released and
    reopened with a ~2 s backoff. NOT thread-safe: read() and close() must
    run on the same (owning) thread."""

    def __init__(self, cap_spec, detector, camera_id=0, camera_name=None,
                 loop=True):
        self.detector = detector
        self.camera_id = camera_id
        self.camera_name = camera_name or f"cam{camera_id}"
        self.cap_spec = cap_spec
        self.is_file = not isinstance(cap_spec, int)
        self.loop = loop and self.is_file
        self.cap = cv2.VideoCapture(cap_spec)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source: {cap_spec!r}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0.0
        self._interval = 1.0 / fps if fps and fps > 0 else 0.0
        self._next_t = time.monotonic()
        self._reopen_at = 0.0
        self._seq = 0
        self._fps = 0.0

    def read(self):
        if self._interval:                       # real-time pacing for files
            now = time.monotonic()
            if now < self._next_t:
                time.sleep(self._next_t - now)
                now = time.monotonic()
            self._next_t += self._interval
            if now - self._next_t > 1.0:         # fell far behind; resync
                self._next_t = now + self._interval

        ok, frame = self.cap.read()
        if (not ok or frame is None) and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            # dead capture (webcam unplugged, unseekable file): reopen with
            # backoff, mirroring PiStreamSource's reconnect contract
            now = time.monotonic()
            if now >= self._reopen_at:
                self._reopen_at = now + 2.0
                self.cap.release()
                self.cap = cv2.VideoCapture(self.cap_spec)
            return None

        t0 = time.time()
        dets = self.detector.detect(frame)
        inst = 1.0 / max(1e-3, time.time() - t0)
        self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._seq += 1
        return FramePacket(
            camera_id=self.camera_id, camera_name=self.camera_name,
            frame_bgr=frame, detections=dets,
            fps=round(self._fps, 1), seq=self._seq, ts=time.time())

    def close(self):
        self.cap.release()


def make_source(spec, *, camera_id, camera_name=None, weights=None,
                device="auto", conf=0.35, iou=0.5, imgsz=640):
    """Build a DetectionSource from a --source spec string (see module doc)."""
    spec = str(spec)
    if spec.startswith("tcp://"):
        rest = spec[len("tcp://"):]
        host, _, port = rest.partition(":")
        return PiStreamSource(host, int(port or DEFAULT_PORT),
                              camera_id, camera_name)
    if spec.startswith("webcam:"):
        cap_spec = int(spec.split(":", 1)[1])
    elif spec.isdigit():
        cap_spec = int(spec)
    else:
        cap_spec = spec
    from detect import YoloDetector    # lazy: pulls in ultralytics/torch
    detector = YoloDetector(weights, imgsz=imgsz, conf=conf, iou=iou,
                            device=device)
    return LocalYoloSource(cap_spec, detector, camera_id, camera_name)
