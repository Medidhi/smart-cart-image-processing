#!/usr/bin/env python3
"""
Grocery detection server — runs on the Raspberry Pi 5 + Hailo-8 AI HAT+.

Captures from the Pi AI Camera, runs YOLOv8 on the Hailo, filters to grocery
classes, annotates the frame, and streams (annotated JPEG + detection JSON) to
any connected laptop client over TCP.

Usage:
  python3 server.py                       # AI Camera, yolov8s, listen on 0.0.0.0:8765
  python3 server.py --source webcam       # UVC webcam instead (off-Pi testing)
  python3 server.py --hef models/yolov8m_h10.hef --thresh 0.35
  python3 server.py --preview             # also show a local window on the Pi

Multi-camera setup (one Pi per camera, laptop/app.py as the viewer):
  python3 server.py --hef models/grocery_yolov8n.hef --labels grocery \\
      --camera-id 0 --camera-name front --no-annotate
--no-annotate streams the CLEAN frame (the laptop draws boxes itself and cuts
re-ID appearance crops, which drawn boxes would contaminate).
"""
import argparse
import socket
import threading
import time
from collections import Counter

import cv2

from detector import HailoObjectDetector, draw
from camera import make_source
from grocery import filter_grocery, color_for
from protocol import send_frame


class Broadcaster:
    """Holds the set of connected client sockets; sends each frame to all of them."""
    def __init__(self):
        self._clients = set()
        self._lock = threading.Lock()

    def add(self, sock):
        with self._lock:
            self._clients.add(sock)

    def remove(self, sock):
        with self._lock:
            self._clients.discard(sock)
        try:
            sock.close()
        except Exception:
            pass

    def broadcast(self, meta, jpeg):
        with self._lock:
            clients = list(self._clients)
        for sock in clients:
            try:
                send_frame(sock, meta, jpeg)
            except Exception:
                self.remove(sock)

    def count(self):
        with self._lock:
            return len(self._clients)


def accept_loop(server_sock, bcast):
    while True:
        try:
            conn, addr = server_sock.accept()
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client connected: {addr}")
        bcast.add(conn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["picamera", "webcam"], default="picamera")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--hef", default="models/yolov8s_h8.hef")
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--camera-id", type=int, default=0,
                    help="stable id for this camera in a multi-camera setup")
    ap.add_argument("--camera-name", default=None,
                    help="human name for this camera (default camN)")
    ap.add_argument("--labels", choices=["coco", "grocery"], default="coco",
                    help="class-name list matching the .hef: coco (stock 80) "
                         "or grocery (custom 43, training/HAILO.md)")
    ap.add_argument("--no-annotate", action="store_true",
                    help="stream the clean frame; the laptop viewer draws "
                         "boxes and needs clean pixels for re-ID crops")
    args = ap.parse_args()
    camera_name = args.camera_name or f"cam{args.camera_id}"

    labels = None
    if args.labels == "grocery":
        from grocery_labels import GROCERY_NAMES
        labels = GROCERY_NAMES
    print(f"[server] loading model {args.hef} (labels={args.labels}) ...")
    det = HailoObjectDetector(args.hef, score_thresh=args.thresh,
                              labels=labels)
    src = make_source(args.source, index=args.index)
    print(f"[server] camera={args.source} ready")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)
    bcast = Broadcaster()
    threading.Thread(target=accept_loop, args=(server_sock, bcast), daemon=True).start()
    print(f"[server] listening on {args.host}:{args.port} — connect the laptop client")

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    fps = 0.0
    try:
        while True:
            frame = src.read()
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            dets = filter_grocery(det.detect(frame))
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, time.time() - t0))

            counts = Counter(d.label for d in dets)
            annotated = None
            if not args.no_annotate or args.preview:
                annotated = draw(frame.copy(), dets, color_for=color_for)
                cv2.putText(annotated, f"{fps:4.1f} FPS  {len(dets)} items",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

            # --no-annotate streams the CLEAN frame: the laptop viewer draws
            # boxes itself and cuts re-ID crops from unmodified pixels.
            out = frame if args.no_annotate else annotated
            ok, buf = cv2.imencode(".jpg", out, enc)
            if not ok:
                continue
            meta = {
                "fps": round(fps, 1),
                "camera_id": args.camera_id,
                "camera_name": camera_name,
                "annotated": not args.no_annotate,
                "counts": dict(counts),
                "detections": [
                    {"label": d.label, "cls_id": d.cls_id,
                     "score": round(d.score, 3),
                     "box": [d.x1, d.y1, d.x2, d.y2], "center": list(d.center)}
                    for d in dets
                ],
            }
            bcast.broadcast(meta, buf.tobytes())

            if args.preview:
                cv2.imshow("grocery-detect (Pi)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        server_sock.close()
        cv2.destroyAllWindows()
        print("[server] stopped")


if __name__ == "__main__":
    main()
