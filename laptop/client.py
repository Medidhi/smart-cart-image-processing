#!/usr/bin/env python3
"""
Grocery detection viewer — native desktop app for the laptop.

Connects to the Pi server, shows the live annotated video on the left and a
table of detected grocery items (class | count | max score) on the right, with
a header showing FPS and total item count.

Usage:
  pip install -r requirements.txt
  python3 client.py --host 192.168.68.62 --port 8765
"""
import argparse
import socket
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from protocol import recv_frame


class StreamThread(QThread):
    frame_ready = pyqtSignal(QImage, dict)
    status = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host, self.port = host, port
        self._running = True

    def run(self):
        while self._running:
            try:
                self.status.emit(f"Connecting to {self.host}:{self.port} …")
                sock = socket.create_connection((self.host, self.port), timeout=10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.status.emit(f"Connected to {self.host}:{self.port}")
                while self._running:
                    meta, jpeg = recv_frame(sock)
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w, _ = rgb.shape
                    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                    self.frame_ready.emit(img, meta)
            except Exception as e:
                self.status.emit(f"Disconnected ({e}). Retrying in 2s …")
                self.msleep(2000)

    def stop(self):
        self._running = False
        self.wait(1500)


class MainWindow(QMainWindow):
    def __init__(self, host, port):
        super().__init__()
        self.setWindowTitle("Grocery Detection — Pi 5 + AI HAT+")
        self.resize(1280, 740)

        self.video = QLabel("Waiting for video …")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(900, 600)
        self.video.setStyleSheet("background:#111; color:#888;")

        self.header = QLabel("—")
        self.header.setStyleSheet("font-size:16px; font-weight:bold; padding:6px;")

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Item", "Count", "Max score"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.status = QLabel("Starting …")
        self.status.setStyleSheet("color:#666; padding:4px;")

        right = QVBoxLayout()
        right.addWidget(self.header)
        right.addWidget(self.table)
        right.addWidget(self.status)
        right_w = QWidget(); right_w.setLayout(right); right_w.setMaximumWidth(340)

        root = QHBoxLayout()
        root.addWidget(self.video, stretch=1)
        root.addWidget(right_w)
        central = QWidget(); central.setLayout(root)
        self.setCentralWidget(central)

        self.thread = StreamThread(host, port)
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.status.connect(self.status.setText)
        self.thread.start()

    def on_frame(self, img: QImage, meta: dict):
        pix = QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video.setPixmap(pix)

        counts = meta.get("counts", {})
        total = sum(counts.values())
        self.header.setText(f"{meta.get('fps', 0):.1f} FPS   ·   {total} items   "
                            f"·   {len(counts)} types")

        # max score per label
        best = {}
        for d in meta.get("detections", []):
            best[d["label"]] = max(best.get(d["label"], 0), d["score"])

        rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.table.setRowCount(len(rows))
        for r, (label, n) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(label))
            self.table.setItem(r, 1, QTableWidgetItem(str(n)))
            self.table.setItem(r, 2, QTableWidgetItem(f"{best.get(label, 0):.2f}"))

    def closeEvent(self, e):
        self.thread.stop()
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.68.62")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args.host, args.port)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
