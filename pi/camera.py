"""
Camera sources. Each source's .read() returns a BGR uint8 frame (or None on EOF).

- Picamera2Source: the Raspberry Pi AI Camera (or any libcamera camera) via Picamera2.
                   Used purely as a frame source; inference runs on the Hailo HAT.
- WebcamSource:    any UVC webcam — for testing the pipeline off-Pi.
"""
import cv2


class Picamera2Source:
    def __init__(self, width=1280, height=720):
        from picamera2 import Picamera2  # imported lazily; only present on the Pi
        self.picam2 = Picamera2()
        cfg = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)})
        self.picam2.configure(cfg)
        self.picam2.start()

    def read(self):
        # Picamera2 "RGB888" main stream is delivered in BGR byte order for OpenCV.
        frame = self.picam2.capture_array()
        return frame if frame is not None else None

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass


class WebcamSource:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


def make_source(kind="picamera", index=0, width=1280, height=720):
    if kind == "picamera":
        return Picamera2Source(width, height)
    if kind == "webcam":
        return WebcamSource(index)
    raise ValueError(kind)
