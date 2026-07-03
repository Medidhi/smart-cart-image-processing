"""
Drawing helpers for the multi-camera viewer. All drawing happens on a COPY of
the clean frame — appearance crops are taken before any of this runs.

- color_for(label): stable per-class BGR color (hashed hue, same every run).
- draw_tracks():    class-colored box + "label #id score" banner per track.
- draw_ring():      emphasis ring + banner for selection / match / new-object.
"""
import hashlib

import cv2
import numpy as np

SELECT_COLOR = (255, 255, 0)     # cyan (BGR) — the clicked object
MATCH_COLOR = (255, 255, 0)      # cyan — same object seen in another camera
DIFFERENT_COLOR = (0, 140, 255)  # orange — looks alike, but no smooth handoff


def color_for(label):
    """Stable per-label BGR color via hashed hue."""
    hue = int(hashlib.md5(label.encode()).hexdigest()[:8], 16) % 180
    px = np.uint8([[[hue, 190, 255]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.box)
        color = color_for(t.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"{t.label} #{t.track_id} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, txt, (x1 + 1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_ring(frame, box, color, text):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    if y2 + th + 12 < frame.shape[0]:
        ty = y2 + th + 10
    else:
        ty = max(th + 4, y1 - 10)
    cv2.rectangle(frame, (x1 - 4, ty - th - 4), (x1 + tw + 2, ty + 4),
                  color, -1)
    cv2.putText(frame, text, (x1 - 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2, cv2.LINE_AA)
    return frame
