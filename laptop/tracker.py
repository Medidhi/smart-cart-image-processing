"""
Per-camera multi-object tracker — stable integer IDs frame to frame.

Hand-written SORT-style IoU tracker (no Kalman, no new deps) so it runs
identically on boxes from a Pi stream and from local YOLO. Constant-velocity
prediction, class-gated IoU cost, optimal assignment via lap.lapjv when
available (greedy fallback), tentative births (min_hits kills one-frame
flicker) and aged deaths. max_age is in SECONDS (wall clock), not frames:
frame-based aging let ghost tracks coast for wildly different real durations
depending on loop speed, long enough to swallow a NEW object appearing
nearby (stealing its identity).

One IouTracker per camera; call update() from a single thread.
"""
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import lap as _lap              # optional extra; resolved ONCE — a failed
except ImportError:                 # per-frame import would rescan sys.path
    _lap = None

_BIG = 1e6


@dataclass
class Track:
    track_id: int
    cls_id: int
    label: str
    box: list                     # [x1, y1, x2, y2] floats, frame pixels
    score: float
    vel: tuple = (0.0, 0.0)
    embedding: np.ndarray = None  # unit vector; EMA-maintained by the app
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    born_ts: float = field(default_factory=time.time)
    last_update_ts: float = field(default_factory=time.time)

    @property
    def center(self):
        return (int((self.box[0] + self.box[2]) / 2),
                int((self.box[1] + self.box[3]) / 2))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _assign(cost, max_cost):
    """Min-cost 1:1 assignment; returns [(row, col), ...] with cost < max_cost.
    Optimal via lap.lapjv when installed, greedy otherwise."""
    if _lap is not None:
        _, x, _ = _lap.lapjv(np.ascontiguousarray(cost), extend_cost=True,
                             cost_limit=max_cost)
        return [(i, int(j)) for i, j in enumerate(x) if j >= 0]
    pairs, used_r, used_c = [], set(), set()
    flat = np.dstack(np.unravel_index(
        np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in flat:
        if cost[r, c] >= max_cost:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return pairs


class IouTracker:
    def __init__(self, iou_min=0.3, max_age=1.5, min_hits=3):
        """max_age: SECONDS a lost track survives before deletion."""
        self.iou_min = iou_min
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """Associate detections with tracks; return the confirmed tracks that
        are visible this frame."""
        for t in self.tracks:                    # constant-velocity predict
            t.box = [t.box[0] + t.vel[0], t.box[1] + t.vel[1],
                     t.box[2] + t.vel[0], t.box[3] + t.vel[1]]

        n_prev = len(self.tracks)
        matches = []
        if self.tracks and detections:
            cost = np.full((n_prev, len(detections)), _BIG)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    # gate on label, not cls_id: an item never changes class,
                    # and labels stay valid when an old server omits cls_id
                    if t.label != d.label:
                        continue
                    v = iou(t.box, d.box)
                    if v >= self.iou_min:
                        cost[i, j] = 1.0 - v
            matches = _assign(cost, max_cost=1.0 - self.iou_min + 1e-6)

        matched_t = {i for i, _ in matches}
        matched_d = {j for _, j in matches}

        for i, j in matches:
            t, d = self.tracks[i], detections[j]
            ocx = (t.box[0] + t.box[2]) / 2
            ocy = (t.box[1] + t.box[3]) / 2
            t.box = [float(d.x1), float(d.y1), float(d.x2), float(d.y2)]
            ncx = (t.box[0] + t.box[2]) / 2
            ncy = (t.box[1] + t.box[3]) / 2
            # (ocx,ocy) is the PREDICTED center, so ncx-ocx is only the
            # residual; the true frame-to-frame motion adds the prediction
            t.vel = (0.6 * t.vel[0] + 0.4 * (ncx - ocx + t.vel[0]),
                     0.6 * t.vel[1] + 0.4 * (ncy - ocy + t.vel[1]))
            t.score = d.score
            t.hits += 1
            t.time_since_update = 0
            t.last_update_ts = time.time()
            if t.hits >= self.min_hits:
                t.confirmed = True

        for i in range(n_prev):
            if i not in matched_t:
                self.tracks[i].time_since_update += 1

        for j, d in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(Track(
                    track_id=self._next_id, cls_id=d.cls_id, label=d.label,
                    box=[float(d.x1), float(d.y1), float(d.x2), float(d.y2)],
                    score=d.score, confirmed=self.min_hits <= 1))
                self._next_id += 1

        now = time.time()
        self.tracks = [t for t in self.tracks
                       if now - t.last_update_ts <= self.max_age]
        return [t for t in self.tracks
                if t.confirmed and t.time_since_update == 0]
