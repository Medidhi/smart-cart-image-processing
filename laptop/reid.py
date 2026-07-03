"""
Cross-camera re-identification of the ONE user-selected object.

Single-query matching: the user clicks a track in one camera (the anchor);
every other camera is searched for the same physical object. Gates, in order:

1. class gate      — candidate.label must equal the selection's (43 fine
                     grocery classes: an Apple can never match Milk). Gating
                     is by label string, not cls_id, so meta from an OLD
                     pi/server.py (no cls_id -> -1) cannot disable the gate.
2. appearance gate — cosine distance to the selection template < tau AND the
                     best candidate beats the runner-up by a margin (refuses
                     ambiguous ties between identical-looking items).
3. continuity gate — the cameras are mounted with OVERLAPPING views, so a
                     true handoff is SMOOTH: the object is co-visible in both
                     cameras, or appears in the target camera within
                     handoff_window seconds of leaving the anchor camera.
                     An appearance match WITHOUT a smooth transition is
                     tagged DIFFERENT OBJECT ('different') and never linked.

tau ('auto') calibrates from guaranteed negatives: two concurrent same-class
tracks in the SAME camera are necessarily different objects, so their
distances bound how close "different" can look. Hysteresis: confirm_frames
consecutive wins to link, release_frames consecutive misses to unlink.
When the anchor track dies while a link is live, the selection re-anchors to
the linked track — the object has walked into the other camera.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from embedding import cosine_distance


@dataclass
class SelectionState:
    anchor_cam: int
    anchor_track_id: int
    cls_id: int
    label: str
    template: np.ndarray            # unit vector; slow-EMA while anchor visible
    last_seen_ts: float
    alive: bool = True
    suspect: int = 0                # consecutive frames the anchor track no
                                    # longer looked like the selection


@dataclass
class MatchResult:
    status: str                     # 'linked' | 'pending' | 'different' | 'none'
    track_id: int = -1
    distance: float = 1.0
    tau: float = 0.0


@dataclass
class _CamMatch:
    track_id: int = -1
    streak: int = 0
    miss: int = 0
    linked: bool = False
    last_ts: float = -1.0           # snapshot dedup: hysteresis advances
    last_result: MatchResult = None  # once per NEW frame from this camera


class CrossCameraMatcher:
    def __init__(self, thresh=None, margin=0.06, confirm_frames=3,
                 release_frames=10, handoff_window=2.0, template_alpha=0.05):
        self.fixed_tau = thresh
        self.margin = margin
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self.handoff_window = handoff_window
        self.template_alpha = template_alpha
        self._neg = deque(maxlen=800)   # (cls_id, distance) hard negatives
        self._cam = {}                  # cam_id -> _CamMatch

    def reset(self):
        self._cam.clear()

    # ── tau auto-calibration ────────────────────────────────────────────────
    def observe_negatives(self, state):
        """Feed same-camera same-class concurrent track pairs (necessarily
        different physical objects) as hard negatives for the threshold.
        Keyed by label (robust to old servers that send no cls_id)."""
        by_label = {}
        for t in state.tracks:
            if t.embedding is not None:
                by_label.setdefault(t.label, []).append(t.embedding)
        for label, embs in by_label.items():
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    self._neg.append(
                        (label, cosine_distance(embs[i], embs[j])))

    def tau(self, label):
        if self.fixed_tau is not None:
            return self.fixed_tau
        ds = [d for lbl, d in self._neg if lbl == label]
        if len(ds) < 20:
            ds = [d for _, d in self._neg]
        if len(ds) < 20:
            return 0.30                 # conservative default, few samples yet
        return float(np.clip(np.percentile(ds, 5) - self.margin, 0.12, 0.45))

    # ── main update ─────────────────────────────────────────────────────────
    def update(self, selection, states):
        """Match the selection against every non-anchor camera.
        states: {camera_id: CameraState-like} (tracks need .track_id, .cls_id,
        .embedding, .born_ts). Mutates selection (template EMA, liveness,
        re-anchoring). Returns {camera_id: MatchResult}."""
        if selection is None:
            return {}
        self._refresh_anchor(selection, states)
        # While the anchor track is alive its own camera is excluded; once it
        # dies the object may reappear ANYWHERE — including the anchor camera
        # (brief occlusion -> new track id) — so then all cameras are matched.
        results = {cam_id: self._match_one(selection, cam_id, state)
                   for cam_id, state in states.items()
                   if not (selection.alive
                           and cam_id == selection.anchor_cam)}
        self._maybe_reanchor(selection, states)
        return results

    def _refresh_anchor(self, selection, states):
        state = states.get(selection.anchor_cam)
        track = None
        if state is not None:
            track = next((t for t in state.tracks
                          if t.track_id == selection.anchor_track_id), None)
        if track is None or track.embedding is None:
            selection.alive = False
            return
        if track.label != selection.label:
            # the track id was inherited by a different-class detection
            # (bad re-association) — unbind; matching will re-acquire
            selection.anchor_track_id = -1
            selection.alive = False
            return
        selection.alive = True
        if state.ts == selection.last_seen_ts:
            return                       # same snapshot: don't re-apply EMA
        selection.last_seen_ts = state.ts
        d = cosine_distance(selection.template, track.embedding)
        if d < self.tau(selection.label):    # drift-protected slow EMA
            selection.suspect = 0
            a = self.template_alpha
            v = (1 - a) * selection.template + a * track.embedding
            selection.template = (v / (np.linalg.norm(v) + 1e-9)
                                  ).astype(np.float32)
        else:
            # anchor identity guard: a ghost track can swallow a NEW object
            # near the frame edge, inheriting the selection. If the anchor
            # persistently stops looking like the selection, unbind it and
            # let cross-camera matching (with its continuity gate) re-acquire.
            selection.suspect += 1
            if selection.suspect >= 5:
                selection.anchor_track_id = -1
                selection.alive = False
                selection.suspect = 0

    def _match_one(self, selection, cam_id, state):
        cm = self._cam.setdefault(cam_id, _CamMatch())
        # dedup: on_state() re-runs the matcher on every camera's event with
        # ALL cached snapshots; hysteresis must advance once per NEW frame
        # from THIS camera or confirm/release rates scale with camera count
        if cm.last_result is not None and state.ts == cm.last_ts:
            return cm.last_result
        cm.last_ts = state.ts
        cm.last_result = self._score_one(selection, cm, state)
        return cm.last_result

    def _score_one(self, selection, cm, state):
        tau = self.tau(selection.label)
        cands = [t for t in state.tracks
                 if t.label == selection.label and t.embedding is not None]
        scored = sorted(((cosine_distance(selection.template, t.embedding), t)
                         for t in cands), key=lambda p: p[0])
        best = scored[0] if scored else None
        second_d = scored[1][0] if len(scored) > 1 else None

        appearance_ok = (best is not None and best[0] < tau and
                         (second_d is None or
                          second_d - best[0] >= self.margin))
        if not appearance_ok:
            cm.streak = 0
            if cm.linked:
                cm.miss += 1
                if cm.miss < self.release_frames:   # ride a brief miss
                    return MatchResult("linked", cm.track_id,
                                       best[0] if best else 1.0, tau)
                cm.linked, cm.track_id, cm.miss = False, -1, 0
            return MatchResult("none", tau=tau)

        d, track = best
        smooth = (selection.alive or
                  (track.born_ts - selection.last_seen_ts)
                  <= self.handoff_window)
        if not smooth:
            # looks the same, but appeared without a smooth transition
            # through the overlap -> tag as a different physical object
            cm.streak = 0
            return MatchResult("different", track.track_id, d, tau)

        cm.miss = 0
        if track.track_id != cm.track_id:
            cm.track_id = track.track_id
            cm.streak = 1
            cm.linked = False
        else:
            cm.streak += 1
        if cm.linked or cm.streak >= self.confirm_frames:
            cm.linked = True
            return MatchResult("linked", track.track_id, d, tau)
        return MatchResult("pending", track.track_id, d, tau)

    def _maybe_reanchor(self, selection, states):
        """Anchor died while a link is live: the object moved cameras —
        transfer the anchor to the linked track and restart match state."""
        if selection.alive:
            return
        for cam_id, cm in list(self._cam.items()):
            if not cm.linked:
                continue
            state = states.get(cam_id)
            track = None
            if state is not None:
                track = next((t for t in state.tracks
                              if t.track_id == cm.track_id), None)
            if track is None:
                continue
            selection.anchor_cam = cam_id
            selection.anchor_track_id = track.track_id
            selection.last_seen_ts = state.ts
            selection.alive = True
            self.reset()               # old anchor cam is a target now
            return
