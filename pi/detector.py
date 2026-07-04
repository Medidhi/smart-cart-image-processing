"""
HailoObjectDetector — runs YOLOv8 on the Hailo-8 AI HAT+ and returns detections.

Reused from ~/depth_detect/detector.py (depth bits removed). Give it any HxWx3 BGR
uint8 frame and it returns a list of Detection objects in ORIGINAL-frame pixel
coords. On-chip NMS means the model output is already decoded boxes+scores per class.
"""
from dataclasses import dataclass
import numpy as np
import cv2
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType,
    HailoSchedulingAlgorithm,
)
from coco_labels import COCO_CLASSES


@dataclass
class Detection:
    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int = -1

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class HailoObjectDetector:
    def __init__(self, hef_path="models/yolov8s_h8.hef", score_thresh=0.4,
                 labels=None, shared=False):
        """labels: index-aligned class-name list matching the .hef NMS output.
        Defaults to the 80 COCO names; pass grocery_labels.GROCERY_NAMES for
        the custom 43-class grocery model.
        shared: open the Hailo through the hailort multi-process service so
        several processes (e.g. one server per CSI camera) can share the one
        device. Requires `systemctl start hailort`. The service scheduler
        activates network groups itself, so detect() skips manual activate."""
        self.labels = labels if labels is not None else COCO_CLASSES
        self.score_thresh = score_thresh
        self.shared = shared
        self.hef = HEF(hef_path)
        if shared:
            params = VDevice.create_params()
            # the multi-process service requires the HailoRT scheduler
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.multi_process_service = True
            params.group_id = "SHARED"
            self.target = VDevice(params)
        else:
            self.target = VDevice()
        cfg = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()

        in_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = in_info.name
        self.in_h, self.in_w, _ = in_info.shape  # 640, 640, 3
        self.output_name = self.hef.get_output_vstream_infos()[0].name

        self.in_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8)
        self.out_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32)

    def _letterbox(self, frame):
        """Resize keeping aspect ratio, pad to model size. Returns padded img + transform."""
        h, w = frame.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def detect(self, frame_bgr):
        orig_h, orig_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_x, pad_y = self._letterbox(rgb)
        batch = np.expand_dims(canvas, axis=0)  # (1,640,640,3) uint8

        with InferVStreams(self.network_group, self.in_params, self.out_params) as pipe:
            if self.shared:
                # multi-process service: the scheduler owns activation
                results = pipe.infer({self.input_name: batch})
            else:
                with self.network_group.activate(self.ng_params):
                    results = pipe.infer({self.input_name: batch})

        # HAILO_NMS_BY_CLASS: results[out] -> array(batch) of list[N] of (n,5)
        # arrays, one entry per class in the compiled .hef
        raw = results[self.output_name][0]
        if len(raw) != len(self.labels) and not getattr(self, "_warned", False):
            self._warned = True
            print(f"[detector] WARNING: .hef emits {len(raw)} classes but "
                  f"{len(self.labels)} labels loaded — wrong --labels choice "
                  f"or class-index invariant broken (training/HAILO.md)")
        dets = []
        for cls_id, boxes in enumerate(raw):
            if boxes is None or len(boxes) == 0:
                continue
            for det in boxes:
                ymin, xmin, ymax, xmax, score = det[:5]
                if score < self.score_thresh:
                    continue
                # NMS coords are normalized to the 640x640 letterboxed canvas
                x1 = (xmin * self.in_w - pad_x) / scale
                y1 = (ymin * self.in_h - pad_y) / scale
                x2 = (xmax * self.in_w - pad_x) / scale
                y2 = (ymax * self.in_h - pad_y) / scale
                dets.append(Detection(
                    label=self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id),
                    score=float(score),
                    x1=max(0, int(x1)), y1=max(0, int(y1)),
                    x2=min(orig_w - 1, int(x2)), y2=min(orig_h - 1, int(y2)),
                    cls_id=cls_id,
                ))
        return dets


def draw(frame, dets, color_for=None):
    """Draw boxes + labels. color_for(label)->(b,g,r) picks a stable per-class color."""
    for d in dets:
        color = color_for(d.label) if color_for else (0, 255, 0)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        txt = f"{d.label} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (d.x1, max(0, d.y1 - th - 6)),
                      (d.x1 + tw + 2, d.y1), color, -1)
        cv2.putText(frame, txt, (d.x1 + 1, max(10, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
