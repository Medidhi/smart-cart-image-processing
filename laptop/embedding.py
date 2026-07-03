"""
Appearance embeddings for cross-camera re-identification.

Embeddings are cut from the CLEAN frame (before any drawing), L2-normalized
unit vectors; distance = cosine (1 - dot). Two embedders:

- DeepEmbedder ('deep', default): frozen ImageNet torchvision backbone
  (mobilenet_v3_small, 576-d). Deep features are what separates two
  same-class items (two apples) — the case cross-camera re-ID lives on.
  First use downloads the backbone weights to TORCH_HOME. torch/torchvision
  are lazy-imported: machines that only view Pi streams can run
  --embedder hist and never import them.
- HistEmbedder ('hist'): HSV histogram + 3x3 spatial mean-color grid,
  numpy/cv2 only. Fallback when torch is unavailable.

embed_batch() is thread-safe (internal lock) — one embedder instance is
shared by all camera threads so backbone weights load exactly once.
"""
import threading

import cv2
import numpy as np


def crop_box(frame, box, inset=0.08):
    """Clamped, inset crop — the inset drops background bleed at the box edge
    (and stray annotation pixels if a Pi was left running --annotate)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 + inset * bw))
    x2 = int(min(w, x2 - inset * bw))
    y1 = int(max(0, y1 + inset * bh))
    y2 = int(min(h, y2 - inset * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def cosine_distance(u, v):
    """1 - dot product of two unit vectors (0 = identical, 2 = opposite)."""
    return float(1.0 - np.dot(u, v))


class HistEmbedder:
    """HSV histogram (16x8) + 3x3 grid of mean HSV. No torch."""

    def __init__(self):
        self._lock = threading.Lock()

    def _one(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8],
                            [0, 180, 0, 256]).flatten()
        hist /= (hist.sum() + 1e-9)
        gh, gw = hsv.shape[0] // 3, hsv.shape[1] // 3
        grid = []
        for gy in range(3):
            for gx in range(3):
                cell = hsv[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
                grid.extend(cell.reshape(-1, 3).mean(axis=0) / 255.0)
        v = np.concatenate([hist, np.asarray(grid, dtype=np.float32)])
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    def embed_batch(self, frame_bgr, boxes):
        with self._lock:
            out = []
            for box in boxes:
                crop = crop_box(frame_bgr, box)
                out.append(None if crop is None else self._one(crop))
            return out


class DeepEmbedder:
    """Frozen ImageNet torchvision backbone -> L2-normalized feature vector.
    All crops of a frame go through ONE batched forward pass."""

    _INPUT = {"mobilenet_v3_small": 128, "mobilenet_v3_large": 160,
              "resnet50": 224}

    def __init__(self, backbone="mobilenet_v3_small", device="auto"):
        import torch          # lazy — deep embedder only
        import torchvision
        if device == "auto":
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._torch = torch
        if backbone == "mobilenet_v3_small":
            m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            net = torch.nn.Sequential(m.features, m.avgpool,
                                      torch.nn.Flatten(1))
        elif backbone == "mobilenet_v3_large":
            m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V1")
            net = torch.nn.Sequential(m.features, m.avgpool,
                                      torch.nn.Flatten(1))
        elif backbone == "resnet50":
            m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            m.fc = torch.nn.Identity()
            net = m
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.size = self._INPUT[backbone]
        self.net = net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor([0.485, 0.456, 0.406],
                                  device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225],
                                 device=device).view(1, 3, 1, 1)
        self._lock = threading.Lock()

    def embed_batch(self, frame_bgr, boxes):
        """Returns a list aligned with boxes (None for degenerate crops)."""
        crops, keep = [], []
        for k, box in enumerate(boxes):
            crop = crop_box(frame_bgr, box)
            if crop is None:
                continue
            crop = cv2.resize(crop, (self.size, self.size),
                              interpolation=cv2.INTER_AREA)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            keep.append(k)
        out = [None] * len(boxes)
        if not crops:
            return out
        torch = self._torch
        batch = torch.from_numpy(np.stack(crops)).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = (batch - self._mean) / self._std
        with self._lock, torch.no_grad():
            feats = self.net(batch)
        feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()
        for k, f in zip(keep, feats):
            out[k] = f.astype(np.float32)
        return out


def make_embedder(kind="deep", backbone="mobilenet_v3_small", device="auto"):
    if kind == "deep":
        return DeepEmbedder(backbone, device)
    if kind == "hist":
        return HistEmbedder()
    raise ValueError(kind)
