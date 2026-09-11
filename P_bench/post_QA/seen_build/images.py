"""Pixel-quality checks; optional ephemeral DINO features for benchmark selection."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


MODEL = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = (336, 252)  # Full 4:3 view, no centre crop; multiples of patch size.
QUALITY_LIMITS = {"black_fraction": 0.05, "largest_black_fraction": 0.02,
                  "minimum_laplacian_variance": 25.0, "minimum_gray_std": 12.0}
EXTREME_DARK_FRACTION = 0.90


def dark_fraction(image):
    """Fraction of native pixels with all RGB channels at most eight."""
    mask = cv2.inRange(np.asarray(image.convert("RGB")), (0, 0, 0), (8, 8, 8))
    return cv2.countNonZero(mask) / mask.size


def quality(image):
    rgb = np.asarray(image.resize((320, 240), Image.Resampling.BILINEAR))
    black = (rgb.max(axis=2) <= 8).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    largest = int(components[1:, cv2.CC_STAT_AREA].max(initial=0)) / black.size
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    std = float(gray.std())
    return {"black_fraction": float(black.mean()),
            "largest_black_fraction": largest, "laplacian_variance": sharpness,
            "gray_std": std,
            "accepted": bool(black.mean() <= QUALITY_LIMITS["black_fraction"] and
                             largest <= QUALITY_LIMITS["largest_black_fraction"] and
                             sharpness >= QUALITY_LIMITS["minimum_laplacian_variance"] and
                             std >= QUALITY_LIMITS["minimum_gray_std"])}


def _decode(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        metrics = quality(image)
        resized = (np.asarray(image.resize(INPUT_SIZE, Image.Resampling.BICUBIC)).copy()
                   if metrics["accepted"] else None)
    return metrics, resized


class ImageBank:
    """Decode each requested view once. Keep only small metrics and features."""

    def __init__(self, device="cuda:0", batch_size=64):
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.torch, self.device, self.batch_size = torch, device, batch_size
        checkpoint = hf_hub_download(f"timm/{MODEL}", "model.safetensors")
        self.model = timm.create_model(MODEL, pretrained=False, num_classes=0,
                                      dynamic_img_size=True, checkpoint_path=checkpoint)
        self.model.eval().to(device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device)[None, :, None, None]
        self.std = torch.tensor(cfg["std"], device=device)[None, :, None, None]
        self.metrics, self.features = {}, {}
        self.timings = Counter()
        self.checkpoint = str(Path(checkpoint).resolve())

    def add(self, paths):
        paths = sorted(set(map(str, paths)) - self.metrics.keys())
        torch = self.torch
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0, len(paths), self.batch_size):
                batch = paths[offset:offset + self.batch_size]
                decoded = list(pool.map(_decode, batch))
                good = []
                for path, (metrics, pixels) in zip(batch, decoded):
                    self.metrics[path] = metrics
                    if pixels is not None:
                        good.append((path, pixels))
                if good:
                    values = torch.from_numpy(np.stack([p for _, p in good]))
                    values = values.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        features = self.model((values - self.mean) / self.std)
                    features = torch.nn.functional.normalize(features.float(), dim=1).cpu()
                    self.features.update((path, feature) for (path, _), feature in zip(good, features))
                if offset == 0 or (offset + self.batch_size) % 1024 == 0:
                    print(f"[images] decoded={len(self.metrics)} encoded={len(self.features)} "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        self.timings["decode_and_dino_seconds"] += time.monotonic() - started

    def good_endpoints(self, paths):
        """C1 options need quality checks, not semantic separation from each other."""
        for path in map(str, paths):
            if path not in self.metrics:
                with Image.open(path) as image:
                    self.metrics[path] = quality(image.convert("RGB"))
            if not self.metrics[path]["accepted"]:
                return False
        return True

    def conflicts(self, paths, threshold):
        """Direct cosine neighbours, blockwise; never an N-by-N stored matrix."""
        torch = self.torch
        matrix = torch.stack([self.features[path] for path in paths]).to(self.device)
        neighbors = [set() for _ in paths]
        with torch.inference_mode():
            for offset in range(0, len(paths), 512):
                similarities = matrix[offset:offset + 512] @ matrix.T
                rows, cols = torch.where(similarities >= threshold)
                for row, col in zip(rows.cpu().tolist(), cols.cpu().tolist()):
                    neighbors[offset + row].add(col)
        return neighbors

    def nearest_pairs(self, paths, ids):
        torch = self.torch
        matrix = torch.stack([self.features[path] for path in paths]).to(self.device)
        nearest, pairs = [], []
        with torch.inference_mode():
            for offset in range(0, len(paths), 512):
                scores = matrix[offset:offset + 512] @ matrix.T
                scores[torch.arange(len(scores)), torch.arange(offset, offset + len(scores))] = -1
                values, indices = scores.max(dim=1)
                for i, (value, j) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), offset):
                    nearest.append(value)
                    if i < j:
                        pairs.append({"left": ids[i], "right": ids[j], "cosine": value})
        return {"maximum_cosine": max(nearest),
                "nearest_cosine_quantiles": dict(zip(("p50", "p90", "p99"),
                    map(float, np.quantile(nearest, (.5, .9, .99))))),
                "nearest_pairs": sorted(pairs, key=lambda row: -row["cosine"])[:50]}
