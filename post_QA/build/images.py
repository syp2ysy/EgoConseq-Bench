"""Unmarked-image metrics, hash-keyed defect reviews, and DINOv2 near-duplicate features."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from pipeline import io_utils

LUMA = np.array([0.299, 0.587, 0.114])  # ITU-R 601, kept in floating point


def validate_review(sha, decision, reason):
    if (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
            or decision not in {"keep", "reject"} or not reason
            or decision == "reject" and reason not in {
                "mesh_artifact", "render_artifact", "dark", "gs_bright_low_contrast"}):
        raise ValueError(f"invalid image review: {sha!r}, {decision!r}, {reason!r}")


class Reviews:
    """Frozen original-image decisions. All datasets require an explicit keep."""

    def __init__(self, path=None):
        self.decisions, self.reasons, self.raw, self.identity = {}, {}, None, None
        self.provenance = None
        if path is None:
            return
        self.raw = Path(path).read_bytes()
        rows = csv.DictReader(io.StringIO(self.raw.decode("utf-8-sig")))
        if not {"image_sha256", "decision", "reason"}.issubset(rows.fieldnames or []):
            raise ValueError("image reviews need image_sha256, decision and reason columns")
        model_protocols = set()
        for row in rows:
            sha, decision, reason = row["image_sha256"], row["decision"], row["reason"]
            validate_review(sha, decision, reason)
            if sha in self.decisions:
                raise ValueError(f"duplicate image review: {sha}")
            self.decisions[sha], self.reasons[sha] = decision, reason
            if row.get("source") == "model":
                model_protocols.add(row.get("protocol_version"))
        self.identity = {"path": str(path), "sha256": hashlib.sha256(self.raw).hexdigest(),
                         "decisions": dict(Counter(self.decisions.values()))}
        manifest = Path(path).with_suffix(".json")
        if model_protocols and not manifest.exists():
            raise ValueError(f"model review table needs its protocol manifest: {manifest}")
        if manifest.exists():
            self.provenance = manifest.read_bytes()
            frozen = json.loads(self.provenance)
            if frozen["table_sha256"] != self.identity["sha256"]:
                raise ValueError(f"review table differs from its frozen manifest: {path}")
            if model_protocols and model_protocols != {frozen["protocol_version"]}:
                raise ValueError(f"model review protocol differs from its frozen manifest: {path}")
            self.identity["provenance_sha256"] = hashlib.sha256(self.provenance).hexdigest()

    def status(self, sha):
        return self.decisions.get(sha, "unreviewed")

    def eligible(self, sha):
        return self.status(sha) == "keep"

    def exclusion_reason(self, sha):
        decision = self.status(sha)
        return self.reasons[sha] if decision == "reject" else decision

    def summary(self, hashes):
        return dict(sorted(Counter(self.status(sha) for sha in set(hashes)).items()))


class Quality(NamedTuple):
    dark_pixel_fraction: float
    edges: float


def quality(path) -> Quality:
    """Dark-area fraction and edge density of an unmarked image, decoded once.

    dark: the share of pixels whose every channel is below 12 (the meta's dark_pixel_fraction).
    edges: the share of interior pixels whose brightness gradient exceeds 8 grey levels, on the float luma
    averaged over 4×4 blocks (640×480 -> 160×120), by central differences (numpy.gradient; its one-sided
    outer ring is left out). Report only: low texture and close-ups are not rendering defects.
    """
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = rgb @ LUMA
    h, w = gray.shape[0] // 4 * 4, gray.shape[1] // 4 * 4
    gray = gray[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    gy, gx = np.gradient(gray)
    maximum = rgb.max(axis=2)
    return Quality(float((maximum < 12).mean()), float((np.hypot(gx, gy)[1:-1, 1:-1] > 8).mean()))


def qualities(paths, workers=16) -> list:
    with Pool(workers) as pool:
        return pool.map(quality, list(paths), chunksize=64)


def dark_fraction(path) -> float:
    """The meta's dark_pixel_fraction: the share of pixels whose every channel is below 12."""
    return float((_decode(path).max(axis=2) < 12).mean())


def dark_fractions(paths, workers=16) -> list:
    with Pool(workers) as pool:
        return pool.map(dark_fraction, list(paths), chunksize=64)


class Qualities:
    """Shared image gate and dark-fraction cache. Initial values come from meta; options are decoded once."""

    def __init__(self, meta, reviews=None):
        self.meta, self.measured = meta, {}
        self.reviews = reviews if reviews is not None else Reviews()
        self.considered = set()  # (dataset, option path), including options awaiting review

    def __call__(self, path):
        if path not in self.measured:
            self.measured[path] = dark_fraction(self.meta / path)
        return self.measured[path]

    def rejection(self, path, sha, spec, strict):
        if not self.reviews.eligible(sha):
            return self.reviews.exclusion_reason(sha), 1.
        dark = self(path)
        if dark > spec["benchmark" if strict else "sft"]["max_dark_fraction"]:
            return "dark", dark
        return None

    def prefetch(self, paths):
        paths = sorted(set(paths) - self.measured.keys())
        self.measured.update(zip(paths, dark_fractions(self.meta / p for p in paths)))


def _decode(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class Dino:
    """Normalised DINOv2 features (float32) of RGB images resized to 336×252 (bicubic, antialiased) and
    standardised with the model's mean and std, all on the GPU."""

    def __init__(self, model):
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = hf_hub_download(f"timm/{model}", "model.safetensors")
        self.model = timm.create_model(model, pretrained=False, num_classes=0, dynamic_img_size=True,
                                       checkpoint_path=weights)
        self.model.eval().to(self.device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=self.device)[:, None, None]
        self.std = torch.tensor(cfg["std"], device=self.device)[:, None, None]
        self.identity = {"model": model, "weights_sha256": io_utils.sha256_file(weights),
                         "input": "RGB 336x252 bicubic antialiased, model mean/std", "precision": "float32"}

    def _embed(self, pixels):
        torch = self.torch
        with torch.inference_mode():
            values = torch.from_numpy(pixels).to(self.device).permute(0, 3, 1, 2).float() / 255
            values = torch.nn.functional.interpolate(values, size=(252, 336), mode="bicubic", antialias=True)
            return torch.nn.functional.normalize(self.model((values - self.mean) / self.std), dim=1)

    def features(self, paths, workers=16, batch=64):
        """Features of many images, a tensor on the GPU: decoded in parallel, embedded in batches."""
        out, chunk = [], []
        with Pool(workers) as pool:
            for pixels in pool.imap(_decode, list(paths), chunksize=8):
                chunk.append(pixels)
                if len(chunk) == batch:
                    out.append(self._embed(np.stack(chunk)))
                    chunk = []
        if chunk:
            out.append(self._embed(np.stack(chunk)))
        return self.torch.cat(out) if out else self.torch.zeros((0, self.model.num_features), device=self.device)

    def embed(self, path):
        """The feature of one image."""
        return self._embed(_decode(path)[None])[0]
