from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class SILogLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 10.0,
        variance_focus: float = 0.85,
        eps: float = 1e-6,
        min_valid_pixels: int = 16,
        interpolate: bool = True,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.variance_focus = float(variance_focus)
        self.eps = float(eps)
        self.min_valid_pixels = int(min_valid_pixels)
        self.interpolate = interpolate

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4 or mask.ndim != 4:
            raise ValueError("pred, target, and mask must all have shape [N,1,H,W]")
        pred = pred.float()
        target = target.float()
        if self.interpolate and pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.bool()
        losses: List[torch.Tensor] = []
        for pred_i, target_i, mask_i in zip(pred, target, mask):
            valid_count = int(mask_i.sum().item())
            if valid_count < self.min_valid_pixels:
                continue
            log_diff = torch.log(pred_i[mask_i].clamp_min(self.eps)) - torch.log(target_i[mask_i].clamp_min(self.eps))
            value = log_diff.pow(2).mean() - self.variance_focus * log_diff.mean().pow(2)
            losses.append(self.alpha * torch.sqrt(value.clamp_min(0.0)))
        if not losses:
            return pred.sum() * 0.0
        return torch.stack(losses).mean()
