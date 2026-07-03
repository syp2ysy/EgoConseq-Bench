import torch
from torch import nn


class DepthReadoutHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        max_depth: float = 10.0,
        patch_size: int = 1,
        eps: float = 1e-3,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        self.max_depth = float(max_depth)
        self.patch_size = int(patch_size)
        self.eps = float(eps)
        out_dim = self.patch_size * self.patch_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.pixel_shuffle = nn.PixelShuffle(self.patch_size) if self.patch_size > 1 else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must have shape [N,C,H,W], got {tuple(features.shape)}")
        x = features.permute(0, 2, 3, 1)
        logits = self.mlp(x).permute(0, 3, 1, 2)
        if self.pixel_shuffle is not None:
            logits = self.pixel_shuffle(logits)
        return torch.sigmoid(logits) * self.max_depth + self.eps
