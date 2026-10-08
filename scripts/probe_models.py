"""Small classifiers used as VRAVEN measurement probes.

Includes tabular MLPs / 1D CNNs plus image CNN2D, diffusion-style UNet
classifiers, and Transformer encoders — all emit class logits.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ProbeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (64, 32),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for size in hidden:
            layers.extend((nn.Linear(current, size), nn.GELU(), nn.Dropout(0.05)))
            current = size
        layers.append(nn.Linear(current, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ProbeCNN1D(nn.Module):
    """1-channel CNN over a fixed-length feature sequence."""

    def __init__(self, input_dim: int, channels: tuple[int, ...] = (16, 32), num_classes: int = 2):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            blocks.extend(
                (
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, ceil_mode=True),
                )
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        with torch.no_grad():
            flat = int(self.backbone(torch.zeros(1, 1, input_dim)).numel())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x.unsqueeze(1)))


class ProbeImageCNN2D(nn.Module):
    """Classic 2D CNN image classifier (never used in earlier probes)."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 32,
        channels: tuple[int, ...] = (16, 32, 64),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        ch = in_channels
        for out_ch in channels:
            blocks.extend(
                (
                    nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
            )
            ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        with torch.no_grad():
            flat = int(self.backbone(torch.zeros(1, in_channels, image_size, image_size)).numel())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))


class _ResBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ProbeDiffusionUNetClassifier(nn.Module):
    """Tiny diffusion-style UNet encoder + class head.

    Optionally noise-conditions the forward pass (t in [0,1]) the way a
    noise-aware diffusion classifier would, while still emitting logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 32,
        base_channels: int = 32,
        num_classes: int = 2,
        noise_conditioned: bool = True,
    ) -> None:
        super().__init__()
        self.noise_conditioned = noise_conditioned
        self.image_size = image_size
        c = base_channels
        self.stem = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(_ResBlock2d(c), nn.Conv2d(c, c * 2, 4, stride=2, padding=1))
        self.down2 = nn.Sequential(_ResBlock2d(c * 2), nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1))
        self.mid = _ResBlock2d(c * 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, c * 4),
            nn.SiLU(),
            nn.Linear(c * 4, c * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c * 4, num_classes)

    def forward(self, x: Tensor, t: Tensor | None = None) -> Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.mid(h)
        if self.noise_conditioned:
            if t is None:
                t = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            elif t.ndim == 1:
                t = t.unsqueeze(-1)
            h = h + self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = self.pool(h).flatten(1)
        return self.head(h)


class _SimpleAttentionBlock(nn.Module):
    """Export-friendly attention block (avoids nested TransformerEncoder asserts)."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class ProbeTransformerClassifier(nn.Module):
    """Token Transformer encoder over a fixed-length feature sequence."""

    def __init__(
        self,
        seq_len: int,
        d_model: int = 48,
        nhead: int = 4,
        layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([_SimpleAttentionBlock(d_model, nhead) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.input_proj(x.unsqueeze(-1)) + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens.mean(dim=1))


class ProbeViTTiny(nn.Module):
    """Tiny patch Vision Transformer for images (unused earlier)."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = patch_size
        self.grid = image_size // patch_size
        n_patches = self.grid * self.grid
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.blocks = nn.ModuleList([_SimpleAttentionBlock(d_model, nhead) for _ in range(layers)])
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        p = self.patch
        # Explicit patchify (more capture-friendly than unfold chains).
        patches = x.reshape(b, c, self.grid, p, self.grid, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, self.grid * self.grid, c * p * p)
        tokens = self.proj(patches)
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0])


def build_probe(kind: str, **kwargs) -> nn.Module:
    builders = {
        "mlp": ProbeMLP,
        "cnn1d": ProbeCNN1D,
        "cnn2d": ProbeImageCNN2D,
        "diffusion_unet": ProbeDiffusionUNetClassifier,
        "transformer": ProbeTransformerClassifier,
        "vit_tiny": ProbeViTTiny,
    }
    if kind not in builders:
        raise ValueError(f"Unknown probe kind {kind!r}; expected one of {sorted(builders)}")
    return builders[kind](**kwargs)
