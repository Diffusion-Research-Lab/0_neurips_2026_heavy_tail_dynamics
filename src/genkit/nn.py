"""Standalone neural networks for low-dimensional and image generative models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _ConditionedMLPBlock(nn.Module):
    """Residual MLP block conditioned on a time embedding."""

    def __init__(self, width: int, time_dim: int, dropout: float, use_norm: bool):
        """Build the linear, normalization, and time-conditioning layers."""
        super().__init__()
        self.norm1 = nn.LayerNorm(width) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(width) if use_norm else nn.Identity()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.tproj = nn.Linear(time_dim, width)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Apply one residual update conditioned on the time embedding."""
        h = self.fc1(self.drop(x))
        h = self.act(self.norm1(h))
        h = h + self.tproj(self.drop(temb))
        h = self.fc2(self.drop(h))
        h = self.norm2(h)
        return self.act(x + h)


class MLPModel(nn.Module):
    """Time-conditioned MLP for 1D/2D vector data."""

    def __init__(
        self,
        dim: int = 2,
        width: int = 64,
        depth: int = 4,
        time_dim: int = 32,
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        """Build a standalone time-conditioned MLP for vector-valued data."""
        super().__init__()
        self.dim = int(dim)
        self.time_dim = int(time_dim)
        self.inp = nn.Linear(self.dim, width)
        self.time = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _ConditionedMLPBlock(width=width, time_dim=time_dim, dropout=dropout, use_norm=use_norm)
            for _ in range(depth)
        )
        self.out = nn.Linear(width, self.dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outputs for flattened vector inputs conditioned on time."""
        x_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)
        if x_flat.shape[1] != self.dim:
            raise ValueError(f"Expected flattened input dimension {self.dim}, got {x_flat.shape[1]}.")

        if t.ndim > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        temb = self.time(timestep_embedding(t, self.time_dim).to(dtype=x.dtype))

        h = self.act(self.inp(x_flat))
        for block in self.blocks:
            h = block(h, temb)
        out = self.out(h)
        return out.reshape(x_shape)


def _conv_nd(dims: int, *args, **kwargs):
    """Construct an N-dimensional convolution layer."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}")


def _avg_pool_nd(dims: int, *args, **kwargs):
    """Construct an N-dimensional average-pooling layer."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimensions: {dims}")


def _norm(channels: int) -> nn.Module:
    """Create the default normalization layer used in the U-Net blocks."""
    return nn.GroupNorm(min(32, channels), channels)


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of a module and return it."""
    for param in module.parameters():
        param.detach().zero_()
    return module


class _TimestepBlock(nn.Module):
    """Interface for modules that consume a timestep embedding."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block to activations `x` conditioned on embedding `emb`."""
        raise NotImplementedError


class _TimestepEmbedSequential(nn.Sequential, _TimestepBlock):
    """Sequential container that forwards timestep embeddings to compatible layers."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Run the sequence while passing embeddings to timestep-aware submodules."""
        for layer in self:
            if isinstance(layer, _TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class _Upsample(nn.Module):
    """Nearest-neighbor upsampling block with an optional convolution."""

    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        """Configure the upsampling path for the requested dimensionality."""
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.conv = _conv_nd(dims, channels, channels, 3, padding=1) if use_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample activations by a factor of two along spatial dimensions."""
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    """Downsampling block with either convolutional or average-pooling reduction."""

    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        """Configure the downsampling operator for the requested dimensionality."""
        super().__init__()
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = _conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = _avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce spatial resolution by a factor of two."""
        return self.op(x)


class _ResBlock(_TimestepBlock):
    """Residual convolutional block conditioned on a timestep embedding."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
    ):
        """Build the residual block and optional scale-shift conditioning path."""
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            _norm(channels),
            nn.SiLU(),
            _conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            _norm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            _zero_module(_conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip = nn.Identity()
        elif use_conv:
            self.skip = _conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip = _conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to activations conditioned on the time embedding."""
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).to(dtype=h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            norm, rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = norm(h) * (1 + scale) + shift
            h = rest(h)
        else:
            h = self.out_layers(h + emb_out)
        return self.skip(x) + h


class _AttentionBlock(nn.Module):
    """Self-attention block over flattened spatial positions."""

    def __init__(self, channels: int, num_heads: int = 1):
        """Initialize multi-head attention over channel activations."""
        super().__init__()
        self.num_heads = num_heads
        self.norm = _norm(channels)
        self.qkv = _conv_nd(1, channels, channels * 3, 1)
        self.proj_out = _zero_module(_conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and add the result back to the input activations."""
        b, c, *spatial = x.shape
        h = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(h)).reshape(b * self.num_heads, -1, h.shape[-1])
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).to(dtype=weight.dtype)
        h = torch.einsum("bts,bcs->bct", weight, v).reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x.reshape(b, c, -1) + h).reshape(b, c, *spatial)


class UNetModel(nn.Module):
    """Minimal U-Net for image data."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions,
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: int | None = None,
        num_heads: int = 1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
    ):
        """Build a compact U-Net with residual and attention blocks."""
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.model_channels = model_channels
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.label_emb = nn.Embedding(num_classes, time_embed_dim) if num_classes is not None else None

        self.input_blocks = nn.ModuleList(
            [_TimestepEmbedSequential(_conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    _ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(_AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(_TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(_TimestepEmbedSequential(_Downsample(ch, conv_resample, dims=dims)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = _TimestepEmbedSequential(
            _ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=use_scale_shift_norm),
            _AttentionBlock(ch, num_heads=num_heads),
            _ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(num_res_blocks + 1):
                layers = [
                    _ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(_AttentionBlock(ch, num_heads=num_heads_upsample))
                if level and block_idx == num_res_blocks:
                    layers.append(_Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(_TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            _norm(ch),
            nn.SiLU(),
            _zero_module(_conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Predict outputs for image-like inputs conditioned on timesteps and labels."""
        if timesteps.ndim > 1:
            timesteps = timesteps.reshape(timesteps.shape[0], -1)[:, 0]

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(device=x.device, dtype=x.dtype))
        if self.label_emb is not None:
            if y is None:
                raise ValueError("Class-conditional UNetModel requires labels.")
            emb = emb + self.label_emb(y)

        h = x
        skips = []
        for module in self.input_blocks:
            h = module(h, emb)
            skips.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)


__all__ = [
    "MLPModel",
    "UNetModel",
    "timestep_embedding",
]
