"""Neural networks for vector and image generative models."""

import math
import torch
import torch.nn as nn
from diffusers import Transformer2DModel, UNet2DModel


def _diffusers_timesteps(
    t: torch.Tensor,
    batch_size: int,
    num_train_timesteps: int,
    discrete: bool,
    device: torch.device,
) -> torch.Tensor:
    """Map genkit times to diffusers timesteps."""
    t = t.reshape(-1).to(device=device)

    if t.numel() == 1:
        t = t.expand(batch_size)

    if torch.is_floating_point(t) and t.max() <= 1.0 and t.min() >= 0.0:
        t = t * (num_train_timesteps - 1)

    return t.round().long() if discrete else t


class MLPModel(nn.Module):
    """Time-conditioned MLP for vector data."""

    class _ConditionedMLPBlock(nn.Module):
        def __init__(self, width: int, time_dim: int, dropout: float, use_norm: bool):
            super().__init__()
            self.norm = nn.LayerNorm(width) if use_norm else nn.Identity()
            self.fc1 = nn.Linear(width, width)
            self.fc2 = nn.Linear(width, width)
            self.tproj = nn.Linear(time_dim, width)
            self.drop = nn.Dropout(dropout)
            self.act = nn.SiLU()
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
            h = self.fc1(self.norm(x)) + self.tproj(temb)
            h = self.fc2(self.drop(self.act(h)))
            return x + h

    @staticmethod
    def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
        """Sinusoidal timestep embedding inherited from Transformer positional encodings."""
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.reshape(-1).to(dtype=torch.get_default_dtype())
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype) / max(half, 1))
        emb = torch.cat([torch.cos(timesteps[:, None] * freqs[None]), torch.sin(timesteps[:, None] * freqs[None])], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        dim: int | None = None,
        width: int = 128,
        depth: int = 4,
        time_dim: int = 32,
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        super().__init__()
        if input_dim is None:
            input_dim = 2 if dim is None else int(dim)
        elif dim is not None and int(dim) != int(input_dim):
            raise ValueError(f"Conflicting input_dim={input_dim} and dim={dim}.")

        self.input_dim = int(input_dim)
        self.output_dim = self.input_dim if output_dim is None else int(output_dim)
        self.time_dim = int(time_dim)
        self.inp = nn.Linear(self.input_dim, width)
        self.time = nn.Sequential(nn.Linear(time_dim, time_dim),
                                  nn.SiLU(),
                                  nn.Linear(time_dim, time_dim))
        self.blocks = nn.ModuleList(self._ConditionedMLPBlock(width, time_dim, dropout, use_norm) for _ in range(depth))
        self.out = nn.Linear(width, self.output_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)

        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected flattened input dimension {self.input_dim}, got {x_flat.shape[1]}.")
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        elif t.ndim != 1:
            raise ValueError(f"Expected t with shape [B] or [B, 1], got {t.shape}.")

        temb = self._timestep_embedding(t, self.time_dim).to(dtype=x.dtype, device=x.device)
        h = self.act(self.inp(x_flat))
        temb = self.time(temb)
        for block in self.blocks:
            h = block(h, temb)

        return self.out(h).reshape(x.shape[0], self.output_dim)


class UNetModel(nn.Module):
    """Thin adapter around diffusers.UNet2DModel."""

    def __init__(
        self,
        sample_size: int | tuple[int, int] | None = None,
        n_steps: int = 1000,
        in_channels: int = 3,
        out_channels: int | None = None,
        width: int = 64,
        model_channels: int | None = None,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        layers_per_block: int = 2,
        num_res_blocks: int | None = None,
        norm_num_groups: int = 8,
        attention: bool = True,
        attention_resolutions: tuple[int, ...] | None = None,
        num_heads: int | None = None,
        conv_resample: bool = True,
        dims: int = 2,
        use_scale_shift_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        if dims != 2:
            raise ValueError(f"UNetModel only supports dims=2, got {dims}.")
        if not conv_resample:
            raise ValueError("UNetModel only supports conv_resample=True.")
        if model_channels is not None:
            width = int(model_channels)
        if num_res_blocks is not None:
            layers_per_block = int(num_res_blocks)

        out_channels = in_channels if out_channels is None else out_channels
        channels = tuple(width * m for m in channel_mult)
        if attention and attention_resolutions is not None:
            attention_set = {int(value) for value in attention_resolutions}
            down_block_types = tuple(
                "AttnDownBlock2D" if 2 ** index in attention_set else "DownBlock2D"
                for index in range(len(channels))
            )
        elif attention:
            down_block_types = ("DownBlock2D",) + ("AttnDownBlock2D",) * (len(channels) - 1)
        else:
            down_block_types = ("DownBlock2D",) * len(channels)

        up_block_types = tuple(
            "AttnUpBlock2D" if block == "AttnDownBlock2D" else "UpBlock2D"
            for block in reversed(down_block_types)
        )
        if num_heads is not None:
            attention_channels = [channel for block, channel in zip(down_block_types, channels) if block == "AttnDownBlock2D"]
            head_channels = attention_channels[0] if attention_channels else channels[0]
            kwargs.setdefault("attention_head_dim", max(head_channels // int(num_heads), 1))
        if use_scale_shift_norm:
            kwargs.setdefault("resnet_time_scale_shift", "scale_shift")

        self.num_train_timesteps = int(n_steps)
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            norm_num_groups=norm_num_groups,
            num_train_timesteps=n_steps,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _diffusers_timesteps(t=t, batch_size=x.shape[0], num_train_timesteps=self.num_train_timesteps, discrete=False, device=x.device)
        return self.model(x, t).sample


class TransformerModel(nn.Module):
    """Thin adapter around diffusers.Transformer2DModel."""

    def __init__(
        self,
        sample_size: int | tuple[int, int],
        n_steps: int,
        in_channels: int = 3,
        out_channels: int | None = None,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        attention_head_dim: int = 64,
        norm_num_groups: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.num_train_timesteps = int(n_steps)
        self.model = Transformer2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels if out_channels is None else out_channels,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_type="ada_norm",
            num_embeds_ada_norm=n_steps,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _diffusers_timesteps(t=t, batch_size=x.shape[0], num_train_timesteps=self.num_train_timesteps, discrete=True, device=x.device)
        return self.model(x, timestep=t).sample


__all__ = [
    "MLPModel",
    "TransformerModel",
    "UNetModel",
]
