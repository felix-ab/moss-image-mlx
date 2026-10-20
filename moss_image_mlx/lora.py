"""Unmerged LoRA adapters for the DiT (kohya / diffusers key formats).

The adapter runs beside the base linear, in fp32: y = base(x) + scale * (alpha / rank) * up(down(x)).
It is never merged into the base weights, so it works on quantized tiers and keeps deltas that are smaller
than bf16 rounding (the few-step DMD adapters depend on that).
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

F32 = mx.float32

# kohya: diffusion_model.layers.3.proj.q.{lora_down.weight,lora_up.weight,alpha}
# diffusers/peft: transformer.layers.3.proj.q.{lora_A.weight,lora_B.weight}
_KEY = re.compile(r"^(?:diffusion_model|transformer)\.(.+?)\.(lora_down|lora_up|lora_A|lora_B|alpha)(?:\.weight)?$")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, down: mx.array, up: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.down, self.up = down.astype(F32), up.astype(F32)
        self.scale = scale

    def __call__(self, x):
        delta = (x.astype(F32) @ self.down.T) @ self.up.T
        return self.base(x) + (delta * self.scale).astype(x.dtype)


def load_lora(path: str | Path) -> dict[str, dict]:
    """Group a LoRA safetensors file by module path: {"layers.3.proj.q": {"down": ..., "up": ..., "alpha": ...}}."""
    mods: dict[str, dict] = {}
    for key, value in mx.load(str(path)).items():
        m = _KEY.match(key)
        if not m:
            raise ValueError(f"unrecognized LoRA key: {key}")
        slot = {"lora_down": "down", "lora_A": "down", "lora_up": "up", "lora_B": "up", "alpha": "alpha"}[m.group(2)]
        mods.setdefault(m.group(1), {})[slot] = value
    return mods


def apply_lora(model: nn.Module, path: str | Path, scale: float = 1.0) -> int:
    """Wrap every matching linear in `model` with an unmerged adapter. Returns the number of modules wrapped."""
    mods = load_lora(path)
    count = 0
    for name, tensors in mods.items():
        *parents, leaf = name.split(".")
        node = model
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else (node[part] if isinstance(node, dict) else getattr(node, part))
        base = node[leaf] if isinstance(node, dict) else getattr(node, leaf)
        rank = tensors["down"].shape[0]
        alpha = float(tensors.get("alpha", rank))
        wrapped = LoRALinear(base, tensors["down"], tensors["up"], scale * alpha / rank)
        if isinstance(node, dict):
            node[leaf] = wrapped
        else:
            setattr(node, leaf, wrapped)
        count += 1
    if count != len(mods):
        raise ValueError("LoRA modules not all applied")
    return count
