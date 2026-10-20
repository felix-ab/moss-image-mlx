"""LoRA adapters: key parsing, unmerged application on bf16 and quantized layers, stacking, CLI spec."""

import mlx.core as mx
import numpy as np
import pytest

from moss_image_mlx.cli import _lora_spec
from moss_image_mlx.dit import MOSSImageTransformer
from moss_image_mlx.lora import LoRALinear, apply_lora, load_lora
from moss_image_mlx.quantize import quantize_dit

MODULES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def _write_lora(path, model, rank=4, alpha=2.0, fmt="kohya"):
    """A random adapter for every proj linear of every block, in kohya or diffusers/PEFT key format."""
    rng = np.random.default_rng(0)
    tensors = {}
    for i, block in enumerate(model.layers):
        for name in MODULES:
            out_dim, in_dim = block.proj[name].weight.shape
            down = rng.standard_normal((rank, in_dim), dtype=np.float32) * 0.1
            up = rng.standard_normal((out_dim, rank), dtype=np.float32) * 0.1
            if fmt == "kohya":
                base = f"diffusion_model.layers.{i}.proj.{name}"
                tensors[f"{base}.lora_down.weight"] = mx.array(down)
                tensors[f"{base}.lora_up.weight"] = mx.array(up)
                tensors[f"{base}.alpha"] = mx.array(alpha, dtype=mx.float32)
            else:
                base = f"transformer.layers.{i}.proj.{name}"
                tensors[f"{base}.lora_A.weight"] = mx.array(down)
                tensors[f"{base}.lora_B.weight"] = mx.array(up)
    mx.save_safetensors(str(path), tensors)
    return tensors


@pytest.fixture
def tiny():
    return MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16)


def test_load_lora_groups_keys(tmp_path, tiny):
    _write_lora(tmp_path / "a.safetensors", tiny)
    mods = load_lora(tmp_path / "a.safetensors")
    assert set(mods) == {f"layers.{i}.proj.{n}" for i in range(2) for n in MODULES}
    assert set(mods["layers.0.proj.q"]) == {"down", "up", "alpha"}


def test_unknown_key_rejected(tmp_path):
    mx.save_safetensors(str(tmp_path / "bad.safetensors"), {"something.else": mx.zeros((2, 2))})
    with pytest.raises(ValueError, match="unrecognized"):
        load_lora(tmp_path / "bad.safetensors")


@pytest.mark.parametrize("fmt", ["kohya", "diffusers"])
def test_apply_matches_formula(tmp_path, tiny, fmt):
    tensors = _write_lora(tmp_path / "a.safetensors", tiny, rank=4, alpha=2.0, fmt=fmt)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    before = tiny.layers[1].proj["o"](x)
    assert apply_lora(tiny, tmp_path / "a.safetensors", scale=0.5) == 12
    layer = tiny.layers[1].proj["o"]
    assert isinstance(layer, LoRALinear)
    prefix = "diffusion_model.layers.1.proj.o" if fmt == "kohya" else "transformer.layers.1.proj.o"
    down = tensors[f"{prefix}.lora_down.weight" if fmt == "kohya" else f"{prefix}.lora_A.weight"]
    up = tensors[f"{prefix}.lora_up.weight" if fmt == "kohya" else f"{prefix}.lora_B.weight"]
    alpha = 2.0 if fmt == "kohya" else 4.0  # diffusers files carry no alpha: defaults to rank
    expect = before + ((x.astype(mx.float32) @ down.T @ up.T) * (0.5 * alpha / 4)).astype(x.dtype)
    assert mx.allclose(layer(x), expect, atol=1e-2, rtol=1e-2).item()


def test_applies_on_quantized_layers(tmp_path, tiny):
    _write_lora(tmp_path / "a.safetensors", tiny)  # shapes from the bf16 layers, before packing
    quantize_dit(tiny, {"bits": 8, "group_size": 64})
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    before = tiny.layers[0].proj["q"](x)
    apply_lora(tiny, tmp_path / "a.safetensors")
    after = tiny.layers[0].proj["q"](x)
    assert after.shape == before.shape and not mx.allclose(after, before).item()


def test_stacking_and_zero_scale(tmp_path, tiny):
    _write_lora(tmp_path / "a.safetensors", tiny)
    _write_lora(tmp_path / "b.safetensors", tiny)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    before = tiny.layers[0].proj["q"](x)
    apply_lora(tiny, tmp_path / "a.safetensors", scale=1.0)
    apply_lora(tiny, tmp_path / "b.safetensors", scale=0.0)  # a no-op adapter on top
    layer = tiny.layers[0].proj["q"]
    assert isinstance(layer.base, LoRALinear)
    assert mx.allclose(layer(x), layer.base(x)).item()
    assert not mx.allclose(layer(x), before).item()


def test_cli_lora_spec():
    assert _lora_spec("a/b.safetensors") == ("a/b.safetensors", 1.0)
    assert _lora_spec("a/b.safetensors:0.7") == ("a/b.safetensors", 0.7)
    assert _lora_spec("/x y/c.safetensors:1") == ("/x y/c.safetensors", 1.0)
    assert _lora_spec("C:/odd:name.safetensors") == ("C:/odd:name.safetensors", 1.0)
