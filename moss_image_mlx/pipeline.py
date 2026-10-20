"""MOSS-Image text-to-image pipeline in MLX.

Follows the official MOSSImagePipeline.__call__: fp32 starting noise from torch's CPU generator (so official seeds
reproduce), sigmas snapped to the training grid, CFG and Euler updates in fp32, block skipping in the negative
branch, optional skip-layer guidance, fp32 VAE decode.

Memory lifecycle (low_memory=True): text encoder -> encode positive and negative -> free; transformer -> denoise ->
free; VAE decoder -> decode. Only one large component is resident at a time.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .dit import load_transformer
from .quantize import describe, quantize_dit, quantize_text_encoder
from .scheduler import sigmas_for, timesteps_for
from .text_encoder import TextEncoder
from .torch_randn import randn
from .vae import load_vae_decoder, to_uint8


@dataclass
class Request:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 50
    guidance: float = 3.0
    shift: float = 5.0
    seed: int = 42
    uncond_skip_layers: list[int] = field(default_factory=lambda: [22])
    slg_layers: list[int] = field(default_factory=list)
    slg_scale: float = 0.0
    slg_start: float = 0.0
    slg_stop: float = 1.0
    max_sequence_length: int = 4096

    def validate(self, depth: int):
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("width and height must be positive multiples of 16")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        if not 0 <= self.slg_start <= self.slg_stop <= 1:
            raise ValueError("require 0 <= slg_start <= slg_stop <= 1")
        for layers in (self.uncond_skip_layers, self.slg_layers):
            if any(type(i) is not int or not 0 <= i < depth for i in layers):
                raise ValueError(f"skip layers must be integer block indices in [0, {depth})")


def _footprint() -> int | None:
    """This process's physical footprint in bytes (macOS proc_pid_rusage), or None if unavailable."""
    try:
        import ctypes
        import os

        buf = (ctypes.c_uint64 * 64)()
        if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.cast(buf, ctypes.c_void_p)):
            return None
        return int(buf[2 + 7])
    except Exception:
        return None


def _free(timeout: float = 3.0):
    """Release dropped MLX buffers and wait until the OS has actually reclaimed them.

    After MLX frees a large component, macOS lowers the process footprint asynchronously (~1 s). If the next stage
    allocates immediately, both are counted at once and the staged peak becomes their sum. Wait (bounded) for at
    least 80% of the released bytes to leave the footprint first."""
    held = mx.get_active_memory() + mx.get_cache_memory()
    start = _footprint()
    gc.collect()
    mx.clear_cache()
    released = held - (mx.get_active_memory() + mx.get_cache_memory())
    if start is None or released < (1 << 30):
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        now = _footprint()
        if now is None or now <= start - 0.8 * released:
            return
        time.sleep(0.05)


def _sample(model, req: Request, positive, negative, latents, on_step):
    req.validate(len(model.layers))
    if latents is None:
        latents = mx.array(randn((1, model.config["latent_channels"], req.height // 16, req.width // 16), req.seed))
    sigmas = sigmas_for(req.steps, req.shift)
    timesteps = timesteps_for(sigmas)
    n = len(timesteps)
    do_cfg = req.guidance != 1.0 and negative is not None
    for i, t in enumerate(timesteps):
        skip_active = req.slg_start <= i / n < req.slg_stop
        cond = model(latents, float(t), positive).astype(mx.float32)
        prediction = cond
        if do_cfg:
            skip = req.uncond_skip_layers if (req.uncond_skip_layers and skip_active) else ()
            uncond = model(latents, float(t), negative, skip).astype(mx.float32)
            prediction = uncond + req.guidance * (cond - uncond)
        if req.slg_layers and req.slg_scale > 0 and skip_active:
            skipped = model(latents, float(t), positive, req.slg_layers).astype(mx.float32)
            if not do_cfg:
                prediction = skipped
            prediction = prediction + req.slg_scale * (cond - skipped)
        dt = mx.array(sigmas[i + 1] - sigmas[i], dtype=mx.float32)  # float32, as torch computes it
        latents = latents.astype(mx.float32) + dt * prediction
        mx.eval(latents)
        if on_step:
            on_step(i, n)
    return latents


class MOSSImagePipeline:
    def __init__(self, model_dir: str | Path, transformer_dir: str | Path | None = None, low_memory: bool = True,
                 dit_dtype=None, dit_quant: dict | None = None, te_quant: dict | None = None,
                 cache_dir: str | Path | None | bool = None, precise: bool = False,
                 loras: list[tuple[str | Path, float]] = ()):
        """model_dir: a diffusers-layout MOSS-Image directory (transformer/, text_encoder/, tokenizer/, vae/).
        transformer_dir overrides the transformer only (Base and SFT share the text encoder and VAE).
        dit_quant / te_quant: quantization specs applied at load (see quantize.py).
        cache_dir: prompt-embedding cache location (None = default, False = disabled).
        precise: keep the DiT's residual stream and elementwise chains in fp32 (matmuls stay in the weight dtype)."""
        self.model_dir = Path(model_dir)
        self.transformer_dir = Path(transformer_dir) if transformer_dir else self.model_dir / "transformer"
        self.low_memory = low_memory
        self.dit_dtype = dit_dtype  # None keeps the checkpoint's bf16; mx.float32 gives an fp32 reference run
        self.dit_quant, self.te_quant, self.precise = dit_quant, te_quant, precise
        self.loras = list(loras)  # [(path, scale)], applied in order, unmerged
        self.text_encoder = self.transformer = self.vae = None
        self.cache = None
        if cache_dir is not False:
            from .cache import PromptCache

            self.cache = PromptCache(self.model_dir / "text_encoder", cache_dir, tag=describe(te_quant))
        self.timings: dict = {}

    # Components -------------------------------------------------------------------------------------------------
    def _load_text_encoder(self):
        if self.text_encoder is None:
            self.text_encoder = TextEncoder(self.model_dir / "text_encoder", self.model_dir / "tokenizer")
            quantize_text_encoder(self.text_encoder.model, self.te_quant)

    def _load_transformer(self):
        if self.transformer is None:
            model = load_transformer(self.transformer_dir, self.dit_dtype)
            quantize_dit(model, self.dit_quant)
            if self.loras:
                from .lora import apply_lora

                for path, scale in self.loras:
                    apply_lora(model, path, scale)
            model.precise = self.precise
            self.transformer = model

    def _load_vae(self):
        if self.vae is None:
            self.vae = load_vae_decoder(self.model_dir / "vae")

    def free_text_encoder(self):
        self.text_encoder = None
        _free()

    def free_transformer(self):
        self.transformer = None
        _free()

    def encode(self, prompts: list[str], max_tokens: int, keep: bool = False) -> list[mx.array]:
        """Embeddings for each prompt; cached prompts never load the text encoder."""
        out = [self.cache.get(p, max_tokens) if self.cache else None for p in prompts]
        missing = [i for i, e in enumerate(out) if e is None]
        if missing:
            self._load_text_encoder()
            for i in missing:
                out[i] = self.text_encoder.encode(prompts[i], max_tokens)
                if self.cache:
                    self.cache.put(prompts[i], max_tokens, out[i])
            if self.low_memory and not keep:
                self.free_text_encoder()
        self.timings["prompts_encoded"] = len(missing)
        return out

    # Generation -------------------------------------------------------------------------------------------------
    def denoise(self, req: Request, positive: mx.array, negative: mx.array | None, latents: mx.array | None = None,
                on_step=None, keep: bool = False) -> mx.array:
        self._load_transformer()
        latents = _sample(self.transformer, req, positive, negative, latents, on_step)
        if self.low_memory and not keep:  # _sample has returned, so no local reference keeps the weights alive
            self.free_transformer()
        return latents

    def decode(self, latents: mx.array, keep: bool = False) -> np.ndarray:
        self._load_vae()
        # The strip decoder's working set is ~2.7 GB at 1024^2, but its many differently sized temporaries would
        # otherwise pile up in MLX's free-buffer cache (measured 16 GB). Cap the cache while decoding (+~1.5 s).
        previous = mx.set_cache_limit(1 << 30)
        try:
            image = self.vae.decode_packed(latents)
            mx.eval(image)
        finally:
            mx.set_cache_limit(previous)
            mx.clear_cache()
        out = to_uint8(image[0])
        if self.low_memory and not keep:
            self.vae = None
            _free()
        return out

    def __call__(self, req: Request, on_step=None) -> np.ndarray:
        return next(iter(self.generate_many([req], on_step=on_step)))[1]

    def generate_many(self, requests: list[Request], on_step=None):
        """Yield (request, image) for each request, loading each component once.

        All prompts are encoded first (the text encoder loads only for prompts not already cached), then the
        transformer stays resident for every request and decodes alternate with denoising. With low_memory the
        encoder is freed before the transformer loads, so the peak is max(encoder, transformer + VAE decode)."""
        t0 = time.perf_counter()
        texts, index = [], {}
        for r in requests:
            for t in ([r.prompt] + ([r.negative_prompt] if r.guidance != 1.0 else [])):
                if (t, r.max_sequence_length) not in index:
                    index[(t, r.max_sequence_length)] = len(texts)
                    texts.append((t, r.max_sequence_length))
        embeds = []
        for limit in sorted({m for _, m in texts}):
            group = [t for t, m in texts if m == limit]
            embeds += list(zip([(t, limit) for t in group], self.encode(group, limit, keep=True)))
        embeds = dict(embeds)
        if self.low_memory and self.text_encoder is not None:
            self.free_text_encoder()
        t1 = time.perf_counter()
        self.timings = {"encode_s": round(t1 - t0, 2), "prompts_encoded": sum(1 for _ in texts)}
        for n, r in enumerate(requests):
            ts = time.perf_counter()
            pos = embeds[(r.prompt, r.max_sequence_length)]
            neg = embeds[(r.negative_prompt, r.max_sequence_length)] if r.guidance != 1.0 else None
            last = n == len(requests) - 1
            latents = self.denoise(r, pos, neg, on_step=on_step, keep=not last)
            td = time.perf_counter()
            image = self.decode(latents, keep=not last)
            te_ = time.perf_counter()
            self.timings.update({"denoise_s": round(td - ts, 2), "decode_s": round(te_ - td, 2),
                                 "total_s": round(te_ - t0, 2), "prompt_tokens": int(pos.shape[1]),
                                 "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)})
            yield r, image


def save(image: np.ndarray, req: Request, path: str | Path, extra: dict | None = None):
    """Write the PNG and an upstream-style parameters.json next to it."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    record = {"schema_version": 2, "parameters": {**asdict(req), "cfg": req.guidance}, **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
