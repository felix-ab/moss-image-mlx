"""moss-mlx: MOSS-Image text-to-image on Apple Silicon (MLX).

Examples:
  moss-mlx "A red ceramic teapot on a wooden table" --model ~/models/MOSS-Image-SFT -o teapot.png
  moss-mlx "A poster titled HELLO" --model ... --seeds 1,2,3,4 -o poster.png      # one load, four images
  moss-mlx --parameters assets/images/005/parameters.json --model ... --transformer .../MOSS-Image-Base/transformer
  moss-mlx --batch jobs.jsonl --model ...      # one JSON object per line (Request fields, or {"parameters": path})
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from .pipeline import MOSSImagePipeline, Request, save

TIERS = {"8bit": "VisualInference/MOSS-Image-SFT-MLX-8bit", "compact": "VisualInference/MOSS-Image-SFT-MLX-compact"}


def default_model() -> str:
    """8bit on Macs with >= 24 GB of memory (9.2 GB peak at 1024^2), compact below (7.7 GB peak)."""
    import subprocess

    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    except Exception:
        mem = 0
    return TIERS["8bit"] if mem >= 24 * 1024**3 else TIERS["compact"]


def resolve_model(model: str | None) -> str:
    """A local directory, a tier name (8bit, compact), or a Hugging Face repo id (downloaded and cached)."""
    model = model or default_model()
    model = TIERS.get(model, model)
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser())
    import subprocess

    print(f"fetching {model} (cached after the first run)", file=sys.stderr)
    # Download in a child process: the Hub client keeps several GB of transfer buffers that this process would
    # otherwise still hold when the text encoder loads (measured +2.7 GB on the first run's peak).
    code = "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))"
    done = subprocess.run([sys.executable, "-c", code, model], stdout=subprocess.PIPE, text=True)
    if done.returncode:
        sys.exit(f"could not download {model}")
    return done.stdout.strip().splitlines()[-1]


def _layers(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split()] if str(text).strip() else []


def quant_spec(text: str | None):
    """'8' -> 8-bit g64 affine; '4g32' -> 4-bit g32; 'nvfp4' / 'mxfp8'; or a JSON spec."""
    if not text or text in ("bf16", "none"):
        return None
    if text.startswith("{"):
        return json.loads(text)
    if text == "nvfp4":
        return {"bits": 4, "group_size": 16, "mode": "nvfp4"}
    if text == "mxfp8":
        return {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    bits, _, group = text.partition("g")
    return {"bits": int(bits), "group_size": int(group or 64)}


def _lora_spec(text: str) -> tuple[str, float]:
    """'path.safetensors' or 'path.safetensors:0.7'."""
    path, sep, scale = text.rpartition(":")
    if sep and scale.replace(".", "", 1).isdigit():
        return path, float(scale)
    return text, 1.0


def _from_parameters(path) -> dict:
    params = json.load(open(path))["parameters"]
    return {"prompt": params["prompt"], "negative_prompt": params.get("negative_prompt", ""),
            "width": params["width"], "height": params["height"], "steps": params["steps"],
            "guidance": float(params["cfg"]), "shift": float(params["shift"]), "seed": params["seed"],
            "uncond_skip_layers": _layers(params.get("uncond_skip_layers", "22"))}


def main(argv=None):
    p = argparse.ArgumentParser(prog="moss-mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", help="Prompt text or layout JSON")
    p.add_argument("--model", help="Tier (8bit, compact), Hugging Face repo id or local directory. Default: 8bit on "
                                   ">= 24 GB Macs, compact below")
    p.add_argument("--transformer", help="Override the transformer directory (e.g. Base with the shared encoder)")
    p.add_argument("--parameters", help="Upstream parameters.json (official example format); CLI flags override it")
    p.add_argument("--batch", help="JSONL file of jobs; rendered with each model component loaded once")
    p.add_argument("--negative-prompt")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--shift", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--seeds", help="Comma-separated seeds: one image per seed, loaded once")
    p.add_argument("--uncond-skip-layers", help="e.g. '22'; empty string disables")
    p.add_argument("-o", "--output", default="moss.png", help="Output PNG (several images get -<n> suffixes)")
    p.add_argument("--dit-dtype", choices=("bfloat16", "float32"), help="Transformer precision (default: as stored)")
    p.add_argument("--dit-quant", help="Quantize the transformer at load: 8, 6, 4g32, nvfp4, mxfp8 or JSON")
    p.add_argument("--te-quant", help="Quantize the text encoder at load (same forms)")
    p.add_argument("--lora", action="append", default=[], metavar="PATH[:SCALE]",
                   help="LoRA safetensors (kohya or diffusers keys), applied unmerged; repeatable, e.g. --lora style.safetensors:0.8")
    p.add_argument("--precise", action="store_true", help="fp32 residual stream and elementwise ops (bf16 matmuls)")
    p.add_argument("--no-cache", action="store_true", help="Do not read or write the prompt-embedding cache")
    p.add_argument("--keep-resident", action="store_true", help="Do not free components between stages")
    args = p.parse_args(argv)
    args.model = resolve_model(args.model)

    base = _from_parameters(args.parameters) if args.parameters else {}
    for name in ("prompt", "negative_prompt", "width", "height", "steps", "guidance", "shift", "seed"):
        if getattr(args, name) is not None:
            base[name] = getattr(args, name)
    if args.uncond_skip_layers is not None:
        base["uncond_skip_layers"] = _layers(args.uncond_skip_layers)

    jobs = []
    if args.batch:
        for line in Path(args.batch).read_text().splitlines():
            if line.strip():
                job = json.loads(line)
                fields = _from_parameters(job.pop("parameters")) if "parameters" in job else {}
                out = job.pop("output", None)
                jobs.append((Request(**{**base, **fields, **job}), out))
    else:
        if "prompt" not in base:
            p.error("a prompt, --parameters or --batch is required")
        req = Request(**base)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [req.seed]
        jobs = [(replace(req, seed=s), None) for s in seeds]
    from .layout import check

    for r, _ in jobs:
        if r.prompt.lstrip().startswith("{"):
            for problem in check(r.prompt, r.width, r.height):
                print(f"layout warning: {problem}", file=sys.stderr)
    out = Path(args.output)
    jobs = [(r, o or (str(out) if len(jobs) == 1 else str(out.with_name(f"{out.stem}-{i}{out.suffix}"))))
            for i, (r, o) in enumerate(jobs)]

    import mlx.core as mx

    dtype = getattr(mx, args.dit_dtype) if args.dit_dtype else None
    pipe = MOSSImagePipeline(args.model, args.transformer, low_memory=not args.keep_resident, dit_dtype=dtype,
                             dit_quant=quant_spec(args.dit_quant), te_quant=quant_spec(args.te_quant),
                             cache_dir=False if args.no_cache else None, precise=args.precise,
                             loras=[_lora_spec(x) for x in args.lora])
    start = time.perf_counter()

    def progress(i, n):
        print(f"\rstep {i + 1}/{n}  {time.perf_counter() - start:7.1f} s", end="", file=sys.stderr, flush=True)

    paths = dict((id(r), o) for r, o in jobs)
    for req, image in pipe.generate_many([r for r, _ in jobs], on_step=progress):
        print(file=sys.stderr)
        save(image, req, paths[id(req)], {"timings": dict(pipe.timings), "precise": args.precise,
                                          "dit_quant": args.dit_quant or "bf16", "te_quant": args.te_quant or "bf16",
                                          "loras": args.lora})
        print(paths[id(req)])
    print(json.dumps(pipe.timings), file=sys.stderr)


if __name__ == "__main__":
    main()
