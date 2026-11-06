#!/usr/bin/env python3
"""Render one clip through the app's own MLX inference path (worker.runtime + worker.invoke):
offline weights, adapter-completeness guard, and the non-clipping float WAV writer.

Used for diagnostics and listening builds outside the app. Same fixed sampler as the app
(8 steps, CFG 1, APG 1, sigma_max 1, fp16). Writes <out> plus <out>.json with the settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_RUNTIME = ROOT / ".local-runtime/stable-audio-3/optimized/mlx"
DECODERS = {"sm-music": "same-s", "medium": "same-l"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--dit", choices=sorted(DECODERS), required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--trigger", default="", help="prepend '<trigger>, ' like the app does for adapters trained with one (e.g. rvkr for v5)")
    ap.add_argument("--profile", default="16gb", help="worker memory-limit profile only; does not change audio")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()  # sa3_mlx resolves relative paths under its own output/ dir
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    from ravi.local import worker
    worker.runtime({"runtime": str(a.runtime), "model_dir": str(a.model_dir), "profile": a.profile})
    prompt = f"{a.trigger}, {a.prompt}" if a.trigger and a.lora and a.strength > 0 else a.prompt
    args = ["--dit", a.dit, "--decoder", DECODERS[a.dit], "--prompt", prompt,
            "--seconds", a.seconds, "--seed", a.seed, "--steps", 8,
            "--cfg", 1.0, "--apg", 1.0, "--init-noise-level", 1.0, "--dit-dtype", "fp16",
            "--free-models", "--out", a.out]
    if a.lora:
        args += ["--lora", a.lora, "--lora-strength", a.strength]
    worker.invoke("sa3_mlx", args)
    a.out.with_suffix(a.out.suffix + ".json").write_text(json.dumps({
        "dit": a.dit, "decoder": DECODERS[a.dit], "prompt": a.prompt, "prompt_sent": prompt, "seconds": a.seconds,
        "seed": a.seed, "lora": str(a.lora) if a.lora else None,
        "strength": a.strength if a.lora else None, "inference_steps": 8, "cfg": 1.0,
        "apg": 1.0, "sigma_max": 1.0, "dit_dtype": "fp16", "runtime": str(a.runtime)}, indent=2))


if __name__ == "__main__":
    main()
