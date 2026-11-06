"""Explicit capability declarations, not promises inferred from installed RAM."""
from __future__ import annotations

import platform
from pathlib import Path

UPSTREAM_COMMIT = "77e0398c26f7d22d801ec1bb036307e8b231f741"
# crop_seconds: the window the trainer samples from each prepared example (random crop).
# example_max_seconds: prepared examples are whole described regions up to this length.
PROFILES = {
    "8gb": {"label": "8 GB · lighter workload", "crop_seconds": 15, "max_seconds": 30,
            "example_max_seconds": 120, "rank": 8, "memory_gb": 4.5},
    "16gb": {"label": "16 GB+ · larger workload", "crop_seconds": 30, "max_seconds": 60,
             "example_max_seconds": 120, "rank": 16, "memory_gb": 10},
    # The cloud recipe's 47 s window: most pieces in a corpus of ~30 s spots fit whole,
    # so training sees real endings with a truthful duration token.
    "32gb": {"label": "32 GB+ · full-length workload", "crop_seconds": 47, "max_seconds": 60,
             "example_max_seconds": 120, "rank": 16, "memory_gb": 24},
}
SAMPLES_PER_LATENT = 4096


def crop_latents(profile: str) -> int:
    """Trainer window in whole latent frames, so every trained frame is real audio."""
    return int(PROFILES[profile]["crop_seconds"] * 44100) // SAMPLES_PER_LATENT


MODELS = {
    "sa3-small": {"name": "Stable Audio 3 Small", "dit": "sm-music", "codec": "same-s",
                  "codec_key": "s", "minimum_profile": "8gb", "download_gb": 2.8},
    "sa3-medium": {"name": "Stable Audio 3 Medium", "dit": "medium", "codec": "same-l",
                   "codec_key": "l", "minimum_profile": "16gb", "download_gb": 9.5},
}

def weight_files(model_id: str) -> list[str]:
    m = MODELS[model_id]
    return [f"dit_{m['dit']}_f16.npz", f"dit_{m['dit']}-base_f16.npz",
            f"same_{m['codec_key']}_encoder_f32.npz", f"same_{m['codec_key']}_decoder_f32.npz",
            "t5gemma_f16.npz"]

def hardware() -> dict:
    import psutil
    return {"system": platform.system(), "machine": platform.machine(),
            "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "available_gb": round(psutil.virtual_memory().available / 2**30, 1),
            "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64"}

def catalog(model_dir: Path, runtime: Path | None) -> list[dict]:
    hw = hardware()
    result = []
    for key, model in MODELS.items():
        missing = [f for f in weight_files(key) if not (model_dir / f).is_file()]
        reason = ("Native MLX needs Apple Silicon." if not hw["apple_silicon"] else
                  "The bundled MLX runtime is missing." if not runtime or not runtime.is_dir() else
                  "Download this model pack in Setup." if missing else None)
        result.append({"id": key, **model, "ready": reason is None, "reason": reason,
                       "missing": missing, "validation": "Experimental · physical 8 GB validation pending",
                       "capabilities": ["prepare", "train", "generate", "edit"] if reason is None else []})
    result.append({"id": "ace-step", "name": "ACE-Step 1.5", "ready": False,
                   "reason": "Mac training and adapter quality validation pending; not enabled in this pilot.",
                   "capabilities": [], "validation": "Not validated"})
    return result
