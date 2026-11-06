"""ACE-Step 1.5 engine for :mod:`ravi.eval.grid` (``engine: acestep`` in a grid config).

The grid drives every engine through the ``sa3_infer.Engine`` interface. This module provides
one for ACE-Step, so an ACE-Step grid runs the same regions, context windows, 50 ms in-mask
splice, stems, lower anchor and sidecars as the SA3 grids, and ``run_eval`` reads it unchanged::

    python -m ravi.eval.grid configs/eval/tier1_ace.yaml [--dry-run]     # in ravi-acestep-train

**Interface geometry.** The engine reports SA3 medium's geometry (44.1 kHz, 4096-sample latent
frames, ``sample_size`` 5292032) and resamples to and from ACE-Step's native 48 kHz inside each
call (polyphase, :func:`ravi.captions.make_acestep_dataset.resample`). Masks, contexts and
stems are therefore sample-identical to the SA3 tier-1 grid, which ``run_eval`` requires to
pair regions across conditions (``_check_sources``). ACE-Step's own latent hop (1920 samples at
48 kHz, 25 Hz) is recorded in ``engine.native``. The grid's
``effective_latent_mask_in_context`` block still describes SA3's 4096-sample frames and does
not apply to these outputs.

**Inpainting** = ACE-Step ``repaint`` (:meth:`ravi.eval.acestep_repaint.AceStepEngine.repaint`):
the 44.1 kHz context the grid passes is resampled to 48 kHz, written as a float32 WAV and
repainted over ``[mask_start_s, mask_end_s)``. Settings: balanced mode, strength 0.5, explicit
chunk mask, no output normalisation. ACE-Step splices the source back outside the mask itself
(25 ms ramps outside the mask), so the grid's ``_raw`` file is ACE-Step's spliced output after
the 48 -> 44.1 kHz round trip; its ``raw_vs_source_outside_mask`` is the resampling error plus
the ramps, not VAE error as for SA3. ``_spliced`` is the grid's own splice of the mask region
into the source context, identical to the SA3 grids.

**Generation** = ACE-Step ``text2music`` on the DiT only (no LM, no CoT, no normalisation) for
``generate.duration`` seconds.

**Prompts.** The grid renders each condition's caption (fields joined with ``", "``). A
trailing ``sections: a, b, c`` field (``with_sections: true``) is split off and sent as
ACE-Step lyrics, one ``[label]`` per line, which is how ``make_acestep_dataset`` writes arm E
for training. Without it the lyrics are ``[Instrumental]``. The caption and lyrics actually
sent are recorded in each sidecar (``engine_call``).

**Determinism.** Every call sets ``torch.manual_seed(seed)`` first (the source VAE encode samples
from the global RNG; ``acestep_repaint`` module docstring).

**Models / LoRA.** ``model: turbo`` (8 steps, shift 3.0, guidance 1.0: the served variant and
what the tier-1 LoRAs are trained on) or ``base`` (50 steps, shift 1.0, guidance 7.0). ``lora``
is the adapter *directory* (``adapter_config.json`` + ``adapter_model.safetensors``, e.g.
``/runs/acestep_B_tier1/model/adapter``). Strength goes through ``set_use_lora`` +
``set_lora_scale`` (0-1). A condition without a LoRA loads a fresh handler: ``unload_lora``
is broken at ``ca1e85fe``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ravi.eval import sa3_infer as si

ENGINE_NAME = "acestep"
INTERFACE_SR = 44100
NATIVE_SR = 48000
NATIVE_LATENT_HOP = 1920  # 48 kHz / 25 Hz latents
#: config dir, steps, cfg (guidance), shift ("Using Your Adapter.md:95-99"; turbo forces guidance 1.0)
MODELS: dict[str, dict[str, Any]] = {
    "turbo": {"config": "acestep-v15-turbo", "steps": 8, "cfg": 1.0, "shift": 3.0},
    "base": {"config": "acestep-v15-base", "steps": 50, "cfg": 7.0, "shift": 1.0},
}
REPAINT = {"mode": "balanced", "strength": 0.5}
SECTIONS_PREFIX = "sections:"
INSTRUMENTAL = "[Instrumental]"


class Backend(Protocol):
    """What the engine needs from ACE-Step (``acestep_repaint.AceStepEngine``); tests fake it."""

    def set_lora(self, scale: float | None) -> None: ...

    def repaint(self, *, src_path: str, start: float, end: float, caption: str, lyrics: str, seed: int,
                steps: int, shift: float, guidance: float, mode: str, strength: float) -> np.ndarray: ...

    def text2music(self, *, caption: str, lyrics: str, duration: float, seed: int, steps: int, shift: float,
                   guidance: float) -> np.ndarray: ...


def split_sections(prompt: str) -> tuple[str, str]:
    """``"rvkr, a caption, Rock, sections: intro, verse"`` -> ``("rvkr, a caption, Rock",
    "[intro]\\n[verse]")``. The grid appends the sections field last; no field -> ``[Instrumental]``."""
    from ravi.captions.make_acestep_dataset import section_lyrics

    p = " ".join(prompt.split())
    low = p.lower()
    if low.startswith(SECTIONS_PREFIX):
        return "", section_lyrics(p)
    i = low.rfind(", " + SECTIONS_PREFIX)
    if i < 0:
        return p, INSTRUMENTAL
    return p[:i], section_lyrics(p[i + 2:])


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from ravi.captions.make_acestep_dataset import resample as poly

    return np.clip(poly(np.asarray(audio, dtype=np.float64), sr_in, sr_out), -1.0, 1.0).astype(np.float32)


def adapter_info(path: str | Path | None) -> dict[str, Any] | None:
    """sha256 of ``adapter_model.safetensors``, ``adapter_config.json`` and, when the adapter
    sits in a ``ravi.train.acestep_lora`` model dir, the run identity from ``../run_meta.json``."""
    if path is None:
        return None
    p = Path(path)
    w = p / "adapter_model.safetensors"
    info: dict[str, Any] = {"path": str(p), "sha256": si.sha256_file(w), "bytes": w.stat().st_size}
    try:
        info["adapter_config"] = json.loads((p / "adapter_config.json").read_text())
    except (OSError, ValueError) as e:
        info["adapter_config_error"] = repr(e)
    rm = p.parent / "run_meta.json"
    if rm.is_file():
        try:
            m = json.loads(rm.read_text())
            info["ravi_run"] = {k: m.get(k) for k in ("run_id", "arm", "status", "acestep_commit", "identity",
                                                      "epoch_plan", "ravi_git_sha")}
        except ValueError:
            pass
    return info


def is_adapter_dir(path: str | Path) -> bool:
    p = Path(path)
    return (p / "adapter_config.json").is_file() and (p / "adapter_model.safetensors").is_file()


class AceGridEngine:
    """``sa3_infer.Engine``-compatible ACE-Step engine (module docstring)."""

    def __init__(self, backend: Backend, model_name: str, lora_path: str | None = None,
                 device: str | None = None, tmp_dir: str | None = None):
        if model_name not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}")
        self.backend = backend
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device
        self.sample_rate = INTERFACE_SR
        self.sample_size = si.DEFAULT_SAMPLE_SIZE
        self.downsampling_ratio = 4096
        self.channels = 2
        self.strength: float | None = 1.0 if lora_path else None
        self.last_call: dict[str, Any] | None = None
        self._tmp = Path(tmp_dir or tempfile.mkdtemp(prefix="acestep_grid_"))
        self._tmp.mkdir(parents=True, exist_ok=True)
        if lora_path:
            self.backend.set_lora(1.0)
        else:
            self.backend.set_lora(None)

    @classmethod
    def load(cls, model_name: str, lora_path: str | None, device: str | None = None) -> "AceGridEngine":
        from ravi.eval.acestep_repaint import AceStepEngine

        if model_name not in MODELS:
            raise SystemExit(f"acestep model must be one of {sorted(MODELS)}")
        if lora_path and not is_adapter_dir(lora_path):
            raise SystemExit(f"{lora_path}: not a PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
        print(f"[acestep_grid] loading {model_name} lora={lora_path}", flush=True)
        backend = AceStepEngine(model_name, lora_path, device or "cuda")
        return cls(backend, model_name, lora_path, device=device or "cuda")

    @property
    def max_context_samples(self) -> int:  # same as sa3_infer.Engine, so contexts match the SA3 grids
        n = self.sample_size - int(si.DURATION_PADDING_S * self.sample_rate)
        return (n // self.downsampling_ratio) * self.downsampling_ratio

    def set_lora_strength(self, strength: float) -> None:
        if not self.lora_path:
            raise ValueError("no LoRA loaded")
        s = float(strength)
        if not 0.0 <= s <= 1.0:
            raise ValueError("ACE-Step clamps the LoRA scale to [0, 1] (lora/controls.py:107)")
        self.backend.set_lora(s)
        self.strength = s

    def generate(self, *, prompt: str, duration: float, steps: int, cfg_scale: float, seed: int,
                 inpaint_audio: np.ndarray | None = None, mask_start_s: float | None = None,
                 mask_end_s: float | None = None) -> np.ndarray:
        caption, lyrics = split_sections(prompt)
        shift = float(MODELS[self.model_name]["shift"])
        call: dict[str, Any] = {"caption": caption, "lyrics": lyrics, "seed": int(seed), "steps": int(steps),
                                "guidance": float(cfg_scale), "shift": shift, "lora_scale": self.strength,
                                "native_sample_rate": NATIVE_SR, "torch_manual_seed": int(seed)}
        if inpaint_audio is not None:
            x = resample(inpaint_audio, INTERFACE_SR, NATIVE_SR)
            src = self._tmp / f"ctx_{os.getpid()}.wav"
            import soundfile as sf

            sf.write(src, x, NATIVE_SR, subtype="FLOAT")
            call.update(task="repaint", repaint_mode=REPAINT["mode"], repaint_strength=REPAINT["strength"],
                        repainting_start=float(mask_start_s), repainting_end=float(mask_end_s),
                        native_source_samples=int(x.shape[0]))
            y = self.backend.repaint(src_path=str(src), start=float(mask_start_s), end=float(mask_end_s),
                                     caption=caption, lyrics=lyrics, seed=int(seed), steps=int(steps), shift=shift,
                                     guidance=float(cfg_scale), mode=REPAINT["mode"], strength=REPAINT["strength"])
        else:
            call.update(task="text2music", duration=float(duration))
            y = self.backend.text2music(caption=caption, lyrics=lyrics, duration=float(duration), seed=int(seed),
                                        steps=int(steps), shift=shift, guidance=float(cfg_scale))
        call["native_output_samples"] = int(y.shape[0])
        self.last_call = call
        return resample(y, NATIVE_SR, INTERFACE_SR)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "engine": ENGINE_NAME,
            "model": self.model_name,
            "model_config": MODELS[self.model_name]["config"],
            "sample_rate": self.sample_rate,
            "sample_size": self.sample_size,
            "downsampling_ratio": self.downsampling_ratio,
            "channels": self.channels,
            "native": {"sample_rate": NATIVE_SR, "latent_hop": NATIVE_LATENT_HOP,
                       "resampling": "scipy resample_poly 160/147 in, 147/160 out"},
            "repaint": dict(REPAINT),
            "device": self.device,
            "acestep_commit": os.environ.get("ACESTEP_COMMIT"),
            "ravi_git_sha": os.environ.get("RAVI_GIT_SHA"),
            "ravi_image_tag": os.environ.get("RAVI_IMAGE_TAG"),
        }
        try:
            import torch

            d["torch"] = torch.__version__
            if torch.cuda.is_available():
                d["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return d
