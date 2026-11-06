"""Stable Audio 3 inference for the inpainting go/no-go (plan Phase 2.3): text-to-audio,
section inpainting and continuation, with or without a rAvI LoRA adapter.

Runs in the ``ravi-sa3-train`` image (``--entrypoint python ... -m ravi.eval.sa3_infer``)
against the pinned upstream checkout (SA3 commit 77e0398c). Subcommands::

    generate --prompt "..." [--prompt-file captions.txt] --duration 30 --seed 0 --out DIR
    inpaint  --audio TRACK.wav --segments TRACK.json (--section-index I | --all-interior)
             --pad-s 1.0 --prompt "..." --seed 0 --out DIR
    continue --audio TRACK.wav --start 60 --pre-s 30 --gen-s 15 --prompt "..." --out DIR

Common options: ``--model medium|medium-base`` (default ``medium``, the post-trained
checkpoint; LoRAs are trained on ``medium-base`` and applied to the post-trained model,
``docs/guides/model-overview.md:129``), ``--lora adapter.safetensors``,
``--lora-strength 1.0 [0.5 0 ...]`` (several values reuse the loaded model and write into
``DIR/strength{v}/``), ``--steps`` / ``--cfg`` (default 8 / 1.0 for ``medium``, 50 / 7.0
for ``medium-base``: ``cli.py:77, 80-84``, ``docs/workflows/inference.md:52, 56-57``).

Upstream calls (``stable_audio_3`` at 77e0398c):

* ``StableAudioModel.from_pretrained(name, device=None, model_half=True)``
  (``model.py:30-62``): ``medium`` / ``medium-base`` resolve to the HF repos
  ``stabilityai/stable-audio-3-medium[-base]`` (``model_configs.py:75-84``) through
  ``hf_hub_download`` (``model_configs.py:12-16``), which reads ``HF_TOKEN`` and ``HF_HOME``
  from the environment itself; this module never handles the token.
* ``model.load_lora([path])`` (``model.py:64-70`` -> ``lora/loader.py:35-140``): reads
  ``lora_config`` (rank, alpha, adapter type, include/exclude) from the safetensors
  metadata (``lora/utils.py:287-302``) and parametrizes the DiT and the conditioner.
* ``model.set_lora_strength(s)`` (``model.py:72-74``): fills every ``lora_strength`` buffer
  of the DiT and the conditioner (``lora/model.py:528-538``). Plain LoRA computes
  ``W + (alpha/rank) * s * BA`` (``lora/model.py:239-242``); DoRA-rows scales the direction
  update by ``s`` and interpolates the learned magnitude toward ``||W0||``
  (``lora/model.py:184-200, 244-260``); ``s == 0`` returns ``W0`` exactly
  (``lora/model.py:245-246``). The buffer starts at 1.0 (``lora/model.py:42``); we always
  set it explicitly.
* ``model.generate(prompt=, duration=, steps=, cfg_scale=, seed=, batch_size=1,
  sample_size=model_config["sample_size"], inpaint_audio=(sample_rate, waveform),
  inpaint_mask_start_seconds=, inpaint_mask_end_seconds=)`` (``model.py:77-109``; the CLI
  passes the same, ``cli.py:270-288``). The tuple is **(sample_rate, waveform)**
  (``model.py:438``, ``cli.py:247``); ``torchaudio.load`` returns the reverse, so the
  docs' ``inpaint_audio = torchaudio.load(...)`` (``docs/workflows/inference.md:92``) is
  the wrong order. The waveform is a ``(channels, samples)`` tensor (``audio_utils.py:65-83``).
  The mask is 1 = keep, 0 = regenerate, built in samples with ``int(sec * sr)``
  (``model.py:203-213``) and downsampled to latent frames with nearest interpolation
  (``model.py:457-462``); we pass ``(n + 0.5) / sr`` so the truncation lands on sample
  ``n``. Output: ``float32 (batch, channels, samples)`` clamped to [-1, 1], truncated to
  ``int(duration * sr)`` samples (``model.py:350-356``); read the rate from
  ``model.model.sample_rate`` (``cli.py:290``).

Inpainting (``inpaint`` and ``continue``): the track is resampled to the model rate and
made stereo if needed (``--audio`` may be any rate/channel count); a context window around
the mask is cut from it (``--context-s``, default 47 s, the LoRA's training window, grown
to keep ``--min-context-s`` on both sides; ``0`` = whole track) and passed with
``duration = context length``. The context start is moved so the mask starts on a latent
frame boundary when possible. Per mask, in ``DIR``:

* ``{stem}_raw.wav``: the model output for the context window, as returned;
* ``{stem}_spliced.wav``: the source context with only ``[mask start, mask end)`` taken
  from the raw output and 50 ms raised-cosine crossfades *inside* the mask, so every sample
  outside the mask is bit-identical to the (resampled) source; checked after writing;
* ``{stem}_orig.wav``: the source context itself (upper anchor: the original region
  re-inserted is the identity);
* ``{stem}.json``: section and mask bounds (track seconds), context bounds, mask relative
  to the context, the latent-frame region the model actually regenerated, the VAE
  round-trip error outside the mask, all parameters and the adapter's ``ravi_*`` metadata.

``stem`` is ``{track}_sec{i}_{label}_{start:.1f}-{end:.1f}`` (padded mask, track seconds,
``i`` = index in the segments JSON) for ``inpaint`` and ``{track}_cont_{start:.1f}-{end:.1f}``
for ``continue``. All WAVs are 32-bit float.

``continue`` regenerates ``[start, start + gen_s)`` from ``pre_s`` seconds of context. The
model sees nothing after ``start`` (zeros); only the 50 ms fade-in of ``_spliced.wav``
mixes in the real audio after ``start`` when the track has it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ravi.eval.masks import (
    DEFAULT_MIN_LEN_S,
    DEFAULT_PAD_S,
    DEFAULT_XFADE_S,
    SectionMask,
    mask_summary,
    section_masks,
    seconds_to_samples,
    splice,
    upper_anchor,
)
from ravi.train.caption_aug import FIELD_SEP, JOIN_SEP, join_fields, split_fields

MODELS = ("medium", "medium-base")
MODEL_DEFAULTS = {
    "medium": {"steps": 8, "cfg": 1.0},  # post-trained: cli.py:77, inference.md:52
    "medium-base": {"steps": 50, "cfg": 7.0},  # inference.md:52, cli.py:83 "try 7.0 for base models"
}
DEFAULT_SAMPLE_SIZE = 5292032  # model.py:87, 120 s at 44.1 kHz; the model config's value wins
DURATION_PADDING_S = 6.0  # model.py:103 duration_padding_sec default
DEFAULT_CONTEXT_S = 47.0  # the LoRA training window (sa3_lora duration 47)
DEFAULT_MIN_CONTEXT_S = 5.0
SLUG_LEN = 40


# --------------------------------------------------------------------------- naming


def render_prompt(prompt: str) -> str:
    """Join ``␞``-separated caption fields with ``", "`` as the trainer shows them to the
    model (``caption_aug.join_sep``); a prompt without ``␞`` is only stripped."""
    if FIELD_SEP in prompt:
        return join_fields(split_fields(prompt, FIELD_SEP), JOIN_SEP)
    return " ".join(prompt.split())


def slugify(prompt: str, maxlen: int = SLUG_LEN) -> str:
    """``{first maxlen chars, [a-z0-9-]}-{sha1(prompt)[:6]}``: readable, collision-free."""
    base = re.sub(r"[^a-z0-9]+", "-", render_prompt(prompt).lower()).strip("-")[:maxlen].rstrip("-")
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:6]
    return f"{base or 'prompt'}-{digest}"


def generate_stem(seed: int, prompt: str) -> str:
    return f"{seed}_{slugify(prompt)}"


def safe_track(track: str) -> str:
    return re.sub(r"[\\/\x00]+", "_", str(track)).strip() or "track"


def inpaint_stem(track: str, index: int, label: str, start: float, end: float) -> str:
    return f"{safe_track(track)}_sec{index}_{label}_{start:.1f}-{end:.1f}"


def continue_stem(track: str, start: float, end: float) -> str:
    return f"{safe_track(track)}_cont_{start:.1f}-{end:.1f}"


def strength_dirname(strength: float) -> str:
    return f"strength{float(strength):g}"


def read_prompts(prompts: Sequence[str] | None, prompt_file: str | None) -> list[str]:
    """``--prompt`` values, then ``--prompt-file`` lines (blank lines and ``#`` comments
    skipped). Duplicates are an error: they would write the same file."""
    out = [p for p in (prompts or [])]
    if prompt_file:
        for line in Path(prompt_file).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                out.append(line.rstrip("\r\n"))
    out = [p for p in out if p.strip()]
    if not out:
        raise SystemExit("no prompts: pass --prompt and/or --prompt-file")
    seen: set[str] = set()
    for p in out:
        if p in seen:
            raise SystemExit(f"duplicate prompt: {p!r}")
        seen.add(p)
    return out


# --------------------------------------------------------------------------- adapter / env


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def adapter_info(path: str | Path | None) -> dict[str, Any] | None:
    """sha256, size, ``lora_config`` and the trainer's ``ravi_*`` metadata keys
    (``ravi.train.sa3_lora.identity_metadata``) of an adapter, read from the header only."""
    if path is None:
        return None
    from ravi.train.sa3_lora import read_safetensors_metadata

    p = Path(path)
    md = read_safetensors_metadata(p)
    ravi = {k: v for k, v in md.items() if k.startswith("ravi_")}
    if "ravi_identity" in ravi:
        try:
            ravi["ravi_identity"] = json.loads(ravi["ravi_identity"])
        except ValueError:
            pass
    try:
        lora_config = json.loads(md["lora_config"]) if "lora_config" in md else None
    except ValueError:
        lora_config = md["lora_config"]
    if not ravi:
        print(f"[sa3_infer] WARNING: {p} has no ravi_* metadata (not written by ravi.train.sa3_lora?)",
              flush=True)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "lora_config": lora_config,
        "ravi_metadata": ravi,
    }


def hf_env() -> dict[str, Any]:
    """What the HF hub client will use; the token value is never read or recorded."""
    return {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
    }


def sa3_commit() -> str | None:
    if os.environ.get("SA3_COMMIT"):
        return os.environ["SA3_COMMIT"]
    p = Path(os.environ.get("SA3_ROOT", "/opt/sa3")) / ".sa3_commit"
    return p.read_text().strip() if p.is_file() else None


# --------------------------------------------------------------------------- audio


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """``(samples, channels)`` float32 and the file's rate. soundfile first (WAV/FLAC and,
    with libsndfile >= 1.1, MP3), torchaudio as a fallback."""
    import soundfile as sf

    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio), int(sr)
    except Exception as e:  # noqa: BLE001 - try the other decoder
        try:
            import torchaudio
        except ImportError:
            raise e from None
        wav, sr = torchaudio.load(str(path))  # (channels, samples), sr: the reverse of generate()'s tuple
        return np.ascontiguousarray(wav.numpy().T.astype(np.float32)), int(sr)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(np.ascontiguousarray(audio.T))
        return np.ascontiguousarray(AF.resample(t, sr_in, sr_out).numpy().T.astype(np.float32))
    except ImportError:
        from scipy.signal import resample_poly

        g = math.gcd(sr_in, sr_out)
        return np.ascontiguousarray(resample_poly(audio, sr_out // g, sr_in // g, axis=0).astype(np.float32))


def to_model_audio(audio: np.ndarray, sr: int, target_sr: int, channels: int = 2) -> tuple[np.ndarray, dict]:
    """Resample to ``target_sr`` and convert channels like upstream's ``set_audio_channels``
    (``audio_utils.py:49-63``: mono duplicated, >2 channels -> the first two)."""
    info = {"input_sr": int(sr), "input_channels": int(audio.shape[1]), "resampled": sr != target_sr}
    a = audio.astype(np.float32, copy=False)
    if channels == 2 and a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    elif a.shape[1] > channels:
        a = a[:, :channels]
    elif channels == 1 and a.shape[1] > 1:
        a = a.mean(axis=1, keepdims=True)
    a = resample(np.ascontiguousarray(a), sr, target_sr)
    info.update(model_sr=int(target_sr), channels=int(a.shape[1]), n_samples=int(a.shape[0]))
    return np.ascontiguousarray(a, dtype=np.float32), info


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sf.write(str(tmp), np.ascontiguousarray(audio, dtype=np.float32), sr, subtype="FLOAT", format="WAV")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    os.replace(tmp, path)


def fix_length(audio: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Crop or zero-pad to ``n`` samples; returns the array and the padding added."""
    if audio.shape[0] >= n:
        return audio[:n], 0
    pad = n - audio.shape[0]
    return np.concatenate([audio, np.zeros((pad, audio.shape[1]), dtype=audio.dtype)]), pad


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    if a.size == 0:
        return {"max_abs": None, "rms": None}
    d = a.astype(np.float64) - b.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(d))), "rms": float(np.sqrt(np.mean(d * d)))}


# --------------------------------------------------------------------------- context


def choose_context(
    m0: int,
    m1: int,
    n_total: int,
    sr: int,
    context_s: float,
    min_context_s: float,
    max_samples: int,
    align: int = 1,
) -> tuple[int, int]:
    """Context window ``[c0, c1)`` (samples) containing the mask ``[m0, m1)``.

    Length: ``context_s`` (``<= 0``: the whole track), at least the mask plus
    ``min_context_s`` on each side, at most ``max_samples`` and the track. Centred on the
    mask, shifted to stay inside the track, then ``c0`` is moved down (or, at the track
    start, up by less than ``align``) so that ``m0 - c0`` is a multiple of ``align``: the
    mask then starts exactly on a latent frame.
    """
    mask_len = m1 - m0
    if mask_len <= 0 or m0 < 0 or m1 > n_total:
        raise ValueError(f"bad mask [{m0}, {m1}) for {n_total} samples")
    if context_s <= 0:
        want = n_total
    else:
        want = max(seconds_to_samples(context_s, sr), mask_len + 2 * seconds_to_samples(min_context_s, sr))
    want = min(want, n_total, max_samples)
    if want < mask_len:
        raise ValueError(
            f"mask of {mask_len / sr:.1f} s does not fit the model's {max_samples / sr:.1f} s context"
        )
    side = (want - mask_len) // 2
    c0 = max(0, min(m0 - side, n_total - want))
    c1 = c0 + want
    if align > 1 and (m0 - c0) % align:
        k = -(-(m0 - c0) // align)  # ceil
        a0 = m0 - k * align
        if a0 >= 0:  # move the start down by < align; keep the mask inside
            a1 = max(a0 + want, m1)
        else:  # at the track start: move it up by < align instead
            a0 = m0 % align
            a1 = min(a0 + want, n_total)
        if a0 <= m0 and m1 <= a1 <= n_total and a1 - a0 <= max_samples:
            c0, c1 = a0, a1
    return int(c0), int(c1)


def effective_latent_region(m0_rel: int, m1_rel: int, ds: int, n: int) -> tuple[int, int]:
    """Samples ``[e0, e1)`` the model regenerates: latent frame ``j`` is masked when sample
    ``j*ds`` is (nearest interpolation, ``model.py:457-462``), i.e. ``j`` in
    ``[ceil(m0/ds), ceil(m1/ds))``."""
    if ds <= 1:
        return m0_rel, m1_rel
    e0 = -(-m0_rel // ds) * ds
    e1 = -(-m1_rel // ds) * ds
    return int(min(e0, n)), int(min(e1, n))


def mask_seconds(n: int, sr: int) -> float:
    """Seconds that upstream's ``int(sec * sr)`` (``model.py:205-212``) maps back to ``n``."""
    return (n + 0.5) / sr


# --------------------------------------------------------------------------- model wrapper


class Engine:
    """Thin wrapper over ``stable_audio_3.StableAudioModel`` that speaks numpy
    ``(samples, channels)``. Tests construct it around a fake model with
    ``to_tensor``/``to_numpy`` replaced."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        lora_path: str | None = None,
        to_tensor: Callable[[np.ndarray], Any] | None = None,
        to_numpy: Callable[[Any], np.ndarray] | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device
        inner = model.model
        self.sample_rate = int(inner.sample_rate)  # cli.py:290
        self.sample_size = int(model.model_config.get("sample_size", DEFAULT_SAMPLE_SIZE))  # cli.py:270
        pt = getattr(inner, "pretransform", None)
        self.downsampling_ratio = int(pt.downsampling_ratio) if pt is not None else 1  # model.py:171-174
        self.channels = int(pt.io_channels) if pt is not None else int(inner.io_channels)  # model.py:441-445
        self._to_tensor = to_tensor or _np_to_torch
        self._to_numpy = to_numpy or _torch_to_np
        self.strength: float | None = 1.0 if lora_path else None

    @classmethod
    def load(cls, model_name: str, lora_path: str | None, device: str | None = None,
             half: bool = True, allow_cpu: bool = False) -> "Engine":
        import torch
        from stable_audio_3 import StableAudioModel

        if not torch.cuda.is_available() and not allow_cpu:
            raise SystemExit("no CUDA device: medium/medium-base need a GPU with flash-attn "
                             "(inference.md:28); pass --allow-cpu to try anyway")
        print(f"[sa3_infer] loading model {model_name!r} (HF_HOME={os.environ.get('HF_HOME')}, "
              f"HF_TOKEN {'set' if hf_env()['HF_TOKEN_set'] else 'NOT set'})", flush=True)
        m = StableAudioModel.from_pretrained(model_name, device=device, model_half=half)  # model.py:31
        if lora_path:
            print(f"[sa3_infer] loading LoRA {lora_path}", flush=True)
            m.load_lora([str(lora_path)])  # model.py:64 -> lora/loader.py:35
        return cls(m, model_name, lora_path, device=str(m.device))

    def set_lora_strength(self, strength: float) -> None:
        if not self.lora_path:
            raise ValueError("no LoRA loaded")
        self.model.set_lora_strength(float(strength))  # model.py:72-74, DiT + conditioner
        self.strength = float(strength)

    @property
    def max_context_samples(self) -> int:
        """Longest input the model takes without ``PadCrop`` cutting it
        (``audio_utils.py:73``): ``sample_size`` minus the 6 s duration padding
        (``model.py:400-429``), whole latent frames."""
        n = self.sample_size - int(DURATION_PADDING_S * self.sample_rate)
        return (n // self.downsampling_ratio) * self.downsampling_ratio

    def generate(
        self,
        *,
        prompt: str,
        duration: float,
        steps: int,
        cfg_scale: float,
        seed: int,
        inpaint_audio: np.ndarray | None = None,
        mask_start_s: float | None = None,
        mask_end_s: float | None = None,
    ) -> np.ndarray:
        kw: dict[str, Any] = dict(
            prompt=prompt,
            duration=float(duration),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            batch_size=1,
            sample_size=self.sample_size,
        )
        if inpaint_audio is not None:
            # (sample_rate, waveform): model.py:438 `in_sr, audio_data = audio_input`, cli.py:247-256
            kw["inpaint_audio"] = (self.sample_rate, self._to_tensor(inpaint_audio))
            kw["inpaint_mask_start_seconds"] = float(mask_start_s)
            kw["inpaint_mask_end_seconds"] = float(mask_end_s)
        out = self._to_numpy(self.model.generate(**kw))  # (1, channels, samples)
        if out.ndim != 3 or out.shape[0] != 1:
            raise RuntimeError(f"unexpected generate() output shape {out.shape}")
        return np.ascontiguousarray(out[0].T, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model_name,
            "sample_rate": self.sample_rate,
            "sample_size": self.sample_size,
            "downsampling_ratio": self.downsampling_ratio,
            "channels": self.channels,
            "device": self.device,
            "sa3_commit": sa3_commit(),
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


def _np_to_torch(audio: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))  # (channels, samples)


def _torch_to_np(t: Any) -> np.ndarray:
    return t.detach().to("cpu").float().numpy()


# --------------------------------------------------------------------------- runs


def resolve_sampling(args: argparse.Namespace) -> tuple[int, float]:
    d = MODEL_DEFAULTS[args.model]
    steps = d["steps"] if args.steps is None else int(args.steps)
    cfg = d["cfg"] if args.cfg is None else float(args.cfg)
    if steps < 1:
        raise SystemExit("--steps must be >= 1")
    return steps, cfg


def strength_plan(args: argparse.Namespace) -> list[float | None]:
    if not args.lora:
        if args.lora_strength is not None:
            raise SystemExit("--lora-strength needs --lora")
        return [None]
    return [1.0] if args.lora_strength is None else [float(s) for s in args.lora_strength]


def out_dir_for(base: Path, strength: float | None, n_strengths: int) -> Path:
    return base / strength_dirname(strength) if (strength is not None and n_strengths > 1) else base


def check_free(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"refusing to overwrite {taken[0]} (and {len(taken) - 1} more); pass --overwrite")


def common_meta(args: argparse.Namespace, engine: Engine, strength: float | None,
                adapter: dict | None, steps: int, cfg: float) -> dict[str, Any]:
    return {
        "tool": "ravi.eval.sa3_infer",
        "command": args.command,
        "argv": list(getattr(args, "_argv", [])),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": engine.describe(),
        "steps": steps,
        "cfg_scale": cfg,
        "lora": adapter,
        "lora_strength": strength,
        "hf": hf_env(),
    }


def cmd_generate(args: argparse.Namespace, engine: Engine) -> list[Path]:
    prompts = read_prompts(args.prompt, args.prompt_file)
    steps, cfg = resolve_sampling(args)
    if args.duration <= 0 or args.duration * engine.sample_rate > engine.sample_size:
        raise SystemExit(f"--duration must be in (0, {engine.sample_size / engine.sample_rate:.1f}] s")
    strengths = strength_plan(args)
    adapter = adapter_info(args.lora)
    base = Path(args.out)
    planned = [
        out_dir_for(base, s, len(strengths)) / f"{generate_stem(seed, p)}{ext}"
        for s in strengths for seed in args.seed for p in prompts for ext in (".wav", ".json")
    ]
    check_free(planned, args.overwrite)
    written: list[Path] = []
    for s in strengths:
        if s is not None:
            engine.set_lora_strength(s)
        d = out_dir_for(base, s, len(strengths))
        for seed in args.seed:
            for p in prompts:
                rendered = render_prompt(p)
                t0 = time.time()
                audio = engine.generate(prompt=rendered, duration=args.duration, steps=steps,
                                        cfg_scale=cfg, seed=seed)
                stem = generate_stem(seed, p)
                wav = d / f"{stem}.wav"
                write_wav(wav, audio, engine.sample_rate)
                meta = common_meta(args, engine, s, adapter, steps, cfg)
                meta.update(
                    prompt=p, prompt_rendered=rendered, seed=seed, duration=args.duration,
                    n_samples=int(audio.shape[0]), seconds=audio.shape[0] / engine.sample_rate,
                    peak=float(np.max(np.abs(audio))) if audio.size else 0.0,
                    elapsed_s=round(time.time() - t0, 3), file=wav.name,
                )
                write_json(d / f"{stem}.json", meta)
                print(f"[sa3_infer] wrote {wav}", flush=True)
                written.append(wav)
    return written


def load_track(path: str, sr: int, channels: int) -> tuple[np.ndarray, dict]:
    audio, in_sr = load_audio(path)
    model_audio, info = to_model_audio(audio, in_sr, sr, channels)
    info["path"] = str(path)
    info["sha256"] = sha256_file(path)
    return model_audio, info


def run_mask(
    engine: Engine,
    *,
    source: np.ndarray,
    m0: int,
    m1: int,
    c0: int,
    c1: int,
    prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    xfade_s: float,
    blank_mask_input: bool,
) -> dict[str, Any]:
    """One inpainting call on the context ``[c0, c1)`` of ``source`` (track samples at the
    model rate) with mask ``[m0, m1)``. ``c1`` may run past the track (continuation); the
    missing source is zeros."""
    sr = engine.sample_rate
    n_ctx = c1 - c0
    ctx = np.zeros((n_ctx, source.shape[1]), dtype=np.float32)
    have = min(c1, source.shape[0]) - c0
    ctx[:have] = source[c0 : c0 + have]
    m0r, m1r = m0 - c0, m1 - c0
    model_in = ctx.copy()
    if blank_mask_input:
        model_in[m0r:m1r] = 0.0
    raw = engine.generate(
        prompt=prompt, duration=n_ctx / sr, steps=steps, cfg_scale=cfg, seed=seed,
        inpaint_audio=model_in, mask_start_s=mask_seconds(m0r, sr), mask_end_s=mask_seconds(m1r, sr),
    )
    raw_len = int(raw.shape[0])
    raw, pad = fix_length(raw, n_ctx)
    xf = seconds_to_samples(xfade_s, sr)
    spliced = splice(ctx, raw[m0r:m1r], m0r, m1r, xf)
    e0, e1 = effective_latent_region(m0r, m1r, engine.downsampling_ratio, n_ctx)
    outside_src = np.concatenate([ctx[:m0r], ctx[m1r:have]]) if have > m1r else ctx[:m0r]
    outside_raw = np.concatenate([raw[:m0r], raw[m1r:have]]) if have > m1r else raw[:m0r]
    return {
        "ctx": ctx,
        "raw": raw,
        "spliced": spliced,
        "orig": upper_anchor(ctx, 0, n_ctx),
        "have": have,
        "meta": {
            "context": {"start_s": c0 / sr, "end_s": c1 / sr, "start_sample": c0, "end_sample": c1,
                        "duration_s": n_ctx / sr, "source_samples": have},
            "mask_in_context": {"start_s": m0r / sr, "end_s": m1r / sr, "start_sample": m0r, "end_sample": m1r,
                                "passed_start_s": mask_seconds(m0r, sr), "passed_end_s": mask_seconds(m1r, sr)},
            "effective_latent_mask_in_context": {
                "start_sample": e0, "end_sample": e1, "start_s": e0 / sr, "end_s": e1 / sr,
                "downsampling_ratio": engine.downsampling_ratio,
                "note": "latent frames regenerated under nearest-neighbour mask downsampling (model.py:457-462)",
            },
            "xfade": {"seconds": xfade_s, "samples": xf, "shape": "raised-cosine, inside the mask"},
            "blank_mask_input": blank_mask_input,
            "raw_output_samples": raw_len,
            "raw_padded_samples": pad,
            "raw_vs_source_outside_mask": diff_stats(outside_raw, outside_src),
        },
    }


def _write_mask_outputs(d: Path, stem: str, res: dict, sr: int, meta: dict, write_orig: bool) -> list[Path]:
    import soundfile as sf

    files = {"raw": d / f"{stem}_raw.wav", "spliced": d / f"{stem}_spliced.wav"}
    if write_orig:
        files["orig"] = d / f"{stem}_orig.wav"
    for k, p in files.items():
        write_wav(p, res[k], sr)
    # the invariant, checked on what is on disk
    back, _ = sf.read(str(files["spliced"]), dtype="float32", always_2d=True)
    m0r, m1r = meta["mask_in_context"]["start_sample"], meta["mask_in_context"]["end_sample"]
    have = res["have"]
    ctx = res["ctx"]
    identical = bool(
        back.shape == ctx.shape
        and np.array_equal(back[:m0r], ctx[:m0r])
        and np.array_equal(back[m1r:have], ctx[m1r:have])
    )
    if not identical:
        raise RuntimeError(f"{files['spliced']}: audio outside the mask differs from the source")
    meta["spliced_outside_mask_bit_identical"] = identical
    meta["files"] = {k: p.name for k, p in files.items()}
    write_json(d / f"{stem}.json", meta)
    for p in files.values():
        print(f"[sa3_infer] wrote {p}", flush=True)
    return list(files.values())


def select_masks(args: argparse.Namespace, doc: dict, duration: float) -> list[SectionMask]:
    masks = section_masks(doc["segments"], duration, pad_s=args.pad_s, min_len_s=args.min_len_s)
    if args.all_interior:
        if not masks:
            raise SystemExit(f"no interior verse/bridge/chorus sections >= {args.min_len_s} s")
        return masks
    by_index = {m.index: m for m in masks}
    missing = [i for i in args.section_index if i not in by_index]
    if missing:
        raise SystemExit(f"section index {missing} is not an interior verse/bridge/chorus section "
                         f">= {args.min_len_s} s; eligible: {mask_summary(masks)}")
    return [by_index[i] for i in args.section_index]


def cmd_inpaint(args: argparse.Namespace, engine: Engine) -> list[Path]:
    from ravi.structure import schema

    steps, cfg = resolve_sampling(args)
    strengths = strength_plan(args)
    adapter = adapter_info(args.lora)
    sr = engine.sample_rate
    source, src_info = load_track(args.audio, sr, engine.channels)
    duration = source.shape[0] / sr
    doc = schema.load(args.segments)
    track = args.track_id or doc["track_id"]
    if doc["track_id"] != Path(args.audio).stem:
        print(f"[sa3_infer] WARNING: segments track_id {doc['track_id']!r} != audio stem "
              f"{Path(args.audio).stem!r}", flush=True)
    if doc.get("duration") and abs(float(doc["duration"]) - duration) > 1.0:
        print(f"[sa3_infer] WARNING: segments duration {doc['duration']:.2f} s, audio {duration:.2f} s",
              flush=True)
    masks = select_masks(args, doc, duration)
    print(f"[sa3_infer] {track}: masks {mask_summary(masks)}", flush=True)
    prompt = render_prompt(args.prompt)
    base = Path(args.out)
    stems = [inpaint_stem(track, m.index, m.label, m.start, m.end) for m in masks]
    check_free([out_dir_for(base, s, len(strengths)) / f"{st}{ext}" for s in strengths for st in stems
                for ext in ("_raw.wav", "_spliced.wav", "_orig.wav", ".json")], args.overwrite)
    written: list[Path] = []
    for s in strengths:
        if s is not None:
            engine.set_lora_strength(s)
        d = out_dir_for(base, s, len(strengths))
        for m, stem in zip(masks, stems):
            m0, m1 = seconds_to_samples(m.start, sr), seconds_to_samples(m.end, sr)
            m1 = min(m1, source.shape[0])
            c0, c1 = choose_context(m0, m1, source.shape[0], sr, args.context_s, args.min_context_s,
                                    engine.max_context_samples, engine.downsampling_ratio)
            t0 = time.time()
            res = run_mask(engine, source=source, m0=m0, m1=m1, c0=c0, c1=c1, prompt=prompt,
                           steps=steps, cfg=cfg, seed=args.seed, xfade_s=args.xfade_ms / 1000.0,
                           blank_mask_input=args.blank_mask_input)
            seg = doc["segments"][m.index]
            meta = common_meta(args, engine, s, adapter, steps, cfg)
            meta.update(
                track_id=track, audio=src_info, segments_json=str(args.segments),
                segments_tool=doc.get("tool"), prompt=args.prompt, prompt_rendered=prompt, seed=args.seed,
                section={"index": m.index, "label": m.label, "start_s": float(seg["start"]),
                         "end_s": float(seg["end"]), "label_raw": seg["label"]},
                mask={"start_s": m.start, "end_s": m.end, "pad_s": args.pad_s, "min_len_s": args.min_len_s,
                      "start_sample": m0, "end_sample": m1},
                **res["meta"],
                elapsed_s=round(time.time() - t0, 3),
            )
            written += _write_mask_outputs(d, stem, res, sr, meta, write_orig=True)
    return written


def cmd_continue(args: argparse.Namespace, engine: Engine) -> list[Path]:
    steps, cfg = resolve_sampling(args)
    strengths = strength_plan(args)
    adapter = adapter_info(args.lora)
    sr = engine.sample_rate
    source, src_info = load_track(args.audio, sr, engine.channels)
    n = source.shape[0]
    m0 = seconds_to_samples(args.start, sr)
    if not 0 < m0 <= n:
        raise SystemExit(f"--start must be inside the track (0, {n / sr:.2f}] s")
    m1 = m0 + seconds_to_samples(args.gen_s, sr)
    ds = engine.downsampling_ratio
    c0 = max(0, m0 - seconds_to_samples(args.pre_s, sr))
    c0 = m0 - ((m0 - c0) // ds) * ds if ds > 1 else c0  # mask start on a latent frame
    if m1 - c0 > engine.max_context_samples:
        raise SystemExit(f"--pre-s + --gen-s exceed the model's {engine.max_context_samples / sr:.1f} s")
    if m1 <= m0 or c0 >= m0:
        raise SystemExit("--gen-s and --pre-s must be > 0")
    track = args.track_id or Path(args.audio).stem
    stem = continue_stem(track, m0 / sr, m1 / sr)
    prompt = render_prompt(args.prompt)
    base = Path(args.out)
    write_orig = m1 <= n
    check_free([out_dir_for(base, s, len(strengths)) / f"{stem}{ext}" for s in strengths
                for ext in ("_raw.wav", "_spliced.wav", "_orig.wav", ".json")], args.overwrite)
    written: list[Path] = []
    for s in strengths:
        if s is not None:
            engine.set_lora_strength(s)
        d = out_dir_for(base, s, len(strengths))
        t0 = time.time()
        res = run_mask(engine, source=source, m0=m0, m1=m1, c0=c0, c1=m1, prompt=prompt, steps=steps,
                       cfg=cfg, seed=args.seed, xfade_s=args.xfade_ms / 1000.0, blank_mask_input=True)
        meta = common_meta(args, engine, s, adapter, steps, cfg)
        meta.update(
            track_id=track, audio=src_info, prompt=args.prompt, prompt_rendered=prompt, seed=args.seed,
            mask={"start_s": m0 / sr, "end_s": m1 / sr, "start_sample": m0, "end_sample": m1,
                  "pre_s": args.pre_s, "gen_s": args.gen_s},
            **res["meta"],
            elapsed_s=round(time.time() - t0, 3),
        )
        written += _write_mask_outputs(d, stem, res, sr, meta, write_orig=write_orig)
    return written


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ravi.eval.sa3_infer", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", choices=MODELS, default="medium",
                       help="medium = post-trained (default; 8 steps, cfg 1), medium-base (50 steps, cfg 7)")
        p.add_argument("--lora", default=None, metavar="PATH", help="adapter .safetensors (e.g. adapter.safetensors)")
        p.add_argument("--lora-strength", type=float, nargs="+", default=None, metavar="S",
                       help="one or more strengths (default 1.0); several -> OUT/strength{S}/")
        p.add_argument("--steps", type=int, default=None, help="default: 8 (medium) / 50 (medium-base)")
        p.add_argument("--cfg", type=float, default=None, help="default: 1.0 (medium) / 7.0 (medium-base)")
        p.add_argument("--out", required=True, metavar="DIR")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--device", default=None, help="cuda (default when available) / cpu")
        p.add_argument("--no-half", action="store_true", help="fp32 model weights (upstream default is fp16)")
        p.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)

    g = sub.add_parser("generate", help="text-to-audio, one WAV per prompt x seed")
    common(g)
    g.add_argument("--prompt", action="append", default=None, help="repeatable")
    g.add_argument("--prompt-file", default=None, help="one prompt per line ('#' comments, blank lines skipped)")
    g.add_argument("--duration", type=float, default=30.0)
    g.add_argument("--seed", type=int, nargs="+", default=[0])

    def masked(p: argparse.ArgumentParser) -> None:
        p.add_argument("--audio", required=True, help="source track (any rate; resampled to the model rate)")
        p.add_argument("--prompt", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--xfade-ms", type=float, default=DEFAULT_XFADE_S * 1000.0)
        p.add_argument("--track-id", default=None, help="name used in output files (default: from the input)")

    i = sub.add_parser("inpaint", help="regenerate interior song sections")
    common(i)
    masked(i)
    i.add_argument("--segments", required=True, help="ravi.structure.schema JSON for the track")
    sel = i.add_mutually_exclusive_group(required=True)
    sel.add_argument("--section-index", type=int, nargs="+", help="index in the JSON's segments list")
    sel.add_argument("--all-interior", action="store_true")
    i.add_argument("--pad-s", type=float, default=DEFAULT_PAD_S)
    i.add_argument("--min-len-s", type=float, default=DEFAULT_MIN_LEN_S)
    i.add_argument("--context-s", type=float, default=DEFAULT_CONTEXT_S,
                   help="context window length (0 = whole track, capped at the model maximum)")
    i.add_argument("--min-context-s", type=float, default=DEFAULT_MIN_CONTEXT_S)
    i.add_argument("--blank-mask-input", action="store_true",
                   help="zero the source inside the mask before VAE encoding (upstream does not)")

    c = sub.add_parser("continue", help="regenerate from --start for --gen-s seconds")
    common(c)
    masked(c)
    c.add_argument("--start", type=float, required=True)
    c.add_argument("--pre-s", type=float, default=30.0, help="seconds of context before --start")
    c.add_argument("--gen-s", type=float, default=15.0)
    return ap


COMMANDS = {"generate": cmd_generate, "inpaint": cmd_inpaint, "continue": cmd_continue}


def main(argv: Sequence[str] | None = None, engine_factory: Callable[[argparse.Namespace], Engine] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args._argv = argv
    strength_plan(args)  # validate before loading anything
    if args.lora and not Path(args.lora).is_file():
        raise SystemExit(f"--lora {args.lora}: no such file")
    if not hf_env()["HF_TOKEN_set"]:
        print("[sa3_infer] WARNING: HF_TOKEN is not set; gated stabilityai/* downloads will fail "
              "unless the files are already in HF_HOME", flush=True)
    if engine_factory is None:
        engine = Engine.load(args.model, args.lora, device=args.device, half=not args.no_half,
                             allow_cpu=args.allow_cpu)
    else:
        engine = engine_factory(args)
    COMMANDS[args.command](args, engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
