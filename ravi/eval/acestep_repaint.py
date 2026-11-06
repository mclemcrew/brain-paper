"""ACE-Step 1.5 repaint check for a rAvI LoRA (SMOKE.md section 7, Tier 0 groundwork).

Repaints one interior section of a full-length track with ACE-Step's own repaint task:
once without the adapter, then with it at each ``--lora-strength``. Each output is checked
against the source outside the region the splice is allowed to touch. Runs in the
``ravi-acestep-train`` image::

    docker run ... --entrypoint python ravi-acestep-train:dev -m ravi.eval.acestep_repaint \\
        --audio TRACK.wav --segments TRACK.json --section-index I --pad-s 1.0 \\
        --caption "rvkr, ..." --lyrics-from-segments --adapter /adapter \\
        --lora-strength 0 0.5 1.0 --seed 1 --out /out/repaint

Upstream calls (ACE-Step-1.5 at ca1e85fe):

* ``AceStepHandler().initialize_service(project_root, config_path, device, use_flash_attention)``
  (``core/generation/handler/init_service_orchestrator.py:48-220``) returns
  ``(message, ok)``. Weights come from ``ACESTEP_CHECKPOINTS_DIR`` (``:122-128``).
  ``use_flash_attention`` defaults to ``False``, so it is passed explicitly.
* ``load_lora(dir)`` (``lora/lifecycle.py:319-325``) takes the directory with
  ``adapter_config.json``. ``set_use_lora`` and ``set_lora_scale`` (``lora/controls.py:35,83``;
  the scale is clamped to [0, 1]).
* ``generate_music(dit_handler, None, GenerationParams(task_type="repaint", ...),
  GenerationConfig(...))`` (``inference.py:514``). Repaint is a direct-conditioning task, so
  the LM is skipped and ``llm_handler=None`` is fine (``inference.py:36-38,633-655``).
  ``result.audios[i]["tensor"]`` is ``[channels, samples]`` float32 on the CPU at
  ``sample_rate`` (``inference.py:1060-1063``).

What the check relies on:

* The source is read with soundfile, converted to stereo 48 kHz
  (``io_audio.py:178-196``) and, for an interior mask, not padded
  (``padding_utils.py:37-60``). This tool writes the source as a float32 48 kHz stereo WAV and
  passes that file, so the model's copy and the reference are the same samples.
* After decoding, the waveform splice puts the source back outside the mask, with linear
  ramps of ``wav_crossfade_s`` **outside** it: ``[start - cf, start)`` and ``[end, end + cf)``
  (``repaint_waveform_splice.py:14-51,76-104``). ``cf`` comes from ``repaint_mode`` and
  ``repaint_strength`` (``generate_music.py:27-48``). In balanced mode at 0.5 that is
  25 ms. The splice is skipped in ``aggressive`` mode.
* ``enable_normalization=False``: otherwise the whole output, including the spliced source,
  is rescaled (``inference.py:998-1012``). The decoder's own peak limiter
  (``generate_music_decode.py:199-203``) runs before the splice, so it cannot touch the
  outside.
* The instruction is set to ``TASK_INSTRUCTIONS["repaint"]`` (``constants.py:130``). The
  REST API and Gradio do this themselves; ``GenerationParams`` defaults to the text2music
  one (``inference.py:112``).

Determinism: the seed only fixes the initial noise (``prepare_noise``,
``models/turbo/modeling_acestep_v15_turbo.py:1780-1817``). Each call also VAE-encodes the source
with ``latent_dist.sample()`` (``core/generation/handler/vae_encode.py:65``), a draw from the
unseeded global torch RNG. Two same-seed calls therefore condition on slightly different
source latents and repaint different audio: on the smoke run, max |diff| 0.34-0.42 inside the
mask, 10 % of the source RMS. Upstream's repaint cache (``inference.py:452-510``) avoids this
only for ACE-generated Gradio outputs. ``AceStepEngine.repaint`` therefore calls
``torch.manual_seed(seed)`` before every call. With that, a same-seed repeat and LoRA scale 0
both reproduce the no-LoRA output exactly, and ``run_check`` requires both (the
``nolora_repeat`` row and ``strength0``).

The no-LoRA condition uses ``set_use_lora(False)`` (PEFT ``disable_adapter_layers``), not
``unload_lora()``. At ``ca1e85fe`` the latter restores the decoder backup with mismatched keys
("Missing keys when restoring decoder", ``lora/lifecycle.py:472-474``) and leaves the adapter
active: its output differed from the base model as much as scale 1 did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from ravi.eval.masks import DEFAULT_PAD_S, SectionMask, section_masks
from ravi.structure.labels import UNLABELED, normalize_label

log = logging.getLogger("ravi.eval.acestep_repaint")

SR = 48000
VARIANTS = {  # config dir, steps, shift, guidance ("Using Your Adapter.md:95-99")
    "turbo": ("acestep-v15-turbo", 8, 3.0, 1.0),
    "base": ("acestep-v15-base", 50, 1.0, 7.0),
}
REPAINT_MODES = ("conservative", "balanced", "aggressive")
OUTSIDE_TOL = 1e-6  # float32 copy of the source; anything above this is a real change


class Engine(Protocol):
    def repaint(self, *, src_path: str, start: float, end: float, caption: str, lyrics: str,
                seed: int, steps: int, shift: float, guidance: float, mode: str,
                strength: float) -> np.ndarray: ...

    def set_lora(self, scale: float | None) -> None: ...

    def wav_crossfade_s(self, mode: str, strength: float) -> float: ...


# --------------------------------------------------------------------------- helpers


def to_stereo_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    """``[frames]`` or ``[frames, ch]`` -> float32 ``[frames, 2]`` at 48 kHz in [-1, 1]."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    elif a.shape[1] > 2:
        a = a[:, :2]
    if sr != SR:
        from ravi.captions.make_acestep_dataset import resample

        a = resample(a, sr, SR)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def segments_lyrics(segments: Iterable[Any]) -> str:
    """Section tags for the whole track, one per line, in segment order (the arm E lyrics
    format of ``ravi.captions.make_acestep_dataset``). Unlabeled segments are skipped."""
    out = []
    for s in segments:
        label = s["label"] if isinstance(s, dict) else s[2] if isinstance(s, tuple) else s.label
        lab = normalize_label(label)
        if lab and lab != UNLABELED:
            out.append(f"[{lab}]")
    return "\n".join(out) if out else "[Instrumental]"


def touched_span(start: float, end: float, cf_s: float, n: int, sr: int = SR) -> tuple[int, int]:
    """Samples ``[lo, hi)`` the splice may change: the mask plus ``cf`` on each side, with
    upstream's ``int()`` rounding (``repaint_waveform_splice.py:91,94-97``)."""
    s0 = max(0, min(int(start * sr), n))
    s1 = max(s0, min(int(end * sr), n))
    cf = int(cf_s * sr)
    return max(0, s0 - cf), min(n, s1 + cf)


def outside_diff(out: np.ndarray, src: np.ndarray, lo: int, hi: int) -> dict[str, Any]:
    """Max |out - src| outside ``[lo, hi)`` over the common length, plus inside stats."""
    n = min(len(out), len(src))
    d = np.abs(out[:n].astype(np.float64) - src[:n].astype(np.float64))
    outside = np.concatenate([d[:lo], d[hi:n]])
    inside = d[lo:hi]
    return {
        "n_out": int(len(out)), "n_src": int(len(src)),
        "outside_max_abs": float(outside.max()) if outside.size else 0.0,
        "inside_rms_diff": float(np.sqrt(np.mean(inside ** 2))) if inside.size else 0.0,
        # The decoder output can differ from the source by a few samples (latent framing);
        # the splice and this comparison both use the common length.
        "outside_ok": bool((outside.max() if outside.size else 0.0) <= OUTSIDE_TOL),
    }


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if x.size else 0.0


def strength_name(s: float | None) -> str:
    return "nolora" if s is None else f"strength{s:g}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_mask(doc: dict, duration: float, index: int | None, pad_s: float) -> SectionMask:
    masks = section_masks(doc["segments"], duration, pad_s=pad_s)
    if not masks:
        raise SystemExit("no interior verse/bridge/chorus section in the segments")
    if index is None:
        return masks[0]
    for m in masks:
        if m.index == index:
            return m
    raise SystemExit(f"section {index} is not an eligible interior section; eligible: {[m.index for m in masks]}")


# --------------------------------------------------------------------------- the check


def run_check(engine: Engine, *, source: np.ndarray, src_path: Path, mask: SectionMask,
              caption: str, lyrics: str, seed: int, variant: str, strengths: Sequence[float],
              adapter: bool, mode: str, repaint_strength: float, out: Path) -> dict[str, Any]:
    import soundfile as sf

    _, steps, shift, guidance = VARIANTS[variant]
    cf_s = engine.wav_crossfade_s(mode, repaint_strength) if mode != "aggressive" else 0.0
    lo, hi = touched_span(mask.start, mask.end, cf_s, len(source))
    plan: list[tuple[str, float | None]] = [("nolora", None), ("nolora_repeat", None)]
    plan += [(strength_name(s), s) for s in (strengths if adapter else [])]
    rows, outs = [], {}
    for name, s in plan:
        engine.set_lora(s)
        y = engine.repaint(src_path=str(src_path), start=mask.start, end=mask.end, caption=caption,
                           lyrics=lyrics, seed=seed, steps=steps, shift=shift, guidance=guidance,
                           mode=mode, strength=repaint_strength)
        sf.write(out / f"{name}.wav", y, SR, subtype="FLOAT")
        outs[name] = y
        row = {"condition": name, "lora_scale": s, **outside_diff(y, source, lo, hi),
               "inside_rms": rms(y[lo:hi]), "source_inside_rms": rms(source[lo:hi])}
        if mode == "aggressive":
            row["outside_ok"] = None  # no splice, nothing to check
        rows.append(row)
        log.info("[ravi] repaint %s: %s", name, json.dumps(row))
    base = outs["nolora"]
    src_rms = rms(source[lo:hi])
    for r in rows:
        y = outs[r["condition"]]
        h = min(hi, len(y), len(base))
        dd = np.abs(y[lo:h].astype(np.float64) - base[lo:h].astype(np.float64)) if h > lo else np.zeros(0)
        r["inside_max_abs_vs_nolora"] = float(dd.max()) if dd.size else 0.0
        r["inside_rel_rms_vs_nolora"] = (rms(dd) / src_rms) if src_rms > 0 else 0.0
    by = {r["condition"]: r for r in rows}
    repeat_ok = by["nolora_repeat"]["inside_max_abs_vs_nolora"] <= OUTSIDE_TOL
    scale0_ok = by["strength0"]["inside_max_abs_vs_nolora"] <= OUTSIDE_TOL if "strength0" in by else None
    outside_ok = all(r["outside_ok"] is not False for r in rows)
    return {"touched_span": [lo, hi], "wav_crossfade_s": cf_s, "steps": steps, "shift": shift,
            "guidance": guidance, "rows": rows, "outside_ok": outside_ok,
            "repeat_ok": repeat_ok, "scale0_ok": scale0_ok,
            "ok": outside_ok and repeat_ok and scale0_ok is not False}


class AceStepEngine:
    """The real thing: AceStepHandler + generate_music (imports only inside the image)."""

    def __init__(self, variant: str, adapter: str | None, device: str = "cuda",
                 flash_attention: bool = True) -> None:
        from acestep.handler import AceStepHandler

        self.h = AceStepHandler()
        msg, ok = self.h.initialize_service(
            project_root=os.environ.get("ACESTEP_ROOT", "/opt/acestep"),
            config_path=VARIANTS[variant][0], device=device,
            use_flash_attention=flash_attention)
        if not ok:
            raise SystemExit(f"initialize_service failed: {msg}")
        log.info("[ravi] %s", str(msg).splitlines()[0] if msg else "initialized")
        self.adapter_loaded = False
        if adapter:
            m = self.h.load_lora(adapter)
            if not str(m).startswith("✅"):
                raise SystemExit(f"load_lora failed: {m}")
            log.info("[ravi] %s", m)
            self.adapter_loaded = True

    def wav_crossfade_s(self, mode: str, strength: float) -> float:
        from acestep.core.generation.handler.generate_music import _resolve_repaint_config

        return float(_resolve_repaint_config(mode, strength)[2])

    def set_lora(self, scale: float | None) -> None:
        if not self.adapter_loaded:
            if scale is not None:
                raise SystemExit("no adapter loaded")
            return
        for m in ([self.h.set_use_lora(False)] if scale is None
                  else [self.h.set_use_lora(True), self.h.set_lora_scale(float(scale))]):
            # "✅ ... (skipped N targets)" means some layers kept their old scale.
            if not str(m).startswith("✅") or "skipped" in str(m):
                raise SystemExit(f"LoRA control failed: {m}")

    def repaint(self, *, src_path, start, end, caption, lyrics, seed, steps, shift, guidance,
                mode, strength) -> np.ndarray:
        from acestep.constants import TASK_INSTRUCTIONS
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        params = GenerationParams(
            task_type="repaint", instruction=TASK_INSTRUCTIONS["repaint"], src_audio=src_path,
            repainting_start=float(start), repainting_end=float(end), chunk_mask_mode="explicit",
            repaint_mode=mode, repaint_strength=float(strength), caption=caption, lyrics=lyrics,
            instrumental=True, inference_steps=steps, shift=shift, guidance_scale=guidance,
            seed=seed, thinking=False, use_cot_metas=False, use_cot_caption=False,
            use_cot_language=False, enable_normalization=False)
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        import torch

        torch.manual_seed(seed)  # the source VAE encode samples from the global RNG (module docstring)
        return self._audio(generate_music(self.h, None, params, cfg, save_dir=None))

    def text2music(self, *, caption, lyrics, duration, seed, steps, shift, guidance) -> np.ndarray:
        """Plain text-to-music on the DiT (no LM: ``thinking`` and every CoT step off), with
        the same per-call global seeding as :meth:`repaint`. ``[frames, 2]`` at 48 kHz."""
        import torch
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        params = GenerationParams(
            task_type="text2music", caption=caption, lyrics=lyrics, instrumental=True, duration=float(duration),
            inference_steps=steps, shift=shift, guidance_scale=guidance, seed=seed, thinking=False,
            use_cot_metas=False, use_cot_caption=False, use_cot_language=False, enable_normalization=False)
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
        torch.manual_seed(seed)
        return self._audio(generate_music(self.h, None, params, cfg, save_dir=None))

    @staticmethod
    def _audio(res: Any) -> np.ndarray:
        if not res.success or not res.audios:
            raise RuntimeError(f"generate_music failed: {res.error or res.status_message}")
        a = res.audios[0]
        if int(a.get("sample_rate", SR)) != SR:
            raise RuntimeError(f"unexpected sample rate {a.get('sample_rate')}")
        return np.asarray(a["tensor"].float().numpy().T, dtype=np.float32)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ravi.eval.acestep_repaint", description=__doc__.split("\n")[0])
    p.add_argument("--audio", required=True, help="full-length source track (any rate)")
    p.add_argument("--segments", required=True, help="ravi.structure.schema JSON for the track")
    p.add_argument("--section-index", type=int, default=None, help="default: first eligible section")
    p.add_argument("--pad-s", type=float, default=DEFAULT_PAD_S)
    p.add_argument("--caption", required=True, help="the fields joined with ', ', trigger first")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--lyrics", default=None, help="default: [Instrumental]")
    g.add_argument("--lyrics-from-segments", action="store_true", help="arm E: one [label] per segment")
    p.add_argument("--adapter", default=None, help="dir with adapter_config.json + adapter_model.safetensors")
    p.add_argument("--lora-strength", type=float, nargs="+", default=[1.0])
    p.add_argument("--variant", choices=sorted(VARIANTS), default="turbo")
    p.add_argument("--repaint-mode", choices=REPAINT_MODES, default="balanced")
    p.add_argument("--repaint-strength", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-flash-attention", action="store_true")
    p.add_argument("--out", required=True)
    return p


def main(argv: Sequence[str] | None = None, engine: Engine | None = None) -> int:
    import soundfile as sf

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(args.segments).read_text())
    audio, sr = sf.read(args.audio, dtype="float64", always_2d=True)
    source = to_stereo_48k(audio, sr)
    duration = len(source) / SR
    mask = pick_mask(doc, duration, args.section_index, args.pad_s)
    src_path = out / "source_48k.wav"
    sf.write(src_path, source, SR, subtype="FLOAT")
    lyrics = segments_lyrics(doc["segments"]) if args.lyrics_from_segments else (args.lyrics or "[Instrumental]")
    if engine is None:
        engine = AceStepEngine(args.variant, args.adapter, args.device, not args.no_flash_attention)
    res = run_check(engine, source=source, src_path=src_path, mask=mask, caption=args.caption,
                    lyrics=lyrics, seed=args.seed, variant=args.variant, strengths=args.lora_strength,
                    adapter=bool(args.adapter), mode=args.repaint_mode,
                    repaint_strength=args.repaint_strength, out=out)
    meta = {
        "audio": args.audio, "track_id": doc.get("track_id"), "duration_s": duration,
        "section": {"index": mask.index, "label": mask.label, "mask_s": [mask.start, mask.end]},
        "caption": args.caption, "lyrics": lyrics, "seed": args.seed, "variant": args.variant,
        "repaint_mode": args.repaint_mode, "repaint_strength": args.repaint_strength,
        "adapter": args.adapter,
        "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
        "acestep_commit": os.environ.get("ACESTEP_COMMIT"), "ravi_git_sha": os.environ.get("RAVI_GIT_SHA"),
        **res,
    }
    (out / "repaint_check.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("ok", "outside_ok", "repeat_ok", "scale0_ok")} | {"rows": [
        {k: r[k] for k in ("condition", "outside_max_abs", "inside_rms_diff", "inside_max_abs_vs_nolora",
                           "inside_rel_rms_vs_nolora")}
        for r in res["rows"]]}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
