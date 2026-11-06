"""Phase 4 evaluation driver: per-condition text-to-music and section-inpainting metrics.

Layout read (``--eval-root``, default ``eval``; ``ravi.eval.grid`` writes exactly this)::

    eval/{condition}/inpaint/{track}_sec{i}_{label}_{start}-{end}[_seed{s}]_{raw,spliced,orig}.wav + .json
    eval/{condition}/gen/{stem}[_seed{s}].wav (+ {stem}.json)
    eval/anchor_lower/inpaint/...           (grid's lower anchor; preferred over the built-in one)

A condition whose ``inpaint/`` (or ``gen/``) holds ``strength{v}/`` subdirectories (a
``sa3_infer --lora-strength`` sweep) expands to one condition per strength, named
``{condition}@strength{v}``. Directories starting with ``.`` or ``_`` are ignored; ``upper``,
``lower`` and ``test_orig`` are reserved.

**Seeds.** ``grid`` names every output ``…_seed{s}``. Per-output rows keep the seed; before any
summary or paired test the rows of one region (or one prompt) are *averaged over its seeds*
(``region_key`` / ``prompt_key`` = the stem without ``_seed{s}``), and ``n_seeds`` is recorded,
so seeds are not counted as independent regions.

Inputs besides the layout: ``--train-dir`` with ``--splits`` (both required together: the
training corpus = the FAD/KAD reference and the memorisation set; a train id in val/test or a
test original in the reference stops the run); ``--test-audio`` (test originals
``{track_id}.wav``: lower-anchor donors and the ``test_orig`` reference row);
``--test-captions`` (required when any ``gen/`` exists): a ``select_regions`` file such as
``eval/regions_test.json`` (``tracks[id].captions[--caption-arm]``, default arm B: the same
captions ``grid`` prompts with), a caption JSONL such as ``captions/midasheng_test.jsonl``
(``track_id`` + ``caption``), or a directory of ``{track_id}.txt``; ``--regions-json`` (default:
``--test-captions`` when that is a regions file) gives each track's ``alt_family`` for the
built-in lower anchor; ``--structure-dir`` (optional, see below).

**Inpainting** (one row per condition x output in ``metrics_regions.parquet``). Regions are the
sidecars of ``sa3_infer inpaint`` (``section`` present; continuations are skipped); the region is
``mask_in_context`` of the sidecar (samples of the context-window WAVs), falling back to
``regenerated_region`` and then to ``mask`` minus the context start. All conditions must agree
on the geometry and on the source audio (``audio.sha256``) of a region. The evaluated audio is
``{stem}_spliced.wav`` (``--variant raw`` for the unspliced output) against ``{stem}_orig.wav``.
Metrics (directions in :data:`REGION_METRICS`):

* ``region_sim_{clap,mert}``: cosine of the pooled embeddings of the regenerated vs the original
  region; ``region_sim_mert_centred``: the same after subtracting the reference corpus' mean MERT
  embedding (raw MERT cosines between unrelated tracks are ~0.9: anisotropy). Before CLAP both
  regions are normalised to -16 LUFS (:func:`ravi.eval.embed.lufs_normalise`; measured levels in
  ``lufs_gen_region`` / ``lufs_orig_region``).
* ``seam_score_{start,end}`` (+ ``_orig_``), ``seam_delta_{start,end}``, ``seam_delta`` (mean of
  the per-seam deltas) and ``abs_seam_delta`` (mean of ``|delta|``); the reference is the
  original *at the same position*, which is ``pad_s`` inside the neighbouring section (see
  :mod:`ravi.eval.seams`).
* ``beat_grid_agree`` (primary rhythmic metric: fraction of the original's beats within ±4 s of a
  seam that have a generated beat within ±70 ms) and ``beat_grid_offset_ms``; ``beat_cv`` (IBI CV,
  secondary), ``beat_cv_orig``, ``beat_cv_delta`` (per-seam difference, then NaN-mean). Grid
  beats: both contexts tracked at the original's global tempo with ``--grid-tightness`` 10
  (:func:`ravi.eval.seams.grid_beats`); IBI-CV beats: whole-context tracking with ``--tempo``
  (default ``global``).
* ``key_agree`` (1/0, NaN when gated), ``key_conf_orig``, ``key_orig``, ``key_gen``.
* ``struct_kept`` with ``--structure-dir``: SongFormer (or any common-schema) documents computed
  externally with the structure image on the evaluated WAVs, at
  ``{structure_dir}/{path of the WAV relative to --eval-root}`` with ``.json`` (e.g.
  ``B/inpaint/{stem}_spliced.json`` and ``B/inpaint/{stem}_orig.json``), times in context seconds.
  For each original section boundary adjacent to the mask (the human ``section`` start/end of
  the sidecar, inside the padded mask), is a predicted boundary within 3 s? ``struct_kept`` = the
  fraction kept on the evaluated audio, ``struct_kept_orig`` on ``_orig``, ``struct_kept_delta``.

Anchors (``--no-anchors`` to skip): ``upper`` = the ``_orig`` context itself (identity). Lower:
an existing ``anchor_lower`` condition (``--lower-anchor``; ``grid`` writes it with a
length-matched donor from another ``alt_family``) is used as is; only without one is ``lower``
built here: the region replaced by a same-length cut from the middle of the next test track
(sorted ids, wrapping) of a *different* ``alt_family`` (from ``--regions-json``; without family
information only a different track, recorded as unchecked), spliced with the 50 ms crossfades.

**Text-to-music** (``metrics_gen.parquet``, one row per clip). ``clap_text`` (cosine of the clip's
CLAP audio embedding, after -16 LUFS normalisation, and its prompt's text embedding; prompt from
the sidecar's ``prompt_rendered``; ``prompt_truncated`` where CLAP's 77-token limit cut it),
``lufs_in`` and, per extractor, ``memo_max_*`` / ``memo_track_*`` (max cosine of any of the clip's
windows to any training window). The memorisation *flag* is CLAP only: ``memo_flag_clap`` at the
fixed ``--memo-thresh`` (0.95) and ``memo_flag_null_clap`` above the null threshold = the 99th
percentile of ``memo_max_clap`` over the ``test_orig`` clips (held-out real music vs the same
training set, same run). ``test_orig`` (the test originals with their captions) is always
evaluated when generated clips are.

Distribution metrics per condition (``metrics_summary.csv`` and ``metrics_dist.parquet``): the
window embeddings of the clips and of the training corpus, projected on the top
``--pca-dims`` (64) principal axes fitted on the *reference only*; FAD∞ (without replacement,
grid floor ``max(3 d, 500)`` rows, :func:`ravi.eval.metrics.fad_inf`; fit range ``n_lo..n_hi``,
slope, R² and ``d`` reported) and KAD (unbiased, sigma = median within the reference), with
clip-level bootstrap 95% CIs (``--dist-boot`` 200 resamples of *clips*,
:func:`ravi.eval.metrics.clip_bootstrap`). Every condition, ``test_orig`` included, is first
subsampled (seeded, recorded) to the smallest clip count among them, so the values are
**relative comparisons at matched n**, not unbiased absolute distances; ``test_orig`` is the
real-music floor at that n. Plain full-dimension FAD is only a diagnostic (``fad_{x}_naive`` in
``metrics_dist.parquet``). The FAD∞ sample size counts windows, as fadtk counts frames.

**Comparisons** (``comparisons.csv``): every pair of conditions on every metric, on the
seed-averaged values, :func:`ravi.eval.stats.paired_compare` with ``d = a - b`` and ``a`` the
non-base condition when ``base`` is involved. Inpainting rows come at two ``level``\\ s with
``--cluster track_id`` (default): ``track`` (primary: the signed-rank test on per-track medians,
CI by bootstrap over tracks) and ``region`` (secondary: pseudo-replicated region-level test, CI
by a cluster bootstrap over tracks). ``p_a_better`` / ``p_a_worse`` are the one-sided p-values in
the metric's direction; ``p_holm_two_sided``, ``p_holm_a_worse``, ``p_holm_a_better`` are Holm
within one (family, metric, level) over all condition pairs (``holm_m`` tests, per-metric scope;
no correction across metrics). The go/no-go reads ``p_holm_a_worse`` on the ``track`` rows.

**Outputs** (``--out-dir``, default ``--eval-root``): ``metrics_regions.parquet`` (per output),
``metrics_regions_avg.parquet`` (per condition x region, seed-averaged), ``metrics_gen.parquet``,
``metrics_dist.parquet``, ``metrics_summary.csv`` (one row per condition), ``comparisons.csv``,
``metrics_meta.json``. Deterministic: sorted inputs, seeded draws, no timestamps in the tables.
``--limit N`` keeps the first N regions (sorted ids), clips per condition and training files.

Example (``docker/eval/README.md``)::

    python -m ravi.eval.run_eval --eval-root eval/tier0 --base base \\
        --test-audio prepared/44100 --train-dir prepared/44100 --splits manifests/splits.json \\
        --test-captions eval/regions_test.json --cache-dir eval/.emb_cache
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ravi.eval import metrics as M
from ravi.eval import seams as S
from ravi.eval import stats as ST
from ravi.eval.masks import DEFAULT_XFADE_S, lower_anchor, seconds_to_samples, splice

RESERVED = ("upper", "lower", "test_orig")
DEFAULT_EXTRACTORS = ("clap", "mert")
DEFAULT_BASE = "base"
DEFAULT_LOWER_ANCHOR = "anchor_lower"
#: extractors whose input is loudness-normalised first (CLAP is level-sensitive; MERT z-scores)
LOUDNESS_NORMALISED = ("clap",)
#: extractors whose memorisation max-cosine is flagged (MERT windows of different tracks > 0.95)
MEMO_FLAG_EXTRACTORS = ("clap",)
#: extractors with a reference-mean-centred region similarity
CENTRED_SIM_EXTRACTORS = ("mert",)
STRUCT_TOL_S = 3.0
SEED_RE = re.compile(r"_seed(-?\d+)$")

# metric -> +1 if higher is better, -1 if lower is better
REGION_METRICS: dict[str, int] = {
    "region_sim_clap": +1,
    "region_sim_mert": +1,
    "region_sim_mert_centred": +1,
    "seam_delta": -1,
    "abs_seam_delta": -1,
    "beat_grid_agree": +1,
    "beat_grid_offset_ms": -1,
    "beat_cv": -1,
    "beat_cv_delta": -1,
    "key_agree": +1,
    "struct_kept": +1,
    "struct_kept_delta": +1,
}
GEN_METRICS: dict[str, int] = {"clap_text": +1}


def log(msg: str) -> None:
    print(f"[run_eval] {msg}", flush=True)


def seed_split(stem: str) -> tuple[str, int | None]:
    """``("{key}", seed)`` for ``{key}_seed{seed}``, else ``(stem, None)``."""
    m = SEED_RE.search(stem)
    return (stem[: m.start()], int(m.group(1))) if m else (stem, None)


# --------------------------------------------------------------------------- discovery


@dataclass
class Region:
    region_id: str  # sidecar stem
    track_id: str
    section_index: int | None
    label: str | None
    m0: int  # mask start in context samples
    m1: int
    mask_start_s: float | None  # track seconds
    mask_end_s: float | None
    context_start_sample: int | None
    sidecar: Path
    files: dict[str, Path] = field(default_factory=dict)
    region_key: str = ""
    seed: int | None = None
    audio_sha256: str | None = None
    section_bounds_ctx_s: tuple[float, ...] = ()  # human section start/end in context seconds


@dataclass
class Condition:
    name: str
    inpaint_dir: Path | None
    gen_dir: Path | None


def _subconditions(name: str, d: Path | None) -> dict[str, Path]:
    if d is None or not d.is_dir():
        return {}
    subs = sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("strength"))
    if subs:
        return {f"{name}@{p.name}": p for p in subs}
    return {name: d}


def discover_conditions(root: Path) -> list[Condition]:
    found: dict[str, Condition] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith((".", "_")):
            continue
        if d.name in RESERVED:
            raise SystemExit(f"{d}: condition name {d.name!r} is reserved for anchors/references")
        inp = _subconditions(d.name, d / "inpaint")
        gen = _subconditions(d.name, d / "gen")
        for name in sorted(set(inp) | set(gen)):
            found[name] = Condition(name, inp.get(name), gen.get(name))
    return list(found.values())


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def _mask_in_context(meta: dict, sr: int) -> tuple[int, int]:
    mic = meta.get("mask_in_context")
    if isinstance(mic, dict) and "start_sample" in mic and "end_sample" in mic:
        return int(mic["start_sample"]), int(mic["end_sample"])
    rr = meta.get("regenerated_region")
    if isinstance(rr, dict):
        if "start_sample" in rr and "end_sample" in rr:
            return int(rr["start_sample"]), int(rr["end_sample"])
        if "start_s" in rr and "end_s" in rr:
            return seconds_to_samples(rr["start_s"], sr), seconds_to_samples(rr["end_s"], sr)
    mask, ctx = meta.get("mask"), meta.get("context") or {}
    if isinstance(mask, dict) and "start_s" in mask and "end_s" in mask:
        c0 = float(ctx.get("start_s", 0.0))
        return seconds_to_samples(float(mask["start_s"]) - c0, sr), seconds_to_samples(float(mask["end_s"]) - c0, sr)
    raise ValueError("sidecar has no mask_in_context / regenerated_region / mask")


def _context_start_s(meta: dict, m0: int, sr: int) -> float:
    c0 = _first(meta, "context", "start_s")
    if c0 is not None:
        return float(c0)
    ms = _first(meta, "mask", "start_s")
    return float(ms) - m0 / sr if ms is not None else 0.0


def _wav_info(path: Path) -> tuple[int, int]:
    import soundfile as sf

    info = sf.info(str(path))
    return int(info.samplerate), int(info.frames)


def read_regions(inpaint_dir: Path, variant: str) -> dict[str, Region]:
    """Inpainting sidecars of one condition, keyed by stem."""
    out: dict[str, Region] = {}
    for js in sorted(inpaint_dir.glob("*.json")):
        try:
            meta = json.loads(js.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"{js}: bad JSON ({e})") from None
        if not isinstance(meta, dict) or "section" not in meta or meta.get("command") not in (None, "inpaint"):
            continue
        stem = js.stem
        files = {k: js.with_name(f"{stem}_{k}.wav") for k in ("raw", "spliced", "orig")}
        gen = files[variant]
        if not gen.exists():
            raise SystemExit(f"{js}: missing {gen.name}")
        sr, n = _wav_info(gen)
        m0, m1 = _mask_in_context(meta, sr)
        if not 0 <= m0 < m1 <= n:
            raise SystemExit(f"{js}: mask [{m0}, {m1}) outside the {n}-sample context")
        sec = meta.get("section") or {}
        c0 = _context_start_s(meta, m0, sr)
        bounds = tuple(float(sec[k]) - c0 for k in ("start_s", "end_s")
                       if sec.get(k) is not None and 0.0 < float(sec[k]) - c0 < n / sr)
        key, seed = seed_split(stem)
        out[stem] = Region(
            region_id=stem,
            track_id=str(meta.get("track_id") or re.sub(r"_sec\d+_.*$", "", stem)),
            section_index=sec.get("index"),
            label=sec.get("label"),
            m0=m0,
            m1=m1,
            mask_start_s=_first(meta, "mask", "start_s"),
            mask_end_s=_first(meta, "mask", "end_s"),
            context_start_sample=_first(meta, "context", "start_sample"),
            sidecar=js,
            files={k: p for k, p in files.items() if p.exists()},
            region_key=key,
            seed=seed if seed is not None else (meta.get("seed") if isinstance(meta.get("seed"), int) else None),
            audio_sha256=_first(meta, "audio", "sha256"),
            section_bounds_ctx_s=bounds,
        )
    return out


def read_gen(gen_dir: Path, captions: dict[str, str] | None) -> list[dict[str, Any]]:
    """Generated clips of one condition with their prompts (sorted by stem)."""
    from ravi.eval.sa3_infer import render_prompt

    clips = []
    for wav in sorted(gen_dir.glob("*.wav")):
        meta_path = wav.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        key, seed = seed_split(wav.stem)
        prompt = meta.get("prompt_rendered") or (render_prompt(meta["prompt"]) if meta.get("prompt") else None)
        if prompt is None and captions:
            prompt = captions.get(str(meta.get("track_id") or key)) or captions.get(wav.stem)
        clips.append({"stem": wav.stem, "prompt_key": key, "path": wav, "prompt": prompt,
                      "seed": meta.get("seed", seed), "grid_seed": seed})
    return clips


def load_test_captions(spec: str | None, arm: str) -> dict[str, str]:
    """Rendered test captions by track id from a regions JSON, a caption JSONL or a directory of
    ``{track_id}.txt`` (module docstring). Missing or empty -> SystemExit."""
    from ravi.eval.sa3_infer import render_prompt

    hint = ("pass --test-captions eval/regions_test.json (select_regions: tracks[id].captions[--caption-arm]) "
            "or captions/midasheng_test.jsonl (track_id + caption), or a directory of {track_id}.txt")
    if not spec:
        raise SystemExit(f"--test-captions is required to evaluate generated clips and test_orig; {hint}")
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"--test-captions {p}: no such file or directory; {hint}")
    raw: dict[str, str] = {}
    if p.is_dir():
        raw = {f.stem: f.read_text(encoding="utf-8") for f in sorted(p.glob("*.txt"))}
    elif p.suffix == ".jsonl":
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{p}:{i + 1}: bad JSON ({e})") from None
            text = d.get("caption") or d.get("prompt")
            if d.get("track_id") is not None and text:
                raw[str(d["track_id"])] = str(text)
    else:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p}: bad JSON ({e})") from None
        tracks = doc.get("tracks") if isinstance(doc, dict) else None
        if not isinstance(tracks, dict):
            raise SystemExit(f"--test-captions {p}: JSON without a 'tracks' mapping; {hint}")
        for tid, t in tracks.items():
            text = _first(t, "captions", arm)
            if text:
                raw[str(tid)] = str(text)
    caps = {k: render_prompt(v) for k, v in raw.items() if v and v.strip()}
    if not caps:
        raise SystemExit(f"--test-captions {p}: no captions found (arm {arm!r} for a regions file); {hint}")
    return caps


def load_families(spec: str | None) -> dict[str, Any]:
    """``alt_family`` by track id from a ``select_regions`` JSON (``tracks`` and ``candidates``)."""
    if not spec or not Path(spec).is_file() or Path(spec).suffix != ".json":
        return {}
    doc = json.loads(Path(spec).read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for tid, t in (doc.get("tracks") or {}).items():
        if isinstance(t, dict) and t.get("alt_family") is not None:
            out[str(tid)] = t["alt_family"]
    for r in list(doc.get("regions") or []) + list(doc.get("candidates") or []):
        if isinstance(r, dict) and r.get("alt_family") is not None:
            out.setdefault(str(r["track_id"]), r["alt_family"])
    return out


# --------------------------------------------------------------------------- audio


def load(path: Path) -> tuple[np.ndarray, int]:
    from ravi.eval.sa3_infer import load_audio

    return load_audio(path)


def match_rate(audio: np.ndarray, sr: int, target_sr: int, channels: int) -> np.ndarray:
    from ravi.eval.sa3_infer import to_model_audio

    a, _ = to_model_audio(audio, sr, target_sr, channels)
    return a


def lufs(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
    from ravi.eval.embed import lufs_normalise

    return lufs_normalise(audio, sr)


# --------------------------------------------------------------------------- structure


def structure_boundaries(path: Path | None) -> np.ndarray | None:
    """Interior boundaries (segment starts/ends, without the first start and the last end) of a
    common-schema structure document; None when there is no document."""
    if path is None or not path.exists():
        return None
    from ravi.structure import schema

    doc = schema.load(path)
    segs = doc["segments"]
    if not segs:
        return np.zeros(0)
    b = sorted({float(s["start"]) for s in segs[1:]} | {float(s["end"]) for s in segs[:-1]})
    return np.asarray(b, dtype=np.float64)


def boundaries_kept(pred: np.ndarray | None, truth: Sequence[float], tol_s: float = STRUCT_TOL_S) -> float:
    """Fraction of ``truth`` boundaries with a predicted boundary within ``tol_s``."""
    if pred is None or not len(truth):
        return math.nan
    if pred.size == 0:
        return 0.0
    return float(np.mean([np.min(np.abs(pred - t)) <= tol_s for t in truth]))


def structure_path(structure_dir: Path | None, root: Path, wav: Path | None) -> Path | None:
    if structure_dir is None or wav is None:
        return None
    try:
        rel = wav.relative_to(root)
    except ValueError:
        return None
    return (structure_dir / rel).with_suffix(".json")


# --------------------------------------------------------------------------- region metrics


class OrigCache:
    """Per-region analysis of the original context, shared by all conditions."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}


def _nanmean(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan


def region_row(
    cond: str, reg: Region, gen_ctx: np.ndarray, orig_ctx: np.ndarray, sr: int,
    extractors: dict[str, Any], cache: OrigCache, args: argparse.Namespace,
    ref_means: dict[str, np.ndarray] | None = None, struct_doc: Path | None = None,
    orig_struct_doc: Path | None = None,
) -> dict[str, Any]:
    if gen_ctx.shape != orig_ctx.shape:
        raise SystemExit(f"{cond}/{reg.region_id}: generated context {gen_ctx.shape} != original {orig_ctx.shape}")
    m0, m1 = reg.m0, reg.m1
    seams_t = [m0 / sr, m1 / sr]
    oc = cache.data.setdefault(reg.region_key or reg.region_id, {})  # the original is the same for every seed
    if "novelty" not in oc:
        oc["novelty"] = S.novelty_curve(orig_ctx, sr, args.kernel_s)
        oc["beats"] = S.beat_times(orig_ctx, sr, args.tempo)
        oc["bpm"] = S.global_tempo(orig_ctx, sr)
        oc["grid_beats"] = S.grid_beats(orig_ctx, sr, oc["bpm"], args.grid_tightness)
        outside = np.concatenate([orig_ctx[:m0], orig_ctx[m1:]])
        oc["key"] = S.estimate_key(outside, sr) if outside.shape[0] > sr else S.KeyEstimate(None, None, None, math.nan, math.nan, math.nan)
        oc["outside"] = outside
        oc["orig_region_lufs"] = lufs(orig_ctx[m0:m1], sr) if any(n in LOUDNESS_NORMALISED for n in extractors) else None
        oc["struct_orig"] = boundaries_kept(structure_boundaries(orig_struct_doc), reg.section_bounds_ctx_s, args.struct_tol_s) \
            if args.structure_dir else math.nan
    row: dict[str, Any] = {
        "condition": cond,
        "region_id": reg.region_id,
        "region_key": reg.region_key or seed_split(reg.region_id)[0],
        "seed": reg.seed,
        "track_id": reg.track_id,
        "section_index": reg.section_index,
        "label": reg.label,
        "mask_start_s": reg.mask_start_s,
        "mask_end_s": reg.mask_end_s,
        "mask_in_context_start_s": m0 / sr,
        "mask_in_context_end_s": m1 / sr,
        "region_s": (m1 - m0) / sr,
        "sr": sr,
    }
    gen_region, orig_region = gen_ctx[m0:m1], orig_ctx[m0:m1]
    for name, ex in extractors.items():
        if name in LOUDNESS_NORMALISED:
            g_n, g_st = lufs(gen_region, sr)
            o_n, o_st = oc["orig_region_lufs"]
            row["lufs_gen_region"], row["lufs_orig_region"] = g_st["lufs_in"], o_st["lufs_in"]
            e = ex.embed([g_n, o_n], sr=sr)
        else:
            e = ex.embed([gen_region, orig_region], sr=sr)
        row[f"region_sim_{name}"] = float(M.region_similarity(e[:1], e[1:])[0])
        if name in CENTRED_SIM_EXTRACTORS and ref_means and name in ref_means:
            row[f"region_sim_{name}_centred"] = float(M.centred_cosine(e[:1], e[1:], ref_means[name])[0])
    g_nov = S.novelty_curve(gen_ctx, sr, args.kernel_s)
    gs = S.seam_score(gen_ctx, sr, seams_t, args.seam_window_s, args.seam_tol_s, novelty=g_nov)
    os_ = S.seam_score(orig_ctx, sr, seams_t, args.seam_window_s, args.seam_tol_s, novelty=oc["novelty"])
    for i, side in enumerate(("start", "end")):
        row[f"seam_score_{side}"] = float(gs[i])
        row[f"seam_score_orig_{side}"] = float(os_[i])
        row[f"seam_delta_{side}"] = float(gs[i] - os_[i])
    row["seam_delta"] = _nanmean(gs - os_)
    row["abs_seam_delta"] = _nanmean(np.abs(gs - os_))
    g_beats = S.beat_times(gen_ctx, sr, args.tempo)
    g_grid = S.grid_beats(gen_ctx, sr, oc["bpm"], args.grid_tightness)
    agree, off = S.beat_grid_agreement(None, None, sr, seams_t, args.beat_window_s, args.beat_tol_s,
                                       gen_beats=g_grid, orig_beats=oc["grid_beats"])
    gcv = S.beat_regularity(gen_ctx, sr, seams_t, args.beat_window_s, beats=g_beats)
    ocv = S.beat_regularity(orig_ctx, sr, seams_t, args.beat_window_s, beats=oc["beats"])
    for i, side in enumerate(("start", "end")):
        row[f"beat_grid_agree_{side}"] = float(agree[i])
        row[f"beat_grid_offset_ms_{side}"] = float(1000.0 * off[i])
        row[f"beat_cv_{side}"] = float(gcv[i])
        row[f"beat_cv_orig_{side}"] = float(ocv[i])
    row["beat_grid_agree"] = _nanmean(agree)
    row["beat_grid_offset_ms"] = 1000.0 * _nanmean(off)
    row["beat_cv"] = _nanmean(gcv)
    row["beat_cv_orig"] = _nanmean(ocv)
    row["beat_cv_delta"] = _nanmean(gcv - ocv)
    agree_k, conf = S.key_agreement(gen_region, oc["outside"], sr, args.key_gate, context_key=oc["key"])
    row["key_agree"] = math.nan if agree_k is None else float(agree_k)
    row["key_conf_orig"] = float(conf)
    row["key_orig"] = oc["key"].key
    row["key_gen"] = S.estimate_key(gen_region, sr).key if agree_k is not None else None
    if args.structure_dir:
        row["struct_n_bounds"] = len(reg.section_bounds_ctx_s)
        row["struct_kept"] = boundaries_kept(structure_boundaries(struct_doc), reg.section_bounds_ctx_s, args.struct_tol_s)
        row["struct_kept_orig"] = oc["struct_orig"]
        row["struct_kept_delta"] = row["struct_kept"] - row["struct_kept_orig"]
    return row


def pick_donor(reg: Region, tracks: list[str], families: dict[str, Any]) -> tuple[str, bool]:
    """The next track after ``reg``'s (sorted, wrapping) of a different ``alt_family``; returns
    ``(track_id, family_checked)``. Without family information for a pair, only a different
    track is required (``family_checked`` False)."""
    i = tracks.index(reg.track_id)
    fam = families.get(reg.track_id)
    fallback = None
    for k in range(1, len(tracks)):
        t = tracks[(i + k) % len(tracks)]
        f = families.get(t)
        if fam is not None and f is not None:
            if f != fam:
                return t, True
        elif fallback is None:
            fallback = t
    if fallback is None:
        raise SystemExit(f"lower anchor: no donor track of another alt_family for {reg.track_id!r}")
    return fallback, False


def lower_anchor_ctx(reg: Region, orig_ctx: np.ndarray, sr: int, donor: Region,
                     test_audio: Path | None) -> tuple[np.ndarray, str]:
    """The original context with the region replaced by a cut from the donor track."""
    src = None
    if test_audio is not None:
        p = test_audio / f"{donor.track_id}.wav"
        if p.exists():
            a, dsr = load(p)
            src, how = match_rate(a, dsr, sr, orig_ctx.shape[1]), f"test-audio:{p.name}"
    if src is None:
        if "orig" not in donor.files:
            raise SystemExit(f"lower anchor: no audio for donor track {donor.track_id!r}")
        a, dsr = load(donor.files["orig"])
        src, how = match_rate(a, dsr, sr, orig_ctx.shape[1]), f"orig-context:{donor.files['orig'].name}"
    fill = lower_anchor(src, reg.m1 - reg.m0)
    if fill.shape[1] != orig_ctx.shape[1]:
        fill = np.repeat(fill[:, :1], orig_ctx.shape[1], axis=1)
    return splice(orig_ctx, fill.astype(orig_ctx.dtype), reg.m0, reg.m1, seconds_to_samples(DEFAULT_XFADE_S, sr)), how


def average_seeds(df, key: str, group_cols: Sequence[str] = ("condition",)):
    """Mean of every numeric column over the seeds of one ``(condition, key)``; other columns
    take their first value; ``n_seeds`` counts the rows averaged."""
    import pandas as pd

    if df is None or len(df) == 0:
        return df
    by = list(group_cols) + [key]
    num = [c for c in df.columns if c not in by and c != "seed" and pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c])]
    other = [c for c in df.columns if c not in by and c not in num and c != "seed"]
    g = df.groupby(by, sort=False)
    out = g[num].mean() if num else g.size().to_frame("_n").drop(columns="_n")
    if other:
        out = out.join(g[other].first())
    out["n_seeds"] = g.size()
    return out.reset_index()


# --------------------------------------------------------------------------- text-to-music


def t2m_clips(
    cond: str, clips: list[dict[str, Any]], extractors: dict[str, Any], ref: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    """Per-clip rows (prompt, CLAP text score, loudness, memorisation max-cosine) and each
    extractor's window embeddings per clip."""
    rows = [{"condition": cond, "stem": c["stem"], "prompt_key": c.get("prompt_key", c["stem"]),
             "prompt": c["prompt"], "seed": c.get("seed"), "file": str(c["path"])} for c in clips]
    chunks_by: dict[str, list[np.ndarray]] = {}
    if not clips:
        return rows, chunks_by
    normed: list[Any] | None = None
    if any(n in LOUDNESS_NORMALISED for n in extractors):
        normed = []
        for r, c in zip(rows, clips):
            a, sr = load(c["path"])
            a_n, st = lufs(a, sr)
            r["lufs_in"], r["lufs_gain_db"], r["lufs_peak_limited"] = st["lufs_in"], st["gain_db"], st["peak_limited"]
            normed.append((a_n, sr))
    paths = [c["path"] for c in clips]
    for name, ex in extractors.items():
        chunks = ex.embed_chunks(normed if name in LOUDNESS_NORMALISED else paths)
        chunks_by[name] = chunks
        pooled = np.stack([ex.pool(c) for c in chunks])
        if name == "clap":
            have = [i for i, c in enumerate(clips) if c["prompt"]]
            if have:
                texts = [clips[i]["prompt"] for i in have]
                sc = M.clap_text_audio_score(ex.embed_text(texts), pooled[have])
                trunc = ex.text_truncated(texts) if hasattr(ex, "text_truncated") else [None] * len(have)
                for j, i in enumerate(have):
                    rows[i]["clap_text"] = float(sc[j])
                    rows[i]["prompt_truncated"] = trunc[j]
            for r in rows:
                r.setdefault("clap_text", math.nan)
                r.setdefault("prompt_truncated", None)
        if name in ref:
            gen_rows = np.concatenate(chunks)
            owner = np.concatenate([np.full(c.shape[0], i) for i, c in enumerate(chunks)])
            memo = M.memorisation_check(gen_rows, ref[name]["rows"], args.memo_thresh)
            for i in range(len(clips)):
                sel = np.flatnonzero(owner == i)
                k = sel[int(np.argmax(memo.max_cos[sel]))]
                rows[i][f"memo_max_{name}"] = float(memo.max_cos[k])
                rows[i][f"memo_track_{name}"] = ref[name]["owner_ids"][int(memo.nearest[k])]
    return rows, chunks_by


def memo_flags(rows: list[dict[str, Any]], null_p99: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    """Set ``memo_flag_{x}`` / ``memo_flag_null_{x}`` on the rows; return the summary counts."""
    out: dict[str, Any] = {}
    for name in MEMO_FLAG_EXTRACTORS:
        col = f"memo_max_{name}"
        if not rows or col not in rows[0]:
            continue
        thr = null_p99.get(name, math.nan)
        for r in rows:
            r[f"memo_flag_{name}"] = bool(r[col] > args.memo_thresh)
            r[f"memo_flag_null_{name}"] = bool(r[col] > thr) if math.isfinite(thr) else None
        out[f"memo_flagged_{name}"] = int(sum(r[f"memo_flag_{name}"] for r in rows))
        out[f"memo_flagged_null_{name}"] = (int(sum(bool(r[f"memo_flag_null_{name}"]) for r in rows))
                                            if math.isfinite(thr) else math.nan)
        out[f"memo_null_p99_{name}"] = thr
    for k in [k for k in (rows[0] if rows else {}) if k.startswith("memo_max_")]:
        out[k] = float(np.max([r[k] for r in rows]))
    return out


def dist_metrics(
    chunks_by: dict[str, list[np.ndarray]], sel: list[int], ref: dict[str, Any], args: argparse.Namespace,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """FAD∞ / KAD (+ clip-bootstrap CIs) of the selected clips in the reference's PCA space.
    Returns (summary columns, diagnostic columns for ``metrics_dist.parquet``)."""
    summ: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    for name, chunks in chunks_by.items():
        if name not in ref:
            continue
        r = ref[name]
        picked = [chunks[i] for i in sel]
        rows = np.concatenate(picked).astype(np.float64)
        owner = np.concatenate([np.full(c.shape[0], j) for j, c in enumerate(picked)])
        summ[f"n_windows_{name}"] = int(rows.shape[0])
        diag[f"fad_{name}_naive"] = M.fad(r["stats_full"], rows) if rows.shape[0] >= 2 else math.nan
        x = M.pca_project(r["pca"], rows) if r["pca"] is not None else rows
        d = int(x.shape[1])
        summ[f"pca_d_{name}"] = d
        summ[f"pca_var_{name}"] = r["pca"].explained if r["pca"] is not None else 1.0
        diag[f"fad_{name}_pca_full_sample"] = M.fad(r["stats"], x) if x.shape[0] >= 2 else math.nan
        min_n = args.fad_inf_min_n or M.fad_inf_min_n(d)
        summ[f"fad_inf_min_n_{name}"] = int(min_n)
        try:
            fi = M.fad_inf(r["stats"], x, replace=False, seed=args.seed, min_n=min_n)
            summ.update({f"fad_inf_{name}": fi.score, f"fad_inf_slope_{name}": fi.slope, f"fad_inf_r2_{name}": fi.r2,
                         f"fad_inf_n_lo_{name}": fi.n_lo, f"fad_inf_n_hi_{name}": fi.n_hi})
            diag[f"fad_inf_status_{name}"] = "ok"
        except ValueError as e:
            summ.update({f"fad_inf_{name}": math.nan, f"fad_inf_slope_{name}": math.nan, f"fad_inf_r2_{name}": math.nan,
                         f"fad_inf_n_lo_{name}": math.nan, f"fad_inf_n_hi_{name}": math.nan})
            diag[f"fad_inf_status_{name}"] = str(e)
            log(f"WARNING {tag} {name}: FAD-inf not computed ({e})")
        summ[f"kad_{name}"] = M.kad(r["kad"], x) if x.shape[0] >= 2 else math.nan
        if args.dist_boot > 0 and len(picked) >= 2:
            b = M.clip_bootstrap(r["stats"], r["kad"], x, owner, n_boot=args.dist_boot, seed=args.seed, min_n=min_n)
        else:
            b = {"fad_inf": (math.nan, math.nan), "kad": (math.nan, math.nan)}
        summ[f"fad_inf_{name}_ci_lo"], summ[f"fad_inf_{name}_ci_hi"] = b["fad_inf"]
        summ[f"kad_{name}_ci_lo"], summ[f"kad_{name}_ci_hi"] = b["kad"]
    return summ, diag


def build_reference(train_paths: list[Path], train_ids: list[str], extractors: dict[str, Any],
                    args: argparse.Namespace) -> dict[str, Any]:
    """Training-corpus window embeddings per extractor (already -16 LUFS: prepare_audio), their
    full-dimension stats (naive FAD), the PCA fitted on them, and the reduced-space FAD / KAD
    reference terms."""
    ref: dict[str, Any] = {}
    if not train_paths:
        return ref
    for name, ex in extractors.items():
        chunks = ex.embed_chunks(train_paths)
        rows = np.concatenate(chunks).astype(np.float64)
        owner_ids = [tid for tid, c in zip(train_ids, chunks) for _ in range(c.shape[0])]
        pca = M.pca_fit(rows, args.pca_dims) if args.pca_dims and args.pca_dims < rows.shape[1] else None
        x = M.pca_project(pca, rows) if pca is not None else rows
        ref[name] = {
            "rows": rows,
            "owner_ids": owner_ids,
            "mean": rows.mean(axis=0),
            "stats_full": M.gaussian_stats(rows),
            "pca": pca,
            "stats": M.gaussian_stats(x),
            "kad": M.kad_reference(x, seed=args.seed),
            "n_files": len(train_paths),
        }
    return ref


# --------------------------------------------------------------------------- comparisons


def compare_all(order: list[str], table, key: str, metrics: dict[str, int], family: str, base: str,
                args: argparse.Namespace, cluster_col: str | None = None, unit: str = "region") -> list[dict[str, Any]]:
    """Paired comparisons of every condition pair on every metric (module docstring)."""
    out: list[dict[str, Any]] = []
    if table is None or len(table) == 0:
        return out
    present = [c for c in order if c in set(table["condition"])]
    clusters = None
    if cluster_col and cluster_col in table.columns:
        clusters = table.drop_duplicates(key).set_index(key)[cluster_col]
    for metric, direction in metrics.items():
        if metric not in table.columns:
            continue
        wide = table.pivot(index=key, columns="condition", values=metric).astype(float)
        cl = None if clusters is None else clusters.reindex(wide.index).astype(str).to_numpy()
        levels = ["track", unit] if cl is not None else [unit]
        better, worse = ("p_greater", "p_less") if direction > 0 else ("p_less", "p_greater")
        for level in levels:
            rows = []
            for i, a in enumerate(present):
                for b in present[i + 1 :]:
                    ca, cb = (b, a) if a == base else (a, b)
                    if ca not in wide.columns or cb not in wide.columns:
                        continue
                    xa, xb = wide[ca].to_numpy(float), wide[cb].to_numpy(float)
                    if level == "track":
                        am, bm, _ = ST.cluster_medians(xa, xb, cl)
                        r = ST.paired_compare(am, bm, n_boot=args.n_boot, seed=args.seed)
                        r["ci_method"] = "bootstrap over tracks (per-track medians)"
                        r["n_clusters"] = r["n"]
                    else:
                        r = ST.paired_compare(xa, xb, n_boot=args.n_boot, seed=args.seed, clusters=cl)
                    rows.append({"family": family, "metric": metric, "level": level, "primary": level == levels[0],
                                 "higher_is_better": direction > 0, "cond_a": ca, "cond_b": cb,
                                 "vs_base": base in (ca, cb), **r,
                                 "p_a_better": r[better], "p_a_worse": r[worse]})
            m = int(np.isfinite([r["p_two_sided"] for r in rows]).sum())
            for col, src in (("p_holm_two_sided", "p_two_sided"), ("p_holm_a_worse", "p_a_worse"),
                             ("p_holm_a_better", "p_a_better")):
                for r, p in zip(rows, ST.holm([r[src] for r in rows])):
                    r[col] = float(p)
            for r in rows:
                r["holm_m"] = m
                r["holm_scope"] = f"{family}/{metric}/{level}: all condition pairs"
            out += rows
    return out


# --------------------------------------------------------------------------- main


def summarise_regions(df, order: list[str]) -> dict[str, dict[str, Any]]:
    """Per condition, on the seed-averaged table: counts and mean / median of every metric."""
    out: dict[str, dict[str, Any]] = {}
    if df is None or len(df) == 0:
        return out
    cols = list(REGION_METRICS) + ["seam_score_start", "seam_score_end", "beat_cv_orig", "struct_kept_orig",
                                   "lufs_gen_region"]
    for cond, g in df.groupby("condition", sort=False):
        s: dict[str, Any] = {"n_regions": int(len(g)), "n_tracks": int(g["track_id"].nunique()),
                             "n_region_outputs": int(g["n_seeds"].sum()),
                             "n_seeds_min": int(g["n_seeds"].min()), "n_seeds_max": int(g["n_seeds"].max())}
        for c in cols:
            if c not in g.columns:
                continue
            v = g[c].to_numpy(float)
            ok = np.isfinite(v)
            s[f"{c}_n"] = int(ok.sum())
            s[f"{c}_mean"] = float(v[ok].mean()) if ok.any() else math.nan
            s[f"{c}_median"] = float(np.median(v[ok])) if ok.any() else math.nan
        out[cond] = s
    return out


def resolve_train(args: argparse.Namespace) -> tuple[list[Path], list[str], list[str]]:
    if args.train_dir is None:
        return [], [], []
    if not args.splits:
        raise SystemExit("--train-dir needs --splits (manifests/splits.json): the reference set must be the "
                         "train split, checked against val/test")
    tdir = Path(args.train_dir)
    sp = json.loads(Path(args.splits).read_text())
    ids = list(sp["train"])
    test_ids = list(sp.get("test", []))
    leak = sorted(set(ids) & (set(test_ids) | set(sp.get("val", []))))
    if leak:
        raise SystemExit(f"splits: {len(leak)} train ids also in val/test, e.g. {leak[0]!r}")
    paths = [tdir / f"{i}.wav" for i in ids]
    missing = [p for p in paths if not p.exists()]
    if missing and not args.limit:
        raise SystemExit(f"--train-dir: {len(missing)} of {len(paths)} train WAVs missing, e.g. {missing[0]}")
    pairs = [(p, i) for p, i in zip(paths, ids) if p.exists()]
    if args.limit:
        pairs = pairs[: args.limit]
    return [p for p, _ in pairs], [i for _, i in pairs], test_ids


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ravi.eval.run_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-root", default="eval")
    ap.add_argument("--out-dir", default=None, help="default: --eval-root")
    ap.add_argument("--base", default=DEFAULT_BASE, help="the no-LoRA control condition")
    ap.add_argument("--conditions", default=None, help="comma list (default: all discovered)")
    ap.add_argument("--test-audio", default=None, help="test originals {track_id}.wav")
    ap.add_argument("--train-dir", default=None, help="training corpus WAVs {track_id}.wav (FAD/KAD reference); needs --splits")
    ap.add_argument("--splits", default=None, help="manifests/splits.json: selects the train ids, guards leakage")
    ap.add_argument("--test-captions", default=None,
                    help="eval/regions_test.json | captions/midasheng_test.jsonl | dir of {track_id}.txt")
    ap.add_argument("--caption-arm", default="B", help="arm of the regions-file captions (default B)")
    ap.add_argument("--regions-json", default=None, help="select_regions file for alt_family (default: --test-captions if a .json)")
    ap.add_argument("--lower-anchor", default=DEFAULT_LOWER_ANCHOR, help="existing lower-anchor condition (grid's)")
    ap.add_argument("--structure-dir", default=None, help="common-schema structure docs mirroring --eval-root")
    ap.add_argument("--struct-tol-s", type=float, default=STRUCT_TOL_S)
    ap.add_argument("--variant", choices=("spliced", "raw"), default="spliced")
    ap.add_argument("--extractors", default=",".join(DEFAULT_EXTRACTORS))
    ap.add_argument("--cache-dir", default=None, help="embedding cache (default: {eval-root}/.emb_cache)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--clap-ckpt", default=None, help="local music_audioset_epoch_15_esc_90.14.pt (default: HF hub)")
    ap.add_argument("--mert-layer", type=int, default=12)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N regions / clips / train files")
    ap.add_argument("--kernel-s", type=float, default=S.DEFAULT_KERNEL_S)
    ap.add_argument("--seam-window-s", type=float, default=1.0)
    ap.add_argument("--seam-tol-s", type=float, default=0.3)
    ap.add_argument("--beat-window-s", type=float, default=4.0)
    ap.add_argument("--beat-tol-s", type=float, default=S.BEAT_TOL_S)
    ap.add_argument("--tempo", choices=("local", "global"), default=S.DEFAULT_TEMPO, help="beat tracker for beat_cv")
    ap.add_argument("--grid-tightness", type=float, default=S.GRID_TIGHTNESS,
                    help="beat_track tightness for the beat-grid agreement (both tracks at the original's tempo)")
    ap.add_argument("--key-gate", type=float, default=S.KEY_CONF_GATE)
    ap.add_argument("--memo-thresh", type=float, default=0.95)
    ap.add_argument("--pca-dims", type=int, default=64, help="PCA dims for FAD/FAD-inf/KAD (0 = full dimension)")
    ap.add_argument("--fad-inf-min-n", type=int, default=0, help="FAD-inf grid floor (default max(3 d, 500))")
    ap.add_argument("--dist-boot", type=int, default=200, help="clip-level bootstrap resamples for FAD-inf/KAD CIs")
    ap.add_argument("--cluster", choices=("track_id", "none"), default="track_id",
                    help="primary inpainting tests on per-track medians (default) or regions only")
    ap.add_argument("--n-boot", type=int, default=ST.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=M.DEFAULT_SEED)
    return ap


def default_extractor_factory(name: str, args: argparse.Namespace):
    from ravi.eval import embed as E

    kw: dict[str, Any] = {"cache_dir": args.cache_dir, "device": args.device}
    if name == "clap":
        kw["ckpt"] = args.clap_ckpt
    elif name == "mert":
        kw["layer"] = args.mert_layer
    return E.get_extractor(name, **kw)


def _versions() -> dict[str, str | None]:
    import importlib

    out = {}
    for m in ("numpy", "scipy", "librosa", "pandas", "soundfile", "pyloudnorm", "torch", "torchaudio",
              "transformers", "laion_clap"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:  # noqa: BLE001 - optional in the test env
            out[m] = None
    return out


def _check_sources(regions_by_cond: dict[str, dict[str, Region]], region_ids: list[str], union: dict[str, Region]) -> None:
    """All conditions agree on each region's geometry and on each track's source audio."""
    sha_by_track: dict[str, tuple[str, Path]] = {}
    for rid in region_ids:
        for name, regs in regions_by_cond.items():
            r = regs.get(rid)
            if r is None:
                continue
            if (r.m0, r.m1) != (union[rid].m0, union[rid].m1):
                raise SystemExit(f"{name}/{rid}: mask differs from {union[rid].sidecar}")
            if r.audio_sha256:
                seen = sha_by_track.setdefault(r.track_id, (r.audio_sha256, r.sidecar))
                if seen[0] != r.audio_sha256:
                    raise SystemExit(f"{r.sidecar}: audio.sha256 {r.audio_sha256[:12]}… differs from "
                                     f"{seen[1]} ({seen[0][:12]}…) for track {r.track_id!r}")


def main(argv: Sequence[str] | None = None,
         extractor_factory: Callable[[str, argparse.Namespace], Any] | None = None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    root = Path(args.eval_root)
    if not root.is_dir():
        raise SystemExit(f"--eval-root {root}: not a directory")
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = str(root / ".emb_cache")
    if args.structure_dir:
        args.structure_dir = str(Path(args.structure_dir))
    structure_dir = Path(args.structure_dir) if args.structure_dir else None
    test_audio = Path(args.test_audio) if args.test_audio else None

    conds = discover_conditions(root)
    if args.conditions:
        want = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = sorted(set(want) - {c.name for c in conds})
        if unknown:
            raise SystemExit(f"--conditions: not found under {root}: {unknown}")
        conds = [c for c in conds if c.name in want or (c.name == args.lower_anchor and not args.no_anchors)]
    if not conds:
        raise SystemExit(f"no conditions under {root} (expected {{condition}}/inpaint or {{condition}}/gen)")
    have_lower = any(c.name == args.lower_anchor and c.inpaint_dir for c in conds) and not args.no_anchors
    if args.no_anchors:
        conds = [c for c in conds if c.name != args.lower_anchor]
    names_order = [c.name for c in conds if c.name != args.lower_anchor]
    if args.base in names_order:
        names_order = [args.base] + [n for n in names_order if n != args.base]
    else:
        log(f"WARNING: base condition {args.base!r} not found; comparisons have no base")
    gen_conds = [c for c in conds if c.gen_dir]
    captions = load_test_captions(args.test_captions, args.caption_arm) if (gen_conds or args.test_captions) else None
    if gen_conds and test_audio is None:
        raise SystemExit("--test-audio is required with generated clips: test_orig is the real-music reference row")
    families = load_families(args.regions_json or args.test_captions)

    factory = extractor_factory or default_extractor_factory
    names = [n for n in args.extractors.split(",") if n]
    extractors = {n: factory(n, args) for n in names}

    # ---- reference first: the region metrics need its mean (centred MERT similarity)
    train_paths, train_ids, test_ids = resolve_train(args)
    ref = build_reference(train_paths, train_ids, extractors, args)
    ref_means = {n: r["mean"] for n, r in ref.items()}

    # ---- inpainting
    regions_by_cond = {c.name: read_regions(c.inpaint_dir, args.variant) for c in conds if c.inpaint_dir}
    union: dict[str, Region] = {}
    for name in names_order + ([args.lower_anchor] if have_lower else []):
        for rid, reg in regions_by_cond.get(name, {}).items():
            union.setdefault(rid, reg)
    region_ids = sorted(union)
    if args.limit:
        keys = sorted({union[r].region_key for r in region_ids})[: args.limit]
        region_ids = [r for r in region_ids if union[r].region_key in set(keys)]
    _check_sources(regions_by_cond, region_ids, union)
    cache = OrigCache()
    region_rows: list[dict[str, Any]] = []
    anchor_src: dict[str, dict[str, Any]] = {}
    tracks = sorted({union[r].track_id for r in region_ids})
    first_by_track: dict[str, Region] = {}
    for rid in region_ids:
        first_by_track.setdefault(union[rid].track_id, union[rid])
    anchored: set[str] = set()
    cond_order = names_order + ([args.lower_anchor] if have_lower else [])
    for rid in region_ids:
        reg = union[rid]
        if "orig" not in reg.files:
            raise SystemExit(f"{reg.sidecar}: missing {rid}_orig.wav (the original context)")
        orig, sr = load(reg.files["orig"])
        orig_doc = structure_path(structure_dir, root, reg.files["orig"])
        for name in cond_order:
            r = regions_by_cond.get(name, {}).get(rid)
            if r is None:
                continue
            gen, gsr = load(r.files[args.variant])
            if gsr != sr:
                raise SystemExit(f"{name}/{rid}: {gsr} Hz vs original context {sr} Hz")
            region_rows.append(region_row(name, r, gen, orig, sr, extractors, cache, args, ref_means,
                                          structure_path(structure_dir, root, r.files[args.variant]), orig_doc))
        if not args.no_anchors and reg.region_key not in anchored:  # anchors once per region, not per seed
            anchored.add(reg.region_key)
            region_rows.append(region_row("upper", reg, orig.copy(), orig, sr, extractors, cache, args, ref_means,
                                          orig_doc, orig_doc))
            if not have_lower and len(tracks) >= 2:
                donor_tid, checked = pick_donor(reg, tracks, families)
                low, how = lower_anchor_ctx(reg, orig, sr, first_by_track[donor_tid], test_audio)
                anchor_src[reg.region_key] = {"donor": donor_tid, "source": how, "family_checked": checked,
                                              "family": families.get(reg.track_id), "donor_family": families.get(donor_tid)}
                row = region_row("lower", reg, low, orig, sr, extractors, cache, args, ref_means, None, orig_doc)
                row["lower_donor"] = f"{donor_tid} ({how})"
                row["lower_family_checked"] = checked
                region_rows.append(row)
        log(f"region {rid}: done")
    regions_df = pd.DataFrame(region_rows)
    lower_name = args.lower_anchor if have_lower else "lower"
    order = names_order + ([] if args.no_anchors else ["upper", lower_name])
    regions_avg = average_seeds(regions_df, "region_key") if len(regions_df) else regions_df

    # ---- text-to-music: test_orig first (it calibrates the memorisation null)
    gen_by_cond = {c.name: read_gen(c.gen_dir, captions) for c in conds if c.gen_dir and c.name in names_order}
    if args.limit:
        gen_by_cond = {k: v[: args.limit] for k, v in gen_by_cond.items()}
    t2m: dict[str, tuple[list[dict[str, Any]], dict[str, list[np.ndarray]]]] = {}
    if test_audio is not None and captions:
        ids = sorted(t for t in captions if (test_audio / f"{t}.wav").exists())
        if test_ids:
            stray = sorted(set(ids) - set(test_ids))
            if stray:
                raise SystemExit(f"--test-captions: {len(stray)} tracks are not in the test split, e.g. {stray[0]!r}")
        if set(ids) & set(train_ids):
            raise SystemExit("test originals overlap the FAD reference set")
        if args.limit:
            ids = ids[: args.limit]
        if not ids and gen_by_cond:
            raise SystemExit(f"--test-captions / --test-audio: no captioned test track has a WAV in {test_audio}")
        clips = [{"stem": i, "prompt_key": i, "path": test_audio / f"{i}.wav", "prompt": captions[i]} for i in ids]
        if clips:
            t2m["test_orig"] = t2m_clips("test_orig", clips, extractors, ref, args)
    for name in names_order:
        if name in gen_by_cond:
            t2m[name] = t2m_clips(name, gen_by_cond[name], extractors, ref, args)
    null_p99: dict[str, float] = {}
    if "test_orig" in t2m:
        for x in MEMO_FLAG_EXTRACTORS:
            vals = [r.get(f"memo_max_{x}", math.nan) for r in t2m["test_orig"][0]]
            null_p99[x] = M.null_threshold(np.asarray(vals, dtype=float), 99.0)
    t2m_order = [n for n in names_order if n in t2m] + (["test_orig"] if "test_orig" in t2m else [])
    n_match = min((len(t2m[n][0]) for n in t2m_order if t2m[n][0]), default=0)
    gen_summary: dict[str, dict[str, Any]] = {}
    dist_rows: list[dict[str, Any]] = []
    gen_rows: list[dict[str, Any]] = []
    subsample: dict[str, list[str]] = {}
    for name in t2m_order:
        rows, chunks_by = t2m[name]
        summ: dict[str, Any] = {"n_gen": len(rows), **memo_flags(rows, null_p99, args)}
        if rows:
            vals = np.array([r["clap_text"] for r in rows], dtype=float) if "clap_text" in rows[0] else np.array([])
            if vals.size:
                summ["clap_text_mean"] = _nanmean(vals)
                summ["clap_text_median"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else math.nan
                summ["n_prompt_truncated"] = int(sum(bool(r.get("prompt_truncated")) for r in rows))
        if ref and rows:
            n = len(rows)
            if n > n_match:
                sel = sorted(np.random.default_rng(args.seed).choice(n, size=n_match, replace=False).tolist())
            else:
                sel = list(range(n))
            subsample[name] = [rows[i]["stem"] for i in sel]
            summ["n_clips_dist"] = len(sel)
            summ["dist_subsampled"] = n > n_match
            ds, diag = dist_metrics(chunks_by, sel, ref, args, name)
            summ.update(ds)
            dist_rows.append({"condition": name, "n_clips_total": n, **ds, **diag})
        for r in rows:
            r["in_dist_subset"] = r["stem"] in set(subsample.get(name, []))
        gen_rows += rows
        gen_summary[name] = summ
    if "test_orig" in t2m:
        order.append("test_orig")
    gen_df = pd.DataFrame(gen_rows)
    gen_avg = average_seeds(gen_df[gen_df["condition"] != "test_orig"], "prompt_key") if len(gen_df) else None

    # ---- outputs
    regions_df.to_parquet(out_dir / "metrics_regions.parquet", index=False)
    if len(regions_avg):
        regions_avg.to_parquet(out_dir / "metrics_regions_avg.parquet", index=False)
    gen_df.to_parquet(out_dir / "metrics_gen.parquet", index=False)
    pd.DataFrame(dist_rows).to_parquet(out_dir / "metrics_dist.parquet", index=False)
    reg_summary = summarise_regions(regions_avg, order)
    summary_rows = []
    for name in order:
        if name not in reg_summary and name not in gen_summary:
            continue
        summary_rows.append({"condition": name, "is_base": name == args.base,
                             "is_anchor": name in ("upper", lower_name),
                             **reg_summary.get(name, {}), **gen_summary.get(name, {})})
    pd.DataFrame(summary_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    cluster_col = None if args.cluster == "none" else "track_id"
    comps = compare_all(order, regions_avg if len(regions_avg) else None, "region_key", REGION_METRICS, "inpaint",
                        args.base, args, cluster_col=cluster_col, unit="region")
    comps += compare_all(order, gen_avg, "prompt_key", GEN_METRICS, "t2m", args.base, args, unit="clip")
    pd.DataFrame(comps).to_csv(out_dir / "comparisons.csv", index=False)
    fit = {name: {x: {k: s.get(f"{k}_{x}") for k in ("fad_inf_n_lo", "fad_inf_n_hi", "fad_inf_r2", "pca_d", "pca_var")}
                  for x in ref} for name, s in gen_summary.items() if "n_clips_dist" in s}
    meta = {
        "tool": "ravi.eval.run_eval",
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "conditions": order,
        "base": args.base,
        "n_regions": int(regions_avg["region_key"].nunique()) if len(regions_avg) else 0,
        "n_region_outputs": len(region_ids),
        "seeds": "per-output rows averaged over _seed{s} per (condition, region_key / prompt_key) before summaries and tests",
        "variant": args.variant,
        "extractors": {n: (ex.config() if hasattr(ex, "config") else {}) for n, ex in extractors.items()},
        "reference": {n: {"n_files": r["n_files"], "n_windows": int(r["rows"].shape[0]),
                          "pca_d": None if r["pca"] is None else r["pca"].k,
                          "pca_explained_variance": None if r["pca"] is None else r["pca"].explained,
                          "kad_bandwidth": r["kad"].bandwidth} for n, r in ref.items()},
        "lower_anchor": {"source": "condition" if have_lower else ("built-in" if not args.no_anchors else None),
                         "name": lower_name, "donors": anchor_src},
        "settings": {k: getattr(args, k) for k in ("kernel_s", "seam_window_s", "seam_tol_s", "beat_window_s",
                                                   "beat_tol_s", "tempo", "grid_tightness", "key_gate", "memo_thresh", "n_boot",
                                                   "seed", "limit", "pca_dims", "dist_boot", "cluster",
                                                   "caption_arm", "struct_tol_s")},
        "test_captions": args.test_captions,
        "fad_inf": {"replace": False, "n_steps": 25, "min_n": args.fad_inf_min_n or "max(3 d, 500)",
                    "space": f"PCA {args.pca_dims} dims fitted on the reference" if args.pca_dims else "full dimension",
                    "fit": fit},
        "kad": {"estimator": "unbiased", "kernel": "rbf", "bandwidth": "median pairwise distance within the reference",
                "scale": M.KAD_SCALE},
        "distribution_metrics": {"matched_n_clips": n_match, "subsample_seed": args.seed,
                                 "clips_used": subsample, "bootstrap": f"{args.dist_boot} clip resamples, percentile 95%",
                                 "note": "relative comparisons at matched n, not unbiased absolute distances"},
        "memorisation": {"flag_extractors": list(MEMO_FLAG_EXTRACTORS), "fixed_thresh": args.memo_thresh,
                         "null": "99th percentile of test_orig memo_max vs the same training set", "null_p99": null_p99},
        "loudness": {"target_lufs": -16.0, "ceiling_dbtp": -1.0, "applied_to": list(LOUDNESS_NORMALISED),
                     "reference": "training corpus already normalised by ravi.data.prepare_audio"},
        "comparisons": {"cluster": args.cluster, "holm": "per (family, metric, level) over all condition pairs; "
                        "two-sided and each one-sided column separately"},
        "structure": {"dir": args.structure_dir, "tol_s": args.struct_tol_s},
        "versions": _versions(),
    }
    (out_dir / "metrics_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"wrote {out_dir}/metrics_regions.parquet ({len(regions_df)} rows), metrics_gen.parquet "
        f"({len(gen_df)} rows), metrics_dist.parquet, metrics_summary.csv, comparisons.csv ({len(comps)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
