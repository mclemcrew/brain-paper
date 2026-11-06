"""Build the seam-localisation listening test (plan Phase 4.3) from a ``ravi.eval.grid`` eval root.

    python -m ravi.eval.make_listening_test --eval-root eval/tier1 --conditions base,B6k,E6k \\
        --segments-dir eval/segments/test --seed 0 --out listening-test-v2

Writes ``{out}/trials.json`` and ``{out}/audio/*.mp3`` for ``listening-test-v2/index.html``.

**Task.** Each trial is a 3-alternative forced choice: three excerpts of one spliced inpainting
output (``{eval_root}/{condition}/inpaint/{stem}_spliced.wav``, the context window the model saw
with the regenerated mask region spliced back in); one excerpt is the regenerated region, the
other two are other human-labelled sections of the same track cut from the *same* file, so all
three share the recording chain, loudness and mastering. The rater picks the regenerated one.
After answering, the rater hears the whole spliced context and rates "sounds like Ravi" on a
5-point scale (1 = not at all, 5 = very much) with an optional comment.

**Excerpts** (:func:`plan_excerpts`, everything in context samples). All three excerpts have the
same length ``core + 2 * roll``: a ``roll`` = 0.5 s pre-roll, a core of ``L`` seconds, a 0.5 s
post-roll, and 20 ms raised-cosine fades at both ends. ``L = min(mask length, --max-core-s)``,
lowered in ``--step-s`` steps down to ``--min-core-s`` until two distractors fit; a region
where none fits is skipped (logged).

* Target: if ``L`` equals the mask length the core is the whole mask (both seams, at ``roll``
  and ``roll + L``). Otherwise it is cropped, anchored at the mask start (core ``[m0, m0+L)``,
  the start seam at ``roll``) or at the mask end (``[m1-L, m1)``), drawn per region from a
  seeded RNG.
* Distractors mirror that geometry: a mask is a human section padded by ``pad_s`` (1 s, from
  the sidecar), so the target core begins ``pad_s`` before a real section boundary. A
  distractor for section ``s`` is placed at its own "virtual mask" (core start
  ``s.start - pad_s``, or core end ``s.end + pad_s`` for an end-anchored trial), then shifted
  as little as needed to fit, rolls included, inside the audio that is neither the mask nor
  the target excerpt; at least half its core must lie in ``s`` (so it *is* that section). The
  two distractors are distinct sections (not the target's, not ``unlabeled``), do not overlap
  each other, and are the first feasible pair in a seeded order of the track's sections.
  Every distractor sample is therefore bit-identical to the original track (``grid`` splices
  with in-mask crossfades), and the only regenerated audio the rater hears is the target's.
* The plan depends only on the region and the seed, not the condition, so every condition
  plays the same three spans of a region (as long as the conditions share the geometry,
  which all SA3 conditions do).

**Loudness.** Nothing is normalised. One gain per output file, ``min(1, 10**(-1/20) / peak)``
over the whole spliced context (so at most -1 dBFS), is applied to all of that output's
exports, which keeps the three excerpts and the context at their relative levels.

**Encoding.** MP3 via ffmpeg ``libmp3lame``, CBR ``--bitrate`` (default 192 kbit/s), at the
file's sample rate (44.1 kHz for SA3), stereo, metadata stripped; ``--format wav`` writes
16-bit PCM WAVs instead (tests, debugging). A ~30 s excerpt is ~0.7 MB at 192k; a 40-trial
block with 47 s contexts is ~50 MB. LAME's encoder delay (~25 ms) falls in the pre-roll.

**Design.** ``--conditions`` (k of them, e.g. base, B6k, E6k); a region is eligible if every
condition has an output for it and :func:`plan_excerpts` succeeds. ``--blocks`` (default
``A,B``, one per rater) of ``--n-trials`` each (a multiple of k; default the largest <= 40):

* ``--n-shared`` trials (default about a third, a multiple of k) are the *same items*
  (region, condition, output file, excerpt files) in every block, drawn with the base seed,
  so inter-rater agreement (Cohen's kappa) is computable on them; each block shows them in its
  own order and with its own A/B/C letters.
* The rest of each block is drawn with that block's seed from regions not yet used by any
  block (falling back to reuse of other blocks' unique regions, with a warning, when the pool
  is too small).
* Within a block: equal trials per condition, every region at most once, trial order
  shuffled, and the answer letter balanced within each condition (each letter is the answer
  ``n_c/3`` times, +-1).
* With several grid seeds per region, the output file is drawn per (condition, region).

**Blinding.** Plain fields hold only an opaque trial id, an opaque item id, the audio file
names (``sha1`` of a seed-derived salt, the item and the role, so nothing orders or names
them), and durations (all three excerpts are equally long). The answer letter, condition,
region, stem and excerpt spans are in ``key`` (base64 of JSON, :func:`obfuscate`); the build
provenance is in the top-level ``meta`` field, also base64. This only stops casual reading:
anyone can decode it, which ``index.html?review=1`` does.

``trials.json`` is deterministic for a given input and arguments (no timestamps); its
``build_id`` (a hash of the design) keys the page's localStorage so a rebuilt test does not
resume stale answers. Score with :mod:`ravi.eval.score_listening_test`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TOOL = "ravi.eval.make_listening_test"
FORMAT = "ravi-listening-test-v2"
LETTERS = ("A", "B", "C")
UNLABELED = "unlabeled"  # ravi.structure.labels.UNLABELED
ROLL_S = 0.5
FADE_S = 0.020
MAX_CORE_S = 10.0
MIN_CORE_S = 4.0
STEP_S = 0.5
MIN_SECTION_OVERLAP = 0.5
PEAK_DBFS = -1.0
DEFAULT_BITRATE = "192k"
FORMATS = ("mp3", "wav")
RATING_SCALE = {
    "question": "How much does this sound like Ravi's music?",
    "min": 1,
    "max": 5,
    "labels": {"1": "Not at all", "2": "Slightly", "3": "Somewhat", "4": "Mostly", "5": "Very much"},
}


def log(msg: str) -> None:
    print(f"[make_listening_test] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- obfuscation


def obfuscate(obj: Any) -> str:
    """base64 of compact JSON (ASCII, so the page can ``atob`` it). Not encryption."""
    return base64.b64encode(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("ascii")).decode("ascii")


def deobfuscate(s: str) -> Any:
    return json.loads(base64.b64decode(s.encode("ascii")).decode("utf-8"))


def _h(*parts: Any, n: int = 12) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:n]


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(_h(*parts, n=16), 16))


# --------------------------------------------------------------------------- excerpt planning


@dataclass(frozen=True)
class Section:
    """A labelled section in context samples (may extend past the context)."""

    index: int
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class Excerpt:
    role: str  # "target" | "other"
    section_index: int
    label: str
    core_start: int
    core_end: int
    start: int  # core_start - roll
    end: int  # core_end + roll

    @property
    def n(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ExcerptPlan:
    core: int
    roll: int
    anchor: str  # "both" (whole mask) | "start" | "end"
    target: Excerpt
    others: tuple[Excerpt, Excerpt]

    @property
    def excerpts(self) -> tuple[Excerpt, Excerpt, Excerpt]:
        return (self.target, *self.others)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _subtract(interval: tuple[int, int], cut: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = interval
    c, d = cut
    out = []
    if c > a:
        out.append((a, min(b, c)))
    if d < b:
        out.append((max(a, d), b))
    return [(x, y) for x, y in out if y > x]


def _place(sec: Section, L: int, roll: int, pad: int, anchor: str, allowed: Sequence[tuple[int, int]],
           min_overlap: float) -> Excerpt | None:
    pref0 = sec.start - pad if anchor in ("both", "start") else sec.end + pad - L
    best: tuple[tuple[int, int], int] | None = None
    for a, b in allowed:
        lo, hi = a + roll, b - roll - L
        if hi < lo:
            continue
        c0 = min(max(pref0, lo), hi)
        ov = _overlap(c0, c0 + L, sec.start, sec.end)
        if ov < min_overlap * L:
            continue
        score = (-ov, abs(c0 - pref0))
        if best is None or score < best[0]:
            best = (score, c0)
    if best is None:
        return None
    c0 = best[1]
    return Excerpt("other", sec.index, sec.label, c0, c0 + L, c0 - roll, c0 + L + roll)


def plan_excerpts(
    n_ctx: int,
    sr: int,
    m0: int,
    m1: int,
    sections: Sequence[Section],
    target_index: int,
    *,
    rng: random.Random,
    pad_s: float = 1.0,
    roll_s: float = ROLL_S,
    max_core_s: float = MAX_CORE_S,
    min_core_s: float = MIN_CORE_S,
    step_s: float = STEP_S,
    min_overlap: float = MIN_SECTION_OVERLAP,
) -> ExcerptPlan | None:
    """Three equal-length excerpts of a context of ``n_ctx`` usable samples with the mask
    ``[m0, m1)`` (context samples): the target and two distractor sections (see the module
    docstring). ``None`` if no core of at least ``min_core_s`` fits."""
    roll = int(round(roll_s * sr))
    pad = int(round(pad_s * sr))
    mask_len = m1 - m0
    l_min = int(round(min_core_s * sr))
    step = max(1, int(round(step_s * sr)))
    # drawn once per region so they do not depend on which L succeeds
    crop_anchor = rng.choice(("start", "end"))
    cands = [s for s in sections if s.index != target_index and s.label != UNLABELED
             and _overlap(s.start, s.end, 0, n_ctx) > 0]
    rng.shuffle(cands)
    if mask_len < l_min:
        return None
    lengths = []
    L = min(mask_len, int(round(max_core_s * sr)))
    while L >= l_min:
        lengths.append(L)
        L -= step
    for L in lengths:
        anchor = "both" if L == mask_len else crop_anchor
        t0 = m1 - L if anchor == "end" else m0
        tgt = Excerpt("target", target_index, "", t0, t0 + L, t0 - roll, t0 + L + roll)
        if tgt.start < 0 or tgt.end > n_ctx:
            continue
        allowed = _subtract((0, n_ctx), (min(m0, tgt.start), max(m1, tgt.end)))
        placed = [e for e in (_place(s, L, roll, pad, anchor, allowed, min_overlap) for s in cands) if e is not None]
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                a, b = placed[i], placed[j]
                if a.section_index != b.section_index and _overlap(a.start, a.end, b.start, b.end) == 0:
                    return ExcerptPlan(L, roll, anchor, tgt, (a, b))
    return None


# --------------------------------------------------------------------------- eval root


@dataclass
class Output:
    condition: str
    stem: str
    track_id: str
    section_index: int
    label: str
    seed: int | None
    sidecar: Path
    spliced: Path
    meta: dict = field(repr=False)

    @property
    def key(self) -> tuple[str, int]:
        return (self.track_id, self.section_index)


def discover(eval_root: Path, condition: str) -> dict[tuple[str, int], list[Output]]:
    """``{(track_id, section_index): [outputs]}`` of ``{eval_root}/{condition}/inpaint``."""
    d = eval_root / condition / "inpaint"
    if not d.is_dir():
        raise SystemExit(f"{d}: no such directory (condition {condition!r})")
    out: dict[tuple[str, int], list[Output]] = {}
    for p in sorted(d.glob("*.json")):
        try:
            meta = json.loads(p.read_text())
        except json.JSONDecodeError:
            log(f"skip {p}: not JSON")
            continue
        if meta.get("command") not in (None, "inpaint") or "mask_in_context" not in meta or "context" not in meta:
            continue
        stem = p.stem
        sp = d / ((meta.get("files") or {}).get("spliced") or f"{stem}_spliced.wav")
        if not sp.is_file():
            log(f"skip {p}: no {sp.name}")
            continue
        o = Output(condition, stem, str(meta["track_id"]), int(meta["section"]["index"]), str(meta["section"]["label"]),
                   meta.get("seed"), p, sp, meta)
        out.setdefault(o.key, []).append(o)
    for v in out.values():
        v.sort(key=lambda o: o.stem)
    return out


def load_segments(track_id: str, meta: Mapping[str, Any], segments_dir: Path | None) -> dict[str, Any] | None:
    paths = []
    if segments_dir is not None:
        paths.append(segments_dir / f"{track_id}.json")
    if meta.get("segments_json"):
        paths.append(Path(str(meta["segments_json"])))
    for p in paths:
        if p.is_file():
            return json.loads(p.read_text())
    return None


def audio_info(path: Path) -> tuple[int, int, int]:
    """``(sample_rate, frames, channels)``."""
    import soundfile as sf

    i = sf.info(str(path))
    return int(i.samplerate), int(i.frames), int(i.channels)


def context_sections(doc: Mapping[str, Any], c0: int, sr: int) -> list[Section]:
    return [Section(i, str(s["label"]).strip().lower(), int(round(float(s["start"]) * sr)) - c0,
                    int(round(float(s["end"]) * sr)) - c0) for i, s in enumerate(doc["segments"])]


@dataclass
class Region:
    """One output's planning inputs, from its sidecar, segments and WAV header."""

    out: Output
    sr: int
    n_ctx: int
    m0: int
    m1: int
    c0: int
    sections: list[Section]
    pad_s: float


def region_of(o: Output, segments_dir: Path | None, require_human: bool = True) -> Region | str:
    """The planning inputs of ``o``, or the reason it cannot be used."""
    doc = load_segments(o.track_id, o.meta, segments_dir)
    if doc is None:
        return "no segments file"
    if require_human and doc.get("tool") not in (None, "human"):
        return f"segments tool {doc.get('tool')!r} is not human"
    sr, frames, _ = audio_info(o.spliced)
    ctx = o.meta["context"]
    mic = o.meta["mask_in_context"]
    have = int(ctx.get("source_samples") or frames)
    n_ctx = min(have, frames)
    pad = (o.meta.get("mask") or {}).get("pad_s")
    return Region(o, sr, n_ctx, int(mic["start_sample"]), int(mic["end_sample"]), int(ctx["start_sample"]),
                  context_sections(doc, int(ctx["start_sample"]), sr), 1.0 if pad is None else float(pad))


def plan_region(reg: Region, seed: int, **kw: Any) -> ExcerptPlan | None:
    rng = _rng("plan", seed, reg.out.track_id, reg.out.section_index)
    return plan_excerpts(reg.n_ctx, reg.sr, reg.m0, reg.m1, reg.sections, reg.out.section_index, rng=rng,
                         pad_s=reg.pad_s, **kw)


# --------------------------------------------------------------------------- design


def balanced(conditions: Sequence[str], n: int, rng: random.Random) -> list[str]:
    k = len(conditions)
    xs = [c for c in conditions for _ in range(n // k)]
    rng.shuffle(xs)
    return xs


def default_n_trials(k: int, target: int = 40) -> int:
    return k * (target // k)


def default_n_shared(k: int, n: int) -> int:
    return k * ((n // k) // 3)


def make_design(keys: Sequence[tuple[str, int]], conditions: Sequence[str], blocks: Sequence[str], n_trials: int,
                n_shared: int, seed: int) -> dict[str, list[tuple[tuple[str, int], str]]]:
    """``{block: [(region, condition), ...]}`` in presentation order (see the module docstring)."""
    k = len(conditions)
    if k < 1 or len(set(conditions)) != k:
        raise SystemExit(f"--conditions must be distinct and non-empty, got {list(conditions)}")
    if n_trials <= 0 or n_trials % k:
        raise SystemExit(f"--n-trials {n_trials} must be a positive multiple of {k} (one per condition)")
    if n_shared < 0 or n_shared % k or n_shared > n_trials:
        raise SystemExit(f"--n-shared {n_shared} must be a multiple of {k} in [0, {n_trials}]")
    if len(set(blocks)) != len(blocks) or not blocks:
        raise SystemExit(f"--blocks must be distinct and non-empty, got {list(blocks)}")
    keys = sorted(keys)
    if len(keys) < n_trials:
        raise SystemExit(f"only {len(keys)} eligible regions for {n_trials} trials per block "
                         "(each region at most once per block): lower --n-trials or relax the excerpt limits")
    r0 = _rng("shared", seed)
    shared_regions = r0.sample(keys, n_shared)
    shared = list(zip(shared_regions, balanced(conditions, n_shared, r0)))
    shared_set = set(shared_regions)
    used = set(shared_regions)
    out: dict[str, list[tuple[tuple[str, int], str]]] = {}
    n_u = n_trials - n_shared
    for b in blocks:
        rb = _rng("block", seed, b)
        pool = [x for x in keys if x not in used]
        if len(pool) < n_u:
            log(f"block {b}: {len(pool)} unused regions < {n_u}: reusing other blocks' unique regions")
            pool = [x for x in keys if x not in shared_set]
        regs = rb.sample(pool, n_u)
        items = shared + list(zip(regs, balanced(conditions, n_u, rb)))
        rb.shuffle(items)
        used |= set(regs)
        out[b] = items
    return out


def answer_letters(conds: Sequence[str], rng: random.Random) -> list[str]:
    """Per trial answer letter, each letter equally often (+-1) within each condition."""
    out: list[str | None] = [None] * len(conds)
    for c in sorted(set(conds)):
        idx = [i for i, x in enumerate(conds) if x == c]
        start = rng.randrange(3)
        letters = [LETTERS[(start + j) % 3] for j in range(len(idx))]
        rng.shuffle(letters)
        for i, l in zip(idx, letters):
            out[i] = l
    return [str(x) for x in out]


# --------------------------------------------------------------------------- audio export


def fade(x: np.ndarray, n: int) -> np.ndarray:
    from ravi.eval.masks import crossfade_ramp

    y = np.array(x, dtype=np.float32, copy=True)
    n = min(n, y.shape[0] // 2)
    if n > 0:
        r = crossfade_ramp(n)[:, None]
        y[:n] *= r
        y[-n:] *= r[::-1]
    return y


def peak_gain(x: np.ndarray, peak_dbfs: float = PEAK_DBFS) -> float:
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    lim = 10 ** (peak_dbfs / 20)
    return 1.0 if pk <= lim else lim / pk


def encode(audio: np.ndarray, sr: int, path: Path, fmt: str, bitrate: str = DEFAULT_BITRATE,
           ffmpeg: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    a = np.ascontiguousarray(audio, dtype=np.float32)
    if fmt == "wav":
        import soundfile as sf

        sf.write(str(tmp), np.clip(a, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    elif fmt == "mp3":
        exe = ffmpeg or shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("ffmpeg not found (needed for --format mp3; --format wav needs none)")
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(a.shape[1]),
               "-i", "pipe:0", "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", str(tmp)]
        subprocess.run(cmd, input=a.astype("<f4").tobytes(), check=True)
    else:
        raise SystemExit(f"--format must be one of {FORMATS}")
    tmp.replace(path)


# --------------------------------------------------------------------------- build


def _excerpt_doc(e: Excerpt, sr: int) -> dict[str, Any]:
    return {"role": e.role, "section_index": e.section_index, "label": e.label,
            "start_sample": e.start, "end_sample": e.end, "core_start_sample": e.core_start,
            "core_end_sample": e.core_end, "start_s": round(e.start / sr, 4), "end_s": round(e.end / sr, 4)}


def build(
    eval_root: str | Path,
    conditions: Sequence[str],
    out: str | Path,
    *,
    segments_dir: str | Path | None = None,
    n_trials: int | None = None,
    n_shared: int | None = None,
    blocks: Sequence[str] = ("A", "B"),
    seed: int = 0,
    fmt: str = "mp3",
    bitrate: str = DEFAULT_BITRATE,
    roll_s: float = ROLL_S,
    fade_s: float = FADE_S,
    max_core_s: float = MAX_CORE_S,
    min_core_s: float = MIN_CORE_S,
    step_s: float = STEP_S,
    require_human: bool = True,
    dry_run: bool = False,
    clean: bool = False,
) -> dict[str, Any]:
    """Build the test; returns the ``trials.json`` document (written unless ``dry_run``)."""
    import soundfile as sf

    eval_root, out = Path(eval_root), Path(out)
    seg_dir = Path(segments_dir) if segments_dir is not None else None
    if fmt not in FORMATS:
        raise SystemExit(f"--format must be one of {FORMATS}")
    k = len(conditions)
    n_trials = default_n_trials(k) if n_trials is None else int(n_trials)
    n_shared = default_n_shared(k, n_trials) if n_shared is None else int(n_shared)
    plan_kw = dict(roll_s=roll_s, max_core_s=max_core_s, min_core_s=min_core_s, step_s=step_s)

    found = {c: discover(eval_root, c) for c in conditions}
    common = sorted(set.intersection(*(set(v) for v in found.values()))) if found else []
    chosen: dict[tuple[str, str, int], tuple[Output, Region, ExcerptPlan]] = {}
    skipped: dict[str, list[str]] = {}
    eligible: list[tuple[str, int]] = []
    for key in common:
        ok = True
        for c in conditions:
            o = _rng("output", seed, c, *key).choice(found[c][key])
            reg = region_of(o, seg_dir, require_human)
            if isinstance(reg, str):
                skipped.setdefault(reg, []).append(o.stem)
                ok = False
                break
            plan = plan_region(reg, seed, **plan_kw)
            if plan is None:
                skipped.setdefault(f"no excerpt plan with core >= {min_core_s} s", []).append(o.stem)
                ok = False
                break
            chosen[(c, *key)] = (o, reg, plan)
        if ok:
            eligible.append(key)
    for c in conditions:
        log(f"{c}: {len(found[c])} regions with outputs")
    log(f"{len(common)} regions in every condition, {len(eligible)} eligible")
    for why, stems in skipped.items():
        log(f"skipped {len(stems)}: {why} (e.g. {stems[0]})")

    design = make_design(eligible, conditions, blocks, n_trials, n_shared, seed)
    salt = _h("lt2-salt", seed, n=16)
    ext = fmt

    def item_id(c: str, key: tuple[str, int], o: Output) -> str:
        return _h("item", salt, c, *key, o.stem, n=10)

    def fname(item: str, role: str) -> str:
        return f"audio/{_h('file', salt, item, role)}.{ext}"

    exports: dict[str, tuple[Output, Region, ExcerptPlan]] = {}
    doc_blocks: dict[str, Any] = {}
    shared_items: set[str] = set()
    counts: dict[str, Any] = {}
    for b, items in design.items():
        rb = _rng("letters", seed, b)
        letters = answer_letters([c for _, c in items], rb)
        trials = []
        for i, ((key, c), ans) in enumerate(zip(items, letters)):
            o, reg, plan = chosen[(c, *key)]
            iid = item_id(c, key, o)
            exports[iid] = (o, reg, plan)
            others = list(plan.others)
            rb.shuffle(others)
            rest = [l for l in LETTERS if l != ans]
            by_letter = {ans: ("target", plan.target)} | {l: (f"other{plan.others.index(e)}", e) for l, e in zip(rest, others)}
            files = {l: fname(iid, role) for l, (role, _) in sorted(by_letter.items())}
            exc_doc = {}
            for l, (role, e) in sorted(by_letter.items()):
                d = _excerpt_doc(e, reg.sr)
                if role == "target":
                    d["label"] = o.label
                exc_doc[l] = d | {"file": files[l]}
            key_doc = {
                "answer": ans, "condition": c, "track_id": o.track_id, "section_index": o.section_index,
                "label": o.label, "stem": o.stem, "seed": o.seed, "item": iid, "anchor": plan.anchor,
                "core_s": round(plan.core / reg.sr, 4), "roll_s": roll_s, "sample_rate": reg.sr,
                "mask_in_context": [reg.m0, reg.m1], "excerpts": exc_doc,
                "context_file": fname(iid, "context"), "gain": None,
            }
            trials.append({
                "id": f"{b}{i + 1:02d}", "index": i, "item": iid, "excerpts": files,
                "context": fname(iid, "context"),
                "excerpt_s": round((plan.core + 2 * plan.roll) / reg.sr, 3),
                "context_s": round(reg.n_ctx / reg.sr, 3),
                "_key": key_doc,
            })
        doc_blocks[b] = {"trials": trials}
        counts[b] = {c: sum(1 for _, x in items if x == c) for c in conditions}
    ids_per_block = [{t["item"] for t in blk["trials"]} for blk in doc_blocks.values()]
    if len(ids_per_block) > 1:
        shared_items = set.intersection(*ids_per_block)

    # gains (need the audio) and export
    gains: dict[str, float] = {}
    if not dry_run:
        adir = out / "audio"
        if clean and adir.is_dir():
            for p in adir.iterdir():
                if p.suffix in (".mp3", ".wav", ".tmp"):
                    p.unlink()
        for n_done, (iid, (o, reg, plan)) in enumerate(sorted(exports.items()), 1):
            audio, sr = sf.read(str(o.spliced), dtype="float32", always_2d=True)
            if sr != reg.sr:
                raise RuntimeError(f"{o.spliced}: sample rate changed")
            audio = audio[: reg.n_ctx]
            g = peak_gain(audio)
            gains[iid] = g
            nf = int(round(fade_s * sr))
            for role, e in [("target", plan.target), ("other0", plan.others[0]), ("other1", plan.others[1])]:
                encode(fade(audio[e.start:e.end] * g, nf), sr, out / fname(iid, role), fmt, bitrate)
            encode(fade(audio * g, nf), sr, out / fname(iid, "context"), fmt, bitrate)
            if n_done % 10 == 0 or n_done == len(exports):
                log(f"exported {n_done}/{len(exports)} items")
    for blk in doc_blocks.values():
        for t in blk["trials"]:
            kd = t.pop("_key")
            kd["gain"] = gains.get(t["item"])
            t["key"] = obfuscate(kd)

    meta = {
        "tool": TOOL, "eval_root": str(eval_root), "conditions": list(conditions), "blocks": list(blocks),
        "segments_dir": None if seg_dir is None else str(seg_dir), "seed": seed, "n_trials": n_trials,
        "n_shared": n_shared, "shared_items": sorted(shared_items), "counts": counts,
        "n_regions_common": len(common), "n_regions_eligible": len(eligible),
        "skipped": {w: len(s) for w, s in skipped.items()},
        "excerpts": {"roll_s": roll_s, "fade_s": fade_s, "max_core_s": max_core_s, "min_core_s": min_core_s,
                     "step_s": step_s, "min_section_overlap": MIN_SECTION_OVERLAP, "peak_dbfs": PEAK_DBFS},
        "argv": sys.argv[1:],
    }
    design_digest = json.dumps({b: [[list(r), c] for r, c in v] for b, v in design.items()}, sort_keys=True)
    doc = {
        "format": FORMAT,
        "version": 1,
        "build_id": _h("build", salt, design_digest, fmt, bitrate, roll_s, max_core_s, min_core_s, n=12),
        "letters": list(LETTERS),
        "require_all_played": True,
        "rating": RATING_SCALE,
        "audio": {"format": fmt, "bitrate": bitrate if fmt == "mp3" else None, "dir": "audio"},
        "blocks": doc_blocks,
        "meta": obfuscate(meta),
    }
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        p = out / "trials.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(p)
        log(f"wrote {p} ({sum(len(b['trials']) for b in doc_blocks.values())} trials, {len(exports)} items, "
            f"{len(shared_items)} shared)")
    for b, cs in counts.items():
        log(f"block {b}: " + ", ".join(f"{c}={n}" for c, n in cs.items()))
    return doc


def decode_keys(doc: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(block, trial_id): key}`` of a ``trials.json`` document."""
    return {(b, t["id"]): deobfuscate(t["key"]) for b, blk in doc["blocks"].items() for t in blk["trials"]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ravi.eval.make_listening_test",
                                 description="Build the 3AFC seam-localisation listening test (plan Phase 4.3).")
    ap.add_argument("--eval-root", required=True, help="ravi.eval.grid out_root ({cond}/inpaint/*_spliced.wav + .json)")
    ap.add_argument("--conditions", required=True, help="comma-separated condition directories, e.g. base,B6k,E6k")
    ap.add_argument("--segments-dir", default="eval/segments/test",
                    help="human segments ({track_id}.json); falls back to the sidecar's segments_json")
    ap.add_argument("--out", default="listening-test-v2", help="page directory (trials.json + audio/)")
    ap.add_argument("--n-trials", type=int, default=None, help="trials per block (multiple of #conditions; default <= 40)")
    ap.add_argument("--n-shared", type=int, default=None,
                    help="trials shared by all blocks for inter-rater agreement (multiple of #conditions; default ~1/3)")
    ap.add_argument("--blocks", default="A,B", help="rater blocks (comma-separated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=FORMATS, default="mp3")
    ap.add_argument("--bitrate", default=DEFAULT_BITRATE, help="MP3 CBR bitrate for ffmpeg libmp3lame")
    ap.add_argument("--roll-s", type=float, default=ROLL_S)
    ap.add_argument("--fade-s", type=float, default=FADE_S)
    ap.add_argument("--max-core-s", type=float, default=MAX_CORE_S)
    ap.add_argument("--min-core-s", type=float, default=MIN_CORE_S)
    ap.add_argument("--step-s", type=float, default=STEP_S)
    ap.add_argument("--allow-nonhuman-segments", action="store_true",
                    help="accept segments files whose tool is not 'human'")
    ap.add_argument("--clean", action="store_true", help="delete existing audio files in {out}/audio first")
    ap.add_argument("--dry-run", action="store_true", help="plan and print counts; write nothing")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    seg = a.segments_dir if a.segments_dir and Path(a.segments_dir).is_dir() else None
    if a.segments_dir and seg is None:
        log(f"--segments-dir {a.segments_dir}: not a directory, using the sidecars' segments_json")
    build(a.eval_root, [c.strip() for c in a.conditions.split(",") if c.strip()], a.out, segments_dir=seg,
          n_trials=a.n_trials, n_shared=a.n_shared, blocks=[b.strip() for b in a.blocks.split(",") if b.strip()],
          seed=a.seed, fmt=a.format, bitrate=a.bitrate, roll_s=a.roll_s, fade_s=a.fade_s, max_core_s=a.max_core_s,
          min_core_s=a.min_core_s, step_s=a.step_s, require_human=not a.allow_nonhuman_segments,
          dry_run=a.dry_run, clean=a.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
