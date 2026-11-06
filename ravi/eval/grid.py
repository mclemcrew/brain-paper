"""Evaluation grid driver (plan Phase 2.3 Tier 0 go/no-go, Phase 3 tiers, Phase 4): run every
condition of a YAML over the evaluation regions and the text-to-music prompts with the
:mod:`ravi.eval.sa3_infer` machinery, in the layout :mod:`ravi.eval.run_eval` reads.

    python -m ravi.eval.grid configs/eval/tier0.yaml [--dry-run] [--conditions base,B3k_s1]

Config (YAML, or JSON; relative paths are relative to the working directory, except the
regions file's ``segments_json`` entries, which are relative to the regions file)::

    regions: eval/regions_test.json        # ravi.eval.select_regions output
    out_root: eval/tier0
    audio_root: null                       # re-roots the regions' audio_path (dir of {track_id}.wav)
    seeds: [1]
    tasks: [inpaint, generate]
    inpaint: {context_s: 47, min_context_s: 5, xfade_ms: 50, blank_mask_input: false}
    generate: {duration: 30, prompts_from: arm, n_prompts: 20, seed: 0, trigger: rvkr,
               sections_window_s: 47}
    anchors: {lower: true, lower_name: anchor_lower, seed: 0}
    defaults: {model: medium, prompt_source: arm_caption, arm: B, caption_root: null}
    conditions:
      - {name: base}
      - {name: B3k_s1, lora: /runs/B_tier0/model/adapter.safetensors, lora_strength: 1.0}

Condition keys (``defaults`` fills the missing ones): ``name``; ``model`` (``medium`` |
``medium-base``); ``lora`` (adapter path or null); ``lora_strength`` (default 1.0 with a
LoRA); ``steps`` / ``cfg`` (default per model, :data:`ravi.eval.sa3_infer.MODEL_DEFAULTS`);
``prompt_source``: ``arm_caption`` (the arm's caption, see below), ``human`` (the bare human
caption, no trigger) or ``none`` (empty prompt); ``arm`` (A/B/C/D/E/F); ``caption_root``
(make_arms layout ``{caption_root}/{arm}/{split}/{track_id}.txt``); ``with_sections``
(true: the prompt ends with ``sections: …`` from the human labels inside the window the model
sees; false: any ``sections:`` field is removed; null: true for arm E, false otherwise);
``sections_from`` (``context``, default: always recompute the field from the window, so an
arm-E caption describes the inference window, not the training crop; ``caption``: keep the
caption file's own field when it has one); ``autocaptions`` (arm D fallback JSONL);
``tasks`` (subset of the global tasks).

Arm captions, first found: the file under ``caption_root``; the caption ``select_regions``
stored in the regions file (``tracks[id].captions[arm]``, built with
:func:`ravi.captions.make_arms.build_caption`, the text make_arms writes; arm E without its
section field); built in-process from the manifest fields stored there (needs pandas). The
section field is then handled per ``with_sections``. The origin is recorded per output
(``grid.prompt_origin``).

Outputs, per condition ``C`` and seed ``s``::

    {out_root}/C/inpaint/{track}_sec{i}_{label}_{start:.1f}-{end:.1f}_seed{s}_{raw,spliced,orig}.wav + .json
    {out_root}/C/gen/{track}_seed{s}.wav + .json
    {out_root}/{lower_name}/inpaint/<same inpaint stems>                           (lower anchor)
    {out_root}/manifest.parquet       one row per output (manifest.csv if no parquet engine)
    {out_root}/grid_log.jsonl         one line per condition run: time, model load, GPU memory
    {out_root}/grid_config.json       the resolved config of the last run

The inpainting stem is ``sa3_infer``'s plus ``_seed{s}``, the same in every condition, so
``run_eval`` pairs regions across conditions by stem; the sidecars are ``sa3_infer``'s
(``section``, ``mask``, ``context``, ``mask_in_context``, …, ``command: "inpaint"``) plus a
``grid`` block. ``_orig.wav`` is the upper anchor. Generation stems are ``{track}_seed{s}``
(paired by stem across conditions); the model seed is derived from ``(s, track_id)``
(:func:`gen_seed`) so arms whose prompts coincide (arm A: ``rvkr`` for every track) still
give distinct clips.

**Lower anchor** (``anchors.lower``): for every region, the same context with the region
replaced by audio from a *different* test track (never the same ``alt_family``): the donor is
the candidate region (any eligible region of the split) closest in length, ties broken by a
seeded hash (:func:`pair_lower_anchors`, independent of input order); the fill is
``len(region)`` samples centred on the donor region (clamped to the donor track, tiled if the
donor track is shorter), spliced with the 50 ms in-mask crossfades. Written as a regular
condition named ``anchor_lower`` because ``run_eval`` reserves ``lower`` (and ``upper``) for
its own anchors and refuses a directory with that name. No model is needed; the context
uses the ``geometry`` block (SA3 medium: 44.1 kHz, 4096-sample latent frames), checked
against every loaded model.

Resumable: an output whose sidecar JSON exists is skipped (``sa3_infer`` writes the JSON
last). A condition with nothing left to do does not load a model; consecutive conditions with
the same ``(model, lora)`` share one loaded model (only the LoRA strength changes).
``--dry-run`` prints the plan, the counts (to do / done), missing inputs, one prompt per
condition and a time estimate, and loads nothing.

**Engines** (top-level ``engine:``): ``sa3`` (default, :mod:`ravi.eval.sa3_infer`, models
``medium`` / ``medium-base``, ``lora`` = adapter ``.safetensors``) or ``acestep``
(:mod:`ravi.eval.acestep_grid`, run inside ``ravi-acestep-train``; models ``turbo`` / ``base``,
``lora`` = PEFT adapter directory). The ACE-Step engine presents SA3's 44.1 kHz geometry, so
both engines produce sample-identical masks, contexts and stems for the same regions file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ravi.eval import sa3_infer as si
from ravi.eval.masks import lower_anchor, seconds_to_samples, splice
from ravi.structure import schema
from ravi.structure.labels import UNLABELED, normalize_label
from ravi.train.caption_aug import FIELD_SEP, split_fields

TOOL = "ravi.eval.grid"
TASKS = ("inpaint", "generate")
PROMPT_SOURCES = ("arm_caption", "human", "none")
ARMS = ("A", "B", "C", "D", "E", "F")
RESERVED = ("upper", "lower", "test_orig")  # run_eval.RESERVED
SECTIONS_PREFIX = "sections:"  # make_arms.SECTIONS_PREFIX (not imported: make_arms needs pandas)
TRIGGER = "rvkr"
#: SA3 medium / medium-base: sa3_infer.DEFAULT_SAMPLE_SIZE, 4096-sample latent frames
#: (ravi.captions.windows.DOWNSAMPLE, docs/workflows/autoencoder.md:19).
NOMINAL_GEOMETRY = {"sample_rate": 44100, "downsampling_ratio": 4096, "sample_size": si.DEFAULT_SAMPLE_SIZE,
                    "channels": 2}
MAX_CONSECUTIVE_FAILURES = 5

EngineFactory = Callable[[str, "str | None"], si.Engine]


def log(msg: str) -> None:
    print(f"[grid] {msg}", flush=True)


# --------------------------------------------------------------------------- config


@dataclass
class Condition:
    name: str
    model: str = "medium"
    lora: str | None = None
    lora_strength: float | None = None
    steps: int = 8
    cfg: float = 1.0
    prompt_source: str = "arm_caption"
    arm: str | None = "B"
    caption_root: str | None = None
    with_sections: bool = False
    sections_from: str = "context"
    autocaptions: str | None = None
    tasks: tuple[str, ...] = TASKS

    @property
    def engine_key(self) -> tuple[str, str | None]:
        return (self.model, self.lora)


@dataclass
class GridConfig:
    regions: str
    out_root: str
    conditions: list[Condition]
    seeds: list[int] = field(default_factory=lambda: [1])
    tasks: tuple[str, ...] = TASKS
    audio_root: str | None = None
    inpaint: dict[str, Any] = field(default_factory=dict)
    generate: dict[str, Any] = field(default_factory=dict)
    anchors: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    half: bool = True
    allow_cpu: bool = False
    source: str | None = None
    engine: str = "sa3"

    @property
    def lower_name(self) -> str | None:
        return self.anchors["lower_name"] if self.anchors.get("lower") else None


INPAINT_DEFAULTS = {"context_s": si.DEFAULT_CONTEXT_S, "min_context_s": si.DEFAULT_MIN_CONTEXT_S,
                    "xfade_ms": si.DEFAULT_XFADE_S * 1000.0, "blank_mask_input": False}
GENERATE_DEFAULTS = {"duration": 30.0, "prompts_from": "arm", "n_prompts": None, "seed": 0,
                     "trigger": TRIGGER, "sections_window_s": si.DEFAULT_CONTEXT_S}
ANCHOR_DEFAULTS = {"lower": True, "lower_name": "anchor_lower", "seed": 0}
SECTIONS_DEFAULTS = {"min_overlap_s": 1.0}
CONDITION_KEYS = {f for f in Condition.__dataclass_fields__}


def load_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml  # in the ravi-sa3-train image (pyyaml via the SA3 lock)

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"{p}: expected a mapping at the top level")
    return data


def _tasks(x: Any, where: str) -> tuple[str, ...]:
    ts = (x,) if isinstance(x, str) else tuple(x)
    bad = [t for t in ts if t not in TASKS]
    if bad or not ts:
        raise SystemExit(f"{where}: tasks must be a non-empty subset of {TASKS}, got {list(ts)}")
    return tuple(t for t in TASKS if t in ts)


ENGINES = ("sa3", "acestep")


def engine_models(engine: str) -> dict[str, dict[str, Any]]:
    """``{model: {"steps", "cfg", ...}}`` of an engine; the first key is its default model."""
    if engine == "acestep":
        from ravi.eval import acestep_grid

        return acestep_grid.MODELS
    return {m: si.MODEL_DEFAULTS[m] for m in si.MODELS}


def lora_present(engine: str, path: str) -> bool:
    if engine == "acestep":
        from ravi.eval import acestep_grid

        return acestep_grid.is_adapter_dir(path)
    return Path(path).is_file()


def lora_info(engine: str, path: str | None) -> dict | None:
    if path is None:
        return None
    if engine == "acestep":
        from ravi.eval import acestep_grid

        return acestep_grid.adapter_info(path)
    return si.adapter_info(path)


def make_condition(raw: Mapping[str, Any], defaults: Mapping[str, Any], global_tasks: tuple[str, ...],
                   engine: str = "sa3") -> Condition:
    d = {**dict(defaults), **dict(raw)}
    models = engine_models(engine)
    unknown = set(d) - CONDITION_KEYS
    if unknown:
        raise SystemExit(f"condition {d.get('name')!r}: unknown keys {sorted(unknown)}")
    name = str(d.get("name") or "").strip()
    if not name or "/" in name or name.startswith((".", "_")) or name in RESERVED or "@" in name:
        raise SystemExit(f"bad condition name {name!r} (empty, '/', '@', leading '.'/'_', or reserved {RESERVED})")
    model = d.get("model") or next(iter(models))
    if model not in models:
        raise SystemExit(f"condition {name}: {engine} model must be one of {tuple(models)}")
    lora = d.get("lora") or None
    strength = d.get("lora_strength")
    if lora is None and strength is not None:
        raise SystemExit(f"condition {name}: lora_strength needs a lora")
    if lora is not None:
        strength = 1.0 if strength is None else float(strength)
    md = models[model]
    steps = int(md["steps"] if d.get("steps") is None else d["steps"])
    cfg = float(md["cfg"] if d.get("cfg") is None else d["cfg"])
    if steps < 1:
        raise SystemExit(f"condition {name}: steps must be >= 1")
    ps = d.get("prompt_source", "arm_caption")
    if ps not in PROMPT_SOURCES:
        raise SystemExit(f"condition {name}: prompt_source must be one of {PROMPT_SOURCES}")
    arm = d.get("arm")
    arm = None if arm is None else str(arm).upper()
    if ps == "arm_caption" and arm not in ARMS:
        raise SystemExit(f"condition {name}: prompt_source arm_caption needs arm in {ARMS}")
    ws = d.get("with_sections")
    ws = (ps == "arm_caption" and arm == "E") if ws is None else bool(ws)
    sf_ = d.get("sections_from", "context")
    if sf_ not in ("context", "caption"):
        raise SystemExit(f"condition {name}: sections_from must be context or caption")
    tasks = _tasks(d["tasks"], f"condition {name}") if d.get("tasks") is not None else global_tasks
    return Condition(name=name, model=model, lora=None if lora is None else str(lora), lora_strength=strength,
                     steps=steps, cfg=cfg, prompt_source=ps, arm=arm, caption_root=d.get("caption_root") or None,
                     with_sections=ws, sections_from=sf_, autocaptions=d.get("autocaptions") or None,
                     tasks=tuple(t for t in tasks if t in global_tasks))


def resolve_config(raw: Mapping[str, Any], overrides: Mapping[str, Any] | None = None,
                   source: str | None = None) -> GridConfig:
    raw = dict(raw)
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    for k in ("regions", "out_root", "audio_root"):
        if k in ov:
            raw[k] = ov[k]
    for k in ("regions", "out_root"):
        if not raw.get(k):
            raise SystemExit(f"config: {k!r} is required")
    gtasks = _tasks(ov.get("tasks", raw.get("tasks", TASKS)), "config")
    engine = str(raw.get("engine") or "sa3")
    if engine not in ENGINES:
        raise SystemExit(f"config: engine must be one of {ENGINES}")
    defaults = dict(raw.get("defaults") or {})
    if "caption_root" in ov:
        defaults["caption_root"] = ov["caption_root"]
    conds_raw = raw.get("conditions") or []
    if not conds_raw:
        raise SystemExit("config: no conditions")
    conds = []
    for c in conds_raw:
        c = dict(c)
        if "caption_root" in ov:
            c["caption_root"] = ov["caption_root"]
        conds.append(make_condition(c, defaults, gtasks, engine))
    names = [c.name for c in conds]
    if len(set(names)) != len(names):
        raise SystemExit(f"config: duplicate condition names {names}")
    seeds = raw.get("seeds", [1])
    seeds = [int(s) for s in ([seeds] if isinstance(seeds, int) else seeds)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("config: seeds must be a non-empty list of distinct ints")
    anchors = {**ANCHOR_DEFAULTS, **dict(raw.get("anchors") or {})}
    if ov.get("no_anchors"):
        anchors["lower"] = False
    if anchors.get("lower"):
        ln = str(anchors["lower_name"])
        if ln in RESERVED or ln in names or "/" in ln:
            raise SystemExit(f"anchors.lower_name {ln!r} is reserved or clashes with a condition")
    if "conditions_filter" in ov:
        want = [x for x in ov["conditions_filter"].split(",") if x]
        missing = [w for w in want if w not in names and w != anchors.get("lower_name")]
        if missing:
            raise SystemExit(f"--conditions: unknown {missing}; have {names}")
        conds = [c for c in conds if c.name in want]
        if anchors.get("lower") and anchors["lower_name"] not in want:
            anchors["lower"] = False
    gen = {**GENERATE_DEFAULTS, **dict(raw.get("generate") or {})}
    geom = {**NOMINAL_GEOMETRY, **dict(raw.get("geometry") or {})}
    return GridConfig(
        regions=str(raw["regions"]), out_root=str(raw["out_root"]), conditions=conds, seeds=seeds, tasks=gtasks,
        audio_root=raw.get("audio_root") or None,
        inpaint={**INPAINT_DEFAULTS, **dict(raw.get("inpaint") or {})}, generate=gen, anchors=anchors,
        geometry=geom, sections={**SECTIONS_DEFAULTS, **dict(raw.get("sections") or {})},
        device=ov.get("device", raw.get("device")), half=not ov.get("no_half", not raw.get("half", True)),
        allow_cpu=bool(ov.get("allow_cpu", raw.get("allow_cpu", False))), source=source, engine=engine,
    )


# --------------------------------------------------------------------------- regions / tracks


class RegionSet:
    """The ``select_regions`` file: regions, donor candidates, per-track metadata."""

    def __init__(self, path: str | Path, audio_root: str | None = None):
        self.path = Path(path)
        self.base = self.path.parent
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.doc = doc
        self.split: str = doc.get("split", "test")
        self.regions: list[dict[str, Any]] = sorted(doc["regions"], key=region_key)
        self.candidates: list[dict[str, Any]] = sorted(doc.get("candidates") or doc["regions"], key=region_key)
        self.tracks: dict[str, dict[str, Any]] = doc.get("tracks") or {}
        self.audio_root = audio_root
        self.pad_s = doc.get("pad_s")
        self.min_len_s = doc.get("min_len_s")
        self._segs: dict[str, dict[str, Any] | None] = {}

    def track(self, tid: str) -> dict[str, Any]:
        t = dict(self.tracks.get(tid) or {})
        for r in self.candidates:
            if r["track_id"] == tid:
                t.setdefault("audio_path", r.get("audio_path"))
                t.setdefault("segments_json", r.get("segments_json"))
                t.setdefault("duration", r.get("track_duration"))
                t.setdefault("alt_family", r.get("alt_family"))
                break
        t["track_id"] = tid
        return t

    def audio_path(self, tid: str) -> Path:
        ap = self.track(tid).get("audio_path")
        if not ap:
            raise SystemExit(f"{self.path}: no audio_path for {tid!r}")
        return Path(self.audio_root) / Path(ap).name if self.audio_root else Path(ap)

    def segments_path(self, tid: str) -> Path | None:
        sp = self.track(tid).get("segments_json")
        if not sp:
            return None
        p = Path(sp)
        return p if p.is_absolute() else self.base / p

    def segments_doc(self, tid: str) -> dict[str, Any] | None:
        if tid not in self._segs:
            p = self.segments_path(tid)
            self._segs[tid] = schema.load(p) if p is not None and p.is_file() else None
        return self._segs[tid]


def region_key(r: Mapping[str, Any]) -> tuple[str, int]:
    return (str(r["track_id"]), int(r["section_index"]))


def inpaint_stem(r: Mapping[str, Any], seed: int) -> str:
    return f"{si.inpaint_stem(r['track_id'], int(r['section_index']), r['label'], float(r['start']), float(r['end']))}_seed{seed}"


def gen_stem(track_id: str, seed: int) -> str:
    return f"{si.safe_track(track_id)}_seed{seed}"


def _h(*parts: Any) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16)


def gen_seed(seed: int, track_id: str) -> int:
    """Model seed for the generation of ``track_id``'s prompt under grid seed ``seed``."""
    return _h("gen", int(seed), track_id) % (2**31 - 1)


# --------------------------------------------------------------------------- lower anchor


def _family(r: Mapping[str, Any]) -> Any:
    return r.get("alt_family")


def pair_lower_anchors(regions: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
                       seed: int = 0) -> dict[tuple[str, int], dict[str, Any]]:
    """Donor region for every region: another track (and another ``alt_family`` when both
    are known), the closest length, ties broken by a hash of ``(seed, region, donor)``.
    The result does not depend on the order of either list."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in regions:
        k = region_key(r)
        fam = _family(r)
        pool = [c for c in candidates if c["track_id"] != r["track_id"]
                and (fam is None or _family(c) is None or _family(c) != fam)]
        if not pool:
            raise SystemExit(f"lower anchor: no donor region for {k} (no other track/family)")
        dur = float(r["end"]) - float(r["start"])
        best = min(pool, key=lambda c: (round(abs((float(c["end"]) - float(c["start"])) - dur), 6),
                                        _h("lower", seed, *k, *region_key(c))))
        out[k] = dict(best)
    return out


def donor_fill(donor_audio: np.ndarray, sr: int, donor: Mapping[str, Any], length: int) -> tuple[np.ndarray, int | None]:
    """``length`` samples of the donor track centred on the donor region (clamped to the
    track; tiled from the start when the track is shorter). Returns the fill and its offset."""
    n = donor_audio.shape[0]
    if n < length:
        return lower_anchor(donor_audio, length), None
    centre = seconds_to_samples((float(donor["start"]) + float(donor["end"])) / 2.0, sr)
    off = int(min(max(centre - length // 2, 0), n - length))
    return lower_anchor(donor_audio, length, offset=off), off


# --------------------------------------------------------------------------- prompts


class Prompter:
    """Builds the prompt of one condition for one track and window."""

    def __init__(self, cfg: GridConfig, rs: RegionSet):
        self.cfg = cfg
        self.rs = rs
        self._autocaps: dict[str, dict[str, str]] = {}
        self._jsonl: dict[str, str] | None = None

    def _autocap(self, path: str | None, tid: str) -> str | None:
        if not path:
            return None
        if path not in self._autocaps:
            self._autocaps[path] = read_caption_jsonl(path)
        return self._autocaps[path].get(tid)

    def caption_fields(self, cond: Condition, tid: str) -> tuple[list[str], str]:
        if cond.prompt_source == "none":
            return [], "none"
        row = self.rs.track(tid)
        if cond.prompt_source == "human":
            text = " ".join(str(row.get("human_caption") or "").replace(FIELD_SEP, " ").split()).rstrip(". ")
            if not text:
                raise SystemExit(f"{cond.name}: no human caption for {tid!r} in {self.rs.path}")
            return [text], "human_caption"
        if cond.caption_root:
            p = Path(cond.caption_root) / str(cond.arm) / self.rs.split / f"{tid}.txt"
            if p.is_file():
                return split_fields(p.read_text(encoding="utf-8").strip()), f"file:{p}"
        pre = (row.get("captions") or {}).get(str(cond.arm))
        if pre is not None and not (cond.arm == "D" and cond.autocaptions):
            return split_fields(pre), f"regions_file:{cond.arm}"
        if "human_caption" not in row:
            raise SystemExit(f"{cond.name}: no caption file for {tid!r} and no track metadata in {self.rs.path} "
                             "to build one (re-run select_regions)")
        from ravi.captions.make_arms import build_caption  # needs pandas

        cap = build_caption(str(cond.arm), row, None, autocaption=self._autocap(cond.autocaptions, tid))
        return list(cap.fields), f"built:make_arms.build_caption({cond.arm})"

    def section_labels(self, tid: str, w0: float, w1: float) -> list[str] | None:
        doc = self.rs.segments_doc(tid)
        if doc is None:
            return None
        min_ov = float(self.cfg.sections["min_overlap_s"])
        labs = []
        for s in doc["segments"]:
            lab = normalize_label(s["label"])
            if lab == UNLABELED:
                continue
            a, b = float(s["start"]), float(s["end"])
            if min(b, w1) - max(a, w0) >= min(min_ov, 0.5 * (b - a)):
                labs.append(lab)
        return labs

    def build(self, cond: Condition, tid: str, window: tuple[float, float],
              fields: list[str] | None = None, origin: str | None = None) -> dict[str, Any]:
        if fields is None:
            fields, origin = self.caption_fields(cond, tid)
        kept = [f for f in fields if not f.lower().startswith(SECTIONS_PREFIX)]
        had = len(kept) != len(fields)
        info: dict[str, Any] = {"prompt_origin": origin, "with_sections": cond.with_sections,
                                "sections_window_s": [round(window[0], 3), round(window[1], 3)]}
        if not cond.with_sections:
            out = kept
            info["sections"] = None
        elif cond.sections_from == "caption" and had:
            out = list(fields)
            info["sections"] = "caption"
        else:
            labs = self.section_labels(tid, *window)
            info["sections"] = labs
            out = kept + ([f"{SECTIONS_PREFIX} {', '.join(labs)}"] if labs else [])
            if not labs:
                info["sections_missing"] = True
        prompt = FIELD_SEP.join(out)
        info["prompt"] = prompt
        info["prompt_rendered"] = si.render_prompt(prompt) if prompt else ""
        return info

    def gen_base_fields(self, cond: Condition, tid: str) -> tuple[list[str], str]:
        src = self.cfg.generate["prompts_from"]
        if src in (None, "arm"):
            return self.caption_fields(cond, tid)
        if self._jsonl is None:
            self._jsonl = read_caption_jsonl(src)
        cap = self._jsonl.get(tid)
        if not cap:
            raise SystemExit(f"generate.prompts_from {src}: no caption for {tid!r}")
        trig = self.cfg.generate.get("trigger")
        return ([trig] if trig else []) + [" ".join(cap.replace(FIELD_SEP, " ").split())], f"jsonl:{src}"


def read_caption_jsonl(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("caption"):
                out[str(rec["track_id"])] = str(rec["caption"])
    return out


# --------------------------------------------------------------------------- plan


@dataclass
class Job:
    condition: str
    task: str  # inpaint | generate
    track_id: str
    seed: int
    stem: str
    out_dir: Path
    region: dict[str, Any] | None = None
    anchor: str | None = None  # "lower" for the lower-anchor condition

    @property
    def sidecar(self) -> Path:
        return self.out_dir / f"{self.stem}.json"

    @property
    def main_path(self) -> Path:
        return self.out_dir / (f"{self.stem}_spliced.wav" if self.task == "inpaint" else f"{self.stem}.wav")

    @property
    def done(self) -> bool:
        return self.sidecar.is_file() and self.main_path.is_file()


def generation_tracks(cfg: GridConfig, rs: RegionSet, limit: int | None = None) -> list[str]:
    """Tracks whose prompts are generated: the split's labelled tracks (``tracks`` entries
    with a ``segments_json``, so ``with_sections`` conditions get a section field for every
    prompt and every condition uses the same tracks), intersected with the JSONL when
    ``prompts_from`` is one; then a ``generate.seed``-ed subset of ``n_prompts``, sorted."""
    src = cfg.generate["prompts_from"]
    labelled = sorted(t for t, rec in rs.tracks.items() if rec.get("segments_json")) \
        or sorted({r["track_id"] for r in rs.candidates})
    if src in (None, "arm"):
        ids = labelled
    else:
        ids = sorted(read_caption_jsonl(src))
        if rs.tracks:
            ids = [t for t in ids if t in set(labelled)]
    n = cfg.generate.get("n_prompts")
    if n is not None and int(n) < len(ids):
        rng = np.random.default_rng(int(cfg.generate.get("seed", 0)))
        ids = sorted(ids[i] for i in rng.choice(len(ids), size=int(n), replace=False))
    return ids[:limit] if limit is not None else ids


def plan(cfg: GridConfig, rs: RegionSet, limit_regions: int | None = None,
         limit_prompts: int | None = None) -> dict[str, list[Job]]:
    """Jobs per condition (the lower anchor first), in execution order."""
    root = Path(cfg.out_root)
    regions = rs.regions[:limit_regions] if limit_regions is not None else rs.regions
    gen_ids = generation_tracks(cfg, rs, limit_prompts) if "generate" in cfg.tasks else []
    jobs: dict[str, list[Job]] = {}
    if cfg.lower_name and "inpaint" in cfg.tasks:
        d = root / cfg.lower_name / "inpaint"
        jobs[cfg.lower_name] = [Job(cfg.lower_name, "inpaint", r["track_id"], s, inpaint_stem(r, s), d, dict(r), "lower")
                                for s in cfg.seeds for r in regions]
    for c in cfg.conditions:
        js: list[Job] = []
        if "inpaint" in c.tasks:
            d = root / c.name / "inpaint"
            js += [Job(c.name, "inpaint", r["track_id"], s, inpaint_stem(r, s), d, dict(r)) for s in cfg.seeds for r in regions]
        if "generate" in c.tasks:
            d = root / c.name / "gen"
            js += [Job(c.name, "generate", t, s, gen_stem(t, s), d) for s in cfg.seeds for t in gen_ids]
        jobs[c.name] = js
    return jobs


# --------------------------------------------------------------------------- geometry


@dataclass
class Geometry:
    sample_rate: int
    downsampling_ratio: int
    sample_size: int
    channels: int = 2

    @property
    def max_context_samples(self) -> int:  # sa3_infer.Engine.max_context_samples
        n = self.sample_size - int(si.DURATION_PADDING_S * self.sample_rate)
        return (n // self.downsampling_ratio) * self.downsampling_ratio

    @classmethod
    def of(cls, engine: si.Engine) -> "Geometry":
        return cls(engine.sample_rate, engine.downsampling_ratio, engine.sample_size, engine.channels)


def region_context(r: Mapping[str, Any], n_total: int, geom: Geometry, inp: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """``(m0, m1, c0, c1)`` exactly as ``sa3_infer.cmd_inpaint`` computes them."""
    sr = geom.sample_rate
    m0, m1 = seconds_to_samples(float(r["start"]), sr), seconds_to_samples(float(r["end"]), sr)
    m1 = min(m1, n_total)
    c0, c1 = si.choose_context(m0, m1, n_total, sr, float(inp["context_s"]), float(inp["min_context_s"]),
                               geom.max_context_samples, geom.downsampling_ratio)
    return m0, m1, c0, c1


# --------------------------------------------------------------------------- execution


class TrackCache:
    """The most recent few tracks at the model rate (regions are sorted by track)."""

    def __init__(self, size: int = 4):
        self.size = size
        self.data: dict[tuple[str, int, int], tuple[np.ndarray, dict]] = {}

    def get(self, path: Path, sr: int, channels: int) -> tuple[np.ndarray, dict]:
        k = (str(path), sr, channels)
        if k not in self.data:
            if len(self.data) >= self.size:
                self.data.pop(next(iter(self.data)))
            self.data[k] = si.load_track(str(path), sr, channels)
        return self.data[k]


def gpu_stats(reset: bool = False) -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    gb = 1024 ** 3
    d = {
        "device": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / gb, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 3),
    }
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return d


def free_engine() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _grid_block(cfg: GridConfig, cond: Condition | None, job: Job, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    g = {"config": cfg.source, "condition": job.condition, "task": job.task, "grid_seed": job.seed,
         "regions_file": cfg.regions, "stem": job.stem}
    if cond is not None:
        g.update(model=cond.model, lora=cond.lora, lora_strength=cond.lora_strength, prompt_source=cond.prompt_source,
                 arm=cond.arm, with_sections=cond.with_sections, sections_from=cond.sections_from)
    if extra:
        g.update(extra)
    return g


def run_inpaint(engine: si.Engine, cond: Condition, job: Job, cfg: GridConfig, rs: RegionSet, prompter: Prompter,
                tracks: TrackCache, adapter: dict | None, argv: Sequence[str]) -> None:
    r = job.region
    assert r is not None
    sr = engine.sample_rate
    source, src_info = tracks.get(rs.audio_path(job.track_id), sr, engine.channels)
    m0, m1, c0, c1 = region_context(r, source.shape[0], Geometry.of(engine), cfg.inpaint)
    pinfo = prompter.build(cond, job.track_id, (c0 / sr, c1 / sr))
    xfade_s = float(cfg.inpaint["xfade_ms"]) / 1000.0
    t0 = time.time()
    res = si.run_mask(engine, source=source, m0=m0, m1=m1, c0=c0, c1=c1, prompt=pinfo["prompt_rendered"],
                      steps=cond.steps, cfg=cond.cfg, seed=job.seed, xfade_s=xfade_s,
                      blank_mask_input=bool(cfg.inpaint["blank_mask_input"]))
    elapsed = time.time() - t0
    doc = rs.segments_doc(job.track_id)
    idx = int(r["section_index"])
    seg = doc["segments"][idx] if doc is not None else {"start": r.get("section_start"), "end": r.get("section_end"),
                                                         "label": r["label"]}
    meta = si.common_meta(SimpleNamespace(command="inpaint", _argv=list(argv)), engine, cond.lora_strength,
                          adapter, cond.steps, cond.cfg)
    meta.update(
        tool=TOOL, grid=_grid_block(cfg, cond, job, {k: v for k, v in pinfo.items() if k not in ("prompt", "prompt_rendered")}),
        track_id=job.track_id, audio=src_info, segments_json=str(rs.segments_path(job.track_id)),
        segments_tool=None if doc is None else doc.get("tool"), prompt=pinfo["prompt"],
        prompt_rendered=pinfo["prompt_rendered"], seed=job.seed,
        section={"index": idx, "label": r["label"], "start_s": float(seg["start"]), "end_s": float(seg["end"]),
                 "label_raw": seg["label"]},
        mask={"start_s": float(r["start"]), "end_s": float(r["end"]), "pad_s": rs.pad_s, "min_len_s": rs.min_len_s,
              "start_sample": m0, "end_sample": m1},
        **res["meta"],
        elapsed_s=round(elapsed, 3),
    )
    if getattr(engine, "last_call", None) is not None:
        meta["engine_call"] = engine.last_call
    si._write_mask_outputs(job.out_dir, job.stem, res, sr, meta, write_orig=True)


def run_generate(engine: si.Engine, cond: Condition, job: Job, cfg: GridConfig, rs: RegionSet, prompter: Prompter,
                 adapter: dict | None, argv: Sequence[str]) -> None:
    sr = engine.sample_rate
    duration = float(cfg.generate["duration"])
    if duration <= 0 or duration * sr > engine.sample_size:
        raise SystemExit(f"generate.duration must be in (0, {engine.sample_size / sr:.1f}] s")
    tdur = rs.track(job.track_id).get("duration") or float(cfg.generate["sections_window_s"])
    window = (0.0, min(float(tdur), float(cfg.generate["sections_window_s"])))
    fields, origin = prompter.gen_base_fields(cond, job.track_id)
    pinfo = prompter.build(cond, job.track_id, window, fields, origin)
    seed = gen_seed(job.seed, job.track_id)
    t0 = time.time()
    audio = engine.generate(prompt=pinfo["prompt_rendered"], duration=duration, steps=cond.steps,
                            cfg_scale=cond.cfg, seed=seed)
    elapsed = time.time() - t0
    wav = job.out_dir / f"{job.stem}.wav"
    si.write_wav(wav, audio, sr)
    meta = si.common_meta(SimpleNamespace(command="generate", _argv=list(argv)), engine, cond.lora_strength,
                          adapter, cond.steps, cond.cfg)
    meta.update(
        tool=TOOL, grid=_grid_block(cfg, cond, job, {k: v for k, v in pinfo.items() if k not in ("prompt", "prompt_rendered")}),
        track_id=job.track_id, prompt=pinfo["prompt"], prompt_rendered=pinfo["prompt_rendered"], seed=seed,
        grid_seed=job.seed, duration=duration, n_samples=int(audio.shape[0]), seconds=audio.shape[0] / sr,
        peak=float(np.max(np.abs(audio))) if audio.size else 0.0, elapsed_s=round(elapsed, 3), file=wav.name,
    )
    if getattr(engine, "last_call", None) is not None:
        meta["engine_call"] = engine.last_call
    si.write_json(job.sidecar, meta)
    log(f"wrote {wav}")


def run_lower_anchor(job: Job, donor: Mapping[str, Any], cfg: GridConfig, rs: RegionSet, tracks: TrackCache,
                     geom: Geometry, argv: Sequence[str]) -> None:
    r = job.region
    assert r is not None
    sr = geom.sample_rate
    t0 = time.time()
    source, src_info = tracks.get(rs.audio_path(job.track_id), sr, geom.channels)
    m0, m1, c0, c1 = region_context(r, source.shape[0], geom, cfg.inpaint)
    n_ctx = c1 - c0
    ctx = np.ascontiguousarray(source[c0:c1])
    m0r, m1r = m0 - c0, m1 - c0
    d_audio, d_info = tracks.get(rs.audio_path(donor["track_id"]), sr, geom.channels)
    fill, off = donor_fill(d_audio, sr, donor, m1r - m0r)
    if fill.shape[1] != ctx.shape[1]:
        fill = np.repeat(fill[:, :1], ctx.shape[1], axis=1)
    xf = seconds_to_samples(float(cfg.inpaint["xfade_ms"]) / 1000.0, sr)
    spliced = splice(ctx, fill, m0r, m1r, xf)
    hard = ctx.copy()
    hard[m0r:m1r] = fill
    res = {"ctx": ctx, "raw": hard, "spliced": spliced, "orig": ctx.copy(), "have": n_ctx}
    doc = rs.segments_doc(job.track_id)
    idx = int(r["section_index"])
    seg = doc["segments"][idx] if doc is not None else {"start": r.get("section_start"), "end": r.get("section_end"),
                                                         "label": r["label"]}
    meta = {
        "tool": TOOL,
        "command": "inpaint",  # run_eval.read_regions keeps sidecars with command None/"inpaint"
        "anchor": "lower",
        "argv": list(argv),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": None,
        "geometry": asdict(geom),
        "grid": _grid_block(cfg, None, job),
        "track_id": job.track_id,
        "audio": src_info,
        "segments_json": str(rs.segments_path(job.track_id)),
        "segments_tool": None if doc is None else doc.get("tool"),
        "prompt": None,
        "seed": job.seed,
        "section": {"index": idx, "label": r["label"], "start_s": float(seg["start"]), "end_s": float(seg["end"]),
                    "label_raw": seg["label"]},
        "mask": {"start_s": float(r["start"]), "end_s": float(r["end"]), "pad_s": rs.pad_s, "min_len_s": rs.min_len_s,
                 "start_sample": m0, "end_sample": m1},
        "context": {"start_s": c0 / sr, "end_s": c1 / sr, "start_sample": c0, "end_sample": c1,
                    "duration_s": n_ctx / sr, "source_samples": n_ctx},
        "mask_in_context": {"start_s": m0r / sr, "end_s": m1r / sr, "start_sample": m0r, "end_sample": m1r},
        "xfade": {"seconds": xf / sr, "samples": xf, "shape": "raised-cosine, inside the mask"},
        "donor": {
            "track_id": donor["track_id"], "section_index": int(donor["section_index"]), "label": donor["label"],
            "region_start_s": float(donor["start"]), "region_end_s": float(donor["end"]),
            "region_duration_s": float(donor["end"]) - float(donor["start"]),
            "duration_diff_s": (float(donor["end"]) - float(donor["start"])) - (float(r["end"]) - float(r["start"])),
            "fill_offset_sample": off, "fill_offset_s": None if off is None else off / sr,
            "tiled": off is None, "alt_family": donor.get("alt_family"), "audio_sha256": d_info.get("sha256"),
        },
        "files_note": "_raw = hard cut (no fades), _spliced = 50 ms in-mask crossfades, _orig = source context",
        "elapsed_s": round(time.time() - t0, 3),
    }
    si._write_mask_outputs(job.out_dir, job.stem, res, sr, meta, write_orig=True)


# --------------------------------------------------------------------------- manifest / logs


MANIFEST_COLUMNS = ("condition", "task", "track_id", "section_index", "label", "seed", "path", "sidecar", "elapsed_s",
                    "model", "lora", "lora_strength", "steps", "cfg", "prompt_source", "arm", "with_sections",
                    "anchor", "prompt_rendered", "stem")


def manifest_rows(cfg: GridConfig, jobs: Mapping[str, Sequence[Job]]) -> list[dict[str, Any]]:
    conds = {c.name: c for c in cfg.conditions}
    rows = []
    for name, js in jobs.items():
        c = conds.get(name)
        for j in js:
            if not j.done:
                continue
            try:
                meta = json.loads(j.sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rows.append({
                "condition": name, "task": j.task, "track_id": j.track_id,
                "section_index": None if j.region is None else int(j.region["section_index"]),
                "label": None if j.region is None else j.region["label"], "seed": j.seed,
                "path": str(j.main_path), "sidecar": str(j.sidecar), "elapsed_s": meta.get("elapsed_s"),
                "model": None if c is None else c.model, "lora": None if c is None else c.lora,
                "lora_strength": None if c is None else c.lora_strength,
                "steps": None if c is None else c.steps, "cfg": None if c is None else c.cfg,
                "prompt_source": None if c is None else c.prompt_source, "arm": None if c is None else c.arm,
                "with_sections": None if c is None else c.with_sections, "anchor": j.anchor,
                "prompt_rendered": meta.get("prompt_rendered"), "stem": j.stem,
            })
    return rows


def write_manifest(cfg: GridConfig, jobs: Mapping[str, Sequence[Job]]) -> Path:
    root = Path(cfg.out_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(cfg, jobs)
    cols = {k: [r[k] for r in rows] for k in MANIFEST_COLUMNS}
    path = root / "manifest.parquet"
    tmp = root / "manifest.parquet.tmp"
    try:
        import pandas as pd

        pd.DataFrame(cols, columns=list(MANIFEST_COLUMNS)).to_parquet(tmp, index=False)
    except ImportError:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table(cols), tmp)
        except ImportError:
            import csv

            path = root / "manifest.csv"
            tmp = root / "manifest.csv.tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
                w.writeheader()
                w.writerows(rows)
            log("WARNING: neither pandas nor pyarrow is installed; wrote manifest.csv instead of manifest.parquet")
    os.replace(tmp, path)
    return path


def append_log(cfg: GridConfig, rec: Mapping[str, Any]) -> None:
    p = Path(cfg.out_root) / "grid_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec), default=str) + "\n")


# --------------------------------------------------------------------------- dry run


def _missing_inputs(cfg: GridConfig, rs: RegionSet, jobs: Mapping[str, Sequence[Job]]) -> dict[str, list[str]]:
    miss: dict[str, list[str]] = {"audio": [], "segments": [], "lora": [], "caption_files": []}
    conds = {c.name: c for c in cfg.conditions}
    tids = {j.track_id for js in jobs.values() for j in js}
    need_segs = {j.track_id for js in jobs.values() for j in js
                 if j.task == "inpaint" or (j.condition in conds and conds[j.condition].with_sections)}
    if cfg.lower_name and cfg.lower_name in jobs:
        donors = pair_lower_anchors([j.region for j in jobs[cfg.lower_name]], rs.candidates,
                                    int(cfg.anchors.get("seed", 0)))
        tids |= {d["track_id"] for d in donors.values()}
    for tid in sorted(tids):
        if not rs.audio_path(tid).is_file():
            miss["audio"].append(str(rs.audio_path(tid)))
    for tid in sorted(need_segs):
        sp = rs.segments_path(tid)
        if sp is None or not sp.is_file():
            miss["segments"].append(f"{tid}: {sp}")
    for c in cfg.conditions:
        if c.lora and not lora_present(cfg.engine, c.lora):
            miss["lora"].append(f"{c.name}: {c.lora}")
        if c.prompt_source == "arm_caption" and c.caption_root:
            for tid in sorted({j.track_id for j in jobs.get(c.name, [])}):
                p = Path(c.caption_root) / str(c.arm) / rs.split / f"{tid}.txt"
                if not p.is_file():
                    miss["caption_files"].append(str(p))
    miss["lora"] = sorted(set(miss["lora"]))
    return miss


def dry_run(cfg: GridConfig, rs: RegionSet, jobs: Mapping[str, Sequence[Job]], est_s: tuple[float, float]) -> dict[str, Any]:
    prompter = Prompter(cfg, rs)
    geom = Geometry(int(cfg.geometry["sample_rate"]), int(cfg.geometry["downsampling_ratio"]),
                    int(cfg.geometry["sample_size"]), int(cfg.geometry["channels"]))
    conds = {c.name: c for c in cfg.conditions}
    summary: dict[str, Any] = {"conditions": {}, "totals": {}}
    print(f"[grid] plan: {cfg.source or '(dict)'}\n"
          f"  regions {rs.path} ({rs.split}): {len(rs.regions)} regions on {len({r['track_id'] for r in rs.regions})} "
          f"tracks; seeds {cfg.seeds}; tasks {list(cfg.tasks)}\n  out_root {cfg.out_root}")
    tot_todo = {"inpaint": 0, "generate": 0, "anchor": 0}
    tot_done = 0
    loads = 0
    prev_key = None
    for name, js in jobs.items():
        c = conds.get(name)
        todo = [j for j in js if not j.done]
        n = {t: sum(1 for j in js if j.task == t) for t in TASKS}
        nt = {t: sum(1 for j in todo if j.task == t) for t in TASKS}
        tot_done += len(js) - len(todo)
        entry = {"inpaint": n["inpaint"], "generate": n["generate"], "todo_inpaint": nt["inpaint"],
                 "todo_generate": nt["generate"], "done": len(js) - len(todo)}
        if c is None:
            tot_todo["anchor"] += len(todo)
            print(f"  - {name:<14} lower anchor (CPU): {n['inpaint']} regions, {len(todo)} to do")
        else:
            tot_todo["inpaint"] += nt["inpaint"]
            tot_todo["generate"] += nt["generate"]
            if todo and c.engine_key != prev_key:
                loads += 1
            if todo:
                prev_key = c.engine_key
            lora = "-" if not c.lora else f"{c.lora} @ {c.lora_strength:g}"
            print(f"  - {name:<14} {c.model} steps {c.steps} cfg {c.cfg:g} lora {lora}\n"
                  f"    {'':<14} prompt {c.prompt_source}{'/' + str(c.arm) if c.prompt_source == 'arm_caption' else ''}"
                  f" sections {c.with_sections} ({c.sections_from}); inpaint {n['inpaint']} ({nt['inpaint']} to do), "
                  f"generate {n['generate']} ({nt['generate']} to do)")
            ex = next((j for j in js if j.task == "inpaint"), None) or next((j for j in js if j.task == "generate"), None)
            if ex is not None:
                try:
                    if ex.task == "inpaint":
                        dur = float(ex.region.get("track_duration") or rs.track(ex.track_id).get("duration") or ex.region["end"])
                        n_tot = seconds_to_samples(dur, geom.sample_rate)
                        _, _, c0, c1 = region_context(ex.region, n_tot, geom, cfg.inpaint)
                        p = prompter.build(c, ex.track_id, (c0 / geom.sample_rate, c1 / geom.sample_rate))
                    else:
                        tdur = rs.track(ex.track_id).get("duration") or cfg.generate["sections_window_s"]
                        f, o = prompter.gen_base_fields(c, ex.track_id)
                        p = prompter.build(c, ex.track_id, (0.0, min(float(tdur), float(cfg.generate["sections_window_s"]))), f, o)
                    print(f"    {'':<14} e.g. {ex.stem}: {p['prompt_rendered']!r} [{p['prompt_origin']}]")
                    entry["example_prompt"] = p["prompt_rendered"]
                except (SystemExit, Exception) as e:  # noqa: BLE001 - a plan should still print
                    print(f"    {'':<14} e.g. prompt: ERROR {e}")
                    entry["example_prompt_error"] = str(e)
        summary["conditions"][name] = entry
    miss = _missing_inputs(cfg, rs, jobs)
    for k, v in miss.items():
        if v:
            print(f"  missing {k}: {len(v)} (e.g. {v[0]})")
    calls = tot_todo["inpaint"] + tot_todo["generate"]
    lo, hi = calls * est_s[0] / 60.0, calls * est_s[1] / 60.0
    print(f"  totals: {tot_todo['inpaint']} inpaint + {tot_todo['generate']} generate model calls to do, "
          f"{tot_todo['anchor']} lower-anchor files, {tot_done} outputs already done; {loads} model load(s)\n"
          f"  estimate: {lo:.0f}-{hi:.0f} min of sampling at {est_s[0]:g}-{est_s[1]:g} s per call, plus ~1 min per load")
    summary["totals"] = {**tot_todo, "done": tot_done, "model_loads": loads, "est_min": [lo, hi]}
    summary["missing"] = miss
    return summary


# --------------------------------------------------------------------------- run


def default_engine_factory(cfg: GridConfig) -> EngineFactory:
    def factory(model: str, lora: str | None) -> si.Engine:
        if cfg.engine == "acestep":
            from ravi.eval.acestep_grid import AceGridEngine

            return AceGridEngine.load(model, lora, device=cfg.device)  # type: ignore[return-value]
        return si.Engine.load(model, lora, device=cfg.device, half=cfg.half, allow_cpu=cfg.allow_cpu)

    return factory


def run_grid(cfg: GridConfig, *, engine_factory: EngineFactory | None = None, dry: bool = False,
             limit_regions: int | None = None, limit_prompts: int | None = None, fail_fast: bool = False,
             est_s: tuple[float, float] = (2.0, 12.0), argv: Sequence[str] = ()) -> dict[str, Any]:
    rs = RegionSet(cfg.regions, cfg.audio_root)
    jobs = plan(cfg, rs, limit_regions, limit_prompts)
    if dry:
        return dry_run(cfg, rs, jobs, est_s)
    for c in cfg.conditions:
        if c.lora and any(not j.done for j in jobs[c.name]) and not lora_present(cfg.engine, c.lora):
            raise SystemExit(f"condition {c.name}: lora {c.lora}: no such {cfg.engine} adapter")
    if cfg.engine == "sa3" and not si.hf_env()["HF_TOKEN_set"]:
        log("WARNING: HF_TOKEN is not set; gated stabilityai/* downloads fail unless HF_HOME has the files")
    root = Path(cfg.out_root)
    root.mkdir(parents=True, exist_ok=True)
    si.write_json(root / "grid_config.json", {
        "source": cfg.source, "argv": list(argv), "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "regions_sha256": si.sha256_file(rs.path),
    })
    factory = engine_factory or default_engine_factory(cfg)
    nominal = Geometry(int(cfg.geometry["sample_rate"]), int(cfg.geometry["downsampling_ratio"]),
                       int(cfg.geometry["sample_size"]), int(cfg.geometry["channels"]))
    prompter = Prompter(cfg, rs)
    tracks = TrackCache()
    conds = {c.name: c for c in cfg.conditions}
    engine: si.Engine | None = None
    engine_key: tuple[str, str | None] | None = None
    summary: dict[str, Any] = {"conditions": {}, "failures": []}
    t_all = time.time()
    for name, js in jobs.items():
        todo = [j for j in js if not j.done]
        c = conds.get(name)
        rec: dict[str, Any] = {"condition": name, "n_jobs": len(js), "n_skipped": len(js) - len(todo), "n_done": 0,
                               "n_failed": 0, "model_load_s": 0.0}
        log(f"condition {name}: {len(todo)} to do, {len(js) - len(todo)} already done")
        if not todo:
            summary["conditions"][name] = rec
            continue
        t_cond = time.time()
        gpu_stats(reset=True)
        adapter = None
        if c is not None:
            if engine is None or engine_key != c.engine_key:
                engine = None
                free_engine()
                t0 = time.time()
                engine = factory(c.model, c.lora)
                engine_key = c.engine_key
                rec["model_load_s"] = round(time.time() - t0, 2)
                g = Geometry.of(engine)
                if (g.sample_rate, g.downsampling_ratio) != (nominal.sample_rate, nominal.downsampling_ratio):
                    log(f"WARNING: model geometry {asdict(g)} != config geometry {asdict(nominal)}: the lower "
                        "anchor's contexts will not match this condition's")
                log(f"loaded {c.model} lora={c.lora} in {rec['model_load_s']} s; gpu {gpu_stats()}")
            else:
                log(f"reusing the loaded {c.model} lora={c.lora}")
            if c.lora:
                engine.set_lora_strength(float(c.lora_strength))
            adapter = lora_info(cfg.engine, c.lora)
            rec.update(model=c.model, lora=c.lora, lora_strength=c.lora_strength, steps=c.steps, cfg=c.cfg)
        donors = pair_lower_anchors([j.region for j in todo], rs.candidates, int(cfg.anchors.get("seed", 0))) \
            if c is None else {}
        consecutive = 0
        call_times: list[float] = []
        for i, j in enumerate(todo, 1):
            t0 = time.time()
            try:
                if c is None:
                    run_lower_anchor(j, donors[region_key(j.region)], cfg, rs, tracks, nominal, argv)
                elif j.task == "inpaint":
                    run_inpaint(engine, c, j, cfg, rs, prompter, tracks, adapter, argv)
                else:
                    run_generate(engine, c, j, cfg, rs, prompter, adapter, argv)
                rec["n_done"] += 1
                consecutive = 0
                call_times.append(time.time() - t0)
                log(f"{name} {i}/{len(todo)} {j.task} {j.stem} {call_times[-1]:.1f} s")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001 - keep the grid going; reported at the end
                rec["n_failed"] += 1
                consecutive += 1
                summary["failures"].append({"condition": name, "stem": j.stem, "error": repr(e)})
                log(f"FAILED {name} {j.task} {j.stem}: {e!r}")
                traceback.print_exc()
                if fail_fast or consecutive >= MAX_CONSECUTIVE_FAILURES:
                    write_manifest(cfg, jobs)
                    raise SystemExit(f"{name}: {consecutive} consecutive failures, stopping") from e
        rec["elapsed_s"] = round(time.time() - t_cond, 2)
        rec["mean_job_s"] = round(float(np.mean(call_times)), 3) if call_times else None
        rec["gpu"] = gpu_stats()
        log(f"condition {name} finished: {rec['n_done']} done, {rec['n_failed']} failed, {rec['elapsed_s']} s "
            f"(mean {rec['mean_job_s']} s/job, load {rec['model_load_s']} s); gpu {rec['gpu']}")
        append_log(cfg, {**rec, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        summary["conditions"][name] = rec
        write_manifest(cfg, jobs)
    engine = None
    free_engine()
    mpath = write_manifest(cfg, jobs)
    summary["manifest"] = str(mpath)
    summary["elapsed_s"] = round(time.time() - t_all, 2)
    log(f"all done in {summary['elapsed_s']} s; {len(summary['failures'])} failures; manifest {mpath}")
    return summary


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ravi.eval.grid", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML (or JSON) grid config")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and counts; load nothing")
    ap.add_argument("--conditions", default=None, help="comma-separated subset (the lower anchor by its name)")
    ap.add_argument("--tasks", default=None, help="comma-separated subset of inpaint,generate")
    ap.add_argument("--no-anchors", action="store_true", help="skip the lower anchor")
    ap.add_argument("--limit-regions", type=int, default=None, help="first N regions (smoke)")
    ap.add_argument("--limit-prompts", type=int, default=None, help="first N generation prompts (smoke)")
    ap.add_argument("--regions", default=None, help="override the config's regions file")
    ap.add_argument("--out-root", default=None, help="override the config's out_root")
    ap.add_argument("--audio-root", default=None, help="override the config's audio_root")
    ap.add_argument("--caption-root", default=None, help="override every condition's caption_root")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-half", action="store_true", help="fp32 model weights")
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fail-fast", action="store_true", help="stop at the first failed output")
    ap.add_argument("--est-s", default="2,12", help="dry-run estimate: seconds per model call, 'lo,hi'")
    return ap


def main(argv: Sequence[str] | None = None, engine_factory: EngineFactory | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    raw = load_config_file(args.config)
    ov: dict[str, Any] = {"regions": args.regions, "out_root": args.out_root, "audio_root": args.audio_root,
                          "caption_root": args.caption_root, "device": args.device}
    if args.tasks:
        ov["tasks"] = [t for t in args.tasks.split(",") if t]
    if args.conditions:
        ov["conditions_filter"] = args.conditions
    if args.no_anchors:
        ov["no_anchors"] = True
    if args.no_half:
        ov["no_half"] = True
    if args.allow_cpu:
        ov["allow_cpu"] = True
    cfg = resolve_config(raw, ov, source=str(args.config))
    lo, hi = (float(x) for x in args.est_s.split(","))
    summary = run_grid(cfg, engine_factory=engine_factory, dry=args.dry_run, limit_regions=args.limit_regions,
                       limit_prompts=args.limit_prompts, fail_fast=args.fail_fast, est_s=(lo, hi), argv=argv)
    return 1 if summary.get("failures") else 0


if __name__ == "__main__":
    sys.exit(main())
