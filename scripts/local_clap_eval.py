#!/usr/bin/env python3
"""Local CLAP evaluation of base vs adapted SA3 renders on the held-out comparison captions.

Renders each held-out caption with every arm through the app's own MLX path
(``worker.runtime`` + ``worker.invoke("sa3_mlx")``, the argument list of
``scripts/render_local.py``: 8 steps, CFG 1, APG 1, init-noise 1, fp16, --free-models, adapter
strength 1; adapter arms get ``<trigger>, `` prepended, base arms none), then scores every render
with the app's offline CLAP ONNX pack (``Xenova/clap-htsat-unfused`` q8, CPU, 2 threads).

Scoring, per render:

* audio: the float WAV is downmixed to mono and resampled to 48 kHz (``analysis.mono_window``),
  normalised to -16 LUFS (``ravi.eval.embed.lufs_normalise``), embedded as the re-normalised mean
  of the L2-normalised embeddings of 10 s windows every 5 s (plus one end-aligned window when the
  last fitted window stops short; a clip under 10 s is one repeat-padded window);
* ``text_score_machine`` / ``text_score_human``: cosine with the CLAP text embedding of the
  caption's machine draft (comparison CSV ``caption``, = cloud arm D text) and of its human
  caption + genre tag (cloud arm B text, ``build_caption('B')`` without the trigger). Both are
  computed for every render whatever the prompt was; ``prompt_source`` says which one it was
  rendered from; ``text_score_shuffled``: cosine with the prompt-source text of the *next*
  caption in the list (a mismatched-caption control);
* ``audio_sim``: cosine with the held-out original's embedding (same windowing, whole track);
* tail check on the written render before normalisation: RMS dB of the last 1.0 s and last
  0.25 s minus RMS dB of the body [1 s, T-1 s]; ``spike_flag`` if the 1 s tail is > +3 dB,
  ``fade_flag`` if < -6 dB.

Statistics (``summary.json`` / ``summary.md`` / ``results_table.tex``): per arm and prompt
source, medians of every score; every adapter arm against the base arm of the same DiT, per
prompt source and metric: pairs matched on (caption, seed), reduced to per-caption medians over
seeds (``stats.cluster_medians``), Wilcoxon signed-rank on those (``stats.paired_compare``),
median difference with a bootstrap 95% CI over captions, Holm across the adapter arms within one
(prompt source, metric). Only paired deltas are meaningful: this CLAP is not the cloud study's
LAION music checkpoint, so absolute scores are not comparable with the cloud tables.

Example (smoke)::

    .venv/bin/python scripts/local_clap_eval.py --limit 2 --seeds 42 \\
      --arm base-small=:sm-music:.local-studio/v5-small-20261105b/models \\
      --arm r1=.local-studio/v5-small-20261105b/report.json#1000:sm-music:.local-studio/v5-small-20261105b/models:rvkr \\
      --out-dir .local-studio/eval-20261105/smoke

``--score-only`` rescoring existing renders and rebuilding the summary never imports the MLX
runtime (``ravi.local.worker`` is imported lazily, only when rendering).

``ADAPTER_PATH`` may be a ``.safetensors`` file or ``<run>/report.json#<step>`` (resolved through
the report's ``checkpoints[step]['path']``, relative to the run folder).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_RUNTIME = ROOT / ".local-runtime/stable-audio-3/optimized/mlx"
DEFAULT_CSV = ROOT / ".local-studio/representative-corpus-20261103/comparison-20.csv"
DEFAULT_PROJECT = ROOT / ".local-studio/v5-medium-20261105/projects/6c2c078c5a4e4102b2f3b2273926a935/project.json"
DEFAULT_TAGGER_ROOT = ROOT / ".local-studio"
DEFAULT_TRAINING_PATTERN = "lora_train|ravi.local.worker|train_selected_corpus"
DECODERS = {"sm-music": "same-s", "medium": "same-l"}
SOURCES = ("machine", "human")
SAMPLER = {"steps": 8, "cfg": 1.0, "apg": 1.0, "init_noise_level": 1.0, "dit_dtype": "fp16", "free_models": True}
CLAP_SR, WINDOW_S, HOP_S = 48000, 10.0, 5.0
TARGET_LUFS = -16.0
SPIKE_DB, FADE_DB = 3.0, -6.0
SCORE_COLUMNS = ["text_score_machine", "text_score_human", "text_score_shuffled", "audio_sim",
                 "tail_db_1s", "tail_db_025s"]
CSV_COLUMNS = ["arm", "caption_id", "seed", "dit", "prompt_source", *SCORE_COLUMNS,
               "spike_flag", "fade_flag", "lufs_mono_in", "seconds", "path"]
COMPARE_METRICS = ["text_score_machine", "text_score_human", "audio_sim"]
MATCHED = {"machine": "text_score_machine", "human": "text_score_human"}
MDE_FACTOR = 2.8  # z_{1-alpha/2} + z_{power} = 1.96 + 0.84 (alpha .05 two-sided, power .8)
DEFAULT_COMPARE = ("r2", "r3")


# --------------------------------------------------------------------------- arms and data

def parse_arm(text: str) -> dict[str, Any]:
    """``NAME=ADAPTER_PATH:DIT:MODEL_DIR[:TRIGGER]``; empty ADAPTER_PATH means the base model."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--arm {text!r}: expected NAME=ADAPTER_PATH:DIT:MODEL_DIR[:TRIGGER]")
    name, rest = text.split("=", 1)
    parts = rest.split(":")
    if len(parts) not in (3, 4) or not name or "/" in name:
        raise argparse.ArgumentTypeError(f"--arm {text!r}: expected NAME=ADAPTER_PATH:DIT:MODEL_DIR[:TRIGGER]")
    adapter, dit, model_dir = parts[:3]
    trigger = parts[3] if len(parts) == 4 else ""
    if dit not in DECODERS:
        raise argparse.ArgumentTypeError(f"--arm {name}: DIT must be one of {sorted(DECODERS)}")
    if trigger and not adapter:
        raise argparse.ArgumentTypeError(f"--arm {name}: a base arm takes no trigger")
    return {"name": name, "adapter_spec": adapter, "adapter": resolve_adapter(adapter) if adapter else None,
            "dit": dit, "model_dir": Path(model_dir).resolve(), "trigger": trigger}


def resolve_adapter(spec: str) -> Path:
    if "#" in spec:
        report, step = spec.rsplit("#", 1)
        report = Path(report).resolve()
        cps = json.loads(report.read_text()).get("checkpoints") or {}
        if step not in cps:
            raise argparse.ArgumentTypeError(f"{report}: no checkpoint {step} (have {sorted(cps)})")
        path = report.parent / cps[step]["path"]
    else:
        path = Path(spec).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"adapter {path} not found")
    return path


def parse_compare(text: str) -> tuple[str, str]:
    """``NAME_A:NAME_B`` -> (A, B); the comparison is A - B."""
    parts = text.split(":")
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise argparse.ArgumentTypeError(f"--compare {text!r}: expected NAME_A:NAME_B (two different arm names)")
    return parts[0], parts[1]


def resolve_compares(arms: Sequence[dict[str, Any]],
                     compares: Sequence[tuple[str, str]] | None) -> list[tuple[str, str]]:
    """Validate ``--compare`` pairs (both adapter arms, same DiT); ``None`` -> r2:r3 if both exist."""
    by_name = {a["name"]: a for a in arms}
    if compares is None:
        return [DEFAULT_COMPARE] if all(n in by_name and by_name[n].get("adapter") for n in DEFAULT_COMPARE) \
            and by_name[DEFAULT_COMPARE[0]]["dit"] == by_name[DEFAULT_COMPARE[1]]["dit"] else []
    out = []
    for a, b in compares:
        for n in (a, b):
            if n not in by_name:
                raise ValueError(f"--compare {a}:{b}: unknown arm {n!r} (have {sorted(by_name)})")
            if not by_name[n].get("adapter"):
                raise ValueError(f"--compare {a}:{b}: {n!r} is a base arm; use the adapter-vs-base tests")
        if by_name[a]["dit"] != by_name[b]["dit"]:
            raise ValueError(f"--compare {a}:{b}: different DiTs ({by_name[a]['dit']} vs {by_name[b]['dit']})")
        if (a, b) not in out:
            out.append((a, b))
    return out


def import_worker():
    """The app's MLX worker; imported only when rendering (never for ``--score-only``)."""
    try:
        from ravi.local import worker
    except ImportError as e:
        raise SystemExit(f"cannot import the MLX render runtime ({e}); use --score-only to rescore existing renders")
    return worker


def base_for(arms: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Adapter arm name -> name of the base arm with the same DiT (exactly one per DiT)."""
    bases: dict[str, list[str]] = {}
    for a in arms:
        if not a.get("adapter"):
            bases.setdefault(a["dit"], []).append(a["name"])
    out = {}
    for a in arms:
        if a.get("adapter"):
            found = bases.get(a["dit"], [])
            if len(found) > 1:
                raise ValueError(f"several base arms for DiT {a['dit']}: {found}")
            if found:
                out[a["name"]] = found[0]
    return out


def load_captions(csv_path: Path, project_json: Path, limit: int | None) -> list[dict[str, Any]]:
    """Held-out captions with both text variants and the decoded original of each."""
    import pandas as pd
    from ravi.captions.make_arms import build_caption, TRIGGER

    rows = list(csv.DictReader(csv_path.open()))
    project = json.loads(project_json.read_text())
    held = {t["sha256"]: t for t in project["tracks"] if t["split"] == "validation"}
    manifest = pd.read_parquet(ROOT / "manifests/tracks.parquet").set_index("audio_filename")
    out = []
    for i, r in enumerate(rows):
        track = held.get(r["sha256"])
        if track is None:
            raise ValueError(f"{r['filename']}: not a validation track of {project_json}")
        cap = build_caption("B", manifest.loc[r["filename"]].to_dict(), None)
        assert cap.fields[0] == TRIGGER and len(cap.fields) > 1, (r["filename"], cap.fallbacks)
        out.append({"caption_id": f"c{i + 1:02d}", "filename": r["filename"], "sha256": r["sha256"],
                    "machine": r["caption"].strip(), "human": ", ".join(cap.fields[1:]),
                    "human_fallbacks": cap.fallbacks,
                    "original": str((project_json.parent / track["path"]).resolve())})
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- CLAP

class ClapEmbedder:
    """Both ONNX encoders of the app's tagger pack, kept resident (``Tagger`` drops the text one).

    The exported text graph takes ``input_ids`` only (no attention mask), so texts are encoded
    one at a time, unpadded, truncated to 77 tokens like ``Tagger``."""

    def __init__(self, store_root: Path):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from ravi.local.analysis import FILES, REPO, REVISION, pack_status

        if not pack_status(store_root)["ready"]:
            raise SystemExit(f"CLAP pack missing under {store_root}/models/tagger (install the tagger in Setup).")
        ort.disable_telemetry_events()
        d = Path(store_root) / "models" / "tagger"
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = False
        self.audio = ort.InferenceSession(str(d / FILES[0]), sess_options=opts, providers=["CPUExecutionProvider"])
        self.text = ort.InferenceSession(str(d / FILES[1]), sess_options=opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>"); self.tokenizer.enable_truncation(max_length=77)
        self._full = Tokenizer.from_file(str(d / "tokenizer.json"))  # token counts of truncated captions
        self.model = {"repo": REPO, "revision": REVISION, "files": list(FILES[:2])}
        self.truncated: dict[str, int] = {}
        self._text_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def _run(self, session, feeds):
        return session.run(None, {i.name: feeds[i.name] for i in session.get_inputs()})[0][0]

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for t in texts:
            if t not in self._text_cache:
                enc = self.tokenizer.encode(t)
                if enc.overflowing:
                    self.truncated[t] = len(self._full.encode(t).ids)
                feeds = {"input_ids": np.array([enc.ids], np.int64),
                         "attention_mask": np.array([enc.attention_mask], np.int64)}
                self._text_cache[t] = self._unit(self._run(self.text, feeds).astype(np.float64))
            rows.append(self._text_cache[t])
        return np.asarray(rows)

    def embed_audio(self, y: np.ndarray) -> np.ndarray:
        """48 kHz mono -> unit vector (mean of 10 s windows, hop 5 s, end-aligned last window)."""
        from ravi.local.analysis import clap_features
        win, hop = int(WINDOW_S * CLAP_SR), int(HOP_S * CLAP_SR)
        if len(y) <= win:
            starts = [0]
        else:
            starts = list(range(0, len(y) - win + 1, hop))
            if starts[-1] + win < len(y):
                starts.append(len(y) - win)
        embs = []
        for s in starts:
            feeds = {"input_features": clap_features(y[s:s + win]), "is_longer": np.array([[False]])}
            embs.append(self._unit(self._run(self.audio, feeds).astype(np.float64)))
        return self._unit(np.mean(embs, axis=0))


def clap_input(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Mono 48 kHz, -16 LUFS (gain only, true peak <= -1 dBTP)."""
    import soundfile as sf
    from ravi.eval.embed import lufs_normalise
    from ravi.local.analysis import mono_window
    with sf.SoundFile(str(path)) as f:
        y = mono_window(f, 0.0, len(f) / f.samplerate, CLAP_SR)
    y, stats = lufs_normalise(y, CLAP_SR, TARGET_LUFS)
    return y, stats


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-12)) if x.size else math.nan


def tail_check(path: Path) -> dict[str, Any]:
    """Tail level of the written render (all channels) relative to its body [1 s, T-1 s]."""
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    n = len(a)
    body = rms_db(a[sr:max(sr, n - sr)])
    t1 = rms_db(a[max(0, n - sr):]) - body
    t025 = rms_db(a[max(0, n - sr // 4):]) - body
    return {"tail_db_1s": t1, "tail_db_025s": t025, "spike_flag": int(t1 > SPIKE_DB),
            "fade_flag": int(t1 < FADE_DB), "seconds": n / sr, "peak": float(np.max(np.abs(a))) if n else 0.0}


# --------------------------------------------------------------------------- rendering

def training_pids(pattern: str) -> list[int]:
    res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    mine = {os.getpid(), os.getppid()}
    return [int(p) for p in res.stdout.split() if p.strip() and int(p) not in mine]


def wait_for_idle_gpu(pattern: str, poll_s: float = 60.0) -> None:
    waited = 0.0
    while True:
        pids = training_pids(pattern)
        if not pids:
            if waited:
                print(f"[pause] no training process after {waited / 60:.0f} min; resuming", flush=True)
            return
        print(f"[pause] training running (pids {pids}); waited {waited / 60:.0f} min; next check in {poll_s:.0f} s",
              flush=True)
        time.sleep(poll_s); waited += poll_s


class Renderer:
    """In-process SA3 MLX renders (one module import, models reloaded per call by sa3_mlx)."""

    def __init__(self, runtime_dir: Path, profile: str, log_path: Path):
        self.runtime_dir, self.profile, self.log_path = Path(runtime_dir), profile, log_path
        self.model_dir: Path | None = None

    def _ensure(self, model_dir: Path) -> None:
        worker = import_worker()
        if self.model_dir == model_dir:
            return
        worker.runtime({"runtime": str(self.runtime_dir), "model_dir": str(model_dir), "profile": self.profile})
        weights = sys.modules["weights"]
        mod = sys.modules.get("sa3_mlx")
        if mod is not None:  # sa3_mlx bound ensure_local at import: rebind to this model dir
            mod.ensure_local, mod.is_present = weights.ensure_local, weights.is_present
        self.model_dir = model_dir

    def render(self, arm: dict[str, Any], prompt: str, seconds: float, seed: int, out: Path) -> dict[str, Any]:
        worker = import_worker()
        self._ensure(arm["model_dir"])
        prompt_sent = f"{arm['trigger']}, {prompt}" if arm["trigger"] and arm["adapter"] else prompt
        tmp = out.with_name(out.stem + ".partial.wav")
        for stale in (tmp, tmp.with_suffix(".levels.json")):
            stale.unlink(missing_ok=True)
        args = ["--dit", arm["dit"], "--decoder", DECODERS[arm["dit"]], "--prompt", prompt_sent,
                "--seconds", seconds, "--seed", seed, "--steps", SAMPLER["steps"],
                "--cfg", SAMPLER["cfg"], "--apg", SAMPLER["apg"], "--init-noise-level", SAMPLER["init_noise_level"],
                "--dit-dtype", SAMPLER["dit_dtype"], "--free-models", "--out", tmp]
        if arm["adapter"]:
            args += ["--lora", arm["adapter"], "--lora-strength", 1.0]
        argv = sys.argv
        t0 = time.monotonic()
        try:
            with self.log_path.open("a") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                print(f"\n=== {arm['name']} {out.name} {time.ctime()}", flush=True)
                worker.invoke("sa3_mlx", args)
        finally:
            sys.argv = argv
        mod = sys.modules["sa3_mlx"]
        assert mod.ensure_local is sys.modules["weights"].ensure_local, "stale model-dir binding in sa3_mlx"
        tmp.with_suffix(".levels.json").replace(out.with_suffix(".levels.json"))
        tmp.replace(out)
        meta = {"arm": arm["name"], "dit": arm["dit"], "decoder": DECODERS[arm["dit"]], "prompt": prompt,
                "prompt_sent": prompt_sent, "seconds": seconds, "seed": seed,
                "lora": str(arm["adapter"]) if arm["adapter"] else None, "strength": 1.0 if arm["adapter"] else None,
                "model_dir": str(arm["model_dir"]), "runtime": str(self.runtime_dir), "profile": self.profile,
                **SAMPLER, "elapsed_s": round(time.monotonic() - t0, 2)}
        out.with_suffix(".wav.json").write_text(json.dumps(meta, indent=2))
        return meta


def render_path(out_dir: Path, arm: str, source: str, cap_id: str, seed: int) -> Path:
    return out_dir / "renders" / arm / source / f"{cap_id}_s{seed}.wav"


def plan(arms, captions, sources, seeds):
    for arm in arms:
        for src in sources:
            for c in captions:
                for seed in seeds:
                    yield arm, src, c, seed


def run_renders(a, arms, captions, sources) -> None:
    worker = import_worker()
    worker.offline()
    renderer = Renderer(a.runtime, a.profile, a.out_dir / "render.log")
    done = skipped = 0
    jobs = list(plan(arms, captions, sources, a.seeds))
    for k, (arm, src, c, seed) in enumerate(jobs, 1):
        out = render_path(a.out_dir, arm["name"], src, c["caption_id"], seed)
        meta_path = out.with_suffix(".wav.json")
        if out.exists():
            if not a.resume:
                raise SystemExit(f"{out} exists; pass --resume to skip existing renders")
            meta = json.loads(meta_path.read_text())
            if meta["prompt"] != c[src] or meta["seed"] != seed or meta["seconds"] != a.seconds \
                    or meta["lora"] != (str(arm["adapter"]) if arm["adapter"] else None):
                raise SystemExit(f"{out} was rendered with different settings; use another --out-dir")
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if a.pause_if_training and arm["dit"] == "medium":
            wait_for_idle_gpu(a.training_pattern)
        meta = renderer.render(arm, c[src], a.seconds, seed, out)
        done += 1
        print(f"[render {k}/{len(jobs)}] {arm['name']}/{src}/{out.name} {meta['elapsed_s']:.1f} s", flush=True)
    print(f"renders: {done} new, {skipped} skipped (existing)", flush=True)


# --------------------------------------------------------------------------- scoring

def run_scores(a, arms, captions, sources) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clap = ClapEmbedder(a.tagger_root)
    text = {src: clap.embed_text([c[src] for c in captions]) for src in SOURCES}
    originals, selfsim = {}, {}
    for c in captions:
        y, _ = clap_input(Path(c["original"]))
        originals[c["caption_id"]] = clap.embed_audio(y)
        y2, _ = clap_input(Path(c["original"]))  # independent second pass: pipeline determinism
        selfsim[c["caption_id"]] = float(originals[c["caption_id"]] @ clap.embed_audio(y2))
    index = {c["caption_id"]: i for i, c in enumerate(captions)}
    rows, missing = [], 0
    for arm, src, c, seed in plan(arms, captions, sources, a.seeds):
        path = render_path(a.out_dir, arm["name"], src, c["caption_id"], seed)
        if not path.exists():
            missing += 1
            continue
        y, stats = clap_input(path)
        emb = clap.embed_audio(y)
        i = index[c["caption_id"]]
        shuffled = text[src][(i + 1) % len(captions)] if len(captions) > 1 else np.full(emb.shape, np.nan)
        rows.append({"arm": arm["name"], "caption_id": c["caption_id"], "seed": seed, "dit": arm["dit"],
                     "prompt_source": src,
                     "text_score_machine": float(text["machine"][i] @ emb),
                     "text_score_human": float(text["human"][i] @ emb),
                     "text_score_shuffled": float(shuffled @ emb),
                     "audio_sim": float(originals[c["caption_id"]] @ emb),
                     **{k: v for k, v in tail_check(path).items() if k != "peak"},
                     "lufs_mono_in": stats["lufs_in"], "path": str(path.relative_to(a.out_dir))})
    if missing:
        print(f"warning: {missing} planned renders missing; scored {len(rows)}", flush=True)
    meta = {"clap": clap.model, "original_self_similarity": selfsim,
            "truncated_captions": clap.truncated, "missing_renders": missing}
    return rows, meta


def write_scores(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------- summary

def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _median(xs: Iterable[float]) -> float:
    v = np.array([x for x in map(_f, xs) if math.isfinite(x)])
    return float(np.median(v)) if v.size else math.nan


def mde_paired(d: Iterable[float]) -> dict[str, float]:
    """Approximate minimum detectable paired effect, alpha .05 two-sided, power .8:
    ``2.8 * sd(d) / sqrt(n)`` (normal approximation; the Wilcoxon test needs ~5% more)."""
    v = np.array([x for x in map(_f, d) if math.isfinite(x)])
    sd = float(np.std(v, ddof=1)) if v.size > 1 else math.nan
    return {"n": int(v.size), "sd_diff": sd, "mde": MDE_FACTOR * sd / math.sqrt(v.size) if v.size > 1 else math.nan}


def paired_arms(rows, src: str, metric: str, a_name: str, b_name: str, n_boot: int) -> dict[str, Any] | None:
    """``a_name - b_name`` on (caption, seed)-matched renders: Wilcoxon + bootstrap CI on per-caption
    medians over seeds (primary), render-level cluster bootstrap (secondary), and the MDE."""
    from ravi.eval.stats import cluster_medians, paired_compare

    key = lambda r: (r["caption_id"], int(r["seed"]))
    b = {key(r): _f(r[metric]) for r in rows if r["arm"] == b_name and r["prompt_source"] == src}
    pairs = [(key(r), _f(r[metric]), b[key(r)]) for r in rows
             if r["arm"] == a_name and r["prompt_source"] == src and key(r) in b]
    if not pairs:
        return None
    av = np.array([p[1] for p in pairs]); bv = np.array([p[2] for p in pairs])
    cl = [p[0][0] for p in pairs]
    am, bm, ids = cluster_medians(av, bv, cl)
    primary = paired_compare(am, bm, n_boot=n_boot)
    secondary = paired_compare(av, bv, n_boot=n_boot, clusters=cl)
    mde = mde_paired(np.asarray(am, dtype=float) - np.asarray(bm, dtype=float))
    return {"prompt_source": src, "metric": metric, "matched_metric": metric == MATCHED[src],
            "level": "caption_median_over_seeds", **primary, "sd_diff": mde["sd_diff"], "mde": mde["mde"],
            "captions": ids,
            "render_level": {k: secondary[k] for k in
                             ("n", "n_nonzero", "median_diff", "ci_low", "ci_high",
                              "ci_method", "n_clusters", "p_two_sided")}}


def build_summary(rows: Sequence[dict[str, Any]], arms: Sequence[dict[str, Any]],
                  n_boot: int = 10_000,
                  compares: Sequence[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Per-arm medians, adapter-vs-same-DiT-base paired comparisons and ``--compare`` adapter pairs.

    ``rows``: score rows (``CSV_COLUMNS``); ``arms``: dicts with ``name``, ``dit`` and a truthy
    ``adapter`` for adapter arms; ``compares``: (A, B) arm-name pairs, ``None`` = r2:r3 if both
    exist (see ``resolve_compares``). Each comparison carries ``mde`` (see ``mde_paired``)."""
    from ravi.eval.stats import holm

    compares = resolve_compares(arms, compares)
    sources = [s for s in SOURCES if any(r["prompt_source"] == s for r in rows)]
    out: dict[str, Any] = {"arms": {}, "comparisons": [], "pairwise": [], "prompt_sources": sources,
                           "mde_approximation": f"{MDE_FACTOR} * sd(paired caption diffs) / sqrt(n); "
                                                "alpha .05 two-sided, power .8"}
    for arm in arms:
        entry = {"dit": arm["dit"], "adapter": bool(arm.get("adapter")), "by_prompt_source": {}}
        for src in sources:
            rs = [r for r in rows if r["arm"] == arm["name"] and r["prompt_source"] == src]
            if not rs:
                continue
            matched = [_f(r[MATCHED[src]]) for r in rs]
            shuffled = [_f(r["text_score_shuffled"]) for r in rs]
            pairs = [(m, s) for m, s in zip(matched, shuffled) if math.isfinite(m) and math.isfinite(s)]
            entry["by_prompt_source"][src] = {
                "n_renders": len(rs), "n_captions": len({r["caption_id"] for r in rs}),
                "median": {k: _median(r[k] for r in rs) for k in SCORE_COLUMNS},
                "matched_gt_shuffled": (sum(m > s for m, s in pairs) / len(pairs)) if pairs else math.nan,
                "spike_count": sum(int(_f(r["spike_flag"]) == 1) for r in rs),
                "fade_count": sum(int(_f(r["fade_flag"]) == 1) for r in rs)}
        out["arms"][arm["name"]] = entry
    bases = base_for(arms)
    out["unpaired_adapter_arms"] = [a["name"] for a in arms if a.get("adapter") and a["name"] not in bases]
    dit = {a["name"]: a["dit"] for a in arms}
    for src in sources:
        for metric in COMPARE_METRICS:
            group = []
            for arm in arms:
                if arm["name"] not in bases:
                    continue
                base = bases[arm["name"]]
                res = paired_arms(rows, src, metric, arm["name"], base, n_boot)
                if res:
                    group.append({"arm": arm["name"], "base": base, "dit": arm["dit"], **res})
            adj = holm([g["p_two_sided"] for g in group])
            for g, p in zip(group, adj):
                g["p_holm"] = float(p)
            out["comparisons"].extend(group)
            pw = []
            for a_name, b_name in compares:
                res = paired_arms(rows, src, metric, a_name, b_name, n_boot)
                if res:
                    pw.append({"arm_a": a_name, "arm_b": b_name, "dit": dit[a_name], **res})
            for g, p in zip(pw, holm([g["p_two_sided"] for g in pw])):
                g["p_holm"] = float(p)
            out["pairwise"].extend(pw)
    out["compares"] = [list(c) for c in compares]
    return out


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


def _num(v: Any, d: int = 3, signed: bool = False) -> str:
    v = _f(v)
    if not math.isfinite(v):
        return "--"
    return f"{v:+.{d}f}" if signed else f"{v:.{d}f}"


def _p(v: Any) -> str:
    v = _f(v)
    if not math.isfinite(v):
        return "--"
    return "<.001" if v < 0.001 else f"{v:.3f}".lstrip("0") if v < 1 else "1.000"


def find_cmp(summary, arm, src, metric):
    return next((c for c in summary["comparisons"]
                 if c["arm"] == arm and c["prompt_source"] == src and c["metric"] == metric), None)


def find_pair(summary, a, b, src, metric):
    return next((c for c in summary.get("pairwise", [])
                 if c["arm_a"] == a and c["arm_b"] == b and c["prompt_source"] == src and c["metric"] == metric), None)


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Local CLAP evaluation", "",
             "Scores are cosines from the q8 ONNX `Xenova/clap-htsat-unfused` (not the cloud LAION music "
             "checkpoint); compare paired deltas only. Tests: Wilcoxon signed-rank on per-caption medians "
             "over seeds, bootstrap 95% CI over captions, Holm across adapter arms per (prompt source, metric). "
             "Pairwise rows (A vs B) are A - B between adapter arms of one DiT, Holm across the --compare "
             "pairs per (prompt source, metric).", ""]
    for src in summary["prompt_sources"]:
        lines += [f"## Prompt source: {src}", "",
                  "| arm | DiT | renders | captions | text (machine) | text (human) | shuffled | matched>shuffled | audio sim | tail 1 s dB | spikes | fades |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, e in summary["arms"].items():
            s = e["by_prompt_source"].get(src)
            if not s:
                continue
            m = s["median"]
            lines.append(f"| {name} | {e['dit']} | {s['n_renders']} | {s['n_captions']} | {_num(m['text_score_machine'])} | "
                         f"{_num(m['text_score_human'])} | {_num(m['text_score_shuffled'])} | "
                         f"{_num(s['matched_gt_shuffled'], 2)} | {_num(m['audio_sim'])} | {_num(m['tail_db_1s'], 2, True)} | "
                         f"{s['spike_count']} | {s['fade_count']} |")
        lines += ["", "| arm vs base | metric | n | median diff [95% CI] | Wilcoxon p | Holm p | MDE |",
                  "|---|---|---|---|---|---|---|"]
        for c in summary["comparisons"]:
            if c["prompt_source"] != src:
                continue
            lines.append(f"| {c['arm']} vs {c['base']} | {c['metric']}{' (matched)' if c['matched_metric'] else ''} | "
                         f"{c['n']} | {_num(c['median_diff'], 4, True)} [{_num(c['ci_low'], 4, True)}, "
                         f"{_num(c['ci_high'], 4, True)}] | {_p(c['p_two_sided'])} | {_p(c['p_holm'])} | {_num(c['mde'], 4)} |")
        pw = [c for c in summary.get("pairwise", []) if c["prompt_source"] == src]
        if pw:
            lines += ["", "| A vs B (A - B) | metric | n | median diff [95% CI] | Wilcoxon p | Holm p | MDE |",
                      "|---|---|---|---|---|---|---|"]
            for c in pw:
                lines.append(f"| {c['arm_a']} vs {c['arm_b']} | {c['metric']}{' (matched)' if c['matched_metric'] else ''} | "
                             f"{c['n']} | {_num(c['median_diff'], 4, True)} [{_num(c['ci_low'], 4, True)}, "
                             f"{_num(c['ci_high'], 4, True)}] | {_p(c['p_two_sided'])} | {_p(c['p_holm'])} | "
                             f"{_num(c['mde'], 4)} |")
        lines.append("")
    mde = mde_lines(summary)
    if mde:
        lines += ["## Minimum detectable effect", "",
                  f"Paired design, n captions, alpha .05 two-sided, power .8: MDE is approximated as "
                  f"{MDE_FACTOR} * sd(paired caption diffs) / sqrt(n) (1.96 + 0.84 = 2.8 standard errors of the "
                  "mean difference; normal approximation, the Wilcoxon test is ~5% less efficient so its true MDE "
                  "is slightly larger). sd is the observed sd of the per-caption base-vs-adapter differences on the "
                  "matched metric; effects smaller than the MDE are not reliably detectable at this n.", "",
                  *mde, ""]
    return "\n".join(lines)


def mde_lines(summary: dict[str, Any]) -> list[str]:
    """One bullet per prompt source: n and the matched-metric MDE of each adapter-vs-base pair."""
    lines = []
    for src in summary["prompt_sources"]:
        cs = [c for c in summary["comparisons"] if c["prompt_source"] == src and c["matched_metric"]]
        if not cs:
            continue
        parts = [f"{c['arm']} vs {c['base']}: n={c['n']}, sd {_num(c['sd_diff'], 4)}, MDE {_num(c['mde'], 4)} "
                 f"(observed |median diff| {_num(abs(_f(c['median_diff'])), 4)})" for c in cs]
        lines.append(f"- {src} ({MATCHED[src]}): " + "; ".join(parts))
    return lines


def _tex(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def results_table_tex(summary: dict[str, Any]) -> str:
    """booktabs table: arm, DiT, n, median matched text score, delta vs base, Holm p per prompt source;
    ``--compare`` rows (A vs B) follow below a rule with Delta = A - B (CLAP column ``--``)."""
    srcs = summary["prompt_sources"]
    label = {"machine": "Machine-caption prompt", "human": "Human caption + genre prompt"}
    cols = "lll" + "rrr" * len(srcs)
    head1 = " & & " + "".join(f" & \\multicolumn{{3}}{{c}}{{{label[s]}}}" for s in srcs) + r" \\"
    rules = "".join(f"\\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(srcs)))
    head2 = r"Arm & DiT & $n$" + r" & CLAP & $\Delta$ & $p_{\mathrm{Holm}}$" * len(srcs) + r" \\"
    body = []
    for name, e in summary["arms"].items():
        n = max((s["n_captions"] for s in e["by_prompt_source"].values()), default=0)
        cells = [_tex(name), _tex(e["dit"]), str(n)]
        for src in srcs:
            s = e["by_prompt_source"].get(src)
            cells.append(_num(s["median"][MATCHED[src]]) if s else "--")
            c = find_cmp(summary, name, src, MATCHED[src])
            cells += [_num(c["median_diff"], 3, True), _p(c["p_holm"]).replace("<", "$<$")] if c else ["--", "--"]
        body.append(" & ".join(cells) + r" \\")
    pairs = list(dict.fromkeys((c["arm_a"], c["arm_b"]) for c in summary.get("pairwise", [])))
    if pairs:
        body.append(r"\midrule")
    for a_name, b_name in pairs:
        e = summary["arms"][a_name]
        n = max((s["n_captions"] for s in e["by_prompt_source"].values()), default=0)
        cells = [_tex(f"{a_name} vs {b_name}"), _tex(e["dit"]), str(n)]
        for src in srcs:
            c = find_pair(summary, a_name, b_name, src, MATCHED[src])
            cells += ["--", _num(c["median_diff"], 3, True), _p(c["p_holm"]).replace("<", "$<$")] if c \
                else ["--", "--", "--"]
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([f"\\begin{{tabular}}{{{cols}}}", r"\toprule", head1, rules, head2, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}", ""])


def write_summary(out_dir: Path, rows, arms, meta, compares=None) -> dict[str, Any]:
    summary = build_summary(rows, arms, compares=compares)
    summary.update(meta)
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2, allow_nan=False))
    (out_dir / "summary.md").write_text(summary_markdown(summary))
    (out_dir / "results_table.tex").write_text(results_table_tex(summary))
    return summary


def read_scores(path: Path) -> list[dict[str, Any]]:
    return list(csv.DictReader(path.open()))


# --------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparison-csv", type=Path, required=True)
    ap.add_argument("--project-json", type=Path, required=True,
                    help="project.json whose split=='validation' tracks hold the decoded originals")
    ap.add_argument("--arm", type=parse_arm, action="append", required=True,
                    metavar="NAME=ADAPTER_PATH:DIT:MODEL_DIR[:TRIGGER]")
    ap.add_argument("--caption-source", choices=["machine", "human-genre", "both"], default="machine",
                    help="prompts to render from (every render is scored against both caption variants)")
    ap.add_argument("--compare", type=parse_compare, action="append", metavar="NAME_A:NAME_B",
                    help="direct paired comparison A - B between two adapter arms of one DiT (repeatable; "
                         "default r2:r3 if both arms exist)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 173])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="skip renders that already exist")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    ap.add_argument("--pause-if-training", action="store_true",
                    help="before each Medium render, wait (60 s polls) while a training process runs")
    ap.add_argument("--training-pattern", default=DEFAULT_TRAINING_PATTERN, help="pgrep -f pattern")
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--profile", default="16gb", help="worker memory-limit profile only; does not change audio")
    ap.add_argument("--tagger-root", type=Path, default=DEFAULT_TAGGER_ROOT,
                    help="store root holding models/tagger (the CLAP ONNX pack)")
    a = ap.parse_args(argv)
    a.out_dir = a.out_dir.resolve()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    names = [arm["name"] for arm in a.arm]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}")
    base_for(a.arm)  # fails on ambiguous bases
    try:
        compares = resolve_compares(a.arm, a.compare)
    except ValueError as e:
        raise SystemExit(str(e))
    sources = {"machine": ["machine"], "human-genre": ["human"], "both": ["machine", "human"]}[a.caption_source]
    captions = load_captions(a.comparison_csv, a.project_json, a.limit)
    (a.out_dir / "captions.json").write_text(json.dumps(captions, indent=2))
    (a.out_dir / "config.json").write_text(json.dumps({
        "arms": [{k: str(v) if isinstance(v, Path) else v for k, v in arm.items()} for arm in a.arm],
        "compares": [list(c) for c in compares], "caption_source": a.caption_source, "seeds": a.seeds, "seconds": a.seconds, "limit": a.limit,
        "comparison_csv": str(a.comparison_csv), "project_json": str(a.project_json), "sampler": SAMPLER,
        "runtime": str(a.runtime), "profile": a.profile,
        "scoring": {"clap_sr": CLAP_SR, "window_s": WINDOW_S, "hop_s": HOP_S, "target_lufs": TARGET_LUFS,
                    "spike_db": SPIKE_DB, "fade_db": FADE_DB}}, indent=2))
    if not a.score_only:
        run_renders(a, a.arm, captions, sources)
    if a.render_only:
        return
    rows, meta = run_scores(a, a.arm, captions, sources)
    write_scores(a.out_dir / "scores.csv", rows)
    summary = write_summary(a.out_dir, rows, a.arm, meta, compares)
    print(summary_markdown(summary))
    ss = list(meta["original_self_similarity"].values())
    print(f"original self-similarity: min {min(ss):.6f}; truncated captions: {len(meta['truncated_captions'])}")


if __name__ == "__main__":
    main()
