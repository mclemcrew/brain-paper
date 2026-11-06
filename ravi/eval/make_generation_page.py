"""Text-to-music comparison page: one card per test caption, one player per model/adapter.

Standard library + ffmpeg only, so the ``transcode`` step can also run on the GPU box with its
system python. Three sub-commands:

  transcode            generation WAVs listed in an eval manifest -> MP3 (128 kb/s), named
                       ``audio/<condition>/<slug(stem)>.mp3``. Sources may be local paths or an
                       ``s3://`` prefix (downloaded one at a time with the aws CLI).
  transcode-originals  the real Ravi track for each test caption -> ``audio/original/<slug(track_id)>.mp3``
                       (first audio stream only; the corpus MP3s carry a cover-art stream).
  build                write ``gen.html`` from the manifests (+ optional metrics_gen.parquet CLAP-text
                       scores) with relative links into ``audio/``.

Examples
  python -m ravi.eval.make_generation_page transcode --manifest eval/tier1_ace/manifest.csv \
      --conditions ACE_base,ACE_B6k,ACE_E6k_nosec --seed 1 \
      --src-root s3://uaudio-model-training-checkpoints/ravi-v2/eval/tier1_ace --manifest-root /eval/tier1_ace \
      --out runs_local/generation_page
  python -m ravi.eval.make_generation_page transcode-originals --manifest eval/tier1/manifest.csv \
      --tracks-dir ~/Downloads/ravi_dataset --out runs_local/generation_page
  python -m ravi.eval.make_generation_page build --manifest eval/tier1/manifest.csv eval/tier3/manifest.csv \
      eval/tier1_ace/manifest.csv --conditions base,A6k,B2k,B4k,B6k,E6k_nosec,C6k,D6k,ACE_base,ACE_B6k,ACE_E6k_nosec \
      --seed 1 --out runs_local/generation_page
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

CONDITIONS = OrderedDict(
    [
        ("base", ("Stable Audio 3", "base model, no fine-tuning")),
        ("A6k", ("SA3 + A", "trigger word only")),
        ("B2k", ("SA3 + B 2k", "human caption + genre tag, 2k steps")),
        ("B4k", ("SA3 + B 4k", "human caption + genre tag, 4k steps")),
        ("B6k", ("SA3 + B 6k", "human caption + genre tag, 6k steps")),
        ("E6k_nosec", ("SA3 + E", "human caption + section list in training; plain prompt")),
        ("E6k", ("SA3 + E, sections in prompt", "as E, with the section list in the prompt too")),
        ("C6k", ("SA3 + C", "ID3 genre tags + BPM")),
        ("D6k", ("SA3 + D", "auto-captions from an audio language model")),
        ("base_sec", ("Stable Audio 3, sections in prompt", "base model with the section list appended")),
        ("B6k_sec", ("SA3 + B 6k, sections in prompt", "B 6k with the section list appended")),
        ("A6k_sec", ("SA3 + A, sections in prompt", "A with the section list appended")),
        ("ACE_base", ("ACE-Step 1.5", "base model, no fine-tuning")),
        ("ACE_B6k", ("ACE-Step + B", "human caption + genre tag")),
        ("ACE_E6k_nosec", ("ACE-Step + E", "human caption + section tags in training; plain prompt")),
        ("ACE_E6k", ("ACE-Step + E, sections in prompt", "as E, with section tags in the prompt too")),
        ("ACE_base_sec", ("ACE-Step 1.5, sections in prompt", "base model with section tags appended")),
        ("ACE_B6k_sec", ("ACE-Step + B, sections in prompt", "B with section tags appended")),
    ]
)

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(s: str) -> str:
    return _SLUG_RE.sub("_", s).strip("_")


def read_generate_rows(manifests: list[Path], conditions: list[str] | None, seed: str | None) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for m in manifests:
        with m.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("task") != "generate":
                    continue
                if conditions and r["condition"] not in conditions:
                    continue
                if seed and str(r["seed"]) != str(seed):
                    continue
                k = (r["condition"], r["track_id"], str(r["seed"]))
                if k in seen:  # tier 3 re-lists tier-1 base/B6k through symlinks
                    continue
                seen.add(k)
                r["_manifest"] = str(m)
                rows.append(r)
    return rows


def ffmpeg_mp3(src: Path, dst: Path, map_first_audio: bool = False, bitrate: str = "128k") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src)]
    if map_first_audio:
        cmd += ["-map", "0:a:0"]
    cmd += ["-vn", "-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    subprocess.run(cmd, check=True)


def cmd_transcode(a: argparse.Namespace) -> int:
    rows = read_generate_rows([Path(m) for m in a.manifest], a.conditions.split(",") if a.conditions else None, a.seed)
    out = Path(a.out) / "audio"
    done = skipped = 0
    tmpdir = Path(tempfile.mkdtemp(prefix="genpage_"))
    try:
        for r in rows:
            rel = r["path"]
            if a.manifest_root and rel.startswith(a.manifest_root):
                rel = rel[len(a.manifest_root):].lstrip("/")
            dst = out / r["condition"] / (slug(r["stem"]) + ".mp3")
            if dst.exists():
                skipped += 1
                continue
            if a.src_root.startswith("s3://"):
                src = tmpdir / Path(rel).name
                cp = ["aws", "s3", "cp", "--only-show-errors", a.src_root.rstrip("/") + "/" + rel, str(src)]
                if a.profile:
                    cp += ["--profile", a.profile]
                subprocess.run(cp, check=True)
            else:
                src = Path(a.src_root) / rel
            ffmpeg_mp3(src, dst)
            if a.src_root.startswith("s3://"):
                src.unlink(missing_ok=True)
            done += 1
            if done % 25 == 0:
                print(f"  {done} transcoded", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"transcoded {done}, already present {skipped}, of {len(rows)} rows -> {out}")
    return 0


def cmd_transcode_originals(a: argparse.Namespace) -> int:
    rows = read_generate_rows([Path(m) for m in a.manifest], None, None)
    track_ids = sorted({r["track_id"] for r in rows})
    tracks = Path(a.tracks_dir).expanduser()
    out = Path(a.out) / "audio" / "original"
    done = missing = 0
    for tid in track_ids:
        dst = out / (slug(tid) + ".mp3")
        if dst.exists():
            continue
        src = tracks / (tid + ".mp3")
        if not src.exists():
            missing += 1
            print(f"  missing source for {tid!r}", file=sys.stderr)
            continue
        ffmpeg_mp3(src, dst, map_first_audio=True)
        done += 1
    print(f"originals: {done} transcoded, {missing} missing, {len(track_ids)} test tracks")
    return 0


def _clap_scores(paths: list[str]) -> dict[tuple[str, str], float]:
    """(condition, stem) -> CLAP text score from metrics_gen.parquet files, if pandas is available."""
    scores: dict[tuple[str, str], float] = {}
    try:
        import pandas as pd  # noqa: WPS433
    except Exception:
        return scores
    for p in paths:
        try:
            g = pd.read_parquet(p, columns=["condition", "stem", "clap_text"])
        except Exception:
            continue
        for c, s, v in g.itertuples(index=False):
            if v == v:
                scores[(str(c), str(s))] = float(v)
    return scores


def _clean_prompt(p: str) -> str:
    p = re.sub(r"^\s*rvkr\s*,?\s*", "", p or "")
    return p.strip()


def cmd_build(a: argparse.Namespace) -> int:
    conds = a.conditions.split(",") if a.conditions else list(CONDITIONS)
    rows = read_generate_rows([Path(m) for m in a.manifest], conds, a.seed)
    out_dir = Path(a.out)
    audio = out_dir / "audio"
    scores = _clap_scores(a.metrics or [])

    by_track: dict[str, dict[str, dict]] = OrderedDict()
    for r in sorted(rows, key=lambda r: r["track_id"]):
        by_track.setdefault(r["track_id"], {})[r["condition"]] = r

    present_conds = [c for c in conds if any(c in d for d in by_track.values())]
    cards: list[str] = []
    n_missing = 0
    for tid, d in by_track.items():
        any_row = next(iter(d.values()))
        caption = _clean_prompt(any_row.get("prompt_rendered", ""))
        title = tid.replace(".mp3", "").replace("_copilot", "").strip()
        orig = audio / "original" / (slug(tid) + ".mp3")
        orig_html = (
            f'<div class="orig"><div class="exhead"><span class="name">Ravi\'s original track</span>'
            f'<span class="desc">the recording this description was written for</span></div>'
            f'<audio controls preload="none" src="audio/original/{html.escape(slug(tid))}.mp3"></audio></div>'
            if orig.exists()
            else '<div class="orig missing">original track not included</div>'
        )
        players = []
        for c in present_conds:
            r = d.get(c)
            name, desc = CONDITIONS.get(c, (c, ""))
            if not r:
                players.append(f'<div class="ex missing" data-cond="{html.escape(c)}"><div class="exhead"><span class="name">{html.escape(name)}</span></div><div class="na">no clip</div></div>')
                continue
            rel = f"audio/{c}/{slug(r['stem'])}.mp3"
            if not (out_dir / rel).exists():
                n_missing += 1
            sc = scores.get((c, r["stem"]))
            sc_html = f'<span class="score" title="CLAP text score: how well the clip matches the words">text match {sc:.2f}</span>' if sc is not None else ""
            fam = "ace" if c.startswith("ACE") else "sa3"
            players.append(
                f'<div class="ex {fam}" data-cond="{html.escape(c)}"><div class="exhead"><span class="name">{html.escape(name)}</span>'
                f'<span class="desc">{html.escape(desc)}</span>{sc_html}</div>'
                f'<audio controls preload="none" src="{html.escape(rel)}"></audio></div>'
            )
        cards.append(
            f'<section class="card" data-track="{html.escape(tid)}"><h3>{html.escape(title)}</h3>'
            f'<p class="caption">“{html.escape(caption)}”</p>{orig_html}<div class="exs">{"".join(players)}</div></section>'
        )

    legend = "".join(
        f'<li><label><input type="checkbox" class="cf" value="{html.escape(c)}" checked> <b>{html.escape(CONDITIONS.get(c, (c, ""))[0])}</b>'
        f' <span class="desc">{html.escape(CONDITIONS.get(c, (c, ""))[1])}</span></label></li>'
        for c in present_conds
    )
    note = html.escape(a.note or "")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>rAvI generations</title>
<style>
:root{{--bg:#fafafa;--fg:#1a1a1a;--card:#fff;--line:#ddd;--acc:#2b6cb0;--orig:#e8f3ea;--sa3:#fff;--ace:#f3eefb;--mut:#666}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--card:#1b1b1b;--line:#333;--acc:#7fb3ff;--orig:#16301c;--sa3:#1b1b1b;--ace:#241d33;--mut:#aaa}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1000px;margin:auto}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:14px 0}}
h3{{margin:0 0 4px}} .caption{{margin:2px 0 10px;font-style:italic;color:var(--acc)}}
audio{{width:100%}} .orig{{background:var(--orig);border:1px solid var(--line);border-radius:6px;padding:8px;margin-bottom:10px}}
.exs{{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(290px,1fr))}}
.ex{{border:1px solid var(--line);border-radius:6px;padding:8px;background:var(--sa3)}} .ex.ace{{background:var(--ace)}}
.exhead{{display:flex;gap:8px;flex-wrap:wrap;align-items:baseline;margin-bottom:4px}} .name{{font-weight:bold}} .desc{{color:var(--mut);font-size:13px}}
.score{{margin-left:auto;font-size:12px;color:var(--mut)}} .na{{color:var(--mut);font-size:13px;padding:8px 0}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0}} input[type=search]{{font:inherit;padding:4px 8px;min-width:260px}}
ul.legend{{list-style:none;padding:0;columns:2;gap:16px}} ul.legend li{{break-inside:avoid;margin:2px 0}} .hidden{{display:none}}
</style></head><body>
<h1>rAvI: the same description, every model</h1>
<p>Each card is one test track from Ravi's catalogue that no model saw in training. Its description (written by
Jessica Strauss) was given as the prompt to the base models and to every fine-tuned adapter; each produced a
30-second clip from the words alone. The green player is Ravi's actual recording, so you can judge how close each
model gets to what the description meant. Purple players are ACE-Step; white are Stable Audio 3. {note}</p>
<details open><summary>Models shown (untick to hide)</summary><ul class="legend">{legend}</ul></details>
<div class="controls"><input id="q" type="search" placeholder="filter by title or words in the description"> <span id="n"></span></div>
<div id="cards">{"".join(cards)}</div>
<script>
const cards=[...document.querySelectorAll('.card')],q=document.getElementById('q'),n=document.getElementById('n');
function apply(){{const v=q.value.trim().toLowerCase();let k=0;for(const c of cards){{const on=!v||c.textContent.toLowerCase().includes(v);c.classList.toggle('hidden',!on);if(on)k++;}}n.textContent=k+' of '+cards.length+' tracks';
const off=new Set([...document.querySelectorAll('.cf')].filter(x=>!x.checked).map(x=>x.value));for(const e of document.querySelectorAll('.ex'))e.classList.toggle('hidden',off.has(e.dataset.cond));}}
q.addEventListener('input',apply);for(const x of document.querySelectorAll('.cf'))x.addEventListener('change',apply);apply();
document.addEventListener('play',e=>{{for(const a of document.querySelectorAll('audio'))if(a!==e.target)a.pause();}},true);
</script></body></html>"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gen.html").write_text(page)
    print(f"wrote {out_dir / 'gen.html'}: {len(by_track)} tracks x {len(present_conds)} conditions; {n_missing} clips referenced but not yet transcoded")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcode")
    t.add_argument("--manifest", nargs="+", required=True)
    t.add_argument("--conditions", default="")
    t.add_argument("--seed", default="1")
    t.add_argument("--src-root", required=True, help="local dir or s3:// prefix that replaces --manifest-root")
    t.add_argument("--manifest-root", default="", help="path prefix used inside the manifest, e.g. /eval/tier1")
    t.add_argument("--profile", default=os.environ.get("AWS_PROFILE", ""))
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_transcode)
    o = sub.add_parser("transcode-originals")
    o.add_argument("--manifest", nargs="+", required=True)
    o.add_argument("--tracks-dir", required=True)
    o.add_argument("--out", required=True)
    o.set_defaults(fn=cmd_transcode_originals)
    b = sub.add_parser("build")
    b.add_argument("--manifest", nargs="+", required=True)
    b.add_argument("--metrics", nargs="*", help="metrics_gen.parquet files for CLAP text scores")
    b.add_argument("--conditions", default="")
    b.add_argument("--seed", default="1")
    b.add_argument("--note", default="")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
