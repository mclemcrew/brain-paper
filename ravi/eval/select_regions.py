"""Pick the section-inpainting evaluation regions (plan Phase 2.3 Tier 0, Phase 4.2).

For the tracks of one split (``test`` or ``val``) that have human labels, every interior
verse/bridge/chorus section of at least ``--min-len-s`` seconds becomes a candidate
region, padded by ``--pad-s`` (:func:`ravi.eval.masks.section_masks`, the same call
``sa3_infer inpaint`` makes). ``--max-tracks N`` keeps a seeded random subset of the
eligible tracks and ``--max-regions-per-track K`` a seeded random subset of each track's
regions; both draws are deterministic for a given ``--seed`` (the per-track draw depends
only on the seed and the track id, so it does not change when ``--max-tracks`` does).

Writes

* ``--out`` (default ``eval/regions_{split}.json``)::

      {"tool", "split", "seed", "pad_s", "min_len_s", "max_tracks", "max_regions_per_track",
       "sample_rate", "n_tracks_split", "n_tracks_eligible", "n_tracks", "n_regions",
       "n_candidates",
       "regions":    [{track_id, section_index, label, start, end, duration,
                       section_start, section_end, track_duration, audio_path,
                       segments_json, alt_family}, ...],      # the evaluated regions
       "candidates": [... same records, every eligible region of the split ...],
       "tracks":     {track_id: {audio_path, segments_json, duration, alt_family,
                                 has_labels, human_caption, human_genre_tag,
                                 caption_is_junk, id3_genre_tokens, id3_bpm,
                                 captions: {arm: "rvkr␞…"}}}}

  ``start`` / ``end`` / ``duration`` are the padded mask in track seconds (what
  ``sa3_infer`` regenerates), ``section_start`` / ``section_end`` the labelled section.
  ``audio_path`` is the prepared 44.1 kHz full track as listed in
  ``manifests/prepared_audio.parquet``, else ``prepared/44100/{track_id}.wav`` by
  convention (relative to the repo; ``ravi.eval.grid --audio-root`` re-roots it); see
  :func:`track_audio` for where ``track_duration`` comes from. ``segments_json`` is relative to the
  directory of ``--out``. ``candidates`` is the donor pool for the grid's lower anchor;
  ``tracks`` carries, for every track of the split (labelled or not), the manifest fields
  behind the arm captions and the captions themselves (``--arms``, default A,B,C,E, plus D
  with ``--autocaptions``), built with :func:`ravi.captions.make_arms.build_caption` exactly
  as make_arms writes them, except that arm E is stored *without* its ``sections:`` field:
  the grid appends the sections of the window the model sees. The grid uses these when
  ``caption_root`` has no file for the track, so it needs neither pandas nor the manifest.
* ``{segments-dir}/{track_id}.json`` (default ``{dir of --out}/segments/{split}/``): the
  human labels in the :mod:`ravi.structure.schema` format (``schema.from_human``), for every
  labelled track of the split.

No timestamps: the same inputs give byte-identical outputs.

Example (on the Mac, where the Audacity labels are)::

    .venv/bin/python -m ravi.eval.select_regions --split test --labels-dir ~/Downloads/labels \\
        --max-tracks 20 --max-regions-per-track 1 --seed 0 --out eval/regions_test_tier0.json
    .venv/bin/python -m ravi.eval.select_regions --split test --labels-dir ~/Downloads/labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ravi.eval.masks import DEFAULT_MIN_LEN_S, DEFAULT_PAD_S, section_masks
from ravi.structure import schema
from ravi.structure.labels import INTERIOR_TARGETS, parse_audacity_labels

TOOL = "ravi.eval.select_regions"
SAMPLE_RATE = 44100
#: manifest columns copied into ``tracks`` (what make_arms.build_caption reads)
TRACK_FIELDS = ("human_caption", "human_genre_tag", "caption_is_junk", "id3_genre_tokens", "id3_bpm")


# --------------------------------------------------------------------------- helpers


def _jsonable(x: Any) -> Any:
    """numpy / pandas scalars and arrays -> plain JSON values; NA -> None."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)) or (hasattr(x, "tolist") and not np.isscalar(x)):
        seq = x.tolist() if hasattr(x, "tolist") else list(x)
        if isinstance(seq, list):
            return [_jsonable(v) for v in seq]
        x = seq
    try:
        import pandas as pd

        if x is pd.NA or (not isinstance(x, (str, bytes)) and pd.isna(x)):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (int, str)):
        return x
    return str(x)


def track_seed(seed: int, track_id: str) -> int:
    """A per-track RNG seed that depends only on ``seed`` and ``track_id``."""
    h = hashlib.sha1(f"{int(seed)}|{track_id}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def load_split(splits_path: Path, split: str) -> list[str]:
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if split not in data or not isinstance(data[split], list):
        raise SystemExit(f"{splits_path}: no list for split {split!r}")
    return sorted(str(t) for t in data[split])


def _labels_file(row: Mapping[str, Any], labels_dir: Path | None) -> Path | None:
    lp = row.get("labels_path")
    if lp is None or (isinstance(lp, float) and math.isnan(lp)) or not str(lp).strip():
        return None
    p = Path(str(lp))
    return (labels_dir / p.name) if labels_dir is not None else p


def _rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:  # different drive (Windows)
        return str(path)


def arm_captions(row: Mapping[str, Any], arms: Sequence[str], autocaption: str | None) -> dict[str, str]:
    """``{arm: caption}`` (fields joined with FIELD_SEP), arm E without ``sections:``; arm D
    only when an auto-caption exists for the track."""
    from ravi.captions.make_arms import build_caption

    out = {}
    for arm in arms:
        if arm == "D" and not autocaption:
            continue
        out[arm] = build_caption(arm, row, None, autocaption=autocaption).text
    return out


def track_audio(tid: str, row: Mapping[str, Any], prep: Any, prepared_root: str | None) -> dict[str, Any]:
    """``audio_path`` (prepared 44.1 kHz WAV, repo-relative) and ``duration`` (seconds).

    Duration source, in order: ``prepared_audio.parquet`` (``n_frames_44k``), the WAV under
    ``--prepared-root`` (``soundfile.info``), the manifest's ``duration_s`` (the MP3's
    duration, within a few ms of the decoded WAV). The path follows the
    ``prepared/44100/{track_id}.wav`` convention of :mod:`ravi.data.prepare_audio` when the
    prepared manifest does not list the track (full preparation ran on the GPU box)."""
    if tid in prep.index and isinstance(prep.loc[tid].get("prepared_path_44k"), str):
        p = prep.loc[tid]
        return {"audio_path": str(p["prepared_path_44k"]),
                "duration": round(float(p["n_frames_44k"]) / SAMPLE_RATE, 6),
                "duration_source": "prepared_manifest"}
    rel = f"prepared/{SAMPLE_RATE}/{tid}.wav"
    if prepared_root:
        wav = Path(prepared_root) / str(SAMPLE_RATE) / f"{tid}.wav"
        if wav.is_file():
            import soundfile as sf

            info = sf.info(str(wav))
            return {"audio_path": rel, "duration": round(info.frames / info.samplerate, 6),
                    "duration_source": "wav"}
    d = _jsonable(row.get("duration_s"))
    return {"audio_path": rel, "duration": None if d is None else round(float(d), 6),
            "duration_source": "manifest_duration_s"}


def region_record(track_id: str, mask: Any, seg: Mapping[str, Any], track: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "section_index": int(mask.index),
        "label": mask.label,
        "start": round(float(mask.start), 6),
        "end": round(float(mask.end), 6),
        "duration": round(float(mask.end - mask.start), 6),
        "section_start": round(float(seg["start"]), 6),
        "section_end": round(float(seg["end"]), 6),
        "track_duration": track["duration"],
        "audio_path": track["audio_path"],
        "segments_json": track["segments_json"],
        "alt_family": track["alt_family"],
    }


# --------------------------------------------------------------------------- core


def select(
    split_ids: Sequence[str],
    tracks: Mapping[str, Mapping[str, Any]],
    segments: Mapping[str, list[dict[str, Any]]],
    *,
    pad_s: float = DEFAULT_PAD_S,
    min_len_s: float = DEFAULT_MIN_LEN_S,
    max_tracks: int | None = None,
    max_regions_per_track: int | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """``(regions, candidates, eligible track ids)``.

    ``segments[track_id]`` are schema segment dicts; ``tracks[track_id]`` needs
    ``duration``, ``audio_path``, ``segments_json`` and ``alt_family``. Tracks without an
    entry in ``segments`` (no labels) are not eligible.
    """
    candidates: dict[str, list[dict[str, Any]]] = {}
    for tid in sorted(split_ids):
        if tid not in segments:
            continue
        segs = segments[tid]
        masks = section_masks(segs, tracks[tid]["duration"], pad_s=pad_s, min_len_s=min_len_s)
        if masks:
            candidates[tid] = [region_record(tid, m, segs[m.index], tracks[tid]) for m in masks]
    eligible = sorted(candidates)
    chosen = list(eligible)
    if max_tracks is not None and max_tracks < len(eligible):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(eligible), size=int(max_tracks), replace=False)
        chosen = sorted(eligible[i] for i in idx)
    regions: list[dict[str, Any]] = []
    for tid in chosen:
        recs = candidates[tid]
        if max_regions_per_track is not None and max_regions_per_track < len(recs):
            rng = np.random.default_rng(track_seed(seed, tid))
            keep = sorted(rng.choice(len(recs), size=int(max_regions_per_track), replace=False))
            recs = [recs[i] for i in keep]
        regions.extend(recs)
    all_candidates = [r for tid in eligible for r in candidates[tid]]
    return regions, all_candidates, eligible


def build(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    if args.max_tracks is not None and args.max_tracks < 1:
        raise SystemExit("--max-tracks must be >= 1")
    if args.max_regions_per_track is not None and args.max_regions_per_track < 1:
        raise SystemExit("--max-regions-per-track must be >= 1")
    out = Path(args.out or f"eval/regions_{args.split}.json")
    base = out.parent
    seg_dir = Path(args.segments_dir) if args.segments_dir else base / "segments" / args.split
    labels_dir = Path(args.labels_dir).expanduser() if args.labels_dir else None

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    autocaps: dict[str, str] = {}
    if args.autocaptions:
        from ravi.captions.make_arms import load_autocaptions

        autocaps = load_autocaptions(Path(args.autocaptions))
        if "D" not in arms:
            arms.append("D")
    split_ids = load_split(Path(args.splits), args.split)
    man = pd.read_parquet(args.manifest).set_index("track_id")
    prep = pd.read_parquet(args.prepared).set_index("track_id")
    missing = [t for t in split_ids if t not in man.index]
    if missing:
        raise SystemExit(f"{len(missing)} {args.split} tracks are not in {args.manifest}: {missing[:3]}")

    tracks: dict[str, dict[str, Any]] = {}
    segments: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}
    for tid in split_ids:
        row = man.loc[tid].to_dict()
        rec: dict[str, Any] = {k: _jsonable(row.get(k)) for k in TRACK_FIELDS}
        rec["alt_family"] = _jsonable(row.get("alt_family"))
        rec["has_labels"] = bool(row.get("has_labels"))
        rec["captions"] = arm_captions(row, arms, autocaps.get(tid))
        rec.update(track_audio(tid, row, prep, args.prepared_root))
        rec["segments_json"] = None
        tracks[tid] = rec
        if not rec["has_labels"]:
            skipped[tid] = "no human labels"
            continue
        if rec["duration"] is None:
            skipped[tid] = "no duration (not in the prepared manifest, no WAV, no duration_s)"
            continue
        lf = _labels_file(row, labels_dir)
        if lf is None or not lf.is_file():
            raise SystemExit(f"{tid}: labels file {lf} not found (pass --labels-dir)")
        segs = parse_audacity_labels(lf, rec["duration"])
        doc = schema.from_human(segs, tid, duration=rec["duration"])
        seg_path = seg_dir / f"{tid}.json"
        schema.save(doc, seg_path)
        rec["segments_json"] = _rel(seg_path, base)
        segments[tid] = doc["segments"]

    regions, candidates, eligible = select(
        split_ids, tracks, segments, pad_s=args.pad_s, min_len_s=args.min_len_s,
        max_tracks=args.max_tracks, max_regions_per_track=args.max_regions_per_track, seed=args.seed,
    )
    doc = {
        "tool": TOOL,
        "split": args.split,
        "seed": int(args.seed),
        "pad_s": float(args.pad_s),
        "min_len_s": float(args.min_len_s),
        "labels": sorted(INTERIOR_TARGETS),
        "max_tracks": args.max_tracks,
        "max_regions_per_track": args.max_regions_per_track,
        "sample_rate": SAMPLE_RATE,
        "arms": arms,
        "autocaptions": args.autocaptions,
        "n_tracks_split": len(split_ids),
        "n_tracks_eligible": len(eligible),
        "n_tracks": len({r["track_id"] for r in regions}),
        "n_regions": len(regions),
        "n_candidates": len(candidates),
        "skipped": skipped,
        "regions": regions,
        "candidates": candidates,
        "tracks": tracks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    labels = {}
    for r in regions:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    print(f"[select_regions] {args.split}: {len(split_ids)} tracks, {len(segments)} labelled, "
          f"{len(eligible)} with interior sections ({len(candidates)} candidate regions); "
          f"kept {doc['n_tracks']} tracks, {len(regions)} regions {labels} -> {out}", flush=True)
    return doc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ravi.eval.select_regions",
                                 description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--manifest", default="manifests/tracks.parquet")
    ap.add_argument("--prepared", default="manifests/prepared_audio.parquet")
    ap.add_argument("--splits", default="manifests/splits.json")
    ap.add_argument("--prepared-root", default="prepared",
                    help="where prepare_audio wrote {sr}/{track_id}.wav (duration fallback)")
    ap.add_argument("--labels-dir", default=None,
                    help="directory of the Audacity label files (default: the manifest's labels_path)")
    ap.add_argument("--arms", default="A,B,C,E", help="arm captions to store per track (default A,B,C,E)")
    ap.add_argument("--autocaptions", default=None,
                    help="auto-caption JSONL (e.g. captions/midasheng_test.jsonl): also stores arm D")
    ap.add_argument("--pad-s", type=float, default=DEFAULT_PAD_S)
    ap.add_argument("--min-len-s", type=float, default=DEFAULT_MIN_LEN_S)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--max-regions-per-track", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default eval/regions_{split}.json")
    ap.add_argument("--segments-dir", default=None, help="default {dir of --out}/segments/{split}")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
