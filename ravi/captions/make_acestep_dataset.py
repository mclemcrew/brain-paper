"""Convert a ``make_arms`` arm split into ACE-Step 1.5's per-file dataset layout at 48 kHz.

Input: ``{arm_dir}/{split}/`` as written by :mod:`ravi.captions.make_arms` (44.1 kHz
``{track_id}.wav`` + ``{track_id}.txt`` captions whose fields are separated by
``FIELD_SEP`` "␞", trigger ``rvkr`` first, ``sections: ...`` last for arm E).

Output: ``{out_dir}/`` with, per clip,

* ``{name}.wav``: the clip resampled to 48 kHz (polyphase, exact ratio 160/147 for
  44.1 kHz input), channels unchanged. PCM_24 by default; a clip whose resampled peak
  would clip in PCM is written as 32-bit float instead and the manifest says so;
* ``{name}.caption.txt``: the caption fields joined with ``", "``;
* ``{name}.lyrics.txt``: only for arms whose ``sections:`` field is rendered as lyrics
  (arm E by default), one section tag per line (``[intro]\\n[verse]\\n...``). Every
  other arm gets no lyrics file, so ACE-Step falls back to ``"[Instrumental]"``;
* ``{name}.json``: ``{"caption": ..., "bpm": <int>, "is_instrumental": true}``; ``bpm``
  comes from the track manifest's ``id3_bpm`` (``--manifest``) and is left out when unknown;

plus ``conversion_manifest.json`` (one record per clip, the input and output sha256) and
the marker ``.make_acestep_dataset``.

How ACE-Step reads this layout (ACE-Step-1.5 @ ca1e85fe, read, not run):

* ``acestep/training/dataset_builder_modules/scan.py:31-36`` walks the directory for
  ``{.wav,.mp3,.flac,.ogg,.opus}`` (``models.py:11``);
* ``audio_io.py:30-38`` reads ``{base}.caption.txt``; ``audio_io.py:41-69`` reads
  ``{base}.json`` (``caption`` overrides the caption file, ``scan.py:84-87``; ``bpm``,
  ``keyscale``, ``timesignature``, ``language``; ``is_instrumental`` is *not* read);
* ``audio_io.py:72-85`` reads lyrics from ``{base}.lyrics.txt`` **and falls back to
  ``{base}.txt``**. A make_arms caption file left next to the audio would therefore be
  read as lyrics, so this tool never writes ``{name}.txt``;
* ``scan.py:66-68``: a lyrics file sets ``is_instrumental = False``; without one the
  lyrics default to ``"[Instrumental]"`` (``scan.py:77``). ``is_instrumental`` is only
  carried as metadata by training_v2 (``acestep/training_v2/preprocess.py:300``);
* ``csv_metadata.py:10-78`` reads *every* ``*.csv`` in the directory and lets its
  ``caption`` column override the captions, so this tool writes no CSV (make_arms'
  ``manifest.csv`` has ``file`` and ``caption`` columns and must not be copied here).

Preprocessing then loads each clip with torchaudio, resamples to 48 kHz if needed, forces
stereo and truncates to ``max_duration`` (default 240 s,
``dataset_builder_modules/preprocess_audio.py:4-21``); the VAE encodes 48 kHz audio to
25 Hz latents (1920 samples per frame, ``core/generation/handler/conditioning_masks.py:55``).
The 47 s make_arms windows are far below 240 s, so nothing is truncated.

Deterministic: clips are processed in ``filelist.txt`` order (else sorted), scipy's
``resample_poly`` has no randomness, and the JSON files are written with sorted keys.

Example::

    python -m ravi.captions.make_acestep_dataset --arm-dir data/arms/E --split train \\
        --manifest manifests/tracks.parquet --out data/acestep/E/train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from ravi.train.caption_aug import FIELD_SEP, JOIN_SEP, CaptionAugConfig, check_trigger, split_fields

log = logging.getLogger("ravi.captions.make_acestep_dataset")

TARGET_SR = 48000
MARKER = ".make_acestep_dataset"
MANIFEST_NAME = "conversion_manifest.json"
SECTIONS_PREFIX = "sections:"  # same prefix as ravi.captions.make_arms.SECTIONS_PREFIX
#: Arms whose ``sections:`` field becomes ACE-Step lyric section tags.
SECTION_LYRICS_ARMS = frozenset({"E"})
#: PCM_24 full scale; a resampled clip above this would clip.
_PCM24_MAX = 1.0 - 2.0 ** -23
#: Sidecar suffixes ACE-Step reads; a name ending in one of these would be ambiguous.
_RESERVED_SUFFIXES = (".caption", ".lyrics")


@dataclass
class Clip:
    name: str  # file stem = make_arms track id
    wav: Path
    txt: Path


def list_clips(split_dir: Path) -> list[Clip]:
    """Clips of a make_arms split, in ``filelist.txt`` order (sorted WAV names if absent)."""
    split_dir = Path(split_dir)
    filelist = split_dir / "filelist.txt"
    if filelist.is_file():
        names = [ln.strip() for ln in filelist.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        names = sorted(p.name for p in split_dir.iterdir() if p.suffix.lower() == ".wav" and not p.name.startswith("."))
    clips = []
    for n in names:
        wav = split_dir / n
        if wav.suffix.lower() != ".wav":
            raise ValueError(f"{filelist}: {n!r} is not a .wav")
        clips.append(Clip(wav.stem, wav, wav.with_suffix(".txt")))
    stems = [c.name for c in clips]
    if len(set(stems)) != len(stems):
        raise ValueError(f"{split_dir}: duplicate clip names")
    return clips


def section_lyrics(field: str) -> str:
    """``"sections: intro, verse, outro"`` -> ``"[intro]\\n[verse]\\n[outro]"``."""
    if not field.lower().startswith(SECTIONS_PREFIX):
        raise ValueError(f"not a sections field: {field!r}")
    labels = [x.strip() for x in field.split(":", 1)[1].split(",") if x.strip()]
    for lab in labels:
        if "[" in lab or "]" in lab or "\n" in lab:
            raise ValueError(f"bad section label {lab!r}")
    return "\n".join(f"[{lab}]" for lab in labels)


def render(caption_text: str, sections_as_lyrics: bool, trigger: str = "rvkr") -> tuple[str, str]:
    """(caption, lyrics) from a make_arms caption. The ``sections:`` field (if any) becomes
    the lyrics when ``sections_as_lyrics``; otherwise it stays a caption field. Lyrics are
    ``""`` when there are none (ACE-Step then uses ``"[Instrumental]"``)."""
    cfg = CaptionAugConfig(trigger=trigger)
    check_trigger(caption_text, cfg)
    fields = [f.strip() for f in split_fields(caption_text, FIELD_SEP)]
    if any(not f for f in fields):
        raise ValueError(f"empty field in caption {caption_text!r}")
    body, lyrics = [], ""
    for f in fields:
        if sections_as_lyrics and f.lower().startswith(SECTIONS_PREFIX):
            if lyrics:
                raise ValueError(f"two sections fields in {caption_text!r}")
            lyrics = section_lyrics(f)
        else:
            body.append(f)
    caption = JOIN_SEP.join(body)
    if "\n" in caption:
        raise ValueError(f"newline in caption {caption!r}")
    return caption, lyrics


def resample(data: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    """``[frames, channels]`` float64 -> float64 at ``sr_out`` (scipy polyphase, Kaiser
    window). Output length is ``ceil(frames * sr_out / sr_in)``."""
    if sr_in == sr_out:
        return data.astype(np.float64, copy=True)
    from scipy.signal import resample_poly

    r = Fraction(sr_out, sr_in)
    out = resample_poly(data.astype(np.float64), r.numerator, r.denominator, axis=0)
    expected = math.ceil(len(data) * r)
    if len(out) != expected:  # resample_poly's documented length; guard against a change
        raise RuntimeError(f"resample_poly returned {len(out)} frames, expected {expected}")
    return out


def load_bpms(manifest: Path | None) -> dict[str, int]:
    """track_id -> BPM from the track manifest's ``id3_bpm`` (same rule as make_arms)."""
    if manifest is None:
        return {}
    import pandas as pd

    from ravi.captions.make_arms import bpm_value

    df = pd.read_parquet(manifest) if str(manifest).endswith(".parquet") else pd.read_csv(manifest)
    if "track_id" not in df.columns or "id3_bpm" not in df.columns:
        raise ValueError(f"{manifest}: needs track_id and id3_bpm columns")
    out = {}
    for row in df[["track_id", "id3_bpm"]].to_dict("records"):
        b = bpm_value(row)
        if b is not None:
            out[str(row["track_id"])] = b
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def convert_clip(clip: Clip, dst_dir: Path, sections_as_lyrics: bool, bpms: Mapping[str, int],
                 subtype: str = "PCM_24", trigger: str = "rvkr") -> dict:
    if clip.name.endswith(_RESERVED_SUFFIXES):
        raise ValueError(f"{clip.name!r} ends in an ACE-Step sidecar suffix {_RESERVED_SUFFIXES}")
    if not clip.txt.is_file():
        raise FileNotFoundError(f"caption missing: {clip.txt}")
    caption, lyrics = render(clip.txt.read_text(encoding="utf-8").strip(), sections_as_lyrics, trigger)
    data, sr = sf.read(str(clip.wav), dtype="float64", always_2d=True)
    if len(data) == 0:
        raise ValueError(f"{clip.wav}: empty")
    out = resample(data, sr)
    peak = float(np.max(np.abs(out)))
    used = subtype
    if subtype != "FLOAT" and peak > _PCM24_MAX:
        used = "FLOAT"
        log.warning("%s: resampled peak %.4f > full scale; writing 32-bit float", clip.name, peak)
    wav = dst_dir / f"{clip.name}.wav"
    sf.write(str(wav), out.astype(np.float32 if used == "FLOAT" else np.float64), TARGET_SR, subtype=used)
    (dst_dir / f"{clip.name}.caption.txt").write_text(caption + "\n", encoding="utf-8")
    if lyrics:
        (dst_dir / f"{clip.name}.lyrics.txt").write_text(lyrics + "\n", encoding="utf-8")
    meta: dict[str, Any] = {"caption": caption, "is_instrumental": True}
    if clip.name in bpms:
        meta["bpm"] = int(bpms[clip.name])
    (dst_dir / f"{clip.name}.json").write_text(_json_text(meta), encoding="utf-8")
    return {
        "name": clip.name,
        "source_wav": str(clip.wav),
        "source_sha256": _sha256(clip.wav),
        "source_sr": int(sr),
        "source_frames": int(len(data)),
        "frames": int(len(out)),
        "seconds": len(out) / TARGET_SR,
        "channels": int(out.shape[1]),
        "subtype": used,
        "peak": round(peak, 6),
        "caption": caption,
        "lyrics": lyrics,
        "bpm": meta.get("bpm"),
        "wav_sha256": _sha256(wav),
    }


def _arm_name(arm_dir: Path) -> str:
    return Path(arm_dir).resolve().name.upper()


def make_acestep_dataset(
    arm_dir: Path,
    split: str,
    out_dir: Path,
    manifest: Path | None = None,
    arm: str | None = None,
    sections: str = "auto",
    limit: int | None = None,
    overwrite: bool = False,
    subtype: str = "PCM_24",
    trigger: str = "rvkr",
) -> dict:
    """Build ``out_dir``; returns the conversion manifest. Builds in ``{out_dir}.tmp`` and
    renames into place only when every clip converted."""
    arm_dir, out_dir = Path(arm_dir), Path(out_dir)
    split_dir = arm_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} not found")
    arm = (arm or _arm_name(arm_dir)).upper()
    if sections == "auto":
        sections_as_lyrics = arm in SECTION_LYRICS_ARMS
    elif sections in ("lyrics", "caption"):
        sections_as_lyrics = sections == "lyrics"
    else:
        raise ValueError("sections must be auto, lyrics or caption")
    if subtype not in ("PCM_24", "PCM_16", "FLOAT"):
        raise ValueError("subtype must be PCM_24, PCM_16 or FLOAT")
    clips = list_clips(split_dir)
    if limit is not None:
        clips = clips[: int(limit)]
    if not clips:
        raise ValueError(f"{split_dir}: no clips")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not (out_dir / MARKER).is_file():
            raise FileExistsError(f"{out_dir} is not empty and has no {MARKER} marker; refusing to touch it")
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to rebuild it")
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        if not (tmp / MARKER).is_file():
            raise FileExistsError(f"{tmp} exists without {MARKER}; remove it by hand")
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / MARKER).write_text("", encoding="utf-8")

    bpms = load_bpms(manifest)
    try:
        recs = []
        for i, c in enumerate(clips, 1):
            recs.append(convert_clip(c, tmp, sections_as_lyrics, bpms, subtype, trigger))
            if i % 50 == 0:
                log.info("converted %d/%d", i, len(clips))
        doc = {
            "arm": arm,
            "split": split,
            "source": str(split_dir.resolve()),
            "sample_rate": TARGET_SR,
            "sections_as_lyrics": sections_as_lyrics,
            "manifest": str(manifest) if manifest else None,
            "n_clips": len(recs),
            "n_with_lyrics": sum(1 for r in recs if r["lyrics"]),
            "n_with_bpm": sum(1 for r in recs if r["bpm"] is not None),
            "n_float": sum(1 for r in recs if r["subtype"] == "FLOAT"),
            "limit": limit,
            "clips": recs,
        }
        (tmp / MANIFEST_NAME).write_text(_json_text(doc), encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ravi.captions.make_acestep_dataset",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--arm-dir", required=True, type=Path, help="make_arms arm directory, e.g. data/arms/E")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True, type=Path, help="output directory (ACE-Step per-file layout)")
    p.add_argument("--manifest", type=Path, help="tracks.parquet (or .csv) with track_id, id3_bpm")
    p.add_argument("--arm", help="arm letter (default: the arm directory name)")
    p.add_argument("--sections", choices=("auto", "lyrics", "caption"), default="auto",
                   help="where the sections: field goes; auto = lyrics for arm E only")
    p.add_argument("--subtype", choices=("PCM_24", "PCM_16", "FLOAT"), default="PCM_24")
    p.add_argument("--limit", type=int, help="first N clips only (smoke runs)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--trigger", default="rvkr")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    doc = make_acestep_dataset(a.arm_dir, a.split, a.out, a.manifest, a.arm, a.sections, a.limit,
                               a.overwrite, a.subtype, a.trigger)
    print(json.dumps({k: v for k, v in doc.items() if k != "clips"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
