"""Build the caption-arm training directories (plan Phase 1 steps 7-8, Phase 3 arm table).

For every arm and split this writes ``{out_root}/{arm}/{split}/`` with, per track,

* ``{track_id}.wav``: the prepared WAV (``prepared/{sr}/{track_id}.wav``) when the track
  fits in the training window, otherwise a WAV cropped to the window
  (:mod:`ravi.captions.windows`) plus ``{track_id}.window.json``. Uncropped WAVs are
  copied by default (``--link copy``); ``--link symlink`` writes *relative* symlinks
  and ``--link hardlink`` hard links (falling back to a copy across file systems);
* ``{track_id}.txt``: the caption, fields joined with ``FIELD_SEP`` ("␞", U+241E) as
  :mod:`ravi.train.caption_aug` expects, trigger ``rvkr`` first, ``sections:`` last;

plus ``{split}/filelist.txt`` (one WAV name per line, relative to the split directory),
``{arm}/manifest.csv``, ``{arm}/trackset.sha256``, ``{arm}/.make_arms`` (marker) and,
only with ``--sr 48000``, ``{arm}/acestep_metadata.csv``. ``{out_root}/arms_summary.json``
is merged across runs (arms not rebuilt keep their entries).

**Training**: point the trainer's data directory at ``{out_root}/{arm}/train`` (not at
``{arm}``, which also holds ``val``); ``filelist.txt`` there lists exactly the training
clips.

Window: ``(int(47 * sr) // 4096) * 4096`` frames, the trainer's ``sample_size``
(2,072,576 frames at 44.1 kHz), so upstream never random-crops (``--duration`` and
``--downsample`` change it).

Arms (Phase 3):

=====  =====================================================================
A      ``rvkr``
B      ``rvkr␞{human caption}␞{genre tag}``; junk caption -> ``rvkr␞{genre tag or ID3 tags}``
C      ``rvkr␞{ID3 tokens, cleaned, comma-joined}␞{bpm} bpm`` (bpm omitted if unknown)
D      ``rvkr␞{auto-caption}`` from ``--autocaptions`` JSONL (missing track = error)
E      B + ``sections: {labels inside the window}``; SongFormer labels for tracks without
       human labels when ``--songformer-dir`` has them, else the field is omitted
F      *experimental* (not pre-registered as a main arm):
       ``rvkr␞Genre: …␞Mood: {first sentence of human caption}␞BPM: …␞Sections: …``
=====  =====================================================================

The human caption loses its trailing period, so the rendered caption reads
"…energetic synths, Electronic" rather than "…synths., Electronic".

ID3 tokens (arm C, and fallbacks in B and F) are cleaned by :func:`clean_id3_tokens`:
typo fixes (Cleassical -> Classical, Chinematic -> Cinematic, Pops -> Pop), the junk
words ``TEST!``, ``MARKUP``, ``DELETE`` and ``alfa`` removed, tokens with ``/`` or a
client code (digits with a dash, letters run into digits) dropped, tokens of more than
four words dropped, and the pre-registered non-genre descriptors
:data:`ID3_NON_GENRE` (Mnemonic, Solo, Sound Design, Vocals, Pause, Run) dropped. What
was dropped is logged per track in the ``id3_dropped`` column of ``manifest.csv``.

Section lists drop ``unlabeled`` filler segments. SongFormer labels are normalised with
:func:`ravi.structure.labels.normalize_segments` and translated to the human vocabulary
by :data:`SONGFORMER_TO_HUMAN`, which is derived from the pre-registered coarse map in
:mod:`ravi.structure.label_map` (labels the human vocabulary shares with the same coarse
class keep their name, other ``body`` labels such as ``pre-chorus`` become ``verse``,
``transition`` labels such as ``inst`` become ``transition``, ``ignore`` (``silence``) is
dropped), so arm E's caption vocabulary does not depend on the label source.

Safety: every track and caption is planned before anything is written; each arm is built
in ``{arm}.tmp`` and swapped in only after every arm has been built and verified, so a
failed run leaves the previous arms intact. ``--overwrite`` only replaces directories
carrying the ``.make_arms`` marker. An arm is refused (unless ``--overwrite``) if an
existing sibling arm in ``out_root`` was built on a different track set or windows
(``trackset.sha256``).

``acestep_metadata.csv`` is written only for ``--sr 48000``: ACE-Step trains on 48 kHz
audio and its ``File`` column points at the WAVs written here.

Example::

    python -m ravi.captions.make_arms --manifest manifests/tracks.parquet \\
        --splits manifests/splits.json --labels-dir ~/Downloads/labels \\
        --prepared-root prepared --sr 44100 --out-root data/arms --arms A,B,C,E \\
        --split train,val
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf

from ravi.captions.windows import DOWNSAMPLE, TRAIN_DURATION_S, Window, choose_window, window_frames
from ravi.data.build_manifest import JUNK_RE
from ravi.structure import label_map, schema
from ravi.structure.labels import UNLABELED, Segment, normalize_segments, parse_audacity_labels
from ravi.structure.labels import VOCAB as HUMAN_VOCAB
from ravi.train.caption_aug import FIELD_SEP, JOIN_SEP, CaptionAugConfig, check_trigger, join_fields

log = logging.getLogger("ravi.captions.make_arms")

TRIGGER = "rvkr"
#: All arms. F is experimental (EXPERIMENTAL_ARMS): built on request, not a main arm.
ALL_ARMS = ("A", "B", "C", "D", "E", "F")
EXPERIMENTAL_ARMS = frozenset({"F"})
SECTIONS_PREFIX = "sections:"
MARKER = ".make_arms"
TRACKSET_FILE = "trackset.sha256"
SUMMARY_FILE = "arms_summary.json"
ACESTEP_SR = 48000
_SUBTYPE_DTYPE = {"PCM_16": "int16", "PCM_24": "int32", "PCM_32": "int32", "FLOAT": "float32", "DOUBLE": "float64"}
_SENTENCE_RE = re.compile(r"^(.+?[.!?])(?=\s|$)")


def _songformer_to_human() -> dict[str, str | None]:
    """SongFormer label -> human (Strauss) label for captions, from ``label_map.COARSE``.

    A label in the human vocabulary with the same coarse class keeps its name (intro,
    verse, chorus, bridge, outro); any other ``body`` label becomes ``verse``, any other
    ``transition`` label becomes ``transition``; ``ignore`` becomes None (dropped).
    """
    human, sf_map = label_map.COARSE["human"], label_map.COARSE["songformer"]
    out: dict[str, str | None] = {}
    for lab, coarse in sf_map.items():
        if coarse == label_map.IGNORE:
            out[lab] = None
        elif lab in HUMAN_VOCAB and human.get(lab) == coarse:
            out[lab] = lab
        elif coarse == "body":
            out[lab] = "verse"
        elif coarse == "transition":
            out[lab] = "transition"
        else:
            raise KeyError(f"no human label for SongFormer {lab!r} (coarse {coarse!r})")
    return out


#: SongFormer label (after ``label_map.normalize_label``) -> human label; None = drop.
SONGFORMER_TO_HUMAN: Mapping[str, str | None] = _songformer_to_human()

#: ID3 typo fixes, applied per word (case-insensitive).
ID3_TYPOS: Mapping[str, str] = {"cleassical": "Classical", "chinematic": "Cinematic", "pops": "Pop"}
#: Junk words removed from inside ID3 tokens (case-insensitive).
ID3_JUNK_WORDS = frozenset({"test!", "markup", "delete", "alfa"})
#: Pre-registered non-genre descriptors: whole tokens dropped (case-insensitive).
ID3_NON_GENRE = frozenset({"mnemonic", "solo", "sound design", "vocals", "pause", "run"})
ID3_MAX_WORDS = 4
#: ``/``, digits next to a dash ("18-71"), or letters run into digits ("UN18"): client codes.
_CLIENT_CODE_RE = re.compile(r"/|\d\s*-|-\s*\d|[A-Za-z]\d")


# --------------------------------------------------------------------------- fields


def _is_missing(x: Any) -> bool:
    """None, NaN, pd.NA, NaT; False for anything array-like or unhashable."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def clean_field(text: Any) -> str:
    """One caption field: no FIELD_SEP, no newlines, single spaces; missing -> ""."""
    if _is_missing(text):
        return ""
    s = str(text).replace(FIELD_SEP, " ")
    return " ".join(s.split())


def _strip_period(s: str) -> str:
    return re.sub(r"\.+$", "", s).rstrip()


def human_caption(row: Mapping[str, Any]) -> str:
    """The human caption as a field: cleaned, trailing period removed."""
    return _strip_period(clean_field(row.get("human_caption")))


def genre_tag(row: Mapping[str, Any]) -> str:
    """The human genre tag, or "" if missing or a junk placeholder ("No Audio", "NA")."""
    tag = clean_field(row.get("human_genre_tag"))
    if not tag or JUNK_RE.search(tag):
        return ""
    return tag


def _raw_id3_tokens(row: Mapping[str, Any]) -> list[str]:
    toks = row.get("id3_genre_tokens")
    if isinstance(toks, str):
        toks = [toks]
    elif not isinstance(toks, (list, tuple, np.ndarray)):
        return []
    return [clean_field(t) for t in toks if clean_field(t)]


def clean_id3_tokens(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
    """(kept tokens, dropped pieces); see the module docstring. Order kept, duplicates
    (case-insensitive, after fixes) removed."""
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        raw = clean_field(raw)
        if not raw:
            continue
        if _CLIENT_CODE_RE.search(raw):
            dropped.append(raw)
            continue
        words = []
        for w in raw.split():
            if w.lower() in ID3_JUNK_WORDS:
                dropped.append(w)
            else:
                words.append(ID3_TYPOS.get(w.lower(), w))
        if not words:
            continue
        if len(words) > ID3_MAX_WORDS:
            dropped.append(raw)
            continue
        tok = " ".join(words)
        if tok.lower() in ID3_NON_GENRE:
            dropped.append(raw)
            continue
        if tok.lower() not in seen:
            seen.add(tok.lower())
            kept.append(tok)
    return kept, dropped


def id3_tags(row: Mapping[str, Any]) -> str:
    """Cleaned ID3 genre tokens, comma-joined ("" if none survive)."""
    return ", ".join(clean_id3_tokens(_raw_id3_tokens(row))[0])


def id3_dropped(row: Mapping[str, Any]) -> list[str]:
    return clean_id3_tokens(_raw_id3_tokens(row))[1]


def bpm_value(row: Mapping[str, Any]) -> int | None:
    b = row.get("id3_bpm")
    if _is_missing(b):
        return None
    b = int(round(float(b)))
    return b if b > 0 else None


def first_sentence(caption: Any) -> str:
    """The caption up to and including the first ``.``, ``!`` or ``?`` that is followed by
    whitespace or the end (the whole caption if there is none), trailing period removed."""
    caption = clean_field(caption)
    m = _SENTENCE_RE.match(caption)
    return _strip_period(m.group(1) if m else caption)


def section_labels(segments: Sequence[Segment]) -> list[str]:
    return [s.label for s in segments if s.label != UNLABELED]


@dataclass
class Caption:
    fields: list[str]
    fallbacks: list[str] = field(default_factory=list)  # counted in n_fallbacks
    omitted: list[str] = field(default_factory=list)  # fields left out because data is missing

    @property
    def text(self) -> str:
        return FIELD_SEP.join(self.fields)

    @property
    def rendered(self) -> str:
        return join_fields(self.fields, JOIN_SEP)

    @property
    def n_words(self) -> int:
        """Whitespace word count of the rendered caption (not tokenizer tokens)."""
        return len(self.rendered.split())


def _human_fields(row: Mapping[str, Any], cap: Caption) -> None:
    """Arm B body: [human caption, genre tag], with the junk-caption fallback."""
    tag = genre_tag(row)
    if bool(row.get("caption_is_junk")):
        if tag:
            cap.fields.append(tag)
            cap.fallbacks.append("junk_caption->genre_tag")
        else:
            tags = id3_tags(row)
            if tags:
                cap.fields.append(tags)
                cap.fallbacks.append("junk_caption->id3_tags")
            else:
                cap.fallbacks.append("junk_caption->trigger_only")
        return
    cap.fields.append(human_caption(row))
    if tag:
        cap.fields.append(tag)
    else:
        cap.omitted.append("genre_tag")


def build_caption(
    arm: str,
    row: Mapping[str, Any],
    sections: Sequence[str] | None,
    sections_source: str = "human",
    autocaption: str | None = None,
) -> Caption:
    """Caption for one track in one arm. ``sections`` are the window's section labels
    (None or empty: no section information)."""
    cap = Caption([TRIGGER])
    if arm == "A":
        pass
    elif arm == "B":
        _human_fields(row, cap)
    elif arm == "C":
        tags = id3_tags(row)
        if tags:
            cap.fields.append(tags)
        else:
            cap.fallbacks.append("no_id3_tags")
        bpm = bpm_value(row)
        if bpm is not None:
            cap.fields.append(f"{bpm} bpm")
        else:
            cap.omitted.append("bpm")
    elif arm == "D":
        text = clean_field(autocaption)
        if not text:
            raise ValueError(f"arm D: no auto-caption for {row.get('track_id')!r}")
        cap.fields.append(text)
    elif arm == "E":
        _human_fields(row, cap)
        if sections:
            cap.fields.append(f"{SECTIONS_PREFIX} {', '.join(sections)}")
            if sections_source != "human":
                cap.fallbacks.append(f"sections_from_{sections_source}")
        else:
            cap.omitted.append("sections")
    elif arm == "F":
        tag = genre_tag(row)
        if not tag:
            tag = id3_tags(row)
            if tag:
                cap.fallbacks.append("genre->id3_tags")
        if tag:
            cap.fields.append(f"Genre: {tag}")
        else:
            cap.omitted.append("genre")
        mood = "" if bool(row.get("caption_is_junk")) else first_sentence(row.get("human_caption"))
        if mood:
            cap.fields.append(f"Mood: {mood}")
        else:
            cap.omitted.append("mood")
        bpm = bpm_value(row)
        if bpm is not None:
            cap.fields.append(f"BPM: {bpm}")
        else:
            cap.omitted.append("bpm")
        if sections:
            cap.fields.append(f"Sections: {', '.join(sections)}")
            if sections_source != "human":
                cap.fallbacks.append(f"sections_from_{sections_source}")
        else:
            cap.omitted.append("sections")
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ALL_ARMS}")
    cap.fields = [f for f in cap.fields if f]
    _validate_caption(cap)
    return cap


def _validate_caption(cap: Caption) -> None:
    cfg = CaptionAugConfig()
    check_trigger(cap.text, cfg)
    for f in cap.fields:
        assert FIELD_SEP not in f and "\n" not in f, f
    tails = [i for i, f in enumerate(cap.fields) if f.lower().startswith(SECTIONS_PREFIX)]
    assert all(i == len(cap.fields) - 1 for i in tails), cap.fields


def acestep_columns(cap: Caption) -> tuple[str, str]:
    """(Caption, lyrics) for ACE-Step: non-section fields joined with ', '; the section
    list as lyric tags ``[intro] [verse] ...``."""
    body, lyrics = [], ""
    for f in cap.fields:
        if f.lower().startswith(SECTIONS_PREFIX):
            labs = [x.strip() for x in f.split(":", 1)[1].split(",") if x.strip()]
            lyrics = " ".join(f"[{x}]" for x in labs)
        else:
            body.append(f)
    return JOIN_SEP.join(body), lyrics


# --------------------------------------------------------------------------- inputs


def load_split_ids(splits_path: Path, splits: Sequence[str]) -> dict[str, list[str]]:
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    out = {}
    for s in splits:
        if s not in data or not isinstance(data[s], list):
            raise KeyError(f"{splits_path}: no split {s!r}")
        out[s] = sorted(str(t) for t in data[s])
    seen: dict[str, str] = {}
    for s, ids in out.items():
        for t in ids:
            if t in seen:
                raise ValueError(f"track {t!r} is in both {seen[t]!r} and {s!r}")
            seen[t] = s
    return out


def load_autocaptions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            tid, cap = rec.get("track_id"), rec.get("caption")
            if not tid or not isinstance(cap, str):
                raise ValueError(f"{path}:{n}: need 'track_id' and string 'caption'")
            out[str(tid)] = cap
    return out


def map_songformer_label(label: str) -> str | None:
    """Human label for a raw SongFormer label (aliases via ``label_map``); None = drop.
    Unknown labels raise KeyError."""
    lab = label_map.normalize_label(label, "songformer")
    if lab not in SONGFORMER_TO_HUMAN:
        raise KeyError(f"unknown SongFormer label {label!r}")
    return SONGFORMER_TO_HUMAN[lab]


def load_songformer_segments(path: Path, duration_s: float | None = None) -> list[Segment]:
    """SongFormer segments mapped to the human vocabulary, ``ignore`` labels dropped, then
    :func:`~ravi.structure.labels.normalize_segments` (sort, snap, clamp, gaps ->
    ``unlabeled``)."""
    doc = schema.load(path)
    segs = []
    for s in doc["segments"]:
        try:
            mapped = map_songformer_label(s["label"])
        except KeyError as e:
            raise KeyError(f"{path}: {e.args[0]}") from None
        if mapped is not None:
            segs.append(Segment(float(s["start"]), float(s["end"]), mapped))
    return normalize_segments(segs, duration_s, Path(path).name)


@dataclass
class TrackPlan:
    track_id: str
    split: str
    row: dict
    wav: Path
    sr: int
    n_frames_total: int
    subtype: str
    segments_source: str  # human | songformer | none
    window: Window
    start_frame: int
    n_frames: int

    @property
    def cropped(self) -> bool:
        return self.window.cropped

    @property
    def window_start(self) -> float:
        return self.start_frame / self.sr

    @property
    def window_end(self) -> float:
        return (self.start_frame + self.n_frames) / self.sr


def plan_track(
    row: Mapping[str, Any],
    split: str,
    prepared_root: Path,
    sr: int,
    labels_dir: Path | None,
    songformer_dir: Path | None,
    duration_s: float = TRAIN_DURATION_S,
    downsample: int = DOWNSAMPLE,
) -> TrackPlan:
    tid = str(row["track_id"])
    wav = Path(prepared_root) / str(sr) / f"{tid}.wav"
    if not wav.is_file():
        raise FileNotFoundError(f"prepared WAV missing: {wav}")
    info = sf.info(str(wav))
    if info.samplerate != sr:
        raise ValueError(f"{wav}: sample rate {info.samplerate} != --sr {sr}")
    duration = info.frames / sr

    segments: list[Segment] = []
    source = "none"
    if bool(row.get("has_labels")):
        lp = row.get("labels_path")
        if _is_missing(lp) or not lp:
            raise ValueError(f"{tid}: has_labels but no labels_path")
        lp = Path(labels_dir) / Path(str(lp)).name if labels_dir else Path(str(lp))
        if not lp.is_file():
            raise FileNotFoundError(f"{tid}: label file missing: {lp}")
        segments = parse_audacity_labels(lp, duration)
        source = "human"
    elif songformer_dir is not None:
        sp = Path(songformer_dir) / f"{tid}.json"
        if sp.is_file():
            segments = load_songformer_segments(sp, duration)
            source = "songformer"

    n = window_frames(sr, duration_s, downsample)
    win = choose_window(segments, duration, n / sr)
    if win.cropped:
        start_frame = min(int(round(win.start * sr)), info.frames - n)
        if start_frame < 0 or n > info.frames:
            raise ValueError(f"{tid}: cannot crop {n} frames from {info.frames}")
    else:
        start_frame, n = 0, info.frames
    return TrackPlan(tid, split, dict(row), wav, sr, info.frames, info.subtype, source, win, start_frame, n)


def trackset_hash(plans: Sequence[TrackPlan], sr: int) -> str:
    """SHA-256 over the sample rate and every (split, track id, start frame, frame count)."""
    payload = json.dumps(
        {"sr": int(sr), "tracks": [[p.split, p.track_id, p.start_frame, p.n_frames] for p in plans]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- outputs


def _place_audio(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        # relative, so the link survives the {arm}.tmp -> {arm} rename and a moved tree
        os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            log.info("hardlink across file systems for %s; copying", src)
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown link mode {mode!r}")


def read_crop(tp: TrackPlan) -> np.ndarray:
    dtype = _SUBTYPE_DTYPE.get(tp.subtype, "float32")
    data, _ = sf.read(str(tp.wav), start=tp.start_frame, frames=tp.n_frames, dtype=dtype, always_2d=True)
    if len(data) != tp.n_frames:
        raise RuntimeError(f"{tp.wav}: read {len(data)} frames, expected {tp.n_frames}")
    return data


def write_crop(tp: TrackPlan, data: np.ndarray, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    sf.write(str(dst), data, tp.sr, subtype=tp.subtype if tp.subtype in _SUBTYPE_DTYPE else None)
    meta = {
        "track_id": tp.track_id,
        "source_wav": str(tp.wav),
        "sr": tp.sr,
        "window_start": tp.window_start,
        "window_end": tp.window_end,
        "start_frame": tp.start_frame,
        "n_frames": tp.n_frames,
        "source_frames": tp.n_frames_total,
        "segments_source": tp.segments_source,
        "segments": [  # relative to the cropped audio
            {"start": s.start - tp.window_start, "end": s.end - tp.window_start, "label": s.label}
            for s in tp.window.segments
        ],
    }
    dst.with_name(f"{tp.track_id}.window.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _is_ours(d: Path) -> bool:
    return (d / MARKER).is_file()


def _remove_ours(d: Path) -> None:
    """Delete ``d`` if it is a make_arms directory (marker present); refuse otherwise."""
    if not (d.exists() or d.is_symlink()):
        return
    if d.is_symlink() or not d.is_dir() or not _is_ours(d):
        raise FileExistsError(f"{d} has no {MARKER} marker; refusing to delete a directory make_arms did not write")
    shutil.rmtree(d)


def _check_targets(out_root: Path, arms: Sequence[str], overwrite: bool, digest: str) -> None:
    """Fail before writing anything if an arm dir may not be replaced, or if a sibling arm
    was built on a different track set."""
    for arm in arms:
        d = out_root / arm
        if d.exists() or d.is_symlink():
            if d.is_dir() and not d.is_symlink() and not any(d.iterdir()):
                continue
            if not overwrite:
                raise FileExistsError(f"{d} exists and is not empty; pass --overwrite to rebuild it")
            if d.is_symlink() or not d.is_dir() or not _is_ours(d):
                raise FileExistsError(f"{d} has no {MARKER} marker; refusing to replace a directory make_arms did not write")
        for suffix in (".tmp", ".old"):
            side = out_root / f"{arm}{suffix}"
            if (side.exists() or side.is_symlink()) and not (side.is_dir() and _is_ours(side)):
                raise FileExistsError(f"{side} exists and has no {MARKER} marker; remove it by hand")
    if not out_root.is_dir():
        return
    for sib in sorted(out_root.iterdir()):
        if sib.name in arms or not sib.is_dir() or sib.suffix in (".tmp", ".old") or not _is_ours(sib):
            continue
        h = sib / TRACKSET_FILE
        if not h.is_file():
            continue
        other = h.read_text(encoding="utf-8").strip()
        if other != digest:
            msg = (f"existing arm {sib.name} was built on a different track set/windows "
                   f"({other[:12]} != {digest[:12]}): arms would not be comparable")
            if not overwrite:
                raise ValueError(msg + "; rebuild it too, or pass --overwrite")
            log.warning("%s (--overwrite given)", msg)


def _verify_audio(arm_dir: Path) -> None:
    """Every .wav must resolve to a regular file (catches dangling or looping symlinks)."""
    for p in arm_dir.rglob("*.wav"):
        try:
            st = os.stat(p)  # follows symlinks
        except OSError as e:
            raise RuntimeError(f"{p} does not resolve to a file: {e}") from None
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(f"{p} does not resolve to a regular file")


def _swap_in(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    _remove_ours(old)
    had_old = final.exists()
    if had_old:
        if any(final.iterdir()):
            os.rename(final, old)
        else:
            final.rmdir()
            had_old = False
    try:
        os.rename(tmp, final)
    except OSError:
        if had_old:
            os.rename(old, final)
        raise
    if had_old:
        shutil.rmtree(old)


def _summ(xs: Sequence[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None}
    return {"mean": round(statistics.fmean(xs), 3), "median": float(statistics.median(xs))}


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


MANIFEST_COLS = ["track_id", "split", "window_start", "window_end", "start_frame", "n_frames", "caption",
                 "caption_words", "file", "cropped", "segments_source", "fallbacks", "omitted", "id3_dropped"]


def _build_arm_dir(
    d: Path,
    plans: Sequence[TrackPlan],
    caps: Sequence[Caption],
    splits: Sequence[str],
    digest: str,
    sr: int,
) -> list[dict]:
    """Write the arm's metadata into ``d`` (audio and captions are already there)."""
    recs = []
    for tp, cap in zip(plans, caps):
        recs.append({
            "track_id": tp.track_id,
            "split": tp.split,
            "window_start": round(tp.window_start, 6),
            "window_end": round(tp.window_end, 6),
            "start_frame": tp.start_frame,
            "n_frames": tp.n_frames,
            "caption": cap.text,
            "caption_words": cap.n_words,
            "file": f"{tp.split}/{tp.track_id}.wav",
            "cropped": tp.cropped,
            "segments_source": tp.segments_source,
            "fallbacks": ";".join(cap.fallbacks),
            "omitted": ";".join(cap.omitted),
            "id3_dropped": ";".join(id3_dropped(tp.row)),
        })
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(recs)
    for s in splits:
        names = [f"{tp.track_id}.wav" for tp in plans if tp.split == s]
        (d / s / "filelist.txt").write_text("".join(n + "\n" for n in names), encoding="utf-8")
    if sr == ACESTEP_SR:
        with open(d / "acestep_metadata.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["File", "BPM", "Key", "Caption", "lyrics"])
            for tp, cap, r in zip(plans, caps, recs):
                c, lyr = acestep_columns(cap)
                bpm = bpm_value(tp.row)
                w.writerow([r["file"], "" if bpm is None else bpm, "", c, lyr])
    (d / TRACKSET_FILE).write_text(digest + "\n", encoding="utf-8")
    return recs


def make_arms(
    manifest: Path,
    splits_path: Path,
    prepared_root: Path,
    out_root: Path,
    arms: Sequence[str],
    splits: Sequence[str] = ("train", "val"),
    sr: int = 44100,
    labels_dir: Path | None = None,
    autocaptions: Path | None = None,
    songformer_dir: Path | None = None,
    link_mode: str = "copy",
    limit: int | None = None,
    overwrite: bool = False,
    duration_s: float = TRAIN_DURATION_S,
    downsample: int = DOWNSAMPLE,
) -> dict:
    """Build the arm directories; returns the (merged) ``arms_summary.json`` content."""
    arms = [a.strip().upper() for a in arms if a.strip()]
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad or not arms:
        raise ValueError(f"unknown arms {bad}; choose from {ALL_ARMS}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"duplicate arms in {arms}")
    if link_mode not in ("symlink", "hardlink", "copy"):
        raise ValueError(f"unknown link mode {link_mode!r}")
    for a in arms:
        if a in EXPERIMENTAL_ARMS:
            log.warning("arm %s is experimental (not a pre-registered main arm)", a)
    auto = None
    if "D" in arms:
        if autocaptions is None:
            raise ValueError("arm D needs --autocaptions")
        auto = load_autocaptions(Path(autocaptions))

    # ---------------------------------------------------------------- plan (no writes)
    df = pd.read_parquet(manifest)
    if df.track_id.duplicated().any():
        raise ValueError(f"{manifest}: duplicate track_id")
    rows = {str(r["track_id"]): r for r in df.to_dict("records")}
    split_ids = load_split_ids(Path(splits_path), splits)
    missing = [t for ids in split_ids.values() for t in ids if t not in rows]
    if missing:
        raise KeyError(f"{len(missing)} split tracks not in manifest, e.g. {missing[:3]}")

    # Deterministic order: manifest order, restricted to the selected splits.
    split_of = {t: s for s, ids in split_ids.items() for t in ids}
    selected = [t for t in df.track_id.astype(str) if t in split_of]
    if limit is not None:
        selected = selected[: int(limit)]
    if auto is not None:
        miss = [t for t in selected if not clean_field(auto.get(t))]
        if miss:
            raise KeyError(f"arm D: {len(miss)} tracks have no auto-caption in {autocaptions}, e.g. {miss[:3]}")
    for t in selected:
        if FIELD_SEP in t or "/" in t or t.startswith(".") or t == "filelist":
            raise ValueError(f"track_id {t!r} cannot be used as a file name")

    plans = [plan_track(rows[t], split_of[t], prepared_root, sr, labels_dir, songformer_dir, duration_s, downsample)
             for t in selected]
    caps: dict[str, list[Caption]] = {
        arm: [build_caption(arm, tp.row, section_labels(tp.window.segments), tp.segments_source,
                            auto.get(tp.track_id) if auto else None) for tp in plans]
        for arm in arms
    }
    digest = trackset_hash(plans, sr)
    out_root = Path(out_root)
    _check_targets(out_root, arms, overwrite, digest)

    # ---------------------------------------------------------------- build {arm}.tmp
    out_root.mkdir(parents=True, exist_ok=True)
    tmps = {arm: out_root / f"{arm}.tmp" for arm in arms}
    records: dict[str, list[dict]] = {}
    try:
        for arm, d in tmps.items():
            _remove_ours(d)
            d.mkdir()
            (d / MARKER).write_text("written by ravi.captions.make_arms\n", encoding="utf-8")
            for s in splits:
                (d / s).mkdir()
        for i, tp in enumerate(plans):
            crop = read_crop(tp) if tp.cropped else None
            for arm, d in tmps.items():
                wav_dst = d / tp.split / f"{tp.track_id}.wav"
                if crop is not None:
                    write_crop(tp, crop, wav_dst)
                else:
                    _place_audio(tp.wav, wav_dst, link_mode)
                wav_dst.with_name(f"{tp.track_id}.txt").write_text(caps[arm][i].text + "\n", encoding="utf-8")
        for arm, d in tmps.items():
            records[arm] = _build_arm_dir(d, plans, caps[arm], splits, digest, sr)
            _verify_audio(d)
            for p in d.rglob("*"):
                assert FIELD_SEP not in p.name, p
    except BaseException:
        for d in tmps.values():
            if d.is_dir() and _is_ours(d):
                shutil.rmtree(d, ignore_errors=True)
        raise

    # ---------------------------------------------------------------- swap in
    for arm, d in tmps.items():
        _swap_in(d, out_root / arm)
    if sr != ACESTEP_SR:
        log.info("acestep_metadata.csv not written: needs --sr %d (got %d)", ACESTEP_SR, sr)

    # ---------------------------------------------------------------- summary (merged)
    n_frames = window_frames(sr, duration_s, downsample)
    run_info: dict[str, Any] = {
        "sr": sr,
        "duration_s": duration_s,
        "downsample": downsample,
        "window_frames": n_frames,
        "window_s": n_frames / sr,
        "splits": list(splits),
        "limit": limit,
        "link_mode": link_mode,
        "n_tracks": len(plans),
        "segments_source": dict(sorted(Counter(tp.segments_source for tp in plans).items())),
        "trackset_sha256": digest,
        "acestep_metadata": sr == ACESTEP_SR,
    }
    summary_path = out_root / SUMMARY_FILE
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prev_arms = summary.get("arms", {}) if isinstance(summary.get("arms"), dict) else {}
    summary.pop("win_s", None)  # pre-window_frames summaries
    summary.update(run_info)
    summary["arms"] = dict(prev_arms)
    for arm in arms:
        recs, cs = records[arm], caps[arm]
        fb = Counter(f for c in cs for f in c.fallbacks)
        om = Counter(f for c in cs for f in c.omitted)
        summary["arms"][arm] = {
            "experimental": arm in EXPERIMENTAL_ARMS,
            "n_tracks": {s: sum(r["split"] == s for r in recs) for s in splits},
            "caption_words": _summ([c.n_words for c in cs]),
            "n_fallbacks": sum(1 for c in cs if c.fallbacks),
            "fallbacks": dict(sorted(fb.items())),
            "omitted_fields": dict(sorted(om.items())),
            "n_cropped": sum(1 for tp in plans if tp.cropped),
            "n_id3_dropped": sum(1 for r in recs if r["id3_dropped"]),
            **{k: run_info[k] for k in ("sr", "window_frames", "limit", "link_mode", "trackset_sha256")},
        }
        if fb or om:
            log.info("arm %s: fallbacks %s, omitted fields %s", arm, dict(fb), dict(om))
    summary["arms"] = dict(sorted(summary["arms"].items()))
    _write_json_atomic(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=Path("manifests/tracks.parquet"))
    ap.add_argument("--splits", type=Path, default=Path("manifests/splits.json"))
    ap.add_argument("--labels-dir", type=Path, default=None,
                    help="directory holding the Audacity label files (default: labels_path from the manifest)")
    ap.add_argument("--prepared-root", type=Path, default=Path("prepared"))
    ap.add_argument("--sr", type=int, default=44100,
                    help="sample rate of prepared/{sr}; acestep_metadata.csv only with 48000")
    ap.add_argument("--out-root", type=Path, default=Path("data/arms"))
    ap.add_argument("--arms", default="A,B,C,E", help=f"comma list from {','.join(ALL_ARMS)} (F is experimental)")
    ap.add_argument("--autocaptions", type=Path, default=None, help="JSONL {track_id, caption}; required for arm D")
    ap.add_argument("--songformer-dir", type=Path, default=None,
                    help="structure/songformer/{track_id}.json, used for tracks without human labels")
    ap.add_argument("--split", default="train,val")
    ap.add_argument("--link", choices=("symlink", "hardlink", "copy"), default="copy",
                    help="how uncropped WAVs are placed in the arm dirs (symlinks are relative)")
    ap.add_argument("--copy", action="store_true", help="shorthand for --link copy")
    ap.add_argument("--limit", type=int, default=None, help="first N selected tracks (manifest order)")
    ap.add_argument("--duration", type=float, default=TRAIN_DURATION_S, help="trainer --duration, seconds")
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE, help="autoencoder downsampling ratio")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing arm dirs (only ones carrying the .make_arms marker)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)
    summary = make_arms(
        manifest=a.manifest,
        splits_path=a.splits,
        prepared_root=a.prepared_root,
        out_root=a.out_root,
        arms=a.arms.split(","),
        splits=[s.strip() for s in a.split.split(",") if s.strip()],
        sr=a.sr,
        labels_dir=a.labels_dir.expanduser() if a.labels_dir else None,
        autocaptions=a.autocaptions,
        songformer_dir=a.songformer_dir,
        link_mode="copy" if a.copy else a.link,
        limit=a.limit,
        overwrite=a.overwrite,
        duration_s=a.duration,
        downsample=a.downsample,
    )
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
