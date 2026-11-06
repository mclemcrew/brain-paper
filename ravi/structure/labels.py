"""Parse and normalise Jessica Strauss's Audacity section labels.

Audacity exports one label per line as ``start<TAB>end<TAB>label`` (seconds).
The raw files contain typos, trailing spaces, a corrupt binary line, micro
overlaps/gaps between neighbouring labels, and labels that stop short of the
end of the audio. :func:`parse_audacity_labels` turns a file into a sorted,
non-overlapping list of :class:`Segment` objects.

Normalisation rules (in order):

1. Decode as UTF-8 with replacement; skip blank lines and Audacity
   spectral-selection lines (those starting with ``\\``).
2. Label -> ``lower().strip()`` then the typo map :data:`TYPO_MAP`.
3. Drop (and log) lines with a label outside :data:`VOCAB`, unparsable or
   non-finite or negative times, ``start == end`` or ``end < start``.
4. Sort by start time.
5. Snap: if a segment starts within :data:`SNAP_TOL` of the previous end
   (micro gap or micro overlap) its start is set to the previous end. Larger
   overlaps are also cut back to the previous end (logged); a segment that
   becomes empty is dropped. Larger interior gaps are filled with an
   ``unlabeled`` segment.
6. Clamp ends to ``duration_s``; drop segments that start at/after it.
7. If the first start is within :data:`SNAP_TOL` of 0 it is snapped to 0;
   if it is later, an ``unlabeled`` segment is prepended.
8. If the last end is more than :data:`TAIL_TOL` short of ``duration_s`` an
   ``unlabeled`` segment is appended.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

VOCAB = frozenset({"intro", "verse", "chorus", "bridge", "transition", "outro"})
UNLABELED = "unlabeled"
INTERIOR_TARGETS = frozenset({"verse", "bridge", "chorus"})
TYPO_MAP = {
    "iverse": "verse",
    "transiton": "transition",
    "brdige": "bridge",
    "brridge": "bridge",
}
SNAP_TOL = 0.5  # seconds
TAIL_TOL = 2.0  # seconds


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def normalize_label(raw: str) -> str:
    lab = raw.strip().lower()
    return TYPO_MAP.get(lab, lab)


def _parse_lines(text: str, src: str, issues: list[str] | None) -> list[Segment]:
    def drop(lineno: int, why: str, line: str) -> None:
        msg = f"{src}:{lineno}: dropped ({why}): {line[:80]!r}"
        log.info(msg)
        if issues is not None:
            issues.append(msg)

    segs: list[Segment] = []
    # Split on "\n" only: str.splitlines() also breaks on \x0c, \x1c etc., which
    # would turn one corrupt binary line into several.
    for lineno, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("\\"):  # Audacity spectral-selection continuation line
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            drop(lineno, "fewer than 3 fields", line)
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            drop(lineno, "unparsable time", line)
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < 0:
            drop(lineno, "non-finite or negative time", line)
            continue
        label = normalize_label("\t".join(parts[2:]))
        if label not in VOCAB:
            drop(lineno, f"label {label!r} not in vocab", line)
            continue
        if end == start:
            drop(lineno, "zero length", line)
            continue
        if end < start:
            drop(lineno, "end before start", line)
            continue
        segs.append(Segment(start, end, label))
    return segs


def normalize_segments(
    segs: Iterable[Segment],
    duration_s: float | None,
    src: str = "<segments>",
    issues: list[str] | None = None,
) -> list[Segment]:
    """Apply steps 4-8 of the module docstring to already-parsed segments."""

    def note(msg: str) -> None:
        log.info("%s: %s", src, msg)
        if issues is not None:
            issues.append(f"{src}: {msg}")

    ordered = sorted(segs, key=lambda s: (s.start, s.end))
    out: list[Segment] = []
    for seg in ordered:
        if out:
            prev_end = out[-1].end
            delta = seg.start - prev_end  # >0 gap, <0 overlap
            if abs(delta) < SNAP_TOL:
                seg = Segment(prev_end, seg.end, seg.label)
            elif delta < 0:
                note(f"overlap of {-delta:.3f}s at {seg.start:.3f} cut back to {prev_end:.3f}")
                seg = Segment(prev_end, seg.end, seg.label)
            else:
                note(f"interior gap of {delta:.3f}s at {prev_end:.3f} filled as {UNLABELED}")
                out.append(Segment(prev_end, seg.start, UNLABELED))
            if seg.end <= seg.start:
                note(f"segment {seg.label!r} empty after snapping; dropped")
                continue
        out.append(seg)

    if duration_s is not None and math.isfinite(duration_s):
        clamped: list[Segment] = []
        for seg in out:
            if seg.start >= duration_s:
                note(f"segment {seg.label!r} starts at {seg.start:.3f} >= duration {duration_s:.3f}; dropped")
                continue
            if seg.end > duration_s:
                seg = Segment(seg.start, duration_s, seg.label)
            clamped.append(seg)
        out = clamped

    if out:
        first = out[0]
        if 0 < first.start < SNAP_TOL:
            out[0] = Segment(0.0, first.end, first.label)
        elif first.start >= SNAP_TOL:
            out.insert(0, Segment(0.0, first.start, UNLABELED))

    if duration_s is not None and math.isfinite(duration_s):
        last_end = out[-1].end if out else 0.0
        if duration_s - last_end > TAIL_TOL:
            out.append(Segment(last_end, float(duration_s), UNLABELED))
    return out


def parse_audacity_labels(
    path: str | Path,
    duration_s: float | None,
    issues: list[str] | None = None,
) -> list[Segment]:
    """Parse an Audacity label export into normalised segments.

    ``duration_s`` is the audio duration used for clamping and tail filling;
    pass ``None`` to skip both. Dropped lines and repairs are logged at INFO
    and, if ``issues`` is given, appended to it as strings.
    """
    p = Path(path)
    text = p.read_bytes().decode("utf-8", errors="replace")
    return normalize_segments(_parse_lines(text, p.name, issues), duration_s, p.name, issues)


def labeled_segments(segments: Sequence[Segment]) -> list[Segment]:
    return [s for s in segments if s.label != UNLABELED]


def interior_sections(segments: Sequence[Segment]) -> list[Segment]:
    """Segments with audio context on both sides (not first/last in the list)
    whose label is verse, bridge or chorus: candidate inpainting targets."""
    return [s for s in segments[1:-1] if s.label in INTERIOR_TARGETS]


def write_segments_json(path: str | Path, segments: Sequence[Segment], **meta) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "segments": [asdict(s) for s in segments]}
    p.write_text(json.dumps(payload, indent=2) + "\n")


def read_segments_json(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    return [Segment(float(s["start"]), float(s["end"]), str(s["label"])) for s in data["segments"]]
