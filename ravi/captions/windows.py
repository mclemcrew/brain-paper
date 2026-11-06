"""Fixed-length training window per track (plan Phase 1 step 8).

Every arm trains on the same window, exactly the trainer's ``sample_size``::

    n_frames = (int(duration_s * sr) // downsample) * downsample   # train_lora.py:105
    win_s    = n_frames / sr

with ``duration_s = 47`` (``--duration 47``) and ``downsample = 4096`` (the SA3
autoencoder's downsampling ratio), the same formula as
:func:`ravi.train.sa3_lora.scan_dataset`. At 44.1 kHz that is 2,072,576 frames
(46.99719 s); at 48 kHz 2,252,800 frames (46.93333 s). :func:`window_frames` and
:func:`window_seconds` compute it; :data:`WIN_S` is the 44.1 kHz value.

Tracks no longer than the window are used whole (the trainer pads them). Longer
tracks are cropped *here*, before training, because upstream ``train_lora.py`` would
otherwise random-crop them and arm E's section summary would no longer describe the
audio the model hears.

Window choice for a long track (deterministic):

* Candidate starts: 0, every label boundary ``b`` with ``b + win_s <= duration``,
  and every ``b - win_s`` with ``b - win_s >= 0`` (window *ending* on a boundary).
* Cost of a start ``s``: distance from ``s`` to the nearest label boundary plus the
  distance from ``s + win_s`` to the nearest label boundary. Boundaries are every
  segment start/end, plus 0 and the track end.
* Start 0 is preferred: a later start wins only if it is *clean*, i.e. its cost is
  at most ``align_tol_s`` (1.0 s: both edges within about a second of a boundary) and
  lower than the cost of starting at 0. Among clean candidates the lowest cost wins,
  ties going to the earlier start.

With no labels the window is ``[0, win_s]``. In practice this means "start at 0 and
cut at ``win_s``" unless shifting the window lands *both* edges on boundaries.

The returned segments are the label segments clipped to the window (absolute times,
in order). A segment that a window edge cuts is kept only if at least
``min_overlap_s`` (1.0 s) of it remains inside, so a sliver of the next section at
the cut is not reported in the caption. Segments that lie entirely inside the window
are always kept, however short (a 0.6 s transition the model does hear).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ravi.structure.labels import Segment

#: Training clip length passed to the trainer (``--duration``), seconds.
TRAIN_DURATION_S = 47.0
#: SA3 autoencoder downsampling ratio; the trainer floors the clip to a multiple of it.
DOWNSAMPLE = 4096
MIN_OVERLAP_S = 1.0
ALIGN_TOL_S = 1.0
_EPS = 1e-9


def window_frames(sr: int, duration_s: float = TRAIN_DURATION_S, downsample: int = DOWNSAMPLE) -> int:
    """The trainer's ``sample_size`` at ``sr``: ``(int(duration_s * sr) // downsample) * downsample``."""
    n = (int(duration_s * sr) // int(downsample)) * int(downsample)
    if n <= 0:
        raise ValueError(f"window of {duration_s} s at {sr} Hz with downsample {downsample} is empty")
    return n


def window_seconds(sr: int, duration_s: float = TRAIN_DURATION_S, downsample: int = DOWNSAMPLE) -> float:
    """``window_frames(...) / sr``."""
    return window_frames(sr, duration_s, downsample) / sr


#: Window length at 44.1 kHz: 2,072,576 / 44,100 = 46.99719 s.
WIN_S = window_seconds(44100)


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    segments: tuple[Segment, ...]  # clipped to [start, end], absolute times
    cropped: bool  # True if the track is longer than the window

    @property
    def duration(self) -> float:
        return self.end - self.start

    def relative_segments(self) -> list[Segment]:
        """Segments with times relative to the window start (i.e. to the cropped audio)."""
        return [Segment(s.start - self.start, s.end - self.start, s.label) for s in self.segments]


def _boundaries(segments: Sequence[Segment], duration_s: float) -> list[float]:
    pts = {0.0, float(duration_s)}
    for s in segments:
        pts.add(float(s.start))
        pts.add(float(s.end))
    return sorted(p for p in pts if -_EPS <= p <= duration_s + _EPS)


def _dist(x: float, pts: Sequence[float]) -> float:
    return min(abs(x - p) for p in pts)


def clip_segments(
    segments: Sequence[Segment], start: float, end: float, min_overlap_s: float = MIN_OVERLAP_S
) -> list[Segment]:
    """Segments clipped to ``[start, end]``. ``min_overlap_s`` applies only to segments a
    window edge cuts; segments entirely inside the window are kept whatever their length."""
    out = []
    for s in sorted(segments, key=lambda s: (s.start, s.end)):
        a, b = max(s.start, start), min(s.end, end)
        if b <= a:
            continue
        contained = s.start >= start - _EPS and s.end <= end + _EPS
        if contained or b - a >= min_overlap_s - _EPS:
            out.append(Segment(a, b, s.label))
    return out


def choose_window(
    segments: Sequence[Segment],
    duration_s: float,
    win_s: float = WIN_S,
    min_overlap_s: float = MIN_OVERLAP_S,
    align_tol_s: float = ALIGN_TOL_S,
) -> Window:
    """Pick the training window for one track; see the module docstring."""
    duration_s = float(duration_s)
    if not duration_s > 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if duration_s <= win_s:
        return Window(0.0, duration_s, tuple(clip_segments(segments, 0.0, duration_s, min_overlap_s)), False)

    pts = _boundaries(segments, duration_s)
    latest = duration_s - win_s
    cands = {0.0}
    for b in pts:
        if b <= latest + _EPS:
            cands.add(min(max(b, 0.0), latest))
        if b - win_s >= -_EPS:
            cands.add(min(max(b - win_s, 0.0), latest))

    def cost(s: float) -> float:
        return _dist(s, pts) + _dist(s + win_s, pts)

    best, best_cost = 0.0, cost(0.0)
    for s in sorted(cands):
        if s == 0.0:
            continue
        c = cost(s)
        if c <= align_tol_s + _EPS and c < best_cost - _EPS:
            best, best_cost = s, c
    start, end = best, best + win_s
    return Window(start, end, tuple(clip_segments(segments, start, end, min_overlap_s)), True)
