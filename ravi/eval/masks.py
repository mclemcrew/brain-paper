"""Inpainting masks over song sections, the splice that restores audio outside a mask,
and the anchors used by the Phase 2.3 listening test.

Times are seconds, sample arrays are float32 ``(n_samples, channels)`` (the
:mod:`soundfile` layout). Pure numpy: no torch, so the eval code and the tests can use it
without a GPU image.

* :func:`section_masks`: one mask per *interior* verse/bridge/chorus section of a
  :mod:`ravi.structure.schema` segmentation, padded by ``pad_s`` on both sides.
* :func:`splice`: copy the source everywhere outside ``[start, end)`` and the fill inside,
  with raised-cosine crossfades that lie *inside* the mask, so every sample outside the
  mask is bit-identical to the source.
* :func:`upper_anchor`: the original region itself (identity: what a perfect inpainting
  would give back).
* :func:`lower_anchor`: a region of the same length cut from a *different* track (the
  plausible-but-wrong reference).

Interior: a section is interior when it is neither the first nor the last *labelled*
segment. ``unlabeled`` filler (a 1 s head the annotator skipped, an early stop) does not
count, so a verse that starts right after a skipped 1 s head is not interior, unlike
:func:`ravi.structure.labels.interior_sections`, which only drops list positions 0 and -1.
The returned ``index`` is the position in the *full* segment list (the JSON's
``segments`` array), so it can be passed back as ``--section-index``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from ravi.structure.labels import INTERIOR_TARGETS, UNLABELED, normalize_label

__all__ = [
    "SectionMask",
    "section_masks",
    "seconds_to_samples",
    "crossfade_ramp",
    "splice",
    "upper_anchor",
    "lower_anchor",
    "mask_summary",
]

DEFAULT_PAD_S = 1.0
DEFAULT_MIN_LEN_S = 4.0
DEFAULT_XFADE_S = 0.050


class SectionMask(NamedTuple):
    """A padded inpainting region; unpacks as ``(start, end, label, index)``."""

    start: float
    end: float
    label: str
    index: int


def _seg(seg: Any) -> tuple[float, float, str]:
    if isinstance(seg, Mapping):
        return float(seg["start"]), float(seg["end"]), str(seg["label"])
    if hasattr(seg, "start") and hasattr(seg, "end") and hasattr(seg, "label"):
        return float(seg.start), float(seg.end), str(seg.label)
    start, end, label = seg
    return float(start), float(end), str(label)


def section_masks(
    segments: Iterable[Any],
    duration: float | None,
    pad_s: float = DEFAULT_PAD_S,
    min_len_s: float = DEFAULT_MIN_LEN_S,
    labels: Iterable[str] = INTERIOR_TARGETS,
) -> list[SectionMask]:
    """Masks for the interior verse/bridge/chorus sections, in segment order.

    ``segments``: schema dicts, :class:`ravi.structure.labels.Segment` objects or
    ``(start, end, label)`` tuples, sorted by start (as :mod:`ravi.structure.schema`
    guarantees). Labels go through :func:`ravi.structure.labels.normalize_label` (case,
    whitespace, typo map). A section shorter than ``min_len_s`` (before padding) is
    skipped. Each kept section becomes ``[start - pad_s, end + pad_s]`` clamped to
    ``[0, duration]`` (no upper clamp if ``duration`` is None).
    """
    if pad_s < 0 or min_len_s < 0:
        raise ValueError("pad_s and min_len_s must be >= 0")
    targets = {normalize_label(x) for x in labels}
    segs = [_seg(s) for s in segments]
    labelled = [i for i, (_, _, lab) in enumerate(segs) if normalize_label(lab) != UNLABELED]
    if len(labelled) < 3:
        return []
    upper = math.inf if duration is None else float(duration)
    out: list[SectionMask] = []
    for i in labelled[1:-1]:
        start, end, lab = segs[i]
        lab = normalize_label(lab)
        if lab not in targets or end - start < min_len_s:
            continue
        m0, m1 = max(0.0, start - pad_s), min(upper, end + pad_s)
        if m1 <= m0:
            continue
        out.append(SectionMask(m0, m1, lab, i))
    return out


def seconds_to_samples(t: float, sr: int) -> int:
    """Nearest sample index (``round``, not upstream's ``int`` truncation)."""
    return int(round(float(t) * sr))


def crossfade_ramp(n: int) -> np.ndarray:
    """Raised-cosine ramp of ``n`` samples rising from just above 0 to just below 1.

    Sample ``k`` has weight ``0.5 - 0.5*cos(pi*(k+0.5)/n)``, so the ramp is symmetric,
    ``ramp + ramp[::-1] == 1`` and neither end is exactly 0 or 1.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(np.pi * (k + 0.5) / n)).astype(np.float32)


def splice(
    source: np.ndarray,
    fill: np.ndarray,
    start: int,
    end: int,
    xfade: int,
) -> np.ndarray:
    """``source`` with ``[start, end)`` replaced by ``fill`` (length ``end - start``).

    The fade from source into fill covers ``[start, start + xfade)`` and the fade back
    covers ``[end - xfade, end)``, both inside the mask, so ``out[:start]`` and
    ``out[end:]`` are exact copies of ``source``. ``xfade`` is reduced to half the mask
    length for a very short mask. At ``end == len(source)`` (continuation) there is no
    second seam and no fade-out.
    """
    source = np.asarray(source)
    fill = np.asarray(fill, dtype=source.dtype)
    if source.ndim != 2 or fill.ndim != 2 or fill.shape[1] != source.shape[1]:
        raise ValueError(f"expected (n, channels) arrays, got {source.shape} and {fill.shape}")
    if not 0 <= start < end <= source.shape[0]:
        raise ValueError(f"bad mask [{start}, {end}) for {source.shape[0]} samples")
    if fill.shape[0] != end - start:
        raise ValueError(f"fill has {fill.shape[0]} samples, mask has {end - start}")
    out = source.copy()
    region = fill.copy()
    to_end = end == source.shape[0]
    xf = max(0, min(int(xfade), (end - start) if to_end else (end - start) // 2))
    if xf:
        w = crossfade_ramp(xf)[:, None]
        src_in = source[start : start + xf]
        region[:xf] = (1.0 - w) * src_in + w * region[:xf]
        if not to_end:
            src_out = source[end - xf : end]
            region[-xf:] = w[::-1] * region[-xf:] + (1.0 - w[::-1]) * src_out
    out[start:end] = region
    return out


def upper_anchor(source: np.ndarray, start: int, end: int) -> np.ndarray:
    """The original ``source[start:end]``: identity, the best any inpainting can do."""
    return np.array(source[start:end], copy=True)


def lower_anchor(other_track_audio: np.ndarray, length: int, offset: int | None = None) -> np.ndarray:
    """``length`` samples of a different track, to splice in where the section was.

    ``offset`` defaults to the middle of the other track (``(n - length) // 2``), which
    for a song usually lands in a body section rather than an intro or a fade. A track
    shorter than ``length`` is tiled. Mono is returned as ``(length, 1)``.
    """
    a = np.asarray(other_track_audio)
    if a.ndim == 1:
        a = a[:, None]
    n = a.shape[0]
    if length <= 0:
        raise ValueError("length must be > 0")
    if n == 0:
        raise ValueError("other track is empty")
    if n < length:
        return np.array(np.tile(a, (math.ceil(length / n), 1))[:length], copy=True)
    off = (n - length) // 2 if offset is None else int(offset)
    if not 0 <= off <= n - length:
        raise ValueError(f"offset {off} out of range for {n} samples and length {length}")
    return np.array(a[off : off + length], copy=True)


def mask_summary(masks: Sequence[SectionMask]) -> str:
    return ", ".join(f"#{m.index} {m.label} {m.start:.1f}-{m.end:.1f}" for m in masks) or "(none)"
