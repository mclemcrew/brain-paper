"""Seam, rhythm and key metrics for section inpainting (plan Phase 4.2).

All functions take mono or multichannel float audio as ``(samples,)``, ``(samples, channels)``
(the :mod:`soundfile` layout used across ``ravi.eval``) and downmix to mono by the channel mean.
Analysis runs at :data:`FEATURE_SR` (22,050 Hz, resampled with :func:`librosa.resample`, its
default ``soxr_hq``). librosa calls are verified against the installed 1.0.0 and the pinned
0.11.0 (``docker/eval``) sources: ``librosa.feature.mfcc(*, y, sr, n_mfcc=20, ...,
**kwargs)`` (kwargs reach the mel spectrogram: ``n_fft``, ``hop_length``),
``librosa.feature.chroma_stft(*, y, sr, norm=inf, n_fft=2048, hop_length=512, ...)``,
``librosa.feature.chroma_cqt(*, y, sr, hop_length=512, ..., n_octaves=7, bins_per_octave=36)``,
``librosa.beat.beat_track(*, y, sr, onset_envelope, hop_length, ..., tightness=100, bpm=None,
units='frames', sparse=True) -> (tempo, beats)`` (``bpm`` may be time-varying, one value per
onset frame, since librosa 0.10), ``librosa.onset.onset_strength(*, y, sr, ...)``,
``librosa.feature.tempo(*, onset_envelope, sr, hop_length, ac_size=8.0, aggregate=np.mean)``
(``aggregate=None`` = one tempo per frame).

Definitions (reported in the paper):

**Novelty curve**, :func:`novelty_curve` (Foote 2000): frames at :data:`FRAME_RATE` = 10 Hz
(hop 2205 samples at 22,050 Hz, ``n_fft`` 4096, centred). Two feature blocks per frame:
MFCC 1-13 (MFCC 0, the loudness term, dropped) and 12-bin STFT chroma. Each block is z-scored
per dimension over the file, then each frame is L2-normalised within the block; the
self-similarity matrix is the *average of the two cosine similarity matrices* (timbre and
harmony weigh equally). Novelty ``N[t] = sum K * S[t-L:t+L, t-L:t+L]`` with a Gaussian-tapered
checkerboard kernel ``K`` of half-width ``L`` = ``kernel_s`` * 10 frames (default 2 s; Gaussian
sigma = L/2), normalised so ``sum |K| = 1``, and the SSM zero-padded at the file edges. ``N`` is
in cosine units (``|N| <= 1``) and is *not* max-normalised per file, so values from a
regenerated and an original file are comparable.

**Seam score**, :func:`seam_score`: the peak sharpness of ``N`` at a seam, from the second
finite difference with step ``h`` = ``window_s`` (default 1 s = 10 frames):
``c[t] = N[t] - (N[t - h] + N[t + h]) / 2 = -Δ²_h N[t] / 2``, the maximum of ``c`` over frames
within ``tol_s`` (default 0.3 s) of the seam. Positive = the novelty peaks at the seam (an
abrupt change); about 0 = no local peak. Indices outside the curve are clamped to its ends.

**Seam delta**, :func:`seam_delta`: ``seam_score(generated) - seam_score(original)`` at the same
seam times; the reference is the original *at the same position*, so whatever novelty the
original has there is subtracted. Lower is better (signed: negative = smoother than the
original there); ``run_eval`` also reports ``abs_seam_delta`` (mean of ``|delta|`` over the two
seams: 0 = the original's local structure is reproduced either way). Geometry caveat: the mask
is the section padded by ``pad_s`` (1 s, ``select_regions``), so a seam sits ``pad_s`` *inside the
neighbouring section*, not on the real section boundary, which lies inside the mask. With the
2 s kernel the original's boundary novelty peak is 1 s from the seam and its curvature there is
usually small or negative; the regenerated audio must re-create that boundary inside the mask
(see the structure-consistency metric in ``run_eval``), and the seam itself should look like
the middle of a section.

**Beat-grid agreement** (primary rhythmic-continuity metric), :func:`beat_grid_agreement`:
beats tracked over the whole original context and over the whole generated context with the
*same* tracker, :func:`grid_beats`: :func:`librosa.beat.beat_track` with ``bpm`` fixed to the
original context's global tempo (:func:`global_tempo`) and a loose ``tightness`` of
:data:`GRID_TIGHTNESS` (10; librosa's default is 100). Both choices are measured (librosa
1.0.0): estimating each file's tempo separately makes the tracker flip phase on near-identical
audio (a corpus track plus -50 dB noise: agreement 0.32 at both seams), which the shared
orig-tempo prior removes (1.0 on all five tracks and both seams); and with ``tightness=100``
the dynamic programme holds the period through a 120 -> 80 BPM jump (agreement 1.0), while 10
lets it follow the generated onsets (0.69; a 250 ms phase jump gives 0.44-0.5 at any tightness;
near-identical real audio 0.91-1.0; the original context with the region replaced by another
track 0.29-0.70). For each seam, the orig beats within ``±window_s`` (default 4 s) and, for each, the
distance to the nearest generated beat. ``agree`` = fraction within ``±tol_s`` (70 ms, the
usual beat-tracking F-measure tolerance), ``offset`` = mean absolute distance (s). 1 / 0 = the
generated audio keeps the original's beat grid across the seam; a tempo or phase jump lowers
``agree`` and raises ``offset``. NaN with fewer than 3 orig beats in the window. Unlike the IBI
CV below this compares against the original's *grid*, so a steady but shifted or re-tempoed
continuation counts as a failure.

**Beat regularity** (secondary), :func:`beat_regularity`: beats from one :func:`librosa.beat.beat_track`
pass over the *whole* file (not per region: short-window tempo estimates octave-error), then
for each seam the inter-beat intervals of the beats within ``±window_s`` (default 4 s) and their
coefficient of variation ``std / mean`` (``ddof=0``). The CV is scale-free, so the tracker's
choice of metrical level (octave ambiguity) does not change it, while a tempo or phase jump at
the seam raises it. NaN with fewer than 3 beats in the window. Lower = steadier.
Tempo mode (``tempo``, default ``"global"``, for every beat function here): ``"global"`` is
librosa's default single-tempo tracker; ``"local"`` passes the per-frame autocorrelation tempo
of :func:`librosa.feature.tempo` (``aggregate=None``, 8 s window) as ``bpm``. The trade-off,
measured with librosa 1.0.0: with one global tempo and ``tightness=100`` the dynamic programme
keeps a constant period across a tempo change (click track 120 -> 80 BPM at the seam: IBI CV
0.023, the same as a steady track, vs 0.20 with ``"local"``), but on *unedited* real music the
local tempo curve flips metrical level (per-frame estimates spanning 99-199 BPM within a track)
and the local tracker's CV is 0.12-0.21 at 5 of 13 positions on five corpus tracks, against
0.018-0.040 everywhere with ``"global"``. A metric that fires on the original cannot measure
the edit, so the default is ``"global"``, the IBI CV is secondary, and the primary rhythmic
metric is the beat-grid agreement (which a global-tempo tracker on the generated audio still
fails when the continuation drifts off the original's grid). ``beat_cv_delta`` in ``run_eval``
is the per-seam difference to the original at the same seams, then the NaN-mean.

**Key**, :func:`estimate_key` / :func:`key_agreement`: CQT chroma (hop 512 at 22,050 Hz)
summed over time, Pearson-correlated with the 24 rotations of the Krumhansl-Kessler major and
minor profiles (Krumhansl 1990); the key is the best correlation and the *confidence* is
``r_best / r_second`` (inf if ``r_second <= 0 < r_best``; NaN for silence). Key agreement
compares the key of the regenerated region with the key of the original's context and is
evaluated only where the context's confidence is ``>= 1.1`` (:data:`KEY_CONF_GATE`) *and* its
best correlation is ``>= 0.6`` (:data:`KEY_MIN_R`); otherwise ``agree`` is None (ambient/atonal
cues are gated out). The correlation floor is ours: the ratio alone is unstable when every
correlation is weak (4 s of white noise, three seeds, librosa 1.0.0: r_best 0.27-0.51 with
ratios up to 2.07), while a I-IV-V-I progression in C scores r = 0.94, ratio 1.18, and single
major/minor triads r = 0.85-0.91. Agreement is exact (same tonic and mode),
so a relative major/minor confusion counts as disagreement.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

__all__ = [
    "FEATURE_SR",
    "FRAME_RATE",
    "KEY_CONF_GATE",
    "KEY_MIN_R",
    "to_mono",
    "novelty_curve",
    "seam_score",
    "seam_delta",
    "beat_times",
    "beat_regularity",
    "beat_grid_agreement",
    "grid_beats",
    "global_tempo",
    "BEAT_TOL_S",
    "GRID_TIGHTNESS",
    "KeyEstimate",
    "estimate_key",
    "key_agreement",
]

FEATURE_SR = 22050
FRAME_RATE = 10
HOP = FEATURE_SR // FRAME_RATE  # 2205
N_FFT = 4096
DEFAULT_KERNEL_S = 2.0
KEY_CONF_GATE = 1.1
KEY_MIN_R = 0.6
BEAT_TOL_S = 0.07
DEFAULT_TEMPO = "global"
GRID_TIGHTNESS = 10.0

# Krumhansl & Kessler (1982) probe-tone profiles, C major / C minor (Krumhansl 1990, p. 30).
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1)
    elif a.ndim != 1:
        raise ValueError(f"expected (samples,) or (samples, channels), got {a.shape}")
    return np.ascontiguousarray(a)


def _prep(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = to_mono(audio)
    if int(sr) != FEATURE_SR:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=FEATURE_SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def _block_norm(f: np.ndarray) -> np.ndarray:
    """z-score each feature dimension over time, then unit-normalise each frame. ``f``: [d, T]."""
    f = f.astype(np.float64)
    sd = f.std(axis=1, keepdims=True)
    f = (f - f.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0)
    nrm = np.linalg.norm(f, axis=0, keepdims=True)
    return np.divide(f, nrm, out=np.zeros_like(f), where=nrm > 1e-12)


def checkerboard_kernel(half: int) -> np.ndarray:
    """``2*half x 2*half`` Gaussian-tapered checkerboard (+ on the diagonal blocks), sum|K| = 1."""
    if half < 1:
        raise ValueError("kernel half-width must be >= 1 frame")
    ax = np.arange(-half, half) + 0.5
    sign = np.sign(ax)
    g = np.exp(-0.5 * (ax / (half / 2.0)) ** 2)
    k = np.outer(sign, sign) * np.outer(g, g)
    return k / np.abs(k).sum()


def ssm(audio: np.ndarray, sr: int) -> np.ndarray:
    """Average of the MFCC and chroma cosine self-similarity matrices at 10 Hz, [T, T]."""
    import librosa

    y = _prep(audio, sr)
    mf = librosa.feature.mfcc(y=y, sr=FEATURE_SR, n_mfcc=14, n_fft=N_FFT, hop_length=HOP)[1:]
    ch = librosa.feature.chroma_stft(y=y, sr=FEATURE_SR, n_fft=N_FFT, hop_length=HOP)
    t = min(mf.shape[1], ch.shape[1])
    a, b = _block_norm(mf[:, :t]), _block_norm(ch[:, :t])
    return 0.5 * (a.T @ a + b.T @ b)


def novelty_curve(audio: np.ndarray, sr: int, kernel_s: float = DEFAULT_KERNEL_S) -> np.ndarray:
    """Foote novelty at :data:`FRAME_RATE` Hz; frame ``i`` is centred at ``i / 10`` s."""
    s = ssm(audio, sr)
    half = max(1, int(round(kernel_s * FRAME_RATE)))
    k = checkerboard_kernel(half)
    n = s.shape[0]
    pad = np.zeros((n + 2 * half, n + 2 * half))
    pad[half : half + n, half : half + n] = s
    out = np.empty(n)
    for i in range(n):
        out[i] = float(np.sum(k * pad[i : i + 2 * half, i : i + 2 * half]))
    return out


def _curvature_at(nov: np.ndarray, t_seams: Sequence[float], window_s: float, tol_s: float) -> np.ndarray:
    n = nov.size
    h = max(1, int(round(window_s * FRAME_RATE)))
    tol = max(0, int(round(tol_s * FRAME_RATE)))
    idx = np.arange(n)
    c = nov - 0.5 * (nov[np.clip(idx - h, 0, n - 1)] + nov[np.clip(idx + h, 0, n - 1)])
    out = []
    for t in t_seams:
        f = int(round(float(t) * FRAME_RATE))
        lo, hi = max(0, f - tol), min(n, f + tol + 1)
        out.append(float(np.max(c[lo:hi])) if lo < hi else math.nan)
    return np.array(out, dtype=np.float64)


def seam_score(
    audio: np.ndarray, sr: int, t_seams: Sequence[float], window_s: float = 1.0,
    tol_s: float = 0.3, kernel_s: float = DEFAULT_KERNEL_S, novelty: np.ndarray | None = None,
) -> np.ndarray:
    """Novelty peak sharpness ``N[t] - (N[t-h] + N[t+h]) / 2`` at each seam (module docstring).

    ``novelty``: a precomputed :func:`novelty_curve` of ``audio`` (skips the recomputation)."""
    nov = novelty_curve(audio, sr, kernel_s) if novelty is None else np.asarray(novelty, np.float64)
    return _curvature_at(nov, t_seams, window_s, tol_s)


def seam_delta(
    gen_audio: np.ndarray, orig_audio: np.ndarray, sr: int, t_seams: Sequence[float],
    window_s: float = 1.0, tol_s: float = 0.3, kernel_s: float = DEFAULT_KERNEL_S,
) -> np.ndarray:
    """``seam_score(gen) - seam_score(orig)`` at the same seams (lower is better)."""
    g = seam_score(gen_audio, sr, t_seams, window_s, tol_s, kernel_s)
    o = seam_score(orig_audio, sr, t_seams, window_s, tol_s, kernel_s)
    return g - o


def beat_times(audio: np.ndarray, sr: int, tempo: str = DEFAULT_TEMPO) -> np.ndarray:
    """Beat times (s) from one whole-file :func:`librosa.beat.beat_track` pass (hop 512);
    ``tempo="local"`` passes the per-frame tempo curve as ``bpm`` (module docstring)."""
    import librosa

    if tempo not in ("local", "global"):
        raise ValueError(f"tempo must be 'local' or 'global', got {tempo!r}")
    y = _prep(audio, sr)
    oenv = librosa.onset.onset_strength(y=y, sr=FEATURE_SR, hop_length=512)
    bpm = None
    if tempo == "local":
        bpm = librosa.feature.tempo(onset_envelope=oenv, sr=FEATURE_SR, hop_length=512, aggregate=None)
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=FEATURE_SR, hop_length=512, bpm=bpm,
                                       units="time")
    return np.asarray(beats, dtype=np.float64)


def _onset_env(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=_prep(audio, sr), sr=FEATURE_SR, hop_length=512)


def global_tempo(audio: np.ndarray, sr: int) -> float:
    """librosa's single global tempo estimate (BPM) of the whole file."""
    import librosa

    return float(np.atleast_1d(librosa.feature.tempo(onset_envelope=_onset_env(audio, sr), sr=FEATURE_SR,
                                                     hop_length=512))[0])


def grid_beats(audio: np.ndarray, sr: int, bpm: float, tightness: float = GRID_TIGHTNESS) -> np.ndarray:
    """Beat times (s) of the whole file tracked at a *given* tempo ``bpm`` (the original's) with
    a loose ``tightness`` (module docstring: beat-grid agreement)."""
    import librosa

    _, beats = librosa.beat.beat_track(onset_envelope=_onset_env(audio, sr), sr=FEATURE_SR, hop_length=512,
                                       bpm=float(bpm), tightness=float(tightness), units="time")
    return np.asarray(beats, dtype=np.float64)


def beat_regularity(
    audio: np.ndarray, sr: int, t_seams: Sequence[float], window_s: float = 4.0,
    beats: np.ndarray | None = None, tempo: str = DEFAULT_TEMPO,
) -> np.ndarray:
    """Inter-beat-interval CV (``std/mean``) of the beats within ``±window_s`` of each seam.

    ``beats``: precomputed :func:`beat_times` of ``audio``."""
    b = beat_times(audio, sr, tempo) if beats is None else np.asarray(beats, np.float64)
    out = []
    for t in t_seams:
        w = b[(b >= float(t) - window_s) & (b <= float(t) + window_s)]
        if w.size < 3:
            out.append(math.nan)
            continue
        ibi = np.diff(w)
        out.append(float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else math.nan)
    return np.array(out, dtype=np.float64)


def beat_grid_agreement(
    gen_audio: np.ndarray | None, orig_audio: np.ndarray | None, sr: int, t_seams: Sequence[float],
    window_s: float = 4.0, tol_s: float = BEAT_TOL_S, gen_beats: np.ndarray | None = None,
    orig_beats: np.ndarray | None = None, tightness: float = GRID_TIGHTNESS, min_beats: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """``(agree, offset_s)`` per seam: the fraction of the original's beats within ``±window_s``
    of the seam that have a generated beat within ``±tol_s``, and their mean absolute distance
    to the nearest generated beat (module docstring). Beats are whole-file :func:`grid_beats`
    of each audio at the original's :func:`global_tempo` unless passed in."""
    bpm = global_tempo(orig_audio, sr) if (orig_beats is None or gen_beats is None) else None
    ob = grid_beats(orig_audio, sr, bpm, tightness) if orig_beats is None else np.asarray(orig_beats, np.float64)
    gb = grid_beats(gen_audio, sr, bpm, tightness) if gen_beats is None else np.asarray(gen_beats, np.float64)
    gb = np.sort(gb)
    agree, offs = [], []
    for t in t_seams:
        w = ob[(ob >= float(t) - window_s) & (ob <= float(t) + window_s)]
        if w.size < min_beats:
            agree.append(math.nan)
            offs.append(math.nan)
            continue
        if gb.size == 0:
            agree.append(0.0)
            offs.append(math.nan)
            continue
        j = np.clip(np.searchsorted(gb, w), 1, max(1, gb.size - 1))
        near = np.minimum(np.abs(w - gb[j - 1]), np.abs(w - gb[np.minimum(j, gb.size - 1)]))
        agree.append(float(np.mean(near <= tol_s + 1e-9)))
        offs.append(float(np.mean(near)))
    return np.array(agree, dtype=np.float64), np.array(offs, dtype=np.float64)


class KeyEstimate(NamedTuple):
    key: str | None  # e.g. "C major", "A minor"; None for silence
    tonic: int | None
    mode: str | None
    r_best: float
    r_second: float
    confidence: float


def estimate_key(audio: np.ndarray, sr: int) -> KeyEstimate:
    """Krumhansl-Kessler key estimate from time-summed CQT chroma (module docstring)."""
    import librosa

    y = _prep(audio, sr)
    if y.size == 0 or not np.any(np.abs(y) > 1e-6):
        return KeyEstimate(None, None, None, math.nan, math.nan, math.nan)
    chroma = librosa.feature.chroma_cqt(y=y, sr=FEATURE_SR, hop_length=512).sum(axis=1)
    if np.std(chroma) < 1e-12:
        return KeyEstimate(None, None, None, math.nan, math.nan, math.nan)
    scores = []
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma, np.roll(prof, tonic))[0, 1])
            scores.append((r, tonic, mode))
    scores.sort(key=lambda s: -s[0])
    (r1, tonic, mode), (r2, _, _) = scores[0], scores[1]
    if r2 > 0:
        conf = r1 / r2
    else:
        conf = math.inf if r1 > 0 else math.nan
    return KeyEstimate(f"{PITCH_CLASSES[tonic]} {mode}", tonic, mode, r1, r2, conf)


def key_agreement(
    gen_region: np.ndarray, context_audio: np.ndarray, sr: int, conf_gate: float = KEY_CONF_GATE,
    min_r: float = KEY_MIN_R, context_key: KeyEstimate | None = None,
) -> tuple[bool | None, float]:
    """``(agree, conf_orig)``: does the regenerated region's key equal the original context's key?

    ``agree`` is None when the context's key confidence is below ``conf_gate`` (or undefined),
    its best correlation is below ``min_r``, or the region is silent. ``context_key``: a
    precomputed :func:`estimate_key` of ``context_audio``."""
    ctx = estimate_key(context_audio, sr) if context_key is None else context_key
    conf = ctx.confidence
    if ctx.key is None or not (conf >= conf_gate) or not (ctx.r_best >= min_r):
        return None, conf
    gen = estimate_key(gen_region, sr)
    if gen.key is None:
        return None, conf
    return gen.key == ctx.key, conf
