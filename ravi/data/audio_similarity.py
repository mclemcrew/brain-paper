"""Find near-duplicate recordings by chroma similarity.

Usage::

    python -m ravi.data.audio_similarity --manifest manifests/tracks.parquet \
        [--cache manifests/chroma_cache.npz] [--out manifests/audio_similarity_pairs.csv] \
        [--threshold 0.70] [--write-links manifests/manual_links.json]

Several corpus entries are alternate mixes, cut-downs or re-uploads of the
same piece without any ``[ALT]`` link or shared title. Splitting such a pair
across train and test leaks the test piece into training. This module scores
every pair of tracks:

1. Decode the first audio stream with ffmpeg to 8 kHz mono float32.
2. 12-bin chroma at 4 frames/s: STFT magnitude (n_fft 4096, hop 2000,
   scaled by ``2 / n_fft`` so a full-scale sinusoid has magnitude ~1), bins
   outside 65-2000 Hz zeroed, folded to pitch classes with
   ``librosa.feature.chroma_stft`` (flat octave weighting, tuning fixed at 0,
   no per-frame normalisation), compressed with ``log1p(10 * C)``; each
   frame's mean over the 12 bins is subtracted so overall level and broadband
   noise do not count. (This mirrors the reviewer's scipy-STFT chroma, whose
   0.70 threshold separated alternate mixes from unrelated tracks.)
3. Similarity of tracks ``a`` and ``b`` = the best cosine similarity
   (over the whole overlapping chroma matrix) across lags of up to
   :data:`MAX_LAG_S` in either direction, with at least :data:`MIN_OVERLAP_S`
   of overlap. Both tracks start at their lag offset, so a cut-down that
   starts like the full mix scores high.

All pairs (``N^2 / 2``) are computed with one matrix product per lag.
Features are cached in ``--cache`` (gitignored) keyed by track id, source
size and the feature parameters.

The CLI writes ``--out``: every pair with similarity >= ``--threshold`` whose
two tracks are not already in the same family *before* audio links (``[ALT]``
refs, duplicate ids, title stems, ``[AKA]`` rule), sorted by similarity.
With ``--write-links`` ``manual_links.json`` is regenerated from its
hand-written entries (any reason not starting with :data:`AUTO_PREFIX`) plus
those pairs, then filtered by :func:`filter_links`: a link survives if both
tracks are at least ``--min-clip-s`` (8 s) long, or its similarity is at
least ``--short-clip-sim`` (0.80), or a title-stem/[ALT]/[AKA] link already
joins the two tracks. The file is ``{"pairs": [[a, b, sim, reason], ...],
"dropped": [[a, b, sim, reason, drop_reason], ...]}``;
:func:`ravi.data.build_manifest.assign_families` unions every kept pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ravi.audio_similarity")

SR = 8000
FPS = 4
HOP = SR // FPS
N_FFT = 4096
MAX_LAG_S = 3.0
MIN_OVERLAP_S = 3.0
THRESHOLD = 0.70
MIN_CLIP_S = 8.0  # see filter_links
SHORT_CLIP_SIM = 0.80
FMIN, FMAX = 65.0, 2000.0
FEATURE_VERSION = 2
PARAMS = {"sr": SR, "fps": FPS, "n_fft": N_FFT, "fmin": FMIN, "fmax": FMAX, "version": FEATURE_VERSION}


def decode_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode('utf-8', 'replace')[-1000:]}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def chroma_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """(12, n_frames) float32 log-chroma with the per-frame mean removed."""
    import librosa

    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)) * (2.0 / N_FFT)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    mag[(freqs < FMIN) | (freqs > FMAX)] = 0.0
    c = librosa.feature.chroma_stft(S=mag, sr=sr, n_fft=N_FFT, norm=None, tuning=0.0, octwidth=None)
    c = np.log1p(10.0 * c)
    c -= c.mean(axis=0, keepdims=True)
    return c.astype(np.float32)


def track_features(path: str | Path) -> np.ndarray:
    return chroma_features(decode_mono(path))


# --------------------------------------------------------------------------- cache


def _cache_key(track_id: str, audio_path: str | Path) -> str:
    try:
        size = Path(audio_path).stat().st_size
    except OSError:
        size = -1
    return f"{track_id}\x1f{size}"


def load_features(df: pd.DataFrame, cache: Path | None, workers: int = 8) -> list[np.ndarray]:
    """Features for every manifest row, reusing ``cache`` where the key matches."""
    keys = [_cache_key(t, p) for t, p in zip(df["track_id"], df["audio_path"])]
    cached: dict[str, np.ndarray] = {}
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if json.loads(str(z["params"])) == PARAMS:
            offs = z["offsets"]
            for i, k in enumerate(z["keys"].tolist()):
                cached[k] = z["features"][:, offs[i] : offs[i + 1]]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    if todo:
        log.info("computing chroma for %d tracks (%d cached)", len(todo), len(keys) - len(todo))
        paths = [df["audio_path"].iloc[i] for i in todo]
        with ThreadPoolExecutor(workers) as ex:
            for i, f in zip(todo, ex.map(track_features, paths)):
                cached[keys[i]] = f
    feats = [cached[k] for k in keys]
    if cache is not None and todo:
        lengths = np.array([f.shape[1] for f in feats])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            keys=np.array(keys),
            offsets=np.concatenate([[0], np.cumsum(lengths)]),
            features=np.concatenate(feats, axis=1),
            params=np.array(json.dumps(PARAMS)),
        )
    return feats


# --------------------------------------------------------------------------- similarity


def pairwise_similarity(
    feats: list[np.ndarray], max_lag: int = int(MAX_LAG_S * FPS), min_overlap: int = int(MIN_OVERLAP_S * FPS)
) -> np.ndarray:
    """Symmetric (N, N) matrix of best lagged cosine similarity; NaN on the
    diagonal and where no lag gives ``min_overlap`` frames of overlap.

    For lag ``L >= 0`` track ``i`` is advanced by ``L`` frames: frames
    ``i[L + t]`` are paired with ``j[t]`` for ``t < min(T_i - L, T_j)``.
    Negative lags are the transpose of positive ones.
    """
    n = len(feats)
    lengths = np.array([f.shape[1] for f in feats])
    t_max = int(lengths.max())
    x = np.zeros((n, 12, t_max), dtype=np.float64)
    for i, f in enumerate(feats):
        x[i, :, : f.shape[1]] = f
    energy = (x**2).sum(axis=1)  # (n, t_max)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(energy, axis=1)], axis=1)  # (n, t_max + 1)
    rows = np.arange(n)[:, None]
    best = np.full((n, n), -np.inf)
    for lag in range(0, max_lag + 1):
        width = t_max - lag
        if width <= 0:
            break
        a = x[:, :, lag:].reshape(n, -1)
        b = x[:, :, :width].reshape(n, -1)
        dot = a @ b.T  # dot[i, j] = sum_t x_i[t + lag] . x_j[t]
        overlap = np.minimum(lengths[:, None] - lag, lengths[None, :])
        overlap = np.clip(overlap, 0, None)
        e_i = cs[rows, np.minimum(lag + overlap, t_max)] - cs[rows, np.minimum(lag, t_max)]
        e_j = cs[np.arange(n)[None, :], overlap] - 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = dot / np.sqrt(e_i * e_j)
        sim[(overlap < min_overlap) | ~np.isfinite(sim)] = -np.inf
        best = np.maximum(best, np.maximum(sim, sim.T))
    best[~np.isfinite(best)] = np.nan
    np.fill_diagonal(best, np.nan)
    return best


def similar_pairs(
    df: pd.DataFrame, sim: np.ndarray, families: list[int] | np.ndarray, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """Pairs ``(i < j)`` with ``sim >= threshold`` in different families, highest first."""
    fam = np.asarray(families)
    iu, ju = np.triu_indices(len(df), k=1)
    s = sim[iu, ju]
    keep = np.isfinite(s) & (s >= threshold) & (fam[iu] != fam[ju])
    tid = df["track_id"].to_numpy()
    dur = df["duration_s"].to_numpy()
    out = pd.DataFrame({
        "track_id_a": tid[iu[keep]],
        "track_id_b": tid[ju[keep]],
        "similarity": np.round(s[keep], 4),
        "family_a": fam[iu[keep]],
        "family_b": fam[ju[keep]],
        "duration_a": np.round(dur[iu[keep]], 3),
        "duration_b": np.round(dur[ju[keep]], 3),
    })
    return out.sort_values(["similarity", "track_id_a", "track_id_b"], ascending=[False, True, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- manual links

#: Reason prefix of machine-generated links; entries with any other reason are
#: hand-written and survive regeneration.
AUTO_PREFIX = "chroma similarity"


def auto_reason(threshold: float) -> str:
    return f"{AUTO_PREFIX} >= {threshold:.2f} (ravi.data.audio_similarity v{FEATURE_VERSION})"


def read_links(path: Path) -> list[list]:
    """The kept ``pairs`` of ``manual_links.json`` (``[]`` if the file is absent)."""
    return _read(path)[0]


def _read(path: Path) -> tuple[list[list], list[list]]:
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    pairs, dropped = data.get("pairs", []), data.get("dropped", [])
    for p in pairs:
        if not (isinstance(p, list) and len(p) == 4 and all(isinstance(t, str) for t in p[:2])):
            raise ValueError(f"{path}: malformed pair {p!r}; expected [track_id_a, track_id_b, sim, reason]")
    for p in dropped:
        if not (isinstance(p, list) and len(p) == 5 and all(isinstance(t, str) for t in p[:2])):
            raise ValueError(f"{path}: malformed dropped pair {p!r}; expected [a, b, sim, reason, drop_reason]")
    return pairs, dropped


def hand_links(path: Path) -> list[list]:
    """Hand-written entries (kept or dropped) as ``[a, b, sim, reason]``."""
    pairs, dropped = _read(path)
    return [list(p) for p in pairs + [d[:4] for d in dropped] if not str(p[3]).startswith(AUTO_PREFIX)]


def merge_links(existing: list[list], pairs: pd.DataFrame, reason: str) -> list[list]:
    """Add CSV pairs to the link list; hand-written entries (and their reasons) win."""
    seen = {frozenset(p[:2]) for p in existing}
    out = [list(p) for p in existing]
    for r in pairs.itertuples(index=False):
        key = frozenset((r.track_id_a, r.track_id_b))
        if key in seen:
            continue
        out.append([r.track_id_a, r.track_id_b, float(r.similarity), reason])
        seen.add(key)
    return _sorted(out)


def _sorted(links: list[list]) -> list[list]:
    return sorted(links, key=lambda p: (-(p[2] if p[2] is not None else 0.0), p[0], p[1]))


def filter_links(
    links: list[list],
    durations: dict[str, float],
    base_family: dict[str, int],
    min_clip_s: float = MIN_CLIP_S,
    short_clip_sim: float = SHORT_CLIP_SIM,
) -> tuple[list[list], list[list]]:
    """Split links into ``(kept, dropped)``.

    A link is kept if both tracks last at least ``min_clip_s``, or its
    similarity is at least ``short_clip_sim``, or the two tracks are already in
    one family without audio links (title stem / [ALT] / [AKA] / duplicate id,
    ``base_family``). Otherwise it is a short clip matching on a generic chord:
    a 4-6 s stinger that scores >= 0.70 against 6-8 unrelated full-length
    tracks is a chroma false positive, not an alternate version. Dropped
    entries are ``[a, b, sim, reason, drop_reason]``.
    """
    kept, dropped = [], []
    for a, b, sim, reason in links:
        short = min(durations[a], durations[b])
        sim_v = float(sim) if sim is not None else float("nan")
        if short >= min_clip_s or sim_v >= short_clip_sim or base_family[a] == base_family[b]:
            kept.append([a, b, sim, reason])
        else:
            dropped.append([a, b, sim, reason,
                            f"shorter track {short:.1f} s < {min_clip_s:g} s and similarity {sim_v:.3f} < "
                            f"{short_clip_sim:.2f}, no title-stem/[ALT]/[AKA] link: likely a generic-chord "
                            f"match of a short clip, not an alternate version"])
    return _sorted(kept), _sorted(dropped)


def write_links(path: Path, pairs: list[list], dropped: list[list] | None = None) -> None:
    def block(rows: list[list]) -> str:
        return ",\n".join("    " + json.dumps(p, ensure_ascii=False) for p in rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    text = '{\n  "pairs": [\n' + block(pairs) + "\n  ]"
    if dropped is not None:
        text += ',\n  "dropped": [\n' + block(dropped) + "\n  ]"
    path.write_text(text + "\n}\n")


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    from ravi.data.build_manifest import families_from_manifest

    ap = argparse.ArgumentParser(description="Chroma near-duplicate search over the corpus.")
    ap.add_argument("--manifest", type=Path, default=Path("manifests/tracks.parquet"))
    ap.add_argument("--cache", type=Path, default=Path("manifests/chroma_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("manifests/audio_similarity_pairs.csv"))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--write-links", type=Path, default=None,
                    help="regenerate this manual_links.json: hand-written entries + CSV pairs, filtered")
    ap.add_argument("--min-clip-s", type=float, default=MIN_CLIP_S,
                    help="links where a track is shorter than this need --short-clip-sim or a non-audio link")
    ap.add_argument("--short-clip-sim", type=float, default=SHORT_CLIP_SIM)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = pd.read_parquet(args.manifest)
    feats = load_features(df, args.cache, args.workers)
    sim = pairwise_similarity(feats)
    families = np.asarray(families_from_manifest(df, manual_links=None))
    pairs = similar_pairs(df, sim, families, args.threshold)
    base = dict(zip(df["track_id"], families.tolist()))
    durations = dict(zip(df["track_id"], df["duration_s"].astype(float)))
    links = merge_links(hand_links(args.write_links) if args.write_links else [], pairs, auto_reason(args.threshold))
    kept, dropped = filter_links(links, durations, base, args.min_clip_s, args.short_clip_sim)
    kept_keys = {frozenset(p[:2]) for p in kept}
    pairs["link_kept"] = [frozenset((a, b)) in kept_keys for a, b in zip(pairs.track_id_a, pairs.track_id_b)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)

    within = [sim[i, j] for f in np.unique(families) for i, j in _pairs_in(np.flatnonzero(families == f))]
    iu, ju = np.triu_indices(len(df), k=1)
    allv = sim[iu, ju][np.isfinite(sim[iu, ju])]
    print(f"pairs scored {len(allv)}; percentiles 50/99/99.9 {np.percentile(allv, [50, 99, 99.9]).round(3)}")
    if within:
        print(f"within-family (pre-audio) pairs {len(within)}; median {np.nanmedian(within):.3f}")
    print(f"{len(pairs)} cross-family pairs >= {args.threshold} ({int(pairs.link_kept.sum())} pass the "
          f"short-clip filter); wrote {args.out}")
    if args.write_links:
        write_links(args.write_links, kept, dropped)
        print(f"{len(kept)} links kept, {len(dropped)} dropped in {args.write_links}")


def _pairs_in(idx: np.ndarray):
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            yield idx[a], idx[b]


if __name__ == "__main__":
    main()
