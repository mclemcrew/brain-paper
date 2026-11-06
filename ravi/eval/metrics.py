"""Distribution and embedding metrics for the rAvI v2 evaluation (plan Phase 4).

Pure numpy (float64 throughout); embeddings come from :mod:`ravi.eval.embed`. Definitions as
reported in the paper:

**FAD** (Kilgour et al. 2019), :func:`fad`: the Fréchet distance between Gaussians fitted to the
reference and generated embeddings, ``||mu_r - mu_g||^2 + tr(C_r) + tr(C_g) - 2 tr((C_r C_g)^{1/2})``
with unbiased (``ddof=1``) covariances. The trace term is computed without a general matrix
square root: ``C_r C_g`` is similar to the symmetric PSD matrix ``C_r^{1/2} C_g C_r^{1/2}``, so
``tr((C_r C_g)^{1/2}) = sum_i sqrt(lambda_i(C_r^{1/2} C_g C_r^{1/2}))``, with ``C_r^{1/2}`` from a
symmetric eigendecomposition and negative round-off eigenvalues clipped to 0. This is exact, real
and stable for rank-deficient covariances (n < d), where ``scipy.linalg.sqrtm(C_r @ C_g)`` can
return complex output (fadtk 1.1.0 ``fad.py:51-110`` works around exactly that).

**FAD∞** (Gui, Gamper, Braun, Emmanouilidou, "Adapting Frechet Audio Distance for Generative
Music Evaluation", ICASSP 2024, arXiv:2311.01616, Sec. 3.3), :func:`fad_inf`: FAD of ``n`` rows
drawn from the generated set for ``n`` on a grid, a least-squares line of FAD against ``1/n``,
and the intercept (``1/n -> 0``) reported as FAD∞, with the slope and R² of the fit. The paper
and fadtk 1.1.0 (``FrechetAudioDistance.score_inf``: 25 sizes ``linspace(500, N)``, rows drawn
*with* replacement, one draw per size) bootstrap with replacement. Note that a with-replacement
draw of ``n`` rows from a fixed ``N``-row set converges to that set's own statistics as
``n -> inf``, so its intercept estimates the *full-sample* FAD (``N`` rows), which still carries
the ``O(1/N)`` bias; drawing *without* replacement (``n <= N``) makes the expected curve
``FAD_pop + b/n`` and the intercept an estimate of the population FAD. We therefore default to
``replace=False`` (our reported FAD∞) and keep ``replace=True`` to reproduce fadtk; both are
tested on Gaussians with a known answer. Default grid: 25 sizes ``linspace(min_n, N)`` with
``min_n = max(3 d, 500)`` (``d`` = embedding dimension), ``n_draws=1`` per size (fadtk), seeded.
The floor matters: for ``n`` comparable to ``d`` the sample covariance is rank-deficient and
``E[FAD(n)]`` is not linear in ``1/n``, so a fit over ``n < d`` extrapolates to the wrong
intercept with a deceptively high R² (768-d Gaussian with a power-law spectrum, ``N = 2000``:
the old ``min(500, N // 2)`` grid gave FAD∞ 0.31 at R² 0.994 for a true 0.54).

**PCA** (:func:`pca_fit`, :func:`pca_project`; ``run_eval --pca-dims``, default 64): FAD, FAD∞
and KAD are computed after projecting both sets onto the top ``k`` principal axes of the
*reference* embeddings (mean and axes fitted on the reference only, so every condition is
projected identically). With ``k = 64`` the grid floor is ``max(192, 500) = 500`` rows, well above
``k``, and in the simulation above PCA'd FAD∞ is within 7% of the true value (unit-tested at 15%).
The reduced-space values are what we report; the full-dimension plain FAD is kept only as a
``_naive`` diagnostic.

**Clip-level bootstrap** (:func:`clip_bootstrap`): windows of one clip are strongly dependent,
so a CI must resample *clips*: each resample draws the condition's clips with replacement, takes
all their windows, and recomputes FAD∞ and KAD; percentile interval over ``n_boot`` (default 200)
seeded resamples. Duplicated clips inflate the within-set terms a little (the resample has fewer
distinct clips), so the interval is approximate and slightly conservative. FAD∞ and KAD depend on
the number of clips and windows, so values are comparable only *between conditions at matched
n* (``run_eval`` subsamples every condition to the same clip count): they are relative
comparisons, not unbiased absolute distances.

**KAD** (Chung, Eu, Lee, Choi, Nam, Chon, "KAD: No More FAD! An Effective and Efficient
Evaluation Metric for Audio Generation", arXiv:2502.15602, 2025, Eqs. 4-5), :func:`kad`:
``alpha * MMD^2_u`` with the unbiased U-statistic estimator (diagonals of the within-set kernel
matrices excluded) and the Gaussian RBF kernel ``exp(-||x - y||^2 / (2 sigma^2))``; ``sigma`` is
the median pairwise Euclidean distance *within the reference set* (as the paper states; kadtk
1.1.0 ``calc_kernel_audio_distance`` instead takes the median of its second argument, the eval
set in ``score()``, which would make the kernel differ per condition) and ``alpha = 100`` (the
paper's and kadtk 1.1.0's ``SCALE_FACTOR``). For a reference set larger than
``bandwidth_max_n`` rows the median is taken over a seeded random subset of that many rows. The
unbiased estimator is not clipped: it is ~0 in expectation for two samples of one distribution and
can be slightly negative (for two *identical* sets of ``n`` rows it equals
``-2 alpha (1 - mean_{i != j} k_ij) / n``, because the cross term counts the ``k(x_i, x_i) = 1``
pairs that the within terms exclude). ``estimator="biased"`` gives the V-statistic (diagonals
included), which is ``>= 0`` and exactly 0 for identical sets; it is reported only as a check.

**CLAP text-audio score**, :func:`clap_text_audio_score`: cosine between each clip's CLAP audio
embedding and its prompt's CLAP text embedding (paired rows; raw cosine, not x100).

**Region similarity**, :func:`region_similarity`: cosine between the embedding of the
regenerated region and the embedding of the original audio of the same region (paired rows).

**Memorisation check**, :func:`memorisation_check`: for every generated row the maximum cosine
to any training row and that row's index; flagged when the maximum exceeds ``thresh`` (0.95,
plan Phase 4.1). A fixed threshold has no calibration (MERT layer-12 windows of *different*
tracks already reach cosine > 0.95), so ``run_eval`` flags on CLAP only and also reports a
null-calibrated threshold: the 99th percentile of the same statistic for the held-out test
originals against the training set (:func:`null_threshold`).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

__all__ = [
    "GaussianStats",
    "gaussian_stats",
    "frechet_distance",
    "fad",
    "FadInf",
    "fad_inf",
    "fad_inf_min_n",
    "Pca",
    "pca_fit",
    "pca_project",
    "KadGen",
    "kad_gen",
    "kad_from_counts",
    "clip_bootstrap",
    "null_threshold",
    "median_pairwise_distance",
    "KadRef",
    "kad_reference",
    "kad",
    "cosine_rows",
    "clap_text_audio_score",
    "region_similarity",
    "centred_cosine",
    "Memorisation",
    "memorisation_check",
]

KAD_SCALE = 100.0
DEFAULT_BANDWIDTH_MAX_N = 5000
DEFAULT_SEED = 20261014
_BLOCK = 2048


# --------------------------------------------------------------------------- FAD


class GaussianStats(NamedTuple):
    mu: np.ndarray
    cov: np.ndarray
    n: int


def _as2d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name}: expected [n, d] embeddings, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: non-finite embeddings")
    return x


def gaussian_stats(emb: np.ndarray) -> GaussianStats:
    """Mean and unbiased covariance (``ddof=1``) of ``[n, d]`` embeddings, ``n >= 2``."""
    x = _as2d(emb, "emb")
    if x.shape[0] < 2:
        raise ValueError(f"need >= 2 rows for a covariance, got {x.shape[0]}")
    return GaussianStats(x.mean(axis=0), np.cov(x, rowvar=False, ddof=1).reshape(x.shape[1], x.shape[1]), x.shape[0])


def _psd_sqrt(c: np.ndarray) -> np.ndarray:
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    """``||mu1 - mu2||^2 + tr(C1) + tr(C2) - 2 tr((C1 C2)^{1/2})`` via the symmetric form
    ``tr((C1^{1/2} C2 C1^{1/2})^{1/2})`` (see module docstring). Clipped at 0 (round-off only)."""
    mu1, mu2 = np.atleast_1d(np.asarray(mu1, np.float64)), np.atleast_1d(np.asarray(mu2, np.float64))
    c1, c2 = np.atleast_2d(np.asarray(cov1, np.float64)), np.atleast_2d(np.asarray(cov2, np.float64))
    if mu1.shape != mu2.shape or c1.shape != c2.shape or c1.shape != (mu1.size, mu1.size):
        raise ValueError(f"shape mismatch: mu {mu1.shape}/{mu2.shape}, cov {c1.shape}/{c2.shape}")
    s1 = _psd_sqrt(c1)
    m = s1 @ (0.5 * (c2 + c2.T)) @ s1
    ev = np.clip(np.linalg.eigvalsh(0.5 * (m + m.T)), 0.0, None)
    diff = mu1 - mu2
    val = float(diff @ diff + np.trace(c1) + np.trace(c2) - 2.0 * np.sum(np.sqrt(ev)))
    return max(0.0, val)


def _stats_of(x: np.ndarray | GaussianStats, name: str) -> GaussianStats:
    return x if isinstance(x, GaussianStats) else gaussian_stats(_as2d(x, name))


def fad(ref_emb: np.ndarray | GaussianStats, gen_emb: np.ndarray | GaussianStats) -> float:
    """Plain FAD between two embedding sets (``[n, d]`` arrays or precomputed
    :class:`GaussianStats`, e.g. of a large reference set computed once)."""
    r, g = _stats_of(ref_emb, "ref_emb"), _stats_of(gen_emb, "gen_emb")
    return frechet_distance(r.mu, r.cov, g.mu, g.cov)


class FadInf(NamedTuple):
    score: float  # intercept of FAD vs 1/n
    slope: float
    r2: float
    points: list  # [(n, fad), ...]
    fad_full: float  # plain FAD with all generated rows
    replace: bool
    n_lo: int  # smallest / largest subset size of the fit
    n_hi: int
    d: int  # embedding dimension


def fad_inf_min_n(d: int) -> int:
    """Default FAD∞ grid floor ``max(3 d, 500)`` (module docstring)."""
    return max(3 * int(d), 500)


def fad_inf(
    ref_emb: np.ndarray | GaussianStats,
    gen_emb: np.ndarray,
    sizes: Sequence[int] | None = None,
    n_steps: int = 25,
    min_n: int | None = None,
    replace: bool = False,
    n_draws: int = 1,
    seed: int = DEFAULT_SEED,
) -> FadInf:
    """FAD extrapolated to infinite generated-set size (Gui et al. 2023; see module docstring).

    ``sizes``: the subset sizes ``n`` (default ``linspace(min_n, N, n_steps)`` rounded, unique;
    ``min_n`` default :func:`fad_inf_min_n` ``= max(3 d, 500)``; ``N < min_n`` raises
    ``ValueError``). Each size is drawn ``n_draws`` times and the FADs averaged before the fit.
    ``replace=False`` (default) draws without replacement (``n <= N``); ``replace=True``
    reproduces fadtk 1.1.0.
    """
    ref = _stats_of(ref_emb, "ref_emb")
    g = _as2d(gen_emb, "gen_emb")
    n_all = g.shape[0]
    if n_all < 4:
        raise ValueError(f"fad_inf needs >= 4 generated rows, got {n_all}")
    if sizes is None:
        lo = max(2, fad_inf_min_n(g.shape[1]) if min_n is None else int(min_n))
        if n_all <= lo:
            raise ValueError(f"fad_inf: {n_all} generated rows <= grid floor min_n={lo} (d={g.shape[1]})")
        sizes = np.unique(np.round(np.linspace(lo, n_all, n_steps)).astype(int))
    sizes = [int(n) for n in sizes]
    if len(set(sizes)) < 2:
        raise ValueError(f"need >= 2 distinct sizes, got {sizes}")
    if min(sizes) < 2 or (not replace and max(sizes) > n_all):
        raise ValueError(f"sizes must be in [2, {n_all}] without replacement, got {sizes}")
    rng = np.random.default_rng(seed)
    points = []
    for n in sizes:
        vals = []
        for _ in range(int(n_draws)):
            idx = rng.choice(n_all, size=n, replace=replace)
            s = gaussian_stats(g[idx])
            vals.append(frechet_distance(ref.mu, ref.cov, s.mu, s.cov))
        points.append((n, float(np.mean(vals))))
    xs = 1.0 / np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else math.nan
    full = gaussian_stats(g)
    return FadInf(float(intercept), float(slope), r2, points,
                  frechet_distance(ref.mu, ref.cov, full.mu, full.cov), bool(replace),
                  int(min(sizes)), int(max(sizes)), int(g.shape[1]))


# --------------------------------------------------------------------------- PCA


class Pca(NamedTuple):
    mean: np.ndarray  # [d]
    components: np.ndarray  # [d, k], columns = principal axes, descending variance
    explained: float  # fraction of the reference variance kept
    k: int
    d: int


def pca_fit(ref_emb: np.ndarray, k: int) -> Pca:
    """Top-``k`` principal axes of the *reference* embeddings (eigendecomposition of the
    ``ddof=1`` covariance). ``k`` is capped at ``min(d, n - 1)``."""
    x = _as2d(ref_emb, "ref_emb")
    n, d = x.shape
    k = int(min(int(k), d, n - 1))
    if k < 1:
        raise ValueError(f"pca_fit: need k >= 1 and >= 2 rows, got k={k}, n={n}")
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1).reshape(d, d)
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(w)[::-1]
    w, v = np.clip(w[order], 0.0, None), v[:, order]
    total = float(w.sum())
    return Pca(mu, v[:, :k], float(w[:k].sum() / total) if total > 0 else math.nan, k, d)


def pca_project(p: Pca, x: np.ndarray) -> np.ndarray:
    x = _as2d(x, "x")
    if x.shape[1] != p.d:
        raise ValueError(f"pca_project: dimension {x.shape[1]} != fitted {p.d}")
    return (x - p.mean) @ p.components


# --------------------------------------------------------------------------- KAD


def _sqdist(x: np.ndarray, y: np.ndarray, xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    return np.maximum(xn[:, None] + yn[None, :] - 2.0 * (x @ y.T), 0.0)


def median_pairwise_distance(x: np.ndarray, max_n: int = DEFAULT_BANDWIDTH_MAX_N, seed: int = DEFAULT_SEED) -> float:
    """Median Euclidean distance over all pairs ``i < j`` of ``x`` (a seeded random subset of
    ``max_n`` rows when ``x`` is larger)."""
    from scipy.spatial.distance import pdist

    x = _as2d(x, "x")
    if x.shape[0] < 2:
        raise ValueError("need >= 2 rows")
    if x.shape[0] > max_n:
        x = x[np.sort(np.random.default_rng(seed).choice(x.shape[0], size=max_n, replace=False))]
    return float(np.median(pdist(x, "euclidean")))


class KadRef(NamedTuple):
    """The reference set's part of KAD, computed once and reused for every condition."""

    x: np.ndarray
    sqnorm: np.ndarray
    bandwidth: float
    kernel: str
    kxx_offdiag_sum: float  # sum of k(x_i, x_j) over i != j


def _kernel(d2: np.ndarray, gamma: float, kernel: str) -> np.ndarray:
    if kernel in ("rbf", "gaussian"):
        return np.exp(-gamma * d2)
    if kernel == "iq":
        return 1.0 / (1.0 + gamma * d2)
    if kernel == "imq":
        return 1.0 / np.sqrt(1.0 + gamma * d2)
    raise ValueError(f"unknown kernel {kernel!r} (rbf, iq, imq)")


def _within_offdiag_sum(x: np.ndarray, xn: np.ndarray, gamma: float, kernel: str) -> float:
    n = x.shape[0]
    total = 0.0
    for i in range(0, n, _BLOCK):
        k = _kernel(_sqdist(x[i : i + _BLOCK], x, xn[i : i + _BLOCK], xn), gamma, kernel)
        rows = np.arange(i, min(i + _BLOCK, n))
        k[rows - i, rows] = 0.0  # drop the diagonal (unbiased estimator)
        total += float(k.sum())
    return total


def _cross_mean(x: np.ndarray, xn: np.ndarray, y: np.ndarray, yn: np.ndarray, gamma: float, kernel: str) -> float:
    total = 0.0
    for i in range(0, x.shape[0], _BLOCK):
        total += float(_kernel(_sqdist(x[i : i + _BLOCK], y, xn[i : i + _BLOCK], yn), gamma, kernel).sum())
    return total / (x.shape[0] * y.shape[0])


def kad_reference(
    ref_emb: np.ndarray, bandwidth: float | str = "median", kernel: str = "rbf",
    bandwidth_max_n: int = DEFAULT_BANDWIDTH_MAX_N, seed: int = DEFAULT_SEED,
) -> KadRef:
    x = _as2d(ref_emb, "ref_emb")
    if x.shape[0] < 2:
        raise ValueError("KAD needs >= 2 reference rows")
    bw = median_pairwise_distance(x, bandwidth_max_n, seed) if bandwidth == "median" else float(bandwidth)
    if not bw > 0:
        raise ValueError(f"bandwidth must be > 0, got {bw}")
    xn = np.einsum("ij,ij->i", x, x)
    gamma = 1.0 / (2.0 * bw * bw)
    _kernel(np.zeros(1), gamma, kernel)  # validate the name
    return KadRef(x, xn, bw, kernel, _within_offdiag_sum(x, xn, gamma, kernel))


def kad(
    ref_emb: np.ndarray | KadRef,
    gen_emb: np.ndarray,
    kernel: str = "rbf",
    bandwidth: float | str = "median",
    scale: float = KAD_SCALE,
    bandwidth_max_n: int = DEFAULT_BANDWIDTH_MAX_N,
    seed: int = DEFAULT_SEED,
    estimator: str = "unbiased",
) -> float:
    """Kernel Audio Distance ``scale * MMD^2_u(ref, gen)`` (Chung et al. 2025; module docstring).

    ``bandwidth="median"``: median pairwise distance within the reference set. Pass a
    :class:`KadRef` from :func:`kad_reference` to reuse the reference term (its bandwidth and
    kernel then apply). ``estimator``: ``"unbiased"`` (U-statistic, the paper's KAD) or
    ``"biased"`` (V-statistic)."""
    if estimator not in ("unbiased", "biased"):
        raise ValueError(f"estimator must be 'unbiased' or 'biased', got {estimator!r}")
    ref = ref_emb if isinstance(ref_emb, KadRef) else kad_reference(ref_emb, bandwidth, kernel, bandwidth_max_n, seed)
    kernel = ref.kernel
    y = _as2d(gen_emb, "gen_emb")
    if y.shape[0] < 2:
        raise ValueError("KAD needs >= 2 generated rows")
    if y.shape[1] != ref.x.shape[1]:
        raise ValueError(f"dimension mismatch {ref.x.shape[1]} vs {y.shape[1]}")
    gamma = 1.0 / (2.0 * ref.bandwidth**2)
    yn = np.einsum("ij,ij->i", y, y)
    m, n = ref.x.shape[0], y.shape[0]
    kyy_off = _within_offdiag_sum(y, yn, gamma, kernel)
    kxy = _cross_mean(ref.x, ref.sqnorm, y, yn, gamma, kernel)
    if estimator == "unbiased":
        kxx, kyy = ref.kxx_offdiag_sum / (m * (m - 1)), kyy_off / (n * (n - 1))
    else:  # k(x, x) = 1 for all three kernels
        kxx, kyy = (ref.kxx_offdiag_sum + m) / (m * m), (kyy_off + n) / (n * n)
    return float(scale * (kxx + kyy - 2.0 * kxy))


class KadGen(NamedTuple):
    """Generated-side kernel sums for :func:`kad_from_counts` (bootstrap over clips)."""

    kyy: np.ndarray  # [n, n] kernel matrix of the generated rows
    kxy_col: np.ndarray  # [n] mean kernel value of each generated row to the reference
    kxx: float  # unbiased reference term
    scale: float


def kad_gen(ref: KadRef, gen_emb: np.ndarray, scale: float = KAD_SCALE) -> KadGen:
    y = _as2d(gen_emb, "gen_emb")
    gamma = 1.0 / (2.0 * ref.bandwidth**2)
    yn = np.einsum("ij,ij->i", y, y)
    kyy = _kernel(_sqdist(y, y, yn, yn), gamma, ref.kernel)
    col = np.zeros(y.shape[0])
    for i in range(0, ref.x.shape[0], _BLOCK):
        col += _kernel(_sqdist(ref.x[i : i + _BLOCK], y, ref.sqnorm[i : i + _BLOCK], yn), gamma, ref.kernel).sum(axis=0)
    m = ref.x.shape[0]
    return KadGen(kyy, col / m, ref.kxx_offdiag_sum / (m * (m - 1)), float(scale))


def kad_from_counts(g: KadGen, counts: np.ndarray) -> float:
    """Unbiased KAD of the multiset that holds generated row ``a`` ``counts[a]`` times: equal to
    :func:`kad` on the expanded rows (every pair of distinct positions, ``k(x, x) = 1``)."""
    c = np.asarray(counts, dtype=np.float64)
    n = float(c.sum())
    if n < 2:
        return math.nan
    kyy = (float(c @ g.kyy @ c) - float(c @ np.diag(g.kyy))) / (n * (n - 1))
    kxy = float(c @ g.kxy_col) / n
    return float(g.scale * (g.kxx + kyy - 2.0 * kxy))


def clip_bootstrap(
    ref_stats: GaussianStats, kref: KadRef | None, gen_emb: np.ndarray, owner: np.ndarray,
    n_boot: int = 200, seed: int = DEFAULT_SEED, ci: float = 0.95, min_n: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Clip-level bootstrap CIs of FAD∞ and KAD (module docstring).

    ``owner[i]``: the clip of generated row ``i``. Each resample draws the clips with
    replacement (as many as there are), takes all of their rows, and recomputes FAD∞
    (:func:`fad_inf`, without replacement inside the resample, same ``min_n``) and KAD
    (:func:`kad_from_counts`, exact for the resampled multiset). Returns
    ``{"fad_inf": (lo, hi), "kad": (lo, hi)}``; a resample too small for FAD∞ is skipped and an
    interval needs at least 10 successful resamples (else ``(nan, nan)``)."""
    g = _as2d(gen_emb, "gen_emb")
    owner = np.asarray(owner)
    clips = np.unique(owner)
    rows_of = [np.flatnonzero(owner == c) for c in clips]
    rng = np.random.default_rng(seed)
    kg = kad_gen(kref, g) if kref is not None else None
    fi, kd = [], []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, clips.size, size=clips.size)
        idx = np.concatenate([rows_of[j] for j in pick])
        try:
            fi.append(fad_inf(ref_stats, g[idx], min_n=min_n, replace=False, seed=int(rng.integers(2**31))).score)
        except ValueError:
            pass
        if kg is not None:
            kd.append(kad_from_counts(kg, np.bincount(idx, minlength=g.shape[0])))
    a = (1.0 - ci) / 2.0

    def _q(v: list[float]) -> tuple[float, float]:
        v = [x for x in v if np.isfinite(x)]
        if len(v) < 10:
            return math.nan, math.nan
        lo, hi = np.quantile(v, [a, 1.0 - a])
        return float(lo), float(hi)

    return {"fad_inf": _q(fi), "kad": _q(kd)}


# --------------------------------------------------------------------------- cosine metrics


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nrm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, nrm, out=np.zeros_like(x), where=nrm > 0)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of paired ``[n, d]`` arrays (0 for a zero vector)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape != b.shape:
        raise ValueError(f"paired embeddings differ in shape: {a.shape} vs {b.shape}")
    return np.einsum("ij,ij->i", _unit(a), _unit(b))


def clap_text_audio_score(text_emb: np.ndarray, audio_emb: np.ndarray) -> np.ndarray:
    """Cosine between each audio clip's CLAP embedding and its own prompt's text embedding."""
    return cosine_rows(text_emb, audio_emb)


def region_similarity(gen_region_emb: np.ndarray, orig_region_emb: np.ndarray) -> np.ndarray:
    """Cosine between each regenerated region and the original audio of the same region."""
    return cosine_rows(gen_region_emb, orig_region_emb)


def centred_cosine(a: np.ndarray, b: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Row-wise cosine of ``a - centre`` and ``b - centre`` (``centre``: the reference mean).
    Removes the common offset that makes every MERT pair look similar (anisotropy)."""
    c = np.asarray(centre, dtype=np.float64)
    return cosine_rows(np.atleast_2d(a) - c, np.atleast_2d(b) - c)


def null_threshold(null_max_cos: np.ndarray, q: float = 99.0) -> float:
    """``q``-th percentile of a null distribution of memorisation scores (NaN if empty)."""
    v = np.asarray(null_max_cos, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else math.nan


class Memorisation(NamedTuple):
    max_cos: np.ndarray  # [n_gen]
    nearest: np.ndarray  # [n_gen] index into train_emb
    flagged: np.ndarray  # [n_gen] bool, max_cos > thresh
    thresh: float


def memorisation_check(gen_emb: np.ndarray, train_emb: np.ndarray, thresh: float = 0.95) -> Memorisation:
    """Max cosine of every generated row to any training row, its argmax, and the flag."""
    g = _unit(np.atleast_2d(gen_emb))
    t = _unit(np.atleast_2d(train_emb))
    if g.shape[1] != t.shape[1]:
        raise ValueError(f"dimension mismatch {g.shape[1]} vs {t.shape[1]}")
    best = np.full(g.shape[0], -np.inf)
    arg = np.zeros(g.shape[0], dtype=np.int64)
    for j in range(0, t.shape[0], _BLOCK):
        s = g @ t[j : j + _BLOCK].T
        k = np.argmax(s, axis=1)
        v = s[np.arange(g.shape[0]), k]
        better = v > best
        best[better], arg[better] = v[better], j + k[better]
    return Memorisation(best, arg, best > thresh, float(thresh))
