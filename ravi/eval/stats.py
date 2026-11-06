"""Paired statistics for the per-region comparisons (plan Phase 2.3 go criterion, Phase 4.2).

Every inpainting metric is computed per region (one test-track section), so two conditions are
compared on the regions both have, with a Wilcoxon signed-rank test on the paired differences
``d = a - b``. Definitions (reported in the paper):

* **Test**: :func:`scipy.stats.wilcoxon` (verified against scipy 1.15.3 docs and installed
  1.18.1: ``wilcoxon(x, y=None, zero_method='wilcox', correction=False,
  alternative='two-sided', method='auto', *, axis=0, nan_policy='propagate')``; ``d = x - y``;
  ``alternative='greater'`` means ``d`` is stochastically greater than a distribution symmetric
  about 0). ``zero_method='wilcox'`` discards zero differences (the scipy default), so ties such as
  two identical anchors do not count; ``method='auto'`` is exact for ``len(d) <= 50`` without
  ties/zeros and asymptotic otherwise. A pair with a NaN on either side is dropped before the test
  (``n`` counts the pairs kept). For a 0/1 metric (key agreement) the signed-rank test on
  differences in {-1, 0, 1} reduces to a two-sided sign test on the discordant pairs.
* **Effect size**: the median of the paired differences, ``median(a - b)`` (zeros included), with
  a percentile bootstrap 95% CI: ``n_boot`` resamples of the pairs with replacement, a seeded
  :class:`numpy.random.Generator` (PCG64), so the CI is deterministic.
* **Clustering** (pseudo-replication): a test track contributes several regions, which are not
  independent. With ``clusters`` (the track id of each pair), :func:`cluster_medians` reduces
  the pairs to one per track (the median of ``a`` and of ``b`` over that track's pairs), and the
  *primary* test is the signed-rank test on those per-track medians; the region-level test is
  kept as secondary and its CI comes from a **cluster bootstrap**
  (:func:`cluster_bootstrap_median_ci`: resample tracks with replacement, keep all their pairs,
  median of the pooled differences).
* **Multiplicity**: :func:`holm` (Holm 1979 step-down), applied by the caller within one
  ``(family, metric, level)`` across all condition pairs, separately to the two-sided and to
  each one-sided column (``run_eval``: the go/no-go reads Holm over ``p_a_worse``). The scope is
  per metric: no correction across metrics.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = ["paired_compare", "holm", "bootstrap_median_ci", "cluster_bootstrap_median_ci", "cluster_medians"]

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20261014


def bootstrap_median_ci(
    d: np.ndarray, n_boot: int = DEFAULT_N_BOOT, ci: float = 0.95, seed: int = DEFAULT_SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``median(d)``; ``(nan, nan)`` for an empty ``d``."""
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return math.nan, math.nan
    if d.size == 1:
        return float(d[0]), float(d[0])
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, min(n_boot, 2_000_000 // d.size))
    for i in range(0, n_boot, chunk):
        k = min(chunk, n_boot - i)
        idx = rng.integers(0, d.size, size=(k, d.size))
        meds[i : i + k] = np.median(d[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(meds, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def cluster_bootstrap_median_ci(
    d: np.ndarray, clusters: Sequence[Any] | np.ndarray, n_boot: int = DEFAULT_N_BOOT, ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile CI of ``median(d)`` resampling *clusters* with replacement (all pairs of a
    drawn cluster are kept). ``(nan, nan)`` for an empty ``d``; one cluster gives a degenerate
    interval at the median."""
    d = np.asarray(d, dtype=np.float64)
    cl = np.asarray(clusters, dtype=object)
    if d.size == 0:
        return math.nan, math.nan
    if d.shape != cl.shape:
        raise ValueError(f"clusters {cl.shape} do not match differences {d.shape}")
    _, inv = np.unique(cl.astype(str), return_inverse=True)
    groups = [d[inv == g] for g in range(inv.max() + 1)]
    if len(groups) == 1:
        m = float(np.median(d))
        return m, m
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        meds[i] = np.median(np.concatenate([groups[j] for j in pick]))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(meds, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def cluster_medians(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, clusters: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-cluster medians of ``a`` and ``b`` over the pairs where both are finite (sorted
    cluster ids; clusters without a finite pair are dropped)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    cl = np.asarray(clusters, dtype=object).astype(str).ravel()
    if not (a.shape == b.shape == cl.shape):
        raise ValueError(f"lengths differ: {a.shape}, {b.shape}, {cl.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    ids = sorted(set(cl[keep]))
    am = np.array([np.median(a[keep & (cl == c)]) for c in ids], dtype=np.float64)
    bm = np.array([np.median(b[keep & (cl == c)]) for c in ids], dtype=np.float64)
    return am, bm, ids


def paired_compare(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
    clusters: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Wilcoxon signed-rank on ``d = a - b`` over pairs where both are finite.

    ``clusters`` (one id per pair): the CI is a cluster bootstrap
    (:func:`cluster_bootstrap_median_ci`) and ``n_clusters`` is reported; the test itself is
    unchanged (call it on :func:`cluster_medians` for the cluster-level test).

    Returns ``n`` (pairs kept), ``n_nonzero``, ``median_a``, ``median_b``, ``median_diff``
    (``median(a - b)``), ``mean_diff``, ``ci_low`` / ``ci_high`` (bootstrap CI of the median
    difference), ``statistic`` (the two-sided W), ``p_two_sided``, ``p_greater`` (H1: ``a > b``),
    ``p_less`` (H1: ``a < b``). The p-values are NaN when fewer than one non-zero difference
    remains (scipy cannot rank nothing), and 1.0 is never invented for that case.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in length: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    cl = None
    if clusters is not None:
        cl = np.asarray(clusters, dtype=object).ravel()
        if cl.shape != keep.shape:
            raise ValueError(f"clusters {cl.shape} do not match pairs {keep.shape}")
        cl = cl[keep]
    a, b = a[keep], b[keep]
    d = a - b
    n = int(d.size)
    nz = int(np.count_nonzero(d))
    out: dict[str, Any] = {
        "n": n,
        "n_nonzero": nz,
        "median_a": float(np.median(a)) if n else math.nan,
        "median_b": float(np.median(b)) if n else math.nan,
        "median_diff": float(np.median(d)) if n else math.nan,
        "mean_diff": float(np.mean(d)) if n else math.nan,
        "statistic": math.nan,
        "p_two_sided": math.nan,
        "p_greater": math.nan,
        "p_less": math.nan,
    }
    if cl is None:
        out["ci_low"], out["ci_high"] = bootstrap_median_ci(d, n_boot=n_boot, ci=ci, seed=seed)
        out["ci_method"] = "bootstrap"
    else:
        out["ci_low"], out["ci_high"] = cluster_bootstrap_median_ci(d, cl, n_boot=n_boot, ci=ci, seed=seed)
        out["ci_method"] = "cluster_bootstrap"
        out["n_clusters"] = int(len(set(cl.astype(str))))
    if nz >= 1:
        two = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
        out["statistic"] = float(two.statistic)
        out["p_two_sided"] = float(two.pvalue)
        out["p_greater"] = float(wilcoxon(d, zero_method="wilcox", alternative="greater", method="auto").pvalue)
        out["p_less"] = float(wilcoxon(d, zero_method="wilcox", alternative="less", method="auto").pvalue)
    return out


def holm(pvals: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Sorted ascending, ``p_(i)`` (``i = 1..m``) becomes ``max_{j <= i} min(1, (m - j + 1) p_(j))``.
    NaN entries are passed through and do not count towards ``m``.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out
