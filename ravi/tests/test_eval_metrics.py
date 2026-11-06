"""Tests for ravi.eval.{metrics,seams,stats,embed,run_eval} on synthetic signals.

No model downloads: CLAP/MERT are replaced by a deterministic band-energy extractor that goes
through the real windowing / pooling / caching code of :class:`ravi.eval.embed.Extractor`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy import linalg
from scipy.stats import wilcoxon

from ravi.eval import embed as E
from ravi.eval import metrics as M
from ravi.eval import run_eval as R
from ravi.eval import seams as S
from ravi.eval import stats as ST

SR = 22050
C4, E4, G4, F4, A4, B4, D5, C5, A3 = 261.63, 329.63, 392.0, 349.23, 440.0, 493.88, 587.33, 523.25, 220.0


# --------------------------------------------------------------------------- signals


def chord(freqs, dur, sr=SR, amp=0.1):
    t = np.arange(int(round(dur * sr))) / sr
    y = np.zeros_like(t)
    for f in freqs:
        for k in (1, 2, 3, 4):
            y += amp / k * np.sin(2 * np.pi * f * k * t)
    return y.astype(np.float32)


def clicks(segments, sr=SR):
    out = []
    for bpm, dur in segments:
        y = np.zeros(int(dur * sr), np.float32)
        per = int(round(sr * 60 / bpm))
        burst = (np.hanning(200) * np.sin(2 * np.pi * 1000 * np.arange(200) / sr)).astype(np.float32)
        for i in range(0, len(y) - 200, per):
            y[i : i + 200] += burst
        out.append(y)
    return np.concatenate(out)


C_PROG = np.concatenate([chord([C4, E4, G4], 2), chord([F4, A4, C5], 2), chord([G4, B4, D5], 2), chord([C4, E4, G4], 2)])


# --------------------------------------------------------------------------- FAD


def test_fad_identical_sets_is_zero():
    x = np.random.default_rng(0).normal(size=(300, 12))
    assert M.fad(x, x) == pytest.approx(0.0, abs=1e-8)


def test_fad_increases_with_mean_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(2000, 8))
    base = rng.normal(size=(2000, 8))
    vals = [M.fad(ref, base + s) for s in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # the mean term dominates: shift s in all 8 dims adds ~8 s^2
    assert vals[-1] - vals[0] == pytest.approx(8 * 4.0, rel=0.05)


def test_frechet_matches_scipy_sqrtm_full_rank():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(80, 6)), 1.5 * rng.normal(size=(90, 6)) + 0.3
    sx, sy = M.gaussian_stats(x), M.gaussian_stats(y)
    covmean = linalg.sqrtm(sx.cov @ sy.cov).real
    ref = (sx.mu - sy.mu) @ (sx.mu - sy.mu) + np.trace(sx.cov) + np.trace(sy.cov) - 2 * np.trace(covmean)
    assert M.fad(x, y) == pytest.approx(float(ref), rel=1e-9)


def test_frechet_rank_deficient_is_real_and_nonnegative():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(10, 40)), rng.normal(size=(12, 40))  # n < d: singular covariances
    v = M.fad(x, y)
    assert np.isfinite(v) and v >= 0
    assert M.fad(x, x) == pytest.approx(0.0, abs=1e-8)


def test_fad_accepts_precomputed_reference_stats():
    rng = np.random.default_rng(4)
    ref, gen = rng.normal(size=(500, 5)), rng.normal(size=(200, 5)) + 0.2
    assert M.fad(M.gaussian_stats(ref), gen) == pytest.approx(M.fad(ref, gen), rel=1e-12)


def _gauss_trials(d=16, n=400, shift=0.25, trials=12):
    mu = np.full(d, shift)
    ref = M.GaussianStats(np.zeros(d), np.eye(d), 10**9)  # the exact reference distribution
    gens = [np.random.default_rng(100 + s).normal(size=(n, d)) + mu for s in range(trials)]
    return ref, gens, float(mu @ mu)


def test_fad_inf_with_replacement_recovers_full_sample_fad():
    """fadtk's bootstrap (rows drawn with replacement) extrapolates to the full-sample FAD."""
    ref, gens, _ = _gauss_trials()
    res = [M.fad_inf(ref, g, replace=True, seed=i, min_n=200) for i, g in enumerate(gens)]
    inf = np.array([r.score for r in res])
    full = np.array([r.fad_full for r in res])
    assert np.mean(inf) == pytest.approx(np.mean(full), rel=0.03)
    assert np.median(np.abs(inf - full) / full) < 0.15


def test_fad_inf_without_replacement_removes_sample_size_bias():
    ref, gens, true = _gauss_trials()
    res = [M.fad_inf(ref, g, replace=False, seed=i, n_draws=3, min_n=200) for i, g in enumerate(gens)]
    inf = np.array([r.score for r in res])
    full = np.array([r.fad_full for r in res])
    assert np.mean(full) - true > 0.15  # plain FAD at n=400, d=16 is biased upwards (~1.16 vs 1.0)
    assert abs(np.mean(inf) - true) < 0.5 * (np.mean(full) - true)


def test_fad_inf_sizes_and_validation():
    g = np.random.default_rng(5).normal(size=(50, 3))
    r = M.fad_inf(g, g + 0.5, sizes=[10, 20, 30, 40, 50])
    assert [p[0] for p in r.points] == [10, 20, 30, 40, 50]
    assert r.replace is False and (r.n_lo, r.n_hi, r.d) == (10, 50, 3)
    with pytest.raises(ValueError):
        M.fad_inf(g, g, sizes=[10, 60])  # > N without replacement
    with pytest.raises(ValueError):
        M.fad_inf(g, g[:3])
    a = M.fad_inf(g, g + 1, seed=7, min_n=20)
    assert a.score == M.fad_inf(g, g + 1, seed=7, min_n=20).score  # deterministic
    # default grid floor max(3 d, 500): 50 rows are not enough
    assert M.fad_inf_min_n(3) == 500 and M.fad_inf_min_n(64) == 500 and M.fad_inf_min_n(768) == 2304
    with pytest.raises(ValueError, match="grid floor"):
        M.fad_inf(g, g + 1)
    big = np.random.default_rng(6).normal(size=(800, 3))
    r = M.fad_inf(big, big + 0.3)
    assert (r.n_lo, r.n_hi) == (500, 800)


def test_pca_fit_on_reference_only():
    rng = np.random.default_rng(11)
    lam = np.array([9.0, 4.0, 1.0, 0.25, 0.01, 0.01])
    ref = rng.normal(size=(4000, 6)) * np.sqrt(lam) + 3.0
    p = M.pca_fit(ref, 3)
    assert p.k == 3 and p.d == 6 and p.components.shape == (6, 3)
    assert np.allclose(p.components.T @ p.components, np.eye(3), atol=1e-10)
    assert p.explained == pytest.approx(14.0 / lam.sum(), rel=0.02)
    z = M.pca_project(p, ref)
    assert np.allclose(z.mean(0), 0, atol=1e-10)
    v = z.var(0, ddof=1)
    assert v[0] > v[1] > v[2] and v[0] == pytest.approx(9.0, rel=0.08)
    gen = rng.normal(size=(50, 6)) + 10.0  # projecting another set does not refit
    assert np.allclose(M.pca_project(p, gen), (gen - p.mean) @ p.components)
    assert M.pca_fit(ref[:3], 64).k == 2  # capped at n - 1
    with pytest.raises(ValueError):
        M.pca_project(p, gen[:, :5])


def test_pca_fad_inf_recovers_true_value_in_768d():
    """M1: 768-d Gaussians with a power-law spectrum that differ only in the leading 16
    directions. The full-dimension FAD-inf over n < d extrapolates far off (with R² ~0.99); PCA to
    64 dims fitted on the reference, grid floor 500, lands within 15% of the true FAD."""
    d, k = 768, 64
    lam = 1.0 / (1 + np.arange(d)) ** 0.8
    sc = np.ones(d)
    sc[:16] = 1.5
    mu = np.zeros(d)
    mu[:8] = 0.2
    true = float(mu @ mu + (lam * (1 - np.sqrt(sc)) ** 2).sum())
    rng = np.random.default_rng(0)
    pca_vals, naive_vals = [], []
    for trial in range(2):
        ref = rng.normal(size=(8000, d)) * np.sqrt(lam)
        gen = rng.normal(size=(2000, d)) * np.sqrt(lam * sc) + mu
        naive = M.fad_inf(ref, gen, min_n=500, seed=trial)  # the old min(500, N // 2) grid
        p = M.pca_fit(ref, k)
        red = M.fad_inf(M.pca_project(p, ref), M.pca_project(p, gen), seed=trial)
        assert red.d == k and red.n_lo == M.fad_inf_min_n(k) == 500
        pca_vals.append(red.score)
        naive_vals.append(naive.score)
        assert abs(red.score - true) / true < 0.15
        assert abs(naive.score - true) / true > 0.15 and naive.r2 > 0.95  # wrong, with a good-looking fit
    assert abs(np.mean(pca_vals) - true) / true < 0.15


# --------------------------------------------------------------------------- KAD


def test_kad_identical_sets():
    x = np.random.default_rng(6).normal(size=(60, 8))
    assert M.kad(x, x, estimator="biased") == pytest.approx(0.0, abs=1e-12)
    ref = M.kad_reference(x)
    gamma = 1 / (2 * ref.bandwidth**2)
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    k = np.exp(-gamma * d2)
    off = (k.sum() - np.trace(k)) / (60 * 59)
    expected = -2 * M.KAD_SCALE * (1 - off) / 60  # U-statistic on identical sets (docstring)
    assert M.kad(x, x) == pytest.approx(expected, rel=1e-9)
    assert abs(M.kad(x, x)) < 2 * M.KAD_SCALE / 60


def test_kad_nonnegative_and_grows_with_shift():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(300, 8))
    kr = M.kad_reference(ref)
    vals = [M.kad(kr, rng.normal(size=(300, 8)) + s) for s in (0.0, 0.5, 1.0, 2.0)]
    assert abs(vals[0]) < 0.5 and all(v >= 0 for v in vals[1:])
    assert all(b > a for a, b in zip(vals, vals[1:]))
    for s in range(5):  # the V-statistic is >= 0 always
        y = np.random.default_rng(50 + s).normal(size=(40, 8)) * (1 + s / 4)
        assert M.kad(ref, y, estimator="biased") >= 0


def test_kad_bandwidth_from_reference_and_reuse():
    rng = np.random.default_rng(8)
    ref, gen = rng.normal(size=(200, 4)), 3 * rng.normal(size=(150, 4))
    kr = M.kad_reference(ref)
    assert kr.bandwidth == pytest.approx(M.median_pairwise_distance(ref))
    assert M.kad(kr, gen) == pytest.approx(M.kad(ref, gen), rel=1e-12)
    # a manual RBF MMD^2_u with sigma from the reference
    g = 1 / (2 * kr.bandwidth**2)
    kxx = np.exp(-g * ((ref[:, None] - ref[None]) ** 2).sum(-1))
    kyy = np.exp(-g * ((gen[:, None] - gen[None]) ** 2).sum(-1))
    kxy = np.exp(-g * ((ref[:, None] - gen[None]) ** 2).sum(-1))
    m, n = len(ref), len(gen)
    mmd = (kxx.sum() - m) / (m * (m - 1)) + (kyy.sum() - n) / (n * (n - 1)) - 2 * kxy.mean()
    assert M.kad(kr, gen) == pytest.approx(100 * mmd, rel=1e-9)
    with pytest.raises(ValueError):
        M.kad(ref, gen, kernel="nope")


def test_kad_from_counts_equals_expanded_rows():
    rng = np.random.default_rng(12)
    ref, gen = rng.normal(size=(120, 5)), rng.normal(size=(40, 5)) + 0.3
    kr = M.kad_reference(ref)
    kg = M.kad_gen(kr, gen)
    assert M.kad_from_counts(kg, np.ones(40)) == pytest.approx(M.kad(kr, gen), rel=1e-10)
    idx = rng.integers(0, 40, size=60)  # a multiset with duplicates
    assert M.kad_from_counts(kg, np.bincount(idx, minlength=40)) == pytest.approx(M.kad(kr, gen[idx]), rel=1e-10)


def _clips(rng, n_clips, per_clip, d, shift):
    """Clip-structured embeddings: a clip offset shared by its windows plus window noise."""
    rows = [rng.normal(size=d) * 0.8 + shift + 0.3 * rng.normal(size=(per_clip, d)) for _ in range(n_clips)]
    return np.concatenate(rows), np.repeat(np.arange(n_clips), per_clip)


def test_clip_bootstrap_ci():
    rng = np.random.default_rng(13)
    ref, _ = _clips(rng, 200, 10, 4, 0.0)
    rs, kr = M.gaussian_stats(ref), M.kad_reference(ref)
    near, own = _clips(rng, 30, 10, 4, 0.2)
    far, own2 = _clips(rng, 30, 10, 4, 1.5)
    ci_near = M.clip_bootstrap(rs, kr, near, own, n_boot=100, seed=1, min_n=40)
    ci_far = M.clip_bootstrap(rs, kr, far, own2, n_boot=100, seed=1, min_n=40)
    assert ci_near == M.clip_bootstrap(rs, kr, near, own, n_boot=100, seed=1, min_n=40)  # seeded
    for ci in (ci_near, ci_far):
        assert all(np.isfinite(ci["fad_inf"])) and ci["fad_inf"][0] < ci["fad_inf"][1]
    assert ci_near["kad"][1] < ci_far["kad"][0] and ci_near["fad_inf"][1] < ci_far["fad_inf"][0]
    point = M.kad(kr, far)
    assert ci_far["kad"][0] <= point * 1.2 and ci_far["kad"][1] >= point * 0.8
    # resampling clips (not windows) gives a wider interval than treating windows as independent
    win_owner = np.arange(far.shape[0])
    ci_win = M.clip_bootstrap(rs, kr, far, win_owner, n_boot=100, seed=1, min_n=40)
    assert np.diff(ci_far["kad"])[0] > 1.5 * np.diff(ci_win["kad"])[0]


# --------------------------------------------------------------------------- cosine metrics


def test_cosine_metrics_paired():
    a = np.array([[1.0, 0, 0], [0, 2.0, 0], [1, 1, 0]])
    b = np.array([[2.0, 0, 0], [0, 0, 1.0], [-1, -1, 0]])
    assert M.region_similarity(a, b) == pytest.approx([1.0, 0.0, -1.0])
    assert M.clap_text_audio_score(a, a) == pytest.approx([1, 1, 1])
    with pytest.raises(ValueError):
        M.region_similarity(a, b[:2])


def test_memorisation_check_finds_copy():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(5000, 16))
    gen = rng.normal(size=(4, 16))
    gen[2] = train[4321] * 3.0  # a scaled copy: cosine 1
    gen[3] = train[17] + 0.05 * rng.normal(size=16)
    res = M.memorisation_check(gen, train, thresh=0.95)
    assert res.nearest[2] == 4321 and res.max_cos[2] == pytest.approx(1.0)
    assert res.nearest[3] == 17 and res.flagged[3]
    assert not res.flagged[0] and not res.flagged[1]
    assert res.max_cos[0] < 0.95


def test_null_threshold_and_centred_cosine():
    assert M.null_threshold(np.arange(101.0)) == pytest.approx(99.0)
    assert np.isnan(M.null_threshold(np.array([np.nan])))
    c = np.array([10.0, 10.0, 0.0])
    a, b = c + np.array([1.0, 0, 0]), c + np.array([0, 1.0, 0])
    assert M.cosine_rows(a[None], b[None])[0] > 0.99  # the shared offset dominates the raw cosine
    assert M.centred_cosine(a, b, c)[0] == pytest.approx(0.0, abs=1e-12)
    assert M.centred_cosine(a, a, c)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- seams


def test_checkerboard_kernel():
    k = S.checkerboard_kernel(5)
    assert k.shape == (10, 10)
    assert np.abs(k).sum() == pytest.approx(1.0)
    assert k[0, 0] > 0 and k[0, -1] < 0 and k.sum() == pytest.approx(0.0, abs=1e-12)


def test_seam_score_detects_hard_cut():
    a = chord([C4, E4, G4], 10)
    b = chord([330.0, 415.3, 523.25], 10)
    smooth = np.concatenate([a, a])
    cut = np.concatenate([a, b])
    s_smooth = S.seam_score(smooth, SR, [10.0])[0]
    s_cut = S.seam_score(cut, SR, [10.0])[0]
    assert s_cut > 0.2 and abs(s_smooth) < 0.05 and s_cut > s_smooth + 0.2
    # away from the seam the cut track has no peak
    assert S.seam_score(cut, SR, [5.0])[0] < 0.05
    # novelty frame rate
    assert S.novelty_curve(cut, SR).shape[0] == pytest.approx(200, abs=2)


def test_seam_delta_zero_for_identical_and_positive_for_new_seam():
    a, b = chord([C4, E4, G4], 8), chord([330.0, 415.3, 523.25], 8)
    orig = np.concatenate([a, a])
    gen = np.concatenate([a, b])
    assert S.seam_delta(orig, orig, SR, [8.0]) == pytest.approx([0.0], abs=1e-12)
    assert S.seam_delta(gen, orig, SR, [8.0])[0] > 0.2
    # a real boundary in the original is subtracted
    assert S.seam_delta(gen, gen, SR, [8.0]) == pytest.approx([0.0], abs=1e-12)
    # stereo input is downmixed
    st = np.stack([orig, orig], axis=1)
    assert S.seam_score(st, SR, [8.0]) == pytest.approx(S.seam_score(orig, SR, [8.0]))


def test_beat_regularity_click_track_vs_tempo_jump():
    steady = clicks([(120, 20)])
    jump = clicks([(120, 10), (80, 10)])
    cv_steady = S.beat_regularity(steady, SR, [10.0], tempo="local")[0]
    cv_jump = S.beat_regularity(jump, SR, [10.0], tempo="local")[0]
    assert cv_steady < 0.05
    assert cv_jump > 0.1 and cv_jump > 3 * cv_steady
    # the global-tempo tracker (the default) keeps a constant period across the jump: the IBI CV
    # cannot see it, which is why the beat-grid agreement is the primary metric
    assert S.DEFAULT_TEMPO == "global"
    assert S.beat_regularity(jump, SR, [10.0])[0] < 0.05
    # fewer than 3 beats in the window -> NaN
    assert np.isnan(S.beat_regularity(steady, SR, [10.0], beats=np.array([9.0, 10.0]))[0])
    # metrical level does not matter for the CV
    b = np.arange(0, 20, 0.5)
    assert S.beat_regularity(steady, SR, [10.0], beats=b)[0] == pytest.approx(
        S.beat_regularity(steady, SR, [10.0], beats=b[::2])[0], abs=1e-12)


def test_beat_grid_agreement_on_beat_lists():
    orig = np.arange(0.0, 20.0, 0.5)  # 120 BPM
    agree, off = S.beat_grid_agreement(None, None, SR, [10.0], gen_beats=orig, orig_beats=orig)
    assert agree[0] == 1.0 and off[0] == 0.0
    shifted = orig + 0.03  # within the 70 ms tolerance
    agree, off = S.beat_grid_agreement(None, None, SR, [10.0], gen_beats=shifted, orig_beats=orig)
    assert agree[0] == 1.0 and off[0] == pytest.approx(0.03)
    # phase jump of 250 ms at the seam: the beats after it (half of the window) miss
    jump = np.where(orig >= 10.0, orig + 0.25, orig)
    agree, off = S.beat_grid_agreement(None, None, SR, [10.0, 3.0], gen_beats=jump, orig_beats=orig)
    assert agree[0] == pytest.approx(8 / 17) and off[0] == pytest.approx(0.25 * 9 / 17)
    assert agree[1] == 1.0  # far from the jump
    # tempo change 120 -> 80 BPM at the seam
    slow = np.concatenate([orig[orig < 10.0], np.arange(10.0, 20.0, 0.75)])
    assert S.beat_grid_agreement(None, None, SR, [10.0], gen_beats=slow, orig_beats=orig)[0][0] < 0.7
    # no generated beats: nothing kept; too few orig beats: NaN
    agree, off = S.beat_grid_agreement(None, None, SR, [10.0], gen_beats=np.zeros(0), orig_beats=orig)
    assert agree[0] == 0.0 and np.isnan(off[0])
    assert np.isnan(S.beat_grid_agreement(None, None, SR, [10.0], gen_beats=orig, orig_beats=np.array([9.9, 10.1]))[0][0])


def test_beat_grid_agreement_on_audio():
    steady = clicks([(120, 20)])
    jump = clicks([(120, 10), (80, 10)])
    a_same, _ = S.beat_grid_agreement(steady, steady, SR, [10.0])
    a_jump, off_jump = S.beat_grid_agreement(jump, steady, SR, [10.0])
    phase = np.concatenate([clicks([(120, 10)]), np.zeros(int(0.25 * SR), np.float32), clicks([(120, 10)])])[: steady.size]
    a_phase, _ = S.beat_grid_agreement(phase, steady, SR, [10.0])
    assert a_same[0] == 1.0
    assert a_jump[0] < 0.75 and off_jump[0] > 0.03  # tempo jump: the loose tracker follows the new onsets
    assert a_phase[0] < 0.6
    # with librosa's default tightness the DP holds the 120 BPM period through the jump
    bpm = S.global_tempo(steady, SR)
    rigid = S.beat_grid_agreement(None, None, SR, [10.0], gen_beats=S.grid_beats(jump, SR, bpm, 100.0),
                                  orig_beats=S.grid_beats(steady, SR, bpm, 100.0))[0]
    assert rigid[0] > 0.95


def test_estimate_key_on_chords():
    k = S.estimate_key(C_PROG, SR)
    assert k.key == "C major" and k.confidence >= S.KEY_CONF_GATE and k.r_best >= S.KEY_MIN_R
    am = S.estimate_key(chord([A3, C4, E4], 4), SR)
    assert am.key == "A minor"
    assert S.estimate_key(np.zeros(SR * 2, np.float32), SR).key is None


def test_key_agreement_major_vs_minor_triad():
    assert S.key_agreement(chord([C4, E4, G4], 4), C_PROG, SR)[0] is True
    agree, conf = S.key_agreement(chord([A3, C4, E4], 4), C_PROG, SR)
    assert agree is False and conf >= 1.1
    # gated: noise context has no reliable key
    noise = (0.1 * np.random.default_rng(0).standard_normal(SR * 4)).astype(np.float32)
    assert S.key_agreement(C_PROG, noise, SR)[0] is None
    # gated by the ratio alone: a lone triad context is below 1.1
    agree, conf = S.key_agreement(C_PROG, chord([C4, E4, G4], 4), SR, conf_gate=10.0)
    assert agree is None


# --------------------------------------------------------------------------- stats


def test_paired_compare_known_data():
    rng = np.random.default_rng(10)
    b = rng.normal(size=40)
    a = b + 1.0 + 0.1 * rng.normal(size=40)
    r = ST.paired_compare(a, b, n_boot=2000)
    assert r["n"] == 40 and r["n_nonzero"] == 40
    assert r["median_diff"] == pytest.approx(np.median(a - b))
    # the median of 40 N(1, 0.1^2) differences has SE ~1.25 * 0.1 / sqrt(40) = 0.020: 95% CI ~ +-0.04
    half = (r["ci_high"] - r["ci_low"]) / 2
    assert r["ci_low"] <= r["median_diff"] <= r["ci_high"] and 0.015 < half < 0.06
    assert r["ci_low"] < 1.0 + 0.02 and r["ci_high"] > 1.0 - 0.02 and r["ci_method"] == "bootstrap"
    assert r["p_greater"] < 1e-6 and r["p_less"] > 0.99
    assert r["p_two_sided"] == pytest.approx(wilcoxon(a - b).pvalue)
    assert r["p_greater"] == pytest.approx(wilcoxon(a, b, alternative="greater").pvalue)
    # small exact case: d = [1, 2, 3, 4, 5] -> one-sided exact p = 1/32
    r = ST.paired_compare([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert r["p_greater"] == pytest.approx(1 / 32) and r["p_two_sided"] == pytest.approx(2 / 32)


def test_paired_compare_nan_and_ties():
    r = ST.paired_compare([1.0, np.nan, 3.0, 2.0], [1.0, 5.0, np.nan, 2.0])
    assert r["n"] == 2 and r["n_nonzero"] == 0
    assert np.isnan(r["p_two_sided"]) and r["median_diff"] == 0.0
    with pytest.raises(ValueError):
        ST.paired_compare([1, 2], [1])
    a = np.arange(30.0)
    assert ST.paired_compare(a + 1, a, n_boot=500) == ST.paired_compare(a + 1, a, n_boot=500)


def test_cluster_medians_and_cluster_bootstrap():
    a = np.array([1.0, 3.0, np.nan, 10.0, 12.0, 0.0])
    b = np.array([0.0, 0.0, 5.0, 1.0, 1.0, np.nan])
    cl = ["t1", "t1", "t1", "t2", "t2", "t3"]
    am, bm, ids = ST.cluster_medians(a, b, cl)
    assert ids == ["t1", "t2"] and am.tolist() == [2.0, 11.0] and bm.tolist() == [0.0, 1.0]
    # 8 tracks x 6 regions with a strong track effect: the cluster CI is wider than the iid one
    rng = np.random.default_rng(14)
    track_eff = rng.normal(size=8)
    d = np.repeat(track_eff, 6) + 0.05 * rng.normal(size=48)
    clusters = np.repeat([f"t{i}" for i in range(8)], 6)
    lo_c, hi_c = ST.cluster_bootstrap_median_ci(d, clusters, n_boot=2000)
    lo_i, hi_i = ST.bootstrap_median_ci(d, n_boot=2000)
    assert hi_c - lo_c > 1.3 * (hi_i - lo_i)
    assert ST.cluster_bootstrap_median_ci(d, clusters, n_boot=500) == ST.cluster_bootstrap_median_ci(d, clusters, n_boot=500)
    r = ST.paired_compare(d, np.zeros(48), n_boot=500, clusters=clusters)
    assert r["ci_method"] == "cluster_bootstrap" and r["n_clusters"] == 8
    assert ST.cluster_bootstrap_median_ci(np.array([1.0, 2.0]), ["x", "x"]) == (1.5, 1.5)


def test_holm():
    adj = ST.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    adj = ST.holm([0.5, np.nan, 0.01])
    assert adj[0] == pytest.approx(0.5) and np.isnan(adj[1]) and adj[2] == pytest.approx(0.02)
    assert ST.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # 2 x 0.8 capped at 1, then monotone
    assert ST.holm([0.04, 0.03]) == pytest.approx([0.06, 0.06])


# --------------------------------------------------------------------------- embed (fake model)


class FakeExtractor(E.Extractor):
    """Log band energies of each window: deterministic, cheap, spectrally meaningful."""

    name = "fake"
    sr = 8000
    normalize_pooled = True

    def __init__(self, cache_dir=None, window_s=2.0, hop_s=1.0, tag="fake"):
        super().__init__(window_s, hop_s, cache_dir, batch_size=4)
        self.name = tag
        self.calls = 0
        self.text_calls = 0

    @property
    def dim(self):
        return 8

    def _embed_windows(self, windows):
        self.calls += 1
        lens = {w.shape[0] for w in windows}
        assert len(lens) == 1, "windows in one batch must share a length"
        out = []
        for w in windows:
            spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[0])))
            v = np.log1p(np.array([b.mean() for b in np.array_split(spec, 8)]))
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out)

    def embed_text(self, texts):
        self.text_calls += 1
        out = []
        for t in texts:
            v = np.zeros(8)
            for i, ch in enumerate(t.encode()):
                v[(ch + i) % 8] += 1
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out)


def test_window_starts():
    assert E.window_starts(5, 10, 3) == [0]
    assert E.window_starts(10, 10, 3) == [0]
    assert E.window_starts(25, 10, 5) == [0, 5, 10, 15]
    assert E.window_starts(27, 10, 5) == [0, 5, 10, 15, 17]
    with pytest.raises(ValueError):
        E.window_starts(0, 10, 5)


def test_extractor_windows_pool_and_cache(tmp_path):
    ex = FakeExtractor(cache_dir=tmp_path / "cache")
    t = np.arange(int(5.5 * 16000)) / 16000
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ch = ex.embed_chunks([y], sr=16000)[0]
    assert ch.shape == (5, 8)  # starts 0,1,2,3 s + an end-aligned window at 3.5 s
    calls = ex.calls
    again = ex.embed_chunks([y], sr=16000)[0]
    assert ex.calls == calls and np.array_equal(ch, again)  # served from the cache
    wav = tmp_path / "x.wav"
    sf.write(wav, np.stack([y, y], 1), 16000, subtype="FLOAT")
    from_file = ex.embed([wav])
    assert from_file.shape == (1, 8) and np.linalg.norm(from_file[0]) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(from_file[0], ex.embed([y], sr=16000)[0], atol=1e-5)
    files = list((tmp_path / "cache" / ex.cache_id).glob("*.npz"))
    assert len(files) == 2  # one array key, one file key
    other = FakeExtractor(cache_dir=tmp_path / "cache", hop_s=0.5)
    assert other.cache_id != ex.cache_id
    with pytest.raises(ValueError):
        ex.embed([y])  # array without sr


def test_lufs_normalise_matches_prepare_audio():
    from ravi.data.prepare_audio import normalise

    rng = np.random.default_rng(15)
    quiet = (0.02 * rng.standard_normal((5 * 44100, 2))).astype(np.float32)
    loud = np.clip(0.9 * rng.standard_normal((5 * 44100, 2)), -1, 1).astype(np.float32)  # peak-limited
    for x in (quiet, loud, _track(3).astype(np.float32)):
        out, st = E.lufs_normalise(x, 44100)
        ref, rst = normalise(x, 44100, -16.0, -1.0)
        assert np.array_equal(out, ref)
        assert st["lufs_in"] == pytest.approx(rst["lufs_in"]) and st["peak_limited"] == rst["peak_limited"]
        assert st["true_peak_dbtp"] <= -1.0 + 1e-6
    _, st = E.lufs_normalise(quiet, 44100)
    assert st["lufs_out"] == pytest.approx(-16.0, abs=0.05) and not st["peak_limited"]
    silent, st = E.lufs_normalise(np.zeros((44100, 2), np.float32), 44100)
    assert st["silent"] and not silent.any()
    tiny, st = E.lufs_normalise(quiet[:1000], 44100)  # shorter than one gating block
    assert st["silent"] and np.array_equal(tiny, quiet[:1000])


def test_clap_quantise_matches_laion_and_is_idempotent():
    k = np.arange(-32767, 32768).astype(np.int16)
    levels = (k / 32767.0).astype(np.float32)  # laion_clap int16_to_float32
    assert np.array_equal(E.clap_quantise(levels), levels)  # a second quantisation is a no-op
    x = np.random.default_rng(16).uniform(-1.2, 1.2, 1000).astype(np.float32)
    q = E.clap_quantise(x)
    expect = ((np.clip(x, -1, 1) * 32767.0).astype(np.int16) / 32767.0).astype(np.float32)
    assert np.array_equal(q, expect)


class _FakeClapModel:
    def __init__(self):
        self.calls = []
        self.tokenize = lambda t: {"input_ids": [0] + list(range(len(t.split()))) + [2]}

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        self.calls.append((x, use_tensor))
        return np.stack([np.r_[np.mean(w), np.std(w), np.zeros(510)] + np.eye(512)[2] for w in x]).astype(np.float32)


def test_clap_extractor_stub_model():
    ex = E.ClapExtractor(batch_size=8)
    ex._model = _FakeClapModel()
    t = np.arange(int(25 * 44100)) / 44100
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ch = ex.embed_chunks([(y, 44100)])[0]
    assert ch.shape == (16, 512)  # 10 s windows, 1 s hop over 25 s
    wins = [w for x, _ in ex._model.calls for w in x]
    assert all(use is False for _, use in ex._model.calls)  # numpy path: laion quantises too
    assert len(wins) == 16 and all(w.dtype == np.float32 and w.shape == (480000,) for w in wins)  # 48 kHz x 10 s
    for w in wins:
        assert np.allclose(w * 32767.0, np.round(w * 32767.0), atol=1e-3)  # int16-quantised
    assert ex.text_truncated(["short prompt", " ".join(["w"] * 76), " ".join(["w"] * 75)]) == [False, True, False]


def test_mert_extractor_stub_forward():
    seen = []

    def fake_forward(batch):
        seen.append(batch.copy())
        b, n = batch.shape
        frames = n // 320
        base = batch[:, : frames * 320].reshape(b, frames, 320).std(axis=2, keepdims=True)
        return [np.broadcast_to(base + layer, (b, frames, 768)) for layer in range(13)]

    ex = E.MertExtractor()
    ex._forward = fake_forward
    rng = np.random.default_rng(17)
    w = rng.standard_normal(5 * 24000).astype(np.float32)
    out = ex._embed_windows([w, 0.1 * w + 0.3, rng.standard_normal(5 * 24000).astype(np.float32)])
    x = seen[0]
    assert x.shape == (3, 120000)
    assert np.allclose(x.mean(axis=1), 0, atol=1e-5) and np.allclose(x.var(axis=1), 1, atol=1e-4)  # per window
    assert np.allclose(x[0], x[1], atol=1e-4)  # gain and offset invariant: each window z-scored alone
    assert out.shape == (3, 768) and np.allclose(out[0], out[1], atol=1e-4)
    assert 12.0 < out[0, 0] < 13.5  # layer 12 (the stub adds the layer index)
    ex4 = E.MertExtractor(layer=4)
    ex4._forward = fake_forward
    assert 4.0 < ex4._embed_windows([w])[0, 0] < 5.5
    assert np.allclose(E.wav2vec2_normalise(w), (w - w.mean()) / np.sqrt(w.var() + 1e-7))
    ex._forward = lambda batch: [np.zeros((1, 2, 768))] * 12
    with pytest.raises(RuntimeError, match="13 hidden states"):
        ex._embed_windows([w])


def test_cache_identity_and_tmp_names(tmp_path):
    clap = E.ClapExtractor()
    cfg = clap.config()
    assert cfg["ckpt_sha256"] == E.CLAP_SHA256 and cfg["revision"] == E.CLAP_HF_REVISION
    assert set(cfg["versions"]) == {"torch", "laion_clap", "transformers"}
    assert set(E.MertExtractor().config()["versions"]) == {"torch", "transformers"}
    ck = tmp_path / "other.pt"
    ck.write_bytes(b"weights-a")
    a = E.ClapExtractor(ckpt=ck)
    ck2 = tmp_path / "other2.pt"
    ck2.write_bytes(b"weights-b")
    b = E.ClapExtractor(ckpt=ck2)
    assert a.config()["ckpt_sha256"] == E.sha256_file(ck) and a.config()["revision"] is None
    assert a.cache_id != b.cache_id != clap.cache_id
    t1, t2 = E._tmp_name(tmp_path / "k.npz"), E._tmp_name(tmp_path / "k.npz")
    assert t1 != t2 and str(os.getpid()) in t1.name and t1.suffix == ".npz"
    ex = FakeExtractor(cache_dir=tmp_path / "c")
    ex.embed_chunks([np.ones(16000, np.float32)], sr=16000)
    assert not list((tmp_path / "c").rglob("*.tmp*"))


def test_real_extractor_configs_without_models():
    clap = E.ClapExtractor()
    mert = E.MertExtractor(layer=4)
    assert clap.sr == 48000 and clap.window_s == 10.0 and clap.dim == 512
    assert mert.sr == 24000 and mert.window_s == 5.0 and mert.config()["layer"] == 4
    assert clap.cache_id.startswith("clap-music-") and mert.cache_id.startswith("mert-v1-95m-")
    with pytest.raises(ValueError):
        E.ClapExtractor(window_s=12.0)
    with pytest.raises(ValueError):
        E.MertExtractor(layer=13)
    with pytest.raises(ValueError):
        E.get_extractor("vggish")


# --------------------------------------------------------------------------- run_eval end to end

EVAL_SR = 16000
C0_S = 2.0  # the context window starts 2 s into the track (sidecar context.start_s != 0)
MASK_CTX = (4.0, 9.0)  # mask in context seconds; the section is the mask minus 1 s padding
FAMILIES = {"t0": 1, "t1": 1, "t2": 2}  # t0's built-in donor must skip t1 (same alt_family)


def _track(seed: int, dur: float = 14.0) -> np.ndarray:
    """A short stereo 'song': four 3.5 s chords with a soft click pulse."""
    roots = [[C4, E4, G4], [F4, A4, C5], [G4, B4, D5], [A3, C4, E4]]
    rng = np.random.default_rng(seed)
    order = rng.permutation(4)
    y = np.concatenate([chord([f * (1 + 0.02 * seed) for f in roots[i]], dur / 4, sr=EVAL_SR) for i in order])
    y = y + 0.5 * clicks([(100 + 10 * seed, dur)], sr=EVAL_SR)[: y.size]
    return np.stack([y, 0.9 * y], 1).astype(np.float32)


def _write_region(d: Path, stem: str, ctx: np.ndarray, gen: np.ndarray, m0: int, m1: int, track: str, idx: int,
                  sha: str, c0: int):
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / f"{stem}_orig.wav", ctx, EVAL_SR, subtype="FLOAT")
    sf.write(d / f"{stem}_spliced.wav", gen, EVAL_SR, subtype="FLOAT")
    sf.write(d / f"{stem}_raw.wav", gen, EVAL_SR, subtype="FLOAT")
    t = lambda n: (n + c0) / EVAL_SR  # noqa: E731 - context sample -> track seconds
    meta = {
        "tool": "ravi.eval.sa3_infer", "command": "inpaint", "track_id": track, "audio": {"sha256": sha},
        "section": {"index": idx, "label": "verse", "start_s": t(m0) + 1, "end_s": t(m1) - 1},
        "mask": {"start_s": t(m0), "end_s": t(m1), "start_sample": m0 + c0, "end_sample": m1 + c0},
        "context": {"start_s": c0 / EVAL_SR, "end_s": t(ctx.shape[0]), "start_sample": c0, "end_sample": c0 + ctx.shape[0]},
        "mask_in_context": {"start_s": m0 / EVAL_SR, "end_s": m1 / EVAL_SR, "start_sample": m0, "end_sample": m1},
    }
    (d / f"{stem}.json").write_text(json.dumps(meta))


def region_stem(tid: str, seed: int) -> str:
    return f"{tid}_sec1_verse_{MASK_CTX[0] + C0_S:.1f}-{MASK_CTX[1] + C0_S:.1f}_seed{seed}"


def _layout(tmp_path: Path) -> dict[str, Path]:
    from ravi.eval.masks import splice

    root = tmp_path / "eval"
    test_audio = tmp_path / "test_audio"
    train = tmp_path / "train"
    for p in (test_audio, train):
        p.mkdir(parents=True)
    tracks = [f"t{i}" for i in range(3)]
    rng = np.random.default_rng(0)
    c0 = int(C0_S * EVAL_SR)
    m0, m1 = int(MASK_CTX[0] * EVAL_SR), int(MASK_CTX[1] * EVAL_SR)
    regions_doc = {"tool": "ravi.eval.select_regions", "tracks": {}}
    for i, tid in enumerate(tracks):
        x = _track(i)
        sf.write(test_audio / f"{tid}.wav", x, EVAL_SR, subtype="FLOAT")
        regions_doc["tracks"][tid] = {"alt_family": FAMILIES[tid], "captions": {"B": f"rvkr␞calm chords {i}␞Pop"}}
        ctx = x[c0:]
        for seed, amp in ((1, 0.2), (2, 0.3)):
            stem = region_stem(tid, seed)
            noise = (amp * rng.standard_normal((m1 - m0, 2))).astype(np.float32)
            near = ctx[m0:m1] + (0.01 * seed * rng.standard_normal((m1 - m0, 2))).astype(np.float32)
            _write_region(root / "base" / "inpaint", stem, ctx, splice(ctx, noise, m0, m1, 800), m0, m1, tid, 1, f"sha-{tid}", c0)
            _write_region(root / "B" / "inpaint", stem, ctx, splice(ctx, near, m0, m1, 800), m0, m1, tid, 1, f"sha-{tid}", c0)
    (tmp_path / "regions_test.json").write_text(json.dumps(regions_doc))
    for i in range(4):
        sf.write(train / f"tr{i}.wav", _track(10 + i), EVAL_SR, subtype="FLOAT")
    (tmp_path / "splits.json").write_text(json.dumps({"train": [f"tr{i}" for i in range(4)], "val": [], "test": tracks}))
    for cond in ("base", "B"):
        g = root / cond / "gen"
        g.mkdir(parents=True)
        for k, (tid, seed) in enumerate((t, s) for s in (1, 2) for t in tracks):
            y = _track(20 + k, 6.0) if cond == "B" else (rng.standard_normal((6 * EVAL_SR, 2)) * 0.3).astype(np.float32)
            stem = f"{tid}_seed{seed}"
            sf.write(g / f"{stem}.wav", y, EVAL_SR, subtype="FLOAT")
            (g / f"{stem}.json").write_text(json.dumps({"track_id": tid, "prompt": f"rvkr␞calm chords {tid}",
                                                         "prompt_rendered": f"rvkr, calm chords {tid}", "seed": 100 + k}))
    return {"root": root, "test_audio": test_audio, "train": train, "caps": tmp_path / "regions_test.json",
            "splits": tmp_path / "splits.json"}


def _argv(p: dict[str, Path], *extra: str) -> list[str]:
    return ["--eval-root", str(p["root"]), "--test-audio", str(p["test_audio"]), "--train-dir", str(p["train"]),
            "--splits", str(p["splits"]), "--test-captions", str(p["caps"]), "--extractors", "clap,mert",
            "--n-boot", "200", "--pca-dims", "4", "--fad-inf-min-n", "6", "--dist-boot", "30", *extra]


class FakeClapLike(FakeExtractor):
    def text_truncated(self, texts):
        return [len(t.split()) > 3 for t in texts]


def _factory(made: dict):
    def f(name, args):
        cls = FakeClapLike if name == "clap" else FakeExtractor
        ex = cls(cache_dir=args.cache_dir, tag=f"fake-{name}")
        made[name] = ex
        return ex
    return f


def test_run_eval_end_to_end(tmp_path, monkeypatch):
    p = _layout(tmp_path)
    made: dict = {}
    fad_inf_calls = []
    real_fad_inf = M.fad_inf

    def spy(*a, **kw):
        fad_inf_calls.append(kw.get("replace", "missing"))
        return real_fad_inf(*a, **kw)

    monkeypatch.setattr(R.M, "fad_inf", spy)
    assert R.main(_argv(p), extractor_factory=_factory(made)) == 0
    assert fad_inf_calls and all(r is False for r in fad_inf_calls)  # every FAD-inf call: without replacement
    monkeypatch.setattr(R.M, "fad_inf", real_fad_inf)
    root = p["root"]
    reg = pd.read_parquet(root / "metrics_regions.parquet")
    assert sorted(reg["condition"].unique()) == ["B", "base", "lower", "upper"]
    assert len(reg) == 3 * (2 + 2 + 1 + 1)  # base/B per seed, anchors once per region
    for c in ("region_sim_mert", "region_sim_mert_centred", "seam_delta_start", "seam_delta_end", "abs_seam_delta",
              "beat_grid_agree", "beat_grid_offset_ms", "beat_cv", "beat_cv_delta", "key_conf_orig", "key_agree",
              "lufs_gen_region", "lufs_orig_region", "region_key", "seed"):
        assert c in reg.columns, c
    # context start != 0: the mask is read in context samples
    assert np.allclose(reg["mask_in_context_start_s"], MASK_CTX[0]) and np.allclose(reg["mask_start_s"], MASK_CTX[0] + C0_S)
    up = reg[reg.condition == "upper"]
    assert np.allclose(up["region_sim_clap"], 1.0) and np.allclose(up["seam_delta"], 0.0)
    assert np.allclose(up["abs_seam_delta"], 0.0) and np.allclose(up["beat_grid_agree"], 1.0)
    assert np.allclose(up["beat_grid_offset_ms"], 0.0) and np.allclose(up["beat_cv_delta"].fillna(0.0), 0.0)
    # per-seam: the noise fill makes both seams sharper than the original at the same position
    base = reg[reg.condition == "base"]
    assert (base["seam_score_start"] > base["seam_score_orig_start"] + 0.02).all()
    assert (base["seam_score_end"] > base["seam_score_orig_end"] + 0.02).all()
    assert np.allclose(base["seam_delta"], (base["seam_delta_start"] + base["seam_delta_end"]) / 2)
    assert np.allclose(up["seam_score_start"], up["seam_score_orig_start"]) and np.allclose(up["seam_score_end"], up["seam_score_orig_end"])
    by = reg.set_index(["condition", "region_id"])
    for t in ("t0", "t1", "t2"):
        for s in (1, 2):
            assert by.loc[("B", region_stem(t, s)), "region_sim_clap"] > by.loc[("base", region_stem(t, s)), "region_sim_clap"]
    low = reg[reg.condition == "lower"].set_index("track_id")
    assert low.loc["t0", "lower_donor"].startswith("t2") and low.loc["t1", "lower_donor"].startswith("t2")
    assert low.loc["t2", "lower_donor"].startswith("t0") and low["lower_family_checked"].all()

    avg = pd.read_parquet(root / "metrics_regions_avg.parquet")
    a = avg.set_index(["condition", "region_key"])
    k0 = region_stem("t0", 1).rsplit("_seed", 1)[0]
    assert a.loc[("base", k0), "n_seeds"] == 2 and a.loc[("upper", k0), "n_seeds"] == 1
    two = reg[(reg.condition == "base") & (reg.track_id == "t0")]["region_sim_clap"]
    assert a.loc[("base", k0), "region_sim_clap"] == pytest.approx(two.mean())

    gen = pd.read_parquet(root / "metrics_gen.parquet")
    assert sorted(gen["condition"].unique()) == ["B", "base", "test_orig"]
    assert gen["clap_text"].notna().all() and gen["lufs_in"].notna().all()
    assert {"memo_max_clap", "memo_max_mert", "memo_flag_clap", "memo_flag_null_clap", "prompt_truncated"} <= set(gen.columns)
    assert "memo_flag_mert" not in gen.columns  # flags on CLAP only
    assert set(gen["memo_track_clap"]) <= {f"tr{i}" for i in range(4)}
    assert gen.groupby("condition")["in_dist_subset"].sum().to_dict() == {"B": 3, "base": 3, "test_orig": 3}

    summ = pd.read_csv(root / "metrics_summary.csv").set_index("condition")
    assert list(summ.index) == ["base", "B", "upper", "lower", "test_orig"]
    for c in ("fad_inf_clap", "fad_inf_clap_ci_lo", "fad_inf_clap_ci_hi", "fad_inf_r2_mert", "fad_inf_n_lo_clap",
              "fad_inf_n_hi_clap", "pca_d_clap", "kad_mert", "kad_mert_ci_lo", "region_sim_clap_mean",
              "clap_text_mean", "memo_null_p99_clap", "memo_flagged_null_clap", "n_clips_dist", "n_seeds_max"):
        assert c in summ.columns, c
    assert not [c for c in summ.columns if c.startswith("fad_") and not c.startswith("fad_inf")]  # no plain FAD
    assert summ.loc["base", "n_regions"] == 3 and summ.loc["base", "n_region_outputs"] == 6 and summ.loc["base", "n_gen"] == 6
    assert (summ.loc[["base", "B", "test_orig"], "n_clips_dist"] == 3).all()
    assert bool(summ.loc["base", "dist_subsampled"]) and not bool(summ.loc["test_orig", "dist_subsampled"])
    assert summ.loc["B", "pca_d_clap"] == 4 and summ.loc["B", "fad_inf_n_lo_clap"] == 6
    assert summ.loc["B", "kad_clap"] < summ.loc["base", "kad_clap"]  # B's clips look like the corpus
    assert summ.loc["B", "n_prompt_truncated"] == 6  # "rvkr, calm chords tX" > 3 words in the fake tokenizer
    dist = pd.read_parquet(root / "metrics_dist.parquet").set_index("condition")
    assert {"fad_clap_naive", "fad_mert_naive", "fad_clap_pca_full_sample"} <= set(dist.columns)
    assert dist.loc["B", "fad_clap_naive"] < dist.loc["base", "fad_clap_naive"]

    comp = pd.read_csv(root / "comparisons.csv")
    rs = comp[(comp.metric == "region_sim_clap")]
    assert set(rs.level) == {"track", "region"}
    tr = rs[rs.level == "track"]
    pairs = {(a, b) for a, b in zip(tr.cond_a, tr.cond_b)}
    assert ("B", "base") in pairs and ("upper", "base") in pairs and ("lower", "base") in pairs
    assert ("B", "upper") in pairs and ("upper", "lower") in pairs and len(tr) == 6
    row = tr[(tr.cond_a == "B") & (tr.cond_b == "base")].iloc[0]
    assert row.n == 3 and row.median_diff > 0 and row.p_a_better == pytest.approx(1 / 8) and bool(row.primary)
    rrow = rs[(rs.level == "region") & (rs.cond_a == "B") & (rs.cond_b == "base")].iloc[0]
    assert rrow.n == 3 and rrow.ci_method == "cluster_bootstrap" and not bool(rrow.primary)
    assert set(comp.metric) >= {"region_sim_clap", "region_sim_mert_centred", "seam_delta", "abs_seam_delta",
                                "beat_grid_agree", "beat_cv", "clap_text"}
    for c in ("p_holm_two_sided", "p_holm_a_worse", "p_holm_a_better", "holm_m", "holm_scope"):
        assert c in comp.columns
    assert "p_holm" not in comp.columns
    g = tr.sort_values("p_a_worse")
    assert np.allclose(g["p_holm_a_worse"], ST.holm(g["p_a_worse"].to_numpy()))
    t2m = comp[comp.metric == "clap_text"]
    assert len(t2m) == 1 and t2m.iloc[0].cond_a == "B" and t2m.iloc[0].n == 3 and t2m.iloc[0].level == "clip"
    meta = json.loads((root / "metrics_meta.json").read_text())
    assert meta["reference"]["clap"]["n_files"] == 4 and meta["reference"]["clap"]["pca_d"] == 4
    assert meta["fad_inf"]["replace"] is False and meta["fad_inf"]["fit"]["B"]["clap"]["pca_d"] == 4
    assert meta["distribution_metrics"]["matched_n_clips"] == 3 and meta["lower_anchor"]["source"] == "built-in"
    assert meta["memorisation"]["null_p99"]["clap"] == pytest.approx(
        np.percentile(gen[gen.condition == "test_orig"]["memo_max_clap"], 99))

    # deterministic, and the second run is served from the embedding cache
    first = {f: (root / f).read_bytes() for f in ("metrics_summary.csv", "comparisons.csv")}
    calls = {k: v.calls for k, v in made.items()}
    made2: dict = {}
    assert R.main(_argv(p), extractor_factory=_factory(made2)) == 0
    assert {f: (root / f).read_bytes() for f in first} == first
    assert all(v.calls == 0 for v in made2.values()) and calls["clap"] > 0
    assert pd.read_parquet(root / "metrics_regions.parquet").equals(reg)


def test_compare_all_direction_mapping():
    """Every metric's better/worse mapping: condition X is better than base in the metric's
    direction on all 8 tracks, so p_a_better is small and p_a_worse large at both levels."""
    import argparse

    expected = {"region_sim_clap": +1, "region_sim_mert": +1, "region_sim_mert_centred": +1, "seam_delta": -1,
                "abs_seam_delta": -1, "beat_grid_agree": +1, "beat_grid_offset_ms": -1, "beat_cv": -1,
                "beat_cv_delta": -1, "key_agree": +1, "struct_kept": +1, "struct_kept_delta": +1}
    assert R.REGION_METRICS == expected and R.GEN_METRICS == {"clap_text": +1}
    args = argparse.Namespace(n_boot=200, seed=0)
    rng = np.random.default_rng(18)
    for metrics, family in ((R.REGION_METRICS, "inpaint"), (R.GEN_METRICS, "t2m")):
        for metric, direction in metrics.items():
            rows = []
            for k in range(8):
                v = float(rng.uniform(0, 1))
                for cond, val in (("base", v), ("X", v + direction * (0.5 + 0.1 * k))):
                    rows.append({"condition": cond, "key": f"r{k}", "track_id": f"t{k}", metric: val})
            out = R.compare_all(["base", "X"], pd.DataFrame(rows), "key", {metric: direction}, family, "base", args,
                                cluster_col="track_id")
            assert {r["level"] for r in out} == {"track", "region"}
            for r in out:
                assert (r["cond_a"], r["cond_b"]) == ("X", "base")
                assert r["p_a_better"] < 0.01 and r["p_a_worse"] > 0.99, (metric, r["level"])
                assert np.sign(r["median_diff"]) == direction and r["higher_is_better"] == (direction > 0)


def test_run_eval_limit_and_guards(tmp_path):
    p = _layout(tmp_path)
    out = tmp_path / "out"
    assert R.main(_argv(p, "--limit", "1", "--no-anchors", "--out-dir", str(out)), extractor_factory=_factory({})) == 0
    reg = pd.read_parquet(out / "metrics_regions.parquet")
    assert sorted(reg["condition"]) == ["B", "B", "base", "base"] and reg["track_id"].nunique() == 1
    gen = pd.read_parquet(out / "metrics_gen.parquet")
    assert (gen.groupby("condition").size() == 1).all()
    # --train-dir without --splits
    argv = _argv(p)
    i = argv.index("--splits")
    with pytest.raises(SystemExit, match="needs --splits"):
        R.main(argv[:i] + argv[i + 2 :], extractor_factory=_factory({}))
    # test captions: missing, empty, and the other two sources
    i = argv.index("--test-captions")
    with pytest.raises(SystemExit, match="--test-captions is required"):
        R.main(argv[:i] + argv[i + 2 :], extractor_factory=_factory({}))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no captions found"):
        R.main(argv[:i] + ["--test-captions", str(empty)] + argv[i + 2 :], extractor_factory=_factory({}))
    with pytest.raises(SystemExit, match="no such file"):
        R.load_test_captions(str(tmp_path / "nope.jsonl"), "B")
    jl = tmp_path / "midasheng_test.jsonl"
    jl.write_text("\n".join(json.dumps({"track_id": f"t{k}", "caption": f"calm chords {k}"}) for k in range(3)) + "\n")
    assert R.load_test_captions(str(jl), "B") == {f"t{k}": f"calm chords {k}" for k in range(3)}
    d = tmp_path / "capdir"
    d.mkdir()
    (d / "t0.txt").write_text("rvkr␞calm␞Pop", encoding="utf-8")
    assert R.load_test_captions(str(d), "B") == {"t0": "rvkr, calm, Pop"}
    assert R.load_test_captions(str(p["caps"]), "B")["t1"] == "rvkr, calm chords 1, Pop"
    with pytest.raises(SystemExit, match="no captions found"):
        R.load_test_captions(str(p["caps"]), "Z")
    # a source-audio mismatch between conditions for the same region
    js = p["root"] / "B" / "inpaint" / f"{region_stem('t1', 2)}.json"
    meta = json.loads(js.read_text())
    meta["audio"]["sha256"] = "different"
    js.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="audio.sha256"):
        R.main(_argv(p), extractor_factory=_factory({}))
    meta["audio"]["sha256"] = "sha-t1"
    js.write_text(json.dumps(meta))
    # a training id that is also a test id is refused
    sp = json.loads(p["splits"].read_text())
    sp["train"].append("t0")
    p["splits"].write_text(json.dumps(sp))
    with pytest.raises(SystemExit, match="also in val/test"):
        R.main(_argv(p), extractor_factory=_factory({}))
    # a reserved condition name is refused
    (p["root"] / "upper" / "inpaint").mkdir(parents=True)
    with pytest.raises(SystemExit, match="reserved"):
        R.main(_argv(p), extractor_factory=_factory({}))


def test_run_eval_strength_sweep_and_mask_fallback(tmp_path):
    p = _layout(tmp_path)
    root = p["root"]
    # B's sweep: move inpaint/ files into strength0.5 and strength1
    src = root / "B" / "inpaint"
    files = sorted(src.iterdir())
    for s in ("strength0.5", "strength1"):
        (src / s).mkdir()
        for f in files:
            (src / s / f.name).write_bytes(f.read_bytes())
    for f in files:
        f.unlink()
    # drop mask_in_context from one base sidecar: falls back to mask - context start (2 s here)
    js = sorted((root / "base" / "inpaint").glob("*.json"))[0]
    meta = json.loads(js.read_text())
    meta.pop("mask_in_context")
    js.write_text(json.dumps(meta))
    conds = [c.name for c in R.discover_conditions(root)]
    assert conds == ["B", "B@strength0.5", "B@strength1", "base"]  # B's gen/ is not swept
    regs = R.read_regions(root / "base" / "inpaint", "spliced")
    r = regs[js.stem]
    assert (r.m0, r.m1) == (int(MASK_CTX[0] * EVAL_SR), int(MASK_CTX[1] * EVAL_SR))
    assert r.section_bounds_ctx_s == pytest.approx((MASK_CTX[0] + 1, MASK_CTX[1] - 1))
    assert r.seed == 1 and r.region_key == js.stem.rsplit("_seed", 1)[0]
    assert R.main(_argv(p, "--extractors", "clap"), extractor_factory=_factory({})) == 0
    summ = pd.read_csv(root / "metrics_summary.csv")
    assert list(summ["condition"])[:4] == ["base", "B", "B@strength0.5", "B@strength1"]


def _write_struct(path: Path, bounds: list[float], dur: float = 12.0):
    edges = [0.0] + list(bounds) + [dur]
    segs = [{"start": a, "end": b, "label": "x"} for a, b in zip(edges, edges[1:])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"track_id": path.stem, "tool": "songformer", "bpm": None, "beats": None,
                                "downbeats": None, "segments": segs, "duration": dur}))


def test_run_eval_grid_lower_anchor_and_structure(tmp_path):
    """An existing anchor_lower condition is used instead of the built-in lower anchor, and
    --structure-dir docs give struct_kept on the evaluated audio and on _orig."""
    import shutil

    p = _layout(tmp_path)
    root = p["root"]
    shutil.copytree(root / "base" / "inpaint", root / "anchor_lower" / "inpaint")
    sd = tmp_path / "structure"
    sec = [MASK_CTX[0] + 1, MASK_CTX[1] - 1]  # the human section boundaries, in context seconds
    for t in ("t0", "t1", "t2"):
        for s in (1, 2):
            stem = region_stem(t, s)
            for cond in ("base", "B", "anchor_lower"):
                _write_struct(sd / cond / "inpaint" / f"{stem}_orig.json", sec)
            _write_struct(sd / "B" / "inpaint" / f"{stem}_spliced.json", [sec[0] + 1.0, sec[1] + 2.5])
            _write_struct(sd / "base" / "inpaint" / f"{stem}_spliced.json", [1.0, 11.5])  # 4 s and 3.5 s off
    assert R.main(_argv(p, "--structure-dir", str(sd), "--extractors", "clap"), extractor_factory=_factory({})) == 0
    reg = pd.read_parquet(root / "metrics_regions.parquet")
    assert sorted(reg["condition"].unique()) == ["B", "anchor_lower", "base", "upper"]
    by = reg.groupby("condition")
    assert np.allclose(by.get_group("B")["struct_kept"], 1.0)
    assert np.allclose(by.get_group("base")["struct_kept"], 0.0) and np.allclose(by.get_group("base")["struct_kept_delta"], -1.0)
    assert np.allclose(by.get_group("upper")["struct_kept"], 1.0) and np.allclose(by.get_group("upper")["struct_kept_delta"], 0.0)
    assert np.allclose(reg["struct_kept_orig"], 1.0) and by.get_group("anchor_lower")["struct_kept"].isna().all()
    summ = pd.read_csv(root / "metrics_summary.csv").set_index("condition")
    assert list(summ.index) == ["base", "B", "upper", "anchor_lower", "test_orig"]
    assert bool(summ.loc["anchor_lower", "is_anchor"])
    meta = json.loads((root / "metrics_meta.json").read_text())
    assert meta["lower_anchor"] == {"source": "condition", "name": "anchor_lower", "donors": {}}
    comp = pd.read_csv(root / "comparisons.csv")
    assert {"struct_kept", "struct_kept_delta"} <= set(comp.metric)
