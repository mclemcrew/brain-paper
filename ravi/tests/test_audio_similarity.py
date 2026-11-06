import json
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ravi.data import audio_similarity as A

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def brute_sim(a, b, max_lag, min_overlap):
    """Direct port of the reviewer's sim3.py ``sim`` (features already centred)."""
    best = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:, lag:], b) if lag >= 0 else (a, b[:, -lag:])
        n = min(x.shape[1], y.shape[1])
        if n < min_overlap:
            continue
        x, y = x[:, :n], y[:, :n]
        best = max(best, (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    return best if np.isfinite(best) else np.nan


def test_vectorised_similarity_matches_brute_force():
    rng = np.random.default_rng(0)
    feats = [rng.standard_normal((12, int(n))).astype(np.float32) for n in rng.integers(5, 60, size=9)]
    base = rng.standard_normal((12, 50))
    feats += [base[:, 3:].astype(np.float32), base[:, :40].astype(np.float32)]  # one lagged copy pair
    S = A.pairwise_similarity(feats, max_lag=4, min_overlap=12)
    for i in range(len(feats)):
        assert np.isnan(S[i, i])
        for j in range(len(feats)):
            if i != j:
                want = brute_sim(feats[i], feats[j], 4, 12)
                if np.isnan(want):
                    assert np.isnan(S[i, j])
                else:
                    assert S[i, j] == pytest.approx(want, abs=1e-6)
    assert S[-1, -2] == pytest.approx(1.0)


def test_lag_window_is_three_seconds():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((12, 200))
    within = [base[:, :150], base[:, 12:162]]  # 3 s at 4 fps
    beyond = [base[:, :150], base[:, 16:166]]  # 4 s
    assert A.pairwise_similarity(within)[0, 1] == pytest.approx(1.0)
    assert A.pairwise_similarity(beyond)[0, 1] < 0.3


def test_short_overlap_is_nan():
    rng = np.random.default_rng(2)
    S = A.pairwise_similarity([rng.standard_normal((12, 8)), rng.standard_normal((12, 100))])
    assert np.isnan(S[0, 1])


def test_chroma_peaks_at_pitch_class():
    t = np.arange(A.SR * 4) / A.SR
    y = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    c = A.chroma_features(y)
    assert c.shape[0] == 12 and abs(c.shape[1] - 4 * A.FPS) <= 1
    assert np.all(np.argmax(c[:, 1:-1], axis=0) == 9)  # librosa chroma starts at C; A = 9
    assert np.allclose(c.mean(axis=0), 0.0, atol=1e-5)  # per-frame mean removed


def test_similar_pairs_and_merge_links():
    df = pd.DataFrame({"track_id": ["a", "b", "c", "d"], "duration_s": [30.0, 31.0, 30.0, 5.0]})
    S = np.array([[np.nan, 0.9, 0.75, 0.2], [0.9, np.nan, 0.6, 0.71], [0.75, 0.6, np.nan, 0.1], [0.2, 0.71, 0.1, np.nan]])
    pairs = A.similar_pairs(df, S, families=[0, 0, 1, 2], threshold=0.70)
    assert pairs[["track_id_a", "track_id_b"]].values.tolist() == [["a", "c"], ["b", "d"]]  # a-b same family
    assert pairs.similarity.tolist() == [0.75, 0.71]
    hand = [["c", "a", 0.75, "listened: same mix"]]
    links = A.merge_links(hand, pairs, "auto")
    assert links == [["c", "a", 0.75, "listened: same mix"], ["b", "d", 0.71, "auto"]]


def test_links_roundtrip(tmp_path):
    p = tmp_path / "manual_links.json"
    A.write_links(p, [["x y", "z", 0.8, "r"]])
    assert json.loads(p.read_text()) == {"pairs": [["x y", "z", 0.8, "r"]]}
    assert A.read_links(p) == [["x y", "z", 0.8, "r"]]
    p.write_text('{"pairs": [["only-one"]]}')
    with pytest.raises(ValueError):
        A.read_links(p)


@needs_ffmpeg
def test_decode_and_cache(tmp_path, monkeypatch):
    t = np.arange(48000 * 5) / 48000
    paths = []
    for k, f in enumerate((440.0, 523.25)):
        p = tmp_path / f"{k}.wav"
        sf.write(p, np.stack([0.2 * np.sin(2 * np.pi * f * t)] * 2, 1), 48000)
        paths.append(str(p))
    assert len(A.decode_mono(paths[0])) == 5 * A.SR
    df = pd.DataFrame({"track_id": ["t0", "t1"], "audio_path": paths})
    cache = tmp_path / "chroma_cache.npz"
    f1 = A.load_features(df, cache, workers=1)
    assert cache.exists()

    def boom(_):
        raise AssertionError("cache not used")

    monkeypatch.setattr(A, "track_features", boom)
    f2 = A.load_features(df, cache, workers=1)
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))
    S = A.pairwise_similarity(f2)
    assert S[0, 1] < 0.5  # A vs C tone


def test_filter_links_short_clip_rule():
    dur = {"long1": 30.0, "long2": 31.0, "sting": 4.0, "sting2": 7.9, "alt": 30.0}
    base = {"long1": 0, "long2": 1, "sting": 2, "sting2": 3, "alt": 3}  # sting2 + alt share a title/[ALT] family
    links = [
        ["long1", "long2", 0.71, "auto"],    # both >= 8 s -> kept
        ["long1", "sting", 0.79, "auto"],    # short clip, < 0.80, no support -> dropped
        ["long2", "sting", 0.80, "auto"],    # short clip but >= 0.80 -> kept
        ["sting2", "alt", 0.72, "auto"],     # short clip, supported by a non-audio link -> kept
    ]
    kept, dropped = A.filter_links(links, dur, base, min_clip_s=8.0, short_clip_sim=0.80)
    assert [p[:2] for p in kept] == [["long2", "sting"], ["sting2", "alt"], ["long1", "long2"]]
    assert len(dropped) == 1 and dropped[0][:4] == ["long1", "sting", 0.79, "auto"]
    assert "4.0 s < 8 s" in dropped[0][4]
    kept2, dropped2 = A.filter_links(links, dur, base, min_clip_s=3.0, short_clip_sim=0.80)
    assert len(kept2) == 4 and not dropped2


def test_regeneration_keeps_hand_entries_and_drops_stale_auto(tmp_path):
    p = tmp_path / "manual_links.json"
    A.write_links(
        p,
        [["a", "b", 0.9, "listened: same mix"], ["c", "d", 0.75, A.auto_reason(0.7)]],
        [["e", "f", 0.72, "reviewer flagged", "dropped: short"], ["g", "h", 0.71, A.auto_reason(0.7), "dropped"]],
    )
    assert A.read_links(p) == [["a", "b", 0.9, "listened: same mix"], ["c", "d", 0.75, A.auto_reason(0.7)]]
    # auto entries are regenerated from the CSV, hand entries (kept or dropped) are carried over
    assert A.hand_links(p) == [["a", "b", 0.9, "listened: same mix"], ["e", "f", 0.72, "reviewer flagged"]]
    p.write_text('{"pairs": [], "dropped": [["a", "b", 0.7, "r"]]}')
    with pytest.raises(ValueError):
        A.read_links(p)
