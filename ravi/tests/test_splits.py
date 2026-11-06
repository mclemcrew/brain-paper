import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ravi.data.splits import (
    GENRES,
    SEED,
    _apportion,
    check_links,
    check_splits,
    coarse_genre,
    eligible_mask,
    make_splits,
)

MANIFESTS = Path(__file__).resolve().parents[2] / "manifests"

TAGS = ["Rock", "Electronic", "Folk", "Ambient", "Jazz", "Cinematic", "Pop", "No Audio"]


def synthetic_manifest(n_fam=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fam in range(n_fam):
        size = int(rng.choice([1, 1, 1, 1, 2, 3, 6]))
        tag = TAGS[int(rng.integers(len(TAGS)))]
        for k in range(size):
            junk = tag == "No Audio"
            has = (not junk) and rng.random() < 0.95
            nseg = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.05, 0.3, 0.3, 0.2, 0.1])) if has else 0
            rows.append(
                dict(
                    track_id=f"{fam:04d}_{k}",
                    alt_family=fam,
                    human_genre_tag=tag,
                    caption_is_junk=junk,
                    has_labels=has,
                    n_segments=nseg,
                    n_interior=int(rng.integers(1, nseg - 1)) if nseg >= 3 and rng.random() < 0.9 else 0,
                    duration_s=float(rng.uniform(15, 90)),
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [20261014, 1, 2, 3])
def test_split_invariants(seed):
    df = synthetic_manifest(seed=seed)
    s = make_splits(df, seed=seed)
    check_splits(df, s)  # also run inside make_splits
    split_of = {t: k for k in ("train", "val", "test") for t in s[k]}
    fam_split = df.assign(split=df.track_id.map(split_of)).groupby("alt_family").split.nunique()
    assert (fam_split == 1).all()
    el = dict(zip(df.track_id, eligible_mask(df)))
    assert all(el[t] for t in s["test"] + s["val"])
    # every family in val/test is wholly eligible
    for k in ("val", "test"):
        fams = set(df[df.track_id.isin(s[k])].alt_family)
        assert eligible_mask(df[df.alt_family.isin(fams)]).all()
    n = len(df)
    assert abs(len(s["test"]) - round(0.15 * n)) <= 2
    assert abs(len(s["val"]) - round(0.05 * n)) <= 2
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == n


def test_split_is_deterministic():
    df = synthetic_manifest()
    assert make_splits(df, seed=7) == make_splits(df, seed=7)
    assert make_splits(df, seed=7)["test"] != make_splits(df, seed=8)["test"]


def test_check_splits_catches_family_leak():
    df = pd.DataFrame(dict(
        track_id=["a", "b", "c"], alt_family=[0, 0, 1], has_labels=True, n_segments=4,
        n_interior=1, duration_s=30.0, human_genre_tag="Rock", caption_is_junk=False,
    ))
    with pytest.raises(AssertionError, match="span"):
        check_splits(df, {"train": ["a", "c"], "val": [], "test": ["b"]})


def test_check_splits_catches_ineligible_test():
    df = pd.DataFrame(dict(
        track_id=["a", "b"], alt_family=[0, 1], has_labels=[True, False], n_segments=[4, 0],
        n_interior=[1, 0], duration_s=30.0, human_genre_tag="Rock", caption_is_junk=False,
    ))
    with pytest.raises(AssertionError, match="ineligible"):
        check_splits(df, {"train": ["a"], "val": [], "test": ["b"]})


def test_family_with_one_ineligible_member_stays_in_train():
    df = synthetic_manifest()
    bad_fam = df.alt_family.max() + 1
    extra = pd.DataFrame([
        dict(track_id="x1", alt_family=bad_fam, human_genre_tag="Rock", caption_is_junk=False,
             has_labels=True, n_segments=5, n_interior=2, duration_s=40.0),
        dict(track_id="x2", alt_family=bad_fam, human_genre_tag="Rock", caption_is_junk=False,
             has_labels=True, n_segments=5, n_interior=2, duration_s=10.0),  # too short
    ])
    df = pd.concat([df, extra], ignore_index=True)
    for seed in range(5):
        s = make_splits(df, seed=seed)
        assert {"x1", "x2"} <= set(s["train"])


@pytest.mark.parametrize(
    "tag,bucket",
    [("Rock", "rock"), ("Indie Rock", "rock"), ("Rocksteady", "pop/other"), ("Electro-Rock", "rock"),
     ("Chillwave", "electronic"), ("Folktronica", "electronic"), ("Folk", "folk/acoustic"),
     ("Latin Jazz", "jazz"), ("Contemporary Classical", "classical/cinematic"), ("Cinematic", "classical/cinematic"),
     ("Dark Ambient", "ambient"), ("Pop", "pop/other"), ("No Audio", "pop/other"), ("", "unknown")],
)
def test_coarse_genre(tag, bucket):
    assert coarse_genre(tag) == bucket
    assert coarse_genre(tag, junk=True) == "unknown"
    assert bucket in GENRES


def test_apportion_respects_caps():
    q = _apportion(10, {"a": 0.5, "b": 0.5}, {"a": 2, "b": 100})
    assert q == {"a": 2, "b": 8}
    assert sum(_apportion(10, {"a": 1, "b": 1}, {"a": 3, "b": 3}).values()) == 6


def test_check_links_catches_split_pair():
    with pytest.raises(AssertionError, match="manual links span"):
        check_links({"train": ["a"], "val": [], "test": ["b"]}, [["a", "b", 0.9, "x"]])
    check_links({"train": ["a", "b"], "val": [], "test": []}, [["a", "b", 0.9, "x"]])


# --- golden tests on the committed manifest ------------------------------------

committed = pytest.mark.skipif(
    not (MANIFESTS / "tracks.parquet").exists() or not (MANIFESTS / "splits.json").exists(),
    reason="committed manifests not present",
)


@pytest.fixture(scope="module")
def committed_manifest():
    return pd.read_parquet(MANIFESTS / "tracks.parquet"), json.loads((MANIFESTS / "splits.json").read_text())


@committed
def test_committed_splits_reproduced_from_committed_manifest(committed_manifest):
    df, splits = committed_manifest
    assert splits["seed"] == SEED
    assert make_splits(df, seed=SEED) == splits


@committed
def test_committed_splits_no_leakage(committed_manifest):
    df, splits = committed_manifest
    check_splits(df, splits)
    links = json.loads((MANIFESTS / "manual_links.json").read_text())["pairs"]
    assert links, "manual_links.json is empty"
    check_links(splits, links)
    fam = dict(zip(df.track_id, df.alt_family))
    assert all(fam[a] == fam[b] for a, b, *_ in links)  # every link was unioned into one family
    split_of = {t: k for k in ("train", "val", "test") for t in splits[k]}
    stems = df[df.title_stem != ""].assign(split=df.track_id.map(split_of)).groupby("title_stem").split.nunique()
    assert (stems == 1).all()


@committed
def test_committed_splits_stratified(committed_manifest):
    """Per-split coarse-genre share within 10 points of the corpus share for
    every class with at least 20 eligible tracks."""
    df, splits = committed_manifest
    split_of = {t: k for k in ("train", "val", "test") for t in splits[k]}
    d = df.assign(
        split=df.track_id.map(split_of),
        genre=[coarse_genre(t, j) for t, j in zip(df.human_genre_tag, df.caption_is_junk)],
        eligible=eligible_mask(df),
    )
    big = d[d.eligible].genre.value_counts()
    big = big[big >= 20].index
    assert len(big) >= 4
    corpus = d.genre.value_counts(normalize=True)
    for k in ("train", "val", "test"):
        share = d[d.split == k].genre.value_counts(normalize=True).reindex(corpus.index, fill_value=0.0)
        for g in big:
            assert abs(share[g] - corpus[g]) <= 0.10, (k, g, share[g], corpus[g])
