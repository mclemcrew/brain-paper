"""Family-atomic, genre-stratified train/val/test split.

Usage::

    python -m ravi.data.splits --manifest manifests/tracks.parquet --out manifests/splits.json

Rules
-----
* A track is *eligible* for evaluation if it has human labels, at least
  ``MIN_SEGMENTS`` labelled segments, lasts at least ``MIN_DURATION_S`` and has
  at least one interior verse/bridge/chorus (a section with audio on both sides).
* ``alt_family`` is atomic: all members land in the same split. Families
  come from :func:`ravi.data.build_manifest.assign_families` ([ALT] refs,
  duplicate ids, title stems, the [AKA] neighbour rule and the audio
  near-duplicate pairs in ``manifests/manual_links.json``); :func:`check_links`
  re-checks that no listed pair spans two splits.
* Val and test families are drawn only from families whose members are *all*
  eligible. Everything not drawn goes to train.
* Targets are fractions of *all* tracks (default 80/5/15). Each split's track
  quota is apportioned across coarse-genre strata in proportion to the
  stratum's share of the whole corpus, capped by what that stratum can supply
  from eligible families; any shortfall is redistributed (water-filling). Within
  a stratum, families are taken in a seeded random order without overshooting
  the quota; a final seeded top-up across strata closes any remaining gap.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261014
FRACTIONS = {"train": 0.80, "val": 0.05, "test": 0.15}
MIN_SEGMENTS = 3
MIN_DURATION_S = 20.0
GENRES = ("rock", "electronic", "folk/acoustic", "ambient", "jazz", "classical/cinematic", "pop/other", "unknown")

# Ordered (first match wins); matched against the lower-cased tag.
_GENRE_RULES: list[tuple[str, str]] = [
    ("classical/cinematic", r"classic|cinematic|score|orchestral|choral|minimalis|^contemporary$|contemporary instrumental|circus|cartoon"),
    ("jazz", r"jazz|swing|cha cha|^fusion$"),
    ("rock", r"\brock\b"),
    ("ambient", r"ambient|new age|ethereal|environmental"),
    ("electronic", r"electr|electo|tronic|edm|house|synth|disco|industrial|jungle|future bass|chillwave|dubstep|techno"),
    ("folk/acoustic", r"folk|acoustic|country|western|flamenco|world|indian|bollywood"),
]


def coarse_genre(tag: str | None, junk: bool = False) -> str:
    """Map a free-text human genre tag to one of :data:`GENRES`."""
    t = (tag or "").strip().lower().rstrip(".")
    if junk or not t:
        return "unknown"
    for bucket, pat in _GENRE_RULES:
        if re.search(pat, t):
            return bucket
    return "pop/other"


def eligible_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["has_labels"].astype(bool)
        & (df["n_segments"] >= MIN_SEGMENTS)
        & (df["duration_s"] >= MIN_DURATION_S)
        & (df["n_interior"] >= 1)
    )


def _family_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(
        eligible=eligible_mask(df),
        genre=[coarse_genre(t, j) for t, j in zip(df["human_genre_tag"], df["caption_is_junk"])],
    )

    def fam_genre(g: pd.Series) -> str:
        known = g[g != "unknown"]
        if known.empty:
            return "unknown"
        counts = known.value_counts()
        return sorted(counts[counts == counts.max()].index)[0]

    fam = d.groupby("alt_family").agg(
        size=("track_id", "size"),
        all_eligible=("eligible", "all"),
        genre=("genre", fam_genre),
    )
    return fam


def _apportion(total: int, weights: dict[str, float], caps: dict[str, int]) -> dict[str, int]:
    """Integer quotas summing to min(total, sum(caps)), proportional to weights, capped."""
    quota = {k: 0 for k in weights}
    remaining = min(total, sum(caps.values()))
    open_keys = {k for k in weights if caps[k] > 0 and weights[k] > 0}
    while remaining > 0 and open_keys:
        wsum = sum(weights[k] for k in open_keys)
        raw = {k: remaining * weights[k] / wsum for k in open_keys}
        give = {k: min(int(np.floor(raw[k])), caps[k] - quota[k]) for k in open_keys}
        # largest remainder for the leftover units
        left = remaining - sum(give.values())
        for k in sorted(open_keys, key=lambda k: (-(raw[k] - np.floor(raw[k])), k)):
            if left <= 0:
                break
            if quota[k] + give[k] < caps[k]:
                give[k] += 1
                left -= 1
        if sum(give.values()) == 0:
            break
        for k, v in give.items():
            quota[k] += v
        remaining -= sum(give.values())
        open_keys = {k for k in open_keys if quota[k] < caps[k]}
    return quota


def make_splits(df: pd.DataFrame, seed: int = SEED, fractions: dict[str, float] = FRACTIONS) -> dict:
    rng = np.random.default_rng(seed)
    fam = _family_table(df)
    n_total = len(df)
    corpus_share = df.assign(
        genre=[coarse_genre(t, j) for t, j in zip(df["human_genre_tag"], df["caption_is_junk"])]
    )["genre"].value_counts(normalize=True).to_dict()

    available = set(fam.index[fam["all_eligible"]])
    assigned: dict[str, list[int]] = {"test": [], "val": []}
    for split in ("test", "val"):
        target = int(round(fractions[split] * n_total))
        pool = fam.loc[sorted(available)]
        caps = pool.groupby("genre")["size"].sum().to_dict()
        weights = {g: corpus_share.get(g, 0.0) for g in GENRES}
        caps = {g: int(caps.get(g, 0)) for g in GENRES}
        quota = _apportion(target, weights, caps)

        chosen: list[int] = []
        for g in GENRES:
            fams = pool.index[pool["genre"] == g].to_numpy()
            fams = fams[rng.permutation(len(fams))]
            got = 0
            for f in fams:
                s = int(fam.at[f, "size"])
                if got + s <= quota[g]:
                    chosen.append(int(f))
                    got += s
                if got == quota[g]:
                    break
        # Top-up across strata if family sizes left the split short.
        n_chosen = int(fam.loc[chosen, "size"].sum()) if chosen else 0
        rest = np.array(sorted(available - set(chosen)))
        rest = rest[rng.permutation(len(rest))]
        for f in rest:
            if n_chosen >= target:
                break
            s = int(fam.at[f, "size"])
            if n_chosen + s <= target:
                chosen.append(int(f))
                n_chosen += s
        assigned[split] = sorted(chosen)
        available -= set(chosen)

    fam_split = {f: "train" for f in fam.index}
    for split, fams in assigned.items():
        for f in fams:
            fam_split[f] = split
    split_of_track = df["alt_family"].map(fam_split)
    out = {s: sorted(df.loc[split_of_track == s, "track_id"].tolist()) for s in ("train", "val", "test")}
    out["seed"] = seed
    out["criteria"] = {
        "fractions_of_all_tracks": fractions,
        "unit": (
            "alt_family (union-find over ID3 [ALT] refs, duplicate copilot ids, identical title stems, "
            "[AKA] singletons with identical ID3 genre within +/-5 ids, and manifests/manual_links.json "
            "audio near-duplicate pairs)"
        ),
        "eligible": {
            "has_labels": True,
            "min_labeled_segments": MIN_SEGMENTS,
            "min_duration_s": MIN_DURATION_S,
            "min_interior_verse_bridge_chorus": 1,
        },
        "val_test_families": "all members eligible",
        "stratify_by": "coarse genre from human_genre_tag",
        "genre_rules": [[b, p] for b, p in _GENRE_RULES] + [["pop/other", "<fallback>"], ["unknown", "junk or empty"]],
        "notes": split_notes(df, out),
    }
    check_splits(df, out)
    return out


def split_notes(df: pd.DataFrame, splits: dict) -> list[str]:
    """Plain-language caveats about the realised split, computed from the data."""
    split_of = {t: s for s in ("train", "val", "test") for t in splits[s]}
    d = df.assign(
        split=df["track_id"].map(split_of),
        eligible=eligible_mask(df),
        genre=[coarse_genre(t, j) for t, j in zip(df["human_genre_tag"], df["caption_is_junk"])],
    )
    fam_ok = d.groupby("alt_family")["eligible"].all()
    pool = int(d["alt_family"].map(fam_ok).sum())
    n_el = int(d["eligible"].sum())
    n_vt = int(d["split"].isin(["val", "test"]).sum())
    el_train = int((d["eligible"] & (d["split"] == "train")).sum())
    notes = [
        f"val+test ({n_vt} tracks) are drawn only from the {pool} tracks in all-eligible families, "
        f"so they drain the labelled pool: of {n_el} eligible tracks, {el_train} remain in train "
        f"(those in families with an ineligible member, or left after the quotas were met).",
    ]
    by_g = pd.crosstab(d["genre"], d["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    for g, row in by_g.iterrows():
        if row["val"] == 0 and row["test"] > 0:
            notes.append(f"{g}: none in val; all {int(row['test'])} held-out {g} tracks are in test.")
    m_test, m_all = d.loc[d["split"] == "test", "duration_s"].mean(), d["duration_s"].mean()
    notes.append(
        f"test mean duration {m_test:.1f} s vs corpus {m_all:.1f} s (val+test are eligible tracks only: "
        f">= {MIN_DURATION_S:g} s with >= {MIN_SEGMENTS} labelled segments, which excludes stingers)."
    )
    notes.append(
        "interior inpainting targets are verse, bridge AND chorus sections with audio on both sides "
        "(ravi.structure.labels.INTERIOR_TARGETS); n_interior counts all three."
    )
    return notes


def check_links(splits: dict, pairs: list[list]) -> None:
    """Assert no ``[track_id_a, track_id_b, ...]`` pair has its tracks in two splits."""
    split_of = {t: s for s in ("train", "val", "test") for t in splits[s]}
    bad = [(p[0], p[1], split_of.get(p[0]), split_of.get(p[1])) for p in pairs
           if p[0] in split_of and p[1] in split_of and split_of[p[0]] != split_of[p[1]]]
    assert not bad, f"manual links span splits: {bad}"


def check_splits(df: pd.DataFrame, splits: dict) -> None:
    ids = {s: set(splits[s]) for s in ("train", "val", "test")}
    all_ids = set(df["track_id"])
    assert ids["train"] | ids["val"] | ids["test"] == all_ids, "splits do not cover every track"
    assert sum(len(v) for v in ids.values()) == len(all_ids), "a track appears in more than one split"
    split_of = {t: s for s, v in ids.items() for t in v}
    fam_splits = df.assign(split=df["track_id"].map(split_of)).groupby("alt_family")["split"].nunique()
    assert (fam_splits == 1).all(), f"families span splits: {fam_splits[fam_splits > 1].index.tolist()}"
    el = dict(zip(df["track_id"], eligible_mask(df)))
    for s in ("val", "test"):
        bad = [t for t in ids[s] if not el[t]]
        assert not bad, f"ineligible tracks in {s}: {bad}"


def report(df: pd.DataFrame, splits: dict) -> str:
    split_of = {t: s for s in ("train", "val", "test") for t in splits[s]}
    d = df.assign(
        split=df["track_id"].map(split_of),
        eligible=eligible_mask(df),
        genre=[coarse_genre(t, j) for t, j in zip(df["human_genre_tag"], df["caption_is_junk"])],
    )
    lines = [f"{'split':6} {'tracks':>6} {'families':>8} {'hours':>7} {'eligible':>8} {'interior':>8}"]
    for s in ("train", "val", "test"):
        x = d[d["split"] == s]
        lines.append(
            f"{s:6} {len(x):6d} {x['alt_family'].nunique():8d} {x['duration_s'].sum() / 3600:7.3f} "
            f"{int(x['eligible'].sum()):8d} {int(x['n_interior'].sum()):8d}"
        )
    lines.append("")
    lines.append(pd.crosstab(d["genre"], d["split"]).reindex(columns=["train", "val", "test"]).to_string())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Make family-atomic train/val/test splits.")
    ap.add_argument("--manifest", type=Path, default=Path("manifests/tracks.parquet"))
    ap.add_argument("--out", type=Path, default=Path("manifests/splits.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--manual-links", type=Path, default=Path("manifests/manual_links.json"))
    args = ap.parse_args(argv)
    df = pd.read_parquet(args.manifest)
    splits = make_splits(df, seed=args.seed)
    if args.manual_links.exists():
        check_links(splits, json.loads(args.manual_links.read_text())["pairs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2) + "\n")
    print(report(df, splits))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
