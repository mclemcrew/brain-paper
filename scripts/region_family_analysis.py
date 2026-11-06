"""Recompute region comparisons with alternate family as the independent unit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ravi.eval.stats import cluster_medians, holm, paired_compare

METRICS = ("region_sim_clap", "region_sim_mert", "region_sim_mert_centred")
BASE_ADAPTERS = {
    "A6k", "A6k_sec", "B2k", "B4k", "B6k", "B6k_sec", "C6k", "D6k", "E6k", "E6k_nosec",
}


def load_regions(root: Path, tiers: tuple[str, ...]) -> pd.DataFrame:
    tracks = pd.read_parquet(root / "manifests/tracks.parquet", columns=["track_id", "alt_family"])
    frames = []
    for tier in tiers:
        frame = pd.read_parquet(root / f"eval/{tier}/metrics_regions_avg.parquet")
        frames.append(frame.merge(tracks, on="track_id", how="left", validate="many_to_one"))
    regions = pd.concat(frames, ignore_index=True)
    if regions["alt_family"].isna().any():
        raise ValueError("A region has no alternate-family assignment")
    return regions.drop_duplicates(["condition", "region_key"])


def compare_scope(regions: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    for metric in METRICS:
        wide = regions.pivot(index="region_key", columns="condition", values=metric).astype(float)
        families = regions.drop_duplicates("region_key").set_index("region_key")["alt_family"]
        clusters = families.reindex(wide.index).astype(str).to_numpy()
        conditions = list(wide.columns)
        scope_rows = []
        for i, first in enumerate(conditions):
            for second in conditions[i + 1 :]:
                cond_a, cond_b = (second, first) if first in {"base", "ACE_base"} else (first, second)
                a, b = wide[cond_a].to_numpy(float), wide[cond_b].to_numpy(float)
                family_a, family_b, _ = cluster_medians(a, b, clusters)
                result = paired_compare(family_a, family_b)
                scope_rows.append({
                    "scope": scope,
                    "metric": metric,
                    "level": "alternate_family",
                    "cond_a": cond_a,
                    "cond_b": cond_b,
                    **result,
                })
        adjusted = holm([row["p_two_sided"] for row in scope_rows])
        for row, p_adjusted in zip(scope_rows, adjusted):
            row["p_holm_scope"] = p_adjusted
            row["holm_scope_m"] = len(scope_rows)
        rows.extend(scope_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scopes = {
        "tier1": ("tier1",),
        "tier3": ("tier3",),
        "tier1_ace": ("tier1_ace",),
        "sa3_combined": ("tier1", "tier3"),
    }
    rows = []
    for scope, tiers in scopes.items():
        rows.extend(compare_scope(load_regions(args.root, tiers), scope))

    combined_base = [
        row for row in rows
        if row["scope"] == "sa3_combined" and row["cond_b"] == "base" and row["cond_a"] in BASE_ADAPTERS
    ]
    for metric in METRICS:
        metric_rows = [row for row in combined_base if row["metric"] == metric]
        adjusted = holm([row["p_two_sided"] for row in metric_rows])
        for row, p_adjusted in zip(metric_rows, adjusted):
            row["p_holm_base10"] = p_adjusted
    for row in rows:
        row.setdefault("p_holm_base10", "")

    columns = [
        "scope", "metric", "level", "cond_a", "cond_b", "n", "n_nonzero", "median_a", "median_b",
        "median_diff", "mean_diff", "statistic", "p_two_sided", "p_greater", "p_less", "ci_low",
        "ci_high", "ci_method", "p_holm_scope", "holm_scope_m", "p_holm_base10",
    ]
    output = pd.DataFrame(rows)[columns]
    if args.output:
        destination = args.output if args.output.is_absolute() else args.root / args.output
        output.to_csv(destination, index=False)
        print(f"Wrote {len(output)} comparisons to {destination}")
    else:
        output.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
