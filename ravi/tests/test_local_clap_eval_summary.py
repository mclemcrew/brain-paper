"""Summary builder of scripts/local_clap_eval.py on synthetic scores (no models, no audio)."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ravi.eval.stats import holm

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "local_clap_eval.py"


@pytest.fixture(scope="module")
def L():
    spec = importlib.util.spec_from_file_location("local_clap_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ARMS = [{"name": "base-small", "dit": "sm-music", "adapter": None},
        {"name": "up", "dit": "sm-music", "adapter": "a.safetensors"},
        {"name": "down", "dit": "sm-music", "adapter": "b.safetensors"},
        {"name": "base-medium", "dit": "medium", "adapter": None},
        {"name": "med", "dit": "medium", "adapter": "c.safetensors"}]
CAPS = [f"c{i:02d}" for i in range(1, 7)]
SEEDS = [42, 173]


def row(arm, dit, cap, seed, value, tail=-1.0):
    return {"arm": arm, "caption_id": cap, "seed": seed, "dit": dit, "prompt_source": "machine",
            "text_score_machine": value, "text_score_human": value - 0.1, "text_score_shuffled": value - 0.2,
            "audio_sim": 0.5, "tail_db_1s": tail, "tail_db_025s": tail, "spike_flag": int(tail > 3),
            "fade_flag": int(tail < -6), "lufs_mono_in": -18.0, "seconds": 15.0, "path": "x.wav"}


@pytest.fixture(scope="module")
def rows():
    rng = np.random.default_rng(0)
    base = {(c, s): float(rng.uniform(0.2, 0.6)) for c in CAPS for s in SEEDS}
    out = []
    for (c, s), v in base.items():
        out.append(row("base-small", "sm-music", c, s, v))
        out.append(row("base-medium", "medium", c, s, v))
        # "up": +0.05 everywhere, a tie on c01 (both seeds), a NaN on (c02, 42)
        up = v if c == "c01" else (math.nan if (c, s) == ("c02", 42) else v + 0.05)
        out.append(row("up", "sm-music", c, s, up, tail=4.0 if c == "c03" else -1.0))
        out.append(row("down", "sm-music", c, s, v - 0.01 * int(c[1:]), tail=-8.0))
        out.append(row("med", "medium", c, s, v + (0.01 if c in ("c01", "c02") else -0.01)))
    return out


def cmp(summary, arm, metric="text_score_machine"):
    return next(c for c in summary["comparisons"] if c["arm"] == arm and c["metric"] == metric)


def test_pairs_ties_and_nan(L, rows):
    s = L.build_summary(rows, ARMS, n_boot=500)
    up = cmp(s, "up")
    assert up["base"] == "base-small" and up["matched_metric"]
    assert up["n"] == 6                    # NaN drops one seed of c02, not the caption
    assert up["n_nonzero"] == 5            # the c01 tie is discarded by the Wilcoxon test
    assert up["median_diff"] == pytest.approx(0.05)
    assert up["p_two_sided"] == pytest.approx(2 / 32)  # exact: 5 positive ranks
    assert up["render_level"]["n"] == 11 and up["render_level"]["n_clusters"] == 6
    assert cmp(s, "med")["base"] == "base-medium"
    assert cmp(s, "down")["median_diff"] < 0


def test_holm_across_adapter_arms(L, rows):
    s = L.build_summary(rows, ARMS, n_boot=500)
    group = [cmp(s, a) for a in ("up", "down", "med")]
    raw = np.array([g["p_two_sided"] for g in group])
    expected = holm(raw)
    assert [g["p_holm"] for g in group] == pytest.approx(list(expected))
    assert all(g["p_holm"] >= g["p_two_sided"] for g in group)


def test_arm_medians_and_flags(L, rows):
    s = L.build_summary(rows, ARMS, n_boot=200)
    up = s["arms"]["up"]["by_prompt_source"]["machine"]
    assert up["n_renders"] == 12 and up["n_captions"] == 6
    assert math.isfinite(up["median"]["text_score_machine"])
    assert up["spike_count"] == 2 and up["fade_count"] == 0
    assert s["arms"]["down"]["by_prompt_source"]["machine"]["fade_count"] == 12
    assert up["matched_gt_shuffled"] == 1.0      # NaN row excluded from the fraction
    assert s["unpaired_adapter_arms"] == []


def test_outputs_are_strict_json_and_booktabs(L, rows):
    s = L.build_summary(rows, ARMS, n_boot=200)
    s["extra"] = math.nan
    json.dumps(L._clean(s), allow_nan=False)
    tex = L.results_table_tex(s)
    assert "\\toprule" in tex and "\\bottomrule" in tex and "base-small" in tex
    assert tex.count("\\\\") == 2 + len(ARMS)
    md = L.summary_markdown(s)
    assert "up vs base-small" in md


def test_string_rows_from_csv(L, rows, tmp_path):
    path = tmp_path / "scores.csv"
    L.write_scores(path, rows)
    back = L.read_scores(path)
    s1 = L.build_summary(rows, ARMS, n_boot=200)
    s2 = L.build_summary(back, ARMS, n_boot=200)
    assert cmp(s2, "up")["n"] == cmp(s1, "up")["n"]
    assert cmp(s2, "up")["median_diff"] == pytest.approx(cmp(s1, "up")["median_diff"], abs=1e-6)


def test_parse_arm_and_bases(L):
    with pytest.raises(Exception):
        L.parse_arm("x=:bad-dit:/tmp")
    with pytest.raises(Exception):
        L.parse_arm("x=:sm-music:/tmp:rvkr")      # a base arm takes no trigger
    b = L.parse_arm("b=:medium:/tmp")
    assert b["adapter"] is None and b["trigger"] == ""
    with pytest.raises(ValueError):
        L.base_for([{"name": "a", "dit": "medium", "adapter": None}, {"name": "b", "dit": "medium", "adapter": None},
                    {"name": "c", "dit": "medium", "adapter": "x"}])


# ------------------------------------------------------------------ --compare, MDE, score-only

ARMS2 = [{"name": "base-small", "dit": "sm-music", "adapter": None},
         {"name": "r2", "dit": "sm-music", "adapter": "r2.safetensors"},
         {"name": "r3", "dit": "sm-music", "adapter": "r3.safetensors"},
         {"name": "base-medium", "dit": "medium", "adapter": None},
         {"name": "m1", "dit": "medium", "adapter": "m1.safetensors"}]


@pytest.fixture(scope="module")
def rows2():
    """r2 = base + 0.02 + per-caption wobble; r3 = r2 + 0.03 + wobble (r3 > r2 on every caption)."""
    rng = np.random.default_rng(1)
    out = []
    for c in CAPS:
        wob2, wob3 = rng.normal(0, 0.005), rng.normal(0, 0.005)
        for s in SEEDS:
            v = float(rng.uniform(0.2, 0.6))
            out.append(row("base-small", "sm-music", c, s, v))
            out.append(row("r2", "sm-music", c, s, v + 0.02 + wob2))
            out.append(row("r3", "sm-music", c, s, v + 0.05 + wob2 + wob3))
            out.append(row("base-medium", "medium", c, s, v))
            out.append(row("m1", "medium", c, s, v + 0.01))
    return out


def pair(summary, a, b, metric="text_score_machine"):
    return next(c for c in summary["pairwise"] if (c["arm_a"], c["arm_b"], c["metric"]) == (a, b, metric))


def test_compare_two_adapters(L, rows2):
    s = L.build_summary(rows2, ARMS2, n_boot=500)          # default r2:r3 (both exist)
    assert s["compares"] == [["r2", "r3"]]
    r = pair(s, "r2", "r3")                                # direction A - B
    assert r["n"] == 6 and r["level"] == "caption_median_over_seeds"
    assert r["median_diff"] < 0 and r["median_diff"] == pytest.approx(-0.03, abs=0.02)
    assert r["p_two_sided"] == pytest.approx(2 / 64)       # exact: all 6 differences negative
    assert r["p_holm"] == pytest.approx(r["p_two_sided"])  # one pair: Holm is the identity
    assert r["render_level"]["n"] == 12 and r["render_level"]["n_clusters"] == 6
    # reversed pair is the exact mirror
    s2 = L.build_summary(rows2, ARMS2, n_boot=500, compares=[("r3", "r2")])
    assert pair(s2, "r3", "r2")["median_diff"] == pytest.approx(-r["median_diff"])
    assert pair(s2, "r3", "r2")["p_two_sided"] == pytest.approx(r["p_two_sided"])
    # adapter-vs-base comparisons are unchanged by the pairwise block
    assert cmp(s, "r2")["base"] == "base-small" and cmp(s, "m1")["base"] == "base-medium"


def test_compare_repeatable_validation_and_default(L, rows2):
    s = L.build_summary(rows2, ARMS2, n_boot=200, compares=[("r2", "r3"), ("r3", "r2")])
    srcs_metrics = {(c["prompt_source"], c["metric"]) for c in s["pairwise"]}
    assert len(s["pairwise"]) == 2 * len(srcs_metrics)
    # Holm over the two pairs: mirror images have equal p, so p_holm == p (both scaled by 2, capped)
    live = [c for c in s["pairwise"] if math.isfinite(c["p_two_sided"])]   # audio_sim is constant -> NaN p
    assert live and all(c["p_holm"] == pytest.approx(min(1.0, 2 * c["p_two_sided"])) for c in live)
    assert L.build_summary(rows2, ARMS2, n_boot=100, compares=[])["pairwise"] == []
    # default only when both r2 and r3 exist
    assert L.build_summary(rows2, ARMS, n_boot=100)["compares"] == []
    assert L.parse_compare("r2:r3") == ("r2", "r3")
    for bad in ("r2", "r2:r2", "a:b:c", ":r3"):
        with pytest.raises(Exception):
            L.parse_compare(bad)
    with pytest.raises(ValueError, match="different DiTs"):
        L.resolve_compares(ARMS2, [("r2", "m1")])
    with pytest.raises(ValueError, match="base arm"):
        L.resolve_compares(ARMS2, [("r2", "base-small")])
    with pytest.raises(ValueError, match="unknown arm"):
        L.resolve_compares(ARMS2, [("r2", "zzz")])


def test_mde(L, rows2):
    s = L.build_summary(rows2, ARMS2, n_boot=200)
    c = cmp(s, "r2")
    am = {}
    for r in rows2:
        if r["prompt_source"] == "machine" and r["arm"] in ("r2", "base-small"):
            am.setdefault((r["arm"], r["caption_id"]), []).append(r["text_score_machine"])
    d = [np.median(am[("r2", cap)]) - np.median(am[("base-small", cap)]) for cap in CAPS]
    assert c["sd_diff"] == pytest.approx(np.std(d, ddof=1))
    assert c["mde"] == pytest.approx(2.8 * np.std(d, ddof=1) / math.sqrt(6))
    assert math.isnan(L.mde_paired([1.0])["mde"])          # n=1 -> NaN
    assert L.mde_paired([0.0, 2.0, math.nan])["mde"] == pytest.approx(2.8 * math.sqrt(2) / math.sqrt(2))
    md = L.summary_markdown(s)
    assert "## Minimum detectable effect" in md and "2.8 * sd(paired caption diffs) / sqrt(n)" in md
    assert "r2 vs base-small: n=6" in md
    assert "| r2 vs r3 |" in md and "| MDE |" in md


def test_pairwise_outputs_strict_json_and_tex_rows(L, rows2):
    s = L.build_summary(rows2, ARMS2, n_boot=200)
    json.dumps(L._clean(s), allow_nan=False)
    tex = L.results_table_tex(s)
    assert "r2 vs r3" in tex
    assert tex.count("\\\\") == 2 + len(ARMS2) + 1        # header rows, arm rows, one --compare row
    assert tex.count("\\midrule") == 2
    body = [l for l in tex.splitlines() if l.startswith("r2 vs r3")][0]
    assert "$-$" not in body and "--" in body and "+" not in body.split("&")[0]
    # no compares -> table is the plain one
    plain = L.results_table_tex(L.build_summary(rows2, ARMS2, n_boot=100, compares=[]))
    assert plain.count("\\midrule") == 1 and "r2 vs r3" not in plain


def test_score_only_path_does_not_import_mlx(L):
    import subprocess, sys
    code = ("import importlib.util, sys;"
            f"spec=importlib.util.spec_from_file_location('m', {str(SCRIPT)!r});m=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(m);"
            "assert 'ravi.local.worker' not in sys.modules and 'mlx' not in sys.modules;"
            "assert callable(m.import_worker)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=SCRIPT.parents[1])
