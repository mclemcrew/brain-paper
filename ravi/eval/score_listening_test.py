"""Score the listening test (plan Phase 4.3) from ``trials.json`` and the raters' response files.

    python -m ravi.eval.score_listening_test listening-test-v2/trials.json \\
        listening-test-v2/responses/responses_*.json --out-md eval/listening_test.md \\
        --out-json eval/listening_test.json

``correct`` is recomputed from the obfuscated key in ``trials.json`` (the page's own
``correct`` field is only checked against it). A response counts for the 3AFC when it has a
choice and for the rating when it has a rating; a response file whose ``build_id`` differs from
``trials.json`` is refused (the trial ids would not mean the same thing), unless
``--ignore-build-id``. If a rater has several files for one block, the later response to a trial
(by ``submitted_at`` / ``rated_at`` / ``choice_at``) wins.

Statistics, per condition (and per rater x condition, and pooled):

* **3AFC accuracy** ``k / n`` with the exact binomial test against chance ``p0 = 1/3``
  (:func:`scipy.stats.binomtest`): ``p_greater`` (H1: detectable, accuracy > 1/3) is the
  primary column, ``p_two_sided`` is also given; ``p_greater_holm`` is Holm over the conditions.
  95% CI: Wilson score interval (:func:`wilson_ci`). Trials are pooled over raters and treated
  as independent (2 raters, each region at most once per rater).
* **"Sounds like Ravi"** mean rating (1-5) with a percentile bootstrap 95% CI over the ratings
  (``--n-boot`` resamples with replacement, seeded PCG64).
* **Inter-rater agreement** on correctness: Cohen's kappa for each pair of raters over the
  *items* both answered (the shared trials, same audio, letters permuted per block), with the
  raw agreement and ``n``. ``kappa`` is NaN when undefined (no common items, or both raters
  constant with the same value so that chance agreement is 1).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ravi.eval.make_listening_test import LETTERS, decode_keys, deobfuscate

TOOL = "ravi.eval.score_listening_test"
CHANCE = 1.0 / 3.0
Z95 = 1.959963984540054
DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20261014


# --------------------------------------------------------------------------- statistics


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; ``(nan, nan)`` for ``n == 0``."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_p(k: int, n: int, p0: float = CHANCE, alternative: str = "greater") -> float:
    if n <= 0:
        return math.nan
    from scipy.stats import binomtest

    return float(binomtest(k, n, p0, alternative=alternative).pvalue)


def bootstrap_mean_ci(x: Sequence[float], n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                      level: float = 0.95) -> tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return (math.nan, math.nan)
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa of two raters' labels on the same items (NaN if undefined)."""
    if len(a) != len(b):
        raise ValueError("cohen_kappa: different lengths")
    n = len(a)
    if n == 0:
        return math.nan
    cats = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    if pe >= 1.0:
        return math.nan
    return (po - pe) / (1 - pe)


def holm(p: Sequence[float]) -> list[float]:
    from ravi.eval.stats import holm as _holm

    return [float(x) for x in _holm(np.asarray(p, dtype=np.float64))] if len(p) else []


# --------------------------------------------------------------------------- loading


def load_trials(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if not str(doc.get("format", "")).startswith("ravi-listening-test-v2"):
        raise SystemExit(f"{path}: not a listening-test-v2 trials.json")
    return doc


def _ts(r: Mapping[str, Any]) -> str:
    return str(r.get("submitted_at") or r.get("rated_at") or r.get("choice_at") or "")


def load_responses(paths: Iterable[str | Path], trials: Mapping[str, Any], ignore_build_id: bool = False,
                   warn=lambda m: print(f"[score_listening_test] {m}", file=sys.stderr)) -> list[dict[str, Any]]:
    """One row per (rater, block, trial) with the key fields joined in."""
    keys = decode_keys(trials)
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        if doc.get("build_id") != trials.get("build_id") and not ignore_build_id:
            raise SystemExit(f"{p}: build_id {doc.get('build_id')!r} != trials.json {trials.get('build_id')!r}")
        rater = str(doc.get("rater") or Path(p).stem.removeprefix("responses_")).strip()
        block = str(doc["block"])
        if block not in trials["blocks"]:
            raise SystemExit(f"{p}: block {block!r} not in trials.json")
        for r in doc.get("responses") or []:
            tid = str(r["trial_id"])
            k = keys.get((block, tid))
            if k is None:
                raise SystemExit(f"{p}: trial {tid!r} not in block {block}")
            if r.get("item") not in (None, k["item"]):
                raise SystemExit(f"{p}: trial {tid} item {r.get('item')!r} != key item {k['item']!r}")
            choice = r.get("choice")
            if choice is not None and choice not in LETTERS:
                raise SystemExit(f"{p}: trial {tid}: bad choice {choice!r}")
            correct = None if choice is None else choice == k["answer"]
            if r.get("correct") is not None and correct is not None and bool(r["correct"]) != correct:
                warn(f"{p}: trial {tid}: page said correct={r['correct']}, key says {correct}")
            rating = r.get("rating")
            row = {
                "rater": rater, "block": block, "trial_id": tid, "item": k["item"], "condition": k["condition"],
                "track_id": k["track_id"], "section_index": k["section_index"], "label": k["label"],
                "answer": k["answer"], "choice": choice, "correct": correct,
                "rating": None if rating is None else float(rating), "comment": r.get("comment") or "",
                "plays": r.get("plays"), "ts": _ts(r), "file": str(p),
            }
            kk = (rater.lower(), block, tid)
            if kk not in best or row["ts"] >= best[kk]["ts"]:
                best[kk] = row
    return sorted(best.values(), key=lambda r: (r["rater"].lower(), r["block"], r["trial_id"]))


# --------------------------------------------------------------------------- scoring


def summarise(rows: Sequence[Mapping[str, Any]], n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    afc = [r for r in rows if r["correct"] is not None]
    rat = [r for r in rows if r["rating"] is not None]
    k = sum(1 for r in afc if r["correct"])
    n = len(afc)
    lo, hi = wilson_ci(k, n)
    ratings = [r["rating"] for r in rat]
    rlo, rhi = bootstrap_mean_ci(ratings, n_boot, seed)
    return {
        "n": n, "k": k, "accuracy": k / n if n else math.nan, "acc_ci_lo": lo, "acc_ci_hi": hi,
        "p_greater": binom_p(k, n, CHANCE, "greater"), "p_two_sided": binom_p(k, n, CHANCE, "two-sided"),
        "n_rating": len(ratings), "rating_mean": float(np.mean(ratings)) if ratings else math.nan,
        "rating_sd": float(np.std(ratings, ddof=1)) if len(ratings) > 1 else math.nan,
        "rating_ci_lo": rlo, "rating_ci_hi": rhi,
    }


def agreement(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_rater: dict[str, dict[str, bool]] = {}
    for r in rows:
        if r["correct"] is not None:
            by_rater.setdefault(r["rater"], {})[r["item"]] = bool(r["correct"])
    out = []
    for a, b in itertools.combinations(sorted(by_rater, key=str.lower), 2):
        common = sorted(set(by_rater[a]) & set(by_rater[b]))
        xa = [by_rater[a][i] for i in common]
        xb = [by_rater[b][i] for i in common]
        out.append({
            "rater_a": a, "rater_b": b, "n_items": len(common),
            "agreement": (sum(x == y for x, y in zip(xa, xb)) / len(common)) if common else math.nan,
            "kappa": cohen_kappa(xa, xb), "acc_a": float(np.mean(xa)) if xa else math.nan,
            "acc_b": float(np.mean(xb)) if xb else math.nan,
        })
    return out


def score(trials: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], n_boot: int = DEFAULT_N_BOOT,
          seed: int = DEFAULT_SEED) -> dict[str, Any]:
    meta = deobfuscate(trials["meta"]) if trials.get("meta") else {}
    conds = list(meta.get("conditions") or sorted({r["condition"] for r in rows}))
    per_cond = []
    for c in conds:
        s = summarise([r for r in rows if r["condition"] == c], n_boot, seed)
        per_cond.append({"condition": c, **s})
    for x, ph in zip(per_cond, holm([x["p_greater"] for x in per_cond])):
        x["p_greater_holm"] = ph  # NaN passes through and does not count towards m
    per_rater = []
    for rater in sorted({r["rater"] for r in rows}, key=str.lower):
        for c in conds:
            sub = [r for r in rows if r["rater"] == rater and r["condition"] == c]
            if sub:
                per_rater.append({"rater": rater, "condition": c, **summarise(sub, n_boot, seed)})
    return {
        "tool": TOOL, "build_id": trials.get("build_id"), "chance": CHANCE, "n_boot": n_boot, "seed": seed,
        "raters": sorted({r["rater"] for r in rows}, key=str.lower), "n_responses": len(rows),
        "per_condition": per_cond, "overall": summarise(rows, n_boot, seed), "per_rater": per_rater,
        "agreement": agreement(rows),
    }


# --------------------------------------------------------------------------- output


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def _p(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def markdown(res: Mapping[str, Any]) -> str:
    lines = [
        "| Condition | 3AFC correct | Accuracy [95% Wilson CI] | p (> 1/3) | p Holm | "
        "\"Sounds like Ravi\" mean [95% CI] | n ratings |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(name: str, s: Mapping[str, Any], holm_p: Any) -> str:
        return (f"| {name} | {s['k']}/{s['n']} | {_f(s['accuracy'])} [{_f(s['acc_ci_lo'])}, {_f(s['acc_ci_hi'])}] | "
                f"{_p(s['p_greater'])} | {_p(holm_p)} | {_f(s['rating_mean'])} [{_f(s['rating_ci_lo'])}, "
                f"{_f(s['rating_ci_hi'])}] | {s['n_rating']} |")

    for s in res["per_condition"]:
        lines.append(row(s["condition"], s, s["p_greater_holm"]))
    lines.append(row("all", res["overall"], None))
    lines += ["", f"Chance = 1/3. Exact one-sided binomial test; Holm over {len(res['per_condition'])} conditions. "
              f"Rating CI: percentile bootstrap, {res['n_boot']} resamples. Raters: {', '.join(res['raters'])}.", ""]
    if res["per_rater"]:
        lines += ["| Rater | Condition | 3AFC correct | Accuracy [95% CI] | Rating mean |", "|---|---|---|---|---|"]
        for s in res["per_rater"]:
            lines.append(f"| {s['rater']} | {s['condition']} | {s['k']}/{s['n']} | {_f(s['accuracy'])} "
                         f"[{_f(s['acc_ci_lo'])}, {_f(s['acc_ci_hi'])}] | {_f(s['rating_mean'])} |")
        lines.append("")
    if res["agreement"]:
        lines += ["| Raters | Shared items | Agreement on correctness | Cohen's kappa |", "|---|---|---|---|"]
        for a in res["agreement"]:
            lines.append(f"| {a['rater_a']} / {a['rater_b']} | {a['n_items']} | {_f(a['agreement'])} | "
                         f"{_f(a['kappa'])} |")
        lines.append("")
    return "\n".join(lines)


def _json_safe(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ravi.eval.score_listening_test",
                                 description="Score the 3AFC listening test (plan Phase 4.3).")
    ap.add_argument("trials", help="listening-test-v2/trials.json")
    ap.add_argument("responses", nargs="+", help="responses_{rater}.json files")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ignore-build-id", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    trials = load_trials(a.trials)
    rows = load_responses(a.responses, trials, a.ignore_build_id)
    res = score(trials, rows, a.n_boot, a.seed)
    md = markdown(res)
    print(md)
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(md + "\n")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(_json_safe({**res, "responses": list(rows)}), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
