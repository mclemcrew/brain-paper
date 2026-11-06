"""Tests for ravi.eval.make_listening_test and ravi.eval.score_listening_test on a synthetic
grid eval root (8 kHz WAVs, no model). The page (listening-test-v2/) is not exercised here."""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ravi.eval import make_listening_test as M
from ravi.eval import score_listening_test as S

SR = 8000
CONDS = ("base", "B6k", "E6k")
SEGS = [(0, 6, "intro"), (6, 16, "verse"), (16, 26, "chorus"), (26, 34, "verse"), (34, 40, "outro")]
N_TRACKS = 8
PAD = 1.0


def track_id(i: int) -> str:
    return f"{i:04d}_copilot.mp3 Song {i}"


def orig_audio(i: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(i)
    t = np.arange(n) / SR
    x = 0.2 * np.sin(2 * np.pi * (110 + 10 * i) * t) + 0.01 * rng.standard_normal(n)
    return np.stack([x, 0.9 * x], axis=1).astype(np.float32)


def write_eval_root(root: Path, conds=CONDS, n_tracks=N_TRACKS, loud: bool = False) -> Path:
    seg_dir = root / "segments"
    seg_dir.mkdir(parents=True)
    for i in range(n_tracks):
        tid = track_id(i)
        dur = SEGS[-1][1]
        doc = {"track_id": tid, "tool": "human", "bpm": None, "beats": None, "downbeats": None,
               "segments": [{"start": a, "end": b, "label": l} for a, b, l in SEGS], "duration": dur}
        (seg_dir / f"{tid}.json").write_text(json.dumps(doc))
        n = int(dur * SR)
        orig = orig_audio(i, n)
        for idx in (1, 2, 3):
            a, b, lab = SEGS[idx]
            m0, m1 = int((a - PAD) * SR), int((b + PAD) * SR)
            stem = f"{tid}_sec{idx}_{lab}_{a - PAD:.1f}-{b + PAD:.1f}_seed1"
            for ci, c in enumerate(conds):
                d = root / "eval" / c / "inpaint"
                d.mkdir(parents=True, exist_ok=True)
                sp = orig.copy()
                level = (3.0 if loud else 0.5) + 0.1 * ci
                sp[m0:m1] = level * np.sin(2 * np.pi * 440 * np.arange(m1 - m0) / SR)[:, None]
                sf.write(str(d / f"{stem}_spliced.wav"), sp, SR, subtype="FLOAT")
                sf.write(str(d / f"{stem}_orig.wav"), orig, SR, subtype="FLOAT")
                meta = {
                    "command": "inpaint", "track_id": tid, "seed": 1,
                    "section": {"index": idx, "label": lab, "start_s": a, "end_s": b},
                    "mask": {"start_s": a - PAD, "end_s": b + PAD, "pad_s": PAD, "start_sample": m0, "end_sample": m1},
                    "context": {"start_s": 0.0, "end_s": dur, "start_sample": 0, "end_sample": n, "source_samples": n},
                    "mask_in_context": {"start_sample": m0, "end_sample": m1},
                    "files": {"spliced": f"{stem}_spliced.wav", "orig": f"{stem}_orig.wav"},
                    "segments_json": "/nonexistent/on/this/box.json",
                }
                (d / f"{stem}.json").write_text(json.dumps(meta))
    return root


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = write_eval_root(tmp_path_factory.mktemp("lt"))
    out = root / "page"
    doc = M.build(root / "eval", CONDS, out, segments_dir=root / "segments", n_trials=9, n_shared=3, seed=7,
                  fmt="wav")
    return root, out, doc


def plain(trial: dict) -> str:
    return json.dumps({k: v for k, v in trial.items() if k != "key"})


# --------------------------------------------------------------------------- planning


def check_plan(p: M.ExcerptPlan, n_ctx, m0, m1, sections, target_index, max_core):
    secs = {s.index: s for s in sections}
    assert p.core <= min(m1 - m0, max_core)
    assert len({e.n for e in p.excerpts}) == 1
    assert all(e.n == p.core + 2 * p.roll for e in p.excerpts)
    for e in p.excerpts:
        assert 0 <= e.start and e.end <= n_ctx
        assert e.core_start - e.start == p.roll and e.end - e.core_end == p.roll
    t = p.target
    assert m0 <= t.core_start and t.core_end <= m1  # the core is regenerated audio only
    if p.anchor in ("both", "start"):
        assert t.core_start == m0
    if p.anchor in ("both", "end"):
        assert t.core_end == m1
    a, b = p.others
    assert a.section_index != b.section_index
    assert target_index not in (a.section_index, b.section_index)
    for e in p.others:
        assert M._overlap(e.start, e.end, m0, m1) == 0  # distractors are untouched audio
        assert M._overlap(e.start, e.end, t.start, t.end) == 0
        s = secs[e.section_index]
        assert M._overlap(e.core_start, e.core_end, s.start, s.end) >= 0.5 * p.core
    assert M._overlap(a.start, a.end, b.start, b.end) == 0


def test_plan_cropped_and_mirrored():
    sections = [M.Section(i, l, a * SR, b * SR) for i, (a, b, l) in enumerate(SEGS)]
    m0, m1 = 15 * SR, 27 * SR  # chorus 16-26 +- 1 s
    p = M.plan_excerpts(40 * SR, SR, m0, m1, sections, 2, rng=random.Random(0), max_core_s=8)
    assert p is not None and p.anchor in ("start", "end") and p.core == 8 * SR
    check_plan(p, 40 * SR, m0, m1, sections, 2, 8 * SR)
    # a distractor sits at its own virtual mask (section start - pad / end + pad), like the target,
    # unless that would cross the mask, the target excerpt or the context edge: then it is clamped
    t = p.target
    f0, f1 = min(m0, t.start), max(m1, t.end)
    for e in p.others:
        s = sections[e.section_index]
        pref = s.start - SR if p.anchor == "start" else s.end + SR - p.core
        assert e.core_start == pref or e.start == 0 or e.end == 40 * SR or e.end == f0 or e.start == f1
    # whole-file layout where the virtual position is free: intro 0-12, verse 12-30, outro 30-40
    secs = [M.Section(0, "intro", 0, 12 * SR), M.Section(1, "verse", 12 * SR, 30 * SR),
            M.Section(2, "outro", 30 * SR, 47 * SR), M.Section(3, "chorus", 47 * SR, 60 * SR)]
    for seed in range(10):
        q = M.plan_excerpts(60 * SR, SR, 29 * SR, 48 * SR, secs, 2, rng=random.Random(seed), max_core_s=6)
        assert q is not None and q.anchor in ("start", "end")
        check_plan(q, 60 * SR, 29 * SR, 48 * SR, secs, 2, 6 * SR)
        for e in q.others:
            s = secs[e.section_index]
            if s.index == 0 and q.anchor == "end":
                assert e.core_end == s.end + SR or e.end == 29 * SR


def test_plan_whole_mask_when_short():
    sections = [M.Section(0, "intro", 0, 10 * SR), M.Section(1, "verse", 10 * SR, 15 * SR),
                M.Section(2, "chorus", 15 * SR, 30 * SR), M.Section(3, "outro", 30 * SR, 45 * SR)]
    m0, m1 = 9 * SR, 16 * SR
    p = M.plan_excerpts(45 * SR, SR, m0, m1, sections, 1, rng=random.Random(1), max_core_s=10)
    assert p is not None and p.anchor == "both" and p.core == m1 - m0
    check_plan(p, 45 * SR, m0, m1, sections, 1, 10 * SR)


def test_plan_none_without_room():
    sections = [M.Section(0, "intro", 0, 2 * SR), M.Section(1, "verse", 2 * SR, 20 * SR),
                M.Section(2, "outro", 20 * SR, 23 * SR)]
    assert M.plan_excerpts(23 * SR, SR, 1 * SR, 21 * SR, sections, 1, rng=random.Random(0)) is None
    # unlabeled sections are never distractors
    sections = [M.Section(0, "unlabeled", 0, 20 * SR), M.Section(1, "verse", 20 * SR, 30 * SR),
                M.Section(2, "unlabeled", 30 * SR, 50 * SR)]
    assert M.plan_excerpts(50 * SR, SR, 19 * SR, 31 * SR, sections, 1, rng=random.Random(0)) is None


def test_plan_property_random_layouts():
    rng = random.Random(123)
    n_ok = 0
    for trial in range(300):
        n_sec = rng.randint(3, 8)
        cuts = sorted(rng.uniform(0, 47) for _ in range(n_sec - 1))
        bounds = [0.0, *cuts, 47.0]
        sections = [M.Section(i, rng.choice(["intro", "verse", "chorus", "outro", "unlabeled"]),
                              int(bounds[i] * SR), int(bounds[i + 1] * SR)) for i in range(n_sec)]
        ti = rng.randrange(1, n_sec - 1) if n_sec > 2 else 0
        s = sections[ti]
        m0, m1 = max(0, s.start - SR), min(47 * SR, s.end + SR)
        p = M.plan_excerpts(47 * SR, SR, m0, m1, sections, ti, rng=random.Random(trial), max_core_s=10)
        if p is not None:
            n_ok += 1
            check_plan(p, 47 * SR, m0, m1, sections, ti, 10 * SR)
            assert p.core >= 4 * SR
    assert n_ok > 50


# --------------------------------------------------------------------------- build


def test_build_balance(built):
    _, _, doc = built
    keys = M.decode_keys(doc)
    assert set(doc["blocks"]) == {"A", "B"}
    items_per_block = []
    for b, blk in doc["blocks"].items():
        ks = [keys[(b, t["id"])] for t in blk["trials"]]
        assert len(ks) == 9
        assert Counter(k["condition"] for k in ks) == {c: 3 for c in CONDS}
        regions = [(k["track_id"], k["section_index"]) for k in ks]
        assert len(set(regions)) == len(regions)  # each region at most once per block
        for c in CONDS:  # answer letters balanced within a condition
            cnt = Counter(k["answer"] for k in ks if k["condition"] == c)
            assert sorted(cnt.values()) == [1, 1, 1]
        items_per_block.append({t["item"] for t in blk["trials"]})
        assert [t["index"] for t in blk["trials"]] == list(range(9))
    shared = items_per_block[0] & items_per_block[1]
    assert len(shared) == 3
    meta = M.deobfuscate(doc["meta"])
    assert sorted(meta["shared_items"]) == sorted(shared)
    assert meta["counts"] == {b: {c: 3 for c in CONDS} for b in "AB"}
    # shared items: same audio files, conditions balanced, own letters per block
    by_item = {b: {t["item"]: t for t in doc["blocks"][b]["trials"]} for b in "AB"}
    assert Counter(keys[("A", by_item["A"][i]["id"])]["condition"] for i in shared) == {c: 1 for c in CONDS}
    for i in shared:
        ta, tb = by_item["A"][i], by_item["B"][i]
        assert set(ta["excerpts"].values()) == set(tb["excerpts"].values())
        ka, kb = keys[("A", ta["id"])], keys[("B", tb["id"])]
        assert ta["excerpts"][ka["answer"]] == tb["excerpts"][kb["answer"]]
    # the unique parts of the two blocks do not reuse regions when the pool is large enough
    ra = {(keys[("A", t["id"])]["track_id"], keys[("A", t["id"])]["section_index"])
          for t in doc["blocks"]["A"]["trials"] if t["item"] not in shared}
    rb = {(keys[("B", t["id"])]["track_id"], keys[("B", t["id"])]["section_index"])
          for t in doc["blocks"]["B"]["trials"] if t["item"] not in shared}
    assert not ra & rb


def test_build_no_key_leak(built):
    root, out, doc = built
    keys = M.decode_keys(doc)
    secrets = set(CONDS) | {track_id(i) for i in range(N_TRACKS)} | {"verse", "chorus", "intro", "outro",
                                                                      "target", "other", "seed1", "sec1", "sec2"}
    top = json.dumps({k: v for k, v in doc.items() if k not in ("blocks", "meta")})
    for s in secrets:
        assert s not in top
    pat = re.compile(r"^audio/[0-9a-f]{12}\.wav$")
    for b, blk in doc["blocks"].items():
        for t in blk["trials"]:
            txt = plain(t)
            for s in secrets:
                assert s not in txt, (s, txt)
            assert set(t) == {"id", "index", "item", "excerpts", "context", "excerpt_s", "context_s", "key"}
            files = [*t["excerpts"].values(), t["context"]]
            assert all(pat.match(f) for f in files)
            assert len(set(files)) == 4
            assert all((out / f).is_file() for f in files)
            k = keys[(b, t["id"])]
            assert k["excerpts"][k["answer"]]["role"] == "target"
            assert t["excerpts"][k["answer"]] == k["excerpts"][k["answer"]]["file"]
    # the key is not plain text in the file either
    raw = (out / "trials.json").read_text()
    for s in secrets:
        assert s not in raw


def test_build_excerpt_audio(built):
    root, out, doc = built
    keys = M.decode_keys(doc)
    roll = int(round(M.ROLL_S * SR))
    nf = int(round(M.FADE_S * SR))
    for b, blk in doc["blocks"].items():
        for t in blk["trials"]:
            k = keys[(b, t["id"])]
            d = root / "eval" / k["condition"] / "inpaint"
            spliced, _ = sf.read(str(d / f"{k['stem']}_spliced.wav"), dtype="float32", always_2d=True)
            orig, _ = sf.read(str(d / f"{k['stem']}_orig.wav"), dtype="float32", always_2d=True)
            m0, m1 = k["mask_in_context"]
            assert k["gain"] == 1.0
            lens = set()
            for l in M.LETTERS:
                y, sr = sf.read(str(out / t["excerpts"][l]), dtype="float32", always_2d=True)
                e = k["excerpts"][l]
                assert sr == SR and y.shape[0] == e["end_sample"] - e["start_sample"]
                lens.add(y.shape[0])
                ref = M.fade(spliced[e["start_sample"]:e["end_sample"]], nf)
                assert np.max(np.abs(y - ref)) < 2e-4
                assert abs(y[0]).max() < 0.02 and abs(y[-1]).max() < 0.02  # faded ends
                if e["role"] == "target":
                    assert e["core_start_sample"] >= m0 and e["core_end_sample"] <= m1
                else:  # untouched source audio
                    seg = slice(e["start_sample"], e["end_sample"])
                    assert e["end_sample"] <= m0 or e["start_sample"] >= m1
                    assert np.array_equal(spliced[seg], orig[seg])
            assert len(lens) == 1
            n = lens.pop()
            assert n == round(k["core_s"] * SR) + 2 * roll
            assert t["excerpt_s"] == pytest.approx(n / SR, abs=1e-3)
            assert 4.0 <= k["core_s"] <= M.MAX_CORE_S
            ctx, _ = sf.read(str(out / t["context"]), dtype="float32", always_2d=True)
            assert ctx.shape == spliced.shape


def test_same_region_same_spans_across_conditions(built):
    _, _, doc = built
    keys = list(M.decode_keys(doc).values())
    spans: dict = {}
    for k in keys:
        r = (k["track_id"], k["section_index"])
        s = sorted((e["role"], e["start_sample"], e["end_sample"]) for e in k["excerpts"].values())
        spans.setdefault(r, set()).add(tuple(s))
    assert all(len(v) == 1 for v in spans.values())


def test_build_deterministic(built, tmp_path):
    root, out, doc = built
    doc2 = M.build(root / "eval", CONDS, tmp_path / "p2", segments_dir=root / "segments", n_trials=9, n_shared=3,
                   seed=7, fmt="wav")
    assert json.dumps(doc2, sort_keys=True) == json.dumps(doc, sort_keys=True)
    assert (tmp_path / "p2" / "trials.json").read_text() == (out / "trials.json").read_text()
    doc3 = M.build(root / "eval", CONDS, tmp_path / "p3", segments_dir=root / "segments", n_trials=9, n_shared=3,
                   seed=8, fmt="wav", dry_run=True)
    assert doc3["build_id"] != doc["build_id"]
    assert not (tmp_path / "p3").exists()
    order = lambda d: [M.deobfuscate(t["key"])["item"] for t in d["blocks"]["A"]["trials"]]  # noqa: E731
    assert order(doc3) != order(doc)


def test_build_gain_and_errors(tmp_path):
    root = write_eval_root(tmp_path / "loud", n_tracks=4, loud=True)
    doc = M.build(root / "eval", CONDS, tmp_path / "pg", segments_dir=root / "segments", n_trials=3, n_shared=0,
                  blocks=["A"], seed=0, fmt="wav")
    for t in doc["blocks"]["A"]["trials"]:
        k = M.deobfuscate(t["key"])
        assert k["gain"] < 1.0
        y, _ = sf.read(str(tmp_path / "pg" / t["context"]), dtype="float32", always_2d=True)
        assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-3
    with pytest.raises(SystemExit, match="multiple of 3"):
        M.build(root / "eval", CONDS, tmp_path / "x", segments_dir=root / "segments", n_trials=4, fmt="wav",
                dry_run=True)
    with pytest.raises(SystemExit, match="eligible regions"):
        M.build(root / "eval", CONDS, tmp_path / "x", segments_dir=root / "segments", n_trials=30, fmt="wav",
                dry_run=True)
    with pytest.raises(SystemExit, match="no such directory"):
        M.build(root / "eval", ["nope"], tmp_path / "x", segments_dir=root / "segments", n_trials=3, dry_run=True)


def test_regions_missing_in_a_condition_are_ineligible(tmp_path):
    root = write_eval_root(tmp_path / "r", n_tracks=3)
    for p in (root / "eval" / "E6k" / "inpaint").glob(f"{track_id(0)}*"):
        p.unlink()
    doc = M.build(root / "eval", CONDS, tmp_path / "pg", segments_dir=root / "segments", n_trials=6, n_shared=0,
                  blocks=["A"], seed=0, fmt="wav", dry_run=True)
    assert M.deobfuscate(doc["meta"])["n_regions_common"] == 6
    assert all(M.deobfuscate(t["key"])["track_id"] != track_id(0) for t in doc["blocks"]["A"]["trials"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_mp3_export(tmp_path):
    root = write_eval_root(tmp_path / "m", n_tracks=2)
    doc = M.build(root / "eval", CONDS, tmp_path / "pg", segments_dir=root / "segments", n_trials=3, n_shared=0,
                  blocks=["A"], seed=0, fmt="mp3")
    t = doc["blocks"]["A"]["trials"][0]
    assert doc["audio"] == {"format": "mp3", "bitrate": "192k", "dir": "audio"}
    for f in t["excerpts"].values():
        p = tmp_path / "pg" / f
        assert p.suffix == ".mp3" and p.stat().st_size > 1000
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "default=nw=1:nk=1", str(p)], capture_output=True, text=True, check=True).stdout)
        assert dur == pytest.approx(t["excerpt_s"], abs=0.1)


# --------------------------------------------------------------------------- scoring


def test_stat_helpers():
    lo, hi = S.wilson_ci(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    lo, hi = S.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert all(math.isnan(x) for x in S.wilson_ci(0, 0))
    # P(X >= 7 | n=10, p=1/3) by hand
    p = sum(math.comb(10, j) * (1 / 3) ** j * (2 / 3) ** (10 - j) for j in range(7, 11))
    assert S.binom_p(7, 10) == pytest.approx(p, rel=1e-9)
    assert S.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert S.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert math.isnan(S.cohen_kappa([1, 1], [1, 1]))
    assert math.isnan(S.cohen_kappa([], []))
    lo, hi = S.bootstrap_mean_ci([1, 2, 3, 4, 5] * 4, n_boot=2000, seed=1)
    assert lo < 3 < hi and (lo, hi) == S.bootstrap_mean_ci([1, 2, 3, 4, 5] * 4, n_boot=2000, seed=1)


def fake_responses(doc: dict, block: str, rater: str, policy, rating_of, build_id=None) -> dict:
    keys = M.decode_keys(doc)
    rs = []
    for t in doc["blocks"][block]["trials"]:
        k = keys[(block, t["id"])]
        ok = policy(k)
        choice = k["answer"] if ok else next(l for l in M.LETTERS if l != k["answer"])
        rs.append({"trial_id": t["id"], "item": t["item"], "index": t["index"], "choice": choice, "correct": ok,
                   "rating": rating_of(k), "comment": "", "plays": {"A": 1, "B": 1, "C": 1, "context": 1},
                   "shown_at": "2026-10-14T10:00:00Z", "choice_at": "2026-10-14T10:00:10Z",
                   "rated_at": "2026-10-14T10:00:20Z", "submitted_at": "2026-10-14T10:00:21Z"})
    return {"format": "ravi-listening-test-v2-responses", "build_id": build_id or doc["build_id"], "rater": rater,
            "block": block, "responses": rs}


def test_score_synthetic(built, tmp_path):
    _, out, doc = built
    ratings = {"base": 2, "B6k": 3, "E6k": 5}
    # Ann always finds base, never E6k, B6k on even section index; Bob the same rule
    pol = lambda k: k["condition"] == "base" or (k["condition"] == "B6k" and k["section_index"] % 2 == 0)  # noqa: E731
    fa, fb = tmp_path / "responses_Ann.json", tmp_path / "responses_Bob.json"
    fa.write_text(json.dumps(fake_responses(doc, "A", "Ann", pol, lambda k: ratings[k["condition"]])))
    fb.write_text(json.dumps(fake_responses(doc, "B", "Bob", pol, lambda k: ratings[k["condition"]])))
    trials = S.load_trials(out / "trials.json")
    rows = S.load_responses([fa, fb], trials)
    assert len(rows) == 18
    res = S.score(trials, rows, n_boot=500)
    pc = {x["condition"]: x for x in res["per_condition"]}
    assert [x["condition"] for x in res["per_condition"]] == list(CONDS)
    assert pc["base"]["k"] == 6 and pc["base"]["n"] == 6 and pc["base"]["accuracy"] == 1.0
    assert pc["E6k"]["k"] == 0
    assert pc["base"]["p_greater"] == pytest.approx((1 / 3) ** 6)
    assert pc["base"]["p_greater_holm"] == pytest.approx(min(1, 3 * (1 / 3) ** 6))
    assert pc["E6k"]["p_greater"] == pytest.approx(1.0)
    exp_b = sum(1 for r in rows if r["condition"] == "B6k" and r["section_index"] % 2 == 0)
    assert pc["B6k"]["k"] == exp_b
    for c, v in ratings.items():
        assert pc[c]["rating_mean"] == v and pc[c]["rating_ci_lo"] == v == pc[c]["rating_ci_hi"]
    (ag,) = res["agreement"]
    assert ag["n_items"] == 3 and ag["agreement"] == 1.0
    # the shared items are one per condition: base correct, E6k wrong -> both categories, kappa 1
    assert ag["kappa"] == pytest.approx(1.0)
    md = S.markdown(res)
    for c in CONDS:
        assert f"| {c} |" in md
    assert "Cohen's kappa" in md and "Ann / Bob" in md


def test_score_disagreement_and_files(built, tmp_path):
    _, out, doc = built
    trials = S.load_trials(out / "trials.json")
    shared = set(M.deobfuscate(doc["meta"])["shared_items"])
    fa = tmp_path / "responses_Ann.json"
    fa.write_text(json.dumps(fake_responses(doc, "A", "Ann", lambda k: True, lambda k: 4)))
    order = sorted(shared)
    fb = tmp_path / "responses_Bob.json"
    fb.write_text(json.dumps(fake_responses(doc, "B", "Bob", lambda k: k["item"] == order[0], lambda k: 1)))
    res = S.score(trials, S.load_responses([fa, fb], trials), n_boot=200)
    (ag,) = res["agreement"]
    assert ag["n_items"] == 3 and ag["agreement"] == pytest.approx(1 / 3)
    assert math.isnan(ag["kappa"]) is False and ag["kappa"] == pytest.approx(0.0)  # Ann constant -> pe = po
    # build mismatch is refused; a later file for the same rater/block wins
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(fake_responses(doc, "A", "Ann", lambda k: True, lambda k: 4, build_id="zzz")))
    with pytest.raises(SystemExit, match="build_id"):
        S.load_responses([bad], trials)
    later = fake_responses(doc, "A", "Ann", lambda k: False, lambda k: 2)
    for r in later["responses"]:
        r["submitted_at"] = "2026-10-15T00:00:00Z"
    fl = tmp_path / "responses_Ann_later.json"
    fl.write_text(json.dumps(later))
    rows = S.load_responses([fa, fl], trials)
    assert len(rows) == 9 and not any(r["correct"] for r in rows)
    # a tampered `correct` only warns; the key decides
    msgs = []
    tam = fake_responses(doc, "A", "Ann", lambda k: False, lambda k: 2)
    for r in tam["responses"]:
        r["correct"] = True
    ft = tmp_path / "tam.json"
    ft.write_text(json.dumps(tam))
    rows = S.load_responses([ft], trials, warn=msgs.append)
    assert len(msgs) == 9 and not any(r["correct"] for r in rows)
    # partial responses: a choice without a rating counts for the 3AFC only
    part = fake_responses(doc, "A", "Ann", lambda k: True, lambda k: None)
    part["responses"] = part["responses"][:4]
    fp = tmp_path / "part.json"
    fp.write_text(json.dumps(part))
    res = S.score(trials, S.load_responses([fp], trials), n_boot=100)
    assert res["overall"]["n"] == 4 and res["overall"]["n_rating"] == 0


def test_score_cli(built, tmp_path):
    _, out, doc = built
    fa = tmp_path / "responses_Ann.json"
    fa.write_text(json.dumps(fake_responses(doc, "A", "Ann", lambda k: True, lambda k: 3)))
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    assert S.main([str(out / "trials.json"), str(fa), "--out-md", str(md), "--out-json", str(js),
                   "--n-boot", "100"]) == 0
    assert "| all | 9/9 |" in md.read_text()
    j = json.loads(js.read_text())
    assert j["overall"]["k"] == 9 and len(j["responses"]) == 9 and j["agreement"] == []
