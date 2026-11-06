"""Tests for ravi.eval.select_regions and ravi.eval.grid (no torch, no GPU: the SA3 model is
faked with an object whose generate() returns silence)."""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ravi.eval import grid as G
from ravi.eval import sa3_infer as si
from ravi.eval import select_regions as SR_
from ravi.structure import schema
from ravi.train.caption_aug import FIELD_SEP

SR = 1000  # fake model rate (small files); the prepared WAVs are written at this rate too
DS = 32
GEOM = {"sample_rate": SR, "downsampling_ratio": DS, "sample_size": 120 * SR, "channels": 2}

# track id -> (duration, segments, alt_family); "t_nolab" has no labels, "t_noint" no interior section
SONGS = {
    "t0 Alpha": (40.0, [(0, 6, "intro"), (6, 18, "verse"), (18, 30, "chorus"), (30, 40, "outro")], 0),
    "t1 Beta": (36.0, [(0, 4, "intro"), (4, 12, "verse"), (12, 13, "transition"), (13, 28, "bridge"),
                       (28, 36, "outro")], 1),
    "t2 Gamma": (50.0, [(0, 5, "intro"), (5, 15, "verse"), (15, 25, "verse"), (25, 38, "bridge"),
                        (38, 50, "outro")], 2),
    "t3 Delta": (30.0, [(0, 3, "intro"), (3, 11, "verse"), (11, 30, "outro")], 3),
    "t4 Eps": (34.0, [(0, 5, "intro"), (5, 14, "verse"), (14, 34, "outro")], 4),
    "t5 Eps alt": (34.0, [(0, 5, "intro"), (5, 14, "verse"), (14, 34, "outro")], 4),  # same family as t4
    "t_noint": (30.0, [(0, 10, "intro"), (10, 12, "transition"), (12, 30, "outro")], 5),
    "t_nolab": (30.0, None, 6),
    "tr0 Train": (32.0, [(0, 8, "intro"), (8, 20, "verse"), (20, 32, "outro")], 7),
}
TEST_IDS = [t for t in SONGS if not t.startswith("tr")]


@pytest.fixture(autouse=True)
def _hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")


def write_fake_adapter(path: Path) -> Path:
    md = {"lora_config": json.dumps({"rank": 16, "alpha": 16}), "ravi_arm": "B", "ravi_step": "3000"}
    header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": md}
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + np.float32(1.0).tobytes())
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> SimpleNamespace:
    labels = tmp_path / "labels"
    labels.mkdir()
    prep = tmp_path / "prepared" / "44100"
    prep.mkdir(parents=True)
    rows, prows = [], []
    rng = np.random.default_rng(0)
    for i, (tid, (dur, segs, fam)) in enumerate(SONGS.items()):
        lp = labels / f"{tid}.mp3_labels.txt"
        if segs is not None:
            lp.write_text("".join(f"{a:.3f}\t{b:.3f}\t{lab}\n" for a, b, lab in segs))
        audio = (0.2 * rng.standard_normal((int(dur * SR), 2))).astype(np.float32)
        sf.write(prep / f"{tid}.wav", audio, SR, subtype="FLOAT")
        rows.append({
            "track_id": tid, "has_labels": segs is not None, "labels_path": f"/nowhere/{lp.name}",
            "human_caption": f"A bright tune number {i}.", "human_genre_tag": "Pop" if i % 2 else "Rock",
            "caption_is_junk": False, "id3_genre_tokens": ["Fun", "Pop"], "id3_bpm": None, "alt_family": fam,
            "duration_s": dur,
        })
        prows.append({"track_id": tid, "prepared_path_44k": f"prepared/44100/{tid}.wav",
                      "n_frames_44k": round(dur * 44100)})
    man = tmp_path / "tracks.parquet"
    pd.DataFrame(rows).to_parquet(man)
    pman = tmp_path / "prepared_audio.parquet"
    pd.DataFrame(prows).to_parquet(pman)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["tr0 Train"], "val": [], "test": TEST_IDS}))
    return SimpleNamespace(root=tmp_path, labels=labels, prep=prep, manifest=man, prepared=pman, splits=splits)


def select(corpus: SimpleNamespace, out: Path, *extra: str) -> dict:
    SR_.main(["--split", "test", "--manifest", str(corpus.manifest), "--prepared", str(corpus.prepared),
              "--splits", str(corpus.splits), "--labels-dir", str(corpus.labels), "--out", str(out), *extra])
    return json.loads(out.read_text())


# --------------------------------------------------------------------------- select_regions


def test_select_regions_counts_segments_and_determinism(corpus):
    out = corpus.root / "eval" / "regions_test.json"
    doc = select(corpus, out)
    # interior verse/bridge/chorus >= 4 s: t0 verse+chorus? chorus (18-30) is interior, verse is
    # interior too (intro first, outro last); t1 verse, bridge; t2 verse, verse, bridge; t3, t4, t5 verse
    want = {("t0 Alpha", 1), ("t0 Alpha", 2), ("t1 Beta", 1), ("t1 Beta", 3), ("t2 Gamma", 1), ("t2 Gamma", 2),
            ("t2 Gamma", 3), ("t3 Delta", 1), ("t4 Eps", 1), ("t5 Eps alt", 1)}
    assert {(r["track_id"], r["section_index"]) for r in doc["regions"]} == want
    assert doc["n_regions"] == len(want) == doc["n_candidates"]
    assert doc["n_tracks_split"] == len(TEST_IDS) and doc["n_tracks_eligible"] == 6
    assert "tr0 Train" not in doc["tracks"] and set(doc["tracks"]) == set(TEST_IDS)
    assert doc["skipped"] == {"t_nolab": "no human labels"}
    r = next(r for r in doc["regions"] if r["track_id"] == "t1 Beta" and r["section_index"] == 3)
    assert (r["start"], r["end"], r["duration"], r["label"]) == (12.0, 29.0, 17.0, "bridge")
    assert (r["section_start"], r["section_end"]) == (13.0, 28.0)
    assert r["audio_path"] == "prepared/44100/t1 Beta.wav" and r["alt_family"] == 1
    # the per-track schema JSON, relative to the regions file
    seg = out.parent / r["segments_json"]
    assert seg == out.parent / "segments" / "test" / "t1 Beta.json"
    sdoc = schema.load(seg)
    assert sdoc["tool"] == "human" and [s["label"] for s in sdoc["segments"]][:2] == ["intro", "verse"]
    assert (out.parent / "segments" / "test" / "t_noint.json").is_file()  # labelled, no region
    assert not (out.parent / "segments" / "test" / "t_nolab.json").exists()
    # arm captions as make_arms writes them (E without sections)
    caps = doc["tracks"]["t1 Beta"]["captions"]
    assert caps["A"] == "rvkr" and caps["B"] == f"rvkr{FIELD_SEP}A bright tune number 1{FIELD_SEP}Pop"
    assert caps["E"] == caps["B"]
    # byte-identical on a rerun
    first = out.read_bytes()
    select(corpus, out)
    assert out.read_bytes() == first


def test_select_regions_limits_are_seeded_and_stable(corpus):
    full = select(corpus, corpus.root / "eval" / "all.json")
    a = select(corpus, corpus.root / "eval" / "a.json", "--max-tracks", "3", "--max-regions-per-track", "1")
    b = select(corpus, corpus.root / "eval" / "b.json", "--max-tracks", "3", "--max-regions-per-track", "1")
    assert a["regions"] == b["regions"]
    assert a["n_tracks"] == 3 and a["n_regions"] == 3
    assert len({r["track_id"] for r in a["regions"]}) == 3
    assert a["candidates"] == full["regions"]  # the donor pool is always the whole split
    # the per-track region draw does not depend on --max-tracks
    k1 = select(corpus, corpus.root / "eval" / "k1.json", "--max-regions-per-track", "1")
    pick = {r["track_id"]: r["section_index"] for r in k1["regions"]}
    assert all(pick[r["track_id"]] == r["section_index"] for r in a["regions"])
    assert k1["n_regions"] == 6
    other = select(corpus, corpus.root / "eval" / "c.json", "--max-tracks", "3", "--seed", "7")
    assert other["seed"] == 7  # a different seed is allowed to pick differently; still valid
    assert all(r in full["regions"] for r in other["regions"])


# --------------------------------------------------------------------------- fake model


class FakeSAM:
    def __init__(self, sr: int = SR, ds: int = DS):
        self.model = SimpleNamespace(sample_rate=sr, io_channels=64,
                                     pretransform=SimpleNamespace(downsampling_ratio=ds, io_channels=2))
        self.model_config = {"sample_size": 120 * sr}
        self.calls: list[dict] = []
        self.strengths: list[float] = []

    def set_lora_strength(self, strength, lora_index=None):
        self.strengths.append(strength)

    def generate(self, **kw):
        self.calls.append(kw)
        return np.zeros((1, 2, int(kw["duration"] * self.model.sample_rate)), dtype=np.float32)


class Factory:
    def __init__(self):
        self.loads: list[tuple[str, str | None]] = []
        self.fakes: list[FakeSAM] = []

    def __call__(self, model, lora):
        self.loads.append((model, lora))
        fake = FakeSAM()
        self.fakes.append(fake)
        return si.Engine(fake, model, lora, to_tensor=lambda a: np.ascontiguousarray(a.T), to_numpy=np.asarray)

    @property
    def calls(self) -> list[dict]:
        return [c for f in self.fakes for c in f.calls]


def boom(model, lora):
    raise AssertionError("the model must not be loaded")


@pytest.fixture
def setup(corpus) -> SimpleNamespace:
    regions = corpus.root / "eval" / "regions_test.json"
    select(corpus, regions)
    adapter = write_fake_adapter(corpus.root / "runs" / "B" / "adapter.safetensors")
    raw = {
        "regions": str(regions),
        "out_root": str(corpus.root / "out"),
        "audio_root": str(corpus.prep),
        "seeds": [1],
        "tasks": ["inpaint", "generate"],
        "geometry": GEOM,
        "inpaint": {"context_s": 20.0},
        "generate": {"duration": 4, "prompts_from": "arm", "n_prompts": 3, "seed": 0},
        "anchors": {"lower": True},
        "defaults": {"model": "medium", "prompt_source": "arm_caption", "arm": "B"},
        "conditions": [
            {"name": "base"},
            {"name": "B_s0", "lora": str(adapter), "lora_strength": 0.0},
            {"name": "B_s1", "lora": str(adapter), "lora_strength": 1.0},
            {"name": "A_sec", "arm": "A", "with_sections": True, "lora": str(adapter)},
        ],
    }
    return SimpleNamespace(corpus=corpus, regions=regions, adapter=adapter, raw=raw, out=corpus.root / "out")


def cfg_of(raw: dict, **ov) -> G.GridConfig:
    return G.resolve_config(raw, ov, source="test")


# --------------------------------------------------------------------------- grid


def test_dry_run_plan_counts_and_loads_nothing(setup, capsys):
    raw = {**setup.raw, "seeds": [1, 2]}
    s = G.run_grid(cfg_of(raw), engine_factory=boom, dry=True)
    n_reg = 10
    assert set(s["conditions"]) == {"anchor_lower", "base", "B_s0", "B_s1", "A_sec"}
    for name in ("base", "B_s0", "B_s1", "A_sec"):
        c = s["conditions"][name]
        assert (c["inpaint"], c["generate"], c["done"]) == (2 * n_reg, 2 * 3, 0)
    assert s["conditions"]["anchor_lower"]["inpaint"] == 2 * n_reg
    t = s["totals"]
    assert (t["inpaint"], t["generate"], t["anchor"], t["model_loads"]) == (4 * 2 * n_reg, 4 * 2 * 3, 2 * n_reg, 2)
    assert s["missing"] == {"audio": [], "segments": [], "lora": [], "caption_files": []}
    assert "sections:" in s["conditions"]["A_sec"]["example_prompt"]
    assert not setup.out.exists()
    out = capsys.readouterr().out
    assert "totals:" in out and "estimate:" in out


def test_cli_dry_run_with_json_config(setup, tmp_path, capsys):
    cfgf = tmp_path / "grid.json"
    cfgf.write_text(json.dumps(setup.raw))
    assert G.main([str(cfgf), "--dry-run", "--conditions", "base"], engine_factory=boom) == 0
    out = capsys.readouterr().out
    assert "base" in out and "B_s1" not in out and "anchor_lower" not in out


def test_run_layout_manifest_sidecars_and_model_reuse(setup):
    f = Factory()
    s = G.run_grid(cfg_of(setup.raw), engine_factory=f)
    assert not s["failures"]
    # base loads once, the three LoRA conditions share one load (only the strength changes)
    assert f.loads == [("medium", None), ("medium", str(setup.adapter))]
    assert f.fakes[1].strengths == [0.0, 1.0, 1.0]
    for cond in ("base", "B_s0", "B_s1", "A_sec"):
        inp = setup.out / cond / "inpaint"
        sidecars = sorted(inp.glob("*.json"))
        assert len(sidecars) == 10
        for js in sidecars:
            stem = js.stem
            assert stem.endswith("_seed1") and "_sec" in stem
            for k in ("raw", "spliced", "orig"):
                assert (inp / f"{stem}_{k}.wav").is_file()
            meta = json.loads(js.read_text())
            assert meta["command"] == "inpaint" and meta["section"]["index"] >= 1
            assert meta["spliced_outside_mask_bit_identical"] is True
            assert meta["grid"]["condition"] == cond
        gen = sorted((setup.out / cond / "gen").glob("*.wav"))
        assert len(gen) == 3 and all(p.with_suffix(".json").is_file() for p in gen)
    # identical stems across conditions (run_eval pairs by stem)
    stems = {c: sorted(p.name for p in (setup.out / c / "inpaint").glob("*.json")) for c in ("base", "B_s1", "anchor_lower")}
    assert stems["base"] == stems["B_s1"] == stems["anchor_lower"]
    gstems = {c: sorted(p.name for p in (setup.out / c / "gen").glob("*.wav")) for c in ("base", "A_sec")}
    assert gstems["base"] == gstems["A_sec"]
    # arm A gives every track the same prompt; the derived seeds keep the clips distinct
    a_gen = [json.loads(p.read_text()) for p in sorted((setup.out / "A_sec" / "gen").glob("*.json"))]
    assert len({m["seed"] for m in a_gen}) == 3 and all(m["grid_seed"] == 1 for m in a_gen)
    # inpainting calls carry the source audio and the condition's sampling
    inpaint_calls = [c for c in f.calls if "inpaint_audio" in c]
    assert len(inpaint_calls) == 40 and all(c["steps"] == 8 and c["cfg_scale"] == 1.0 for c in inpaint_calls)
    # manifest: one row per output
    man = pd.read_parquet(setup.out / "manifest.parquet")
    assert len(man) == 10 + 4 * (10 + 3)
    assert set(man["task"]) == {"inpaint", "generate"}
    assert set(man.loc[man.condition == "anchor_lower", "anchor"]) == {"lower"}
    row = man[(man.condition == "B_s1") & (man.task == "inpaint")].iloc[0]
    assert row["lora_strength"] == 1.0 and Path(row["path"]).name.endswith("_spliced.wav")
    assert man.loc[man.task == "generate", "section_index"].isna().all()
    assert {"condition", "task", "track_id", "section_index", "seed", "path", "elapsed_s"} <= set(man.columns)
    assert (setup.out / "grid_log.jsonl").read_text().count("\n") == 5


def test_resume_skips_existing_outputs(setup):
    G.run_grid(cfg_of(setup.raw), engine_factory=Factory())
    f2 = Factory()
    s = G.run_grid(cfg_of(setup.raw), engine_factory=f2)
    assert f2.loads == [] and all(c["n_done"] == 0 for c in s["conditions"].values())
    victim = sorted((setup.out / "B_s1" / "inpaint").glob("*.json"))[3]
    victim.unlink()
    f3 = Factory()
    G.run_grid(cfg_of(setup.raw), engine_factory=f3)
    assert f3.loads == [("medium", str(setup.adapter))] and len(f3.calls) == 1
    assert f3.fakes[0].strengths == [1.0] and victim.is_file()
    assert len(pd.read_parquet(setup.out / "manifest.parquet")) == 10 + 4 * 13


def test_lower_anchor_pairing_deterministic_length_matched_other_family(setup):
    doc = json.loads(setup.regions.read_text())
    regs, cands = doc["regions"], doc["candidates"]
    p1 = G.pair_lower_anchors(regs, cands, seed=0)
    shuffled_r, shuffled_c = regs[:], cands[:]
    random.Random(3).shuffle(shuffled_r)
    random.Random(4).shuffle(shuffled_c)
    assert G.pair_lower_anchors(shuffled_r, shuffled_c, seed=0) == p1
    for r in regs:
        d = p1[G.region_key(r)]
        assert d["track_id"] != r["track_id"] and d["alt_family"] != r["alt_family"]
        dur = r["end"] - r["start"]
        ok = [c for c in cands if c["track_id"] != r["track_id"] and c["alt_family"] != r["alt_family"]]
        best = min(abs((c["end"] - c["start"]) - dur) for c in ok)
        assert abs((d["end"] - d["start"]) - dur) == pytest.approx(best)
    # t4's only same-length match would be its alt version t5 (same family): never chosen
    t4 = next(r for r in regs if r["track_id"] == "t4 Eps")
    assert p1[G.region_key(t4)]["track_id"] != "t5 Eps alt"


def test_lower_anchor_files_splice_only_inside_the_mask(setup):
    cfg = cfg_of(setup.raw, conditions_filter="anchor_lower")
    G.run_grid(cfg, engine_factory=boom)
    inp = setup.out / "anchor_lower" / "inpaint"
    sidecars = sorted(inp.glob("*.json"))
    assert len(sidecars) == 10 and not (setup.out / "base").exists()
    for js in sidecars:
        meta = json.loads(js.read_text())
        assert meta["anchor"] == "lower" and meta["command"] == "inpaint"
        assert meta["donor"]["track_id"] != meta["track_id"]
        a, _ = sf.read(inp / f"{js.stem}_spliced.wav", dtype="float32", always_2d=True)
        o, _ = sf.read(inp / f"{js.stem}_orig.wav", dtype="float32", always_2d=True)
        m0, m1 = meta["mask_in_context"]["start_sample"], meta["mask_in_context"]["end_sample"]
        assert a.shape == o.shape == (meta["context"]["end_sample"] - meta["context"]["start_sample"], 2)
        assert np.array_equal(a[:m0], o[:m0]) and np.array_equal(a[m1:], o[m1:])
        assert not np.allclose(a[m0:m1], o[m0:m1])
        # the fill is the donor's audio centred on the donor region
        src, _ = sf.read(setup.corpus.prep / f"{meta['donor']['track_id']}.wav", dtype="float32", always_2d=True)
        off = meta["donor"]["fill_offset_sample"]
        raw, _ = sf.read(inp / f"{js.stem}_raw.wav", dtype="float32", always_2d=True)
        assert np.array_equal(raw[m0:m1], src[off:off + (m1 - m0)])


def test_outputs_readable_by_run_eval(setup):
    run_eval = pytest.importorskip("ravi.eval.run_eval")
    G.run_grid(cfg_of(setup.raw), engine_factory=Factory())
    conds = {c.name: c for c in run_eval.discover_conditions(setup.out)}
    assert set(conds) == {"anchor_lower", "base", "B_s0", "B_s1", "A_sec"}
    regs = run_eval.read_regions(conds["base"].inpaint_dir, "spliced")
    low = run_eval.read_regions(conds["anchor_lower"].inpaint_dir, "spliced")
    assert len(regs) == 10 and set(regs) == set(low)
    r = next(iter(regs.values()))
    assert r.track_id in TEST_IDS and set(r.files) == {"raw", "spliced", "orig"}
    clips = run_eval.read_gen(conds["A_sec"].gen_dir, None)
    assert len(clips) == 3 and all(c["prompt"].startswith("rvkr, sections:") for c in clips)


def test_prompts_sections_and_sources(setup, tmp_path):
    raw = setup.raw
    cfg = cfg_of({**raw, "conditions": [
        {"name": "b"}, {"name": "b_sec", "with_sections": True}, {"name": "e", "arm": "E"},
        {"name": "e_nosec", "arm": "E", "with_sections": False}, {"name": "h", "prompt_source": "human"},
        {"name": "n", "prompt_source": "none"},
        {"name": "e_file", "arm": "E", "caption_root": str(tmp_path / "arms")},
        {"name": "e_file_keep", "arm": "E", "caption_root": str(tmp_path / "arms"), "sections_from": "caption"},
    ]})
    rs = G.RegionSet(cfg.regions, cfg.audio_root)
    pr = G.Prompter(cfg, rs)
    c = {x.name: x for x in cfg.conditions}
    tid = "t2 Gamma"
    win = (14.0, 40.0)  # verse (15-25 overlaps 10 s, 5-15 overlaps 1 s), bridge, outro (38-50: 2 s)
    assert c["e"].with_sections and not c["b"].with_sections
    assert pr.build(c["b"], tid, win)["prompt_rendered"] == "rvkr, A bright tune number 2, Rock"
    assert pr.build(c["b_sec"], tid, win)["prompt_rendered"] == \
        "rvkr, A bright tune number 2, Rock, sections: verse, verse, bridge, outro"
    assert pr.build(c["e"], tid, win)["prompt_rendered"] == pr.build(c["b_sec"], tid, win)["prompt_rendered"]
    assert pr.build(c["e"], tid, (0.0, 12.0))["sections"] == ["intro", "verse"]
    assert pr.build(c["e_nosec"], tid, win)["prompt_rendered"] == "rvkr, A bright tune number 2, Rock"
    assert pr.build(c["h"], tid, win)["prompt_rendered"] == "A bright tune number 2"
    assert pr.build(c["n"], tid, win)["prompt_rendered"] == ""
    # a caption file wins; its sections field is recomputed from the window unless sections_from=caption
    f = tmp_path / "arms" / "E" / "test" / f"{tid}.txt"
    f.parent.mkdir(parents=True)
    f.write_text(FIELD_SEP.join(["rvkr", "from file", "sections: intro, verse, outro"]))
    p = pr.build(c["e_file"], tid, win)
    assert p["prompt_rendered"] == "rvkr, from file, sections: verse, verse, bridge, outro"
    assert p["prompt_origin"].startswith("file:")
    assert pr.build(c["e_file_keep"], tid, win)["prompt_rendered"] == "rvkr, from file, sections: intro, verse, outro"


def test_generation_prompts_from_jsonl(setup, tmp_path):
    jl = tmp_path / "auto.jsonl"
    jl.write_text("".join(json.dumps({"track_id": t, "caption": f"auto {i}"}) + "\n" for i, t in enumerate(TEST_IDS[:4])))
    raw = {**setup.raw, "tasks": ["generate"], "anchors": {"lower": False},
           "generate": {"duration": 2, "prompts_from": str(jl), "n_prompts": None},
           "conditions": [{"name": "base"}]}
    f = Factory()
    G.run_grid(cfg_of(raw), engine_factory=f)
    metas = [json.loads(p.read_text()) for p in sorted((setup.out / "base" / "gen").glob("*.json"))]
    assert [m["track_id"] for m in metas] == sorted(TEST_IDS[:4])
    assert all(m["prompt_rendered"].startswith("rvkr, auto ") for m in metas)
    assert not (setup.out / "base" / "inpaint").exists()


def test_config_validation(setup):
    with pytest.raises(SystemExit, match="reserved"):
        cfg_of({**setup.raw, "conditions": [{"name": "lower"}]})
    with pytest.raises(SystemExit, match="lora_strength needs a lora"):
        cfg_of({**setup.raw, "conditions": [{"name": "x", "lora_strength": 0.5}]})
    with pytest.raises(SystemExit, match="duplicate"):
        cfg_of({**setup.raw, "conditions": [{"name": "x"}, {"name": "x"}]})
    with pytest.raises(SystemExit, match="unknown keys"):
        cfg_of({**setup.raw, "conditions": [{"name": "x", "strength": 1}]})
    c = cfg_of({**setup.raw, "conditions": [{"name": "mb", "model": "medium-base"}]}).conditions[0]
    assert (c.steps, c.cfg) == (50, 7.0)
    assert G.SECTIONS_PREFIX == "sections:"


@pytest.mark.parametrize("name, n_cond, seeds", [("tier0", 4, [1]), ("tier1", 8, [1, 2])])
def test_shipped_configs_resolve(name, n_cond, seeds):
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[2] / "configs" / "eval" / f"{name}.yaml"
    cfg = G.resolve_config(yaml.safe_load(path.read_text()), source=str(path))
    assert len(cfg.conditions) == n_cond and cfg.seeds == seeds and cfg.lower_name == "anchor_lower"
    assert all(c.model == "medium" and c.prompt_source == "arm_caption" for c in cfg.conditions)
    if name == "tier0":
        assert [c.lora_strength for c in cfg.conditions] == [None, 0.0, 0.5, 1.0]
        assert {c.lora for c in cfg.conditions[1:]} == {"/runs/B_tier0/model/adapter.safetensors"}
    else:
        secs = {c.name: c.with_sections for c in cfg.conditions}
        assert secs == {"base": False, "base_sec": True, "A6k": False, "A6k_sec": True, "B6k": False,
                        "B6k_sec": True, "E6k": True, "E6k_nosec": False}
