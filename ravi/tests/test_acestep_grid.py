"""Tests for ravi.eval.acestep_grid and the grid's ``engine: acestep`` option (no torch, no
ACE-Step: the backend is faked). The regions corpus comes from test_eval_grid."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from ravi.eval import acestep_grid as AG
from ravi.eval import grid as G
from ravi.eval import sa3_infer as si
from ravi.tests.test_eval_grid import FakeSAM, _hf_token, corpus, select  # noqa: F401 - fixtures

NSR = AG.NATIVE_SR


class FakeBackend:
    """ACE-Step stand-in: repaint keeps the 48 kHz source outside the mask and fills the mask
    with seeded noise (scale-dependent); text2music returns seeded noise."""

    def __init__(self):
        self.scale: float | None = None
        self.calls: list[dict] = []

    def set_lora(self, scale):
        self.scale = scale

    def _rng(self, seed):
        return np.random.default_rng(seed * 1000 + int(100 * (self.scale if self.scale is not None else -1)))

    def repaint(self, *, src_path, start, end, caption, lyrics, seed, steps, shift, guidance, mode, strength):
        src, sr = sf.read(src_path, dtype="float32", always_2d=True)
        assert sr == NSR and src.shape[1] == 2
        self.calls.append(dict(task="repaint", start=start, end=end, caption=caption, lyrics=lyrics, seed=seed,
                               steps=steps, shift=shift, guidance=guidance, mode=mode, strength=strength,
                               scale=self.scale, n=len(src)))
        y = src.copy()
        s0, s1 = int(start * NSR), int(end * NSR)
        y[s0:s1] = 0.1 * self._rng(seed).standard_normal((s1 - s0, 2))
        return y

    def text2music(self, *, caption, lyrics, duration, seed, steps, shift, guidance):
        self.calls.append(dict(task="text2music", caption=caption, lyrics=lyrics, duration=duration, seed=seed,
                               steps=steps, shift=shift, guidance=guidance, scale=self.scale))
        return (0.1 * self._rng(seed).standard_normal((int(duration * NSR), 2))).astype(np.float32)


class AceFactory:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.loads: list[tuple[str, str | None]] = []
        self.backends: list[FakeBackend] = []

    def __call__(self, model, lora):
        self.loads.append((model, lora))
        b = FakeBackend()
        self.backends.append(b)
        return AG.AceGridEngine(b, model, lora, tmp_dir=str(self.tmp))

    @property
    def calls(self) -> list[dict]:
        return [c for b in self.backends for c in b.calls]


def write_adapter_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 64, "lora_alpha": 128}))
    (path / "adapter_model.safetensors").write_bytes(b"\0" * 64)
    (path.parent / "run_meta.json").write_text(json.dumps({"run_id": "ACE_B-x", "arm": "ACE_B", "status": "completed",
                                                           "identity": {"prefix": "ACE_B_s42_habc123"}}))
    return path


@pytest.fixture
def ace(corpus) -> SimpleNamespace:  # noqa: F811 - the imported fixture
    regions = corpus.root / "eval" / "regions_test.json"
    select(corpus, regions)
    adapter = write_adapter_dir(corpus.root / "runs" / "acestep_B_tier1" / "model" / "adapter")
    common = {
        "regions": str(regions),
        "audio_root": str(corpus.prep),
        "seeds": [1],
        "tasks": ["inpaint", "generate"],
        "inpaint": {"context_s": 20.0},
        "generate": {"duration": 2, "prompts_from": "arm", "n_prompts": 2, "seed": 0},
        "anchors": {"lower": False},
    }
    raw = {**common, "engine": "acestep", "out_root": str(corpus.root / "ace"),
           "defaults": {"prompt_source": "arm_caption", "arm": "B"},
           "conditions": [
               {"name": "ACE_base"},
               {"name": "ACE_B6k", "lora": str(adapter), "lora_strength": 1.0},
               {"name": "ACE_B6k_s0", "lora": str(adapter), "lora_strength": 0.0},
               {"name": "ACE_E6k", "arm": "E", "with_sections": True, "lora": str(adapter)},
           ]}
    sa3 = {**common, "out_root": str(corpus.root / "sa3"), "defaults": {"model": "medium", "arm": "B"},
           "conditions": [{"name": "base"}]}
    return SimpleNamespace(corpus=corpus, raw=raw, sa3=sa3, adapter=adapter, out=corpus.root / "ace")


def cfg_of(raw: dict) -> G.GridConfig:
    return G.resolve_config(raw, {}, source="test")


def test_split_sections():
    assert AG.split_sections("rvkr, a calm tune, Rock, sections: intro, verse, chorus") == \
        ("rvkr, a calm tune, Rock", "[intro]\n[verse]\n[chorus]")
    assert AG.split_sections("rvkr, sections: verse") == ("rvkr", "[verse]")
    assert AG.split_sections("sections: verse, outro") == ("", "[verse]\n[outro]")
    assert AG.split_sections("rvkr, a calm tune, Rock") == ("rvkr, a calm tune, Rock", "[Instrumental]")
    assert AG.split_sections("  rvkr,   x ") == ("rvkr, x", "[Instrumental]")


def test_condition_models_and_defaults(ace):
    cfg = cfg_of(ace.raw)
    assert cfg.engine == "acestep"
    assert [(c.model, c.steps, c.cfg) for c in cfg.conditions] == [("turbo", 8, 1.0)] * 4
    c = cfg_of({**ace.raw, "conditions": [{"name": "b", "model": "base"}]}).conditions[0]
    assert (c.steps, c.cfg) == (50, 7.0)
    with pytest.raises(SystemExit, match="acestep model must be one of"):
        cfg_of({**ace.raw, "conditions": [{"name": "m", "model": "medium"}]})
    with pytest.raises(SystemExit, match="engine must be one of"):
        cfg_of({**ace.raw, "engine": "foo"})
    # SA3 configs are unchanged: no engine key -> sa3, default model medium
    s = cfg_of(ace.sa3)
    assert s.engine == "sa3" and s.conditions[0].model == "medium"


def test_lora_must_be_an_adapter_dir(ace, tmp_path):
    raw = {**ace.raw, "conditions": [{"name": "x", "lora": str(ace.adapter / "adapter_model.safetensors")}]}
    with pytest.raises(SystemExit, match="no such acestep adapter"):
        G.run_grid(cfg_of(raw), engine_factory=AceFactory(tmp_path))
    s = G.run_grid(cfg_of(raw), engine_factory=AceFactory(tmp_path), dry=True)
    assert s["missing"]["lora"] == [f"x: {ace.adapter / 'adapter_model.safetensors'}"]


def test_run_layout_prompts_lyrics_and_seeding(ace, tmp_path):
    f = AceFactory(tmp_path / "tmp")
    s = G.run_grid(cfg_of(ace.raw), engine_factory=f)
    assert not s["failures"]
    assert f.loads == [("turbo", None), ("turbo", str(ace.adapter))]
    # one load for the three LoRA conditions; the scale follows each condition
    assert [c["scale"] for c in f.backends[1].calls if c["task"] == "repaint"][::10] == [1.0, 0.0, 1.0]
    rep = [c for c in f.calls if c["task"] == "repaint"]
    assert len(rep) == 4 * 10 and all((c["steps"], c["shift"], c["guidance"]) == (8, 3.0, 1.0) for c in rep)
    assert all((c["mode"], c["strength"]) == ("balanced", 0.5) for c in rep)
    for cond in ("ACE_base", "ACE_B6k", "ACE_B6k_s0", "ACE_E6k"):
        inp = ace.out / cond / "inpaint"
        sidecars = sorted(inp.glob("*.json"))
        assert len(sidecars) == 10
        for js in sidecars:
            meta = json.loads(js.read_text())
            assert meta["spliced_outside_mask_bit_identical"] is True
            assert meta["engine"]["engine"] == "acestep" and meta["engine"]["native"]["sample_rate"] == NSR
            call = meta["engine_call"]
            assert call["task"] == "repaint" and call["torch_manual_seed"] == meta["seed"]
            m = meta["mask_in_context"]
            assert (call["repainting_start"], call["repainting_end"]) == (m["passed_start_s"], m["passed_end_s"])
            for k in ("raw", "spliced", "orig"):
                assert sf.info(str(inp / f"{js.stem}_{k}.wav")).samplerate == 44100
            if cond == "ACE_E6k":
                assert call["lyrics"].startswith("[") and "sections:" not in call["caption"]
                assert "sections:" in meta["prompt_rendered"]
            else:
                assert call["lyrics"] == "[Instrumental]"
        if cond != "ACE_base":
            assert json.loads(sidecars[0].read_text())["lora"]["adapter_config"]["r"] == 64
        gen = sorted((ace.out / cond / "gen").glob("*.json"))
        assert len(gen) == 2
        g = json.loads(gen[0].read_text())
        assert g["engine_call"]["task"] == "text2music" and g["engine_call"]["duration"] == 2.0
        assert sf.info(str(ace.out / cond / "gen" / g["file"])).samplerate == 44100
        assert g["n_samples"] == 2 * 44100
    # the repainted region differs between scale 0 and 1 (the fake backend's noise depends on it)
    a = sorted((ace.out / "ACE_B6k" / "inpaint").glob("*_raw.wav"))[0]
    b = ace.out / "ACE_B6k_s0" / "inpaint" / a.name
    assert not np.array_equal(sf.read(str(a))[0], sf.read(str(b))[0])


def test_masks_and_contexts_match_the_sa3_grid(ace, tmp_path):
    """run_eval pairs regions across conditions by mask samples: the ACE-Step grid must match SA3's."""
    G.run_grid(cfg_of(ace.raw), engine_factory=AceFactory(tmp_path / "tmp"))

    def sa3_factory(model, lora):
        return si.Engine(FakeSAM(sr=44100, ds=4096), model, lora, to_tensor=lambda x: np.ascontiguousarray(x.T),
                         to_numpy=np.asarray)

    G.run_grid(cfg_of(ace.sa3), engine_factory=sa3_factory)
    sa3_dir = Path(ace.sa3["out_root"]) / "base" / "inpaint"
    ace_dir = ace.out / "ACE_base" / "inpaint"
    names = sorted(p.name for p in sa3_dir.glob("*.json"))
    assert names == sorted(p.name for p in ace_dir.glob("*.json")) and len(names) == 10
    for n in names:
        a, b = json.loads((sa3_dir / n).read_text()), json.loads((ace_dir / n).read_text())
        for k in ("mask", "context", "mask_in_context"):
            assert {x: a[k][x] for x in a[k] if "sample" in x} == {x: b[k][x] for x in b[k] if "sample" in x}, (n, k)
        assert a["audio"]["sha256"] == b["audio"]["sha256"]
        assert np.array_equal(sf.read(str(sa3_dir / n.replace(".json", "_orig.wav")))[0],
                              sf.read(str(ace_dir / n.replace(".json", "_orig.wav")))[0])


def test_outputs_readable_by_run_eval(ace, tmp_path):
    run_eval = pytest.importorskip("ravi.eval.run_eval")
    G.run_grid(cfg_of(ace.raw), engine_factory=AceFactory(tmp_path / "tmp"))
    conds = {c.name: c for c in run_eval.discover_conditions(ace.out)}
    assert set(conds) == {"ACE_base", "ACE_B6k", "ACE_B6k_s0", "ACE_E6k"}
    regs = run_eval.read_regions(conds["ACE_E6k"].inpaint_dir, "spliced")
    assert len(regs) == 10 and set(next(iter(regs.values())).files) == {"raw", "spliced", "orig"}
    assert len(run_eval.read_gen(conds["ACE_B6k"].gen_dir, None)) == 2


def test_engine_strength_range_and_resampling(tmp_path):
    e = AG.AceGridEngine(FakeBackend(), "turbo", str(tmp_path), tmp_dir=str(tmp_path))
    with pytest.raises(ValueError, match="clamps"):
        e.set_lora_strength(1.5)
    with pytest.raises(ValueError):
        AG.AceGridEngine(FakeBackend(), "turbo", None, tmp_dir=str(tmp_path)).set_lora_strength(1.0)
    ctx = (0.1 * np.random.default_rng(0).standard_normal((44100, 2))).astype(np.float32)
    y = e.generate(prompt="rvkr, x", duration=1.0, steps=8, cfg_scale=1.0, seed=3, inpaint_audio=ctx,
                   mask_start_s=0.25, mask_end_s=0.5)
    assert y.shape == (44100, 2) and y.dtype == np.float32
    # outside the mask: the 44.1 -> 48 -> 44.1 kHz round trip only (polyphase, small error)
    assert np.abs(y[:8000] - ctx[:8000]).max() < 0.05
    assert e.last_call["native_source_samples"] == 48000


def test_shipped_tier1_ace_config_resolves():
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[2] / "configs" / "eval" / "tier1_ace.yaml"
    cfg = G.resolve_config(yaml.safe_load(path.read_text()), source=str(path))
    assert cfg.engine == "acestep" and cfg.seeds == [1, 2] and cfg.lower_name == "anchor_lower"
    assert cfg.regions == "/eval/regions_test.json" and cfg.inpaint["context_s"] == 47
    assert {c.name: (c.with_sections, c.arm) for c in cfg.conditions} == {
        "ACE_base": (False, "B"), "ACE_base_sec": (True, "B"), "ACE_B6k": (False, "B"), "ACE_B6k_sec": (True, "B"),
        "ACE_E6k": (True, "E"), "ACE_E6k_nosec": (False, "E")}
    assert all(c.model == "turbo" and (c.steps, c.cfg) == (8, 1.0) for c in cfg.conditions)
    assert {c.lora for c in cfg.conditions} == {None, "/runs/acestep_B_tier1/model/adapter",
                                               "/runs/acestep_E_tier1/model/adapter"}
    sa3 = G.resolve_config(yaml.safe_load((path.parent / "tier1.yaml").read_text()))
    for k in ("regions", "audio_root", "seeds", "inpaint", "generate", "geometry"):
        assert getattr(cfg, k) == getattr(sa3, k), k
