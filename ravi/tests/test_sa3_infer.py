"""Tests for ravi.eval.masks and ravi.eval.sa3_infer (no torch, no GPU: the SA3 model is faked)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from ravi.eval import masks as mk
from ravi.eval import sa3_infer as si
from ravi.structure import schema
from ravi.structure.labels import Segment
from ravi.train.caption_aug import FIELD_SEP

SR = 8000  # fake model rate: small files; the logic is rate-agnostic
DS = 256  # fake downsampling ratio


# --------------------------------------------------------------------------- helpers


SONG = [
    (0.0, 8.0, "intro"),
    (8.0, 20.0, "verse"),
    (20.0, 32.0, "chorus"),
    (32.0, 34.0, "transition"),
    (34.0, 44.0, "bridge"),
    (44.0, 54.0, "chorus"),
    (54.0, 60.0, "outro"),
]


def write_fake_adapter(path: Path, metadata: dict[str, str]) -> Path:
    header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": metadata}
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + np.float32(1.0).tobytes())
    return path


ADAPTER_MD = {
    "lora_config": json.dumps({"rank": 16, "alpha": 16, "adapter_type": "dora-rows", "exclude": ["seconds_total"]}),
    "ravi_arm": "B",
    "ravi_seed": "42",
    "ravi_cfg_hash": "abc123def456",
    "ravi_step": "300",
    "ravi_identity": json.dumps({"model": "medium-base", "rank": 16}),
}


class FakeSAM:
    """Stands in for stable_audio_3.StableAudioModel: the attributes Engine reads and a
    generate() that mimics upstream's output shape and truncation (model.py:350-356)."""

    def __init__(self, sr: int = SR, ds: int = DS, sample_size: int | None = None, fill: str = "silence"):
        self.model = SimpleNamespace(
            sample_rate=sr,
            io_channels=64,
            pretransform=SimpleNamespace(downsampling_ratio=ds, io_channels=2),
        )
        self.model_config = {"sample_size": sample_size or 120 * sr}
        self.calls: list[dict] = []
        self.strengths: list[float] = []
        self.fill = fill

    def set_lora_strength(self, strength, lora_index=None):
        self.strengths.append(strength)

    def generate(self, **kw):
        self.calls.append(kw)
        sr = self.model.sample_rate
        n = int(kw["duration"] * sr)  # upstream truncation
        if self.fill == "silence":
            return np.zeros((1, 2, n), dtype=np.float32)
        rng = np.random.default_rng(kw["seed"])
        return (0.1 * rng.standard_normal((1, 2, n))).astype(np.float32)


def make_engine(lora: str | None = None, **kw) -> tuple[si.Engine, FakeSAM]:
    fake = FakeSAM(**kw)
    eng = si.Engine(fake, "medium", lora, to_tensor=lambda a: np.ascontiguousarray(a.T), to_numpy=np.asarray)
    return eng, fake


def run(argv: list[str], engine: si.Engine) -> int:
    return si.main(argv, engine_factory=lambda args: engine)


@pytest.fixture(autouse=True)
def _hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")


@pytest.fixture
def track(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.standard_normal((60 * SR, 2))).astype(np.float32)
    wav = tmp_path / "data" / "t1.wav"
    wav.parent.mkdir(parents=True)
    sf.write(wav, audio, SR, subtype="FLOAT")
    seg = schema.save(schema.from_human([Segment(*s) for s in SONG], "t1", duration=60.0), tmp_path / "data" / "t1.json")
    return wav, seg, audio


# --------------------------------------------------------------------------- masks


def test_section_masks_interior_targets_padded_and_indexed():
    got = mk.section_masks(SONG, 60.0)
    assert [tuple(m) for m in got] == [
        (7.0, 21.0, "verse", 1),
        (19.0, 33.0, "chorus", 2),
        (33.0, 45.0, "bridge", 4),
        (43.0, 55.0, "chorus", 5),
    ]
    start, end, label, index = got[0]  # plain 4-tuple unpacking
    assert (start, end, label, index) == (7.0, 21.0, "verse", 1)


def test_section_masks_excludes_first_and_last_labelled_and_short():
    segs = [
        (0.0, 1.0, "unlabeled"),  # skipped annotator head: the verse is still the first section
        (1.0, 10.0, "verse"),
        (10.0, 13.0, "chorus"),  # 3 s < min_len_s
        (13.0, 25.0, "Chorus "),  # case/whitespace normalised
        (25.0, 35.0, "brdige"),  # typo map
        (35.0, 50.0, "verse"),
        (50.0, 52.0, "unlabeled"),  # trailing: the last verse is still the last section
    ]
    got = mk.section_masks(segs, 52.0, pad_s=0.5)
    assert [(m.index, m.label) for m in got] == [(3, "chorus"), (4, "bridge")]
    assert got[0][:2] == (12.5, 25.5)
    assert mk.section_masks(segs, 52.0, min_len_s=2.0)[0].index == 2


def test_section_masks_clamps_and_accepts_segment_objects():
    segs = [Segment(0, 0.5, "intro"), Segment(0.5, 10, "verse"), Segment(10, 10.8, "outro")]
    (m,) = mk.section_masks(segs, 10.8, pad_s=1.0)
    assert (m.start, m.end) == (0.0, 10.8)
    (m,) = mk.section_masks([dict(start=a, end=b, label=c) for a, b, c in [(0, 2, "intro"), (2, 9, "bridge"),
                                                                           (9, 12, "outro")]], None, pad_s=5)
    assert (m.start, m.end) == (0.0, 14.0)  # no upper clamp without a duration
    assert mk.section_masks(SONG[:2], 20.0) == []
    with pytest.raises(ValueError):
        mk.section_masks(SONG, 60.0, pad_s=-1)


def test_crossfade_ramp_is_symmetric_and_open():
    r = mk.crossfade_ramp(10)
    assert np.allclose(r + r[::-1], 1.0)
    assert 0 < r[0] < 0.05 and 0.95 < r[-1] < 1
    assert np.all(np.diff(r) > 0)
    assert mk.crossfade_ramp(0).size == 0


def test_splice_keeps_outside_bit_identical_and_fades_inside():
    rng = np.random.default_rng(1)
    src = rng.standard_normal((1000, 2)).astype(np.float32)
    fill = rng.standard_normal((400, 2)).astype(np.float32)
    out = mk.splice(src, fill, 300, 700, 50)
    assert np.array_equal(out[:300], src[:300]) and np.array_equal(out[700:], src[700:])
    assert np.array_equal(out[350:650], fill[50:350])  # untouched middle
    w = mk.crossfade_ramp(50)[:, None]
    assert np.allclose(out[300:350], (1 - w) * src[300:350] + w * fill[:50])
    assert np.allclose(out[650:700], w[::-1] * fill[350:] + (1 - w[::-1]) * src[650:700])
    # the input is not modified
    assert not np.shares_memory(out, src)


def test_splice_short_mask_and_continuation():
    src = np.ones((100, 1), dtype=np.float32)
    fill = np.zeros((10, 1), dtype=np.float32)
    out = mk.splice(src, fill, 40, 50, 50)  # xfade capped at 5 samples
    assert np.array_equal(out[:40], src[:40]) and np.array_equal(out[50:], src[50:])
    assert np.all(out[40:50] < 1) and np.all(out[40:50] > 0)
    out = mk.splice(src, np.zeros((30, 1), np.float32), 70, 100, 10)  # mask to the end: fade-in only
    assert np.array_equal(out[:70], src[:70])
    assert np.all(out[80:] == 0)
    with pytest.raises(ValueError):
        mk.splice(src, fill, 40, 60, 5)  # wrong fill length
    with pytest.raises(ValueError):
        mk.splice(src, fill, 95, 105, 5)


def test_anchors():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((1000, 2)).astype(np.float32)
    up = mk.upper_anchor(a, 100, 300)
    assert np.array_equal(up, a[100:300]) and not np.shares_memory(up, a)
    assert np.array_equal(mk.splice(a, up, 100, 300, 0), a)  # identity without a fade
    lo = mk.lower_anchor(a, 200)
    assert np.array_equal(lo, a[400:600])  # middle of the other track
    assert np.array_equal(mk.lower_anchor(a, 200, offset=0), a[:200])
    tiled = mk.lower_anchor(a[:30], 100)
    assert tiled.shape == (100, 2) and np.array_equal(tiled[30:60], a[:30])
    assert mk.lower_anchor(np.arange(10, dtype=np.float32), 4).shape == (4, 1)
    with pytest.raises(ValueError):
        mk.lower_anchor(a, 200, offset=900)


# --------------------------------------------------------------------------- naming / small helpers


def test_prompt_rendering_and_names():
    cap = FIELD_SEP.join(["rvkr", "Warm piano, soft strings", "Jazz"])
    assert si.render_prompt(cap) == "rvkr, Warm piano, soft strings, Jazz"
    assert si.render_prompt("  calm   piano ") == "calm piano"
    stem = si.generate_stem(7, cap)
    assert stem.startswith("7_rvkr-warm-piano-soft-strings-jazz-") and len(stem.split("-")[-1]) == 6
    assert si.generate_stem(7, "a b") != si.generate_stem(7, "a  b")  # raw prompt is hashed
    assert si.slugify("!!!").startswith("prompt-")
    assert len(si.slugify("x" * 200)) == si.SLUG_LEN + 7
    assert si.inpaint_stem("My Song", 3, "chorus", 19.0, 33.04) == "My Song_sec3_chorus_19.0-33.0"
    assert si.inpaint_stem("a/b", 1, "verse", 1, 2) == "a_b_sec1_verse_1.0-2.0"
    assert si.strength_dirname(0.5) == "strength0.5" and si.strength_dirname(1.0) == "strength1"


def test_read_prompts(tmp_path: Path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\nfirst\n\n  \nsecond" + FIELD_SEP + "x\n", encoding="utf-8")
    assert si.read_prompts(["cli"], str(f)) == ["cli", "first", "second" + FIELD_SEP + "x"]
    with pytest.raises(SystemExit):
        si.read_prompts(["a", "a"], None)
    with pytest.raises(SystemExit):
        si.read_prompts(None, None)


def test_mask_seconds_survive_upstream_int_truncation():
    for sr in (8000, 44100, 48000):
        for n in list(range(0, 5000, 7)) + [2072576, 5292031, 44100 * 61 + 1]:
            assert int(si.mask_seconds(n, sr) * sr) == n  # model.py:205-212


def test_effective_latent_region():
    assert si.effective_latent_region(0, 1000, 256, 4096) == (0, 1024)
    assert si.effective_latent_region(256, 512, 256, 4096) == (256, 512)
    assert si.effective_latent_region(257, 4000, 256, 4096) == (512, 4096)
    assert si.effective_latent_region(10, 20, 1, 100) == (10, 20)


def test_choose_context():
    sr, n = 100, 6000  # 60 s
    c0, c1 = si.choose_context(2000, 2600, n, sr, 20, 5, 10_000)
    assert c1 - c0 == 2000 and c0 <= 2000 and 2600 <= c1
    assert abs((2000 - c0) - (c1 - 2600)) <= 1
    # long mask: the window grows to keep min_context on both sides
    c0, c1 = si.choose_context(1000, 4000, n, sr, 20, 5, 10_000)
    assert (c0, c1) == (500, 4500)
    # near the start: shifted, not shrunk
    assert si.choose_context(100, 500, n, sr, 20, 5, 10_000) == (0, 2000)
    # whole track, and the model maximum
    assert si.choose_context(100, 500, n, sr, 0, 5, 10_000) == (0, 6000)
    c0, c1 = si.choose_context(100, 500, n, sr, 0, 5, 3000)
    assert c1 - c0 == 3000 and c0 <= 100
    with pytest.raises(ValueError):
        si.choose_context(100, 4000, n, sr, 20, 5, 3000)
    # latent alignment: the mask starts on a frame of the context
    for m0, m1 in [(2013, 2600), (37, 900), (5000, 5990), (1, 5999)]:
        c0, c1 = si.choose_context(m0, m1, n, sr, 20, 5, 10_000, align=64)
        assert c0 <= m0 and m1 <= c1 <= n and c1 - c0 <= 10_000
        if c0 > 0 or m0 % 64 == 0:
            assert (m0 - c0) % 64 == 0


# --------------------------------------------------------------------------- generate


def test_generate_names_lengths_and_sidecar(tmp_path: Path):
    adapter = write_fake_adapter(tmp_path / "adapter.safetensors", ADAPTER_MD)
    eng, fake = make_engine(lora=str(adapter))
    pf = tmp_path / "captions.txt"
    cap = FIELD_SEP.join(["rvkr", "Warm piano", "Jazz"])
    pf.write_text(cap + "\n", encoding="utf-8")
    out = tmp_path / "gen"
    run(["generate", "--lora", str(adapter), "--prompt", "lo-fi beat", "--prompt-file", str(pf),
         "--duration", "3", "--seed", "1", "2", "--out", str(out)], eng)
    wavs = sorted(p.name for p in out.glob("*.wav"))
    assert wavs == sorted(f"{s}_{si.slugify(p)}.wav" for s in (1, 2) for p in ("lo-fi beat", cap))
    a, sr = sf.read(out / f"1_{si.slugify(cap)}.wav", always_2d=True)
    assert sr == SR and a.shape == (3 * SR, 2) and sf.info(out / f"1_{si.slugify(cap)}.wav").subtype == "FLOAT"
    meta = json.loads((out / f"1_{si.slugify(cap)}.json").read_text())
    assert meta["prompt"] == cap and meta["prompt_rendered"] == "rvkr, Warm piano, Jazz"
    assert (meta["steps"], meta["cfg_scale"], meta["seed"], meta["lora_strength"]) == (8, 1.0, 1, 1.0)
    assert meta["lora"]["ravi_metadata"]["ravi_arm"] == "B"
    assert meta["lora"]["ravi_metadata"]["ravi_identity"] == {"model": "medium-base", "rank": 16}
    assert meta["lora"]["lora_config"]["adapter_type"] == "dora-rows"
    assert meta["lora"]["sha256"] == si.sha256_file(adapter)
    assert meta["hf"]["HF_TOKEN_set"] is True and "hf_dummy" not in json.dumps(meta)
    # upstream call: rendered prompt, model-default sampling, no inpaint args, model's sample_size
    kw = fake.calls[0]
    assert kw["prompt"] in ("lo-fi beat", "rvkr, Warm piano, Jazz") and kw["steps"] == 8 and kw["cfg_scale"] == 1.0
    assert kw["batch_size"] == 1 and kw["sample_size"] == 120 * SR and "inpaint_audio" not in kw
    assert fake.strengths == [1.0]  # set explicitly even at the default


def test_generate_strength_sweep_base_defaults_and_guards(tmp_path: Path):
    adapter = write_fake_adapter(tmp_path / "adapter.safetensors", ADAPTER_MD)
    eng, fake = make_engine(lora=str(adapter))
    out = tmp_path / "gen"
    argv = ["generate", "--model", "medium-base", "--lora", str(adapter), "--lora-strength", "0", "0.5", "1",
            "--prompt", "x", "--duration", "1", "--out", str(out)]
    run(argv, eng)
    assert fake.strengths == [0.0, 0.5, 1.0]
    assert sorted(p.name for p in out.iterdir()) == ["strength0", "strength0.5", "strength1"]
    meta = json.loads(next((out / "strength0.5").glob("*.json")).read_text())
    assert (meta["steps"], meta["cfg_scale"], meta["lora_strength"]) == (50, 7.0, 0.5)
    with pytest.raises(SystemExit, match="overwrite"):
        run(argv, eng)
    run(argv + ["--overwrite", "--steps", "12", "--cfg", "3"], eng)
    assert fake.calls[-1]["steps"] == 12 and fake.calls[-1]["cfg_scale"] == 3.0
    eng2, _ = make_engine()
    with pytest.raises(SystemExit, match="needs --lora"):
        run(["generate", "--lora-strength", "0.5", "--prompt", "x", "--out", str(out)], eng2)
    with pytest.raises(SystemExit):
        run(["generate", "--prompt", "x", "--duration", "500", "--out", str(tmp_path / "g2")], eng2)


def test_generate_without_lora(tmp_path: Path):
    eng, fake = make_engine()
    run(["generate", "--prompt", "x", "--duration", "1", "--out", str(tmp_path / "g")], eng)
    meta = json.loads(next((tmp_path / "g").glob("*.json")).read_text())
    assert meta["lora"] is None and meta["lora_strength"] is None and fake.strengths == []


# --------------------------------------------------------------------------- inpaint


def _outside_equal(a: np.ndarray, b: np.ndarray, m0: int, m1: int) -> bool:
    return np.array_equal(a[:m0], b[:m0]) and np.array_equal(a[m1:], b[m1:])


def test_inpaint_all_interior_files_invariant_and_crossfade(tmp_path: Path, track):
    wav, seg, audio = track
    eng, fake = make_engine()  # silence inside the mask makes the fades readable
    out = tmp_path / "inp"
    run(["inpaint", "--audio", str(wav), "--segments", str(seg), "--all-interior", "--prompt",
         "rvkr" + FIELD_SEP + "calm", "--seed", "3", "--context-s", "30", "--out", str(out)], eng)
    stems = ["t1_sec1_verse_7.0-21.0", "t1_sec2_chorus_19.0-33.0", "t1_sec4_bridge_33.0-45.0",
             "t1_sec5_chorus_43.0-55.0"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{s}{x}" for s in stems for x in ("_raw.wav", "_spliced.wav", "_orig.wav", ".json"))
    assert len(fake.calls) == 4
    for stem, kw in zip(stems, fake.calls):
        meta = json.loads((out / f"{stem}.json").read_text())
        c0, c1 = meta["context"]["start_sample"], meta["context"]["end_sample"]
        m0r, m1r = meta["mask_in_context"]["start_sample"], meta["mask_in_context"]["end_sample"]
        ctx = audio[c0:c1]
        # upstream call: (sample_rate, (channels, samples)) tuple, mask seconds within the context
        sr_passed, wave = kw["inpaint_audio"]
        assert sr_passed == SR and wave.shape == (2, c1 - c0) and np.array_equal(wave.T, ctx)
        assert int(kw["inpaint_mask_start_seconds"] * SR) == m0r and int(kw["inpaint_mask_end_seconds"] * SR) == m1r
        assert kw["duration"] == pytest.approx((c1 - c0) / SR) and kw["seed"] == 3
        assert kw["prompt"] == "rvkr, calm"
        assert m0r % DS == 0  # mask starts on a latent frame
        assert meta["effective_latent_mask_in_context"]["start_sample"] == m0r
        assert c1 - c0 == 30 * SR
        # files
        raw, _ = sf.read(out / f"{stem}_raw.wav", dtype="float32", always_2d=True)
        spl, _ = sf.read(out / f"{stem}_spliced.wav", dtype="float32", always_2d=True)
        org, _ = sf.read(out / f"{stem}_orig.wav", dtype="float32", always_2d=True)
        assert raw.shape == spl.shape == org.shape == ctx.shape
        assert np.array_equal(org, ctx)  # upper anchor = identity
        assert _outside_equal(spl, ctx, m0r, m1r)  # bit-identical outside the mask
        xf = meta["xfade"]["samples"]
        assert xf == 400  # 50 ms at 8 kHz
        w = mk.crossfade_ramp(xf)[:, None]
        assert np.allclose(spl[m0r:m0r + xf], (1 - w) * ctx[m0r:m0r + xf], atol=1e-7)
        assert np.allclose(spl[m1r - xf:m1r], (1 - w[::-1]) * ctx[m1r - xf:m1r], atol=1e-7)
        assert np.all(spl[m0r + xf:m1r - xf] == 0)  # the (silent) model output inside
        assert meta["spliced_outside_mask_bit_identical"] is True
        assert meta["mask"]["start_s"] == float(stem.split("_")[-1].split("-")[0])
        assert meta["section"]["label"] in stem and meta["lora"] is None
        # silence vs. noise: the round-trip error outside the mask is the source's own level
        assert meta["raw_vs_source_outside_mask"]["rms"] == pytest.approx(0.3, rel=0.05)


def test_inpaint_noise_model_section_index_and_strengths(tmp_path: Path, track):
    wav, seg, audio = track
    adapter = write_fake_adapter(tmp_path / "adapter.safetensors", ADAPTER_MD)
    eng, fake = make_engine(lora=str(adapter), fill="noise")
    out = tmp_path / "inp"
    run(["inpaint", "--lora", str(adapter), "--lora-strength", "0", "1", "--audio", str(wav), "--segments",
         str(seg), "--section-index", "4", "--prompt", "p", "--xfade-ms", "10", "--out", str(out)], eng)
    assert fake.strengths == [0.0, 1.0]
    for sub in ("strength0", "strength1"):
        stem = "t1_sec4_bridge_33.0-45.0"
        meta = json.loads((out / sub / f"{stem}.json").read_text())
        c0, c1 = meta["context"]["start_sample"], meta["context"]["end_sample"]
        m0r, m1r = meta["mask_in_context"]["start_sample"], meta["mask_in_context"]["end_sample"]
        raw, _ = sf.read(out / sub / f"{stem}_raw.wav", dtype="float32", always_2d=True)
        spl, _ = sf.read(out / sub / f"{stem}_spliced.wav", dtype="float32", always_2d=True)
        xf = meta["xfade"]["samples"]
        assert xf == 80
        assert _outside_equal(spl, audio[c0:c1], m0r, m1r)
        assert np.array_equal(spl[m0r + xf:m1r - xf], raw[m0r + xf:m1r - xf])
        assert meta["section"] == {"index": 4, "label": "bridge", "start_s": 34.0, "end_s": 44.0,
                                   "label_raw": "bridge"}
        assert meta["mask"]["start_s"] == 33.0 and meta["mask"]["end_s"] == 45.0
        # default 47 s context (the training window), capped by the 60 s track
        assert c1 - c0 == 47 * SR and meta["lora"]["ravi_metadata"]["ravi_step"] == "300"
    # raw output shorter than the context by upstream's int() truncation is padded, never misaligned
    assert meta["raw_output_samples"] in (c1 - c0, c1 - c0 - 1)


def test_inpaint_rejects_ineligible_section_and_resamples(tmp_path: Path, track):
    wav, seg, audio = track
    eng, _ = make_engine()
    with pytest.raises(SystemExit, match="eligible"):
        run(["inpaint", "--audio", str(wav), "--segments", str(seg), "--section-index", "0", "--prompt", "p",
             "--out", str(tmp_path / "x")], eng)
    # 16 kHz mono input -> model rate (8 kHz) stereo
    mono = tmp_path / "data" / "t1_16k.wav"
    sf.write(mono, audio[:, :1].repeat(2, axis=0), 2 * SR, subtype="FLOAT")
    eng, fake = make_engine()
    out = tmp_path / "rs"
    run(["inpaint", "--audio", str(mono), "--segments", str(seg), "--section-index", "1", "--prompt", "p",
         "--track-id", "t1", "--blank-mask-input", "--out", str(out)], eng)
    meta = json.loads((out / "t1_sec1_verse_7.0-21.0.json").read_text())
    assert meta["audio"]["resampled"] is True and meta["audio"]["input_channels"] == 1
    assert meta["audio"]["input_sr"] == 2 * SR and meta["audio"]["channels"] == 2
    _, wave = fake.calls[0]["inpaint_audio"]
    m0r, m1r = meta["mask_in_context"]["start_sample"], meta["mask_in_context"]["end_sample"]
    assert wave.shape[0] == 2 and np.all(wave[:, m0r:m1r] == 0) and np.any(wave[:, :m0r] != 0)
    assert np.array_equal(wave[0], wave[1])
    assert meta["blank_mask_input"] is True


# --------------------------------------------------------------------------- continue


def test_continue_mask_to_end_and_no_future_input(tmp_path: Path, track):
    wav, _, audio = track
    eng, fake = make_engine()
    out = tmp_path / "cont"
    run(["continue", "--audio", str(wav), "--start", "40", "--pre-s", "20", "--gen-s", "10", "--prompt", "p",
         "--out", str(out)], eng)
    stem = "t1_cont_40.0-50.0"
    meta = json.loads((out / f"{stem}.json").read_text())
    c0, c1 = meta["context"]["start_sample"], meta["context"]["end_sample"]
    m0r, m1r = meta["mask_in_context"]["start_sample"], meta["mask_in_context"]["end_sample"]
    assert c1 == 50 * SR and m1r == c1 - c0 and (40 * SR - c0) % DS == 0 and 40 * SR - c0 <= 20 * SR
    _, wave = fake.calls[0]["inpaint_audio"]
    assert np.all(wave[:, m0r:] == 0) and np.array_equal(wave[:, :m0r].T, audio[c0:40 * SR])
    spl, _ = sf.read(out / f"{stem}_spliced.wav", dtype="float32", always_2d=True)
    assert np.array_equal(spl[:m0r], audio[c0:40 * SR])
    assert np.all(spl[m0r + meta["xfade"]["samples"]:] == 0)
    assert (out / f"{stem}_orig.wav").is_file()
    # past the end of the track: no _orig.wav
    run(["continue", "--audio", str(wav), "--start", "55", "--gen-s", "10", "--prompt", "p", "--out",
         str(out)], eng)
    assert not (out / "t1_cont_55.0-65.0_orig.wav").exists()
    assert (out / "t1_cont_55.0-65.0_spliced.wav").is_file()
