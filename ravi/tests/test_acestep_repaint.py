"""Tests for ravi.eval.acestep_repaint with a fake engine that reproduces upstream's waveform
splice (no torch, ACE-Step or GPU). The GenerationParams contract test reads
``acestep/inference.py`` as text from ``ACESTEP_SRC`` and is skipped without one."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ravi.eval import acestep_repaint as r

SR = r.SR


class FakeEngine:
    """Noise inside the mask, upstream's linear ramps outside it, source elsewhere.
    ``leak`` adds a change outside the touched span (a broken splice); ``lora_gain`` makes
    the LoRA scale change the repainted content."""

    def __init__(self, cf_s: float = 0.025, leak: bool = False, short: int = 0,
                 unseeded: bool = False, scale0_leaks: bool = False) -> None:
        self.cf_s, self.leak, self.short = cf_s, leak, short
        self.unseeded, self.scale0_leaks = unseeded, scale0_leaks
        self.scale: float | None = None
        self.calls: list[dict] = []

    def wav_crossfade_s(self, mode, strength):
        return self.cf_s

    def set_lora(self, scale):
        self.scale = scale

    def repaint(self, *, src_path, start, end, caption, lyrics, seed, steps, shift, guidance, mode, strength):
        self.calls.append(dict(start=start, end=end, caption=caption, lyrics=lyrics, seed=seed,
                               steps=steps, shift=shift, guidance=guidance, scale=self.scale))
        src, sr = sf.read(src_path, dtype="float32", always_2d=True)
        assert sr == SR and src.shape[1] == 2
        n = len(src) - self.short
        key = seed + int(1000 * (self.scale or 0)) + (len(self.calls) if self.unseeded else 0)
        key += 7 if self.scale0_leaks and self.scale == 0.0 else 0
        rng = np.random.default_rng(key)
        pred = (src[:n] + 0.01 * rng.standard_normal(src[:n].shape)).astype(np.float32)  # VAE error everywhere
        s0, s1, cf = int(start * SR), int(end * SR), int(self.cf_s * SR)
        pred[s0:s1] = 0.3 * rng.standard_normal((s1 - s0, 2))
        m = np.zeros(n, dtype=np.float32)
        m[s0:s1] = 1.0
        m[s0 - cf:s0] = np.linspace(0.0, 1.0, cf + 2)[1:-1]
        m[s1:s1 + cf] = np.linspace(1.0, 0.0, cf + 2)[1:-1]
        out = m[:, None] * pred + (1.0 - m[:, None]) * src[:n]
        if self.leak:
            out[: SR // 10] += 1e-3
        return out.astype(np.float32)


def write_track(tmp: Path, seconds: float = 40.0, sr: int = 44100) -> tuple[Path, Path]:
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.2 * np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1)
    wav = tmp / "My Track.mp3 X.wav"
    sf.write(wav, audio, sr, subtype="PCM_24")
    segs = [{"start": 0.0, "end": 8.0, "label": "intro"},
            {"start": 8.0, "end": 20.0, "label": "Verse"},
            {"start": 20.0, "end": 32.0, "label": "chorus"},
            {"start": 32.0, "end": seconds, "label": "outro"}]
    doc = {"track_id": "My Track.mp3 X", "tool": "human", "duration": seconds, "segments": segs}
    js = tmp / "My Track.mp3 X.json"
    js.write_text(json.dumps(doc))
    return wav, js


def test_touched_span_matches_upstream_rounding():
    assert r.touched_span(9.0, 21.0, 0.025, 48000 * 40) == (9 * SR - 1200, 21 * SR + 1200)
    assert r.touched_span(0.01, 5.0, 0.025, 10 * SR) == (0, 5 * SR + 1200)
    assert r.touched_span(9.0, 50.0, 0.025, 10 * SR) == (9 * SR - 1200, 10 * SR)


def test_segments_lyrics():
    segs = [{"label": "Intro"}, {"label": "verse "}, {"label": ""}, {"label": "chorus"}]
    assert r.segments_lyrics(segs).splitlines()[:2] == ["[intro]", "[verse]"]
    assert "[chorus]" in r.segments_lyrics(segs)
    assert r.segments_lyrics([]) == "[Instrumental]"


def test_to_stereo_48k_resamples_and_duplicates_mono():
    x = r.to_stereo_48k(np.zeros(44100), 44100)
    assert x.shape == (48000, 2) and x.dtype == np.float32


def test_check_passes_with_upstream_splice(tmp_path):
    wav, js = write_track(tmp_path)
    eng = FakeEngine(short=7)  # decoder output a few samples short of the source
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    rc = r.main(["--audio", str(wav), "--segments", str(js), "--caption", "rvkr, calm, Rock",
                 "--lyrics-from-segments", "--adapter", str(adapter), "--lora-strength", "0", "0.5", "1",
                 "--seed", "3", "--out", str(tmp_path / "out")], engine=eng)
    assert rc == 0
    meta = json.loads((tmp_path / "out" / "repaint_check.json").read_text())
    assert meta["outside_ok"] is True
    assert [x["condition"] for x in meta["rows"]] == ["nolora", "nolora_repeat", "strength0", "strength0.5",
                                                      "strength1"]
    assert [c["scale"] for c in eng.calls] == [None, None, 0.0, 0.5, 1.0]
    assert (meta["ok"], meta["repeat_ok"], meta["scale0_ok"]) == (True, True, True)
    # first eligible interior section (verse, index 1), [8, 20] padded by 1 s on both sides
    assert meta["section"]["index"] == 1 and meta["section"]["mask_s"] == [7.0, 21.0]
    assert eng.calls[0]["lyrics"] == "[intro]\n[verse]\n[chorus]\n[outro]"
    assert (eng.calls[0]["steps"], eng.calls[0]["shift"], eng.calls[0]["guidance"]) == (8, 3.0, 1.0)
    assert all(x["outside_max_abs"] <= r.OUTSIDE_TOL for x in meta["rows"])
    assert all(x["inside_rms_diff"] > 0.1 for x in meta["rows"])
    assert [x["inside_max_abs_vs_nolora"] == 0.0 for x in meta["rows"]] == [True, True, True, False, False]
    assert meta["rows"][4]["inside_rel_rms_vs_nolora"] > 0.5
    for name in ("source_48k", "nolora", "strength0.5"):
        info = sf.info(str(tmp_path / "out" / f"{name}.wav"))
        assert info.samplerate == SR and info.channels == 2


def test_check_fails_when_outside_changes(tmp_path):
    wav, js = write_track(tmp_path)
    rc = r.main(["--audio", str(wav), "--segments", str(js), "--caption", "rvkr, x",
                 "--out", str(tmp_path / "out")], engine=FakeEngine(leak=True))
    assert rc == 1
    meta = json.loads((tmp_path / "out" / "repaint_check.json").read_text())
    assert meta["outside_ok"] is False and len(meta["rows"]) == 2
    assert meta["lyrics"] == "[Instrumental]"


def test_check_fails_when_crossfade_is_wider_than_assumed(tmp_path):
    wav, js = write_track(tmp_path)
    eng = FakeEngine(cf_s=0.05)
    eng.wav_crossfade_s = lambda mode, strength: 0.025
    rc = r.main(["--audio", str(wav), "--segments", str(js), "--caption", "rvkr, x",
                 "--out", str(tmp_path / "out")], engine=eng)
    assert rc == 1


def test_section_index_must_be_interior(tmp_path):
    wav, js = write_track(tmp_path)
    with pytest.raises(SystemExit, match="eligible"):
        r.main(["--audio", str(wav), "--segments", str(js), "--section-index", "0", "--caption", "rvkr, x",
                "--out", str(tmp_path / "out")], engine=FakeEngine())


def test_upstream_generation_params_fields():
    root = os.environ.get("ACESTEP_SRC")
    if not root:
        pytest.skip("ACESTEP_SRC not set (point it at an ACE-Step-1.5 checkout)")
    src = (Path(root) / "acestep" / "inference.py").read_text()
    body = src.split("class GenerationParams", 1)[1].split("\nclass ", 1)[0]
    fields = set(re.findall(r"^    (\w+): ", body, re.M))
    used = {"task_type", "instruction", "src_audio", "repainting_start", "repainting_end", "chunk_mask_mode",
            "repaint_mode", "repaint_strength", "caption", "lyrics", "instrumental", "inference_steps",
            "shift", "guidance_scale", "seed", "thinking", "use_cot_metas", "use_cot_caption",
            "use_cot_language", "enable_normalization"}
    assert used <= fields, used - fields
    consts = (Path(root) / "acestep" / "constants.py").read_text()
    assert '"repaint": "Repaint the mask area based on the given conditions:"' in consts


def test_check_fails_when_same_seed_is_not_reproducible(tmp_path):
    wav, js = write_track(tmp_path)
    rc = r.main(["--audio", str(wav), "--segments", str(js), "--caption", "rvkr, x",
                 "--out", str(tmp_path / "out")], engine=FakeEngine(unseeded=True))
    meta = json.loads((tmp_path / "out" / "repaint_check.json").read_text())
    assert rc == 1 and meta["outside_ok"] is True and meta["repeat_ok"] is False and meta["scale0_ok"] is None


def test_check_fails_when_scale0_differs_from_nolora(tmp_path):
    wav, js = write_track(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    rc = r.main(["--audio", str(wav), "--segments", str(js), "--caption", "rvkr, x", "--adapter", str(adapter),
                 "--lora-strength", "0", "1", "--out", str(tmp_path / "out")], engine=FakeEngine(scale0_leaks=True))
    meta = json.loads((tmp_path / "out" / "repaint_check.json").read_text())
    assert rc == 1 and meta["repeat_ok"] is True and meta["scale0_ok"] is False
