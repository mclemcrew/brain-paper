"""ravi.captions.autocaption: prompt, post-processing, audio prep, JSONL resume.

The model call is mocked (FakeBackend): no torch/transformers needed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ravi.captions import autocaption as ac

IDS = ["0001_copilot.mp3 Agent X NS", "0002_a", "0003_b", "0004_c"]


# ------------------------------------------------------------------ prompt


def test_prompt_exact_text():
    assert ac.build_prompt(15) == (
        "Describe this piece of instrumental music in one sentence of at most 15 words, "
        "covering its genre, mood, main instruments and how fast it feels."
    )
    assert ac.build_prompt(15, "alt") == (
        "What does this instrumental track sound like? Answer in one sentence of at most 15 words, "
        "naming its genre, mood, main instruments and how fast it feels."
    )
    assert "at most 8 words" in ac.build_prompt(8)
    for v in ac.PROMPTS:  # nothing negated left to parrot
        assert not any(w in ac.build_prompt(15, v).lower() for w in ("no lyrics", "bpm", " not ", "no "))


def test_prompt_sha_changes_with_wording_and_cap(monkeypatch):
    a = ac.prompt_sha(15)
    assert a == ac.prompt_sha(15) and a != ac.prompt_sha(12)
    monkeypatch.setitem(ac.PROMPTS, "alt", "Describe it in {n} words.")
    assert ac.prompt_sha(15) != a
    with pytest.raises(ValueError):
        ac.build_prompt(0)


# ------------------------------------------------------------------ post-processing


@pytest.mark.parametrize(
    "raw, want",
    [
        ('"An upbeat electronic track with punchy drums."', "an upbeat electronic track with punchy drums"),
        ("“Warm jazz trio, brushed drums.”", "warm jazz trio, brushed drums"),
        ("Caption: Dreamy ambient pads, slow and spacious.", "dreamy ambient pads, slow and spacious"),
        ("\n\nWarm jazz trio, brushed drums.\nSecond line here", "warm jazz trio, brushed drums"),
        ("EDM banger, energetic, big synth leads.", "EDM banger, energetic, big synth leads"),
        ("Lo-fi beat", "lo-fi beat"),
    ],
)
def test_postprocess_single_line_quotes_period_case(raw, want):
    assert ac.postprocess(raw, 15).caption == want


@pytest.mark.parametrize(
    "raw, want, stripped",
    [
        ("Upbeat house groove at 124 BPM, bright synths", "upbeat house groove, bright synths", ["at 124 BPM"]),
        ("Mellow lo-fi hip hop at around 85 bpm, dusty piano", "mellow lo-fi hip hop, dusty piano",
         ["at around 85 bpm"]),
        ("EDM banger (128bpm), energetic", "EDM banger, energetic", ["128bpm"]),
        ("Driving techno, 128-132 BPM, dark", "driving techno, dark", ["128-132 BPM"]),
        ("Funk at 110 beats per minute with slap bass", "funk with slap bass", ["at 110 beats per minute"]),
        ("A 90-bpm boom bap beat", "boom bap beat", ["A 90-bpm"]),
    ],
)
def test_postprocess_strips_bpm(raw, want, stripped):
    r = ac.postprocess(raw, 15)
    assert r.caption == want
    assert r.bpm_stripped == stripped


def test_postprocess_keeps_numbers_without_bpm():
    r = ac.postprocess("80s synthwave with retro drums", 15)
    assert r.caption == "80s synthwave with retro drums" and r.bpm_stripped == []


def test_word_cap_cuts_at_last_comma_boundary():
    raw = ("Driving rock instrumental featuring distorted electric guitars, pounding drums, "
           "bass and an anthemic uplifting energetic mood throughout.")
    r = ac.postprocess(raw, 15)
    # 17 words; the last comma that keeps >= 8 words is after "drums," (9 words); a comma beats
    # the later "bass and" word boundary
    assert r.caption == "driving rock instrumental featuring distorted electric guitars, pounding drums"
    assert r.truncated and r.n_words == 9 and r.n_words_before_cap == 17


def test_word_cap_cuts_before_conjunction():
    raw = "Calm ambient piano piece with soft strings and gentle pads evoking a dreamy nostalgic late night mood"
    r = ac.postprocess(raw, 15)
    assert r.caption == "calm ambient piano piece with soft strings and gentle pads"
    assert r.n_words == 10


def test_word_cap_hard_cut_drops_dangling_words():
    # No boundary in the upper half of the budget -> hard cut; trailing "the" is dropped.
    r = ac.postprocess("w1 w2 w3 w4 w5 w6 w7 w8 w9 the w11 w12", 10)
    assert r.caption == "w1 w2 w3 w4 w5 w6 w7 w8 w9"
    assert r.truncated and r.n_words == 9


def test_word_cap_ignores_boundaries_below_half_budget():
    # The only comma is after word 2 (< ceil(10/2)), so the cut is hard at 10.
    r = ac.postprocess("Ambient drone, w3 w4 w5 w6 w7 w8 w9 w10 w11 w12", 10)
    assert r.n_words == 10 and r.caption.endswith("w10")


def test_word_cap_noop_when_short_and_strips_trailing_comma():
    r = ac.postprocess("Soft piano, ", 15)
    assert r.caption == "soft piano" and not r.truncated


@pytest.mark.parametrize("max_words", [3, 5, 8, 15])
def test_word_cap_always_respected(max_words):
    raw = "Bright, bouncy, funky, retro disco groove with slap bass, strings, horns and claps all night long"
    r = ac.postprocess(raw, max_words)
    assert 1 <= r.n_words <= max_words
    assert not r.caption.endswith((",", ";", ":", "."))


def test_postprocess_empty():
    assert ac.postprocess("  \n\"\" ", 15).caption == ""


# ------------------------------------------------------------------ audio


def test_resample_mono_44k_to_16k():
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1)
    y = ac.resample_mono(x, sr, 16000)
    assert y.dtype == np.float32 and y.ndim == 1 and y.shape[0] == 32000
    assert abs(np.sqrt(np.mean(y[1000:-1000] ** 2)) - np.sqrt(0.5)) < 0.01


def test_load_track_audio_crop_and_sr_check(tmp_path: Path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros((44100 * 3, 2), dtype=np.float32), 44100, subtype="FLOAT")
    y, src, fed = ac.load_track_audio(p, 44100, 16000, None)
    assert src == pytest.approx(3.0) and fed == pytest.approx(3.0) and y.shape == (48000,)
    y, src, fed = ac.load_track_audio(p, 44100, 16000, 1.5)
    assert src == pytest.approx(3.0) and fed == pytest.approx(1.5)
    with pytest.raises(ValueError, match="sample rate"):
        ac.load_track_audio(p, 48000, 16000, None)


def test_strip_prompt_tokens():
    inp = np.array([[0, 5, 6], [7, 8, 9]])
    full = np.array([[0, 5, 6, 1, 2], [7, 8, 9, 3, 4]])
    assert ac.strip_prompt_tokens(full, inp).tolist() == [[1, 2], [3, 4]]
    new_only = np.array([[1, 2], [3, 4]])
    assert ac.strip_prompt_tokens(new_only, inp).tolist() == [[1, 2], [3, 4]]


# ------------------------------------------------------------------ driver with a fake model


class FakeBackend:
    name = "midasheng"
    hf_id = ac.MIDASHENG_ID
    revision = ac.MIDASHENG_REVISION
    sample_rate = 16000
    max_audio_s = None

    def __init__(self, fail_batches: bool = False, fail_len: int | None = None):
        self.calls: list[tuple[int, str]] = []
        self.lengths: list[int] = []
        self.fail_batches = fail_batches
        self.fail_len = fail_len

    def generate(self, audios, prompt):
        self.calls.append((len(audios), prompt))
        if self.fail_batches and len(audios) > 1:
            raise RuntimeError("CUDA out of memory (fake)")
        out = []
        for a in audios:
            assert a.dtype == np.float32 and a.ndim == 1
            if self.fail_len is not None and a.shape[0] == self.fail_len:
                raise RuntimeError("bad clip (fake)")
            self.lengths.append(a.shape[0])
            out.append(f'"Upbeat funk at 120 BPM with slap bass, {a.shape[0] // 16000} s."')
        return out


@pytest.fixture
def corpus(tmp_path: Path):
    pd.DataFrame({"track_id": IDS}).to_parquet(tmp_path / "tracks.parquet")
    d = tmp_path / "prepared" / "44100"
    d.mkdir(parents=True)
    for i, t in enumerate(IDS):
        sf.write(d / f"{t}.wav", np.zeros((44100 * (i + 1), 2), dtype=np.float32), 44100, subtype="FLOAT")
    return tmp_path


def _cfg(root: Path, **kw) -> ac.RunConfig:
    base = dict(manifest=root / "tracks.parquet", prepared_root=root / "prepared", sr=44100,
                out=root / "captions" / "midasheng.jsonl")
    base.update(kw)
    return ac.RunConfig(**base)


def _records(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_run_writes_schema_and_prompt(corpus: Path):
    be = FakeBackend()
    s = ac.run(_cfg(corpus, limit=2), backend=be)
    assert s == {"selected": 2, "skipped": 0, "ok": 2, "failed": 0}
    recs = _records(corpus / "captions" / "midasheng.jsonl")
    assert [r["track_id"] for r in recs] == IDS[:2]
    assert set(recs[0]) == {"track_id", "caption", "raw", "model", "n_words", "audio_seconds", "elapsed_s",
                            "audio_source", "audio_dir", "split", "prompt_variant", "echo_retry",
                            "echo_reason", "raw_primary"}
    assert recs[0]["prompt_variant"] == "primary" and recs[0]["echo_retry"] is False
    assert recs[0]["audio_source"] == "prepared" and recs[0]["split"] is None
    assert recs[0]["audio_dir"] == str(corpus / "prepared" / "44100")
    assert recs[0]["caption"] == "upbeat funk with slap bass, 1 s"
    assert recs[0]["raw"].startswith('"Upbeat funk at 120 BPM')
    assert recs[1]["audio_seconds"] == pytest.approx(2.0) and recs[0]["model"] == ac.MIDASHENG_ID
    assert recs[0]["n_words"] == len(recs[0]["caption"].split())
    assert all(p == ac.build_prompt(15) for _, p in be.calls)
    assert be.lengths == [16000, 32000]  # 44.1 kHz stereo -> 16 kHz mono
    meta = json.loads((corpus / "captions" / "midasheng.jsonl.meta.json").read_text())
    assert meta["revision"] == ac.MIDASHENG_REVISION and meta["prompt"] == ac.build_prompt(15)
    assert meta["audio_source"] == "prepared" and meta["audio_dir"] is None
    assert meta["audio_seconds"] == {IDS[0]: 1.0, IDS[1]: 2.0}


def test_resume_skips_done_and_limit_is_stable(corpus: Path):
    out = corpus / "captions" / "midasheng.jsonl"
    ac.run(_cfg(corpus, limit=2), backend=FakeBackend())
    with pytest.raises(FileExistsError):
        ac.run(_cfg(corpus), backend=FakeBackend())
    be = FakeBackend()
    assert ac.run(_cfg(corpus, limit=2, resume=True), backend=be)["ok"] == 0
    assert be.calls == []
    be = FakeBackend()
    s = ac.run(_cfg(corpus, resume=True), backend=be)
    assert s == {"selected": 4, "skipped": 2, "ok": 2, "failed": 0}
    assert [r["track_id"] for r in _records(out)] == IDS
    assert len(be.calls) == 2


def test_resume_repairs_torn_tail(corpus: Path):
    out = corpus / "captions" / "midasheng.jsonl"
    ac.run(_cfg(corpus, limit=2), backend=FakeBackend())
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"track_id": "0003_b", "capt')  # crash mid-write
    ac.run(_cfg(corpus, resume=True), backend=FakeBackend())
    recs = _records(out)
    assert [r["track_id"] for r in recs] == IDS


def test_resume_rejects_corrupt_middle_line(corpus: Path):
    out = corpus / "captions" / "midasheng.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text('{"track_id": "0002_a", "caption": "x", "model": "%s"}\nnot json\n' % ac.MIDASHENG_ID)
    with pytest.raises(ValueError, match="not valid JSON"):
        ac.run(_cfg(corpus, resume=True), backend=FakeBackend())


def test_resume_rejects_changed_settings(corpus: Path):
    ac.run(_cfg(corpus, limit=1), backend=FakeBackend())
    with pytest.raises(ValueError, match="max_words"):
        ac.run(_cfg(corpus, resume=True, max_words=10), backend=FakeBackend())


def test_resume_rejects_other_model_records(corpus: Path):
    out = corpus / "captions" / "midasheng.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"track_id": IDS[0], "caption": "x", "model": ac.QWEN_ID}) + "\n")
    with pytest.raises(ValueError, match="Qwen"):
        ac.run(_cfg(corpus, resume=True), backend=FakeBackend())


def test_failures_logged_and_retried_on_resume(corpus: Path):
    (corpus / "prepared" / "44100" / f"{IDS[1]}.wav").unlink()
    be = FakeBackend(fail_len=48000)  # IDS[2] is 3 s -> 48000 samples at 16 kHz
    s = ac.run(_cfg(corpus), backend=be)
    assert s["ok"] == 2 and s["failed"] == 2
    fails = _records(corpus / "captions" / "_failures.jsonl")
    assert [(f["track_id"], f["stage"]) for f in fails] == [(IDS[1], "load"), (IDS[2], "generate")]
    assert all(f["model"] == ac.MIDASHENG_ID and f["error"] for f in fails)
    # Fix the inputs; --resume retries exactly the failed ids.
    sf.write(corpus / "prepared" / "44100" / f"{IDS[1]}.wav", np.zeros((44100, 2), np.float32), 44100)
    be = FakeBackend()
    s = ac.run(_cfg(corpus, resume=True), backend=be)
    assert s == {"selected": 4, "skipped": 2, "ok": 2, "failed": 0}
    assert sorted(r["track_id"] for r in _records(corpus / "captions" / "midasheng.jsonl")) == sorted(IDS)


def test_batch_failure_falls_back_to_single_items(corpus: Path):
    be = FakeBackend(fail_batches=True)
    s = ac.run(_cfg(corpus, batch_size=3), backend=be)
    assert s["ok"] == 4 and s["failed"] == 0
    assert [n for n, _ in be.calls] == [3, 1, 1, 1, 1]


def test_empty_caption_is_a_failure(corpus: Path):
    class Empty(FakeBackend):
        def generate(self, audios, prompt):
            self.calls.append((len(audios), prompt))
            return ['""' for _ in audios]

    be = Empty()
    s = ac.run(_cfg(corpus, limit=1), backend=be)
    assert s["failed"] == 1 and not (corpus / "captions" / "midasheng.jsonl").exists()
    assert [p for _, p in be.calls] == [ac.build_prompt(15), ac.build_prompt(15, "alt")]
    f = _records(corpus / "captions" / "_failures.jsonl")[0]
    assert f["stage"] == "echo" and f["raw"] == '""' and "empty" in f["error"]


def test_track_id_selection_and_unknown(corpus: Path):
    be = FakeBackend()
    ac.run(_cfg(corpus, track_ids=[IDS[3], IDS[0]]), backend=be)
    assert [r["track_id"] for r in _records(corpus / "captions" / "midasheng.jsonl")] == [IDS[3], IDS[0]]
    with pytest.raises(KeyError):
        ac.select_tracks(IDS, ["nope"], None)


def test_max_audio_seconds_crop(corpus: Path):
    be = FakeBackend()
    ac.run(_cfg(corpus, max_audio_seconds=2.0), backend=be)
    assert be.lengths == [16000, 32000, 32000, 32000]
    recs = _records(corpus / "captions" / "midasheng.jsonl")
    assert [r["audio_seconds"] for r in recs] == [1.0, 2.0, 2.0, 2.0]


def test_cli_parses_and_requires_paths(corpus: Path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ac, "run", lambda cfg: seen.setdefault("cfg", cfg) and {"failed": 0})
    rc = ac.main(["--manifest", str(corpus / "tracks.parquet"), "--prepared-root", str(corpus / "prepared"),
                  "--sr", "44100", "--out", str(corpus / "c.jsonl"), "--model", "qwen-omni", "--limit", "3",
                  "--track-id", IDS[0], IDS[1], "--max-words", "12", "--batch-size", "2", "--resume"])
    cfg = seen["cfg"]
    assert rc == 0 and cfg.model == "qwen-omni" and cfg.limit == 3 and cfg.track_ids == IDS[:2]
    assert cfg.max_words == 12 and cfg.batch_size == 2 and cfg.resume
    with pytest.raises(SystemExit):
        ac.main(["--sr", "44100"])


def test_qwen_spec_caps_audio_at_300s():
    assert ac.SPECS["qwen-omni"].max_audio_s == 300.0
    assert ac.SPECS["midasheng"].max_audio_s is None
    assert ac.SPECS["midasheng"].sample_rate == ac.SPECS["qwen-omni"].sample_rate == 16000


# ------------------------------------------------------------------ --audio-dir / --split

SPLITS = {"train": [IDS[0], IDS[2]], "val": [IDS[1]], "test": [IDS[3]], "seed": 1, "criteria": {}}


@pytest.fixture
def arms(corpus: Path):
    """splits.json + make_arms-style windowed dirs data/arms/A/{split}/ (0.5 s WAVs + sidecars)."""
    (corpus / "splits.json").write_text(json.dumps(SPLITS))
    for split in ("train", "val"):
        d = corpus / "arms" / "A" / split
        d.mkdir(parents=True)
        for t in SPLITS[split]:
            sf.write(d / f"{t}.wav", np.zeros((22050, 2), np.float32), 44100, subtype="FLOAT")
            (d / f"{t}.wav.json").write_text("{}")
            (d / f"{t}.window.json").write_text('{"start": 3.0}')
            (d / f"{t}.txt").write_text("rvkr")
    return corpus


def test_split_selection_full_track_test_split(arms: Path):
    be = FakeBackend()
    ac.run(_cfg(arms, splits_path=arms / "splits.json", split=["test"]), backend=be)
    recs = _records(arms / "captions" / "midasheng.jsonl")
    assert [(r["track_id"], r["split"], r["audio_source"]) for r in recs] == [(IDS[3], "test", "prepared")]
    assert recs[0]["audio_seconds"] == pytest.approx(4.0)  # full prepared track


def test_audio_dir_split_template_reads_windowed_wavs(arms: Path):
    be = FakeBackend()
    tmpl = str(arms / "arms" / "A" / "{split}")
    s = ac.run(_cfg(arms, prepared_root=None, audio_dir=tmpl, splits_path=arms / "splits.json",
                    split=["train", "val"]), backend=be)
    assert s == {"selected": 3, "skipped": 0, "ok": 3, "failed": 0}
    recs = _records(arms / "captions" / "midasheng.jsonl")
    assert [r["track_id"] for r in recs] == IDS[:3]  # manifest order
    assert [r["split"] for r in recs] == ["train", "val", "train"]
    assert all(r["audio_source"] == "audio_dir" and r["audio_seconds"] == 0.5 for r in recs)
    assert recs[1]["audio_dir"] == str(arms / "arms" / "A" / "val")
    assert be.lengths == [8000, 8000, 8000]  # the windowed 0.5 s WAVs, not the prepared ones
    meta = json.loads((arms / "captions" / "midasheng.jsonl.meta.json").read_text())
    assert meta["audio_source"] == "audio_dir" and meta["audio_dir"] == tmpl
    assert meta["splits"] == ["train", "val"]
    assert meta["audio_seconds"] == {t: 0.5 for t in IDS[:3]}


def test_audio_dir_train_then_val_resume_same_file(arms: Path):
    tmpl = str(arms / "arms" / "A" / "{split}")
    kw = dict(prepared_root=None, audio_dir=tmpl, splits_path=arms / "splits.json")
    ac.run(_cfg(arms, split=["train"], **kw), backend=FakeBackend())
    ac.run(_cfg(arms, split=["val"], resume=True, **kw), backend=FakeBackend())
    recs = _records(arms / "captions" / "midasheng.jsonl")
    assert sorted(r["track_id"] for r in recs) == sorted(IDS[:3])
    meta = json.loads((arms / "captions" / "midasheng.jsonl.meta.json").read_text())
    assert meta["splits"] == ["train", "val"] and len(meta["audio_seconds"]) == 3


def test_resume_refuses_mixing_windowed_and_full_track(arms: Path):
    tmpl = str(arms / "arms" / "A" / "{split}")
    ac.run(_cfg(arms, prepared_root=None, audio_dir=tmpl, splits_path=arms / "splits.json", split=["train"]),
           backend=FakeBackend())
    with pytest.raises(ValueError, match="windowed and full-track"):
        ac.run(_cfg(arms, splits_path=arms / "splits.json", split=["test"], resume=True), backend=FakeBackend())
    # Same refusal from the records alone (meta sidecar lost).
    (arms / "captions" / "midasheng.jsonl.meta.json").unlink()
    with pytest.raises(ValueError, match="separate files"):
        ac.run(_cfg(arms, splits_path=arms / "splits.json", split=["test"], resume=True), backend=FakeBackend())


def test_resume_refuses_other_audio_dir(arms: Path):
    kw = dict(prepared_root=None, splits_path=arms / "splits.json", split=["train"])
    ac.run(_cfg(arms, audio_dir=str(arms / "arms" / "A" / "{split}"), **kw), backend=FakeBackend())
    with pytest.raises(ValueError, match="audio_dir"):
        ac.run(_cfg(arms, audio_dir=str(arms / "arms" / "B" / "{split}"), resume=True, **kw), backend=FakeBackend())


def test_audio_dir_plain_and_wrong_rate(arms: Path):
    d = arms / "arms" / "A" / "train"
    sf.write(d / f"{IDS[2]}.wav", np.zeros((24000, 2), np.float32), 48000)  # not --sr 44100
    be = FakeBackend()
    s = ac.run(_cfg(arms, prepared_root=None, audio_dir=d, track_ids=[IDS[0], IDS[2]]), backend=be)
    assert s["ok"] == 1 and s["failed"] == 1
    f = _records(arms / "captions" / "_failures.jsonl")[0]
    assert f["track_id"] == IDS[2] and f["stage"] == "load" and "sample rate" in f["error"]


def test_split_errors(arms: Path):
    with pytest.raises(ValueError, match="no --split"):
        ac.run(_cfg(arms, audio_dir=str(arms / "arms" / "A" / "{split}")), backend=FakeBackend())
    with pytest.raises(ValueError, match="--split must be"):
        ac.run(_cfg(arms, splits_path=arms / "splits.json", split=["dev"]), backend=FakeBackend())
    with pytest.raises(KeyError, match="selected splits"):
        ac.run(_cfg(arms, splits_path=arms / "splits.json", split=["test"], track_ids=[IDS[0]]),
               backend=FakeBackend())
    (arms / "splits.json").write_text(json.dumps({**SPLITS, "test": ["ghost"]}))
    with pytest.raises(KeyError, match="not in manifest"):
        ac.run(_cfg(arms, splits_path=arms / "splits.json", split=["test"]), backend=FakeBackend())
    (arms / "splits.json").write_text(json.dumps({**SPLITS, "val": [IDS[0]]}))
    with pytest.raises(ValueError, match="both"):
        ac.load_splits(arms / "splits.json", ["train", "val"])


def test_cli_audio_dir_and_split(corpus: Path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ac, "run", lambda cfg: seen.setdefault("cfg", cfg) and {"failed": 0})
    rc = ac.main(["--audio-dir", "data/arms/A/{split}", "--splits", "manifests/splits.json",
                  "--split", "train, val", "--out", str(corpus / "c.jsonl")])
    cfg = seen["cfg"]
    assert rc == 0 and cfg.prepared_root is None and cfg.audio_dir == "data/arms/A/{split}"
    assert cfg.split == ["train", "val"] and cfg.splits_path == Path("manifests/splits.json")
    with pytest.raises(SystemExit):
        ac.main(["--out", str(corpus / "c.jsonl")])  # neither --prepared-root nor --audio-dir


# ------------------------------------------------------------------ echo detection + retry

PILOT_ECHO = "Instrumental, no lyrics, no BPM numbers."


@pytest.mark.parametrize(
    "caption, reason",
    [
        (PILOT_ECHO, "substring"),  # the v1 prompt's constraint clause, verbatim
        ("instrumental, no lyrics, no bpm numbers", "substring"),
        ("genre, mood, main instruments and how fast it feels", "substring"),
        ("covering its genre and mood and main instruments", "overlap"),
        ("instrumental music, genre and mood", "overlap"),
        ("lo-fi beat", "few_content_words"),
        ("calm piano music", "few_content_words"),
        ("", "empty"),
    ],
)
def test_is_echo_flags(caption, reason):
    assert ac.is_echo(ac.postprocess(caption, 15).caption, ac.echo_prompts(15)) == reason


@pytest.mark.parametrize(
    "caption",
    [
        "upbeat electronic track with punchy drums and bright synths",
        "warm jazz trio, brushed drums, walking bass, relaxed swing",
        "calm acoustic guitar with soft piano and a steady drum beat",
        "fast energetic rock with distorted guitars and a driving mood",  # 'mood' overlaps, 1/7
    ],
)
def test_is_echo_passes_real_captions(caption):
    assert ac.is_echo(caption, ac.echo_prompts(15)) is None


class EchoBackend(FakeBackend):
    """Echoes on the primary prompt for clips whose length (in s) is in ``echo_primary``; on the
    alt prompt for those in ``echo_alt``."""

    def __init__(self, echo_primary=(), echo_alt=()):
        super().__init__()
        self.echo_primary, self.echo_alt = set(echo_primary), set(echo_alt)

    def generate(self, audios, prompt):
        self.calls.append((len(audios), prompt))
        bad = self.echo_alt if "What does" in prompt else self.echo_primary
        secs = [a.shape[0] // 16000 for a in audios]
        return [PILOT_ECHO if n in bad else f"Mellow funk with slap bass and keys, {n} s" for n in secs]


def test_echo_retry_with_alt_prompt(corpus: Path):
    be = EchoBackend(echo_primary={1, 3})
    s = ac.run(_cfg(corpus, batch_size=4), backend=be)
    assert s == {"selected": 4, "skipped": 0, "ok": 4, "failed": 0}
    # one primary batch of 4, then one alt batch with only the two echoed clips
    assert [(n, p) for n, p in be.calls] == [(4, ac.build_prompt(15)), (2, ac.build_prompt(15, "alt"))]
    recs = {r["track_id"]: r for r in _records(corpus / "captions" / "midasheng.jsonl")}
    assert list(recs) == IDS  # manifest order kept
    r0 = recs[IDS[0]]
    assert r0["prompt_variant"] == "alt" and r0["echo_retry"] is True
    assert r0["echo_reason"] == "substring" and r0["raw_primary"] == PILOT_ECHO
    assert r0["caption"] == "mellow funk with slap bass and keys, 1 s"
    r1 = recs[IDS[1]]
    assert r1["prompt_variant"] == "primary" and r1["echo_retry"] is False and r1["raw_primary"] is None


def test_echo_twice_is_a_failure_not_a_caption(corpus: Path):
    be = EchoBackend(echo_primary={1}, echo_alt={1})
    s = ac.run(_cfg(corpus, limit=2), backend=be)
    assert s["ok"] == 1 and s["failed"] == 1
    assert [r["track_id"] for r in _records(corpus / "captions" / "midasheng.jsonl")] == [IDS[1]]
    f = _records(corpus / "captions" / "_failures.jsonl")[0]
    assert f["track_id"] == IDS[0] and f["stage"] == "echo"
    assert f["raw"] == PILOT_ECHO and f["raw_primary"] == PILOT_ECHO
    # --resume retries it (with both prompts again)
    be = EchoBackend()
    assert ac.run(_cfg(corpus, limit=2, resume=True), backend=be)["ok"] == 1


def test_resume_refuses_old_prompt_records_and_meta(corpus: Path):
    out = corpus / "captions" / "midasheng.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"track_id": IDS[0], "caption": "x", "model": ac.MIDASHENG_ID}) + "\n")
    with pytest.raises(ValueError, match="older prompt"):
        ac.run(_cfg(corpus, resume=True), backend=FakeBackend())
    out.unlink()
    ac.run(_cfg(corpus, limit=1), backend=FakeBackend())
    meta_path = corpus / "captions" / "midasheng.jsonl.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["prompt_version"] == 2 and meta["prompts"]["alt"] == ac.build_prompt(15, "alt")
    meta["prompt_sha"] = "0123456789abcdef"  # a run with other wordings
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="prompt_sha"):
        ac.run(_cfg(corpus, resume=True), backend=FakeBackend())


# ------------------------------------------------------------------ clause trimming (pilot cases)


def test_trim_pilot_0003_cuts_at_comma_not_mid_clause():
    raw = ("A gentle acoustic guitar melody with soft piano and a steady drum beat, "
           "creating a calm and peaceful atmosphere.")
    r = ac.postprocess(raw, 15)
    assert r.caption == "a gentle acoustic guitar melody with soft piano and a steady drum beat"
    assert r.n_words == 13 and not r.caption.endswith("calm")


def test_trim_prefers_comma_over_later_and():
    words = "w1 w2 w3 w4 w5 w6 w7 w8 w9, w10 w11 w12 w13 w14 and w16 w17 w18"
    assert ac.truncate_words(words, 15) == ("w1 w2 w3 w4 w5 w6 w7 w8 w9", True)


def test_trim_uses_and_when_no_comma():
    words = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 and w13 w14 w15 w16 w17"
    assert ac.truncate_words(words, 15) == ("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11", True)


def test_trim_boundary_must_keep_8_words_else_hard_cut():
    # comma after word 7 (< 8) does not count; no other boundary -> hard cut at 15
    words = "w1 w2 w3 w4 w5 w6 w7, w8 w9 w10 w11 w12 w13 w14 w15 w16 w17"
    text, cut = ac.truncate_words(words, 15)
    assert cut and text.split()[-1] == "w15" and len(text.split()) == 15
    # a comma after exactly word 8 is accepted
    words = "w1 w2 w3 w4 w5 w6 w7 w8, w9 w10 w11 w12 w13 w14 w15 w16 w17"
    assert ac.truncate_words(words, 15) == ("w1 w2 w3 w4 w5 w6 w7 w8", True)
