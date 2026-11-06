"""Batch driver for the structure tools: selection, skipping, failure logging,
run-config guard, sanity checks; plus the runner adapters.

The real tools need a GPU image; here fakes stand in: a fake ``analyze`` for
the driver, fake ``allin1`` / ``huggingface_hub`` modules for the allin1
adapter and the weight checks, and a numpy-backed fake ``torch`` with stub
MuQ/MusicFM/SongFormer models to drive :class:`SongFormerRunner` itself (its
window loop, including the tails that hang upstream)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ravi.structure import batch, run_allin1, run_songformer, schema

IDS = ["0001_copilot.mp3 Agent X NS", "0002_a", "0003_b", "0004_c", "0005_d"]
#: IDS[1]'s MP3 is not named {track_id}.mp3, so the --audio-root remap must use audio_filename
RENAMED = "renamed 0002 (master).mp3"


def _segments(dur: float = 30.0) -> list[dict]:
    return [{"start": 0.0, "end": dur / 2, "label": "Verse"}, {"start": dur / 2, "end": dur, "label": "chorus"}]


@pytest.fixture
def corpus(tmp_path: Path):
    """Manifest + splits + empty prepared WAV (with sidecar) and MP3 placeholders for IDS."""
    mp3_dir = tmp_path / "mp3"
    mp3_dir.mkdir()
    fnames = [RENAMED if t == IDS[1] else f"{t}.mp3" for t in IDS]
    df = pd.DataFrame(
        {
            "track_id": IDS,
            "audio_filename": fnames,
            "audio_path": [f"/nonexistent/{f}" for f in fnames],
            "duration_s": [30.0, 31.0, 32.0, 33.0, 34.0],
        }
    )
    manifest = tmp_path / "tracks.parquet"
    df.to_parquet(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": IDS[:3], "val": [IDS[3]], "test": [IDS[4]], "seed": 1}))
    prepared = tmp_path / "prepared"
    (prepared / "44100").mkdir(parents=True)
    for t, f in zip(IDS, fnames):
        wav = prepared / "44100" / f"{t}.wav"
        wav.touch()
        # the sidecar prepare_audio writes next to every WAV ({track_id}.wav.json)
        wav.with_name(wav.name + ".json").write_text(json.dumps({
            "track_id": t,
            "params": {"params_version": 2, "sr": 44100, "subtype": "FLOAT", "target_lufs": -16.0,
                       "ceiling_dbtp": -1.0, "resampler": "swr", "source_name": f, "source_bytes": 1234},
            "stats": {"n_frames": 0},
        }))
        (mp3_dir / f).touch()
    return types.SimpleNamespace(root=tmp_path, manifest=manifest, splits=splits, prepared=prepared, mp3=mp3_dir,
                                 df=df)


def _tracks(c, **kw):
    kw.setdefault("source", "prepared")
    return batch.load_tracks(c.manifest, prepared_root=c.prepared, splits_path=c.splits, **kw)


def fake_convert(raw, track, duration):
    return schema.from_songformer(raw, track.track_id, duration=duration)


class FakeTool:
    """Returns two segments per track ending at the manifest duration; ``fail`` ids
    raise (whole call), ``soft_fail`` come back as an Exception element,
    ``bad_output`` is schema-invalid, ``short_output`` ends 5 s early, ``empty`` has
    no segments."""

    def __init__(self, fail=(), soft_fail=(), bad_output=(), short_output=(), empty=(), sleep=0.0):
        self.fail, self.soft_fail, self.bad_output = set(fail), set(soft_fail), set(bad_output)
        self.short_output, self.empty, self.sleep = set(short_output), set(empty), sleep
        self.calls: list[list[str]] = []

    def __call__(self, tracks):
        ids = [t.track_id for t in tracks]
        self.calls.append(ids)
        if self.sleep:
            time.sleep(self.sleep)
        if self.fail & set(ids):
            raise RuntimeError(f"boom in {sorted(self.fail & set(ids))}")
        out = []
        for t in tracks:
            if t.track_id in self.soft_fail:
                out.append(ValueError("soft"))
            elif t.track_id in self.bad_output:
                out.append([{"start": 5.0, "end": 1.0, "label": "verse"}])  # end < start
            elif t.track_id in self.short_output:
                out.append(_segments(t.duration_s - 5.0))
            elif t.track_id in self.empty:
                out.append([])
            else:
                out.append(_segments(t.duration_s))
        return out


def _run(c, tool_fn, tracks=None, **kw):
    kw.setdefault("tool", "songformer")
    kw.setdefault("out_dir", c.root / "out" / kw["tool"])
    kw.setdefault("convert", fake_convert)
    kw.setdefault("duration_fn", lambda t: (t.duration_s, "manifest"))
    return batch.run(tracks if tracks is not None else _tracks(c), analyze=tool_fn, **kw)


def _failures(out_dir: Path) -> list[dict]:
    p = batch.failures_path(out_dir)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _parse(argv):
    ap = argparse.ArgumentParser()
    batch.add_common_args(ap)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_load_tracks_prepared_paths_order_and_sidecar(corpus):
    tr = _tracks(corpus)
    assert [t.track_id for t in tr] == IDS
    # the .wav.json sidecar next to it is read, never taken for the input
    assert tr[0].audio_path == corpus.prepared / "44100" / f"{IDS[0]}.wav"
    assert tr[0].prepared_params["params_version"] == 2
    assert tr[1].duration_s == 31.0
    # a WAV without sidecar still loads, with no params
    (corpus.prepared / "44100" / f"{IDS[2]}.wav.json").unlink()
    assert _tracks(corpus)[2].prepared_params is None


def test_split_filter_union_and_limit(corpus):
    assert [t.track_id for t in _tracks(corpus, split=["test"])] == [IDS[4]]
    assert [t.track_id for t in _tracks(corpus, split=["val", "test"])] == IDS[3:]
    assert [t.track_id for t in _tracks(corpus, split=["train"], limit=2)] == IDS[:2]
    assert [t.track_id for t in _tracks(corpus, track_ids=[IDS[2]])] == [IDS[2]]
    with pytest.raises(ValueError, match="not in manifest"):
        _tracks(corpus, track_ids=["nope"])


def test_limit_must_be_non_negative_and_empty_selection_is_an_error(corpus):
    with pytest.raises(ValueError, match="limit must be >= 0"):
        batch.select_rows(corpus.df, limit=-1)
    with pytest.raises(ValueError, match="no tracks selected"):
        _tracks(corpus, limit=0)
    with pytest.raises(ValueError, match="no tracks selected"):  # track outside the split
        _tracks(corpus, split=["test"], track_ids=[IDS[0]])
    with pytest.raises(SystemExit):
        _parse(["--limit", "-1"])
    with pytest.raises(SystemExit):
        _parse(["--max-consecutive-failures", "-2"])


def test_split_ids_missing_from_manifest_is_an_error(corpus):
    corpus.splits.write_text(json.dumps({"train": IDS[:3], "val": [IDS[3], "ghost"], "test": [IDS[4]]}))
    with pytest.raises(ValueError, match="not in the manifest.*ghost"):
        _tracks(corpus, split=["val"])
    assert [t.track_id for t in _tracks(corpus, split=["test"])] == [IDS[4]]  # other splits unaffected


def test_original_source_is_default_and_audio_root_uses_audio_filename(corpus):
    tr = batch.load_tracks(corpus.manifest, audio_root=corpus.mp3)
    assert tr[0].audio_path == corpus.mp3 / f"{IDS[0]}.mp3"
    assert tr[1].audio_path == corpus.mp3 / RENAMED  # not {track_id}.mp3
    assert tr[1].audio_path.is_file() and tr[0].prepared_params is None
    tr = batch.load_tracks(corpus.manifest, source="original")
    assert tr[1].audio_path == Path(f"/nonexistent/{RENAMED}")
    assert _parse([]).source == "original"


def test_audio_filename_column_required_for_original_or_audio_root(corpus):
    corpus.df.drop(columns=["audio_filename"]).to_parquet(corpus.manifest, index=False)
    with pytest.raises(ValueError, match="audio_filename"):
        batch.load_tracks(corpus.manifest, source="original")
    with pytest.raises(ValueError, match="audio_filename"):
        _tracks(corpus, audio_root=corpus.mp3)
    assert len(_tracks(corpus)) == 5  # prepared without a remap does not need it


def test_cli_args_round_trip(corpus):
    args = _parse(["--manifest", str(corpus.manifest), "--splits", str(corpus.splits), "--source", "prepared",
                   "--prepared-root", str(corpus.prepared), "--split", "train", "--limit", "1"])
    assert [t.track_id for t in batch.tracks_from_args(args)] == [IDS[0]]
    assert args.require_cuda is True and args.max_consecutive_failures == 10
    assert _parse(["--no-require-cuda"]).require_cuda is False


def test_input_meta_and_fingerprint(corpus):
    prep = _tracks(corpus)
    m = batch.input_meta(prep, source="prepared", sr=44100)
    assert m["source"] == "prepared" and m["input_sr"] == 44100
    assert batch.input_meta(prep[:2], source="prepared", sr=44100) == m  # selection-independent
    assert m["prepared_params"]["params_version"] == 2 and "source_bytes" not in m["prepared_params"]
    orig = batch.load_tracks(corpus.manifest, audio_root=corpus.mp3)
    assert batch.input_meta(orig, source="original", sr=44100) == {"source": "original", "input_sr": None}
    fp = batch.input_fingerprint(prep, source="prepared", sr=44100)
    assert fp == batch.input_fingerprint(prep[:2], source="prepared", sr=44100)  # selection-independent
    assert fp != batch.input_fingerprint(orig, source="original", sr=44100)
    side = corpus.prepared / "44100" / f"{IDS[0]}.wav.json"
    d = json.loads(side.read_text())
    d["params"]["params_version"] = 3
    side.write_text(json.dumps(d))
    assert batch.input_fingerprint(_tracks(corpus), source="prepared", sr=44100) != fp


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_writes_schema_docs_with_timing_meta_and_raw(corpus):
    tool = FakeTool()
    s = _run(corpus, tool, meta={"tool_version": "x", "host": "h1"}, batch_size=2)
    out = corpus.root / "out" / "songformer"
    assert (s.written, s.failed, s.skipped) == (5, 0, 0)
    assert tool.calls == [IDS[:2], IDS[2:4], IDS[4:]]
    doc = schema.load(batch.output_path(out, IDS[0]))
    assert doc["tool"] == "songformer" and doc["track_id"] == IDS[0]
    assert doc["segments"][0]["label"] == "verse"  # converter lower-cases
    assert doc["duration"] == 30.0
    assert doc["timing"]["batch_size"] == 2 and doc["timing"]["wall_s"] >= 0
    assert doc["meta"]["tool_version"] == "x" and doc["meta"]["run_id"] == s.run_id
    assert doc["meta"]["input_path"].endswith(f"{IDS[0]}.wav") and doc["meta"]["input_bytes"] == 0
    assert doc["meta"]["duration_source"] == "manifest"
    assert doc["meta"]["prepared_params"]["source_name"] == f"{IDS[0]}.mp3"
    assert json.loads(batch.raw_path(out, IDS[0]).read_text()) == _segments(30.0)
    # the run config: meta minus run_id/host
    assert json.loads(batch.config_path(out).read_text()) == {"tool_version": "x"}
    # eval reads only top-level *.json: exactly the 5 docs, no temp files left behind
    assert sorted(p.stem for p in out.glob("*.json")) == sorted(IDS)
    assert not list(out.rglob("*.tmp"))
    assert not batch.failures_path(out).exists()


def test_skips_existing_outputs_and_overwrite(corpus):
    _run(corpus, FakeTool(), batch_size=10)
    tool = FakeTool()
    s = _run(corpus, tool, batch_size=10)
    assert (s.skipped, s.written) == (5, 0) and tool.calls == []
    # remove one output -> only that one reruns
    batch.output_path(corpus.root / "out" / "songformer", IDS[3]).unlink()
    s = _run(corpus, tool, batch_size=10)
    assert tool.calls == [[IDS[3]]] and (s.skipped, s.written) == (4, 1)
    s = _run(corpus, FakeTool(), overwrite=True)
    assert (s.skipped, s.written) == (0, 5)


def test_config_mismatch_refuses_resume_and_overwrite_clears(corpus):
    out = corpus.root / "out" / "songformer"
    meta = {"source": "original", "checkpoint": {"commit": "abc"}, "host": "a"}
    _run(corpus, FakeTool(), meta=meta, tracks=_tracks(corpus)[:3])
    # same config on another host, another run id: resumes
    s = _run(corpus, FakeTool(), meta={**meta, "host": "b"})
    assert (s.skipped, s.written) == (3, 2)
    changed = {**meta, "source": "prepared", "checkpoint": {"commit": "def"}}
    tool = FakeTool()
    with pytest.raises(batch.ConfigMismatch) as ei:
        _run(corpus, tool, meta=changed)
    msg = str(ei.value)
    assert 'source: "original" -> "prepared"' in msg and 'checkpoint.commit: "abc" -> "def"' in msg
    assert "--overwrite" in msg and tool.calls == []
    # --overwrite with a different config clears everything, then runs the selection
    s = _run(corpus, tool, meta=changed, overwrite=True, tracks=_tracks(corpus)[:2])
    assert s.written == 2 and sorted(p.stem for p in out.glob("*.json")) == sorted(IDS[:2])
    assert sorted(p.stem for p in (out / "_raw").glob("*.json")) == sorted(IDS[:2])
    assert json.loads(batch.config_path(out).read_text())["source"] == "prepared"


def test_outputs_without_config_are_refused(corpus):
    out = corpus.root / "out" / "songformer"
    _run(corpus, FakeTool(), tracks=_tracks(corpus)[:1])
    batch.config_path(out).unlink()  # e.g. results of a run before the guard existed
    with pytest.raises(batch.ConfigMismatch, match="configuration is unknown"):
        _run(corpus, FakeTool())
    assert _run(corpus, FakeTool(), overwrite=True).written == 5


def test_batch_failure_falls_back_to_single_tracks_and_continues(corpus):
    tool = FakeTool(fail={IDS[1]})
    s = _run(corpus, tool, batch_size=3)
    out = corpus.root / "out" / "songformer"
    assert s.written == 4 and s.failed == 1 and s.failed_ids == [IDS[1]]
    # chunk 1 failed as a whole, then retried one by one; chunk 2 untouched
    assert tool.calls == [IDS[:3], [IDS[0]], [IDS[1]], [IDS[2]], IDS[3:]]
    assert not batch.output_path(out, IDS[1]).exists()
    (rec,) = _failures(out)
    assert rec["track_id"] == IDS[1] and rec["stage"] == "analyze" and rec["tool"] == "songformer"
    assert "RuntimeError: boom" in rec["error"] and "Traceback" in rec["traceback"]
    assert rec["run_id"] == s.run_id
    # rerun retries only the failure; the log is append-only
    tool2 = FakeTool(fail={IDS[1]})
    _run(corpus, tool2, batch_size=3)
    assert tool2.calls == [[IDS[1]]]
    assert len(_failures(out)) == 2


def test_soft_failure_bad_output_and_missing_input(corpus):
    (corpus.prepared / "44100" / f"{IDS[4]}.wav").unlink()
    tool = FakeTool(soft_fail={IDS[0]}, bad_output={IDS[2]})
    s = _run(corpus, tool, batch_size=5)
    out = corpus.root / "out" / "songformer"
    assert s.written == 2 and sorted(s.failed_ids) == sorted([IDS[0], IDS[2], IDS[4]])
    stages = {r["track_id"]: r["stage"] for r in _failures(out)}
    assert stages == {IDS[4]: "input", IDS[0]: "analyze", IDS[2]: "convert"}
    assert tool.calls == [IDS[:4]]  # missing input never reaches the tool
    # the raw output of a convert failure is kept for debugging; no doc written
    assert batch.raw_path(out, IDS[2]).exists() and not batch.output_path(out, IDS[2]).exists()


def test_sanity_check_rejects_empty_and_truncated_output(corpus):
    tool = FakeTool(short_output={IDS[1]}, empty={IDS[3]})
    s = _run(corpus, tool, batch_size=5)
    out = corpus.root / "out" / "songformer"
    assert s.written == 3 and sorted(s.failed_ids) == sorted([IDS[1], IDS[3]])
    errs = {r["track_id"]: (r["stage"], r["error"]) for r in _failures(out)}
    assert errs[IDS[1]][0] == "check" and "last segment ends at 26.000s" in errs[IDS[1]][1]
    assert errs[IDS[3]] == ("check", "ValueError: tool returned no segments")
    assert not batch.output_path(out, IDS[1]).exists()
    # within tolerance passes; unknown duration checks only non-emptiness
    batch.sanity_check({"segments": _segments(30.0)}, 30.9)
    batch.sanity_check({"segments": _segments(30.0)}, None)
    with pytest.raises(ValueError):
        batch.sanity_check({"segments": _segments(30.0)}, 31.0)


def test_wrong_result_count_rejects_the_batch_then_fails_each_track(corpus):
    tool_calls = []

    def short(tracks):
        tool_calls.append(len(tracks))
        return [_segments(t.duration_s) for t in tracks][:-1]  # always one result short

    s = _run(corpus, short, tracks=_tracks(corpus)[:2], batch_size=2)
    # the 2-track call returns 1 result -> rejected; each 1-track retry returns 0 -> failure
    assert tool_calls == [2, 1, 1]
    assert s.written == 0 and s.failed_ids == IDS[:2]
    errs = [r["error"] for r in _failures(corpus.root / "out" / "songformer")]
    assert errs == ["RuntimeError: analyze returned 0 results for 1 tracks"] * 2


def test_wall_s_is_shared_only_by_tracks_that_ran(corpus):
    s = _run(corpus, FakeTool(soft_fail={IDS[1]}, sleep=0.05), tracks=_tracks(corpus)[:3], batch_size=3)
    out = corpus.root / "out" / "songformer"
    assert s.written == 2
    t = schema.load(batch.output_path(out, IDS[0]))["timing"]
    assert t["batch_size"] == 3 and t["batch_ran"] == 2 and t["batch_wall_s"] >= 0.05
    assert t["wall_s"] == pytest.approx(t["batch_wall_s"] / 2, abs=1e-3)


def test_timed_results_keep_their_own_wall_time(corpus):
    def analyze(tracks):
        return [batch.Timed(_segments(t.duration_s), 1.5 + i) for i, t in enumerate(tracks)]

    _run(corpus, analyze, tracks=_tracks(corpus)[:2], batch_size=2)
    out = corpus.root / "out" / "songformer"
    assert [schema.load(batch.output_path(out, i))["timing"]["wall_s"] for i in IDS[:2]] == [1.5, 2.5]


def test_aborts_after_max_consecutive_failures(corpus):
    tool = FakeTool(soft_fail=set(IDS[1:4]))
    s = _run(corpus, tool, batch_size=1, max_consecutive_failures=2)
    assert s.aborted and "2 consecutive failures" in s.aborted
    assert s.written == 1 and s.failed_ids == IDS[1:3]
    assert tool.calls == [[IDS[0]], [IDS[1]], [IDS[2]]]  # IDS[3:] never attempted
    assert "ABORTED" in s.line()
    # a success resets the streak; 0 disables the limit
    tool = FakeTool(soft_fail={IDS[0], IDS[2], IDS[4]})
    assert _run(corpus, tool, out_dir=corpus.root / "o2", batch_size=1, max_consecutive_failures=2).aborted is None
    s = _run(corpus, FakeTool(soft_fail=set(IDS)), out_dir=corpus.root / "o3", max_consecutive_failures=0)
    assert s.aborted is None and s.failed == 5


def test_missing_inputs_count_towards_the_abort(corpus):
    tracks = batch.load_tracks(corpus.manifest)  # original: /nonexistent/...
    tool = FakeTool()
    s = _run(corpus, tool, tracks=tracks, max_consecutive_failures=3)
    assert s.aborted and s.failed == 3 and tool.calls == []


def test_converter_must_match_tool_and_track(corpus):
    def wrong(raw, track, duration):
        return schema.from_songformer(raw, "someone_else", duration=duration)

    s = _run(corpus, FakeTool(), convert=wrong, tracks=_tracks(corpus)[:1])
    assert s.failed == 1 and _failures(corpus.root / "out" / "songformer")[0]["stage"] == "convert"


def test_rejects_unknown_tool(corpus):
    with pytest.raises(ValueError):
        _run(corpus, FakeTool(), tool="human")


def test_probe_duration_reads_wav_and_falls_back(tmp_path):
    p = tmp_path / "x.wav"
    sf.write(p, np.zeros((44100 * 2, 2), np.float32), 44100)
    assert batch.probe_duration(batch.Track("x", p, 9.0)) == pytest.approx(2.0)
    assert batch.probe_duration_source(batch.Track("x", p, 9.0))[1] == "input_file"
    assert batch.probe_duration_source(batch.Track("y", tmp_path / "missing.wav", 9.0)) == (9.0, "manifest")
    assert batch.probe_duration_source(batch.Track("z", tmp_path / "missing.wav")) == (None, "none")


def test_json_default_handles_arrays_and_scalars():
    s = json.dumps({"a": np.arange(3), "b": np.float32(1.5), "c": np.array(2.0), "p": Path("/x")},
                   default=batch._json_default)
    assert json.loads(s) == {"a": [0, 1, 2], "b": 1.5, "c": 2.0, "p": "/x"}


def test_require_cuda(monkeypatch):
    monkeypatch.setattr(batch, "_cuda_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA required"):
        batch.resolve_device(None, require_cuda=True)
    assert batch.resolve_device(None, require_cuda=False) == "cpu"
    monkeypatch.setattr(batch, "_cuda_available", lambda: True)
    assert batch.resolve_device(None, require_cuda=True) == "cuda"
    assert batch.resolve_device("cuda:1", require_cuda=True) == "cuda:1"
    with pytest.raises(SystemExit):
        batch.resolve_device("cpu", require_cuda=True)


def test_hf_cached_file_info_records_cache_misses(monkeypatch, tmp_path):
    snap = tmp_path / "models--r" / "snapshots" / "c0ffee"
    snap.mkdir(parents=True)
    blob = tmp_path / "models--r" / "blobs" / "e7a9"
    blob.parent.mkdir()
    blob.touch()
    (snap / "w.bin").symlink_to(blob)
    answers = {"hit": str(snap / "w.bin"), "miss": None, "gone": object()}
    fake = types.SimpleNamespace(try_to_load_from_cache=lambda repo, fn, revision=None: answers[fn])
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    hit = batch.hf_cached_file_info("r", "hit", "c0ffee")
    assert hit["commit"] == "c0ffee" and hit["blob"] == "e7a9" and "error" not in hit
    assert batch.hf_cached_file_info("r", "miss")["error"] == "not in the local HF cache"
    assert "non-existent" in batch.hf_cached_file_info("r", "gone")["error"]


# ---------------------------------------------------------------------------
# allin1 adapter (no torch / allin1 needed)
# ---------------------------------------------------------------------------


@dataclass
class _Seg:
    start: float
    end: float
    label: str


@dataclass
class _Result:  # shape of allin1.typings.AnalysisResult
    path: Path
    bpm: int
    beats: list
    downbeats: list
    beat_positions: list
    segments: list
    activations: object = None
    embeddings: object = None


def test_allin1_adapter_matches_results_by_path(corpus, monkeypatch):
    calls = []

    def fake_analyze(paths, **kw):
        calls.append((list(paths), kw))
        # allin1 resolves and *sorts* paths and returns results in that order
        rs = sorted(Path(p).expanduser().resolve() for p in paths)
        return [
            _Result(p, np.int64(120), [np.float64(0.5), 1.0], [0.5], [1, 2],
                    [_Seg(np.float64(0.0), 1.0, "start"), _Seg(1.0, 2.0, "Chorus")])
            for p in rs
        ]

    monkeypatch.setitem(sys.modules, "allin1", types.SimpleNamespace(analyze=fake_analyze))
    analyze = run_allin1.make_analyzer(model="harmonix-all", device="cpu", work_dir=corpus.root / "w")
    tracks = list(reversed(_tracks(corpus)[:3]))
    tracks.append(batch.Track("glob?", corpus.root / "what?.wav"))
    out = analyze(tracks)
    assert [Path(o["path"]).name for o in out[:3]] == [t.audio_path.name for t in tracks[:3]]
    assert isinstance(out[3], ValueError)  # never passed to allin1
    (paths, kw), = calls
    assert len(paths) == 3 and kw["out_dir"] is None and kw["device"] == "cpu"
    # fresh per-call scratch under work_dir, removed afterwards
    assert kw["demix_dir"].parent.parent == corpus.root / "w" and kw["demix_dir"].name == "demix"
    assert kw["keep_byproducts"] is False and not kw["demix_dir"].parent.exists()
    raw = out[0]
    assert raw["bpm"] == 120.0 and isinstance(raw["beats"][0], float)
    json.dumps(raw)  # plain JSON types only
    doc = run_allin1.convert(raw, tracks[0], 2.0)
    assert doc["tool"] == "allin1" and doc["track_id"] == tracks[0].track_id
    assert [s["label"] for s in doc["segments"]] == ["start", "chorus"] and doc["bpm"] == 120.0


def test_allin1_glob_guard_checks_the_resolved_path(tmp_path, monkeypatch):
    weird = tmp_path / "we*ird"
    weird.mkdir()
    (weird / "x.wav").touch()
    (tmp_path / "link").symlink_to(weird)
    called = []
    monkeypatch.setitem(sys.modules, "allin1", types.SimpleNamespace(analyze=lambda paths, **kw: called.append(1)))
    analyze = run_allin1.make_analyzer(model="harmonix-all", device="cpu", work_dir=tmp_path / "w")
    (out,) = analyze([batch.Track("x", tmp_path / "link" / "x.wav")])  # no glob char until resolved
    assert isinstance(out, ValueError) and "we*ird" in str(out) and called == []


def test_allin1_end_to_end_with_fake_module(corpus, monkeypatch):
    def fake_analyze(paths, **kw):
        out = []
        for p in sorted(paths):
            dur = 30.0 if IDS[0] in p else 2.0  # all but IDS[0] end far from their duration
            out.append(_Result(Path(p).resolve(), 100, [0.6], [0.6], [1], [_Seg(0.0, dur, "verse")]))
        return out

    monkeypatch.setitem(sys.modules, "allin1", types.SimpleNamespace(analyze=fake_analyze))
    analyze = run_allin1.make_analyzer(model="harmonix-all", device="cpu", work_dir=corpus.root / "w")
    out = corpus.root / "out" / "allin1"
    tracks = batch.load_tracks(corpus.manifest, audio_root=corpus.mp3)
    s = batch.run(tracks, tool="allin1", out_dir=out, analyze=analyze, convert=run_allin1.convert,
                  batch_size=4, duration_fn=lambda t: t.duration_s)
    assert s.written == 1 and s.failed == 4 and {r["stage"] for r in _failures(out)} == {"check"}
    doc = schema.load(batch.output_path(out, IDS[0]))
    assert doc["beats"] == [0.6] and doc["bpm"] == 100.0
    assert doc["meta"]["input_path"] == str(corpus.mp3 / f"{IDS[0]}.mp3")
    assert doc["meta"]["duration_source"] == "duration_fn" and "prepared_params" not in doc["meta"]


def test_allin1_pins_checkpoint_revision(monkeypatch):
    seen = []

    def hf_hub_download(repo_id=None, filename=None, cache_dir=None, revision=None):
        seen.append((repo_id, filename, revision))
        return "/x"

    loaders = types.SimpleNamespace(hf_hub_download=hf_hub_download)
    models = types.SimpleNamespace(loaders=loaders)
    monkeypatch.setitem(sys.modules, "allin1", types.SimpleNamespace(models=models))
    monkeypatch.setitem(sys.modules, "allin1.models", models)
    monkeypatch.setitem(sys.modules, "allin1.models.loaders", loaders)
    run_allin1.pin_checkpoint_revision()
    run_allin1.pin_checkpoint_revision()  # idempotent: not wrapped twice
    loaders.hf_hub_download(repo_id=run_allin1.HF_REPO, filename="harmonix-fold0-0vra4ys2.pth", cache_dir=None)
    loaders.hf_hub_download(repo_id="other/repo", filename="f")
    assert seen == [(run_allin1.HF_REPO, "harmonix-fold0-0vra4ys2.pth", run_allin1.HF_REV),
                    ("other/repo", "f", None)]
    assert run_allin1.HF_REV == "379e5fd010b3fdd0ee8381ff8cbcfa51d70b5c19"


def test_allin1_keep_byproducts_work_dir(tmp_path):
    base = tmp_path / "_work"
    assert run_allin1.prepare_work_dir(base, "fp1", keep_byproducts=False, overwrite=False) == base
    d = run_allin1.prepare_work_dir(base, "fp1", keep_byproducts=True, overwrite=False)
    assert d == base / "fp1" and d.is_dir()
    (d / "spec").mkdir()
    (d / "spec" / "x.npy").touch()
    with pytest.raises(SystemExit, match="not empty"):
        run_allin1.prepare_work_dir(base, "fp1", keep_byproducts=True, overwrite=False)
    # another input configuration gets its own dir
    assert run_allin1.prepare_work_dir(base, "fp2", keep_byproducts=True, overwrite=False) == base / "fp2"
    d = run_allin1.prepare_work_dir(base, "fp1", keep_byproducts=True, overwrite=True)
    assert d.is_dir() and not any(d.iterdir())


# ---------------------------------------------------------------------------
# SongFormer adapter and runner (fake torch / models)
# ---------------------------------------------------------------------------


def test_songformer_adapter_isolates_per_track_errors_and_times_each(corpus):
    def runner(path):
        if "0002" in str(path):
            raise RuntimeError("cuda oom")
        return {"segments": [{"label": "intro", "start": 0.0, "end": 4.0},
                             {"label": "pre-chorus", "start": 4.0, "end": 9.5}]}

    analyze = run_songformer.make_analyzer(runner)
    out = analyze(_tracks(corpus)[:3])
    assert all(isinstance(o, batch.Timed) and o.wall_s >= 0 for o in out)
    assert isinstance(out[1].result, RuntimeError) and out[0].result["segments"][1]["label"] == "pre-chorus"
    doc = run_songformer.convert(out[0].result, _tracks(corpus)[0], 9.5)
    assert doc["tool"] == "songformer" and doc["beats"] is None and len(doc["segments"]) == 2


class _FT:
    """Just enough of a torch.Tensor (numpy-backed) for SongFormerRunner.__call__."""

    def __init__(self, a):
        self.a = np.asarray(a.a if isinstance(a, _FT) else a)

    shape = property(lambda self: self.a.shape)

    def __getitem__(self, k):
        return _FT(self.a[k])

    def to(self, *args, **kwargs):
        return self

    def unsqueeze(self, d):
        return _FT(np.expand_dims(self.a, d))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


_fake_torch = types.SimpleNamespace(
    tensor=_FT, Tensor=_FT, from_numpy=_FT, long="long",
    no_grad=contextlib.nullcontext,
    cuda=types.SimpleNamespace(empty_cache=lambda: None),
    concatenate=lambda xs, dim=None, axis=None: _FT(np.concatenate([x.a for x in xs],
                                                                   axis=dim if dim is not None else axis)),
)

SR24 = 24000
HOP_24K = 960  # MuQ / MusicFM: 25 Hz frames at 24 kHz


class _FakeEncoder:
    def __init__(self, calls):
        self.calls = calls

    def _hidden(self, x):
        self.calls.append(x.shape[-1])
        return [_FT(np.zeros((1, math.ceil(x.shape[-1] / HOP_24K), 4)))] * 11

    def __call__(self, x, output_hidden_states=True):  # MuQ
        return {"hidden_states": self._hidden(x)}

    def get_predictions(self, x):  # MusicFM
        return None, self._hidden(x)


class _FakeModel:
    def infer(self, input_embeddings, dataset_ids, label_id_masks, with_logits):
        n = math.ceil(input_embeddings.shape[1] / 3)  # 25 Hz -> 8.333 Hz
        return None, {"function_logits": _FT(np.zeros((1, n, 128))), "boundary_logits": _FT(np.zeros((1, n)))}


def _fake_runner(monkeypatch, n_samples: int, rule_post_processing: bool = True):
    calls: list[int] = []
    monkeypatch.setitem(sys.modules, "librosa", types.SimpleNamespace(
        load=lambda path, sr=None: (np.zeros(n_samples, np.float32), sr)))
    up = types.SimpleNamespace(
        INPUT_SAMPLING_RATE=SR24, TIME_DUR=420, AFTER_DOWNSAMPLING_FRAME_RATES=25 / 3,
        DATASET_LABEL="x", DATASET_IDS=[5], DATASET_LABEL_TO_DATASET_ID={"x": 5},
        postprocess_functional_structure=lambda logits, hp: [
            (0.0, "intro"), (logits["boundary_logits"].shape[1] / (25 / 3), "end")],
        rule_post_processing=lambda m: [(t, "verse" if lab == "intro" else lab) for t, lab in m],
    )
    r = run_songformer.SongFormerRunner.__new__(run_songformer.SongFormerRunner)  # skip model loading
    r.up, r.np, r.torch, r.device, r.hp = up, np, _fake_torch, "cpu", None
    r.rule_post_processing = rule_post_processing
    r.win_size = r.hop_size = 420
    r.num_classes = 128
    enc = _FakeEncoder(calls)
    r.muq, r.musicfm, r.model = enc, enc, _FakeModel()
    r.dataset_id2label_mask = {5: np.zeros(128, bool)}
    return r, calls


@pytest.fixture
def deadline():
    """Turn a hang (upstream's `continue` without advancing) into a test failure."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def boom(*_):
        raise TimeoutError("SongFormerRunner did not terminate")

    old = signal.signal(signal.SIGALRM, boom)
    signal.alarm(20)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


_UPSTREAM_PAD_DIVIDE = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")


@_UPSTREAM_PAD_DIVIDE
@pytest.mark.parametrize(
    "n_samples, extra_windows",
    [
        (SR24 * 420 + 500, 0),   # tail of 500 samples: upstream hangs
        (SR24 * 420 + 1024, 0),  # 1024: still <= 1024, upstream hangs
        (SR24 * 420 + 1025, 1),  # 1025: processed as a (tiny) last window
        (SR24 * 420, 0),         # exact multiple: loop ends on start_idx >= len
        (SR24 * 32, None),       # single short window
    ],
)
def test_songformer_runner_window_tails_terminate(monkeypatch, deadline, n_samples, extra_windows):
    r, calls = _fake_runner(monkeypatch, n_samples)
    out = r("x.wav")
    if extra_windows is not None:
        # first 420 s window: one 420 s pass + fourteen 30 s passes, x2 encoders
        assert len(calls) == 2 * (15 + 2 * extra_windows)  # x2: MuQ and MusicFM share the fake
        assert calls.count(SR24 * 420) == 2
    assert out["n_samples_24k"] == n_samples
    assert out["rule_post_processing"] is True
    (seg,) = out["segments"]
    assert seg["label"] == "verse" and seg["start"] == 0.0
    assert out["segments_before_rule_post_processing"][0]["label"] == "intro"
    # ends at the frames actually covered: ~ the audio length (within a frame or so)
    assert abs(seg["end"] - n_samples / SR24) < 1.0
    doc = run_songformer.convert(out, batch.Track("x", Path("x.wav")), n_samples / SR24)
    batch.sanity_check(doc, n_samples / SR24)


@pytest.mark.parametrize("n_samples", [0, 1000, 1024])
def test_songformer_runner_rejects_audio_shorter_than_one_window(monkeypatch, deadline, n_samples):
    r, calls = _fake_runner(monkeypatch, n_samples)
    with pytest.raises(ValueError, match="audio shorter than 1025 samples"):
        r("x.wav")
    assert calls == []


@_UPSTREAM_PAD_DIVIDE
def test_songformer_runner_without_rule_post_processing(monkeypatch, deadline):
    r, _ = _fake_runner(monkeypatch, SR24 * 10, rule_post_processing=False)
    out = r("x.wav")
    assert out["segments"] == out["segments_before_rule_post_processing"]
    assert out["segments"][0]["label"] == "intro" and out["rule_post_processing"] is False


def test_songformer_pins_w2v2_config_revision():
    seen = []

    class Base:
        @classmethod
        def from_pretrained(cls, name, revision=None):
            seen.append((cls.__name__, name, revision))

    class Config(Base):
        pass

    with run_songformer.pinned_from_pretrained(Config, run_songformer.W2V2_CONFIG_REPO, "rev1"):
        Config.from_pretrained(run_songformer.W2V2_CONFIG_REPO)
        Config.from_pretrained("other")
        Config.from_pretrained(run_songformer.W2V2_CONFIG_REPO, revision="explicit")
    Config.from_pretrained(run_songformer.W2V2_CONFIG_REPO)  # restored
    assert "from_pretrained" not in vars(Config)
    assert seen == [("Config", run_songformer.W2V2_CONFIG_REPO, "rev1"), ("Config", "other", None),
                    ("Config", run_songformer.W2V2_CONFIG_REPO, "explicit"),
                    ("Config", run_songformer.W2V2_CONFIG_REPO, None)]


def test_songformer_fetch_assets_checks_md5_and_muq_sha256(tmp_path, monkeypatch):
    files = tmp_path / "hf"
    files.mkdir()
    downloads = []

    def hf_hub_download(repo, filename, revision=None):
        downloads.append((repo, filename, revision))
        p = files / repo.replace("/", "--") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(f"{repo}/{filename}".encode())
        return str(p)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download))
    home = tmp_path / "SongFormer"
    md5 = home / "src" / "SongFormer" / "ckpts" / "md5sum.txt"
    md5.parent.mkdir(parents=True)

    def md5_of(repo, fn):
        return hashlib.md5(f"{repo}/{fn}".encode()).hexdigest()

    md5.write_text(f"{md5_of(run_songformer.SONGFORMER_REPO, run_songformer.SONGFORMER_CKPT)}  SongFormer.safetensors\n"
                   f"{md5_of(run_songformer.MUSICFM_REPO, 'msd_stats.json')}  MusicFM/msd_stats.json\n"
                   f"{md5_of(run_songformer.MUSICFM_REPO, 'pretrained_msd.pt')}  MusicFM/pretrained_msd.pt\n")
    muq_sha = hashlib.sha256(f"{run_songformer.MUQ_REPO}/model.safetensors".encode()).hexdigest()

    # the real pinned digest does not match the fake file
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        run_songformer.fetch_assets(home)
    monkeypatch.setattr(run_songformer, "MUQ_SHA256", muq_sha)
    a = run_songformer.fetch_assets(home)
    assert a.md5_verified is True and a.muq_weights.name == "model.safetensors"
    assert (run_songformer.W2V2_CONFIG_REPO, "config.json", run_songformer.W2V2_CONFIG_REV) in downloads
    assert run_songformer.fetch_assets(home, verify_md5=False).md5_verified is False
    md5.write_text(md5.read_text().splitlines()[0] + "\n")  # upstream md5 missing for MusicFM files
    assert run_songformer.fetch_assets(home).md5_verified is False
    assert run_songformer.MUQ_SHA256 == muq_sha  # (monkeypatched)


def test_pinned_constants():
    assert run_songformer.W2V2_CONFIG_REV == "6b36ef01c6443c67ae7ed0822876d091ab50e4aa"
    sha = "273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6"
    assert run_songformer.MUQ_SHA256 == sha and len(sha) == 64
