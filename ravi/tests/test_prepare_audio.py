import json
import shutil
import subprocess

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

from ravi.data import prepare_audio as pa
from ravi.data.prepare_audio import normalise, true_peak_dbtp

SR = 44100


def test_normalise_hits_target_when_headroom():
    rng = np.random.default_rng(0)
    x = (0.01 * rng.standard_normal((SR * 5, 2))).astype(np.float32)  # quiet noise, low crest
    y, st = normalise(x, SR, -14.0, -1.0)
    assert not st["peak_limited"]
    assert pyln.Meter(SR).integrated_loudness(y.astype(np.float64)) == pytest.approx(-14.0, abs=0.05)
    assert st["true_peak_dbtp"] <= -1.0


def test_normalise_reduces_gain_to_respect_ceiling():
    rng = np.random.default_rng(1)
    x = (0.05 * rng.standard_normal((SR * 5, 2))).astype(np.float32)
    x[SR, :] = 0.99  # a single near-full-scale spike
    y, st = normalise(x, SR, -14.0, -1.0)
    assert st["peak_limited"]
    assert st["true_peak_dbtp"] == pytest.approx(-1.0, abs=0.05)
    assert st["lufs_out"] < -14.0


def test_true_peak_catches_intersample_overs():
    t = np.arange(SR) / SR
    # fs/4 sine sampled at 45 degrees: sample peak 0.707 of true peak
    x = np.sin(2 * np.pi * (SR / 4) * t + np.pi / 4).astype(np.float32)[:, None].repeat(2, 1)
    assert np.max(np.abs(x)) == pytest.approx(np.sqrt(0.5), abs=1e-3)
    assert true_peak_dbtp(x) == pytest.approx(0.0, abs=0.3)


def test_silence_left_untouched():
    x = np.zeros((SR * 2, 2), np.float32)
    y, st = normalise(x, SR, -14.0, -1.0)
    assert st["silent"] and np.all(y == 0)


def test_correction_step_lands_on_target_and_records_residual():
    rng = np.random.default_rng(2)
    x = (0.003 * rng.standard_normal((SR * 4, 2))).astype(np.float32)
    y, st = normalise(x, SR, -16.0, -1.0)
    assert not st["peak_limited"]
    assert abs(st["lufs_residual"]) < 0.01
    assert st["lufs_out"] == pytest.approx(pyln.Meter(SR).integrated_loudness(y.astype(np.float64)), abs=1e-9)
    assert "correction_db" in st and abs(st["correction_db"]) < 0.05


# --- decode + per-WAV sidecar (needs ffmpeg) -------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)


def make_mp3_with_cover(tmp_path, seconds=2, sr=48000):
    """Mono MP3 with an embedded PNG cover (attached_pic video stream), like the corpus."""
    png, mp3 = tmp_path / "cover.png", tmp_path / "0001_copilot.mp3 Test NS.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64",
                    "-frames:v", "1", str(png)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(png), "-f", "lavfi",
                    "-i", f"sine=f=440:d={seconds}:sample_rate={sr}", "-map", "0:v", "-map", "1:a",
                    "-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1", "-c:v", "png",
                    "-disposition:v", "attached_pic", "-id3v2_version", "3", str(mp3)], check=True)
    return mp3


@needs_ffmpeg
@pytest.mark.parametrize("sr", [44100, 48000])
def test_decode_selects_audio_stream_and_upmixes(tmp_path, sr):
    mp3 = make_mp3_with_cover(tmp_path)
    types = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                            str(mp3)], capture_output=True, text=True, check=True).stdout.split()
    assert sorted(types) == ["audio", "video"]  # cover art is a real second stream
    a = pa.decode(str(mp3), sr)
    assert a.dtype == np.float32 and a.ndim == 2 and a.shape[1] == 2
    assert a.shape[0] == 2 * sr  # LAME delay/padding trimmed by ffmpeg: exact length
    assert np.array_equal(a[:, 0], a[:, 1])  # mono source duplicated to both channels
    assert 0.01 < np.abs(a).max() < 1.0


@needs_ffmpeg
def test_decode_error_includes_ffmpeg_stderr(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not an mp3 at all")
    with pytest.raises(RuntimeError, match="ffmpeg exited") as ei:
        pa.decode(str(bad), 44100)
    assert len(str(ei.value)) > len(f"ffmpeg exited 1 decoding {bad}: ")  # stderr text is attached


@needs_ffmpeg
def test_process_one_writes_sidecar_and_skips_only_on_matching_params(tmp_path):
    mp3 = make_mp3_with_cover(tmp_path)
    out = tmp_path / "prepared" / "44100" / "t.wav"
    job = {"track_id": "t", "audio_path": str(mp3), "sr": 44100, "out_path": str(out), "subtype": "FLOAT",
           "target_lufs": -16.0, "ceiling_dbtp": -1.0, "resampler": "swr"}
    r1 = pa.process_one(job)
    assert r1["error"] is None and not r1["skipped"]
    side = json.loads(pa.sidecar_path(out).read_text())
    assert side["params"] == pa.job_params(job)
    assert side["params"]["source_name"] == mp3.name and side["params"]["params_version"] == pa.PARAMS_VERSION
    assert side["stats"]["n_frames"] == sf.info(str(out)).frames == 88200
    assert set(side["stats"]) >= {"lufs_in", "gain_db", "correction_db", "lufs_out", "lufs_residual",
                                  "true_peak_dbtp", "peak_limited", "silent"}

    r2 = pa.process_one(job)
    assert r2["skipped"] and r2["lufs_out"] == side["stats"]["lufs_out"]

    r3 = pa.process_one({**job, "target_lufs": -20.0})  # different parameters -> regenerated
    assert not r3["skipped"]
    assert json.loads(pa.sidecar_path(out).read_text())["params"]["target_lufs"] == -20.0

    pa.sidecar_path(out).unlink()  # WAV without a sidecar is not trusted
    assert not pa.process_one(job)["skipped"]


@needs_ffmpeg
def test_process_one_reports_ffmpeg_error(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"garbage")
    r = pa.process_one({"track_id": "b", "audio_path": str(bad), "sr": 44100, "out_path": str(tmp_path / "b.wav"),
                        "subtype": "FLOAT", "target_lufs": -16.0, "ceiling_dbtp": -1.0, "resampler": "swr"})
    assert r["error"].startswith("RuntimeError: ffmpeg exited")
    assert not (tmp_path / "b.wav").exists()
