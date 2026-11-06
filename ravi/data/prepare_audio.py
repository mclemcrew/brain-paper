"""Decode, resample and loudness-normalise the corpus to WAV.

Usage::

    python -m ravi.data.prepare_audio --manifest manifests/tracks.parquet \
        --out-root prepared --sr 44100 --sr 48000 [--limit 5] [--workers 8]

For every manifest row and every target sample rate:

1. ``ffmpeg -map 0:a:0`` decodes only the audio stream (the MP3s carry an
   embedded cover-art video stream), downmix/upmix to stereo, resample with
   ``aresample`` (swr, long filter; ``--resampler soxr`` if your ffmpeg has it)
   and pipe float32 PCM to Python.
2. Integrated loudness (ITU-R BS.1770-4, pyloudnorm) is measured and a single
   gain brings it to ``--target-lufs`` (default -16 LUFS; the masters are ~-11 LUFS with peaks over 0 dBTP, so -14 is unreachable with gain alone for ~23% of tracks).
3. True peak is estimated by 4x polyphase oversampling; if it exceeds
   ``--ceiling-dbtp`` (default -1 dBTP) the gain is reduced until it does not
   (simple gain reduction, no limiter), so such tracks end up quieter than the
   target. That is recorded in the sidecar.
4. Loudness is re-measured on the gained (float32) signal and, for tracks that
   were not peak-limited, one correction gain ``target - measured`` is applied
   (itself capped by the true-peak ceiling). Gain is linear, so the residual is
   only float rounding plus the shift of the BS.1770 relative gate; it is
   recorded as ``lufs_residual = lufs_out - target``. On corpus tracks that
   were not peak-limited the first-pass residual was already ~1e-10 LU (the
   correction is a safeguard, not a material change); for peak-limited
   tracks the residual is the intended shortfall below the target.
5. Written atomically to ``{out_root}/{sr}/{track_id}.wav`` (float32 by default
   or ``--subtype PCM_24``) together with ``{track_id}.wav.json``: the
   processing parameters (:data:`PARAMS_VERSION`, sample rate, subtype,
   target, ceiling, resampler, source file name and size) and the stats. An
   existing WAV is skipped only if its sidecar's parameters equal the current
   ones; otherwise it is regenerated.

Per-track results also go to the table ``manifests/prepared_audio.parquet``
(one row per track_id, columns suffixed ``_44k``/``_48k``), merged with any
earlier runs. The MP3s remain the source of truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

SR_SUFFIX = {44100: "44k", 48000: "48k"}
#: Bump when the processing chain changes so stale WAVs are regenerated.
PARAMS_VERSION = 2


def sr_suffix(sr: int) -> str:
    return SR_SUFFIX.get(sr, f"{sr // 1000}k")


def decode(path: str, sr: int, resampler: str = "swr") -> np.ndarray:
    """Decode the first audio stream of ``path`` to float32 stereo at ``sr``. Shape (n, 2)."""
    if resampler == "swr":
        af = f"aresample={sr}:resampler=swr:filter_size=128:phase_shift=10:cutoff=0.97"
    elif resampler == "soxr":
        af = f"aresample={sr}:resampler=soxr:precision=28"
    else:
        raise ValueError(resampler)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
        "-i", path, "-map", "0:a:0", "-vn",
        "-af", af, "-ac", "2", "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode} decoding {path}: {stderr[-2000:]}")
    raw = proc.stdout
    if len(raw) % 8:
        raise RuntimeError(f"ffmpeg returned a partial frame for {path}: {stderr[-2000:]}")
    audio = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg produced no audio for {path}: {stderr[-2000:]}")
    return audio


def true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float:
    up = resample_poly(audio.astype(np.float64), oversample, 1, axis=0)
    peak = float(np.max(np.abs(up)))
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def normalise(audio: np.ndarray, sr: int, target_lufs: float, ceiling_dbtp: float) -> tuple[np.ndarray, dict]:
    meter = pyln.Meter(sr)
    lufs_in = float(meter.integrated_loudness(audio.astype(np.float64)))
    if not math.isfinite(lufs_in):
        # Silence (or too short to gate). Leave untouched and say so.
        return audio, {"lufs_in": lufs_in, "gain_db": 0.0, "correction_db": 0.0, "lufs_out": lufs_in,
                       "lufs_residual": float("nan"), "true_peak_dbtp": true_peak_dbtp(audio),
                       "peak_limited": False, "silent": True}
    tp_in = true_peak_dbtp(audio)
    gain_db = target_lufs - lufs_in
    tp_after = tp_in + gain_db
    limited = tp_after > ceiling_dbtp
    if limited:
        gain_db -= tp_after - ceiling_dbtp
    src = audio.astype(np.float64)
    out = (src * 10 ** (gain_db / 20)).astype(np.float32)
    lufs_first = float(meter.integrated_loudness(out.astype(np.float64)))
    correction_db = 0.0
    if not limited and math.isfinite(lufs_first):
        # One correction step; never push the true peak over the ceiling.
        correction_db = min(target_lufs - lufs_first, ceiling_dbtp - (tp_in + gain_db))
        if correction_db != 0.0:
            gain_db += correction_db
            out = (src * 10 ** (gain_db / 20)).astype(np.float32)
    lufs_out = float(meter.integrated_loudness(out.astype(np.float64))) if correction_db else lufs_first
    return out, {
        "lufs_in": lufs_in,
        "gain_db": gain_db,
        "correction_db": correction_db,
        "lufs_out": lufs_out,
        "lufs_residual": lufs_out - target_lufs,
        "true_peak_dbtp": true_peak_dbtp(out),
        "peak_limited": bool(limited),
        "silent": False,
    }


def sidecar_path(out_path: str | Path) -> Path:
    p = Path(out_path)
    return p.with_name(p.name + ".json")


def job_params(job: dict) -> dict:
    """Everything that determines the bytes of the output WAV."""
    src = Path(job["audio_path"])
    return {
        "params_version": PARAMS_VERSION,
        "sr": int(job["sr"]),
        "subtype": job["subtype"],
        "target_lufs": float(job["target_lufs"]),
        "ceiling_dbtp": float(job["ceiling_dbtp"]),
        "resampler": job["resampler"],
        "source_name": src.name,
        "source_bytes": src.stat().st_size,
    }


def _read_sidecar(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def process_one(job: dict) -> dict:
    """Worker: one (track, sr). Returns a flat result dict (never raises)."""
    out_path = Path(job["out_path"])
    side = sidecar_path(out_path)
    res = {"track_id": job["track_id"], "sr": job["sr"], "prepared_path": str(out_path), "error": None}
    try:
        params = job_params(job)
        old = _read_sidecar(side) if out_path.exists() else None
        if old is not None and old.get("params") == params:
            info = sf.info(str(out_path))
            if info.frames == old.get("stats", {}).get("n_frames"):
                res.update(skipped=True, **old["stats"])
                return res
        audio = decode(job["audio_path"], job["sr"], job["resampler"])
        out, stats = normalise(audio, job["sr"], job["target_lufs"], job["ceiling_dbtp"])
        stats["n_frames"] = int(out.shape[0])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
        sf.write(str(tmp), out, job["sr"], subtype=job["subtype"], format="WAV")
        os.replace(tmp, out_path)
        payload = {"track_id": job["track_id"], "params": params, "stats": stats}
        _write_atomic(side, (json.dumps(payload, indent=2, allow_nan=True) + "\n").encode())
        res.update(skipped=False, **stats)
    except Exception as e:  # noqa: BLE001 - collected and reported by the parent
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def merge_sidecar(sidecar: Path, results: list[dict]) -> pd.DataFrame:
    stat_cols = ["prepared_path", "n_frames", "lufs_in", "gain_db", "correction_db", "lufs_out", "lufs_residual",
                 "true_peak_dbtp", "peak_limited", "silent"]
    import pandas as pd

    old = pd.read_parquet(sidecar) if sidecar.exists() else pd.DataFrame({"track_id": pd.Series(dtype=str)})
    new = old.set_index("track_id")
    for r in results:
        if r.get("error"):
            continue
        suf = sr_suffix(r["sr"])
        for c in stat_cols:
            if c not in r:
                continue  # skipped files keep whatever stats an earlier run recorded
            col = f"{c}_{suf}"
            if col not in new.columns:
                new[col] = pd.Series(dtype=object)
            new.loc[r["track_id"], col] = r[c]
    new = new.reset_index().rename(columns={"index": "track_id"})
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(sidecar, index=False)
    return new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Decode + loudness-normalise corpus audio to WAV.")
    ap.add_argument("--manifest", type=Path, default=Path("manifests/tracks.parquet"))
    ap.add_argument("--out-root", type=Path, default=Path("prepared"))
    ap.add_argument("--sidecar", type=Path, default=Path("manifests/prepared_audio.parquet"))
    ap.add_argument("--sr", type=int, action="append", choices=[44100, 48000],
                    help="target sample rate(s); default both 44100 (SA3) and 48000 (ACE-Step)")
    ap.add_argument("--subtype", default="FLOAT", choices=["FLOAT", "PCM_24"])
    ap.add_argument("--target-lufs", type=float, default=-16.0)
    ap.add_argument("--ceiling-dbtp", type=float, default=-1.0)
    ap.add_argument("--resampler", default="swr", choices=["swr", "soxr"])
    ap.add_argument("--audio-root", type=Path, default=None,
                    help="directory holding the MP3s if it differs from the manifest's audio_path")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest rows")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    srs = args.sr or [44100, 48000]
    import pandas as pd

    df = pd.read_parquet(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    jobs = []
    for row in df.itertuples(index=False):
        src = str(args.audio_root / row.audio_filename) if args.audio_root else row.audio_path
        for sr in srs:
            jobs.append({
                "track_id": row.track_id, "audio_path": src, "sr": sr,
                "out_path": str(args.out_root / str(sr) / f"{row.track_id}.wav"),
                "subtype": args.subtype, "target_lufs": args.target_lufs,
                "ceiling_dbtp": args.ceiling_dbtp, "resampler": args.resampler,
            })

    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="prepare"):
            results.append(f.result())

    errors = [r for r in results if r["error"]]
    merge_sidecar(args.sidecar, results)
    done = [r for r in results if not r["error"] and not r.get("skipped")]
    limited = sum(bool(r.get("peak_limited")) for r in done)
    print(f"jobs {len(jobs)}  written {len(done)}  skipped {sum(bool(r.get('skipped')) for r in results)}  "
          f"peak-limited {limited}  errors {len(errors)}")
    for r in errors:
        print(f"ERROR {r['track_id']} @ {r['sr']}: {r['error']}")
    print(f"sidecar {args.sidecar}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
