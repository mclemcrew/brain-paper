"""One short-lived worker per stage: release Metal allocations between stages.

All workers except the explicit model installer refuse network connections.
Upstream scripts are pinned and called with constructed argument lists, never shells.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .catalog import MODELS, PROFILES, UPSTREAM_COMMIT, crop_latents, weight_files
from .store import Store, atomic_json, digest, waveform, read_blocks
from .recipe import training_recipe

def offline():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      DO_NOT_TRACK="1", GRADIO_ANALYTICS_ENABLED="False")
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise PermissionError("Network access is disabled in composer workers. Install missing models in Setup.")
    sys.addaudithook(audit)

def announce(message, **extra):
    print("RAVI " + json.dumps({"message": message, **extra}), flush=True)

def runtime(spec):
    root = Path(spec["runtime"])
    sys.path[:0] = [str(root / "scripts"), str(root)]
    weights = importlib.import_module("weights")
    def local_only(relative, verbose=True):
        path = Path(spec["model_dir"]) / Path(relative).name
        if not path.is_file(): raise FileNotFoundError(f"Missing {path.name}. Install the model pack in Setup.")
        return path
    weights.ensure_local = local_only
    weights.is_present = lambda relative: (Path(spec["model_dir"]) / Path(relative).name).is_file()
    import mlx.core as mx
    import atexit
    prior_peaks = []
    reset_peak = mx.reset_peak_memory
    def preserve_peak():
        prior_peaks.append(mx.get_peak_memory())
        reset_peak()
    mx.reset_peak_memory = preserve_peak
    atexit.register(lambda: announce("MLX memory", peak_mlx_gb=round(max([mx.get_peak_memory(), *prior_peaks]) / 2**30, 3)))
    limit = int(PROFILES[spec["profile"]]["memory_gb"] * 2**30)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(min(256 * 2**20, limit // 10))
    if hasattr(mx, "set_wired_limit"):
        setter = mx.set_wired_limit
        mx.set_wired_limit = lambda n: setter(min(n, limit))
    return mx

def invoke(module, args):
    sys.argv = [module, *map(str, args)]
    implementation = importlib.import_module(module)
    if module == "sa3_mlx":
        from .adapter_guard import require_complete_adapter_merge
        implementation.save_wav = write_render
        merge_module = importlib.import_module("models.defs.lora_merge")
        with require_complete_adapter_merge(merge_module):
            implementation.main()
    else:
        implementation.main()

def write_render(path, audio, sample_rate=44100):
    """Prevent the upstream PCM16 writer from hard-clipping decoder output."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or not audio.size:
        raise ValueError("Expected non-empty channel-first audio")
    if not np.isfinite(audio).all(): raise ValueError("Render contains non-finite samples")
    peak = float(np.max(np.abs(audio)))
    gain = min(1.0, .98 / peak) if peak else 1.0
    sf.write(path, (audio * gain).T, sample_rate, subtype="FLOAT")
    atomic_json(Path(path).with_suffix(".levels.json"), {
        "raw_sample_peak": peak, "applied_gain": gain,
        "gain_db": 20 * math.log10(gain), "export": "float WAV; sample-peak headroom, no hard clipping"})

def install(spec):
    # Plain HTTP supports predictable timeout/retry and resumable downloads.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(spec["model_dir"])
    root.mkdir(parents=True, exist_ok=True)
    repo = "stabilityai/stable-audio-3-optimized"
    # Resolve once, so all files in a pack belong to the same revision.
    revision = HfApi().model_info(repo).sha
    records = {}
    for i, name in enumerate(weight_files(spec["model_id"])):
        announce(f"Downloading {name}", progress=i / len(weight_files(spec["model_id"])))
        downloaded = Path(hf_hub_download(repo, f"MLX/{name}", revision=revision,
                                        cache_dir=str(root / ".download-cache")))
        dst = root / name
        tmp = dst.with_suffix(".partial")
        # A real file, not a symlink into a disposable external cache.
        try:
            if tmp.exists(): tmp.unlink()
            os.link(downloaded.resolve(), tmp)
        except OSError:
            shutil.copyfile(downloaded, tmp)
        tmp.replace(dst)
        records[name] = {"sha256": digest(dst), "bytes": dst.stat().st_size}
    atomic_json(root / f"{spec['model_id']}.json", {"repo": repo, "revision": revision, "files": records,
                                                       "runtime_commit": UPSTREAM_COMMIT})

def resample(audio, original_sr, target_sr=44100):
    if original_sr == target_sr: return audio
    from scipy.signal import resample_poly
    g = math.gcd(original_sr, target_sr)
    return resample_poly(audio, target_sr // g, original_sr // g, axis=0).astype(np.float32)

def prepare(spec):
    from ravi.data.prepare_audio import normalise
    from .metadata import crop_plan
    from .recipe import TARGET_LUFS, TRUE_PEAK_CEILING_DBTP
    project = spec["project"]
    store = Store(Path(spec["root"]))
    output = Path(spec["prepared_dir"])
    output.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[spec["profile"]]
    tracks = [t for t in project["tracks"] if t["split"] == "train"]
    count, levels = 0, {}
    for index, t in enumerate(tracks):
        announce(f"Preparing {t['name']}", progress=index / max(1, len(tracks)))
        plan = crop_plan(t, profile["example_max_seconds"])
        if not plan["clips"]: continue
        audio, sr = sf.read(store.asset_path(project["id"], t["path"]), dtype="float32", always_2d=True)
        if len(audio) < max(c["end_frame"] for c in plan["clips"]):
            raise ValueError("Source duration changed; import this recording again.")
        a = resample(audio, sr)
        if a.shape[1] == 1: a = np.repeat(a, 2, axis=1)
        # One gain for the whole recording (as the cloud preparation does), so fades,
        # quiet intros and endings keep their level relative to the rest of the piece.
        a, stats = normalise(a, 44100, TARGET_LUFS, TRUE_PEAK_CEILING_DBTP)
        if stats["silent"]: continue
        levels[t["id"]] = {k: stats[k] for k in ("lufs_in", "gain_db", "lufs_out", "true_peak_dbtp", "peak_limited")}
        for part, clip in enumerate(plan["clips"]):
            block = a[round(clip["start_frame"] * 44100 / sr):round(clip["end_frame"] * 44100 / sr)]
            if not len(block) or float(np.max(np.abs(block))) < 1e-4: continue  # digital silence
            stem = f"{t['id']}_{part:05d}"
            sf.write(output / f"{stem}.wav", block, 44100, subtype="FLOAT")
            atomic_json(output / f"{stem}.json", {"text": clip["caption"], "prompt": clip["caption"],
                "source_track": t["id"], "source_start": clip["start"], "source_end": clip["end"],
                "section_id": clip["section_id"], "section_label": clip["label"],
                "metadata_revision": t.get("metadata_revision", 0), "caption_format": clip["caption_format"], "bpm": t["bpm"]})
            count += 1
    if not count: raise ValueError("No audible training clips of at least one second were found.")
    atomic_json(output / "preparation.json", {"clips": count,
                "normalization": f"one gain per track to {TARGET_LUFS:g} LUFS, true peak <= {TRUE_PEAK_CEILING_DBTP:g} dBTP",
                "example_max_seconds": profile["example_max_seconds"], "crop_seconds": profile["crop_seconds"],
                "track_levels": levels, "runtime_commit": UPSTREAM_COMMIT})

def encode(spec):
    runtime(spec)
    invoke("pre_encode_mlx", ["--audio-dir", spec["prepared_dir"], "--output-dir", spec["latents_dir"],
        "--codec", MODELS[spec["model_id"]]["codec"], "--max-duration", PROFILES[spec["profile"]]["example_max_seconds"]])
    # Do not depend on which tags upstream's decoder preserves.
    for src in Path(spec["prepared_dir"]).glob("*.json"):
        if src.name == "preparation.json": continue
        sidecar = Path(spec["latents_dir"]) / src.name
        latent = sidecar.with_suffix(".npy")
        if not sidecar.exists() or not latent.exists():
            raise RuntimeError(f"Encoding did not complete for {src.stem}. Retry preparation before training.")
        encoded = np.load(latent, mmap_mode="r")
        if encoded.ndim != 2 or not np.isfinite(encoded).all():
            raise RuntimeError(f"Invalid cached latent: {src.stem}")
        if sidecar.exists():
            info = json.loads(sidecar.read_text())
            meta = json.loads(src.read_text())
            info.update({k: meta[k] for k in ("text", "prompt", "source_track", "source_start", "source_end", "section_id", "section_label", "metadata_revision", "bpm")})
            info["caption_format"] = meta.get("caption_format", "prose-v0")
            atomic_json(sidecar, info)
    encode_silence(spec)
    atomic_json(Path(spec["latents_dir"]) / "ready.json", {"cache_key": spec["cache_key"], "completed": time.time()})

def encode_silence(spec):
    """Encoded digital silence the trainer uses to pad examples shorter than its window
    (PreEncodedLatentDataset reads <latents>/silence.npy; it is never served as a sample)."""
    frames = crop_latents(spec["profile"])
    work = Path(spec["job_dir"]) / "silence"
    (work / "audio").mkdir(parents=True, exist_ok=True)
    sf.write(work / "audio" / "silence.wav", np.zeros(((frames + 1) * 4096, 2), np.float32), 44100, subtype="FLOAT")
    invoke("pre_encode_mlx", ["--audio-dir", work / "audio", "--output-dir", work / "latents",
        "--codec", MODELS[spec["model_id"]]["codec"], "--max-duration", PROFILES[spec["profile"]]["crop_seconds"] + 1])
    silence = np.load(work / "latents" / "silence.npy")
    if silence.ndim != 2 or silence.shape[1] < frames or not np.isfinite(silence).all():
        raise RuntimeError("Could not encode the silence padding latent.")
    np.save(Path(spec["latents_dir"]) / "silence.npy", silence[:, :frames].astype(np.float32))

def checkpoint_every(steps):
    return min(250, steps) if steps <= 2000 else 500

def train(spec):
    p = PROFILES[spec["profile"]]
    recipe = training_recipe(spec["profile"])
    if (spec.get("recipe") is not None or spec.get("adapter_path")) and spec.get("recipe") != recipe:
        raise ValueError("Training recipe differs from this worker. Submit a new run.")
    trainer = Path(spec["runtime"]) / "scripts" / "lora_train_mlx.py"
    if "--gen-mask" not in trainer.read_text():
        raise RuntimeError("The bundled trainer is missing the rAvI generation-mask patch. "
                           "Run scripts/apply_runtime_patches.py and rebuild.")
    runtime(spec)
    prompt_config = Path(spec["job_dir"]) / "prompt.json"
    atomic_json(prompt_config, {"use_tags": True, "tag_keys": ["prompt"], "hide_tag_names": True,
                               "shuffle": False, "use_paths": False,
                               "trigger": recipe["trigger"], "trigger_pct": 100})
    args = ["--dit", MODELS[spec["model_id"]]["dit"], "--latents-dir", spec["latents_dir"],
        "--lr", recipe["learning_rate"], "--name", spec["id"], "--save-dir", spec["checkpoint_root"],
        "--batch-size", recipe["batch_size"], "--max-steps", spec["steps"], "--checkpoint-every", checkpoint_every(spec["steps"]),
        "--rank", recipe["rank"], "--alpha", recipe["alpha"], "--adapter-type", recipe["adapter_type"], "--seed", spec["seed"],
        "--latent-crop-length", recipe["crop_latents"],
        "--random-crop" if recipe["random_crop"] else "--no-random-crop",
        "--no-conditioner-lora", "--exclude", ",".join(recipe["exclude"]), "--grad-checkpoint", "--no-t5-cache",
        "--beta1", recipe["beta1"], "--beta2", recipe["beta2"], "--eps", recipe["eps"],
        "--weight-decay", recipe["weight_decay"], "--lr-scheduler", recipe["lr_scheduler"],
        "--timestep-sampler", recipe["timestep_sampler"], "--dist-shift", recipe["dist_shift"],
        "--use-effective-length" if recipe["use_effective_length"] else "--no-use-effective-length",
        "--gen-mask", recipe["gen_mask"], "--cfg-dropout-prob", recipe["cfg_dropout_prob"],
        "--gradient-clip-val", recipe["gradient_clip_val"], "--prompt-config", prompt_config, "--demo-every", 0]
    if spec.get("adapter_path"):
        args += ["--lora-ckpt-path", spec["adapter_path"]]
    announce("Training; checkpoints are retained. Continuation restores adapter weights, not optimizer state.")
    invoke("lora_train_mlx", args)

def infer(spec):
    runtime(spec)
    m = MODELS[spec["model_id"]]
    output = Path(spec["job_dir"]) / "render.wav"
    prompt = spec["prompt"]
    if spec.get("adapter_path") and spec.get("trigger") and float(spec.get("strength", 1)) > 0:
        prompt = f"{spec['trigger']}, {prompt}" if prompt else spec["trigger"]
    args = ["--dit", m["dit"], "--decoder", m["codec"], "--prompt", prompt,
            "--seconds", spec["seconds"], "--seed", spec["seed"], "--steps", 8,
            "--cfg", 1.0, "--apg", 1.0, "--init-noise-level", 1.0, "--dit-dtype", "fp16",
            "--free-models", "--out", output]
    if spec.get("adapter_path"): args += ["--lora", spec["adapter_path"], "--lora-strength", spec.get("strength", 1)]
    if spec["kind"] == "edit":
        store = Store(Path(spec["root"]))
        source = store.asset_path(spec["project_id"], spec["source_path"])
        with sf.SoundFile(source) as f:
            sr, channels = f.samplerate, f.channels
            total = len(f) / sr
            start, end = spec["start"], spec["end"]
            duration = min(PROFILES[spec["profile"]]["max_seconds"], total)
            offset = max(0, min(start - (duration - (end - start)) / 2, total - duration))
            offset = round(offset * sr) / sr
            f.seek(round(offset * sr))
            context = f.read(round(duration * sr), dtype="float32", always_2d=True)
        ref = Path(spec["job_dir"]) / "context.wav"
        ctx = resample(context, sr)
        if channels == 1: ctx = np.repeat(ctx, 2, axis=1)
        sf.write(ref, ctx, 44100, subtype="PCM_16")
        args[args.index("--seconds") + 1] = len(ctx) / 44100
        args += ["--init-audio", ref, "--inpaint-range", f"{start-offset},{end-offset}"]
    atomic_json(Path(spec["job_dir"]) / "effective-inference.json", {
        "inference_steps": 8, "cfg": 1.0, "apg": 1.0, "sigma_max": 1.0, "dit_dtype": "fp16",
        "seconds": args[args.index("--seconds") + 1], "seed": spec["seed"],
        "adapter_path": spec.get("adapter_path"), "strength": spec.get("strength", 1),
        "prompt_sent": prompt, "trigger": spec.get("trigger") if prompt != spec["prompt"] else None})
    announce("Rendering on this Mac")
    invoke("sa3_mlx", args)
    if spec["kind"] == "edit":
        from ravi.eval.masks import splice
        generated, gen_sr = sf.read(output, dtype="float32", always_2d=True)
        generated = resample(generated, gen_sr, sr)
        if channels == 1: generated = generated.mean(axis=1, keepdims=True)
        lo, hi = round(start * sr), round(end * sr)
        off = round(offset * sr)
        fill = generated[lo-off:hi-off]
        if len(fill) != hi-lo: raise ValueError("Render length does not cover the requested edit")
        with sf.SoundFile(source) as src:
            src.seek(lo)
            original = src.read(hi-lo, dtype="float32", always_2d=True)
        # Fade only inside the selected region. A trailing sentinel keeps both fades.
        padded = np.concatenate([original, np.zeros((1, channels), np.float32)])
        fill = splice(padded, fill, 0, hi-lo, round(.05*sr))[:-1]
        final = Path(spec["job_dir"]) / "edited.wav"
        with sf.SoundFile(source) as src, sf.SoundFile(final, "w", samplerate=sr, channels=channels, subtype="FLOAT") as dst:
            pos = 0
            for block in read_blocks(src):
                a, b = max(lo, pos), min(hi, pos+len(block))
                if a < b: block[a-pos:b-pos] = fill[a-lo:b-lo]
                dst.write(block)
                pos += len(block)
        output = final
    publish_audio(spec, output)

def publish_audio(spec, source):
    store = Store(Path(spec["root"]))
    rel = f"renders/{spec['id']}.wav"
    target = store.asset_path(spec["project_id"], rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    asset = {"id": spec["id"], "path": rel, "name": spec.get("prompt", "Variation")[:80],
             "kind": spec["kind"], "created": time.time(), "model_id": spec["model_id"],
             "adapter_id": spec.get("adapter_id"), "seed": spec["seed"], "prompt": spec["prompt"],
             "source_path": spec.get("source_path"), "region": [spec.get("start"), spec.get("end")],
             "runtime_commit": UPSTREAM_COMMIT, **waveform(target)}
    levels = Path(spec["job_dir"]) / "render.levels.json"
    if levels.exists(): asset["render_levels"] = json.loads(levels.read_text())
    atomic_json(Path(spec["job_dir"]) / "result.json", asset)

def transcribe(spec):
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from basic_pitch.inference import predict, Model
    import basic_pitch
    model_paths = list(Path(basic_pitch.__file__).parent.rglob("*.onnx"))
    if not model_paths: raise RuntimeError("The bundled Basic Pitch ONNX model is missing.")
    store = Store(Path(spec["root"]))
    source = store.asset_path(spec["project_id"], spec["source_path"])
    low, high = spec.get("min_note", 36), spec.get("max_note", 96)
    _, midi, _ = predict(str(source), Model(str(model_paths[0])),
                         minimum_frequency=440 * 2 ** ((low-69)/12), maximum_frequency=440 * 2 ** ((high-69)/12))
    import pretty_midi
    timed_midi = pretty_midi.PrettyMIDI(initial_tempo=spec.get("bpm", 120))
    timed_midi.instruments = midi.instruments
    midi = timed_midi
    grid = spec.get("quantize", False)
    if grid:
        tick = 60 / spec["bpm"] / 4
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.start = round(note.start / tick) * tick
                note.end = max(note.start + tick, round(note.end / tick) * tick)
    rel = f"renders/{spec['id']}.mid"
    target = store.asset_path(spec["project_id"], rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(target))
    notes = [{"pitch": int(n.pitch), "start": float(n.start), "end": float(n.end), "velocity": int(n.velocity)}
             for i in midi.instruments for n in i.notes]
    atomic_json(Path(spec["job_dir"]) / "result.json", {"id": spec["id"], "path": rel,
        "name": "MIDI draft", "kind": "transcribe", "source_path": spec["source_path"],
        "created": time.time(), "notes": notes, "duration": float(midi.get_end_time())})

def analyze(spec):
    from .analysis import Tagger, propose, analysis_key
    store = Store(Path(spec["root"]))
    results = []
    tagger = None
    tracks = spec["project"]["tracks"]
    for index, t in enumerate(tracks):
        key = analysis_key(t)
        cache = store.project_dir(spec["project_id"]) / "analysis" / f"{key}.json"
        announce(f"Analyzing {index+1}/{len(tracks)}: {t['name']}", progress=index/max(1,len(tracks)))
        if cache.exists() and not spec.get("force_analysis"):
            result = json.loads(cache.read_text())
        else:
            if tagger is None: tagger = Tagger(store.root)
            result = propose(store.asset_path(spec["project_id"], t["path"]), tagger)
            result.update(id=t["id"], sha256=t["sha256"], created=time.time(), key=key)
            atomic_json(cache, result)  # Survives cancellation; next run skips completed files.
        results.append(result)
    atomic_json(Path(spec["job_dir"]) / "analysis.json", results)

def install_tagger(spec):
    from .analysis import install_pack
    install_pack(spec["root"])

def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    if spec["stage"] not in ("install", "install_tagger"): offline()
    Path(spec["job_dir"]).mkdir(parents=True, exist_ok=True)
    os.chdir(spec["job_dir"])
    started = time.monotonic()
    {"install": install, "prepare": prepare, "encode": encode, "train": train,
     "infer": infer, "transcribe": transcribe, "analyze": analyze, "install_tagger": install_tagger}[spec["stage"]](spec)
    announce("Stage complete", elapsed_seconds=time.monotonic()-started)

if __name__ == "__main__": main()
