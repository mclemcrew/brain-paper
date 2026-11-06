"""Persistent, serial jobs. The web server never owns a model allocation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .catalog import MODELS, PROFILES, UPSTREAM_COMMIT, catalog, hardware
from .store import Store, atomic_json

def number(value, lo, hi, label):
    n = float(value)
    if not math.isfinite(n) or not lo <= n <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}.")
    return n


def worker_progress(tail):
    """Read complete worker announcements from a bounded log tail."""
    latest = {}
    # The final line may still be being written by the worker.
    for line in tail.splitlines(keepends=True):
        if not line.endswith('\n') or not line.startswith('RAVI '): continue
        try:
            event = json.loads(line[5:])
            progress = event.get('progress')
            if isinstance(progress, (int, float)) and math.isfinite(progress) and 0 <= progress <= 1:
                latest = {'progress': progress, 'message': str(event.get('message', ''))}
        except (ValueError, TypeError, AttributeError):
            continue
    return latest

class Jobs:
    def __init__(self, store: Store, runtime: Path | None, worker_binary: Path | None = None):
        self.store, self.runtime = store, runtime
        self.worker_binary = worker_binary
        self.root = store.root / "jobs"
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.active = None
        self.proc = None
        for path in self.root.glob("*/job.json"):
            j = json.loads(path.read_text())
            if j["status"] in ("running", "queued", "cancelling"):
                j.update(status="interrupted", message="The app stopped. Saved checkpoints are available for continuation.")
                atomic_json(path, j)

    def list(self):
        jobs = [json.loads(p.read_text()) for p in self.root.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def checkpoints(self, pid):
        result = []
        for j in self.list():
            if j.get("project_id") != pid or j["kind"] not in ("train", "calibrate"): continue
            if j["status"] in ("running", "queued", "cancelling"): continue
            for p in (self.root / j["id"] / "checkpoints").rglob("*.safetensors"):
                if p.stat().st_size < 8: continue
                try:
                    from safetensors import safe_open
                    with safe_open(p, framework="numpy") as checkpoint:
                        if not list(checkpoint.keys()): continue
                except Exception:
                    continue  # Never offer a partially written checkpoint after a crash.
                step = re.search(r"step=(\d+)", p.name)
                rel = str(p.relative_to(self.store.root))
                result.append({"id": hashlib.sha256(rel.encode()).hexdigest()[:24], "path": rel,
                    "name": f"{j.get('model_id', '')} · step {step[1] if step else '?'}",
                    "step": int(step[1]) if step else 0, "job_id": j["id"], "model_id": j["model_id"],
                    "cache_key": j.get("cache_key"), "profile": j.get("profile"), "created": p.stat().st_mtime})
        return sorted(result, key=lambda c: c["created"], reverse=True)

    def submit(self, request: dict):
        with self.lock:
            if self.active: raise ValueError("Another job is running. Finish or cancel it before starting another.")
            kind = request.get("kind")
            if kind not in ("install", "prepare", "train", "calibrate", "generate", "edit", "transcribe", "analyze", "install_tagger"):
                raise ValueError("Unknown job type")
            p = self.store.ensure_working_audio(request["project_id"])
            if kind in ("prepare", "train", "calibrate"):
                from .metadata import readiness
                state = readiness(p)
                if not state["ready"]:
                    raise ValueError(" ".join(b["message"] for b in state["blockers"]))
            model_id = request.get("model_id", p["model_id"])
            profile = p["profile"]
            if model_id not in MODELS: raise ValueError("This model is not available in the pilot")
            if kind not in ("install", "transcribe", "analyze", "install_tagger"):
                c = next(m for m in catalog(self.store.root / "models", self.runtime) if m["id"] == model_id)
                if not c["ready"]: raise ValueError(c["reason"])
                if profile == "8gb" and model_id != "sa3-small":
                    raise ValueError("Use Stable Audio 3 Small with the 8 GB workload profile.")
                if profile == "16gb" and hardware()["memory_gb"] < 14:
                    raise ValueError("Use the lighter 8 GB workload on this computer.")
                if profile == "32gb" and hardware()["memory_gb"] < 28:
                    raise ValueError("Use a lighter workload on this computer.")
                required = 2 if profile == "8gb" else 4
                if hardware()["available_gb"] < required:
                    raise ValueError("Close other memory-intensive applications before starting this job.")
            if kind in ("train", "calibrate") and self.runtime:
                trainer = Path(self.runtime) / "scripts" / "lora_train_mlx.py"
                if trainer.is_file() and "--gen-mask" not in trainer.read_text():
                    raise ValueError("This build's trainer is missing the generation-mask fix. Rebuild the app.")
            if kind in ("install", "install_tagger") and request.get("license_accepted") is not True:
                raise ValueError("Review and accept the model terms before downloading.")
            ident = uuid.uuid4().hex
            job_dir = self.root / ident
            job_dir.mkdir()
            from .recipe import training_recipe
            recipe = training_recipe(profile)
            snapshot = {"tracks": [{k: t.get(k) for k in ("id", "sha256", "caption", "split", "family", "bpm", "sections", "musical_details", "reviewed", "training_authorization", "metadata_revision")} for t in p["tracks"]],
                        "description": p["description"], "model": model_id, "profile": profile,
                        "caption_format": "musical-details-v1", "recipe": recipe, "upstream": UPSTREAM_COMMIT}
            key = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
            cache = self.store.project_dir(p["id"]) / "cache" / key
            spec = {"id": ident, "kind": kind, "root": str(self.store.root), "project_id": p["id"],
                "project": p, "profile": profile, "model_id": model_id, "job_dir": str(job_dir),
                "runtime": str(self.runtime or ""), "model_dir": str(self.store.root / "models"),
                "cache_key": key, "prepared_dir": str(cache / "audio"), "latents_dir": str(cache / "latents"),
                "checkpoint_root": str(job_dir / "checkpoints"),
                "force_analysis": request.get("force_analysis") is True,
                "seed": int(number(request.get("seed", 42), 0, 2**31-1, "Seed")),
                "steps": 20 if kind == "calibrate" else int(number(request.get("steps", 1000), 1, 10000, "Steps")),
                "prompt": str(request.get("prompt", ""))[:4000],
                "seconds": number(request.get("seconds", 15), 1, PROFILES[profile]["max_seconds"], "Duration"),
                "strength": number(request.get("strength", 1), 0, 1.5, "Personalization strength")}
            if kind in ("generate", "edit"):
                spec["steps"] = spec["inference_steps"] = 8
            if kind in ("prepare", "train", "calibrate"):
                train = [t for t in p["tracks"] if t["split"] == "train"]
                # Retain every recipe choice; this is explicitly not the paper's mask recipe.
                spec["recipe"] = recipe
                free = __import__("shutil").disk_usage(self.store.root).free
                estimate = sum(t["duration"] for t in train) * 44100 * 2 * 4 + 1024**3
                if free < estimate: raise ValueError("Not enough disk space for prepared audio and checkpoints.")
            if request.get("adapter_id"):
                cp = next((c for c in self.checkpoints(p["id"]) if c["id"] == request["adapter_id"]), None)
                if not cp or cp["model_id"] != model_id: raise ValueError("Choose an adapter trained for this model.")
                if kind in ("train", "calibrate") and (cp["cache_key"] != key or cp["profile"] != profile):
                    raise ValueError("The collection or recipe changed. Start a new training run.")
                if kind in ("train", "calibrate") and spec["steps"] <= cp["step"]:
                    raise ValueError("Continuation target must exceed the saved checkpoint step.")
                spec.update(adapter_id=cp["id"], adapter_path=str(self.store.root / cp["path"]))
                if kind in ("generate", "edit"):
                    # Adapters trained with a trigger token get it in their prompts too.
                    trained = json.loads((self.root / cp["job_id"] / "job.json").read_text())
                    spec["trigger"] = (trained.get("recipe") or {}).get("trigger")
            if kind in ("edit", "transcribe"):
                item = next((a for a in p["tracks"] + p["assets"] if a["id"] == request.get("source_id")), None)
                if not item or item["path"].endswith(".mid"): raise ValueError("Choose source audio.")
                spec["source_path"] = item["path"]
                if kind == "edit":
                    start = number(request.get("start", 0), 0, item["duration"], "Selection start")
                    end = number(request.get("end", 0), 0, item["duration"], "Selection end")
                    if end <= start or end-start > PROFILES[profile]["max_seconds"]-2:
                        raise ValueError("Select a non-empty passage with room for surrounding context.")
                    spec.update(start=start, end=end)
                else:
                    spec.update(min_note=int(number(request.get("min_note", 36), 0, 127, "Lowest MIDI note")),
                        max_note=int(number(request.get("max_note", 96), 0, 127, "Highest MIDI note")),
                        quantize=request.get("quantize") is True,
                        bpm=number(request.get("bpm", 120), 20, 300, "Tempo"))
                    if spec["min_note"] >= spec["max_note"]: raise ValueError("Lowest note must be below highest note.")
                    if item["duration"] > 300: raise ValueError("Transcribe a passage of five minutes or less.")
            job = {k: spec[k] for k in ("id", "kind", "project_id", "model_id", "profile", "cache_key", "steps")}
            job.update(status="queued", message="Waiting for local worker", created=time.time(), progress=0,
                       elapsed_seconds=0, peak_rss_gb=0, recipe=spec.get("recipe"), checkpoints=[])
            if kind in ("generate", "edit"):
                job["inference_steps"] = 8
            atomic_json(job_dir / "metadata-snapshot.json", snapshot)
            atomic_json(job_dir / "request.json", spec)
            atomic_json(job_dir / "job.json", job)
            self.active = job
            # Credentials are transient, never persisted in a job or command line.
            threading.Thread(target=self._run, args=(spec, job, request.get("hf_token", "")), daemon=True).start()
            return job.copy()

    def cancel(self, jid):
        with self.lock:
            if not self.active or self.active["id"] != jid: raise ValueError("Job is no longer running")
            self.active.update(status="cancelling", message="Stopping; keeping saved checkpoints…")
            self._save()
            if self.proc and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)

    def _save(self):
        if self.active: atomic_json(self.root / self.active["id"] / "job.json", self.active)

    def _run(self, spec, job, token):
        import psutil
        started = time.monotonic()
        start_swap = psutil.swap_memory().used
        try:
            kind = spec["kind"]
            stages = {"install": ["install"], "install_tagger": ["install_tagger"], "prepare": ["prepare", "encode"],
                "train": ["prepare", "encode", "train"], "calibrate": ["prepare", "encode", "train"],
                "generate": ["infer"], "edit": ["infer"], "transcribe": ["transcribe"], "analyze": ["analyze"]}[kind]
            if (Path(spec["latents_dir"]) / "ready.json").exists():
                stages = [s for s in stages if s not in ("prepare", "encode")]
            with self.lock:
                if job["status"] == "cancelling":
                    job.update(status="cancelled", message="Stopped before processing began.")
                    self._save()
                    return
                job["status"] = "running"
            for stage_index, stage in enumerate(stages):
                with self.lock:
                    if job["status"] == "cancelling": break
                    job.update(stage=stage, progress=stage_index / len(stages), message={"prepare": "Preparing bounded audio clips", "encode": "Encoding audio once",
                        "train": "Training your personal model", "infer": "Rendering audio", "install": "Downloading model files",
                        "transcribe": "Finding notes locally", "analyze": "Drafting local tags and sections", "install_tagger": "Downloading local tagger"}[stage])
                    self._save()
                    stage_spec = {**spec, "stage": stage}
                    stage_path = Path(spec["job_dir"]) / f"{stage}.json"
                    atomic_json(stage_path, stage_spec)
                    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
                    env["NUMBA_CACHE_DIR"] = str(self.store.root / "cache" / "numba")
                    if stage == "install" and token: env["HF_TOKEN"] = token
                    elif stage not in ("install", "install_tagger"):
                        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"): env.pop(key, None)
                    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
                    cmd = [sys.executable, "--worker", str(stage_path)] if getattr(sys, "frozen", False) else [sys.executable, "-m", "ravi.local.worker", str(stage_path)]
                    if self.worker_binary:
                        cmd = [str(self.worker_binary), "--worker", str(stage_path)]
                    if stage not in ("install", "install_tagger") and sys.platform == "darwin":
                        if not Path("/usr/bin/sandbox-exec").exists():
                            raise RuntimeError("The macOS network isolation helper is unavailable; processing was not started.")
                        cmd = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", *cmd]
                    log_path = Path(spec["job_dir"]) / f"{stage}.log"
                    log = log_path.open("w")
                    self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                stage_start = time.monotonic()
                cancel_at = None
                while self.proc.poll() is None:
                    time.sleep(.5)
                    with self.lock:
                        elapsed = time.monotonic()-started
                        job["elapsed_seconds"] = round(elapsed, 1)
                        try:
                            process = psutil.Process(self.proc.pid)
                            rss = sum(x.memory_info().rss for x in [process, *process.children(recursive=True)] if x.is_running())
                            job["peak_rss_gb"] = max(job["peak_rss_gb"], round(rss / 2**30, 3))
                        except psutil.Error: pass
                        job["swap_growth_gb"] = round(max(0, psutil.swap_memory().used-start_swap) / 2**30, 3)
                        with log_path.open("rb") as lf:
                            lf.seek(max(0, log_path.stat().st_size - 16000))
                            tail = lf.read().decode(errors="replace")
                        steps = re.findall(r"Step\s+(\d+)", tail)
                        event = worker_progress(tail)
                        if event:
                            job['progress'] = min(.99, (stage_index + event['progress']) / len(stages))
                            if event['message']: job['message'] = event['message']
                        if stage == "train" and steps:
                            job["step"] = int(steps[-1])
                            job["progress"] = min(.99, (stage_index + job["step"] / spec["steps"]) / len(stages))
                        if elapsed > 12*3600 and job["status"] != "cancelling":
                            job.update(status="cancelling", message="12-hour limit reached; saving progress.")
                            self.proc.send_signal(signal.SIGINT)
                        if job["status"] == "cancelling":
                            cancel_at = cancel_at or time.monotonic()
                            if time.monotonic()-cancel_at > 20: self.proc.kill()
                        self._save()
                log.close()
                if job["status"] == "cancelling": break
                if self.proc.returncode != 0:
                    with log_path.open("rb") as failed_log:
                        failed_log.seek(max(0, log_path.stat().st_size - 2400))
                        tail = failed_log.read().decode(errors="replace")
                    tail = re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", tail)
                    raise RuntimeError(tail or "Worker exited without a result")
                with log_path.open("rb") as completed_log:
                    completed_log.seek(max(0, log_path.stat().st_size - 16000))
                    for line in completed_log.read().decode(errors="replace").splitlines():
                        if line.startswith("RAVI "):
                            try:
                                event = json.loads(line[5:])
                                job["peak_mlx_gb"] = max(job.get("peak_mlx_gb", 0), event.get("peak_mlx_gb", 0))
                            except (ValueError, TypeError): pass
                job.setdefault("stage_seconds", {})[stage] = round(time.monotonic()-stage_start, 2)
            result = Path(spec["job_dir"]) / "result.json"
            with self.lock:
                if job["status"] == "cancelling": job.update(status="cancelled", message="Stopped. Saved checkpoints are retained.")
                else:
                    if result.exists(): self.store.add_asset(spec["project_id"], json.loads(result.read_text()))
                    if kind == "analyze":
                        self.store.apply_suggestions(spec["project_id"], json.loads((Path(spec["job_dir"]) / "analysis.json").read_text()))
                    job.update(status="completed", progress=1, message="Complete · saved on this Mac")
                job["elapsed_seconds"] = round(time.monotonic()-started, 2)
                job["hardware"] = hardware()
                job["validation_note"] = "Measured on the recorded hardware; a workload profile does not simulate an 8 GB machine."
                self._save()
        except Exception as exc:
            with self.lock:
                job.update(status="failed", message=str(exc)[-2400:])
                self._save()
        finally:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try: self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            with self.lock: self.active, self.proc = None, None

    def close(self):
        with self.lock:
            if self.active: self.cancel(self.active["id"])
            process = self.proc
        if process and process.poll() is None:
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
