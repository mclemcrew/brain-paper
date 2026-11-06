"""Atomic project storage. Originals are immutable; every edit creates a new asset."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_blocks(stream, frames=65536):
    """Stop at decoded EOF, even when a compressed header overstates its length."""
    while True:
        block = stream.read(frames, dtype="float32", always_2d=True)
        if not len(block): break
        yield block

def waveform(path: Path, bins=240) -> dict:
    with sf.SoundFile(path) as f:
        if not f.frames or f.channels > 2:
            raise ValueError("Use a non-empty mono or stereo audio file.")
        width = max(1, int(np.ceil(f.frames / bins)))
        peaks = []
        for _ in range(bins):
            # Bounded allocations even for hour-long files.
            remaining, peak = width, 0.0
            while remaining:
                block = f.read(min(remaining, 65536), dtype="float32", always_2d=True)
                if not len(block): break
                if not np.isfinite(block).all(): raise ValueError("Audio contains invalid samples.")
                peak = max(peak, float(np.max(np.abs(block))))
                remaining -= len(block)
            peaks.append(round(peak, 5))
        return {"duration": f.frames / f.samplerate, "sample_rate": f.samplerate,
                "channels": f.channels, "peaks": peaks}

def family_name(name: str) -> str:
    # A suggestion, explicitly editable by the composer before splitting.
    return re.sub(r"[\W_]+", " ", re.sub(
        r"(?i)[ _\-]+(?:\[?alt\]?|mix|version|ver|v|take)(?:[ _\-]*\d+|\b).*$", "", Path(name).stem
    )).strip().lower() or Path(name).stem.lower()

class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def project_dir(self, pid: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", pid): raise ValueError("Invalid project identifier")
        return self.root / "projects" / pid

    def get(self, pid: str) -> dict:
        return json.loads((self.project_dir(pid) / "project.json").read_text())

    def save(self, p: dict):
        p["updated"] = time.time()
        atomic_json(self.project_dir(p["id"]) / "project.json", p)

    def list(self):
        return sorted([json.loads(p.read_text()) for p in self.root.glob("projects/*/project.json")],
                      key=lambda p: p["updated"], reverse=True)

    def create(self, name: str) -> dict:
        if not name.strip(): raise ValueError("Give this collection a name.")
        p = {"id": uuid.uuid4().hex, "name": name.strip()[:120], "description": "",
             "tracks": [], "assets": [], "selected_asset": None, "selection_history": [],
             "profile": "16gb", "model_id": "sa3-medium", "created": time.time()}
        self.save(p)
        return p

    def asset_path(self, pid: str, relative: str) -> Path:
        base = self.project_dir(pid).resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base): raise ValueError("Asset must belong to this project")
        return path

    def import_file(self, pid: str, source: Path, name: str | None = None) -> dict:
        name = Path(name or source.name).name
        info = waveform(source)
        sha = digest(source)
        with self.lock:
            p = self.get(pid)
            if any(t["sha256"] == sha for t in p["tracks"]):
                return {"project": p, "duplicate": True}
            rel = f"originals/{sha}{source.suffix.lower()}"
            dst = self.asset_path(pid, rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dst)
            original_rel = rel
            rel = self.working_copy(pid, rel)
            info = waveform(self.asset_path(pid, rel))
            tags = {}
            try:
                import mutagen
                audio = mutagen.File(source, easy=True)
                if audio and audio.tags:
                    tags = {k: ", ".join(map(str, audio.tags.get(k, []))) for k in ("genre", "bpm", "title")}
            except Exception:
                pass
            caption = ", ".join(filter(None, [tags.get("genre"), f"{tags['bpm']} bpm" if tags.get("bpm") else ""]))
            track = {"id": sha[:24], "name": name, "path": rel, "sha256": sha,
                     "original_path": original_rel,
                     "family": family_name(name), "split": "train", "caption": caption,
                     "bpm": tags.get("bpm", ""), "metadata_source": "File tags" if caption else "Not described yet", **info}
            p["tracks"].append(track)
            self.save(p)
            return {"project": p, "duplicate": False}

    def working_copy(self, pid: str, relative: str) -> str:
        if Path(relative).suffix.lower() not in (".mp3", ".ogg"):
            return relative
        rel = f"decoded/{Path(relative).stem}.wav"
        dst = self.asset_path(pid, rel)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + '.' + uuid.uuid4().hex + '.tmp.wav')
            try:
                with sf.SoundFile(self.asset_path(pid, relative)) as src:
                    if src.channels not in (1, 2): raise ValueError("Use mono or stereo audio.")
                    with sf.SoundFile(tmp, "w", samplerate=src.samplerate, channels=src.channels, subtype="FLOAT") as out:
                        for block in read_blocks(src):
                            if not np.isfinite(block).all(): raise ValueError("Audio contains invalid samples.")
                            out.write(block)
                waveform(tmp)  # Reject empty or invalid decodes before publishing.
                tmp.replace(dst)
            finally:
                tmp.unlink(missing_ok=True)
        return rel

    def ensure_working_audio(self, pid: str) -> dict:
        """Upgrade pilot collections imported before canonical MP3 decoding."""
        with self.lock:
            p = self.get(pid)
            changed = False
            for t in p["tracks"]:
                rel = self.working_copy(pid, t["path"])
                if rel != t["path"]:
                    t["original_path"] = t["path"]
                    t["path"] = rel
                    t.update(waveform(self.asset_path(pid, rel)))
                    changed = True
            if changed: self.save(p)
            return p

    def update(self, pid: str, changes: dict) -> dict:
        with self.lock:
            p = self.get(pid)
            for key in ("name", "description", "profile", "model_id"):
                if key in changes: p[key] = str(changes[key])[:2000]
            if "onboarding" in changes:
                value = changes["onboarding"]
                if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                        or type(value.get("step")) is not int or not 0 <= value["step"] <= 4):
                    raise ValueError("Invalid onboarding progress")
                p["onboarding"] = {"enabled": value["enabled"], "step": value["step"]}
            from .catalog import PROFILES, MODELS
            if p["profile"] not in PROFILES or p["model_id"] not in MODELS:
                raise ValueError("Unknown hardware profile or model")
            if "track" in changes:
                edit = changes["track"]
                track = next(t for t in p["tracks"] if t["id"] == edit["id"])
                from .metadata import edit_track
                edit_track(track, edit)
                # All alternates follow the edited family's split.
                for t in p["tracks"]:
                    if t["family"] == track["family"]: t["split"] = track["split"]
            if "append_captions" in changes:
                from .metadata import edit_track
                ids = set(changes["append_captions"]["ids"])
                text = str(changes["append_captions"]["text"]).strip()[:2000]
                if not text or not ids.issubset({t["id"] for t in p["tracks"]}): raise ValueError("Choose recordings and text to append.")
                for track in p["tracks"]:
                    if track["id"] in ids: edit_track(track, {"caption": (track["caption"] + " " + text).strip()})
            if "review_tracks" in changes:
                from .metadata import edit_track
                ids = set(changes["review_tracks"])
                if not ids.issubset({t["id"] for t in p["tracks"]}): raise ValueError("Unknown recording")
                for track in p["tracks"]:
                    if track["id"] in ids: edit_track(track, {"reviewed": True})
            if "authorize_training" in changes:
                from .metadata import edit_track
                ids = set(changes["authorize_training"])
                if not ids.issubset({t["id"] for t in p["tracks"]}): raise ValueError("Unknown recording")
                for track in p["tracks"]:
                    if track["id"] in ids: edit_track(track, {"training_authorized": True})
            self.save(p)
            return p

    def apply_suggestions(self, pid, suggestions):
        with self.lock:
            p = self.get(pid)
            for suggestion in suggestions:
                track = next(t for t in p["tracks"] if t["id"] == suggestion["id"])
                if suggestion.get("sha256") != track["sha256"]: continue
                track["suggestion"] = suggestion
                track["suggested_bpm"] = suggestion.get("suggested_bpm")
            self.save(p)

    def split(self, pid: str) -> dict:
        with self.lock:
            p = self.get(pid)
            families = sorted({t["family"] for t in p["tracks"] if t["split"] != "exclude"},
                              key=lambda x: hashlib.sha256(x.encode()).hexdigest())
            if len(families) < 3: raise ValueError("Import at least three different piece families to reserve validation audio.")
            held = set(families[:max(1, round(len(families) * .2))])
            for t in p["tracks"]:
                if t["split"] != "exclude": t["split"] = "validation" if t["family"] in held else "train"
            self.save(p)
            return p

    def add_asset(self, pid: str, asset: dict):
        with self.lock:
            p = self.get(pid)
            p["assets"].append(asset)
            self.save(p)

    def select(self, pid: str, aid: str | None, undo=False):
        with self.lock:
            p = self.get(pid)
            if undo:
                if p["selection_history"]: p["selected_asset"] = p["selection_history"].pop()
            else:
                if not any(a["id"] == aid for a in p["assets"]): raise ValueError("Unknown version")
                p["selection_history"].append(p["selected_asset"])
                p["selected_asset"] = aid
            self.save(p)
            return p
