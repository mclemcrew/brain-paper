"""Run ACE-Step 1.5 LoRA training (upstream ``training_v2`` "fixed" trainer) for one caption arm.

Entry point of the ``ravi-acestep-train`` image
(``ENTRYPOINT ["python", "-m", "ravi.train.acestep_lora"]``). Works the same under
``docker run`` on EC2 and as a SageMaker Training Job. Conventions follow
:mod:`ravi.train.sa3_lora`. Every upstream reference below is to ACE-Step-1.5 at
``ca1e85fe9430179831e6bc6be790c332190a3866``.

Configuration, lowest to highest precedence:
  1. built-in defaults (:data:`DEFAULTS`)
  2. ``--config FILE`` (YAML or JSON), or a ``config`` key in hyperparameters.json
  3. SageMaker ``/opt/ml/input/config/hyperparameters.json`` (values may be JSON-encoded strings)
  4. ``--set key=value`` and ``--s3-out URI`` on the command line

Input: a directory in ACE-Step's per-file layout (``{name}.wav`` + ``{name}.caption.txt``
[+ ``{name}.lyrics.txt``] + ``{name}.json``), as written by
:mod:`ravi.captions.make_acestep_dataset`.

What a run does:
  * preflight (:func:`scan_data`): every clip has a caption that starts with the trigger;
    no ``*.csv`` (``csv_metadata.py:10-78`` would override captions) and no legacy
    ``{name}.txt`` (read as *lyrics*, ``audio_io.py:72-85``) in the data dir; unique stems
    (tensors are named ``{stem}.pt``, ``training_v2/preprocess.py:237,279``); no clip longer
    than ``max_duration`` (silently truncated, ``preprocess_audio.py:17-19``); CUDA visible;
    flash-attn importable;
  * model weights: ``ACE-Step/Ace-Step1.5`` at a pinned HF revision into ``weights_dir``
    (:data:`MAIN_COMPONENTS`; plus ``ACE-Step/acestep-v15-base`` for the base variant);
  * dataset JSON: upstream ``DatasetBuilder.scan_directory`` (the Gradio "Load Data" step,
    ``dataset_builder_modules/scan.py:15-126``) reads the per-file layout; its samples are
    written in ``save_dataset``'s format (``serialization.py:26-29``) and checked against
    the files;
  * preprocessing: ``python -m acestep.training_v2.cli.train_fixed --preprocess
    --dataset-json ...`` (``train_fixed.py:134-195``; two passes: VAE + Qwen3 text encoder,
    then the DiT condition encoder, ``training_v2/preprocess.py:118-143``). It returns 0
    even when files fail (``train_fixed.py:188-195``), so the wrapper counts the ``.pt``
    files. The tensors are cached in ``work_dir`` by data fingerprint and variant;
  * training: ``python -m acestep.training_v2.cli.train_fixed --plain --yes ...`` in a child
    process (through :func:`child_train`, which fixes resume, see below). ``run_fixed``
    also returns 0 after a ``[FAIL]`` update (``trainer_fixed.py:170-172`` turns exceptions
    into messages), so success means ``final/adapter_model.safetensors`` exists and the log
    has no ``[FAIL]`` line. Without ``--yes`` it would ask "Start training?" and, on EOF,
    exit 0 without training (``ui/config_panel.py:266-291``);
  * every ``output_dir/checkpoints/epoch_{N}_loss_{L}/`` (``trainer_fixed.py:514-517``) is
    copied to ``/opt/ml/checkpoints/{prefix}_epoch{N:04d}/`` with a ``ravi_meta.json`` and,
    if ``s3_out`` is set, uploaded;
  * the final PEFT adapter (``save_pretrained`` of the PeftModel decoder into
    ``output_dir/final``, ``trainer_fixed.py:551-554`` + ``trainer_helpers.py:145-184``)
    is copied to ``/opt/ml/model/adapter/`` (``adapter_config.json`` +
    ``adapter_model.safetensors``), loadable with ``AceStepHandler.load_lora(dir)``
    (``core/generation/handler/lora/lifecycle.py:191-271``);
  * ``run_meta.json`` (argv, ACE-Step commit, weights revision, epoch plan, versions,
    attention implementation, loss curve parsed from the log) in the output and model dirs.

Steps and epochs. ``training_v2`` has no step cap: it trains ``--epochs`` full epochs and
``steps_per_epoch = ceil(len(loader) / grad_accum)`` (``trainer_fixed.py:325``) with
``len(loader) = ceil(n_clips / batch)`` (``shuffle=True, drop_last=False``, ``training/data_module.py:280-297``)
and the remainder flushed as one optimizer step at each epoch end
(``trainer_fixed.py:476-501``). So

    steps_per_epoch = ceil(ceil(n / batch) / accum) = ceil(n / (batch * accum))
    epochs          = ceil(steps / steps_per_epoch)      # rounded UP
    actual_steps    = epochs * steps_per_epoch           # >= steps, < steps + steps_per_epoch

e.g. 580 clips, batch 1, accum 4: 145 steps/epoch; ``steps: 6000`` -> 42 epochs = 6090 steps.
The cosine schedule is computed over ``actual_steps`` (``trainer_fixed.py:326-336``).
``--save-every`` is in epochs: ``save_every_epochs``, or
``ceil(checkpoint_every_steps / steps_per_epoch)``.

Resume. A checkpoint directory holds the adapter plus ``training_state.pt`` (optimizer,
scheduler, epoch, global step, ``trainer_helpers.py:187-234``), and ``--resume-from``
restores them (``trainer_fixed.py:384-398``). Upstream reloads the LoRA weights with
``decoder.load_state_dict(state_dict, strict=False)`` (``trainer_helpers.py:395-398``), but
``save_pretrained`` strips the adapter name from the keys (``...lora_A.weight`` vs the
PeftModel's ``...lora_A.default.weight``), so with ``strict=False`` nothing is loaded and the
run silently continues from freshly initialised LoRA weights with a restored optimizer.
Any exception during resume also becomes "starting fresh" (``trainer_fixed.py:393-398``).
:func:`child_train` therefore wraps upstream's resume: it reloads the adapter with
``peft.set_peft_model_state_dict``, verifies every tensor, and exits non-zero on any
problem. Upstream checkpoints and ``--resume-from`` need paths under the process's
working directory (``training/path_safety.py:35,53-91``), so the trainer runs with
``cwd = work_dir`` and the checkpoint is staged into ``work_dir/resume/`` first.

Checkpoints are named ``{arm}_s{seed}_h{hash6}_epoch{N:04d}``; ``hash`` is the sha256 of
:func:`identity_fields` (variant, weights revision, LoRA and optimiser settings, steps,
epochs, data fingerprint). Modes: ``none`` (default off SageMaker), ``auto`` (default on
SageMaker: newest checkpoint of this identity, local or S3), ``publish`` (only publish a
finished run), or an explicit directory / ``s3://`` prefix (identity still checked). A
resumed run uses seed ``seed + completed_epochs`` so the noise stream does not replay.
The data order of the remaining epochs differs from an uninterrupted run.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ravi.train.sa3_lora import (
    S3Sink,
    _load_file,
    _set_dotted,
    _unwrap_json_string,
    _write_failure,
    pin_single_gpu,
    split_s3,
)

# --------------------------------------------------------------------------- constants

ACESTEP_DEFAULT_ROOT = "/opt/acestep"
SM_HP_PATH = "/opt/ml/input/config/hyperparameters.json"
TRAIN_MODULE = "acestep.training_v2.cli.train_fixed"
MAIN_WEIGHTS_REPO = "ACE-Step/Ace-Step1.5"  # model_downloader.py:287
MAIN_WEIGHTS_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"  # HF, 2026-02-03
#: Everything inference's check_main_model_exists requires (model_downloader.py:307-312, 386-402).
#: Training needs only the DiT, vae/ and the text encoder (model_loader.py:283,309); the 5 Hz LM
#: is fetched too because AceStepHandler.initialize_service otherwise auto-downloads the
#: whole main repo at its *current* revision (init_service_downloads.py:31-36).
MAIN_COMPONENTS = ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
#: Variant -> checkpoint subdir (cli/args.py:18-22), weights source, and the inference
#: schedule the variant is served with (docs/sidestep/Using Your Adapter.md:93-99).
VARIANTS: dict[str, dict[str, Any]] = {
    "turbo": {"dir": "acestep-v15-turbo", "repo": MAIN_WEIGHTS_REPO, "revision": MAIN_WEIGHTS_REVISION,
              "shift": 3.0, "inference_steps": 8},
    "base": {"dir": "acestep-v15-base", "repo": "ACE-Step/acestep-v15-base",
             "revision": "e432212fec32b8965a14ffa57ae653438d6abd14", "shift": 1.0, "inference_steps": 50},
}
#: Audio the upstream directory scan picks up (dataset_builder_modules/models.py:11).
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".opus")
ACESTEP_SR = 48000
WEIGHTS_MARKER = ".ravi_weights.json"
CKPT_FILES = ("adapter_config.json", "adapter_model.safetensors", "training_state.pt")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
META_NAME = "ravi_meta.json"

DEFAULTS: dict[str, Any] = {
    "arm": "unnamed",
    "run_id": None,
    "data_dir": None,  # falls back to SM_CHANNEL_TRAIN
    "smoke": False,
    "model_variant": "turbo",  # turbo | base
    "acestep_root": None,  # ACESTEP_ROOT env or /opt/acestep
    "weights_dir": None,  # ACESTEP_CHECKPOINTS_DIR env or {acestep_root}/checkpoints
    "weights_revision": None,  # default: VARIANTS[variant]["revision"]
    "download_weights": True,
    # LoRA (cli/args.py:285-291)
    "rank": 64,
    "alpha": 128,
    "dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_type": "both",
    # optimisation (cli/args.py:263-278, 325-328)
    "lr": 1e-4,
    "batch_size": 1,
    "grad_accum": 4,
    "steps": 6000,  # target optimizer steps; epochs are derived (module docstring)
    "save_every_epochs": None,  # default: ceil(checkpoint_every_steps / steps_per_epoch)
    "checkpoint_every_steps": 500,
    "warmup_steps": 100,
    "scheduler": "cosine",
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "cfg_ratio": 0.15,
    "gradient_checkpointing": True,
    "shift": None,  # default per variant; metadata only for training (Using Your Adapter.md:101)
    "num_inference_steps": None,  # default per variant
    "precision": "bf16",
    "seed": 42,
    "num_workers": 4,
    "log_every": 10,
    "max_duration": 240.0,
    "trigger": "rvkr",
    "resume": "default",  # none | auto | publish | <dir or s3:// prefix>; default auto on SageMaker
    "s3_out": None,
    "aws_region": None,
    "require_cuda": True,
    "require_flash_attn": True,
    "work_dir": None,
    "checkpoint_dir": None,
    "model_dir": None,
    "output_data_dir": None,
    "poll_seconds": 15,
}

STRING_KEYS = frozenset({
    "arm", "run_id", "data_dir", "model_variant", "acestep_root", "weights_dir", "weights_revision",
    "attention_type", "scheduler", "optimizer", "precision", "trigger", "resume", "s3_out",
    "aws_region", "work_dir", "checkpoint_dir", "model_dir", "output_data_dir", "config",
})

# --------------------------------------------------------------------------- config


def _coerce(value: Any, key: str | None = None) -> Any:
    if key in STRING_KEYS:
        if isinstance(value, str):
            return _unwrap_json_string(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key in ("arm", "run_id"):
            return str(value)
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null", "~"):
        return None
    return s


def _merge(cfg: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        value = _coerce(value, key)
        if "." in key:
            _set_dotted(cfg, key, value)
        else:
            cfg[key] = value


def load_config(config_path: str | None = None, hp_path: str | None = None,
                sets: list[str] | None = None, s3_out: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    hp: dict = {}
    hp_file = Path(hp_path or os.environ.get("RAVI_HP_PATH", SM_HP_PATH))
    if hp_file.is_file():
        hp = json.loads(hp_file.read_text())
    cfg_file = config_path or (_coerce(hp.get("config"), "config") if hp.get("config") else None)
    if cfg_file:
        _merge(cfg, _load_file(cfg_file))
    _merge(cfg, {k: v for k, v in hp.items() if k != "config" and not k.startswith(("sagemaker_", "_"))})
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        _merge(cfg, {k.strip(): v})
    if s3_out:
        cfg["s3_out"] = s3_out
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg["_sources"] = {"config": str(cfg_file) if cfg_file else None,
                       "hyperparameters": str(hp_file) if hp else None, "sets": list(sets or [])}
    return cfg


def resolve(cfg: dict) -> dict:
    """Fill environment-dependent defaults and validate."""
    cfg = copy.deepcopy(cfg)
    cfg["data_dir"] = cfg["data_dir"] or os.environ.get("SM_CHANNEL_TRAIN")
    if not cfg["data_dir"]:
        raise ValueError("data_dir (or SM_CHANNEL_TRAIN) is required")
    if cfg["model_variant"] not in VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(VARIANTS)}, got {cfg['model_variant']!r}")
    var = VARIANTS[cfg["model_variant"]]
    for key in ("rank", "alpha", "batch_size", "grad_accum", "steps", "checkpoint_every_steps", "log_every"):
        v = cfg[key]
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"{key} must be an integer (cli/args.py declares type=int), got {v}")
        cfg[key] = int(v)
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("warmup_steps", "num_workers", "seed"):
        cfg[key] = int(cfg[key])
        if cfg[key] < 0:
            raise ValueError(f"{key} must be >= 0")
    if cfg["save_every_epochs"] is not None:
        cfg["save_every_epochs"] = int(cfg["save_every_epochs"])
        if cfg["save_every_epochs"] <= 0:
            raise ValueError("save_every_epochs must be positive")
    for key in ("lr", "dropout", "weight_decay", "max_grad_norm", "cfg_ratio", "max_duration"):
        cfg[key] = float(cfg[key])
    if not 0.0 <= cfg["cfg_ratio"] < 1.0:
        raise ValueError("cfg_ratio must be in [0, 1)")
    if not 0.0 <= cfg["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    tm = cfg["target_modules"]
    if isinstance(tm, str):
        tm = [t for t in re.split(r"[\s,]+", tm) if t]
    if not tm or not all(isinstance(t, str) and t for t in tm):
        raise ValueError("target_modules must be a non-empty list of module names")
    cfg["target_modules"] = list(tm)
    choices = {  # cli/args.py:203,275,276,291
        "attention_type": ("self", "cross", "both"),
        "precision": ("auto", "bf16", "fp16", "fp32"),
        "optimizer": ("adamw", "adamw8bit", "adafactor", "prodigy"),
        "scheduler": ("cosine", "cosine_restarts", "linear", "constant", "constant_with_warmup"),
    }
    for key, allowed in choices.items():
        if cfg[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {cfg[key]!r}")
    cfg["shift"] = float(cfg["shift"] if cfg["shift"] is not None else var["shift"])
    cfg["num_inference_steps"] = int(cfg["num_inference_steps"] if cfg["num_inference_steps"] is not None
                                     else var["inference_steps"])
    cfg["weights_revision"] = cfg["weights_revision"] or var["revision"]
    cfg["gradient_checkpointing"] = bool(cfg["gradient_checkpointing"])
    cfg["smoke"] = bool(cfg["smoke"])

    on_sm = Path(SM_HP_PATH).is_file() or "SM_TRAINING_ENV" in os.environ
    mode = cfg["resume"]
    if mode in (None, False) or str(mode).strip().lower() in ("none", "false", "off", "no", ""):
        mode = "none"
    elif str(mode).strip().lower() == "default":
        mode = "auto" if on_sm else "none"
    cfg["resume"] = str(mode).strip()
    cfg["acestep_root"] = cfg["acestep_root"] or os.environ.get("ACESTEP_ROOT", ACESTEP_DEFAULT_ROOT)
    cfg["weights_dir"] = (cfg["weights_dir"] or os.environ.get("ACESTEP_CHECKPOINTS_DIR")
                          or str(Path(cfg["acestep_root"]) / "checkpoints"))
    if not cfg["run_id"]:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cfg["run_id"] = os.environ.get("TRAINING_JOB_NAME") or f"{cfg['arm']}-{stamp}"
    cfg["checkpoint_dir"] = cfg["checkpoint_dir"] or "/opt/ml/checkpoints"
    cfg["model_dir"] = cfg["model_dir"] or os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    cfg["output_data_dir"] = cfg["output_data_dir"] or os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    cfg["work_dir"] = cfg["work_dir"] or os.environ.get("RAVI_WORK_DIR") or "/opt/ml/ravi_work"
    # realpath, not just absolute: path_safety.py:23-29 compares symlink-resolved paths.
    for key in ("data_dir", "acestep_root", "weights_dir", "checkpoint_dir", "model_dir",
                "output_data_dir", "work_dir"):
        cfg[key] = os.path.realpath(os.path.expanduser(str(cfg[key])))
    if cfg["work_dir"] == os.sep:
        raise ValueError("work_dir must not be /: it is the trainer's path-safety root (path_safety.py:87)")
    if cfg["resume"] not in ("none", "auto", "publish") and not cfg["resume"].startswith("s3://"):
        cfg["resume"] = os.path.realpath(os.path.expanduser(cfg["resume"]))
    cfg["_on_sagemaker"] = on_sm
    return cfg


# --------------------------------------------------------------------------- epochs


def steps_per_epoch(n_clips: int, batch_size: int, grad_accum: int) -> int:
    """Optimizer steps per epoch of ``training_v2`` (trainer_fixed.py:325, 476-501)."""
    if n_clips <= 0 or batch_size <= 0 or grad_accum <= 0:
        raise ValueError("n_clips, batch_size and grad_accum must be positive")
    return math.ceil(math.ceil(n_clips / batch_size) / grad_accum)


def plan_epochs(target_steps: int, n_clips: int, batch_size: int, grad_accum: int,
                save_every_epochs: int | None = None, checkpoint_every_steps: int = 500) -> dict:
    """Epoch count for at least ``target_steps`` optimizer steps (rounded up)."""
    spe = steps_per_epoch(n_clips, batch_size, grad_accum)
    epochs = max(1, math.ceil(target_steps / spe))
    save_every = int(save_every_epochs) if save_every_epochs else max(1, math.ceil(checkpoint_every_steps / spe))
    return {
        "n_clips": n_clips,
        "batches_per_epoch": math.ceil(n_clips / batch_size),
        "steps_per_epoch": spe,
        "target_steps": target_steps,
        "epochs": epochs,
        "actual_steps": epochs * spe,
        "overshoot_steps": epochs * spe - target_steps,
        "save_every_epochs": save_every,
        "checkpoint_epochs": [e for e in range(save_every, epochs + 1, save_every)],
        "clips_seen": epochs * n_clips,
    }


# --------------------------------------------------------------------------- data


def _read_text(p: Path) -> str | None:
    """Like upstream ``_read_text_file`` (audio_io.py:10-27): stripped, None if absent/empty."""
    if not p.is_file():
        return None
    s = p.read_text(encoding="utf-8").strip()
    return s or None


def _audio_info(path: Path) -> tuple[int, int, int] | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return int(info.frames), int(info.samplerate), int(info.channels)
    except Exception:
        return None


def scan_data(cfg: dict) -> tuple[dict, list[dict]]:
    """Validate the per-file layout the way the upstream scanner and preprocessing see it.
    Returns (summary, one record per clip in upstream's order)."""
    root = Path(cfg["data_dir"])
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir {root} does not exist")
    csvs = sorted(p.name for p in root.iterdir() if p.suffix.lower() == ".csv")
    if csvs:
        raise ValueError(f"{root} contains {csvs}: upstream reads every CSV in the dataset dir and its caption "
                         "column overrides the captions (csv_metadata.py:10-78, scan.py:99-108)")
    audio = []
    for dirpath, _dirs, files in os.walk(root):  # scan.py:31-36 (recursive, all extensions)
        for name in files:
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                audio.append(Path(dirpath) / name)
    audio.sort(key=str)  # scan.py:44 sorts the joined paths
    trigger = str(cfg["trigger"])
    recs, problems, warnings = [], [], []
    stems: dict[str, str] = {}
    max_dur = float(cfg["max_duration"])
    for a in audio:
        base = a.with_suffix("")
        rel = str(a.relative_to(root))
        if a.stem in stems:
            problems.append(f"{rel}: stem {a.stem!r} also used by {stems[a.stem]} (tensors are named by stem)")
        stems[a.stem] = rel
        legacy = Path(str(base) + ".txt")
        if legacy.exists():
            problems.append(f"{legacy.name}: upstream reads {{name}}.txt as LYRICS (audio_io.py:72-85); remove it")
        caption = _read_text(Path(str(base) + ".caption.txt"))
        meta: dict = {}
        jp = Path(str(base) + ".json")
        if jp.is_file():
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                problems.append(f"{jp.name}: {e}")
            if not isinstance(meta, dict):
                problems.append(f"{jp.name}: not a JSON object")
                meta = {}
        if meta.get("caption"):  # scan.py:84-87: the JSON caption wins
            caption = str(meta["caption"]).strip()
        if not caption:
            problems.append(f"{rel}: no caption (.caption.txt or .json caption); upstream would mark it unlabeled")
        elif caption != trigger and not caption.startswith(trigger + ", "):
            problems.append(f"{rel}: caption does not start with the trigger {trigger!r}: {caption[:60]!r}")
        lyrics = _read_text(Path(str(base) + ".lyrics.txt"))
        info = _audio_info(a)
        if info is None:
            problems.append(f"{rel}: unreadable")
            continue
        frames, sr, ch = info
        seconds = frames / sr
        if seconds > max_dur + 1e-6:
            problems.append(f"{rel}: {seconds:.2f}s > max_duration {max_dur}s; upstream truncates silently "
                            "(preprocess_audio.py:17-19)")
        if sr != ACESTEP_SR:
            warnings.append(f"{rel}: {sr} Hz, upstream resamples to 48 kHz (preprocess_audio.py:8-10)")
        if ch > 2:
            warnings.append(f"{rel}: {ch} channels, upstream keeps the first 2 (preprocess_audio.py:14-15)")
        recs.append({
            "name": a.stem, "file": rel, "audio_path": str(a), "bytes": a.stat().st_size,
            "frames": frames, "sample_rate": sr, "channels": ch, "seconds": round(seconds, 6),
            "caption": caption, "lyrics": lyrics or "[Instrumental]", "has_lyrics_file": lyrics is not None,
            "json": {k: meta[k] for k in sorted(meta)},
        })
    if not audio:
        problems.append(f"no audio files ({', '.join(AUDIO_EXTS)}) under {root}")
    if problems:
        raise ValueError(f"{len(problems)} data problems in {root}:\n  " + "\n  ".join(problems[:30]))
    for w in warnings[:10]:
        print(f"[ravi] WARNING: {w}", flush=True)
    fp_payload = [[r["file"], r["bytes"], r["frames"], r["sample_rate"], r["caption"], r["lyrics"], r["json"]]
                  for r in recs]
    fingerprint = hashlib.sha256(json.dumps(fp_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    secs = sorted(r["seconds"] for r in recs)
    summary = {
        "data_dir": str(root),
        "n_clips": len(recs),
        "n_with_lyrics": sum(r["has_lyrics_file"] for r in recs),
        "n_with_bpm": sum(1 for r in recs if r["json"].get("bpm")),
        "n_not_48k": sum(1 for r in recs if r["sample_rate"] != ACESTEP_SR),
        "seconds_min": secs[0],
        "seconds_median": secs[len(secs) // 2],
        "seconds_max": secs[-1],
        "fingerprint": fingerprint,
        "n_warnings": len(warnings),
    }
    return summary, recs


def check_dataset_json(path: Path, recs: list[dict]) -> dict:
    """Compare upstream's scan (written to ``path``) with :func:`scan_data`."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = doc.get("samples", [])
    by_path = {os.path.realpath(s.get("audio_path", "")): s for s in samples}
    problems = []
    if len(samples) != len(recs):
        problems.append(f"upstream scan found {len(samples)} samples, the wrapper {len(recs)}")
    for r in recs:
        s = by_path.get(os.path.realpath(r["audio_path"]))
        if s is None:
            problems.append(f"{r['file']}: missing from the upstream scan")
            continue
        if not s.get("labeled"):
            problems.append(f"{r['file']}: upstream marks it unlabeled")
        if (s.get("caption") or "").strip() != r["caption"]:
            problems.append(f"{r['file']}: caption {s.get('caption')!r} != {r['caption']!r}")
        if (s.get("lyrics") or "[Instrumental]").strip() != r["lyrics"]:
            problems.append(f"{r['file']}: lyrics {s.get('lyrics')!r} != {r['lyrics']!r}")
        if r["json"].get("bpm") and s.get("bpm") != r["json"]["bpm"]:
            problems.append(f"{r['file']}: bpm {s.get('bpm')!r} != {r['json']['bpm']!r}")
        if s.get("custom_tag"):
            problems.append(f"{r['file']}: unexpected custom_tag {s['custom_tag']!r} (would be prepended)")
    md = doc.get("metadata", {})
    if md.get("genre_ratio", 0):
        problems.append(f"genre_ratio {md['genre_ratio']} != 0 (genre would replace captions)")
    if problems:
        raise ValueError("upstream dataset scan disagrees with the files:\n  " + "\n  ".join(problems[:30]))
    return {"n_samples": len(samples), "metadata": md}


def build_dataset_json(data_dir: str, out_json: Path, name: str) -> dict:
    """Run upstream's ``DatasetBuilder.scan_directory`` (needs ACE-Step) and write its samples
    in ``save_dataset``'s format. ``save_dataset`` itself is not used because it only writes
    under the path-safety root (serialization.py:31-33), and the data dir may be read-only."""
    from acestep.training.dataset_builder import DatasetBuilder
    from acestep.training.path_safety import set_safe_root

    set_safe_root(data_dir)  # scan_directory validates every path against it (scan.py:17-20)
    builder = DatasetBuilder()
    samples, status = builder.scan_directory(data_dir)
    print(f"[ravi] upstream scan: {status}", flush=True)
    builder.metadata.name = name
    builder.metadata.num_samples = len(samples)
    doc = {"metadata": builder.metadata.to_dict(), "samples": [s.to_dict() for s in samples]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_name(out_json.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_json)
    return doc


def tensor_status(tensor_dir: Path, recs: list[dict]) -> dict:
    want = {r["name"] for r in recs}
    have = {p.name[: -len(".pt")] for p in tensor_dir.glob("*.pt") if not p.name.endswith(".tmp.pt")} \
        if tensor_dir.is_dir() else set()
    tmp = sorted(p.name for p in tensor_dir.glob("*.tmp.pt")) if tensor_dir.is_dir() else []
    return {"n_expected": len(want), "n_present": len(want & have), "missing": sorted(want - have)[:20],
            "extra": sorted(have - want)[:20], "leftover_tmp": tmp[:20],
            "complete": want == have and not tmp}


# --------------------------------------------------------------------------- weights


def weights_layout(cfg: dict) -> list[tuple[str, str, list[str] | None, Path]]:
    """(repo, revision, allow_patterns, local_dir) downloads that make ``weights_dir`` usable."""
    var = VARIANTS[cfg["model_variant"]]
    wd = Path(cfg["weights_dir"])
    main_rev = cfg["weights_revision"] if var["repo"] == MAIN_WEIGHTS_REPO else MAIN_WEIGHTS_REVISION
    out = [(MAIN_WEIGHTS_REPO, main_rev, [f"{c}/*" for c in MAIN_COMPONENTS], wd)]
    if var["repo"] != MAIN_WEIGHTS_REPO:
        out.append((var["repo"], cfg["weights_revision"], None, wd / var["dir"]))
    return out


def missing_weights(cfg: dict) -> list[str]:
    wd = Path(cfg["weights_dir"])
    vdir = wd / VARIANTS[cfg["model_variant"]]["dir"]
    need = [vdir / "config.json", vdir / "model.safetensors", vdir / "silence_latent.pt"]
    need += [wd / c / "config.json" for c in MAIN_COMPONENTS]
    return [str(p) for p in dict.fromkeys(need) if not p.is_file()]


def ensure_weights(cfg: dict) -> dict:
    """Make sure ``weights_dir`` has the variant at the pinned revision; download if allowed.

    ``.ravi_weights.json`` records ``{repo: revision}`` of what this wrapper downloaded. A
    directory filled by hand (no marker) is used as is and reported as unverified."""
    wd = Path(cfg["weights_dir"])
    marker = wd / WEIGHTS_MARKER
    recorded = json.loads(marker.read_text()) if marker.is_file() else {}
    plan = weights_layout(cfg)
    for repo, rev, _pat, _local in plan:
        if recorded.get(repo) not in (None, rev):
            raise ValueError(f"{wd} holds {repo}@{recorded[repo]}, config wants {rev}; use another weights_dir")
    missing = missing_weights(cfg)
    if missing:
        if not cfg["download_weights"]:
            raise FileNotFoundError(f"model weights missing and download_weights is false: {missing}")
        from huggingface_hub import snapshot_download

        for repo, rev, pat, local in plan:
            print(f"[ravi] downloading {repo}@{rev} {pat or '*'} -> {local}", flush=True)
            local.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo, revision=rev, local_dir=str(local), allow_patterns=pat)
            recorded[repo] = rev
            wd.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        missing = missing_weights(cfg)
        if missing:
            raise FileNotFoundError(f"model weights still missing after download: {missing}")
    vdir = wd / VARIANTS[cfg["model_variant"]]["dir"]
    files = {str(p.relative_to(wd)): p.stat().st_size for p in sorted(vdir.iterdir()) if p.is_file()}
    verified = all(recorded.get(repo) == rev for repo, rev, _p, _l in plan)
    if not verified:
        print(f"[ravi] WARNING: {wd} has no {WEIGHTS_MARKER} for the pinned revision(s); using it unverified",
              flush=True)
    return {"weights_dir": str(wd), "revisions": recorded, "verified_revision": verified, "variant_files": files}


# --------------------------------------------------------------------------- argv


def build_preprocess_argv(cfg: dict, dataset_json: Path, tensor_dir: Path) -> list[str]:
    """``train_fixed --preprocess`` (train_fixed.py:198-216; flags in cli/args.py)."""
    return [
        "--plain", "--yes",
        "--checkpoint-dir", str(cfg["weights_dir"]),
        "--model-variant", cfg["model_variant"],
        "--device", "cuda" if cfg["require_cuda"] else "auto",
        "--precision", cfg["precision"],
        "--preprocess",
        "--dataset-json", str(dataset_json),
        "--tensor-output", str(tensor_dir),
        "--max-duration", repr(float(cfg["max_duration"])),
    ]


def build_train_argv(cfg: dict, plan: dict, tensor_dir: Path, output_dir: Path, seed: int,
                     resume_from: str | None) -> list[str]:
    """``train_fixed`` training argv; every flag exists in cli/args.py (tested)."""
    argv = [
        "--plain", "--yes",
        "--checkpoint-dir", str(cfg["weights_dir"]),
        "--model-variant", cfg["model_variant"],
        "--device", "cuda" if cfg["require_cuda"] else "auto",
        "--precision", cfg["precision"],
        "--num-devices", "1",
        "--dataset-dir", str(tensor_dir),
        "--num-workers", str(cfg["num_workers"]),
        "--lr", repr(cfg["lr"]),
        "--batch-size", str(cfg["batch_size"]),
        "--gradient-accumulation", str(cfg["grad_accum"]),
        "--epochs", str(plan["epochs"]),
        "--warmup-steps", str(cfg["warmup_steps"]),
        "--weight-decay", repr(cfg["weight_decay"]),
        "--max-grad-norm", repr(cfg["max_grad_norm"]),
        "--seed", str(seed),
        "--shift", repr(cfg["shift"]),
        "--num-inference-steps", str(cfg["num_inference_steps"]),
        "--optimizer-type", cfg["optimizer"],
        "--scheduler-type", cfg["scheduler"],
        "--gradient-checkpointing" if cfg["gradient_checkpointing"] else "--no-gradient-checkpointing",
        "--adapter-type", "lora",
        "--rank", str(cfg["rank"]),
        "--alpha", str(cfg["alpha"]),
        "--dropout", repr(cfg["dropout"]),
        "--target-modules", *cfg["target_modules"],
        "--attention-type", cfg["attention_type"],
        "--output-dir", str(output_dir),
        "--save-every", str(plan["save_every_epochs"]),
        "--log-every", str(cfg["log_every"]),
        "--cfg-ratio", repr(cfg["cfg_ratio"]),
    ]
    if resume_from:
        argv += ["--resume-from", str(resume_from)]
    return argv


# --------------------------------------------------------------------------- identity / resume

IDENTITY_KEYS = (
    "model_variant", "weights_revision", "rank", "alpha", "dropout", "target_modules", "attention_type",
    "lr", "batch_size", "grad_accum", "steps", "epochs", "warmup_steps", "scheduler", "optimizer",
    "weight_decay", "max_grad_norm", "cfg_ratio", "shift", "precision", "max_duration", "data",
)


def identity_fields(cfg: dict, plan: dict, data_fingerprint: str) -> dict:
    return {
        "model_variant": cfg["model_variant"],
        "weights_revision": cfg["weights_revision"],
        "rank": int(cfg["rank"]),
        "alpha": int(cfg["alpha"]),
        "dropout": float(cfg["dropout"]),
        "target_modules": list(cfg["target_modules"]),
        "attention_type": cfg["attention_type"],
        "lr": float(cfg["lr"]),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum": int(cfg["grad_accum"]),
        "steps": int(cfg["steps"]),
        "epochs": int(plan["epochs"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "scheduler": cfg["scheduler"],
        "optimizer": cfg["optimizer"],
        "weight_decay": float(cfg["weight_decay"]),
        "max_grad_norm": float(cfg["max_grad_norm"]),
        "cfg_ratio": float(cfg["cfg_ratio"]),
        "shift": float(cfg["shift"]),
        "precision": cfg["precision"],
        "max_duration": float(cfg["max_duration"]),
        "data": data_fingerprint,
    }


def run_identity(cfg: dict, plan: dict, data_fingerprint: str) -> dict:
    fields = identity_fields(cfg, plan, data_fingerprint)
    assert tuple(fields) == IDENTITY_KEYS
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return {"arm": str(cfg["arm"]), "seed": int(cfg["seed"]), "cfg_hash": digest,
            "prefix": f"{cfg['arm']}_s{int(cfg['seed'])}_h{digest[:6]}", "fields": fields}


def ckpt_dir_name(prefix: str, epoch: int) -> str:
    return f"{prefix}_epoch{epoch:04d}"


def final_dir_name(prefix: str) -> str:
    return f"{prefix}_final"


def parse_ckpt_epoch(name: str, prefix: str) -> int | None:
    m = re.match(rf"^{re.escape(prefix)}_epoch(\d{{4,}})$", name)
    return int(m.group(1)) if m else None


UPSTREAM_CKPT_RE = re.compile(r"^epoch_(\d+)_loss_")  # trainer_fixed.py:516


def is_complete_ckpt(d: Path, need_state: bool = True) -> bool:
    files = CKPT_FILES if need_state else ADAPTER_FILES
    return d.is_dir() and (d / META_NAME).is_file() and all((d / f).is_file() for f in files)


def ckpt_meta(identity: dict, epoch: int, global_step: int, kind: str) -> dict:
    return {"arm": identity["arm"], "seed": identity["seed"], "cfg_hash": identity["cfg_hash"],
            "prefix": identity["prefix"], "epoch": epoch, "global_step": global_step, "kind": kind,
            "identity": identity["fields"]}


def check_identity(d: Path, identity: dict) -> dict:
    """Refuse a checkpoint written by a different run identity; return its ravi_meta."""
    mp = Path(d) / META_NAME
    if not mp.is_file():
        raise ValueError(f"{d} has no {META_NAME}; refusing to resume from it")
    md = json.loads(mp.read_text())
    got = {k: md.get(k) for k in ("arm", "seed", "cfg_hash")}
    want = {k: identity[k] for k in ("arm", "seed", "cfg_hash")}
    if got != want:
        saved = md.get("identity") or {}
        diffs = {k: (saved.get(k), v) for k, v in identity["fields"].items() if saved.get(k) != v}
        raise ValueError(f"{d} belongs to a different run (saved {got}, this run {want}); "
                         f"differing config (saved, this run): {diffs}")
    return md


class ResumePlan:
    def __init__(self, epoch: int = 0, global_step: int = 0, path: str | None = None,
                 source: str | None = None, action: str = "train"):
        self.epoch, self.global_step, self.path, self.source, self.action = epoch, global_step, path, source, action

    def as_dict(self) -> dict:
        return dict(vars(self))


def _s3_list_dirs(s3: S3Sink, prefix: str) -> dict[str, set[str]]:
    """{dir name: {file names}} under ``{s3_out}/{arm}/checkpoints/`` starting with prefix."""
    out: dict[str, set[str]] = {}
    if not s3.enabled:
        return out
    base = f"{s3.prefix}/checkpoints/"
    for page in s3.client.get_paginator("list_objects_v2").paginate(Bucket=s3.bucket, Prefix=base + prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(base):]
            if "/" in rel:
                d, f = rel.split("/", 1)
                out.setdefault(d, set()).add(f)
    return out


def _s3_download_dir(s3: S3Sink, uri: str, dest: Path) -> Path:
    bucket, key = split_s3(uri)
    tmp = dest.with_name(dest.name + ".download")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = 0
    for page in s3.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(key.rstrip("/")) + 1:]
            if rel and "/" not in rel:
                s3.client.download_file(bucket, obj["Key"], str(tmp / rel))
                n += 1
    if not n:
        raise FileNotFoundError(f"nothing under {uri}")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


def find_resume(cfg: dict, identity: dict, plan: dict, s3: S3Sink) -> ResumePlan:
    """Newest complete checkpoint of this identity (``auto``/``publish``) or the given one."""
    mode = cfg["resume"]
    prefix = identity["prefix"]
    ckpt_root = Path(cfg["checkpoint_dir"])
    if mode == "none":
        return ResumePlan()
    if mode in ("auto", "publish"):
        final_local = ckpt_root / final_dir_name(prefix)
        cands: list[tuple[int, int, str]] = []  # (epoch, local-first, source)
        if is_complete_ckpt(final_local, need_state=False):
            cands.append((plan["epochs"], 2, str(final_local)))
        if ckpt_root.is_dir():
            for d in ckpt_root.iterdir():
                e = parse_ckpt_epoch(d.name, prefix)
                if e is not None and is_complete_ckpt(d):
                    cands.append((e, 1, str(d)))
        for d, files in _s3_list_dirs(s3, prefix).items():
            need = set(ADAPTER_FILES) | {META_NAME}
            if d == final_dir_name(prefix) and need <= files:
                cands.append((plan["epochs"], 0, s3.uri(f"checkpoints/{d}")))
            e = parse_ckpt_epoch(d, prefix)
            if e is not None and (need | {"training_state.pt"}) <= files:
                cands.append((e, 0, s3.uri(f"checkpoints/{d}")))
        if not cands:
            if mode == "publish":
                raise FileNotFoundError(f"resume=publish but no finished {prefix} checkpoint locally or on S3")
            return ResumePlan()
        epoch, _, src = max(cands)
        path = Path(src) if not src.startswith("s3://") else _s3_download_dir(s3, src, ckpt_root / src.rstrip("/").rsplit("/", 1)[-1])
    else:
        src = mode
        if src.startswith("s3://"):
            path = _s3_download_dir(s3, src, Path(cfg["work_dir"]) / "resume_download" / src.rstrip("/").rsplit("/", 1)[-1])
        else:
            path = Path(src)
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint {src} is not a directory")
    md = check_identity(path, identity)
    is_final = md.get("kind") == "final"
    if not is_final and not (path / "training_state.pt").is_file():
        raise ValueError(f"{path} has no training_state.pt; cannot resume from it")
    rp = ResumePlan(int(md["epoch"]), int(md.get("global_step", 0)), str(path), src)
    return _resume_action(mode, rp, plan, is_final)


def _resume_action(mode: str, rp: ResumePlan, plan: dict, is_final: bool = False) -> ResumePlan:
    epochs = plan["epochs"]
    if rp.epoch > epochs:
        raise ValueError(f"checkpoint {rp.source} is at epoch {rp.epoch}, past epochs={epochs}")
    if rp.epoch == epochs or is_final:
        if mode != "publish":
            raise ValueError(f"{rp.source} already reached epochs={epochs}. Set resume=publish to re-publish its "
                             "adapter, or resume=none to train again from scratch")
        rp.action = "publish"
    elif mode == "publish":
        raise ValueError(f"resume=publish needs a finished run (epoch {epochs}); newest is epoch {rp.epoch}")
    return rp


# --------------------------------------------------------------------------- environment

ENV_CHECK = r"""
import json, os, sys, importlib.metadata as md
out = {"python": sys.version.split()[0]}
import torch
out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
out["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
if out["gpu_count"]:
    out["gpu"] = torch.cuda.get_device_name(0)
    out["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    out["gpu_bf16"] = torch.cuda.is_bf16_supported()
for pkg in ("flash-attn", "peft", "lightning", "transformers", "torchaudio", "torchcodec", "safetensors",
            "huggingface-hub", "ace-step"):
    try:
        out[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        out[pkg] = None
try:
    import flash_attn  # what training_v2/model_loader.py:25-50 checks
    out["flash_attn_import"] = "ok"
except Exception as e:
    out["flash_attn_import"] = f"{type(e).__name__}: {e}"
try:
    import acestep.training_v2.cli.args  # noqa: F401
    out["acestep_import"] = "ok"
except Exception as e:
    out["acestep_import"] = f"{type(e).__name__}: {e}"
print("RAVI_ENV " + json.dumps(out))
"""


def check_environment(cfg: dict) -> dict:
    res = subprocess.run([sys.executable, "-c", ENV_CHECK], capture_output=True, text=True)
    line = next((l for l in res.stdout.splitlines() if l.startswith("RAVI_ENV ")), None)
    if res.returncode or line is None:
        raise RuntimeError(f"environment check failed:\n{res.stdout}\n{res.stderr}")
    env = json.loads(line[len("RAVI_ENV "):])
    if env["acestep_import"] != "ok":
        raise RuntimeError(f"ACE-Step not importable: {env['acestep_import']}")
    if cfg["require_cuda"] and not env["cuda_available"]:
        raise RuntimeError("CUDA not available; detect_gpu would fall back to CPU (training_v2/gpu_utils.py)")
    if cfg["require_flash_attn"]:
        if env["flash_attn_import"] != "ok":
            raise RuntimeError("flash-attn not importable (" + env["flash_attn_import"] + "); training_v2 would "
                               "fall back to sdpa (model_loader.py:148-152). Set require_flash_attn=false to accept that")
        if env.get("gpu_capability") and env["gpu_capability"][0] < 8:
            raise RuntimeError(f"GPU capability {env['gpu_capability']} < 8.0: model_loader.py:38-44 skips flash-attn")
    if cfg["precision"] == "bf16" and env.get("gpu_count") and not env.get("gpu_bf16"):
        raise RuntimeError("precision bf16 requested but the GPU has no bf16 support")
    return env


def acestep_commit(root: str) -> str | None:
    if os.environ.get("ACESTEP_COMMIT"):
        return os.environ["ACESTEP_COMMIT"]
    marker = Path(root) / ".acestep_commit"
    if marker.is_file():
        return marker.read_text().strip()
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- log parsing

STEP_RE = re.compile(r"Epoch (\d+)/(\d+), Step (\d+), Loss: (\S+)")  # trainer_fixed.py:457
EPOCH_RE = re.compile(r"\[OK\] Epoch (\d+)/(\d+) in ([0-9.]+)s, Loss: (\S+)")  # trainer_fixed.py:510
ATTN_RE = re.compile(r"\[OK\] Model loaded with attn_implementation=(\S+)")  # model_loader.py:165
FAIL_RE = re.compile(r"^\s*\[FAIL\]", re.M)  # ui/progress.py:418 prints every TrainingUpdate msg


def _float(x: str) -> float | None:
    try:
        return float(x.rstrip(",;"))
    except ValueError:
        return None


def parse_train_log(text: str) -> dict:
    steps = [{"epoch": int(e), "step": int(s), "loss": _float(l)} for e, _, s, l in STEP_RE.findall(text)]
    epochs = [{"epoch": int(e), "seconds": float(t), "loss": _float(l)} for e, _, t, l in EPOCH_RE.findall(text)]
    attn = ATTN_RE.findall(text)
    fails = [ln.strip() for ln in text.splitlines() if FAIL_RE.match(ln)]
    return {"loss_curve": steps, "epoch_losses": epochs, "attn_implementation": attn[-1] if attn else None,
            "fail_lines": fails[:20], "complete": "[OK] Training complete!" in text}


# --------------------------------------------------------------------------- child processes

KILL_GRACE_SECONDS = 30


class Interrupted(RuntimeError):
    def __init__(self, signum: int, exit_code: int | None):
        self.signum, self.exit_code = signum, exit_code
        super().__init__(f"interrupted by signal {signum}; child exit code {exit_code}")


def _run_child(cmd: list[str], env: dict, cwd: Path, log_path: Path,
               poller: Callable[[bool], None] | None = None, poll_seconds: float = 15) -> tuple[int, int | None]:
    """Run ``cmd``, tee its output to ``log_path``, call ``poller(final)`` periodically.

    Returns ``(exit_code, signal_or_None)``. SIGTERM/SIGINT go to the child as SIGTERM,
    followed by SIGKILL after :data:`KILL_GRACE_SECONDS`. If the wrapper itself fails, the
    child is killed rather than orphaned."""
    print(f"[ravi] launching (cwd {cwd}): {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    got: list[int] = []
    timers: list[threading.Timer] = []
    stop = threading.Event()

    def _kill_if_alive():
        if proc.poll() is None:
            print(f"[ravi] child still alive {KILL_GRACE_SECONDS}s after SIGTERM; sending SIGKILL", flush=True)
            proc.kill()

    def _forward(signum, _frame):
        print(f"[ravi] received signal {signum}; stopping child", flush=True)
        got.append(signum)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            if not timers:
                t = threading.Timer(KILL_GRACE_SECONDS, _kill_if_alive)
                t.daemon = True
                t.start()
                timers.append(t)

    def _loop():
        while not stop.wait(poll_seconds):
            try:
                poller(False)
            except Exception:
                traceback.print_exc()

    old = {sig: signal.signal(sig, _forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    thread = None
    try:
        if poller is not None:
            thread = threading.Thread(target=_loop, daemon=True)
            thread.start()
        with open(log_path, "ab") as log:
            fd = proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk.replace(b"\r", b"\n"))
        code = proc.wait()
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for t in timers:
            t.cancel()
        stop.set()
        if thread is not None:
            thread.join(timeout=60)
    sig = got[0] if got else None
    if poller is not None:
        try:
            poller(True)
        except Exception:
            if code == 0 and sig is None:
                raise
            traceback.print_exc()
    print(f"[ravi] child exited with code {code}" + (f" after signal {sig}" if sig else ""), flush=True)
    return code, sig


def _copy_dir_atomic(src: Path, dst: Path, files: tuple[str, ...] | None, extra: dict[str, str]) -> None:
    """Copy ``files`` (all regular files if None) of ``src`` plus ``extra`` {name: text} to
    ``dst`` via ``{dst}.partial`` + rename (same filesystem as ``dst``)."""
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    names = files if files is not None else tuple(p.name for p in sorted(src.iterdir()) if p.is_file())
    for n in names:
        shutil.copy2(src / n, tmp / n)
    for n, text in extra.items():
        (tmp / n).write_text(text)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)


class Syncer:
    """Publishes upstream epoch checkpoints from the trainer's output dir."""

    def __init__(self, cfg: dict, identity: dict, plan: dict, output_dir: Path, s3: S3Sink, meta: "RunMeta"):
        self.cfg, self.identity, self.plan, self.s3, self.meta = cfg, identity, plan, s3, meta
        self.src_root = output_dir / "checkpoints"
        self.dst_root = Path(cfg["checkpoint_dir"])
        self._sizes: dict[Path, tuple] = {}
        self._done: set[int] = set()
        self._lock = threading.Lock()

    def _stable(self, d: Path, final: bool) -> bool:
        if not all((d / f).is_file() for f in CKPT_FILES):
            return False
        sig = tuple(sorted((p.name, p.stat().st_size) for p in d.iterdir() if p.is_file()))
        prev = self._sizes.get(d)
        self._sizes[d] = sig
        return final or prev == sig

    def poll(self, final: bool = False) -> None:
        with self._lock:
            if not self.src_root.is_dir():
                return
            for d in sorted(self.src_root.iterdir()):
                m = UPSTREAM_CKPT_RE.match(d.name)
                if not m or not d.is_dir():
                    continue
                epoch = int(m.group(1))
                if epoch in self._done or not self._stable(d, final):
                    continue
                step = epoch * self.plan["steps_per_epoch"]
                name = ckpt_dir_name(self.identity["prefix"], epoch)
                dst = self.dst_root / name
                meta = ckpt_meta(self.identity, epoch, step, "epoch")
                meta["upstream_dir"] = d.name
                _copy_dir_atomic(d, dst, None, {META_NAME: json.dumps(meta, indent=2, sort_keys=True)})
                uris = self._upload_dir(dst, f"checkpoints/{name}")
                self.meta.data["checkpoints"].append({"epoch": epoch, "global_step": step, "path": str(dst),
                                                      "s3": uris})
                print(f"[ravi] checkpoint epoch {epoch} (step {step}): {dst}" + (f" -> {uris}" if uris else ""),
                      flush=True)
                self._done.add(epoch)
                self.meta.refresh_log()
                self.meta.write(self.s3)

    def _upload_dir(self, d: Path, rel: str) -> str | None:
        if not self.s3.enabled:
            return None
        names = sorted(p.name for p in d.iterdir() if p.is_file() and p.name != META_NAME) + [META_NAME]
        for n in names:  # ravi_meta.json last: its presence marks a complete upload
            self.s3.upload(d / n, f"{rel}/{n}")
        return self.s3.uri(rel)


# --------------------------------------------------------------------------- run meta


class RunMeta:
    def __init__(self, cfg: dict, paths: list[Path], log_path: Path):
        self.paths, self.log_path = paths, log_path
        self.data: dict[str, Any] = {
            "arm": cfg["arm"], "run_id": cfg["run_id"], "status": "starting",
            "acestep_commit": acestep_commit(cfg["acestep_root"]),
            "ravi_git_sha": os.environ.get("RAVI_GIT_SHA"), "image": os.environ.get("RAVI_IMAGE_TAG"),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "config_sources": cfg.get("_sources"), "on_sagemaker": cfg.get("_on_sagemaker"),
            "start_time": dt.datetime.now(dt.timezone.utc).isoformat(), "end_time": None,
            "wall_seconds": None, "checkpoints": [], "loss_curve": [],
        }
        self._t0 = time.time()

    def refresh_log(self) -> None:
        if self.log_path.is_file():
            parsed = parse_train_log(self.log_path.read_text(errors="replace"))
            prior = self.data.get("loss_curve_prior", [])
            self.data["loss_curve"] = prior + parsed["loss_curve"]
            self.data["epoch_losses"] = parsed["epoch_losses"]
            self.data["attn_implementation"] = parsed["attn_implementation"] or self.data.get("attn_implementation")

    def finish(self, status: str, exit_code: int | None) -> None:
        self.data.update(status=status, exit_code=exit_code,
                         end_time=dt.datetime.now(dt.timezone.utc).isoformat(),
                         wall_seconds=round(time.time() - self._t0, 1))

    def write(self, s3: S3Sink | None = None) -> None:
        text = json.dumps(self.data, indent=2, default=str)
        for p in self.paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.tmp")
            tmp.write_text(text)
            os.replace(tmp, p)
        if s3 is not None and s3.enabled and self.paths:
            s3.upload(self.paths[0], f"runs/{self.data['run_id']}/run_meta.json")


# --------------------------------------------------------------------------- main flow


def _child_env(cfg: dict) -> dict:
    env = dict(os.environ)
    pin_single_gpu(env)
    ravi_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (ravi_root, env.get("PYTHONPATH")) if p)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["ACESTEP_CHECKPOINTS_DIR"] = cfg["weights_dir"]
    return env


def publish_adapter(src: Path, model_dir: Path) -> Path:
    dst = model_dir / "adapter"
    for f in ADAPTER_FILES:
        if not (src / f).is_file():
            raise RuntimeError(f"{src} has no {f}")
    _copy_dir_atomic(src, dst, ADAPTER_FILES, {})
    return dst


def run(cfg: dict, dry_run: bool = False) -> int:
    cfg = resolve(cfg)
    summary, recs = scan_data(cfg)
    print(f"[ravi] data: {json.dumps(summary)}", flush=True)
    plan = plan_epochs(cfg["steps"], summary["n_clips"], cfg["batch_size"], cfg["grad_accum"],
                       cfg["save_every_epochs"], cfg["checkpoint_every_steps"])
    identity = run_identity(cfg, plan, summary["fingerprint"])
    print(f"[ravi] epochs: {json.dumps(plan)}", flush=True)
    work = Path(cfg["work_dir"])
    run_dir = work / str(cfg["run_id"])
    # Tensors depend on the data, the variant (silence latent + DiT condition encoder,
    # training_v2/preprocess.py:205-207,366), the weights, the precision and the truncation.
    tkey = hashlib.sha256(json.dumps([summary["fingerprint"], cfg["model_variant"], cfg["weights_revision"],
                                      cfg["precision"], cfg["max_duration"]]).encode()).hexdigest()[:12]
    tensor_dir = work / f"tensors_{cfg['model_variant']}_{tkey}"
    dataset_json = work / f"dataset_{summary['fingerprint'][:12]}.json"
    output_dir = run_dir / "acestep_out"
    out_dir = Path(cfg["output_data_dir"])
    s3 = S3Sink(cfg["s3_out"], cfg["arm"], cfg["aws_region"]) if not dry_run else S3Sink(None, cfg["arm"], None)

    rp = ResumePlan() if dry_run else find_resume(cfg, identity, plan, s3)
    seed = cfg["seed"] + rp.epoch
    staged = str(work / "resume" / Path(rp.path).name) if rp.path and rp.action == "train" else None
    pre_argv = build_preprocess_argv(cfg, dataset_json, tensor_dir)
    train_argv = build_train_argv(cfg, plan, tensor_dir, output_dir, seed, staged)
    pre_cmd = [sys.executable, "-m", TRAIN_MODULE, *pre_argv]
    train_cmd = [sys.executable, "-m", "ravi.train.acestep_lora", "_child-train", "--", *train_argv]
    if dry_run:
        print(json.dumps({"identity": identity, "resume": cfg["resume"], "epoch_plan": plan,
                          "preprocess_cmd": pre_cmd, "train_cmd": train_cmd}, indent=2))
        return 0

    for d in (run_dir, Path(cfg["checkpoint_dir"]), Path(cfg["model_dir"]), out_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    for old_log in (log_path, out_dir / "preprocess.log"):  # one log per attempt; the loss curve is re-parsed
        if old_log.is_file():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.replace(old_log, old_log.with_name(f"{old_log.stem}.{stamp}{old_log.suffix}"))
    meta = RunMeta(cfg, [out_dir / "run_meta.json", Path(cfg["model_dir"]) / "run_meta.json",
                         Path(cfg["checkpoint_dir"]) / f"{identity['prefix']}_run_meta.json"], log_path)
    prev = Path(cfg["checkpoint_dir"]) / f"{identity['prefix']}_run_meta.json"
    if rp.epoch and prev.is_file():
        try:
            meta.data["loss_curve_prior"] = [r for r in json.loads(prev.read_text()).get("loss_curve", [])
                                             if r.get("epoch", 0) <= rp.epoch]
        except (OSError, json.JSONDecodeError):
            pass
    meta.data.update(
        data=summary, epoch_plan=plan, identity={k: identity[k] for k in ("prefix", "cfg_hash", "fields")},
        resume_mode=cfg["resume"], resume=rp.as_dict() if rp.path else None, effective_seed=seed,
        preprocess_cmd=pre_cmd, train_argv=train_argv,
        upstream_defaults_in_effect={
            "bias": "none", "lora_target_resolution": "PEFT suffix match on q/k/v/o_proj (self and cross attention)",
            "timestep_sampling": "continuous logit-normal from the variant's config.json (cli/config_builder.py:66-75)",
            "shuffle": True, "drop_last": False, "num_devices": 1, "loss_dtype": "float32",
        },
    )
    try:
        meta.data["env"] = env = check_environment(cfg)
        print(f"[ravi] env: {json.dumps(env)}", flush=True)
        child_env = _child_env(cfg)
        if rp.action == "publish":
            print(f"[ravi] resume=publish: publishing the adapter of {rp.path}", flush=True)
            final_src = Path(rp.path)
            exit_code = 0
        else:
            meta.data["weights"] = ensure_weights(cfg)
            meta.data["status"] = "preprocessing"
            meta.write(s3)
            build_dataset_json(cfg["data_dir"], dataset_json, f"ravi_{cfg['arm']}")
            meta.data["dataset_json"] = {"path": str(dataset_json), **check_dataset_json(dataset_json, recs)}
            ts = tensor_status(tensor_dir, recs)
            if ts["extra"]:
                raise RuntimeError(f"{tensor_dir} has tensors for clips not in the data: {ts['extra']}")
            if not ts["complete"]:
                for p in tensor_dir.glob("*.tmp.pt"):  # stale pass-1 intermediates
                    p.unlink()
                code, sig = _run_child(pre_cmd, child_env, work, out_dir / "preprocess.log",
                                       poll_seconds=cfg["poll_seconds"])
                if sig is not None:
                    raise Interrupted(sig, code)
                attn = parse_train_log((out_dir / "preprocess.log").read_text(errors="replace"))["attn_implementation"]
                meta.data["preprocess_attn_implementation"] = attn
                if cfg["require_flash_attn"] and attn and attn != "flash_attention_2":
                    raise RuntimeError(f"preprocessing loaded the DiT with attn_implementation={attn}, not flash")
                ts = tensor_status(tensor_dir, recs)
                if code != 0 or not ts["complete"]:
                    raise RuntimeError(f"preprocessing incomplete (exit {code}): {ts}")
            meta.data["tensors"] = {"dir": str(tensor_dir), **ts}
            print(f"[ravi] tensors: {json.dumps(meta.data['tensors'])}", flush=True)
            if staged:
                shutil.rmtree(staged, ignore_errors=True)
                shutil.copytree(rp.path, staged)
            shutil.rmtree(output_dir, ignore_errors=True)  # a stale final/ must not count as success
            meta.data["status"] = "running"
            meta.write(s3)
            syncer = Syncer(cfg, identity, plan, output_dir, s3, meta)
            exit_code, sig = _run_child(train_cmd, child_env, work, log_path, syncer.poll, cfg["poll_seconds"])
            if sig is not None:
                raise Interrupted(sig, exit_code)
            parsed = parse_train_log(log_path.read_text(errors="replace"))
            meta.refresh_log()
            if exit_code != 0 or parsed["fail_lines"]:
                raise RuntimeError(f"training failed (exit {exit_code}): {parsed['fail_lines'][:5]}")
            final_src = output_dir / "final"
            if not all((final_src / f).is_file() for f in ADAPTER_FILES):
                raise RuntimeError(f"training exited 0 but {final_src} has no PEFT adapter")
            if cfg["require_flash_attn"] and parsed["attn_implementation"] not in (None, "flash_attention_2"):
                meta.data["warnings"] = [f"trained with attn_implementation={parsed['attn_implementation']}"]
            fname = final_dir_name(identity["prefix"])
            fmeta = ckpt_meta(identity, plan["epochs"], plan["actual_steps"], "final")
            _copy_dir_atomic(final_src, Path(cfg["checkpoint_dir"]) / fname, ADAPTER_FILES,
                             {META_NAME: json.dumps(fmeta, indent=2, sort_keys=True)})
            syncer._upload_dir(Path(cfg["checkpoint_dir"]) / fname, f"checkpoints/{fname}")
            tb = output_dir / "runs"
            if tb.is_dir():
                shutil.copytree(tb, out_dir / "tensorboard", dirs_exist_ok=True)
        adapter = publish_adapter(final_src, Path(cfg["model_dir"]))
        meta.data["final_adapter"] = {"path": str(adapter), "epochs": plan["epochs"],
                                      "global_step": plan["actual_steps"],
                                      "s3": s3.uri(f"checkpoints/{final_dir_name(identity['prefix'])}")}
        status = "completed" if not s3.errors else "completed_with_s3_errors"
        meta.data["s3_errors"] = s3.errors
        meta.finish(status, exit_code)
        meta.write(s3)
        if s3.enabled and log_path.is_file():
            s3.upload(log_path, f"runs/{cfg['run_id']}/train.log")
        print(f"[ravi] {status}: adapter at {adapter}", flush=True)
        return 0
    except BaseException as e:
        interrupted = isinstance(e, Interrupted)
        meta.data["error"] = f"{type(e).__name__}: {e}"
        meta.refresh_log()
        meta.finish("interrupted" if interrupted else "failed", e.exit_code if interrupted else None)
        if s3.enabled and log_path.is_file():
            s3.upload(log_path, f"runs/{cfg['run_id']}/train.log")
        meta.write(s3)
        _write_failure(meta.data["error"])
        raise


# --------------------------------------------------------------------------- trainer child


def _reload_lora_weights(trainer: Any, resume_path: str) -> int:
    """Load the checkpoint's adapter into the training PeftModel and verify every tensor."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    d = Path(resume_path)
    d = d.parent if d.is_file() else d
    f = d / "adapter_model.safetensors"
    if not f.is_file():
        raise SystemExit(f"[ravi] FATAL: {f} missing")
    saved = load_file(str(f))
    dec = trainer.module.model.decoder
    while hasattr(dec, "_forward_module"):  # Fabric wrapper, as trainer_helpers.py:175-178
        dec = dec._forward_module
    res = set_peft_model_state_dict(dec, saved)
    unexpected = list(getattr(res, "unexpected_keys", None) or [])
    if unexpected:
        raise SystemExit(f"[ravi] FATAL: resume: {len(unexpected)} unexpected adapter keys, e.g. {unexpected[:3]}")
    now = get_peft_model_state_dict(dec)
    missing = sorted(set(now) - set(saved))
    bad = [k for k, v in saved.items() if k not in now or not torch.equal(now[k].detach().cpu().to(v.dtype), v)]
    if not saved or missing or bad:
        raise SystemExit(f"[ravi] FATAL: resume: adapter not restored (missing {missing[:3]}, mismatched {bad[:3]})")
    return len(saved)


def child_train(argv: list[str]) -> int:
    """Run ``train_fixed.main()`` with a verified resume (see the module docstring)."""
    import acestep.training_v2.trainer_fixed as tf

    orig = tf.resume_checkpoint  # looked up by FixedLoRATrainer._resume_checkpoint (trainer_fixed.py:211-214)

    def resume_checkpoint(trainer, resume_path, optimizer, scheduler):
        try:
            result = yield from orig(trainer, resume_path, optimizer, scheduler)
        except Exception as e:  # would become "starting fresh" (trainer_fixed.py:393-398)
            raise SystemExit(f"[ravi] FATAL: upstream resume raised {type(e).__name__}: {e}") from None
        if result is None:
            raise SystemExit(f"[ravi] FATAL: upstream resume loaded nothing from {resume_path}")
        n = _reload_lora_weights(trainer, resume_path)
        print(f"[ravi] resume: epoch {result[0]}, step {result[1]}, {n} LoRA tensors restored and verified",
              flush=True)
        return result

    tf.resume_checkpoint = resume_checkpoint
    from acestep.training_v2.cli import train_fixed

    sys.argv = [TRAIN_MODULE, *argv]
    return int(train_fixed.main() or 0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_child-train":
        rest = argv[1:]
        return child_train(rest[1:] if rest[:1] == ["--"] else rest)
    if argv and argv[0] == "train":  # SageMaker runs `<entrypoint> train`
        argv = argv[1:]
    download_only = bool(argv) and argv[0] == "download-weights"
    if download_only:
        argv = argv[1:]
    p = argparse.ArgumentParser(prog="python -m ravi.train.acestep_lora", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="YAML/JSON run config")
    p.add_argument("--hyperparameters", help=f"hyperparameters.json (default {SM_HP_PATH} if present)")
    p.add_argument("--s3-out", help="s3://bucket/prefix for checkpoints and run_meta")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    p.add_argument("--dry-run", action="store_true", help="validate data and print the commands; no training")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config, args.hyperparameters, args.set, args.s3_out)
        if download_only:
            if not cfg["data_dir"]:
                cfg["data_dir"] = "/"  # not used; resolve() requires one
            print(json.dumps(ensure_weights(resolve(cfg)), indent=2))
            return 0
        return run(cfg, dry_run=args.dry_run)
    except Exception as e:
        _write_failure(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
