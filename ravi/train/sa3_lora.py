"""Run Stable Audio 3 LoRA training (upstream ``scripts/train_lora.py``) for one caption arm.

Entry point of the ``ravi-sa3-train`` image (``ENTRYPOINT ["python", "-m", "ravi.train.sa3_lora"]``).
Works the same under ``docker run`` on EC2 and as a SageMaker Training Job.

Configuration, lowest to highest precedence:
  1. built-in defaults (:data:`DEFAULTS`)
  2. ``--config FILE`` (YAML or JSON), or a ``config`` key in hyperparameters.json
  3. SageMaker ``/opt/ml/input/config/hyperparameters.json`` (values may be JSON-encoded
     strings; dotted keys like ``caption_aug.p_lowercase`` set nested values)
  4. ``--set key=value`` and ``--s3-out URI`` on the command line

What a run does:
  * preflight: every audio file in the data dir has a non-empty sibling ``.txt``
    (upstream silently swaps in a random other clip when one is missing,
    ``train_lora.py:79-80`` + ``stable_audio_3/data/dataset.py:297-298``), clip lengths
    vs ``--duration``, CUDA visible, flash-attn importable (upstream silently falls back
    without it, ``stable_audio_3/models/transformer.py:22-36``);
  * builds the train_lora.py argv and runs it in a child process through
    :mod:`ravi.train._launch_train_lora`, which installs the caption augmentation hook;
  * after each Lightning checkpoint (``*.ckpt``, LoRA tensors + ``lora_config`` only,
    ``stable_audio_3/training/diffusion.py:634-641``) converts it to ``.safetensors``
    with upstream's ``convert_lora_ckpt_to_safetensors``, writes it to the checkpoint dir
    (``/opt/ml/checkpoints``) and, if ``s3_out`` is set, uploads it with boto3;
  * copies demo WAVs (written by upstream to the child's CWD,
    ``diffusion.py:937-939``) to the output dir and S3;
  * writes ``run_meta.json`` (argv, SA3 commit, versions, start/end, checkpoints,
    loss curve from Lightning's CSV logger) after every checkpoint and at the end.

Resume: upstream checkpoints hold only LoRA weights; ``on_save_checkpoint`` clears the
optimizer state and step counter (``diffusion.py:634-641``), so Lightning cannot resume
them. ``--lora_checkpoint`` (``train_lora.py:143-145, 362-366``) is a *weights-only warm
start*: AdamW moments restart from zero and the step counter from 0.

Checkpoints are named ``{arm}_s{seed}_h{hash6}_step{N:06d}.safetensors``; ``hash`` is the
sha256 of :func:`identity_fields` (model, rank, alpha, adapter type, dropout, lr, steps,
batch size, duration, include/exclude, data dir basename, caption_aug, inpaint mask
probabilities). The full hash, arm, seed and step are also in the safetensors metadata
(``ravi_*`` keys), and a resume refuses any file whose identity differs. Modes:

* ``none`` (default off SageMaker): always train from scratch.
* ``auto`` (default on SageMaker, for spot restarts): newest checkpoint of this identity,
  the larger step of the local checkpoint dir and S3; trains ``steps - N`` more steps with
  seed ``seed + N`` and renames new checkpoints with the ``N`` offset. A checkpoint that
  already reached ``steps`` is an error, not a silent re-publish.
* ``publish``: like ``auto`` but requires the checkpoint at ``steps`` and only publishes it.
* a path / ``s3://`` URI: that checkpoint (identity still checked).

Upstream's LR is constant (no scheduler in ``train_lora.py:155-166``), so the LR schedule
is unaffected by a resume; the optimizer reset is not.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from ravi.train.caption_aug import CaptionAugConfig, check_trigger

# --------------------------------------------------------------------------- config

SA3_DEFAULT_ROOT = "/opt/sa3"
SM_HP_PATH = "/opt/ml/input/config/hyperparameters.json"

ADAPTER_TYPES = (
    "lora", "dora", "dora-rows", "dora-cols", "bora",
    "lora-xs", "dora-rows-xs", "dora-cols-xs", "bora-xs",
)  # train_lora.py:318-331
BASE_MODELS = ("medium-base", "small-music-base", "small-sfx-base")  # model_configs.py:109-111
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus")  # data/dataset.py:91
# The duration conditioner, named as in docs/workflows/lora.md:91-95,115
# ("conditioners.seconds_total.embedder.embedding.1").
DURATION_CONDITIONER_PATTERN = "seconds_total"

DEFAULTS: dict[str, Any] = {
    "arm": "unnamed",
    "run_id": None,
    "data_dir": None,  # falls back to SM_CHANNEL_TRAIN
    "encoded_dir": None,
    "model": "medium-base",
    "rank": 16,
    "lora_alpha": None,  # upstream default: = rank
    "adapter_type": "dora-rows",
    "dropout": 0.0,
    "lr": 1e-4,
    "steps": 6000,
    "batch_size": 1,
    "duration": 47.0,
    "seed": 42,
    "checkpoint_every": 500,
    "demo_every": 500,
    "log_every": 50,
    "num_workers": 3,  # g6e.xlarge has 4 vCPUs; upstream default 8
    "base_precision": "bf16",
    "exclude_duration_conditioner": True,
    "include": None,
    "exclude": None,  # extra patterns on top of the duration conditioner
    "svd_bases_path": None,
    # default: "auto" on SageMaker (spot restarts), "none" elsewhere.
    # none | auto | publish | <path or s3:// uri of a {arm}_s{seed}_h{hash}_step{N}.safetensors>
    "resume": "default",
    "s3_out": None,  # s3://bucket/prefix ; run files go to {s3_out}/{arm}/...
    "caption_aug": {},  # CaptionAugConfig fields
    "inpaint_mask_type_probabilities": None,  # None = upstream [0.1, 0.8, 0.1]
    "hf_token_ssm_param": None,  # e.g. /uaudio/hf_token ; used only if HF_TOKEN unset
    "aws_region": None,
    "require_cuda": True,
    "require_flash_attn": True,
    "smoke": False,  # smoke/debug run: relaxes long_clip_policy's default to warn
    "long_clip_policy": None,  # error | warn ; default error (warn when smoke: true)
    "sample_rate": 44100,  # preflight only; the trainer reads it from the model config
    "downsampling_ratio": 4096,  # preflight only (docs/workflows/autoencoder.md:19)
    "sa3_root": None,  # SA3_ROOT env or /opt/sa3
    "work_dir": None,
    "checkpoint_dir": None,
    "model_dir": None,
    "output_data_dir": None,
    "poll_seconds": 15,
}


# Keys whose value is a string (names, paths, URIs, enum choices). They are never parsed
# as JSON/bool/None, so arm "123", run_id "true" or a path "null" stay strings. The only
# transformation is unwrapping a JSON-encoded string, which is how the SageMaker Python
# SDK serialises str hyperparameters ('"B"' -> 'B').
STRING_KEYS = frozenset({
    "arm", "run_id", "data_dir", "encoded_dir", "model", "adapter_type", "base_precision",
    "svd_bases_path", "resume", "s3_out", "hf_token_ssm_param", "aws_region", "sa3_root",
    "work_dir", "checkpoint_dir", "model_dir", "output_data_dir", "long_clip_policy", "config",
    "caption_aug.field_sep", "caption_aug.join_sep", "caption_aug.trigger",
})


def _unwrap_json_string(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] == '"':
        try:
            out = json.loads(v)
        except json.JSONDecodeError:
            return value
        if isinstance(out, str):
            return out
    return value


def _coerce(value: Any, key: str | None = None) -> Any:
    if key in STRING_KEYS:
        if isinstance(value, str):
            return _unwrap_json_string(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key in ("arm", "run_id"):
            return str(value)  # YAML `arm: 123`
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


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _merge(cfg: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        value = _coerce(value, key)
        if "." in key:
            _set_dotted(cfg, key, value)
        elif isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value


def _load_file(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # installed in the image

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_config(
    config_path: str | None = None,
    hp_path: str | None = None,
    sets: list[str] | None = None,
    s3_out: str | None = None,
) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    hp: dict = {}
    hp_file = Path(hp_path or os.environ.get("RAVI_HP_PATH", SM_HP_PATH))
    if hp_file.is_file():
        hp = json.loads(hp_file.read_text())
    cfg_file = config_path or (_coerce(hp.get("config"), "config") if hp.get("config") else None)
    if cfg_file:
        _merge(cfg, _load_file(cfg_file))
    # sagemaker_* keys are injected by the SageMaker SDK for script-mode jobs, and
    # _-prefixed ones (e.g. _tuning_objective_metric) by hyperparameter tuning; not ours.
    _merge(cfg, {k: v for k, v in hp.items()
                 if k != "config" and not k.startswith(("sagemaker_", "_"))})
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
    cfg["_sources"] = {
        "config": str(cfg_file) if cfg_file else None,
        "hyperparameters": str(hp_file) if hp else None,
        "sets": list(sets or []),
    }
    return cfg


def resolve(cfg: dict) -> dict:
    """Fill environment-dependent defaults and validate."""
    cfg = copy.deepcopy(cfg)
    if not cfg["data_dir"] and not cfg["encoded_dir"]:
        cfg["data_dir"] = os.environ.get("SM_CHANNEL_TRAIN")
    if bool(cfg["data_dir"]) == bool(cfg["encoded_dir"]):
        raise ValueError("exactly one of data_dir (or SM_CHANNEL_TRAIN) / encoded_dir is required")
    if cfg["model"] not in BASE_MODELS:
        raise ValueError(f"train_lora.py only accepts base models {BASE_MODELS}, got {cfg['model']!r}")
    if cfg["adapter_type"] not in ADAPTER_TYPES:
        raise ValueError(f"adapter_type {cfg['adapter_type']!r} not in {ADAPTER_TYPES}")
    for key in ("steps", "checkpoint_every", "demo_every", "log_every", "rank", "batch_size"):
        cfg[key] = int(cfg[key])
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    cfg["num_workers"] = int(cfg["num_workers"])
    cfg["seed"] = int(cfg["seed"])
    cfg["lr"] = float(cfg["lr"])
    cfg["duration"] = float(cfg["duration"])
    if cfg["steps"] % cfg["checkpoint_every"]:
        # ModelCheckpoint(every_n_train_steps=...) (train_lora.py:224-226) never saves at
        # the end of training, so a non-multiple loses the final weights.
        raise ValueError(
            f"steps={cfg['steps']} must be a multiple of checkpoint_every={cfg['checkpoint_every']}: "
            "upstream only saves every checkpoint_every steps, never at the end"
        )
    aug = CaptionAugConfig.from_dict(cfg["caption_aug"])
    cfg["caption_aug"] = aug.to_dict()
    if aug.enabled and cfg["num_workers"] < 1:
        raise ValueError("caption augmentation only runs in DataLoader workers; set num_workers >= 1")
    if cfg["inpaint_mask_type_probabilities"] is not None:
        from ravi.train._launch_train_lora import validate_mask_probs

        cfg["inpaint_mask_type_probabilities"] = validate_mask_probs(cfg["inpaint_mask_type_probabilities"])
    cfg["smoke"] = bool(cfg["smoke"])
    if cfg["long_clip_policy"] is None:
        cfg["long_clip_policy"] = "warn" if cfg["smoke"] else "error"
    if cfg["long_clip_policy"] not in ("warn", "error"):
        raise ValueError("long_clip_policy must be warn or error")
    for key in ("include", "exclude"):  # "a b" -> ["a", "b"] (substring patterns have no spaces)
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].split() or None

    on_sm = Path(SM_HP_PATH).is_file() or "SM_TRAINING_ENV" in os.environ
    mode = cfg["resume"]
    if mode in (None, False) or str(mode).strip().lower() in ("none", "false", "off", "no", ""):
        mode = "none"
    elif str(mode).strip().lower() == "default":
        # Off SageMaker a stale checkpoint dir on the box must not be picked up silently.
        mode = "auto" if on_sm else "none"
    cfg["resume"] = str(mode).strip()
    cfg["sa3_root"] = cfg["sa3_root"] or os.environ.get("SA3_ROOT", SA3_DEFAULT_ROOT)
    if not cfg["run_id"]:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cfg["run_id"] = os.environ.get("TRAINING_JOB_NAME") or f"{cfg['arm']}-{stamp}"
    cfg["checkpoint_dir"] = cfg["checkpoint_dir"] or "/opt/ml/checkpoints"
    cfg["model_dir"] = cfg["model_dir"] or os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    cfg["output_data_dir"] = cfg["output_data_dir"] or os.environ.get(
        "SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"
    )
    cfg["work_dir"] = cfg["work_dir"] or os.environ.get("RAVI_WORK_DIR") or "/opt/ml/ravi_work"
    # Absolute paths: the trainer runs with cwd = its demo dir (demos land in its cwd,
    # diffusion.py:937-939), so a relative data_dir/save_dir/checkpoint would resolve there.
    for key in ("data_dir", "encoded_dir", "svd_bases_path", "sa3_root", "checkpoint_dir",
                "model_dir", "output_data_dir", "work_dir"):
        if cfg[key]:
            cfg[key] = str(Path(str(cfg[key])).expanduser().resolve())
    if cfg["resume"] not in ("none", "auto", "publish") and not cfg["resume"].startswith("s3://"):
        cfg["resume"] = str(Path(cfg["resume"]).expanduser().resolve())
    cfg["_on_sagemaker"] = on_sm
    return cfg


# --------------------------------------------------------------------------- argv


def exclude_patterns(cfg: dict) -> list[str]:
    patterns = list(cfg.get("exclude") or [])
    if cfg["exclude_duration_conditioner"] and DURATION_CONDITIONER_PATTERN not in patterns:
        patterns.append(DURATION_CONDITIONER_PATTERN)
    return patterns


def build_train_lora_argv(
    cfg: dict, save_dir: Path, steps: int, seed: int, lora_checkpoint: str | None
) -> list[str]:
    """argv for scripts/train_lora.py; every flag exists in train_lora.py:300-383."""
    argv = ["--model", cfg["model"]]
    if cfg["data_dir"]:
        argv += ["--data_dir", str(cfg["data_dir"])]
    else:
        argv += ["--encoded_dir", str(cfg["encoded_dir"])]
    argv += [
        "--rank", str(cfg["rank"]),
        "--adapter_type", cfg["adapter_type"],
        "--dropout", repr(float(cfg["dropout"])),
        "--base_precision", cfg["base_precision"],
        "--lr", repr(cfg["lr"]),
        "--steps", str(steps),
        "--batch_size", str(cfg["batch_size"]),
        "--duration", repr(cfg["duration"]),
        "--seed", str(seed),
        "--logger", "csv",
        "--name", str(cfg["arm"]),
        "--save_dir", str(save_dir),
        "--checkpoint_every", str(cfg["checkpoint_every"]),
        "--log_every", str(cfg["log_every"]),
        "--demo_every", str(cfg["demo_every"]),
        "--num_workers", str(cfg["num_workers"]),
    ]
    if cfg["lora_alpha"] is not None:
        argv += ["--lora_alpha", repr(float(cfg["lora_alpha"]))]
    if cfg["include"]:
        argv += ["--include", *map(str, cfg["include"])]
    excl = exclude_patterns(cfg)
    if excl:
        argv += ["--exclude", *excl]
    if cfg["svd_bases_path"]:
        argv += ["--svd_bases_path", str(cfg["svd_bases_path"])]
    if lora_checkpoint:
        argv += ["--lora_checkpoint", str(lora_checkpoint)]
    return argv


def build_child_cmd(cfg: dict, train_argv: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "ravi.train._launch_train_lora", "--sa3-root", str(cfg["sa3_root"])]
    # Always passed: with enabled=False the hook still renders field_sep as join_sep.
    cmd += ["--caption-aug-json", CaptionAugConfig.from_dict(cfg["caption_aug"]).to_json()]
    if cfg["inpaint_mask_type_probabilities"] is not None:
        cmd += ["--inpaint-mask-probs", json.dumps(list(cfg["inpaint_mask_type_probabilities"]))]
    return cmd + ["--", *train_argv]


# --------------------------------------------------------------------------- preflight


def list_audio_files(data_dir: Path) -> list[Path]:
    """Same file discovery as upstream get_audio_filenames (data/dataset.py:88-116)."""
    filelist = data_dir / "filelist.txt"
    if filelist.exists():
        return [data_dir / line.strip() for line in filelist.read_text().splitlines() if line.strip()]
    out = []
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                out.append(Path(root) / name)
    return sorted(out)


def scan_dataset(cfg: dict) -> dict:
    """Validate the data dir the way the upstream loader will see it."""
    sr = int(cfg["sample_rate"])
    ds = int(cfg["downsampling_ratio"])
    sample_size = (int(cfg["duration"] * sr) // ds) * ds  # train_lora.py:105
    summary: dict[str, Any] = {
        "sample_size": sample_size,
        "window_seconds": sample_size / sr,
        "used_filelist": False,
    }
    if cfg["encoded_dir"]:
        root = Path(cfg["encoded_dir"])
        npys = [p for p in root.rglob("*.npy") if p.name != "silence.npy"]
        missing = [str(p) for p in npys if not p.with_suffix(".json").exists()]
        no_prompt = []
        for p in npys:
            j = p.with_suffix(".json")
            if j.exists():
                md = json.loads(j.read_text())
                if not isinstance(md.get("prompt"), str) or not md["prompt"].strip():
                    no_prompt.append(str(j))
        summary.update(n_items=len(npys), missing_json=missing[:20], missing_prompt=no_prompt[:20])
        if missing or no_prompt:
            raise ValueError(f"encoded_dir problems: {len(missing)} missing .json, {len(no_prompt)} without prompt")
        if len(npys) < 4:
            raise ValueError("need >= 4 items: the demo DataLoader uses batch_size=4, drop_last=True (train_lora.py:228-238)")
        return summary

    root = Path(cfg["data_dir"])
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir {root} does not exist")
    summary["used_filelist"] = (root / "filelist.txt").exists()
    files = list_audio_files(root)
    aug = CaptionAugConfig.from_dict(cfg["caption_aug"])
    missing_txt, empty_txt, bad_trigger, durations = [], [], [], {}
    for f in files:
        txt = f.with_suffix(".txt")  # same rule as train_lora.py:78
        if not txt.exists():
            missing_txt.append(str(f))
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            empty_txt.append(str(f))
        elif aug.enabled:
            try:
                check_trigger(text, aug)
            except ValueError:
                bad_trigger.append(str(txt))
    unreadable = []
    for f in files:
        frames_sr = audio_frames(f)
        if frames_sr is None:
            unreadable.append(str(f))
        else:
            # length after upstream resampling to the model rate (dataset.py:237-239)
            durations[str(f)] = frames_sr[0] * sr / frames_sr[1]
    tol = int(0.1 * sr)
    n_long = sum(1 for n in durations.values() if n > sample_size + tol)
    n_short = sum(1 for n in durations.values() if n < sample_size - tol)
    secs = sorted(n / sr for n in durations.values())
    summary.update(
        n_audio=len(files),
        missing_txt=missing_txt[:20],
        n_missing_txt=len(missing_txt),
        empty_txt=empty_txt[:20],
        n_empty_txt=len(empty_txt),
        n_bad_trigger=len(bad_trigger),
        n_duration_unknown=len(files) - len(durations),
        n_longer_than_window=n_long,  # random-cropped upstream (data/utils.py:42-43)
        n_shorter_than_window=n_short,  # zero-padded + attention-masked (data/utils.py:64-71)
        seconds_min=secs[0] if secs else None,
        seconds_median=secs[len(secs) // 2] if secs else None,
        seconds_max=secs[-1] if secs else None,
    )
    if unreadable:
        # upstream catches the load error and silently substitutes a random other clip
        # (dataset.py:313-315), so an unreadable file would just vanish from training
        raise ValueError(f"{len(unreadable)} audio files are unreadable: {unreadable[:20]}")
    if len(files) < 4:
        raise ValueError("need >= 4 clips: the demo DataLoader uses batch_size=4, drop_last=True (train_lora.py:228-238)")
    if missing_txt or empty_txt:
        raise ValueError(
            f"{len(missing_txt)} clips without .txt and {len(empty_txt)} with an empty caption, e.g. "
            f"{(missing_txt + empty_txt)[:3]}. Upstream would silently replace them with random other clips."
        )
    if bad_trigger:
        raise ValueError(
            f"{len(bad_trigger)} captions do not start with the trigger field {aug.trigger!r} "
            f"(field_sep {aug.field_sep!r}), e.g. {bad_trigger[:3]}"
        )
    if n_long:
        msg = (
            f"{n_long} clips are longer than the {sample_size / sr:.3f}s window (max {secs[-1]:.2f}s). Upstream "
            "random-crops them but sets seconds_total to the FULL file length (data/utils.py:42-52), so the "
            "duration conditioner sees a wrong length, and because use_effective_length_for_schedule=True "
            "(train_lora.py:180) the noise-schedule shift is computed from that seconds_total, i.e. for a "
            "longer sequence than the crop actually trained (training/diffusion.py:311-319). Crop the clips "
            "to the window, or set long_clip_policy=warn (default for smoke: true)"
        )
        if cfg["long_clip_policy"] == "error":
            raise ValueError(msg)
        print(f"[ravi] WARNING: {msg}", flush=True)
    return summary


def audio_frames(path: Path) -> tuple[int, int] | None:
    """(frames, sample_rate) via soundfile, falling back to torchaudio (upstream's loader,
    dataset.py:235) for formats the bundled libsndfile lacks. None if neither can read it."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames > 0 and info.samplerate > 0:
            return int(info.frames), int(info.samplerate)
    except Exception:
        pass
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        if info.num_frames > 0 and info.sample_rate > 0:
            return int(info.num_frames), int(info.sample_rate)
    except Exception:
        pass
    return None


ENV_CHECK = r"""
import json, sys, importlib.metadata as md
out = {"python": sys.version.split()[0]}
import torch
out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
out["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
out["gpus"] = [
    {"index": i, "name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}
    for i in range(out["gpu_count"])
]
if out["gpus"]:
    out["gpu"] = out["gpus"][0]["name"]
    out["gpu_capability"] = out["gpus"][0]["capability"]
for pkg in ("flash-attn", "pytorch-lightning", "transformers", "torchaudio", "stable-audio-3"):
    try:
        out[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        out[pkg] = None
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # both used by transformer.py:23,31
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
    out["flash_attn_import"] = "ok"
except Exception as e:
    out["flash_attn_import"] = f"{type(e).__name__}: {e}"
print("RAVI_ENV " + json.dumps(out))
"""


def check_environment(cfg: dict) -> dict:
    res = subprocess.run([sys.executable, "-c", ENV_CHECK], capture_output=True, text=True)
    line = next((l for l in res.stdout.splitlines() if l.startswith("RAVI_ENV ")), None)
    if res.returncode or line is None:
        raise RuntimeError(f"environment check failed:\n{res.stdout}\n{res.stderr}")
    env = json.loads(line[len("RAVI_ENV "):])
    if cfg["require_cuda"] and not env["cuda_available"]:
        raise RuntimeError("CUDA not available; train_lora.py would silently train on CPU (train_lora.py:97-99)")
    if cfg["require_flash_attn"] and env["flash_attn_import"] != "ok":
        raise RuntimeError(
            "flash-attn not importable (" + env["flash_attn_import"] + "); SA3 Medium requires it "
            "and stable_audio_3/models/transformer.py:22-36 silently disables it"
        )
    if env.get("gpu_count", 0) > 1:
        print(
            f"[ravi] WARNING: {env['gpu_count']} GPUs visible; the trainer is pinned to one "
            "(CUDA_VISIBLE_DEVICES), see pin_single_gpu", flush=True
        )
    return env


def pin_single_gpu(env: dict) -> None:
    """Expose exactly one GPU to the child. Upstream builds ``pl.Trainer(devices="auto",
    strategy="auto")`` (train_lora.py:271-276), which on a multi-GPU box would start DDP,
    and nothing in this wrapper (checkpoint step offsets, seeds, one Syncer) supports that."""
    vis = env.get("CUDA_VISIBLE_DEVICES")
    if vis is None:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    elif len([d for d in vis.split(",") if d.strip()]) > 1:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={vis!r} exposes several GPUs; single-GPU training only")


def sa3_commit(sa3_root: str) -> str | None:
    if os.environ.get("SA3_COMMIT"):
        return os.environ["SA3_COMMIT"]
    marker = Path(sa3_root) / ".sa3_commit"
    if marker.is_file():
        return marker.read_text().strip()
    try:
        return subprocess.run(
            ["git", "-C", sa3_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def ensure_hf_token(cfg: dict, env: dict) -> None:
    if env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or not cfg["hf_token_ssm_param"]:
        return
    import boto3

    ssm = boto3.client("ssm", region_name=cfg["aws_region"]) if cfg["aws_region"] else boto3.client("ssm")
    value = ssm.get_parameter(Name=cfg["hf_token_ssm_param"], WithDecryption=True)["Parameter"]["Value"]
    env["HF_TOKEN"] = value  # child env only; never logged or written to run_meta


# --------------------------------------------------------------------------- S3


def split_s3(uri: str) -> tuple[str, str]:
    m = re.match(r"^s3://([^/]+)/?(.*)$", uri)
    if not m:
        raise ValueError(f"not an s3 uri: {uri}")
    return m.group(1), m.group(2).strip("/")


class S3Sink:
    """Uploads under ``{s3_out}/{arm}/``. ``enabled`` only governs uploads and listing;
    :meth:`download` works without ``s3_out`` (``resume: s3://...``)."""

    def __init__(self, base_uri: str | None, arm: str, region: str | None):
        self.enabled = bool(base_uri)
        self.errors: list[str] = []
        self.region = region
        self._client = None
        self.bucket, self.prefix = None, None
        if self.enabled:
            self.bucket, prefix = split_s3(base_uri)
            self.prefix = "/".join(p for p in (prefix, str(arm)) if p)

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region) if self.region else boto3.client("s3")
        return self._client

    def uri(self, rel: str) -> str | None:
        return f"s3://{self.bucket}/{self.prefix}/{rel}" if self.enabled else None

    def upload(self, path: Path, rel: str) -> str | None:
        if not self.enabled:
            return None
        try:
            self.client.upload_file(str(path), self.bucket, f"{self.prefix}/{rel}")
            return self.uri(rel)
        except Exception as e:  # recorded in run_meta; a finished training still exits 0
            msg = f"upload {path} -> {self.uri(rel)} failed: {e}"
            print(f"[ravi] ERROR: S3 {msg}", flush=True)
            self.errors.append(msg)
            return None

    def latest_checkpoint(self, run_prefix: str) -> tuple[int, str] | None:
        """Newest ``{run_prefix}_step{N}.safetensors`` under ``{s3_out}/{arm}/checkpoints/``."""
        if not self.enabled:
            return None
        best = None
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/checkpoints/"):
            for obj in page.get("Contents", []):
                step = parse_ckpt_step(obj["Key"].rsplit("/", 1)[-1], run_prefix)
                if step is not None and (best is None or step > best[0]):
                    best = (step, f"s3://{self.bucket}/{obj['Key']}")
        return best

    def exists(self, uri: str) -> bool:
        bucket, key = split_s3(uri)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def download(self, uri: str, dest: Path) -> Path:
        bucket, key = split_s3(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(dest))
        return dest


# --------------------------------------------------------------------------- checkpoints


# Everything that makes two runs' checkpoints non-interchangeable. Seed and arm are in
# the file name; these go into the hash. Steps is included: a 300-step and a 6000-step run
# are different experiments even though both reach step 300.
IDENTITY_KEYS = (
    "model", "rank", "lora_alpha", "adapter_type", "dropout", "lr", "steps", "batch_size",
    "duration", "include", "exclude", "data", "caption_aug", "inpaint_mask_type_probabilities",
)


def identity_fields(cfg: dict) -> dict:
    data = cfg.get("data_dir") or cfg.get("encoded_dir")
    alpha = cfg.get("lora_alpha")
    return {
        "model": cfg["model"],
        "rank": int(cfg["rank"]),
        "lora_alpha": float(alpha if alpha is not None else cfg["rank"]),  # upstream: None -> rank
        "adapter_type": cfg["adapter_type"],
        "dropout": float(cfg.get("dropout") or 0.0),
        "lr": float(cfg["lr"]),
        "steps": int(cfg["steps"]),
        "batch_size": int(cfg["batch_size"]),
        "duration": float(cfg["duration"]),
        "include": sorted(map(str, cfg.get("include") or [])) or None,
        "exclude": sorted(exclude_patterns(cfg)) or None,
        "data": ("encoded:" if cfg.get("encoded_dir") and not cfg.get("data_dir") else "")
        + Path(str(data).rstrip("/")).name,
        "caption_aug": CaptionAugConfig.from_dict(cfg.get("caption_aug")).to_dict(),
        "inpaint_mask_type_probabilities": (
            [float(x) for x in cfg["inpaint_mask_type_probabilities"]]
            if cfg.get("inpaint_mask_type_probabilities") is not None else None
        ),
    }


def run_identity(cfg: dict) -> dict:
    fields = identity_fields(cfg)
    assert tuple(fields) == IDENTITY_KEYS
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return {
        "arm": str(cfg["arm"]),
        "seed": int(cfg["seed"]),
        "cfg_hash": digest,
        "prefix": f"{cfg['arm']}_s{int(cfg['seed'])}_h{digest[:6]}",
        "fields": fields,
    }


def ckpt_name(run_prefix: str, step: int, ext: str = "safetensors") -> str:
    return f"{run_prefix}_step{step:06d}.{ext}"


def parse_ckpt_step(name: str, run_prefix: str | None = None, ext: str = "safetensors") -> int | None:
    """Step of ``{run_prefix}_step{N}.{ext}``; with run_prefix=None any prefix matches."""
    m = re.match(rf"^(?P<prefix>.+)_step(?P<step>\d+)\.{re.escape(ext)}$", name)
    if not m or (run_prefix is not None and m.group("prefix") != str(run_prefix)):
        return None
    return int(m.group("step"))


def lightning_step(path: Path) -> int | None:
    # ModelCheckpoint default filename "epoch={epoch}-step={step}.ckpt"
    m = re.search(r"step=(\d+)", path.name)
    return int(m.group(1)) if m else None


# safetensors layout: u64 little-endian header length, JSON header, raw tensor bytes.
# Header offsets are relative to the data section, so the header can be rewritten
# without touching (or loading) the tensors.
def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return dict(header.get("__metadata__") or {})


def write_safetensors_metadata(src: str | Path, dst: str | Path, extra: dict[str, Any]) -> None:
    with open(src, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        md = dict(header.get("__metadata__") or {})
        md.update({k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in extra.items()})
        header["__metadata__"] = md
        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            shutil.copyfileobj(f, out)


def identity_metadata(identity: dict, step: int) -> dict[str, str]:
    return {
        "ravi_arm": identity["arm"],
        "ravi_seed": str(identity["seed"]),
        "ravi_cfg_hash": identity["cfg_hash"],
        "ravi_step": str(step),
        "ravi_identity": json.dumps(identity["fields"], sort_keys=True),
    }


def check_identity(path: str | Path, identity: dict) -> int:
    """Refuse a checkpoint from a different run identity; return its step."""
    md = read_safetensors_metadata(path)
    if "ravi_cfg_hash" not in md:
        raise ValueError(f"{path} has no ravi identity metadata; refusing to resume from it")
    got = {"arm": md.get("ravi_arm"), "seed": md.get("ravi_seed"), "cfg_hash": md.get("ravi_cfg_hash")}
    want = {"arm": identity["arm"], "seed": str(identity["seed"]), "cfg_hash": identity["cfg_hash"]}
    if got != want:
        diffs = {}
        try:
            saved = json.loads(md.get("ravi_identity", "{}"))
            diffs = {k: (saved.get(k), v) for k, v in identity["fields"].items() if saved.get(k) != v}
        except json.JSONDecodeError:
            pass
        raise ValueError(
            f"{path} belongs to a different run (saved {got}, this run {want}); "
            f"differing config (saved, this run): {diffs}"
        )
    return int(md["ravi_step"])


def convert_ckpt(src: Path, dst: Path, metadata: dict[str, str], tmp_dir: Path) -> None:
    """Lightning .ckpt -> .safetensors (upstream helper) + ravi identity metadata."""
    from stable_audio_3.models.lora.utils import convert_lora_ckpt_to_safetensors, load_lora_checkpoint

    _sd, lora_cfg = load_lora_checkpoint(str(src))
    if not lora_cfg:
        raise RuntimeError(f"{src} has no lora_config; not an upstream LoRA checkpoint")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw = tmp_dir / f".{dst.name}.raw.partial"
    tmp = tmp_dir / f".{dst.name}.partial"
    convert_lora_ckpt_to_safetensors(str(src), str(raw))
    write_safetensors_metadata(raw, tmp, metadata)
    raw.unlink()
    publish_file(tmp, dst)


def publish_file(tmp: Path, dst: Path) -> None:
    """Move a finished temp file into place. Atomic when both are on one filesystem; across
    bind mounts (``docker run -v``) rename fails with EXDEV and we copy instead, which is
    only non-atomic on a destination nobody syncs mid-copy."""
    try:
        os.replace(tmp, dst)
    except OSError:
        shutil.copy2(tmp, dst)
        tmp.unlink()


class Syncer:
    """Polls the trainer's save_dir and demo dir, publishes new files."""

    def __init__(self, cfg: dict, save_dir: Path, demo_dir: Path, step_offset: int, s3: S3Sink, meta: "RunMeta"):
        self.cfg = cfg
        self.identity = cfg.get("_identity") or run_identity(cfg)
        self.partial_dir = Path(cfg["work_dir"]) / "partial"
        self.save_dir = save_dir
        self.demo_dir = demo_dir
        self.offset = step_offset
        self.s3 = s3
        self.meta = meta
        self.ckpt_dir = Path(cfg["checkpoint_dir"])
        self.out_demo_dir = Path(cfg["output_data_dir"]) / "demos"
        self._sizes: dict[Path, int] = {}
        self._done: set[Path] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _stable(self, p: Path, final: bool) -> bool:
        size = p.stat().st_size
        prev = self._sizes.get(p)
        self._sizes[p] = size
        return final or (prev is not None and prev == size and size > 0)

    def poll(self, final: bool = False) -> None:
        with self._lock:
            changed = False
            for ckpt in sorted(self.save_dir.rglob("*.ckpt")):
                if ckpt in self._done or not self._stable(ckpt, final):
                    continue
                local = lightning_step(ckpt)
                if local is None:
                    continue
                step = self.offset + local
                dst = self.ckpt_dir / ckpt_name(self.identity["prefix"], step)
                try:
                    convert_ckpt(ckpt, dst, identity_metadata(self.identity, step), self.partial_dir)
                except Exception as e:
                    if final:
                        raise
                    print(f"[ravi] checkpoint {ckpt} not ready ({e}); retrying", flush=True)
                    continue
                # The Lightning .ckpt holds the same LoRA tensors in fp32; the .safetensors is
                # fp16 (save_lora_safetensors casts, lora/utils.py:275-285). Resume from the .ckpt.
                fp32 = self.ckpt_dir / ckpt_name(self.identity["prefix"], step, "ckpt")
                self.partial_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.partial_dir / f".{fp32.name}.partial"
                shutil.copy2(ckpt, tmp)
                publish_file(tmp, fp32)
                uri = self.s3.upload(dst, f"checkpoints/{dst.name}")
                uri_fp32 = self.s3.upload(fp32, f"checkpoints/{fp32.name}")
                self.meta.add_checkpoint(step, dst, uri, fp32, uri_fp32)
                print(f"[ravi] checkpoint step {step}: {dst}" + (f" -> {uri}" if uri else ""), flush=True)
                self._done.add(ckpt)
                changed = True
            for wav in sorted(self.demo_dir.glob("demo_*.wav")):
                if wav in self._done or not self._stable(wav, final):
                    continue
                m = re.match(r"^demo_cfg_(.+)_(\d+)\.wav$", wav.name)
                name = (
                    f"demo_step{self.offset + int(m.group(2)):06d}_cfg{m.group(1)}.wav" if m else wav.name
                )
                self.out_demo_dir.mkdir(parents=True, exist_ok=True)
                dst = self.out_demo_dir / name
                shutil.copy2(wav, dst)
                uri = self.s3.upload(dst, f"runs/{self.cfg['run_id']}/demos/{name}")
                self.meta.data["demos"].append({"file": str(dst), "s3": uri})
                self._done.add(wav)
                changed = True
            if changed or final:
                self.meta.refresh_losses(self.save_dir, self.offset)
                self.meta.write(self.s3)

    def _loop(self) -> None:
        while not self._stop.wait(self.cfg["poll_seconds"]):
            try:
                self.poll()
            except Exception:
                traceback.print_exc()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)


# --------------------------------------------------------------------------- run meta


def read_loss_curve(save_dir: Path, offset: int) -> list[dict]:
    """Rows of Lightning's CSVLogger metrics.csv (train_lora.py:217-219). Upstream's
    StaggeredLogger writes averaged metrics every log_every steps with
    step=global_step (stable_audio_3/training/utils.py:183-247)."""
    rows = []
    for f in sorted(save_dir.rglob("metrics.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                vals = {k: float(v) for k, v in r.items() if k.startswith("train/") and v not in ("", None)}
                if vals and r.get("step") not in (None, ""):
                    rows.append({"step": offset + int(float(r["step"])), **vals})
    return rows


LOSS_RE = re.compile(r"train/loss=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def losses_from_log(text: str) -> list[float]:
    """Fallback: progress-bar values (no step numbers)."""
    return [float(x) for x in LOSS_RE.findall(text)]


class RunMeta:
    def __init__(self, cfg: dict, path_list: list[Path]):
        self.paths = path_list
        public_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        self.data: dict[str, Any] = {
            "arm": cfg["arm"],
            "run_id": cfg["run_id"],
            "status": "starting",
            "sa3_commit": sa3_commit(cfg["sa3_root"]),
            "ravi_git_sha": os.environ.get("RAVI_GIT_SHA"),
            "image": os.environ.get("RAVI_IMAGE_TAG"),
            "config": public_cfg,
            "config_sources": cfg.get("_sources"),
            "on_sagemaker": cfg.get("_on_sagemaker"),
            "start_time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "end_time": None,
            "wall_seconds": None,
            "checkpoints": [],
            "demos": [],
            "loss_curve": [],
        }
        self._t0 = time.time()

    def add_checkpoint(self, step: int, path: Path, uri: str | None,
                       ckpt: Path | None = None, ckpt_uri: str | None = None) -> None:
        self.data["checkpoints"].append({"step": step, "path": str(path), "s3": uri,
                                         "ckpt_fp32": str(ckpt) if ckpt else None, "ckpt_fp32_s3": ckpt_uri})

    def refresh_losses(self, save_dir: Path, offset: int) -> None:
        curve = read_loss_curve(save_dir, offset)
        if curve:
            prior = [r for r in self.data.get("loss_curve_prior", []) if r["step"] <= offset]
            self.data["loss_curve"] = prior + curve
            self.data["loss_curve_source"] = "metrics.csv"

    def finish(self, status: str, exit_code: int | None) -> None:
        self.data["status"] = status
        self.data["exit_code"] = exit_code
        self.data["end_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.data["wall_seconds"] = round(time.time() - self._t0, 1)

    def write(self, s3: S3Sink | None = None) -> None:
        text = json.dumps(self.data, indent=2, default=str)
        for p in self.paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.tmp")
            tmp.write_text(text)
            os.replace(tmp, p)
        if s3 is not None and s3.enabled and self.paths:
            s3.upload(self.paths[0], f"runs/{self.data['run_id']}/run_meta.json")


# --------------------------------------------------------------------------- resume


@dataclasses.dataclass
class ResumePlan:
    offset: int = 0  # completed steps
    checkpoint: str | None = None  # local path passed as --lora_checkpoint (fp32 .ckpt if found)
    source: str | None = None
    action: str = "train"  # train | publish
    safetensors: str | None = None  # identity-checked adapter; what "publish" copies


def local_checkpoints(ckpt_dir: Path, run_prefix: str) -> list[tuple[int, Path]]:
    if not ckpt_dir.is_dir():
        return []
    out = []
    for p in ckpt_dir.iterdir():
        step = parse_ckpt_step(p.name, run_prefix)
        if step is not None:
            out.append((step, p))
    return sorted(out)


def find_resume(cfg: dict, s3: S3Sink, work_dir: Path) -> ResumePlan:
    """Pick the checkpoint to warm-start from. Only checkpoints of this run identity
    (arm, seed, config hash) are considered; ``auto`` takes the newest of local and S3."""
    mode = cfg["resume"]
    identity = cfg.get("_identity") or run_identity(cfg)
    prefix = identity["prefix"]
    ckpt_dir = Path(cfg["checkpoint_dir"])
    if mode == "none":
        return ResumePlan()

    if mode in ("auto", "publish"):
        cands: list[tuple[int, int, str]] = []  # (step, local-first tiebreak, source)
        local = local_checkpoints(ckpt_dir, prefix)
        if local:
            cands.append((local[-1][0], 1, str(local[-1][1])))
        remote = s3.latest_checkpoint(prefix)
        if remote:
            cands.append((remote[0], 0, remote[1]))
        if not cands:
            if mode == "publish":
                raise FileNotFoundError(f"resume=publish but no {prefix}_step*.safetensors locally or on S3")
            return ResumePlan()
        step, _, src = max(cands)
        if src.startswith("s3://"):
            path = s3.download(src, ckpt_dir / ckpt_name(prefix, step))
        else:
            path = Path(src)
    else:
        src = str(mode)
        if src.startswith("s3://"):
            path = s3.download(src, work_dir / "resume" / src.rsplit("/", 1)[-1])
        else:
            path = Path(src)
            if not path.is_file():
                raise FileNotFoundError(src)
    step = check_identity(path, identity)
    weights = _fp32_sibling(path, src, s3) or path
    return _resume_action(cfg, mode, ResumePlan(step, str(weights), src, safetensors=str(path)))


def _fp32_sibling(path: Path, src: str, s3: S3Sink) -> Path | None:
    """The ``.ckpt`` next to a ``.safetensors`` (same stem), fetched from S3 if needed.
    Its tensors are fp32; the .safetensors copy is fp16."""
    local = path.with_suffix(".ckpt")
    if local.is_file():
        return local
    if src.startswith("s3://") and src.endswith(".safetensors"):
        uri = src[: -len(".safetensors")] + ".ckpt"
        try:
            if s3.exists(uri):
                return s3.download(uri, local)
        except Exception as e:
            print(f"[ravi] WARNING: could not fetch {uri} ({e}); resuming from the fp16 .safetensors", flush=True)
    return None


def _resume_action(cfg: dict, mode: str, plan: ResumePlan) -> ResumePlan:
    steps = cfg["steps"]
    if plan.offset > steps:
        raise ValueError(f"checkpoint {plan.source} is at step {plan.offset}, past steps={steps}")
    if plan.offset == steps:
        if mode != "publish":
            raise ValueError(
                f"{plan.source} already reached steps={steps}. Set resume=publish to re-publish it as "
                "the final adapter, or resume=none to train again from scratch"
            )
        plan.action = "publish"
    elif mode == "publish":
        raise ValueError(f"resume=publish needs a checkpoint at step {steps}; newest is {plan.offset}")
    if plan.offset % cfg["checkpoint_every"]:
        raise ValueError(f"resume step {plan.offset} is not a multiple of checkpoint_every")
    return plan


def check_resume_compat(path: str, cfg: dict) -> None:
    """Refuse to warm-start from an adapter trained with a different LoRA layout."""
    if not str(path).endswith(".safetensors"):
        return
    saved = json.loads(read_safetensors_metadata(path).get("lora_config", "{}"))
    want = {
        "rank": cfg["rank"],
        "adapter_type": cfg["adapter_type"],
        "exclude": exclude_patterns(cfg) or None,
        "include": cfg["include"] or None,
    }
    diffs = {k: (saved.get(k), v) for k, v in want.items() if k in saved and saved.get(k) != v}
    if diffs:
        raise ValueError(f"resume checkpoint {path} has a different LoRA config (saved, wanted): {diffs}")


# --------------------------------------------------------------------------- main


def _write_failure(msg: str) -> None:
    failure = Path("/opt/ml/output/failure")
    try:
        if failure.parent.is_dir():
            failure.write_text(msg[-1000:])
    except OSError:
        pass


def run(cfg: dict, dry_run: bool = False) -> int:
    cfg = resolve(cfg)
    identity = cfg["_identity"] = run_identity(cfg)
    work_dir = Path(cfg["work_dir"]) / str(cfg["run_id"])
    out_dir = Path(cfg["output_data_dir"])
    s3 = S3Sink(cfg["s3_out"], cfg["arm"], cfg["aws_region"]) if not dry_run else S3Sink(None, cfg["arm"], None)

    data_summary = scan_dataset(cfg)
    print(f"[ravi] data: {json.dumps(data_summary)}", flush=True)

    plan = ResumePlan() if dry_run else find_resume(cfg, s3, work_dir)
    offset, lora_ckpt, resume_src = plan.offset, plan.checkpoint, plan.source
    if plan.safetensors:
        check_resume_compat(plan.safetensors, cfg)
    remaining = cfg["steps"] - offset
    save_dir = work_dir / f"segment_from_{offset:06d}"
    demo_dir = save_dir / "demos"
    train_argv = build_train_lora_argv(cfg, save_dir, max(remaining, 0), cfg["seed"] + offset, lora_ckpt)
    cmd = build_child_cmd(cfg, train_argv)

    if dry_run:
        print(json.dumps({"identity": identity, "resume": cfg["resume"],
                          "train_lora_argv": train_argv, "child_cmd": cmd}, indent=2))
        return 0

    for d in (save_dir, demo_dir, Path(cfg["checkpoint_dir"]), Path(cfg["model_dir"]), out_dir):
        d.mkdir(parents=True, exist_ok=True)
    meta = RunMeta(cfg, [out_dir / "run_meta.json", Path(cfg["model_dir"]) / "run_meta.json"])
    prev_meta = Path(cfg["checkpoint_dir"]) / f"{identity['prefix']}_run_meta.json"
    if offset and prev_meta.is_file():
        try:
            meta.data["loss_curve_prior"] = json.loads(prev_meta.read_text()).get("loss_curve", [])
        except (OSError, json.JSONDecodeError):
            pass
    meta.paths.append(prev_meta)
    meta.data.update(
        data=data_summary,
        identity={k: identity[k] for k in ("prefix", "cfg_hash", "fields")},
        resume_mode=cfg["resume"],
        resume={"completed_steps": offset, "from": resume_src, "action": plan.action,
                "mode": "weights-only warm start"} if offset else None,
        train_lora_argv=train_argv,
        child_cmd=cmd,
        effective_seed=cfg["seed"] + offset,
        duration_conditioner_excluded=cfg["exclude_duration_conditioner"],
        upstream_defaults_in_effect={
            "cfg_dropout_prob": 0.1,
            "inpaint_mask_type_probabilities": cfg["inpaint_mask_type_probabilities"] or [0.1, 0.8, 0.1],
            "mask_padding_attention": True,
            "silence_extension_scale_seconds": 4.0,
            "mask_loss_weight": 1.0,
            "random_crop_of_longer_clips": True,
            "phase_flip_p": 0.5,
            "optimizer": "AdamW(wd=0.01, betas=(0.9, 0.95)), constant LR",
            "precision": "bf16-mixed",
        },
    )

    try:
        meta.data["env"] = check_environment(cfg)
        print(f"[ravi] env: {json.dumps(meta.data['env'])}", flush=True)
        if plan.action == "publish":  # only reachable with resume=publish (see _resume_action)
            print(f"[ravi] resume=publish: {plan.safetensors} is at step {offset}; publishing it as the final adapter",
                  flush=True)
            final = Path(plan.safetensors)
            exit_code = 0
        else:
            child_env = dict(os.environ)
            ensure_hf_token(cfg, child_env)
            pin_single_gpu(child_env)
            ravi_root = str(Path(__file__).resolve().parents[2])
            child_env["PYTHONPATH"] = os.pathsep.join(p for p in (ravi_root, child_env.get("PYTHONPATH")) if p)
            child_env.setdefault("PYTHONUNBUFFERED", "1")
            child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
            meta.data["status"] = "running"
            meta.write(s3)
            exit_code, got_signal = _run_child(
                cmd, child_env, demo_dir, out_dir / "train.log", cfg, save_dir, offset, s3, meta
            )
            if got_signal is not None:
                raise Interrupted(got_signal, exit_code)
            if exit_code != 0:
                raise RuntimeError(f"train_lora.py exited with code {exit_code}")
            final = Path(cfg["checkpoint_dir"]) / ckpt_name(identity["prefix"], cfg["steps"])
            if not final.is_file():
                raise RuntimeError(f"training finished but {final} was not produced")
        shutil.copy2(final, Path(cfg["model_dir"]) / "adapter.safetensors")
        meta.data["final_adapter"] = {
            "path": str(Path(cfg["model_dir"]) / "adapter.safetensors"),
            "step": cfg["steps"],
            "s3": s3.uri(f"checkpoints/{final.name}"),
        }
        for f in save_dir.rglob("metrics.csv"):
            shutil.copy2(f, out_dir / f"metrics_from_{offset:06d}.csv")
        if not meta.data["loss_curve"]:
            log = out_dir / "train.log"
            if log.is_file():
                meta.data["loss_curve_progress_bar"] = losses_from_log(log.read_text(errors="replace"))
        _upload_log(s3, out_dir, cfg)
        status = "completed" if not s3.errors else "completed_with_s3_errors"
        meta.data["s3_errors"] = s3.errors
        meta.finish(status, exit_code)
        meta.write(s3)
        if s3.errors:
            # Training succeeded and every file is in the local checkpoint/model dirs (which
            # SageMaker syncs/packs itself), so the job is not failed over an upload.
            bar = "!" * 78
            print(f"[ravi] {bar}\n[ravi] WARNING: training completed but {len(s3.errors)} S3 uploads failed:", flush=True)
            for err in s3.errors:
                print(f"[ravi]   {err}", flush=True)
            print(f"[ravi] the files are in {cfg['checkpoint_dir']} and {cfg['model_dir']}\n[ravi] {bar}", flush=True)
        return 0
    except BaseException as e:
        interrupted = isinstance(e, Interrupted)
        meta.data["error"] = f"{type(e).__name__}: {e}"
        meta.finish("interrupted" if interrupted else "failed",
                    e.exit_code if interrupted else getattr(e, "code", None))
        _upload_log(s3, out_dir, cfg)
        meta.write(s3)
        _write_failure(meta.data["error"])
        raise


class Interrupted(RuntimeError):
    """The wrapper received SIGTERM/SIGINT (SageMaker stop, spot interruption, docker stop)."""

    def __init__(self, signum: int, exit_code: int | None):
        self.signum, self.exit_code = signum, exit_code
        super().__init__(f"interrupted by signal {signum}; trainer exit code {exit_code}")


def _upload_log(s3: S3Sink, out_dir: Path, cfg: dict) -> None:
    log = out_dir / "train.log"
    if s3.enabled and log.is_file():
        s3.upload(log, f"runs/{cfg['run_id']}/train.log")


KILL_GRACE_SECONDS = 30


def _run_child(cmd, env, cwd, log_path, cfg, save_dir, offset, s3, meta) -> tuple[int, int | None]:
    """Run the trainer, tee its output, publish checkpoints while it runs.

    Returns ``(exit_code, signal_received_or_None)``. On SIGTERM/SIGINT the trainer gets
    SIGTERM and, if still alive after :data:`KILL_GRACE_SECONDS`, SIGKILL. If the wrapper
    itself fails, the trainer is killed rather than orphaned."""
    print(f"[ravi] launching: {' '.join(cmd)}", flush=True)
    syncer = Syncer(cfg, save_dir, cwd, offset, s3, meta)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    got_signal: list[int] = []
    killer: list[threading.Timer] = []

    def _kill_if_alive():
        if proc.poll() is None:
            print(f"[ravi] trainer still alive {KILL_GRACE_SECONDS}s after SIGTERM; sending SIGKILL", flush=True)
            proc.kill()

    def _forward(signum, _frame):
        print(f"[ravi] received signal {signum}; stopping trainer", flush=True)
        got_signal.append(signum)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            if not killer:
                t = threading.Timer(KILL_GRACE_SECONDS, _kill_if_alive)
                t.daemon = True
                t.start()
                killer.append(t)

    old = {sig: signal.signal(sig, _forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        syncer.start()
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
        if proc.poll() is None:  # the wrapper raised while the trainer was running
            proc.kill()
            proc.wait()
        for t in killer:
            t.cancel()
        syncer.stop()
    sig = got_signal[0] if got_signal else None
    try:
        syncer.poll(final=True)
    except Exception:
        if code == 0 and sig is None:
            raise
        # trainer died mid-write; keep whatever was published already
        traceback.print_exc()
    if sig is not None:
        print(f"[ravi] trainer interrupted (signal {sig}), exit code {code}", flush=True)
    else:
        print(f"[ravi] trainer exited with code {code}", flush=True)
    return code, sig


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":  # SageMaker runs `<entrypoint> train`
        argv = argv[1:]
    p = argparse.ArgumentParser(prog="python -m ravi.train.sa3_lora", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="YAML/JSON run config")
    p.add_argument("--hyperparameters", help=f"hyperparameters.json (default {SM_HP_PATH} if present)")
    p.add_argument("--s3-out", help="s3://bucket/prefix for checkpoints, demos, run_meta")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    p.add_argument("--dry-run", action="store_true", help="validate data and print the argv; no training")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.hyperparameters, args.set, args.s3_out)
    try:
        return run(cfg, dry_run=args.dry_run)
    except Exception as e:
        _write_failure(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
