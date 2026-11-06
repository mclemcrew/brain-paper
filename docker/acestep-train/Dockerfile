# syntax=docker/dockerfile:1.7
#
# rAvI v2: ACE-Step 1.5 LoRA training image (EC2 `docker run` and SageMaker Training Jobs).
# Build from the REPO ROOT (the build context must contain ravi/):
#
#   docker buildx build --platform linux/amd64 -f docker/acestep-train/Dockerfile \
#       --build-arg RAVI_GIT_SHA=$(git rev-parse --short HEAD) -t ravi-acestep-train:dev --load .
#
# Base image. ACE-Step pins torch==2.10.0+cu128 / torchaudio==2.10.0+cu128 for Linux x86_64
# (pyproject.toml:14-16, index pytorch-cu128 at :64-67, :84-95) and requires-python
# ">=3.11,<3.13" (pyproject.toml:6). The torch cu128 wheels bring their own CUDA runtime and
# cuDNN as pip packages, so the base only provides the driver-facing CUDA 12.8 layer; this is
# the same base as ACE-Step's own Dockerfile (Dockerfile:28,33: nvidia/cuda:12.8.1-runtime-
# ubuntu22.04). Their Dockerfile is not used directly: it copies the build context as the
# project (Dockerfile:66), takes an unpinned uv:latest (:62), installs Python from the
# deadsnakes PPA (:53-58) and starts Gradio/the REST server. We keep its base and its
# `uv sync --frozen --no-dev` (Dockerfile:71) at a pinned commit, with uv-managed Python 3.11.
# Digest: multi-arch index digest of the tag from Docker Hub (2026-10-14).
#
# Python 3.11, not 3.12: the only Linux flash-attn in uv.lock is the cp311 prebuilt wheel
# flash_attn-2.8.3+cu128torch2.10-cp311 (uv.lock:609-620), pulled in by the vendored
# nano-vllm (acestep/third_parts/nano-vllm/pyproject.toml, marker python_version == '3.11').
# On 3.12 the lock installs no flash-attn and training silently uses sdpa.

ARG BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9
FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM ${BASE_IMAGE}

# ACE-Step 1.5 source, pinned. Recorded in the image env and in every run_meta.json.
ARG ACESTEP_REPO=https://github.com/ace-step/ACE-Step-1.5.git
ARG ACESTEP_COMMIT=ca1e85fe9430179831e6bc6be790c332190a3866
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/acestep/.venv

# ffmpeg: torchaudio 2.10 decodes through torchcodec, which loads the FFmpeg shared libraries
# (ACE-Step's preprocessing reads audio with torchaudio.load, preprocess_audio.py:6;
# Ubuntu 22.04 ships FFmpeg 4.4); libsndfile1: soundfile (source audio for repaint,
# io_audio.py:33); git: the pinned checkout; tini: PID 1 that forwards SIGTERM to the
# wrapper and reaps DataLoader worker zombies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates ffmpeg libsndfile1 tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# --- ACE-Step at the pinned commit (training_v2 and the vendored nano-vllm are not on PyPI).
RUN git clone --filter=blob:none "${ACESTEP_REPO}" /opt/acestep \
 && git -C /opt/acestep checkout --detach "${ACESTEP_COMMIT}" \
 && test "$(git -C /opt/acestep rev-parse HEAD)" = "${ACESTEP_COMMIT}" \
 && echo "${ACESTEP_COMMIT}" > /opt/acestep/.acestep_commit

WORKDIR /opt/acestep

# --- Locked environment, exactly uv.lock (--frozen = never re-resolve). Installs ace-step
#     itself (editable), torch/torchaudio 2.10.0+cu128, peft 0.18.1, lightning 2.6.1,
#     transformers 4.57.6, torchcodec 0.10.0 and flash-attn 2.8.3 (prebuilt cp311 wheel).
#     The export is the constraints file for the wrapper deps below; path and URL
#     requirements (nano-vllm, flash-attn) are dropped from it since nothing below needs them.
RUN uv python install "${PYTHON_VERSION}" \
 && uv sync --frozen --no-dev --python "${PYTHON_VERSION}" \
 && uv export --frozen --no-dev --no-hashes --no-emit-project -o /opt/acestep/constraints.txt \
 && sed -i -e '/^-/d' -e '/^\.\//d' -e '/ @ /d' /opt/acestep/constraints.txt \
 && rm -rf /tmp/uv-cache

# --- Wrapper deps, constrained so nothing in the ACE-Step lock moves. pyyaml (6.0.3) and
#     huggingface-hub (0.36.0) are already in the lock; boto3 is wrapper-only (S3 sync).
RUN uv pip install --python /opt/acestep/.venv/bin/python -c /opt/acestep/constraints.txt \
        "boto3==1.40.50" pyyaml \
 && rm -rf /tmp/uv-cache

# ACESTEP_CHECKPOINTS_DIR is where ACE-Step looks for weights (model_downloader.py:344-360,
# init_service_orchestrator.py:116-121) and the wrapper's default weights_dir. Weights are
# not baked in (~10 GB); mount a host dir there or let the wrapper download them.
ENV PATH=/opt/acestep/.venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/acestep/.venv \
    ACESTEP_ROOT=/opt/acestep \
    ACESTEP_COMMIT=${ACESTEP_COMMIT} \
    ACESTEP_CHECKPOINTS_DIR=/opt/acestep/checkpoints \
    HF_HOME=/opt/hf_home \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/ml/code

# Build-time sanity (CPU only: the CUDA kernels need the GPU box, SMOKE.md section 3).
# 1. versions and the flash-attn import (training_v2/model_loader.py:25-50 only checks the
#    import and the GPU capability before asking transformers for flash_attention_2);
# 2. torchaudio.load through torchcodec/FFmpeg on a real file;
# 3. the upstream APIs the wrapper and the repaint check call;
# 4. the training_v2 parser accepts the wrapper's flags (same argv as ravi.train.acestep_lora).
RUN python - <<'EOF'
import dataclasses, importlib.metadata as md, inspect, sys, tempfile
import numpy as np, soundfile as sf, torch, torchaudio
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
import flash_attn  # noqa: F401
from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
want = {"peft": "0.18.1", "lightning": "2.6.1", "transformers": "4.57.6", "torchcodec": "0.10.0",
        "torchaudio": "2.10.0+cu128", "flash-attn": "2.8.3", "boto3": "1.40.50"}
got = {k: md.version(k) for k in want}
assert all(got[k] == v or got[k].startswith(v + "+") for k, v in want.items()), got
with tempfile.NamedTemporaryFile(suffix=".wav") as f:
    sf.write(f.name, np.zeros((4800, 2), dtype="float32"), 48000, subtype="PCM_24")
    wav, sr = torchaudio.load(f.name)
    assert sr == 48000 and tuple(wav.shape) == (2, 4800), (sr, wav.shape)
import peft
from peft import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: F401
import acestep.training_v2.trainer_fixed as tf
assert inspect.isgeneratorfunction(tf.resume_checkpoint)
from acestep.training.dataset_builder import DatasetBuilder
from acestep.training.path_safety import set_safe_root  # noqa: F401
assert hasattr(DatasetBuilder(), "scan_directory")
from acestep.inference import GenerationParams, GenerationConfig, generate_music  # noqa: F401
fields = {f.name for f in dataclasses.fields(GenerationParams)}
assert {"src_audio", "repainting_start", "repainting_end", "chunk_mask_mode", "repaint_mode",
        "repaint_strength", "instrumental", "enable_normalization", "thinking"} <= fields
from acestep.handler import AceStepHandler
assert all(hasattr(AceStepHandler, m) for m in ("initialize_service", "load_lora", "set_use_lora", "set_lora_scale"))
from acestep.core.generation.handler.generate_music import _resolve_repaint_config
assert _resolve_repaint_config("balanced", 0.5) == (0.5, 12, 0.025)
from acestep.training_v2.cli.args import build_fixed_standalone_parser
p = build_fixed_standalone_parser()
a = p.parse_args(["--plain", "--yes", "--checkpoint-dir", "/c", "--model-variant", "turbo", "--device", "cuda",
                  "--precision", "bf16", "--num-devices", "1", "--dataset-dir", "/t", "--num-workers", "4",
                  "--lr", "0.0001", "--batch-size", "1", "--gradient-accumulation", "4", "--epochs", "42",
                  "--warmup-steps", "100", "--weight-decay", "0.01", "--max-grad-norm", "1.0", "--seed", "42",
                  "--shift", "3.0", "--num-inference-steps", "8", "--optimizer-type", "adamw",
                  "--scheduler-type", "cosine", "--gradient-checkpointing", "--adapter-type", "lora",
                  "--rank", "64", "--alpha", "128", "--dropout", "0.1",
                  "--target-modules", "q_proj", "k_proj", "v_proj", "o_proj", "--attention-type", "both",
                  "--output-dir", "/o", "--save-every", "4", "--log-every", "10", "--cfg-ratio", "0.15",
                  "--resume-from", "/r"])
assert (a.rank, a.alpha, a.cfg_ratio, a.epochs, a.save_every, a.gradient_checkpointing, a.yes) == \
    (64, 128, 0.15, 42, 4, True, True), a
b = p.parse_args(["--plain", "--yes", "--checkpoint-dir", "/c", "--model-variant", "turbo", "--device", "cuda",
                  "--precision", "bf16", "--preprocess", "--dataset-json", "/d.json", "--tensor-output", "/t",
                  "--max-duration", "240.0"])
assert b.preprocess and b.dataset_json == "/d.json", b
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "|", " ".join(f"{k} {v}" for k, v in got.items()))
EOF

COPY ravi /opt/ml/code/ravi
WORKDIR /opt/ml/code

# Last, so a new git SHA only rebuilds this metadata layer, not the environment above.
ARG RAVI_GIT_SHA=unknown
ARG RAVI_IMAGE_TAG=unknown
ENV RAVI_GIT_SHA=${RAVI_GIT_SHA} \
    RAVI_IMAGE_TAG=${RAVI_IMAGE_TAG}

# SageMaker runs `docker run <image> train`; the wrapper ignores a leading `train`.
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "ravi.train.acestep_lora"]
