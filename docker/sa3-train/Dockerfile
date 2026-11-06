# syntax=docker/dockerfile:1.7
#
# rAvI v2: Stable Audio 3 LoRA training image (EC2 `docker run` and SageMaker Training Jobs).
# Build from the REPO ROOT (the build context must contain ravi/):
#
#   docker buildx build --platform linux/amd64 -f docker/sa3-train/Dockerfile -t ravi-sa3-train .
#
# Base image choice. SA3 pins torch==2.7.1 / torchaudio==2.7.1 from the cu126 index
# (pyproject.toml, [tool.uv.sources]) and Python 3.10 (.python-version). The suggested
# pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel ships Python 3.11 in /opt/conda (checked:
# opt/conda/lib/python3.11/site-packages/torch in the image layer), so it disagrees with
# .python-version and its torch would go unused once uv installs the locked env. We use the
# matching NVIDIA CUDA 12.6 *runtime* base and let uv install Python 3.10 + the exact locked
# wheels (torch 2.7.1+cu126, which bundle their own CUDA libraries) from uv.lock. flash-attn
# comes as a prebuilt wheel, so no nvcc is needed. Only a flash-attn source build
# (FLASH_ATTN_WHEEL=) needs the devel base; build_and_push.sh switches to it automatically:
#
#   --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04@sha256:b3e7fba84d169f46939f00c25be7d016f712a8d651f4756d6a55e693d84d94f2
#
# Both digests are the multi-arch index digests from Docker Hub (2026-10-14).

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:5f0d2d827f6436b3cb7468fd8acbdc8c1d41261614e579ae49afe6141da51133
FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM ${BASE_IMAGE}

# Stable Audio 3 source, pinned. Recorded in the image env and in every run_meta.json.
ARG SA3_REPO=https://github.com/Stability-AI/stable-audio-3.git
ARG SA3_COMMIT=77e0398c26f7d22d801ec1bb036307e8b231f741
ARG PYTHON_VERSION=3.10
# Prebuilt Flash-Attention 2 wheel for cu126 / torch 2.7 / cp310: the exact URL the SA3
# README ("Flash Attention" section) recommends. Set to "" to build from source instead.
ARG FLASH_ATTN_WHEEL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
ARG FLASH_ATTN_VERSION=2.6.3
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/sa3/.venv

# ffmpeg: torchaudio's decoding backend (Ubuntu 22.04 ships FFmpeg 4.4); libsndfile1: soundfile;
# tini: PID 1 that forwards SIGTERM to the wrapper and reaps DataLoader worker zombies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates ffmpeg libsndfile1 tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# --- Stable Audio 3 at the pinned commit (scripts/train_lora.py is not part of the wheel,
#     so we need the checkout, not just `pip install git+...`).
RUN git clone --filter=blob:none "${SA3_REPO}" /opt/sa3 \
 && git -C /opt/sa3 checkout --detach "${SA3_COMMIT}" \
 && test "$(git -C /opt/sa3 rev-parse HEAD)" = "${SA3_COMMIT}" \
 && test "$(cat /opt/sa3/.python-version)" = "${PYTHON_VERSION}" \
 && echo "${SA3_COMMIT}" > /opt/sa3/.sa3_commit

WORKDIR /opt/sa3

# --- Locked environment: stable-audio-3 + the `lora` extra (pytorch_lightning==2.5.5, dill),
#     torch/torchaudio 2.7.1+cu126 exactly as in uv.lock. --frozen = never re-resolve.
RUN uv python install "${PYTHON_VERSION}" \
 && uv sync --frozen --no-dev --extra lora --python "${PYTHON_VERSION}" \
 && uv export --frozen --no-dev --extra lora --no-hashes --no-emit-project -o /opt/sa3/constraints.txt \
 && sed -i '/^-/d' /opt/sa3/constraints.txt \
 && rm -rf /tmp/uv-cache

# --- Flash-Attention 2 (not declared in SA3's pyproject; README says Medium requires it).
#     Source-build fallback follows the README: no build isolation, MAX_JOBS to bound RAM.
#     flash-attn 2.6.x compiles sm_80 + sm_90 itself; sm_80 SASS runs on A10G (8.6) and L40S (8.9).
RUN if [ -n "${FLASH_ATTN_WHEEL}" ]; then \
        uv pip install --python /opt/sa3/.venv/bin/python --no-deps "${FLASH_ATTN_WHEEL}"; \
    else \
        command -v nvcc >/dev/null || { echo "flash-attn source build needs nvcc: pass --build-arg BASE_IMAGE=<cudnn-devel image>" >&2; exit 1; } \
     && uv pip install --python /opt/sa3/.venv/bin/python -c /opt/sa3/constraints.txt ninja packaging psutil setuptools wheel \
     && MAX_JOBS="${MAX_JOBS}" FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
        uv pip install --python /opt/sa3/.venv/bin/python --no-build-isolation --no-deps \
            --no-binary flash-attn "flash-attn==${FLASH_ATTN_VERSION}"; \
    fi \
 && rm -rf /tmp/uv-cache

# --- Wrapper deps, constrained so nothing in the SA3 lock moves. pyyaml is unpinned here on
#     purpose: it is already in SA3's lock (6.0.3, via huggingface-hub / lightning), so the
#     constraints file decides its version; pinning another version makes the solve fail.
RUN uv pip install --python /opt/sa3/.venv/bin/python -c /opt/sa3/constraints.txt \
        "boto3==1.40.50" pyyaml "matplotlib>=3.10.8" wandb pillow \
 && rm -rf /tmp/uv-cache

ENV PATH=/opt/sa3/.venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/sa3/.venv \
    SA3_ROOT=/opt/sa3 \
    SA3_COMMIT=${SA3_COMMIT} \
    HF_HOME=/opt/hf_home \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/opt/ml/code

# Build-time sanity. The CUDA kernels of flash-attn cannot be exercised without a GPU, but
# the import is what upstream does (stable_audio_3/models/transformer.py:22-36) and what fails
# on an ABI mismatch, so it is checked here; ravi.train.sa3_lora repeats it at start-up and
# refuses to train if it fails (upstream would silently fall back).
RUN python - <<'EOF'
import importlib.metadata as md, sys
import torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
import pytorch_lightning, dill, torchaudio, boto3, yaml, safetensors, matplotlib, wandb, PIL  # noqa: F401
assert pytorch_lightning.__version__ == "2.5.5", pytorch_lightning.__version__
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI, "| flash-attn", md.version("flash-attn"),
      "| lightning", pytorch_lightning.__version__)
EOF

COPY ravi /opt/ml/code/ravi
WORKDIR /opt/ml/code

# Last, so a new git SHA only rebuilds this metadata layer, not the environment above.
ARG RAVI_GIT_SHA=unknown
ARG RAVI_IMAGE_TAG=unknown
ENV RAVI_GIT_SHA=${RAVI_GIT_SHA} \
    RAVI_IMAGE_TAG=${RAVI_IMAGE_TAG}

# SageMaker runs `docker run <image> train`; the wrapper ignores a leading `train`.
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "ravi.train.sa3_lora"]
