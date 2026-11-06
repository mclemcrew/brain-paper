# syntax=docker/dockerfile:1.7
#
# rAvI v2: Stable Audio 3 inference image for SageMaker (bring-your-own-container).
# Build from the REPO ROOT:
#
#   docker buildx build --platform linux/amd64 -f docker/sa3-serve/Dockerfile \
#       --build-arg BASE_IMAGE=<ravi-sa3-train image> -t sagemaker-stable-audio-3-finetuned .
#
# Why this image and not the stock pytorch-inference:2.6.0-gpu-py312-cu124 DLC: stable-audio-3
# (77e0398c) pins torch==2.7.1 / torchaudio==2.7.1, Python 3.10 (.python-version) and needs
# Flash-Attention 2. The training image already has exactly that environment, installed from
# SA3's uv.lock, and it is the environment the adapters were trained and the inpainting splice
# was verified in (docker/sa3-train/SMOKE.md). This image adds only a web server.
#
# The handler is NOT copied in: SageMaker extracts model.tar.gz (uapkg
# Sagemaker_Deployment/stable-audio-3-finetuned, built by its build_artifact.sh) to
# /opt/ml/model, and server.py imports /opt/ml/model/code/inference.py.

ARG BASE_IMAGE=ravi-sa3-train:dev
FROM ${BASE_IMAGE}

# Web server, constrained by the SA3 lock so nothing in the verified environment moves.
ARG FASTAPI_VERSION=0.115.12
ARG UVICORN_VERSION=0.34.0
RUN uv pip install --python /opt/sa3/.venv/bin/python -c /opt/sa3/constraints.txt \
        "fastapi==${FASTAPI_VERSION}" "uvicorn==${UVICORN_VERSION}" \
 && rm -rf /tmp/uv-cache

COPY docker/sa3-serve/server.py /opt/sa3-serve/server.py

# HF_HUB_OFFLINE stays unset: the handler sets it itself when it loads local base weights and
# only goes online for its last-resort Hugging Face fallback.
ENV SM_MODEL_DIR=/opt/ml/model \
    SA3_HANDLER_DIR=/opt/ml/model/code \
    SAGEMAKER_BIND_TO_PORT=8080 \
    HF_HOME=/tmp/hf_home

# Build-time sanity: the same imports the handler needs (flash-attn's CUDA kernels need a GPU,
# so the handler checks them again at start-up and refuses to serve without them).
RUN python - <<'PY'
import sys, torch, fastapi, uvicorn, soundfile, boto3, numpy
import importlib.metadata as md
from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
import stable_audio_3
from stable_audio_3 import StableAudioModel  # noqa: F401
from stable_audio_3.loading_utils import load_diffusion_cond  # noqa: F401
from stable_audio_3.models.lora import remove_lora  # noqa: F401
import stable_audio_3.models.transformer as T
assert T.flash_attn_func is not None, "stable_audio_3 did not pick up flash_attn"
assert torch.__version__.startswith("2.7.1"), torch.__version__
print("python", sys.version.split()[0], "| torch", torch.__version__, "| flash-attn", md.version("flash-attn"),
      "| fastapi", fastapi.__version__, "| uvicorn", uvicorn.__version__, "| soundfile", soundfile.__version__,
      "libsndfile", soundfile.__libsndfile_version__)
PY

EXPOSE 8080
WORKDIR /opt/sa3-serve
ARG RAVI_SERVE_GIT_SHA=unknown
ENV RAVI_SERVE_GIT_SHA=${RAVI_SERVE_GIT_SHA}
# SageMaker runs `docker run <image> serve`; server.py ignores the argument.
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/opt/sa3-serve/server.py"]
