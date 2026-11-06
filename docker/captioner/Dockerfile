# syntax=docker/dockerfile:1.7
#
# rAvI v2: auto-captioner (plan Phase 1 step 6). MiDashengLM-7B (default) and Qwen2.5-Omni-7B
# (fallback) under transformers 4.52.4. Build from the REPO ROOT:
#
#   DOCKER_BUILDKIT=1 docker build -f docker/captioner/Dockerfile -t ravi-captioner \
#       --build-arg RAVI_GIT_SHA=$(git rev-parse HEAD) .
#
# Run: see docker/captioner/README.md. ENTRYPOINT is `python -m ravi.captions.autocaption`,
# so arguments go straight to the script.
#
# Base image: transformers 4.52.4 needs torch >= 2.1 (no upper bound); MiDashengLM's remote code
# also imports torchaudio.functional. pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime ships
# Python 3.11 with torch 2.6.0+cu124 and torchaudio 2.6.0+cu124 in /opt/conda (checked in the
# image layer), so nothing torch-related is installed here. CUDA 12.4 covers the L40S (sm_89);
# the DLAMI driver (>= 550) is new enough. No flash-attn: both models run on SDPA.
# Digest: the linux/amd64 manifest of the tag on Docker Hub (2026-10-14; the tag is single-arch).

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.6.0
ARG TORCHAUDIO_VERSION=2.6.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libsndfile1: soundfile (its wheel bundles one too; the system copy is a fallback).
# tini: PID 1 that forwards SIGTERM/SIGINT to python, so `docker stop` / Ctrl-C end the run
# between fsynced JSONL lines (a torn last line is repaired by --resume anyway).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python - <<EOF
import sys, torch, torchaudio
assert sys.version_info[:2] == (3, 11), sys.version  # requirements.txt is resolved for 3.11
assert torch.__version__.split("+")[0] == "${TORCH_VERSION}", torch.__version__
assert torchaudio.__version__.split("+")[0] == "${TORCHAUDIO_VERSION}", torchaudio.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
EOF

# --- Everything else, fully pinned, no resolver (see the header of requirements.txt).
COPY docker/captioner/requirements.txt /opt/captioner/requirements.txt
RUN pip install --no-deps -r /opt/captioner/requirements.txt

# --- Build-time sanity 1: every requirement of the packages we installed (and of torch) is met.
#     (A global `pip check` would also judge the base image's conda tooling.)
RUN python - <<'EOF'
import importlib.metadata as md, re
from packaging.requirements import Requirement
names = [re.split(r"[=<>!~ ]", l.strip())[0] for l in open("/opt/captioner/requirements.txt")
         if l.strip() and not l.lstrip().startswith("#")]
names += ["torch", "torchaudio"]
bad = []
for n in names:
    for r in md.requires(n) or []:
        req = Requirement(r)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            v = md.version(req.name)
        except md.PackageNotFoundError:
            bad.append(f"{n}: missing {req}"); continue
        if req.specifier and not req.specifier.contains(v, prereleases=True):
            bad.append(f"{n}: needs {req}, have {v}")
assert not bad, "\n".join(bad)
print(f"dependency check ok ({len(names)} packages)")
EOF

# --- Build-time sanity 2: the symbols both models need exist in this transformers (no GPU,
#     no weights). MiDashengLM's remote code (modeling/configuration/processing_midashenglm.py)
#     imports the Qwen2.5-Omni text classes, can_return_tuple and AllKwargsForChatTemplate.
RUN python - <<'EOF'
import importlib.metadata as md
import librosa, soundfile, scipy.signal, torchaudio.functional  # noqa: F401
import transformers
assert transformers.__version__ == "4.52.4", transformers.__version__
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,  # noqa: F401
                          Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
                          Wav2Vec2FeatureExtractor, Qwen2TokenizerFast)
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig  # noqa: F401
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel  # noqa: F401
from transformers.processing_utils import AllKwargsForChatTemplate  # noqa: F401
from transformers.utils import can_return_tuple  # noqa: F401
assert hasattr(Qwen2_5OmniForConditionalGeneration, "disable_talker")
print("transformers", transformers.__version__, "| accelerate", md.version("accelerate"),
      "| tokenizers", md.version("tokenizers"), "| huggingface-hub", md.version("huggingface-hub"),
      "| librosa", md.version("librosa"), "| numpy", md.version("numpy"))
EOF

ARG RAVI_GIT_SHA=unknown
ENV HF_HOME=/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    RAVI_GIT_SHA=${RAVI_GIT_SHA} \
    PYTHONPATH=/app
VOLUME ["/hf"]

COPY ravi /app/ravi
WORKDIR /app
RUN python -m ravi.captions.autocaption --help > /dev/null

ENTRYPOINT ["tini", "--", "python", "-m", "ravi.captions.autocaption"]
CMD ["--help"]
