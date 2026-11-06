# syntax=docker/dockerfile:1.7
#
# rAvI v2: evaluation image (plan Phase 4): CLAP (laion_clap 1.1.7, music checkpoint) and
# MERT-v1-95M (transformers 4.52.4) embeddings, FAD / FAD-inf / KAD, seam / beat / key metrics
# and the paired statistics. Build from the REPO ROOT:
#
#   DOCKER_BUILDKIT=1 docker build -f docker/eval/Dockerfile -t ravi-eval \
#       --build-arg RAVI_GIT_SHA=$(git rev-parse HEAD) .
#
# Run: see docker/eval/README.md. ENTRYPOINT is `python -m`, so the first argument is the
# module: `ravi.eval.run_eval` (the driver) or `ravi.eval.embed --self-test`.
#
# Base image: the captioner's (pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime, Python 3.11,
# torch/torchaudio 2.6.0+cu124 in /opt/conda; same digest, see docker/captioner/Dockerfile).
# CUDA 12.4 covers the L40S (sm_89) on the g6e box and runs on CPU without a GPU (both
# extractors fall back to cpu when torch.cuda.is_available() is False). No flash-attn needed.

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.6.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libsndfile1: soundfile (its wheel bundles one too; the system copy is a fallback).
# tini: PID 1 that forwards SIGTERM/SIGINT.
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

# laion_clap imports torchvision (clap_module/utils.py: torchvision.ops.misc.FrozenBatchNorm2d;
# training/data.py: torchvision.transforms). 0.21.0 is the release paired with torch 2.6.0; pip
# leaves it alone if the base already has it.
RUN pip install --no-deps "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}" \
 && python -c "import torchvision; assert torchvision.__version__.split('+')[0] == '${TORCHVISION_VERSION}', torchvision.__version__"

# --- Everything else, fully pinned, no resolver (see the header of requirements.txt).
COPY docker/eval/requirements.txt /opt/eval/requirements.txt
RUN pip install --no-deps -r /opt/eval/requirements.txt

# --- Build-time sanity 1: every requirement of the packages we installed (and of torch) is met.
RUN python - <<'EOF'
import importlib.metadata as md, re
from packaging.requirements import Requirement
names = [re.split(r"[=<>!~ ]", l.strip())[0] for l in open("/opt/eval/requirements.txt")
         if l.strip() and not l.lstrip().startswith("#")]
names += ["torch", "torchaudio", "torchvision"]
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

# --- Build-time sanity 2: the APIs ravi.eval relies on exist (no GPU, no weights).
RUN python - <<'EOF'
import inspect
import laion_clap, librosa, scipy, scipy.stats, transformers
from laion_clap import CLAP_Module
assert transformers.__version__ == "4.52.4", transformers.__version__
assert librosa.__version__ == "0.11.0", librosa.__version__
sig = inspect.signature(CLAP_Module.__init__).parameters
assert {"enable_fusion", "device", "amodel"} <= set(sig), sig
for m in ("load_ckpt", "get_audio_embedding_from_data", "get_text_embedding"):
    assert hasattr(CLAP_Module, m), m
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor  # noqa: F401
from transformers.models.hubert.modeling_hubert import (  # noqa: F401  (MERT remote code)
    HubertFeatureEncoder, HubertModel, HubertEncoderStableLayerNorm, HubertEncoder,
    HubertEncoderLayer, HubertPositionalConvEmbedding, HubertAttention, HubertFeedForward)
assert "bpm" in inspect.signature(librosa.beat.beat_track).parameters
assert "aggregate" in inspect.signature(librosa.feature.tempo).parameters
assert {"zero_method", "alternative", "method"} <= set(inspect.signature(scipy.stats.wilcoxon).parameters)
print("laion_clap ok | transformers", transformers.__version__, "| librosa", librosa.__version__,
      "| scipy", scipy.__version__)
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
RUN python -m ravi.eval.run_eval --help > /dev/null && python -m ravi.eval.embed --help > /dev/null

ENTRYPOINT ["tini", "--", "python", "-m"]
CMD ["ravi.eval.run_eval", "--help"]
