#!/usr/bin/env bash
# Build the SA3 inference (BYOC) image for linux/amd64 and push it to ECR.
#
#   docker/sa3-serve/build_and_push.sh [tag]         # default tag: <git sha>[-dirty]-<UTC yyyymmdd-HHMMSS>
#   PUSH=0 PROFILE= docker/sa3-serve/build_and_push.sh   # build + load locally (EC2 box, instance role)
#   BASE_IMAGE=297416720456.dkr.ecr.us-east-1.amazonaws.com/ravi-sa3-train:<tag> docker/sa3-serve/build_and_push.sh
#
# BASE_IMAGE defaults to the local `ravi-sa3-train:dev` (the image on the GPU box the adapters
# were trained with). The repo/tag naming follows the existing BYOC endpoints in the dev account
# (sagemaker-model-docker-sgmse-dereverb:<sha>-dirty-<yyyymmdd-HHMMSS>).
#
# Env: PROFILE (default dev-sso; unset on an EC2 box with an instance role), REGION, ACCOUNT, REPO.
set -euo pipefail

ACCOUNT=${ACCOUNT:-297416720456}
REGION=${REGION:-us-east-1}
REPO=${REPO:-sagemaker-stable-audio-3-finetuned}
PROFILE=${PROFILE-dev-sso}
PUSH=${PUSH:-1}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}
BASE_IMAGE=${BASE_IMAGE:-ravi-sa3-train:dev}

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/../.." && pwd))
cd "$ROOT"
SHA=$(git rev-parse --short=10 HEAD 2>/dev/null || echo nogit)
if [[ "$SHA" != nogit && -n "$(git status --porcelain -- docker/sa3-serve)" ]]; then
  if [[ "$PUSH" == "1" && "$ALLOW_DIRTY" != "1" ]]; then
    echo "refusing to push: uncommitted changes in docker/sa3-serve/ (ALLOW_DIRTY=1 to override)" >&2
    exit 1
  fi
  SHA="${SHA}-dirty"
fi
TAG=${1:-${SHA}-$(date -u +%Y%m%d-%H%M%S)}
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

AWS=(aws --region "$REGION")
[[ -n "$PROFILE" ]] && AWS+=(--profile "$PROFILE")

if [[ "$PUSH" == "1" ]]; then
  "${AWS[@]}" ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
    || "${AWS[@]}" ecr create-repository --repository-name "$REPO" \
         --image-scanning-configuration scanOnPush=true >/dev/null
  "${AWS[@]}" ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
  OUTPUT=(--push)
else
  OUTPUT=(--load)
fi

# --provenance=false: SageMaker rejects the multi-manifest index that buildx attestations create.
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --progress=plain \
  -f docker/sa3-serve/Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "RAVI_SERVE_GIT_SHA=${SHA}" \
  -t "$IMAGE" \
  "${OUTPUT[@]}" \
  .

echo "$IMAGE"
