#!/usr/bin/env bash
# Build the SA3 LoRA training image for linux/amd64 and push it to ECR.
#
#   docker/sa3-train/build_and_push.sh [tag]          # default tag: <ravi git sha>-sa3-<sa3 sha[:8]>
#   PUSH=0 docker/sa3-train/build_and_push.sh         # build + load locally, no push
#   FLASH_ATTN_WHEEL= docker/sa3-train/build_and_push.sh   # build flash-attn from source (devel base)
#   PUSH_LATEST=1 docker/sa3-train/build_and_push.sh  # also move the :latest tag
#
# A push refuses a dirty ravi/ or docker/sa3-train/ (the image would not match its SHA tag)
# unless ALLOW_DIRTY=1; the tag then carries a -dirty suffix.
#
# Env: PROFILE (default dev-sso; unset it on an EC2 box with an instance role), REGION, ACCOUNT,
# REPO, BASE_IMAGE (default: the Dockerfile's digest-pinned cudnn-runtime image).
set -euo pipefail

ACCOUNT=${ACCOUNT:-297416720456}
REGION=${REGION:-us-east-1}
REPO=${REPO:-ravi-sa3-train}
PROFILE=${PROFILE-dev-sso}
PUSH=${PUSH:-1}
PUSH_LATEST=${PUSH_LATEST:-0}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}
SA3_COMMIT=${SA3_COMMIT:-77e0398c26f7d22d801ec1bb036307e8b231f741}
DEVEL_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04@sha256:b3e7fba84d169f46939f00c25be7d016f712a8d651f4756d6a55e693d84d94f2

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
RAVI_SHA=$(git rev-parse --short=12 HEAD)
DIRTY=$(git status --porcelain -- ravi docker/sa3-train)
if [[ -n "$DIRTY" ]]; then
  if [[ "$PUSH" == "1" && "$ALLOW_DIRTY" != "1" ]]; then
    echo "refusing to push: uncommitted changes in ravi/ or docker/sa3-train/ (set ALLOW_DIRTY=1 to override):" >&2
    echo "$DIRTY" >&2
    exit 1
  fi
  RAVI_SHA="${RAVI_SHA}-dirty"
fi
TAG=${1:-${RAVI_SHA}-sa3-${SA3_COMMIT:0:8}}
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

AWS=(aws --region "$REGION")
[[ -n "$PROFILE" ]] && AWS+=(--profile "$PROFILE")

BUILD_ARGS=(--build-arg "SA3_COMMIT=${SA3_COMMIT}" --build-arg "RAVI_GIT_SHA=${RAVI_SHA}" --build-arg "RAVI_IMAGE_TAG=${IMAGE}")
if [[ -n "${FLASH_ATTN_WHEEL+x}" ]]; then
  BUILD_ARGS+=(--build-arg "FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL}")
  # a source build needs nvcc, which only the devel base has
  [[ -z "$FLASH_ATTN_WHEEL" && -z "${BASE_IMAGE:-}" ]] && BASE_IMAGE=$DEVEL_IMAGE
fi
[[ -n "${BASE_IMAGE:-}" ]] && BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
[[ -n "${MAX_JOBS:-}" ]] && BUILD_ARGS+=(--build-arg "MAX_JOBS=${MAX_JOBS}")

if [[ "$PUSH" == "1" ]]; then
  "${AWS[@]}" ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
    || "${AWS[@]}" ecr create-repository --repository-name "$REPO" \
         --image-scanning-configuration scanOnPush=true >/dev/null
  "${AWS[@]}" ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
  OUTPUT=(--push)
else
  OUTPUT=(--load)
fi

TAGS=(-t "$IMAGE")
[[ "$PUSH_LATEST" == "1" ]] && TAGS+=(-t "${REGISTRY}/${REPO}:latest")

# --provenance=false: SageMaker rejects the multi-manifest index that buildx attestations create.
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --progress=plain \
  -f docker/sa3-train/Dockerfile \
  "${BUILD_ARGS[@]}" \
  "${TAGS[@]}" \
  "${OUTPUT[@]}" \
  .

echo "$IMAGE"
