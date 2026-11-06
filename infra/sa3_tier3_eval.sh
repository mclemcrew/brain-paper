#!/usr/bin/env bash
# SA3 tier-3 evaluation on the GPU box: waits for the arm-C run (TIER3_C_DONE in
# ~/ravi/logs/sa3_tier3_C.log, TIER3_C_EXIT=0 in ~/ravi/logs/C_tier3.log), backs up the ACE-Step
# adapters, links the B 2k/4k checkpoints and the tier-1 controls, then runs the grid
# (configs/eval/tier3.yaml: D6k, C6k, B2k, B4k + base, B6k, anchor_lower) and run_eval on
# ~/ravi/eval/tier3, and backs up the metric tables to S3. Ends with TIER3_EVAL_DONE (or
# TIER3_EVAL_FAILED: <reason>). Does not stop the instance. Safe to rerun: the grid resumes
# (outputs with a sidecar are skipped) and every link is replaced in place.
#
#   cp infra/sa3_tier3_eval.sh ~/ravi/runs/ && \
#     nohup ~/ravi/runs/sa3_tier3_eval.sh > ~/ravi/logs/sa3_tier3_eval.log 2>&1 < /dev/null &
#
# Logs: ~/ravi/logs/tier3_grid.log (GRID3_EXIT=), ~/ravi/logs/tier3_run_eval.log
# (TIER3_RUN_EVAL_EXIT=). Host -> container paths: ~/ravi/runs -> /runs (ro), ~/ravi/eval -> /eval
# (grid) and /app/eval (run_eval); every link below is relative so it resolves on both sides.
set -u
cd "$HOME/ravi" || exit 1
L=$HOME/ravi/logs
R=$HOME/ravi/runs
E=$HOME/ravi/eval
S3=s3://uaudio-model-training-checkpoints/ravi-v2
mkdir -p "$L"
die() { echo "TIER3_EVAL_FAILED: $* ($(date -Is))"; exit 1; }

# The grid container (tier-1 invocation). A function, not a string, so the token never sits
# in a variable that could be echoed; do not add `set -x`.
grid() {
  docker run --rm --gpus all --network host -e HF_TOKEN="$(cat ~/.hf_token)" -e HF_HOME=/opt/hf_home \
    -v "$HOME/ravi/ravi:/opt/ml/code/ravi:ro" -v "$HOME/ravi/configs:/configs:ro" -v "$HOME/ravi/eval:/eval" \
    -v /data:/data:ro -v "$HOME/ravi/runs:/runs:ro" -v "$HOME/ravi/hf_home:/opt/hf_home" \
    --entrypoint python ravi-sa3-train:dev -m ravi.eval.grid "$@"
}

# ---- 1. wait for arm C
echo "=== queued $(date -Is): waiting for TIER3_C_DONE in $L/sa3_tier3_C.log"
until grep -q TIER3_C_DONE "$L/sa3_tier3_C.log" 2>/dev/null; do sleep 120; done
c_exit=$(grep -o 'TIER3_C_EXIT=[0-9]*' "$L/C_tier3.log" 2>/dev/null | tail -1)
echo "=== TIER3_C_DONE seen $(date -Is); ${c_exit:-no TIER3_C_EXIT= line in $L/C_tier3.log}"

# ---- 2. ACE-Step adapter backups (independent of the SA3 result, so before the C check)
for a in B E; do
  if aws s3 sync "$R/acestep_${a}_tier1/model/" "$S3/adapters/acestep_${a}_tier1/model/" --region us-east-1; then
    echo "=== ACE $a adapter backed up -> $S3/adapters/acestep_${a}_tier1/model/"
  else
    echo "=== WARNING: ACE $a adapter backup failed (rc $?); continuing"
  fi
done

if [ "$c_exit" != "TIER3_C_EXIT=0" ]; then
  echo "=== arm C training did not succeed (${c_exit:-no exit line}); not evaluating a broken adapter. See $L/C_tier3.log"
  die "arm C run failed"
fi

# ---- adapters the grid needs (host paths of the /runs/... in configs/eval/tier3.yaml)
for f in B_tier1/model/adapter.safetensors D_tier2/model/adapter.safetensors C_tier3/model/adapter.safetensors; do
  [ -s "$R/$f" ] || die "missing adapter $R/$f"
done

# ---- 3. B 2k / 4k checkpoints -> stable relative links (checkpoints/{arm}_s{seed}_h{hash6}_step{N:06d})
link_ckpt() {
  local step=$1 n name prefix
  local matches=()
  shopt -s nullglob
  matches=("$R"/B_tier1/checkpoints/B_*_step"${step}".safetensors)
  shopt -u nullglob
  n=${#matches[@]}
  [ "$n" -eq 1 ] || die "expected exactly one $R/B_tier1/checkpoints/B_*_step${step}.safetensors, found $n: ${matches[*]:-none}"
  name=$(basename "${matches[0]}")
  # Same run as B6k: model/adapter.safetensors is a copy of {prefix}_step006000.safetensors.
  prefix=${name%_step"${step}".safetensors}
  cmp -s "$R/B_tier1/checkpoints/${prefix}_step006000.safetensors" "$R/B_tier1/model/adapter.safetensors" \
    || die "$name: ${prefix}_step006000.safetensors is missing or differs from B_tier1/model/adapter.safetensors (not the B6k run)"
  ln -sfn "checkpoints/$name" "$R/B_tier1/ckpt_step${step}.safetensors" \
    || die "cannot create $R/B_tier1/ckpt_step${step}.safetensors (owned by root? sudo chown $(id -un) $R/B_tier1)"
  [ -s "$R/B_tier1/ckpt_step${step}.safetensors" ] || die "$R/B_tier1/ckpt_step${step}.safetensors does not resolve"
  echo "=== B step $step: $R/B_tier1/ckpt_step${step}.safetensors -> checkpoints/$name"
}
link_ckpt 002000
link_ckpt 004000

# ---- 4. tier-1 controls, reused (run_eval discovers conditions from directories and follows links)
mkdir -p "$E/tier3" || die "cannot create $E/tier3"
for c in base B6k anchor_lower; do
  [ -d "$E/tier1/$c" ] || die "no tier-1 outputs at $E/tier1/$c"
  dst=$E/tier3/$c
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then die "$dst exists and is not a link; refusing to replace it"; fi
  ln -sfn "../tier1/$c" "$dst" || die "cannot create $dst"
  echo "=== $dst -> ../tier1/$c"
done

# The grid would write any missing control output through the link into eval/tier1: require
# the linked controls to be complete (the grid plan counts done = sidecar + wav).
echo "=== grid dry-run $(date -Is)" >> "$L/tier3_grid.log"
ctl=$(grid /configs/eval/tier3.yaml --dry-run --conditions base,B6k,anchor_lower 2>&1)
rc=$?
echo "$ctl" >> "$L/tier3_grid.log"
echo "$ctl" | grep -E 'totals:| to do\)'
echo "$ctl" | grep -Fq 'totals: 0 inpaint + 0 generate model calls to do, 0 lower-anchor files' \
  || { [ $rc -ne 0 ] && echo "$ctl" | tail -5; die "tier-1 controls are incomplete or unreadable via eval/tier3 (rc $rc); see $L/tier3_grid.log"; }

# ---- 5. grid. Each condition writes ~15 GB of audio (tier 1); require 75 GB free on / first
# and wait (not die) so the script can be queued before the disk is cleared.
MIN_FREE_GB=75
while :; do
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] && break
  echo "=== waiting for disk: ${free_gb:-?} GB free on /, need $MIN_FREE_GB GB ($(date -Is))"
  sleep 600
done
echo "=== disk ok: ${free_gb} GB free"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
plan=$(grid /configs/eval/tier3.yaml --dry-run 2>&1)
rc=$?
echo "$plan" >> "$L/tier3_grid.log"
echo "$plan" | grep -E 'totals:|missing'
[ $rc -eq 0 ] || die "grid dry-run failed (rc $rc); see $L/tier3_grid.log"
if echo "$plan" | grep -Eq '^  missing (lora|audio|segments):'; then die "grid inputs missing (see above)"; fi
for attempt in 1 2; do  # the second pass resumes: done outputs are skipped, only failures rerun
  echo "=== grid start (pass $attempt) $(date -Is)" | tee -a "$L/tier3_grid.log"
  grid /configs/eval/tier3.yaml >> "$L/tier3_grid.log" 2>&1
  rc=$?
  echo "GRID3_EXIT=$rc" | tee -a "$L/tier3_grid.log"
  [ $rc -eq 0 ] && break
done
[ $rc -eq 0 ] || die "grid failed twice (GRID3_EXIT=$rc); see $L/tier3_grid.log. Rerun this script to resume"

# ---- 6. run_eval (tier-1 invocation, eval root tier3)
echo "=== run_eval start $(date -Is)"
docker run --rm --gpus all --network host -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf -v /hf:/hf \
  -v "$HOME/ravi/ravi:/app/ravi:ro" -v "$HOME/ravi/eval:/app/eval" -v "$HOME/ravi/manifests:/app/manifests:ro" \
  -v "$HOME/ravi/captions:/app/captions:ro" -v /data:/data:ro --entrypoint python ravi-eval -m ravi.eval.run_eval \
  --eval-root /app/eval/tier3 --base base --train-dir /data/ravi-corpus/prepared/44100 \
  --test-audio /data/ravi-corpus/prepared/44100 --splits /app/manifests/splits.json \
  --test-captions /app/eval/regions_test.json --caption-arm B --cache-dir /app/eval/.emb_cache \
  > "$L/tier3_run_eval.log" 2>&1
ev=$?
echo "TIER3_RUN_EVAL_EXIT=$ev" | tee -a "$L/tier3_run_eval.log"

# ---- 7. metric tables + grid manifest/config/log to S3 (top-level files only: no audio, no
# sidecars, links not followed)
if aws s3 sync "$E/tier3/" "$S3/eval/tier3/" --region us-east-1 --no-follow-symlinks \
     --exclude "*" --include "*.csv" --include "*.parquet" --include "*.json" --include "*.jsonl" --exclude "*/*"; then
  echo "=== tier-3 metrics backed up -> $S3/eval/tier3/"
else
  echo "=== WARNING: tier-3 metrics backup failed (rc $?)"
fi
[ $ev -eq 0 ] || die "run_eval failed (TIER3_RUN_EVAL_EXIT=$ev); see $L/tier3_run_eval.log"
echo "TIER3_EVAL_DONE $(date -Is)"
