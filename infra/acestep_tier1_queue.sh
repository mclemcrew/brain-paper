#!/usr/bin/env bash
# ACE-Step 1.5 tier-1 LoRA queue on the GPU box: waits for the SA3 arm-D run (TIER2_D_EXIT= in
# ~/ravi/logs/D_tier2.log) and an idle GPU, then trains configs/acestep/B_tier1.yaml and
# E_tier1.yaml one after the other with ravi-acestep-train:dev.
#
#   cp infra/acestep_tier1_queue.sh ~/ravi/runs/ && \
#     nohup ~/ravi/runs/acestep_tier1_queue.sh > ~/ravi/logs/acestep_tier1_queue.log 2>&1 < /dev/null &
#
# Inputs: ~/ravi/acestep/data/{B,E}/train (ravi.captions.make_acestep_dataset, 580 clips each),
# ~/ravi/acestep/ckpt (pinned weights, `download-weights`), ~/ravi/configs/acestep/*.yaml.
# Outputs: ~/ravi/runs/acestep_{B,E}_tier1/{checkpoints,model,output,work}; the adapter is
# model/adapter/{adapter_config.json,adapter_model.safetensors}. Logs: ~/ravi/logs/acestep_{B,E}_tier1.log
# with ACE_{B,E}_EXIT= at the end; ACE_TIER1_QUEUE_DONE in the queue log when both are done.
# Containers run as root: clear old run dirs with sudo. Stop a run with `docker stop -t 60 acestep_B_tier1`.
set -u
IMG=${IMG:-ravi-acestep-train:dev}
L=$HOME/ravi/logs
A=$HOME/ravi/acestep
ARMS=${ARMS:-"B E"}
mkdir -p "$L"
echo "=== queued $(date -Is): arms $ARMS; waiting for TIER2_D_EXIT= in $L/D_tier2.log"
until grep -q "TIER2_D_EXIT=" "$L/D_tier2.log" 2>/dev/null; do sleep 60; done
echo "=== $(grep "TIER2_D_EXIT=" "$L/D_tier2.log" | tail -1) seen $(date -Is)"
for ARM in $ARMS; do
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
  R=$HOME/ravi/runs/acestep_${ARM}_tier1
  mkdir -p "$R"/{checkpoints,model,output,work}
  echo "=== ACE $ARM start $(date -Is)"
  s=$(date +%s)
  docker run --rm --name "acestep_${ARM}_tier1" --gpus all --shm-size 8g \
    -v "$A/data/$ARM/train:/data/train:ro" \
    -v "$A/ckpt:/opt/acestep/checkpoints" -v "$A/hf_home:/opt/hf_home" \
    -v "$R/checkpoints:/opt/ml/checkpoints" -v "$R/model:/opt/ml/model" \
    -v "$R/output:/opt/ml/output" -v "$R/work:/opt/ml/ravi_work" \
    -v "$HOME/ravi/configs/acestep/${ARM}_tier1.yaml:/config/run.yaml:ro" \
    -v "$HOME/ravi/ravi:/opt/ml/code/ravi:ro" \
    "$IMG" --config /config/run.yaml > "$L/acestep_${ARM}_tier1.log" 2>&1
  code=$?
  echo "ACE_${ARM}_EXIT=$code" >> "$L/acestep_${ARM}_tier1.log"
  echo "ACE_${ARM}_EXIT=$code seconds=$(( $(date +%s) - s )) $(date -Is)"
done
echo "ACE_TIER1_QUEUE_DONE $(date -Is)"
