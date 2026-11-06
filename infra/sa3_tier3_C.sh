#!/usr/bin/env bash
# SA3 tier-3 arm C on the GPU box: waits for the ACE-Step tier-1 eval chain (ACE_TIER1_EVAL_DONE in
# ~/ravi/logs/acestep_tier1_eval.log) and an idle GPU, then trains configs/sa3/C_tier3.yaml with the
# same docker invocation as runs/tier1_queue.sh. Ends with TIER3_C_EXIT= in ~/ravi/logs/C_tier3.log.
#
#   cp infra/sa3_tier3_C.sh ~/ravi/runs/ && \
#     nohup ~/ravi/runs/sa3_tier3_C.sh > ~/ravi/logs/sa3_tier3_C.log 2>&1 < /dev/null &
set -u
cd "$HOME/ravi"
L=$HOME/ravi/logs
echo "=== queued $(date -Is): waiting for ACE_TIER1_EVAL_DONE in $L/acestep_tier1_eval.log"
until grep -q ACE_TIER1_EVAL_DONE "$L/acestep_tier1_eval.log" 2>/dev/null; do sleep 120; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
echo "=== C start $(date -Is)"
arm=C
mkdir -p runs/${arm}_tier3/{checkpoints,model,output,work}
docker run --rm --name sa3_C_tier3 --gpus all --shm-size 16g --network host -e HF_TOKEN=$(cat ~/.hf_token) -e HF_HOME=/opt/hf_home \
  -v /data/ravi-corpus/arms/$arm/train:/data/train:ro -v $HOME/ravi/hf_home:/opt/hf_home \
  -v $HOME/ravi/runs/${arm}_tier3/checkpoints:/opt/ml/checkpoints -v $HOME/ravi/runs/${arm}_tier3/model:/opt/ml/model \
  -v $HOME/ravi/runs/${arm}_tier3/output:/opt/ml/output -v $HOME/ravi/runs/${arm}_tier3/work:/opt/ml/ravi_work \
  -v $HOME/ravi/configs/sa3/${arm}_tier3.yaml:/config/run.yaml:ro ravi-sa3-train:dev --config /config/run.yaml > "$L/C_tier3.log" 2>&1
rc=$?
echo "TIER3_C_EXIT=$rc" >> "$L/C_tier3.log"
echo "=== C done $(date -Is) rc=$rc"
aws s3 sync "$HOME/ravi/runs/${arm}_tier3/model/" "s3://uaudio-model-training-checkpoints/ravi-v2/adapters/C_tier3/model/" --region us-east-1 && echo "=== C adapter backed up"
echo TIER3_C_DONE
