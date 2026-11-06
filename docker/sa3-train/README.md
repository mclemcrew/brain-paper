# `ravi-sa3-train`: Stable Audio 3 LoRA training image

This image runs Stability AI's own `scripts/train_lora.py` from
[stable-audio-3](https://github.com/Stability-AI/stable-audio-3), pinned at commit
`77e0398c26f7d22d801ec1bb036307e8b231f741` (2026-10-11), through the wrapper
`ravi.train.sa3_lora`. The same image runs under `docker run` on the EC2 GPU box and as a
SageMaker Training Job. See [SMOKE.md](SMOKE.md) for the first run on EC2.

| Piece | Version | Source of the pin |
|---|---|---|
| Base | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:5f0d2d82…` (`BASE_IMAGE` build-arg; the `cudnn-devel` variant only for a flash-attn source build) | the CUDA 12.6 wheels in SA3's `pyproject.toml` |
| Python | 3.10 (installed by uv) | SA3 `.python-version` |
| torch / torchaudio | 2.7.1+cu126 | SA3 `pyproject.toml` + `uv.lock`, installed with `uv sync --frozen --extra lora` |
| pytorch_lightning | 2.5.5 | SA3 `lora` extra |
| flash-attn | 2.6.3 (`+cu126torch2.7-cp310` prebuilt wheel) | URL taken from SA3 README, "Flash Attention" |
| uv | 0.9.5 | |
| tini | Ubuntu 22.04 package | PID 1: forwards SIGTERM to the wrapper, reaps worker zombies |
| boto3 / pyyaml | 1.40.50 / 6.0.3 | boto3 is wrapper-only; pyyaml comes from SA3's `uv.lock` (installed unpinned under the exported lock as constraints) |

`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel` was not used: it ships Python 3.11, which
does not match SA3's `.python-version`.

## Build

```bash
docker/sa3-train/build_and_push.sh            # buildx linux/amd64, --provenance=false, push to
                                              # 297416720456.dkr.ecr.us-east-1.amazonaws.com/ravi-sa3-train:<sha>-sa3-77e0398c
PUSH=0 PROFILE= docker/sa3-train/build_and_push.sh   # local build only (EC2 box, instance role)
FLASH_ATTN_WHEEL= MAX_JOBS=4 docker/sa3-train/build_and_push.sh  # compile flash-attn (switches to the devel base)
PUSH_LATEST=1 docker/sa3-train/build_and_push.sh     # also move :latest (off by default)
```

A push refuses uncommitted changes under `ravi/` or `docker/sa3-train/` unless
`ALLOW_DIRTY=1` (the tag then ends in `-dirty`). `RAVI_GIT_SHA` / `RAVI_IMAGE_TAG` are set in
the last layer, so a new commit does not rebuild the environment.

The build context is the repo root. `Dockerfile.dockerignore` keeps everything except
`ravi/` out of it.

## Configuration

`python -m ravi.train.sa3_lora [train] [--config run.yaml] [--hyperparameters hp.json] [--s3-out s3://…] [--set k=v]… [--dry-run]`

Precedence, lowest first: defaults < `--config` YAML/JSON (or a `config` hyperparameter
naming such a file) < `hyperparameters.json` < `--set` / `--s3-out`. Dotted keys set
nested values (`caption_aug.p_lowercase=0.25`). Unknown keys are an error. Values of
non-string keys may be JSON-encoded strings (`"6000"`, `"true"`, `'{"p_lowercase": 0.25}'`).
String keys (`arm`, `run_id`, paths, URIs, enum choices) are never parsed, so `arm: "123"`
stays a string; only a JSON-quoted string (`'"B"'`, as the SageMaker SDK sends it) is unwrapped.

| Key | Default | Maps to |
|---|---|---|
| `arm` | `unnamed` | `--name`; prefix of every checkpoint name |
| `data_dir` | `$SM_CHANNEL_TRAIN` | `--data_dir` (audio + sibling `.txt`) |
| `encoded_dir` | – | `--encoded_dir` (pre-encoded latents; captions in the JSON `prompt`) |
| `model` | `medium-base` | `--model` (base checkpoints only) |
| `rank`, `lora_alpha` | 16, = rank | `--rank`, `--lora_alpha` |
| `adapter_type` | `dora-rows` | `--adapter_type` |
| `lr` | 1e-4 | `--lr` (AdamW, wd 0.01, betas 0.9/0.95, constant LR) |
| `steps`, `checkpoint_every` | 6000, 500 | `--steps`, `--checkpoint_every`; steps must be a multiple |
| `demo_every`, `log_every` | 500, 50 | `--demo_every`, `--log_every` |
| `duration` | 47 | `--duration` (window = 46.997 s after alignment to 4096 samples) |
| `seed` | 42 | `--seed` (+N when resuming from step N); part of the checkpoint name |
| `batch_size`, `num_workers` | 1, 3 | `--batch_size`, `--num_workers` |
| `base_precision` | `bf16` | `--base_precision` |
| `exclude_duration_conditioner` | true | adds `seconds_total` to `--exclude` |
| `include`, `exclude` | – | extra `--include` / `--exclude` substrings (list, or one whitespace-separated string) |
| `caption_aug` | disabled | online augmentation in DataLoader workers; an arm opts in with `enabled: true`. Captions are fields separated by `field_sep` (`␞`, U+241E), shown to the model joined by `join_sep` (`", "`); first field must be `trigger` (`rvkr`); `non_droppable_fields`, `atomic_tail_prefixes` (`sections:` kept verbatim and last). See `ravi/train/caption_aug.py` |
| `inpaint_mask_type_probabilities` | upstream `[0.1, 0.8, 0.1]` | overrides `train_lora.py:179` (mask-aware ablation only) |
| `resume` | `auto` on SageMaker, `none` elsewhere | `none` / `auto` (newest checkpoint of this run identity, max of local and S3; a finished one is an error) / `publish` (publish the step-`steps` checkpoint without training) / path or `s3://` URI of a ravi `.safetensors` |
| `s3_out` | – | checkpoints, demos and `run_meta.json` are uploaded under `{s3_out}/{arm}/` |
| `hf_token_ssm_param` | – | SSM parameter to read `HF_TOKEN` from if it is not in the environment |
| `long_clip_policy` | `error` (`warn` if `smoke: true`) | clips longer than the window are random-cropped upstream with `seconds_total` (and hence the noise-schedule shift) taken from the full file |
| `smoke` | false | marks a smoke/debug run; only changes `long_clip_policy`'s default |

## SageMaker conventions honoured

| Convention | Use |
|---|---|
| `docker run <image> train` | a leading `train` argument is ignored |
| `/opt/ml/input/config/hyperparameters.json` | merged over the YAML. Values may be JSON-encoded strings; keys starting `sagemaker_` are ignored |
| `SM_CHANNEL_TRAIN` | default `data_dir` (channel `train`, File mode) |
| `/opt/ml/checkpoints` | `{arm}_s{seed}_h{hash6}_step{N:06d}.safetensors` (fp16, `ravi_*` identity in the metadata) and `.ckpt` (fp32 Lightning file, used for warm-start resume) after every checkpoint, plus `{arm}_s{seed}_h{hash6}_run_meta.json`. Set `checkpoint_local_path` to this and `checkpoint_s3_uri` so SageMaker syncs it and restores it on spot restarts, which `resume: auto` (the SageMaker default) then picks up. Temp files are written under `/opt/ml/ravi_work/partial`, never here |
| `SM_MODEL_DIR` (`/opt/ml/model`) | `adapter.safetensors` (final step) + `run_meta.json`, packed by SageMaker into `model.tar.gz` |
| `SM_OUTPUT_DATA_DIR` (`/opt/ml/output/data`) | `train.log`, `metrics_from_{N}.csv`, `demos/demo_step{N}_cfg{c}.wav`, `run_meta.json` |
| `/opt/ml/output/failure` | one-line reason on failure; the exit code is non-zero. S3 upload errors after a successful training are logged loudly and recorded (`status: completed_with_s3_errors`) but exit 0 |
| `TRAINING_JOB_NAME` | default `run_id` |
| SIGTERM (stop / spot interruption) | `tini` passes it to the wrapper, which sends SIGTERM to the trainer and SIGKILL after 30 s; the run is marked `interrupted`, `train.log` is uploaded, and the last checkpoint stays in `/opt/ml/checkpoints` |
| Multi-GPU instances | the trainer is pinned to one GPU (`CUDA_VISIBLE_DEVICES=0` unless set); `run_meta.env.gpus` lists all of them |

Scratch files (Lightning `.ckpt`, the CSV logger, raw demos) go to `/opt/ml/ravi_work/{run_id}`
(`RAVI_WORK_DIR` overrides). The model weights download to `HF_HOME=/opt/hf_home`. For a
SageMaker job, pass `HF_HOME=/opt/ml/ravi_work/hf` in `environment` to keep the ~several-GB
download on the job's storage volume, and set `hf_token_ssm_param=/uaudio/hf_token` rather
than a plaintext `HF_TOKEN` env var (the plan's Phase 0 found one such leak).

```python
from sagemaker.estimator import Estimator
est = Estimator(
    image_uri="297416720456.dkr.ecr.us-east-1.amazonaws.com/ravi-sa3-train:<tag>",
    role="<sagemaker execution role>", instance_type="ml.g6e.xlarge", instance_count=1,
    volume_size=200, max_run=6 * 3600,
    checkpoint_s3_uri="s3://uaudio-ravi-corpus-dev-297416720456/adapters/B/checkpoints-sm/",
    checkpoint_local_path="/opt/ml/checkpoints",
    environment={"HF_HOME": "/opt/ml/ravi_work/hf"},
    hyperparameters={"arm": "B", "steps": 6000, "checkpoint_every": 2000, "demo_every": 2000,
                     "hf_token_ssm_param": "/uaudio/hf_token",
                     "s3_out": "s3://uaudio-ravi-corpus-dev-297416720456/adapters"},
)
est.fit({"train": "s3://uaudio-ravi-corpus-dev-297416720456/arms/B/"})
```

## What the upstream trainer does (read from the pinned source)

- **Captions**: `scripts/train_lora.py:77-81` reads `Path(audio).with_suffix(".txt")`
  verbatim (`.strip()` only). If the `.txt` is missing, the clip is rejected and
  `SampleDataset` silently returns a random other clip (`stable_audio_3/data/dataset.py:297-298`).
  The wrapper therefore refuses to start if any clip lacks a caption. Upstream has no
  caption augmentation. `ravi/train/caption_aug.py` replaces that function in the child
  process.
- **CFG dropout**: p = 0.1 on the text embedding (`training/diffusion.py:55`,
  `models/dit.py:441-450`). The wrapper's augmentation does not add another empty-caption
  dropout.
- **Short clips** (< window): zero-padded to the window and marked in a `padding_mask`
  (`data/utils.py:64-71`). Padding is attention-masked (`mask_padding_attention=True`,
  `train_lora.py:171`). Loss is computed on the real audio plus a random silence tail drawn
  from an exponential with a 4 s mean (`silence_extension_scale_seconds=4.0`,
  `train_lora.py:172`; `training/utils.py:280-329`; `training/diffusion.py:386-403`).
  `seconds_total = ceil(len/sr)` (`data/utils.py:52`). Clips are not skipped, except
  all-silent ones (< −60 dBFS, `dataset.py:259-261`).
- **Long clips** (> window): randomly cropped (`SampleDataset(random_crop=True)` is the
  default, `dataset.py:187`, `data/utils.py:42-43`). `seconds_total` stays the length of
  the **whole file**, not the crop (`data/utils.py:52`).
- **Inpaint masks during LoRA training**: `inpainting_config={"mask_kwargs":
  {"mask_type_probabilities": [0.1, 0.8, 0.1]}}` (`train_lora.py:179`). Per sample, that is
  10% random segments, 80% full generation and 10% causal continuation
  (`models/inpainting.py:164-236`). The loss covers the generated region plus a
  context-reconstruction term (`training/diffusion.py:405-417, 475-489`).
- **Duration conditioner**: by default LoRA goes on every `Linear`/`Conv1d` in the DiT and
  the conditioner, including `conditioners.seconds_total.embedder.embedding.1`.
  `--exclude seconds_total` (`train_lora.py:345-350`, `docs/workflows/lora.md:91-95`)
  skips it. The exclusion is saved in the adapter and re-applied at load time.
- **Checkpoints**: Lightning `.ckpt` files containing only the LoRA tensors and
  `lora_config`. The optimizer state and step counter are cleared (`diffusion.py:634-641`).
  They are written only every `checkpoint_every` steps, never at the end of training. The
  wrapper converts each one to `.safetensors` with upstream's
  `convert_lora_ckpt_to_safetensors`; DoRA row norms are not baked in, so they are
  recomputed against whichever base the adapter is loaded onto.
- **Resume** = weights-only warm start via `--lora_checkpoint` (AdamW moments reset). The
  wrapper passes the fp32 `.ckpt`: `save_lora_safetensors` casts the adapter to fp16
  (`models/lora/utils.py:275-285`), so resuming from the `.safetensors` would round the
  weights once per resume.
- **Demos**: at global steps 1, `demo_every+1`, … (`diffusion.py:899-900`), 4 clips from
  the first batch, 50 steps, CFG 2/4/7, written to the trainer's working directory
  (`diffusion.py:937-939`). The wrapper collects them from there. The demo prompts are the
  un-augmented captions.
