# `ravi-acestep-train`: ACE-Step 1.5 LoRA training image

This image runs ACE-Step's own corrected trainer (`acestep/training_v2`, the integrated
Side-Step "fixed" subcommand) from [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5),
pinned at commit `ca1e85fe9430179831e6bc6be790c332190a3866` (2026-08-29), through the wrapper
`ravi.train.acestep_lora`. The same image runs under `docker run` on the EC2 GPU box and as a
SageMaker Training Job. See [SMOKE.md](SMOKE.md) for the first run on EC2. Every
`file:line` below refers to that commit.

| Piece | Version | Source of the pin |
|---|---|---|
| Base | `nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:4a801ef9…` (`BASE_IMAGE` build-arg) | ACE-Step's own `Dockerfile:28,33`; the torch cu128 wheels bring CUDA and cuDNN as pip packages |
| Python | 3.11 (installed by uv) | `pyproject.toml:6` allows 3.11–3.12; flash-attn is only locked for cp311 (`uv.lock:609-620`) |
| torch / torchaudio | 2.10.0+cu128 | `pyproject.toml:14-16` (Linux x86_64) + `uv.lock`, installed with `uv sync --frozen --no-dev` |
| peft / lightning / transformers | 0.18.1 / 2.6.1 / 4.57.6 | `uv.lock` |
| torchcodec | 0.10.0 | `uv.lock`; torchaudio 2.10 decodes through it, so FFmpeg (apt, 4.4) is installed |
| flash-attn | 2.8.3+cu128torch2.10, cp311 prebuilt wheel | `uv.lock:609-620`, pulled in by the vendored `acestep/third_parts/nano-vllm` |
| uv | 0.9.5 | same as `ravi-sa3-train` (upstream uses an unpinned `uv:latest`, `Dockerfile:62`) |
| tini | Ubuntu 22.04 package | PID 1: forwards SIGTERM to the wrapper, reaps DataLoader worker zombies |
| boto3 / pyyaml | 1.40.50 / 6.0.3 | boto3 is wrapper-only; pyyaml comes from the lock (installed under the exported lock as constraints) |
| Model weights | `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22` (HF, 2026-02-03); base variant `ACE-Step/acestep-v15-base` @ `e432212fec32b8965a14ffa57ae653438d6abd14` | not baked in; mounted or downloaded by the wrapper (below) |

ACE-Step's `Dockerfile` is not used as a base. It copies the build context as the project
(`Dockerfile:66`), installs Python from the deadsnakes PPA (`:53-58`), takes an unpinned uv and
starts Gradio or the REST server. This image keeps its CUDA base and its
`uv sync --frozen --no-dev` (`Dockerfile:71`), but uses a pinned git checkout, a uv-managed
Python and the wrapper entrypoint.

## Build

```bash
docker buildx build --platform linux/amd64 -f docker/acestep-train/Dockerfile \
    --build-arg RAVI_GIT_SHA=$(git rev-parse --short HEAD) -t ravi-acestep-train:dev --load .
```

The build context is the repo root. `Dockerfile.dockerignore` keeps everything except `ravi/`
(without tests) out of it. The last build step runs a sanity check on the CPU. It checks the
versions, the flash-attn import, a `torchaudio.load` through torchcodec/FFmpeg, and the upstream
APIs the wrapper and the repaint check call. It also checks that
`build_fixed_standalone_parser()` accepts the wrapper's full argv. To push, tag it for ECR
(`297416720456.dkr.ecr.us-east-1.amazonaws.com/ravi-acestep-train:<sha>-acestep-ca1e85fe`) the
same way as `docker/sa3-train/build_and_push.sh`.

## Model variant: turbo

The wrapper trains on `turbo` (`acestep-v15-turbo`, `--model-variant turbo`) by default. This
is the variant we serve. It is distilled for 8-step inference at shift 3.0 with guidance baked
in: the handler forces `guidance_scale` to 1.0 for turbo,
`core/generation/handler/generate_music.py:311-318`. It is also what the existing
`ace-step-finetuned-dev` endpoint and the canvas run, and what `repaint` is served with. A LoRA
trained on another variant would have to be tested on a schedule we do not serve.

`base` (`acestep-v15-base`, `--set model_variant=base`, shift 1.0, 50 steps) is the
documented alternative. Side-Step's docs say the corrected trainer's continuous timestep sampling
and CFG dropout match how base and SFT were trained. They also say turbo was "Not trained with
CFG" (`docs/sidestep/Model Management.md:91-97`). Training turbo with `--cfg-ratio 0.15` is
therefore a deliberate choice: the null condition is learnt but never used at turbo inference.
The smoke run and Tier 1 check that it does not hurt. If it does, `cfg_ratio: 0` or `base` are
the fallbacks. `fixed` reads `timestep_mu`, `timestep_sigma` and `data_proportion` from the
variant's `config.json` (`cli/config_builder.py:66-75`; turbo: -0.4 / 1.0 / 0.5).
`--shift` and `--num-inference-steps` are metadata only for training
(`docs/sidestep/Using Your Adapter.md:101`).

## Flash-Attention

It is not required, but it is installed and the wrapper requires it by default.
`training_v2/model_loader.py:25-50` uses `flash_attention_2` only when three things hold: the
device is CUDA, the GPU's compute capability is at least 8.0, and `import flash_attn` works.
Otherwise it falls back to `sdpa`, then `eager` (`model_loader.py:148-170`), and logs
`[OK] Model loaded with attn_implementation=...` (`:165`). Inference auto-detects it the same way
(`docs/en/GPU_COMPATIBILITY.md:68`). The repaint check passes `use_flash_attention=True` to
`initialize_service`, because its default is `False`
(`core/generation/handler/init_service_orchestrator.py:48-60`).

With `require_flash_attn: true` (default) the wrapper refuses to start when the import fails or
the GPU is older than 8.0. It also refuses to train when preprocessing loaded the DiT with
anything but `flash_attention_2`. The run's `attn_implementation` is recorded in
`run_meta.json`. The L40S is 8.9 and the A10G 8.6. The import is checked at build time; the kernel runs only on
the GPU box (SMOKE.md section 3).

## Data

The input is ACE-Step's per-file layout (`docs/en/LoRA_Training_Tutorial.md:36-46`), written
from a make_arms arm by `ravi.captions.make_acestep_dataset`:

```bash
python -m ravi.captions.make_acestep_dataset --arm-dir data/arms/E --split train \
    --manifest manifests/tracks.parquet --out data/acestep/E/train
```

Each clip becomes `{name}.wav` (48 kHz, polyphase 160/147 from the 44.1 kHz window), plus
`{name}.caption.txt` (the `␞` fields joined with `", "`, trigger first) and `{name}.json`
(`caption`, `bpm` from `id3_bpm` if known, `is_instrumental: true`). For arm E only, the
`sections:` field becomes `{name}.lyrics.txt` with one tag per line (`[intro]\n[verse]\n...`).
The other arms get no lyrics file, so ACE-Step uses `[Instrumental]`
(`dataset_builder_modules/scan.py:77`). Traps the converter and the wrapper's preflight guard
against:

* `{name}.txt` is read as **lyrics** (`audio_io.py:72-85`). A make_arms caption file copied next
  to the audio would become lyrics, so it must not be there.
* Every `*.csv` in the directory is read, and its `caption` column overrides the captions
  (`csv_metadata.py:10-78`, `scan.py:99-108`). make_arms' `manifest.csv` has both columns.
* The JSON `caption` overrides `.caption.txt` (`scan.py:84-87`). `is_instrumental` is not read
  from the JSON. A lyrics file flips it to `False` (`scan.py:66-68`), but training_v2 only
  carries it as metadata (`training_v2/preprocess.py:300`).
* Tensors are named `{stem}.pt` (`training_v2/preprocess.py:237,279`), so stems must be unique
  across subdirectories.
* Clips longer than `max_duration` (240 s) are cut off without a warning
  (`preprocess_audio.py:17-19`).

The text encoder sees `SFT_GEN_PROMPT` with the caption and a metas block (`bpm`,
`timesignature`, `keyscale`, `duration`) (`training_v2/preprocess_prompt.py:16-74`,
`constants.py:160-168`). Lyrics go through the Qwen3 token embedding at up to 512 tokens
(`preprocess_lyrics.py:4-26`).

## What a run does

1. Preflight on the data dir. The model weights must be present at the pinned revision; they are
   downloaded into `weights_dir` (default `/opt/acestep/checkpoints` =
   `ACESTEP_CHECKPOINTS_DIR`) when missing. The download covers turbo, `vae`,
   `Qwen3-Embedding-0.6B` and the 1.7B LM, about 10 GB. The LM is not needed for training, but
   without it `AceStepHandler.initialize_service` auto-downloads the whole main repo at its
   current, unpinned revision (`init_service_downloads.py:31-36`,
   `model_downloader.py:386-402`). `download-weights` does only this step:
   `docker run ... ravi-acestep-train:dev download-weights`.
2. **Dataset JSON**: upstream `DatasetBuilder.scan_directory` reads the layout (the Gradio "Load
   Data" step). Its samples are written in `save_dataset`'s format and compared with the files:
   caption, lyrics, bpm, labeled, no custom tag, `genre_ratio` 0.
3. **Preprocessing**:
   `python -m acestep.training_v2.cli.train_fixed --plain --yes --preprocess --dataset-json ...
   --tensor-output ...` (`train_fixed.py:134-195`). It runs two passes with sequential model
   loading: VAE + text encoder, then the DiT condition encoder (`training_v2/preprocess.py:118-143`).
   It exits 0 even when clips fail (`train_fixed.py:188-195`), so the wrapper counts the `.pt`
   files. The tensors are cached in `work_dir` by data fingerprint, variant, weights revision,
   precision and `max_duration`.
4. **Training**: `train_fixed` with the flags below, in a child process (SIGTERM, then SIGKILL
   after 30 s). `run_fixed` also exits 0 after a `[FAIL]` message (`trainer_fixed.py:170-172`),
   so success means `final/adapter_model.safetensors` exists and the log has no `[FAIL]` line.
   Without `--yes` it would prompt and, on EOF, exit 0 without training
   (`ui/config_panel.py:266-291`).
5. Each `checkpoints/epoch_{N}_loss_{L}/` (`trainer_fixed.py:514-523`) is copied to
   `/opt/ml/checkpoints/{arm}_s{seed}_h{hash6}_epoch{N:04d}/` with a `ravi_meta.json`, and
   uploaded to `{s3_out}/{arm}/checkpoints/` if `s3_out` is set.
6. The final adapter goes to `/opt/ml/model/adapter/` (`adapter_config.json` +
   `adapter_model.safetensors`). This is PEFT `save_pretrained` of the decoder into
   `output_dir/final` (`trainer_fixed.py:551-554`, `trainer_helpers.py:145-184,237-240`).
   `run_meta.json` goes to `/opt/ml/model/` and `/opt/ml/output/data/`. Load the adapter with
   `AceStepHandler.load_lora("/opt/ml/model/adapter")` (`lora/lifecycle.py:191-271,319-325`; it
   needs the directory with `adapter_config.json`) or `POST /v1/lora/load`
   (`api/http/lora_routes.py:60-81`).

### Flags (all in `acestep/training_v2/cli/args.py`)

| Config key | Flag | Default | args.py |
|---|---|---|---|
| (always) | `--plain`, `--yes` | | 49-61 |
| `weights_dir` | `--checkpoint-dir` | `/opt/acestep/checkpoints` | 162-167 |
| `model_variant` | `--model-variant` | `turbo` | 168-177 (map 18-22) |
| `require_cuda` | `--device cuda` | | 193-198 |
| `precision` | `--precision` | `bf16` | 199-205 |
| (always) | `--num-devices 1` | | 206-211 |
| (derived) | `--dataset-dir` | tensor cache | 232-237 |
| `num_workers` | `--num-workers` | 4 | 238-243 |
| `lr` | `--lr` | 1e-4 | 265 |
| `batch_size` | `--batch-size` | 1 | 266 |
| `grad_accum` | `--gradient-accumulation` | 4 | 267 |
| `steps` | `--epochs` (derived, below) | 6000 steps | 268 |
| `warmup_steps` | `--warmup-steps` | 100 | 269 |
| `weight_decay` / `max_grad_norm` | `--weight-decay` / `--max-grad-norm` | 0.01 / 1.0 | 270-271 |
| `seed` | `--seed` | 42 (+ completed epochs on resume) | 272 |
| `shift` / `num_inference_steps` | `--shift` / `--num-inference-steps` | 3.0 / 8 (turbo) | 273-274 |
| `optimizer` / `scheduler` | `--optimizer-type` / `--scheduler-type` | adamw / cosine | 275-276 |
| `gradient_checkpointing` | `--gradient-checkpointing` / `--no-gradient-checkpointing` | on | 277 |
| (always) | `--adapter-type lora` | | 282 |
| `rank` / `alpha` / `dropout` | `--rank` / `--alpha` / `--dropout` | 64 / 128 / 0.1 | 286-288 |
| `target_modules` | `--target-modules` | q_proj k_proj v_proj o_proj | 289 |
| `attention_type` | `--attention-type` | both | 291 |
| (derived) | `--output-dir` | `work_dir/{run_id}/acestep_out` | 305 |
| `save_every_epochs` / `checkpoint_every_steps` | `--save-every` | ceil(500 / steps per epoch) | 306 |
| (resume) | `--resume-from` | staged checkpoint | 307 |
| `log_every` | `--log-every` | 10 | 312 |
| `cfg_ratio` | `--cfg-ratio` | 0.15 | 328 |
| preprocessing | `--preprocess`, `--dataset-json`, `--tensor-output`, `--max-duration` | 240 s | 318-322 |

`ravi/tests/test_acestep_lora.py` reads `args.py` as text (with `ACESTEP_SRC` set to a checkout)
and checks that every flag the wrapper emits exists there. The image build parses the same argv
with the real parser.

### Steps, epochs and their rounding

`training_v2` has no step cap; it trains whole epochs. Per epoch it takes
`ceil(len(loader) / grad_accum)` optimizer steps (`trainer_fixed.py:325`). The loader length is
`ceil(n_clips / batch)` because `drop_last=False` (`training/data_module.py:280-297`), and a
leftover partial accumulation is flushed as one step at the end of each epoch
(`trainer_fixed.py:476-501`):

```
steps_per_epoch = ceil(ceil(n / batch) / accum) = ceil(n / (batch * accum))
epochs          = ceil(steps / steps_per_epoch)          # rounded UP
actual_steps    = epochs * steps_per_epoch               # steps <= actual < steps + steps_per_epoch
save_every      = save_every_epochs or ceil(checkpoint_every_steps / steps_per_epoch)
```

For 580 clips at batch 1 and accum 4 that is 145 steps per epoch, so `steps: 6000` becomes 42
epochs = **6090** steps (+1.5 %), with checkpoints every 4 epochs (580 steps). The smoke run uses
20 clips, so 5 steps per epoch and 20 epochs = exactly 100 steps. The cosine schedule and warmup
run over `actual_steps` (`trainer_fixed.py:326-336`). The final epoch is always saved as
`final/`, whether or not it is a multiple of `save_every`. Each step's loss is the mean over
`accum` micro-batches. With few clips the last step of an epoch averages fewer, which is
upstream's behaviour. `run_meta.json["epoch_plan"]` records all of this.

## Resume

A checkpoint directory holds the adapter plus `training_state.pt` (optimizer, scheduler, epoch,
global step; `trainer_helpers.py:187-234`). `--resume-from` restores these
(`trainer_fixed.py:384-398`, `lora_checkpoint.py:151-254`).

Upstream has a bug here. It reloads the LoRA weights with
`decoder.load_state_dict(state_dict, strict=False)` (`trainer_helpers.py:395-398`), but PEFT's
`save_pretrained` drops the adapter name from the keys (`...lora_A.weight`, while the live
PeftModel has `...lora_A.default.weight`). With `strict=False` nothing matches and nothing loads.
A resumed run would keep its optimizer state but start from freshly initialised LoRA weights,
without any error. Any exception during resume also turns into "starting fresh"
(`trainer_fixed.py:393-398`).

`ravi.train.acestep_lora` launches the trainer through its `_child-train` entry. That entry
wraps upstream's `resume_checkpoint`: it reloads the adapter with
`peft.set_peft_model_state_dict`, compares every tensor with the file, and exits non-zero on any
problem. Upstream also only accepts paths under the process's working directory
(`training/path_safety.py:35,53-91`). The trainer therefore runs with `cwd = work_dir`, and the
checkpoint is copied to `work_dir/resume/` first.

The resume guard works like SA3's. Checkpoints are named `{arm}_s{seed}_h{hash6}_epoch{N:04d}`.
The hash covers the variant, weights revision, LoRA settings, optimiser and schedule, `steps`,
the derived `epochs`, `cfg_ratio`, `shift`, `precision`, `max_duration` and a fingerprint of the
data (file names, sizes, captions, lyrics, JSON). The available modes:

* `none` (default off SageMaker): always train from scratch.
* `auto` (default on SageMaker): the newest complete checkpoint of this identity, local or S3. A
  finished run is an error.
* `publish`: only republish a finished run.
* An explicit directory or `s3://` prefix: that checkpoint, with the identity still checked.

The resumed run uses seed `seed + completed_epochs`, so the noise stream does not replay. The
data order of the remaining epochs differs from an uninterrupted run.

## Configuration

`--config FILE` (YAML or JSON), SageMaker `hyperparameters.json` (JSON-encoded strings are
fine), then `--set key=value`; see `DEFAULTS` in `ravi/train/acestep_lora.py`. Paths default to
the SageMaker layout: `/opt/ml/checkpoints`, `/opt/ml/model`, `/opt/ml/output/data`,
`/opt/ml/ravi_work` (the trainer's cwd and the tensor cache), and `SM_CHANNEL_TRAIN` for the
data. `--dry-run` validates the data and prints both child commands.

## Repaint

At inference, an adapter is used through ACE-Step's Python API. Use
`GenerationParams(task_type="repaint", src_audio=..., repainting_start=..., repainting_end=...,
chunk_mask_mode="explicit", ...)` (`inference.py:56-226`; `docs/en/INFERENCE.md:510-524`).
`ravi.eval.acestep_repaint` makes the full call and checks the result (SMOKE.md section 7).
Things that differ from what one might assume:

* `repaint_latent_crossfade_frames` and `repaint_wav_crossfade_sec` are accepted but ignored. The
  handler derives both from `repaint_mode` / `repaint_strength`
  (`core/generation/handler/generate_music.py:27-48,237-238,421`). In balanced mode at 0.5 that
  gives 12 latent frames and 25 ms.
* The waveform splice puts the original audio back outside the mask. Its crossfade ramps sit
  **outside** the repaint region, `[start - cf, start)` and `[end, end + cf)`
  (`repaint_waveform_splice.py:14-51`). The splice runs after the decoder's peak normalisation
  (`generate_music_decode.py:199-203`, `generate_music.py:480-495`) and is skipped in
  `aggressive` mode.
* The REST API and Gradio swap in the repaint instruction
  `"Repaint the mask area based on the given conditions:"` (`constants.py:130`,
  `api/job_generation_setup.py:24-60`). `GenerationParams` defaults to the text2music one
  (`inference.py:112`), so pass it explicitly.
* `instrumental=True` is not forwarded to the DiT (`inference.py:845-905`); pass
  `lyrics="[Instrumental]"`. `enable_normalization` defaults to `True` and rescales the whole
  output, including the spliced original (`inference.py:998-1012`).
* The seed does not make repaint reproducible. It fixes only the initial noise (`prepare_noise`,
  `models/turbo/modeling_acestep_v15_turbo.py:1780-1817`); the source is VAE-encoded with
  `latent_dist.sample()` from the unseeded global RNG (`core/generation/handler/vae_encode.py:65`)
  on every call. Call `torch.manual_seed(seed)` before each `generate_music` when outputs must
  be paired across conditions (same-seed runs otherwise differed by about 10 % of the source
  RMS inside the mask).
* For a no-LoRA control on a handler with an adapter, use `set_use_lora(False)`. `unload_lora()`
  at this commit restores the decoder with mismatched keys and leaves the adapter active
  (`lora/lifecycle.py:472-474` logs "Missing keys when restoring decoder").

## Tier-1 runs and evaluation

* **Data:** `~/ravi/acestep/data/{B,E}/{train,val}` on the GPU box, from
  `ravi.captions.make_acestep_dataset --arm-dir /data/ravi-corpus/arms/{B,E} --split {train,val}`.
  Each train split is 580 clips. Arm E has 528 lyrics files; the other 52 clips have no
  `sections:` field.
* **Configs:** `configs/acestep/{B,E}_tier1.yaml`. Same budget as SA3 tier 1: 6000 steps, which
  becomes 42 epochs = 6090 steps, with checkpoints every 7 epochs (1015 steps).
* **Queue:** `infra/acestep_tier1_queue.sh`, copied to `~/ravi/runs/`. It waits for the SA3 arm-D
  run and an idle GPU, then trains B, then E. Results go to `~/ravi/runs/acestep_{B,E}_tier1/`,
  logs to `~/ravi/logs/acestep_{B,E}_tier1.log` (`ACE_{B,E}_EXIT=`), and the queue prints
  `ACE_TIER1_QUEUE_DONE` at the end.
* **Wall time on the A10G,** from the smoke run (1.38 s per optimizer step at batch 1 × accum 4,
  about 2 s per clip to preprocess): about 2.3 h of training plus about 0.3 h of preprocessing
  per arm, so 5-6 h for both. Longer clips than the smoke's (median 33 s) would push this up.
* **Evaluation:** `configs/eval/tier1_ace.yaml` runs `ravi.eval.grid` with `engine: acestep`
  (`ravi.eval.acestep_grid`). It uses the same regions, seeds, contexts and prompts as the SA3
  tier-1 grid and writes `/eval/tier1_ace`. The run is 3192 model calls; at 4-8 s per call that
  is 3.5-7 h.
