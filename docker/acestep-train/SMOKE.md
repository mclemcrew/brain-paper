# ACE-Step 1.5 LoRA smoke test on the EC2 GPU box (Phase 2.4)

The goal is to show, on the GPU box, five things. The box is a g5.2xlarge (1x A10G 24 GB,
compute capability 8.6, 8 vCPU) since 2026-10-15; it was a g6e.xlarge (1x L40S 48 GB, 8.9,
4 vCPU). The smoke run peaks at 5 GiB of VRAM, so either works, but only one GPU job at a time
fits on the A10G:

* the image builds and flash-attn really runs;
* ACE-Step's `training_v2` preprocesses and trains 100 LoRA steps on 20 arm E clips without
  errors, with a loss that does not blow up;
* the PEFT adapter loads in ACE-Step inference;
* `repaint` with the adapter leaves everything outside the mask untouched;
* resume restores the adapter weights.

Run every command from the repo root unless it says otherwise. `file:line` references are to
ACE-Step-1.5 at `ca1e85fe` (see [README.md](README.md)).

## 0. Box and access

```bash
# on the Mac
INSTANCE_PROFILE=<role with ECR/S3/SSM read> infra/launch_gpu_box.sh   # Deep Learning Base OSS Nvidia AMI, Docker + NVIDIA toolkit
IP=$(cat infra/.gpu_box_ip)
```

The ACE-Step weight repos are public, so no HF token is needed. As with SA3, the IMDS hop limit
is 1. Add `--network host` to a `docker run` that needs the instance role (`s3_out`).

Disk: the weights take about 10 GB and the image about 15 GB. The tensor cache for 20 clips is
small, but the full arm E (580 clips) needs a few GB in `~/ravi/acestep/work`.

## 1. Smoke data (arm E, 20 clips)

Build arm E with `make_arms` (same manifest, splits and labels as Phase 1). Then convert it to
ACE-Step's per-file layout with `ravi.captions.make_acestep_dataset`, on the Mac:

```bash
.venv/bin/python -m ravi.captions.make_arms --manifest manifests/tracks.parquet \
    --splits manifests/splits.json --labels-dir ~/Downloads/labels \
    --prepared-root prepared --sr 44100 --out-root data/arms --arms E --split train,val
.venv/bin/python -m ravi.captions.make_acestep_dataset --arm-dir data/arms/E --split train \
    --manifest manifests/tracks.parquet --limit 20 --out data/acestep/E_smoke/train
.venv/bin/python -m ravi.captions.make_acestep_dataset --arm-dir data/arms/E --split val \
    --manifest manifests/tracks.parquet --out data/acestep/E/val
ls data/acestep/E_smoke/train | head -4
rsync -a --exclude prepared --exclude data --exclude .venv --exclude node_modules ./ ubuntu@$IP:~/brain-paper/
rsync -a data/acestep/ ubuntu@$IP:~/ravi/data/acestep/
ssh -i ~/.ssh/fxjepa-oct2026.pem ubuntu@$IP
```

Each clip becomes four files: `{track_id}.wav` (48 kHz PCM_24), `{track_id}.caption.txt`,
`{track_id}.json` and, for arm E, `{track_id}.lyrics.txt`. The directory also gets
`conversion_manifest.json` and a `.make_acestep_dataset` marker. Track ids contain spaces and
`.mp3`, which ACE-Step handles. Check one clip:

```bash
cd data/acestep/E_smoke/train
f="$(ls *.caption.txt | head -1)"; s="${f%.caption.txt}"
cat "$f"; echo; cat "$s.lyrics.txt"; echo; cat "$s.json"
```

Expected:

* The caption reads `rvkr, <human caption>, <genre>`. The fields are joined with `", "`, and no
  `␞` and no `sections:` field remain.
* The lyrics hold one `[label]` per line, e.g. `[intro]\n[verse]\n[transition]`.
* The JSON is `{"caption": ..., "is_instrumental": true}`, plus `bpm` when the manifest has an
  `id3_bpm`.

There must be no `{track_id}.txt` file and no `*.csv` in the directory. The wrapper's preflight
refuses both (README "Data"), because ACE-Step would read them as lyrics or as caption overrides.

## 2. Build (on the box)

```bash
cd ~/brain-paper
docker buildx build --platform linux/amd64 -f docker/acestep-train/Dockerfile \
    --build-arg RAVI_GIT_SHA=$(git rev-parse --short HEAD) -t ravi-acestep-train:dev --load . 2>&1 | tee ~/build_acestep.log
IMG=ravi-acestep-train:dev
```

The sanity step should end with a line like this:
`python 3.11.x | torch 2.10.0+cu128 cuda 12.8 | peft 0.18.1 lightning 2.6.1 transformers 4.57.6
torchcodec 0.10.0 torchaudio 2.10.0+cu128 flash-attn 2.8.3... boto3 1.40.50`.

If it fails, the assertion shows which pin or upstream API moved. A `uv sync --frozen` error
means the checkout and `uv.lock` disagree. Rebuild from a clean clone rather than re-resolving.

## 3. GPU + flash-attn kernel check

```bash
docker run --rm -i --gpus all --entrypoint python "$IMG" - <<'EOF'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q)
ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, q, q))).transpose(1, 2)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "flash-attn", flash_attn.__version__,
      "max|diff|", (o - ref).abs().max().item())
EOF
```

Expected: `NVIDIA A10G (8, 6) flash-attn 2.8.3 max|diff| <about 1e-2>` (2026-10-15: 0.00195),
or `NVIDIA L40S (8, 9) ...` on a g6e. An `undefined symbol`
error means the prebuilt wheel does not match torch 2.10. Without flash-attn the wrapper refuses
to train (`require_flash_attn: true`). `--set require_flash_attn=false` accepts sdpa, and the
choice is recorded in `run_meta.json`.

## 4. Weights

```bash
mkdir -p ~/ravi/acestep/{ckpt,hf_home}
docker run --rm \
  -v ~/ravi/acestep/ckpt:/opt/acestep/checkpoints \
  -v ~/ravi/acestep/hf_home:/opt/hf_home \
  "$IMG" download-weights
cat ~/ravi/acestep/ckpt/.ravi_weights.json
```

Expected: one `[ravi] downloading ACE-Step/Ace-Step1.5@19671f40... ['acestep-v15-turbo/*', 'vae/*',
'Qwen3-Embedding-0.6B/*', 'acestep-5Hz-lm-1.7B/*']` line. The JSON should show
`"verified_revision": true` and `variant_files` with `config.json`, `model.safetensors` and
`silence_latent.pt`. A second run downloads nothing. All later commands mount the same dir, so
neither the trainer nor `initialize_service` downloads anything at an unpinned revision.

## 5. Dry run (no GPU work; validates the data and prints both child commands)

```bash
docker run --rm \
  -v ~/ravi/data/acestep/E_smoke/train:/data/train:ro \
  -v ~/brain-paper/docker/acestep-train/smoke.yaml:/config/smoke.yaml:ro \
  "$IMG" --config /config/smoke.yaml --set require_cuda=false --dry-run
```

Expected output:

* `[ravi] data: {"n_clips": 20, "n_with_lyrics": 20, "n_not_48k": 0, ..., "n_warnings": 0}`.
* `[ravi] epochs: {"n_clips": 20, "steps_per_epoch": 5, "target_steps": 100, "epochs": 20,
  "actual_steps": 100, "overshoot_steps": 0, "save_every_epochs": 10, "checkpoint_epochs": [10, 20], ...}`.
* A JSON block with `"identity": {"prefix": "E_smoke_s42_h<hash6>", ...}` and `"resume": "none"`.
* `preprocess_cmd`: `... -m acestep.training_v2.cli.train_fixed --plain --yes ... --model-variant turbo
  ... --preprocess --dataset-json /opt/ml/ravi_work/dataset_<fp12>.json --tensor-output
  /opt/ml/ravi_work/tensors_turbo_<key12> --max-duration 240.0`.
* `train_cmd`: `... -m ravi.train.acestep_lora _child-train -- --plain --yes ... --epochs 20
  --warmup-steps 10 ... --rank 64 --alpha 128 --dropout 0.1 --target-modules q_proj k_proj v_proj
  o_proj --attention-type both ... --save-every 10 --log-every 5 --cfg-ratio 0.15`, with no
  `--resume-from`.

## 6. 100-step run

```bash
mkdir -p ~/ravi/acestep/smoke/{checkpoints,model,output,work}
docker run --rm --gpus all --shm-size 8g \
  -v ~/ravi/data/acestep/E_smoke/train:/data/train:ro \
  -v ~/ravi/acestep/ckpt:/opt/acestep/checkpoints \
  -v ~/ravi/acestep/hf_home:/opt/hf_home \
  -v ~/ravi/acestep/smoke/checkpoints:/opt/ml/checkpoints \
  -v ~/ravi/acestep/smoke/model:/opt/ml/model \
  -v ~/ravi/acestep/smoke/output:/opt/ml/output \
  -v ~/ravi/acestep/smoke/work:/opt/ml/ravi_work \
  -v ~/brain-paper/docker/acestep-train/smoke.yaml:/config/smoke.yaml:ro \
  "$IMG" --config /config/smoke.yaml 2>&1 | tee ~/ravi/acestep/smoke/console.log
```

To stop a run, use `docker stop -t 60 <container>`, not the default 10 s. tini forwards SIGTERM
to the wrapper, which stops the trainer (SIGKILL after 30 s) and writes `run_meta.json` with
`status: interrupted`.

### Expected console lines

| Line | Source |
|---|---|
| `[ravi] env: {"python": "3.11...", "torch": "2.10.0+cu128", ..., "gpu": "NVIDIA A10G", "gpu_capability": [8, 6], "flash-attn": "2.8.3...", "flash_attn_import": "ok", "acestep_import": "ok"}` | wrapper preflight |
| `[ravi] upstream scan: ✅ Found 20 audio files in /data/train` | `dataset_builder_modules/scan.py:116`; the wrapper then compares the samples with the files |
| `[ravi] launching (cwd /opt/ml/ravi_work): ... --preprocess ...` | wrapper |
| `[OK] Model loaded with attn_implementation=flash_attention_2` (preprocess.log) | `training_v2/model_loader.py:165` |
| `[ravi] tensors: {"dir": ".../tensors_turbo_<key12>", "n_expected": 20, "n_present": 20, "complete": true, ...}` | wrapper |
| `[ravi] launching (cwd /opt/ml/ravi_work): ... _child-train -- ...` | wrapper |
| `[OK] Model loaded with attn_implementation=flash_attention_2` | `model_loader.py:165` |
| `Epoch 1/20, Step 1, Loss: ...` ... `Epoch 20/20, Step 100, Loss: ...` (every 5 steps) | `trainer_fixed.py:457` |
| `[OK] Epoch N/20 in Xs, Loss: ...` (20 lines) | `trainer_fixed.py:510` |
| `[ravi] checkpoint epoch 10 (step 50): /opt/ml/checkpoints/E_smoke_s42_h<hash6>_epoch0010` and the same for epoch 20 (step 100) | wrapper |
| `[OK] Training complete! LoRA saved to .../acestep_out/final` | `trainer_fixed.py:563` |
| `[ravi] child exited with code 0` then `[ravi] completed: adapter at /opt/ml/model/adapter` | wrapper |

These must not appear:

* any `[FAIL]` line (the wrapper fails the run on one);
* `attn_implementation=sdpa` or `=eager`;
* the `␞` separator anywhere in the logs.

```bash
grep -cE "^\s*\[FAIL\]|attn_implementation=(sdpa|eager)" ~/ravi/acestep/smoke/console.log   # expect 0
grep -c "$(printf '\xe2\x90\x9e')" ~/ravi/acestep/smoke/console.log                          # expect 0
```

### Expected files

```
~/ravi/acestep/smoke/checkpoints/  E_smoke_s42_h<hash6>_epoch0010/  adapter_config.json adapter_model.safetensors training_state.pt ravi_meta.json
                                   E_smoke_s42_h<hash6>_epoch0020/  (the same)
                                   E_smoke_s42_h<hash6>_final/      adapter_config.json adapter_model.safetensors ravi_meta.json
                                   E_smoke_s42_h<hash6>_run_meta.json
~/ravi/acestep/smoke/model/        adapter/{adapter_config.json,adapter_model.safetensors}  run_meta.json
~/ravi/acestep/smoke/output/data/  train.log  preprocess.log  run_meta.json  tensorboard/
~/ravi/acestep/smoke/work/         dataset_<fp12>.json  tensors_turbo_<key12>/*.pt (20)  <run_id>/acestep_out/
```

### Pass checks

```bash
python3 - <<'EOF'
import json, math
m = json.load(open("/home/ubuntu/ravi/acestep/smoke/output/data/run_meta.json"))
c = [r["loss"] for r in m["loss_curve"]]
k = max(1, len(c) // 3)
print(m["status"], m["acestep_commit"], m["env"]["gpu"], m["attn_implementation"], m["preprocess_attn_implementation"])
print("epoch_plan", m["epoch_plan"]["epochs"], m["epoch_plan"]["actual_steps"], [x["epoch"] for x in m["checkpoints"]])
print("points", len(c), "finite", all(x is not None and math.isfinite(x) for x in c),
      "first third", sum(c[:k]) / k, "last third", sum(c[-k:]) / k)
EOF
docker run --rm -v ~/ravi/acestep/smoke/model/adapter:/a:ro --entrypoint python "$IMG" -c "
import json; from safetensors import safe_open
cfg = json.load(open('/a/adapter_config.json'))
print({k: cfg[k] for k in ('peft_type', 'r', 'lora_alpha', 'lora_dropout', 'target_modules')})
f = safe_open('/a/adapter_model.safetensors', 'pt'); keys = list(f.keys())
print(len(keys), 'tensors, e.g.', keys[0]); assert not any('.default.' in k for k in keys)"
```

A run passes when:

- `status` is `completed`, `acestep_commit` is `ca1e85fe…`, and both attention fields are
  `flash_attention_2`.
- The plan is 20 epochs and 100 steps, with checkpoints at epochs 10 and 20.
- There are 20 loss points (every 5 steps), all finite, and the last-third mean is not above the
  first-third mean. The loss is noisy, so expect a small decrease, not a large one.
- `adapter_config.json` shows `LORA`, `r` 64, `lora_alpha` 128, `lora_dropout` 0.1 and the four
  target modules.
- The adapter keys are PEFT `save_pretrained` keys (`...lora_A.weight`, no `.default.`). This is
  the format both `load_lora` and the wrapper's resume reload expect.

## 7. Repaint check (`ravi.eval.acestep_repaint`)

This step loads the adapter in ACE-Step inference and repaints one interior section of a
full-length val track. It repaints twice without the adapter (`nolora`, `nolora_repeat`), then
at LoRA scale 0, 0.5 and 1.0, all with the same seed on the turbo schedule we serve (8 steps,
shift 3.0, guidance 1.0). Each output is checked against the source outside the region the
waveform splice may touch. The repeat and scale 0 must reproduce `nolora` exactly.

Repainting needs a full-length val track with its human labels in the common structure schema.
The arm's `val/` clips are windowed, so they will not do. On the Mac, pick the first val track
that has an interior verse/bridge/chorus section. If you already ran the SA3 SMOKE.md
section 6 snippet, `prepared/val_full/` is the same and you can skip this.

```bash
.venv/bin/python - <<'EOF'
import json, shutil
from pathlib import Path
import pandas as pd, soundfile as sf
from ravi.eval.masks import section_masks
from ravi.structure import schema
from ravi.structure.labels import parse_audacity_labels

LABELS = Path("~/Downloads/labels").expanduser()          # the make_arms --labels-dir
m = pd.read_parquet("manifests/tracks.parquet").set_index("track_id")
out = Path("prepared/val_full"); out.mkdir(parents=True, exist_ok=True)
for tid in json.load(open("manifests/splits.json"))["val"]:
    r = m.loc[tid]
    wav = Path("prepared/44100") / f"{tid}.wav"
    if not r.has_labels or not wav.exists():
        continue
    dur = sf.info(str(wav)).frames / 44100
    segs = parse_audacity_labels(LABELS / Path(r.labels_path).name, dur)
    masks = section_masks(segs, dur)                       # pad 1 s, min 4 s, not first/last
    if not masks:
        continue
    schema.save(schema.from_human(segs, tid, duration=dur), out / f"{tid}.json")
    shutil.copy2(wav, out / f"{tid}.wav")
    sel = {"track_id": tid, "section_index": masks[0].index, "label": masks[0].label,
           "mask": [masks[0].start, masks[0].end]}
    (out / "selected.json").write_text(json.dumps(sel) + "\n"); print(sel)
    break
EOF
rsync -a prepared/val_full/ ubuntu@$IP:~/ravi/data/val_full/
```

On the box (keep the quotes; track ids contain spaces). The `ravi` bind mount lets an image
built before `ravi/eval/acestep_repaint.py` existed run the current code:

```bash
SEL=~/ravi/data/val_full/selected.json
TID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['track_id'])" $SEL)"
IDX="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['section_index'])" $SEL)"
CAPTION="$(cat ~/ravi/data/acestep/E/val/"$TID".caption.txt)"
mkdir -p ~/ravi/acestep/smoke/repaint
docker run --rm --gpus all \
  -v ~/brain-paper/ravi:/opt/ml/code/ravi:ro \
  -v ~/ravi/acestep/ckpt:/opt/acestep/checkpoints \
  -v ~/ravi/acestep/hf_home:/opt/hf_home \
  -v ~/ravi/acestep/smoke/model/adapter:/adapter:ro \
  -v ~/ravi/data/val_full:/data:ro \
  -v ~/ravi/acestep/smoke/repaint:/out \
  --entrypoint python "$IMG" -m ravi.eval.acestep_repaint \
    --audio "/data/$TID.wav" --segments "/data/$TID.json" --section-index "$IDX" --pad-s 1.0 \
    --caption "$CAPTION" --lyrics-from-segments \
    --adapter /adapter --lora-strength 0 0.5 1.0 --seed 1 --out /out \
  2>&1 | tee ~/ravi/acestep/smoke/repaint/console.log
```

What the tool does (module docstring for the upstream line references):

* It writes the source as `source_48k.wav` (float32, stereo, 48 kHz, polyphase from 44.1 kHz) and
  passes that file as `src_audio`, so the model's copy and the reference are the same samples.
* It calls `generate_music(handler, None, GenerationParams(task_type="repaint",
  instruction=TASK_INSTRUCTIONS["repaint"], repainting_start/end=<padded mask>,
  chunk_mask_mode="explicit", repaint_mode="balanced", repaint_strength=0.5, caption=...,
  lyrics=<one [label] per segment>, inference_steps=8, shift=3.0, guidance_scale=1.0, seed=1,
  thinking=False, enable_normalization=False), GenerationConfig(batch_size=1,
  use_random_seed=False, seeds=[1]))`. The LM is not loaded, since repaint skips it.
* It uses `load_lora("/adapter")`, then `set_use_lora(False)` for `nolora` and `nolora_repeat`,
  and `set_use_lora(True)` + `set_lora_scale(s)` for each strength. A scale message that reports
  skipped targets is an error. `unload_lora()` is not used as the no-LoRA control: at
  `ca1e85fe` it restores the decoder with mismatched keys ("Missing keys when restoring
  decoder") and the adapter stays active (its output differed from the base model as much as
  scale 1 did).
* It calls `torch.manual_seed(seed)` before every generation. The seed passed to
  `generate_music` only fixes the initial noise; each call also VAE-encodes the source with
  `latent_dist.sample()` (`core/generation/handler/vae_encode.py:65`), which draws from the
  unseeded global RNG. Without the global seed, two same-seed no-LoRA runs differed by up to
  0.34-0.42 inside the mask (10 % of the source RMS), which is what made scale 0 look different
  from no-LoRA in the first smoke run. With it, repeats are bit-identical.
* The splice may only touch `[start - 25 ms, end + 25 ms)`, because balanced at 0.5 resolves to
  a 25 ms waveform crossfade. Outside that span the output must equal the source within 1e-6.

Expected console lines: `✅ LoRA 'default' loaded from /adapter` (or the adapter name upstream
picks), one `[ravi] repaint <condition>: {...}` per condition and a final JSON summary. The tool
exits 1 if any condition changes the outside, if `nolora_repeat` differs from `nolora`, or if
`strength0` differs from `nolora`. Expected files in `~/ravi/acestep/smoke/repaint/`:
`source_48k.wav`, `nolora.wav`, `nolora_repeat.wav`, `strength0.wav`, `strength0.5.wav`,
`strength1.wav` and
`repaint_check.json`. The JSON records the section, the mask, the touched span, the crossfade,
the caption and lyrics, the adapter's sha256 and the ACE-Step commit.

```bash
python3 - <<'EOF'
import json
m = json.load(open("/home/ubuntu/ravi/acestep/smoke/repaint/repaint_check.json"))
print(m["track_id"], m["section"], "touched", m["touched_span"], "cf", m["wav_crossfade_s"])
print("ok", m["ok"], "outside_ok", m["outside_ok"], "repeat_ok", m["repeat_ok"], "scale0_ok", m["scale0_ok"])
for r in m["rows"]:
    print(f'{r["condition"]:12s} outside max {r["outside_max_abs"]:.2e}  inside rms diff vs source {r["inside_rms_diff"]:.4f}'
          f'  vs nolora max {r["inside_max_abs_vs_nolora"]:.3g} rel rms {r["inside_rel_rms_vs_nolora"]:.3f}'
          f'  inside rms {r["inside_rms"]:.4f}'
          f' (source {r["source_inside_rms"]:.4f})  n_out {r["n_out"]} n_src {r["n_src"]}')
EOF
```

A run passes when:

- `ok` is `true`: `outside_ok`, `repeat_ok` and `scale0_ok` all hold.
- Every `outside_max_abs` is at most 1e-6.
- Inside the mask every condition differs from the source (`inside_rms_diff` well above 0), and
  `inside_rms` is on the order of the source's: not silence, not clipping noise.
- `nolora_repeat` and `strength0` match `nolora` inside the mask (max diff 0). At scale 0 the
  PEFT delta is exactly zero (`scaling[adapter] = base * 0`), and the global seed makes the
  rest of the pipeline repeat. `strength0.5` and `strength1` differ from `nolora`, with
  `inside_rel_rms_vs_nolora` growing with the scale.
- `n_out` is within a latent frame (1920 samples) of `n_src`.
- You have listened to the mask and its seams in `nolora.wav` and `strength1.wav`.

This only checks that the adapter loads and repaint behaves as documented. The repaint go/no-go
is Tier 0.

## 8. Resume path

Resume is off by default outside SageMaker (`resume: none`). To exercise it, start the section 6
command again with fresh `checkpoints`, `model` and `output` dirs (the `work` dir can stay; the
tensors are reused, and every run gets a new `run_id`). Stop it after the epoch-10 checkpoint
line, then rerun it with `--set resume=auto`:

The containers run as root, so the files they write are root-owned: clear old run dirs with
`sudo rm -rf`. A leftover `..._final` from an earlier run makes `resume=auto` refuse to train.
Start `docker stop` only after `docker ps` shows the container; stopping a container that does
not exist yet fails, and the run then trains to the end.

```bash
docker stop -t 60 <container>      # after "[ravi] checkpoint epoch 10 (step 50): ..."
# same docker run as in section 6, plus:  --set resume=auto
```

Expected:

* The first run's `run_meta.json` has `status: interrupted`.
* The second run's `run_meta.json` has `resume: {"epoch": 10, "global_step": 50, "action":
  "train", ...}` and `effective_seed: 52`, and its argv has
  `--resume-from /opt/ml/ravi_work/resume/E_smoke_s42_h<hash6>_epoch0010`.
* The train log has `[ravi] resume: epoch 10, step 50, <N> LoRA tensors restored and
  verified`. That line comes from the wrapper's reload with `set_peft_model_state_dict`, which
  works around upstream's `strict=False` reload that loads nothing (README "Resume").
* Training continues at `Epoch 11/20` and ends with a new `..._epoch0020` and `..._final`.

Two guards to check while you are at it:

- Rerunning with `--set resume=auto` once `_final` exists fails with "already reached
  epochs=20. Set resume=publish ...". `--set resume=publish` republishes it to `model/adapter/`
  without training.
- `--set resume=auto --set seed=43` does not pick up the seed-42 checkpoints. It is a new
  identity (`E_smoke_s43_h<hash6>`) and trains from epoch 0.

## Reference run (2026-10-15, g5.2xlarge, A10G)

| Step | Result |
|---|---|
| Build | exit 0 in 869 s (uv sync 253 s, export 485 s), image 20.9 GB |
| Weights | 258 s, 9.4 GB, `verified_revision: true` |
| Kernel check | `NVIDIA A10G (8, 6) flash-attn 2.8.3 max\|diff\| 0.00195` |
| 100-step run | exit 0 in 241 s; `flash_attention_2` for preprocessing and training; 6.7-7.9 s per epoch; peak VRAM 5.0 GiB |
| Loss per epoch | 1.924, 1.555, 1.351, 1.291, 1.212, 1.190, 1.120, 1.068, 1.068, 1.073, 1.046, 1.002, 1.006, 1.021, 1.017, 0.996, 1.002, 0.981, 1.029, 0.986 |
| Repaint (`0026_copilot.mp3 Tonight NS`, verse [3.73, 26.96] s) | 64 s; outside max diff 0.0 for every row; vs `nolora` (max / rel RMS): repeat 0 / 0, scale 0 0 / 0, scale 0.5 0.925 / 0.709, scale 1 1.153 / 0.964 |
| Resume | stopped at epoch 10; rerun with `resume=auto`: 384 LoRA tensors restored and verified, seed 52, epochs 11-20, `completed` |
| Guards | `auto` after finish: "already reached epochs=20" (exit 1); `publish`: exit 0; `seed=43`: prefix `E_smoke_s43_h42e5f2` |

## 9. Push

Tag the image for ECR and push it the same way as `docker/sa3-train/build_and_push.sh` (the
instance role needs `ecr:*`; or build and push on the Mac with `PROFILE=dev-sso`):

```bash
REPO=297416720456.dkr.ecr.us-east-1.amazonaws.com/ravi-acestep-train
TAG=$(git rev-parse --short HEAD)-acestep-ca1e85fe
aws ecr describe-repositories --region us-east-1 --repository-names ravi-acestep-train >/dev/null 2>&1 \
  || aws ecr create-repository --region us-east-1 --repository-name ravi-acestep-train
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${REPO%%/*}
docker tag "$IMG" "$REPO:$TAG" && docker push "$REPO:$TAG"
```
