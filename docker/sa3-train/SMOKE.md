# SA3 LoRA smoke test on the EC2 GPU box (Phase 2, step 2)

The goal is to show, on a g6e.xlarge (1x L40S 48 GB, 4 vCPU), that the image builds, that
flash-attn really runs, that 300 LoRA steps train without errors with a loss that does not
blow up, that demos render, and that the adapter loads in `stable_audio_3` inference
(including inpainting). Run every command from the repo root unless it says otherwise.

## 0. Box and access

```bash
# on the Mac
INSTANCE_PROFILE=<role with ECR/S3/SSM read> infra/launch_gpu_box.sh   # Deep Learning Base OSS Nvidia AMI, Docker + NVIDIA toolkit
IP=$(cat infra/.gpu_box_ip)
```

You need HF access to `stabilityai/stable-audio-3-medium-base`, `stabilityai/stable-audio-3-medium`
and `google/t5gemma-b-b-ul2` with the account behind SSM `/uaudio/hf_token` (Phase 0, item 1).

`launch_gpu_box.sh` sets `HttpTokens=required` and keeps the default IMDS hop limit of 1. A
container on the default bridge network then cannot reach the instance role. Run the
container with `--network host` whenever it needs AWS credentials (`s3_out`,
`hf_token_ssm_param`), or pass `HF_TOKEN` in from the host as shown below.

## 1. Smoke data (arm B, built by `make_arms`)

The arms come from `ravi.captions.make_arms`. Build arm B for the train and val splits on
the Mac (same manifest, splits and label exports as Phase 1):

```bash
.venv/bin/python -m ravi.captions.make_arms --manifest manifests/tracks.parquet \
    --splits manifests/splits.json --labels-dir ~/Downloads/labels \
    --prepared-root prepared --sr 44100 --out-root data/arms --arms B --split train,val
python3 -c "import json; print(json.load(open('data/arms/arms_summary.json'))['arms']['B']['n_tracks'])"
rsync -a --exclude prepared --exclude data --exclude .venv --exclude node_modules ./ ubuntu@$IP:~/brain-paper/
rsync -a data/arms/B/ ubuntu@$IP:~/ravi/data/arms/B/
ssh -i ~/.ssh/fxjepa-oct2026.pem ubuntu@$IP
```

This writes `data/arms/B/{train,val}/` with one `{track_id}.wav` and one `{track_id}.txt` per
track, plus `filelist.txt`, `manifest.csv` and `trackset.sha256`. The trainer reads
`{arm}/train` (here `~/ravi/data/arms/B/train`, mounted at `/data/train`), never `{arm}`
itself, which also holds `val`. The smoke run trains on all of `B/train`; 300 steps take the
same time whatever the number of clips. Write down `N_TRAIN`, the `train` count printed above.

Window: every clip is at most the trainer's `sample_size`,
`(int(47*sr)//4096)*4096` = **2,072,576 samples** at 44.1 kHz (46.997 s, `train_lora.py:105`).
`make_arms` crops longer tracks to a deterministic window (start 0 unless a window whose
edges sit on section boundaries is cleaner, `ravi/captions/windows.py`;
`{track_id}.window.json` records where), so upstream never random-crops.
`long_clip_policy: error` in `smoke.yaml` checks this.

Caption files hold fields separated by `␞` (U+241E, `caption_aug.field_sep`), the trigger
`rvkr` first: `rvkr␞{human caption}␞{genre tag}`. A junk caption falls back to
`rvkr␞{genre tag or ID3 tags}`. Augmentation shuffles and drops whole fields only, so
commas inside the human caption are kept. The model always sees the fields joined with
`", "` (`join_sep`), including in the unaugmented demos.

`smoke.yaml` protects field 1 (the human caption) from being dropped
(`non_droppable_fields: [1]`), so `p_drop_field` can only drop the genre tag.

## 2. Build (on the box)

```bash
cd ~/brain-paper
PUSH=0 PROFILE= docker/sa3-train/build_and_push.sh | tee ~/build.log
IMG=$(tail -1 ~/build.log)           # 297416720456.dkr.ecr.us-east-1.amazonaws.com/ravi-sa3-train:<sha>-sa3-77e0398c
docker tag "$IMG" ravi-sa3-train:dev # the name section 6 uses
```

The build log should end its sanity step with
`python 3.10.x | torch 2.7.1+cu126 cuda 12.6 cxx11abi ... | flash-attn 2.6.3 | lightning 2.5.5`.

## 3. GPU + flash-attn kernel check

```bash
docker run --rm -i --gpus all --entrypoint python "$IMG" - <<'EOF'
import torch, flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q)
ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, q, q))).transpose(1, 2)
print(torch.cuda.get_device_name(0), "flash-attn", flash_attn.__version__, "max|diff|", (o - ref).abs().max().item())
EOF
```

Expected: `NVIDIA L40S flash-attn 2.6.3 max|diff| <about 1e-2>`. An `undefined symbol` error
means the wheel does not match torch's ABI. In that case rebuild with
`FLASH_ATTN_WHEEL= MAX_JOBS=4 PUSH=0 PROFILE= docker/sa3-train/build_and_push.sh`.

## 4. Dry run (no GPU work; validates data and prints the exact train_lora.py argv)

```bash
docker run --rm \
  -v ~/ravi/data/arms/B/train:/data/train:ro \
  -v ~/brain-paper/docker/sa3-train/smoke.yaml:/config/smoke.yaml:ro \
  "$IMG" --config /config/smoke.yaml --dry-run
```

Expected: `[ravi] data: {"sample_size": 2072576, ..., "n_audio": <N_TRAIN>, "n_missing_txt": 0,
"n_bad_trigger": 0, "n_longer_than_window": 0, ...}`, then `"identity": {"prefix":
"B_smoke_s42_h<hash6>", ...}`, `"resume": "none"` and an argv containing
`--model medium-base ... --rank 16 --adapter_type dora-rows ... --steps 300 ... --duration 47.0
... --checkpoint_every 100 ... --exclude seconds_total`. Every checkpoint of this run is named
after that prefix; the hash covers model, rank, alpha, adapter type, lr, steps, duration,
include/exclude, data dir name and the caption_aug block, so changing any of them starts a
new identity instead of resuming the old one.

## 5. 300-step run

```bash
export HF_TOKEN=$(aws ssm get-parameter --region us-east-1 --name /uaudio/hf_token \
  --with-decryption --query Parameter.Value --output text)
mkdir -p ~/ravi/hf_home ~/ravi/smoke/{checkpoints,model,output,work}
docker run --rm --gpus all --shm-size 16g --network host \
  -e HF_TOKEN \
  -v ~/ravi/data/arms/B/train:/data/train:ro \
  -v ~/ravi/hf_home:/opt/hf_home \
  -v ~/ravi/smoke/checkpoints:/opt/ml/checkpoints \
  -v ~/ravi/smoke/model:/opt/ml/model \
  -v ~/ravi/smoke/output:/opt/ml/output \
  -v ~/ravi/smoke/work:/opt/ml/ravi_work \
  -v ~/brain-paper/docker/sa3-train/smoke.yaml:/config/smoke.yaml:ro \
  "$IMG" --config /config/smoke.yaml 2>&1 | tee ~/ravi/smoke/console.log
```

Stop a run with `docker stop -t 60 <container>`, not the default 10 s: the wrapper sends the
trainer SIGTERM and waits up to 30 s before SIGKILL, then writes `run_meta.json` with
`status: interrupted`.

`--shm-size` is needed because the 3 DataLoader workers pass 47 s stereo clips through
shared memory, and Docker's 64 MB default is too small for that. The first run also
downloads medium-base and T5Gemma into `~/ravi/hf_home`.

### Expected console lines

| Line | Source |
|---|---|
| `[ravi] env: {... "gpu": "NVIDIA L40S", "flash_attn_import": "ok", ...}` | wrapper preflight |
| `[ravi] caption augmentation installed: {..., "field_sep": "\u241e", ...}` | `ravi/train/_launch_train_lora.py` |
| `Found <N_TRAIN> files` | `stable_audio_3/data/dataset.py:230` |
| `LoRA config: rank=16, alpha=16, adapter_type=dora-rows` and `  exclude: ['seconds_total']` | `training/diffusion.py:93-97` |
| `lora layers: <N>` | `training/diffusion.py:117` |
| `Demo sample 0..3: prompt=rvkr, ... seconds_total=47` (clean captions, fields joined with `, `) | `train_lora.py:240-244` |
| `[ravi] checkpoint step 100/200/300: /opt/ml/checkpoints/B_smoke_s42_h<hash6>_step000{100,200,300}.safetensors` | wrapper |
| `[ravi] trainer exited with code 0` | wrapper |

These must not appear: `flash_attn not installed` (`models/transformer.py:26`),
`Couldn't load file` (`data/dataset.py:314`, a clip silently replaced by another) and the
field separator `␞` in any prompt (it must always be rendered as `, `):

```bash
grep -cE "flash_attn not installed|Couldn't load file" ~/ravi/smoke/console.log   # expect 0
grep -c "prompt=.*$(printf '\xe2\x90\x9e')" ~/ravi/smoke/console.log            # expect 0
```

### Expected files

```
~/ravi/smoke/checkpoints/  B_smoke_s42_h<hash6>_step000{100,200,300}.safetensors   (fp16, for inference)
                           B_smoke_s42_h<hash6>_step000{100,200,300}.ckpt          (fp32, for resume)
                           B_smoke_s42_h<hash6>_run_meta.json
~/ravi/smoke/model/        adapter.safetensors (= step 300)  run_meta.json
~/ravi/smoke/output/data/  train.log  metrics_from_000000.csv  run_meta.json
~/ravi/smoke/output/data/demos/  demo_step000001_cfg{2,4,7}.wav  demo_step000101_cfg{2,4,7}.wav
                                 demo_step000201_cfg{2,4,7}.wav
```

Demos are rendered at global steps 1, 101 and 201 (`diffusion.py:899-900`), not at 300.
Each WAV holds the 4 demo clips back to back. These are inpaint-callback demos with
randomly drawn mask types (mostly full generation). The step-1 demo is effectively the
base model.

### Pass checks

```bash
python3 - <<'EOF'
import json
m = json.load(open("/home/ubuntu/ravi/smoke/output/data/run_meta.json"))
c = [r["train/loss"] for r in m["loss_curve"]]
k = max(1, len(c) // 3)
print(m["status"], m["sa3_commit"], m["env"]["gpu"], m["env"]["flash_attn_import"])
print("points", len(c), "first third", sum(c[:k]) / k, "last third", sum(c[-k:]) / k)
print("argv", " ".join(m["train_lora_argv"]))
EOF
docker run --rm -v ~/ravi/smoke/model:/m --entrypoint python "$IMG" -c "
import json; from safetensors import safe_open
f = safe_open('/m/adapter.safetensors', 'pt'); keys = list(f.keys())
print(len(keys), 'tensors;', json.loads(f.metadata()['lora_config']))
print('seconds_total keys:', [k for k in keys if 'seconds_total' in k])"
```

A run passes when:

- `status` is `completed` and `sa3_commit` is `77e0398c…`.
- There are 30 loss points (every 10 steps), all finite, and the last-third mean is not above
  the first-third mean. Flow-matching loss is noisy, and the adapter starts from the base
  model's own behaviour, so expect a small decrease, not a large one.
- `lora_config` shows `rank 16`, `adapter_type dora-rows`, `exclude ['seconds_total']`, and
  `seconds_total keys: []`. The metadata also carries `ravi_arm`, `ravi_seed`, `ravi_step`
  and `ravi_cfg_hash`.
- You have listened to the demos: step 201 should not be noise or silence.

## 6. Inference check (`ravi.eval.sa3_infer`)

Load the adapter in inference with `ravi.eval.sa3_infer`: generate from text, then inpaint
one interior section of a val track. The default model is the post-trained `medium`
(8 steps, cfg 1.0), which is the serving target. The LoRA was trained on `medium-base`
(upstream: "LoRAs are trained on the base checkpoint ... can be applied to the post-trained
model", `docs/guides/model-overview.md:129`). Use `--model medium-base` for the base model
(50 steps, cfg 7 by default).

Inpainting needs a full-length val track and its human labels in the common structure
schema. The arm's `val/` WAVs are cropped to the 47 s window, so they will not do. On the
Mac, pick the first val track that has an interior verse/bridge/chorus section:

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

On the box (track ids contain spaces, so keep the quotes). The `ravi` bind mount lets an
image built before `ravi/eval/` existed run the current code:

```bash
export HF_TOKEN=$(aws ssm get-parameter --region us-east-1 --name /uaudio/hf_token \
  --with-decryption --query Parameter.Value --output text)
mkdir -p ~/ravi/smoke/infer
RUN=(docker run --rm --gpus all --network host -e HF_TOKEN
     -v ~/brain-paper/ravi:/opt/ml/code/ravi:ro
     -v ~/ravi/smoke/model:/adapter:ro
     -v ~/ravi/data:/data:ro
     -v ~/ravi/hf_home:/opt/hf_home
     -v ~/ravi/smoke/infer:/out
     --entrypoint python ravi-sa3-train:dev -m ravi.eval.sa3_infer)

# (a) two val captions, without and with the LoRA
find ~/ravi/data/arms/B/val -name '*.txt' ! -name filelist.txt | sort | head -2 \
  | while read -r f; do cat "$f"; echo; done > ~/ravi/smoke/infer/prompts.txt
"${RUN[@]}" generate --prompt-file /out/prompts.txt --duration 30 --seed 1 --out /out/gen/nolora
"${RUN[@]}" generate --lora /adapter/adapter.safetensors --lora-strength 1.0 \
    --prompt-file /out/prompts.txt --duration 30 --seed 1 --out /out/gen/lora
# optional, the training base model:
"${RUN[@]}" generate --model medium-base --lora /adapter/adapter.safetensors \
    --prompt-file /out/prompts.txt --duration 30 --seed 1 --out /out/gen/base_lora

# (b) inpaint one interior section of the val track, without the LoRA and at strength 0 / 0.5 / 1.0
SEL=~/ravi/data/val_full/selected.json
TID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['track_id'])" $SEL)"
IDX="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['section_index'])" $SEL)"
PROMPT="$(cat ~/ravi/data/arms/B/val/"$TID".txt)"
"${RUN[@]}" inpaint --audio "/data/val_full/$TID.wav" --segments "/data/val_full/$TID.json" \
    --section-index "$IDX" --pad-s 1.0 --prompt "$PROMPT" --seed 1 --out /out/inpaint/nolora
"${RUN[@]}" inpaint --lora /adapter/adapter.safetensors --lora-strength 0 0.5 1.0 \
    --audio "/data/val_full/$TID.wav" --segments "/data/val_full/$TID.json" \
    --section-index "$IDX" --pad-s 1.0 --prompt "$PROMPT" --seed 1 --out /out/inpaint/lora
```

Expected files:

```
~/ravi/smoke/infer/gen/{nolora,lora,base_lora}/  1_<slug>-<hash6>.wav + .json   (2 prompts each)
~/ravi/smoke/infer/inpaint/nolora/               <track>_sec<i>_<label>_<start>-<end>_{raw,spliced,orig}.wav + .json
~/ravi/smoke/infer/inpaint/lora/strength{0,0.5,1}/   the same four files per strength
```

Each run with the LoRA prints `Loading LoRA 0 from /adapter/adapter.safetensors`
(`lora/loader.py:76`). Each sidecar JSON records the prompt as given and as rendered (`␞`
becomes `, `), steps, cfg, seed, strength, the adapter's sha256, `lora_config` and `ravi_*`
metadata, the SA3 commit and the GPU. Inpainting sidecars also record the section and the
padded mask in track seconds, the context window (47 s by default, `--context-s`), the
latent-frame region the model actually regenerated, and
`raw_vs_source_outside_mask` (the VAE round-trip error). `_spliced.wav` is the source with
only the mask taken from the model, using 50 ms fades inside the mask. The tool re-reads
it from disk and fails if any sample outside the mask differs from the source.
`_orig.wav` is the source context itself, the upper anchor.

Checks:

```bash
docker run --rm -i -v ~/ravi/smoke/infer:/out --entrypoint python ravi-sa3-train:dev - <<'EOF'
import glob, json, numpy as np, soundfile as sf
def raw(d): return sorted(glob.glob(f"/out/inpaint/{d}/*_raw.wav"))[0]
a = {d: sf.read(raw(d))[0] for d in ("nolora", "lora/strength0", "lora/strength0.5", "lora/strength1")}
for d in a:
    print(f"{d:18s} max|diff vs nolora| {np.abs(a[d] - a['nolora']).max():.3g}")
for j in sorted(glob.glob("/out/inpaint/**/*.json", recursive=True)):
    m = json.load(open(j))
    print(j.split("/out/")[1], m["lora_strength"], m["spliced_outside_mask_bit_identical"],
          "outside rms", round(m["raw_vs_source_outside_mask"]["rms"], 4))
for d in ("nolora", "lora"):
    g = sorted(glob.glob(f"/out/gen/{d}/*.wav"))
    print(d, [round(float(np.sqrt(np.mean(sf.read(f)[0] ** 2))), 4) for f in g])
EOF
```

A run passes when:

- Strength 0 matches no LoRA. For DoRA-rows, strength 0 returns the base weight unchanged
  (`lora/model.py:245-246`), so the difference is 0, or float noise if the kernels are not
  deterministic. Strength 0.5 and 1.0 differ from no LoRA.
- `spliced_outside_mask_bit_identical` is `true` everywhere, and the outside-mask RMS error is
  small compared with the track level. A large value means the mask or context is misplaced.
- None of the generated or inpainted files is silent or noise. Listen to the inpainted region
  and its seams in `_spliced.wav`.

This only checks that the adapter loads and runs. The inpainting go/no-go is Tier 0
(Phase 2.3).

## 7. Optional: resume path

Resume is off by default outside SageMaker (`resume: none`). To exercise it, start the
step-5 command again with fresh dirs, stop it after the step-200 checkpoint line, then rerun
it with `--set resume=auto`:

```bash
docker stop -t 60 <container>      # after "[ravi] checkpoint step 200: ..."
# same docker run as in step 5, plus:  --set resume=auto
```

Expected: the first run's `run_meta.json` has `status: interrupted`; the second has
`resume: {"completed_steps": 200, "action": "train", ...}`, a child argv with
`--steps 100 --seed 242 --lora_checkpoint /opt/ml/checkpoints/B_smoke_s42_h<hash6>_step000200.ckpt`
(the fp32 `.ckpt`, not the fp16 `.safetensors`), and a new `..._step000300.safetensors`. The
resume is weights-only: AdamW state is not restored, because upstream does not save it.

Two guards to check while you are at it:

- Rerunning with `--set resume=auto` once step 300 exists fails with "already reached
  steps=300. Set resume=publish ...". `--set resume=publish` copies it to
  `model/adapter.safetensors` without training.
- `--set resume=auto --set seed=43` does not pick up the seed-42 checkpoints: it is a new
  identity (`B_smoke_s43_h<hash6>`) and trains from step 0.

## 8. Push

```bash
PROFILE= docker/sa3-train/build_and_push.sh   # needs ecr:* on the instance role; or run on the Mac with PROFILE=dev-sso
```
