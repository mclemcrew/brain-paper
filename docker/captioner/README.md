# Auto-captioner image (MiDashengLM-7B, Qwen2.5-Omni-7B fallback)

One CUDA image for plan Phase 1 step 6: `python -m ravi.captions.autocaption` writes one short
caption per track to JSONL, which `ravi.captions.make_arms --autocaptions` reads for arm D.

| `--model` | checkpoint | license | pinned revision | audio in |
|---|---|---|---|---|
| `midasheng` (default) | HF `mispeech/midashenglm-7b-1021-bf16` | Apache-2.0 | `f7b0fdb745fc` | 16 kHz mono, whole clip (no hard cap; logged) |
| `qwen-omni` | HF `Qwen/Qwen2.5-Omni-7B` (thinker only, talker disabled) | Apache-2.0 | `ae9e1690543f` | 16 kHz mono, cropped at 300 s |

Stack: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (digest-pinned; Python 3.11, torch and
torchaudio 2.6.0+cu124 from the base), `transformers==4.52.4` (the version in MiDashengLM's
`config.json`; its remote code needs >= 4.52), everything else pinned in `requirements.txt`.
MiDashengLM loads with `trust_remote_code=True` (its model and processor classes live in the
HF repo, not in transformers). The verified upstream calls, the audio constraints and the
post-processing rules are in the docstring of `ravi/captions/autocaption.py`.

## Build (on the g6e box, from the repo root)

```bash
cd ~/brain-paper
DOCKER_BUILDKIT=1 docker build -f docker/captioner/Dockerfile -t ravi-captioner \
    --build-arg RAVI_GIT_SHA=$(git rev-parse HEAD) .
```

The build asserts the base's Python/torch/torchaudio/CUDA versions, checks that every installed
package's requirements are met, imports every transformers symbol the two models need and runs
`--help`. It needs no GPU.

## Host layout

```
/data/ravi-corpus/prepared/44100/{track_id}.wav   python -m ravi.data.prepare_audio --sr 44100
~/brain-paper/manifests/tracks.parquet            the repo's manifest + splits.json (mounted at /app/manifests)
~/brain-paper/data/arms/A/{train,val}/            make_arms' windowed WAVs (mounted at /app/data/arms)
~/brain-paper/captions/                           output (mounted at /app/captions)
/hf                                               HF cache volume (HF_HOME=/hf in the image), ~16 GB per model
```

```bash
sudo mkdir -p /hf && sudo chown $USER /hf
mkdir -p ~/brain-paper/captions
```

## Fetch the weights (once, needs network)

```bash
RUN="docker run --rm --gpus all --network host --shm-size 8g \
     -v /data:/data:ro -v /hf:/hf \
     -v $HOME/brain-paper/manifests:/app/manifests:ro -v $HOME/brain-paper/captions:/app/captions \
     ravi-captioner"

$RUN --prefetch                      # MiDashengLM @ pinned revision
$RUN --prefetch --model qwen-omni    # only if the fallback is needed
```

Afterwards add `-e HF_HUB_OFFLINE=1` so nothing is re-resolved mid-run. The MiDashengLM
remote code (`modeling_midashenglm.py` etc.) comes from the same pinned snapshot.

## Run on the L40S

```bash
RUN="docker run --rm --gpus all --network host --shm-size 8g -e HF_HUB_OFFLINE=1 \
     -v /data:/data:ro -v /hf:/hf \
     -v $HOME/brain-paper/manifests:/app/manifests:ro -v $HOME/brain-paper/captions:/app/captions \
     -v $HOME/brain-paper/data/arms:/app/data/arms:ro \
     ravi-captioner"
COMMON="--manifest manifests/tracks.parquet --splits manifests/splits.json --sr 44100"
FULL="--prepared-root /data/ravi-corpus/prepared"     # full prepared tracks
WIN="--audio-dir data/arms/A/{split}"                 # make_arms' windowed WAVs (any arm: same audio)

# 1. pilot: the first 3 manifest tracks, plus the longest clip (184 s), into a separate file
$RUN $COMMON $FULL --out captions/pilot_midasheng.jsonl --limit 3
$RUN $COMMON $FULL --out captions/pilot_midasheng.jsonl --resume --track-id "0450_copilot.mp3 Me and my friends"

# 2. train/val: captions of the exact 46.997 s windows the arms train on (arm D input)
$RUN $COMMON $WIN  --split train,val --out captions/midasheng.jsonl --resume
#    test: captions of the full tracks, a separate file
$RUN $COMMON $FULL --split test      --out captions/midasheng_test.jsonl --resume
#    rerun the same command after a crash or failures

# optional: batching once the pilot shows batch-size 1 and 4 give the same captions
$RUN $COMMON $WIN --split train,val --out captions/midasheng.jsonl --resume --batch-size 4

# fallback model
$RUN $COMMON $WIN --split train,val --out captions/qwen-omni.jsonl --model qwen-omni --resume
```

`--audio-dir` reads `{DIR}/{track_id}.wav` (`{split}` is replaced by each track's split) and
ignores the `.wav.json` / `.window.json` / `.txt` siblings; the WAV rate must equal `--sr`.
make_arms' default `--link symlink` makes *absolute* symlinks to the prepared WAVs for tracks
that fit the window: they resolve in the container only if make_arms ran on the box with
`--prepared-root /data/ravi-corpus/prepared` (mounted at the same path). Otherwise build the
arms with `--link hardlink` or `--link copy`; a dangling link shows up as a `load` failure.

The container runs as root, so `sudo chown -R $USER ~/brain-paper/captions` afterwards.
Exit code 1 means at least one track failed (see `_failures.jsonl`); rerun with `--resume`.

Flags: `--model {midasheng,qwen-omni}`, `--limit N` (first N of the selection), `--track-id ID
...` (repeatable), `--max-words 15`, `--batch-size 1`, `--resume`, `--max-audio-seconds S`
(crop to the first S seconds; default whole clip), `--revision` (override the pinned commit),
`--failures PATH`, `--device cuda:0`, `--log-level`, `--audio-dir DIR`, `--splits PATH`
(default `manifests/splits.json`), `--split train,val|test` (default: whole manifest).

## Outputs

```
captions/midasheng.jsonl            {"track_id","caption","raw","model","n_words","audio_seconds","elapsed_s",
                                     "audio_source","audio_dir","split",
                                     "prompt_variant","echo_retry","echo_reason","raw_primary"}
captions/midasheng.jsonl.meta.json  model id + revision, prompt, max_words, decoding, sr, audio_source,
                                    audio_dir (as given), splits so far, audio_seconds per track,
                                    transformers, RAVI_GIT_SHA
captions/_failures.jsonl            {"track_id","model","stage","error","time"[,"raw","raw_primary"]},
                                    stage = load | generate | echo
```

* Append-only, fsynced per line. Without `--resume` an existing non-empty output is refused.
  With `--resume`, ids already in the file are skipped, failed ids are retried, a torn last line
  is cut, and a file written with a different model, revision, prompt, `--max-words`,
  `--max-audio-seconds`, `--sr`, audio source (`prepared` vs `audio_dir`) or `--audio-dir` is
  refused, so windowed and full-track captions never share a file. train then val into one file
  with the same `--audio-dir .../{split}` is fine.
* `caption` is post-processed (first line, no quotes / trailing period / `<n> bpm` phrases,
  <= `--max-words` cut at a clause boundary, first letter lowercased); `raw` is the decoded model
  output, verbatim. BPM removals and word-cap cuts are logged per track.
* Prompt v2 (wording in `ravi/captions/autocaption.py`, `PROMPTS`) states the task positively.
  An answer that echoes a prompt (substring, >= 60% prompt content words, or < 4 content words)
  is retried once with the `alt` wording (`echo_retry: true`); a second echo is an `echo`
  failure, not a caption.
* `elapsed_s`: audio load + inference + post-processing per track (batch time / batch size),
  model load excluded.

## Check in the 3-clip pilot

1. The model loads with `torch_dtype=bfloat16` and the remote code runs under 4.52.4 (no
   `ImportError` / `AttributeError` from `modeling_midashenglm.py`); GPU memory ~17 GB.
2. `raw` is only the answer (MiDashengLM's `generate` returns new tokens only; the script
   strips an echoed prompt either way). No `<|im_end|>` or system-prompt text in `raw`.
3. Captions are on-prompt: one line, genre/mood/instruments/tempo feel, no lyrics talk, few BPM
   numbers. Look at how often the post-processor had to strip BPM or cut words (log lines
   `stripped BPM`, `words cut to`); if often, adjust the prompt, not the post-processor.
4. The 184 s clip: caption quality vs. the ~30 s clips (upstream benchmarks use 30 s; the
   encoder handles longer input in 10.08 s windows). Train/val use the <= 47 s windows anyway;
   this matters for the full-track test captions. If it degrades, use `--max-audio-seconds 30`
   into a fresh file.
5. `elapsed_s` per clip, to size the full run, and whether `--batch-size 4` reproduces the
   batch-size-1 captions exactly (left padding is the processor default; greedy decoding in
   bf16 can still differ slightly across batch sizes).
