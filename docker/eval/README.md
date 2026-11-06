# Evaluation image (CLAP + MERT embeddings, FAD∞ / KAD, seam / beat / key metrics)

One image for plan Phase 4 (and the Phase 2.3 Tier-0 go/no-go): `python -m ravi.eval.run_eval`
reads the inference outputs of `ravi.eval.sa3_infer` (and ACE-Step outputs laid out the same
way), computes per-condition text-to-music and per-region inpainting metrics, and writes the
tables the paper reads.

| extractor | checkpoint | pinned revision | audio in | rows |
|---|---|---|---|---|
| `clap` | HF `lukewys/laion_clap` / `music_audioset_epoch_15_esc_90.14.pt` (LAION-CLAP music: music + AudioSet + LAION-Audio-630k; `amodel='HTSAT-base'`, non-fusion), sha256 `fae3e9c0…6dedd` | `b3708341862f` | 48 kHz mono, 10 s windows, 1 s hop | 512-d, unit norm; text tower too |
| `mert` | HF `m-a-p/MERT-v1-95M` (`trust_remote_code`) | `12af15fef9d0` | 24 kHz mono, 5 s windows, 1 s hop | 768-d, layer 12 mean over each window |

Stack: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (digest-pinned, the captioner's base;
Python 3.11, torch/torchaudio 2.6.0+cu124), torchvision 0.21.0+cu124, `laion-clap==1.1.7`,
`transformers==4.52.4`, `librosa==0.11.0`, `scipy==1.15.3`, `numpy==1.26.4` (laion-clap needs
numpy < 2), `pyloudnorm==0.1.1` (the -16 LUFS normalisation before CLAP), everything else pinned
in `requirements.txt`. Every metric definition, with the
upstream calls it relies on, is in the module docstrings: `ravi/eval/embed.py`,
`ravi/eval/metrics.py`, `ravi/eval/seams.py`, `ravi/eval/stats.py`, `ravi/eval/run_eval.py`.

## Build (on the g6e box, from the repo root)

```bash
cd ~/brain-paper
DOCKER_BUILDKIT=1 docker build -f docker/eval/Dockerfile -t ravi-eval \
    --build-arg RAVI_GIT_SHA=$(git rev-parse HEAD) .
```

The build asserts the base's Python/torch/torchaudio/CUDA versions, checks that every installed
package's requirements are met, imports the laion_clap / transformers (MERT's HuBERT classes) /
librosa / scipy symbols ravi.eval uses and runs both `--help`s. It needs no GPU and downloads no
weights.

## Weights (once, into the HF cache volume)

laion_clap fetches `roberta-base` (tokenizer + text encoder) from the hub by that name at load
time, so it goes into the cache too. ~2.4 GB (CLAP) + 0.4 GB (MERT) + 0.5 GB (RoBERTa).

```bash
mkdir -p /data/hf
docker run --rm -v /data/hf:/hf --entrypoint huggingface-cli ravi-eval \
    download lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt \
    --revision b3708341862f581175dba5c356a4ebf74a9b6651
docker run --rm -v /data/hf:/hf --entrypoint huggingface-cli ravi-eval \
    download m-a-p/MERT-v1-95M --revision 12af15fef9d0ac838c3f475bfbbf26d2060dd4f5
docker run --rm -v /data/hf:/hf --entrypoint huggingface-cli ravi-eval download roberta-base

# verify the CLAP checkpoint (the extractor also checks this sha256 on every load)
sha256sum /data/hf/hub/models--lukewys--laion_clap/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt
# fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd
```

Without the cache the first run downloads the same files itself. A local copy of the CLAP
checkpoint can be passed instead with `--clap-ckpt /path/music_audioset_epoch_15_esc_90.14.pt`.
After the downloads, `-e HF_HUB_OFFLINE=1` keeps runs off the network.

Self-test (loads both models, embeds a tone and noise, embeds two prompts, checks the cache
round trip; GPU optional):

```bash
docker run --rm --gpus all -v /data/hf:/hf -v ~/brain-paper/eval:/app/eval ravi-eval \
    ravi.eval.embed --self-test --cache-dir /app/eval/.emb_cache
```

## Eval layout

`run_eval` reads one directory per condition. `python -m ravi.eval.grid configs/eval/tier0.yaml`
writes exactly this (and `sa3_infer` does too when `--out` points at `eval/{condition}/inpaint` or
`eval/{condition}/gen`):

```
eval/tier0/base/inpaint/{track}_sec{i}_{label}_{start}-{end}_seed{s}_{raw,spliced,orig}.wav + .json
eval/tier0/base/gen/{track}_seed{s}.wav + .json
eval/tier0/B3k_s1/…
eval/tier0/anchor_lower/inpaint/…          # grid's lower anchor (donor of another alt_family)
eval/B/inpaint/strength0/…  strength0.5/…  strength1/…   # sa3_infer --lora-strength sweep
```

A `strength{v}/` sweep becomes the conditions `B@strength0`, `B@strength0.5`, `B@strength1`.
`upper`, `lower` and `test_orig` are reserved names. Outputs of one region (or prompt) under
several `_seed{s}` are **averaged over the seeds** before any summary or test (`n_seeds` is
recorded). When an `anchor_lower` condition exists it is the lower anchor; only without one does
`run_eval` build `lower` itself (donor = next test track of a different `alt_family`, read from
`--regions-json`). All conditions must agree on each region's mask and source audio
(`audio.sha256` in the sidecars).

Inputs besides the layout:

* `--train-dir prepared/44100 --splits manifests/splits.json` (always together): the training
  corpus, the FAD / KAD reference and the memorisation set (only `splits["train"]` ids are read; a
  train id that is also in val/test, or a test original inside the reference set, stops the run);
* `--test-audio prepared/44100`: the test originals `{track_id}.wav` (built-in lower-anchor donors
  and the `test_orig` reference row, which is always evaluated when there are generated clips);
* `--test-captions` (required with generated clips; a missing or empty source stops the run):
  the test captions for `test_orig` and the fallback prompt of a clip without a sidecar prompt.
  There are two real sources:
  * `eval/regions_test.json` (`ravi.eval.select_regions`): `tracks[id].captions[arm]`, the
    make_arms captions `grid` prompts with; pick the arm with `--caption-arm` (default `B`). This
    file also gives the `alt_family` per track (used as `--regions-json` by default);
  * `captions/midasheng_test.jsonl` (MiDashengLM auto-captions of the full test tracks, the arm D
    source): one `{"track_id", "caption"}` per line.

  A directory of `{track_id}.txt` also works (`data/arms/{arm}/` holds only train/val, so there is
  no `test/` directory to point at).
* `--structure-dir` (optional): structure documents in the common schema
  (`ravi/structure/schema.py`) for the evaluated WAVs, at the WAV's path relative to
  `--eval-root` with `.json` (e.g. `{dir}/B/inpaint/{stem}_spliced.json` and
  `{dir}/B/inpaint/{stem}_orig.json`), times in context seconds. They are computed **outside this
  image**, with SongFormer in the structure image (`docker/structure`, `ravi.structure.run_songformer`);
  its batch runner is manifest-driven, so the eval WAVs need a manifest of their own (or a small
  driver) before they can be run there. Gives `struct_kept`.

## Run

```bash
docker run --rm --gpus all -v /data/hf:/hf \
    -v ~/brain-paper/eval:/app/eval \
    -v /data/ravi-corpus/prepared:/app/prepared:ro \
    -v ~/brain-paper/manifests:/app/manifests:ro \
    -v ~/brain-paper/captions:/app/captions:ro \
    ravi-eval ravi.eval.run_eval \
        --eval-root eval/tier0 --base base \
        --test-audio prepared/44100 \
        --train-dir prepared/44100 --splits manifests/splits.json \
        --test-captions eval/regions_test.json --caption-arm B \
        --cache-dir eval/.emb_cache
# or: --test-captions captions/midasheng_test.jsonl --regions-json eval/regions_test.json
```

Smoke (first 2 regions / clips / train files per condition, CLAP only; with so few clips FAD∞
falls below its grid floor and is reported as NaN with the reason in `metrics_dist.parquet`):

```bash
docker run --rm --gpus all -v /data/hf:/hf -v ~/brain-paper/eval:/app/eval \
    -v /data/ravi-corpus/prepared:/app/prepared:ro -v ~/brain-paper/manifests:/app/manifests:ro \
    ravi-eval ravi.eval.run_eval --eval-root eval/tier0 --test-audio prepared/44100 \
        --train-dir prepared/44100 --splits manifests/splits.json \
        --test-captions eval/regions_test.json \
        --extractors clap --limit 2 --n-boot 1000 --dist-boot 50 --out-dir eval/tier0/_smoke
```

Useful options: `--variant raw` (score the unspliced model output instead of `_spliced`),
`--conditions base,B@strength1`, `--no-anchors`, `--mert-layer 4`, `--pca-dims 64` (0 = full
dimension), `--fad-inf-min-n`, `--dist-boot 200`, `--cluster none`, `--tempo local`,
`--grid-tightness 10`, `--key-gate 1.1`, `--memo-thresh 0.95`, `--device cpu`. Without `--gpus`
both extractors run on CPU (slow for the training corpus: ~19,500 s of audio → ~19k CLAP and ~19k
MERT windows; the embeddings are cached by file sha256 plus the extractor configuration, which
includes the checkpoint sha256, the HF revision and the torch / laion_clap / transformers
versions, so this is paid once per software stack).

## Outputs (in `--out-dir`, default `--eval-root`)

| file | rows | contents |
|---|---|---|
| `metrics_regions.parquet` | condition × output (seed) | `region_key`, `seed`, `region_sim_{clap,mert}`, `region_sim_mert_centred`, `lufs_{gen,orig}_region`, `seam_score_{start,end}` (+ `_orig_`), `seam_delta_{start,end}`, `seam_delta`, `abs_seam_delta`, `beat_grid_agree{,_start,_end}`, `beat_grid_offset_ms{,…}`, `beat_cv{,_start,_end}`, `beat_cv_orig`, `beat_cv_delta`, `key_agree` (1/0/NaN = gated), `key_conf_orig`, `key_orig`, `key_gen`, `struct_kept`, `struct_kept_orig`, `struct_kept_delta` (with `--structure-dir`), geometry; `lower_donor`, `lower_family_checked` for the built-in lower anchor |
| `metrics_regions_avg.parquet` | condition × region | the same, averaged over seeds, with `n_seeds` (what the summary and the tests use) |
| `metrics_gen.parquet` | condition × clip | `prompt`, `prompt_truncated` (CLAP's 77-token limit), `clap_text`, `lufs_in`, `memo_max_{clap,mert}`, `memo_track_*`, `memo_flag_clap` (> 0.95), `memo_flag_null_clap` (> null p99), `in_dist_subset` |
| `metrics_dist.parquet` | condition | everything of the distribution metrics, plus the diagnostics `fad_{x}_naive` (plain FAD, full dimension), `fad_{x}_pca_full_sample`, `fad_inf_status_{x}` |
| `metrics_summary.csv` | condition | `n_regions`, `n_tracks`, `n_region_outputs`, `n_seeds_{min,max}`, `{metric}_{n,mean,median}`, `n_gen`, `n_clips_dist`, `dist_subsampled`, `fad_inf_{x}` with `_ci_lo/_ci_hi`, slope, R², fit range `n_lo..n_hi`, `pca_d`, `pca_var`, `kad_{x}` with `_ci_lo/_ci_hi`, `clap_text_{mean,median}`, `n_prompt_truncated`, `memo_max_*`, `memo_flagged_clap`, `memo_flagged_null_clap`, `memo_null_p99_clap` (no plain FAD) |
| `comparisons.csv` | metric × level × condition pair | Wilcoxon signed-rank on the seed-averaged pairs: `level` = `track` (primary: per-track medians, CI by bootstrap over tracks) or `region` (secondary, cluster-bootstrap CI) for inpainting, `clip` for `clap_text`; `n`, `median_diff` (a − b) with 95% CI, `p_two_sided`, `p_greater`, `p_less`, `p_a_better`, `p_a_worse`, `p_holm_two_sided`, `p_holm_a_worse`, `p_holm_a_better`, `holm_m`, `holm_scope`; `a` is the non-base condition when `base` is involved |
| `metrics_meta.json` | — | inputs, extractor configurations (cache identity), reference sizes and PCA, FAD∞ grid and per-condition fit range / R² / d, matched clip count and the clips used, memorisation null, loudness, lower-anchor source and donors, settings, package versions |

Go/no-go (plan Phase 2.3): read the `B@strength1` vs `base` rows of `comparisons.csv` with
`level == "track"` for `region_sim_clap`, `region_sim_mert_centred`, `abs_seam_delta` and
`beat_grid_agree`, using `p_holm_a_worse` (Holm over all condition pairs of that metric and
level: the correction is per metric, not across metrics), and `fad_inf_*` / `kad_*` with their
CIs in `metrics_summary.csv` against the `test_orig` row.

## Choices to report in the paper

* **FAD∞ / KAD are relative comparisons at matched n.** Every condition and `test_orig` is
  subsampled (seeded, recorded) to the same number of clips; the values depend on n and are not
  unbiased absolute distances. Report them next to `test_orig` (real held-out music at the same
  n) and with the clip-level bootstrap CIs (200 resamples of clips: the windows of one clip are
  not independent).
* **PCA before FAD∞ / KAD** (64 dims, fitted on the reference only) and a FAD∞ grid floor of
  `max(3 d, 500)` rows: with 512-/768-d embeddings and subsets of a few hundred to a few thousand
  windows, n < d makes the FAD-vs-1/n curve non-linear and the full-dimension intercept wrong
  despite R² ≈ 0.99 (simulation in `ravi/tests/test_eval_metrics.py`: 0.31 vs a true 0.54;
  PCA'd within 15%). Plain FAD is only a diagnostic (`fad_*_naive`).
* **FAD∞ draws without replacement** (default). Gui et al. 2024 / fadtk draw with replacement,
  which extrapolates to the full-sample FAD and keeps its O(1/N) bias; without replacement the
  intercept estimates the population FAD (both are unit-tested on Gaussians with a known answer).
* **KAD bandwidth from the reference set** (as the KAD paper states; kadtk 1.1.0's code takes the
  median of the eval set) so the kernel is the same for every condition; unbiased estimator,
  α = 100.
* **Rhythm: beat-grid agreement is primary**, the IBI CV secondary. The CV uses librosa's
  global-tempo tracker (`--tempo global`): the time-varying-tempo tracker flips metrical level on
  unedited corpus music (CV 0.12–0.21 vs 0.02–0.04), but the global tracker cannot see a tempo
  jump. The grid agreement tracks both contexts at the *original's* tempo with `tightness` 10
  (shared prior: no phase flips on near-identical audio; loose: follows a 120 → 80 BPM jump) and
  counts the original's beats near each seam with a generated beat within ±70 ms.
* **Key gate**: confidence ratio ≥ 1.1 *and* best correlation ≥ 0.6 (the ratio alone passes
  white noise).
* **Loudness**: every generated clip, test original and region is normalised to −16 LUFS (gain
  only, −1 dBTP ceiling, the `prepare_audio` procedure the reference corpus went through) before
  CLAP, which is level-sensitive. MERT is not normalised: its feature extractor z-scores every
  window, so it is gain-invariant already.
* **Memorisation** uses window embeddings (any generated window vs any training window) and is
  flagged on CLAP only (MERT layer-12 windows of different tracks already exceed 0.95), both at
  the fixed 0.95 and at a null threshold: the 99th percentile of the same statistic for the test
  originals against the training set in the same run.
* **MERT region similarity** is also reported mean-centred with the reference mean
  (`region_sim_mert_centred`): raw MERT cosines between unrelated tracks are ~0.9.
* **Seam delta** compares with the original *at the same position*, which is `pad_s` (1 s) inside
  the neighbouring section, not on the real boundary (inside the mask); `abs_seam_delta` is
  reported with the signed mean.
* **Pseudo-replication**: several regions come from one track, so the primary inpainting tests
  are on per-track medians; region-level tests are secondary with cluster-bootstrap CIs.
* **Seeds** are averaged per region / prompt before pairing, never counted as extra samples.
