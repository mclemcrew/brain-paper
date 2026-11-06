# Research reproduction

We compare caption conditions on a composer corpus, retaining the data splits and evaluation settings needed to interpret each comparison. Source recordings are private and not bundled. Retained manifests, splits, configuration and metrics describe the recorded experiments; they are not a substitute for access to the inputs.

## Pipeline

1. Build and inspect the corpus manifest with `ravi.data.build_manifest`; prepare audio with `ravi.data.prepare_audio` and assign family-disjoint splits with `ravi.data.splits`.
2. Construct caption conditions with `ravi.captions.make_arms`: trigger only, human caption plus genre, ID3 tags plus BPM, machine captions, or human caption plus sections. Inspect `--help` before supplying local input paths.
3. Use the matching configuration under `configs/sa3/` or `configs/acestep/`. Docker READMEs document their separate training environments and weight requirements.
4. Use `ravi.eval.grid` and `ravi.eval.run_eval` with the corresponding evaluation configuration. Retain prompt, seed, adapter identity, duration and effective inference settings.
5. Compare held-out results using `ravi.eval.stats`. Retained `eval/` tables and their metadata document historical outputs. `eval/decisions.md` is a dated experiment record, not current app setup guidance.

Install the base project with `uv sync --locked`. Run `uv run --locked pytest ravi/tests -q`. Real-corpus tests use `RAVI_AUDIO_DIR`, `RAVI_DESC_DIR` and `RAVI_LABELS_DIR`; absent inputs produce explicit skips. Model-backed reproduction additionally needs the matching Docker environment and authorized weights. Do not treat skipped integration tests as verified reproduction.

## Local Medium recipe

The local v5 recipe uses full-generation zero-mask conditioning, whole-region preparation, one normalization gain per track, and the `rvkr` trigger. The duration conditioner stays frozen. Continuation restores weights and step count, not optimizer state.

This differs from cloud training: fp16 rather than bf16, full-generation masks rather than the mixed inpainting schedule, and no caption-field augmentation. The 16 GB workload samples 30-second windows; the 32 GB workload uses 47 seconds. Record the recipe and workload with every comparison.

Keep comparison families outside training. Preserve the distinction between human-reviewed descriptions, machine drafts and permission to train on unverified metadata. Compare checkpoints against the original model with matched prompts and seeds; do not infer preference from training loss or select the last checkpoint automatically.

We selected Medium for the pilot following composer feedback that favored its listening examples over Small. This observation informs the pilot configuration, although it does not establish a preference across composers or identify a universally preferable checkpoint. Preserve negative judgments and ending artifacts. Metric improvements and successful execution do not establish musical usefulness.
