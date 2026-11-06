# rAvI Studio

A local Mac workspace for describing recordings, fine-tuning Stable Audio 3, comparing generated passages, and exporting audio to a DAW. The repository also contains the research pipeline for comparing metadata conditions.

This is an experimental research pilot. Medium is the primary model: the composer preferred its listening examples and found Small less useful. This feedback does not establish that every trained adapter improves on the original model.

## Start here

Use an Apple Silicon Mac running macOS 13.3 or newer, with at least 16 GB RAM. Medium downloads about 9.5 GB of weights; leave additional space for recordings, decoded audio, checkpoints, and exports. The 32 GB workload requires suitable hardware. Small is an optional diagnostic, not the default listening experience.

A native pilot build is being prepared. Until an app is attached to a tagged release, use the source instructions below. Model weights and the private recording corpus are not bundled.

1. Follow the [composer quickstart](docs/COMPOSER_QUICKSTART.md).
2. For source installation or packaging, use the [developer guide](docs/LOCAL_STUDIO.md).
3. For experimental methods and reproduction, use the [research guide](docs/RESEARCH.md).

## Run from source

Install Python 3.12, uv, Git, Node.js 22 and npm. Research audio preparation also needs ffmpeg and ffprobe on PATH.

```sh
python3 scripts/bootstrap_studio.py
.venv/bin/python -m ravi.local.server --root .local-studio
```

Open the local URL printed in Terminal. Install Medium through Settings after reviewing its model terms. Downloads require internet; processing runs locally afterward. Native packaging additionally requires Rust and Xcode command-line tools.

## Repository

- `desktop/` and `ravi/local/`: composer interface and local processing.
- `ravi/data/`, `ravi/captions/`, `ravi/train/`, `ravi/structure/`, `ravi/eval/`: research pipeline.
- `configs/`, `docker/`, `manifests/`, `captions/`, `eval/`: experiment configuration and retained evidence.
- `ravi/tests/`: regression tests.
- `web/`, `training/`, `listening-test/`: legacy v1 work, not the Studio setup path.

## Limits

A short training test verifies execution, not learned style or musical quality. Compare original and adapted takes using the same prompt, duration and seed. Keep alternate mixes in the same piece family and reserve independent comparison families. MIDI transcription is optional and can produce incorrect notes.

Windows, Linux and DAW plug-ins are not supported by this local pilot. Downloaded-app launch, signing and fresh-install checks must pass before a native build is advertised as ready.

## License

Repository code is MIT; see LICENSE. Models, upstream runtime code and bundled dependencies retain their own terms. The MIT license does not grant rights to model weights or the private corpus.
