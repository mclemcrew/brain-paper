# Build and verify Studio

Use Apple Silicon macOS 13.3+, Python 3.12, uv, Git, Node.js 22/npm. Native builds also require Rust and Xcode command-line tools.

```sh
python3 scripts/bootstrap_studio.py
.venv/bin/python -m ravi.local.server --root .local-studio
```

The Python server supplies the session token and local API; Vite alone is not a working Studio. Browser and native sessions use separate application-data stores.

Bootstrap checks out the Stable Audio commit declared in `ravi/local/catalog.py` and applies the tracked generation-mask patch. Never substitute an unpatched trainer. The v5 recipe records effective settings and rejects incompatible continuation. Basic Pitch 0.4.0 uses its ONNX model without its legacy TensorFlow dependency set.

```sh
.venv/bin/python -m pytest ravi/tests -q
node --test desktop/src/onboarding.test.mjs desktop/src/workflow.test.mjs
npm --prefix desktop run check
npm --prefix desktop run build
.venv/bin/python scripts/build_studio.py
```

Native output: `desktop/src-tauri/target/release/bundle/macos/rAvI Studio.app`. Weights are downloaded separately. The development build is ad-hoc signed; do not describe it as notarized.

After explicit model installation, this creates a synthetic integration collection and runs generation, short training, adapted generation, editing and transcription:

```sh
.venv/bin/python scripts/smoke_studio.py --root .local-studio --model sa3-medium --profile 16gb
```

Use `--worker-binary .build/dist/ravi-local/ravi-local` to test the packaged worker. Small diagnostics require explicit `--model sa3-small --profile 8gb`. Synthetic checks do not establish music quality.

Before distributing an app, verify a downloaded copy with fresh storage, real file-picker import, Medium download, short training, continuation, cancellation, matched renders, export and restart. Record hardware and memory pressure. Test offline processing after installation. Do not count a Small smoke run as Medium acceptance. Check third-party notices and downloaded-app launch restrictions before publication.

## Icon source

Edit `desktop/src-tauri/icons/icon.svg`, then run `npm --prefix desktop run tauri -- icon src-tauri/icons/icon.svg --output src-tauri/icons`. The native bundle explicitly selects the resulting ICNS and PNG. Rebuild after changing the icon.
