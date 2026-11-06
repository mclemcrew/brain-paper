"""Explicit developer setup; downloads code/dependencies, never composer audio."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ravi.local.catalog import UPSTREAM_COMMIT

def run(*args): subprocess.run(list(map(str,args)), check=True, cwd=ROOT)
runtime = ROOT / '.local-runtime/stable-audio-3'
if not runtime.exists():
    run('git', 'clone', 'https://github.com/Stability-AI/stable-audio-3.git', runtime)
if runtime.exists(): run(sys.executable, ROOT / 'scripts/apply_runtime_patches.py', '--reverse')
run('git', '-C', runtime, 'checkout', '--detach', UPSTREAM_COMMIT)
run(sys.executable, ROOT / 'scripts/apply_runtime_patches.py')
if not (ROOT / '.venv').exists(): run('uv', 'venv', '--python', '3.12')
run('uv', 'sync', '--locked', '--extra', 'studio', '--extra', 'midi', '--extra', 'package')
# Basic Pitch's legacy TensorFlow constraint excludes Python 3.12. We use its
# bundled ONNX model; all needed runtime dependencies are declared in our extras.
run('uv', 'pip', 'install', '--python', ROOT / '.venv/bin/python', '--no-deps', 'basic-pitch==0.4.0')
run('npm', '--prefix', 'desktop', 'ci')
run('npm', '--prefix', 'desktop', 'run', 'build')
print('Ready: .venv/bin/python -m ravi.local.server --root .local-studio')
