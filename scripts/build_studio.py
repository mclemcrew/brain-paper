"""Build an offline-capable native app. Run after bootstrap_studio.py."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['PYINSTALLER_CONFIG_DIR'] = str(ROOT / '.build/pyinstaller-cache')
def run(*args, **kw): subprocess.run(list(map(str, args)), check=True, **kw)
run(sys.executable, ROOT / 'scripts/apply_runtime_patches.py')
run('npm', '--prefix', 'desktop', 'run', 'check')
run('npm', '--prefix', 'desktop', 'run', 'build')
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--name', 'ravi-local',
    '--distpath', '.build/dist', '--workpath', '.build/work', '--specpath', '.build',
    '--paths', ROOT, '--collect-all', 'mlx', '--collect-all', 'basic_pitch',
    '--paths', ROOT / '.local-runtime/stable-audio-3/optimized/mlx',
    '--paths', ROOT / '.local-runtime/stable-audio-3/optimized/mlx/scripts',
    '--hidden-import', 'sa3_mlx', '--hidden-import', 'lora_train_mlx', '--hidden-import', 'pre_encode_mlx',
    '--collect-all', 'onnxruntime', '--collect-all', 'tokenizers', '--collect-all', 'sentencepiece',
    '--collect-all', 'librosa', '--collect-all', 'resampy',
    '--collect-submodules', 'ravi.local', '--collect-submodules', 'safetensors',
    '--exclude-module', 'pyarrow', '--exclude-module', 'pandas',
    '--hidden-import', 'PIL', '--hidden-import', 'pyloudnorm', '--hidden-import', 'huggingface_hub',
    '--add-data', f'{ROOT / "desktop/dist"}:studio-ui',
    '--add-data', f'{ROOT / ".local-runtime/stable-audio-3/optimized/mlx"}:sa3-mlx',
    '--add-data', f'{ROOT / ".local-runtime/stable-audio-3/LICENSE"}:third-party/stable-audio-3',
    'scripts/local_entry.py')
resources = ROOT / 'desktop/src-tauri/resources/ravi-local'
if resources.exists(): shutil.rmtree(resources)  # Generated bundle only; avoid stale excluded dependencies.
shutil.copytree(ROOT / '.build/dist/ravi-local', resources, dirs_exist_ok=True)
env = dict(os.environ)
local_cargo = ROOT / '.local-runtime/cargo'
if local_cargo.exists():
    env.update(CARGO_HOME=str(local_cargo), RUSTUP_HOME=str(ROOT / '.local-runtime/rustup'))
    env['PATH'] = str(local_cargo / 'bin') + os.pathsep + env['PATH']
run('npm', 'run', 'tauri', '--', 'build', cwd=ROOT / 'desktop', env=env)
