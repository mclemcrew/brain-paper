"""Real pinned parser/merge/loader functions, with NumPy standing in for MLX.

No model is instantiated: the exact load_dit function is extracted from source
and its DiT constructor is a recorder. Tiny safetensors exercise real file parsing.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ravi.local.adapter_guard import AdapterApplicationError, require_complete_adapter_merge


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    safetensors = pytest.importorskip('safetensors.numpy')
    from safetensors import safe_open
    defs = Path(__file__).resolve().parents[2] / '.local-runtime/stable-audio-3/optimized/mlx/models/defs'
    if not (defs/'lora_merge.py').exists():
        pytest.skip('Pinned optional SA3 runtime is not installed')
    def load(path, return_metadata=False):
        if str(path).endswith('.npz'):
            with np.load(path) as archive:
                return dict(archive)
        with safe_open(path, framework='numpy') as archive:
            arrays = {key: archive.get_tensor(key) for key in archive.keys()}
            metadata = archive.metadata()
        return (arrays, metadata) if return_metadata else arrays
    mx = ModuleType('mlx.core')
    mx.float16, mx.float32, mx.array, mx.load = np.float16, np.float32, np.array, load
    mx.eval = lambda *args: None
    mlx = ModuleType('mlx'); mlx.core = mx
    monkeypatch.setitem(sys.modules, 'mlx', mlx)
    monkeypatch.setitem(sys.modules, 'mlx.core', mx)
    for package in ['audit_models', 'audit_models.defs']:
        module = ModuleType(package); module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    name = 'audit_models.defs.lora_merge'
    spec = importlib.util.spec_from_file_location(name, defs/'lora_merge.py')
    merger = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, merger)
    spec.loader.exec_module(merger)
    class Recorder:
        def __init__(self, **kwargs): self.weights = {}
        def load_weights(self, weights, strict=False): self.weights = dict(weights)
        def parameters(self): return self.weights
    def loader(filename):
        tree = ast.parse((defs/filename).read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_dit')
        namespace = {'__package__':'audit_models.defs', 'mx':mx, 'DiT':Recorder}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(defs/filename), 'exec'), namespace)
        return namespace['load_dit']
    weights = tmp_path/'weights.npz'; np.savez(weights, **{'a.weight':np.eye(2,dtype=np.float32)})
    def checkpoint(extra=False, adapter_type='lora'):
        tensors = {}
        for layer in ['a', 'missing'] if extra else ['a']:
            prefix = layer+'.parametrizations.weight.0.'
            tensors[prefix+'lora_A'] = np.ones((1,2),dtype=np.float32)
            tensors[prefix+'lora_B'] = np.ones((2,1),dtype=np.float32)
            if adapter_type == 'dora-rows':
                tensors[prefix+'magnitude'] = np.ones(2,dtype=np.float32)
        path = tmp_path/'adapter.safetensors'
        safetensors.save_file(tensors, path, metadata={'lora_config':json.dumps({'adapter_type':adapter_type,'rank':1,'alpha':1})})
        return str(path)
    return SimpleNamespace(merger=merger, loader=loader, weights=weights, checkpoint=checkpoint)


@pytest.mark.parametrize('filename', ['dit_mlx.py', 'dit_mlx_medium.py'])
@pytest.mark.parametrize('strength', [0., 1.])
def test_real_loader_resolves_guard_and_applies_exact_delta(pinned, filename, strength):
    loader = pinned.loader(filename)
    with require_complete_adapter_merge(pinned.merger):
        model = loader(pinned.weights, lora_specs=[{'path':pinned.checkpoint(), 'strength':strength, 'steps':None}], num_steps=8)
    np.testing.assert_array_equal(model.weights['a.weight'], np.eye(2)+strength*np.ones((2,2)))


def test_baseline_does_not_require_adapter(pinned):
    with require_complete_adapter_merge(pinned.merger):
        model = pinned.loader('dit_mlx.py')(pinned.weights)
        assert pinned.merger.merge_loras_into_weights({}, []) == {'merged':0,'skipped':[],'adapters':0}
    np.testing.assert_array_equal(model.weights['a.weight'],np.eye(2))


@pytest.mark.parametrize('strength', [0., 1.])
def test_real_dora_checkpoint(pinned, strength):
    with require_complete_adapter_merge(pinned.merger):
        model = pinned.loader('dit_mlx.py')(pinned.weights, dtype=np.float32,
            lora_specs=[{'path':pinned.checkpoint(adapter_type='dora-rows'),'strength':strength,'steps':None}],num_steps=8)
    merged = (np.eye(2)+np.ones((2,2)))/np.sqrt(5)
    np.testing.assert_allclose(model.weights['a.weight'], np.eye(2)+strength*(merged-np.eye(2)),atol=1e-7)


@pytest.mark.parametrize('filename', ['dit_mlx.py', 'dit_mlx_medium.py'])
def test_real_loader_cannot_silently_skip_unknown_target(pinned, filename):
    with pytest.raises(AdapterApplicationError, match='absent'), require_complete_adapter_merge(pinned.merger):
        pinned.loader(filename)(pinned.weights, lora_specs=[{'path':pinned.checkpoint(extra=True),'strength':1.,'steps':None}], num_steps=8)
