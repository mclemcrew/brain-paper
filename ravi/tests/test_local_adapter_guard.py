"""Exercise the studio boundary without loading model weights or requiring MLX."""
from types import SimpleNamespace

import numpy as np
import pytest

from ravi.local.adapter_guard import AdapterApplicationError, require_complete_adapter_merge


def runtime(layers=None, stats=None):
    params = {'lora_A': np.ones((1, 2)), 'lora_B': np.ones((2, 1))}
    calls = []
    module = SimpleNamespace(
        _PARAMS_FOR={'lora': ('lora_A', 'lora_B')},
        _resolve_path=lambda p: p,
        _parse_adapter=lambda p: ('lora', 1.0, {'a': params} if layers is None else layers),
        _layer_to_npz_key=lambda name: name + '.weight',
        _weight_as_2d=lambda w: (w, lambda x: x),
        _check_shapes=lambda *args: None,
    )
    def merge(weights, paths, strength=1.0, log=None):
        calls.append(paths)
        return stats if stats is not None else {'merged': 1, 'skipped': [], 'adapters': len(paths)}
    def prepare(weights, specs, num_steps, log=None, gate_all=False):
        for spec in specs:
            module.merge_loras_into_weights(weights, [spec['path']], strength=spec['strength'], log=log)
    module.merge_loras_into_weights = merge
    module.prepare_loras = prepare
    return module, calls, params


def test_complete_adapter_reaches_merge_and_restores_runtime():
    module, calls, _ = runtime()
    original = module.merge_loras_into_weights
    with require_complete_adapter_merge(module):
        module.prepare_loras({'a.weight': np.eye(2)}, [{'path': 'test', 'strength': 1, 'steps': None}], 8)
    assert calls == [['test']]
    assert module.merge_loras_into_weights is original


@pytest.mark.parametrize('targets', ['empty', 'unmatched', 'partial', 'missing_param', 'nan'])
def test_invalid_adapter_rejected_before_model_mutation(targets):
    module, calls, params = runtime()
    layers = {'a': dict(params)}
    if targets == 'empty': layers = {}
    if targets == 'unmatched': layers = {'unknown': params}
    if targets == 'partial': layers['unknown'] = params
    if targets == 'missing_param': del layers['a']['lora_B']
    if targets == 'nan': layers['a']['lora_A'] = np.array([[np.nan, 0]])
    module._parse_adapter = lambda _: ('lora', 1., layers)
    original = module.merge_loras_into_weights
    with pytest.raises(AdapterApplicationError), require_complete_adapter_merge(module):
        module.merge_loras_into_weights({'a.weight': np.eye(2)}, ['test'])
    assert calls == []
    assert module.merge_loras_into_weights is original


@pytest.mark.parametrize('stats', [None, {'merged': 0, 'skipped': [], 'adapters': 1},
    {'merged': 1, 'skipped': ['unknown'], 'adapters': 1},
    {'merged': 1, 'skipped': [], 'adapters': 0}])
def test_runtime_must_report_complete_merge(stats):
    module, _, _ = runtime()
    module.merge_loras_into_weights = lambda *args, **kwargs: stats
    with pytest.raises(AdapterApplicationError), require_complete_adapter_merge(module):
        module.merge_loras_into_weights({'a.weight': np.eye(2)}, ['test'])


def test_step_gating_cannot_bypass_verification():
    module, calls, _ = runtime()
    with pytest.raises(AdapterApplicationError), require_complete_adapter_merge(module):
        module.prepare_loras({'a.weight': np.eye(2)}, [{'path': 'test', 'strength': 1, 'steps': (2, 8)}], 8)
    assert calls == []


def test_strength_zero_still_requires_a_valid_adapter():
    module, calls, _ = runtime(layers={})
    with pytest.raises(AdapterApplicationError), require_complete_adapter_merge(module):
        module.merge_loras_into_weights({'a.weight': np.eye(2)}, ['test'], strength=0)
    assert calls == []


@pytest.mark.parametrize('strength,scaling,kind', [(np.nan, 1., 'lora'), (1., np.inf, 'lora'), (1., 1., 'unknown')])
def test_invalid_strength_scaling_or_type(strength, scaling, kind):
    module, calls, params = runtime()
    module._parse_adapter = lambda _: (kind, scaling, {'a': params})
    with pytest.raises(AdapterApplicationError), require_complete_adapter_merge(module):
        module.merge_loras_into_weights({'a.weight': np.eye(2)}, ['test'], strength=strength)
    assert calls == []


def test_shape_failure_prevents_merge():
    module, calls, _ = runtime()
    def wrong_shape(*args):
        raise ValueError('adapter rank or weight dimensions do not match')
    module._check_shapes = wrong_shape
    with pytest.raises(ValueError, match='dimensions'), require_complete_adapter_merge(module):
        module.merge_loras_into_weights({'a.weight': np.eye(2)}, ['test'])
    assert calls == []


def test_all_adapters_preflight_before_any_mutation():
    module, calls, params = runtime()
    module._parse_adapter = lambda p: ('lora', 1., {'a' if p == 'good' else 'missing': params})
    specs = [{'path': p, 'strength': 1, 'steps': None} for p in ['good', 'bad']]
    with pytest.raises(AdapterApplicationError), require_complete_adapter_merge(module):
        module.prepare_loras({'a.weight': np.eye(2)}, specs, 8)
    assert calls == []
