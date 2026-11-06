"""Require complete adapter application in the pinned SA3 inference runtime.

Installed only around the local CLI invocation. Upstream accepts unmatched
targets with a warning; the studio must not publish that as a successful render.
No MLX import is needed until the caller supplies the runtime module.
"""
from contextlib import contextmanager
import math

import numpy as np


class AdapterApplicationError(ValueError):
    pass


def _expected_targets(module, weights, paths, strength):
    if not math.isfinite(float(strength)):
        raise AdapterApplicationError('Adapter strength must be finite.')
    expected = set()
    for path in paths:
        adapter_type, scaling, layers = module._parse_adapter(module._resolve_path(path))
        required = module._PARAMS_FOR.get(adapter_type)
        if not required or not layers or not math.isfinite(float(scaling)):
            raise AdapterApplicationError('Adapter has no valid target layers or scaling.')
        for layer, params in layers.items():
            key = module._layer_to_npz_key(layer)
            if key not in weights:
                raise AdapterApplicationError(f'Adapter target {layer!r} is absent from this model. No render was published.')
            missing = set(required) - params.keys()
            if missing:
                raise AdapterApplicationError(f'Adapter target {layer!r} is missing {sorted(missing)}.')
            if any(not np.isfinite(np.asarray(params[name])).all() for name in required):
                raise AdapterApplicationError(f'Adapter target {layer!r} contains non-finite values.')
            # Use the pinned runtime layout conversion and checks for Linear /
            # Conv1d and every supported adapter type; do not duplicate its math.
            weight, _ = module._weight_as_2d(weights[key])
            module._check_shapes(layer, weight, params, adapter_type)
            expected.add(key)
    if not expected:
        raise AdapterApplicationError('Adapter has no target layers; refusing an unadapted render.')
    return expected


@contextmanager
def require_complete_adapter_merge(module):
    """Guard the full-interval adapter path used by the studio's SA3 CLI.

The CLI imports this module during main(), so replacing its functions before
main() catches the loader. No output is written before this merge completes.
Step-gated adapters are deliberately rejected: the studio does not expose them
and their separate application path would bypass the merge verification.
"""
    original_merge = module.merge_loras_into_weights
    original_prepare = module.prepare_loras

    def checked_merge(weights, lora_paths, strength=1.0, log=lambda _: None):
        paths = list(lora_paths or [])
        if not paths:
            # An intentionally unadapted baseline is distinct from a supplied
            # checkpoint that contains no usable target layers.
            return original_merge(weights, paths, strength=strength, log=log)
        expected = _expected_targets(module, weights, paths, strength)
        stats = original_merge(weights, paths, strength=strength, log=log)
        if (not isinstance(stats, dict) or stats.get('skipped') != []
                or stats.get('merged') != len(expected)
                or stats.get('adapters') != len(paths)):
            raise AdapterApplicationError('Adapter did not apply completely; refusing to render.')
        return stats

    def checked_prepare(weights, specs, num_steps, log=lambda _: None, gate_all=False):
        if gate_all or any(spec.get('steps') is not None for spec in specs):
            raise AdapterApplicationError('Step-gated adapters are not supported by this studio.')
        # Validate every adapter before the first can mutate the model.
        for spec in specs:
            _expected_targets(module, weights, [spec['path']], spec['strength'])
        return original_prepare(weights, specs, num_steps, log=log, gate_all=False)

    module.merge_loras_into_weights = checked_merge
    module.prepare_loras = checked_prepare
    try:
        yield
    finally:
        module.merge_loras_into_weights = original_merge
        module.prepare_loras = original_prepare
