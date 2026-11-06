"""Child process: load upstream ``scripts/train_lora.py``, apply ravi patches, run its main().

Invoked by :mod:`ravi.train.sa3_lora`; not meant to be called by hand::

    python -m ravi.train._launch_train_lora --sa3-root /opt/sa3 \
        --caption-aug-json '{"enabled": true, ...}' \
        [--inpaint-mask-probs '[0.1, 0.8, 0.1]'] -- <train_lora.py argv>

Why a module import and not ``python scripts/train_lora.py``: ``scripts/`` is not part
of the installed ``stable_audio_3`` package, but the file has no side effects at import
time apart from setting ``HF_HUB_DISABLE_PROGRESS_BARS`` (``train_lora.py:31-33``) and
its ``main()`` is guarded by ``if __name__ == "__main__"`` (``train_lora.py:390``). So it
can be loaded by path, ``caption_metadata_fn`` swapped, and ``main()`` called with the
exact argv, which is still parsed by upstream's own argparse. Running it as an opaque
script would leave no way to patch the caption reader.

Why a separate process at all: the parent watches the checkpoint directory, uploads to
S3 and writes ``run_meta.json`` even if training dies (CUDA OOM, segfault, SIGTERM from
SageMaker); that is simpler when the trainer is a child it can wait on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from ravi.train.caption_aug import (
    CaptionAugConfig,
    patch_pre_encoded_dataset,
    patch_train_lora_module,
    verify_upstream_caption_fn,
)

MODULE_NAME = "sa3_train_lora"


def load_train_lora(sa3_root: Path):
    path = sa3_root / "scripts" / "train_lora.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; is SA3 checked out at {sa3_root}?")
    verify_upstream_caption_fn(path)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    # Registered so dill/pickle can resolve references into it from DataLoader workers.
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def validate_mask_probs(probs) -> list[float]:
    try:
        vals = [float(p) for p in probs]
    except (TypeError, ValueError):
        raise ValueError(f"inpaint mask probabilities must be a list of numbers: {probs!r}") from None
    if len(vals) not in (3, 4) or any(v < 0 for v in vals) or abs(sum(vals) - 1.0) > 1e-6:
        raise ValueError(f"inpaint mask probabilities must be 3 or 4 non-negative values summing to 1: {probs}")
    return vals


def patch_inpaint_mask_probs(module, probs: list[float]) -> None:
    """Override ``inpainting_config`` (train_lora.py:179) without editing upstream.

    Upstream hard-codes ``{"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}}``
    = [RANDOM_SEGMENTS, FULL_MASK, CAUSAL_MASK] (stable_audio_3/models/inpainting.py:164-172).
    A 4-element list adds RANDOM_SPANS. Only used for the mask-aware ablation.
    """
    probs = validate_mask_probs(probs)
    base = module.DiffusionCondTrainingWrapper

    class PatchedWrapper(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["inpainting_config"] = {"mask_kwargs": {"mask_type_probabilities": list(probs)}}
            super().__init__(*args, **kwargs)

    PatchedWrapper.__name__ = base.__name__
    module.DiffusionCondTrainingWrapper = PatchedWrapper


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: _launch_train_lora [options] -- <train_lora argv>")
    split = argv.index("--")
    ours, theirs = argv[:split], argv[split + 1 :]

    p = argparse.ArgumentParser()
    p.add_argument("--sa3-root", required=True, type=Path)
    p.add_argument("--caption-aug-json", default=None)
    p.add_argument("--inpaint-mask-probs", default=None)
    args = p.parse_args(ours)

    module = load_train_lora(args.sa3_root)

    if args.caption_aug_json:
        # Installed even when disabled: the hook then only renders field_sep as join_sep,
        # so the model never sees the separator character.
        cfg = CaptionAugConfig.from_dict(json.loads(args.caption_aug_json))
        patch_train_lora_module(module, cfg)
        patch_pre_encoded_dataset(module.PreEncodedDataset, cfg)
        state = "installed" if cfg.enabled else "disabled (field_sep -> join_sep only)"
        print(f"[ravi] caption augmentation {state}: {cfg.to_json()}", flush=True)
    else:
        print("[ravi] caption hook not installed (upstream caption_metadata_fn)", flush=True)

    if args.inpaint_mask_probs:
        probs = json.loads(args.inpaint_mask_probs)
        patch_inpaint_mask_probs(module, probs)
        print(f"[ravi] inpaint mask_type_probabilities overridden: {probs}", flush=True)

    sys.argv = [str(args.sa3_root / "scripts" / "train_lora.py"), *theirs]
    print(f"[ravi] train_lora argv: {json.dumps(theirs)}", flush=True)
    module.main()


if __name__ == "__main__":
    main()
