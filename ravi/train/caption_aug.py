"""Online caption augmentation for SA3 LoRA training.

Where it hooks in (verified against stable-audio-3 @ 77e0398c26f7d22d801ec1bb036307e8b231f741):

* ``scripts/train_lora.py:77-81`` defines the only caption reader on the
  ``--data_dir`` path::

      def caption_metadata_fn(info, audio):
          txt = Path(info["path"]).with_suffix(".txt")
          if not txt.exists():
              return {"__reject__": True}
          return {"prompt": txt.read_text().strip()}

  ``train()`` looks the name up as a module global when it builds the dataset
  (``train_lora.py:121-131``, ``custom_metadata_fn=caption_metadata_fn``), so
  replacing the module attribute before ``main()`` runs is enough.
  ``SampleDataset`` dill-pickles the callable (``stable_audio_3/data/dataset.py:227-228``)
  and calls it in every ``__getitem__`` (``dataset.py:291-295``), i.e. inside the
  DataLoader worker, once per draw. The caption is therefore re-augmented every
  time a clip is drawn, which is what "online" means here.
* On the ``--encoded_dir`` path there is no caption hook: ``PreEncodedDataset`` is
  built without a ``custom_metadata_fn`` (``train_lora.py:114-119``) and the caption
  is the ``"prompt"`` key of the JSON sidecar written by ``pre_encode_dataset.py``.
  :func:`patch_pre_encoded_dataset` wraps ``PreEncodedDataset.__getitem__`` to
  augment that key instead.

Classifier-free-guidance dropout already exists upstream, so it is **not** doubled:
``DiffusionCondTrainingWrapper`` defaults ``cfg_dropout_prob=0.1``
(``stable_audio_3/training/diffusion.py:55``), ``train_lora.py`` does not override it
(``train_lora.py:168-189``), and it is applied in the DiT forward pass by replacing the
cross-attention (text) embedding with zeros per batch element
(``stable_audio_3/models/dit.py:441-450``). That zero embedding is what inference
uses as the unconditional branch (``dit.py:512-538``). Feeding an *empty string*
instead would give a T5Gemma embedding of "", which is not the null embedding, so
the plan's ``p=0.1 empty caption`` would be a second, different kind of dropout.
``p_empty`` therefore defaults to 0.0 and exists only for ablations.

Randomness. The augmentation itself is a pure function of ``(caption, rng)``. The
hook builds a fresh ``numpy.random.Generator`` per call from ``[seed, draw]`` where
``draw`` comes from numpy's global RNG. PyTorch's DataLoader worker loop reseeds
numpy's global RNG per worker and per epoch from a base seed drawn from the main
process's torch RNG, and that RNG is fixed by ``pl.seed_everything(seed)``
(``train_lora.py:95``). torch's RNG is deliberately *not* used in workers:
``train_lora.py:140`` resets it to ``seed + worker_id`` on every worker start, so
with non-persistent workers it would replay the same stream every epoch.

The demo batch is fetched in the main process (``train_lora.py:228-245``,
``num_workers=0``). The hook does not augment outside DataLoader workers, so the
demos are rendered with the clean caption (fields joined with ``join_sep``). The wrapper
requires ``num_workers >= 1`` when augmentation is on, so training is never silently
unaugmented.

Caption format: fields separated by ``field_sep`` (default ``"␞"``, U+241E), first field
the trigger. See :class:`CaptionAugConfig`. Splitting on commas would cut free-text
sentences and ordered section lists into pieces, so commas are never a separator.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Exact text of the upstream hook. The launcher refuses to patch a train_lora.py whose
# caption reader differs from this, because CaptionAugHook re-implements it.
UPSTREAM_CAPTION_FN_SOURCE = '''def caption_metadata_fn(info, audio):
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}
'''


FIELD_SEP = "␞"  # "␞" SYMBOL FOR RECORD SEPARATOR; never occurs in natural captions
JOIN_SEP = ", "


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value)


@dataclasses.dataclass(frozen=True)
class CaptionAugConfig:
    """Augmentation settings. Off unless an arm opts in with ``enabled: true``.

    Caption format. A caption file holds *fields* separated by :data:`FIELD_SEP`
    (``"␞"``, U+241E), e.g. ``rvkr␞A warm, mellow piano ballad␞Jazz␞sections: intro, verse``.
    Fields are split, shuffled and dropped only at ``field_sep``; commas inside a field
    (free-text sentences, section lists) are never touched. The text the model sees is
    the fields joined with ``join_sep`` (``", "``), on every path: augmented draws in
    DataLoader workers, the unaugmented demo batch in the main process, and
    ``enabled=False``. A caption without ``field_sep`` is a single field.
    """

    enabled: bool = False
    field_sep: str = FIELD_SEP
    join_sep: str = JOIN_SEP
    keep_first_field: bool = True  # field 0 must equal `trigger`; never moved, dropped or re-cased
    trigger: str = "rvkr"
    shuffle: bool = True  # uniformly permute the body fields
    p_lowercase: float = 0.5  # lowercase all body fields
    p_drop_field: float = 0.1  # drop one droppable body field
    p_empty: float = 0.0  # replace the whole caption with "" (NOT CFG; see module docstring)
    # Fields that are never dropped (still shuffled and re-cased). An int is an index into
    # the caption's field list (0 = trigger); a str protects every field containing it
    # (case-insensitive). E.g. [1] keeps arm B's human sentence in every draw.
    non_droppable_fields: tuple[int | str, ...] = ()
    # A field starting with one of these prefixes (case-insensitive) is atomic: never
    # shuffled, dropped or re-cased, and always placed last (after the body), in its
    # original relative order. Arm E's "sections: intro, verse, outro" must keep its order.
    atomic_tail_prefixes: tuple[str, ...] = ("sections:",)
    seed: int = 0
    augment_in_main_process: bool = False  # demos are drawn in the main process

    def __post_init__(self) -> None:
        for name in ("p_lowercase", "p_drop_field", "p_empty"):
            p = getattr(self, name)
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"{name}={p} is not a probability")
        if not isinstance(self.field_sep, str) or not self.field_sep:
            raise ValueError("field_sep must be a non-empty string")
        if not isinstance(self.join_sep, str):
            raise ValueError("join_sep must be a string")
        if self.keep_first_field and not str(self.trigger).strip():
            raise ValueError("keep_first_field needs a non-empty trigger")
        for f in self.non_droppable_fields:
            if isinstance(f, bool) or not isinstance(f, (int, str)):
                raise ValueError(f"non_droppable_fields entries must be int or str, got {f!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CaptionAugConfig":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown caption_aug keys: {sorted(unknown)}")
        for key in ("atomic_tail_prefixes", "non_droppable_fields"):
            if key in d:
                d[key] = _as_tuple(d[key])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["atomic_tail_prefixes"] = list(self.atomic_tail_prefixes)
        d["non_droppable_fields"] = list(self.non_droppable_fields)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def split_fields(caption: str, field_sep: str = FIELD_SEP) -> list[str]:
    """Split on ``field_sep`` only, strip whitespace and drop empty fields."""
    return [f for f in (x.strip() for x in caption.split(field_sep)) if f]


def join_fields(fields: Iterable[str], join_sep: str = JOIN_SEP) -> str:
    return join_sep.join(fields)


def render_caption(caption: str, cfg: CaptionAugConfig) -> str:
    """The unaugmented text the model sees: fields joined with ``join_sep``."""
    fields = split_fields(caption, cfg.field_sep)
    return join_fields(fields, cfg.join_sep) if fields else caption.strip()


def check_trigger(caption: str, cfg: CaptionAugConfig) -> None:
    """Raise if ``keep_first_field`` is on and field 0 is not the trigger."""
    if not cfg.keep_first_field:
        return
    fields = split_fields(caption, cfg.field_sep)
    first = fields[0] if fields else ""
    if first != cfg.trigger:
        raise ValueError(
            f"caption's first field is {first[:60]!r}, expected trigger {cfg.trigger!r} "
            f"(fields are separated by {cfg.field_sep!r}); caption: {caption[:120]!r}"
        )


def _is_tail(field: str, prefixes: Sequence[str]) -> bool:
    low = field.lower()
    return any(low.startswith(p.lower()) for p in prefixes)


def _protected(index: int, field: str, rules: Sequence[int | str]) -> bool:
    low = field.lower()
    for r in rules:
        if isinstance(r, int) and r == index:
            return True
        if isinstance(r, str) and r.lower() in low:
            return True
    return False


def augment_caption(caption: str, rng: np.random.Generator, cfg: CaptionAugConfig) -> str:
    """Return one augmented draw of ``caption``.

    Fields: ``head`` (field 0, the trigger, if keep_first_field), ``tail`` (atomic-tail
    fields, verbatim, in order) and ``body`` (everything else). Each draw:
      1. with p_empty   -> return ""
      2. with p_drop    -> remove one uniformly chosen droppable body field (if any)
      3. if shuffle     -> uniformly permute the body
      4. with p_lower   -> lowercase the body
    and returns ``join_sep.join(head + body + tail)``.
    """
    if not cfg.enabled:
        return render_caption(caption, cfg)
    check_trigger(caption, cfg)
    if cfg.p_empty > 0.0 and rng.random() < cfg.p_empty:
        return ""

    fields = split_fields(caption, cfg.field_sep)
    if not fields:
        return caption.strip()

    indexed = list(enumerate(fields))
    head = indexed[:1] if cfg.keep_first_field else []
    others = indexed[len(head):]
    tail = [x for x in others if _is_tail(x[1], cfg.atomic_tail_prefixes)]
    body = [x for x in others if not _is_tail(x[1], cfg.atomic_tail_prefixes)]

    if cfg.p_drop_field > 0.0 and rng.random() < cfg.p_drop_field:
        droppable = [k for k, (i, f) in enumerate(body) if not _protected(i, f, cfg.non_droppable_fields)]
        if droppable:
            del body[droppable[int(rng.integers(len(droppable)))]]

    if cfg.shuffle and len(body) > 1:
        body = [body[i] for i in rng.permutation(len(body))]

    body_text = [f for _, f in body]
    if cfg.p_lowercase > 0.0 and rng.random() < cfg.p_lowercase:
        body_text = [f.lower() for f in body_text]

    return join_fields([f for _, f in head] + body_text + [f for _, f in tail], cfg.join_sep)


def _in_dataloader_worker() -> bool:
    try:
        import torch.utils.data
    except ImportError:  # unit tests run without torch
        return False
    return torch.utils.data.get_worker_info() is not None


def _draw_rng(seed: int) -> np.random.Generator:
    # numpy's global RNG is reseeded per worker per epoch by torch's DataLoader.
    draw = int(np.random.randint(0, 2**32, dtype=np.uint64))
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, draw])


class CaptionAugHook:
    """Drop-in replacement for ``train_lora.caption_metadata_fn``.

    Reads the sibling .txt exactly as upstream does, then augments the caption. With
    ``enabled=False`` it only renders ``field_sep`` as ``join_sep`` (identity for a caption
    without ``field_sep``), so the launcher installs it for every run.
    Defined at module level so dill pickles it by reference into the workers.
    """

    def __init__(self, cfg: CaptionAugConfig | dict | None = None):
        self.cfg = cfg if isinstance(cfg, CaptionAugConfig) else CaptionAugConfig.from_dict(cfg)

    # Same behaviour as train_lora.py:77-81.
    @staticmethod
    def read_caption(info: dict) -> dict:
        txt = Path(info["path"]).with_suffix(".txt")
        if not txt.exists():
            return {"__reject__": True}
        return {"prompt": txt.read_text().strip()}

    def augment(self, caption: str) -> str:
        if not self.cfg.enabled:
            return render_caption(caption, self.cfg)
        check_trigger(caption, self.cfg)
        if not self.cfg.augment_in_main_process and not _in_dataloader_worker():
            # demo batch (train_lora.py:228-245): clean caption, but still joined with
            # join_sep so the model never sees field_sep
            return render_caption(caption, self.cfg)
        return augment_caption(caption, _draw_rng(self.cfg.seed), self.cfg)

    def __call__(self, info: dict, audio: Any = None) -> dict:
        out = self.read_caption(info)
        if "prompt" in out:
            out["prompt"] = self.augment(out["prompt"])
        return out

    # dill pickles instances by reference to the class plus __dict__; keep it plain.
    def __getstate__(self) -> dict:
        return {"cfg": self.cfg.to_dict()}

    def __setstate__(self, state: dict) -> None:
        self.cfg = CaptionAugConfig.from_dict(state["cfg"])


def verify_upstream_caption_fn(train_lora_path: str | Path) -> None:
    """Fail loudly if the pinned upstream caption reader is not the one we re-implement.

    Reads the file as text; does not import or execute it."""
    src = Path(train_lora_path).read_text()
    if UPSTREAM_CAPTION_FN_SOURCE not in src:
        raise RuntimeError(
            f"{train_lora_path}: caption_metadata_fn differs from the version "
            "ravi.train.caption_aug re-implements (stable-audio-3 @ 77e0398). "
            "Re-verify the hook before training."
        )


def patch_train_lora_module(module: Any, cfg: CaptionAugConfig) -> CaptionAugHook:
    """Replace ``caption_metadata_fn`` in a loaded train_lora module."""
    if not hasattr(module, "caption_metadata_fn"):
        raise RuntimeError("train_lora module has no caption_metadata_fn to patch")
    hook = CaptionAugHook(cfg)
    module.caption_metadata_fn = hook
    return hook


def patch_pre_encoded_dataset(dataset_cls: Any, cfg: CaptionAugConfig) -> None:
    """Wrap ``PreEncodedDataset.__getitem__`` so the JSON ``prompt`` is augmented."""
    if getattr(dataset_cls.__getitem__, "_ravi_caption_aug", False):
        return
    hook = CaptionAugHook(cfg)
    original = dataset_cls.__getitem__

    def __getitem__(self, idx):
        latents, info = original(self, idx)
        # The upstream error path recurses through self[...] (dataset.py:433-476),
        # i.e. through this wrapper; the marker stops a second augmentation.
        if isinstance(info.get("prompt"), str) and not info.get("__ravi_caption_aug__"):
            info["prompt"] = hook.augment(info["prompt"])
            info["__ravi_caption_aug__"] = True
        return latents, info

    __getitem__._ravi_caption_aug = True  # type: ignore[attr-defined]
    dataset_cls.__getitem__ = __getitem__
