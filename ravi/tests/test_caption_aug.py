"""Tests for ravi.train.caption_aug (no torch needed)."""

from __future__ import annotations

import itertools
import pickle
from collections import Counter

import numpy as np
import pytest

from ravi.train import caption_aug as ca
from ravi.train.caption_aug import CaptionAugConfig, CaptionAugHook, augment_caption, split_fields

S = ca.FIELD_SEP
CAPTION = S.join(["rvkr", "Upbeat Acoustic Pop", "Ukulele And Claps", "Bright"])
N = 10_000


def on(**kw):
    return CaptionAugConfig(enabled=True, **kw)


def sep(*fs):
    return S.join(fs)


def draws(caption, cfg, seed=0, n=N):
    rng = np.random.default_rng(seed)
    return [augment_caption(caption, rng, cfg) for _ in range(n)]


def fields(s):
    # outputs are joined with join_sep=", "; only valid for captions without inner commas
    return [f.strip() for f in s.split(",")] if s else []


def test_deterministic_with_seed():
    cfg = on()
    assert draws(CAPTION, cfg, seed=7, n=200) == draws(CAPTION, cfg, seed=7, n=200)
    assert draws(CAPTION, cfg, seed=7, n=200) != draws(CAPTION, cfg, seed=8, n=200)


def test_trigger_preserved_and_fields_conserved():
    cfg = on(p_empty=0.1)
    orig = split_fields(CAPTION)
    for out in draws(CAPTION, cfg):
        if out == "":
            continue
        f = fields(out)
        assert f[0] == "rvkr"
        rest = [x.lower() for x in f[1:]]
        assert len(rest) in (len(orig) - 1, len(orig) - 2)
        assert set(rest) <= {x.lower() for x in orig[1:]}
        assert len(set(rest)) == len(rest)


def test_default_has_no_empty_caption_because_upstream_does_cfg_dropout():
    # cfg_dropout_prob=0.1 in stable_audio_3/training/diffusion.py:55 / dit.py:441-450
    assert on().p_empty == 0.0
    assert "" not in draws(CAPTION, on(), n=2000)


def _rate(xs):
    return sum(xs) / len(xs)


def test_probabilities_approximately_honoured():
    cfg = on(p_empty=0.1, p_drop_field=0.1, p_lowercase=0.5)
    outs = draws(CAPTION, cfg, seed=123)
    n_rest = len(split_fields(CAPTION)) - 1
    empty = [o == "" for o in outs]
    kept = [o for o in outs if o]
    dropped = [len(fields(o)) - 1 < n_rest for o in kept]
    lowered = [all(x == x.lower() for x in fields(o)[1:]) for o in kept]
    # 10k draws: binomial sd <= 0.005, so +-0.02 is ~4 sd
    assert abs(_rate(empty) - 0.1) < 0.015
    assert abs(_rate(dropped) - 0.1) < 0.015
    assert abs(_rate(lowered) - 0.5) < 0.02


def test_shuffle_uniform_over_permutations():
    cfg = on(p_drop_field=0.0, p_lowercase=0.0)
    outs = draws(sep("rvkr", "a", "b", "c"), cfg, seed=5)
    counts = Counter(tuple(fields(o)[1:]) for o in outs)
    assert set(counts) == set(itertools.permutations(["a", "b", "c"]))
    for c in counts.values():
        assert abs(c / N - 1 / 6) < 0.02


def test_drop_is_uniform_over_non_trigger_fields():
    cfg = on(p_drop_field=1.0, p_lowercase=0.0)
    outs = draws(sep("rvkr", "a", "b", "c"), cfg, seed=9)
    missing = Counter(({"a", "b", "c"} - set(fields(o)[1:])).pop() for o in outs)
    for c in missing.values():
        assert abs(c / N - 1 / 3) < 0.02


def test_trigger_never_lowercased_or_dropped():
    cfg = on(p_drop_field=1.0, p_lowercase=1.0, trigger="RVKR")
    for out in draws(sep("RVKR", "A", "B"), cfg, n=500):
        assert fields(out)[0] == "RVKR"


def test_trigger_only_caption():
    cfg = on(p_drop_field=1.0, p_lowercase=1.0)
    assert set(draws("rvkr", cfg, n=100)) == {"rvkr"}


TAIL = "sections: intro, verse, transition, verse, outro"


def test_atomic_tail_always_last_and_verbatim():
    # tail deliberately placed in the middle of the caption; it must end up last
    cap = sep("rvkr", "calm piano", TAIL, "strings", "Ambient")
    cfg = on(p_drop_field=0.5, p_lowercase=0.5)
    for out in draws(cap, cfg):
        assert out.endswith(", " + TAIL), out
        assert out.count("sections:") == 1


def test_atomic_tail_never_dropped_even_when_only_body_field():
    cfg = on(p_drop_field=1.0)
    assert set(draws(sep("rvkr", TAIL), cfg, n=200)) == {"rvkr, " + TAIL}


def test_comma_inside_sentence_survives_verbatim():
    sentence = "A warm, mellow piano ballad, with brushed drums"
    cap = sep("rvkr", sentence, "Jazz", "Slow")
    cfg = on(p_drop_field=0.0, p_lowercase=0.0)
    outs = draws(cap, cfg, n=2000)
    assert all(sentence in o for o in outs)
    assert all(o.startswith("rvkr, ") for o in outs)
    # still shuffled at field level
    assert len({o.index(sentence) for o in outs}) > 1


def test_non_droppable_fields_by_index_and_substring():
    sentence = "A warm, mellow piano ballad"
    cap = sep("rvkr", sentence, "Jazz", "Slow", "Soft")
    for rule in ([1], ["mellow piano"], [1, "jazz"]):
        cfg = on(p_drop_field=1.0, p_lowercase=0.0, non_droppable_fields=tuple(rule))
        outs = draws(cap, cfg, n=1000)
        assert all(sentence in o for o in outs), rule
        if "jazz" in rule:
            assert all("Jazz" in o for o in outs)
        # exactly one of the remaining droppable fields went missing each time
        assert all(sum(f in o for f in ("Jazz", "Slow", "Soft")) == 2 for o in outs)


def test_nothing_droppable_drops_nothing():
    cfg = on(p_drop_field=1.0, non_droppable_fields=(1, 2))
    outs = draws(sep("rvkr", "a", "b"), cfg, n=200)
    assert all(sorted(fields(o)[1:]) in (["a", "b"],) for o in outs)


def test_first_field_not_trigger_raises():
    with pytest.raises(ValueError, match="trigger"):
        augment_caption(sep("Jazz", "rvkr", "a"), np.random.default_rng(0), on())
    # old comma-only captions are one field, so they fail loudly too
    with pytest.raises(ValueError, match="trigger"):
        augment_caption("rvkr, a, b", np.random.default_rng(0), on())
    with pytest.raises(ValueError, match="trigger"):
        augment_caption(sep("xyz", "a"), np.random.default_rng(0), on(trigger="abc"))
    assert augment_caption(sep("abc", "a"), np.random.default_rng(0), on(trigger="abc", shuffle=False)).startswith("abc")


def test_keep_first_field_false_shuffles_everything():
    cfg = on(keep_first_field=False, p_drop_field=0.0, p_lowercase=0.0)
    outs = draws(sep("x", "a", "b"), cfg, seed=3)
    counts = Counter(tuple(fields(o)) for o in outs)
    assert set(counts) == set(itertools.permutations(["x", "a", "b"]))
    # no trigger check: first field may be anything
    cfg = on(keep_first_field=False, p_drop_field=1.0)
    assert all(len(fields(o)) == 2 for o in draws(sep("Jazz", "a", "b"), cfg, n=200))


def test_disabled_renders_join_sep_only():
    cfg = CaptionAugConfig()  # default: disabled
    assert cfg.enabled is False
    cap = sep("rvkr", "A warm, mellow ballad", " Jazz ")
    assert augment_caption(cap, np.random.default_rng(0), cfg) == "rvkr, A warm, mellow ballad, Jazz"
    # a caption without field_sep is passed through (stripped) exactly as upstream
    assert augment_caption("  rvkr,a ,  b ", np.random.default_rng(0), cfg) == "rvkr,a ,  b"
    # custom separators
    cfg = CaptionAugConfig(field_sep="|", join_sep="; ")
    assert augment_caption("rvkr|a, b|c", np.random.default_rng(0), cfg) == "rvkr; a, b; c"


def test_split_fields_only_on_field_sep():
    assert split_fields(sep("rvkr", "a, b", "", "  c ")) == ["rvkr", "a, b", "c"]
    assert split_fields("rvkr, a, b") == ["rvkr, a, b"]


def test_whitespace_normalised():
    cfg = on(shuffle=False, p_lowercase=0.0, p_drop_field=0.0)
    assert augment_caption(sep("rvkr", "a ", "  b", "", ""), np.random.default_rng(0), cfg) == "rvkr, a, b"


def test_config_validation():
    with pytest.raises(ValueError):
        CaptionAugConfig(p_lowercase=1.5)
    with pytest.raises(ValueError):
        CaptionAugConfig.from_dict({"p_shuffle": 0.5})
    with pytest.raises(ValueError):
        CaptionAugConfig(field_sep="")
    with pytest.raises(ValueError):
        CaptionAugConfig(non_droppable_fields=(1.5,))
    cfg = CaptionAugConfig.from_dict({"atomic_tail_prefixes": "sections:", "non_droppable_fields": 1})
    assert cfg.atomic_tail_prefixes == ("sections:",)
    assert cfg.non_droppable_fields == (1,)
    full = on(non_droppable_fields=(1, "jazz"))
    assert CaptionAugConfig.from_dict(full.to_dict()) == full
    assert CaptionAugConfig.from_dict(__import__("json").loads(full.to_json())) == full
    assert ca.FIELD_SEP == "\u241e"


# --------------------------------------------------------------------------- hook


def test_hook_reads_sibling_txt_like_upstream(tmp_path, monkeypatch):
    wav = tmp_path / "0915_copilot.mp3.wav"
    wav.write_bytes(b"")
    (tmp_path / "0915_copilot.mp3.txt").write_text("  " + sep("rvkr", "a, x", "b") + " \n")
    hook = CaptionAugHook(on())
    # main process (demo batch): unaugmented, stripped like train_lora.py:81, field_sep rendered
    monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: False)
    assert hook({"path": str(wav)}, None) == {"prompt": "rvkr, a, x, b"}
    # missing caption -> upstream reject marker
    assert hook({"path": str(tmp_path / "nope.wav")}, None) == {"__reject__": True}
    # worker: augmented, trigger kept
    monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: True)
    np.random.seed(0)
    outs = {hook({"path": str(wav)}, None)["prompt"] for _ in range(200)}
    assert all(o.startswith("rvkr") for o in outs)
    assert len(outs) > 1


def test_hook_worker_stream_follows_numpy_global_seed(tmp_path, monkeypatch):
    wav = tmp_path / "x.wav"
    (tmp_path / "x.txt").write_text(CAPTION)
    monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: True)
    hook = CaptionAugHook({"seed": 3, "enabled": True})

    def stream(global_seed):
        np.random.seed(global_seed)
        return [hook({"path": str(wav)})["prompt"] for _ in range(50)]

    assert stream(1) == stream(1)
    assert stream(1) != stream(2)


def test_hook_pickles():
    hook = CaptionAugHook({"p_lowercase": 0.25})
    clone = pickle.loads(pickle.dumps(hook))
    assert clone.cfg == hook.cfg


def test_verify_upstream_caption_fn(tmp_path):
    good = tmp_path / "train_lora.py"
    good.write_text("from pathlib import Path\n\n\n" + ca.UPSTREAM_CAPTION_FN_SOURCE)
    ca.verify_upstream_caption_fn(good)
    bad = tmp_path / "bad.py"
    bad.write_text(ca.UPSTREAM_CAPTION_FN_SOURCE.replace(".strip()", ""))
    with pytest.raises(RuntimeError):
        ca.verify_upstream_caption_fn(bad)


def test_patch_train_lora_module():
    class M:
        def caption_metadata_fn(info, audio):  # noqa: N805
            return {}

    hook = ca.patch_train_lora_module(M, CaptionAugConfig())
    assert M.caption_metadata_fn is hook


def test_patch_pre_encoded_dataset_augments_prompt_once(monkeypatch):
    monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: True)

    class DS:
        def __getitem__(self, idx):
            return "latents", {"prompt": sep("rvkr", "a", "b", "c")}

    ca.patch_pre_encoded_dataset(DS, on(p_drop_field=0.0, p_lowercase=0.0))
    ca.patch_pre_encoded_dataset(DS, on())  # idempotent
    np.random.seed(0)
    outs = {DS()[0][1]["prompt"] for _ in range(100)}
    assert all(o.startswith("rvkr, ") and len(fields(o)) == 4 for o in outs)
    assert len(outs) > 1


def test_hook_disabled_still_renders_in_workers_and_main(tmp_path, monkeypatch):
    wav = tmp_path / "x.wav"
    (tmp_path / "x.txt").write_text(sep("rvkr", "A warm, mellow ballad", "Jazz"))
    hook = CaptionAugHook(CaptionAugConfig())
    for in_worker in (False, True):
        monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: in_worker)
        assert hook({"path": str(wav)})["prompt"] == "rvkr, A warm, mellow ballad, Jazz"


def test_hook_main_process_checks_trigger(tmp_path, monkeypatch):
    wav = tmp_path / "x.wav"
    (tmp_path / "x.txt").write_text(sep("Jazz", "rvkr"))
    monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: False)
    with pytest.raises(ValueError, match="trigger"):
        CaptionAugHook(on())({"path": str(wav)})


def test_golden_upstream_caption_fn_matches():
    """The pinned upstream caption reader is the one CaptionAugHook re-implements."""
    import os
    from pathlib import Path

    root = os.environ.get("SA3_ROOT")
    if not root or not (Path(root) / "scripts" / "train_lora.py").is_file():
        pytest.skip("SA3_ROOT not set (point it at a stable-audio-3 checkout)")
    ca.verify_upstream_caption_fn(Path(root) / "scripts" / "train_lora.py")
