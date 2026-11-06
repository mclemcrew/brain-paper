"""Tests for ravi.captions.make_acestep_dataset and ravi.train.acestep_lora (no torch, ACE-Step or GPU).

The upstream argv contract test reads ``acestep/training_v2/cli/args.py`` as text from an
ACE-Step-1.5 checkout given by ``ACESTEP_SRC`` (e.g. the pinned clone at
``ca1e85fe9430179831e6bc6be790c332190a3866``) and is skipped without one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ravi.captions import make_acestep_dataset as mad
from ravi.train import acestep_lora as a

FS = "␞"


# --------------------------------------------------------------------------- helpers


def write_arm(root: Path, arm: str = "E", n: int = 3, seconds: float = 1.0, sr: int = 44100,
              captions: list[str] | None = None, amp: float = 0.3) -> Path:
    d = root / arm / "train"
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        tid = f"track {i:02d}"
        t = np.arange(int(seconds * sr)) / sr
        x = np.stack([amp * np.sin(2 * np.pi * (220 + 110 * i) * t), 0.5 * amp * np.sin(2 * np.pi * 330 * t)], 1)
        sf.write(d / f"{tid}.wav", x, sr, subtype="PCM_16")
        cap = captions[i] if captions else FS.join(
            ["rvkr", f"calm piano {i}, with strings", "Jazz", "sections: intro, verse, transition, verse, outro"])
        (d / f"{tid}.txt").write_text(cap, encoding="utf-8")
        names.append(f"{tid}.wav")
    (d / "filelist.txt").write_text("".join(x + "\n" for x in names))
    (d / f"{names[0][:-4]}.window.json").write_text("{}")  # make_arms sidecar, must be ignored
    return root / arm


def tree_digest(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def converted(tmp_path: Path, arm: str = "E", n: int = 3, **kw) -> Path:
    arm_dir = write_arm(tmp_path / "arms", arm, n, **kw)
    out = tmp_path / "ace" / arm
    mad.make_acestep_dataset(arm_dir, "train", out)
    return out


def make_cfg(tmp_path: Path, data_dir: Path, **over) -> dict:
    cfg = a.load_config(hp_path=str(tmp_path / "no_hp.json"))
    cfg.update({
        "arm": "E", "data_dir": str(data_dir), "checkpoint_dir": str(tmp_path / "ckpt"),
        "model_dir": str(tmp_path / "model"), "output_data_dir": str(tmp_path / "out"),
        "work_dir": str(tmp_path / "work"), "weights_dir": str(tmp_path / "weights"),
        "steps": 100, "seed": 42, "require_cuda": False, "require_flash_attn": False,
    })
    cfg.update(over)
    return cfg


def resolved(tmp_path: Path, data_dir: Path, **over) -> dict:
    return a.resolve(make_cfg(tmp_path, data_dir, **over))


# --------------------------------------------------------------------------- dataset conversion


def test_arm_e_sections_become_lyrics_and_caption_is_comma_joined(tmp_path):
    out = converted(tmp_path, "E")
    files = sorted(p.name for p in out.iterdir())
    assert "track 00.lyrics.txt" in files and "track 00.caption.txt" in files and "track 00.json" in files
    assert not [f for f in files if f.endswith(".txt") and not f.endswith((".caption.txt", ".lyrics.txt"))]
    assert not [f for f in files if f.endswith(".csv") or f.endswith(".window.json")]
    assert (out / "track 00.caption.txt").read_text() == "rvkr, calm piano 0, with strings, Jazz\n"
    assert (out / "track 00.lyrics.txt").read_text() == "[intro]\n[verse]\n[transition]\n[verse]\n[outro]\n"
    meta = json.loads((out / "track 00.json").read_text())
    assert meta == {"caption": "rvkr, calm piano 0, with strings, Jazz", "is_instrumental": True}


def test_other_arms_have_no_lyrics_file(tmp_path):
    caps = [FS.join(["rvkr", "warm synth pads", "Electronic"])] * 2
    out = converted(tmp_path, "B", n=2, captions=caps)
    assert not list(out.glob("*.lyrics.txt"))
    assert (out / "track 01.caption.txt").read_text() == "rvkr, warm synth pads, Electronic\n"
    doc = json.loads((out / mad.MANIFEST_NAME).read_text())
    assert doc["sections_as_lyrics"] is False and doc["n_with_lyrics"] == 0


def test_sections_field_kept_in_caption_outside_arm_e(tmp_path):
    caps = [FS.join(["rvkr", "Genre: Jazz", "Sections: intro, outro"])]
    out = converted(tmp_path, "F", n=1, captions=caps)
    assert (out / "track 00.caption.txt").read_text() == "rvkr, Genre: Jazz, Sections: intro, outro\n"
    assert not list(out.glob("*.lyrics.txt"))


def test_resampled_to_48k_with_expected_length(tmp_path):
    out = converted(tmp_path, "E", n=1, seconds=1.0)
    info = sf.info(str(out / "track 00.wav"))
    assert info.samplerate == 48000 and info.channels == 2 and info.subtype == "PCM_24"
    assert info.frames == math.ceil(44100 * 48000 / 44100)  # ceil(n * 160 / 147)
    y, _ = sf.read(str(out / "track 00.wav"))
    t = np.arange(len(y)) / 48000
    # a 220 Hz sine stays a 220 Hz sine (away from the edges)
    ref = 0.3 * np.sin(2 * np.pi * 220 * t)
    assert np.max(np.abs(y[2000:-2000, 0] - ref[2000:-2000])) < 2e-3


def test_bpm_from_manifest(tmp_path):
    import pandas as pd

    man = tmp_path / "tracks.parquet"
    pd.DataFrame({"track_id": ["track 00", "track 01", "track 02"],
                  "id3_bpm": [120.4, float("nan"), 0.0]}).to_parquet(man)
    arm_dir = write_arm(tmp_path / "arms", "E", 3)
    out = tmp_path / "ace" / "E"
    doc = mad.make_acestep_dataset(arm_dir, "train", out, manifest=man)
    assert json.loads((out / "track 00.json").read_text())["bpm"] == 120
    assert "bpm" not in json.loads((out / "track 01.json").read_text())
    assert "bpm" not in json.loads((out / "track 02.json").read_text())
    assert doc["n_with_bpm"] == 1


def test_conversion_is_deterministic_and_limit_takes_filelist_order(tmp_path):
    arm_dir = write_arm(tmp_path / "arms", "E", 3)
    d1, d2 = tmp_path / "o1", tmp_path / "o2"
    mad.make_acestep_dataset(arm_dir, "train", d1)
    mad.make_acestep_dataset(arm_dir, "train", d2)
    t1, t2 = tree_digest(d1), tree_digest(d2)
    t1.pop(mad.MANIFEST_NAME), t2.pop(mad.MANIFEST_NAME)  # holds the source path, otherwise equal
    assert t1 == t2
    m1 = json.loads((d1 / mad.MANIFEST_NAME).read_text())
    m2 = json.loads((d2 / mad.MANIFEST_NAME).read_text())
    assert m1 == m2
    d3 = tmp_path / "o3"
    mad.make_acestep_dataset(arm_dir, "train", d3, limit=2)
    assert sorted(p.name for p in d3.glob("*.wav")) == ["track 00.wav", "track 01.wav"]


def test_missing_trigger_is_an_error_and_nothing_is_left_behind(tmp_path):
    arm_dir = write_arm(tmp_path / "arms", "B", 2, captions=[FS.join(["rvkr", "ok"]), FS.join(["calm", "rvkr"])])
    out = tmp_path / "ace" / "B"
    with pytest.raises(ValueError, match="trigger"):
        mad.make_acestep_dataset(arm_dir, "train", out)
    assert not out.exists() and not out.with_name("B.tmp").exists()


def test_refuses_foreign_dir_and_needs_overwrite(tmp_path):
    arm_dir = write_arm(tmp_path / "arms", "E", 1)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="marker"):
        mad.make_acestep_dataset(arm_dir, "train", foreign, overwrite=True)
    out = tmp_path / "ace"
    mad.make_acestep_dataset(arm_dir, "train", out)
    with pytest.raises(FileExistsError, match="overwrite"):
        mad.make_acestep_dataset(arm_dir, "train", out)
    mad.make_acestep_dataset(arm_dir, "train", out, overwrite=True)
    assert (out / "track 00.wav").is_file()


def test_clipping_clip_written_as_float(tmp_path):
    arm_dir = tmp_path / "arms" / "B"
    d = arm_dir / "train"
    d.mkdir(parents=True)
    x = np.sign(np.sin(2 * np.pi * 1000 * np.arange(44100) / 44100)) * 0.999  # square wave: Gibbs overshoot
    sf.write(d / "sq.wav", np.stack([x, x], 1), 44100, subtype="PCM_16")
    (d / "sq.txt").write_text(FS.join(["rvkr", "square"]))
    out = tmp_path / "ace"
    doc = mad.make_acestep_dataset(arm_dir, "train", out)
    assert doc["clips"][0]["peak"] > 1.0 and doc["n_float"] == 1
    assert sf.info(str(out / "sq.wav")).subtype == "FLOAT"


def test_section_lyrics_rejects_brackets():
    assert mad.section_lyrics("sections: intro, verse") == "[intro]\n[verse]"
    with pytest.raises(ValueError):
        mad.section_lyrics("sections: [intro]")
    with pytest.raises(ValueError):
        mad.section_lyrics("genre: x")


# --------------------------------------------------------------------------- preflight


def test_scan_accepts_converted_output(tmp_path):
    data = converted(tmp_path, "E", n=3)
    summary, recs = a.scan_data(resolved(tmp_path, data))
    assert summary["n_clips"] == 3 and summary["n_with_lyrics"] == 3 and summary["n_not_48k"] == 0
    assert recs[0]["lyrics"] == "[intro]\n[verse]\n[transition]\n[verse]\n[outro]"
    assert recs[0]["caption"].startswith("rvkr, ")


@pytest.mark.parametrize("damage, match", [
    (lambda d: (d / "track 00.txt").write_text("rvkr␞x"), "LYRICS"),
    (lambda d: (d / "manifest.csv").write_text("file,caption\n"), "CSV"),
    (lambda d: ((d / "track 01.caption.txt").unlink(), (d / "track 01.json").write_text("{}")), "no caption"),
    (lambda d: (d / "track 02.json").write_text(json.dumps({"caption": "calm, rvkr"})), "trigger"),
    (lambda d: (d / "sub").mkdir() or sf.write(d / "sub" / "track 00.flac", np.zeros((480, 2)), 48000), "stem"),
])
def test_scan_rejects_layouts_upstream_would_misread(tmp_path, damage, match):
    data = converted(tmp_path, "E", n=3)
    damage(data)
    with pytest.raises(ValueError, match=match):
        a.scan_data(resolved(tmp_path, data))


def test_scan_rejects_clips_longer_than_max_duration(tmp_path):
    data = converted(tmp_path, "E", n=1, seconds=2.0)
    with pytest.raises(ValueError, match="truncates"):
        a.scan_data(resolved(tmp_path, data, max_duration=1.5))


def test_fingerprint_tracks_captions(tmp_path):
    data = converted(tmp_path, "E", n=2)
    fp1 = a.scan_data(resolved(tmp_path, data))[0]["fingerprint"]
    (data / "track 00.caption.txt").write_text("rvkr, changed\n")
    (data / "track 00.json").write_text(json.dumps({"caption": "rvkr, changed", "is_instrumental": True}))
    assert a.scan_data(resolved(tmp_path, data))[0]["fingerprint"] != fp1


def test_check_dataset_json_matches_upstream_shape(tmp_path):
    data = converted(tmp_path, "E", n=2)
    _, recs = a.scan_data(resolved(tmp_path, data))
    samples = [{"audio_path": r["audio_path"], "filename": Path(r["audio_path"]).name, "caption": r["caption"],
                "lyrics": r["lyrics"], "labeled": True, "custom_tag": "", "bpm": None} for r in recs]
    doc = {"metadata": {"genre_ratio": 0, "custom_tag": ""}, "samples": samples}
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(doc))
    assert a.check_dataset_json(p, recs)["n_samples"] == 2
    samples[1]["lyrics"] = "[Instrumental]"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="lyrics"):
        a.check_dataset_json(p, recs)


# --------------------------------------------------------------------------- epochs


@pytest.mark.parametrize("n", range(1, 40))
@pytest.mark.parametrize("b, k", [(1, 1), (1, 4), (2, 4), (3, 2), (4, 3)])
def test_steps_per_epoch_identity(n, b, k):
    assert a.steps_per_epoch(n, b, k) == math.ceil(n / (b * k))


def test_epoch_plan_examples():
    full = a.plan_epochs(6000, 580, 1, 4)
    assert (full["steps_per_epoch"], full["epochs"], full["actual_steps"], full["overshoot_steps"]) == (145, 42, 6090, 90)
    assert full["save_every_epochs"] == 4 and full["checkpoint_epochs"][:3] == [4, 8, 12]
    smoke = a.plan_epochs(100, 20, 1, 4, checkpoint_every_steps=50)
    assert (smoke["steps_per_epoch"], smoke["epochs"], smoke["actual_steps"], smoke["save_every_epochs"]) == (5, 20, 100, 10)
    assert a.plan_epochs(100, 20, 1, 4, save_every_epochs=3)["checkpoint_epochs"] == [3, 6, 9, 12, 15, 18]


@pytest.mark.parametrize("steps, n", [(1, 1), (7, 3), (999, 580), (6000, 581), (12345, 616)])
def test_epoch_plan_rounds_up_by_less_than_one_epoch(steps, n):
    p = a.plan_epochs(steps, n, 1, 4)
    assert steps <= p["actual_steps"] < steps + p["steps_per_epoch"]


def test_train_argv_carries_the_plan(tmp_path):
    data = converted(tmp_path, "E", n=20, seconds=0.2)
    cfg = resolved(tmp_path, data, checkpoint_every_steps=50)
    summary, _ = a.scan_data(cfg)
    plan = a.plan_epochs(cfg["steps"], summary["n_clips"], cfg["batch_size"], cfg["grad_accum"],
                         cfg["save_every_epochs"], cfg["checkpoint_every_steps"])
    argv = a.build_train_argv(cfg, plan, tmp_path / "t", tmp_path / "o", 42, None)
    val = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert (val("--epochs"), val("--save-every"), val("--model-variant")) == ("20", "10", "turbo")
    assert (val("--rank"), val("--alpha"), val("--dropout"), val("--cfg-ratio")) == ("64", "128", "0.1", "0.15")
    assert (val("--lr"), val("--batch-size"), val("--gradient-accumulation")) == ("0.0001", "1", "4")
    assert (val("--shift"), val("--num-inference-steps")) == ("3.0", "8")
    i = argv.index("--target-modules")
    assert argv[i + 1:i + 5] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert "--gradient-checkpointing" in argv and "--resume-from" not in argv and argv[:2] == ["--plain", "--yes"]


# --------------------------------------------------------------------------- upstream contract


def _upstream_args() -> Path:
    root = os.environ.get("ACESTEP_SRC")
    if not root:
        pytest.skip("ACESTEP_SRC not set (point it at an ACE-Step-1.5 checkout)")
    path = Path(root) / "acestep" / "training_v2" / "cli" / "args.py"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return path


def upstream_flags(src: str) -> set[str]:
    """Every long option in an ``add_argument`` call (aliases included), plus ``--no-X`` for
    ``BooleanOptionalAction`` flags."""
    flags: set[str] = set()
    for m in re.finditer(r"add_argument\(([^)]*)\)", src, re.S):
        body = m.group(1)
        names = re.findall(r"""["'](--[A-Za-z0-9_-]+)["']""", body.split("help=")[0])
        flags |= set(names)
        if "BooleanOptionalAction" in body:
            flags |= {"--no-" + n[2:] for n in names}
    return flags


def test_every_emitted_flag_exists_in_upstream_args(tmp_path):
    src = _upstream_args().read_text()  # read as text only; never imported
    upstream = upstream_flags(src)
    assert {"--cfg-ratio", "--epochs", "--save-every", "--gradient-checkpointing", "--no-gradient-checkpointing",
            "--preprocess", "--dataset-json", "--tensor-output", "--plain", "--yes"} <= upstream
    data = converted(tmp_path, "E", n=2)
    emitted: set[str] = set()
    for over in ({}, {"gradient_checkpointing": False, "model_variant": "base"}):
        cfg = resolved(tmp_path, data, **over)
        plan = a.plan_epochs(cfg["steps"], 2, 1, 4)
        for argv in (a.build_preprocess_argv(cfg, tmp_path / "ds.json", tmp_path / "t"),
                     a.build_train_argv(cfg, plan, tmp_path / "t", tmp_path / "o", 43, str(tmp_path / "r"))):
            emitted |= {x for x in argv if x.startswith("--")}
    missing = emitted - upstream
    assert not missing, f"flags not accepted by upstream cli/args.py: {sorted(missing)}"
    assert {"--resume-from", "--no-gradient-checkpointing", "--cfg-ratio"} <= emitted


def test_upstream_choices_match_validation():
    src = _upstream_args().read_text()
    for flag, key in (("--optimizer-type", "optimizer"), ("--scheduler-type", "scheduler"),
                      ("--attention-type", "attention_type"), ("--precision", "precision")):
        m = re.search(rf'"{flag}".*?choices=\[([^\]]*)\]', src, re.S)
        assert m, flag
        up = set(re.findall(r'"([^"]+)"', m.group(1)))
        cfg_ok = {"optimizer": "adamw", "scheduler": "cosine", "attention_type": "both", "precision": "bf16"}[key]
        assert cfg_ok in up
    m = re.search(r"VARIANT_DIR_MAP = \{(.*?)\}", src, re.S)
    for v, spec in a.VARIANTS.items():
        assert f'"{v}": "{spec["dir"]}"' in m.group(1)


# --------------------------------------------------------------------------- config


def test_config_precedence_and_sagemaker_json_strings(tmp_path):
    data = converted(tmp_path, "E", n=1)
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({"rank": 32, "arm": "B", "steps": 200}))
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"config": f'"{cfg_file}"', "rank": "16", "arm": '"123"',
                              "target_modules": '["q_proj", "v_proj"]', "sagemaker_program": "x"}))
    cfg = a.load_config(hp_path=str(hp), sets=[f"data_dir={data}", "steps=300"])
    assert (cfg["rank"], cfg["arm"], cfg["steps"], cfg["target_modules"]) == (16, "123", 300, ["q_proj", "v_proj"])
    cfg = a.load_config(hp_path=str(tmp_path / "none.json"), sets=["target_modules=q_proj k_proj"])
    cfg.update(data_dir=str(data), work_dir=str(tmp_path / "w"))
    assert a.resolve(cfg)["target_modules"] == ["q_proj", "k_proj"]
    with pytest.raises(ValueError, match="unknown"):
        a.load_config(hp_path=str(tmp_path / "none.json"), sets=["ranks=3"])


@pytest.mark.parametrize("over, match", [
    ({"alpha": 128.5}, "integer"), ({"model_variant": "xl_turbo"}, "model_variant"),
    ({"scheduler": "step"}, "scheduler"), ({"cfg_ratio": 1.0}, "cfg_ratio"), ({"work_dir": "/"}, "work_dir"),
])
def test_resolve_rejects(tmp_path, over, match):
    with pytest.raises(ValueError, match=match):
        resolved(tmp_path, tmp_path, **over)


def test_variant_defaults(tmp_path):
    t = resolved(tmp_path, tmp_path)
    b = resolved(tmp_path, tmp_path, model_variant="base")
    assert (t["shift"], t["num_inference_steps"], t["weights_revision"]) == (3.0, 8, a.MAIN_WEIGHTS_REVISION)
    assert (b["shift"], b["num_inference_steps"]) == (1.0, 50) and b["weights_revision"] != t["weights_revision"]
    layout = a.weights_layout(b)
    assert layout[0][0] == a.MAIN_WEIGHTS_REPO and layout[1][3].name == "acestep-v15-base"


def test_resume_default_is_none_off_sagemaker_and_auto_on_it(tmp_path, monkeypatch):
    monkeypatch.delenv("SM_TRAINING_ENV", raising=False)
    assert resolved(tmp_path, tmp_path)["resume"] == "none"
    monkeypatch.setenv("SM_TRAINING_ENV", "{}")
    assert resolved(tmp_path, tmp_path)["resume"] == "auto"


# --------------------------------------------------------------------------- resume guard


def _identity(tmp_path, data, **over):
    cfg = resolved(tmp_path, data, **over)
    summary, _ = a.scan_data(cfg)
    plan = a.plan_epochs(cfg["steps"], summary["n_clips"], cfg["batch_size"], cfg["grad_accum"],
                         cfg["save_every_epochs"], cfg["checkpoint_every_steps"])
    return cfg, plan, a.run_identity(cfg, plan, summary["fingerprint"])


def fake_ckpt(root: Path, identity: dict, epoch: int, kind: str = "epoch", complete: bool = True,
              meta_override: dict | None = None) -> Path:
    name = a.final_dir_name(identity["prefix"]) if kind == "final" else a.ckpt_dir_name(identity["prefix"], epoch)
    d = root / name
    d.mkdir(parents=True)
    files = a.ADAPTER_FILES if kind == "final" else a.CKPT_FILES
    for f in files[: None if complete else -1]:
        (d / f).write_text(f)
    meta = a.ckpt_meta(identity, epoch, epoch * 5, kind)
    meta.update(meta_override or {})
    (d / a.META_NAME).write_text(json.dumps(meta))
    return d


def test_identity_hash_sensitivity(tmp_path):
    data = converted(tmp_path, "E", n=2)
    _, _, base = _identity(tmp_path, data)
    assert re.fullmatch(r"E_s42_h[0-9a-f]{6}", base["prefix"])
    for over in ({"rank": 32}, {"lr": 2e-4}, {"cfg_ratio": 0.1}, {"steps": 200}, {"model_variant": "base"},
                 {"target_modules": ["q_proj"]}):
        assert _identity(tmp_path, data, **over)[2]["cfg_hash"] != base["cfg_hash"], over
    for over in ({"num_workers": 1}, {"log_every": 5}, {"save_every_epochs": 2}, {"gradient_checkpointing": False}):
        assert _identity(tmp_path, data, **over)[2]["cfg_hash"] == base["cfg_hash"], over
    assert _identity(tmp_path, data, seed=43)[2]["prefix"] != base["prefix"]


def test_resume_none_ignores_checkpoints(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data)
    fake_ckpt(Path(cfg["checkpoint_dir"]), ident, 10)
    rp = a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    assert rp.path is None and rp.epoch == 0


def test_auto_takes_newest_complete_checkpoint_of_this_identity(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, resume="auto", steps=40)  # 2 clips: 1 step/epoch, 40 epochs
    root = Path(cfg["checkpoint_dir"])
    fake_ckpt(root, ident, 10)
    fake_ckpt(root, ident, 20)
    fake_ckpt(root, ident, 30, complete=False)  # no training_state.pt: still being written
    _, _, other = _identity(tmp_path, data, seed=7)
    fake_ckpt(root, other, 35)  # other identity: never considered
    rp = a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    assert (rp.epoch, rp.action, Path(rp.path).name) == (20, "train", a.ckpt_dir_name(ident["prefix"], 20))


def test_auto_with_no_checkpoint_trains_from_scratch(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, resume="auto")
    assert a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None)).path is None


def test_same_prefix_different_hash_is_refused(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, resume="auto", steps=40)
    fake_ckpt(Path(cfg["checkpoint_dir"]), ident, 10,
              meta_override={"cfg_hash": "0" * 64, "identity": {**ident["fields"], "rank": 8}})
    with pytest.raises(ValueError, match=r"different run.*rank"):
        a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))


def test_finished_run_needs_explicit_publish(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, resume="auto", steps=40)
    fake_ckpt(Path(cfg["checkpoint_dir"]), ident, plan["epochs"])
    with pytest.raises(ValueError, match="resume=publish"):
        a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    cfg["resume"] = "publish"
    assert a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None)).action == "publish"


def test_final_dir_counts_as_finished(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, resume="auto", steps=40)
    fake_ckpt(Path(cfg["checkpoint_dir"]), ident, 20)
    fake_ckpt(Path(cfg["checkpoint_dir"]), ident, plan["epochs"], kind="final")
    with pytest.raises(ValueError, match="already reached"):
        a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    cfg["resume"] = "publish"
    rp = a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    assert rp.action == "publish" and Path(rp.path).name.endswith("_final")


def test_publish_without_finished_run_fails(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, resume="publish", steps=40)
    with pytest.raises(FileNotFoundError):
        a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    fake_ckpt(Path(cfg["checkpoint_dir"]), ident, 10)
    with pytest.raises(ValueError, match="needs a finished run"):
        a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))


def test_explicit_path_still_checks_identity(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, steps=40)
    _, _, other = _identity(tmp_path, data, steps=80)
    d = fake_ckpt(tmp_path / "elsewhere", other, 10)
    cfg["resume"] = str(d)
    with pytest.raises(ValueError, match="different run"):
        a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None))
    d2 = fake_ckpt(tmp_path / "elsewhere2", ident, 10)
    cfg["resume"] = str(d2)
    assert a.find_resume(cfg, ident, plan, a.S3Sink(None, "E", None)).epoch == 10


# --------------------------------------------------------------------------- run orchestration (fakes)


def _fake_upstream(monkeypatch, tmp_path, resume_epochs: list[int] | None = None):
    """Replace weights, the upstream scan and the two child processes."""
    calls = []
    monkeypatch.setattr(a, "check_environment", lambda cfg: {"fake": True})
    monkeypatch.setattr(a, "ensure_weights", lambda cfg: {"fake": True})

    def build_json(data_dir, out_json, name):
        _, recs = a.scan_data(a.resolve(make_cfg(tmp_path, Path(data_dir))))
        doc = {"metadata": {"genre_ratio": 0}, "samples": [
            {"audio_path": r["audio_path"], "caption": r["caption"], "lyrics": r["lyrics"], "labeled": True,
             "bpm": r["json"].get("bpm")} for r in recs]}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc))
        return doc

    monkeypatch.setattr(a, "build_dataset_json", build_json)

    def child(cmd, env, cwd, log_path, poller=None, poll_seconds=15):
        calls.append({"cmd": cmd, "cwd": cwd})
        args = cmd[cmd.index("--plain"):]
        val = lambda f: args[args.index(f) + 1]  # noqa: E731
        if "--preprocess" in args:
            jd = json.loads(Path(val("--dataset-json")).read_text())
            t = Path(val("--tensor-output"))
            t.mkdir(parents=True, exist_ok=True)
            for s in jd["samples"]:
                (t / f"{Path(s['audio_path']).stem}.pt").write_text("pt")
            log_path.write_text("[OK] Model loaded with attn_implementation=sdpa\n")
            return 0, None
        out = Path(val("--output-dir"))
        epochs, every = int(val("--epochs")), int(val("--save-every"))
        lines = []
        for e in range(1, epochs + 1):
            lines.append(f"Epoch {e}/{epochs}, Step {e}, Loss: {1.0 / e:.4f}")
            if e % every == 0:
                d = out / "checkpoints" / f"epoch_{e}_loss_0.1000"
                d.mkdir(parents=True)
                for f in a.CKPT_FILES:
                    (d / f).write_text(f)
        (out / "final").mkdir(parents=True)
        for f in a.ADAPTER_FILES:
            (out / "final" / f).write_text("final " + f)
        lines.append("[OK] Training complete! LoRA saved to x")
        log_path.write_text("\n".join(lines) + "\n")
        if poller:
            poller(True)
        return 0, None

    monkeypatch.setattr(a, "_run_child", child)
    return calls


def test_run_end_to_end_with_fake_upstream(tmp_path, monkeypatch):
    data = converted(tmp_path, "E", n=2)
    calls = _fake_upstream(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, data, steps=6, save_every_epochs=2)
    assert a.run(cfg) == 0
    assert [("--preprocess" in c["cmd"]) for c in calls] == [True, False]
    assert all(Path(c["cwd"]) == Path(os.path.realpath(tmp_path / "work")) for c in calls)
    assert sorted(p.name for p in (tmp_path / "model" / "adapter").iterdir()) == sorted(a.ADAPTER_FILES)
    meta = json.loads((tmp_path / "model" / "run_meta.json").read_text())
    assert meta["status"] == "completed" and meta["epoch_plan"]["epochs"] == 6
    assert [c["epoch"] for c in meta["checkpoints"]] == [2, 4, 6] and len(meta["loss_curve"]) == 6
    ckpts = sorted(p.name for p in (tmp_path / "ckpt").iterdir() if p.is_dir())
    prefix = meta["identity"]["prefix"]
    assert ckpts == [f"{prefix}_epoch0002", f"{prefix}_epoch0004", f"{prefix}_epoch0006", f"{prefix}_final"]
    md = json.loads((tmp_path / "ckpt" / f"{prefix}_epoch0004" / a.META_NAME).read_text())
    assert (md["epoch"], md["global_step"], md["cfg_hash"]) == (4, 4, meta["identity"]["cfg_hash"])
    # second run: tensors are cached, finished run is refused without publish
    calls.clear()
    with pytest.raises(ValueError, match="already reached"):
        a.run(make_cfg(tmp_path, data, steps=6, save_every_epochs=2, resume="auto"))
    calls.clear()
    assert a.run(make_cfg(tmp_path, data, steps=6, save_every_epochs=2, resume="publish")) == 0
    assert calls == []


def test_run_resume_stages_checkpoint_and_offsets_seed(tmp_path, monkeypatch):
    data = converted(tmp_path, "E", n=2)
    calls = _fake_upstream(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, data, steps=6, save_every_epochs=2, resume="auto")
    _, plan, ident = _identity(tmp_path, data, steps=6, save_every_epochs=2)
    fake_ckpt(tmp_path / "ckpt", ident, 4)
    assert a.run(cfg) == 0
    train = calls[-1]["cmd"]
    staged = Path(train[train.index("--resume-from") + 1])
    assert staged.parent == Path(os.path.realpath(tmp_path / "work")) / "resume"  # under the trainer's cwd
    assert (staged / "training_state.pt").is_file()
    assert train[train.index("--seed") + 1] == "46"
    assert train[train.index("_child-train") + 1] == "--"


def test_run_fails_on_upstream_fail_line_despite_exit_zero(tmp_path, monkeypatch):
    data = converted(tmp_path, "E", n=2)
    _fake_upstream(monkeypatch, tmp_path)
    real = a._run_child

    def child(cmd, env, cwd, log_path, poller=None, poll_seconds=15):
        code, sig = real(cmd, env, cwd, log_path, poller, poll_seconds)
        if "--preprocess" not in cmd:
            log_path.write_text(log_path.read_text() + "[FAIL] Training failed: boom\n")
        return code, sig

    monkeypatch.setattr(a, "_run_child", child)
    with pytest.raises(RuntimeError, match="boom"):
        a.run(make_cfg(tmp_path, data, steps=2))
    assert json.loads((tmp_path / "out" / "run_meta.json").read_text())["status"] == "failed"


def test_run_fails_when_preprocessing_drops_a_clip(tmp_path, monkeypatch):
    data = converted(tmp_path, "E", n=3)
    _fake_upstream(monkeypatch, tmp_path)
    real = a._run_child

    def child(cmd, env, cwd, log_path, poller=None, poll_seconds=15):
        code, sig = real(cmd, env, cwd, log_path, poller, poll_seconds)
        if "--preprocess" in cmd:
            next(Path(cmd[cmd.index("--tensor-output") + 1]).glob("*.pt")).unlink()
        return code, sig

    monkeypatch.setattr(a, "_run_child", child)
    with pytest.raises(RuntimeError, match="preprocessing incomplete"):
        a.run(make_cfg(tmp_path, data, steps=2))


def test_dry_run_prints_commands(tmp_path, capsys):
    data = converted(tmp_path, "E", n=2)
    assert a.run(make_cfg(tmp_path, data), dry_run=True) == 0
    out = capsys.readouterr().out
    doc = json.loads(out[out.index("{\n"):])
    assert doc["preprocess_cmd"][2] == a.TRAIN_MODULE and "--preprocess" in doc["preprocess_cmd"]
    assert doc["train_cmd"][2:5] == ["ravi.train.acestep_lora", "_child-train", "--"]


# --------------------------------------------------------------------------- log parsing / child


def test_parse_train_log():
    text = ("[OK] Model loaded with attn_implementation=flash_attention_2\n"
            "Epoch 1/2, Step 10, Loss: 0.5123\n[OK] Epoch 1/2 in 12.5s, Loss: 0.5000\n"
            "  [FAIL] Training failed: x\n[OK] Training complete! LoRA saved to /o/final\n")
    p = a.parse_train_log(text)
    assert p["loss_curve"] == [{"epoch": 1, "step": 10, "loss": 0.5123}]
    assert p["epoch_losses"] == [{"epoch": 1, "seconds": 12.5, "loss": 0.5}]
    assert p["attn_implementation"] == "flash_attention_2" and p["complete"] and len(p["fail_lines"]) == 1


def test_child_sigterm_then_sigkill(tmp_path, monkeypatch):
    import signal
    import sys
    import threading
    import time

    monkeypatch.setattr(a, "KILL_GRACE_SECONDS", 0.5)
    script = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "print('ready', flush=True); time.sleep(30)")
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    t0 = time.time()
    code, sig = a._run_child([sys.executable, "-c", script], dict(os.environ), tmp_path, tmp_path / "l.log")
    assert sig == signal.SIGTERM and code == -signal.SIGKILL and time.time() - t0 < 10
    assert "ready" in (tmp_path / "l.log").read_text()


def test_syncer_waits_for_stable_checkpoint(tmp_path):
    data = converted(tmp_path, "E", n=2)
    cfg, plan, ident = _identity(tmp_path, data, steps=40)
    out = tmp_path / "o"
    meta = a.RunMeta(cfg, [tmp_path / "rm.json"], tmp_path / "train.log")
    sy = a.Syncer(cfg, ident, plan, out, a.S3Sink(None, "E", None), meta)
    d = out / "checkpoints" / "epoch_10_loss_0.2000"
    d.mkdir(parents=True)
    for f in a.CKPT_FILES[:-1]:
        (d / f).write_text(f)
    sy.poll()
    sy.poll()
    assert not meta.data["checkpoints"]  # training_state.pt not written yet
    (d / "training_state.pt").write_text("state")
    sy.poll()
    assert not meta.data["checkpoints"]  # first sighting
    sy.poll()
    dst = Path(cfg["checkpoint_dir"]) / a.ckpt_dir_name(ident["prefix"], 10)
    assert meta.data["checkpoints"][0]["epoch"] == 10 and a.is_complete_ckpt(dst)
    assert json.loads((dst / a.META_NAME).read_text())["global_step"] == 10 * plan["steps_per_epoch"]
