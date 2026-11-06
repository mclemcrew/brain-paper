"""Tests for ravi.train.sa3_lora (no torch, boto3 or GPU needed)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from ravi.train import sa3_lora as s

FS = "␞"


# --------------------------------------------------------------------------- helpers


def write_fake_safetensors(path: Path, metadata: dict[str, str] | None = None) -> Path:
    header = {"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + np.float32(1.5).tobytes())
    return path


def write_dataset(root: Path, n: int = 4, seconds: float = 1.0, caption: str | None = None) -> Path:
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sf.write(root / f"c{i}.wav", np.zeros((int(seconds * 44100), 2), dtype="float32"), 44100)
        (root / f"c{i}.txt").write_text(caption or FS.join(["rvkr", f"clip {i}, calm", "Jazz"]))
    return root


def make_cfg(tmp_path: Path, **over) -> dict:
    base = {
        "arm": "B",
        "data_dir": str(tmp_path / "data"),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "model_dir": str(tmp_path / "model"),
        "output_data_dir": str(tmp_path / "out"),
        "work_dir": str(tmp_path / "work"),
        "steps": 300,
        "checkpoint_every": 100,
        "seed": 42,
        "require_cuda": False,
        "require_flash_attn": False,
    }
    base.update(over)
    cfg = s.load_config(hp_path=str(tmp_path / "no_hp.json"))
    for k, v in base.items():
        cfg[k] = v
    return cfg


def resolved(tmp_path, **over) -> dict:
    cfg = s.resolve(make_cfg(tmp_path, **over))
    cfg["_identity"] = s.run_identity(cfg)
    return cfg


def put_ckpt(cfg: dict, step: int, identity: dict | None = None, directory: Path | None = None) -> Path:
    ident = identity or cfg["_identity"]
    d = directory or Path(cfg["checkpoint_dir"])
    return write_fake_safetensors(
        d / s.ckpt_name(ident["prefix"], step), {"lora_config": "{}", **s.identity_metadata(ident, step)}
    )


class FakeS3:
    """Stands in for S3Sink: objects are local files keyed by s3:// URI."""

    def __init__(self, objects: dict[str, Path] | None = None):
        self.enabled = True
        self.errors: list[str] = []
        self.objects = dict(objects or {})
        self.uploads: list[tuple[str, str]] = []

    def latest_checkpoint(self, run_prefix):
        best = None
        for uri in self.objects:
            step = s.parse_ckpt_step(uri.rsplit("/", 1)[-1], run_prefix)
            if step is not None and (best is None or step > best[0]):
                best = (step, uri)
        return best

    def download(self, uri, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[uri].read_bytes())
        return dest

    def uri(self, rel):
        return f"s3://b/p/{rel}"

    def upload(self, path, rel):
        self.uploads.append((str(path), rel))
        return self.uri(rel)


NO_S3 = s.S3Sink(None, "B", None)


# --------------------------------------------------------------------------- safetensors metadata


def test_safetensors_metadata_roundtrip(tmp_path):
    src = write_fake_safetensors(tmp_path / "a.safetensors", {"lora_config": '{"rank": 16}'})
    dst = tmp_path / "b.safetensors"
    s.write_safetensors_metadata(src, dst, {"ravi_seed": "42", "x": {"a": 1}})
    md = s.read_safetensors_metadata(dst)
    assert md == {"lora_config": '{"rank": 16}', "ravi_seed": "42", "x": '{"a": 1}'}
    assert dst.read_bytes()[-4:] == np.float32(1.5).tobytes()
    (n,) = struct.unpack("<Q", dst.read_bytes()[:8])
    assert n % 8 == 0
    try:
        from safetensors.numpy import load_file
    except ImportError:
        return
    assert load_file(str(dst))["t"][0] == 1.5


# --------------------------------------------------------------------------- identity


def test_identity_name_and_hash_sensitivity(tmp_path):
    a = resolved(tmp_path)["_identity"]
    assert a["prefix"] == f"B_s42_h{a['cfg_hash'][:6]}"
    assert s.ckpt_name(a["prefix"], 300) == f"{a['prefix']}_step000300.safetensors"
    # lora_alpha None == rank (upstream default)
    assert resolved(tmp_path, lora_alpha=16)["_identity"]["cfg_hash"] == a["cfg_hash"]
    for over in (
        {"lr": 2e-4}, {"rank": 8}, {"steps": 600}, {"duration": 30.0}, {"adapter_type": "lora"},
        {"model": "small-music-base"}, {"exclude": "foo"}, {"include": ["bar"]}, {"lora_alpha": 32},
        {"data_dir": str(tmp_path / "other")}, {"caption_aug": {"enabled": True}},
    ):
        assert resolved(tmp_path, **over)["_identity"]["cfg_hash"] != a["cfg_hash"], over
    # seed and arm are in the name, not the hash
    b = resolved(tmp_path, seed=43)["_identity"]
    assert b["cfg_hash"] == a["cfg_hash"] and b["prefix"] != a["prefix"]
    # data dir: basename only (SageMaker channel path vs. EC2 bind mount)
    assert resolved(tmp_path, data_dir=str(tmp_path / "x" / "data") + "/")["_identity"]["cfg_hash"] == a["cfg_hash"]


def test_parse_ckpt_step_requires_exact_prefix():
    assert s.parse_ckpt_step("B_s42_habcdef_step000300.safetensors", "B_s42_habcdef") == 300
    assert s.parse_ckpt_step("B_s42_habcdef_step000300.safetensors", "B_s43_habcdef") is None
    assert s.parse_ckpt_step("B_step000300.safetensors", "B_s42_habcdef") is None
    assert s.parse_ckpt_step("B_smoke_s42_habcdef_step000300.safetensors", "B_s42_habcdef") is None
    assert s.parse_ckpt_step("B_s42_habcdef_step000300.ckpt", "B_s42_habcdef") is None
    assert s.parse_ckpt_step("B_s42_habcdef_step000300.ckpt", "B_s42_habcdef", ext="ckpt") == 300


# --------------------------------------------------------------------------- resume (M3)


def test_resume_default_is_none_off_sagemaker_and_auto_on_it(tmp_path, monkeypatch):
    monkeypatch.delenv("SM_TRAINING_ENV", raising=False)
    assert resolved(tmp_path)["resume"] == "none"
    monkeypatch.setenv("SM_TRAINING_ENV", "{}")
    assert resolved(tmp_path)["resume"] == "auto"
    assert resolved(tmp_path, resume="none")["resume"] == "none"
    assert resolved(tmp_path, resume=None)["resume"] == "none"


def test_resume_none_ignores_existing_checkpoints(tmp_path):
    cfg = resolved(tmp_path, resume="none")
    put_ckpt(cfg, 300)
    assert s.find_resume(cfg, NO_S3, tmp_path) == s.ResumePlan()


def test_finished_checkpoint_of_other_seed_is_not_published(tmp_path):
    old = resolved(tmp_path, seed=42, resume="auto")
    put_ckpt(old, 300)
    write_fake_safetensors(Path(old["checkpoint_dir"]) / "B_step000300.safetensors", {})  # pre-identity name
    new = resolved(tmp_path, seed=43, resume="auto")
    plan = s.find_resume(new, NO_S3, tmp_path)
    assert plan.offset == 0 and plan.checkpoint is None and plan.action == "train"
    with pytest.raises(FileNotFoundError):
        s.find_resume(resolved(tmp_path, seed=43, resume="publish"), NO_S3, tmp_path)


def test_finished_checkpoint_needs_explicit_publish(tmp_path):
    cfg = resolved(tmp_path, resume="auto")
    put_ckpt(cfg, 300)
    with pytest.raises(ValueError, match="resume=publish"):
        s.find_resume(cfg, NO_S3, tmp_path)
    cfg["resume"] = "publish"
    plan = s.find_resume(cfg, NO_S3, tmp_path)
    assert (plan.offset, plan.action) == (300, "publish")


def test_publish_requires_final_step(tmp_path):
    cfg = resolved(tmp_path, resume="publish")
    put_ckpt(cfg, 200)
    with pytest.raises(ValueError, match="step 300"):
        s.find_resume(cfg, NO_S3, tmp_path)


@pytest.mark.parametrize("local_step,remote_step,expect", [(100, 200, "s3"), (200, 100, "local"), (200, 200, "local")])
def test_auto_takes_max_of_local_and_s3(tmp_path, local_step, remote_step, expect):
    cfg = resolved(tmp_path, resume="auto")
    put_ckpt(cfg, local_step)
    remote_file = put_ckpt(cfg, remote_step, directory=tmp_path / "bucket")
    uri = f"s3://b/p/B/checkpoints/{remote_file.name}"
    plan = s.find_resume(cfg, FakeS3({uri: remote_file}), tmp_path)
    assert plan.offset == max(local_step, remote_step)
    assert plan.source == (uri if expect == "s3" else str(Path(cfg["checkpoint_dir"]) / remote_file.name.replace(
        f"step{remote_step:06d}", f"step{local_step:06d}")))
    assert Path(plan.checkpoint).parent == Path(cfg["checkpoint_dir"])
    assert Path(plan.checkpoint).is_file()


def test_s3_only_checkpoint_downloaded(tmp_path):
    cfg = resolved(tmp_path, resume="auto")
    remote_file = put_ckpt(cfg, 100, directory=tmp_path / "bucket")
    uri = f"s3://b/p/B/checkpoints/{remote_file.name}"
    plan = s.find_resume(cfg, FakeS3({uri: remote_file}), tmp_path)
    assert plan.offset == 100 and plan.source == uri
    assert Path(plan.checkpoint) == Path(cfg["checkpoint_dir"]) / remote_file.name


def test_identity_mismatch_in_metadata_is_refused(tmp_path):
    cfg = resolved(tmp_path, resume="auto")
    other = resolved(tmp_path, lr=3e-4)["_identity"]
    # right file name, wrong identity inside (e.g. a renamed file)
    bad = dict(other, prefix=cfg["_identity"]["prefix"])
    put_ckpt(cfg, 100, identity=bad)
    with pytest.raises(ValueError, match="different run"):
        s.find_resume(cfg, NO_S3, tmp_path)
    # explicit path: same check, and a file without ravi metadata is refused
    plain = write_fake_safetensors(tmp_path / "plain_step000100.safetensors", {"lora_config": "{}"})
    with pytest.raises(ValueError, match="no ravi identity"):
        s.find_resume(dict(cfg, resume=str(plain)), NO_S3, tmp_path)


def test_explicit_path_resume(tmp_path):
    cfg = resolved(tmp_path)
    path = put_ckpt(cfg, 200, directory=tmp_path / "elsewhere")
    plan = s.find_resume(dict(cfg, resume=str(path)), NO_S3, tmp_path)
    assert (plan.offset, plan.checkpoint, plan.action) == (200, str(path), "train")
    final = put_ckpt(cfg, 300, directory=tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="resume=publish"):
        s.find_resume(dict(cfg, resume=str(final)), NO_S3, tmp_path)


class _ChildDone(Exception):
    pass


def _run_with_fake_child(tmp_path, monkeypatch, cfg, produce_final=True, result=(0, None), s3=None):
    """Run s.run() with the environment check and child process stubbed out."""
    seen = {}
    monkeypatch.setattr(s, "check_environment", lambda cfg: {"gpus": []})
    if s3 is not None:
        monkeypatch.setattr(s, "S3Sink", lambda *a, **k: s3)

    def fake_child(cmd, env, cwd, log_path, cfg_, save_dir, offset, s3_, meta):
        seen.update(cmd=cmd, env=env, offset=offset, cwd=cwd)
        log_path.write_text("train/loss=0.5\n")
        if produce_final:
            put_ckpt(cfg_, cfg_["steps"])
        return result

    monkeypatch.setattr(s, "_run_child", fake_child)
    code = s.run(cfg)
    return code, seen


def test_run_seed_arithmetic_on_resume(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    cfg = make_cfg(tmp_path, resume="auto")
    r = s.resolve(cfg)
    r["_identity"] = s.run_identity(r)
    put_ckpt(r, 200)
    code, seen = _run_with_fake_child(tmp_path, monkeypatch, cfg)
    assert code == 0
    argv = seen["cmd"][seen["cmd"].index("--") + 1 :]
    assert argv[argv.index("--steps") + 1] == "100"
    assert argv[argv.index("--seed") + 1] == "242"
    assert argv[argv.index("--lora_checkpoint") + 1].endswith(s.ckpt_name(r["_identity"]["prefix"], 200))
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text())
    assert meta["status"] == "completed" and meta["resume"]["completed_steps"] == 200
    assert meta["effective_seed"] == 242


def test_run_finished_other_seed_trains_from_scratch(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    old = resolved(tmp_path, seed=42)
    put_ckpt(old, 300)
    cfg = make_cfg(tmp_path, seed=43, resume="auto")
    code, seen = _run_with_fake_child(tmp_path, monkeypatch, cfg)
    argv = seen["cmd"][seen["cmd"].index("--") + 1 :]
    assert code == 0 and "--lora_checkpoint" not in argv
    assert argv[argv.index("--steps") + 1] == "300" and argv[argv.index("--seed") + 1] == "43"


def test_run_publish_does_not_launch_child(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    r = resolved(tmp_path)
    final = put_ckpt(r, 300)
    cfg = make_cfg(tmp_path, resume="publish")
    code, seen = _run_with_fake_child(tmp_path, monkeypatch, cfg)
    assert code == 0 and seen == {}
    assert (tmp_path / "model" / "adapter.safetensors").read_bytes() == final.read_bytes()


# --------------------------------------------------------------------------- long clips (M4)


def test_long_clips_error_by_default_with_reason(tmp_path):
    write_dataset(tmp_path / "data", seconds=3.0)
    cfg = resolved(tmp_path, duration=2.0)
    assert cfg["long_clip_policy"] == "error"
    with pytest.raises(ValueError) as e:
        s.scan_dataset(cfg)
    assert "seconds_total" in str(e.value) and "noise-schedule shift" in str(e.value)
    assert "diffusion.py:311-319" in str(e.value)


def test_long_clips_warn_in_smoke_mode(tmp_path, capsys):
    write_dataset(tmp_path / "data", seconds=3.0)
    cfg = resolved(tmp_path, duration=2.0, smoke=True)
    assert cfg["long_clip_policy"] == "warn"
    assert s.scan_dataset(cfg)["n_longer_than_window"] == 4
    assert "WARNING" in capsys.readouterr().out
    # explicit setting wins over the smoke default
    assert resolved(tmp_path, smoke=True, long_clip_policy="error")["long_clip_policy"] == "error"
    with pytest.raises(ValueError):
        resolved(tmp_path, long_clip_policy="crop")


# --------------------------------------------------------------------------- config


def test_config_precedence_with_sagemaker_json_string_hyperparameters(tmp_path):
    conf = tmp_path / "run.json"
    conf.write_text(json.dumps({"arm": "fromfile", "steps": 1000, "lr": 3e-4, "rank": 8,
                                "caption_aug": {"enabled": True, "p_lowercase": 0.1}}))
    hp = tmp_path / "hp.json"
    # SageMaker Python SDK: every value a string, str values JSON-encoded
    hp.write_text(json.dumps({
        "config": json.dumps(str(conf)),
        "arm": '"123"',
        "steps": "6000",
        "lr": "1e-4",
        "exclude_duration_conditioner": "true",
        "caption_aug": '{"p_lowercase": 0.25}',
        "caption_aug.p_drop_field": "0.2",
        "caption_aug.trigger": '"007"',
        "s3_out": '"s3://b/adapters"',
        "resume": '"none"',
        "data_dir": "null",
        "sagemaker_program": "x",
        "sagemaker_region": '"us-east-1"',
        "_tuning_objective_metric": "loss",
        "_anything": "1",
    }))
    cfg = s.load_config(None, str(hp), ["steps=400", "caption_aug.enabled=false"])
    assert cfg["arm"] == "123" and isinstance(cfg["arm"], str)
    assert cfg["steps"] == 400  # --set beats hyperparameters beats file
    assert cfg["lr"] == 1e-4 and cfg["rank"] == 8  # hp beats file; file beats default
    assert cfg["exclude_duration_conditioner"] is True
    assert cfg["caption_aug"] == {"enabled": False, "p_lowercase": 0.25, "p_drop_field": 0.2, "trigger": "007"}
    assert cfg["s3_out"] == "s3://b/adapters" and cfg["resume"] == "none"
    assert cfg["data_dir"] == "null"  # paths are never coerced
    assert cfg["_sources"]["config"] == str(conf)


@pytest.mark.parametrize("raw", ["123", '"123"', "true", "null", "none", "1e5", "0915", "007", "B_smoke"])
def test_string_keys_never_coerced(tmp_path, raw):
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"arm": raw, "run_id": raw}))
    cfg = s.load_config(None, str(hp))
    expect = json.loads(raw) if raw.startswith('"') else raw
    assert cfg["arm"] == expect and cfg["run_id"] == expect


def test_set_and_yaml_style_ints_for_arm(tmp_path):
    cfg = s.load_config(hp_path=str(tmp_path / "none.json"), sets=["arm=0915", "steps=200", "resume=none"])
    assert cfg["arm"] == "0915" and cfg["steps"] == 200 and cfg["resume"] == "none"
    conf = tmp_path / "c.json"
    conf.write_text(json.dumps({"arm": 123}))
    assert s.load_config(str(conf), str(tmp_path / "none.json"))["arm"] == "123"


def test_unknown_keys_rejected_but_underscore_hp_keys_ignored(tmp_path):
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"_tuning_objective_metric": "x"}))
    s.load_config(None, str(hp))
    hp.write_text(json.dumps({"stepz": "1"}))
    with pytest.raises(ValueError, match="unknown config keys"):
        s.load_config(None, str(hp))


def test_exclude_string_split_on_whitespace(tmp_path):
    cfg = resolved(tmp_path, exclude="foo  bar", include="blocks.0")
    assert cfg["exclude"] == ["foo", "bar"] and cfg["include"] == ["blocks.0"]
    assert s.exclude_patterns(cfg) == ["foo", "bar", "seconds_total"]
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"exclude": '["a", "b c"]'}))
    assert s.load_config(None, str(hp))["exclude"] == ["a", "b c"]


# --------------------------------------------------------------------------- preflight


def test_unreadable_audio_is_an_error(tmp_path):
    write_dataset(tmp_path / "data")
    (tmp_path / "data" / "broken.wav").write_bytes(b"RIFF....not audio")
    (tmp_path / "data" / "broken.txt").write_text(FS.join(["rvkr", "x"]))
    with pytest.raises(ValueError, match="unreadable.*broken.wav"):
        s.scan_dataset(resolved(tmp_path))


def test_preflight_checks_trigger_when_augmenting(tmp_path):
    write_dataset(tmp_path / "data", caption="rvkr, comma, separated")
    s.scan_dataset(resolved(tmp_path))  # augmentation off: any caption is fine
    with pytest.raises(ValueError, match="trigger"):
        s.scan_dataset(resolved(tmp_path, caption_aug={"enabled": True}))


def test_pin_single_gpu():
    env = {}
    s.pin_single_gpu(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    env = {"CUDA_VISIBLE_DEVICES": "3"}
    s.pin_single_gpu(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    with pytest.raises(RuntimeError):
        s.pin_single_gpu({"CUDA_VISIBLE_DEVICES": "0,1"})


def test_child_env_pins_one_gpu(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _, seen = _run_with_fake_child(tmp_path, monkeypatch, make_cfg(tmp_path))
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"



# --------------------------------------------------------------------------- run outcome / S3


def test_s3_errors_after_successful_training_exit_zero(tmp_path, monkeypatch, capsys):
    write_dataset(tmp_path / "data")
    fake = FakeS3()
    fake.errors.append("upload x failed")
    code, _ = _run_with_fake_child(tmp_path, monkeypatch, make_cfg(tmp_path, s3_out="s3://b/p"), s3=fake)
    assert code == 0
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text())
    assert meta["status"] == "completed_with_s3_errors" and meta["s3_errors"] == ["upload x failed"]
    assert "S3 uploads failed" in capsys.readouterr().out
    assert any(rel.endswith("train.log") for _, rel in fake.uploads)


def test_interrupted_run_is_reported_and_log_uploaded(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    fake = FakeS3()
    with pytest.raises(s.Interrupted):
        _run_with_fake_child(tmp_path, monkeypatch, make_cfg(tmp_path, s3_out="s3://b/p"),
                             produce_final=False, result=(-15, 15), s3=fake)
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text())
    assert meta["status"] == "interrupted" and meta["exit_code"] == -15
    assert any(rel.endswith("train.log") for _, rel in fake.uploads)


def test_failed_run_uploads_log(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    fake = FakeS3()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        _run_with_fake_child(tmp_path, monkeypatch, make_cfg(tmp_path, s3_out="s3://b/p"),
                             produce_final=False, result=(1, None), s3=fake)
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text())
    assert meta["status"] == "failed"
    assert any(rel.endswith("train.log") for _, rel in fake.uploads)


def test_s3sink_download_without_s3_out(tmp_path, monkeypatch):
    import sys
    import types

    calls = []

    class Client:
        def download_file(self, bucket, key, dest):
            calls.append((bucket, key))
            Path(dest).write_text("x")

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *a, **k: Client()))
    sink = s.S3Sink(None, "B", None)
    assert not sink.enabled and sink.upload(tmp_path / "f", "x") is None
    assert sink.latest_checkpoint("B_s1_habcdef") is None
    sink.download("s3://bkt/a/b.safetensors", tmp_path / "d" / "b.safetensors")
    assert calls == [("bkt", "a/b.safetensors")]


def test_paths_resolved_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = s.resolve(make_cfg(tmp_path, data_dir="data", checkpoint_dir="./ck", work_dir="w", resume="ck/x.safetensors"))
    for k in ("data_dir", "checkpoint_dir", "work_dir", "resume"):
        assert Path(cfg[k]).is_absolute(), k
    assert cfg["data_dir"] == str((tmp_path / "data").resolve())


def test_resume_prefers_fp32_ckpt_sibling(tmp_path):
    cfg = resolved(tmp_path, resume="auto")
    st = put_ckpt(cfg, 100)
    plan = s.find_resume(cfg, NO_S3, tmp_path)
    assert plan.checkpoint == str(st) == plan.safetensors
    st.with_suffix(".ckpt").write_bytes(b"fp32")
    plan = s.find_resume(cfg, NO_S3, tmp_path)
    assert plan.checkpoint == str(st.with_suffix(".ckpt")) and plan.safetensors == str(st)


# --------------------------------------------------------------------------- Syncer


def _syncer(tmp_path, monkeypatch, offset=0):
    cfg = resolved(tmp_path, run_id="r1")
    save_dir = tmp_path / "work" / "seg"
    demo_dir = save_dir / "demos"
    demo_dir.mkdir(parents=True)
    converted = []

    def fake_convert(src, dst, metadata, tmp_dir):
        assert Path(tmp_dir) == Path(cfg["work_dir"]) / "partial"
        converted.append((src.name, dst.name, metadata["ravi_step"]))
        write_fake_safetensors(dst, {"lora_config": "{}", **metadata})

    monkeypatch.setattr(s, "convert_ckpt", fake_convert)
    fake = FakeS3()
    meta = s.RunMeta(cfg, [tmp_path / "out" / "run_meta.json"])
    return cfg, s.Syncer(cfg, save_dir, demo_dir, offset, fake, meta), save_dir, demo_dir, converted, fake, meta


def test_syncer_waits_for_stable_files_and_publishes_both_formats(tmp_path, monkeypatch):
    cfg, sync, save_dir, demo_dir, converted, fake, meta = _syncer(tmp_path, monkeypatch, offset=200)
    prefix = cfg["_identity"]["prefix"]
    ck_dir = save_dir / "lightning_logs" / "version_0" / "checkpoints"
    ck_dir.mkdir(parents=True)
    done = ck_dir / "epoch=0-step=100.ckpt"
    done.write_bytes(b"a" * 10)
    growing = ck_dir / "epoch=0-step=200.ckpt"
    growing.write_bytes(b"b" * 5)
    (demo_dir / "demo_cfg_4_101.wav").write_bytes(b"w" * 8)

    sync.poll()  # first sighting: nothing is stable yet
    assert converted == []
    growing.write_bytes(b"b" * 50)  # still being written
    sync.poll()
    assert converted == [("epoch=0-step=100.ckpt", s.ckpt_name(prefix, 300), "300")]
    ckpt_dir = Path(cfg["checkpoint_dir"])
    assert (ckpt_dir / s.ckpt_name(prefix, 300, "ckpt")).read_bytes() == b"a" * 10
    assert s.read_safetensors_metadata(ckpt_dir / s.ckpt_name(prefix, 300))["ravi_cfg_hash"] == cfg["_identity"]["cfg_hash"]
    rels = [rel for _, rel in fake.uploads]
    assert f"checkpoints/{s.ckpt_name(prefix, 300)}" in rels and f"checkpoints/{s.ckpt_name(prefix, 300, 'ckpt')}" in rels
    assert (Path(cfg["output_data_dir"]) / "demos" / "demo_step000301_cfg4.wav").is_file()
    assert "runs/r1/demos/demo_step000301_cfg4.wav" in rels
    assert not list((Path(cfg["work_dir"]) / "partial").glob("*"))  # temp files moved away

    sync.poll()  # growing file now stable
    assert [c[2] for c in converted] == ["300", "400"]
    sync.poll(final=True)  # nothing new; no double publish
    assert len(converted) == 2 and len(meta.data["checkpoints"]) == 2
    assert meta.data["checkpoints"][0]["ckpt_fp32"].endswith(".ckpt")


def test_syncer_final_poll_publishes_unstable_file(tmp_path, monkeypatch):
    cfg, sync, save_dir, *_ , converted, fake, meta = _syncer(tmp_path, monkeypatch)
    (save_dir / "epoch=0-step=100.ckpt").write_bytes(b"x")
    sync.poll(final=True)
    assert [c[2] for c in converted] == ["100"]


# --------------------------------------------------------------------------- child lifecycle


def _child_runner(tmp_path, monkeypatch, script):
    cfg, sync, save_dir, demo_dir, converted, fake, meta = _syncer(tmp_path, monkeypatch)
    import sys

    cmd = [sys.executable, "-c", script]
    return lambda: s._run_child(cmd, dict(__import__("os").environ), demo_dir, tmp_path / "out" / "train.log",
                                cfg, save_dir, 0, fake, meta)


def test_child_exit_code_and_log(tmp_path, monkeypatch):
    run = _child_runner(tmp_path, monkeypatch, "import sys; print('hello\\rworld'); sys.exit(3)")
    assert run() == (3, None)
    assert (tmp_path / "out" / "train.log").read_text() == "hello\nworld\n"


def test_sigterm_forwarded_then_sigkill_after_grace(tmp_path, monkeypatch):
    import os
    import signal
    import threading
    import time

    monkeypatch.setattr(s, "KILL_GRACE_SECONDS", 0.5)
    # child ignores SIGTERM, so only the SIGKILL fallback can stop it
    script = ("import signal, time, sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "print('ready', flush=True); time.sleep(30)")
    run = _child_runner(tmp_path, monkeypatch, script)
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    t0 = time.time()
    code, sig = run()
    assert sig == signal.SIGTERM and code == -signal.SIGKILL
    assert time.time() - t0 < 10
    assert signal.getsignal(signal.SIGTERM) is not None  # handler restored


def test_child_killed_if_wrapper_raises(tmp_path, monkeypatch):
    import subprocess

    procs = []
    real_popen = subprocess.Popen

    def popen(*a, **k):
        procs.append(real_popen(*a, **k))
        return procs[-1]

    monkeypatch.setattr(s.subprocess, "Popen", popen)
    monkeypatch.setattr(s.Syncer, "start", lambda self: (_ for _ in ()).throw(OSError("boom")))
    run = _child_runner(tmp_path, monkeypatch, "import time; time.sleep(30)")
    with pytest.raises(OSError, match="boom"):
        run()
    assert procs[0].poll() is not None


# --------------------------------------------------------------------------- upstream contract


def _upstream_train_lora() -> Path:
    import os

    root = os.environ.get("SA3_ROOT")
    if not root:
        pytest.skip("SA3_ROOT not set (point it at a stable-audio-3 checkout)")
    path = Path(root) / "scripts" / "train_lora.py"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return path


def test_every_emitted_flag_exists_in_upstream_train_lora(tmp_path):
    import re

    src = _upstream_train_lora().read_text()  # read as text only; never imported
    upstream = set(re.findall(r"""add_argument\(\s*["'](--[A-Za-z0-9_-]+)["']""", src))
    assert {"--model", "--data_dir", "--steps", "--lora_checkpoint"} <= upstream
    full = resolved(tmp_path, lora_alpha=8, include=["blocks"], exclude=["foo"], svd_bases_path="/x.pt",
                    dropout=0.1)
    enc = resolved(tmp_path, data_dir=None, encoded_dir=str(tmp_path / "enc"))
    emitted = set()
    for cfg in (full, enc):
        argv = s.build_train_lora_argv(cfg, tmp_path / "save", 100, 7, "/ck.ckpt")
        emitted |= {a for a in argv if a.startswith("--")}
    missing = emitted - upstream
    assert not missing, f"flags not accepted by upstream train_lora.py: {sorted(missing)}"
    assert {"--lora_alpha", "--include", "--exclude", "--svd_bases_path", "--encoded_dir"} <= emitted
