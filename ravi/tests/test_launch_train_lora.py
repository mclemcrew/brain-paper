"""Tests for ravi.train._launch_train_lora against a fake scripts/train_lora.py."""

from __future__ import annotations

import json
import sys

import pytest

from ravi.train import _launch_train_lora as launch
from ravi.train import caption_aug as ca

FS = ca.FIELD_SEP

FAKE_TRAIN_LORA = '''
import json, os, sys
from pathlib import Path


class PreEncodedDataset:
    def __getitem__(self, idx):
        return "latents", {"prompt": "rvkr\\u241ea\\u241eb"}


class DiffusionCondTrainingWrapper:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


''' + ca.UPSTREAM_CAPTION_FN_SOURCE + '''

def main():
    out = {"argv": sys.argv}
    out["caption_fn_type"] = type(caption_metadata_fn).__name__
    out["prompt"] = caption_metadata_fn({"path": os.environ["FAKE_WAV"]}, None)["prompt"]
    out["pre_encoded_prompt"] = PreEncodedDataset()[0][1]["prompt"]
    w = DiffusionCondTrainingWrapper(inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}})
    out["inpainting_config"] = w.kwargs["inpainting_config"]
    out["wrapper_name"] = DiffusionCondTrainingWrapper.__name__
    Path(os.environ["FAKE_OUT"]).write_text(json.dumps(out))
'''


@pytest.fixture
def fake_sa3(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "train_lora.py").write_text(FAKE_TRAIN_LORA)
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    wav.with_suffix(".txt").write_text(FS.join(["rvkr", "A warm, mellow ballad", "Jazz"]) + "\n")
    monkeypatch.setenv("FAKE_WAV", str(wav))
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "out.json"))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.delitem(sys.modules, launch.MODULE_NAME, raising=False)
    # the launcher runs in the main process, where the hook never augments
    monkeypatch.setattr(ca, "_in_dataloader_worker", lambda: False)
    yield root, tmp_path / "out.json"
    sys.modules.pop(launch.MODULE_NAME, None)


def run_launcher(root, *ours, theirs=("--model", "medium-base", "--steps", "10")):
    launch.main(["--sa3-root", str(root), *ours, "--", *theirs])


def test_patched_caption_fn_is_used_and_argv_passed_through(fake_sa3):
    root, out = fake_sa3
    aug = ca.CaptionAugConfig(enabled=True).to_json()
    run_launcher(root, "--caption-aug-json", aug, theirs=("--model", "medium-base", "--exclude", "a", "b"))
    res = json.loads(out.read_text())
    assert res["argv"] == [str(root / "scripts" / "train_lora.py"), "--model", "medium-base", "--exclude", "a", "b"]
    assert res["caption_fn_type"] == "CaptionAugHook"
    assert res["prompt"] == "rvkr, A warm, mellow ballad, Jazz"
    assert res["pre_encoded_prompt"] == "rvkr, a, b"
    assert res["inpainting_config"]["mask_kwargs"]["mask_type_probabilities"] == [0.1, 0.8, 0.1]


def test_hook_installed_even_when_disabled(fake_sa3):
    root, out = fake_sa3
    run_launcher(root, "--caption-aug-json", ca.CaptionAugConfig().to_json())
    res = json.loads(out.read_text())
    assert res["caption_fn_type"] == "CaptionAugHook"
    assert res["prompt"] == "rvkr, A warm, mellow ballad, Jazz"  # field_sep never reaches the model


def test_without_caption_json_upstream_fn_is_untouched(fake_sa3):
    root, out = fake_sa3
    run_launcher(root)
    res = json.loads(out.read_text())
    assert res["caption_fn_type"] == "function"
    assert res["prompt"] == FS.join(["rvkr", "A warm, mellow ballad", "Jazz"])


def test_guard_refuses_mutated_upstream_source(fake_sa3):
    root, out = fake_sa3
    path = root / "scripts" / "train_lora.py"
    path.write_text(path.read_text().replace('txt.read_text().strip()', 'txt.read_text().strip().lower()'))
    with pytest.raises(RuntimeError, match="differs"):
        run_launcher(root, "--caption-aug-json", ca.CaptionAugConfig().to_json())
    assert not out.exists()


def test_inpaint_mask_probabilities_override(fake_sa3):
    root, out = fake_sa3
    run_launcher(root, "--inpaint-mask-probs", "[0.25, 0.25, 0.25, 0.25]")
    res = json.loads(out.read_text())
    assert res["inpainting_config"] == {"mask_kwargs": {"mask_type_probabilities": [0.25, 0.25, 0.25, 0.25]}}
    assert res["wrapper_name"] == "DiffusionCondTrainingWrapper"


@pytest.mark.parametrize(
    "probs", ["[0.5, 0.6, 0.1]", "[0.5, 0.5]", "[0.2, 0.2, 0.2, 0.2, 0.2]", "[1.2, -0.1, -0.1]", '["a", 1, 0]']
)
def test_bad_inpaint_mask_probabilities_rejected(fake_sa3, probs):
    root, out = fake_sa3
    with pytest.raises(ValueError, match="mask probabilities"):
        run_launcher(root, "--inpaint-mask-probs", probs)
    assert not out.exists()


def test_missing_separator_is_usage_error(fake_sa3):
    root, _ = fake_sa3
    with pytest.raises(SystemExit):
        launch.main(["--sa3-root", str(root)])


def test_missing_train_lora_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        launch.load_train_lora(tmp_path)


def test_wrapper_validates_mask_probabilities_before_launch(tmp_path):
    from ravi.train import sa3_lora

    cfg = sa3_lora.load_config(hp_path=str(tmp_path / "none.json"))
    cfg.update(data_dir=str(tmp_path), inpaint_mask_type_probabilities=[1.2, -0.1, -0.1])
    with pytest.raises(ValueError, match="non-negative"):
        sa3_lora.resolve(cfg)
    cfg["inpaint_mask_type_probabilities"] = ["0.1", "0.8", "0.1"]
    assert sa3_lora.resolve(cfg)["inpaint_mask_type_probabilities"] == [0.1, 0.8, 0.1]
