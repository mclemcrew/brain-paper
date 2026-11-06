"""Contract tests for the Stable Audio 3 SageMaker handler in uapkg
(``Sagemaker_Deployment/stable-audio-3-finetuned/code``): request validation, mask planning,
the splice that keeps audio outside the mask bit-identical, and a full ``predict_fn`` pass
with the SA3 model faked (no torch, no GPU, no AWS).

The handler lives in another repo; set ``RAVI_SA3_SERVING_CODE`` to its ``code/`` directory if
it is not at the default path. The whole module is skipped when it is absent.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

DEFAULT_CODE = (Path.home() / "Development/uapkg/packages/system/algo/Sagemaker_Deployment"
                / "stable-audio-3-finetuned/code")
CODE = Path(os.environ.get("RAVI_SA3_SERVING_CODE", DEFAULT_CODE)).expanduser()
if not (CODE / "inference.py").is_file():
    pytest.skip(f"SA3 serving handler not found at {CODE} (set RAVI_SA3_SERVING_CODE)", allow_module_level=True)

sys.path.insert(0, str(CODE))
import inference as inf  # noqa: E402
import serving_core as core  # noqa: E402
from ravi_vendor import masks as vmasks  # noqa: E402
from ravi_vendor import sa3_infer_helpers as vh  # noqa: E402

from ravi.eval import masks as bp_masks  # noqa: E402
from ravi.eval import sa3_infer as bp_infer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SR = 8000  # fake model rate (the logic is rate-agnostic; the real model is 44.1 kHz)
DS = 256  # fake latent downsampling ratio
BUCKET = "ngmc-poc-uploads-dev-297416720456"


# --------------------------------------------------------------------------- vendoring drift


def test_vendored_masks_and_labels_match_brain_paper():
    src = (REPO / "ravi/eval/masks.py").read_text(encoding="utf-8")
    vend = (CODE / "ravi_vendor/masks.py").read_text(encoding="utf-8")
    assert vend == src.replace("from ravi.structure.labels import", "from .labels import"), \
        "re-run vendor_ravi.py in the uapkg package"
    assert (CODE / "ravi_vendor/labels.py").read_text(encoding="utf-8") == \
        (REPO / "ravi/structure/labels.py").read_text(encoding="utf-8")
    for name in ("splice", "crossfade_ramp", "seconds_to_samples"):
        assert inspect.getsource(getattr(vmasks, name)) == inspect.getsource(getattr(bp_masks, name))


@pytest.mark.parametrize("name", [
    "sha256_file", "hf_env", "sa3_commit", "load_audio", "resample", "to_model_audio", "fix_length",
    "diff_stats", "choose_context", "effective_latent_region", "mask_seconds", "Engine",
    "_np_to_torch", "_torch_to_np",
])
def test_vendored_sa3_infer_definitions_are_verbatim(name):
    assert inspect.getsource(getattr(vh, name)) == inspect.getsource(getattr(bp_infer, name))


def test_vendored_constants_match():
    for name in ("MODEL_DEFAULTS", "DEFAULT_SAMPLE_SIZE", "DURATION_PADDING_S", "DEFAULT_CONTEXT_S",
                 "DEFAULT_MIN_CONTEXT_S"):
        assert getattr(vh, name) == getattr(bp_infer, name)
    assert core.DEFAULTS["context_s"] == bp_infer.DEFAULT_CONTEXT_S
    assert core.DEFAULTS["min_context_s"] == bp_infer.DEFAULT_MIN_CONTEXT_S
    assert core.DEFAULTS["pad_s"] == bp_masks.DEFAULT_PAD_S
    assert core.DEFAULTS["crossfade_ms"] == bp_masks.DEFAULT_XFADE_S * 1000
    assert core.DEFAULTS["steps"] == bp_infer.MODEL_DEFAULTS["medium"]["steps"]
    assert core.DEFAULTS["cfg_scale"] == bp_infer.MODEL_DEFAULTS["medium"]["cfg"]


CANVAS_FIELD_TYPES = {"number", "string", "enum", "boolean", "s3_audio", "time_range"}  # ngmc_poc lambda/types.ts


def test_parameter_schema_matches_the_canvas_field_format_and_the_handler():
    schema = json.loads((CODE.parent / "parameter_schema.json").read_text())
    assert schema["schema_version"] == 2 and isinstance(schema["fields"], list)
    fields = {f["name"]: f for f in schema["fields"]}
    assert len(fields) == len(schema["fields"])
    assert {f["type"] for f in schema["fields"]} <= CANVAS_FIELD_TYPES
    sent = set()  # the flat parameter names the canvas emits (time_range `x` -> x_start_s / x_end_s)
    for f in schema["fields"]:
        sent |= {f"{f['name']}_start_s", f"{f['name']}_end_s"} if f["type"] == "time_range" else {f["name"]}
        assert set(f.get("modes", [])) <= set(core.MODES), f["name"]
    assert sent <= core.KNOWN_FIELDS
    assert {"mode", "prompt", "adapter", "lora_strength", "source_audio_s3", "mask_start_s", "mask_end_s",
            "pad_s", "crossfade_ms", "steps", "cfg_scale", "seed", "duration_seconds"} <= sent
    assert [o["value"] for o in fields["mode"]["options"]] == list(core.MODES)
    assert fields["adapter"]["options"][0]["value"] == core.BASE_ADAPTER
    for k in ("steps", "cfg_scale", "lora_strength", "pad_s", "crossfade_ms", "context_s", "pre_s"):
        assert fields[k]["default"] == core.DEFAULTS[k], k
    assert fields["duration_seconds"]["default"] == core.DEFAULTS["duration_seconds"]
    assert fields["seed"]["default"] == -1 and parse({"prompt": "p", "seed": -1}).seed_was_random
    assert fields["mask"]["modes"] == ["inpaint"] and fields["source_audio_s3"]["modes"] == ["inpaint", "continue"]


# --------------------------------------------------------------------------- request parsing


ADAPTERS = ["A", "B", "E"]


def parse(body, **kw):
    kw.setdefault("adapters", ADAPTERS)
    return core.parse_request(json.dumps(body).encode(), "application/json", **kw)


def inpaint_body(**kw):
    b = {"mode": "inpaint", "prompt": "rvkr, warm piano", "source_audio_s3": f"s3://{BUCKET}/u/track.wav",
         "mask_start_s": 10.0, "mask_end_s": 20.0, "inference_id": "abc-123"}
    b.update(kw)
    return b


def test_generate_defaults_and_random_seed_is_reported():
    r = parse({"prompt": "  rvkr,   upbeat  ", "inference_id": "x1"})
    assert (r.mode, r.prompt, r.steps, r.cfg_scale, r.adapter, r.lora_strength) == \
        ("generate", "rvkr, upbeat", 8, 1.0, "base_model", 1.0)
    assert r.duration_seconds == 30.0 and r.seed_was_random and 0 <= r.seed <= core.SEED_MAX
    for s in (None, -1, "-1", "", -1.0, "-1.0"):
        assert parse({"prompt": "p", "seed": s}).seed_was_random
    r = parse({"prompt": "p", "seed": "42", "steps": "8", "cfg_scale": "1.5", "duration_seconds": 12})
    assert (r.seed, r.seed_was_random, r.steps, r.cfg_scale, r.duration_seconds) == (42, False, 8, 1.5, 12.0)


@pytest.mark.parametrize("body,match", [
    ({}, "prompt is required"),
    ({"prompt": "   "}, "prompt is required"),
    ({"prompt": "p", "mode": "remix"}, "mode must be"),
    ({"prompt": "p", "mode": 0}, "mode must be"),
    ({"prompt": "p", "mode": False}, "mode must be"),
    ({"prompt": "p", "steps": 0}, "steps must be >= 1"),
    ({"prompt": "p", "steps": 8.5}, "integer"),
    ({"prompt": "p", "steps": True}, "number"),
    ({"prompt": "p", "cfg_scale": "nan"}, "finite"),
    ({"prompt": "p", "seed": -5}, "seed must be >= 0"),
    ({"prompt": "p", "duration_seconds": 0}, "duration_seconds must be > 0"),
    ({"prompt": "p", "duration_seconds": 500}, "<= 120"),
    ({"prompt": "p", "lora_strength": 3, "adapter": "B"}, "lora_strength must be <= 2"),
    ({"prompt": "p", "adapter": "Z"}, "unknown adapter 'Z'"),
    ({"prompt": "p", "adapter": "../x"}, "not a valid arm id"),
])
def test_bad_requests_are_refused(body, match):
    with pytest.raises(core.RequestError, match=match):
        parse(body)


def test_content_type_and_body_shape():
    with pytest.raises(core.RequestError, match="content type"):
        core.parse_request(b'{"prompt": "p"}', "audio/wav")
    with pytest.raises(core.RequestError, match="valid JSON"):
        core.parse_request(b"{nope", "application/json")
    with pytest.raises(core.RequestError, match="valid JSON"):
        core.parse_request(b"[" * 100000 + b"]" * 100000, "application/json")  # RecursionError
    with pytest.raises(core.RequestError, match="JSON object"):
        core.parse_request(b"[1]", "application/json; charset=utf-8")
    assert core.parse_request({"prompt": "p"}, None).prompt == "p"  # dicts pass straight through
    assert issubclass(core.RequestError, ValueError)  # the toolkit maps ValueError to a client error


def test_adapter_resolution_aliases_and_legacy_checkpoint():
    assert parse({"prompt": "p", "adapter": "b"}).adapter == "B"
    for alias in ("base_model", "BASE", "none"):
        assert parse({"prompt": "p", "adapter": alias}).adapter == "base_model"
    r = parse({"prompt": "p", "checkpoint": "e"})
    assert r.adapter == "E" and any("legacy field checkpoint" in w for w in r.warnings)
    r = parse({"prompt": "p", "checkpoint": "ai_descriptions"})  # the old canvas value
    assert r.adapter == "base_model" and any("not an adapter" in w for w in r.warnings)
    r = parse({"prompt": "p", "adapter": "A", "checkpoint": "B"})
    assert r.adapter == "A" and any("checkpoint 'B' ignored" in w for w in r.warnings)
    assert parse({"prompt": "p"}, default_adapter="B").adapter == "B"
    # before model_fn the adapter list is unknown: shape only
    assert core.parse_request({"prompt": "p", "adapter": "Q"}, None).adapter == "Q"


def test_inference_id_and_unknown_fields():
    r = parse({"prompt": "p", "inference_id": "gen-ab+c/=d-1", "node_id": "n1"})
    assert r.inference_id == "gen-ab_c__d-1" and not r.inference_id_generated
    assert any("node_id" in w for w in r.warnings) and any("sanitized" in w for w in r.warnings)
    r = parse({"prompt": "p"})
    assert r.inference_id.startswith("sa3-") and r.inference_id_generated


def test_inpaint_fields_lists_and_validation():
    r = parse(inpaint_body(mask_start_s=[30, "5"], mask_end_s="[40, 9.5]", pad_s=0.5, crossfade_ms=20))
    assert r.mask_start_s == (30.0, 5.0) and r.mask_end_s == (40.0, 9.5)
    assert (r.pad_s, r.crossfade_ms, r.context_s) == (0.5, 20.0, 47.0)
    for kw, match in [
        ({"source_audio_s3": None}, "source_audio_s3"),
        ({"source_audio_s3": "https://x/y.wav"}, "s3://bucket/key"),
        ({"source_audio_s3": "s3://someone-elses-bucket/a.wav"}, "not allowed"),
        ({"mask_end_s": None}, "required in inpaint"),
        ({"mask_start_s": [1, 2], "mask_end_s": [3]}, "2 values"),
        ({"mask_start_s": 5, "mask_end_s": 5.05}, "at least 0.1 s"),
        ({"mask_start_s": -1}, "< 0"),
        ({"context_s": 6}, "context_s must be 0"),
        ({"mask_start_s": list(range(9)), "mask_end_s": [x + 1 for x in range(9)]}, "at most 8"),
    ]:
        with pytest.raises(core.RequestError, match=match):
            parse(inpaint_body(**kw))
    assert parse(inpaint_body(source_audio_s3="s3://any-bucket/a.mp3"), allowed_buckets=None).source_audio_s3
    r = parse(inpaint_body(source_audio_s3=f"s3://{BUCKET}/input/123/take_1"))  # canvas keeps user file names
    assert any("decoding by content" in w for w in r.warnings)
    with pytest.raises(core.RequestError, match="prefix"):
        parse(inpaint_body(source_audio_s3=f"s3://{BUCKET}/input/123/"))
    with pytest.raises(core.RequestError, match="s3://bucket/key"):  # file:// only with the env flag
        parse(inpaint_body(source_audio_s3="file:///etc/passwd.wav"))
    assert parse(inpaint_body(source_audio_s3="file:///tmp/x.wav"), allow_file_sources=True).source_audio_s3


def test_source_bucket_allow_list_is_exact_per_account():
    ok = [f"s3://{BUCKET}/input/1/a.wav", "s3://uaudio-training-artifacts-297416720456-us-east-1/ravi/a.wav"]
    bad = ["s3://uaudio-model-inference-evil/x.wav", "s3://ngmc-poc-uploads-dev-297416720456-evil/x.wav",
           "s3://uaudio-inference-297416720456-us-east-1/output/abc.out", "s3://ngmc-poc-uploads-dev-111111111111/x.wav"]
    for uri in ok:
        assert parse(inpaint_body(source_audio_s3=uri)).source_audio_s3 == uri
    for uri in bad:
        with pytest.raises(core.RequestError, match="not allowed"):
            parse(inpaint_body(source_audio_s3=uri))
    with pytest.raises(core.RequestError, match="not allowed"):  # an empty list allows nothing
        parse(inpaint_body(), allowed_buckets=[])
    assert core.source_buckets_for_account("545063352976")[0] == "ngmc-poc-uploads-*-545063352976"


def test_continue_fields():
    r = parse(inpaint_body(mode="continue", mask_start_s=None, mask_end_s=None))
    assert r.mask_start_s == () and r.duration_seconds == 15.0 and r.pre_s == 30.0
    r = parse(inpaint_body(mode="continue", mask_start_s=12, mask_end_s=None, duration_seconds=8))
    assert r.mask_start_s == (12.0,) and r.duration_seconds == 8.0
    r = parse(inpaint_body(mode="continue", mask_start_s=12, mask_end_s=20, duration_seconds=3))
    assert r.duration_seconds == 8.0 and any("mask_end_s sets" in w for w in r.warnings)
    with pytest.raises(core.RequestError, match="one mask_start_s"):
        parse(inpaint_body(mode="continue", mask_start_s=[1, 2], mask_end_s=None))
    with pytest.raises(core.RequestError, match="needs mask_start_s"):
        parse(inpaint_body(mode="continue", mask_start_s=None, mask_end_s=5))


# --------------------------------------------------------------------------- masks / splice


def test_plan_inpaint_regions_pad_clamp_sort_merge():
    n = 60 * SR
    rs = core.plan_inpaint_regions([30, 0.5, 20], [40, 9.5, 29.5], 1.0, n, SR)
    assert [(r.start_s, r.end_s) for r in rs] == [(0.0, 10.5), (19.0, 41.0)]  # 20-29.5 and 30-40 merged
    assert rs[1].requested == ((20.0, 29.5), (30.0, 40.0))
    assert rs[0].start_sample == 0 and rs[1].end_sample == 41 * SR
    r = core.plan_inpaint_regions([55], [70], 1.0, n, SR)[0]  # runs past the end: clamped
    assert (r.end_s, r.end_sample) == (60.0, n)
    with pytest.raises(core.RequestError, match="after the end"):
        core.plan_inpaint_regions([61], [65], 1.0, n, SR)
    # same rounding as the vendored seconds_to_samples (round, not int)
    r = core.plan_inpaint_regions([10.00006], [11], 0.0, n, SR)[0]
    assert r.start_sample == vmasks.seconds_to_samples(10.00006, SR)


def test_plan_continue_alignment_and_limits():
    n = 40 * SR
    m0, m1, c0 = core.plan_continue(None, 10.0, 30.0, n, SR, DS, 120 * SR)
    assert m0 == n and m1 == n + 10 * SR and (m0 - c0) % DS == 0 and m0 - c0 <= 30 * SR
    m0, m1, c0 = core.plan_continue(5.0, 10.0, 30.0, n, SR, DS, 120 * SR)
    assert c0 == m0 - (m0 // DS) * DS and m1 == m0 + 10 * SR
    with pytest.raises(core.RequestError, match="inside the source"):
        core.plan_continue(41.0, 10.0, 30.0, n, SR, DS, 120 * SR)
    with pytest.raises(core.RequestError, match="exceed"):
        core.plan_continue(None, 100.0, 30.0, n, SR, DS, 120 * SR)


def test_outside_mask_identical():
    rng = np.random.default_rng(1)
    src = rng.standard_normal((1000, 2)).astype(np.float32)
    out = src.copy()
    out[100:200] += 1
    out[500:600] += 1
    assert core.outside_mask_identical(out, src, [(500, 600), (100, 200)])
    assert not core.outside_mask_identical(out, src, [(100, 200)])
    out2 = out.copy()
    out2[200] += 1e-7
    assert not core.outside_mask_identical(out2, src, [(100, 200), (500, 600)])
    assert core.outside_mask_identical(np.concatenate([src[:300], out[:50]]), src, [(300, 1000)], keep_end=300)


def test_wav_roundtrip_is_exact_and_sidecar_uri():
    a = np.random.default_rng(2).uniform(-1, 1, (1234, 2)).astype(np.float32)
    back, sr = core.decode_audio_bytes(core.encode_wav(a, 44100))
    assert sr == 44100 and np.array_equal(back, a)
    assert core.sidecar_uri("s3://b-1/sidecars/", "i") == "s3://b-1/sidecars/i.json"
    assert core.sidecar_uri("s3://b-1/x/y", "i") == "s3://b-1/x/y/i.json"
    assert core.sidecar_uri("s3://b-1", "i") == "s3://b-1/i.json"


# --------------------------------------------------------------------------- handler with a fake model


class FakeSAM:
    """The attributes h.Engine reads, a generate() with upstream's output truncation
    (model.py:350-356) that records what it was given, and LoRA hooks."""

    def __init__(self):
        self.model = SimpleNamespace(sample_rate=SR, io_channels=64,
                                     pretransform=SimpleNamespace(downsampling_ratio=DS, io_channels=2))
        self.model_config = {"sample_size": 120 * SR}
        self.device = "cpu"
        self.calls: list[dict] = []
        self.strengths: list[float] = []

    def set_lora_strength(self, strength, lora_index=None):
        self.strengths.append(float(strength))

    def generate(self, **kw):
        self.calls.append(kw)
        n = int(kw["duration"] * SR)
        rng = np.random.default_rng(kw["seed"])
        return (0.2 * rng.standard_normal((1, 2, n))).astype(np.float32)


def write_adapter(path: Path, arm: str) -> Path:
    import struct

    md = {"lora_config": json.dumps({"rank": 16, "adapter_type": "dora-rows"}), "ravi_arm": arm}
    header = json.dumps({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": md}).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + np.float32(1).tobytes())
    return path


@pytest.fixture
def service(tmp_path):
    fake = FakeSAM()
    eng = vh.Engine(fake, "medium", None, to_tensor=lambda a: np.ascontiguousarray(a.T), to_numpy=np.asarray)
    eng.strength = None
    events: list[tuple] = []
    paths = {arm: write_adapter(tmp_path / "adapters" / f"{arm}.safetensors", arm) for arm in ADAPTERS}
    am = inf.AdapterManager(eng, inf.discover_adapters(tmp_path / "adapters"),
                            attach=lambda sam, p: events.append(("attach", p.stem)),
                            detach=lambda sam: events.append(("detach",)))
    assert sorted(am.paths) == sorted(paths)
    sources: dict[str, bytes] = {}
    sidecars: dict[str, dict] = {}
    cfg = json.loads(json.dumps(inf.CONFIG_DEFAULTS))
    svc = inf.Service(eng, am, cfg, fetch=lambda uri, max_bytes: sources[uri],
                      put_sidecar=lambda uri, payload: sidecars.__setitem__(uri, json.loads(json.dumps(payload, default=str))),
                      sidecar_prefix="s3://uaudio-inference-297416720456-us-east-1/sidecars/")
    return SimpleNamespace(svc=svc, fake=fake, events=events, sources=sources, sidecars=sidecars)


def put_source(service, name: str, audio: np.ndarray, sr: int, subtype: str = "FLOAT") -> str:
    import io

    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype=subtype, format="WAV")
    uri = f"s3://{BUCKET}/uploads/{name}.wav"
    service.sources[uri] = buf.getvalue()
    return uri


def run_handler(service, body: dict):
    inf._SERVICE = service.svc
    try:
        req = inf.input_fn(json.dumps(body).encode(), "application/json")
        pred = inf.predict_fn(req, service.svc)
        wav, ctype = inf.output_fn(pred, "application/json")
    finally:
        inf._SERVICE = None
    assert ctype == "audio/wav"
    audio, sr = sf.read(__import__("io").BytesIO(wav), dtype="float32", always_2d=True)
    assert sr == SR and audio.shape[1] == 2
    return audio, pred


def test_generate_through_the_handler(service):
    audio, pred = run_handler(service, {"prompt": "rvkr, piano", "duration_seconds": 3.3, "seed": 7,
                                        "inference_id": "g1"})
    assert audio.shape[0] == int(3.3 * SR)  # upstream truncation
    call = service.fake.calls[-1]
    assert (call["steps"], call["cfg_scale"], call["seed"], call["prompt"]) == (8, 1.0, 7, "rvkr, piano")
    assert "inpaint_audio" not in call
    sc = service.sidecars["s3://uaudio-inference-297416720456-us-east-1/sidecars/g1.json"]
    assert sc["request"]["mode"] == "generate" and sc["outside_mask_bit_identical"] is None
    assert sc["output"]["samples"] == audio.shape[0] and sc["adapter"] == {"id": "base_model"}
    assert set(sc["timings"]) >= {"adapter_s", "model_s", "encode_verify_s", "total_s"}


def test_inpaint_is_bit_identical_outside_padded_masks(service):
    rng = np.random.default_rng(3)
    src = (0.3 * rng.standard_normal((60 * SR, 2))).astype(np.float32)
    uri = put_source(service, "t1", src, SR)
    body = {"mode": "inpaint", "prompt": "rvkr, verse", "source_audio_s3": uri, "mask_start_s": [30, 10],
            "mask_end_s": [36, 18], "pad_s": 1.0, "seed": 1, "adapter": "B", "lora_strength": 0.5,
            "inference_id": "inp-1"}
    audio, pred = run_handler(service, body)
    assert audio.shape == src.shape
    spans = [(round(9 * SR), round(19 * SR)), (round(29 * SR), round(37 * SR))]  # computed independently
    keep = np.ones(src.shape[0], bool)
    for a, b in spans:
        keep[a:b] = False
        assert not np.array_equal(audio[a:b], src[a:b])
    assert np.array_equal(audio[keep], src[keep])
    # the model saw a context window around each mask, with the mask passed so that
    # upstream's int(sec * sr) lands exactly on the mask samples
    calls = service.fake.calls[-2:]
    for call, (a, b) in zip(calls, spans):
        sr_in, ctx = call["inpaint_audio"]
        assert sr_in == SR and ctx.shape[0] == 2
        n_ctx = ctx.shape[1]
        assert n_ctx == round(47 * SR) and call["duration"] == n_ctx / SR
        m0r = int(call["inpaint_mask_start_seconds"] * SR)
        m1r = int(call["inpaint_mask_end_seconds"] * SR)
        assert m1r - m0r == b - a
    sc = pred.sidecar
    assert sc["outside_mask_bit_identical"] is True and len(sc["masks"]) == 2
    assert [(m["start_sample"], m["end_sample"]) for m in sc["masks"]] == spans
    assert sc["adapter"]["id"] == "B" and sc["adapter"]["sha256"] and sc["lora_strength_effective"] == 0.5
    assert sc["adapter"]["lora_config"]["adapter_type"] == "dora-rows"
    assert sc["source"]["resampled"] is False and sc["crossfade_samples"] == round(0.05 * SR)
    assert service.events == [("attach", "B")] and service.fake.strengths[-1] == 0.5


def test_inpaint_resamples_other_rates_and_pcm16(service):
    rng = np.random.default_rng(4)
    src_sr = 11025
    src = (0.3 * rng.standard_normal((20 * src_sr, 1))).astype(np.float32)
    uri = put_source(service, "t2", src, src_sr, subtype="PCM_16")
    audio, pred = run_handler(service, {"mode": "inpaint", "prompt": "p", "source_audio_s3": uri,
                                        "mask_start_s": 8, "mask_end_s": 12, "seed": 0, "inference_id": "inp-2"})
    pcm = sf.read(__import__("io").BytesIO(service.sources[uri]), dtype="float32", always_2d=True)[0]
    expected, _ = vh.to_model_audio(pcm, src_sr, SR, 2)
    assert audio.shape == expected.shape
    a, b = round(7 * SR), round(13 * SR)
    assert np.array_equal(audio[:a], expected[:a]) and np.array_equal(audio[b:], expected[b:])
    assert pred.sidecar["source"]["resampled"] is True and pred.sidecar["source"]["input_channels"] == 1


def test_continue_append_and_mid_track(service):
    rng = np.random.default_rng(5)
    src = (0.3 * rng.standard_normal((40 * SR, 2))).astype(np.float32)
    uri = put_source(service, "t3", src, SR)
    audio, pred = run_handler(service, {"mode": "continue", "prompt": "p", "source_audio_s3": uri,
                                        "duration_seconds": 10, "seed": 2, "inference_id": "c1"})
    assert audio.shape[0] == src.shape[0] + 10 * SR and np.array_equal(audio[: src.shape[0]], src)
    assert pred.sidecar["crossfade_samples"] == 0  # nothing to fade from when appending
    call = service.fake.calls[-1]
    ctx = call["inpaint_audio"][1]
    m0r = int(call["inpaint_mask_start_seconds"] * SR)
    assert m0r % DS == 0 and not ctx[:, m0r:].any()  # nothing after the start is shown
    audio, pred = run_handler(service, {"mode": "continue", "prompt": "p", "source_audio_s3": uri,
                                        "mask_start_s": 20, "mask_end_s": 28, "seed": 2, "inference_id": "c2"})
    m0 = 20 * SR
    assert audio.shape[0] == 28 * SR and np.array_equal(audio[:m0], src[:m0])
    assert not np.array_equal(audio[m0:], src[m0:28 * SR])
    assert pred.sidecar["crossfade_samples"] == round(0.05 * SR)
    assert pred.sidecar["masks"][0]["source_after_start_dropped_s"] == 12.0


def test_adapter_switching_is_lazy_and_base_uses_strength_zero(service):
    src = np.zeros((10 * SR, 2), np.float32)
    uri = put_source(service, "t4", src, SR)

    def go(adapter, strength=1.0):
        run_handler(service, {"prompt": "p", "duration_seconds": 1, "adapter": adapter, "lora_strength": strength,
                              "seed": 0})

    go("base_model")
    assert service.events == [] and service.fake.strengths == []
    go("A", 0.7)
    go("A", 1.2)
    go("base_model")
    go("E")
    assert service.events == [("attach", "A"), ("detach",), ("attach", "E")]
    assert service.fake.strengths == [0.7, 1.2, 0.0, 1.0]
    assert uri  # source unused


def test_predict_refuses_unknown_adapter_and_bad_masks(service):
    with pytest.raises(core.RequestError, match="unknown adapter"):
        service.svc.run(core.parse_request({"prompt": "p", "adapter": "Q"}, None))
    src = np.zeros((10 * SR, 2), np.float32)
    uri = put_source(service, "t5", src, SR)
    inf._SERVICE = service.svc
    try:
        with pytest.raises(core.RequestError, match="after the end"):
            inf.predict_fn(inf.input_fn(json.dumps({"mode": "inpaint", "prompt": "p", "source_audio_s3": uri,
                                                    "mask_start_s": 11, "mask_end_s": 12}), "application/json"),
                           service.svc)
    finally:
        inf._SERVICE = None


def test_sidecar_failure_does_not_fail_the_request(service):
    def boom(uri, payload):
        raise RuntimeError("no s3")

    service.svc.put_sidecar = boom
    audio, pred = run_handler(service, {"prompt": "p", "duration_seconds": 1, "seed": 0})
    assert pred.sidecar_uri is None and audio.shape[0] == SR


def test_local_model_config_points_t5gemma_at_the_local_copy(tmp_path):
    base = tmp_path / "medium"
    (base / inf.T5GEMMA_DIR).mkdir(parents=True)
    (base / inf.T5GEMMA_DIR / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"")
    cfg = {"model": {"conditioning": {"configs": [
        {"id": "prompt", "type": "t5gemma",
         "config": {"max_length": 256, "repo_id": "stabilityai/stable-audio-3-medium", "subfolder": inf.T5GEMMA_DIR}},
        {"id": "seconds_total", "type": "number", "config": {}}]}}}
    (base / "model_config.json").write_text(json.dumps(cfg))
    assert inf.base_dir_complete(base)
    out = inf.local_model_config(base)
    t5 = out["model"]["conditioning"]["configs"][0]["config"]
    assert t5 == {"max_length": 256, "model_path": str(base / inf.T5GEMMA_DIR)}
    (base / inf.T5GEMMA_DIR / "config.json").unlink()
    assert not inf.base_dir_complete(base)


def test_failed_attach_detaches_and_leaves_the_base(service):
    calls = []

    def bad_attach(sam, p):
        calls.append(("attach", p.stem))
        raise RuntimeError("shape mismatch")

    am = service.svc.adapters
    am._attach, am._detach = bad_attach, (lambda sam: calls.append(("detach",)))
    with pytest.raises(RuntimeError, match="attaching adapter B failed"):
        run_handler(service, {"prompt": "p", "duration_seconds": 1, "adapter": "B", "seed": 0})
    assert calls == [("attach", "B"), ("detach",)] and am.active is None and service.svc.engine.lora_path is None
    run_handler(service, {"prompt": "p", "duration_seconds": 1, "seed": 0})  # base still serves
    assert service.fake.strengths == []  # no strength-0 call: nothing is attached


def test_bad_masks_are_refused_before_the_adapter_switch_and_gpu(service):
    src = np.zeros((60 * SR, 2), np.float32)
    uri = put_source(service, "t6", src, SR)
    service.svc.engine.sample_size = 30 * SR  # model window 30 - 6 s padding = 24 s
    n_calls = len(service.fake.calls)
    with pytest.raises(core.RequestError, match="does not fit"):  # 2nd mask (29.9 s) longer than the window
        run_handler(service, {"mode": "inpaint", "prompt": "p", "source_audio_s3": uri, "mask_start_s": [1, 30],
                              "mask_end_s": [5, 59.9], "pad_s": 0, "adapter": "A"})
    assert service.events == [] and len(service.fake.calls) == n_calls  # nothing attached, nothing generated


def test_source_guards(service, tmp_path, monkeypatch):
    long = np.zeros((int(1.5 * SR), 2), np.float32)
    uri = put_source(service, "t7", long, SR)
    service.svc.config["max_source_s"] = 1.0
    with pytest.raises(core.RequestError, match="the limit is 1 s"):  # checked from the header, before decoding
        run_handler(service, {"mode": "continue", "prompt": "p", "source_audio_s3": uri})
    service.svc.config["max_source_s"] = 600.0
    nan = np.zeros((2 * SR, 2), np.float32)
    nan[5] = np.nan
    uri = put_source(service, "t8", nan, SR)
    with pytest.raises(core.RequestError, match="NaN"):
        run_handler(service, {"mode": "continue", "prompt": "p", "source_audio_s3": uri})
    monkeypatch.setenv("SA3_ALLOW_FILE_SOURCES", "1")
    with pytest.raises(core.RequestError, match="does not exist"):
        inf.fetch_source(f"file://{tmp_path}/missing.wav", 10)
    monkeypatch.delenv("SA3_ALLOW_FILE_SOURCES")


def test_verify_manifest(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"1234")
    assert inf.verify_manifest(tmp_path) is None  # no manifest: warning only
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"repo_id": "r", "files": {"a.bin": {"bytes": 4}}}))
    assert inf.verify_manifest(tmp_path)["files"] == 1
    (tmp_path / "a.bin").write_bytes(b"12")
    with pytest.raises(RuntimeError, match="do not match"):
        inf.verify_manifest(tmp_path)
