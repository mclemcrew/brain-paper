r"""Auto-caption the corpus with an audio-language model (plan Phase 1 step 6).

Usage (inside the ``docker/captioner`` image; see its README)::

    python -m ravi.captions.autocaption \
        --manifest manifests/tracks.parquet --prepared-root /data/ravi-corpus/prepared \
        --sr 44100 --out captions/midasheng.jsonl \
        [--model midasheng|qwen-omni] [--limit N] [--track-id ID ...] \
        [--max-words 15] [--batch-size 1] [--resume]
        [--audio-dir DIR] [--splits manifests/splits.json --split train,val|test]

    # train/val: caption the exact windowed audio make_arms writes; test: the full track
    python -m ravi.captions.autocaption --audio-dir data/arms/A/{split} --split train,val \
        --sr 44100 --out captions/midasheng.jsonl --resume
    python -m ravi.captions.autocaption --prepared-root /data/ravi-corpus/prepared --split test \
        --sr 44100 --out captions/midasheng_test.jsonl --resume

    python -m ravi.captions.autocaption --prefetch [--model qwen-omni]   # weights -> HF_HOME

Models
------
``midasheng`` (default): Xiaomi MiDashengLM-7B, HF ``mispeech/midashenglm-7b-1021-bf16``
(Apache-2.0), pinned to revision :data:`MIDASHENG_REVISION`. ``qwen-omni`` (fallback):
``Qwen/Qwen2.5-Omni-7B`` (Apache-2.0 per its LICENSE file), pinned to :data:`QWEN_REVISION`.
Both run under ``transformers==4.52.4`` (the ``transformers_version`` in MiDashengLM's
``config.json`` and ``generation_config.json``).

Upstream API used (verified 2026-10-14 against the files at the pinned revisions)
-------------------------------------------------------------------------------
MiDashengLM, model card https://huggingface.co/mispeech/midashenglm-7b-1021-bf16 and
README https://github.com/xiaomi-research/dasheng-lm::

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": user_prompt},
        {"type": "audio", "path": "/path/to/example.wav"},
        # or "url": "https://example.com/example.wav"
        # or "audio": np.random.randn(16000)
    ]}]
    with torch.no_grad():
        model_inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            add_special_tokens=True, return_dict=True,
        ).to(device=model.device, dtype=model.dtype)
        generation = model.generate(**model_inputs)
        output = tokenizer.batch_decode(generation, skip_special_tokens=True)

* ``trust_remote_code=True`` is required: ``config.json`` ``auto_map`` points
  ``AutoModelForCausalLM`` at ``modeling_midashenglm.MiDashengLMModel`` and
  ``preprocessor_config.json`` points ``AutoProcessor`` at
  ``processing_midashenglm.MiDashengLMProcessor``; neither is in transformers.
* The remote code imports ``Qwen2_5OmniTextConfig`` / ``Qwen2_5OmniThinkerTextModel``
  (absent in 4.51.3, present in 4.52.4), ``transformers.utils.can_return_tuple`` and
  ``processing_utils.AllKwargsForChatTemplate``; and ``torchaudio.functional`` for its
  mel front end. transformers' ``load_audio`` (used by ``apply_chat_template`` even for
  an ndarray) calls ``requires_backends(load_audio, ["librosa"])``, so librosa is needed.
* We pass the audio as a 1-D float32 ndarray under ``"audio"`` (the third documented form),
  resampled here, so transformers never decodes files itself. The processor raises if a
  ``sampling_rate`` other than its own is passed.
* ``MiDashengLMModel.generate`` builds ``inputs_embeds`` and calls
  ``self.decoder.generate(inputs_embeds=...)`` without ``input_ids``, so the output holds
  only new tokens (the card decodes it whole). :func:`strip_prompt_tokens` handles both
  cases anyway.
* Deviation from the card: we load with ``torch_dtype=torch.bfloat16``. Without it,
  transformers 4.52 materialises the bf16 checkpoint in float32 (~2x memory).
* Chat template (``chat_template.jinja``): with no system turn it inserts the Qwen
  system prompt; the audio item renders as ``<|audio_bos|><|AUDIO|><|audio_eos|>``.
  The processor warns if a different system prompt is supplied, so we supply none.

Qwen2.5-Omni-7B, model card https://huggingface.co/Qwen/Qwen2.5-Omni-7B::

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
    text = processor.batch_decode(text_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)

* ``model.disable_talker()`` drops the speech head (~2 GB); batch inference needs
  ``return_audio=False``. No ``qwen_omni_utils`` is needed: we pass the resampled arrays.
* In transformers 4.52.4 ``Qwen2_5OmniForConditionalGeneration.generate`` routes
  ``thinker_*`` kwargs to the thinker and sets ``max_new_tokens=thinker_max_new_tokens``
  (default 1024) itself, so a plain ``max_new_tokens`` is ignored: we pass
  ``thinker_max_new_tokens`` / ``thinker_do_sample``. The thinker is called with
  ``input_ids``, so its output includes the prompt and is sliced off.

Audio input constraints
-----------------------
* Both models: 16 kHz mono float. MiDashengLM ``preprocessor_config.json``:
  ``Wav2Vec2FeatureExtractor``, ``sampling_rate: 16000``, ``feature_size: 1``,
  ``do_normalize: false`` (the processor asserts it); the encoder config has
  ``sample_rate: 16000``. Qwen2.5-Omni ``preprocessor_config.json``:
  ``WhisperFeatureExtractor``, ``sampling_rate: 16000``.
* We read ``{prepared_root}/{sr}/{track_id}.wav`` (stereo, -16 LUFS, <= -1 dBTP, see
  ``ravi.data.prepare_audio``), or with ``--audio-dir DIR`` ``{DIR}/{track_id}.wav`` (e.g.
  ``make_arms``' 46.997 s windowed WAVs; ``{split}`` in DIR becomes the track's split; the
  ``.wav.json`` / ``.window.json`` / ``.txt`` siblings are ignored; the WAV rate must still
  equal ``--sr``), average the channels and resample with
  ``scipy.signal.resample_poly`` (44100 -> 16000 is up 160 / down 441). No gain change:
  MiDashengLM does no normalisation and the prepared level is well inside full scale.
* MiDashengLM has no hard maximum length. Its Dasheng encoder splits the patch sequence
  into windows of ``target_length // 4`` = 252 patches = 1008 mel frames x 160 hop =
  10.08 s, encodes each window independently and concatenates
  (``DashengAudioTransformer.forward``); the LM sees ~5 audio tokens per second
  (``calculate_mel_frames_dasheng``: 100 frames/s // 4 // 5). Upstream benchmarks use 30 s
  inputs, so clips are passed whole by default and their length is logged; the corpus
  ranges 2.3-184 s (median 32 s). ``--max-audio-seconds`` crops if the pilot shows that
  long clips caption worse.
* Qwen2.5-Omni: ``chunk_length: 300`` / ``n_samples: 4800000`` and the processor forces
  ``padding="max_length"``, so audio beyond 300 s would be cut silently; we crop at 300 s
  ourselves and log it (no corpus clip is that long).

Decoding and post-processing
----------------------------
Prompt (:data:`PROMPTS`, v2, stated positively; v1's "Instrumental, no lyrics, no BPM
numbers." clause was parroted verbatim on 2 of 4 pilot clips)::

    primary: Describe this piece of instrumental music in one sentence of at most {n} words,
             covering its genre, mood, main instruments and how fast it feels.
    alt:     What does this instrumental track sound like? Answer in one sentence of at most
             {n} words, naming its genre, mood, main instruments and how fast it feels.

Every track gets ``primary``. If the post-processed answer is an echo (:func:`is_echo`: a
contiguous run of a prompt's words, >= 60% of its content words taken from a prompt, or fewer
than 4 content words; checked against both wordings and v1) the track is asked once more with
``alt``; a second echo is a failure (stage ``echo``, both raws logged), never a caption.

Greedy (``do_sample=False``, ``num_beams=1``), ``max_new_tokens=48``. The raw text is
kept in ``raw``; ``caption`` is :func:`postprocess` of it: first non-empty line, leading
``Caption:``-style labels and surrounding quotes removed, ``<digits> bpm`` phrases
stripped (logged), trailing period removed, cut to ``--max-words`` at the last clause
boundary that keeps >= 8 words (half the cap if it is below 8): a comma/semicolon/colon/dash
if there is one in range, else before and/with/but/while/featuring/...; a hard cut with
dangling function words dropped only if neither exists,
and the first letter lowercased (unless the first word is an all-caps acronym like EDM).

Output
------
``--out`` is JSONL, one record per captioned track, appended and fsynced line by line::

    {"track_id", "caption", "raw", "model", "n_words", "audio_seconds", "elapsed_s",
     "audio_source", "audio_dir", "split",
     "prompt_variant", "echo_retry", "echo_reason", "raw_primary"}

``model`` is the HF repo id, ``audio_seconds`` the length fed to the model (after any
crop), ``audio_source`` ``prepared`` or ``audio_dir``, ``audio_dir`` the directory the WAV
was read from, ``split`` the track's split when ``--split`` is given, ``elapsed_s`` the
wall time of load + inference + post-processing (a batch's time divided by its size; model
loading excluded). Failures are appended to
``{out_dir}/_failures.jsonl`` (``--failures``) with stage and error; they are retried on
the next ``--resume``. ``prompt_variant`` is ``primary`` or ``alt``; ``echo_retry`` says the
primary answer was an echo (``echo_reason``, its text in ``raw_primary``).
``{out}.meta.json`` records model id, revision, both prompt wordings and their
``prompt_sha`` (compared on ``--resume``; records without ``prompt_variant`` are refused),
decoding settings, ``audio_source`` / ``audio_dir`` (as given, ``{split}`` unexpanded), the
splits captioned so far and ``audio_seconds`` per track; ``--resume`` refuses to continue a
file made with different settings or a different audio source (records are checked too), so
windowed and full-track captions can never share a file. Captioning train then val into one
file with the same ``--audio-dir .../{split}`` template is allowed.
An existing non-empty ``--out`` without ``--resume`` is an error. On ``--resume`` a torn
last line (crash mid-write) is cut off before appending.

Selection: manifest order, then ``--split`` (tracks of those splits in ``--splits``; split ids
missing from the manifest are an error), then ``--track-id`` (subset, in the given order), then
``--limit N`` (first N of that), then ``--resume`` skips ids already in ``--out``. So
``--limit 3`` is always the same three tracks and re-running it with ``--resume`` is a
no-op.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

log = logging.getLogger("ravi.captions.autocaption")

#: Prompt wordings. v1 ("Caption this instrumental music in at most {n} words: genre, mood,
#: main instruments, tempo feel. Instrumental, no lyrics, no BPM numbers. One line.") made
#: MiDashengLM parrot its negated constraint clause on 2 of 4 pilot clips, so v2 states the task
#: positively. "alt" is used once for a track whose "primary" answer is an echo (:func:`is_echo`).
PROMPTS = {
    "primary": (
        "Describe this piece of instrumental music in one sentence of at most {n} words, "
        "covering its genre, mood, main instruments and how fast it feels."
    ),
    "alt": (
        "What does this instrumental track sound like? Answer in one sentence of at most {n} words, "
        "naming its genre, mood, main instruments and how fast it feels."
    ),
}
PROMPT_VERSION = 2
#: Earlier prompts: an answer that parrots one of these is an echo too.
LEGACY_PROMPTS = (
    "Caption this instrumental music in at most {n} words: genre, mood, main instruments, "
    "tempo feel. Instrumental, no lyrics, no BPM numbers. One line.",
)
PROMPT_TEMPLATE = PROMPTS["primary"]
DEFAULT_MAX_WORDS = 15
MAX_NEW_TOKENS = 48

MIDASHENG_ID = "mispeech/midashenglm-7b-1021-bf16"
#: HF commit of MIDASHENG_ID checked on 2026-10-14 (lastModified 2026-03-17).
MIDASHENG_REVISION = "f7b0fdb745fc7997a62443007191b50f1c94702b"
QWEN_ID = "Qwen/Qwen2.5-Omni-7B"
#: HF commit of QWEN_ID checked on 2026-10-14 (lastModified 2025-04-30).
QWEN_REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
#: Default system prompt of both chat templates (Qwen2.5-Omni's card asks for it verbatim).
QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech."
)


def build_prompt(max_words: int = DEFAULT_MAX_WORDS, variant: str = "primary") -> str:
    if max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}")
    return PROMPTS[variant].format(n=max_words)


def prompt_sha(max_words: int) -> str:
    """Hash of every prompt wording a run may use (+ PROMPT_VERSION); --resume compares it."""
    import hashlib

    blob = json.dumps({"version": PROMPT_VERSION, "prompts": {k: build_prompt(max_words, k) for k in PROMPTS}},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- echo detection

#: Not content: function words plus generic nouns every caption or prompt may contain.
_NON_CONTENT = frozenset(
    "a an the and or but nor of in on at to for by from with into over as is are be it its this that "
    "these those no not one some very most at least than how what does do sound sounds like "
    "music musical track piece song tune audio clip recording instrumental".split()
)
ECHO_OVERLAP = 0.6
ECHO_MIN_CONTENT_WORDS = 4


def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def is_echo(caption: str, prompts: Sequence[str]) -> str | None:
    """Why ``caption`` is a prompt echo (or too thin to use), else None.

    * ``empty``;
    * ``substring``: its normalised words are a contiguous run of a prompt's words;
    * ``overlap``: >= 60% of its content words occur in one prompt;
    * ``few_content_words``: fewer than 4 content words (``_NON_CONTENT`` excluded).
    """
    words = _norm_words(caption)
    if not words:
        return "empty"
    cap = " " + " ".join(words) + " "
    content = [w for w in words if w not in _NON_CONTENT]
    for p in prompts:
        pw = _norm_words(p)
        if cap in " " + " ".join(pw) + " ":
            return "substring"
        pset = set(pw)
        if content and sum(w in pset for w in content) / len(content) >= ECHO_OVERLAP:
            return "overlap"
    if len(content) < ECHO_MIN_CONTENT_WORDS:
        return "few_content_words"
    return None


def echo_prompts(max_words: int) -> list[str]:
    return [build_prompt(max_words, k) for k in PROMPTS] + [p.format(n=max_words) for p in LEGACY_PROMPTS]


# --------------------------------------------------------------------------- post-processing

_QUOTES = "\"'`“”‘’«»"
_LABEL_RE = re.compile(r"^\s*(?:[-*•]\s+)?(?:\*\*)?(?:caption|description|answer)(?:\*\*)?\s*[:\-–—]\s*", re.I)
_NUM = r"\d+(?:[.,]\d+)?"
_BPM_RE = re.compile(
    r"(?:\b(?:at|around|about|approximately|approx\.?|roughly|near|of|a|an)\s+){0,2}"
    r"(?:[~≈]\s*)?"
    rf"{_NUM}(?:\s*(?:-|–|to)\s*{_NUM})?"
    r"\s*-?\s*(?:bpm|b\.p\.m\.?|beats\s+per\s+minute)\b",
    re.I,
)
#: Word-cap cuts keep at least this many words when a boundary allows it (else a hard cut).
MIN_CLAUSE_WORDS = 8
#: A clause may start before these words (second choice after punctuation boundaries).
_CLAUSE_STARTERS = frozenset(
    "and with but while featuring over plus as then into against under alongside "
    "backed driven layered evoking creating building giving".split()
)
#: Words that must not end a caption after a hard cut.
_DANGLING = frozenset(
    "a an the and or with of in on over to for by from into as at but while featuring plus "
    "its their some very".split()
)
_DASHES = {"-", "–", "—", "--"}


@dataclass
class PostResult:
    caption: str
    n_words: int
    n_words_before_cap: int
    bpm_stripped: list[str] = field(default_factory=list)
    truncated: bool = False


def _strip_quotes(s: str) -> str:
    prev = None
    while s != prev:
        prev = s
        s = s.strip().strip(_QUOTES).strip()
    return s


def _tidy(s: str) -> str:
    s = re.sub(r"\(\s*[,;:]?\s*\)|\[\s*\]", " ", s)  # parentheses left empty by BPM removal
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,;:.!?)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)  # ", ," -> ","
    s = re.sub(r"\b(a|an)\s+([,;:]|$)", r"\2", s)  # "a 120 bpm, groove" leftovers
    return s.strip().strip(",;:").strip()


def _punct_boundary(words: Sequence[str], i: int) -> bool:
    """A cut after ``words[:i]`` falls on , ; : or a dash (needs ``i < len(words)``)."""
    last, nxt = words[i - 1], words[i]
    return last[-1] in ",;:" or last[-1] in "–—" or nxt in _DASHES


def _word_boundary(words: Sequence[str], i: int) -> bool:
    """A cut after ``words[:i]`` falls before 'and' / 'with' / ... (needs ``i < len(words)``)."""
    return words[i].lower().strip(",;:") in _CLAUSE_STARTERS


def truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    """Cap ``text`` at ``max_words`` words, cutting at the last clause boundary that keeps
    >= ``MIN_CLAUSE_WORDS`` (8; half the cap if the cap is below 8): a comma/semicolon/colon/dash
    if there is one, else before and/with/... A hard cut (dangling function words dropped) only
    when no boundary exists in that range."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    min_keep = MIN_CLAUSE_WORDS if max_words >= MIN_CLAUSE_WORDS else max(1, math.ceil(max_words / 2))
    span = range(max_words, min_keep - 1, -1)
    # Punctuation first: "calm and peaceful" must not be split at "and" when a comma
    # earlier in the window ends a clause.
    cut = next((i for i in span if _punct_boundary(words, i)), None)
    if cut is None:
        cut = next((i for i in span if _word_boundary(words, i)), None)
    if cut is None:
        cut = max_words
        while cut > 1 and words[cut - 1].lower().strip(",;:") in _DANGLING:
            cut -= 1
    kept = list(words[:cut])
    while len(kept) > 1 and kept[-1] in _DASHES:
        kept.pop()
    return " ".join(kept).rstrip(",;:-–— ").strip(), True


def _lower_first(s: str) -> str:
    if not s or re.match(r"[A-Z0-9]{2,}\b", s):  # keep EDM, UK, 80s-style leading tokens
        return s
    return s[0].lower() + s[1:]


def postprocess(raw: str, max_words: int = DEFAULT_MAX_WORDS) -> PostResult:
    """Turn raw model output into a one-line caption of at most ``max_words`` words."""
    lines = [ln for ln in (raw or "").splitlines() if ln.strip()]
    s = lines[0] if lines else ""
    s = _LABEL_RE.sub("", s)
    s = _strip_quotes(s)
    bpm = [m.group(0).strip() for m in _BPM_RE.finditer(s)]
    if bpm:
        s = _BPM_RE.sub(" ", s)
    s = _tidy(s)
    s = _strip_quotes(s).rstrip(".!… ").strip()
    n_before = len(s.split())
    s, truncated = truncate_words(s, max_words)
    s = _strip_quotes(s).rstrip(".!…,;: ").strip()
    s = _lower_first(s)
    return PostResult(s, len(s.split()), n_before, bpm, truncated)


# --------------------------------------------------------------------------- audio


def resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """(n,) or (n, ch) float -> (m,) float32 mono at ``sr_out`` (channel mean, polyphase)."""
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    elif x.ndim != 1:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {x.shape}")
    if sr_in != sr_out:
        g = math.gcd(sr_in, sr_out)
        x = resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


@dataclass
class LoadedAudio:
    track_id: str
    audio: np.ndarray
    source_seconds: float
    audio_seconds: float


def load_track_audio(
    path: Path, expect_sr: int, target_sr: int, max_seconds: float | None
) -> tuple[np.ndarray, float, float]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != expect_sr:
        raise ValueError(f"{path}: sample rate {sr} != --sr {expect_sr}")
    if x.shape[0] == 0:
        raise ValueError(f"{path}: empty file")
    src_s = x.shape[0] / sr
    if max_seconds is not None and src_s > max_seconds:
        x = x[: int(round(max_seconds * sr))]
    y = resample_mono(x, sr, target_sr)
    return y, src_s, y.shape[0] / target_sr


# --------------------------------------------------------------------------- backends


class Backend(Protocol):
    name: str
    hf_id: str
    revision: str
    sample_rate: int
    max_audio_s: float | None

    def generate(self, audios: list[np.ndarray], prompt: str) -> list[str]: ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    revision: str
    sample_rate: int
    max_audio_s: float | None


SPECS: dict[str, ModelSpec] = {
    "midasheng": ModelSpec("midasheng", MIDASHENG_ID, MIDASHENG_REVISION, 16000, None),
    "qwen-omni": ModelSpec("qwen-omni", QWEN_ID, QWEN_REVISION, 16000, 300.0),
}


def strip_prompt_tokens(generated, input_ids):
    """Drop the echoed prompt if ``generate`` returned prompt + new tokens (works for torch/np)."""
    n = input_ids.shape[1]
    if generated.shape[1] > n and bool((generated[:, :n] == input_ids).all()):
        return generated[:, n:]
    return generated


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MiDashengBackend:
    def __init__(self, spec: ModelSpec, revision: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.name, self.hf_id, self.revision = spec.name, spec.hf_id, revision
        self.sample_rate, self.max_audio_s = spec.sample_rate, spec.max_audio_s
        self._torch = torch
        kw = dict(revision=revision, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.bfloat16, **kw)
            .to(_device(device))
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **kw)
        self.processor = AutoProcessor.from_pretrained(spec.hf_id, **kw)
        fe_sr = int(self.processor.feature_extractor.sampling_rate)
        if fe_sr != self.sample_rate:
            raise RuntimeError(f"processor expects {fe_sr} Hz, this script resamples to {self.sample_rate}")

    def generate(self, audios: list[np.ndarray], prompt: str) -> list[str]:
        convs = [
            [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio": a}]}]
            for a in audios
        ]
        with self._torch.no_grad():
            inputs = self.processor.apply_chat_template(
                convs if len(convs) > 1 else convs[0],
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=True,
                return_dict=True,
            ).to(device=self.model.device, dtype=self.model.dtype)
            gen = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1
            )
        gen = strip_prompt_tokens(gen, inputs["input_ids"])
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class QwenOmniBackend:
    def __init__(self, spec: ModelSpec, revision: str, device: str | None = None):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.name, self.hf_id, self.revision = spec.name, spec.hf_id, revision
        self.sample_rate, self.max_audio_s = spec.sample_rate, spec.max_audio_s
        self._torch = torch
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            spec.hf_id, revision=revision, torch_dtype=torch.bfloat16, device_map=_device(device)
        ).eval()
        self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(spec.hf_id, revision=revision)
        fe_sr = int(self.processor.feature_extractor.sampling_rate)
        if fe_sr != self.sample_rate:
            raise RuntimeError(f"processor expects {fe_sr} Hz, this script resamples to {self.sample_rate}")

    def generate(self, audios: list[np.ndarray], prompt: str) -> list[str]:
        convs = [
            [
                {"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": prompt}]},
            ]
            for _ in audios
        ]
        text = self.processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
        with self._torch.no_grad():
            inputs = self.processor(
                text=text, audio=list(audios), return_tensors="pt", padding=True, use_audio_in_video=False
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            gen = self.model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                thinker_max_new_tokens=MAX_NEW_TOKENS,
                thinker_do_sample=False,
                thinker_num_beams=1,
            )
        gen = strip_prompt_tokens(gen, inputs["input_ids"])
        return self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)


_BACKENDS: dict[str, Callable[..., Backend]] = {"midasheng": MiDashengBackend, "qwen-omni": QwenOmniBackend}


def load_backend(name: str, revision: str | None = None, device: str | None = None) -> Backend:
    spec = SPECS[name]
    rev = revision or spec.revision
    log.info("loading %s (%s @ %s)", name, spec.hf_id, rev)
    t0 = time.perf_counter()
    be = _BACKENDS[name](spec, rev, device)
    log.info("loaded in %.1f s", time.perf_counter() - t0)
    return be


def prefetch(name: str, revision: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    spec = SPECS[name]
    path = snapshot_download(spec.hf_id, revision=revision or spec.revision)
    log.info("%s @ %s -> %s", spec.hf_id, revision or spec.revision, path)
    return path


# --------------------------------------------------------------------------- JSONL bookkeeping


def repair_torn_tail(path: Path) -> bool:
    """Cut a trailing partial line (no final newline) left by a crash. Returns True if cut."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return False
        f.seek(0)
        data = f.read()
        keep = data.rfind(b"\n") + 1
        f.truncate(keep)
    log.warning("%s: removed torn last line (%d bytes)", path, len(data) - keep)
    return True


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not valid JSON ({e}); fix or remove the line") from e
    return out


def done_ids(path: Path) -> set[str]:
    return {r["track_id"] for r in read_jsonl(path)}


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


#: Keys of the meta sidecar that must match for --resume to continue a file.
META_KEYS = (
    "model", "revision", "prompt_sha", "max_words", "max_new_tokens", "max_audio_seconds", "sr",
    "audio_source", "audio_dir",
)
#: Values assumed for keys missing from a meta file written before they existed.
META_DEFAULTS = {"audio_source": "prepared", "audio_dir": None}


def run_meta(
    backend: Backend, prompt: str, max_words: int, sr: int, max_audio_s: float | None,
    audio_source: str = "prepared", audio_dir: str | None = None, splits: list[str] | None = None,
) -> dict:
    meta = {
        "model": backend.hf_id,
        "model_key": backend.name,
        "revision": backend.revision,
        "prompt": prompt,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": prompt_sha(max_words),
        "prompts": {k: build_prompt(max_words, k) for k in PROMPTS},
        "echo_rule": {"overlap": ECHO_OVERLAP, "min_content_words": ECHO_MIN_CONTENT_WORDS, "retries": 1},
        "max_words": max_words,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "model_sample_rate": backend.sample_rate,
        "max_audio_seconds": max_audio_s,
        "sr": sr,
        "audio_source": audio_source,
        "audio_dir": audio_dir,
        "splits": splits,
    }
    try:
        import importlib.metadata as md

        meta["transformers_version"] = md.version("transformers")
    except Exception:  # not installed (tests)
        meta["transformers_version"] = None
    meta["ravi_git_sha"] = os.environ.get("RAVI_GIT_SHA")
    return meta


def write_meta(meta_path: Path, meta: dict, out: Path) -> None:
    """Write the sidecar atomically, with the per-track ``audio_seconds`` of every record in ``out``."""
    meta = dict(meta)
    meta["audio_seconds"] = {r["track_id"]: r.get("audio_seconds") for r in read_jsonl(out)}
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, meta_path)


# --------------------------------------------------------------------------- driver


SPLIT_NAMES = ("train", "val", "test")


def load_splits(path: Path, names: Sequence[str]) -> dict[str, str]:
    """``{track_id: split}`` for the requested splits of a ``ravi.data.splits`` JSON."""
    bad = [n for n in names if n not in SPLIT_NAMES]
    if bad or not names:
        raise ValueError(f"--split must be a comma list of {SPLIT_NAMES}, got {list(names)}")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for n in names:
        if n not in doc:
            raise KeyError(f"{path} has no split {n!r}")
        for t in doc[n]:
            if t in out and out[t] != n:
                raise ValueError(f"{path}: track {t!r} is in both {out[t]!r} and {n!r}")
            out[t] = n
    return out


def select_tracks(
    manifest_ids: Sequence[str], track_ids: Sequence[str] | None, limit: int | None,
    split_of: dict[str, str] | None = None,
) -> list[str]:
    ids = list(dict.fromkeys(manifest_ids))
    if split_of is not None:
        known = set(ids)
        unknown = [t for t in split_of if t not in known]
        if unknown:
            raise KeyError(f"{len(unknown)} split track_ids not in manifest, e.g. {unknown[:3]}")
        ids = [t for t in ids if t in split_of]
    if track_ids:
        known = set(ids)
        unknown = [t for t in track_ids if t not in known]
        if unknown:
            where = "the selected splits" if split_of is not None else "manifest"
            raise KeyError(f"--track-id not in {where}: {unknown}")
        ids = list(dict.fromkeys(track_ids))
    if limit is not None:
        ids = ids[:limit]
    return ids


def _batches(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@dataclass
class RunConfig:
    manifest: Path
    prepared_root: Path | None
    sr: int
    out: Path
    failures: Path | None = None
    model: str = "midasheng"
    revision: str | None = None
    limit: int | None = None
    track_ids: list[str] | None = None
    max_words: int = DEFAULT_MAX_WORDS
    batch_size: int = 1
    resume: bool = False
    max_audio_seconds: float | None = None
    device: str | None = None
    #: Read ``{audio_dir}/{track_id}.wav`` instead of ``{prepared_root}/{sr}/...``. ``{split}``
    #: in it is replaced by the track's split (needs ``split``), e.g. ``data/arms/A/{split}``.
    audio_dir: str | Path | None = None
    splits_path: Path = Path("manifests/splits.json")
    split: list[str] | None = None


def run(cfg: RunConfig, backend: Backend | None = None) -> dict:
    """Caption the selected tracks. ``backend`` overrides model loading (tests)."""
    import pandas as pd

    if cfg.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if cfg.limit is not None and cfg.limit < 0:
        raise ValueError("--limit must be >= 0")
    prompt = build_prompt(cfg.max_words, "primary")
    alt_prompt = build_prompt(cfg.max_words, "alt")
    echo_ps = echo_prompts(cfg.max_words)
    out = cfg.out
    failures = cfg.failures or out.parent / "_failures.jsonl"
    meta_path = out.with_name(out.name + ".meta.json")

    if out.exists() and out.stat().st_size > 0 and not cfg.resume:
        raise FileExistsError(f"{out} exists; pass --resume to continue it or remove it")

    audio_source = "audio_dir" if cfg.audio_dir is not None else "prepared"
    audio_dir = str(cfg.audio_dir) if cfg.audio_dir is not None else None
    if audio_dir is not None and "{split}" in audio_dir and not cfg.split:
        raise ValueError("--audio-dir contains {split} but no --split was given")
    if audio_dir is None and cfg.prepared_root is None:
        raise ValueError("need --prepared-root or --audio-dir")
    split_of = load_splits(cfg.splits_path, cfg.split) if cfg.split else None

    manifest = pd.read_parquet(cfg.manifest, columns=["track_id"])
    ids = select_tracks(manifest["track_id"].tolist(), cfg.track_ids, cfg.limit, split_of)

    def input_dir(tid: str) -> Path:
        if audio_dir is None:
            return Path(cfg.prepared_root) / str(cfg.sr)
        if "{split}" in audio_dir:
            return Path(audio_dir.replace("{split}", split_of[tid]))
        return Path(audio_dir)

    spec = SPECS[cfg.model]
    max_audio_s = cfg.max_audio_seconds if cfg.max_audio_seconds is not None else spec.max_audio_s
    if spec.max_audio_s is not None and max_audio_s is not None:
        max_audio_s = min(max_audio_s, spec.max_audio_s)

    # Resume checks need only the spec, so do them before the (slow) model load.
    expect = {
        "model": spec.hf_id, "revision": cfg.revision or spec.revision, "prompt_sha": prompt_sha(cfg.max_words),
        "max_words": cfg.max_words, "max_new_tokens": MAX_NEW_TOKENS,
        "max_audio_seconds": max_audio_s, "sr": cfg.sr,
        "audio_source": audio_source, "audio_dir": audio_dir,
    }
    old_meta: dict = {}
    done: set[str] = set()
    if cfg.resume and out.exists():
        repair_torn_tail(out)
        recs = read_jsonl(out)
        done = {r["track_id"] for r in recs}
        other = sorted({r.get("model") for r in recs} - {spec.hf_id})
        if other:
            raise ValueError(f"{out} holds captions from {other}, not {spec.hf_id}")
        if any("prompt_variant" not in r for r in recs):
            raise ValueError(f"{out} holds captions made with an older prompt (no prompt_variant); "
                             "start a new file")
        sources = sorted({r.get("audio_source", "prepared") for r in recs} - {audio_source})
        if sources:
            raise ValueError(f"{out} holds captions of {sources} audio; this run reads {audio_source!r} "
                             "(windowed and full-track captions must go to separate files)")
        if meta_path.exists():
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            old = {k: old_meta.get(k, META_DEFAULTS.get(k)) for k in META_KEYS}
            diff = {k: (old[k], expect[k]) for k in META_KEYS if old[k] != expect[k]}
            if diff:
                raise ValueError(f"--resume with different settings than {meta_path}: {diff}")
    pending = [t for t in ids if t not in done]
    log.info(
        "%d selected, %d already in %s, %d to caption (model=%s, batch=%d, max_words=%d)",
        len(ids), len(ids) - len(pending), out, len(pending), cfg.model, cfg.batch_size, cfg.max_words,
    )
    summary = {"selected": len(ids), "skipped": len(ids) - len(pending), "ok": 0, "failed": 0}
    if not pending:
        return summary

    if backend is None:
        backend = load_backend(cfg.model, cfg.revision, cfg.device)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits_done = sorted(set(old_meta.get("splits") or []) | set(cfg.split or []), key=SPLIT_NAMES.index) or None
    meta = run_meta(backend, prompt, cfg.max_words, cfg.sr, max_audio_s, audio_source, audio_dir, splits_done)
    write_meta(meta_path, meta, out)

    def fail(tid: str, stage: str, err: BaseException | str, raw: str | None = None, **extra) -> None:
        summary["failed"] += 1
        rec = {"track_id": tid, "model": backend.hf_id, "stage": stage, "error": f"{type(err).__name__}: {err}"
               if isinstance(err, BaseException) else str(err), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if raw is not None:
            rec["raw"] = raw
        rec.update(extra)
        log.error("%s: %s failed: %s", tid, stage, rec["error"])
        append_jsonl(failures, rec)

    def generate(items: list[LoadedAudio], prompt: str) -> list[tuple[LoadedAudio, str | None]]:
        try:
            raws = backend.generate([it.audio for it in items], prompt)
            if len(raws) != len(items):
                raise RuntimeError(f"backend returned {len(raws)} outputs for {len(items)} inputs")
            return list(zip(items, raws, strict=True))
        except Exception as e:
            if len(items) == 1:
                fail(items[0].track_id, "generate", e)
                return [(items[0], None)]
            log.warning("batch of %d failed (%s: %s); retrying one by one", len(items), type(e).__name__, e)
            return [r for it in items for r in generate([it], prompt)]

    def post(it: LoadedAudio, raw: str) -> PostResult:
        pp = postprocess(raw, cfg.max_words)
        if pp.bpm_stripped:
            log.warning("%s: stripped BPM %s from %r", it.track_id, pp.bpm_stripped, raw)
        if pp.truncated:
            log.info("%s: %d words cut to %d", it.track_id, pp.n_words_before_cap, pp.n_words)
        return pp

    for batch in _batches(pending, cfg.batch_size):
        t0 = time.perf_counter()
        loaded: list[LoadedAudio] = []
        for tid in batch:
            try:
                y, src_s, fed_s = load_track_audio(
                    input_dir(tid) / f"{tid}.wav", cfg.sr, backend.sample_rate, max_audio_s
                )
            except Exception as e:
                fail(tid, "load", e)
                continue
            if fed_s < src_s:
                log.info("%s: %.1f s audio, cropped to %.1f s", tid, src_s, fed_s)
            else:
                log.info("%s: %.1f s audio", tid, src_s)
            loaded.append(LoadedAudio(tid, y, src_s, fed_s))
        if not loaded:
            continue
        # (item, raw, post, prompt_variant, echo reason of the primary answer, primary raw)
        accepted: list[tuple[LoadedAudio, str, PostResult, str, str | None, str | None]] = []
        retry: list[tuple[LoadedAudio, str, str]] = []
        for it, raw in generate(loaded, prompt):
            if raw is None:
                continue
            pp = post(it, raw)
            reason = is_echo(pp.caption, echo_ps)
            if reason is None:
                accepted.append((it, raw, pp, "primary", None, None))
            else:
                log.warning("%s: primary answer is an echo (%s): %r; retrying with the alt prompt", it.track_id,
                            reason, raw)
                retry.append((it, raw, reason))
        if retry:
            first = {it.track_id: (raw, reason) for it, raw, reason in retry}
            for it, raw in generate([it for it, _, _ in retry], alt_prompt):
                raw1, reason1 = first[it.track_id]
                if raw is None:
                    continue
                pp = post(it, raw)
                reason2 = is_echo(pp.caption, echo_ps)
                if reason2 is None:
                    accepted.append((it, raw, pp, "alt", reason1, raw1))
                else:
                    fail(it.track_id, "echo", f"primary: {reason1}, alt: {reason2}", raw=raw, raw_primary=raw1)
        per_item = (time.perf_counter() - t0) / len(loaded)
        order = {it.track_id: i for i, it in enumerate(loaded)}
        for it, raw, pp, variant, echo_reason, raw_primary in sorted(accepted, key=lambda a: order[a[0].track_id]):
            append_jsonl(out, {
                "track_id": it.track_id,
                "caption": pp.caption,
                "raw": raw,
                "model": backend.hf_id,
                "n_words": pp.n_words,
                "audio_seconds": round(it.audio_seconds, 3),
                "elapsed_s": round(per_item, 3),
                "audio_source": audio_source,
                "audio_dir": str(input_dir(it.track_id)),
                "split": split_of.get(it.track_id) if split_of else None,
                "prompt_variant": variant,
                "echo_retry": variant != "primary",
                "echo_reason": echo_reason,
                "raw_primary": raw_primary,
            })
            summary["ok"] += 1
            log.info("%s: %s", it.track_id, pp.caption)
    write_meta(meta_path, meta, out)
    log.info("done: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ravi.captions.autocaption", description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, default=Path("manifests/tracks.parquet"))
    p.add_argument("--prepared-root", type=Path, help="contains {sr}/{track_id}.wav")
    p.add_argument("--audio-dir", default=None,
                   help="read {DIR}/{track_id}.wav instead (e.g. make_arms' windowed WAVs); "
                        "'{split}' in DIR is replaced by each track's split, e.g. data/arms/A/{split}")
    p.add_argument("--splits", type=Path, default=Path("manifests/splits.json"), dest="splits_path")
    p.add_argument("--split", default=None, help="comma list, e.g. train,val or test (default: whole manifest)")
    p.add_argument("--sr", type=int, default=44100, help="which prepared rate to read (default 44100)")
    p.add_argument("--out", type=Path, help="JSONL output, e.g. captions/midasheng.jsonl")
    p.add_argument("--failures", type=Path, default=None, help="default: {out dir}/_failures.jsonl")
    p.add_argument("--model", choices=sorted(SPECS), default="midasheng")
    p.add_argument("--revision", default=None, help="HF revision (default: the pinned commit)")
    p.add_argument("--limit", type=int, default=None, help="first N selected tracks")
    p.add_argument("--track-id", nargs="+", action="extend", dest="track_ids", default=None)
    p.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="append, skipping track_ids already in --out")
    p.add_argument("--max-audio-seconds", type=float, default=None,
                   help="crop clips to their first N s (default: whole clip; qwen-omni caps at 300)")
    p.add_argument("--device", default=None, help="default: cuda if available")
    p.add_argument("--prefetch", action="store_true", help="download the model snapshot and exit")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    if args.prefetch:
        prefetch(args.model, args.revision)
        return 0
    missing = [f for f in ("out",) if getattr(args, f) is None]
    if args.prepared_root is None and args.audio_dir is None:
        missing.append("prepared_root (or --audio-dir)")
    if missing:
        build_parser().error("required: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    cfg = RunConfig(
        manifest=args.manifest, prepared_root=args.prepared_root, sr=args.sr, out=args.out,
        failures=args.failures, model=args.model, revision=args.revision, limit=args.limit,
        track_ids=args.track_ids, max_words=args.max_words, batch_size=args.batch_size,
        resume=args.resume, max_audio_seconds=args.max_audio_seconds, device=args.device,
        audio_dir=args.audio_dir, splits_path=args.splits_path,
        split=[x.strip() for x in args.split.split(",") if x.strip()] if args.split else None,
    )
    summary = run(cfg)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
