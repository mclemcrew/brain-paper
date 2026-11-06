"""Audio (and text) embedding extractors for the rAvI v2 evaluation (plan Phase 4).

Common interface (every extractor)::

    ex = get_extractor("clap")            # or "mert"
    ex.embed(paths_or_arrays, sr=None)    # -> np.ndarray [n, d], one pooled row per input
    ex.embed_chunks(paths_or_arrays, sr)  # -> list of [k_i, d] window embeddings per input
    ex.embed_text(texts)                  # CLAP only -> [n, d]

Inputs are file paths or float arrays ``(samples,)`` / ``(samples, channels)``; arrays take the
``sr`` argument (or are passed as ``(array, sr)`` tuples). Audio is downmixed to mono (channel
mean) and resampled to the model's native rate with :func:`ravi.eval.sa3_infer.resample`
(``torchaudio.functional.resample`` when torch is installed, as in ``docker/eval``; else
``scipy.signal.resample_poly``; the resampler is part of the cache key).

**Windowing** (both models): windows of ``window_s`` starting every ``hop_s`` from 0 while they
fit; if the last fitted window ends before the input does, one more window aligned to the end is
added, so the tail is covered without zero padding; an input shorter than one window is a single
window of its own length. The pooled row (:meth:`Extractor.embed`) is the mean of the window
embeddings (for CLAP, re-normalised to unit length). FAD/KAD use the window rows
(:meth:`Extractor.embed_chunks`), cosine metrics the pooled rows.

**CLAP** (:class:`ClapExtractor`): LAION-CLAP (Wu, Chen et al., ICASSP 2023; ``laion_clap``
1.1.7, https://github.com/LAION-AI/CLAP) with the *music* checkpoint
``music_audioset_epoch_15_esc_90.14.pt`` ("trained on music + Audioset + LAION-Audio-630k",
ESC-50 zero-shot 90.14%, GTZAN 71%: LAION-CLAP README), which uses the larger audio encoder:
``CLAP_Module(enable_fusion=False, amodel='HTSAT-base')`` then ``load_ckpt(path)`` (README; all
new checkpoints are non-fusion). fadtk's ``clap-laion-music`` (Gui et al. 2024) is the same
checkpoint and model. Downloaded with ``huggingface_hub.hf_hub_download`` from
``lukewys/laion_clap`` at revision :data:`CLAP_HF_REVISION`; LFS sha256 :data:`CLAP_SHA256`
(verified after download). 48 kHz mono; 10 s windows (``clip_samples`` 480,000 in
``HTSAT-base.json``), hop 1 s by default (fadtk's L-CLAP setting, Gui et al. Table 1). Each
window is quantised to int16 and back (:func:`clap_quantise`, laion_clap's
``int16_to_float32(float32_to_int16(x))``, done explicitly here so it is tested; applying it a
second time is the identity on its own outputs, checked over all 65,536 int16 levels) and goes
through ``CLAP_Module.get_audio_embedding_from_data(x, use_tensor=False)``
(``hook.py:157-191``), which quantises again (a no-op now) and calls
``get_audio_features(..., 480000, data_truncating='rand_trunc', data_filling='repeatpad')``
(``training/data.py:402-497``): a window of exactly 480,000 samples is neither cropped nor
padded, so the random crop never fires and the output is deterministic; a shorter input is
repeat-padded (the training-time default). Output rows are L2-normalised
(``clap_module/model.py:741``), 512-d. Text: ``get_text_embedding(texts)`` (RoBERTa tokenizer,
max 77 tokens including ``<s>``/``</s>``, longer prompts are silently truncated:
:meth:`ClapExtractor.text_truncated` reports which; ``RobertaTokenizer/RobertaModel.from_pretrained
('roberta-base')`` are fetched from the HF hub by laion_clap itself), L2-normalised
(``model.py:716``).

**Loudness** (CLAP only): CLAP embeddings move with input level (the same 20 s segment at -12 dB
has cosine ~0.9 to itself at 0 dB, comparable to the gap between conditions), and generated
clips are not loudness-matched. ``run_eval`` therefore passes every generated clip, test original
and region through :func:`lufs_normalise` before CLAP: ITU-R BS.1770-4 integrated loudness
(pyloudnorm) to -16 LUFS with a single gain, reduced so the 4x-oversampled true peak stays at or
below -1 dBTP, plus one correction step; this is exactly ``ravi.data.prepare_audio.normalise``,
the procedure the training corpus (the FAD/KAD reference) went through. MERT is left alone: its
``Wav2Vec2FeatureExtractor`` (``do_normalize: true``) z-scores every window
(:func:`wav2vec2_normalise`), so MERT embeddings are already gain-invariant.

**MERT** (:class:`MertExtractor`): ``m-a-p/MERT-v1-95M`` (Li et al., "MERT", ICLR 2024) at HF
revision :data:`MERT_HF_REVISION`, through ``transformers`` with ``trust_remote_code=True``
(``AutoModel`` -> ``modeling_MERT.MERTModel``, a ``HubertModel`` subclass) and
``Wav2Vec2FeatureExtractor`` (``do_normalize: true``, ``sampling_rate: 24000`` in
``preprocessor_config.json``), as on the model card. 24 kHz mono, 75 frames/s, 768-d, 13 hidden
states (index 0 = the pre-transformer projection, 12 = the last layer). We use **layer 12** (the
last layer, fadtk's default ``MERT-v1-95M``) mean-pooled over the frames of each **5 s window**
(MERT's pre-training clip length), hop 1 s; ``layer`` is configurable (Gui et al. found layer 4
best for musical-quality prediction). Each window is z-scored on its own
(:func:`wav2vec2_normalise`, the processor's ``zero_mean_unit_var_norm``: ``(x - mean) /
sqrt(var + 1e-7)``; :meth:`MertExtractor.load` checks it against the processor's output) and the
batch goes straight to the model (:meth:`MertExtractor._forward`). ``config.conv_pos_batch_norm = False`` is set before
loading, as fadtk does (MERT's config sets it too: "fix bug with hf > 4.42"). Windows are batched
by equal length and passed without an attention mask (group-norm feature encoder; no padding is
ever needed). The remote code imports ``HubertFeatureEncoder, HubertModel,
HubertEncoderStableLayerNorm, HubertEncoder, HubertEncoderLayer, HubertPositionalConvEmbedding,
HubertAttention, HubertFeedForward`` from ``transformers.models.hubert.modeling_hubert``; all
eight exist in the pinned transformers 4.52.4.

**Cache**: ``{cache_dir}/{extractor.cache_id}/{key}.npz`` holding ``chunks`` [k, d] and
``starts_s`` [k]. ``key`` is the sha256 of the file bytes (paths) or of the float32 samples plus
rate and shape (arrays); ``cache_id`` is the extractor name plus a hash of its configuration
(window, hop, layer, checkpoint name *and sha256*, HF revision, resampler, and the installed
torch / laion_clap / transformers versions). A cached file is trusted only if its stored
``cache_id`` matches. Writes go to a per-process temporary name (pid + random suffix) and are
renamed into place, so concurrent runs sharing a cache cannot interleave.

Self-test inside the image (downloads the weights): ``python -m ravi.eval.embed --self-test``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "Extractor",
    "ClapExtractor",
    "MertExtractor",
    "get_extractor",
    "window_starts",
    "sha256_file",
    "load_mono",
    "lufs_normalise",
    "clap_quantise",
    "wav2vec2_normalise",
    "EXTRACTORS",
]

CLAP_HF_REPO = "lukewys/laion_clap"
CLAP_CKPT = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_HF_REVISION = "b3708341862f581175dba5c356a4ebf74a9b6651"  # repo HEAD, last modified 2023-04-08
CLAP_SHA256 = "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"  # LFS oid, 2,352,471,003 B
MERT_HF_REPO = "m-a-p/MERT-v1-95M"
MERT_HF_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"  # last modified 2025-05-25
CLAP_TEXT_MAX_TOKENS = 77  # laion_clap hook.tokenizer: max_length=77, truncation=True
TARGET_LUFS = -16.0  # ravi.data.prepare_audio defaults
CEILING_DBTP = -1.0

AudioInput = "str | Path | np.ndarray | tuple[np.ndarray, int]"


# --------------------------------------------------------------------------- helpers


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray, sr: int) -> str:
    a = np.ascontiguousarray(a, dtype=np.float32)
    h = hashlib.sha256()
    h.update(f"{a.shape}|{int(sr)}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _resampler_name() -> str:
    try:
        import torchaudio  # noqa: F401

        return f"torchaudio-{torchaudio.__version__.split('+')[0]}"
    except ImportError:
        import scipy

        return f"scipy-resample_poly-{scipy.__version__}"


def load_mono(item: Any, sr: int | None, target_sr: int) -> np.ndarray:
    """Mono float32 at ``target_sr`` from a path, an array (with ``sr``) or ``(array, sr)``."""
    from ravi.eval.sa3_infer import load_audio, resample

    if isinstance(item, (str, Path)):
        audio, in_sr = load_audio(item)
    elif isinstance(item, tuple) and len(item) == 2:
        audio, in_sr = np.asarray(item[0]), int(item[1])
    else:
        if sr is None:
            raise ValueError("arrays need sr= (or pass (array, sr) tuples)")
        audio, in_sr = np.asarray(item), int(sr)
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim != 2:
        raise ValueError(f"expected (samples,) or (samples, channels), got {a.shape}")
    a = a.mean(axis=1, keepdims=True)
    a = resample(np.ascontiguousarray(a), in_sr, target_sr)
    return np.ascontiguousarray(a[:, 0], dtype=np.float32)


def _pkg_versions(names: Sequence[str]) -> dict[str, str | None]:
    """Installed distribution versions (``None`` if absent), without importing the packages."""
    from importlib import metadata

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def _no_grad():
    try:
        import torch
    except ImportError:  # stub models in the tests
        return contextlib.nullcontext()
    return torch.no_grad()


def _tmp_name(path: Path) -> Path:
    return path.with_name(f"{path.stem}.{os.getpid()}.{secrets.token_hex(4)}.tmp{path.suffix}")


def true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float:
    """4x-oversampled true peak (``ravi.data.prepare_audio.true_peak_dbtp``)."""
    from scipy.signal import resample_poly

    up = resample_poly(np.asarray(audio, np.float64), oversample, 1, axis=0)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def lufs_normalise(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS,
                   ceiling_dbtp: float = CEILING_DBTP) -> tuple[np.ndarray, dict[str, Any]]:
    """Gain-only loudness normalisation, the procedure of ``ravi.data.prepare_audio.normalise``
    (module docstring): returns ``(audio, stats)`` with ``lufs_in``, ``gain_db``, ``lufs_out``,
    ``true_peak_dbtp``, ``peak_limited``, ``silent``. Silence (or a clip too short to gate) is
    returned unchanged."""
    import pyloudnorm as pyln

    a = np.asarray(audio, dtype=np.float32)
    meter = pyln.Meter(int(sr))
    try:
        lufs_in = float(meter.integrated_loudness(a.astype(np.float64)))
    except ValueError:  # shorter than one 400 ms gating block
        lufs_in = -math.inf
    if not math.isfinite(lufs_in):
        return a, {"lufs_in": lufs_in, "gain_db": 0.0, "lufs_out": lufs_in, "true_peak_dbtp": true_peak_dbtp(a),
                   "peak_limited": False, "silent": True}
    tp_in = true_peak_dbtp(a)
    gain_db = target_lufs - lufs_in
    limited = tp_in + gain_db > ceiling_dbtp
    if limited:
        gain_db -= tp_in + gain_db - ceiling_dbtp
    src = a.astype(np.float64)
    out = (src * 10 ** (gain_db / 20)).astype(np.float32)
    lufs_first = float(meter.integrated_loudness(out.astype(np.float64)))
    correction_db = 0.0
    if not limited and math.isfinite(lufs_first):
        correction_db = min(target_lufs - lufs_first, ceiling_dbtp - (tp_in + gain_db))
        if correction_db != 0.0:
            gain_db += correction_db
            out = (src * 10 ** (gain_db / 20)).astype(np.float32)
    lufs_out = float(meter.integrated_loudness(out.astype(np.float64))) if correction_db else lufs_first
    return out, {"lufs_in": lufs_in, "gain_db": float(gain_db), "lufs_out": lufs_out,
                 "true_peak_dbtp": true_peak_dbtp(out), "peak_limited": bool(limited), "silent": False}


def clap_quantise(x: np.ndarray) -> np.ndarray:
    """laion_clap's ``int16_to_float32(float32_to_int16(x))`` (``training/data.py:93-99``)."""
    q = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    return (q / 32767.0).astype(np.float32)


def wav2vec2_normalise(x: np.ndarray) -> np.ndarray:
    """``Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`` for one unpadded window."""
    x = np.asarray(x, np.float32)
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


def window_starts(n: int, win: int, hop: int) -> list[int]:
    """Window starts covering ``n`` samples (module docstring): every ``hop`` while a full
    window fits, plus one end-aligned window for an uncovered tail; ``[0]`` if ``n <= win``."""
    if n <= 0:
        raise ValueError("empty input")
    if win <= 0 or hop <= 0:
        raise ValueError("window and hop must be > 0")
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


# --------------------------------------------------------------------------- base class


class Extractor:
    """Windowing, pooling, batching and caching; subclasses implement :meth:`_embed_windows`."""

    name: str = "base"
    sr: int = 16000
    normalize_pooled: bool = False

    def __init__(self, window_s: float, hop_s: float, cache_dir: str | Path | None = None,
                 batch_size: int = 16) -> None:
        if window_s <= 0 or hop_s <= 0:
            raise ValueError("window_s and hop_s must be > 0")
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = int(batch_size)
        self._resampler = _resampler_name()

    # -- identity
    def config(self) -> dict[str, Any]:
        return {"name": self.name, "sr": self.sr, "window_s": self.window_s, "hop_s": self.hop_s,
                "resampler": self._resampler}

    @property
    def cache_id(self) -> str:
        blob = json.dumps(self.config(), sort_keys=True).encode()
        return f"{self.name}-{hashlib.sha256(blob).hexdigest()[:12]}"

    @property
    def dim(self) -> int:
        raise NotImplementedError

    # -- model call
    def _embed_windows(self, windows: list[np.ndarray]) -> np.ndarray:
        """``windows``: mono float32 arrays at :attr:`sr`, all the same length; -> [k, d]."""
        raise NotImplementedError

    # -- public
    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / self.cache_id / f"{key}.npz"

    def _key(self, item: Any, sr: int | None) -> str | None:
        if self.cache_dir is None:
            return None
        if isinstance(item, (str, Path)):
            return sha256_file(item)
        if isinstance(item, tuple) and len(item) == 2:
            return sha256_array(item[0], int(item[1]))
        return sha256_array(item, int(sr) if sr is not None else -1)

    def _compute(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win = int(round(self.window_s * self.sr))
        hop = int(round(self.hop_s * self.sr))
        starts = window_starts(audio.shape[0], win, hop)
        wins = [audio[s : s + win] for s in starts]
        out: list[np.ndarray | None] = [None] * len(wins)
        by_len: dict[int, list[int]] = {}
        for i, w in enumerate(wins):
            by_len.setdefault(w.shape[0], []).append(i)
        for _, idx in sorted(by_len.items()):
            for b in range(0, len(idx), self.batch_size):
                sel = idx[b : b + self.batch_size]
                emb = np.asarray(self._embed_windows([wins[i] for i in sel]), dtype=np.float32)
                for j, i in enumerate(sel):
                    out[i] = emb[j]
        return np.stack(out), np.asarray(starts, dtype=np.float64) / self.sr

    def embed_chunks(self, items: Sequence[Any], sr: int | None = None) -> list[np.ndarray]:
        """Window embeddings ``[k_i, d]`` per input (cached)."""
        res = []
        for item in items:
            key = self._key(item, sr)
            path = self._cache_path(key) if key else None
            if path is not None and path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z["cache_id"]) == self.cache_id:
                        res.append(z["chunks"].astype(np.float32))
                        continue
            audio = load_mono(item, sr, self.sr)
            chunks, starts = self._compute(audio)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = _tmp_name(path)
                np.savez(tmp, chunks=chunks, starts_s=starts, cache_id=np.array(self.cache_id))
                os.replace(tmp, path)
            res.append(chunks)
        return res

    def pool(self, chunks: np.ndarray) -> np.ndarray:
        v = np.asarray(chunks, dtype=np.float64).mean(axis=0)
        if self.normalize_pooled:
            n = np.linalg.norm(v)
            v = v / n if n > 0 else v
        return v.astype(np.float32)

    def embed(self, items: Sequence[Any], sr: int | None = None) -> np.ndarray:
        """One pooled row per input: ``[n, d]``."""
        return np.stack([self.pool(c) for c in self.embed_chunks(items, sr)]) if len(items) else np.zeros((0, self.dim), np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no text tower")


# --------------------------------------------------------------------------- CLAP


class ClapExtractor(Extractor):
    name = "clap-music"
    sr = 48000
    normalize_pooled = True

    def __init__(self, ckpt: str | Path | None = None, device: str | None = None,
                 window_s: float = 10.0, hop_s: float = 1.0, cache_dir: str | Path | None = None,
                 batch_size: int = 16, verify_sha: bool = True) -> None:
        super().__init__(window_s, hop_s, cache_dir, batch_size)
        if window_s > 10.0:
            raise ValueError("CLAP windows are at most 10 s (480,000 samples); longer would be random-cropped")
        self.ckpt = ckpt
        self.device = device
        self.verify_sha = verify_sha
        self._model = None
        self._ckpt_sha: str | None = None

    def _ckpt_identity(self) -> tuple[str, str | None]:
        """(sha256, HF revision) of the checkpoint: the pinned constants for the hub download,
        else the sha256 of the local file (hashed once)."""
        if self.ckpt is None:
            return CLAP_SHA256, CLAP_HF_REVISION
        if self._ckpt_sha is None:
            self._ckpt_sha = sha256_file(self.ckpt) if Path(self.ckpt).exists() else f"missing:{self.ckpt}"
        return self._ckpt_sha, None

    def config(self) -> dict[str, Any]:
        sha, rev = self._ckpt_identity()
        return {**super().config(), "ckpt": CLAP_CKPT if self.ckpt is None else Path(self.ckpt).name,
                "ckpt_sha256": sha, "revision": rev, "amodel": "HTSAT-base", "enable_fusion": False,
                "quantise": "int16", "versions": _pkg_versions(("torch", "laion_clap", "transformers"))}

    @property
    def dim(self) -> int:
        return 512

    def load(self):
        if self._model is not None:
            return self._model
        import torch
        import laion_clap

        path = self.ckpt
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(CLAP_HF_REPO, CLAP_CKPT, revision=CLAP_HF_REVISION)
        if self.verify_sha:
            sha = sha256_file(path)
            if Path(path).name == CLAP_CKPT and sha != CLAP_SHA256:
                raise RuntimeError(f"{path}: sha256 {sha} != {CLAP_SHA256}")
            if self.ckpt is not None:
                self._ckpt_sha = sha
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        model.load_ckpt(str(path), verbose=False)
        model.eval()
        self._model = model
        return model

    def _embed_windows(self, windows: list[np.ndarray]) -> np.ndarray:
        model = self.load()
        with _no_grad():
            emb = model.get_audio_embedding_from_data(x=[clap_quantise(w) for w in windows], use_tensor=False)
        return np.asarray(emb, dtype=np.float32)

    def text_token_counts(self, texts: Sequence[str]) -> list[int]:
        """RoBERTa token counts (with ``<s>``/``</s>``) of laion_clap's own tokenizer."""
        tok = self.load().tokenize
        return [len(tok(str(t))["input_ids"]) for t in texts]

    def text_truncated(self, texts: Sequence[str]) -> list[bool]:
        """True where laion_clap truncates the prompt (> :data:`CLAP_TEXT_MAX_TOKENS` tokens)."""
        return [n > CLAP_TEXT_MAX_TOKENS for n in self.text_token_counts(texts)]

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised CLAP text embeddings [n, 512] (cached by sha256 of the text)."""
        texts = [str(t) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        cdir = None if self.cache_dir is None else self.cache_dir / f"{self.cache_id}-text"
        for i, t in enumerate(texts):
            p = None if cdir is None else cdir / f"{hashlib.sha256(t.encode('utf-8')).hexdigest()}.npy"
            if p is not None and p.exists():
                out[i] = np.load(p)
            else:
                todo.append((i, p))
        if todo:
            model = self.load()
            for b in range(0, len(todo), 64):
                batch = todo[b : b + 64]
                with _no_grad():
                    emb = model.get_text_embedding([texts[i] for i, _ in batch], use_tensor=False)
                for (i, p), e in zip(batch, np.asarray(emb, np.float32)):
                    out[i] = e
                    if p is not None:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        tmp = _tmp_name(p)
                        np.save(tmp, e)
                        os.replace(tmp, p)
        return np.stack(out) if out else np.zeros((0, 512), np.float32)


# --------------------------------------------------------------------------- MERT


class MertExtractor(Extractor):
    name = "mert-v1-95m"
    sr = 24000

    def __init__(self, layer: int = 12, device: str | None = None, window_s: float = 5.0,
                 hop_s: float = 1.0, cache_dir: str | Path | None = None, batch_size: int = 16,
                 revision: str = MERT_HF_REVISION) -> None:
        super().__init__(window_s, hop_s, cache_dir, batch_size)
        if not 0 <= int(layer) <= 12:
            raise ValueError("MERT-v1-95M has hidden states 0..12")
        self.layer = int(layer)
        self.device = device
        self.revision = revision
        self._model = None
        self._proc = None

    def config(self) -> dict[str, Any]:
        return {**super().config(), "layer": self.layer, "revision": self.revision, "pool": "mean",
                "input_norm": "per-window z-score", "versions": _pkg_versions(("torch", "transformers"))}

    @property
    def dim(self) -> int:
        return 768

    def load(self):
        if self._model is not None:
            return self._model, self._proc
        import torch
        from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

        cfg = AutoConfig.from_pretrained(MERT_HF_REPO, trust_remote_code=True, revision=self.revision)
        cfg.conv_pos_batch_norm = False
        model = AutoModel.from_pretrained(MERT_HF_REPO, trust_remote_code=True, config=cfg, revision=self.revision)
        proc = Wav2Vec2FeatureExtractor.from_pretrained(MERT_HF_REPO, trust_remote_code=True, revision=self.revision)
        if int(proc.sampling_rate) != self.sr:
            raise RuntimeError(f"MERT processor rate {proc.sampling_rate} != {self.sr}")
        if not getattr(proc, "do_normalize", False):
            raise RuntimeError("MERT processor has do_normalize off; wav2vec2_normalise would not match it")
        probe = np.random.default_rng(0).standard_normal(self.sr).astype(np.float32) * 0.1 + 0.01
        ref = np.asarray(proc([probe], sampling_rate=self.sr, return_tensors="np")["input_values"][0], np.float32)
        if not np.allclose(ref, wav2vec2_normalise(probe), atol=1e-5):
            raise RuntimeError("wav2vec2_normalise differs from the MERT processor's normalisation")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        self._model, self._proc, self._device = model, proc, device
        return model, proc

    def _forward(self, batch: np.ndarray):
        """Model call on normalised windows ``[b, samples]``: the 13 hidden states."""
        import torch

        model, _ = self.load()
        with torch.no_grad():
            out = model(input_values=torch.from_numpy(batch).to(self._device), output_hidden_states=True)
        return out.hidden_states

    def _embed_windows(self, windows: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([wav2vec2_normalise(w) for w in windows])
        hs = self._forward(batch)
        if len(hs) != 13:
            raise RuntimeError(f"expected 13 hidden states, got {len(hs)}")
        h = hs[self.layer]  # [b, frames, 768]
        if hasattr(h, "detach"):  # torch: pool on the device, as before
            return h.detach().float().mean(dim=1).cpu().numpy()
        return np.asarray(h, dtype=np.float32).mean(axis=1)


# --------------------------------------------------------------------------- registry


EXTRACTORS: dict[str, Callable[..., Extractor]] = {"clap": ClapExtractor, "mert": MertExtractor}


def get_extractor(name: str, **kwargs: Any) -> Extractor:
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name](**kwargs)


def _self_test(cache_dir: str | None) -> int:
    sr = 44100
    t = np.arange(int(12.3 * sr)) / sr
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.05 * np.random.default_rng(0).standard_normal(t.size)).astype(np.float32)
    for name in ("clap", "mert"):
        ex = get_extractor(name, cache_dir=cache_dir)
        chunks = ex.embed_chunks([tone, noise], sr=sr)
        pooled = ex.embed([tone, noise], sr=sr)
        again = ex.embed([tone], sr=sr)
        print(f"{ex.cache_id}: windows {[c.shape for c in chunks]} pooled {pooled.shape} "
              f"cos(tone, noise) {float(pooled[0] @ pooled[1] / np.linalg.norm(pooled[0]) / np.linalg.norm(pooled[1])):.3f} "
              f"repeat max|diff| {float(np.max(np.abs(again[0] - pooled[0]))):.2e}")
        if name == "clap":
            te = ex.embed_text(["a sine tone", "white noise"])
            print(f"  text {te.shape}, cos(tone audio, 'a sine tone') {float(te[0] @ pooled[0]):.3f}, "
                  f"cos(tone audio, 'white noise') {float(te[1] @ pooled[0]):.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true", help="load CLAP and MERT and embed two test signals")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test(args.cache_dir)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
