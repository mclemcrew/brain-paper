"""Bounded, offline CPU music tagging and acoustic boundary proposals.

CLAP is a candidate-ranking model, not a free-form captioner. Scores are cosine
similarities, never confidence probabilities. No suggestion is auto-approved.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import soundfile as sf

REPO = 'Xenova/clap-htsat-unfused'
REVISION = 'c28f2883575e590e04d3146ff0713c2448d691ba'
VERSION = 'clap-q8-acoustic-v2'
FILES = ('onnx/audio_model_quantized.onnx', 'onnx/text_model_quantized.onnx', 'tokenizer.json', 'preprocessor_config.json')
CANDIDATES = {
    'instrumentation': ['piano', 'acoustic guitar', 'electric guitar', 'bowed strings', 'orchestra', 'brass', 'woodwinds', 'synthesizers', 'percussion', 'bass guitar', 'organ', 'vocals'],
    'character': ['gentle and reflective', 'dark and tense', 'bright and playful', 'energetic and driving', 'spacious and atmospheric', 'solemn and dramatic'],
    'texture': ['a sparse solo', 'a small acoustic ensemble', 'a dense ensemble', 'layered electronic textures', 'a rhythmic groove'],
}


def pack_status(root):
    directory = Path(root) / 'models' / 'tagger'
    return {'ready': all((directory / f).is_file() for f in FILES),
            'model': REPO, 'revision': REVISION, 'download_mb': 165,
            'license_url': 'https://huggingface.co/laion/clap-htsat-unfused',
            'description': 'Local CLAP draft tags + acoustic section suggestions; review required.'}


def install_pack(root):
    import os
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import hf_hub_download
    directory = Path(root) / 'models' / 'tagger'
    for filename in FILES:
        hf_hub_download(REPO, filename, revision=REVISION, local_dir=directory)


def analysis_key(track):
    return hashlib.sha256(json.dumps([track['sha256'], VERSION, REVISION, CANDIDATES], sort_keys=True).encode()).hexdigest()


def mono_window(f, start, seconds, target_sr):
    from scipy.signal import resample_poly
    f.seek(min(len(f), round(start * f.samplerate)))
    y = f.read(round(seconds * f.samplerate), dtype='float32', always_2d=True).mean(axis=1)
    g = math.gcd(f.samplerate, target_sr)
    return resample_poly(y, target_sr // g, f.samplerate // g).astype(np.float32)


def clap_features(y):
    import librosa
    # Match unfused HF ClapFeatureExtractor: Slaney filters, repeatpad, 10 s.
    if not len(y) or not np.isfinite(y).all(): raise ValueError('Invalid analysis audio.')
    y = y[:480000]
    if len(y) < 480000: y = np.pad(np.tile(y, 480000 // len(y)), (0, 480000 % len(y)))
    mel = librosa.feature.melspectrogram(y=y.astype(np.float64), sr=48000, n_fft=1024,
        hop_length=480, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney', pad_mode='reflect')
    return (10 * np.log10(np.maximum(mel, 1e-10))).T[None, None].astype(np.float32)


class Tagger:
    def __init__(self, root):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        ort.disable_telemetry_events()
        directory = Path(root) / 'models' / 'tagger'
        if not pack_status(root)['ready']: raise ValueError('Install the local tagger in Setup first.')
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = False
        self.audio = ort.InferenceSession(str(directory / FILES[0]), sess_options=opts, providers=['CPUExecutionProvider'])
        text = ort.InferenceSession(str(directory / FILES[1]), sess_options=opts, providers=['CPUExecutionProvider'])
        tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        tokenizer.enable_padding(pad_id=1, pad_token='<pad>'); tokenizer.enable_truncation(max_length=77)
        self.labels = [(category, label) for category, labels in CANDIDATES.items() for label in labels]
        embeddings = []
        for category, label in self.labels:
            phrase = f'Music featuring {label}.' if category == 'instrumentation' else f'The music is {label}.'
            encoded = tokenizer.encode(phrase)
            feeds = {'input_ids': np.array([encoded.ids], np.int64), 'attention_mask': np.array([encoded.attention_mask], np.int64)}
            output = text.run(None, {i.name: feeds[i.name] for i in text.get_inputs()})[0]
            embeddings.append(output[0])
        self.text = np.asarray(embeddings)
        self.text /= np.maximum(np.linalg.norm(self.text, axis=1, keepdims=True), 1e-8)
        del text  # Text encoder is released before corpus audio processing.

    def rank(self, audio):
        features = clap_features(audio)
        feeds = {'input_features': features, 'is_longer': np.array([[False]])}
        embedding = self.audio.run(None, {i.name: feeds[i.name] for i in self.audio.get_inputs()})[0][0]
        embedding /= max(np.linalg.norm(embedding), 1e-8)
        scores = self.text @ embedding
        return {category: sorted([{'label': label, 'similarity': round(float(scores[i]), 4)}
                                  for i, (group, label) in enumerate(self.labels) if group == category],
                                 key=lambda x: x['similarity'], reverse=True)[:3] for category in CANDIDATES}


def structured_suggestions(tags, start, end):
    """Keep ranked evidence separate from composer-approved observations."""
    result = {}
    for group, field in [('instrumentation', 'instruments'), ('character', 'mood'), ('texture', 'texture')]:
        if not tags.get(group): continue
        best = tags[group][0]
        result[field] = {'value': best['label'], 'reviewed': False, 'source': REPO,
                         'suggestion': {'value': best['label'], 'model': REPO,
                                        'revision': REVISION, 'version': VERSION,
                                        'start': start, 'end': end,
                                        'score': best['similarity'], 'score_type': 'ranking_similarity'}}
    return result


def propose(path, tagger):
    import librosa
    from scipy.signal import find_peaks
    # Stream the entire recording in bounded 10 s windows. Only tiny summary vectors
    # survive a window; no quadratic whole-track similarity matrix is allocated.
    vectors, ranks, tempos, starts = [], [], [], []
    with sf.SoundFile(path) as f:
        duration = len(f) / f.samplerate
        for start in np.arange(0, duration, 10.):
            y = mono_window(f, float(start), min(10., duration-start), 48000)
            if len(y) < 4800: continue
            ranks.append(tagger.rank(y)); starts.append(float(start))
            z = librosa.resample(y, orig_sr=48000, target_sr=12000)
            mel = librosa.feature.melspectrogram(y=z, sr=12000, n_fft=1024, hop_length=512, n_mels=24)
            v = np.log1p(mel).mean(axis=1); vectors.append(v / max(np.linalg.norm(v), 1e-8))
            if start < 40 and np.max(np.abs(z)) > 1e-5:
                tempo, _ = librosa.beat.beat_track(y=z, sr=12000, hop_length=256)
                tempos.append(float(np.asarray(tempo).item()))
    if not starts: raise ValueError('Audio is too short for local analysis.')
    novelty = np.array([0.] + [float(np.linalg.norm(b-a)) for a,b in zip(vectors, vectors[1:])])
    peaks, _ = find_peaks(novelty, height=max(.15, float(np.median(novelty)+np.std(novelty))), distance=2)
    boundaries = [0.] + [starts[i] for i in peaks if 5 <= starts[i] <= duration-5] + [duration]
    def summarize(indices):
        grouped = {}
        for category, labels in CANDIDATES.items():
            # Missing top-three labels get no vote; return rankings, not confidence.
            values = {label: [] for label in labels}
            for i in indices:
                for row in ranks[i][category]: values[row['label']].append(row['similarity'])
            grouped[category] = sorted([{'label': label, 'similarity': round(sum(v)/len(indices), 4)}
                                       for label,v in values.items() if v], key=lambda x:x['similarity'], reverse=True)[:3]
        caption = f"{grouped['character'][0]['label'].capitalize()} music with {grouped['instrumentation'][0]['label']}; {grouped['texture'][0]['label']}."
        return caption, grouped
    caption, tags = summarize(list(range(len(starts))))
    sections = []
    for i, (a,b) in enumerate(zip(boundaries, boundaries[1:])):
        text, local_tags = summarize([j for j,s in enumerate(starts) if a <= s < b])
        sections.append({'id': f'proposal-{i}', 'start': a, 'end': b, 'label': f'Section {i+1}', 'caption': text,
                         'musical_details': structured_suggestions(local_tags, a, b)})
    return {'caption': caption, 'tags': tags, 'sections': sections,
            'musical_details': structured_suggestions(tags, 0., duration),
            'suggested_bpm': round(float(np.median(tempos)),1) if tempos else None,
            'model': REPO, 'revision': REVISION, 'version': VERSION,
            'method': 'CLAP candidate rankings; 10-second acoustic-change boundaries. Neither is verified musical analysis.',
            'windows': len(starts), 'coverage_seconds': duration}
