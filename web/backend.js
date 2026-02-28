/**
 * brAIn Pipeline — Pluggable Inference Backend
 *
 * Provides a unified interface for model inference, abstracting the underlying
 * backend (mock for demo, SageMaker for production).
 *
 * Usage:
 *   import { createBackend } from './backend.js';
 *   const backend = await createBackend(); // reads config.json
 *   const models = await backend.fetchModels();
 */

// ─── Abstract Interface ─────────────────────────────────────────────────────

class InferenceBackend {
  /** Upload an audio file. Returns { audioKey, blobUrl } */
  async uploadAudio(file) { throw new Error('Not implemented'); }

  /** Fetch available models. Returns array of model descriptors */
  async fetchModels() { throw new Error('Not implemented'); }

  /**
   * Generate music from a text prompt.
   * @param {Object} params - { prompt, duration, modelId, checkpoint, onProgress }
   * @returns {Object} { audioUrl, duration }
   */
  async generateMusic(params) { throw new Error('Not implemented'); }

  /**
   * Analyze audio with a captioning/analysis model.
   * @param {string} audioKey - key from uploadAudio
   * @param {string} modelId - analysis model endpoint name
   * @param {Object} params - model-specific parameters
   * @returns {Object} { caption, tags, ... }
   */
  async analyzeAudio(audioKey, modelId, params) { throw new Error('Not implemented'); }

  /**
   * Run generic inference (for extensibility).
   * @param {Object} params - { audioKey, modelId, parameters }
   * @returns {Object} result
   */
  async runInference(params) { throw new Error('Not implemented'); }
}

// ─── Mock Backend ────────────────────────────────────────────────────────────

class MockBackend extends InferenceBackend {
  constructor() {
    super();
    this._uploads = new Map();
    this._counter = 0;
  }

  async uploadAudio(file) {
    const key = `mock-audio-${++this._counter}`;
    const blobUrl = URL.createObjectURL(file);
    this._uploads.set(key, { file, blobUrl });
    return { audioKey: key, blobUrl };
  }

  async fetchModels() {
    return [
      {
        endpoint_name: 'stable-audio-finetuned',
        model_id: 'stable-audio-finetuned',
        display_name: 'Stable Audio Open (Finetuned)',
        description: 'Finetuned on personal corpus — 3 metadata conditioning approaches',
        model_type: 'generation',
        parameter_schema: {
          prompt: { type: 'text', label: 'Prompt', required: true },
          duration: { type: 'number', label: 'Duration (s)', default: 30, min: 5, max: 120 },
          checkpoint: {
            type: 'select', label: 'Checkpoint',
            options: [
              { value: 'human-captions', label: 'Human Captions' },
              { value: 'ai-descriptions', label: 'AI Descriptions' },
              { value: 'industry-tags', label: 'Industry Tags' },
            ]
          }
        }
      },
      {
        endpoint_name: 'musicgen-finetuned',
        model_id: 'musicgen-finetuned',
        display_name: 'MusicGen (Finetuned)',
        description: 'Finetuned MusicGen-medium on personal corpus',
        model_type: 'generation',
        parameter_schema: {
          prompt: { type: 'text', label: 'Prompt', required: true },
          duration: { type: 'number', label: 'Duration (s)', default: 15, min: 5, max: 30 },
        }
      },
      {
        endpoint_name: 'audio-flamingo',
        model_id: 'audio-flamingo',
        display_name: 'Audio Flamingo',
        description: 'Audio captioning and analysis',
        model_type: 'analysis',
        parameter_schema: {
          question: { type: 'text', label: 'Question', default: 'Describe this audio.' }
        }
      }
    ];
  }

  async generateMusic({ prompt, duration = 30, modelId, checkpoint, onProgress }) {
    const totalMs = 2000 + Math.random() * 1000; // 2-3 sec simulated
    const steps = 20;
    const stepMs = totalMs / steps;

    for (let i = 1; i <= steps; i++) {
      await _sleep(stepMs);
      if (onProgress) onProgress(i / steps);
    }

    // Return a silent audio blob as placeholder
    const sampleRate = 44100;
    const numSamples = sampleRate * Math.min(duration, 5); // 5s max for mock
    const buffer = new AudioContext().createBuffer(1, numSamples, sampleRate);
    // Fill with a simple sine tone so there's *something* audible
    const data = buffer.getChannelData(0);
    const freq = 220 + Math.random() * 220; // A3–A4
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const envelope = Math.min(1, t * 4) * Math.max(0, 1 - (t - duration + 0.5) * 4);
      data[i] = Math.sin(2 * Math.PI * freq * t) * 0.3 * envelope;
    }

    const wavBlob = _audioBufferToWav(buffer);
    const audioUrl = URL.createObjectURL(wavBlob);

    return { audioUrl, duration: buffer.duration };
  }

  async analyzeAudio(audioKey, modelId, params) {
    await _sleep(1000 + Math.random() * 1000); // 1-2 sec

    return {
      caption: 'A dramatic orchestral piece featuring sweeping string arrangements with ' +
        'building intensity. The composition blends cinematic tension with emotional depth, ' +
        'utilizing layered brass and woodwind sections. Moderate tempo, approximately 110 BPM.',
      tags: ['orchestral', 'cinematic', 'dramatic', 'strings', 'brass'],
      confidence: 0.87
    };
  }

  async runInference({ audioKey, modelId, parameters }) {
    // Generic passthrough — delegates to generateMusic or analyzeAudio
    const models = await this.fetchModels();
    const model = models.find(m => m.endpoint_name === modelId);
    if (model?.model_type === 'analysis') {
      return this.analyzeAudio(audioKey, modelId, parameters);
    }
    return this.generateMusic({ ...parameters, modelId });
  }
}

// ─── SageMaker Backend ───────────────────────────────────────────────────────
//
// Production backend that calls real AWS endpoints via API Gateway.
//
// Required config:
//   {
//     "backend": "sagemaker",
//     "apiUrl": "https://YOUR_API_GATEWAY_ID.execute-api.REGION.amazonaws.com/prod",
//     "wsUrl":  "wss://YOUR_WS_API_ID.execute-api.REGION.amazonaws.com/prod"
//   }
//
// API Contract:
//
//   GET  /models                     → [{ endpoint_name, model_id, display_name, model_type, parameter_schema }]
//   POST /upload    { audio: File }  → { audioKey }
//   POST /generate  { prompt, duration, modelId, checkpoint }
//                                    → WebSocket job: progress events → { audioUrl }
//   POST /analyze   { audioKey, modelId, params }
//                                    → { caption, tags, confidence }
//   POST /infer     { audioKey, modelId, parameters }
//                                    → WebSocket job: { resultUrl }

class SageMakerBackend extends InferenceBackend {
  constructor(config) {
    super();
    this.apiUrl = config.apiUrl;
    this.wsUrl = config.wsUrl;
    this.authToken = config.authToken || null;
  }

  _headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.authToken) h['Authorization'] = this.authToken;
    return h;
  }

  async uploadAudio(file) {
    const form = new FormData();
    form.append('audio', file);
    const res = await fetch(`${this.apiUrl}/upload`, {
      method: 'POST',
      headers: this.authToken ? { Authorization: this.authToken } : {},
      body: form,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    const { audioKey } = await res.json();
    const blobUrl = URL.createObjectURL(file);
    return { audioKey, blobUrl };
  }

  async fetchModels() {
    const res = await fetch(`${this.apiUrl}/models`, { headers: this._headers() });
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
    return res.json();
  }

  async generateMusic({ prompt, duration, modelId, checkpoint, onProgress }) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      ws.onopen = () => {
        ws.send(JSON.stringify({
          action: 'generate',
          prompt, duration, modelId, checkpoint,
        }));
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'progress' && onProgress) {
          onProgress(msg.progress);
        } else if (msg.type === 'complete') {
          ws.close();
          resolve({ audioUrl: msg.audioUrl, duration: msg.duration });
        } else if (msg.type === 'error') {
          ws.close();
          reject(new Error(msg.message));
        }
      };
      ws.onerror = (err) => reject(new Error('WebSocket error'));
    });
  }

  async analyzeAudio(audioKey, modelId, params) {
    const res = await fetch(`${this.apiUrl}/analyze`, {
      method: 'POST',
      headers: this._headers(),
      body: JSON.stringify({ audioKey, modelId, params }),
    });
    if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
    return res.json();
  }

  async runInference({ audioKey, modelId, parameters }) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      ws.onopen = () => {
        ws.send(JSON.stringify({
          action: 'infer',
          audioKey, modelId, parameters,
        }));
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'complete') {
          ws.close();
          resolve(msg.result);
        } else if (msg.type === 'error') {
          ws.close();
          reject(new Error(msg.message));
        }
      };
      ws.onerror = () => reject(new Error('WebSocket error'));
    });
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function _sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/** Encode an AudioBuffer to a WAV Blob */
function _audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const numSamples = buffer.length;
  const dataSize = numSamples * blockAlign;
  const headerSize = 44;
  const arrayBuffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(arrayBuffer);

  // WAV header
  _writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  _writeString(view, 8, 'WAVE');
  _writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  _writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Interleave channels
  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) channels.push(buffer.getChannelData(ch));

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

function _writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/**
 * Create the appropriate backend based on config.json.
 * @returns {InferenceBackend}
 */
export async function createBackend() {
  let config = { backend: 'mock' };
  try {
    const res = await fetch('./config.json');
    if (res.ok) config = await res.json();
  } catch (e) {
    console.warn('[backend] Could not load config.json, using mock backend');
  }

  if (config.backend === 'sagemaker') {
    console.log('[backend] Using SageMaker backend');
    return new SageMakerBackend(config);
  }

  console.log('[backend] Using Mock backend');
  return new MockBackend();
}

export { InferenceBackend, MockBackend, SageMakerBackend };
