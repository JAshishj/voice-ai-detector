/**
 * Browser audio helpers: decode any dropped/recorded blob to 16 kHz mono,
 * render preview peaks for the waveform canvas, and encode a WAV base64
 * payload the API accepts. The server also resamples, but sending clean
 * 16 kHz mono WAV keeps behavior identical across browsers.
 */

export interface PreparedAudio {
  /** 16 kHz mono samples, peak-normalized */
  samples: Float32Array;
  sampleRate: 16000;
  /** Downsampled peak envelope for canvas rendering */
  peaks: Float32Array;
  durationSec: number;
  wavBase64: string;
}

const PEAK_BINS = 240;

async function decodeToMono16k(blob: Blob): Promise<Float32Array> {
  const raw = await blob.arrayBuffer();
  const Ctx =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext;
  const ctx = new Ctx();
  try {
    const decoded = await ctx.decodeAudioData(raw.slice(0));
    const targetLen = Math.floor((decoded.length / decoded.sampleRate) * 16000);
    const offline = new OfflineAudioContext(1, Math.max(1, targetLen), 16000);
    const src = offline.createBufferSource();
    // Downmix to mono manually so channel count can't surprise us.
    const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    const out = mono.getChannelData(0);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < decoded.length; i++) out[i] += ch[i] / decoded.numberOfChannels;
    }
    src.buffer = mono;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  } finally {
    void ctx.close();
  }
}

function toPeaks(samples: Float32Array, bins = PEAK_BINS): Float32Array {
  const peaks = new Float32Array(bins);
  const per = Math.max(1, Math.floor(samples.length / bins));
  for (let b = 0; b < bins; b++) {
    let peak = 0;
    const start = b * per;
    for (let i = start; i < Math.min(start + per, samples.length); i += 4) {
      const v = Math.abs(samples[i]);
      if (v > peak) peak = v;
    }
    peaks[b] = peak;
  }
  const max = Math.max(0.0001, ...peaks);
  for (let b = 0; b < bins; b++) peaks[b] /= max;
  return peaks;
}

function encodeWav(samples: Float32Array): Blob {
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true);
  view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result ?? "");
      resolve(url.slice(url.indexOf(",") + 1));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

export async function prepareAudio(blob: Blob): Promise<PreparedAudio> {
  const samples = await decodeToMono16k(blob);
  if (samples.length === 0) throw new Error("empty-audio");
  let peak = 0;
  for (let i = 0; i < samples.length; i += 7) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  if (peak > 0) for (let i = 0; i < samples.length; i++) samples[i] /= peak;
  const wavBase64 = await blobToBase64(encodeWav(samples));
  return {
    samples,
    sampleRate: 16000,
    peaks: toPeaks(samples),
    durationSec: samples.length / 16000,
    wavBase64,
  };
}
