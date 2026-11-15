// Turns the replay on screen into a video file. Every drawn frame is copied onto a 16:9 canvas
// (the picture between the black bars, a caption under it).
//
// StepFilm is the real thing: the film is made frame by frame, not in real time. Each frame is
// handed to an H.264 encoder and stamped with its own time (1/60 s after the one before), so the
// file is smooth at 60 frames a second however slow the computer is, and the sound track is
// rendered afterwards from a tape of the sounds (audio.js). The result is a plain MP4 that
// Facebook, WhatsApp and phones take as it is.
//
// LiveFilm is the fallback for browsers without WebCodecs: it records the canvas as it plays.
import { Muxer, ArrayBufferTarget } from '../../lib/mp4-muxer.mjs';

export const FPS = 60;
const RATE = 48000;
const SIZES = { big: [1920, 1080, 9e6], small: [1280, 720, 5e6] };
const AVC = ['avc1.64002a', 'avc1.4d002a', 'avc1.42002a'];           // high, main, baseline: level 4.2 carries 1080p at 60
const LIVE = ['video/mp4;codecs=avc1.640028,mp4a.40.2', 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];

// What this browser can do: { step: { big, small, audio } } or { live: mime } or null. Asked once.
let caps;
export function filmCaps() {
  return caps || (caps = (async () => {
    try {
      if (typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined') {
        const step = {};
        for (const k of ['big', 'small']) {
          const [width, height, bitrate] = SIZES[k];
          for (const codec of AVC) {
            const s = await VideoEncoder.isConfigSupported({ codec, width, height, bitrate, framerate: FPS, avc: { format: 'avc' } }).catch(() => null);
            if (s && s.supported) { step[k] = codec; break; }
          }
        }
        if (typeof AudioEncoder !== 'undefined') {
          for (const [name, codec] of [['aac', 'mp4a.40.2'], ['opus', 'opus']]) {      // AAC where the browser has it (not on Linux)
            const s = await AudioEncoder.isConfigSupported({ codec, sampleRate: RATE, numberOfChannels: 2, bitrate: 160e3 }).catch(() => null);
            if (s && s.supported) { step.audio = [name, codec]; break; }
          }
        }
        if (step.small) return { step };
      }
    } catch (e) { /* fall through to the recorder */ }
    const live = typeof MediaRecorder === 'undefined' ? null : LIVE.find((m) => MediaRecorder.isTypeSupported(m));
    return live ? { live } : null;
  })());
}

class Base {
  // src: the game's canvas. caption: { text, rtl } for the lower bar
  constructor(src, caption, big) {
    this.src = src; this.caption = caption;
    const c = this.canvas = document.createElement('canvas'); [c.width, c.height] = SIZES[big ? 'big' : 'small'];
    this.g = c.getContext('2d', { alpha: false });
  }
  // fade: 0 picture .. 1 black
  draw(fade) {
    const g = this.g, W = this.canvas.width, H = this.canvas.height, s = this.src;
    // the same strip of the picture that the black bars leave on screen
    const sh = Math.min(s.height * 0.9, s.width / 2.39), sw = Math.min(s.width, sh * 2.39), dh = Math.round(W * sh / sw), dy = Math.round((H - dh) / 2);
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.drawImage(s, (s.width - sw) / 2, (s.height - sh) / 2, sw, sh, 0, dy, W, dh);
    if (fade > 0) { g.fillStyle = `rgba(0,0,0,${Math.min(1, fade)})`; g.fillRect(0, dy, W, dh); }
    const k = H / 720, y = dy + dh + (H - dy - dh) / 2;
    g.textBaseline = 'middle'; g.fillStyle = 'rgba(255,255,255,.82)';
    g.font = `900 ${22 * k}px system-ui, sans-serif`; g.direction = 'ltr'; g.textAlign = 'left'; g.fillText('FÖHN', 36 * k, y);
    g.font = `600 ${17 * k}px system-ui, sans-serif`; g.direction = this.caption.rtl ? 'rtl' : 'ltr'; g.textAlign = 'right'; g.fillText(this.caption.text, W - 36 * k, y);
  }
}

export class StepFilm extends Base {
  // dur: about how long the film will be (s), to keep a long one from filling the memory
  constructor(src, caption, step, dur) {
    const big = src.width >= 1700 && !!step.big;
    super(src, caption, big);
    this.step = true; this.ext = 'mp4'; this.n = 0; this.err = null; this.audio = step.audio || null;
    const [width, height, top] = SIZES[big ? 'big' : 'small'];
    const bitrate = Math.round(Math.max(2.5e6, Math.min(top, 3.2e9 / Math.max(1, dur))));       // at most some 400 MB
    this.target = new ArrayBufferTarget();
    this.mux = new Muxer({ target: this.target, fastStart: 'in-memory', firstTimestampBehavior: 'offset',
      video: { codec: 'avc', width, height, frameRate: FPS },
      audio: this.audio ? { codec: this.audio[0], sampleRate: RATE, numberOfChannels: 2 } : undefined });
    this.enc = new VideoEncoder({ output: (chunk, meta) => this.mux.addVideoChunk(chunk, meta), error: (e) => { this.err = e; } });
    this.enc.configure({ codec: step[big ? 'big' : 'small'], width, height, bitrate, framerate: FPS, avc: { format: 'avc' }, latencyMode: 'quality' });
  }
  get time() { return this.n / FPS; }                       // the time in the film of the next frame
  get full() { return this.enc.encodeQueueSize > 3; }       // the encoder is behind: draw nothing new until it catches up
  grab(fade) {
    if (this.err || this.enc.state !== 'configured') return;
    this.draw(fade);
    const f = new VideoFrame(this.canvas, { timestamp: Math.round(this.n * 1e6 / FPS), duration: Math.round(1e6 / FPS) });
    this.enc.encode(f, { keyFrame: this.n % (FPS * 2) === 0 }); f.close(); this.n++;
  }
  // sound: an AudioBuffer at 48 kHz, or null. Resolves with the finished file.
  async finish(sound) {
    await this.enc.flush(); this.enc.close();
    if (this.err) throw this.err;
    if (this.audio && sound) {
      let err = null;
      const ae = new AudioEncoder({ output: (chunk, meta) => this.mux.addAudioChunk(chunk, meta), error: (e) => { err = e; } });
      ae.configure({ codec: this.audio[1], sampleRate: RATE, numberOfChannels: 2, bitrate: 160e3 });
      const L = sound.getChannelData(0), R = sound.numberOfChannels > 1 ? sound.getChannelData(1) : L;
      const len = Math.min(sound.length, Math.round(this.time * RATE)), N = 4800;
      for (let o = 0; o < len && !err; o += N) {
        const n = Math.min(N, len - o), data = new Float32Array(n * 2);
        data.set(L.subarray(o, o + n), 0); data.set(R.subarray(o, o + n), n);
        const ad = new AudioData({ format: 'f32-planar', sampleRate: RATE, numberOfFrames: n, numberOfChannels: 2, timestamp: Math.round(o * 1e6 / RATE), data });
        ae.encode(ad); ad.close();
        if (ae.encodeQueueSize > 20) await new Promise((r) => setTimeout(r, 0));
      }
      await ae.flush(); ae.close();
      if (err) throw err;
    }
    this.mux.finalize();
    return new Blob([this.target.buffer], { type: 'video/mp4' });
  }
  cancel() { try { if (this.enc.state !== 'closed') this.enc.close(); } catch (e) { /* already gone */ } }
}

export class LiveFilm extends Base {
  // sound: a MediaStream or null
  constructor(src, caption, mime, sound) {
    super(src, caption, src.width >= 1700);
    this.step = false; this.mime = mime; this.ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const stream = this.canvas.captureStream(60);
    if (sound) for (const tr of sound.getAudioTracks()) stream.addTrack(tr);
    this.rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: SIZES[this.canvas.width > 1300 ? 'big' : 'small'][2], audioBitsPerSecond: 160e3 });
    this.chunks = []; this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.started = false;
  }
  grab(fade) { this.draw(fade); if (!this.started) { this.started = true; this.rec.start(1000); } }
  finish() {
    return new Promise((res) => {
      if (this.rec.state === 'inactive') return res(null);
      this.rec.onstop = () => res(new Blob(this.chunks, { type: this.mime.split(';')[0] }));
      this.rec.stop();
    });
  }
  cancel() { if (this.rec.state !== 'inactive') { this.rec.onstop = null; this.rec.stop(); } this.chunks = []; }
}
