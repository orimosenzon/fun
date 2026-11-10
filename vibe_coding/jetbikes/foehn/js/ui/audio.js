// All sound is synthesised live with Web Audio: turbine whine, jet roar, after-burner,
// wind, and the game's cues. No samples.
export class Audio {
  constructor() { this.ctx = null; this.muted = false; this.vol = 0.8; this.streak = 0; }
  start() {
    if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    const c = this.ctx = new AC();
    this.master = c.createGain(); this.master.gain.value = this.vol;
    const comp = c.createDynamicsCompressor(); comp.threshold.value = -14; comp.ratio.value = 5;
    this.master.connect(comp); comp.connect(c.destination);
    // noise source
    const len = c.sampleRate * 3, buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
    let b0 = 0; for (let i = 0; i < len; i++) { const w = Math.random() * 2 - 1; b0 = 0.985 * b0 + 0.015 * w; d[i] = w * 0.5 + b0 * 6; }
    const noise = () => { const s = c.createBufferSource(); s.buffer = buf; s.loop = true; s.start(); return s; };
    const chain = (src, type, f, q) => { const fl = c.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q; const g = c.createGain(); g.gain.value = 0; src.connect(fl); fl.connect(g); g.connect(this.master); return { f: fl, g }; };
    this.roar = chain(noise(), 'lowpass', 300, 0.7);
    this.hiss = chain(noise(), 'bandpass', 2600, 0.8);
    this.wind = chain(noise(), 'bandpass', 500, 0.6);
    this.abn = chain(noise(), 'lowpass', 140, 1.2);
    this.whine = [0, 1].map((i) => { const o = c.createOscillator(); o.type = 'sawtooth'; o.start(); const r = chain(o, 'bandpass', 2000, 6); r.o = o; r.det = 1 + i * 0.013; return r; });
    this.sub = c.createOscillator(); this.sub.type = 'sine'; this.sub.frequency.value = 55; this.sub.start();
    this.subG = c.createGain(); this.subG.gain.value = 0; this.sub.connect(this.subG); this.subG.connect(this.master);
  }
  setMuted(m) { this.muted = m; if (this.master) this.master.gain.value = m ? 0 : this.vol; }
  // every frame
  engine(b, alive, camDist = 8, cockpit = false) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime, set = (p, v, k = 0.06) => p.setTargetAtTime(v, t, k);
    let N = 0, T = 0, ab = 0;
    for (const e of b.eng) { N += e.N / 4; T += e.T / 4; ab = Math.max(ab, e.ab); }
    const th = alive ? Math.min(1.5, T / 1400) : 0, att = 1 / (1 + camDist / 60), ck = cockpit ? 0.7 : 1;
    set(this.roar.g.gain, th * 0.55 * att * ck); set(this.roar.f.frequency, 180 + th * 900);
    set(this.hiss.g.gain, th * 0.10 * att); set(this.hiss.f.frequency, 1800 + N * 2500);
    for (const w of this.whine) { set(w.o.frequency, (500 + N * 2900) * w.det); set(w.f.frequency, 500 + N * 2900); set(w.g.gain, alive ? (0.012 + 0.03 * N) * att : 0); }
    set(this.abn.g.gain, ab * (0.9 + 0.3 * Math.random()) * att, 0.03); set(this.subG.gain, ab * 0.35 * att + th * 0.08 * att);
    const V = b.airspeed;
    set(this.wind.g.gain, alive ? Math.min(0.5, V * V / 22000) * (cockpit ? 1.3 : 0.8) : 0, 0.2); set(this.wind.f.frequency, 300 + V * 14);
  }
  blip(freq, dur = 0.12, type = 'sine', gain = 0.2, slide = 0, delay = 0) {
    if (!this.ctx) return;
    const c = this.ctx, t = c.currentTime + delay, o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t); if (slide) o.frequency.exponentialRampToValueAtTime(freq * slide, t + dur);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + 0.008); g.gain.exponentialRampToValueAtTime(0.0008, t + dur);
    o.connect(g); g.connect(this.master); o.start(t); o.stop(t + dur + 0.02);
  }
  gate(perfect, last) {
    const scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24], n = scale[Math.min(this.streak, scale.length - 1)]; this.streak++;
    const f = 523.25 * Math.pow(2, n / 12);
    this.blip(f, 0.35, 'triangle', 0.22); this.blip(f * 1.5, 0.45, 'sine', 0.14, 0, 0.06);
    if (perfect) this.blip(f * 2, 0.5, 'sine', 0.1, 0, 0.12);
    if (last) [0, 4, 7, 12].forEach((s, i) => this.blip(523.25 * Math.pow(2, s / 12), 0.6, 'triangle', 0.2, 0, 0.15 + i * 0.11));
  }
  miss() { this.streak = 0; this.blip(180, 0.3, 'sawtooth', 0.16, 0.6); }
  count(go) { this.blip(go ? 880 : 440, go ? 0.5 : 0.16, 'square', 0.12); }
  click() { this.blip(700, 0.05, 'triangle', 0.1); }
  warn() { this.blip(1250, 0.07, 'square', 0.07); }
  crash(water) {
    if (!this.ctx) return;
    this.streak = 0;
    const c = this.ctx, t = c.currentTime, len = c.sampleRate * 2.2, buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, water ? 1.5 : 3);
    const s = c.createBufferSource(); s.buffer = buf;
    const f = c.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(water ? 3000 : 2400, t); f.frequency.exponentialRampToValueAtTime(water ? 500 : 90, t + 1.6);
    const g = c.createGain(); g.gain.value = water ? 0.6 : 1.3;
    s.connect(f); f.connect(g); g.connect(this.master); s.start();
    if (!water) this.blip(70, 1.2, 'sine', 0.9, 0.4);
  }
}
