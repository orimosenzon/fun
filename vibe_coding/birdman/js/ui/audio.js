// All sound is synthesised live: the wind over his ears, the wingbeat, the whump of the wings opening,
// a glider pilot's vario that chirps when the air is rising, and a few chimes.
export class Audio {
  constructor() { this.ctx = null; this.muted = false; this.vol = 0.8; this.varioT = 0; this.varioOn = true; }
  start() {
    if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    const c = (this.ctx = new AC());
    this.master = c.createGain(); this.master.gain.value = this.muted ? 0 : this.vol;
    const comp = c.createDynamicsCompressor(); comp.threshold.value = -16; comp.ratio.value = 4;
    this.master.connect(comp); comp.connect(c.destination);
    const len = c.sampleRate * 3, buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
    let b0 = 0; for (let i = 0; i < len; i++) { const w = Math.random() * 2 - 1; b0 = 0.98 * b0 + 0.02 * w; d[i] = w * 0.4 + b0 * 7; }
    this.noiseBuf = buf;
    const chain = (type, f, q) => { const s = c.createBufferSource(); s.buffer = buf; s.loop = true; s.start(); const fl = c.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q; const g = c.createGain(); g.gain.value = 0; s.connect(fl); fl.connect(g); g.connect(this.master); return { f: fl, g }; };
    this.wind = chain('bandpass', 400, 0.5);
    this.rush = chain('highpass', 1800, 0.4);
    this.flap = chain('lowpass', 500, 0.8);
    this.sea = chain('lowpass', 420, 0.3);
  }
  setMuted(m) { this.muted = m; if (this.master) this.master.gain.value = m ? 0 : this.vol; }
  frame(bird, dt, camY) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime, set = (p, v, k = 0.08) => p.setTargetAtTime(v, t, k);
    const air = bird.mode === 'air', V = air ? bird.airspeed : bird.airspeed * 0.4;
    set(this.wind.g.gain, Math.min(0.55, (V * V) / 5200) + 0.012); set(this.wind.f.frequency, 220 + V * 13); this.wind.f.Q.value = 0.5 + bird.stall * 1.5;
    set(this.rush.g.gain, Math.min(0.3, Math.max(0, V - 28) / 160));
    // the wingbeat: loud on the way down
    const beat = Math.max(0, -bird.L.phiDot) * bird.fold;
    set(this.flap.g.gain, Math.min(0.7, beat * 0.2), 0.03); set(this.flap.f.frequency, 260 + beat * 170, 0.03);
    set(this.sea.g.gain, 0.10 * Math.exp(-Math.max(0, camY) / 140) * (0.75 + 0.25 * Math.sin(t * 0.7)), 0.3);
    // vario: only when soaring, not when he is doing the work himself
    if (this.varioOn && air && bird.amp < 0.1 && bird.fold > 0.8 && bird.climb > 0.5) {
      this.varioT -= dt;
      if (this.varioT <= 0) { const k = Math.min(bird.climb, 6); this.blip(620 + k * 150, 0.07 + 0.02 * (6 - k) * 0.3, 'sine', 0.045); this.varioT = 0.42 - k * 0.05; }
    } else this.varioT = 0;
  }
  blip(freq, dur = 0.12, type = 'sine', gain = 0.2, slide = 0, delay = 0) {
    if (!this.ctx) return;
    const c = this.ctx, t = c.currentTime + delay, o = c.createOscillator(), g = c.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t); if (slide) o.frequency.exponentialRampToValueAtTime(freq * slide, t + dur);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + 0.008); g.gain.exponentialRampToValueAtTime(0.0008, t + dur);
    o.connect(g); g.connect(this.master); o.start(t); o.stop(t + dur + 0.02);
  }
  burst(dur, f0, f1, gain, type = 'lowpass') {
    if (!this.ctx) return;
    const c = this.ctx, t = c.currentTime, s = c.createBufferSource(); s.buffer = this.noiseBuf; s.loopStart = Math.random() * 2; s.loop = true;
    const f = c.createBiquadFilter(); f.type = type; f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const g = c.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + 0.02); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    s.connect(f); f.connect(g); g.connect(this.master); s.start(t); s.stop(t + dur + 0.05);
  }
  // the wings catching the air: a canvas whump, deeper and louder the faster he was falling
  spread(speed) { const k = Math.min(1, speed / 60); this.burst(0.5 + 0.5 * k, 900 + 1500 * k, 90, 0.35 + 0.6 * k); this.blip(95 - 30 * k, 0.5, 'sine', 0.3 + 0.5 * k, 0.5); }
  tuck() { this.burst(0.25, 500, 1800, 0.2, 'bandpass'); }
  chime(last) { [0, 7, 12].forEach((s, i) => this.blip(587.33 * Math.pow(2, s / 12), 0.5, 'triangle', 0.16, 0, i * 0.07)); if (last) [0, 4, 7, 12, 16].forEach((s, i) => this.blip(392 * Math.pow(2, s / 12), 0.9, 'triangle', 0.17, 0, 0.3 + i * 0.12)); }
  land(q) { this.burst(0.25, 500, 120, 0.4); this.blip(80, 0.15, 'sine', 0.3); }
  crash() { this.burst(0.9, 1800, 80, 0.9); this.blip(60, 0.6, 'sine', 0.7, 0.5); }
  splash() { this.burst(1.3, 4000, 400, 0.7, 'bandpass'); }
  step() { this.burst(0.07, 300, 120, 0.12); }
  click() { this.blip(700, 0.05, 'triangle', 0.1); }
}
