// Procedural jet sound: looped white noise through a low-pass filter (rumble)
// plus a sawtooth oscillator (turbine whine). Both track total jet power.
export class JetAudio {
  constructor() {
    this.ok = false;
    this.muted = false;
  }

  init() {
    if (this.ok) return;
    const C = window.AudioContext || window.webkitAudioContext;
    if (!C) return;
    const ctx = this.ctx = new C();

    const len = 2 * ctx.sampleRate;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 300;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0;
    src.connect(this.filter).connect(this.gain).connect(ctx.destination);
    src.start();

    this.osc = ctx.createOscillator();
    this.osc.type = 'sawtooth';
    this.osc.frequency.value = 90;
    this.wGain = ctx.createGain();
    this.wGain.gain.value = 0;
    this.osc.connect(this.wGain).connect(ctx.destination);
    this.osc.start();

    this.ok = true;
  }

  // power ~0..1.6, speed in m/s
  update(power, speed) {
    if (!this.ok || this.muted) return;
    const t = this.ctx.currentTime;
    this.gain.gain.setTargetAtTime(0.06 + power * 0.2, t, 0.08);
    this.filter.frequency.setTargetAtTime(220 + power * 1500 + speed * 8, t, 0.1);
    this.wGain.gain.setTargetAtTime(0.008 + power * 0.035, t, 0.1);
    this.osc.frequency.setTargetAtTime(85 + power * 150 + speed * 1.2, t, 0.15);
  }

  beep() {
    if (!this.ok || this.muted) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(880, t);
    o.frequency.exponentialRampToValueAtTime(1420, t + 0.12);
    g.gain.setValueAtTime(0.25, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
    o.connect(g).connect(ctx.destination);
    o.start(t);
    o.stop(t + 0.3);
  }

  // RCS pulse: a short pressurized-gas "chuff"
  pulse() {
    if (!this.ok || this.muted) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const dur = 0.16;
    const buf = ctx.createBuffer(1, dur * ctx.sampleRate, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass';
    f.frequency.value = 1900;
    f.Q.value = 0.8;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.35, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(f).connect(g).connect(ctx.destination);
    src.start(t);
  }

  // crash explosion: filtered noise burst sweeping down + a sub-bass thump
  boom() {
    if (!this.ok || this.muted) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const dur = 1.3;
    const buf = ctx.createBuffer(1, dur * ctx.sampleRate, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.setValueAtTime(950, t);
    f.frequency.exponentialRampToValueAtTime(55, t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.85, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(f).connect(g).connect(ctx.destination);
    src.start(t);
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(72, t);
    o.frequency.exponentialRampToValueAtTime(34, t + 0.5);
    const og = ctx.createGain();
    og.gain.setValueAtTime(0.5, t);
    og.gain.exponentialRampToValueAtTime(0.001, t + 0.65);
    o.connect(og).connect(ctx.destination);
    o.start(t);
    o.stop(t + 0.7);
  }

  toggleMute() {
    this.muted = !this.muted;
    if (this.ok && this.muted) {
      const t = this.ctx.currentTime;
      this.gain.gain.setTargetAtTime(0, t, 0.05);
      this.wGain.gain.setTargetAtTime(0, t, 0.05);
    }
    return this.muted;
  }
}
