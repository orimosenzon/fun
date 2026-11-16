// All sound is synthesised live with Web Audio: turbine whine, jet roar, after-burner,
// wind, and the game's cues. No samples.
export class Audio {
  // ctx: only for rendering a film's sound track (see renderSound below); the game makes its own on the first key
  constructor(ctx = null) {
    this.ctx = null; this.muted = false; this.vol = 0.8; this.streak = 0;
    this.tape = null; this.tapeT = 0;      // while a film is recorded: every sound that is made, with its time in the film
    this.at = -1;                          // a fixed time to play at, instead of now
    if (ctx) this.build(ctx);
  }
  start() {
    if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
    this.build(new AC());
  }
  now() { return this.at >= 0 ? this.at : this.ctx.currentTime; }
  note(name, args) { if (this.tape) this.tape.push([this.tapeT, name, args]); }
  build(c) {
    this.ctx = c;
    this.master = c.createGain(); this.master.gain.value = this.muted ? 0 : this.vol;
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
    this.sidej = chain(noise(), 'bandpass', 3400, 1.3);       // side nozzles
    this.rain = chain(noise(), 'highpass', 2200, 0.5);
    this.gale = chain(noise(), 'bandpass', 320, 2.2);         // the wind itself, heard when it gusts
    this.whine = [0, 1].map((i) => { const o = c.createOscillator(); o.type = 'sawtooth'; o.start(); const r = chain(o, 'bandpass', 2000, 6); r.o = o; r.det = 1 + i * 0.013; return r; });
    this.sub = c.createOscillator(); this.sub.type = 'sine'; this.sub.frequency.value = 55; this.sub.start();
    this.subG = c.createGain(); this.subG.gain.value = 0; this.sub.connect(this.subG); this.subG.connect(this.master);
  }
  // the sound of the game as a stream, for recording a film (null before the first key or tap)
  tap() { if (!this.ctx) return null; if (!this._tap) { this._tap = this.ctx.createMediaStreamDestination(); this.master.connect(this._tap); } return this._tap.stream; }
  setMuted(m) { this.muted = m; if (this.master) this.master.gain.value = m ? 0 : this.vol; }
  // every frame
  engine(b, alive, camDist = 8, cockpit = false) { this.eng(b.eng.N, b.thrust / 4, b.abOut, Math.abs(b.sd[0]) + Math.abs(b.sd[1]), b.kick, b.airspeed, alive, camDist, cockpit); }
  eng(N, T, ab, sd, kick, V, alive, camDist, cockpit) {
    this.note('eng', [N, T, ab, sd, kick, V, alive, camDist, cockpit]);
    if (!this.ctx) return;
    const t = this.now(), set = (p, v, k = 0.06) => p.setTargetAtTime(v, t, k);
    // one big engine: its spool speed sets the pitch of the whine, the thrust of all five nozzles the roar
    const th = alive ? Math.min(1.5, T / 1400) : 0, att = 1 / (1 + camDist / 60), ck = cockpit ? 0.7 : 1;
    set(this.roar.g.gain, th * 0.6 * att * ck); set(this.roar.f.frequency, 150 + th * 800);
    set(this.hiss.g.gain, th * 0.10 * att); set(this.hiss.f.frequency, 1800 + N * 2500);
    for (const w of this.whine) { set(w.o.frequency, (320 + N * 2100) * w.det); set(w.f.frequency, 320 + N * 2100); set(w.g.gain, alive ? (0.012 + 0.03 * N) * att : 0); }
    set(this.abn.g.gain, ab * (1.0 + 0.3 * Math.random()) * att, 0.03); set(this.subG.gain, ab * 0.4 * att + th * 0.1 * att);
    const sj = alive ? Math.min(1, sd / 3000) : 0;
    set(this.sidej.g.gain, sj * (0.16 + 0.3 * kick) * att, 0.02); set(this.sidej.f.frequency, 3000 + 1500 * kick);
    set(this.wind.g.gain, alive ? Math.min(0.5, V * V / 22000) * (cockpit ? 1.3 : 0.8) : 0, 0.2); set(this.wind.f.frequency, 300 + V * 14);
  }
  // rain on the helmet (0..1) and the wind that is blowing right now (m/s)
  weather(rain, wind, on) {
    this.note('weather', [rain, wind, on]);
    if (!this.ctx) return;
    const t = this.now();
    this.rain.g.gain.setTargetAtTime(on ? rain * 0.10 : 0, t, 0.3);
    this.gale.g.gain.setTargetAtTime(on ? Math.min(0.55, wind * wind / 900) : 0, t, 0.15); this.gale.f.frequency.setTargetAtTime(240 + wind * 14, t, 0.2);
  }
  blip(freq, dur = 0.12, type = 'sine', gain = 0.2, slide = 0, delay = 0) {
    this.note('blip', [freq, dur, type, gain, slide, delay]);
    if (!this.ctx) return;
    const c = this.ctx, t = this.now() + delay, o = c.createOscillator(), g = c.createGain();
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
  // noise through a filter that closes: the body of a shot, a pop, a splash
  burst(dur, f0, f1, gain, pow = 3) {
    this.note('burst', [dur, f0, f1, gain, pow]);
    if (!this.ctx) return;
    const c = this.ctx, t = this.now(), n = Math.floor(c.sampleRate * dur), buf = c.createBuffer(1, n, c.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, pow);
    const s = c.createBufferSource(); s.buffer = buf;
    const f = c.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const g = c.createGain(); g.gain.value = gain;
    s.connect(f); f.connect(g); g.connect(this.master); s.start(t);
  }
  shot() { this.burst(0.45, 5200, 220, 1.1, 4); this.blip(120, 0.22, 'sine', 0.7, 0.35); this.blip(900, 0.06, 'square', 0.12, 0.3); }
  pop() { this.burst(0.16, 7000, 900, 0.9, 2); this.blip(420, 0.12, 'triangle', 0.25, 0.4); }
  thud(water) { this.burst(water ? 0.5 : 0.25, water ? 3000 : 1400, water ? 500 : 160, 0.3, 2.5); }
  miss() { this.streak = 0; this.blip(180, 0.3, 'sawtooth', 0.16, 0.6); }
  count(go) { this.blip(go ? 880 : 440, go ? 0.5 : 0.16, 'square', 0.12); }
  click() { this.blip(700, 0.05, 'triangle', 0.1); }
  warn() { this.blip(1250, 0.07, 'square', 0.07); }
  crash(water) { this.streak = 0; this.boom(water); if (!water) this.blip(70, 1.2, 'sine', 0.9, 0.4); }
  boom(water) {
    this.note('boom', [water]);
    if (!this.ctx) return;
    const c = this.ctx, t = this.now(), len = c.sampleRate * 2.2, buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, water ? 1.5 : 3);
    const s = c.createBufferSource(); s.buffer = buf;
    const f = c.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(water ? 3000 : 2400, t); f.frequency.exponentialRampToValueAtTime(water ? 500 : 90, t + 1.6);
    const g = c.createGain(); g.gain.value = water ? 0.6 : 1.3;
    s.connect(f); f.connect(g); g.connect(this.master); s.start(t);
  }
}

// The sound track of a film: the taped sounds played again, each at its own moment, into a
// buffer. Faster than real time and exactly in step with the picture. A long film is rendered a
// minute at a time (thousands of scheduled changes on one context get slow); each minute starts
// a few seconds early so that sounds still ringing are there, and the joins are cross-faded.
export async function renderSound(tape, dur, rate = 48000) {
  const OC = window.OfflineAudioContext || window.webkitOfflineAudioContext; if (!OC || !tape.length) return null;
  const SEG = 60, PRE = 3, XF = 0.03, len = Math.max(rate, Math.ceil(dur * rate)), ch = [new Float32Array(len), new Float32Array(len)];
  for (let S = 0, i0 = 0; S * rate < len; S += SEG) {
    const pre = S ? PRE : 0, from = S - pre, to = S + SEG + XF;
    while (i0 < tape.length && tape[i0][0] < from) i0++;
    const a = new Audio(new OC(2, Math.ceil((to - from) * rate), rate));
    for (let i = i0; i < tape.length && tape[i][0] < to; i++) { a.at = tape[i][0] - from; a[tape[i][1]](...tape[i][2]); }
    const buf = await a.ctx.startRendering(), o = Math.round(S * rate), skip = Math.round(pre * rate), xf = S ? Math.round(XF * rate) : 0, n = Math.min(buf.length - skip, len - o);
    for (let c = 0; c < 2; c++) {
      const src = buf.getChannelData(c), dst = ch[c];
      for (let k = 0; k < n; k++) dst[o + k] = k < xf ? dst[o + k] * (1 - k / xf) + src[skip + k] * (k / xf) : src[skip + k];
    }
  }
  return { length: len, numberOfChannels: 2, getChannelData: (c) => ch[c] };
}
