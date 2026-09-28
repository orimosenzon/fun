// Real-time engine sound (Web Audio, no samples), same model as the film's soundtrack:
// per engine a turbine whine at the shaft frequency (+harmonics) and a band of jet roar whose
// level climbs steeply with thrust; plus wind rush, jet-on-ground hiss / water splash,
// distance attenuation and Doppler shift when the camera is not riding with the bike.

export class JetAudio {
  constructor() { this.ctx = null; this.muted = false; }

  start() {
    if (this.ctx) { this.ctx.resume(); return; }
    const ctx = (this.ctx = new (window.AudioContext || window.webkitAudioContext)());
    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const noise = () => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.loopStart = Math.random(); s.start(); return s; };
    this.master = ctx.createGain(); this.master.gain.value = 0.7;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -16; comp.ratio.value = 4;
    this.master.connect(comp).connect(ctx.destination);
    this.engines = [];
    for (let k = 0; k < 5; k++) {
      const pan = ctx.createStereoPanner();
      pan.connect(this.master);
      const whineG = ctx.createGain(); whineG.gain.value = 0;
      const oscs = [1, 2, 3].map((h, i) => {
        const o = ctx.createOscillator(); o.type = i === 0 ? 'triangle' : 'sine';
        const og = ctx.createGain(); og.gain.value = [0.5, 0.22, 0.1][i];
        o.connect(og).connect(whineG); o.start();
        return { o, h };
      });
      whineG.connect(pan);
      const roarSrc = noise();
      const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 0.6; bp.frequency.value = 500;
      const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 3000;
      const roarG = ctx.createGain(); roarG.gain.value = 0;
      roarSrc.connect(bp).connect(lp).connect(roarG).connect(pan);
      this.engines.push({ oscs, whineG, roarG, bp, lp, pan, shaft: k < 4 ? 1671 * (1 + 0.004 * (k - 2)) : 1416 });
    }
    const mk = (type, f, q = 0.7) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; b.Q.value = q; return b; };
    this.windF = mk('lowpass', 400); this.windG = ctx.createGain(); this.windG.gain.value = 0;
    noise().connect(this.windF).connect(this.windG).connect(this.master);
    this.hissF = mk('highpass', 1200); this.hissG = ctx.createGain(); this.hissG.gain.value = 0;
    noise().connect(this.hissF).connect(mk('lowpass', 6000)).connect(this.hissG).connect(this.master);
    this.splashF = mk('bandpass', 900, 0.5); this.splashG = ctx.createGain(); this.splashG.gain.value = 0;
    noise().connect(this.splashF).connect(this.splashG).connect(this.master);
    this.rumbleF = mk('lowpass', 110); this.rumbleG = ctx.createGain(); this.rumbleG.gain.value = 0;
    noise().connect(this.rumbleF).connect(this.rumbleG).connect(this.master);
    this.noiseBuf = buf;
  }

  setMuted(m) { this.muted = m; if (this.master) this.master.gain.setTargetAtTime(m ? 0 : 0.7, this.ctx.currentTime, 0.05); }

  // s: {engines:[{N,T,Tmax,pos,dir}], camPos, camRight, bikeVel, onboard, airspeed, blast, wet}
  update(s) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const now = this.ctx.currentTime, tc = 0.04;
    let loud = 0;
    s.engines.forEach((e, k) => {
      const E = this.engines[k];
      const dx = e.pos[0] - s.camPos[0], dy = e.pos[1] - s.camPos[1], dz = e.pos[2] - s.camPos[2];
      const r = Math.max(1.5, Math.hypot(dx, dy, dz));
      // Doppler for a listener that is not moving with the bike
      const vr = s.onboard ? 0 : (s.bikeVel[0] * dx + s.bikeVel[1] * dy + s.bikeVel[2] * dz) / r - (s.camVel[0] * dx + s.camVel[1] * dy + s.camVel[2] * dz) / r;
      const dop = 343 / (343 + Math.max(-150, Math.min(150, vr)));
      const att = Math.min(1, 6 / r);
      const frac = Math.max(0, e.T / e.Tmax);
      const f0 = E.shaft * e.N * dop;
      E.oscs.forEach(({ o, h }) => o.frequency.setTargetAtTime(Math.max(20, f0 * h), now, tc));
      const whine = Math.min(1, e.N / 0.34) ** 1.5 * (0.05 + 0.08 * e.N) * att;
      const roar = (0.02 + 0.5 * frac ** 1.6) * att * (k === 4 ? 1.1 : 0.75);
      E.whineG.gain.setTargetAtTime(whine, now, tc);
      E.roarG.gain.setTargetAtTime(roar, now, tc);
      E.bp.frequency.setTargetAtTime((250 + 900 * frac) * dop, now, tc);
      E.lp.frequency.setTargetAtTime((1500 + 4000 * frac) * dop * (0.3 + 0.7 * Math.exp(-r / 300)), now, tc);
      const pan = (dx * s.camRight[0] + dy * s.camRight[1] + dz * s.camRight[2]) / r;
      E.pan.pan.setTargetAtTime(Math.max(-1, Math.min(1, pan * 0.8)), now, tc);
      loud += roar;
    });
    const V = s.airspeed;
    this.windG.gain.setTargetAtTime(Math.min(0.5, (V / 55) ** 2 * (s.onboard ? 0.45 : 0.18)), now, 0.1);
    this.windF.frequency.setTargetAtTime(250 + V * 18, now, 0.1);
    this.hissG.gain.setTargetAtTime(Math.min(0.5, s.blast * (1 - s.wet) * 0.35), now, 0.05);
    this.splashG.gain.setTargetAtTime(Math.min(0.6, s.blast * s.wet * 0.5), now, 0.05);
    this.rumbleG.gain.setTargetAtTime(Math.min(0.6, s.fuelFlow * 1.2), now, 0.1);
  }

  silenceEngines() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.engines.forEach((E) => { E.whineG.gain.setTargetAtTime(0, now, 0.05); E.roarG.gain.setTargetAtTime(0, now, 0.05); });
  }

  boom(water = false) {
    if (!this.ctx) return;
    const ctx = this.ctx, now = ctx.currentTime;
    const s = ctx.createBufferSource(); s.buffer = this.noiseBuf;
    const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(water ? 2500 : 1800, now); f.frequency.exponentialRampToValueAtTime(90, now + 2.5);
    const g = ctx.createGain(); g.gain.setValueAtTime(water ? 0.9 : 1.6, now); g.gain.exponentialRampToValueAtTime(0.001, now + (water ? 2 : 3.5));
    s.connect(f).connect(g).connect(this.master); s.start(now); s.stop(now + 4);
    if (!water) {
      const o = ctx.createOscillator(); o.frequency.setValueAtTime(70, now); o.frequency.exponentialRampToValueAtTime(28, now + 0.8);
      const og = ctx.createGain(); og.gain.setValueAtTime(1.2, now); og.gain.exponentialRampToValueAtTime(0.001, now + 1.2);
      o.connect(og).connect(this.master); o.start(now); o.stop(now + 1.3);
    }
  }

  chime(up = 0) {
    if (!this.ctx) return;
    const ctx = this.ctx, now = ctx.currentTime;
    [0, 4, 7].forEach((st, i) => {
      const o = ctx.createOscillator(); o.type = 'sine';
      o.frequency.value = 660 * 2 ** ((st + up) / 12);
      const g = ctx.createGain(); g.gain.setValueAtTime(0, now + i * 0.06);
      g.gain.linearRampToValueAtTime(0.18, now + i * 0.06 + 0.01); g.gain.exponentialRampToValueAtTime(0.001, now + i * 0.06 + 0.5);
      o.connect(g).connect(this.master); o.start(now + i * 0.06); o.stop(now + i * 0.06 + 0.6);
    });
  }

  beep() {
    if (!this.ctx) return;
    const ctx = this.ctx, now = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = 'square'; o.frequency.value = 1150;
    const g = ctx.createGain(); g.gain.setValueAtTime(0.06, now); g.gain.setValueAtTime(0, now + 0.12);
    o.connect(g).connect(this.master); o.start(now); o.stop(now + 0.15);
  }
}
