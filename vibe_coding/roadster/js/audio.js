// קול מסונתז ב-WebAudio, בלי קבצים: יללת המנוע החשמלי והממיר, שאגת המאווררים,
// שריקת הגז הקר, רוח, חריקת צמיגים ומכות.
export class Audio {
  constructor() { this.ctx = null; this.muted = false; }
  start() {
    if (this.ctx) { this.ctx.resume(); return; }
    const C = window.AudioContext || window.webkitAudioContext;
    if (!C) return;
    const ctx = this.ctx = new C();
    this.master = ctx.createGain(); this.master.gain.value = 0.55; this.master.connect(ctx.destination);
    // רעש לבן משותף
    const len = ctx.sampleRate * 2, buf = ctx.createBuffer(1, len, ctx.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const noise = () => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(); return s; };
    const chain = (src, ...nodes) => { let n = src; for (const x of nodes) { n.connect(x); n = x; } return n; };

    // מנוע: שני מתנדים (בסיס וממיר בתדר גבוה)
    this.motor = ctx.createOscillator(); this.motor.type = 'sawtooth';
    this.motor2 = ctx.createOscillator(); this.motor2.type = 'triangle';
    this.motorF = ctx.createBiquadFilter(); this.motorF.type = 'lowpass'; this.motorF.frequency.value = 1200;
    this.motorG = ctx.createGain(); this.motorG.gain.value = 0;
    this.motor.connect(this.motorF); this.motor2.connect(this.motorF);
    chain(this.motorF, this.motorG, this.master);
    this.motor.start(); this.motor2.start();
    // מאווררים
    this.fanF = ctx.createBiquadFilter(); this.fanF.type = 'bandpass'; this.fanF.Q.value = 1.2;
    this.fanG = ctx.createGain(); this.fanG.gain.value = 0;
    chain(noise(), this.fanF, this.fanG, this.master);
    this.fanTone = ctx.createOscillator(); this.fanTone.type = 'sine';
    this.fanToneG = ctx.createGain(); this.fanToneG.gain.value = 0;
    chain(this.fanTone, this.fanToneG, this.master); this.fanTone.start();
    // גז קר: רעש רחב-סרט ועמוק
    this.jetF = ctx.createBiquadFilter(); this.jetF.type = 'lowpass'; this.jetF.frequency.value = 3000;
    this.jetG = ctx.createGain(); this.jetG.gain.value = 0;
    chain(noise(), this.jetF, this.jetG, this.master);
    this.jetLowF = ctx.createBiquadFilter(); this.jetLowF.type = 'lowpass'; this.jetLowF.frequency.value = 160;
    this.jetLowG = ctx.createGain(); this.jetLowG.gain.value = 0;
    chain(noise(), this.jetLowF, this.jetLowG, this.master);
    // רוח
    this.windF = ctx.createBiquadFilter(); this.windF.type = 'lowpass'; this.windF.frequency.value = 500;
    this.windG = ctx.createGain(); this.windG.gain.value = 0;
    chain(noise(), this.windF, this.windG, this.master);
    // צמיגים
    this.squealF = ctx.createBiquadFilter(); this.squealF.type = 'bandpass'; this.squealF.frequency.value = 900; this.squealF.Q.value = 8;
    this.squealG = ctx.createGain(); this.squealG.gain.value = 0;
    chain(noise(), this.squealF, this.squealG, this.master);
    // מילוי אוויר
    this.fillF = ctx.createBiquadFilter(); this.fillF.type = 'highpass'; this.fillF.frequency.value = 4000;
    this.fillG = ctx.createGain(); this.fillG.gain.value = 0;
    chain(noise(), this.fillF, this.fillG, this.master);
    this._noise = noise;
  }
  setMuted(m) { this.muted = m; if (this.master) this.master.gain.value = m ? 0 : 0.55; }
  set(param, v, tc = 0.05) { if (this.ctx) param.setTargetAtTime(v, this.ctx.currentTime, tc); }

  update(car, view, dt) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const inside = view === 'cockpit' ? 0.7 : 1;
    const wAvg = Math.abs(car.wheels[2].omega + car.wheels[3].omega) / 2;
    const motorRpm = wAvg * 9; // יחס העברה ~9:1
    this.set(this.motor.frequency, Math.min(4000, 40 + motorRpm * 0.55));
    this.set(this.motor2.frequency, Math.min(9000, 300 + motorRpm * 0.6));
    const load = Math.min(1, car.motorPower / 6e5);
    this.set(this.motorG.gain, (0.02 + load * 0.12 + Math.min(0.06, motorRpm / 30000)) * inside);
    this.set(this.motorF.frequency, Math.min(12000, 600 + motorRpm * 0.5 + load * 2000));
    // מאווררים: הספק ∝ n³, צליל ∝ n
    const n = car.fan.n;
    this.set(this.fanF.frequency, 200 + n * 2200);
    this.set(this.fanG.gain, n * n * 0.22 * inside);
    this.set(this.fanTone.frequency, 120 + n * 1400);
    this.set(this.fanToneG.gain, n * n * 0.025);
    // סילונים
    const T = car.thrustTotal / 80000;
    this.set(this.jetG.gain, Math.min(0.9, T * 0.8), 0.02);
    this.set(this.jetLowG.gain, Math.min(1.2, T * 1.4), 0.02);
    this.set(this.jetF.frequency, 1500 + T * 5000, 0.03);
    // רוח
    const v = car.speed;
    this.set(this.windG.gain, Math.min(0.5, (v / 90) ** 2 * 0.5) * (view === 'cockpit' ? 0.5 : 1));
    this.set(this.windF.frequency, 300 + v * 15);
    // חריקה
    let slip = 0; for (const w of car.wheels) if (w.contact) slip = Math.max(slip, Math.abs(w.kappa) * 2 + Math.abs(w.alpha) * 3);
    this.set(this.squealG.gain, Math.max(0, Math.min(0.25, (slip - 0.35) * 0.4)) * Math.min(1, v / 5));
    this.set(this.fillG.gain, car._refilling ? 0.08 : 0);
    // מכות
    for (const e of car.events) if (e.type === 'impact') this.thump(Math.min(1, e.speed / 12));
  }
  thump(a) {
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.setValueAtTime(90, ctx.currentTime); o.frequency.exponentialRampToValueAtTime(35, ctx.currentTime + 0.25);
    g.gain.setValueAtTime(0.6 * a, ctx.currentTime); g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    o.connect(g); g.connect(this.master); o.start(); o.stop(ctx.currentTime + 0.4);
  }
  beep(f = 880, dur = 0.12, vol = 0.2) {
    if (!this.ctx) return;
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = f; g.gain.setValueAtTime(vol, ctx.currentTime); g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
    o.connect(g); g.connect(this.master); o.start(); o.stop(ctx.currentTime + dur + 0.02);
  }
}
