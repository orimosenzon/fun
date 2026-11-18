// MOS 6581 SID: three voices (triangle, sawtooth, pulse, noise, sync, ring
// modulation), ADSR envelopes, the multimode filter, and the readable OSC3/ENV3
// registers that games use as a random number source.
// Clocked in steps of a few CPU cycles and averaged down to the output rate.

const RATE = [9, 32, 63, 95, 149, 220, 267, 313, 392, 977, 1954, 3126, 3907, 11720, 19532, 31251];
const ATTACK = 0, DECAY = 1, RELEASE = 2;
const STEP = 8;   // CPU cycles per internal step

class Voice {
  constructor() { this.reset(); }
  reset() {
    this.freq = 0; this.pw = 0; this.ctrl = 0; this.ad = 0; this.sr = 0;
    this.acc = 0; this.lfsr = 0x7ffff8; this.msbRise = false;
    this.state = RELEASE; this.level = 0; this.rateCount = 0; this.expCount = 0;
  }

  setCtrl(v) {
    const was = this.ctrl;
    this.ctrl = v;
    if ((v & 1) && !(was & 1)) this.state = ATTACK;
    else if (!(v & 1) && (was & 1)) this.state = RELEASE;
    if (v & 8) { this.acc = 0; this.lfsr = 0x7ffff8; }
  }

  clockOsc(n) {
    if (this.ctrl & 8) { this.msbRise = false; return; }
    const prev = this.acc;
    const next = prev + this.freq * n;
    // Noise shifts on each rising edge of accumulator bit 19.
    let shifts = Math.floor((next + 0x80000) / 0x100000) - Math.floor((prev + 0x80000) / 0x100000);
    while (shifts-- > 0) {
      const r = this.lfsr;
      this.lfsr = ((r << 1) | (((r >> 22) ^ (r >> 17)) & 1)) & 0x7fffff;
    }
    this.msbRise = next >= 0x1000000;
    this.acc = next & 0xffffff;
  }

  clockEnv(n) {
    const rate = this.state === ATTACK ? this.ad >> 4 : this.state === DECAY ? this.ad & 15 : this.sr & 15;
    const period = RATE[rate];
    this.rateCount += n;
    while (this.rateCount >= period) {
      this.rateCount -= period;
      if (this.state === ATTACK) {
        if (++this.level >= 255) { this.level = 255; this.state = DECAY; this.expCount = 0; }
        continue;
      }
      const l = this.level;
      const exp = l > 93 ? 1 : l > 54 ? 2 : l > 26 ? 4 : l > 14 ? 8 : l > 6 ? 16 : 30;
      if (++this.expCount < exp) continue;
      this.expCount = 0;
      if (this.state === DECAY) { if (l > (this.sr >> 4) * 17) this.level = l - 1; }
      else if (l > 0) this.level = l - 1;
    }
  }

  // 12-bit waveform output. ringMsb: the modulating voice's top bit.
  wave(ringMsb) {
    const c = this.ctrl, acc = this.acc;
    if (!(c & 0xf0)) return 0;
    let out = 0xfff;
    if (c & 0x10) {
      const flip = ((c & 4) ? (acc ^ ringMsb) : acc) & 0x800000;
      out &= ((flip ? ~acc : acc) >> 11) & 0xfff;
    }
    if (c & 0x20) out &= acc >> 12;
    if (c & 0x40) out &= ((c & 8) || (acc >> 12) >= this.pw) ? 0xfff : 0;
    if (c & 0x80) {
      const r = this.lfsr;
      out &= (((r >> 22) & 1) << 11) | (((r >> 20) & 1) << 10) | (((r >> 16) & 1) << 9) | (((r >> 13) & 1) << 8)
        | (((r >> 11) & 1) << 7) | (((r >> 7) & 1) << 6) | (((r >> 4) & 1) << 5) | (((r >> 2) & 1) << 4);
    }
    return out;
  }
}

export class SID {
  constructor(hz, sampleRate = 44100) {
    this.hz = hz; this.sampleRate = sampleRate;
    this.cyclesPerSample = hz / sampleRate;
    this.voices = [new Voice(), new Voice(), new Voice()];
    this.out = new Float32Array(16384);   // produced samples; the owner drains it
    this.outLen = 0;
    this.reset();
  }

  reset() {
    for (const v of this.voices) v.reset();
    this.reg = new Uint8Array(32);
    this.time = 0;            // CPU cycles clocked so far
    this.sampleAcc = 0;       // cycles gathered towards the next sample
    this.sumDirect = 0; this.sumFilt = 0;
    this.lp = 0; this.bp = 0;
    this.lastWrite = 0;
  }

  read(r) {
    r &= 31;
    if (r === 0x1b) return this.voices[2].wave(this.voices[1].acc) >> 4;
    if (r === 0x1c) return this.voices[2].level;
    if (r === 0x19 || r === 0x1a) return 0xff;   // paddles, not connected
    return this.lastWrite;
  }

  write(r, v) {
    r &= 31;
    this.reg[r] = v; this.lastWrite = v;
    if (r < 21) {
      const voice = this.voices[Math.floor(r / 7)];
      switch (r % 7) {
        case 0: voice.freq = (voice.freq & 0xff00) | v; break;
        case 1: voice.freq = (voice.freq & 0x00ff) | (v << 8); break;
        case 2: voice.pw = (voice.pw & 0xf00) | v; break;
        case 3: voice.pw = (voice.pw & 0x0ff) | ((v & 15) << 8); break;
        case 4: voice.setCtrl(v); break;
        case 5: voice.ad = v; break;
        default: voice.sr = v; break;
      }
    }
  }

  // Bring the chip up to the given CPU cycle count.
  clockTo(cycles) {
    let left = cycles - this.time;
    if (left <= 0) return;
    this.time = cycles;
    const v = this.voices, reg = this.reg;
    while (left > 0) {
      let n = left < STEP ? left : STEP;
      const room = this.cyclesPerSample - this.sampleAcc;
      if (n > room) n = Math.max(1, Math.ceil(room));
      left -= n;

      for (let i = 0; i < 3; i++) { v[i].clockOsc(n); v[i].clockEnv(n); }
      // Hard sync: voice i restarts when voice i-1 wraps.
      for (let i = 0; i < 3; i++) if ((v[i].ctrl & 2) && v[(i + 2) % 3].msbRise) v[i].acc = 0;

      const routing = reg[0x17], modeVol = reg[0x18];
      let direct = 0, filt = 0;
      for (let i = 0; i < 3; i++) {
        const s = (v[i].wave(v[(i + 2) % 3].acc) - 2048) * v[i].level;
        if ((routing >> i) & 1) filt += s;
        else if (i !== 2 || !(modeVol & 0x80)) direct += s;
      }
      this.sumDirect += direct * n; this.sumFilt += filt * n;
      this.sampleAcc += n;
      if (this.sampleAcc >= this.cyclesPerSample) this.emit(modeVol);
    }
  }

  emit(modeVol) {
    const reg = this.reg;
    const scale = 1 / (this.sampleAcc * 2048 * 255 * 3);
    const direct = this.sumDirect * scale, input = this.sumFilt * scale;
    this.sampleAcc -= this.cyclesPerSample;
    this.sumDirect = 0; this.sumFilt = 0;

    // State-variable filter at the output rate. The cutoff curve is a rough
    // fit to a 6581 (about 200 Hz to 12 kHz over the register range).
    const fcReg = (reg[0x15] & 7) | (reg[0x16] << 3);
    const fc = Math.min(200 + fcReg * 5.8, this.sampleRate / 6.5);
    const f = 2 * Math.sin(Math.PI * fc / this.sampleRate);
    const damp = 1 / (0.707 + (reg[0x17] >> 4) / 15 * 1.6);
    const hp = input - this.lp - damp * this.bp;
    this.bp += f * hp;
    this.lp += f * this.bp;
    let mix = direct;
    if (modeVol & 0x10) mix += this.lp;
    if (modeVol & 0x20) mix += this.bp;
    if (modeVol & 0x40) mix += hp;
    mix *= (modeVol & 15) / 15;
    if (this.outLen < this.out.length) this.out[this.outLen++] = mix > 1 ? 1 : mix < -1 ? -1 : mix;
  }

  // Hand over and forget the samples produced so far.
  drain() { const s = this.out.slice(0, this.outLen); this.outLen = 0; return s; }
}
