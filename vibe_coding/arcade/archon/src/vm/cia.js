// MOS 6526 CIA: two timers, interrupt control, time-of-day clock, two ports.
// The owner supplies readA/readB (the pins as seen from outside) and is told
// about port writes through onPortA.

export class CIA {
  constructor(hz) {
    this.hz = hz;
    this.readPinsA = () => 0xff;
    this.readPinsB = () => 0xff;
    this.onPortA = () => {};
    this.reset();
  }

  reset() {
    this.pra = 0; this.prb = 0; this.ddra = 0; this.ddrb = 0;
    this.ta = 0xffff; this.tb = 0xffff; this.la = 0xffff; this.lb = 0xffff;
    this.cra = 0; this.crb = 0;
    this.icr = 0; this.mask = 0; this.irq = false;
    this.tod = 0; this.todFrac = 0; this.todStopped = true; this.alarm = 0; this.todLatch = -1;
  }

  // What the chip drives combined with what the outside pulls low.
  get portA() { return (this.pra | ~this.ddra) & 0xff; }
  get portB() { return (this.prb | ~this.ddrb) & 0xff; }

  raise(bit) {
    this.icr |= bit;
    if (this.icr & this.mask) this.irq = true;
  }

  clock(n) {
    let underA = 0;
    if ((this.cra & 0x21) === 0x01) {
      let t = this.ta - n;
      if (t < 0) {
        if (this.cra & 0x08) { underA = 1; t = this.la; this.cra &= ~1; }
        else { const period = this.la + 1; underA = 1 + Math.floor((-t - 1) / period); t += period * underA; }
        this.raise(1);
      }
      this.ta = t;
    }
    if (this.crb & 1) {
      const count = (this.crb & 0x40) ? underA : (this.crb & 0x20) ? 0 : n;
      if (count) {
        let t = this.tb - count;
        if (t < 0) {
          if (this.crb & 0x08) { t = this.lb; this.crb &= ~1; }
          else { const period = this.lb + 1; t += period * (1 + Math.floor((-t - 1) / period)); }
          this.raise(2);
        }
        this.tb = t;
      }
    }
    if (!this.todStopped) {
      this.todFrac += n * 10;
      while (this.todFrac >= this.hz) {
        this.todFrac -= this.hz;
        this.tod = (this.tod + 1) % 864000;   // tenths of a second in a day
        if (this.tod === this.alarm) this.raise(4);
      }
    }
  }

  todByte(t, which) {
    const bcd = (v) => ((Math.floor(v / 10) << 4) | (v % 10));
    const tenths = t % 10, s = Math.floor(t / 10) % 60, m = Math.floor(t / 600) % 60, h24 = Math.floor(t / 36000);
    if (which === 0) return tenths;
    if (which === 1) return bcd(s);
    if (which === 2) return bcd(m);
    const h12 = h24 % 12 || 12;
    return bcd(h12) | (h24 >= 12 ? 0x80 : 0);
  }

  setTod(which, v) {
    const dec = (b) => (b >> 4) * 10 + (b & 15);
    let t = (this.crb & 0x80) ? this.alarm : this.tod;
    let tenths = t % 10, s = Math.floor(t / 10) % 60, m = Math.floor(t / 600) % 60, h = Math.floor(t / 36000);
    if (which === 0) tenths = v & 15;
    else if (which === 1) s = dec(v & 0x7f) % 60;
    else if (which === 2) m = dec(v & 0x7f) % 60;
    else h = (dec(v & 0x1f) % 12) + (v & 0x80 ? 12 : 0);
    t = tenths + s * 10 + m * 600 + h * 36000;
    if (this.crb & 0x80) this.alarm = t;
    else { this.tod = t; if (which === 3) this.todStopped = true; if (which === 0) this.todStopped = false; }
  }

  read(r) {
    switch (r & 15) {
      // A pin reads low if either side pulls it low, even when it is an output
      // (a joystick switch shorts the line to ground).
      case 0: return this.portA & this.readPinsA();
      case 1: return this.portB & this.readPinsB();
      case 2: return this.ddra;
      case 3: return this.ddrb;
      case 4: return this.ta & 0xff;
      case 5: return this.ta >> 8;
      case 6: return this.tb & 0xff;
      case 7: return this.tb >> 8;
      case 8: { const t = this.todLatch >= 0 ? this.todLatch : this.tod; this.todLatch = -1; return this.todByte(t, 0); }
      case 9: return this.todByte(this.todLatch >= 0 ? this.todLatch : this.tod, 1);
      case 10: return this.todByte(this.todLatch >= 0 ? this.todLatch : this.tod, 2);
      case 11: this.todLatch = this.tod; return this.todByte(this.tod, 3);
      case 12: return 0;
      case 13: { const v = this.icr | (this.irq ? 0x80 : 0); this.icr = 0; this.irq = false; return v; }
      case 14: return this.cra;
      default: return this.crb;
    }
  }

  write(r, v) {
    switch (r & 15) {
      case 0: this.pra = v; this.onPortA(); break;
      case 1: this.prb = v; break;
      case 2: this.ddra = v; this.onPortA(); break;
      case 3: this.ddrb = v; break;
      case 4: this.la = (this.la & 0xff00) | v; break;
      case 5: this.la = (this.la & 0x00ff) | (v << 8); if (!(this.cra & 1)) this.ta = this.la; break;
      case 6: this.lb = (this.lb & 0xff00) | v; break;
      case 7: this.lb = (this.lb & 0x00ff) | (v << 8); if (!(this.crb & 1)) this.tb = this.lb; break;
      case 8: case 9: case 10: case 11: this.setTod((r & 15) - 8, v); break;
      case 12: break;
      case 13:
        if (v & 0x80) this.mask |= v & 0x1f; else this.mask &= ~v;
        if (this.icr & this.mask) this.irq = true;
        break;
      case 14: this.cra = v & 0xef; if (v & 0x10) this.ta = this.la; break;
      default: this.crb = v & 0xef; if (v & 0x10) this.tb = this.lb; break;
    }
  }
}
