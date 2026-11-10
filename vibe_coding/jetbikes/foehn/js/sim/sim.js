// Fixed-step simulation driver: physics at 600 Hz, flight computer at 120 Hz.
import { Bike } from './bike.js';
import { FCS } from './fcs.js';
import { calm } from './wind.js';

export const PHYS_HZ = 600, SUB = 5;   // 5 physics steps per control tick
const H = 1 / PHYS_HZ;

export class Sim {
  constructor(wind = calm) {
    this.bike = new Bike();
    this.fcs = new FCS(this.bike);
    this.wind = wind;
    this.windW = [0, 0, 0];
    this.acc = 0; this.t = 0;
  }
  reset(pos, yaw, speed = 0) { this.bike.reset(pos, yaw, speed); this.fcs.reset(); this.acc = 0; }
  // one control tick (1/120 s)
  tick(inp) {
    const b = this.bike;
    this.windW = this.wind(b.p, this.t, b.agl);
    const cmd = this.fcs.update(SUB * H, inp);
    for (let i = 0; i < SUB; i++) b.step(H, cmd, this.windW);
    this.t += SUB * H;
  }
  // advance by a variable frame time; returns number of ticks run
  advance(dt, inp, onTick) {
    this.acc += Math.min(dt, 0.1);
    let n = 0;
    while (this.acc >= SUB * H) { this.tick(inp); this.acc -= SUB * H; n++; if (onTick) onTick(); }
    return n;
  }
}
