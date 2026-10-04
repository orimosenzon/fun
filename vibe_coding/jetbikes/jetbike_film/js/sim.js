// Runs the whole flight once (deterministically) and records it at 240 Hz.
// Rendering, sound and the HUD only ever read this recording.

import { JetBike, makeWind } from './physics.js';
import { FlightComputer } from './control.js';
import { buildMission, TL } from './mission.js';
import { massProps, FUEL0, G } from './vehicle.js';
import { qSlerp, lerp, len } from './vmath.js';

export const REC_HZ = 240;
const PHYS_HZ = 2000;
const CTRL_DIV = 10; // controller at 200 Hz

// fields recorded per sample
const FIELDS = {
  p: 3, q: 4, v: 3, w: 3, acc: 3, T: 4, N: 4, vanes: 8, mainT: 1, mainN: 1, egt: 5,
  fuel: 1, contact: 1, ref: 3, wind: 3, sigma: 1, airspeed: 1,
};

export function runSimulation() {
  const cg0 = massProps(FUEL0).cg;
  const mission = buildMission(cg0[1] - 0.012);
  const bike = new JetBike([0, 0, 0], 0);
  const fc = new FlightComputer();
  const wind = makeWind();
  const dt = 1 / PHYS_HZ;
  const nRec = Math.ceil(mission.tEnd * REC_HZ) + 1;
  const rec = {};
  let stride = 0;
  const off = {};
  for (const [k, n] of Object.entries(FIELDS)) { off[k] = stride; stride += n; }
  const data = new Float64Array(nRec * stride);

  let mode = 'ground';
  let tTouch = null, tShutdown = null;
  const events = { liftoff: null, touchdown: null, shutdown: null };
  let cmd = fc.cmd;
  let step = 0;
  const W = () => bike.mp.m * G;

  for (let r = 0; r < nRec; r++) {
    const tRec = r / REC_HZ;
    while (bike.t < tRec - 1e-9) {
      const t = bike.t;
      // engine start sequence
      TL.liftStart.forEach((ts, i) => { if (t >= ts && bike.lift[i].phase === 'off' && tShutdown === null) bike.lift[i].start(); });
      if (t >= TL.mainStart && bike.main.phase === 'off' && tShutdown === null) bike.main.start();

      if (step % CTRL_DIV === 0) {
        const ref = mission.sample(t);
        if (mode === 'ground' && t >= TL.liftoff) mode = 'flight';
        if (mode === 'ground') {
          const u = Math.max(0, Math.min(1, (t - TL.spoolUp) / (TL.liftoff - TL.spoolUp)));
          cmd = fc.update(dt * CTRL_DIV, bike, ref, 'ground', lerp(0, 0.93 * W(), u * u * (3 - 2 * u)));
        } else if (mode === 'flight') {
          if (events.liftoff === null && bike.contactN < 1) events.liftoff = t;
          if (t > mission.tTouchRef - 6 && bike.contactN > 0.08 * W()) {
            tTouch ??= t;
            if (t - tTouch > 0.3) { mode = 'landed'; events.touchdown = tTouch; }
          } else tTouch = null;
          cmd = fc.update(dt * CTRL_DIV, bike, ref, 'flight');
        } else if (mode === 'landed') {
          const u = Math.min(1, (t - events.touchdown) / 2.0);
          cmd = fc.update(dt * CTRL_DIV, bike, ref, 'landed', lerp(0.55 * W(), 0, u));
          if (t - events.touchdown > 3.2 && tShutdown === null) {
            tShutdown = t; events.shutdown = t;
            bike.lift.forEach((e) => e.stop()); bike.main.stop();
            mode = 'off';
          }
        } else {
          cmd = fc.update(dt * CTRL_DIV, bike, ref, 'off');
        }
      }
      bike.step(dt, cmd, wind);
      step++;
    }
    // record
    const o = r * stride;
    const put = (k, arr) => { for (let i = 0; i < arr.length; i++) data[o + off[k] + i] = arr[i]; };
    put('p', bike.p); put('q', bike.q); put('v', bike.v); put('w', bike.w); put('acc', bike.acc);
    put('T', bike.lift.map((e) => e.T)); put('N', bike.lift.map((e) => e.N));
    put('vanes', bike.vanes.flat()); put('mainT', [bike.main.T]); put('mainN', [bike.main.N]);
    put('egt', [...bike.lift.map((e) => e.egt), bike.main.egt]);
    put('fuel', [bike.fuel]); put('contact', [bike.contactN]);
    put('ref', mission.sample(tRec).p); put('wind', wind(bike.p, tRec));
    put('sigma', [bike.sigma ?? 1]); put('airspeed', [bike.airspeed ?? 0]);
  }
  return new Recording(data, stride, off, nRec, mission, events);
}

export class Recording {
  constructor(data, stride, off, n, mission, events) {
    Object.assign(this, { data, stride, off, n, mission, events });
    this.duration = (n - 1) / REC_HZ;
  }
  // interpolated state at time t
  at(t) {
    const f = Math.max(0, Math.min(this.n - 1.0001, t * REC_HZ));
    const i = Math.floor(f), u = f - i;
    const o0 = i * this.stride, o1 = o0 + this.stride;
    const d = this.data, off = this.off;
    const get = (k, n) => {
      const out = new Array(n);
      for (let j = 0; j < n; j++) out[j] = lerp(d[o0 + off[k] + j], d[o1 + off[k] + j], u);
      return out;
    };
    const q0 = [0, 1, 2, 3].map((j) => d[o0 + off.q + j]);
    const q1 = [0, 1, 2, 3].map((j) => d[o1 + off.q + j]);
    const vanes = get('vanes', 8);
    return {
      t, p: get('p', 3), q: qSlerp(q0, q1, u), v: get('v', 3), w: get('w', 3), acc: get('acc', 3),
      T: get('T', 4), N: get('N', 4), vanes: [0, 1, 2, 3].map((k) => [vanes[2 * k], vanes[2 * k + 1]]),
      mainT: get('mainT', 1)[0], mainN: get('mainN', 1)[0], egt: get('egt', 5),
      fuel: get('fuel', 1)[0], contact: get('contact', 1)[0], ref: get('ref', 3), wind: get('wind', 3),
      sigma: get('sigma', 1)[0], airspeed: get('airspeed', 1)[0],
    };
  }
  speedAt(t) { return len(this.at(t).v); }
}
