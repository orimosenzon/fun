// The animals of the valley: cattle and horses on the low pastures, sheep on the alps, deer at
// the edge of the forest. They graze, look up, and run when the bike comes howling past.
// They are scenery with a little life: nothing collides with them. Free of any DOM, so the
// tests can run it; render/fauna.js draws what this decides.
import { T, groundAt, normalAt, isWater } from './terrain.js';
import { canopyAt } from './scenery.js';
import { hitCastle } from './castle.js';
import { EVENTS } from './courses.js';

// alt: where they live (m). herd: how many together. spread: size of the herd (m). shy: how near
// the bike may come (m). speed: m/s at the gallop. stride: metres per cycle of the gallop.
export const KINDS = {
  cow: { alt: [900, 1750], herd: [5, 11], spread: 21, shy: 34, speed: 6.5, stride: 3.2, n: 13, with: ['bull', 0.12] },
  holstein: { alt: [900, 1500], herd: [5, 10], spread: 19, shy: 34, speed: 6.5, stride: 3.2, n: 10 },
  sheep: { alt: [1250, 2050], herd: [9, 20], spread: 16, shy: 45, speed: 7, stride: 1.9, n: 14 },
  horse: { alt: [900, 1400], herd: [3, 6], spread: 17, shy: 60, speed: 13, stride: 4.6, n: 8, with: ['whitehorse', 0.3] },
  deer: { alt: [900, 1850], herd: [2, 5], spread: 12, shy: 95, speed: 14, stride: 4.2, n: 22, edge: true, with: ['stag', 0.22] },
};
export const POSES = ['stand', 'graze', 'run0', 'run1', 'run2', 'run3'];
const SPEC = { ...KINDS, bull: KINDS.cow, whitehorse: KINDS.horse, stag: KINDS.deer };
const GRAZE = 1, STAND = 0, RUN = 2;

// may an animal stand here?
export function pasture(x, z, kind = KINDS.cow) {
  const h = T.half - 60;
  if (Math.abs(x) > h || Math.abs(z) > h) return false;
  if (isWater(x, z) || normalAt(x, z)[1] < 0.955) return false;                     // steeper than 17 degrees
  if (canopyAt(x, z, true) > -1e8) return false;                                    // trees or a roof in this patch
  const y = groundAt(x, z);
  if (y < kind.alt[0] - 40 || y > kind.alt[1] + 60) return false;
  if (hitCastle([x, y + 1, z], 6)) return false;
  for (const p of Object.values(T.meta.pads)) if (Math.hypot(x - p[0], z - p[2]) < Math.min(p[3], 34) + 6) return false;
  return true;
}

export class Fauna {
  constructor(seed = 20261115) {
    let s = seed >>> 0; const rnd = this.rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const list = [], R = (a, b) => a + (b - a) * rnd();
    // herds like to be where the riders pass: beside the nodes of the events, and here and there in the valley
    const spots = [];
    for (const ev of EVENTS) for (const c of ev.cps()) if (c.x !== undefined && !c.under) spots.push([c.x, c.z]);
    const edgeOK = (x, z) => { for (let a = 0; a < 6.2; a += 0.8) if (canopyAt(x + Math.cos(a) * 45, z + Math.sin(a) * 45, true) > -1e8) return true; return false; };
    for (const [name, K] of Object.entries(KINDS)) {
      let made = 0;
      for (let tries = 0; tries < 6000 && made < K.n; tries++) {
        let x, z;
        if (spots.length && rnd() < 0.6) { const sp = spots[Math.floor(rnd() * spots.length)], a = R(0, 6.283), d = R(45, 260); x = sp[0] + Math.cos(a) * d; z = sp[1] + Math.sin(a) * d; }
        else { x = R(-T.half, T.half); z = R(-T.half, T.half); }
        if (!pasture(x, z, K) || groundAt(x, z) < K.alt[0] || groundAt(x, z) > K.alt[1]) continue;
        if (K.edge && !edgeOK(x, z)) continue;
        if (list.some((o) => o.lead && Math.hypot(o.x - x, o.z - z) < 150)) continue;            // herds keep apart
        const n = Math.round(R(K.herd[0], K.herd[1])), face = R(0, 6.283); let put = 0;
        for (let k = 0; k < n * 6 && put < n; k++) {
          const a = R(0, 6.283), d = Math.sqrt(rnd()) * K.spread, ax = x + Math.cos(a) * d, az = z + Math.sin(a) * d;
          if (!pasture(ax, az, K) || list.some((o) => Math.hypot(o.x - ax, o.z - az) < 3)) continue;
          const kind = K.with && (put === 0 ? name === 'deer' && rnd() < 0.5 : rnd() < K.with[1]) ? K.with[0] : name;
          list.push({ kind, lead: put === 0, herd: made + name, x: ax, z: az, y: groundAt(ax, az), yaw: face + R(-1.3, 1.3), size: R(0.88, 1.1) * (kind === name ? 1 : 1), state: rnd() < 0.7 ? GRAZE : STAND, t: R(2, 14), ph: 0, dir: 0, pose: 0 });
          put++;
        }
        if (put) made++;
      }
    }
    this.list = list;
    this.start = list.map((a) => ({ ...a }));
    for (const a of list) a.pose = a.state === GRAZE ? 1 : 0;
  }
  // back to where everything stood (a new run)
  reset() { this.list.forEach((a, i) => Object.assign(a, this.start[i])); for (const a of this.list) a.pose = a.state === GRAZE ? 1 : 0; }

  // p, v: the bike (world). Animals far from it are left alone.
  update(dt, p, v) {
    if (dt <= 0) return;
    const rnd = this.rnd, sp = Math.hypot(v[0], v[1], v[2]);
    for (const a of this.list) {
      const dx = a.x - p[0], dz = a.z - p[2];
      if (Math.abs(dx) > 700 || Math.abs(dz) > 700) continue;
      const K = SPEC[a.kind], d = Math.hypot(dx, a.y - p[1], dz);
      // a jet at speed is heard from farther off; one that hovers quietly may come closer
      const scare = K.shy * (0.75 + Math.min(0.9, sp / 70));
      if (d < scare && d > 0.5) {
        const away = Math.atan2(-dz, dx);
        if (a.state !== RUN) { a.state = RUN; a.dir = away + (rnd() - 0.5) * 0.9; a.t = 2.5 + rnd() * 3; }
        else { let e = away - a.dir; e = Math.atan2(Math.sin(e), Math.cos(e)); a.dir += e * Math.min(1, dt * 1.2); a.t = Math.max(a.t, 1.5); }
      }
      a.t -= dt;
      if (a.state === RUN) {
        const step = K.speed * Math.min(1, a.t / 0.8 + 0.25) * dt;                    // slowing to a stop at the end
        let moved = false;
        for (const turn of [0, 0.7, -0.7, 1.5, -1.5]) {
          const h = a.dir + turn, nx = a.x + Math.cos(h) * step, nz = a.z - Math.sin(h) * step;
          // look a little ahead as well, or it stops with its nose in the lake
          if (!pasture(nx, nz, RUNS) || !pasture(a.x + Math.cos(h) * 6, a.z - Math.sin(h) * 6, RUNS)) continue;
          a.x = nx; a.z = nz; a.dir = h; moved = true; break;
        }
        a.y = groundAt(a.x, a.z);
        let e = a.dir - a.yaw; e = Math.atan2(Math.sin(e), Math.cos(e)); a.yaw += e * Math.min(1, dt * 7);
        a.ph = (a.ph + step / K.stride) % 1; a.pose = 2 + Math.floor(a.ph * 4);
        if (a.t <= 0 || !moved) { a.state = STAND; a.t = 2 + rnd() * 5; a.pose = 0; }
      } else if (a.t <= 0) {
        a.state = a.state === GRAZE ? (rnd() < 0.6 ? STAND : GRAZE) : GRAZE;
        a.t = a.state === GRAZE ? 5 + rnd() * 14 : 2 + rnd() * 5; a.pose = a.state === GRAZE ? 1 : 0;
      }
    }
  }
}
// where a frightened animal may run: anywhere an animal can stand, whatever the height
const RUNS = { alt: [0, 9000] };
