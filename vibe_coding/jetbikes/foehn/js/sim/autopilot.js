// A demo rider. It uses exactly the rider's controls (turn, climb, target speed, side-step):
// pure pursuit of the event's reference line, a speed limit from the curvature ahead, and a
// hover-and-settle routine for the landing pads. Used for the attract mode, for headless
// testing of every course, and to calibrate the medal times.
import { clamp, wrapPi, yprOf, len } from './vmath.js';
import { V_MAX } from './fcs.js';

export class Autopilot {
  constructor(game, skill = 1) { this.game = game; this.i = 0; this.skill = skill; this.boostT = 0; }
  // after a respawn: pick the line up again at the check-point just passed
  reset() { const g = this.game; this.i = g.cp > 0 ? g.cps[g.cp - 1].pathIndex : 0; this.around = null; }

  update() {
    const g = this.game, b = g.bike, path = g.build.path, p = b.p;
    const out = { turn: 0, pitch: 0, strafe: 0, vt: 0, boost: false, brake: false };
    if (g.state !== 'fly' || path.length < 4) return out;
    // progress along the reference line (never backwards, never past the next check-point's node)
    const cp = g.nextCp;
    const cap = cp ? cp.pathIndex + 3 : path.length - 1;
    // Missed the ring (already beyond its plane, and it is still the next one)? Go round: first back
    // to a point on the ring's axis well before it, then slowly down the axis and through.
    if (this.around && (g.cp !== this.around.cp || !cp || cp.type !== 'ring')) { this.i = g.cp > 0 ? g.cps[g.cp - 1].pathIndex : 0; this.around = null; }
    if (cp && cp.type === 'ring') {
      const n = cp.n, along = (p[0] - cp.p[0]) * n[0] + (p[1] - cp.p[1]) * n[1] + (p[2] - cp.p[2]) * n[2];
      if (along > 6 && (this.around ? this.around.ph === 2 : this.i >= cp.pathIndex - 1)) this.around = { cp: g.cp, ph: 1 };
      if (this.around) {
        // between walls there is no room for a wide circle: turn almost on the spot, a little way back
        const tight = cp.slow && cp.r < 7.5;
        const a = this.around, back = a.ph === 1 ? (tight ? 26 : 230) : Math.max(0, -along - (tight ? 8 : 45));
        const q = [cp.p[0] - n[0] * back, cp.p[1] - n[1] * back + (a.ph === 1 && !tight ? 25 : 0), cp.p[2] - n[2] * back];
        const dx = q[0] - p[0], dz = q[2] - p[2], dh = Math.hypot(dx, dz);
        if (a.ph === 1 && dh < (tight ? 7 : 60)) a.ph = 2;
        const [yaw0] = yprOf(b.q), V0 = len(b.v);
        out.turn = clamp(-2.6 * wrapPi(Math.atan2(-dz, dx) - yaw0), -1, 1);
        out.vt = tight ? 7 : a.ph === 1 ? 38 : 28;
        out.pitch = clamp((Math.max(V0, 15) * (q[1] - p[1]) / Math.max(dh, 25)) / Math.max(8, 0.62 * V0), -1, 1);
        return out;
      }
    }
    let best = 1e18, bi = this.i;
    for (let k = this.i; k <= Math.min(cap, this.i + 80, path.length - 1); k++) {
      const q = path[k], d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 * 0.3 + (q[2] - p[2]) ** 2;
      if (d < best) { best = d; bi = k; }
    }
    this.i = bi;
    const V = len(b.v), [yaw] = yprOf(b.q);
    const fh = [Math.cos(yaw), -Math.sin(yaw)], rh = [Math.sin(yaw), Math.cos(yaw)];

    // ----- landing -----
    if (cp && cp.type === 'land') {
      const dx = cp.p[0] - p[0], dz = cp.p[2] - p[2], d = Math.hypot(dx, dz);
      if (d < 110 && path[bi].s > cp.s - 160) {
        const hAbove = p[1] - b.mp.cg[1] - cp.p[1];
        // velocity command towards the pad, slowing as it gets close
        const sp = Math.min(22, 0.42 * d + (d > 3 ? 0.8 : 0));
        const wx = dx / (d || 1) * sp, wz = dz / (d || 1) * sp;
        out.vt = clamp(wx * fh[0] + wz * fh[1], -7, 22);
        out.strafe = clamp((wx * rh[0] + wz * rh[1]) / 7, -1, 1);
        const hv = Math.hypot(b.v[0], b.v[2]);
        if (d < 2.5 && hv < 1.5) out.pitch = -0.5;                           // settle
        else out.pitch = clamp(0.3 * (20 - hAbove), -0.6, 0.8) * (hAbove < 20 || d < 12 ? 1 : 0.3);
        out.brake = V > sp + 8;
        return out;
      }
    }
    // just took off from a pad: climb clear first
    const liftOff = Math.hypot(b.v[0], b.v[2]) < 14 && path[bi][1] - p[1] > 4 && b.agl < 30;   // climb clear of trees before moving off

    // ----- pure pursuit -----
    const vCap = path[bi].v;                     // set inside the fortress
    const look = vCap ? clamp(0.5 * V, 8, 30) : clamp(0.75 * V, 22, 80);
    let k = bi;
    while (k < path.length - 1 && path[k].s - path[bi].s < look) k++;
    const tgt = path[k];
    const dx = tgt[0] - p[0], dz = tgt[2] - p[2], dh = Math.hypot(dx, dz) || 1;
    const err = wrapPi(Math.atan2(-dz, dx) - yaw);
    out.turn = clamp(-2.6 * err, -0.88, 0.88);   // short of full stick: the demo rider plans its own speed, no hairpin
    const vzDes = V * (tgt[1] - p[1]) / Math.max(dh, 20) + 0.9 * (path[bi][1] - p[1]);
    out.pitch = liftOff ? 1 : clamp(vzDes / Math.max(8, 0.62 * V), -1, 1);

    // ----- speed: what the bends ahead allow -----
    const aLat = 8 * this.skill;
    let vLim = V_MAX + 30;
    const hdg = (a, c) => Math.atan2(-(path[c][2] - path[a][2]), path[c][0] - path[a][0]);
    const horizon = 3.0 * V + 60;
    if (vCap) vLim = vCap;
    for (let j = bi; j < path.length - 12 && path[j].s - path[bi].s < horizon; j += 3) {
      if (path[j].v) vLim = Math.min(vLim, Math.sqrt(path[j].v ** 2 + 2 * 7 * Math.max(0, path[j].s - path[bi].s - 0.6 * V)));
      const dpsi = Math.abs(wrapPi(hdg(j + 6, j + 12) - hdg(j, j + 6)));
      const ds = path[j + 9].s - path[j + 3].s;
      const dg = ((path[j + 12][1] - path[j + 6][1]) - (path[j + 6][1] - path[j][1])) / Math.max(ds, 1);
      // a crest can only be followed with what gravity gives (about half a g); a dip can be pulled
      const kappa = Math.max(dpsi, dg < 0 ? -2.6 * dg : 1.1 * dg) / Math.max(ds, 1);
      if (kappa > 1e-4) {
        const vc = Math.sqrt(aLat / kappa), dist = path[j].s - path[bi].s;
        vLim = Math.min(vLim, Math.sqrt(vc * vc + 2 * 7 * Math.max(0, dist - 0.6 * V)));   // braking distance at 7 m/s^2
      }
    }
    // slow down for a landing ahead
    for (let q = g.cp; q < g.cps.length; q++) if (g.cps[q].type === 'land') {
      const d = Math.max(0, g.cps[q].s - path[bi].s - 55); vLim = Math.min(vLim, Math.sqrt(2 * 5 * d) + 16); break;
    }
    if (Math.abs(err) > 0.5) vLim = Math.min(vLim, 30);
    // well below the line (after take-off, or a steep wall ahead): gain height before speed
    const below = path[bi][1] - p[1];
    if (below > 5) vLim = Math.min(vLim, Math.max(3, 45 - 3.5 * below));
    out.vt = Math.min(vLim, V_MAX);
    if (liftOff) out.vt = 0;
    out.boost = this.skill >= 1 && vLim > V_MAX + 10 && V > 40 && b.heat < 0.7 && Math.abs(err) < 0.12;
    out.brake = V > vLim + 14;
    return out;
  }
}
