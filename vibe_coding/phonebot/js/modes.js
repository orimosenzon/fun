// "המוח": מיפוי, מעקב אחרי החתול, יריב סומו, ולמידת חיזוק.
import { rayNearest, wrapAngle, randomFreePose, insideBox, clearance } from './world.js';
import { BODY_R } from './robot.js';

export const FOV = 70 * Math.PI / 180;
const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

// ---------- מפת תפוסה (occupancy grid) ----------
export class OccGrid {
  constructor(world, res = 0.1) {
    this.res = res; this.x0 = world.x0; this.y0 = world.y0;
    this.nx = Math.ceil(world.W / res); this.ny = Math.ceil(world.H / res);
    this.lo = new Float32Array(this.nx * this.ny);
    this.version = 0;
  }
  idx(x, y) {
    const i = Math.floor((x - this.x0) / this.res), j = Math.floor((y - this.y0) / this.res);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return -1;
    return j * this.nx + i;
  }
  center(k) { return { x: this.x0 + (k % this.nx + 0.5) * this.res, y: this.y0 + (Math.floor(k / this.nx) + 0.5) * this.res }; }
  // חיישן עומק מדומה: רשת עומק מהמצלמה, רעש שגדל בריבוע המרחק
  integrate(robot, world, nRays = 36, maxD = 4) {
    for (let r = 0; r < nRays; r++) {
      const a = robot.th - FOV / 2 + FOV * (r + 0.5) / nRays;
      const dx = Math.cos(a), dy = Math.sin(a);
      const hit = rayNearest(world.segs, robot.x, robot.y, dx, dy, maxD);
      const d = Math.max(0.05, hit.d + gauss() * 0.015 * hit.d * hit.d);
      const end = Math.min(d, maxD);
      let last = -1;
      for (let t = 0.05; t < end - this.res * 0.5; t += this.res * 0.5) {
        const k = this.idx(robot.x + dx * t, robot.y + dy * t);
        if (k < 0 || k === last) continue;
        last = k;
        this.lo[k] = Math.max(-4, this.lo[k] - 0.35);
      }
      if (hit.seg && d < maxD) {
        const k = this.idx(robot.x + dx * d, robot.y + dy * d);
        if (k >= 0) this.lo[k] = Math.min(4, this.lo[k] + 0.9);
      }
    }
    this.version++;
  }
  areaKnown() {
    let n = 0;
    for (const v of this.lo) if (v < -0.3) n++;
    return n * this.res * this.res;
  }
}

// ---------- חוקר אוטונומי: חזיתות (frontier exploration) ----------
export class Explorer {
  constructor(grid) {
    this.grid = grid; this.path = []; this.target = null; this.state = 'spin';
    this.spun = 0; this.replanT = 0; this.black = new Set(); this.stuckT = 0; this.ref = null; this.backT = 0;
    this.done = false; this.nearT = 0;
  }
  plan(robot) {
    const g = this.grid, n = g.nx * g.ny, lo = g.lo;
    const occ = new Uint8Array(n);
    for (let k = 0; k < n; k++) if (lo[k] > 0.3) {
      const i = k % g.nx, j = (k / g.nx) | 0;
      for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) {
        const ii = i + di, jj = j + dj;
        if (ii >= 0 && jj >= 0 && ii < g.nx && jj < g.ny && di * di + dj * dj <= 5) occ[jj * g.nx + ii] = 1;
      }
    }
    const free = k => lo[k] < -0.3 && !occ[k];
    const unknown = k => Math.abs(lo[k]) <= 0.3;
    const start = g.idx(robot.x, robot.y);
    if (start < 0) return null;
    const prev = new Int32Array(n).fill(-1);
    prev[start] = start;
    const queue = [start];
    let found = -1, fallback = -1;
    const nb = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    for (let h = 0; h < queue.length; h++) {
      const k = queue[h], i = k % g.nx, j = (k / g.nx) | 0;
      if (k !== start && !this.black.has(k)) {
        let front = false;
        for (let q = 0; q < 4; q++) {
          const ii = i + nb[q][0], jj = j + nb[q][1];
          if (ii >= 0 && jj >= 0 && ii < g.nx && jj < g.ny && unknown(jj * g.nx + ii)) { front = true; break; }
        }
        if (front) {
          const c = g.center(k);
          if (Math.hypot(c.x - robot.x, c.y - robot.y) > 0.6) { found = k; break; }
          if (fallback < 0) fallback = k;
        }
      }
      for (const [di, dj] of nb) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= g.nx || jj >= g.ny) continue;
        const kk = jj * g.nx + ii;
        if (prev[kk] >= 0 || !free(kk)) continue;
        prev[kk] = k; queue.push(kk);
      }
    }
    if (found < 0) found = fallback;
    if (found < 0) return null;
    const path = [];
    for (let k = found; k !== start; k = prev[k]) path.push(g.center(k));
    path.reverse();
    return { path, goal: found };
  }
  control(robot, world, dt) {
    if (this.done) return [0, 0];
    if (this.state === 'spin') {
      this.spun += Math.abs(robot.w) * dt;
      if (this.spun > Math.PI * 2.1) this.state = 'go';
      return [0.35, -0.35];
    }
    if (this.backT > 0) { this.backT -= dt; return this.backT > 0.5 ? [-0.4, -0.4] : [0.4, -0.4]; }

    // תקיעות
    this.stuckT += dt;
    if (!this.ref) this.ref = { x: robot.x, y: robot.y };
    if (this.stuckT > 3) {
      if (Math.hypot(robot.x - this.ref.x, robot.y - this.ref.y) < 0.08) { this.backT = 1.2; this.replanT = 0; if (this.goal != null) this.black.add(this.goal); }
      this.stuckT = 0; this.ref = { x: robot.x, y: robot.y };
    }

    this.replanT -= dt;
    if (this.replanT <= 0) {
      this.replanT = 0.6;
      const p = this.plan(robot);
      if (!p) { this.done = true; this.path = []; return [0, 0]; }
      this.path = p.path; this.goal = p.goal;
    }
    if (!this.path.length) return [0.3, -0.3];
    // נקודת מבט קדימה לאורך המסלול
    let tgt = this.path[this.path.length - 1];
    for (const pt of this.path) if (Math.hypot(pt.x - robot.x, pt.y - robot.y) > 0.35) { tgt = pt; break; }
    const goalPt = this.path[this.path.length - 1];
    if (Math.hypot(goalPt.x - robot.x, goalPt.y - robot.y) < 0.3) {
      this.nearT += dt;
      if (this.nearT > 2.5) { this.black.add(this.goal); this.nearT = 0; this.replanT = 0; }
    } else this.nearT = 0;
    const err = wrapAngle(Math.atan2(tgt.y - robot.y, tgt.x - robot.x) - robot.th);
    // ביטחון: קיר קרוב מקדימה
    const front = rayNearest(world.segs, robot.x, robot.y, Math.cos(robot.th), Math.sin(robot.th), 1).d;
    let base = 0.55 * Math.max(0, Math.cos(err)) * Math.min(1, (front - BODY_R) / 0.3);
    if (Math.abs(err) > 0.9) base = 0;
    const turn = Math.max(-0.45, Math.min(0.45, 1.1 * err));
    return [base + turn, base - turn];
  }
}

// ---------- החתול ----------
export class Cat {
  constructor(world) {
    this.world = world;
    const p = randomFreePose(world, 0.4);
    this.x = p.x; this.y = p.y; this.th = p.th; this.v = 0; this.wait = 1; this.goal = null; this.dragged = false;
  }
  update(dt) {
    if (this.dragged) return;
    if (this.wait > 0) { this.wait -= dt; this.v = 0; return; }
    if (!this.goal) {
      const p = randomFreePose(this.world, 0.35);
      this.goal = p;
    }
    const dx = this.goal.x - this.x, dy = this.goal.y - this.y, d = Math.hypot(dx, dy);
    if (d < 0.15) { this.goal = null; this.wait = 1 + Math.random() * 4; return; }
    const want = Math.atan2(dy, dx);
    this.th += Math.max(-3 * dt, Math.min(3 * dt, wrapAngle(want - this.th)));
    this.v = 0.45;
    const nx = this.x + Math.cos(this.th) * this.v * dt, ny = this.y + Math.sin(this.th) * this.v * dt;
    if (clearance(this.world, nx, ny) > 0.12 && !insideBox(this.world, nx, ny, 0.1)) { this.x = nx; this.y = ny; }
    else { this.goal = null; this.wait = 0.3; this.th += Math.PI / 2; }
  }
}

export function seeTarget(robot, world, tx, ty, maxD = 6) {
  const dx = tx - robot.x, dy = ty - robot.y, d = Math.hypot(dx, dy);
  const bearing = wrapAngle(Math.atan2(dy, dx) - robot.th);
  if (Math.abs(bearing) > FOV / 2 || d > maxD) return null;
  const hit = rayNearest(world.segs, robot.x, robot.y, dx / d, dy / d, d);
  if (hit.seg) return null;
  return { d, bearing };
}

export class Follower {
  constructor() { this.lastSide = 1; this.lostT = 0; this.seen = null; }
  control(robot, world, cat) {
    const s = seeTarget(robot, world, cat.x, cat.y);
    this.seen = s;
    if (s) {
      this.lostT = 0; this.lastSide = Math.sign(s.bearing) || 1;
      const fwd = Math.max(-0.4, Math.min(0.8, 0.9 * (s.d - 0.7)));
      const turn = Math.max(-0.5, Math.min(0.5, 1.6 * s.bearing));
      return [fwd + turn, fwd - turn];
    }
    this.lostT += 1 / 60;
    if (this.lostT < 5) return [0.35 * this.lastSide, -0.35 * this.lastSide];
    // חיפוש: נסיעה עם הימנעות ממכשולים
    const f = rayNearest(world.segs, robot.x, robot.y, Math.cos(robot.th), Math.sin(robot.th), 1.5).d;
    if (f < 0.5) return [0.4, -0.4];
    if (this.lostT > 9) this.lostT = 0;
    return [0.5, 0.5];
  }
}

// ---------- יריב סומו ----------
export class SumoAI {
  constructor(R, dirty = false) { this.R = R; this.dirty = dirty; this.mode = 'stalk'; this.t = 0; this.side = Math.random() < 0.5 ? 1 : -1; this.dodgeT = 0; }
  control(me, foe, dt) {
    this.t += dt;
    const r = Math.hypot(me.x, me.y);
    const out = Math.atan2(me.y, me.x);
    // קרוב לשפה ונוסע החוצה: לחזור למרכז
    const movingOut = me.vx * Math.cos(out) + me.vy * Math.sin(out) > 0.05;
    if (r > this.R - 0.35 && (Math.cos(wrapAngle(me.th - out)) > 0 || movingOut)) {
      const err = wrapAngle(Math.atan2(-me.y, -me.x) - me.th);
      if (Math.abs(err) > 2) return [-0.6, -0.6];
      return [0.3 + 0.6 * err, 0.3 - 0.6 * err];
    }
    const toMe = Math.atan2(me.y - foe.y, me.x - foe.x);
    const d = Math.hypot(foe.x - me.x, foe.y - me.y);
    // היריב דוהר אליי חזיתית: להתחמק הצידה
    const foeSpeed = foe.vx * Math.cos(toMe) + foe.vy * Math.sin(toMe);
    if (this.dodgeT <= 0 && d < 0.9 && foeSpeed > 0.25 && Math.abs(wrapAngle(foe.th - toMe)) < 0.5 && Math.random() < 0.04) this.dodgeT = 0.6;
    if (this.dodgeT > 0) {
      this.dodgeT -= dt;
      const want = toMe + this.side * Math.PI / 2;
      const err = wrapAngle(want - me.th);
      return [0.9 + 0.8 * err, 0.9 - 0.8 * err];
    }
    // הטריק המלוכלך: התושבת חזקה אחורה, אז נוגחים ברוורס
    if (this.dirty) {
      const err = wrapAngle(Math.atan2(foe.y - me.y, foe.x - me.x) - (me.th + Math.PI));
      if (Math.abs(err) > 0.35) return [1.2 * err, -1.2 * err].map(v => Math.max(-0.6, Math.min(0.6, v)));
      return [-1 + 0.8 * err, -1 - 0.8 * err];
    }
    // לכוון אל החזית או הגב של היריב: שם הציר החלש של הטלפון שלו
    const fc = Math.cos(foe.th), fs = Math.sin(foe.th);
    const front = { x: foe.x + fc * 0.35, y: foe.y + fs * 0.35 };
    const back = { x: foe.x - fc * 0.35, y: foe.y - fs * 0.35 };
    const dF = Math.hypot(front.x - me.x, front.y - me.y), dB = Math.hypot(back.x - me.x, back.y - me.y);
    let tgt = dF < dB ? front : back;
    if (d < 0.6 || this.mode === 'ram') tgt = { x: foe.x, y: foe.y };
    if (d < 0.6) this.mode = 'ram'; else if (d > 1.0) this.mode = 'stalk';
    const err = wrapAngle(Math.atan2(tgt.y - me.y, tgt.x - me.x) - me.th);
    const fwd = this.mode === 'ram' ? 1 : 0.6 * Math.max(0, Math.cos(err));
    const turn = Math.max(-0.6, Math.min(0.6, 1.4 * err));
    return [fwd + turn, fwd - turn];
  }
}

// ---------- Q-learning להימנעות ממכשולים ----------
export const RL_RAYS = [-0.6, -0.3, 0, 0.3, 0.6];
export const RL_ACTIONS = [
  { u: [0.8, 0.8], name: 'קדימה' },
  { u: [-0.3, 0.7], name: 'שמאלה' },
  { u: [0.7, -0.3], name: 'ימינה' },
];
export class QLearner {
  constructor() { this.reset(); }
  reset() {
    this.Q = new Float32Array(243 * 3);
    this.eps = 1; this.alpha = 0.15; this.gamma = 0.9;
    this.state = -1; this.action = 0; this.acc = 0; this.t = 0;
    this.episodes = []; this.epDist = 0; this.epTime = 0; this.steps = 0; this.greedy = false;
    this.bins = [2, 2, 2, 2, 2];
  }
  observe(robot, world) {
    let s = 0, m = 1;
    this.rays = [];
    for (let i = 0; i < RL_RAYS.length; i++) {
      const a = robot.th + RL_RAYS[i];
      const d = rayNearest(world.segs, robot.x, robot.y, Math.cos(a), Math.sin(a), 2).d;
      const b = d < 0.3 ? 0 : d < 0.7 ? 1 : 2;
      this.bins[i] = b; this.rays.push(d);
      s += b * m; m *= 3;
    }
    return s;
  }
  choose(s) {
    if (!this.greedy && Math.random() < this.eps) return (Math.random() * 3) | 0;
    let best = 0;
    for (let a = 1; a < 3; a++) if (this.Q[s * 3 + a] > this.Q[s * 3 + best]) best = a;
    return best;
  }
  learn(s, a, r, s2, terminal) {
    let mx = 0;
    if (!terminal) mx = Math.max(this.Q[s2 * 3], this.Q[s2 * 3 + 1], this.Q[s2 * 3 + 2]);
    const k = s * 3 + a;
    this.Q[k] += this.alpha * (r + this.gamma * mx - this.Q[k]);
  }
  // צעד פיזיקה אחד, החלטה כל 0.12 שניות
  tick(robot, world, dt) {
    this.t += dt; this.epTime += dt;
    const sp = robot.speed;
    this.acc += sp * dt * 10;
    this.epDist += Math.max(0, sp) * dt;
    const crashed = robot.impact > 0.05;
    const timeout = this.epTime > 60;
    if (crashed || timeout || this.t >= 0.12 || this.state < 0) {
      const s2 = this.observe(robot, world);
      if (this.state >= 0) {
        const r = this.acc - (this.action ? 0.05 : 0) - (crashed ? 10 : 0);
        if (!this.greedy) this.learn(this.state, this.action, r, s2, crashed);
      }
      this.acc = 0; this.t = 0;
      if (crashed || timeout) {
        this.episodes.push({ dist: this.epDist, crash: crashed });
        if (this.episodes.length > 2000) this.episodes.shift();
        this.epDist = 0; this.epTime = 0; this.state = -1;
        robot.resetPose(randomFreePose(world, 0.4));
        return;
      }
      this.state = s2;
      this.action = this.choose(s2);
      if (!this.greedy) this.eps = Math.max(0.03, this.eps * 0.9993);
      this.steps++;
    }
    const u = RL_ACTIONS[this.action].u;
    robot.cmdNow(u[0], u[1]);
  }
}
