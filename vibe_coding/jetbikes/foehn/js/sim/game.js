// Game rules, free of any DOM or rendering so the whole thing also runs headless in tests:
// events, check-points, landings, crashes, respawns, timing, medals, ghost recording.
import { Sim } from './sim.js';
import { windFor } from './wind.js';
import { V_MAX, V_MIN } from './fcs.js';
import { G, PROBES, FUEL0 } from './bike.js';
import { T, groundAt } from './terrain.js';
import { S, hitScenery, canopyAt } from './scenery.js';
import { buildEvent, eventById } from './courses.js';
import { indoor } from './castle.js';
import { Gun } from './gun.js';
import { clamp, dot, sub, len, yprOf, qrot } from './vmath.js';

export const CRASH_WAIT = 2.4;
const BODY_PTS = [PROBES[0], PROBES[1], PROBES[2], PROBES[4], PROBES[5], [0, 0.5, 0]];

export class Game {
  constructor() {
    this.sim = new Sim();
    this.bike = this.sim.bike;
    this.sim.fcs.obstacle = canopyAt;
    this.state = 'idle';       // idle | countdown | fly | crashed | finished
    this.events = [];          // drained by the UI every frame
    this.build = null; this.cp = 0; this.time = 0; this.penalty = 0; this.vt = 0;
    this.best = null;          // { time, splits, ghost } supplied by the UI layer
    this.ghost = []; this.splits = [];
    this.score = 0; this.combo = 1; this.lowT = 0;
    this.crashes = 0; this.misses = 0; this.landT = 0; this.stateT = 0; this.tick = 0;
    this.respawnPose = null;
    this.gun = new Gun();
    this.windLevel = 1; this.windSeed = 7;   // the UI sets these (0 calm .. 3 storm; a new seed for every run)
    this.fig = { loop: 0, roll: 0, inv: false, loops: 0, rolls: 0 };   // aerobatic figures: how far round the bike has come
  }
  // Aerobatic mode (sim/fcs.js, acroLaw): the stick turns the bike itself. Only in the air.
  get acro() { return this.sim.fcs.aerobatic; }
  toggleAcro(on = !this.sim.fcs.aerobatic) {
    const f = this.sim.fcs, was = f.aerobatic;
    if (this.state !== 'fly' && on) return false;
    const now = f.setAcro(on, 'rider');                                 // (leaving is reported from onTick, like the safety net)
    if (now && !was) { this.fig.loop = this.fig.roll = 0; this.fig.inv = false; this.emit('acro', { on: true }); }
    else if (on && !now && !f.rec) this.emit('acro', { on: false, why: 'air' });          // refused: the bike stands on the ground
    return now;
  }
  emit(type, data = {}) { this.events.push({ type, ...data }); }

  load(id, best = null) {
    const ev = eventById(id);
    this.ev = ev; this.build = buildEvent(ev); this.best = best;
    this.cps0 = this.build.real;
    this.setWind(this.windLevel, this.windSeed);
    this.restart();
  }
  setWind(level, seed = this.windSeed) { this.windLevel = level; this.windSeed = seed; if (!this.ev) return;
    // inside the halls of the fortress the walls keep most of the wind out
    const w = windFor(this.ev.wind, level, seed);
    this.sim.wind = Object.assign((p, t, agl) => { const v = w(p, t, agl), k = 1 - 0.88 * indoor(p); if (k < 1) { v[0] *= k; v[1] *= k; v[2] *= k; } return v; }, { dir: w.dir, speed: w.speed });
  }
  restart() {
    const st = this.build.start;
    if (this.ev.env) this.setWind(this.ev.env.wind);
    // balloons are shot in any order: the one that is hit changes places with the next in line, so cp still counts them
    this.cps = this.cps0.slice(); this.cps0.forEach((c, k) => { c.k = k; c.popped = false; });
    this.gun.reset();
    this.sim.reset(st.p, st.yaw); this.sim.t = 0;
    this.cp = 0; this.time = 0; this.penalty = 0; this.vt = 0; this.ghost = []; this.splits = [];
    this.score = 0; this.combo = 1; this.lowT = 0; this.crashes = 0; this.misses = 0; this.landT = 0; this.tick = 0;
    this.fig = { loop: 0, roll: 0, inv: false, loops: 0, rolls: 0 };
    this.respawnPose = { p: st.p, yaw: st.yaw, speed: 0 };
    this.state = 'countdown'; this.stateT = this.ev.id === 'free' ? 0.01 : 3;
    this.prevP = this.bike.p.slice();
    this.emit('load');
  }
  get total() { return this.time + this.penalty; }
  get nextCp() { return this.cps[this.cp] || null; }
  get gunOn() { return !!this.ev.gun || this.ev.id === 'free'; }
  popTarget(c, off) {
    const j = this.cps.indexOf(c); if (j < this.cp || c.popped) return;
    this.cps[j] = this.cps[this.cp]; this.cps[this.cp] = c; c.popped = true;
    this.emit('pop', { k: c.k, p: c.p.slice() });
    this.pass(c, off);
  }

  // ctl: { turn, pitch, strafe, throttle (-1..1), boost, brake } or autopilot { vt }
  update(dt, ctl) {
    const b = this.bike;
    if (this.state === 'countdown') {
      const before = Math.ceil(this.stateT);
      this.stateT -= dt;
      const after = Math.ceil(this.stateT);
      if (after !== before && after > 0) this.emit('count', { n: after });
      this.sim.advance(dt, { turn: 0, pitch: 0, vt: 0 });
      if (this.stateT <= 0) { this.state = 'fly'; this.emit('go'); }
      return;
    }
    if (this.state === 'crashed') {
      this.stateT -= dt;
      if (this.ev.id !== 'free') this.time += dt;
      if (this.stateT <= 0) this.respawn();
      return;
    }
    if (this.state !== 'fly' && this.state !== 'finished') return;

    // throttle lever -> target speed
    if (ctl.vt !== undefined) this.vt = ctl.vt;
    else {
      const fwd = dot(b.v, qrot(b.q, [1, 0, 0]));
      this.vt = clamp(this.vt + (ctl.throttle || 0) * 20 * dt, V_MIN, V_MAX);
      if (ctl.brake) this.vt = Math.max(0, Math.min(this.vt, fwd) - 30 * dt);
      if (ctl.boost && !b.abLock) this.vt = clamp(Math.max(this.vt, fwd + 3), V_MIN, V_MAX + 30);
      else if (this.vt > V_MAX) this.vt = Math.max(V_MAX, Math.min(this.vt, fwd));
    }
    // touchdown cuts the engine: resting on the skids with no throttle, climb or boost asked
    // for, any leftover target speed (a few km/h, or backwards after S) would keep the flight
    // computer awake and lift the bike off again
    if (b.contactN > 0.25 * b.mp.m * G && len(b.v) < 4 && !ctl.boost && (ctl.pitch || 0) <= 0.05 && (ctl.vt !== undefined ? Math.abs(ctl.vt) < 3 : !(ctl.throttle > 0))) this.vt = 0;
    const inp = this.state === 'finished' ? { turn: 0, pitch: 0, vt: 0, brake: true }
      : { turn: ctl.turn || 0, pitch: ctl.pitch || 0, strafe: ctl.strafe || 0, vt: this.vt, boost: !!ctl.boost, brake: !!ctl.brake };
    if (ctl.fire && this.gunOn && this.state === 'fly') this.gun.trigger = true;
    this.sim.advance(dt, inp, () => this.onTick());
    if (this.state === 'fly' && this.ev.id !== 'free') this.time += dt;
  }

  onTick() {
    const b = this.bike, p = b.p;
    this.tick++;
    this.gun.step(1 / 120, this);
    // --- aerobatics: the computer may have taken the bike back; smoke; figures flown ---
    const fc = this.sim.fcs, fig = this.fig;
    if (fc.bail) { this.emit('acro', { on: false, why: fc.bail }); fc.bail = null; }
    b.smoke = fc.aerobatic && this.state === 'fly';
    if (fc.aerobatic) {
      // a loop is once round in pitch that went over the top (a flat circle on its side is a turn);
      // a roll is once round the nose. A pause between the quarters of a roll does not spoil it.
      fig.loop += b.w[2] / 120; fig.roll += b.w[0] / 120;
      if (Math.abs(b.w[2]) < 0.2) fig.loop *= 1 - 1 / 180;
      if (Math.abs(b.w[0]) < 0.3) fig.roll *= 1 - 1 / 480;
      if (Math.abs(fig.loop) > 1.2 && qrot(b.q, [0, 1, 0])[1] < -0.5) fig.inv = true;
      if (Math.abs(fig.loop) >= 6.1) { if (fig.inv && this.state === 'fly') { fig.loops++; this.score += 400; this.emit('trick', { kind: 'loop', n: fig.loops }); } fig.loop -= Math.sign(fig.loop) * 2 * Math.PI; fig.inv = false; }
      if (Math.abs(fig.roll) >= 6.1) { if (this.state === 'fly') { fig.rolls++; this.score += 250; this.emit('trick', { kind: 'roll', n: fig.rolls }); } fig.roll -= Math.sign(fig.roll) * 2 * Math.PI; }
    }
    if (this.state !== 'fly') { this.prevP = p.slice(); return; }
    // --- crashes ---
    let reason = b.crash;
    if (!reason && (b.agl < 40 || this.tick % 4 === 0)) {
      for (const d of BODY_PTS) { const h = hitScenery(b.toWorld(d)); if (h) { reason = h; break; } }
    }
    const lim = T.half - 70;
    if (!reason && (Math.abs(p[0]) > lim || Math.abs(p[2]) > lim)) reason = 'bounds';
    if (!reason && this.tick % 30 === 0 && (Math.abs(p[0]) > lim - 260 || Math.abs(p[2]) > lim - 260)) this.emit('warn', { text: 'bounds' });
    if (!reason && b.fuel <= 0 && b.contactN > 0 && len(b.v) < 1) reason = 'fuel';
    // --- check-points ---
    const c = this.nextCp;
    if (!reason && c) {
      if (c.type === 'ring') {
        const d0 = dot(sub(this.prevP, c.p), c.n), d1 = dot(sub(p, c.p), c.n);
        // The rings are flown in their order and in their direction, always. A ring that was
        // missed stays the next one: come round and fly through it the right way.
        if ((d0 < 0) !== (d1 < 0)) {
          const fwd = d0 < 0, t = d0 / (d0 - d1);
          const hit = [this.prevP[0] + (p[0] - this.prevP[0]) * t - c.p[0], this.prevP[1] + (p[1] - this.prevP[1]) * t - c.p[1], this.prevP[2] + (p[2] - this.prevP[2]) * t - c.p[2]];
          const r = len(hit);
          if (r >= c.r - 0.9 && r < c.r + 1.1) reason = 'ring';                 // the rim is solid from either side
          else if (r < c.r) { if (fwd) this.pass(c, r / c.r); else this.emit('wrongway', { index: this.cp }); }
          else if (fwd && r < c.r * 12) { this.misses++; this.emit('miss', { index: this.cp, off: r, dy: hit[1] }); }
        }
      } else if (c.type === 'land') {
        const onPad = Math.hypot(p[0] - c.p[0], p[2] - c.p[2]) < c.r;
        if (onPad && b.contactN > 0.5 * b.mp.m * G && len(b.v) < 1.0) { this.landT += 1 / 120; if (this.landT > 0.45) { this.landT = 0; this.pass(c, Math.hypot(p[0] - c.p[0], p[2] - c.p[2]) / c.r); } } else this.landT = 0;
        // parked beside the circle: say so, otherwise it looks as if the landing just didn't count
        if (!onPad && b.contactN > 0.5 * b.mp.m * G && len(b.v) < 1.0 && this.tick % 60 === 0) this.emit('warn', { text: 'offpad' });
      }
    }
    if (reason) { this.crash(reason); this.prevP = p.slice(); return; }
    // --- ghost ---
    if (this.tick % 6 === 0 && this.ev.id !== 'free') this.ghost.push(p[0], p[1], p[2], b.q[0], b.q[1], b.q[2], b.q[3]);
    // --- free flight: proximity score, refuelling on pads ---
    if (this.ev.id === 'free') {
      const V = len(b.v);
      if (V > 28 && b.agl < 14 && b.contactN === 0) {
        this.lowT += 1 / 120; this.combo = 1 + Math.min(9, Math.floor(this.lowT / 2.5));
        this.score += (V / 30) * (1 + (14 - b.agl) / 4) * this.combo / 120 * 10;
      } else if (b.agl > 30 || V < 15) { this.lowT = 0; this.combo = 1; }
      if (b.contactN > 0 && V < 1 && b.fuel < FUEL0 - 0.5) { b.fuel = Math.min(FUEL0, b.fuel + 6 / 120); if (this.tick % 60 === 0) this.emit('refuel'); }
    }
    this.prevP = p.slice();
  }

  pass(c, off) {
    const t = this.total, i = this.cp;
    this.splits.push(t);
    const delta = this.best && this.best.splits && this.best.splits[i] !== undefined ? t - this.best.splits[i] : null;
    this.emit(c.type === 'land' ? 'land' : 'gate', { index: i, time: t, delta, perfect: off < 0.3, last: i === this.cps.length - 1 });
    // an event with a day of its own: this check-point turns the hour, the weather and the wind
    if (c.stage) { this.setWind(c.stage.wind); this.emit('stage', { env: c.stage }); }
    this.advance(c, true);
  }
  advance(c, passed) {
    const b = this.bike;
    // where a crash puts you back: at the check-point just passed, flying along the course
    if (c.type === 'target') this.respawnPose = { p: [b.p[0], Math.max(b.p[1], groundAt(b.p[0], b.p[2]) + 30), b.p[2]], yaw: yprOf(b.q)[0], speed: 30, pitch: 0 };   // after a balloon: where the shot was fired from, with air underneath
    else if (c.type === 'land') this.respawnPose = { p: c.p, yaw: yprOf(b.q)[0], speed: 0 };
    else this.respawnPose = { p: c.p, yaw: Math.atan2(-c.n[2], c.n[0]), speed: c.slow ? clamp(len(b.v), 12, Math.min(c.v, 20)) : clamp(len(b.v), 30, 45), pitch: Math.asin(clamp(c.n[1], -1, 1)) };
    this.cp++;
    if (this.cp >= this.cps.length) this.finish();
  }
  finish() {
    const t = this.total, m = this.ev.medals;
    const medal = !m ? 0 : t <= m[0] ? 3 : t <= m[1] ? 2 : t <= m[2] ? 1 : 0;
    const isBest = !this.best || t < this.best.time;
    this.state = 'finished';
    this.result = { time: t, medal, isBest, prev: this.best ? this.best.time : null, crashes: this.crashes, misses: this.misses, shots: this.gun.fired, hits: this.gun.hits, splits: this.splits.slice(), ghost: new Float32Array(this.ghost) };
    this.emit('finish', this.result);
  }
  crash(reason) {
    const b = this.bike;
    this.crashes++; this.combo = 1; this.lowT = 0;
    this.emit('crash', { reason, p: b.p.slice(), v: b.v.slice(), water: reason === 'water' });
    this.state = 'crashed'; this.stateT = CRASH_WAIT;
  }
  respawn() {
    const r = this.respawnPose, b = this.bike, fuel = b.fuel;
    if (r.speed > 0) {
      this.sim.reset(r.p, r.yaw, r.speed);
      b.v[1] = Math.sin(r.pitch || 0) * r.speed;
      this.vt = r.speed;
    } else { this.sim.reset([r.p[0], groundAt(r.p[0], r.p[2]), r.p[2]], r.yaw); this.vt = 0; }
    b.fuel = Math.max(fuel, 10);
    this.state = 'fly'; this.prevP = b.p.slice();
    this.emit('respawn');
  }
  // pose of the best-run ghost at race time t: [x,y,z, qw,qx,qy,qz] or null
  ghostAt(t) {
    const g = this.best && this.best.ghost;
    if (!g || g.length < 14) return null;
    const f = t * 20, i = Math.floor(f);
    if (i < 0 || (i + 1) * 7 >= g.length) return null;
    const u = f - i, o = [];
    for (let k = 0; k < 7; k++) o.push(g[i * 7 + k] + (g[(i + 1) * 7 + k] - g[i * 7 + k]) * u);
    return o;
  }
}
