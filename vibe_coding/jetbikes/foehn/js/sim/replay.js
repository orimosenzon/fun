// The flight recorder. Every drawn frame of a run it keeps the whole state of the bike (position,
// attitude, every nozzle, the engine), so the replay shows exactly what happened: nothing is
// simulated again, and a replay can run slow, jump about, or be watched from anywhere.
// Free of any DOM so it also runs in the tests.
const STATES = ['idle', 'countdown', 'fly', 'crashed', 'finished'];
const HDR = 8;                 // t, state, check-point, lean, wind x y z, speed target
const CH = 1024;               // frames per block of memory
const MAX_BLOCKS = 56;         // about a quarter of an hour at 60 frames a second; older frames are dropped
const TAIL = 7;                // seconds kept after the finish line

// every number of the bike that is worth keeping: [object, key, length (0 = scalar), is boolean]
function layoutOf(bike) {
  const out = [];
  const scan = (o) => {
    for (const k of Object.keys(o)) {
      if (k[0] === '_') continue;
      const v = o[k];
      if (typeof v === 'number') out.push([o, k, 0, false]);
      else if (typeof v === 'boolean') out.push([o, k, 0, true]);
      else if (Array.isArray(v) && v.length && v.every((x) => typeof x === 'number')) out.push([o, k, v.length, false]);
    }
  };
  scan(bike); scan(bike.eng);
  return out;
}

export class Recorder {
  constructor(game) {
    this.game = game;
    this.fields = layoutOf(game.bike);
    this.W = HDR + this.fields.reduce((s, f) => s + (f[2] || 1), 0);
    let o = HDR; this.off = {};
    for (const f of this.fields) { if (f[0] === game.bike) this.off[f[1]] = o; o += f[2] || 1; }
    this.reset();
  }
  reset() { this.blocks = []; this.n = 0; this.base = 0; this.t = 0; this.events = []; this.tail = 0; }
  get t0() { return this.n > this.base ? this.row(this.base)[this.ro(this.base)] : 0; }
  get t1() { return this.n > this.base ? this.row(this.n - 1)[this.ro(this.n - 1)] : 0; }
  get duration() { return this.t1 - this.t0; }
  row(i) { return this.blocks[(i - this.base) >> 10]; }
  ro(i) { return ((i - this.base) & (CH - 1)) * this.W; }

  // once per frame while the run is being flown
  push(dt, lean = 0) {
    const g = this.game;
    if (g.state === 'finished') { this.tail += dt; if (this.tail > TAIL) return; } else this.tail = 0;
    this.t += dt;
    const k = this.n - this.base;
    if ((k >> 10) >= this.blocks.length) {
      if (this.blocks.length >= MAX_BLOCKS) { this.blocks.shift(); this.base += CH; }
      this.blocks.push(new Float32Array(CH * this.W));
    }
    this.write(this.row(this.n), this.ro(this.n), lean);
    this.n++;
  }
  write(a, o, lean) {
    const g = this.game, w = g.sim.windW;
    a[o] = this.t; a[o + 1] = STATES.indexOf(g.state); a[o + 2] = g.cp; a[o + 3] = lean; a[o + 4] = w[0]; a[o + 5] = w[1]; a[o + 6] = w[2]; a[o + 7] = g.vt;
    let j = o + HDR;
    for (const [obj, key, n, bool] of this.fields) {
      const v = obj[key];
      if (n) for (let i = 0; i < n; i++) a[j++] = v[i]; else a[j++] = bool ? +v : v;
    }
  }
  mark(type, data = {}) { this.events.push({ ...data, type, t: this.t }); }

  // ---- keeping a flight for later (the history of flights) ----
  names() { return this.fields.map(([obj, key, n]) => [obj === this.game.bike ? 'b' : 'e', key, n || 1]); }
  // the whole recording as one array, with the names of its columns
  dump() {
    const cnt = this.n - this.base, frames = new Float32Array(cnt * this.W);
    for (let k = 0; k < cnt; k += CH) { const m = Math.min(CH, cnt - k) * this.W; frames.set(this.blocks[k >> 10].subarray(0, m), k * this.W); }
    return { hdr: HDR, W: this.W, layout: this.names(), frames, events: this.events.filter((e) => e.t >= this.t0) };
  }
  // put a kept flight back. The columns are matched by name, so a flight recorded before the
  // bike gained (or lost) a part still plays: what it does not know stays at zero.
  load(d) {
    this.reset();
    if (!d || d.hdr !== HDR || !d.frames || !d.W) return false;
    const at = {}; let o = d.hdr;
    for (const [tag, key, n] of d.layout) { at[tag + '.' + key] = [o, n]; o += n; }
    const src = new Int32Array(this.W).fill(-1); for (let j = 0; j < HDR; j++) src[j] = j;
    let j = HDR;
    for (const [tag, key, n] of this.names()) { const s = at[tag + '.' + key]; for (let i = 0; i < n; i++, j++) if (s && i < s[1]) src[j] = s[0] + i; }
    const cnt = Math.min(Math.floor(d.frames.length / d.W), MAX_BLOCKS * CH);
    for (let i = 0; i < cnt; i++) {
      if ((i >> 10) >= this.blocks.length) this.blocks.push(new Float32Array(CH * this.W));
      const a = this.blocks[i >> 10], ao = (i & (CH - 1)) * this.W, so = i * d.W;
      for (let c = 0; c < this.W; c++) a[ao + c] = src[c] < 0 ? 0 : d.frames[so + src[c]];
    }
    this.n = cnt; this.t = this.t1; this.events = (d.events || []).slice();
    return cnt > 1;
  }

  // the state of everything right now, to put back when the replay is over
  snapshot() { const a = new Float32Array(this.W); this.write(a, 0, 0); return { a, cp: this.game.cp, state: this.game.state }; }
  restore(s) { this.read(s.a, 0, s.a, 0, 0); this.game.cp = s.cp; this.game.state = s.state; }

  // index of the last frame at or before time t
  find(t) {
    let lo = this.base, hi = this.n - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (this.row(m)[this.ro(m)] <= t) lo = m; else hi = m - 1; }
    return lo;
  }
  // put the recorded moment t back into the game. Returns the rider's lean.
  apply(t) {
    if (this.n <= this.base) return 0;
    const i = this.find(t), j = Math.min(i + 1, this.n - 1);
    const a = this.row(i), ao = this.ro(i), b = this.row(j), bo = this.ro(j);
    let u = b[bo] > a[ao] ? Math.min(1, Math.max(0, (t - a[ao]) / (b[bo] - a[ao]))) : 0;
    // never blend across a respawn or a change of state: the bike would streak across the valley
    const P = this.off.p;
    if (a[ao + 1] !== b[bo + 1] || Math.hypot(b[bo + P] - a[ao + P], b[bo + P + 1] - a[ao + P + 1], b[bo + P + 2] - a[ao + P + 2]) > 30) u = 0;
    return this.read(a, ao, b, bo, u);
  }
  read(a, ao, b, bo, u) {
    const g = this.game, mix = (k) => a[ao + k] + (b[bo + k] - a[ao + k]) * u;
    g.state = STATES[a[ao + 1]] || 'fly'; g.cp = a[ao + 2]; g.vt = mix(7);
    const w = g.sim.windW; w[0] = mix(4); w[1] = mix(5); w[2] = mix(6);
    let j = HDR;
    for (const [obj, key, n, bool] of this.fields) {
      if (n) { const v = obj[key]; for (let i = 0; i < n; i++, j++) v[i] = mix(j); } else { obj[key] = bool ? a[ao + j] > 0.5 : mix(j); j++; }
    }
    const q = g.bike.q, l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; q[0] /= l; q[1] /= l; q[2] /= l; q[3] /= l;
    return mix(3);
  }

  // what the director needs to know about moment t (clamped to the recording)
  at(t) {
    const i = this.find(t), j = Math.min(i + 1, this.n - 1);
    const a = this.row(i), ao = this.ro(i), b = this.row(j), bo = this.ro(j), o = this.off;
    let u = b[bo] > a[ao] ? Math.min(1, Math.max(0, (t - a[ao]) / (b[bo] - a[ao]))) : 0;
    if (a[ao + 1] !== b[bo + 1] || Math.hypot(b[bo + o.p] - a[ao + o.p], b[bo + o.p + 1] - a[ao + o.p + 1], b[bo + o.p + 2] - a[ao + o.p + 2]) > 30) u = 0;
    const v3 = (k) => [a[ao + k] + (b[bo + k] - a[ao + k]) * u, a[ao + k + 1] + (b[bo + k + 1] - a[ao + k + 1]) * u, a[ao + k + 2] + (b[bo + k + 2] - a[ao + k + 2]) * u];
    const q = [0, 1, 2, 3].map((k) => a[ao + o.q + k] + (b[bo + o.q + k] - a[ao + o.q + k]) * u), l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return { p: v3(o.p), v: v3(o.v), q: q.map((x) => x / l), state: STATES[a[ao + 1]], agl: a[ao + o.agl], cp: a[ao + 2] };
  }
}
