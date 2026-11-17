// The director of the replay. It gets the finished recording, so unlike a live camera it knows
// where the bike is going to be: it can stand beside the track before the bike arrives, wait
// behind a ring, and be in place for a crash before it happens. It cuts the flight into shots
// of a few seconds each, never two of the same kind in a row, and checks every shot against
// the mountains, the trees and the walls so the bike is not hidden.
// A shot is a pure function of time: camera position, look-at point, up and lens.
// Free of any DOM or rendering, so the tests can run it.
import { surfaceAt, groundAt } from './terrain.js';
import { hitCastle, indoor } from './castle.js';
import { hitScenery } from './scenery.js';
import { qrot, clamp, lerp, sstep, norm, sub, len, cross, madd, add } from './vmath.js';

const UP = [0, 1, 0], DEG = 180 / Math.PI;
const lensFor = (d, size, lo, hi) => clamp(2 * Math.atan(size / Math.max(d, 0.5)) * DEG, lo, hi);
const DUR = { flyby: [3, 5], headon: [2.6, 4], chase: [3, 5], lead: [3, 4.6], track: [3.5, 6], heli: [4, 6.5], onboard: [2.6, 4.2], orbit: [4, 7] };

export function planReplay(rec, cps, cg = [0, 0.7, 0], rnd = Math.random) {
  const t0 = rec.t0, t1 = rec.t1, R = (a, b) => a + (b - a) * rnd();
  const evs = rec.events.filter((e) => e.t >= t0);
  // stretches of flight with no jump in them: a respawn puts the bike somewhere else
  const cutsAt = [t0, ...evs.filter((e) => e.type === 'respawn').map((e) => e.t), t1 + 1e-3];
  const segs = cutsAt.slice(0, -1).map((a, i) => ({ a, b: cutsAt[i + 1] - 1e-3 })).filter((s) => s.b > s.a);
  const segOf = (t) => segs.find((s) => t <= s.b) || segs[segs.length - 1];

  const maker = (seg) => {
    const P = (t) => rec.at(clamp(t, seg.a, seg.b));
    // which way the bike is travelling around moment t (its nose, when it is hardly moving)
    const dirOf = (t, win = 0.4) => {
      const d = sub(P(t + win).p, P(t - win).p);
      if (len(d) > 1.5) return norm(d);
      const f = qrot(P(t).q, [1, 0, 0]); return norm([f[0], 0, f[2]]);
    };
    const rightOf = (d) => { const r = cross(d, UP); return len(r) > 0.05 ? norm(r) : [0, 0, 1]; };
    const body = (s) => [s.p[0], s.p[1] + 0.25, s.p[2]];
    const floor = (p, h) => { p[1] = Math.max(p[1], surfaceAt(p[0], p[2]) + h); return p; };
    // a camera on a tripod: the operator follows the bike a moment late and zooms to hold it in frame
    const tripod = (C, size, lo, hi, ph) => (t, o) => {
      const s = P(t - 0.05), d = len(sub(P(t - 0.2).p, C)), j = d * 0.0035;
      o.p = C.slice(); o.l = body(s);
      o.l[0] += j * Math.sin(t * 1.9 + ph); o.l[1] += j * Math.sin(t * 2.7 + ph * 2); o.l[2] += j * Math.sin(t * 2.3 + ph * 3);
      o.up = UP; o.fov = lensFor(d, size, lo, hi); return o;
    };

    return {
      P, dirOf,
      // beside the track, close to the ground: the bike comes, fills the frame and is gone
      flyby(a, b, side) {
        const tm = lerp(a, b, R(0.4, 0.62)), s = P(tm), d = dirOf(tm), r = rightOf(d);
        const C = madd(madd(s.p, r, side * R(7, 30)), d, R(-4, 4)), g = surfaceAt(C[0], C[2]);
        C[1] = s.p[1] - g < 45 ? g + R(1.2, 3.6) : s.p[1] + R(-8, 4);
        return tripod(floor(C, 1), R(5, 10), 9, 62, R(0, 6));
      },
      // on the line of flight with a long lens: the bike grows until it passes right over the camera
      headon(a, b, side) {
        const s = P(b + 0.22), d = dirOf(b), r = rightOf(d);
        const C = madd(s.p, r, side * R(2.4, 4.5)), g = surfaceAt(C[0], C[2]);
        C[1] = s.p[1] - g < 14 ? g + R(0.9, 1.6) : s.p[1] - R(1, 2.2);
        return tripod(floor(C, 0.8), R(4, 6.5), 5, 70, R(0, 6));
      },
      // waiting behind the ring, a little off its axis
      ring(c, side) {
        const n = norm(c.n), r = rightOf(n);
        const C = madd(madd(c.p, n, c.r * 1.5 + R(4, 9)), r, side * c.r * R(0.4, 0.62)); C[1] -= c.r * R(0.1, 0.45);
        return tripod(floor(C, 1), R(6, 9), 20, 64, R(0, 6));
      },
      // the place of the accident, seen from a safe distance
      crash(tc, side) {
        const s = P(tc - 0.03), d = dirOf(tc - 0.5), r = rightOf(d);
        const C = madd(madd(s.p, d, R(12, 24)), r, side * R(14, 26)); C[1] += R(3, 9);
        return tripod(floor(C, 2), R(8, 11), 14, 48, R(0, 6));
      },
      // the first shot: a crane rising beside the bike on its pad
      open(a, b, side) {
        const s = P(a), d = dirOf(a), r = rightOf(d), C0 = madd(madd(s.p, d, R(11, 15)), r, side * R(5, 9));
        const g = surfaceAt(C0[0], C0[2]);
        return (t, o) => { const u = clamp((t - a) / (b - a), 0, 1), e = u * u * (3 - 2 * u); o.p = [C0[0] - r[0] * side * 4 * e, Math.max(g + 0.7, s.p[1] - 0.3) + 4.5 * e, C0[2] - r[2] * side * 4 * e]; o.l = body(P(t)); o.up = UP; o.fov = lerp(46, 30, e); return o; };
      },
      // behind and low, the way the rider's friend on a second bike would see it
      chase(a, b, side) {
        const low = rnd() < 0.45, dist = low ? 3.7 : 5.6, h = low ? 0.25 : 0.9, f = low ? 80 : 66;
        return (t, o) => {
          const now = P(t), s = now, d = dirOf(t - 0.12), r = rightOf(d);       // it swings out a little late in the turns
          o.p = floor([s.p[0] - d[0] * dist + r[0] * side * 0.9, s.p[1] - d[1] * dist + h, s.p[2] - d[2] * dist + r[2] * side * 0.9], 0.7);
          o.l = [now.p[0] + d[0] * 8, now.p[1] + d[1] * 8 + 0.6, now.p[2] + d[2] * 8];
          const steepK = sstep(0.45, 0.85, Math.abs(d[1]));                     // on the way up or down a loop the world's "up" lies along the lens: take the bike's own
          o.up = norm(madd([0, 1 - steepK, 0], qrot(now.q, UP), 0.5 + 1.5 * steepK)); o.fov = f; return o;
        };
      },
      // flying backwards ahead of the bike, on the rider's face
      lead(a, b, side) {
        const off = R(1.6, 2.8), h = R(0.3, 0.9);
        return (t, o) => {
          const s = P(t), d = dirOf(t, 0.6), r = rightOf(d), u = (t - a) / (b - a);
          o.p = floor([s.p[0] + d[0] * (7 - 1.8 * u) + r[0] * side * off, s.p[1] + d[1] * (7 - 1.8 * u) + h, s.p[2] + d[2] * (7 - 1.8 * u) + r[2] * side * off], 0.7);
          o.l = body(s); o.l[1] += 0.3; o.up = Math.abs(d[1]) > 0.7 ? qrot(s.q, UP) : UP; o.fov = 40; return o;
        };
      },
      // a camera ship flying alongside, drifting slowly from the nose to the tail
      track(a, b, side) {
        const dist = R(10, 17), h = R(0.2, 2.4), slide = R(4, 7) * (rnd() < 0.5 ? 1 : -1);
        return (t, o) => {
          const s = P(t), d = dirOf(t, 0.8), r = rightOf(d), u = (t - a) / (b - a), k = lerp(slide, -slide, u);
          o.p = floor([s.p[0] + r[0] * side * dist + d[0] * k, s.p[1] + h, s.p[2] + r[2] * side * dist + d[2] * k], 1);
          o.l = body(s); o.up = UP; o.fov = lensFor(dist, 4.2, 22, 34); return o;
        };
      },
      // high above and behind, the valley under the bike
      heli(a, b, side) {
        const up = R(26, 52), back = R(28, 44), out = R(18, 34), size = R(8, 13);
        return (t, o) => {
          const s0 = P(t - 0.9), s1 = P(t - 0.45), s = P(t), d = dirOf(t - 0.45, 0.9), r = rightOf(d);
          const c = [(s0.p[0] + s1.p[0] + s.p[0]) / 3, (s0.p[1] + s1.p[1] + s.p[1]) / 3, (s0.p[2] + s1.p[2] + s.p[2]) / 3];
          o.p = floor([c[0] - d[0] * back + r[0] * side * out, c[1] + up, c[2] - d[2] * back + r[2] * side * out], 12);
          o.l = body(s); o.up = UP; o.fov = lensFor(len(sub(o.p, s.p)), size, 12, 40); return o;
        };
      },
      // bolted to the bike itself
      onboard(a, b, side) {
        const k = rnd(), m = k < 0.36 ? { p: [-1.8, 1.3, 0.6 * side], l: [4, 0.75, -0.25 * side], f: 82 }              // over the tail, looking ahead past the rider
          : k < 0.7 ? { p: [1.85, 0.98, 0.45 * side], l: [-0.2, 1.22, 0], f: 62 }                                      // on the nose, looking back at the rider
            : { p: [-0.15, 0.3, 1.2 * side], l: [1.7, 0.22, 0.3 * side], f: 86 };                                      // down by the skids, on the front lift nozzle
        return (t, o) => {
          const s = P(t), w = (d) => add(s.p, qrot(s.q, sub(d, cg)));
          o.p = w(m.p); o.l = w(m.l); o.up = qrot(s.q, UP); o.fov = m.f; o.on = true; return o;
        };
      },
      // a slow circle around a bike that is standing still or hovering
      orbit(a, b, side) {
        const a0 = R(0, 6.28), rad = R(6, 9), h = R(0.9, 2.6), w = R(0.22, 0.36) * side;
        return (t, o) => {
          const s = P(t), an = a0 + w * (t - a);
          o.p = floor([s.p[0] + Math.cos(an) * rad, s.p[1] + h, s.p[2] + Math.sin(an) * rad], 0.6);
          o.l = body(s); o.l[1] += 0.3; o.up = UP; o.fov = 42; return o;
        };
      },
    };
  };

  // is the bike really in the picture, the whole shot long?  Returns the number of bad samples.
  const tmp = {};
  // dense: a camera that travels (the orbit) is looked at four times a second, or it can pass
  // through a tree between two looks
  const blocked = (cam, P, a, b, maxD, dense) => {
    let bad = 0;
    const N = dense ? Math.max(8, Math.ceil((b - a) * 4)) : 8;
    for (let i = 0; i <= N; i++) {
      const t = lerp(a, b, i / N), o = cam(t, tmp), s = P(t), d = len(sub(o.p, s.p));
      if (o.p[1] < surfaceAt(o.p[0], o.p[2]) + 0.3 || hitCastle(o.p, 0.5) || hitScenery(o.p, 0.6)) return 99;
      if (d > maxD || d < 1.2) { bad += 3; continue; }
      const steps = clamp(Math.ceil(d / 6), 6, 40);
      for (let k = 1; k < steps; k++) {
        const u = k / steps, q = [lerp(o.p[0], s.p[0], u), lerp(o.p[1], s.p[1] + 0.6, u), lerp(o.p[2], s.p[2], u)];
        if (q[1] < groundAt(q[0], q[2]) + 0.05 || hitCastle(q, 0.1) || hitScenery(q, 0)) { bad++; break; }
      }
    }
    return bad;
  };

  // ---- the moments that get a shot of their own ----
  const anchors = [], zones = [];
  for (const e of evs) {
    if (e.type !== 'crash') continue;
    const seg = segOf(e.t - 1e-3);
    anchors.push({ a: Math.max(seg.a, e.t - 1.8), b: seg.b, kind: 'crash', t: e.t });
    zones.push({ a: e.t - 0.75, b: e.t + 1.1, r: 0.3 });
  }
  let lastRing = -9;
  for (const e of evs) {
    if (e.type !== 'gate' || !cps || !cps[e.index] || cps[e.index].type !== 'ring') continue;
    if (!e.last && (rnd() > 0.45 || e.t - lastRing < 9)) continue;
    const seg = segOf(e.t), A = { a: Math.max(seg.a, e.t - 2.4), b: Math.min(seg.b, e.t + (e.last ? 1.7 : 0.9)), kind: 'ring', c: cps[e.index] };
    if (anchors.some((x) => A.a < x.b + 0.5 && A.b > x.a - 0.5)) continue;
    anchors.push(A); lastRing = e.t;
    if (e.last) zones.push({ a: e.t - 0.7, b: e.t + 0.6, r: 0.35 });
  }
  for (const e of evs) if (e.type === 'pop') zones.push({ a: e.t - 0.55, b: e.t + 0.3, r: 0.4 });      // the round on its way to a balloon, slowly
  anchors.sort((x, y) => x.a - y.a);

  // ---- cut the flight into shots ----
  const shots = [], hist = [];
  const go = evs.find((e) => e.type === 'go');
  let t = t0, guard = 0;
  while (t < t1 - 0.05 && guard++ < 4000) {
    const seg = segOf(t + 2e-3), mk = maker(seg);
    const A = anchors.find((x) => x.b > t + 0.4 && !x.used);
    const pick = (type, a, b, tries, maxD, tol) => {
      for (let i = 0; i < tries; i++) {
        const side = (i % 2 ? -1 : 1) * (rnd() < 0.5 ? -1 : 1);
        const cam = type === 'ring' ? mk.ring(A.c, side) : type === 'crash' ? mk.crash(A.t, side) : mk[type](a, b, side);
        if (type === 'onboard' || blocked(cam, mk.P, a, b, maxD, type === 'orbit') <= tol) return { a, b, type, cam };
      }
      return null;
    };
    let shot = null;
    if (A && A.a - t < 1.7 && A.a <= seg.b) {
      A.used = true;
      if (A.kind === 'ring' && hist[hist.length - 1] === 'ring') continue;      // two rings close together: the second one gets an ordinary shot
      shot = pick(A.kind, t, A.b, 6, 500, 2);
      if (!shot) continue;                                   // no clear view from there: ordinary shots take over
    } else {
      const limit = Math.min(seg.b, A ? A.a : Infinity, t1);
      const mid = mk.P(t + 1.5), V = len(mid.v), slow = V < 7 && len(sub(mk.P(t + 3).p, mk.P(t).p)) < 18, inside = indoor(mid.p) > 0.5;
      const first = !shots.length && rec.at(t).state === 'countdown';
      const ended = rec.at(t).state === 'finished';
      const w = slow || ended ? { orbit: 8, flyby: 1, track: 1, heli: 0.5, lead: 0.5, onboard: 0.3 }
        : { flyby: 3, headon: 1.5, chase: 1.5, lead: 1.3, track: 1.8, heli: mid.agl > 25 ? 2.2 : 0.7, onboard: 1.3 };
      if (inside) { w.heli = 0; w.chase = (w.chase || 0) + 1.5; }
      for (const [back, k] of [[1, 0], [2, 0.45]]) { const h = hist[hist.length - back]; if (w[h]) w[h] *= k; }
      for (let n = 0; n < 9 && !shot; n++) {
        let type;
        if (first && n === 0) type = 'open';
        else { let x = rnd() * Object.values(w).reduce((s, v) => s + v, 0); for (const k in w) { x -= w[k]; if (x <= 0) { type = k; break; } } }
        if (!type) break;
        let b = type === 'open' ? Math.min(go ? go.t + 0.9 : t + 4, t + 5) : t + R(...DUR[type]);
        if (limit - b < 1.7) b = limit;
        b = Math.min(b, limit);
        const still = type === 'flyby' || type === 'headon';
        shot = pick(type, t, b, 3, type === 'heli' ? 400 : still ? 420 : 60, still ? 2 : 1);
        if (type !== 'open') w[type] = 0;
      }
      if (!shot) { const b = Math.min(limit, t + R(...DUR.onboard) > limit - 1.7 ? limit : t + R(...DUR.onboard)); shot = pick('onboard', t, b, 1, 0, 0); }
    }
    hist.push(shot.type); shots.push(shot); t = shot.b;
  }
  if (shots.length) shots[shots.length - 1].b = t1;

  // slow motion through the crashes and the last ring
  const rate = (t) => {
    let r = 1;
    for (const z of zones) { const k = clamp(Math.min(t - z.a, z.b - t) / 0.3, 0, 1); r = Math.min(r, lerp(1, z.r, k)); }
    return r;
  };
  return { shots, rate, t0, t1, shotAt(t) { let lo = 0, hi = shots.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (shots[m].a <= t) lo = m; else hi = m - 1; } return shots[lo]; } };
}
