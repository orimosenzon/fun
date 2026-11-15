// The flight recorder and the replay director, headless. node tests/replay.mjs [id] [-v]
// Flies events with the demo rider while recording, then checks that the recording plays back
// exactly, that the shots cover the whole flight without gaps, and that in (nearly) every
// moment of every shot the camera is in free air and has the bike in view.
import { loadWorld } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { Autopilot } from '../js/sim/autopilot.js';
import { buildScenery, hitScenery } from '../js/sim/scenery.js';
import { hitCastle } from '../js/sim/castle.js';
import { surfaceAt, groundAt } from '../js/sim/terrain.js';
import { Recorder } from '../js/sim/replay.js';
import { planReplay } from '../js/sim/director.js';
import { len, sub, norm, dot } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
const only = process.argv[2] && !process.argv[2].startsWith('-') ? process.argv[2] : null, verbose = process.argv.includes('-v');
let fail = 0;
const bad = (m) => { fail++; console.log('  FAIL ' + m); };
for (const [id, level] of [['first', 1], ['lake', 3], ['castle', 1], ['pass', 2]]) {
  if (only && id !== only) continue;
  const g = new Game(); g.windLevel = level; g.load(id);
  const ap = new Autopilot(g), rec = new Recorder(g);
  const truth = [];
  let tail = 0;
  while (tail < 4 && g.total < 500) {
    g.update(1 / 60, g.state === 'fly' ? ap.update() : {});
    rec.push(1 / 60, 0);
    for (const e of g.events) { if (['crash', 'gate', 'land', 'respawn', 'go', 'miss'].includes(e.type)) rec.mark(e.type, e); if (e.type === 'respawn') ap.reset(); }
    g.events.length = 0;
    if (rec.n % 97 === 0) truth.push({ t: rec.t, p: g.bike.p.slice(), q: g.bike.q.slice(), L: g.bike.L.slice(), N: g.bike.eng.N, cp: g.cp, state: g.state });
    if (g.state === 'finished') tail += 1 / 60;
  }
  // --- playback is exact, and leaves the game as it found it ---
  const live = rec.snapshot(), pEnd = g.bike.p.slice();
  let err = 0;
  for (const s of truth) { rec.apply(s.t); err = Math.max(err, len(sub(g.bike.p, s.p)), Math.abs(g.bike.eng.N - s.N), Math.abs(g.bike.L[0] - s.L[0]) / 1000); if (g.cp !== s.cp || g.state !== s.state) err = 9; }
  if (err > 0.01) bad(`${id}: playback differs from the flight by ${err}`);
  rec.restore(live);
  if (len(sub(g.bike.p, pEnd)) > 0.01 || g.state !== 'finished') bad(`${id}: state not restored after the replay`);

  // --- a flight kept in the history comes back the same, also when the bike has changed since ---
  const kept = structuredClone(rec.dump()), g2 = new Game(); g2.load(id);
  const rec2 = new Recorder(g2);
  if (!rec2.load(kept) || rec2.n !== rec.n - rec.base || Math.abs(rec2.duration - rec.duration) > 1e-3) bad(`${id}: kept flight did not load`);
  else { const tm = rec.t0 + rec.duration * 0.37, x = rec.at(tm), y = rec2.at(tm); if (len(sub(x.p, y.p)) > 1e-3 || x.cp !== y.cp || rec2.events.length !== rec.events.length) bad(`${id}: kept flight differs`); }
  const old = structuredClone(kept); old.layout = [['b', 'gone', 2], ...old.layout.filter((f) => f[1] !== 'heat')];       // a part that no longer exists, and one that did not exist yet
  { const w = old.W + 2 - 1, n = old.frames.length / old.W, f = new Float32Array(n * w), hi = kept.layout.findIndex((q) => q[1] === 'heat'); let ho = kept.hdr; for (let k = 0; k < hi; k++) ho += kept.layout[k][2];
    for (let i = 0; i < n; i++) { const r = kept.frames.subarray(i * kept.W, (i + 1) * kept.W); f.set(r.subarray(0, kept.hdr), i * w); f.set(r.subarray(kept.hdr, ho), i * w + kept.hdr + 2); f.set(r.subarray(ho + 1), i * w + ho + 2); }
    old.frames = f; old.W = w; }
  if (!rec2.load(old)) bad(`${id}: flight from an older bike did not load`);
  else { const tm = rec.t0 + rec.duration * 0.61, x = rec.at(tm), y = rec2.at(tm); rec2.apply(tm); if (len(sub(x.p, y.p)) > 1e-3 || len(sub(x.q, y.q)) > 1e-3 || g2.bike.heat !== 0) bad(`${id}: flight from an older bike plays wrong`); }

  // --- the edit ---
  for (let seed = 1; seed <= 3; seed++) {
    let s = seed * 7919; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const plan = planReplay(rec, g.cps, g.bike.mp.cg, rnd), sh = plan.shots;
    if (!sh.length || Math.abs(sh[0].a - rec.t0) > 1e-6 || Math.abs(sh[sh.length - 1].b - rec.t1) > 1e-6) bad(`${id}: shots do not span the flight`);
    let hidden = 0, total = 0, off = 0, same = 0, short = 0, long = 0;
    const kinds = {};
    sh.forEach((x, i) => {
      kinds[x.type] = (kinds[x.type] || 0) + 1;
      if (i && Math.abs(x.a - sh[i - 1].b) > 1e-6) bad(`${id}: gap or overlap at shot ${i}`);
      if (i && x.type === sh[i - 1].type && x.type !== 'onboard') { same++; if (process.env.DBG) console.log("   same", x.type, sh[i - 1].a.toFixed(1), x.a.toFixed(1), x.b.toFixed(1)); }
      const d = x.b - x.a; if (d < 1.2) short++; if (d > 9.5) long++;
      const o = {};
      for (let k = 0; k <= 20; k++) {
        const t = x.a + (x.b - x.a) * (k + 0.5) / 21; x.cam(t, o); const b = rec.at(t);
        total++;
        if (![...o.p, ...o.l, ...o.up, o.fov].every(Number.isFinite)) { bad(`${id}: ${x.type} gives a broken camera`); return; }
        if (x.type === 'onboard') continue;
        if (o.p[1] < surfaceAt(o.p[0], o.p[2]) + 0.2 || hitCastle(o.p, 0.3) || hitScenery(o.p, 0.3)) bad(`${id}: ${x.type} camera inside something at t=${t.toFixed(1)}`);
        // the bike inside the picture
        const view = norm(sub(o.l, o.p)), to = sub(b.p, o.p), ang = Math.acos(Math.min(1, dot(view, norm(to)))) * 180 / Math.PI;
        if (ang > o.fov * 0.5) off++;
        const n = Math.max(6, Math.ceil(len(to) / 5)); let blocked = false;
        for (let j = 1; j < n && !blocked; j++) { const u = j / n, q = [o.p[0] + to[0] * u, o.p[1] + (to[1] + 0.6) * u, o.p[2] + to[2] * u]; blocked = q[1] < groundAt(q[0], q[2]) || !!hitCastle(q, 0.05) || !!hitScenery(q, 0); }
        if (blocked) hidden++;
      }
    });
    if (same > 0) bad(`${id}: ${same} cuts to the same kind of shot`);
    if (hidden / total > 0.08) bad(`${id}: bike hidden in ${(100 * hidden / total).toFixed(0)}% of the samples`);
    if (off / total > 0.06) bad(`${id}: bike out of frame in ${(100 * off / total).toFixed(0)}% of the samples`);
    if (short > 2 || long > 0) bad(`${id}: ${short} shots under 1.2 s, ${long} over 9.5 s`);
    if (seed === 1 || verbose) console.log(`${id.padEnd(7)} ${rec.duration.toFixed(0)} s, ${rec.n} frames x ${rec.W} numbers, ${g.crashes} crashes · seed ${seed}: ${sh.length} shots, mean ${(rec.duration / sh.length).toFixed(1)} s, hidden ${(100 * hidden / total).toFixed(1)}%, out of frame ${(100 * off / total).toFixed(1)}% · ${Object.entries(kinds).map(([k, v]) => k + ' ' + v).join(', ')}`);
    if (verbose) console.log(sh.map((x) => `    ${x.a.toFixed(1).padStart(6)} ${(x.b - x.a).toFixed(1).padStart(4)}s ${x.type}`).join('\n'));
  }
}
console.log(fail ? `${fail} FAILED` : 'replay: all good');
process.exit(fail ? 1 : 0);
