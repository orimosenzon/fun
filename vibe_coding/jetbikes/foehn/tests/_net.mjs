// The safety net of aerobatic mode: random aerobatics low over the valley. node tests/_net.mjs [runs] [secs] [key=value]
import { loadWorld, deg } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { buildScenery } from '../js/sim/scenery.js';
import { groundAt, surfaceAt, T } from '../js/sim/terrain.js';
import { FEEL, ACRO, NET } from '../js/sim/fcs.js';
import { mulberry32 } from '../js/sim/noise.js';
import { len, qrot, yprOf } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
const runs = +(process.argv[2] || 40), secs = +(process.argv[3] || 60), verbose = process.argv.includes('-v');
for (const a of process.argv.slice(4)) { const [k, v] = a.split('='); for (const o of [FEEL, ACRO, NET]) if (k in o) o[k] = +v; }
let crashes = 0, bails = 0, acroT = 0, lowest = 1e9, recMax = 0;
for (let seed = 1; seed <= runs; seed++) {
  const r = mulberry32(seed * 7919), g = new Game();
  g.load('free'); g.update(0.02, {}); g.events.length = 0;
  const L = T.lakes[0], flat = process.env.FLAT, cx = (L.x0 + L.x1) / 2, cz = (L.z0 + L.z1) / 2;
  let x, z; if (flat) { x = cx; z = cz; } else do { x = (r() - 0.5) * 7000; z = (r() - 0.5) * 7000; } while (surfaceAt(x, z) > 1900);
  const h0 = 60 + r() * 240, V0 = 25 + r() * 70;
  g.sim.reset([x, surfaceAt(x, z) + h0, z], r() * 6.28, V0); g.state = 'fly'; g.prevP = g.bike.p.slice();
  const b = g.bike;
  let ctl = { turn: 0, pitch: 0, strafe: 0, vt: V0 }, next = 0, t = 0, bailT = -1, bailH = 0, wasAcro = false;
  while (t < secs && g.state === 'fly') {
    if (!process.env.PLAIN && !g.acro && !g.sim.fcs.rec && (bailT < 0 || t - bailT > 2.5)) { if (bailT >= 0) recMax = Math.max(recMax, 0); g.toggleAcro(true); }
    if (t >= next) {
      const pick = () => [-1, -1, 0, 0, 1, 1, 1][(r() * 7) | 0];
      ctl = { turn: r() < 0.6 ? pick() : 0, pitch: r() < 0.7 ? pick() : 0, strafe: r() < 0.15 ? pick() : 0, vt: [20, 40, 55, 70, 95][(r() * 5) | 0], boost: r() < 0.2 };
      const lim = T.half - 600; if (Math.abs(b.p[0]) > lim || Math.abs(b.p[2]) > lim || b.p[1] > 3300 || (flat && (Math.hypot(b.p[0] - x, b.p[2] - z) > 300 || b.agl > 500))) { g.sim.reset([x, surfaceAt(x, z) + h0, z], r() * 6.28, V0); g.prevP = b.p.slice(); }
      next = t + 0.3 + r() * 2.2;
    }
    g.update(1 / 60, ctl); t += 1 / 60;
    if (+process.env.DUMP === seed && bailT >= 0 && t - bailT < 3 && Math.round(t * 60) % 6 === 0) { const f = g.sim.fcs, y = yprOf(b.q); console.log(`   t=${(t - bailT).toFixed(2)} agl=${b.agl.toFixed(1)} V=${len(b.v).toFixed(0)} vz=${b.v[1].toFixed(1)} up=${qrot(b.q, [0, 1, 0])[1].toFixed(2)} ypr=${y.map((v) => deg(v).toFixed(0))} des=${deg(f.tele.thetaDes).toFixed(0)},${deg(f.tele.phiDes).toFixed(0)} err=${deg(f.tele.attErr).toFixed(0)} w=${b.w.map((v) => v.toFixed(1))} a=${f.tele.aCmd.map((v) => v.toFixed(0))} vzDes=${f.tele.vzDes.toFixed(0)} fl=${f.tele.floor} L=${b.lift.toFixed(0)} R=${b.R.toFixed(0)} tilt=${deg(b.tilt).toFixed(0)} N=${b.eng.N.toFixed(2)} sat=${f.tele.sat} vt=${g.vt.toFixed(0)}`); }
    if (g.acro) { acroT += 1 / 60; lowest = Math.min(lowest, b.agl); }
    for (const e of g.events) {
      if (e.type === 'acro' && !e.on && e.why === 'net') { bails++; bailT = t; bailH = b.agl; if (verbose) console.log(`  seed ${seed} t=${t.toFixed(1)} net at agl=${b.agl.toFixed(0)} V=${len(b.v).toFixed(0)} vz=${b.v[1].toFixed(0)} up=${qrot(b.q, [0, 1, 0])[1].toFixed(2)}`); }
      if (e.type === 'crash') { crashes++; console.log(`seed ${seed} t=${t.toFixed(1)} CRASH ${e.reason} ${g.acro ? 'in acro' : `${(t - bailT).toFixed(1)} s after the net (it took over at agl=${bailH.toFixed(0)})`} V=${len(e.v).toFixed(0)} vz=${e.v[1].toFixed(0)} agl=${b.agl.toFixed(1)}`); }
    }
    g.events.length = 0;
  }
}
console.log(`${runs} runs x ${secs} s: ${crashes} crashes, net took over ${bails} times, ${(acroT / 60).toFixed(1)} min of aerobatics, lowest ${lowest.toFixed(1)} m`);
