// The nose gun and the balloon range, headless. node tests/gun.mjs
import { loadWorld, deg } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { buildScenery, hitScenery } from '../js/sim/scenery.js';
import { GUN } from '../js/sim/gun.js';
import { FEEL } from '../js/sim/fcs.js';
import { surfaceAt } from '../js/sim/terrain.js';
import { qFromYPR, qrot, yprOf, len, sub, dot } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
let fail = 0;
const ok = (c, m) => { if (!c) fail++; console.log(`${c ? 'ok  ' : 'FAIL'} ${m}`); };
const drain = (g, log) => { for (const e of g.events) log.push(e); g.events.length = 0; };

// --- the range itself: every balloon in free air, above the trees around it, reachable ---
const g = new Game(); g.load('hunt');
ok(g.cps.length === 10 && g.cps.every((c) => c.type === 'target'), `ten balloons (${g.cps.length})`);
ok(g.cps.every((c) => !hitScenery(c.p, 2.5) && c.p[1] - surfaceAt(c.p[0], c.p[2]) > 15), 'every balloon hangs in free air');

// put the bike somewhere, pointing the gun so the round arrives at the balloon (drop allowed for)
const aimAt = (c, dist, miss = 0) => {
  const b = g.bike, yaw = 0.4, T = dist / FEEL.muzzle, drop = 0.5 * 9.81 * T * T;
  const from = [c.p[0] - Math.cos(yaw) * dist, c.p[1] - 0.74, c.p[2] + Math.sin(yaw) * dist];       // the muzzle is 0.74 above... the cg correction is small
  let q = qFromYPR(yaw, Math.atan2(drop + miss, dist), 0);
  if (qrot(q, [1, 0, 0])[1] < 0) q = qFromYPR(yaw, -Math.atan2(drop + miss, dist), 0);
  b.p = from; b.q = q; b.v = [0, 0, 0]; b.w = [0, 0, 0]; b.heat = 0; b.abLock = false; g.state = 'fly'; g.gun.cool = 0; g.gun.shots.length = 0;
  // aim exactly: correct for where the muzzle really is
  const m = b.toWorld([1.22, 0.74, 0]), f = qrot(q, [1, 0, 0]);
  b.p = [b.p[0] + (c.p[0] - (m[0] + f[0] * dist)), b.p[1] + (c.p[1] + drop + miss - (m[1] + f[1] * dist)), b.p[2] + (c.p[2] - (m[2] + f[2] * dist))];
};
const shoot = () => { const log = []; g.gun.trigger = true; for (let i = 0; i < 600 && (i === 0 || g.gun.shots.length); i++) { g.gun.step(1 / 120, g); drain(g, log); } return log; };

for (const dist of [80, 200, 350]) {
  g.restart(); g.events.length = 0;
  const c = g.cps[3]; aimAt(c, dist);
  const log = shoot(), pop = log.find((e) => e.type === 'pop'), end = log.find((e) => e.type === 'shotEnd');
  ok(pop && pop.k === 3 && g.cp === 1 && g.cps[0] === c && g.penalty === 0, `a well-aimed round pops the balloon at ${dist} m (any order: it becomes number 1), flight ${(dist / FEEL.muzzle).toFixed(2)} s, drop ${(0.5 * 9.81 * (dist / FEEL.muzzle) ** 2).toFixed(1)} m`);
  if (!pop) console.log('     ended:', end && end.end, end && sub(end.p, c.p).map((v) => v.toFixed(1)));
}
g.restart(); g.events.length = 0; aimAt(g.cps[3], 200, 4);
{ const log = shoot(); ok(!log.some((e) => e.type === 'pop') && g.penalty === GUN.missPenalty && g.cp === 0, `4 m high at 200 m is a miss, and costs ${GUN.missPenalty} s`); }
// no gravity compensation at 350 m: the round falls short of the centre by more than the balloon
g.restart(); g.events.length = 0;
{ const c = g.cps[3]; aimAt(c, 350, -0.5 * 9.81 * (350 / FEEL.muzzle) ** 2); const log = shoot(); ok(!log.some((e) => e.type === 'pop'), 'aiming straight at a balloon 350 m away falls under it: the drop has to be allowed for'); }
// rate of fire and heat
g.restart(); g.events.length = 0; aimAt(g.cps[0], 200, 30);
{ g.gun.trigger = true; g.gun.step(1 / 120, g); g.gun.trigger = true; g.gun.step(1 / 120, g); ok(g.gun.fired === 1, 'one round at a time'); let n = 0; for (let i = 0; i < 120 * 30; i++) { g.gun.trigger = true; g.gun.step(1 / 120, g); if (g.bike.heat >= 1) g.bike.abLock = true; } n = g.gun.fired; ok(n >= 4 && n <= 8, `held trigger without cooling: ${n} rounds until the hot core stops the gun`); }

// --- the whole event: all ten, the clock stops, shots are counted ---
g.restart(); g.events.length = 0;
{ const log = []; for (const c of g.cps0) { aimAt(c, 60); const l = shoot(); log.push(...l); if (!l.some((e) => e.type === 'pop')) console.log('     balloon', c.k, 'not hit:', JSON.stringify(l.find((e) => e.type === 'shotEnd'))); } ok(g.state === 'finished' && g.result.shots === 10 && g.result.hits === 10 && log.filter((e) => e.type === 'gate').length === 10, `ten hits finish the event (${g.state}, ${g.result ? g.result.hits : 0}/${g.result ? g.result.shots : 0})`); }

// --- recoil in real flight: the shot disturbs the bike and the flight computer brings it back ---
for (const [name, speed] of [['hover', 0], ['at 40 m/s', 40]]) {
  const h = new Game(); h.load('hunt'); h.state = 'fly';
  const st = h.build.start; h.sim.reset([st.p[0], st.p[1] + 60, st.p[2]], 0, Math.max(speed, 1)); h.vt = speed; h.bike.tilt = speed ? h.bike.tilt : Math.PI / 2;
  for (let i = 0; i < 60 * 6; i++) h.update(1 / 60, { vt: speed }); h.events.length = 0;
  const p0 = yprOf(h.bike.q)[1], f0 = dot(h.bike.v, qrot(h.bike.q, [1, 0, 0])), y0 = h.bike.p[1];
  let dp = 0, dv = 0, up = 0;
  for (let i = 0; i < 60 * 2; i++) { h.update(1 / 60, { vt: speed, fire: i === 0 }); const p = yprOf(h.bike.q)[1] - p0; if (Math.abs(p) > Math.abs(dp)) dp = p; dv = Math.min(dv, dot(h.bike.v, qrot(h.bike.q, [1, 0, 0])) - f0); up = Math.max(up, Math.abs(h.bike.p[1] - y0)); }
  const left = yprOf(h.bike.q)[1] - p0, noseUp = qrot(h.bike.q, [1, 0, 0])[1];
  ok(h.gun.fired === 1 && Math.abs(dp) > 0.004 && Math.abs(dp) < 0.2 && Math.abs(left) < 0.02 && dv < -0.3 && h.state === 'fly', `${name}: a shot throws the nose ${deg(dp).toFixed(1)} deg, takes ${(-dv).toFixed(1)} m/s, moves the height ${up.toFixed(1)} m; 2 s later the nose is back within ${deg(Math.abs(left)).toFixed(2)} deg`);
  if (name === 'hover') ok((dp > 0) === (qrot(qFromYPR(0, 0.1, 0), [1, 0, 0])[1] > 0), 'the nose is thrown up, not down');
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
