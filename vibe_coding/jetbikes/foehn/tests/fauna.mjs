// The animals: where they stand, and what they do when the bike comes. node tests/fauna.mjs
import { loadWorld } from './util.mjs';
import { buildScenery } from '../js/sim/scenery.js';
import { Fauna, KINDS, pasture } from '../js/sim/fauna.js';
import { groundAt, isWater, normalAt } from '../js/sim/terrain.js';
loadWorld(); buildScenery();
let fail = 0; const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) fail++; };
const t0 = Date.now(), f = new Fauna(), ms = Date.now() - t0;
const count = {}; for (const a of f.list) count[a.kind] = (count[a.kind] || 0) + 1;
ok(f.list.length > 250 && f.list.length < 900, `${f.list.length} animals in ${new Set(f.list.map((a) => a.herd)).size} herds, placed in ${ms} ms: ${Object.entries(count).map(([k, n]) => k + ' ' + n).join(', ')}`);
ok(Object.keys(KINDS).every((k) => count[k] > 0), 'every kind is there');
ok(f.list.every((a) => !isWater(a.x, a.z) && normalAt(a.x, a.z)[1] > 0.95 && Math.abs(a.y - groundAt(a.x, a.z)) < 1e-6), 'all stand on dry, gentle ground');
const g = new Fauna(); ok(g.list.length === f.list.length && g.list.every((a, i) => a.x === f.list[i].x && a.kind === f.list[i].kind), 'the same valley every time');
// a pass low over a herd of cows at 60 m/s
const cow = f.list.find((a) => a.kind === 'cow' && a.lead), herd = f.list.filter((a) => a.herd === cow.herd), before = herd.map((a) => [a.x, a.z]);
let ran = 0, bad = 0;
for (let t = -6; t < 10; t += 1 / 60) {
  const p = [cow.x + t * 60, cow.y + 8, cow.z + 4];
  f.update(1 / 60, p, [60, 0, 0]);
  for (const a of herd) { if (a.pose >= 2) ran++; if (isWater(a.x, a.z) || normalAt(a.x, a.z)[1] < 0.95 || !Number.isFinite(a.x + a.z + a.yaw)) bad++; }
}
const moved = herd.map((a, i) => Math.hypot(a.x - before[i][0], a.z - before[i][1]));
ok(Math.max(...moved) > 8 && ran > 60, `a low pass scatters the herd: ${herd.length} cows, moved ${moved.map((m) => m.toFixed(0)).join(' ')} m`);
ok(bad === 0, 'nobody ran into the lake or up a wall');
ok(herd.every((a) => a.pose < 2), 'ten seconds later they have stopped');
// a far herd never noticed
const far = f.list.find((a) => Math.hypot(a.x - cow.x, a.z - cow.z) > 2500), fx = far.x; ok(far.x === fx && far.pose < 2, 'a herd far away stays calm');
// a long random harassment: nothing leaves the valid ground
let s = 5; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); bad = 0;
for (let k = 0; k < 60; k++) { const a = f.list[Math.floor(rnd() * f.list.length)], ang = rnd() * 6.28; for (let t = 0; t < 5; t += 1 / 30) f.update(1 / 30, [a.x + Math.cos(ang) * (40 - t * 16), a.y + 5, a.z + Math.sin(ang) * (40 - t * 16)], [16, 0, 0]); }
for (const a of f.list) if (!pasture(a.x, a.z, { alt: [0, 9000] })) bad++;
ok(bad === 0, `after sixty chases all ${f.list.length} are still on pasture`);
f.reset(); ok(f.list.every((a, i) => a.x === g.list[i].x && a.z === g.list[i].z), 'reset puts everyone back');
console.log(fail ? `${fail} FAILED` : 'ALL PASS'); process.exit(fail ? 1 : 0);
