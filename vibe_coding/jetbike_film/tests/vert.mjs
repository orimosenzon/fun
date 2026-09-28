import { buildMission } from '../js/mission.js';
import { surfaceAt } from '../js/terrain.js';
const m = buildMission(0.863);
let maxA = 0, minA = 0, minAgl = 1e9;
for (let t = m.tPath0; t < m.tPath1; t += 0.1) { const r = m.sample(t); maxA = Math.max(maxA, r.a[1]); minA = Math.min(minA, r.a[1]); minAgl = Math.min(minAgl, r.p[1] - 0.863 - surfaceAt(r.p[0], r.p[2])); }
console.log('vertical acc range', minA.toFixed(2), maxA.toFixed(2), 'min AGL', minAgl.toFixed(1), 'tPath1', m.tPath1.toFixed(1));
for (let t = 44; t < 100; t += 4) { const r = m.sample(t); console.log(t, 'y', r.p[1].toFixed(1), 'agl', (r.p[1] - surfaceAt(r.p[0], r.p[2])).toFixed(1), 'vy', r.v[1].toFixed(1), 'ay', r.a[1].toFixed(2)); }
