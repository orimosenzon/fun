import { buildMission } from '../js/mission.js';
import { surfaceAt } from '../js/terrain.js';
const m = buildMission(0.863);
console.log('tPath0', m.tPath0.toFixed(1), 'tPath1', m.tPath1.toFixed(1), 'touch', m.tTouchRef.toFixed(1), 'len', m.pathLength.toFixed(0));
for (let t = m.tPath0; t < m.tPath1; t += 3) { const r = m.sample(t); console.log(t.toFixed(0), r.p.map(v => v.toFixed(0)).join(','), 'agl', (r.p[1] - surfaceAt(r.p[0], r.p[2])).toFixed(1), 'v', Math.hypot(...r.v).toFixed(1), 'a', Math.hypot(...r.a).toFixed(2)); }
