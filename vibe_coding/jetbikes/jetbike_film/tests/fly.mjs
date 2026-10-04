import { runSimulation } from '../js/sim.js';
import { GAINS } from '../js/control.js';
if (process.env.G) for (const [k, v] of Object.entries(JSON.parse(process.env.G))) { if (Array.isArray(v)) v.forEach((x, i) => GAINS[k][i] = x); else GAINS[k] = v; }
import { surfaceAt } from '../js/terrain.js';
import { qrot, len, sub } from '../js/vmath.js';
const t0 = Date.now();
const rec = runSimulation();
console.log('sim time', ((Date.now() - t0) / 1000).toFixed(2), 's; duration', rec.duration.toFixed(1), 'events', JSON.stringify(rec.events), 'path', rec.mission.pathLength.toFixed(0), 'm');
let maxErr = 0, minAgl = 1e9, maxTilt = 0;
const every = +(process.argv[2] || 1);
for (let t = 0; t <= rec.duration; t += 0.05) {
  const s = rec.at(t);
  const err = len(sub(s.ref, s.p));
  const up = qrot(s.q, [0, 1, 0]); const fwd = qrot(s.q, [1, 0, 0]);
  const tilt = Math.acos(up[1]) * 57.3;
  const agl = s.p[1] - surfaceAt(s.p[0], s.p[2]);
  if (t > 11 && t < rec.events.touchdown - 2) { maxErr = Math.max(maxErr, err); minAgl = Math.min(minAgl, agl); maxTilt = Math.max(maxTilt, tilt); }
  if (Math.abs(t / every - Math.round(t / every)) < 1e-6) {
    const pitch = Math.asin(fwd[1]) * 57.3; const right = qrot(s.q, [0, 0, 1]); const bank = Math.asin(-right[1]) * 57.3;
    console.log(`t=${t.toFixed(1).padStart(5)} p=(${s.p.map(v => v.toFixed(1)).join(',')}) err=${err.toFixed(2)} spd=${(len(s.v) * 3.6).toFixed(0)}kmh agl=${agl.toFixed(1)} pitch=${pitch.toFixed(1)} bank=${bank.toFixed(1)} T=[${s.T.map(v => v.toFixed(0)).join(' ')}] main=${s.mainT.toFixed(0)} fuel=${s.fuel.toFixed(1)} Nc=${s.contact.toFixed(0)} w=${len(s.w).toFixed(2)}`);
  }
}
console.log('maxErr', maxErr.toFixed(2), 'minAGL', minAgl.toFixed(2), 'maxTilt', maxTilt.toFixed(1));
