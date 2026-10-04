import { buildMission } from '../js/mission.js';
const m = buildMission(0.863);
for (let t = 50; t < 80; t += 1) { const r = m.sample(t); const v = Math.hypot(...r.v); const a = r.a; 
  // lateral accel = |a - (a.v)v/|v|^2|
  const av = (a[0]*r.v[0]+a[1]*r.v[1]+a[2]*r.v[2])/(v*v); const al = Math.hypot(a[0]-av*r.v[0], a[1]-av*r.v[1], a[2]-av*r.v[2]);
  console.log(t, r.p.map(x=>x.toFixed(0)).join(','), 'v', v.toFixed(1), 'alat', al.toFixed(2), 'R', (v*v/al).toFixed(0), 'yaw', (r.yaw*57.3).toFixed(0)); }
