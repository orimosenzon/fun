// סריקת מרחב העבודה של יד ימין באחיזת "לחיצת יד": לכל מרחק קדימה וגובה, שגיאת מיקום וסיבוב
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { rotZYX, rotError } from '../js/ctrl/linalg.js';
import { feetMid } from '../js/tasks/prims.js';
const crouch = +(process.argv[2] || 0.12), lean = +(process.argv[3] || 0.06), lat = +(process.argv[4] || -0.2);
const R = rotZYX(0, -Math.PI / 2, 0);
const fwds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4], zs = [0.85, 0.92, 1.0, 1.08];
console.log('crouch', crouch, 'lean', lean, 'lateral', lat, '   cells: posErr(cm)/rotErr(deg)');
console.log('z\\fwd  ' + fwds.map(f => f.toFixed(2).padStart(9)).join(''));
for (const z of zs) {
  let row = z.toFixed(2) + '  ';
  for (const fw of fwds) {
    const w = new World(mj, {}); const r = new Robot(w); const m = w.model, d = w.data;
    r.lean = lean; r.crouch = crouch;
    const fm = feetMid(r), pos = [fm[0] + fw, fm[1] + lat, z];
    for (let i = 0; i < 1200; i++) { r.handTargets.r = { pos, R, w: 120, kp: 220, wr: 30 }; mj.mj_step1(m, d); r.control(); mj.mj_step2(m, d); }
    const g = r.wbc.sitePos(r.wbc.ids.r_grip), e = rotError(R, r.wbc.bodyMat(r.wbc.ids.r_hand));
    row += `${(Math.hypot(g[0] - pos[0], g[1] - pos[1], g[2] - pos[2]) * 100).toFixed(0)}/${(Math.hypot(...e) * 57.3).toFixed(0)}`.padStart(9);
    w.dispose();
  }
  console.log(row);
}
