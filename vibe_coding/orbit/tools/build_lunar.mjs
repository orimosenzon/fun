// build_lunar.mjs — מחשב מסלולי ירח במודל תלת־גופי מישורי (כדור הארץ + ירח במסלול מעגלי)
// ושומר אותם ל-js/lunar_data.js. הרצה: node tools/build_lunar.mjs
import { writeFileSync } from 'fs';

const MU_E = 3.986004418e14, MU_M = 4.9048695e12;
const D = 384400e3, RE = 6378137, RM = 1737400;
const W = Math.sqrt((MU_E + MU_M) / D ** 3); // מהירות זוויתית של הירח (כוכבית, ≈ 27.3 ימים)

const moonAt = t => [D * Math.cos(W * t), D * Math.sin(W * t)];

function acc(t, x, y) {
  const [mx, my] = moonAt(t);
  const r3 = Math.hypot(x, y) ** 3;
  const dx = x - mx, dy = y - my;
  const d3 = Math.hypot(dx, dy) ** 3;
  const m3 = D ** 3; // תאוצה עקיפה: כדור הארץ מואץ לכיוון הירח (מערכת מרכזת בכדור הארץ)
  return [-MU_E * x / r3 - MU_M * dx / d3 - MU_M * mx / m3, -MU_E * y / r3 - MU_M * dy / d3 - MU_M * my / m3];
}

// RK4 עם צעד מסתגל לפי המרחק מהגופים
function propagate(s0, t0, tEnd, stop) {
  let [x, y, vx, vy] = s0, t = t0;
  const out = [[t, x, y, vx, vy]];
  while (t < tEnd) {
    const [mx, my] = moonAt(t);
    const rE = Math.hypot(x, y), rM = Math.hypot(x - mx, y - my);
    let h = Math.min(rE / Math.hypot(vx, vy) * 0.01, rM / Math.hypot(vx, vy) * 0.01, 600);
    h = Math.max(h, 1);
    const f = (tt, X, Y, VX, VY) => { const a = acc(tt, X, Y); return [VX, VY, a[0], a[1]]; };
    const k1 = f(t, x, y, vx, vy);
    const k2 = f(t + h / 2, x + h / 2 * k1[0], y + h / 2 * k1[1], vx + h / 2 * k1[2], vy + h / 2 * k1[3]);
    const k3 = f(t + h / 2, x + h / 2 * k2[0], y + h / 2 * k2[1], vx + h / 2 * k2[2], vy + h / 2 * k2[3]);
    const k4 = f(t + h, x + h * k3[0], y + h * k3[1], vx + h * k3[2], vy + h * k3[3]);
    x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    vx += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    vy += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
    t += h;
    out.push([t, x, y, vx, vy]);
    if (stop && stop(t, x, y, vx, vy)) break;
  }
  return out;
}

// מצב התחלתי: מסלול חניה מעגלי, הבערת TLI בזווית phi (ביחס לכיוון הירח בזמן 0) עם dv
function tliState(alt, phi, dv) {
  const r = RE + alt, v = Math.sqrt(MU_E / r) + dv;
  return [r * Math.cos(phi), r * Math.sin(phi), -v * Math.sin(phi), v * Math.cos(phi)];
}
function perilune(traj) {
  let best = { d: 1e20 };
  for (const p of traj) {
    const [mx, my] = moonAt(p[0]);
    const d = Math.hypot(p[1] - mx, p[2] - my);
    if (d < best.d) best = { d, p };
  }
  return best;
}
// האם המעבר ליד הירח הוא מאחוריו (הצד הרחוק): המכפלה הווקטורית של (מיקום יחסי) ו(מהירות יחסית)
function passSense(pl) {
  const [t, x, y, vx, vy] = pl.p;
  const [mx, my] = moonAt(t);
  const vmx = -W * my, vmy = W * mx;
  return (x - mx) * (vy - vmy) - (y - my) * (vx - vmx);
}

function solve(cfg) {
  // לכל גודל הבערה dv: סריקת זווית ההבערה phi ואז חיתוך בינארי כדי לפגוע בגובה הפריסלנה המבוקש
  // מהצד הנכון של הירח. מבין הפתרונות בוחרים את זה שמתאים לזמן הטיסה או לחזרה החופשית.
  const T = cfg.days * 86400;
  const run = (phi, dv) => {
    const traj = propagate(tliState(cfg.alt, phi, dv), 0, T, (t, x, y) => Math.hypot(x, y) < RE + 50e3 && t > 86400);
    const pl = perilune(traj);
    return { traj, pl, alt: pl.d - RM, sense: Math.sign(passSense(pl)) };
  };
  let best = null;
  for (let dv = cfg.dvMin; dv <= cfg.dvMax; dv += cfg.dvStep ?? 5) {
    const step = Math.PI / 180;
    let prev = null;
    for (let phi = -Math.PI; phi < Math.PI; phi += step) {
      const r = run(phi, dv);
      const f = r.sense === cfg.sense ? r.alt - cfg.periAlt : null;
      if (prev && prev.f !== null && f !== null && Math.sign(prev.f) !== Math.sign(f) && r.pl.d < 60000e3) {
        let lo = prev.phi, hi = phi, flo = prev.f, sol = null;
        for (let k = 0; k < 40; k++) {
          const mid = (lo + hi) / 2, rm = run(mid, dv);
          const fm = rm.alt - cfg.periAlt;
          sol = { ...rm, phi: mid, dv };
          if (Math.abs(fm) < 200) break;
          if (Math.sign(fm) === Math.sign(flo)) { lo = mid; flo = fm; } else hi = mid;
        }
        if (sol && sol.sense === cfg.sense && Math.abs(sol.alt - cfg.periAlt) < 2000) {
          let cost = 0;
          if (cfg.tof) cost += Math.abs(sol.pl.p[0] / 3600 - cfg.tof);
          if (cfg.freeReturn) {
            let minR = 1e20;
            for (const p of sol.traj) if (p[0] > sol.pl.p[0]) minR = Math.min(minR, Math.hypot(p[1], p[2]));
            cost += Math.abs(minR - RE - 50e3) / 1000;
            sol.returnPerigee = minR - RE;
          }
          sol.cost = cost;
          if (!best || cost < best.cost) best = sol;
        }
      }
      prev = { phi, f };
    }
  }
  return best;
}

// כניסה למסלול ירחי מעגלי בפריסלנה, כמה הקפות, ואז הבערת TEI חזרה
function lunarOrbitAndReturn(best, orbits, periAlt, tRet) {
  const [t0, x, y, vx, vy] = best.pl.p;
  const [mx, my] = moonAt(t0);
  const vmx = -W * my, vmy = W * mx;
  const rx = x - mx, ry = y - my, rr = Math.hypot(rx, ry);
  const sense = Math.sign(rx * (vy - vmy) - ry * (vx - vmx));
  const vc = Math.sqrt(MU_M / rr);
  const tx = -sense * ry / rr, ty = sense * rx / rr; // כיוון משיקי באותו כיוון סיבוב
  const loiDv = Math.hypot(vx - vmx - vc * tx, vy - vmy - vc * ty);
  const sLO = [x, y, vmx + vc * tx, vmy + vc * ty];
  const Tlo = 2 * Math.PI * Math.sqrt(rr ** 3 / MU_M);
  const lo = propagate(sLO, t0, t0 + orbits * Tlo);
  // TEI: חיפוש תוספת מהירות משיקית שמחזירה לגובה כניסה של 50 ק"מ
  const last = lo[lo.length - 1];
  let bestR = null;
  for (let ang = 0; ang < 2 * Math.PI; ang += Math.PI / 36) {
    // זמן הבערה: סיבוב נוסף בזווית ang
    const dtA = Tlo * ang / (2 * Math.PI);
    const seg = propagate(last.slice(1), last[0], last[0] + dtA + 1);
    const s = seg[seg.length - 1];
    const [mx2, my2] = moonAt(s[0]);
    const vmx2 = -W * my2, vmy2 = W * mx2;
    const vrx = s[3] - vmx2, vry = s[4] - vmy2, vr = Math.hypot(vrx, vry);
    for (let dv = 750; dv <= 1300; dv += 10) {
      const st = [s[1], s[2], s[3] + vrx / vr * dv, s[4] + vry / vr * dv];
      const back = propagate(st, s[0], s[0] + 4 * 86400, (t, X, Y) => Math.hypot(X, Y) < RE + 50e3);
      let minR = 1e20; for (const p of back) minR = Math.min(minR, Math.hypot(p[1], p[2]));
      const cost = Math.abs(minR - RE - 50e3) / 1000 + Math.abs((back[back.length - 1][0] - s[0]) / 3600 - (tRet ?? 60)) * 20;
      if (!bestR || cost < bestR.cost) bestR = { cost, dv, back, pre: seg, tTEI: s[0] };
    }
  }
  return { lo, loiDv, teiDv: bestR.dv, pre: bestR.pre, back: bestR.back, Tlo };
}

const pack = (traj, every = 600) => {
  const out = [];
  let last = -1e9;
  for (const p of traj) if (p[0] - last >= every || p === traj[traj.length - 1]) { out.push([Math.round(p[0]), Math.round(p[1] / 1000), Math.round(p[2] / 1000)]); last = p[0]; }
  return out;
};
// דגימה צפופה ליד הירח
const packAdaptive = (traj) => {
  const out = []; let last = -1e9;
  for (const p of traj) {
    const [mx, my] = moonAt(p[0]);
    const near = Math.hypot(p[1] - mx, p[2] - my) < 40000e3 || Math.hypot(p[1], p[2]) < 40000e3;
    if (p[0] - last >= (near ? 60 : 900) || p === traj[traj.length - 1]) { out.push([Math.round(p[0]), Math.round(p[1] / 1000), Math.round(p[2] / 1000)]); last = p[0]; }
  }
  return out;
};

const result = {};

// אפולו 11: מסלול חניה 185 ק"מ, TLI, פריסלנה 110 ק"מ מאחורי הירח, הכנסה למסלול, 30 הקפות (מוצגות 4), חזרה
{
  const b = solve({ alt: 185e3, periAlt: 111e3, days: 5, dvMin: 3130, dvMax: 3200, sense: -1, tof: 73 });
  const trans = b.traj.filter(p => p[0] <= b.pl.p[0]);
  const ret = lunarOrbitAndReturn(b, 4, 111e3, 60);
  console.log('Apollo11 TLI dv', b.dv.toFixed(1), 'phi', b.phi.toFixed(3), 'perilune alt km', ((b.pl.d - RM) / 1000).toFixed(1), 'TOF h', (b.pl.p[0] / 3600).toFixed(1), 'LOI', ret.loiDv.toFixed(0), 'TEI', ret.teiDv, 'return h', ((ret.back[ret.back.length - 1][0] - ret.back[0][0]) / 3600).toFixed(1));
  result.apollo11 = {
    tliDv: b.dv, tof: b.pl.p[0], loiDv: ret.loiDv, teiDv: ret.teiDv,
    out: packAdaptive(trans), lunar: packAdaptive([...ret.lo, ...ret.pre]), back: packAdaptive(ret.back),
  };
}
// ארטמיס 2: חזרה חופשית סביב הירח, מעבר בגובה 6,545 ק"מ מעל הצד הרחוק
{
  const b = solve({ alt: 185e3, periAlt: 6545e3, days: 10, dvMin: 3100, dvMax: 3180, dvStep: 2, sense: -1, freeReturn: true });
  console.log('ArtemisII TLI dv', b.dv.toFixed(1), 'perilune alt km', ((b.pl.d - RM) / 1000).toFixed(1), 'TOF h', (b.pl.p[0] / 3600).toFixed(1), 'total h', (b.traj[b.traj.length - 1][0] / 3600).toFixed(1), 'cost', b.cost.toFixed(1), 'retPerigee km', (b.returnPerigee/1000).toFixed(0));
  result.artemis2 = { tliDv: b.dv, tof: b.pl.p[0], total: b.traj[b.traj.length - 1][0], out: packAdaptive(b.traj) };
}
// צ'אנג'-אה (כללי): העברה ישירה לפריסלנה 100 ק"מ
{
  const b = solve({ alt: 200e3, periAlt: 100e3, days: 7, dvMin: 3050, dvMax: 3140, sense: -1, tof: 112 });
  const trans = b.traj.filter(p => p[0] <= b.pl.p[0]);
  const ret = lunarOrbitAndReturn(b, 3, 100e3, 110);
  console.log('Change TLI dv', b.dv.toFixed(1), 'TOF h', (b.pl.p[0] / 3600).toFixed(1), 'LOI', ret.loiDv.toFixed(0));
  result.change = { tliDv: b.dv, tof: b.pl.p[0], loiDv: ret.loiDv, teiDv: ret.teiDv, out: packAdaptive(trans), lunar: packAdaptive([...ret.lo, ...ret.pre]), back: packAdaptive(ret.back) };
}

result.W = W;
writeFileSync(new URL('../js/lunar_data.js', import.meta.url),
  '// נוצר אוטומטית ע"י tools/build_lunar.mjs — אין לערוך ידנית.\n// מסלולים במישור מסלול הירח, מערכת אינרציאלית שמרכזה בכדור הארץ. כל נקודה: [t s, x km, y km]; הירח ב-t=0 על ציר x.\nexport const LUNAR = ' + JSON.stringify(result) + ';\n');
console.log('written');
