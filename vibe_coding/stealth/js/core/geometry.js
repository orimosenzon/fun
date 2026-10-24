// מחולל מטוס פרמטרי: גוף (עגול או עם צלע חדה), כנף, זנבות ומייצב. אותה רשת משמשת לציור ולחישוב ה-RCS.
// x קדימה (האף ב-L/2), y שמאלה, z למעלה.

export const PART = { FUS: 0, WING: 1, TAIL: 2, HSTAB: 3, NOZZLE: 4 };
export const PART_NAMES = ['גוף', 'כנף', 'זנבות', 'מייצב גובה', 'פתח פליטה'];

const D2R = Math.PI / 180;

export const PRESETS = {
  conventional: {
    name: 'קרב קונבנציונלי (בהשראת F-15)', L: 19.4,
    fus: { section: 'round', W: 1.45, HU: 1.15, HL: 0.95, side: 0, noseFrac: 0.3, tailFrac: 0.16, nozzle: 0.55 },
    wing: { span: 13, cr: 6.6, ct: 1.9, sweep: 45, x0: 0.45, t: 0.05, te: 'straight', teeth: 4, dihedral: -2 },
    tail: { type: 'twin', cant: 0, h: 3.0, cr: 3.0, ct: 1.2, sweep: 40, xFrac: 0.76, yFrac: 0.85 },
    hstab: { on: true, span: 8.6, cr: 3.0, ct: 1.0, sweep: 50, xFrac: 0.8 },
    align: false, coat: 'none', inlet: 'pitot', canopy: 'clear', nozzleType: 'round', antenna: 'dish',
  },
  faceted: {
    name: 'רב-פאות (בהשראת F-117)', L: 20.1,
    fus: { section: 'chine', W: 1.6, HU: 1.35, HL: 0.6, side: 42, noseFrac: 0.42, tailFrac: 0.2, nozzle: 0.75 },
    wing: { span: 13.2, cr: 12.5, ct: 1.2, sweep: 67.5, x0: 0.18, t: 0.06, te: 'straight', teeth: 3, dihedral: 0 },
    tail: { type: 'twin', cant: 50, h: 2.7, cr: 3.2, ct: 1.5, sweep: 52, xFrac: 0.76, yFrac: 0.6 },
    hstab: { on: false, span: 7, cr: 2.5, ct: 1, sweep: 45, xFrac: 0.8 },
    align: false, coat: 'paint', inlet: 'grille', canopy: 'coated', nozzleType: 'slot', antenna: 'tilted',
  },
  wing: {
    name: 'כנף מעופפת (בהשראת B-2)', L: 21,
    fus: { section: 'chine', W: 2.6, HU: 1.6, HL: 0.7, side: 55, noseFrac: 0.45, tailFrac: 0.35, nozzle: 0.6 },
    wing: { span: 52.4, cr: 20, ct: 2, sweep: 33, x0: 0.0, t: 0.07, te: 'saw', teeth: 3, dihedral: 0 },
    tail: { type: 'none', cant: 0, h: 2, cr: 2, ct: 1, sweep: 33, xFrac: 0.8, yFrac: 0.6 },
    hstab: { on: false, span: 7, cr: 2.5, ct: 1, sweep: 33, xFrac: 0.8 },
    align: true, coat: 'thick', inlet: 'sduct_ram', canopy: 'coated', nozzleType: 'slot', antenna: 'tilted',
  },
  fighter: {
    name: 'קרב חמקן (בהשראת F-22)', L: 18.9,
    fus: { section: 'chine', W: 1.7, HU: 1.15, HL: 0.85, side: 28, noseFrac: 0.33, tailFrac: 0.14, nozzle: 0.65 },
    wing: { span: 13.6, cr: 9.2, ct: 1.6, sweep: 42, x0: 0.36, t: 0.05, te: 'straight', teeth: 4, dihedral: -3 },
    tail: { type: 'twin', cant: 28, h: 2.8, cr: 3.3, ct: 1.3, sweep: 42, xFrac: 0.72, yFrac: 0.9 },
    hstab: { on: true, span: 8.8, cr: 3.1, ct: 1.1, sweep: 42, xFrac: 0.84 },
    align: true, coat: 'paint', inlet: 'sduct_ram', canopy: 'coated', nozzleType: 'slot', antenna: 'tilted',
  },
};

export function clonePreset(k) { return JSON.parse(JSON.stringify(PRESETS[k])); }

// יישור קצוות: כל קצה מוביל באותה זווית כמו הכנף
export function applyAlign(p) {
  if (!p.align) return p;
  p.tail.sweep = p.wing.sweep;
  p.hstab.sweep = p.wing.sweep;
  return p;
}

// פרופיל הגוף לאורך s∈[0,1] (0 = אף)
function fusProfile(p, s) {
  const f = p.fus;
  let k = 1;
  if (s < f.noseFrac) {
    const u = s / f.noseFrac;
    k = f.section === 'round' ? Math.sqrt(Math.max(0, 1 - (1 - u) * (1 - u))) : Math.pow(u, 0.85);
  } else if (s > 1 - f.tailFrac) {
    const u = (s - (1 - f.tailFrac)) / f.tailFrac;
    k = 1 - (1 - f.nozzle) * u;
  }
  return { w: f.W * k, hu: f.HU * k, hl: f.HL * k };
}

function section(p, s) {
  const { w, hu, hl } = fusProfile(p, s), f = p.fus;
  if (f.section === 'round') {
    const M = 24, pts = [];
    for (let i = 0; i < M; i++) {
      const th = (i / M) * 2 * Math.PI;
      const sn = Math.sin(th);
      pts.push([w * Math.cos(th), (sn >= 0 ? hu : hl) * sn]);
    }
    return pts;
  }
  const ta = Math.tan(f.side * D2R);
  const wt = Math.max(0.08 * w, w - hu * ta), wb = Math.max(0.08 * w, w - hl * ta);
  return [[wt, hu], [w, 0], [wb, -hl], [-wb, -hl], [-w, 0], [-wt, hu]];
}

// גובה החלק העליון של הגוף ב-y נתון (לשתילת הזנבות)
function topZ(p, s, y) {
  const { w, hu } = fusProfile(p, s), f = p.fus;
  const ay = Math.min(Math.abs(y), w);
  if (f.section === 'round') return hu * Math.sqrt(Math.max(0, 1 - (ay / w) ** 2));
  const wt = Math.max(0.08 * w, w - hu * Math.tan(f.side * D2R));
  return ay <= wt ? hu : hu * (w - ay) / (w - wt);
}

export function buildAircraft(p0) {
  const p = applyAlign(JSON.parse(JSON.stringify(p0)));
  const L = p.L, pos = [], tri = [], part = [];
  const plates = []; // לחישוב פילוג השטח
  const addV = (x, y, z) => { pos.push(x, y, z); return pos.length / 3 - 1; };
  const addT = (a, b, c, pt, hint) => {
    const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
    const ux = pos[3 * b] - ax, uy = pos[3 * b + 1] - ay, uz = pos[3 * b + 2] - az;
    const wx = pos[3 * c] - ax, wy = pos[3 * c + 1] - ay, wz = pos[3 * c + 2] - az;
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    if (nx * nx + ny * ny + nz * nz < 1e-14) return;
    const h = typeof hint === 'function' ? hint((ax + pos[3 * b] + pos[3 * c]) / 3, (ay + pos[3 * b + 1] + pos[3 * c + 1]) / 3, (az + pos[3 * b + 2] + pos[3 * c + 2]) / 3) : hint;
    if (nx * h[0] + ny * h[1] + nz * h[2] < 0) tri.push(a, c, b); else tri.push(a, b, c);
    part.push(pt);
  };
  const xOf = (s) => L / 2 - s * L;

  // ---- גוף ----
  const NS = 30, st = [];
  for (let i = 0; i <= NS; i++) { const u = i / NS; st.push(u < 0.5 ? 0.5 * Math.pow(2 * u, 1.6) : u); }
  st[0] = 0;
  const rings = [];
  const tip = addV(L / 2, 0, 0);
  for (let i = 1; i < st.length; i++) {
    const s = st[i], x = xOf(s);
    rings.push({ s, x, idx: section(p, s).map(([y, z]) => addV(x, y, z)) });
  }
  const fusHint = (x, y, z) => [0, y, z];
  const M = rings[0].idx.length;
  for (let m = 0; m < M; m++) addT(tip, rings[0].idx[m], rings[0].idx[(m + 1) % M], PART.FUS, fusHint);
  for (let r = 0; r + 1 < rings.length; r++) {
    const A = rings[r].idx, B = rings[r + 1].idx;
    for (let m = 0; m < M; m++) {
      const m2 = (m + 1) % M;
      addT(A[m], B[m], B[m2], PART.FUS, fusHint);
      addT(A[m], B[m2], A[m2], PART.FUS, fusHint);
    }
  }
  const last = rings[rings.length - 1];
  const cz = last.idx.reduce((s, i) => s + pos[3 * i + 2], 0) / M;
  const cap = addV(last.x, 0, cz);
  for (let m = 0; m < M; m++) addT(cap, last.idx[m], last.idx[(m + 1) % M], PART.NOZZLE, [-1, 0, 0]);

  // ---- משטח עילוי כללי ----
  // R: שורש הקצה המוביל, S: כיוון המוטה, N: כיוון העובי, span: אורך, sweep במעלות
  function plate(o) {
    const { R, S, N, span, cr, ct, sweep, t, pt, rootFn, teFn } = o;
    const tl = Math.tan(sweep * D2R);
    // תחנות לאורך המוטה, כולל נקודות שבירה של שפת שיניים
    const nv = o.nv || 7, etas = [];
    const kinks = [0, ...(o.kinks || []).filter((e) => e > 0.05 && e < span - 0.05), span];
    for (let k = 0; k + 1 < kinks.length; k++) {
      const sub = o.kinks && o.kinks.length ? Math.max(2, Math.ceil(nv * (kinks[k + 1] - kinks[k]) / span)) : nv;
      for (let q = 0; q < sub; q++) etas.push(kinks[k] + (kinks[k + 1] - kinks[k]) * q / sub);
    }
    etas.push(span);
    const xLE = (eta) => -eta * tl;
    const xTE = teFn ? (eta) => Math.min(teFn(eta), xLE(eta) - 0.25) : (eta) => xLE(eta) - (cr + (ct - cr) * eta / span);
    const NU = o.nu || 9, us = [];
    for (let i = 0; i <= NU; i++) us.push(0.5 - 0.5 * Math.cos(Math.PI * i / NU));
    const grid = []; // grid[j][i] = [top, bottom]
    for (let j = 0; j < etas.length; j++) {
      const eta = etas[j], le = xLE(eta), te = xTE(eta), c = le - te, row = [];
      for (let i = 0; i < us.length; i++) {
        const u = us[i];
        let bx = R[0] + le - u * c, by = R[1] + eta * S[1], bz = R[2] + eta * S[2];
        if (j === 0 && rootFn) { const q = rootFn(bx); by = q[1]; bz = q[2]; }
        const th = t * c * 2 * u * (1 - u);
        if (i === 0 || i === us.length - 1) { const v = addV(bx, by, bz); row.push([v, v]); }
        else row.push([addV(bx + th * N[0], by + th * N[1], bz + th * N[2]), addV(bx - th * N[0], by - th * N[1], bz - th * N[2])]);
      }
      grid.push(row);
    }
    const nN = [-N[0], -N[1], -N[2]];
    for (let j = 0; j + 1 < grid.length; j++) for (let i = 0; i + 1 < us.length; i++) {
      for (let sd = 0; sd < 2; sd++) {
        const a = grid[j][i][sd], b = grid[j][i + 1][sd], c = grid[j + 1][i + 1][sd], d = grid[j + 1][i][sd];
        const h = sd === 0 ? N : nN;
        addT(a, b, c, pt, h); addT(a, c, d, pt, h);
      }
    }
    const tipRow = grid[grid.length - 1];
    for (let i = 0; i + 1 < us.length; i++) {
      addT(tipRow[i][0], tipRow[i + 1][0], tipRow[i + 1][1], pt, S);
      addT(tipRow[i][0], tipRow[i + 1][1], tipRow[i][1], pt, S);
    }
    plates.push({ R, S, N, span, xLE, xTE, t, etas, rootFn });
  }

  // ---- כנפיים ----
  const w = p.wing, xA = xOf(w.x0), semi = w.span / 2, Wf = p.fus.W;
  const dh = (w.dihedral || 0) * D2R;
  const tlw = Math.tan(w.sweep * D2R);
  const r0 = Math.min(Wf, semi * 0.5);
  const teG = (y) => teGlobal(w, y); // יחסית ל-xA
  for (const sg of [1, -1]) {
    // הכנף מתחילה בדופן הגוף; המוטה נמדד מקו האמצע
    const rootFn = (x) => { const s = (L / 2 - x) / L; return [x, sg * fusProfile(p, Math.min(1, Math.max(0, s))).w, 0]; };
    const R = [xA - r0 * tlw, sg * r0, 0];
    const o = { R, S: [0, sg * Math.cos(dh), Math.sin(dh)], N: [0, -sg * Math.sin(dh), Math.cos(dh)], span: semi - r0, cr: w.cr + (w.ct - w.cr) * r0 / semi, ct: w.ct, sweep: w.sweep, t: w.t, pt: PART.WING, rootFn, nv: 7 };
    if (w.te === 'saw') {
      o.teFn = (eta) => teG(r0 + eta) + r0 * tlw;
      o.kinks = []; for (let i = 1; i < w.teeth; i++) o.kinks.push(semi * i / w.teeth - r0);
    }
    plate(o);
  }

  // ---- זנבות ----
  const tl = p.tail;
  if (tl.type !== 'none') {
    const xr = xOf(tl.xFrac), sMid = Math.min(1, tl.xFrac + tl.cr / L / 2);
    const fins = tl.type === 'single' ? [0] : [1, -1];
    for (const sg of fins) {
      const y = sg * tl.yFrac * fusProfile(p, sMid).w;
      const z = topZ(p, sMid, y);
      const g = (tl.type === 'single' ? 0 : tl.cant) * D2R;
      const S = sg === 0 ? [0, 0, 1] : [0, sg * Math.sin(g), Math.cos(g)];
      const N = sg === 0 ? [0, 1, 0] : [0, sg * Math.cos(g), -Math.sin(g)];
      plate({ R: [xr, y, z - 0.05], S, N, span: tl.h, cr: tl.cr, ct: tl.ct, sweep: tl.sweep, t: 0.045, pt: PART.TAIL, nv: 5, nu: 7 });
    }
  }
  // ---- מייצב גובה ----
  const hs = p.hstab;
  if (hs.on) {
    const xr = xOf(hs.xFrac), sMid = Math.min(1, hs.xFrac + hs.cr / L / 2);
    const yr = fusProfile(p, sMid).w;
    for (const sg of [1, -1]) {
      plate({ R: [xr, sg * yr, 0], S: [0, sg, 0], N: [0, 0, 1], span: Math.max(0.5, hs.span / 2 - yr), cr: hs.cr, ct: hs.ct, sweep: hs.sweep, t: 0.04, pt: PART.HSTAB, nv: 5, nu: 7 });
    }
  }

  return { p, pos, tri, part, plates, rings, area: areaInfo(p, plates) };
}

// שפת הנגרר בשיני מסור, כפונקציה של המרחק מקו האמצע (x יחסית לקודקוד הקצה המוביל)
// קטעים מקבילים לקצה המוביל לסירוגין, כמו ב-B-2
export function teGlobal(w, y) {
  const semi = w.span / 2, seg = semi / w.teeth, tl = Math.tan(w.sweep * D2R);
  let x = -w.cr, e = 0, sgn = 1;
  for (let i = 0; i < w.teeth; i++) { const d = Math.min(seg, y - e); if (d <= 0) break; x += sgn * tl * d; e += d; sgn = -sgn; }
  return x;
}

// פילוג שטח החתך A(x), נפח, שטח כנף ייחוס
function areaInfo(p, plates) {
  const L = p.L, N = 120, xs = [], A = [];
  for (let i = 0; i <= N; i++) {
    const s = i / N, x = L / 2 - s * L;
    const sec = section(p, s);
    let a = 0;
    for (let m = 0; m < sec.length; m++) { const q = sec[m], r = sec[(m + 1) % sec.length]; a += q[0] * r[1] - r[0] * q[1]; }
    a = Math.abs(a) / 2;
    for (const pl of plates) {
      const n = 60, de = pl.span / n;
      for (let k = 0; k < n; k++) {
        const eta = (k + 0.5) * de, le = pl.R[0] + pl.xLE(eta), te = pl.R[0] + pl.xTE(eta), c = le - te;
        if (x > le || x < te || c <= 0) continue;
        const u = (le - x) / c;
        a += 2 * pl.t * c * 2 * u * (1 - u) * de;
      }
    }
    xs.push(x); A.push(a);
  }
  let V = 0; const dx = L / N;
  for (let i = 0; i < N; i++) V += 0.5 * (A[i] + A[i + 1]) * dx;
  // שטח כנף ייחוס: הטרפז הממשיך עד קו האמצע
  const w = p.wing, semi = w.span / 2, tlw = Math.tan(w.sweep * D2R);
  let Sref = 0; const n = 200;
  for (let k = 0; k < n; k++) {
    const y = (k + 0.5) * semi / n;
    let c;
    if (w.te === 'saw') c = Math.max(0.25, -y * tlw - teGlobal(w, y));
    else c = w.cr + (w.ct - w.cr) * y / semi;
    Sref += 2 * c * semi / n;
  }
  return { xs, A, V, Sref, Amax: Math.max(...A) };
}
