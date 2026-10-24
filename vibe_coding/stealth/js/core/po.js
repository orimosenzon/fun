// מנוע RCS: אופטיקה פיזיקלית (PO) על משולשים בנוסחת גורדון, ועוד מקדם קצה מפושט.
// בלי three, רץ גם ב-node. יחידות SI. מערכת צירים: x קדימה (לאף), y שמאלה, z למעלה.
// r̂ הוא וקטור יחידה מהמטוס אל המכ"ם. אזימוט 0 = מול האף.

export const C0 = 299792458;

export function dirFromAzEl(azDeg, elDeg) {
  const a = azDeg * Math.PI / 180, e = elDeg * Math.PI / 180;
  return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
}

// הפרש מחולק ראשון של e^{jx}: (e^{jx}-e^{jy})/(x-y) = j e^{jm} sinc(d/2), יציב גם כש-x≈y
function d1(x, y, out) {
  const m = 0.5 * (x + y), h = 0.5 * (x - y);
  const s = Math.abs(h) < 1e-6 ? 1 - h * h / 6 : Math.sin(h) / h;
  // j·e^{jm}·s = (-sin m + j cos m)·s
  out[0] = -Math.sin(m) * s; out[1] = Math.cos(m) * s;
}
const _p = [0, 0], _q = [0, 0];

// ∫_T e^{j(w·r)} dS לפי הרמיט-ג'נוקי: 2A·(-f[α,β,γ]) כאשר f(x)=e^{jx}
// α,β,γ הם הפאזות בקודקודים. מחזיר [re, im] ב-out.
export function triIntegral(al, be, ga, area, out) {
  // בוחרים את הזוג עם ההפרש הגדול ביותר כקצוות, כדי שהחילוק יהיה יציב
  let x = al, y = be, z = ga;
  const dab = Math.abs(al - be), dbg = Math.abs(be - ga), dag = Math.abs(al - ga);
  if (dab >= dbg && dab >= dag) { x = al; z = be; y = ga; }
  else if (dbg >= dab && dbg >= dag) { x = be; z = ga; y = al; }
  const big = Math.abs(x - z);
  if (big < 1e-4) {
    const mu = (al + be + ga) / 3;
    out[0] = area * Math.cos(mu); out[1] = area * Math.sin(mu);
    return out;
  }
  d1(x, y, _p); d1(y, z, _q);
  const inv = 1 / (x - z);
  // f[x,y,z] = (d1(x,y) - d1(y,z)) / (x - z); האינטגרל = -2A·f
  out[0] = -2 * area * (_p[0] - _q[0]) * inv;
  out[1] = -2 * area * (_p[1] - _q[1]) * inv;
  return out;
}

// הכנת רשת: מערכים שטוחים לכל משולש + קצוות חדים (feature edges)
// mesh: { pos:[x,y,z,...], tri:[i,j,k,...], part:[...per tri] }
export function prepare(mesh, featureDeg = 35) {
  const P = mesh.pos, T = mesh.tri, nt = T.length / 3;
  const v = new Float64Array(nt * 9), n = new Float64Array(nt * 3), A = new Float64Array(nt);
  const part = new Uint8Array(nt);
  for (let t = 0; t < nt; t++) {
    const i = T[3 * t] * 3, j = T[3 * t + 1] * 3, k = T[3 * t + 2] * 3;
    for (let c = 0; c < 3; c++) { v[9 * t + c] = P[i + c]; v[9 * t + 3 + c] = P[j + c]; v[9 * t + 6 + c] = P[k + c]; }
    const ux = P[j] - P[i], uy = P[j + 1] - P[i + 1], uz = P[j + 2] - P[i + 2];
    const wx = P[k] - P[i], wy = P[k + 1] - P[i + 1], wz = P[k + 2] - P[i + 2];
    let cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
    const L = Math.hypot(cx, cy, cz) || 1;
    A[t] = 0.5 * L; n[3 * t] = cx / L; n[3 * t + 1] = cy / L; n[3 * t + 2] = cz / L;
    part[t] = mesh.part ? mesh.part[t] : 0;
  }
  // ריתוך קודקודים לפי מיקום, כדי לזהות שכנות גם בין חלקים
  const key = (i) => `${Math.round(P[3 * i] * 1e4)},${Math.round(P[3 * i + 1] * 1e4)},${Math.round(P[3 * i + 2] * 1e4)}`;
  const weld = new Map(), id = new Int32Array(P.length / 3);
  for (let i = 0; i < id.length; i++) { const kk = key(i); if (!weld.has(kk)) weld.set(kk, weld.size); id[i] = weld.get(kk); }
  const em = new Map();
  for (let t = 0; t < nt; t++) for (let e = 0; e < 3; e++) {
    const a = T[3 * t + e], b = T[3 * t + (e + 1) % 3];
    const ia = id[a], ib = id[b]; if (ia === ib) continue;
    const kk = ia < ib ? ia + '_' + ib : ib + '_' + ia;
    let r = em.get(kk); if (!r) { r = { a, b, t: [] }; em.set(kk, r); }
    r.t.push(t);
  }
  const cosT = Math.cos(featureDeg * Math.PI / 180);
  const edges = [];
  for (const r of em.values()) {
    if (r.t.length !== 2) continue;
    const t1 = r.t[0], t2 = r.t[1];
    const dot = n[3 * t1] * n[3 * t2] + n[3 * t1 + 1] * n[3 * t2 + 1] + n[3 * t1 + 2] * n[3 * t2 + 2];
    if (dot > cosT) continue;
    const ax = P[3 * r.a], ay = P[3 * r.a + 1], az = P[3 * r.a + 2];
    const bx = P[3 * r.b], by = P[3 * r.b + 1], bz = P[3 * r.b + 2];
    const L = Math.hypot(bx - ax, by - ay, bz - az);
    if (L < 1e-6) continue;
    const turn = Math.acos(Math.max(-1, Math.min(1, dot))); // זווית הפנייה בין הפאות
    edges.push({
      c: [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2], t: [(bx - ax) / L, (by - ay) / L, (bz - az) / L], L,
      n1: [n[3 * t1], n[3 * t1 + 1], n[3 * t1 + 2]], n2: [n[3 * t2], n[3 * t2 + 1], n[3 * t2 + 2]],
      w: Math.min(1, turn / (Math.PI * 0.6)), part: part[t1], a: [ax, ay, az], b: [bx, by, bz],
    });
  }
  return { nt, v, n, A, part, edges };
}

// gamma: פונקציה part → מקדם החזרה (גודל) של הציפוי. extras: פיזורים נקודתיים (פתחי אוויר, תא טייס)
// מחזיר משרעת מרוכבת √σ (במטרים), כך ש-σ=|amp|²
export function amplitude(P, r, k, gamma, extras, opt = {}) {
  const rx = r[0], ry = r[1], rz = r[2], w = 2 * k;
  const lam = 2 * Math.PI / k, scale = 2 * Math.sqrt(Math.PI) / lam;
  let re = 0, im = 0;
  const out = [0, 0];
  const v = P.v, n = P.n, A = P.A, part = P.part;
  if (!opt.noPO) {
    for (let t = 0; t < P.nt; t++) {
      const c = n[3 * t] * rx + n[3 * t + 1] * ry + n[3 * t + 2] * rz;
      if (c <= 0) continue; // רק פאות מוארות
      const o = 9 * t;
      const al = w * (v[o] * rx + v[o + 1] * ry + v[o + 2] * rz);
      const be = w * (v[o + 3] * rx + v[o + 4] * ry + v[o + 5] * rz);
      const ga = w * (v[o + 6] * rx + v[o + 7] * ry + v[o + 8] * rz);
      triIntegral(al, be, ga, A[t], out);
      const g = gamma ? gamma[part[t]] : 1;
      re += c * g * out[0]; im += c * g * out[1];
    }
    re *= scale; im *= scale;
  }
  if (!opt.noEdges) {
    const inv = 1 / Math.sqrt(Math.PI);
    for (const e of P.edges) {
      const s1 = e.n1[0] * rx + e.n1[1] * ry + e.n1[2] * rz, s2 = e.n2[0] * rx + e.n2[1] * ry + e.n2[2] * rz;
      if (s1 < -0.02 && s2 < -0.02) continue; // הקצה מוסתר
      const ct = e.t[0] * rx + e.t[1] * ry + e.t[2] * rz;
      const x = k * e.L * ct;
      const sinc = Math.abs(x) < 1e-6 ? 1 : Math.sin(x) / x;
      const g = (gamma ? gamma[e.part] : 1) * (opt.edgeGamma ?? 1);
      const amp = e.w * e.L * inv * sinc * Math.sqrt(Math.max(0, 1 - ct * ct)) * g;
      const ph = w * (e.c[0] * rx + e.c[1] * ry + e.c[2] * rz);
      re += amp * Math.cos(ph); im += amp * Math.sin(ph);
    }
  }
  if (extras) for (const x of extras) {
    const a = x.amp(r, k);
    if (a <= 0) continue;
    const ph = w * (x.p[0] * rx + x.p[1] * ry + x.p[2] * rz) + (x.phase || 0);
    re += a * Math.cos(ph); im += a * Math.sin(ph);
  }
  return [re, im];
}

export function sigma(P, r, k, gamma, extras, opt) {
  const a = amplitude(P, r, k, gamma, extras, opt);
  return a[0] * a[0] + a[1] * a[1];
}

// סכום לא קוהרנטי: כל פאה בנפרד, בלי התאבכות. נותן "בועה" חלקה שמראה לאן הולכת האנרגיה
export function sigmaIncoherent(P, r, k, gamma) {
  const rx = r[0], ry = r[1], rz = r[2], w = 2 * k, lam = 2 * Math.PI / k, s2 = 4 * Math.PI / (lam * lam);
  let sum = 0; const out = [0, 0];
  for (let t = 0; t < P.nt; t++) {
    const c = P.n[3 * t] * rx + P.n[3 * t + 1] * ry + P.n[3 * t + 2] * rz;
    if (c <= 0) continue;
    const o = 9 * t, v = P.v;
    triIntegral(w * (v[o] * rx + v[o + 1] * ry + v[o + 2] * rz), w * (v[o + 3] * rx + v[o + 4] * ry + v[o + 5] * rz),
      w * (v[o + 6] * rx + v[o + 7] * ry + v[o + 8] * rz), P.A[t], out);
    const g = gamma ? gamma[P.part[t]] : 1;
    sum += s2 * c * c * g * g * (out[0] * out[0] + out[1] * out[1]);
  }
  for (const e of P.edges) {
    const s1 = e.n1[0] * rx + e.n1[1] * ry + e.n1[2] * rz, s2b = e.n2[0] * rx + e.n2[1] * ry + e.n2[2] * rz;
    if (s1 < -0.02 && s2b < -0.02) continue;
    const ct = e.t[0] * rx + e.t[1] * ry + e.t[2] * rz, x = k * e.L * ct;
    const sinc = Math.abs(x) < 1e-6 ? 1 : Math.sin(x) / x;
    const g = gamma ? gamma[e.part] : 1;
    sum += (e.w * e.L * sinc * g) ** 2 * (1 - ct * ct) / Math.PI;
  }
  return sum;
}

// תבנית קוטבית: σ כתלות באזימוט בגובה נתון
export function polar(P, k, elDeg, n, gamma, extras, opt) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = sigma(P, dirFromAzEl(i * 360 / n, elDeg), k, gamma, extras, opt);
  return out;
}

// ממוצע בחלון זוויתי (בערכים לינאריים), כמו שמודדים בפועל: הפסגות הצרות מתפזרות
export function smooth(arr, winDeg) {
  const n = arr.length, h = Math.max(0, Math.round(winDeg / 360 * n / 2));
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = -h; j <= h; j++) s += arr[(i + j + n) % n];
    out[i] = s / (2 * h + 1);
  }
  return out;
}

export const dB = (s) => 10 * Math.log10(Math.max(s, 1e-12));

// ממוצע σ בגזרה סביב אזימוט מרכזי
export function sectorMean(arr, centerDeg, halfDeg) {
  const n = arr.length; let s = 0, c = 0;
  for (let i = 0; i < n; i++) {
    let a = i * 360 / n - centerDeg; a = ((a + 540) % 360) - 180;
    if (Math.abs(a) <= halfDeg) { s += arr[i]; c++; }
  }
  return c ? s / c : 0;
}
