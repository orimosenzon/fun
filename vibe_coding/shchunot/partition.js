// חלוקת רשת תאים ל-k שכונות רציפות, עם איזון בין אוכלוסייה לשטח.
// עובד גם בדפדפן (window.Partition) וגם ב-Node (module.exports).
//
// שלבים:
//  1. זרעים בשיטת k-means++ משוקללת (מחולל אקראי עם seed, כדי שהתוצאה תהיה ניתנת לשחזור).
//  2. דיאגרמת חזקה מאוזנת: כל תא הולך למרכז שממזער d² - b_j, וה-b_j מתכווננים
//     עד שלכל שכונה "משקל" דומה (ערבוב של אוכלוסייה ושטח לפי alpha). זה נותן אזורים קומפקטיים.
//  3. תיקון רציפות: חלקים מנותקים מוצמדים לשכנה.
//  4. חיפוש מקומי: העברת תאי גבול בין שכונות כשזה משפר את פונקציית המטרה,
//     בלי לנתק אף שכונה.
(function (root) {
  'use strict';

  function rng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  // מכין מבנה רשת מהנתונים הגולמיים
  // opts.inhabitedOnly: שטח של אזורים סטטיסטיים בלי תושבים (תעשייה, שטח פתוח) לא נספר בשטח
  function prepare(D, opts) {
    opts = opts || {};
    const n = D.cells.length;
    const X = new Float64Array(n), Y = new Float64Array(n);
    const A = new Float64Array(n), Ar = new Float64Array(n), P = new Float64Array(n);
    const idx = new Map();
    const cs = D.source.cell;
    const empty = new Set(D.stats.filter((s) => !s.pop).map((s) => s.id));
    D.cells.forEach((c, k) => {
      X[k] = (c[0] + 0.5) * cs; Y[k] = (c[1] + 0.5) * cs;
      Ar[k] = c[2]; A[k] = opts.inhabitedOnly && empty.has(c[4]) ? 0 : c[2]; P[k] = c[3];
      idx.set(c[0] + ',' + c[1], k);
    });
    // שכנים: 4 צדדים, ו-8 לבדיקת טבעת
    const N4 = [], N8 = [];
    const d4 = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const d8 = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
    D.cells.forEach((c) => {
      N4.push(d4.map(([dx, dy]) => idx.has((c[0] + dx) + ',' + (c[1] + dy)) ? idx.get((c[0] + dx) + ',' + (c[1] + dy)) : -1));
      N8.push(d8.map(([dx, dy]) => idx.has((c[0] + dx) + ',' + (c[1] + dy)) ? idx.get((c[0] + dx) + ',' + (c[1] + dy)) : -1));
    });
    let Pt = 0, At = 0;
    for (let k = 0; k < n; k++) { Pt += P[k]; At += A[k]; }
    return { n, X, Y, A, Ar, P, N4, N8, Pt, At, idx };
  }

  function objective(G, reg, k, opt) {
    const { pops, areas } = loads(G, reg, k);
    const Pt = G.Pt / k, At = G.At / k;
    let bal = 0;
    for (let j = 0; j < k; j++) {
      bal += opt.alpha * (pops[j] / Pt - 1) ** 2 + (1 - opt.alpha) * (areas[j] / At - 1) ** 2;
    }
    let cut = 0;
    for (let c = 0; c < G.n; c++) for (const m of G.N4[c]) if (m > c && reg[m] !== reg[c]) cut++;
    return bal + opt.mu * cut / CUT_SCALE;
  }
  const CUT_SCALE = 1500;

  function loads(G, reg, k) {
    const pops = new Float64Array(k), areas = new Float64Array(k);
    for (let c = 0; c < G.n; c++) { pops[reg[c]] += G.P[c]; areas[reg[c]] += G.A[c]; }
    return { pops, areas };
  }

  function seeds(G, k, W, rand) {
    const C = [];
    let tot = 0; for (let c = 0; c < G.n; c++) tot += W[c];
    let r = rand() * tot, first = 0;
    for (let c = 0; c < G.n; c++) { r -= W[c]; if (r <= 0) { first = c; break; } }
    C.push([G.X[first], G.Y[first]]);
    const d2 = new Float64Array(G.n).fill(Infinity);
    while (C.length < k) {
      const [cx, cy] = C[C.length - 1];
      let s = 0;
      for (let c = 0; c < G.n; c++) {
        const d = (G.X[c] - cx) ** 2 + (G.Y[c] - cy) ** 2;
        if (d < d2[c]) d2[c] = d;
        s += d2[c] * (W[c] + 1e-9);
      }
      let t = rand() * s, pick = 0;
      for (let c = 0; c < G.n; c++) { t -= d2[c] * (W[c] + 1e-9); if (t <= 0) { pick = c; break; } }
      C.push([G.X[pick], G.Y[pick]]);
    }
    return C;
  }

  function powerBalance(G, k, W, C) {
    const reg = new Int32Array(G.n);
    const b = new Float64Array(k);
    const S = G.At / k; // סקאלת מרחק בריבוע
    const target = 1 / k;
    for (let it = 0; it < 120; it++) {
      for (let c = 0; c < G.n; c++) {
        let best = 0, bv = Infinity;
        for (let j = 0; j < k; j++) {
          const v = (G.X[c] - C[j][0]) ** 2 + (G.Y[c] - C[j][1]) ** 2 - b[j];
          if (v < bv) { bv = v; best = j; }
        }
        reg[c] = best;
      }
      const L = new Float64Array(k), sx = new Float64Array(k), sy = new Float64Array(k), sw = new Float64Array(k);
      for (let c = 0; c < G.n; c++) {
        const j = reg[c]; L[j] += W[c];
        const w = W[c] + 1e-6; sx[j] += G.X[c] * w; sy[j] += G.Y[c] * w; sw[j] += w;
      }
      const step = 0.6 * S * (it < 80 ? 1 : 0.4);
      for (let j = 0; j < k; j++) {
        b[j] += step * (target - L[j]) * k;
        if (sw[j] > 0 && it % 3 === 0 && it < 100) {
          C[j][0] = 0.5 * C[j][0] + 0.5 * sx[j] / sw[j];
          C[j][1] = 0.5 * C[j][1] + 0.5 * sy[j] / sw[j];
        }
      }
    }
    return reg;
  }

  function components(G, reg, j) {
    const seen = new Map();
    const comps = [];
    for (let c = 0; c < G.n; c++) {
      if (reg[c] !== j || seen.has(c)) continue;
      const comp = [c]; seen.set(c, comps.length);
      for (let q = 0; q < comp.length; q++) {
        for (const m of G.N4[comp[q]]) if (m >= 0 && reg[m] === j && !seen.has(m)) { seen.set(m, comps.length); comp.push(m); }
      }
      comps.push(comp);
    }
    return comps;
  }

  // מצמיד חלקים מנותקים לשכנה הגדולה. מחזיר רשימת שכונות ריקות (אם יש).
  function fixContiguity(G, reg, k) {
    for (let round = 0; round < 6; round++) {
      let changed = false;
      for (let j = 0; j < k; j++) {
        const comps = components(G, reg, j);
        if (comps.length <= 1) continue;
        comps.sort((a, b) => b.length - a.length);
        for (let q = 1; q < comps.length; q++) {
          const cnt = new Map();
          for (const c of comps[q]) for (const m of G.N4[c]) if (m >= 0 && reg[m] !== j) cnt.set(reg[m], (cnt.get(reg[m]) || 0) + 1);
          let best = -1, bv = -1;
          for (const [r, v] of cnt) if (v > bv) { bv = v; best = r; }
          if (best >= 0) { for (const c of comps[q]) reg[c] = best; changed = true; }
        }
      }
      if (!changed) break;
    }
  }

  // האם הסרת c מהשכונה שלה משאירה אותה רציפה?
  function removable(G, reg, c) {
    const j = reg[c];
    const ring = G.N8[c].map((m) => m >= 0 && reg[m] === j);
    // שכני 4 באותה שכונה: אינדקסים זוגיים בטבעת
    let same4 = 0; for (let q = 0; q < 8; q += 2) if (ring[q]) same4++;
    if (same4 === 0) return false; // תא בודד: לא מוציאים (משאיר שכונה ריקה או מנותקת)
    if (same4 === 1) return true;
    // מספר רצפים של תאים באותה שכונה בטבעת (בלי פינות שלא נוגעות בצד)
    let runs = 0;
    for (let q = 0; q < 8; q++) if (ring[q] && !ring[(q + 7) % 8]) runs++;
    if (runs === 1) return true;
    return fullCheck(G, reg, c);
  }

  function fullCheck(G, reg, c) {
    const j = reg[c];
    const start = G.N4[c].find((m) => m >= 0 && reg[m] === j);
    const need = G.N4[c].filter((m) => m >= 0 && reg[m] === j);
    const seen = new Set([start, c]);
    const st = [start];
    while (st.length) {
      const x = st.pop();
      for (const m of G.N4[x]) if (m >= 0 && reg[m] === j && !seen.has(m)) { seen.add(m); st.push(m); }
    }
    return need.every((m) => seen.has(m));
  }

  function localSearch(G, reg, k, opt, maxPass) {
    const { pops, areas } = loads(G, reg, k);
    const cnt = new Int32Array(k);
    for (let c = 0; c < G.n; c++) cnt[reg[c]]++;
    const Pt = G.Pt / k, At = G.At / k, a = opt.alpha;
    const term = (p, ar) => a * (p / Pt - 1) ** 2 + (1 - a) * (ar / At - 1) ** 2;
    const mu = opt.mu / CUT_SCALE;
    for (let pass = 0; pass < maxPass; pass++) {
      let moves = 0;
      for (let c = 0; c < G.n; c++) {
        const r = reg[c];
        let bestS = -1, bestD = -1e-12;
        const tried = new Set();
        for (const m of G.N4[c]) {
          if (m < 0) continue;
          const s = reg[m];
          if (s === r || tried.has(s)) continue;
          tried.add(s);
          let dcut = 0;
          for (const q of G.N4[c]) if (q >= 0) dcut += (reg[q] !== s) - (reg[q] !== r);
          const d = term(pops[r] - G.P[c], areas[r] - G.A[c]) + term(pops[s] + G.P[c], areas[s] + G.A[c])
            - term(pops[r], areas[r]) - term(pops[s], areas[s]) + mu * dcut;
          if (d < bestD) { bestD = d; bestS = s; }
        }
        if (bestS >= 0 && cnt[r] > 1 && removable(G, reg, c)) {
          pops[r] -= G.P[c]; areas[r] -= G.A[c]; cnt[r]--;
          pops[bestS] += G.P[c]; areas[bestS] += G.A[c]; cnt[bestS]++;
          reg[c] = bestS; moves++;
        }
      }
      if (!moves) break;
    }
    return reg;
  }

  function sumAr(G) {
    if (!G._sumAr) { let s = 0; for (let c = 0; c < G.n; c++) s += G.Ar[c]; G._sumAr = s; }
    return G._sumAr;
  }

  // שכונה שנשארה ריקה מקבלת תא גבול מהשכונה הכבדה ביותר, וגדלה בחיפוש המקומי
  function fillEmpty(G, reg, k, W) {
    for (let guard = 0; guard < k; guard++) {
      const L = new Float64Array(k), cnt = new Int32Array(k);
      for (let c = 0; c < G.n; c++) { L[reg[c]] += W[c]; cnt[reg[c]]++; }
      const j = cnt.findIndex((v) => v === 0);
      if (j < 0) return;
      let r = 0; for (let q = 1; q < k; q++) if (L[q] > L[r]) r = q;
      // תא בשכונה r שהכי רחוק ממרכזה וניתן להוצאה
      let sx = 0, sy = 0, sw = 0;
      for (let c = 0; c < G.n; c++) if (reg[c] === r) { sx += G.X[c]; sy += G.Y[c]; sw++; }
      sx /= sw; sy /= sw;
      let best = -1, bd = -1;
      for (let c = 0; c < G.n; c++) {
        if (reg[c] !== r) continue;
        const d = ((G.X[c] - sx) ** 2 + (G.Y[c] - sy) ** 2) * (W[c] + 1e-7);
        if (d > bd && removable(G, reg, c)) { bd = d; best = c; }
      }
      if (best < 0) return;
      reg[best] = j;
    }
  }

  // opt: {k, alpha (0=שטח בלבד, 1=אוכלוסייה בלבד), mu (קומפקטיות), seed, restarts}
  function solve(G, opt) {
    const k = opt.k;
    // משקל לשלב הגאומטרי: תמיד קצת שטח אמיתי, כדי שגם שדות ריקים "ימשכו" מישהו.
    // ההתחלה הגאומטרית נעשית בכמה ערבובים שונים (לא רק alpha עצמו), כי לפעמים
    // התחלה "מוטה אוכלוסייה" מגיעה לפתרון טוב יותר גם כשהמטרה היא איזון.
    const geoAlphas = [...new Set([opt.alpha, Math.min(0.9, opt.alpha + 0.2), 0.75])];
    const mkW = (al) => {
      const W = new Float64Array(G.n);
      const ap = Math.min(al, 0.95);
      for (let c = 0; c < G.n; c++) W[c] = ap * G.P[c] / G.Pt + (1 - ap) * (0.9 * G.A[c] / G.At + 0.1 * G.Ar[c] / sumAr(G));
      return W;
    };
    // גם בחיפוש המקומי: אחוז קטן של שטח, כדי ששכונה תקועה תמיד תוכל לגדול
    opt = Object.assign({}, opt, { alpha: Math.min(opt.alpha, 0.98) });
    let best = null, bestObj = Infinity;
    const restarts = opt.restarts || 2;
    for (let t = 0; t < restarts * geoAlphas.length; t++) {
      const W = mkW(geoAlphas[t % geoAlphas.length]);
      const rand = rng((opt.seed || 1) * 7919 + t * 104729 + k * 31);
      const C = seeds(G, k, W, rand);
      const reg = powerBalance(G, k, W, C);
      fixContiguity(G, reg, k);
      fillEmpty(G, reg, k, W);
      localSearch(G, reg, k, opt, 60);
      const o = objective(G, reg, k, opt);
      if (o < bestObj) { bestObj = o; best = reg; }
    }
    return { reg: best, obj: bestObj };
  }

  function summary(G, reg, k) {
    const { pops, areas } = loads(G, reg, k);
    const Pt = G.Pt / k, At = G.At / k;
    const pd = Array.from(pops, (p) => p / Pt - 1), ad = Array.from(areas, (a) => a / At - 1);
    const rms = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
    let cut = 0;
    for (let c = 0; c < G.n; c++) for (const m of G.N4[c]) if (m > c && reg[m] !== reg[c]) cut++;
    return {
      pops: Array.from(pops), areas: Array.from(areas), pd, ad,
      maxP: Math.max(...pd.map(Math.abs)), maxA: Math.max(...ad.map(Math.abs)),
      rmsP: rms(pd), rmsA: rms(ad), cutKm: cut * 0.1,
    };
  }

  const API = { prepare, solve, summary, objective, components, removable, loads };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.Partition = API;
})(typeof window !== 'undefined' ? window : globalThis);
