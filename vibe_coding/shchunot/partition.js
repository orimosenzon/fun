// חלוקת גרף של בלוקים ל-k שכונות רציפות, עם איזון בין אוכלוסייה לשטח.
// בלוק = שטח שתחום בגבולות טבעיים (רחובות, ואדיות, שולי חורשות, מסילה; ראו build_data.py).
// שכונה היא אוסף של בלוקים שלמים, ולכן כל גבול בין שכונות עובר לאורך גבול טבעי.
// לכל שכנות יש "מחיר": אורך הגבול המשותף כפול משקל לפי סוגו (כביש ראשי, ואדי וחורשה
// זולים, סמטה יקרה), כך שהאלגוריתם מעדיף לחתוך לאורך גבולות בולטים.
// עובד גם בדפדפן (window.Partition) וגם ב-Node (module.exports).
//
// שלבים:
//  1. זרעים בשיטת k-means++ משוקללת (מחולל אקראי עם seed, כדי שהתוצאה תהיה ניתנת לשחזור).
//  2. דיאגרמת חזקה מאוזנת: כל בלוק הולך למרכז שממזער d² - b_j, וה-b_j מתכווננים
//     עד שלכל שכונה "משקל" דומה (ערבוב של אוכלוסייה ושטח לפי alpha).
//  3. תיקון רציפות: חלקים מנותקים מוצמדים לשכנה.
//  4. חיפוש מקומי: העברת בלוקי גבול בין שכונות כשזה משפר את פונקציית המטרה,
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

  // בונה גרף: nodes = [{x, y, area, pop}], edges = [[a, b, len, cost]]
  function graph(nodes, edges, areaOf) {
    const n = nodes.length;
    const X = new Float64Array(n), Y = new Float64Array(n), A = new Float64Array(n), Ar = new Float64Array(n), P = new Float64Array(n);
    nodes.forEach((b, q) => { X[q] = b.x; Y[q] = b.y; Ar[q] = b.area; A[q] = areaOf ? areaOf(q) : b.area; P[q] = b.pop; });
    const N = Array.from({ length: n }, () => []), W = Array.from({ length: n }, () => []), Ln = Array.from({ length: n }, () => []);
    for (const [a, b, len, cost] of edges) {
      N[a].push(b); W[a].push(cost); Ln[a].push(len);
      N[b].push(a); W[b].push(cost); Ln[b].push(len);
    }
    let Pt = 0, At = 0;
    for (let q = 0; q < n; q++) { Pt += P[q]; At += A[q]; }
    return { n, X, Y, A, Ar, P, N, W, Ln, Pt, At };
  }
  const nodesOf = (D) => D.blocks.map((b) => ({ x: b[0], y: b[1], area: b[2], pop: b[3] }));

  // קיטום: אילו בלוקים נשארים בתחום החלוקה.
  // trim = אחוז התושבים שמותר שיגורו בשטח הקטום. הבלוקים נקטמים מהדליל לצפוף:
  // ב-0 נקטמים רק בלוקים בלי תושבים בכלל (חקלאות, תעשייה, שטח פתוח), ומעליו גם
  // בלוקים דלילים. אחר כך חורים בתוך היישוב חוזרים פנימה: חור שגרים בו תמיד,
  // וחור ריק רק אם הוא קטן (גן, מגרש, בית ספר). נשאר רק הרצף הגדול.
  // בלוק קטום עדיין שייך לשכונה: לזו של הבלוק המיושב הקרוב. התושבים שלו נספרים, השטח לא.
  const HOLE_MAX = 100000; // מ"ר
  function trimMask(D, trim) {
    const B = D.blocks, n = B.length;
    const G = graph(nodesOf(D), D.adj);
    const ord = [...Array(n).keys()].sort((a, b) => B[a][3] / B[a][2] - B[b][3] / B[b][2]);
    const keep = new Uint8Array(n).fill(1);
    let cum = 0;
    for (const c of ord) {
      if (B[c][3] > 0 && cum + B[c][3] > trim / 100 * G.Pt) break;
      keep[c] = 0; cum += B[c][3];
    }
    const comps = (want) => {
      const seen = new Uint8Array(n), out = [];
      for (let c = 0; c < n; c++) {
        if (keep[c] !== want || seen[c]) continue;
        const q = [c]; seen[c] = 1;
        for (let i = 0; i < q.length; i++) for (const m of G.N[q[i]]) if (keep[m] === want && !seen[m]) { seen[m] = 1; q.push(m); }
        out.push(q);
      }
      return out;
    };
    for (const q of comps(0)) {
      if (q.some((c) => B[c][5])) continue; // נוגע בגבול המועצה: לא חור
      const area = q.reduce((s, c) => s + B[c][2], 0);
      if (area <= HOLE_MAX || q.some((c) => B[c][3] > 0)) for (const c of q) keep[c] = 1;
    }
    const main = comps(1).sort((a, b) => b.length - a.length)[0];
    keep.fill(0);
    for (const c of main) keep[c] = 1;
    return keep;
  }

  // opts.trim (0 ומעלה): קיטום שטחים לא מיושבים. G נשאר על כל הבלוקים (לתצוגה ולסיכום),
  //   ו-G.sub הוא גרף הבלוקים שנשארו, שעליו פותרים.
  // opts.inhabitedOnly: בלי קיטום, אבל שטח של אזורים סטטיסטיים בלי תושבים לא נספר.
  function prepare(D, opts) {
    opts = opts || {};
    const nodes = nodesOf(D);
    if (!(opts.trim >= 0)) {
      const empty = new Set(D.stats.filter((s) => !s.pop).map((s) => s.id));
      return graph(nodes, D.adj, opts.inhabitedOnly ? (q) => (empty.has(D.blocks[q][4]) ? 0 : D.blocks[q][2]) : null);
    }
    const keep = trimMask(D, opts.trim);
    const n = nodes.length;
    const kept = []; for (let c = 0; c < n; c++) if (keep[c]) kept.push(c);
    const subOf = new Int32Array(n).fill(-1); kept.forEach((c, q) => { subOf[c] = q; });
    // בלוק קטום → הבלוק המיושב הקרוב
    const host = new Int32Array(n);
    for (let c = 0; c < n; c++) {
      if (keep[c]) { host[c] = c; continue; }
      let best = -1, bd = Infinity;
      for (const q of kept) { const d = (nodes[q].x - nodes[c].x) ** 2 + (nodes[q].y - nodes[c].y) ** 2; if (d < bd) { bd = d; best = q; } }
      host[c] = best;
    }
    const settled = (q) => D.blocks[q][8];
    const subNodes = kept.map((c) => Object.assign({}, nodes[c], { pop: 0, area: settled(c) }));
    for (let c = 0; c < n; c++) subNodes[subOf[host[c]]].pop += nodes[c].pop;
    const subEdges = D.adj.filter((e) => keep[e[0]] && keep[e[1]]).map((e) => [subOf[e[0]], subOf[e[1]], e[2], e[3]]);
    // השטח שנספר הוא השטח המיושב בבלוק (רדיוס 30 מ' סביב בתים), כדי שפרדס עם משק אחד
    // לא ייחשב כאילו כולו שכונה
    const G = graph(nodes, D.adj, (q) => (keep[q] ? settled(q) : 0));
    G.sub = graph(subNodes, subEdges);
    G.keep = keep; G.host = host; G.subOf = subOf; G.trim = opts.trim;
    G.trimPop = 0; G.trimArea = 0;
    for (let c = 0; c < n; c++) if (!keep[c]) { G.trimPop += G.P[c]; G.trimArea += G.Ar[c]; }
    return G;
  }

  const live = (G, c) => !G.keep || G.keep[c];
  // מחיר גבול: מטר של רחוב שכונתי (משקל 3) לכל 100 מ' כמו צלע של תא ברשת הישנה
  const CUT_SCALE = 450000;

  function cutCost(G, reg) {
    let cut = 0;
    for (let c = 0; c < G.n; c++) G.N[c].forEach((m, q) => { if (m > c && reg[m] !== reg[c] && live(G, c) && live(G, m)) cut += G.W[c][q]; });
    return cut;
  }

  function objective(G, reg, k, opt) {
    const { pops, areas } = loads(G, reg, k);
    const Pt = G.Pt / k, At = G.At / k;
    let bal = 0;
    for (let j = 0; j < k; j++) {
      bal += opt.alpha * (pops[j] / Pt - 1) ** 2 + (1 - opt.alpha) * (areas[j] / At - 1) ** 2;
    }
    return bal + opt.mu * cutCost(G, reg) / CUT_SCALE;
  }

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
    const S = G.At / k;
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

  // רכיבי קשירות של שכונה j (רק בלוקים שבתחום החלוקה)
  function components(G, reg, j) {
    const seen = new Uint8Array(G.n);
    const comps = [];
    for (let c = 0; c < G.n; c++) {
      if (reg[c] !== j || seen[c] || !live(G, c)) continue;
      const comp = [c]; seen[c] = 1;
      for (let q = 0; q < comp.length; q++) {
        for (const m of G.N[comp[q]]) if (reg[m] === j && !seen[m] && live(G, m)) { seen[m] = 1; comp.push(m); }
      }
      comps.push(comp);
    }
    return comps;
  }

  function fixContiguity(G, reg, k) {
    for (let round = 0; round < 6; round++) {
      let changed = false;
      for (let j = 0; j < k; j++) {
        const comps = components(G, reg, j);
        if (comps.length <= 1) continue;
        comps.sort((a, b) => b.length - a.length);
        for (let q = 1; q < comps.length; q++) {
          const cnt = new Map();
          for (const c of comps[q]) G.N[c].forEach((m, e) => { if (reg[m] !== j) cnt.set(reg[m], (cnt.get(reg[m]) || 0) + G.Ln[c][e]); });
          let best = -1, bv = -1;
          for (const [r, v] of cnt) if (v > bv) { bv = v; best = r; }
          if (best >= 0) { for (const c of comps[q]) reg[c] = best; changed = true; }
        }
      }
      if (!changed) break;
    }
  }

  // האם הוצאת c מהשכונה שלה משאירה אותה רציפה?
  function removable(G, reg, c) {
    const j = reg[c];
    const same = G.N[c].filter((m) => reg[m] === j);
    if (same.length === 0) return false;
    if (same.length === 1) return true;
    const need = new Set(same);
    const seen = new Uint8Array(G.n); seen[c] = 1; seen[same[0]] = 1;
    const st = [same[0]]; need.delete(same[0]);
    while (st.length && need.size) {
      const x = st.pop();
      for (const m of G.N[x]) if (reg[m] === j && !seen[m]) { seen[m] = 1; need.delete(m); st.push(m); }
    }
    return need.size === 0;
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
        for (const m of G.N[c]) {
          const s = reg[m];
          if (s === r || tried.has(s)) continue;
          tried.add(s);
          let dcut = 0;
          G.N[c].forEach((q, e) => { dcut += G.W[c][e] * ((reg[q] !== s) - (reg[q] !== r)); });
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

  // שכונה שנשארה ריקה מקבלת בלוק גבול מהשכונה הכבדה ביותר, וגדלה בחיפוש המקומי
  function fillEmpty(G, reg, k, W) {
    for (let guard = 0; guard < k; guard++) {
      const L = new Float64Array(k), cnt = new Int32Array(k);
      for (let c = 0; c < G.n; c++) { L[reg[c]] += W[c]; cnt[reg[c]]++; }
      const j = cnt.findIndex((v) => v === 0);
      if (j < 0) return;
      let r = 0; for (let q = 1; q < k; q++) if (L[q] > L[r]) r = q;
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

  // opt: {k, alpha (0=שטח בלבד, 1=אוכלוסייה בלבד), mu (מחיר גבולות), seed, restarts}
  function solve(G, opt) {
    if (G.sub) {
      const r = solve(G.sub, opt);
      const reg = new Int32Array(G.n);
      for (let c = 0; c < G.n; c++) reg[c] = r.reg[G.subOf[G.host[c]]];
      return { reg, obj: r.obj };
    }
    const k = opt.k;
    const geoAlphas = [...new Set([opt.alpha, Math.min(0.9, opt.alpha + 0.2), 0.75])];
    const mkW = (al) => {
      const W = new Float64Array(G.n);
      const ap = Math.min(al, 0.95);
      for (let c = 0; c < G.n; c++) W[c] = ap * G.P[c] / G.Pt + (1 - ap) * (0.9 * G.A[c] / (G.At || 1) + 0.1 * G.Ar[c] / sumAr(G));
      return W;
    };
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
      localSearch(G, reg, k, opt, 80);
      const o = objective(G, reg, k, opt);
      if (o < bestObj) { bestObj = o; best = reg; }
    }
    return { reg: best, obj: bestObj };
  }

  // NATURAL: משקל למטר עד שבו גבול נחשב "בולט" (כביש ראשי, מאסף, ואדי, חורשה, פארק, מסילה)
  const NATURAL = 1.7;
  function summary(G, reg, k) {
    const { pops, areas } = loads(G, reg, k);
    const Pt = G.Pt / k, At = G.At / k;
    const pd = Array.from(pops, (p) => p / Pt - 1), ad = Array.from(areas, (a) => a / At - 1);
    const rms = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
    let len = 0, strong = 0;
    for (let c = 0; c < G.n; c++) G.N[c].forEach((m, e) => {
      if (m > c && reg[m] !== reg[c] && live(G, c) && live(G, m)) {
        len += G.Ln[c][e];
        if (G.W[c][e] <= NATURAL * G.Ln[c][e]) strong += G.Ln[c][e];
      }
    });
    return {
      pops: Array.from(pops), areas: Array.from(areas), pd, ad,
      maxP: Math.max(...pd.map(Math.abs)), maxA: Math.max(...ad.map(Math.abs)),
      rmsP: rms(pd), rmsA: rms(ad), cutKm: len / 1000, strongShare: len ? strong / len : 1,
    };
  }

  const API = { prepare, trimMask, solve, summary, objective, components, removable, loads, NATURAL };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.Partition = API;
})(typeof window !== 'undefined' ? window : globalThis);
