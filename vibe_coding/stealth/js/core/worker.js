// עובד רקע: כל החישובים הכבדים (תבניות קוטביות, בועה תלת-ממדית, טבלאות למבחן החדירה)
import * as M from './model.js';
import * as PO from './po.js';

let cacheKey = '', model = null;
function getModel(p) {
  const k = JSON.stringify(p);
  if (k !== cacheKey) { model = M.build(p); cacheKey = k; }
  return model;
}

self.onmessage = (ev) => {
  const { id, type, p, opts } = ev.data;
  try {
    const m = getModel(p);
    if (type === 'polar') {
      const t0 = performance.now();
      const raw = M.polarFor(m, opts.band, opts.el, opts.n || 1440, opts);
      const sm = PO.smooth(raw, opts.win || 10);
      self.postMessage({ id, raw, sm, ms: performance.now() - t0 });
    } else if (type === 'bubble') {
      // סכום לא קוהרנטי על רשת כיוונים
      const f = M.BANDS[opts.band].f, k = M.kOf(f), g = M.gammaFor(m.ac.p, f);
      const na = opts.na || 72, ne = opts.ne || 37, out = new Float64Array(na * ne);
      for (let j = 0; j < ne; j++) {
        const el = -90 + 180 * j / (ne - 1);
        for (let i = 0; i < na; i++) out[j * na + i] = PO.sigmaIncoherent(m.P, PO.dirFromAzEl(i * 360 / na, el), k, g);
      }
      self.postMessage({ id, out, na, ne });
    } else if (type === 'table') {
      // σ מוחלק לכל תדר בכמה זוויות גובה, למבחן החדירה
      const res = {};
      for (const b of opts.bands) {
        res[b] = opts.els.map((el) => PO.smooth(M.polarFor(m, b, el, 720, opts), 6));
        self.postMessage({ id, progress: b });
      }
      self.postMessage({ id, res, els: opts.els });
    } else if (type === 'facets') {
      // תרומת כל פאה לכיוון המכ"ם (למפת חום על המטוס)
      const f = M.BANDS[opts.band].f, k = M.kOf(f), g = M.gammaFor(m.ac.p, f), r = opts.r;
      const P = m.P, w = 2 * k, lam = 2 * Math.PI / k, s2 = 4 * Math.PI / (lam * lam), o2 = [0, 0];
      const out = new Float32Array(P.nt);
      for (let t = 0; t < P.nt; t++) {
        const c = P.n[3 * t] * r[0] + P.n[3 * t + 1] * r[1] + P.n[3 * t + 2] * r[2];
        if (c <= 0) continue;
        const o = 9 * t, v = P.v;
        PO.triIntegral(w * (v[o] * r[0] + v[o + 1] * r[1] + v[o + 2] * r[2]), w * (v[o + 3] * r[0] + v[o + 4] * r[1] + v[o + 5] * r[2]), w * (v[o + 6] * r[0] + v[o + 7] * r[1] + v[o + 8] * r[2]), P.A[t], o2);
        out[t] = s2 * c * c * g[P.part[t]] ** 2 * (o2[0] ** 2 + o2[1] ** 2);
      }
      self.postMessage({ id, out });
    }
  } catch (e) {
    self.postMessage({ id, error: String(e && e.stack || e) });
  }
};
