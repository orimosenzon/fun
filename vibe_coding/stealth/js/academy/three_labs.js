// מעבדות תלת-ממד באקדמיה: בועת RCS קוהרנטית, ואלומת קרניים על מטוס
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as PO from '../core/po.js';
import { buildAircraft, PRESETS } from '../core/geometry.js';
import { polarDraw } from './main.js';

const $ = (s) => document.querySelector(s);
const T = (x, y, z) => new THREE.Vector3(x, z, -y);
const c0 = 299792458;

function scene3(el, dist) {
  const r = new THREE.WebGLRenderer({ antialias: true }); r.setPixelRatio(Math.min(2, devicePixelRatio)); r.setClearColor(0x070b14);
  el.prepend(r.domElement);
  const sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  cam.position.set(-dist * 0.7, dist * 0.45, dist * 0.6);
  const ctl = new OrbitControls(cam, r.domElement); ctl.enableDamping = true;
  sc.add(new THREE.HemisphereLight(0xa9c6ff, 0x1a1f2b, 1.0));
  const d = new THREE.DirectionalLight(0xffffff, 1.4); d.position.set(-5, 10, 4); sc.add(d);
  const fit = () => { const w = el.clientWidth, h = el.clientHeight; r.setSize(w, h, false); r.domElement.style.width = w + 'px'; r.domElement.style.height = h + 'px'; cam.aspect = w / h; cam.updateProjectionMatrix(); };
  new ResizeObserver(fit).observe(el); fit();
  let vis = true; new IntersectionObserver((e) => { vis = e[0].isIntersecting; }).observe(el);
  const loop = () => { requestAnimationFrame(loop); if (!vis) return; ctl.update(); r.render(sc, cam); };
  loop();
  return { r, sc, cam, ctl };
}

function meshFrom(mesh, color = 0x56709a) {
  const P = PO.prepare(mesh), a = new Float32Array(P.nt * 9);
  for (let t = 0; t < P.nt; t++) for (let k = 0; k < 3; k++) { const v = T(P.v[9 * t + 3 * k], P.v[9 * t + 3 * k + 1], P.v[9 * t + 3 * k + 2]); a.set([v.x, v.y, v.z], 9 * t + 3 * k); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(a, 3)); g.computeVertexNormals();
  return { P, obj: new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color, flatShading: true, metalness: 0.4, roughness: 0.5, side: THREE.DoubleSide })) };
}

// ---- גופים פשוטים כרשתות משולשים (נורמלים החוצה) ----
function shapes() {
  const plate = { pos: [0.5, 0.5, 0, -0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, -0.5, 0], tri: [0, 1, 2, 0, 2, 3] };
  const cube = (() => {
    const pos = [], tri = [], h = 0.5;
    const faces = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
    for (const [n, u, v] of faces) {
      const b = pos.length / 3;
      for (const [su, sv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) pos.push(h * (n[0] + su * u[0] + sv * v[0]), h * (n[1] + su * u[1] + sv * v[1]), h * (n[2] + su * u[2] + sv * v[2]));
      tri.push(b, b + 1, b + 2, b, b + 2, b + 3);
    }
    return { pos, tri };
  })();
  const cyl = (() => {
    const pos = [], tri = [], N = 32, r = 0.3, L = 1.6;
    for (let i = 0; i < N; i++) { const a = 2 * Math.PI * i / N; pos.push(-L / 2, r * Math.cos(a), r * Math.sin(a), L / 2, r * Math.cos(a), r * Math.sin(a)); }
    for (let i = 0; i < N; i++) { const a = 2 * i, b = 2 * ((i + 1) % N); tri.push(a, b, b + 1, a, b + 1, a + 1); }
    const c1 = pos.length / 3; pos.push(-L / 2, 0, 0); const c2 = c1 + 1; pos.push(L / 2, 0, 0);
    for (let i = 0; i < N; i++) { const a = 2 * i, b = 2 * ((i + 1) % N); tri.push(c1, b, a, c2, a + 1, b + 1); }
    return { pos, tri };
  })();
  const sphere = (() => {
    const g = new THREE.IcosahedronGeometry(0.5, 3), p = g.attributes.position.array, pos = [], tri = [];
    for (let i = 0; i < p.length; i += 3) pos.push(p[i], -p[i + 2], p[i + 1]);
    for (let i = 0; i < pos.length / 3; i += 3) tri.push(i, i + 1, i + 2);
    return { pos, tri };
  })();
  const cone = (() => {
    const pos = [0.8, 0, 0], tri = [], N = 32, r = 0.35;
    for (let i = 0; i < N; i++) { const a = 2 * Math.PI * i / N; pos.push(-0.6, r * Math.cos(a), r * Math.sin(a)); }
    pos.push(-0.6, 0, 0); const c = N + 1;
    for (let i = 0; i < N; i++) { const a = 1 + i, b = 1 + (i + 1) % N; tri.push(0, a, b, c, b, a); }
    return { pos, tri };
  })();
  const dihedral = { pos: [0, -0.6, 0, 0, 0.6, 0, -0.8, -0.6, 0.8, -0.8, 0.6, 0.8, -0.8, -0.6, -0.8, -0.8, 0.6, -0.8], tri: [0, 2, 1, 1, 2, 3, 0, 1, 4, 1, 5, 4] };
  // הנורמלים צריכים להצביע החוצה; לפינה הכפולה הצד הפנימי הוא הצד המחזיר
  return {
    plate: ['לוח 1×1', plate], cube: ['קובייה', cube], cyl: ['גליל', cyl], sphere: ['כדור', sphere], cone: ['חרוט', cone], dihedral: ['פינה כפולה', dihedral],
  };
}

function fixNormals(mesh, inward = false) {
  // מכוון את המשולשים כך שהנורמל מצביע מהמרכז החוצה (או פנימה לפינה)
  const P = mesh.pos, T2 = mesh.tri;
  let cx = 0, cy = 0, cz = 0; const n = P.length / 3; for (let i = 0; i < n; i++) { cx += P[3 * i]; cy += P[3 * i + 1]; cz += P[3 * i + 2]; }
  cx /= n; cy /= n; cz /= n;
  for (let t = 0; t < T2.length; t += 3) {
    const a = T2[t] * 3, b = T2[t + 1] * 3, c = T2[t + 2] * 3;
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], wx = P[c] - P[a], wy = P[c + 1] - P[a + 1], wz = P[c + 2] - P[a + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const mx = (P[a] + P[b] + P[c]) / 3 - cx, my = (P[a + 1] + P[b + 1] + P[c + 1]) / 3 - cy, mz = (P[a + 2] + P[b + 2] + P[c + 2]) / 3 - cz;
    let s = nx * mx + ny * my + nz * mz; if (inward) s = -s;
    if (s < 0) { const tmp = T2[t + 1]; T2[t + 1] = T2[t + 2]; T2[t + 2] = tmp; }
  }
  return mesh;
}

export function labPO3D() {
  const S = shapes();
  for (const k of ['cube', 'cyl', 'sphere', 'cone']) fixNormals(S[k][1]);
  fixNormals(S.dihedral[1], true);
  const { sc } = scene3($('#po3d-view'), 4.2);
  let cur = 'plate', obj = null, bubble = null, P = null;
  sc.add(new THREE.AxesHelper(0.9));
  const build = () => {
    if (obj) sc.remove(obj);
    const m = meshFrom(S[cur][1]); P = m.P; obj = m.obj; sc.add(obj);
    compute();
  };
  let tmr = 0;
  const compute = () => {
    clearTimeout(tmr);
    tmr = setTimeout(() => {
      const g = +$('#po3d-f').value, k = 2 * Math.PI * g * 1e9 / c0, el = +$('#po3d-e').value;
      $('#po3d-fv').textContent = `${g.toFixed(1)} GHz (λ = ${(c0 / g / 1e7).toFixed(1)} ס"מ)`; $('#po3d-ev').textContent = el + '°';
      const na = 96, ne = 49, vals = new Float64Array(na * ne);
      for (let j = 0; j < ne; j++) for (let i = 0; i < na; i++) vals[j * na + i] = PO.sigma(P, PO.dirFromAzEl(i * 360 / na, -90 + 180 * j / (ne - 1)), k, null, null, { noEdges: true });
      let mx = 0; for (const v of vals) mx = Math.max(mx, v);
      const top = PO.dB(mx), lo = top - 40, R = 1.7;
      const pos = [], col = [], idx = [], c = new THREE.Color();
      for (let j = 0; j < ne; j++) for (let i = 0; i <= na; i++) {
        const v = vals[j * na + (i % na)], u = Math.max(0, (PO.dB(v) - lo) / 40);
        const e = (-90 + 180 * j / (ne - 1)) * Math.PI / 180, a = i * 2 * Math.PI / na, rr = R * (0.02 + 0.98 * u);
        const p = T(rr * Math.cos(e) * Math.cos(a), rr * Math.cos(e) * Math.sin(a), rr * Math.sin(e));
        pos.push(p.x, p.y, p.z); c.setHSL(0.62 - 0.62 * u, 0.95, 0.3 + 0.3 * u); col.push(c.r, c.g, c.b);
      }
      for (let j = 0; j + 1 < ne; j++) for (let i = 0; i < na; i++) { const a = j * (na + 1) + i, b = a + 1, cc = a + na + 1, d = cc + 1; idx.push(a, b, d, a, d, cc); }
      if (bubble) { sc.remove(bubble); bubble.geometry.dispose(); }
      const g2 = new THREE.BufferGeometry(); g2.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g2.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); g2.setIndex(idx);
      bubble = new THREE.Mesh(g2, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false }));
      sc.add(bubble);
      const pol = new Float64Array(720);
      for (let i = 0; i < 720; i++) pol[i] = PO.sigma(P, PO.dirFromAzEl(i / 2, el), k, null, null, { noEdges: true });
      let pm = 0; for (const v of pol) pm = Math.max(pm, v);
      const hi = Math.ceil(PO.dB(pm) / 10) * 10;
      polarDraw($('#po3d-polar'), [{ arr: pol, color: '#7cf0c5', w: 1.6 }], hi - 50, hi);
      $('#po3d-note').innerHTML = `המקסימום: <b style="direction:ltr;display:inline-block">${top.toFixed(1)} dBsm</b>. הבועה מראה את σ בכל כיוון, 40 dB מהמרכז לשפה. זה סכום קוהרנטי: האונות והחריצים הם התאבכות אמיתית בין חלקי הגוף. ${cur === 'sphere' ? 'בכדור: σ כמעט זהה לכל כיוון ושווה πa² = ' + PO.dB(Math.PI * 0.25).toFixed(1) + ' dBsm. הריפרוף הקטן בא מהחלוקה למשולשים.' : cur === 'dihedral' ? 'בפינה הכפולה: האופטיקה הפיזיקלית סופרת רק פגיעה אחת, ולכן היא מפספסת את ההחזרה הכפולה החזקה. זה בדיוק מה שמעקב קרניים (פרק 7) משלים.' : 'הגדילו את התדר ותראו את האונות מתחדדות: רוחב אונה ≈ λ/גודל.'}`;
    }, 60);
  };
  seg($('#po3d-shape'), Object.entries(S).map(([k, v]) => [k, v[0]]), cur, (k) => { cur = k; build(); });
  $('#po3d-f').oninput = compute; $('#po3d-e').oninput = compute;
  build();
  return compute;
}

function seg(el, items, cur, cb) {
  el.innerHTML = '';
  for (const [k, n] of items) {
    const b = document.createElement('button'); b.className = 'btn' + (k === cur ? ' on' : ''); b.textContent = n;
    b.onclick = () => { el.querySelectorAll('.btn').forEach((x) => x.classList.toggle('on', x === b)); cb(k); };
    el.appendChild(b);
  }
}

// ---- אלומת קרניים ----
export function labRays() {
  const { sc, cam, ctl } = scene3($('#rays-view'), 34);
  let cur = 'faceted', P = null, obj = null, lines = null, radar = null, span = 20;
  const build = () => {
    if (obj) sc.remove(obj);
    const ac = buildAircraft(PRESETS[cur]); const m = meshFrom(ac, 0x3d5580); P = m.P; obj = m.obj; sc.add(obj);
    span = Math.max(ac.p.L, ac.p.wing.span);
    trace();
  };
  const trace = () => {
    const az = +$('#rays-a').value, el = +$('#rays-e').value, two = $('#rays-2').checked;
    $('#rays-av').textContent = az + '°'; $('#rays-ev').textContent = el + '°';
    const r = PO.dirFromAzEl(az, el), d0 = [-r[0], -r[1], -r[2]];
    // בסיס ניצב לכיוון
    const up = Math.abs(r[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    const u = norm(cross(r, up)), v = cross(r, u);
    const R = span * 0.6, n = 44, start = span * 1.2;
    const segs = [], cols = []; let hits = 0, back = 0, total = 0;
    const cosTol = Math.cos(4 * Math.PI / 180);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const a = -R + 2 * R * (i + 0.5) / n, b = -R + 2 * R * (j + 0.5) / n; if (a * a + b * b > R * R) continue;
      total++;
      let o = [r[0] * start + u[0] * a + v[0] * b, r[1] * start + u[1] * a + v[1] * b, r[2] * start + u[2] * a + v[2] * b], d = d0.slice();
      let h = hit(P, o, d, -1); if (!h) continue;
      hits++;
      const pts = [o, h.p]; let last = h.t, dd = reflect(d, h.n), bounce = 1;
      if (two) { const h2 = hit(P, h.p, dd, last); if (h2) { pts.push(h2.p); dd = reflect(dd, h2.n); bounce = 2; } }
      const end = pts[pts.length - 1];
      const ret = dot(dd, r) > cosTol;
      if (ret) back++;
      const out = [end[0] + dd[0] * span * 0.7, end[1] + dd[1] * span * 0.7, end[2] + dd[2] * span * 0.7];
      pts.push(out);
      for (let q = 0; q + 1 < pts.length; q++) {
        const A = T(...pts[q]), B = T(...pts[q + 1]);
        segs.push(A.x, A.y, A.z, B.x, B.y, B.z);
        const c = q === 0 ? [0.35, 0.3, 0.12] : ret ? [1, 0.25, 0.35] : bounce === 2 && q === 1 ? [0.9, 0.55, 0.2] : [1, 0.83, 0.3];
        cols.push(...c, ...c);
      }
    }
    if (lines) { sc.remove(lines); lines.geometry.dispose(); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }));
    sc.add(lines);
    if (!radar) { radar = new THREE.Mesh(new THREE.SphereGeometry(0.8, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffd34d })); sc.add(radar); }
    radar.position.copy(T(r[0] * start, r[1] * start, r[2] * start));
    $('#rays-k').innerHTML = [
      ['פגעו במטוס', hits + '', `מתוך ${total}`],
      ['חזרו למכ"ם', back + '', hits ? (100 * back / hits).toFixed(1) + '% מהפוגעות' : ''],
    ].map(([t, vv, c]) => `<div class="kpi"><span>${t}</span><b class="${t.startsWith('חזרו') ? (back ? 'bad' : 'good') : ''}">${vv}</b><i>${c}</i></div>`).join('');
  };
  seg($('#rays-shape'), [['conventional', 'קונבנציונלי'], ['faceted', 'רב-פאות'], ['fighter', 'קרב חמקן'], ['wing', 'כנף מעופפת']], cur, (k) => { cur = k; build(); });
  $('#rays-a').oninput = trace; $('#rays-e').oninput = trace; $('#rays-2').onchange = trace;
  cam.position.set(-30, 20, 26); ctl.update();
  build();
  return trace;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
const reflect = (d, n) => { const k = 2 * dot(d, n); return [d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2]]; };

// מולר-טרומבור על כל המשולשים
function hit(P, o, d, skip) {
  let best = null;
  const v = P.v;
  for (let t = 0; t < P.nt; t++) {
    if (t === skip) continue;
    const q = 9 * t;
    const e1x = v[q + 3] - v[q], e1y = v[q + 4] - v[q + 1], e1z = v[q + 5] - v[q + 2];
    const e2x = v[q + 6] - v[q], e2y = v[q + 7] - v[q + 1], e2z = v[q + 8] - v[q + 2];
    const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
    const det = e1x * px + e1y * py + e1z * pz; if (Math.abs(det) < 1e-12) continue;
    const inv = 1 / det, tx = o[0] - v[q], ty = o[1] - v[q + 1], tz = o[2] - v[q + 2];
    const uu = (tx * px + ty * py + tz * pz) * inv; if (uu < 0 || uu > 1) continue;
    const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
    const vv = (d[0] * qx + d[1] * qy + d[2] * qz) * inv; if (vv < 0 || uu + vv > 1) continue;
    const tt = (e2x * qx + e2y * qy + e2z * qz) * inv; if (tt <= 1e-6) continue;
    if (!best || tt < best.d) best = { d: tt, t };
  }
  if (!best) return null;
  let n = [P.n[3 * best.t], P.n[3 * best.t + 1], P.n[3 * best.t + 2]];
  if (dot(n, d) > 0) n = [-n[0], -n[1], -n[2]];
  return { p: [o[0] + d[0] * best.d, o[1] + d[1] * best.d, o[2] + d[2] * best.d], n, t: best.t };
}
