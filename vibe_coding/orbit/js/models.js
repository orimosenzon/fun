// models.js — מודלים תלת־ממדיים פרוצדורליים של רקטות, חלליות ולוויינים, במטרים אמיתיים.
// ציר הרקטה הוא +Y והבסיס ב-y=0. מידות מתוך מפרטי היצרנים, נאס"א וויקיפדיה (מפורטים ב-SPECS).
import * as THREE from 'three';

// ---------- חומרים ----------
const M = {};
function mat(key, params) {
  if (!M[key]) M[key] = new THREE.MeshStandardMaterial(params);
  return M[key];
}
const white = () => mat('white', { color: 0xf1f1ee, roughness: 0.55, metalness: 0.05 });
const offWhite = () => mat('offwhite', { color: 0xe4e2dc, roughness: 0.6, metalness: 0.05 });
const black = () => mat('black', { color: 0x161616, roughness: 0.6, metalness: 0.1 });
const darkGray = () => mat('dgray', { color: 0x3a3c40, roughness: 0.55, metalness: 0.4 });
const gray = () => mat('gray', { color: 0x8d9096, roughness: 0.45, metalness: 0.5 });
const steel = () => mat('steel', { color: 0xc8cbd0, roughness: 0.28, metalness: 0.92 });
const nozzleMat = () => mat('nozzle', { color: 0x4a4038, roughness: 0.5, metalness: 0.8, side: THREE.DoubleSide });
const copper = () => mat('copper', { color: 0x9a6a45, roughness: 0.4, metalness: 0.85, side: THREE.DoubleSide });
const gold = () => mat('gold', { color: 0xd9a63a, roughness: 0.32, metalness: 1.0 });
const silverFoil = () => mat('silverfoil', { color: 0xdedede, roughness: 0.22, metalness: 0.95 });
const aluminum = () => mat('alu', { color: 0xe8e8ea, roughness: 0.08, metalness: 1.0 });
const tileMat = () => mat('tiles', { color: 0x1b1b1d, roughness: 0.85, metalness: 0.05, map: hexTexture() });
const solarMat = () => mat('solar', { color: 0xffffff, roughness: 0.35, metalness: 0.5, map: solarTexture('#1b2c66', '#0d1838'), side: THREE.DoubleSide });
const solarGold = () => mat('solargold', { color: 0xffffff, roughness: 0.4, metalness: 0.5, map: solarTexture('#7a5a22', '#3d2c10'), side: THREE.DoubleSide });
const radiatorMat = () => mat('radiator', { color: 0xf4f4f4, roughness: 0.7, metalness: 0.0, side: THREE.DoubleSide });

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  draw(g, w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
let _hex, _solar = {};
function hexTexture() {
  if (_hex) return _hex;
  _hex = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = '#1e1e20'; g.fillRect(0, 0, w, h);
    g.strokeStyle = '#0c0c0d'; g.lineWidth = 2;
    const s = 16, hh = s * Math.sqrt(3);
    for (let y = -hh; y < h + hh; y += hh) for (let x = -s * 3; x < w + s * 3; x += s * 3) {
      for (const [ox, oy] of [[0, 0], [s * 1.5, hh / 2]]) {
        g.beginPath();
        for (let k = 0; k < 6; k++) { const a = k * Math.PI / 3; g.lineTo(x + ox + s * Math.cos(a), y + oy + s * Math.sin(a)); }
        g.closePath(); g.stroke();
      }
    }
  });
  _hex.wrapS = _hex.wrapT = THREE.RepeatWrapping;
  _hex.repeat.set(24, 10);
  return _hex;
}
function solarTexture(c1, c2) {
  const k = c1 + c2;
  if (_solar[k]) return _solar[k];
  const t = canvasTex(128, 128, (g, w, h) => {
    g.fillStyle = c2; g.fillRect(0, 0, w, h);
    g.fillStyle = c1;
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) g.fillRect(x * 16 + 1, y * 16 + 1, 14, 14);
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(4, 4);
  _solar[k] = t;
  return t;
}

// ---------- גאומטריה בסיסית ----------
function cyl(rTop, rBot, h, y0, material, seg = 48, open = false) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, seg, 1, open), material);
  m.position.y = y0 + h / 2;
  return m;
}
function lathe(pts, material, seg = 48, phiStart = 0, phiLen = Math.PI * 2) {
  const v = pts.map(([r, y]) => new THREE.Vector2(Math.max(r, 0.0001), y));
  return new THREE.Mesh(new THREE.LatheGeometry(v, seg, phiStart, phiLen), material);
}
function box(w, h, d, x, y, z, material) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  m.position.set(x, y, z);
  return m;
}
function sphere(r, material, seg = 32) {
  return new THREE.Mesh(new THREE.SphereGeometry(r, seg, seg / 2), material);
}
// קשת אוז'יב משיקה: נקודות [r,y] מהבסיס (רדיוס R בגובה y0) עד לחוד בגובה y0+L
function ogive(R, L, y0, n = 24, tipR = 0) {
  const rho = (R * R + L * L) / (2 * R);
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const x = L * i / n; // מרחק מהבסיס
    const r = Math.sqrt(rho * rho - x * x) + R - rho;
    pts.push([Math.max(tipR, r), y0 + x]);
  }
  pts.push([0, y0 + L]);
  return pts;
}
// פעמון נחיר: רדיוס צוואר rt ורדיוס יציאה re, באורך L, כשהצוואר למעלה ב-yTop
function nozzle(rt, re, L, yTop, material = nozzleMat()) {
  const pts = [];
  for (let i = 0; i <= 16; i++) {
    const s = i / 16;
    pts.push([rt + (re - rt) * Math.pow(s, 0.65), yTop - L * s]);
  }
  return lathe(pts, material, 32);
}
// גליל צבוע בטקסטורת קנבס (u סביב הציר, v לאורך)
function paintedCyl(r, h, y0, draw, rTop = r, seg = 64) {
  const tex = canvasTex(1024, Math.max(64, Math.round(1024 * h / (2 * Math.PI * r))), draw);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, r, h, seg, 1, true),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55, metalness: 0.05 }));
  m.position.y = y0 + h / 2;
  return m;
}
// טקסט אנכי על גליל, בזווית u (0..1) ובטווח v
function vText(g, w, h, text, u, vTop, vBot, color = '#111', font = 'bold') {
  g.save();
  const cx = u * w, top = vTop * h, bot = vBot * h;
  g.translate(cx, (top + bot) / 2);
  g.rotate(Math.PI / 2);
  g.fillStyle = color;
  const len = bot - top;
  let size = 200;
  g.font = `${font} ${size}px Arial, sans-serif`;
  const tw = g.measureText(text).width;
  size = Math.min(size * len / tw, w * 0.06);
  g.font = `${font} ${size}px Arial, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, 0, 0);
  g.restore();
}
function flag(g, x, y, fw, fh) { // דגל ארה"ב פשוט (אנכי, מסובב)
  g.save(); g.translate(x, y);
  for (let i = 0; i < 13; i++) { g.fillStyle = i % 2 ? '#fff' : '#b22234'; g.fillRect(i * fw / 13, 0, fw / 13, fh); }
  g.fillStyle = '#3c3b6e'; g.fillRect(0, 0, fw * 7 / 13, fh * 0.4);
  g.restore();
}
function gridFin(w, h, depth, material) {
  const grp = new THREE.Group();
  const n = 6, t = 0.05;
  grp.add(box(w, t * 2, depth, 0, h / 2, 0, material));
  grp.add(box(w, t * 2, depth, 0, -h / 2, 0, material));
  grp.add(box(t * 2, h, depth, -w / 2, 0, 0, material));
  grp.add(box(t * 2, h, depth, w / 2, 0, 0, material));
  for (let i = 1; i < n; i++) {
    const a = new THREE.Mesh(new THREE.BoxGeometry(t, Math.hypot(w, h) * 0.95, depth), material);
    a.rotation.z = Math.PI / 4; a.position.x = (i / n - 0.5) * w * 0.9; grp.add(a);
    const b = a.clone(); b.rotation.z = -Math.PI / 4; grp.add(b);
  }
  return grp;
}
function ring(n, radius, fn) { for (let i = 0; i < n; i++) fn(i * 2 * Math.PI / n, Math.cos(i * 2 * Math.PI / n) * radius, Math.sin(i * 2 * Math.PI / n) * radius); }
function solarPanel(w, h, material = solarMat()) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material.clone());
  m.material.map = material.map.clone();
  m.material.map.repeat.set(Math.max(1, w / 1.2), Math.max(1, h / 1.2));
  m.material.map.needsUpdate = true;
  return m;
}

// ================= רקטות =================

// Falcon 9 Block 5 — גובה 69.8 מ', קוטר 3.66 מ', כיסוי מטען 13.1×5.2 מ'
function falcon9() {
  const g = new THREE.Group();
  const R = 1.83;
  // שלב ראשון, לבן עם לוגו SpaceX אנכי
  g.add(paintedCyl(R, 36.0, 1.2, (c, w, h) => {
    c.fillStyle = '#f2f2f0'; c.fillRect(0, 0, w, h);
    c.fillStyle = '#d8d8d6'; c.fillRect(0, h * 0.62, w, h * 0.004); // תפר בין המכלים
    vText(c, w, h, 'SPACEX', 0.0, 0.08, 0.45, '#111');
    vText(c, w, h, 'SPACEX', 0.5, 0.08, 0.45, '#111');
  }));
  g.add(cyl(R, R, 1.2, 0, darkGray()));                // אוקטאווב
  g.add(cyl(R, R, 6.2, 37.2, black()));                // תא בין שלבים מסיבי פחמן, שחור
  // 9 מנועי מרלין: 8 בטבעת + 1 במרכז
  g.add(nozzle(0.22, 0.46, 1.0, 0.05));
  ring(8, 1.25, (a, x, z) => { const n = nozzle(0.22, 0.46, 1.0, 0.05); n.position.set(x, 0, z); g.add(n); });
  // ארבע רגלי נחיתה מקופלות (סיבי פחמן) ושני "פינים" קטנים בבסיס
  ring(4, R + 0.12, (a, x, z) => {
    const leg = box(0.75, 9.5, 0.25, x, 1.0 + 4.75, z, black());
    leg.rotation.y = -a + Math.PI / 2;
    g.add(leg);
  });
  // ארבעה כנפוני סריג מטיטניום, מקופלים כלפי מעלה בראש התא
  ring(4, R + 0.1, (a, x, z) => {
    const f = gridFin(1.2, 1.5, 0.25, gray());
    f.position.set(Math.cos(a) * (R + 0.15), 42.4, Math.sin(a) * (R + 0.15));
    f.rotation.y = -a + Math.PI / 2;
    g.add(f);
  });
  // שלב שני
  g.add(cyl(R, R, 12.3, 43.4, white()));
  g.add(cyl(R, R, 0.4, 55.7, darkGray()));            // מתאם מטען
  // כיסוי מטען (Fairing): 5.2 מ' קוטר, 13.1 מ' אורך
  const fr = 2.6;
  g.add(lathe([[R, 56.1], [fr, 57.3]], white()));
  g.add(cyl(fr, fr, 6.0, 57.3, white()));
  g.add(lathe(ogive(fr, 6.5, 63.3, 24, 0.25), white()));
  g.add(cyl(fr * 1.001, fr * 1.001, 6.0, 57.3, mat('fairseam', { color: 0xcccccc, wireframe: true, transparent: true, opacity: 0 })));
  return g;
}

// Saturn V (תצורת אפולו 11) — 110.6 מ', S-IC 42.1 מ', S-II 24.8, S-IVB 17.9, IU 0.9, SLA 8.5, חללית ומגדל חילוץ
function saturnV() {
  const g = new THREE.Group();
  const R = 5.03;
  // S-IC: תבנית הגלגול השחור־לבן המפורסמת, כיתוב USA ודגל
  g.add(paintedCyl(R, 42.1, 0, (c, w, h) => {
    c.fillStyle = '#f2f2ef'; c.fillRect(0, 0, w, h);
    const band = (v0, v1, phase) => { c.fillStyle = '#121212'; for (let q = 0; q < 4; q++) if ((q + phase) % 2 === 0) c.fillRect(q * w / 4, v0 * h, w / 4, (v1 - v0) * h); };
    band(0.0, 0.07, 0);    // חצאית קדמית
    band(0.42, 0.52, 1);   // אזור בין המכלים
    band(0.82, 1.0, 0);    // חצאית אחורית (מבנה דחף)
    c.fillStyle = '#121212'; c.fillRect(0, 0.07 * h, w, 0.004 * h);
    vText(c, w, h, 'USA', 0.125 + 0.25, 0.55, 0.78, '#111');
    vText(c, w, h, 'UNITED STATES', 0.125 + 0.75, 0.1, 0.40, '#111');
    flag(c, (0.125 + 0.75) * w - 25, 0.43 * h, 50, 70);
  }));
  // מנועי F-1: 5 נחירים, קוטר יציאה 3.7 מ'
  g.add(nozzle(0.55, 1.85, 5.8, 0.6));
  ring(4, 3.2, (a, x, z) => { const n = nozzle(0.55, 1.85, 5.8, 0.6); n.position.set(x, 0, z); n.translateX(0); g.add(n); });
  // כיסויי מנוע חרוטיים וארבעה סנפירים
  ring(4, 3.6, (a, x, z) => {
    const fair = lathe([[2.0, -1.2], [1.6, 2], [0.8, 7]], white(), 24);
    fair.position.set(Math.cos(a) * 3.9, 0, Math.sin(a) * 3.9); g.add(fair);
    const sh = new THREE.Shape();
    sh.moveTo(0, -1.1); sh.lineTo(3.2, -1.1); sh.lineTo(3.2, 1.4); sh.lineTo(0, 6.8); sh.closePath();
    const fin = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.25, bevelEnabled: false }), white());
    fin.position.set(Math.cos(a) * (R + 0.6), 0, Math.sin(a) * (R + 0.6));
    fin.rotation.y = -a;
    fin.translateZ(-0.12);
    g.add(fin);
  }, 0);
  // S-II (כולל התא בין השלבים)
  g.add(paintedCyl(R, 24.8, 42.1, (c, w, h) => {
    c.fillStyle = '#f2f2ef'; c.fillRect(0, 0, w, h);
    c.fillStyle = '#121212'; c.fillRect(0, 0.0, w, 0.03 * h);
    c.fillStyle = '#cfcfcb'; c.fillRect(0, 0.78 * h, w, 0.006 * h);
  }));
  // תא בין S-II ל-S-IVB: חרוט קטום 10.06 → 6.6 מ'
  g.add(lathe([[R, 66.9], [3.3, 72.3]], white()));
  // S-IVB: לבן עם אזורי שחור בחלק העליון ושני מודולי APS
  g.add(paintedCyl(3.3, 12.5, 72.3, (c, w, h) => {
    c.fillStyle = '#f2f2ef'; c.fillRect(0, 0, w, h);
    c.fillStyle = '#121212'; c.fillRect(0, 0, w, 0.14 * h);
    for (let q = 0; q < 4; q++) if (q % 2 === 0) c.fillRect(q * w / 4, 0.14 * h, w / 4, 0.12 * h);
  }));
  ring(2, 3.35, (a, x, z) => { g.add(box(0.9, 1.5, 0.6, x, 73.3, z, white())); });
  g.add(cyl(3.3, 3.3, 0.9, 84.8, offWhite()));          // יחידת המכשור (IU)
  // SLA — מתאם החללית שמסתיר את רכב הנחיתה: 6.6 → 3.9 מ', 8.5 מ'
  g.add(lathe([[3.3, 85.7], [1.96, 94.2]], white()));
  // מודול שירות (SM): כסוף, 3.9 מ'
  g.add(cyl(1.96, 1.96, 3.9, 94.2, aluminum()));
  // מודול פיקוד (CM) מכוסה ב"כיסוי הגנת ההמראה" הלבן, ומגדל החילוץ
  g.add(lathe([[1.96, 98.1], [0.42, 101.3]], white()));
  // מגדל חילוץ (LES): מסבך + מנוע + חוד
  const truss = new THREE.Group();
  ring(4, 0.55, (a, x, z) => { const b = cyl(0.05, 0.05, 3.0, 101.2, mat('lesred', { color: 0xc23b22, roughness: 0.6 }), 6); b.position.x = x; b.position.z = z; truss.add(b); });
  g.add(truss);
  g.add(cyl(0.33, 0.33, 4.2, 104.2, white(), 24));
  g.add(cyl(0.33, 0.45, 0.6, 104.0, darkGray(), 24));
  g.add(lathe([[0.33, 108.4], [0.2, 109.6], [0.08, 110.4], [0.01, 110.6]], darkGray(), 24));
  return g;
}

// Soyuz-2.1b עם חללית סויוז MS — 46.3 מ': 4 מאיצים חרוטיים (19.6 מ'), ליבה (27.1 מ'), בלוק I, כיסוי ומגדל חילוץ
function soyuz() {
  const g = new THREE.Group();
  const rocketGray = mat('soyuzgray', { color: 0xc9cbc6, roughness: 0.6, metalness: 0.15 });
  // ליבה (בלוק A): הצרה בתחתית, רחבה למעלה
  g.add(lathe([[1.03, 0], [1.03, 1.0], [1.20, 6], [1.40, 14], [1.475, 19.5], [1.475, 27.1], [0.01, 27.1]], rocketGray));
  // ארבעה מאיצים (בלוקים B, V, G, D): חרוט שקצהו נוגע בליבה
  ring(4, 2.34, (a, x, z) => {
    const b = lathe([[1.34, 0], [1.34, 1.2], [1.05, 7], [0.75, 13.5], [0.42, 17.5], [0.15, 19.4], [0.01, 19.6]], rocketGray);
    b.position.set(x, 0, z);
    b.rotation.set(0, 0, 0);
    // הטיה קלה פנימה כך שהחוד נצמד לליבה
    const tilt = Math.atan2(2.34 - 1.5, 19.6) * 0.8;
    b.rotateOnWorldAxis(new THREE.Vector3(-Math.sin(a), 0, Math.cos(a)), tilt);
    g.add(b);
    // סנפיר אוויר קטן בבסיס המאיץ, מכוון החוצה
    const fin = box(0.06, 1.3, 1.0, Math.cos(a) * 3.85, 1.0, Math.sin(a) * 3.85, darkGray());
    fin.rotation.y = -a + Math.PI / 2;
    g.add(fin);
    // ארבעה נחירים לכל מאיץ (RD-107A)
    for (let k = 0; k < 4; k++) {
      const n = nozzle(0.12, 0.36, 0.9, 0.1);
      n.position.set(x + Math.cos(k * Math.PI / 2 + Math.PI / 4) * 0.55, 0, z + Math.sin(k * Math.PI / 2 + Math.PI / 4) * 0.55);
      g.add(n);
    }
  });
  for (let k = 0; k < 4; k++) { const n = nozzle(0.12, 0.36, 0.9, 0.1); n.position.set(Math.cos(k * Math.PI / 2 + Math.PI / 4) * 0.5, 0, Math.sin(k * Math.PI / 2 + Math.PI / 4) * 0.5); g.add(n); }
  // מסבך בין־שלבי פתוח (תכונה ייחודית לסויוז)
  for (let i = 0; i < 12; i++) {
    const a = i * Math.PI / 6;
    const s = box(0.08, 1.7, 0.08, Math.cos(a) * 1.4, 27.95, Math.sin(a) * 1.4, darkGray());
    s.rotation.z = (i % 2 ? 0.35 : -0.35);
    s.rotation.y = -a;
    g.add(s);
  }
  // בלוק I (שלב שלישי): 6.7 מ', קוטר 2.66
  g.add(cyl(1.33, 1.33, 6.7, 28.8, rocketGray));
  // כיסוי מטען לבן עם החללית בפנים, ומגדל חילוץ (SAS) עם כנפונים
  g.add(lathe([[1.33, 35.5], [1.5, 36.2], [1.5, 40.8], [1.25, 41.6], [0.62, 42.4], [0.40, 42.7]], white()));
  ring(4, 1.5, (a, x, z) => { const f = box(0.06, 1.5, 0.9, Math.cos(a) * 1.75, 39.4, Math.sin(a) * 1.75, white()); f.rotation.y = -a + Math.PI / 2; g.add(f); });
  g.add(cyl(0.18, 0.40, 1.6, 42.7, darkGray(), 24));
  g.add(cyl(0.18, 0.18, 1.6, 44.3, gray(), 16));
  g.add(lathe([[0.18, 45.9], [0.01, 46.3]], darkGray(), 16));
  return g;
}

// Long March 5 — 56.97 מ': ליבה 5 מ' (33.2 מ'), 4 מאיצים 3.35 מ' (27.6 מ'), שלב שני 11.5, כיסוי 5.2 מ'
function longMarch5() {
  const g = new THREE.Group();
  const cnRed = '#c8102e';
  g.add(paintedCyl(2.5, 33.2, 0, (c, w, h) => {
    c.fillStyle = '#f1f0ea'; c.fillRect(0, 0, w, h);
    vText(c, w, h, '中国航天', 0.0, 0.15, 0.45, cnRed);
    // דגל סין קטן
    c.fillStyle = '#de2910'; c.fillRect(w * 0.5 - 30, h * 0.18, 60, 90);
    c.fillStyle = '#ffde00'; c.beginPath(); c.arc(w * 0.5 - 12, h * 0.18 + 18, 8, 0, 7); c.fill();
  }));
  g.add(nozzle(0.3, 0.75, 1.6, 0.2)); const n2 = nozzle(0.3, 0.75, 1.6, 0.2); n2.position.x = 1.1; g.add(n2);
  g.children[g.children.length - 2].position.x = -1.1;
  ring(4, 2.5 + 1.675 + 0.05, (a, x, z) => {
    const b = new THREE.Group();
    b.add(paintedCyl(1.675, 22.6, 0, (c, w, h) => {
      c.fillStyle = '#f1f0ea'; c.fillRect(0, 0, w, h);
      c.fillStyle = cnRed; c.fillRect(0, 0.0, w, 0.015 * h);
    }));
    b.add(lathe([[1.675, 22.6], [1.2, 25.4], [0.5, 27.2], [0.05, 27.6]], white()));
    for (const s of [-0.75, 0.75]) { const n = nozzle(0.28, 0.62, 1.5, 0.2); n.position.x = s; b.add(n); }
    b.position.set(x, 0, z);
    b.rotation.y = -a;
    g.add(b);
  });
  g.add(cyl(2.5, 2.5, 11.5, 33.2, white()));
  // כיסוי מטען 5.2 מ'
  g.add(lathe([[2.5, 44.7], [2.6, 45.2]], white()));
  g.add(cyl(2.6, 2.6, 5.3, 45.2, white()));
  g.add(lathe(ogive(2.6, 6.47, 50.5, 24, 0.2), white()));
  return g;
}

// Starship V3 — 124.4 מ' (Super Heavy 72.3 מ' + ספינה 52.1 מ'), קוטר 9 מ', פלדת אל־חלד
function starship() {
  const g = new THREE.Group();
  const R = 4.5;
  // Super Heavy
  g.add(cyl(R, R, 66.0, 0, steel(), 64));
  // טבעת ההפרדה החמה המשולבת (V3): פתחי אוורור כהים
  g.add(paintedCyl(R, 6.3, 66.0, (c, w, h) => {
    c.fillStyle = '#b9bcc1'; c.fillRect(0, 0, w, h);
    c.fillStyle = '#16181b';
    for (let i = 0; i < 24; i++) c.fillRect(i * w / 24 + 8, h * 0.15, w / 24 - 16, h * 0.7);
  }));
  // 33 מנועי רפטור 3: 3 במרכז, 10 בטבעת פנימית, 20 בטבעת חיצונית
  const raptor = (x, z) => { const n = nozzle(0.3, 0.65, 1.6, 0.0, copper()); n.position.set(x, 0, z); g.add(n); };
  ring(3, 0.75, (a, x, z) => raptor(x, z));
  ring(10, 2.15, (a, x, z) => raptor(x, z));
  ring(20, 3.75, (a, x, z) => raptor(x, z));
  g.add(cyl(R, R, 0.25, -0.05, darkGray(), 64)); // מגן חום תחתון
  // שלושה כנפוני סריג גדולים (פריסה 90/90/180)
  for (const deg of [0, 90, 180]) {
    const a = deg * Math.PI / 180;
    const f = gridFin(4.2, 3.0, 0.5, darkGray());
    f.position.set(Math.cos(a) * (R + 0.35), 62.5, Math.sin(a) * (R + 0.35));
    f.rotation.y = -a + Math.PI / 2;
    g.add(f);
  }
  // צלעות (chines) לאורך המאיץ
  for (const s of [1, -1]) g.add(box(0.35, 50, 0.6, 0, 8 + 25, s * (R + 0.2), steel()));
  // הספינה
  const y0 = 72.3;
  g.add(cyl(R, R, 34.0, y0, steel(), 64));
  g.add(lathe(ogive(R, 18.1, y0 + 34.0, 28, 0.6), steel(), 64));
  // מגן חום: אריחים משושים שחורים על צד הרוח (חצי היקף + האף)
  const hsMat = tileMat();
  const hs = new THREE.Mesh(new THREE.CylinderGeometry(R + 0.03, R + 0.03, 34.0, 64, 1, true, Math.PI / 2, Math.PI), hsMat);
  hs.position.y = y0 + 17; g.add(hs);
  const nose = lathe(ogive(R + 0.03, 18.1, y0 + 34.0, 28, 0.62), hsMat, 64, Math.PI / 2, Math.PI);
  g.add(nose);
  // דשים: 2 אחוריים גדולים ו-2 קדמיים קטנים, בגבול האריחים
  const flap = (w, h, y, side, top) => {
    const sh = new THREE.Shape();
    sh.moveTo(0, 0); sh.lineTo(0, h); sh.lineTo(w * (top ? 0.55 : 0.9), h); sh.lineTo(w, h * (top ? 0.2 : 0.15)); sh.lineTo(w, 0); sh.closePath();
    const m = new THREE.Mesh(new THREE.ExtrudeGeometry(sh, { depth: 0.35, bevelEnabled: false }), steel());
    m.position.set(side * R, y, -0.18);
    if (side < 0) m.scale.x = -1;
    g.add(m);
    const t = m.clone(); t.material = hsMat; t.scale.z = 0.05; t.position.z = -0.24 - 0.02; g.add(t);
  };
  flap(3.6, 10.5, y0 + 0.5, 1, false); flap(3.6, 10.5, y0 + 0.5, -1, false);
  flap(2.0, 7.0, y0 + 38.5, 1, true); flap(2.0, 7.0, y0 + 38.5, -1, true);
  // 3 רפטור ים + 3 רפטור ואקום
  ring(3, 1.05, (a, x, z) => { const n = nozzle(0.3, 0.65, 1.5, y0 + 0.6, copper()); n.position.x = x; n.position.z = z; g.add(n); });
  ring(3, 3.05, (a, x, z) => { const n = nozzle(0.35, 1.18, 3.0, y0 + 1.6, copper()); n.position.x = x; n.position.z = z; g.add(n); }, 0);
  // פתח פריסת סטארלינק ("פז")
  g.add(box(6.0, 0.6, 0.1, 0, y0 + 24, R + 0.02, darkGray()));
  return g;
}

// ================= חלליות ולוויינים =================

// ספוטניק 1 — כדור אלומיניום מלוטש בקוטר 58 ס"מ, 4 אנטנות שוט (2.4 ו-2.9 מ'), 83.6 ק"ג
function sputnik() {
  const g = new THREE.Group();
  g.add(sphere(0.29, aluminum(), 48));
  const seam = new THREE.Mesh(new THREE.TorusGeometry(0.29, 0.006, 8, 64), gray());
  seam.rotation.x = Math.PI / 2; g.add(seam);
  const antMat = mat('ant', { color: 0xbfc3c8, metalness: 1, roughness: 0.3 });
  const lens = [2.4, 2.4, 2.9, 2.9];
  [[0.35, 0.35], [-0.35, 0.35], [0.35, -0.35], [-0.35, -0.35]].forEach(([dx, dz], i) => {
    const L = lens[i];
    const a = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.01, L, 6), antMat);
    // האנטנות נטויות לאחור בזווית של כ-35° מהציר
    const dir = new THREE.Vector3(dx, -1.0, dz).normalize();
    a.position.copy(new THREE.Vector3(dx * 0.3, -0.05, dz * 0.3).add(dir.clone().multiplyScalar(L / 2)));
    a.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    g.add(a);
  });
  return g;
}

// Starlink V2 Mini — גוף שטוח 4.1×2.7 מ', שתי כנפיים סולאריות בפריסה של כ-30 מ', כ-800 ק"ג
function starlinkV2mini() {
  const g = new THREE.Group();
  g.add(box(4.1, 0.25, 2.7, 0, 0, 0, darkGray()));
  // ארבעה אנטנות מערך מופע (phased array) בצד הפונה לכדור הארץ
  for (const [x, z] of [[-1.0, -0.65], [1.0, -0.65], [-1.0, 0.65], [1.0, 0.65]]) g.add(box(1.7, 0.04, 1.1, x, -0.145, z, mat('pa', { color: 0xe8e6df, roughness: 0.8 })));
  // שתי כנפיים סולאריות על זרועות
  for (const s of [-1, 1]) {
    g.add(box(1.0, 0.06, 0.06, s * 2.55, 0.15, 0, gray()));
    const p = solarPanel(12.4, 4.1);
    p.rotation.x = -Math.PI / 2;
    p.position.set(s * (3.05 + 6.2), 0.15, 0);
    g.add(p);
  }
  return g;
}

// לוויין תקשורת גאוסטציונרי טיפוסי (פלטפורמה מסוג Boeing 702 / Eurostar): גוף ~3.5 מ', כנפיים כ-40 מ', 2 מחזירי אנטנה
function geoComsat() {
  const g = new THREE.Group();
  g.add(box(2.6, 3.6, 2.4, 0, 0, 0, gold()));
  g.add(box(2.62, 3.0, 2.42, 0, 0, 0, radiatorMat())); // רדיאטורים בצפון ובדרום
  for (const s of [-1, 1]) {
    // מחזירי אנטנה פרבוליים בקוטר 2.3 מ' במזרח ובמערב
    const dish = new THREE.Mesh(new THREE.SphereGeometry(2.0, 32, 12, 0, Math.PI * 2, 0, 0.6), mat('dish', { color: 0xf2f0e8, roughness: 0.6, side: THREE.DoubleSide }));
    dish.rotation.z = s * Math.PI / 2;
    dish.position.set(s * 2.8, 0.6, 0);
    dish.rotation.y = 0.4 * s;
    g.add(dish);
    g.add(box(0.1, 0.1, 0.1, s * 1.4, 0.6, 0, gray()));
    // כנף סולארית: 4 פאנלים של 2.3×5 מ' על זרוע
    g.add(box(0.08, 0.08, 3.0, 0, 0, s * 2.7, gray()));
    for (let k = 0; k < 4; k++) {
      const p = solarPanel(2.4, 4.6);
      p.position.set(0, 0, s * (4.4 + k * 4.75));
      p.rotation.y = Math.PI / 2;
      p.rotation.order = 'YXZ';
      p.rotation.x = 0;
      const q = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 4.6), p.material);
      q.rotation.x = -Math.PI / 2; q.rotation.z = Math.PI / 2;
      q.position.set(0, 0, s * (4.6 + k * 4.75));
      g.add(q);
    }
  }
  g.add(cyl(0.35, 0.15, 0.6, 1.8, gray(), 16)); // אנטנת תקשורת עליונה
  return g;
}

// תחנת החלל הבינלאומית — מסבך 109 מ', 8 כנפיים סולאריות (35×12 מ'), מודולים לאורך כ-73 מ'
function iss() {
  const g = new THREE.Group();
  const trussMat = mat('truss', { color: 0xbab8b0, roughness: 0.6, metalness: 0.4 });
  g.add(box(94, 2.6, 2.6, 0, 0, 0, trussMat)); // המסבך המשולב
  // מודולים לחוצים לאורך ציר Z (Zvezda–Zarya–Unity–Destiny–Harmony)
  const mod = (len, r, z, m = offWhite()) => { const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 24), m); c.rotation.x = Math.PI / 2; c.position.set(0, -4, z); g.add(c); };
  mod(13.1, 2.1, -32);   // זבזדה
  mod(12.6, 2.05, -19.5); // זריה
  mod(5.5, 2.25, -10.5);  // יוניטי
  mod(8.5, 2.15, -3.5);   // דסטיני
  mod(7.2, 2.2, 4.5);     // הרמוני
  g.add(box(1, 4, 1, 0, -2, -3.5, trussMat));
  // קולומבוס וקיבו בצידי הרמוני
  const side = (len, x) => { const c = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, len, 24), offWhite()); c.rotation.z = Math.PI / 2; c.position.set(x, -4, 4.5); g.add(c); };
  side(6.9, -5.6); side(11.2, 7.8);
  // כנפי זבזדה הקטנות
  for (const s of [-1, 1]) { const p = solarPanel(13, 3.4); p.rotation.x = -Math.PI / 2; p.position.set(s * 8.5, -4, -34); g.add(p); }
  // 4 זוגות כנפיים סולאריות גדולות: P4, P6, S4, S6 — לכל אחת 2 שמיכות (35×12 מ' לכנף)
  const wingPos = [-41, -30, 30, 41];
  for (const x of wingPos) {
    for (const s of [-1, 1]) {
      const p = solarPanel(11.6, 34.7, solarGold());
      p.rotation.x = 0;
      p.position.set(x, s * (34.7 / 2 + 1.5), 0);
      g.add(p);
      g.add(box(0.3, 34.7, 0.3, x, s * (34.7 / 2 + 1.5), 0.05, trussMat));
    }
  }
  // רדיאטורים לבנים
  for (const x of [-13, 13]) for (const k of [0, 1, 2]) {
    const r = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 13.6), radiatorMat());
    r.rotation.y = Math.PI / 2; r.rotation.z = Math.PI / 2;
    r.position.set(x + k * 3.6 * Math.sign(x), -9, 0);
    g.add(r);
  }
  return g;
}

// טלסקופ החלל האבל — 13.2 מ' × 4.2 מ', שני מערכים סולאריים של 7.1×2.6 מ'
function hubble() {
  const g = new THREE.Group();
  const body = new THREE.Group();
  body.add(cyl(2.1, 2.1, 6.5, 0, silverFoil()));        // חלק אחורי (מכשירים)
  body.add(cyl(1.55, 2.1, 0.5, 6.5, silverFoil()));
  body.add(cyl(1.55, 1.55, 6.0, 7.0, silverFoil()));     // מגן האור
  // מכסה הצמצם הפתוח
  const door = new THREE.Mesh(new THREE.CircleGeometry(1.6, 32), mat('door', { color: 0xd8d8d8, metalness: 0.9, roughness: 0.3, side: THREE.DoubleSide }));
  door.position.set(0, 13.0 + 1.2, -1.6); door.rotation.x = -0.4; body.add(door);
  g.add(body);
  for (const s of [-1, 1]) {
    const p = solarPanel(2.6, 7.1, solarMat());
    p.position.set(s * (2.1 + 0.4 + 3.55), 4.0, 0);
    p.rotation.z = Math.PI / 2; p.rotation.y = 0.0;
    g.add(p);
    g.add(box(0.8, 0.1, 0.1, s * 2.3, 4.0, 0, gray()));
    // אנטנות שבח גבוה על זרועות
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.65, 16, 8, 0, Math.PI * 2, 0, 0.7), mat('hga', { color: 0xeeeeee, side: THREE.DoubleSide }));
    d.position.set(0, 7.5, s * 2.9); d.rotation.x = s * Math.PI / 2; g.add(d);
    g.add(box(0.06, 0.06, 0.9, 0, 7.5, s * 2.4, gray()));
  }
  g.rotation.z = Math.PI / 2;
  const w = new THREE.Group(); w.add(g); return w;
}

// תחנת החלל הסינית טיאנגונג — טיאנחה 16.6 מ', וונטיאן ומנגטיאן 17.9 מ' כל אחד, קוטר 4.2 מ', צורת T
function tiangong() {
  const g = new THREE.Group();
  const m = offWhite();
  const cylX = (len, r, x, z) => { const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 24), m); c.rotation.z = Math.PI / 2; c.position.set(x, 0, z); g.add(c); };
  const cylZ = (len, r, x, z) => { const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 24), m); c.rotation.x = Math.PI / 2; c.position.set(x, 0, z); g.add(c); };
  // טיאנחה לאורך Z, עם צומת העגינה הכדורי בקדמתו
  cylZ(16.6 - 2.8, 2.1, 0, -6.9);
  const node = sphere(1.4, m); node.position.set(0, 0, 1.4); g.add(node);
  cylZ(9.5, 1.4, 0, -9.0); // החלק הצר של טיאנחה
  // וונטיאן ומנגטיאן לרוחב, מחוברים לצומת
  cylX(17.9, 2.1, -(1.4 + 8.95), 1.4);
  cylX(17.9, 2.1, (1.4 + 8.95), 1.4);
  // כנפיים סולאריות גדולות בקצות מודולי המעבדה (כ-27 מ' כל כנף)
  for (const s of [-1, 1]) for (const t of [-1, 1]) {
    const p = solarPanel(4.2, 27, solarMat());
    p.position.set(s * 16.5, t * (13.5 + 2.3), 1.4);
    g.add(p);
  }
  // כנפיים של טיאנחה
  for (const t of [-1, 1]) { const p = solarPanel(3.2, 13, solarMat()); p.position.set(0, t * (6.5 + 2.1), -6.5); p.rotation.y = Math.PI / 2; g.add(p); }
  return g;
}

// ווסטוק 1 — כדור הנחיתה בקוטר 2.3 מ' + מודול מכשירים (שני חרוטים), אורך כולל 4.4 מ', 4,725 ק"ג
function vostok() {
  const g = new THREE.Group();
  const s = sphere(1.15, mat('vostokgray', { color: 0x9aa3a8, roughness: 0.5, metalness: 0.6 }));
  s.position.y = 2.9; g.add(s);
  g.add(lathe([[0.4, 1.9], [1.2, 1.2], [1.25, 0.9], [0.95, 0.2], [0.5, 0.0]], gray()));
  // 16 כדורי חנקן ואוקסיגן סביב המודול
  ring(16, 1.25, (a, x, z) => { const b = sphere(0.18, darkGray(), 12); b.position.set(x, 1.15, z); g.add(b); });
  const ant = cyl(0.015, 0.015, 1.8, 3.9, gray(), 6); g.add(ant);
  return g;
}

// אפולו CSM + LM — מודול פיקוד (3.9×3.2 מ'), מודול שירות (3.9×7.5 מ' כולל נחיר), רכב נחיתה (4.2 מ' רוחב ללא רגליים)
function apolloCSM() {
  const g = new THREE.Group();
  g.add(lathe([[1.95, 7.5], [1.95, 7.6], [0.42, 10.7], [0.3, 10.9], [0.01, 10.9]], aluminum()));
  g.add(cyl(1.95, 1.95, 4.0, 3.5, aluminum()));
  g.add(nozzle(0.4, 1.25, 3.5, 3.5, mat('spsnozzle', { color: 0x5a5a5a, metalness: 0.7, roughness: 0.4, side: THREE.DoubleSide })));
  g.add(box(0.5, 0.5, 0.5, 1.95, 7.0, 0, gray()));
  return g;
}
function apolloLM() {
  const g = new THREE.Group();
  // שלב ירידה: מתומן עטוף בנייר זהב, רגליים פרושות
  const desc = new THREE.Mesh(new THREE.CylinderGeometry(2.1, 2.1, 1.65, 8), gold());
  desc.position.y = 1.9; g.add(desc);
  ring(4, 2.1, (a, x, z) => {
    const leg = cyl(0.07, 0.07, 3.0, 0, gold(), 8);
    leg.position.set(x * 1.35, 1.2, z * 1.35); leg.rotation.set(Math.sin(a) * 0.45, 0, -Math.cos(a) * 0.45);
    g.add(leg);
    const pad = cyl(0.45, 0.45, 0.12, 0, gray(), 16); pad.position.set(x * 1.9, 0.0, z * 1.9); g.add(pad);
  });
  // שלב עלייה
  const asc = new THREE.Mesh(new THREE.DodecahedronGeometry(1.7, 0), mat('lmasc', { color: 0xb8b8b4, metalness: 0.5, roughness: 0.5, flatShading: true }));
  asc.position.y = 3.9; asc.scale.set(1.15, 0.85, 1.0); g.add(asc);
  g.add(box(0.6, 0.6, 0.1, -0.7, 4.0, 1.5, black()));
  g.add(box(0.6, 0.6, 0.1, 0.7, 4.0, 1.5, black()));
  return g;
}

// Crew Dragon — קפסולה 4.4 מ' + טאנק (trunk) עם תאים סולאריים; גובה 8.1 מ', קוטר 4 מ'
function crewDragon() {
  const g = new THREE.Group();
  g.add(paintedCyl(1.85, 3.7, 0, (c, w, h) => {
    c.fillStyle = '#f2f2f0'; c.fillRect(0, 0, w, h);
    c.fillStyle = '#16213f'; c.fillRect(0, 0, w / 2, h); // חצי מכוסה תאים סולאריים
  }));
  ring(4, 1.85, (a, x, z) => { if (Math.abs(Math.sin(a * 2)) < 0.1) { const f = box(0.08, 1.4, 0.8, x * 1.15, 0.8, z * 1.15, white()); f.rotation.y = -a + Math.PI / 2; g.add(f); } });
  g.add(cyl(2.0, 2.0, 0.4, 3.7, black()));              // מגן חום
  g.add(lathe([[2.0, 4.1], [1.1, 7.0], [0.85, 7.3]], white()));
  g.add(lathe([[0.85, 7.3], [0.7, 7.8], [0.3, 8.1], [0.01, 8.15]], white()));
  // חלונות קטנים ופס שחור של מנועי SuperDraco
  ring(4, 1.75, (a, x, z) => { const p = box(0.5, 0.6, 0.05, x * 0.95, 5.0, z * 0.95, black()); p.rotation.y = -a + Math.PI / 2; g.add(p); });
  return g;
}

// שנז'ואו — אורך 9.25 מ': מודול מסלול (2.8 מ', ⌀2.25), מודול חזרה (2.5 מ', ⌀2.52), מודול שירות (2.94 מ', ⌀2.5–2.8), כנפיים 17 מ'
function shenzhou() {
  const g = new THREE.Group();
  const sz = mat('szwhite', { color: 0xe7e4dc, roughness: 0.6 });
  g.add(cyl(1.25, 1.4, 2.94, 0, sz));                         // מודול שירות
  for (const s of [-1, 1]) { const p = solarPanel(2.0, 7.0); p.position.set(s * (1.4 + 3.6), 1.5, 0); p.rotation.z = Math.PI / 2; g.add(p); }
  g.add(lathe([[1.26, 2.94], [1.26, 3.2], [0.85, 5.4], [0.6, 5.44]], mat('szdesc', { color: 0x8e8a7c, roughness: 0.5, metalness: 0.4 }))); // מודול חזרה (פעמון)
  g.add(lathe([[0.6, 5.44], [1.125, 5.9], [1.125, 7.9], [0.6, 8.24]], sz));            // מודול מסלול
  for (const s of [-1, 1]) { const p = solarPanel(1.0, 2.2); p.position.set(s * (1.125 + 1.2), 7.0, 0); p.rotation.z = Math.PI / 2; g.add(p); }
  g.add(cyl(0.45, 0.45, 1.0, 8.24, gray(), 16));
  return g;
}

// דמות אדם לסקאלה, 1.8 מ'
function human() {
  const g = new THREE.Group();
  const m = mat('human', { color: 0xff7a3c, roughness: 0.7 });
  g.add(cyl(0.18, 0.2, 0.85, 0.0, m, 12));
  g.add(cyl(0.22, 0.2, 0.65, 0.85, m, 12));
  const h = sphere(0.12, m, 12); h.position.y = 1.65; g.add(h);
  return g;
}

export const BUILDERS = { falcon9, saturn5: saturnV, soyuz, cz5: longMarch5, starship, sputnik, starlink: starlinkV2mini, geo: geoComsat, iss, hubble, tiangong, vostok, apolloCSM, apolloLM, dragon: crewDragon, shenzhou, human };

const cache = {};
export function buildModel(id) {
  if (!cache[id]) cache[id] = BUILDERS[id]();
  return cache[id].clone();
}

// גובה הרקטה (למיקום מצלמה ולפלומה)
export const ROCKET_HEIGHTS = { falcon9: 69.8, saturn5: 110.6, soyuz: 46.3, cz5: 56.97, starship: 124.4 };
