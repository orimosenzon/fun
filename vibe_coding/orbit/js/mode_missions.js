// mode_missions.js — לשונית "משימות": ציר זמן היסטורי עם סצנה תלת־ממדית לכל משימה
import * as THREE from 'three';
import * as P from './physics.js';
import { MISSIONS } from './data.js';
import { LUNAR } from './lunar_data.js';
import { buildModel } from './models.js';
import { RE_KM, eciToThree, ecefDir, makeLine } from './world.js';
import { $, $$, fmt, fmtDateHe, fmtDur, h } from './util.js';

const FILTERS = [
  ['all', 'הכל'], ['us', 'ארה"ב'], ['ussr', 'ברית המועצות'], ['cn', 'סין'], ['spacex', 'SpaceX'], ['il', 'ישראל'],
];
const matchFilter = (m, f) => f === 'all' ||
  (f === 'us' && (m.flag === '🇺🇸')) || (f === 'ussr' && m.flag === '☭') || (f === 'cn' && m.flag === '🇨🇳') ||
  (f === 'spacex' && /SpaceX|Falcon|Starship|Dragon/.test(m.country + m.rocket)) || (f === 'il' && m.flag === '🇮🇱');

export const missionsMode = {
  enter(world, panel) {
    this.world = world;
    this.panel = panel;
    this.filter = this.filter ?? 'all';
    this.group = new THREE.Group(); world.scene.add(this.group);
    this.labels = [];
    const id = location.hash.split('/')[1];
    if (id && MISSIONS.find(m => m.id === id)) this.open(id);
    else this.list();
  },

  list() {
    const { panel, world } = this;
    this.clearScene();
    world.clock.time = Date.now();
    world.setWarps([0, 1, 60, 600], 60);
    world.moonOverride = false;
    world.controls.target.set(0, 0, 0);
    world.camera.position.set(0, 9000, 30000);
    history.replaceState(null, '', '#missions');
    const items = MISSIONS.filter(m => matchFilter(m, this.filter));
    panel.innerHTML = `
      <h2>משימות</h2>
      <p class="lead">מספוטניק ב-1957 ועד הטיסה המסלולית הראשונה של סטארשיפ בספטמבר 2026. בחרו משימה כדי לראות את המסלול שלה בתלת־ממד, בתאריך האמיתי.</p>
      <div class="filters">${FILTERS.map(([k, v]) => `<button class="chip ${k === this.filter ? 'on' : ''}" data-f="${k}">${v}</button>`).join('')}</div>
      <div class="mission-list">${items.map(m => `
        <button class="mission" data-id="${m.id}"><span class="fl">${m.flag}</span><span><b>${m.title}</b><span>${m.subtitle}</span></span><span class="yr">${m.date.slice(0, 4)}</span></button>`).join('')}</div>`;
    $('.filters', panel).onclick = e => { const b = e.target.closest('button'); if (b) { this.filter = b.dataset.f; this.list(); } };
    $('.mission-list', panel).onclick = e => { const b = e.target.closest('button'); if (b) this.open(b.dataset.id); };
  },

  open(id) {
    const { panel, world } = this;
    const m = MISSIONS.find(x => x.id === id);
    this.m = m;
    history.replaceState(null, '', '#missions/' + id);
    const d = new Date(m.date);
    panel.innerHTML = `
      <button class="back" id="back">→ כל המשימות</button>
      <h2>${m.flag} ${m.title}</h2>
      <p class="lead">${m.subtitle}</p>
      <dl class="facts">
        <dt>תאריך</dt><dd>${fmtDateHe(d)}</dd>
        <dt>רקטה</dt><dd>${m.rocket}</dd>
        <dt>אתר שיגור</dt><dd>${m.site.name}</dd>
        ${m.crew ? `<dt>צוות</dt><dd>${m.crew}</dd>` : ''}
        ${m.facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}
      </dl>
      ${m.launchPreset ? `<button class="btn" id="sim">🚀 לסמלץ את השיגור הזה</button>` : ''}
      ${m.text.map(p => `<p>${p}</p>`).join('')}
      ${m.timeline ? `<h3>ציר הזמן</h3><div class="timeline">${m.timeline.map(([t, x]) => `<div><span class="t">${t}</span> ${x}</div>`).join('')}</div>` : ''}
      <div id="scene-info" class="note"></div>
      <div class="row" id="camrow"></div>
    `;
    $('#back', panel).onclick = () => this.list();
    if (m.launchPreset) $('#sim', panel).onclick = () => {
      world.MODES_launch_preset = m.launchPreset;
      import('./mode_launch.js').then(({ launchMode }) => { launchMode.pendingPreset = m.launchPreset; world.setMode('launch'); });
    };
    this.buildScene(m);
  },

  clearScene() {
    const w = this.world;
    this.group.clear();
    this.labels.forEach(L => w.removeLabel(L));
    this.labels = [];
    this.anim = null;
    if (this.siteMarker) { w.earth.remove(this.siteMarker); this.siteMarker = null; }
    w.moonOverride = false;
    w.camera.up.set(0, 1, 0);
  },

  addSite(m) {
    const w = this.world;
    const d = ecefDir(m.site.lat, m.site.lon);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(40, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff6b6b }));
    dot.position.set(d[0] * RE_KM, d[2] * RE_KM, -d[1] * RE_KM);
    w.earth.add(dot);
    this.siteMarker = dot;
    const L = w.addLabel(m.site.name, 'site'); L.obj = dot; this.labels.push(L);
  },

  buildScene(m) {
    const w = this.world;
    this.clearScene();
    const sc = m.scene;
    const t0 = new Date(m.date).getTime();
    w.clock.time = t0;
    w.updateCelestial();
    this.addSite(m);
    const info = $('#scene-info', this.panel);
    if (sc.type === 'orbit') this.sceneOrbit(m, t0, info);
    else if (sc.type === 'lunar') this.sceneLunar(m, t0, info);
    else if (sc.type === 'suborbital') this.sceneSuborbital(m, t0, info);
    else {
      w.setWarps([0, 1, 60, 600], 60);
      w.controls.target.set(0, 0, 0);
      const s = eciToThree(P.moonPosECI(new Date(t0))).multiplyScalar(0.001);
      w.camera.position.copy(s.clone().multiplyScalar(0.5)).add(new THREE.Vector3(0, 150000, 0)).add(s.clone().cross(new THREE.Vector3(0, 1, 0)).normalize().multiplyScalar(150000));
      const L = w.addLabel('הירח', 'dim'); L.obj = w.moon; this.labels.push(L);
      info.textContent = 'כדור הארץ והירח במיקומם האמיתי בתאריך השיגור, בקנה מידה אמיתי.';
    }
  },

  // מסלול קפלרי: הצומת העולה מחושב כך שהמסלול עובר מעל אתר השיגור בזמן השיגור
  sceneOrbit(m, t0, info) {
    const w = this.world, sc = m.scene;
    const R = P.R_EARTH;
    const a = R + (sc.peri + sc.apo) / 2 * 1000;
    const e = (sc.apo - sc.peri) * 1000 / (2 * a);
    const inc = sc.inc * Math.PI / 180;
    const phi = m.site.lat * Math.PI / 180;
    const alpha = P.gmst(new Date(t0)) + m.site.lon * Math.PI / 180;
    const u0 = Math.asin(Math.max(-1, Math.min(1, Math.sin(phi) / Math.sin(inc))));
    const raan = alpha - Math.atan2(Math.cos(inc) * Math.sin(u0), Math.cos(u0));
    // הכניסה למסלול כ-10 דקות ו-~35° אחרי השיגור, בנקודת הפריגיאה
    const argp = u0 + 0.6;
    const tIns = t0 + 600e3;
    const T = P.periodOf(a);
    const el = { a, e, inc, raan, argp, tIns, T };
    this.anim = { type: 'orbit', el };
    const pts = [];
    for (let k = 0; k <= 360; k++) pts.push(eciToThree(P.keplerToECI(a, e, inc, raan, argp, k * Math.PI / 180)).multiplyScalar(0.001));
    this.group.add(makeLine(pts, sc.retro ? 0xf472b6 : 0x5ac8ff, 0.9));
    // מודל או נקודה
    if (sc.model) {
      this.sat = buildModel(sc.model);
      this.satSize = new THREE.Box3().setFromObject(this.sat).getSize(new THREE.Vector3()).length();
    } else {
      this.sat = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      this.satSize = 60;
    }
    this.group.add(this.sat);
    this.satLabel = w.addLabel(m.title, 'leo big'); this.labels.push(this.satLabel);
    w.setWarps([0, 1, 10, 60, 300, 1800], 60);
    w.clock.time = tIns;
    // מצלמה מעל אתר השיגור
    const p = eciToThree(P.keplerToECI(a, e, inc, raan, argp, 0)).multiplyScalar(0.001);
    w.controls.target.set(0, 0, 0);
    w.camera.position.copy(p.clone().normalize().multiplyScalar(RE_KM * 3.6)).add(new THREE.Vector3(0, RE_KM * 0.8, 0));
    info.innerHTML = `מסלול ${fmt(sc.peri)} × ${fmt(sc.apo)} ק"מ בנטייה ${fmt(sc.inc, 2)}°, זמן הקפה ${fmtDur(T)}. המישור מחושב מאתר השיגור ומשעת השיגור, ותאורת השמש היא של אותו רגע.${sc.retro ? ' המסלול רטרוגרדי: הלוויין נע מערבה, נגד סיבוב כדור הארץ.' : ''}`;
  },

  sceneLunar(m, t0, info) {
    const w = this.world, sc = m.scene;
    const L = { ...LUNAR[sc.key], W: LUNAR.W };
    // t=0 בנתונים = הבערת ההזרקה לירח; משגרים ומשייטים במסלול חניה לפני כן
    const tTLI = t0 + (sc.key === 'apollo11' ? 2.73 : 3) * 3600e3;
    const p0 = P.moonPosECI(new Date(tTLI + L.tof * 1000));
    const p1 = P.moonPosECI(new Date(tTLI + L.tof * 1000 + 6 * 3600e3));
    const n = new THREE.Vector3().crossVectors(eciToThree(p0), eciToThree(p1)).normalize();
    // מכוונים כך שהירח של המודל יהיה בכיוון האמיתי שלו בזמן ההגעה
    const exArr = eciToThree(p0).normalize();
    const ang = L.W * L.tof;
    const ex = exArr.clone().applyAxisAngle(n, -ang);
    const ey = new THREE.Vector3().crossVectors(n, ex);
    const map = (x, y) => ex.clone().multiplyScalar(x).addScaledVector(ey, y);
    const segs = [['out', 0xfbbf24], ['lunar', 0x5ac8ff], ['back', 0x34d399]].filter(([k]) => L[k]);
    const all = [];
    for (const [k, col] of segs) {
      const pts = L[k].map(([t, x, y]) => map(x, y));
      // מסלולי הירח נשמרו במערכת אינרציאלית: הם נעים יחד עם הירח, ולכן מציירים אותם יחסית אליו
      if (k === 'lunar') {
        const rel = L[k].map(([t, x, y]) => { const mx = 384400 * Math.cos(L.W * t), my = 384400 * Math.sin(L.W * t); return [x - mx, y - my]; });
        this.lunarRel = { pts: rel, t: L[k].map(p => p[0]) };
        const line = makeLine(rel.map(([x, y]) => map(x, y)), col, 0.95);
        this.lunarLine = line; this.group.add(line);
      } else this.group.add(makeLine(pts, col, 0.9));
      L[k].forEach(p => all.push(p));
    }
    all.sort((a, b) => a[0] - b[0]);
    this.anim = { type: 'lunar', all, map, W: L.W, tTLI };
    w.moonOverride = true;
    this.craft = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.group.add(this.craft);
    this.satLabel = w.addLabel(m.title, 'leo big'); this.labels.push(this.satLabel);
    const Lm = w.addLabel('הירח', 'dim'); Lm.obj = w.moon; this.labels.push(Lm);
    w.setWarps([0, 600, 3600, 3 * 3600, 6 * 3600], 3 * 3600);
    w.clock.time = tTLI;
    w.timeDisplay = () => {
      const s = (w.clock.time - t0) / 1000;
      const d = Math.floor(s / 86400), hh = Math.floor(s % 86400 / 3600), mm = Math.floor(s % 3600 / 60);
      return `${new Date(w.clock.time).toISOString().slice(0, 10)} · יום ${d}, ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')} מהשיגור`;
    };
    // מצלמה מעל מישור המסלול
    const overview = () => { w.controls.target.copy(map(200000, 0)); w.camera.position.copy(map(200000, -330000)).addScaledVector(n, 330000); };
    overview();
    const row = $('#camrow', this.panel);
    row.innerHTML = `<button class="btn secondary" id="cAll">מבט על המסלול</button><button class="btn secondary" id="cMoon">הירח מקרוב</button><button class="btn secondary" id="cEarth">כדור הארץ</button><button class="btn secondary" id="cRestart">מההתחלה</button>`;
    $('#cAll', row).onclick = () => { overview(); this.followMoon = false; };
    $('#cMoon', row).onclick = () => { this.followMoon = true; w.camera.position.copy(w.moon.position).addScaledVector(n, 16000).add(ey.clone().multiplyScalar(-9000)); w.controls.target.copy(w.moon.position); };
    $('#cEarth', row).onclick = () => { this.followMoon = false; w.controls.target.set(0, 0, 0); w.camera.position.copy(n.clone().multiplyScalar(60000)).add(ey.clone().multiplyScalar(-30000)); };
    $('#cRestart', row).onclick = () => { w.clock.time = tTLI; };
    const parts = [`ההבערה לירח: ${fmt(L.tliDv)} מ׳/שנ׳`, `זמן ההגעה: ${fmt(L.tof / 3600, 1)} שעות`];
    if (L.loiDv) parts.push(`בלימה למסלול ירחי: ${fmt(L.loiDv)} מ׳/שנ׳`, `חזרה: ${fmt(L.teiDv)} מ׳/שנ׳`);
    info.innerHTML = `מסלול מחושב באינטגרציה נומרית של כדור הארץ, הירח והחללית (מודל תלת־גופי מישורי). ${parts.join(' · ')}. <span style="color:#fbbf24">צהוב</span>: בדרך לירח${L.lunar ? ', <span style="color:#5ac8ff">תכלת</span>: הקפות סביב הירח, <span style="color:#34d399">ירוק</span>: חזרה' : ''}.`;
  },

  sceneSuborbital(m, t0, info) {
    const w = this.world;
    // המחשה איכותית של מסלול "כמעט מסלולי": עלייה, שיוט וירידה לאוקיינוס ההודי
    const phi = m.site.lat * Math.PI / 180;
    const lam = P.gmst(new Date(t0)) + m.site.lon * Math.PI / 180;
    const s = [Math.cos(phi) * Math.cos(lam), Math.cos(phi) * Math.sin(lam), Math.sin(phi)];
    const east = [-Math.sin(lam), Math.cos(lam), 0];
    const north = [-Math.sin(phi) * Math.cos(lam), -Math.sin(phi) * Math.sin(lam), Math.cos(phi)];
    const A = 97 * Math.PI / 180;
    const d = east.map((e, i) => Math.sin(A) * e + Math.cos(A) * north[i]);
    const span = 200 * Math.PI / 180;
    const pts = [];
    for (let k = 0; k <= 400; k++) {
      const f = k / 400, th = f * span;
      const alt = 190 * Math.min(1, Math.sin(Math.min(1, f / 0.12) * Math.PI / 2)) * Math.min(1, (1 - f) / 0.1);
      const r = RE_KM + Math.max(0, alt);
      pts.push(eciToThree([r * (Math.cos(th) * s[0] + Math.sin(th) * d[0]), r * (Math.cos(th) * s[1] + Math.sin(th) * d[1]), r * (Math.cos(th) * s[2] + Math.sin(th) * d[2])]));
    }
    this.group.add(makeLine(pts, 0xfbbf24, 0.9));
    this.anim = { type: 'path', pts, dur: 65 * 60, t0 };
    this.sat = buildModel('starship');
    this.satSize = 124;
    this.group.add(this.sat);
    this.satLabel = w.addLabel(m.title, 'leo big'); this.labels.push(this.satLabel);
    w.setWarps([0, 10, 60, 300], 60);
    w.controls.target.set(0, 0, 0);
    w.camera.position.copy(pts[200]).normalize().multiplyScalar(RE_KM * 3.4).add(new THREE.Vector3(0, RE_KM, 0));
    info.textContent = 'המחשה: מסלול תת־מסלולי מסטארבייס עד האוקיינוס ההודי, כשעה וחמש דקות. הצורה איכותית ולא מבוססת על נתוני טלמטריה.';
  },

  placeSat(world, pos, next) {
    const s = this.sat;
    if (!s) return;
    s.position.copy(pos);
    const y = pos.clone().normalize();
    const vel = next.clone().sub(pos).normalize();
    const z = new THREE.Vector3().crossVectors(vel, y).normalize();
    const x = new THREE.Vector3().crossVectors(y, z);
    s.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
    const dist = world.camera.position.distanceTo(pos);
    s.scale.setScalar(Math.max(0.001, dist * 0.05 / this.satSize));
    this.satLabel.pos.copy(pos);
  },

  update(world) {
    const A = this.anim;
    if (!A) return;
    const t = world.clock.time;
    if (A.type === 'orbit') {
      const { a, e, inc, raan, argp, tIns, T } = A.el;
      const M = 2 * Math.PI * ((t - tIns) / 1000) / T;
      const p = eciToThree(P.keplerToECI(a, e, inc, raan, argp, M)).multiplyScalar(0.001);
      const q = eciToThree(P.keplerToECI(a, e, inc, raan, argp, M + 0.002)).multiplyScalar(0.001);
      this.placeSat(world, p, q);
    } else if (A.type === 'path') {
      const f = (((t - A.t0) / 1000) / A.dur) % 1;
      const i = Math.max(0, Math.min(A.pts.length - 2, Math.floor(Math.max(0, f) * (A.pts.length - 1))));
      this.placeSat(world, A.pts[i], A.pts[i + 1]);
    } else if (A.type === 'lunar') {
      const tt = (t - A.tTLI) / 1000;
      // הירח לפי אותו מודל מעגלי שבו חושב המסלול
      const mx = 384400 * Math.cos(A.W * tt), my = 384400 * Math.sin(A.W * tt);
      const prevMoon = world.moon.position.clone();
      world.moon.position.copy(A.map(mx, my));
      world.orientMoon();
      if (this.lunarLine) this.lunarLine.position.copy(world.moon.position);
      if (this.followMoon) {
        const dm = world.moon.position.clone().sub(prevMoon);
        world.camera.position.add(dm); world.controls.target.add(dm);
      }
      const all = A.all;
      if (tt < 0 || tt > all[all.length - 1][0]) { this.craft.visible = false; this.satLabel.visible = false; return; }
      let lo = 0, hi = all.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (all[mid][0] <= tt) lo = mid; else hi = mid; }
      const a = all[lo], b = all[hi];
      const f = (tt - a[0]) / Math.max(1, b[0] - a[0]);
      const p = A.map(a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f);
      this.craft.visible = true; this.satLabel.visible = true;
      this.craft.position.copy(p);
      const dist = world.camera.position.distanceTo(p);
      this.craft.scale.setScalar(dist * 0.004);
      this.satLabel.pos.copy(p);
    }
  },

  exit(world) {
    this.clearScene();
    world.scene.remove(this.group);
    world.timeDisplay = null;
  },
};
