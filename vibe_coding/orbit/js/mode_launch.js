// mode_launch.js — לשונית "שיגור": סימולציה פיזיקלית מלאה של עלייה למסלול, מוצגת בתלת־ממד
import * as THREE from 'three';
import * as P from './physics.js';
import { ROCKETS, ROCKET_ORDER } from './rockets.js';
import { buildModel, ROCKET_HEIGHTS } from './models.js';
import { RE_KM, eciToThree, ecefDir, ecefToEci, makeLine } from './world.js';
import { barsHTML } from './mode_orbits.js';
import { $, $$, fmt, fmtT, fmtMass, fmtDur, drawChart } from './util.js';

// חלוקת המודל לחלקים לפי גובה מרכז כל רכיב (מטרים) ומרחקו מהציר
const PARTS = {
  falcon9: (y) => y < 43.4 ? 0 : y > 56.05 ? 'fairing' : 1,
  saturn5: (y) => y < 42.1 ? 0 : y < 72.3 ? 1 : y < 94.2 ? 2 : y > 101.3 ? 'les' : 'payload',
  soyuz: (y, r) => (r > 1.6 && y < 20) ? 'boosters' : y < 28.8 ? 0 : y < 35.5 ? 1 : 'fairing',
  cz5: (y, r) => r > 3 ? 'boosters' : y < 33.2 ? 0 : y < 44.7 ? 1 : 'fairing',
  starship: (y) => y < 71.5 ? 0 : 1,
};
// צבע הסילון לפי סוג הדלק
const PLUME = {
  kerolox: { color: 0xffa040, op: 0.85 },
  hydrolox: { color: 0x9fc4ff, op: 0.35 },
  methalox: { color: 0x7f8cff, op: 0.6 },
};
const FUEL = {
  falcon9: ['kerolox', 'kerolox'], saturn5: ['kerolox', 'hydrolox', 'hydrolox'], soyuz: ['kerolox', 'kerolox'],
  cz5: ['hydrolox', 'hydrolox'], starship: ['methalox', 'methalox'],
};
const EVENT_NAMES = {
  maxq: 'לחץ דינמי מרבי (Max-Q)', boosterSep: 'הפרדת המאיצים', stageSep: 'כיבוי מנועים והפרדת שלב',
  fairing: 'הפרדת כיסוי המטען', les: 'השלכת מגדל החילוץ', orbit: 'כיבוי מנוע: הגענו למסלול!', impact: 'פגיעה בקרקע',
  burn2: 'הבערה שנייה', coast: 'שיוט במסלול חניה',
};
const TARGETS = {
  leo: 'מסלול נמוך מעגלי',
  gto: 'מסלול העברה גאוסטציונרי (GTO)',
  tli: 'הזרקה לירח (TLI)',
};

export const launchMode = {
  enter(world, panel) {
    this.world = world;
    this.rid = this.rid ?? 'falcon9';
    this.target = this.target ?? 'leo';
    const R = ROCKETS[this.rid];
    this.payload = this.payload ?? R.defaultPayload;
    this.alt = this.alt ?? 200;
    this.inc = this.inc ?? (R.defaultInclination ?? R.site.lat);
    world.setWarps([0, 1, 5, 20, 60, 300, 3600], 5);
    world.moonOverride = false;
    world.controls.minDistance = 0.02;
    this.group = new THREE.Group(); world.scene.add(this.group);
    this.labels = [];
    this.debris = [];

    // הגדרה מוקדמת מתוך משימה (למשל אפולו 11)
    const pre = this.pendingPreset; this.pendingPreset = null;
    if (pre) {
      this.rid = pre.rocket; this.payload = pre.payload; this.target = pre.target;
      this.alt = pre.alt; this.inc = pre.inc;
    }

    panel.innerHTML = `
      <h2>שיגור</h2>
      <p class="lead">סימולציה מלאה: דחף שמשתנה עם הלחץ החיצוני, גרר אוויר לפי מספר מאך, אטמוספרה סטנדרטית, סיבוב כדור הארץ, שלבים שנזרקים, והנחיה במעגל סגור לשלבים העליונים.</p>
      <h3>רקטה</h3>
      <div class="rockets" id="rockets">${ROCKET_ORDER.map(id => `<button class="rocket-pick" data-id="${id}"><b>${ROCKETS[id].nameHe}</b><span>${ROCKETS[id].country}</span></button>`).join('')}</div>
      <p class="small" id="rinfo"></p>
      <div class="ctl"><label>מטען <b id="plv"></b></label><input type="range" id="pl" min="0" max="1000" step="1"></div>
      <p class="small" id="plnote"></p>
      <div class="ctl"><label>יעד</label><select id="tgt">${Object.entries(TARGETS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select></div>
      <div class="ctl" id="altc"><label>גובה המסלול <b id="altv"></b></label><input type="range" id="alt" min="160" max="1500" step="5"></div>
      <div class="ctl"><label>נטייה <b id="incv"></b></label><input type="range" id="inc" min="0" max="180" step="0.1"></div>
      <div class="row"><button class="btn" id="go">🚀 שגר</button><button class="btn secondary" id="camBtn">מצלמה: עוקבת</button><button class="btn secondary" id="maxBtn">מה המטען המרבי?</button></div>
      <div id="maxout"></div>
      <div id="result"></div>
      <h3>טלמטריה</h3>
      <canvas class="chart" id="ch1"></canvas>
      <canvas class="chart" id="ch2"></canvas>
      <div id="events" class="timeline"></div>
      <p class="note">מה רואים: בשניות הראשונות הרקטה עולה אנכית, ואז נוטה מעט ("בעיטת הטיה") ומשאירה לכובד לכופף את המסלול: סיבוב כובד (gravity turn), בלי זווית התקפה, כדי שהאוויר לא ישבור אותה. הלחץ הדינמי מגיע לשיא (Max-Q) כדקה אחרי ההמראה. רוב המהירות נבנית בכלל מעל האטמוספרה, אופקית.</p>
    `;
    this.panel = panel;
    $('#rockets', panel).addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      this.rid = b.dataset.id;
      const R2 = ROCKETS[this.rid];
      this.payload = R2.defaultPayload;
      this.inc = R2.defaultInclination ?? R2.site.lat;
      this.syncUI(); this.prepare();
    });
    $('#pl', panel).addEventListener('input', e => { this.payload = this.plFromSlider(+e.target.value); this.syncUI(); });
    $('#tgt', panel).addEventListener('change', e => { this.target = e.target.value; if (this.target !== 'leo') this.alt = 185; this.syncUI(); });
    $('#alt', panel).addEventListener('input', e => { this.alt = +e.target.value; this.syncUI(); });
    $('#inc', panel).addEventListener('input', e => { this.inc = +e.target.value; this.syncUI(); });
    $('#go', panel).onclick = () => this.launch();
    $('#camBtn', panel).onclick = () => { this.follow = !this.follow; $('#camBtn', panel).textContent = this.follow ? 'מצלמה: עוקבת' : 'מצלמה: חופשית'; if (!this.follow) this.overview(); };
    $('#maxBtn', panel).onclick = () => this.computeMax();
    this.follow = true;
    this.syncUI();
    this.prepare();
  },

  plMax() { const R = ROCKETS[this.rid]; return (R.published.leo ?? 30000) * 1.4; },
  plFromSlider(s) { return Math.round(this.plMax() * (s / 1000) ** 2 / 100) * 100; },
  plToSlider(p) { return 1000 * Math.sqrt(p / this.plMax()); },

  syncUI() {
    const { panel } = this;
    const R = ROCKETS[this.rid];
    $$('#rockets .rocket-pick', panel).forEach(b => b.classList.toggle('on', b.dataset.id === this.rid));
    const est = R.stages.some(s => s.core.est || s.core.estDry || s.boosters?.est) || R.fairing?.est;
    $('#rinfo', panel).innerHTML = `${R.name} · ${fmt(R.height, 1)} מ׳ · ${R.stages.length} שלבים · ${R.site.name}${est ? ' · <span class="est">חלק מהמסות הן הערכות (לא פורסמו)</span>' : ''}`;
    $('#pl', panel).value = this.plToSlider(this.payload);
    $('#plv', panel).textContent = fmtMass(this.payload);
    $('#plnote', panel).textContent = this.payload === R.defaultPayload ? R.defaultPayloadNote : '';
    $('#tgt', panel).value = this.target;
    $('#altc', panel).style.display = this.target === 'leo' ? '' : 'none';
    $('#alt', panel).value = this.alt;
    $('#altv', panel).textContent = this.target === 'leo' ? `${fmt(this.alt)} ק"מ` : '';
    const lat = R.site.lat;
    const inc = $('#inc', panel);
    inc.min = lat; inc.max = 180 - lat;
    this.inc = Math.min(Math.max(this.inc, lat), 180 - lat);
    inc.value = this.inc;
    $('#incv', panel).textContent = `${fmt(this.inc, 1)}°${this.inc > 90 ? ' (שיגור מערבה)' : Math.abs(this.inc - lat) < 0.05 ? ' (מזרחה, מינימום לאתר)' : ''}`;
  },

  // בניית הרקטה על כן השיגור, לפני שיגור
  prepare() {
    const { world } = this;
    this.clear3D();
    const R = ROCKETS[this.rid];
    this.R = R;
    // שיגור בבוקר לפי השעון המקומי באתר (שעה סולארית 9:30), כדי שהקרקע תהיה מוארת
    const now = new Date();
    let utcH = 9.5 - R.site.lon / 15;
    utcH = ((utcH % 24) + 24) % 24;
    this.t0 = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) + utcH * 3600e3;
    world.clock.time = this.t0;
    world.clock.paused = true;
    world.updateCelestial();
    this.theta0 = world.earth.rotation.y;
    // קטע קרקע מפורט ואתר השיגור
    this.patch = world.groundPatch(R.site.lat, R.site.lon, 80);
    const sDir = ecefDir(R.site.lat, R.site.lon);
    this.siteEcef = sDir;
    // מגדל שיגור פשוט במערכת כדור הארץ
    this.pad = new THREE.Group();
    const towerH = R.height * 1.05;
    const tower = new THREE.Mesh(new THREE.BoxGeometry(8, towerH, 8), new THREE.MeshStandardMaterial({ color: 0x5b5f66, roughness: 0.8, wireframe: true }));
    tower.position.set(R.diameter / 2 + 12, towerH / 2, 0);
    this.pad.add(tower);
    const slab = new THREE.Mesh(new THREE.CylinderGeometry(60, 60, 2, 32), new THREE.MeshStandardMaterial({ color: 0x8a8a84, roughness: 0.95 }));
    slab.position.y = -1; this.pad.add(slab);
    this.pad.scale.setScalar(0.001);
    const up = new THREE.Vector3(sDir[0], sDir[2], -sDir[1]);
    this.pad.position.copy(up).multiplyScalar(RE_KM + 0.001);
    this.pad.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
    world.earth.add(this.pad);
    // הרקטה ומיפוי חלקיה
    this.rocket = buildModel(this.rid);
    this.parts = {};
    const fn = PARTS[this.rid];
    for (const ch of [...this.rocket.children]) {
      const b = new THREE.Box3().setFromObject(ch);
      const c = b.getCenter(new THREE.Vector3());
      const key = fn(c.y, Math.hypot(c.x, c.z));
      (this.parts[key] ??= []).push(ch);
    }
    this.rocketRoot = new THREE.Group();
    this.rocketRoot.add(this.rocket);
    this.rocket.scale.setScalar(0.001);
    this.group.add(this.rocketRoot);
    // סילון
    this.plume = new THREE.Group();
    const cone = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 24, 1, true), new THREE.MeshBasicMaterial({ color: 0xffa040, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    cone.rotation.x = Math.PI; cone.position.y = -0.5;
    const core = cone.clone(); core.material = cone.material.clone(); core.material.color.set(0xffffff); core.scale.set(0.45, 0.6, 0.45); core.position.y = -0.3;
    this.plume.add(cone, core);
    this.plumeCone = cone; this.plumeCore = core;
    this.rocketRoot.add(this.plume);
    this.plume.visible = false;
    this.placeOnPad();
    // מצלמה ליד הכן
    const pos = this.rocketRoot.position;
    const upv = pos.clone().normalize();
    const side = new THREE.Vector3().crossVectors(upv, new THREE.Vector3(0, 1, 0)).normalize();
    world.controls.target.copy(pos).addScaledVector(upv, R.height * 0.0005);
    world.camera.position.copy(world.controls.target).addScaledVector(side, R.height * 0.0028).addScaledVector(upv, R.height * 0.0004);
    world.camera.up.copy(upv);
    this.sim = null;
    this.tAnim = 0;
    $('#hud').hidden = true;
    $('#result', this.panel).innerHTML = '';
    $('#events', this.panel).innerHTML = '';
    this.drawCharts(null);
    world.timeDisplay = () => 'T−00:00';
  },

  placeOnPad() {
    const w = this.world;
    const s = ecefToEci(this.siteEcef, w.earth.rotation.y);
    const up = eciToThree(s).normalize();
    this.rocketRoot.position.copy(up).multiplyScalar(RE_KM + 0.0012);
    this.rocketRoot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
  },

  clear3D() {
    const w = this.world;
    if (this.patch) { w.earth.remove(this.patch); this.patch.geometry.dispose(); this.patch = null; }
    if (this.pad) { w.earth.remove(this.pad); this.pad = null; }
    this.group.clear();
    this.debris = [];
    w.camera.up.set(0, 1, 0);
  },

  launch() {
    const R = this.R;
    const { world } = this;
    // אם הרקטה כבר טסה, מחזירים אותה לכן
    if (this.sim) this.prepare();
    const btn = $('#go', this.panel);
    btn.disabled = true; btn.textContent = 'מחשב מסלול…';
    setTimeout(() => {
      const targetAlt = (this.target === 'leo' ? this.alt : 185) * 1000;
      const kick = P.optimizeKick(R, this.payload, targetAlt, this.inc);
      const sim = P.simulateAscent({ rocket: R, payload: this.payload, targetAlt, inclination: this.inc, kick, dt: 0.05 });
      sim.events.push({ t: sim.maxQt, name: 'maxq' });
      sim.events.sort((a, b) => a.t - b.t);
      this.sim = sim;
      this.buildPath();
      this.planAfter();
      this.showResult();
      btn.disabled = false; btn.textContent = '🚀 שגר שוב';
      this.tAnim = 0;
      this.chartsFinal = false;
      this.evIdx = 0;
      this.stageShown = 0;
      world.clock.paused = false;
      this.plume.visible = true;
      $('#hud').hidden = false;
      world.timeDisplay = () => 'T+' + fmtT(this.tAnim);
      this.drawCharts(0);
    }, 30);
  },

  // מישור המסלול בתלת־ממד: נקודת השיגור s וכיוון השיגור d (לפי אזימוט שמתאים לנטייה)
  basis() {
    const R = this.R;
    const phi = R.site.lat * Math.PI / 180;
    const lam = this.theta0 + R.site.lon * Math.PI / 180;
    const s = [Math.cos(phi) * Math.cos(lam), Math.cos(phi) * Math.sin(lam), Math.sin(phi)];
    const east = [-Math.sin(lam), Math.cos(lam), 0];
    const north = [-Math.sin(phi) * Math.cos(lam), -Math.sin(phi) * Math.sin(lam), Math.cos(phi)];
    const sinA = Math.max(-1, Math.min(1, Math.cos(this.inc * Math.PI / 180) / Math.cos(phi)));
    const cosA = Math.sqrt(1 - sinA * sinA);
    const d = east.map((e, i) => sinA * e + cosA * north[i]);
    return { s, d };
  },
  toThree(r, th, out = new THREE.Vector3()) {
    const { s, d } = this.B;
    const c = Math.cos(th), sn = Math.sin(th), k = r / 1000;
    return eciToThree([k * (c * s[0] + sn * d[0]), k * (c * s[1] + sn * d[1]), k * (c * s[2] + sn * d[2])], out);
  },

  buildPath() {
    this.B = this.basis();
    const pts = this.sim.log.map(e => this.toThree(P.R_EARTH + e.alt, e.theta));
    this.pathLine = makeLine(pts, 0xfbbf24, 0.95);
    this.pathLine.geometry.setDrawRange(0, 0);
    this.pathLine.frustumCulled = false;
    this.group.add(this.pathLine);
  },

  // אחרי הכניסה למסלול: שיוט, ואם היעד GTO/TLI, הבערה שנייה של השלב העליון
  planAfter() {
    const sim = this.sim;
    this.after = null;
    if (sim.result !== 'orbit') {
      // נפילה בליסטית אחרי שנגמר הדלק
      this.after = { kind: 'fall', state: this.state2D(sim), t: sim.t };
      return;
    }
    const st = this.state2D(sim);
    const plan = { kind: 'coast', state: st, t: sim.t, burnAt: null, dv: 0 };
    if (this.target !== 'leo') {
      const dvAvail = P.upperStageDv(sim, this.payload);
      const r1 = sim.r;
      const rApo = this.target === 'gto' ? P.GEO_RADIUS : P.MOON_DIST;
      const need = P.visViva(r1, (r1 + rApo) / 2) - sim.v;
      plan.burnAt = sim.t + 1500;
      plan.dv = Math.min(need, dvAvail);
      plan.need = need; plan.avail = dvAvail;
    }
    this.after = plan;
    this.drawOrbit(st, 0x5ac8ff);
  },
  state2D(sim) {
    // מצב קרטזי במישור (x לאורך s, y לאורך d), במטרים
    const c = Math.cos(sim.theta), s = Math.sin(sim.theta);
    return { x: sim.r * c, y: sim.r * s, vx: sim.vr * c - sim.vt * s, vy: sim.vr * s + sim.vt * c };
  },
  drawOrbit(st, color) {
    const r = Math.hypot(st.x, st.y), v2 = st.vx ** 2 + st.vy ** 2;
    const a = 1 / (2 / r - v2 / P.MU);
    if (a <= 0) return;
    const T = P.periodOf(a);
    const pts = [];
    let s = { ...st };
    const n = 400, dt = T / n;
    for (let i = 0; i <= n; i++) { pts.push(this.toThree(Math.hypot(s.x, s.y), Math.atan2(s.y, s.x))); s = rk2body(s, dt, 8); }
    const line = makeLine(pts, color, 0.7);
    line.frustumCulled = false;
    this.group.add(line);
    this.orbitLines = (this.orbitLines ?? []);
    this.orbitLines.push(line);
  },

  showResult() {
    const { panel, sim, R } = this;
    const L = sim.loss;
    const vFinal = sim.v;
    const items = [
      { key: 'מהירות בסוף העלייה', v: vFinal, color: '#5aa9ff' },
      { key: 'הפסד כובד', v: L.gravity, color: '#f87171' },
      { key: 'הפסד היגוי', v: L.steering, color: '#fb923c' },
      { key: 'גרר אוויר', v: L.drag, color: '#facc15' },
      { key: this.inc > 90 ? 'קנס השיגור מערבה' : 'מתנת סיבוב כדור הארץ', v: -sim.v0, color: '#34d399' },
    ];
    const total = items.reduce((s, i) => s + i.v, 0);
    let verdict = '';
    if (sim.result === 'orbit') {
      const left = sim.upperPropLeft;
      verdict = `<div class="verdict ok">✓ <b>הגענו למסלול</b> של ${fmt(sim.peri / 1000)} × ${fmt(sim.apo / 1000)} ק"מ אחרי ${fmtDur(sim.t)}. בשלב העליון נשארו ${fmtMass(left)} דלק, שהם עוד ${fmt(P.upperStageDv(sim, this.payload))} מ׳/שנ׳ של Δv.</div>`;
      if (this.after?.need) {
        const a = this.after;
        const name = this.target === 'gto' ? 'למסלול העברה גאוסטציונרי' : 'לירח';
        verdict += a.avail >= a.need
          ? `<div class="verdict ok">✓ יש מספיק דלק להבערה ${name}: נדרשים ${fmt(a.need)} מ׳/שנ׳, יש ${fmt(a.avail)}. ההבערה תתבצע אחרי כ-25 דקות של שיוט.</div>`
          : `<div class="verdict fail">✗ <b>אין מספיק דלק</b> להבערה ${name}: נדרשים ${fmt(a.need)} מ׳/שנ׳ ויש רק ${fmt(a.avail)}. צריך להקטין את המטען.</div>`;
      }
    } else {
      const vc = P.circularSpeed(sim.r);
      verdict = `<div class="verdict fail">✗ <b>לא הגענו למסלול.</b> הדלק נגמר בגובה ${fmt((sim.r - P.R_EARTH) / 1000)} ק"מ במהירות ${fmt(sim.v)} מ׳/שנ׳, בעוד שמסלול מעגלי שם דורש ${fmt(vc)}. המטען כבד מדי לרקטה הזו, והיא תיפול בחזרה.</div>`;
    }
    $('#result', panel).innerHTML = verdict + `
      <h3>לאן הלך הדלק? (Δv שהמנועים סיפקו)</h3>
      ${barsHTML(items, total, 11000)}
      <p class="small">סכום ההבערות של כל השלבים: ${fmt(L.ideal)} מ׳/שנ׳. הפסד הכובד הוא הזמן שהמנועים "מחזיקים" את הרקטה נגד הכובד במקום להאיץ אותה; הפסד ההיגוי נובע מכך שהדחף לא מכוון בדיוק בכיוון התנועה. Max-Q: ${fmt(sim.maxQ / 1000, 1)} קילו־פסקל אחרי ${fmt(sim.maxQt)} שניות; תאוצה מרבית ${fmt(sim.maxG, 1)}g.</p>`;
    $('#events', panel).innerHTML = sim.events.map(e => `<div><span class="t">T+${fmtT(e.t)}</span> ${EVENT_NAMES[e.name] ?? e.name}${e.alt != null ? ` · ${fmt(e.alt / 1000)} ק"מ` : ''}</div>`).join('') +
      (this.after?.burnAt ? `<div><span class="t">T+${fmtT(this.after.burnAt)}</span> ${EVENT_NAMES.burn2} (${fmt(this.after.dv)} מ׳/שנ׳)</div>` : '');
  },

  computeMax() {
    const R = this.R;
    const out = $('#maxout', this.panel);
    out.innerHTML = '<div class="verdict info">מחשב (חיפוש בינארי על המטען, עשרות סימולציות)…</div>';
    setTimeout(() => {
      const alt = (this.target === 'leo' ? this.alt : 185) * 1000;
      let extra = 0;
      if (this.target !== 'leo') { const r1 = P.R_EARTH + alt; const rA = this.target === 'gto' ? P.GEO_RADIUS : P.MOON_DIST; extra = P.visViva(r1, (r1 + rA) / 2) - P.circularSpeed(r1); }
      const m = P.maxPayload(R, alt, extra, this.inc);
      const pub = this.target === 'leo' ? R.published.leo : this.target === 'gto' ? R.published.gto : R.published.tli;
      out.innerHTML = `<div class="verdict info">בסימולציה: עד <b>${fmtMass(m)}</b> ${this.target === 'leo' ? `למסלול של ${fmt(this.alt)} ק"מ` : TARGETS[this.target]} בנטייה ${fmt(this.inc, 1)}°.
        ${pub ? `<br>הערך המפורסם: ${fmtMass(pub)}. ` : ''}<span class="small">רקטות אמיתיות שומרות רזרבות דלק, מוגבלות בעומסים מבניים ובאילוצי בטיחות, ולכן הערכים המפורסמים נמוכים בדרך כלל ב-10–20% מהסימולציה האידיאלית.</span></div>`;
      this.payload = Math.round(m / 100) * 100;
      this.syncUI();
    }, 30);
  },

  update(world, dtReal, dtSim) {
    if (!this.sim) {
      if (this.rocketRoot) this.placeOnPad();
      return;
    }
    const sim = this.sim, log = sim.log;
    this.tAnim += dtSim;
    const t = this.tAnim;
    const tEnd = log[log.length - 1].t;
    let pos, fwd, e;
    if (t <= tEnd) {
      // אינטרפולציה בטלמטריה
      let i = Math.min(log.length - 2, Math.floor(t));
      while (i > 0 && log[i].t > t) i--;
      while (i < log.length - 2 && log[i + 1].t < t) i++;
      const a = log[i], b = log[i + 1];
      const f = Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t)));
      e = {};
      for (const k of ['alt', 'theta', 'v', 'vRel', 'm', 'F', 'q', 'g', 'pitch', 'M']) e[k] = a[k] + (b[k] - a[k]) * f;
      e.stage = a.stage;
      const r = P.R_EARTH + e.alt;
      pos = this.toThree(r, e.theta);
      const up = pos.clone().normalize();
      const down = this.toThree(r, e.theta + 1e-6).sub(pos).normalize();
      fwd = up.clone().multiplyScalar(Math.sin(e.pitch)).addScaledVector(down, Math.cos(e.pitch)).normalize();
      this.pathLine.geometry.setDrawRange(0, i + 2);
      if (e.alt < 1 && t < 3) { // על הכן, כדור הארץ מסתובב
        pos = this.rocketRoot.position.clone();
      }
      this.hud(e, t);
      this.plume.visible = e.F > 0;
      this.updatePlume(e);
      this.state = null;
    } else {
      // אחרי סוף העלייה: תנועה בכובד בלבד (עם הבערה שנייה אם תוכננה)
      const A = this.after;
      if (!this.state) { this.state = { ...A.state }; this.stT = A.t; this.burnDone = false; this.plume.visible = false; this.pathLine.geometry.setDrawRange(0, log.length); }
      let target = t;
      if (A.burnAt && !this.burnDone && target >= A.burnAt) target = A.burnAt;
      while (this.stT < target - 1e-6) {
        const step = Math.min(target - this.stT, 20);
        this.state = rk2body(this.state, step, Math.max(1, Math.ceil(step / 5)));
        this.stT += step;
      }
      if (A.burnAt && !this.burnDone && t >= A.burnAt) {
        const s = this.state, v = Math.hypot(s.vx, s.vy);
        s.vx += s.vx / v * A.dv; s.vy += s.vy / v * A.dv;
        this.burnDone = true;
        this.drawOrbit(s, A.dv >= A.need - 1 ? 0xa78bfa : 0xf87171);
        this.flash(EVENT_NAMES.burn2 + ` · ${fmt(A.dv)} מ׳/שנ׳`);
      }
      const r = Math.hypot(this.state.x, this.state.y), th = Math.atan2(this.state.y, this.state.x);
      if (r < P.R_EARTH) { world.clock.rate = 0; this.flash(EVENT_NAMES.impact); }
      pos = this.toThree(r, th);
      const vel = this.toThree(Math.hypot(this.state.x + this.state.vx, this.state.y + this.state.vy), Math.atan2(this.state.y + this.state.vy, this.state.x + this.state.vx)).sub(pos).normalize();
      fwd = vel;
      this.hudCoast(r, t);
    }
    // מיקום והכוונה של הרקטה
    const prev = this.rocketRoot.position.clone();
    const prevQ = this.rocketRoot.quaternion.clone();
    this.rocketRoot.position.copy(pos);
    // סיבוב מינימלי מהכיוון הקודם, כדי שהרקטה לא "תתגלגל" סביב צירה
    const curUp = new THREE.Vector3(0, 1, 0).applyQuaternion(prevQ);
    this.rocketRoot.quaternion.premultiply(new THREE.Quaternion().setFromUnitVectors(curUp, fwd));
    // אירועים: הפרדות
    while (this.evIdx < sim.events.length && sim.events[this.evIdx].t <= t) this.fire(sim.events[this.evIdx++]);
    // שלבים שנזרקו: נופלים בבליסטיקה פשוטה
    for (const d of this.debris) {
      if (dtSim > 0) {
        const r = d.p.length();
        const g = d.p.clone().multiplyScalar(-P.MU / 1e9 / r ** 3);
        d.v.addScaledVector(g, dtSim);
        if (r > RE_KM) d.p.addScaledVector(d.v, dtSim);
        d.age += dtSim;
        d.obj.rotateX(0.15 * dtSim);
      }
      d.obj.position.copy(d.p);
      d.obj.visible = d.age < 240;
    }
    // מצלמה עוקבת
    if (this.follow) {
      // המצלמה שומרת על מיקומה ביחס לרקטה, כולל סיבוב יחד איתה
      const dq = this.rocketRoot.quaternion.clone().multiply(prevQ.clone().invert());
      const camOff = world.camera.position.clone().sub(prev).applyQuaternion(dq);
      const tgtOff = world.controls.target.clone().sub(prev).applyQuaternion(dq);
      world.camera.position.copy(this.rocketRoot.position).add(camOff);
      world.controls.target.copy(this.rocketRoot.position).add(tgtOff);
      world.camera.up.copy(this.rocketRoot.position).normalize();
    }
  },

  updatePlume(e) {
    const R = this.R;
    const fuel = FUEL[this.rid][Math.min(e.stage, FUEL[this.rid].length - 1)];
    const pl = PLUME[fuel];
    const atm = P.atmosphere(e.alt);
    const vac = 1 - Math.min(1, atm.p / 101325);
    const len = R.height * (0.35 + 0.9 * vac) * Math.min(1, e.F / 1e6 + 0.3);
    const wid = R.diameter * (0.45 + 2.2 * vac * vac);
    this.plume.scale.set(wid * 0.001, len * 0.001, wid * 0.001);
    // בסיס הסילון בתחתית השלב הפעיל
    const base = { falcon9: [0, 43.4], saturn5: [0, 42.1, 72.3], soyuz: [0, 28.8], cz5: [0, 33.2], starship: [0, 72.3] }[this.rid][e.stage] ?? 0;
    this.plume.position.set(0, base * 0.001, 0);
    this.plumeCone.material.color.setHex(pl.color);
    this.plumeCone.material.opacity = pl.op * (0.6 + 0.4 * Math.random()) * (1 - 0.5 * vac);
    this.plumeCore.material.opacity = pl.op * 0.7;
  },

  fire(ev) {
    const name = ev.name;
    let key = null;
    if (name === 'boosterSep') key = 'boosters';
    else if (name === 'stageSep') key = ev.stage;
    else if (name === 'fairing') key = 'fairing';
    else if (name === 'les') key = 'les';
    this.flash(EVENT_NAMES[name] ?? name);
    if (key == null || !this.parts[key]) return;
    // מעבירים את החלקים לאובייקט נופל נפרד
    const g = new THREE.Group();
    g.position.copy(this.rocketRoot.position);
    g.quaternion.copy(this.rocketRoot.quaternion);
    const inner = new THREE.Group(); inner.scale.setScalar(0.001); g.add(inner);
    for (const ch of this.parts[key]) { ch.parent?.remove(ch); inner.add(ch); }
    this.group.add(g);
    // מהירות: של הרקטה ברגע ההפרדה, פחות מעט
    const v = this.velocityNow();
    const sideKick = key === 'fairing' ? 0.006 : 0;
    this.debris.push({ obj: g, p: g.position.clone(), v: v.multiplyScalar(0.999).add(new THREE.Vector3(sideKick, 0, 0)), age: 0 });
  },
  velocityNow() {
    const t = this.tAnim, log = this.sim.log;
    const i = Math.min(log.length - 2, Math.max(0, Math.floor(t)));
    const a = log[i], b = log[i + 1];
    const pa = this.toThree(P.R_EARTH + a.alt, a.theta), pb = this.toThree(P.R_EARTH + b.alt, b.theta);
    return pb.sub(pa).divideScalar(b.t - a.t);
  },

  flash(text) {
    this.lastEvent = text;
    this.lastEventT = performance.now();
  },

  hud(e, t) {
    const R = this.R;
    const stageName = R.stages[Math.min(e.stage, R.stages.length - 1)].name;
    const evt = this.lastEvent && performance.now() - this.lastEventT < 5000 ? this.lastEvent : '';
    $('#hud').innerHTML = `
      <div><div class="k">זמן</div><div class="v">T+${fmtT(t)}</div></div>
      <div><div class="k">גובה</div><div class="v">${fmt(e.alt / 1000, 1)} km</div></div>
      <div><div class="k">מהירות יחסית לקרקע</div><div class="v">${fmt(e.vRel * 3.6)} km/h</div></div>
      <div><div class="k">מהירות מסלולית</div><div class="v">${fmt(e.v * 3.6)} km/h</div></div>
      <div><div class="k">תאוצה</div><div class="v">${fmt(e.g, 2)} g</div></div>
      <div><div class="k">לחץ דינמי</div><div class="v">${fmt(e.q / 1000, 1)} kPa</div></div>
      <div><div class="k">${stageName}</div><div class="v">${fmt(e.m / 1000, 0)} t</div></div>
      <div class="ev">${evt}</div>`;
    this.chartAcc = (this.chartAcc ?? 0) + 1;
    if (this.chartAcc % 6 === 0) this.drawCharts(t);
  },
  hudCoast(r, t) {
    if (!this.chartsFinal) { this.chartsFinal = true; this.drawCharts(this.sim.log[this.sim.log.length - 1].t); }
    const evt = this.lastEvent && performance.now() - this.lastEventT < 5000 ? this.lastEvent : (this.after.kind === 'fall' ? 'נפילה בליסטית' : 'במסלול, בלי מנועים');
    const v = Math.hypot(this.state.vx, this.state.vy);
    $('#hud').innerHTML = `
      <div><div class="k">זמן</div><div class="v">T+${fmtT(t)}</div></div>
      <div><div class="k">גובה</div><div class="v">${fmt((r - P.R_EARTH) / 1000, 0)} km</div></div>
      <div><div class="k">מהירות מסלולית</div><div class="v">${fmt(v * 3.6)} km/h</div></div>
      <div><div class="k">מנועים</div><div class="v">כבויים</div></div>
      <div><div class="k">תאוצה</div><div class="v">0 g</div></div>
      <div><div class="k">לחץ דינמי</div><div class="v">0</div></div>
      <div><div class="k">מסה</div><div class="v">${fmt(this.sim.mass / 1000, 1)} t</div></div>
      <div class="ev">${evt}</div>`;
  },

  drawCharts(t) {
    const c1 = $('#ch1', this.panel), c2 = $('#ch2', this.panel);
    if (!c1) return;
    if (!this.sim) { drawChart(c1, [], { title: 'גובה (ק"מ, צהוב) ומהירות (מ׳/שנ׳ ÷10, תכלת)' }); drawChart(c2, [], { title: 'תאוצה (g ×10, כתום) ולחץ דינמי (kPa, ורוד)' }); return; }
    const log = this.sim.log;
    drawChart(c1, [
      { data: log.map(e => [e.t, e.alt / 1000]), color: '#fbbf24' },
      { data: log.map(e => [e.t, e.v / 10]), color: '#5ac8ff' },
    ], { title: 'גובה (ק"מ, צהוב) ומהירות (מ׳/שנ׳ ÷10, תכלת)', marker: t, xLabel: 'שניות' });
    drawChart(c2, [
      { data: log.map(e => [e.t, e.g * 10]), color: '#fb923c' },
      { data: log.map(e => [e.t, e.q / 1000]), color: '#f472b6' },
    ], { title: 'תאוצה (g ×10, כתום) ולחץ דינמי (kPa, ורוד)', marker: t, xLabel: 'שניות' });
  },

  overview() {
    const w = this.world;
    w.controls.target.set(0, 0, 0);
    w.camera.up.set(0, 1, 0);
    w.camera.position.copy(this.rocketRoot.position).normalize().multiplyScalar(RE_KM * 3.2).add(new THREE.Vector3(0, RE_KM, 0));
  },

  exit(world) {
    this.clear3D();
    world.scene.remove(this.group);
    world.clock.paused = false;
    world.controls.minDistance = RE_KM * 1.05;
    world.camera.up.set(0, 1, 0);
    this.sim = null;
  },
};

// אינטגרציה של בעיית שני הגופים במישור (RK4 קלאסי), במטרים
function rk2body(s, dt, n) {
  const h = dt / n;
  const f = (x, y, vx, vy) => { const r3 = Math.hypot(x, y) ** 3; return [vx, vy, -P.MU * x / r3, -P.MU * y / r3]; };
  let X = [s.x, s.y, s.vx, s.vy];
  for (let i = 0; i < n; i++) {
    const k1 = f(...X);
    const k2 = f(...X.map((v, j) => v + h / 2 * k1[j]));
    const k3 = f(...X.map((v, j) => v + h / 2 * k2[j]));
    const k4 = f(...X.map((v, j) => v + h * k3[j]));
    X = X.map((v, j) => v + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
  }
  return { x: X[0], y: X[1], vx: X[2], vy: X[3] };
}
