// mode_orbits.js — לשונית "מסלולים": בוחרים גובה ונטייה, ורואים מה זה דורש
import * as THREE from 'three';
import * as P from './physics.js';
import { ROCKETS } from './rockets.js';
import { ORBIT_PRESETS } from './data.js';
import { buildModel } from './models.js';
import { RE_KM, eciToThree, makeLine } from './world.js';
import { $, $$, fmt, fmtDur, fmtMass, h } from './util.js';

export const SITES = [
  { id: 'cape', name: 'קייפ קנוורל, פלורידה', lat: 28.5, lon: -80.6 },
  { id: 'kourou', name: 'קורו, גיאנה הצרפתית', lat: 5.2, lon: -52.8 },
  { id: 'wenchang', name: 'ון־צ\'אנג, סין', lat: 19.6, lon: 110.95 },
  { id: 'starbase', name: 'סטארבייס, טקסס', lat: 26.0, lon: -97.2 },
  { id: 'baikonur', name: 'בייקונור, קזחסטן', lat: 45.9, lon: 63.3 },
  { id: 'palmachim', name: 'פלמחים (שיגור מערבה)', lat: 31.9, lon: 34.7, retro: true },
];

// הפסדים טיפוסיים בעלייה למסלול נמוך, מחושבים פעם אחת מסימולציה של פלקון 9
let ASCENT_LOSS = null;
function ascentLoss() {
  if (!ASCENT_LOSS) {
    const R = ROCKETS.falcon9;
    const res = P.simulateAscent({ rocket: R, payload: 15000, targetAlt: 200e3, inclination: 28.5, kick: P.optimizeKick(R, 15000, 200e3, 28.5), record: false, dt: 0.2 });
    ASCENT_LOSS = res.loss;
  }
  return ASCENT_LOSS;
}

// תקציב דלתא־וי מהקרקע למסלול מעגלי בגובה altKm ובנטייה incDeg, מאתר site
export function dvBudget(altKm, incDeg, site, toMoon = false) {
  const lat = site.lat;
  let inc = incDeg;
  if (site.retro) inc = Math.max(inc, 180 - lat); // שיגור מערבה: מסלול רטרוגרדי
  // אם הנטייה המבוקשת קטנה מקו הרוחב, משגרים מזרחה ומשנים מישור בהמשך
  const launchInc = site.retro ? inc : Math.max(inc, lat);
  const planeChange = site.retro ? 0 : Math.max(0, lat - inc) * Math.PI / 180;
  const gift = P.OMEGA_EARTH * P.R_EARTH * Math.cos(launchInc * Math.PI / 180);
  const loss = ascentLoss();
  const park = Math.min(altKm, 200) * 1000;
  const vPark = P.circularSpeed(P.R_EARTH + park);
  const items = [
    { key: 'מהירות מסלולית ב-200 ק"מ', v: vPark, color: '#5aa9ff' },
    { key: 'הפסד כובד', v: loss.gravity, color: '#f87171' },
    { key: 'הפסד היגוי', v: loss.steering, color: '#fb923c' },
    { key: 'גרר אוויר', v: loss.drag, color: '#facc15' },
    { key: gift >= 0 ? 'מתנת סיבוב כדור הארץ' : 'קנס השיגור מערבה', v: -gift, color: gift >= 0 ? '#34d399' : '#f43f5e' },
  ];
  if (toMoon) {
    const r1 = P.R_EARTH + park, a = (r1 + P.MOON_DIST) / 2;
    items.push({ key: 'הזרקה לירח (TLI)', v: P.visViva(r1, a) - vPark, color: '#a78bfa' });
    items.push({ key: 'בלימה למסלול ירחי', v: 894, color: '#c4b5fd' });
  } else if (altKm > 200.5 || planeChange > 0) {
    const r1 = P.R_EARTH + park, r2 = P.R_EARTH + altKm * 1000;
    const hm = P.hohmann(r1, r2);
    const vApoT = P.visViva(r2, hm.a);
    const dv2 = P.combinedBurn(vApoT, P.circularSpeed(r2), planeChange);
    if (altKm > 200.5) items.push({ key: 'הבערה 1: העלאת השיא', v: hm.dv1, color: '#a78bfa' });
    items.push({ key: planeChange > 0 ? 'הבערה 2: עיגול ושינוי נטייה' : 'הבערה 2: עיגול המסלול', v: dv2, color: '#c4b5fd' });
  }
  const total = items.reduce((s, i) => s + i.v, 0);
  const extraDv = items.slice(5).reduce((s, i) => s + i.v, 0);
  return { items, total, extraDv, launchInc, planeChange };
}

export function barsHTML(items, total, scale) {
  const max = scale ?? Math.max(...items.map(i => Math.abs(i.v)), 1);
  return `<div class="bars">${items.map(i => `
    <div class="bar-row"><span>${i.key}</span>
      <div class="track"><div class="fill" style="width:${Math.min(100, Math.abs(i.v) / max * 100)}%;background:${i.color};${i.v < 0 ? 'opacity:.55;' : ''}"></div></div>
      <span class="val">${i.v < 0 ? '−' : ''}${fmt(Math.abs(i.v) / 1000, 2)}</span></div>`).join('')}
    <div class="bar-row total"><span>סה"כ (ק"מ/שנייה)</span><span></span><span class="val">${fmt(total / 1000, 2)}</span></div></div>`;
}

const ALT_MIN = 160, ALT_MAX = 384400 - 6378;
const sliderToAlt = s => ALT_MIN * Math.pow(ALT_MAX / ALT_MIN, s / 1000);
const altToSlider = a => 1000 * Math.log(a / ALT_MIN) / Math.log(ALT_MAX / ALT_MIN);

export const orbitsMode = {
  enter(world, panel) {
    this.world = world;
    this.alt = this.alt ?? 480;
    this.inc = this.inc ?? 53;
    this.preset = this.preset ?? 'starlink';
    this.site = this.site ?? SITES[0];
    this.u = 0;
    world.clock.time = Date.now();
    world.setWarps([0, 1, 10, 60, 300, 1800, 7200], 60);
    world.controls.minDistance = RE_KM * 1.05;
    world.moonOverride = false;

    this.group = new THREE.Group();
    world.scene.add(this.group);
    this.earthGroup = new THREE.Group();
    world.earth.add(this.earthGroup);

    // טבעת גאוסטציונרית קבועה להתמצאות
    const geoPts = [];
    for (let k = 0; k <= 256; k++) { const a = k / 256 * Math.PI * 2; geoPts.push(new THREE.Vector3(Math.cos(a) * 42164, 0, -Math.sin(a) * 42164)); }
    this.geoRing = makeLine(geoPts, 0xf59e0b, 0.25, true);
    this.group.add(this.geoRing);

    this.satLabel = world.addLabel('', 'leo big');
    this.siteLabel = null;

    panel.innerHTML = `
      <h2>מסלולים</h2>
      <p class="lead">ככל שהמסלול גבוה יותר, הלוויין נע לאט יותר, אבל להגיע אליו עולה הרבה יותר. בחרו גובה ונטייה וראו מה זה דורש.</p>
      <div class="presets" id="presets">${ORBIT_PRESETS.map(p => `<button class="chip" data-id="${p.id}">${p.name}</button>`).join('')}</div>
      <div class="ctl"><label>גובה <b id="altv"></b></label><input type="range" id="alt" min="0" max="1000" step="1"></div>
      <div class="ctl"><label>נטייה (זווית מישור המסלול ביחס לקו המשווה) <b id="incv"></b></label><input type="range" id="inc" min="0" max="180" step="0.1"></div>
      <p class="note" id="pnote"></p>
      <div class="cards" id="cards"></div>
      <h3>כמה זה עולה: תקציב דלתא־וי</h3>
      <p class="small">דלתא־וי (Δv, "שינוי מהירות") הוא המטבע של טיסות חלל: סך כל שינויי המהירות שהמנועים צריכים לספק. ההפסדים חושבו בסימולציית שיגור מלאה של פלקון 9.</p>
      <div class="ctl"><label>אתר שיגור</label><select id="site">${SITES.map(s => `<option value="${s.id}">${s.name} (${fmt(s.lat, 1)}°)</option>`).join('')}</select></div>
      <div id="bars"></div>
      <div id="rocketq" class="verdict info"></div>
      <div id="single" class="note"></div>
    `;
    $('#presets', panel).addEventListener('click', e => { const b = e.target.closest('button'); if (b) this.applyPreset(b.dataset.id); });
    $('#alt', panel).addEventListener('input', e => { this.alt = sliderToAlt(+e.target.value); this.preset = null; this.refresh(true); });
    $('#inc', panel).addEventListener('input', e => { this.inc = +e.target.value; this.preset = null; this.refresh(true); });
    $('#site', panel).value = this.site.id;
    $('#site', panel).addEventListener('change', e => { this.site = SITES.find(s => s.id === e.target.value); this.refresh(false); });
    this.panel = panel;
    if (this.preset) this.applyPreset(this.preset); else this.refresh(true);
    this.frameCamera(true);
  },

  applyPreset(id) {
    const p = ORBIT_PRESETS.find(x => x.id === id);
    this.preset = id;
    this.alt = p.alt; this.inc = p.inc;
    this.refresh(true);
    this.frameCamera(false);
  },

  frameCamera(initial) {
    const r = RE_KM + this.alt;
    const d = Math.max(RE_KM * 3.2, r * 2.6);
    const cam = this.world.camera;
    const dir = cam.position.clone().normalize();
    if (initial || dir.lengthSq() === 0) dir.set(0.3, 0.45, 0.85).normalize();
    cam.position.copy(dir.multiplyScalar(Math.min(d, 1.4e6)));
    this.world.controls.target.set(0, 0, 0);
  },

  refresh(geomChanged) {
    const { world, panel } = this;
    const altKm = this.alt, inc = this.inc;
    const preset = ORBIT_PRESETS.find(x => x.id === this.preset);
    const isMoon = this.preset === 'moon';
    $('#alt', panel).value = altToSlider(altKm);
    $('#altv', panel).textContent = `${fmt(altKm)} ק"מ`;
    $('#inc', panel).value = inc;
    $('#incv', panel).textContent = `${fmt(inc, 1)}°`;
    $$('#presets .chip', panel).forEach(c => c.classList.toggle('on', c.dataset.id === this.preset));
    $('#pnote', panel).textContent = preset?.note ?? 'מסלול מעגלי מותאם אישית.';
    $('#pnote', panel).style.display = 'block';

    const r = P.R_EARTH + altKm * 1000;
    const v = P.circularSpeed(r), T = P.periodOf(r);
    const elev = altKm < 2000 ? 25 : 10;
    const lam = P.footprintAngle(altKm * 1000, elev * Math.PI / 180);
    const cover = P.capFraction(lam);
    const rtt = 4 * altKm * 1000 / P.C_LIGHT;
    const life = P.decayTime(altKm * 1000, 50);
    const lam10 = P.footprintAngle(altKm * 1000, 10 * Math.PI / 180);
    const relRate = 2 * Math.PI / T - (Math.abs(inc) < 1 ? P.OMEGA_EARTH : 0);
    const isGeo = Math.abs(altKm - 35786) < 50 && inc < 1;
    const sky = isGeo ? 'עומד במקום' : isMoon ? 'כמו הירח' : fmtDur(2 * lam10 / Math.abs(relRate));
    this.T = T;

    $('#cards', panel).innerHTML = `
      <div class="card"><div class="k">מהירות מסלולית</div><div class="v">${fmt(v / 1000, 2)} <small>ק"מ/שנ׳</small></div></div>
      <div class="card"><div class="k">= קמ"ש</div><div class="v">${fmt(v * 3.6)}</div></div>
      <div class="card"><div class="k">זמן הקפה</div><div class="v">${fmtDur(T)}</div></div>
      <div class="card"><div class="k">הקפות ביממה</div><div class="v">${fmt(86400 / T, T > 86400 * 3 ? 3 : 2)}</div></div>
      <div class="card"><div class="k">השהיה מינימלית הלוך־חזור${isMoon ? '' : ' דרך הלוויין'}</div><div class="v">${rtt < 1 ? fmt(rtt * 1000, rtt < 0.1 ? 1 : 0) + ' <small>מילישנייה</small>' : fmt(rtt, 2) + ' <small>שנ׳</small>'}</div></div>
      <div class="card"><div class="k">משך מעבר בשמיים (מעל 10°)</div><div class="v">${sky}</div></div>
      ${isMoon ? '' : `<div class="card"><div class="k">חלק מכדור הארץ שרואה אותו (מעל ${elev}°)</div><div class="v">${fmt(cover * 100, cover < 0.01 ? 2 : 1)}%</div></div>
      <div class="card"><div class="k">נשאר במסלול בלי הנעה</div><div class="v">${life === Infinity ? 'מאות שנים ויותר' : fmtDur(life)}</div></div>`}
    `;
    this.updateBudget(isMoon);

    if (geomChanged) this.rebuild();
  },

  updateBudget(isMoon) {
    const { panel } = this;
    const b = dvBudget(this.alt, this.inc, this.site, isMoon);
    this.budget = b;
    $('#bars', panel).innerHTML = barsHTML(b.items, b.total, 11000);
    // שלב יחיד: כמה מהרקטה יכול להיות מבנה ומטען (Isp=350 שנ')
    const mr = P.massRatioFor(b.total, 350);
    $('#single', panel).innerHTML = `לפי משוואת הרקטה של ציולקובסקי, רקטה חד־שלבית עם מנוע טוב (Isp של 350 שניות) צריכה יחס מסה של <b>${fmt(mr, 1)}</b> כדי לספק ${fmt(b.total / 1000, 1)} ק"מ לשנייה: רק <b>${fmt(100 / mr, 1)}%</b> מהמסה בהמראה יכולים להיות מבנה, מנועים ומטען. כל השאר דלק. לכן רקטות בנויות משלבים שנזרקים בדרך.`;
    // מטען מרבי של פלקון 9 למסלול הזה (חישוב ברקע)
    const q = $('#rocketq', panel);
    q.innerHTML = 'מחשב כמה פלקון 9 יכול להביא לכאן…';
    clearTimeout(this.pt);
    const site = this.site;
    this.pt = setTimeout(() => {
      const R = ROCKETS.falcon9;
      const pl = P.maxPayload(R, 200e3, b.extraDv, b.launchInc);
      const frac = pl / 549000 * 100;
      q.innerHTML = pl < 50
        ? `פלקון 9 (בלי החזרת המאיץ) <b>לא מסוגל</b> להביא לכאן מטען מ${site.name}.`
        : `פלקון 9 (בלי החזרת המאיץ) יכול להביא לכאן כ-<b>${fmtMass(pl)}</b> מ${site.name}: <b>${fmt(frac, 1)}%</b> ממסת ההמראה של 549 טון. <span class="small">(סימולציה; הערכים המפורסמים: 22.8 טון למסלול נמוך, 8.3 טון למסלול העברה גאוסטציונרי.)</span>`;
    }, 250);
  },

  rebuild() {
    const { world } = this;
    // ניקוי
    for (const o of [this.orbitLine, this.sat, this.track, this.foot]) { if (o) o.parent?.remove(o); }
    const altKm = this.alt, inc = this.inc * Math.PI / 180;
    const r = RE_KM + altKm;
    this.r = r;
    // מישור המסלול: צומת עולה לאורך x
    const Pv = [1, 0, 0], Q = [0, Math.cos(inc), Math.sin(inc)];
    this.Pv = Pv; this.Q = Q;
    const pts = [];
    for (let k = 0; k <= 360; k++) {
      const a = k * Math.PI / 180;
      pts.push(eciToThree([r * (Math.cos(a) * Pv[0] + Math.sin(a) * Q[0]), r * (Math.cos(a) * Pv[1] + Math.sin(a) * Q[1]), r * (Math.cos(a) * Pv[2] + Math.sin(a) * Q[2])]));
    }
    this.orbitLine = makeLine(pts, this.alt > 30000 && this.inc < 1 ? 0xf59e0b : 0x5ac8ff, 0.85);
    this.group.add(this.orbitLine);
    // לוויין
    const preset = ORBIT_PRESETS.find(x => x.id === this.preset);
    const modelId = preset ? preset.model : (altKm > 10000 ? 'geo' : 'starlink');
    if (modelId) {
      this.sat = buildModel(modelId);
      const box = new THREE.Box3().setFromObject(this.sat);
      this.satSize = box.getSize(new THREE.Vector3()).length();
      this.group.add(this.sat);
    } else this.sat = null;
    this.satLabel.el.textContent = preset ? preset.name : 'לוויין';
    this.satLabel.visible = !!modelId;
    // התחלה מעל ישראל בערך (אורך 35° מזרח), כדי שיהיה קל לראות לוויין גאוסטציונרי עומד
    this.u = P.gmst(new Date(world.clock.time)) + 35 * Math.PI / 180;
    if (Math.abs(inc) > 0.01) this.u = 0.6;
    // עקבת קרקע
    this.trackPts = [];
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2500 * 3), 3));
    tg.setDrawRange(0, 0);
    this.track = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xfcd34d, transparent: true, opacity: 0.85, depthWrite: false }));
    this.track.frustumCulled = false;
    this.earthGroup.add(this.track);
    // כתם כיסוי
    const elev = altKm < 2000 ? 25 : 10;
    const lam = P.footprintAngle(altKm * 1000, elev * Math.PI / 180);
    if (this.preset !== 'moon') {
      this.foot = new THREE.Mesh(new THREE.SphereGeometry(RE_KM + 30, 192, 16, 0, Math.PI * 2, 0, lam),
        new THREE.MeshBasicMaterial({ color: altKm > 10000 ? 0xf59e0b : 0x38bdf8, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }));
      this.earthGroup.add(this.foot);
    } else this.foot = null;
  },

  satPos(u) {
    const r = this.r, Pv = this.Pv, Q = this.Q;
    return [r * (Math.cos(u) * Pv[0] + Math.sin(u) * Q[0]), r * (Math.cos(u) * Pv[1] + Math.sin(u) * Q[1]), r * (Math.cos(u) * Pv[2] + Math.sin(u) * Q[2])];
  },

  update(world, dtReal, dtSim) {
    if (!this.r) return;
    this.u += 2 * Math.PI / this.T * dtSim;
    const p = this.satPos(this.u);
    const pos = eciToThree(p);
    if (this.preset === 'moon') { world.moonOverride = true; world.moon.position.copy(pos); world.orientMoon(); }
    else world.moonOverride = false;
    if (this.sat) {
      this.sat.position.copy(pos);
      // מכוונים את הלוויין: צד ה"תחתון" כלפי כדור הארץ, ציר x בכיוון התנועה
      const down = pos.clone().negate().normalize();
      const vel = eciToThree(this.satPos(this.u + 0.001)).sub(pos).normalize();
      const y = down.clone().negate();
      const z = new THREE.Vector3().crossVectors(vel, y).normalize();
      const x = new THREE.Vector3().crossVectors(y, z);
      this.sat.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
      // הגדלה כדי שיהיה נראה (מצוין בתווית)
      const dist = world.camera.position.distanceTo(pos);
      const s = Math.max(0.001, dist * 0.045 / this.satSize);
      this.sat.scale.setScalar(s);
      this.satLabel.pos.copy(pos);
      this.satLabel.el.textContent = (ORBIT_PRESETS.find(x => x.id === this.preset)?.name ?? 'לוויין') + (s > 0.0011 ? ` (מוגדל פי ${fmt(s * 1000)})` : '');
    }
    // עקבת קרקע ונקודת כיסוי במערכת כדור הארץ
    const local = world.earth.worldToLocal(pos.clone()).normalize();
    if (this.foot) this.foot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), local);
    const tp = local.clone().multiplyScalar(RE_KM + 6);
    const last = this.trackPts[this.trackPts.length - 1];
    if (!last || last.distanceTo(tp) > 40) {
      this.trackPts.push(tp);
      if (this.trackPts.length > 2500) this.trackPts.shift();
      const attr = this.track.geometry.attributes.position;
      this.trackPts.forEach((q, i) => attr.setXYZ(i, q.x, q.y, q.z));
      attr.needsUpdate = true;
      this.track.geometry.setDrawRange(0, this.trackPts.length);
    }
  },

  exit(world) {
    world.scene.remove(this.group);
    world.earth.remove(this.earthGroup);
    world.removeLabel(this.satLabel);
    world.moonOverride = false;
    this.r = null;
  },
};
