// mode_orbits.js — לשונית "מסלולים": בוחרים גובה ונטייה, ורואים מה זה דורש
import * as THREE from 'three';
import * as P from './physics.js';
import { ROCKETS } from './rockets.js';
import { ORBIT_PRESETS } from './data.js';
import { buildModel } from './models.js';
import { RE_KM, eciToThree, makeLine } from './world.js';
import { $, $$, fmt, fmtDur, fmtMass, h } from './util.js';
import { settings, speedNum, speedUnit, altSpeedStr } from './settings.js';

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

const COLORS = { iss: '#a78bfa', tiangong: '#f472b6', starlink: '#38bdf8', hubble: '#34d399', gps: '#f87171', geo: '#f59e0b', moon: '#d1d5db', custom: '#facc15' };
// כל מסלול מקבל צומת עולה אחר, כדי שהקווים לא יתלכדו
const RAAN = { iss: 0, tiangong: 70, starlink: 140, hubble: 210, gps: 280, geo: 0, moon: 30, custom: 110 };

function orbitStats(o) {
  const altKm = o.alt, r = P.R_EARTH + altKm * 1000;
  const v = P.circularSpeed(r), T = P.periodOf(r);
  const isMoon = o.id === 'moon';
  const elev = altKm < 2000 ? 25 : 10;
  const lam = P.footprintAngle(altKm * 1000, elev * Math.PI / 180);
  const lam10 = P.footprintAngle(altKm * 1000, 10 * Math.PI / 180);
  const relRate = 2 * Math.PI / T - (Math.abs(o.inc) < 1 ? P.OMEGA_EARTH : 0);
  const isGeo = Math.abs(altKm - 35786) < 50 && o.inc < 1;
  return {
    v, T, isMoon, elev, cover: P.capFraction(lam), rtt: 4 * altKm * 1000 / P.C_LIGHT,
    life: P.decayTime(altKm * 1000, 50),
    sky: isGeo ? 'עומד במקום' : isMoon ? 'כמו הירח' : fmtDur(2 * lam10 / Math.abs(relRate)),
  };
}
const fmtRtt = rtt => rtt < 1 ? fmt(rtt * 1000, rtt < 0.1 ? 1 : 0) + ' מ"ש' : fmt(rtt, 2) + ' שנ׳';
const fmtLife = l => l === Infinity ? 'מאות שנים' : fmtDur(l);

export const orbitsMode = {
  enter(world, panel) {
    this.world = world;
    this.site = this.site ?? SITES[0];
    if (!this.orbits) {
      this.orbits = {};
      for (const p of ORBIT_PRESETS) this.orbits[p.id] = { ...p, color: COLORS[p.id] };
      this.orbits.custom = { id: 'custom', name: 'מותאם אישית', alt: 1000, inc: 30, model: 'starlink', color: COLORS.custom, note: 'מסלול מעגלי שבחרתם בעצמכם.' };
      this.visible = new Set(['starlink', 'geo']);
      this.focus = 'starlink';
    }
    world.clock.time = Date.now();
    world.setWarps([0, 1, 10, 60, 300, 1800, 7200], 60);
    world.controls.minDistance = RE_KM * 1.05;
    world.moonOverride = false;

    this.group = new THREE.Group();
    world.scene.add(this.group);
    this.earthGroup = new THREE.Group();
    world.earth.add(this.earthGroup);
    this.objs = {};

    panel.innerHTML = `
      <h2>מסלולים</h2>
      <p class="lead">ככל שהמסלול גבוה יותר, הלוויין נע לאט יותר, אבל להגיע אליו עולה הרבה יותר. סמנו כמה מסלולים כדי להשוות ביניהם, ולחצו על שם של מסלול כדי לראות אותו בפירוט.</p>
      <div class="presets" id="presets">${Object.values(this.orbits).map(o => `
        <span class="chip ochip" data-id="${o.id}" style="--c:${o.color}">
          <input type="checkbox" aria-label="הצג את ${o.name}"><button class="oname">${o.name}</button></span>`).join('')}</div>
      <div class="cmpwrap"><table class="cmp otable" id="otable"></table></div>
      <p class="small">הלוויינים בתצוגה מוגדלים מאוד, אחרת לא היו נראים: לוויין סטארלינק בגודלו האמיתי, ממרחק כזה, קטן משבריר של פיקסל. המסלולים וכדור הארץ בקנה מידה אמיתי.</p>
      <h3 id="fhead"></h3>
      <p class="note" id="pnote"></p>
      <div id="customctl">
        <div class="ctl"><label>גובה <b id="altv"></b></label><input type="range" id="alt" min="0" max="1000" step="1"></div>
        <div class="ctl"><label>נטייה (זווית מישור המסלול ביחס לקו המשווה) <b id="incv"></b></label><input type="range" id="inc" min="0" max="180" step="0.1"></div>
      </div>
      <div class="cards" id="cards"></div>
      <h3>כמה זה עולה: תקציב דלתא־וי</h3>
      <p class="small">דלתא־וי (Δv, "שינוי מהירות") הוא המטבע של טיסות חלל: סך כל שינויי המהירות שהמנועים צריכים לספק. ההפסדים חושבו בסימולציית שיגור מלאה של פלקון 9.</p>
      <div class="ctl"><label>אתר שיגור</label><select id="site">${SITES.map(s => `<option value="${s.id}">${s.name} (${fmt(s.lat, 1)}°)</option>`).join('')}</select></div>
      <div id="bars"></div>
      <div id="rocketq" class="verdict info"></div>
      <div id="single" class="note"></div>
    `;
    this.panel = panel;
    $('#presets', panel).addEventListener('click', e => {
      const chip = e.target.closest('.ochip'); if (!chip) return;
      const id = chip.dataset.id;
      if (e.target.matches('input')) this.toggle(id, e.target.checked);
      else { this.visible.add(id); this.setFocus(id); this.syncScene(true); }
    });
    $('#otable', panel).addEventListener('click', e => { const th = e.target.closest('[data-id]'); if (th) this.setFocus(th.dataset.id); });
    const c = this.orbits.custom;
    $('#alt', panel).addEventListener('input', e => { c.alt = sliderToAlt(+e.target.value); this.customChanged(); });
    $('#inc', panel).addEventListener('input', e => { c.inc = +e.target.value; this.customChanged(); });
    $('#site', panel).value = this.site.id;
    $('#site', panel).addEventListener('change', e => { this.site = SITES.find(s => s.id === e.target.value); this.refreshTable(); this.refreshFocus(); });
    this.syncScene(true);
    this.setFocus(this.focus);
  },

  toggle(id, on) {
    if (on) this.visible.add(id); else this.visible.delete(id);
    if (!on && this.focus === id) this.focus = [...this.visible][0] ?? null;
    if (on && !this.focus) this.focus = id;
    this.syncScene(true);
    this.setFocus(this.focus);
  },

  customChanged() {
    const c = this.orbits.custom;
    this.visible.add('custom');
    this.focus = 'custom';
    this.rebuildOrbit(c);
    this.setFocus('custom');
  },

  setFocus(id) {
    this.focus = id;
    $$('#presets .ochip', this.panel).forEach(ch => {
      ch.classList.toggle('on', ch.dataset.id === id);
      ch.querySelector('input').checked = this.visible.has(ch.dataset.id);
    });
    this.refreshTable();
    this.refreshFocus();
    this.rebuildFocusDecor();
  },

  // מסגור המצלמה לפי המסלול הגבוה ביותר שמוצג
  frameCamera() {
    const maxAlt = Math.max(400, ...[...this.visible].map(id => this.orbits[id].alt));
    const d = Math.min(Math.max(RE_KM * 3.2, (RE_KM + maxAlt) * 2.6), 1.4e6);
    const cam = this.world.camera;
    const dir = cam.position.clone().normalize();
    if (dir.lengthSq() === 0 || !this.framed) dir.set(0.3, 0.45, 0.85).normalize();
    this.framed = true;
    cam.position.copy(dir.multiplyScalar(d));
    this.world.controls.target.set(0, 0, 0);
  },

  syncScene(reframe) {
    for (const id of Object.keys(this.objs)) if (!this.visible.has(id)) this.removeOrbit(id);
    for (const id of this.visible) if (!this.objs[id]) this.rebuildOrbit(this.orbits[id]);
    this.world.moonOverride = this.visible.has('moon');
    if (reframe) this.frameCamera();
  },

  removeOrbit(id) {
    const o = this.objs[id];
    if (!o) return;
    this.group.remove(o.line); if (o.sat) this.group.remove(o.sat);
    this.world.removeLabel(o.label);
    delete this.objs[id];
  },

  rebuildOrbit(orb) {
    const prevU = this.objs[orb.id]?.u;
    this.removeOrbit(orb.id);
    const { world } = this;
    const r = RE_KM + orb.alt, inc = orb.inc * Math.PI / 180, raan = (RAAN[orb.id] ?? 0) * Math.PI / 180;
    const Pv = [Math.cos(raan), Math.sin(raan), 0];
    const Q = [-Math.cos(inc) * Math.sin(raan), Math.cos(inc) * Math.cos(raan), Math.sin(inc)];
    const pos = u => [r * (Math.cos(u) * Pv[0] + Math.sin(u) * Q[0]), r * (Math.cos(u) * Pv[1] + Math.sin(u) * Q[1]), r * (Math.cos(u) * Pv[2] + Math.sin(u) * Q[2])];
    const pts = [];
    for (let k = 0; k <= 360; k++) pts.push(eciToThree(pos(k * Math.PI / 180)));
    const line = makeLine(pts, new THREE.Color(orb.color).getHex(), 0.5);
    this.group.add(line);
    let sat = null, satSize = 1;
    if (orb.model) {
      sat = buildModel(orb.model);
      satSize = new THREE.Box3().setFromObject(sat).getSize(new THREE.Vector3()).length();
      this.group.add(sat);
    }
    const label = world.addLabel(orb.name, 'big');
    label.el.style.color = orb.color;
    label.visible = !!orb.model;
    // לוויין גאוסטציונרי מתחיל מעל ישראל (35° מזרח) כדי שיהיה קל לראות אותו עומד
    let u = prevU ?? (orb.inc < 0.01 ? P.gmst(new Date(world.clock.time)) + 35 * Math.PI / 180 - raan : 0.6);
    this.objs[orb.id] = { line, sat, satSize, label, pos, u, T: P.periodOf((P.R_EARTH + orb.alt * 1000)) };
    if (orb.id === this.focus) this.rebuildFocusDecor();
  },

  // עקבת קרקע וכתם כיסוי רק למסלול המודגש
  rebuildFocusDecor() {
    for (const o of [this.track, this.foot]) if (o) o.parent?.remove(o);
    this.track = this.foot = null;
    for (const [id, o] of Object.entries(this.objs)) o.line.material.opacity = id === this.focus ? 0.95 : 0.45;
    const orb = this.orbits[this.focus];
    if (!orb || !this.objs[orb.id]) return;
    this.trackPts = [];
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2500 * 3), 3));
    tg.setDrawRange(0, 0);
    this.track = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: orb.color, transparent: true, opacity: 0.85, depthWrite: false }));
    this.track.frustumCulled = false;
    this.earthGroup.add(this.track);
    if (orb.id !== 'moon') {
      const elev = orb.alt < 2000 ? 25 : 10;
      const lam = P.footprintAngle(orb.alt * 1000, elev * Math.PI / 180);
      this.foot = new THREE.Mesh(new THREE.SphereGeometry(RE_KM + 30, 192, 16, 0, Math.PI * 2, 0, lam),
        new THREE.MeshBasicMaterial({ color: orb.color, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide }));
      this.earthGroup.add(this.foot);
    }
  },

  refreshTable() {
    const ids = Object.keys(this.orbits).filter(id => this.visible.has(id));
    const t = $('#otable', this.panel);
    if (!ids.length) { t.innerHTML = '<tr><td class="small">סמנו מסלול אחד או יותר.</td></tr>'; return; }
    const S = ids.map(id => ({ o: this.orbits[id], s: orbitStats(this.orbits[id]), b: dvBudget(this.orbits[id].alt, this.orbits[id].inc, this.site, id === 'moon') }));
    const row = (name, f) => `<tr><td>${name}</td>${S.map(x => `<td style="color:${x.o.color}">${f(x)}</td>`).join('')}</tr>`;
    t.innerHTML = `
      <tr><th></th>${S.map(x => `<th data-id="${x.o.id}" class="${x.o.id === this.focus ? 'fcs' : ''}" style="color:${x.o.color}">${x.o.name}</th>`).join('')}</tr>
      ${row('גובה (ק"מ)', x => fmt(x.o.alt))}
      ${row(`מהירות (${speedUnit().label})`, x => speedNum(x.s.v))}
      ${row('זמן הקפה', x => fmtDur(x.s.T))}
      ${row('השהיה הלוך־חזור', x => fmtRtt(x.s.rtt))}
      ${row('רואה מכדור הארץ', x => x.s.isMoon ? '—' : fmt(x.s.cover * 100, x.s.cover < 0.01 ? 2 : 1) + '%')}
      ${row('חיים בלי הנעה', x => x.s.isMoon ? '—' : fmtLife(x.s.life))}
      ${row('Δv מהקרקע (ק"מ/שנ׳)', x => fmt(x.b.total / 1000, 1))}`;
  },

  refreshFocus() {
    const { panel } = this;
    const orb = this.orbits[this.focus];
    const show = !!orb;
    for (const sel of ['#cards', '#bars', '#rocketq', '#single', '#pnote']) $(sel, panel).style.display = show ? '' : 'none';
    $('#customctl', panel).style.display = 'block';
    const c = this.orbits.custom;
    $('#alt', panel).value = altToSlider(c.alt);
    $('#altv', panel).textContent = `${fmt(c.alt)} ק"מ`;
    $('#inc', panel).value = c.inc;
    $('#incv', panel).textContent = `${fmt(c.inc, 1)}°`;
    $('#fhead', panel).innerHTML = show ? `<span style="color:${orb.color}">●</span> ${orb.name}${orb.id === 'custom' ? '' : ` · ${fmt(orb.alt)} ק"מ`}` : '';
    $('#fhead', panel).insertAdjacentHTML('beforeend', `<span class="small" style="display:block;font-weight:400">${orb?.id === 'custom' ? 'הזיזו את המחוונים כדי לשנות אותו' : 'המחוונים למטה שולטים במסלול "מותאם אישית"'}</span>`);
    if (!show) return;
    $('#pnote', panel).textContent = orb.note;
    const s = orbitStats(orb);
    $('#cards', panel).innerHTML = `
      <div class="card"><div class="k">מהירות מסלולית</div><div class="v">${speedNum(s.v)} <small>${speedUnit().label}</small></div></div>
      <div class="card"><div class="k">זה בערך</div><div class="v">${altSpeedStr(s.v).replace(/ (\S+)$/, ' <small>$1</small>')}</div></div>
      <div class="card"><div class="k">זמן הקפה</div><div class="v">${fmtDur(s.T)}</div></div>
      <div class="card"><div class="k">הקפות ביממה</div><div class="v">${fmt(86400 / s.T, s.T > 86400 * 3 ? 3 : 2)}</div></div>
      <div class="card"><div class="k">השהיה מינימלית הלוך־חזור${s.isMoon ? '' : ' דרך הלוויין'}</div><div class="v">${fmtRtt(s.rtt).replace(' מ"ש', ' <small>מילישנייה</small>')}</div></div>
      <div class="card"><div class="k">משך מעבר בשמיים (מעל 10°)</div><div class="v">${s.sky}</div></div>
      ${s.isMoon ? '' : `<div class="card"><div class="k">חלק מכדור הארץ שרואה אותו (מעל ${s.elev}°)</div><div class="v">${fmt(s.cover * 100, s.cover < 0.01 ? 2 : 1)}%</div></div>
      <div class="card"><div class="k">נשאר במסלול בלי הנעה</div><div class="v">${s.life === Infinity ? 'מאות שנים ויותר' : fmtDur(s.life)}</div></div>`}`;
    this.updateBudget(orb);
  },

  updateBudget(orb) {
    const { panel } = this;
    const b = dvBudget(orb.alt, orb.inc, this.site, orb.id === 'moon');
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
      const pl = P.maxPayload(ROCKETS.falcon9, 200e3, b.extraDv, b.launchInc);
      q.innerHTML = pl < 50
        ? `פלקון 9 (בלי החזרת המאיץ) <b>לא מסוגל</b> להביא ל${orb.name} מטען מ${site.name}.`
        : `פלקון 9 (בלי החזרת המאיץ) יכול להביא ל${orb.name} כ-<b>${fmtMass(pl)}</b> מ${site.name}: <b>${fmt(pl / 549000 * 100, 1)}%</b> ממסת ההמראה של 549 טון. <span class="small">(סימולציה; הערכים המפורסמים: 22.8 טון למסלול נמוך, 8.3 טון למסלול העברה גאוסטציונרי.)</span>`;
    }, 250);
  },

  update(world, dtReal, dtSim) {
    for (const [id, o] of Object.entries(this.objs)) {
      o.u += 2 * Math.PI / o.T * dtSim;
      const pos = eciToThree(o.pos(o.u));
      if (id === 'moon') { world.moon.position.copy(pos); world.orientMoon(); }
      if (o.sat) {
        o.sat.position.copy(pos);
        // הצד ה"תחתון" כלפי כדור הארץ, ציר x בכיוון התנועה
        const vel = eciToThree(o.pos(o.u + 0.001)).sub(pos).normalize();
        const y = pos.clone().normalize();
        const z = new THREE.Vector3().crossVectors(vel, y).normalize();
        const x = new THREE.Vector3().crossVectors(y, z);
        o.sat.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
        // הגדלה כדי שיהיה נראה (מצוין בתווית)
        const s = Math.max(0.001, world.camera.position.distanceTo(pos) * 0.045 * settings.satSize / o.satSize);
        o.sat.scale.setScalar(s);
        o.label.pos.copy(pos);

      }
      if (id !== this.focus || !this.track) continue;
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
    }
  },

  onSettings(k) { if (k === 'speedUnit') { this.refreshTable(); this.refreshFocus(); } },

  exit(world) {
    for (const id of Object.keys(this.objs)) this.removeOrbit(id);
    world.scene.remove(this.group);
    world.earth.remove(this.earthGroup);
    world.moonOverride = false;
    this.track = this.foot = null;
  },
};
