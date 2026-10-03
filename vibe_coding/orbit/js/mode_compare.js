// mode_compare.js — סטארלינק מול לוויין גאוסטציונרי: אותה משימה (אינטרנט), פיזיקה הפוכה
import * as THREE from 'three';
import * as P from './physics.js';
import { PLACES } from './data.js';
import { buildModel } from './models.js';
import { RE_KM, eciToThree, ecefDir, ecefToEci, makeLine } from './world.js';
import { dvBudget } from './mode_orbits.js';
import { $, fmt, h } from './util.js';
import { settings, speedStr } from './settings.js';

// מעטפת אחת של סטארלינק: 72 מישורים × 22 לוויינים, נטייה 53° (המעטפת הראשונה, שהונמכה ב-2026 ל-480 ק"מ)
const PLANES = 72, PER = 22, INC = 53 * Math.PI / 180, ALT = 480, F = 39;
const GEO_SATS = [
  { name: 'עמוס 17 (17° מזרח)', lon: 17 },
  { name: 'לוויין גאו (137° מזרח)', lon: 137 },
  { name: 'לוויין גאו (103° מערב)', lon: -103 },
];

export const compareMode = {
  enter(world, panel) {
    this.world = world;
    world.clock.time = Date.now();
    world.setWarps([0, 1, 10, 60, 300], 10);
    world.moonOverride = false;
    world.controls.minDistance = RE_KM * 1.02;
    this.group = new THREE.Group(); world.scene.add(this.group);
    this.eg = new THREE.Group(); world.earth.add(this.eg);
    this.labels = [];

    // לווייני סטארלינק כנקודות
    const n = PLANES * PER;
    this.slPos = new Float32Array(n * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.slPos, 3));
    this.slPoints = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x7dd3fc, size: 3 * settings.satSize, sizeAttenuation: false, transparent: true, opacity: 0.95, depthWrite: false }));
    this.slPoints.frustumCulled = false;
    this.group.add(this.slPoints);
    this.rS = RE_KM + ALT;
    this.nS = 2 * Math.PI / P.periodOf((this.rS) * 1000);

    // טבעת גאו + שלושה לוויינים (הרעיון של ארתור סי. קלארק, 1945) עם כתמי כיסוי
    const rG = P.GEO_RADIUS / 1000;
    const ring = [];
    for (let k = 0; k <= 256; k++) { const a = k / 256 * Math.PI * 2; ring.push(new THREE.Vector3(Math.cos(a) * rG, 0, -Math.sin(a) * rG)); }
    this.group.add(makeLine(ring, 0xf59e0b, 0.45));
    this.geo = GEO_SATS.map(s => {
      const m = buildModel('geo');
      m.scale.setScalar(25 * settings.satSize);
      const d = ecefDir(0, s.lon);
      m.position.set(d[0] * rG, d[2] * rG, -d[1] * rG);
      // הצד עם האנטנות פונה לכדור הארץ
      m.lookAt(0, 0, 0);
      this.eg.add(m);
      const lam = P.footprintAngle(35786e3, 10 * Math.PI / 180);
      const cap = new THREE.Mesh(new THREE.SphereGeometry(RE_KM + 30, 192, 24, 0, Math.PI * 2, 0, lam),
        new THREE.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.045, depthWrite: false, side: THREE.DoubleSide }));
      cap.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(d[0], d[2], -d[1]));
      this.eg.add(cap);
      // קו גבול הכיסוי
      const rr = RE_KM + 32, edge = [];
      for (let k = 0; k <= 128; k++) { const a = k / 128 * Math.PI * 2; edge.push(new THREE.Vector3(Math.sin(lam) * Math.cos(a) * rr, Math.cos(lam) * rr, Math.sin(lam) * Math.sin(a) * rr)); }
      const el = makeLine(edge, 0xf59e0b, 0.75);
      el.quaternion.copy(cap.quaternion);
      this.eg.add(el);
      const L = world.addLabel(s.name, 'geo'); L.obj = m; this.labels.push(L);
      return { ...s, mesh: m };
    });

    // הצופה בפרדס חנה וקווי התקשורת
    const pl = PLACES.pardesHanna;
    const od = ecefDir(pl.lat, pl.lon);
    this.obsEcef = od;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(35, 16, 8), new THREE.MeshBasicMaterial({ color: 0xff6b6b }));
    dot.position.set(od[0] * RE_KM, od[2] * RE_KM, -od[1] * RE_KM);
    this.eg.add(dot);
    const Lp = world.addLabel(pl.name, 'site'); Lp.obj = dot; this.labels.push(Lp);
    this.lineGeo = makeLine([new THREE.Vector3(), new THREE.Vector3()], 0xf59e0b, 0.9);
    this.lineSL = makeLine([new THREE.Vector3(), new THREE.Vector3()], 0x7dd3fc, 1);
    this.lineGeo.frustumCulled = this.lineSL.frustumCulled = false;
    this.group.add(this.lineGeo, this.lineSL);
    this.slLabel = world.addLabel('', 'leo below');
    this.labels.push(this.slLabel);

    const bS = dvBudget(ALT, 53, { lat: 28.5 });
    const bG = dvBudget(35786, 0, { lat: 28.5 });
    const lamS = P.footprintAngle(ALT * 1e3, 25 * Math.PI / 180), lamG = P.footprintAngle(35786e3, 10 * Math.PI / 180);

    panel.innerHTML = `
      <h2>סטארלינק מול גאוסטציונרי</h2>
      <p class="lead">שתי דרכים לספק אינטרנט מהחלל, עם פיזיקה הפוכה כמעט בכל פרמטר.</p>
      <div class="legend"><span><i style="background:#7dd3fc"></i>סטארלינק, ${fmt(ALT)} ק"מ</span><span><i style="background:#f59e0b"></i>גאוסטציונרי, 35,786 ק"מ</span><span><i style="background:#ff6b6b"></i>${pl.name}</span></div>
      <div class="row" style="margin:8px 0">
        <button class="btn secondary" id="camNear">מקרוב: מעטפת סטארלינק</button>
        <button class="btn secondary" id="camFar">מרחוק: מסלול הגאו</button>
      </div>
      <h3>מה קורה עכשיו מעל ${pl.name}</h3>
      <div class="cards" id="live"></div>
      <h3>השוואה</h3>
      <table class="cmp">
        <tr><th></th><th>סטארלינק</th><th>גאוסטציונרי</th></tr>
        <tr><td>גובה</td><td class="leo">480–560 ק"מ</td><td class="geo">35,786 ק"מ (פי 70)</td></tr>
        <tr><td>מהירות</td><td class="leo" id="spdS">${speedStr(P.circularSpeed(this.rS * 1e3))}</td><td class="geo" id="spdG">${speedStr(P.circularSpeed(P.GEO_RADIUS))}</td></tr>
        <tr><td>הקפה</td><td class="leo">${fmt(P.periodOf(this.rS * 1e3) / 60, 0)} דקות</td><td class="geo">23:56 שעות, כמו סיבוב כדור הארץ</td></tr>
        <tr><td>בשמיים</td><td class="leo">חוצה את השמיים בדקות; המשתמש עובר מלוויין ללוויין כל הזמן</td><td class="geo">עומד במקום</td></tr>
        <tr><td>אנטנה בבית</td><td class="leo">מערך מופע שמכוון את האלומה אלקטרונית</td><td class="geo">צלחת קבועה שמכוונים פעם אחת</td></tr>
        <tr><td>השהיה פיזיקלית מינימלית, הלוך־חזור</td><td class="leo">${fmt(4 * ALT * 1e3 / P.C_LIGHT * 1000, 1)} מילישניות</td><td class="geo">${fmt(4 * 35786e3 / P.C_LIGHT * 1000, 0)} מילישניות</td></tr>
        <tr><td>השהיה בפועל (טיפוסית)</td><td class="leo">כ-25–40 מ"ש</td><td class="geo">כ-600 מ"ש ומעלה</td></tr>
        <tr><td>שטח שלוויין אחד רואה</td><td class="leo">${fmt(P.capFraction(lamS) * 100, 2)}% מכדור הארץ (מעל 25°)</td><td class="geo">${fmt(P.capFraction(lamG) * 100, 0)}% (מעל 10°)</td></tr>
        <tr><td>כמה לוויינים צריך</td><td class="leo">אלפים: כ-11,100 פעילים בספטמבר 2026</td><td class="geo">3 לכיסוי כמעט מלא (חוץ מהקטבים)</td></tr>
        <tr><td>מסת לוויין</td><td class="leo">כ-575–800 ק"ג (V2 Mini)</td><td class="geo">3–6.5 טון</td></tr>
        <tr><td>Δv מהקרקע (מקייפ קנוורל)</td><td class="leo">${fmt(bS.total / 1000, 1)} ק"מ/שנ׳</td><td class="geo">${fmt(bG.total / 1000, 1)} ק"מ/שנ׳</td></tr>
        <tr><td>שיגור פלקון 9</td><td class="leo">כ-22–29 לוויינים בבת אחת</td><td class="geo">לרוב לוויין אחד, למסלול העברה</td></tr>
        <tr><td>סוף החיים</td><td class="leo">כ-5 שנים, ואז נשרף באטמוספרה</td><td class="geo">15 שנה ויותר, ואז מועבר ל"מסלול בית קברות" כ-300 ק"מ גבוה יותר</td></tr>
      </table>
      <h3>למה גאו כל כך יקר להגיע אליו?</h3>
      <p>להגיע ל-480 ק"מ דורש בערך ${fmt(bS.total / 1000, 1)} ק"מ לשנייה של שינוי מהירות. למסלול גאוסטציונרי צריך עוד שתי הבערות: אחת שמותחת את המסלול לאליפסה שהשיא שלה ב-35,786 ק"מ (מסלול העברה, GTO), ואחת בשיא, שמעגלת אותו ובאותה הזדמנות מבטלת את הנטייה של ${fmt(28.5, 1)}° של קייפ קנוורל. בסך הכול כ-${fmt(bG.total / 1000, 1)} ק"מ לשנייה.</p>
      <p>בגלל משוואת הרקטה המחיר הזה אקספוננציאלי: פלקון 9 מעלה כ-22.8 טון למסלול נמוך, אבל רק 8.3 טון למסלול ההעברה, ואת ההבערה האחרונה הלוויין עושה בעצמו, עם מנוע ודלק משלו שתופסים חלק ניכר ממסתו.</p>
      <h3>ולמה סטארלינק צריך כל כך הרבה לוויינים?</h3>
      <p>לוויין נמוך רואה רק פיסה קטנה מכדור הארץ, והוא נע מעליה במהירות של כ-27,000 קמ"ש. כדי שתמיד יהיה לוויין מעליך צריך לרשת את כל השמיים בלוויינים. בתמורה האות עובר מרחק קצר פי 70, וההשהיה נמוכה מספיק לשיחות וידאו ולמשחקים.</p>
      <p class="note">הנקודות התכולות הן מעטפת אחת בלבד (1,584 לוויינים, 72 מישורים × 22, נטייה 53°), במסלולים אמיתיים בחישוב קפלר. הקונסטלציה המלאה כוללת עוד מעטפות בנטיות 43°, 53.2°, 70° ו-97.6°.</p>
      <p class="small">שני הלוויינים הגאוסטציונריים הנוספים מוצבים כדי להמחיש את רעיון שלושת הלוויינים של ארתור סי. קלארק (1945). עמוס 17 של חלל־תקשורת נמצא באמת ב-17° מזרח.</p>
    `;
    $('#camNear', panel).onclick = () => this.cam('near');
    $('#camFar', panel).onclick = () => this.cam('far');
    this.panel = panel;
    this.cam('far');
  },

  cam(which) {
    const w = this.world;
    const theta = w.earth.rotation.y;
    const o = ecefToEci(this.obsEcef, theta);
    const ov = eciToThree(o);
    w.controls.target.set(0, 0, 0);
    if (which === 'near') w.camera.position.copy(ov).multiplyScalar(RE_KM * 2.3).add(new THREE.Vector3(0, 2500, 0));
    else w.camera.position.copy(ov).multiplyScalar(130000).add(new THREE.Vector3(0, 45000, 0));
  },

  update(world, dtReal, dtSim) {
    const t = world.clock.time / 1000;
    const pos = this.slPos;
    const theta = world.earth.rotation.y;
    // מיקום הצופה ב-ECI (ק"מ)
    const o = ecefToEci(this.obsEcef, theta).map(x => x * RE_KM);
    const oN = ecefToEci(this.obsEcef, theta);
    let best = null;
    let k = 0;
    for (let p = 0; p < PLANES; p++) {
      const raan = p / PLANES * 2 * Math.PI;
      const cO = Math.cos(raan), sO = Math.sin(raan);
      for (let s = 0; s < PER; s++, k++) {
        const u = this.nS * t + s / PER * 2 * Math.PI + p * F * 2 * Math.PI / (PLANES * PER);
        const cu = Math.cos(u), su = Math.sin(u);
        const x = this.rS * (cO * cu - sO * su * Math.cos(INC));
        const y = this.rS * (sO * cu + cO * su * Math.cos(INC));
        const z = this.rS * (su * Math.sin(INC));
        pos[k * 3] = x; pos[k * 3 + 1] = z; pos[k * 3 + 2] = -y;
        // זווית הגבהה מעל האופק של הצופה
        const dx = x - o[0], dy = y - o[1], dz = z - o[2];
        const d = Math.hypot(dx, dy, dz);
        const sinEl = (dx * oN[0] + dy * oN[1] + dz * oN[2]) / d;
        if (sinEl > Math.sin(25 * Math.PI / 180) && (!best || sinEl > best.sinEl)) best = { sinEl, d, x, y, z };
      }
    }
    this.slPoints.geometry.attributes.position.needsUpdate = true;
    const ov = eciToThree(o);
    // קו לעמוס 17
    const amos = this.geo[0].mesh.getWorldPosition(new THREE.Vector3());
    this.lineGeo.geometry.setFromPoints([ov, amos]);
    const dGeo = ov.distanceTo(amos);
    const elGeo = Math.asin(amos.clone().sub(ov).normalize().dot(ov.clone().normalize())) * 180 / Math.PI;
    if (best) {
      const sv = eciToThree([best.x, best.y, best.z]);
      this.lineSL.geometry.setFromPoints([ov, sv]);
      this.lineSL.visible = true;
      this.slLabel.pos.copy(sv); this.slLabel.visible = true;
      this.slLabel.el.textContent = 'לוויין סטארלינק מעליך';
    } else { this.lineSL.visible = false; this.slLabel.visible = false; }
    // תצוגה חיה (מתעדכנת כמה פעמים בשנייה)
    this.acc = (this.acc ?? 0) + dtReal;
    if (this.acc > 0.25) {
      this.acc = 0;
      const ms = d => fmt(2 * d * 1000 / P.C_LIGHT * 1000, d > 5000 ? 0 : 1);
      $('#live', this.panel).innerHTML = `
        <div class="card"><div class="k">מרחק ללוויין הסטארלינק הגבוה בשמיים</div><div class="v" style="color:var(--leo)">${best ? fmt(best.d) + ' <small>ק"מ</small>' : '—'}</div></div>
        <div class="card"><div class="k">מרחק לעמוס 17</div><div class="v" style="color:var(--geo)">${fmt(dGeo)} <small>ק"מ</small></div></div>
        <div class="card"><div class="k">זמן לאות: אליו וממנו לקרקע</div><div class="v" style="color:var(--leo)">${best ? ms(best.d) + ' <small>מ"ש</small>' : '—'}</div></div>
        <div class="card"><div class="k">זמן לאות: אליו וממנו לקרקע</div><div class="v" style="color:var(--geo)">${ms(dGeo)} <small>מ"ש</small></div></div>
        <div class="card"><div class="k">הגבהה מעל האופק</div><div class="v" style="color:var(--leo)">${best ? fmt(Math.asin(best.sinEl) * 180 / Math.PI, 0) + '°' : '—'}</div></div>
        <div class="card"><div class="k">הגבהה מעל האופק</div><div class="v" style="color:var(--geo)">${fmt(elGeo, 0)}° <small>תמיד</small></div></div>`;
    }
  },

  onSettings(k) {
    if (k === 'speedUnit') {
      $('#spdS', this.panel).textContent = speedStr(P.circularSpeed(this.rS * 1e3));
      $('#spdG', this.panel).textContent = speedStr(P.circularSpeed(P.GEO_RADIUS));
    }
    if (k === 'satSize') {
      this.slPoints.material.size = 3 * settings.satSize;
      this.geo.forEach(g => g.mesh.scale.setScalar(25 * settings.satSize));
    }
  },

  exit(world) {
    world.scene.remove(this.group);
    world.earth.remove(this.eg);
    this.labels.forEach(L => world.removeLabel(L));
  },
};
