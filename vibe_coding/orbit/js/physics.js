// physics.js — קבועים, מכניקה מסלולית, אטמוספרה וסימולציית שיגור.
// כל היחידות SI (מטר, שנייה, ק"ג) אלא אם כתוב אחרת. המודול רץ גם בדפדפן וגם ב-node.

export const MU = 3.986004418e14;          // GM של כדור הארץ, m^3/s^2 (WGS-84)
export const R_EARTH = 6378137;            // רדיוס משווני, m
export const R_MEAN = 6371000;             // רדיוס ממוצע, m
export const OMEGA_EARTH = 7.2921159e-5;   // מהירות סיבוב זוויתית (יום כוכבי), rad/s
export const SIDEREAL_DAY = 86164.0905;    // s
export const G0 = 9.80665;                 // תאוצת כובד תקנית, m/s^2
export const C_LIGHT = 299792458;          // m/s
export const MU_MOON = 4.9048695e12;       // m^3/s^2
export const R_MOON = 1737400;             // m
export const MOON_DIST = 384400e3;         // מרחק ממוצע מרכז־מרכז, m
export const GEO_RADIUS = Math.cbrt(MU * (SIDEREAL_DAY / (2 * Math.PI)) ** 2); // ≈ 42,164 km

// ---------- מסלולים מעגליים ואליפטיים ----------

export const circularSpeed = r => Math.sqrt(MU / r);
export const periodOf = a => 2 * Math.PI * Math.sqrt(a ** 3 / MU);
export const visViva = (r, a) => Math.sqrt(MU * (2 / r - 1 / a));
export const escapeSpeed = r => Math.sqrt(2 * MU / r);

// העברת הוהמן בין שני מסלולים מעגליים
export function hohmann(r1, r2) {
  const a = (r1 + r2) / 2;
  const dv1 = visViva(r1, a) - circularSpeed(r1);
  const dv2 = circularSpeed(r2) - visViva(r2, a);
  return { a, dv1, dv2, total: dv1 + dv2, tof: periodOf(a) / 2 };
}

// הבערה משולבת: הגעה למהירות מעגלית ושינוי נטייה באותה הבערה (חוק הקוסינוסים)
export function combinedBurn(v1, v2, dInc) {
  return Math.sqrt(v1 * v1 + v2 * v2 - 2 * v1 * v2 * Math.cos(dInc));
}

// משוואת הרקטה של ציולקובסקי
export const rocketDv = (isp, m0, m1) => isp * G0 * Math.log(m0 / m1);
export const massRatioFor = (dv, isp) => Math.exp(dv / (isp * G0));

// כיסוי: זווית מרכזית של "כתם" הכיסוי מלוויין בגובה h, לזווית הגבהה מינימלית eps
export function footprintAngle(h, epsRad) {
  const r = R_MEAN + h;
  return Math.acos((R_MEAN / r) * Math.cos(epsRad)) - epsRad;
}
// חלק משטח כדור הארץ שבכתם
export const capFraction = lam => (1 - Math.cos(lam)) / 2;
// טווח נטוי מהקרקע ללוויין בזווית הגבהה eps
export function slantRange(h, epsRad) {
  const r = R_MEAN + h;
  return Math.sqrt(r * r - (R_MEAN * Math.cos(epsRad)) ** 2) - R_MEAN * Math.sin(epsRad);
}

// ---------- אטמוספרה: US Standard Atmosphere 1976 ----------
// שכבות עד 86 ק"מ (גובה גאופוטנציאלי), ומעליהן טבלה של צפיפות.
const LAYERS = [ // [h_base m, T_base K, lapse K/m, P_base Pa]
  [0, 288.15, -0.0065, 101325],
  [11000, 216.65, 0, 22632.06],
  [20000, 216.65, 0.001, 5474.889],
  [32000, 228.65, 0.0028, 868.0187],
  [47000, 270.65, 0, 110.9063],
  [51000, 270.65, -0.0028, 66.93887],
  [71000, 214.65, -0.002, 3.956420],
  [84852, 186.946, 0, 0.3733836],
];
// צפיפות מעל 86 ק"מ (מתוך טבלאות 1976), [h km, rho kg/m^3]
const HIGH = [
  [86, 6.958e-6], [90, 3.416e-6], [100, 5.604e-7], [110, 9.708e-8], [120, 2.222e-8],
  [130, 8.152e-9], [140, 3.831e-9], [150, 2.076e-9], [160, 1.233e-9], [180, 5.194e-10],
  [200, 2.541e-10], [250, 6.073e-11], [300, 1.916e-11], [350, 7.014e-12], [400, 2.803e-12],
  [450, 1.184e-12], [500, 5.215e-13], [600, 1.137e-13], [700, 3.070e-14], [800, 1.136e-14],
  [900, 5.759e-15], [1000, 3.561e-15],
];
const R_AIR = 287.053, GAMMA = 1.4;

export function atmosphere(hGeom) {
  if (hGeom < 0) hGeom = 0;
  if (hGeom >= 86000) {
    const hk = hGeom / 1000;
    if (hk >= 1000) return { rho: 0, p: 0, T: 1000, a: 600 };
    let i = 0;
    while (HIGH[i + 1][0] < hk) i++;
    const [h0, r0] = HIGH[i], [h1, r1] = HIGH[i + 1];
    const rho = r0 * Math.exp(Math.log(r1 / r0) * (hk - h0) / (h1 - h0));
    return { rho, p: rho * R_AIR * 200, T: 186.87, a: 274 };
  }
  const h = R_MEAN * hGeom / (R_MEAN + hGeom); // גאופוטנציאלי
  let L = LAYERS[0];
  for (const l of LAYERS) if (h >= l[0]) L = l;
  const [hb, Tb, lap, Pb] = L;
  const T = Tb + lap * (h - hb);
  const p = lap === 0
    ? Pb * Math.exp(-G0 * (h - hb) / (R_AIR * Tb))
    : Pb * Math.pow(Tb / T, G0 / (R_AIR * lap));
  return { rho: p / (R_AIR * T), p, T, a: Math.sqrt(GAMMA * R_AIR * T) };
}

// מקדם גרר טיפוסי לגוף רקטה ארוך כפונקציה של מספר מאך
export function dragCoeff(M) {
  const tbl = [[0, 0.30], [0.8, 0.32], [1.0, 0.45], [1.2, 0.50], [1.6, 0.45], [2.5, 0.37], [4, 0.30], [6, 0.27], [10, 0.25]];
  if (M >= 10) return 0.25;
  let i = 0;
  while (tbl[i + 1][0] < M) i++;
  const [m0, c0] = tbl[i], [m1, c1] = tbl[i + 1];
  return c0 + (c1 - c0) * (M - m0) / (m1 - m0);
}

// ---------- סימולציית שיגור ----------
// הרקטה מתוארת כרשימת שלבים. לכל שלב: מנוע מרכזי (core) ואופציונלית מאיצי צד (boosters)
// שבוערים במקביל ונזרקים כשנגמר להם הדלק. מנוע: thrustSL/thrustVac [N], ispSL/ispVac [s].
// ספיקת המסה נגזרת מהדחף בוואקום ומה-Isp בוואקום, והדחף בגובה נמוך יורד בגלל לחץ חיצוני.

function engineState(e, p) {
  const mdot = e.thrustVac / (e.ispVac * G0);
  const Ae = (e.thrustVac - e.thrustSL) / 101325; // שטח יציאה אפקטיבי
  return { mdot, thrust: Math.max(0, e.thrustVac - Ae * p) };
}

/*
  opts: {
    rocket, payload [kg], targetAlt [m], latitude [deg],
    kick [deg]  — זווית הטיה ראשונית (אם חסר, נמצאת בחיפוש),
    dt, record (bool)
  }
  מחזיר: מצב סופי, לוג טלמטריה, ותקציב דלתא־וי מפורק.
  המודל דו־ממדי, במישור המסלול, עם שיגור מזרחה מקו רוחב נתון (הנטייה = קו הרוחב).
*/
export function simulateAscent(opts) {
  const { rocket, payload, targetAlt, record = true } = opts;
  const dt = opts.dt ?? 0.1;
  // במישור המסלול, תרומת סיבוב כדור הארץ היא ωR·cos(i) (בשיגור לנטייה i ≥ קו הרוחב)
  const lat = (opts.inclination ?? rocket.site.lat) * Math.PI / 180;
  const kick = (opts.kick ?? rocket.kick) * Math.PI / 180;
  const rT = R_EARTH + targetAlt;
  const wAir = OMEGA_EARTH * Math.cos(lat); // סיבוב האטמוספרה במישור המסלול (קירוב)

  // בניית מצב המסה
  const stages = rocket.stages.map(s => ({
    ...s,
    coreProp: s.core.prop,
    boosterProp: s.boosters ? s.boosters.prop * s.boosters.count : 0,
    boostersOn: !!s.boosters,
  }));
  let si = 0;
  let fairingOn = !!rocket.fairing;
  let lesOn = !!rocket.les;
  const massNow = () => {
    let m = payload + (fairingOn ? rocket.fairing.mass : 0) + (lesOn ? rocket.les.mass : 0);
    for (let k = si; k < stages.length; k++) {
      const s = stages[k];
      m += s.core.dry + s.coreProp;
      if (s.boostersOn) m += s.boosters.dry * s.boosters.count + s.boosterProp;
    }
    return m;
  };

  // מצב קינמטי: קואורדינטות קוטביות במישור, אינרציאלי
  let r = R_EARTH, theta = 0, vr = 0, vt = OMEGA_EARTH * Math.cos(lat) * R_EARTH;
  const v0 = vt;
  let t = 0;
  const log = [];
  const events = [];
  const loss = { gravity: 0, drag: 0, steering: 0, ideal: 0 };
  let maxQ = 0, maxQt = 0, maxG = 0;
  let pitch = Math.PI / 2; // זווית הדחף מעל האופק המקומי
  let phase = 'ascent';
  let coastUntil = 0;
  let result = null;
  let lastLog = -1;

  const ev = (name, extra = {}) => events.push({ t, name, alt: r - R_EARTH, v: Math.hypot(vr, vt), ...extra });

  // ביצוע הנחיה במעגל סגור לשלבים העליונים: מתכננים תאוצה רדיאלית שמביאה
  // את הגובה ואת המהירות האנכית ליעד בדיוק כשהמהירות האופקית מגיעה למעגלית.
  function closedLoopPitch(aThrust, mdot, m) {
    const vc = Math.sqrt(MU / rT);
    const dvNeed = Math.hypot(vc - vt, vr);
    const ve = aThrust * m / mdot;
    let T = (m / mdot) * (1 - Math.exp(-dvNeed / ve));
    T = Math.max(T, 8);
    const a0 = 6 * (rT - r) / (T * T) - 4 * vr / T;
    const need = a0 + MU / (r * r) - vt * vt / r;
    return Math.asin(Math.max(-0.75, Math.min(0.95, need / aThrust)));
  }

  const maxT = 4000;
  while (t < maxT) {
    const h = r - R_EARTH;
    const atm = atmosphere(h);
    const st = stages[si];
    if (!st) { result = 'no-stages'; break; }

    // דחף וספיקה של כל המנועים הפעילים
    let F = 0, mdot = 0, mdotCore = 0, mdotB = 0;
    const throttle = st.throttle ?? 1;
    if (phase !== 'coast' && st.coreProp > 0 && t >= (st.coreDelay ?? 0)) {
      const e = engineState(st.core, atm.p);
      F += e.thrust * throttle; mdotCore = e.mdot * throttle;
    }
    if (st.boostersOn && st.boosterProp > 0) {
      const e = engineState(st.boosters, atm.p);
      F += e.thrust * st.boosters.count; mdotB = e.mdot * st.boosters.count;
    }
    let m = massNow();
    // הגבלת תאוצה (מצערת) אם הוגדרה
    if (st.maxG && F / m > st.maxG * G0) {
      const scale = Math.max(st.minThrottle ?? 0.4, st.maxG * G0 * m / F);
      F *= scale; mdotCore *= scale; mdotB *= scale;
    }
    // כיבוי מנוע מרכזי מתוכנן (למשל CECO בסטורן 5)
    if (st.cutAt && t >= st.cutAt.t) { F *= st.cutAt.factor; mdotCore *= st.cutAt.factor; mdotB *= st.cutAt.factor; }
    mdot = mdotCore + mdotB;

    // מהירות יחסית לאוויר
    const vAirT = vt - wAir * r;
    const vRel = Math.hypot(vr, vAirT);
    const q = 0.5 * atm.rho * vRel * vRel;
    const M = vRel / atm.a;
    const D = q * dragCoeff(M) * rocket.area;
    if (q > maxQ) { maxQ = q; maxQt = t; }

    // הנחיה
    if (phase === 'ascent') {
      if (si === 0) {
        if (t < rocket.vertical) pitch = Math.PI / 2;
        else if (t < rocket.vertical + 10) pitch = Math.PI / 2 - kick * (t - rocket.vertical) / 10;
        else {
          // סיבוב כובד: הדחף בכיוון המהירות יחסית לאוויר (זווית התקפה אפס)
          const fpa = Math.atan2(vr, vAirT);
          pitch = Math.min(pitch, fpa);
        }
      } else if (F > 0) {
        pitch = closedLoopPitch(F / m, mdot, m);
      }
    }

    // תאוצות
    const ux = Math.cos(pitch), uy = Math.sin(pitch); // x = אופקי (משיקי), y = רדיאלי
    const aT = F / m;
    const dragT = vRel > 0 ? -D / m * vAirT / vRel : 0;
    const dragR = vRel > 0 ? -D / m * vr / vRel : 0;
    const g = MU / (r * r);
    const ar = aT * uy + dragR - g + vt * vt / r;
    const at = aT * ux + dragT - vr * vt / r;
    const gLoad = Math.hypot(aT * ux + dragT, aT * uy + dragR) / G0;
    if (gLoad > maxG) maxG = gLoad;

    // הפסדים (אינטגרלים)
    if (F > 0) {
      const vmag = Math.hypot(vr, vt);
      const cosA = vmag > 0 ? (ux * vt + uy * vr) / vmag : 1;
      loss.ideal += aT * dt;
      loss.steering += aT * (1 - cosA) * dt;
      loss.gravity += g * (vr / Math.max(vmag, 1)) * dt;
      loss.drag += D / m * dt;
    }

    // רישום
    if (record && t - lastLog >= 1) {
      lastLog = t;
      log.push({ t, alt: h, theta, vr, vt, v: Math.hypot(vr, vt), vRel, m, F, q, g: gLoad, pitch, stage: si, M });
    }

    // אינטגרציה (אוילר סמי־אימפליציטי, צעד קטן)
    vr += ar * dt; vt += at * dt;
    r += vr * dt; theta += vt / r * dt;
    st.coreProp -= mdotCore * dt;
    st.boosterProp -= mdotB * dt;
    t += dt;

    if (r < R_EARTH - 1) { result = 'crash'; ev('impact'); break; }

    // אירועים
    if (st.boostersOn && st.boosterProp <= 0) {
      st.boostersOn = false; st.boosterProp = 0;
      ev('boosterSep', { stage: si });
    }
    if (fairingOn && h > (rocket.fairing.sepAlt ?? 110000)) { fairingOn = false; ev('fairing'); }
    if (lesOn && si >= 1 && t > (rocket.les.sepAfter ?? 0)) { lesOn = false; ev('les'); }

    // כיבוי לפי אנרגיה: כשחצי הציר הראשי הגיע ליעד ושיא המסלול קרוב
    const v2 = vr * vr + vt * vt;
    const a = 1 / (2 / r - v2 / MU);
    if (si > 0 && phase === 'ascent' && a >= rT - 200 && F > 0) {
      ev('orbit');
      result = 'orbit';
      break;
    }

    if (st.coreProp <= 0 && !(st.boostersOn && st.boosterProp > 0)) {
      st.coreProp = 0;
      ev('stageSep', { stage: si });
      si++;
      if (si >= stages.length) {
        result = 'burnout';
        break;
      }
      // בשלב העליון, התחלה מחודשת של הנחיה לפי זווית הטיסה הנוכחית
    }
  }

  // מסלול סופי
  const v2 = vr * vr + vt * vt;
  const a = 1 / (2 / r - v2 / MU);
  const hAng = r * vt;
  const e = Math.sqrt(Math.max(0, 1 - hAng * hAng / (MU * a)));
  const peri = a * (1 - e) - R_EARTH, apo = a * (1 + e) - R_EARTH;
  const inOrbit = a > 0 && peri > 120000;
  // דלק שנשאר בשלב העליון
  const up = stages[Math.min(si, stages.length - 1)];
  const lastStage = stages[stages.length - 1];
  return {
    result: inOrbit ? 'orbit' : (result === 'orbit' ? 'orbit' : 'fail'),
    t, r, theta, vr, vt, v: Math.sqrt(v2), a, e, peri, apo,
    mass: massNow(),
    stageIndex: si,
    upperPropLeft: si < stages.length ? up.coreProp : 0,
    upperStage: lastStage,
    upperDry: lastStage.core.dry,
    loss, maxQ, maxQt, maxG, events, log, v0,
    kick: kick * 180 / Math.PI,
  };
}

// חיפוש זווית הטיה שממזערת את הדלק הנצרך (ממקסמת את הדלק שנשאר בשלב העליון)
export function optimizeKick(rocket, payload, targetAlt, latitude) {
  let best = null;
  const tryK = k => {
    const res = simulateAscent({ rocket, payload, targetAlt, inclination: latitude, kick: k, record: false, dt: 0.25 });
    const score = res.result === 'orbit' ? res.upperPropLeft + 1e7 : -Math.abs(res.peri - targetAlt) + res.apo * 0;
    if (!best || score > best.score) best = { k, score, res };
    return score;
  };
  for (let k = 1; k <= 30; k += 1.5) tryK(k);
  // עידון סביב הטוב ביותר
  const k0 = best.k;
  for (let k = k0 - 1.5; k <= k0 + 1.5; k += 0.25) if (k > 0.2) tryK(k);
  return best.k;
}

// כמה דלק נשאר לשלב העליון אחרי הכנסה למסלול חניה, וכמה דלתא־וי זה נותן
export function upperStageDv(res, payload) {
  const st = res.upperStage;
  const isp = st.core.ispVac;
  const m0 = st.core.dry + res.upperPropLeft + payload;
  const m1 = st.core.dry + payload;
  return rocketDv(isp, m0, m1);
}

// מטען מקסימלי למסלול נתון (חיפוש בינארי). target: {alt, extraDv} — extraDv מבוצע ע"י השלב העליון אחרי מסלול החניה
export function maxPayload(rocket, parkAlt, extraDv = 0, latitude) {
  let lo = 0, hi = rocket.maxPayloadGuess ?? 200000;
  for (let i = 0; i < 18; i++) {
    const mid = (lo + hi) / 2;
    const kick = optimizeKick(rocket, mid, parkAlt, latitude);
    const res = simulateAscent({ rocket, payload: mid, targetAlt: parkAlt, inclination: latitude, kick, record: false, dt: 0.25 });
    const ok = res.result === 'orbit' && (extraDv <= 0 || upperStageDv(res, mid) >= extraDv);
    if (ok) lo = mid; else hi = mid;
  }
  return lo;
}

// ---------- אסטרונומיה בסיסית: מיקום השמש והירח ----------
// אלגוריתמים בדיוק נמוך (Astronomical Almanac / Meeus), מספיקים לתאורה ולתצוגה.

export function julianDate(date) { return date.getTime() / 86400000 + 2440587.5; }

// זמן כוכבי בגריניץ' (רדיאנים)
export function gmst(date) {
  const d = julianDate(date) - 2451545.0;
  const deg = 280.46061837 + 360.98564736629 * d;
  return ((deg % 360) + 360) % 360 * Math.PI / 180;
}

// כיוון השמש במערכת משוונית (ECI), וקטור יחידה [x,y,z]
export function sunDirECI(date) {
  const n = julianDate(date) - 2451545.0;
  const L = (280.460 + 0.9856474 * n) * Math.PI / 180;
  const gA = (357.528 + 0.9856003 * n) * Math.PI / 180;
  const lam = L + (1.915 * Math.sin(gA) + 0.020 * Math.sin(2 * gA)) * Math.PI / 180;
  const eps = (23.439 - 0.0000004 * n) * Math.PI / 180;
  return [Math.cos(lam), Math.cos(eps) * Math.sin(lam), Math.sin(eps) * Math.sin(lam)];
}

// מיקום הירח במערכת משוונית (מטרים), דיוק של כמה עשיריות מעלה
export function moonPosECI(date) {
  const T = (julianDate(date) - 2451545.0) / 36525;
  const d2r = Math.PI / 180;
  const L0 = 218.316 + 481267.881 * T;
  const Mm = 134.963 + 477198.867 * T;
  const Ms = 357.529 + 35999.050 * T;
  const D = 297.850 + 445267.111 * T;
  const F = 93.272 + 483202.018 * T;
  const lon = L0 + 6.289 * Math.sin(Mm * d2r) - 1.274 * Math.sin((Mm - 2 * D) * d2r)
    + 0.658 * Math.sin(2 * D * d2r) - 0.214 * Math.sin(2 * Mm * d2r) - 0.186 * Math.sin(Ms * d2r)
    - 0.114 * Math.sin(2 * F * d2r);
  const lat = 5.128 * Math.sin(F * d2r) + 0.281 * Math.sin((Mm + F) * d2r)
    - 0.277 * Math.sin((Mm - F) * d2r) - 0.173 * Math.sin((F - 2 * D) * d2r);
  const dist = 385001 - 20905 * Math.cos(Mm * d2r) - 3699 * Math.cos((2 * D - Mm) * d2r) - 2956 * Math.cos(2 * D * d2r);
  const l = lon * d2r, b = lat * d2r, eps = 23.439 * d2r;
  const x = Math.cos(b) * Math.cos(l), y = Math.cos(b) * Math.sin(l), z = Math.sin(b);
  return [x * dist * 1000, (y * Math.cos(eps) - z * Math.sin(eps)) * dist * 1000, (y * Math.sin(eps) + z * Math.cos(eps)) * dist * 1000];
}

// ---------- מסלול כללי: מאלמנטים קפלריים למיקום ----------
// מחזיר מיקום ECI במטרים, עבור a, e, i, raan, argp, M (רדיאנים)
export function keplerToECI(a, e, i, raan, argp, M) {
  let E = M;
  for (let k = 0; k < 12; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const nu = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E / 2), Math.sqrt(1 - e) * Math.cos(E / 2));
  const rr = a * (1 - e * Math.cos(E));
  const xp = rr * Math.cos(nu), yp = rr * Math.sin(nu);
  const cO = Math.cos(raan), sO = Math.sin(raan), cw = Math.cos(argp), sw = Math.sin(argp), ci = Math.cos(i), si = Math.sin(i);
  return [
    xp * (cO * cw - sO * sw * ci) - yp * (cO * sw + sO * cw * ci),
    xp * (sO * cw + cO * sw * ci) - yp * (sO * sw - cO * cw * ci),
    xp * (sw * si) + yp * (cw * si),
  ];
}

// ---------- דעיכת מסלול בגלל גרר ----------
// da/dt = -ρ·sqrt(μa)/B, כש-B = m/(Cd·A) מקדם בליסטי [kg/m^2]. מחזיר שניות עד ירידה ל-120 ק"מ.
// מודל גס: אטמוספרה סטנדרטית (פעילות שמש בינונית). בשיא מחזור השמש הדעיכה מהירה בהרבה.
export function decayTime(altM, B = 100) {
  let h = altM, t = 0;
  if (h > 1000e3) return Infinity;
  while (h > 120e3) {
    const dh = Math.max(200, h * 0.002);
    const a = R_EARTH + h;
    const rate = atmosphere(h).rho * Math.sqrt(MU * a) / B;
    if (rate <= 0) return Infinity;
    t += dh / rate;
    if (t > 3.15e7 * 1e5) return Infinity;
    h -= dh;
  }
  return t;
}
