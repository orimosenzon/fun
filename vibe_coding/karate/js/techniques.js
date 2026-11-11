// techniques.js: התנוחות והטכניקות. קוד טהור.
// כל התנוחות מוגדרות לעמידת שמאל-קדימה (הידארי); עמידת ימין מתקבלת בשיקוף.
// טכניקה היא רצף תנוחות מפתח בזמנים נתונים, במערכת הצירים של תחילת הטכניקה.
// advance הוא המרחק שהלוחם התקדם בסופה, sw אומר שהעמידה התחלפה.

import { D2R, Ry, mv, add, sub, chestOf, clonePose, mirrorPose, shiftPose, solve, lerpPose } from './body.js';

const d = (...a) => a.map((x) => x * D2R);
const PI = Math.PI;

// גובהי מטרה
const Y_JODAN = 1.50, Y_CHUDAN = 1.18;

function hikite(p, s) {
  // היקיטה: האגרוף נמשך אל המותן, כף היד למעלה, המרפק לאחור
  const sg = s === 'L' ? -1 : 1;
  const R = Ry(p.rot[0]);
  const abs = add(p.pel, mv(R, [-0.03, 0.20, sg * 0.20]));
  p['h' + s] = sub(abs, chestOf(p));
  p['e' + s] = mv(R, [-1, -0.15, sg * 0.25]);
  p['g' + s] = [PI, 0];
}

// בניית תנוחה מתנוחת בסיס. hLabs/hRabs: מיקום שורש כף היד במערכת הלוחם (לא יחסית לחזה).
// hLbody/hRbody: יחסית לחזה, במערכת הגוף המסובב (קדימה, למעלה, ימינה).
function mk(base, o = {}) {
  const p = clonePose(base);
  for (const k in o) {
    if (k.endsWith('abs') || k.endsWith('body') || k === 'hiki') continue;
    p[k] = Array.isArray(o[k]) ? o[k].slice() : o[k];
  }
  const c = chestOf(p);
  const R = Ry(p.rot[0] + p.tw);
  for (const s of ['L', 'R']) {
    if (o['h' + s + 'abs']) p['h' + s] = sub(o['h' + s + 'abs'], c);
    if (o['h' + s + 'body']) p['h' + s] = mv(R, o['h' + s + 'body']);
    if (o['e' + s + 'body']) p['e' + s] = mv(R, o['e' + s + 'body']);
  }
  if (o.hiki) for (const s of o.hiki) hikite(p, s);
  return p;
}

const BASE = {
  pel: [0, 0.86, 0], rot: [0, 0, 0], tw: 0, head: [0, 0],
  fL: [0, 0, -0.1], fLr: [0, 0, 0], kL: [1, 0, 0],
  fR: [0, 0, 0.1], fRr: [0, 0, 0], kR: [1, 0, 0],
  hL: [0, -0.5, -0.25], eL: [-1, 0, -0.3], gL: [PI / 2, 0],
  hR: [0, -0.5, 0.25], eR: [-1, 0, 0.3], gR: [PI / 2, 0],
};

// ---------- עמידות יסוד ----------

// קמאה חופשית כפי שנראית בקומיטה של שוטוקאן אושימה (דוג'ו תל אביב): עמידה ארוכה ונמוכה,
// גו זקוף בהאנמי, שתי הידיים פתוחות מול הגוף, הקדמית בגובה הכתף והאחורית מול מקלעת השמש.
export const KAMAE = mk(BASE, {
  pel: [0, 0.80, 0], rot: d(40, 3, 0),
  fL: [0.36, 0, -0.11], fLr: d(5, 0, 0), kL: [1, 0, 0.05],
  fR: [-0.36, 0, 0.13], fRr: d(50, 0, 0), kR: [0.65, 0, 0.75],
  hLabs: [0.50, 1.29, -0.04], eL: [-0.2, -1, -0.35], gL: [PI / 2, 1],
  hRabs: [0.25, 1.11, 0.06], eR: [-0.4, -1, 0.5], gR: [PI / 2, 1],
});

// מוסובי-דאצ'י: עקבים צמודים, לקידה
export const STAND = mk(BASE, {
  pel: [0, 0.945, 0],
  fL: [0, 0, -0.06], fLr: d(-25, 0, 0), kL: [1, 0, -0.3],
  fR: [0, 0, 0.06], fRr: d(25, 0, 0), kR: [1, 0, 0.3],
  hLabs: [0.02, 0.87, -0.25], eL: [-1, 0, -0.3], gL: [PI / 2, 1],
  hRabs: [0.02, 0.87, 0.25], eR: [-1, 0, 0.3], gR: [PI / 2, 1],
});

const BOW = mk(STAND, {
  pel: [-0.05, 0.945, 0], rot: d(0, 28, 0),
  hLabs: [0.07, 0.85, -0.24], hRabs: [0.07, 0.85, 0.24],
});

// האצ'יג'י-דאצ'י (יוי): רגליים ברוחב הכתפיים, אגרופים לפני הגוף
const YOI = mk(STAND, {
  pel: [0, 0.935, 0],
  fL: [0, 0, -0.17], fLr: d(-15, 0, 0), fR: [0, 0, 0.17], fRr: d(15, 0, 0),
  hLabs: [0.15, 0.86, -0.13], eL: [-0.5, 0, -1], gL: [0, 0],
  hRabs: [0.15, 0.86, 0.13], eR: [-0.5, 0, 1], gR: [0, 0],
});

// ---------- טכניקות ----------
const T = {};
function def(id, o) { o.id = id; o.dur = o.keys[o.keys.length - 1].t; T[id] = o; return o; }
const K = (t, pose, ease = 'io', kime = false) => ({ t, pose, ease, kime });

// --- צוקי ---
for (const [lv, y] of [['c', Y_CHUDAN], ['j', Y_JODAN]]) {
  const cls = lv === 'j' ? 'J' : 'C';

  // קיזאמי-צוקי (מאה-טה): היד הקדמית, עם יורי-אשי קצר
  {
    const kime = mk(KAMAE, {
      pel: [0.20, 0.80, 0], rot: d(55, 6, 0),
      fL: [0.56, 0, -0.11], fR: [-0.20, 0, 0.13],
      hLabs: [1.05, y, -0.02], eL: [0, -1, -0.3], gL: [0, 0], hiki: ['R'],
    });
    def('kizami_' + lv, {
      name: 'kizami-zuki', he: 'קיזאמי־צוקי', kind: 'attack', lv,
      keys: [K(0.12, kime, 'punch', true), K(0.27, kime), K(0.48, shiftPose(KAMAE, 0.16))],
      hit: { from: 0.07, to: 0.17, limb: 'handL', dmg: 6, cls },
      advance: 0.16,
    });
  }

  // גיאקו-צוקי: היד האחורית, המותן נדחפת קדימה, רגל אחורית ישרה
  {
    const kime = mk(KAMAE, {
      pel: [0.18, 0.735, 0], rot: d(-8, 7, 0), tw: -14 * D2R,
      fL: [0.62, 0, -0.14], fLr: d(0, 0, 0),
      fRr: d(40, 0, 0), kR: [0.5, 0, 0.6],
      hRabs: [lv === 'j' ? 0.90 : 0.98, y, 0.02], eR: [0, -1, 0.3], gR: [0, 0], hiki: ['L'],
    });
    def('gyaku_' + lv, {
      name: 'gyaku-zuki', he: 'גיאקו־צוקי', kind: 'attack', lv,
      keys: [K(0.15, kime, 'punch', true), K(0.36, kime), K(0.60, KAMAE)],
      hit: { from: 0.09, to: 0.21, limb: 'handR', dmg: 10, cls },
      advance: 0,
    });
  }

  // אוי-צוקי: צעד מלא קדימה, האגרוף מגיע עם נחיתת הרגל
  {
    const mid = mk(KAMAE, {
      pel: [0.36, 0.75, 0], rot: d(15, 6, 0),
      fLr: d(-20, 0, 0),
      fR: [0.30, 0.03, 0.10], fRr: d(10, 10, 0), kR: [1, 0, 0.1],
      hLabs: [0.78, 1.24, -0.03], gL: [PI / 2, 0], hiki: ['R'],
    });
    const kime = mk(KAMAE, {
      // הגוף כולו מושלך אל המכה: גו נוטה קדימה, הכתף המכה נדחפת פנימה, רגל אחורית ישרה
      pel: [0.87, 0.715, 0], rot: d(-16, 10, 0), tw: -12 * D2R,
      fL: [0.36, 0, -0.11], fLr: d(-40, 0, 0), kL: [0.6, 0, -0.6],
      fR: [1.24, 0, 0.14], fRr: d(0, 0, 0), kR: [1, 0, 0],
      hRabs: [1.71, y - 0.02, 0.0], eR: [0, -1, 0.3], gR: [0, 0], hiki: ['L'],
    });
    def('oi_' + lv, {
      name: 'oi-zuki', he: 'אוי־צוקי', kind: 'attack', lv,
      keys: [K(0.19, mid, 'in'), K(0.35, kime, 'lunge', true), K(0.62, kime), K(0.92, shiftPose(mirrorPose(KAMAE), 0.88))],
      hit: { from: 0.27, to: 0.42, limb: 'handR', dmg: 14, cls },
      advance: 0.88, sw: true,
    });
  }
}

// --- מאה-גרי: הרגל האחורית, הרמת ברך, שליחה וחזרה מהירה ---
for (const lv of ['c', 'j']) {
  const chamber = mk(KAMAE, {
    pel: [0.20, 0.86, 0], rot: d(8, 2, 0), fLr: d(-15, 0, 0),
    fR: [0.56, 0.42, 0.08], fRr: d(0, 60, 0), kR: [1, 0.5, 0],
  });
  const ext = lv === 'c'
    ? mk(KAMAE, {
      pel: [0.27, 0.87, 0], rot: d(0, -3, 0), fLr: d(-20, 0, 0),
      fR: [1.09, 0.87, 0.03], fRr: d(0, -35, 0), kR: [0, 1, 0],
    })
    : mk(KAMAE, {
      pel: [0.27, 0.88, 0], rot: d(0, -9, 0), fLr: d(-20, 0, 0),
      fR: [0.97, 1.30, 0.03], fRr: d(0, -55, 0), kR: [0, 1, 0],
    });
  def('mae_' + lv, {
    name: 'mae-geri', he: 'מאה־גרי', kind: 'attack', lv,
    keys: [K(0.17, chamber, 'out'), K(0.25, ext, 'kick', true), K(0.28, ext), K(0.37, chamber, 'out'), K(0.60, KAMAE)],
    hit: { from: 0.19, to: 0.30, limb: 'footR', dmg: lv === 'j' ? 15 : 12, cls: lv === 'j' ? 'J' : 'G' },
    advance: 0,
  });
}

// --- מאוואשי-גרי: הרגל האחורית במעגל, כף הרגל התומכת מסתובבת ---
for (const lv of ['c', 'j']) {
  const arms = { hRbody: [0.30, -0.14, 0.12], eRbody: [-0.3, -1, 0.4], hLbody: [0.26, -0.16, -0.10], eLbody: [-0.3, -1, -0.4] };
  const chamber = mk(KAMAE, {
    pel: [0.18, 0.86, -0.03], rot: d(-40, 0, -10),
    fLr: d(-60, 0, 0), kL: [0.6, 0, -0.6],
    fR: [0.15, 0.83, 0.66], fRr: d(-60, 30, 0), kR: [1, 0.3, 0.4], ...arms,
  });
  const ext = lv === 'c'
    ? mk(KAMAE, {
      pel: [0.22, 0.87, -0.04], rot: d(-80, 0, -16),
      fLr: d(-100, 0, 0), kL: [0.2, 0, -1],
      fR: [1.10, 0.94, 0.0], fRr: d(-80, 35, 0), kR: [0, 0.4, 1], ...arms,
    })
    : mk(KAMAE, {
      pel: [0.22, 0.87, -0.04], rot: d(-80, 0, -28),
      fLr: d(-100, 0, 0), kL: [0.2, 0, -1],
      fR: [0.88, 1.40, 0.0], fRr: d(-80, 35, 0), kR: [0, 0.4, 1], ...arms,
    });
  def('mawashi_' + lv, {
    name: 'mawashi-geri', he: 'מאוואשי־גרי', kind: 'attack', lv,
    keys: [K(0.19, chamber, 'out'), K(0.29, ext, 'kick', true), K(0.35, ext), K(0.46, chamber, 'out'), K(0.70, KAMAE)],
    hit: { from: 0.22, to: 0.35, limb: 'footR', dmg: lv === 'j' ? 18 : 14, cls: lv === 'j' ? 'J' : 'C' },
    advance: 0,
  });
}

// --- יוקו-גרי קקומי: הרגל האחורית נאספת אל הקדמית, והקדמית נשלחת בצד הגוף ---
{
  const stepup = mk(KAMAE, {
    pel: [0.22, 0.85, 0], rot: d(70, 0, 0),
    fR: [0.14, 0, 0.04], fRr: d(85, 0, 0), kR: [0.2, 0, 1],
  });
  const chamber = mk(stepup, {
    pel: [0.21, 0.88, 0], rot: d(85, 0, 8),
    fL: [0.37, 0.52, -0.02], fLr: d(80, 0, 0), kL: [1, 0.7, 0],
    hLbody: [0.02, -0.20, -0.30], hRbody: [0.22, -0.16, 0.10],
  });
  const ext = mk(stepup, {
    pel: [0.22, 0.89, 0], rot: d(90, 0, 25),
    fL: [1.18, 0.90, 0], fLr: d(90, 10, 0), kL: [0, 1, 0],
    hLbody: [0.0, -0.30, -0.42], eLbody: [-0.3, -0.5, -0.3], hRbody: [0.24, -0.14, 0.08],
  });
  def('yoko', {
    name: 'yoko-geri kekomi', he: 'יוקו־גרי קקומי', kind: 'attack', lv: 'c',
    keys: [K(0.15, stepup), K(0.28, chamber), K(0.38, ext, 'kick', true), K(0.48, ext), K(0.60, chamber, 'out'), K(0.82, shiftPose(KAMAE, 0.50))],
    hit: { from: 0.31, to: 0.46, limb: 'footL', dmg: 16, cls: 'C' },
    advance: 0.50,
  });
}

// --- אושירו-גרי: סיבוב לאחור על הרגל הקדמית ובעיטת עקב ישרה ---
{
  const tuck = { hLbody: [0.22, -0.18, -0.12], eLbody: [-0.2, -1, -0.5], hRbody: [0.22, -0.18, 0.12], eRbody: [-0.2, -1, 0.5] };
  const pivot = mk(KAMAE, {
    pel: [0.30, 0.86, -0.04], rot: d(150, 5, 0),
    fLr: d(150, 0, 0), kL: [-0.87, 0, 0.5],
    fR: [0.18, 0.06, 0.05], fRr: d(160, 20, 0), kR: [-0.9, 0, 0.3], ...tuck,
  });
  const chamber = mk(KAMAE, {
    pel: [0.34, 0.86, -0.05], rot: d(180, 20, 0),
    fLr: d(180, 0, 0), kL: [-1, 0, 0],
    fR: [0.36, 0.42, -0.12], fRr: d(180, 40, 0), kR: [-1, -0.3, 0], ...tuck,
  });
  const ext = mk(KAMAE, {
    pel: [0.42, 0.86, -0.05], rot: d(180, 40, 0),
    fLr: d(180, 0, 0), kL: [-1, 0, 0],
    fR: [1.26, 0.90, -0.08], fRr: d(180, 50, 0), kR: [0, -1, 0], ...tuck,
  });
  const fin = shiftPose(mirrorPose(KAMAE), 0.72);
  fin.rot[0] += 2 * PI; fin.fRr[0] += 2 * PI; fin.fLr[0] += 2 * PI; // ממשיכים את הסיבוב באותו כיוון
  def('ushiro', {
    name: 'ushiro-geri', he: 'אושירו־גרי', kind: 'attack', lv: 'c',
    keys: [K(0.18, pivot), K(0.30, chamber), K(0.40, ext, 'kick', true), K(0.50, ext), K(0.62, chamber, 'out'), K(0.86, fin)],
    hit: { from: 0.33, to: 0.48, limb: 'footR', dmg: 18, cls: 'G' },
    advance: 0.72, sw: true,
  });
}

// --- חסימות: היד הקדמית חוסמת, השנייה בהיקיטה ---
{
  const st = { pel: [-0.03, 0.76, 0], rot: d(50, 0, 0), fL: [0.40, 0, -0.13], fR: [-0.40, 0, 0.15] };
  const stPrep = { pel: [-0.01, 0.81, 0], rot: d(40, 2, 0) };
  const blockDef = (id, name, he, prep, kime, covers) => def(id, {
    name, he, kind: 'block',
    keys: [K(0.08, prep), K(0.16, kime, 'punch', true), K(0.42, kime), K(0.58, KAMAE)],
    block: { from: 0.08, to: 0.44, covers }, hold: 0.30,
    advance: 0,
  });

  // אגה-אוקה: האמה עולה באלכסון אל מעל המצח
  blockDef('age', 'age-uke', 'אגה־אוקה',
    mk(KAMAE, { ...stPrep, hRabs: [0.36, 1.45, 0.0], gR: [PI / 2, 0], hLabs: [0.16, 1.08, 0.08], gL: [PI, 0] }),
    mk(KAMAE, { ...st, hLabs: [0.22, 1.60, 0.10], eL: [0.2, -0.3, -1], gL: [150 * D2R, 0], hiki: ['R'] }),
    ['J']);

  // אודה-אוקה: מבחוץ פנימה, המרפק בזווית ישרה
  blockDef('ude', 'ude-uke', 'אודה־אוקה',
    mk(KAMAE, { ...stPrep, hLabs: [0.05, 1.50, -0.30], eL: [-0.3, -0.2, -1], gL: [0, 0], hRabs: [0.34, 1.20, 0.0] }),
    mk(KAMAE, { ...st, hLabs: [0.50, 1.27, 0.04], eL: [0, -1, -0.15], gL: [PI, 0], hiki: ['R'] }),
    ['C']);

  // גדאן-בראי: מהכתף הנגדית למטה, מעל הברך הקדמית
  blockDef('gedan', 'gedan-barai', 'גדאן־בראי',
    mk(KAMAE, { ...stPrep, hLabs: [0.10, 1.42, 0.16], eL: [0, -1, -0.2], gL: [PI, 0], hRabs: [0.38, 1.0, -0.02] }),
    mk(KAMAE, { ...st, hLabs: [0.52, 0.80, -0.15], eL: [-0.6, 0.2, -0.6], gL: [0, 0], hiki: ['R'] }),
    ['G']);

  // שוטו-אוקה בקוקוטסו-דאצ'י: המשקל על הרגל האחורית, יד סכין
  const koku = {
    pel: [-0.17, 0.735, 0.03], rot: d(80, 0, 0),
    fL: [0.38, 0, -0.07], fLr: d(0, 0, 0), kL: [1, 0, 0],
    fR: [-0.36, 0, 0.14], fRr: d(90, 0, 0), kR: [-0.1, 0, 1],
  };
  blockDef('shuto', 'shuto-uke', 'שוטו־אוקה',
    mk(KAMAE, { ...koku, pel: [-0.07, 0.80, 0.01], rot: d(60, 0, 0), hLabs: [0.0, 1.42, 0.16], gL: [PI, 1], hRabs: [0.22, 1.15, 0.05], gR: [0, 1] }),
    mk(KAMAE, { ...koku, hLabs: [0.44, 1.19, -0.06], eL: [0, -1, -0.1], gL: [60 * D2R, 1], hRabs: [-0.02, 1.02, 0.03], eR: [-0.5, -0.5, 0.7], gR: [PI, 1] }),
    ['C']);
}

// --- תנועה: יורי-אשי ---
def('stepF', {
  name: '', he: '', kind: 'move',
  keys: [K(0.10, mk(KAMAE, { pel: [0.10, 0.81, 0], fL: [0.54, 0.02, -0.11] }), 'lin'), K(0.20, shiftPose(KAMAE, 0.18), 'lin')],
  advance: 0.18,
});
def('stepB', {
  name: '', he: '', kind: 'move',
  keys: [K(0.10, mk(KAMAE, { pel: [-0.10, 0.81, 0], fR: [-0.54, 0.02, 0.13] }), 'lin'), K(0.20, shiftPose(KAMAE, -0.18), 'lin')],
  advance: -0.18,
});

// --- תגובות לפגיעה ---
{
  const pj = mk(KAMAE, { pel: [-0.08, 0.82, 0], rot: d(35, -16, 0), head: [0, -25 * D2R] });
  def('hitJ', { name: '', he: '', kind: 'react', keys: [K(0.08, pj, 'out'), K(0.22, pj), K(0.46, KAMAE)], advance: 0 });
  const pc = mk(KAMAE, { pel: [-0.10, 0.77, 0], rot: d(35, 24, 0), head: [0, 10 * D2R], hLabs: [0.25, 1.0, -0.08], hRabs: [0.18, 0.98, 0.08] });
  def('hitC', { name: '', he: '', kind: 'react', keys: [K(0.08, pc, 'out'), K(0.24, pc), K(0.50, KAMAE)], advance: 0 });

  // נוקאאוט: הברכיים קורסות, נפילה לאחור
  const k1 = mk(KAMAE, { pel: [-0.15, 0.62, 0], rot: d(20, -10, 0), head: [0, -20 * D2R], hLabs: [0.10, 0.95, -0.25], hRabs: [0.0, 0.90, 0.25], gL: [PI / 2, 1], gR: [PI / 2, 1] });
  const k2 = mk(KAMAE, {
    pel: [-0.42, 0.27, 0], rot: d(8, -45, 0), head: [0, -10 * D2R],
    fL: [0.24, 0, -0.15], fLr: d(-10, 0, 0), kL: [0, 1, -0.2], fR: [0.02, 0, 0.18], fRr: d(15, 0, 0), kR: [0, 1, 0.3],
    hLabs: [-0.55, 0.30, -0.40], hRabs: [-0.62, 0.30, 0.40], gL: [0, 1], gR: [0, 1],
  });
  const k3 = mk(KAMAE, {
    pel: [-0.55, 0.13, 0], rot: d(0, -86, 0), head: [0, -38 * D2R],
    fL: [0.25, 0, -0.16], fLr: d(-12, 0, 0), kL: [0, 1, -0.2], fR: [0.16, 0, 0.20], fRr: d(15, 0, 0), kR: [0, 1, 0.3],
    hLabs: [-0.92, 0.05, -0.52], eL: [0, -1, -1], hRabs: [-1.12, 0.05, 0.50], eR: [0, -1, 1], gL: [PI, 1], gR: [PI, 1],
  });
  def('ko', { name: '', he: '', kind: 'ko', keys: [K(0.25, k1, 'out'), K(0.62, k2, 'in'), K(0.90, k3, 'out')], advance: 0, stay: true });
}

// --- טקס: ריי בתחילת הקרב, יאמה בסופו ---
def('rei', {
  name: 'rei', he: 'רֵיי', kind: 'cine',
  keys: [K(0.5, STAND), K(1.1, BOW), K(1.6, BOW), K(2.2, STAND), K(2.6, STAND), K(3.1, KAMAE)],
  advance: 0,
});
def('yame', {
  name: 'yame', he: 'יאמה', kind: 'cine',
  keys: [K(0.6, KAMAE), K(1.3, YOI), K(2.0, YOI), K(2.5, STAND), K(3.1, BOW), K(3.6, BOW), K(4.2, STAND)],
  advance: 0, stay: true,
});

export const TECH = T;

// גרסאות משוקפות (לעמידת ימין-קדימה), מחושבות פעם אחת
for (const id in T) {
  const t = T[id];
  t.keysM = t.keys.map((k) => ({ t: k.t, ease: k.ease, kime: k.kime, pose: mirrorPose(k.pose) }));
}

const EASE = {
  lin: (u) => u,
  in: (u) => u * u,
  out: (u) => 1 - (1 - u) * (1 - u),
  io: (u) => u * u * (3 - 2 * u),
};
export const ease = (name, u) => (EASE[name] || EASE.io)(u);

// שרשרת הגוף. בתנועת קימה הגוף לא נע כמקשה אחת: המותן והרגליים מובילות,
// הכתפיים אחריהן, והיד או הרגל המכה מואצת אחרונה ונעצרת בבת אחת.
const sm = (u) => { u = Math.max(0, Math.min(1, u)); return u * u * (3 - 2 * u); };
const late = (u, start, pw) => Math.pow(Math.max(0, (u - start) / (1 - start)), pw);
const CHAIN = {
  //        אגן וסבסוב         כפות רגליים        כיוון כף רגל       פיתול חזה            ידיים
  punch: { core: (u) => sm(u / 0.70), feet: (u) => sm(u / 0.70), frot: (u) => sm(u / 0.70), tw: (u) => sm((u - 0.15) / 0.7), arms: (u) => late(u, 0.28, 2.3) },
  lunge: { core: (u) => u, feet: (u) => u, frot: (u) => sm(u), tw: (u) => sm((u - 0.3) / 0.6), arms: (u) => late(u, 0.40, 2.0) },
  kick: { core: (u) => sm(u / 0.60), feet: (u) => late(u, 0.10, 2.3), frot: (u) => sm(u / 0.60), tw: (u) => sm(u / 0.6), arms: (u) => sm(u / 0.7) },
};
const GROUP = {
  pel: 'core', rot: 'core', head: 'core', tw: 'tw',
  fL: 'feet', fR: 'feet', kL: 'feet', kR: 'feet', fLr: 'frot', fRr: 'frot',
  hL: 'arms', hR: 'arms', eL: 'arms', eR: 'arms', gL: 'arms', gR: 'arms',
};
function lerpChain(a, b, u, chain, out) {
  for (const k in a) {
    const t = chain[GROUP[k]](u), x = a[k], y = b[k];
    if (Array.isArray(x)) {
      const o = out[k] || (out[k] = new Array(x.length));
      for (let i = 0; i < x.length; i++) o[i] = x[i] + (y[i] - x[i]) * t;
    } else out[k] = x + (y - x) * t;
  }
  return out;
}

export function poseAt(tech, t, from, mirrored = false, out = {}) {
  const keys = mirrored ? tech.keysM : tech.keys;
  let prev = from, t0 = 0;
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    if (t <= k.t || i === keys.length - 1) {
      const u = Math.max(0, Math.min(1, (t - t0) / (k.t - t0)));
      if (CHAIN[k.ease]) return lerpChain(prev, k.pose, u, CHAIN[k.ease], out);
      return lerpPose(prev, k.pose, ease(k.ease, u), out);
    }
    prev = k.pose; t0 = k.t;
  }
  return out;
}

export const REACH = {};
for (const id in T) {
  const t = T[id];
  if (!t.hit) continue;
  let best = -1, y = 0;
  for (let s = t.hit.from; s <= t.hit.to; s += 0.005) {
    const J = solve(poseAt(t, s, KAMAE));
    const p = J[t.hit.limb];
    if (p[0] > best) { best = p[0]; y = p[1]; }
  }
  REACH[id] = { x: best, y };
}

// מיפוי פעולה + גובה לטכניקה
export function techFor(action, level) {
  switch (action) {
    case 'kizami': return level === 'down' ? T.kizami_c : T.kizami_j;
    case 'gyaku': return level === 'up' ? T.gyaku_j : T.gyaku_c;
    case 'oi': return level === 'up' ? T.oi_j : T.oi_c;
    case 'mae': return level === 'up' ? T.mae_j : T.mae_c;
    case 'mawashi': return level === 'up' ? T.mawashi_j : T.mawashi_c;
    case 'yoko': return T.yoko;
    case 'ushiro': return T.ushiro;
    case 'block': return level === 'up' ? T.age : level === 'down' ? T.gedan : T.ude;
    case 'shuto': return T.shuto;
  }
  return null;
}
