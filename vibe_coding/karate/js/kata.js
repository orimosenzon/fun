// kata.js: מנוע הקאטות. קוד טהור (בלי three) כדי שאפשר לבדוק אותו ב-node.
// קאטה היא רשימת תנועות. כל תנועה אומרת: באיזו עמידה, לאיזה כיוון, איזו רגל זזה, ומה עושה כל יד.
// המנוע מחשב ממנה תנוחה מלאה (בפורמט של body.js) במערכת הצירים של הדוג'ו:
// x אל החזית (שומן), z ימינה, y למעלה. הכיוון F הוא הכיוון שאליו עומדים בתחילת הקאטה.

import { D2R, Ry, mv, add, sub, chestOf, clonePose } from './body.js';

const PI = Math.PI;
export const DIR = { F: 0, FR: 45, R: 90, BR: 135, B: 180, BL: -135, L: -90, FL: -45 };

// ---------- עמידות ----------
// מוגדרות לרגל שמאל מובילה (lead=L), מבט אל +x. fr היא הרגל המובילה, bk השנייה.
// [x, z] מיקום הקרסול יחסית למרכז העמידה. yaw זווית כף הרגל (חיובי = ימינה), pt הרמת עקב, lift גובה הקרסול.
// h גובה האגן, px הסטת האגן קדימה, hip סיבוב האגן ברירת מחדל, kn כיווני הברכיים [קדימה, הצידה כלפי חוץ].
// he/en/ja: שמות.
export const STANCE = {
  zen: { he: 'זנקוטסו־דאצ׳י', ja: 'zenkutsu-dachi', fr: [0.46, -0.14], bk: [-0.46, 0.14], frYaw: 0, bkYaw: 40, h: 0.745, px: 0.07, lean: 4, kf: [1, -0.1], kb: [0.5, 0.8], hipH: 45, hipG: -12 },
  kok: { he: 'קוקוטסו־דאצ׳י', ja: 'kokutsu-dachi', fr: [0.36, -0.03], bk: [-0.35, 0.05], frYaw: 0, bkYaw: 88, h: 0.735, px: -0.25, lean: 0, kf: [1, -0.05], kb: [-0.1, 1], hipH: 65, hipG: 20, hip0: 65 },
  kiba: { he: 'קיבה־דאצ׳י', ja: 'kiba-dachi', fr: [0, -0.40], bk: [0, 0.40], frYaw: 4, bkYaw: -4, h: 0.725, px: 0, lean: 0, kf: [0.55, 0.85], kb: [0.55, 0.85], side: true },
  shiko: { he: 'שיקו־דאצ׳י', ja: 'shiko-dachi', fr: [0, -0.42], bk: [0, 0.42], frYaw: -40, bkYaw: 40, h: 0.69, px: 0, lean: 0, kf: [0.4, 0.95], kb: [0.4, 0.95], side: true },
  neko: { he: 'נקו־אשי־דאצ׳י', ja: 'nekoashi-dachi', fr: [0.30, -0.05], bk: [-0.04, 0.07], frYaw: 0, bkYaw: 35, frPt: 32, frLift: 0.045, h: 0.735, px: -0.05, lean: 0, kf: [1, 0], kb: [0.6, 0.7], hipH: 45, hipG: 10, hip0: 45 },
  fudo: { he: 'פודו־דאצ׳י', ja: 'fudo-dachi', fr: [0.42, -0.17], bk: [-0.42, 0.17], frYaw: -12, bkYaw: 45, h: 0.705, px: 0, lean: 0, kf: [1, 0.35], kb: [0.4, 0.9], hipH: 40, hipG: 0, hip0: 30 },
  hangetsu: { he: 'הנגטסו־דאצ׳י', ja: 'hangetsu-dachi', fr: [0.34, -0.17], bk: [-0.34, 0.17], frYaw: 25, bkYaw: -8, h: 0.77, px: 0.0, lean: 0, kf: [1, -0.35], kb: [1, -0.25], hipH: 25, hipG: -10, hip0: 20 },
  heisoku: { he: 'הייסוקו־דאצ׳י', ja: 'heisoku-dachi', fr: [0, -0.055], bk: [0, 0.055], frYaw: 0, bkYaw: 0, h: 0.945, px: 0, lean: 0, kf: [1, 0], kb: [1, 0], side: true },
  musubi: { he: 'מוסובי־דאצ׳י', ja: 'musubi-dachi', fr: [0, -0.06], bk: [0, 0.06], frYaw: -27, bkYaw: 27, h: 0.945, px: 0, lean: 0, kf: [1, 0.3], kb: [1, 0.3], side: true },
  hachiji: { he: 'האצ׳יג׳י־דאצ׳י', ja: 'hachiji-dachi', fr: [0, -0.17], bk: [0, 0.17], frYaw: -15, bkYaw: 15, h: 0.935, px: 0, lean: 0, kf: [1, 0.2], kb: [1, 0.2], side: true },
  heiko: { he: 'הייקו־דאצ׳י', ja: 'heiko-dachi', fr: [0, -0.15], bk: [0, 0.15], frYaw: 0, bkYaw: 0, h: 0.935, px: 0, lean: 0, kf: [1, 0.1], kb: [1, 0.1], side: true },
  teiji: { he: 'טייג׳י־דאצ׳י', ja: 'teiji-dachi', fr: [0.24, -0.03], bk: [-0.04, 0.05], frYaw: 0, bkYaw: 80, h: 0.925, px: -0.03, lean: 0, kf: [1, 0], kb: [0.3, 1], hip0: 50, hipH: 50, hipG: 10 },
  // קוסה־דאצ'י בקפיצה קדימה (הייאן יודאן): הרגל המובילה נטועה, השנייה משוכלת מאחוריה על כרית כף הרגל
  kosa: { he: 'קוסה־דאצ׳י', ja: 'kosa-dachi', fr: [0.10, 0.02], bk: [-0.14, -0.07], frYaw: 0, bkYaw: 35, bkPt: 45, bkLift: 0.05, h: 0.70, px: 0.02, lean: 8, kf: [1, -0.1], kb: [0.8, -0.5] },
  // צעד שיכול הצידה (טקי): הרגל המובילה חוצה מלפנים אל הצד של השנייה
  kosaY: { he: 'קוסה־דאצ׳י', ja: 'kosa-dachi', fr: [0.07, 0.08], bk: [-0.05, -0.08], frYaw: 15, bkYaw: -10, h: 0.75, px: 0, lean: 0, kf: [1, 0.3], kb: [1, -0.3], side: true },
  // עמידת עגור על רגל אחת: הרגל המובילה מורמת אל ברך הרגל התומכת
  tsuru: { he: 'צורו־אשי־דאצ׳י', ja: 'tsuru-ashi-dachi', fr: [0.07, 0.02], bk: [0, 0.07], frYaw: 0, bkYaw: 0, frPt: 55, frLift: 0.40, h: 0.90, px: 0, lean: 0, kf: [1, -0.3], kb: [1, 0], side: true, oneLeg: true },
  // עמידה זקופה כללית לתנועות ביניים
  shizen: { he: 'שיזנטאי', ja: 'shizentai', fr: [0, -0.15], bk: [0, 0.15], frYaw: -10, bkYaw: 10, h: 0.935, px: 0, lean: 0, kf: [1, 0.2], kb: [1, 0.2], side: true },
  // ישיבה על ברך אחת (כמו באמפי): הברך האחורית על הרצפה
  hiza: { he: 'כריעה', ja: 'hiza-dachi', fr: [0.30, -0.12], bk: [-0.40, 0.12], frYaw: 0, bkYaw: 10, bkPt: 70, bkLift: 0.10, h: 0.50, px: 0.0, lean: 10, kf: [1, -0.1], kb: [0, -1], hip0: 30, hipH: 40, hipG: 0 },
  // שכיבה נמוכה על הרצפה (קוואנקו): הידיים על הרצפה
  fuse: { he: 'ירידה לרצפה', ja: 'fuse', fr: [0.12, -0.14], bk: [-0.70, 0.12], frYaw: 0, bkYaw: 20, bkPt: 40, bkLift: 0.03, h: 0.40, px: -0.05, lean: 55, kf: [1, -0.4], kb: [0, -1], hip0: 0 },
};

// ---------- טכניקות ידיים ----------
// כל פונקציה מקבלת s = -1 לשמאל או +1 לימין ומחזירה, במערכת "כיוון המטרה" (x קדימה, z ימינה), יחסית לחזה:
// h מיקום שורש כף היד, e כיוון המרפק, g [גלגול, פתוח]. p: תנוחת ההכנה של היד הזאת.
// o: מה עושה היד השנייה בהכנה (מקבלת את הצד שלה). pair: מה עושה היד השנייה בסיום (במקום היקיטה).
// hip: 'h' טכניקת חסימה (האנמי), 's' אגרוף (שומן). he/ja: שם להצגה.
const Y = { j: 0.24, c: -0.07, g: -0.40 };
const LVHE = { j: 'ג׳ודאן', c: 'צ׳ודאן', g: 'גדאן' };
const fwdOther = (s) => ({ h: [0.46, -0.06, 0], e: [0, -1, s * 0.3], g: [0, 0] });
const fwdOtherOpen = (s) => ({ h: [0.44, -0.10, 0], e: [0, -1, s * 0.3], g: [0, 1] });

export const ARM = {
  hiki: { he: '', hip: '' },
  hold: { he: '', hip: '' },
  tsuki: { he: 'צוקי', ja: 'tsuki', hip: 's', lv: true, f: (s, lv) => ({ h: [0.57, Y[lv], -s * 0.01], e: [0, -1, s * 0.3], g: [0, 0] }) },
  tate: { he: 'טטה־צוקי', ja: 'tate-zuki', hip: 's', lv: true, f: (s, lv) => ({ h: [0.55, Y[lv], -s * 0.01], e: [0, -1, s * 0.15], g: [PI / 2, 0] }) },
  ura: { he: 'אורה־צוקי', ja: 'ura-zuki', hip: 's', f: (s) => ({ h: [0.40, -0.16, -s * 0.0], e: [0, -1, s * 0.05], g: [PI, 0] }) },
  kagi: { he: 'קאגי־צוקי', ja: 'kagi-zuki', hip: '', body: true, f: (s) => ({ h: [0.24, -0.06, -s * 0.20], e: [0, -0.25, s * 1], g: [0, 0] }) },
  age: {
    he: 'אגה־אוקה', ja: 'age-uke', hip: 'h',
    f: (s) => ({ h: [0.27, 0.36, -s * 0.03], e: [0.25, -0.2, s * 1], g: [2.6, 0] }),
    p: (s) => ({ h: [0.30, -0.20, -s * 0.08], e: [0, -1, s * 0.5], g: [PI, 0] }),
    o: (s) => ({ h: [0.42, 0.06, -s * 0.02], e: [0, -1, s * 0.3], g: [0, 0] }),
  },
  soto: {
    he: 'אודה־אוקה מבחוץ פנימה', ja: 'soto ude-uke', hip: 'h',
    f: (s) => ({ h: [0.40, 0.03, s * 0.05], e: [0, -1, s * 0.15], g: [PI, 0] }),
    p: (s) => ({ h: [-0.08, 0.20, s * 0.36], e: [-0.5, -0.3, s * 1], g: [0, 0] }),
    o: fwdOther,
  },
  uchi: {
    he: 'אודה־אוקה', ja: 'ude-uke', hip: 'h',
    f: (s) => ({ h: [0.40, 0.03, s * 0.20], e: [0.1, -1, 0], g: [2.7, 0] }),
    p: (s) => ({ h: [0.12, -0.15, -s * 0.18], e: [0, -1, s * 1], g: [0, 0] }),
    o: fwdOther,
  },
  gedan: {
    he: 'גדאן־בראי', ja: 'gedan-barai', hip: 'h',
    f: (s) => ({ h: [0.44, -0.42, s * 0.15], e: [-0.4, 0.3, s * 0.8], g: [0, 0] }),
    p: (s) => ({ h: [0.08, 0.14, -s * 0.16], e: [0, -1, s * 0.3], g: [PI, 0] }),
    o: (s) => ({ h: [0.42, -0.32, 0], e: [0, -1, s * 0.3], g: [0, 0] }),
  },
  shuto: {
    he: 'שוטו־אוקה', ja: 'shuto-uke', hip: 'h', pair: 'shuto2',
    f: (s) => ({ h: [0.40, 0.03, s * 0.08], e: [0, -1, s * 0.1], g: [1.0, 1] }),
    p: (s) => ({ h: [0.04, 0.18, -s * 0.17], e: [0, -1, s * 0.3], g: [PI, 1] }),
    o: fwdOtherOpen,
  },
  shuto2: { he: '', hip: '', f: (s) => ({ h: [0.15, -0.17, s * 0.02], e: [-0.5, -0.4, s * 1], g: [PI, 1] }) },
  // מורוטה־אוקה: האגרוף השני תומך מתחת למרפק של היד החוסמת
  morote2: { he: '', hip: '', f: (s) => ({ h: [0.24, -0.14, -s * 0.13], e: [0, -1, s * 0.5], g: [PI / 2, 0] }) },
  // שוטו־אוצ׳י: מכת סכין מבחוץ פנימה בגובה הצוואר
  shutouchi: {
    he: 'שוטו־אוצ׳י', ja: 'shuto-uchi', hip: 's',
    f: (s) => ({ h: [0.46, 0.18, s * 0.02], e: [0, -1, s * 0.6], g: [PI, 1] }),
    p: (s) => ({ h: [-0.05, 0.20, s * 0.30], e: [-0.4, 0, s * 1], g: [0, 1] }),
  },
  // אורקן: מכת גב האגרוף, מהמרפק, אל הפנים
  uraken: {
    he: 'אורקן־אוצ׳י', ja: 'uraken-uchi', hip: '',
    f: (s) => ({ h: [0.40, 0.22, s * 0.02], e: [0.2, -0.3, s * 1], g: [PI / 2 + 0.3, 0] }),
    p: (s) => ({ h: [0.10, 0.02, -s * 0.10], e: [0, -0.3, s * 1], g: [PI / 2, 0] }),
  },
  tettsui: {
    he: 'טטסוי־אוצ׳י', ja: 'tettsui-uchi', hip: 's',
    f: (s) => ({ h: [0.46, -0.02, -s * 0.0], e: [0, -1, s * 0.6], g: [PI / 2, 0] }),
    p: (s) => ({ h: [-0.02, 0.28, s * 0.20], e: [0, 0, s * 1], g: [PI / 2, 0] }),
  },
  // אמפי: מכת מרפק
  empi: { he: 'מאה־אמפי', ja: 'mae-empi', hip: 's', f: (s) => ({ h: [0.10, -0.02, -s * 0.06], e: [1, -0.2, s * 0.1], g: [0, 0] }) },
  yokoEmpi: { he: 'יוקו־אמפי', ja: 'yoko-empi', hip: '', f: (s) => ({ h: [0.06, -0.04, s * 0.02], e: [0, -0.2, s * 1], g: [0, 0] }) },
  ageEmpi: { he: 'אגה־אמפי', ja: 'age-empi', hip: 's', f: (s) => ({ h: [0.05, 0.12, s * 0.02], e: [0.3, 1, s * 0.2], g: [PI / 2, 0] }) },
  ushiroEmpi: { he: 'אושירו־אמפי', ja: 'ushiro-empi', hip: '', f: (s) => ({ h: [0.02, -0.32, s * 0.24], e: [-1, 0, s * 0.2], g: [PI, 0] }) },
  nukite: { he: 'נוקיטה', ja: 'nukite', hip: 's', pair: 'osae', f: (s) => ({ h: [0.57, -0.07, -s * 0.01], e: [0, -1, s * 0.3], g: [PI / 2, 1] }) },
  osae: { he: 'אוסאה־אוקה', ja: 'osae-uke', hip: 'h', f: (s) => ({ h: [0.30, -0.22, -s * 0.03], e: [-0.3, -1, s * 0.9], g: [0, 1] }) },
  // כף יד פתוחה מקבלת (מטרה למרפק או לבעיטה)
  palm: { he: '', hip: '', f: (s) => ({ h: [0.36, 0.0, -s * 0.10], e: [0, -1, s * 0.6], g: [PI / 2, 1] }) },
  // כף יד פתוחה מורמת לצד (הייאן יודאן): מטרה לבעיטה ולאורקן
  palmSide: { he: '', hip: '', f: (s) => ({ h: [0.40, 0.08, s * 0.04], e: [0, -1, s * 0.4], g: [PI / 2, 1] }) },
  haishu: { he: 'האישו־אוקה', ja: 'haishu-uke', hip: 'h', f: (s) => ({ h: [0.38, 0.12, s * 0.12], e: [0, -1, s * 0.3], g: [PI / 2 + 0.4, 1] }), p: (s) => ({ h: [0.06, -0.18, -s * 0.16], e: [0, -1, s], g: [0, 1] }) },
  // יד פתוחה מעל המצח, גב היד כלפי חוץ (היד השנייה בהייאן יודאן)
  hitai: { he: '', hip: '', f: (s) => ({ h: [0.20, 0.30, -s * 0.06], e: [0.3, -0.1, s * 1], g: [2.4, 1] }) },
  haito: { he: 'הייטו־אוצ׳י', ja: 'haito-uchi', hip: 's', f: (s) => ({ h: [0.46, 0.20, -s * 0.06], e: [0, -1, s * 0.5], g: [PI, 1] }), p: (s) => ({ h: [0.00, -0.30, s * 0.30], e: [0, -1, s], g: [PI, 1] }) },
  teisho: { he: 'טיישו', ja: 'teisho', hip: 's', f: (s) => ({ h: [0.50, -0.06, -s * 0.0], e: [0, -1, s * 0.3], g: [PI / 2, 1] }) },
  sukui: { he: 'סוקוי־אוקה', ja: 'sukui-uke', hip: 'h', f: (s) => ({ h: [0.40, -0.18, s * 0.04], e: [0, -1, s * 0.5], g: [PI, 1] }), p: (s) => ({ h: [0.30, -0.45, s * 0.20], e: [0, -1, s], g: [PI, 1] }) },
  nagashi: { he: 'נגשי־אוקה', ja: 'nagashi-uke', hip: '', f: (s) => ({ h: [0.22, 0.12, -s * 0.12], e: [0.2, -1, s * 0.8], g: [PI / 2, 1] }) },
  // ירידה לרצפה: כף היד על הרצפה
  floor: { he: '', hip: '', f: (s) => ({ h: [0.40, -0.80, s * 0.14], e: [-0.3, 0, s * 1], g: [0, 1] }) },
  // תנוחות פתיחה וסגירה
  yoi: { he: '', hip: '', f: (s) => ({ h: [0.16, -0.38, s * 0.12], e: [-0.5, 0, s * 1], g: [0, 0] }) },
  side: { he: '', hip: '', f: (s) => ({ h: [0.02, -0.38, s * 0.24], e: [-1, 0, s * 0.3], g: [PI / 2, 1] }) },
  // תנוחות שתי ידיים
  kakiwake: {
    he: 'קקיוואקה־אוקה', ja: 'kakiwake-uke', hip: '', two: true,
    f: (s) => ({ h: [0.36, 0.05, s * 0.23], e: [0, -1, s * 0.2], g: [2.85, 0] }),
    p: (s) => ({ h: [0.24, -0.16, -s * 0.06], e: [0, -1, s], g: [0, 0] }),
  },
  jujiG: {
    he: 'ג׳וג׳י־אוקה גדאן', ja: 'gedan juji-uke', hip: '', two: true,
    f: (s) => ({ h: [0.42, -0.36, -s * 0.03], e: [0, -1, s * 0.4], g: [0, 0] }),
  },
  jujiJ: {
    he: 'ג׳וג׳י־אוקה ג׳ודאן', ja: 'jodan juji-uke', hip: '', two: true,
    f: (s) => ({ h: [0.30, 0.32, -s * 0.04], e: [0.3, -0.4, s * 1], g: [2.4, 1] }),
  },
  morote: { he: 'מורוטה־צוקי', ja: 'morote-zuki', hip: '', two: true, lv: true, f: (s, lv) => ({ h: [0.55, Y[lv], s * 0.10], e: [0, -1, s * 0.3], g: [0, 0] }) },
  // יאמה־צוקי: יד עליונה לג'ודאן, תחתונה אורה־צוקי
  yamaHi: { he: 'יאמה־צוקי', ja: 'yama-zuki', hip: '', f: (s) => ({ h: [0.50, 0.22, -s * 0.02], e: [0, -1, s * 0.5], g: [0, 0] }) },
  yamaLo: { he: '', hip: '', f: (s) => ({ h: [0.42, -0.26, -s * 0.02], e: [0, -1, s * 0.1], g: [PI, 0] }) },
  // אגרופים צמודים במותן (קושי־גמאה); הצד ניתן ב-@
  koshiTop: { he: 'קושי־גמאה', ja: 'koshi-gamae', hip: '', f: (s) => ({ h: [-0.04, -0.30, 0.20], e: [-1, -0.3, s * 0.4], g: [0, 0] }) },
  koshiBot: { he: '', hip: '', f: (s) => ({ h: [-0.04, -0.37, 0.20], e: [-1, -0.3, s * 0.4], g: [PI, 0] }) },
  // ידיים אחת על השנייה מול הבטן (תחילת ג'יון ובסאי)
  cover: { he: '', hip: '', f: (s) => ({ h: [0.24, -0.30, s * 0.00], e: [-0.4, -0.5, s * 1], g: [PI / 2, s < 0 ? 1 : 0] }) },
  // מנג׳י־אוקה: גדאן־בראי לצד אחד ואוצ׳י־אוקה ג'ודאן לצד השני (הכיוונים ניתנים ב-@)
  ageUchi: { he: 'ג׳ודאן־אוקה', ja: 'jodan uchi-uke', hip: '', f: (s) => ({ h: [0.30, 0.25, s * 0.18], e: [0, -1, s * 0.3], g: [2.7, 0] }) },
  // --- תוספות לקאטות הייאן ---
  haiwan: { he: 'האיוואן־אוקה ג׳ודאן', ja: 'jodan haiwan-uke', hip: 'h', f: (s) => ({ h: [0.30, 0.28, s * 0.06], e: [0.2, -1, s * 0.25], g: [1.9, 0] }), p: (s) => ({ h: [0.10, -0.25, -s * 0.15], e: [0, -1, s], g: [PI, 0] }) },
  haiwanO: { he: 'האישו־אוקה ג׳ודאן', ja: 'jodan haishu-uke', hip: 'h', f: (s) => ({ h: [0.30, 0.28, s * 0.06], e: [0.2, -1, s * 0.25], g: [1.9, 1] }), p: (s) => ({ h: [0.10, -0.25, -s * 0.15], e: [0, -1, s], g: [PI, 1] }) },
  hitaiF: { he: '', hip: '', f: (s) => ({ h: [0.17, 0.31, -s * 0.03], e: [0.2, 0.1, s * 1], g: [2.5, 0] }), p: (s) => ({ h: [0.12, -0.28, s * 0.10], e: [0, -1, s], g: [PI, 0] }) },
  uraIn: { he: '', hip: '', body: true, f: (s) => ({ h: [0.30, 0.02, -s * 0.12], e: [0, -1, s * 0.3], g: [PI, 0] }) },
  shoulder: { he: '', hip: '', body: true, f: (s) => ({ h: [0.13, 0.03, -s * 0.15], e: [0.3, -0.7, s * 1], g: [PI / 2, 0] }) },
  kLbot: { he: 'קושי־גמאה', ja: 'koshi-gamae', hip: '', body: true, f: () => ({ h: [-0.02, -0.31, -0.20], e: [-1, -0.2, -0.3], g: [PI, 0] }) },
  kLtop: { he: '', hip: '', body: true, f: () => ({ h: [0.0, -0.24, -0.19], e: [0.4, -0.5, 1], g: [PI / 2, 0] }) },
  kRbot: { he: 'קושי־גמאה', ja: 'koshi-gamae', hip: '', body: true, f: () => ({ h: [-0.02, -0.31, 0.20], e: [-1, -0.2, 0.3], g: [PI, 0] }) },
  kRtop: { he: '', hip: '', body: true, f: () => ({ h: [0.0, -0.24, 0.19], e: [0.4, -0.5, -1], g: [PI / 2, 0] }) },
  hipFist: { he: '', hip: '', body: true, f: (s) => ({ h: [0.03, -0.30, s * 0.21], e: [-0.25, 0, s * 1], g: [PI / 2, 0] }) },
  overShoulder: { he: '', hip: '', body: true, f: (s) => ({ h: [0.03, 0.05, -s * 0.22], e: [0.8, -0.5, s * 0.3], g: [0, 0] }) },
  tateShuto: { he: 'טטה־שוטו־אוקה', ja: 'tate shuto-uke', hip: '', f: (s) => ({ h: [0.52, -0.04, 0], e: [0, -1, s * 0.3], g: [PI / 2, 1] }) },
  palmT: { he: '', hip: '', f: (s) => ({ h: [0.25, -0.03, -s * 0.03], e: [0, -1, s * 1], g: [PI / 2, 1] }) },
  grab: { he: '', hip: '', two: true, f: (s) => ({ h: [0.42, 0.22, s * 0.10], e: [0, -1, s * 0.5], g: [PI / 2, 1] }) },
  pullDown: { he: '', hip: '', two: true, f: (s) => ({ h: [0.20, -0.36, s * 0.17], e: [-0.6, 0, s * 1], g: [0, 0] }) },
  gedanShuto: { he: 'גדאן שוטו־אוצ׳י', ja: 'gedan shuto-uchi', hip: '', f: (s) => ({ h: [0.42, -0.42, -s * 0.04], e: [0, 0.2, s * 1], g: [PI / 2, 1] }) },
  manjiHi: { he: '', hip: '', body: true, f: (s) => ({ h: [-0.06, 0.34, s * 0.30], e: [0, -1, s * 0.6], g: [2.9, 0] }) },
  ageZuki: { he: '', hip: '', f: (s) => ({ h: [0.30, 0.30, 0], e: [0, -1, s * 0.3], g: [PI, 0] }) },
  osaeLow: { he: 'אוסאה־אוקה', ja: 'osae-uke', hip: '', two: true, body: true, f: (s) => ({ h: [0.14, -0.30, 0.13 + s * 0.03], e: [-0.3, -0.3, s * 1], g: [PI / 2, 1] }) },
};

// ---------- כלים ----------
const rad = (x) => x * D2R;
const unwrapNear = (a, ref) => { while (a - ref > PI) a -= 2 * PI; while (a - ref < -PI) a += 2 * PI; return a; };
const rot2 = (yaw, x, z) => [Math.cos(yaw) * x - Math.sin(yaw) * z, Math.sin(yaw) * x + Math.cos(yaw) * z];
const toRel = (yaw, v) => mv(Ry(yaw), v);

// מפרק מחרוזת יד: 'tsuki_j', 'gedan', 'uraken@L', 'koshiTop@R'
function parseArm(spec) {
  if (!spec) return null;
  let [t, aim] = spec.split('@');
  let lv = 'c';
  const m = t.match(/^(.*)_([jcg])$/);
  if (m) { t = m[1]; lv = m[2]; }
  if (!ARM[t]) throw new Error('טכניקת יד לא מוכרת: ' + spec);
  return { t, lv, aim: aim !== undefined ? aim : null };
}
const aimYaw = (aim, fallback) => {
  if (aim === null || aim === undefined || aim === '') return fallback;
  if (aim in DIR) return rad(DIR[aim]);
  return rad(+aim);
};

// היקיטה: האגרוף ליד המותן, כף היד למעלה
function hikite(p, s, hipYaw) {
  const sg = s === 'L' ? -1 : 1;
  const R = Ry(hipYaw);
  const abs = add(p.pel, mv(R, [-0.03, 0.20, sg * 0.20]));
  return { h: sub(abs, chestOf(p)), e: mv(R, [-1, -0.15, sg * 0.25]), g: [PI, 0] };
}

// ---------- בעיטות ----------
// מוגדרות יחסית לאגן, במערכת כיוון הבעיטה (x אל המטרה). foot: מיקום הקרסול, kn: כיוון הברך.
export const KICK = {
  mae: { he: 'מאה־גרי', ja: 'mae-geri', lv: { c: [0.80, 0.02], j: [0.70, 0.42], g: [0.80, -0.30] }, ch: [0.32, -0.30], kn: [1, 0.6, 0], fp: 50 },
  yoko: { he: 'יוקו־גרי קאגה', ja: 'yoko-geri keage', lv: { c: [0.85, 0.08], j: [0.78, 0.40], g: [0.85, -0.25] }, ch: [0.22, -0.30], kn: [0.2, 1, 0], fp: 0, sideways: true },
  yokoK: { he: 'יוקו־גרי קקומי', ja: 'yoko-geri kekomi', lv: { c: [0.88, 0.04], j: [0.80, 0.36], g: [0.88, -0.25] }, ch: [0.25, -0.25], kn: [0.2, 1, 0], fp: 0, sideways: true },
  mika: { he: 'מיקזוקי־גרי', ja: 'mikazuki-geri', lv: { c: [0.42, 0.30], j: [0.40, 0.40], g: [0.40, 0.0] }, ch: [0.20, -0.25], kn: [1, 0.6, 0], fp: 20, inward: true },
  nami: { he: 'נמי־גאשי', ja: 'nami-gaeshi', lv: { c: [0.18, -0.48], j: [0.18, -0.48], g: [0.18, -0.48] }, ch: [0.18, -0.48], kn: [1, 0.3, 0], fp: 20, inward: true, small: true },
  hiza: { he: 'הייזה־גרי', ja: 'hiza-geri', lv: { c: [0.28, -0.40], j: [0.28, -0.40], g: [0.28, -0.40] }, ch: [0.20, -0.40], kn: [1, 1, 0], fp: 60, small: true },
};

// ---------- הרכבת תנוחה ----------
// st: מצב קודם {feet:{L,R}, fy:{L,R}, hip, head, pose}. מחזיר {pose, prep, post?, state, info}
export function composeMove(prev, m) {
  const S = STANCE[m.st];
  if (!S) throw new Error('עמידה לא מוכרת: ' + m.st);
  const lead = m.lead || 'L';
  const ls = lead === 'L' ? 1 : -1;               // שיקוף: לרגל ימין מובילה הופכים z וזוויות
  const other = lead === 'L' ? 'R' : 'L';
  const th = rad(DIR[m.dir] ?? +m.dir);

  // מיקומי כפות הרגליים במערכת העמידה
  const loc = {}, yaw = {}, pt = {}, lift = {};
  loc[lead] = [S.fr[0], S.fr[1] * ls]; loc[other] = [S.bk[0], S.bk[1] * ls];
  yaw[lead] = rad(S.frYaw * ls); yaw[other] = rad(S.bkYaw * ls);
  pt[lead] = rad(S.frPt || 0); pt[other] = rad(S.bkPt || 0);
  lift[lead] = S.frLift || 0; lift[other] = S.bkLift || 0;
  if (m.wide) { for (const f of ['L', 'R']) { loc[f][0] *= m.wide; loc[f][1] *= m.wide; } }

  // איפה המרכז: הרגל שלא זזה נשארת במקומה
  const mvf = m.mv || 'pivot';
  let c;
  const pf = prev.feet;
  if (mvf === 'L' || mvf === 'R') {
    const anc = mvf === 'L' ? 'R' : 'L';
    const r = rot2(th, loc[anc][0], loc[anc][1]);
    c = [pf[anc][0] - r[0], pf[anc][1] - r[1]];
  } else {
    const mid = [(pf.L[0] + pf.R[0]) / 2, (pf.L[1] + pf.R[1]) / 2];
    const lm = [(loc.L[0] + loc.R[0]) / 2, (loc.L[1] + loc.R[1]) / 2];
    const r = rot2(th, lm[0], lm[1]);
    c = [mid[0] - r[0], mid[1] - r[1]];
    if (m.by || m.byz) { const b = rot2(th, m.by || 0, m.byz || 0); c = [c[0] + b[0], c[1] + b[1]]; }
    if (m.go) { const b = rot2(rad(DIR[m.go[0]]), m.go[1], 0); c = [c[0] + b[0], c[1] + b[1]]; }
  }
  const feet = {};
  for (const f of ['L', 'R']) { const r = rot2(th, loc[f][0], loc[f][1]); feet[f] = [c[0] + r[0], c[1] + r[1]]; }

  // כיוון האגן: לפי העמידה ולפי סוג הטכניקה ביד המובילה
  const arms = { L: parseArm(m.L), R: parseArm(m.R) };
  let hipOff = S.hip0 !== undefined ? S.hip0 : 0;
  if (!S.side) {
    const main = ['L', 'R'].find((a) => arms[a] && ARM[arms[a].t].hip);
    if (main) {
      const kind = ARM[arms[main].t].hip;
      const same = main === lead;
      if (kind === 'h') hipOff = same ? (S.hipH ?? 45) : (S.hipG ?? -12);
      else if (kind === 's') { const fr = m.st === 'zen' || m.st === 'fudo' || m.st === 'hangetsu'; hipOff = same ? (fr ? 8 : (S.hip0 ?? 45) - 15) : (fr ? -10 : (S.hipG ?? 0)); }
    }
  }
  if (m.hip !== undefined) hipOff = m.hip;
  let hip = th + rad(hipOff * ls);
  if (m.turn === '+' || m.turn === '-') {
    let d = hip - prev.hip;
    d = ((d % (2 * PI)) + 2 * PI) % (2 * PI);          // [0, 2π)
    if (m.turn === '-' && d > 1e-6) d -= 2 * PI;
    hip = prev.hip + d;
  } else {
    hip = unwrapNear(hip, prev.hip);
    // סיבוב של 180 בדיוק: הכיוון לפי הרגל שזזה (רגל ימין זזה: מסתובבים ימינה)
    if (Math.abs(Math.abs(hip - prev.hip) - PI) < 0.02 && (mvf === 'L' || mvf === 'R')) {
      const want = mvf === 'R' ? 1 : -1;
      if (Math.sign(hip - prev.hip) !== want) hip += want * 2 * PI;
    }
  }
  const lookYaw = unwrapNear(aimYaw(m.look, th), hip);

  // תנוחה
  const p = {
    pel: [0, 0, 0], rot: [hip, rad(S.lean + (m.lean || 0)), 0], tw: rad(m.tw || 0), head: [lookYaw, rad(m.headP || 0)],
    fL: [0, 0, 0], fLr: [0, 0, 0], kL: [1, 0, 0], fR: [0, 0, 0], fRr: [0, 0, 0], kR: [1, 0, 0],
    hL: [0, 0, 0], eL: [0, 0, 0], gL: [0, 0], hR: [0, 0, 0], eR: [0, 0, 0], gR: [0, 0],
  };
  const pc = rot2(th, S.px + (m.px || 0), 0);
  p.pel = [c[0] + pc[0], S.h + (m.dh || 0), c[1] + pc[1]];
  const fy = {};
  for (const f of ['L', 'R']) {
    p['f' + f] = [feet[f][0], lift[f], feet[f][1]];
    fy[f] = unwrapNear(th + yaw[f], prev.fy[f]);
    p['f' + f + 'r'] = [fy[f], pt[f], 0];
    const kk = f === lead ? S.kf : S.kb;
    const out = f === 'L' ? -1 : 1;                   // כלפי חוץ
    p['k' + f] = toRel(th + yaw[f], [kk[0], 0, kk[1] * out]);
  }

  // ידיים
  const prevArms = prev.armSpec || { L: null, R: null };
  const single = ['L', 'R'].filter((a) => arms[a]);
  const res = {};
  for (const a of ['L', 'R']) {
    let sp = arms[a];
    if (!sp) {
      const o = arms[a === 'L' ? 'R' : 'L'];
      if (o && ARM[o.t].two) sp = { ...o };
      else if (o && ARM[o.t].pair) sp = { t: ARM[o.t].pair, lv: 'c', aim: o.aim };
      else if (o) sp = { t: 'hiki', lv: 'c', aim: null };
      else sp = { t: 'hold', lv: 'c', aim: null };
    }
    res[a] = sp;
  }
  const armOut = {}, armPrep = {};
  for (const a of ['L', 'R']) {
    const sp = res[a], s = a === 'L' ? -1 : 1, A = ARM[sp.t];
    const ay = aimYaw(sp.aim, A.body ? hip : lookYaw);
    if (sp.t === 'hold') {
      // היד נשארת במקומה יחסית לגוף: מסתובבת יחד עם האגן
      const dR = Ry(hip - prev.hip);
      armOut[a] = prev.pose ? { h: mv(dR, prev.pose['h' + a]), e: mv(dR, prev.pose['e' + a]), g: prev.pose['g' + a].slice() } : null; continue;
    }
    if (sp.t === 'hiki') { armOut[a] = { hiki: true }; continue; }
    const r = A.f(s, sp.lv);
    armOut[a] = { h: toRel(ay, r.h), e: toRel(ay, r.e), g: r.g };
    if (A.p) { const q = A.p(s, sp.lv); armPrep[a] = { h: toRel(ay, q.h), e: toRel(ay, q.e), g: q.g }; }
    const o = a === 'L' ? 'R' : 'L';
    if (A.o && res[o].t === 'hiki') { const q = A.o(-s, sp.lv); armPrep[o] = { h: toRel(ay, q.h), e: toRel(ay, q.e), g: q.g }; }
  }
  for (const a of ['L', 'R']) {
    if (!armOut[a] || armOut[a].hiki) armOut[a] = hikite(p, a, hip);
    p['h' + a] = armOut[a].h; p['e' + a] = armOut[a].e; p['g' + a] = armOut[a].g;
  }

  // בעיטה: הרגל שאינה תומכת נשלחת. התנוחה "pose" היא שיא הבעיטה, ו-post היא האיסוף.
  let post = null, kickInfo = null;
  if (m.kick) {
    const K = KICK[m.kick], leg = m.leg || other, sup = leg === 'L' ? 'R' : 'L';
    const kd = aimYaw(m.to, th), lv = m.lv || 'c';
    const sgn = leg === 'L' ? -1 : 1;
    // התומכת נשארת במקום: במצב בעיטה אין צעד, הרגל הבועטת יוצאת מהמקום שבו הייתה
    p['f' + sup] = [pf[sup][0], 0, pf[sup][1]];
    p['f' + leg] = [pf[leg][0], 0, pf[leg][1]];
    const hipY = 0.88;
    const back = toRel(kd, [K.sideways ? -0.06 : -0.04, 0, 0]);
    p.pel = [pf[sup][0] + back[0] + (K.sideways ? toRel(kd, [0, 0, 0])[0] : 0), hipY, pf[sup][1] + back[2]];
    if (K.sideways) {
      // הגוף בניצב לכיוון הבעיטה, נוטה מעט הרחק ממנה
      p.rot[2] = rad(14) * (leg === 'L' ? 1 : -1);
    }
    const [rx, ry] = K.lv[lv];
    const zOff = K.inward ? -sgn * 0.12 : (K.sideways ? 0 : sgn * 0.04);
    const tgt = add([p.pel[0], p.pel[1], p.pel[2]], toRel(kd, [rx, ry, zOff]));
    p['f' + leg] = [tgt[0], Math.max(0.05, tgt[1] - 0.08), tgt[2]];
    p['f' + leg + 'r'] = [unwrapNear(kd + (K.sideways ? -sgn * PI / 2 : 0), fy[leg]), rad(K.fp), 0];
    p['k' + leg] = toRel(kd, K.kn);
    p['k' + sup] = toRel(th, [1, 0, 0]);
    p['f' + sup + 'r'] = [unwrapNear(th + (K.sideways ? rad(-sgn * 60) : rad(sgn * 15)), prev.fy[sup]), 0, 0];
    const chm = add([p.pel[0], p.pel[1], p.pel[2]], toRel(kd, [K.ch[0], K.ch[1], zOff * 0.5]));
    post = clonePose(p);
    post['f' + leg] = [chm[0], Math.max(0.05, chm[1] - 0.04), chm[2]];
    post['k' + leg] = toRel(kd, [1, 0.8, 0]);
    post['f' + leg + 'r'] = [p['f' + leg + 'r'][0], rad(30), 0];
    // אחרי האיסוף: הרגל חוזרת אל מקומה הקודם, או נוחתת לפי שדה land בתנועה הבאה
    kickInfo = { leg, sup };
    feet[leg] = [pf[leg][0], pf[leg][1]]; feet[sup] = [pf[sup][0], pf[sup][1]];
  }

  // תנוחת הכנה: הרגל הזזה עוברת ליד התומכת, הידיים בהכנה, הראש כבר מביט אל הכיוון החדש
  const prevPose = prev.pose;
  const prep = clonePose(p);
  if (prevPose) {
    const mid = (k, t = 0.5) => prevPose[k].map((v, i) => v + (p[k][i] - v) * t);
    prep.pel = mid('pel'); prep.pel[1] = Math.min(prevPose.pel[1], p.pel[1]) + (m.air ? -0.12 : 0);
    prep.rot = mid('rot', m.prepTurn ?? 0.55);
    prep.rot[2] = 0;
    if (mvf === 'L' || mvf === 'R') {
      const anc = mvf === 'L' ? 'R' : 'L', s = mvf === 'L' ? -1 : 1;
      const a = prevPose['f' + anc], hy = prep.rot[0];
      const side = rot2(hy, 0, s * 0.10);
      prep['f' + mvf] = [a[0] + side[0], 0.035, a[2] + side[1]];
      prep['f' + anc] = a.slice();
      prep['f' + mvf + 'r'] = mid('f' + mvf + 'r');
      prep['f' + anc + 'r'] = mid('f' + anc + 'r', 0.6);
      prep['k' + mvf] = toRel(hy, [1, 0, 0]);
      prep['k' + anc] = mid('k' + anc);
    } else {
      for (const f of ['L', 'R']) { prep['f' + f] = mid('f' + f); prep['f' + f + 'r'] = mid('f' + f + 'r'); prep['k' + f] = mid('k' + f); }
    }
    if (m.kick) {
      // הכנה לבעיטה: ברך מורמת
      const leg = kickInfo.leg;
      prep.pel = p.pel.slice(); prep.rot = p.rot.slice();
      prep['f' + leg] = post['f' + leg].slice(); prep['k' + leg] = post['k' + leg].slice(); prep['f' + leg + 'r'] = post['f' + leg + 'r'].slice();
      for (const f of ['L', 'R']) if (f !== leg) { prep['f' + f] = p['f' + f].slice(); prep['f' + f + 'r'] = p['f' + f + 'r'].slice(); prep['k' + f] = p['k' + f].slice(); }
    }
    prep.head = p.head.slice();
    // ידיים בהכנה: מה שהטכניקה מגדירה, אחרת היד נשארת איפה שהייתה (אגרוף יוצא מההיקיטה)
    for (const a of ['L', 'R']) {
      const q = armPrep[a];
      if (q) { prep['h' + a] = q.h; prep['e' + a] = q.e; prep['g' + a] = q.g; }
      else if (res[a].t === 'hiki' || res[a].t === 'hold') { prep['h' + a] = mid('h' + a, 0.3); prep['e' + a] = mid('e' + a, 0.3); prep['g' + a] = mid('g' + a, 0.3); }
      else { prep['h' + a] = mid('h' + a, 0.15); prep['e' + a] = mid('e' + a, 0.15); prep['g' + a] = prevPose['g' + a].slice(); }
    }
  }

  // קפיצה: תנוחה באוויר בין ההכנה לנחיתה
  let air = null;
  if (m.air) {
    air = clonePose(p);
    const mid = (k) => prevPose[k].map((v, i) => v + (p[k][i] - v) * 0.5);
    air.pel = mid('pel'); air.pel[1] = 1.0 + (m.airH || 0.35);
    air.rot = mid('rot');
    for (const f of ['L', 'R']) {
      const fm = mid('f' + f);
      air['f' + f] = [fm[0], air.pel[1] - 0.62, fm[2]];
      air['k' + f] = toRel(air.rot[0], [1, 0.4, 0]);
    }
  }

  const state = { feet, fy, hip, pose: post || p, armSpec: res, dir: th };
  if (kickInfo) state.feet = feet;
  return { pose: p, prep: prevPose ? prep : null, air, post, state };
}

export function initialState(dir = 0) {
  return { feet: { L: [0, -0.06], R: [0, 0.06] }, fy: { L: 0, R: 0 }, hip: rad(dir), pose: null, armSpec: null };
}

// ---------- קאטה שלמה: רשימת תנועות -> ציר זמן ----------
// tempo: n רגיל, s איטי, f מהיר (מחובר לבא), x מהיר במיוחד. kiai: true. sub: true = אותה ספירה כמו הקודמת.
const TEMPO = { n: [0.95, 0.40], s: [2.6, 0.50], f: [0.55, 0.06], x: [0.38, 0.04], m: [1.5, 0.45] };

export function buildKata(kata) {
  let st = initialState();
  const out = [];
  // תנוחת פתיחה
  const startMove = { st: kata.start || 'hachiji', dir: 'F', L: kata.startArms || 'yoi', R: kata.startArms || 'yoi' };
  const open = composeMove(st, startMove);
  st = open.state;
  let count = 0;
  const lastM = kata.moves[kata.moves.length - 1];
  const moves = kata.moves.concat([{ ...startMove, mv: FRONT[lastM.st] ? (lastM.lead || 'L') : undefined, ...(kata.end || {}), he: 'יאמה', ja: 'yame', t: 'm', soft: true, sub: true, yame: true }]);
  let cont = false;
  for (const m of moves) {
    const c = composeMove(st, m);
    if (!m.sub && !cont) count++;
    cont = !!m.via;
    const [dur, hold] = TEMPO[m.t || 'n'];
    const [vd] = TEMPO.f;
    out.push({ m, count, dur: m.dur || (m.via && !m.t ? vd : dur), hold: m.via ? (m.hold ?? 0.05) : (m.hold ?? hold) + (m.kiai ? 0.5 : 0), ...c });
    st = c.state;
  }
  // תנועת מעבר (via) נושאת את השם של התנועה שאחריה
  for (let i = out.length - 1; i >= 0; i--) {
    const s = out[i];
    s.label = s.m.via && out[i + 1] ? out[i + 1].label : rawLabel(s.m);
    s.stance = s.m.via && out[i + 1] ? out[i + 1].stance : (STANCE[s.m.st] ? { he: STANCE[s.m.st].he, ja: STANCE[s.m.st].ja } : { he: '', ja: '' });
    if (s.m.kick && !s.m.st2) s.stance = { he: '', ja: '' };
  }
  return { open: open.pose, steps: out, count };
}

// שם התנועה לתצוגה
const LVJA = { j: 'jodan', c: 'chudan', g: 'gedan' };
const FRONT = { zen: 1, fudo: 1, hangetsu: 1, kok: 1, neko: 1 };
function rawLabel(m) {
  if (m.he !== undefined) return { he: m.he, ja: m.ja || '' };
  const parts = [];
  if (m.kick) { const K = KICK[m.kick]; parts.push({ he: K.he + (m.lv === 'j' ? ' ' + LVHE.j : ''), ja: K.ja }); }
  for (const a of ['L', 'R']) {
    const spec = m[a]; if (!spec) continue;
    const sp = parseArm(spec), A = ARM[sp.t];
    if (!A.he) continue;
    let he = A.he, ja = A.ja;
    if (sp.t === 'tsuki' && FRONT[m.st] && !m.kick) { const oi = (m.lead || 'L') === a; he = oi ? 'אוי־צוקי' : 'גיאקו־צוקי'; ja = oi ? 'oi-zuki' : 'gyaku-zuki'; }
    if (A.lv) { he += ' ' + LVHE[sp.lv]; ja += ' ' + LVJA[sp.lv]; }
    if (!parts.some((x) => x.he === he)) parts.push({ he, ja });
  }
  return { he: parts.map((x) => x.he).join(' + '), ja: parts.map((x) => x.ja).join(' + ') };
}
export function moveLabel(step) { return step.label || rawLabel(step.m); }
export function stanceLabel(step) { if (step.stance) return step.stance; const S = STANCE[step.m.st]; return S ? { he: S.he, ja: S.ja } : { he: '', ja: '' }; }
// תחילת התנועה שהצעד i שייך אליה (מדלגים אחורה על צעדי מעבר)
export function moveHead(K, i) { while (i > 0 && K.steps[i - 1].m.via) i--; return i; }

// ---------- ניגון ----------
// ציר זמן: לכל צעד [התחלה, הכנה, (אוויר), שיא, (איסוף), סוף החזקה]
export function timeline(K) {
  let t = 0.8; // דממה לפני
  const marks = [];
  for (const s of K.steps) {
    const t0 = t, d = s.dur;
    const tp = t0 + d * (s.air ? 0.35 : 0.5);
    const ta = s.air ? t0 + d * 0.65 : null;
    const tk = t0 + d;
    const tpost = s.post ? tk + 0.18 : null;
    t = (tpost || tk) + s.hold;
    marks.push({ t0, tp, ta, tk, tpost, t1: t });
  }
  return { marks, total: t + 1.0 };
}

const sm = (u) => { u = Math.max(0, Math.min(1, u)); return u * u * (3 - 2 * u); };
const late = (u, s, pw) => Math.pow(Math.max(0, (u - s) / (1 - s)), pw);
function lerpChain(a, b, u, chain, out) {
  for (const k in a) {
    const g = k[0] === 'h' && k !== 'head' || k[0] === 'e' || k[0] === 'g' ? 'arms' : (k === 'pel' || k === 'rot' || k === 'head' ? 'core' : (k === 'tw' ? 'core' : 'feet'));
    const t = chain[g](u), x = a[k], y = b[k];
    if (Array.isArray(x)) { const o = out[k] || (out[k] = new Array(x.length)); for (let i = 0; i < x.length; i++) o[i] = x[i] + (y[i] - x[i]) * t; }
    else out[k] = x + (y - x) * t;
  }
  return out;
}
const CH_PREP = { core: (u) => sm(u), feet: (u) => sm(u), arms: (u) => sm(u) };
const CH_KIME = { core: (u) => sm(u / 0.75), feet: (u) => sm(u / 0.65), arms: (u) => late(u, 0.25, 2.0) };
const CH_SOFT = { core: (u) => sm(u), feet: (u) => sm(u), arms: (u) => sm(u) };

// תנוחה בזמן t. מחזיר {pose, i (אינדקס הצעד), phase, kimeAge}
export function kataPoseAt(K, TL, t, out = {}) {
  const { marks } = TL;
  const n = K.steps.length;
  if (t <= marks[0].t0) return { pose: lerpChain(K.open, K.open, 0, CH_SOFT, out), i: -1, kimeAge: 9 };
  for (let i = 0; i < n; i++) {
    const s = K.steps[i], k = marks[i];
    if (t > k.t1 && i < n - 1) continue;
    const from = i === 0 ? K.open : (K.steps[i - 1].post || K.steps[i - 1].pose);
    const prep = s.prep || from;
    if (t <= k.tp) return { pose: lerpChain(from, prep, (t - k.t0) / (k.tp - k.t0), CH_PREP, out), i, kimeAge: 9 };
    if (s.air && t <= k.ta) return { pose: lerpChain(prep, s.air, (t - k.tp) / (k.ta - k.tp), CH_SOFT, out), i, kimeAge: 9 };
    const a = s.air || prep, ts = s.air ? k.ta : k.tp;
    if (t <= k.tk) return { pose: lerpChain(a, s.pose, (t - ts) / (k.tk - ts), s.m.soft ? CH_SOFT : CH_KIME, out), i, kimeAge: 9 };
    if (s.post && t <= k.tpost) return { pose: lerpChain(s.pose, s.post, (t - k.tk) / (k.tpost - k.tk), CH_SOFT, out), i, kimeAge: t - k.tk };
    const fin = s.post || s.pose;
    return { pose: lerpChain(fin, fin, 0, CH_SOFT, out), i, kimeAge: t - k.tk };
  }
  return { pose: K.steps[n - 1].pose, i: n - 1, kimeAge: 9 };
}
