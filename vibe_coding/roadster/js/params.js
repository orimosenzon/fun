// נתוני הרכב. כל ערך מסומן במקור שלו:
//  'tesla'  = נאמר באירוע (15/10/2026) או מופיע באתר טסלה
//  'report' = דווח בכתבות (paultan, electrek, techcrunch) ולא אושר רשמית
//  'est'    = הערכה הנדסית שלנו, כי טסלה לא פרסמה
import { BAR, PSI } from './gas.js';

export const SOURCES = {
  tesla: 'טסלה',
  report: 'דיווח',
  est: 'הערכה',
};

// מיקומי מנועי הסילון בגוף הרכב: x שמאלה, y למעלה, z קדימה, ראשית במרכז הכובד
const MAIN4 = (F) => [
  { id: 'M1', group: 'main', pos: [0.62, -0.17, -2.14], dir: [0, 0, 1], F },
  { id: 'M2', group: 'main', pos: [0.24, -0.20, -2.16], dir: [0, 0, 1], F },
  { id: 'M3', group: 'main', pos: [-0.24, -0.20, -2.16], dir: [0, 0, 1], F },
  { id: 'M4', group: 'main', pos: [-0.62, -0.17, -2.14], dir: [0, 0, 1], F },
];
const CORNERS = (F) => [
  { id: 'FL', group: 'lift', pos: [0.84, -0.31, 1.58], dir: [0, 1, 0], F },
  { id: 'FR', group: 'lift', pos: [-0.84, -0.31, 1.58], dir: [0, 1, 0], F },
  { id: 'RL', group: 'lift', pos: [0.84, -0.31, -1.66], dir: [0, 1, 0], F },
  { id: 'RR', group: 'lift', pos: [-0.84, -0.31, -1.66], dir: [0, 1, 0], F },
];
// שני נחירי סבסוב בחרטום: כל אחד נושף הצידה ודוחף את החרטום לצד השני
const YAW2 = (F) => [
  { id: 'YL', group: 'yaw', pos: [0.55, -0.12, 1.98], dir: [-1, 0, 0], F },
  { id: 'YR', group: 'yaw', pos: [-0.55, -0.12, 1.98], dir: [1, 0, 0], F },
];

const COMMON = {
  length: 4.35, width: 2.0, height: 1.15,
  wheelbase: 2.68, trackF: 1.70, trackR: 1.68,
  cgHeight: 0.42,             // est
  frontWeight: 0.43,          // est, מנוע אחורי כפול וסוללה במרכז
  wheelRadius: 0.345,         // est, צמיגי Pirelli P Zero Trofeo RS
  wheelInertia: 1.3,          // kg·m², est
  suspension: { rest: 0.25, k: 260000, bumpAt: 0.085, kBump: 3.0e6, cBump: 9000, cReb: 12000, arbF: 80000, arbR: 60000, staticComp: 0.025 }, // קפיצים קשיחים כמו ברכב מרוץ: היניקה מוסיפה 4.5 טון
  brakeTorque: 4200,          // N·m לגלגל, est
  fan: true,                  // מערכת יניקה בכל הדגמים ('tesla')
  floorY: -0.33,              // גובה הרצפה בגוף הרכב (90 מ"מ מעל הקרקע במנוחה)
  fillPressure: 5500 * PSI,   // report: הקריאה "Thruster 5,500 psi" במסך הרכב
};

export const VARIANTS = {
  performance: {
    ...COMMON,
    key: 'performance',
    name: 'Performance',
    nameHe: 'פרפורמנס',
    tagline: 'מוכן לכביש. שני מושבים, אווירודינמיקה אקטיבית, יניקה רב-מצבית',
    seats: 2,
    massDry: 1860,            // est, כולל נהג
    tankVolume: 0,
    thrusters: [],
    motorPower: 800e3,        // est
    wheelTorqueMax: 11000,    // est, סה"כ בגלגלים
    frontTorqueShare: 0.4,
    vMax: 94,                 // tesla/report: 210 mph
    claimed060: 1.6,          // tesla: "פחות מ-1.6 שניות"
  },
  thrust: {
    ...COMMON,
    key: 'thrust',
    name: 'Thrust',
    nameHe: 'ת\'ראסט',
    tagline: 'מוכן למסלול. מנועי גז קר אחוריים לתאוצה',
    seats: 2,
    massDry: 1930,            // est: 1,860 + מכל 200 ליטר וצנרת
    tankVolume: 0.2,          // report (electrek): 200 ליטר
    thrusters: MAIN4(9500),   // est: מכויל ל-0–60 של 0.9 שניות
    motorPower: 1000e3,       // report: "מעל מגה-ואט"
    wheelTorqueMax: 12000,
    frontTorqueShare: 0.4,
    vMax: 103,                // tesla: 230 mph
    claimed060: 0.9,
  },
  rocket: {
    ...COMMON,
    key: 'rocket',
    name: 'Rocket',
    nameHe: 'רוקט',
    tagline: 'מוכן לאוויר. מושב אחד, 10 נחירים, מערכת טיסה מלאה',
    seats: 1,
    massDry: 2010,            // est: מושב אחד פחות, 600 ליטר מכלים ו-10 נחירים
    tankVolume: 0.6,          // tesla (מוראווי): 600 ליטר
    thrusters: [...MAIN4(20000), ...CORNERS(22000), ...YAW2(4000)], // tesla: 10 נחירים, מעל 80 קילוניוטון אחורה. הפינות מכוילות ל-3.3g בבלימת הנפילה מהקיר (הקריאה בהדגמה)
    motorPower: 1000e3,
    wheelTorqueMax: 12000,
    frontTorqueShare: 0.4,
    vMax: 104,
    claimed060: 0.59,
  },
};

export const AERO = {
  // CdA, ClA (מ"ר) לכל מצב של הכנף האחורית והמשטחים הקדמיים
  modes: {
    road:  { nameHe: 'כביש', CdA: 0.58, ClA: 0.25, wing: 0 },
    low:   { nameHe: 'גרר נמוך', CdA: 0.62, ClA: 0.55, wing: 0.25 },
    high:  { nameHe: 'כוח הצמדה', CdA: 0.82, ClA: 1.55, wing: 1 },
  },
  copLong: -0.15,   // מרכז הלחץ קצת מאחורי מרכז הכובד (יציבות)
};

export const FAN_MODES = {
  off:  { nameHe: 'כבוי' },
  auto: { nameHe: 'אוטומטי' },
  max:  { nameHe: 'Max Grip' },
};

export { BAR, PSI };
