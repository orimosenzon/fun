// מפרט הרובוט: מידות, מסות, מפרקים ומפעילים.
// כל ערך מסומן במקור שלו: tesla (נאמר רשמית ע"י טסלה/מאסק), report (דווח/דליפה/פטנט), est (הערכה הנדסית שלנו).
// ציר z למעלה, x קדימה, y שמאלה (המוסכמה של MuJoCo). מטרים, קילוגרמים, ניוטון-מטר.

export const SRC = { tesla: 'tesla', report: 'report', est: 'est' };

export const BODY = {
  height: { v: 1.73, src: 'tesla', note: 'AI Day 2021: 5 רגל 8 אינץ\'. לא פורסם מחדש ל-V3' },
  mass: { v: 57, src: 'tesla', note: 'AI Day 2021 (125 ליברות). ל-V3 לא פורסמה מסה' },
  payload: { v: 20, src: 'tesla', note: 'AI Day 2021: 45 ליברות נשיאה' },
  battery: { v: 2.3, src: 'tesla', note: 'AI Day 2022: 2.3 קוט"ש, 52 וולט, בתוך החזה' },
  busVoltage: { v: 52, src: 'tesla' },
  powerIdle: { v: 100, src: 'tesla', note: 'AI Day 2022: כ-100 וואט בישיבה/עמידה' },
  powerWalk: { v: 500, src: 'tesla', note: 'AI Day 2022: כ-500 וואט בהליכה מהירה' },
  actuators: { v: 28, src: 'tesla', note: '28 מפעילים מבניים (בלי הידיים), משישה סוגים' },
  handActuators: { v: 25, src: 'tesla', note: 'מאסק: 50 מפעילים ברובוט, 25 בכל אמה+יד. בהמשך אמר שהתכנון השתנה' },
  handDof: { v: 22, src: 'report', note: 'פטנטים מ-2024/2026 ודליפת האפליקציה: 22 דרגות חופש ליד ולשורש כף היד' },
  parts: { v: 10000, src: 'tesla', note: 'מאסק, 2026: כ-10,000 חלקים ייחודיים' },
};

// ששת סוגי המפעילים שהוצגו ב-AI Day 2022
export const ACTUATOR_TYPES = {
  R20: { kind: 'rotary', rating: 20, unit: 'N·m', src: 'tesla', label: 'סיבובי 20 נ"מ' },
  R110: { kind: 'rotary', rating: 110, unit: 'N·m', src: 'tesla', label: 'סיבובי 110 נ"מ' },
  R180: { kind: 'rotary', rating: 180, unit: 'N·m', src: 'tesla', label: 'סיבובי 180 נ"מ' },
  L500: { kind: 'linear', rating: 500, unit: 'N', src: 'tesla', label: 'קווי 500 ניוטון' },
  L3900: { kind: 'linear', rating: 3900, unit: 'N', src: 'tesla', label: 'קווי 3,900 ניוטון' },
  L8000: { kind: 'linear', rating: 8000, unit: 'N', src: 'tesla', label: 'קווי 8,000 ניוטון' },
  TENDON: { kind: 'tendon', rating: 0, unit: '', src: 'report', label: 'מנוע באמה + גיד' },
};

// גיאומטריה בסיסית (est, לפי פרופורציות אנושיות לגובה 173 ס"מ ולתמונות)
export const G = {
  hipZ: 0.905,        // גובה מפרק הירך מהרצפה בעמידה ישרה
  hipY: 0.088,        // חצי המרווח בין מפרקי הירכיים
  thigh: 0.415,
  shin: 0.405,
  ankleZ: 0.085,      // גובה הקרסול מעל הסוליה
  footLen: 0.245, footW: 0.098, footBack: 0.068,  // הסוליה: מהעקב לבהונות, מאחורי הקרסול
  pelvisZ: 0.975,     // מרכז האגן (גוף הבסיס הצף)
  waistZ: 1.045,      // מפרק המותן
  shoulderZ: 1.425, shoulderY: 0.205,
  neckZ: 1.49,
  headC: 1.615,       // מרכז הראש
  upperArm: 0.285, forearm: 0.255,
  palmLen: 0.098, palmW: 0.084, palmT: 0.032,
};

// המפרקים של הגוף. torque: מומנט שיא; speed: מהירות בלי עומס (רד'/ש'); armature: אינרציה מוחזרת של הרוטור (ק"ג·מ"ר)
// type: סוג המפעיל, לפי ההקצאה הסבירה בין 28 המפעילים. range ברדיאנים.
export const JOINTS = [
  // רגליים: 6 לכל רגל
  ...['l', 'r'].flatMap(s => [
    { name: `${s}_hip_yaw`, group: 'leg', type: 'R110', torque: 110, speed: 7, armature: 0.08, range: s === 'l' ? [-0.5, 0.9] : [-0.9, 0.5], damping: 0.6, frictionloss: 0.4 },
    { name: `${s}_hip_roll`, group: 'leg', type: 'R180', torque: 180, speed: 6, armature: 0.15, range: s === 'l' ? [-0.4, 0.75] : [-0.75, 0.4], damping: 0.6, frictionloss: 0.4 },
    { name: `${s}_hip_pitch`, group: 'leg', type: 'R180', torque: 180, speed: 6, armature: 0.15, range: [-2.3, 0.7], damping: 0.6, frictionloss: 0.4 },
    { name: `${s}_knee`, group: 'leg', type: 'L8000', torque: 300, speed: 7, armature: 0.18, range: [0, 2.45], damping: 0.6, frictionloss: 0.5 },
    { name: `${s}_ankle_pitch`, group: 'leg', type: 'L3900', torque: 170, speed: 8, armature: 0.07, range: [-0.9, 0.7], damping: 0.4, frictionloss: 0.3 },
    { name: `${s}_ankle_roll`, group: 'leg', type: 'L3900', torque: 90, speed: 8, armature: 0.05, range: [-0.45, 0.45], damping: 0.4, frictionloss: 0.3 },
  ]),
  // מותן: 2
  { name: 'waist_yaw', group: 'torso', type: 'R180', torque: 180, speed: 5, armature: 0.12, range: [-1.1, 1.1], damping: 0.8, frictionloss: 0.4 },
  { name: 'waist_pitch', group: 'torso', type: 'R180', torque: 180, speed: 5, armature: 0.15, range: [-0.35, 1.0], damping: 0.8, frictionloss: 0.4 },
  // צוואר: 2 (Gen 2 הוסיף צוואר עם 2 דרגות חופש)
  { name: 'neck_yaw', group: 'neck', type: 'R20', torque: 20, speed: 8, armature: 0.006, range: [-1.2, 1.2], damping: 0.15, frictionloss: 0.05 },
  { name: 'neck_pitch', group: 'neck', type: 'R20', torque: 20, speed: 8, armature: 0.006, range: [-0.5, 0.7], damping: 0.15, frictionloss: 0.05 },
  // ידיים: 5 בזרוע + 2 בשורש כף היד
  ...['l', 'r'].flatMap(s => [
    { name: `${s}_shoulder_pitch`, group: 'arm', type: 'R110', torque: 110, speed: 7, armature: 0.05, range: [-3.1, 1.0], damping: 0.3, frictionloss: 0.2 },
    { name: `${s}_shoulder_roll`, group: 'arm', type: 'R110', torque: 110, speed: 7, armature: 0.05, range: s === 'l' ? [-0.25, 2.9] : [-2.9, 0.25], damping: 0.3, frictionloss: 0.2 },
    { name: `${s}_shoulder_yaw`, group: 'arm', type: 'R20', torque: 40, speed: 9, armature: 0.02, range: [-1.6, 1.6], damping: 0.2, frictionloss: 0.1 },
    { name: `${s}_elbow`, group: 'arm', type: 'L3900', torque: 110, speed: 8, armature: 0.03, range: [-2.4, 0], damping: 0.2, frictionloss: 0.1 },
    { name: `${s}_forearm_roll`, group: 'arm', type: 'R20', torque: 20, speed: 10, armature: 0.006, range: [-1.7, 1.7], damping: 0.08, frictionloss: 0.03 },
    { name: `${s}_wrist_flex`, group: 'wrist', type: 'L500', torque: 18, speed: 9, armature: 0.003, range: [-1.1, 1.1], damping: 0.05, frictionloss: 0.02 },
    { name: `${s}_wrist_dev`, group: 'wrist', type: 'L500', torque: 14, speed: 9, armature: 0.003, range: [-0.45, 0.55], damping: 0.05, frictionloss: 0.02 },
  ]),
];

// אצבעות: 4 דרגות חופש לכל אצבע (פטנט V3): הרחקה, MCP, PIP, DIP. מונעות בגידים ממנועים באמה.
export const FINGERS = [
  // name, x on palm (forward), len of 3 phalanges, radius
  { key: 'thumb', segs: [0.046, 0.034, 0.028], r: 0.0105 },
  { key: 'index', x: 0.031, segs: [0.044, 0.026, 0.021], r: 0.0088 },
  { key: 'middle', x: 0.010, segs: [0.048, 0.029, 0.022], r: 0.009 },
  { key: 'ring', x: -0.011, segs: [0.045, 0.027, 0.021], r: 0.0086 },
  { key: 'little', x: -0.031, segs: [0.036, 0.022, 0.019], r: 0.0078 },
];
export const FINGER_JOINTS = ['abd', 'mcp', 'pip', 'dip'];
export const FINGER_FORCE = 22;   // ניוטון בקצה האצבע (est). מומנט במפרק = כוח × זרוע
export const FINGER_KP = 0.9;     // נ"מ לרדיאן, ההגבר של מפעיל המיקום

// מסות (est) כך שהסכום ≈ 57 ק"ג
export const MASS = {
  pelvis: 7.6, abdomen: 1.2, torso: 17.0, head: 2.3,
  upperArm: 1.9, forearm: 2.1, palm: 0.42, finger: 0.035,
  thigh: 5.4, shin: 2.9, foot: 1.0,
};

export function totalMass() {
  const fingerCount = FINGERS.length * 3;
  return MASS.pelvis + MASS.abdomen + MASS.torso + MASS.head +
    2 * (MASS.upperArm + MASS.forearm + MASS.palm + fingerCount * MASS.finger) +
    2 * (MASS.thigh + MASS.shin + MASS.foot);
}

// תנוחה נומינלית: ברכיים מעט כפופות, כמו שרובוטים עומדים בפועל
export const NOMINAL = {
  knee: 0.34, hipPitch: -0.17, anklePitch: -0.17,
  shoulderRoll: 0.12, elbow: -0.35, shoulderPitch: 0.06,
};
