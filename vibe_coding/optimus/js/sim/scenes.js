// הסצנות: מעבדה עם עמדות עבודה. כל חפץ סטטי (שולחן, מגש) הוא גיאומטריה קבועה; כל חפץ חופשי הוא גוף עם מסה.
// מיקומים במערכת MuJoCo: x קדימה מנקודת ההתחלה של הרובוט, y שמאלה, z למעלה.

// שולחן: משטח + 4 רגליים. כל החלקים מתחילים ב-"table" כדי לקבל את המראה הנכון
function table(name, cx, cy, hx, hy, top = 0.76, yaw = 0) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const R = (x, y) => [cx + c * x - s * y, cy + s * x + c * y];
  const out = [{ name: `table_${name}`, pos: [cx, cy, top - 0.02], size: [hx, hy, 0.02], yaw }];
  for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const [x, y] = R(sx * (hx - 0.04), sy * (hy - 0.04));
    out.push({ name: `tableLeg_${name}_${sx}${sy}`, pos: [x, y, (top - 0.04) / 2], size: [0.022, 0.022, (top - 0.04) / 2], yaw });
  }
  return out;
}
// מגש: תחתית + 4 דפנות נמוכות
function tray(name, cx, cy, hx, hy, z, wall = 0.03) {
  const t = 0.006;
  if (!wall) return [{ name: `tray_${name}_b`, pos: [cx, cy, z + t], size: [hx, hy, t] }];
  return [
    { name: `tray_${name}_b`, pos: [cx, cy, z + t], size: [hx, hy, t] },
    { name: `tray_${name}_n`, pos: [cx + hx, cy, z + wall], size: [t, hy, wall] },
    { name: `tray_${name}_s`, pos: [cx - hx, cy, z + wall], size: [t, hy, wall] },
    { name: `tray_${name}_e`, pos: [cx, cy + hy, z + wall], size: [hx, t, wall] },
    { name: `tray_${name}_w`, pos: [cx, cy - hy, z + wall], size: [hx, t, wall] },
  ];
}

export const STATIONS = {
  tableA: { pos: [0.95, 0], top: 0.76, approach: [0.48, 0, 0] },        // מול הרובוט
  tableB: { pos: [0.1, -1.75], top: 0.76, approach: [0.1, -1.25, -Math.PI / 2] },
  // עמדת תאי הסוללה: שורה של 3 תאי 4680 במגש מקור (מימין), ומגש יעד משמאל. הרובוט פונה ל-+y
  cells: { pos: [0.1, 1.75], top: 0.76, approach: [0.23, 1.32, Math.PI / 2],
    src: { x: [0.51, 0.43, 0.35], y: 1.55, tray: [0.43, 1.55] },
    dst: { x: [-0.12, -0.03, 0.06], y: 1.55, tray: [-0.03, 1.55] },  // ממלאים משמאל לימין: היד תמיד מימין לתא שהיא מניחה
    pickStand: [0.23, 1.32], placeStand: [-0.33, 1.32], lift: 0.1, liftDst: 0.17 },
};

export const SCENES = {
  lab: {
    name: 'מעבדה',
    build() {
      const A = STATIONS.tableA, B = STATIONS.tableB, C = STATIONS.cells;
      const statics = [
        ...table('a', A.pos[0], A.pos[1], 0.3, 0.6, A.top),
        ...table('b', B.pos[0], B.pos[1], 0.55, 0.3, B.top),
        ...table('c', C.pos[0], C.pos[1], 0.8, 0.3, C.top),
        // המגשים על במות של 10 ס"מ, כמו במדפים במפעל: כך האצבעות לא נגררות על השולחן
        { name: 'step_src', pos: [C.src.tray[0], C.src.tray[1], C.top + C.lift / 2], size: [0.145, 0.06, C.lift / 2] },
        { name: 'step_dst', pos: [C.dst.tray[0], C.dst.tray[1], C.top + C.liftDst / 2], size: [0.195, 0.07, C.liftDst / 2] },
        ...tray('src', C.src.tray[0], C.src.tray[1], 0.14, 0.055, C.top + C.lift, 0),
        ...tray('dst', C.dst.tray[0], C.dst.tray[1], 0.19, 0.065, C.top + C.liftDst, 0),
      ];
      // תא 4680: קוטר 46 מ"מ, גובה 80 מ"מ, כ-355 גרם
      const cells = C.src.x.map((x, i) => ({ name: `cell${i + 1}`, pos: [x, C.src.y, C.top + C.lift + 0.012 + 0.04], geoms: [{ type: 'cylinder', size: [0.0232, 0.04], mass: 0.355 }], friction: 0.8 }));
      const objects = [
        { name: 'box1', pos: [A.pos[0] - 0.16, 0.0, A.top + 0.09], geoms: [{ type: 'box', size: [0.11, 0.16, 0.09], mass: 2.2 }], friction: 0.9 },
        { name: 'cup1', pos: [A.pos[0] - 0.12, -0.38, A.top + 0.055], geoms: [{ type: 'cylinder', size: [0.037, 0.05], mass: 0.25 }], friction: 0.8 },
        ...cells,
        { name: 'ball1', pos: [-1.6, -1.3, 0.11], geoms: [{ type: 'sphere', size: [0.11], mass: 0.45 }], friction: 0.7 },
      ];
      return { statics, objects, looks: { cup1: 'cup' } };
    },
  },
  empty: {
    name: 'רצפה ריקה',
    build() { return { statics: [], objects: [] }; },
  },
};
