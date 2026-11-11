// The story: chapters, each a short list of steps (fly through a ring, reach a height, land in a circle).
import { POI, groundAt } from './sim/world.js';

export const PROLOGUE = [
  'יונה היה הדוור של האי. שלושים שנה אותו סיבוב: הכפר, הטחנה, המגדלור, והעלייה הארוכה אל הבקתה שבהר.',
  'בוקר אחד, בחורשה שמעל המצוק, מעד על שורש מוזר. כתום מבפנים, מריר כמו קליפת אשכולית. הוא טעם ממנו. למה? גם הוא לא ידע.',
  'למחרת הרים את שק הדואר ביד אחת. אחרי שבוע עקר עמוד גדר בלי להתכוון. הידיים שלו כבר לא היו ידיים של דוור.',
  'בלילות, במחסן שעל המצוק, תפר מפרשים ישנים על מוטות במבוק. הוא ידע בדיוק מה הוא בונה, ולא סיפר לאף אחד.',
  'הבוקר הרוח באה מהים. יונה עומד על שפת המצוק, והכנפיים מקופלות על גבו.',
];
export const EPILOGUE = 'מאז, הדואר באי מגיע מהשמיים. ואם תשאלו את יונה מה השתנה, הוא יגיד שהסיבוב נשאר אותו סיבוב. רק הדרך נהייתה יפה יותר.';

const K = (s) => `<kbd>${s}</kbd>`;
const ring = (x, y, z, text, r = 17) => ({ type: 'ring', p: [x, y, z], r, text });
const above = (key, dy, text, r = 16) => ({ type: 'ring', p: [POI[key][0] + 12, groundAt(POI[key][0] + 12, POI[key][2] + 14) + dy, POI[key][2] + 14], r, text });

// where the plateau ends, walking west from the take-off stones
function cliffEdge() { const z = POI.cliff[2]; let x = POI.cliff[0]; while (x > -2600 && groundAt(x - 6, z) > groundAt(x, z) - 4) x -= 3; return x; }

export function buildChapters() {
  const ex = cliffEdge(), cz = POI.cliff[2], W = Math.PI;
  const [bx, , bz] = POI.beach, [px, py, pz] = POI.peak;
  const home = { type: 'land', p: [POI.cliff[0], POI.cliff[1], cz], r: 55 };
  return [
    {
      title: 'קפיצת אמונה', sub: 'פרק ראשון',
      intro: 'מתחת למצוק יש מאה ושבעים מטר של אוויר. הכנפיים עוד לא נוסו.',
      start: { kind: 'ground', x: ex + 34, z: cz, heading: W }, tutorialFall: true,
      steps: [
        { type: 'cond', test: (b) => b.mode === 'air', text: `רוץ אל שפת המצוק ${K('W')} או ${K('↑')}` },
        { type: 'cond', test: (b) => b.fold > 0.9, text: `פרוש כנפיים! ${K('רווח')}`, big: true },
        ring(ex - 190, 162, cz + 110, `אתה עף. ${K('←')} ${K('→')} להטות ולפנות, ${K('↑')} ${K('↓')} להרים ולהוריד את האף`),
        ring(ex - 250, 128, cz + 520, 'עבור בטבעות. הרוח שעולה על המצוק מחזיקה אותך באוויר'),
        ring(bx - 60, 82, bz - 420, 'המשך דרומה, אל המפרץ הקטן'),
        { type: 'land', p: [bx, POI.beach[1], bz], r: 60, text: `נחת בעיגול שעל החוף. רגע לפני המגע משוך למעלה ${K('↑')} כדי לבלום` },
      ],
      done: 'נחתת על החול, שלם. שלושים שנה הלכת ברגל על האי הזה.',
    },
    {
      title: 'משק כנפיים', sub: 'פרק שני',
      intro: 'לגלוש למטה זה דבר אחד. לעלות בחזרה, בכוח הזרועות, זה כבר סיפור אחר.',
      start: { kind: 'ground', x: bx + 50, z: bz, heading: W },
      steps: [
        { type: 'cond', test: (b) => b.mode === 'air' && b.agl > 6, text: `החזק ${K('רווח')}: ריצה במשק כנפיים אל מול הרוח, עד שהרגליים עוזבות את החול` },
        ring(bx - 190, 40, bz - 220, `המשך לנפנף ${K('רווח')}. שים לב למד הכוח: הזרועות מתעייפות`),
        ring(ex - 150, 95, cz + 560, 'היצמד למצוק. האוויר שעולה עליו ירים אותך בלי מאמץ'),
        ring(ex - 120, 165, cz + 120, 'נוח קצת. בגלישה הכוח חוזר'),
        ring(ex - 110, 235, cz - 330, 'עוד קצת גובה'),
        { ...home, text: 'חזור ונחת ליד המחסן, על ראש המצוק' },
      ],
      done: 'חזרת למעלה בכוחות עצמך. השרירים בוערים, אבל אתה מחייך כמו ילד.',
    },
    {
      title: 'אוויר חם', sub: 'פרק שלישי',
      intro: 'ההר גבוה מדי בשביל זרועות, אפילו כאלה. אבל העיטים מגיעים לשם כל יום, ואף פעם לא נראים מתאמצים.',
      start: { kind: 'ground', x: POI.fields[0] + 330, z: POI.fields[2] + 40, heading: W },
      steps: [
        { type: 'cond', test: (b) => b.mode === 'air' && b.agl > 10, text: `המרא מהשדה ${K('רווח')}` },
        { type: 'alt', y: 420, text: 'מעל השדות החמים האוויר עולה. חפש את העיטים שחגים, היכנס מתחתם וחוג איתם' },
        { type: 'alt', y: 760, text: 'נפלא. הישאר במעגל, או דלג אל העמוד הבא, מתחת לענן הבא' },
        ring(px, py + 75, pz, 'עכשיו אל הפסגה', 26),
      ],
      done: 'כל האי פרוש מתחתיך. מכאן אפילו המגדלור נראה כמו צעצוע.',
    },
    {
      title: 'צלילת בז', sub: 'פרק רביעי',
      intro: 'בז נודד מקפל כנפיים ונופל מהשמיים במהירות של רכבת. יונה תמיד רצה לדעת איך זה מרגיש.',
      start: { kind: 'air', p: [px - 260, py + 400, pz], heading: W, speed: 14 },
      steps: [
        { type: 'speed', v: 55.6, text: `קפל כנפיים ${K('Shift')} והורד את האף ${K('↓')}. הגע ל-200 קמ"ש` },
        { type: 'cond', test: (b) => b.fold > 0.9 && b.gload > 2, text: `עכשיו פרוש ${K('שחרר Shift')} ומשוך ${K('↑')}`, big: true },
        ring(-950, 330, pz + 100, 'מערבה, אל הים', 22),
        ring(ex + 260, 232, cz, 'מעל המצוק', 20),
        ring(-2830, 34, -440, 'ובין שני הסלעים שבים', 22),
      ],
      done: 'מאתיים קמ"ש, ובלי כרטיס. הצעיף כמעט נשאר מאחור.',
    },
    {
      title: 'דואר אוויר', sub: 'פרק חמישי',
      intro: 'שק הדואר מלא, והסיבוב אותו סיבוב. רק שהיום יונה לא הולך ברגל.',
      start: { kind: 'ground', x: ex + 34, z: cz, heading: W }, timed: true,
      steps: [
        { type: 'cond', test: (b) => b.mode === 'air' && b.fold > 0.9, text: `רוץ אל שפת המצוק ופרוש כנפיים ${K('רווח')}` },
        above('light', 14, 'מכתב ראשון: שומר המגדלור'),
        { type: 'ring', p: [POI.boat[0], 16, POI.boat[2]], r: 16, text: 'גלויה לדייג שבסירה' },
        above('village', 16, 'חבילה לכיכר הכפר'),
        above('mill', 14, 'חשבון חשמל לטוחן. מצטערים'),
        above('hut', 14, 'ומכתב אהבה לרועה שבהר. זה רחוק וגבוה: חפש אוויר חם', 18),
        { ...home, text: 'הביתה. נחת ליד המחסן' },
      ],
      done: EPILOGUE,
    },
    { title: 'טיסה חופשית', sub: 'האי שלך', free: true, intro: 'אין משימות. רק רוח, כנפיים ואי שלם.', start: { kind: 'ground', x: ex + 34, z: cz, heading: W }, steps: [] },
  ];
}

export class Game {
  constructor(bird, ui) {
    this.bird = bird; this.ui = ui; this.chapters = buildChapters();
    this.ci = 0; this.si = 0; this.state = 'idle'; this.time = 0; this.timeScale = 1; this.fallT = 0; this.doneT = 0;
    try { this.unlocked = +(localStorage.getItem('birdman.unlocked') || 0); this.best = JSON.parse(localStorage.getItem('birdman.best') || '{}'); } catch { this.unlocked = 0; this.best = {}; }
  }
  get chapter() { return this.chapters[this.ci]; }
  get step() { return this.chapter.steps[this.si]; }
  start(i) {
    this.ci = i; this.si = 0; this.time = 0; this.state = 'play'; this.doneT = 0;
    this.check = { ...this.chapter.start, si: 0 };
    this.spawn();
    this.ui.markers(this.chapter.steps.filter((s) => s.type === 'ring' || s.type === 'land'));
    this.ui.title(this.chapter.title, this.chapter.sub);
    this.ui.objective(this.step ? this.step.text : 'עוף לאן שתרצה', this.step && this.step.big);
  }
  spawn() {
    const c = this.check, b = this.bird;
    if (c.kind === 'ground') b.placeGround(c.x, c.z, c.heading); else b.placeAir(c.p, c.heading, c.speed);
    b.events.length = 0; this.si = c.si; this.fallT = 0;
    this.ui.respawned();
    this.ui.objective(this.step ? this.step.text : 'עוף לאן שתרצה', this.step && this.step.big);
  }
  // index of the current step among the markers (rings and landing circles)
  get markerIndex() { let n = 0; for (let i = 0; i < this.si; i++) { const t = this.chapter.steps[i].type; if (t === 'ring' || t === 'land') n++; } return n; }
  update(dt) {
    const b = this.bird, ch = this.chapter;
    this.timeScale = 1;
    if (this.state !== 'play' && this.state !== 'done') return;
    if (this.state === 'play' && this.si > 0) this.time += dt;
    // the first fall: the world holds its breath until the wings open
    if (ch.tutorialFall && this.si === 1 && b.mode === 'air') { this.fallT += dt; if (this.fallT > 0.45) this.timeScale = 0.22; }
    if ((b.mode === 'crash' || b.mode === 'water') && b.crashT > 2.3) { this.spawn(); return; }
    for (const e of b.events) {
      if (e.type === 'land') { this.ui.flash(e.q > 0.6 ? 'נחיתה מושלמת' : e.q > 0.3 ? 'נחיתה טובה' : 'נחיתה קשה', ''); }
      if (e.type === 'crash') this.ui.flash('אאוץ׳', 'נסה שוב. לפני הקרקע האט ומשוך למעלה');
      if (e.type === 'splash') this.ui.flash('פלופ', 'הכנפיים לא אוהבות מים');
    }
    const s = this.step;
    if (!s || this.state !== 'play') { if (this.state === 'done') { this.doneT += dt; if (this.doneT > 2.2 && !this.cardShown) { this.cardShown = true; this.ui.chapterDone(this); } } return; }
    let ok = false;
    if (s.type === 'cond') ok = s.test(b);
    else if (s.type === 'alt') ok = b.p[1] >= s.y;
    else if (s.type === 'speed') ok = b.mode === 'air' && b.airspeed >= s.v;
    else if (s.type === 'ring') ok = Math.hypot(b.p[0] - s.p[0], b.p[1] - s.p[1], b.p[2] - s.p[2]) < s.r;
    else if (s.type === 'land') ok = b.mode === 'ground' && Math.abs(b.gspeed) < 3 && Math.hypot(b.p[0] - s.p[0], b.p[2] - s.p[2]) < s.r;
    if (ok) {
      this.si++;
      if (s.type === 'ring' || s.type === 'alt' || s.type === 'speed') this.ui.chime(this.si === ch.steps.length);
      if (s.type === 'land') { this.check = { kind: 'ground', x: b.p[0], z: b.p[2], heading: b.heading, si: this.si }; this.ui.chime(true); }
      if (this.si >= ch.steps.length) {
        this.state = 'done'; this.cardShown = false; this.doneT = 0;
        if (this.ci + 1 > this.unlocked) this.unlocked = this.ci + 1;
        if (ch.timed && (!this.best[this.ci] || this.time < this.best[this.ci])) this.best[this.ci] = +this.time.toFixed(1);
        try { localStorage.setItem('birdman.unlocked', this.unlocked); localStorage.setItem('birdman.best', JSON.stringify(this.best)); } catch {}
        this.ui.objective('', false);
      } else this.ui.objective(this.step.text, this.step.big);
    }
  }
}
