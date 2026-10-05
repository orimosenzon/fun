/* batch.js: פענוח רשימת קטעים מטקסט.
 *
 * כל שורה עם טווח זמנים = סרטון אחד. כמה טווחים באותה שורה ("And", "+", פסיק,
 * מה שבא) מתחברים לסרטון אחד לפי הסדר. שורות בלי טווח (כותרת כמו "Session 9:")
 * מדולגות.
 *
 * זמנים: 02.49 = 2 דקות ו-49 שניות, 1.00.14 = שעה ו-14 שניות. נקודתיים עובדים
 * באותה צורה (2:49, 1:00:14). ככה יורם כותב אותם ב-Pages.
 */
window.C = window.C || {};

C.batch = (() => {
  const T = String.raw`\d{1,2}(?:[.:]\d{1,2}){1,2}`;
  const RANGE = new RegExp(`(${T})\\s*[-–—]\\s*(${T})`, 'g');
  const LONE = new RegExp(`(^|[^\\d.:])${T}($|[^\\d.:])`);

  /** "1.00.14" → 3614, או NaN אם דקות/שניות מעל 59 */
  function clock(str) {
    const p = str.split(/[.:]/).map(Number);
    if (p.slice(1).some((x) => x > 59)) return NaN;
    return p.reduce((acc, x) => acc * 60 + x, 0);
  }

  /**
   * @returns [{ n, line, text, ranges: [[from, to]], errors: [key] }]
   *   n = מספר הסרטון (1..), line = מספר השורה בטקסט (1..)
   */
  function parse(text) {
    const lines = String(text || '').replace(/\r\n?|\u2028|\u2029/g, '\n').split('\n');
    const items = [];
    lines.forEach((raw, i) => {
      const s = raw.trim();
      if (!s) return;
      const ranges = [];
      const errors = [];
      for (const m of s.matchAll(RANGE)) {
        const a = clock(m[1]), b = clock(m[2]);
        if (isNaN(a) || isNaN(b)) errors.push('bBadTime');
        else if (b <= a) errors.push('bEndBeforeStart');
        else ranges.push([a, b]);
      }
      const rest = s.replace(RANGE, ' ');
      if (!ranges.length && !errors.length) {
        // שורה עם זמן בודד בלי טווח: כנראה טעות הקלדה, ולא כותרת
        if (LONE.test(rest)) items.push({ n: 0, line: i + 1, text: s, ranges, errors: ['bNoRange'] });
        return;
      }
      if (LONE.test(rest)) errors.push('bStrayTime');
      items.push({ n: 0, line: i + 1, text: s, ranges, errors });
    });
    let n = 0;
    for (const it of items) if (it.ranges.length) it.n = ++n;
    return items;
  }

  /** קטע שחופף ברובו לקטע קודם: כנראה תיקון של אותה שורה ולא סרטון נוסף */
  function similarTo(items) {
    const out = new Map();
    const ov = (r, q) => Math.max(0, Math.min(r[1], q[1]) - Math.max(r[0], q[0]));
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < i; j++) {
        const a = items[i], b = items[j];
        if (!a.n || !b.n) continue;
        const hit = a.ranges.some((r) => b.ranges.some((q) => ov(r, q) > 0.5 * Math.min(r[1] - r[0], q[1] - q[0])));
        if (hit) { out.set(a, b.n); break; }
      }
    }
    return out;
  }

  const length = (it) => it.ranges.reduce((s, [a, b]) => s + b - a, 0);

  return { parse, similarTo, length, clock };
})();
