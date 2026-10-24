// תקציב המדינה: עוגה שלחיצה על פלח הופכת אותו לעוגה משלו, עד התקנה.
// מבוסס על "לאן הולך הכסף" של המועצה (transparency/budget), עם מה שמתווסף ברמה הארצית:
// שלושה שלבים לכל תקנה (בחוק / אחרי שינויים / בפועל), תיקון לאינפלציה ושקל לתושב,
// ו"מי קיבל את הכסף" בזמן אמת ממפתח התקציב.
// הנתונים נבנים ב-tools/build_budget.py:
//   data/years.json            השנים, השלבים שיש לכל אחת והסיכומים
//   data/budget_<שנה>.json     העץ של השנה. לכל עלה v = [מקורי, מאושר, ביצוע] באלפי ₪
//   data/series/top.json       כל צומת עד רמת המשרד לאורך השנים
//   data/series/<צד>_<סעיף>.json  מה שבתוך משרד אחד, נטען כשנכנסים אליו
//   data/context.json          מדד ואוכלוסייה (tools/fetch_context.py)

const MAX_SLICES = 8;
const DUR = 420;
const PRINCIPAL = ['S62', 'S63'];            // קרן החוב, וקרן החוב לביטוח הלאומי
const PHASE_NAME = ['כפי שאושר בחוק', 'אחרי שינויים', 'בפועל'];
const PHASE_SHORT = ['בחוק', 'אחרי שינויים', 'בפועל'];
const OB = 'https://next.obudget.org';

// תיקוני כתיב בשמות כפי שהם בקובץ
const RENAME = {
  'בטחון וסדר ציבורי': 'ביטחון וסדר ציבורי',
  'בטחון': 'ביטחון',
  'בטחון פנים': 'ביטחון פנים',
  'בטחון-אחר': 'ביטחון: שונות',
  'קליטת עליה': 'קליטת עלייה',
  'כלכלה ותעשיה': 'כלכלה ותעשייה',
  'קרן': 'החזר קרן החוב',
  'קרן - ביטוח לאומי': 'החזר קרן החוב לביטוח הלאומי',
  'ריבית': 'ריבית על החוב',
  'הכנסות למימון גירעון': 'הלוואות והכנסות הון',
  'תקבולי מע"מ ברוטו': 'מע"מ',
  'משרד הבטחון': 'משרד הביטחון',
};
// הסברים קצרים רק היכן שהשם לבדו לא מספיק
const EXPLAIN = {
  S62: 'החזר של הלוואות שהמדינה לקחה בעבר. רובו ממומן בהלוואות חדשות (מחזור החוב), ולכן הוא מופיע גם בצד ההכנסות, כ"הלוואות". אפשר להסתיר אותו בתיבה "כולל החזר קרן החוב".',
  S63: 'החזר של אגרות החוב שהמדינה הנפיקה לביטוח הלאומי, שמשקיע בהן את הכספים שלו.',
  S61: 'הריבית על החוב של המדינה. זו הוצאה של ממש, שלא חוזרת.',
  S25: 'הכסף שהמדינה מעבירה לביטוח הלאומי: השתתפות בקצבאות (זקנה, ילדים, נכות) ופיצוי על ההנחות בדמי הביטוח. רוב הקצבאות עצמן משולמות מדמי הביטוח הלאומי, שלא עוברים בתקציב.',
  S86: 'הלוואות שהמדינה לוקחת (בעיקר אגרות חוב בארץ ובחו"ל), גבייה של הלוואות שנתנה, ומכירת נכסים. ההלוואות מממנות את הגירעון ואת ההחזר של הלוואות ישנות.',
  SM: 'הכנסות שמשרד גובה לעצמו ומוציא בהתאם: אגרות, מכירות, וגם כסף מחו"ל, למשל הסיוע הביטחוני האמריקאי שנרשם במשרד הביטחון.',
  S73: 'כסף שלא שויך מראש למשרד, ומחולק במהלך השנה.',
  S71: 'פנסיה לגמלאי המדינה שבפנסיה תקציבית, ופיצויים.',
  P54: 'בעיקר הוצאות שעוברות דרך האוצר: השתתפות בפנסיה צוברת, ביטוחים והתחשבנויות.',
  S81: 'מס הכנסה, מס חברות, מס שבח ומס רכישה.',
  S82: 'מע"מ, מס קנייה, מכס ומס על הדלק.',
};

const $ = (s) => document.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* מצב פרטי */ } },
};
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ---------- עיצוב מספרים ----------
const nf = new Intl.NumberFormat('he-IL');
function money(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toLocaleString('he-IL', { maximumFractionDigits: a >= 1e11 ? 0 : 1 })} מיליארד ₪`;
  if (a >= 1e6) return `${(v / 1e6).toLocaleString('he-IL', { maximumFractionDigits: a >= 1e8 ? 0 : 1 })} מיליון ₪`;
  if (a >= 1e3) return `${Math.round(v / 1e3).toLocaleString('he-IL')} אלף ₪`;
  return `${nf.format(Math.round(v))} ₪`;
}
const moneyFull = (v) => `${nf.format(Math.round(v))} ₪`;
function pctNum(p) {
  if (p === 0) return '0%';
  if (Math.abs(p) < 0.1) return 'פחות מ-0.1%';
  return `${p.toLocaleString('he-IL', { maximumFractionDigits: Math.abs(p) < 10 ? 1 : 0 })}%`;
}
const pct = (part, whole) => pctNum(whole ? (part / whole) * 100 : 0);
function change(cur, prev) {
  if (prev == null) return null;
  if (!prev && !cur) return null;
  if (!prev) return { text: 'סעיף חדש', cls: 'up' };
  if (!cur) return { text: `בוטל ב-${state.year}`, cls: 'down' };
  const d = ((cur - prev) / prev) * 100;
  if (Math.abs(d) < 0.5) return { text: 'ללא שינוי', cls: '' };
  const s = `${Math.abs(d).toLocaleString('he-IL', { maximumFractionDigits: Math.abs(d) < 10 ? 1 : 0 })}%`;
  return d > 0 ? { text: `▲ ${s}`, cls: 'up' } : { text: `▼ ${s}`, cls: 'down' };
}
const nameOf = (n) => RENAME[n] || n;

// ---------- מצב ----------
// כל מה שרואים על המסך נשמר ב-state, וכל state נכתב לכתובת (#y=2026&s=e&p=...)
const state = { year: null, side: 'expense', item: null, view: 'one', unit: 'nis', phase: 0, debt: true };
let years = [], YEARS = [], ctx = null, books = {}, trees = {}, current, grand;
const SER = {}, serLoaded = {}, serCache = {};
const COLORS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const colorOf = (d, i) => (d.kind === 'more' ? css('--other') : css(COLORS[i % COLORS.length]));
const yearMeta = (y = state.year) => years.find((m) => m.year === y);
const sideWord = () => (state.side === 'expense' ? 'ההוצאות' : 'ההכנסות');
const yi = (y) => YEARS.indexOf(y);

// השלב שמוצג בפועל: אם אין לשנה את השלב שנבחר (2026 בלי ביצוע, 2020 בלי חוק) לוקחים את הקרוב
function phaseOf(y = state.year) {
  const m = yearMeta(y);
  if (!m) return state.phase;
  if (m.phases.includes(state.phase)) return state.phase;
  return m.phases.reduce((a, b) => (Math.abs(b - state.phase) < Math.abs(a - state.phase) ? b : a));
}

function pathOf(n) { const p = []; while (n) { p.unshift(n); n = n.parent; } return p; }
function taxes() { const v = +$('#tax').value; return v > 0 ? v : 0; }
function myShare(v) { const a = taxes(); return a && state.side === 'expense' ? (v / grand) * a : 0; }
const cpiNow = () => ctx.cpi[ctx.cpi.length - 1];
function realOf(v, y) { const i = ctx.years.indexOf(y); return i < 0 ? null : v * cpiNow() / ctx.cpi[i]; }
function popOf(y) { const i = ctx.years.indexOf(y); return i < 0 ? null : ctx.pop[i]; }

// ---------- בניית העץ ----------
const KIND = { D: 'domain', S: 'sub', P: 'office', A: 'area', G: 'program', I: 'item' };
function buildTree(book, side) {
  const s = side[0];
  const ph = phaseOf(book.year);
  const root = { name: `תקציב ${book.year}`, kind: 'root', sk: `${s}|ROOT`, key: 'ROOT', children: [] };
  function make(raw, parent) {
    if (side === 'expense' && !state.debt && PRINCIPAL.includes(raw.k)) return null;
    const n = { name: nameOf(raw.n), raw: raw.n, key: raw.k, sk: `${s}|${raw.k}`, kind: KIND[raw.k[0]], parent };
    n.sec = raw.k[0] === 'P' ? raw.k.slice(1) : parent.sec;
    if (raw.c) {
      n.children = raw.c.map((c) => make(c, n)).filter(Boolean);
      if (!n.children.length) return null;
    } else {
      n.leaf = raw; n.code = raw.k.slice(1);
      n.v3 = raw.v.map((x) => x * 1000);
      n.value = Math.max(n.v3[ph], 0);
      n.salary = raw.s ? Math.max(raw.s[ph] * 1000, 0) : 0;
      n.jobs = raw.j ? raw.j[ph] : 0;
      if (!n.v3.some((x) => x > 0)) return null;
    }
    return n;
  }
  root.children = book[side].map((c) => make(c, root)).filter(Boolean);
  // סכומים מלמטה למעלה, ומיון מהגדול לקטן
  (function sum(n) {
    if (!n.children) return;
    n.children.forEach(sum);
    n.children.sort((a, b) => b.value - a.value);
    n.value = d3.sum(n.children, (c) => c.value);
    n.v3 = [0, 1, 2].map((i) => d3.sum(n.children, (c) => c.v3[i]));
    n.salary = d3.sum(n.children, (c) => c.salary);
    n.jobs = d3.sum(n.children, (c) => c.jobs);
  })(root);
  // רמה עם ילד אחד (למשל "חינוך" ← "משרד החינוך"): מדלגים על הרמה הכפולה
  (function flatten(n) {
    if (!n.children) return;
    while (n.kind !== 'root' && n.children.length === 1 && n.children[0].children) {
      const only = n.children[0];
      n.children = only.children;
      n.sec = n.sec || only.sec;
      n.alias = n.alias || only;         // הקוד של המשרד, לשאילתות ולסדרות
      if (n.key === 'D6' && n.alias === only) n.name = only.name;   // בלי הקרן נשארת רק "ריבית על החוב"
    }
    n.children.forEach((c) => { c.parent = n; flatten(c); });
  })(root);
  return root;
}

// ילדים להצגה: עד MAX_SLICES, ואם יש יותר – 7 הגדולים ופלח "עוד"
function slicesOf(node) {
  if (node._slices) return node._slices;
  const kids = node.children.filter((c) => c.value > 0);
  if (kids.length <= MAX_SLICES) return (node._slices = kids);
  const top = kids.slice(0, MAX_SLICES - 1);
  const rest = kids.slice(MAX_SLICES - 1);
  const more = {
    name: `עוד ${rest.length} סעיפים קטנים`, kind: 'more', children: rest, parent: node, sec: node.sec,
    value: d3.sum(rest, (c) => c.value), v3: [0, 1, 2].map((i) => d3.sum(rest, (c) => c.v3[i])),
    salary: d3.sum(rest, (c) => c.salary), jobs: d3.sum(rest, (c) => c.jobs),
  };
  return (node._slices = [...top, more]);
}

// ---------- סדרות לאורך השנים ----------
// קובץ הסדרות של משרד נטען רק כשנכנסים אליו
async function ensureSeries(node) {
  const files = new Set(['top']);
  if (node && node.sec) files.add(`${state.side[0]}_${node.sec}`);
  await Promise.all([...files].map(async (f) => {
    if (serLoaded[f]) return serLoaded[f];
    serLoaded[f] = fetch(`data/series/${f}.json`).then((r) => r.json()).then((d) => Object.assign(SER, d.nodes))
      .catch(() => { delete serLoaded[f]; });
    return serLoaded[f];
  }));
}
// סדרה של מפתח אחד: שלושה מערכים (מקורי, מאושר, ביצוע) באורך כל השנים, בשקלים
function rawSeries(sk) {
  if (serCache[sk]) return serCache[sk];
  const e = SER[sk];
  if (!e) return null;
  const off = yi(e[0]);
  return (serCache[sk] = [1, 2, 3].map((k) => YEARS.map((_, i) => {
    const v = e[k][i - off];
    return i - off >= 0 && v != null ? v * 1000 : null;
  })));
}
function seriesOf(node) {
  if (node.kind === 'more' || (node.kind === 'root' && state.side === 'expense' && !state.debt)) {
    const parts = node.children.map(seriesOf);
    return [0, 1, 2].map((k) => YEARS.map((_, i) => {
      const xs = parts.map((p) => p[k][i]).filter((x) => x != null);
      return xs.length ? d3.sum(xs) : null;
    }));
  }
  // לצומת שקופל יש גם את הסדרה של הילד היחיד. הסדרה שלו עצמו עדיפה (P00 של ההכנסות יושב מתחת לכל סוגי המסים),
  // חוץ מהחזרי החוב בלי הקרן, שם נשארה רק הריבית
  const own = rawSeries(node.sk), al = node.alias && rawSeries(node.alias.sk);
  if (node.key === 'D6' && !state.debt && al) return al;
  return own || al || [0, 1, 2].map(() => YEARS.map(() => null));
}
// ערך בשנה, בשלב שמוצג באותה שנה
function valAt(s, i) { const y = YEARS[i]; const p = phaseOf(y); return s[p][i]; }
function prevOf(node) {
  const i = yi(state.year);
  if (i <= 0) return null;
  const s = seriesOf(node);
  const v = valAt(s, i - 1);
  return v == null ? 0 : v;
}
// מכנה לאחוזים: כל הצד, בלי קרן החוב אם היא מוסתרת
function denom() {
  const s = seriesOf(trees[state.side]);
  return YEARS.map((_, i) => valAt(s, i));
}

// ---------- העוגה ----------
const W = 460, R = 210, R0 = 120;
const svg = d3.select('#pie').attr('viewBox', `${-W / 2} ${-W / 2} ${W} ${W}`);
const gSlices = svg.append('g');
const gLabels = svg.append('g');
const gCenter = svg.append('g').attr('class', 'center');
const gHint = svg.append('g').attr('class', 'hint');
const arc = d3.arc().innerRadius(R0).outerRadius(R).cornerRadius(4).padAngle(0.008);
const pieLayout = d3.pie().value((d) => d.value).sort(null);
const tip = $('#tooltip');
let arcsNow = [];

function layout(node) {
  const sl = slicesOf(node);
  return pieLayout(sl).map((a, i) => ({ ...a, color: colorOf(a.data, i), key: a.data.name + '|' + i }));
}
function arcTween(from, to) {
  // רק הזוויות: ב-datum יש גם את הצומת עם הפניה להורה, ו-d3.interpolate היה נכנס אליו לעומק
  const pick = (a) => ({ startAngle: a.startAngle, endAngle: a.endAngle, padAngle: a.padAngle ?? 0.008 });
  const i = d3.interpolate(pick(from), pick(to));
  return (t) => arc(i(t));
}
function textColorFor(hex) {
  const c = d3.rgb(hex);
  const lum = (0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b) / 255;
  return lum > 0.6 ? '#0b0b0b' : '#ffffff';
}
function drawLabels(arcs) {
  const lab = gLabels.selectAll('text').data(arcs.filter((a) => a.endAngle - a.startAngle > 0.32), (a) => a.key);
  lab.exit().remove();
  lab.enter().append('text').attr('class', 'pct').attr('text-anchor', 'middle').attr('dy', '0.35em')
    .merge(lab)
    .attr('transform', (a) => `translate(${arc.centroid(a)})`)
    .attr('fill', (a) => textColorFor(a.color))
    .text((a) => pct(a.data.value, current.value))
    .attr('opacity', 0).transition().duration(200).attr('opacity', 1);
}
function drawCenter(node) {
  gCenter.selectAll('*').remove();
  const up = node.kind !== 'root';
  gCenter.classed('up', up).on('click', up ? () => goUp() : null);
  gCenter.append('circle').attr('r', R0 - 4).attr('fill', 'transparent');
  const name = node.kind === 'root' ? `כל ${sideWord()} ${state.year}`
    : node.kind === 'more' ? `הקטנים של ${node.parent.name}` : node.name;
  const lines = wrap(name, 16);
  lines.forEach((l, i) => gCenter.append('text').attr('class', 'center-name').attr('text-anchor', 'middle')
    .attr('y', -22 - (lines.length - 1 - i) * 18).text(l));
  gCenter.append('text').attr('class', 'center-total').attr('text-anchor', 'middle').attr('y', 12).text(money(node.value));
  gCenter.append('text').attr('class', up ? 'center-back' : 'center-hint').attr('text-anchor', 'middle').attr('y', 42)
    .text(up ? 'לחצו כאן לחזרה ↩' : 'לחצו על פלח לפירוט 👆');
}
function wrap(s, n) {
  const words = s.split(' '); const out = [''];
  for (const w of words) {
    if ((out[out.length - 1] + ' ' + w).trim().length > n && out[out.length - 1]) out.push(w);
    else out[out.length - 1] = (out[out.length - 1] + ' ' + w).trim();
  }
  return out.slice(0, 3);
}
function bindSlices(sel) {
  // הכיוון החוצה מהמרכז, לאנימציית הרמז
  sel.each(function (a) {
    const m = (a.startAngle + a.endAngle) / 2;
    this.style.setProperty('--dx', `${(Math.sin(m) * 7).toFixed(1)}px`);
    this.style.setProperty('--dy', `${(-Math.cos(m) * 7).toFixed(1)}px`);
  });
  sel.on('click', (ev, a) => select(a.data))
    .on('mousemove', (ev, a) => showTip(ev, a))
    .on('mouseleave', () => { tip.hidden = true; hot(null); });
}
function placeTip(t, ev, box) {
  t.hidden = false;
  let x = ev.clientX - box.left + 14, y = ev.clientY - box.top + 14;
  if (x + t.offsetWidth > box.width) x = ev.clientX - box.left - t.offsetWidth - 14;
  t.style.left = `${Math.max(0, x)}px`; t.style.top = `${y}px`;
}
function showTip(ev, a) {
  const d = a.data;
  const ch = change(d.value, prevOf(d));
  const mine = myShare(d.value);
  const inside = d.kind === 'more' ? `<span class="muted">ביחד: ${esc(d.children.slice(0, 12).map((c) => c.name).join(', '))}${d.children.length > 12 ? '…' : ''}</span><br>` : '';
  tip.innerHTML = `<b>${esc(d.name)}</b>${inside}${money(d.value)}<br>
    <span class="muted">${pct(d.value, current.value)} מ${esc(current.kind === 'root' ? 'כל התקציב' : current.name)}` +
    (current.kind !== 'root' ? ` · ${pct(d.value, grand)} מכל התקציב` : '') + '</span>' +
    (ch ? `<br><span class="chg ${ch.cls}">${ch.text}</span> <span class="muted">לעומת ${state.year - 1}</span>` : '') +
    (mine ? `<br><span class="mine-share">מהמסים שלכם: ${moneyFull(mine)}</span>` : '');
  placeTip(tip, ev, $('.pie-box').getBoundingClientRect());
  hot(d);
}
function hot(d) {
  gSlices.selectAll('path.slice').classed('dim', (a) => d && a.data !== d);
  document.querySelectorAll('#list li').forEach((li) => li.classList.toggle('hot', !!d && li._d === d));
  d3.selectAll('#chart .line').classed('dim', (s) => d && s.node !== d);
}
// רמז חד-פעמי: הפלח הגדול בולט ופועם ואצבע מקישה עליו, עד הלחיצה הראשונה (נזכר בדפדפן)
let hintPlanned = false;
function planHint() {
  if (hintPlanned || store.get('pieLearned')) return;
  hintPlanned = true;
  setTimeout(playHint, 1000);
}
function playHint() {
  if (store.get('pieLearned') || busy || !arcsNow.length) return;
  const big = arcsNow.reduce((a, b) => (b.endAngle - b.startAngle > a.endAngle - a.startAngle ? b : a));
  gSlices.selectAll('path.slice').filter((a) => a.key === big.key).classed('nudge', true);
  gLabels.selectAll('text').filter((a) => a.key === big.key).classed('hint-hide', true);
  const [x, y] = arc.centroid(big);
  gHint.append('text').attr('class', 'hint-hand').attr('text-anchor', 'middle').attr('x', x).attr('y', y + 30).text('👆');
  setTimeout(clearHint, 3 * 1300);
}
function clearHint() {
  gHint.selectAll('*').remove();
  gSlices.selectAll('path.nudge').classed('nudge', false);
  gLabels.selectAll('text.hint-hide').classed('hint-hide', false);
}
function learned() { store.set('pieLearned', '1'); clearHint(); }
function afterMove(node, arcs) {
  arcsNow = arcs;
  planHint();
  drawCenter(node); drawSide(node, arcs); drawCrumbs(node); drawChart(); drawWho(node); syncURL();
}

// ציור רגיל, בלי אנימציה
function render(node) {
  current = node;
  const arcs = layout(node);
  gSlices.selectAll('path:not(.slice)').remove();
  const p = gSlices.selectAll('path.slice').data(arcs, (a) => a.key);
  p.exit().remove();
  const all = p.enter().append('path').attr('class', 'slice').merge(p)
    .attr('fill', (a) => a.color).attr('d', arc).each(function (a) { this._cur = a; });
  bindSlices(all);
  drawLabels(arcs);
  afterMove(node, arcs);
}
// כניסה לפלח: הפלח מתרחב לכל העיגול, ואז הילדים שלו נפרשים מתוכו
function zoomIn(target) {
  const paths = gSlices.selectAll('path.slice');
  let hitIndex = -1;
  paths.each(function (a, i) { if (a.data === target) hitIndex = i; });
  gLabels.selectAll('text').remove();
  tip.hidden = true;
  const t1 = paths.transition().duration(DUR).ease(d3.easeCubicInOut)
    .attrTween('d', function (a, i) {
      const to = i === hitIndex ? { ...a, startAngle: 0, endAngle: 2 * Math.PI, padAngle: 0 }
        : i < hitIndex ? { ...a, startAngle: 0, endAngle: 0 } : { ...a, startAngle: 2 * Math.PI, endAngle: 2 * Math.PI };
      return arcTween(this._cur, to);
    });
  const color = hitIndex >= 0 ? d3.select(paths.nodes()[hitIndex]).attr('fill') : css('--other');
  t1.end().catch(() => {}).then(() => {
    current = target;
    const arcs = layout(target);
    const bg = gSlices.append('path').attr('fill', color).attr('d', arc({ startAngle: 0, endAngle: 2 * Math.PI, padAngle: 0 }));
    gSlices.selectAll('path.slice').remove();
    const ps = gSlices.selectAll('path.slice').data(arcs, (a) => a.key).enter().append('path').attr('class', 'slice')
      .attr('fill', (a) => a.color).each(function (a) { this._cur = a; });
    ps.transition().duration(DUR).ease(d3.easeCubicOut)
      .attrTween('d', (a) => arcTween({ ...a, startAngle: 0, endAngle: 0 }, a));
    bg.transition().duration(DUR).attr('opacity', 0).remove();
    bindSlices(ps);
    afterMove(target, arcs);
    setTimeout(() => drawLabels(arcs), DUR);
  });
}
// יציאה: הילדים מתקפלים, ואז ההורה חוזר כשהפלח שממנו באנו מתכווץ למקומו
function zoomOut(to) {
  const from = current;
  const child = pathOf(from).find((n) => n.parent === to) || from;
  gLabels.selectAll('text').remove();
  tip.hidden = true;
  gSlices.selectAll('path.slice').transition().duration(DUR * 0.7).ease(d3.easeCubicIn)
    .attrTween('d', function (a) { return arcTween(this._cur, { ...a, startAngle: 0, endAngle: 0 }); })
    .end().catch(() => {}).then(() => {
      current = to;
      const arcs = layout(to);
      let hit = arcs.findIndex((a) => a.data === child);
      if (hit < 0) hit = arcs.findIndex((a) => a.data.kind === 'more' && (child.kind === 'more' || a.data.children.includes(child)));
      gSlices.selectAll('path.slice').remove();
      const ps = gSlices.selectAll('path.slice').data(arcs, (a) => a.key).enter().append('path').attr('class', 'slice')
        .attr('fill', (a) => a.color).each(function (a) { this._cur = a; });
      ps.transition().duration(DUR).ease(d3.easeCubicInOut).attrTween('d', (a, i) => {
        const start = i === hit ? { ...a, startAngle: 0, endAngle: 2 * Math.PI, padAngle: 0 }
          : i < hit ? { ...a, startAngle: 0, endAngle: 0 } : { ...a, startAngle: 2 * Math.PI, endAngle: 2 * Math.PI };
        return arcTween(hit < 0 ? { ...a, startAngle: 0, endAngle: 0 } : start, a);
      });
      bindSlices(ps);
      afterMove(to, arcs);
      setTimeout(() => drawLabels(arcs), DUR);
    });
}

let busy = false;
async function select(d) {
  if (busy) return;
  learned();
  if (d.kind === 'item') { openItem(d); return; }
  busy = true;
  await ensureSeries(d);
  setTimeout(() => { busy = false; }, DUR * 2 + 50);
  zoomIn(d);
}
function goUp() { if (current.parent && !busy) goTo(current.parent); }
async function goTo(n) {
  if (n === current) return;
  busy = true;
  await ensureSeries(n);
  setTimeout(() => { busy = false; }, DUR * 2 + 50);
  if (pathOf(current).includes(n)) zoomOut(n);
  else { render(n); busy = false; }
}

// ---------- הצד: כותרת ורשימה ----------
function codeOf(node) {
  const n = node.alias && !['area', 'program', 'item'].includes(node.kind) ? node.alias : node;
  if (!n.key || n.key === 'ROOT' || !'PAGI'.includes(n.key[0])) return null;
  return n.key.slice(1);
}
function obLink(node) {
  const c = codeOf(node);
  if (!c || state.side !== 'expense') return '';
  return `<a href="${OB}/i/budget/00${c}/${state.year}" target="_blank" rel="noopener">במפתח התקציב</a>`;
}
function drawSide(node, arcs) {
  const info = $('#node-info');
  const ph = phaseOf();
  const ch = change(node.value, prevOf(node));
  const isRoot = node.kind === 'root';
  const title = isRoot ? (state.side === 'expense' ? 'לאן הולך הכסף' : 'מאיפה בא הכסף')
    : node.kind === 'more' ? `הסעיפים הקטנים של ${node.parent.name}` : node.name;
  const explain = EXPLAIN[node.key] || (node.alias && EXPLAIN[node.alias.key]) || '';
  const facts = [];
  facts.push(`<span><b>${money(node.value)}</b> ב-${state.year}, ${PHASE_SHORT[ph]}</span>`);
  if (!isRoot) facts.push(`<span><b>${pct(node.value, grand)}</b> מכל ${sideWord()}</span>`);
  if (ch) facts.push(`<span>לעומת ${state.year - 1}: <b class="chg ${ch.cls}">${ch.text}</b></span>`);
  const pop = popOf(state.year);
  if (pop && node.value) {
    const per = node.value / pop;
    facts.push(`<span>לתושב: <b>${per >= 10 ? moneyFull(per) : `${per.toLocaleString('he-IL', { maximumFractionDigits: 2 })} ₪`}</b> בשנה</span>`);
  }
  const m = yearMeta();
  if (state.side === 'expense' && m.phases.includes(0) && m.phases.includes(2) && node.v3[0] > 0) {
    const r = node.v3[2] / node.v3[0] * 100;
    const cls = r > 105 ? 'up' : r < 95 ? 'down' : '';
    facts.push(`<span>בפועל מול החוק: <b class="chg ${cls}">${pctNum(r)}</b></span>`);
  }
  if (state.side === 'expense' && node.salary > 0) facts.push(`<span>שכר: <b>${pct(node.salary, node.value)}</b></span>`);
  if (state.side === 'expense' && node.jobs > 0) facts.push(`<span>משרות: <b>${Math.round(node.jobs).toLocaleString('he-IL')}</b></span>`);
  const mine = myShare(node.value);
  if (mine && !isRoot) facts.push(`<span class="mine-share">מהמסים שלכם: <b>${moneyFull(mine)}</b></span>`);
  const link = obLink(node);
  const code = codeOf(node);
  info.innerHTML = `<h2>${esc(title)}</h2>
    <p class="sub">${node.kind === 'more'
    ? `${node.children.length} סעיפים קטנים של ${esc(node.parent.name)}, שמוצגים כאן יחד כי כל אחד מהם קטן מכדי לקבל פלח משלו. לחצו על סעיף כדי לראות רק אותו.`
    : `${node.children.length} ${isRoot ? 'תחומים' : 'סעיפים'}`}${code ? ` · קוד ${code}` : ''}${link ? ` · ${link}` : ''}</p>
    ${explain ? `<p class="sub">${esc(explain)}</p>` : ''}
    <div class="facts">${facts.join('')}</div>`;

  const colorFor = new Map(arcs.map((a) => [a.data, a.color]));
  const list = $('#list');
  list.innerHTML = '';
  const max = d3.max(node.children, (c) => c.value) || 1;
  const kids = node.children.filter((c) => c.value > 0 || c.v3.some((x) => x > 0));
  for (const c of kids.slice(0, 300)) {
    const li = document.createElement('li');
    li._d = colorFor.has(c) ? c : null;
    const col = colorFor.get(c) || css('--other');
    const ch2 = change(c.value, prevOf(c));
    const mine2 = myShare(c.value);
    const salaryTag = c.leaf && c.leaf.t === 'שכר' ? '<span class="tag">שכר</span>' : '';
    li.innerHTML = `<span class="sw" style="background:${col}"></span>
      <span class="name" title="${esc(c.name)}">${esc(c.name)}${salaryTag}</span>
      <span class="amt">${money(c.value)}</span>
      <span class="bar"><i style="width:${(c.value / max) * 100}%;background:${col}"></i></span>
      <span class="meta">${pct(c.value, node.value)}${ch2 ? ` · <span class="chg ${ch2.cls}">${ch2.text}</span>` : ''}${mine2 ? ` · <span class="mine-share">${moneyFull(mine2)}</span>` : ''}</span>
      ${c.kind === 'item' ? '<span class="go det">פרטים</span>' : '<span class="go" aria-hidden="true">›</span>'}`;
    li.addEventListener('click', () => { learned(); if (c.kind === 'item') openItem(c); else select(c); });
    li.addEventListener('mouseenter', () => hot(li._d));
    li.addEventListener('mouseleave', () => hot(null));
    list.appendChild(li);
  }
  if (kids.length > 300) list.insertAdjacentHTML('beforeend', `<li class="leaf-hint">ועוד ${kids.length - 300} סעיפים. אפשר לחפש אותם בחיפוש.</li>`);
}

function drawCrumbs(node) {
  const nav = $('#crumbs');
  nav.innerHTML = '';
  const p = pathOf(node);
  p.forEach((n, i) => {
    if (i) nav.insertAdjacentHTML('beforeend', '<span class="sep">‹</span>');
    const label = n.kind === 'root' ? `כל ${sideWord()}` : n.name;
    if (i === p.length - 1) nav.insertAdjacentHTML('beforeend', `<span class="here">${esc(label)}</span>`);
    else {
      const b = document.createElement('button');
      b.textContent = label;
      b.addEventListener('click', () => !busy && goTo(n));
      nav.appendChild(b);
    }
  });
}

// ---------- הגרף לאורך השנים ----------
// view 'one': עמודה לכל שנה של הצומת הנוכחי. view 'cmp': קו לכל פלח, באותם צבעים כמו בעוגה.
const chartTip = $('#chart-tip');
function chartSeries(node, view) {
  if (view === 'cmp' && node.children && node.kind !== 'item') {
    return arcsNow.map((a) => ({ node: a.data, name: a.data.name, color: a.color, s: seriesOf(a.data) }));
  }
  return [{ node, name: node.kind === 'root' ? `כל ${sideWord()}` : node.name, color: css('--s1'), s: seriesOf(node) }];
}
function drawChart() {
  const canCmp = current.kind !== 'item' && current.children && current.children.length > 1;
  const view = canCmp ? state.view : 'one';
  document.querySelectorAll('[data-view]').forEach((b) => {
    b.setAttribute('aria-selected', String(b.dataset.view === view));
    b.hidden = b.dataset.view === 'cmp' && !canCmp;
  });
  document.querySelectorAll('[data-unit]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.unit === state.unit)));
  $('#chart-title').textContent = view === 'cmp' ? 'איך זה התחלק לאורך השנים' : 'לאורך השנים';
  plot($('#chart'), chartSeries(current, view), view === 'cmp' ? 'lines' : 'bars', chartTip, $('#chart-note'));
}

const UNIT_NOTE = {
  nis: 'שקלים של כל שנה, בלי תיקון לאינפלציה. כדי להשוות לאורך זמן, "₪ היום" מדויק יותר.',
  real: () => `במחירי ${ctx.years[ctx.years.length - 1]}, לפי מדד המחירים לצרכן.`,
  cap: () => `לתושב, במחירי ${ctx.years[ctx.years.length - 1]}: הסכום מחולק במספר תושבי ישראל באותה שנה.`,
  pct: () => `אחוז מכל ${sideWord()} של אותה שנה${state.side === 'expense' && !state.debt ? ', בלי החזר קרן החוב' : ''}.`,
};
function plot(el, list, mode, tipEl, noteEl) {
  const ys = YEARS;
  const den = denom();
  const u = state.unit;
  const conv = (v, i) => {
    if (v == null) return null;
    if (u === 'pct') return den[i] ? (v / den[i]) * 100 : null;
    if (u === 'real') return realOf(v, ys[i]);
    if (u === 'cap') { const r = realOf(v, ys[i]); const p = popOf(ys[i]); return r != null && p ? r / p : null; }
    return v;
  };
  // הערך של כל שנה בשלב שמוצג בה, והסימון: הביצוע (או החוק, כשמוצג הביצוע)
  const val = (s, i) => conv(valAt(s.s, i), i);
  const mark = (s, i) => {
    const p = phaseOf(ys[i]);
    const q = p === 2 ? 0 : 2;
    return yearMeta(ys[i]).phases.includes(q) ? conv(s.s[q][i], i) : null;
  };
  const n1 = (v) => v.toLocaleString('he-IL', { maximumFractionDigits: 1 });
  const fmtY = (v) => {
    if (!v) return '0';
    if (u === 'pct') return `${n1(v)}%`;
    if (u === 'cap') return `${n1(v)} ₪`;
    return v >= 1e9 ? `${n1(v / 1e9)} מיליארד` : v >= 1e6 ? `${n1(v / 1e6)} מיליון` : `${n1(v / 1e3)} אלף`;
  };
  const fmtV = (v) => (u === 'pct' ? pctNum(v) : u === 'cap' ? `${v >= 10 ? nf.format(Math.round(v)) : v.toLocaleString('he-IL', { maximumFractionDigits: 2 })} ₪` : money(v));

  const w = Math.max(280, el.clientWidth || 480), h = 230;
  const m = { top: 12, right: 12, bottom: 26, left: 76 };
  const svgc = d3.select(el).html('').append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', '100%').attr('role', 'img')
    .attr('aria-label', `גרף לאורך השנים: ${list.map((s) => s.name).join(', ')}`);
  // ציר הזמן משמאל לימין, כמו בעיתונות גם בעברית
  const x = d3.scaleBand().domain(ys).range([m.left, w - m.right]).padding(0.22);
  const all = list.flatMap((s) => ys.map((_, i) => [val(s, i), mode === 'bars' ? mark(s, i) : null])).flat().filter((v) => v != null);
  const maxY = d3.max(all) || 1;
  const y = d3.scaleLinear().domain([0, maxY * 1.08]).nice(4).range([h - m.bottom, m.top]);
  const g = svgc.append('g');
  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', m.left).attr('x2', w - m.right).attr('y1', y).attr('y2', y);
  const every = w < 420 ? 5 : w < 640 ? 3 : 2;
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h - m.bottom})`)
    .call(d3.axisBottom(x).tickSizeOuter(0).tickValues(ys.filter((yr) => (2026 - yr) % every === 0)).tickFormat((d) => (w < 520 ? `'${String(d).slice(2)}` : d)));
  g.append('g').attr('class', 'axis').attr('transform', `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6).tickFormat(fmtY)).call((a) => a.select('.domain').remove());

  const sel = ys.indexOf(state.year);
  if (mode === 'bars') {
    const s = list[0];
    g.selectAll('rect.bar').data(ys.map((yr, i) => ({ yr, i, v: val(s, i) })).filter((d) => d.v != null)).join('rect')
      .attr('class', 'bar').attr('x', (d) => x(d.yr)).attr('width', x.bandwidth())
      .attr('y', (d) => y(Math.max(d.v, 0))).attr('height', (d) => Math.abs(y(0) - y(d.v))).attr('rx', 2)
      .attr('fill', s.color).attr('opacity', (d) => (d.i === sel ? 1 : 0.45));
    g.selectAll('line.fc').data(ys.map((yr, i) => ({ yr, v: mark(s, i) })).filter((d) => d.v != null)).join('line')
      .attr('class', 'fc').attr('x1', (d) => x(d.yr) - 1).attr('x2', (d) => x(d.yr) + x.bandwidth() + 1)
      .attr('y1', (d) => y(d.v)).attr('y2', (d) => y(d.v));
  } else {
    const line = d3.line().defined((d) => d != null).x((_, i) => x(ys[i]) + x.bandwidth() / 2).y((d) => y(d));
    g.selectAll('path.line').data(list).join('path').attr('class', 'line')
      .attr('d', (s) => line(ys.map((_, i) => val(s, i)))).attr('stroke', (s) => s.color);
  }
  if (sel >= 0) {
    g.append('rect').attr('class', 'sel-year').attr('x', x(state.year) - 2).attr('width', x.bandwidth() + 4)
      .attr('y', m.top).attr('height', h - m.bottom - m.top).lower();
  }
  // ריחוף: כל השנה כעמודת פגיעה אחת. לחיצה עוברת לאותה שנה.
  const hover = g.append('line').attr('class', 'cross').attr('y1', m.top).attr('y2', h - m.bottom).attr('opacity', 0);
  g.selectAll('rect.hit').data(ys).join('rect').attr('class', 'hit')
    .attr('x', (d) => x(d) - (x.step() - x.bandwidth()) / 2).attr('width', x.step())
    .attr('y', m.top).attr('height', h - m.bottom - m.top)
    .on('mousemove', (ev, yr) => {
      const i = ys.indexOf(yr);
      hover.attr('x1', x(yr) + x.bandwidth() / 2).attr('x2', x(yr) + x.bandwidth() / 2).attr('opacity', mode === 'lines' ? 1 : 0);
      const p = phaseOf(yr);
      const rows = list.map((s) => ({ s, v: val(s, i), f: mark(s, i) })).filter((r) => r.v != null).sort((a, b) => b.v - a.v);
      tipEl.innerHTML = `<b>${yr}</b>` + (rows.length ? rows.slice(0, 9).map((r) => (list.length > 1
        ? `<div class="row"><i style="background:${r.s.color}"></i>${esc(r.s.name)}<span>${fmtV(r.v)}</span></div>`
        : `<div>${PHASE_SHORT[p]}: ${fmtV(r.v)}</div>${r.f != null ? `<div class="muted">${p === 2 ? 'בחוק' : 'בפועל'}: ${fmtV(r.f)}</div>` : ''}`)).join('')
        : '<div class="muted">אין סעיף כזה בשנה הזו</div>') +
        (yr !== state.year ? '<div class="muted">לחצו כדי לראות את העוגה של השנה הזו</div>' : '');
      placeTip(tipEl, ev, el.getBoundingClientRect());
    })
    .on('mouseleave', () => { tipEl.hidden = true; hover.attr('opacity', 0); })
    .on('click', (ev, yr) => { tipEl.hidden = true; if (yr !== state.year) setYear(yr); });

  const notes = [];
  if (mode === 'bars' && ys.some((_, i) => mark(list[0], i) != null)) {
    notes.push(phaseOf() === 2
      ? '<span class="key-fc"></span> הקו הכהה: התקציב כפי שאושר בחוק באותה שנה.'
      : '<span class="key-fc"></span> הקו הכהה: כמה הוצא בפועל באותה שנה.');
  }
  const un = UNIT_NOTE[u];
  notes.push(typeof un === 'function' ? un() : un);
  const missing = ys.filter((_, i) => list.every((s) => valAt(s.s, i) == null));
  if (missing.length && missing.length < ys.length) {
    const span = missing.length > 6 ? `${missing.length} שנים` : missing.join(', ');
    notes.push(`אין נתון ל-${span}: הסעיף לא הופיע בתקציב באותן שנים, או הופיע בקוד אחר.`);
  }
  noteEl.innerHTML = notes.map((n) => `<p>${n}</p>`).join('');
}

document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => { state.view = b.dataset.view; drawChart(); syncURL(); }));
document.querySelectorAll('[data-unit]').forEach((b) => b.addEventListener('click', () => {
  state.unit = b.dataset.unit; drawChart(); if (!$('#sheet').hidden && state.item) openItemByCode(state.item); syncURL();
}));
let rsz;
window.addEventListener('resize', () => { clearTimeout(rsz); rsz = setTimeout(() => current && drawChart(), 150); });

// ---------- מפתח התקציב: מי קיבל את הכסף, ושינויים במהלך השנה ----------
const obCache = {};
async function obq(sql) {
  if (obCache[sql]) return obCache[sql];
  const r = await fetch(`${OB}/api/query?query=${encodeURIComponent(sql)}`);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.status);
  return (obCache[sql] = j.rows || []);
}
const KIND_HE = { municipality: 'רשות מקומית', association: 'עמותה', 'ottoman-association': 'אגודה עותמנית', company: 'חברה',
  cooperative: 'אגודה שיתופית', 'gov-company': 'חברה ממשלתית', 'public-company': 'חברה ציבורית', 'health-service': 'קופת חולים',
  'religious-council': 'מועצה דתית', 'municipal-company': 'חברה עירונית', 'municipal-council': 'מועצה אזורית' };
function entityLink(id, name, kind) {
  const label = esc(name || 'לא ידוע');
  const k = kind && KIND_HE[kind] ? ` <span class="muted">${KIND_HE[kind]}</span>` : '';
  return id ? `<a href="${OB}/i/org/${encodeURIComponent(kind || 'company')}/${encodeURIComponent(id)}" target="_blank" rel="noopener">${label}</a>${k}` : `${label}${k}`;
}
async function whoHTML(code, year) {
  const like = `00${code}%`;
  const [sup, con] = await Promise.all([
    obq(`select coalesce(entity_name, recipient) as name, entity_id, entity_kind, sum(amount_paid) as paid, count(*) as n
         from supports_by_payment_year where budget_code like '${like}' and year_paid=${year}
         group by 1,2,3 order by paid desc nulls last limit 12`).catch(() => null),
    obq(`select coalesce(entity_name, supplier_name->>0) as name, entity_id, entity_kind, sum(volume) as vol, count(*) as n,
         max(purpose) as purpose from contract_spending where budget_code like '${like}'
         and order_date >= '${year}-01-01' and order_date < '${year + 1}-01-01'
         group by 1,2,3 order by vol desc nulls last limit 12`).catch(() => null),
  ]);
  const block = (title, rows, amt, empty, note) => {
    if (rows == null) return `<h4>${title}</h4><p class="muted">מפתח התקציב לא ענה כרגע. נסו שוב בעוד רגע.</p>`;
    const good = rows.filter((r) => r[amt] > 0);
    if (!good.length) return `<h4>${title}</h4><p class="muted">${empty}</p>`;
    const tot = d3.sum(good, (r) => r[amt]);
    return `<h4>${title}</h4><ol class="who-list">${good.map((r) => `<li><span class="who-name">${entityLink(r.entity_id, r.name, r.entity_kind)}${r.purpose ? `<span class="who-purpose">${esc(r.purpose)}</span>` : ''}</span><span class="who-amt">${money(r[amt])}</span></li>`).join('')}</ol>` +
      `<p class="muted small">${note} ${good.length >= 12 ? 'הגדולים מבין המדווחים' : `סה"כ ${money(tot)}`}.</p>`;
  };
  return block('תמיכות', sup, 'paid', `לא דווחו תמיכות מהסעיף הזה ב-${year}.`, `תשלומים ב-${year} לפי דיווח המשרד.`) +
    block('התקשרויות ורכש', con, 'vol', `לא דווחו התקשרויות שנחתמו ב-${year} בסעיף הזה.`, `היקף ההתקשרויות שנחתמו ב-${year}.`);
}
async function changesHTML(code, year) {
  // הבקשות לשינוי נרשמות ברמת תכנית (6 ספרות), ולכן תקנה מחפשת לפי התכנית שלה
  const pre = `00${code.slice(0, 6)}`;
  let rows;
  try {
    rows = await obq(`select req_title->>0 as t, change_title->>0 as k, date->>0 as d, change_list, committee_id
      from budget_changes where year=${year} and budget_code_title::text like '%"${pre}%' order by d limit 60`);
  } catch { return '<p class="muted">מפתח התקציב לא ענה כרגע.</p>'; }
  const items = [];
  for (const r of rows) {
    const mine = (r.change_list || []).filter((c) => (c.budget_code_title || '').startsWith(pre));
    const diff = d3.sum(mine, (c) => c.net_expense_diff || 0);
    if (!diff) continue;
    const why = mine.map((c) => c.ai_change_explanation).find(Boolean);
    items.push({ ...r, diff, why });
  }
  if (!items.length) return `<p class="muted">לא נמצאו העברות תקציביות בתכנית הזו ב-${year}.</p>`;
  const tot = d3.sum(items, (r) => r.diff);
  return `<ol class="who-list">${items.map((r) => `<li><span class="who-name">${esc(r.k || r.t || 'שינוי')}
      <span class="who-purpose">${esc(r.d || '')}${r.t && r.t !== r.k ? ` · ${esc(r.t)}` : ''}${r.why ? `<br>${esc(r.why)}` : ''}</span></span>
      <span class="who-amt chg ${r.diff > 0 ? 'up' : 'down'}">${r.diff > 0 ? '+' : '−'}${money(Math.abs(r.diff))}</span></li>`).join('')}</ol>
    <p class="muted small">סה"כ ${tot > 0 ? 'תוספת' : 'הפחתה'} של ${money(Math.abs(tot))} בתכנית כולה במהלך השנה. ההסברים נכתבו בבינה מלאכותית במפתח התקציב, מתוך מסמכי הבקשה.</p>`;
}
function drawWho(node) {
  const box = $('#who');
  const code = codeOf(node);
  const show = state.side === 'expense' && code && node.kind !== 'more';
  box.hidden = !show;
  if (!show) return;
  box.open = false;
  $('#who-sum').textContent = `מי קיבל את הכסף של ${node.name} ב-${state.year}?`;
  $('#who-body').innerHTML = '';
  box._load = async () => {
    const body = $('#who-body');
    body.innerHTML = '<p class="muted">טוען ממפתח התקציב…</p>';
    const html = await whoHTML(code, state.year);
    if (current === node) body.innerHTML = html;
  };
}
$('#who').addEventListener('toggle', () => { const b = $('#who'); if (b.open && b._load) { const f = b._load; b._load = null; f(); } });

// ---------- כרטיס תקנה ----------
function openItem(d) {
  state.item = d.code;
  const path = pathOf(d).slice(1, -1).filter((n) => n.kind !== 'more').map((n) => n.name).join(' ‹ ');
  const ch = change(d.value, prevOf(d));
  const mine = myShare(d.value);
  const m = yearMeta();
  const leaf = d.leaf;
  const rows = [];
  for (const p of [0, 1, 2]) if (m.phases.includes(p)) rows.push([`${PHASE_NAME[p]}`, moneyFull(Math.max(d.v3[p], 0))]);
  if (m.phases.includes(0) && m.phases.includes(2) && d.v3[0] > 0) rows.push(['בפועל מול החוק', pctNum(d.v3[2] / d.v3[0] * 100)]);
  const ph = phaseOf();
  if (leaf.j && leaf.j[ph]) rows.push(['משרות (שיא כוח אדם)', leaf.j[ph].toLocaleString('he-IL')]);
  if (leaf.h && leaf.h[ph]) rows.push(['הרשאה להתחייב (לשנים הבאות)', moneyFull(leaf.h[ph] * 1000)]);
  if (leaf.m && leaf.m[ph]) rows.push(['הוצאה מותנית בהכנסה', moneyFull(leaf.m[ph] * 1000)]);
  if (mine) rows.push(['מהמסים שלכם', moneyFull(mine)]);
  const live = state.side === 'expense';
  $('#sheet-body').innerHTML = `
    <h3 id="sheet-title">${esc(d.name)}</h3>
    <p class="path">${esc(path)} · ${state.year}</p>
    <div class="big">${moneyFull(d.value)}</div>
    <div>${PHASE_SHORT[ph]} · ${pct(d.value, grand)} מכל ${sideWord()}${ch ? ` · <span class="chg ${ch.cls}">${ch.text}</span> לעומת ${state.year - 1}` : ''}</div>
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    <h4 class="sheet-chart-title">לאורך השנים</h4>
    <div class="chart-box"><div id="sheet-chart" class="chart"></div><div id="sheet-chart-tip" class="tooltip" hidden></div></div>
    <div id="sheet-chart-note" class="chart-note"></div>
    ${live ? `<h4 class="sheet-chart-title">מי קיבל את הכסף</h4><div id="sheet-who" class="who-body"><p class="muted">טוען ממפתח התקציב…</p></div>
    <h4 class="sheet-chart-title">שינויים במהלך ${state.year}</h4><div id="sheet-changes" class="who-body"><p class="muted">טוען…</p></div>` : ''}
    <p class="foot">${leaf.t ? `סוג: ${esc(leaf.t)} · ` : ''}תקנה ${d.code}${live ? ` · <a href="${OB}/i/budget/00${d.code}/${state.year}" target="_blank" rel="noopener">התקנה במפתח התקציב</a>` : ''}</p>`;
  $('#sheet').hidden = false;
  plot($('#sheet-chart'), [{ node: d, name: d.name, color: css('--s1'), s: seriesOf(d) }], 'bars', $('#sheet-chart-tip'), $('#sheet-chart-note'));
  $('#sheet-close').focus();
  if (live) {
    const code = d.code, year = state.year;
    whoHTML(code, year).then((h) => { if (state.item === code) $('#sheet-who').innerHTML = h; });
    changesHTML(code, year).then((h) => { if (state.item === code) $('#sheet-changes').innerHTML = h; });
  }
  syncURL();
}
function findItem(root, code) {
  let hit = null;
  (function walk(n) { if (hit) return; if (n.code === code) hit = n; (n.children || []).forEach(walk); })(root);
  return hit;
}
function openItemByCode(code) { const n = findItem(trees[state.side], code); if (n) openItem(n); }
function closeSheet() { $('#sheet').hidden = true; state.item = null; syncURL(); }
$('#sheet-close').addEventListener('click', closeSheet);
$('#sheet').addEventListener('click', (e) => { if (e.target.id === 'sheet') closeSheet(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { if (!$('#sheet').hidden) closeSheet(); else goUp(); }
});

// ---------- חיפוש ----------
function allNodes(root) {
  const out = [];
  (function walk(n) { if (n.kind !== 'root') out.push(n); (n.children || []).forEach(walk); })(root);
  return out;
}
const results = $('#search-results');
$('#search').addEventListener('input', (e) => {
  const q = e.target.value.trim();
  if (q.length < 2) { results.hidden = true; return; }
  const digits = /^\d+$/.test(q);
  const hits = [];
  for (const s of ['expense', 'income']) {
    for (const n of allNodes(trees[s])) {
      const code = n.code || (n.key && n.key.slice(1));
      if (digits ? code && code.startsWith(q) : n.name.includes(q)) hits.push({ n, s });
    }
  }
  hits.sort((a, b) => b.n.value - a.n.value);
  results.innerHTML = hits.slice(0, 30).map((h, i) => {
    const path = pathOf(h.n).slice(1, -1).map((x) => x.name).join(' ‹ ');
    return `<li data-i="${i}">${esc(h.n.name)} · ${money(h.n.value)}<span class="path">${h.s === 'income' ? 'הכנסה · ' : ''}${esc(path)}</span></li>`;
  }).join('') || `<li>לא נמצא בתקציב ${state.year}</li>`;
  results.hidden = false;
  results.querySelectorAll('li[data-i]').forEach((li) => li.addEventListener('click', async () => {
    const h = hits[+li.dataset.i];
    results.hidden = true; e.target.value = '';
    if (h.s !== state.side) setSide(h.s, false);
    const target = h.n.kind === 'item' ? h.n.parent : h.n;
    await ensureSeries(target);
    render(target);
    if (h.n.kind === 'item') openItem(h.n);
  }));
});
document.addEventListener('click', (e) => { if (!e.target.closest('.search')) results.hidden = true; });

// ---------- הכתובת: כל המצב בתוכה ----------
// #y=2026&s=e&p=D2.S22&i=20460503&v=cmp&u=real&f=e&d=0
// p: המפתחות הקבועים של כל רמה (D2, S22, P20...), m = פלח "עוד". המפתח הוא הקוד, אז הוא שורד מעבר בין שנים.
function pathIds(node) { return pathOf(node).slice(1).map((n) => (n.kind === 'more' ? 'm' : n.key)); }
function nodeByIds(root, ids) {
  let n = root;
  for (const id of ids) {
    const c = id === 'm' ? slicesOf(n).find((x) => x.kind === 'more')
      : (n.children || []).find((x) => x.key === id || (x.alias && x.alias.key === id));
    if (!c || c.kind === 'item') break;
    n = c;
  }
  return n;
}
let lastHash = '';
function syncURL() {
  const q = new URLSearchParams();
  q.set('y', state.year);
  if (state.side !== 'expense') q.set('s', 'i');
  const ids = current ? pathIds(current) : [];
  if (ids.length) q.set('p', ids.join('.'));
  if (state.item) q.set('i', state.item);
  if (state.view !== 'one') q.set('v', state.view);
  if (state.unit !== 'nis') q.set('u', state.unit);
  if (state.phase) q.set('f', 'ore'[state.phase]);
  if (!state.debt) q.set('d', '0');
  const h = `#${q.toString()}`;
  if (h !== location.hash) history.replaceState(null, '', h);
  lastHash = h;
}
function readURL() {
  const q = new URLSearchParams(location.hash.slice(1));
  return {
    year: +q.get('y') || null,
    side: q.get('s') === 'i' ? 'income' : 'expense',
    ids: (q.get('p') || '').split('.').filter(Boolean),
    item: q.get('i'),
    view: q.get('v') === 'cmp' ? 'cmp' : 'one',
    unit: ['real', 'cap', 'pct'].includes(q.get('u')) ? q.get('u') : 'nis',
    phase: Math.max(0, 'ore'.indexOf(q.get('f') || 'o')),
    debt: q.get('d') !== '0',
  };
}
window.addEventListener('hashchange', () => { if (location.hash !== lastHash) applyURL(); });

// ---------- שיתוף ----------
const shortCache = {};
function isgd(url) {
  return new Promise((resolve, reject) => {
    const cb = `isgd_${Date.now()}`;
    const s = document.createElement('script');
    const done = () => { delete window[cb]; s.remove(); };
    window[cb] = (r) => { done(); r && r.shorturl ? resolve(r.shorturl) : reject(new Error('is.gd')); };
    s.onerror = () => { done(); reject(new Error('is.gd')); };
    s.src = `https://is.gd/create.php?format=json&callback=${cb}&url=${encodeURIComponent(url)}`;
    document.head.appendChild(s);
  });
}
async function dagd(url) {
  const r = await fetch(`https://da.gd/s?url=${encodeURIComponent(url)}`);
  const t = (await r.text()).trim();
  if (!r.ok || !t.startsWith('https://')) throw new Error('da.gd');
  return t;
}
const timeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
async function shorten(url) {
  if (shortCache[url]) return shortCache[url];
  if (/^https?:\/\/(localhost|127\.)/.test(url)) return url;
  for (const f of [dagd, isgd]) {
    try { return (shortCache[url] = await timeout(f(url), 4000)); } catch { /* הבא בתור */ }
  }
  return url;
}
function toast(html) {
  const t = $('#toast');
  t.innerHTML = html; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, 6000);
}
function shareTitle() {
  const where = current.kind === 'root' ? `כל ${sideWord()}` : current.kind === 'more' ? `הסעיפים הקטנים של ${current.parent.name}` : current.name;
  return `${where} · תקציב המדינה ${state.year}`;
}
$('#share').addEventListener('click', async () => {
  const btn = $('#share');
  btn.disabled = true; btn.textContent = 'מכין קישור…';
  syncURL();
  const url = await shorten(location.href);
  btn.disabled = false; btn.textContent = 'שיתוף';
  const title = shareTitle();
  if (navigator.share && matchMedia('(pointer: coarse)').matches) {
    try { await navigator.share({ title, text: title, url }); return; } catch (e) { if (e.name === 'AbortError') return; }
  }
  try {
    await navigator.clipboard.writeText(url);
    toast(`הקישור הועתק: <a href="${esc(url)}" target="_blank" rel="noopener" dir="ltr">${esc(url)}</a>`);
  } catch {
    toast(`הקישור: <a href="${esc(url)}" target="_blank" rel="noopener" dir="ltr">${esc(url)}</a>`);
  }
});

// ---------- שנים, שלבים וצדדים ----------
async function loadYear(y) {
  if (!books[y]) books[y] = await (await fetch(`data/budget_${y}.json`)).json();
  return books[y];
}
function buildTrees(d) { trees = { expense: buildTree(d, 'expense'), income: buildTree(d, 'income') }; }
function drawYears() {
  const bar = $('#years');
  bar.innerHTML = years.map((m) => `<button data-year="${m.year}" aria-pressed="${m.year === state.year}">${m.year}</button>`).join('');
  bar.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => setYear(+b.dataset.year)));
  const on = bar.querySelector('[aria-pressed="true"]');
  if (on) bar.scrollLeft = on.offsetLeft - bar.clientWidth / 2 + on.offsetWidth / 2;
}
function drawPhase() {
  const m = yearMeta();
  const ph = phaseOf();
  document.querySelectorAll('[data-phase]').forEach((b) => {
    const p = +b.dataset.phase;
    b.setAttribute('aria-selected', String(p === ph));
    b.disabled = !m.phases.includes(p);
  });
  $('#debt').checked = state.debt;
}
function drawHeader() {
  $('#year').textContent = state.year;
  const tot = trees.expense.value;
  $('#grand-total').textContent = money(tot);
  const pop = popOf(state.year);
  $('#per-person').textContent = pop ? `כ-${nf.format(Math.round(tot / pop / 100) * 100)} ₪ לכל תושב (${PHASE_SHORT[phaseOf()]})` : '';
  const notes = [];
  const m = yearMeta();
  if (!m.phases.includes(0)) notes.push(`ב-${state.year} לא עבר חוק תקציב: המדינה פעלה בתקציב המשכי (חלק מהתקציב הקודם בכל חודש). לכן אין "כפי שאושר בחוק", ומוצג התקציב אחרי השינויים.`);
  if (!m.phases.includes(2)) notes.push(`ל-${state.year} עוד אין נתוני ביצוע, אז מוצג התקציב כפי שאושר בחוק.`);
  if (state.side === 'expense' && !state.debt) notes.push('בלי החזר קרן החוב: רק הוצאה שנשארת בפועל, כולל הריבית.');
  $('#year-note').textContent = notes.join(' ');
  $('#year-note').hidden = !notes.length;
}

async function setYear(y, ids = null) {
  const keep = ids || (current ? pathIds(current) : []);
  const item = state.item;
  state.year = y;
  const [book] = await Promise.all([loadYear(y), ensureSeries(null)]);
  buildTrees(book);
  grand = trees[state.side].value;
  drawYears(); drawPhase(); drawHeader();
  if (!$('#sheet').hidden) $('#sheet').hidden = true;
  state.item = null;
  let target = nodeByIds(trees[state.side], keep);
  await ensureSeries(target);
  // nodeByIds עוצר כשהוא לא מוצא, ואז אולי רק עכשיו נטענה הסדרה; מספיק
  render(target);
  if (item) { const n = findItem(trees[state.side], item); if (n) { await ensureSeries(n); openItem(n); } }
  drawMoves();
}
function rebuild() {
  // שינוי של שלב או של קרן החוב: אותה שנה, אותו מקום
  const keep = current ? pathIds(current) : [];
  buildTrees(books[state.year]);
  grand = trees[state.side].value;
  drawPhase(); drawHeader();
  render(nodeByIds(trees[state.side], keep));
  drawMoves();
}
function setSide(s, draw = true) {
  state.side = s;
  document.querySelectorAll('.seg button[data-side]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.side === s)));
  grand = trees[s].value;
  drawHeader();
  if (draw) render(trees[s]);
}
document.querySelectorAll('.seg button[data-side]').forEach((b) => b.addEventListener('click', () => setSide(b.dataset.side)));
document.querySelectorAll('[data-phase]').forEach((b) => b.addEventListener('click', () => { state.phase = +b.dataset.phase; rebuild(); }));
$('#debt').addEventListener('change', (e) => { state.debt = e.target.checked; rebuild(); });

// ---------- המסים שלי ----------
$('#tax').addEventListener('input', () => {
  store.set('tax', $('#tax').value);
  drawSide(current, arcsNow);
});
// הערכה גסה: מס הכנסה לפי מדרגות 2025–2026 (מוקפאות), ומע"מ על החלק מהשכר נטו שהולך לצריכה.
// בלי ביטוח לאומי ומס בריאות: הם לא נכנסים לתקציב המדינה.
const BRACKETS = [[7010, 0.10], [10060, 0.14], [16150, 0.20], [22440, 0.31], [46690, 0.35], [Infinity, 0.47]];
const POINT = 242, SURTAX_FROM = 60130, VAT = 0.18, SPEND = 0.6;
function estimate(gross, points) {
  let tax = 0, lo = 0;
  for (const [hi, r] of BRACKETS) { if (gross > lo) tax += (Math.min(gross, hi) - lo) * r; lo = hi; }
  if (gross > SURTAX_FROM) tax += (gross - SURTAX_FROM) * 0.03;
  tax = Math.max(0, tax - points * POINT);
  const net = gross - tax - gross * 0.12;            // בערך: ביטוח לאומי, בריאות ופנסיה
  const vat = Math.max(0, net) * SPEND * VAT / (1 + VAT);
  return { income: tax * 12, vat: vat * 12 };
}
$('#est-open').addEventListener('click', () => { $('#est').hidden = !$('#est').hidden; });
function drawEstimate() {
  const g = +$('#est-salary').value, p = +$('#est-points').value || 0;
  const btn = $('#est-use');
  if (!(g > 0)) { $('#est-out').textContent = ''; btn.disabled = true; return; }
  const e = estimate(g, p);
  $('#est-out').textContent = `מס הכנסה: כ-${moneyFull(Math.round(e.income / 100) * 100)} בשנה. מע"מ על הקניות: כ-${moneyFull(Math.round(e.vat / 100) * 100)} ` +
    `(בהנחה ש-60% מהנטו הולך לקניות שחייבות במע"מ). יחד: כ-${moneyFull(Math.round((e.income + e.vat) / 100) * 100)}. הערכה גסה בלבד: בלי הפקדות, זיכויים נוספים ומסים עקיפים אחרים.`;
  btn.disabled = false;
  btn._v = Math.round((e.income + e.vat) / 100) * 100;
}
$('#est-salary').addEventListener('input', drawEstimate);
$('#est-points').addEventListener('input', drawEstimate);
$('#est-use').addEventListener('click', () => {
  $('#tax').value = $('#est-use')._v; store.set('tax', $('#tax').value);
  $('#est').hidden = true; drawSide(current, arcsNow);
});

// ---------- מה בולט ----------
// מחושב מהסדרות של המשרדים (רמה P): פער בין החוק לביצוע, ושינוי ריאלי בעשור
function drawMoves() {
  const offices = [];
  (function walk(n) {
    if (n.kind === 'office' || (n.alias && n.alias.kind === 'office')) { offices.push(n); return; }
    (n.children || []).forEach(walk);
  })(trees.expense);
  const show = (el, rows, fmt) => {
    $(el).innerHTML = rows.map((r) => `<li data-k="${r.n.alias ? r.n.alias.key : r.n.key}"><span class="name">${esc(r.n.name)}</span><span class="mv-val">${fmt(r)}</span></li>`).join('') || '<li class="muted">אין נתונים</li>';
    $(el).querySelectorAll('li[data-k]').forEach((li) => li.addEventListener('click', async () => {
      if (state.side !== 'expense') setSide('expense', false);
      const n = offices.find((o) => (o.alias ? o.alias.key : o.key) === li.dataset.k);
      if (!n) return;
      await ensureSeries(n);
      render(n);
      document.querySelector('.stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
  };
  // הסדרה של הצומת עצמו, אם יש (קרן החוב וקרן החוב לביטוח הלאומי חולקות את אותו סעיף 84)
  const sk = (n) => (SER[n.sk] ? n.sk : n.alias.sk);
  // פער: השנה האחרונה שיש לה ביצוע, עד השנה שנבחרה
  const ey = [...years].reverse().find((m) => m.year <= state.year && m.phases.includes(2) && m.phases.includes(0));
  const i = ey ? yi(ey.year) : -1;
  const gaps = i < 0 ? [] : offices.map((n) => {
    const s = rawSeries(sk(n)); if (!s) return null;
    const o = s[0][i], e = s[2][i];
    return o > 5e8 || e > 5e8 ? { n, o: o || 0, e: e || 0, d: (e || 0) - (o || 0) } : null;
  }).filter(Boolean).sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, 6);
  $('#mv-gap-h').textContent = ey ? `איפה ${ey.year} בפועל רחוקה מהחוק` : 'איפה הביצוע רחוק מהחוק';
  show('#mv-gap', gaps, (r) => `<span class="chg ${r.d > 0 ? 'up' : 'down'}">${r.d > 0 ? '+' : '−'}${money(Math.abs(r.d))}</span><span class="muted"> ${r.o ? pctNum(r.e / r.o * 100) : ''}</span>`);
  // עשור: שינוי ריאלי, בשלב שמוצג בכל שנה
  const y0 = state.year - 10, j = yi(y0), k = yi(state.year);
  const moves = j < 0 ? [] : offices.map((n) => {
    const s = rawSeries(sk(n)); if (!s) return null;
    const a = valAt(s, j), b = valAt(s, k);
    if (!a || !b) return null;
    const ra = realOf(a, y0), rb = realOf(b, state.year);
    return Math.max(ra, rb) > 1e9 ? { n, ra, rb, ch: (rb / ra - 1) * 100 } : null;
  }).filter(Boolean);
  const fmtCh = (r) => `<span class="chg ${r.ch > 0 ? 'up' : 'down'}">${r.ch > 0 ? '▲' : '▼'} ${pctNum(Math.abs(r.ch))}</span><span class="muted"> ${money(r.ra)} ← ${money(r.rb)}</span>`;
  $('#mv-grow-h').textContent = `מה גדל הכי הרבה מאז ${y0}`;
  $('#mv-shrink-h').textContent = `מה הצטמק הכי הרבה מאז ${y0}`;
  show('#mv-grow', [...moves].sort((a, b) => b.ch - a.ch).slice(0, 6), fmtCh);
  show('#mv-shrink', [...moves].sort((a, b) => a.ch - b.ch).filter((r) => r.ch < 0).slice(0, 6), fmtCh);
  $('#moves-lede').textContent = `משרדים שתקציבם מעל מיליארד ₪. העשור מחושב במחירי ${ctx.years[ctx.years.length - 1]}, כדי שהאינפלציה לא תיראה כמו גידול. לחצו על שורה כדי לפתוח את המשרד בעוגה.`;
  $('#moves-note').textContent = j < 0 ? '' : 'משרד שהקוד שלו השתנה בעשור (למשל אחרי פיצול או איחוד) לא יופיע, או יופיע כגידול חד. כדאי לבדוק בגרף.';
}

// צבעים נקראים מ-CSS, אז מחליפים כשמצב כהה משתנה
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(current));

async function applyURL() {
  const u = readURL();
  Object.assign(state, { view: u.view, unit: u.unit, side: u.side, phase: u.phase, debt: u.debt });
  document.querySelectorAll('.seg button[data-side]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.side === state.side)));
  const y = years.some((m) => m.year === u.year) ? u.year : years[years.length - 1].year;
  current = null;
  state.item = u.item || null;
  await setYear(y, u.ids);
}

// ---------- התחלה ----------
(async () => {
  const [ys, c] = await Promise.all([fetch('data/years.json').then((r) => r.json()), fetch('data/context.json').then((r) => r.json())]);
  years = ys; ctx = c; YEARS = ys.map((m) => m.year);
  $('#ctx-notes').textContent = c.notes.length ? `${c.notes.join('. ')}.` : '';
  const saved = store.get('tax');
  if (saved) $('#tax').value = saved;
  await applyURL();
})();
