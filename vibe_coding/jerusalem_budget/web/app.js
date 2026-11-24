// לאן הולך הכסף של ירושלים: עוגת התקציב הרגיל של העירייה. לחיצה על פלח הופכת אותו
// לעוגה משלו, ומתחת לכל עוגה גרף של אותו פלח לאורך השנים.
// שתי חלוקות לאותם סעיפים: לפי נושא (פרקי המבנה האחיד) ולפי מבנה העירייה (מינהל ← אגף).
// הנתונים נבנים ב-tools/build_years.py:
//   data/years.json         השנים, המקור של כל אחת והערות
//   data/budget_<שנה>.json  כל הסעיפים של שנה אחת
//   data/series.json        כל צומת לאורך השנים, לפי מפתח קבוע

const MAX_SLICES = 8;          // עד 8 פלחים; מעבר לזה 7 הגדולים ו"עוד"
const DUR = 420;
const CITY = 'ירושלים';

// תיקוני כתיב בשמות כפי שהופיעו בספר
const RENAME = {
  'שרותי תמיכה מערכת לוגיסטית': 'שירותי תמיכה במערכת הלוגיסטית',
  'מידע ואירגון': 'מידע וארגון',
  'הגנה אזרחית': 'הגנה אזרחית (הג"א)',
};
// הסברים קצרים רק היכן שהשם לבדו לא מספיק. המפתח הוא המפתח הקבוע של הצומת.
const EXPLAIN = {
  'e|F|76': 'רוב הסכום הוא הנחות בארנונה שהחוק מחייב (אזרחים ותיקים, נכות, מצב כלכלי, מוסדות פטורים). הן נרשמות פעמיים: כהכנסה מארנונה בצד ההכנסות וכהוצאה כאן, ולכן הן מנפחות את שני צדי התקציב באותו סכום. השאר: השתתפויות העירייה בגופים אחרים.',
  'i|F|11': 'כולל את ההנחות בארנונה, שנרשמות גם כהוצאה (ראו "הנחות בארנונה והשתתפויות" בצד ההוצאות). הארנונה שנגבית בפועל היא הסכום פחות ההנחות. הספר מפצל את הגבייה בין מערב העיר, מזרח העיר ומוסדות.',
  'i|F|19': 'בעיקר "מענק בירה": תוספת שהמדינה נותנת לירושלים כבירה, מעבר למענקי האיזון שמקבלות רשויות אחרות.',
  'e|F|63': 'החזר הלוואות שהעירייה לקחה בעבר (קרן וריבית), ועמלות בנקים.',
  'e|F|99': 'פנסיה תקציבית לעובדי עירייה שפרשו, ופיצויים.',
  'e|M|העמסות - הפחתות': 'כאן נרשמות בספר ההפחתות של עלויות שחולקו בין היחידות. באתר הזה העלויות האלה מוצגות פעם אחת, ביחידה שמספקת את השירות.',
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
  if (!yearMeta() || yearMeta().derived_from) return null;  // ל-2016 אין עמודת שנה קודמת
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
// group: 'f' לפי נושא (פרק במבנה האחיד), 'o' לפי מבנה העירייה (מינהל ← אגף)
const state = { year: null, side: 'expense', item: null, view: 'one', unit: 'nis', group: 'f' };
let years = [], series = { years: [], nodes: {} }, books = {}, trees = {}, current, grand;
const COLORS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const colorOf = (d, i) => (d.kind === 'more' ? css('--other') : css(COLORS[i % COLORS.length]));
const yearMeta = () => years.find((y) => y.year === state.year);
const sideWord = () => (state.side === 'expense' ? 'ההוצאות' : 'ההכנסות');

function pathOf(n) { const p = []; while (n) { p.unshift(n); n = n.parent; } return p; }
function arnona() { const v = +$('#arnona').value; return v > 0 ? v : 0; }
function myShare(v) { const a = arnona(); return a && state.side === 'expense' ? (v / grand) * a : 0; }

// ---------- בניית העץ ----------
// side: 'expense' או 'income'. group: 'f' (נושא ← יחידה ← סעיף) או 'o' (מינהל ← אגף ← יחידה ← סעיף).
// סעיף: c קוד, n שם, s צד, p תקציב השנה, b תקציב השנה הקודמת (מעודכן), m מינהל, a אגף, u יחידה,
//        pg עמוד ו-src קובץ (בשנות ה-PDF), d דברי הסבר, j משרות ביחידה, k "שכר", x תומלל ידנית.
function buildTree(data, side, group) {
  const s = side[0];
  const root = { name: `תקציב ${data.year}`, kind: 'root', sk: s, children: [] };
  const kids = new Map();   // sk → צומת
  const child = (parent, sk, name, kind, extra = {}) => {
    let n = kids.get(sk);
    if (!n) { n = { name: nameOf(name), raw: name, kind, sk, children: [], ...extra }; kids.set(sk, n); parent.children.push(n); }
    return n;
  };
  const chapters = data.chapters[side];
  for (const it of data.items) {
    if (it.s !== s) continue;
    if (it.p <= 0 && it.b <= 0) continue;
    const u4 = it.c.slice(0, 4);
    let parent;
    if (group === 'f') {
      const ch = it.c.slice(0, 2);
      const dom = child(root, `${s}|F|${ch}`, chapters[ch] || `פרק ${ch}`, 'domain');
      parent = child(dom, `${s}|U|${u4}`, it.u || u4, 'dept', { page: it.pg, src: it.src, agaf: it.a });
    } else {
      const m = child(root, `${s}|M|${it.m}`, it.m || 'לא משויך', 'domain');
      const a = child(m, `${s}|A|${it.m}|${it.a}`, it.a || it.m, 'dept', { page: it.pg, src: it.src });
      parent = child(a, `${s}|U|${u4}`, it.u || u4, 'dept', { page: it.pg, src: it.src });
    }
    parent.children.push({
      name: nameOf(it.n) || it.c, kind: 'item', item: it, page: it.pg, src: it.src, sk: `${s}|I|${it.c}`,
      value: Math.max(it.p, 0), prev: Math.max(it.b, 0), jobs: it.j || 0,
    });
  }
  // סכומים מלמטה למעלה, ומיון מהגדול לקטן
  (function sum(n) {
    if (!n.children) return;
    n.children.forEach(sum);
    n.children.sort((a, b) => b.value - a.value);
    n.value = d3.sum(n.children, (c) => c.value);
    n.prev = d3.sum(n.children, (c) => c.prev);
    n.salary = d3.sum(n.children, (c) => c.salary ?? (c.item && c.item.k === 'שכר' ? c.value : 0));
    n.jobs = d3.sum(n.children, (c) => c.jobs || 0);
  })(root);
  // צומת עם ילד אחד (אגף עם יחידה אחת, או יחידה בשם האגף): מדלגים על הרמה הכפולה
  (function flatten(n) {
    if (!n.children) return;
    while (n.kind !== 'root' && n.children.length === 1 && n.children[0].children) {
      const only = n.children[0];
      n.children = only.children;
      n.page = n.page || only.page;
      n.src = n.src || only.src;
    }
    n.children.forEach((c) => { c.parent = n; flatten(c); });
  })(root);
  // אגפים שונים נותנים ליחידות שלהם אותו שם ("חינוך מיוחד", "הנהלת האגף"): מוסיפים את שם האגף, ואם צריך את הקוד
  (function dedupe(n) {
    if (!n.children) return;
    const count = d3.rollup(n.children.filter((c) => c.kind !== 'item'), (v) => v.length, (c) => c.name);
    for (const c of n.children) if (c.kind !== 'item' && count.get(c.name) > 1 && c.agaf && c.agaf !== c.name) c.name = `${c.name} · ${c.agaf}`;
    const again = d3.rollup(n.children.filter((c) => c.kind !== 'item'), (v) => v.length, (c) => c.name);
    for (const c of n.children) if (c.kind !== 'item' && again.get(c.name) > 1) c.name = `${c.name} (${c.sk.split('|').pop()})`;
    n.children.forEach(dedupe);
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
    name: `עוד ${rest.length} סעיפים קטנים`, kind: 'more', children: rest, parent: node,
    value: d3.sum(rest, (c) => c.value), prev: d3.sum(rest, (c) => c.prev),
  };
  return (node._slices = [...top, more]);
}

// ---------- סדרות לאורך השנים ----------
function seriesOf(node) {
  if (node.kind === 'more') {
    const parts = node.children.map(seriesOf);
    const sumAt = (k) => series.years.map((_, i) => {
      const xs = parts.map((p) => p[k][i]).filter((x) => x != null);
      return xs.length ? d3.sum(xs) : null;
    });
    return { v: sumAt('v'), f: sumAt('f') };
  }
  const e = series.nodes[node.sk] || {};
  return { v: e.v || series.years.map(() => null), f: e.f || series.years.map(() => null) };
}
// מכנה לאחוזים: כל התקציב של אותו צד באותה שנה
function denom() {
  return series.nodes[state.side[0]].v.map((t) => t || 0);
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
  const ch = change(d.value, d.prev);
  const mine = myShare(d.value);
  // פלח "עוד" הוא כמה סעיפים יחד, ולכן אומרים במפורש מה בתוכו
  const inside = d.kind === 'more' ? `<span class="muted">ביחד: ${esc(d.children.map((c) => c.name).join(', '))}</span><br>` : '';
  tip.innerHTML = `<b>${esc(d.name)}</b>${inside}${money(d.value)}<br>
    <span class="muted">${pct(d.value, current.value)} מ${esc(current.kind === 'root' ? 'כל התקציב' : current.name)}` +
    (current.kind !== 'root' ? ` · ${pct(d.value, grand)} מכל התקציב` : '') + '</span>' +
    (ch ? `<br><span class="chg ${ch.cls}">${ch.text}</span> <span class="muted">לעומת ${state.year - 1}</span>` : '') +
    (mine ? `<br><span class="mine-share">מהארנונה שלכם: ${moneyFull(mine)}</span>` : '');
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
  drawCenter(node); drawSide(node, arcs); drawCrumbs(node); drawChart(); syncURL();
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
    // רקע בצבע ההורה שנעלם בזמן שהילדים נפרשים
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
function select(d) {
  if (busy) return;
  learned();
  if (d.kind === 'item') { openItem(d); return; }
  busy = true; setTimeout(() => { busy = false; }, DUR * 2 + 50);
  zoomIn(d);
}
function goUp() { if (current.parent && !busy) goTo(current.parent); }
function goTo(n) {
  if (n === current) return;
  busy = true; setTimeout(() => { busy = false; }, DUR * 2 + 50);
  if (pathOf(current).includes(n)) zoomOut(n);
  else { render(n); busy = false; }
}

// ---------- הצד: כותרת ורשימה ----------
// קישור למקור: בשנות ה-PDF לעמוד המדויק בקובץ שממנו נקרא הסעיף, ובשנות האקסל לקובץ האקסל
function sourceFor(page, src) {
  const m = yearMeta();
  if (!m) return '';
  const url = src || (state.side === 'income' && m.source_income) || m.source;
  if (!url) return '';
  const xlsx = url.includes('.xlsx');
  const label = xlsx ? (m.derived_from ? `בקובץ האקסל של ${m.derived_from}` : 'בקובץ האקסל של העירייה') : 'בספר התקציב';
  return `<a href="${encodeURI(url)}${page && !xlsx ? `#page=${page}` : ''}" target="_blank" rel="noopener">${label}${page && !xlsx ? `, עמ' ${page}` : ''}</a>`;
}

function drawSide(node, arcs) {
  const info = $('#node-info');
  const ch = change(node.value, node.prev);
  const isRoot = node.kind === 'root';
  const title = isRoot ? (state.side === 'expense' ? 'לאן הולך הכסף' : 'מאיפה בא הכסף')
    : node.kind === 'more' ? `הסעיפים הקטנים של ${node.parent.name}` : node.name;
  const explain = EXPLAIN[node.sk] || '';
  const facts = [];
  facts.push(`<span><b>${money(node.value)}</b> ב-${state.year}</span>`);
  if (!isRoot) facts.push(`<span><b>${pct(node.value, grand)}</b> מכל ${sideWord()}</span>`);
  if (ch) facts.push(`<span>לעומת ${state.year - 1}: <b class="chg ${ch.cls}">${ch.text}</b></span>`);
  if (state.side === 'expense' && node.salary > 0) facts.push(`<span>שכר: <b>${pct(node.salary, node.value)}</b></span>`);
  if (state.side === 'expense' && node.jobs > 0) facts.push(`<span>משרות: <b>${node.jobs.toLocaleString('he-IL', { maximumFractionDigits: 1 })}</b></span>`);
  const mine = myShare(node.value);
  if (mine && !isRoot) facts.push(`<span class="mine-share">מהארנונה שלכם: <b>${moneyFull(mine)}</b></span>`);
  const src = node.kind !== 'more' && (node.page || node.src) ? ` · ${sourceFor(node.page, node.src)}` : '';
  const rootWord = state.group === 'f' ? 'נושאים' : 'מינהלים';
  info.innerHTML = `<h2>${esc(title)}</h2>
    <p class="sub">${node.kind === 'more'
    ? `${node.children.length} סעיפים קטנים של ${esc(node.parent.name)}, שמוצגים כאן יחד כי כל אחד מהם קטן מכדי לקבל פלח משלו. אלה סעיפים נפרדים: לחצו על סעיף כדי לראות רק אותו.`
    : `${node.children.length} ${isRoot ? rootWord : 'סעיפים'}`}${src}</p>
    ${explain ? `<p class="sub">${esc(explain)}</p>` : ''}
    <div class="facts">${facts.join('')}</div>`;

  const colorFor = new Map(arcs.map((a) => [a.data, a.color]));
  const list = $('#list');
  list.innerHTML = '';
  const max = d3.max(node.children, (c) => c.value) || 1;
  for (const c of node.children) {
    const li = document.createElement('li');
    // סעיף שנכנס לפלח "עוד" לא מדליק את הפלח האפור, כדי שלא ייראה שהפלח כולו הוא הסעיף
    li._d = colorFor.has(c) ? c : null;
    const col = colorFor.get(c) || css('--other');
    const ch2 = change(c.value, c.prev);
    const mine2 = myShare(c.value);
    const salaryTag = c.item && c.item.k === 'שכר' ? '<span class="tag">שכר</span>' : '';
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
    return arcsNow.map((a) => ({ node: a.data, name: a.data.name, color: a.color, ...seriesOf(a.data) }));
  }
  return [{ node, name: node.kind === 'root' ? `כל ${sideWord()}` : node.name, color: css('--s1'), ...seriesOf(node) }];
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

function plot(el, list, mode, tipEl, noteEl) {
  const ys = series.years;
  const den = denom();
  const asPct = state.unit === 'pct';
  const val = (s, i, k = 'v') => {
    const v = s[k][i];
    if (v == null) return null;
    return asPct ? (den[i] ? (v / den[i]) * 100 : null) : v;
  };
  const num = (x) => x.toLocaleString('he-IL', { maximumFractionDigits: 1 });
  const fmtY = (v) => (asPct ? `${d3.format('.3~')(v)}%` : v >= 1e9 ? `${num(v / 1e9)} מיליארד` : v >= 1e6 ? `${num(v / 1e6)} מ׳` : `${num(v / 1e3)} א׳`);
  const fmtV = (v) => (asPct ? pctNum(v) : money(v));

  const w = Math.max(280, el.clientWidth || 480), h = 230;
  const m = { top: 12, right: 12, bottom: 26, left: 46 };
  const svgc = d3.select(el).html('').append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', '100%').attr('role', 'img')
    .attr('aria-label', `גרף לאורך השנים: ${list.map((s) => s.name).join(', ')}`);
  // ציר הזמן משמאל לימין, כמו בעיתונות גם בעברית
  const x = d3.scaleBand().domain(ys).range([m.left, w - m.right]).padding(0.25);
  const all = list.flatMap((s) => ys.map((_, i) => [val(s, i), mode === 'bars' ? val(s, i, 'f') : null])).flat().filter((v) => v != null);
  const maxY = d3.max(all) || 1;
  const y = d3.scaleLinear().domain([0, maxY * 1.08]).nice(4).range([h - m.bottom, m.top]);
  const g = svgc.append('g');
  g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', m.left).attr('x2', w - m.right).attr('y1', y).attr('y2', y);
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h - m.bottom})`)
    .call(d3.axisBottom(x).tickSizeOuter(0).tickFormat((d) => (w < 420 ? `'${String(d).slice(2)}` : d)));
  g.append('g').attr('class', 'axis').attr('transform', `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6).tickFormat(fmtY)).call((a) => a.select('.domain').remove());

  const sel = ys.indexOf(state.year);
  if (mode === 'bars') {
    const s = list[0];
    g.selectAll('rect.bar').data(ys.map((yr, i) => ({ yr, i, v: val(s, i) })).filter((d) => d.v != null)).join('rect')
      .attr('class', 'bar').attr('x', (d) => x(d.yr)).attr('width', x.bandwidth())
      .attr('y', (d) => y(d.v)).attr('height', (d) => y(0) - y(d.v)).attr('rx', 3)
      .attr('fill', s.color).attr('opacity', (d) => (d.i === sel ? 1 : 0.45));
    // הביצוע הצפוי: קו קצר על כל עמודה
    g.selectAll('line.fc').data(ys.map((yr, i) => ({ yr, v: val(s, i, 'f') })).filter((d) => d.v != null)).join('line')
      .attr('class', 'fc').attr('x1', (d) => x(d.yr) - 2).attr('x2', (d) => x(d.yr) + x.bandwidth() + 2)
      .attr('y1', (d) => y(d.v)).attr('y2', (d) => y(d.v));
  } else {
    const line = d3.line().defined((d) => d != null).x((_, i) => x(ys[i]) + x.bandwidth() / 2).y((d) => y(d));
    g.selectAll('path.line').data(list).join('path').attr('class', 'line')
      .attr('d', (s) => line(ys.map((_, i) => val(s, i)))).attr('stroke', (s) => s.color);
    g.selectAll('g.dots').data(list).join('g').attr('class', 'dots').attr('fill', (s) => s.color)
      .selectAll('circle').data((s) => ys.map((yr, i) => ({ yr, v: val(s, i) })).filter((d) => d.v != null)).join('circle')
      .attr('cx', (d) => x(d.yr) + x.bandwidth() / 2).attr('cy', (d) => y(d.v)).attr('r', 2.5);
  }
  // השנה המוצגת בעוגה
  if (sel >= 0) {
    g.append('rect').attr('class', 'sel-year').attr('x', x(state.year) - 3).attr('width', x.bandwidth() + 6)
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
      const rows = list.map((s) => ({ s, v: val(s, i), f: val(s, i, 'f') })).filter((r) => r.v != null)
        .sort((a, b) => b.v - a.v);
      const meta = years.find((m2) => m2.year === yr);
      tipEl.innerHTML = `<b>${yr}</b>` + (rows.length ? rows.map((r) => (list.length > 1
        ? `<div class="row"><i style="background:${r.s.color}"></i>${esc(r.s.name)}<span>${fmtV(r.v)}</span></div>`
        : `<div>תקציב: ${fmtV(r.v)}</div>${r.f != null ? `<div class="muted">אחרי עדכון: ${fmtV(r.f)}</div>` : ''}`)).join('')
        : '<div class="muted">אין סעיף כזה בשנה הזו</div>') +
        (meta && meta.derived_from ? `<div class="muted">מהספר של ${meta.derived_from}</div>` : '') +
        (yr !== state.year ? '<div class="muted">לחצו כדי לראות את העוגה של השנה הזו</div>' : '');
      placeTip(tipEl, ev, el.getBoundingClientRect());
    })
    .on('mouseleave', () => { tipEl.hidden = true; hover.attr('opacity', 0); })
    .on('click', (ev, yr) => { tipEl.hidden = true; if (yr !== state.year) setYear(yr); });

  const notes = [];
  if (mode === 'bars' && list[0].f.some((v) => v != null)) {
    notes.push('<span class="key-fc"></span> הקו הכהה: התקציב המעודכן של אותה שנה, אחרי השינויים שנעשו במהלכה, כפי שנרשם בספר של השנה שאחריה. זה עדיין לא ביצוע בפועל.');
  }
  if (asPct) notes.push(`אחוז מכל ${sideWord()} של אותה שנה.`);
  else notes.push('שקלים של כל שנה, בלי תיקון לאינפלציה. כדי להשוות סדרי עדיפויות, האחוזים מדויקים יותר.');
  const missing = ys.filter((_, i) => list.every((s) => s.v[i] == null));
  if (missing.length && missing.length < ys.length) notes.push(`אין נתון ל-${missing.join(', ')}: הסעיף לא הופיע בספר של אותה שנה, או הופיע בקוד אחר${state.group === 'o' ? ' או במינהל אחר (העירייה שינתה את המבנה שלה כמה פעמים)' : ''}.`);
  noteEl.innerHTML = notes.map((n) => `<p>${n}</p>`).join('');
}

document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => { state.view = b.dataset.view; drawChart(); syncURL(); }));
document.querySelectorAll('[data-unit]').forEach((b) => b.addEventListener('click', () => {
  state.unit = b.dataset.unit; drawChart(); if (!$('#sheet').hidden && state.item) openItemByCode(state.item); syncURL();
}));
let rsz;
window.addEventListener('resize', () => { clearTimeout(rsz); rsz = setTimeout(() => current && drawChart(), 150); });

// ---------- כרטיס סעיף ----------
function openItem(d) {
  state.item = d.item.c;
  const it = d.item;
  const path = pathOf(d).slice(1, -1).filter((n) => n.kind !== 'more').map((n) => n.name).join(' ‹ ');
  const ch = change(d.value, d.prev);
  const mine = myShare(d.value);
  const derived = yearMeta().derived_from;
  const rows = [[`תקציב ${state.year}`, moneyFull(d.value)]];
  if (!derived) rows.push([`תקציב ${state.year - 1} (מעודכן)`, moneyFull(d.prev)]);
  if (it.j) rows.push(['משרות ביחידה', it.j.toLocaleString('he-IL')]);
  if (mine) rows.push(['מהארנונה שלכם', moneyFull(mine)]);
  const where = [it.m, it.a, it.u].filter((x, i, a) => x && a.indexOf(x) === i).join(' ‹ ');
  $('#sheet-body').innerHTML = `
    <h3 id="sheet-title">${esc(d.name)}</h3>
    <p class="path">${esc(path)} · ${state.year}</p>
    <div class="big">${moneyFull(d.value)}</div>
    <div>${pct(d.value, grand)} מכל ${sideWord()}${ch ? ` · <span class="chg ${ch.cls}">${ch.text}</span> לעומת ${state.year - 1}` : ''}</div>
    ${it.d ? `<div class="expl"><small>דברי ההסבר בספר התקציב</small>${esc(it.d)}</div>` : ''}
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    <h4 class="sheet-chart-title">לאורך השנים</h4>
    <div class="chart-box"><div id="sheet-chart" class="chart"></div><div id="sheet-chart-tip" class="tooltip" hidden></div></div>
    <div id="sheet-chart-note" class="chart-note"></div>
    <p class="foot">${where ? `${esc(where)} · ` : ''}${it.k ? `${it.k} · ` : ''}סעיף ${it.c.slice(0, 4)}-${it.c.slice(4, 7)}-${it.c.slice(7)} · ${sourceFor(d.page, d.src)}
      ${it.x ? '<br>העמוד הזה בספר צויר כקווים ולא כטקסט, והסעיף תומלל ידנית. <a href="data.html#manual">פרטים</a>' : ''}</p>`;
  $('#sheet').hidden = false;
  plot($('#sheet-chart'), [{ node: d, name: d.name, color: css('--s1'), ...seriesOf(d) }], 'bars', $('#sheet-chart-tip'), $('#sheet-chart-note'));
  $('#sheet-close').focus();
  syncURL();
}
function findItem(root, code) {
  let hit = null;
  (function walk(n) { if (hit) return; if (n.item && n.item.c === code) hit = n; (n.children || []).forEach(walk); })(root);
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
  const hits = [];
  for (const s of ['expense', 'income']) {
    for (const n of allNodes(trees[s])) {
      if (n.name.includes(q) || (n.item && n.item.d && n.item.d.includes(q))) hits.push({ n, s });
    }
  }
  hits.sort((a, b) => b.n.value - a.n.value);
  results.innerHTML = hits.slice(0, 30).map((h, i) => {
    const path = pathOf(h.n).slice(1, -1).map((x) => x.name).join(' ‹ ');
    return `<li data-i="${i}">${esc(h.n.name)} · ${money(h.n.value)}<span class="path">${h.s === 'income' ? 'הכנסה · ' : ''}${esc(path)}</span></li>`;
  }).join('') || `<li>לא נמצא בתקציב ${state.year}</li>`;
  results.hidden = false;
  results.querySelectorAll('li[data-i]').forEach((li) => li.addEventListener('click', () => {
    const h = hits[+li.dataset.i];
    results.hidden = true; e.target.value = '';
    if (h.s !== state.side) setSide(h.s, false);
    const target = h.n.kind === 'item' ? h.n.parent : h.n;
    render(target);
    if (h.n.kind === 'item') openItem(h.n);
  }));
});
document.addEventListener('click', (e) => { if (!e.target.closest('.search')) results.hidden = true; });

// ---------- הכתובת: כל המצב בתוכה ----------
// #y=2026&s=e&p=k3f.a91&i=1812000110&v=cmp&u=pct
// p: מזהה קצר לכל רמה בעוגה (גיבוב של המפתח הקבוע, אז הוא שורד מעבר בין שנים), m = פלח "עוד"
function shortId(sk) {
  let h = 5381;
  for (const ch of sk) h = ((h * 33) ^ ch.codePointAt(0)) >>> 0;
  return h.toString(36).slice(0, 4);
}
function pathIds(node) {
  return pathOf(node).slice(1).map((n) => (n.kind === 'more' ? 'm' : shortId(n.sk)));
}
function nodeByIds(root, ids) {
  let n = root;
  for (const id of ids) {
    const c = id === 'm' ? slicesOf(n).find((x) => x.kind === 'more') : (n.children || []).find((x) => x.sk && shortId(x.sk) === id);
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
  if (state.group !== 'f') q.set('g', state.group);
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
    unit: q.get('u') === 'pct' ? 'pct' : 'nis',
    group: q.get('g') === 'o' ? 'o' : 'f',
  };
}
window.addEventListener('hashchange', () => { if (location.hash !== lastHash) applyURL(); });

// ---------- שיתוף ----------
const shortCache = {};
function isgd(url) {
  // ל-is.gd אין CORS, אבל יש JSONP. גיבוי בלבד: ב-10/2026 הוא החזיר "database insert failed"
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
  return `${where} · תקציב עיריית ${CITY} ${state.year}`;
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

// ---------- שנים וצדדים ----------
async function loadYear(y) {
  if (!books[y]) books[y] = await (await fetch(`data/budget_${y}.json`)).json();
  return books[y];
}
function buildTrees(d) {
  trees = { expense: buildTree(d, 'expense', state.group), income: buildTree(d, 'income', state.group) };
}
function drawYears() {
  const bar = $('#years');
  bar.innerHTML = years.map((m) => `<button data-year="${m.year}" aria-pressed="${m.year === state.year}">${m.year}</button>`).join('');
  bar.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => setYear(+b.dataset.year)));
  const on = bar.querySelector('[aria-pressed="true"]');
  // השנה הנבחרת באמצע הסרגל (scrollIntoView היה גולל גם את הדף)
  if (on) bar.scrollLeft = on.offsetLeft - bar.clientWidth / 2 + on.offsetWidth / 2;
}
function drawHeader() {
  $('#year').textContent = state.year;
  const m = yearMeta();
  // הסכום הרשמי של הספר. בעוגה הוא יכול להיות שונה מעט (העמסות שלא מתקזזות, ראו data.html)
  $('#grand-total').textContent = `${money(m.total_expense)}`;
  $('#year-note').innerHTML = m.note ? `${esc(m.note)} <a href="data.html#y${m.year}">עוד על הנתונים של ${m.year}</a>` : '';
  $('#year-note').hidden = !m.note;
  $('#source-link').href = encodeURI(m.source);
}

async function setYear(y, ids = null) {
  const keep = ids || (current ? pathIds(current) : []);
  const item = state.item;
  state.year = y;
  buildTrees(await loadYear(y));
  grand = trees[state.side].value;
  drawYears(); drawHeader();
  if (!$('#sheet').hidden) $('#sheet').hidden = true;
  state.item = null;
  render(nodeByIds(trees[state.side], keep));
  if (item) { const n = findItem(trees[state.side], item); if (n) openItem(n); }
}

function setSide(s, draw = true) {
  state.side = s;
  document.querySelectorAll('.seg button[data-side]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.side === s)));
  grand = trees[s].value;
  if (draw) render(trees[s]);
}
document.querySelectorAll('.seg button[data-side]').forEach((b) => b.addEventListener('click', () => setSide(b.dataset.side)));

// החלפת חלוקה: בונים את העצים מחדש וחוזרים לשורש (המסלול בחלוקה האחרת לא קיים)
function setGroup(g) {
  if (g === state.group) return;
  state.group = g;
  document.querySelectorAll('.seg button[data-group]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.group === g)));
  buildTrees(books[state.year]);
  grand = trees[state.side].value;
  render(trees[state.side]);
}
document.querySelectorAll('.seg button[data-group]').forEach((b) => b.addEventListener('click', () => setGroup(b.dataset.group)));

$('#arnona').addEventListener('input', () => {
  store.set('arnona', $('#arnona').value);
  drawSide(current, arcsNow);
});

// צבעים נקראים מ-CSS, אז מחליפים כשמצב כהה משתנה
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(current));

async function applyURL() {
  const u = readURL();
  state.view = u.view || 'one';
  state.unit = u.unit || 'nis';
  state.side = u.side;
  state.group = u.group;
  document.querySelectorAll('.seg button[data-side]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.side === state.side)));
  document.querySelectorAll('.seg button[data-group]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.group === state.group)));
  const y = years.some((m) => m.year === u.year) ? u.year : years[years.length - 1].year;
  current = null;
  state.item = u.item || null;
  await setYear(y, u.ids);
}

// ---------- התחלה ----------
(async () => {
  const [ys, sr] = await Promise.all([fetch('data/years.json').then((r) => r.json()), fetch('data/series.json').then((r) => r.json())]);
  years = ys; series = sr;
  const saved = store.get('arnona');
  if (saved) $('#arnona').value = saved;
  await applyURL();
})();
