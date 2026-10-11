// לאן הולך הכסף: עוגת התקציב של המועצה. לחיצה על פלח הופכת אותו לעוגה משלו.
// הנתונים: data/budget_<שנה>.json, נבנים מספר התקציב ב-tools/parse_budget.py.

const YEAR = 2026;
const SOURCE = 'https://www.pardes-hanna-karkur.muni.il/wp-content/uploads/2026/01/1_%D7%AA%D7%A7%D7%A6%D7%99%D7%91-2026-ea99798703c0aba343738025777216ff.pdf';
const MAX_SLICES = 8;          // עד 8 פלחים; מעבר לזה 7 הגדולים ו"עוד"
const DUR = 420;

// תיקוני כתיב בשמות כפי שהופיעו בספר
const RENAME = {
  'ספריות ערוניות': 'ספריות עירוניות',
  'פיקוח וטירנרי': 'פיקוח וטרינרי',
  'שרותים חקלאים': 'שירותים חקלאיים',
  'מינהל שרותים חקלאיים': 'שירותים חקלאיים',
  'שרותים דתיים יהודיים': 'שירותים דתיים',
  'שרות משפטי': 'שירות משפטי',
  'שרות פסיכולוגי חינוכי': 'שירות פסיכולוגי חינוכי (שפ"ח)',
  'תרבות,אומנות וספורט': 'תרבות, אמנות וספורט',
  'תרבות, אומנות וספורט': 'תרבות, אמנות וספורט',
  'קליטת עליה': 'קליטת עלייה',
  'בלתי רגילים': 'פנסיה, הנחות בארנונה ובלתי צפוי',
  'תקבולים כלליים': 'ארנונה ומענקים כלליים',
  "הוצ' מיוחדות ובלתי נצפות מראש": 'הוצאות בלתי צפויות',
  'הנחות ממיסים לפי חוק או תקנות': 'הנחות בארנונה לפי חוק',
  'מינהל שפ"ע': 'מינהל שפ"ע (שיפור פני העיר)',
};
// הסברים קצרים רק היכן שהשם לבדו לא מספיק
const EXPLAIN = {
  'הנחות ממיסים לפי חוק או תקנות': 'הנחות בארנונה שהחוק מחייב (אזרחים ותיקים, נכות, מצב כלכלי ועוד). בספר הן נרשמות פעמיים: כהכנסה מארנונה וכהוצאה, ולכן הן מופיעות בשני צדי התקציב.',
  'תקבולים כלליים': 'הכסף שאינו צבוע לשירות מסוים: בעיקר ארנונה, ומענקים כלליים ממשרד הפנים.',
  'פרעון מלוות': 'החזר הלוואות שהמועצה לקחה בעבר.',
};

const $ = (s) => document.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* מצב פרטי */ } },
};

// ---------- עיצוב מספרים ----------
const nf = new Intl.NumberFormat('he-IL');
function money(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toLocaleString('he-IL', { maximumFractionDigits: a >= 1e8 ? 0 : 1 })} מיליון ₪`;
  if (a >= 1e3) return `${Math.round(v / 1e3).toLocaleString('he-IL')} אלף ₪`;
  return `${nf.format(Math.round(v))} ₪`;
}
const moneyFull = (v) => `${nf.format(Math.round(v))} ₪`;
function pct(part, whole) {
  const p = whole ? (part / whole) * 100 : 0;
  if (p === 0) return '0%';
  if (p < 0.1) return 'פחות מ-0.1%';
  return `${p.toLocaleString('he-IL', { maximumFractionDigits: p < 10 ? 1 : 0 })}%`;
}
function change(cur, prev) {
  if (!prev && !cur) return null;
  if (!prev) return { text: 'סעיף חדש', cls: 'up' };
  if (!cur) return { text: 'בוטל ב-2026', cls: 'down' };
  const d = ((cur - prev) / prev) * 100;
  if (Math.abs(d) < 0.5) return { text: 'ללא שינוי', cls: '' };
  const s = `${Math.abs(d).toLocaleString('he-IL', { maximumFractionDigits: Math.abs(d) < 10 ? 1 : 0 })}%`;
  return d > 0 ? { text: `▲ ${s}`, cls: 'up' } : { text: `▼ ${s}`, cls: 'down' };
}
const nameOf = (n) => RENAME[n] || n;

// ---------- בניית העץ ----------
// side: 'expense' או 'income'. בספר הוצאה חיובית (ח) והכנסה שלילית (ז).
function buildTree(data, side) {
  const sign = side === 'expense' ? 1 : -1;
  const root = { name: `תקציב ${data.year}`, kind: 'root', children: [] };
  for (const d of data.domains) {
    const dom = { name: nameOf(d.name), raw: d.name, kind: 'domain', children: [] };
    for (const dp of d.depts) {
      const dept = { name: nameOf(dp.name), raw: dp.name, kind: 'dept', page: dp.page, children: [] };
      for (const it of dp.items) {
        if (it.section !== side) continue;
        const v = sign * it.p, prev = sign * it.b_prev;
        if (v <= 0 && prev <= 0) continue;
        dept.children.push({
          name: nameOf(it.name) || it.code, kind: 'item', item: it, page: dp.page,
          value: Math.max(v, 0), prev: Math.max(prev, 0), forecast: sign * it.f_prev,
        });
      }
      if (dept.children.length) dom.children.push(dept);
    }
    if (dom.children.length) root.children.push(dom);
  }
  // סכומים מלמטה למעלה, ומיון מהגדול לקטן
  (function sum(n) {
    if (!n.children) return;
    n.children.forEach(sum);
    n.children.sort((a, b) => b.value - a.value);
    n.value = d3.sum(n.children, (c) => c.value);
    n.prev = d3.sum(n.children, (c) => c.prev);
    n.salary = d3.sum(n.children, (c) => c.salary ?? (c.item && c.item.kind === 'שכר' ? c.value : 0));
    n.jobs = d3.sum(n.children, (c) => c.jobs ?? (c.item && c.item.kind === 'שכר' ? c.item.jobs || 0 : 0));
  })(root);
  // תחום עם מחלקה אחת (או מחלקה עם שם זהה): מדלגים על הרמה הכפולה
  (function flatten(n) {
    if (!n.children) return;
    if (n.kind !== 'root' && n.children.length === 1 && n.children[0].children) {
      const only = n.children[0];
      n.children = only.children;
      n.page = n.page || only.page;
    }
    n.children.forEach((c) => { c.parent = n; flatten(c); });
  })(root);
  root.children.forEach(function key(c, i) { c.idx = i; (c.children || []).forEach(key); });
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
    name: `עוד ${rest.length} סעיפים`, kind: 'more', children: rest, parent: node,
    value: d3.sum(rest, (c) => c.value), prev: d3.sum(rest, (c) => c.prev),
  };
  return (node._slices = [...top, more]);
}

// ---------- מצב ----------
let data, side = 'expense', trees = {}, current, grand;
const COLORS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const colorOf = (d, i) => (d.kind === 'more' ? css('--other') : css(COLORS[i % COLORS.length]));

function pathOf(n) { const p = []; while (n) { p.unshift(n); n = n.parent; } return p; }
function arnona() { const v = +$('#arnona').value; return v > 0 ? v : 0; }
function myShare(v) { const a = arnona(); return a ? (v / grand) * a : 0; }

// ---------- העוגה ----------
const W = 460, R = 210, R0 = 120;
const svg = d3.select('#pie').attr('viewBox', `${-W / 2} ${-W / 2} ${W} ${W}`);
const gSlices = svg.append('g');
const gLabels = svg.append('g');
const gCenter = svg.append('g').attr('class', 'center');
const arc = d3.arc().innerRadius(R0).outerRadius(R).cornerRadius(4).padAngle(0.008);
const pieLayout = d3.pie().value((d) => d.value).sort(null);
const tip = $('#tooltip');

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
  const name = node.kind === 'root' ? (side === 'expense' ? 'כל ההוצאות' : 'כל ההכנסות') : node.name;
  const lines = wrap(name, 16);
  lines.forEach((l, i) => gCenter.append('text').attr('class', 'center-name').attr('text-anchor', 'middle')
    .attr('y', -22 - (lines.length - 1 - i) * 18).text(l));
  gCenter.append('text').attr('class', 'center-total').attr('text-anchor', 'middle').attr('y', 12).text(money(node.value));
  if (up) gCenter.append('text').attr('class', 'center-back').attr('text-anchor', 'middle').attr('y', 40).text('↩ חזרה');
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
  sel.on('click', (ev, a) => select(a.data))
    .on('mousemove', (ev, a) => showTip(ev, a))
    .on('mouseleave', () => { tip.hidden = true; hot(null); });
}

function showTip(ev, a) {
  const d = a.data;
  const ch = change(d.value, d.prev);
  const mine = myShare(d.value);
  tip.innerHTML = `<b>${esc(d.name)}</b>${money(d.value)}<br>
    <span class="muted">${pct(d.value, current.value)} מ${esc(current.kind === 'root' ? 'כל התקציב' : current.name)}` +
    (current.kind !== 'root' ? ` · ${pct(d.value, grand)} מכל התקציב` : '') + '</span>' +
    (ch ? `<br><span class="chg ${ch.cls}">${ch.text}</span> <span class="muted">לעומת 2025</span>` : '') +
    (mine ? `<br><span class="mine-share">מהארנונה שלכם: ${moneyFull(mine)}</span>` : '');
  tip.hidden = false;
  const box = $('.pie-box').getBoundingClientRect();
  let x = ev.clientX - box.left + 14, y = ev.clientY - box.top + 14;
  if (x + tip.offsetWidth > box.width) x = ev.clientX - box.left - tip.offsetWidth - 14;
  tip.style.left = `${Math.max(0, x)}px`; tip.style.top = `${y}px`;
  hot(d);
}

function hot(d) {
  gSlices.selectAll('path.slice').classed('dim', (a) => d && a.data !== d);
  document.querySelectorAll('#list li').forEach((li) => li.classList.toggle('hot', !!d && li._d === d));
}

// ציור רגיל, בלי אנימציה
function render(node) {
  current = node;
  const arcs = layout(node);
  const p = gSlices.selectAll('path.slice').data(arcs, (a) => a.key);
  p.exit().remove();
  const all = p.enter().append('path').attr('class', 'slice').merge(p)
    .attr('fill', (a) => a.color).attr('d', arc).each(function (a) { this._cur = a; });
  bindSlices(all);
  drawLabels(arcs);
  drawCenter(node);
  drawSide(node, arcs);
  drawCrumbs(node);
  writeHash(node);
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
    drawCenter(target); drawSide(target, arcs); drawCrumbs(target); writeHash(target);
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
      drawCenter(to); drawSide(to, arcs); drawCrumbs(to); writeHash(to);
      setTimeout(() => drawLabels(arcs), DUR);
    });
}

let busy = false;
function select(d) {
  if (busy) return;
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
function drawSide(node, arcs) {
  const info = $('#node-info');
  const ch = change(node.value, node.prev);
  const isRoot = node.kind === 'root';
  const title = isRoot ? (side === 'expense' ? 'לאן הולך הכסף' : 'מאיפה בא הכסף') : node.name;
  const explain = EXPLAIN[node.raw] || '';
  const facts = [];
  facts.push(`<span><b>${money(node.value)}</b></span>`);
  if (!isRoot) facts.push(`<span><b>${pct(node.value, grand)}</b> מכל ${side === 'expense' ? 'ההוצאות' : 'ההכנסות'}</span>`);
  if (ch) facts.push(`<span>לעומת 2025: <b class="chg ${ch.cls}">${ch.text}</b></span>`);
  if (side === 'expense' && node.salary > 0) facts.push(`<span>שכר: <b>${pct(node.salary, node.value)}</b></span>`);
  if (side === 'expense' && node.jobs > 0) facts.push(`<span>משרות: <b>${node.jobs.toLocaleString('he-IL', { maximumFractionDigits: 1 })}</b></span>`);
  const mine = myShare(node.value);
  if (mine && !isRoot) facts.push(`<span class="mine-share">מהארנונה שלכם: <b>${moneyFull(mine)}</b></span>`);
  const page = node.page ? ` · <a href="${SOURCE}#page=${node.page}" target="_blank" rel="noopener">בספר התקציב, עמ' ${node.page}</a>` : '';
  info.innerHTML = `<h2>${esc(title)}</h2>
    <p class="sub">${node.kind === 'more' ? 'הסעיפים הקטנים יותר' : `${node.children.length} ${isRoot ? 'תחומים' : 'סעיפים'}`}${page}</p>
    ${explain ? `<p class="sub">${esc(explain)}</p>` : ''}
    <div class="facts">${facts.join('')}</div>`;

  const colorFor = new Map(arcs.map((a) => [a.data, a.color]));
  const moreArc = arcs.find((a) => a.data.kind === 'more');
  const list = $('#list');
  list.innerHTML = '';
  const max = d3.max(node.children, (c) => c.value) || 1;
  for (const c of node.children) {
    const li = document.createElement('li');
    li._d = colorFor.has(c) ? c : (moreArc ? moreArc.data : c);
    const col = colorFor.get(c) || css('--other');
    const ch2 = change(c.value, c.prev);
    const mine2 = myShare(c.value);
    const salaryTag = c.item && c.item.kind === 'שכר' ? '<span class="tag">שכר</span>' : '';
    li.innerHTML = `<span class="sw" style="background:${col}"></span>
      <span class="name" title="${esc(c.name)}">${esc(c.name)}${salaryTag}</span>
      <span class="amt">${money(c.value)}</span>
      <span class="bar"><i style="width:${(c.value / max) * 100}%;background:${col}"></i></span>
      <span class="meta">${pct(c.value, node.value)}${ch2 ? ` · <span class="chg ${ch2.cls}">${ch2.text}</span>` : ''}${mine2 ? ` · <span class="mine-share">${moneyFull(mine2)}</span>` : ''}</span>`;
    li.addEventListener('click', () => (c.kind === 'item' ? openItem(c) : select(c)));
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
    const label = n.kind === 'root' ? (side === 'expense' ? 'כל ההוצאות' : 'כל ההכנסות') : n.name;
    if (i === p.length - 1) nav.insertAdjacentHTML('beforeend', `<span class="here">${esc(label)}</span>`);
    else {
      const b = document.createElement('button');
      b.textContent = label;
      b.addEventListener('click', () => !busy && goTo(n));
      nav.appendChild(b);
    }
  });
}

// ---------- כרטיס סעיף ----------
function openItem(d) {
  const it = d.item;
  const path = pathOf(d).slice(1, -1).filter((n) => n.kind !== 'more').map((n) => n.name).join(' ‹ ');
  const ch = change(d.value, d.prev);
  const mine = myShare(d.value);
  const rows = [
    ['תקציב 2026', moneyFull(d.value)],
    ['תקציב 2025', moneyFull(d.prev)],
    ['ביצוע צפוי ל-2025', moneyFull(Math.max(d.forecast, 0))],
  ];
  if (it.jobs) rows.push(['משרות', it.jobs.toLocaleString('he-IL')]);
  if (mine) rows.push(['מהארנונה שלכם', moneyFull(mine)]);
  $('#sheet-body').innerHTML = `
    <h3 id="sheet-title">${esc(d.name)}</h3>
    <p class="path">${esc(path)}</p>
    <div class="big">${moneyFull(d.value)}</div>
    <div>${pct(d.value, grand)} מכל ${side === 'expense' ? 'ההוצאות' : 'ההכנסות'}${ch ? ` · <span class="chg ${ch.cls}">${ch.text}</span> לעומת 2025` : ''}</div>
    ${it.details ? `<div class="expl"><small>ההסבר בספר התקציב</small>${esc(it.details)}</div>` : ''}
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    <p class="foot">${it.kind ? `סוג: ${it.kind} · ` : ''}כרטיס ${it.code} ·
      <a href="${SOURCE}#page=${d.page}" target="_blank" rel="noopener">בספר התקציב, עמ' ${d.page}</a></p>`;
  $('#sheet').hidden = false;
  $('#sheet-close').focus();
}
function closeSheet() { $('#sheet').hidden = true; }
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
      if (n.name.includes(q) || (n.item && n.item.details.includes(q))) hits.push({ n, s });
    }
  }
  hits.sort((a, b) => b.n.value - a.n.value);
  results.innerHTML = hits.slice(0, 30).map((h, i) => {
    const path = pathOf(h.n).slice(1, -1).map((x) => x.name).join(' ‹ ');
    return `<li data-i="${i}">${esc(h.n.name)} · ${money(h.n.value)}<span class="path">${h.s === 'income' ? 'הכנסה · ' : ''}${esc(path)}</span></li>`;
  }).join('') || '<li>לא נמצא</li>';
  results.hidden = false;
  results.querySelectorAll('li[data-i]').forEach((li) => li.addEventListener('click', () => {
    const h = hits[+li.dataset.i];
    results.hidden = true; e.target.value = '';
    if (h.s !== side) setSide(h.s, false);
    const target = h.n.kind === 'item' ? h.n.parent : h.n;
    render(target);
    if (h.n.kind === 'item') openItem(h.n);
  }));
});
document.addEventListener('click', (e) => { if (!e.target.closest('.search')) results.hidden = true; });

// ---------- קישור ישיר (#expense/3/1) ----------
function writeHash(node) {
  const p = pathOf(node).slice(1);
  const stop = p.findIndex((n) => n.kind === 'more');
  const idx = (stop < 0 ? p : p.slice(0, stop)).map((n) => n.idx);
  const h = `#${side}${idx.length ? '/' + idx.join('/') : ''}`;
  if (location.hash !== h) history.replaceState(null, '', h);
}
function readHash() {
  const [s, ...idx] = decodeURIComponent(location.hash.slice(1)).split('/');
  const sd = s === 'income' ? 'income' : 'expense';
  let n = trees[sd];
  for (const i of idx) { const c = n.children && n.children[+i]; if (!c || c.kind === 'item') break; n = c; }
  return { sd, n };
}

function setSide(s, draw = true) {
  side = s;
  document.querySelectorAll('.seg button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.side === s)));
  grand = trees[s].value;
  if (draw) render(trees[s]);
}
document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => setSide(b.dataset.side)));

$('#arnona').addEventListener('input', () => {
  store.set('arnona', $('#arnona').value);
  drawSide(current, layout(current));
});

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// צבעים נקראים מ-CSS, אז מחליפים כשמצב כהה משתנה
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(current));

// ---------- התחלה ----------
fetch(`data/budget_${YEAR}.json`).then((r) => r.json()).then((d) => {
  data = d;
  trees.expense = buildTree(d, 'expense');
  trees.income = buildTree(d, 'income');
  $('#year').textContent = d.year;
  $('#grand-total').textContent = `${money(trees.expense.value)} בשנה`;
  $('#source-link').href = SOURCE;
  const saved = store.get('arnona');
  if (saved) $('#arnona').value = saved;
  // הקישור הישיר כבר מכוון, אז לא פותחים מחדש את מה שהחיפוש הציג
  const { sd, n } = readHash();
  setSide(sd, false);
  render(n);
});
