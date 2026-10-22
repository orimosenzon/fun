// "ישראל מול העולם": הוצאות הממשלה הכללית לפי ייעוד (COFOG) כאחוז מהתוצר, מול מדינות ה-OECD.
// הנתונים: data/oecd.json (tools/fetch_context.py). צורת התצוגה היא "הדגשה": ישראל בצבע ההדגשה,
// השאר באפור. בלי ציון של "טוב" ו"רע": הוצאה גבוהה על ביטחון או על ריבית היא נסיבות, לא בחירה בלבד.
(() => {
  const FUNCS = [
    ['ALL', 'הכל', 'כל הוצאות הממשלה'],
    ['GF02', 'ביטחון', 'ביטחון'],
    ['GF09', 'חינוך', 'חינוך'],
    ['GF07', 'בריאות', 'בריאות'],
    ['GF10', 'רווחה', 'רווחה וביטחון סוציאלי (קצבאות, פנסיה, אבטלה)'],
    ['GF01', 'שירותים כלליים', 'שירותים ציבוריים כלליים, כולל ריבית על החוב'],
    ['GF04', 'כלכלה ותחבורה', 'ענייני כלכלה: תחבורה, אנרגיה, חקלאות ותעשייה'],
    ['GF03', 'משטרה ובתי משפט', 'סדר ציבורי ובטיחות: משטרה, כבאות, בתי משפט ובתי סוהר'],
    ['GF08', 'תרבות ודת', 'תרבות, ספורט ודת'],
  ];
  const HOME = 'ISR';
  let data = null, func = 'GF02';
  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v, d = 1) => (v == null ? '–' : `${Number(v).toLocaleString('he-IL', { minimumFractionDigits: d, maximumFractionDigits: d })}%`);
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  const valOf = (c, y) => {
    const r = data.data[c] && data.data[c][y];
    if (!r) return null;
    return func === 'ALL' ? Object.values(r).reduce((a, b) => a + b, 0) : r[func];
  };
  const nameOf = (c) => data.countries[c] || c;
  const yearsOf = () => [...new Set(Object.values(data.data).flatMap((d) => Object.keys(d)))].map(Number).sort();
  // השנה האחרונה שיש בה ישראל ולפחות 20 מדינות
  function latest() {
    const ys = yearsOf().filter((y) => valOf(HOME, y) != null && Object.keys(data.data).filter((c) => valOf(c, y) != null).length >= 20);
    return ys[ys.length - 1];
  }
  const cohort = (y) => Object.keys(data.data).map((c) => ({ c, v: valOf(c, y) })).filter((r) => r.v != null);

  // ---------------------------------------------------------------- tooltip
  const tip = () => $('#w-tip');
  function showTip(ev, lines) {
    const t = tip();
    t.replaceChildren(...lines.map(([v, l, key]) => {
      const row = el('div', 'v-tip-row');
      if (key) { const k = el('span', 'v-key'); k.style.background = key; row.append(k); }
      row.append(el('strong', null, v), el('span', 'v-tip-l', l));
      return row;
    }));
    t.hidden = false;
    const box = t.parentElement.getBoundingClientRect();
    const x = ev.clientX - box.left, yy = ev.clientY - box.top;
    const w = t.offsetWidth;
    t.style.left = `${Math.max(4, Math.min(box.width - w - 4, x - w / 2))}px`;
    t.style.top = `${Math.max(4, yy - t.offsetHeight - 14)}px`;
  }
  const hideTip = () => { tip().hidden = true; };

  function drawTiles(y) {
    const box = $('#w-kpis');
    box.replaceChildren();
    const rows = cohort(y);
    const home = rows.find((r) => r.c === HOME);
    const others = rows.filter((r) => r.c !== HOME).map((r) => r.v);
    if (!home) return;
    const med = d3.median(others);
    const rank = 1 + others.filter((v) => v > home.v).length;
    const y0 = y - 10, past = valOf(HOME, y0);
    const add = (label, value, ...subs) => {
      const t = el('div', 'v-tile');
      t.append(el('div', 'v-label', label), el('div', 'v-value', value), ...subs.map((s) => el('div', 'v-sub', s)));
      box.append(t);
    };
    add(`ישראל, ${y}`, fmt(home.v), 'מהתוצר', `מקום ${rank} מתוך ${rows.length}, מהגבוה לנמוך`);
    add('חציון OECD', fmt(med), `${others.length} מדינות עם נתון ל-${y}`,
      home.v > med ? `ישראל מעל החציון ב-${fmt(home.v - med)} מהתוצר` : `ישראל מתחת לחציון ב-${fmt(med - home.v)} מהתוצר`);
    if (past != null) add(`ישראל, ${y0}`, fmt(past), 'מהתוצר', `שינוי בעשור: ${home.v >= past ? '+' : '−'}${fmt(Math.abs(home.v - past))} מהתוצר`);
  }

  // ---------------------------------------------------------------- מדינות, מהגבוה לנמוך
  function drawStrip(y) {
    const host = $('#w-strip');
    host.replaceChildren();
    const rows = cohort(y).sort((a, b) => b.v - a.v);
    const F = FUNCS.find((f) => f[0] === func);
    $('#w-strip-title').textContent = `${F[1]}: אחוז מהתוצר, ${y}`;
    const W = host.clientWidth || 500, bh = 15, m = { t: 6, r: 44, b: 6, l: 92 };
    const H = m.t + m.b + rows.length * bh;
    const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.v)]).nice().range([m.l, W - m.r]);
    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
      .attr('role', 'img').attr('aria-label', `${F[2]} כאחוז מהתוצר לפי מדינה`);
    const accent = css('--s1'), other = css('--other'), ink = css('--text'), ink3 = css('--text-3');
    const g = svg.append('g').selectAll('g').data(rows).join('g').attr('transform', (r, i) => `translate(0,${m.t + i * bh})`);
    g.append('rect').attr('x', m.l).attr('y', 2).attr('height', bh - 4).attr('rx', 2)
      .attr('width', (r) => Math.max(1, x(r.v) - m.l)).attr('fill', (r) => (r.c === HOME ? accent : other));
    g.append('text').attr('x', m.l - 6).attr('y', bh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11.5)
      .attr('font-family', 'inherit').attr('fill', (r) => (r.c === HOME ? ink : ink3))
      .attr('font-weight', (r) => (r.c === HOME ? 700 : 400)).text((r) => nameOf(r.c));
    g.append('text').attr('x', (r) => x(r.v) + 4).attr('y', bh / 2 + 4).attr('font-size', 11).attr('font-family', 'inherit')
      .attr('fill', (r) => (r.c === HOME ? ink : ink3)).text((r) => fmt(r.v));
    const med = d3.median(rows.filter((r) => r.c !== HOME), (r) => r.v);
    svg.append('line').attr('x1', x(med)).attr('x2', x(med)).attr('y1', m.t).attr('y2', H - m.b)
      .attr('stroke', ink3).attr('stroke-dasharray', '3 3').attr('stroke-opacity', 0.7);
    g.append('rect').attr('x', 0).attr('width', W).attr('height', bh).attr('fill', 'transparent')
      .on('pointermove', (ev, r) => showTip(ev, [[nameOf(r.c), '', r.c === HOME ? accent : other], [fmt(r.v), 'מהתוצר'], [fmt(med), 'חציון OECD']]))
      .on('pointerleave', hideTip);
  }

  // ---------------------------------------------------------------- ישראל מול החציון לאורך השנים
  function drawTrend() {
    const host = $('#w-trend');
    host.replaceChildren();
    const F = FUNCS.find((f) => f[0] === func);
    $('#w-trend-title').textContent = `${F[1]} לאורך השנים`;
    const ys = yearsOf().filter((y) => cohort(y).length >= 20);
    const pts = ys.map((y) => {
      const vs = cohort(y).filter((r) => r.c !== HOME).map((r) => r.v).sort(d3.ascending);
      return { y, h: valOf(HOME, y), med: d3.quantile(vs, 0.5), q1: d3.quantile(vs, 0.25), q3: d3.quantile(vs, 0.75), n: vs.length };
    });
    const W = host.clientWidth || 500, H = 280, m = { t: 14, r: 18, b: 30, l: 44 };
    const x = d3.scaleLinear().domain(d3.extent(ys)).range([m.l, W - m.r]);
    const hi = d3.max(pts, (p) => Math.max(p.q3, p.h || 0));
    const yS = d3.scaleLinear().domain([0, hi]).nice().range([H - m.b, m.t]);
    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
      .attr('role', 'img').attr('aria-label', `${F[2]}: ישראל מול חציון ה-OECD לאורך השנים`);
    const grid = css('--line'), ink3 = css('--text-3'), other = css('--other'), accent = css('--s1'), surface = css('--surface');
    const gx = svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(Math.min(ys.length, W < 420 ? 4 : 8)).tickFormat(d3.format('d')).tickSize(0).tickPadding(8));
    const gy = svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(yS).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat((v) => `${v}%`));
    for (const g of [gx, gy]) {
      g.select('.domain').remove();
      g.selectAll('line').attr('stroke', grid);
      g.selectAll('text').attr('fill', ink3).attr('font-size', 12).attr('font-family', 'inherit');
    }
    svg.append('path').datum(pts).attr('fill', other).attr('fill-opacity', 0.25)
      .attr('d', d3.area().x((p) => x(p.y)).y0((p) => yS(p.q1)).y1((p) => yS(p.q3)));
    const line = (f) => d3.line().defined((p) => f(p) != null).x((p) => x(p.y)).y((p) => yS(f(p)));
    svg.append('path').datum(pts).attr('fill', 'none').attr('stroke', other).attr('stroke-width', 2).attr('d', line((p) => p.med));
    svg.append('path').datum(pts).attr('fill', 'none').attr('stroke', accent).attr('stroke-width', 2.5).attr('d', line((p) => p.h));
    svg.append('g').selectAll('circle').data(pts.filter((p) => p.h != null)).join('circle')
      .attr('cx', (p) => x(p.y)).attr('cy', (p) => yS(p.h)).attr('r', 3).attr('fill', accent).attr('stroke', surface).attr('stroke-width', 1.5);
    const cross = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).attr('stroke', ink3).attr('stroke-opacity', 0.6).attr('visibility', 'hidden');
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px] = d3.pointer(ev);
        const p = pts.reduce((a, b) => (Math.abs(x(b.y) - px) < Math.abs(x(a.y) - px) ? b : a));
        cross.attr('x1', x(p.y)).attr('x2', x(p.y)).attr('visibility', 'visible');
        showTip(ev, [[String(p.y), ''], [fmt(p.h), 'ישראל', accent], [fmt(p.med), `חציון ${p.n} מדינות`, other]]);
      })
      .on('pointerleave', () => { cross.attr('visibility', 'hidden'); hideTip(); });
  }

  function drawTable(y) {
    const t = $('#w-table');
    t.replaceChildren();
    const fs = FUNCS.filter((f) => f[0] !== 'ALL');
    const head = el('tr');
    for (const h of ['מדינה', 'הכל', ...fs.map((f) => f[1])]) head.append(el('th', null, h));
    const thead = el('thead'); thead.append(head);
    const tb = el('tbody');
    const rows = Object.keys(data.data).filter((c) => data.data[c][y]).map((c) => ({ c, r: data.data[c][y] }));
    rows.sort((a, b) => (b.r[func] ?? Object.values(b.r).reduce((s, v) => s + v, 0)) - (a.r[func] ?? Object.values(a.r).reduce((s, v) => s + v, 0)));
    for (const { c, r } of rows) {
      const tr = el('tr', c === HOME ? 'home' : null);
      tr.append(el('td', null, nameOf(c)), el('td', null, fmt(Object.values(r).reduce((s, v) => s + v, 0))), ...fs.map((f) => el('td', null, fmt(r[f[0]]))));
      tb.append(tr);
    }
    t.append(thead, tb);
    $('#w-table-sum').textContent = `כל ${rows.length} המדינות, כל התחומים, ${y}`;
  }

  function draw() {
    if (!data) return;
    const y = latest();
    drawTiles(y); drawStrip(y); drawTrend(); drawTable(y);
  }

  function init() {
    const bar = $('#w-funcs');
    if (!bar) return;
    bar.innerHTML = FUNCS.map((f) => `<button role="tab" data-func="${f[0]}" aria-selected="${f[0] === func}" title="${f[2]}">${f[1]}</button>`).join('');
    bar.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      func = b.dataset.func;
      bar.querySelectorAll('button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
      draw();
    }));
    fetch('data/oecd.json').then((r) => r.json()).then((d) => { data = d; draw(); });
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  }
  init();
})();
