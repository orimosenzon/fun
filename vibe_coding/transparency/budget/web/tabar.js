// "תב"רים וקרנות הרשות": התקציב הבלתי רגיל, מקורות המימון שלו, קרנות הרשות, ואישורי המליאה.
// הנתונים: data/tabar.json (tools/build_tabar.py). אותה שפה חזותית כמו shefa.js. כל הסכומים בקובץ באלפי ש"ח,
// חוץ מסכומי האישורים (approvals), שהם בשקלים.
(() => {
  const HOME = 'פרדס חנה-כרכור';
  let D = null;
  let domYear = null;

  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmtN = (v, d = 0) => (v == null ? '–'
    : Number(v).toLocaleString('he-IL', { minimumFractionDigits: d, maximumFractionDigits: d }));
  const fmtK = (v, d = 1) => (v == null ? '–' : `${fmtN(v / 1000, d)} מ' ₪`);  // אלפי ש"ח ← מיליונים
  const fmtS = (v) => (v == null ? '–' : v >= 1e6 ? `${fmtN(v / 1e6, 1)} מ' ₪` : `${fmtN(v)} ₪`);  // שקלים
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  const Y = (y) => D.years.find((r) => r.year === y);
  const audited = () => D.years.filter((r) => !r.provisional);

  // ---------------------------------------------------------------- tiles
  function tile(label, value, subs = []) {
    const t = el('div', 'v-tile');
    t.append(el('div', 'v-label', label), el('div', 'v-value', value));
    for (const s of subs) if (s) t.append(el('div', 'v-sub', s));
    return t;
  }
  const fill = (id, tiles) => $(id).replaceChildren(...tiles.filter(Boolean));

  // ---------------------------------------------------------------- tooltip
  function showTip(ev, lines) {
    const t = $('#sh-tip');
    t.replaceChildren(...lines.map(([v, l, key]) => {
      const row = el('div', 'v-tip-row');
      if (key) { const k = el('span', 'v-key'); k.style.background = key; row.append(k); }
      row.append(el('strong', null, v), el('span', 'v-tip-l', l));
      return row;
    }));
    t.hidden = false;
    const x = ev.clientX + window.scrollX, y = ev.clientY + window.scrollY;
    const w = t.offsetWidth;
    t.style.left = `${Math.max(8, Math.min(document.documentElement.clientWidth - w - 8, x - w / 2))}px`;
    t.style.top = `${y - t.offsetHeight - 14}px`;
  }
  const hideTip = () => { $('#sh-tip').hidden = true; };

  function axes(svg, x, yS, W, H, m, xFmt, yFmt) {
    const grid = css('--line'), ink3 = css('--text-3');
    const gx = svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickFormat(xFmt).tickSize(0).tickPadding(8));
    const gy = svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(yS).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(yFmt));
    for (const g of [gx, gy]) {
      g.select('.domain').remove();
      g.selectAll('line').attr('stroke', grid);
      g.selectAll('text').attr('fill', ink3).attr('font-size', 12).attr('font-family', 'inherit');
    }
  }
  function frame(host, H = 250) {
    host.replaceChildren();
    const W = host.clientWidth || 520, m = { t: 12, r: 14, b: 28, l: 44 };
    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);
    return { svg, W, H, m };
  }
  // שנים בציר: כולל 2020 כחור, כדי שהפער ייראה
  const yearAxis = (ys) => d3.range(d3.min(ys), d3.max(ys) + 1);
  const short = (W) => (v, i) => (W < 420 && i % 2 ? '' : String(v));

  // ---------------------------------------------------------------- grouped / stacked bars
  // series: [{name, color, vals: {year: value}}]; stacked או זה לצד זה. ערך null = אין נתון לשנה.
  function barChart(id, series, { stacked = false, yFmt, axisFmt, notes = {} } = {}) {
    const host = $(id);
    const { svg, W, H, m } = frame(host);
    const ys = yearAxis(series.flatMap((s) => Object.keys(s.vals).map(Number)));
    const x = d3.scaleBand().domain(ys).range([m.l, W - m.r]).padding(0.25);
    const inner = stacked ? null : d3.scaleBand().domain(series.map((s) => s.name)).range([0, x.bandwidth()]).padding(0.08);
    const tot = (y) => d3.sum(series, (s) => Math.max(0, s.vals[y] || 0));
    const max = stacked ? d3.max(ys, tot) : d3.max(series.flatMap((s) => Object.values(s.vals)));
    const yS = d3.scaleLinear().domain([0, max]).nice().range([H - m.b, m.t]);
    axes(svg, x, yS, W, H, m, short(W), axisFmt || yFmt);
    const surface = css('--surface');
    for (const y of ys) {
      let acc = 0;
      series.forEach((s) => {
        const v = s.vals[y];
        if (v == null || v <= 0) return;
        const x0 = stacked ? x(y) : x(y) + inner(s.name), w = stacked ? x.bandwidth() : inner.bandwidth();
        const y0 = yS(acc), y1 = yS(acc + v);
        svg.append('rect').attr('x', x0).attr('width', w).attr('y', y1).attr('height', Math.max(0, y0 - y1))
          .attr('fill', s.color).attr('stroke', stacked ? surface : 'none').attr('stroke-width', stacked ? 1 : 0)
          .attr('rx', stacked ? 0 : 2);
        if (stacked) acc += v;
      });
      if (!series.some((s) => s.vals[y] != null)) {
        svg.append('text').attr('x', x(y) + x.bandwidth() / 2).attr('y', H - m.b - 6).attr('text-anchor', 'middle')
          .attr('fill', css('--text-3')).attr('font-size', 11).text('אין נתון');
      }
    }
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b)
      .attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px] = d3.pointer(ev);
        const yv = ys.reduce((a, b) => (Math.abs(x(b) + x.bandwidth() / 2 - px) < Math.abs(x(a) + x.bandwidth() / 2 - px) ? b : a));
        const lines = [[String(yv), '']];
        for (const s of [...series].reverse()) lines.push([s.vals[yv] != null ? yFmt(s.vals[yv]) : '–', s.name, s.color]);
        if (stacked && series.some((s) => s.vals[yv] != null)) lines.push([yFmt(tot(yv)), 'סה"כ']);
        if (notes[yv]) lines.push([notes[yv], '']);
        showTip(ev, lines);
      })
      .on('pointerleave', hideTip);
  }

  // ---------------------------------------------------------------- line chart (פרדס חנה מול חציון)
  function lineChart(id, series, { yFmt, axisFmt } = {}) {
    const host = $(id);
    const { svg, W, H, m } = frame(host, 240);
    const xs = [...new Set(series.flatMap((s) => s.pts.map((p) => p.x)))].sort((a, b) => a - b);
    const x = d3.scaleLinear().domain(d3.extent(xs)).range([m.l, W - m.r]);
    const vals = series.flatMap((s) => s.pts.map((p) => p.y)).filter((v) => v != null);
    const yS = d3.scaleLinear().domain([0, d3.max(vals)]).nice().range([H - m.b, m.t]);
    const xAxis = d3.scaleLinear().domain(x.domain()).range(x.range());
    const grid = css('--line'), ink3 = css('--text-3'), surface = css('--surface');
    const gx = svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(xAxis).tickValues(xs).tickFormat(short(W)).tickSize(0).tickPadding(8));
    const gy = svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(yS).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(axisFmt || yFmt));
    for (const g of [gx, gy]) {
      g.select('.domain').remove();
      g.selectAll('line').attr('stroke', grid);
      g.selectAll('text').attr('fill', ink3).attr('font-size', 12).attr('font-family', 'inherit');
    }
    const all = yearAxis(xs);
    for (const s of series) {
      const byX = new Map(s.pts.map((p) => [p.x, p]));
      const data = all.map((v) => byX.get(v) || { x: v, y: null });
      svg.append('path').datum(data).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2)
        .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
        .attr('d', d3.line().defined((p) => p.y != null).x((p) => x(p.x)).y((p) => yS(p.y)));
      svg.append('g').selectAll('circle').data(s.pts.filter((p) => p.y != null)).join('circle')
        .attr('cx', (p) => x(p.x)).attr('cy', (p) => yS(p.y)).attr('r', 3.5)
        .attr('fill', s.color).attr('stroke', surface).attr('stroke-width', 2);
    }
    const cross = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).attr('stroke', ink3)
      .attr('stroke-opacity', 0.6).attr('visibility', 'hidden');
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b)
      .attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px] = d3.pointer(ev);
        const xv = xs.reduce((a, b) => (Math.abs(x(b) - px) < Math.abs(x(a) - px) ? b : a));
        cross.attr('x1', x(xv)).attr('x2', x(xv)).attr('visibility', 'visible');
        const lines = [[String(xv), '']];
        for (const s of series) {
          const p = s.pts.find((q) => q.x === xv);
          lines.push([p && p.y != null ? yFmt(p.y) : '–', s.name, s.color]);
          if (p && p.note) lines.push([p.note, '']);
        }
        showTip(ev, lines);
      })
      .on('pointerleave', () => { cross.attr('visibility', 'hidden'); hideTip(); });
  }
  function legend(id, series) {
    $(id).replaceChildren(...series.map((s) => {
      const sp = el('span');
      const k = el('i', 'v-swatch'); k.style.background = s.color; k.style.borderRadius = '2px';
      sp.append(k, document.createTextNode(s.name));
      return sp;
    }));
  }

  // ---------------------------------------------------------------- data helpers
  const ext = (s) => (s.gov || 0) + (s.other || 0) + (s.owners || 0);
  const fundsShare = (r) => (r.sources.funds != null && r.receipts ? r.sources.funds / r.receipts : null);
  function peerMedian(y, key) {
    const v = D.peer_rows.filter((p) => p.y === y && p.n !== HOME && p[key] != null).map((p) => p[key]);
    return v.length ? d3.median(v) : null;
  }
  const homePeer = (y) => D.peer_rows.find((p) => p.y === y && p.n === HOME) || {};

  // ---------------------------------------------------------------- sections
  function hero() {
    const last = audited().at(-1), y25 = Y(2025);
    const share = fundsShare(last);
    const med = peerMedian(last.year, 'funds_share');
    fill('#hero', [
      tile(`שולם בתב"רים ב-${last.year}`, fmtK(last.payments),
        [`${fmtN(100 * last.payments / last.regular_payments)}% נוסף על התקציב הרגיל`, `ב-2025: ${fmtK(y25.payments)}`]),
      tile('תב"רים פתוחים', fmtN(last.count_open),
        [`בסוף ${last.year}. תקציב מאושר: ${fmtK(last.approved_total, 0)}`]),
      tile('מקרנות הרשות', `${fmtN(100 * share)}%`,
        [`מהכסף שנכנס לתב"רים ב-${last.year}`, med != null ? `חציון רשויות דומות: ${fmtN(med)}%` : null]),
      tile('נשאר בקרנות', fmtK(last.funds.total.close),
        [`בסוף ${last.year}, לעומת ${fmtK(last.funds.total.open)} בתחילתה`,
          `ב-2025 נלקחה הלוואה של ${fmtK(y25.sources.loans, 0)}`]),
    ]);
  }

  function size() {
    const rows = D.years;
    const s = [
      { name: 'תקציב רגיל', color: css('--other'), vals: Object.fromEntries(rows.map((r) => [r.year, r.regular_payments])) },
      { name: 'תב"רים', color: css('--s1'), vals: Object.fromEntries(rows.map((r) => [r.year, r.payments])) },
    ];
    legend('#size-legend', s);
    const notes = Object.fromEntries(rows.map((r) => [r.year,
      `התב"רים: ${fmtN(100 * r.payments / r.regular_payments)}% מהתקציב הרגיל`]));
    barChart('#size-chart', s, { yFmt: (v) => fmtN(v / 1000, 1), axisFmt: (v) => fmtN(v / 1000), notes });
    const a = audited();
    const tot = d3.sum(a.filter((r) => r.year >= 2021), (r) => r.payments) + Y(2025).payments;
    fill('#size-kpis', [
      tile('שולם בתב"רים 2021–2025', fmtK(tot, 0), ['חמש שנים, כולל 2025 לפני ביקורת']),
      tile('תב"רים מול התקציב הרגיל', `${fmtN(100 * a.at(-1).payments / a.at(-1).regular_payments)}%`,
        [`ב-${a.at(-1).year}. ב-${a[0].year}: ${fmtN(100 * a[0].payments / a[0].regular_payments)}%`]),
      tile(`תב"רים שנפתחו עד סוף ${a.at(-1).year}`, fmtN(a.at(-1).count),
        [`${fmtN(a.at(-1).count_open)} מהם עדיין פתוחים`]),
    ]);
  }

  function sources() {
    const rows = D.years;
    const v = (f) => Object.fromEntries(rows.map((r) => [r.year, f(r.sources)]));
    const s = [
      { name: 'קרנות הרשות', color: css('--s1'), vals: v((x) => x.funds) },
      { name: 'ממשלה וגורמים חיצוניים', color: css('--s3'), vals: v((x) => ext(x)) },
      { name: 'הלוואות', color: css('--s2'), vals: v((x) => x.loans) },
      { name: 'תקציב רגיל ואחר', color: css('--other'), vals: v((x) => (x.regular || 0) + (x.rest || 0)) },
    ];
    legend('#sources-legend', s);
    const notes = { 2025: 'לפי הדוח הרבעוני, לא מבוקר' };
    barChart('#sources-chart', s, { stacked: true, yFmt: (x) => fmtN(x / 1000, 1), axisFmt: (x) => fmtN(x / 1000), notes });
    const a = audited(), last = a.at(-1), y25 = Y(2025);
    const sumF = d3.sum(a.filter((r) => r.year >= 2021), (r) => r.sources.funds) + y25.sources.funds;
    const sumAll = d3.sum(a.filter((r) => r.year >= 2021), (r) => r.receipts) + y25.sources.funds + ext(y25.sources)
      + y25.sources.loans + y25.sources.regular;
    const edu = last.ministries['משרד החינוך'];
    fill('#sources-kpis', [
      tile('מקרנות הרשות, 2021–2025', fmtK(sumF, 0), [`${fmtN(100 * sumF / sumAll)}% מכל מה שנכנס לתב"רים`]),
      tile(`ממשרדי הממשלה ב-${last.year}`, fmtK(last.sources.gov), [edu ? `מזה משרד החינוך: ${fmtK(edu)}` : null]),
      tile('הלוואות ב-2025', fmtK(y25.sources.loans, 0),
        [`יתרת ההלוואות לפירעון: ${fmtK(D.loans['2024'])} ← ${fmtK(D.loans['2025'])}`]),
    ]);
  }

  function funds() {
    const rows = audited().filter((r) => r.funds.total);
    const t = (f) => Object.fromEntries(rows.map((r) => [r.year, f(r.funds.total)]));
    const s = [
      { name: 'נכנס לקרנות', color: css('--s3'), vals: t((x) => x.in) },
      { name: 'הועבר לתב"רים', color: css('--s1'), vals: t((x) => x.to_tabar) },
      { name: 'הועבר לתקציב הרגיל', color: css('--s4'), vals: t((x) => x.to_regular) },
      { name: 'יתרה בסוף השנה', color: css('--s2'), vals: t((x) => x.close) },
    ];
    s[3].vals[2025] = Y(2025).funds.total.close;
    legend('#funds-legend', s);
    barChart('#funds-chart', s, { yFmt: (x) => fmtN(x / 1000, 1), axisFmt: (x) => fmtN(x / 1000),
      notes: { 2025: 'יתרה לפי הדוח הרבעוני, לא מבוקר' } });

    const lv = audited().filter((r) => r.levy);
    const ls = [
      { name: 'חיובים', color: css('--other'), vals: Object.fromEntries(lv.map((r) => [r.year, r.levy.charged])) },
      { name: 'נגבה', color: css('--s3'), vals: Object.fromEntries(lv.map((r) => [r.year, r.levy.collected])) },
    ];
    legend('#levy-legend', ls);
    barChart('#levy-chart', ls, { yFmt: (x) => fmtN(x / 1000, 1), axisFmt: (x) => fmtN(x / 1000) });

    const last = audited().at(-1), f = last.funds, prev = Y(last.year - 1).funds.total;
    const hs = f['השבחה'] || {};
    fill('#funds-kpis', [
      tile(`יצא מהקרנות ב-${last.year}`, fmtK(f.total.to_tabar + f.total.to_regular, 1),
        [`נכנס: ${fmtK(f.total.in)}`, `${fmtK(f.total.to_tabar)} לתב"רים, ${fmtK(f.total.to_regular)} לתקציב הרגיל`]),
      tile('יתרה בסוף השנה', fmtK(f.total.close), [`${prev.close ? `${fmtK(prev.close)} שנה קודם` : ''}`,
        `2025: ${fmtK(Y(2025).funds.total.close)}, לפי הדוח הרבעוני`]),
      tile(`נגבה היטל השבחה ב-${last.year}`, fmtK(last.levy.collected),
        [`${fmtN(100 * last.levy.collected / last.levy.charged)}% מהחיובים`, hs.to_regular ? `${fmtK(hs.to_regular)} ממנו הועברו לתקציב הרגיל` : null]),
    ]);

    const box = $('#funds-callout');
    box.replaceChildren(el('h3', null, 'מה רואים כאן'));
    const ul = el('ul');
    const y21 = Y(2021).funds.total;
    for (const txt of [
      `ב-2021 העבירה המועצה ${fmtK(y21.to_tabar, 0)} מהקרנות לתב"רים, והיתרה ירדה מ-${fmtK(y21.open, 0)} ל-${fmtK(y21.close, 0)}.`,
      `ב-${last.year} זה קרה שוב: ${fmtK(f.total.to_tabar, 0)} לתב"רים ו-${fmtK(f.total.to_regular, 0)} לתקציב הרגיל, ` +
        `והיתרה ירדה מ-${fmtK(f.total.open)} ל-${fmtK(f.total.close)}. הכסף שנצבר בקרנות נוצל.`,
      `ב-2025 נכנסו לתב"רים ${fmtK(Y(2025).sources.loans, 0)} מהלוואות, לעומת ${fmtK(last.sources.loans)} שנה קודם. ` +
        `יתרת ההלוואות של המועצה עלתה מ-${fmtK(D.loans['2024'])} ל-${fmtK(D.loans['2025'])}, והחזרן יבוא מהתקציב הרגיל בשנים הבאות.`,
      `מקרנות הפיתוח מותר להעביר לתקציב הרגיל סכום שמכסה את הוצאות התכנון וההנדסה (ועוד 15% תקורה). ` +
        `ב-${last.year} הועברו לתקציב הרגיל ${fmtK(f.total.to_regular)} מכל הקרנות, לעומת ${fmtK(Y(last.year - 1).funds.total.to_regular)} שנה קודם` +
        (hs.to_regular ? `. מזה ${fmtK(hs.to_regular)} מקרן היטל השבחה.` : '.'),
    ]) ul.append(el('li', null, txt));
    const src = el('p', 'note');
    const a1 = el('a', null, 'דוח מבקר המועצה על היטלי השבחה');
    a1.href = D.docs.audit_levy; a1.target = '_blank'; a1.rel = 'noopener';
    src.append('לקריאה נוספת: ', a1, '.');
    box.append(ul, src);
  }

  function where() {
    const ys = audited().filter((r) => Object.keys(r.domains).length).map((r) => r.year);
    if (!domYear) domYear = ys.at(-1);
    const btns = $('#dom-years');
    btns.replaceChildren(...ys.map((y) => {
      const b = el('button', null, String(y));
      b.type = 'button';
      b.setAttribute('aria-pressed', String(y === domYear));
      b.addEventListener('click', () => { domYear = y; where(); });
      return b;
    }));
    const r = Y(domYear);
    const entries = Object.entries(r.domains);
    const max = entries[0][1];
    const box = $('#dom-bars');
    box.replaceChildren(...entries.filter(([, v]) => v >= 50).map(([k, v]) => {
      const row = el('div', 'tb-bar');
      const bar = el('div', 'tb-bar-fill'); bar.style.width = `${(100 * v / max).toFixed(1)}%`;
      const track = el('div', 'tb-bar-track'); track.append(bar);
      row.append(el('span', 'tb-bar-l', k), track,
        el('span', 'tb-bar-v', `${fmtK(v)} · ${fmtN(100 * v / r.payments)}%`));
      return row;
    }));
    // תיק התב"רים הפתוחים
    const last = audited().filter((x) => x.portfolio).at(-1);
    $('#pf-year').textContent = String(last.year);
    const t = $('#pf-table');
    const head = el('tr');
    for (const h of ['תחום', 'תב"רים פתוחים', 'תקציב מאושר', 'שולם עד היום', 'שולם ב-' + last.year]) head.append(el('th', null, h));
    const thead = el('thead'); thead.append(head);
    const tb = el('tbody');
    const pf = Object.entries(last.portfolio).filter(([, v]) => v.approved).sort((a, b) => b[1].approved - a[1].approved);
    for (const [k, v] of pf) {
      const tr = el('tr');
      tr.append(el('td', null, k), el('td', null, fmtN(v.open)), el('td', null, fmtK(v.approved)),
        el('td', null, fmtK(v.spent)), el('td', null, fmtK(last.domains[k] || 0)));
      tb.append(tr);
    }
    const tr = el('tr', 'home');
    tr.append(el('td', null, 'סה"כ'), el('td', null, fmtN(last.count_open)), el('td', null, fmtK(last.approved_total)),
      el('td', null, fmtK(d3.sum(pf, ([, v]) => v.spent))), el('td', null, fmtK(last.payments)));
    tb.append(tr);
    t.replaceChildren(thead, tb);
  }

  // ---------------------------------------------------------------- approvals
  const fromFunds = (a) => (a.sources || []).some((s) => /קרנות/.test(s.src));
  function projects() {
    const A = D.approvals || [];
    const sec = $('#projects');
    if (!A.length) { sec.hidden = true; return; }
    // הקטנה נשמרת כסכום חיובי; בסיכומים ובתצוגה היא שלילית
    const amt = (a) => (a.amount == null ? null : a.action === 'הקטנה' ? -a.amount : a.amount);
    const byYear = d3.rollup(A, (v) => ({ n: v.length, sum: d3.sum(v, (a) => amt(a) || 0),
      missing: v.filter((a) => a.amount == null).length,
      funds: d3.sum(v, (a) => d3.sum((a.sources || []).filter((s) => /קרנות/.test(s.src)), (s) => s.amount || 0)) }),
    (a) => a.date.slice(0, 4));
    const yrs = [...byYear.keys()].sort();
    const meetings = new Set(A.map((a) => a.date)).size;
    const lastDate = A.map((a) => a.date).sort().at(-1);
    fill('#ap-kpis', [
      tile('החלטות תב"ר במליאה', fmtN(A.length), [`ב-${fmtN(meetings)} ישיבות, ${yrs[0]}–${yrs.at(-1)}`]),
      ...yrs.slice(-3).map((y) => {
        const b = byYear.get(y);
        return tile(`אושר ב-${y}${y === lastDate.slice(0, 4) ? ` (עד ${+lastDate.slice(5, 7)}/${y})` : ''}`, fmtS(b.sum), [
          `${fmtN(b.n)} החלטות${b.missing ? `, ב-${fmtN(b.missing)} מהן לא נאמר סכום` : ''}`,
          b.funds ? `מקרנות הרשות, כשנאמר במפורש: ${fmtS(b.funds)}` : null]);
      }),
    ]);
    const render = () => {
      const q = $('#ap-search').value.trim();
      const onlyF = $('#ap-funds').checked;
      const byAmt = $('#ap-sort').value === 'amount';
      const rows = A.filter((a) => (!onlyF || fromFunds(a))
        && (!q || `${a.num} ${a.name} ${(a.sources || []).map((s) => s.src).join(' ')}`.includes(q)))
        .sort(byAmt ? (a, b) => (amt(b) ?? -1e12) - (amt(a) ?? -1e12)
          : (a, b) => (b.date + b.num).localeCompare(a.date + a.num));
      const t = $('#ap-table');
      const head = el('tr');
      for (const h of ['תאריך', 'תב"ר', 'שם', 'החלטה', 'סכום', 'מקורות']) head.append(el('th', null, h));
      const thead = el('thead'); thead.append(head);
      const tb = el('tbody');
      const lim = render.all || q ? 400 : 30;
      for (const a of rows.slice(0, lim)) {
        const tr = el('tr');
        const src = (a.sources || []).map((s) => (s.amount ? `${s.src} ${fmtS(s.amount)}` : s.src)).join(' · ');
        const link = el('a', null, a.date.split('-').reverse().join('/'));
        link.href = a.pdf + (a.page ? `#page=${a.page}` : ''); link.target = '_blank'; link.rel = 'noopener';
        const td = el('td'); td.append(link);
        tr.append(td, el('td', null, a.num || ''), el('td', null, a.name), el('td', null, a.action || ''),
          el('td', null, a.amount != null ? (amt(a) < 0 ? `−${fmtS(a.amount)}` : fmtS(a.amount)) + (a.certain === false ? ' ?' : '') : '–'),
          el('td', 'tb-src', src));
        if (a.note) tr.title = a.note;
        if (a.certain === false) tr.classList.add('tb-unsure');
        tb.append(tr);
      }
      t.replaceChildren(thead, tb);
      const more = $('#ap-more');
      more.textContent = `${fmtN(rows.length)} שורות${rows.length > lim ? `, מוצגות ${lim} הראשונות` : ''}. ` +
        'הסכום הוא של ההחלטה עצמה (תב"ר חדש או ההגדלה), לא של כל התב"ר. לחיצה על התאריך פותחת את הפרוטוקול בעמוד ' +
        'הנכון. "?" מסמן סכום שלא נשמע או לא נכתב בבירור; ריחוף על השורה מציג הערה. ';
      if (rows.length > lim && !render.all) {
        const b = el('button', 'share sh-copy', `הצגת כל ${fmtN(rows.length)} השורות`);
        b.type = 'button';
        b.onclick = () => { render.all = true; render(); };
        more.append(b);
      }
    };
    $('#ap-search').oninput = render;
    $('#ap-funds').onchange = render;
    $('#ap-sort').onchange = render;
    render();
  }

  function peersSec() {
    const ys = [...new Set(D.peer_rows.map((p) => p.y))].sort();
    const two = (key, fmt, axis) => [
      { name: 'חציון רשויות דומות', color: css('--other'), pts: ys.map((y) => ({ x: y, y: peerMedian(y, key) })) },
      { name: HOME, color: css('--s1'), pts: ys.map((y) => ({ x: y, y: homePeer(y)[key] ?? null })) },
    ];
    lineChart('#pay-chart', two('pay_res'), { yFmt: (v) => `${fmtN(v)} ₪`, axisFmt: (v) => fmtN(v) });
    lineChart('#share-chart', two('funds_share'), { yFmt: (v) => `${fmtN(v)}%`, axisFmt: (v) => fmtN(v) });
  }

  function asks() {
    const items = [
      ['רשימת התב"רים המלאה',
        'אבקש את רשימת כל התב"רים הפתוחים של המועצה נכון לסוף 2025, ולכל תב"ר: מספר, שם, תחום, שנת פתיחה, התקציב המאושר ' +
        'ומקורות המימון שאושרו (קרנות הרשות, משרדי ממשלה לפי משרד, הלוואות, אחר), הביצוע המצטבר בהוצאות ובהכנסות, ' +
        'והתב"רים שנסגרו ב-2024–2025 עם העודף או הגירעון הסופי של כל אחד.'],
      ['קרנות הרשות לפי קרן',
        'אבקש את התנועה בכל אחת מקרנות הרשות בשנים 2021–2025 (יתרת פתיחה, תקבולים, העברות לתב"רים, העברות לתקציב הרגיל, ' +
        'יתרת סגירה), כולל שמות הקרנות שמופיעות בביאור 5 לדוח המבוקר כ"קרנות אחרות", ואת הפירוט של ההעברות מכל קרן ' +
        'לפי מספר תב"ר.'],
      ['"פיתוח כלכלי" בתב"רים',
        'בדוח המבוקר לשנת 2024 (נספח 1 לטופס 3) שולמו 37.6 מיליון ₪ בתב"רים בתחום "פיתוח כלכלי", ותקציבם המאושר 97.6 ' +
        'מיליון ₪. אבקש את רשימת התב"רים שמסווגים בתחום הזה, עם התקציב, מקורות המימון והביצוע של כל אחד.'],
      ['ההלוואות של 2025',
        'אבקש פירוט של ההלוואות שהמועצה נטלה ב-2025 (כ-21 מיליון ₪ לפי הדוח הרבעוני): המלווה, הסכום, התנאים ותקופת ' +
        'ההחזר, אישור המליאה ומשרד הפנים, ואילו תב"רים הן מממנות.'],
      ['תוכנית פיתוח רב-שנתית',
        'אבקש את תוכנית הפיתוח (תוכנית העבודה לתקציבי פיתוח) של המועצה לשנים 2022–2026, כפי שפורסמו בעבר לשנים ' +
        '2014–2021, ואת תחזית ההכנסות של קרנות הרשות לשנים הקרובות.'],
    ];
    const ol = $('#asks');
    ol.replaceChildren(...items.map(([title, text]) => {
      const li = el('li');
      const btn = el('button', 'share sh-copy', 'העתקה');
      btn.type = 'button';
      btn.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(text); toast('הנוסח הועתק'); } catch { toast('לא הצלחתי להעתיק'); }
      });
      li.append(el('strong', null, title), el('p', null, text), btn);
      return li;
    }));
  }

  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg; t.hidden = false;
    clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, 2200);
  }

  function about() {
    $('#doc-summary').href = D.docs.summary_2025;
    $('#doc-q4').href = D.docs.q4_2025;
    const p = $('#plans');
    D.plans.forEach((pl, i) => {
      const a = el('a', null, String(pl.year));
      a.href = pl.url; a.target = '_blank'; a.rel = 'noopener';
      if (pl.total) a.title = `אומדן כולל: ${fmtK(pl.total, 0)}`;
      if (i) p.append(', ');
      p.append(a);
    });
    $('#peers').textContent = `רשויות דומות: ${D.peer_rule}: ${D.peers.join(', ')}.`;
  }

  function draw() {
    if (!D) return;
    hero(); size(); sources(); funds(); where(); peersSec();
  }

  fetch('data/tabar.json').then((r) => r.json()).then((d) => {
    D = d;
    draw();
    projects();
    asks();
    about();
  });
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
})();
