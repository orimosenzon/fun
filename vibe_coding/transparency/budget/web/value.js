// "מה מקבלים תמורת הכסף": עלות ליחידת תוצאה, מול רשויות דומות.
// הנתונים: data/benchmark.json (tools/build_benchmark.py). צורת התצוגה היא "הדגשה": פרדס חנה-כרכור
// בצבע ההדגשה, הרשויות הדומות באפור. אין כאן ציון או דירוג של "טוב" ו"רע": עלות גבוהה יכולה להיות
// שירות תדיר יותר, ולכן מוצגת השוואה בלבד.
(() => {
  const DOMAINS = {
    waste: {
      label: 'פסולת',
      cost: { key: 'per_ton', name: 'שקל לטון פסולת', fmt: (v) => `${fmtN(v)} ₪` },
      out: { key: 'rec', name: 'אחוז מחזור', fmt: (v) => `${fmtN(v, 1)}%` },
      tiles: [
        { key: 'per_ton', name: 'עלות לטון פסולת', fmt: (v) => `${fmtN(v)} ₪` },
        { key: 'per_res', name: 'עלות פינוי וניקיון לתושב, בשנה', fmt: (v) => `${fmtN(v)} ₪` },
        { key: 'kg', name: 'פסולת לתושב ביום', fmt: (v) => `${fmtN(v, 2)} ק"ג` },
        { key: 'rec', name: 'מחזור', fmt: (v) => `${fmtN(v, 1)}%` },
      ],
      note: 'העלות היא כל התשלומים בפרק "תברואה" בדוח הכספי המבוקר: פינוי אשפה וגזם, הטמנה, ' +
        'וגם ניקיון רחובות. הטונות מחושבים מהלמ"ס (ק"ג ליום לנפש כפול אוכלוסייה). ' +
        'אחוז המחזור כולל גזם, ולכן ביישוב עם הרבה גינות פרטיות הוא גבוה יותר מטבעו. ' +
        'ל-2020 אין דוח מבוקר במאגר הפתוח.',
    },
    edu: {
      label: 'חינוך',
      cost: { key: 'per_pupil', name: 'שקל לתלמיד', fmt: (v) => `${fmtN(v)} ₪` },
      out: { key: 'bag', name: 'זכאות לבגרות', fmt: (v) => `${fmtN(v, 1)}%` },
      tiles: [
        { key: 'per_pupil', name: 'הוצאת המועצה על חינוך לתלמיד', fmt: (v) => `${fmtN(v)} ₪` },
        { key: 'bag', name: 'זכאים לבגרות מבין תלמידי י"ב', fmt: (v) => `${fmtN(v, 1)}%` },
      ],
      note: 'ההוצאה היא פרק החינוך בתקציב הרגיל של המועצה, כולל הכסף שמגיע ממשרד החינוך ועובר דרכה. ' +
        'שכר המורים שמשרד החינוך משלם ישירות לא נכלל, ולכן זו לא העלות המלאה של תלמיד. ' +
        'הבגרות היא של תלמידים שגרים ביישוב. נתוני הוצאה לתלמיד קיימים בקובצי הלמ"ס מ-2021.',
    },
  };
  const HOME = 'פרדס חנה-כרכור';
  let data = null;
  let domain = 'waste';

  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function fmtN(v, d = 0) {
    return v == null ? '–' : Number(v).toLocaleString('he-IL', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function latestYear(key) {
    const ys = data.years.filter((y) => data.rows.some((r) => r.y === y && r.n === HOME && r[key] != null));
    return ys[ys.length - 1];
  }
  function cohort(y) {
    const names = new Set([...data.peers, HOME]);
    return data.rows.filter((r) => r.y === y && names.has(r.n));
  }

  // ---------------------------------------------------------------- tiles
  function drawTiles(D, y) {
    const box = $('#v-kpis');
    box.replaceChildren();
    const rows = cohort(y);
    for (const t of D.tiles) {
      const home = rows.find((r) => r.n === HOME);
      const peers = rows.filter((r) => r.n !== HOME && r[t.key] != null).map((r) => r[t.key]);
      if (!home || home[t.key] == null || !peers.length) continue;
      const med = d3.median(peers);
      const rank = 1 + peers.filter((v) => v < home[t.key]).length;
      const tile = el('div', 'v-tile');
      tile.append(el('div', 'v-label', t.name), el('div', 'v-value', t.fmt(home[t.key])),
        el('div', 'v-sub', `חציון רשויות דומות: ${t.fmt(med)}`),
        el('div', 'v-sub', `מקום ${rank} מתוך ${peers.length + 1}, מהנמוך לגבוה`));
      box.append(tile);
    }
  }

  // ---------------------------------------------------------------- tooltip
  const tip = () => $('#v-tip');
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
  function hideTip() { tip().hidden = true; }

  // ---------------------------------------------------------------- scatter: cost vs outcome
  function drawScatter(D, y) {
    const host = $('#v-scatter');
    host.replaceChildren();
    const rows = cohort(y).filter((r) => r[D.cost.key] != null && r[D.out.key] != null);
    $('#v-scatter-title').textContent = `${D.cost.name} מול ${D.out.name}, ${y}`;
    const W = host.clientWidth || 600, H = Math.min(360, Math.max(260, W * 0.62));
    const m = { t: 16, r: 18, b: 42, l: 52 };
    const x = d3.scaleLinear().domain(d3.extent(rows, (r) => r[D.cost.key])).nice().range([m.l, W - m.r]);
    const yS = d3.scaleLinear().domain(d3.extent(rows, (r) => r[D.out.key])).nice().range([H - m.b, m.t]);
    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
      .attr('role', 'img').attr('aria-label', `${D.cost.name} מול ${D.out.name}`);
    const grid = css('--line'), ink2 = css('--text-2'), ink3 = css('--text-3');
    const gx = svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(-(H - m.t - m.b)).tickFormat((v) => `₪${d3.format('~s')(v).replace('k', 'K')}`));
    const gy = svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(yS).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat((v) => `${v}%`));
    for (const g of [gx, gy]) {
      g.select('.domain').remove();
      g.selectAll('line').attr('stroke', grid);
      g.selectAll('text').attr('fill', ink3).attr('font-size', 12).attr('font-family', 'inherit');
    }
    svg.append('text').attr('x', W - m.r).attr('y', H - 6).attr('text-anchor', 'end')
      .attr('fill', ink2).attr('font-size', 12).text(`${D.cost.name} ←`);
    svg.append('text').attr('x', m.l).attr('y', m.t - 4).attr('text-anchor', 'start')
      .attr('fill', ink2).attr('font-size', 12).text(`↑ ${D.out.name}`);

    // החציון של הרשויות הדומות בשני הצירים: קווי ייחוס דקים
    const peers = rows.filter((r) => r.n !== HOME);
    const mx = d3.median(peers, (r) => r[D.cost.key]), my = d3.median(peers, (r) => r[D.out.key]);
    svg.append('line').attr('x1', x(mx)).attr('x2', x(mx)).attr('y1', m.t).attr('y2', H - m.b).attr('stroke', ink3).attr('stroke-opacity', 0.5);
    svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', yS(my)).attr('y2', yS(my)).attr('stroke', ink3).attr('stroke-opacity', 0.5);
    svg.append('text').attr('x', x(mx) + 4).attr('y', m.t + 10).attr('fill', ink3).attr('font-size', 11).attr('text-anchor', 'start').text('חציון');

    const surface = css('--surface'), other = css('--other'), accent = css('--s1');
    const sorted = [...peers, ...rows.filter((r) => r.n === HOME)];
    const dots = svg.append('g').selectAll('g').data(sorted).join('g')
      .attr('transform', (r) => `translate(${x(r[D.cost.key])},${yS(r[D.out.key])})`)
      .attr('tabindex', 0).attr('role', 'img')
      .attr('aria-label', (r) => `${r.n}: ${D.cost.fmt(r[D.cost.key])}, ${D.out.fmt(r[D.out.key])}`);
    dots.append('circle').attr('r', 12).attr('fill', 'transparent');  // אזור פגיעה של 24px
    dots.append('circle').attr('class', 'v-dot').attr('r', (r) => (r.n === HOME ? 6 : 4.5))
      .attr('fill', (r) => (r.n === HOME ? accent : other)).attr('stroke', surface).attr('stroke-width', 2);
    const home = rows.find((r) => r.n === HOME);
    if (home) {
      const hx = x(home[D.cost.key]), left = hx > W / 2;
      svg.append('text').attr('x', hx + (left ? -10 : 10)).attr('y', yS(home[D.out.key]) + 4)
        .attr('text-anchor', left ? 'end' : 'start').attr('fill', css('--text')).attr('font-size', 13)
        .attr('font-weight', 500).text(HOME);
    }
    const on = (ev, r) => {
      dots.selectAll('.v-dot').attr('stroke', (d) => (d === r ? css('--text') : surface));
      showTip(ev, [[r.n, '', r.n === HOME ? accent : other], [D.cost.fmt(r[D.cost.key]), D.cost.name], [D.out.fmt(r[D.out.key]), D.out.name]]);
    };
    dots.on('pointermove', on).on('focus', (ev, r) => {
      const b = ev.currentTarget.getBoundingClientRect();
      on({ clientX: b.x + b.width / 2, clientY: b.y }, r);
    }).on('pointerleave blur', () => { dots.selectAll('.v-dot').attr('stroke', surface); hideTip(); });
  }

  // ---------------------------------------------------------------- trend: home vs peer median
  function drawTrend(D) {
    const host = $('#v-trend');
    host.replaceChildren();
    const key = D.cost.key;
    const ys = data.years.filter((y) => data.summary[y] && data.summary[y][key]);
    const pts = ys.map((y) => ({ y, s: data.summary[y][key], h: (data.rows.find((r) => r.y === y && r.n === HOME) || {})[key] }));
    $('#v-trend-title').textContent = `${D.cost.name} לאורך השנים`;
    const W = host.clientWidth || 600, H = 260, m = { t: 14, r: 18, b: 30, l: 52 };
    const x = d3.scaleLinear().domain(d3.extent(ys)).range([m.l, W - m.r]);
    const hi = d3.max(pts, (p) => Math.max(p.s.q3, p.h || 0));
    const yS = d3.scaleLinear().domain([0, hi]).nice().range([H - m.b, m.t]);
    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
      .attr('role', 'img').attr('aria-label', `${D.cost.name} לאורך השנים, מול חציון הרשויות הדומות`);
    const grid = css('--line'), ink3 = css('--text-3'), other = css('--other'), accent = css('--s1'), surface = css('--surface');
    const gx = svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues(ys).tickFormat(d3.format('d')).tickSize(0).tickPadding(8));
    const gy = svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(yS).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat((v) => `₪${d3.format('~s')(v).replace('k', 'K')}`));
    for (const g of [gx, gy]) {
      g.select('.domain').remove();
      g.selectAll('line').attr('stroke', grid);
      g.selectAll('text').attr('fill', ink3).attr('font-size', 12).attr('font-family', 'inherit');
    }
    // שנה בלי נתונים (2020 בפסולת) נשארת חור בקו, לא קו ישר שממציא ערך
    const all = d3.range(ys[0], ys[ys.length - 1] + 1).map((y) => pts.find((p) => p.y === y) || { y });
    svg.append('path').datum(all).attr('fill', other).attr('fill-opacity', 0.25)
      .attr('d', d3.area().defined((p) => p.s).x((p) => x(p.y)).y0((p) => yS(p.s.q1)).y1((p) => yS(p.s.q3)));
    const line = (f) => d3.line().defined((p) => f(p) != null).x((p) => x(p.y)).y((p) => yS(f(p)));
    svg.append('path').datum(all).attr('fill', 'none').attr('stroke', other).attr('stroke-width', 2)
      .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', line((p) => (p.s ? p.s.med : null)));
    svg.append('path').datum(all).attr('fill', 'none').attr('stroke', accent).attr('stroke-width', 2)
      .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', line((p) => p.h));
    svg.append('g').selectAll('circle').data(pts.filter((p) => p.h != null)).join('circle')
      .attr('cx', (p) => x(p.y)).attr('cy', (p) => yS(p.h)).attr('r', 4).attr('fill', accent).attr('stroke', surface).attr('stroke-width', 2);

    const cross = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).attr('stroke', ink3).attr('stroke-opacity', 0.6).attr('visibility', 'hidden');
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b)
      .attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px] = d3.pointer(ev);
        const p = pts.reduce((a, b) => (Math.abs(x(b.y) - px) < Math.abs(x(a.y) - px) ? b : a));
        cross.attr('x1', x(p.y)).attr('x2', x(p.y)).attr('visibility', 'visible');
        showTip(ev, [[String(p.y), ''], [p.h != null ? D.cost.fmt(p.h) : '–', HOME, accent],
          [D.cost.fmt(p.s.med), `חציון ${p.s.n} רשויות דומות`, other]]);
      })
      .on('pointerleave', () => { cross.attr('visibility', 'hidden'); hideTip(); });
  }

  // ---------------------------------------------------------------- table
  function drawTable(D, y) {
    const t = $('#v-table');
    t.replaceChildren();
    const head = el('tr');
    for (const h of ['רשות', 'תושבים', D.cost.name, D.out.name]) head.append(el('th', null, h));
    const thead = el('thead'); thead.append(head);
    const tb = el('tbody');
    const rows = cohort(y).filter((r) => r[D.cost.key] != null).sort((a, b) => a[D.cost.key] - b[D.cost.key]);
    for (const r of rows) {
      const tr = el('tr', r.n === HOME ? 'home' : null);
      tr.append(el('td', null, r.n), el('td', null, fmtN(r.pop)), el('td', null, D.cost.fmt(r[D.cost.key])),
        el('td', null, r[D.out.key] != null ? D.out.fmt(r[D.out.key]) : '–'));
      tb.append(tr);
    }
    t.append(thead, tb);
    $('#v-table-sum').textContent = `כל ${rows.length} הרשויות בהשוואה, ${y}`;
  }

  function draw() {
    if (!data) return;
    const D = DOMAINS[domain];
    const y = latestYear(D.cost.key);
    drawTiles(D, y);
    drawScatter(D, y);
    drawTrend(D);
    drawTable(D, y);
    $('#v-note').textContent = D.note;
  }

  function init() {
    const sec = $('#value');
    if (!sec) return;
    fetch('data/benchmark.json').then((r) => r.json()).then((d) => {
      data = d;
      $('#v-peers').textContent = `${d.peer_rule}: ${d.peers.join(', ')}.`;
      draw();
    });
    sec.querySelectorAll('[data-domain]').forEach((b) => b.addEventListener('click', () => {
      domain = b.dataset.domain;
      sec.querySelectorAll('[data-domain]').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
      draw();
    }));
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  }
  init();
})();
