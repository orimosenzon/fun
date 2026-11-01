// "שפ"ע במספרים": פסולת, גזם, גינון, שטח ציבורי, תנועה והוצאה לתושב.
// הנתונים: data/shefa.json (tools/build_shefa.py). אותה שפה חזותית כמו value.js: פרדס חנה-כרכור בצבע
// ההדגשה, הרשויות הדומות באפור, וקווים של 2px עם ריחוף שמראה את כל הסדרות בשנה.
(() => {
  const HOME = 'פרדס חנה-כרכור';
  let D = null;

  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmtN = (v, d = 0) => (v == null ? '–'
    : Number(v).toLocaleString('he-IL', { minimumFractionDigits: d, maximumFractionDigits: d }));
  const fmtM = (v) => (v == null ? '–' : `${fmtN(v / 1e6, 1)} מ' ₪`);
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  const homeRows = () => D.rows.filter((r) => r.n === HOME).sort((a, b) => a.y - b.y);
  const homeAt = (y) => D.rows.find((r) => r.n === HOME && r.y === y) || {};
  function latest(key) {
    const r = homeRows().filter((x) => x[key] != null);
    return r[r.length - 1];
  }
  function peersAt(y, key) {
    return D.rows.filter((r) => r.y === y && r.n !== HOME && r[key] != null).map((r) => r[key]);
  }

  // ---------------------------------------------------------------- tiles
  function tile(label, value, subs = []) {
    const t = el('div', 'v-tile');
    t.append(el('div', 'v-label', label), el('div', 'v-value', value));
    for (const s of subs) if (s) t.append(el('div', 'v-sub', s));
    return t;
  }
  // אריח עם השוואה לחציון הרשויות הדומות באותה שנה
  function cmpTile(label, key, fmt, extra) {
    const r = latest(key);
    if (!r) return null;
    const peers = peersAt(r.y, key);
    const rank = 1 + peers.filter((v) => v < r[key]).length;
    return tile(label, fmt(r[key]), [
      `${r.y}. חציון רשויות דומות: ${fmt(d3.median(peers))}`,
      `מקום ${rank} מתוך ${peers.length + 1}, מהנמוך לגבוה`, extra]);
  }
  function fill(id, tiles) {
    const box = $(id);
    box.replaceChildren(...tiles.filter(Boolean));
  }

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

  // ---------------------------------------------------------------- line chart
  // series: [{name, color, pts: [{x, y, note?}]}]. שנה בלי ערך נשארת חור בקו.
  function lineChart(id, series, { yFmt, axisFmt, zero = true, bars = false } = {}) {
    const host = $(id);
    host.replaceChildren();
    const xs = [...new Set(series.flatMap((s) => s.pts.map((p) => p.x)))].sort((a, b) => a - b);
    const W = host.clientWidth || 520, H = 240, m = { t: 12, r: 14, b: 28, l: 48 };
    const x = bars
      ? d3.scaleBand().domain(xs).range([m.l, W - m.r]).padding(0.3)
      : d3.scaleLinear().domain(d3.extent(xs)).range([m.l, W - m.r]);
    const cx = (v) => (bars ? x(v) + x.bandwidth() / 2 : x(v));
    const vals = series.flatMap((s) => s.pts.map((p) => p.y)).filter((v) => v != null);
    const yS = d3.scaleLinear().domain([zero ? 0 : d3.min(vals) * 0.9, d3.max(vals)]).nice().range([H - m.b, m.t]);
    const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
      .attr('role', 'img').attr('aria-label', series.map((s) => s.name).join(', '));
    const grid = css('--line'), ink3 = css('--text-3'), surface = css('--surface');
    const every = W < 420 ? 2 : 1;
    const gx = svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call((bars ? d3.axisBottom(x) : d3.axisBottom(x).tickValues(xs)).tickFormat((v, i) => (i % every ? '' : String(v)))
        .tickSize(0).tickPadding(8));
    const gy = svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(yS).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(axisFmt || yFmt));
    for (const g of [gx, gy]) {
      g.select('.domain').remove();
      g.selectAll('line').attr('stroke', grid);
      g.selectAll('text').attr('fill', ink3).attr('font-size', 12).attr('font-family', 'inherit');
    }
    if (bars) {
      const s = series[0];
      svg.append('g').selectAll('path').data(s.pts.filter((p) => p.y != null)).join('path')
        .attr('fill', s.color)
        .attr('d', (p) => {  // עמודה עם קצה עליון מעוגל 4px, צמודה לבסיס
          const x0 = x(p.x), w = x.bandwidth(), y0 = yS(0), y1 = yS(p.y), r = Math.min(4, w / 2, y0 - y1);
          return `M${x0},${y0}V${y1 + r}Q${x0},${y1} ${x0 + r},${y1}H${x0 + w - r}Q${x0 + w},${y1} ${x0 + w},${y1 + r}V${y0}Z`;
        });
    } else {
      const all = d3.range(xs[0], xs[xs.length - 1] + 1);
      for (const s of series) {
        const byX = new Map(s.pts.map((p) => [p.x, p]));
        const data = all.map((v) => byX.get(v) || { x: v, y: null });
        svg.append('path').datum(data).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2)
          .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
          .attr('d', d3.line().defined((p) => p.y != null).x((p) => x(p.x)).y((p) => yS(p.y)));
        svg.append('g').selectAll('circle').data(s.pts.filter((p) => p.y != null)).join('circle')
          .attr('cx', (p) => x(p.x)).attr('cy', (p) => yS(p.y)).attr('r', s.dots === false ? 0 : 3.5)
          .attr('fill', s.color).attr('stroke', surface).attr('stroke-width', 2);
      }
    }
    const cross = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).attr('stroke', ink3)
      .attr('stroke-opacity', 0.6).attr('visibility', 'hidden');
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b)
      .attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px] = d3.pointer(ev);
        const xv = xs.reduce((a, b) => (Math.abs(cx(b) - px) < Math.abs(cx(a) - px) ? b : a));
        cross.attr('x1', cx(xv)).attr('x2', cx(xv)).attr('visibility', bars ? 'hidden' : 'visible');
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
      const k = el('i', 'v-line'); k.style.background = s.color;
      sp.append(k, document.createTextNode(s.name));
      return sp;
    }));
  }
  // פרדס חנה-כרכור מול חציון הרשויות הדומות
  function homeVsPeers(id, key, yFmt, axisFmt) {
    const ys = D.years.filter((y) => D.summary[y] && D.summary[y][key] != null);
    lineChart(id, [
      { name: 'חציון רשויות דומות', color: css('--other'), pts: ys.map((y) => ({ x: y, y: D.summary[y][key] })) },
      { name: HOME, color: css('--s1'), pts: ys.map((y) => ({ x: y, y: homeAt(y)[key] ?? null })) },
    ], { yFmt, axisFmt });
  }

  // ---------------------------------------------------------------- sections
  function hero() {
    const st = D.statutory, a = D.audit;
    const pop = st.pop;
    fill('#hero', [
      cmpTile('הוצאה של המועצה לתושב, בשנה', 'per_res', (v) => `${fmtN(v)} ₪`),
      tile('גזם וגרוטאות שפונו', `${fmtN(a.yard_tons)} טון`, [`ב-${a.year}, לפי מבקר המועצה`,
        `כ-${fmtN(Math.round(a.yard_cost / a.yard_tons / 10) * 10)} ₪ לטון`]),
      tile('כל הפסולת ביישוב, כולל גזם', `${fmtN(Math.round(latest('tons').tons / 100) * 100)} טון`,
        [`${latest('tons').y}, מחושב מהלמ"ס`]),
      tile('שטח ציבורי פתוח בתוכניות', `${fmtN(st.dunam['שטח ציבורי פתוח'])} דונם`,
        [`${fmtN(st.dunam['שטח ציבורי פתוח'] * 1000 / pop, 1)} מ"ר לתושב`, 'כולל מגרשים שעוד לא פותחו']),
    ]);
  }

  function money() {
    const r = latest('per_res');
    const first = homeRows().find((x) => x.per_res);
    fill('#money-kpis', [
      cmpTile('הוצאה לתושב', 'per_res', (v) => `${fmtN(v)} ₪`),
      tile('סך ההוצאות בתקציב הרגיל', fmtM(r.expense * 1000), [`${r.y}, בפועל לפי הלמ"ס`]),
      tile(`שינוי מ-${first.y}`, `${fmtN((r.per_res / first.per_res - 1) * 100)}%`,
        [`${fmtN(first.per_res)} ← ${fmtN(r.per_res)} ₪ לתושב`,
          `חציון הרשויות הדומות: ${fmtN((D.summary[r.y].per_res / D.summary[first.y].per_res - 1) * 100)}%`]),
    ]);
    homeVsPeers('#money-chart', 'per_res', (v) => `${fmtN(v)} ₪`, (v) => `${fmtN(v / 1000, 0)}K`);
  }

  function waste() {
    const a = D.audit, b = D.budget, last = b.years.length - 1;
    const yard = b.lines['1712300751'];
    fill('#waste-kpis', [
      cmpTile('פסולת לתושב ביום', 'kg', (v) => `${fmtN(v, 2)} ק"ג`),
      cmpTile('אחוז מחזור', 'rec', (v) => `${fmtN(v, 1)}%`, 'כולל גזם, ולכן גבוה ביישוב עם הרבה חצרות'),
      tile(`פינוי גזם בתקציב ${b.years[last]}`, fmtM(yard.plan[last]),
        [`ב-${b.years[0]}: ${fmtM(yard.plan[0])}`, `פי ${fmtN(yard.plan[last] / yard.plan[0], 1)}`]),
      tile('אשפה ביתית שנאספה', `${fmtN(a.household_tons)} טון`,
        [`ב-${a.year}, לפי מבקר המועצה`, `החוזה תומחר לפי ${fmtN(a.household_priced_tons)} טון`]),
    ]);

    const box = $('#audit');
    box.replaceChildren();
    box.append(el('h3', null, `מה מצא מבקר המועצה (שנת ${a.year}, הוצג במליאה ב-6/3/2025)`));
    const ul = el('ul');
    for (const t of [
      `גזם וגרוטאות: כ-${fmtN(a.yard_tons)} טון בשנה, בעלות של ${fmtM(a.yard_cost)}. כלומר כ-${fmtN(Math.round(a.yard_cost / a.yard_tons / 10) * 10)} ₪ לטון. ` +
        'הגזם נשקל במאזני גשר במועצה, והקבלן מקבל תשלום קבוע לטון. בתעודות השקילה, השקילה האחרונה של היום הייתה לרוב לפני 13:00, אף שיום העבודה בחוזה נמשך עד אחר הצהריים.',
      `אשפה ביתית: כ-${fmtN(a.household_tons)} טון, בחוזה של מחיר קבוע שלא תלוי במשקל (התשלומים ב-2023: כ-${fmtM(a.household_paid_2023)}). ` +
        'במבנה כזה למועצה אין תמריץ כספי להפחית או למחזר. בדיון נאמר שהמכרז החדש ישנה את זה.',
      'פחים כתומים (אריזות): המבקר כתב שהאיסוף "אינו אפקטיבי מבחינת מיצוי הפוטנציאל", בגלל מיקום הפחים, ערבוב עם פסולת רגילה והיעדר אכיפה.',
      'לא נקבעו מדדי ביצוע לקבלנים, ובשנה שנבדקה כמעט לא הוטלו קנסות על הפרות.',
    ]) ul.append(el('li', null, t));
    const src = el('p', 'note');
    const link = el('a', null, 'הפרוטוקול באתר המועצה, עמודים 16–24');
    link.href = a.url; link.target = '_blank'; link.rel = 'noopener';
    src.append('המקור: ', link, '. אלה הטונות היחידים לפי סוג שפורסמו.');
    box.append(ul, src);
    $('#audit-link').href = a.url;

    // תקציב לפי מחלקה. "איסוף אשפה" בלי הגזם, שמוצג כסדרה משלו
    const ys = b.years, dep = b.depts;
    const sub = (arr, minus) => arr.map((v, i) => (v == null ? null : v - (minus[i] || 0)));
    const defs = [
      { name: 'איסוף אשפה', plan: sub(dep['אסוף אשפה'].plan, yard.plan), act: sub(dep['אסוף אשפה'].actual, yard.actual), color: css('--s1') },
      { name: 'פינוי גזם וגרוטאות', plan: yard.plan, act: yard.actual, color: css('--s2') },
      { name: 'ניקוי רחובות', plan: dep['ניקוי רחובות'].plan, act: dep['ניקוי רחובות'].actual, color: css('--s3') },
      { name: 'גנים ונטיעות', plan: dep['גנים ונטיעות'].plan, act: dep['גנים ונטיעות'].actual, color: css('--s4') },
    ];
    const series = defs.map((d) => ({
      name: d.name, color: d.color,
      pts: ys.map((y, i) => ({ x: y, y: d.plan[i], note: d.act[i] != null ? `צפי ביצוע: ${fmtN(d.act[i] / 1e6, 1)}` : null })),
    }));
    legend('#spend-legend', series);
    lineChart('#spend-chart', series, { yFmt: (v) => fmtN(v / 1e6, 1), axisFmt: (v) => fmtN(v / 1e6) });

    const tons = homeRows().filter((r) => r.tons);
    lineChart('#tons-chart', [{ name: 'טונות בשנה', color: css('--s1'), pts: tons.map((r) => ({ x: r.y, y: r.tons })) }],
      { yFmt: (v) => fmtN(v), axisFmt: (v) => `${fmtN(v / 1000)}K`, bars: true });
  }

  function garden() {
    const b = D.budget, last = b.years.length - 1, g = b.depts['גנים ונטיעות'];
    const pop = D.statutory.pop;
    fill('#garden-kpis', [
      tile(`גנים ונטיעות בתקציב ${b.years[last]}`, fmtM(g.plan[last]),
        [`${fmtN(g.plan[last] / pop)} ₪ לתושב`, `ב-${b.years[0]}: ${fmtM(g.plan[0])}`]),
      cmpTile('מים לגינון ציבורי, לתושב', 'water_garden_res', (v) => `${fmtN(v, 1)} מ"ק`,
        `סה"כ ${fmtN(latest('water_garden').water_garden)} אלף מ"ק`),
      tile(`גיזום עצים בתקציב ${b.years[last]}`, fmtM(b.lines['1746001750'].plan[last]),
        [`ב-${b.years[0]}: ${fmtM(b.lines['1746001750'].plan[0])}`]),
    ]);
    const cols = [css('--s1'), css('--s2')];
    const series = ['1746000432', '1746001750'].map((c, i) => ({
      name: b.lines[c].label, color: cols[i],
      pts: b.years.map((y, j) => ({ x: y, y: b.lines[c].plan[j],
        note: b.lines[c].actual[j] != null ? `צפי ביצוע: ${fmtN(b.lines[c].actual[j] / 1e6, 2)}` : null })),
    }));
    legend('#lines-legend', series);
    lineChart('#lines-chart', series, { yFmt: (v) => fmtN(v / 1e6, 2), axisFmt: (v) => fmtN(v / 1e6) });
    homeVsPeers('#water-chart', 'water_garden_res', (v) => fmtN(v, 1));
  }

  function space() {
    const st = D.statutory, pop = st.pop, land = D.land[HOME];
    const per = (k) => `${fmtN(st.dunam[k] * 1000 / pop, 1)} מ"ר לתושב`;
    fill('#space-kpis', [
      tile('שטח ציבורי פתוח (שצ"פ) בתוכניות', `${fmtN(st.dunam['שטח ציבורי פתוח'])} דונם`, [per('שטח ציבורי פתוח')]),
      tile('מגרשים למבני ציבור בתוכניות', `${fmtN(st.dunam['מבנים ומוסדות ציבור'])} דונם`, [per('מבנים ומוסדות ציבור')]),
      tile('גינון ופארקים בפועל', `${fmtN(land['2024'].parks * 1000)} דונם`,
        [`2013: ${fmtN(land['2013'].parks * 1000)}, 2021: ${fmtN(land['2021'].parks * 1000)}`, 'לפי הלמ"ס, מספר נמוך מהמציאות']),
    ]);
    const cats = [['residential', 'מגורים'], ['education', 'חינוך והשכלה'], ['public', 'שירותים ציבוריים'],
      ['culture', 'תרבות, פנאי וספורט'], ['welfare', 'בריאות ורווחה'], ['parks', 'גינון לנוי ופארק ציבורי'],
      ['forest', 'יער וחורש'], ['total', 'כל שטח השיפוט']];
    const years = Object.keys(land).sort();
    const t = $('#land-table');
    const head = el('tr');
    for (const h of ['שימוש', ...years.map((y) => `סקר ${y}`)]) head.append(el('th', null, h));
    const thead = el('thead'); thead.append(head);
    const tb = el('tbody');
    for (const [k, name] of cats) {
      const tr = el('tr', k === 'total' ? 'home' : null);
      tr.append(el('td', null, name), ...years.map((y) => el('td', null, fmtN(land[y][k], 2))));
      tb.append(tr);
    }
    t.replaceChildren(thead, tb);
  }

  function traffic() {
    fill('#traffic-kpis', [
      tile('כלי רכב רשומים ביישוב', fmtN(latest('vehicles').vehicles),
        [`${latest('vehicles').y}. מתוכם ${fmtN(latest('vehicles').private)} פרטיים`,
          `ב-${homeRows()[0].y}: ${fmtN(homeRows()[0].vehicles)}`]),
      cmpTile('רכב פרטי לכל 1,000 תושבים', 'private_k', (v) => fmtN(v)),
      cmpTile('תאונות עם נפגעים, לכל 10,000 תושבים', 'acc_10k', (v) => fmtN(v, 1),
        `${fmtN(latest('accidents').accidents)} תאונות ב-${latest('accidents').y}`),
    ]);
    homeVsPeers('#cars-chart', 'private_k', (v) => fmtN(v));
    homeVsPeers('#acc-chart', 'acc_10k', (v) => fmtN(v, 1));
    $('#traffic-n').textContent = fmtN(D.traffic.length);
    const t = $('#counts-table');
    const head = el('tr');
    for (const h of ['תאריך', 'מקום', 'שעות', 'הזמין', 'מס\' ספירה']) head.append(el('th', null, h));
    const thead = el('thead'); thead.append(head);
    const tb = el('tbody');
    for (const c of D.traffic) {
      const tr = el('tr');
      tr.append(el('td', null, c.date.split('-').reverse().join('/')), el('td', null, c.name), el('td', null, c.hours),
        el('td', null, c.by), el('td', null, String(c.id)));
      tb.append(tr);
    }
    t.replaceChildren(thead, tb);
  }

  function asks() {
    const recent = D.traffic.filter((c) => c.date >= '2023').map((c) => `${c.id} (${c.name}, ${c.date.slice(0, 4)})`);
    const items = [
      ['פסולת לפי סוג',
        'אבקש את דוחות "טופס 1" (כמויות פסולת ומחזור שנתי של הרשות המקומית) שהמועצה הגישה למשרד להגנת הסביבה לשנים 2019–2025: ' +
        'טונות לפי זרם (מעורב, גזם, גרוטאות, אריזות, נייר וקרטון, זכוכית, מכלי משקה, פסולת בניין) ולפי אתר קליטה, ואת העלות השנתית לכל זרם (איסוף, הובלה, היטל הטמנה וקליטה).'],
      ['גזם: טונות ומחיר לטון',
        'אבקש את כמויות הגזם והגרוטאות שנשקלו בכל חודש מ-2020 ועד היום, את המחיר לטון לפי כל חוזה (כולל החוזה שנחתם אחרי המכרז החדש), ' +
        'ואת התשלומים בפועל לקבלן הגזם בכל שנה.'],
      ['גינון ציבורי',
        'אבקש את פירוט ההוצאה על גינון ציבורי לשנים 2020–2025 בפועל: חוזי קבלני הגינון והתחזוקה ותמורתם, ' +
        'השטח המגונן שהמועצה מתחזקת (בדונמים) בכל שנה, ומספר העצים הציבוריים שנגזמו ונכרתו.'],
      ['שטחים ציבוריים',
        'אבקש את רשימת המגרשים בייעוד ציבורי (שצ"פ ומבני ציבור) שבבעלות המועצה או בניהולה, עם שטח, גוש וחלקה ומצב פיתוח, ' +
        'ואת רשימת המגרשים שייעודם הציבורי שונה או בוטל מאז 2015.'],
      ['ספירות תנועה',
        `אבקש את קובצי ספירות התנועה שבוצעו בתוך תחום המועצה, ממאגר ספירות התנועה של משרד התחבורה: ${recent.join(', ')}, ` +
        'וכן כל סקר תנועה או ספירה שהמועצה הזמינה מאז 2020 (למשל לצורכי ועדת התנועה או תוכניות בנייה).'],
      ['ביצוע תקציב',
        'אבקש את דוחות ביצוע התקציב הרבעוניים של המועצה לשנים 2023–2025 ברמת סעיף, כדי שאפשר יהיה להשוות בין התקציב שתוכנן לבין מה שהוצא בפועל.'],
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

  function draw() {
    if (!D) return;
    hero(); money(); waste(); garden(); space(); traffic();
  }

  fetch('data/shefa.json').then((r) => r.json()).then((d) => {
    D = d;
    $('#peers').textContent = `רשויות דומות: עיריות ומועצות מקומיות באשכול חברתי-כלכלי 6–8, עם 20–90 אלף תושבים: ${d.peers.join(', ')}.`;
    draw();
    asks();
  });
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
})();
