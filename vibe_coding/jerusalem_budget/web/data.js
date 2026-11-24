// על הנתונים: כל מספר בדף מחושב מ-data/years.json ומקובצי השנים, אותם קבצים שהעוגה טוענת.

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const nf = new Intl.NumberFormat('he-IL');
const mil = (v) => `${(v / 1e6).toLocaleString('he-IL', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} מיליון ₪`;
const pct = (a, b) => `${b ? Math.round((a / b) * 100) : 0}%`;
const dateHe = (iso) => new Date(iso).toLocaleDateString('he-IL', { day: 'numeric', month: 'long', year: 'numeric' });
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let years = [];
const books = {};
async function book(y) {
  if (!books[y]) books[y] = await (await fetch(`data/budget_${y}.json`)).json();
  return books[y];
}

// ---------- בשורה התחתונה ----------
function drawTrust() {
  const checks = years.reduce((a, m) => a + (m.checks ? m.checks[0] : 0), 0);
  const bad = years.reduce((a, m) => a + (m.checks ? m.checks[1] : 0), 0);
  const cards = [
    { lvl: 'high', t: 'הסכומים ושורות התקציב', v: 'אפשר לסמוך',
      d: `כל סעיף נקרא מקובץ של העירייה ומקושר לעמוד שלו. ${nf.format(checks)} שורות סה"כ בספרים נבדקו מול הסעיפים${bad ? `, ${bad} מהן לא התאימו (ראו למטה)` : ', וכולן התאימו'}.` },
    { lvl: 'mid', t: 'השוואה בין שנים', v: 'טובה, עם הסתייגויות',
      d: 'לפי נושא היא יציבה, כי היא נשענת על קוד הסעיף. לפי מינהל יש קפיצות, כי העירייה שינתה את המבנה שלה. ובלי תיקון לאינפלציה.' },
    { lvl: 'mid', t: 'דברי ההסבר', v: 'חלקי',
      d: 'רק ל-2023–2026, ולא לכל סעיף. נקראו אוטומטית מ-PDF, ולפעמים יש בהם פגם קטן בפיסוק.' },
    { lvl: 'low', t: 'כמה הוצא בפועל', v: 'לא מכאן',
      d: 'זה תקציב מתוכנן. הביצוע בפועל יכול להיות שונה מאוד, והוא לא נכלל באתר.' },
  ];
  $('#trust').innerHTML = cards.map((c) => `<div class="trust ${c.lvl}">
    <div class="trust-t">${c.t}</div><div class="trust-v">${c.v}</div><p>${c.d}</p></div>`).join('');
  $('#checks-count').textContent = `בסך הכול ${nf.format(checks)} שורות סיכום ב-2023–2026` +
    (bad ? `. ${bad} לא התאימו, וכולן מוסברות בדף הזה.` : ', וכולן התאימו.');
}

// ---------- עדכניות ----------
function drawFresh() {
  const last = years[years.length - 1];
  const fetched = last.fetched;
  const ageDays = Math.round((Date.now() - new Date(fetched)) / 864e5);
  const nextYear = last.year + 1;
  $('#fresh').innerHTML = `
    <div class="fresh-row"><span class="k">השנה האחרונה</span><span class="v">${last.year}</span>
      <span class="d">התקציב שהעירייה פרסמה לשנה הזו. זה התקציב כפי שהוגש ואושר בתחילת השנה, לא אחרי שינויים שנעשו במהלכה.</span></div>
    <div class="fresh-row"><span class="k">הקבצים ירדו מאתר העירייה</span><span class="v">${dateHe(fetched)}</span>
      <span class="d">לפני ${nf.format(ageDays)} ימים. אם העירייה תיקנה קובץ אחרי התאריך הזה, התיקון לא נכנס.</span></div>
    <div class="fresh-row"><span class="k">התקציב המעודכן של ${last.year - 1}</span><span class="v">מתוך הספר של ${last.year}</span>
      <span class="d">כל ספר מביא גם את התקציב המעודכן של השנה שלפניו. לכן ${last.year} עצמה תקבל "תקציב מעודכן" רק כשיתפרסם הספר של ${nextYear}.</span></div>
    <div class="fresh-row"><span class="k">מתי יהיה ${nextYear}</span><span class="v">כשהעירייה תפרסם</span>
      <span class="d">תקציב העירייה מאושר בדרך כלל סביב תחילת השנה, ולפעמים באיחור של כמה חודשים (תקציב 2025 אושר בוועדת הכספים במרץ 2025). אחרי הפרסום צריך להריץ מחדש את הבנייה.</span></div>`;
}

// ---------- גרף השנים ----------
function drawChart() {
  const el = $('#totals-chart');
  const w = Math.max(300, el.clientWidth || 600), h = 240;
  const m = { top: 14, right: 40, bottom: 26, left: 48 };
  const svg = d3.select(el).html('').append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', '100%')
    .attr('role', 'img').attr('aria-label', 'הסכום הרשמי של התקציב בכל שנה, וחלק הסעיפים שיש להם דברי הסבר');
  const x = d3.scaleBand().domain(years.map((d) => d.year)).range([m.left, w - m.right]).padding(0.28);
  const y = d3.scaleLinear().domain([0, d3.max(years, (d) => d.total_expense) * 1.08]).nice(4).range([h - m.bottom, m.top]);
  const yp = d3.scaleLinear().domain([0, 1]).range([h - m.bottom, m.top]);
  svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(4)).join('line')
    .attr('x1', m.left).attr('x2', w - m.right).attr('y1', y).attr('y2', y);
  svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${h - m.bottom})`)
    .call(d3.axisBottom(x).tickSizeOuter(0).tickFormat((d) => (w < 480 ? `'${String(d).slice(2)}` : d)));
  svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(6).tickFormat((v) => `${v / 1e9} מיליארד`)).call((a) => a.select('.domain').remove());
  svg.append('g').attr('class', 'axis').attr('transform', `translate(${w - m.right},0)`)
    .call(d3.axisRight(yp).ticks(2).tickSize(0).tickPadding(6).tickFormat(d3.format('.0%'))).call((a) => a.select('.domain').remove());
  svg.selectAll('rect.bar').data(years).join('rect').attr('class', 'bar')
    .attr('x', (d) => x(d.year)).attr('width', x.bandwidth()).attr('rx', 3)
    .attr('y', (d) => y(d.total_expense)).attr('height', (d) => y(0) - y(d.total_expense))
    .attr('fill', css('--s1')).attr('opacity', (d) => (d.derived_from ? 0.45 : 0.85));
  const share = (d) => d.details / d.items;
  svg.append('path').attr('fill', 'none').attr('stroke', css('--s2')).attr('stroke-width', 2)
    .attr('d', d3.line().x((d) => x(d.year) + x.bandwidth() / 2).y((d) => yp(share(d)))(years));
  svg.selectAll('circle.ex').data(years).join('circle').attr('class', 'ex').attr('r', 3.5).attr('fill', css('--s2'))
    .attr('cx', (d) => x(d.year) + x.bandwidth() / 2).attr('cy', (d) => yp(share(d)));
  const tip = $('#totals-tip');
  svg.selectAll('rect.hit').data(years).join('rect').attr('class', 'hit').attr('fill', 'transparent')
    .attr('x', (d) => x(d.year) - (x.step() - x.bandwidth()) / 2).attr('width', x.step()).attr('y', m.top).attr('height', h - m.top - m.bottom)
    .on('mousemove', (ev, d) => {
      tip.innerHTML = `<b>${d.year}</b><div>תקציב: ${mil(d.total_expense)}</div>
        <div class="muted">${d.format === 'xlsx' ? 'מקובץ אקסל' : 'מספר PDF'}${d.derived_from ? ` (מהקובץ של ${d.derived_from})` : ''}</div>
        <div>דברי הסבר: ${pct(d.details, d.items)} מהסעיפים</div><div class="muted">לחצו לפרטים בטבלה</div>`;
      tip.hidden = false;
      const box = el.getBoundingClientRect();
      let left = ev.clientX - box.left + 12;
      if (left + tip.offsetWidth > box.width) left = ev.clientX - box.left - tip.offsetWidth - 12;
      tip.style.left = `${Math.max(0, left)}px`; tip.style.top = `${ev.clientY - box.top + 12}px`;
    })
    .on('mouseleave', () => { tip.hidden = true; })
    .on('click', (ev, d) => openYear(d.year, true));
}

// ---------- טבלת השנים ----------
function yearStatus(m) {
  if (m.derived_from) return { cls: 'mid', t: 'נגזרת' };
  if (m.note || (m.checks && m.checks[1])) return { cls: 'mid', t: 'עם הערה' };
  if (Math.abs(m.load_gap) > 1e6) return { cls: 'mid', t: 'פער העמסות' };
  return { cls: 'high', t: 'תקין' };
}
function yearDetails(m) {
  const out = [];
  const file = (u, label) => `<a href="${encodeURI(u)}" target="_blank" rel="noopener">${label}</a>`;
  out.push(`<b>מקור:</b> ${file(m.source, m.format === 'xlsx' ? 'קובץ האקסל' : 'ספר ההוצאות (PDF)')}` +
    (m.source_income && m.source_income !== m.source ? ` · ${file(m.source_income, 'ספר ההכנסות (PDF)')}` : ''));
  if (m.derived_from) out.push(m.note);
  if (m.from_minhal_files && m.from_minhal_files.length) {
    out.push(`הקובץ הכולל של ${m.from_minhal_files.map((s) => (s === 'expense' ? 'ההוצאות' : 'ההכנסות')).join(' ושל ')} שמור בגופן שלא ניתן לפענוח, ולכן הסעיפים נקראו מהקבצים של כל מינהל בנפרד. הסכום תואם לסכום הרשמי.`);
  }
  if (m.checks) out.push(`${nf.format(m.checks[0])} שורות סה"כ נבדקו${m.checks[1] ? `, ${m.checks[1]} לא התאימו` : ', כולן התאימו'}.`);
  if (m.checks && m.checks[1] && m.year === 2024) out.push('שלוש השורות שלא התאימו ב-2024 הן של "העמסות - הפחתות": בספר הסה"כ מודפס בלי סימן המינוס. הסכום עצמו נכון.');
  if (m.checks && m.checks[1] && m.year === 2023) out.push('השורה שלא התאימה ב-2023: סה"כ של "אגף לפיתוח אסטרטגי עירוני" במינהל התכנון. הסכום של המינהל כולו תקין, אבל כמה סעיפים בעמוד הזה אולי משויכים לאגף הלא נכון.');
  if (Math.abs(m.load_gap) > 1e5) {
    out.push(`<b>פער העמסות:</b> ${mil(m.load_gap)} (${(m.load_gap / m.total_expense * 100).toLocaleString('he-IL', { maximumFractionDigits: 2 })}% מהתקציב). ` +
      'בשנה הזו חלק מההפחתות המרכזיות (דלק, ליסינג, טלפונים) רשומות בלי "העמסה" תואמת אצל היחידות, ולכן הוצאתן מגדילה את סכום העוגה מעט מעבר לסכום הרשמי.');
  }
  if (m.manual) out.push(...m.manual.map((t) => `<b>השלמה ידנית:</b> ${esc(t)}`));
  if (m.note && !m.derived_from) out.push(esc(m.note));
  return out.map((p) => `<p>${p}</p>`).join('');
}
function drawTable() {
  const t = $('#years-table');
  t.innerHTML = `<thead><tr><th>שנה</th><th>מקור</th><th>סכום רשמי</th><th>בעוגה</th><th>סעיפים</th><th>עם הסבר</th><th>מצב</th></tr></thead>
    <tbody>${years.slice().reverse().map((m) => {
    const st = yearStatus(m);
    const diff = m.shown_expense - m.total_expense;
    return `<tr class="yr" data-year="${m.year}" id="row${m.year}" tabindex="0" aria-expanded="false">
        <td><b>${m.year}</b></td><td>${m.format === 'xlsx' ? 'אקסל' : 'PDF'}${m.derived_from ? ` (${m.derived_from})` : ''}</td>
        <td>${mil(m.total_expense)}</td>
        <td>${Math.abs(diff) < 5e5 ? '<span class="ok">זהה</span>' : `${diff > 0 ? '+' : ''}${mil(diff)}`}</td>
        <td>${nf.format(m.items)}</td><td>${pct(m.details, m.items)}</td>
        <td><span class="lvl ${st.cls}">${st.t}</span></td></tr>
      <tr class="yr-det" data-for="${m.year}" hidden><td colspan="7">${yearDetails(m)}</td></tr>`;
  }).join('')}</tbody>`;
  t.querySelectorAll('tr.yr').forEach((tr) => {
    const go = () => openYear(+tr.dataset.year);
    tr.addEventListener('click', go);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
}
function openYear(y, scroll) {
  const tr = document.querySelector(`tr.yr[data-year="${y}"]`);
  const det = document.querySelector(`tr.yr-det[data-for="${y}"]`);
  const open = det.hidden;
  det.hidden = !open;
  tr.setAttribute('aria-expanded', String(open));
  if (scroll) { det.hidden = false; tr.setAttribute('aria-expanded', 'true'); tr.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
}

// ---------- בדקו בעצמכם ----------
async function checkSum() {
  const y = +$('#chk-year').value;
  const m = years.find((d) => d.year === y);
  const out = $('#chk-sum');
  out.innerHTML = '<p class="muted">מסכם…</p>';
  const b = await book(y);
  const sum = (s) => b.items.filter((i) => i.s === s).reduce((a, i) => a + i.p, 0);
  const e = sum('e'), i = sum('i');
  const row = (label, mine, official) => {
    const d = mine - official;
    return `<tr><td>${label}</td><td>${nf.format(Math.round(mine))} ₪</td><td>${nf.format(official)} ₪</td>
      <td>${Math.abs(d) < 1 ? '<span class="ok">✓ זהה</span>' : `<span class="diff">${d > 0 ? '+' : ''}${nf.format(Math.round(d))} ₪</span>`}</td></tr>`;
  };
  out.innerHTML = `<p>סיכמתי עכשיו ${nf.format(b.items.length)} סעיפים של ${y}, בדפדפן שלכם:</p>
    <table class="chk-table"><thead><tr><th></th><th>סכום הסעיפים בעוגה</th><th>הסכום בקובץ העירייה</th><th>הפרש</th></tr></thead>
    <tbody>${row('הוצאות', e, m.total_expense)}${row('הכנסות', i, m.total_income)}</tbody></table>
    ${Math.abs(e - m.total_expense) >= 1 ? `<p class="muted">ההפרש בהוצאות: ${m.year === 2025 ? 'ראו "2025: הספר לא מסתכם בדיוק" למטה' : '<a href="#loadings">העמסות שלא מתקזזות</a> בשנה הזו'}.</p>` : ''}`;
}
async function randomItem() {
  const y = +$('#chk-year').value;
  const b = await book(y);
  // סעיף אקראי מכל הסעיפים החיוביים של השנה
  const pool = b.items.filter((i) => i.p > 0);
  const it = pool[Math.floor(Math.random() * pool.length)];
  const m = years.find((d) => d.year === y);
  const url = it.src || (it.s === 'i' ? m.source_income || m.source : m.source);
  const xlsx = url.includes('.xlsx');
  const code = `${it.c.slice(0, 4)}-${it.c.slice(4, 7)}-${it.c.slice(7)}`;
  const out = $('#chk-item');
  out.hidden = false;
  out.innerHTML = `<div class="rand">
    <div class="rand-name">${esc(it.n)}</div>
    <div class="rand-path">${esc([it.m, it.a, it.u].filter((x, i, a) => x && a.indexOf(x) === i).join(' ‹ '))} · ${it.s === 'e' ? 'הוצאה' : 'הכנסה'}</div>
    <div class="rand-num">${nf.format(it.p)} ₪ <span class="muted">ב-${y}</span>${it.b ? ` · ${nf.format(it.b)} ₪ <span class="muted">ב-${y - 1} (מעודכן)</span>` : ''}</div>
    <p>${xlsx
    ? `פתחו את <a href="${encodeURI(url)}" target="_blank" rel="noopener">קובץ האקסל של העירייה</a>, בגיליון "הוצאה/הכנסה לפי יחידה ארגונית", וחפשו את הסעיף <b dir="ltr">${it.c}</b>.`
    : `פתחו את <a href="${encodeURI(url)}#page=${it.pg}" target="_blank" rel="noopener">עמוד ${it.pg} בספר של העירייה</a> וחפשו את הסעיף <b dir="ltr">${code}</b>.`}
    ${it.x ? ' (הסעיף הזה תומלל ידנית.)' : ''}</p></div>`;
}

// ---------- התחלה ----------
(async () => {
  years = await (await fetch('data/years.json')).json();
  drawTrust(); drawFresh(); drawChart(); drawTable();
  const gap = years.filter((m) => Math.abs(m.load_gap) > 1e5);
  $('#load-gap').innerHTML = gap.length
    ? `ברוב השנים ההעמסות מתקזזות בדיוק. ב-${gap.map((m) => m.year).join(', ')} נשאר פער קטן, כי חלק מההפחתות המרכזיות (דלק, ליסינג, טלפונים) רשומות בלי "העמסה" תואמת אצל היחידות. בשנים האלה סכום העוגה גבוה מהסכום הרשמי ב-${gap.map((m) => `${(m.load_gap / 1e6).toLocaleString('he-IL', { maximumFractionDigits: 1 })} מיליון ב-${m.year}`).join(', ')} (לכל היותר כחצי אחוז).`
    : '';
  const sel = $('#chk-year');
  sel.innerHTML = years.slice().reverse().map((m) => `<option value="${m.year}">${m.year}</option>`).join('');
  sel.addEventListener('change', () => { $('#chk-item').hidden = true; checkSum(); });
  $('#chk-random').addEventListener('click', randomItem);
  checkSum();
  // קישור ישיר לשנה (#y2025) או למגבלה (#loadings)
  const h = location.hash.slice(1);
  if (/^y\d{4}$/.test(h) && years.some((m) => `y${m.year}` === h)) {
    const lim = document.getElementById(h);
    if (lim && lim.tagName === 'DETAILS') { lim.open = true; lim.scrollIntoView(); }
    openYear(+h.slice(1), !lim || lim.tagName !== 'DETAILS');
  } else if (h) {
    const el = document.getElementById(h);
    if (el && el.tagName === 'DETAILS') el.open = true;
    if (el) el.scrollIntoView();
  }
  let rsz;
  window.addEventListener('resize', () => { clearTimeout(rsz); rsz = setTimeout(drawChart, 150); });
})();
