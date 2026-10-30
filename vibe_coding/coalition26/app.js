// מי יקים את הממשלה? הלוגיקה, החישובים והאנימציות.
(() => {
'use strict';
const D = window.DATA;
const $ = (s, el = document) => el.querySelector(s);
const ROLE_W = { in: 1, support: 0.5, abstain: 0.2 };
const ROLE_NAME = { in: 'בקואליציה', support: 'תומכת מבחוץ', abstain: 'נמנעת' };
const PM = { N: { party: 'L', name: 'בנימין נתניהו' }, E: { party: 'Y', name: 'גדי איזנקוט' } };
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- מפלגות ----------
const BASE = Object.fromEntries(D.parties.map(p => [p.id, p]));
const LX = Object.assign({}, BASE.L, D.likudAfter);
function party(id) { return id === 'Lx' ? LX : BASE[id]; }
function idsFor(mode) { return D.parties.map(p => (mode === 'E' && p.id === 'L') ? 'Lx' : p.id); }
function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}
function color(id) { return party(id).color[isDark() ? 1 : 0]; }
function seats(id) { return party(id).seats; }

// ---------- מצב ----------
let state = { mode: null, roles: {} };

function active(st = state) { return Object.keys(st.roles); }
function rolesOf(id) { return D.roles[id] || ['in']; }

// פסילה הדדית חוסמת כל תפקיד, גם הימנעות (ב-2021 סמוטריץ' חסם אפילו ממשלה שנשענת על הימנעות של רע"מ)
function blockReason(id, st = state, role) {
  if (!st.mode) return 'בחרו קודם מי מרכיב את הממשלה';
  if (id === PM[st.mode].party) return null;
  const mb = D.modeBlocked[st.mode][id === 'Lx' ? 'L' : id];
  if (mb && !(st.mode === 'E' && id === 'Lx')) return mb;
  if (role === undefined) {
    const ok = rolesOf(id).find(r => !blockReason(id, st, r));
    return ok ? null : blockReason(id, st, rolesOf(id)[0]);
  }
  for (const other of active(st)) {
    if (other === id) continue;
    for (const [a, b, why] of D.impossible) {
      if ((a === id && b === other) || (b === id && a === other)) return why;
    }
  }
  return null;
}

function tally(st = state) {
  let inS = 0, sup = 0, abs = 0;
  for (const [id, r] of Object.entries(st.roles)) {
    if (r === 'in') inS += seats(id); else if (r === 'support') sup += seats(id); else abs += seats(id);
  }
  const forS = inS + sup;
  return { inS, sup, abs, forS, against: 120 - forS - abs, pass: forS > 120 - forS - abs };
}

// מפלגה היא "ציר" אם עזיבתה (מעבר לנגד) מפילה את הממשלה
function pivots(st = state) {
  const t = tally(st);
  if (!t.pass) return [];
  return Object.entries(st.roles).filter(([id, r]) => r !== 'abstain').filter(([id, r]) => {
    const s = seats(id);
    return !(t.forS - s > t.against + s);
  }).map(([id]) => id);
}

// ---------- תוצאות מדיניות ----------
function outcomes(st = state) {
  const piv = new Set(pivots(st));
  const pm = st.mode ? PM[st.mode].party : null;
  const res = {};
  for (const iss of D.issues) {
    let num = 0, den = 0;
    for (const [id, r] of Object.entries(st.roles)) {
      const pos = iss.pos[id]; if (!pos) continue;
      const [p, c] = pos;
      let w = seats(id) * c * ROLE_W[r];
      if (id === pm) w *= 1.5;
      if (piv.has(id) && c === 3) w *= 1.6;
      num += w * p; den += w;
    }
    res[iss.id] = den ? num / den : 0;
  }
  return res;
}
const bucket = v => String(Math.max(-2, Math.min(2, Math.round(v))));

function helper(st, out) {
  return {
    mode: st.mode,
    has: id => id in st.roles,
    isIn: id => st.roles[id] === 'in',
    role: id => st.roles[id],
    out: k => out[k]
  };
}

function brokenPledges(st, out) {
  const h = helper(st, out);
  return D.pledges.map(p => ({ p, sev: p.trig(h) })).filter(x => x.sev > 0).sort((a, b) => b.sev - a.sev);
}

function concessions(st, out) {
  const rows = [];
  for (const id of active(st)) {
    const gives = [], comp = [], wins = [];
    for (const iss of D.issues) {
      const pos = iss.pos[id]; if (!pos) continue;
      const [p, c, want] = pos;
      if (c < 2) continue;
      const gap = Math.abs(p - out[iss.id]);
      const item = { iss, want, gap, c };
      if (gap >= 1.25) gives.push(item); else if (gap >= 0.6) comp.push(item); else wins.push(item);
    }
    gives.sort((a, b) => b.gap * b.c - a.gap * a.c);
    rows.push({ id, role: st.roles[id], gives, comp, wins });
  }
  return rows;
}

function spreadOf(st) {
  let tot = 0, wtot = 0;
  for (const iss of D.issues) {
    let sw = 0, sm = 0; const pts = [];
    for (const [id, r] of Object.entries(st.roles)) {
      if (r !== 'in') continue;
      const pos = iss.pos[id]; if (!pos) continue;
      const w = seats(id) * pos[1]; if (!w) continue;
      pts.push([pos[0], w]); sw += w; sm += w * pos[0];
    }
    if (!sw) continue;
    const mu = sm / sw;
    const sd = Math.sqrt(pts.reduce((a, [p, w]) => a + w * (p - mu) ** 2, 0) / sw);
    tot += sd * sw; wtot += sw;
  }
  return wtot ? tot / wtot : 0;
}

// האם מפלגות ה"נגד" יכולות להרכיב יחד 61 (בשביל אי-אמון קונסטרוקטיבי)?
function altMajority(st) {
  const ids = idsFor(st.mode).filter(id => !(id in st.roles));
  let best = 0;
  const n = ids.length;
  for (let mask = 1; mask < (1 << n); mask++) {
    const set = ids.filter((_, i) => mask & (1 << i));
    let ok = true;
    outer: for (const [a, b] of D.impossible) {
      if (set.includes(a) && set.includes(b)) { ok = false; break outer; }
    }
    if (!ok) continue;
    const s = set.reduce((a, id) => a + seats(id), 0);
    if (s > best) best = s;
  }
  return best;
}

function stability(st, out, broken) {
  const t = tally(st);
  const f = [];
  const nParties = Object.values(st.roles).filter(r => r !== 'abstain').length;
  if (t.inS >= 61) f.push(['גודל הרוב', Math.min(18, (t.inS - 61) * 2.5), `${t.inS} ח"כים בקואליציה, ${t.inS - 60} מעל הסף`]);
  else f.push(['ממשלת מיעוט', -10, `רק ${t.inS} ח"כים בתוך הקואליציה; כל חוק תלוי במי שבחוץ`]);
  const sp = spreadOf(st);
  f.push(['פערים אידאולוגיים', -Math.round(sp * 14), sp > 1 ? 'השותפות רחוקות זו מזו בנושאי ליבה' : sp > 0.6 ? 'מחלוקות משמעותיות בכמה נושאים' : 'השותפות קרובות יחסית']);
  const pv = pivots(st);
  if (pv.length) f.push(['מפלגות עם כוח הפלה', -Math.min(15, pv.length * 3), `${pv.length} מפלגות יכולות להפיל את הממשלה לבד`]);
  const sev = broken.reduce((a, b) => a + b.sev, 0);
  if (sev) f.push(['הבטחות שנשברות', -Math.min(20, Math.round(sev * 1.5)), `${broken.length} הבטחות, וזה מזמין מרד ח"כים ועריקות`]);
  if (nParties > 3) f.push(['מספר שותפות', -(nParties - 3) * 3, `${nParties} מפלגות שצריך לרצות`]);
  const sup = Object.values(st.roles).filter(r => r === 'support').length;
  const abs = Object.values(st.roles).filter(r => r === 'abstain').length;
  if (sup) f.push(['תלות בתמיכה מבחוץ', -4 * sup, 'מי שבחוץ לא מחויב לשום דבר']);
  if (abs) f.push(['תלות בהימנעות', -6 * abs, 'מספיק שהנמנעים יצביעו נגד']);
  const rot = (st.mode === 'N' && 'Y' in st.roles) || (st.mode === 'E' && 'Lx' in st.roles);
  if (rot) f.push(['רוטציה', -6, 'רוטציות בישראל נוטות להתפרק לפני מועד ההחלפה']);
  const alt = altMajority(st);
  if (alt < 61) f.push(['אין חלופה באופוזיציה', 10, `המתנגדים יכולים להגיע לכל היותר ל-${alt}, ובלי 61 אי אפשר להחליף ממשלה בלי בחירות`]);
  else f.push(['יש חלופה באופוזיציה', -5, `מפלגות מחוץ לממשלה יכולות להגיע ל-${alt}`]);
  const score = Math.max(5, Math.min(95, Math.round(80 + f.reduce((a, x) => a + x[1], 0))));
  let label, life, cls;
  if (score >= 72) { label = 'יציבה'; life = 'יכולה לשרוד קדנציה מלאה'; cls = 'good'; }
  else if (score >= 52) { label = 'סבירה'; life = 'כשנתיים עד שלוש'; cls = 'warn'; }
  else if (score >= 32) { label = 'שבירה'; life = 'כשנה עד שנתיים'; cls = 'serious'; }
  else { label = 'שבירה מאוד'; life = 'חודשים עד שנה'; cls = 'crit'; }
  return { score, label, life, cls, factors: f };
}

function tensions(st) {
  const ids = Object.entries(st.roles).filter(([, r]) => r !== 'abstain').map(([id]) => id);
  const out = [];
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const a = ids[i], b = ids[j];
    let t = 0; const top = [];
    for (const iss of D.issues) {
      const pa = iss.pos[a], pb = iss.pos[b]; if (!pa || !pb) continue;
      const v = Math.abs(pa[0] - pb[0]) * Math.min(pa[1], pb[1]);
      t += v; if (v >= 4) top.push([iss.name, v]);
    }
    top.sort((x, y) => y[1] - x[1]);
    out.push({ a, b, t, issues: top.map(x => x[0]) });
  }
  return out.sort((x, y) => y.t - x.t);
}

function matchScenario(st) {
  return D.scenarios.find(s => s.mode === st.mode &&
    Object.keys(s.roles).length === Object.keys(st.roles).length &&
    Object.entries(s.roles).every(([id, r]) => st.roles[id] === r)) || null;
}

// ---------- המקום בכנסת (חצי עיגול) ----------
const SVG_W = 440, SVG_H = 262, CX = SVG_W / 2, CY = 236, R = 160;
const SEATS = (() => {
  const rows = 6, r0 = 0.42, r1 = 1;
  const radii = d3.range(rows).map(i => r0 + (r1 - r0) * i / (rows - 1));
  const tot = d3.sum(radii);
  const counts = radii.map(r => Math.round(120 * r / tot));
  counts[rows - 1] += 120 - d3.sum(counts);
  const out = [];
  radii.forEach((r, i) => {
    const k = counts[i];
    for (let j = 0; j < k; j++) out.push({ r, th: Math.PI * j / (k - 1) });
  });
  out.sort((a, b) => a.th - b.th || b.r - a.r);
  return out;
})();
const xy = (r, th) => [CX + R * r * Math.cos(th), CY - R * r * Math.sin(th)];
const BOUNDARY_61 = (SEATS[59].th + SEATS[60].th) / 2;

let svg, gSeats, gLabels, gMarker, tip;

function seatOrder(st) {
  // מחזיר רשימת קבוצות לפי הסדר מימין לשמאל
  const ids = st.mode ? idsFor(st.mode) : D.parties.map(p => p.id);
  if (!st.mode) {
    const order = ['L', 'O', 'RZ', 'S', 'U', 'R', 'J', 'YB', 'D', 'B', 'Y'];
    return order.map(id => ({ id, role: 'neutral' }));
  }
  const pm = PM[st.mode].party;
  const ins = active(st).filter(id => st.roles[id] === 'in' && id !== pm).sort((a, b) => seats(b) - seats(a));
  const sup = active(st).filter(id => st.roles[id] === 'support').sort((a, b) => seats(b) - seats(a));
  const abs = active(st).filter(id => st.roles[id] === 'abstain');
  const opp = ids.filter(id => !(id in st.roles)).sort((a, b) => seats(a) - seats(b));
  return [
    { id: pm, role: 'in' }, ...ins.map(id => ({ id, role: 'in' })),
    ...sup.map(id => ({ id, role: 'support' })), ...abs.map(id => ({ id, role: 'abstain' })),
    ...opp.map(id => ({ id, role: 'opp' }))
  ];
}

function initHemicycle() {
  svg = d3.select('#hemi').attr('viewBox', `0 0 ${SVG_W} ${SVG_H}`);
  gMarker = svg.append('g').attr('class', 'marker61');
  const [x1, y1] = xy(0.34, BOUNDARY_61), [x2, y2] = xy(1.09, BOUNDARY_61);
  gMarker.append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2);
  const [lx, ly] = xy(0.27, BOUNDARY_61);
  gMarker.append('text').attr('x', lx).attr('y', ly).attr('text-anchor', 'middle').text('61');
  gSeats = svg.append('g');
  gLabels = svg.append('g').attr('class', 'plabels');
  tip = $('#tip');
  const all = [];
  D.parties.forEach(p => { for (let k = 0; k < p.seats; k++) all.push({ key: `${p.id}-${k}`, base: p.id, k }); });
  gSeats.selectAll('circle').data(all, d => d.key).enter().append('circle')
    .attr('r', 7.2).attr('class', 'seat')
    .on('mouseenter', (ev, d) => showTip(ev, d)).on('mousemove', (ev, d) => showTip(ev, d))
    .on('mouseleave', () => tip.hidden = true);
}

function showTip(ev, d) {
  const id = d.cur || d.base, p = party(id);
  const role = state.mode ? (state.roles[id] ? ROLE_NAME[state.roles[id]] : 'באופוזיציה') : '';
  tip.innerHTML = `<b>${p.name}</b> · ${p.seats} מנדטים<br>${p.leader}${role ? `<br><span>${role}</span>` : ''}`;
  tip.hidden = false;
  const box = $('#stage').getBoundingClientRect();
  tip.style.left = (ev.clientX - box.left) + 'px';
  tip.style.top = (ev.clientY - box.top - 12) + 'px';
}

function seatStyle(sel, role, id) {
  const c = color(id);
  if (role === 'in' || role === 'neutral') sel.attr('fill', c).attr('stroke', c).attr('stroke-width', 0).attr('fill-opacity', 1).attr('stroke-opacity', 1);
  else if (role === 'support') sel.attr('fill', c).attr('stroke', c).attr('stroke-width', 1.8).attr('fill-opacity', 0.4).attr('stroke-opacity', 1);
  else if (role === 'abstain') sel.attr('fill', 'var(--surface)').attr('stroke', c).attr('stroke-width', 1.8).attr('fill-opacity', 1).attr('stroke-opacity', 0.9);
  else sel.attr('fill', c).attr('stroke', c).attr('stroke-width', 0).attr('fill-opacity', 0.26).attr('stroke-opacity', 1);
}

let animToken = 0;
function layoutHemicycle(st, animate) {
  const token = ++animToken;
  const groups = seatOrder(st);
  const target = new Map(); // key -> {r, th, role, id, gi}
  let idx = 0;
  const labelData = [];
  groups.forEach((g, gi) => {
    const base = g.id === 'Lx' ? 'L' : g.id;
    const n = seats(g.id);
    const ths = [];
    for (let k = 0; k < n; k++) {
      const s = SEATS[idx++];
      target.set(`${base}-${k}`, { r: s.r, th: s.th, role: g.role, id: g.id, gi });
      ths.push(s.th);
    }
    labelData.push({ id: g.id, base, th: d3.mean(ths), role: g.role, gi });
  });

  const joinGroups = groups.filter(g => g.role === 'in' || g.role === 'support' || g.role === 'abstain');
  const STEP = reduceMotion || !animate ? 0 : 520;
  const BASE_DELAY = reduceMotion || !animate ? 0 : 450;
  const DUR = reduceMotion || !animate ? 0 : 900;

  gSeats.selectAll('circle').each(function (d) {
    const t = target.get(d.key);
    d.cur = t.id;
    const sel = d3.select(this);
    const from = d.pos || { r: t.r, th: t.th };
    const joining = t.role === 'in' || t.role === 'support' || t.role === 'abstain';
    const delay = joining ? BASE_DELAY + t.gi * STEP + d.k * 9 : d.k * 4;
    const lift = joining && animate ? 0.28 : 0.08;
    d.pos = { r: t.r, th: t.th };
    const tr = sel.transition('move').delay(delay).duration(DUR).ease(d3.easeCubicInOut)
      .attrTween('cx', () => tt => { const r = from.r + (t.r - from.r) * tt + Math.sin(Math.PI * tt) * lift; return xy(r, from.th + (t.th - from.th) * tt)[0]; })
      .attrTween('cy', () => tt => { const r = from.r + (t.r - from.r) * tt + Math.sin(Math.PI * tt) * lift; return xy(r, from.th + (t.th - from.th) * tt)[1]; });
    if (!DUR) { const [x, y] = xy(t.r, t.th); sel.attr('cx', x).attr('cy', y); }
    const st2 = sel.transition('style').delay(delay + DUR * 0.55).duration(DUR ? 350 : 0);
    seatStyle(st2, t.role, t.id);
  });

  // תוויות: רק למי שבממשלה (במצב הפתיחה: לשלושת הגושים). המרווח בין תוויות תלוי בזווית:
  // בראש הקשת הן יושבות זו ליד זו, בצדדים זו מעל זו
  let labels;
  if (!st.mode) {
    const byBase = Object.fromEntries(labelData.map(l => [l.base, l]));
    labels = D.blocs.map((b, i) => ({ key: 'bloc' + i, text: b.name, n: b.seats, th: d3.mean(b.ids, id => byBase[id].th), role: 'neutral', gi: 0 }));
  } else {
    labels = labelData.filter(l => l.role !== 'opp').map(l => ({ key: l.base, text: party(l.id).short, n: seats(l.id), th: l.th, role: l.role, gi: l.gi }));
  }
  labels.sort((a, b) => a.th - b.th);
  const LR = 1.12 * R, width = l => (l.text.length + String(l.n).length + 1) * 6 + 6;
  labels.forEach((l, i) => {
    l.lth = l.th;
    if (!i) return;
    const prev = labels[i - 1], mid = (prev.lth + l.th) / 2;
    const need = (Math.abs(Math.sin(mid)) * (width(prev) + width(l)) / 2 + Math.abs(Math.cos(mid)) * 14) / LR;
    l.lth = Math.max(l.th, prev.lth + need);
  });
  const over = Math.max(0, ...labels.map(l => l.lth - Math.PI));
  if (over) labels.forEach(l => { l.lth -= over * (l.lth / Math.PI); });

  const lab = gLabels.selectAll('g.pl').data(labels, d => d.key);
  lab.exit().transition('lab').duration(DUR ? 250 : 0).style('opacity', 0).remove();
  const ent = lab.enter().append('g').attr('class', 'pl').style('opacity', 0);
  ent.append('text').attr('dy', '0.35em');
  lab.merge(ent).each(function (l) {
    const g = d3.select(this);
    const [x, y] = xy(1.12, l.lth);
    const c = Math.cos(l.lth);
    // ב-RTL "end" הוא הקצה השמאלי של הטקסט, ולכן בצד ימין מעגנים ב-end
    const anchor = c > 0.3 ? 'end' : c < -0.3 ? 'start' : 'middle';
    const delay = l.role === 'neutral' ? 0 : BASE_DELAY + l.gi * STEP + DUR * 0.6;
    g.select('text').attr('text-anchor', anchor)
      .html(`<tspan class="pl-name">${l.text}</tspan><tspan class="pl-seats"> ${l.n}</tspan>`);
    if (!g.attr('transform')) g.attr('transform', `translate(${x},${y})`);
    g.transition('lab').delay(delay).duration(DUR ? 450 : 0).attr('transform', `translate(${x},${y})`).style('opacity', 1);
  });
  $('#opp-line').innerHTML = st.mode ? 'באופוזיציה: ' + groups.filter(g => g.role === 'opp').reverse()
    .map(g => `<span style="--pc:${color(g.id)}"><i class="dot"></i>${party(g.id).short} ${seats(g.id)}</span>`).join(' ') : '';

  // מונה במרכז
  const t = tally(st);
  const counter = $('#count');
  const sub = $('#count-sub');
  if (!st.mode) {
    counter.textContent = '120';
    sub.textContent = '53 · 19 · 48';
    gMarker.classed('hit', false);
    return;
  }
  sub.textContent = `מתוך 61 דרושים`;
  let cum = 0;
  const steps = joinGroups.filter(g => g.role !== 'abstain').map((g, i) => {
    cum += seats(g.id);
    return { at: BASE_DELAY + i * STEP + DUR * 0.7, val: cum };
  });
  if (!animate || reduceMotion) {
    counter.textContent = t.forS;
    gMarker.classed('hit', t.forS >= 61);
    return;
  }
  counter.textContent = '0';
  gMarker.classed('hit', false);
  let shown = 0;
  steps.forEach(s => {
    setTimeout(() => {
      if (token !== animToken) return;
      const from = shown; shown = s.val;
      d3.select(counter).transition().duration(380).tween('text', () => tt => { counter.textContent = Math.round(from + (s.val - from) * tt); });
      if (from < 61 && s.val >= 61) {
        gMarker.classed('hit', true);
        const burst = $('#burst'); burst.classList.remove('go'); void burst.offsetWidth; burst.classList.add('go');
      }
    }, s.at);
  });
  return BASE_DELAY + Math.max(0, joinGroups.length - 1) * STEP + DUR;
}

// ---------- הצבעת אמון ----------
function renderVote(st, delay) {
  const box = $('#vote');
  const t = tally(st);
  if (!st.mode) { box.className = 'vote idle'; box.innerHTML = '<p class="vote-hint">בחרו תרחיש מוכן או בנו קואליציה בעצמכם, ותראו איך היא מתיישבת בכנסת</p>'; return; }
  const pct = v => (v / 120 * 100).toFixed(2) + '%';
  box.className = 'vote';
  box.innerHTML = `
    <div class="vote-bar" aria-hidden="true">
      <span class="vb-for" style="width:0"></span><span class="vb-abs" style="width:0"></span><span class="vb-against" style="width:0"></span>
      <i class="vb-mid"></i>
    </div>
    <div class="vote-nums">
      <span><b class="n-for">${t.forS}</b> בעד</span>
      ${t.abs ? `<span><b>${t.abs}</b> נמנעים</span>` : ''}
      <span><b class="n-against">${t.against}</b> נגד</span>
    </div>
    <div class="verdict ${t.pass ? 'ok' : 'no'}" hidden>${verdictText(st, t)}</div>`;
  const tok = animToken;
  const run = () => {
    if (tok !== animToken) return;
    box.classList.add('shown');
    box.querySelector('.vb-for').style.width = pct(t.forS);
    box.querySelector('.vb-abs').style.width = pct(t.abs);
    box.querySelector('.vb-against').style.width = pct(t.against);
    setTimeout(() => { const v = box.querySelector('.verdict'); if (v) { v.hidden = false; v.classList.add('stamp'); } }, reduceMotion ? 0 : 650);
  };
  if (delay) setTimeout(run, delay); else requestAnimationFrame(run);
}

function verdictText(st, t) {
  if (t.pass) {
    const minority = t.inS < 61;
    return `<span class="v-icon">✓</span><div><b>הממשלה מקבלת אמון</b><small>${minority ? `ממשלת מיעוט: ${t.inS} ח"כים בפנים, השאר ${t.sup ? 'תומכים' : ''}${t.sup && t.abs ? ' או ' : ''}${t.abs ? 'נמנעים' : ''} מבחוץ` : `רוב של ${t.inS} ח"כים בתוך הקואליציה`}</small></div>`;
  }
  if (t.forS === t.against) return `<span class="v-icon">✗</span><div><b>תיקו: ${t.forS} מול ${t.against}</b><small>תיקו לא מספיק. ח"כ אחד שעובר צד היה מכריע</small></div>`;
  const need = Math.ceil((t.against - t.forS + 1) / 2);
  return `<span class="v-icon">✗</span><div><b>אין רוב</b><small>צריך שעוד ${need} ח"כים לפחות יעברו מ"נגד" ל"בעד"</small></div>`;
}

// ---------- בנייה ----------
function renderModes() {
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('on', b.dataset.mode === state.mode);
    b.setAttribute('aria-pressed', b.dataset.mode === state.mode);
  });
}

function renderChips() {
  const wrap = $('#chips');
  const ids = state.mode ? idsFor(state.mode) : D.parties.map(p => p.id);
  wrap.innerHTML = '';
  for (const id of ids) {
    const p = party(id);
    const role = state.roles[id];
    const pm = state.mode && PM[state.mode].party === id;
    const reason = role ? null : blockReason(id);
    const chip = document.createElement('div');
    chip.className = 'chip' + (role ? ' on role-' + role : '') + (reason ? ' blocked' : '') + (pm ? ' pm' : '');
    chip.style.setProperty('--pc', color(id));
    const roles = rolesOf(id);
    chip.innerHTML = `
      <button class="chip-main" type="button" ${pm ? 'disabled' : ''} aria-pressed="${!!role}">
        <span class="sw"></span>
        <span class="chip-txt"><b>${p.name}</b><small>${p.leader}</small></span>
        <span class="chip-n">${p.seats}</span>
        ${reason ? '<span class="lock" aria-hidden="true">🔒</span>' : ''}
        ${pm ? '<span class="pm-tag">מרכיב</span>' : ''}
      </button>
      ${role && roles.length > 1 ? `<div class="roleseg" role="group" aria-label="תפקיד">${roles.map(r => {
        const others = { mode: state.mode, roles: Object.fromEntries(Object.entries(state.roles).filter(([k]) => k !== id)) };
        const why = blockReason(id, others, r);
        return `<button type="button" data-r="${r}" class="${r === role ? 'on' : ''}" ${why ? `aria-disabled="true" data-why="${why.replace(/"/g, '&quot;')}"` : ''}>${ROLE_NAME[r]}</button>`;
      }).join('')}</div>` : ''}`;
    chip.querySelector('.chip-main').addEventListener('click', () => {
      if (!state.mode) { flash('בחרו קודם מי מרכיב את הממשלה'); return; }
      if (reason) { flash(`<b>${p.name}</b>: ${reason}`); return; }
      const roles2 = { ...state.roles };
      if (role) delete roles2[id]; else roles2[id] = rolesOf(id).find(r => !blockReason(id, state, r));
      setState({ mode: state.mode, roles: roles2 }, true);
    });
    chip.querySelectorAll('.roleseg button').forEach(b => b.addEventListener('click', () => {
      if (b.dataset.why) { flash(`<b>${p.name}</b> לא יכולה להיות ${ROLE_NAME[b.dataset.r]}: ${b.dataset.why}`); return; }
      setState({ mode: state.mode, roles: { ...state.roles, [id]: b.dataset.r } }, true);
    }));
    wrap.appendChild(chip);
  }
  renderHint();
}

function renderHint() {
  const el = $('#hint');
  if (!state.mode) { el.innerHTML = ''; return; }
  const t = tally(state);
  if (t.pass) { el.innerHTML = ''; return; }
  // אילו מפלגות זמינות היו מביאות לרוב אם יתווספו?
  const sug = idsFor(state.mode).filter(id => !(id in state.roles) && !blockReason(id)).filter(id => {
    const st2 = { mode: state.mode, roles: { ...state.roles, [id]: rolesOf(id).find(r => !blockReason(id, state, r)) } };
    return tally(st2).pass;
  });
  const avail = idsFor(state.mode).filter(id => !(id in state.roles) && !blockReason(id));
  if (sug.length) el.innerHTML = `מספיקה תוספת של אחת מאלה: ${sug.map(id => `<b style="--pc:${color(id)}">${party(id).name}</b>`).join(' · ')}`;
  else if (avail.length) el.innerHTML = `עוד אין רוב. אפשר להוסיף: ${avail.map(id => `<b style="--pc:${color(id)}">${party(id).name}</b>`).join(' · ')}`;
  else el.innerHTML = 'אין יותר מפלגות שיכולות להצטרף לצירוף הזה. נסו להוציא מפלגה או להחליף מרכיב.';
}

let flashTimer;
function flash(html) {
  const el = $('#flash');
  el.innerHTML = html; el.hidden = false;
  el.classList.remove('show'); void el.offsetWidth; el.classList.add('show');
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { el.hidden = true; }, 5200);
}

// ---------- גלריית תרחישים ----------
function renderGallery() {
  const g = $('#gallery');
  g.innerHTML = '';
  for (const s of D.scenarios) {
    const t = tally(s);
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'scard';
    card.dataset.id = s.id;
    const order = seatOrder(s).filter(x => x.role !== 'opp');
    card.innerHTML = `
      <span class="scard-top"><span class="scard-tag">${s.tag}</span><span class="scard-n ${t.pass ? '' : 'fail'}">${t.forS}</span></span>
      <b class="scard-title">${s.title}</b>
      <span class="scard-pm">${PM[s.mode].name}${(s.mode === 'N' && s.roles.Y) ? ' ⇄ איזנקוט' : ''}${s.mode === 'E' && s.roles.Lx ? ' ⇄ יו"ר הליכוד' : ''}</span>
      <span class="mini" aria-hidden="true">${order.map(x => `<i class="r-${x.role}" style="flex:${seats(x.id)};--pc:${color(x.id)}"></i>`).join('')}<i class="mini-rest" style="flex:${120 - order.reduce((a, x) => a + seats(x.id), 0)}"></i><em></em></span>`;
    card.addEventListener('click', () => {
      setState({ mode: s.mode, roles: { ...s.roles } }, true);
      if (window.innerWidth < 900) $('#stage').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    });
    g.appendChild(card);
  }
}
function markGallery(sc) {
  document.querySelectorAll('.scard').forEach(c => c.classList.toggle('on', sc && c.dataset.id === sc.id));
}

// ---------- ניתוח ----------
function srcLink(k) { const s = D.sources[k]; return s ? `<a href="${s.url}" target="_blank" rel="noopener">${s.label}</a>` : ''; }
function pname(id) { return id === 'CH' ? 'גוש השינוי' : party(id).name; }
function pdot(id) { return id === 'CH' ? '<i class="dot" style="--pc:var(--ink2)"></i>' : `<i class="dot" style="--pc:${color(id)}"></i>`; }
const SEV = { 3: ['קשה', 'crit'], 2: ['בינונית', 'serious'], 1: ['קלה', 'warn'] };

function renderAnalysis() {
  const el = $('#analysis');
  if (!state.mode) {
    el.innerHTML = `<div class="empty-an"><p>כאן יופיע הניתוח של כל ממשלה שתרכיבו: מי מוותר על מה, אילו הבטחות נשברות, כמה זה יציב ומה זה אומר לחיים שלנו.</p></div>`;
    return;
  }
  const t = tally(state);
  const out = outcomes(state);
  const broken = brokenPledges(state, out);
  const stab = stability(state, out, broken);
  const sc = matchScenario(state);
  const conc = concessions(state, out);
  const tens = tensions(state).filter(x => x.t >= 8);
  const pm = PM[state.mode];
  const rot = (state.mode === 'N' && state.roles.Y) ? 'רוטציה: נתניהו ראשון, איזנקוט אחריו (לפי ההצעה שדווחה)' :
    (state.mode === 'E' && state.roles.Lx) ? 'רוטציה בין איזנקוט ליו"ר הליכוד החדש' : '';

  let html = '';
  // כותרת
  html += `<section class="an-card an-head">
    <div class="an-kicker">${t.pass ? (t.inS < 61 ? 'ממשלת מיעוט שעוברת' : 'ממשלה עם רוב') : 'עדיין אין ממשלה'}</div>
    <h2>${sc ? sc.title : 'הקואליציה שבניתם'}</h2>
    <p class="an-pm">ראש הממשלה: <b>${pm.name}</b>${rot ? `<br><span>${rot}</span>` : ''}</p>
    ${sc ? `<p class="lede">${sc.lede}</p>${sc.story.map(p => `<p>${p}</p>`).join('')}
      <h4>מה צריך לקרות כדי שזה יקרה</h4><ul class="needs">${sc.needs.map(n => `<li>${n}</li>`).join('')}</ul>
      <p class="prec"><b>תקדים:</b> ${sc.precedent}</p>` :
      `<p class="lede">${composeLede(t)}</p>`}
    <button type="button" class="share" id="share">🔗 העתיקו קישור לתרחיש הזה</button>
  </section>`;

  if (!t.pass) {
    html += `<section class="an-card"><p class="muted">הניתוח המלא מוצג כשיש לצירוף רוב בהצבעת אמון. בינתיים, הנה מה שכבר אפשר לראות:</p></section>`;
  }

  // יציבות
  html += `<section class="an-card">
    <h3>כמה יציבה ממשלה כזו?</h3>
    <div class="stab">
      <div class="gauge" role="img" aria-label="ציון יציבות ${stab.score} מתוך 100: ${stab.label}">
        <svg viewBox="0 0 120 70"><path class="g-bg" d="M10 62 A50 50 0 0 1 110 62"/><path class="g-fg ${stab.cls}" d="M10 62 A50 50 0 0 1 110 62" pathLength="100" style="stroke-dasharray:0 100" data-v="${stab.score}"/></svg>
        <div class="g-num">${stab.score}</div>
      </div>
      <div class="stab-txt"><b class="st-${stab.cls}">${stab.label}</b><span>תוחלת חיים משוערת: ${stab.life}</span></div>
    </div>
    <ul class="factors">${stab.factors.map(([n, v, why]) => `<li><span class="f-n">${n}</span><span class="f-bar"><i class="${v >= 0 ? 'pos' : 'neg'}" style="width:${Math.min(100, Math.abs(v) * 5)}%"></i></span><span class="f-v">${v > 0 ? '+' : ''}${v}</span><small>${why}</small></li>`).join('')}</ul>
    <p class="note">הציון הוא מודל פשוט שלנו, לא תחזית. הוא מחבר את גודל הרוב, המרחק האידאולוגי בין השותפות, מספר המפלגות שיכולות להפיל את הממשלה לבד, ההבטחות שנשברו, והאם האופוזיציה יכולה בכלל להרכיב חלופה (כדי להחליף ממשלה בלי בחירות צריך 61 ח"כים שמסכימים על ראש ממשלה אחר). בנוסף, כל ממשלה חדשה תצטרך להעביר תקציב 2027 תוך חודשים ספורים, אחרת הכנסת מתפזרת.</p>
  </section>`;

  // הבטחות
  html += `<section class="an-card">
    <h3>הבטחות שנשברות <span class="count">${broken.length}</span></h3>
    ${broken.length ? `<ul class="pledges">${broken.map(({ p, sev }) => `<li class="sev-${SEV[sev][1]}">
      <div class="pl-top">${pdot(p.party)}<b>${pname(p.party)}</b><span class="sevpill ${SEV[sev][1]}">${sev === 3 ? '⚠' : sev === 2 ? '◆' : '•'} הפרה ${SEV[sev][0]}</span></div>
      <div class="pl-text">${p.text.startsWith('"') ? p.text : `"${p.text}"`}</div>
      <small>${p.quote}. מקור: ${srcLink(p.src)}</small></li>`).join('')}</ul>` : '<p class="muted">אף הבטחה מרכזית לא נשברת בצירוף הזה.</p>'}
  </section>`;

  // מי מוותר על מה
  html += `<section class="an-card">
    <h3>מי מקבל מה, ומי מוותר על מה</h3>
    <div class="conc">${conc.map(c => `<div class="cc" style="--pc:${color(c.id)}">
      <div class="cc-h"><i class="dot"></i><b>${party(c.id).name}</b><small>${ROLE_NAME[c.role]} · ${seats(c.id)}</small></div>
      ${c.wins.length ? `<div class="cc-row win"><span>מקבלת</span><ul>${c.wins.map(w => `<li>${w.iss.icon} ${w.want}</li>`).join('')}</ul></div>` : ''}
      ${c.comp.length ? `<div class="cc-row comp"><span>מתפשרת</span><ul>${c.comp.map(w => `<li>${w.iss.icon} ${w.want}</li>`).join('')}</ul></div>` : ''}
      ${c.gives.length ? `<div class="cc-row give"><span>מוותרת</span><ul>${c.gives.map(w => `<li>${w.iss.icon} ${w.want}</li>`).join('')}</ul></div>` : ''}
    </div>`).join('')}</div>
  </section>`;

  // מה זה אומר עלינו
  html += `<section class="an-card">
    <h3>מה זה אומר עלינו</h3>
    <p class="muted small">לכל נושא: איפה עומדת כל מפלגה בממשלה (עיגול לפי גודל), והיכן צפויה המדיניות לנחות (◆). ככל שמפלגה רחוקה מה-◆, היא מוותרת יותר.</p>
    <div class="issues">${D.issues.map(iss => issueRow(iss, out[iss.id])).join('')}</div>
  </section>`;

  // קווי שבר
  html += `<section class="an-card">
    <h3>קווי השבר בתוך הממשלה</h3>
    ${tens.length ? `<div class="net-wrap"><svg id="net" viewBox="0 0 320 250" role="img" aria-label="מפת מתחים בין השותפות"></svg></div>
    <ul class="tens">${tens.slice(0, 5).map(x => `<li>${pdot(x.a)}<b>${party(x.a).short}</b> ↔ ${pdot(x.b)}<b>${party(x.b).short}</b><small>${x.issues.slice(0, 3).join(', ') || 'פערים מצטברים'}</small></li>`).join('')}</ul>` :
      '<p class="muted">אין מתחים חריפים בין השותפות.</p>'}
  </section>`;

  el.innerHTML = html;
  requestAnimationFrame(() => {
    const g = el.querySelector('.g-fg');
    if (g) g.style.strokeDasharray = `${g.dataset.v} 100`;
    el.querySelectorAll('.iss-out').forEach(d => { d.style.right = d.dataset.x + '%'; });
  });
  if (tens.length) drawNet(tens);
  $('#share').addEventListener('click', () => {
    const url = location.href;
    (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject()).then(() => flash('הקישור הועתק')).catch(() => flash(url));
  });
}

function composeLede(t) {
  const ins = active(state).filter(id => state.roles[id] === 'in').map(id => party(id).name);
  const sup = active(state).filter(id => state.roles[id] === 'support').map(id => party(id).name);
  const abs = active(state).filter(id => state.roles[id] === 'abstain').map(id => party(id).name);
  let s = `בתוך הקואליציה: ${ins.join(', ')} (${t.inS} מנדטים).`;
  if (sup.length) s += ` תומכות מבחוץ: ${sup.join(', ')}.`;
  if (abs.length) s += ` נמנעות: ${abs.join(', ')}.`;
  return s;
}

function issueRow(iss, o) {
  // -2 בצד ימין, +2 בצד שמאל (במסך מימין לשמאל)
  const xp = v => (v + 2) / 4 * 100; // מרחק מהקצה הימני, באחוזים
  const groups = {};
  for (const id of active(state)) {
    const pos = iss.pos[id]; if (!pos || pos[1] === 0) continue;
    (groups[pos[0]] = groups[pos[0]] || []).push(id);
  }
  const dots = Object.entries(groups).map(([v, ids]) => ids.map((id, i) => {
    const r = 4 + Math.sqrt(seats(id)) * 1.1;
    return `<i class="iss-dot" title="${party(id).name}: ${iss.pos[id][2]}" style="right:calc(${xp(+v)}% - ${r}px);top:${6 + i * 17 + (8 - r)}px;width:${r * 2}px;height:${r * 2}px;--pc:${color(id)};opacity:${state.roles[id] === 'in' ? 1 : 0.55}"></i>`;
  }).join('')).join('');
  const maxStack = Math.max(1, ...Object.values(groups).map(g => g.length));
  return `<div class="iss">
    <div class="iss-h"><span>${iss.icon}</span><b>${iss.name}</b></div>
    <div class="iss-axis" style="height:${34 + maxStack * 17}px">
      <span class="pole r">${iss.minus}</span><span class="pole l">${iss.plus}</span>
      <div class="iss-track">${dots}<i class="iss-out" data-x="${xp(o)}" style="right:50%"><em>◆</em></i></div>
    </div>
    <p class="iss-txt">${iss.outcomes[bucket(o)]}</p>
  </div>`;
}

function drawNet(tens) {
  const net = d3.select('#net');
  const ids = Object.entries(state.roles).filter(([, r]) => r !== 'abstain').map(([id]) => id);
  const cx = 160, cy = 125, rr = 88;
  const pos = {};
  ids.forEach((id, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / ids.length; pos[id] = [cx + rr * Math.cos(a), cy + rr * Math.sin(a)]; });
  const maxT = d3.max(tens, d => d.t) || 1;
  net.selectAll('line').data(tens).enter().append('line')
    .attr('x1', d => pos[d.a][0]).attr('y1', d => pos[d.a][1]).attr('x2', d => pos[d.a][0]).attr('y2', d => pos[d.a][1])
    .attr('class', 'net-edge').attr('stroke-width', d => 1 + 6 * d.t / maxT).attr('stroke-opacity', d => 0.25 + 0.6 * d.t / maxT)
    .transition().delay((d, i) => 200 + i * 80).duration(reduceMotion ? 0 : 600)
    .attr('x2', d => pos[d.b][0]).attr('y2', d => pos[d.b][1]);
  net.selectAll('line').append('title').text(d => `${party(d.a).name} ↔ ${party(d.b).name}: ${d.issues.join(', ')}`);
  const nodes = net.selectAll('g.nn').data(ids).enter().append('g').attr('class', 'nn').attr('transform', id => `translate(${pos[id]})`);
  nodes.append('circle').attr('r', id => 6 + Math.sqrt(seats(id)) * 2.2).attr('fill', id => color(id));
  nodes.append('text').attr('dy', id => (pos[id][1] < cy ? -1 : 1) * (12 + Math.sqrt(seats(id)) * 2.2) + (pos[id][1] < cy ? 0 : 6)).attr('text-anchor', 'middle').text(id => party(id).short);
}

// ---------- תוצאות האמת ----------
function renderResults() {
  const bar = $('#resbar');
  const order = ['L', 'O', 'RZ', 'S', 'U', 'R', 'J', 'YB', 'D', 'B', 'Y'];
  bar.innerHTML = order.map(id => `<i style="flex:${seats(id)};--pc:${color(id)}" title="${party(id).name}: ${seats(id)}"><span>${seats(id)}</span></i>`).join('');
  const tb = $('#restable tbody');
  tb.innerHTML = D.parties.map(p => `<tr><td><i class="dot" style="--pc:${color(p.id)}"></i>${p.name}</td><td>${p.leader}</td><td class="num">${p.votes.toLocaleString('he-IL')}</td><td class="num">${p.pct.toFixed(2)}%</td><td class="num"><b>${p.seats}</b></td><td class="about">${p.about}</td></tr>`).join('');
  $('#res-note').textContent = D.election.note;
  $('#timeline').innerHTML = D.timeline.map(t => `<li><b>${t.date}</b><span>${t.text}</span></li>`).join('');
  $('#sources').innerHTML = Object.values(D.sources).map(s => `<li><a href="${s.url}" target="_blank" rel="noopener">${s.label}</a></li>`).join('');
}

// ---------- קישור לשיתוף ----------
function encode(st) {
  if (!st.mode) return '';
  const r = Object.entries(st.roles).map(([id, role]) => id + (role === 'in' ? '' : role === 'support' ? '.s' : '.a')).join(',');
  return `#${st.mode}:${r}`;
}
function decode(h) {
  const m = /^#([NE]):(.*)$/.exec(decodeURIComponent(h || ''));
  if (!m) return null;
  const roles = {};
  for (const part of m[2].split(',').filter(Boolean)) {
    const [id, r] = part.split('.');
    if (!party(id)) continue;
    roles[id] = r === 's' ? 'support' : r === 'a' ? 'abstain' : 'in';
  }
  const st = { mode: m[1], roles: {} };
  st.roles[PM[st.mode].party] = 'in';
  // הוספה בסדר, עם בדיקת חוקיות, כדי שקישור ידני לא יעקוף את הכללים
  for (const [id, r] of Object.entries(roles)) {
    if (id in st.roles) continue;
    const want = rolesOf(id).includes(r) ? r : rolesOf(id)[0];
    if (blockReason(id, st, want)) continue;
    st.roles[id] = want;
  }
  return st;
}

// ---------- הכל ביחד ----------
function setState(st, animate) {
  state = st;
  renderModes();
  renderChips();
  const total = layoutHemicycle(state, animate);
  renderVote(state, animate && !reduceMotion ? (total || 0) + 150 : 0);
  renderAnalysis();
  markGallery(matchScenario(state));
  $('#pm-line').innerHTML = state.mode ? `${PM[state.mode].name} מרכיב` : 'תוצאות הבחירות';
  const h = encode(state);
  if (h !== location.hash) history.replaceState(null, '', h || location.pathname + location.search);
}

function init() {
  renderResults();
  initHemicycle();
  renderGallery();
  document.querySelectorAll('.mode-btn').forEach(b => b.addEventListener('click', () => {
    const mode = b.dataset.mode;
    if (state.mode === mode) return;
    setState({ mode, roles: { [PM[mode].party]: 'in' } }, true);
  }));
  $('#reset').addEventListener('click', () => setState({ mode: null, roles: {} }, true));
  $('#theme').addEventListener('click', () => {
    const cur = isDark();
    document.documentElement.dataset.theme = cur ? 'light' : 'dark';
    try { localStorage.setItem('coalition-theme', document.documentElement.dataset.theme); } catch (e) {}
    setState(state, false); renderResults(); renderGallery(); markGallery(matchScenario(state));
  });
  const st = decode(location.hash);
  setState(st || { mode: null, roles: {} }, false);
  if (st) setTimeout(() => setState(st, true), 50);
  window.addEventListener('hashchange', () => { const s = decode(location.hash); if (s && encode(s) !== encode(state)) setState(s, true); });
}

try { const t = localStorage.getItem('coalition-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
document.addEventListener('DOMContentLoaded', init);
})();
