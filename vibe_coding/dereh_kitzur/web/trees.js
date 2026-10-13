/* The tree explorer - "סייר העצים".
 *
 * The survey layer (build_trees.py → data/trees.json, registered in layers.js)
 * is almost two thousand pins, and a map of two thousand green dots answers no
 * question. Every tree carries a dozen things the survey recorded about it, and
 * this module turns those into cuts: colour the map by any one of them, filter
 * by any combination, read the counts, and jump to a handful of views worth
 * seeing ("the ancient trees", "hollow trees next to a pavement").
 *
 * It lives in the side panel as a third view next to the list and the detail
 * pane, not in a sheet over the map: the point of a filter here is to watch the
 * map change while you set it.
 *
 * What it changes is only presentation. Each tree's `color` and `group` follow
 * the dimension it is coloured by, and `hidden` its place in the filter; the
 * survey's own values stay in `tree` untouched. layers.js skips hidden points
 * on the map, in the key and in the list.
 *
 * The state is in the address bar as `tr=` so a cut can be sent to somebody:
 *     tr=c.age_health.חולה*חלול_near.מדרכה_q.קנקר_o
 * The separators are the few characters a query string leaves unescaped, so
 * the link stays readable (app.js rewrites the whole query on every move).
 * c = colour dimension, then one `dim.values` per filter, q = free text,
 * o = the explorer is open.
 */
'use strict';

const Trees = (() => {

  const HEALTH = ['בריא', 'חולה', 'מנוון', 'חלול', 'לא תועד'];
  const HEALTH_C = { 'בריא': '#2e7d32', 'חולה': '#ef6c00', 'מנוון': '#ad1457', 'חלול': '#8d6e63', 'לא תועד': '#9e9e9e' };
  // Categorical, in a fixed order, and grey for the rest - never a ninth hue.
  const CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'];
  const OTHER = '#9e9e9e';
  // Ordered things go light to dark in one hue. Light enough at the near end
  // to tell apart, dark enough at the far end to still read as green on the
  // satellite basemap.
  const SEQ = ['#c5e1a5', '#9ccc65', '#66bb6a', '#2e7d32', '#1b5e20', '#0b3d12'];

  const bucket = (edges, labels) => (v) => {
    if (v == null) return 'לא נמדד';
    for (let i = 0; i < edges.length; i++) if (v < edges[i]) return labels[i];
    return labels[labels.length - 1];
  };
  const MONTHS = ['ינו׳', 'פבר׳', 'מרץ', 'אפר׳', 'מאי', 'יוני', 'יולי', 'אוג׳', 'ספט׳', 'אוק׳', 'נוב׳', 'דצמ׳'];

  /* Every cut. `of` gives a tree's value - or values, for the ones a tree can
   * be several of at once (it can need pruning and support, and stand by a
   * pavement and a road). `order` fixes the order and the colours of ordered
   * cuts; the rest are ordered by size. */
  const DIMS = [
    { id: 'health', name: 'מצב בריאות', of: (p) => p.tree.health || 'לא תועד', order: HEALTH, colors: HEALTH_C },
    { id: 'species', name: 'מין', of: (p) => p.name, top: 7 },
    { id: 'age', name: 'גיל', of: (p) => p.tree.age || 'לא תועד',
      order: ['צעיר', 'מתבגר', 'בוגר', 'ותיק', 'עתיק', 'לא תועד'], seq: true },
    { id: 'height', name: 'גובה', unit: true,
      of: (p) => bucket([5, 10, 15, 20], ['עד 5 מ׳', '5–10 מ׳', '10–15 מ׳', '15–20 מ׳', 'מעל 20 מ׳'])(p.tree.h),
      order: ['עד 5 מ׳', '5–10 מ׳', '10–15 מ׳', '15–20 מ׳', 'מעל 20 מ׳', 'לא נמדד'], seq: true },
    { id: 'trunk', name: 'קוטר הגזע', unit: true,
      of: (p) => bucket([20, 40, 60, 100], ['עד 20 ס״מ', '20–40 ס״מ', '40–60 ס״מ', '60–100 ס״מ', 'מטר ומעלה'])(p.tree.d),
      order: ['עד 20 ס״מ', '20–40 ס״מ', '40–60 ס״מ', '60–100 ס״מ', 'מטר ומעלה', 'לא נמדד'], seq: true },
    { id: 'care', name: 'טיפול נדרש', multi: true,
      of: (p) => ((p.tree.needs || []).length ? p.tree.needs : ['לא צוין טיפול']),
      order: ['גיזום', 'תמיכה', 'קשירה', 'השקיה', 'חיפוי', 'לא צוין טיפול'] },
    { id: 'follow', name: 'מתי לחזור לבדוק', of: (p) => p.tree.follow || 'לא צוין',
      order: ['בעוד חודש', 'בעוד חצי שנה', 'בעוד שנה', 'בעוד שנתיים', 'בעוד חמש שנים', 'לא צוין'],
      colors: { 'בעוד חודש': '#c62828', 'בעוד חצי שנה': '#ef6c00', 'בעוד שנה': '#f9a825',
        'בעוד שנתיים': '#7cb342', 'בעוד חמש שנים': '#2e7d32', 'לא צוין': '#9e9e9e' } },
    { id: 'near', name: 'צמוד ל', multi: true,
      of: (p) => ((p.tree.near || []).length ? p.tree.near : ['שום דבר מאלה']),
      order: ['מדרכה', 'כביש', 'חניה', 'חומה', 'ביוב', 'שום דבר מאלה'] },
    { id: 'record', name: 'סוג הרשומה',
      of: (p) => (p.tree.many ? 'שדרה או חורשה' : 'עץ בודד'), order: ['עץ בודד', 'שדרה או חורשה'],
      colors: { 'עץ בודד': '#2e7d32', 'שדרה או חורשה': '#4a3aa7' } },
    { id: 'month', name: 'מתי נסקר', of: (p) => monthOf(p.tree.date), seq: true, chrono: true }
  ];
  const dim = (id) => DIMS.find((d) => d.id === id);
  function monthOf(iso) {
    if (!iso) return 'לא ידוע';
    const [y, m] = iso.split('-');
    return `${MONTHS[+m - 1]} ${y}`;
  }
  const valuesOf = (d, p) => [].concat(d.of(p));

  /* The picture of each tree on the map: its shape is the way the species
   * grows, its colour whatever the explorer colours by (health, unless
   * somebody picks otherwise). Ori, 13/10/2026: "the shape the kind of tree,
   * the colour its state". The forms themselves are drawn in layers.js; this
   * says which species grows which way, by a word in its Hebrew name, first
   * match wins. 12 species are 83 % of the survey, and each of those has a
   * form of its own or its family's. An avenue or a grove is three of them. */
  const FORMS = [
    ['cypress', 'ברוש', (n) => n === 'ברוש מצוי'],
    ['palm', 'דקל', (n) => /דקל|תמר|וושינגטוניה/.test(n)],
    ['conifer', 'מחטני אחר', (n) => /ברוש|תויה|אראוקריה|קזוארינה/.test(n)],
    ['grevillea', 'גרווילאה', (n) => n.includes('גרווילאה')],
    ['oak', 'אלון', (n) => n.startsWith('אלון')],
    ['eucalyptus', 'איקליפטוס', (n) => n.includes('איקליפטוס')],
    ['weeping', 'שיטה', (n) => n.startsWith('שיטה') || n.startsWith('שיטת')],
    ['vase', 'סיגלון', (n) => n.includes('סיגלון')],
    ['ficus', 'פיקוס', (n) => n.startsWith('פיקוס')],
    ['umbrella', 'צמרת שטוחה: מכנף, צאלון, אורן הגלעין',
      (n) => /מכנף|צאלון|אלביציה|אורן הגלעין/.test(n)],
    ['pine', 'אורן', (n) => n.startsWith('אורן')],
    ['olive', 'זית', (n) => n.startsWith('זית')],
    ['round', 'עץ רחב-עלים אחר', () => true]
  ];
  const formOf = (p) => FORMS.find(([, , is]) => is(p.name))[0];
  const iconOf = (p) => `tree-${formOf(p)}-${p.tree.many ? 'g' : 's'}-${p.color.slice(1).toLowerCase()}`;

  /** The shapes the survey actually uses, most common first, and the grove. */
  function shapeKey() {
    const n = new Map();
    trees().forEach((p) => { const f = formOf(p); n.set(f, (n.get(f) || 0) + 1); });
    const rows = FORMS.filter(([f]) => n.get(f)).sort((a, b) => (a[0] === 'round') - (b[0] === 'round')
      || n.get(b[0]) - n.get(a[0])).map(([f, name]) => [f, name, false]);
    return [...rows, ['round', 'שדרה או חורשה', true]];
  }

  /* Views worth a click. Each is a whole state, not a tweak of the current one,
   * so that pressing it twice gives the same map. */
  const PRESETS = [
    { name: 'העצים העתיקים', c: 'species', f: { age: ['עתיק'] } },
    { name: 'אלוני התבור', c: 'health', f: { species: ['אלון התבור'] } },
    { name: 'חלולים ליד מדרכה או כביש', c: 'near', f: { health: ['חלול'], near: ['מדרכה', 'כביש'] } },
    { name: 'הענקים: מעל 20 מטר', c: 'species', f: { height: ['מעל 20 מ׳'] } },
    { name: 'לבדוק בעוד חודש', c: 'health', f: { follow: ['בעוד חודש'] } },
    { name: 'צריכים גיזום', c: 'health', f: { care: ['גיזום'] } },
    { name: 'שדרות וחורשות', c: 'species', f: { record: ['שדרה או חורשה'] } },
    { name: 'ההערה מזכירה כריתה', c: 'health', f: {}, q: 'כרית' }
  ];

  const state = { c: 'health', f: {}, q: '', open: false };
  let resumeExplorer = false;

  const layer = () => Layers.byId(Layers.TREES_ID);
  const trees = () => (layer() ? layer().waypoints : []);

  function passes(p, skip) {
    for (const [id, vals] of Object.entries(state.f)) {
      if (id === skip || !vals.length) continue;
      const d = dim(id);
      if (!d || !valuesOf(d, p).some((v) => vals.includes(v))) return false;
    }
    if (state.q) {
      const q = state.q;
      if (!(p.name.includes(q) || (p.note || '').includes(q))) return false;
    }
    return true;
  }

  /** The values of one cut, in display order, with their counts among the trees
   *  that pass every *other* filter - so a row's number is what you would get
   *  by ticking it now. */
  function facet(d) {
    const counts = new Map();
    trees().forEach((p) => {
      if (!passes(p, d.id)) return;
      valuesOf(d, p).forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
    });
    // Ticked values stay listed even at zero, or there is no way to untick them.
    (state.f[d.id] || []).forEach((v) => { if (!counts.has(v)) counts.set(v, 0); });
    let vals = [...counts.keys()];
    if (d.order) vals.sort((a, b) => idx(d.order, a) - idx(d.order, b));
    else if (d.chrono) vals.sort((a, b) => chronoKey(a) - chronoKey(b));
    else vals.sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b, 'he'));
    return vals.map((v) => ({ v, n: counts.get(v) }));
  }
  const idx = (arr, v) => { const i = arr.indexOf(v); return i < 0 ? 999 : i; };
  function chronoKey(label) {
    const m = label.split(' ');
    return m.length === 2 ? +m[1] * 12 + MONTHS.indexOf(m[0]) : 1e9;
  }

  /** Colour for every value of the colouring cut. Species is coloured by how
   *  common it is overall, so the seven colours stay put while you filter. */
  let colorCache = { key: null, map: null };
  function colors(d) {
    if (colorCache.key === d.id) return colorCache.map;
    const m = new Map();
    if (d.colors) Object.entries(d.colors).forEach(([k, c]) => m.set(k, c));
    else if (d.seq) {
      const all = d.order ? d.order.filter((v) => !/^לא /.test(v))
        : [...new Set(trees().map((p) => d.of(p)))].filter((v) => v !== 'לא ידוע').sort((a, b) => chronoKey(a) - chronoKey(b));
      all.forEach((v, i) => m.set(v, SEQ[Math.round((i / Math.max(1, all.length - 1)) * (SEQ.length - 1))]));
    } else {
      const counts = new Map();
      trees().forEach((p) => valuesOf(d, p).forEach((v) => counts.set(v, (counts.get(v) || 0) + 1)));
      const ranked = [...counts.keys()].filter((v) => !/^(לא |שום )/.test(v))
        .sort((a, b) => counts.get(b) - counts.get(a));
      ranked.slice(0, d.top || CAT.length).forEach((v, i) => m.set(v, CAT[i]));
    }
    colorCache = { key: d.id, map: m };
    return m;
  }
  const colorFor = (d, v) => colors(d).get(v) || OTHER;

  /** Push the state onto the trees and redraw. */
  function apply() {
    const l = layer();
    if (!l) return;
    const d = dim(state.c) || dim('health');
    const cm = colors(d);
    l.waypoints.forEach((p) => {
      const vs = valuesOf(d, p);
      // A tree that is several things at once takes the first of them that the
      // filter is asking about, so a map filtered to "pruning" is coloured as
      // pruning and not as whatever need came first in the survey's columns.
      const want = state.f[d.id] || [];
      const v = vs.find((x) => want.includes(x)) || vs[0];
      p.group = cm.has(v) ? v : (d.id === 'species' ? 'מינים אחרים' : v);
      p.color = cm.get(v) || OTHER;
      p.icon = iconOf(p);
      p.hidden = !passes(p);
    });
    // The key is built from the layer's groups. Ordered as the cut is, and only
    // the ones somebody can see - layers.js counts and drops the empty ones.
    const order = facet(d).map((r) => r.v).filter((v) => cm.has(v));
    l.groups = [...order.map((v) => ({ name: v, color: cm.get(v) })),
      ...(d.id === 'species' ? [{ name: 'מינים אחרים', color: OTHER }] : []),
      ...[...cm.keys()].filter((v) => !order.includes(v)).map((v) => ({ name: v, color: cm.get(v) }))];
    l.filtered = Object.values(state.f).some((v) => v.length) || !!state.q;
    Layers.restyle(l.id);
    writeUrl();
    if (state.open) render();
  }

  /* ---------- the panel ---------- */

  function view() {
    let v = document.getElementById('trees-view');
    if (v) return v;
    v = document.createElement('div');
    v.id = 'trees-view';
    v.hidden = true;
    v.innerHTML = `
      <header class="detail-head">
        <button class="back" data-tx="close" aria-label="חזרה לרשימה">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6L4 12l6 6 1.4-1.4L7.8 13H20v-2H7.8l3.6-3.6z"/></svg>
          <span>לרשימה</span>
        </button>
        <span class="tip">לחיצה על עץ במפה פותחת את הכרטיס שלו</span>
      </header>
      <div id="tx" class="detail tx"></div>`;
    document.getElementById('panel').appendChild(v);
    v.addEventListener('click', onClick);
    v.addEventListener('input', (e) => {
      if (e.target.id !== 'tx-q') return;
      clearTimeout(onQ.t);
      onQ.t = setTimeout(() => { state.q = e.target.value.trim(); apply(); }, 250);
    });
    return v;
  }
  const onQ = {};

  function open() {
    if (!layer()) return;
    Layers.turnOn(Layers.TREES_ID);
    state.open = true;
    resumeExplorer = false;
    document.getElementById('detail-view').hidden = true;
    document.getElementById('list-view').hidden = true;
    view().hidden = false;
    apply();
  }

  function close() {
    state.open = false;
    resumeExplorer = false;
    const v = document.getElementById('trees-view');
    if (v) v.hidden = true;
    document.getElementById('list-view').hidden = false;
    writeUrl();
    if (typeof renderList === 'function') renderList();
  }

  /** app.js calls this when a detail pane opens: the explorer steps aside, and
   *  the back button of that pane brings it back rather than the list. */
  function onDetail() {
    const v = document.getElementById('trees-view');
    if (v && !v.hidden) {
      v.hidden = true;
      resumeExplorer = true;
    }
  }
  /** And this from the back button. True when it took the panel back. */
  function resume() {
    if (!resumeExplorer || !state.open) return false;
    resumeExplorer = false;
    document.getElementById('list-view').hidden = true;
    view().hidden = false;
    render();
    return true;
  }

  const fmt = (n) => n.toLocaleString('he-IL');

  function render() {
    const box = document.getElementById('tx');
    if (!box) return;
    const all = trees();
    const shown = all.filter((p) => !p.hidden);
    const trunkSum = shown.reduce((s, p) => s + (p.tree.many || 1), 0);
    const d = dim(state.c) || dim('health');
    const rows = facet(d);
    const max = Math.max(1, ...rows.map((r) => r.n));
    const ticked = state.f[d.id] || [];
    const filters = Object.entries(state.f).filter(([, v]) => v.length);

    box.innerHTML = `
      <h2>סייר העצים</h2>
      <p class="tx-warn">המצב כפי שנמדד בסקר, אוקטובר 2018 עד יולי 2019. חלק מהעצים
        נכרתו מאז, ומצבם של אחרים השתנה.</p>
      <div class="tx-count">
        <span><b>${fmt(shown.length)}</b> ${shown.length === all.length ? 'רשומות' : `מתוך ${fmt(all.length)} רשומות`}
          ${trunkSum > shown.length ? `<span class="tx-sub">(כ-${fmt(trunkSum)} עצים, כי חלק מהרשומות הן שדרה או חורשה)</span>` : ''}</span>
        <span class="tx-acts">
          ${shown.length ? '<button class="tx-btn" data-tx="fit">התמקד במפה</button>' : ''}
          ${filters.length || state.q ? '<button class="tx-btn" data-tx="reset">ניקוי הסינון</button>' : ''}
        </span>
      </div>
      ${filters.length || state.q ? `<div class="tx-active">${filters.map(([id, vs]) =>
        `<span class="tx-tag">${escapeHtml(dim(id).name)}: ${escapeHtml(vs.join(' או '))}
          <button data-tx="drop" data-d="${id}" aria-label="הסרת הסינון">&times;</button></span>`).join('')}${
        state.q ? `<span class="tx-tag">מכיל "${escapeHtml(state.q)}"
          <button data-tx="dropq" aria-label="הסרת החיפוש">&times;</button></span>` : ''}</div>` : ''}

      <h3>מעניין לראות</h3>
      <div class="tx-presets">${PRESETS.map((pr, i) => {
        const n = countFor(pr);
        return n ? `<button class="tx-chip${isPreset(pr) ? ' on' : ''}" data-tx="preset" data-i="${i}">
          ${escapeHtml(pr.name)} <span class="tx-n">${fmt(n)}</span></button>` : '';
      }).join('')}</div>

      <h3>צבע לפי</h3>
      <div class="tx-dims" role="group" aria-label="צבע לפי">${DIMS.map((x) =>
        `<button class="tx-chip${x.id === d.id ? ' on' : ''}" data-tx="color" data-d="${x.id}">${escapeHtml(x.name)}</button>`).join('')}</div>

      <div class="tx-shapes" aria-label="מה אומרת צורת העץ">
        <span class="tx-shapes-t">הצורה במפה היא סוג העץ, הצבע הוא מה שבחרתם למעלה:</span>
        ${shapeKey().map(([f, n, g]) => `<span class="tx-form"><span class="tx-shape" data-form="${f}"${g ? ' data-grove="1"' : ''}></span><span>${escapeHtml(n)}</span></span>`).join('')}
      </div>
      <p class="tx-hint">הצבע לפי מה שבחרתם למעלה. לחיצה על שורה מסננת לפיה. אפשר לבחור כמה.</p>
      <ul class="tx-rows">${rows.map((r) => {
        const on = ticked.includes(r.v);
        return `<li><button class="tx-row${on ? ' on' : ''}${r.n ? '' : ' empty'}" data-tx="tick" data-d="${d.id}" data-v="${escapeHtml(r.v)}"
            aria-pressed="${on}">
          <span class="lg-dot" style="--c:${colorFor(d, r.v)}"></span>
          <span class="tx-nm">${escapeHtml(r.v)}</span>
          <span class="tx-bar"><i style="width:${(r.n / max) * 100}%;background:${colorFor(d, r.v)}"></i></span>
          <span class="tx-n">${fmt(r.n)}</span>
        </button></li>`;
      }).join('')}</ul>

      <h3>חיפוש בהערות ובמינים</h3>
      <input id="tx-q" class="tx-q" type="search" value="${escapeHtml(state.q)}"
        placeholder="למשל: קנקר, תיכון, שדרה, כריתה" autocomplete="off" aria-label="חיפוש בהערות">

      <h3>סינון לפי חתכים נוספים</h3>
      ${DIMS.filter((x) => x.id !== d.id).map((x) => {
        const fr = facet(x);
        const sel = state.f[x.id] || [];
        const list = x.id === 'species' && fr.length > 14 && !sel.length ? fr.slice(0, 14) : fr;
        return `<details class="tx-more"${sel.length ? ' open' : ''}>
          <summary>${escapeHtml(x.name)}${sel.length ? ` <span class="tx-n">${sel.length} נבחרו</span>` : ''}</summary>
          <div class="tx-chips">${list.map((r) => `<button class="tx-chip small${sel.includes(r.v) ? ' on' : ''}"
            data-tx="tick" data-d="${x.id}" data-v="${escapeHtml(r.v)}" ${r.n || sel.includes(r.v) ? '' : 'disabled'}>
            ${escapeHtml(r.v)} <span class="tx-n">${fmt(r.n)}</span></button>`).join('')}
            ${list.length < fr.length ? `<span class="tx-sub">ועוד ${fr.length - list.length} מינים: חפשו אותם בחיפוש למעלה</span>` : ''}</div>
        </details>`;
      }).join('')}

      <a class="src" href="${escapeHtml((layer() || {}).waypoints[0].url)}" target="_blank" rel="noopener">
        סקר העצים המלא בפרדספדיה ↗</a>`;
    box.querySelectorAll('.tx-shape').forEach((el) => {
      el.appendChild(Layers.treeCanvas(el.dataset.form, '#4f8a4b', 0.7, !!el.dataset.grove));
    });
  }

  function countFor(pr) {
    const keep = { f: state.f, q: state.q };
    state.f = pr.f; state.q = pr.q || '';
    const n = trees().filter((p) => passes(p)).length;
    state.f = keep.f; state.q = keep.q;
    return n;
  }
  function isPreset(pr) {
    const norm = (f) => JSON.stringify(Object.entries(f).filter(([, v]) => v.length).sort());
    return norm(pr.f) === norm(state.f) && (pr.q || '') === state.q && pr.c === state.c;
  }

  function onClick(e) {
    const b = e.target.closest('[data-tx]');
    if (!b) return;
    const act = b.dataset.tx;
    if (act === 'close') { close(); return; }
    if (act === 'fit') { fit(); return; }
    if (act === 'reset') { state.f = {}; state.q = ''; }
    if (act === 'color') { state.c = b.dataset.d; colorCache.key = null; }
    if (act === 'drop') delete state.f[b.dataset.d];
    if (act === 'dropq') state.q = '';
    if (act === 'tick') {
      const cur = new Set(state.f[b.dataset.d] || []);
      cur.has(b.dataset.v) ? cur.delete(b.dataset.v) : cur.add(b.dataset.v);
      state.f[b.dataset.d] = [...cur];
      if (!cur.size) delete state.f[b.dataset.d];
    }
    if (act === 'preset') {
      const pr = PRESETS[+b.dataset.i];
      state.f = JSON.parse(JSON.stringify(pr.f));
      state.q = pr.q || '';
      state.c = pr.c;
      colorCache.key = null;
      apply();
      fit();
      return;
    }
    apply();
  }

  function fit() {
    const shown = trees().filter((p) => !p.hidden && p.lat != null);
    if (!shown.length || typeof map === 'undefined' || !map) return;
    let w = 180, s = 90, ea = -180, n = -90;
    shown.forEach((p) => { w = Math.min(w, p.lng); ea = Math.max(ea, p.lng); s = Math.min(s, p.lat); n = Math.max(n, p.lat); });
    if (shown.length === 1) map.flyTo({ center: [w, s], zoom: 18, duration: 900 });
    else map.fitBounds([[w, s], [ea, n]], { padding: 50, maxZoom: 17.5, duration: 900 });
  }

  /* ---------- one tree's card ----------
   * Called by app.js inside the detail pane of a place that has `tree`. */
  function detailBlock(it) {
    const t = it.tree;
    const same = trees().filter((p) => p.name === it.name);
    const rank = (key) => {
      const mine = t[key];
      const vals = same.map((p) => p.tree[key]).filter((v) => v != null);
      if (mine == null || vals.length < 8) return '';
      const below = vals.filter((v) => v < mine).length;
      const pctl = Math.round((below / vals.length) * 100);
      const word = key === 'h' ? ['גבוה', 'מהגבוהים'] : ['עבה', 'מהעבים'];
      if (pctl >= 75) return ` · ${pctl >= 95 ? word[1] : `${word[0]} מ-${pctl}%`} מבין ${fmt(vals.length)} עצי ${escapeHtml(it.name)} בסקר`;
      return '';
    };
    const row = (k, v) => (v == null || v === '' ? '' : `<tr><th>${k}</th><td>${v}</td></tr>`);
    const date = t.date ? new Date(t.date).toLocaleDateString('he-IL') : null;
    return `
      ${t.many ? `<p class="tx-many">הרשומה הזו מייצגת קבוצה של כ-${fmt(t.many)} עצים, שדרה או
        חורשה, והנקודה היא אחד מהם. הפרטים בהערה למעלה.</p>` : ''}
      <table class="tx-facts">
        ${row('מצב בריאות', `<span class="lg-dot" style="--c:${HEALTH_C[t.health] || OTHER}"></span> ${escapeHtml(t.health)}`)}
        ${row('גיל', t.age && escapeHtml(t.age))}
        ${row('גובה', t.h != null ? `${t.h} מ׳<span class="tx-sub">${rank('h')}</span>` : null)}
        ${row('קוטר הגזע', t.d != null ? `${t.d} ס״מ<span class="tx-sub">${rank('d')}</span>` : null)}
        ${row('היקף הנוף', t.c != null ? `${t.c} מ׳` : null)}
        ${row('נדרש', (t.needs || []).length ? escapeHtml(t.needs.join(', ')) : null)}
        ${row('הגנה נדרשת', t.protect && escapeHtml(t.protect))}
        ${row('צמוד ל', (t.near || []).length ? escapeHtml(t.near.join(', ')) : null)}
        ${row('לחזור לבדוק', t.follow && escapeHtml(t.follow))}
        ${row('מספר בסקר', t.num && escapeHtml(t.num))}
        ${row('נמדד', date)}
      </table>
      <p class="tx-warn small">כך נמצא העץ ${date ? `ב-${date}` : 'בסקר'}. ייתכן שמצבו השתנה מאז, או שנכרת.</p>
      <div class="acts">
        <button class="act" data-tx-same="${escapeHtml(it.name)}"><span class="lbl">כל עצי ${escapeHtml(it.name)} על המפה
          <span class="hint">${fmt(same.length)} רשומות בסקר, בסייר העצים</span></span></button>
      </div>`;
  }

  // The button inside the card lives in app.js's pane, so it is caught here.
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-tx-same]');
    if (!b) return;
    state.f = { species: [b.dataset.txSame] };
    state.q = '';
    state.c = 'health';
    colorCache.key = null;
    open();
    fit();
  });

  /* ---------- the address bar ---------- */

  function writeUrl() {
    const parts = [];
    if (state.c !== 'health') parts.push(`c.${state.c}`);
    Object.entries(state.f).forEach(([id, vs]) => { if (vs.length) parts.push(`${id}.${vs.join('*')}`); });
    if (state.q) parts.push(`q.${state.q}`);
    if (state.open) parts.push('o');
    const params = new URLSearchParams(location.search);
    if (parts.length) params.set('tr', parts.join('_')); else params.delete('tr');
    const query = params.toString().replace(/%2C/g, ',');
    try {
      history.replaceState(null, '', `${location.pathname}${query ? '?' + query : ''}${location.hash}`);
    } catch (err) { /* file:// */ }
  }

  /** Read `tr=` once, after the layers are up. */
  function init() {
    const raw = new URLSearchParams(location.search).get('tr');
    if (!layer()) return;
    // Even with nothing in the link: the pictures on the map come from apply().
    if (!raw) { apply(); return; }
    raw.split('_').forEach((part) => {
      if (part === 'o') { state.open = true; return; }
      const i = part.indexOf('.');
      if (i < 0) return;
      const k = part.slice(0, i), v = part.slice(i + 1);
      if (k === 'c' && dim(v)) state.c = v;
      else if (k === 'q') state.q = v;
      else if (dim(k)) state.f[k] = v.split('*').filter(Boolean);
    });
    Layers.turnOn(Layers.TREES_ID);
    if (state.open) open(); else apply();
  }

  return { open, close, init, detailBlock, onDetail, resume, apply };
})();
