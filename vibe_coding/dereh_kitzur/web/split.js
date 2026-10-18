/* Split - one long recording into the trails that are inside it.
 *
 * A walk with somebody who knows the moshava is recorded as one line of a few
 * kilometres. Most of that line is streets, a good part of it is shortcuts
 * that are already on the map, and somewhere around a third is new. Before
 * 18/10/2026 the way to get the new third out was the shape editor: zoom in,
 * tap a point, "פצל כאן", thirty times; save thirty-one trails called "(2)" to
 * "(31)"; then visit nineteen of them to remove them and twelve to rename them.
 *
 * Here the cutting is done first, by looking at where the line runs:
 *
 *   within KNOWN_M of a shortcut already on the map   -> known  (blue)
 *   within STREET_M of a named street (walknet.json)  -> street (grey, dashed)
 *   anywhere else                                     -> new    (green)
 *
 * Runs shorter than MIN_RUN_M are folded into the run before them, so that
 * crossing a road is not a piece of its own. What is left is a list: the new
 * pieces are kept and named after the streets at their two ends, the way the
 * trails on this map are already named ("הטוריים - מלאכי"), and the rest are
 * shown as not kept. Everything after that is the walker's judgement:
 *
 *   tap a piece             keep it or not, or cut it in two where tapped
 *   drag a ✂                moves the boundary - only along the recording, so
 *                           there is no point to hit and no need to zoom in
 *   tap a ✂                 remove that cut, joining the two pieces
 *   tap a row in the list   flies the map to that piece
 *
 * Saving writes every kept piece as a trail of its own, in the shortcuts
 * layer, credited to whoever made the recording (Store.splitOut, or
 * Drafts.splitOut for a recording that was never published). The recording
 * itself stays where it is unless the box at the bottom of the list says
 * otherwise, and remembers what it was split into.
 *
 * Drawn as an SVG over the map, as shape.js is and for the same reason: a ✂
 * that trails the finger by a worker round trip is one you cannot place.
 */
'use strict';

const Split = (() => {

  const STREET_M = 12;          // this close to a named street: walking the street
  const KNOWN_M = 10;           // this close to a shortcut on the map: already there
  const MIN_RUN_M = 25;         // a run shorter than this joins the one before it
  const NAME_M = 150;           // a street further than this does not name an end
  const MIN_LEN_M = 300;        // shorter than this is a trail, not a recording
  const MIN_POINTS = 40;
  const CELL = 50;              // metres, the spatial index's grid
  const EDGE_HIT = 16;          // px from a piece that still selects it
  const CUT_GAP_PX = 34;        // a ✂ closer than this to the last one is a tick
  const SLOP_MOUSE = 10;
  const SLOP_TOUCH = 18;
  const UNDO_MAX = 200;
  const LIVE = 'dk.split.v1';
  const LIVE_MAX_MS = 24 * 60 * 60 * 1000;

  let s = null;                 // the session, see start()
  let drag = null;
  let paintQueued = false;

  /* ---------- geometry ----------
   *
   * Everything is measured in a flat local frame in metres. Across a moshava a
   * few kilometres wide the error of that is far below a GPS fix's. */

  const KY = 110540;
  const kx = (lat) => 111320 * Math.cos(lat * Math.PI / 180);

  function segDist(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  /** A grid of stretches, so that "the nearest street" is a look at nine
   *  cells and not at thirty thousand stretches. */
  function grid(KX) {
    const cells = new Map();
    const key = (gx, gy) => gx + ',' + gy;
    return {
      add(a, b, ref) {
        const ax = a[1] * KX, ay = a[0] * KY, bx = b[1] * KX, by = b[0] * KY;
        for (let gx = Math.floor(Math.min(ax, bx) / CELL); gx <= Math.floor(Math.max(ax, bx) / CELL); gx++) {
          for (let gy = Math.floor(Math.min(ay, by) / CELL); gy <= Math.floor(Math.max(ay, by) / CELL); gy++) {
            const k = key(gx, gy);
            if (!cells.has(k)) cells.set(k, []);
            cells.get(k).push([ax, ay, bx, by, ref]);
          }
        }
      },
      /** The nearest stretch to [lat, lng] within `reach` cells, skipping
       *  `skip` (a ref), as {d, ref}. */
      near(p, reach = 1, skip) {
        const x = p[1] * KX, y = p[0] * KY;
        const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL);
        let best = { d: Infinity, ref: null };
        for (let gx = cx - reach; gx <= cx + reach; gx++) {
          for (let gy = cy - reach; gy <= cy + reach; gy++) {
            const list = cells.get(key(gx, gy));
            if (!list) continue;
            for (const e of list) {
              if (skip != null && e[4] === skip) continue;
              const d = segDist(x, y, e[0], e[1], e[2], e[3]);
              if (d < best.d) best = { d, ref: e[4] };
            }
          }
        }
        return best;
      }
    };
  }

  let streets = null;           // the named streets, indexed once per load

  async function loadStreets() {
    if (streets) return streets;
    const base = await Route.load();
    const g = grid(kx(base.lat[0] || 32.47));
    base.ways.forEach((w) => {
      if (w[0] < 0) return;     // a footway with no name says nothing about a street
      for (let k = 2; k < w.length; k++) {
        g.add([base.lat[w[k - 1]], base.lng[w[k - 1]]], [base.lat[w[k]], base.lng[w[k]]], w[0]);
      }
    });
    streets = { grid: g, names: base.names };
    return streets;
  }

  /** The shortcuts a recording may walk along without that being news: every
   *  published trail layer except a private one (fenced paths are exactly the
   *  thing a walk should be allowed to rediscover) and except the recording's
   *  own layer when it has one. The walks with Shahar sit in a layer of their
   *  own, and the walk of the 16th going over the walk of the 15th is not the
   *  16th finding nothing new. */
  function knownTrails(it) {
    const own = Layers.layerOf(it.id);
    const skip = own && own.id !== Layers.TRAILS_ID ? own.id : null;
    return Layers.trailLayers()
      .filter((l) => !l.private && l.id !== skip)
      .flatMap((l) => l.segments)
      .filter((g) => g.id !== it.id && !g.trip && g.path && g.path.length > 1);
  }

  /** Street, known or new, for every point; then runs; then pieces. */
  function analyse(path, st, known) {
    const KX = kx(path[0][0]);
    const kg = grid(KX);
    known.forEach((g) => {
      for (let i = 1; i < g.path.length; i++) kg.add(g.path[i - 1], g.path[i], g.id);
    });
    const cls = path.map((p) => {
      if (kg.near(p).d < KNOWN_M) return 'T';
      if (st.grid.near(p).d < STREET_M) return 'S';
      return 'O';
    });

    const cum = cumulative(path);
    const L = (i, j) => cum[j] - cum[i];
    const runs = [];
    cls.forEach((c, i) => {
      const last = runs[runs.length - 1];
      if (last && last.kind === c) last.b = i;
      else runs.push({ kind: c, a: i, b: i });
    });
    // Fold the short ones in. A run reaches to where the next one starts, so
    // that consecutive pieces share their boundary point.
    runs.forEach((r, k) => { if (runs[k + 1]) r.b = runs[k + 1].a; });
    const out = [];
    runs.forEach((r) => {
      const last = out[out.length - 1];
      if (last && (L(r.a, r.b) < MIN_RUN_M || last.kind === r.kind)) last.b = r.b;
      else out.push({ ...r });
    });
    const merged = [];
    out.forEach((r) => {
      const last = merged[merged.length - 1];
      if (last && last.kind === r.kind) last.b = r.b;
      else merged.push(r);
    });
    // A short first run has nothing before it to fold into.
    if (merged.length > 1 && L(merged[0].a, merged[0].b) < MIN_RUN_M) {
      merged[1].a = 0;
      merged.shift();
    }
    return merged.map((r) => ({ a: r.a, b: r.b, kind: r.kind, keep: r.kind === 'O', name: '', custom: false }));
  }

  function cumulative(path) {
    const cum = [0];
    for (let i = 1; i < path.length; i++) cum.push(cum[i - 1] + Layers.metres(path[i - 1], path[i]));
    return cum;
  }

  /** "רחוב - רחוב", from the nearest named street at each end. Both ends on
   *  the same street name the far end after the next street out, since "הזית -
   *  הזית" says nothing. */
  function suggest(p) {
    const reach = Math.ceil(NAME_M / CELL);
    const A = streets.grid.near(s.path[p.a], reach);
    const a = A.d < NAME_M ? A.ref : null;
    let B = streets.grid.near(s.path[p.b], reach);
    if (B.ref === a) B = streets.grid.near(s.path[p.b], reach, a);
    const b = B.d < NAME_M ? B.ref : null;
    const nm = (r) => streets.names[r];
    if (p.kind === 'S' && a != null) return nm(a);
    if (a != null && b != null) return `${nm(a)} - ${nm(b)}`;
    if (a != null || b != null) return `ליד ${nm(a != null ? a : b)}`;
    return `${s.name} · קטע`;
  }

  function rename() {
    s.pieces.forEach((p) => { if (!p.custom) p.name = suggest(p); });
  }

  const len = (p) => Math.round(s.cum[p.b] - s.cum[p.a]);
  const kept = () => s.pieces.filter((p) => p.keep);
  const metresText = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + ' ק"מ' : m + ' מ׳');

  /* ---------- editing, with undo ---------- */

  const copy = (pieces) => pieces.map((p) => ({ ...p }));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  function edit(fn) {
    const before = copy(s.pieces);
    fn();
    rename();
    if (same(before, s.pieces)) return false;
    s.undo.push(before);
    if (s.undo.length > UNDO_MAX) s.undo.shift();
    s.redo = [];
    saveLive();
    return true;
  }

  function undo() {
    if (!s || !s.undo.length) return;
    s.redo.push(copy(s.pieces));
    s.pieces = s.undo.pop();
    s.sel = null;
    saveLive();
    paintAll();
  }

  function redo() {
    if (!s || !s.redo.length) return;
    s.undo.push(copy(s.pieces));
    s.pieces = s.redo.pop();
    s.sel = null;
    saveLive();
    paintAll();
  }

  function toggle(k) {
    edit(() => { s.pieces[k].keep = !s.pieces[k].keep; });
  }

  /** Cut piece k at point i of the recording. The first half keeps a name
   *  somebody typed; the second is named afresh. */
  function cutAt(k, i) {
    const p = s.pieces[k];
    if (i <= p.a || i >= p.b) return;
    edit(() => {
      s.pieces.splice(k, 1, { ...p, b: i }, { ...p, a: i, custom: false, name: '' });
    });
    s.sel = { t: 'p', k };
  }

  /** Remove the cut between piece k and k+1. The joined piece is kept if
   *  either was, is of the longer one's kind, and keeps a typed name. */
  function join(k) {
    const A = s.pieces[k];
    const B = s.pieces[k + 1];
    if (!A || !B) return;
    edit(() => {
      const longer = len(A) >= len(B) ? A : B;
      const named = A.custom ? A : (B.custom ? B : null);
      s.pieces.splice(k, 2, {
        a: A.a, b: B.b, kind: longer.kind, keep: A.keep || B.keep,
        name: named ? named.name : '', custom: !!named
      });
    });
    s.sel = { t: 'p', k };
  }

  /* ---------- drawing ---------- */

  function host() { return map.getContainer(); }
  const px = (pt) => map.project([pt[1], pt[0]]);
  const ptsAttr = (pp) => pp.map((p) => p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ');

  function schedulePaint() {
    if (paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => { paintQueued = false; paint(); });
  }

  /** The number a kept piece goes by, on the map and in the list. */
  function numbers() {
    const n = [];
    let c = 0;
    s.pieces.forEach((p) => { n.push(p.keep ? ++c : 0); });
    return n;
  }

  function paint() {
    if (!s || !map) return;
    const W = host().clientWidth;
    const H = host().clientHeight;
    const seen = (p, pad = 40) => p.x > -pad && p.y > -pad && p.x < W + pad && p.y < H + pad;
    s.proj = s.path.map(px);
    const nums = numbers();
    const sel = s.sel;

    let under = '';
    let over = '';
    s.pieces.forEach((p, k) => {
      const pts = ptsAttr(s.proj.slice(p.a, p.b + 1));
      const cls = p.keep ? 'keep' : (p.kind === 'T' ? 'known' : 'street');
      const on = sel && sel.t === 'p' && sel.k === k ? ' on' : '';
      if (p.keep) under += `<polyline class="sp-case${on}" points="${pts}"/>`;
      over += `<polyline class="sp-line ${cls}${on}" points="${pts}"/>`;
    });

    let marks = '';
    s.pieces.forEach((p, k) => {
      if (!nums[k]) return;
      const m = s.proj[Math.floor((p.a + p.b) / 2)];
      if (!seen(m)) return;
      marks += `<g class="sp-num" transform="translate(${m.x.toFixed(1)} ${m.y.toFixed(1)})">`
        + `<circle r="11"/><text dy=".35em">${nums[k]}</text></g>`;
    });
    // The cuts last, on top: they are what the finger is reaching for. Zoomed
    // out, thirty of them pile up into a heap nobody can grab one out of, so
    // a cut too close to the last one drawn is a small tick until zoomed in.
    let last = null;
    s.ticks = 0;
    s.pieces.forEach((p, k) => {
      if (!k) return;
      const c = s.proj[p.a];
      if (!seen(c)) return;
      const on = sel && sel.t === 'c' && sel.k === k - 1 ? ' on' : '';
      const crowded = !on && last && Math.hypot(c.x - last.x, c.y - last.y) < CUT_GAP_PX;
      if (crowded) {
        s.ticks += 1;
        marks += `<circle class="sp-tick" cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="4"/>`;
        return;
      }
      last = c;
      marks += `<g class="sp-cut${on}" data-k="${k - 1}" transform="translate(${c.x.toFixed(1)} ${c.y.toFixed(1)})">`
        + '<circle class="hit" r="20"/><circle class="dot" r="12"/>'
        + '<text dy=".36em">✂</text></g>';
    });

    s.svg.innerHTML = under + over + marks;
    placePop();
    paintBar();
  }

  function paintAll() {
    paint();
    paintList();
  }

  /* ---------- the small menu by the selection ---------- */

  function popHtml() {
    if (!s.sel || drag) return '';
    if (s.sel.t === 'c') return '<button data-op="join">✕ הסר את החיתוך</button>';
    const p = s.pieces[s.sel.k];
    if (!p) return '';
    return (p.keep ? '<button data-op="toggle">לא לשמור</button>'
      : '<button data-op="toggle" class="strong">✚ לשמור כשביל</button>')
      + (s.sel.i != null ? '<button data-op="cut">✂ חתוך כאן</button>' : '');
  }

  function placePop() {
    const pop = s.pop;
    const html = popHtml();
    if (!html) { pop.hidden = true; pop.dataset.html = ''; return; }
    if (pop.dataset.html !== html) { pop.innerHTML = html; pop.dataset.html = html; }
    let at;
    if (s.sel.t === 'c') at = s.proj[s.pieces[s.sel.k + 1].a];
    else if (s.sel.i != null) at = s.proj[s.sel.i];
    else {
      const p = s.pieces[s.sel.k];
      at = s.proj[Math.floor((p.a + p.b) / 2)];
    }
    if (!at) { pop.hidden = true; return; }
    pop.hidden = false;
    const W = host().clientWidth;
    const x = Math.max(8, Math.min(W - pop.offsetWidth - 8, at.x - pop.offsetWidth / 2));
    const barBottom = el('split-bar').getBoundingClientRect().bottom - host().getBoundingClientRect().top;
    let y = at.y - pop.offsetHeight - 24;
    if (y < barBottom + 6) y = at.y + 24;
    pop.style.left = x + 'px';
    pop.style.top = y + 'px';
  }

  function onPop(e) {
    const btn = e.target.closest('[data-op]');
    if (!btn || !s || !s.sel) return;
    e.stopPropagation();
    const op = btn.dataset.op;
    const sel = s.sel;
    if (op === 'join') join(sel.k);
    else if (op === 'toggle') toggle(sel.k);
    else if (op === 'cut' && sel.i != null) cutAt(sel.k, sel.i);
    paintAll();
  }

  /* ---------- the bar ---------- */

  function say(text, bad) {
    el('split-state').textContent = text;
    el('split-bar').classList.toggle('bad', !!bad);
  }

  function paintBar() {
    const k = kept();
    const total = k.reduce((m, p) => m + len(p), 0);
    el('split-len').textContent = metresText(Math.round(s.cum[s.cum.length - 1]));
    let state;
    if (s.busy) state = s.busy;
    else if (s.sel && s.sel.t === 'c') state = 'גרור את ✂ לאורך הקו כדי להזיז את הגבול, או הסר את החיתוך.';
    else if (s.sel) {
      const p = s.pieces[s.sel.k];
      state = p.keep ? `"${p.name}" · ${metresText(len(p))}. ייכנס למפה כשביל.`
        : `${metresText(len(p))} ${p.kind === 'T' ? 'על שביל שכבר במפה' : p.kind === 'S' ? 'ברחוב' : ''}. לא יישמר.`;
    } else if (!k.length) state = 'אף קטע לא מסומן לשמירה. לחץ על קטע בקו כדי לשמור אותו.';
    else state = `${k.length} שבילים, ${metresText(total)}. לחץ על קטע כדי לשמור או לא, גרור ✂ כדי להזיז גבול.`
      + (s.ticks ? ' התקרב כדי לראות את כל הגבולות.' : '');
    if (!s.busy) say(state);
    el('split-undo').disabled = !s.undo.length;
    el('split-redo').disabled = !s.redo.length;
    const save = el('split-save');
    save.disabled = !!s.busy || !k.length;
    save.textContent = k.length ? `שמור ${k.length}` : 'שמור';
  }

  /* ---------- the list ---------- */

  function paintList() {
    if (!s) return;
    const nums = numbers();
    const k = kept();
    el('split-sum').textContent = k.length
      ? `${k.length} שבילים חדשים, ${metresText(k.reduce((m, p) => m + len(p), 0))}. השמות מוצעים לפי הרחובות בקצוות, ואפשר לשנות אותם.`
      : 'שום קטע לא מסומן לשמירה.';
    const sel = s.sel && s.sel.t === 'p' ? s.sel.k : -1;
    const focused = document.activeElement && document.activeElement.closest('#split-list')
      ? document.activeElement.dataset.k : null;
    el('split-list').innerHTML = s.pieces.map((p, i) => {
      const on = i === sel ? ' on' : '';
      if (p.keep) {
        return `<li class="sp-row keep${on}" data-k="${i}">
          <span class="sp-n">${nums[i]}</span>
          <input class="sp-name" data-k="${i}" value="${escapeHtml(p.name)}" aria-label="שם השביל ${nums[i]}" maxlength="80">
          <span class="sp-len">${metresText(len(p))}</span>
          <button class="sp-tg on" data-tg="${i}" aria-pressed="true" title="לא לשמור"></button>
        </li>`;
      }
      const what = p.kind === 'T' ? 'על שביל שכבר במפה' : p.kind === 'S' ? `ברחוב ${escapeHtml(p.name)}` : 'מחוץ לרחובות';
      return `<li class="sp-row drop k-${p.kind}${on}" data-k="${i}">
        <span class="sp-ic" aria-hidden="true"></span>
        <span class="sp-what">${what} · ${metresText(len(p))} · לא יישמר</span>
        <button class="sp-tg" data-tg="${i}" aria-pressed="false" title="לשמור כשביל"></button>
      </li>`;
    }).join('');
    if (focused != null) {
      const input = el('split-list').querySelector(`.sp-name[data-k="${focused}"]`);
      if (input) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }
    }
    const row = el('split-list').querySelector('.sp-row.on');
    if (row && s.scrollTo) { row.scrollIntoView({ block: 'nearest' }); s.scrollTo = false; }
  }

  function onListClick(e) {
    if (!s) return;
    const tg = e.target.closest('[data-tg]');
    if (tg) {
      const k = +tg.dataset.tg;
      toggle(k);
      s.sel = { t: 'p', k };
      paintAll();
      return;
    }
    if (e.target.closest('input')) return;
    const row = e.target.closest('.sp-row');
    if (!row) return;
    focusPiece(+row.dataset.k);
  }

  function focusPiece(k) {
    const p = s.pieces[k];
    if (!p) return;
    s.sel = { t: 'p', k };
    const b = new maplibregl.LngLatBounds();
    s.path.slice(Math.max(0, p.a - 3), p.b + 4).forEach(([la, ln]) => b.extend([ln, la]));
    map.fitBounds(b, { padding: pad(), maxZoom: 18.5, duration: 500 });
    paintAll();
  }

  function onNameInput(e) {
    const input = e.target.closest('.sp-name');
    if (!input || !s) return;
    const p = s.pieces[+input.dataset.k];
    if (!p) return;
    // Typing is not an undo step per letter: one step when the field is left.
    if (!s.typing) s.typing = copy(s.pieces);
    p.name = input.value;
    p.custom = input.value.trim() !== '';
    saveLive();
    paintBar();
  }

  function onNameDone() {
    if (!s || !s.typing) return;
    const before = s.typing;
    s.typing = null;
    if (same(before, s.pieces)) return;
    s.undo.push(before);
    s.redo = [];
    // An emptied field goes back to the suggestion rather than to nothing.
    // The fields are updated in place, not redrawn: this runs on the way to
    // the next field, and redrawing the list would take that one away.
    rename();
    el('split-list').querySelectorAll('.sp-name').forEach((input) => {
      const p = s.pieces[+input.dataset.k];
      if (p && input !== document.activeElement && input.value !== p.name) input.value = p.name;
    });
    saveLive();
    paint();
  }

  /* ---------- taps on the map ---------- */

  /** The piece under a tap, and the point of the recording nearest to it. */
  function pieceAt(point) {
    let best = null;
    let bestD = EDGE_HIT;
    for (let i = 0; i < s.proj.length - 1; i++) {
      const a = s.proj[i];
      const b = s.proj[i + 1];
      const d = segDist(point.x, point.y, a.x, a.y, b.x, b.y);
      if (d < bestD) {
        bestD = d;
        const near = Math.hypot(point.x - a.x, point.y - a.y) <= Math.hypot(point.x - b.x, point.y - b.y) ? i : i + 1;
        best = { i: near, seg: i };
      }
    }
    if (!best) return null;
    const k = s.pieces.findIndex((p) => best.seg >= p.a && best.seg < p.b);
    return k < 0 ? null : { t: 'p', k, i: best.i };
  }

  function onTap(point) {
    if (!s || drag) return;
    const hit = pieceAt(point);
    s.sel = hit;
    if (hit) {
      const p = s.pieces[hit.k];
      // Too close to an end to be worth a cut there.
      if (hit.i - p.a < 2 || p.b - hit.i < 2) hit.i = null;
      s.scrollTo = true;
    }
    paintAll();
  }

  function bindTaps() {
    const canvas = map.getCanvas();
    let start = null;
    let down = 0;
    const press = (e) => {
      down += 1;
      if (down > 1 || (e.button != null && e.button !== 0)) { start = null; return; }
      start = { x: e.clientX, y: e.clientY, touch: e.pointerType === 'touch' };
    };
    const release = (e) => {
      down = Math.max(0, down - 1);
      const from = start;
      start = null;
      if (!from || down) return;
      if (Math.hypot(e.clientX - from.x, e.clientY - from.y) > (from.touch ? SLOP_TOUCH : SLOP_MOUSE)) return;
      const box = canvas.getBoundingClientRect();
      onTap({ x: e.clientX - box.left, y: e.clientY - box.top });
    };
    const abandon = () => { down = 0; start = null; };
    canvas.addEventListener('pointerdown', press);
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', abandon);
    return () => {
      canvas.removeEventListener('pointerdown', press);
      canvas.removeEventListener('pointerup', release);
      canvas.removeEventListener('pointercancel', abandon);
    };
  }

  /* ---------- dragging a ✂ ----------
   *
   * The ✂ between piece k and k+1 can go anywhere strictly between the start
   * of k and the end of k+1, and nowhere off the recording: on every move it
   * lands on the recorded point nearest the pointer within that range. */

  function onDown(e) {
    const g = e.target.closest('.sp-cut');
    if (!g || !s || drag || (e.button != null && e.button > 0)) return;
    e.preventDefault();
    e.stopPropagation();
    drag = {
      id: e.pointerId,
      k: +g.dataset.k,
      from: { x: e.clientX, y: e.clientY },
      touch: e.pointerType === 'touch',
      moved: false,
      before: null
    };
    map.dragPan.disable();
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onCancel);
  }

  function onMove(e) {
    if (!drag || e.pointerId !== drag.id) return;
    if (!drag.moved) {
      if (Math.hypot(e.clientX - drag.from.x, e.clientY - drag.from.y) < (drag.touch ? 7 : 3)) return;
      drag.moved = true;
      drag.before = copy(s.pieces);
      s.sel = { t: 'c', k: drag.k };
      document.body.classList.add('sp-dragging');
    }
    const box = map.getCanvas().getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    const A = s.pieces[drag.k];
    const B = s.pieces[drag.k + 1];
    let best = B.a;
    let bestD = Infinity;
    for (let i = A.a + 1; i < B.b; i++) {
      const q = s.proj[i];
      const d = (q.x - x) * (q.x - x) + (q.y - y) * (q.y - y);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best !== B.a) {
      A.b = best;
      B.a = best;
      schedulePaint();
    }
  }

  function release() {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onCancel);
    document.body.classList.remove('sp-dragging');
    map.dragPan.enable();
  }

  function onCancel(e) {
    if (!drag || e.pointerId !== drag.id) return;
    release();
    if (drag.moved && drag.before) s.pieces = drag.before;
    drag = null;
    paintAll();
  }

  function onUp(e) {
    if (!drag || e.pointerId !== drag.id) return;
    release();
    const d = drag;
    drag = null;
    if (d.moved) {
      rename();
      if (!same(d.before, s.pieces)) {
        s.undo.push(d.before);
        if (s.undo.length > UNDO_MAX) s.undo.shift();
        s.redo = [];
        saveLive();
      }
      s.sel = null;
    } else {
      const was = s.sel && s.sel.t === 'c' && s.sel.k === d.k;
      s.sel = was ? null : { t: 'c', k: d.k };
    }
    paintAll();
  }

  /* ---------- keys ---------- */

  function onKey(e) {
    if (!s || drag) return;
    if (e.target.closest && e.target.closest('input, textarea, select, [contenteditable]')) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
    if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
    if (e.key === 'Escape' && s.sel) { e.stopPropagation(); s.sel = null; paintAll(); }
  }

  /* ---------- surviving a reload ----------
   *
   * Naming twelve trails is the long part, and a locked phone should not cost
   * it. Same promise and same limits as the shape editor's mirror. */

  function saveLive() {
    try {
      if (!s) { localStorage.removeItem(LIVE); return; }
      localStorage.setItem(LIVE, JSON.stringify({
        kind: s.kind, id: s.id, name: s.name, n: s.path.length, pieces: s.pieces, at: Date.now()
      }));
    } catch (err) {
      /* private mode or a full quota: the editor still works, unmirrored */
    }
  }

  let offered = false;

  function restore() {
    if (offered || s || typeof map === 'undefined' || !map || !map.getSource('src-trails')) return;
    let rec = null;
    try {
      rec = JSON.parse(localStorage.getItem(LIVE) || 'null');
    } catch (err) { /* nothing worth offering */ }
    if (!rec || !rec.id || !Array.isArray(rec.pieces)) return;
    if (Date.now() - (rec.at || 0) > LIVE_MAX_MS) { localStorage.removeItem(LIVE); return; }
    const it = Layers.item(rec.id);
    // A recording that changed since is a different line; the cuts would land
    // on the wrong points.
    if (!it || !it.path || it.path.length !== rec.n || (rec.kind === 'trail' && !Store.isEditor())) return;
    offered = true;
    if (!confirm(`נשאר פירוק של "${rec.name}" שלא נשמר.\n\nלהמשיך מאיפה שהפסקת?`)) {
      localStorage.removeItem(LIVE);
      return;
    }
    start(it, rec.pieces);
  }

  /* ---------- in and out ---------- */

  /** A recording worth splitting: a line the shape editor could edit, long
   *  enough and dense enough to be a walk rather than a shortcut. */
  function canSplit(it) {
    if (typeof Shape === 'undefined' || !Shape.canShape(it)) return false;
    return it.path.length >= MIN_POINTS && Layers.pathLength(it.path) >= MIN_LEN_M;
  }

  async function open(it) {
    if (!map) { alert('פירוק הקלטה דורש את המפה, והדפדפן הזה לא מציג אותה.'); return; }
    if (!canSplit(it)) return;
    start(it, null);
  }

  const pad = () => ({
    top: el('split-bar').getBoundingClientRect().height + 40,
    bottom: 50, left: 40, right: 40
  });

  async function start(it, pieces) {
    if (s) close(true);
    if (Shape.isOn()) Shape.close(true);
    stopNav();
    Arrange.close(true);
    if (Drafts.isDrafting()) Drafts.stop();
    if (Route.isOn()) Route.close();
    deselect();

    let st;
    try {
      st = await loadStreets();
    } catch (err) {
      alert('לא הצלחתי לטעון את מפת הרחובות, ובלעדיה אי אפשר למצוא את הקטעים. ' + err.message);
      return;
    }

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'split-svg');
    svg.addEventListener('pointerdown', onDown);
    const pop = document.createElement('div');
    pop.className = 'shape-pop split-pop';
    pop.hidden = true;
    pop.addEventListener('click', onPop);
    pop.addEventListener('pointerdown', (e) => e.stopPropagation());
    host().append(svg, pop);

    const path = it.path.map((p) => p.slice());
    const layer = Layers.layerOf(it.id);
    s = {
      kind: it.draft ? 'draft' : 'trail',
      id: it.id,
      name: it.name,
      path,
      cum: cumulative(path),
      pieces: pieces ? copy(pieces) : analyse(path, st, knownTrails(it)),
      // Kept by default only where it does not end up drawn twice: a
      // recording in a layer of its own stays there; one in the shortcuts
      // layer, or a draft, is replaced by its pieces.
      keepOrig: !it.draft && !!layer && layer.id !== Layers.TRAILS_ID,
      sel: null,
      proj: [],
      busy: '',
      undo: [],
      redo: [],
      typing: null,
      pitch: map.getPitch(),
      svg,
      pop,
      offTaps: bindTaps()
    };
    rename();

    el('split-name').textContent = it.name;
    el('split-keep-orig').checked = s.keepOrig;
    el('split-keep-lbl').textContent = s.kind === 'draft'
      ? 'להשאיר גם את הטיוטה המקורית במכשיר'
      : `להשאיר את ההקלטה המקורית${layer ? ` בשכבה "${layer.name}"` : ''}`;
    s.view = { list: !el('list-view').hidden, detail: !el('detail-view').hidden };
    el('list-view').hidden = true;
    el('detail-view').hidden = true;
    el('split-view').hidden = false;
    document.body.classList.add('splitting');
    el('split-bar').hidden = false;
    Layers.setShaping(it.id);
    map.doubleClickZoom.disable();
    map.on('move', paint);
    window.addEventListener('keydown', onKey, true);
    saveLive();

    const b = new maplibregl.LngLatBounds();
    path.forEach(([lat, lng]) => b.extend([lng, lat]));
    map.jumpTo({ pitch: 0 });
    map.fitBounds(b, { padding: pad(), maxZoom: 18, duration: 500 });
    paintAll();
  }

  const changed = () => s && (s.undo.length > 0 || s.typing);

  function close(force) {
    if (!s) return true;
    if (!force && changed() && !confirm('לצאת בלי לשמור את הפירוק?')) return false;
    if (drag) { release(); drag = null; }
    s.offTaps();
    s.svg.remove();
    s.pop.remove();
    map.off('move', paint);
    map.doubleClickZoom.enable();
    window.removeEventListener('keydown', onKey, true);
    const { pitch, view } = s;
    s = null;
    saveLive();
    Layers.setShaping(null);
    document.body.classList.remove('splitting');
    el('split-bar').hidden = true;
    el('split-bar').classList.remove('bad');
    el('split-view').hidden = true;
    el('split-list').innerHTML = '';
    el('list-view').hidden = !view.list;
    el('detail-view').hidden = !view.detail;
    if (pitch) map.easeTo({ pitch, duration: 400 });
    return true;
  }

  async function save() {
    if (!s || s.busy) return;
    onNameDone();
    const pieces = kept().map((p, k) => ({
      path: s.path.slice(p.a, p.b + 1),
      name: p.name.trim() || `${s.name} · ${k + 1}`
    }));
    if (!pieces.length) return;
    const keepOrig = el('split-keep-orig').checked;
    const { kind, id, name } = s;
    s.busy = kind === 'trail' ? `שומר ${pieces.length} שבילים במסד המשותף…` : 'שומר…';
    say(s.busy);
    paintBar();
    try {
      let ids;
      if (kind === 'trail') {
        const out = await Store.splitOut(id, pieces, keepOrig, name);
        await reloadShared(out.doc);
        ids = out.ids;
      } else {
        ids = await Drafts.splitOut(id, pieces, keepOrig);
      }
      close(true);
      if (ids.length) select(keepOrig ? id : ids[0]);
    } catch (err) {
      if (!s) return;
      s.busy = '';
      paintBar();
      say('לא נשמר: ' + err.message + ' הפירוק עדיין כאן.', true);
    }
  }

  function wire() {
    el('split-save').addEventListener('click', save);
    el('split-cancel').addEventListener('click', () => close());
    el('split-undo').addEventListener('click', undo);
    el('split-redo').addEventListener('click', redo);
    const list = el('split-list');
    list.addEventListener('click', onListClick);
    list.addEventListener('input', onNameInput);
    list.addEventListener('focusout', onNameDone);
    list.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.closest('.sp-name')) e.target.blur();
    });
  }

  return { open, close, wire, restore, canSplit, isOn: () => !!s,
           kind: () => (s ? s.kind : null),
           // for tests
           _state: () => s, _analyse: analyse, _loadStreets: loadStreets, _known: knownTrails };
})();
