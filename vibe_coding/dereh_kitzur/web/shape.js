/* Shape - redrawing a trail that is already there.
 *
 * Until 7/10/2026 the only way to correct a line was to draw it again from
 * nothing (a draft) or not at all (a published trail). A GPS walk is usually
 * right for most of its length and wrong at a few corners, and a shortcut gets
 * longer when a new gap opens in a fence, so the job is nearly always "fix
 * this bit", and that is what this editor is for.
 *
 * Everything is direct manipulation on the line itself, the way a map editor
 * on a phone has to work:
 *
 *   drag a point            moves it. Hold near the edge and the map scrolls.
 *   drag a ⊕                the small plus halfway along every stretch: pulls
 *                           a new point out of the line. A tap adds it in place.
 *   tap a point             selects it: delete it, split the trail there, or
 *                           (at an end) carry the line on from it.
 *   tap a stretch           selects it: delete it, or add a point in it.
 *                           Deleting a stretch in the middle cuts the trail.
 *   carry on from an end    every tap on the map adds a point, until ✓. A tap
 *                           on the end of another piece joins the two.
 *   drop an end on an end   joins two pieces back into one.
 *
 * Ends snap to the ends of the other shortcuts, so a trail that is meant to
 * meet another one actually does - which is what the router needs to walk
 * from one to the next.
 *
 * Several pieces are allowed while editing. On save each piece becomes a trail
 * of its own (the longest keeps the original's id and everything on it); see
 * Store.reshape. A trail is one continuous walk everywhere else in the app.
 *
 * Drawn as an SVG over the map rather than as MapLibre layers, for the reason
 * draft.js gives for its ghost line: a GeoJSON source reaches the screen a
 * worker round trip after setData, and a dragged point that trails the finger
 * by a few frames is a point you cannot place.
 */
'use strict';

const Shape = (() => {

  const HIT_V = 17;            // px a finger may miss a point by
  const HIT_M = 14;            // ... and a ⊕
  const EDGE_HIT = 14;         // px from a stretch that still selects it
  const SNAP_PX = 18;          // an end this close to another end lands on it
  const MID_MIN_PX = 46;       // a stretch shorter than this on screen gets no ⊕
  const DENSE_PX = 9;          // points closer than this: zoom in to edit them
  const SIMPLIFY_M = 2;        // what "פשט" may move the line by, at most
  const UNDO_MAX = 200;
  const EDGE = 70;             // px from the visible edge where a drag scrolls
  const PAN_MAX = 14;
  const SLOP_MOUSE = 10;       // see SLOP_* in draft.js
  const SLOP_TOUCH = 18;
  const LIVE = 'dk.shape.v1';
  const LIVE_MAX_MS = 24 * 60 * 60 * 1000;

  let s = null;                // the session, see start()
  let drag = null;
  let raf = 0;
  let paintQueued = false;

  /* ---------- geometry ---------- */

  const copy = (lines) => lines.map((l) => l.map((p) => p.slice()));
  const round = (ll) => [+ll.lat.toFixed(6), +ll.lng.toFixed(6)];
  const px = (pt) => map.project([pt[1], pt[0]]);
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  /** Distance from p to the stretch a-b, in whatever units the three share. */
  function segDist(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2)) : 0;
    return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
  }

  /** Douglas-Peucker in metres. A walk recorded every four metres is three
   *  hundred points of GPS noise along a straight path, and three hundred
   *  handles is not a line anybody can edit. The ends always stay. */
  function simplify(line, tol) {
    if (line.length < 3) return line.slice();
    const kx = 111320 * Math.cos(line[0][0] * Math.PI / 180);
    const P = line.map(([la, ln]) => ({ x: ln * kx, y: la * 110540 }));
    const keep = new Uint8Array(line.length);
    keep[0] = keep[line.length - 1] = 1;
    const stack = [[0, line.length - 1]];
    while (stack.length) {
      const [a, b] = stack.pop();
      let best = -1;
      let at = -1;
      for (let i = a + 1; i < b; i++) {
        const d = segDist(P[i], P[a], P[b]);
        if (d > best) { best = d; at = i; }
      }
      if (best > tol) { keep[at] = 1; stack.push([a, at], [at, b]); }
    }
    return line.filter((_, i) => keep[i]);
  }

  const total = () => s.lines.reduce((n, l) => n + Layers.pathLength(l), 0);
  const points = () => s.lines.reduce((n, l) => n + l.length, 0);
  const changed = () => !same(s.lines.filter((l) => l.length > 1), [s.orig]);

  /* ---------- editing, with undo ----------
   *
   * Every change goes through edit(): a copy of the lines before, the change,
   * and the copy kept only if something actually moved. Snapshots rather than
   * inverse operations because the lines are small (a few hundred numbers)
   * and a snapshot cannot get an inverse wrong. */

  function edit(fn) {
    const before = copy(s.lines);
    fn();
    // A line of one point is only allowed while it is being carried on;
    // anywhere else it is a stray dot that would be saved as nothing.
    s.lines = s.lines.filter((l) => l.length > 1 || (s.extend && s.extend.line === l));
    if (same(before, s.lines)) return false;
    remember(before);
    return true;
  }

  function remember(before) {
    s.undo.push(before);
    if (s.undo.length > UNDO_MAX) s.undo.shift();
    s.redo = [];
    saveLive();
  }

  function undo() {
    if (!s || !s.undo.length) return;
    s.redo.push(copy(s.lines));
    s.lines = s.undo.pop();
    s.sel = null;
    s.extend = null;
    saveLive();
    paint();
  }

  function redo() {
    if (!s || !s.redo.length) return;
    s.undo.push(copy(s.lines));
    s.lines = s.redo.pop();
    s.sel = null;
    s.extend = null;
    saveLive();
    paint();
  }

  function deleteVertex(l, i) {
    edit(() => { s.lines[l].splice(i, 1); });
    s.sel = null;
  }

  /** Remove the stretch between point i and i+1. At either end of the line
   *  that trims it; in the middle it leaves two pieces. */
  function deleteEdge(l, i) {
    edit(() => {
      const line = s.lines[l];
      s.lines.splice(l, 1, line.slice(0, i + 1), line.slice(i + 1));
    });
    s.sel = null;
  }

  /** Two pieces that share point i. */
  function splitAt(l, i) {
    edit(() => {
      const line = s.lines[l];
      s.lines.splice(l, 1, line.slice(0, i + 1), line.slice(i));
    });
    s.sel = null;
  }

  function insertMid(l, i) {
    const line = s.lines[l];
    const a = line[i];
    const b = line[i + 1];
    edit(() => {
      line.splice(i + 1, 0, [+((a[0] + b[0]) / 2).toFixed(6), +((a[1] + b[1]) / 2).toFixed(6)]);
    });
    s.sel = { t: 'v', l, i: i + 1 };
  }

  /** Join the `endA` of one piece to the `endB` of another. The result runs
   *  from A's far end through the join to B's far end, so neither direction
   *  the two pieces happened to be drawn in matters. */
  function joinLines(A, endA, B, endB) {
    const a = endA === 'end' ? A.slice() : A.slice().reverse();
    const b = endB === 'start' ? B.slice() : B.slice().reverse();
    const near = Layers.metres(a[a.length - 1], b[0]) < 0.5;
    const merged = a.concat(near ? b.slice(1) : b);
    s.lines = s.lines.filter((l) => l !== A && l !== B);
    s.lines.push(merged);
    s.extend = null;
    s.sel = null;
  }

  function startExtend(l, end) {
    s.extend = { line: s.lines[l], end };
    s.sel = null;
    s.hover = null;
    paint();
  }

  function stopExtend() {
    if (!s.extend) return;
    s.extend = null;
    s.hover = null;
    s.lines = s.lines.filter((l) => l.length > 1);
    paint();
  }

  /** Start a fresh line, for when everything was deleted. */
  function drawNew() {
    s.lines.push([]);
    s.extend = { line: s.lines[s.lines.length - 1], end: 'end' };
    s.sel = null;
    paint();
  }

  /* ---------- snapping ---------- */

  /** The places an end may snap to: the ends of the other pieces (a join) and
   *  the ends of the other shortcuts on the map (a junction). `skip` is the
   *  end being moved, so it does not snap to itself. */
  function snapTargets(skip) {
    const out = [];
    s.lines.forEach((line, l) => {
      if (line.length < 1) return;
      [['start', 0], ['end', line.length - 1]].forEach(([end, i]) => {
        if (skip && skip.line === line && (skip.end === end || line.length === 1)) return;
        if (line.length === 1 && end === 'end') return;
        out.push({ pt: line[i], join: skip && skip.line !== line ? { line, end } : null });
      });
    });
    // shown() and not .on: a private layer is on for everybody and drawn only
    // for an editor, and a draft must not snap to a fence it cannot see.
    Layers.trailLayers().filter((layer) => Layers.shown(layer)).forEach((layer) => {
      layer.segments.forEach((seg) => {
        if (seg.id === s.id || !seg.path || seg.path.length < 2) return;
        out.push({ pt: seg.path[0] });
        out.push({ pt: seg.path[seg.path.length - 1] });
      });
    });
    return out;
  }

  function snapAt(point, skip) {
    let best = null;
    let bestD = SNAP_PX;
    snapTargets(skip).forEach((t) => {
      const p = px(t.pt);
      const d = Math.hypot(p.x - point.x, p.y - point.y);
      if (d < bestD) { bestD = d; best = { ...t, x: p.x, y: p.y }; }
    });
    return best;
  }

  /** Which end of its line point i is, if it is one. */
  function endOf(l, i) {
    const line = s.lines[l];
    if (!line) return null;
    if (i === line.length - 1) return 'end';
    if (i === 0) return 'start';
    return null;
  }

  /* ---------- drawing ---------- */

  function host() { return map.getContainer(); }

  function schedulePaint() {
    if (paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => { paintQueued = false; paint(); });
  }

  const ptsAttr = (pp) => pp.map((p) => p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ');

  function paint() {
    if (!s || !map) return;
    const W = host().clientWidth;
    const H = host().clientHeight;
    const seen = (p, pad = 40) => p.x > -pad && p.y > -pad && p.x < W + pad && p.y < H + pad;
    const proj = s.lines.map((line) => line.map(px));

    // Too dense to tell apart: the ends stay grabbable and the rest wait for
    // a closer zoom, rather than a carpet of overlapping handles.
    const gaps = [];
    proj.forEach((pp) => {
      for (let i = 1; i < pp.length; i++) gaps.push(Math.hypot(pp[i].x - pp[i - 1].x, pp[i].y - pp[i - 1].y));
    });
    gaps.sort((a, b) => a - b);
    s.dense = gaps.length > 3 && gaps[Math.floor(gaps.length / 2)] < DENSE_PX;

    let out = '';
    if (s.orig.length > 1) out += `<polyline class="sh-orig" points="${ptsAttr(s.orig.map(px))}"/>`;
    proj.forEach((pp) => {
      if (pp.length < 2) return;
      const pts = ptsAttr(pp);
      out += `<polyline class="sh-case" points="${pts}"/><polyline class="sh-line" points="${pts}"/>`;
    });

    if (s.sel && s.sel.t === 'e' && proj[s.sel.l] && proj[s.sel.l][s.sel.i + 1]) {
      const a = proj[s.sel.l][s.sel.i];
      const b = proj[s.sel.l][s.sel.i + 1];
      out += `<line class="sh-sel-edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`;
    }

    if (s.extend && s.extend.line.length) {
      const line = s.extend.line;
      const from = px(s.extend.end === 'end' ? line[line.length - 1] : line[0]);
      if (s.hover) {
        out += `<line class="sh-band" x1="${from.x}" y1="${from.y}" x2="${s.hover.x}" y2="${s.hover.y}"/>`;
      }
    }

    proj.forEach((pp, l) => {
      pp.forEach((p, i) => {
        const end = endOf(l, i);
        const on = s.sel && s.sel.t === 'v' && s.sel.l === l && s.sel.i === i;
        const active = s.extend && s.extend.line === s.lines[l] && end === s.extend.end;
        if (s.dense && !end && !on) return;
        if (!seen(p)) return;
        out += `<g class="sh-v${end ? ' end' : ''}${on ? ' on' : ''}${active ? ' active' : ''}"`
          + ` data-l="${l}" data-i="${i}" transform="translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})">`
          + `<circle class="hit" r="${HIT_V}"/><circle class="dot" r="${active ? 9 : on ? 8 : end ? 7.5 : 5.5}"/></g>`;
      });
    });

    // After the points, so a ⊕ is on top: it is the smaller target, and where
    // a point's generous touch area overlaps it the visible ⊕ should win.
    if (!s.dense && !s.extend && !drag) {
      proj.forEach((pp, l) => {
        for (let i = 0; i < pp.length - 1; i++) {
          const a = pp[i];
          const b = pp[i + 1];
          if (Math.hypot(b.x - a.x, b.y - a.y) < MID_MIN_PX) continue;
          const m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
          if (!seen(m)) continue;
          out += `<g class="sh-mid" data-l="${l}" data-i="${i}" transform="translate(${m.x.toFixed(1)} ${m.y.toFixed(1)})">`
            + `<circle class="hit" r="${HIT_M}"/><circle class="dot" r="7"/>`
            + '<path d="M-3.5 0H3.5M0 -3.5V3.5"/></g>';
        }
      });
    }

    if (s.snap) out += `<circle class="sh-snap" cx="${s.snap.x}" cy="${s.snap.y}" r="13"/>`;

    s.svg.innerHTML = out;
    placePop(proj);
    paintBar();
  }

  /* ---------- the small menu by the selection ---------- */

  function popHtml() {
    if (s.extend) {
      return '<button data-op="done-extend" class="strong">✓ סיום הוספה</button>';
    }
    if (!s.sel) return '';
    if (s.sel.t === 'v') {
      const end = endOf(s.sel.l, s.sel.i);
      return '<button data-op="del-v">🗑 מחק נקודה</button>'
        + (end ? '<button data-op="extend" class="strong">✚ המשך מכאן</button>'
          : '<button data-op="split">✂ פצל כאן</button>');
    }
    return '<button data-op="del-e">🗑 מחק קטע</button>'
      + '<button data-op="mid">✚ נקודה באמצע</button>';
  }

  function placePop(proj) {
    const pop = s.pop;
    const html = drag ? '' : popHtml();
    if (!html) { pop.hidden = true; pop.dataset.html = ''; return; }
    if (pop.dataset.html !== html) { pop.innerHTML = html; pop.dataset.html = html; }

    let at = null;
    if (s.extend && s.extend.line.length) {
      const line = s.extend.line;
      at = px(s.extend.end === 'end' ? line[line.length - 1] : line[0]);
    } else if (s.extend) {
      at = { x: host().clientWidth / 2, y: host().clientHeight - 140 };
    } else if (s.sel && proj[s.sel.l]) {
      const pp = proj[s.sel.l];
      if (s.sel.t === 'v') at = pp[s.sel.i];
      else if (pp[s.sel.i + 1]) at = { x: (pp[s.sel.i].x + pp[s.sel.i + 1].x) / 2,
                                       y: (pp[s.sel.i].y + pp[s.sel.i + 1].y) / 2 };
    }
    if (!at) { pop.hidden = true; return; }
    pop.hidden = false;
    const W = host().clientWidth;
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    const x = Math.max(8, Math.min(W - w - 8, at.x - w / 2));
    // Above the point, unless that is under the bar - then below it.
    const barBottom = el('shape-bar').getBoundingClientRect().bottom - host().getBoundingClientRect().top;
    let y = at.y - h - 22;
    if (y < barBottom + 6) y = at.y + 22;
    pop.style.left = x + 'px';
    pop.style.top = y + 'px';
  }

  function onPop(e) {
    const btn = e.target.closest('[data-op]');
    if (!btn || !s) return;
    e.stopPropagation();
    const op = btn.dataset.op;
    const sel = s.sel;
    if (op === 'done-extend') stopExtend();
    else if (op === 'del-v' && sel) deleteVertex(sel.l, sel.i);
    else if (op === 'split' && sel) splitAt(sel.l, sel.i);
    else if (op === 'extend' && sel) { startExtend(sel.l, endOf(sel.l, sel.i)); return; }
    else if (op === 'del-e' && sel) deleteEdge(sel.l, sel.i);
    else if (op === 'mid' && sel) insertMid(sel.l, sel.i);
    paint();
  }

  /* ---------- the bar ---------- */

  function say(text, bad) {
    el('shape-state').textContent = text;
    el('shape-bar').classList.toggle('bad', !!bad);
  }

  function paintBar() {
    const len = total();
    el('shape-len').textContent = len >= 1000 ? (len / 1000).toFixed(2) + ' ק"מ' : len + ' מ׳';
    const pieces = s.lines.filter((l) => l.length > 1).length;

    let state;
    if (s.busy) state = s.busy;
    else if (s.extend) {
      state = s.lines.length > 1
        ? 'לחץ על המפה כדי להמשיך את הקו. לחיצה על קצה של חלק אחר תחבר ביניהם.'
        : 'לחץ על המפה כדי להמשיך את הקו.';
    } else if (!pieces) state = 'הקו נמחק כולו. צייר מחדש, או בטל.';
    else if (s.sel && s.sel.t === 'v') state = 'נקודה נבחרה. גרור אותה, או בחר פעולה.';
    else if (s.sel) state = 'קטע נבחר.';
    else if (s.dense) state = 'התקרב כדי לראות את הנקודות ולגרור אותן.';
    else if (pieces > 1) state = `השביל מחולק ל-${pieces} חלקים. גרור קצה אל קצה כדי לחבר.`;
    else state = 'גרור נקודה כדי להזיז אותה, או ⊕ כדי להוסיף. לחיצה על נקודה או קטע: מחיקה ועוד.';
    if (!s.busy) say(state);

    el('shape-undo').disabled = !s.undo.length;
    el('shape-redo').disabled = !s.redo.length;
    el('shape-save').disabled = !!s.busy || !pieces || !changed();
    el('shape-reset').hidden = !changed();
    el('shape-new').hidden = !!pieces || !!s.extend;

    const n = points();
    // Not on every frame of a pan: only when the lines have changed.
    const key = s.undo.length + ':' + s.redo.length + ':' + n;
    if (s.simpKey !== key) {
      s.simpKey = key;
      s.fewer = s.lines.reduce((k, l) => k + simplify(l, SIMPLIFY_M).length, 0);
    }
    const fewer = s.fewer;
    const simp = el('shape-simplify');
    simp.hidden = n < 12 || fewer >= n;
    simp.textContent = `פשט: ${fewer} נקודות במקום ${n}`;
  }

  /* ---------- taps on the map ---------- */

  function nearestEdge(point) {
    let best = null;
    let bestD = EDGE_HIT;
    s.lines.forEach((line, l) => {
      const pp = line.map(px);
      for (let i = 0; i < pp.length - 1; i++) {
        const d = segDist(point, pp[i], pp[i + 1]);
        if (d < bestD) { bestD = d; best = { t: 'e', l, i }; }
      }
    });
    return best;
  }

  function extendTo(point, lngLat) {
    const ex = s.extend;
    const snap = snapAt(point, ex.line.length ? ex : null);
    if (snap && snap.join && ex.line.length) {
      const before = copy(s.lines);
      joinLines(ex.line, ex.end, snap.join.line, snap.join.end);
      remember(before);
      paint();
      return;
    }
    const pt = snap ? snap.pt.slice() : round(lngLat);
    edit(() => {
      if (ex.end === 'end') ex.line.push(pt);
      else ex.line.unshift(pt);
    });
    paint();
  }

  function onTap(point, lngLat) {
    if (!s || drag) return;
    if (s.extend) { extendTo(point, lngLat); return; }
    s.sel = nearestEdge(point);
    paint();
  }

  /** Taps bound by hand rather than through MapLibre's `click`, for the reason
   *  draft.js gives at onMapTap: `click` gives up after three pixels of drift,
   *  and a finger nearly always drifts more. */
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
      const point = { x: e.clientX - box.left, y: e.clientY - box.top };
      onTap(point, map.unproject([point.x, point.y]));
    };
    const abandon = () => { down = 0; start = null; };
    // The rubber band from the end being carried on to the pointer. Mouse
    // only: on a touch screen there is no pointer until the tap itself.
    const hover = (e) => {
      if (!s || !s.extend || e.pointerType !== 'mouse' || e.buttons) return;
      const box = canvas.getBoundingClientRect();
      s.hover = { x: e.clientX - box.left, y: e.clientY - box.top };
      schedulePaint();
    };
    canvas.addEventListener('pointerdown', press);
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', abandon);
    canvas.addEventListener('pointermove', hover);
    return () => {
      canvas.removeEventListener('pointerdown', press);
      canvas.removeEventListener('pointerup', release);
      canvas.removeEventListener('pointercancel', abandon);
      canvas.removeEventListener('pointermove', hover);
    };
  }

  /* ---------- dragging a point or a ⊕ ----------
   *
   * Listened for on window rather than on the handle, because the handles are
   * redrawn on every frame of the drag and a listener on one would be thrown
   * away with it. The edge scrolling is arrange.js's, for the same reason it
   * exists there: the place a point belongs is often just off the screen. */

  function onDown(e) {
    const g = e.target.closest('.sh-v, .sh-mid');
    if (!g || !s || drag || (e.button != null && e.button > 0)) return;
    e.preventDefault();
    e.stopPropagation();
    drag = {
      id: e.pointerId,
      kind: g.classList.contains('sh-mid') ? 'm' : 'v',
      l: +g.dataset.l,
      i: +g.dataset.i,
      from: { x: e.clientX, y: e.clientY },
      pointer: { x: e.clientX, y: e.clientY },
      touch: e.pointerType === 'touch',
      moved: false,
      before: null,
      last: 0
    };
    map.dragPan.disable();
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onCancel);
    raf = requestAnimationFrame(tick);
  }

  function onMove(e) {
    if (!drag || e.pointerId !== drag.id) return;
    drag.pointer = { x: e.clientX, y: e.clientY };
    if (!drag.moved) {
      const far = Math.hypot(e.clientX - drag.from.x, e.clientY - drag.from.y);
      if (far < (drag.touch ? 7 : 3)) return;
      drag.moved = true;
      drag.before = copy(s.lines);
      if (drag.kind === 'm') {
        const line = s.lines[drag.l];
        line.splice(drag.i + 1, 0, line[drag.i].slice());
        drag.i += 1;
        drag.kind = 'v';
      }
      s.sel = { t: 'v', l: drag.l, i: drag.i };
      s.extend = null;
      document.body.classList.add('sh-dragging');
    }
    follow();
  }

  /** Put the point wherever the pointer is pointing at the ground, or on the
   *  end it is close enough to snap to. */
  function follow() {
    const rect = map.getCanvas().getBoundingClientRect();
    const point = { x: drag.pointer.x - rect.left, y: drag.pointer.y - rect.top };
    const end = endOf(drag.l, drag.i);
    const line = s.lines[drag.l];
    const snap = end ? snapAt(point, { line, end }) : null;
    s.snap = snap;
    line[drag.i] = snap ? snap.pt.slice() : round(map.unproject([point.x, point.y]));
    paint();
  }

  function usable() {
    const canvas = map.getCanvas().getBoundingClientRect();
    const bar = el('shape-bar').getBoundingClientRect();
    const panel = el('panel').getBoundingClientRect();
    const overlapsX = panel.right > canvas.left + 1 && panel.left < canvas.right - 1;
    return {
      left: canvas.left,
      right: canvas.right,
      top: Math.max(canvas.top, bar.bottom),
      bottom: overlapsX ? Math.min(canvas.bottom, panel.top) : canvas.bottom
    };
  }

  function tick() {
    if (!drag) return;
    if (drag.moved) {
      const box = usable();
      const now = performance.now();
      const dt = Math.min(64, now - (drag.last || now));
      drag.last = now;
      const push = (v, lo, hi) => {
        if (v < lo + EDGE) return v - (lo + EDGE);
        if (v > hi - EDGE) return v - (hi - EDGE);
        return 0;
      };
      const speed = (d) => Math.max(-PAN_MAX, Math.min(PAN_MAX, d * 0.32)) * (dt / 16.7);
      const dx = push(drag.pointer.x, box.left, box.right);
      const dy = push(drag.pointer.y, box.top, box.bottom);
      if (dx || dy) {
        map.panBy([speed(dx), speed(dy)], { duration: 0, animate: false });
        follow();
      }
    }
    raf = requestAnimationFrame(tick);
  }

  function release() {
    cancelAnimationFrame(raf);
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onCancel);
    document.body.classList.remove('sh-dragging');
    map.dragPan.enable();
  }

  function onCancel(e) {
    if (!drag || e.pointerId !== drag.id) return;
    release();
    if (drag.moved && drag.before) s.lines = drag.before;
    drag = null;
    s.snap = null;
    paint();
  }

  function onUp(e) {
    if (!drag || e.pointerId !== drag.id) return;
    release();
    const d = drag;
    drag = null;
    const snap = s.snap;
    s.snap = null;

    if (d.moved) {
      const line = s.lines[d.l];
      if (snap && snap.join) joinLines(line, endOf(d.l, d.i), snap.join.line, snap.join.end);
      // A drag is a move and nothing more. Leaving the point selected put
      // its menu up over the very ⊕ the next drag was reaching for.
      s.sel = null;
      remember(d.before);
      paint();
      return;
    }

    // A tap on a handle.
    if (d.kind === 'm') { insertMid(d.l, d.i); paint(); return; }
    const end = endOf(d.l, d.i);
    if (s.extend) {
      const line = s.lines[d.l];
      // The end being carried on: that is "done".
      if (line === s.extend.line && end === s.extend.end) { stopExtend(); return; }
      // Another piece's end: join to it.
      if (end && line !== s.extend.line && s.extend.line.length) {
        const before = copy(s.lines);
        joinLines(s.extend.line, s.extend.end, line, end);
        remember(before);
        paint();
        return;
      }
      // Anything else: a point there, as if the map had been tapped.
      extendTo(px(line[d.i]), { lat: line[d.i][0], lng: line[d.i][1] });
      return;
    }
    const was = s.sel && s.sel.t === 'v' && s.sel.l === d.l && s.sel.i === d.i;
    s.sel = was ? null : { t: 'v', l: d.l, i: d.i };
    paint();
  }

  /* ---------- keys ---------- */

  function onKey(e) {
    if (!s || drag) return;
    if (e.target.closest && e.target.closest('input, textarea, select, [contenteditable]')) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
    if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
    if (e.key === 'Escape') {
      e.stopPropagation();
      if (s.extend) stopExtend();
      else if (s.sel) { s.sel = null; paint(); }
      return;
    }
    if (e.key === 'Enter' && s.extend) { stopExtend(); return; }
    if ((e.key === 'Delete' || e.key === 'Backspace') && s.sel) {
      e.preventDefault();
      if (s.sel.t === 'v') deleteVertex(s.sel.l, s.sel.i);
      else deleteEdge(s.sel.l, s.sel.i);
      paint();
    }
  }

  /* ---------- surviving a reload ----------
   *
   * Same promise as the drafts editor (see "surviving a reload" in draft.js):
   * fixing a long trail is many small moves, and a locked phone should not
   * cost them. Mirrored on every change, offered back on the next load. */

  function saveLive() {
    try {
      if (!s) { localStorage.removeItem(LIVE); return; }
      localStorage.setItem(LIVE, JSON.stringify({
        kind: s.kind, id: s.id, name: s.name, lines: s.lines, at: Date.now()
      }));
    } catch (err) {
      /* private mode or a full quota: the editor still works, unmirrored */
    }
  }

  let offered = false;

  /** Called once the drafts are loaded and again when the worker confirms an
   *  editor, since a published trail can only be reopened by one. Asks once. */
  function restore() {
    if (offered || s || typeof map === 'undefined' || !map || !map.getSource('src-trails')) return;
    let rec = null;
    try {
      rec = JSON.parse(localStorage.getItem(LIVE) || 'null');
    } catch (err) { /* nothing worth offering */ }
    if (!rec || !rec.id || !Array.isArray(rec.lines)) return;
    if (Date.now() - (rec.at || 0) > LIVE_MAX_MS) { localStorage.removeItem(LIVE); return; }
    const it = Layers.item(rec.id);
    if (!it || !it.path || (rec.kind === 'trail' && !Store.isEditor())) return;
    offered = true;
    if (!confirm(`נשארה עריכה של התוואי של "${rec.name}" שלא נשמרה.\n\nלהמשיך מאיפה שהפסקת?`)) {
      localStorage.removeItem(LIVE);
      return;
    }
    start(rec.kind, it, rec.lines);
  }

  /* ---------- in and out ---------- */

  /** May this item's line be edited here? A draft by anybody, a published
   *  shortcut by an editor. A trip is a recipe of other trails and is edited
   *  by re-chaining them, not here. */
  function canShape(it) {
    if (!it || it.trip || it.pending || !it.path || it.path.length < 2) return false;
    if (it.draft) return true;
    const layer = Layers.layerOf(it.id);
    return !!layer && layer.kind === 'trails' && layer.id !== 'drafts' && Store.isEditor();
  }

  function open(it) {
    if (!map) { alert('עריכת תוואי דורשת את המפה, והדפדפן הזה לא מציג אותה.'); return; }
    if (!canShape(it)) return;
    start(it.draft ? 'draft' : 'trail', it, [it.path]);
  }

  function start(kind, it, lines) {
    if (s) close(true);
    stopNav();
    Arrange.close(true);
    if (Drafts.isDrafting()) Drafts.stop();
    if (Route.isOn()) Route.close();
    deselect();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'shape-svg');
    svg.addEventListener('pointerdown', onDown);
    const pop = document.createElement('div');
    pop.className = 'shape-pop';
    pop.hidden = true;
    pop.addEventListener('click', onPop);
    pop.addEventListener('pointerdown', (e) => e.stopPropagation());
    host().append(svg, pop);

    s = {
      kind,
      id: it.id,
      name: it.name,
      orig: it.path.map((p) => p.slice()),
      lines: copy(lines).filter((l) => l.length > 1),
      sel: null,
      extend: null,
      hover: null,
      snap: null,
      dense: false,
      busy: '',
      undo: [],
      redo: [],
      pitch: map.getPitch(),
      svg,
      pop,
      offTaps: bindTaps()
    };

    el('shape-name').textContent = it.name;
    document.body.classList.add('shaping');
    el('shape-bar').hidden = false;
    Layers.setShaping(it.id);
    map.doubleClickZoom.disable();
    map.on('move', paint);
    window.addEventListener('keydown', onKey, true);
    saveLive();

    // Flat, and framed. A point dragged on a map tilted to 58 degrees moves
    // ten metres per pixel at the top of the screen and one at the bottom.
    const b = new maplibregl.LngLatBounds();
    [it.path, ...s.lines].forEach((l) => l.forEach(([lat, lng]) => b.extend([lng, lat])));
    map.jumpTo({ pitch: 0 });
    const top = el('shape-bar').getBoundingClientRect().height + 40;
    map.fitBounds(b, { padding: { top, bottom: 70, left: 40, right: 40 }, maxZoom: 18.5, duration: 500 });
    paint();
  }

  function close(force) {
    if (!s) return true;
    if (!force && changed() && !confirm('לצאת בלי לשמור את השינויים בתוואי?')) return false;
    if (drag) { release(); drag = null; }
    s.offTaps();
    s.svg.remove();
    s.pop.remove();
    map.off('move', paint);
    map.doubleClickZoom.enable();
    window.removeEventListener('keydown', onKey, true);
    const pitch = s.pitch;
    s = null;
    saveLive();                       // no session, so this drops the mirror
    Layers.setShaping(null);
    document.body.classList.remove('shaping');
    el('shape-bar').hidden = true;
    el('shape-bar').classList.remove('bad');
    if (pitch) map.easeTo({ pitch, duration: 400 });
    return true;
  }

  async function save() {
    if (!s || s.busy) return;
    stopExtend();
    const pieces = s.lines.filter((l) => l.length > 1);
    if (!pieces.length) return;
    if (!changed()) { close(true); return; }
    if (pieces.length > 1 && !confirm(
      `השביל מחולק עכשיו ל-${pieces.length} חלקים, וכל חלק יישמר כשביל נפרד.\n`
      + `הארוך יישאר "${s.name}" עם התמונות והקישורים, והשאר ייקראו "${s.name} (2)" וכן הלאה.\n\n`
      + 'לשמור כך? (אפשר גם לבטל ולחבר את החלקים: גרור קצה אל קצה.)')) return;

    const { kind, id } = s;
    s.busy = kind === 'trail' ? 'שומר במסד המשותף…' : 'שומר…';
    say(s.busy);
    paintBar();
    try {
      if (kind === 'trail') await reloadShared(await Store.reshape(id, pieces, s.name));
      else await Drafts.reshape(id, pieces);
      close(true);
      select(id);
    } catch (err) {
      if (!s) return;
      s.busy = '';
      paintBar();
      say('לא נשמר: ' + err.message + ' השינויים עדיין כאן.', true);
    }
  }

  function wire() {
    el('shape-save').addEventListener('click', save);
    el('shape-cancel').addEventListener('click', () => close());
    el('shape-undo').addEventListener('click', undo);
    el('shape-redo').addEventListener('click', redo);
    el('shape-reset').addEventListener('click', () => {
      if (!s) return;
      edit(() => { s.lines = [s.orig.map((p) => p.slice())]; });
      s.sel = null;
      s.extend = null;
      paint();
    });
    el('shape-simplify').addEventListener('click', () => {
      if (!s) return;
      edit(() => { s.lines = s.lines.map((l) => simplify(l, SIMPLIFY_M)); });
      s.sel = null;
      paint();
    });
    el('shape-new').addEventListener('click', () => { if (s) drawNew(); });
  }

  return { open, close, wire, restore, canShape, isOn: () => !!s,
           kind: () => (s ? s.kind : null),
           // for tests
           _state: () => s, _simplify: simplify };
})();
