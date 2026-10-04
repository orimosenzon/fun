/* Walking directions from anywhere to anywhere, through the shortcuts.
 *
 * Ori, 3/10/2026. Until now the app could take you *to a trail*. The question
 * somebody actually has is "how do I get from here to the kindergarten", and
 * nobody else can answer it: 83% of these shortcuts are on no map Google has,
 * so Google routes around them. This app has both halves - the streets, from
 * OpenStreetMap (data/walknet.json, built by build_walknet.py), and the
 * shortcuts, which residents walked and sent in - so it can route through
 * the whole of it, here in the browser, with no service and no key.
 *
 * Every route is worked out twice, once with the shortcuts and once without,
 * and the difference is the point: "12 minutes, and 7 of them saved". The
 * route without them is drawn too, thin and grey, so you see what you are
 * cutting.
 *
 * Two parts. `RouteEngine` is the graph and Dijkstra, pure and DOM-free so a
 * test can drive it under node. `Route` is the screen: picking the two ends,
 * the bar, the line on the map, the detail pane, the link.
 */
'use strict';

const RouteEngine = (() => {

  const WALK_M_PER_MIN = 80;        // 4.8 km/h, an unhurried adult
  const CELL = 60;                  // spatial grid, metres
  const TRAIL_SNAP = 80;            // how far a trail's end may be from the network
  const POINT_SNAP = 400;           // how far a chosen point may be from any path

  // Edge kinds. A link is the bit of nothing between a trail's recorded end and
  // the street it comes out on: GPS starts recording a step or two in, and the
  // trail is unusable without it. It belongs to the trail, so it is left out of
  // the without-shortcuts route along with the trail itself.
  const STREET = 0, TRAIL = 1, LINK = 2, OFF = 3;

  /** walknet.json -> plain arrays, once. */
  function decode(net) {
    const n = net.pts.length / 2;
    const lat = new Float64Array(n), lng = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      lat[i] = net.o[0] + net.pts[2 * i] / net.scale;
      lng[i] = net.o[1] + net.pts[2 * i + 1] / net.scale;
    }
    let s = 90, w = 180, nn = -90, e = -180;
    for (let i = 0; i < n; i++) {
      if (lat[i] < s) s = lat[i]; if (lat[i] > nn) nn = lat[i];
      if (lng[i] < w) w = lng[i]; if (lng[i] > e) e = lng[i];
    }
    return { lat, lng, ways: net.ways, names: net.names, bounds: [[s, w], [nn, e]] };
  }

  /** The graph for one question: the streets, every shortcut, and the two
   *  points asked about, all joined up.
   *
   *  Built fresh for every route. It takes a few tens of milliseconds, and in
   *  exchange there is no state to keep in step with trails being added,
   *  approved or switched off while the app is open. */
  function build(base, trails, ends) {
    const lat0 = base.lat[0] || 32.47;
    const KX = 111320 * Math.cos(lat0 * Math.PI / 180), KY = 111320;
    const X = [], Y = [];
    const node = (la, ln) => { X.push((ln - base.lng[0]) * KX); Y.push((la - base.lat[0]) * KY); return X.length - 1; };
    const toLatLng = (i) => [base.lat[0] + Y[i] / KY, base.lng[0] + X[i] / KX];

    for (let i = 0; i < base.lat.length; i++) node(base.lat[i], base.lng[i]);

    const eA = [], eB = [], eKind = [], eRef = [];
    const edge = (a, b, kind, ref) => {
      if (a === b) return -1;
      eA.push(a); eB.push(b); eKind.push(kind); eRef.push(ref);
      return eA.length - 1;
    };

    for (const w of base.ways) {
      for (let k = 2; k < w.length; k++) edge(w[k - 1], w[k], STREET, w[0]);
    }
    const trailEnds = [];
    trails.forEach((t, ti) => {
      if (!t.path || t.path.length < 2) return;
      let prev = node(t.path[0][0], t.path[0][1]);
      const first = prev;
      for (let k = 1; k < t.path.length; k++) {
        const cur = node(t.path[k][0], t.path[k][1]);
        edge(prev, cur, TRAIL, ti);
        prev = cur;
      }
      trailEnds.push([first, ti], [prev, ti]);
    });

    // The grid: every edge, in every cell its bounding box touches.
    const grid = new Map();
    const key = (cx, cy) => cx * 1e6 + cy;
    const cellsOf = (x0, y0, x1, y1, fn) => {
      for (let cx = Math.floor(Math.min(x0, x1) / CELL); cx <= Math.floor(Math.max(x0, x1) / CELL); cx++) {
        for (let cy = Math.floor(Math.min(y0, y1) / CELL); cy <= Math.floor(Math.max(y0, y1) / CELL); cy++) fn(key(cx, cy));
      }
    };
    const fixed = eA.length;
    for (let e = 0; e < fixed; e++) {
      cellsOf(X[eA[e]], Y[eA[e]], X[eB[e]], Y[eB[e]], (k) => {
        const c = grid.get(k);
        if (c) c.push(e); else grid.set(k, [e]);
      });
    }

    /** Nearest point on any edge `ok` accepts, within `max` metres. */
    function nearest(x, y, max, ok) {
      let best = null;
      const seen = new Set();
      cellsOf(x - max, y - max, x + max, y + max, (k) => {
        const c = grid.get(k);
        if (!c) return;
        for (const e of c) {
          if (seen.has(e) || !ok(e)) continue;
          seen.add(e);
          const ax = X[eA[e]], ay = Y[eA[e]], bx = X[eB[e]], by = Y[eB[e]];
          const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
          const t = len ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len)) : 0;
          const d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
          if (d <= max && (!best || d < best.d)) best = { e, t, d, x: ax + t * dx, y: ay + t * dy };
        }
      });
      return best;
    }

    // Points that land part-way along an edge. Collected first and stitched in
    // at the end, so that two of them on one long street are joined to each
    // other and not only to the street's two ends.
    const splits = new Map();
    function splitAt(hit) {
      if (hit.t < 1e-6) return eA[hit.e];
      if (hit.t > 1 - 1e-6) return eB[hit.e];
      const list = splits.get(hit.e) || [];
      for (const s of list) if (Math.abs(s.t - hit.t) < 1e-6) return s.n;
      X.push(hit.x); Y.push(hit.y);
      const n = X.length - 1;
      list.push({ t: hit.t, n });
      splits.set(hit.e, list);
      return n;
    }

    // Where a shortcut crosses a street on the way, it can be joined there:
    // the circular trail crosses a dozen roads, and only its two ends would
    // otherwise connect.
    for (let e = 0; e < fixed; e++) {
      if (eKind[e] !== TRAIL) continue;
      const ax = X[eA[e]], ay = Y[eA[e]], bx = X[eB[e]], by = Y[eB[e]];
      const seen = new Set();
      cellsOf(ax, ay, bx, by, (k) => {
        for (const f of grid.get(k) || []) {
          if (seen.has(f) || eKind[f] !== STREET) continue;
          seen.add(f);
          const cx = X[eA[f]], cy = Y[eA[f]], dx = X[eB[f]], dy = Y[eB[f]];
          const r1 = bx - ax, r2 = by - ay, s1 = dx - cx, s2 = dy - cy;
          const den = r1 * s2 - r2 * s1;
          if (Math.abs(den) < 1e-9) continue;
          const t = ((cx - ax) * s2 - (cy - ay) * s1) / den;
          const u = ((cx - ax) * r2 - (cy - ay) * r1) / den;
          if (t <= 0 || t >= 1 || u <= 0 || u >= 1) continue;
          const at = splitAt({ e: f, t: u, x: ax + t * r1, y: ay + t * r2 });
          const list = splits.get(e) || [];
          list.push({ t, n: at });
          splits.set(e, list);
        }
      });
    }

    // Each end of each shortcut, onto the nearest thing that is not itself.
    for (const [n, ti] of trailEnds) {
      const hit = nearest(X[n], Y[n], TRAIL_SNAP,
        (e) => !(eKind[e] === TRAIL && eRef[e] === ti));
      if (hit) edge(n, splitAt(hit), LINK, ti);
    }

    // The two points asked about. Each gets two landings: the nearest path of
    // any kind, for the route with the shortcuts, and the nearest street, for
    // the one without - otherwise a destination at the mouth of a shortcut has
    // no without-shortcuts route at all.
    const asked = ends.map((p) => {
      const x = (p.lng - base.lng[0]) * KX, y = (p.lat - base.lat[0]) * KY;
      const any = nearest(x, y, POINT_SNAP, () => true);
      const street = nearest(x, y, POINT_SNAP, (e) => eKind[e] === STREET);
      if (!any) return null;
      const n = node(p.lat, p.lng);
      edge(n, splitAt(any), OFF, -1);
      if (street && street.e !== any.e) edge(n, splitAt(street), OFF, -1);
      return n;
    });

    for (const [e, list] of splits) {
      list.sort((a, b) => a.t - b.t);
      let prev = eA[e];
      for (const s of list) { edge(prev, s.n, eKind[e], eRef[e]); prev = s.n; }
      edge(prev, eB[e], eKind[e], eRef[e]);
    }

    // Adjacency, compressed: every edge both ways.
    const N = X.length, E = eA.length;
    const len = new Float64Array(E);
    const deg = new Int32Array(N + 1);
    for (let e = 0; e < E; e++) {
      len[e] = Math.hypot(X[eB[e]] - X[eA[e]], Y[eB[e]] - Y[eA[e]]);
      deg[eA[e] + 1]++; deg[eB[e] + 1]++;
    }
    for (let i = 0; i < N; i++) deg[i + 1] += deg[i];
    const adj = new Int32Array(2 * E), fill = deg.slice(0, N);
    for (let e = 0; e < E; e++) { adj[fill[eA[e]]++] = e; adj[fill[eB[e]]++] = e; }

    return { N, X, Y, eA, eB, eKind, eRef, len, off: deg, adj, toLatLng, from: asked[0], to: asked[1] };
  }

  /** Cheapest path by Dijkstra, as a list of edges from `from` to `to`.
   *  `ok(kind)` says which edges may be used; `cost(e)`, when given, is what
   *  an edge costs instead of its length. The returned `length` is always the
   *  real distance walked, whatever the costs were. */
  function shortest(g, ok, cost) {
    if (g.from == null || g.to == null) return null;
    const dist = new Float64Array(g.N).fill(Infinity);
    const via = new Int32Array(g.N).fill(-1);
    // A binary heap of [distance, node] pairs, laid out flat.
    const hd = [], hn = [];
    const push = (d, n) => {
      let i = hd.length; hd.push(d); hn.push(n);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (hd[p] <= hd[i]) break;
        [hd[p], hd[i]] = [hd[i], hd[p]]; [hn[p], hn[i]] = [hn[i], hn[p]]; i = p;
      }
    };
    const pop = () => {
      const d = hd[0], n = hn[0], ld = hd.pop(), ln = hn.pop();
      if (hd.length) {
        hd[0] = ld; hn[0] = ln;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < hd.length && hd[l] < hd[m]) m = l;
          if (r < hd.length && hd[r] < hd[m]) m = r;
          if (m === i) break;
          [hd[m], hd[i]] = [hd[i], hd[m]]; [hn[m], hn[i]] = [hn[i], hn[m]]; i = m;
        }
      }
      return [d, n];
    };
    dist[g.from] = 0;
    push(0, g.from);
    while (hd.length) {
      const [d, n] = pop();
      if (d > dist[n]) continue;
      if (n === g.to) break;
      for (let k = g.off[n]; k < g.off[n + 1]; k++) {
        const e = g.adj[k];
        if (!ok(g.eKind[e])) continue;
        const m = g.eA[e] === n ? g.eB[e] : g.eA[e];
        const nd = d + (cost ? cost(e) : g.len[e]);
        if (nd < dist[m]) { dist[m] = nd; via[m] = e; push(nd, m); }
      }
    }
    if (!isFinite(dist[g.to])) return null;
    const edges = [];
    let length = 0;
    for (let n = g.to; n !== g.from;) {
      const e = via[n];
      edges.push(e);
      length += g.len[e];
      n = g.eA[e] === n ? g.eB[e] : g.eA[e];
    }
    edges.reverse();
    return { edges, length };
  }

  // Preferring the shortcuts (Ori, 4/10/2026: "use the shortcuts as much as
  // possible - plain walking directions are what Google already gives"). A
  // metre of shortcut is made to cost less than a metre of street, so the
  // route bends toward them. How much less is tried from the strongest
  // preference down, and the first one whose route is no more than DETOUR
  // longer than the plain shortest walk is kept: a shortcut is worth a small
  // detour, not a walk around the block for its own sake.
  const TRAIL_WEIGHTS = [0.45, 0.6, 0.75, 0.9, 1];
  const DETOUR = 1.25, DETOUR_MIN_M = 200;

  // Alternatives, by the penalty method: every edge of a route already found
  // is made dearer and the search runs again, which pushes it onto other
  // streets and other shortcuts. A candidate is kept if it is not much longer
  // than the best route and mostly does not walk the same ground as any route
  // already kept.
  const ALT_PENALTY = 1.8, ALT_TRIES = 8, ALT_MAX = 3;
  const ALT_LONGER = 1.4, ALT_OVERLAP = 0.7;

  const isTrail = (k) => k === TRAIL || k === LINK;
  const capOf = (plain) => Math.max(plain * DETOUR, plain + DETOUR_MIN_M);

  /** The recommended route: the most shortcut-loving one within the detour. */
  function preferred(g) {
    const plain = shortest(g, () => true);
    if (!plain) return null;
    const cap = capOf(plain.length);
    for (const w of TRAIL_WEIGHTS) {
      const cost = (e) => (isTrail(g.eKind[e]) ? g.len[e] * w : g.len[e]);
      const p = shortest(g, () => true, cost);
      if (p && p.length <= cap) return { path: p, weight: w, plain };
    }
    return { path: plain, weight: 1, plain };
  }

  /** Up to ALT_MAX other good ways, each different enough from the rest. */
  function alternatives(g, best, weight, plain) {
    const penalty = new Float64Array(g.len.length).fill(1);
    const kept = [best];
    const sets = [new Set(best.edges)];
    const limit = Math.min(best.length * ALT_LONGER, capOf(plain.length) * 1.15);
    const hit = (p) => p.edges.forEach((e) => { if (g.eKind[e] !== OFF) penalty[e] *= ALT_PENALTY; });
    hit(best);
    for (let i = 0; i < ALT_TRIES && kept.length <= ALT_MAX; i++) {
      const cost = (e) => g.len[e] * penalty[e] * (isTrail(g.eKind[e]) ? weight : 1);
      const p = shortest(g, () => true, cost);
      if (!p) break;
      hit(p);
      if (p.length > limit) continue;
      const shared = (set) => p.edges.reduce((s, e) => s + (set.has(e) ? g.len[e] : 0), 0);
      if (sets.some((set) => shared(set) > ALT_OVERLAP * p.length)) continue;
      kept.push(p);
      sets.push(new Set(p.edges));
    }
    return kept.slice(1).sort((a, b) => a.length - b.length);
  }

  /** A path of edges as legs: runs of one street, one shortcut, or open ground. */
  function legs(g, path, base, trails) {
    const out = [];
    let at = g.from;
    for (const e of path.edges) {
      const next = g.eA[e] === at ? g.eB[e] : g.eA[e];
      const kind = g.eKind[e];
      const leg = kind === STREET
        ? { kind: 'street', name: base.names[g.eRef[e]] || '' }
        : kind === OFF ? { kind: 'off', name: '' }
          : { kind: 'trail', name: trails[g.eRef[e]].name, id: trails[g.eRef[e]].id };
      const last = out[out.length - 1];
      const same = last && last.kind === leg.kind && last.name === leg.name && last.id === leg.id;
      if (same) {
        last.length += g.len[e];
        last.path.push(g.toLatLng(next));
      } else {
        out.push({ ...leg, length: g.len[e], path: [g.toLatLng(at), g.toLatLng(next)] });
      }
      at = next;
    }
    // A short street with no name is usually the last metres of a junction or
    // a service road. Folded into the street before it, so the list names the
    // turns that matter. Open ground and shortcuts always stay their own leg.
    const folded = [];
    for (const l of out) {
      const prev = folded[folded.length - 1];
      if (prev && prev.kind === 'street' && l.kind === 'street' && (!l.name && l.length < 60 || l.name === prev.name)) {
        prev.length += l.length;
        prev.path.push(...l.path.slice(1));
      } else folded.push(l);
    }
    return folded;
  }

  /** Turn-by-turn steps from the legs (Ori, 4/10/2026: a dot to walk toward
   *  was not navigation). Each step is where a leg begins: how far along the
   *  route that is, which way to turn there, and onto what. The bits of open
   *  ground at either end are not steps of their own. */
  function steps(L) {
    const M = 111320;
    const vec = (a, b) => [(b[1] - a[1]) * M * Math.cos(a[0] * Math.PI / 180), (b[0] - a[0]) * M];
    const bearing = (a, b) => { const [x, y] = vec(a, b); return Math.atan2(x, y) * 180 / Math.PI; };
    const lenOf = (a, b) => Math.hypot(...vec(a, b));
    // A point about `m` metres from one end of a path, to take a bearing over
    // the stretch you actually see rather than its last kink.
    const reach = (path, m, fromEnd) => {
      const pts = fromEnd ? path.slice().reverse() : path;
      let left = m;
      for (let i = 1; i < pts.length; i++) {
        const d = lenOf(pts[i - 1], pts[i]);
        if (d >= left || i === pts.length - 1) return pts[i];
        left -= d;
      }
      return pts[pts.length - 1];
    };
    const out = [];
    let at = 0, prev = null;
    L.forEach((l, i) => {
      // A few metres of street between two turns is a jog, not a step of its
      // own: "slight left onto X, then 30 m on, left again" is noise.
      const jog = l.kind === 'street' && l.length < 35 && prev && L.slice(i + 1).some((n) => n.kind !== 'off');
      if (jog) out[out.length - 1].length += l.length;
      else if (l.kind !== 'off') {
        const step = { at, kind: l.kind, name: l.name, id: l.id, pt: l.path[0], length: l.length, turn: 'start', angle: 0 };
        if (prev) {
          const inB = bearing(reach(prev.path, 20, true), prev.path[prev.path.length - 1]);
          const outB = bearing(l.path[0], reach(l.path, 20, false));
          const d = ((outB - inB + 540) % 360) - 180;     // + is clockwise: a right turn
          step.angle = d;
          const a = Math.abs(d), side = d > 0 ? 'right' : 'left';
          step.turn = a < 25 ? 'straight' : a < 60 ? `slight-${side}` : a < 150 ? side : 'back';
        }
        out.push(step);
        prev = l;
      } else if (prev) out[out.length - 1].length += l.length;
      at += l.length;
    });
    // Straight on along the same street, after a jog was folded away, is the
    // step before it continuing.
    const merged = [];
    for (const st of out) {
      const last = merged[merged.length - 1];
      if (last && st.turn === 'straight' && st.kind === last.kind && st.name === last.name) last.length += st.length;
      else merged.push(st);
    }
    merged.push({ at, kind: 'end', name: '', pt: L[L.length - 1].path.slice(-1)[0], length: 0, turn: 'end', angle: 0 });
    return merged;
  }

  /** A step in words, in two halves: the turn ("פנה ימינה") and what it is
   *  onto ("לשביל כיכר פעם"). The navigation bar puts them on two lines. */
  const TURN = { start: 'צא', straight: 'המשך ישר', 'slight-right': 'פנה קלות ימינה',
                 'slight-left': 'פנה קלות שמאלה', right: 'פנה ימינה', left: 'פנה שמאלה',
                 back: 'הסתובב', end: 'הגעת ליעד' };
  const turnWords = (s) => TURN[s.turn];
  // A name that already says what it is ("שביל השבילים", "רחוב הנשיא") is
  // not given the word a second time.
  const onto = (s) => (s.kind === 'end' ? ''
    : /^(שביל|רחוב|דרך|שדרות|סמטת|כביש)\s/.test(s.name) ? `ל${s.name}`
      : s.kind === 'trail' ? `לשביל ${s.name}`
        : s.name ? `לרחוב ${s.name}` : 'לדרך בלי שם');
  const say = (s) => `${turnWords(s)} ${onto(s)}`.trim();

  /** The whole answer: the recommended route through the shortcuts, a few
   *  alternatives, and the streets-only walk to compare against. */
  function route(base, trails, from, to) {
    const g = build(base, trails, [from, to]);
    if (g.from == null) return { error: 'from' };
    if (g.to == null) return { error: 'to' };
    const best = preferred(g);
    if (!best) return { error: 'none' };
    const without = shortest(g, (k) => k === STREET || k === OFF);
    const pathOf = (p) => {
      const pts = [g.toLatLng(g.from)];
      let at = g.from;
      for (const e of p.edges) { at = g.eA[e] === at ? g.eB[e] : g.eA[e]; pts.push(g.toLatLng(at)); }
      return pts;
    };
    const pack = (p) => {
      const L = legs(g, p, base, trails);
      const T = L.filter((l) => l.kind === 'trail');
      return {
        length: p.length,
        minutes: p.length / WALK_M_PER_MIN,
        path: pathOf(p),
        legs: L,
        steps: steps(L),
        trailLength: T.reduce((s, l) => s + l.length, 0),
        trailsUsed: [...new Set(T.map((l) => l.id))]
      };
    };
    return {
      ...pack(best.path),
      alternatives: alternatives(g, best.path, best.weight, best.plain).map(pack),
      without: without && { length: without.length, minutes: without.length / WALK_M_PER_MIN,
                            path: pathOf(without) }
    };
  }

  /** The same answer with alternative `i` as the recommended route. */
  function choose(r, i) {
    const alts = r.alternatives.slice();
    const [picked] = alts.splice(i, 1);
    if (!picked) return r;
    const { alternatives: _a, without, ...current } = r;
    alts.push(current);
    alts.sort((a, b) => a.length - b.length);
    return { ...picked, alternatives: alts, without };
  }

  return { decode, route, choose, say, turnWords, onto, WALK_M_PER_MIN };
})();

if (typeof module !== 'undefined') module.exports = { RouteEngine };

/* ---------- the screen ---------- */

const Route = (typeof document === 'undefined') ? null : (() => {
  const SRC = 'src-route';
  const LAYERS = ['route-alt-hit', 'route-alt', 'route-case', 'route-main', 'route-trail', 'route-off'];
  const BLUE = '#0d47a1', BLUE_FADED = '#5b8fd6', GOLD = '#ffc928';

  let base = null, loading = null;
  let from = null, to = null;      // {lat, lng, mine?}
  let picking = null;              // 'from' | 'to' | null
  let result = null;
  let markers = { from: null, to: null };

  function load() {
    if (base) return Promise.resolve(base);
    if (!loading) {
      loading = fetch('data/walknet.json')
        .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then((net) => (base = RouteEngine.decode(net)))
        .catch((err) => { loading = null; throw err; });
    }
    return loading;
  }

  /** The shortcuts a route may use: every published one, whether or not its
   *  layer is switched on - turning the gold lines off to see the map better
   *  should not make the directions worse - plus the circular trail. Never a
   *  private layer: those are the paths somebody has fenced off, and the one
   *  place they must not send anybody is through the fence. Never a draft or
   *  the queue, which nobody has checked yet. */
  function trails() {
    const layers = Layers.trailLayers().filter((l) => !l.private);
    const sovev = Layers.byId(Layers.SOVEV_ID);
    if (sovev) layers.push(sovev);
    return layers.flatMap((l) => l.segments).filter((s) => s.path && s.path.length > 1);
  }

  const bar = () => el('route-bar');
  const say = (strong, line) => {
    el('route-main').textContent = strong;
    el('route-sub').textContent = line;
  };
  const isOn = () => !bar().hidden;
  const isPicking = () => !!picking;

  /** Which buttons the bar shows, by what is wanted now. Measured after, so
   *  the map controls under the bar move down when it grows a second row. */
  function acts() {
    const ready = !!result && !picking;
    el('route-mine').hidden = picking !== 'from';
    el('route-go').hidden = !ready;
    el('route-from').hidden = !ready;
    el('route-to').hidden = !ready;
    el('route-swap').hidden = !ready;
    requestAnimationFrame(() => document.body.style.setProperty('--route-h', `${bar().offsetHeight}px`));
  }

  /** Ask for the start: a tap anywhere, or the button for where you are. */
  function askFrom() {
    picking = 'from';
    say('בחר נקודת יציאה (א)', to
      ? 'לחץ על המפה, או על "המיקום שלי".'
      : 'לחץ על המפה, או על "המיקום שלי". אחר כך בוחרים את היעד (ב).');
    acts();
  }

  function askTo() {
    picking = 'to';
    say('עכשיו לחץ על היעד (ב)', 'אפשר לגרור את שתי הסיכות אחר כך.');
    acts();
  }

  /** Both ends set: route, else ask for the one missing. */
  function next() {
    if (!from) askFrom();
    else if (!to) askTo();
    else { picking = null; acts(); run(); }
  }

  /** Open the bar and ask for the start, then the destination - or only the
   *  start, when a place's page already gave the destination.
   *
   *  The start is a choice (Ori, 4/10/2026): a tap anywhere on the map, or
   *  "המיקום שלי", which finds you by GPS. It used to be taken from the GPS
   *  on its own whenever you were in the moshava, which left no way to plan a
   *  walk from somewhere else. Taps go in reading order: א, then ב. */
  function open(dest) {
    stopNavIfOurs();
    from = to = result = null;
    clear();
    bar().hidden = false;
    document.body.classList.add('routing');
    load().catch(() => say('לא הצלחתי לטעון את מפת הרחובות', 'בדוק את החיבור ונסה שוב.'));
    if (dest) {
      to = { lat: dest.lat, lng: dest.lng, label: dest.label };
      paintMarker('to');
    }
    collapsePanel();
    askFrom();
  }

  /** "המיקום שלי": the GPS position as the start - if it is in or near the
   *  moshava, since a phone in Tel Aviv cannot walk to a kindergarten here. */
  function useMine() {
    const want = picking;
    say('מחפש את המיקום שלך…', 'אפשר גם פשוט ללחוץ על המפה.');
    const use = (p) => {
      if (!isOn() || picking !== want || want !== 'from') return;   // a tap got there first
      const [[s, w], [n, e]] = base ? base.bounds : [[32.43, 34.93], [32.52, 35.03]];
      if (!p) {
        say('לא הצלחתי למצוא את המיקום שלך', 'צריך לאשר גישה למיקום. בינתיים אפשר ללחוץ על המפה.');
        return;
      }
      if (!(p.lat > s && p.lat < n && p.lng > w && p.lng < e)) {
        say('אתה לא במושבה כרגע', 'לחץ על המפה במקום שממנו תצא.');
        return;
      }
      from = { lat: p.lat, lng: p.lng, mine: true };
      paintMarker('from');
      next();
    };
    if (!navigator.geolocation) { use(null); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => { here = { lat: pos.coords.latitude, lng: pos.coords.longitude }; drawMe(); use(here); },
      () => use(null),
      { enableHighAccuracy: true, maximumAge: 15000, timeout: 12000 });
  }

  /** "שנה את א" / "שנה את ב": the next tap moves that end. The route stays on
   *  the map until it does. */
  function repick(which) {
    stopNavIfOurs();
    if (which === 'from') askFrom();
    else askTo();
  }

  /** A tap on the map while the bar waits for one. */
  function pick(lngLat) {
    const p = { lat: lngLat.lat, lng: lngLat.lng };
    if (picking === 'from') { from = p; paintMarker('from'); }
    else if (picking === 'to') { to = p; paintMarker('to'); }
    else return;
    picking = null;
    next();
  }

  function paintMarker(which) {
    if (!map) return;
    const p = which === 'from' ? from : to;
    if (!markers[which]) {
      const node = document.createElement('div');
      node.className = `route-pin route-${which}`;
      node.textContent = which === 'from' ? 'א' : 'ב';
      const m = new maplibregl.Marker({ element: node, draggable: true });
      m.on('dragend', () => {
        const ll = m.getLngLat();
        const q = { lat: ll.lat, lng: ll.lng };
        if (which === 'from') from = q; else to = { ...q, label: '' };
        if (from && to) run();
      });
      markers[which] = m;
    }
    markers[which].setLngLat([p.lng, p.lat]).addTo(map);
  }

  async function run(keepCamera = false) {
    say('מחשב מסלול…', '');
    try {
      await load();
    } catch (err) {
      say('לא הצלחתי לטעון את מפת הרחובות', 'בדוק את החיבור ונסה שוב.');
      return;
    }
    const t0 = performance.now();
    const r = RouteEngine.route(base, trails(), from, to);
    console.info(`דרך קיצור: מסלול חושב ב-${Math.round(performance.now() - t0)} מ"ש`);
    if (r.error) {
      result = null;
      clearLine();
      say(r.error === 'none' ? 'לא מצאתי דרך בין שתי הנקודות' : 'הנקודה רחוקה מכל דרך',
          r.error === 'none' ? 'נסה להזיז אחת מהן.' : 'גרור אותה קרוב יותר לרחוב או לשביל.');
      acts();
      return;
    }
    result = r;
    paintLine();
    paintBar();
    acts();
    if (!keepCamera) frame();
    showDetail();
    if (typeof scheduleSync === 'function') scheduleSync();
  }

  const mins = (m) => `${Math.max(1, Math.round(m))} דק׳`;

  /** What the shortcuts bought you, in the words somebody would use. */
  function saving(r) {
    if (!r.trailsUsed.length) return null;
    if (!r.without) return 'בלי קיצורי הדרך אין דרך בכלל';
    const saved = r.without.minutes - r.minutes;
    // The route prefers the shortcuts, so it may walk a little longer than the
    // streets would: said plainly, with what it buys.
    if (saved <= -0.75) return `${fmt(r.trailLength)} בשבילים, ${mins(-saved)} יותר מברחובות`;
    if (saved < 0.75) return `${fmt(r.trailLength)} בשבילים, באותו זמן כמו ברחובות`;
    return `חוסך ${mins(saved)} לעומת הרחובות`;
  }

  function paintBar() {
    const r = result;
    const s = saving(r);
    const alts = r.alternatives.length
      ? ` · ${plural(r.alternatives.length, 'עוד דרך אחת', 'דרכים נוספות')} בכחול בהיר` : '';
    say(`${mins(r.minutes)} · ${fmt(r.length)}`,
        (s ? `${s} · ${plural(r.trailsUsed.length, 'קיצור דרך אחד', 'קיצורי דרך')}`
           : 'אין קיצור דרך שעוזר כאן') + alts);
  }

  /** Make alternative `i` the recommended route, keeping the camera. */
  function choose(i) {
    if (!result || !result.alternatives[i]) return;
    stopNavIfOurs();
    result = RouteEngine.choose(result, i);
    paintLine();
    paintBar();
    showDetail();
  }

  /** A tap on the map while a route is shown: on a faded line, choose it. */
  function altAt(point) {
    if (!map || !result || !map.getLayer('route-alt-hit')) return null;
    const f = map.queryRenderedFeatures(point, { layers: ['route-alt-hit'] })[0];
    return f ? f.properties.alt : null;
  }

  function tapAlt(point) {
    const i = altAt(point);
    if (i == null) return false;
    choose(i);
    return true;
  }

  /* ---------- on the map ---------- */

  function geo() {
    const r = result;
    const line = (path, props) => ({ type: 'Feature', properties: props,
      geometry: { type: 'LineString', coordinates: path.map(([la, ln]) => [ln, la]) } });
    // Alternatives first, so they draw underneath the recommended route.
    const feats = r.alternatives.map((a, i) => line(a.path, { kind: 'alt', alt: i }));
    r.legs.forEach((l) => feats.push(line(l.path, { kind: l.kind })));
    return { type: 'FeatureCollection', features: feats };
  }

  // The recommended route is strong blue from end to end, and the alternatives
  // the same blue, faded (Ori, 4/10/2026). The shortcuts inside the route keep
  // a gold stripe down the middle - the colour of the shortcuts layer - so you
  // still see where you leave the street. The streets-only walk is no longer
  // drawn: the alternatives took its place, and its minutes stay in the pane.
  function paintLine() {
    if (!map || !result) return;
    // isStyleLoaded is false while tiles are still arriving, which right after
    // opening the app is most of the time. Waiting beats a route that only
    // draws on the second try.
    if (!map.isStyleLoaded()) { map.once('idle', paintLine); return; }
    const data = geo();
    if (map.getSource(SRC)) { map.getSource(SRC).setData(data); return; }
    map.addSource(SRC, { type: 'geojson', data });
    const kind = (k) => ['==', ['get', 'kind'], k];
    const onRoute = ['in', ['get', 'kind'], ['literal', ['street', 'trail']]];
    const round = { 'line-cap': 'round', 'line-join': 'round' };
    // Wide and invisible, so a finger finds a faded line without aiming.
    map.addLayer({ id: 'route-alt-hit', type: 'line', source: SRC, filter: kind('alt'),
      layout: round, paint: { 'line-color': '#000', 'line-width': 24, 'line-opacity': 0.001 } });
    map.addLayer({ id: 'route-alt', type: 'line', source: SRC, filter: kind('alt'),
      layout: round, paint: { 'line-color': BLUE_FADED, 'line-width': 6, 'line-opacity': 0.55 } });
    map.addLayer({ id: 'route-case', type: 'line', source: SRC, filter: onRoute,
      layout: round, paint: { 'line-color': '#fff', 'line-width': 10.5, 'line-opacity': 0.95 } });
    map.addLayer({ id: 'route-main', type: 'line', source: SRC, filter: onRoute,
      layout: round, paint: { 'line-color': BLUE, 'line-width': 7 } });
    map.addLayer({ id: 'route-trail', type: 'line', source: SRC, filter: kind('trail'),
      layout: round, paint: { 'line-color': GOLD, 'line-width': 2.5 } });
    map.addLayer({ id: 'route-off', type: 'line', source: SRC, filter: kind('off'),
      layout: { 'line-cap': 'round' },
      paint: { 'line-color': BLUE, 'line-width': 3, 'line-dasharray': [0.8, 1.2] } });
    // Bound by layer id, so they outlive the layer being removed and re-added.
    if (!hoverWired) {
      hoverWired = true;
      map.on('mouseenter', 'route-alt-hit', () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', 'route-alt-hit', () => { map.getCanvas().style.cursor = ''; });
    }
  }
  let hoverWired = false;

  function clearLine() {
    if (!map) return;
    LAYERS.forEach((id) => { if (map.getLayer(id)) map.removeLayer(id); });
    if (map.getSource(SRC)) map.removeSource(SRC);
  }

  function clear() {
    clearLine();
    Object.values(markers).forEach((m) => m && m.remove());
    markers = { from: null, to: null };
  }

  function frame() {
    if (!map || !result) return;
    const b = new maplibregl.LngLatBounds();
    result.path.forEach(([la, ln]) => b.extend([ln, la]));
    if (result.without) result.without.path.forEach(([la, ln]) => b.extend([ln, la]));
    // On a phone the panel lies over the bottom of the map, however short. It
    // may still be sliding down when this runs, so the measure is capped: a
    // padding taller than the map makes MapLibre refuse to move at all.
    const under = window.innerWidth <= 760
      ? Math.min(el('panel').getBoundingClientRect().height, window.innerHeight * 0.2) : 0;
    // The camera is worked out as if the map were flat and then given back its
    // tilt. fitBounds on a tilted map (the app opens at 58°) leaves the route
    // small and off to the top, since it measures the box on the flat ground.
    const cam = map.cameraForBounds(b, { padding: { top: 90, bottom: 40 + under, left: 40, right: 40 },
                                         maxZoom: 17.5, bearing: 0 });
    if (cam) map.easeTo({ ...cam, bearing: map.getBearing(), pitch: map.getPitch(), duration: 800 });
  }

  /* ---------- the pane ---------- */

  function showDetail() {
    const r = result;
    const s = saving(r);
    // The directions as turns, in order (4/10/2026). The same steps drive the
    // navigation bar, and the one it is on is highlighted here as you walk.
    const GLYPH = { start: '●', straight: '↑', 'slight-right': '↗', 'slight-left': '↖',
                    right: '→', left: '←', back: '↶', end: '🏁' };
    const rows = r.steps.map((st, i) => {
      const attr = st.kind === 'trail' && st.id ? ` data-route-trail="${escapeHtml(st.id)}"` : '';
      return `<li class="route-leg ${st.kind}" data-step="${i}"${attr}>
          <span class="route-glyph">${GLYPH[st.turn]}</span>
          <span class="route-nm">${escapeHtml(RouteEngine.say(st))}${st.kind === 'trail' ? ' 🥾' : ''}</span>
          <span class="route-len">${st.length >= 1 ? fmt(st.length) : ''}</span></li>`;
    }).join('');
    // The other ways, as rows that pick them - the same as tapping the faded
    // line on the map, for whoever reads the pane before the map.
    const alts = r.alternatives.length ? `
      <h3 class="route-alts-h">דרכים נוספות <span>(בכחול בהיר במפה)</span></h3>
      <ul class="route-alts">${r.alternatives.map((a, i) => `
        <li><button class="route-alt" data-route-alt="${i}">
          <b>${mins(a.minutes)}</b><span>${fmt(a.length)}</span>
          <span class="route-alt-tr">${a.trailsUsed.length
            ? `🥾 ${plural(a.trailsUsed.length, 'קיצור דרך אחד', 'קיצורי דרך')}` : 'רחובות בלבד'}</span>
        </button></li>`).join('')}</ul>` : '';
    el('detail').innerHTML = `
      <h2 class="route-title">מסלול הליכה</h2>
      <div class="route-compare">
        <div class="route-num"><b>${mins(r.minutes)}</b><span>${fmt(r.length)} דרך קיצורי הדרך</span></div>
        ${r.without && r.trailsUsed.length ? `<div class="route-num grey"><b>${mins(r.without.minutes)}</b>
          <span>${fmt(r.without.length)} ברחובות בלבד</span></div>` : ''}
      </div>
      <p class="route-say">${s ? escapeHtml(s) + '.' : 'אין כאן קיצור דרך שעוזר, וזה המסלול ברחובות.'}
        ${r.trailsUsed.length ? 'הפס הזהוב בתוך הקו הכחול מסמן את קיצורי הדרך.' : ''}</p>
      <ol class="route-legs">${rows}</ol>
      ${alts}
      <div class="acts">
        <button class="act act-nav" data-route="go">${icon(I_WALK)}
          <span class="lbl">התחל ניווט
            <span class="hint">תוך כדי הליכה: המפה עוקבת אחריך, והפס למעלה אומר מה הפנייה הבאה ובעוד כמה מטרים</span></span></button>
        <button class="act act-sub" data-route="close"><span class="lbl">סגור את המסלול</span></button>
      </div>
      <p class="sheet-credit">הרחובות מ-OpenStreetMap. קיצורי הדרך מהיוזמה. ההליכה מחושבת לפי
        4.8 קמ"ש. שביל שנחסם? ספר ליוזמה.</p>`;
    selectedId = null;
    Layers.highlight(null);
    markSelection(null);
    el('list-view').hidden = true;
    el('detail-view').hidden = false;
  }

  /* ---------- walking it ---------- */

  /** Highlight the step navigation is on, in the pane, if it is showing. */
  function markStep(i) {
    document.querySelectorAll('#detail .route-leg[data-step]').forEach((li) =>
      li.classList.toggle('now', +li.dataset.step === i));
  }

  function navigate() {
    if (!result) return;
    startNav({ name: 'המסלול', route: true, path: result.path, steps: result.steps,
               entries: [{ lat: result.path[0][0], lng: result.path[0][1] }] });
  }

  function stopNavIfOurs() {
    if (nav && nav.item && nav.item.route) stopNav();
  }

  function swap() {
    if (!from || !to) return;
    [from, to] = [{ ...to, mine: false }, { ...from }];
    paintMarker('from'); paintMarker('to');
    run();
  }

  function close() {
    stopNavIfOurs();
    picking = null;
    result = from = to = null;
    clear();
    bar().hidden = true;
    document.body.classList.remove('routing');
    if (!el('detail-view').hidden && el('detail').querySelector('.route-title')) deselect();
    if (typeof scheduleSync === 'function') scheduleSync();
  }

  /* ---------- in the link ---------- */

  const r5 = (v) => +v.toFixed(5);
  const linkValue = () => (from && to ? [r5(from.lat), r5(from.lng), r5(to.lat), r5(to.lng)].join(',') : null);

  function fromLink(raw) {
    const n = String(raw || '').split(',').map(Number);
    if (n.length !== 4 || n.some((v) => !isFinite(v))) return;
    stopNavIfOurs();
    clear();
    bar().hidden = false;
    document.body.classList.add('routing');
    from = { lat: n[0], lng: n[1] };
    to = { lat: n[2], lng: n[3] };
    picking = null;
    paintMarker('from'); paintMarker('to');
    // A link that carries a camera already framed the shot it was sent with.
    run(new URLSearchParams(location.search).has('map'));
  }

  function wire() {
    el('route-stop').addEventListener('click', (e) => { e.stopPropagation(); close(); });
    el('route-go').addEventListener('click', (e) => { e.stopPropagation(); navigate(); });
    el('route-swap').addEventListener('click', (e) => { e.stopPropagation(); swap(); });
    el('route-mine').addEventListener('click', (e) => { e.stopPropagation(); useMine(); });
    el('route-from').addEventListener('click', (e) => { e.stopPropagation(); repick('from'); });
    el('route-to').addEventListener('click', (e) => { e.stopPropagation(); repick('to'); });
    el('route-bar').addEventListener('click', () => { if (result) { frame(); showDetail(); } });
    el('route-ask').addEventListener('click', () => open(null));
    el('detail').addEventListener('click', (e) => {
      const act = e.target.closest('[data-route]');
      if (act) { act.dataset.route === 'go' ? navigate() : close(); return; }
      const alt = e.target.closest('[data-route-alt]');
      if (alt) { choose(+alt.dataset.routeAlt); return; }
      const leg = e.target.closest('[data-route-trail]');
      if (leg && Layers.item(leg.dataset.routeTrail)) select(leg.dataset.routeTrail);
    });
  }

  /** Called after a basemap swap, which throws every added layer away. */
  function repaint() { if (result) { clearLine(); paintLine(); } }

  return { open, pick, isOn, isPicking, close, wire, repaint, linkValue, fromLink, load,
           choose, tapAlt, altAt, markStep, result: () => result };
})();
