/* תלת-ממד - the moshava as a model, the third background on the globe button.
 *
 * Ori, 13/10/2026: "add a third option to the map/satellite button, a 3D
 * model of the moshava, taken from pardes3d - and in that mode let me tilt
 * the view further than the 3D button does."
 *
 * The model is pardes3d's, but not its meshes. Godot gets triangles; a map
 * in a browser is better off with what the triangles were built from - a
 * footprint and a height - because MapLibre extrudes those itself, sorts
 * them against each other in depth, and draws the trails, the labels and
 * the selection on the same ground. So pardes3d/pipeline/export_web.py
 * writes the very same buildings (OSM plus Microsoft's, heights calibrated
 * on GHSL, the same plaster colours), the same 27 thousand trees from the
 * survey and the canopy map, and the same walls along the parcel lines, as
 * compact lists in data/model3d/. They load the first time the mode is
 * chosen and are kept.
 *
 * Trees are four extrusions each: a trunk, and a crown in three steps -
 * narrower, widest, narrower - which is as round as an extrusion gets. From any distance at all that reads as a tree; close up it is a
 * stepped octagon, which is the honest price of not bringing a 3D engine into
 * the map. The 120 thousand trees pardes3d only *imagines* in the orchards
 * are left out - the satellite picture already shows the orchards, and they
 * would be four times the geometry.
 *
 * The ground is the satellite photograph, laid flat. Terrain is off in this
 * mode for the reason the flight turns it off (see explore.js): looked at
 * from near the horizon the DEM, which stops at zoom 14, smears the picture
 * into vertical streaks, and the real relief - 46 metres across the whole
 * moshava - is smaller than the buildings standing on it.
 *
 * ---- looking at it ----
 *
 * A model is only a model if you can look at it from the side, and the
 * map's own ways of tilting are hidden: a right-drag, a ctrl-drag, a
 * two-finger drag on a phone. The 3D button jumps between two angles. So
 * this mode adds two things beside the button column:
 *
 *   - The angle slider. A vertical track: the top is straight down, the
 *     bottom is the horizon (85 degrees, the most MapLibre 4 allows). Drag
 *     it and the map tilts as you drag; tap a point on it and it goes there.
 *     It follows the map, so a gesture tilt moves the thumb too. Vertical,
 *     because the motion it stands for is vertical: dragging down lowers
 *     your eyes towards the horizon.
 *
 *   - The person. Tap it and you are standing on the street nearest the
 *     point you were looking at, at eye height, looking down the street. Then the
 *     controls become Street View's, the one way of walking through a place
 *     on a screen that people already know: drag to look around, tap the
 *     ground to walk there; on a keyboard the arrows walk and turn too. The
 *     same person, now an arrow pointing up, takes you back to where you
 *     were.
 */
'use strict';

const Model3D = (() => {
  const RAD = Math.PI / 180;
  const M_LAT = 111320;
  const EYE = 1.7;            // metres above the ground in the street
  const PITCH_MAX = 85;       // MapLibre 4's ceiling
  const STREET_PITCH = 85;    // where you look on arriving: straight down the street
  const STREET_PITCH_MIN = 45;
  const WALK_SPEED = 9;       // m/s of a tap-walk: brisk, so that crossing a block is a second
  const TAP_MAX = 120;        // metres; a tap further than this walks this far towards it
  const FOCAL = 1.5;          // MapLibre's camera distance in container heights (fov 36.87)

  const TREE_GREENS = ['#4c7536', '#5a8441', '#3f6630', '#678c45', '#52793a', '#46703a'];
  const TRUNK = '#6b5340';
  const HEDGE = '#41662f';
  const BLOCK = '#a9a59c';

  let data = null;            // the GeoJSON, built once
  let loading = null;
  let active = false;         // is the 3D background the one showing
  let street = null;          // standing in the street: {lat, lng, brg, pitch, back}
  let walkAnim = 0;
  const keys = new Set();

  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

  /* ---------- the data ---------- */

  function unpack(row, start, origin, q) {
    const out = [];
    let x = 0, y = 0;
    for (let i = start; i + 1 < row.length; i += 2) {
      x += row[i]; y += row[i + 1];
      out.push([origin[0] + x / q, origin[1] + y / q]);
    }
    return out;
  }

  /** A regular polygon of `n` sides around a point, radius in metres. */
  function ngon(lng, lat, r, n, rot) {
    const kx = 1 / (M_LAT * Math.cos(lat * RAD)), ky = 1 / M_LAT;
    const ring = [];
    for (let i = 0; i < n; i++) {
      const a = rot + (i / n) * 2 * Math.PI;
      ring.push([lng + Math.cos(a) * r * kx, lat + Math.sin(a) * r * ky]);
    }
    ring.push(ring[0]);
    return [ring];
  }

  /** A line thickened into a polygon `w` metres wide, for a wall. */
  function ribbon(pts, w) {
    if (pts.length < 2) return null;
    const lat = pts[0][1];
    const kx = M_LAT * Math.cos(lat * RAD), ky = M_LAT;
    const P = pts.map(([x, y]) => [x * kx, y * ky]);
    const L = [], R = [];
    for (let i = 0; i < P.length; i++) {
      const a = P[Math.max(i - 1, 0)], b = P[Math.min(i + 1, P.length - 1)];
      let dx = b[0] - a[0], dy = b[1] - a[1];
      const len = Math.hypot(dx, dy) || 1;
      dx /= len; dy /= len;
      const ox = -dy * w / 2, oy = dx * w / 2;
      L.push([(P[i][0] + ox) / kx, (P[i][1] + oy) / ky]);
      R.push([(P[i][0] - ox) / kx, (P[i][1] - oy) / ky]);
    }
    const ring = L.concat(R.reverse());
    ring.push(ring[0]);
    return [ring];
  }

  const hash = (a, b) => {
    const s = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
    return s - Math.floor(s);
  };

  async function load() {
    if (data) return data;
    if (loading) return loading;
    loading = (async () => {
      const get = (n) => fetch(`data/model3d/${n}.json`).then((r) => {
        if (!r.ok) throw new Error(`${n}: ${r.status}`);
        return r.json();
      });
      const [b, t, w] = await Promise.all([get('buildings'), get('trees'), get('walls')]);

      const houses = [];
      for (const row of b.rows) {
        const ring = unpack(row, 3, b.origin, b.q);
        if (ring.length < 3) continue;
        ring.push(ring[0]);
        index.add(ring);
        const h = row[0];
        // `r` is the roof's colour, drawn by a second layer over the same
        // footprints (see attach).
        houses.push({ type: 'Feature',
          properties: { h, c: rgb(b.wall_colors[row[1]]), r: rgb(b.roof_colors[row[2]]) },
          geometry: { type: 'Polygon', coordinates: [ring] } });
      }

      const trees = [];
      for (const [x, y, h0, c0, kind] of t.rows) {
        const lng = t.origin[0] + x / t.q, lat = t.origin[1] + y / t.q;
        const h = Math.max(h0 || 6, 2.5), r = Math.max((c0 || 5) / 2, 1);
        const v = hash(x, y);
        const green = TREE_GREENS[Math.floor(v * TREE_GREENS.length)];
        const rot = v * Math.PI;
        trees.push(
          { type: 'Feature', properties: { b: 0, h: h * 0.42, c: TRUNK },
            geometry: { type: 'Polygon', coordinates: ngon(lng, lat, Math.max(0.16, r * 0.07), 4, rot) } },
          { type: 'Feature', properties: { b: h * 0.38, h: h * 0.48, c: green },
            geometry: { type: 'Polygon', coordinates: ngon(lng, lat, r * 0.8, 8, rot) } },
          { type: 'Feature', properties: { b: h * 0.48, h: h * 0.8, c: green },
            geometry: { type: 'Polygon', coordinates: ngon(lng, lat, r, 8, rot) } },
          { type: 'Feature', properties: { b: h * 0.8, h, c: green },
            geometry: { type: 'Polygon', coordinates: ngon(lng, lat, r * 0.62, 8, rot) } });
        void kind;
      }

      const walls = [];
      for (const row of w.rows) {
        const [kind, h, thick] = row;
        const geo = ribbon(unpack(row, 3, w.origin, w.q), thick);
        if (!geo) continue;
        walls.push({ type: 'Feature',
          properties: { h, c: kind === 2 ? HEDGE : kind === 1 ? BLOCK : '#e4dccb' },
          geometry: { type: 'Polygon', coordinates: geo } });
      }

      const fc = (features) => ({ type: 'FeatureCollection', features });
      data = { houses: fc(houses), trees: fc(trees), walls: fc(walls),
        attribution: b.attribution, raw: b, rawWalls: w };
      return data;
    })();
    try {
      return await loading;
    } catch (err) {
      loading = null;
      throw err;
    }
  }

  /* ---------- the buildings as obstacles ----------
   *
   * Standing in the street you should not be able to land inside a house or
   * walk through one. The footprints go into a grid of 40-metre cells, in
   * local metres around the moshava, which is all a tap-walk needs: is this
   * point inside a building, and where does this short segment first cross a
   * wall. */
  const index = (() => {
    const CELL = 40;
    const cells = new Map();
    const LAT0 = 32.4735, KX = M_LAT * Math.cos(LAT0 * RAD);
    const local = (lng, lat) => [(lng - 34.9765) * KX, (lat - LAT0) * M_LAT];
    const key = (i, j) => i * 100003 + j;
    function add(ring) {
      const pts = ring.map(([lng, lat]) => local(lng, lat));
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const [x, y] of pts) {
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      }
      const poly = { pts, x0, y0, x1, y1 };
      for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++) {
        for (let j = Math.floor(y0 / CELL); j <= Math.floor(y1 / CELL); j++) {
          const k = key(i, j);
          if (!cells.has(k)) cells.set(k, []);
          cells.get(k).push(poly);
        }
      }
    }
    function near(x0, y0, x1, y1) {
      const out = new Set();
      for (let i = Math.floor(Math.min(x0, x1) / CELL); i <= Math.floor(Math.max(x0, x1) / CELL); i++) {
        for (let j = Math.floor(Math.min(y0, y1) / CELL); j <= Math.floor(Math.max(y0, y1) / CELL); j++) {
          for (const p of cells.get(key(i, j)) || []) out.add(p);
        }
      }
      return out;
    }
    function inside(p, x, y) {
      if (x < p.x0 || x > p.x1 || y < p.y0 || y > p.y1) return false;
      let c = false;
      const q = p.pts;
      for (let i = 0, j = q.length - 1; i < q.length; j = i++) {
        if ((q[i][1] > y) !== (q[j][1] > y) &&
            x < (q[j][0] - q[i][0]) * (y - q[i][1]) / (q[j][1] - q[i][1]) + q[i][0]) c = !c;
      }
      return c;
    }
    /** The building a point is inside, or null. */
    function at(lng, lat) {
      const [x, y] = local(lng, lat);
      for (const p of near(x, y, x, y)) if (inside(p, x, y)) return p;
      return null;
    }
    /** A point inside a building moved out through the nearest wall, `gap`
     *  metres clear of it. */
    function pushOut(lng, lat, gap) {
      let [x, y] = local(lng, lat);
      for (let round = 0; round < 4; round++) {
        let hit = null;
        for (const p of near(x, y, x, y)) if (inside(p, x, y)) { hit = p; break; }
        if (!hit) break;
        let best = Infinity, bx = x, by = y;
        const q = hit.pts;
        for (let i = 0; i + 1 < q.length; i++) {
          const [ax, ay] = q[i], [cx, cy] = q[i + 1];
          const dx = cx - ax, dy = cy - ay;
          const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
          const px = ax + t * dx, py = ay + t * dy;
          const d = Math.hypot(px - x, py - y);
          if (d < best) { best = d; bx = px; by = py; }
        }
        const len = Math.hypot(bx - x, by - y) || 1;
        x = bx + (bx - x) / len * gap;
        y = by + (by - y) / len * gap;
      }
      return { lng: 34.9765 + x / KX, lat: LAT0 + y / M_LAT };
    }
    /** How far along the segment from a to b (0..1) the first wall is, or 1. */
    function firstWall(a, b) {
      const [ax, ay] = local(a.lng, a.lat), [bx, by] = local(b.lng, b.lat);
      let tMin = 1;
      for (const p of near(ax, ay, bx, by)) {
        const q = p.pts;
        for (let i = 0; i + 1 < q.length; i++) {
          const [cx, cy] = q[i], [dx, dy] = q[i + 1];
          const rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
          const den = rx * sy - ry * sx;
          if (Math.abs(den) < 1e-9) continue;
          const t = ((cx - ax) * sy - (cy - ay) * sx) / den;
          const u = ((cx - ax) * ry - (cy - ay) * rx) / den;
          if (t >= 0 && t < tMin && u >= 0 && u <= 1) tMin = t;
        }
      }
      return tMin;
    }
    return { add, at, pushOut, firstWall };
  })();

  /* ---------- on the map ---------- */

  /** Put the model on the map. Called from applyOverlays after every style
   *  load of the 3D background, because a style swap throws sources away. */
  async function attach() {
    if (!map) return;
    active = true;
    map.setMaxPitch(PITCH_MAX);
    paintControls();
    let d;
    try {
      d = await load();
    } catch (err) {
      console.error('model3d: the model did not load', err);
      toast('המודל התלת־ממדי לא נטען. בדוק את החיבור.');
      return;
    }
    if (!active || !map.getStyle()) return;
    const src = (id, fc, extra) => {
      if (!map.getSource(id)) map.addSource(id, { type: 'geojson', data: fc, ...extra });
    };
    // Built to zoom 17 so the trees keep their octagons; past that the tiles
    // are overzoomed. The attribution rides on the houses.
    src('m3d-houses', d.houses, { maxzoom: 17, tolerance: 0.2, attribution: d.attribution });
    src('m3d-trees', d.trees, { maxzoom: 17, tolerance: 0.1 });
    src('m3d-walls', d.walls, { maxzoom: 17, tolerance: 0.1 });

    const add = (spec) => { if (!map.getLayer(spec.id)) map.addLayer(spec); };
    const ext = (id, minzoom, base) => ({
      id, source: id, type: 'fill-extrusion', minzoom,
      paint: {
        'fill-extrusion-color': ['get', 'c'],
        'fill-extrusion-height': ['get', 'h'],
        'fill-extrusion-base': base ? ['get', 'b'] : 0,
        'fill-extrusion-vertical-gradient': true
      }
    });
    add(ext('m3d-walls', 16, false));
    add(ext('m3d-trees', 14.5, true));
    add(ext('m3d-houses', 12, false));
    // A roof is the top 40 cm of the house in its own colour: white tar,
    // grey, now and then red tiles. Seen from above, it is what tells the
    // houses apart. The same source as the walls, so the footprints are
    // held once.
    add({
      id: 'm3d-roofs', source: 'm3d-houses', type: 'fill-extrusion', minzoom: 12,
      paint: {
        'fill-extrusion-color': ['get', 'r'],
        'fill-extrusion-height': ['+', ['get', 'h'], 0.4],
        'fill-extrusion-base': ['get', 'h']
      }
    });
    restack();
    // The faces (facades.js) take over from the plain boxes once Three.js
    // is here; until then, or without it, the boxes are the houses.
    if (typeof Facades !== 'undefined') {
      Facades.attach(map, d.raw, d.rawWalls, 'm3d-roofs').then((ok) => { if (ok && active) restack(); });
    }
  }

  /* Where the model sits among the app's own layers. With no terrain, MapLibre
   * draws layers strictly in order and a line drawn after a building is drawn
   * across it - a trail running behind a house showed through the house. A
   * line or a fill on the ground is exactly what a building should hide, and
   * a pin or a label is what it should not. So the buildings go above every
   * line and fill and below every circle and label, and they are put back
   * there whenever the app adds a layer, which it does at the top. */
  const EXTRUDED = ['m3d-walls', 'm3d-trees', 'm3d-houses', 'm3d-facades', 'm3d-roofs'];
  const onTop = (l) => l.type === 'symbol' || l.type === 'circle' || l.type === 'heatmap';
  let restacking = false;

  function restack() {
    if (!active || restacking || !map.getStyle() || !map.getLayer('m3d-roofs')) return;
    const layers = map.getStyle().layers;
    const ours = layers.map((l, i) => (EXTRUDED.includes(l.id) ? i : -1)).filter((i) => i >= 0);
    const lo = Math.min(...ours), hi = Math.max(...ours);
    const wrong = layers.some((l, i) => !EXTRUDED.includes(l.id) &&
      ((i > lo && !onTop(l)) || (i < hi && onTop(l) && l.source !== 'sat')));
    if (!wrong) return;
    restacking = true;
    try {
      EXTRUDED.forEach((id) => { if (map.getLayer(id)) map.moveLayer(id); });
      layers.filter((l) => onTop(l) && !EXTRUDED.includes(l.id)).forEach((l) => map.moveLayer(l.id));
    } finally {
      restacking = false;
    }
  }

  /** Leaving the 3D background. The layers go with the style; what is ours
   *  to undo is the steeper tilt and the street. */
  function detach() {
    if (street) leaveStreet(false);
    active = false;
    if (map) {
      map.setMaxPitch(80);
      if (map.getPitch() > 80) map.setPitch(80);
    }
    paintControls();
  }

  /* ---------- the angle slider ---------- */

  let ui = null;

  function buildControls() {
    if (ui) return ui;
    const box = document.createElement('div');
    box.id = 'm3d-ctl';
    box.className = 'm3d-ctl';
    box.hidden = true;
    box.innerHTML = `
      <div class="m3d-tilt" role="slider" tabindex="0" aria-orientation="vertical"
           aria-label="זווית המבט" aria-valuemin="0" aria-valuemax="${PITCH_MAX}">
        <span class="m3d-end m3d-top" title="מלמעלה" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M12 4a1 1 0 011 1v9.6l3.3-3.3 1.4 1.4L12 18.4l-5.7-5.7 1.4-1.4 3.3 3.3V5a1 1 0 011-1z"/><path d="M4 20h16v2H4z"/></svg>
        </span>
        <div class="m3d-track"><div class="m3d-fill"></div><div class="m3d-thumb"></div></div>
        <span class="m3d-end m3d-bottom" title="אל האופק" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M2 15h20v2H2z"/><path d="M5 12a7 7 0 0114 0h-2a5 5 0 00-10 0z"/></svg>
        </span>
      </div>
      <button type="button" class="m3d-walk" title="לעמוד ברחוב" aria-label="לעמוד ברחוב">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.5 5.5a2 2 0 100-4 2 2 0 000 4zM9.8 8.9L7 23h2.1l1.8-8 2.1 2v6h2v-7.5l-2.1-2 .6-3A7 7 0 0019 13v-2a5 5 0 01-4.2-2.4l-1-1.6c-.4-.6-1-1-1.8-1-.3 0-.5 0-.8.2L6 8.3V13h2V9.6l1.8-.7z"/></svg>
      </button>`;
    document.body.appendChild(box);

    const hint = document.createElement('div');
    hint.className = 'm3d-hint';
    hint.hidden = true;
    document.body.appendChild(hint);

    const exit = document.createElement('button');
    exit.type = 'button';
    exit.className = 'm3d-exit';
    exit.hidden = true;
    exit.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l-7 7h4.5v8h5v-8H19z"/></svg> חזרה למעלה';
    exit.addEventListener('click', () => leaveStreet(true));
    document.body.appendChild(exit);

    ui = { box, hint, exit, slider: box.querySelector('.m3d-tilt'),
      track: box.querySelector('.m3d-track'), thumb: box.querySelector('.m3d-thumb'),
      fill: box.querySelector('.m3d-fill'), walk: box.querySelector('.m3d-walk') };

    // Drag anywhere on the slider, track or not: a thumb 20 pixels wide on
    // a phone is hard to catch, and the whole strip means only one thing.
    let dragging = false;
    const pitchAt = (clientY) => {
      const r = ui.track.getBoundingClientRect();
      return Math.min(Math.max((clientY - r.top) / r.height, 0), 1) * PITCH_MAX;
    };
    ui.slider.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.m3d-end')) return;
      dragging = true;
      ui.slider.setPointerCapture(e.pointerId);
      setPitch(pitchAt(e.clientY), 250);
      e.preventDefault();
    });
    ui.slider.addEventListener('pointermove', (e) => {
      if (dragging) setPitch(pitchAt(e.clientY), 0);
    });
    const up = () => {
      if (!dragging) return;
      dragging = false;
      if (typeof scheduleSync === 'function') scheduleSync();
    };
    ui.slider.addEventListener('pointerup', up);
    ui.slider.addEventListener('pointercancel', up);
    box.querySelector('.m3d-top').addEventListener('click', () => setPitch(0, 600));
    box.querySelector('.m3d-bottom').addEventListener('click', () => setPitch(PITCH_MAX, 600));
    ui.slider.addEventListener('keydown', (e) => {
      const step = { ArrowDown: 5, ArrowUp: -5, PageDown: 15, PageUp: -15 }[e.key];
      if (step === undefined) return;
      setPitch(map.getPitch() + step, 150);
      e.preventDefault();
    });
    ui.walk.addEventListener('click', () => (street ? leaveStreet(true) : enterStreet()));

    map.on('pitch', paintThumb);
    // The list pulled up or down, a phone turned: the same eyes, a new frame.
    map.on('resize', () => { if (street && !map.isEasing()) placeCamera(); });
    // After the app adds a layer, once the style has settled.
    let pending = 0;
    map.on('styledata', () => {
      if (!active || pending) return;
      pending = requestAnimationFrame(() => { pending = 0; restack(); });
    });
    return ui;
  }

  function setPitch(p, duration) {
    p = Math.min(Math.max(p, 0), PITCH_MAX);
    if (street) {
      street.pitch = Math.min(Math.max(p, STREET_PITCH_MIN), PITCH_MAX);
      placeCamera();
      return;
    }
    if (duration) map.easeTo({ pitch: p, duration });
    else map.setPitch(p);
  }

  function paintThumb() {
    if (!ui || ui.box.hidden) return;
    const p = map.getPitch();
    const f = p / PITCH_MAX;
    ui.thumb.style.top = `${(f * 100).toFixed(1)}%`;
    ui.fill.style.height = `${(f * 100).toFixed(1)}%`;
    ui.slider.setAttribute('aria-valuenow', Math.round(p));
    ui.slider.setAttribute('aria-valuetext', `${Math.round(p)} מעלות`);
  }

  function paintControls() {
    if (!map) return;
    buildControls();
    ui.box.hidden = !active;
    ui.walk.classList.toggle('on', !!street);
    ui.walk.title = street ? 'חזרה למעלה' : 'לעמוד ברחוב';
    ui.exit.hidden = !street;
    document.body.classList.toggle('m3d-on', active);
    document.body.classList.toggle('m3d-street', !!street);
    paintThumb();
  }

  let hintTimer = 0;
  function say(text, ms) {
    ui.hint.textContent = text;
    ui.hint.hidden = false;
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => { ui.hint.hidden = true; }, ms || 4000);
  }

  function toast(text) {
    buildControls();
    say(text, 5000);
  }

  /* ---------- in the street ---------- */

  /** Hand the map the camera of somebody standing at street.lat/lng, eyes
   *  EYE metres up, looking along street.brg and street.pitch. The map's
   *  camera is a centre point and a zoom, so the centre is the spot on the
   *  ground the eyes look at and the zoom is the one that puts the camera
   *  EYE metres above it - explore.js's zoomFor, with the ground flat. */
  function poseFor(s) {
    const h = map.getContainer().clientHeight || 800;
    const ahead = EYE * Math.tan(s.pitch * RAD);
    const c = step(s.lat, s.lng, s.brg, ahead);
    const mpp = EYE / (FOCAL * h * Math.cos(s.pitch * RAD));
    const zoom = Math.log2(40075016.686 * Math.cos(c.lat * RAD) / mpp / 512);
    return { center: [c.lng, c.lat], zoom, bearing: s.brg, pitch: s.pitch };
  }

  function placeCamera() {
    map.jumpTo(poseFor(street));
  }

  function step(lat, lng, brg, m) {
    return {
      lat: lat + m * Math.cos(brg * RAD) / M_LAT,
      lng: lng + m * Math.sin(brg * RAD) / (M_LAT * Math.cos(lat * RAD))
    };
  }

  /** The nearest street to a point, within 150 m, and which way along it to
   *  face: of its two directions, the one nearer the bearing you had. This is
   *  what the little man in Street View does when you drop him - you always
   *  land on a road, looking down it, and never with your nose in a wall,
   *  which is where the middle of the screen turned out to put you more often
   *  than not. The streets are the vector tiles under the asphalt layer, so
   *  they are there only once the map has loaded them (zoom 14 and up). */
  const STREETS = ['primary', 'trunk', 'secondary', 'tertiary', 'minor', 'street',
    'street_limited', 'service', 'path', 'track'];

  function snapToStreet(c, brg) {
    let feats = [];
    try {
      feats = map.querySourceFeatures('omt', { sourceLayer: 'transportation' });
    } catch (err) {
      return null;
    }
    const kx = M_LAT * Math.cos(c.lat * RAD);
    let best = null;
    for (const f of feats) {
      if (!STREETS.includes(f.properties.class)) continue;
      const g = f.geometry;
      const lines = g.type === 'LineString' ? [g.coordinates]
        : g.type === 'MultiLineString' ? g.coordinates : [];
      for (const line of lines) {
        for (let i = 0; i + 1 < line.length; i++) {
          const ax = (line[i][0] - c.lng) * kx, ay = (line[i][1] - c.lat) * M_LAT;
          const bx = (line[i + 1][0] - c.lng) * kx, by = (line[i + 1][1] - c.lat) * M_LAT;
          const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
          if (L2 < 0.01) continue;
          const t = Math.max(0, Math.min(1, -(ax * dx + ay * dy) / L2));
          const px = ax + t * dx, py = ay + t * dy;
          // Footpaths count, but a road a few metres further is preferred: you
          // see the most of a street from its middle.
          const d = Math.hypot(px, py) + (f.properties.class === 'path' ? 8 : 0);
          if (d < 150 && (!best || d < best.d)) best = { d, px, py, dx, dy };
        }
      }
    }
    if (!best) return null;
    const along = (Math.atan2(best.dx, best.dy) / RAD + 360) % 360;
    const diff = Math.abs(((along - brg + 540) % 360) - 180);
    const p = index.pushOut(c.lng + best.px / kx, c.lat + best.py / M_LAT, 1.5);
    return { ...p, brg: diff <= 90 ? along : (along + 180) % 360 };
  }

  function enterStreet() {
    if (street || !active) return;
    const back = { center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(),
      pitch: map.getPitch() };
    // You land on the street nearest the spot you were looking at - the
    // middle of the screen - facing down it.
    const c = map.getCenter();
    let brg = map.getBearing();
    let at = snapToStreet(c, brg);
    if (at) brg = at.brg;
    else at = index.pushOut(c.lng, c.lat, 1.5);
    street = { lat: at.lat, lng: at.lng, brg, pitch: STREET_PITCH, back };
    map.setMaxZoom(24);
    for (const h of ['dragPan', 'dragRotate', 'scrollZoom', 'touchZoomRotate', 'doubleClickZoom',
      'keyboard', 'touchPitch']) {
      if (map[h]) map[h].disable();
    }
    attachStreetInput();
    paintControls();
    // On a phone the list takes the bottom half of the screen, and the street
    // wants all of it. Pulled down to its handle; it comes back up by hand.
    // First, because the camera's height is worked out from the map's, and
    // the map grows when the list goes.
    const before = map.getContainer().clientHeight;
    if (typeof collapsePanel === 'function') collapsePanel();
    const go = () => { if (street) map.easeTo({ ...poseFor(street), duration: 1600, essential: true }); };
    if (map.getContainer().clientHeight !== before || window.innerWidth <= 760) setTimeout(go, 300);
    else go();
    say(matchMedia('(pointer: coarse)').matches
      ? 'גרירה: להסתכל מסביב · הקשה על הקרקע: ללכת לשם'
      : 'גרירה: להסתכל מסביב · לחיצה על הקרקע: ללכת לשם · חיצים: הליכה וסיבוב', 6000);
  }

  function leaveStreet(animate) {
    if (!street) return;
    const back = street.back;
    cancelAnimationFrame(walkAnim);
    street = null;
    detachStreetInput();
    for (const h of ['dragPan', 'dragRotate', 'scrollZoom', 'touchZoomRotate', 'doubleClickZoom',
      'keyboard', 'touchPitch']) {
      if (map[h]) map[h].enable();
    }
    // Back above where you walked to rather than where you came down: the
    // walk is how you got somewhere, so the view from above follows you.
    const here = map.getCenter();
    const target = { center: here, zoom: Math.min(back.zoom, 18), bearing: map.getBearing(),
      pitch: Math.min(back.pitch, 70) };
    map.setMaxZoom(22);
    if (animate) map.easeTo({ ...target, duration: 1200 });
    else map.jumpTo(target);
    paintControls();
    if (typeof scheduleSync === 'function') scheduleSync();
  }

  /* Drag to look, tap to walk. A pointer that moves less than a few pixels
   * between down and up is a tap; anything more is a look. One finger only:
   * a second finger is ignored rather than turned into a zoom, because in
   * the street there is nothing to zoom - you walk closer instead. */
  let input = null;

  function attachStreetInput() {
    const canvas = map.getCanvasContainer();
    let down = null;
    const onDown = (e) => {
      if (down || e.button > 0) return;
      cancelAnimationFrame(walkAnim);
      down = { id: e.pointerId, x: e.clientX, y: e.clientY, brg: street.brg, pitch: street.pitch,
        moved: false };
      canvas.setPointerCapture(e.pointerId);
    };
    const onMove = (e) => {
      if (!down || e.pointerId !== down.id) return;
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (!down.moved && Math.hypot(dx, dy) < 6) return;
      down.moved = true;
      // Grab-the-world, as in Street View: drag left and the scene moves
      // left with your finger, which means you turned right. The rate is
      // set so a drag across the screen is about a field of view and a half.
      const h = canvas.clientHeight || 800;
      const degPerPx = 60 / h;
      street.brg = (down.brg - dx * degPerPx + 360) % 360;
      street.pitch = Math.min(Math.max(down.pitch - dy * degPerPx, STREET_PITCH_MIN), PITCH_MAX);
      placeCamera();
    };
    const onUp = (e) => {
      if (!down || e.pointerId !== down.id) return;
      const tap = !down.moved;
      down = null;
      if (tap) walkToward(e);
    };
    const onWheel = (e) => {
      e.preventDefault();
      walk(-Math.sign(e.deltaY) * 6, 0);
    };
    const onKey = (e) => {
      if (e.target.closest && e.target.closest('input, textarea, [contenteditable]')) return;
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'KeyW', 'KeyS', 'KeyA', 'KeyD',
        'Escape'].includes(e.code)) return;
      e.preventDefault();
      if (e.code === 'Escape') { leaveStreet(true); return; }
      if (e.type === 'keydown') {
        if (!keys.size) startKeyLoop();
        keys.add(e.code);
      } else keys.delete(e.code);
    };
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    addEventListener('keydown', onKey);
    addEventListener('keyup', onKey);
    input = { canvas, onDown, onMove, onUp, onWheel, onKey };
  }

  function detachStreetInput() {
    if (!input) return;
    const { canvas, onDown, onMove, onUp, onWheel, onKey } = input;
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerup', onUp);
    canvas.removeEventListener('pointercancel', onUp);
    canvas.removeEventListener('wheel', onWheel);
    removeEventListener('keydown', onKey);
    removeEventListener('keyup', onKey);
    keys.clear();
    input = null;
  }

  /** Walk `fwd` metres ahead and `side` to the right, now. */
  function walk(fwd, side) {
    let p = step(street.lat, street.lng, street.brg, fwd);
    p = step(p.lat, p.lng, street.brg + 90, side);
    // A step into a wall is not taken; a step looking further ahead (half a
    // metre) keeps the camera from ending up with its nose in the plaster.
    const probe = step(street.lat, street.lng, street.brg, fwd + Math.sign(fwd) * 0.5);
    if (index.firstWall(street, probe) < 1) return;
    street.lat = p.lat;
    street.lng = p.lng;
    placeCamera();
  }

  /** Arrows and WASD, held: up and down walk, left and right turn. */
  function startKeyLoop() {
    let last = performance.now();
    const tick = (now) => {
      if (!street || !keys.size) return;
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const has = (...k) => k.some((x) => keys.has(x));
      const fwd = (has('ArrowUp', 'KeyW') ? 1 : 0) - (has('ArrowDown', 'KeyS') ? 1 : 0);
      const turn = (has('ArrowRight', 'KeyD') ? 1 : 0) - (has('ArrowLeft', 'KeyA') ? 1 : 0);
      street.brg = (street.brg + turn * 70 * dt + 360) % 360;
      walk(fwd * 6 * dt, 0);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /** A tap on the ground walks you there, the way a tap does in Street View.
   *  A tap at the sky, or further than TAP_MAX, walks that far towards it. */
  function walkToward(e) {
    const r = map.getCanvasContainer().getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const s = street;
    let target;
    const horizon = r.height * (0.5 - FOCAL / Math.tan(s.pitch * RAD));
    if (y > horizon + 4) {
      const ll = map.unproject([x, y]);
      target = { lat: ll.lat, lng: ll.lng };
    }
    const kx = M_LAT * Math.cos(s.lat * RAD);
    let dE, dN;
    if (target) {
      dE = (target.lng - s.lng) * kx;
      dN = (target.lat - s.lat) * M_LAT;
    } else {
      // Above the horizon: the direction of the tap, as far as TAP_MAX.
      const a = s.brg + ((x - r.width / 2) / r.height) * 37;
      dE = Math.sin(a * RAD) * TAP_MAX;
      dN = Math.cos(a * RAD) * TAP_MAX;
    }
    let d = Math.hypot(dE, dN);
    if (d < 0.5) return;
    // Stop a step short: walking onto the exact spot you tapped puts it
    // under your feet and out of the picture.
    const go = Math.min(d, TAP_MAX) - Math.min(2, d * 0.2);
    dE *= go / d; dN *= go / d; d = go;
    const from = { lat: s.lat, lng: s.lng };
    let to = { lat: s.lat + dN / M_LAT, lng: s.lng + dE / kx };
    // A tap on a house: the ground point behind its wall is where unproject
    // lands, since it does not see extrusions. Walk up to the wall and stop
    // a metre and a half short of it.
    const wall = index.firstWall(from, to);
    if (wall < 1) {
      const k = Math.max(0, (wall * d - 1.5) / d);
      if (k * d < 0.5) return;
      dE *= k; dN *= k; d *= k;
      to = { lat: s.lat + dN / M_LAT, lng: s.lng + dE / kx };
    }
    // Turn to face where you are going, the short way round, as you set off.
    const brgTo = (Math.atan2(dE, dN) / RAD + 360) % 360;
    let turn = ((brgTo - s.brg + 540) % 360) - 180;
    if (Math.abs(turn) < 25) turn = 0;      // close enough to ahead: keep looking where you were
    const brg0 = s.brg;
    const T = Math.max(0.5, d / WALK_SPEED) * 1000;
    const t0 = performance.now();
    cancelAnimationFrame(walkAnim);
    const frame = (now) => {
      if (!street) return;
      const k = Math.min((now - t0) / T, 1);
      const e2 = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      street.lat = from.lat + (to.lat - from.lat) * e2;
      street.lng = from.lng + (to.lng - from.lng) * e2;
      street.brg = (brg0 + turn * Math.min(k * 2, 1) + 360) % 360;
      placeCamera();
      if (k < 1) walkAnim = requestAnimationFrame(frame);
    };
    walkAnim = requestAnimationFrame(frame);
  }

  return {
    attach, detach, load,
    isActive: () => active,
    inStreet: () => !!street,
    enterStreet, leaveStreet,
    // for the browser tests
    _state: () => street && { ...street, back: undefined },
    _inside: (lng, lat) => !!index.at(lng, lat),
    /** Turn to the nearest wall within 50 m and walk 100 m at it. */
    _walkInto() {
      if (!street) return { ok: false, why: 'not in the street' };
      let best = null;
      for (let b = 0; b < 360; b += 5) {
        const t = index.firstWall(street, step(street.lat, street.lng, b, 50));
        if (t < 1 && (!best || t < best.t)) best = { b, t };
      }
      if (!best) return { ok: false, why: 'no wall within 50 m' };
      street.brg = best.b;
      const from = { lat: street.lat, lng: street.lng };
      for (let i = 0; i < 200; i++) walk(0.5, 0);
      const went = Math.hypot((street.lat - from.lat) * M_LAT,
        (street.lng - from.lng) * M_LAT * Math.cos(from.lat * RAD));
      return { ok: !index.at(street.lng, street.lat) && went <= best.t * 50 + 0.01,
        wall: best.t * 50, went };
    }
  };
})();
