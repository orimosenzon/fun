/* Layers - the map is no longer one dataset.
 *
 * Three kinds of layer share one registry:
 *   trails   the initiative's own shortcuts, from the community My Maps
 *   network  the moshava's cycling-network plan, existing and proposed
 *   drafts   trails recorded on this device, held by draft.js
 *
 * Everything downstream - the list, the search, the detail pane, navigation -
 * reads from `visibleSegments()`, so turning a layer off removes it from the
 * whole app rather than only from the map.
 *
 * These files are plain scripts with no bundler, so they share one global
 * scope: `map`, `select`, `escapeHtml` and friends come from app.js, which is
 * loaded last and calls in here only after its data has arrived.
 */
'use strict';

const Layers = (() => {

  const list = [];                    // draw order, bottom first
  const index = new Map();            // segment/waypoint id -> {layer, item}
  const feature = new Map();          // item id -> {src, fid} for feature-state

  let onChange = () => {};            // set by app.js, re-renders the list

  const byId = (id) => list.find((l) => l.id === id);
  const item = (id) => (index.get(id) || {}).item;
  const layerOf = (id) => (index.get(id) || {}).layer;

  /* Stored per browser, so a visitor who turns the cycling plan on keeps it on.
   * Only the on/off flags are stored - never the data itself, which is rebuilt
   * from the JSON on every load. */
  const PREF = 'dk.layers.v1';

  function loadPrefs() {
    try {
      return JSON.parse(localStorage.getItem(PREF) || '{}');
    } catch (err) {
      return {};
    }
  }

  function savePrefs() {
    const on = {};
    list.forEach((l) => { on[l.id] = l.on; });
    try {
      localStorage.setItem(PREF, JSON.stringify(on));
    } catch (err) {
      /* private mode: the toggles simply do not persist */
    }
  }

  /* ---------- registry ---------- */

  function add(layer) {
    layer.segments = layer.segments || [];
    layer.waypoints = layer.waypoints || [];
    list.push(layer);
    reindex();
    return layer;
  }

  function reindex() {
    index.clear();
    list.forEach((layer) => {
      layer.segments.forEach((s) => index.set(s.id, { layer, item: s }));
      layer.waypoints.forEach((w) => index.set(w.id, { layer, item: w }));
    });
  }

  function init(trails, network) {
    const prefs = loadPrefs();

    add({
      id: 'trails',
      kind: 'trails',
      name: 'שבילי היוזמה',
      short: 'שבילים',
      color: '#1b5e20',
      note: 'קיצורי הדרך שמופו על ידי יוזמת דרך קיצור, מתוך המפה החיה ב-My Maps.',
      source: trails.source,
      on: prefs.trails !== false,          // the point of the app; on unless muted
      segments: trails.segments,
      waypoints: trails.waypoints
    });

    (network ? network.layers : []).forEach((l) => add({
      ...l,
      kind: 'network',
      on: prefs[l.id] === true             // planning data is opt-in
    }));

    // Populated by draft.js once IndexedDB answers.
    add({
      id: 'drafts',
      kind: 'drafts',
      name: 'הטיוטות שלי',
      short: 'טיוטות',
      color: '#8e24aa',
      dash: true,
      note: 'שבילים שהקלטת או ציירת במכשיר הזה. נשמרים כאן בלבד, עד שתשלח אותם.',
      on: prefs.drafts !== false,
      segments: []
    });

    // The list is drawn bottom-first, but the plan should sit *under* the
    // trails on the map, so the map order is the reverse of the panel order.
    list.sort((a, b) => order(a) - order(b));
    reindex();
  }

  const RANK = { network: 0, trails: 1, drafts: 2 };
  const order = (l) => RANK[l.kind];

  /* ---------- what the rest of the app sees ---------- */

  const visible = () => list.filter((l) => l.on);
  const visibleSegments = () => visible().flatMap((l) => l.segments);
  const visibleWaypoints = () => visible().flatMap((l) => l.waypoints);

  function stats() {
    const segs = visibleSegments();
    return {
      segments: segs.length,
      length: segs.reduce((sum, s) => sum + (s.length || 0), 0),
      waypoints: visibleWaypoints().length,
      photos: segs.reduce((sum, s) => sum + (s.photos ? s.photos.length : 0), 0)
    };
  }

  /* ---------- map ---------- */

  const srcId = (id) => `src-${id}`;
  const lineId = (id) => `ln-${id}`;
  const hitId = (id) => `hit-${id}`;

  function geojson(layer) {
    return {
      type: 'FeatureCollection',
      features: layer.segments.map((seg, i) => {
        feature.set(seg.id, { src: srcId(layer.id), fid: i });
        return {
          type: 'Feature',
          id: i,
          properties: { id: seg.id, color: seg.color || layer.color },
          geometry: {
            type: 'LineString',
            coordinates: seg.path.map(([lat, lng]) => [lng, lat])
          }
        };
      })
    };
  }

  /** Build every source and layer. Re-run after each style change, because
   *  setStyle drops anything we added. */
  function addToMap() {
    if (typeof map === 'undefined' || !map) return;
    feature.clear();

    list.forEach((layer) => {
      const src = srcId(layer.id);
      if (map.getSource(src)) {
        map.getSource(src).setData(geojson(layer));
      } else {
        map.addSource(src, { type: 'geojson', data: geojson(layer) });
      }
      if (map.getLayer(lineId(layer.id))) return;

      const sel = ['boolean', ['feature-state', 'sel'], false];
      const dim = ['boolean', ['feature-state', 'dim'], false];
      const paint = {
        'line-color': ['get', 'color'],
        'line-width': ['case', sel, 8, layer.kind === 'network' ? 3.5 : 5],
        // Unselected lines stay clearly readable: picking one trail should not
        // stop you browsing straight on to the next one from the map.
        'line-opacity': ['case', sel, 1, dim, 0.55,
          layer.kind === 'network' ? 0.75 : 0.82]
      };
      // A dashed line reads as "not there yet" for the proposed network, and as
      // "not published yet" for a draft.
      if (layer.dash) paint['line-dasharray'] = [2, 1.6];

      map.addLayer({
        id: lineId(layer.id),
        type: 'line',
        source: src,
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint
      });

      // A fat transparent line on top, so a fingertip does not have to land on
      // the stroke itself.
      map.addLayer({
        id: hitId(layer.id),
        type: 'line',
        source: src,
        paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 22 }
      });

      map.on('click', hitId(layer.id), (e) => {
        if (e.features && e.features.length) select(e.features[0].properties.id, false);
      });
      map.on('mouseenter', hitId(layer.id), () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', hitId(layer.id), () => { map.getCanvas().style.cursor = ''; });
    });

    applyVisibility();
  }

  function applyVisibility() {
    if (typeof map === 'undefined' || !map) return;
    list.forEach((layer) => {
      const v = layer.on ? 'visible' : 'none';
      [lineId(layer.id), hitId(layer.id)].forEach((id) => {
        if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', v);
      });
    });
  }

  /** Push a layer's segments back into its map source, after drafts change. */
  function refresh(id) {
    const layer = byId(id);
    if (!layer) return;
    reindex();
    if (typeof map !== 'undefined' && map && map.getSource(srcId(id))) {
      map.getSource(srcId(id)).setData(geojson(layer));
    }
    onChange();
  }

  /** Highlight one item and dim the rest, across every layer. */
  function highlight(id) {
    if (typeof map === 'undefined' || !map) return;
    feature.forEach(({ src, fid }, ourId) => {
      if (!map.getSource(src)) return;
      map.setFeatureState({ source: src, id: fid },
        { sel: ourId === id, dim: id != null && ourId !== id });
    });
  }

  /* ---------- the layer sheet ---------- */

  function render() {
    const box = document.getElementById('layer-list');
    box.innerHTML = list.slice().reverse().map((layer) => {
      const n = layer.segments.length;
      const metres = layer.segments.reduce((sum, s) => sum + (s.length || 0), 0);
      const sub = n
        ? `${n} מקטעים · ${metres >= 1000 ? (metres / 1000).toFixed(1) + ' ק"מ' : metres + ' מ׳'}`
        : (layer.kind === 'drafts' ? 'אין עדיין. הקלט או צייר שביל.' : 'ריק');

      return `<label class="lay${layer.on ? ' on' : ''}" data-id="${layer.id}">
        <input type="checkbox" ${layer.on ? 'checked' : ''}>
        <span class="lay-swatch${layer.dash ? ' dash' : ''}"
              style="--c:${layer.color}"></span>
        <span class="lay-txt">
          <span class="lay-nm">${escapeHtml(layer.name)}</span>
          <span class="lay-sub">${escapeHtml(sub)}</span>
          <span class="lay-note">${escapeHtml(layer.note || '')}</span>
        </span>
      </label>`;
    }).join('');

    box.querySelectorAll('.lay input').forEach((box2) => {
      box2.addEventListener('change', () => {
        const layer = byId(box2.closest('.lay').dataset.id);
        layer.on = box2.checked;
        box2.closest('.lay').classList.toggle('on', layer.on);
        savePrefs();
        applyVisibility();
        onChange();
      });
    });

    const credit = list.find((l) => l.credit);
    document.getElementById('layer-credit').textContent =
      credit ? 'רשת הרכיבה: ' + credit.credit : '';
  }

  function openSheet() {
    render();
    document.getElementById('layer-sheet').hidden = false;
  }

  const closeSheet = () => { document.getElementById('layer-sheet').hidden = true; };

  return {
    list, init, add, byId, item, layerOf, reindex,
    visible, visibleSegments, visibleWaypoints, stats,
    addToMap, applyVisibility, refresh, highlight,
    openSheet, closeSheet, render,
    set onChange(fn) { onChange = fn; }
  };
})();
