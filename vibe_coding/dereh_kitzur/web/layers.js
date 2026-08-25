/* Layers - the map is no longer one dataset.
 *
 * Four kinds of layer share one registry:
 *   trails   the initiative's own shortcuts, plus any further trail layer an
 *            editor has made. A trail carries the id of its layer; the ones
 *            that carry none belong to the initiative's original layer.
 *   network  the moshava's cycling-network plan, existing and proposed
 *   places   places drawn from pardespedia, each with the wiki's own summary,
 *            photo and a link back to the article
 *   drafts   trails recorded on this device, held by draft.js
 *
 * Everything downstream - the list, the search, the detail pane, navigation -
 * reads from `visibleSegments()` and `visibleWaypoints()`, so turning a layer
 * off removes it from the whole app rather than only from the map.
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
  const wired = new Set();            // layer ids whose map handlers are bound

  let onChange = () => {};            // set by app.js, re-renders the list

  const byId = (id) => list.find((l) => l.id === id);
  const item = (id) => (index.get(id) || {}).item;
  const layerOf = (id) => (index.get(id) || {}).layer;

  const TRAILS_ID = 'trails';
  const PLACES_ID = 'places';
  const PENDING_ID = 'pending';

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
    return layer;
  }

  function reindex() {
    index.clear();
    list.forEach((layer) => {
      layer.segments.forEach((s) => index.set(s.id, { layer, item: s }));
      layer.waypoints.forEach((w) => index.set(w.id, { layer, item: w }));
    });
  }

  /** A pardespedia article turned into something the list and the detail pane
   *  can treat like any other point on the map.
   *
   *  The position lives under `geo` in the file, because build_places.py has to
   *  tell a hand-dropped pin from one it guessed off an address, and only ever
   *  overwrite the guesses. The rest of the app just wants lat and lng. */
  function toPlace(raw, colours) {
    const geo = raw.geo || null;
    return {
      ...raw,
      lat: geo ? geo.lat : null,
      lng: geo ? geo.lng : null,
      unplaced: !geo,
      approx: !!geo && (geo.source === 'nearby' || geo.source === 'street'),
      geoSource: geo ? geo.source : null,
      place: true,
      color: colours[raw.group] || '#6a1b9a'
    };
  }

  /** The trail layers, rebuilt from the document.
   *
   *  `on` comes from a lookup rather than a stored map so the same code serves
   *  the first load, where it reads the browser's preferences, and a reload
   *  after a write, where it reads whatever is currently switched on. */
  function buildTrailLayers(trails, isOn) {
    const custom = trails.layers || [];
    const mine = new Set(custom.map((l) => l.id));

    // A trail whose layer was deleted, or which names a layer this app has
    // never heard of, belongs with the initiative's own rather than nowhere.
    const home = (it) => (it.layer && mine.has(it.layer) ? it.layer : TRAILS_ID);

    add({
      id: TRAILS_ID,
      kind: 'trails',
      name: 'שבילי היוזמה',
      short: 'שבילים',
      color: '#1b5e20',
      note: 'קיצורי הדרך שמופו על ידי יוזמת דרך קיצור.',
      source: trails.source,
      on: isOn(TRAILS_ID),                 // the point of the app; on unless muted
      segments: trails.segments.filter((s) => home(s) === TRAILS_ID),
      waypoints: trails.waypoints.filter((w) => home(w) === TRAILS_ID)
    });

    custom.forEach((l) => add({
      ...l,
      kind: 'trails',
      own: true,                           // made in the app, so it can be edited here
      on: isOn(l.id),
      segments: trails.segments.filter((s) => home(s) === l.id),
      waypoints: trails.waypoints.filter((w) => home(w) === l.id)
    }));
  }

  function init(trails, network, places) {
    const prefs = loadPrefs();
    buildTrailLayers(trails, (id) => prefs[id] !== false);

    (network ? network.layers : []).forEach((l) => add({
      ...l,
      kind: 'network',
      on: prefs[l.id] === true             // planning data is opt-in
    }));

    if (places && places.places && places.places.length) {
      const colours = {};
      (places.groups || []).forEach((g) => { colours[g.name] = g.color; });
      add({
        id: PLACES_ID,
        kind: 'places',
        name: 'מקומות מפרדספדיה',
        short: 'מקום',
        color: '#7b1fa2',
        note: 'בתי קפה, גנים, מוסדות ואתרי הנצחה, עם התקציר והתמונה מהערך בוויקי.',
        credit: 'pardespedia.info',
        groups: places.groups || [],
        on: prefs[PLACES_ID] === true,     // hundreds of pins; opt-in
        waypoints: places.places.map((p) => toPlace(p, colours))
      });
    }

    // Populated from the worker, and only while edit mode is on: a trail nobody
    // has looked at yet is not something to show a visitor as if it were part
    // of the map.
    add({
      id: PENDING_ID,
      kind: 'pending',
      name: 'ממתינים לאישור',
      short: 'ממתין',
      color: '#f9a825',
      dash: true,
      note: 'שבילים שתושבים שלחו ועוד לא אושרו. גלוי רק במצב עריכה.',
      on: true,
      segments: []
    });

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

  const RANK = { network: 0, places: 1, trails: 2, pending: 3, drafts: 4 };
  const order = (l) => RANK[l.kind];

  /** Rebuild the trail layers after a write, without disturbing the drafts
   *  layer or anyone's on/off choices. */
  function resetTrails(trails) {
    const was = {};
    list.forEach((l) => { if (l.kind === 'trails') was[l.id] = l.on; });

    const others = list.filter((l) => l.kind !== 'trails');
    list.length = 0;
    buildTrailLayers(trails, (id) => was[id] !== false);
    others.forEach((l) => list.push(l));
    list.sort((a, b) => order(a) - order(b));
    reindex();
    savePrefs();
    if (typeof map !== 'undefined' && map) addToMap();
    onChange();
  }

  /** Fill the review queue layer from what the worker holds. */
  function setPending(items) {
    const layer = byId(PENDING_ID);
    if (!layer) return;
    layer.segments = (items || [])
      .filter((it) => it.path && it.path.length > 1)
      .map((it) => ({ ...it, color: '#f9a825', pending: true }));
    reindex();
    if (typeof map !== 'undefined' && map && map.getSource(srcId(PENDING_ID))) {
      map.getSource(srcId(PENDING_ID)).setData(geojson(layer));
    }
    applyVisibility();
    onChange();
  }

  /** Rebuild the places layer in place, after a pin is dropped or removed. */
  function resetPlaces(places) {
    const layer = byId(PLACES_ID);
    if (!layer || !places) return;
    const colours = {};
    (places.groups || []).forEach((g) => { colours[g.name] = g.color; });
    layer.waypoints = places.places.map((p) => toPlace(p, colours));
    reindex();
    if (typeof map !== 'undefined' && map && map.getSource(srcId(PLACES_ID))) {
      map.getSource(srcId(PLACES_ID)).setData(geojson(layer));
    }
    onChange();
  }

  /* ---------- what the rest of the app sees ---------- */

  const shown = (l) => l.on && !(l.kind === 'pending' && !Store.isEditor());
  const visible = () => list.filter(shown);
  const visibleSegments = () => visible().flatMap((l) => l.segments);
  const visibleWaypoints = () => visible().flatMap((l) => l.waypoints);

  /** Waypoints drawn as HTML markers, which is the initiative's own handful.
   *  Places are hundreds of points and go through a GL layer instead. */
  const markerWaypoints = () =>
    visible().filter((l) => l.kind !== 'places').flatMap((l) => l.waypoints);

  const trailLayers = () => list.filter((l) => l.kind === 'trails' && l.id !== 'drafts');

  function stats() {
    // The queue is not part of the map yet, so it gets its own number rather
    // than inflating the count of what is actually out there.
    const segs = visible().filter((l) => l.kind !== 'pending').flatMap((l) => l.segments);
    const waiting = visible().filter((l) => l.kind === 'pending')
      .reduce((n, l) => n + l.segments.length, 0);
    const wps = visible().filter((l) => l.kind !== 'places').flatMap((l) => l.waypoints);
    const places = visible().filter((l) => l.kind === 'places').flatMap((l) => l.waypoints);
    return {
      segments: segs.length,
      length: segs.reduce((sum, s) => sum + (s.length || 0), 0),
      waiting,
      waypoints: wps.length,
      places: places.length,
      unplaced: places.filter((p) => p.unplaced).length,
      photos: [...segs, ...wps, ...places]
        .reduce((sum, s) => sum + (s.photos ? s.photos.length : 0), 0)
    };
  }

  /* ---------- map ---------- */

  const srcId = (id) => `src-${id}`;
  const lineId = (id) => `ln-${id}`;
  const dotId = (id) => `pt-${id}`;
  const labelId = (id) => `lb-${id}`;
  const hitId = (id) => `hit-${id}`;

  const drawnIds = (layer) => (layer.kind === 'places'
    ? [dotId(layer.id), labelId(layer.id), hitId(layer.id)]
    : [lineId(layer.id), hitId(layer.id)]);

  function geojson(layer) {
    if (layer.kind === 'places') {
      let i = 0;
      return {
        type: 'FeatureCollection',
        features: layer.waypoints.filter((p) => !p.unplaced).map((p) => {
          feature.set(p.id, { src: srcId(layer.id), fid: i });
          return {
            type: 'Feature',
            id: i++,
            properties: { id: p.id, color: p.color, name: p.name },
            geometry: { type: 'Point', coordinates: [p.lng, p.lat] }
          };
        })
      };
    }
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

  function addPlaceLayers(layer) {
    const src = srcId(layer.id);
    const sel = ['boolean', ['feature-state', 'sel'], false];
    const dim = ['boolean', ['feature-state', 'dim'], false];

    map.addLayer({
      id: dotId(layer.id),
      type: 'circle',
      source: src,
      paint: {
        'circle-color': ['get', 'color'],
        // A zoom expression has to be the outermost one: MapLibre rejects a
        // ["zoom"] nested inside a ["case"], so the selected state is decided
        // at each stop rather than around the whole thing.
        'circle-radius': ['interpolate', ['linear'], ['zoom'],
          12, ['case', sel, 7, 3.5],
          15, ['case', sel, 9, 5.5],
          18, ['case', sel, 12, 8]],
        'circle-opacity': ['case', dim, 0.45, 0.9],
        'circle-stroke-color': '#fff',
        'circle-stroke-width': ['case', sel, 3, 1.5]
      }
    });

    // Hebrew labels need the style's glyph set to carry the Hebrew range.
    // OpenFreeMap's Noto Sans does, and the satellite style in app.js points
    // its `glyphs` at the same endpoint so both look the same.
    map.addLayer({
      id: labelId(layer.id),
      type: 'symbol',
      source: src,
      minzoom: 14.5,
      layout: {
        'text-field': ['get', 'name'],
        'text-font': ['Noto Sans Regular'],
        'text-size': 12,
        'text-anchor': 'top',
        'text-offset': [0, 0.8],
        'text-max-width': 9,
        'text-padding': 4,
        'text-optional': true          // a label that will not fit is dropped,
      },                               // the dot stays
      paint: {
        'text-color': '#2b2b2b',
        'text-halo-color': 'rgba(255,255,255,0.9)',
        'text-halo-width': 1.6,
        'text-opacity': ['case', ['boolean', ['feature-state', 'dim'], false], 0.5, 1]
      }
    });

    map.addLayer({
      id: hitId(layer.id),
      type: 'circle',
      source: src,
      paint: { 'circle-color': '#000', 'circle-opacity': 0, 'circle-radius': 16 }
    });
  }

  function addLineLayers(layer) {
    const src = srcId(layer.id);
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
  }

  /** Build every source and layer. Re-run after each style change, because
   *  setStyle drops anything we added. */
  function addToMap() {
    if (typeof map === 'undefined' || !map) return;
    feature.clear();

    // A layer deleted since the last run leaves its GL layers and its source
    // behind. Sources go last: removing one still in use throws.
    const live = new Set(list.map((l) => l.id));
    const gone = new Set();
    map.getStyle().layers.forEach((gl) => {
      const match = /^(?:ln|pt|lb|hit)-(.+)$/.exec(gl.id);
      if (match && !live.has(match[1]) && map.getLayer(gl.id)) {
        map.removeLayer(gl.id);
        gone.add(match[1]);
      }
    });
    gone.forEach((id) => {
      if (map.getSource(srcId(id))) map.removeSource(srcId(id));
      wired.delete(id);
    });

    list.forEach((layer) => {
      const src = srcId(layer.id);
      if (map.getSource(src)) {
        map.getSource(src).setData(geojson(layer));
      } else {
        map.addSource(src, { type: 'geojson', data: geojson(layer) });
      }
      if (map.getLayer(hitId(layer.id))) return;

      if (layer.kind === 'places') addPlaceLayers(layer);
      else addLineLayers(layer);

      // A layer-scoped listener outlives the layer it names, and setStyle wipes
      // every layer we added - so without this set, each basemap switch bound
      // another copy of the same three handlers.
      if (wired.has(layer.id)) return;
      wired.add(layer.id);
      map.on('click', hitId(layer.id), (e) => {
        if (e.features && e.features.length) select(e.features[0].properties.id, false);
      });
      map.on('mouseenter', hitId(layer.id), () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', hitId(layer.id), () => { map.getCanvas().style.cursor = ''; });
    });

    applyVisibility();
  }

  /* While the arrange tool is open it draws its own draggable marker for every
   * place, and the layer's dots would sit underneath every one of them. */
  let arranging = false;
  const setArranging = (value) => { arranging = value; applyVisibility(); };

  function applyVisibility() {
    if (typeof map === 'undefined' || !map) return;
    list.forEach((layer) => {
      const hide = (arranging && layer.kind === 'places')
        || (layer.kind === 'pending' && !Store.isEditor());
      const v = layer.on && !hide ? 'visible' : 'none';
      drawnIds(layer).forEach((id) => {
        if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', v);
      });
    });
  }

  /** Push a layer's items back into its map source, after drafts change. */
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

  function summary(layer) {
    if (layer.kind === 'places') {
      const n = layer.waypoints.length;
      const missing = layer.waypoints.filter((p) => p.unplaced).length;
      return `${n} מקומות` + (missing ? ` · ${missing} עוד לא ממוקמים` : '');
    }
    const n = layer.segments.length + layer.waypoints.length;
    if (!n) return layer.kind === 'drafts' ? 'אין עדיין. הקלט או צייר שביל.' : 'ריק';
    const metres = layer.segments.reduce((sum, s) => sum + (s.length || 0), 0);
    const bits = [`${layer.segments.length} מקטעים`];
    if (metres) bits.push(metres >= 1000 ? (metres / 1000).toFixed(1) + ' ק"מ' : metres + ' מ׳');
    if (layer.waypoints.length) bits.push(`${layer.waypoints.length} נקודות`);
    return bits.join(' · ');
  }

  function render() {
    const box = document.getElementById('layer-list');
    const editable = Store.isEditor();

    box.innerHTML = list.slice().reverse()
      .filter((layer) => layer.kind !== 'pending' || Store.isEditor())
      .map((layer) => `
      <div class="lay-row">
        <label class="lay${layer.on ? ' on' : ''}" data-id="${layer.id}">
          <input type="checkbox" ${layer.on ? 'checked' : ''}>
          <span class="lay-swatch${layer.dash ? ' dash' : ''}"
                style="--c:${layer.color}"></span>
          <span class="lay-txt">
            <span class="lay-nm">${escapeHtml(layer.name)}</span>
            <span class="lay-sub">${escapeHtml(summary(layer))}</span>
            <span class="lay-note">${escapeHtml(layer.note || '')}</span>
          </span>
        </label>
        ${editable && layer.own ? `<button class="lay-edit" data-edit="${layer.id}"
          aria-label="עריכת השכבה ${escapeHtml(layer.name)}">עריכה</button>` : ''}
      </div>
      ${editable && layer.kind === 'places' ? `
        <button class="lay-add places" data-arrange="1">סידור מיקומי המקומות${
          layer.waypoints.filter((p) => p.unplaced).length
            ? ` · ${layer.waypoints.filter((p) => p.unplaced).length} עוד לא ממוקמים` : ''}
        </button>` : ''}`).join('') + (editable ? `
      <button class="lay-add" data-newlayer="1">+ שכבת שבילים חדשה</button>` : '');

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

    const credits = list.filter((l) => l.on && l.credit).map((l) => l.credit);
    document.getElementById('layer-credit').textContent =
      credits.length ? 'מקורות: ' + credits.join(' · ') : '';
  }

  function openSheet() {
    render();
    document.getElementById('layer-sheet').hidden = false;
  }

  const closeSheet = () => { document.getElementById('layer-sheet').hidden = true; };

  return {
    list, init, add, byId, item, layerOf, reindex, resetTrails, resetPlaces,
    visible, visibleSegments, visibleWaypoints, markerWaypoints, trailLayers, stats,
    addToMap, applyVisibility, refresh, highlight, setArranging, setPending,
    openSheet, closeSheet, render,
    TRAILS_ID, PLACES_ID, PENDING_ID,
    set onChange(fn) { onChange = fn; }
  };
})();
