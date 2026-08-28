/* דרך קיצור - trail browser for the walking shortcuts of Pardes Hanna-Karkur.
 *
 * Data arrives as layers (see layers.js): the initiative's own trails from
 * data/trails.json, the moshava's cycling-network plan from data/layers.json,
 * and whatever the visitor has recorded themselves, held by draft.js.
 *
 * No API key is used anywhere: base tiles are open, and Street View is reached
 * by handing Google a URL rather than embedding a paid panorama widget.
 */
'use strict';

/* Elevation for the tilted view. Terrarium tiles are free and need no key.
 * Local relief is only about 46 m across the whole moshava and 4 m along a
 * typical trail, so it needs exaggerating to read as anything at all. */
const DEM = {
  type: 'raster-dem',
  tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
  tileSize: 256,
  encoding: 'terrarium',
  maxzoom: 14,
  attribution: 'Elevation: Mapzen / AWS Open Data'
};
const TERRAIN_X = 2.5;
const TILTED = 58;

const BASEMAPS = [
  { name: 'רחובות', style: 'https://tiles.openfreemap.org/styles/liberty' },
  {
    name: 'לוויין',
    style: {
      version: 8,
      // Hand-built styles carry no glyphs, and without them the place labels
      // silently render as nothing. Point at the same font endpoint the street
      // style uses, whose Noto Sans covers the Hebrew range.
      glyphs: 'https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf',
      sources: {
        sat: {
          type: 'raster',
          tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
          tileSize: 256,
          maxzoom: 19,
          attribution: 'Esri, Maxar, Earthstar Geographics'
        }
      },
      layers: [{ id: 'sat', type: 'raster', source: 'sat' }]
    }
  }
];

const el = (id) => document.getElementById(id);

/* The tilted view is drawn with WebGL, which a handful of old phones lack.
 * Everything that is not the map - list, search, photos, Street View links -
 * works without it, so the map is optional rather than fatal. */
function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext &&
      (c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl')));
  } catch (err) {
    return false;
  }
}

// Probe the canvas directly. maplibregl.supported() looks like the obvious
// check but was removed in MapLibre v3, so calling it silently reports "no
// WebGL" on every browser and the map never gets built.
const hasGL = !!window.maplibregl && webglAvailable();

/* Hebrew in a GL label is laid out glyph by glyph in logical order, which
 * renders it backwards: דרך הנדיב comes out בידנה ךרד. WebGL has no bidi of
 * its own, so the reordering has to be loaded in.
 *
 * This was invisible until now only because the app drew every label of its
 * own as an HTML marker, where the browser does the reordering. The basemap's
 * own street names have been mirrored the whole time. Four hundred places are
 * too many for HTML markers - they need the collision handling a symbol layer
 * has - so the plugin goes in and the street names come out right as well.
 *
 * It is served from here rather than from a CDN. The plugin runs inside
 * MapLibre's worker, which is built from a blob URL, and a cross-origin
 * importScripts into a blob worker fails with nothing but "failed to import
 * scripts" - no status, no way to retry. Local also means the labels are still
 * the right way round with no signal, which is the situation on a trail. */
const RTL_PLUGIN = 'vendor/mapbox-gl-rtl-text.min.js';

if (hasGL) {
  try {
    // Signatures differ across MapLibre majors: older ones take a callback,
    // newer ones return a promise. Tolerate either, and carry on without it -
    // mirrored labels are bad, a blank map is worse.
    const pending = maplibregl.setRTLTextPlugin(RTL_PLUGIN, () => {}, false);
    if (pending && pending.catch) pending.catch(() => {});
  } catch (err) {
    console.warn('RTL text plugin unavailable', err);
  }
}

const map = hasGL ? new maplibregl.Map({
  container: 'map',
  style: BASEMAPS[0].style,
  center: [34.966, 32.4755],
  zoom: 13.5,
  pitch: 0,
  maxPitch: 80,
  attributionControl: { compact: true }
}) : null;

if (map) {
  // visualizePitch gives the compass-and-tilt puck, the same control Google
  // offers for looking at the map from an angle.
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');
  window.__map = map;   // handle for debugging and for the browser tests
}

let DATA = null;          // the trails document, for its source link and bounds
let PLACES = null;        // the pardespedia document, kept for rebuilds
let baseIndex = 0;
let here = null;          // {lat, lng} once geolocation succeeds
let hereMarker = null;
let selectedId = null;
let sortMode = 'length';
let wpMarkers = [];       // waypoint pins, rebuilt when layers are toggled

/* ---------- helpers ---------- */

const metres = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + ' ק"מ' : m + ' מ׳');

/* Hebrew says "קישור אחד" and "2 קישורים", never "1 קישורים". */
const plural = (n, one, many) => (n === 1 ? one : `${n} ${many}`);

function distance(a, b) {
  const R = 6371000, rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad, dLng = (b.lng - a.lng) * rad;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

/** Where an item sits, for "nearest to me" sorting: a segment counts from
 *  whichever of its two entrances is closer, since that is what you walk to. */
function anchors(item) {
  if (item.entries) return item.entries.map((e) => ({ lat: e.lat, lng: e.lng }));
  return [{ lat: item.lat, lng: item.lng }];
}

function nearestMetres(item) {
  // A pardespedia place with no pin yet has no position to measure from, and
  // an unguarded distance() would sort it to the top of "nearest to me" on a
  // NaN rather than dropping it to the bottom.
  if (!here || item.lat == null && !item.entries) return Infinity;
  return Math.min(...anchors(item).map((a) => distance(here, a)));
}

/* Official Google URL schemes. Free, no key, and on a phone they open the
 * Google Maps app rather than a browser tab. */
const panoUrl = (lat, lng, heading) =>
  `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lng}` +
  `&heading=${heading || 0}&pitch=0&fov=80`;

const walkUrl = (lat, lng) =>
  `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}` +
  `&travelmode=walking&hl=iw`;

function icon(path) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`;
}

const I_PANO = 'M12 2a7 7 0 00-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 00-7-7zm0 9.5A2.5 2.5 0 1112 6.5a2.5 2.5 0 010 5z';
const I_WALK = 'M13.5 5.5a2 2 0 100-4 2 2 0 000 4zM9.8 8.9L7 23h2.1l1.8-8 2.1 2v6h2v-7.5l-2.1-2 .6-3A7 7 0 0019 13v-2a5 5 0 01-4.2-2.4l-1-1.6c-.4-.6-1-1-1.8-1-.3 0-.5 0-.8.2L6 8.3V13h2V9.6l1.8-.7z';

/* ---------- map layers ---------- */

function setBasemap(i) {
  if (!map) return;
  baseIndex = i;
  el('basemap').classList.toggle('on', i > 0);
  el('basemap').title = 'רקע: ' + BASEMAPS[i].name;
  // setStyle drops every source and layer we added, so applyOverlays runs
  // again on the style.load that follows.
  map.setStyle(BASEMAPS[i].style);
}

/* Everything we add on top of whichever base style is loaded. Re-run on every
 * style change, because a style swap wipes custom sources and layers. */
function applyOverlays() {
  if (!map) return;
  if (!map.getSource('dem')) map.addSource('dem', DEM);
  map.setTerrain({ source: 'dem', exaggeration: TERRAIN_X });

  // The vector style extrudes buildings at high zoom, but only 1 of 400
  // buildings here carries a height in OSM, so they all come out the same
  // default box. That is noise, not information - the relief is the point.
  for (const layer of map.getStyle().layers) {
    if (layer.type === 'fill-extrusion') {
      map.setLayoutProperty(layer.id, 'visibility', 'none');
    }
  }

  if (DATA) {
    Layers.addToMap();
    drawWaypoints();
    if (Drafts.isDrafting()) Drafts.paintEditor();
  }
}

function drawWaypoints() {
  if (!map) return;
  wpMarkers.forEach((m) => m.remove());
  wpMarkers = [];

  // Waypoints are HTML markers rather than a symbol layer: markers use the
  // browser's own font, which sidesteps the whole question of whether the
  // style's glyph set covers Hebrew.
  Layers.markerWaypoints().forEach((wp) => {
    const node = document.createElement('div');
    node.className = 'pin';
    node.innerHTML = `<i></i><b>${escapeHtml(wp.name)}</b>`;
    node.addEventListener('click', (e) => { e.stopPropagation(); select(wp.id, false); });
    wpMarkers.push(
      new maplibregl.Marker({ element: node, anchor: 'left' })
        .setLngLat([wp.lng, wp.lat])
        .addTo(map)
    );
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ---------- selection ---------- */

const byId = (id) => Layers.item(id);

/* `fit` recentres the map on the item. Picking from the list should do that,
 * because you have no idea where the trail is. Tapping a trail already drawn
 * on the map should not: you are looking straight at it, and yanking the
 * viewport around makes browsing from one trail to the next impossible. */
function select(id, fit = true) {
  selectedId = id;
  const item = byId(id);
  if (!item) return;

  Layers.highlight(id);
  if (fit && map) {
    if (item.path) {
      const b = new maplibregl.LngLatBounds();
      item.path.forEach(([lat, lng]) => b.extend([lng, lat]));
      map.fitBounds(b, { padding: 60, maxZoom: 18, duration: 700 });
    } else if (item.lat != null) {
      map.easeTo({ center: [item.lng, item.lat], zoom: 18, duration: 700 });
    }
    // A place with no pin yet has nowhere to fly to. The detail pane opens
    // anyway, which is where the editor drops one.
  }
  showDetail(item);
}

function deselect() {
  selectedId = null;
  Layers.highlight(null);
  el('detail-view').hidden = true;
  el('list-view').hidden = false;
  renderList();
}

/* ---------- list ---------- */

/* The list shows exactly what the map shows. Turning a layer off has to remove
 * it from both, otherwise the list offers trails you cannot see. */
function items() {
  const q = el('search').value.trim();
  let all = [...Layers.visibleSegments(), ...Layers.visibleWaypoints()];

  // A place nobody has pinned yet is not on the map, so for a reader it is a
  // row that goes nowhere. An editor is exactly the person who can fix that,
  // so for an editor it stays, and the "לא ממוקמים" sort gathers them up.
  if (!Store.isEditor() && sortMode !== 'unplaced') {
    all = all.filter((it) => !it.unplaced);
  }

  if (q) {
    const needle = q.toLowerCase();
    all = all.filter((it) =>
      [it.name, it.note, it.group, it.address,
       (it.connects || []).join(' '), (it.streets || []).join(' ')]
        .join(' ').toLowerCase().includes(needle));
  }

  if (sortMode === 'length') {
    all.sort((a, b) => (b.length || 0) - (a.length || 0));
  } else if (sortMode === 'photos') {
    all = all.filter((it) => it.photos && it.photos.length);
    all.sort((a, b) => b.photos.length - a.photos.length);
  } else if (sortMode === 'near') {
    all.sort((a, b) => nearestMetres(a) - nearestMetres(b));
  } else if (sortMode === 'unplaced') {
    all = all.filter((it) => it.unplaced);
    all.sort((a, b) => a.name.localeCompare(b.name, 'he'));
  }
  return all;
}

function subtitle(it) {
  const bits = [];
  if (it.length) bits.push(metres(it.length));
  else if (it.place) bits.push(it.group || 'מקום');
  else bits.push('נקודת ציון');
  if (it.unplaced) bits.push('עוד לא ממוקם על המפה');
  else if (it.approx) bits.push('מיקום מקורב');
  if (here) {
    const d = nearestMetres(it);
    if (isFinite(d)) bits.push(d < 1000 ? `${Math.round(d)} מ׳ ממך` : `${(d / 1000).toFixed(1)} ק"מ ממך`);
  }
  if (it.photos && it.photos.length) bits.push(`${it.photos.length} תמונות`);
  if (it.kind) bits.push(it.kind);
  if (it.note) bits.push(it.note);
  return bits.join(' · ');
}

/* With several layers on at once, the colour swatch alone no longer says where
 * a row came from. Only tag the rows that are not the initiative's own. */
function badge(it) {
  const layer = Layers.layerOf(it.id);
  if (!layer || layer.id === Layers.TRAILS_ID) return '';
  // Places all sit in one layer but come in groups, and "בית קפה" says far
  // more on a row than the layer's own name repeated four hundred times.
  const [text, colour] = it.place
    ? [it.group || layer.short, it.color]
    : [layer.short, layer.color];
  return `<span class="tag" style="--c:${colour}">${escapeHtml(text)}</span>`;
}

function renderList() {
  const list = el('list');
  const rows = items();

  if (!rows.length) {
    list.innerHTML = !Layers.visible().length
      ? '<li class="empty-msg">כל השכבות כבויות.<br>פתח את כפתור השכבות והדלק אחת.</li>'
      : sortMode === 'unplaced'
        ? '<li class="empty-msg">כל המקומות ממוקמים. 🎉</li>'
        : '<li class="empty-msg">לא נמצא שביל תואם.</li>';
    return;
  }

  list.innerHTML = rows.map((it) => {
    const glyph = it.place ? (it.unplaced ? '📌' : '🏛️') : (it.path ? (it.draft ? '✏️' : '🥾') : '📍');
    const thumb = it.photos && it.photos.length
      ? `<img class="thumb" src="${it.photos[0].thumb}" alt="" loading="lazy">`
      : `<div class="thumb empty">${glyph}</div>`;
    return `<li class="row${it.id === selectedId ? ' on' : ''}" data-id="${it.id}"
              style="color:${it.color || '#8d6e63'}">
      <span class="swatch"></span>
      ${thumb}
      <span class="txt">
        <span class="nm">${badge(it)}${escapeHtml(it.name)}</span>
        <span class="sub">${escapeHtml(subtitle(it))}</span>
      </span>
    </li>`;
  }).join('');

  list.querySelectorAll('.row').forEach((row) => {
    row.addEventListener('click', () => select(row.dataset.id));
  });
}

/* ---------- detail ---------- */

/** Which way the camera should look at an entrance.
 *
 *  build_data.py works this out properly for the initiative's trails, standing
 *  the camera on the nearest street. Cycling segments and drafts have no such
 *  pass, so aim along the line itself: these run on or beside roads, and
 *  looking down the route is what you want anyway. */
function entryHeading(it, i) {
  const e = it.entries[i];
  if (e.heading != null) return e.heading;
  if (!it.path || it.path.length < 2) return 0;
  const at = i === 0 ? 0 : it.path.length - 1;
  const to = i === 0 ? 1 : it.path.length - 2;
  return Math.round(bearingTo(
    { lat: it.path[at][0], lng: it.path[at][1] },
    { lat: it.path[to][0], lng: it.path[to][1] }));
}

function panoActs(it, labels, hint) {
  return it.entries.map((e, i) => {
    const [vLat, vLng] = e.view || [e.lat, e.lng];
    const weak = e.likely === false;
    const text = weak ? `${e.road} מ׳ מהכביש הקרוב, ייתכן שאין כאן צילום 360` : hint;
    return `<a class="act${weak ? ' weak' : ''}"
        href="${panoUrl(vLat, vLng, entryHeading(it, i))}" target="_blank" rel="noopener">
      ${icon(I_PANO)}
      <span class="lbl">סטריט ויו · ${escapeHtml(labels[i])}
        <span class="hint">${escapeHtml(text)}</span></span>
    </a>`;
  }).join('');
}

/* In-app navigation first: Google would route *around* a shortcut it does not
 * know exists. The hand-off stays as a secondary way to reach the area. */
function navActs(it, hint) {
  const first = (it.entries && it.entries[0]) || it;
  if (first.lat == null) return '';
  return `<button class="act act-nav" id="go">
      ${icon(I_WALK)}
      <span class="lbl">נווט אליי לכאן
        <span class="hint">${escapeHtml(hint)}</span></span>
    </button>
    <a class="act act-sub" href="${walkUrl(first.lat, first.lng)}" target="_blank" rel="noopener">
      <span class="lbl">פתח בגוגל מפות
        <span class="hint">מנווט עד השכונה, לא דרך השביל</span></span>
    </a>`;
}

/** Whatever the item links out to: the wiki article behind a place, plus any
 *  link an editor attached. Rendered for everybody, not only editors - a link
 *  nobody can follow is not a link. */
function linksBlock(it) {
  const links = [];
  if (it.place && it.url) {
    links.push({ url: it.url, title: 'הערך המלא בפרדספדיה', lead: true });
  }
  (it.links || []).forEach((l) => links.push(l));
  if (!links.length) return '';

  return `
    <h3>קישורים</h3>
    <div class="acts">
      ${links.map((l) => `<a class="act act-link${l.lead ? ' act-nav' : ''}"
          href="${escapeHtml(l.url)}" target="_blank" rel="noopener noreferrer">
        <span class="lbl">${escapeHtml(l.title)}
          <span class="hint">${escapeHtml(hostOf(l.url))}</span></span>
      </a>`).join('')}
    </div>`;
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch (err) {
    return '';
  }
}

/** A pardespedia place. The wiki wrote the words and took the photo; this pane
 *  adds the two things a map can offer on top - where it is, and how to walk
 *  there. */
function placeBody(it) {
  if (it.unplaced) {
    return `
      <p class="unplaced">המקום הזה עוד לא מוקם על המפה. בפרדספדיה אין קואורדינטות,
        ולערך הזה לא נמצאה כתובת שאפשר לפענח אוטומטית.</p>
      ${Store.isEditor() ? `
        <div class="acts">
          <button class="act act-nav" data-place="pin"><span class="lbl">נעץ על המפה
            <span class="hint">לחיצה אחת על המקום המדויק, ונשמר לכולם</span></span></button>
        </div>
        <p id="pub-msg" class="pub-msg" hidden></p>` : ''}
      ${linksBlock(it)}`;
  }

  return `
    ${it.address ? `<p class="addr">${escapeHtml(it.address)}</p>` : ''}
    <h3>הגעה</h3>
    <div class="acts">
      ${panoActs(it, ['המקום'], 'מבט 360° מהרחוב')}
      ${navActs(it, 'ניווט בתוך האפליקציה, גם דרך קיצורי הדרך')}
    </div>
    ${linksBlock(it)}
    ${Store.isEditor() ? `
      <h3>מיקום</h3>
      <div class="acts">
        <button class="act" data-place="pin"><span class="lbl">הזזת הסיכה
          <span class="hint">${escapeHtml(GEO_SOURCE[it.geoSource] || '')}</span></span></button>
        ${it.geoSource === 'manual' ? `<button class="act act-sub" data-place="unpin">
          <span class="lbl">ביטול המיקום הידני
            <span class="hint">יחזור להשערה האוטומטית בבנייה הבאה</span></span></button>` : ''}
      </div>
      <p id="pub-msg" class="pub-msg" hidden></p>` : ''}`;
}

const GEO_SOURCE = {
  manual: 'מיקום שנקבע ידנית',
  google: 'לפי גוגל מפות',
  osm: 'זוהה לפי שם ב-OpenStreetMap',
  address: 'פוענח מכתובת שבערך',
  street: 'רמת רחוב בלבד, לא מספר בית',
  nearby: 'מקורב, לפי מקום סמוך שמוזכר בערך'
};

/** What an editor can change about a trail that is already published. */
/** Who put this on the map.
 *
 *  `by` is written on publish and carried through approval, so it names the
 *  person who walked or drew the trail rather than whoever waved it through.
 *  It is shown to everybody: a shortcut on this map exists because a neighbour
 *  went and mapped it, and that should be visible without opening a commit log.
 *
 *  The trails imported from My Maps carry no name at all, so those fall back to
 *  the date alone rather than claiming an author nobody recorded. */
function creditLine(it) {
  const bits = [];
  if (it.by) bits.push(`מופה בידי ${escapeHtml(it.by)}`);
  if (it.added) bits.push(`נוסף ${new Date(it.added).toLocaleDateString('he-IL')}`);
  return bits.length ? `<p class="src">${bits.join(' · ')}</p>` : '';
}

function editorBlock(it, layer) {
  if (!Store.isEditor()) return '';
  const others = Layers.trailLayers().filter((l) => l.id !== layer.id);
  return `
    <h3>עריכה</h3>
    <div class="acts">
      <button class="act" data-pub="rename"><span class="lbl">שינוי שם והערה</span></button>
      <label class="act" style="cursor:pointer"><span class="lbl">הוספת תמונות
        <span class="hint">מוקטנות ומועלות לריפו הנתונים</span></span>
        <input type="file" accept="image/*" multiple hidden data-pub="photos"></label>
      <button class="act" data-pub="links"><span class="lbl">קישורים
        <span class="hint">${(it.links || []).length
          ? plural(it.links.length, 'קישור אחד', 'קישורים') : 'אתר, כתבה, ערך בוויקי'}</span></span></button>
      ${others.length ? `<button class="act" data-pub="move"><span class="lbl">העברה לשכבה אחרת
        <span class="hint">כרגע ב"${escapeHtml(layer.name)}"</span></span></button>` : ''}
      <button class="act danger" data-pub="remove"><span class="lbl">הסרה מהמסד
        <span class="hint">נשמר בהיסטוריה, אפשר לשחזר</span></span></button>
    </div>
    <p id="pub-msg" class="pub-msg" hidden></p>`;
}

function showDetail(it) {
  const layer = Layers.layerOf(it.id) || {};
  const chips = [];

  if (it.length) chips.push(`<span class="chip accent">${metres(it.length)}</span>`);
  else if (it.place) chips.push(`<span class="chip accent" style="--c:${it.color}">${escapeHtml(it.group || 'מקום')}</span>`);
  else chips.push('<span class="chip accent">נקודת ציון</span>');
  if (layer.kind !== 'trails' && !it.place) {
    chips.push(`<span class="chip layer" style="--c:${layer.color}">${escapeHtml(layer.name)}</span>`);
  }
  if (layer.kind === 'trails' && layer.id !== Layers.TRAILS_ID) {
    chips.push(`<span class="chip layer" style="--c:${layer.color}">${escapeHtml(layer.name)}</span>`);
  }
  if (it.approx) chips.push('<span class="chip">מיקום מקורב</span>');
  if (it.status) chips.push(`<span class="chip">${escapeHtml(it.status)}</span>`);
  if (it.grade) chips.push(`<span class="chip">רשת ${escapeHtml(it.grade)}</span>`);
  if (it.kind) chips.push(`<span class="chip">${escapeHtml(it.kind)}</span>`);
  if (it.detour && it.detour > 1.15) chips.push(`<span class="chip">מתפתל ×${it.detour}</span>`);
  if (here) {
    const d = nearestMetres(it);
    if (isFinite(d)) {
      chips.push(`<span class="chip">${d < 1000 ? Math.round(d) + ' מ׳' : (d / 1000).toFixed(1) + ' ק"מ'} ממך</span>`);
    }
  }

  // Entries are the two ends you can walk to. Trails from the dataset carry
  // them worked out properly; anything else - a queued submission, a segment
  // from the cycling plan - has only a line, so take its ends.
  //
  // An unpinned place has neither, and an entry built from nothing would send
  // Street View to the middle of the Atlantic.
  const ends = (path) => [
    { lat: path[0][0], lng: path[0][1] },
    { lat: path[path.length - 1][0], lng: path[path.length - 1][1] }
  ];
  const entries = it.entries
    || (it.path && it.path.length > 1 ? ends(it.path)
      : it.lat == null ? [] : [{ lat: it.lat, lng: it.lng, heading: 0, view: null }]);
  it = { ...it, entries };

  let body;
  if (it.place) {
    body = placeBody(it);
  } else if (layer.kind === 'network') {
    const labels = ['תחילת המקטע', 'סוף המקטע'];
    body = `
      ${it.streets && it.streets.length
        ? `<p class="note">עובר לאורך ${escapeHtml(it.streets.join(', '))}.</p>` : ''}
      <h3>הגעה</h3>
      <div class="acts">
        ${panoActs(it, labels, 'מבט 360° לאורך המקטע')}
        ${navActs(it, 'ניווט בתוך האפליקציה, לאורך תוואי המקטע')}
      </div>
      ${linksBlock(it)}
      <p class="src">${escapeHtml(layer.credit || '')}</p>`;
  } else if (it.pending) {
    body = `
      <p class="unplaced">שביל שהתקבל ועוד לא אושר. הוא לא מופיע למי שרק פותח
        את האפליקציה.</p>
      <h3>הגעה</h3>
      <div class="acts">${navActs(it, 'ניווט לפי התוואי שנשלח')}</div>
      ${linksBlock(it)}
      <h3>אישור</h3>
      <div class="acts">
        <button class="act act-nav" data-queue="approve"><span class="lbl">אשר והוסף למפה
          <span class="hint">ייכנס מיד לכל מי שפותח את האפליקציה</span></span></button>
        <button class="act danger" data-queue="reject"><span class="lbl">דחה
          <span class="hint">יוסר מהתור. נשמר בהיסטוריה</span></span></button>
      </div>
      <p id="pub-msg" class="pub-msg" hidden></p>
      <p class="src">${it.by ? `נשלח על ידי ${escapeHtml(it.by)} · ` : ''}${
        it.submitted ? new Date(it.submitted).toLocaleDateString('he-IL') : ''}</p>`;
  } else if (it.draft) {
    body = `
      <h3>הגעה</h3>
      <div class="acts">${navActs(it, 'ניווט לפי התוואי שהקלטת')}</div>
      ${linksBlock(it)}
      ${Drafts.detailExtras(it)}`;
  } else {
    const names = it.connects && it.connects.length === entries.length
      ? it.connects
      : entries.map((_, i) => (entries.length > 1 ? `כניסה ${i + 1}` : 'המקום'));
    body = `
      <h3>הגעה</h3>
      <div class="acts">
        ${panoActs(it, names, 'מבט 360° מהרחוב אל הכניסה')}
        ${navActs(it, 'ניווט בתוך האפליקציה, לפי מסלול השביל עצמו')}
      </div>
      ${linksBlock(it)}
      ${creditLine(it)}
      ${editorBlock(it, layer)}`;
  }

  // Photos can be removed only where this app owns them: the initiative's own
  // trails. A pardespedia photo belongs to the wiki article and is changed
  // there, not here.
  const canEditPhotos = Store.isEditor() && !it.place && !it.draft && layer.kind === 'trails';
  const photos = it.photos || [];
  const gallery = photos.length ? `
    <h3>תמונות (${photos.length})</h3>
    <div class="gallery${canEditPhotos ? ' editable' : ''}">
      ${photos.map((p, i) => `<span class="shot">
        <img src="${p.thumb}" data-i="${i}" alt="${escapeHtml(it.name)}" loading="lazy">
        ${canEditPhotos ? `<button class="shot-x" data-drop="${i}"
          aria-label="הסרת התמונה">&times;</button>` : ''}
      </span>`).join('')}
    </div>` : '';

  el('detail').innerHTML = `
    <h2>${escapeHtml(it.name)}</h2>
    <div class="chips">${chips.join('')}</div>
    ${it.note ? `<p class="note">${escapeHtml(it.note)}</p>` : ''}
    ${body}
    ${gallery}
    ${it.place
      ? `<a class="src" href="${escapeHtml(it.url)}" target="_blank" rel="noopener">
           הטקסט והתמונה מתוך פרדספדיה, הוויקי של המושבה ↗</a>`
      : layer.kind === 'trails'
        ? `<a class="src" href="${DATA.source}" target="_blank" rel="noopener">המפה המקורית ב-Google My Maps ↗</a>`
        : ''}`;

  el('detail').querySelectorAll('.gallery img').forEach((img) => {
    img.addEventListener('click', () => openLightbox(it, +img.dataset.i));
  });
  const go = el('go');
  if (go) go.addEventListener('click', () => startNav(it));
  if (it.draft) Drafts.wireDetail(it, el('detail'));
  wirePublished(it);

  el('detail').scrollTop = 0;
  el('list-view').hidden = true;
  el('detail-view').hidden = false;
}

/** A status line inside the detail pane, shared by every editing action. */
function detailSay() {
  const msg = el('pub-msg');
  return (text, bad) => {
    if (!msg) return;
    msg.hidden = false;
    msg.textContent = text;
    msg.className = 'pub-msg' + (bad ? ' bad' : '');
  };
}

/** Editing an item that is already in the shared dataset: a published trail,
 *  or the position of a pardespedia place. */
function wirePublished(it) {
  const say = detailSay();

  el('detail').querySelectorAll('[data-drop]').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('להסיר את התמונה הזאת מהשביל?')) return;
      btn.disabled = true;
      say('מסיר תמונה…');
      try {
        await reloadShared(await Store.removePhoto(it.id, +btn.dataset.drop, it.name));
        select(it.id, false);
      } catch (err) {
        btn.disabled = false;
        say('נכשל: ' + err.message, true);
      }
    });
  });

  el('detail').querySelectorAll('[data-queue]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const ok = btn.dataset.queue === 'approve';
      if (!ok && !confirm(`לדחות את "${it.name}"?\nהוא יוסר מהתור.`)) return;
      btn.disabled = true;
      say(ok ? 'מאשר…' : 'מסיר…');
      try {
        if (ok) {
          const { id, doc } = await Store.approve(it.id);
          await reloadShared(doc);
          await refreshQueue();
          select(id);
        } else {
          await Store.reject(it.id, it.name);
          deselect();
          await refreshQueue();
        }
      } catch (err) {
        btn.disabled = false;
        say('נכשל: ' + err.message, true);
      }
    });
  });

  el('detail').querySelectorAll('[data-place]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (btn.dataset.place === 'pin') { startPinning(it); return; }
      if (!confirm(`לבטל את המיקום הידני של "${it.name}"?`)) return;
      btn.disabled = true;
      say('מבטל…');
      try {
        await reloadPlaces(await Store.unpinPlace(it.id, it.name));
        select(it.id, false);
      } catch (err) {
        btn.disabled = false;
        say('נכשל: ' + err.message, true);
      }
    });
  });

  el('detail').querySelectorAll('[data-pub]').forEach((node) => {
    const act = node.dataset.pub;

    if (act === 'photos') {
      node.addEventListener('change', async () => {
        if (!node.files || !node.files.length) return;
        say('מעלה…');
        try {
          await reloadShared(await Store.addPhotos(it.id, [...node.files], it.name, say));
          select(it.id, false);
        } catch (err) {
          say('העלאה נכשלה: ' + err.message, true);
        }
      });
      return;
    }

    node.addEventListener('click', async () => {
      try {
        if (act === 'links') { linksForm(it); return; }
        if (act === 'move') { moveForm(it); return; }

        if (act === 'rename') {
          const name = prompt('שם השביל:', it.name);
          if (name == null) return;
          const note = prompt('הערה (אפשר להשאיר ריק):', it.note || '');
          if (note == null) return;
          node.disabled = true;
          say('שומר…');
          await reloadShared(await Store.rename(it.id, name.trim() || it.name, note.trim()));
          select(it.id, false);
        } else if (act === 'remove') {
          if (!confirm(`להסיר את "${it.name}" מהמסד המשותף?\n` +
                       'השינוי נשמר בהיסטוריה ואפשר לשחזר אותו.')) return;
          node.disabled = true;
          say('מסיר…');
          await reloadShared(await Store.remove(it.id, it.name));
          deselect();
        }
      } catch (err) {
        node.disabled = false;
        say('נכשל: ' + err.message, true);
      }
    });
  });
}

/* ---------- the small editor forms ----------
 *
 * Links, a new trail layer, moving a trail between layers. All three are a few
 * fields and a save button, so they share one sheet rather than each growing
 * its own markup in index.html.
 */

const openForm = (html) => {
  el('form-card').innerHTML = html;
  el('form-sheet').hidden = false;
};

const closeForm = () => { el('form-sheet').hidden = true; };

/** A repeating url/title pair, used both here and by draft.js when a trail is
 *  first written, so a link can be attached before the trail even exists. */
const LinkRows = {
  row(link) {
    return `<div class="link-row">
      <input class="l-url" type="url" inputmode="url" spellcheck="false"
             placeholder="https://…" value="${escapeHtml((link && link.url) || '')}">
      <input class="l-title" type="text" maxlength="60"
             placeholder="איך לקרוא לקישור" value="${escapeHtml((link && link.title) || '')}">
      <button type="button" class="link-x" aria-label="הסרת הקישור">&times;</button>
    </div>`;
  },

  html(links) {
    const rows = (links && links.length ? links : [null]).map((l) => LinkRows.row(l)).join('');
    return `<div class="link-rows">${rows}</div>
      <button type="button" class="add-row" data-act="add-link">+ עוד קישור</button>`;
  },

  read(root) {
    return [...root.querySelectorAll('.link-row')].map((row) => ({
      url: row.querySelector('.l-url').value.trim(),
      title: row.querySelector('.l-title').value.trim()
    })).filter((l) => l.url);
  },

  /** One delegated listener covers rows that do not exist yet. */
  wire(root) {
    root.addEventListener('click', (e) => {
      if (e.target.closest('[data-act="add-link"]')) {
        root.querySelector('.link-rows').insertAdjacentHTML('beforeend', LinkRows.row(null));
        root.querySelector('.link-rows').lastElementChild.querySelector('.l-url').focus();
        return;
      }
      const x = e.target.closest('.link-x');
      if (!x) return;
      const rows = root.querySelector('.link-rows');
      if (rows.children.length > 1) x.closest('.link-row').remove();
      else x.closest('.link-row').querySelectorAll('input').forEach((i) => { i.value = ''; });
    });
  }
};

function linksForm(it) {
  openForm(`
    <header class="sheet-head">
      <h2>קישורים</h2>
      <button class="sheet-x" data-act="close-form" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">קישורים שיופיעו במסך של "${escapeHtml(it.name)}": אתר, כתבה,
      ערך בפרדספדיה, אלבום תמונות. השאר שורה ריקה כדי למחוק אותה.</p>
    ${LinkRows.html(it.links)}
    <p id="form-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="save-links"><b>שמור קישורים</b></button>`);
  formTarget = it;
}

/* The palette a new layer picks from. Free colour entry on a phone is a colour
 * wheel nobody can hit precisely, and eight distinguishable colours is what a
 * map can carry anyway. */
const LAYER_COLOURS = ['#0b7285', '#c2255c', '#5f3dc4', '#e8590c',
                       '#2b8a3e', '#1864ab', '#a61e4d', '#495057'];

function layerForm(layer) {
  const now = layer || { name: '', color: LAYER_COLOURS[0], note: '', dash: false };
  openForm(`
    <header class="sheet-head">
      <h2>${layer ? 'עריכת שכבה' : 'שכבת שבילים חדשה'}</h2>
      <button class="sheet-x" data-act="close-form" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">${layer ? 'השינוי חל על כל השבילים בשכבה.'
      : 'שכבה היא קבוצה של שבילים שאפשר להדליק ולכבות יחד. אחרי שתיווצר, כל שביל שתפרסם יוכל להיכנס אליה.'}</p>
    <label class="fld"><span>שם השכבה</span>
      <input id="lay-name" type="text" maxlength="40" value="${escapeHtml(now.name)}"
             placeholder="למשל: מסלולי בוקר, שבילים לעגלה"></label>
    <label class="fld"><span>תיאור (לא חובה)</span>
      <textarea id="lay-note" rows="2" maxlength="160"
                placeholder="מה נכנס לשכבה הזאת">${escapeHtml(now.note || '')}</textarea></label>
    <div class="fld"><span>צבע</span>
      <div class="swatches" id="lay-colours">
        ${LAYER_COLOURS.map((c) => `<button type="button" class="sw${c === now.color ? ' on' : ''}"
          data-colour="${c}" style="--c:${c}" aria-label="צבע ${c}"></button>`).join('')}
      </div>
    </div>
    <label class="check"><input type="checkbox" id="lay-dash" ${now.dash ? 'checked' : ''}>
      <span>קו מקווקו, לשבילים שעוד לא קיימים בשטח</span></label>
    <p id="form-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="save-layer"><b>${layer ? 'שמור' : 'צור שכבה'}</b></button>
    ${layer ? `<button class="big-act ghost danger" data-act="drop-layer"><b>מחיקת השכבה</b>
      <span>השבילים שבתוכה יחזרו לשכבת "דרכי קיצור", ולא יימחקו</span></button>` : ''}`);
  formTarget = layer || null;
  formColour = now.color;
}

function moveForm(it) {
  const layers = Layers.trailLayers();
  const current = Layers.layerOf(it.id) || {};
  openForm(`
    <header class="sheet-head">
      <h2>העברה לשכבה</h2>
      <button class="sheet-x" data-act="close-form" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">לאיזו שכבה "${escapeHtml(it.name)}" שייך.</p>
    <div class="picks" id="layer-pick">
      ${layers.map((l) => `<label class="pick${l.id === current.id ? ' on' : ''}">
        <input type="radio" name="target" value="${l.id}" ${l.id === current.id ? 'checked' : ''}>
        <span class="lay-swatch" style="--c:${l.color}"></span>
        <span>${escapeHtml(l.name)}</span>
      </label>`).join('')}
    </div>
    <p id="form-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="save-move"><b>העבר</b></button>`);
  formTarget = it;
}

let formTarget = null;      // what the open form is about
let formColour = null;      // the swatch picked in the layer form

function formError(err) {
  const box = el('form-err');
  if (!box) return;
  box.textContent = err;
  box.hidden = false;
}

async function formAction(act, btn) {
  if (act === 'close-form') { closeForm(); return; }
  if (act === 'colour') return;

  const label = btn.querySelector('b');
  const was = label ? label.textContent : '';
  if (label) label.textContent = 'שומר…';
  btn.disabled = true;

  try {
    if (act === 'save-links') {
      const links = LinkRows.read(el('form-card'));
      await reloadShared(await Store.setLinks(formTarget.id, links, formTarget.name));
      closeForm();
      select(formTarget.id, false);

    } else if (act === 'save-layer') {
      const name = el('lay-name').value.trim();
      if (!name) throw new Error('צריך שם לשכבה.');
      const patch = {
        name,
        note: el('lay-note').value.trim(),
        color: formColour,
        dash: el('lay-dash').checked
      };
      await reloadShared(formTarget
        ? await Store.editLayer(formTarget.id, patch)
        : await Store.addLayer(patch));
      closeForm();
      Layers.render();

    } else if (act === 'drop-layer') {
      if (!confirm(`למחוק את השכבה "${formTarget.name}"?\n` +
                   'השבילים שבתוכה יעברו לשכבת "דרכי קיצור" ולא יימחקו.')) {
        btn.disabled = false;
        if (label) label.textContent = was;
        return;
      }
      await reloadShared(await Store.removeLayer(formTarget.id, formTarget.name));
      closeForm();
      Layers.render();

    } else if (act === 'save-move') {
      const picked = el('form-card').querySelector('input[name=target]:checked');
      if (!picked) throw new Error('צריך לבחור שכבה.');
      const target = picked.value === Layers.TRAILS_ID ? null : picked.value;
      await reloadShared(await Store.setLayer(formTarget.id, target, formTarget.name));
      closeForm();
      select(formTarget.id, false);
    }
  } catch (err) {
    formError(err.message);
    btn.disabled = false;
    if (label) label.textContent = was;
  }
}

/* ---------- placing a place ----------
 *
 * Pardespedia knows what a place is and what it looks like, and has never
 * known where it is. The whole tool lives in arrange.js, because correcting
 * these is bulk work: most of the derived positions are wrong, and the only
 * way to fix one is for somebody who lives here to look at the map.
 *
 * The panel gets out of the way first. A tap on the map means something else
 * entirely while the tool is open, and the map has to be the thing you see.
 */
function startPinning(it) {
  document.documentElement.style.setProperty('--panel-h', '128px');
  setTimeout(() => { if (map) map.resize(); }, 60);
  Arrange.open(it);
}

/* ---------- lightbox ---------- */

let lbPhotos = [], lbIndex = 0, lbTitle = '';

function openLightbox(item, index) {
  lbPhotos = item.photos;
  lbTitle = item.name;
  lbIndex = index;
  el('lightbox').hidden = false;
  document.addEventListener('keydown', lbKeys);
  paintLightbox();
}

function closeLightbox() {
  el('lightbox').hidden = true;
  el('lb-img').src = '';
  document.removeEventListener('keydown', lbKeys);
}

function paintLightbox() {
  const img = el('lb-img');
  const photo = lbPhotos[lbIndex];
  img.classList.add('loading');
  el('lb-spin').hidden = false;
  img.src = photo.full;                       // full resolution, not the thumb
  img.alt = lbTitle;
  img.onload = () => { img.classList.remove('loading'); el('lb-spin').hidden = true; };
  img.onerror = () => { el('lb-spin').hidden = true; img.classList.remove('loading'); };

  el('lb-cap-text').textContent = lbTitle;
  el('lb-count').textContent = lbPhotos.length > 1 ? `(${lbIndex + 1}/${lbPhotos.length})` : '';
  const many = lbPhotos.length > 1;
  document.querySelector('.lb-prev').hidden = !many;
  document.querySelector('.lb-next').hidden = !many;
}

function step(delta) {
  lbIndex = (lbIndex + delta + lbPhotos.length) % lbPhotos.length;
  paintLightbox();
}

function lbKeys(e) {
  if (e.key === 'Escape') closeLightbox();
  // In RTL the visual "next" arrow points left, so the keys are mirrored.
  else if (e.key === 'ArrowLeft') step(1);
  else if (e.key === 'ArrowRight') step(-1);
}

/* ---------- in-app navigation ----------
 *
 * 83% of these shortcuts do not exist in OpenStreetMap, and they are not in
 * Google's network either - that is the whole reason the initiative exists.
 * So no routing service can guide you along one. We have the geometry, so we
 * navigate off it directly: walk you to the nearest entrance, then count down
 * the trail itself. No key, no network calls, works with no signal.
 */

let nav = null;      // {item, watchId, target}
let facing = null;   // compass heading in degrees, when the device reports one

function bearingTo(from, to) {
  const rad = Math.PI / 180;
  const dLng = (to.lng - from.lng) * rad;
  const y = Math.sin(dLng) * Math.cos(to.lat * rad);
  const x = Math.cos(from.lat * rad) * Math.sin(to.lat * rad) -
    Math.sin(from.lat * rad) * Math.cos(to.lat * rad) * Math.cos(dLng);
  return (Math.atan2(y, x) / rad + 360) % 360;
}

/** Closest point on a trail, plus how far it is and how much trail remains
 *  in each direction. Distances are along the path, not straight lines. */
function projectOnPath(pos, path) {
  let best = { d: Infinity, i: 0, t: 0, point: null };
  for (let i = 0; i < path.length - 1; i++) {
    const a = { lat: path[i][0], lng: path[i][1] };
    const b = { lat: path[i + 1][0], lng: path[i + 1][1] };
    const my = 111320, mx = 111320 * Math.cos(pos.lat * Math.PI / 180);
    const px = (pos.lng - a.lng) * mx, py = (pos.lat - a.lat) * my;
    const bx = (b.lng - a.lng) * mx, by = (b.lat - a.lat) * my;
    const len = bx * bx + by * by;
    const t = len === 0 ? 0 : Math.max(0, Math.min(1, (px * bx + py * by) / len));
    const d = Math.hypot(px - t * bx, py - t * by);
    if (d < best.d) {
      best = { d, i, t,
        point: { lat: a.lat + (b.lat - a.lat) * t, lng: a.lng + (b.lng - a.lng) * t } };
    }
  }
  let before = 0, after = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const seg = distance({ lat: path[i][0], lng: path[i][1] },
                         { lat: path[i + 1][0], lng: path[i + 1][1] });
    if (i < best.i) before += seg;
    else if (i > best.i) after += seg;
    else { before += seg * best.t; after += seg * (1 - best.t); }
  }
  return { ...best, before, after };
}

function startNav(item) {
  if (!navigator.geolocation) { alert('הדפדפן לא תומך באיתור מיקום.'); return; }
  stopNav();
  nav = { item, watchId: null, endIdx: null };
  document.body.classList.add('nav-active');
  el('nav').hidden = false;
  el('nav-dist').textContent = '—';
  el('nav-state').textContent = 'מחפש מיקום…';
  askForCompass();

  nav.watchId = navigator.geolocation.watchPosition(
    (pos) => {
      here = { lat: pos.coords.latitude, lng: pos.coords.longitude };
      drawMe();
      // While moving, GPS course is a better "which way am I facing" than a
      // compass the user may never have granted.
      if (pos.coords.speed > 0.6 && pos.coords.heading != null) facing = pos.coords.heading;
      paintNav();
    },
    () => {
      el('nav').classList.add('stale');
      el('nav-state').textContent = 'אין גישה למיקום. צריך לאשר, ורק מעל https.';
    },
    { enableHighAccuracy: true, maximumAge: 2000, timeout: 15000 }
  );
}

function stopNav() {
  if (nav && nav.watchId != null) navigator.geolocation.clearWatch(nav.watchId);
  nav = null;
  document.body.classList.remove('nav-active');
  el('nav').hidden = true;
  el('nav').classList.remove('on-trail', 'stale');
}

function paintNav() {
  if (!nav || !here) return;
  const item = nav.item;
  const bar = el('nav');
  bar.classList.remove('stale');

  let target, label, state, onTrail = false;

  if (item.path) {
    const p = projectOnPath(here, item.path);
    if (p.d < 25) {
      onTrail = true;
      // Lock the exit the first time we find ourselves on the trail, and keep
      // it. Re-deciding on every fix makes the arrow spin around near the
      // midpoint, and sends you back the way you came.
      if (nav.endIdx == null) {
        const last = item.path.length - 1;
        if (facing != null) {
          // Best signal: pick whichever end lies ahead of where we are walking.
          const toLast = bearingTo(here, { lat: item.path[last][0], lng: item.path[last][1] });
          const diff = Math.abs(((toLast - facing + 540) % 360) - 180);
          nav.endIdx = diff < 90 ? last : 0;
        } else {
          nav.endIdx = p.after >= p.before ? last : 0;
        }
      }
      const end = item.path[nav.endIdx];
      target = { lat: end[0], lng: end[1] };
      label = fmt(nav.endIdx === item.path.length - 1 ? p.after : p.before);
      state = 'על השביל · עד היציאה';
    } else {
      nav.endIdx = null;    // off the trail again; decide afresh on re-entry
      const ends = item.entries.map((e) => ({ lat: e.lat, lng: e.lng }));
      target = ends.reduce((a, b) => distance(here, a) <= distance(here, b) ? a : b);
      label = fmt(distance(here, target));
      state = 'אל הכניסה לשביל';
    }
  } else {
    target = { lat: item.lat, lng: item.lng };
    label = fmt(distance(here, target));
    state = item.name;
  }

  bar.classList.toggle('on-trail', onTrail);
  el('nav-dist').textContent = label;
  el('nav-state').textContent = state;

  const course = bearingTo(here, target);
  // With a heading we can point where to actually walk; without one the arrow
  // is north-up, so say so rather than sending someone the wrong way.
  if (facing == null) {
    el('nav-state').textContent = state + ' · ' + compass(course) + ' (חץ לפי צפון)';
  }
  document.querySelector('.nav-arrow').style.transform =
    `rotate(${course - (facing || 0)}deg)`;
}

const fmt = (m) => (m >= 1000 ? (m / 1000).toFixed(1) + ' ק"מ' : Math.round(m) + ' מ׳');

function compass(deg) {
  const names = ['צפון', 'צפון-מזרח', 'מזרח', 'דרום-מזרח',
                 'דרום', 'דרום-מערב', 'מערב', 'צפון-מערב'];
  return names[Math.round(deg / 45) % 8];
}

function askForCompass() {
  const use = (e) => {
    const h = e.webkitCompassHeading != null ? e.webkitCompassHeading
      : (e.absolute && e.alpha != null ? 360 - e.alpha : null);
    if (h != null) { facing = h; }
  };
  const attach = () => {
    window.addEventListener('deviceorientationabsolute', use, true);
    window.addEventListener('deviceorientation', use, true);
  };
  const req = window.DeviceOrientationEvent && DeviceOrientationEvent.requestPermission;
  if (typeof req === 'function') req.call(DeviceOrientationEvent).then((s) => {
    if (s === 'granted') attach();
  }).catch(() => {});
  else attach();
}

/* ---------- geolocation ---------- */

function drawMe() {
  if (!map) return;
  if (!hereMarker) {
    const node = document.createElement('div');
    node.className = 'me';
    hereMarker = new maplibregl.Marker({ element: node });
  }
  hereMarker.setLngLat([here.lng, here.lat]).addTo(map);
}

function locate() {
  if (!navigator.geolocation) {
    alert('הדפדפן לא תומך באיתור מיקום.');
    return;
  }
  el('locate').classList.add('on');
  navigator.geolocation.getCurrentPosition((pos) => {
    here = { lat: pos.coords.latitude, lng: pos.coords.longitude };
    drawMe();
    if (map) map.easeTo({ center: [here.lng, here.lat], zoom: 17, duration: 700 });

    sortMode = 'near';
    document.querySelectorAll('.sort').forEach((b) =>
      b.classList.toggle('on', b.dataset.sort === 'near'));
    if (selectedId) showDetail(byId(selectedId)); else renderList();
  }, () => {
    el('locate').classList.remove('on');
    alert('לא הצלחתי לאתר את המיקום. צריך לאשר גישה למיקום, ובדפדפן זה עובד רק ב-https.');
  }, { enableHighAccuracy: true, timeout: 10000 });
}

/* ---------- panel drag (mobile) ---------- */

function wireGrip() {
  const grip = el('grip');
  let startY = 0, startH = 0, dragging = false;

  const begin = (y) => {
    dragging = true;
    startY = y;
    startH = el('panel').getBoundingClientRect().height;
  };
  const move = (y) => {
    if (!dragging) return;
    const h = Math.min(window.innerHeight * 0.9,
      Math.max(90, startH + (startY - y)));
    document.documentElement.style.setProperty('--panel-h', h + 'px');
  };
  const end = () => { dragging = false; if (map) map.resize(); };

  grip.addEventListener('pointerdown', (e) => { begin(e.clientY); grip.setPointerCapture(e.pointerId); });
  grip.addEventListener('pointermove', (e) => move(e.clientY));
  grip.addEventListener('pointerup', end);
  grip.addEventListener('pointercancel', end);
  grip.addEventListener('click', () => {
    const collapsed = el('panel').getBoundingClientRect().height < 140;
    document.documentElement.style.setProperty('--panel-h', collapsed ? '45vh' : '96px');
    setTimeout(() => { if (map) map.resize(); }, 260);
  });
}

/* ---------- panel width (desktop) ----------
 *
 * On a wide screen the list is a column beside the map, and 380px is a guess
 * that suits nobody exactly: reading trail notes wants it wider, looking at
 * where the paths actually run wants it gone. So the edge is draggable and the
 * tab folds it away entirely.
 *
 * Everything reads from one custom property, `--panel-w`, which is also what
 * the map, the floating buttons and the navigation bar are positioned against.
 * Setting it is the whole implementation; nothing else has to be told.
 */

const PANEL_PREF = 'dk.panel.v1';
const PANEL_MIN = 280;                 // narrower than this and names wrap badly
const PANEL_FOLD = 200;                // drag past here and it folds instead

function panelMax() {
  return Math.min(window.innerWidth * 0.6, 720);
}

function setPanelWidth(w) {
  document.documentElement.style.setProperty('--panel-w', Math.round(w) + 'px');
}

function foldPanel(off, animate) {
  document.body.classList.toggle('panel-anim', !!animate);
  document.body.classList.toggle('panel-off', off);
  el('panel-fold').setAttribute('aria-label', off ? 'הצגת הרשימה' : 'הסתרת הרשימה');
  el('panel').setAttribute('aria-hidden', off ? 'true' : 'false');
  savePanel();
  // The map only learns its new size when told, and only after the transition
  // has actually moved the edge.
  setTimeout(() => { if (map) map.resize(); }, animate ? 240 : 0);
}

function savePanel() {
  try {
    localStorage.setItem(PANEL_PREF, JSON.stringify({
      w: parseInt(document.documentElement.style.getPropertyValue('--panel-w'), 10) || 380,
      off: document.body.classList.contains('panel-off')
    }));
  } catch (err) {
    /* private mode: the panel simply opens at its default next time */
  }
}

function wirePanelWidth() {
  const grip = el('grip-v');
  const fold = el('panel-fold');

  let pref = {};
  try {
    pref = JSON.parse(localStorage.getItem(PANEL_PREF) || '{}');
  } catch (err) {
    pref = {};
  }
  if (pref.w) setPanelWidth(Math.min(Math.max(pref.w, PANEL_MIN), panelMax()));
  if (pref.off) foldPanel(true, false);

  let startX = 0, startW = 0, toRight = true, dragging = false, queued = false;
  // A drag ends with a synthetic click on the same element. Without this,
  // dragging the edge shut folds the panel and the click that follows opens it
  // straight back up.
  let dragged = false;

  // Which way the panel grows depends on the writing direction, so ask the
  // element where it is rather than assuming RTL.
  const begin = (e) => {
    if (document.body.classList.contains('panel-off')) return;
    const rect = el('panel').getBoundingClientRect();
    dragging = true;
    startX = e.clientX;
    startW = rect.width;
    dragged = false;
    toRight = rect.right >= window.innerWidth - 2;
    grip.classList.add('dragging');
    document.body.classList.remove('panel-anim');
    grip.setPointerCapture(e.pointerId);
  };

  const move = (e) => {
    if (!dragging) return;
    const delta = e.clientX - startX;
    if (Math.abs(delta) > 3) dragged = true;
    const raw = startW + (toRight ? -delta : delta);
    setPanelWidth(Math.min(Math.max(raw, PANEL_FOLD - 60), panelMax()));
    // Resizing the map on every pointer event outruns the frame; one per frame
    // is what the eye gets anyway.
    if (!queued && map) {
      queued = true;
      requestAnimationFrame(() => { queued = false; map.resize(); });
    }
  };

  const end = () => {
    if (!dragging) return;
    dragging = false;
    grip.classList.remove('dragging');
    const w = el('panel').getBoundingClientRect().width;
    if (w < PANEL_FOLD) {
      setPanelWidth(380);              // what it reopens to
      foldPanel(true, true);
      return;
    }
    setPanelWidth(Math.max(w, PANEL_MIN));
    savePanel();
    if (map) map.resize();
  };

  grip.addEventListener('pointerdown', (e) => {
    if (e.target === fold) return;     // the tab is a button, not a handle
    begin(e);
  });
  grip.addEventListener('pointermove', move);
  grip.addEventListener('pointerup', end);
  grip.addEventListener('pointercancel', end);

  // Folded, the strip is the only thing left on screen, so the whole of it
  // reopens rather than only the 26px tab.
  grip.addEventListener('click', (e) => {
    if (dragged) { dragged = false; return; }
    if (!document.body.classList.contains('panel-off') && e.target !== fold) return;
    foldPanel(!document.body.classList.contains('panel-off'), true);
  });

  grip.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 60 : 20;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      foldPanel(!document.body.classList.contains('panel-off'), true);
      return;
    }
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const rect = el('panel').getBoundingClientRect();
    const grow = (e.key === 'ArrowLeft') === (rect.right >= window.innerWidth - 2);
    setPanelWidth(Math.min(Math.max(rect.width + (grow ? step : -step),
                                    PANEL_MIN), panelMax()));
    savePanel();
    if (map) map.resize();
  });

  // A window narrowed after the fact must not leave the panel wider than the
  // window allows.
  window.addEventListener('resize', () => {
    const w = parseInt(
      document.documentElement.style.getPropertyValue('--panel-w'), 10);
    if (w && w > panelMax()) { setPanelWidth(panelMax()); savePanel(); }
  });
}

/* ---------- swipe on the lightbox ---------- */

function wireSwipe() {
  const stage = document.querySelector('.lb-stage');
  let x0 = null;
  stage.addEventListener('touchstart', (e) => { x0 = e.touches[0].clientX; }, { passive: true });
  stage.addEventListener('touchend', (e) => {
    if (x0 === null || lbPhotos.length < 2) return;
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 45) step(dx < 0 ? 1 : -1);
    x0 = null;
  });
}

/* ---------- boot ---------- */

function paintStats() {
  const s = Layers.stats();
  const bits = [];
  if (s.segments) bits.push(`${s.segments} מקטעים`, metres(s.length));
  if (s.waypoints) bits.push(`${s.waypoints} נקודות ציון`);
  if (s.places) bits.push(`${s.places} מקומות`);
  if (s.photos) bits.push(`${s.photos} תמונות`);
  if (s.waiting) bits.push(plural(s.waiting, 'שביל אחד ממתין לאישור', 'שבילים ממתינים לאישור'));
  if (!bits.length) bits.push('אין שכבה דלוקה');
  if (Store.offline) bits.push('לא מחובר, מציג עותק שמור');
  el('stats').textContent = bits.join(' · ');

  const who = Store.editor();
  el('editor-btn').textContent = who ? (who === 'עורך' ? 'עריכה דלוקה' : `עריכה · ${who}`)
    : 'מצב עריכה';
  el('editor-btn').classList.toggle('on', !!who);

  // Only an editor can do anything about an unplaced place, so the shortcut to
  // them only appears for one - and only while there are any left.
  const sortUnplaced = el('sort-unplaced');
  const worth = !!who && s.unplaced > 0;
  sortUnplaced.hidden = !worth;
  sortUnplaced.textContent = `לא ממוקמים (${s.unplaced})`;
  if (!worth && sortMode === 'unplaced') {
    sortMode = 'length';
    document.querySelectorAll('.sort').forEach((b) =>
      b.classList.toggle('on', b.dataset.sort === 'length'));
  }
}

/** Everything that has to happen after a layer is toggled or a draft changes.
 *
 *  The open item has to be checked against its *layer*, not merely against the
 *  index: the index holds every item whether or not its layer is showing, so
 *  asking only "does it still exist" leaves the detail pane open on a trail the
 *  user just hid - and skips the list repaint on the way out. */
function repaint() {
  drawWaypoints();
  paintStats();
  const layer = selectedId ? Layers.layerOf(selectedId) : null;
  if (selectedId && (!layer || !layer.on)) deselect();   // deselect repaints the list
  else if (selectedId) showDetail(byId(selectedId));
  else renderList();
}

async function boot() {
  el('basemap').title = 'רקע: ' + BASEMAPS[0].name;
  // Fires for the initial style and again after every setBasemap.
  if (map) map.on('style.load', applyOverlays);

  const { trails, network, places } = await Store.load();
  DATA = trails;
  PLACES = places;
  Layers.init(trails, network, places);
  Layers.onChange = repaint;

  // The list, the search and the buttons come up as soon as the data lands.
  // Waiting for the map style first would leave the whole panel dead on a weak
  // connection, which is exactly the connection you have out on a trail.
  renderList();
  paintStats();
  Drafts.init();
  wireControls();
  // Confirming the stored token needs the network, so it must not hold up the
  // list. The editor badge and the publish buttons appear a moment later.
  Store.resume().then(() => { repaint(); refreshQueue(); });

  if (map) {
    await new Promise((done) => (map.isStyleLoaded() ? done() : map.once('load', done)));
    const [[s1, s2], [n1, n2]] = DATA.bounds;
    map.fitBounds([[s2, s1], [n2, n1]], { padding: 24, duration: 0 });
    Layers.addToMap();
    drawWaypoints();
  }
}

/** Show the shared dataset again after a write, so the trail reappears as an
 *  ordinary trail of the initiative rather than merely vanishing from drafts.
 *
 *  A write hands back the document it just stored; use that rather than
 *  fetching, which would go through a CDN that has not caught up yet. */
async function reloadShared(doc) {
  try {
    const trails = doc || (await Store.load()).trails;
    DATA = trails;
    // resetTrails rather than refresh: a write can create a layer or move a
    // trail between layers, so which layer holds what has to be worked out
    // again, not only the geometry inside one of them.
    Layers.resetTrails(trails);
  } catch (err) {
    console.error('refresh failed', err);
  }
}

/** Pull the review queue and hand it to the layer registry.
 *
 *  Only worth a request while edit mode is on: for everybody else the queue is
 *  invisible by design, and asking for it would be a round trip that changes
 *  nothing on screen. */
async function refreshQueue() {
  if (!Store.isEditor()) { Layers.setPending([]); return; }
  try {
    const doc = await Store.queue();
    Layers.setPending(doc.items || []);
  } catch (err) {
    console.error('queue unavailable', err);
  }
}

/** Same, for the pardespedia layer after a pin is dropped or cleared. */
function reloadPlaces(doc) {
  try {
    PLACES = doc || PLACES;
    Layers.resetPlaces(PLACES);
  } catch (err) {
    console.error('places refresh failed', err);
  }
}

/* ---------- edit mode ----------
 *
 * Two different things, and they are worth keeping apart.
 *
 * Adding to the map is open to everybody: record a walk, send it in, done. No
 * account and no password, because a resident who knows a shortcut is exactly
 * who this app wants to hear from.
 *
 * Deciding what the map *says* takes the editor's password, checked by the
 * worker. That is the switch below. Without it, publishing a trail, approving
 * one out of the queue and removing one were all a tap away for anyone who
 * opened the app.
 */

function editorSheet() {
  const head = `
    <header class="sheet-head">
      <h2>מצב עריכה</h2>
      <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
    </header>`;

  if (!Store.WORKER) {
    el('editor-card').innerHTML = `${head}
      <p class="sheet-lead">שרת הכתיבה עוד לא נפרס, ולכן האפליקציה במצב קריאה בלבד.
        כל השאר עובד: המפה, הרשימה, החיפוש, הניווט, וגם הקלטת שביל חדש, שנשמר
        במכשיר שלך עד שאפשר יהיה לשלוח אותו.</p>
      <p class="sheet-credit">ההוראות ב-<code>worker/README.md</code> בריפו.</p>`;
    el('editor-sheet').hidden = false;
    return;
  }

  if (Store.writable() === false) {
    el('editor-card').innerHTML = `${head}
      <p class="sheet-lead">העריכה לא זמינה כרגע. או שאין חיבור לרשת, או שהכתיבה
        הושהתה זמנית. הרשימה, המפה, התמונות והניווט עובדים כרגיל, ושביל שתקליט
        נשמר במכשיר שלך וממתין.</p>`;
    el('editor-sheet').hidden = false;
    return;
  }

  el('editor-card').innerHTML = Store.isEditor() ? `${head}
    <p class="sheet-lead">מצב עריכה דלוק. שביל שתפרסם נכנס למפה מיד וכל מי שיפתח
      את האפליקציה יראה אותו.</p>
    <label class="fld"><span>איך לקרוא לך (לא חובה)</span>
      <input id="ed-name" type="text" maxlength="40" value="${escapeHtml(Store.named())}"
             placeholder="השם שיירשם ליד השינויים שלך"></label>
    <button class="big-act primary" data-act="save-name"><b>שמור שם</b></button>
    <button class="big-act" data-act="out"><b>כבה מצב עריכה</b>
      <span>הכפתורים ייעלמו מהמסך והסיסמה תישכח במכשיר. הטיוטות שלך נשארות.</span></button>` : `${head}
    <p class="sheet-lead">כדי להוסיף שביל למפה לא צריך שום דבר מכאן: מקליטים אותו
      במסך הטיוטות ושולחים ליוזמה. מצב עריכה הוא משהו אחר, והוא מיועד למי שמאשר
      מה נכנס למפה.</p>
    <label class="fld"><span>סיסמת עריכה</span>
      <input id="ed-key" type="password" autocomplete="current-password"
             placeholder="הסיסמה שנשמרת בשרת"></label>
    <label class="fld"><span>איך לקרוא לך (לא חובה)</span>
      <input id="ed-name" type="text" maxlength="40" value="${escapeHtml(Store.named())}"
             placeholder="השם שיירשם ליד השינויים שלך"></label>
    <p id="ed-msg" class="pub-msg" hidden></p>
    <p class="sheet-credit">כל שינוי נשמר בהיסטוריה, אז אפשר לשחזר כל דבר.</p>
    <button class="big-act primary" data-act="in"><b>הדלק מצב עריכה</b></button>`;

  el('editor-sheet').hidden = false;
}

async function editorAction(act) {
  const name = el('ed-name') ? el('ed-name').value : null;
  if (act === 'close') { el('editor-sheet').hidden = true; return; }
  if (act === 'in') {
    const msg = el('ed-msg');
    const key = el('ed-key') ? el('ed-key').value : '';
    const say = (text) => { if (msg) { msg.textContent = text; msg.hidden = !text; } };
    if (!key.trim()) { say('צריך סיסמה.'); return; }
    say('בודק…');
    // Refused covers both a wrong password and a worker that did not answer,
    // because from here the two look the same and neither lets you edit.
    if (!(await Store.enable(name, key))) {
      say('הסיסמה לא התקבלה. בדוק אותה, ואת החיבור לרשת.');
      return;
    }
    el('editor-sheet').hidden = true;
    repaint();
    refreshQueue();
    return;
  }
  if (act === 'save-name') { await Store.enable(name); editorSheet(); repaint(); return; }
  if (act === 'out') {
    Store.disable();
    Arrange.close(true);
    Layers.setPending([]);
    editorSheet();
    repaint();
  }
}

function wireControls() {
  el('editor-btn').addEventListener('click', editorSheet);
  el('editor-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'editor-sheet') { el('editor-sheet').hidden = true; return; }
    const btn = e.target.closest('[data-act]');
    if (btn) editorAction(btn.dataset.act);
  });

  el('layers').addEventListener('click', Layers.openSheet);
  el('layer-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'layer-sheet' || e.target.closest('[data-act="close"]')) {
      Layers.closeSheet();
      return;
    }
    if (e.target.closest('[data-newlayer]')) { layerForm(null); return; }
    if (e.target.closest('[data-arrange]')) {
      Layers.closeSheet();
      startPinning(null);
      return;
    }
    const edit = e.target.closest('[data-edit]');
    if (edit) layerForm(Layers.byId(edit.dataset.edit));
  });

  el('form-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'form-sheet') { closeForm(); return; }
    const swatch = e.target.closest('[data-colour]');
    if (swatch) {
      formColour = swatch.dataset.colour;
      el('lay-colours').querySelectorAll('.sw').forEach((s) =>
        s.classList.toggle('on', s === swatch));
      return;
    }
    const pick = e.target.closest('.pick');
    if (pick) {
      el('layer-pick').querySelectorAll('.pick').forEach((p) =>
        p.classList.toggle('on', p === pick));
      return;
    }
    const btn = e.target.closest('[data-act]');
    if (btn) formAction(btn.dataset.act, btn);
  });
  LinkRows.wire(el('form-sheet'));

  Arrange.wire();

  el('search').addEventListener('input', renderList);
  el('back').addEventListener('click', deselect);
  el('nav-stop').addEventListener('click', stopNav);
  el('locate').addEventListener('click', locate);
  el('basemap').addEventListener('click', () => {
    setBasemap((baseIndex + 1) % BASEMAPS.length);
  });

  if (map) {
    el('tilt').addEventListener('click', () => {
      const flat = map.getPitch() < 10;
      map.easeTo({ pitch: flat ? TILTED : 0, duration: 700 });
      el('tilt').classList.toggle('on', flat);
    });
    map.on('pitchend', () => el('tilt').classList.toggle('on', map.getPitch() >= 10));
  } else {
    el('tilt').hidden = el('basemap').hidden = true;
  }

  document.querySelectorAll('.sort').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.sort === 'near' && !here) { locate(); return; }
      sortMode = btn.dataset.sort;
      document.querySelectorAll('.sort').forEach((b) => b.classList.toggle('on', b === btn));
      renderList();
    });
  });

  document.querySelector('.lb-close').addEventListener('click', closeLightbox);
  document.querySelector('.lb-prev').addEventListener('click', () => step(-1));
  document.querySelector('.lb-next').addEventListener('click', () => step(1));
  el('lightbox').addEventListener('click', (e) => {
    if (e.target.id === 'lightbox' || e.target.classList.contains('lb-stage')) closeLightbox();
  });

  // A click on a trail also reaches the map, so only clear the selection when
  // the tap actually landed on empty ground - and never while drafting, where
  // a tap on the map is how you place a point.
  if (map) map.on('click', (e) => {
    // While drafting or arranging, a tap on the map means something other than
    // "clear the selection".
    if (!selectedId || Drafts.isDrafting() || Arrange.isOn()) return;
    const hits = Layers.list
      .map((l) => `hit-${l.id}`)
      .filter((id) => map.getLayer(id));
    if (!map.queryRenderedFeatures(e.point, { layers: hits }).length) deselect();
  });

  wireGrip();
  wirePanelWidth();
  wireSwipe();
}

if (!hasGL) {
  el('map').innerHTML =
    '<p class="no-gl">הדפדפן הזה לא תומך בתצוגת המפה (WebGL).<br>' +
    'הרשימה, התמונות והסטריט ויו עובדים כרגיל.</p>';
}

boot().catch((err) => {
  el('stats').textContent = 'שגיאה בטעינת הנתונים.';
  console.error(err);
});
