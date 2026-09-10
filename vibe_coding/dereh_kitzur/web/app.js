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
  { id: 'streets', name: 'רחובות', style: 'https://tiles.openfreemap.org/styles/liberty' },
  {
    id: 'sat',
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

/* ---------- the view, in the address bar ----------
 *
 * The layer switches have lived in the URL since the layer sheet existed, so
 * that "look at the shortcuts together with the festival" is a link. Everything
 * else about what is on screen did not, which made that half a promise: the
 * person opening the link got your layers over their own position, their own
 * zoom, their own flat-or-tilted, their own street-or-satellite.
 *
 *     ?map=32.47410,34.96980,15.2,0,58&bg=sat&sel=p52&layers=trails,trips
 *
 * `map` is where the camera is: latitude, longitude, zoom, bearing, pitch. The
 * 3D button is not a separate flag because it never was one - it sets the
 * pitch, and the pitch is in there. `bg` names the base style and is left out
 * for the default one. `sel` is the open item, which dims every other line on
 * the map and opens the detail pane, so it is part of the picture too.
 *
 * Written with replaceState after the map settles, never during a drag: the
 * back button belongs to the map, and Safari throttles history writes. */

const VIEW_KEY = 'map';
const BG_KEY = 'bg';
const SEL_KEY = 'sel';

/** The camera the URL asks for, or null when it says nothing about it. */
function urlView() {
  const raw = new URLSearchParams(location.search).get(VIEW_KEY);
  if (!raw) return null;
  const n = raw.split(',').map(Number);
  if (n.length < 3 || n.slice(0, 3).some((v) => !isFinite(v))) return null;
  const [lat, lng, zoom, bearing, pitch] = n;
  if (Math.abs(lat) > 90 || Math.abs(lng) > 180) return null;
  return {
    center: [lng, lat],
    zoom: Math.min(Math.max(zoom, 0), 22),
    bearing: isFinite(bearing) ? bearing : 0,
    pitch: Math.min(Math.max(isFinite(pitch) ? pitch : 0, 0), 80)
  };
}

/** The base style the URL asks for, as an index, or null. */
function urlBasemap() {
  const raw = new URLSearchParams(location.search).get(BG_KEY);
  if (!raw) return null;
  const i = BASEMAPS.findIndex((b) => b.id === raw);
  return i < 0 ? null : i;
}

let syncTimer = null;

/** Put the camera, the base style and the open item back in the address bar.
 *
 *  Layers.syncUrl owns the `layers` parameter and this owns the other three;
 *  both read whatever is currently there and rewrite the whole query, so the
 *  two never erase each other. */
function syncView() {
  // The flight moves the camera every frame, and every one of those fires a
  // moveend. Writing the address bar sixty times a second is both pointless -
  // the camera is not a place you would link to mid-flight - and enough
  // history writes for Safari to start refusing them.
  if (!map || Explore.isOn()) return;
  const params = new URLSearchParams(location.search);
  const c = map.getCenter();
  const round = (v, d) => +v.toFixed(d);
  params.set(VIEW_KEY, [round(c.lat, 5), round(c.lng, 5), round(map.getZoom(), 2),
    Math.round(map.getBearing()), Math.round(map.getPitch())].join(','));

  if (baseIndex > 0) params.set(BG_KEY, BASEMAPS[baseIndex].id);
  else params.delete(BG_KEY);

  if (selectedId) params.set(SEL_KEY, selectedId);
  else params.delete(SEL_KEY);

  const query = params.toString().replace(/%2C/g, ',');
  try {
    history.replaceState(null, '', `${location.pathname}?${query}${location.hash}`);
  } catch (err) {
    /* opened straight off the filesystem; everything still works */
  }
}

/** Coalesce a drag, a zoom and a rotate that all end at once into one write. */
function scheduleSync() {
  clearTimeout(syncTimer);
  syncTimer = setTimeout(syncView, 250);
}

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
  // Opens tilted. The moshava sits on the western slope of the Carmel foothills
  // and a shortcut is very often a way through a dip that a flat map draws as
  // an ordinary gap between two streets. The relief is the reason to have a map
  // here at all, so it is what you see on arrival rather than a mode to find.
  pitch: TILTED,
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
let selMarker = null;     // the ring around the selected point, see markSelection

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
  //
  // `diff: false` is what makes that sentence true rather than lucky. Left to
  // itself, setStyle first tries to *patch* the running style into the new
  // one, and a patch fires `styledata` and never `style.load` - so the trails,
  // the terrain and the navigation line are diffed away and nothing puts them
  // back. Whether the patch is attempted at all depends on how far the two
  // styles are apart, which is why an ordinary background swap looked fine for
  // months and leaving the flight, which has touched the layers on the way in,
  // did not. A full reload every time is a few tiles more work and one
  // behaviour instead of two.
  map.setStyle(BASEMAPS[i].style, { diff: false });
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
    // Changing the basemap throws away every source on the style, the
    // navigation line included. Without this, switching to satellite mid-walk
    // silently loses the one thing telling you where you are heading.
    if (nav && here) paintNav();
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
    // Centre anchoring, because the element is a point with no size of its own
    // and the dot is drawn around it. See .pin in app.css.
    wpMarkers.push(
      new maplibregl.Marker({ element: node })
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
/** A ring around the selected point, drawn over every layer.
 *
 *  An HTML marker rather than another circle layer, for two reasons: it sits
 *  above everything without having to be reinserted whenever a layer is added
 *  or the basemap changes, and the arrival pulse is then three lines of CSS
 *  instead of an animation loop driven from JavaScript.
 *
 *  Only points get one. A trail or a trip is a line, and framing the line is
 *  already unambiguous - a ring on top of it would say less, not more. */
function markSelection(item) {
  if (selMarker) { selMarker.remove(); selMarker = null; }
  if (!map || !item || item.path || item.lat == null || item.lng == null) return;
  const node = document.createElement('div');
  node.className = 'sel-ring';
  selMarker = new maplibregl.Marker({ element: node })
    .setLngLat([item.lng, item.lat])
    .addTo(map);
}

function select(id, fit = true) {
  selectedId = id;
  const item = byId(id);
  if (!item) return;

  Layers.highlight(id);
  markSelection(item);
  if (fit && map) {
    if (item.path) {
      const b = new maplibregl.LngLatBounds();
      item.path.forEach(([lat, lng]) => b.extend([lng, lat]));
      map.fitBounds(b, { padding: 60, maxZoom: 18, duration: 700 });
    } else if (item.lat != null) {
      // Never zoom *out* to arrive. Somebody who has already zoomed to a
      // street knows where they are, and dropping them back to 18 undoes that
      // for no reason. The floor is what matters: close enough that the ring
      // and the name are both readable.
      map.easeTo({ center: [item.lng, item.lat],
                   zoom: Math.max(map.getZoom(), 18), duration: 700 });
    }
    // A place with no pin yet has nowhere to fly to. The detail pane opens
    // anyway, which is where the editor drops one.
  }
  showDetail(item);
  // A `fit` flies the camera, and `moveend` will sync once it lands. Selecting
  // without moving - from the list, or a link that already framed the shot -
  // has no move to wait for.
  if (!fit) scheduleSync();
}

function deselect() {
  selectedId = null;
  Layers.highlight(null);
  markSelection(null);
  el('detail-view').hidden = true;
  el('list-view').hidden = false;
  renderList();
  scheduleSync();
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
      // The craft and the wiki's own categories are what somebody actually
      // types: "קרמיקה", "ברים ופאבים". Neither is drawn in the list, so
      // without this they were the one thing on a place you could not search.
      [it.name, it.note, it.group, it.address, it.craft, (it.cats || []).join(' '),
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

  /* Your own unfinished business floats above every sort (10/9/2026).
   *
   * A shortcut is short - that is what makes it a shortcut - so a trail
   * somebody had just recorded landed near the bottom of seventy-odd rows under
   * the default "longest first", and under "with photos" it was filtered out
   * altogether for having none yet. Somebody drew two and reported that they
   * had "gone quiet", which is exactly right: the app had just invited them to
   * add a trail and then buried what they added.
   *
   * These rows are not content to browse. They are a to-do list of two or three
   * things that are yours and not finished, and they belong where you left
   * them. Ordinary trails keep the order the sort asked for. */
  const unfinished = (it) => it.draft || it.pending;
  return [...all.filter(unfinished), ...all.filter((it) => !unfinished(it))];
}

function subtitle(it) {
  const bits = [];
  if (it.length) bits.push(metres(it.length));
  else if (it.place) bits.push(it.group || 'מקום');
  else bits.push('נקודת ציון');
  // Said on the row itself, because the difference between "saved" and "sent"
  // is the whole of what somebody has to know after recording a trail, and the
  // only place it was said before was inside the trail's own page.
  if (it.draft) bits.push('שמור אצלך, עוד לא נשלח');
  else if (it.pending) bits.push(it.mine ? 'ממתין לאישור' : 'בתור לאישור');
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
      ? `<img class="thumb" src="${it.photos[0].thumb}" alt="" loading="lazy"
          referrerpolicy="no-referrer">`
      : `<div class="thumb empty">${glyph}</div>`;
    // A trail carrying no colour of its own is drawn in its layer's, on the
    // map and so in the list beside it too.
    const colour = it.color || (Layers.layerOf(it.id) || {}).color || '#8d6e63';
    return `<li class="row${it.id === selectedId ? ' on' : ''}" data-id="${it.id}"
              style="color:${colour}">
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

/** Whatever the item links out to: the write-up behind a place, plus any link
 *  an editor attached. Rendered for everybody, not only editors - a link
 *  nobody can follow is not a link.
 *
 *  What the first link is called comes from the layer, because a place here is
 *  always somebody else's write-up and the two sources call theirs different
 *  things: an article in the wiki, a page on the festival's site. */
function linksBlock(it) {
  const layer = Layers.layerOf(it.id) || {};
  const links = [];
  if (it.place && it.url) {
    links.push({ url: it.url, title: layer.linkTitle || 'הערך המלא', lead: true });
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

/** A place somebody else wrote up. They wrote the words and took the photo;
 *  this pane adds the two things a map can offer on top - where it is, and how
 *  to walk there.
 *
 *  Only a layer marked `pinnable` gets the positioning controls. The wiki holds
 *  no coordinates, so this app is where a pardespedia place's position is
 *  decided; the festival placed its own pins on its own map, and offering to
 *  drag those would be offering to be wrong about somebody else's data. */
/** Which document owns what somebody attaches to this item.
 *
 *  'trails'  the initiative's own shortcuts, whose photos live inside
 *            trails.json alongside the geometry, as they always have.
 *  'places'  nothing new is written here any more; pardespedia videos added
 *            before 7/9/2026 sit in places.json and are removed from it.
 *  'media'   everything else - the cycling plan, the conservation lists, the
 *            planning schemes, the blocks, the land uses, the festival, Houten,
 *            Curitiba. Those files are rebuilt by scripts, so the attachment
 *            goes in the side-car instead of into the file it hangs off. */
const IN_TRAILS_DOC = ['trails', 'waypoints', 'trips'];

function mediaHome(it, layer) {
  // The three kinds are the three arrays inside trails.json - the shortcuts,
  // the initiative's own waypoints and the walks - which is exactly the set
  // `Store`'s `find` reaches. The drafts layer is of kind `trails` too and is
  // not in that file at all, hence the second test.
  if (IN_TRAILS_DOC.includes(layer.kind) && layer.id !== 'drafts'
      && !it.draft && !it.pending) return 'trails';
  return 'media';
}

/** Attaching a picture, a video, a link or a line of explanation to anything on
 *  the map.
 *
 *  Until 7/9/2026 this existed twice and covered two layers: the whole block
 *  inside `editorBlock` for a trail, and a video-only version for a pardespedia
 *  place. Every other layer offered nothing, because there was nowhere to put
 *  the answer - the file a photo would have gone into is rebuilt from its
 *  source every time the script runs. With the side-car there is somewhere, and
 *  the reason to special-case a layer is gone, so this is one block shown on
 *  everything.
 *
 *  A draft is the exception, and not by omission: it lives in this browser's
 *  IndexedDB and has its own photo handling in draft.js, which works offline.
 *  A trail waiting in the queue is the other: it is not on the map yet. */
function mediaActs(it, layer) {
  if (!Store.isEditor() || it.draft || it.pending) return '';
  const home = mediaHome(it, layer);
  // On a trail this is part of the wider editing block, which carries its own
  // heading, its own status line and the rename, colour, move and delete
  // actions this one has no business offering.
  if (home === 'trails') return '';
  // The counts name what this block can change, which is what the side-car
  // holds - not what the item shows. An item whose source shipped four links
  // and to which nobody has attached one has nothing here yet.
  const links = (it.linksExtra || []).length;
  const noted = !!it.noteExtra;
  return `
    <h3>הוספה</h3>
    <div class="acts">
      <label class="act" style="cursor:pointer"><span class="lbl">הוספת תמונות
        <span class="hint">מוקטנות ומועלות לריפו הנתונים</span></span>
        <input type="file" accept="image/*" multiple hidden data-pub="photos"></label>
      <button class="act" data-pub="video"><span class="lbl">הוספת סרטון יוטיוב
        <span class="hint">יוצג יחד עם התמונות, ונפתח בגדול בדפדוף</span></span></button>
      <button class="act" data-pub="note"><span class="lbl">${noted ? 'שינוי ההערה' : 'הערה'}
        <span class="hint">${noted ? 'השאר ריק כדי לחזור למה שהמקור אומר'
          : 'שורה משלך על המקום הזה, במקום מה שהמקור אומר'}</span></span></button>
      <button class="act" data-pub="links"><span class="lbl">קישורים
        <span class="hint">${links ? plural(links, 'קישור אחד', 'קישורים')
          : 'אתר, כתבה, ערך בוויקי'}</span></span></button>
    </div>
    <p class="src">נשמר בנפרד מהשכבה עצמה, כדי שבנייה מחדש שלה לא תמחק אותו.</p>`;
}

function placeBody(it) {
  const layer = Layers.layerOf(it.id) || {};

  if (it.unplaced) {
    return `
      <p class="unplaced">המקום הזה עוד לא מוקם על המפה. בפרדספדיה אין קואורדינטות,
        ולערך הזה לא נמצאה כתובת שאפשר לפענח אוטומטית.</p>
      ${Store.isEditor() && layer.pinnable ? `
        <div class="acts">
          <button class="act act-nav" data-place="pin"><span class="lbl">נעץ על המפה
            <span class="hint">לחיצה אחת על המקום המדויק, ונשמר לכולם</span></span></button>
        </div>` : ''}
      ${linksBlock(it)}
      ${mediaActs(it, layer)}`;
  }

  return `
    ${it.address ? `<p class="addr">${escapeHtml(it.address)}</p>` : ''}
    ${contactBlock(it)}
    <h3>הגעה</h3>
    <div class="acts">
      ${panoActs(it, ['המקום'], 'מבט 360° מהרחוב')}
      ${navActs(it, 'ניווט בתוך האפליקציה, גם דרך קיצורי הדרך')}
    </div>
    ${linksBlock(it)}
    ${Store.isEditor() && layer.pinnable ? `
      <h3>מיקום</h3>
      <div class="acts">
        <button class="act" data-place="pin"><span class="lbl">הזזת הסיכה
          <span class="hint">${escapeHtml(GEO_SOURCE[it.geoSource] || '')}</span></span></button>
        ${it.geoSource === 'manual' ? `<button class="act act-sub" data-place="unpin">
          <span class="lbl">ביטול המיקום הידני
            <span class="hint">יחזור להשערה האוטומטית בבנייה הבאה</span></span></button>` : ''}
      </div>` : ''}
    ${mediaActs(it, layer)}`;
}

/** The practical half of a festival entry: whom to ring, and whether you can
 *  get in. The pardespedia places carry none of this, so the block simply does
 *  not appear for them. */
function contactBlock(it) {
  const rows = [];
  if (it.phone) {
    rows.push(`<a class="act act-link" href="tel:${escapeHtml(it.phone.replace(/[^\d+]/g, ''))}">
      <span class="lbl">${escapeHtml(it.phone)}<span class="hint">חיוג</span></span></a>`);
  }
  if (it.access) {
    rows.push(`<span class="act act-sub"><span class="lbl">${escapeHtml(it.access)}
      <span class="hint">נגישות</span></span></span>`);
  }
  if (it.saturday) {
    rows.push(`<span class="act act-sub"><span class="lbl">${escapeHtml(it.saturday)}
      <span class="hint">פתיחה בשבת</span></span></span>`);
  }
  return rows.length ? `<h3>פרטים</h3><div class="acts">${rows.join('')}</div>` : '';
}

const GEO_SOURCE = {
  manual: 'מיקום שנקבע ידנית',
  google: 'לפי גוגל מפות',
  osm: 'זוהה לפי שם ב-OpenStreetMap',
  address: 'פוענח מכתובת שבערך',
  street: 'רמת רחוב בלבד, לא מספר בית',
  nearby: 'מקורב, לפי מקום סמוך שמוזכר בערך',
  // The two the conservation appendix produces. It gives a block and a parcel
  // rather than a point, so even the exact ones are the middle of a plot; and
  // a fifth of its parcels no longer exist in today's cadastre.
  parcel: 'מרכז החלקה שבנספח, לא המבנה עצמו',
  neighbour: 'החלקה שבנספח כבר לא קיימת, לפי חלקה סמוכה במספור',
  plan: 'מרכז שטח התכנית, לפי הקו הכחול',
  block: 'מרכז הגוש בקדסטר',
  pardespedia: 'לפי המיקום של אותו מקום בפרדספדיה',
  shimur: 'לפי אותו אתר בנספח השימור',
  festival: 'הסיכה שהפסטיבל עצמו הניח'
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
      <button class="act" data-pub="video"><span class="lbl">הוספת סרטון יוטיוב
        <span class="hint">יוצג יחד עם התמונות, ונפתח בגדול בדפדוף</span></span></button>
      <button class="act" data-pub="colour"><span class="lbl">צבע השביל
        <span class="hint"><span class="dot" style="--c:${it.color || layer.color}"></span>
          איך הקו נראה על המפה</span></span></button>
      <button class="act" data-pub="links"><span class="lbl">קישורים
        <span class="hint">${(it.links || []).length
          ? plural(it.links.length, 'קישור אחד', 'קישורים') : 'אתר, כתבה, ערך בוויקי'}</span></span></button>
      ${others.length ? `<button class="act" data-pub="move"><span class="lbl">העברה לשכבה אחרת
        <span class="hint">כרגע ב"${escapeHtml(layer.name)}"</span></span></button>` : ''}
      <button class="act danger" data-pub="remove"><span class="lbl">הסרה מהמסד
        <span class="hint">נשמר בהיסטוריה, אפשר לשחזר</span></span></button>
    </div>`;
}

/** The shortcuts a trip threads together, in the order it walks them.
 *
 *  This is the half of a trip that a flat copy of its coordinates could never
 *  show, and the reason it is stored as a recipe: the walk is an argument for
 *  those particular shortcuts, and each one is a tap away.
 *
 *  A shortcut somebody has since deleted leaves a hole. Saying so is the whole
 *  point of keeping the reference - the alternative is a trip that quietly
 *  jumps a few hundred metres and looks fine. */
/** The caveat on a trip nobody has walked yet.
 *
 *  `build_trips.py` chains shortcuts that are on the map and routes between
 *  them over OpenStreetMap, so the line is plausible everywhere and verified
 *  nowhere: a gate that is locked, a fence that went up last spring and a
 *  stretch that is knee-deep in mud after rain all look identical to it.
 *
 *  A flag on the trip and not a sentence pasted into `note`, so that clearing
 *  it once somebody has actually walked the route is deleting a field rather
 *  than editing prose - which is the difference between a promise that gets
 *  kept and one that does not. */
function unwalkedNote(it) {
  if (!it.trip || !it.unwalked) return '';
  return `<p class="unplaced">המסלול הזה נבנה אוטומטית מדרכי קיצור שכבר על המפה,
    ועדיין לא הלכו בו מקצה לקצה. הקטעים עצמם ממופים, אבל אף אחד עוד לא בדק
    שהמעבר ביניהם פתוח בפועל. הלכתם בו? ספרו ליוזמה ונוריד את השורה הזאת.</p>`;
}

function tripParts(it) {
  // The other direction, on a shortcut's own page: which walks come through
  // here. Deleting a shortcut stops being a local act once a trip is built on
  // it, and this is where that becomes visible before somebody does it.
  if (!it.trip && !it.place && !it.draft) {
    const walks = Layers.tripsUsing(it.id);
    if (!walks.length) return '';
    return `
      <h3>טיולים שעוברים כאן</h3>
      <ol class="trip-chain">
        ${walks.map((t) => `<li><button class="chain-link" data-goto="${escapeHtml(t.id)}">
          ${escapeHtml(t.name)}<span class="rev"> · ${
            t.length >= 1000 ? (t.length / 1000).toFixed(1) + ' ק"מ' : t.length + ' מ׳'
          }</span></button></li>`).join('')}
      </ol>`;
  }

  if (!it.trip) return '';
  const uses = it.uses || [];
  const missing = (it.missing || []).length;
  if (!uses.length && !missing) return '';
  return `
    <h3>עובר דרך</h3>
    <ol class="trip-chain">
      ${uses.map((u) => `<li><button class="chain-link" data-goto="${escapeHtml(u.id)}">
        ${escapeHtml(u.name)}${u.reversed ? '<span class="rev"> (בכיוון ההפוך)</span>' : ''}
      </button></li>`).join('')}
    </ol>
    ${missing ? `<p class="unplaced">${missing === 1
      ? 'שביל אחד שהטיול עבר בו כבר לא קיים במפה, והמסלול כאן קטוע.'
      : `${missing} שבילים שהטיול עבר בהם כבר לא קיימים במפה, והמסלול כאן קטוע.`}
      </p>` : ''}`;
}

function showDetail(it) {
  const layer = Layers.layerOf(it.id) || {};
  const chips = [];

  if (it.length) chips.push(`<span class="chip accent">${metres(it.length)}</span>`);
  else if (it.place) chips.push(`<span class="chip accent" style="--c:${it.color}">${escapeHtml(it.group || 'מקום')}</span>`);
  else chips.push('<span class="chip accent">נקודת ציון</span>');
  // "אמנות | ציור" - the festival's own two-level labelling of what somebody
  // makes, which is the first thing a visitor picking a studio wants to know.
  if (it.craft) chips.push(`<span class="chip">${escapeHtml(it.craft)}</span>`);
  if (layer.kind !== 'trails' && !it.place) {
    chips.push(`<span class="chip layer" style="--c:${layer.color}">${escapeHtml(layer.name)}</span>`);
  }
  if (layer.kind === 'trails' && layer.id !== Layers.TRAILS_ID) {
    chips.push(`<span class="chip layer" style="--c:${layer.color}">${escapeHtml(layer.name)}</span>`);
  }
  if (it.trip) {
    if (it.group) chips.push(`<span class="chip accent" style="--c:${it.color}">${escapeHtml(it.group)}</span>`);
    chips.push(`<span class="chip">${it.loop ? 'מעגלי' : 'מקצה לקצה'}</span>`);
    if (it.minutes) chips.push(`<span class="chip">כ-${it.minutes} דק׳ הליכה</span>`);
    if (it.uses && it.uses.length) {
      chips.push(`<span class="chip">${it.uses.length} ${
        it.uses.length === 1 ? 'דרך קיצור' : 'דרכי קיצור'}</span>`);
    }
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
    // A network layer is lines, except where it is not: Curitiba carries half a
    // dozen pins for the places its bike network was built to reach, and they
    // arrive here with a position and no path at all. Calling that "start of
    // the segment" would be the label lying about what was clicked.
    const spot = !it.path || it.path.length < 2;
    const labels = spot ? ['המקום'] : ['תחילת המקטע', 'סוף המקטע'];
    body = `
      ${it.streets && it.streets.length
        ? `<p class="note">עובר לאורך ${escapeHtml(it.streets.join(', '))}.</p>` : ''}
      <h3>הגעה</h3>
      <div class="acts">
        ${panoActs(it, labels, spot ? 'מבט 360° מהרחוב' : 'מבט 360° לאורך המקטע')}
        ${navActs(it, spot ? 'ניווט בתוך האפליקציה, עד המקום'
          : 'ניווט בתוך האפליקציה, לאורך תוואי המקטע')}
      </div>
      ${linksBlock(it)}
      <p class="src">${escapeHtml(layer.credit || '')}</p>
      ${mediaActs(it, layer)}`;
  } else if (it.pending) {
    // The same queued trail, seen from the two ends of the same transaction.
    // An editor is deciding whether it goes on the map; the person who walked
    // it is waiting to hear, and what they can still do meanwhile is add the
    // photographs, which on a shortcut usually means a second walk.
    body = Store.isEditor() ? `
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
        it.submitted ? new Date(it.submitted).toLocaleDateString('he-IL') : ''}</p>` : `
      <p class="waiting">שלחת את השביל הזה והוא <b>ממתין לאישור</b>. בינתיים רק
        אתה רואה אותו. ברגע שמישהו מהיוזמה יאשר אותו הוא ייכנס למפה של כולם, על
        שמך.</p>
      <h3>הגעה</h3>
      <div class="acts">${navActs(it, 'ניווט לפי התוואי ששלחת')}</div>
      ${linksBlock(it)}
      <h3>להוסיף לשביל</h3>
      <div class="acts">
        <label class="act" style="cursor:pointer"><span class="lbl">הוספת תמונות
          <span class="hint">נשלחות עכשיו ומצטרפות לשביל שממתין</span></span>
          <input type="file" accept="image/*" multiple hidden data-mine="photos"></label>
        <button class="act" data-mine="video"><span class="lbl">הוספת סרטון
          <span class="hint">קישור ליוטיוב</span></span></button>
      </div>
      <p id="pub-msg" class="pub-msg" hidden></p>
      <p class="src">נשלח${it.by ? ` על שם ${escapeHtml(it.by)}` : ''}${
        it.submitted ? ` · ${new Date(it.submitted).toLocaleDateString('he-IL')}` : ''}</p>`;
  } else if (it.draft) {
    // The navigation block is handed to the drafts module rather than printed
    // above it. On a trail somebody has just walked, "navigate me there" is not
    // the question, and having it first was most of why the send button went
    // unnoticed.
    body = Drafts.detailExtras(it, `
      <h3>הגעה</h3>
      <div class="acts">${navActs(it, 'ניווט לפי התוואי שהקלטת')}</div>
      ${linksBlock(it)}`);
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

  // Photos that live on somebody else's host are asked for without a referrer.
  // Google's image hosting, which is where fifteen years of מקום שמור sit,
  // answers 429 to a request that says it comes from an origin it does not
  // know - localhost above all - and Chrome then blocks the reply for having
  // arrived as HTML where an image was expected. Nothing here needs to say
  // where it is asking from.
  const photos = it.photos || [];
  const clips = photos.filter((p) => p.yt).length;

  // What this app may remove, which is what it put there. The gallery is the
  // source's own pictures followed by whatever an editor attached, and
  // `mediaBase` is the boundary between the two - so `base` many pictures came
  // with the item and the rest are in the side-car.
  //
  // A pardespedia photo belongs to the wiki article: the next build of
  // places.json brings it back, so a delete button over it would promise
  // something it cannot keep. A Houten path's own photos are the same story.
  // The attached ones, on either, are ours to remove.
  const base = (it.mediaBase || photos).length;
  const home = mediaHome(it, layer);
  // Which document the item's *own* photos are in, when this app may touch them
  // at all. A ternary and not `home === 'trails' || …`, which yields `true` and
  // then never matches the string it is compared against - that read as a trail
  // whose eight photos had lost their delete buttons.
  //
  // Pardespedia videos added before the side-car existed still sit in
  // places.json, and removing one has to go there. They are the only entries in
  // any base list this app wrote that are not a trail's.
  const ownsBase = home === 'trails' ? 'trails'
    : layer.id === Layers.PLACES_ID ? 'places' : false;
  const editable = Store.isEditor() && !it.draft;
  const canDrop = (p, i) => editable
    && (i >= base || ownsBase === 'trails' || (ownsBase === 'places' && !!p.yt));

  const heading = !clips ? 'תמונות'
    : clips === photos.length ? (clips === 1 ? 'סרטון' : 'סרטונים')
      : 'תמונות וסרטונים';

  const gallery = photos.length ? `
    <h3>${heading} (${photos.length})</h3>
    <div class="gallery${editable ? ' editable' : ''}">
      ${photos.map((p, i) => `<span class="shot${p.yt ? ' video' : ''}">
        <img src="${p.thumb}" data-i="${i}"
          title="${escapeHtml(p.cap || (p.yt ? 'סרטון · ' + it.name : it.name))}"
          alt="${escapeHtml(p.cap || it.name)}" loading="lazy"
          referrerpolicy="no-referrer">
        ${canDrop(p, i) ? `<button class="shot-x" data-drop="${i}"
          aria-label="${p.yt ? 'הסרת הסרטון' : 'הסרת התמונה'}">&times;</button>` : ''}
      </span>`).join('')}
    </div>` : '';

  // One status line per detail pane and never two: `detailSay` finds it by id,
  // and a second element with the same id is one that never shows a word. It
  // used to be emitted by whichever branch happened to have editing controls,
  // which is three places to get that wrong; now there is one.
  const msg = Store.isEditor() ? '<p id="pub-msg" class="pub-msg" hidden></p>' : '';

  el('detail').innerHTML = `
    <h2>${escapeHtml(it.name)}</h2>
    <div class="chips">${chips.join('')}</div>
    ${it.note ? `<p class="note">${escapeHtml(it.note)}</p>` : ''}
    ${unwalkedNote(it)}
    ${tripParts(it)}
    ${body}
    ${msg}
    ${gallery}
    ${it.place
      ? `<a class="src" href="${escapeHtml(it.url)}" target="_blank" rel="noopener">
           ${escapeHtml(layer.sourceLine || 'המקור')} ↗</a>`
      : layer.kind === 'trails'
        ? `<a class="src" href="${DATA.source}" target="_blank" rel="noopener">המפה המקורית ב-Google My Maps ↗</a>`
        : ''}`;

  el('detail').querySelectorAll('.gallery img').forEach((img) => {
    img.addEventListener('click', () => openLightbox(it, +img.dataset.i));
  });
  el('detail').querySelectorAll('[data-goto]').forEach((btn) => {
    btn.addEventListener('click', () => select(btn.dataset.goto, true));
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

/* ---------- media on a layer ----------
 *
 * The side-car is keyed by item id, and a layer's id is a key in it like any
 * other. That is the whole of the feature on the storage side: no new
 * document, no change to what the worker will accept, and two editors
 * attaching to two different layers in the same second still merge.
 *
 * What it is for: a clip that explains a whole layer. The Not Just Bikes video
 * about Houten is about Houten, not about the fourteenth piece of its ring
 * road, and until this existed the only place to file it was on some segment
 * chosen for being long.
 */

/** The layer sheet's own status line, the counterpart of `detailSay`. */
function laySay(text, bad) {
  const msg = el('lay-msg');
  if (!msg) return;
  msg.hidden = !text;
  msg.textContent = text || '';
  msg.className = 'pub-msg' + (bad ? ' bad' : '');
}

/** Write to the side-car under a layer's id, then repaint the sheet.
 *
 *  `Layers.resetMedia` re-merges and redraws the map; the sheet is a separate
 *  repaint because it is the thing the person is looking at, and it is the one
 *  place the layer's own gallery is drawn. */
async function onLayerMedia(work, busy) {
  laySay(busy);
  try {
    Layers.resetMedia(await work());
    Layers.render();
    laySay('');
  } catch (err) {
    laySay('נכשל: ' + err.message, true);
  }
}

/** Show the document a write handed back.
 *
 *  Three documents, three ways back, and refreshing the wrong one leaves the
 *  pane showing what was just removed. The side-car is the cheap one: nothing
 *  about the map changed, only what hangs off one item, so the layers are
 *  re-merged rather than rebuilt. */
async function reloadAfter(home, doc) {
  if (home === 'media') Layers.resetMedia(doc);
  else if (home === 'places') reloadPlaces(doc);
  else await reloadShared(doc);
}

/** Editing an item that is already in the shared dataset: a published trail,
 *  or the position of a pardespedia place. */
/** Ask for a YouTube address and file it with the pictures.
 *
 *  One function for every kind of item, because the only difference is which
 *  document is written and which one is re-read afterwards.
 *
 *  `prompt` and not a form: this is one field, pasted from a phone's share
 *  sheet nine times out of ten, and every other single-field action in this app
 *  asks the same way. The address is checked in the store rather than here, so
 *  that whatever paths reach it get the same answer. */
async function addVideoTo(it, home, say) {
  const url = prompt('כתובת של סרטון יוטיוב:\n(אפשר גם קישור קצר של youtu.be)');
  if (url == null || !url.trim()) return;
  say('משבץ סרטון…');
  try {
    await reloadAfter(home, await Store.addVideo(it.id, url, it.name, home));
    select(it.id, false);
  } catch (err) {
    say('נכשל: ' + err.message, true);
  }
}

function wirePublished(it) {
  const say = detailSay();

  const layer = Layers.layerOf(it.id) || {};
  const home = mediaHome(it, layer);
  // Where the item's own pictures end and the attached ones begin. Anything at
  // or past this point is in the side-car whatever the item is; anything before
  // it is in the document the item came from.
  const base = (it.mediaBase || it.photos || []).length;

  el('detail').querySelectorAll('[data-drop]').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const clip = !!btn.closest('.shot.video');
      if (!confirm(clip ? 'להסיר את הסרטון הזה?' : 'להסיר את התמונה הזאת?')) return;
      btn.disabled = true;
      say(clip ? 'מסיר סרטון…' : 'מסיר תמונה…');
      try {
        const i = +btn.dataset.drop;
        // A pardespedia video added before the side-car existed is the one
        // thing this app may remove that is in neither the side-car nor
        // trails.json.
        const from = i >= base ? 'media' : (it.place ? 'places' : home);
        const doc = await Store.removePhoto(
          it.id, i >= base ? i - base : i, it.name, from);
        await reloadAfter(from, doc);
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

  /* Attaching to a trail of one's own that is still in the queue. Both writes
   * go to the two paths the worker leaves open - the queue itself and the
   * content-addressed images - so this needs no password, which is the same
   * reason sending the trail needed none. */
  el('detail').querySelectorAll('[data-mine]').forEach((node) => {
    if (node.dataset.mine === 'photos') {
      node.addEventListener('change', async () => {
        if (!node.files || !node.files.length) return;
        node.disabled = true;
        try {
          await Store.addPendingPhotos(it.id, [...node.files], it.name, say);
          await refreshQueue();
          select(it.id, false);
        } catch (err) {
          node.disabled = false;
          say('הצירוף נכשל: ' + err.message, true);
        }
      });
      return;
    }
    node.addEventListener('click', async () => {
      const url = prompt('כתובת של סרטון יוטיוב:', '');
      if (url === null) return;
      node.disabled = true;
      say('מצרף סרטון…');
      try {
        await Store.addPendingVideo(it.id, url, it.name);
        await refreshQueue();
        select(it.id, false);
      } catch (err) {
        node.disabled = false;
        say('הצירוף נכשל: ' + err.message, true);
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

    if (act === 'video') {
      node.addEventListener('click', () => addVideoTo(it, home, say));
      return;
    }

    if (act === 'photos') {
      node.addEventListener('change', async () => {
        if (!node.files || !node.files.length) return;
        say('מעלה…');
        try {
          await reloadAfter(home, await Store.addPhotos(
            it.id, [...node.files], it.name, say, home));
          select(it.id, false);
        } catch (err) {
          say('העלאה נכשלה: ' + err.message, true);
        }
      });
      return;
    }

    node.addEventListener('click', async () => {
      try {
        if (act === 'links') { linksForm(it, home); return; }

        if (act === 'note') {
          const note = prompt('הערה על המקום הזה:', it.noteExtra || '');
          if (note == null) return;
          node.disabled = true;
          say('שומר…');
          await reloadAfter('media', await Store.setNote(it.id, note, it.name));
          select(it.id, false);
          return;
        }
        if (act === 'move') { moveForm(it); return; }
        if (act === 'colour') { colourForm(it); return; }

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

function linksForm(it, home) {
  openForm(`
    <header class="sheet-head">
      <h2>קישורים</h2>
      <button class="sheet-x" data-act="close-form" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">קישורים שיופיעו במסך של "${escapeHtml(it.name)}": אתר, כתבה,
      ערך בפרדספדיה, אלבום תמונות. השאר שורה ריקה כדי למחוק אותה.${
        home === 'media' && (it.linksBase || []).length
          ? ' הקישורים שהמקור עצמו נושא נשארים ואינם נערכים כאן.' : ''}</p>
    ${LinkRows.html(home === 'media' ? (it.linksExtra || []) : it.links)}
    <p id="form-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="save-links"><b>שמור קישורים</b></button>`);
  formTarget = it;
  formHome = home || 'trails';
}

/* The palette a new layer picks from. Free colour entry on a phone is a colour
 * wheel nobody can hit precisely, and eight distinguishable colours is what a
 * map can carry anyway. */
const LAYER_COLOURS = ['#0b7285', '#c2255c', '#5f3dc4', '#e8590c',
                       '#2b8a3e', '#1864ab', '#a61e4d', '#495057'];

/* And the one a single trail picks from. It opens with the initiative's own
 * green, and carries the colours the trails imported from My Maps already use,
 * so recolouring one to match its neighbour is a matter of picking the same
 * swatch rather than guessing at a hex. */
const TRAIL_COLOURS = ['#097138', '#0b7285', '#1864ab', '#1a237e', '#5f3dc4',
                       '#880e4f', '#a52714', '#e65100', '#f57c00', '#817717',
                       '#495057'];

/** A row of colour buttons, shared by the layer form, the trail form and the
 *  sheet where a trail is first written.
 *
 *  Which one is picked lives in the DOM rather than in a variable, the way the
 *  link rows below do it, so a form can be opened and thrown away without
 *  leaving a stale choice behind it. */
const Swatches = {
  /** `inherit` adds a first, colourless swatch meaning "whatever the layer
   *  is". A colour already in use that is not in the palette is added too,
   *  so opening the form on such a trail cannot quietly recolour it. */
  html(colours, current, inherit) {
    const all = current && !colours.includes(current) ? [current, ...colours] : colours;
    const one = (c, on) => `<button type="button"
      class="sw${c ? '' : ' none'}${on ? ' on' : ''}" data-colour="${c}"
      style="--c:${c || 'transparent'}"
      aria-label="${c ? 'צבע ' + c : 'צבע השכבה'}"></button>`;
    return `<div class="swatches">
      ${inherit ? one('', !current) : ''}
      ${all.map((c) => one(c, c === current)).join('')}
    </div>`;
  },

  /** The colour picked inside `root`, or '' for the layer's own. */
  read(root) {
    const on = root.querySelector('.swatches .sw.on');
    return on ? on.dataset.colour : '';
  },

  /** One delegated listener, so a picker rendered later still works. */
  wire(root) {
    root.addEventListener('click', (e) => {
      const sw = e.target.closest('[data-colour]');
      if (!sw) return;
      sw.parentElement.querySelectorAll('.sw').forEach((s) =>
        s.classList.toggle('on', s === sw));
    });
  }
};

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
      ${Swatches.html(LAYER_COLOURS, now.color, false)}
    </div>
    <label class="check"><input type="checkbox" id="lay-dash" ${now.dash ? 'checked' : ''}>
      <span>קו מקווקו, לשבילים שעוד לא קיימים בשטח</span></label>
    <p id="form-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="save-layer"><b>${layer ? 'שמור' : 'צור שכבה'}</b></button>
    ${layer ? `<button class="big-act ghost danger" data-act="drop-layer"><b>מחיקת השכבה</b>
      <span>השבילים שבתוכה יחזרו לשכבת "דרכי קיצור", ולא יימחקו</span></button>` : ''}`);
  formTarget = layer || null;
}

/** The colour one trail is drawn in, over and above its layer's. */
function colourForm(it) {
  const layer = Layers.layerOf(it.id) || {};
  openForm(`
    <header class="sheet-head">
      <h2>צבע השביל</h2>
      <button class="sheet-x" data-act="close-form" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">באיזה צבע "${escapeHtml(it.name)}" ייראה על המפה, אצל כל
      מי שפותח את האפליקציה. הריק שבהתחלה הוא צבע השכבה עצמה${
        layer.name ? `, "${escapeHtml(layer.name)}"` : ''}.</p>
    ${Swatches.html(TRAIL_COLOURS, it.color || '', true)}
    <p id="form-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="save-colour"><b>שמור צבע</b></button>`);
  formTarget = it;
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
let formHome = 'trails';    // and which document its answer is written to

function formError(err) {
  const box = el('form-err');
  if (!box) return;
  box.textContent = err;
  box.hidden = false;
}

async function formAction(act, btn) {
  if (act === 'close-form') { closeForm(); return; }

  const label = btn.querySelector('b');
  const was = label ? label.textContent : '';
  if (label) label.textContent = 'שומר…';
  btn.disabled = true;

  try {
    if (act === 'save-links') {
      const links = LinkRows.read(el('form-card'));
      await reloadAfter(formHome, await Store.setLinks(
        formTarget.id, links, formTarget.name, formHome));
      closeForm();
      select(formTarget.id, false);

    } else if (act === 'save-layer') {
      const name = el('lay-name').value.trim();
      if (!name) throw new Error('צריך שם לשכבה.');
      const patch = {
        name,
        note: el('lay-note').value.trim(),
        color: Swatches.read(el('form-card')) || LAYER_COLOURS[0],
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

    } else if (act === 'save-colour') {
      await reloadShared(await Store.setColor(
        formTarget.id, Swatches.read(el('form-card')), formTarget.name));
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
  // Dropped rather than set to '', for the reason spelled out in
  // showPhotoStage: an empty src is not empty, it is this page.
  el('lb-img').removeAttribute('src');
  showPhotoStage();                           // and with it, silence the player
  lbReset();
  document.removeEventListener('keydown', lbKeys);
}

/** Put the stage back to showing pictures.
 *
 *  Silencing the player is the part that matters, and it is why this is a
 *  function rather than two lines repeated: a player left loaded goes on
 *  playing behind a closed lightbox, and the sound comes from nowhere with
 *  nothing on screen to stop it.
 *
 *  `src = ''` is the obvious way to do that and it is wrong. An empty string is
 *  resolved against the document, so the frame does not go blank - it loads
 *  *this app* inside itself, a second full copy running invisibly behind the
 *  map. Measured: closing the lightbox fetched index.html again. about:blank is
 *  a real navigation away from YouTube, which is what actually stops the sound,
 *  and dropping the attribute afterwards keeps the next read from resolving to
 *  the page URL all over again. */
function showPhotoStage() {
  const video = el('lb-video');
  if (video.getAttribute('src')) video.src = 'about:blank';
  video.removeAttribute('src');
  video.hidden = true;
  el('lb-img').hidden = false;
}

function paintLightbox() {
  const img = el('lb-img');
  const photo = lbPhotos[lbIndex];
  lbReset();                                  // a new photo arrives fitted

  if (photo.yt) {
    img.hidden = true;
    img.removeAttribute('src');
    img.onerror = null;                       // the photo's fallback is not this one's
    el('lb-spin').hidden = true;
    const video = el('lb-video');
    video.hidden = false;
    // autoplay, because getting here already took a deliberate press; rel=0 so
    // what follows comes from the same channel; nocookie so that opening the
    // lightbox does not set a tracking cookie for somebody who never pressed play.
    video.src = 'https://www.youtube-nocookie.com/embed/'
      + encodeURIComponent(photo.yt) + '?autoplay=1&rel=0';
    paintLightboxCaption(photo);
    return;
  }

  showPhotoStage();
  img.classList.add('loading');
  el('lb-spin').hidden = false;
  img.src = photo.full;                       // full resolution, not the thumb
  img.alt = photo.cap || lbTitle;
  img.onload = () => { img.classList.remove('loading'); el('lb-spin').hidden = true; };

  // A photo that lives on somebody else's host can refuse the big rendition
  // while still serving the small one: both מקום שמור hosts throttle by the
  // hour, and the thumbnail is already in this browser's cache because the
  // gallery just drew it. Showing the small version beats showing the broken
  // picture glyph over a caption, so falling back is the last thing tried
  // before giving up.
  img.onerror = () => {
    el('lb-spin').hidden = true;
    img.classList.remove('loading');
    if (photo.thumb && img.src !== photo.thumb) img.src = photo.thumb;
  };

  paintLightboxCaption(photo);
}

/** The caption and the counter, shared by both kinds of item.
 *
 *  The caption a photo arrives with says more than the name of the place: the
 *  year, the photographer, the archive the print came from. Where מקום שמור
 *  wrote one it is shown instead of the name, credit and all. */
function paintLightboxCaption(photo) {
  el('lb-cap-text').textContent = photo.cap || lbTitle;
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

/* ---------- zooming inside the lightbox ----------
 *
 * A photo is stored 1600px wide, and what somebody wants to see in it is often
 * a few dozen of those pixels: which gate, which sign, where exactly the path
 * leaves the road. Fitted to the screen that detail is unreadable, so the wheel
 * magnifies towards the pointer and drag moves the picture underneath it.
 *
 * The scale and the offset live here rather than on the element, because every
 * change needs the previous values to work out the next ones.
 */

const LB_MAX = 8;                             // beyond this a 1600px photo is mush
let lbScale = 1, lbTx = 0, lbTy = 0, lbDragged = false;

function lbApply() {
  const img = el('lb-img');
  img.style.transform = `translate(${lbTx}px, ${lbTy}px) scale(${lbScale})`;
  img.classList.toggle('zoomed', lbScale > 1);
}

function lbReset() {
  lbScale = 1;
  lbTx = 0;
  lbTy = 0;
  lbApply();
}

/** Keep the picture over the stage: it may be moved exactly as far as it
 *  overhangs, and no further, so no edge ever shows a black gap and the photo
 *  can never be pushed off into the dark and lost.
 *
 *  An axis the picture does not yet fill has no overhang, so it is held in the
 *  middle. That is not a compromise: with nothing cut off in that direction
 *  there is nothing there to move to, and the centre is the only honest place
 *  for it. Aiming with the wheel therefore takes hold in each axis exactly when
 *  that axis starts hiding something - immediately in the one the fitted photo
 *  already fills, and from the moment it overflows in the other. */
function lbClamp(fit) {
  const box = lbStageBox();
  const overX = Math.max(0, (fit.w * lbScale - box.w) / 2);
  const overY = Math.max(0, (fit.h * lbScale - box.h) / 2);
  lbTx = Math.min(overX, Math.max(-overX, lbTx));
  lbTy = Math.min(overY, Math.max(-overY, lbTy));
}

/** How big the picture is on screen when it is not magnified, to sub-pixel
 *  accuracy, worked back out of what is currently drawn.
 *
 *  offsetWidth would read the same number more simply, but it is rounded to
 *  whole pixels, and half a pixel of rounding becomes four pixels of black at
 *  the edge once the photo is eight times its fitted size.
 *
 *  Only meaningful while what is on screen still matches lbScale: before a
 *  change, or after lbApply has drawn one. */
function lbFitted() {
  const r = el('lb-img').getBoundingClientRect();
  return { w: r.width / lbScale, h: r.height / lbScale };
}

/** The area a photo is allowed to occupy: the stage minus its padding.
 *
 *  Not the stage's full rectangle. It is padded at the top to keep a fitted
 *  photo clear of the close button, and that band is meant to stay dark. The
 *  picture rests centred in the area inside the padding, so measuring the same
 *  area here is what keeps the limits symmetrical around where it sits. */
function lbStageBox() {
  const stage = document.querySelector('.lb-stage');
  const cs = getComputedStyle(stage);
  return {
    w: stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight),
    h: stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)
  };
}

/** Scale by `factor`, holding still whatever sits under the pointer.
 *
 *  Without that anchoring the wheel would zoom the middle of the screen and the
 *  thing being examined would slide away exactly when it got interesting. */
function lbZoomAt(factor, cx, cy) {
  const r = el('lb-img').getBoundingClientRect();
  if (!r.width) return;                       // nothing loaded yet
  const next = Math.min(LB_MAX, Math.max(1, lbScale * factor));
  if (next === lbScale) return;

  const fit = { w: r.width / lbScale, h: r.height / lbScale };
  const fx = (cx - r.left) / r.width;
  const fy = (cy - r.top) / r.height;
  // The element scales about its own centre, so a point at fraction f of it
  // moves by size*(Δscale)*(f - ½); cancelling that is what pins it down.
  lbTx += fit.w * (next - lbScale) * (0.5 - fx);
  lbTy += fit.h * (next - lbScale) * (0.5 - fy);
  lbScale = next;

  if (lbScale === 1) { lbTx = 0; lbTy = 0; } else lbClamp(fit);
  lbApply();
}

function wireLightboxZoom() {
  const stage = document.querySelector('.lb-stage');

  stage.addEventListener('wheel', (e) => {
    e.preventDefault();                       // no page behind this to scroll
    // A mouse wheel sends one large delta per notch and a trackpad a stream of
    // small ones; a browser may also report lines instead of pixels. Through an
    // exponent both end up moving the same amount per unit scrolled.
    const dy = e.deltaY * (e.deltaMode === 1 ? 16 : 1);
    lbZoomAt(Math.exp(-dy * 0.0015), e.clientX, e.clientY);
  }, { passive: false });

  // Dragging is only meaningful once there is more picture than screen. Touch
  // is left alone: there a horizontal drag already means "next photo".
  let drag = null;
  stage.addEventListener('pointerdown', (e) => {
    if (lbScale === 1 || e.pointerType === 'touch') return;
    drag = { x: e.clientX, y: e.clientY };
    lbDragged = false;
    stage.setPointerCapture(e.pointerId);
  });

  stage.addEventListener('pointermove', (e) => {
    if (!drag) return;
    lbTx += e.clientX - drag.x;
    lbTy += e.clientY - drag.y;
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 0) lbDragged = true;
    drag = { x: e.clientX, y: e.clientY };
    lbClamp(lbFitted());
    lbApply();
  });

  const release = (e) => {
    if (!drag) return;
    drag = null;
    if (stage.hasPointerCapture(e.pointerId)) stage.releasePointerCapture(e.pointerId);
  };
  stage.addEventListener('pointerup', release);
  stage.addEventListener('pointercancel', release);
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

/* ---------- the line you are meant to walk ----------
 *
 * The bar alone was a distance, a compass word and an arrow, and somebody
 * seeing it for the first time had no way in: nothing on the map said where
 * the target was, so there was nothing for the number to be a number *of*.
 * The arrow made it worse rather than better, because without a compass in the
 * device it points to north rather than to where you are looking.
 *
 * A dashed line from where you are to where you are going answers all of it at
 * once, with no wording at all. It is deliberately straight: this is a bearing
 * and a distance, not a route, and drawing it as a route would promise a way
 * through that nobody has checked.
 */

const NAV_SRC = 'src-nav';

function navGeoJSON(target) {
  return {
    type: 'FeatureCollection',
    features: [
      { type: 'Feature',
        geometry: { type: 'LineString',
                    coordinates: [[here.lng, here.lat], [target.lng, target.lat]] },
        properties: {} },
      { type: 'Feature',
        geometry: { type: 'Point', coordinates: [target.lng, target.lat] },
        properties: {} }
    ]
  };
}

function paintNavLine(target) {
  if (!map || !nav || !here || !target || !map.isStyleLoaded()) return;
  const data = navGeoJSON(target);
  if (map.getSource(NAV_SRC)) { map.getSource(NAV_SRC).setData(data); return; }

  map.addSource(NAV_SRC, { type: 'geojson', data });
  map.addLayer({
    id: 'nav-line',
    type: 'line',
    source: NAV_SRC,
    filter: ['==', ['geometry-type'], 'LineString'],
    layout: { 'line-cap': 'round' },
    paint: {
      'line-color': '#0f4c1a',
      'line-width': 4,
      'line-opacity': 0.9,
      // Dashes rather than a solid line, for the same reason it is straight:
      // a solid line on a map of walking routes reads as one more route.
      'line-dasharray': [1.6, 1.4]
    }
  });
  map.addLayer({
    id: 'nav-target',
    type: 'circle',
    source: NAV_SRC,
    filter: ['==', ['geometry-type'], 'Point'],
    paint: {
      'circle-radius': 9,
      'circle-color': '#0f4c1a',
      'circle-stroke-color': '#fff',
      'circle-stroke-width': 3
    }
  });
}

function clearNavLine() {
  if (!map) return;
  ['nav-line', 'nav-target'].forEach((id) => {
    if (map.getLayer(id)) map.removeLayer(id);
  });
  if (map.getSource(NAV_SRC)) map.removeSource(NAV_SRC);
}

/* ---------- leaving the moshava, and coming back ----------
 *
 * Every layer here used to be within a few kilometres of every other one, so
 * the map's own view was always roughly the right view. The האוטן layer broke
 * that: it is a town in the Netherlands, and switching it on while looking at
 * Pardes Hanna turns a layer on that is three thousand kilometres off screen -
 * which is indistinguishable from a layer that does not work.
 *
 * So a layer may carry `bounds`, and turning it on flies there. That is the
 * only reason to tick that particular box, and doing it silently would be the
 * surprising choice rather than the polite one.
 */

/** Frame a layer that lives somewhere else.
 *
 *  Switching it on does not re-frame when the map is already looking at it:
 *  somebody who has zoomed into one Dutch street and toggles the layer off and
 *  on to compare should not be yanked back out to the whole town. Asking for it
 *  by name - the "טוס לשם" button - always flies, because from a street that is
 *  what "show me the layer" has to mean. */
function frameLayer(layer, force = true) {
  if (!map || !layer || !layer.bounds) return;
  const [[s, w], [n, e]] = layer.bounds;
  const here = map.getCenter();
  const inside = here.lat > s && here.lat < n && here.lng > w && here.lng < e;
  if (inside && !force) return;
  map.fitBounds([[w, s], [e, n]], { padding: 40, duration: 1200 });
}

/** The way back. Shown only once the map has actually left the area, which is
 *  the only time it means anything - and it is measured against the trails'
 *  own bounds rather than a hardcoded point, so it stays true if the dataset
 *  ever grows past the moshava. */
function updateHomeButton() {
  const btn = el('go-home');
  if (!btn || !map || !DATA || !DATA.bounds) return;
  const [[s1, s2], [n1, n2]] = DATA.bounds;
  const here = map.getCenter();
  // A degree of latitude is about 111 km, so a fifth of a degree outside the
  // dataset is roughly twenty kilometres away: far enough that nothing on this
  // map is on screen, near enough that a drive up the coast does not trigger it.
  const pad = 0.2;
  btn.hidden = here.lat > s1 - pad && here.lat < n1 + pad
    && here.lng > s2 - pad && here.lng < n2 + pad;
}

function goHome() {
  if (!map || !DATA || !DATA.bounds) return;
  const [[s1, s2], [n1, n2]] = DATA.bounds;
  map.fitBounds([[s2, s1], [n2, n1]], { padding: 24, duration: 1200 });
}

/** Put both ends on screen, once, when the first fix arrives.
 *
 *  Once and not on every fix: after this the map is the person's to pan, and
 *  a viewport that re-frames itself every few seconds cannot be read. */
function frameNav(target) {
  if (!map || nav.framed) return;
  nav.framed = true;
  const b = new maplibregl.LngLatBounds();
  b.extend([here.lng, here.lat]);
  b.extend([target.lng, target.lat]);
  map.fitBounds(b, { padding: 90, maxZoom: 17, duration: 800 });
}

function startNav(item) {
  if (!navigator.geolocation) { alert('הדפדפן לא תומך באיתור מיקום.'); return; }
  stopNav();
  nav = { item, watchId: null, endIdx: null, framed: false };
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
  clearNavLine();
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

  const course = bearingTo(here, target);
  // With a heading we can point where to actually walk; without one the arrow
  // is north-up, so say so rather than sending someone the wrong way. The
  // wording is spelled out because "(חץ לפי צפון)" told the truth to somebody
  // who already knew what it meant, and nothing to anybody else.
  el('nav-state').textContent = facing == null
    ? `${state} · ${compass(course)} · החץ מיושר לצפון`
    : `${state} · ${compass(course)}`;

  document.querySelector('.nav-arrow').style.transform =
    `rotate(${course - (facing || 0)}deg)`;

  // The line is the part that actually explains the bar, so it is drawn on
  // every fix and the framing happens once, on the first.
  paintNavLine(target);
  frameNav(target);
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

  paintUnsent();

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

/** The strip that says a trail is sitting on this phone and going nowhere.
 *
 *  Counted off the drafts layer rather than off IndexedDB, so it is right the
 *  instant a draft is saved, sent or deleted - every one of those repaints.
 *  Hidden entirely at zero: a permanent bar reading "0 unsent" would be one
 *  more thing to read past, and this one is meant to be unmissable. */
function paintUnsent() {
  const bar = el('unsent');
  const layer = Layers.byId('drafts');
  const n = layer ? layer.segments.length : 0;
  bar.hidden = !n;
  // Two stacked cards over a panel that is 45% of a phone leave about one row
  // of list. The invitation's explainer is onboarding copy - "walk it with your
  // phone, no account needed" - and somebody who already has a draft has done
  // exactly that, so it is the half that goes. The button itself stays: having
  // one unsent trail is no reason to be unable to start a second.
  el('add').classList.toggle('compact', !!n);
  if (!n) return;

  // An editor's own drafts are not waiting to be sent anywhere; they are
  // waiting to be published, which is the same sentence with a different verb
  // and a different button at the end of it. Both forms of each verb are
  // written out rather than derived: Hebrew inflection is not string surgery.
  const [one, many] = Store.isEditor()
    ? ['שביל אחד שעוד לא פורסם', `${n} שבילים שעוד לא פורסמו`]
    : ['שביל אחד שעוד לא נשלח', `${n} שבילים שעוד לא נשלחו`];
  bar.innerHTML = `
    <span class="unsent-ic" aria-hidden="true">✏️</span>
    <span class="unsent-txt">
      <b>${n === 1 ? one : many}</b>
      <span>שמור אצלך בלבד. לחץ כדי לפתוח ${n === 1 ? 'אותו' : 'את האחרון'}.</span>
    </span>`;
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
  else if (selectedId) {
    // The ring is re-hung rather than left where it was: a repaint follows a
    // pin being dragged or a layer being rebuilt, and the selected place may
    // not be where the ring is any more.
    markSelection(byId(selectedId));
    showDetail(byId(selectedId));
  }
  else renderList();
}

async function boot() {
  const wantedBg = urlBasemap();
  if (wantedBg) setBasemap(wantedBg);
  el('basemap').title = 'רקע: ' + BASEMAPS[baseIndex].name;
  // Fires for the initial style and again after every setBasemap.
  if (map) map.on('style.load', applyOverlays);

  const data = await Store.load();
  DATA = data.trails;
  PLACES = data.places;
  Layers.init(data);
  Layers.onChange = repaint;

  // The list, the search and the buttons come up as soon as the data lands.
  // Waiting for the map style first would leave the whole panel dead on a weak
  // connection, which is exactly the connection you have out on a trail.
  renderList();
  paintStats();
  Store.statVisit();
  Drafts.init();
  wireControls();
  // After the panel is up and wired, and before the network is waited on: the
  // invitation is about what this app is for, and that does not depend on the
  // worker being reachable.
  welcome();
  // Confirming the stored token needs the network, so it must not hold up the
  // list. The editor badge and the publish buttons appear a moment later.
  Store.resume().then(() => { repaint(); refreshQueue(); });

  if (map) {
    await new Promise((done) => (map.isStyleLoaded() ? done() : map.once('load', done)));
    // A link that carries a camera is somebody saying "look at this". It wins
    // over the opening view, which is only ever a guess at where to start.
    const view = urlView();
    if (view) map.jumpTo(view);
    else {
      const [[s1, s2], [n1, n2]] = DATA.bounds;
      map.fitBounds([[s2, s1], [n2, n1]], { padding: 24, duration: 0 });
    }
    Layers.addToMap();
    drawWaypoints();
    el('tilt').classList.toggle('on', map.getPitch() >= 10);

    // Read what the link asked for *before* writing anything back: syncView
    // rewrites the whole query from the live state, and with nothing selected
    // yet that erases the very parameter naming what to select.
    const wanted = new URLSearchParams(location.search).get(SEL_KEY);

    // From here the address bar tracks the map. `moveend` covers panning,
    // zooming, rotating and tilting alike.
    ['moveend', 'pitchend', 'rotateend'].forEach((ev) => map.on(ev, scheduleSync));
    // The way back appears and disappears with the same move. `move` and not
    // `moveend`, so a link that lands on Houten shows it during the flight
    // rather than only once the camera has settled.
    map.on('move', updateHomeButton);
    updateHomeButton();

    if (wanted && Layers.item(wanted)) select(wanted, !view);
    syncView();
  }

  // Last, and outside the `if (map)`: reopening an interrupted recording draws
  // on the map, so it has to come after the style and the layers are up, and it
  // still has to happen on a browser that never got a map at all.
  Drafts.restore();
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
 *  An editor gets all of it, because deciding what goes on the map is what the
 *  queue is for. Somebody who has sent a trail in gets their own rows and
 *  nothing else, so that "waiting for approval" is a line on the map they can
 *  go and look at rather than a sentence they were shown once. Everybody else
 *  is not asking a question the queue answers, so no request is made at all. */
async function refreshQueue() {
  const ledger = Store.sent();
  if (!Store.isEditor() && !ledger.length) { Layers.setPending([]); return; }
  try {
    const doc = await Store.queue();
    const items = doc.items || [];
    if (Store.isEditor()) { Layers.setPending(items); return; }
    const ids = new Set(ledger.map((row) => row.id));
    Layers.setPending(items.filter((item) => ids.has(item.id)), { mine: true });
    settle(ledger, items);
  } catch (err) {
    console.error('queue unavailable', err);
  }
}

/** Tell somebody what became of a trail they sent, once, and then stop.
 *
 *  A row that has left the queue was either approved onto the map or turned
 *  down, and this app cannot tell which from the outside: both look like an id
 *  that is no longer there. So it looks for the trail by name among the
 *  published ones and says only what it can actually see. Guessing "approved"
 *  would eventually congratulate somebody on a trail that was rejected. */
function settle(ledger, items) {
  const live = new Set(items.map((item) => item.id));
  const gone = ledger.filter((row) => !live.has(row.id));
  if (!gone.length) return;

  const published = (name) => [Layers.TRAILS_ID, Layers.TRIPS_ID]
    .map((id) => Layers.byId(id))
    .some((layer) => layer && layer.segments.some(
      (seg) => seg.name.trim() === String(name || '').trim()));

  const landed = gone.filter((row) => published(row.name));
  const rest = gone.filter((row) => !published(row.name));
  gone.forEach((row) => Store.forget(row.id));

  const lines = [];
  if (landed.length) {
    lines.push(`<p class="sheet-lead">${landed.length === 1
      ? `<b>${escapeHtml(landed[0].name)}</b> אושר ונמצא עכשיו על המפה, לעיני כולם.`
      : `${landed.length} מהשבילים ששלחת אושרו ונמצאים עכשיו על המפה.`} תודה. 🥾</p>`);
  }
  if (rest.length) {
    lines.push(`<p class="sheet-credit">${rest.length === 1
      ? `<b>${escapeHtml(rest[0].name)}</b> כבר לא בתור`
      : `${rest.length} מהשבילים ששלחת כבר לא בתור`}. או שהוא נוסף למפה בשם
      אחר, או שהיוזמה החליטה שלא להוסיף אותו. אפשר לשאול אותם.</p>`);
  }
  notice(`
    <header class="sheet-head">
      <h2>מה קרה למה ששלחת</h2>
      <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
    </header>
    ${lines.join('')}
    <button class="big-act primary" data-act="close"><b>סגירה</b></button>`);
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

/* ---------- the invitation ----------
 *
 * This map exists because residents walked the shortcuts on it: 41 of the 49
 * are on no other map in the world. Nothing on the screen said so. A visitor
 * met a finished-looking product with fifty trails and a search box, and the
 * only way in was a chip called "+ שביל חדש" sitting between "הארוכים" and
 * "הקרובים", which reads as a filter.
 *
 * Two things now say it. A permanent button at the top of the list, and this
 * sheet on a first visit - shown once per browser and then never again, because
 * an invitation that repeats is an obstacle.
 */

const K_WELCOME = 'dk.welcome.v1';

/** One sheet for anything the app has to say on arrival: the invitation on a
 *  first visit, and what became of a submitted trail on a later one. Never both
 *  at once - the second needs a trail to have been sent, which needs the first
 *  to be long past. */
function notice(html) {
  el('welcome-card').innerHTML = html;
  el('welcome-sheet').hidden = false;
}

function seenWelcome() {
  try {
    return localStorage.getItem(K_WELCOME) === 'yes';
  } catch (err) {
    return true;                        // no storage: never nag
  }
}

function markWelcome() {
  try {
    localStorage.setItem(K_WELCOME, 'yes');
  } catch (err) {
    /* private mode; it will be offered again next time, which is harmless */
  }
}

function welcome() {
  if (seenWelcome()) return;
  markWelcome();                        // shown counts, whatever they choose
  const s = Layers.stats();
  notice(`
    <header class="sheet-head">
      <h2>את המפה הזאת בנו תושבים</h2>
      <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">רוב קיצורי הדרך שכאן לא מופיעים בשום מפה אחרת. הם כאן
      כי מישהו מהמושבה הלך בהם עם הטלפון ושלח אותם.${s.segments
        ? ` ${s.segments} מקטעים עד עכשיו.` : ''}</p>
    <p class="sheet-lead">מכיר קיצור דרך שעוד לא כאן? לך בו מקצה לקצה והאפליקציה
      תצייר אותו לפי ה-GPS. אפשר גם לצייר אותו על המפה מהבית, ולצרף תמונות
      וסרטון.</p>
    <button class="big-act primary" data-act="add"><b>להוסיף דרך קיצור</b>
      <span>בלי הרשמה, בלי חשבון ובלי סיסמה. רק שם שייכתב לידה</span></button>
    <button class="big-act ghost" data-act="close"><b>אחר כך</b>
      <span>הכפתור נשאר בראש הרשימה</span></button>`);
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

  // `?local` reads the files next to the app and writes to the live dataset
  // all the same, which is a combination worth saying out loud: a save lands
  // in the repo and then the screen goes back to showing the older copy.
  const localData = Store.RAW === './' ? `
    <p class="pub-msg bad">האפליקציה קוראת נתונים מקומיים (<code>?local</code>),
      אבל כותבת למסד האמיתי. מה שתשמור ייכתב לריפו ואחרי רענון המסך יראה שוב את
      העותק הישן. להורדת <code>?local</code> מהכתובת יש מסד אחד בלבד.</p>` : '';

  el('editor-card').innerHTML = Store.isEditor() ? `${head}${localData}
    <p class="sheet-lead">מצב עריכה דלוק. שביל שתפרסם נכנס למפה מיד וכל מי שיפתח
      את האפליקציה יראה אותו.</p>
    <label class="fld"><span>איך לקרוא לך (לא חובה)</span>
      <input id="ed-name" type="text" maxlength="40" value="${escapeHtml(Store.named())}"
             placeholder="השם שיירשם ליד השינויים שלך"></label>
    <button class="big-act primary" data-act="save-name"><b>שמור שם</b></button>
    <button class="big-act" data-act="out"><b>כבה מצב עריכה</b>
      <span>הכפתורים ייעלמו מהמסך והסיסמה תישכח במכשיר. הטיוטות שלך נשארות.</span></button>` : `${head}
    <p class="sheet-lead">כדי להוסיף שביל למפה לא צריך שום דבר מכאן: לוחצים על
      "הוסף דרך קיצור" בראש הרשימה, הולכים בשביל, ושולחים. מצב עריכה הוא משהו
      אחר, והוא מיועד למי שמאשר מה נכנס למפה.</p>
    <label class="fld"><span>איך לקרוא לך</span>
      <input id="ed-name" type="text" maxlength="40" value="${escapeHtml(Store.named())}"
             placeholder="השם שיירשם ליד מה שתוסיף"></label>
    <button class="big-act" data-act="save-name"><b>שמור שם</b>
      <span>זה הקרדיט שיופיע ליד שביל ששלחת. נשמר במכשיר שלך בלבד</span></button>
    <label class="fld"><span>סיסמת עריכה</span>
      <input id="ed-key" type="password" autocomplete="current-password"
             placeholder="הסיסמה שנשמרת בשרת"></label>
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
  // `setName` and not `enable`: the two used to be the same call, and the name
  // is not a permission - somebody without the password setting what to be
  // credited as would have been refused by the door they were not knocking on.
  if (act === 'save-name') { Store.setName(name); editorSheet(); repaint(); return; }
  if (act === 'out') {
    Store.disable();
    Arrange.close(true);
    editorSheet();
    repaint();
    // Not setPending([]): leaving edit mode does not undo having sent a trail,
    // and this browser may well have some of its own still waiting.
    refreshQueue();
  }
}

function wireControls() {
  el('welcome-sheet').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-act]');
    if (e.target.id !== 'welcome-sheet' && !btn) return;
    el('welcome-sheet').hidden = true;
    // The point of the sheet. Going through the button on the panel rather than
    // calling askMode directly keeps one entry into drafting, so there is one
    // place to change when it moves again.
    if (btn && btn.dataset.act === 'add') el('add').click();
  });

  // The drafts layer is newest first, so the first row is the one just made.
  el('unsent').addEventListener('click', () => {
    const layer = Layers.byId('drafts');
    if (layer && layer.segments.length) select(layer.segments[0].id);
  });

  el('editor-btn').addEventListener('click', editorSheet);
  el('editor-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'editor-sheet') { el('editor-sheet').hidden = true; return; }
    const btn = e.target.closest('[data-act]');
    if (btn) editorAction(btn.dataset.act);
  });

  el('go-home').addEventListener('click', goHome);

  el('layers').addEventListener('click', () => { laySay(''); Layers.openSheet(); });
  el('layer-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'layer-sheet' || e.target.closest('[data-act="close"]')) {
      Layers.closeSheet();
      return;
    }

    // The layer's own gallery. Before the tile itself, because the remove
    // button sits on top of one and a press on it is not a press on the photo.
    const drop = e.target.closest('[data-lay-drop]');
    if (drop) {
      const layer = Layers.byId(drop.dataset.layDrop);
      const i = +drop.dataset.i;
      const clip = !!(layer.photos[i] || {}).yt;
      if (!confirm(clip ? 'להסיר את הסרטון הזה?' : 'להסיר את התמונה הזאת?')) return;
      // The index in the gallery, less what the layer's own document shipped:
      // the side-car's list starts where that one ends.
      const at = i - (layer.mediaBase || layer.photos).length;
      onLayerMedia(() => Store.removePhoto(layer.id, at, layer.name, 'media'),
        clip ? 'מסיר סרטון…' : 'מסיר תמונה…');
      return;
    }
    const shot = e.target.closest('[data-shot]');
    if (shot) {
      openLightbox(Layers.byId(shot.dataset.shot), +shot.dataset.i);
      return;
    }
    const clip = e.target.closest('[data-lay-video]');
    if (clip) {
      const layer = Layers.byId(clip.dataset.layVideo);
      const url = prompt(`סרטון יוטיוב על "${layer.name}", לשכבה כולה:\n`
        + '(אפשר גם קישור קצר של youtu.be)');
      if (url == null || !url.trim()) return;
      onLayerMedia(() => Store.addVideo(layer.id, url, layer.name, 'media'),
        'משבץ סרטון…');
      return;
    }

    if (e.target.closest('[data-clearall]')) { Layers.clearAll(); return; }
    if (e.target.closest('[data-newlayer]')) { layerForm(null); return; }
    if (e.target.closest('[data-arrange]')) {
      Layers.closeSheet();
      startPinning(null);
      return;
    }
    // A layer somewhere else stays reachable after the flight that switching it
    // on gave you: come back a week later with it still ticked and the map on
    // the moshava, and this is the way to it that does not involve guessing that
    // toggling the box twice is what does it.
    const fly = e.target.closest('[data-fly]');
    if (fly) {
      Layers.closeSheet();
      frameLayer(Layers.byId(fly.dataset.fly));
      return;
    }
    const edit = e.target.closest('[data-edit]');
    if (edit) layerForm(Layers.byId(edit.dataset.edit));
  });

  // Pictures onto a layer. A file input answers `change` and not `click`, and
  // the sheet is repainted on every write, so this is delegated on the sheet
  // for the same reason the clicks are: the input that fired it may already
  // have been replaced by the time the upload finishes.
  el('layer-sheet').addEventListener('change', (e) => {
    const input = e.target.closest('[data-lay-photos]');
    if (!input || !input.files || !input.files.length) return;
    const layer = Layers.byId(input.dataset.layPhotos);
    const files = [...input.files];
    onLayerMedia(() => Store.addPhotos(layer.id, files, layer.name, laySay, 'media'),
      'מעלה…');
  });

  el('form-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'form-sheet') { closeForm(); return; }
    if (e.target.closest('[data-colour]')) return;   // Swatches.wire has it
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
  Swatches.wire(el('form-sheet'));

  Arrange.wire();

  el('search').addEventListener('input', renderList);
  el('back').addEventListener('click', deselect);
  el('nav-stop').addEventListener('click', stopNav);

  // Tapping the bar frames both ends again. Walking with the map open means
  // panning it, and after a couple of pans neither you nor the target is on
  // screen; without this the only way back was to stop and start again.
  el('nav').addEventListener('click', (e) => {
    if (e.target.closest('#nav-stop') || !nav || !here) return;
    nav.framed = false;
    paintNav();
  });
  el('locate').addEventListener('click', locate);
  // Needs WebGL and a map to fly over, so it goes away with the other two.
  if (map) el('explore').addEventListener('click', () => Explore.enter());
  else el('explore').hidden = true;
  el('basemap').addEventListener('click', () => {
    setBasemap((baseIndex + 1) % BASEMAPS.length);
    syncView();
  });

  if (map) {
    // The map opens tilted, and `pitchend` only ever fires after somebody has
    // moved it - so without this the button would sit unlit over a tilted map
    // until the first drag.
    el('tilt').classList.toggle('on', map.getPitch() >= 10);
    el('tilt').addEventListener('click', () => {
      const flat = map.getPitch() < 10;
      map.easeTo({ pitch: flat ? TILTED : 0, duration: 700 });
      el('tilt').classList.toggle('on', flat);
      scheduleSync();
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
    // A pan that ended over the dark surround is still a click on it, and
    // closing there would throw away the very view being framed.
    if (lbDragged) { lbDragged = false; return; }
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
  wireLightboxZoom();
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
