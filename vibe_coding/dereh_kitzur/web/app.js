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

let DATA = null;          // the trails layer's own file, for its source link
let baseIndex = 0;
let here = null;          // {lat, lng} once geolocation succeeds
let hereMarker = null;
let selectedId = null;
let sortMode = 'length';
let wpMarkers = [];       // waypoint pins, rebuilt when layers are toggled

/* ---------- helpers ---------- */

const metres = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + ' ק"מ' : m + ' מ׳');

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
  if (!here) return Infinity;
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
  Layers.visibleWaypoints().forEach((wp) => {
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
    } else {
      map.easeTo({ center: [item.lng, item.lat], zoom: 18, duration: 700 });
    }
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

  if (q) {
    const needle = q.toLowerCase();
    all = all.filter((it) =>
      [it.name, it.note, (it.connects || []).join(' '), (it.streets || []).join(' ')]
        .join(' ').toLowerCase().includes(needle));
  }

  if (sortMode === 'length') {
    all.sort((a, b) => (b.length || 0) - (a.length || 0));
  } else if (sortMode === 'photos') {
    all = all.filter((it) => it.photos && it.photos.length);
    all.sort((a, b) => b.photos.length - a.photos.length);
  } else if (sortMode === 'near') {
    all.sort((a, b) => nearestMetres(a) - nearestMetres(b));
  }
  return all;
}

function subtitle(it) {
  const bits = [];
  if (it.length) bits.push(metres(it.length));
  else bits.push('נקודת ציון');
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
  if (!layer || layer.kind === 'trails') return '';
  return `<span class="tag" style="--c:${layer.color}">${escapeHtml(layer.short)}</span>`;
}

function renderList() {
  const list = el('list');
  const rows = items();

  if (!rows.length) {
    list.innerHTML = Layers.visible().length
      ? '<li class="empty-msg">לא נמצא שביל תואם.</li>'
      : '<li class="empty-msg">כל השכבות כבויות.<br>פתח את כפתור השכבות והדלק אחת.</li>';
    return;
  }

  list.innerHTML = rows.map((it) => {
    const thumb = it.photos && it.photos.length
      ? `<img class="thumb" src="${it.photos[0].thumb}" alt="" loading="lazy">`
      : `<div class="thumb empty">${it.path ? (it.draft ? '✏️' : '🥾') : '📍'}</div>`;
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
  const first = it.entries ? it.entries[0] : it;
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

function showDetail(it) {
  const layer = Layers.layerOf(it.id) || {};
  const chips = [];

  if (it.length) chips.push(`<span class="chip accent">${metres(it.length)}</span>`);
  else chips.push('<span class="chip accent">נקודת ציון</span>');
  if (layer.kind !== 'trails') {
    chips.push(`<span class="chip layer" style="--c:${layer.color}">${escapeHtml(layer.name)}</span>`);
  }
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

  const entries = it.entries || [{ lat: it.lat, lng: it.lng, heading: 0, view: null }];
  it = { ...it, entries };

  let body;
  if (layer.kind === 'network') {
    const labels = ['תחילת המקטע', 'סוף המקטע'];
    body = `
      ${it.streets && it.streets.length
        ? `<p class="note">עובר לאורך ${escapeHtml(it.streets.join(', '))}.</p>` : ''}
      <h3>הגעה</h3>
      <div class="acts">
        ${panoActs(it, labels, 'מבט 360° לאורך המקטע')}
        ${navActs(it, 'ניווט בתוך האפליקציה, לאורך תוואי המקטע')}
      </div>
      <p class="src">${escapeHtml(layer.credit || '')}</p>`;
  } else if (it.draft) {
    body = `
      <h3>הגעה</h3>
      <div class="acts">${navActs(it, 'ניווט לפי התוואי שהקלטת')}</div>
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
      ${Store.isEditor() ? `
        <h3>עריכה</h3>
        <div class="acts">
          <button class="act" data-pub="rename"><span class="lbl">שינוי שם והערה</span></button>
          <button class="act danger" data-pub="remove"><span class="lbl">הסרה מהמסד
            <span class="hint">נשמר בהיסטוריה, אפשר לשחזר</span></span></button>
        </div>
        <p id="pub-msg" class="pub-msg" hidden></p>` : ''}`;
  }

  const photos = it.photos || [];
  const gallery = photos.length ? `
    <h3>תמונות (${photos.length})</h3>
    <div class="gallery">
      ${photos.map((p, i) => `<img src="${p.thumb}" data-i="${i}" alt="${escapeHtml(it.name)}" loading="lazy">`).join('')}
    </div>` : '';

  el('detail').innerHTML = `
    <h2>${escapeHtml(it.name)}</h2>
    <div class="chips">${chips.join('')}</div>
    ${it.note ? `<p class="note">${escapeHtml(it.note)}</p>` : ''}
    ${body}
    ${gallery}
    ${layer.kind === 'trails'
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

/** Editing a trail that is already in the shared dataset. */
function wirePublished(it) {
  el('detail').querySelectorAll('[data-pub]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const msg = el('pub-msg');
      const say = (text, bad) => {
        msg.hidden = false;
        msg.textContent = text;
        msg.className = 'pub-msg' + (bad ? ' bad' : '');
      };

      try {
        if (btn.dataset.pub === 'rename') {
          const name = prompt('שם השביל:', it.name);
          if (name == null) return;
          const note = prompt('הערה (אפשר להשאיר ריק):', it.note || '');
          if (note == null) return;
          btn.disabled = true;
          say('שומר…');
          await reloadShared(await Store.rename(it.id, name.trim() || it.name, note.trim()));
          select(it.id, false);
        } else {
          if (!confirm(`להסיר את "${it.name}" מהמסד המשותף?\n` +
                       'השינוי נשמר בהיסטוריה ואפשר לשחזר אותו.')) return;
          btn.disabled = true;
          say('מסיר…');
          await reloadShared(await Store.remove(it.id, it.name));
          deselect();
        }
      } catch (err) {
        btn.disabled = false;
        say('נכשל: ' + err.message, true);
      }
    });
  });
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
  const bits = s.segments
    ? [`${s.segments} מקטעים`, metres(s.length), `${s.waypoints} נקודות ציון`,
       `${s.photos} תמונות`]
    : ['אין שכבה דלוקה'];
  if (Store.offline) bits.push('לא מחובר, מציג עותק שמור');
  el('stats').textContent = bits.join(' · ');

  const who = Store.editor();
  el('editor-btn').textContent = who ? `עורך: ${who}` : 'מצב עורך';
  el('editor-btn').classList.toggle('on', !!who);
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

  const { trails, network } = await Store.load();
  DATA = trails;
  Layers.init(trails, network);
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
  Store.resume().then((who) => { if (who) repaint(); });

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
    const layer = Layers.byId('trails');
    layer.segments = trails.segments;
    layer.waypoints = trails.waypoints;
    Layers.refresh('trails');
  } catch (err) {
    console.error('refresh failed', err);
  }
}

/* ---------- editor mode ----------
 *
 * Reading needs nothing. Publishing needs a token, pasted once per device and
 * scoped to the data repo alone, which is the whole reason the data sits in a
 * repo of its own: a token that leaks off someone's phone can touch published
 * trail data and nothing else, and every change it makes is a commit.
 */

function editorSheet() {
  const who = Store.editor();
  el('editor-card').innerHTML = who ? `
    <header class="sheet-head">
      <h2>מצב עורך</h2>
      <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">מחובר בתור <b>${escapeHtml(who)}</b>. שביל שתפרסם ייכנס
      למסד המשותף מיד, וכל מי שיפתח את האפליקציה יראה אותו.</p>
    <button class="big-act" data-act="out"><b>התנתק מהמכשיר הזה</b>
      <span>המפתח יימחק מהדפדפן. הטיוטות המקומיות נשארות.</span></button>` : `
    <header class="sheet-head">
      <h2>מצב עורך</h2>
      <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
    </header>
    <p class="sheet-lead">כדי לפרסם שבילים למסד המשותף צריך מפתח אישי. זה חד פעמי
      למכשיר. בלי מפתח האפליקציה עובדת רגיל, והשבילים שתקליט נשארים אצלך ואפשר
      לשלוח אותם ליוזמה כקובץ.</p>
    <ol class="steps">
      <li><a href="${Store.TOKEN_HELP}" target="_blank" rel="noopener">יצירת מפתח בגיטהאב ↗</a></li>
      <li>מגבילים לריפו אחד בלבד:
        <code>Repository access › Only select repositories</code>
        <code>${Store.REPO}</code></li>
      <li>נותנים הרשאת כתיבה לתוכן:
        <code>Permissions › Contents › Read and write</code></li>
      <li>מייצרים, מעתיקים, ומדביקים כאן</li>
    </ol>
    <label class="fld"><span>המפתח</span>
      <input id="tok" type="password" autocomplete="off" spellcheck="false"
             placeholder="github_pat_…"></label>
    <p id="tok-err" class="tok-err" hidden></p>
    <button class="big-act primary" data-act="in"><b>התחבר</b></button>`;

  el('editor-sheet').hidden = false;
}

async function editorAction(act, btn) {
  if (act === 'close') { el('editor-sheet').hidden = true; return; }
  if (act === 'out') { Store.signOut(); editorSheet(); repaint(); return; }
  if (act !== 'in') return;

  const err = el('tok-err');
  err.hidden = true;
  btn.disabled = true;
  btn.querySelector('b').textContent = 'בודק…';
  try {
    await Store.signIn(el('tok').value);
    el('editor-sheet').hidden = true;
    repaint();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
    btn.disabled = false;
    btn.querySelector('b').textContent = 'התחבר';
  }
}

function wireControls() {
  el('editor-btn').addEventListener('click', editorSheet);
  el('editor-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'editor-sheet') { el('editor-sheet').hidden = true; return; }
    const btn = e.target.closest('[data-act]');
    if (btn) editorAction(btn.dataset.act, btn);
  });

  el('layers').addEventListener('click', Layers.openSheet);
  el('layer-sheet').addEventListener('click', (e) => {
    if (e.target.id === 'layer-sheet' || e.target.closest('[data-act="close"]')) {
      Layers.closeSheet();
    }
  });

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
    if (!selectedId || Drafts.isDrafting()) return;
    const hits = Layers.list
      .map((l) => `hit-${l.id}`)
      .filter((id) => map.getLayer(id));
    if (!map.queryRenderedFeatures(e.point, { layers: hits }).length) deselect();
  });

  wireGrip();
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
