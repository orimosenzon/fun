/* דרך הנדיב · מה קורה עכשיו
 *
 * Copyright (C) 2026 Ori Mosenzon
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version. It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License (the LICENSE file beside this one) for more details.
 *
 * One page over the festival's programme. The programme comes from the nightly
 * build of דרך קיצור (dereh_kitzur/build_hanadiv.py), which pulls the
 * festival's API and places every venue on the map; that file lands in the
 * derech-kitzur-data repo, so this page is as fresh as last night's run with
 * no deploy. A copy in data/ is the fallback when GitHub is unreachable.
 *
 * The whole interface hangs on one value, the playhead: a day and a minute.
 * The strip, the map, the list and the timetable all redraw from it.
 */
'use strict';

const DATA_URLS = [
  'https://raw.githubusercontent.com/orimosenzon/derech-kitzur-data/main/data/hanadiv.json',
  'data/hanadiv.json',
];
const CENTER = [34.978, 32.4735];
const SOON = 60;            // minutes ahead that count as "starting soon"
const LONG = 240;           // events this long are exhibitions, open all day
const WALK_M_PER_MIN = 75;  // with 1.3 for streets not being straight lines
const STEP = 5;

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const hm = m => { m = Math.round(m); const h = Math.floor(m / 60) % 24; return `${String(h).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`; };
const rng = (a, b) => `<span dir="ltr">${hm(a)}–${hm(b)}</span>`;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

const S = {
  events: [], venues: new Map(), groups: [], colors: {}, days: [],
  day: null, t: 600, lo: 480, hi: 1440,
  view: 'map', filt: new Set(), kids: false, access: false, q: '',
  plan: new Set(), onlyMine: true, me: null, sel: null, playing: null, moreLater: false, shared: null,
};

/* ---------- data ---------- */

function parseDur(s) {
  s = (s || '').trim();
  let m = 0;
  if (/^שעתיים/.test(s)) m = 120;
  else if (/^שעה/.test(s)) m = 60;
  const h = s.match(/^(\d+)\s*שעות/); if (h) m = +h[1] * 60;
  if (/וחצי/.test(s)) m += 30;
  const d = s.match(/(\d+)\s*דק/); if (d) m += +d[1];
  return m || 60;
}

function splitAddress(a) {
  const [street, ...rest] = String(a || '').split(' · ');
  let place = rest.join(' · ').replace(/["״]/g, '').trim();
  let st = street.replace(/\s+0$/, '').trim();
  if (st.length > 34) {  // "בווייז - האולם ... ולהמשיך לפי השילוט" is directions, not a street
    if (!place) place = st.split(/[.,]/)[0];
    st = '';
  }
  place = place.replace(/\.\s*בוו?ייז.*$/, '').replace(/\.$/, '');
  return { street: st, place: place || st, directions: street.length > 34 ? street.replace(/\s+0$/, '') : '' };
}

async function load() {
  for (const url of DATA_URLS) {
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 7000);
      const r = await fetch(url, { signal: ctl.signal });
      clearTimeout(timer);
      if (r.ok) return await r.json();
    } catch (e) { /* next */ }
  }
  throw new Error('no data');
}

function prepare(doc) {
  S.groups = doc.groups.map(g => g.name);
  doc.groups.forEach(g => S.colors[g.name] = g.color);
  S.festival = doc.festival;
  S.updated = doc.updated;
  for (const p of doc.places) {
    const [h, m] = p.hour.split(':').map(Number);
    const durLabel = (p.when.split('·').pop() || '').trim();
    const dur = parseDur(durLabel);
    const a = splitAddress(p.address);
    const vkey = (a.street || a.place).replace(/\s+/g, ' ');
    const e = {
      id: p.id.replace(/^hanadiv-/, ''), name: p.name.replace(/\s*·\s*יום (חמישי|שישי|שבת)\s*$/, ''),
      group: p.group, craft: p.craft, date: p.date, start: h * 60 + m, dur, durLabel,
      audience: p.audience, note: p.note, url: p.url, links: p.links || [], photos: p.photos || [],
      street: a.street, place: a.place, directions: a.directions, access: p.access || '', seats: p.seats, avail: p.avail || null, vkey,
      geo: p.geo ? { lat: p.geo.lat, lng: p.geo.lng } : null,
    };
    e.end = e.start + dur;
    e.long = dur >= LONG;
    e.c = S.colors[e.group] || '#607d8b';
    S.events.push(e);
    let v = S.venues.get(vkey);
    if (!v) S.venues.set(vkey, v = { key: vkey, street: a.street, place: a.place, events: [], pts: [] });
    v.events.push(e);
    if (e.geo) v.pts.push(e.geo);
  }
  for (const v of S.venues.values()) {
    v.events.sort((a, b) => a.date.localeCompare(b.date) || a.start - b.start);
    if (v.pts.length) v.geo = { lat: avg(v.pts.map(p => p.lat)), lng: avg(v.pts.map(p => p.lng)) };
    v.events.forEach(e => { e.venue = v; e.vgeo = v.geo || null; });
    // a complex (a school, a culture hall) runs several stages under one address
    v.stages = [...new Set(v.events.map(e => e.place))];
    v.title = v.stages.length > 1 ? v.street || v.place : v.place;
  }
  S.events.sort((a, b) => a.date.localeCompare(b.date) || a.start - b.start || a.name.localeCompare(b.name, 'he'));
  S.byId = new Map(S.events.map(e => [e.id, e]));
  S.days = [...new Set(S.events.map(e => e.date))].sort();
  S.lo = Math.floor(Math.min(...S.events.map(e => e.start)) / 60) * 60;
  S.hi = Math.min(26 * 60, Math.ceil(Math.max(...S.events.map(e => e.start + Math.min(e.dur, 120))) / 60) * 60);
}
const avg = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

/* ---------- helpers over the playhead ---------- */

function passes(e) {
  if (S.filt.size && !S.filt.has(e.group)) return false;
  if (S.kids && e.audience === 'רק למבוגרים') return false;
  if (S.access && !/(^|· )נגיש/.test(e.access)) return false;
  if (S.q && !qMatch(e)) return false;
  return true;
}

/* Search in Hebrew: "חייזרים" should find "החייזרי", and "לחמים" should find "לחם". Both sides
   lose their final letters and case; each word of the query also tries itself without a
   plural ending and without the little words glued to its front (ו, ה, ב, ל, מ, ש, כ). */
const FINALS = { 'ם': 'מ', 'ן': 'נ', 'ץ': 'צ', 'ף': 'פ', 'ך': 'כ' };
const qNorm = s => s.toLowerCase().replace(/[םןץףך]/g, c => FINALS[c]).replace(/["'״׳]/g, '');
function qForms(w) {
  const out = new Set([w]);
  for (const f of [...out]) { const s = f.replace(/^ו?[הבלמשכ]/, ''); if (s.length >= 4) out.add(s); }
  for (const f of [...out]) { const s = f.replace(/(ימ|ות|יות)$/, ''); if (s.length >= 3) out.add(s); }
  return [...out];
}
function qMatch(e) {
  const hay = e._hay || (e._hay = qNorm(`${e.name} ${e.craft} ${e.group} ${e.note} ${e.place} ${e.street}`));
  if (S._q !== S.q) { S._q = S.q; S._qw = qNorm(S.q).split(/\s+/).filter(Boolean).map(qForms); }
  return S._qw.every(forms => forms.some(f => hay.includes(f)));
}
// under the search box: how many, on which days, and that Enter goes to the first
function drawQInfo() {
  const box = $('#qinfo');
  const none = !!S.q && !S.events.some(passes);
  box.classList.toggle('none', none);
  $('#q').classList.toggle('none', none);
  if (!S.q) { box.hidden = true; return; }
  const hits = S.events.filter(passes);
  box.hidden = false;
  if (!hits.length) {
    // the other filters may be what hides it; say so, and offer the way out
    const other = S.filt.size || S.kids || S.access;
    box.innerHTML = `${ICON.search}<span><b>לא נמצא אף אירוע עם „${esc(S.q)}”</b>
      ${other ? 'גם הסינון לפי סוג פעיל.' : 'באף אחד משלושת הימים. אפשר לנסות מילה אחרת או קצרה יותר.'}</span>
      ${other ? '<button class="more" data-clearfilt>לחפש בלי הסינון</button>' : ''}`;
    return;
  }
  const per = S.days.map(d => [d, hits.filter(e => e.date === d).length]).filter(([, n]) => n);
  box.innerHTML = `<b>${hits.length === 1 ? 'אירוע אחד' : hits.length + ' אירועים'}</b>: ${per.map(([d, n]) => `${dayName(d, true)} ${n}`).join(' · ')}
    <button class="more" data-qgo>להראות ${hits.length === 1 ? 'אותו' : 'את הראשון'} ←</button>`;
}
// Enter, or the button: the first hit still to come (or the first at all), on the map
function qGo() {
  const hits = S.events.filter(passes).sort((a, b) => a.date.localeCompare(b.date) || a.start - b.start);
  if (!hits.length) {
    // Enter with nothing to show still answers: the box shakes and the line says why
    const q = $('#q'); q.classList.remove('shake'); void q.offsetWidth; q.classList.add('shake');
    return toast(`לא נמצא אירוע עם „${S.q}”`);
  }
  const n = israelNow(), e = hits.find(x => x.date > n.date || (x.date === n.date && x.end > n.m)) || hits[0];
  stopPlay(); S.follow = false;
  $('#q').blur();
  if (S.view !== 'map') setView('map');
  setDay(e.date); setT(e.start);
  focusEvent(e);
}
const dayEvents = () => S.events.filter(e => e.date === S.day && passes(e));
function status(e, t = S.t) {
  if (e.date !== S.day) return e.date < S.day ? 'past' : 'later';
  if (e.start <= t && t < e.end) return 'live';
  if (e.end <= t) return 'past';
  return e.start - t <= SOON ? 'soon' : 'later';
}
function dist(a, b) {
  const R = 6371000, toR = Math.PI / 180;
  const dLat = (b.lat - a.lat) * toR, dLng = (b.lng - a.lng) * toR;
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * toR) * Math.cos(b.lat * toR) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}
const walkMin = (a, b) => Math.max(1, Math.round(dist(a, b) * 1.3 / WALK_M_PER_MIN));
function mins(n) {
  n = Math.round(n);
  if (n < 60) return `${n} דק׳`;
  const h = Math.floor(n / 60), m = n % 60;
  const hs = h === 1 ? 'שעה' : h === 2 ? 'שעתיים' : `${h} שעות`;
  return m ? `${hs} ו-${m} דק׳` : hs;
}
function dayName(d, short) {
  const w = new Date(d + 'T12:00:00').toLocaleDateString('he-IL', { weekday: 'long' }).replace('יום ', '');
  const [, mo, da] = d.split('-');
  return short ? w : `${w} ${+da}.${+mo}`;
}
function israelNow() {
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Jerusalem', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(new Date()).map(p => [p.type, p.value]));
  let date = `${parts.year}-${parts.month}-${parts.day}`, m = +parts.hour * 60 + +parts.minute;
  if (m < 180) {  // the small hours belong to last night
    const d = new Date(date + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() - 1);
    date = d.toISOString().slice(0, 10); m += 1440;
  }
  return { date, m };
}
const festivalNow = () => { const n = israelNow(); return S.days.includes(n.date) ? n : null; };
/* "Now" is said only of the real present: the festival is on today, and the playhead stands
   on today at the clock's time. Anywhere else the app is a look ahead (or back), and every
   label speaks of the chosen time instead, so nothing reads as happening when it is not. */
function isNow() {
  const n = festivalNow();
  return !!n && n.date === S.day && Math.abs(n.m - S.t) <= STEP;
}
function relLabel(e) {
  const st = status(e), now = isNow();
  if (st === 'live') return e.long ? { cls: now ? 'live' : 'at', txt: `פתוח עד ${hm(e.end)}` }
    : now ? { cls: 'live', txt: `קורה עכשיו · עוד ${mins(e.end - S.t)}` } : { cls: 'at', txt: `ב-${hm(S.t)} באמצע · עד ${hm(e.end)}` };
  if (st === 'soon') return now ? { cls: 'soon', txt: e.start === S.t ? 'מתחיל עכשיו' : `בעוד ${mins(e.start - S.t)}` }
    : { cls: 'soon', txt: e.start === S.t ? `מתחיל ב-${hm(S.t)}` : `${mins(e.start - S.t)} אחרי ${hm(S.t)}` };
  if (st === 'past') return { cls: '', txt: 'הסתיים' };
  return null;
}

/* ---------- the strip ---------- */

const PAD = 16;  // room for the first and last hour labels
function xOf(t, W) { return W - PAD - (t - S.lo) / (S.hi - S.lo) * (W - 2 * PAD); }  // right to left: morning on the right
function tOf(x, W) { return S.lo + (W - PAD - x) / (W - 2 * PAD) * (S.hi - S.lo); }

function drawStrip() {
  const svg = $('#stripSvg');
  const W = svg.clientWidth, H = svg.clientHeight, base = H - 14;
  if (!H) return;  // put away by the viewer; it draws again when it comes back
  const evs = dayEvents().filter(e => !e.long);
  const BIN = 10, bins = [];
  let peak = 1;
  for (let t = S.lo; t < S.hi; t += BIN) {
    const per = {};
    let n = 0;
    for (const e of evs) if (e.start < t + BIN && e.end > t) { per[e.group] = (per[e.group] || 0) + 1; n++; }
    bins.push({ t, per, n }); peak = Math.max(peak, n);
  }
  const bw = (W - 2 * PAD) / bins.length;
  let out = '';
  for (let h = S.lo; h <= S.hi; h += 60) {
    const x = xOf(h, W);
    out += `<line x1="${x}" x2="${x}" y1="0" y2="${base}" stroke="#000" stroke-opacity=".06"/>`;
    if ((h / 60) % 2 === 0 || W > 700) out += `<text x="${x}" y="${H - 2}" text-anchor="middle">${hm(h)}</text>`;
  }
  for (const b of bins) {
    let y = base;
    const x = xOf(b.t + BIN, W);
    for (const g of S.groups) {
      const n = b.per[g]; if (!n) continue;
      const h = n / peak * (base - 4);
      y -= h;
      out += `<rect x="${x + .5}" y="${y}" width="${Math.max(bw - 1, 1)}" height="${h}" fill="${S.colors[g]}" rx="1"/>`;
    }
  }
  out += `<line x1="${PAD}" x2="${W - PAD}" y1="${base}" y2="${base}" stroke="#000" stroke-opacity=".15"/>`;
  // my plan rides on top of the strip, so dragging the clock shows when I am busy
  for (const e of planList().filter(e => e.date === S.day && !e.long)) {
    const x1 = xOf(Math.min(e.end, S.hi), W), x2 = xOf(e.start, W);
    out += `<rect class="mineband" x="${x1}" y="${base - 7}" width="${Math.max(x2 - x1, 5)}" height="7" rx="3"/>`;
  }
  svg.innerHTML = out;
  placePlayhead();
}
function placePlayhead() {
  const W = $('#stripSvg').clientWidth;
  $('#playhead').style.left = xOf(S.t, W) + 'px';
  // looking at another time of today, the real present stays marked on the strip
  const n = festivalNow(), tick = $('#realTick');
  tick.hidden = !n || n.date !== S.day || isNow() || n.m < S.lo || n.m > S.hi;
  if (!tick.hidden) tick.style.left = xOf(n.m, W) + 'px';
}
// under the chosen time, when it is not the present: what the real time is, and a way back to it
function realNowHTML() {
  const r = israelNow(), n = festivalNow();
  let where;
  if (n) where = n.date === S.day ? 'היום' : `היום ${dayName(n.date, true)}`;
  else if (r.date < S.days[0]) {
    const d = Math.round((new Date(S.days[0]) - new Date(r.date)) / 864e5);
    where = `הפסטיבל מתחיל ${d === 1 ? 'מחר' : `בעוד ${d} ימים`}`;
  } else where = 'הפסטיבל כבר נגמר';
  return `<button data-gonow>השעה עכשיו ${hm(r.m % 1440)} · ${where}</button>`;
}

function setupStrip() {
  const strip = $('#strip');
  let dragging = false;
  const fromEvent = ev => {
    const r = strip.getBoundingClientRect();
    const t = tOf(clamp(ev.clientX - r.left, 0, r.width), r.width);
    setT(Math.round(t / STEP) * STEP);
  };
  strip.addEventListener('pointerdown', ev => { letGo(); dragging = true; strip.setPointerCapture(ev.pointerId); stopPlay(); fromEvent(ev); });
  strip.addEventListener('pointermove', ev => dragging && fromEvent(ev));
  strip.addEventListener('pointerup', () => { dragging = false; });
  strip.addEventListener('keydown', ev => {
    const d = { ArrowLeft: STEP, ArrowRight: -STEP, PageUp: 60, PageDown: -60 }[ev.key];
    if (d) { ev.preventDefault(); letGo(); stopPlay(); setT(S.t + d); }
  });
  new ResizeObserver(() => drawStrip()).observe(strip);
}

/* ---------- time of day: the map darkens towards evening ---------- */

function sky() {
  const n = clamp((S.t - 16.5 * 60) / 100, 0, 1);       // sunset is about 16:50 at the end of October
  const dusk = clamp(1 - Math.abs(S.t - 17 * 60) / 50, 0, 1);
  if (S.shade) S.shade.style.background = `rgba(${Math.round(40 - 20 * n)},${Math.round(45 - 10 * n)},${Math.round(70 + 20 * n)},${(.42 * n).toFixed(3)})`;
  if (S.glow) S.glow.style.background = `rgba(255,140,60,${(.18 * dusk).toFixed(3)})`;
  // a blended layer costs a full-screen blend every frame even when it is clear, so by day it is gone
  if (S.shade) S.shade.style.display = n ? '' : 'none';
  if (S.glow) S.glow.style.display = dusk ? '' : 'none';
  const mix = (a, b, k) => a.map((x, i) => Math.round(x + (b[i] - x) * k));
  const day = [253, 243, 223], eve = [251, 226, 205], night = [226, 226, 240];
  const c = S.t < 17 * 60 ? mix(day, eve, dusk) : mix(eve, night, n);
  document.documentElement.style.setProperty('--sky', `rgb(${c})`);
}

/* ---------- map ---------- */

/* Two looks for the map, as in דרך קיצור: the aerial photograph (Esri's, free and keyless) laid
   in under the street names, and a tilted view over the relief. The relief across the moshava
   is only some 46 m, so it is exaggerated to read as anything. Both are remembered. */
const SAT = {
  type: 'raster', tileSize: 256, maxzoom: 18,  // Esri has no zoom 19 here, so 18 is stretched
  tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
  attribution: 'Esri, Maxar, Earthstar Geographics',
};
const DEM = {
  type: 'raster-dem', tileSize: 256, maxzoom: 14, encoding: 'terrarium',
  tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
  attribution: 'Elevation: Mapzen / AWS Open Data',
};
const TILTED = 58, TERRAIN_X = 2.5;
S.sat = false; S.tilt = false;  // set from the address in main()

let map;
const markers = new Map();
function setupMap() {
  // without it Hebrew street names come out mirrored
  try {
    const pending = maplibregl.setRTLTextPlugin('vendor/mapbox-gl-rtl-text.min.js', false);
    if (pending && pending.catch) pending.catch(() => {});
  } catch (e) { /* mirrored labels are bad, a blank map is worse */ }
  map = new maplibregl.Map({
    container: 'map', style: 'https://tiles.openfreemap.org/styles/liberty',
    center: S.cam ? [S.cam.lng, S.cam.lat] : CENTER, zoom: S.cam ? S.cam.zoom : 13.4,
    bearing: S.cam ? S.cam.bearing : 0, pitch: S.cam ? S.cam.pitch : S.tilt ? TILTED : 0, maxPitch: 65,
    // a phone's screen of 3 pixels to the point would draw 9 for each one seen, over the relief twice
    pixelRatio: Math.min(devicePixelRatio || 1, 2),
    attributionControl: { compact: true, customAttribution: `אורי מוסנזון · <a href="${SRC}" target="_blank" rel="noopener">GPL-3.0</a>` },
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');
  map.addControl({
    onAdd() {
      const box = document.createElement('div');
      box.className = 'maplibregl-ctrl maplibregl-ctrl-group lookctl';
      box.innerHTML = `<button type="button" id="satBtn" title="צילום אוויר" aria-label="צילום אוויר">${ICON.sat}</button>
        <button type="button" id="tiltBtn" title="מבט בהטיה, תלת־ממד" aria-label="מבט בהטיה">3D</button>
        <button type="button" id="showBtn" title="מה להציג" aria-label="מה להציג">${ICON.sliders}</button>
        <button type="button" id="cleanBtn" title="מפה בלבד" aria-label="מפה בלבד, מסך מלא">${ICON.full}</button>`;
      box.querySelector('#satBtn').onclick = () => { S.sat = !S.sat; applyLook(); };
      box.querySelector('#showBtn').onclick = ev => { ev.stopPropagation(); toggleShowPanel(); };
      box.querySelector('#cleanBtn').onclick = () => setClean(!S.clean);
      box.querySelector('#tiltBtn').onclick = () => {
        S.tilt = !S.tilt; applyLook();
        map.easeTo({ pitch: S.tilt ? TILTED : 0, bearing: S.tilt ? map.getBearing() : 0, duration: 700 });
      };
      return box;
    },
    onRemove() {},
  }, 'top-right');
  const cc = map.getCanvasContainer();
  for (const k of ['shade', 'glow']) {
    const d = document.createElement('div');
    d.style.cssText = 'position:absolute;inset:0;pointer-events:none;mix-blend-mode:multiply;transition:background .4s';
    cc.appendChild(d); S[k] = d;
  }
  S.glow.style.mixBlendMode = 'soft-light';
  for (const v of S.venues.values()) {
    if (!v.geo) continue;
    const el = document.createElement('div');
    el.className = 'vm hide';
    el.innerHTML = '<div class="dot"><i class="img"></i><span class="lbl"></span></div>';
    el.addEventListener('click', ev => { ev.stopPropagation(); S.focus = null; openVenue(v); markFocus(); });
    const mk = new maplibregl.Marker({ element: el, anchor: 'center' }).setLngLat([v.geo.lng, v.geo.lat]).addTo(map);
    // Over the relief MapLibre asks the graphics card, picture by picture and several times a
    // second, whether a hill hides it. Each ask stalls the card until it answers, and with 90
    // pictures that was most of why the 3D view crawled. The moshava has no hill to hide behind.
    mk._updateOpacity = () => {};
    markers.set(v.key, { el, mk, v });
  }
  map.on('load', () => {
    // the photograph goes in under the labels and the 3D buildings, so street names stay on top
    map.addSource('sat', SAT);
    map.addSource('dem', DEM);
    const before = (map.getStyle().layers.find(l => l.type === 'symbol' || l.type === 'fill-extrusion') || {}).id;
    map.addLayer({ id: 'sat', type: 'raster', source: 'sat', layout: { visibility: 'none' }, paint: { 'raster-fade-duration': 0 } }, before);
    applyLook(); applyLabels();
    // tilting with two fingers (or ctrl-drag) counts as asking for the 3D view
    map.on('sourcedata', ev => { if (ev.sourceId === 'dem' && !map.isMoving()) groundCamera(); });
    map.on('moveend', groundCamera);
    map.on('pitchend', () => { const t = map.getPitch() >= 10; if (t !== S.tilt) { S.tilt = t; applyLook(); } });
    map.addSource('route', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    map.addLayer({ id: 'route-case', type: 'line', source: 'route', paint: { 'line-color': '#fff', 'line-width': 7, 'line-opacity': .9 }, layout: { 'line-cap': 'round', 'line-join': 'round' } });
    map.addLayer({ id: 'route', type: 'line', source: 'route', paint: { 'line-color': '#1d2b3a', 'line-width': 3.5, 'line-dasharray': [1.5, 1.2] }, layout: { 'line-cap': 'round', 'line-join': 'round' } });
    // arrowheads along the line, pointing from each event to the next one
    map.addImage('route-arrow', arrowImage(), { pixelRatio: 2 });
    map.addLayer({ id: 'route-arrows', type: 'symbol', source: 'route', layout: {
      'symbol-placement': 'line', 'symbol-spacing': 110, 'icon-image': 'route-arrow',
      'icon-rotation-alignment': 'map', 'icon-allow-overlap': true, 'icon-ignore-placement': true } });
    drawRoute();
    // frame the moshava: venues further than 3 km (the outskirts, Caesarea's beach) are left out of the opening view
    const near = [...S.venues.values()].filter(v => v.geo && dist(v.geo, { lat: CENTER[1], lng: CENTER[0] }) < (innerWidth < 600 ? 1600 : 3000));
    if (near.length && !S.sel && !S.cam) {
      const b = new maplibregl.LngLatBounds();
      near.forEach(v => b.extend([v.geo.lng, v.geo.lat]));
      map.fitBounds(b, { padding: fitPad(), duration: 0, maxZoom: 15 });
    }
    // a link can carry an open popup on the map, with the event it was showing
    const at = S.init && S.init.at && S.venues.get(S.init.at);
    if (at && at.geo && S.view !== 'grid') {
      const f = S.init.f && S.byId.get(S.init.f);
      S.focus = f && f.vkey === at.key ? f : null;
      openVenue(at, false); markFocus();
    }
    S.camReady = true;
    map.on('moveend', setHashSoon);
    setHashSoon();
  });
  map.on('click', () => closePopup());
  map.on('movestart', () => map.getContainer().classList.add('moving'));
  map.on('moveend', () => map.getContainer().classList.remove('moving'));
  // The pictures shrink as the map zooms out, so the whole moshava stays readable on a phone.
  // Zoomed right in they take on a real size on the ground, about a quarter of a house,
  // and keep growing with the map until the picture is plain to see.
  const scale = () => {
    const z = map.getZoom();
    const mpp = 156543 * Math.cos(map.getCenter().lat * Math.PI / 180) / 2 ** z / 2;  // metres per css pixel (512px tiles)
    const k = Math.min(Math.max(clamp(.62 + (z - 12.6) * .2, .6, 1.15), GROUND_M / mpp / 34), 7);
    const c = map.getContainer();
    c.style.setProperty('--k', k.toFixed(3));
    c.classList.toggle('close', k > 1.6);
    if (S.popup && S.sel) S.popup.setOffset(popOffset(S.sel));
  };
  map.on('zoom', scale); scale();
}
function applyLook() {
  setHashSoon();
  $('#satBtn')?.classList.toggle('on', S.sat);
  $('#tiltBtn')?.classList.toggle('on', S.tilt);
  document.body.classList.toggle('sat', S.sat);
  if (!map || !map.getLayer('sat')) return;
  map.setLayoutProperty('sat', 'visibility', S.sat ? 'visible' : 'none');
  // grey boxes standing on a photograph of the same roofs are noise
  for (const l of map.getStyle().layers) if (l.type === 'fill-extrusion') map.setLayoutProperty(l.id, 'visibility', S.sat ? 'none' : 'visible');
  map.setTerrain(S.tilt ? { source: 'dem', exaggeration: TERRAIN_X } : null);
  groundCamera();
}
/* MapLibre keeps the camera's ground at sea level until the first drag. The moshava stands
   about 70 m up, 170 m once the relief is exaggerated, so zoomed in close the tilted camera
   sat inside the hill: a grey blur, the pictures faded out, and the first drag then jumped
   four zoom levels in. Lifting the ground under the camera as soon as the heights arrive
   keeps the zoom where it was. */
function groundCamera() {
  if (!map || !map.terrain) return;
  const h = map.terrain.getElevationForLngLatZoom(map.getCenter(), map.getZoom());
  if (Math.abs(h - map.transform.elevation) > 1) { map.transform.elevation = h; map.triggerRepaint(); }
}

/* ---------- how much is on the screen ---------- */
/* Each viewer picks what stays on screen: the hour strip, the kind chips, the list, and on
   the map the places already over, the ones later in the day, and the street names. "Map
   only" clears everything but the map and a small clock, full screen where the browser
   allows it (not on an iPhone). The choices live in the address, like the rest of the state:
   the plain address shows everything, and a link brings back exactly what was on screen. */
const UI_DEF = { strip: true, chips: true, list: true, done: true, idle: true, labels: true };
S.ui = { ...UI_DEF };
S.clean = false;
const SHOW_ROWS = [
  ['על המסך', [['strip', 'סרגל השעות'], ['chips', 'סינון לפי סוג'], ['list', 'רשימת האירועים ליד המפה']]],
  ['על המפה', [['idle', 'מה שמתחיל מאוחר יותר היום'], ['done', 'מה שכבר נגמר'], ['labels', 'שמות רחובות ומקומות']]],
];
function showPanelHTML() {
  const sw = ([k, label]) => `<label class="sw"><span>${label}</span><input type="checkbox" data-ui="${k}" ${S.ui[k] ? 'checked' : ''}><i></i></label>`;
  return `<div class="sphead"><b>מה להציג</b><button class="iconbtn" data-showclose aria-label="סגירה">${ICON.x}</button></div>
    <button class="btn cleango" data-clean>${ICON.full}מפה בלבד, מסך מלא</button>
    ${SHOW_ROWS.map(([h, rows]) => `<div class="spsec"><h5>${h}</h5>${rows.map(sw).join('')}</div>`).join('')}
    ${Object.keys(UI_DEF).some(k => S.ui[k] !== UI_DEF[k]) ? '<button class="more" data-uireset>להחזיר הכל</button>' : ''}`;
}
function toggleShowPanel(open = $('#showPanel').hidden) {
  const p = $('#showPanel');
  p.hidden = !open;
  $('#showBtn')?.classList.toggle('on', open);
  if (open) p.innerHTML = showPanelHTML();
}
function applyUI() {
  layoutUI();
  applyLabels();
  if (map) map.resize();
  render();
  if (S.popup) requestAnimationFrame(fitPopup);
}
function layoutUI() {
  const R = document.documentElement.classList;
  R.toggle('clean', S.clean);
  R.toggle('nostrip', !S.ui.strip);
  R.toggle('nochips', !S.ui.chips);
  // the plan lives in the list, so asking for the plan brings the list back
  R.toggle('nolist', !S.ui.list && S.view === 'map');
  $('#sheet').hidden = S.clean || (!S.ui.list && S.view === 'map');
  const fold = $('#sheetFold'), off = !S.ui.list;
  fold.classList.toggle('off', off);
  fold.title = off ? 'הצגת רשימת האירועים' : 'הסתרת רשימת האירועים';
  fold.setAttribute('aria-label', fold.title);
  $('#cleanBtn')?.classList.toggle('on', S.clean);
  if ($('#cleanBtn')) $('#cleanBtn').innerHTML = S.clean ? ICON.shrink : ICON.full;
}
function applyLabels() {
  if (!map || !map.isStyleLoaded()) return;
  for (const l of map.getStyle().layers) if (l.type === 'symbol') map.setLayoutProperty(l.id, 'visibility', S.ui.labels ? 'visible' : 'none');
}
function setClean(on) {
  if (on === S.clean) return;
  S.clean = on;
  if (on) {
    toggleShowPanel(false);
    if (S.view !== 'map') setView('map');
    const d = document.documentElement;
    if (d.requestFullscreen && !document.fullscreenElement) d.requestFullscreen().then(() => { S.fs = true; }, () => {});
  } else if (document.fullscreenElement) { S.fs = false; document.exitFullscreen().catch(() => {}); }
  applyUI();
}
// leaving the browser's full screen (the back gesture, Esc) leaves map-only too
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement && S.fs) { S.fs = false; setClean(false); } });

// flyTo's padding would stay on the map and skew every later move, so a fly takes the
// same padding as a one-off offset of the centre instead
const GROUND_M = 12;  // a picture's width on the ground, once zoomed in close
// the popup clears the top of its place's picture, however big the picture has grown
function popOffset(v) {
  const m = markers.get(v.key);
  return m && m.el.firstChild.offsetHeight ? m.el.firstChild.offsetHeight / 2 + 8 : 22;
}
const off = p => [(p.left - p.right) / 2, (p.top - p.bottom) / 2];
function fitPad() {
  const wide = innerWidth >= 900;
  if (wide || $('#sheet').hidden) return { top: 40, bottom: 40, left: 40, right: 40 };
  // on a phone, frame what is left above the sheet at the height it is snapping to
  const H = $('#stage').clientHeight, snapTo = $('#sheet').dataset.snap;
  const sheet = snapTo === 'peek' ? 128 : snapTo === 'full' ? H * .92 : H * .5;
  return { top: 30, bottom: Math.min(sheet + 30, H - 120), left: 30, right: 30 };
}

function drawMarkers() {
  const planDay = S.view === 'plan' ? planList().filter(e => e.date === S.day) : null;
  const planVenues = new Map();
  if (planDay) planDay.forEach((e, i) => { if (!planVenues.has(e.vkey)) planVenues.set(e.vkey, i + 1); });
  // a place shows the picture of the event that matters at the playhead: what is on, else what
  // starts next. Among those, one of mine wins, so the map shows my evening first.
  const pick = list => list.find(e => S.plan.has(e.id)) || list[0];
  for (const { el, v } of markers.values()) {
    const evs = v.events.filter(e => e.date === S.day && passes(e));
    const dot = el.firstChild;
    let cls = 'hide', txt = '', rep = null;
    if (evs.length) {
      const live = evs.filter(e => status(e) === 'live');
      const liveShort = live.filter(e => !e.long);
      const soon = evs.filter(e => status(e) === 'soon');
      const later = evs.filter(e => status(e) === 'later');
      if (live.length) {
        cls = 'live' + (liveShort.length > 1 ? ' multi' : '');
        rep = pick(liveShort.length ? liveShort : live);
        txt = liveShort.length > 1 ? liveShort.length : '';
      } else if (soon.length) {
        cls = 'soon'; rep = pick(soon); txt = `${rep.start - S.t}′`;
      } else if (later.length) { cls = 'idle'; rep = pick(later); }
      else { cls = 'done'; rep = evs.at(-1); }
      // the viewer may clear away what is not on now; a place in my plan stays
      if ((cls === 'idle' && !S.ui.idle || cls === 'done' && !S.ui.done) && !evs.some(e => S.plan.has(e.id) && status(e) !== 'past')) cls = 'hide';
    }
    // the plan adds a glow and a star; "only mine" leaves just the planned places
    const mine = evs.filter(e => S.plan.has(e.id));
    const myNow = mine.filter(e => status(e) !== 'past');
    if (myNow.length) cls += ' plan';
    if (onlyMine()) {
      if (!mine.length) cls = 'hide';
      else {
        cls = cls.replace(/\b(idle|done|hide)\b/, '') + ' mine';
        if (!/live|soon/.test(cls)) { rep = myNow[0] || mine.at(-1); if (!myNow.length) cls += ' over'; }
      }
    }
    if (S.sel && S.sel.key === v.key) cls += ' sel';
    if (S.focus && S.focus.vkey === v.key) rep = S.focus;
    el.className = 'vm ' + cls;
    el.style.setProperty('--c', rep ? rep.c : '#9aa3ab');
    const img = rep && rep.photos[0] ? rep.photos[0].thumb : '';
    if (el.dataset.img !== img) { el.dataset.img = img; dot.firstChild.style.backgroundImage = img ? `url("${img}")` : ''; }
    el.title = rep ? rep.name : '';
    const lbl = dot.lastChild;
    if (lbl.textContent !== String(txt)) lbl.textContent = txt;
    let badge = el.querySelector('.planno');
    const no = planDay && planVenues.get(v.key);
    if (no) { if (!badge) { badge = document.createElement('div'); badge.className = 'planno'; el.appendChild(badge); } badge.textContent = no; }
    else if (badge) badge.remove();
  }
  if (S.focus) {
    const m = markers.get(S.focus.vkey);
    if (m) {
      m.el.classList.add('focus');
      m.el.classList.remove('hide', 'done');
    }
  }
  if (S.popup && S.sel) S.popup.setHTML(S.focus ? eventPopHTML(S.focus) : venueHTML(S.sel));
}

function eventPopHTML(e) {
  const r = e.date === S.day ? relLabel(e) : null;
  const more = e.venue.events.filter(o => o.date === S.day && o.id !== e.id).length;
  const ph = e.photos[0] ? `<div class="pph" style="background-image:url('${esc(e.photos[0].thumb)}')"></div>` : '';
  return `<div class="pop epop ${isMine(e) ? 'mine' : ''}" style="--c:${e.c}">${ph}${isMine(e) ? '<div class="myflag">★ בתוכנית שלך</div>' : ''}
    <div class="kind">${esc(e.group)}${e.craft && e.craft !== e.group ? ' · ' + esc(e.craft) : ''}</div>
    <h4>${esc(e.name)}</h4>
    <div class="addr">${dayName(e.date, true)} ${rng(e.start, e.end)}${r ? ` <span class="tag ${r.cls}">${r.txt}</span>` : ''}</div>
    <div class="addr">${esc(e.place)}${e.street && e.street !== e.place ? ' · ' + esc(e.street) : ''}</div>
    <div class="pact"><button class="btn" data-detail="${e.id}">לפרטים</button>
      <button class="btn ghost ${S.plan.has(e.id) ? 'on' : ''}" data-star="${e.id}">${STAR}</button>
      <a class="btn ghost" target="_blank" rel="noopener" href="https://waze.com/ul?ll=${e.vgeo.lat},${e.vgeo.lng}&navigate=yes">${ICON.nav}</a>
      ${waLinks(e).slice(0, 1).map(l => `<a class="btn wa" target="_blank" rel="noopener" href="${esc(l.url)}" title="${esc(l.title)}">${ICON.wa}</a>`).join('')}</div>
    <div class="pmore">${more ? `<button class="more" data-venuelist>עוד ${more} כאן היום</button>` : ''}<button class="more" data-place="${esc(e.vkey)}">${ICON.list}לוח המקום</button></div></div>`;
}

/* A pick in the list (or in a popup) flies to the place, lights it up and
   shows the event there; the full card is one more tap. */
function focusEvent(e) {
  if (!e.vgeo) return openDetail(e);
  if (S.view === 'grid') setView('map');
  if (e.date !== S.day) setDay(e.date);
  const keep = S.popup && S.sel === e.venue;
  S.focus = e;
  if (!keep) openVenue(e.venue, false);
  else S.popup.setHTML(eventPopHTML(e));
  const wide = innerWidth >= 900;
  if (!wide) snap('peek');
  // the popup sits above the place (and on a phone the sheet below it), so the place goes low on the screen
  const pad = { top: Math.min(wide ? 330 : 300, map.getContainer().clientHeight * .58), bottom: wide ? 20 : 140, left: 10, right: 10 };
  map.flyTo({ center: [e.vgeo.lng, e.vgeo.lat], zoom: Math.max(map.getZoom(), 15), offset: off(pad), duration: 700 });
  map.once('moveend', fitPopup);
  drawMarkers(); markFocus(true); setHashSoon();
}
function markFocus(scroll) {
  $$('#sheetBody .ev').forEach(c => c.classList.toggle('focus', !!S.focus && c.dataset.ev === S.focus.id));
  if (scroll && innerWidth >= 900) {
    const c = $(`#sheetBody .ev.focus`);
    if (c) c.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

let meMarker;
function drawMe() {
  if (!S.me || !map) return;
  if (!meMarker) {
    const el = document.createElement('div'); el.className = 'medot';
    meMarker = new maplibregl.Marker({ element: el }).setLngLat([S.me.lng, S.me.lat]).addTo(map);
  } else meMarker.setLngLat([S.me.lng, S.me.lat]);
}

function venueHTML(v) {
  const evs = v.events.filter(e => e.date === S.day);
  const W = S.hi - S.lo;
  const bars = evs.map(e => `<b style="--c:${e.c};right:${(e.start - S.lo) / W * 100}%;width:${Math.min(e.dur, S.hi - e.start) / W * 100}%"></b>`).join('');
  const now = `<i style="right:${(S.t - S.lo) / W * 100}%"></i>`;
  const items = evs.map(e => {
    const st = status(e);
    return `<li class="${st} ${isMine(e) ? 'mine' : ''}" data-ev="${e.id}"><span class="h">${hm(e.start)}</span><span class="n" style="color:${e.c}">●</span><span class="n">${esc(e.name)}</span>${isMine(e) ? '<span class="ms">★</span>' : ''}</li>`;
  }).join('') || `<li>אין אירועים ב${dayName(S.day, true)}. ${v.events.length} בימים אחרים.</li>`;
  return `<div class="pop"><h4>${esc(v.title)}</h4><div class="addr">${esc(v.stages.length > 1 ? v.stages.length + ' במות' : v.street)}</div>
    ${evs.length ? `<div class="mini">${bars}${now}</div>` : ''}<ul>${items}</ul>
    <div class="pmore"><button class="more" data-place="${esc(v.key)}">${ICON.list}לוח הזמנים של המקום · ${v.events.length}</button></div></div>`;
}
function openVenue(v, fly) {
  const f = S.focus && S.focus.vkey === v.key ? S.focus : null;
  closePopup();
  S.focus = f;
  S.sel = v;
  if (!v.geo) return;
  S.popup = new maplibregl.Popup({ closeButton: false, maxWidth: '300px', offset: popOffset(v), focusAfterOpen: false, anchor: 'bottom' })
    .setLngLat([v.geo.lng, v.geo.lat]).setHTML(S.focus ? eventPopHTML(S.focus) : venueHTML(v)).addTo(map);
  S.popup.on('close', () => { if (S.sel === v) { S.sel = null; S.focus = null; drawMarkers(); markFocus(); setHashSoon(); } });
  if (fly) { map.flyTo({ center: [v.geo.lng, v.geo.lat], zoom: Math.max(map.getZoom(), 15.5), offset: off(fitPad()) }); map.once('moveend', fitPopup); }
  else requestAnimationFrame(fitPopup);
  if ($('#sheet').dataset.snap === 'full') snap('peek');
  drawMarkers(); setHashSoon();
}
/* A popup opened near the edge of the map would run off it (or hide under the sheet on a
   phone), so the map slides until all of it is in sight. When the map is too short to hold
   it, the popup rises over the header instead, which it is drawn above. */
function fitPopup() {
  if (!S.popup || !map) return;
  const el = S.popup.getElement(); if (!el) return;
  const r = el.getBoundingClientRect(), m = map.getContainer().getBoundingClientRect();
  const sheet = innerWidth < 900 && !$('#sheet').hidden ? $('#sheet').getBoundingClientRect().top : m.bottom;
  const bottom = Math.min(m.bottom, sheet) - 10, left = m.left + 10, right = m.right - 10;
  const top = r.height <= bottom - m.top - 10 ? m.top + 10 : 8;
  let dx = 0, dy = 0;
  if (r.top < top) dy = r.top - top;
  else if (r.bottom > bottom) dy = Math.min(r.bottom - bottom, r.top - top);
  if (r.left < left) dx = r.left - left;
  else if (r.right > right) dx = r.right - right;
  if (dx || dy) map.panBy([dx, dy], { duration: 350 });
}
function closePopup() { if (S.popup) { const p = S.popup; S.popup = null; p.remove(); } S.sel = null; S.focus = null; setHashSoon(); }

/* ---------- the list ---------- */

function card(e, opts = {}) {
  const r = relLabel(e);
  const st = status(e);
  const ph = e.photos[0] ? `background-image:url('${esc(e.photos[0].thumb)}')` : '';
  const walk = S.me && e.vgeo ? `<span class="tag walk">${walkMin(S.me, e.vgeo)} דק׳ הליכה</span>` : '';
  const kid = e.audience !== 'רק למבוגרים' ? `<span class="tag kid">${e.audience === 'רק לילדים' ? 'לילדים' : 'למשפחה'}</span>` : '';
  const prog = st === 'live' && !e.long ? `<div class="prog" style="width:${(S.t - e.start) / e.dur * 100}%"></div>` : '';
  const day = opts.showDay ? `${dayName(e.date, true)} · ` : '';
  const clash = !opts.noClash && S.plan.size ? clashWith(e) : null;
  const cl = clash ? `<span class="tag clash" title="${esc(clash.name)}">חופף לתוכנית</span>` : '';
  return `<div class="ev ${opts.dim && st === 'past' ? 'past' : ''} ${isMine(e) ? 'mine' : ''}" data-ev="${e.id}" style="--c:${e.c}">
    <div class="ph" style="${ph}"></div>
    <div class="body"><div class="nm">${esc(e.name)}</div>
      <div class="where">${esc(e.place)}${e.street && e.street !== e.place ? ' · ' + esc(e.street) : ''}${e.vgeo ? '' : ' · לא על המפה'}</div>
      <div class="when"><span class="hrs">${day}${rng(e.start, e.end)}</span>${r && !opts.noRel ? `<span class="tag ${r.cls}">${r.txt}</span>` : ''}${walk}${kid}${cl}${availTag(e)}${opts.join && !isOver(e) ? `<a class="tag join" data-join href="${esc(joinUrl(e))}" target="_blank" rel="noopener">להרשמה ↗</a>` : ''}</div></div>
    <button class="star ${S.plan.has(e.id) ? 'on' : ''}" data-star="${e.id}" aria-label="לתוכנית שלי">${STAR}</button>${prog}</div>`;
}
// The festival site opens its own sign-up for ?join=1 (added for us by its developer on 27.10.2026).
// Login and registration stay there; the map never touches users or seats.
const isOver = (e, n = israelNow()) => e.date < n.date || (e.date === n.date && e.end <= n.m);
const joinUrl = e => e.url + (e.url.includes('?') ? '&' : '?') + 'join=1';
const STAR = '<svg viewBox="0 0 24 24"><path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/></svg>';

function byNear(a, b) {
  if (!S.me) return 0;
  return (a.vgeo ? dist(S.me, a.vgeo) : 1e9) - (b.vgeo ? dist(S.me, b.vgeo) : 1e9);
}

function drawList() { $('#sheetBody').innerHTML = (S.q ? searchHTML() : myNextHTML() + listHTML()); markFocus(); }
// While searching, the list is the results: every hit on all three days, in time order,
// so a hit on Saturday shows up even when the clock is on Thursday.
function searchHTML() {
  const hits = S.events.filter(passes).sort((a, b) => a.date.localeCompare(b.date) || a.start - b.start);
  if (!hits.length) return `<div class="empty">לא נמצא אף אירוע עם „${esc(S.q)}”. <button class="more" data-qclear>לנקות את החיפוש</button></div>`;
  let h = `<div class="sec"><h3>תוצאות החיפוש</h3><span class="meta">${hits.length === 1 ? 'אירוע אחד' : hits.length + ' אירועים'} · <button class="linkish" data-qclear>ניקוי</button></span></div>`;
  for (const d of S.days) {
    const evs = hits.filter(e => e.date === d);
    if (evs.length) h += `<h4 class="qday">${dayName(d)}</h4>` + evs.map(e => card(e, { noRel: d !== S.day })).join('');
  }
  return h;
}
function listHTML() {
  const evs = dayEvents();
  const live = evs.filter(e => status(e) === 'live' && !e.long).sort((a, b) => byNear(a, b) || a.end - b.end);
  const open = evs.filter(e => status(e) === 'live' && e.long);
  const soon = evs.filter(e => status(e) === 'soon').sort((a, b) => a.start - b.start || byNear(a, b));
  const later = evs.filter(e => status(e) === 'later');
  let h = '';
  const filtered = S.filt.size || S.kids || S.access || S.q;
  if (!evs.length) {
    h += `<div class="empty">${filtered ? `אין ב${dayName(S.day, true)} אירועים שמתאימים לסינון. <button class="more" data-clear>לנקות את הסינון</button>` : 'אין אירועים ביום הזה.'}</div>`;
  }
  h += `<div class="sec"><h3>${live.length && isNow() ? '<span class="livedot"></span>קורה עכשיו, ' : `${dayName(S.day, true)}, `}ב-${hm(S.t)}</h3><span class="meta">${live.length} אירועים</span></div>`;
  if (live.length) h += live.map(e => card(e)).join('');
  else if (evs.length) {
    const next = evs.find(e => e.start > S.t);
    h += `<div class="empty">${next ? `בשעה הזאת שקט. הבא מתחיל ב-${hm(next.start)}: ${esc(next.name)}.` : 'היום נגמר. מחר עוד.'}</div>`;
  }
  if (soon.length) h += `<div class="sec"><h3>${isNow() ? 'מתחיל בשעה הקרובה' : `מתחיל עד ${hm(S.t + 60)}`}</h3><span class="meta">${soon.length}</span></div>` + soon.map(e => card(e)).join('');
  if (open.length) h += `<div class="sec"><h3>פתוח לאורך היום</h3><span class="meta">תערוכות ומיצגים</span></div>` + open.map(e => card(e)).join('');
  if (later.length) {
    const shown = S.moreLater ? later : later.slice(0, 5);
    h += `<div class="sec"><h3>בהמשך היום</h3>${later.length > 5 && !S.moreLater ? `<button class="more" data-more>כל ה-${later.length}</button>` : `<span class="meta">${later.length}</span>`}</div>`;
    h += shown.map(e => card(e)).join('');
  }
  const unplaced = evs.filter(e => !e.vgeo);
  if (unplaced.length && !filtered) h += `<div class="empty" style="font-size:12.5px">${unplaced.length} אירועים היום בלי נקודה על המפה (כתובת שלא הצלחנו למקם). הם מופיעים ברשימה.</div>`;
  h += `<div class="empty" style="font-size:12px;color:var(--ink3)">${updatedLabel()}</div>${CREDIT}`;
  return h;
}
const SRC = 'https://github.com/orimosenzon/fun/tree/master/vibe_coding/hanadiv';
const CREDIT = `<footer class="site-footer">
  <a href="${SRC}" target="_blank" rel="noopener" title="קוד המקור בגיטהאב" aria-label="GitHub"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#8a939c" style="fill:#8a939c;stroke:none" aria-hidden="true"> <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.4 3-.405 1.02.005 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/> </svg></a>
  <span class="footer-sep">·</span>
  <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener" title="GNU General Public License v3.0"><svg style="stroke:none" xmlns="http://www.w3.org/2000/svg" width="116" height="22" viewBox="0 0 116 22" role="img" aria-label="license: GPL-3.0"> <title>license: GPL-3.0</title> <defs> <clipPath id="gpl-clip-h"> <rect width="116" height="22" rx="4" fill="#fff"/> </clipPath> </defs> <g clip-path="url(#gpl-clip-h)"> <rect width="56" height="22" fill="#2a2a2a"/> <rect x="56" width="60" height="22" fill="#0d9488"/> </g> <text x="28" y="15" fill="#bbb" font-family="Segoe UI,DejaVu Sans,sans-serif" font-size="10.5" text-anchor="middle">license</text> <text x="86" y="15.5" fill="#fff" font-family="Segoe UI,DejaVu Sans,sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">GPL-3.0</text> </svg></a>
  <span class="footer-sep">·</span>
  <span class="footer-credit"><img src="avatar.png" alt="">אורי מוסנזון</span>
</footer>`;
const updatedAt = () => new Date(S.updated).toLocaleString('he-IL', { timeZone: 'Asia/Jerusalem', day: 'numeric', month: 'numeric', hour: '2-digit', minute: '2-digit' });
function updatedLabel() {
  if (!S.updated) return '';
  return `התוכנייה נמשכת מאתר הפסטיבל כל לילה. עדכון אחרון: ${updatedAt()}`;
}
/* Seats, as the festival counts them (its `availability`, read with the nightly pull).
   Only "full" and "few left" earn a tag; the card says the number and how old it is. */
const FEW = 10;
function availTag(e) {
  const a = e.avail;
  if (!a || isOver(e)) return '';
  if (a.isFull) return `<span class="tag full">${a.registrationOpen ? 'מלא · רשימת המתנה' : 'מלא'}</span>`;
  return a.availableSeats <= FEW ? `<span class="tag few">${a.availableSeats === 1 ? 'נשאר מקום אחד' : `נשארו ${a.availableSeats}`}</span>` : '';
}
function availLine(e) {
  const a = e.avail;
  if (!a) return e.seats ? ` · עד ${e.seats} משתתפים` : '';
  const when = S.updated ? ` <small>(נכון ל-${updatedAt()})</small>` : '';
  if (a.isFull) return `<br><b class="availfull">מלא</b>${a.registrationOpen ? ', אפשר להצטרף לרשימת ההמתנה' : ''}${when}`;
  return `<br>${a.availableSeats === 1 ? 'נשאר מקום אחד' : `נשארו ${a.availableSeats} מקומות`}${e.seats ? ` מתוך ${e.seats}` : ''}${when}`;
}

/* ---------- my plan ---------- */

const onlyMine = () => S.view === 'plan' && S.onlyMine && S.plan.size > 0;
let planMemo = null;
function planList() { return planMemo || (planMemo = [...S.plan].map(id => S.byId.get(id)).filter(Boolean).sort((a, b) => a.date.localeCompare(b.date) || a.start - b.start)); }
const isMine = e => S.plan.has(e.id);
// the planned event that an unplanned one would collide with
const clashWith = e => isMine(e) ? null : planList().find(o => o.date === e.date && o.start < e.end && e.start < o.end && !o.long && !e.long) || null;
// what of mine is on or still to come at the playhead, on the chosen day
function myNext() {
  const left = planList().filter(e => e.date === S.day && e.end > S.t && !e.long);
  return left.find(e => e.start <= S.t) || left[0] || null;
}
function savePlan() {
  planMemo = null;
  try { localStorage.setItem('hanadiv-plan', JSON.stringify([...S.plan])); } catch (e) { /* private window */ }
  if (typeof planChanged === 'function') planChanged();  // sync.js: up to the account
  const n = S.plan.size;
  $('#planCount').textContent = n || '';
}
function toggleStar(id) {
  const e = S.byId.get(id);
  if (S.plan.has(id)) {
    S.plan.delete(id);
    toast('הוסר מהתוכנית', { label: 'ביטול', fn: () => { S.plan.add(id); savePlan(); render(); redrawAll(); redrawDetail(); } });
  } else {
    S.plan.add(id);
    planMemo = null;
    const clash = planList().find(o => o.id !== id && o.date === e.date && o.start < e.end && e.start < o.end);
    toast(clash ? `נוסף, אבל חופף ל"${clash.name}"` : `נוסף לתוכנית שלך ★ (${S.plan.size})`);
    const b = $('#planCount'); b.classList.remove('bump'); void b.offsetWidth; b.classList.add('bump');
  }
  savePlan(); render(); redrawAll();
}
// the plan shows on the timetable, the strip and the day tabs too
function redrawAll() { drawStrip(); drawDays(); if (S.view === 'grid') { const g = $('#grid'), x = g.scrollLeft, y = g.scrollTop; drawGrid(); g.scrollLeft = x; g.scrollTop = y; } }

/* The banner at the top of the list: what of mine is on now or next, and when to leave for it. */
function myNextHTML() {
  const e = myNext(); if (!e) return '';
  const st = status(e);
  const ph = e.photos[0] ? `background-image:url('${esc(e.photos[0].thumb)}')` : '';
  const w = S.me && e.vgeo ? walkMin(S.me, e.vgeo) : null;
  const now = isNow();
  const head = st === 'live' ? `${now ? 'עכשיו' : 'ב-' + hm(S.t)} בתוכנית שלך · עד ${hm(e.end)}`
    : `הבא בתוכנית שלך · ${e.start === S.t ? (now ? 'מתחיל עכשיו' : 'מתחיל ב-' + hm(S.t)) : now ? 'בעוד ' + mins(e.start - S.t) : 'ב-' + hm(e.start)}`;
  let go = '';
  if (w != null && st !== 'live') go = now && e.start - w <= S.t ? `<em class="go">לצאת עכשיו · ${w} דק׳ הליכה</em>` : `<em>לצאת ב-${hm(e.start - w)} · ${w} דק׳ הליכה</em>`;
  const rest = planList().filter(o => o.date === S.day && o.start > e.start && !o.long).length;
  return `<div class="mynext ${st}" data-ev="${e.id}" style="--c:${e.c}"><div class="ph" style="${ph}"></div>
    <div class="body"><small>★ ${head}</small><b>${esc(e.name)}</b>
      <span>${rng(e.start, e.end)} · ${esc(e.place)}</span>${go}
      ${rest ? `<span class="rest">ואחריו עוד ${rest} בתוכנית היום</span>` : ''}</div></div>`;
}

// walking time between two events, guessing ten minutes when one of them is not on the map
const hop = (a, b) => a.vgeo && b.vgeo ? (a.vkey === b.vkey ? 0 : walkMin(a.vgeo, b.vgeo)) : 10;
/* Events that fit in a free window of the plan, with the walk there and on to the next. */
function fits(prev, next, d) {
  return S.events.filter(o => o.date === d && !isMine(o) && !o.long && passes(o)
    && (!prev || o.start >= prev.end + hop(prev, o))
    && (!next || o.end + hop(o, next) <= next.start)
    && !(d === S.day && o.start < S.t))
    .sort((a, b) => a.start - b.start || (prev ? hop(prev, a) - hop(prev, b) : 0));
}
function gapHTML(prev, next, d, label) {
  const key = `${d}:${prev ? prev.id : ''}>${next ? next.id : ''}`;
  const f = fits(prev, next, d);
  if (!f.length) return '';
  const open = S.openGap === key;
  let h = `<button class="gapfit ${open ? 'on' : ''}" data-gap="${esc(key)}">${ICON.plus}${label} · ${f.length} אירועים${open ? ' ▴' : ' ▾'}</button>`;
  if (open) h += `<div class="fitlist">${f.slice(0, 8).map(o => card(o, { noRel: true, noClash: true })).join('')}${f.length > 8 ? `<div class="empty" style="font-size:12.5px">ועוד ${f.length - 8}. אפשר לצמצם עם הסינון למעלה.</div>` : ''}</div>`;
  return h;
}
function planMini(evs, d) {
  const W = S.hi - S.lo;
  let h = '';
  for (let t = Math.ceil((S.lo + 60) / 120) * 120; t < S.hi - 30; t += 120) h += `<u style="right:${(t - S.lo) / W * 100}%">${hm(t)}</u>`;
  h += evs.map(e => `<b style="--c:${e.c};right:${(e.start - S.lo) / W * 100}%;width:${Math.max(Math.min(e.dur, S.hi - e.start) / W * 100, 1.5)}%" data-ev="${e.id}" title="${esc(e.name)}"></b>`).join('');
  if (d === S.day) h += `<i style="right:${(S.t - S.lo) / W * 100}%"></i>`;
  return `<div class="pmini">${h}</div>`;
}

function drawPlan() {
  let h = '';
  if (S.shared) {
    const fresh = S.shared.filter(id => !S.plan.has(id));
    h += `<div class="shared">מישהו שיתף איתכם תוכנית של ${S.shared.length} אירועים.
      ${fresh.length ? `<div class="planbar"><button class="btn warm" data-import>להוסיף ${fresh.length} לתוכנית שלי</button><button class="btn ghost" data-dismiss>לא תודה</button></div>` : '<br>כולם כבר בתוכנית שלכם.'}</div>`;
    h += S.shared.map(id => S.byId.get(id)).filter(Boolean).map(e => card(e, { showDay: true, noRel: true })).join('');
  }
  const list = planList();
  if (!list.length) {
    h += `<div class="sec"><h3>התוכנית שלי</h3></div><div class="planhelp">
      <p><b>בונים תוכנית בשלושה צעדים:</b></p>
      <ol><li>לוחצים ★ ליד כל אירוע שרוצים: ברשימה, בבועה על המפה, בכרטיס האירוע או בלוח הזמנים של מקום.</li>
      <li>כאן הם מסתדרים לפי שעות, עם זמן ההליכה ביניהם ואזהרה כששניים מתנגשים, ובין אירוע לאירוע מוצע מה עוד נכנס בחלון.</li>
      <li>התוכנית זוהרת בכל האפליקציה: על המפה, בלוח הזמנים, בפס השעות ובימים. ואפשר לשלוח אותה לחברים בקישור.</li></ol></div>`;
    if (typeof syncBarHTML === 'function') h += syncBarHTML();
    $('#sheetBody').innerHTML = h + '<div class="divider"></div>' + listHTML(); markFocus(); return;
  }
  h += myNextHTML();
  h += `<div class="sec"><h3>התוכנית שלי</h3><span class="meta">${list.length} אירועים</span></div>
    <div class="planbar"><button class="btn ${S.onlyMine ? 'warm' : 'ghost'}" data-onlymine>${S.onlyMine ? 'רק התוכנית שלי' : 'הכל, עם התוכנית'}</button><button class="btn" data-share>${ICON.share}שיתוף</button><button class="btn gcal" data-cal>${ICON.gcal}ליומן Google</button><button class="btn ghost" data-clear-plan>ניקוי</button></div>
    ${typeof syncBarHTML === 'function' ? syncBarHTML() : ''}`;
  for (const d of S.days) {
    const all = list.filter(e => e.date === d);
    if (!all.length) continue;
    const evs = all.filter(e => !e.long), long = all.filter(e => e.long);
    const mins_ = evs.reduce((a, e) => a + e.dur, 0);
    h += `<div class="planday ${d === S.day ? 'today' : ''}"><h4 data-day="${d}"><span>${dayName(d)}</span><small>${all.length} אירועים${mins_ ? ' · ' + mins(mins_) : ''}</small></h4>${planMini(all, d)}`;
    if (evs[0]) h += gapHTML(null, evs[0], d, 'מה אפשר לפני');
    evs.forEach((e, i) => {
      h += card(e, { noRel: d !== S.day, dim: true, join: true });
      const nx = evs[i + 1];
      if (!nx) return;
      const gap = nx.start - e.end;
      const walk = e.vgeo && nx.vgeo ? (e.vkey === nx.vkey ? 0 : walkMin(e.vgeo, nx.vgeo)) : null;
      let cls = '', txt;
      if (gap < 0) { cls = 'bad'; txt = `חופף לבא אחריו (${mins(-gap)}). צריך לבחור, או לצאת מוקדם`; }
      else if (walk != null && walk > gap) { cls = 'bad'; txt = `${mins(walk)} הליכה, ויש רק ${gap ? mins(gap) : 'אפס דקות'}`; }
      else if (walk != null && walk + 5 > gap) { cls = 'tight'; txt = `צפוף: ${mins(walk)} הליכה, ${mins(gap)} ביניהם`; }
      else txt = `${gap ? mins(gap) + ' הפסקה' : 'מיד אחרי'}${walk ? ` · ${mins(walk)} הליכה` : walk === 0 ? ' · באותו מקום' : ''}`;
      h += `<div class="gap ${cls}">${txt}</div>`;
      if (gap >= 30) h += gapHTML(e, nx, d, 'מה נכנס בחלון');
    });
    if (evs.length) h += gapHTML(evs.at(-1), null, d, 'מה אפשר אחרי');
    if (long.length) h += `<div class="sec"><h3 style="font-size:14px">פתוח לאורך היום</h3></div>` + long.map(e => card(e, { noRel: true, join: true })).join('');
    h += '</div>';
  }
  if (S.onlyMine) h += `<div class="empty">על המפה מופיעים רק המקומות שבתוכנית, ממוספרים לפי הסדר. <button class="more" data-onlymine>להציג גם את כל השאר</button></div>`;
  $('#sheetBody').innerHTML = h + (S.onlyMine ? '' : '<div class="divider"></div>' + listHTML()); markFocus();
}

// The day's plan as a dashed line from place to place, in time order. It is straight, not
// along the streets, so each leg carries its walking time to say what the line is.
let hopMarkers = [];
// a white-edged arrowhead pointing east; the line layer turns it along each leg
function arrowImage() {
  const n = 48, c = document.createElement('canvas');
  c.width = c.height = n;
  const g = c.getContext('2d');
  g.lineJoin = 'round';
  g.beginPath(); g.moveTo(10, 8); g.lineTo(40, 24); g.lineTo(10, 40); g.lineTo(17, 24); g.closePath();
  g.strokeStyle = '#fff'; g.lineWidth = 7; g.stroke();
  g.fillStyle = '#1d2b3a'; g.fill();
  return g.getImageData(0, 0, n, n);
}

function drawRoute() {
  if (!map || !map.getSource('route')) return;
  const evs = S.view === 'plan' ? planList().filter(e => e.date === S.day && e.vgeo) : [];
  const pts = evs.map(e => [e.vgeo.lng, e.vgeo.lat]);
  map.getSource('route').setData({ type: 'FeatureCollection', features: pts.length > 1 ? [{ type: 'Feature', geometry: { type: 'LineString', coordinates: pts } }] : [] });
  hopMarkers.forEach(m => m.remove()); hopMarkers = [];
  for (let i = 1; i < evs.length; i++) {
    const a = evs[i - 1], b = evs[i];
    if (a.vkey === b.vkey) continue;
    const w = walkMin(a.vgeo, b.vgeo), gap = b.start - a.end;
    const el = document.createElement('div');
    el.className = 'hop' + (gap < w && !a.long && !b.long ? ' late' : '');
    el.title = `הליכה מ${a.place} אל ${b.place}, בערך ${w} דקות${gap < w ? ' (יותר מהזמן שיש בין האירועים)' : ''}`;
    el.innerHTML = `${ICON.walk}${w} דק׳`;
    const mk = new maplibregl.Marker({ element: el, anchor: 'center' })
      .setLngLat([(a.vgeo.lng + b.vgeo.lng) / 2, (a.vgeo.lat + b.vgeo.lat) / 2]).addTo(map);
    mk._updateOpacity = () => {};  // the same graphics-card stall as the venue pictures
    hopMarkers.push(mk);
  }
}

/* ---------- timetable ---------- */

const PPM = 2.4, NAMEW = 128;
function drawGrid() {
  const g = $('#grid');
  const evs = dayEvents();
  const rows = new Map();
  evs.forEach(e => { if (!rows.has(e.vkey)) rows.set(e.vkey, []); rows.get(e.vkey).push(e); });
  const W = (S.hi - S.lo) * PPM + NAMEW + 20;
  const X = t => NAMEW + (t - S.lo) * PPM;
  let h = `<div class="gt" style="width:${W}px"><div class="hours">`;
  for (let t = S.lo; t <= S.hi; t += 60) h += `<span style="right:${X(t)}px">${hm(t)}</span>`;
  h += '</div>';
  for (let t = S.lo; t <= S.hi; t += 60) h += `<div class="grid-hour" style="right:${X(t)}px"></div>`;
  // places with something on now or next come first, places whose day is over go last
  const key = list => Math.min(...list.map(e => (e.end <= S.t ? 1e5 + e.start : Math.max(e.start, S.t)) + (e.long ? 5000 : 0)));
  const ordered = [...rows.values()].sort((a, b) => key(a) - key(b) || a[0].start - b[0].start);
  h += myRowHTML(X);
  for (const list of ordered) {
    const lanes = [];
    for (const e of list) {
      const end = e.start + Math.min(e.dur, S.hi - e.start);
      let l = lanes.findIndex(x => x <= e.start);
      if (l < 0) { l = lanes.length; lanes.push(0); }
      lanes[l] = end; e._lane = l;
    }
    const n = lanes.length, rowH = Math.max(54, n * 28 + 12), bh = (rowH - 12) / n - 2;
    const v = list[0].venue;
    const vmine = list.some(isMine);
    h += `<div class="row ${vmine ? 'hasmine' : ''}" style="height:${rowH}px"><div class="vname" data-place="${esc(v.key)}" title="לוח הזמנים של המקום"><b>${vmine ? '<s>★</s>' : ''}${esc(v.title)}</b>${esc(v.stages.length > 1 ? v.stages.length + ' במות' : v.street !== v.place ? v.street : '')}</div>`;
    for (const e of list) {
      const w = Math.max(Math.min(e.dur, S.hi - e.start) * PPM - 3, 16);
      h += `<div class="bar ${n > 1 ? 'split' : ''} ${S.plan.has(e.id) ? 'planned' : ''}" data-ev="${e.id}" style="--c:${e.c};right:${X(e.start)}px;width:${w}px;top:${6 + e._lane * (bh + 2)}px;height:${bh}px" title="${esc(e.name)}"><b>${esc(e.name)}</b><span>${rng(e.start, e.end)}</span></div>`;
    }
    h += '</div>';
  }
  if (!ordered.length) h += `<div class="empty" style="padding:20px">אין אירועים שמתאימים לסינון.</div>`;
  h += `<div class="nowline" style="right:${X(S.t)}px"></div></div>`;
  g.innerHTML = h;
  tickGrid();
}
/* The first row of the timetable is my own: every planned event of the day on one line,
   pinned under the hours so it stays in sight while scrolling through the places. */
function myRowHTML(X) {
  const evs = planList().filter(e => e.date === S.day);
  if (!evs.length) return `<div class="row myrow empty" style="height:40px"><div class="vname"><b>★ התוכנית שלי</b><span>${S.plan.size ? 'אין לך אירועים ביום הזה' : 'לוחצים על אירוע ואז ★, והוא יופיע כאן'}</span></div></div>`;
  const lanes = [];
  for (const e of evs) {
    let l = lanes.findIndex(x => x <= e.start);
    if (l < 0) { l = lanes.length; lanes.push(0); }
    lanes[l] = e.start + Math.min(e.dur, S.hi - e.start); e._ml = l;
  }
  const n = lanes.length, rowH = Math.max(54, n * 30 + 12), bh = (rowH - 12) / n - 2;
  let h = `<div class="row myrow" style="height:${rowH}px"><div class="vname"><b>★ התוכנית שלי</b>${evs.length} אירועים</div>`;
  for (const e of evs) {
    const w = Math.max(Math.min(e.dur, S.hi - e.start) * PPM - 3, 16);
    h += `<div class="bar planned ${n > 1 ? 'split' : ''}" data-ev="${e.id}" style="--c:${e.c};right:${X(e.start)}px;width:${w}px;top:${6 + e._ml * (bh + 2)}px;height:${bh}px" title="${esc(e.name)}"><b>${esc(e.name)}</b><span>${esc(e.place)}</span></div>`;
  }
  return h + '</div>';
}
function tickGrid() {
  const g = $('#grid');
  const line = $('.nowline', g); if (!line) return;
  line.style.right = NAMEW + (S.t - S.lo) * PPM + 'px';
  $$('.bar', g).forEach(b => b.classList.toggle('past', status(S.byId.get(b.dataset.ev)) === 'past'));
}
// The timetable follows the playhead: the red line stays near the right edge, with what is
// on and what comes next to its left. While the playhead moves the bars only fade and the
// view slides; once it rests, the rows are re-ordered for the new time and the view goes
// back to the top, where the places with something on now are.
function followGrid(smooth) {
  const g = $('#grid');
  const x = NAMEW + (S.t - S.lo) * PPM;
  g.scrollTo({ left: -(x - NAMEW - 90), behavior: smooth ? 'smooth' : 'auto' });
}
function scrollGridToNow() { followGrid(true); $('#grid').scrollTo({ top: 0 }); }
let regridTimer;
function regridSoon() {
  clearTimeout(regridTimer);
  regridTimer = setTimeout(() => { if (S.view === 'grid' && !S.playing) { drawGrid(); followGrid(); $('#grid').scrollTop = 0; } }, 350);
}

/* ---------- detail card ---------- */

const ICON = {
  search: '<svg viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21M8 8l5 5M13 8l-5 5"/></svg>',
  clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  pin: '<svg viewBox="0 0 24 24"><path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/></svg>',
  ppl: '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.2"/><path d="M3 20c.6-3.4 3-5.5 6-5.5s5.4 2.1 6 5.5"/><circle cx="17" cy="9" r="2.4"/><path d="M16.5 14.6c2.4.2 4 1.9 4.5 4.6"/></svg>',
  acc: '<svg viewBox="0 0 24 24"><path d="M4 12h16M12 4v16"/></svg>',
  nav: '<svg viewBox="0 0 24 24"><path d="m3 11 18-8-8 18-2-8z"/></svg>',
  ext: '<svg viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>',
  cal: '<svg viewBox="0 0 24 24"><rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/></svg>',
  // a calendar page in Google's four colours
  gcal: '<svg class="gc" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="17" rx="3" fill="#fff" stroke="#4285f4" stroke-width="2"/><path d="M3 9.5h18" stroke="#4285f4" stroke-width="2"/><path d="M8 2.5v3.5M16 2.5v3.5" stroke="#ea4335" stroke-width="2" stroke-linecap="round"/><path d="m8.5 15 2.3 2.2 4.7-4.7" stroke="#34a853" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/><circle cx="18" cy="18" r="1.4" fill="#fbbc05" stroke="none"/></svg>',
  share: '<svg viewBox="0 0 24 24"><circle cx="18" cy="5" r="2.5"/><circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="19" r="2.5"/><path d="m8.2 10.8 7.6-4.4M8.2 13.2l7.6 4.4"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  cloud: '<svg viewBox="0 0 24 24"><path d="M7 18h10.5a4 4 0 0 0 .6-7.95A6 6 0 0 0 6.5 9.5 4.3 4.3 0 0 0 7 18z"/></svg>',
  walk: '<svg viewBox="0 0 24 24"><circle cx="13" cy="4.5" r="2"/><path d="m9 21 2.5-6 2.5 2v4M10.5 9.5 13 8l2.5 4 3 1M13 8l-1.5 6.5M10.5 9.5 7 11.5v3"/></svg>',
  list: '<svg viewBox="0 0 24 24"><path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/></svg>',
  sat: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.6 3.9 5.6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-5.6-3.9-9s1.3-6.4 3.9-9z"/></svg>',
  sliders: '<svg viewBox="0 0 24 24"><path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/></svg>',
  full: '<svg viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  shrink: '<svg viewBox="0 0 24 24"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  wa: '<svg viewBox="0 0 24 24"><path d="M3.5 20.5 4.8 16A8.6 8.6 0 1 1 8 19.2z"/><path d="M9 8.2c.2 3.4 3.3 6.5 6.8 6.8l1-1.6-2.1-1-1 .9c-1.1-.5-2.4-1.8-2.9-2.9l.9-1-1-2.1z"/></svg>',
};

/* The festival form had two free link fields. Some hosts put a WhatsApp group there,
   and some wrote a phone number into the description instead. */
const isWa = url => /(^|\.)(whatsapp\.com|wa\.me)$/.test((() => { try { return new URL(url).hostname; } catch (e) { return ''; } })());
function waLinks(e) {
  const wa = e.links.filter(l => isWa(l.url));
  const groups = wa.filter(l => /chat\.whatsapp/.test(l.url)).length;
  let n = 0;
  return wa.map(l => ({ url: l.url, title: !/chat\.whatsapp/.test(l.url) ? 'הודעה בווטסאפ' : groups > 1 ? `קבוצת ווטסאפ ${++n}` : 'קבוצת הווטסאפ' }));
}
function noteHTML(note) {
  return esc(note).replace(/ · /g, '\n').replace(/(^|[^\d])(05\d)[- ]?(\d{3})[- ]?(\d{4})(?!\d)/g, (m, pre, a, b, c) =>
    `${pre}<a href="tel:${a}${b}${c}">${a}-${b}${c}</a> <a class="wa-inline" target="_blank" rel="noopener" href="https://wa.me/972${a.slice(1)}${b}${c}" title="ווטסאפ">${ICON.wa}</a>`);
}

/* A place's whole festival on one card: every day, every stage. */
function openPlace(v) {
  if (!v) return;
  S.detail = { place: v };
  const nav = v.geo ? `<a class="btn ghost" target="_blank" rel="noopener" href="https://waze.com/ul?ll=${v.geo.lat},${v.geo.lng}&navigate=yes">${ICON.nav}ווייז</a>
     <a class="btn ghost" target="_blank" rel="noopener" href="https://www.google.com/maps/dir/?api=1&destination=${v.geo.lat},${v.geo.lng}&travelmode=walking&hl=iw">גוגל מפות</a>
     <button class="btn ghost" data-placemap="${esc(v.key)}">${ICON.pin}על המפה</button>` : '';
  const multi = v.stages.length > 1;
  const W = S.hi - S.lo;
  let days = '';
  for (const d of S.days) {
    const evs = v.events.filter(e => e.date === d);
    if (!evs.length) continue;
    const today = d === S.day;
    const bars = evs.map(e => `<b style="--c:${e.c};right:${(e.start - S.lo) / W * 100}%;width:${Math.min(e.dur, S.hi - e.start) / W * 100}%"></b>`).join('');
    days += `<section class="pday ${today ? 'today' : ''}" data-pday="${d}"><h3>${dayName(d)}<span>${evs.length} אירועים</span></h3>
      <div class="mini">${bars}${today ? `<i style="right:${(S.t - S.lo) / W * 100}%"></i>` : ''}</div>`;
    for (const e of evs) {
      const st = today ? status(e) : '';
      const r = today ? relLabel(e) : null;
      days += `<div class="srow ${st} ${isMine(e) ? 'mine' : ''}" data-ev="${e.id}" style="--c:${e.c}">
        <span class="h">${rng(e.start, e.end)}</span>
        <span class="n"><b>${esc(e.name)}</b><small>${multi ? esc(e.place) + ' · ' : ''}${esc(e.craft || e.group)}${r && (st === 'live' || st === 'soon') ? ` · <em class="${r.cls}">${r.txt}</em>` : ''}</small></span>
        ${waLinks(e).length ? `<span class="wa-dot" title="יש קבוצת ווטסאפ">${ICON.wa}</span>` : ''}
        <button class="star ${S.plan.has(e.id) ? 'on' : ''}" data-star="${e.id}" aria-label="לתוכנית שלי">${STAR}</button></div>`;
    }
    days += '</section>';
  }
  $('#detailCard').style.setProperty('--c', 'var(--accent)');
  $('#detailCard').innerHTML = `<button class="x" data-close aria-label="סגירה">${ICON.x}</button>
    <div class="phead"><div class="kind">${ICON.list}לוח הזמנים של המקום</div>
      <h2>${esc(v.title)}</h2>
      <div class="addr">${multi ? `${v.stages.length} במות: ${v.stages.map(esc).join(' · ')}` : esc(v.street && v.street !== v.place ? v.street : '')}</div></div>
    <div class="inner">
      <div class="actions">${nav}<button class="btn ghost" data-shareplace>${ICON.share}שיתוף</button></div>
      ${days}
    </div>`;
  if (!S.pushed) { history.pushState({ card: 1 }, ''); S.pushed = true; }
  $('#detail').hidden = false;
  const today = $('#detailCard .pday.today');
  $('#detailCard').scrollTop = today && today !== $('#detailCard .pday') ? today.offsetTop - 60 : 0;
  setHash();
}
function redrawDetail() {
  if (!S.detail) return;
  const y = $('#detailCard').scrollTop;
  S.detail.place ? openPlace(S.detail.place) : openDetail(S.detail);
  $('#detailCard').scrollTop = y;
}

function openDetail(e) {
  if (!e) return;
  S.detail = e;
  const v = e.venue;
  const gallery = e.photos.length ? `<div class="gallery">${e.photos.map(p => `<img src="${esc(p.full)}" alt="" loading="lazy">`).join('')}</div>` : `<div class="gallery" style="height:70px"></div>`;
  const r = e.date === S.day ? relLabel(e) : null;
  const W = S.hi - S.lo;
  const tl = `<div class="timeline"><b style="right:${(e.start - S.lo) / W * 100}%;width:${Math.min(e.dur, S.hi - e.start) / W * 100}%"></b>${e.date === S.day ? `<i style="right:${(S.t - S.lo) / W * 100}%"></i>` : ''}</div>`;
  const nav = v.geo ? `<a class="btn ghost" target="_blank" rel="noopener" href="https://waze.com/ul?ll=${v.geo.lat},${v.geo.lng}&navigate=yes">${ICON.nav}ווייז</a>
     <a class="btn ghost" target="_blank" rel="noopener" href="https://www.google.com/maps/dir/?api=1&destination=${v.geo.lat},${v.geo.lng}&travelmode=walking&hl=iw">גוגל מפות</a>` : '';
  const others = v.events.filter(o => o.id !== e.id);
  const walk = S.me && v.geo ? ` · ${walkMin(S.me, v.geo)} דק׳ הליכה ממך` : '';
  const wa = waLinks(e), links = e.links.filter(l => !isWa(l.url));
  $('#detailCard').style.setProperty('--c', e.c);
  $('#detailCard').innerHTML = `<button class="x" data-close aria-label="סגירה">${ICON.x}</button>${gallery}
    <div class="inner">
      <div class="kind"><i></i>${esc(e.group)}${e.craft && e.craft !== e.group ? ' · ' + esc(e.craft) : ''}</div>
      <h2>${esc(e.name)}</h2>
      ${isMine(e) ? '<span class="tag minetag">★ בתוכנית שלך</span> ' : ''}${r ? `<span class="tag ${r.cls}">${r.txt}</span>` : ''}
      ${(() => { const c = clashWith(e); return c ? `<div class="clashnote">חופף ל<b>${esc(c.name)}</b> שבתוכנית שלך (${rng(c.start, c.end)})</div>` : ''; })()}
      <div class="facts">
        <div>${ICON.clock}<span>${dayName(e.date)} · ${rng(e.start, e.end)} (${esc(e.durLabel)})${tl}</span></div>
        <div>${ICON.pin}<span><b>${esc(e.place)}</b>${e.street && e.street !== e.place ? ', ' + esc(e.street) : ''}${walk}${e.directions ? `<br><small>${esc(e.directions)}</small>` : ''}
          ${v.events.length > 1 ? `<br><button class="linkish" data-place="${esc(v.key)}">לוח הזמנים של המקום (${v.events.length} אירועים)</button>` : ''}</span></div>
        <div>${ICON.ppl}<span>${esc(e.audience)}${availLine(e)}</span></div>
        ${e.access ? `<div>${ICON.acc}<span>${esc(e.access)}</span></div>` : ''}
      </div>
      <div class="actions">
        <button class="btn ${S.plan.has(e.id) ? 'warm' : ''}" data-star="${e.id}">${STAR}${S.plan.has(e.id) ? 'בתוכנית שלי' : 'לתוכנית שלי'}</button>
        ${wa.map(l => `<a class="btn wa" target="_blank" rel="noopener" href="${esc(l.url)}">${ICON.wa}${esc(l.title)}</a>`).join('')}
        <a class="btn join" target="_blank" rel="noopener" href="${esc(joinUrl(e))}">${ICON.ext}${e.avail?.isFull ? 'לרשימת ההמתנה באתר הנדיב' : 'הרשמה באתר הנדיב'}</a>
        <a class="btn ghost" target="_blank" rel="noopener" href="${esc(e.url)}">${ICON.ext}עמוד האירוע</a>
        ${nav}
        <a class="btn ghost" href="${esc(gcalUrl(e))}" target="_blank" rel="noopener" data-gadd="${e.id}">${ICON.gcal}ליומן Google</a>
        <button class="btn ghost" data-ics="${e.id}">${ICON.cal}קובץ יומן</button>
        ${v.geo ? `<button class="btn ghost" data-showmap="${e.id}">${ICON.pin}על המפה</button>` : ''}
      </div>
      <div class="note">${noteHTML(e.note)}</div>
      ${links.length ? `<div class="links">${links.map(l =>`<a target="_blank" rel="noopener" href="${esc(l.url)}">${esc(l.title)}</a>`).join('')}</div>` : ''}
      ${others.length ? `<div class="same"><h4>עוד באותו מקום (${others.length})</h4>${others.map(o => card(o, { showDay: true, noRel: o.date !== S.day })).join('')}</div>` : ''}
      ${e.long || isMine(e) ? '' : (() => { const nb = alsoThen(e); return nb.length ? `<div class="same"><h4>באותה שעה, קרוב (${nb.length})</h4>${nb.map(o => card(o, { noRel: true })).join('')}</div>` : ''; })()}
    </div>`;
  if (!S.pushed) { history.pushState({ card: 1 }, ''); S.pushed = true; }
  $('#detail').hidden = false;
  $('#detailCard').scrollTop = 0;
  setHash();
}
// unplanned events on at the same time within a short walk: the alternatives to this one
function alsoThen(e) {
  if (!e.vgeo) return [];
  return S.events.filter(o => o.date === e.date && o.id !== e.id && o.vkey !== e.vkey && !o.long && o.vgeo
    && Math.abs(o.start - e.start) <= 30 && walkMin(o.vgeo, e.vgeo) <= 12)
    .sort((a, b) => walkMin(a.vgeo, e.vgeo) - walkMin(b.vgeo, e.vgeo)).slice(0, 4);
}
function closeDetail(fromBack) {
  $('#detail').hidden = true; S.detail = null;
  if (S.pushed) { S.pushed = false; if (!fromBack) { history.back(); return; } }
  setHash();
}
addEventListener('popstate', () => { if (S.detail) closeDetail(true); else S.pushed = false; });

/* ---------- calendar ---------- */

function icsTime(date, m) {
  // The festival falls after the clocks go back (25/10/2026), so Israel is UTC+2.
  const t = new Date(`${date}T00:00:00+02:00`).getTime() + m * 60000;
  return new Date(t).toISOString().replace(/[-:]/g, '').replace(/\.\d+/, '');
}
function downloadIcs(evs, name) {
  const lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//pardespedia//hanadiv//HE', 'CALSCALE:GREGORIAN'];
  const fold = s => s.replace(/[\\;,]/g, c => '\\' + c).replace(/\n/g, '\\n');
  for (const e of evs) {
    lines.push('BEGIN:VEVENT', `UID:hanadiv-${e.id}@pardespedia.info`, `DTSTAMP:${icsTime('2026-10-01', 0)}`,
      `DTSTART:${icsTime(e.date, e.start)}`, `DTEND:${icsTime(e.date, e.end)}`,
      `SUMMARY:${fold(e.name)}`, `LOCATION:${fold([e.place, e.street, 'פרדס חנה-כרכור'].filter(Boolean).join(', '))}`,
      `DESCRIPTION:${fold(e.url + '\n\n' + e.note.slice(0, 600))}`, `URL:${e.url}`);
    if (e.vgeo) lines.push(`GEO:${e.vgeo.lat};${e.vgeo.lng}`);
    lines.push('END:VEVENT');
  }
  lines.push('END:VCALENDAR');
  const blob = new Blob([lines.join('\r\n')], { type: 'text/calendar' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name + '.ics'; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* Google Calendar. Writing into someone's calendar straight from the page needs a calendar
   permission that Google only grants an app after weeks of review, and until then it greets
   people with an "unverified app" warning. So each event goes through Google's own "new event"
   page instead, filled in and waiting for one tap on Save; the panel lists the plan and ticks
   off what was sent. The ticks are this browser's only (a convenience, nothing to sync). */
function gcalUrl(e) {
  const q = new URLSearchParams({
    action: 'TEMPLATE', text: e.name, ctz: 'Asia/Jerusalem',
    dates: `${icsTime(e.date, e.start)}/${icsTime(e.date, e.end)}`,
    location: [e.place, e.street, 'פרדס חנה-כרכור'].filter(Boolean).join(', '),
    details: `${e.note.slice(0, 700)}\n\nהרשמה: ${joinUrl(e)}\nבמפת הפסטיבל: ${location.origin}${location.pathname}#e=${e.id}`,
  });
  return 'https://calendar.google.com/calendar/render?' + q;
}
const GCAL = 'hanadiv-gcal';
function gcalDone() { try { return new Set(JSON.parse(localStorage.getItem(GCAL)) || []); } catch (e) { return new Set(); } }
function gcalAdded(id) {
  const s = gcalDone(); s.add(id);
  try { localStorage.setItem(GCAL, JSON.stringify([...s])); } catch (e) { /* private window */ }
  // after the tab with Google opens, the list shows what is left
  if (!$('#calPanel').hidden) setTimeout(openCal, 300);
}
function openCal() {
  toggleShowPanel(false); closeShare();
  const box = $('#calPanel'), evs = planList(), done = gcalDone();
  box.hidden = false;
  const left = evs.filter(e => !done.has(e.id)).length;
  let h = `<div class="sphead"><b>${ICON.gcal}ליומן Google</b><button class="iconbtn" data-calclose aria-label="סגירה">${ICON.x}</button></div>`;
  if (!evs.length) return box.innerHTML = h + '<p class="shnote">התוכנית ריקה. מסמנים ★ על אירועים, והם יופיעו כאן.</p>';
  h += `<p class="shnote">לוחצים "הוספה", גוגל פותח את האירוע מוכן עם השעה, המקום והקישור, ונשאר רק ללחוץ "שמירה".
    ${left ? `<b>${left === evs.length ? `${evs.length} אירועים בתוכנית` : `נשארו ${left} מתוך ${evs.length}`}</b>` : '<b>כל התוכנית ביומן.</b>'}</p><div class="callist">`;
  let day = '';
  for (const e of evs) {
    if (e.date !== day) { day = e.date; h += `<h5>${dayName(day)}</h5>`; }
    const ok = done.has(e.id);
    h += `<div class="calrow ${ok ? 'done' : ''}"><span><b>${esc(e.name)}</b><small>${rng(e.start, e.end)} · ${esc(e.place)}</small></span>
      <a class="btn ${ok ? 'ghost' : 'gcal'}" href="${esc(gcalUrl(e))}" target="_blank" rel="noopener" data-gadd="${e.id}">${ok ? '✓ נוסף' : 'הוספה'}</a></div>`;
  }
  h += `</div><p class="calfile">הכל בבת אחת, בקובץ: <button class="more" data-ics-all>הורדת קובץ יומן</button><br>
    נפתח מיד באייפון ובאאוטלוק. ליומן Google במחשב: הגדרות, ואז "ייבוא וייצוא".</p>`;
  box.innerHTML = h;
}
function closeCal() { $('#calPanel').hidden = true; }

/* ---------- chips, days, views ---------- */

function drawDays() {
  const today = festivalNow();
  $('#days').innerHTML = S.days.map(d => {
    const n = S.events.filter(e => e.date === d && passes(e)).length;
    const my = planList().filter(e => e.date === d).length;
    return `<button role="tab" data-day="${d}" class="${d === S.day ? 'on' : ''}">${dayName(d, true)}${my ? ` <span class="mystars">★${my}</span>` : ''} <small>${d.slice(8)}.${d.slice(5, 7)}${today && today.date === d ? ' · <span class="today">היום</span>' : ''}<span class="cnt"> · ${n} אירועים</span></small></button>`;
  }).join('');
}
function drawChips() {
  const counts = {};
  S.events.forEach(e => counts[e.group] = (counts[e.group] || 0) + 1);
  const gs = S.groups.filter(g => counts[g]);
  $('#chips').innerHTML =
    gs.map(g => `<button class="chip ${S.filt.has(g) ? 'on' : S.filt.size ? 'off' : ''}" data-g="${esc(g)}" style="--c:${S.colors[g]}"><i></i>${esc(g)}</button>`).join('') +
    `<button class="chip sep ${S.kids ? 'on' : ''}" data-kids style="--c:#1d7a52">מתאים לילדים</button>` +
    `<button class="chip ${S.access ? 'on' : ''}" data-access style="--c:#1a5bb8">נגיש</button>`;
}

function setView(v) {
  if (v !== S.view) closePopup();
  S.view = v;
  document.body.className = 'view-' + v;
  if (v !== 'map' && S.clean) { S.clean = false; if (document.fullscreenElement) { S.fs = false; document.exitFullscreen().catch(() => {}); } }
  layoutUI();
  $$('#views button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
  $('#grid').hidden = v !== 'grid';
  if (v === 'plan') snap('mid');
  render();
  if (v === 'grid') { drawGrid(); scrollGridToNow(); }
  else if (map) map.resize();  // before any camera move: a resize cuts an animation short
  if (v === 'plan' && map) {
    const pts = planList().filter(e => e.date === S.day && e.vgeo);
    if (pts.length) {
      const b = new maplibregl.LngLatBounds(); pts.forEach(e => b.extend([e.vgeo.lng, e.vgeo.lat]));
      map.fitBounds(b, { padding: fitPad(), maxZoom: 15.5 });
    }
  }
  setHash();
}

/* ---------- state changes ---------- */

let raf = 0;
function render() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    const evs = dayEvents();
    const live = evs.filter(e => status(e) === 'live' && !e.long).length;
    const soon = evs.filter(e => status(e) === 'soon').length;
    $('#timeLabel').innerHTML = `<span class="tday">${dayName(S.day, true)} </span>${hm(S.t)}`;
    const now = isNow();
    document.documentElement.classList.toggle('preview', !now);
    $('#countLabel').innerHTML = now
      ? `<b>${live}</b><span class="tlong"> קורים</span> עכשיו · ${soon}<span class="tlong"> מתחילים בשעה הקרובה</span><span class="tshort"> בקרוב</span>`
      : `<b>${live}</b> קורים בשעה הזאת · ${soon}<span class="tlong"> מתחילים בשעה שאחריה</span><span class="tshort"> אחריה</span>`;
    $('#realNow').innerHTML = now ? '' : realNowHTML();
    if (S.clean) $('#cleanbar').innerHTML = `<b>${dayName(S.day, true)} ${hm(S.t)}</b><span><b class="n">${live}</b> ${now ? 'קורים עכשיו' : 'קורים בשעה הזאת'} · ${soon} ${now ? 'בקרוב' : 'אחריה'}</span>${ICON.shrink}`;
    placePlayhead();
    sky();
    if (map) drawMarkers();
    drawRoute();
    if (S.view === 'plan') drawPlan(); else drawList();
    if (S.view === 'grid') { tickGrid(); followGrid(); regridSoon(); }
    $('#nowBtn').classList.toggle('on', now);
    setHashSoon();
  });
}
function setT(t) {
  t = clamp(t, S.lo, S.hi);
  if (t === S.t) return;
  S.t = t; render();
}
function setDay(d) {
  if (d === S.day) return;
  S.day = d; S.moreLater = false;
  closePopup();
  drawDays(); drawStrip();
  if (S.view === 'grid') { drawGrid(); scrollGridToNow(); }
  if (S.view === 'plan') setView('plan');
  render();
}
function filtersChanged() {
  drawChips(); drawDays(); drawStrip(); drawQInfo();
  if (S.view === 'grid') drawGrid();
  render();
}

/* "Now" is the real time of day, in Israel. During the festival it also picks today;
   before it, the time is shown on the chosen festival day. While following, the
   playhead moves with the clock; touching the strip or pressing play lets go. */
function goNow(quiet) {
  const n = israelNow();
  S.follow = true;
  const today = S.days.includes(n.date);
  if (today) setDay(n.date);
  const m = Math.round(clamp(n.m, S.lo, S.hi) / STEP) * STEP;
  S.t === m ? render() : setT(m);
  if (quiet) return;
  if (!today) {
    const days = Math.round((new Date(S.days[0]) - new Date(n.date)) / 864e5);
    toast(days > 0 ? `הפסטיבל מתחיל בעוד ${days === 1 ? 'יום' : days + ' ימים'}. זו השעה עכשיו, ב${dayName(S.day, true)}.` : 'הפסטיבל נגמר. זו השעה עכשיו.');
  } else if (n.m !== m) toast(n.m < S.lo ? 'עוד מוקדם, האירועים מתחילים ב-' + hm(S.lo) : 'האירועים של היום נגמרו');
}
function letGo() { if (S.follow) { S.follow = false; $('#nowBtn').classList.remove('on'); } }

/* ---------- play ---------- */

function play() {
  if (S.playing) return stopPlay();
  letGo();
  if (S.t >= S.hi - STEP) S.t = S.lo;
  $('#playBtn').innerHTML = '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
  S.playing = setInterval(() => {
    if (S.t >= S.hi) return stopPlay();
    setT(S.t + STEP);
  }, 90);
}
function stopPlay() {
  if (!S.playing) return;
  clearInterval(S.playing); S.playing = null;
  $('#playBtn').innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
}

/* ---------- sheet ---------- */

function snap(s) { $('#sheet').dataset.snap = s; setHashSoon(); }
function setupSheet() {
  const sheet = $('#sheet'), grab = $('#grab');
  let y0 = null, h0 = 0, moved = false;
  const offset = () => sheet.getBoundingClientRect().top - sheet.parentElement.getBoundingClientRect().top;
  grab.addEventListener('pointerdown', ev => {
    if (innerWidth >= 900) return;
    y0 = ev.clientY; h0 = offset(); moved = false;
    grab.setPointerCapture(ev.pointerId); sheet.classList.add('dragging');
  });
  grab.addEventListener('pointermove', ev => {
    if (y0 == null) return;
    const dy = ev.clientY - y0;
    if (Math.abs(dy) > 4) moved = true;
    const H = sheet.parentElement.clientHeight;
    const top = clamp(h0 + dy, H * .08, H - 128);
    sheet.style.height = `${H - top}px`;
  });
  const end = ev => {
    if (y0 == null) return;
    sheet.classList.remove('dragging'); sheet.style.height = '';
    const H = sheet.parentElement.clientHeight;
    if (!moved) { snap(sheet.dataset.snap === 'full' ? 'peek' : sheet.dataset.snap === 'peek' ? 'mid' : 'full'); y0 = null; return; }
    const top = h0 + (ev.clientY - y0);
    const frac = top / H;
    snap(frac < .3 ? 'full' : frac < .7 ? 'mid' : 'peek');
    y0 = null;
  };
  grab.addEventListener('pointerup', end);
  grab.addEventListener('pointercancel', end);
}

/* ---------- url ---------- */

/* The address holds the whole state of the app, so a reload or a link brings back exactly
   what was on screen: day and time (none while following the real clock), view, open card,
   what is hidden, map-only, filters and search, aerial photo and 3D, the camera, the popup on
   the map and how high the phone's sheet is. Only what differs from the plain address is
   written, so the plain address is the app with everything showing. The personal plan stays
   in the browser; a shared link may carry it as an offer to add. */
let hashTimer;
function setHashSoon() { clearTimeout(hashTimer); hashTimer = setTimeout(setHash, 400); }
function stateParams() {
  const p = new URLSearchParams();
  p.set('d', S.day.slice(8));
  if (!S.follow) p.set('t', hm(S.t));  // following the clock: a reload should come back to now
  if (S.view !== 'map') p.set('v', S.view);
  if (S.detail) S.detail.place ? p.set('place', S.detail.place.key) : p.set('e', S.detail.id);
  const hide = Object.keys(UI_DEF).filter(k => !S.ui[k]);
  if (hide.length) p.set('hide', hide.join(','));
  if (S.clean) p.set('full', '1');
  if (S.filt.size) p.set('g', [...S.filt].join(','));
  if (S.kids) p.set('kids', '1');
  if (S.access) p.set('acc', '1');
  if (S.q) p.set('q', S.q);
  if (!S.onlyMine) p.set('all', '1');
  if (S.sat) p.set('sat', '1');
  if (S.tilt) p.set('3d', '1');
  if (S.sel && S.view !== 'grid') { p.set('at', S.sel.key); if (S.focus) p.set('f', S.focus.id); }
  const sh = $('#sheet').dataset.snap;
  if (innerWidth < 900 && sh !== 'mid') p.set('sh', sh);
  if (map && S.camReady) {
    const c = map.getCenter(), b = Math.round(map.getBearing()), pt = Math.round(map.getPitch());
    p.set('m', [c.lat.toFixed(5), c.lng.toFixed(5), +map.getZoom().toFixed(2), ...(b || pt ? [b, pt] : [])].join(','));
  }
  return p;
}
const query = p => p.toString().replace(/%3A/g, ':').replace(/%2C/g, ',');
function setHash() { history.replaceState(null, '', '#' + query(stateParams())); }
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const d = p.get('d') && S.days.find(x => x.slice(8) === p.get('d').padStart(2, '0'));
  const t = p.get('t') && /^(\d\d?):(\d\d)$/.exec(p.get('t'));
  const m = (p.get('m') || '').split(',').map(Number);
  const list = k => (p.get(k) || '').split(',').filter(Boolean);
  return {
    day: d, t: t ? +t[1] * 60 + +t[2] : null, view: p.get('v'), e: p.get('e'), place: p.get('place'),
    plan: p.get('plan') ? p.get('plan').split(/[,.]/).filter(id => S.byId.has(id)) : null,
    hide: list('hide').filter(k => k in UI_DEF), full: p.has('full'),
    g: list('g').filter(g => S.groups.includes(g)), kids: p.has('kids'), acc: p.has('acc'), q: p.get('q') || '',
    all: p.has('all'), sat: p.has('sat'), tilt: p.has('3d'), at: p.get('at'), f: p.get('f'),
    sh: ['peek', 'mid', 'full'].includes(p.get('sh')) ? p.get('sh') : null,
    cam: m.length >= 3 && m.slice(0, 3).every(Number.isFinite) && Math.abs(m[0] - CENTER[1]) < 1 && Math.abs(m[1] - CENTER[0]) < 1
      ? { lat: m[0], lng: m[1], zoom: clamp(m[2], 8, 20), bearing: m[3] || 0, pitch: clamp(m[4] || 0, 0, 65) } : null,
  };
}

/* ---------- misc ---------- */

let toastTimer;
function toast(msg, act) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  if (act) {
    const b = document.createElement('button'); b.textContent = act.label;
    b.onclick = ev => { ev.stopPropagation(); t.hidden = true; act.fn(); };
    t.appendChild(b);
  }
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, act ? 4500 : 2600);
}

function locate() {
  const btn = $('#meBtn');
  if (S.watch != null) {
    navigator.geolocation.clearWatch(S.watch); S.watch = null; S.me = null; btn.classList.remove('on');
    if (meMarker) { meMarker.remove(); meMarker = null; }
    render(); return;
  }
  if (!navigator.geolocation) return toast('הדפדפן לא נותן מיקום');
  toast('מחפש איפה אתם...');
  let first = true;
  S.watch = navigator.geolocation.watchPosition(pos => {
    const me = { lat: pos.coords.latitude, lng: pos.coords.longitude };
    if (dist(me, { lat: CENTER[1], lng: CENTER[0] }) > 15000) {
      if (first) toast('אתם רחוקים מהפסטיבל, אז אין טעם במרחקי הליכה. בפרדס חנה זה יעבוד.');
      navigator.geolocation.clearWatch(S.watch); S.watch = null; return;
    }
    S.me = me; btn.classList.add('on'); drawMe();
    if (first) { map.flyTo({ center: [me.lng, me.lat], zoom: 15, offset: off(fitPad()) }); toast('הרשימה מסודרת עכשיו לפי מה שהכי קרוב'); }
    first = false; render();
  }, () => { toast('לא קיבלנו מיקום. אפשר לאשר אותו בהגדרות הדפדפן.'); S.watch = null; }, { enableHighAccuracy: true, maximumAge: 30000 });
}

/* Sharing sends the state on screen as a short link. The long address goes to is.gd, which
   answers browsers directly and keeps the part after # through its redirect; if it does not
   answer, the full address is shared instead. The plan joins only when asked for. */
const shortened = new Map();
async function shorten(url) {
  if (shortened.has(url)) return shortened.get(url);
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 6000);
  try {
    const r = await fetch('https://is.gd/create.php?format=json&url=' + encodeURIComponent(url), { signal: ctl.signal });
    const j = await r.json();
    if (!j.shorturl) throw new Error(j.errormessage || 'no link');
    shortened.set(url, j.shorturl);
    return j.shorturl;
  } finally { clearTimeout(timer); }
}
function shareLink(withPlan) {
  const p = stateParams();
  if (withPlan && S.plan.size) { p.set('plan', [...S.plan].join(',')); p.set('v', 'plan'); }
  return `${location.origin}${location.pathname}#${query(p)}`;
}
let shareSeq = 0;
function share(withPlan = S.view === 'plan' && S.plan.size > 0) {
  toggleShowPanel(false);
  const box = $('#sharePanel');
  box.hidden = false;
  box.innerHTML = `<div class="sphead"><b>שיתוף</b><button class="iconbtn" data-shareclose aria-label="סגירה">${ICON.x}</button></div>
    <p class="shnote">מי שיפתח את הקישור יראה בדיוק את מה שעל המסך שלך: ${S.follow ? 'השעה שלו עכשיו' : `${dayName(S.day, true)} ב-${hm(S.t)}`}, התצוגה, המפה, הסינון ומה שמוסתר.</p>
    ${S.plan.size ? `<label class="sw"><span>לצרף את התוכנית שלי (${S.plan.size})</span><input type="checkbox" data-shareplan ${withPlan ? 'checked' : ''}><i></i></label>` : ''}
    <div class="shlink"><input id="shareUrl" readonly dir="ltr" value="מקצר את הקישור..."><button class="btn warm" data-copy>העתקה</button></div>
    <div class="shbtns">${navigator.share ? `<button class="btn" data-native>${ICON.share}שליחה</button>` : ''}<a class="btn wa" id="shareWa" target="_blank" rel="noopener">${ICON.wa}וואטסאפ</a></div>`;
  fillShare(withPlan);
}
async function fillShare(withPlan) {
  const seq = ++shareSeq, long = shareLink(withPlan);
  let url = long;
  try { url = await shorten(long); } catch (e) { /* the long one works too */ }
  if (seq !== shareSeq || $('#sharePanel').hidden) return;
  const inp = $('#shareUrl'); inp.value = url; inp.dataset.plan = withPlan ? '1' : '';
  const text = withPlan ? `התוכנית שלי לפסטיבל דרך הנדיב (${S.plan.size} אירועים)` : 'דרך הנדיב · מה קורה עכשיו';
  $('#shareWa').href = 'https://wa.me/?text=' + encodeURIComponent(`${text}\n${url}`);
}
function closeShare() { shareSeq++; $('#sharePanel').hidden = true; }
// a host can send their place's own schedule to their guests
async function sharePlace(v) {
  let url = `${location.origin}${location.pathname}#place=${encodeURIComponent(v.key)}`;
  try { url = await shorten(url); } catch (e) { /* the long one works too */ }
  return shareUrl(url, `${v.title} בפסטיבל דרך הנדיב: כל האירועים`);
}
async function shareUrl(url, text) {
  if (navigator.share) { try { await navigator.share({ title: text, url }); return; } catch (e) { if (e.name === 'AbortError') return; } }
  try { await navigator.clipboard.writeText(url); toast('הקישור הועתק. אפשר להדביק בוואטסאפ'); }
  catch (e) { prompt('הקישור:', url); }
}

function setupEvents() {
  $('#sharePanel').addEventListener('change', ev => { if (ev.target.dataset.shareplan != null) { $('#shareUrl').value = 'מקצר את הקישור...'; fillShare(ev.target.checked); } });
  $('#showPanel').addEventListener('change', ev => {
    const k = ev.target.dataset.ui; if (!k) return;
    S.ui[k] = ev.target.checked; applyUI(); $('#showPanel').innerHTML = showPanelHTML();
  });
  document.addEventListener('click', ev => {
    const t = ev.target;
    if (t.closest('[data-clean]')) return setClean(!S.clean);
    if (t.closest('[data-gonow]')) { stopPlay(); return goNow(); }
    if (t.closest('[data-showclose]')) return toggleShowPanel(false);
    if (t.closest('[data-fold]')) { S.ui.list = !S.ui.list; return applyUI(); }
    if (t.closest('[data-shareclose]')) return closeShare();
    if (t.closest('[data-copy]')) {
      const u = $('#shareUrl').value;
      return navigator.clipboard.writeText(u).then(() => toast('הקישור הועתק'), () => { $('#shareUrl').select(); });
    }
    if (t.closest('[data-native]')) {
      const u = $('#shareUrl').value;
      return navigator.share({ title: 'דרך הנדיב · מה קורה עכשיו', url: u }).catch(() => {});
    }
    if (t.closest('[data-qgo]')) return qGo();
    if (t.closest('[data-qclear]')) { $('#q').value = ''; S.q = ''; return filtersChanged(); }
    if (t.closest('[data-clearfilt]')) { S.filt.clear(); S.kids = S.access = false; return filtersChanged(); }
    if (t.closest('[data-calclose]')) return closeCal();
    const ga = t.closest('[data-gadd]'); if (ga) return gcalAdded(ga.dataset.gadd);
    if (t.closest('[data-join]')) return;
    if (t.closest('[data-ics-all]')) return downloadIcs(planList(), 'דרך הנדיב - התוכנית שלי');
    if (t.closest('#sharePanel, #calPanel')) return;
    if (!$('#sharePanel').hidden && !t.closest('[data-share]')) closeShare();
    if (!$('#calPanel').hidden && !t.closest('[data-cal]')) closeCal();
    if (t.closest('[data-cal]')) return openCal();
    if (t.closest('[data-uireset]')) { S.ui = { ...UI_DEF }; applyUI(); return toggleShowPanel(true); }
    if (!$('#showPanel').hidden && !t.closest('#showPanel')) toggleShowPanel(false);
    const star = t.closest('[data-star]');
    if (star) { ev.stopPropagation(); toggleStar(star.dataset.star); redrawDetail(); return; }
    const gp = t.closest('[data-gap]'); if (gp) { S.openGap = S.openGap === gp.dataset.gap ? null : gp.dataset.gap; return render(); }
    const pl = t.closest('[data-place]'); if (pl) return openPlace(S.venues.get(pl.dataset.place));
    const pm = t.closest('[data-placemap]');
    if (pm) { const v = S.venues.get(pm.dataset.placemap); closeDetail(); setView('map'); return openVenue(v, true); }
    if (t.closest('[data-shareplace]')) return sharePlace(S.detail.place);
    if (t.closest('[data-close]')) return closeDetail();
    const ics = t.closest('[data-ics]'); if (ics) { const e = S.byId.get(ics.dataset.ics); return downloadIcs([e], e.name.slice(0, 40)); }
    if (t.closest('[data-ics-all]')) return downloadIcs(planList(), 'דרך הנדיב - התוכנית שלי');
    if (t.closest('[data-share]')) return t.closest('#shareBtn') && !$('#sharePanel').hidden ? closeShare() : share();
    if (t.closest('[data-clear-plan]')) { if (confirm('לנקות את כל התוכנית?')) { S.plan.clear(); savePlan(); render(); redrawAll(); } return; }
    if (t.closest('[data-import]')) { S.shared.forEach(id => S.plan.add(id)); S.shared = null; savePlan(); toast('התוכנית נוספה'); setHash(); redrawAll(); return render(); }
    if (t.closest('[data-dismiss]')) { S.shared = null; return render(); }
    if (t.closest('[data-onlymine]')) { S.onlyMine = !S.onlyMine; return render(); }
    if (t.closest('[data-more]')) { S.moreLater = true; return render(); }
    if (t.closest('[data-clear]')) { S.filt.clear(); S.kids = S.access = false; S.q = ''; $('#q').value = ''; return filtersChanged(); }
    const sm = t.closest('[data-showmap]');
    if (sm) { const e = S.byId.get(sm.dataset.showmap); closeDetail(); if (e.date !== S.day) setDay(e.date); setT(e.start); return focusEvent(e); }
    const vn = t.closest('[data-venue]');
    if (vn) { const v = S.venues.get(vn.dataset.venue); setView('map'); return v.geo && openVenue(v, true); }
    const det = t.closest('[data-detail]'); if (det) return openDetail(S.byId.get(det.dataset.detail));
    if (t.closest('[data-venuelist]')) { S.focus = null; S.popup.setHTML(venueHTML(S.sel)); markFocus(); drawMarkers(); return requestAnimationFrame(fitPopup); }
    const evEl = t.closest('[data-ev]');
    if (evEl) {
      const e = S.byId.get(evEl.dataset.ev);
      // the list and the popups point at the map; the timetable and the full card open the card
      return t.closest('#sheetBody, .maplibregl-popup') && S.view !== 'grid' ? focusEvent(e) : openDetail(e);
    }
    const day = t.closest('[data-day]'); if (day) return setDay(day.dataset.day);
    const g = t.closest('[data-g]');
    if (g) { const n = g.dataset.g; S.filt.has(n) ? S.filt.delete(n) : S.filt.add(n); return filtersChanged(); }
    if (t.closest('[data-kids]')) { S.kids = !S.kids; return filtersChanged(); }
    if (t.closest('[data-access]')) { S.access = !S.access; return filtersChanged(); }
    const view = t.closest('[data-view]'); if (view) return setView(view.dataset.view);
  });
  document.addEventListener('keydown', ev => {
    if (ev.key !== 'Escape') return;
    if (S.detail) closeDetail();
    else if (!$('#sharePanel').hidden) closeShare();
    else if (!$('#calPanel').hidden) closeCal();
    else if (!$('#showPanel').hidden) toggleShowPanel(false);
    else if (S.clean) setClean(false);
  });
  // pointing at a card lights up its place on the map
  const hl = on => ev => {
    const c = ev.target.closest('#sheetBody .ev'); if (!c) return;
    const e = S.byId.get(c.dataset.ev), m = e && markers.get(e.vkey);
    if (m) m.el.classList.toggle('hl', on);
  };
  $('#sheetBody').addEventListener('mouseover', hl(true));
  $('#sheetBody').addEventListener('mouseout', hl(false));
  $('#playBtn').onclick = play;
  $('#nowBtn').onclick = () => { stopPlay(); goNow(); };
  $('#meBtn').onclick = locate;
  $('#searchBtn').onclick = () => {
    const row = $('#searchRow'); row.hidden = !row.hidden;
    $('#searchBtn').classList.toggle('on', !row.hidden);
    if (!row.hidden) $('#q').focus(); else if (S.q) { S.q = ''; $('#q').value = ''; filtersChanged(); }
  };
  let qt;
  $('#q').addEventListener('input', () => { clearTimeout(qt); qt = setTimeout(() => { S.q = $('#q').value.trim(); filtersChanged(); }, 180); });
  $('#q').addEventListener('keydown', ev => {
    if (ev.key !== 'Enter') return;
    clearTimeout(qt); S.q = $('#q').value.trim(); filtersChanged(); qGo();
  });
}

/* ---------- start ---------- */

async function main() {
  let doc;
  try { doc = await load(); }
  catch (e) { $('#sheetBody').innerHTML = '<div class="empty">לא הצלחנו לטעון את התוכנייה. כדאי לנסות שוב בעוד רגע.</div>'; return; }
  prepare(doc);
  try { (JSON.parse(localStorage.getItem('hanadiv-plan')) || []).forEach(id => S.byId.has(id) && S.plan.add(id)); } catch (e) { /* none */ }
  savePlan();
  const h = readHash();
  const now = festivalNow();
  S.day = h.day || (now && now.date) || S.days[0];
  // a shared link keeps its time; otherwise the app opens on the real time of day, following it
  S.follow = h.t == null;
  S.t = h.t != null ? h.t : Math.round(clamp(israelNow().m, S.lo, S.hi) / STEP) * STEP;
  if (h.plan && h.plan.length && h.plan.some(id => !S.plan.has(id))) S.shared = h.plan;
  S.init = h;
  h.hide.forEach(k => S.ui[k] = false);
  h.g.forEach(g => S.filt.add(g));
  S.kids = h.kids; S.access = h.acc; S.onlyMine = !h.all;
  if (h.q) { S.q = h.q; $('#q').value = h.q; $('#searchRow').hidden = false; $('#searchBtn').classList.add('on'); drawQInfo(); }
  S.sat = h.sat; S.tilt = h.tilt || !!(h.cam && h.cam.pitch >= 10); S.cam = h.cam;
  // map-only from a link: set before the map frames itself (the browser's full screen needs a tap, so not that)
  S.clean = h.full && !S.shared && (h.view || 'map') === 'map';
  $('#sub').textContent = `פסטיבל ${dayName(S.days[0]).split(' ')[1]}–${dayName(S.days.at(-1)).split(' ')[1]} · ${S.events.length} אירועים ב-${S.venues.size} מקומות`;
  setupEvents(); setupStrip(); setupSheet(); setupMap();
  drawDays(); drawChips(); drawStrip();
  setView(S.shared ? 'plan' : h.view && ['map', 'grid', 'plan'].includes(h.view) ? h.view : 'map');
  if (innerWidth < 900) snap(h.sh || 'mid');
  if (h.e && S.byId.has(h.e)) openDetail(S.byId.get(h.e));
  else if (h.place && S.venues.has(h.place)) openPlace(S.venues.get(h.place));
  setInterval(() => { if (S.follow && !S.playing) goNow(true); else render(); drawDays(); }, 30000);
  if (typeof syncBoot === 'function') syncBoot();  // the real clock under a look-ahead ticks too
}
main();
