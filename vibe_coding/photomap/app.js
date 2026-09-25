// מפת התמונות: קורא את data/library.json (ואת trips.json אם יש) ומציג כל פריט במקום שצולם.
// כל הזמנים כאן הם "זמן מקומי במקום הצילום" (taken + tz), ומוצגים כאילו הם UTC.

const NEAR_METERS = 60;  // לחיצה על סמן בודד פותחת גם את מה שצולם בטווח הזה ממנו

const map = L.map('map', { zoomControl: false, worldCopyJump: true });
L.control.zoom({ position: 'bottomleft' }).addTo(map);
const layers = {
  'מפה': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '© OpenStreetMap' }),
  'לוויין': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19, attribution: 'Imagery © Esri' }),
};
layers['מפה'].addTo(map);
L.control.layers(layers, null, { position: 'bottomleft' }).addTo(map);

const cluster = L.markerClusterGroup({
  maxClusterRadius: 70,
  showCoverageOnHover: false,
  zoomToBoundsOnClick: false,
  spiderfyOnMaxZoom: false,
  iconCreateFunction: c => {
    const items = c.getAllChildMarkers().map(m => m.item).sort(byTime);
    return pinIcon(items[items.length - 1], c.getChildCount());
  },
});
map.addLayer(cluster);
const routeLayer = L.layerGroup().addTo(map);

let lib, trips = [];
const markers = new Map();        // id -> marker, נוצרים פעם אחת
const hiddenAlbums = new Set();
let range = null;                 // [from, to] בזמן מקומי, או null = הכל
let activeTrip = null;

// ---------- עזרים ----------

const localTime = it => it.taken + it.tz;
const byTime = (a, b) => a.taken - b.taken;
const MONTH_FMT = { timeZone: 'UTC', month: 'long', year: 'numeric' };
const DAY_FMT = { timeZone: 'UTC', day: 'numeric', month: 'long', year: 'numeric' };
const fmt = (t, o) => new Date(t).toLocaleString('he-IL', o);

function fmtDate(it) {
  return fmt(localTime(it), { ...DAY_FMT, hour: '2-digit', minute: '2-digit' });
}

function fmtSpan(a, b) {
  const da = fmt(a, DAY_FMT), db = fmt(b, DAY_FMT);
  if (da === db) return da;
  const A = new Date(a), B = new Date(b);
  if (A.getUTCFullYear() === B.getUTCFullYear() && A.getUTCMonth() === B.getUTCMonth())
    return `${A.getUTCDate()}–${B.getUTCDate()} ב${fmt(b, { timeZone: 'UTC', month: 'long' })} ${B.getUTCFullYear()}`;
  return `${da} – ${db}`;
}

function pinIcon(it, count = 1) {
  const cls = ['pin', it.type === 'video' && count === 1 ? 'video' : '', count > 1 ? 'stack' : ''].join(' ');
  const badge = count > 1 ? `<span class="count">${count}</span>` : '';
  return L.divIcon({
    html: `<div class="${cls}" style="background-image:url('${it.thumb}')">${badge}</div>`,
    className: '', iconSize: [52, 52], iconAnchor: [26, 26],
  });
}

const inAlbums = () => Object.values(lib.items).filter(i => !hiddenAlbums.has(i.album));
const inRange = it => !range || (localTime(it) >= range[0] && localTime(it) <= range[1]);
const visible = () => inAlbums().filter(inRange);

// ---------- מפה ----------

function markerFor(it) {
  if (!markers.has(it.id)) {
    const m = L.marker([it.lat, it.lon], { icon: pinIcon(it) });
    m.item = it;
    m.on('click', () => {
      const here = L.latLng(it.lat, it.lon);
      const near = visible().filter(o => o.lat != null && here.distanceTo([o.lat, o.lon]) < NEAR_METERS).sort(byTime);
      openLightbox(near, near.indexOf(it));
    });
    markers.set(it.id, m);
  }
  return markers.get(it.id);
}

function render() {
  cluster.clearLayers();
  cluster.addLayers(visible().filter(i => i.lat != null).map(markerFor));
  timeline.draw();
}

// לחיצה על קבוצה תמיד פותחת את כל התמונות שבה. מתפצלים רק בזום של המפה
cluster.on('clusterclick', e => {
  openLightbox(e.layer.getAllChildMarkers().map(m => m.item).sort(byTime), 0);
});

function fitTo(items) {
  const pts = items.filter(i => i.lat != null).map(i => [i.lat, i.lon]);
  if (pts.length) map.fitBounds(pts, { padding: [70, 70], maxZoom: 16 });
}

// ---------- טיולים ----------

function drawRoute(items) {
  routeLayer.clearLayers();
  // מדלגים על נקודות צפופות, אחרת עשר תמונות מאותו מקום הופכות את הקו לקשקוש
  const pts = [];
  for (const it of items) {
    const p = L.latLng(it.lat, it.lon);
    if (!pts.length || pts[pts.length - 1].distanceTo(p) > 25) pts.push(p);
  }
  if (pts.length < 2) return;
  L.polyline(pts, { color: '#fff', weight: 6, opacity: .7, interactive: false }).addTo(routeLayer);
  L.polyline(pts, { color: getComputedStyle(document.body).getPropertyValue('--accent'), weight: 3,
                    opacity: .85, interactive: false }).addTo(routeLayer);
  L.circleMarker(pts[0], { radius: 7, color: '#fff', weight: 3, fillColor: '#2a9d5c', fillOpacity: 1 })
    .bindTooltip('התחלה').addTo(routeLayer);
}

function selectTrip(trip) {
  if (activeTrip === trip) return clearTrip();
  activeTrip = trip;
  const items = trip.items.map(id => lib.items[id]).filter(Boolean).sort(byTime);
  range = [trip.start - 60e3, trip.end + 60e3];
  drawRoute(items);
  fitTo(items);
  render();
  renderTrips();
}

function clearTrip() {
  activeTrip = null;
  routeLayer.clearLayers();
  range = null;
  render();
  renderTrips();
}

let tripKind = 'trip';

function renderTrips() {
  const ul = document.getElementById('trips');
  const tabs = document.getElementById('trip-tabs');
  const count = k => trips.filter(t => (t.kind || 'trip') === k).length;
  tabs.hidden = false;
  tabs.querySelectorAll('button').forEach(b => {
    b.textContent = `${b.dataset.kind === 'trip' ? 'טיולים' : 'יציאות יום'} (${count(b.dataset.kind)})`;
    b.classList.toggle('on', b.dataset.kind === tripKind);
    b.onclick = () => { tripKind = b.dataset.kind; renderTrips(); };
  });
  ul.innerHTML = '';
  const shown = trips.filter(t => (t.kind || 'trip') === tripKind);
  if (!shown.length) {
    ul.innerHTML = '<li class="hint">עוד לא זוהו. הם יופיעו כשיהיו יותר תמונות.</li>';
    return;
  }
  shown.forEach(trip => {
    const cover = lib.items[trip.cover];
    const li = document.createElement('li');
    li.classList.toggle('on', trip === activeTrip);
    const meta = [fmtSpan(trip.start, trip.end), `${trip.count} פריטים`];
    if (trip.km >= 2) meta.push(`${trip.km} ק"מ`);
    li.innerHTML = `<img src="${cover?.thumb ?? ''}" alt="" loading="lazy"><div class="name"><b></b><small></small></div>`;
    li.querySelector('b').textContent = trip.name;
    li.querySelector('small').textContent = meta.join(' · ');
    li.onclick = () => selectTrip(trip);
    ul.appendChild(li);
  });
}

// ---------- פאנל ----------

function renderPanel() {
  const all = Object.values(lib.items);
  const located = all.filter(i => i.lat != null).length;
  const videos = all.filter(i => i.type === 'video').length;
  document.getElementById('stats').textContent =
    `${all.length - videos} תמונות · ${videos} סרטונים · ${located} על המפה`;

  const ul = document.getElementById('albums');
  ul.innerHTML = '';
  Object.values(lib.albums).forEach(al => {
    const items = al.items.map(id => lib.items[id]).filter(Boolean).sort(byTime);
    const n = items.filter(i => i.lat != null).length;
    const li = document.createElement('li');
    li.innerHTML = `<input type="checkbox" ${hiddenAlbums.has(al.id) ? '' : 'checked'} title="הצג/הסתר">
      <img src="${items[0]?.thumb ?? ''}" alt="">
      <button class="name"><b></b><small>${items.length} פריטים${n < items.length ? `, ${n} ממוקמים` : ''}</small></button>`;
    li.querySelector('b').textContent = al.title;
    li.querySelector('input').onchange = e => {
      e.target.checked ? hiddenAlbums.delete(al.id) : hiddenAlbums.add(al.id);
      render();
    };
    li.querySelector('.name').onclick = () => fitTo(items);
    ul.appendChild(li);
  });

  const lost = all.filter(i => i.lat == null).sort(byTime);
  const btn = document.getElementById('unlocated');
  btn.hidden = !lost.length;
  btn.textContent = `${lost.length} בלי מיקום`;
  btn.onclick = () => openLightbox(lost, 0);
}

document.getElementById('collapse').onclick = () => document.getElementById('panel').classList.toggle('collapsed');

// ---------- ציר זמן ----------
// היסטוגרמה של כמות הפריטים לאורך הזמן, עם בחירת טווח בגרירה.
// הזמן זורם מימין לשמאל, כמו הקריאה בעברית וכמו החצים בגלריה.

const UNITS = {
  hour:  { floor: d => d.setUTCMinutes(0, 0, 0),         next: d => d.setUTCHours(d.getUTCHours() + 1) },
  day:   { floor: d => d.setUTCHours(0, 0, 0, 0),         next: d => d.setUTCDate(d.getUTCDate() + 1) },
  month: { floor: d => (d.setUTCHours(0, 0, 0, 0), d.setUTCDate(1)), next: d => d.setUTCMonth(d.getUTCMonth() + 1) },
  year:  { floor: d => (d.setUTCHours(0, 0, 0, 0), d.setUTCMonth(0, 1)), next: d => d.setUTCFullYear(d.getUTCFullYear() + 1) },
};
const LABEL_OF = { hour: 'day', day: 'month', month: 'year', year: 'year' };
const floorT = (t, u) => { const d = new Date(t); UNITS[u].floor(d); return +d; };
const nextT = (t, u) => { const d = new Date(t); UNITS[u].next(d); return +d; };

function edges(t0, t1, u) {
  const out = [floorT(t0, u)];
  while (out[out.length - 1] <= t1) out.push(nextT(out[out.length - 1], u));
  return out;
}

const timeline = (() => {
  const canvas = document.getElementById('tl-canvas');
  const label = document.getElementById('tl-label');
  const reset = document.getElementById('tl-reset');
  const ctx = canvas.getContext('2d');
  let T0 = 0, T1 = 1, W = 0, H = 0, bins = [], unit = 'day', first = 0, last = 0;
  const AXIS = 16;

  const x = t => W * (1 - (t - T0) / (T1 - T0));
  const t = px => T0 + (1 - px / W) * (T1 - T0);
  const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();

  function layout(items) {
    const times = items.map(localTime);
    if (!times.length) { bins = []; return; }
    const lo = Math.min(...times), hi = Math.max(...times);
    first = lo; last = hi;
    unit = ['hour', 'day', 'month', 'year'].find(u => edges(lo, hi, u).length <= 150) || 'year';
    const e = edges(lo, hi, unit);
    T0 = e[0]; T1 = e[e.length - 1];
    bins = e.slice(0, -1).map((a, i) => ({ a, b: e[i + 1], n: 0 }));
    for (const tm of times) {
      let lo2 = 0, hi2 = bins.length - 1;
      while (lo2 < hi2) { const m = (lo2 + hi2 + 1) >> 1; bins[m].a <= tm ? lo2 = m : hi2 = m - 1; }
      bins[lo2].n++;
    }
  }

  function draw() {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const items = inAlbums().filter(i => i.lat != null);   // רק מה שעל המפה
    layout(items);

    // תווית
    const shown = items.filter(inRange).length;
    if (!bins.length) label.textContent = '';
    else {
      const [a, b] = range || [first, last];
      const span = unit === 'hour' && fmt(a, DAY_FMT) === fmt(b, DAY_FMT)
        ? `${fmt(a, DAY_FMT)}, ${fmt(a, { timeZone: 'UTC', hour: '2-digit', minute: '2-digit' })}–${fmt(b, { timeZone: 'UTC', hour: '2-digit', minute: '2-digit' })}`
        : fmtSpan(a, b);
      label.innerHTML = `<span></span><small>${shown} פריטים</small>`;
      label.firstChild.textContent = activeTrip ? `${activeTrip.name}: ${span}` : range ? span : `כל התקופה: ${span}`;
    }
    reset.hidden = !range;
    if (!bins.length) return;

    // עמודות. שורש ריבועי, כדי שחודש עם 5 תמונות לא ייעלם ליד חודש עם 500
    const max = Math.sqrt(Math.max(...bins.map(b => b.n)));
    const hBars = H - AXIS - 2, accent = css('--accent'), bar = css('--bar');
    for (const b of bins) {
      if (!b.n) continue;
      const x1 = x(b.b), x0 = x(b.a), h = Math.max(2, hBars * Math.sqrt(b.n) / max);
      const inside = !range || (b.b > range[0] && b.a <= range[1]);
      ctx.fillStyle = inside ? accent : bar;
      ctx.fillRect(x1 + .5, hBars - h + 2, Math.max(1, x0 - x1 - 1), h);
    }

    // טווח נבחר
    if (range) {
      const xa = x(range[0]), xb = x(range[1]);
      ctx.fillStyle = css('--bg') + '99';
      ctx.fillRect(xa, 0, W - xa, hBars + 2);
      ctx.fillRect(0, 0, xb, hBars + 2);
      ctx.fillStyle = css('--fg');
      for (const hx of [xa, xb]) ctx.fillRect(hx - 1, 0, 2, hBars + 2);
    }

    // ציר
    ctx.fillStyle = css('--muted');
    ctx.font = '11px Heebo, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    // תוויות ביחידה הגדולה הבאה (שנים מעל חודשים), ואם יוצאות פחות משלוש, ביחידה של העמודות
    let lu = LABEL_OF[unit];
    if (edges(T0, T1 - 1, lu).filter(e => e >= T0 && e < T1).length < 3) lu = unit;
    const lfmt = { hour: { timeZone: 'UTC', hour: '2-digit', minute: '2-digit' },
                   day: { timeZone: 'UTC', day: 'numeric', month: 'short' },
                   month: { timeZone: 'UTC', month: 'short', year: '2-digit' },
                   year: { timeZone: 'UTC', year: 'numeric' } }[lu];
    let lastX = Infinity;
    for (const e of edges(T0, T1 - 1, lu)) {
      const ex = x(Math.max(e, T0));
      if (lastX - ex < 64 || ex < 20) continue;
      ctx.fillRect(ex, hBars + 2, 1, 4);
      ctx.fillText(fmt(Math.max(e, T0), lfmt), ex - 3, H);
      lastX = ex;
    }
  }

  // גרירה: על ידית = שינוי קצה, בתוך הטווח = הזזה, בחוץ = טווח חדש. לחיצה בלי גרירה = העמודה שמתחת
  let drag = null;
  canvas.addEventListener('pointerdown', e => {
    if (!bins.length) return;
    canvas.setPointerCapture(e.pointerId);
    const px = e.offsetX, tp = t(px);
    if (range && Math.abs(px - x(range[0])) < 8) drag = { mode: 'from' };
    else if (range && Math.abs(px - x(range[1])) < 8) drag = { mode: 'to' };
    else if (range && tp > range[0] && tp < range[1]) drag = { mode: 'move', t0: tp, r0: [...range] };
    else drag = { mode: 'new', t0: tp };
    drag.x0 = px;
  });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return;
    const tp = Math.min(T1, Math.max(T0, t(e.offsetX)));
    if (drag.mode === 'new') { if (Math.abs(e.offsetX - drag.x0) < 4) return; range = [Math.min(drag.t0, tp), Math.max(drag.t0, tp)]; }
    else if (drag.mode === 'from') range = [Math.min(tp, range[1]), range[1]];
    else if (drag.mode === 'to') range = [range[0], Math.max(tp, range[0])];
    else {
      const d = Math.min(T1 - drag.r0[1], Math.max(T0 - drag.r0[0], tp - drag.t0));
      range = [drag.r0[0] + d, drag.r0[1] + d];
    }
    leaveTrip();
    draw();
  });
  canvas.addEventListener('pointerup', e => {
    if (!drag) return;
    if (drag.mode === 'new' && Math.abs(e.offsetX - drag.x0) < 4) {
      const tp = t(e.offsetX), b = bins.find(b => b.a <= tp && tp < b.b);
      if (b && b.n) { range = [b.a, b.b - 1]; leaveTrip(); }
    }
    drag = null;
    render();
    // אם אין בטווח החדש אף פריט בתצוגה הנוכחית, המפה עוברת אליהם
    const sel = visible().filter(i => i.lat != null);
    if (sel.length && !sel.some(i => map.getBounds().contains([i.lat, i.lon]))) fitTo(sel);
  });
  reset.onclick = () => (activeTrip ? clearTrip() : (range = null, render()));
  addEventListener('resize', draw);

  // שינוי ידני של הטווח מבטל את הטיול הנבחר, אבל לא את הטווח
  function leaveTrip() {
    if (!activeTrip) return;
    activeTrip = null;
    routeLayer.clearLayers();
    renderTrips();
  }

  return { draw };
})();

// ---------- גלריה ----------

const lb = document.getElementById('lightbox');
const stage = lb.querySelector('.lb-stage');
const caption = lb.querySelector('.lb-caption');
const strip = lb.querySelector('.lb-strip');
let lbItems = [], lbIndex = 0;

function openLightbox(items, index) {
  lbItems = items;
  strip.innerHTML = '';
  strip.hidden = items.length < 2;
  items.forEach((it, i) => {
    const t = document.createElement('img');
    t.src = it.thumb; t.loading = 'lazy';
    t.onclick = () => show(i);
    strip.appendChild(t);
  });
  lb.hidden = false;
  show(Math.max(0, index));
}

function show(i) {
  lbIndex = i;
  const it = lbItems[i];
  viewer.close();
  if (it.type === 'video') stage.innerHTML = `<video src="${it.video}" poster="${it.large}" controls autoplay playsinline></video>`;
  else viewer.open(it);
  const album = lib.albums[it.album];
  caption.innerHTML = `${fmtDate(it)}<small></small>`;
  const where = it.folders?.length ? it.folders.join(', ') : album?.title ?? '';
  caption.querySelector('small').textContent = `${where} · ${i + 1}/${lbItems.length}`;
  lb.querySelector('.lb-prev').disabled = i === 0;
  lb.querySelector('.lb-next').disabled = i === lbItems.length - 1;
  strip.querySelector('.on')?.classList.remove('on');
  strip.children[i]?.classList.add('on');
  strip.children[i]?.scrollIntoView({ block: 'nearest', inline: 'center' });
  if (it.lat != null) map.panTo([it.lat, it.lon], { animate: false });
}

function closeLightbox() {
  lb.hidden = true;
  viewer.close();
  stage.innerHTML = '';  // עוצר סרטון שמתנגן
}

// ---------- צופה עם זום ----------
// התמונה מצוירת על canvas, רק החלק שנראה על המסך, כך שגם זום של פי מאות נשאר מהיר.
// מעבר ל-PIXELATED פיקסלים של מסך לכל פיקסל של תמונה מכבים את ההחלקה, והפיקסלים נראים כריבועים.
// המצב נשמר ביחידות של "התמונה בגודל המותאם למסך", ולכן אפשר להחליף באמצע לגרסה ברזולוציה מלאה.

const viewer = (() => {
  const MAX_PX = 80;       // בזום המקסימלי, פיקסל אחד של התמונה = 80 פיקסלים של מסך
  const PIXELATED = 3;
  let canvas, ctx, badge, img, item, W, H, fw, fh, s, tx, ty, full, badgeTimer, drag, dragged = false;

  function open(it) {
    item = it;
    stage.innerHTML = '<canvas class="lb-canvas"></canvas><div class="lb-zoom" hidden></div><div class="lb-loading">טוען…</div>';
    canvas = stage.querySelector('canvas');
    badge = stage.querySelector('.lb-zoom');
    ctx = canvas.getContext('2d');
    load(it.large, () => { stage.querySelector('.lb-loading')?.remove(); fit(); });
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('dblclick', onDbl);
    canvas.addEventListener('click', onClick);
  }

  function load(src, done) {
    const im = new Image();
    im.onload = () => { if (item && (src === item.large || src === item.orig)) { img = im; done?.(); draw(); } };
    im.src = src;
  }

  function close() {
    item = img = null;
    full = false;
  }

  // מידות מקור אמיתיות: אם יש גרסה מלאה בדרך, לפיה, כדי שהזום המקסימלי יגיע לפיקסלים האמיתיים
  const naturalW = () => (item.orig && item.w) || img.naturalWidth;

  function fit() {
    const r = stage.getBoundingClientRect();
    W = r.width; H = r.height;
    const dpr = devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    const k = Math.min(W / img.naturalWidth, H / img.naturalHeight, 1);
    fw = img.naturalWidth * k; fh = img.naturalHeight * k;
    s = 1; tx = (W - fw) / 2; ty = (H - fh) / 2;
  }

  function clampPan() {
    const w = fw * s, h = fh * s;
    tx = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, tx));
    ty = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, ty));
  }

  function draw() {
    if (!img || !canvas) return;
    const dpr = devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const w = fw * s, h = fh * s, nw = img.naturalWidth, nh = img.naturalHeight;
    // רק החלק הנראה: מלבן היעד על המסך, ומלבן המקור המתאים בתמונה
    const x0 = Math.max(0, tx), y0 = Math.max(0, ty), x1 = Math.min(W, tx + w), y1 = Math.min(H, ty + h);
    if (x1 <= x0 || y1 <= y0) return;
    const pxPerPixel = w / nw * dpr;
    ctx.imageSmoothingEnabled = pxPerPixel < PIXELATED;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, (x0 - tx) / w * nw, (y0 - ty) / h * nh, (x1 - x0) / w * nw, (y1 - y0) / h * nh,
                  x0, y0, x1 - x0, y1 - y0);
    // כשמתקרבים מעבר לרזולוציה של הגרסה שנטענה, מביאים את המקור המלא (אם יש)
    if (!full && item.orig && w / nw > 1) { full = true; load(item.orig); }
  }

  function zoomAt(cx, cy, next) {
    const max = MAX_PX * naturalW() / fw;
    next = Math.min(max, Math.max(1, next));
    tx = cx - (cx - tx) * next / s;
    ty = cy - (cy - ty) * next / s;
    s = next;
    clampPan();
    draw();
    showBadge();
  }

  function showBadge() {
    const pct = Math.round(fw * s / naturalW() * 100);
    badge.textContent = s === 1 ? 'מותאם למסך' : `${pct.toLocaleString('he-IL')}%`;
    badge.hidden = false;
    clearTimeout(badgeTimer);
    badgeTimer = setTimeout(() => { badge.hidden = true; }, 1200);
    canvas.style.cursor = s > 1 ? 'grab' : '';
  }

  function onWheel(e) {
    e.preventDefault();
    if (!img) return;
    const d = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;   // שורות -> פיקסלים
    zoomAt(e.offsetX, e.offsetY, s * Math.exp(-d * 0.0022));
  }

  function onDbl(e) {
    if (!img) return;
    // לחיצה כפולה: מהתאמה למסך לגודל אמיתי (100%), ומכל זום אחר חזרה להתאמה
    zoomAt(e.offsetX, e.offsetY, s > 1.01 ? 1 : Math.max(2, naturalW() / fw));
  }

  function onDown(e) {
    dragged = false;
    if (!img) return;
    drag = { x: e.clientX, y: e.clientY, tx, ty };
    canvas.setPointerCapture(e.pointerId);
    canvas.onpointermove = ev => {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) dragged = true;
      if (s === 1) return;
      canvas.style.cursor = 'grabbing';
      tx = drag.tx + dx; ty = drag.ty + dy;
      clampPan();
      draw();
    };
    canvas.onpointerup = () => {
      canvas.onpointermove = canvas.onpointerup = null;
      if (s > 1) canvas.style.cursor = 'grab';
    };
  }

  // לחיצה בודדת מחוץ לתמונה (או בזמן טעינה), בלי גרירה ובלי זום, סוגרת
  function onClick(e) {
    if (dragged || e.detail !== 1) return;
    const outside = !img || e.offsetX < tx || e.offsetX > tx + fw * s || e.offsetY < ty || e.offsetY > ty + fh * s;
    if (outside && (!img || s === 1)) closeLightbox();
  }

  addEventListener('resize', () => { if (img && canvas) { fit(); draw(); } });
  return { open, close };
})();

lb.querySelector('.lb-close').onclick = closeLightbox;
lb.querySelector('.lb-prev').onclick = () => lbIndex > 0 && show(lbIndex - 1);
lb.querySelector('.lb-next').onclick = () => lbIndex < lbItems.length - 1 && show(lbIndex + 1);
stage.onclick = e => { if (e.target === stage) closeLightbox(); };   // סביב סרטון
document.addEventListener('keydown', e => {
  if (lb.hidden) return;
  if (e.key === 'Escape') closeLightbox();
  // בעברית "הבא" הוא שמאלה
  if (e.key === 'ArrowLeft' && lbIndex < lbItems.length - 1) show(lbIndex + 1);
  if (e.key === 'ArrowRight' && lbIndex > 0) show(lbIndex - 1);
});

// ---------- טעינה ----------

const getJSON = url => fetch(url, { cache: 'no-store' }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

getJSON('data/library.json')
  .then(async data => {
    lib = data;
    if (!Object.keys(lib.items).length) throw new Error('empty');
    trips = (await getJSON('data/trips.json').catch(() => ({ trips: [] }))).trips;
    renderPanel();
    renderTrips();
    render();
    fitTo(Object.values(lib.items));
  })
  .catch(() => {
    document.getElementById('empty').hidden = false;
    map.setView([31.5, 34.9], 7);
  });
