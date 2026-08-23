/* Arrange - moving places to where they actually are.
 *
 * build_places.py can only place a pardespedia article as well as its sources
 * allow, and those sources are thin: OpenStreetMap knows 442 named points in
 * the whole moshava, and not one house number in this town resolves, so most
 * positions are the middle of a road. A lot of them are simply wrong, and no
 * amount of further scripting fixes that - the information is in the head of
 * somebody who lives here.
 *
 * So this is a bulk correction tool, built for someone working through a few
 * hundred pins in one sitting rather than fixing one and starting again:
 *
 *   drag              pick a pin up and put it down. No mode, no dialog.
 *   edge auto-pan     hold a pin near the edge of the map and the map scrolls
 *                     under it, so the destination does not have to be on
 *                     screen when you start. This is the whole difficulty with
 *                     dragging on a map, and dropping-panning-repicking is
 *                     what makes the job unbearable.
 *   batched save      every move is held locally and written in one commit.
 *                     A commit per drag would be a round trip per drag.
 *   a queue           places with no position at all are handed over one at a
 *                     time, dropped in the middle of the map to be dragged.
 *
 * Each pin carries a link to its wiki article and a Google Maps search for its
 * name, because the fastest way to find out where a cafe really is, is to look
 * it up - Google's own positions for these are usually right, and there is no
 * free API that will hand them over.
 */
'use strict';

const Arrange = (() => {

  const EDGE = 88;              // px from the visible edge where auto-pan starts
  const PAN_MAX = 14;           // px per frame at the very edge
  const MAX_PINS = 220;         // beyond this the map is unusable anyway

  let on = false;
  let entries = new Map();      // place id -> {place, marker, el, lngLat}
  const moves = new Map();      // place id -> {lat, lng}, not yet written
  let drag = null;              // {entry, pointer:{x,y}}
  let raf = 0;
  let focused = null;           // the place whose details the bar is showing
  let queue = [];               // places with no position yet
  let dropped = 0;              // how many the queue has handed over since a save

  const layer = () => Layers.byId(Layers.PLACES_ID);

  /* ---------- entering and leaving ---------- */

  function open(place) {
    if (!map) { alert('הכלי דורש את המפה, והדפדפן הזה לא מציג אותה.'); return; }
    if (!Store.isEditor()) return;
    if (on) { if (place) focus(place, true); return; }

    stopNav();
    if (Drafts.isDrafting()) Drafts.stop();
    deselect();

    on = true;
    document.body.classList.add('arranging');
    el('arrange-bar').hidden = false;

    // The layer's own dots would sit under every marker we are about to draw.
    Layers.setArranging(true);
    if (!layer().on) { layer().on = true; Layers.applyVisibility(); }

    queue = layer().waypoints.filter((p) => p.unplaced);
    map.on('moveend', render);
    window.addEventListener('keydown', onKey);

    if (place && place.lat != null) {
      map.easeTo({ center: [place.lng, place.lat], zoom: Math.max(map.getZoom(), 17.5), duration: 500 });
    }
    render();
    focus(place || null, false);
    paintBar();
  }

  function close(force) {
    if (!on) return;
    if (!force && moves.size &&
        !confirm(`יש ${moves.size} מיקומים שעוד לא נשמרו. לצאת ולוותר עליהם?`)) return;
    endDrag(true);
    on = false;
    moves.clear();
    dropped = 0;
    clearMarkers();
    focused = null;
    document.body.classList.remove('arranging');
    el('arrange-bar').hidden = true;
    Layers.setArranging(false);
    map.off('moveend', render);
    window.removeEventListener('keydown', onKey);
  }

  function onKey(e) {
    if (e.key === 'Escape') close();
  }

  /* ---------- the pins ---------- */

  function clearMarkers() {
    entries.forEach((entry) => entry.marker.remove());
    entries = new Map();
  }

  /** Only what is on screen, plus anything moved this session so a pin you
   *  just placed cannot quietly vanish when the map shifts. */
  function inView() {
    const bounds = map.getBounds();
    const pad = 0.0025;
    const seen = new Set();
    const out = [];
    layer().waypoints.forEach((p) => {
      const at = moves.get(p.id) || (p.unplaced ? null : { lat: p.lat, lng: p.lng });
      if (!at) return;
      const near = at.lat > bounds.getSouth() - pad && at.lat < bounds.getNorth() + pad &&
                   at.lng > bounds.getWest() - pad && at.lng < bounds.getEast() + pad;
      if (near || moves.has(p.id)) {
        seen.add(p.id);
        out.push({ place: p, at });
      }
    });
    return { list: out.slice(0, MAX_PINS), clipped: out.length > MAX_PINS, seen };
  }

  function render() {
    if (!on || drag) return;                 // never rebuild under a live drag
    const { list, clipped } = inView();
    const want = new Set(list.map((r) => r.place.id));

    entries.forEach((entry, id) => {
      if (!want.has(id)) { entry.marker.remove(); entries.delete(id); }
    });

    list.forEach(({ place, at }) => {
      const existing = entries.get(place.id);
      if (existing) {
        existing.place = place;
        existing.marker.setLngLat([at.lng, at.lat]);
        return;
      }
      entries.set(place.id, make(place, at));
    });

    el('arrange-count').textContent = clipped
      ? `מציג ${MAX_PINS} מקומות, יש עוד. התקרב.` : '';
    paintBar();
  }

  function make(place, at) {
    const node = document.createElement('div');
    node.className = 'am' + (moves.has(place.id) ? ' moved' : '') +
      (place.approx ? ' approx' : '');
    node.style.setProperty('--c', place.color);
    node.innerHTML = `<i></i><b>${escapeHtml(place.name)}</b>`;
    node.title = place.name;

    // Only the dot is grabbable. The name label beside it doubles the width of
    // the element, and with pins this close together - a courtyard's worth of
    // businesses sit on one ring - the label of one pin covers the dot of the
    // next, so you reach for one and pick up its neighbour.
    const entry = {
      place, el: node, dot: node.querySelector('i'),
      lngLat: { lat: at.lat, lng: at.lng }, marker: null
    };
    entry.dot.addEventListener('pointerdown', (e) => beginDrag(entry, e));
    entry.marker = new maplibregl.Marker({ element: node, anchor: 'left' })
      .setLngLat([at.lng, at.lat]).addTo(map);
    return entry;
  }

  /* ---------- dragging ---------- */

  /* MapLibre's own draggable Marker is not enough here. It moves the pin only
   * when a pointermove arrives, so a map panning underneath a held pin leaves
   * the pin behind - and panning underneath a held pin is exactly the feature
   * that makes this usable. Driving the drag by hand also means one gesture
   * can cross the whole town. */
  function beginDrag(entry, e) {
    if (!on || e.button > 0) return;
    e.preventDefault();
    e.stopPropagation();
    entry.dot.setPointerCapture(e.pointerId);
    drag = { entry, pointer: { x: e.clientX, y: e.clientY }, id: e.pointerId, moved: false };

    map.dragPan.disable();
    entry.el.classList.add('drag');
    document.body.classList.add('am-dragging');

    entry.dot.addEventListener('pointermove', onPointerMove);
    entry.dot.addEventListener('pointerup', onPointerUp);
    entry.dot.addEventListener('pointercancel', onPointerUp);
    raf = requestAnimationFrame(tick);
    focus(entry.place, false);
  }

  function onPointerMove(e) {
    if (!drag || e.pointerId !== drag.id) return;
    if (Math.abs(e.clientX - drag.pointer.x) + Math.abs(e.clientY - drag.pointer.y) > 2) {
      drag.moved = true;
    }
    drag.pointer = { x: e.clientX, y: e.clientY };
    follow();
  }

  const onPointerUp = (e) => { if (drag && e.pointerId === drag.id) endDrag(false); };

  /** Put the pin wherever the pointer is now pointing at the ground. */
  function follow() {
    const rect = map.getCanvas().getBoundingClientRect();
    const point = [drag.pointer.x - rect.left, drag.pointer.y - rect.top];
    const at = map.unproject(point);
    drag.entry.lngLat = { lat: at.lat, lng: at.lng };
    drag.entry.marker.setLngLat(at);
  }

  /** The part of the map a person can actually see.
   *
   *  The canvas is the whole window, but the bar covers its top and the panel
   *  covers its bottom. Measuring the scroll zone from the canvas edge puts it
   *  underneath both of them, where the pointer can never reach - the first
   *  version did exactly that and the map never scrolled at all. */
  function usable() {
    const canvas = map.getCanvas().getBoundingClientRect();
    const bar = el('arrange-bar').getBoundingClientRect();
    const panel = el('panel').getBoundingClientRect();
    // On a phone the panel is a sheet across the bottom of the map. On a wide
    // screen it is a column beside it, which the canvas already excludes - and
    // clipping the bottom to a full-height panel's top edge would leave no
    // usable height at all, so the map would never scroll downwards.
    const overlapsX = panel.right > canvas.left + 1 && panel.left < canvas.right - 1;
    return {
      left: canvas.left,
      right: canvas.right,
      top: Math.max(canvas.top, bar.bottom),
      bottom: overlapsX ? Math.min(canvas.bottom, panel.top) : canvas.bottom,
      canvas
    };
  }

  /** While the pointer is held near an edge, scroll the map under it.
   *
   *  follow() runs again after every pan, so the pin stays under the pointer
   *  while the ground beneath moves: hold at the edge and the pin travels. */
  function tick() {
    if (!drag) return;
    const box = usable();

    // Scroll by time, not by frame. A phone rendering the map at fifteen
    // frames a second would otherwise scroll at a quarter of the speed of a
    // desktop, and the cap stops a stalled frame from throwing the pin across
    // the town when rendering catches up.
    const now = performance.now();
    const dt = Math.min(64, now - (drag.last || now));
    drag.last = now;

    // Distance past the start of the edge zone, negative towards the start of
    // the axis. Beyond the edge entirely - over the bar, over the panel - it
    // keeps growing, and the clamp in speed() caps it.
    const push = (v, lo, hi) => {
      if (v < lo + EDGE) return v - (lo + EDGE);
      if (v > hi - EDGE) return v - (hi - EDGE);
      return 0;
    };
    const speed = (d) =>
      Math.max(-PAN_MAX, Math.min(PAN_MAX, d * 0.32)) * (dt / 16.7);

    const dx = push(drag.pointer.x, box.left, box.right);
    const dy = push(drag.pointer.y, box.top, box.bottom);
    if (dx || dy) {
      map.panBy([speed(dx), speed(dy)], { duration: 0, animate: false });
      follow();
      drag.moved = true;
    }
    raf = requestAnimationFrame(tick);
  }

  function endDrag(silent) {
    if (!drag) return;
    cancelAnimationFrame(raf);
    const { entry } = drag;
    entry.dot.removeEventListener('pointermove', onPointerMove);
    entry.dot.removeEventListener('pointerup', onPointerUp);
    entry.dot.removeEventListener('pointercancel', onPointerUp);
    entry.el.classList.remove('drag');
    document.body.classList.remove('am-dragging');
    map.dragPan.enable();

    const wasDragged = drag.moved;
    drag = null;

    if (!silent && wasDragged) {
      moves.set(entry.place.id, {
        lat: +entry.lngLat.lat.toFixed(6),
        lng: +entry.lngLat.lng.toFixed(6)
      });
      entry.el.classList.add('moved');
      entry.el.classList.remove('approx');
    }
    if (!silent) paintBar();
  }

  /* ---------- the bar ---------- */

  function focus(place, recentre) {
    focused = place || null;
    if (place && recentre && place.lat != null && map) {
      map.easeTo({ center: [place.lng, place.lat], zoom: Math.max(map.getZoom(), 17.5) });
    }
    entries.forEach((entry) => entry.el.classList.toggle('on', !!place && entry.place.id === place.id));
    paintBar();
  }

  /* Google's own position for a small business is usually right, and there is
   * no free way to read it, so the practical answer is a search link and a
   * human. hl=iw so it opens in Hebrew. */
  const lookUp = (name) =>
    'https://www.google.com/maps/search/?api=1&query=' +
    encodeURIComponent(`${name} פרדס חנה כרכור`) + '&hl=iw';

  function paintBar() {
    const pending = moves.size;
    el('arrange-save').disabled = !pending;
    el('arrange-save').textContent = pending ? `שמור ${pending}` : 'שמור';

    const left = queue.filter((p) => !moves.has(p.id)).length;
    el('arrange-next').hidden = !left;
    el('arrange-next').textContent = `הנח מקום חסר (${left})`;

    if (focused) {
      el('arrange-name').textContent = focused.name;
      el('arrange-links').innerHTML =
        `<a href="${escapeHtml(focused.url)}" target="_blank" rel="noopener">הערך</a>` +
        `<a href="${lookUp(focused.name)}" target="_blank" rel="noopener">חפש בגוגל מפות ↗</a>` +
        (moves.has(focused.id) ? '<span class="am-flag">הוזז</span>'
          : focused.approx ? '<span class="am-flag warn">מיקום מקורב</span>' : '');
    } else {
      el('arrange-name').textContent = pending
        ? `${pending} מיקומים ממתינים לשמירה` : 'גרור סיכה למקום הנכון';
      el('arrange-links').innerHTML =
        '<span class="am-hint">החזק ליד קצה המפה והמפה תיגלל</span>';
    }
  }

  /** Hand over the next place that has no position at all: drop it in the
   *  middle of the map, where it is already under the cursor to be dragged. */
  function next() {
    const place = queue.find((p) => !moves.has(p.id));
    if (!place) return;

    // The first one lands dead centre. Later ones step around a small ring, or
    // a second place dropped before the first was moved would sit exactly on
    // top of it and only one of the two could ever be picked up.
    const canvas = map.getCanvas();
    const angle = dropped * 1.9;
    const r = dropped ? 34 : 0;
    const at = map.unproject([
      canvas.clientWidth / 2 + r * Math.cos(angle),
      canvas.clientHeight / 2 + r * Math.sin(angle)
    ]);
    dropped++;
    moves.set(place.id, {
      lat: +at.lat.toFixed(6),
      lng: +at.lng.toFixed(6)
    });
    render();
    focus(place, false);
    const entry = entries.get(place.id);
    if (entry) {
      entry.el.classList.add('moved', 'fresh');
      setTimeout(() => entry.el.classList.remove('fresh'), 1200);
    }
  }

  async function save() {
    if (!moves.size) return;
    const btn = el('arrange-save');
    btn.disabled = true;
    el('arrange-name').textContent = `שומר ${moves.size} מיקומים…`;
    try {
      const doc = await Store.movePlaces([...moves.entries()]
        .map(([id, at]) => ({ id, ...at })));
      moves.clear();
      dropped = 0;
      reloadPlaces(doc);
      queue = layer().waypoints.filter((p) => p.unplaced);
      clearMarkers();
      render();
      focus(null, false);
      el('arrange-name').textContent = 'נשמר.';
    } catch (err) {
      btn.disabled = false;
      el('arrange-name').textContent = 'לא נשמר: ' + err.message;
    }
  }

  function wire() {
    el('arrange-save').addEventListener('click', save);
    el('arrange-next').addEventListener('click', next);
    el('arrange-done').addEventListener('click', () => close());
  }

  return { open, close, wire, next, isOn: () => on, isDragging: () => !!drag };
})();
