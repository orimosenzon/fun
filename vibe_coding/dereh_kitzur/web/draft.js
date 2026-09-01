/* Drafts - adding a trail from inside the app.
 *
 * Two ways in, because they suit different moments:
 *   walking  the phone records the trail while you walk it. This is the honest
 *            one here: 41 of the 49 known shortcuts exist in no map at all, so
 *            there is nothing on screen to trace over.
 *   drawing  tap the corners on the map, for a trail you already know.
 *
 * Drafts live in IndexedDB on this device only, with their photos and links,
 * until somebody does something with them. An editor publishes one straight
 * into the shared dataset; everyone else sends it as a KML or GPX file and an
 * editor opens it here and publishes it. That is the same review queue either
 * way, carried by WhatsApp rather than by a server this app does not have.
 */
'use strict';

const Drafts = (() => {

  const DB = 'derech-kitzur';
  const STORE = 'drafts';
  const MAX_PX = 1600;              // photos are resized before they are stored
  const MIN_STEP_M = 4;             // GPS noise below this is not movement
  const MAX_ACC_M = 30;             // a fix vaguer than this is not a position

  const MODE_TEXT = {
    walk: 'הוקלט בהליכה',
    draw: 'שורטט על המפה',
    import: 'התקבל כקובץ',
    trip: 'טיול משורשר'
  };

  /* How near a tap has to land on a shortcut to mean it. The hit line under
   * each trail is 22px wide, and a finger is wider than a tap point. */
  const TAP_PX = 10;

  /* How far the pointer may travel between press and release and still be a
   * tap rather than a pan.
   *
   * MapLibre's own `click` allows three pixels, and cancels the event outright
   * beyond that. Measured on this map: six taps with 3px of drift produced six
   * pointer-downs and *zero* clicks. That is not a slow tap, it is a lost one -
   * nothing appears, so the person taps again, and again. A fingertip drifts
   * more than three pixels nearly every time, and so does a trackpad.
   *
   * These are the ordinary platform figures instead: about ten pixels for a
   * mouse, and the wider touch slop for a finger. The cost is that a very
   * short deliberate pan now also drops a point, which "בטל נקודה" takes back
   * immediately - much the better way round. */
  const SLOP_MOUSE = 10;
  const SLOP_TOUCH = 18;

  /** Acknowledge a tap where the finger landed, straight away.
   *
   *  The point itself is drawn by MapLibre, into WebGL, on the map's own render
   *  loop - so how fast it appears is not this app's to decide. Measured on a
   *  machine falling back to software rendering, a single map frame took about
   *  a second and blocked the main thread, and the new point trailed the tap by
   *  a second or two. Somebody watching that has no way to tell a slow tap from
   *  a lost one, so they tap again, and again, and end up with four points they
   *  did not ask for.
   *
   *  This is an ordinary DOM element instead. It costs nothing, it does not
   *  wait for a map frame, and it paints on the compositor as soon as the main
   *  thread yields - always sooner than the point it is promising. It says one
   *  thing only, and it is the thing that was missing: "your tap landed". */
  function ping(point) {
    const host = document.getElementById('map');
    if (!host) return;
    const dot = document.createElement('span');
    dot.className = 'tap-ping';
    dot.style.left = point.x + 'px';
    dot.style.top = point.y + 'px';
    host.appendChild(dot);
    // Belt and braces: animationend does not fire if the animation is skipped
    // for a person who asked for reduced motion.
    dot.addEventListener('animationend', () => dot.remove());
    setTimeout(() => dot.remove(), 900);
  }

  /** Taps on the map, bound ourselves rather than through MapLibre's `click`.
   *  Returns the function that unbinds them.
   *
   *  A second finger means a pinch and never a tap, so any pointer arriving
   *  while one is already down abandons the gesture. */
  function onMapTap(handler) {
    const canvas = map.getCanvas();
    let start = null;
    let down = 0;

    const press = (e) => {
      down += 1;
      if (down > 1 || (e.button != null && e.button !== 0)) { start = null; return; }
      start = { x: e.clientX, y: e.clientY, touch: e.pointerType === 'touch' };
    };

    const release = (e) => {
      down = Math.max(0, down - 1);
      const from = start;
      start = null;
      if (!from || down) return;
      const slop = from.touch ? SLOP_TOUCH : SLOP_MOUSE;
      if (Math.hypot(e.clientX - from.x, e.clientY - from.y) > slop) return;
      const box = canvas.getBoundingClientRect();
      // A plain {x, y} is a PointLike everywhere this is passed on to.
      const point = { x: e.clientX - box.left, y: e.clientY - box.top };
      ping(point);
      handler({ point, lngLat: map.unproject([point.x, point.y]) });
    };

    const abandon = () => { down = 0; start = null; };

    canvas.addEventListener('pointerdown', press);
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', abandon);
    return () => {
      canvas.removeEventListener('pointerdown', press);
      canvas.removeEventListener('pointerup', release);
      canvas.removeEventListener('pointercancel', abandon);
    };
  }

  let db = null;
  let rows = [];                    // raw records, newest first
  let urls = [];                    // object URLs to revoke on rebuild
  let ed = null;                    // the live editor, see startEditor()

  /* ---------- storage ---------- */

  function open() {
    return new Promise((done, fail) => {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) {
          req.result.createObjectStore(STORE, { keyPath: 'id' });
        }
      };
      req.onsuccess = () => done(req.result);
      req.onerror = () => fail(req.error);
    });
  }

  function tx(mode, run) {
    return new Promise((done, fail) => {
      const t = db.transaction(STORE, mode);
      const out = run(t.objectStore(STORE));
      t.oncomplete = () => done(out && out.result !== undefined ? out.result : out);
      t.onerror = () => fail(t.error);
    });
  }

  const put = (rec) => tx('readwrite', (s) => s.put(rec));
  const drop = (id) => tx('readwrite', (s) => s.delete(id));
  const readAll = () => tx('readonly', (s) => s.getAll());

  /* ---------- geometry ---------- */

  function pathLength(path) {
    let sum = 0;
    for (let i = 1; i < path.length; i++) {
      sum += distance({ lat: path[i - 1][0], lng: path[i - 1][1] },
                      { lat: path[i][0], lng: path[i][1] });
    }
    return Math.round(sum);
  }

  /** A stored record, presented the way the rest of the app expects a segment. */
  function toSegment(rec) {
    const photos = (rec.photos || []).map((blob) => {
      const url = URL.createObjectURL(blob);
      urls.push(url);
      return { thumb: url, full: url };
    });
    // A trip draft holds its recipe and not its line, exactly like a published
    // one, so it is resolved here against whatever the shortcuts look like now.
    //
    // That resolution can come back empty - every shortcut it names is gone, or
    // the trails index is not populated at this instant - and this function
    // used to reach straight into path[0] for its entries. One such record threw,
    // `sync` threw with it, and *every* draft on the device vanished from the
    // list at once. A draft is somebody's unrepeatable afternoon: nothing here
    // may throw, and a record that cannot be drawn still has to be listed so
    // its owner can see it and decide what to do with it.
    const built = rec.parts ? Layers.resolveTrip(rec.parts) : null;
    const path = (built ? built.path : rec.path) || [];
    const ends = path.length
      ? [{ lat: path[0][0], lng: path[0][1] },
         { lat: path[path.length - 1][0], lng: path[path.length - 1][1] }]
      : [];
    return {
      id: rec.id,
      name: rec.name,
      note: rec.note || '',
      photos,
      links: rec.links || [],
      layer: rec.layer || '',
      path,
      ...(rec.parts ? {
        trip: true,
        parts: rec.parts,
        uses: built.uses,
        missing: built.missing,
        difficulty: rec.difficulty || '',
        group: rec.difficulty || '',
        minutes: rec.minutes || Math.round((pathLength(path) / 1000) * 15),
        loop: Layers.isLoop(path)
      } : {}),
      length: pathLength(path),
      // Empty leaves it to the drafts layer's own purple, which is how a draft
      // recorded before a colour could be picked still looks.
      color: rec.color || '',
      draft: true,
      mode: rec.mode,
      created: rec.created,
      entries: ends
    };
  }

  /** Rebuild the drafts layer from storage and tell the app to repaint.
   *
   *  Every record is converted inside its own try, so that one that cannot be
   *  converted costs its owner that one draft rather than all of them. A record
   *  that fails is left in storage untouched and reported to the console; it is
   *  never dropped from IndexedDB on the strength of a rendering error. */
  function sync() {
    urls.forEach(URL.revokeObjectURL);
    urls = [];
    const layer = Layers.byId('drafts');
    layer.segments = rows.map((rec) => {
      try {
        return toSegment(rec);
      } catch (err) {
        console.error('דרך קיצור: טיוטה שלא ניתן להציג', rec && rec.id, err);
        return null;
      }
    }).filter(Boolean);
    Layers.refresh('drafts');
  }

  async function reload() {
    rows = (await readAll()).sort((a, b) => b.created - a.created);
    sync();
  }

  /* ---------- building a trip ----------
   *
   * A trip is a continuous route that chains shortcuts already on the map with
   * pieces drawn between them, so the editor has to tell two taps apart on the
   * same map: one that means "walk along this shortcut" and one that means
   * "the route goes through here". A tap that lands on a visible shortcut is
   * the first; everything else is the second.
   *
   * What is kept is the recipe - which shortcut, which way round, which drawn
   * points - and never the line it produces. layers.js resolves it. */

  /** The shortcut under a tap, if the tap was on one. */
  function trailUnder(point) {
    const layers = Layers.trailHitLayers();
    // An empty list would ask MapLibre for every layer on the map, which would
    // make a tap on a plan or a cadastral block chain something absurd.
    if (!layers.length) return null;
    const box = [[point.x - TAP_PX, point.y - TAP_PX],
                 [point.x + TAP_PX, point.y + TAP_PX]];
    const hits = map.queryRenderedFeatures(box, { layers });
    return hits.length ? hits[0].properties.id : null;
  }

  /** Re-resolve the recipe into the line the bar and the map are showing. */
  function rebuildTrip() {
    const built = Layers.resolveTrip(ed.parts);
    ed.path = built.path;
    ed.uses = built.uses;
    paintEditor();
  }

  function pushDrawn(pt) {
    ed.gap = null;
    const last = ed.parts[ed.parts.length - 1];
    // Consecutive taps extend the same drawn run rather than each becoming a
    // part of its own, so that undo takes back a point and not a whole piece.
    if (last && last.draw) last.draw.push(pt);
    else ed.parts.push({ draw: [pt] });
    rebuildTrip();
  }

  /** Add a shortcut to the end of the trip, whichever way round it has to go.
   *
   *  A trail carries a direction - the order its path was recorded in - and a
   *  trip almost never wants all of them the same way. The end nearer to where
   *  the trip currently stops is the end it joins by, and if that is the far
   *  end of the trail then the trail goes in reversed. */
  function chainTrail(id) {
    const seg = Layers.item(id);
    if (!seg || !seg.path || seg.path.length < 2) return;

    if (!ed.path.length) {
      ed.gap = null;
      ed.parts.push({ trail: id });
      rebuildTrip();
      return;
    }

    const end = ed.path[ed.path.length - 1];
    const toStart = Layers.metres(end, seg.path[0]);
    const toEnd = Layers.metres(end, seg.path[seg.path.length - 1]);
    const reversed = toEnd < toStart;
    const gap = Math.min(toStart, toEnd);

    // Beyond the threshold this is ground nobody has walked, and joining it
    // with a straight line would draw a trip through back gardens. Say what is
    // missing and let the person draw it.
    if (gap > Layers.TRIP_GAP_M) {
      ed.gap = { name: seg.name, m: Math.round(gap) };
      paintBar();
      return;
    }
    ed.gap = null;
    ed.parts.push(reversed ? { trail: id, reversed: true } : { trail: id });
    rebuildTrip();
  }

  /** Take back the last thing added: a point from the run being drawn, or the
   *  whole shortcut that was chained. */
  function undoTrip() {
    const last = ed.parts[ed.parts.length - 1];
    if (!last) return;
    if (last.draw && last.draw.length > 1) last.draw.pop();
    else ed.parts.pop();
    ed.gap = null;
    rebuildTrip();
  }

  /* ---------- the editor ---------- */

  const EDIT_SRC = 'src-editor';

  function editorGeoJSON() {
    const line = ed.path.map(([lat, lng]) => [lng, lat]);
    return {
      type: 'FeatureCollection',
      features: [
        { type: 'Feature', geometry: { type: 'LineString', coordinates: line }, properties: {} },
        ...ed.path.map(([lat, lng], i) => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [lng, lat] },
          properties: { end: i === 0 || i === ed.path.length - 1 ? 1 : 0 }
        }))
      ]
    };
  }

  function paintEditor() {
    if (!map) return;
    if (!map.getSource(EDIT_SRC)) {
      map.addSource(EDIT_SRC, { type: 'geojson', data: editorGeoJSON() });
      map.addLayer({
        id: 'editor-line',
        type: 'line',
        source: EDIT_SRC,
        filter: ['==', ['geometry-type'], 'LineString'],
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#8e24aa', 'line-width': 5, 'line-opacity': 0.95 }
      });
      map.addLayer({
        id: 'editor-pts',
        type: 'circle',
        source: EDIT_SRC,
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-radius': ['case', ['==', ['get', 'end'], 1], 6, 3.5],
          'circle-color': '#fff',
          'circle-stroke-color': '#8e24aa',
          'circle-stroke-width': 2.5
        }
      });
    } else {
      map.getSource(EDIT_SRC).setData(editorGeoJSON());
    }
    paintBar();
  }

  function clearEditorLayers() {
    if (!map) return;
    ['editor-line', 'editor-pts'].forEach((id) => {
      if (map.getLayer(id)) map.removeLayer(id);
    });
    if (map.getSource(EDIT_SRC)) map.removeSource(EDIT_SRC);
  }

  function paintBar() {
    if (!ed) return;
    const bar = el('draft-bar');
    const len = pathLength(ed.path);
    el('draft-len').textContent = len >= 1000
      ? (len / 1000).toFixed(2) + ' ק"מ' : len + ' מ׳';

    let state;
    if (ed.mode === 'walk') {
      state = ed.paused ? 'מושהה. לחץ המשך כדי לחזור להקליט.'
        : (ed.path.length ? `מקליט · ${ed.path.length} נקודות` : 'מחפש מיקום מדויק…');
      if (ed.weak && !ed.paused) state = 'הקליטה פועלת, אבל הדיוק חלש כרגע…';
    } else if (ed.mode === 'trip') {
      const n = (ed.uses || []).length;
      state = ed.gap
        ? `פער ${ed.gap.m} מ׳ עד "${ed.gap.name}". צייר אותו קודם.`
        : !ed.path.length
          ? 'לחץ על דרך קיצור לשרשור, או על המפה לציור'
          : n
            ? `${n} ${n === 1 ? 'שביל משורשר' : 'שבילים משורשרים'}`
            : 'קטע מצויר. אפשר לשרשר עכשיו דרך קיצור.';
    } else {
      // The bar shares its width with three buttons, so the wording stays
      // short enough not to be cut off on a narrow phone.
      state = ed.path.length < 2
        ? 'לחץ על המפה לסימון התוואי'
        : `${ed.path.length} נקודות סומנו`;
    }
    el('draft-state').textContent = state;
    // A gap is not an error state, but it is the one thing in the bar that
    // wants to be noticed, and `paused` is already the bar's "look here".
    bar.classList.toggle('paused', !!ed.paused || !!(ed.gap));
    el('draft-undo').hidden = ed.mode !== 'draw' && ed.mode !== 'trip';
    el('draft-pause').hidden = ed.mode !== 'walk';
    el('draft-pause').textContent = ed.paused ? 'המשך' : 'השהה';
    el('draft-done').disabled = ed.path.length < 2;
  }

  function startEditor(mode, existing) {
    stopEditor(true);
    stopNav();          // the three share the top bar, and a tap on the map
    Arrange.close(true);
    deselect();         // the panel shrinks away; leave nothing stale behind it
    ed = {
      mode,
      path: existing && existing.path ? existing.path.slice() : [],
      editing: existing || null,
      paused: false,
      watch: null,
      weak: false,
      offTap: null,
      parts: [],
      uses: [],
      gap: null
    };
    document.body.classList.add('drafting');
    el('draft-bar').hidden = false;
    paintEditor();

    if (mode === 'draw') {
      if (map) ed.offTap = onMapTap((e) => {
        ed.path.push([+e.lngLat.lat.toFixed(6), +e.lngLat.lng.toFixed(6)]);
        paintEditor();
      });
      if (map && existing && existing.path.length) fitTo(existing.path);
      return;
    }

    if (mode === 'trip') {
      // You cannot chain what you cannot see, so the shortcuts come on whether
      // or not this browser had them on. Nothing is turned off again at the
      // end: having switched them on is a choice the person can now unmake.
      Layers.turnOn(Layers.TRAILS_ID);
      ed.parts = existing && existing.parts
        ? existing.parts.map((part) => ({ ...part,
            ...(part.draw ? { draw: part.draw.map((pt) => pt.slice()) } : {}) }))
        : [];
      rebuildTrip();
      if (map) ed.offTap = onMapTap((e) => {
        const id = trailUnder(e.point);
        if (id) chainTrail(id);
        else pushDrawn([+e.lngLat.lat.toFixed(6), +e.lngLat.lng.toFixed(6)]);
      });
      if (map && ed.path.length) fitTo(ed.path);
      return;
    }

    if (!navigator.geolocation) {
      alert('הדפדפן לא תומך באיתור מיקום, אז אי אפשר להקליט בהליכה. אפשר לצייר על המפה במקום.');
      stopEditor();
      return;
    }

    ed.watch = navigator.geolocation.watchPosition((pos) => {
      const { latitude: lat, longitude: lng, accuracy: acc } = pos.coords;
      here = { lat, lng };
      drawMe();
      if (map && ed.path.length < 2) map.easeTo({ center: [lng, lat], zoom: 18, duration: 500 });

      // A vague fix is worse than no fix: it bends the trail sideways by more
      // than the trail is wide, and that error is then stored for good.
      ed.weak = acc > MAX_ACC_M;
      if (ed.paused || ed.weak) { paintBar(); return; }

      const last = ed.path[ed.path.length - 1];
      if (last && distance({ lat: last[0], lng: last[1] }, { lat, lng }) < MIN_STEP_M) {
        paintBar();
        return;                       // standing still, not walking
      }
      ed.path.push([+lat.toFixed(6), +lng.toFixed(6)]);
      paintEditor();
    }, () => {
      el('draft-state').textContent = 'אין גישה למיקום. צריך לאשר, ורק מעל https.';
      el('draft-bar').classList.add('paused');
    }, { enableHighAccuracy: true, maximumAge: 0, timeout: 20000 });
  }

  function stopEditor(quiet) {
    if (ed && ed.watch != null) navigator.geolocation.clearWatch(ed.watch);
    if (ed && ed.offTap) ed.offTap();
    ed = null;
    clearEditorLayers();
    document.body.classList.remove('drafting');
    const bar = el('draft-bar');
    if (bar) { bar.hidden = true; bar.classList.remove('paused'); }
    if (!quiet) closeSheet();
  }

  function fitTo(path) {
    const b = new maplibregl.LngLatBounds();
    path.forEach(([lat, lng]) => b.extend([lng, lat]));
    map.fitBounds(b, { padding: 70, maxZoom: 18, duration: 600 });
  }

  /* ---------- photos ---------- */

  /** Downscale before storing. Phone photos are 3-6 MB each and IndexedDB is
   *  the user's own disk quota, so full-size originals would fill it fast. */
  function shrink(file) {
    return new Promise((done) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, MAX_PX / Math.max(img.width, img.height));
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(img.src);
        canvas.toBlob((blob) => done(blob || file), 'image/webp', 0.82);
      };
      img.onerror = () => done(file);
      img.src = URL.createObjectURL(file);
    });
  }

  async function attachPhotos(id, files) {
    const rec = rows.find((r) => r.id === id);
    if (!rec) return;
    rec.photos = rec.photos || [];
    for (const file of files) rec.photos.push(await shrink(file));
    rec.updated = Date.now();
    await put(rec);
    await reload();
  }

  /* ---------- export ---------- */

  // KML wants aabbggrr, which is neither the order nor the position of the
  // alpha channel that CSS uses.
  function kmlColour(hex) {
    const h = hex.replace('#', '');
    return 'ff' + h.slice(4, 6) + h.slice(2, 4) + h.slice(0, 2);
  }

  const xml = (s) => String(s || '').replace(/[&<>]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function toKML(segs) {
    const places = segs.map((s) => `  <Placemark>
    <name>${xml(s.name)}</name>
    <description>${xml([s.note, `אורך ${s.length} מ׳`,
      MODE_TEXT[s.mode] || MODE_TEXT.draw,
      'נוסף דרך אפליקציית דרך קיצור'].filter(Boolean).join(' · '))}</description>
    <styleUrl>#dk</styleUrl>
    <LineString><tessellate>1</tessellate><coordinates>
      ${s.path.map(([lat, lng]) => `${lng},${lat},0`).join(' ')}
    </coordinates></LineString>
  </Placemark>`).join('\n');

    return `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>דרך קיצור · שבילים חדשים</name>
  <description>${xml(`${segs.length} שבילים שנוספו מהאפליקציה, ${new Date().toLocaleDateString('he-IL')}`)}</description>
  <Style id="dk"><LineStyle><color>${kmlColour('#8e24aa')}</color><width>4</width></LineStyle></Style>
${places}
</Document>
</kml>
`;
  }

  function toGPX(segs) {
    const tracks = segs.map((s) => `  <trk><name>${xml(s.name)}</name><trkseg>
${s.path.map(([lat, lng]) => `    <trkpt lat="${lat}" lon="${lng}"/>`).join('\n')}
  </trkseg></trk>`).join('\n');
    return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="דרך קיצור" xmlns="http://www.topografix.com/GPX/1/1">
${tracks}
</gpx>
`;
  }

  async function share(segs, format) {
    if (!segs.length) return;
    const kml = format === 'gpx';
    const text = kml ? toGPX(segs) : toKML(segs);
    const name = kml ? 'derech-kitzur.gpx' : 'derech-kitzur.kml';
    const type = kml ? 'application/gpx+xml' : 'application/vnd.google-earth.kml+xml';
    const file = new File([text], name, { type });

    // On a phone the share sheet reaches WhatsApp directly, which is how these
    // actually get to Yoav. On a desktop it falls back to a download.
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: 'שבילים חדשים · דרך קיצור' });
        return;
      } catch (err) {
        if (err && err.name === 'AbortError') return;
      }
    }
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  /* ---------- sheets ---------- */

  const closeSheet = () => { el('draft-sheet').hidden = true; };

  function openSheet(html) {
    el('draft-card').innerHTML = html;
    el('draft-sheet').hidden = false;
  }

  function askMode() {
    openSheet(`
      <header class="sheet-head">
        <h2>הוספת שביל</h2>
        <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
      </header>
      <p class="sheet-lead">רוב קיצורי הדרך כאן לא מופיעים בשום מפה, ולכן אין מה
        לשרטט מעליו. הדרך המדויקת היא פשוט ללכת בשביל עם הטלפון.</p>
      <button class="big-act" data-act="walk">
        <b>הקלטה בהליכה</b>
        <span>לך בשביל מקצה לקצה, והאפליקציה מציירת אותו לפי ה-GPS.</span>
      </button>
      <button class="big-act" data-act="draw">
        <b>ציור על המפה</b>
        <span>סמן את התוואי בלחיצות. מתאים לשביל שאתה כבר מכיר, מהבית.</span>
      </button>
      <button class="big-act" data-act="trip">
        <b>טיול חדש</b>
        <span>מסלול רציף שמשרשר קיצורי דרך קיימים. לוחצים על שביל כדי לצרף
          אותו, ועל המפה כדי לצייר את מה שביניהם.</span>
      </button>
      <label class="big-act ghost" style="cursor:pointer">
        <b>פתיחת קובץ שקיבלת</b>
        <span>שביל ששלח לך מישהו, ב-KML או GPX. נפתח לבדיקה על המפה לפני פרסום.</span>
        <input type="file" accept=".kml,.gpx,.json,application/xml,text/xml" hidden id="d-import">
      </label>
      <p class="sheet-credit">${Store.isEditor()
        ? 'אתה במצב עורך, אז אפשר לפרסם ישירות למסד המשותף מסך הפרטים.'
        : 'השביל נשמר במכשיר הזה בלבד, ומסך הפרטים אפשר לשלוח אותו ליוזמה.'}</p>`);
  }

  /** The extra three a trip carries and a trail does not: how far and how long,
   *  how hard, and whether it brings you back to where you left the car.
   *
   *  The minutes are pre-filled from the distance at a flat-ground pace and are
   *  a suggestion, not a claim: the field is editable and whatever is typed
   *  wins. Circular or point-to-point is not asked at all - the two ends either
   *  meet or they do not, and asking somebody to confirm what the map already
   *  knows is a question with a right answer. */
  function tripFields(cur) {
    const len = pathLength(ed.path);
    const uses = ed.uses || [];
    const loop = Layers.isLoop(ed.path, len);
    const mins = cur.minutes || Math.round((len / 1000) * 15);
    const hard = cur.difficulty || 'קל';

    return `
      <p class="sheet-lead">${len >= 1000 ? (len / 1000).toFixed(2) + ' ק"מ' : len + ' מ׳'}
        · ${loop ? 'מסלול מעגלי' : 'מקצה לקצה'}${uses.length
          ? ` · עובר ב-${uses.length} ${uses.length === 1 ? 'דרך קיצור' : 'דרכי קיצור'}`
          : ' · מצויר ביד'}</p>
      ${uses.length ? `<p class="sheet-credit trip-uses">${
        uses.map((u) => escapeHtml(u.name)).join(' ← ')}</p>` : ''}
      <div class="fld"><span>דרגת קושי</span>
        <div class="picks tight" id="d-hard-pick">
          ${Layers.DIFFICULTY.map((d) => `<label class="pick${d.name === hard ? ' on' : ''}">
            <input type="radio" name="hard" value="${d.name}" ${d.name === hard ? 'checked' : ''}>
            <span class="lay-swatch" style="--c:${d.color}"></span>
            <span>${d.name}</span></label>`).join('')}
        </div>
        <input type="hidden" id="d-hard" value="${escapeHtml(hard)}">
      </div>
      <label class="fld"><span>זמן הליכה משוער (דקות)</span>
        <input id="d-mins" type="number" min="1" max="900" value="${mins}"></label>`;
  }

  function askDetails() {
    const cur = ed.editing || {};
    const len = pathLength(ed.path);

    if (ed.mode === 'trip') {
      openSheet(`
        <header class="sheet-head">
          <h2>${cur.id ? 'עדכון הטיול' : 'טיול חדש'}</h2>
          <button class="sheet-x" data-act="back" aria-label="חזרה">&times;</button>
        </header>
        ${tripFields(cur)}
        <label class="fld"><span>שם הטיול</span>
          <input id="d-name" type="text" maxlength="60" value="${escapeHtml(cur.name || '')}"
                 placeholder="למשל: סובב שמורת החורש"></label>
        <label class="fld"><span>הערה (לא חובה)</span>
          <textarea id="d-note" rows="2" maxlength="240"
                    placeholder="מתאים לעגלה, צל רוב הדרך, יש ברזייה באמצע…"
                    >${escapeHtml(cur.note || '')}</textarea></label>
        <div class="fld"><span>קישורים (לא חובה)</span>
          ${LinkRows.html(cur.links)}</div>
        <button class="big-act primary" data-act="save"><b>שמור טיול</b></button>
        <button class="big-act ghost" data-act="resume"><b>חזרה למסלול</b>
          <span>לשרשר עוד שביל או להוסיף קטע מצויר</span></button>`);
      setTimeout(() => el('d-name').focus(), 60);
      return;
    }

    // A layer to publish into is only worth asking about once an editor has
    // made one, and only to somebody who can publish at all.
    const targets = Store.isEditor() ? Layers.trailLayers() : [];
    const picker = targets.length > 1 ? `
      <label class="fld"><span>שכבה</span>
        <select id="d-layer">
          ${targets.map((l) => `<option value="${l.id === Layers.TRAILS_ID ? '' : l.id}"
            ${l.id === (cur.layer || Layers.TRAILS_ID) ? 'selected' : ''}
            >${escapeHtml(l.name)}</option>`).join('')}
        </select></label>` : '';

    openSheet(`
      <header class="sheet-head">
        <h2>${cur.id ? 'עדכון השביל' : 'שביל חדש'}</h2>
        <button class="sheet-x" data-act="back" aria-label="חזרה">&times;</button>
      </header>
      <p class="sheet-lead">${ed.path.length} נקודות ·
        ${len >= 1000 ? (len / 1000).toFixed(2) + ' ק"מ' : len + ' מ׳'}</p>
      <label class="fld"><span>שם השביל</span>
        <input id="d-name" type="text" maxlength="60" value="${escapeHtml(cur.name || '')}"
               placeholder="למשל: מהצפירה לבית הכנסת"></label>
      <label class="fld"><span>הערה (לא חובה)</span>
        <textarea id="d-note" rows="2" maxlength="240"
                  placeholder="מדרגות בקצה, חסום בחורף, מתאים לעגלה…">${escapeHtml(cur.note || '')}</textarea></label>
      ${picker}
      <div class="fld"><span>צבע השביל</span>
        ${Swatches.html(TRAIL_COLOURS, cur.color || '#097138', true)}
      </div>
      <div class="fld"><span>קישורים (לא חובה)</span>
        ${LinkRows.html(cur.links)}</div>
      <button class="big-act primary" data-act="save"><b>שמור שביל</b></button>
      <button class="big-act ghost" data-act="resume"><b>חזרה לתוואי</b>
        <span>להוסיף עוד נקודות או להמשיך להקליט</span></button>`);
    setTimeout(() => el('d-name').focus(), 60);
  }

  async function save() {
    const name = el('d-name').value.trim();
    if (!name) { el('d-name').focus(); return; }
    const rec = ed.editing && ed.editing.id
      ? rows.find((r) => r.id === ed.editing.id)
      : { id: 'draft-' + Date.now(), created: Date.now(), photos: [] };

    rec.name = name;
    rec.note = el('d-note').value.trim();
    rec.links = LinkRows.read(el('draft-card'));
    rec.color = Swatches.read(el('draft-card'));
    rec.layer = el('d-layer') ? el('d-layer').value : (rec.layer || '');
    if (ed.mode === 'trip') {
      rec.parts = ed.parts;
      delete rec.path;                 // the recipe is the record; the line is not
      rec.difficulty = (el('d-hard') || {}).value || '';
      const typed = parseInt((el('d-mins') || {}).value, 10);
      rec.minutes = Number.isFinite(typed) && typed > 0 ? typed : 0;
    } else {
      rec.path = ed.path;
    }
    rec.mode = ed.mode;
    rec.updated = Date.now();

    await put(rec);
    const id = rec.id;
    stopEditor(true);
    closeSheet();
    await reload();
    select(id);
  }

  /* ---------- detail pane, rendered by app.js ---------- */

  /** True once a trail with the same name shows up in the initiative's own
   *  layer, meaning the hand-off to My Maps has come full circle. */
  function landed(seg) {
    const trails = Layers.byId('trails');
    return trails.segments.some((s) => s.name.trim() === seg.name.trim());
  }

  function detailExtras(seg) {
    const done = landed(seg);

    // With edit mode on, a trail goes straight onto the map. Without it, it
    // goes into the queue - which needs nothing from the sender, no account
    // and no key. Until the worker existed this hand-off was a file sent over
    // WhatsApp, and a file in a chat is a thing somebody has to remember.
    const publish = Store.isEditor() ? `
      <button class="act act-nav" data-draft="publish"><span class="lbl">פרסם למפה
        <span class="hint">ייכנס מיד לכל מי שפותח את האפליקציה</span></span></button>` : `
      <button class="act act-nav" data-draft="submit"${Store.writable() === false ? ' disabled' : ''}>
        <span class="lbl">שלח ליוזמה
        <span class="hint">${Store.writable() === false
          ? 'אין חיבור כרגע. השביל נשמר אצלך ואפשר לשלוח אחר כך'
          : 'נכנס לתור, ומישהו מהיוזמה יאשר אותו למפה'}</span></span></button>`;

    return `
      ${done ? `<p class="landed">השביל הזה כבר מופיע במסד המשותף, אז אפשר
        למחוק את הטיוטה.</p>` : ''}
      <h3>${Store.isEditor() ? 'פרסום' : 'לשלוח ליוזמה'}</h3>
      <div class="acts">
        ${publish}
        <button class="act act-sub" data-draft="kml">
          <span class="lbl">ייצוא KML<span class="hint">קובץ, לייבוא ידני ל-My Maps</span></span></button>
        <button class="act act-sub" data-draft="gpx"><span class="lbl">ייצוא GPX
          <span class="hint">לאפליקציות הליכה וניווט</span></span></button>
      </div>
      <p id="pub-msg" class="pub-msg" hidden></p>
      <h3>עריכה</h3>
      <div class="acts">
        <label class="act" style="cursor:pointer"><span class="lbl">הוספת תמונות
          <span class="hint">נשמרות במכשיר, מוקטנות אוטומטית</span></span>
          <input type="file" accept="image/*" multiple hidden data-draft="photos"></label>
        <button class="act" data-draft="edit"><span class="lbl">עריכת התוואי
          <span class="hint">להוסיף או להסיר נקודות</span></span></button>
        <button class="act" data-draft="rename"><span class="lbl">שם, הערה וקישורים
          <span class="hint">${(seg.links || []).length
            ? plural(seg.links.length, 'קישור אחד', 'קישורים') : 'אתר, כתבה, ערך בוויקי'}</span></span></button>
        <button class="act danger" data-draft="delete"><span class="lbl">מחיקת הטיוטה</span></button>
      </div>
      <p class="src">נוצר ${new Date(seg.created).toLocaleDateString('he-IL')} ·
        ${MODE_TEXT[seg.mode] || MODE_TEXT.draw} · שמור במכשיר הזה בלבד</p>`;
  }

  /** Send a draft into the shared dataset, then retire the local copy.
   *
   *  Deleting only after the write has come back means a failed publish leaves
   *  the trail exactly where it was, rather than losing a walk. */
  async function publish(seg, btn) {
    const msg = el('pub-msg');
    const rec = rows.find((r) => r.id === seg.id);
    const say = (text) => { msg.hidden = false; msg.textContent = text; msg.className = 'pub-msg'; };

    btn.disabled = true;
    say('מפרסם…');
    try {
      const { id, doc } = seg.trip
        ? await Store.publishTrip(seg, (rec && rec.photos) || [], say)
        : await Store.publish(seg, (rec && rec.photos) || [], say);
      await drop(seg.id);
      await reload();
      await reloadShared(doc);
      // Publishing consumes the draft, and a trip lands in a layer that is off
      // by default - so without this the thing you just published vanishes from
      // the screen at the moment you publish it. That is exactly what it looks
      // like when work is lost, and it is how the first published trip was
      // mistaken for a trip that had been eaten.
      if (seg.trip) Layers.turnOn(Layers.TRIPS_ID);
      select(id);            // open it in its new life as a published trail
    } catch (err) {
      btn.disabled = false;
      msg.className = 'pub-msg bad';
      msg.hidden = false;
      msg.textContent = 'הפרסום נכשל: ' + err.message + ' הטיוטה נשארה אצלך.';
    }
  }

  /** Send a draft into the review queue.
   *
   *  Same care as publish(): the local copy is only dropped once the write has
   *  come back, so a failed send leaves the walk exactly where it was. */
  async function send(seg, btn) {
    const msg = el('pub-msg');
    const rec = rows.find((r) => r.id === seg.id);
    const say = (text) => { msg.hidden = false; msg.textContent = text; msg.className = 'pub-msg'; };

    btn.disabled = true;
    say('שולח…');
    try {
      await Store.submit(seg, (rec && rec.photos) || [], say);
      await drop(seg.id);
      deselect();
      await reload();
      await refreshQueue();
      alert(seg.trip
        ? 'נשלח, תודה. הטיול ממתין לאישור ויופיע על המפה בקרוב.'
        : 'נשלח, תודה. השביל ממתין לאישור ויופיע על המפה בקרוב.');
    } catch (err) {
      btn.disabled = false;
      msg.className = 'pub-msg bad';
      msg.hidden = false;
      msg.textContent = 'השליחה נכשלה: ' + err.message + ' הטיוטה נשארה אצלך.';
    }
  }

  /* ---------- importing a trail somebody sent ---------- */

  function parseIncoming(text, filename) {
    const trails = [];

    if (/^\s*\{/.test(text)) {                     // our own JSON export
      const doc = JSON.parse(text);
      (doc.segments || [doc]).forEach((s) => {
        if (s.path && s.path.length > 1) trails.push({ name: s.name, note: s.note, path: s.path });
      });
      return trails;
    }

    const doc = new DOMParser().parseFromString(text, 'application/xml');
    if (doc.querySelector('parsererror')) throw new Error('הקובץ לא נקרא כ-KML או GPX תקין.');

    // KML: <Placemark><LineString><coordinates>lng,lat,alt ...
    doc.querySelectorAll('Placemark').forEach((pm) => {
      const coords = pm.querySelector('LineString > coordinates');
      if (!coords) return;
      const path = coords.textContent.trim().split(/\s+/).map((t) => {
        const [lng, lat] = t.split(',').map(Number);
        return [lat, lng];
      }).filter(([lat, lng]) => isFinite(lat) && isFinite(lng));
      if (path.length > 1) {
        trails.push({
          name: (pm.querySelector('name') || {}).textContent || filename,
          note: (pm.querySelector('description') || {}).textContent || '',
          path
        });
      }
    });

    // GPX: <trk><trkseg><trkpt lat lon>
    doc.querySelectorAll('trk').forEach((trk) => {
      const path = [...trk.querySelectorAll('trkpt')]
        .map((p) => [+p.getAttribute('lat'), +p.getAttribute('lon')]);
      if (path.length > 1) {
        trails.push({ name: (trk.querySelector('name') || {}).textContent || filename, note: '', path });
      }
    });

    if (!trails.length) throw new Error('לא מצאתי בקובץ אף תוואי.');
    return trails;
  }

  /** Bring a received file in as ordinary drafts, so an editor can look at it
   *  on the map, fix the name, and publish or discard it. */
  async function importFile(file) {
    const text = await file.text();
    const found = parseIncoming(text, file.name.replace(/\.[^.]+$/, ''));
    let last = null;
    for (const t of found) {
      const rec = {
        id: 'draft-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 6),
        name: (t.name || '').trim() || 'שביל שהתקבל',
        note: (t.note || '').trim(),
        path: t.path,
        mode: 'import',
        created: Date.now(),
        updated: Date.now(),
        photos: []
      };
      await put(rec);
      last = rec.id;
    }
    await reload();
    closeSheet();
    if (last) select(last);
    return found.length;
  }

  /** Wire the buttons detailExtras() just wrote. */
  function wireDetail(seg, root) {
    root.querySelectorAll('[data-draft]').forEach((node) => {
      const act = node.dataset.draft;
      if (act === 'photos') {
        node.addEventListener('change', async () => {
          if (node.files && node.files.length) {
            await attachPhotos(seg.id, [...node.files]);
            select(seg.id, false);
          }
        });
        return;
      }
      node.addEventListener('click', async () => {
        if (act === 'kml' || act === 'gpx') return share([seg], act);
        if (act === 'publish') return publish(seg, node);
        if (act === 'submit') return send(seg, node);
        if (act === 'edit') {
          deselect();
          startEditor(seg.mode === 'walk' ? 'draw' : seg.mode, seg);
          return;
        }
        if (act === 'rename') {
          ed = { mode: seg.mode, path: seg.path.slice(), editing: seg, paused: true };
          askDetails();
          return;
        }
        if (act === 'delete') {
          if (!confirm(`למחוק את "${seg.name}"? אין דרך לשחזר.`)) return;
          await drop(seg.id);
          deselect();
          await reload();
        }
      });
    });
  }

  /* ---------- wiring ---------- */

  function wire() {
    el('add').addEventListener('click', askMode);
    LinkRows.wire(el('draft-sheet'));
    Swatches.wire(el('draft-sheet'));

    el('draft-sheet').addEventListener('change', async (e) => {
      if (e.target.id !== 'd-import' || !e.target.files.length) return;
      try {
        const n = await importFile(e.target.files[0]);
        if (n > 1) alert(`נפתחו ${n} תוואים מהקובץ. כולם ברשימה תחת הטיוטות שלך.`);
      } catch (err) {
        alert('לא הצלחתי לקרוא את הקובץ. ' + err.message);
      }
    });

    el('draft-sheet').addEventListener('click', async (e) => {
      if (e.target.id === 'draft-sheet') { closeSheet(); return; }
      const btn = e.target.closest('[data-act]');
      if (!btn) return;
      const act = btn.dataset.act;
      if (act === 'close') closeSheet();
      else if (act === 'walk' || act === 'draw' || act === 'trip') {
        closeSheet(); startEditor(act);
      }
      else if (act === 'save') save();
      else if (act === 'resume' || act === 'back') {
        closeSheet();
        // Coming back from the rename form, there is no live editor to return to.
        if (ed && ed.paused && ed.editing && !ed.watch
            && ed.mode !== 'draw' && ed.mode !== 'trip') stopEditor(true);
      }
    });

    // The difficulty picker writes into the hidden field the save reads, and
    // paints the chosen one, the same way the colour swatches do.
    el('draft-sheet').addEventListener('change', (e) => {
      if (e.target.name !== 'hard') return;
      el('d-hard').value = e.target.value;
      el('d-hard-pick').querySelectorAll('.pick').forEach((p) => {
        p.classList.toggle('on', p.contains(e.target));
      });
    });

    el('draft-undo').addEventListener('click', () => {
      if (!ed) return;
      if (ed.mode === 'trip') { undoTrip(); return; }
      ed.path.pop();
      paintEditor();
    });

    el('draft-pause').addEventListener('click', () => {
      if (!ed) return;
      ed.paused = !ed.paused;
      paintBar();
    });

    el('draft-done').addEventListener('click', () => {
      if (!ed || ed.path.length < 2) return;
      ed.paused = true;
      askDetails();
    });

    el('draft-cancel').addEventListener('click', () => {
      if (ed && ed.path.length > 1 && !confirm('לבטל את השביל? מה שנרשם יימחק.')) return;
      stopEditor();
    });
  }

  /** Ask the browser to stop treating this storage as disposable.
   *
   *  Without this, IndexedDB is "best effort": a browser under disk pressure
   *  may clear it, and a trail somebody walked and photographed is gone with
   *  no warning and no copy anywhere. Chrome grants the request silently to a
   *  site the person actually uses, and refusing costs nothing - the drafts
   *  keep working either way, they are merely evictable again. */
  async function keepStorage() {
    try {
      if (navigator.storage && navigator.storage.persist) {
        const already = await navigator.storage.persisted();
        if (!already) await navigator.storage.persist();
      }
    } catch (err) {
      /* not offered here; the drafts still save */
    }
  }

  async function init() {
    wire();
    keepStorage();                    // not awaited: the drafts must not wait on it
    try {
      db = await open();
      await reload();
    } catch (err) {
      console.error('drafts unavailable', err);
    }
  }

  return { init, detailExtras, wireDetail, share, isDrafting: () => !!ed, paintEditor,
           stop: () => stopEditor() };
})();
