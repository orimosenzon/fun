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
    import: 'התקבל כקובץ'
  };

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
    const path = rec.path;
    return {
      id: rec.id,
      name: rec.name,
      note: rec.note || '',
      photos,
      links: rec.links || [],
      layer: rec.layer || '',
      path,
      length: pathLength(path),
      color: '#8e24aa',
      draft: true,
      mode: rec.mode,
      created: rec.created,
      entries: [
        { lat: path[0][0], lng: path[0][1] },
        { lat: path[path.length - 1][0], lng: path[path.length - 1][1] }
      ]
    };
  }

  /** Rebuild the drafts layer from storage and tell the app to repaint. */
  function sync() {
    urls.forEach(URL.revokeObjectURL);
    urls = [];
    const layer = Layers.byId('drafts');
    layer.segments = rows
      .filter((r) => r.path && r.path.length > 1)
      .map(toSegment);
    Layers.refresh('drafts');
  }

  async function reload() {
    rows = (await readAll()).sort((a, b) => b.created - a.created);
    sync();
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
    } else {
      // The bar shares its width with three buttons, so the wording stays
      // short enough not to be cut off on a narrow phone.
      state = ed.path.length < 2
        ? 'לחץ על המפה לסימון התוואי'
        : `${ed.path.length} נקודות סומנו`;
    }
    el('draft-state').textContent = state;
    bar.classList.toggle('paused', !!ed.paused);
    el('draft-undo').hidden = ed.mode !== 'draw';
    el('draft-pause').hidden = ed.mode !== 'walk';
    el('draft-pause').textContent = ed.paused ? 'המשך' : 'השהה';
    el('draft-done').disabled = ed.path.length < 2;
  }

  function startEditor(mode, existing) {
    stopEditor(true);
    stopNav();          // the three share the top bar, and a tap on the map
    stopPinning();
    deselect();         // the panel shrinks away; leave nothing stale behind it
    ed = {
      mode,
      path: existing ? existing.path.slice() : [],
      editing: existing || null,
      paused: false,
      watch: null,
      weak: false
    };
    document.body.classList.add('drafting');
    el('draft-bar').hidden = false;
    paintEditor();

    if (mode === 'draw') {
      ed.onClick = (e) => {
        ed.path.push([+e.lngLat.lat.toFixed(6), +e.lngLat.lng.toFixed(6)]);
        paintEditor();
      };
      if (map) map.on('click', ed.onClick);
      if (map && existing && existing.path.length) fitTo(existing.path);
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
    if (ed && ed.onClick && map) map.off('click', ed.onClick);
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
      <label class="big-act ghost" style="cursor:pointer">
        <b>פתיחת קובץ שקיבלת</b>
        <span>שביל ששלח לך מישהו, ב-KML או GPX. נפתח לבדיקה על המפה לפני פרסום.</span>
        <input type="file" accept=".kml,.gpx,.json,application/xml,text/xml" hidden id="d-import">
      </label>
      <p class="sheet-credit">${Store.isEditor()
        ? 'אתה במצב עורך, אז אפשר לפרסם ישירות למסד המשותף מסך הפרטים.'
        : 'השביל נשמר במכשיר הזה בלבד, ומסך הפרטים אפשר לשלוח אותו ליוזמה.'}</p>`);
  }

  function askDetails() {
    const cur = ed.editing || {};
    const len = pathLength(ed.path);

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
    rec.layer = el('d-layer') ? el('d-layer').value : (rec.layer || '');
    rec.path = ed.path;
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

    // An editor publishes straight into the shared dataset. Everyone else
    // sends the trail as a file, and an editor reviews it before it lands -
    // which is the same queue, just carried by WhatsApp instead of a server.
    const publish = Store.isEditor() ? `
      <button class="act act-nav" data-draft="publish"><span class="lbl">פרסם למסד המשותף
        <span class="hint">ייכנס מיד לכל מי שפותח את האפליקציה</span></span></button>` : `
      <button class="act act-nav" data-draft="kml"><span class="lbl">שלח ליוזמה
        <span class="hint">נשלח כקובץ, ואחד מהעורכים יוסיף אותו למפה</span></span></button>`;

    return `
      ${done ? `<p class="landed">השביל הזה כבר מופיע במסד המשותף, אז אפשר
        למחוק את הטיוטה.</p>` : ''}
      <h3>${Store.isEditor() ? 'פרסום' : 'לשלוח ליוזמה'}</h3>
      <div class="acts">
        ${publish}
        ${Store.isEditor() ? `<button class="act act-sub" data-draft="kml">
          <span class="lbl">ייצוא KML<span class="hint">קובץ, לייבוא ידני ל-My Maps</span></span></button>` : ''}
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
      const { id, doc } = await Store.publish(seg, (rec && rec.photos) || [], say);
      await drop(seg.id);
      await reload();
      await reloadShared(doc);
      select(id);            // open it in its new life as a published trail
    } catch (err) {
      btn.disabled = false;
      msg.className = 'pub-msg bad';
      msg.hidden = false;
      msg.textContent = 'הפרסום נכשל: ' + err.message + ' הטיוטה נשארה אצלך.';
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
      else if (act === 'walk' || act === 'draw') { closeSheet(); startEditor(act); }
      else if (act === 'save') save();
      else if (act === 'resume' || act === 'back') {
        closeSheet();
        // Coming back from the rename form, there is no live editor to return to.
        if (ed && ed.paused && ed.editing && !ed.watch && ed.mode !== 'draw') stopEditor(true);
      }
    });

    el('draft-undo').addEventListener('click', () => {
      if (!ed) return;
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

  async function init() {
    wire();
    try {
      db = await open();
      await reload();
    } catch (err) {
      console.error('drafts unavailable', err);
    }
  }

  return { init, detailExtras, wireDetail, share, isDrafting: () => !!ed, paintEditor };
})();
