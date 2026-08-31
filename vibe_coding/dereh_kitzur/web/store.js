/* Store - where the trails and the places actually live.
 *
 * Until 22/8/2026 the initiative's Google My Maps layer was the source of
 * truth, and this app was a read-only view of it. Google offers no way to
 * write to a My Maps layer from code, so every new trail needed a manual
 * import and the app could never be the place you added one.
 *
 * So the direction is reversed. The dataset lives in its own public repo,
 * this app reads it and writes to it, and My Maps now receives a generated
 * copy instead of feeding us.
 *
 * Three documents in that repo:
 *   data/trails.json   the initiative's trails, its waypoints, and the trail
 *                      layers an editor has created. Written from the app.
 *   data/layers.json   the moshava's cycling plan. Written by build_network.py.
 *   data/places.json   places drawn from pardespedia. Written by
 *                      build_places.py, except for the positions, which an
 *                      editor pins by hand from the map - the wiki has no
 *                      coordinates at all, so a good half of them can only be
 *                      placed by someone who knows where the place is.
 *   data/art2026.json  the אמנות במושבה festival map. Written by build_art.py,
 *                      positions included: the festival placed its own pins and
 *                      this app never writes to this file.
 *   data/shimur.json   the conservation appendix of the master plan, and
 *   data/makom_shamur.json  the מקום שמור documentation project. Both written
 *                      by build_shimur.py; neither is written from the app.
 *
 * Reading needs nothing at all: the files are public and come straight off a
 * CDN. Writing goes through a small Cloudflare worker that holds the only
 * GitHub credential involved.
 *
 * There is no login. Until 25/8/2026 each editor pasted their own fine-grained
 * token, which worked and meant every editor needed a GitHub account - so in
 * practice there was one editor and there was only ever going to be one. The
 * point of a community map is that the community can edit it, so the credential
 * moved to the worker and the barrier came down. What the worker will and will
 * not accept is in worker/README.md; every write is still a commit, so nothing
 * is ever really lost.
 */
'use strict';

const Store = (() => {

  const OWNER = 'orimosenzon';
  const REPO = 'derech-kitzur-data';
  const BRANCH = 'main';

  /* Reading the data that sits next to the app instead of the published repo
   * is the only way to look at a data change - a new layer, a rebuilt network -
   * before pushing it. Two ways in, because a console-only switch is no use
   * when you just want to open a link: `?local` on the URL, or `window.DK_RAW`
   * for a script driving the page.
   *
   * Read side only, and it grants nothing: everything it can reach is already
   * public. It also refuses to engage off localhost, so a `?local` link that
   * escapes into the wild shows the real data rather than a stale copy. */
  const onLocalhost = typeof location !== 'undefined' &&
    /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  const askedLocal = typeof location !== 'undefined' &&
    new URLSearchParams(location.search).has('local');

  const RAW = (typeof window !== 'undefined' && window.DK_RAW) ||
    (askedLocal && onLocalhost ? './' :
      `https://raw.githubusercontent.com/${OWNER}/${REPO}/${BRANCH}/`);

  if (RAW === './') console.info('דרך קיצור: נתונים מקומיים, לא מהריפו המפורסם');

  /* Every write goes through here. The app carries no GitHub credential at
   * all: it cannot, because everything shipped to a browser is public. The
   * worker holds one and checks what may be written. See worker/README.md for
   * what that protects and what it deliberately does not.
   *
   * Deployed 27/8/2026. Empty here would mean read-only for everyone, which is
   * what the app did between the rewrite and the deploy.
   *
   * `window.DK_WORKER` points this at a `wrangler dev` on localhost while
   * working on the write path. It is not a way in: the worker accepts no
   * credential from anyone, so choosing a different one grants nothing that
   * curl would not. */
  const WORKER = (typeof window !== 'undefined' && window.DK_WORKER) ||
    'https://derech-kitzur.orimosenzon.workers.dev';

  const TRAILS = 'data/trails.json';
  const PLACES = 'data/places.json';
  const ART = 'data/art2026.json';
  const SHIMUR = 'data/shimur.json';
  const MAKOM = 'data/makom_shamur.json';
  const PLANS = 'data/plans.json';
  const BLOCKS = 'data/blocks.json';

  const K_TRAILS = 'dk.cache.trails.v2';
  const K_NET = 'dk.cache.network.v2';
  const K_PLACES = 'dk.cache.places.v1';
  const K_ART = 'dk.cache.art.v1';
  const K_SHIMUR = 'dk.cache.shimur.v1';
  const K_MAKOM = 'dk.cache.makom.v1';
  const K_PLANS = 'dk.cache.plans.v1';
  const K_BLOCKS = 'dk.cache.blocks.v1';
  const K_ON = 'dk.editing.v1';         // the edit toggle, per browser
  const K_NAME = 'dk.name.v1';          // what to write in `by`, if given
  const K_KEY = 'dk.key.v1';            // the editor's password, once verified

  // `approved` starts false and only the worker can turn it true, so every
  // path that asks "may this browser edit" is closed until it has answered.
  const state = { offline: false, writable: null, checked: false, approved: false };

  /* Photo paths are stored relative to the data repo, so they resolve the same
   * whether the app is served from Pages, from localhost, or from a file the
   * editor is reviewing before publishing. Pardespedia photos arrive as
   * absolute URLs and pass straight through. */
  const asset = (p) => (!p || /^(https?:|blob:|data:)/.test(p) ? p : RAW + p);

  function absolutise(doc) {
    if (!doc) return doc;
    [...(doc.segments || []), ...(doc.waypoints || []), ...(doc.places || [])]
      .forEach((it) => {
        (it.photos || []).forEach((ph) => {
          ph.thumb = asset(ph.thumb);
          ph.full = asset(ph.full);
        });
      });
    return doc;
  }

  /* ---------- reading ---------- */

  function cached(key) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (err) {
      return null;
    }
  }

  function cache(key, doc) {
    try {
      localStorage.setItem(key, JSON.stringify(doc));
    } catch (err) {
      /* quota or private mode: the app simply refetches next time */
    }
  }

  async function fetchJson(path) {
    const res = await fetch(RAW + path, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res.json();
  }

  /** raw.githubusercontent serves from a CDN with a five minute cache, which
   *  is fine for a visitor and wrong for the person who just published: their
   *  own trail would disappear for five minutes. Somebody with editing switched
   *  on reads through the worker instead, which is always current.
   *
   *  Two things this asks that look wrong and are not:
   *
   *  `editing()` and not `isEditor()`, because the worker has not answered yet
   *  when the app first loads - approval arrives a moment later - so the first
   *  read of the session, the one right after a save, would be the stale one.
   *  That is the exact case this function exists for. Reading through the
   *  worker is ungated, so it needs no approval to ask.
   *
   *  And `?local` wins over both, because it is an explicit request to look at
   *  the files sitting next to the app rather than at the published dataset. */
  async function canonical(path) {
    if (RAW !== './' && editing()) {
      try {
        const { json } = await getFile(path);
        if (json) return json;
      } catch (err) {
        /* worker down or offline; the public copy still works */
      }
    }
    return fetchJson(path);
  }

  /** The canonical data, with three fallbacks so the app opens on a dead
   *  connection: the live repo, then the last copy we saw, then the seed
   *  bundled with the app itself. */
  async function load() {
    let trails = null;
    let network = null;
    let places = null;
    let art = null;
    let shimur = null;
    let makom = null;
    let plans = null;
    let blocks = null;
    try {
      [trails, network, places, art, shimur, makom, plans, blocks] = await Promise.all([
        canonical(TRAILS),
        fetchJson('data/layers.json').catch(() => null),
        // The places file is younger than the repo, and a copy also ships with
        // the app, so a miss here is ordinary rather than a failure.
        canonical(PLACES).catch(() => bundled('data/places.json')),
        // Nothing writes to the festival file from here, so it never needs the
        // worker's always-current read - the CDN copy is the current one.
        fetchJson(ART).catch(() => bundled('data/art2026.json')),
        // Same for the two conservation layers: build_shimur.py writes them,
        // the app only reads them.
        fetchJson(SHIMUR).catch(() => bundled('data/shimur.json')),
        fetchJson(MAKOM).catch(() => bundled('data/makom_shamur.json')),
        // And the planning schemes, which build_plans.py writes.
        fetchJson(PLANS).catch(() => bundled('data/plans.json')),
        // The cadastral blocks, which build_cadastre.py writes.
        fetchJson(BLOCKS).catch(() => bundled('data/blocks.json'))
      ]);
      cache(K_TRAILS, trails);
      if (network) cache(K_NET, network);
      if (places) cache(K_PLACES, places);
      if (art) cache(K_ART, art);
      if (shimur) cache(K_SHIMUR, shimur);
      if (makom) cache(K_MAKOM, makom);
      if (plans) cache(K_PLANS, plans);
      if (blocks) cache(K_BLOCKS, blocks);
    } catch (err) {
      state.offline = true;
      trails = cached(K_TRAILS);
      network = cached(K_NET);
      places = cached(K_PLACES);
      art = cached(K_ART);
      shimur = cached(K_SHIMUR);
      makom = cached(K_MAKOM);
      plans = cached(K_PLANS);
      blocks = cached(K_BLOCKS);
      if (!trails) {
        // First ever visit, with no connection. The copy shipped with the app
        // is stale by definition, but it beats an empty map.
        trails = await bundled('data/trails.json');
        network = await bundled('data/layers.json');
        places = await bundled('data/places.json');
        art = await bundled('data/art2026.json');
        shimur = await bundled('data/shimur.json');
        makom = await bundled('data/makom_shamur.json');
        plans = await bundled('data/plans.json');
        blocks = await bundled('data/blocks.json');
      }
    }
    return {
      trails: absolutise(trails),
      network,
      places: absolutise(places),
      art: absolutise(art),
      shimur: absolutise(shimur),
      makom: absolutise(makom),
      plans: absolutise(plans),
      blocks: absolutise(blocks),
      offline: state.offline
    };
  }

  const bundled = (path) => fetch(path).then((r) => r.json()).catch(() => null);

  /* ---------- editing ----------
   *
   * Contributing needs nothing: a resident who walked a shortcut records it and
   * sends it in, with no account, no password and no GitHub. That path is the
   * point of the whole app and it stays wide open.
   *
   * Deciding what goes *on the map* is a different act, and since 28/8/2026 it
   * takes a password. Until then edit mode was a switch in localStorage, which
   * meant anybody who opened the app could publish, approve and delete - and
   * the worker, taking no credential at all, accepted the same from curl.
   *
   * The password is held by the worker as a secret and never travels except as
   * a header on a write. This app only ever learns whether the one it holds is
   * the right one, from /health.
   *
   * The name stays optional and unverified. It goes in the commit and in `by`,
   * so a change has a face rather than being anonymous by default.
   */

  const editing = () => localStorage.getItem(K_ON) === 'yes';
  const named = () => localStorage.getItem(K_NAME) || '';
  const keyed = () => localStorage.getItem(K_KEY) || '';

  /** May this browser change the map: the switch is on, a worker is up, and it
   *  has confirmed the password this browser holds. */
  const isEditor = () => editing() && !!WORKER && state.writable !== false
    && state.approved;
  const editor = () => (isEditor() ? (named() || 'עורך') : null);

  /** Ask the worker about a password without storing it. */
  async function verify(key) {
    if (!WORKER || !key) return false;
    try {
      const res = await fetch(`${WORKER}/health`,
        { cache: 'no-store', headers: { 'X-DK-Key': key } });
      const body = res.ok ? await res.json() : null;
      return !!body && body.editor === true;
    } catch (err) {
      return false;                     // offline: not a wrong password, but
    }                                   // not something we can act on either
  }

  /** Turn edit mode on, or just update the name once it is already on.
   *
   *  Async, and false means refused: the password is checked with the worker
   *  before anything is stored, so a typo is caught here rather than at the end
   *  of a walk when there is a trail to publish. */
  async function enable(name, key) {
    if (name != null) localStorage.setItem(K_NAME, String(name).trim().slice(0, 40));
    if (key != null && String(key).trim()) {
      const trimmed = String(key).trim();
      if (!(await verify(trimmed))) return false;
      localStorage.setItem(K_KEY, trimmed);
      state.approved = true;
    }
    if (!state.approved) return false;
    localStorage.setItem(K_ON, 'yes');
    return true;
  }

  /** Leaving edit mode forgets the password too.
   *
   *  This is a phone that goes out on trails, so "off" should mean the next
   *  person holding it cannot turn it back on. Getting back in is one paste. */
  function disable() {
    localStorage.removeItem(K_ON);
    localStorage.removeItem(K_KEY);
    state.approved = false;
  }

  /** Ask the worker whether it is alive and accepting writes.
   *
   *  A worker that is down, or switched to read-only, should present as "you
   *  cannot publish right now" rather than letting somebody record a walk and
   *  discover at the last step that it cannot be saved. */
  async function resume() {
    if (!WORKER) { state.writable = false; state.checked = true; return null; }
    try {
      const res = await fetch(`${WORKER}/health`,
        { cache: 'no-store', ...(keyed() ? { headers: { 'X-DK-Key': keyed() } } : {}) });
      const body = res.ok ? await res.json() : null;
      state.writable = !!body && body.ok && !body.readOnly;
      state.approved = !!body && body.editor === true;
      // A key the worker no longer recognises - rotated, or typed into a phone
      // that has since been handed on - should stop presenting as edit mode.
      if (keyed() && body && body.editor === false) disable();
    } catch (err) {
      state.writable = false;         // offline, or no worker deployed yet
    }
    state.checked = true;
    return editor();
  }

  const writable = () => state.writable;

  /* ---------- writing ---------- */

  function b64(bytes) {
    let s = '';
    // String.fromCharCode chokes on a whole megabyte of arguments at once.
    for (let i = 0; i < bytes.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(s);
  }

  const b64text = (text) => b64(new TextEncoder().encode(text));

  async function sha1(buf) {
    const digest = await crypto.subtle.digest('SHA-1', buf);
    return [...new Uint8Array(digest)]
      .map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  /** Read a file through the worker, which reads it through the GitHub API and
   *  so always answers with the current version.
   *
   *  raw.githubusercontent is a CDN with a five minute cache, which is right
   *  for a visitor and wrong for somebody who is about to merge an edit into
   *  what they just read. */
  async function readFile(path) {
    if (!WORKER) throw new Error('אין שרת כתיבה מוגדר.');
    const res = await fetch(`${WORKER}/file?path=${encodeURIComponent(path)}`,
                            { cache: 'no-store' });
    if (!res.ok) throw new Error(`קריאה נכשלה (${res.status})`);
    return res.json();
  }

  async function getFile(path) {
    const body = await readFile(path);
    if (!body.content) return { sha: null, json: null };
    const text = new TextDecoder().decode(
      Uint8Array.from(atob(body.content.replace(/\n/g, '')), (c) => c.charCodeAt(0)));
    return { sha: body.sha, json: JSON.parse(text) };
  }

  /** Is this file already in the repo, and under which sha?
   *
   *  Deliberately not getFile: that one parses what it read as JSON, which is
   *  right for the three data documents and throws on the bytes of a photo. A
   *  caller asking only whether a file exists would read that throw as "no",
   *  try to create a file that is already there, and be told - by GitHub, then
   *  by the worker, then by the app - "conflict", forever. */
  async function fileSha(path) {
    try {
      return (await readFile(path)).sha || null;
    } catch (err) {
      return null;
    }
  }

  async function putFile(path, base64, message, sha) {
    if (!WORKER) throw new Error('אין שרת כתיבה מוגדר.');
    const res = await fetch(`${WORKER}/file`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        // Sent whenever this browser holds one. Submitting a trail needs no key
        // and the worker asks for none on that path, so one code serves both.
        ...(keyed() ? { 'X-DK-Key': keyed() } : {})
      },
      body: JSON.stringify({
        path, content: base64, sha: sha || null,
        message: `${message}${named() ? ` (${named()})` : ''}`
      })
    });
    if (res.status === 409) {
      // `conflict` is the flag withDoc retries on. The wording is for the one
      // case that reaches a person: three retries that all lost the race, or a
      // write outside withDoc, where the bare English word said nothing.
      const err = new Error('מישהו אחר כתב באותו רגע.');
      err.conflict = true;
      throw err;
    }
    if (!res.ok) {
      // The worker explains its refusals in Hebrew - too big, too fast, does
      // not look like the document it replaces - and those are worth showing.
      const body = await res.json().catch(() => null);
      throw new Error((body && body.error) || `כתיבה נכשלה (${res.status})`);
    }
    return res.json();
  }

  /** Read a document, let the caller change it, write it back.
   *
   *  Two editors publishing at the same second would otherwise have the second
   *  write silently drop the first trail, so a rejected sha is retried against
   *  freshly read content rather than forced through. */
  async function withDoc(path, key, mutate, message, after, seed) {
    for (let attempt = 0; attempt < 3; attempt++) {
      let { sha, json } = await getFile(path);
      if (!json) {
        // The file is not in the repo yet. The copy shipped with the app is
        // the right starting point: it is what every reader is already seeing.
        json = await bundled(path) || seed;
        if (!json) throw new Error(`${path} חסר בריפו הנתונים.`);
      }
      const doc = json;
      mutate(doc);
      if (after) after(doc);
      doc.updated = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
      try {
        await putFile(path, b64text(JSON.stringify(doc)), message, sha);
        cache(key, doc);
        return doc;
      } catch (err) {
        if (!err.conflict || attempt === 2) throw err;
      }
    }
    return null;
  }

  const withTrails = (mutate, message) =>
    withDoc(TRAILS, K_TRAILS, mutate, message, restat);

  const withPlaces = (mutate, message) =>
    withDoc(PLACES, K_PLACES, mutate, message, (doc) => {
      doc.stats = {
        places: doc.places.length,
        located: doc.places.filter((p) => p.geo).length,
        photos: doc.places.reduce((n, p) => n + (p.photos || []).length, 0)
      };
    });

  function restat(doc) {
    const segs = doc.segments;
    doc.stats = {
      segments: segs.length,
      trips: (doc.trips || []).length,
      waypoints: doc.waypoints.length,
      total_length: segs.reduce((n, s) => n + (s.length || 0), 0),
      photos: [...segs, ...doc.waypoints].reduce((n, s) => n + s.photos.length, 0)
    };
    const pts = [...segs.flatMap((s) => s.path),
                 ...doc.waypoints.map((w) => [w.lat, w.lng])];
    doc.bounds = [
      [Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1]))],
      [Math.max(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[1]))]
    ];
  }

  /** The one item with this id, wherever it is in the document.
   *
   *  Trips are the third array, and they belong here for the same reason as
   *  the other two: renaming, recolouring, adding a link or a photo and
   *  deleting all reach an item through this, and a trip that could be
   *  published and then never touched again would be a dead end. */
  const find = (doc, id) =>
    (doc.segments || []).find((s) => s.id === id) ||
    (doc.waypoints || []).find((w) => w.id === id) ||
    (doc.trips || []).find((t) => t.id === id);

  /* ---------- photos ---------- */

  /** Re-encode to the same two renditions build_data.py writes, and name the
   *  file after a hash of its bytes exactly as that script does. Content
   *  addressing means republishing the same photo cannot duplicate it. */
  function encodeImage(blob, longest, quality) {
    return new Promise((done, fail) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, longest / Math.max(img.width, img.height));
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(img.src);
        canvas.toBlob((out) => (out ? done(out) : fail(new Error('encode failed'))),
                      'image/webp', quality);
      };
      img.onerror = () => fail(new Error('לא הצלחתי לקרוא את התמונה'));
      img.src = URL.createObjectURL(blob);
    });
  }

  async function uploadPhoto(blob, name, onStep) {
    const full = await encodeImage(blob, 1600, 0.82);
    const fullBytes = new Uint8Array(await full.arrayBuffer());
    const key = (await sha1(fullBytes)).slice(0, 14);

    const thumb = await encodeImage(blob, 500, 0.78);
    const thumbBytes = new Uint8Array(await thumb.arrayBuffer());

    const rel = { thumb: `img/${key}_t.webp`, full: `img/${key}.webp` };

    // An identical photo published before already sits there under these names,
    // and its bytes are these bytes - the name is their hash - so there is
    // nothing to write. The two are asked about separately because a publish
    // that died between them left the full image in the repo and no thumbnail,
    // and this is where that heals.
    const [hasFull, hasThumb] = await Promise.all(
      [fileSha(rel.full), fileSha(rel.thumb)]);
    if (hasFull && hasThumb) return rel;

    if (onStep) onStep(`מעלה תמונה…`);
    if (!hasFull) await putFile(rel.full, b64(fullBytes), `תמונה לשביל ${name}`);
    if (!hasThumb) await putFile(rel.thumb, b64(thumbBytes), `תמונה ממוזערת לשביל ${name}`);
    return rel;
  }

  async function uploadAll(blobs, name, onStep) {
    const photos = [];
    let n = 0;
    for (const blob of blobs || []) {
      n++;
      if (onStep && (blobs.length > 1)) onStep(`מעלה תמונה ${n} מתוך ${blobs.length}…`);
      photos.push(await uploadPhoto(blob, name, onStep));
    }
    return photos;
  }

  /* ---------- links ---------- */

  /** Only http(s) survives. A javascript: or data: URL in a shared dataset is
   *  a script anyone can run in everybody else's browser, and this is a file
   *  several people write to. */
  function cleanLinks(links) {
    return (links || []).map((l) => {
      let url = String(l.url || '').trim();
      if (!url) return null;
      if (!/^https?:\/\//i.test(url)) {
        if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return null;   // some other scheme
        url = 'https://' + url;
      }
      let host = '';
      try {
        host = new URL(url).hostname.replace(/^www\./, '');
      } catch (err) {
        return null;
      }
      return { url, title: String(l.title || '').trim().slice(0, 60) || host };
    }).filter(Boolean).slice(0, 8);
  }

  /* ---------- the operations the app offers ---------- */

  /** Publish a local draft into the shared dataset. */
  async function publish(draft, blobs, onStep) {
    const photos = await uploadAll(blobs, draft.name, onStep);
    if (onStep) onStep('מוסיף למסד…');

    const seg = {
      id: 'app-' + Date.now().toString(36),
      name: draft.name,
      note: draft.note || '',
      photos,
      links: cleanLinks(draft.links),
      path: draft.path,
      length: draft.length,
      // No colour means "draw me in my layer's colour", so the field is left
      // out rather than written empty.
      ...(draft.color ? { color: draft.color } : {}),
      connects: [],
      entries: [
        { lat: draft.path[0][0], lng: draft.path[0][1] },
        { lat: draft.path[draft.path.length - 1][0], lng: draft.path[draft.path.length - 1][1] }
      ],
      origin: 'app',
      mode: draft.mode,
      added: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      by: named()
    };
    if (draft.layer) seg.layer = draft.layer;

    // Hand the caller the document we just wrote. Re-reading it would go
    // through the CDN and come back without the trail that was just added.
    const doc = await withTrails((doc2) => { doc2.segments.push(seg); },
      `שביל חדש: ${draft.name}`);
    return { id: seg.id, doc: absolutise(doc) };
  }

  /** Delete a trail or a trip. Both arrays are filtered rather than the caller
   *  having to say which kind it was; an id lives in exactly one of them.
   *
   *  Deleting a *trail* that trips walk along is allowed and leaves them each
   *  short of a piece, which their own page then says out loud. The place to
   *  stop somebody doing that by accident is the trail's page, which lists the
   *  trips that pass through it, not here. */
  const remove = async (id, name) => absolutise(await withTrails((doc) => {
    doc.segments = doc.segments.filter((s) => s.id !== id);
    if (doc.trips) doc.trips = doc.trips.filter((t) => t.id !== id);
  }, `הסרת שביל: ${name}`));

  const rename = async (id, name, note) => absolutise(await withTrails((doc) => {
    const it = find(doc, id);
    if (it) { it.name = name; it.note = note; }
  }, `עדכון שביל: ${name}`));

  /** The colour one trail is drawn in.
   *
   *  Empty removes the field, and the trail then takes its layer's colour -
   *  which is what a trail that carries no colour at all has always done.
   *
   *  The value is checked rather than trusted: it ends up inside a style
   *  attribute in everyone's browser, and this file is one several people
   *  write to. */
  const setColor = async (id, color, name) => {
    if (color && !/^#[0-9a-f]{6}$/i.test(color)) throw new Error('צבע לא תקין.');
    return absolutise(await withTrails((doc) => {
      const it = find(doc, id);
      if (!it) return;
      if (color) it.color = color;
      else delete it.color;
    }, `צבע השביל: ${name}`));
  };

  /** Links on an existing trail or waypoint. */
  const setLinks = async (id, links, name) => absolutise(await withTrails((doc) => {
    const it = find(doc, id);
    if (it) it.links = cleanLinks(links);
  }, `קישורים: ${name}`));

  /** Photos onto an existing trail or waypoint. */
  async function addPhotos(id, blobs, name, onStep) {
    const photos = await uploadAll(blobs, name, onStep);
    if (onStep) onStep('משייך לשביל…');
    return absolutise(await withTrails((doc) => {
      const it = find(doc, id);
      if (it) it.photos = [...(it.photos || []), ...photos];
    }, `תמונות לשביל ${name}`));
  }

  /** Drop a photo from a trail.
   *
   *  The file itself stays in the repo. Names are a hash of the bytes, so the
   *  same photo published twice is one file, and deleting it here could blank
   *  it somewhere else. An orphan webp costs a few hundred kilobytes; a
   *  missing one costs a photo. */
  const removePhoto = async (id, index, name) => absolutise(await withTrails((doc) => {
    const it = find(doc, id);
    if (it && it.photos) it.photos.splice(index, 1);
  }, `הסרת תמונה: ${name}`));

  /* ---------- trail layers ----------
   *
   * Every trail used to land in one layer. A layer here is just a named,
   * coloured bucket recorded in trails.json, and a trail carries the id of its
   * bucket - so making one and moving trails into it are ordinary edits to the
   * same file, in the same commit stream, rather than a second thing to keep
   * in sync.
   */

  const addLayer = async (layer) => absolutise(await withTrails((doc) => {
    doc.layers = doc.layers || [];
    doc.layers.push({
      id: 'tl-' + Date.now().toString(36),
      name: layer.name,
      short: layer.short || layer.name.split(' ')[0].slice(0, 10),
      color: layer.color,
      note: layer.note || '',
      dash: !!layer.dash,
      added: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      by: named()
    });
  }, `שכבה חדשה: ${layer.name}`));

  const editLayer = async (id, patch) => absolutise(await withTrails((doc) => {
    const layer = (doc.layers || []).find((l) => l.id === id);
    if (layer) Object.assign(layer, patch);
  }, `עדכון שכבה: ${patch.name || id}`));

  /** Remove a layer. Its trails move back to the initiative's own layer rather
   *  than disappearing with it - deleting a bucket should never delete walks. */
  const removeLayer = async (id, name) => absolutise(await withTrails((doc) => {
    doc.layers = (doc.layers || []).filter((l) => l.id !== id);
    doc.segments.forEach((s) => { if (s.layer === id) delete s.layer; });
    (doc.waypoints || []).forEach((w) => { if (w.layer === id) delete w.layer; });
  }, `הסרת שכבה: ${name}`));

  const setLayer = async (id, layerId, name) => absolutise(await withTrails((doc) => {
    const it = find(doc, id);
    if (!it) return;
    if (layerId) it.layer = layerId;
    else delete it.layer;
  }, `העברת ${name} לשכבה אחרת`));

  /* ---------- places ---------- */

  /** Pin a pardespedia place onto the map, or move a pin that is off.
   *
   *  build_places.py marks how it derived a position - a name that matched
   *  OpenStreetMap, an address it geocoded, a neighbour it borrowed from - and
   *  never touches one marked manual. So a pin dropped here is permanent, and
   *  a rebuild that would otherwise re-guess leaves it alone. */
  const pinPlace = async (id, lat, lng, name) => absolutise(await withPlaces((doc) => {
    const place = doc.places.find((p) => p.id === id);
    if (place) {
      place.geo = {
        lat: +lat.toFixed(6),
        lng: +lng.toFixed(6),
        source: 'manual',
        by: named(),
        at: new Date().toISOString().replace(/\.\d+Z$/, 'Z')
      };
    }
  }, `מיקום ידני: ${name}`));

  /** Take a pin off, back to whatever the next rebuild works out. */
  const unpinPlace = async (id, name) => absolutise(await withPlaces((doc) => {
    const place = doc.places.find((p) => p.id === id);
    if (place) delete place.geo;
  }, `ביטול מיקום: ${name}`));

  /** Many pins at once, in one commit.
   *
   *  Correcting these positions is bulk work - most of them are wrong and only
   *  a person who lives here can say where they belong. A write per drag would
   *  mean a round trip per drag and a commit log of several hundred entries for
   *  one afternoon's work. */
  /* ---------- the queue ----------
   *
   * A trail somebody sent in, not yet on the map. Before the worker this queue
   * ran on WhatsApp: export a file, send it, and an editor imported it by hand.
   * That worked and it lost trails, because a file in a chat is a thing
   * somebody has to remember.
   *
   * Approving is deliberately two writes rather than one, in this order: add
   * to trails first, drop from pending second. If the second fails the trail is
   * on the map twice over, which somebody will notice and can fix. The other
   * order loses a walk.
   */

  const PENDING = 'data/pending.json';
  const K_PENDING = 'dk.cache.pending.v1';
  const emptyQueue = () => ({ version: 1, updated: '', items: [] });

  const withPending = (mutate, message) =>
    withDoc(PENDING, K_PENDING, mutate, message, null, emptyQueue());

  async function queue() {
    if (!WORKER) return emptyQueue();
    try {
      const { json } = await getFile(PENDING);
      return json || emptyQueue();
    } catch (err) {
      return emptyQueue();
    }
  }

  /** A trip straight onto the map. Only an editor gets here.
   *
   *  What is written is the recipe and not the line: which shortcut, which way
   *  round, which drawn points. `trips` is a second array in the same document
   *  as the shortcuts it names, so a trail and the trips built on it are always
   *  written and read together and can never disagree about what exists. */
  async function publishTrip(draft, blobs, onStep) {
    const photos = await uploadAll(blobs, draft.name, onStep);
    if (onStep) onStep('מוסיף למסד…');

    const trip = {
      id: 'trip-' + Date.now().toString(36),
      name: draft.name,
      note: draft.note || '',
      photos,
      links: cleanLinks(draft.links),
      parts: draft.parts,
      ...(draft.difficulty ? { difficulty: draft.difficulty } : {}),
      ...(draft.minutes ? { minutes: draft.minutes } : {}),
      origin: 'app',
      added: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      by: named()
    };
    const doc = await withTrails((d) => { (d.trips = d.trips || []).push(trip); },
      `טיול חדש: ${draft.name}`);
    return { id: trip.id, doc: absolutise(doc) };
  }

  /** Send a trail in for review. Needs nothing from the sender. */
  async function submit(draft, blobs, onStep) {
    const photos = await uploadAll(blobs, draft.name, onStep);
    if (onStep) onStep('שולח…');
    const item = {
      id: 'sub-' + Date.now().toString(36),
      name: draft.name,
      note: draft.note || '',
      photos,
      links: cleanLinks(draft.links),
      // A trip goes into the queue as its recipe, a trail as its line. The
      // reviewer sees both drawn the same way; only the approval differs.
      ...(draft.parts
        ? { parts: draft.parts,
            ...(draft.difficulty ? { difficulty: draft.difficulty } : {}),
            ...(draft.minutes ? { minutes: draft.minutes } : {}) }
        : { path: draft.path }),
      length: draft.length,
      // The colour the sender picked, carried through to approval. The queue
      // draws every item in its own yellow regardless, so this is not what it
      // looks like while it waits.
      ...(draft.color ? { color: draft.color } : {}),
      mode: draft.mode,
      submitted: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      by: named()
    };
    await withPending((doc) => { doc.items.push(item); },
      `${draft.parts ? 'טיול' : 'שביל'} שהתקבל: ${draft.name}`);
    return item.id;
  }

  /** Move a queued trail onto the map. */
  async function approve(id) {
    const doc = await queue();
    const item = (doc.items || []).find((i) => i.id === id);
    if (!item) throw new Error('השביל כבר לא בתור.');

    // A queued trip is approved into the trips array. Everything else about
    // the queue - who sent it, its photos, dropping it afterwards - is the
    // same, so only the shape of what is appended differs.
    if (item.parts) {
      const trip = {
        id: 'trip-' + Date.now().toString(36),
        name: item.name,
        note: item.note || '',
        photos: item.photos || [],
        links: item.links || [],
        parts: item.parts,
        ...(item.difficulty ? { difficulty: item.difficulty } : {}),
        ...(item.minutes ? { minutes: item.minutes } : {}),
        origin: 'app',
        added: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
        by: item.by || '',
        approved_by: named()
      };
      const withTrip = await withTrails((d) => { (d.trips = d.trips || []).push(trip); },
        `אישור טיול: ${item.name}`);
      await withPending((d) => { d.items = d.items.filter((i) => i.id !== id); },
        `הוסר מהתור: ${item.name}`);
      return { id: trip.id, doc: absolutise(withTrip) };
    }

    const seg = {
      id: 'app-' + Date.now().toString(36),
      name: item.name,
      note: item.note || '',
      photos: item.photos || [],
      links: item.links || [],
      path: item.path,
      length: item.length,
      // Items queued before a sender could pick a colour were stamped with the
      // queue's own yellow, which was never a statement about the map.
      ...(item.color && item.color !== '#f9a825' ? { color: item.color } : {}),
      connects: [],
      entries: [
        { lat: item.path[0][0], lng: item.path[0][1] },
        { lat: item.path[item.path.length - 1][0], lng: item.path[item.path.length - 1][1] }
      ],
      origin: 'app',
      mode: item.mode,
      added: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      by: item.by || '',
      approved_by: named()
    };
    const trails = await withTrails((d) => { d.segments.push(seg); },
      `אישור שביל: ${item.name}`);
    await withPending((d) => { d.items = d.items.filter((i) => i.id !== id); },
      `הוסר מהתור: ${item.name}`);
    return { id: seg.id, doc: absolutise(trails) };
  }

  const reject = (id, name) => withPending(
    (d) => { d.items = d.items.filter((i) => i.id !== id); },
    `נדחה מהתור: ${name}`);

  const movePlaces = async (changes) => absolutise(await withPlaces((doc) => {
    const at = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
    const byId = new Map(doc.places.map((p) => [p.id, p]));
    changes.forEach(({ id, lat, lng }) => {
      const place = byId.get(id);
      if (!place) return;
      place.geo = { lat, lng, source: 'manual', by: named(), at };
    });
  }, `עדכון מיקומים: ${changes.length} מקומות`));

  return {
    RAW, OWNER, REPO, WORKER,
    load, asset, cleanLinks,
    isEditor, editor, editing, named, enable, disable, resume, writable,
    publish, publishTrip, remove, rename, setLinks, setColor, addPhotos, removePhoto,
    addLayer, editLayer, removeLayer, setLayer,
    pinPlace, unpinPlace, movePlaces,
    queue, submit, approve, reject,
    get offline() { return state.offline; }
  };
})();
