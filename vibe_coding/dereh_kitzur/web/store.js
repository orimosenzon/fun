/* Store - where the trails actually live.
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
 * A visitor needs nothing to read. An editor pastes a fine-grained token once
 * per device; it is scoped to the data repo alone, so the worst a leaked one
 * can do is edit trail data that is public anyway - and every write is a
 * commit, which means nothing is ever really lost.
 */
'use strict';

const Store = (() => {

  const OWNER = 'orimosenzon';
  const REPO = 'derech-kitzur-data';
  const BRANCH = 'main';

  const RAW = `https://raw.githubusercontent.com/${OWNER}/${REPO}/${BRANCH}/`;
  const API = `https://api.github.com/repos/${OWNER}/${REPO}`;
  const TOKEN_HELP = `https://github.com/settings/personal-access-tokens/new`;

  const K_TRAILS = 'dk.cache.trails.v2';
  const K_NET = 'dk.cache.network.v2';
  const K_TOKEN = 'dk.token.v1';

  const state = { offline: false, editor: null };

  /* Photo paths are stored relative to the data repo, so they resolve the same
   * whether the app is served from Pages, from localhost, or from a file the
   * editor is reviewing before publishing. */
  const asset = (p) => (!p || /^(https?:|blob:|data:)/.test(p) ? p : RAW + p);

  function absolutise(doc) {
    [...(doc.segments || []), ...(doc.waypoints || [])].forEach((it) => {
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
   *  own trail would disappear for five minutes. An editor already holds a
   *  token, so read through the API, which is always current. */
  async function canonical() {
    if (isEditor()) {
      try {
        const { json } = await getFile('data/trails.json');
        if (json) return json;
      } catch (err) {
        /* token expired or offline; the public copy still works */
      }
    }
    return fetchJson('data/trails.json');
  }

  /** The canonical data, with three fallbacks so the app opens on a dead
   *  connection: the live repo, then the last copy we saw, then the seed
   *  bundled with the app itself. */
  async function load() {
    let trails = null;
    let network = null;
    try {
      [trails, network] = await Promise.all([
        canonical(),
        fetchJson('data/layers.json').catch(() => null)
      ]);
      cache(K_TRAILS, trails);
      if (network) cache(K_NET, network);
    } catch (err) {
      state.offline = true;
      trails = cached(K_TRAILS);
      network = cached(K_NET);
      if (!trails) {
        // First ever visit, with no connection. The copy shipped with the app
        // is stale by definition, but it beats an empty map.
        trails = await fetch('data/trails.json').then((r) => r.json());
        network = await fetch('data/layers.json').then((r) => r.json()).catch(() => null);
      }
    }
    return { trails: absolutise(trails), network, offline: state.offline };
  }

  /* ---------- the editor's token ---------- */

  const token = () => localStorage.getItem(K_TOKEN) || '';
  const isEditor = () => !!token();
  const editor = () => state.editor;

  const auth = (extra) => ({
    Authorization: `Bearer ${token()}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    ...extra
  });

  /** Check a token before storing it, so a typo fails at the settings screen
   *  rather than halfway through a publish. */
  async function signIn(value) {
    const probe = await fetch(API, {
      headers: {
        Authorization: `Bearer ${value.trim()}`,
        Accept: 'application/vnd.github+json'
      }
    });
    if (probe.status === 401) throw new Error('המפתח לא תקף או פג תוקפו.');
    if (!probe.ok) throw new Error('אין למפתח הזה גישה לריפו הנתונים.');
    const repo = await probe.json();
    if (!repo.permissions || !repo.permissions.push) {
      throw new Error('המפתח קורא אבל לא כותב. צריך הרשאת Contents: Read and write.');
    }
    localStorage.setItem(K_TOKEN, value.trim());

    const me = await fetch('https://api.github.com/user', { headers: auth() });
    state.editor = me.ok ? (await me.json()).login : 'עורך';
    return state.editor;
  }

  function signOut() {
    localStorage.removeItem(K_TOKEN);
    state.editor = null;
  }

  /** Restore the editor's name on boot without blocking anything. */
  async function resume() {
    if (!isEditor()) return null;
    try {
      const me = await fetch('https://api.github.com/user', { headers: auth() });
      if (me.status === 401) { signOut(); return null; }
      state.editor = me.ok ? (await me.json()).login : 'עורך';
    } catch (err) {
      state.editor = 'עורך';        // offline; the token is still probably fine
    }
    return state.editor;
  }

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

  async function getFile(path) {
    // The read has to be fresh, or a publish merges into a stale file and
    // drops whatever was added meanwhile. A Cache-Control header would be the
    // obvious way to say so, but GitHub does not list it in
    // Access-Control-Allow-Headers, so sending one fails the CORS preflight
    // and the whole request never leaves the browser. A throwaway query
    // parameter changes the cache key instead, and the API ignores it.
    const res = await fetch(`${API}/contents/${path}?ref=${BRANCH}&t=${Date.now()}`, {
      headers: auth()
    });
    if (res.status === 404) return { sha: null, json: null };
    if (!res.ok) throw new Error(`קריאה נכשלה (${res.status})`);
    const body = await res.json();
    const text = new TextDecoder().decode(
      Uint8Array.from(atob(body.content.replace(/\n/g, '')), (c) => c.charCodeAt(0)));
    return { sha: body.sha, json: JSON.parse(text) };
  }

  async function putFile(path, base64, message, sha) {
    const res = await fetch(`${API}/contents/${path}`, {
      method: 'PUT',
      headers: auth({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ message, content: base64, branch: BRANCH, ...(sha ? { sha } : {}) })
    });
    if (res.status === 409 || res.status === 422) {
      const err = new Error('conflict');
      err.conflict = true;
      throw err;
    }
    if (!res.ok) throw new Error(`כתיבה נכשלה (${res.status})`);
    return res.json();
  }

  /** Read trails.json, let the caller change it, write it back.
   *
   *  Two editors publishing at the same second would otherwise have the second
   *  write silently drop the first trail, so a rejected sha is retried against
   *  freshly read content rather than forced through. */
  async function withTrails(mutate, message) {
    for (let attempt = 0; attempt < 3; attempt++) {
      const { sha, json } = await getFile('data/trails.json');
      const doc = json;
      mutate(doc);
      restat(doc);
      doc.updated = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
      try {
        await putFile('data/trails.json', b64text(JSON.stringify(doc)), message, sha);
        cache(K_TRAILS, doc);
        return doc;
      } catch (err) {
        if (!err.conflict || attempt === 2) throw err;
      }
    }
    return null;
  }

  function restat(doc) {
    const segs = doc.segments;
    doc.stats = {
      segments: segs.length,
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

    // An identical photo published before already sits there under this name.
    const exists = await fetch(`${API}/contents/${rel.full}?ref=${BRANCH}`, { headers: auth() });
    if (exists.ok) return rel;

    if (onStep) onStep(`מעלה תמונה…`);
    await putFile(rel.full, b64(fullBytes), `תמונה לשביל ${name}`);
    await putFile(rel.thumb, b64(thumbBytes), `תמונה ממוזערת לשביל ${name}`);
    return rel;
  }

  /* ---------- the operations the app offers ---------- */

  /** Publish a local draft into the shared dataset. */
  async function publish(draft, blobs, onStep) {
    const photos = [];
    for (const blob of blobs || []) {
      photos.push(await uploadPhoto(blob, draft.name, onStep));
    }
    if (onStep) onStep('מוסיף למסד…');

    const seg = {
      id: 'app-' + Date.now().toString(36),
      name: draft.name,
      note: draft.note || '',
      photos,
      path: draft.path,
      length: draft.length,
      color: '#097138',
      connects: [],
      entries: [
        { lat: draft.path[0][0], lng: draft.path[0][1] },
        { lat: draft.path[draft.path.length - 1][0], lng: draft.path[draft.path.length - 1][1] }
      ],
      origin: 'app',
      mode: draft.mode,
      added: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      by: state.editor || ''
    };

    // Hand the caller the document we just wrote. Re-reading it would go
    // through the CDN and come back without the trail that was just added.
    const doc = await withTrails((doc2) => { doc2.segments.push(seg); },
      `שביל חדש: ${draft.name}`);
    return { id: seg.id, doc: absolutise(doc) };
  }

  const remove = async (id, name) => absolutise(await withTrails(
    (doc) => { doc.segments = doc.segments.filter((s) => s.id !== id); },
    `הסרת שביל: ${name}`));

  const rename = async (id, name, note) => absolutise(await withTrails((doc) => {
    const seg = doc.segments.find((s) => s.id === id);
    if (seg) { seg.name = name; seg.note = note; }
  }, `עדכון שביל: ${name}`));

  return {
    RAW, OWNER, REPO, TOKEN_HELP,
    load, asset, isEditor, editor, signIn, signOut, resume,
    publish, remove, rename,
    get offline() { return state.offline; }
  };
})();
