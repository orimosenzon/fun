/* דרך קיצור - the write path.
 *
 * The app is a static page on GitHub Pages, and the dataset is a public GitHub
 * repo. Writing to a repo needs a GitHub credential, and a static page has
 * nowhere to keep one: anything shipped to the browser is readable by anyone.
 *
 * Until now that credential was a fine-grained token each editor pasted into
 * their own browser. It worked, and it meant every editor needed a GitHub
 * account and a five-step token dance - which in a town of eleven thousand
 * people means exactly one editor, forever.
 *
 * So the token moves here instead. This worker holds it as a secret, and the
 * app calls the worker with no credential at all. That is a deliberate choice
 * with a real cost: the dataset becomes writable by anyone who finds this URL.
 * The guards below are what make that survivable rather than safe -
 *
 *   allowlist    only three data files and content-addressed webp images can
 *                be written. Nothing else in the repo is reachable, not
 *                workflows, not the app's own code.
 *   editor key   the two files that *are* the map - the trails and the places -
 *                need a shared secret. See GATED below.
 *   schema       a document has to still look like the document it replaces,
 *                with a sane number of items, or the write is refused.
 *   size         a hard cap per request and per file.
 *   rate limit   per IP, in a KV namespace, so a script cannot hammer it.
 *   git          every write is a commit by a bot account. Nothing is ever
 *                really destroyed, and `git revert` undoes a bad day.
 *
 * See README for the kill switch and for what this does and does not protect.
 */

const REPO_API = 'https://api.github.com/repos';

/* Exactly what may be written, and nothing else. Image names are the first
 * fourteen hex characters of the SHA-1 of the file's own bytes, which is how
 * the app and build_data.py both name them. */
const ALLOWED = [
  /^data\/trails\.json$/,
  /^data\/places\.json$/,
  /^data\/pending\.json$/,
  /^img\/[0-9a-f]{14}(_t)?\.webp$/
];

/* The files an editor's key is needed for.
 *
 * Approving a submitted trail is exactly a write to data/trails.json, so this
 * list is what makes "only an editor decides what goes on the map" true rather
 * than a convention. The two paths left out are left out on purpose:
 *
 *   data/pending.json   a resident sending a trail in writes here, and asking
 *                       them for a secret first would mean nobody ever sends
 *                       one. The queue is a waiting room, not the map.
 *   img/*.webp          those submissions carry photos. Names are the hash of
 *                       the bytes, so a write here can only ever add a file
 *                       nobody is pointing at yet.
 *
 * The cost of that: somebody who finds this URL can flood the queue or empty
 * it. Both are visible, neither reaches the map, and every one is a commit.
 */
const GATED = [
  /^data\/trails\.json$/,
  /^data\/places\.json$/
];

const MAX_BODY = 6 * 1024 * 1024;        // one photo, comfortably
const RATE_MAX = 40;                     // writes per window, per IP
const RATE_WINDOW = 300;                 // seconds

/* Wrong passwords per window, per IP.
 *
 * Checking a key is the one operation here that answers a question worth
 * asking repeatedly, so without this the endpoint is a free oracle and the
 * password is only as good as the number of guesses per second allows. A
 * correct key costs nothing from this budget; only a wrong one does. */
const AUTH_MAX = 10;

/* The app is served from GitHub Pages and this worker lives on a different
 * host, so every call is cross-origin.
 *
 * The origin is not restricted, and restricting it would be theatre: CORS
 * governs what a *browser* will hand back to a page, and this endpoint takes
 * no credential, so anyone who wants to write can use curl and never involve
 * a browser at all. The guards that matter are the allowlist, the schema check
 * and the rate limit. */
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, PUT, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, X-DK-Key',
  'Access-Control-Max-Age': '86400'
};


const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { 'Content-Type': 'application/json; charset=utf-8', ...CORS }
});

const gh = (env, path, init = {}) => fetch(
  `${REPO_API}/${env.OWNER}/${env.REPO}/${path}`,
  {
    ...init,
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'derech-kitzur-worker',
      ...(init.headers || {})
    }
  }
);

/* ---------- guards ---------- */

const allowed = (path) => ALLOWED.some((re) => re.test(path));
const gated = (path) => GATED.some((re) => re.test(path));

/** Compare without letting the time taken say how much of the key was right.
 *  Both sides are hashed first so the comparison runs over a fixed length
 *  whatever the two strings were. */
async function sameSecret(a, b) {
  const digest = async (s) => new Uint8Array(
    await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)));
  const [x, y] = await Promise.all([digest(a), digest(b)]);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

/** Is this request carrying the editor's key?
 *
 *  With no EDITOR_KEY configured this answers no, and the gated paths become
 *  unwritable by anybody. That direction is deliberate: a worker deployed
 *  without its secret should refuse edits rather than accept every one. */
async function isEditor(env, request, ip) {
  const given = request.headers.get('X-DK-Key') || '';
  if (!env.EDITOR_KEY || !given) return false;
  // Spent budget means stop answering, right or wrong: an oracle that still
  // replies once the limit is hit is not limited at all.
  if (await overRate(env, 'a', ip, AUTH_MAX, false)) return false;
  if (await sameSecret(given, env.EDITOR_KEY)) return true;
  await overRate(env, 'a', ip, AUTH_MAX, true);
  return false;
}

/** Count one event against a per-IP budget, and say whether it is spent.
 *
 *  Without KV bound the worker still runs - losing the limiter should not take
 *  the whole write path down with it. `bump: false` reads the budget without
 *  spending any of it, which is how a correct password avoids being charged
 *  for the guesses somebody else made from the same address. */
async function overRate(env, bucket, ip, max, bump = true) {
  if (!env.RATE) return false;
  const key = `${bucket}:${ip}`;
  const now = Math.floor(Date.now() / 1000);
  let state;
  try {
    state = await env.RATE.get(key, 'json');
  } catch (err) {
    return false;
  }
  if (!state || now - state.since > RATE_WINDOW) {
    state = { since: now, n: 0 };
  }
  if (!bump) return state.n > max;
  state.n += 1;
  try {
    await env.RATE.put(key, JSON.stringify(state), { expirationTtl: RATE_WINDOW * 2 });
  } catch (err) {
    /* a failed write to the limiter is not a reason to refuse the edit */
  }
  return state.n > max;
}

/** A replacement document has to still be the kind of document it replaces.
 *
 *  This is the guard that matters most. It cannot tell a good edit from a
 *  malicious one, but it does stop the whole dataset being replaced with
 *  something else, truncated to nothing, or blown up to a size that would
 *  make the app unusable for everyone. */
function sane(path, text) {
  let doc;
  try {
    doc = JSON.parse(text);
  } catch (err) {
    return 'הקובץ אינו JSON תקין.';
  }
  if (!doc || typeof doc !== 'object') return 'מבנה לא צפוי.';

  const check = (key, min, max) => {
    const list = doc[key];
    if (!Array.isArray(list)) return `${key} חסר.`;
    if (list.length > max) return `${key}: ${list.length} פריטים, יותר מדי.`;
    if (list.length < min) return `${key}: ${list.length} פריטים, נראה כמו מחיקה בטעות.`;
    return null;
  };

  if (path === 'data/trails.json') {
    // `trips` is only checked once it exists. A document written before trips
    // were a thing has no such key, and demanding one here would lock every
    // write out of an older dataset the moment this is deployed.
    return check('segments', 1, 3000) || check('waypoints', 0, 3000)
      || (doc.trips === undefined ? null : check('trips', 0, 1000));
  }
  if (path === 'data/places.json') {
    return check('places', 50, 5000);
  }
  if (path === 'data/pending.json') {
    return check('items', 0, 500);
  }
  return null;
}

/* ---------- the two operations ---------- */

async function read(env, path) {
  if (!allowed(path)) return json({ error: 'path not allowed' }, 403);
  // A throwaway parameter defeats any cache between here and GitHub: a publish
  // that merges into a stale read silently drops whatever arrived meanwhile.
  const res = await gh(env, `contents/${path}?ref=${env.BRANCH || 'main'}&t=${Date.now()}`);
  if (res.status === 404) return json({ sha: null, content: null });
  if (!res.ok) return json({ error: `github ${res.status}` }, 502);
  const body = await res.json();
  return json({ sha: body.sha, content: body.content, encoding: body.encoding });
}

async function write(env, request, ip) {
  if (env.READ_ONLY === 'true') {
    return json({ error: 'הכתיבה מושהית זמנית.' }, 503);
  }
  if (await overRate(env, 'w', ip, RATE_MAX)) {
    return json({ error: 'יותר מדי שינויים בזמן קצר. נסה בעוד כמה דקות.' }, 429);
  }

  const raw = await request.text();
  if (raw.length > MAX_BODY) return json({ error: 'הבקשה גדולה מדי.' }, 413);

  let body;
  try {
    body = JSON.parse(raw);
  } catch (err) {
    return json({ error: 'bad request' }, 400);
  }
  const { path, content, message, sha } = body;
  if (typeof path !== 'string' || typeof content !== 'string') {
    return json({ error: 'bad request' }, 400);
  }
  if (!allowed(path)) return json({ error: 'path not allowed' }, 403);

  // Checked after the allowlist and before anything is parsed, so a wrong key
  // never gets as far as a schema opinion about the body it sent.
  if (gated(path) && !(await isEditor(env, request, ip))) {
    return json({ error: 'רק עורך יכול לשנות את המפה עצמה. הזן סיסמת עריכה.' }, 401);
  }

  if (path.endsWith('.json')) {
    let text;
    try {
      text = new TextDecoder().decode(
        Uint8Array.from(atob(content), (c) => c.charCodeAt(0)));
    } catch (err) {
      return json({ error: 'bad encoding' }, 400);
    }
    const complaint = sane(path, text);
    if (complaint) return json({ error: complaint }, 422);
  }

  const res = await gh(env, `contents/${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      branch: env.BRANCH || 'main',
      message: String(message || 'עדכון מהאפליקציה').slice(0, 200),
      ...(sha ? { sha } : {})
    })
  });

  // A rejected sha means somebody else wrote first. The app knows how to read
  // again and retry, so this has to come back distinguishable.
  //
  // GitHub says far more than the status does - a rejected sha, a ruleset, a
  // blocked push and a body it did not like all arrive as 409 or 422 - and
  // swallowing that leaves an editor staring at the word "conflict" with no way
  // to find out which. It goes to the log, where `wrangler tail` picks it up.
  if (!res.ok) {
    const why = await res.text().catch(() => '');
    console.log(`github ${res.status} on PUT ${path}: ${why.slice(0, 400)}`);
    if (res.status === 409 || res.status === 422) return json({ error: 'conflict' }, 409);
    return json({ error: `github ${res.status}` }, 502);
  }
  return json({ ok: true });
}

/* ---------- counting ----------
 *
 * Whether anybody uses this is a question the project could not answer at all,
 * and every decision about what to build next was being made without it.
 *
 * What is stored is one integer per day per event, and nothing else. No IP, no
 * cookie, no user agent, no session id, no path, no referrer - there is nothing
 * in this store that could be traced to a person even by whoever holds the keys
 * to it, which is the only version of counting worth having on a map that shows
 * where residents walk. The counts live in the project's own Cloudflare account
 * and reach no third party.
 *
 * Read-modify-write, so two visits landing in the same instant can cost a
 * count. That is the right trade here: this is a village-scale number used to
 * tell "nobody" from "a few dozen", and an occasional lost tick does not change
 * that answer. It is not billing.
 */

const STAT_EVENTS = new Set([
  'open',      // the app was opened
  'draft',     // somebody started recording or drawing a trail
  'trip',      // somebody started building a trip
  'send'       // somebody sent a trail in for review
]);

const STAT_TTL = 120 * 24 * 60 * 60;     // four months, then the day drops out
const STAT_MAX = 60;                     // events per window, per IP

const statKey = (event) => `d:${new Date().toISOString().slice(0, 10)}|${event}`;

async function count(env, request, ip) {
  // Counting is optional by construction: without the binding the endpoint
  // answers politely and the app carries on, so a worker deployed without the
  // namespace is degraded rather than broken.
  if (!env.STATS) return json({ ok: true, counted: false });
  if (await overRate(env, 's', ip, STAT_MAX)) return json({ ok: true, counted: false });

  let event = '';
  try {
    // text/plain rather than a JSON content type, so the beacon the app sends
    // stays a simple request and needs no preflight it cannot make.
    event = String(JSON.parse(await request.text()).event || '');
  } catch (err) {
    return json({ error: 'bad body' }, 400);
  }
  // A fixed list, so nobody can turn this into free storage of their own.
  if (!STAT_EVENTS.has(event)) return json({ error: 'unknown event' }, 400);

  const key = statKey(event);
  const now = parseInt(await env.STATS.get(key), 10) || 0;
  await env.STATS.put(key, String(now + 1), { expirationTtl: STAT_TTL });
  return json({ ok: true, counted: true });
}

/** Every day counted, newest first. Open in a browser and read it.
 *
 *  Deliberately ungated: these are aggregate numbers about a public map, and
 *  the point of measuring is that the answer is one tap away on a phone. */
async function stats(env) {
  if (!env.STATS) return json({ error: 'המדידה כבויה' }, 503);
  const { keys } = await env.STATS.list({ prefix: 'd:' });
  const days = {};
  await Promise.all(keys.map(async ({ name }) => {
    const [date, event] = name.slice(2).split('|');
    const n = parseInt(await env.STATS.get(name), 10) || 0;
    days[date] = days[date] || { date };
    days[date][event] = n;
  }));
  const rows = Object.values(days).sort((a, b) => b.date.localeCompare(a.date));
  const sum = (event) => rows.reduce((n, d) => n + (d[event] || 0), 0);
  return json({
    days: rows,
    total: { open: sum('open'), draft: sum('draft'), trip: sum('trip'), send: sum('send') }
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    // `editor` is how the app finds out whether the key it is holding is the
    // right one, so that a wrong password is refused when it is typed rather
    // than an hour later at the end of a walk.
    if (url.pathname === '/health') {
      return json({ ok: true, repo: `${env.OWNER}/${env.REPO}`,
                    readOnly: env.READ_ONLY === 'true',
                    editor: await isEditor(env, request, ip) });
    }

    if (url.pathname === '/file') {
      if (request.method === 'GET') {
        return read(env, url.searchParams.get('path') || '');
      }
      if (request.method === 'PUT') {
        return write(env, request, ip);
      }
    }

    if (url.pathname === '/stat' && request.method === 'POST') {
      return count(env, request, ip);
    }

    if (url.pathname === '/stats') {
      return stats(env);
    }

    return json({ error: 'not found' }, 404);
  }
};
