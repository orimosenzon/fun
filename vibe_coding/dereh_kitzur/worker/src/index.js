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
 *   schema       a document has to still look like the document it replaces,
 *                with a sane number of items, or the write is refused.
 *   size         a hard cap per request and per file.
 *   rate limit   per IP, in a KV namespace, so a script cannot hammer it.
 *   git          every write is a commit by a bot account. Nothing is ever
 *                really destroyed, and `git revert` undoes a bad day.
 *
 * There is deliberately no login. See README for what that does and does not
 * protect, and for the kill switch when it stops being a good trade.
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

const MAX_BODY = 6 * 1024 * 1024;        // one photo, comfortably
const RATE_MAX = 40;                     // writes per window, per IP
const RATE_WINDOW = 300;                 // seconds

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
  'Access-Control-Allow-Methods': 'GET, PUT, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
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

/** Rate limit per IP. Without KV bound the worker still runs - losing the
 *  limiter should not take the whole write path down with it. */
async function overRate(env, ip) {
  if (!env.RATE) return false;
  const key = `w:${ip}`;
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
  state.n += 1;
  try {
    await env.RATE.put(key, JSON.stringify(state), { expirationTtl: RATE_WINDOW * 2 });
  } catch (err) {
    /* a failed write to the limiter is not a reason to refuse the edit */
  }
  return state.n > RATE_MAX;
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
    return check('segments', 1, 3000) || check('waypoints', 0, 3000);
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
  if (await overRate(env, ip)) {
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
  if (res.status === 409 || res.status === 422) return json({ error: 'conflict' }, 409);
  if (!res.ok) return json({ error: `github ${res.status}` }, 502);
  return json({ ok: true });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (url.pathname === '/health') {
      return json({ ok: true, repo: `${env.OWNER}/${env.REPO}`,
                    readOnly: env.READ_ONLY === 'true' });
    }

    if (url.pathname === '/file') {
      if (request.method === 'GET') {
        return read(env, url.searchParams.get('path') || '');
      }
      if (request.method === 'PUT') {
        return write(env, request, ip);
      }
    }

    return json({ error: 'not found' }, 404);
  }
};
