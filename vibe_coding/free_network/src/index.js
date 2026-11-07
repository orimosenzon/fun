import { Hono } from 'hono';
import { deleteCookie, getCookie, setCookie } from 'hono/cookie';
import { writeLog } from './log.js';

const SESSION_COOKIE = 'session';
const SESSION_MS = 90 * 24 * 60 * 60 * 1000;
const LOGIN_TOKEN_MS = 15 * 60 * 1000;
const INVITE_MS = 7 * 24 * 60 * 60 * 1000;
const LOGIN_REQUESTS_PER_HOUR = 5;
const PAGE_SIZE = 30;
const MAX_POST = 5000;
const MAX_COMMENT = 2000;
const MAX_NAME = 60;
const MAX_IMAGE_BYTES = 1024 * 1024;
const MAX_IMAGES_PER_POST = 4;
const IMAGES_PER_DAY = 40;
// R2 has no spending cap of its own and bills past its free tier (10 GB, 1M writes a month).
// These limits keep the whole network inside it, whatever anyone does.
const STORAGE_LIMIT_BYTES = 8 * 1024 ** 3;
const UPLOADS_PER_DAY = 2000;
const YOUTUBE_ID = /^[\w-]{11}$/;
const VIDEO_CACHE_SECONDS = 24 * 60 * 60;
const GOOGLE_STATE_COOKIE = 'google_state';
const GOOGLE_STATE_PATH = '/api/auth/google';
const DEMO_SESSION = 'demo';
const DEMO_WRITES = ['/api/auth/demo', '/api/auth/logout'];

const FRIEND_IDS = `
  SELECT addressee_id FROM friendships WHERE requester_id = ?1 AND status = 'accepted'
  UNION
  SELECT requester_id FROM friendships WHERE addressee_id = ?1 AND status = 'accepted'`;

function randomToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export function findYoutubeId(text) {
  for (const [raw] of text.matchAll(/https?:\/\/[^\s<>"]+/g)) {
    try {
      const url = new URL(raw);
      const host = url.hostname.replace(/^(www|m|music)\./, '');
      let id = null;
      if (host === 'youtu.be') id = url.pathname.slice(1);
      else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
        id = url.searchParams.get('v') ?? url.pathname.match(/^\/(?:embed|shorts|live)\/([^/]+)/)?.[1];
      }
      if (/^[\w-]{11}$/.test(id ?? '')) return id;
    } catch {
      // Not a valid URL, keep looking.
    }
  }
  return null;
}

// Rebuilds a JPEG without its metadata, which is where phones put the GPS location.
// Browsers strip it too, but the server must not depend on that. Returns null if not a JPEG.
export function stripJpegMetadata(bytes) {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  const kept = [bytes.subarray(0, 2)];
  let i = 2;
  while (i + 4 <= bytes.length) {
    if (bytes[i] !== 0xff) return null;
    const marker = bytes[i + 1];
    if (marker === 0xda) {
      kept.push(bytes.subarray(i));
      const out = new Uint8Array(kept.reduce((n, part) => n + part.length, 0));
      let at = 0;
      for (const part of kept) {
        out.set(part, at);
        at += part.length;
      }
      return out;
    }
    const end = i + 2 + ((bytes[i + 2] << 8) | bytes[i + 3]);
    if (end > bytes.length) return null;
    const isMetadata = (marker >= 0xe1 && marker <= 0xef) || marker === 0xfe;
    if (!isMetadata) kept.push(bytes.subarray(i, end));
    i = end;
  }
  return null;
}

function cleanText(value, max) {
  if (typeof value !== 'string') return null;
  const text = value.trim();
  return text && text.length <= max ? text : null;
}

async function readJson(c) {
  try {
    return await c.req.json();
  } catch {
    return {};
  }
}

function parseId(value) {
  return /^\d{1,15}$/.test(value) ? Number(value) : null;
}

function isLocalhost(c) {
  return ['localhost', '127.0.0.1'].includes(new URL(c.req.url).hostname);
}

function isDemo(c) {
  return c.env.DEMO === '1';
}

function isAdmin(c) {
  return !!c.env.ADMIN_EMAIL && c.get('user')?.email === c.env.ADMIN_EMAIL;
}

async function imageUsage(c) {
  const row = await c.env.DB
    .prepare('SELECT COALESCE(SUM(bytes), 0) AS bytes, COUNT(CASE WHEN created_at > ? THEN 1 END) AS today FROM images')
    .bind(Date.now() - 24 * 60 * 60 * 1000)
    .first();
  return {
    ...row,
    storageLimit: Number(c.env.STORAGE_LIMIT_BYTES ?? STORAGE_LIMIT_BYTES),
    uploadLimit: Number(c.env.UPLOADS_PER_DAY ?? UPLOADS_PER_DAY),
  };
}

function startSession(c, token) {
  setCookie(c, SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: 'Lax',
    secure: new URL(c.req.url).protocol === 'https:',
    path: '/',
    maxAge: SESSION_MS / 1000,
  });
}

async function createSession(c, userId) {
  const session = randomToken();
  await c.env.DB
    .prepare('INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)')
    .bind(await sha256(session), userId, Date.now() + SESSION_MS)
    .run();
  startSession(c, session);
}

function googleRedirectUri(c) {
  return `${new URL(c.req.url).origin}/api/auth/google/callback`;
}

// Gmail ignores dots and case in the name part, so an invitation matches however it was typed.
export function normalizeGoogleEmail(value) {
  const email = String(value).trim().toLowerCase();
  const [name, domain] = email.split('@');
  if (!name || !domain) return null;
  return ['gmail.com', 'googlemail.com'].includes(domain) ? `${name.split('+')[0].replace(/\./g, '')}@gmail.com` : email;
}

// Trades the code from Google's redirect for Google's id of this person. The token arrives
// straight from Google over TLS, so its claims are checked but its signature need not be.
async function googleSubject(c, code) {
  const response = await fetch(c.env.GOOGLE_TOKEN_URL ?? 'https://oauth2.googleapis.com/token', {
    method: 'POST',
    body: new URLSearchParams({
      code,
      client_id: c.env.GOOGLE_CLIENT_ID,
      client_secret: c.env.GOOGLE_CLIENT_SECRET,
      redirect_uri: googleRedirectUri(c),
      grant_type: 'authorization_code',
    }),
  }).catch(() => null);
  const tokens = response?.ok ? await response.json().catch(() => null) : null;
  try {
    const payload = tokens.id_token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    const claims = JSON.parse(atob(payload));
    const fromGoogle = ['https://accounts.google.com', 'accounts.google.com'].includes(claims.iss);
    if (!fromGoogle || claims.aud !== c.env.GOOGLE_CLIENT_ID || typeof claims.sub !== 'string') return null;
    const verified = claims.email_verified === true || claims.email_verified === 'true';
    return { sub: claims.sub, email: verified && claims.email ? normalizeGoogleEmail(claims.email) : null };
  } catch {
    return null;
  }
}

function serializePost(row, viewer) {
  return {
    id: row.id,
    author: { id: row.author_id, name: row.author_name, avatar: row.author_avatar },
    body: row.body,
    youtubeId: row.youtube_id,
    audience: row.audience,
    createdAt: row.created_at,
    commentCount: row.comment_count,
    images: JSON.parse(row.image_ids),
    mine: viewer?.id === row.author_id,
  };
}

function serializeComment(row, viewer, post) {
  return {
    id: row.id,
    author: { id: row.author_id, name: row.author_name, avatar: row.author_avatar },
    body: row.body,
    youtubeId: row.body ? findYoutubeId(row.body) : null,
    image: row.image_id,
    parentId: row.parent_id,
    createdAt: row.created_at,
    removedBy: row.removed_by,
    canRemove: !row.removed_by && !!viewer && (viewer.id === row.author_id || viewer.id === post.author_id),
  };
}

// An uploaded image that its owner has not used yet, in a post, a comment or on their profile.
const UNUSED_IMAGE = `
  SELECT 1 FROM images WHERE id = ?1 AND owner_id = ?2 AND post_id IS NULL AND comment_id IS NULL
    AND NOT EXISTS (SELECT 1 FROM users WHERE avatar_id = ?1 OR cover_id = ?1)`;

const POST_SELECT = `
  SELECT p.*, u.name AS author_name, u.avatar_id AS author_avatar,
    (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
    (SELECT json_group_array(id) FROM (SELECT id FROM images WHERE post_id = p.id ORDER BY position)) AS image_ids
  FROM posts p JOIN users u ON u.id = p.author_id`;

async function areFriends(db, a, b) {
  const row = await db
    .prepare(`SELECT 1 FROM friendships WHERE status = 'accepted'
      AND ((requester_id = ?1 AND addressee_id = ?2) OR (requester_id = ?2 AND addressee_id = ?1))`)
    .bind(a, b)
    .first();
  return !!row;
}

async function loadVisiblePost(c, id) {
  const db = c.env.DB;
  const viewer = c.get('user');
  const post = await db.prepare(`${POST_SELECT} WHERE p.id = ? AND p.deleted_at IS NULL`).bind(id).first();
  if (!post) return null;
  if (post.audience === 'public' || viewer?.id === post.author_id) return post;
  return viewer && (await areFriends(db, viewer.id, post.author_id)) ? post : null;
}

const app = new Hono();

// The demo is public and has no moderation, so nothing a visitor sends is ever stored.
// This runs first so that refused writes do not fill the log either.
app.use('/api/*', async (c, next) => {
  if (isDemo(c) && !['GET', 'HEAD'].includes(c.req.method) && !DEMO_WRITES.includes(c.req.path)) {
    return c.json({ error: 'זו גרסת הדגמה לצפייה בלבד, אז שום דבר לא נשמר כאן' }, 403);
  }
  await next();
});

// Only the demo routes /why through the worker, to keep it closed there.
app.get('/why*', (c) => c.redirect('/'));

function requestLog(c, fields) {
  return {
    source: 'server',
    requestId: c.get('requestId'),
    userId: c.get('user')?.id,
    method: c.req.method,
    path: c.req.path,
    ...fields,
  };
}

app.use('/api/*', async (c, next) => {
  const requestId = crypto.randomUUID().slice(0, 8);
  c.set('requestId', requestId);
  const started = Date.now();
  await next();
  c.header('X-Request-Id', requestId);

  // 401 and 404 are normal traffic. Any other refusal may mean the app sent something wrong.
  const { status } = c.res;
  if (status >= 400 && status < 500 && status !== 401 && status !== 404) {
    const { error } = await c.res.clone().json().catch(() => ({}));
    await writeLog(c.env.DB, requestLog(c, { level: 'warn', status, message: error }));
  }
  console.log(JSON.stringify({ requestId, method: c.req.method, path: c.req.path, status, ms: Date.now() - started }));
});

app.onError(async (error, c) => {
  await writeLog(c.env.DB, requestLog(c, { level: 'error', status: 500, message: error.message, stack: error.stack }));
  return c.json({ error: 'משהו השתבש בשרת', requestId: c.get('requestId') }, 500);
});

app.post('/api/client-log', async (c) => {
  const { message, stack, path } = await readJson(c);
  if (typeof message !== 'string') return c.json({ error: 'חסרה הודעה' }, 400);
  await writeLog(c.env.DB, {
    source: 'client',
    level: 'error',
    requestId: c.get('requestId'),
    path: typeof path === 'string' ? path.slice(0, 200) : null,
    message,
    stack: typeof stack === 'string' ? stack : null,
  });
  return c.json({ ok: true });
});

app.use('/api/*', async (c, next) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token) {
    const user = await c.env.DB
      .prepare(`SELECT u.id, u.name, u.email, u.avatar_id, u.cover_id, u.google_sub FROM sessions s JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND s.expires_at > ?`)
      .bind(await sha256(token), Date.now())
      .first();
    if (user) c.set('user', user);
  }
  await next();
});

async function requireUser(c, next) {
  if (!c.get('user')) return c.json({ error: 'צריך להתחבר' }, 401);
  await next();
}

async function requireAdmin(c, next) {
  if (!c.get('user')) return c.json({ error: 'צריך להתחבר' }, 401);
  if (!isAdmin(c)) return c.json({ error: 'אין הרשאה' }, 403);
  await next();
}

app.post('/api/auth/request', async (c) => {
  const { email: rawEmail } = await readJson(c);
  const email = typeof rawEmail === 'string' ? rawEmail.trim().toLowerCase() : '';
  if (email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return c.json({ error: 'כתובת המייל לא תקינה' }, 400);
  }

  const now = Date.now();
  const recent = await c.env.DB
    .prepare('SELECT COUNT(*) AS n FROM login_tokens WHERE email = ? AND created_at > ?')
    .bind(email, now - 60 * 60 * 1000)
    .first();
  if (recent.n >= LOGIN_REQUESTS_PER_HOUR) return c.json({ error: 'יותר מדי ניסיונות, נסו שוב בעוד שעה' }, 429);

  // The dev shortcut works only on localhost, even if the variable leaks into production.
  const devLogin = c.env.DEV_LOGIN === '1' && isLocalhost(c);
  if (!devLogin) return c.json({ error: 'שליחת מיילים עוד לא הוגדרה' }, 503);

  const token = randomToken();
  await c.env.DB
    .prepare('INSERT INTO login_tokens (token_hash, email, created_at, expires_at) VALUES (?, ?, ?, ?)')
    .bind(await sha256(token), email, now, now + LOGIN_TOKEN_MS)
    .run();
  return c.json({ ok: true, devLink: `${new URL(c.req.url).origin}/login#token=${token}` });
});

// The link in the email opens a page that POSTs here, so mail scanners that prefetch links
// cannot use up the token.
app.post('/api/auth/verify', async (c) => {
  const { token } = await readJson(c);
  if (typeof token !== 'string') return c.json({ error: 'הקישור לא תקין' }, 400);

  const now = Date.now();
  const used = await c.env.DB
    .prepare(`UPDATE login_tokens SET used_at = ?1
      WHERE token_hash = ?2 AND used_at IS NULL AND expires_at > ?1 RETURNING email, invited_by`)
    .bind(now, await sha256(token))
    .first();
  if (!used) return c.json({ error: 'הקישור פג או שכבר השתמשו בו' }, 400);

  await c.env.DB
    .prepare('INSERT INTO users (email, created_at) VALUES (?, ?) ON CONFLICT (email) DO NOTHING')
    .bind(used.email, now)
    .run();
  const user = await c.env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(used.email).first();
  if (used.invited_by) {
    await c.env.DB
      .prepare(`INSERT INTO friendships (requester_id, addressee_id, status, created_at)
        VALUES (?, ?, 'accepted', ?) ON CONFLICT DO NOTHING`)
      .bind(used.invited_by, user.id, now)
      .run();
  }

  await createSession(c, user.id);
  return c.json({ ok: true });
});

// Google only ever signs in someone who already linked it from their page, so the network stays
// invitation-only. The state cookie proves the redirect back belongs to this browser.
app.get('/api/auth/google', (c) => {
  if (!c.env.GOOGLE_CLIENT_ID) return c.json({ error: 'לא נמצא' }, 404);
  const mode = c.req.query('link') === '1' ? 'link' : 'login';
  if (mode === 'link' && !c.get('user')) return c.redirect('/login');

  const state = randomToken();
  setCookie(c, GOOGLE_STATE_COOKIE, `${mode}.${state}`, {
    httpOnly: true,
    sameSite: 'Lax',
    secure: new URL(c.req.url).protocol === 'https:',
    path: GOOGLE_STATE_PATH,
    maxAge: 600,
  });
  const url = new URL('https://accounts.google.com/o/oauth2/v2/auth');
  url.search = new URLSearchParams({
    client_id: c.env.GOOGLE_CLIENT_ID,
    redirect_uri: googleRedirectUri(c),
    response_type: 'code',
    scope: 'openid email',
    prompt: 'select_account',
    state,
  });
  return c.redirect(url.href);
});

app.get('/api/auth/google/callback', async (c) => {
  if (!c.env.GOOGLE_CLIENT_ID) return c.json({ error: 'לא נמצא' }, 404);
  const [mode, state] = (getCookie(c, GOOGLE_STATE_COOKIE) ?? '').split('.');
  deleteCookie(c, GOOGLE_STATE_COOKIE, { path: GOOGLE_STATE_PATH });
  const finish = (result) => c.redirect(`/?google=${result}`);

  const code = c.req.query('code');
  if (!state || state !== c.req.query('state') || !code) return finish('failed');
  const google = await googleSubject(c, code);
  if (!google) return finish('failed');
  const subject = google.sub;

  const owner = await c.env.DB.prepare('SELECT id FROM users WHERE google_sub = ?').bind(subject).first();
  if (mode === 'link') {
    const user = c.get('user');
    if (!user) return finish('failed');
    if (owner && owner.id !== user.id) return finish('taken');
    await c.env.DB.prepare('UPDATE users SET google_sub = ? WHERE id = ?').bind(subject, user.id).run();
    return finish('linked');
  }
  if (owner) {
    await createSession(c, owner.id);
    return finish('login');
  }

  // Someone new signs in with the address they were invited by: their account starts linked.
  const invite = google.email && await c.env.DB
    .prepare('DELETE FROM google_invites WHERE email = ? RETURNING invited_by')
    .bind(google.email)
    .first();
  if (!invite) return finish('unknown');
  const now = Date.now();
  const user = await c.env.DB
    .prepare('INSERT INTO users (email, google_sub, created_at) VALUES (?, ?, ?) RETURNING id')
    .bind(`member-${randomToken().slice(0, 16).toLowerCase()}@invite.invalid`, subject, now)
    .first();
  await c.env.DB
    .prepare(`INSERT INTO friendships (requester_id, addressee_id, status, created_at) VALUES (?, ?, 'accepted', ?)`)
    .bind(invite.invited_by, user.id, now)
    .run();
  await createSession(c, user.id);
  return finish('joined');
});

// Every visitor shares one guest account. Its session row comes from tools/demo_seed.sql.
app.post('/api/auth/demo', (c) => {
  if (!isDemo(c)) return c.json({ error: 'לא נמצא' }, 404);
  startSession(c, DEMO_SESSION);
  return c.json({ ok: true });
});

app.post('/api/auth/logout', async (c) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token && !isDemo(c)) await c.env.DB.prepare('DELETE FROM sessions WHERE token_hash = ?').bind(await sha256(token)).run();
  deleteCookie(c, SESSION_COOKIE, { path: '/' });
  return c.json({ ok: true });
});

app.get('/api/me', async (c) => {
  const user = c.get('user');
  const waiting = user && await c.env.DB
    .prepare(`SELECT COUNT(*) AS n FROM friendships WHERE addressee_id = ? AND status = 'pending'`)
    .bind(user.id)
    .first();
  return c.json({
    user: user ? {
      id: user.id,
      name: user.name,
      avatar: user.avatar_id,
      cover: user.cover_id,
      admin: isAdmin(c),
      friendRequests: waiting.n,
      googleLinked: !!user.google_sub,
    } : null,
    demo: isDemo(c),
    inviteOnly: c.env.INVITE_ONLY === '1',
    googleLogin: !!c.env.GOOGLE_CLIENT_ID,
  });
});

app.put('/api/me', requireUser, async (c) => {
  const name = cleanText((await readJson(c)).name, MAX_NAME);
  if (!name) return c.json({ error: 'צריך שם, עד 60 תווים' }, 400);
  await c.env.DB.prepare('UPDATE users SET name = ? WHERE id = ?').bind(name, c.get('user').id).run();
  return c.json({ ok: true });
});

app.delete('/api/me/google', requireUser, async (c) => {
  await c.env.DB.prepare('UPDATE users SET google_sub = NULL WHERE id = ?').bind(c.get('user').id).run();
  return c.json({ ok: true });
});

// Sets or clears a profile picture. The picture it replaces is really deleted.
app.put('/api/me/images', requireUser, async (c) => {
  const { kind, imageId } = await readJson(c);
  const column = { avatar: 'avatar_id', cover: 'cover_id' }[kind];
  if (!column) return c.json({ error: 'סוג תמונה לא תקין' }, 400);

  const user = c.get('user');
  const usable = imageId === null
    || (typeof imageId === 'string' && await c.env.DB.prepare(UNUSED_IMAGE).bind(imageId, user.id).first());
  if (!usable) return c.json({ error: 'תמונה לא תקינה' }, 400);

  await c.env.DB.prepare(`UPDATE users SET ${column} = ? WHERE id = ?`).bind(imageId, user.id).run();
  const previous = user[column];
  if (previous) {
    await c.env.IMAGES.delete(previous);
    await c.env.DB.prepare('DELETE FROM images WHERE id = ?').bind(previous).run();
  }
  return c.json({ ok: true });
});

// How many people a friends post will reach. The composer shows this before publishing.
app.get('/api/audience', requireUser, async (c) => {
  const row = await c.env.DB
    .prepare(`SELECT COUNT(*) AS n FROM (${FRIEND_IDS})`)
    .bind(c.get('user').id)
    .first();
  return c.json({ friends: row.n });
});

// The whole feed algorithm: every post by you and your friends, newest first.
app.get('/api/feed', requireUser, async (c) => {
  const user = c.get('user');
  const before = parseId(c.req.query('before') ?? '') ?? Number.MAX_SAFE_INTEGER;
  const { results } = await c.env.DB
    .prepare(`${POST_SELECT}
      WHERE p.deleted_at IS NULL AND p.id < ?2
        AND (p.author_id = ?1 OR p.author_id IN (${FRIEND_IDS}))
      ORDER BY p.id DESC LIMIT ?3`)
    .bind(user.id, before, PAGE_SIZE)
    .all();
  return c.json({ posts: results.map((row) => serializePost(row, user)), hasMore: results.length === PAGE_SIZE });
});

app.post('/api/posts', requireUser, async (c) => {
  const input = await readJson(c);
  const user = c.get('user');
  const body = typeof input.body === 'string' ? input.body.trim() : '';
  const imageIds = Array.isArray(input.images) ? [...new Set(input.images)] : [];
  if (body.length > MAX_POST) return c.json({ error: 'הפוסט ארוך מדי' }, 400);
  if (!body && !imageIds.length) return c.json({ error: 'הפוסט ריק' }, 400);
  if (imageIds.length > MAX_IMAGES_PER_POST) return c.json({ error: 'עד 4 תמונות בפוסט' }, 400);
  if (!['friends', 'public'].includes(input.audience)) return c.json({ error: 'קהל לא תקין' }, 400);

  for (const imageId of imageIds) {
    const owned = typeof imageId === 'string' && await c.env.DB.prepare(UNUSED_IMAGE).bind(imageId, user.id).first();
    if (!owned) return c.json({ error: 'תמונה לא תקינה' }, 400);
  }

  const row = await c.env.DB
    .prepare(`INSERT INTO posts (author_id, body, youtube_id, audience, created_at)
      VALUES (?, ?, ?, ?, ?) RETURNING id`)
    .bind(user.id, body, findYoutubeId(body), input.audience, Date.now())
    .first();
  if (imageIds.length) {
    await c.env.DB.batch(imageIds.map((imageId, position) => c.env.DB
      .prepare('UPDATE images SET post_id = ?, position = ? WHERE id = ?')
      .bind(row.id, position, imageId)));
  }
  return c.json({ id: row.id }, 201);
});

// Every comment comes back, in the order it was written, with no paging and no ranking.
// The browser groups replies under the comment they answer.
app.get('/api/posts/:id', async (c) => {
  const id = parseId(c.req.param('id'));
  const post = id && (await loadVisiblePost(c, id));
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  const viewer = c.get('user');
  const { results } = await c.env.DB
    .prepare(`SELECT m.*, u.name AS author_name, u.avatar_id AS author_avatar,
      (SELECT id FROM images WHERE comment_id = m.id) AS image_id
      FROM comments m JOIN users u ON u.id = m.author_id
      WHERE m.post_id = ? ORDER BY m.id`)
    .bind(id)
    .all();
  return c.json({
    post: serializePost(post, viewer),
    comments: results.map((row) => serializeComment(row, viewer, post)),
  });
});

app.delete('/api/posts/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const result = await c.env.DB
    .prepare('UPDATE posts SET deleted_at = ? WHERE id = ? AND author_id = ? AND deleted_at IS NULL')
    .bind(Date.now(), id, c.get('user').id)
    .run();
  if (!result.meta.changes) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  // Deleting a post really deletes its images and those in its comments, not only the links.
  const attached = 'post_id = ?1 OR comment_id IN (SELECT id FROM comments WHERE post_id = ?1)';
  const { results } = await c.env.DB.prepare(`SELECT id FROM images WHERE ${attached}`).bind(id).all();
  if (results.length) {
    await c.env.IMAGES.delete(results.map((image) => image.id));
    await c.env.DB.prepare(`DELETE FROM images WHERE ${attached}`).bind(id).run();
  }
  return c.json({ ok: true });
});

app.post('/api/images', requireUser, async (c) => {
  const user = c.get('user');
  const now = Date.now();
  const today = await c.env.DB
    .prepare('SELECT COUNT(*) AS n FROM images WHERE owner_id = ? AND created_at > ?')
    .bind(user.id, now - 24 * 60 * 60 * 1000)
    .first();
  if (today.n >= IMAGES_PER_DAY) return c.json({ error: 'הגעת למכסת התמונות היומית' }, 429);

  const data = new Uint8Array(await c.req.arrayBuffer());
  if (data.length > MAX_IMAGE_BYTES) return c.json({ error: 'התמונה גדולה מדי' }, 413);
  const clean = stripJpegMetadata(data);
  if (!clean) return c.json({ error: 'רק תמונות JPEG' }, 415);

  const usage = await imageUsage(c);
  if (usage.bytes + clean.length > usage.storageLimit || usage.today >= usage.uploadLimit) {
    return c.json({ error: 'אי אפשר להעלות תמונות כרגע, הגענו למגבלת האחסון של הרשת' }, 429);
  }

  const id = randomToken();
  await c.env.IMAGES.put(id, clean, { httpMetadata: { contentType: 'image/jpeg' } });
  await c.env.DB
    .prepare('INSERT INTO images (id, owner_id, created_at, bytes) VALUES (?, ?, ?, ?)')
    .bind(id, user.id, now, clean.length)
    .run();
  return c.json({ id }, 201);
});

// Every request is checked against the post, so a copied image address does not bypass the audience.
app.get('/api/images/:id', async (c) => {
  const id = c.req.param('id');
  const image = await c.env.DB
    .prepare(`SELECT owner_id, comment_id,
      COALESCE(post_id, (SELECT post_id FROM comments WHERE id = images.comment_id)) AS post_id,
      EXISTS (SELECT 1 FROM users WHERE avatar_id = images.id OR cover_id = images.id) AS profile
      FROM images WHERE id = ?`)
    .bind(id)
    .first();
  const post = image?.post_id && (await loadVisiblePost(c, image.post_id));
  const visible = image?.profile || post || (image && !image.post_id && c.get('user')?.id === image.owner_id);
  const object = visible && (await c.env.IMAGES.get(id));
  if (!object) return c.json({ error: 'התמונה לא נמצאה' }, 404);
  return new Response(object.body, {
    headers: {
      'Content-Type': 'image/jpeg',
      'Cache-Control': 'private, no-cache',
      'X-Content-Type-Options': 'nosniff',
    },
  });
});

app.post('/api/posts/:id/comments', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const post = id && (await loadVisiblePost(c, id));
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  const input = await readJson(c);
  const user = c.get('user');
  const body = typeof input.body === 'string' ? input.body.trim() : '';
  const image = input.image ?? null;
  if (body.length > MAX_COMMENT || (!body && !image)) return c.json({ error: 'התגובה ריקה או ארוכה מדי' }, 400);
  if (image !== null) {
    const owned = typeof image === 'string' && await c.env.DB.prepare(UNUSED_IMAGE).bind(image, user.id).first();
    if (!owned) return c.json({ error: 'תמונה לא תקינה' }, 400);
  }

  let parentId = null;
  if (input.parentId !== undefined && input.parentId !== null) {
    const parent = await c.env.DB
      .prepare('SELECT id, parent_id FROM comments WHERE id = ? AND post_id = ? AND removed_by IS NULL')
      .bind(input.parentId, id)
      .first();
    if (!parent) return c.json({ error: 'התגובה לא נמצאה' }, 404);
    parentId = parent.parent_id ?? parent.id;
  }

  const row = await c.env.DB
    .prepare('INSERT INTO comments (post_id, author_id, body, created_at, parent_id) VALUES (?, ?, ?, ?, ?) RETURNING id')
    .bind(id, user.id, body, Date.now(), parentId)
    .first();
  if (image) await c.env.DB.prepare('UPDATE images SET comment_id = ? WHERE id = ?').bind(row.id, image).run();
  return c.json({ id: row.id }, 201);
});

app.delete('/api/comments/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const user = c.get('user');
  const comment = id && (await c.env.DB
    .prepare(`SELECT m.author_id, p.author_id AS post_owner_id FROM comments m
      JOIN posts p ON p.id = m.post_id WHERE m.id = ? AND m.removed_by IS NULL`)
    .bind(id)
    .first());
  if (!comment) return c.json({ error: 'התגובה לא נמצאה' }, 404);

  const removedBy = user.id === comment.author_id ? 'author' : user.id === comment.post_owner_id ? 'post_owner' : null;
  if (!removedBy) return c.json({ error: 'אין הרשאה' }, 403);
  await c.env.DB.prepare('UPDATE comments SET body = NULL, removed_by = ? WHERE id = ?').bind(removedBy, id).run();

  const image = await c.env.DB.prepare('SELECT id FROM images WHERE comment_id = ?').bind(id).first();
  if (image) {
    await c.env.IMAGES.delete(image.id);
    await c.env.DB.prepare('DELETE FROM images WHERE id = ?').bind(image.id).run();
  }
  return c.json({ ok: true });
});

// The server fetches a video's title and picture, so that Google never learns who is reading.
// Nothing about the reader is passed on, and only the player itself needs a click.
app.get('/api/youtube/:id', async (c) => {
  const id = c.req.param('id');
  if (!YOUTUBE_ID.test(id)) return c.json({ error: 'לא נמצא' }, 404);
  const watch = encodeURIComponent(`https://www.youtube.com/watch?v=${id}`);
  const origin = c.env.YOUTUBE_ORIGIN ?? 'https://www.youtube.com';
  const response = await fetch(`${origin}/oembed?format=json&url=${watch}`).catch(() => null);
  const details = response?.ok ? await response.json().catch(() => null) : null;
  c.header('Cache-Control', `public, max-age=${VIDEO_CACHE_SECONDS}`);
  return c.json({ title: typeof details?.title === 'string' ? details.title : null });
});

app.get('/api/youtube/:id/thumb', async (c) => {
  const id = c.req.param('id');
  if (!YOUTUBE_ID.test(id)) return c.json({ error: 'לא נמצא' }, 404);
  const origin = c.env.YOUTUBE_ORIGIN ?? 'https://i.ytimg.com';
  const response = await fetch(`${origin}/vi/${id}/hqdefault.jpg`).catch(() => null);
  if (!response?.ok) return c.json({ error: 'לא נמצא' }, 404);
  return new Response(response.body, {
    headers: {
      'Content-Type': 'image/jpeg',
      'Cache-Control': `public, max-age=${VIDEO_CACHE_SECONDS}`,
      'X-Content-Type-Options': 'nosniff',
    },
  });
});

app.get('/api/users', requireUser, async (c) => {
  const q = (c.req.query('q') ?? '').trim();
  if (q.length < 2) return c.json({ users: [] });
  const pattern = `%${q.replace(/[\\%_]/g, '\\$&')}%`;
  const { results } = await c.env.DB
    .prepare(`SELECT id, name, avatar_id AS avatar FROM users WHERE name LIKE ? ESCAPE '\\' AND name <> '' AND id <> ? ORDER BY name LIMIT 20`)
    .bind(pattern, c.get('user').id)
    .all();
  return c.json({ users: results });
});

// Someone's page: their pictures, and the posts this viewer is allowed to see.
app.get('/api/users/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const viewer = c.get('user');
  const person = id && (await c.env.DB
    .prepare(`SELECT id, name, avatar_id AS avatar, cover_id AS cover FROM users WHERE id = ? AND name <> ''`)
    .bind(id)
    .first());
  if (!person) return c.json({ error: 'המשתמש לא נמצא' }, 404);

  const link = await c.env.DB
    .prepare(`SELECT requester_id, status FROM friendships
      WHERE (requester_id = ?1 AND addressee_id = ?2) OR (requester_id = ?2 AND addressee_id = ?1)`)
    .bind(viewer.id, id)
    .first();
  const friendship = id === viewer.id ? 'self'
    : link?.status === 'accepted' ? 'friends'
      : link ? (link.requester_id === viewer.id ? 'outgoing' : 'incoming') : null;

  const before = parseId(c.req.query('before') ?? '') ?? Number.MAX_SAFE_INTEGER;
  const seesAll = friendship === 'self' || friendship === 'friends';
  const { results } = await c.env.DB
    .prepare(`${POST_SELECT}
      WHERE p.deleted_at IS NULL AND p.author_id = ?1 AND p.id < ?2 AND (?3 OR p.audience = 'public')
      ORDER BY p.id DESC LIMIT ?4`)
    .bind(id, before, seesAll ? 1 : 0, PAGE_SIZE)
    .all();
  return c.json({
    user: person,
    friendship,
    posts: results.map((row) => serializePost(row, viewer)),
    hasMore: results.length === PAGE_SIZE,
  });
});

app.get('/api/friends', requireUser, async (c) => {
  const { results } = await c.env.DB
    .prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, f.status, f.requester_id = ?1 AS outgoing
      FROM friendships f
      JOIN users u ON u.id = CASE WHEN f.requester_id = ?1 THEN f.addressee_id ELSE f.requester_id END
      WHERE (f.requester_id = ?1 OR f.addressee_id = ?1) AND u.name <> ''
      ORDER BY u.name`)
    .bind(c.get('user').id)
    .all();
  const person = ({ id, name, avatar }) => ({ id, name, avatar });
  return c.json({
    friends: results.filter((r) => r.status === 'accepted').map(person),
    incoming: results.filter((r) => r.status === 'pending' && !r.outgoing).map(person),
    outgoing: results.filter((r) => r.status === 'pending' && r.outgoing).map(person),
  });
});

// Asking someone who already asked you accepts their request.
app.post('/api/friends/:id', requireUser, async (c) => {
  const other = parseId(c.req.param('id'));
  const me = c.get('user').id;
  const target = other && other !== me
    && (await c.env.DB.prepare(`SELECT id FROM users WHERE id = ? AND name <> ''`).bind(other).first());
  if (!target) return c.json({ error: 'המשתמש לא נמצא' }, 404);

  const accepted = await c.env.DB
    .prepare(`UPDATE friendships SET status = 'accepted' WHERE requester_id = ? AND addressee_id = ?`)
    .bind(other, me)
    .run();
  if (accepted.meta.changes) return c.json({ status: 'accepted' });

  await c.env.DB
    .prepare(`INSERT INTO friendships (requester_id, addressee_id, status, created_at)
      VALUES (?, ?, 'pending', ?) ON CONFLICT DO NOTHING`)
    .bind(me, other, Date.now())
    .run();
  return c.json({ status: 'pending' });
});

// Cancels a request, declines one, or ends a friendship.
app.delete('/api/friends/:id', requireUser, async (c) => {
  await c.env.DB
    .prepare(`DELETE FROM friendships
      WHERE (requester_id = ?1 AND addressee_id = ?2) OR (requester_id = ?2 AND addressee_id = ?1)`)
    .bind(c.get('user').id, parseId(c.req.param('id')))
    .run();
  return c.json({ ok: true });
});

app.get('/api/admin/overview', requireAdmin, async (c) => {
  const db = c.env.DB;
  const [counts, members] = await db.batch([
    db.prepare(`SELECT (SELECT COUNT(*) FROM users) AS users,
      (SELECT COUNT(*) FROM posts WHERE deleted_at IS NULL) AS posts,
      (SELECT COUNT(*) FROM comments WHERE removed_by IS NULL) AS comments`),
    db.prepare('SELECT id, name, avatar_id AS avatar, created_at AS joinedAt FROM users ORDER BY id'),
  ]);
  const { results: googleInvites } = await db
    .prepare('SELECT email, created_at AS createdAt FROM google_invites ORDER BY created_at DESC')
    .all();
  return c.json({ usage: await imageUsage(c), ...counts.results[0], members: members.results, googleInvites });
});

app.post('/api/admin/google-invites', requireAdmin, async (c) => {
  const raw = (await readJson(c)).email;
  const email = typeof raw === 'string' && raw.length <= 254 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(raw.trim())
    ? normalizeGoogleEmail(raw) : null;
  if (!email) return c.json({ error: 'כתובת לא תקינה' }, 400);
  await c.env.DB
    .prepare('INSERT INTO google_invites (email, invited_by, created_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING')
    .bind(email, c.get('user').id, Date.now())
    .run();
  return c.json({ email }, 201);
});

app.delete('/api/admin/google-invites/:email', requireAdmin, async (c) => {
  await c.env.DB.prepare('DELETE FROM google_invites WHERE email = ?').bind(c.req.param('email')).run();
  return c.json({ ok: true });
});

// Without a mail service, the admin hands out login links: a new member, or a new link for
// someone who lost their session. The link works once. Members get an address nobody can mail.
app.post('/api/admin/invites', requireAdmin, async (c) => {
  const { userId } = await readJson(c);
  let email = `member-${randomToken().slice(0, 16).toLowerCase()}@invite.invalid`;
  let invitedBy = c.get('user').id;
  if (userId !== undefined) {
    const user = await c.env.DB.prepare('SELECT email FROM users WHERE id = ?').bind(userId).first();
    if (!user) return c.json({ error: 'המשתמש לא נמצא' }, 404);
    email = user.email;
    invitedBy = null;
  }
  const token = randomToken();
  const now = Date.now();
  await c.env.DB
    .prepare('INSERT INTO login_tokens (token_hash, email, created_at, expires_at, invited_by) VALUES (?, ?, ?, ?, ?)')
    .bind(await sha256(token), email, now, now + INVITE_MS, invitedBy)
    .run();
  return c.json({ link: `${new URL(c.req.url).origin}/login#token=${token}` }, 201);
});

app.notFound((c) => c.json({ error: 'לא נמצא' }, 404));

export default app;
