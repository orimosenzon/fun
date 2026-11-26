import { Hono } from 'hono';
import { deleteCookie, getCookie, setCookie } from 'hono/cookie';
import { outsideNews, israelNews, PICTURE_HOSTS } from './digest.js';
import { deviceName, writeJournal, writeLog } from './log.js';
import { pushDevice, pushReady, sendPush } from './push.js';
import { pagePicture, pageUrl, readPage } from './page.js';
import { CONTACT_SCOPES, DEFAULT_NOTE, MAX_NOTE, SEND_SCOPE, googleContacts, inviteLetter, seal, sendFromGmail, unseal } from './mail.js';

const SESSION_COOKIE = 'session';
const SESSION_MS = 90 * 24 * 60 * 60 * 1000;
const LOGIN_TOKEN_MS = 15 * 60 * 1000;
const INVITE_MS = 7 * 24 * 60 * 60 * 1000;
// A link sent by email waits in a private inbox, so it may wait longer.
const EMAIL_INVITE_MS = 30 * 24 * 60 * 60 * 1000;
const NO_MAILBOX = '@invite.invalid';
const LOGIN_REQUESTS_PER_HOUR = 5;
const PAGE_SIZE = 30;
const MAX_POST = 5000;
const MAX_COMMENT = 2000;
const MAX_MESSAGE = 4000;
const MESSAGES_PAGE = 50;
// A removal is reported again for a few seconds, so one that landed between two questions is not missed.
const REMOVAL_OVERLAP_MS = 10000;
const MAX_NAME = 60;
const MAX_IMAGE_BYTES = 1024 * 1024;
const MAX_IMAGES_PER_POST = 4;
const IMAGES_PER_DAY = 40;
// R2 has no spending cap of its own and bills past its free tier (10 GB, 1M writes a month).
// These limits keep the whole network inside it, whatever anyone does.
const STORAGE_LIMIT_BYTES = 8 * 1024 ** 3;
const UPLOADS_PER_DAY = 2000;
// A place is kept to two decimals of a degree, about a kilometre: a neighbourhood, not a house.
const PLACE_DECIMALS = 2;
// "What's up" is one line, and it is erased a week after it was written.
const MAX_STATUS = 140;
const STATUS_MS = 7 * 24 * 60 * 60 * 1000;
const DAY_MS = 24 * 60 * 60 * 1000;
const YOUTUBE_ID = /^[\w-]{11}$/;
const VIDEO_CACHE_SECONDS = 24 * 60 * 60;
const GOOGLE_STATE_COOKIE = 'google_state';
const GOOGLE_STATE_PATH = '/api/auth/google';

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

function place(row) {
  return row.lat === null || row.lat === undefined ? null : { lat: row.lat, lng: row.lng };
}

function status(row, now = Date.now()) {
  return row.status_text && row.status_at > now - STATUS_MS ? { text: row.status_text, at: row.status_at } : null;
}

function isLocalhost(c) {
  return ['localhost', '127.0.0.1'].includes(new URL(c.req.url).hostname);
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

// Enough of a session's hash to tell in the journal that two lines came from one device.
const sessionMark = (hash) => hash?.slice(0, 8) ?? null;
const SEEN_EVERY_MS = 60 * 60 * 1000;

// One line in the journal about this request. Whoever it is about is the logged in member,
// unless the line says otherwise (a login is about the one who just got in).
async function journal(c, kind, detail, about = {}) {
  await writeJournal(c.env.DB, {
    kind,
    detail,
    userId: about.userId ?? c.get('user')?.id,
    session: about.session ?? c.get('session'),
    device: deviceName(c.req.header('User-Agent')),
    requestId: c.get('requestId'),
  });
}

// Opens a session for the member on this device. `how` is the way they got in, for the journal.
// The session this browser held until now, of whichever account, ends: it has no other holder.
async function createSession(c, userId, how) {
  const session = randomToken();
  const hash = await sha256(session);
  const now = Date.now();
  const old = getCookie(c, SESSION_COOKIE);
  const oldHash = old ? await sha256(old) : null;
  const before = oldHash && (await c.env.DB.prepare('SELECT user_id FROM sessions WHERE token_hash = ?').bind(oldHash).first());
  await c.env.DB.batch([
    c.env.DB.prepare('INSERT INTO sessions (token_hash, user_id, expires_at, created_at, seen_at, device) VALUES (?, ?, ?, ?, ?, ?)')
      .bind(hash, userId, now + SESSION_MS, now, now, deviceName(c.req.header('User-Agent'))),
    c.env.DB.prepare('DELETE FROM sessions WHERE token_hash = ?').bind(oldHash),
    c.env.DB.prepare('DELETE FROM push_devices WHERE session_hash = ?').bind(oldHash),
  ]);
  startSession(c, session);
  const replaced = before ? (before.user_id === userId ? ', replacing their older session here' : `, replacing the session of member ${before.user_id} here`) : '';
  await journal(c, 'login', `${how}${replaced}`, { userId, session: sessionMark(hash) });
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

// Trades the code from Google's redirect for Google's id of this person, with the verified
// address, name and picture of the account. The token arrives straight from Google over TLS,
// so its claims are checked but its signature need not be.
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
    const claims = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(payload), (ch) => ch.charCodeAt(0))));
    const fromGoogle = ['https://accounts.google.com', 'accounts.google.com'].includes(claims.iss);
    if (!fromGoogle || claims.aud !== c.env.GOOGLE_CLIENT_ID || typeof claims.sub !== 'string') return null;
    const verified = claims.email_verified === true || claims.email_verified === 'true';
    return {
      sub: claims.sub,
      email: verified && claims.email ? normalizeGoogleEmail(claims.email) : null,
      address: verified && claims.email ? String(claims.email) : null,
      refreshToken: typeof tokens.refresh_token === 'string' ? tokens.refresh_token : null,
      scopes: typeof tokens.scope === 'string' ? tokens.scope.split(' ') : [],
      name: cleanText(claims.name, MAX_NAME),
      picture: typeof claims.picture === 'string' ? claims.picture : null,
    };
  } catch {
    return null;
  }
}

// A member's first picture is the one on their Google account, if they have one and have not
// chosen a picture here. Google's server is asked by ours, once. Anything unexpected is skipped
// quietly, and the member keeps the picture drawn from their name.
async function importGooglePicture(c, userId, picture) {
  try {
    const origin = c.env.GOOGLE_PICTURE_ORIGIN ?? 'https://lh3.googleusercontent.com';
    const url = new URL(picture);
    if (url.origin !== origin) return;
    const user = await c.env.DB.prepare('SELECT avatar_id FROM users WHERE id = ?').bind(userId).first();
    if (!user || user.avatar_id) return;
    // Google ends the address with the size it serves, like "=s96-c".
    url.pathname = url.pathname.replace(/=s\d+(-c)?$/, '=s400-c');
    const response = await fetch(url);
    if (!response.ok) return;
    const data = new Uint8Array(await response.arrayBuffer());
    // The letter Google draws for an account without a photo is not a JPEG, so it stops here too.
    const clean = data.length <= MAX_IMAGE_BYTES ? stripJpegMetadata(data) : null;
    if (!clean) return;
    const usage = await imageUsage(c);
    if (usage.bytes + clean.length > usage.storageLimit || usage.today >= usage.uploadLimit) return;
    const id = randomToken();
    await c.env.IMAGES.put(id, clean, { httpMetadata: { contentType: 'image/jpeg' } });
    await c.env.DB.batch([
      c.env.DB.prepare('INSERT INTO images (id, owner_id, created_at, bytes) VALUES (?, ?, ?, ?)').bind(id, userId, Date.now(), clean.length),
      c.env.DB.prepare('UPDATE users SET avatar_id = ? WHERE id = ? AND avatar_id IS NULL').bind(id, userId),
    ]);
  } catch {
    // No picture from Google this time.
  }
}

function serializePost(row, viewer, reactions) {
  return {
    id: row.id,
    author: { id: row.author_id, name: row.author_name, avatar: row.author_avatar },
    body: row.body,
    youtubeId: row.youtube_id,
    page: row.page_url ? {
      url: row.page_url,
      title: row.page_title,
      text: row.page_text,
      picture: row.page_picture ? `/api/posts/${row.id}/page-picture` : null,
      framable: row.page_framable !== 0,
    } : null,
    audience: row.audience,
    createdAt: row.created_at,
    editedAt: row.edited_at ?? null,
    commentCount: row.comment_count,
    images: JSON.parse(row.image_ids),
    mine: viewer?.id === row.author_id,
    reactions: reactions?.get(row.id) ?? [],
    shares: null,
    shared: null,
  };
}

function serializeComment(row, viewer, post, reactions) {
  return {
    id: row.id,
    author: { id: row.author_id, name: row.author_name, avatar: row.author_avatar },
    body: row.body,
    youtubeId: row.body ? findYoutubeId(row.body) : null,
    image: row.image_id,
    parentId: row.parent_id,
    createdAt: row.created_at,
    editedAt: row.removed_by ? null : row.edited_at ?? null,
    removedBy: row.removed_by,
    canEdit: !row.removed_by && viewer?.id === row.author_id,
    canRemove: !row.removed_by && !!viewer && (viewer.id === row.author_id || viewer.id === post.author_id),
    reactions: reactions?.get(row.id) ?? [],
  };
}

// An uploaded image that its owner has not used yet, in a post, a comment, a message or on their profile.
const UNUSED_IMAGE = `
  SELECT 1 FROM images WHERE id = ?1 AND owner_id = ?2 AND post_id IS NULL AND comment_id IS NULL AND message_id IS NULL
    AND NOT EXISTS (SELECT 1 FROM users WHERE avatar_id = ?1 OR cover_id = ?1)`;

// The five feelings that the network draws itself. Anything else is a single standard emoji.
const FEELINGS = ['love', 'funny', 'surprise', 'bored', 'anger'];
// The emoji that say the same as a drawn feeling count as it, so they are shown and counted together.
const SAME_FEELING = { '❤️': 'love', '😂': 'funny', '🤣': 'funny', '😮': 'surprise', '😲': 'surprise', '😑': 'bored', '🥱': 'bored', '😠': 'anger', '😡': 'anger' };
// Built from a string because the bundler does not read the "v" flag yet.
const ONE_EMOJI = new RegExp('^\\p{RGI_Emoji}$', 'v');
const MAX_REACTION_NAMES = 60;
const MAX_SHARE_NAMES = 200;

function feelingKind(value) {
  if (typeof value !== 'string' || value.length > 40) return null;
  if (FEELINGS.includes(value)) return value;
  return SAME_FEELING[value] ?? (ONE_EMOJI.test(value) ? value : null);
}

// What was felt about each of these posts (or comments), grouped by feeling, the most common
// first. Whoever is not logged in sees the counts without the names.
async function reactionsOf(db, column, ids, viewer) {
  const byTarget = new Map(ids.map((id) => [id, []]));
  if (!ids.length) return byTarget;
  const { results } = await db
    .prepare(`SELECT r.${column} AS target, r.kind, r.user_id, u.name FROM reactions r JOIN users u ON u.id = r.user_id
      WHERE r.${column} IN (SELECT value FROM json_each(?)) ORDER BY r.created_at, r.rowid`)
    .bind(JSON.stringify(ids))
    .all();
  for (const row of results) {
    const groups = byTarget.get(row.target);
    let group = groups.find((item) => item.kind === row.kind);
    if (!group) groups.push(group = { kind: row.kind, count: 0, mine: false, people: [] });
    group.count++;
    if (viewer?.id === row.user_id) group.mine = true;
    if (viewer && group.people.length < MAX_REACTION_NAMES) group.people.push({ id: row.user_id, name: row.name });
  }
  for (const groups of byTarget.values()) groups.sort((a, b) => b.count - a.count);
  return byTarget;
}

const postReactions = (db, rows, viewer) => reactionsOf(db, 'post_id', rows.map((row) => row.id), viewer);

// Posts as the page shows them: with what was felt about each, how many times it was shared,
// and for a share, the post it quotes. The quote comes only to a viewer who may read the
// original post, so sharing never widens the audience that its writer chose.
async function showPosts(db, rows, viewer) {
  const viewerId = viewer?.id ?? 0;
  const list = (ids) => JSON.stringify([...new Set(ids)]);
  const quotedIds = rows.map((row) => row.shared_id).filter(Boolean);
  const quoted = !quotedIds.length ? [] : (await db
    .prepare(`${POST_SELECT} WHERE p.deleted_at IS NULL AND p.id IN (SELECT value FROM json_each(?2)) AND ${VISIBLE_TO}`)
    .bind(viewerId, list(quotedIds))
    .all()).results;
  const originals = [...rows.filter((row) => !row.shared_id), ...quoted];
  const [reactions, counted] = await Promise.all([
    postReactions(db, rows, viewer),
    !originals.length ? { results: [] } : db
      .prepare(`SELECT shared_id, COUNT(*) AS n, MAX(CASE WHEN author_id = ?1 THEN id END) AS mine FROM posts
        WHERE shared_id IN (SELECT value FROM json_each(?2)) AND deleted_at IS NULL GROUP BY shared_id`)
      .bind(viewerId, list(originals.map((row) => row.id)))
      .all(),
  ]);
  const counts = new Map(counted.results.map((row) => [row.shared_id, row]));
  const original = (row, feelings) => ({
    ...serializePost(row, viewer, feelings),
    shares: { count: counts.get(row.id)?.n ?? 0, mine: counts.get(row.id)?.mine ?? null },
  });
  return rows.map((row) => {
    if (!row.shared_id) return original(row, reactions);
    const source = quoted.find((other) => other.id === row.shared_id);
    return { ...serializePost(row, viewer, reactions), shared: { post: source ? original(source) : null, gone: !!row.shared_gone } };
  });
}

// Sets, replaces or takes back the feeling of the member about one post or one comment. Whoever
// wrote it (owner) is told, once, if they may still read the post it is on.
async function react(c, column, id, post, owner) {
  const db = c.env.DB;
  const user = c.get('user');
  const clear = db.prepare(`DELETE FROM reactions WHERE ${column} = ? AND user_id = ?`).bind(id, user.id);
  if (c.req.method === 'DELETE') await clear.run();
  else {
    const kind = feelingKind((await readJson(c)).kind);
    if (!kind) return c.json({ error: 'אפשר לבחור רגש אחד, או אימוג׳י אחד' }, 400);
    await db.batch([clear, db.prepare(`INSERT INTO reactions (${column}, user_id, kind, created_at) VALUES (?, ?, ?, ?)`).bind(id, user.id, kind, Date.now())]);
    const onPost = column === 'post_id';
    if ((await readersOf(db, post, [owner])).length) {
      await notify(c, user, [{ user: owner, kind: 'felt', post: post.id, comment: onPost ? null : id, feeling: kind, onShare: onPost && !!post.shared_id }]);
    }
  }
  return c.json({ reactions: (await reactionsOf(db, column, [id], user)).get(id) });
}

const POST_SELECT = `
  SELECT p.*, u.name AS author_name, u.avatar_id AS author_avatar,
    (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
    (SELECT deleted_at IS NOT NULL FROM posts WHERE id = p.shared_id) AS shared_gone,
    (SELECT json_group_array(id) FROM (SELECT id FROM images WHERE post_id = p.id ORDER BY position)) AS image_ids
  FROM posts p JOIN users u ON u.id = p.author_id`;

// Whether the viewer (?1) may read the post p: it is public, their own, or written by a friend.
const VISIBLE_TO = `(p.audience = 'public' OR p.author_id = ?1 OR p.author_id IN (${FRIEND_IDS}))`;

async function areFriends(db, a, b) {
  const row = await db
    .prepare(`SELECT 1 FROM friendships WHERE status = 'accepted'
      AND ((requester_id = ?1 AND addressee_id = ?2) OR (requester_id = ?2 AND addressee_id = ?1))`)
    .bind(a, b)
    .first();
  return !!row;
}

// The two people of a conversation are kept smaller id first, so each pair has one row.
function conversationOf(db, a, b) {
  return db.prepare('SELECT * FROM conversations WHERE low_id = ? AND high_id = ?').bind(Math.min(a, b), Math.max(a, b)).first();
}

const MESSAGE_SELECT = 'SELECT m.*, (SELECT id FROM images WHERE message_id = m.id) AS image_id FROM messages m';

// Messages from the other side that this member has not seen yet, in all their conversations.
const UNREAD_MESSAGES = `
  SELECT COUNT(*) AS n FROM conversations k JOIN messages m ON m.conversation_id = k.id
    AND m.id > CASE WHEN k.low_id = ?1 THEN k.low_read ELSE k.high_read END
  WHERE (k.low_id = ?1 OR k.high_id = ?1) AND m.sender_id <> ?1 AND m.removed_at IS NULL`;

function serializeMessage(row, viewerId) {
  return {
    id: row.id,
    mine: row.sender_id === viewerId,
    body: row.body,
    image: row.image_id ?? null,
    at: row.created_at,
    removed: !!row.removed_at,
  };
}

// Whoever joins by invitation starts as a friend of the person who invited them.
async function befriend(db, inviter, joined, now) {
  await db
    .prepare(`INSERT INTO friendships (requester_id, addressee_id, status, created_at)
      VALUES (?, ?, 'accepted', ?) ON CONFLICT DO NOTHING`)
    .bind(inviter, joined, now)
    .run();
}

async function loadVisiblePost(c, id) {
  const db = c.env.DB;
  const viewer = c.get('user');
  const post = await db.prepare(`${POST_SELECT} WHERE p.id = ? AND p.deleted_at IS NULL`).bind(id).first();
  if (!post) return null;
  if (post.audience === 'public' || viewer?.id === post.author_id) return post;
  return viewer && (await areFriends(db, viewer.id, post.author_id)) ? post : null;
}

// ---------- Notices ----------

// The kinds of notices, in the groups that the page filters by and that a member can silence.
const NOTICE_GROUPS = {
  comments: ['comment', 'reply', 'thread'],
  feelings: ['felt'],
  shares: ['share', 'reshare'],
  friends: ['request', 'accepted', 'joined'],
};
const PUSH_GROUPS = ['messages', ...Object.keys(NOTICE_GROUPS)];
const TOLD_ONCE = ['felt', 'share', 'reshare', 'request', 'accepted', 'joined'];
const NOTICES_PAGE = 30;
const MAX_PUSH_DEVICES = 10;
const FEELING_EMOJI = { love: '❤️', funny: '😂', surprise: '😮', bored: '😑', anger: '😠' };
const noticeGroup = (kind) => Object.keys(NOTICE_GROUPS).find((group) => NOTICE_GROUPS[group].includes(kind));

// What a notice says after the name of whoever did it. The words name the thing that happened
// and not the person, so they read the same about a woman and about a man.
function noticeText(kind, { onShare = false, onComment = false } = {}) {
  switch (kind) {
    case 'comment': return onShare ? 'תגובה לשיתוף שלך' : 'תגובה לפוסט שלך';
    case 'reply': return 'תשובה לתגובה שלך';
    case 'thread': return 'תגובה נוספת בשרשור שכתבת בו';
    case 'felt': return onComment ? 'על התגובה שלך' : onShare ? 'על השיתוף שלך' : 'על הפוסט שלך';
    case 'share': return 'שיתוף של הפוסט שלך';
    case 'reshare': return 'שיתוף נוסף, דרך השיתוף שלך';
    case 'request': return 'בקשת חברות';
    case 'accepted': return 'בקשת החברות שלך אושרה';
    default: return 'הצטרפות לרשת, בהזמנתך';
  }
}

// Where a notice leads: to the comment itself, to the share if the member may open it, to the
// requests, or to the person.
function noticePath(kind, { post, comment, share, shareVisible, actor }) {
  if (kind === 'request') return '/friends';
  if (kind === 'accepted' || kind === 'joined') return `/u/${actor}`;
  if (kind === 'share' || kind === 'reshare') return `/p/${shareVisible ? share : post}`;
  return `/p/${post}${comment ? `?c=${comment}` : ''}`;
}

// The notices of the member (?1) that still stand. A notice falls away when its post or comment
// was deleted, when the member may no longer read the post, when the feeling was taken back or
// the request is no longer waiting. So the page and the count never tell of what is gone.
const NOTICE_FROM = `
  FROM notices n JOIN users u ON u.id = n.actor_id
  LEFT JOIN posts p ON p.id = n.post_id
  LEFT JOIN comments m ON m.id = n.comment_id
  LEFT JOIN posts s ON s.id = n.share_id
  WHERE n.user_id = ?1 AND u.name <> ''
    AND (n.post_id IS NULL OR (p.deleted_at IS NULL AND ${VISIBLE_TO}))
    AND (n.comment_id IS NULL OR m.removed_by IS NULL)
    AND (n.share_id IS NULL OR s.deleted_at IS NULL)
    AND CASE n.kind
      WHEN 'felt' THEN CASE WHEN n.comment_id IS NULL
        THEN EXISTS (SELECT 1 FROM reactions r WHERE r.post_id = n.post_id AND r.user_id = n.actor_id)
        ELSE EXISTS (SELECT 1 FROM reactions r WHERE r.comment_id = n.comment_id AND r.user_id = n.actor_id) END
      WHEN 'request' THEN EXISTS (SELECT 1 FROM friendships f
        WHERE f.requester_id = n.actor_id AND f.addressee_id = ?1 AND f.status = 'pending')
      WHEN 'accepted' THEN n.actor_id IN (${FRIEND_IDS})
      ELSE 1 END`;

// The notices that came since the member last opened their notices page, on any device.
const UNSEEN_NOTICES = `SELECT COUNT(*) AS unseen ${NOTICE_FROM} AND n.at > (SELECT notices_seen_at FROM users WHERE id = ?1)`;

const NOTICE_SELECT = `
  SELECT n.id, n.kind, n.at, n.post_id, n.comment_id, n.share_id,
    u.id AS actor_id, u.name AS actor_name, u.avatar_id AS actor_avatar,
    SUBSTR(p.body, 1, 160) AS post_body, p.shared_id IS NOT NULL AND p.author_id = ?1 AS on_share,
    SUBSTR(m.body, 1, 240) AS comment_body, EXISTS (SELECT 1 FROM images WHERE comment_id = m.id) AS comment_image,
    (SELECT SUBSTR(top.body, 1, 160) FROM comments top WHERE top.id = m.parent_id AND top.removed_by IS NULL) AS thread_body,
    CASE WHEN n.kind = 'felt' THEN (SELECT r.kind FROM reactions r WHERE r.user_id = n.actor_id
      AND (r.comment_id = n.comment_id OR (n.comment_id IS NULL AND r.post_id = n.post_id))) END AS feeling,
    s.audience = 'public' OR s.author_id IN (${FRIEND_IDS}) AS share_visible, SUBSTR(s.body, 1, 240) AS share_body
  ${NOTICE_FROM}`;

// A notice as the page shows it: who, what happened, what they wrote (words), and which of the
// member's own words it was about.
function serializeNotice(row, seenAt) {
  const wrote = NOTICE_GROUPS.comments.includes(row.kind);
  const onComment = row.kind === 'felt' && row.comment_id !== null;
  const words = wrote ? row.comment_body : row.share_id && row.share_visible ? row.share_body : null;
  const about = row.kind === 'reply' || row.kind === 'thread' ? row.thread_body : onComment ? row.comment_body : row.post_body;
  return {
    id: row.id,
    kind: row.kind,
    at: row.at,
    fresh: row.at > seenAt,
    actor: { id: row.actor_id, name: row.actor_name, avatar: row.actor_avatar },
    text: noticeText(row.kind, { onShare: !!row.on_share, onComment }),
    feeling: row.feeling ?? null,
    words: words || null,
    image: wrote && !!row.comment_image,
    about: about || null,
    path: noticePath(row.kind, { post: row.post_id, comment: row.comment_id, share: row.share_id, shareVisible: !!row.share_visible, actor: row.actor_id }),
  };
}

// Work that goes on after the answer was sent, so nobody waits for a push service.
function later(c, work) {
  const done = work.catch((error) => writeLog(c.env.DB, { source: 'server', level: 'error', message: `push: ${error.message}`, stack: error.stack }));
  try {
    c.executionCtx.waitUntil(done);
  } catch {
    // Nothing keeps the request open here, and the work goes on by itself.
  }
}

// Pushes a message to every device where the member asked for it and is still logged in,
// unless they silenced this kind. A device that has unsubscribed is forgotten. Returns how many
// devices took it.
async function pushTo(c, userId, group, message) {
  if (!pushReady(c.env)) return 0;
  const db = c.env.DB;
  const { results: devices } = await db
    .prepare(`SELECT d.endpoint, d.p256dh, d.auth FROM push_devices d
      JOIN sessions s ON s.token_hash = d.session_hash AND s.user_id = d.user_id AND s.expires_at > ?2
      JOIN users u ON u.id = d.user_id
      WHERE d.user_id = ?1 AND INSTR(',' || u.push_muted || ',', ',' || ?3 || ',') = 0`)
    .bind(userId, Date.now(), group)
    .all();
  if (!devices.length) return 0;
  // The number that the icon of the app carries: everything that waits for the member.
  const [unseen, unread] = (await db.batch([db.prepare(UNSEEN_NOTICES).bind(userId), db.prepare(UNREAD_MESSAGES).bind(userId)]))
    .map((part) => part.results[0]);
  const sender = new URL(c.req.url).origin;
  const taken = await Promise.all(devices.map(async (device) => {
    const status = await sendPush(c.env, device, { ...message, count: unseen.unseen + unread.n }, sender).catch((error) => error.message);
    if (status >= 200 && status < 300) return 1;
    if (status === 404 || status === 410) await db.prepare('DELETE FROM push_devices WHERE endpoint = ?').bind(device.endpoint).run();
    else await writeLog(db, { source: 'server', level: 'warn', userId, message: `push to ${new URL(device.endpoint).hostname}: ${status}` });
    return 0;
  }));
  return taken.reduce((sum, one) => sum + one, 0);
}

// Tells members what the actor has just done around them: a line on their notices page, and a
// push. A notice of a kind that is told once, and is already there, is not told again unless a
// day has passed, so nobody can ring a person over and over.
async function notify(c, actor, notices) {
  const db = c.env.DB;
  const now = Date.now();
  const fresh = [];
  for (const notice of notices) {
    const { user, kind, post = null, comment = null, share = null } = notice;
    if (user === actor.id) continue;
    if (TOLD_ONCE.includes(kind)) {
      const old = await db
        .prepare('SELECT id, at FROM notices WHERE user_id = ? AND actor_id = ? AND kind = ? AND IFNULL(post_id, 0) = ? AND IFNULL(comment_id, 0) = ?')
        .bind(user, actor.id, kind, post ?? 0, comment ?? 0)
        .first();
      if (old) {
        const again = old.at < now - DAY_MS;
        await db.prepare('UPDATE notices SET share_id = ?, at = ? WHERE id = ?').bind(share, again ? now : old.at, old.id).run();
        if (again) fresh.push(notice);
        continue;
      }
    }
    const row = await db
      .prepare(`INSERT INTO notices (user_id, actor_id, kind, at, post_id, comment_id, share_id) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING RETURNING id`)
      .bind(user, actor.id, kind, now, post, comment, share)
      .first();
    if (row) fresh.push(notice);
  }
  // Someone who has not chosen a name yet is told about on the page, once they have one.
  if (!actor.name || !fresh.length) return;
  later(c, Promise.all(fresh.map((notice) => {
    const text = noticeText(notice.kind, { onShare: notice.onShare, onComment: notice.kind === 'felt' && !!notice.comment });
    const body = notice.feeling ? `${FEELING_EMOJI[notice.feeling] ?? notice.feeling} ${text}` : notice.words ? `${text}: ${notice.words}` : text;
    return pushTo(c, notice.user, noticeGroup(notice.kind), {
      title: actor.name,
      body,
      path: noticePath(notice.kind, { ...notice, actor: actor.id }),
    });
  })));
}

// Of these members, those who may read the post: everyone when it is public, and otherwise its
// writer and the writer's friends. Nobody is told about words they are not allowed to open.
async function readersOf(db, post, ids) {
  if (post.audience === 'public' || !ids.length) return ids;
  const { results } = await db
    .prepare(`SELECT value AS id FROM json_each(?2) WHERE value = ?1 OR value IN (${FRIEND_IDS})`)
    .bind(post.author_id, JSON.stringify(ids))
    .all();
  return results.map((row) => row.id);
}

const app = new Hono();

// The Android app (android/) shows the network without an address bar only because the site
// vouches for it here, by the fingerprint of the key the APK is signed with.
const ANDROID_APP = {
  package: 'il.pardes.freenetwork',
  fingerprint: '19:B0:34:EE:64:B7:F3:48:A6:DA:41:93:28:64:DC:87:06:F3:D5:FB:2C:2E:10:A7:62:70:E5:0B:BF:AB:B8:F0',
};

app.get('/.well-known/assetlinks.json', (c) => c.json([{
  relation: ['delegate_permission/common.handle_all_urls'],
  target: { namespace: 'android_app', package_name: ANDROID_APP.package, sha256_cert_fingerprints: [ANDROID_APP.fingerprint] },
}]));

// The philosophy on /why is public. The claims about Facebook have not been checked by a
// lawyer yet, so only members see them: for everyone else they are cut out of the page.
app.get('/why*', async (c) => {
  const page = await c.env.ASSETS.fetch(new URL('/why', c.req.url));
  const shown = (await sessionUser(c))
    ? page
    : new HTMLRewriter().on('[data-unreviewed]', { element: (element) => element.remove() }).transform(page);
  const response = new Response(shown.body, shown);
  response.headers.set('Cache-Control', 'private, no-store');
  return response;
});

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

async function sessionUser(c) {
  const token = getCookie(c, SESSION_COOKIE);
  if (!token) return null;
  const hash = await sha256(token);
  const now = Date.now();
  const user = await c.env.DB
    .prepare(`SELECT u.id, u.name, u.email, u.avatar_id, u.cover_id, u.google_sub, u.lat, u.lng, u.status_text, u.status_at, u.notices_seen_at, u.push_muted, s.seen_at AS session_seen
      FROM sessions s JOIN users u ON u.id = s.user_id
      WHERE s.token_hash = ? AND s.expires_at > ?`)
    .bind(hash, now)
    .first();
  if (!user) return null;
  c.set('session', sessionMark(hash));
  // When the session was last used is kept to the hour: enough to tell a device that is in use
  // from one that was forgotten, without a write on every request.
  if (now - (user.session_seen ?? 0) > SEEN_EVERY_MS) {
    later(c, c.env.DB.prepare('UPDATE sessions SET seen_at = ?, device = ? WHERE token_hash = ?').bind(now, deviceName(c.req.header('User-Agent')), hash).run());
  }
  return user;
}

app.use('/api/*', async (c, next) => {
  const user = await sessionUser(c);
  if (user) c.set('user', user);
  await next();
});

// The browser tells that the network was opened in it, and how: as the installed app, from the
// home screen or in a plain browser, and what brought it here. Only words from a short list of
// letters are kept, so nothing a person wrote can ride in.
const JOURNAL_WORD = /^[\w.:/-]{1,80}$/;

app.post('/api/journal', async (c) => {
  const input = await readJson(c);
  const words = ['mode', 'display', 'from', 'path', 'app', 'note']
    .filter((name) => typeof input[name] === 'string' && JOURNAL_WORD.test(input[name]))
    .map((name) => `${name}=${input[name]}`);
  await journal(c, 'open', words.join(' '));
  return c.json({ ok: true });
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
      WHERE token_hash = ?2 AND used_at IS NULL AND expires_at > ?1 RETURNING email, invited_by, name`)
    .bind(now, await sha256(token))
    .first();
  if (!used) return c.json({ error: 'הקישור פג או שכבר השתמשו בו' }, 400);

  const created = await c.env.DB
    .prepare('INSERT INTO users (email, name, created_at) VALUES (?, ?, ?) ON CONFLICT (email) DO NOTHING')
    .bind(used.email, used.name ?? '', now)
    .run();
  const user = await c.env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(used.email).first();
  if (used.invited_by) await befriend(c.env.DB, used.invited_by, user.id, now);
  if (used.invited_by && created.meta.changes) await notify(c, { id: user.id, name: used.name ?? '' }, [{ user: used.invited_by, kind: 'joined' }]);

  await createSession(c, user.id, created.meta.changes ? 'link, joined' : 'link');
  return c.json({ ok: true, joined: !!created.meta.changes });
});

// Google signs in only someone who linked it from their page, or whose invitation was sent to
// the address Google vouches for, so the network stays invitation-only. The state cookie proves
// the redirect back belongs to this browser.
app.get('/api/auth/google', (c) => {
  if (!c.env.GOOGLE_CLIENT_ID) return c.json({ error: 'לא נמצא' }, 404);
  // quiet: the installed app asks Google, without showing anything, who is signed in on the
  // phone. A member Google vouches for is let in; anyone else just gets the login screen.
  const mode = c.req.query('mail') === '1' ? 'mail' : c.req.query('link') === '1' ? 'link' : c.req.query('quiet') === '1' ? 'quiet' : 'login';
  if (mode === 'link' && !c.get('user')) return c.redirect('/login');
  if (mode === 'mail' && !isAdmin(c)) return c.redirect('/');

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
    // To send invitations, Google is asked for sending, and for reading the contacts so that an
    // address completes itself. Never for reading mail. The admin may tick only the sending.
    scope: mode === 'mail' ? ['openid', 'email', SEND_SCOPE, ...CONTACT_SCOPES].join(' ') : 'openid email profile',
    prompt: mode === 'mail' ? 'consent' : mode === 'quiet' ? 'none' : 'select_account',
    ...(mode === 'mail' ? { access_type: 'offline' } : {}),
    state,
  });
  return c.redirect(url.href);
});

app.get('/api/auth/google/callback', async (c) => {
  if (!c.env.GOOGLE_CLIENT_ID) return c.json({ error: 'לא נמצא' }, 404);
  const [mode, state] = (getCookie(c, GOOGLE_STATE_COOKIE) ?? '').split('.');
  deleteCookie(c, GOOGLE_STATE_COOKIE, { path: GOOGLE_STATE_PATH });
  // A quiet attempt that did not end in a login says nothing: the login screen is all that follows.
  // What Google said when it sent no code (the member has to choose an account, for example).
  const refusal = /^[a-z_]{1,40}$/.test(c.req.query('error') ?? '') ? c.req.query('error') : null;
  const finish = async (result) => {
    await journal(c, 'google', `${mode || 'no state cookie'}: ${result}${refusal ? ` (${refusal})` : ''}`);
    return c.redirect(`/?google=${mode === 'quiet' && result !== 'login' ? 'quiet' : result}`);
  };

  const code = c.req.query('code');
  if (!state || state !== c.req.query('state') || !code) return finish('failed');
  const google = await googleSubject(c, code);
  if (!google) return finish('failed');
  const subject = google.sub;

  const db = c.env.DB;
  if (mode === 'mail') {
    const back = async (result) => {
      await journal(c, 'google', `mail: ${result}`);
      return c.redirect(`/admin?google=${result}`);
    };
    if (!isAdmin(c)) return finish('failed');
    if (!google.refreshToken || !google.address || !google.scopes.includes(SEND_SCOPE)) return back('mail-refused');
    await db
      .prepare(`INSERT INTO mailers (user_id, address, refresh_token, created_at, contacts) VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT (user_id) DO UPDATE SET address = ?2, refresh_token = ?3, created_at = ?4, contacts = ?5`)
      .bind(c.get('user').id, google.address, await seal(c.env.GOOGLE_CLIENT_SECRET, google.refreshToken), Date.now(),
        CONTACT_SCOPES.some((scope) => google.scopes.includes(scope)) ? 1 : 0)
      .run();
    return back('mail');
  }
  const owner = await db.prepare('SELECT id FROM users WHERE google_sub = ?').bind(subject).first();
  if (mode === 'link') {
    const user = c.get('user');
    if (!user) return finish('failed');
    if (owner && owner.id !== user.id) return finish('taken');
    await db.prepare('UPDATE users SET google_sub = ? WHERE id = ?').bind(subject, user.id).run();
    await importGooglePicture(c, user.id, google.picture);
    return finish('linked');
  }
  if (owner) {
    await createSession(c, owner.id, mode === 'quiet' ? 'google, quiet' : 'google');
    return finish('login');
  }
  if (!google.email) return finish('unknown');

  // A member who was invited at this address: Google vouches that it is theirs, so the account
  // is linked from now on.
  const member = await db.prepare('SELECT id FROM users WHERE email = ? AND google_sub IS NULL').bind(google.email).first();
  if (member) {
    await db.prepare('UPDATE users SET google_sub = ? WHERE id = ?').bind(subject, member.id).run();
    await importGooglePicture(c, member.id, google.picture);
    await createSession(c, member.id, mode === 'quiet' ? 'google by address, quiet' : 'google by address');
    return finish('login');
  }

  // Someone new whose invitation still waits: signing in with that address takes it. Only when
  // they asked to, not in a quiet attempt.
  if (mode === 'quiet') return finish('unknown');
  const now = Date.now();
  const invite = await db
    .prepare(`UPDATE login_tokens SET used_at = ?1 WHERE token_hash = (SELECT token_hash FROM login_tokens
        WHERE email = ?2 AND invited_by IS NOT NULL AND used_at IS NULL AND expires_at > ?1 ORDER BY created_at DESC LIMIT 1)
      RETURNING invited_by, name`)
    .bind(now, google.email)
    .first();
  if (!invite) return finish('unknown');
  const user = await db
    .prepare('INSERT INTO users (email, name, google_sub, created_at) VALUES (?, ?, ?, ?) RETURNING id')
    .bind(google.email, invite.name ?? google.name ?? '', subject, now)
    .first();
  await befriend(db, invite.invited_by, user.id, now);
  await notify(c, { id: user.id, name: invite.name ?? google.name ?? '' }, [{ user: invite.invited_by, kind: 'joined' }]);
  await importGooglePicture(c, user.id, google.picture);
  await createSession(c, user.id, 'google, joined');
  return finish('joined');
});

app.post('/api/auth/logout', async (c) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token) {
    const hash = await sha256(token);
    await journal(c, 'logout', null);
    // A device that was logged out is not told anything any more.
    await c.env.DB.batch([
      c.env.DB.prepare('DELETE FROM sessions WHERE token_hash = ?').bind(hash),
      c.env.DB.prepare('DELETE FROM push_devices WHERE session_hash = ?').bind(hash),
    ]);
  }
  deleteCookie(c, SESSION_COOKIE, { path: '/' });
  return c.json({ ok: true });
});

app.get('/api/me', async (c) => {
  const user = c.get('user');
  const [waiting, unread, notices] = !user ? [] : (await c.env.DB.batch([
    c.env.DB.prepare(`SELECT COUNT(*) AS n FROM friendships WHERE addressee_id = ? AND status = 'pending'`).bind(user.id),
    c.env.DB.prepare(UNREAD_MESSAGES).bind(user.id),
    c.env.DB.prepare(UNSEEN_NOTICES).bind(user.id),
  ])).map((part) => part.results[0]);
  return c.json({
    user: user ? {
      id: user.id,
      name: user.name,
      avatar: user.avatar_id,
      cover: user.cover_id,
      admin: isAdmin(c),
      friendRequests: waiting.n,
      unreadMessages: unread.n,
      notices: notices.unseen,
      pushMuted: user.push_muted ? user.push_muted.split(',') : [],
      googleLinked: !!user.google_sub,
      place: place(user),
      status: status(user),
    } : null,
    inviteOnly: c.env.INVITE_ONLY === '1',
    googleLogin: !!c.env.GOOGLE_CLIENT_ID,
    // The public half of the key that pushes are signed with. Without one there are no pushes.
    pushKey: pushReady(c.env) ? c.env.VAPID_PUBLIC : null,
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

// Sharing a place is one decision: it is stored when shared and erased when not. The member
// picks the spot, and it is rounded here so the server never holds anything more exact.
app.put('/api/me/place', requireUser, async (c) => {
  const { lat, lng } = await readJson(c);
  const valid = Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180;
  if (!valid) return c.json({ error: 'מיקום לא תקין' }, 400);
  const round = (value) => Number(value.toFixed(PLACE_DECIMALS));
  const saved = { lat: round(lat), lng: round(lng) };
  await c.env.DB.prepare('UPDATE users SET lat = ?, lng = ?, place_at = ? WHERE id = ?').bind(saved.lat, saved.lng, Date.now(), c.get('user').id).run();
  return c.json({ place: saved });
});

app.delete('/api/me/place', requireUser, async (c) => {
  await c.env.DB.prepare('UPDATE users SET lat = NULL, lng = NULL, place_at = NULL WHERE id = ?').bind(c.get('user').id).run();
  return c.json({ ok: true });
});

// One line for friends. Writing a new one replaces the old, and deleting really deletes.
app.put('/api/me/status', requireUser, async (c) => {
  const text = cleanText((await readJson(c)).text, MAX_STATUS);
  if (!text) return c.json({ error: `צריך משפט אחד, עד ${MAX_STATUS} תווים` }, 400);
  const at = Date.now();
  await c.env.DB.prepare('UPDATE users SET status_text = ?, status_at = ? WHERE id = ?').bind(text, at, c.get('user').id).run();
  return c.json({ status: { text, at } });
});

app.delete('/api/me/status', requireUser, async (c) => {
  await c.env.DB.prepare('UPDATE users SET status_text = NULL, status_at = NULL WHERE id = ?').bind(c.get('user').id).run();
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

// How many people a friends post will reach, which the composer shows before publishing, and
// the audience the member chose for the last post they wrote, which the next one starts from.
// A share does not count: it always goes to friends, and nobody chose that.
app.get('/api/audience', requireUser, async (c) => {
  const row = await c.env.DB
    .prepare(`SELECT (SELECT COUNT(*) FROM (${FRIEND_IDS})) AS n,
      (SELECT audience FROM posts WHERE author_id = ?1 AND shared_id IS NULL ORDER BY id DESC LIMIT 1) AS last`)
    .bind(c.get('user').id)
    .first();
  return c.json({ friends: row.n, last: row.last ?? 'friends' });
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
  return c.json({ posts: await showPosts(c.env.DB, results, user), hasMore: results.length === PAGE_SIZE });
});

// The live page of a post, as the writer gave it: '' or nothing for none, null when the address
// cannot be a live page. Its title and picture are read now, once, for every reader to come.
async function livePage(c, raw) {
  if (raw === undefined || raw === null || (typeof raw === 'string' && !raw.trim())) return { url: null };
  const where = { own: new URL(c.req.url).host, local: c.env.DEV_LOGIN === '1' && isLocalhost(c) };
  const url = pageUrl(raw, where);
  if (!url) return null;
  return { url, ...(await readPage(url, where)) };
}

const PAGE_PROBLEM = 'אפשר לשבץ רק כתובת מלאה של דף, שמתחילה ב-https://';
const pageColumns = (page) => [page.url, page.title ?? null, page.text ?? null, page.picture ?? null, page.url ? (page.framable ? 1 : 0) : null];

app.post('/api/posts', requireUser, async (c) => {
  const input = await readJson(c);
  const user = c.get('user');
  const body = typeof input.body === 'string' ? input.body.trim() : '';
  const imageIds = Array.isArray(input.images) ? [...new Set(input.images)] : [];
  if (body.length > MAX_POST) return c.json({ error: 'הפוסט ארוך מדי' }, 400);
  const page = await livePage(c, input.page);
  if (!page) return c.json({ error: PAGE_PROBLEM }, 400);
  if (!body && !imageIds.length && !page.url) return c.json({ error: 'הפוסט ריק' }, 400);
  if (imageIds.length > MAX_IMAGES_PER_POST) return c.json({ error: 'עד 4 תמונות בפוסט' }, 400);
  if (!['friends', 'public'].includes(input.audience)) return c.json({ error: 'קהל לא תקין' }, 400);

  for (const imageId of imageIds) {
    const owned = typeof imageId === 'string' && await c.env.DB.prepare(UNUSED_IMAGE).bind(imageId, user.id).first();
    if (!owned) return c.json({ error: 'תמונה לא תקינה' }, 400);
  }

  const row = await c.env.DB
    .prepare(`INSERT INTO posts (author_id, body, youtube_id, audience, created_at, page_url, page_title, page_text, page_picture, page_framable)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id`)
    .bind(user.id, body, findYoutubeId(body), input.audience, Date.now(), ...pageColumns(page))
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
  const [[shown], commentFeelings] = await Promise.all([
    showPosts(c.env.DB, [post], viewer),
    reactionsOf(c.env.DB, 'comment_id', results.filter((row) => !row.removed_by).map((row) => row.id), viewer),
  ]);
  return c.json({
    post: shown,
    comments: results.map((row) => serializeComment(row, viewer, post, commentFeelings)),
  });
});

app.on(['PUT', 'DELETE'], '/api/posts/:id/reaction', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const post = id && (await loadVisiblePost(c, id));
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);
  return react(c, 'post_id', id, post, post.author_id);
});

app.on(['PUT', 'DELETE'], '/api/comments/:id/reaction', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const comment = id && (await c.env.DB.prepare('SELECT post_id, author_id FROM comments WHERE id = ? AND removed_by IS NULL').bind(id).first());
  const post = comment && (await loadVisiblePost(c, comment.post_id));
  if (!post) return c.json({ error: 'התגובה לא נמצאה' }, 404);
  return react(c, 'comment_id', id, post, comment.author_id);
});

// Only the words of a post can change, and only by its writer. The audience stays as the
// people who answered knew it, and readers are told that the post was edited.
app.put('/api/posts/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const user = c.get('user');
  const post = id && (await c.env.DB.prepare(`${POST_SELECT} WHERE p.id = ? AND p.author_id = ? AND p.deleted_at IS NULL`).bind(id, user.id).first());
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  const input = await readJson(c);
  if (typeof input.body !== 'string') return c.json({ error: 'הפוסט ריק' }, 400);
  const body = input.body.trim();
  if (body.length > MAX_POST) return c.json({ error: 'הפוסט ארוך מדי' }, 400);
  // The live page changes only when it is given: '' takes it away. An address that stays the
  // same is read again, so a page that changed its title shows the new one.
  const page = input.page === undefined || post.shared_id ? undefined : await livePage(c, input.page);
  if (page === null) return c.json({ error: PAGE_PROBLEM }, 400);
  const pageAfter = page === undefined ? post.page_url : page.url;
  // A share may stay without words of its own: the post it quotes speaks for it.
  if (!body && post.image_ids === '[]' && !post.shared_id && !pageAfter) return c.json({ error: 'הפוסט ריק' }, 400);
  const pageChanged = page !== undefined && page.url !== post.page_url;
  if (body !== post.body || pageChanged) {
    Object.assign(post, { body, youtube_id: findYoutubeId(body), edited_at: Date.now() });
    await c.env.DB.prepare('UPDATE posts SET body = ?, youtube_id = ?, edited_at = ? WHERE id = ?').bind(post.body, post.youtube_id, post.edited_at, id).run();
  }
  if (page !== undefined) {
    const [url, title, text, picture, framed] = pageColumns(page);
    Object.assign(post, { page_url: url, page_title: title, page_text: text, page_picture: picture, page_framable: framed });
    await c.env.DB.prepare('UPDATE posts SET page_url = ?, page_title = ?, page_text = ?, page_picture = ?, page_framable = ? WHERE id = ?')
      .bind(url, title, text, picture, framed, id).run();
  }
  return c.json({ post: (await showPosts(c.env.DB, [post], user))[0] });
});

app.delete('/api/posts/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const now = Date.now();
  const result = await c.env.DB
    .prepare('UPDATE posts SET deleted_at = ? WHERE id = ? AND author_id = ? AND deleted_at IS NULL')
    .bind(now, id, c.get('user').id)
    .run();
  if (!result.meta.changes) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  // A share with no words of its own is only the quote, so it goes with the post it quoted.
  // A share that says something stays, and tells its readers that the original was deleted.
  const { results: quotes } = await c.env.DB
    .prepare(`UPDATE posts SET deleted_at = ? WHERE shared_id = ? AND body = '' AND deleted_at IS NULL RETURNING id`)
    .bind(now, id)
    .all();
  const gone = JSON.stringify([id, ...quotes.map((quote) => quote.id)]);

  // Deleting a post really deletes its images and those in its comments, not only the links,
  // and what people felt about it and about its comments.
  const posts = 'SELECT value FROM json_each(?1)';
  const attached = `post_id IN (${posts}) OR comment_id IN (SELECT id FROM comments WHERE post_id IN (${posts}))`;
  await c.env.DB.batch([
    c.env.DB.prepare(`DELETE FROM reactions WHERE ${attached}`).bind(gone),
    c.env.DB.prepare(`DELETE FROM notices WHERE post_id IN (${posts}) OR share_id IN (${posts})`).bind(gone),
  ]);
  const { results } = await c.env.DB.prepare(`SELECT id FROM images WHERE ${attached}`).bind(gone).all();
  if (results.length) {
    await c.env.IMAGES.delete(results.map((image) => image.id));
    await c.env.DB.prepare(`DELETE FROM images WHERE ${attached}`).bind(gone).run();
  }
  return c.json({ ok: true });
});

// One click publishes a post again under the name of whoever shares it, for their friends.
// Sharing from someone's share still quotes the original post, and remembers the way it came.
app.post('/api/posts/:id/share', requireUser, async (c) => {
  const db = c.env.DB;
  const user = c.get('user');
  const id = parseId(c.req.param('id'));
  const found = id && (await loadVisiblePost(c, id));
  const original = found?.shared_id ? await loadVisiblePost(c, found.shared_id) : found;
  if (!original) return c.json({ error: 'הפוסט לא נמצא' }, 404);
  if (original.author_id === user.id) return c.json({ error: 'הפוסט הזה שלך, הוא כבר על הקיר שלך' }, 400);

  // Sharing a post that the member already shared changes nothing.
  const made = await db
    .prepare(`INSERT INTO posts (author_id, body, audience, created_at, shared_id, shared_via)
      VALUES (?, '', 'friends', ?, ?, ?) ON CONFLICT DO NOTHING`)
    .bind(user.id, Date.now(), original.id, found.id === original.id ? null : found.id)
    .run();
  const [shown] = await showPosts(db, [original], user);
  if (made.meta.changes) {
    // The writer is told, and so is whoever's share it came through. The share opens for the
    // writer only if the two are friends, and otherwise the notice leads to their own post.
    const share = shown.shares.mine;
    const told = [{ user: original.author_id, kind: 'share', post: original.id, share, shareVisible: await areFriends(db, user.id, original.author_id) }];
    if (found.id !== original.id && found.author_id !== original.author_id) {
      told.push({ user: found.author_id, kind: 'reshare', post: found.id, share, shareVisible: true });
    }
    await notify(c, user, told);
  }
  return c.json({ id: shown.shares.mine, shares: shown.shares }, 201);
});

// Whoever wrote a post sees exactly who shared it, and through whose share it reached them.
// The share itself opens only if its writer is a friend, like any post for friends.
app.get('/api/posts/:id/shares', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const user = c.get('user');
  const post = id && (await c.env.DB
    .prepare('SELECT id FROM posts WHERE id = ? AND author_id = ? AND deleted_at IS NULL AND shared_id IS NULL')
    .bind(id, user.id)
    .first());
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);
  const { results } = await c.env.DB
    .prepare(`SELECT p.id, p.created_at, u.id AS user_id, u.name, u.avatar_id AS avatar, ${VISIBLE_TO} AS visible,
        w.id AS via_id, w.name AS via_name
      FROM posts p JOIN users u ON u.id = p.author_id
      LEFT JOIN posts v ON v.id = p.shared_via LEFT JOIN users w ON w.id = v.author_id
      WHERE p.shared_id = ?2 AND p.deleted_at IS NULL ORDER BY p.id DESC LIMIT ${MAX_SHARE_NAMES}`)
    .bind(user.id, id)
    .all();
  return c.json({ shares: results.map((row) => ({
    user: { id: row.user_id, name: row.name, avatar: row.avatar },
    at: row.created_at,
    postId: row.visible ? row.id : null,
    via: row.via_id ? { id: row.via_id, name: row.via_name } : null,
  })) });
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
// An image in a message opens only for the two people of that conversation.
app.get('/api/images/:id', async (c) => {
  const id = c.req.param('id');
  const viewer = c.get('user');
  const image = await c.env.DB
    .prepare(`SELECT owner_id, comment_id,
      COALESCE(post_id, (SELECT post_id FROM comments WHERE id = images.comment_id)) AS post_id,
      EXISTS (SELECT 1 FROM users WHERE avatar_id = images.id OR cover_id = images.id) AS profile,
      EXISTS (SELECT 1 FROM messages m JOIN conversations k ON k.id = m.conversation_id
        WHERE m.id = images.message_id AND ?2 IN (k.low_id, k.high_id)) AS in_conversation
      FROM images WHERE id = ?1`)
    .bind(id, viewer?.id ?? 0)
    .first();
  const post = image?.post_id && (await loadVisiblePost(c, image.post_id));
  const visible = image?.profile || post || image?.in_conversation || (image && !image.post_id && viewer?.id === image.owner_id);
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

  // Who is told: whoever wrote the comment that was answered, the writer of the post, and the
  // others who wrote in the same thread. Each of them once, and only if they may read the post.
  const told = new Map();
  if (parentId) {
    const [top, others] = await c.env.DB.batch([
      c.env.DB.prepare('SELECT author_id FROM comments WHERE id = ?').bind(parentId),
      c.env.DB.prepare('SELECT DISTINCT author_id FROM comments WHERE parent_id = ? AND removed_by IS NULL AND id < ?').bind(parentId, row.id),
    ]);
    told.set(top.results[0].author_id, 'reply');
    for (const other of others.results) if (!told.has(other.author_id)) told.set(other.author_id, 'thread');
  }
  if (told.get(post.author_id) !== 'reply') told.set(post.author_id, 'comment');
  told.delete(user.id);
  const readers = await readersOf(c.env.DB, post, [...told.keys()]);
  await notify(c, user, readers.map((reader) => ({
    user: reader, kind: told.get(reader), post: id, comment: row.id, words: body ? body.slice(0, 140) : 'תמונה', onShare: !!post.shared_id,
  })));
  return c.json({ id: row.id }, 201);
});

// A comment is rewritten only by whoever wrote it. The owner of the post may remove it, not reword it.
app.put('/api/comments/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const user = c.get('user');
  const comment = id && (await c.env.DB
    .prepare(`SELECT m.body, (SELECT id FROM images WHERE comment_id = m.id) AS image_id FROM comments m
      JOIN posts p ON p.id = m.post_id WHERE m.id = ? AND m.author_id = ? AND m.removed_by IS NULL AND p.deleted_at IS NULL`)
    .bind(id, user.id)
    .first());
  if (!comment) return c.json({ error: 'התגובה לא נמצאה' }, 404);

  const input = await readJson(c);
  const body = typeof input.body === 'string' ? input.body.trim() : '';
  if (body.length > MAX_COMMENT || (!body && !comment.image_id)) return c.json({ error: 'התגובה ריקה או ארוכה מדי' }, 400);
  if (body !== (comment.body ?? '')) await c.env.DB.prepare('UPDATE comments SET body = ?, edited_at = ? WHERE id = ?').bind(body, Date.now(), id).run();
  return c.json({ ok: true });
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
  await c.env.DB.batch([
    c.env.DB.prepare('UPDATE comments SET body = NULL, removed_by = ? WHERE id = ?').bind(removedBy, id),
    c.env.DB.prepare('DELETE FROM reactions WHERE comment_id = ?').bind(id),
    c.env.DB.prepare('DELETE FROM notices WHERE comment_id = ?').bind(id),
  ]);

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

// The picture of a post's live page, for whoever may read the post.
app.get('/api/posts/:id/page-picture', async (c) => {
  const id = parseId(c.req.param('id'));
  const post = id && (await loadVisiblePost(c, id));
  const picture = post?.page_picture && (await pagePicture(post.page_picture));
  return picture || c.json({ error: 'לא נמצא' }, 404);
});

// A picture from outside, passed through this server so the reader's browser contacts nobody else.
const pictureUrl = (src) => `/api/picture?src=${encodeURIComponent(src)}`;
const withPicture = (item) => ({ ...item, image: item.image ? pictureUrl(item.image) : null });

app.get('/api/picture', requireUser, async (c) => {
  let src;
  try {
    src = new URL(c.req.query('src') ?? '');
  } catch {
    return c.json({ error: 'לא נמצא' }, 404);
  }
  if (src.protocol !== 'https:' || !PICTURE_HOSTS.includes(src.host)) return c.json({ error: 'לא נמצא' }, 404);
  const response = await fetch(src, {
    headers: { 'User-Agent': 'free-network-digest/1.0 (https://free-network.orimosenzon.workers.dev)' },
    cf: { cacheEverything: true, cacheTtl: 86400 },
  }).catch(() => null);
  const type = response?.headers.get('Content-Type') ?? '';
  if (!response?.ok || !/^image\/(jpeg|png|gif|webp)/.test(type)) return c.json({ error: 'לא נמצא' }, 404);
  return new Response(response.body, {
    headers: { 'Content-Type': type, 'Cache-Control': 'private, max-age=86400', 'X-Content-Type-Options': 'nosniff' },
  });
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

// Aerial pictures of the earth, for the globe seen from close up. They pass through this server,
// so whoever holds the pictures never learns who looked at which place. Only members get them,
// and only as far in as the pictures go.
const TILE_ZOOM = [3, 17];
const TILE_CACHE_SECONDS = 30 * 24 * 60 * 60;

app.get('/api/tiles/:z/:x/:y', requireUser, async (c) => {
  const [z, x, y] = ['z', 'x', 'y'].map((name) => parseId(c.req.param(name)));
  const inside = z !== null && x !== null && y !== null && z >= TILE_ZOOM[0] && z <= TILE_ZOOM[1] && x < 2 ** z && y < 2 ** z;
  if (!inside) return c.json({ error: 'לא נמצא' }, 404);
  const origin = c.env.TILES_ORIGIN ?? 'https://server.arcgisonline.com';
  const response = await fetch(`${origin}/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`, {
    cf: { cacheEverything: true, cacheTtl: TILE_CACHE_SECONDS },
  }).catch(() => null);
  const type = response?.ok ? response.headers.get('content-type') ?? '' : '';
  if (!type.startsWith('image/')) return c.json({ error: 'התצלום לא הגיע' }, 502);
  return new Response(response.body, {
    headers: {
      'Content-Type': type.split(';')[0],
      'Cache-Control': `private, max-age=${TILE_CACHE_SECONDS}`,
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
  // How many friends they have is told to whoever may see who they are: the person and their friends.
  const friends = seesAll ? (await c.env.DB.prepare(`SELECT COUNT(*) AS n FROM users WHERE id IN (${FRIEND_IDS}) AND name <> ''`).bind(id).first()).n : null;
  return c.json({
    user: person,
    friendship,
    friends,
    posts: await showPosts(c.env.DB, results, viewer),
    hasMore: results.length === PAGE_SIZE,
  });
});

// The friends of a member, for the member and for their friends only: this is how people who
// share a friend find each other. Beside each one is what they are to the reader, and how many
// friends the two have in common.
app.get('/api/users/:id/friends', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const viewer = c.get('user').id;
  const db = c.env.DB;
  const person = id && (await db.prepare(`SELECT id, name, avatar_id AS avatar FROM users WHERE id = ? AND name <> ''`).bind(id).first());
  if (!person) return c.json({ error: 'המשתמש לא נמצא' }, 404);
  if (id !== viewer && !(await areFriends(db, viewer, id))) return c.json({ error: 'רק חברים רואים את רשימת החברים' }, 403);
  const { results } = await db
    .prepare(`SELECT u.id, u.name, u.avatar_id AS avatar,
        (SELECT f.status || (f.requester_id = ?1) FROM friendships f
          WHERE (f.requester_id = ?1 AND f.addressee_id = u.id) OR (f.requester_id = u.id AND f.addressee_id = ?1)) AS link,
        (SELECT COUNT(*) FROM friendships g WHERE g.status = 'accepted'
          AND ((g.requester_id = u.id AND g.addressee_id IN (${FRIEND_IDS})) OR (g.addressee_id = u.id AND g.requester_id IN (${FRIEND_IDS})))) AS mutual
      FROM users u WHERE u.id IN (${FRIEND_IDS.replaceAll('?1', '?2')}) AND u.name <> '' ORDER BY u.name LIMIT 1000`)
    .bind(viewer, id)
    .all();
  const LINKS = { accepted0: 'friends', accepted1: 'friends', pending1: 'outgoing', pending0: 'incoming' };
  return c.json({
    user: person,
    friends: results.map((row) => ({
      id: row.id, name: row.name, avatar: row.avatar, mutual: row.id === viewer ? 0 : row.mutual,
      friendship: row.id === viewer ? 'self' : LINKS[row.link] ?? null,
    })),
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

// Friends of the reader's friends who are not connected to the reader yet, each with exactly
// the friends they have in common. The order is plain and said on the page: whoever shares more
// friends comes first, and then by name.
app.get('/api/friends/suggested', requireUser, async (c) => {
  const { results } = await c.env.DB
    .prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, COUNT(*) AS n,
        json_group_array(json_object('id', m.id, 'name', m.name)) AS through
      FROM (SELECT requester_id AS mine, addressee_id AS theirs FROM friendships WHERE status = 'accepted'
        UNION ALL SELECT addressee_id, requester_id FROM friendships WHERE status = 'accepted') e
      JOIN users u ON u.id = e.theirs JOIN users m ON m.id = e.mine
      WHERE e.mine IN (${FRIEND_IDS}) AND e.theirs <> ?1 AND u.name <> '' AND m.name <> ''
        AND e.theirs NOT IN (SELECT requester_id FROM friendships WHERE addressee_id = ?1
          UNION SELECT addressee_id FROM friendships WHERE requester_id = ?1)
      GROUP BY u.id ORDER BY n DESC, u.name LIMIT 40`)
    .bind(c.get('user').id)
    .all();
  return c.json({ people: results.map((row) => ({ id: row.id, name: row.name, avatar: row.avatar, through: JSON.parse(row.through) })) });
});

// The globe: every friend, and the place of those who chose to share one. Only friends are
// ever listed, so a place reaches nobody else.
app.get('/api/globe', requireUser, async (c) => {
  const { results } = await c.env.DB
    .prepare(`SELECT id, name, avatar_id AS avatar, lat, lng, status_text, status_at FROM users
      WHERE id IN (${FRIEND_IDS}) AND name <> '' ORDER BY name`)
    .bind(c.get('user').id)
    .all();
  return c.json({ friends: results.map((row) => ({
    id: row.id, name: row.name, avatar: row.avatar, place: place(row), status: status(row)?.text ?? null,
  })) });
});

// The update: a short, finite list of what happened lately, in widening circles. Friends come
// from the network itself, with no ranking: each list is simply the newest first. The rest
// comes from sources outside it, which the server reads so the reader's browser does not.
app.get('/api/update', requireUser, async (c) => {
  const db = c.env.DB;
  const me = c.get('user').id;
  const now = Date.now();
  const person = (row) => ({ id: row.id, name: row.name, avatar: row.avatar });
  const friend = `u.id IN (${FRIEND_IDS}) AND u.name <> ''`;
  const seen = parseId(c.req.query('seen') ?? '');
  // An expired line is erased, not just hidden.
  await db.prepare('UPDATE users SET status_text = NULL, status_at = NULL WHERE status_at <= ?').bind(now - STATUS_MS).run();

  const [statuses, fresh, replies, felt, shared, moved, joined, quiet] = (await db.batch([
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, u.status_text, u.status_at FROM users u
      WHERE ${friend} AND u.status_text IS NOT NULL ORDER BY u.status_at DESC LIMIT 8`).bind(me),
    // Posts the reader has not seen yet. What was seen is known only to their device, which says so
    // here. A device that remembers nothing gets the last three days.
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, COUNT(*) AS n, MAX(p.id) AS newest FROM posts p
      JOIN users u ON u.id = p.author_id
      WHERE ${friend} AND p.deleted_at IS NULL AND p.created_at > ?2 AND p.id > ?3
      GROUP BY u.id ORDER BY newest DESC LIMIT 5`).bind(me, now - (seen ? 14 : 3) * DAY_MS, seen ?? 0),
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, m.post_id, m.created_at, p.author_id = ?1 AS on_mine
      FROM comments m JOIN posts p ON p.id = m.post_id JOIN users u ON u.id = m.author_id
      LEFT JOIN comments parent ON parent.id = m.parent_id
      WHERE m.author_id <> ?1 AND m.removed_by IS NULL AND p.deleted_at IS NULL AND m.created_at > ?2
        AND (p.author_id = ?1 OR parent.author_id = ?1)
      ORDER BY m.id DESC LIMIT 4`).bind(me, now - 7 * DAY_MS),
    // What others felt this week about the reader's posts and comments.
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, r.kind, r.created_at, p.id AS post_id, r.post_id IS NOT NULL AS on_post
      FROM reactions r JOIN users u ON u.id = r.user_id
      LEFT JOIN comments m ON m.id = r.comment_id
      JOIN posts p ON p.id = COALESCE(r.post_id, m.post_id)
      WHERE r.user_id <> ?1 AND r.created_at > ?2 AND p.deleted_at IS NULL
        AND ((r.post_id IS NOT NULL AND p.author_id = ?1) OR m.author_id = ?1)
      ORDER BY r.created_at DESC, r.rowid DESC LIMIT 4`).bind(me, now - 7 * DAY_MS),
    // Who shared the reader's posts this week. The share opens only when its writer is a friend.
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, p.id AS share_id, p.shared_id AS post_id, p.created_at,
        p.author_id IN (${FRIEND_IDS}) AS visible
      FROM posts p JOIN users u ON u.id = p.author_id JOIN posts o ON o.id = p.shared_id
      WHERE o.author_id = ?1 AND o.deleted_at IS NULL AND p.deleted_at IS NULL AND p.created_at > ?2
      ORDER BY p.id DESC LIMIT 4`).bind(me, now - 7 * DAY_MS),
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, u.lat, u.lng, u.place_at FROM users u
      WHERE ${friend} AND u.lat IS NOT NULL AND u.place_at > ?2 ORDER BY u.place_at DESC LIMIT 3`).bind(me, now - 14 * DAY_MS),
    db.prepare(`SELECT u.id, u.name, u.avatar_id AS avatar, u.created_at FROM users u
      WHERE ${friend} AND u.created_at > ?2 ORDER BY u.created_at DESC LIMIT 3`).bind(me, now - 14 * DAY_MS),
    // Friends who have been here a month and said nothing in that time: a nudge to get in touch.
    db.prepare(`SELECT * FROM (SELECT u.id, u.name, u.avatar_id AS avatar,
        (SELECT MAX(created_at) FROM posts WHERE author_id = u.id AND deleted_at IS NULL) AS last
      FROM users u WHERE ${friend} AND u.created_at < ?2 AND u.status_text IS NULL)
      WHERE last IS NULL OR last < ?2 ORDER BY last IS NULL, last LIMIT 2`).bind(me, now - 30 * DAY_MS),
  ])).map((part) => part.results);

  const news = await outsideNews(c.env, now);
  return c.json({
    friends: {
      statuses: statuses.map((row) => ({ user: person(row), text: row.status_text, at: row.status_at })),
      fresh: fresh.map((row) => ({ user: person(row), count: row.n })),
      replies: replies.map((row) => ({ user: person(row), postId: row.post_id, at: row.created_at, onMine: !!row.on_mine })),
      felt: felt.map((row) => ({ user: person(row), kind: row.kind, postId: row.post_id, at: row.created_at, onPost: !!row.on_post })),
      shared: shared.map((row) => ({ user: person(row), postId: row.post_id, shareId: row.visible ? row.share_id : null, at: row.created_at })),
      moved: moved.map((row) => ({ user: person(row), place: place(row), at: row.place_at })),
      joined: joined.map((row) => ({ user: person(row), at: row.created_at })),
      quiet: quiet.filter((row) => ![...moved, ...joined].some((other) => other.id === row.id))
        .map((row) => ({ user: person(row), last: row.last })),
    },
    community: {
      events: news.events.slice(0, 12).map(withPicture),
      moreEvents: Math.max(0, news.events.length - 12),
      plans: news.plans.slice(0, 5),
    },
    israel: israelNews(news).map(withPicture),
    country: news.country.slice(0, 6).map(withPicture),
    world: news.world.slice(0, 6).map(withPicture),
  });
});

// The notices of the member, the newest first, in pages that go back to the beginning. Nothing
// is folded together and nothing is left out: every comment, feeling and share is its own line.
app.get('/api/notices', requireUser, async (c) => {
  const user = c.get('user');
  const kinds = NOTICE_GROUPS[c.req.query('group')];
  const [at, id] = (c.req.query('before') ?? '').split('.').map((part) => parseId(part));
  const now = Date.now();
  const { results } = await c.env.DB
    .prepare(`${NOTICE_SELECT}
      AND (?2 IS NULL OR n.kind IN (SELECT value FROM json_each(?2)))
      AND (n.at < ?3 OR (n.at = ?3 AND n.id < ?4))
      ORDER BY n.at DESC, n.id DESC LIMIT ?5`)
    .bind(user.id, kinds ? JSON.stringify(kinds) : null, at ?? Number.MAX_SAFE_INTEGER, id ?? Number.MAX_SAFE_INTEGER, NOTICES_PAGE + 1)
    .all();
  const page = results.slice(0, NOTICES_PAGE);
  return c.json({
    notices: page.map((row) => serializeNotice(row, user.notices_seen_at)),
    before: results.length > NOTICES_PAGE ? `${page.at(-1).at}.${page.at(-1).id}` : null,
    now,
  });
});

// The member has seen their notices up to the moment the page was given to them. The mark only
// moves forward, and never past now, so what comes later is still counted.
app.put('/api/notices/seen', requireUser, async (c) => {
  const { upTo } = await readJson(c);
  const now = Date.now();
  await c.env.DB
    .prepare('UPDATE users SET notices_seen_at = MAX(notices_seen_at, ?) WHERE id = ?')
    .bind(Number.isSafeInteger(upTo) && upTo > 0 && upTo < now ? upTo : now, c.get('user').id)
    .run();
  return c.json({ ok: true });
});

// This device asks to be told also when the network is closed. It is tied to the session, and
// a device that was used by someone else before now belongs to whoever is logged in on it.
app.put('/api/push', requireUser, async (c) => {
  if (!pushReady(c.env)) return c.json({ error: 'אין כאן התראות למכשיר' }, 404);
  const device = pushDevice(await readJson(c), { local: c.env.DEV_LOGIN === '1' && isLocalhost(c) });
  if (!device) return c.json({ error: 'המכשיר לא נרשם להתראות, נסו שוב' }, 400);
  const db = c.env.DB;
  const user = c.get('user');
  await db.batch([
    db.prepare(`INSERT INTO push_devices (endpoint, user_id, session_hash, p256dh, auth, created_at) VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT (endpoint) DO UPDATE SET user_id = excluded.user_id, session_hash = excluded.session_hash,
        p256dh = excluded.p256dh, auth = excluded.auth`)
      .bind(device.endpoint, user.id, await sha256(getCookie(c, SESSION_COOKIE)), device.p256dh, device.auth, Date.now()),
    db.prepare(`DELETE FROM push_devices WHERE user_id = ?1 AND endpoint NOT IN
      (SELECT endpoint FROM push_devices WHERE user_id = ?1 ORDER BY created_at DESC LIMIT ${MAX_PUSH_DEVICES})`).bind(user.id),
  ]);
  await journal(c, 'push', `on, through ${new URL(device.endpoint).hostname}`);
  return c.json({ ok: true });
});

app.delete('/api/push', requireUser, async (c) => {
  const { endpoint } = await readJson(c);
  const gone = await c.env.DB.prepare('DELETE FROM push_devices WHERE endpoint = ? AND user_id = ?').bind(String(endpoint), c.get('user').id).run();
  if (gone.meta.changes) await journal(c, 'push', 'off');
  return c.json({ ok: true });
});

// The kinds of things the member does not want pushed, on all their devices. They still appear
// on the notices page.
app.put('/api/push/muted', requireUser, async (c) => {
  const { muted } = await readJson(c);
  const kept = PUSH_GROUPS.filter((group) => Array.isArray(muted) && muted.includes(group));
  await c.env.DB.prepare('UPDATE users SET push_muted = ? WHERE id = ?').bind(kept.join(','), c.get('user').id).run();
  return c.json({ muted: kept });
});

// A push to the member's own devices, so they can see that it arrives and what it looks like.
app.post('/api/push/test', requireUser, async (c) => {
  const delivered = await pushTo(c, c.get('user').id, 'test', {
    title: 'הרשת החופשית', body: 'זו התראה לדוגמה. ככה זה ייראה כשמישהו יכתוב לך.', path: '/notices', tag: 'test', show: true,
  });
  return c.json({ delivered });
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
  if (accepted.meta.changes) {
    await notify(c, c.get('user'), [{ user: other, kind: 'accepted' }]);
    return c.json({ status: 'accepted' });
  }

  const asked = await c.env.DB
    .prepare(`INSERT INTO friendships (requester_id, addressee_id, status, created_at)
      VALUES (?, ?, 'pending', ?) ON CONFLICT DO NOTHING`)
    .bind(me, other, Date.now())
    .run();
  if (asked.meta.changes) await notify(c, c.get('user'), [{ user: other, kind: 'request' }]);
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

// Private messages. A conversation is between two members, and only the two of them ever get
// anything from these routes. It is addressed by the other member's id.
async function correspondent(c) {
  const id = parseId(c.req.param('userId'));
  if (!id || id === c.get('user').id) return null;
  return c.env.DB.prepare(`SELECT id, name, avatar_id AS avatar FROM users WHERE id = ? AND name <> ''`).bind(id).first();
}

// Every conversation of the reader, the one with the newest message first.
app.get('/api/messages', requireUser, async (c) => {
  const me = c.get('user').id;
  const { results } = await c.env.DB
    .prepare(`SELECT u.id, u.name, u.avatar_id AS avatar,
        m.sender_id, SUBSTR(m.body, 1, 140) AS body, m.created_at, m.removed_at,
        EXISTS (SELECT 1 FROM images WHERE message_id = m.id) AS has_image,
        (SELECT COUNT(*) FROM messages x WHERE x.conversation_id = k.id AND x.sender_id <> ?1 AND x.removed_at IS NULL
          AND x.id > CASE WHEN k.low_id = ?1 THEN k.low_read ELSE k.high_read END) AS unread
      FROM conversations k
      JOIN users u ON u.id = CASE WHEN k.low_id = ?1 THEN k.high_id ELSE k.low_id END
      JOIN messages m ON m.id = (SELECT MAX(id) FROM messages WHERE conversation_id = k.id)
      WHERE k.low_id = ?1 OR k.high_id = ?1
      ORDER BY m.id DESC LIMIT 200`)
    .bind(me)
    .all();
  return c.json({ conversations: results.map((row) => ({
    user: { id: row.id, name: row.name, avatar: row.avatar },
    last: { mine: row.sender_id === me, body: row.body, image: !!row.has_image, at: row.created_at, removed: !!row.removed_at },
    unread: row.unread,
  })) });
});

// One conversation. Asked plainly it is the newest page, and "before" pages back. "after" brings
// only what was written since, with the ids of messages removed since "since": that is what an
// open conversation asks every few seconds, as there is no live connection.
app.get('/api/messages/:userId', requireUser, async (c) => {
  const db = c.env.DB;
  const me = c.get('user').id;
  const person = await correspondent(c);
  if (!person) return c.json({ error: 'המשתמש לא נמצא' }, 404);

  const now = Date.now();
  const [conversation, canSend] = await Promise.all([conversationOf(db, me, person.id), areFriends(db, me, person.id)]);
  const base = { user: person, canSend, now };
  if (!conversation) return c.json({ ...base, messages: [], removed: [], hasMore: false, readUpTo: 0 });
  base.readUpTo = conversation.low_id === me ? conversation.low_read : conversation.high_read;
  const message = (row) => serializeMessage(row, me);

  const after = parseId(c.req.query('after') ?? '');
  if (after !== null) {
    const since = (parseId(c.req.query('since') ?? '') ?? now) - REMOVAL_OVERLAP_MS;
    const [fresh, gone] = await db.batch([
      db.prepare(`${MESSAGE_SELECT} WHERE m.conversation_id = ? AND m.id > ? ORDER BY m.id LIMIT ?`).bind(conversation.id, after, MESSAGES_PAGE * 4),
      db.prepare('SELECT id FROM messages WHERE conversation_id = ? AND removed_at >= ? AND id <= ?').bind(conversation.id, since, after),
    ]);
    return c.json({ ...base, messages: fresh.results.map(message), removed: gone.results.map((row) => row.id) });
  }

  const before = parseId(c.req.query('before') ?? '') ?? Number.MAX_SAFE_INTEGER;
  const { results } = await db
    .prepare(`${MESSAGE_SELECT} WHERE m.conversation_id = ? AND m.id < ? ORDER BY m.id DESC LIMIT ?`)
    .bind(conversation.id, before, MESSAGES_PAGE + 1)
    .all();
  return c.json({ ...base, messages: results.slice(0, MESSAGES_PAGE).reverse().map(message), removed: [], hasMore: results.length > MESSAGES_PAGE });
});

// Writing is between friends only. The first message opens the conversation.
app.post('/api/messages/:userId', requireUser, async (c) => {
  const db = c.env.DB;
  const me = c.get('user').id;
  const person = await correspondent(c);
  if (!person) return c.json({ error: 'המשתמש לא נמצא' }, 404);
  if (!(await areFriends(db, me, person.id))) return c.json({ error: 'אפשר לכתוב רק לחברים' }, 403);

  const input = await readJson(c);
  const body = typeof input.body === 'string' ? input.body.trim() : '';
  const image = input.image ?? null;
  if (body.length > MAX_MESSAGE || (!body && !image)) return c.json({ error: 'ההודעה ריקה או ארוכה מדי' }, 400);
  if (image !== null) {
    const owned = typeof image === 'string' && await db.prepare(UNUSED_IMAGE).bind(image, me).first();
    if (!owned) return c.json({ error: 'תמונה לא תקינה' }, 400);
  }

  const now = Date.now();
  const mine = me < person.id ? 'low_read' : 'high_read';
  const conversation = await db
    .prepare(`INSERT INTO conversations (low_id, high_id, created_at) VALUES (?, ?, ?)
      ON CONFLICT (low_id, high_id) DO UPDATE SET created_at = created_at RETURNING id`)
    .bind(Math.min(me, person.id), Math.max(me, person.id), now)
    .first();
  const row = await db
    .prepare('INSERT INTO messages (conversation_id, sender_id, body, created_at) VALUES (?, ?, ?, ?) RETURNING id')
    .bind(conversation.id, me, body || null, now)
    .first();
  // Whoever answers has seen what they answer.
  await db.batch([
    db.prepare(`UPDATE conversations SET ${mine} = MAX(${mine}, ?) WHERE id = ?`).bind(row.id, conversation.id),
    ...(image ? [db.prepare('UPDATE images SET message_id = ? WHERE id = ?').bind(row.id, image)] : []),
  ]);
  // Messages from one person take the place of each other on the device, so a conversation
  // does not pile up there.
  later(c, pushTo(c, person.id, 'messages', {
    title: c.get('user').name, body: body ? body.slice(0, 140) : 'תמונה', path: `/messages/${me}`, tag: `m${me}`,
  }));
  return c.json({ message: { id: row.id, mine: true, body: body || null, image, at: now, removed: false } }, 201);
});

// Marks the conversation as seen, up to a message the reader's screen really showed. The mark
// never passes the newest message, so nothing written later can be skipped by it.
app.put('/api/messages/:userId/read', requireUser, async (c) => {
  const db = c.env.DB;
  const me = c.get('user').id;
  const other = parseId(c.req.param('userId'));
  const conversation = other && other !== me && (await conversationOf(db, me, other));
  if (!conversation) return c.json({ error: 'השיחה לא נמצאה' }, 404);

  const { upTo } = await readJson(c);
  const mine = conversation.low_id === me ? 'low_read' : 'high_read';
  await db
    .prepare(`UPDATE conversations SET ${mine} = MAX(${mine},
      MIN(?1, COALESCE((SELECT MAX(id) FROM messages WHERE conversation_id = ?2), 0))) WHERE id = ?2`)
    .bind(Number.isSafeInteger(upTo) && upTo > 0 ? upTo : Number.MAX_SAFE_INTEGER, conversation.id)
    .run();
  return c.json({ ok: true });
});

// Only the writer removes a message. Its text is erased and its picture is really deleted. A
// mark stays in its place, for both sides, so the conversation still reads in order.
app.delete('/api/messages/:userId/:id', requireUser, async (c) => {
  const db = c.env.DB;
  const id = parseId(c.req.param('id'));
  const removed = id && (await db
    .prepare('UPDATE messages SET body = NULL, removed_at = ? WHERE id = ? AND sender_id = ? AND removed_at IS NULL RETURNING id')
    .bind(Date.now(), id, c.get('user').id)
    .first());
  if (!removed) return c.json({ error: 'ההודעה לא נמצאה' }, 404);

  const image = await db.prepare('SELECT id FROM images WHERE message_id = ?').bind(id).first();
  if (image) {
    await c.env.IMAGES.delete(image.id);
    await db.prepare('DELETE FROM images WHERE id = ?').bind(image.id).run();
  }
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
  const { results: emailInvites } = await db
    .prepare(`SELECT email, name, MAX(created_at) AS createdAt, MAX(expires_at) AS expiresAt FROM login_tokens
      WHERE invited_by IS NOT NULL AND used_at IS NULL AND expires_at > ? AND email NOT LIKE ?
      GROUP BY email ORDER BY createdAt DESC`)
    .bind(Date.now(), `%${NO_MAILBOX}`)
    .all();
  const mailer = await db.prepare('SELECT address, contacts FROM mailers WHERE user_id = ?').bind(c.get('user').id).first();
  return c.json({
    usage: await imageUsage(c),
    ...counts.results[0],
    members: members.results,
    emailInvites,
    mailer: mailer && { address: mailer.address, contacts: !!mailer.contacts },
    defaultNote: DEFAULT_NOTE,
  });
});

// The admin's own contacts, for completing an address while an invitation is typed. They are
// read from Google now, handed to the admin's browser, and stored nowhere.
app.get('/api/admin/contacts', requireAdmin, async (c) => {
  c.header('Cache-Control', 'private, no-store');
  const mailer = await c.env.DB.prepare('SELECT refresh_token, contacts FROM mailers WHERE user_id = ?').bind(c.get('user').id).first();
  if (!mailer?.contacts) return c.json({ contacts: null, problem: 'unconnected' });
  const { contacts = null, problem = null } = await googleContacts(c.env, mailer.refresh_token);
  return c.json({ contacts, problem });
});

// Stops the sending: the key is erased here, and Google is told to stop honouring it.
app.delete('/api/admin/mailer', requireAdmin, async (c) => {
  const row = await c.env.DB.prepare('DELETE FROM mailers WHERE user_id = ? RETURNING refresh_token').bind(c.get('user').id).first();
  if (row) {
    const token = await unseal(c.env.GOOGLE_CLIENT_SECRET, row.refresh_token).catch(() => null);
    if (token) {
      await fetch(c.env.GOOGLE_REVOKE_URL ?? 'https://oauth2.googleapis.com/revoke', { method: 'POST', body: new URLSearchParams({ token }) }).catch(() => null);
    }
  }
  return c.json({ ok: true });
});

// An invitation to an email address. The letter goes out from the admin's own Gmail if they
// connected it, and otherwise comes back for them to send by hand. The address stays on the
// account: it is how the member comes back with Google. Inviting the same address again
// replaces the link that was waiting.
app.post('/api/admin/email-invites', requireAdmin, async (c) => {
  const input = await readJson(c);
  const raw = input.email;
  const email = typeof raw === 'string' && raw.length <= 254 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(raw.trim())
    ? normalizeGoogleEmail(raw) : null;
  if (!email || email.endsWith(NO_MAILBOX)) return c.json({ error: 'כתובת לא תקינה' }, 400);
  const name = input.name === undefined || input.name === '' ? null : cleanText(input.name, MAX_NAME);
  if (input.name && !name) return c.json({ error: 'השם ארוך מדי, עד 60 תווים' }, 400);
  const note = input.note === undefined || input.note === '' ? null : cleanText(input.note, MAX_NOTE);
  if (input.note && !note) return c.json({ error: `המכתב ארוך מדי, עד ${MAX_NOTE} תווים` }, 400);

  const db = c.env.DB;
  const member = await db.prepare('SELECT 1 FROM users WHERE email = ?').bind(email).first();
  if (member) return c.json({ error: 'הכתובת הזו כבר שייכת לחבר ברשת' }, 409);
  const token = randomToken();
  const now = Date.now();
  await db.batch([
    db.prepare('DELETE FROM login_tokens WHERE email = ? AND invited_by IS NOT NULL AND used_at IS NULL').bind(email),
    db.prepare('INSERT INTO login_tokens (token_hash, email, created_at, expires_at, invited_by, name) VALUES (?, ?, ?, ?, ?, ?)')
      .bind(await sha256(token), email, now, now + EMAIL_INVITE_MS, c.get('user').id, name),
  ]);
  const site = new URL(c.req.url).origin;
  const expiresAt = now + EMAIL_INVITE_MS;
  const letter = inviteLetter({ name, sender: c.get('user').name, note, link: `${site}/login#token=${token}`, site, expiresAt });

  // The letter is addressed as the admin typed it. Gmail ignores the dots, but people do not.
  const mailer = input.send === false ? null : await db.prepare('SELECT refresh_token FROM mailers WHERE user_id = ?').bind(c.get('user').id).first();
  const outcome = mailer ? await sendFromGmail(c.env, mailer.refresh_token, { to: raw.trim(), ...letter }) : null;
  if (outcome === 'expired') await db.prepare('DELETE FROM mailers WHERE user_id = ?').bind(c.get('user').id).run();
  return c.json({ email, name, link: `${site}/login#token=${token}`, expiresAt, subject: letter.subject, text: letter.text, sent: outcome === 'sent', sendProblem: outcome === 'sent' ? null : outcome }, 201);
});

app.delete('/api/admin/email-invites/:email', requireAdmin, async (c) => {
  await c.env.DB
    .prepare('DELETE FROM login_tokens WHERE email = ? AND invited_by IS NOT NULL AND used_at IS NULL')
    .bind(c.req.param('email'))
    .run();
  return c.json({ ok: true });
});

// Without a mail service, the admin hands out login links: a new member, or a new link for
// someone who lost their session. The link works once. Members get an address nobody can mail.
app.post('/api/admin/invites', requireAdmin, async (c) => {
  const { userId } = await readJson(c);
  let email = `member-${randomToken().slice(0, 16).toLowerCase()}${NO_MAILBOX}`;
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
